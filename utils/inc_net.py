import copy
import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from convs.linears import CosineLinear

def get_convnet(args, pretrained=False):
    name = args["convnet_type"].lower()
    if name == 'cnn1d':
        from convs.cnn1d import CNN1DConvNet
        return CNN1DConvNet()

    else:
        raise NotImplementedError("Unknown type {}".format(name))


def _feat(module, x):
    """Lay vector dac trung tu mot module, chap nhan ca hai giao dien
    (extract_vector(x) hoac forward(x)["features"])."""
    if hasattr(module, "extract_vector"):
        return module.extract_vector(x)
    return module(x)["features"]


class VectorGate(nn.Module):
    """Cong theo tung chieu giua nhanh stability va nhanh plasticity.

    stab_dim — so chieu nhanh stability.
    plas_dim — so chieu nhanh plasticity; mac dinh bang stab_dim, tuc giu
               nguyen chu ky cu VectorGate(feature_dim).
    Dau ra co plas_dim chieu.
    """
    def __init__(self, stab_dim, plas_dim=None):
        super(VectorGate, self).__init__()
        plas_dim = stab_dim if plas_dim is None else plas_dim
        self.gate = nn.Sequential(
            nn.Linear(stab_dim + plas_dim, plas_dim),
            nn.Sigmoid()
        )
    def forward(self, phi_x, a_x):
        combined = torch.cat([phi_x, a_x], dim=1)
        return self.gate(combined)


class AFSICIDSNet(nn.Module):
    """Mang AFSIC-IoV.

    KHONG GIAN DAC TRUNG — hai che do, chon bang co expand_feature_space:

      false (mac dinh, hanh vi cu): z = g (*) phi_x + (1-g) (*) a_x. Hai nhanh
          tron vao DUNG 64 chieu cua task 0. Ca 13 lop phai song trong khong
          gian da duoc nan boi 3 lop cua task 0 — do la ly do prototype dau
          task 1 co cos(w_i,w_j) = 0,9917: encoder dong bang khong tach noi
          cac lop moi nen trung binh lop cua chung roi gan nhu cung mot huong.

      true: z = [ phi_x , g (*) a_x ]. Nhanh moi NOI vao thay vi tron, nen
          feature_dim no theo task: 64 -> 128 -> 192 -> 256 -> 320, dung day
          so cua DER trong HFIN. Cong g van giu vai tro cu — quyet dinh bao
          nhieu tin hieu nhanh plasticity duoc nap vao — nhung khong con phai
          danh doi voi nhanh stability tren cung mot o nho.

      CANH BAO: bat co nay doi kien truc nen checkpoint task >= 1 cu KHONG
      con tuong thich. Checkpoint task 0 van dung duoc vi task 0 khong co
      nhanh nao.
    """
    def __init__(self, args, pretrained=False):
        super(AFSICIDSNet, self).__init__()
        self.args = args
        self.convnet = get_convnet(args, pretrained)
        self.base_dim = self.convnet.out_dim
        self.feature_dim = self.base_dim
        self._stability_dim = self.base_dim
        self._expand_mode = bool(args.get("expand_feature_space", False))
        self.fc = None
        self.stability_encoder = None
        self.plasticity_adapter = None
        self.gate = None
        self._device = args["device"][0]

    def extract_vector(self, x):
        if self.stability_encoder is None:
            return _feat(self.convnet, x)

        self.stability_encoder.eval()
        with torch.no_grad():
            phi_x = _feat(self.stability_encoder, x)

        a_x = _feat(self.plasticity_adapter, x)
        g = self.gate(phi_x, a_x)

        if self._expand_mode:
            z = torch.cat([phi_x, g * a_x], dim=1)
        else:
            z = g * phi_x + (1.0 - g) * a_x
        # Dac ta muc 5.3:  z = Norm( g (*) h_s + (1-g) (*) h_a )
        #
        # Ban goc bo buoc Norm. Thuc te anh huong nho vi moi noi dung z deu
        # tu chuan hoa (CosineLinear.forward, compute_fsp_loss,
        # compute_proto_loss, compute_local_prototypes), nhung de khop dac
        # ta thi bat gated_fusion_norm.
        #
        # CANH BAO: bat co nay doi bieu dien dac trung nen MOI checkpoint
        # hien co tro nen khong tuong thich — phai chay lai tu task 0.
        if self.args.get("gated_fusion_norm", False):
            z = F.normalize(z, p=2, dim=1)
        return z

    def forward(self, x):
        z = self.extract_vector(x)
        out = self.fc(z)
        out.update({"features": z})
        return out

    def freeze_stability_encoder(self):
        if self.stability_encoder is not None:
            for p in self.stability_encoder.parameters():
                p.requires_grad = False
            self.stability_encoder.eval()

    def unfreeze_adapter(self):
        if self.plasticity_adapter is not None:
            for p in self.plasticity_adapter.parameters():
                p.requires_grad = True
            self.plasticity_adapter.train()
        if self.gate is not None:
            for p in self.gate.parameters():
                p.requires_grad = True
            self.gate.train()

    def unfreeze_incremental_params(self):
        """Alias for unfreeze_adapter, matching instruction API."""
        self.unfreeze_adapter()
        if self.fc is not None:
            for p in self.fc.parameters():
                p.requires_grad = True

    def update_fc(self, nb_classes):
        """Dung lai classifier cho nb_classes lop o so chieu HIEN TAI.

        Khi expand_feature_space bat, feature_dim no ra sau moi transition nen
        fc cu co it cot hon. Cach xu ly giong DER: chep nguyen phan trong so cu
        vao cac cot cu, dat 0 cho cac cot MOI — tuc lop cu khong phan ung voi
        chieu dac trung cua task moi cho toi khi chinh no hoc duoc.
        """
        fc = CosineLinear(self.feature_dim, nb_classes, sigma=True)
        if self.fc is not None:
            nb_output = self.fc.out_features
            old_dim = int(self.fc.weight.shape[1])
            keep = min(old_dim, self.feature_dim)
            fc.weight.data[:nb_output, :keep] = self.fc.weight.data[:nb_output, :keep]
            if self.feature_dim > old_dim:
                fc.weight.data[:nb_output, old_dim:] = 0.0
            if self.fc.sigma is not None:
                fc.sigma.data = self.fc.sigma.data
        del self.fc
        self.fc = fc

    def transition_to_incremental_stage(self):
        _expand = self._expand_mode

        class FrozenFeatureExtractor(nn.Module):
            def __init__(self, extractor):
                super().__init__()
                self.extractor = copy.deepcopy(extractor)
                for p in self.extractor.parameters():
                    p.requires_grad = False
                self.extractor.eval()
            def forward(self, x):
                return self.extractor(x)
            def extract_vector(self, x):
                return _feat(self.extractor, x)

        class FusedFeatureExtractor(nn.Module):
            def __init__(self, stability, plasticity, gate, expand):
                super().__init__()
                self.stability = copy.deepcopy(stability)
                self.plasticity = copy.deepcopy(plasticity)
                self.gate = copy.deepcopy(gate)
                self.expand = bool(expand)
                for p in self.parameters():
                    p.requires_grad = False
                self.eval()
            def extract_vector(self, x):
                phi_x = _feat(self.stability, x)
                a_x = _feat(self.plasticity, x)
                g = self.gate(phi_x, a_x)
                if self.expand:
                    return torch.cat([phi_x, g * a_x], dim=1)
                return g * phi_x + (1.0 - g) * a_x
            def forward(self, x):
                return {"features": self.extract_vector(x)}

        if self.stability_encoder is None:
            self.stability_encoder = FrozenFeatureExtractor(self.convnet)
            # Sau giai doan incremental dau tien, convnet goc chi con la nhanh
            # stability dong bang. No khong duoc xuat hien nhu tham so hoc duoc.
            for p in self.convnet.parameters():
                p.requires_grad = False
            self.convnet.eval()
            self._stability_dim = self.base_dim
        else:
            self.stability_encoder = FusedFeatureExtractor(
                self.stability_encoder, self.plasticity_adapter, self.gate, _expand)
            # Nhanh stability moi tai tao dung phep hop nhat cua task truoc,
            # nen so chieu cua no chinh la feature_dim TRUOC transition nay.
            self._stability_dim = self.feature_dim

        _plastic = bool(self.args.get("plastic_source_trainable", False))

        class BottleneckFeatureAdapter(nn.Module):
            """Nhanh plasticity.

            MAC DINH (plastic_source_trainable=False, hanh vi cu): frozen_source
            bi dong bang VA duoc goi trong torch.no_grad(), nen ca nhanh
            plasticity chi la mot MLP residual tren 64 con so da dong bang tu
            task 0. No KHONG BAO GIO nhin thay du lieu goc. Neu lop cua task moi
            khong tach duoc trong khong gian dac trung task 0 thi khong do rong
            adapter nao cuu duoc — thong tin da mat truoc do.

            plastic_source_trainable=True: mo dong bang va cho gradient chay qua,
            nen nhanh plasticity trich duoc dac trung MOI tu du lieu goc.

            expand_feature_space=True: feature_source la ban sao cua convnet GOC,
            tuc nhanh nay doc thang du lieu tho (31 chieu) chu khong phai 64 con
            so dau ra cua nhanh stability. Cong voi phep NOI o extract_vector,
            day dung la co che DER cua HFIN: moi task mot backbone moi, khong
            gian dac trung no theo task.
            """
            def __init__(self, feature_source, feature_dim, bottleneck_dim, plastic=False):
                super().__init__()
                self.plastic = plastic
                self.frozen_source = copy.deepcopy(feature_source)
                for p in self.frozen_source.parameters():
                    p.requires_grad = plastic
                if not plastic:
                    self.frozen_source.eval()
                self.adapter = nn.Sequential(
                    nn.Linear(feature_dim, bottleneck_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(bottleneck_dim, feature_dim),
                )

            def extract_vector(self, x):
                if self.plastic:
                    base = _feat(self.frozen_source, x)
                else:
                    self.frozen_source.eval()
                    with torch.no_grad():
                        base = _feat(self.frozen_source, x)
                return base + self.adapter(base)

            def forward(self, x):
                return {"features": self.extract_vector(x)}

        if _expand:
            # Nhanh moi doc du lieu THO, luon 64 chieu ra.
            _source, _src_dim = self.convnet, self.base_dim
        else:
            _source, _src_dim = self.stability_encoder, self._stability_dim

        bottleneck_dim = int(self.args.get("adapter_bottleneck", max(8, _src_dim // 4)))
        self.plasticity_adapter = BottleneckFeatureAdapter(
            _source, _src_dim, bottleneck_dim, plastic=_plastic)
        self.gate = VectorGate(self._stability_dim, _src_dim if _expand else self._stability_dim)
        self.feature_dim = (self._stability_dim + _src_dim) if _expand else self._stability_dim

        # fc duoc dung o so chieu CU trong incremental_train; sau khi khong gian
        # dac trung no ra thi phai dung lai cho khop.
        if self.fc is not None and int(self.fc.weight.shape[1]) != self.feature_dim:
            self.update_fc(self.fc.out_features)

        logging.info(
            "[NET] transition: stability_dim=%d + plasticity_dim=%d -> feature_dim=%d "
            "(expand_feature_space=%s, bottleneck=%d)",
            self._stability_dim, _src_dim if _expand else 0, self.feature_dim,
            _expand, bottleneck_dim)
        self.to(self._device)

    def init_new_class_weights_from_prototypes(self, prototypes, class_ids):
        for cid in class_ids:
            if cid < self.fc.out_features:
                if isinstance(prototypes, dict):
                    if cid not in prototypes:
                        continue
                    proto = prototypes[cid]
                else:
                    if cid >= len(prototypes):
                        continue
                    proto = prototypes[cid]
                if isinstance(proto, np.ndarray):
                    proto = torch.from_numpy(proto).float()
                proto = proto.to(self.fc.weight.device)
                if proto.numel() != self.fc.weight.shape[1]:
                    # Prototype tinh o so chieu khac (vd. con luu tu task truoc
                    # khi khong gian dac trung chua no). Bo qua thay vi hong.
                    continue
                proto_norm = proto / (torch.norm(proto, p=2) + 1e-8)
                self.fc.weight.data[cid] = proto_norm

    def get_trainable_incremental_params(self):
        params = []
        if self.plasticity_adapter is not None:
            # plastic_source_trainable=True thi frozen_source cung nam trong
            # .parameters() voi requires_grad=True nen tu dong duoc huan luyen.
            params.extend(p for p in self.plasticity_adapter.parameters() if p.requires_grad)
        if self.gate is not None:
            params.extend(self.gate.parameters())
        if self.fc is not None:
            params.extend(self.fc.parameters())
        return params

    def get_incremental_state_dict(self):
        """Return state dict of only incremental (adapter/gate/fc) parameters."""
        state = {}
        full_sd = self.state_dict()
        for k, v in full_sd.items():
            if "plasticity_adapter.frozen_source" in k and not self._expand_mode:
                continue
            if any(sub in k for sub in ["plasticity_adapter", "gate", "fc"]):
                state[k] = v
        return state

    def load_incremental_state_dict(self, state_dict):
        """Load only incremental parameters from a state dict."""
        own_sd = self.state_dict()
        for k, v in state_dict.items():
            if k in own_sd:
                own_sd[k] = v
        self.load_state_dict(own_sd)
