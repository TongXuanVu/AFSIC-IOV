import copy
import math
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
        return CNN1DConvNet(norm_type=args.get("norm_type", "batch"))

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
        # Dac ta 5.1: encoder dung chung h_s duoc HOC o moi stage (client gui
        # Delta theta len server, L_prox keo theta ve theta^{t-1}). Mac dinh
        # false = hanh vi cu: tu task 1 nhanh stability bi dong bang.
        self._shared_trainable = bool(args.get("shared_encoder_trainable", False))
        # Dac ta 5.4: P(y=c|x) = softmax(gamma * cos(z, p~_c)); KHONG co trong so
        # phan loai hoc duoc. Mac dinh false = hanh vi cu (CosineLinear hoc duoc).
        self._proto_clf = bool(args.get("prototype_classifier", False))
        if self._proto_clf and args.get("fixed_classifier", False):
            raise ValueError("prototype_classifier va fixed_classifier loai tru nhau")
        # So KHOI LA trong vector dac trung hien tai. Task 0 co 1 khoi; moi lan
        # transition them 1. Dung cho block_norm.
        self._num_blocks = 1
        # block_norm: chuan hoa TUNG KHOI ve chuan don vi truoc khi noi.
        #
        # VI SAO. Do tren checkpoint that (task 4, 5 khoi 64 chieu):
        #     nang luong ||z_khoi||^2 :  t0 93,07%  t1 2,02%  t2 1,79%
        #                                t3 1,76%   t4 1,37%
        # Khoi task 0 nuot 93% nang luong. Ma fc la CosineLinear nen
        #     logit_c = sigma * cos(z, w_c)  <=  sigma * ||z_khoi(c)|| / ||z||
        # tuc mot lop dat trong so trong khoi t4 bi CHAN TREN o
        #     2,21 * 0,1168 = 0,258
        # con lop task 0 dat toi 2,21 * 0,9647 = 2,13. Do duoc: logit trung
        # binh cua Benign 2,1085 (dung bang tran), cua systematic -0,3966 —
        # va systematic THANG 0 lan tren 10.905 mau. Lop moi khong thua vi hoc
        # kem, no khong bao gio duoc phep thang.
        #
        # Chuan hoa moi khoi ve chuan don vi cho moi khoi cung tran cosine
        # 1/sqrt(so khoi), nen cac task canh tranh cong bang.
        self._block_norm = bool(args.get("block_norm", False))
        # block_norm_gamma: NGHIENG ve task moi. Moi lan no ra, khoi cu bi chia
        # cho gamma so voi khoi moi, nen khoi cua task j co chuan ti le
        # gamma^j — cang moi cang lon.
        #   gamma = 1.0  -> moi khoi bang nhau (dung block_norm thuan)
        #   gamma > 1.0  -> uu tien task moi
        # Day la MOT tham so, khong phai hang so tuy tien: bao cao kem duong
        # do nhay theo gamma thi phan bien kiem chung duoc.
        self._block_gamma = float(args.get("block_norm_gamma", 1.0))
        self.fc = None
        self.stability_encoder = None
        self.plasticity_adapter = None
        self.gate = None
        self._device = args["device"][0]

    def extract_vector(self, x):
        if self.stability_encoder is None:
            return _feat(self.convnet, x)

        if self._shared_trainable:
            # Encoder dung chung duoc huan luyen: giu graph, che do train/eval
            # theo net.train()/net.eval().
            phi_x = _feat(self.stability_encoder, x)
        else:
            self.stability_encoder.eval()
            with torch.no_grad():
                phi_x = _feat(self.stability_encoder, x)

        a_x = _feat(self.plasticity_adapter, x)
        g = self.gate(phi_x, a_x)

        if self._expand_mode:
            if self._block_norm:
                # phi_x da chua _num_blocks-1 khoi don vi -> giu chuan
                # sqrt(so khoi do); khoi moi ve chuan 1. Quy nap lai thi MOI
                # khoi la deu co chuan 1.
                _b_truoc = max(1, int(self._num_blocks) - 1)
                _g = max(1e-6, float(self._block_gamma))
                z = torch.cat([
                    self._chuan_hoa_khoi(phi_x, math.sqrt(_b_truoc) / _g),
                    self._chuan_hoa_khoi(g * a_x, 1.0),
                ], dim=1)
            else:
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
        if self._shared_trainable:
            return   # 5.1: encoder dung chung duoc huan luyen va gop o moi stage
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
            if self.args.get("fixed_classifier", False) or self._proto_clf:
                self.fc.weight.requires_grad = False   # neo co dinh / prototype classifier

    @staticmethod
    def _chuan_hoa_khoi(v, muc_tieu=1.0):
        """Dua mot khoi ve chuan `muc_tieu` (mac dinh 1)."""
        return v / (v.norm(dim=1, keepdim=True) + 1e-8) * muc_tieu

    def update_fc(self, nb_classes):
        """Dung lai classifier cho nb_classes lop o so chieu HIEN TAI.

        Khi expand_feature_space bat, feature_dim no ra sau moi transition nen
        fc cu co it cot hon. Cach xu ly giong DER: chep nguyen phan trong so cu
        vao cac cot cu, dat 0 cho cac cot MOI — tuc lop cu khong phan ung voi
        chieu dac trung cua task moi cho toi khi chinh no hoc duoc.
        """
        fc = CosineLinear(self.feature_dim, nb_classes, sigma=True)
        nb_output = 0
        if self.fc is not None:
            nb_output = self.fc.out_features
            old_dim = int(self.fc.weight.shape[1])
            keep = min(old_dim, self.feature_dim)
            fc.weight.data[:nb_output, :keep] = self.fc.weight.data[:nb_output, :keep]
            if self.feature_dim > old_dim:
                fc.weight.data[:nb_output, old_dim:] = 0.0
            if self.fc.sigma is not None:
                fc.sigma.data = self.fc.sigma.data
        self._apply_cosine_sigma(fc)
        if self.args.get("fixed_classifier", False):
            # Goi tu incremental_train (so lop tang): lop cu = nb_output.
            # Goi tu transition (so lop giu nguyen, chi no so chieu): SINH LAI
            # neo cho lop moi cua task nay o so chieu MOI — neu khong, neo lop
            # moi chi nam trong khoi dac trung cu (dong bang) va khoi moi = 0.
            if nb_classes > nb_output:
                self._fixed_known = nb_output
            self._set_fixed_anchors(fc, min(nb_output, getattr(self, "_fixed_known", nb_output)))
        if self._proto_clf:
            # 5.4: trong so = prototype p~_c, ghi boi calibrate sau moi lan gop,
            # khong hoc bang gradient.
            fc.weight.requires_grad = False
        del self.fc
        self.fc = fc

    def _set_fixed_anchors(self, fc, nb_keep):
        """Phuong an B: bo phan loai CO DINH theo 'neo' truc giao.

        VI SAO. Tren CAN-IoV 49/50 client task 0 chi co MOT lop. Client chi
        thay mau duong nen CE cuc bo keo fc cua lop minh ve moi huong va day
        cac lop khac ra — client Benign va client DoS keo fc nguoc chieu nhau,
        mo hinh gop dao giua 'toan Benign' va 'toan DoS' (log debug/final).
        Giu cac vector lop CO DINH va tach xa nhau thi khong con xung dot o fc:
        client chi con keo dac trung mau cua minh ve neo cua lop minh.
        Tham chieu: FedAwS (Yu et al., ICML 2020, "Federated Learning with Only
        Positive Labels"), FedETF (Li et al., ICCV 2023).

        Hang 0..nb_keep-1 (lop cu) giu nguyen — da truc chuan tu truoc, phan
        chieu moi = 0 van truc chuan. Hang moi: sinh ngau nhien CO DINH THEO
        SEED roi truc giao hoa voi hang cu va voi nhau (QR), nen moi client va
        global sinh ra CUNG mot ma tran. requires_grad=False: optimizer bo qua.
        """
        W = fc.weight.data
        C, D = W.shape
        if C > D:
            raise ValueError(f"fixed_classifier: {C} lop > {D} chieu, khong truc giao duoc")
        _seed = self.args.get("seed", 0)
        _seed = int(_seed[0] if isinstance(_seed, (list, tuple)) else _seed)
        g = torch.Generator().manual_seed(1000003 * (_seed + 1) + 7919 * C + D)
        R = torch.randn(D, C, generator=g, dtype=torch.float64)
        if nb_keep > 0:
            old = F.normalize(W[:nb_keep].double().cpu(), p=2, dim=1)
            R[:, :nb_keep] = old.t()
        # QR tren [cu | ngau nhien]: nb_keep cot dau tai tao dung khong gian cu,
        # cac cot sau truc giao voi no.
        Q, Rr = torch.linalg.qr(R)
        Q = Q * torch.sign(torch.diagonal(Rr)).unsqueeze(0)   # giu dung dau cot cu
        A = Q.t().float()
        if nb_keep > 0:
            A[:nb_keep] = F.normalize(W[:nb_keep].float().cpu(), p=2, dim=1)
        W.copy_(A.to(W.device))
        fc.weight.requires_grad = False

    def _apply_cosine_sigma(self, fc):
        """Dat thang do (nhiet do nghich) cua CosineLinear: logit = sigma*cos.

        [DO] ckpt task 0 (LR1 FINAL): sigma hoc duoc chi 2,22 -> logit nam trong
        [-2,22 ; 2,22]. Voi dac trung va trong so deu chuan hoa, loss softmax co
        CAN DUOI khong the vuot qua khi thang do nho (NormFace, Wang et al.,
        ACM MM 2017): log(1 + (C-1)*exp(-s*C/(C-1))). Voi s = 2,22 can nay la
        0,30 / 0,51 / 0,63 / 0,73 cho C = 6 / 9 / 11 / 13 lop — tang theo so
        lop, khop voi viec task 3-4 khong bao gio thoat trang thai sup.
        Mau dung (da phan loai dung) khong bao gio bao hoa nen gradient cua
        99% Benign lan at lop hiem, mo hinh bap benh giua hai nghiem suy bien.

        cosine_sigma = None (mac dinh) -> GIU NGUYEN hanh vi cu.
        cosine_sigma = s -> gan sigma = s; cosine_sigma_trainable (mac dinh
        false) quyet dinh co cho hoc tiep hay khong. Doi sigma KHONG doi argmax
        cua checkpoint cu (moi logit nhan cung mot so duong).
        """
        _s = self.args.get("cosine_sigma") if isinstance(getattr(self, "args", None), dict) else None
        if _s is None or getattr(fc, "sigma", None) is None:
            return
        fc.sigma.data.fill_(float(_s))
        fc.sigma.requires_grad = bool(self.args.get("cosine_sigma_trainable", False))

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
            def __init__(self, stability, plasticity, gate, expand,
                         block_norm=False, num_blocks_prev=1, block_gamma=1.0,
                         fusion_norm=False):
                super().__init__()
                self.stability = copy.deepcopy(stability)
                # Bai bao 5.3: z = Norm(g*h_s + (1-g)*h_a). Mang chinh ap Norm khi
                # gated_fusion_norm=True, nen nhanh stability cung PHAI ap Norm.
                self.fusion_norm = bool(fusion_norm)
                self.plasticity = copy.deepcopy(plasticity)
                self.gate = copy.deepcopy(gate)
                self.expand = bool(expand)
                # PHAI dung Y HET quy tac cua mang chinh, neu khong dac trung
                # luc suy luan khac luc huan luyen.
                self.block_norm = bool(block_norm)
                self.num_blocks_prev = int(num_blocks_prev)
                self.block_gamma = float(block_gamma)
                for p in self.parameters():
                    p.requires_grad = False
                self.eval()
            def extract_vector(self, x):
                phi_x = _feat(self.stability, x)
                a_x = _feat(self.plasticity, x)
                g = self.gate(phi_x, a_x)
                if self.expand:
                    if self.block_norm:
                        _n = lambda v, m: v / (v.norm(dim=1, keepdim=True) + 1e-8) * m
                        _g = max(1e-6, float(self.block_gamma))
                        return torch.cat([
                            _n(phi_x, math.sqrt(max(1, self.num_blocks_prev)) / _g),
                            _n(g * a_x, 1.0),
                        ], dim=1)
                    return torch.cat([phi_x, g * a_x], dim=1)
                z = g * phi_x + (1.0 - g) * a_x
                if self.fusion_norm:
                    z = F.normalize(z, p=2, dim=1)
                return z
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
                self.stability_encoder, self.plasticity_adapter, self.gate, _expand,
                block_norm=self._block_norm,
                num_blocks_prev=max(1, int(self._num_blocks) - 1),
                block_gamma=self._block_gamma,
                fusion_norm=bool(self.args.get("gated_fusion_norm", False)))
            # Nhanh stability moi tai tao dung phep hop nhat cua task truoc,
            # nen so chieu cua no chinh la feature_dim TRUOC transition nay.
            self._stability_dim = self.feature_dim

        if self._shared_trainable:
            # Hai lop boc o tren dong bang tham so trong __init__; mo lai de
            # nhanh stability hoc duoc (dac ta 5.1).
            for p in self.stability_encoder.parameters():
                p.requires_grad = True

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

        class RawInputAdapter(nn.Module):
            """Dac ta 5.2: h_a = A_psi(x). MLP bottleneck NHO doc thang dau vao tho x
            (khong boc/sao chep encoder). Ten thuoc tinh `adapter` de khop bo loc
            gop trong so (plasticity_adapter.adapter) va loss RS (L1 tren A_psi)."""
            def __init__(self, in_dim, out_dim, bottleneck_dim):
                super().__init__()
                self.adapter = nn.Sequential(
                    nn.Linear(in_dim, bottleneck_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(bottleneck_dim, out_dim),
                )
            def extract_vector(self, x):
                return self.adapter(x.flatten(1) if x.dim() > 2 else x)
            def forward(self, x):
                return {"features": self.extract_vector(x)}

        _adapter_input = str(self.args.get("adapter_input", "features")).lower()
        if _adapter_input not in ("features", "raw"):
            raise ValueError(f"adapter_input phai la 'features' hoac 'raw', nhan {_adapter_input!r}")

        if _expand:
            # Nhanh moi doc du lieu THO, luon 64 chieu ra.
            _source, _src_dim = self.convnet, self.base_dim
        else:
            _source, _src_dim = self.stability_encoder, self._stability_dim

        bottleneck_dim = int(self.args.get("adapter_bottleneck", max(8, _src_dim // 4)))
        if _adapter_input == "raw":
            self.plasticity_adapter = RawInputAdapter(
                int(self.args.get("adapter_input_dim", 31)), _src_dim, bottleneck_dim)
        else:
            self.plasticity_adapter = BottleneckFeatureAdapter(
                _source, _src_dim, bottleneck_dim, plastic=_plastic)
        self.gate = VectorGate(self._stability_dim, _src_dim if _expand else self._stability_dim)
        # adapter_identity_init: khoi tao NHANH MOI SAO CHO z == dac trung cu luc mo stage.
        #
        # [DO tren mau exemplar that + checkpoint task 0, cau hinh afsic_paper_t14]
        # Khoi tao ngau nhien (adapter raw 31->64->64, cong sigmoid ~0,5) cho
        # |h_a| = 1,12 > |h_s| = 0,98, nen z = Norm(g*h_s + (1-g)*h_a) chi con cos
        # 0,56 so voi dac trung task 0 (p10 0,39, min 0,19). Prototype lop cu
        # (tinh tu h_s) vi vay khong con khop z ngay truoc khi huan luyen, va Benign
        # bi keo sang prototype lop moi (Old Acc 34% o round 1 task 1).
        #
        # Cach lam: lop Linear CUOI cua adapter = 0 -> h_a = 0; cong: trong so = 0,
        # bias = gate_init_bias (mac dinh 0 -> g = 0,5 DEU moi chieu). Khi do
        # z = Norm(g*h_s) ~ h_s theo HUONG (g deu nen khong doi huong) => logit lop
        # cu khong doi so voi mo hinh task truoc. Gradient van chay: dL/dW_cuoi =
        # (kich hoat an)^T * dL/dh_a khac 0 vi (1-g) = 0,5. Chi ap cho adapter raw.
        if bool(self.args.get("adapter_identity_init", False)):
            _last = None
            if _adapter_input == "raw":
                _last = self.plasticity_adapter.adapter[-1]
            if _last is not None:
                nn.init.zeros_(_last.weight)
                nn.init.zeros_(_last.bias)
            _gl = self.gate.gate[0]
            nn.init.zeros_(_gl.weight)
            nn.init.constant_(_gl.bias, float(self.args.get("gate_init_bias", 0.0)))
        self.feature_dim = (self._stability_dim + _src_dim) if _expand else self._stability_dim
        # Cot dau tien cua khoi dac trung MOI (chi co nghia khi expand). Dung de
        # khoi tao phan trong so cua lop CU tren khoi moi (xem
        # init_old_class_new_block_from_prototypes).
        self._new_block_start = int(self._stability_dim) if _expand else None
        if _expand:
            self._num_blocks = int(self._num_blocks) + 1

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

    def init_old_class_new_block_from_prototypes(self, prototypes, class_ids):
        """Dien khoi dac trung MOI cho trong so cua cac lop CU.

        VAN DE: update_fc dat 0 cho cac cot moi cua lop cu (cach cua DER). DER
        dung bo phan loai TUYEN TINH nen cot 0 vo hai. Nhung fc o day la
        CosineLinear: dac trung duoc chuan hoa CA vector [phi, g*a], nen voi
        w_cu = [w, 0] thi cos(z, w_cu) <= |phi| / |z| < 1 — lop cu bi TRAN tren
        mot cach he thong, trong khi lop moi (khoi tao tu prototype tren CA
        khong gian) dat cos gan 1 voi moi dac trung ReLU.
        [DO] afsic-b2-goc: round 1 cua task 1/2/3/4 Old Acc = 0,75 / 0,75 /
        0,75 / 0,75 %, New Acc = 6 / 40 / 82 %: MOI mau (ca 99% Benign) bi doan
        la lop moi. Task 3 va 4 khong bao gio hoi phuc trong 30 round.

        CACH SUA: giu NGUYEN huong da hoc cua lop cu tren khoi cu, chi thay 0 o
        khoi moi bang phan khoi-moi cua prototype lop cu (tinh o khong gian
        HIEN TAI, trainer da tinh san luc dang ky task). Do lon khoi cu duoc dat
        bang |phan khoi-cu cua prototype| de ti le hai khoi dung nhu du lieu that.
        Chi dien khi khoi moi con dung bang 0 (vua no ra), goi lai la vo hai.
        Tra ve so lop da dien.
        """
        start = getattr(self, "_new_block_start", None)
        W = self.fc.weight.data
        D = int(W.shape[1])
        if start is None or start <= 0 or start >= D:
            return 0
        done = 0
        for cid in class_ids:
            if cid >= W.shape[0]:
                continue
            if isinstance(prototypes, dict):
                proto = prototypes.get(cid)
            else:
                proto = prototypes[cid] if cid < len(prototypes) else None
            if proto is None:
                continue
            if isinstance(proto, np.ndarray):
                proto = torch.from_numpy(proto)
            proto = proto.float().to(W.device).flatten()
            if proto.numel() != D:
                continue
            if float(W[cid, start:].abs().sum()) > 0.0:
                continue
            w_old = W[cid, :start]
            n_w = float(torch.norm(w_old, p=2))
            if n_w < 1e-8:
                continue
            p_old, p_new = proto[:start], proto[start:]
            row = torch.cat([w_old / n_w * torch.norm(p_old, p=2), p_new])
            W[cid] = row / (torch.norm(row, p=2) + 1e-8)
            done += 1
        return done

    def init_new_class_weights_from_prototypes(self, prototypes, class_ids, center=None):
        """Ghi w_c = prototype chuan hoa (imprinting).

        center (tuy chon): vector tru di truoc khi chuan hoa, w_c = (p_c - center)/|.|.
        [DO] tren checkpoint task 0 that: cos giua prototype cac lop = 0,94-0,99
        (dac trung ReLU cung huong). Prototype tho vi the co cos ~0,95 voi MOI
        mau -> lop moi thang tuyet doi luc khoi tao task: 96,6 % mau bi doan la
        lop moi (khop Old Acc 0,75 % cua afsic-b2-goc). Tru tam cac lop CU
        (trung binh prototype lop cu) bo di phan huong chung, giu phan phan biet.
        """
        if center is not None:
            if isinstance(center, np.ndarray):
                center = torch.from_numpy(center)
            center = center.float().to(self.fc.weight.device).flatten()
            if center.numel() != self.fc.weight.shape[1]:
                center = None
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
                if center is not None:
                    proto = proto.float().flatten() - center
                proto_norm = proto / (torch.norm(proto, p=2) + 1e-8)
                self.fc.weight.data[cid] = proto_norm

    def get_trainable_incremental_params(self):
        params = []
        if self._shared_trainable and self.stability_encoder is not None:
            params.extend(p for p in self.stability_encoder.parameters() if p.requires_grad)
        if self.plasticity_adapter is not None:
            # plastic_source_trainable=True thi frozen_source cung nam trong
            # .parameters() voi requires_grad=True nen tu dong duoc huan luyen.
            params.extend(p for p in self.plasticity_adapter.parameters() if p.requires_grad)
        if self.gate is not None:
            params.extend(self.gate.parameters())
        if self.fc is not None:
            params.extend(p for p in self.fc.parameters() if p.requires_grad)
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
