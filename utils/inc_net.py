import copy
import logging
import numpy as np
import torch
from torch import nn
from convs.linears import CosineLinear

def get_convnet(args, pretrained=False):
    name = args["convnet_type"].lower()
    if name == 'cnn1d':
        from convs.cnn1d import CNN1DConvNet
        return CNN1DConvNet()
    
    else:
        raise NotImplementedError("Unknown type {}".format(name))



class VectorGate(nn.Module):
    def __init__(self, feature_dim):
        super(VectorGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
    def forward(self, phi_x, a_x):
        combined = torch.cat([phi_x, a_x], dim=1)
        return self.gate(combined)


class AFSICIDSNet(nn.Module):
    def __init__(self, args, pretrained=False):
        super(AFSICIDSNet, self).__init__()
        self.args = args
        self.convnet = get_convnet(args, pretrained)
        self.feature_dim = self.convnet.out_dim
        self.fc = None
        self.stability_encoder = None
        self.plasticity_adapter = None
        self.gate = None
        self._device = args["device"][0]

    def extract_vector(self, x):
        if self.stability_encoder is None:
            if hasattr(self.convnet, "extract_vector"):
                return self.convnet.extract_vector(x)
            else:
                return self.convnet(x)["features"]
        else:
            self.stability_encoder.eval()
            with torch.no_grad():
                if hasattr(self.stability_encoder, "extract_vector"):
                    phi_x = self.stability_encoder.extract_vector(x)
                else:
                    phi_x = self.stability_encoder(x)["features"]
            
            if hasattr(self.plasticity_adapter, "extract_vector"):
                a_x = self.plasticity_adapter.extract_vector(x)
            else:
                a_x = self.plasticity_adapter(x)["features"]
            
            g = self.gate(phi_x, a_x)
            z = g * phi_x + (1.0 - g) * a_x
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
        fc = CosineLinear(self.feature_dim, nb_classes, sigma=True)
        if self.fc is not None:
            nb_output = self.fc.out_features
            fc.weight.data[:nb_output] = self.fc.weight.data[:nb_output]
            if self.fc.sigma is not None:
                fc.sigma.data = self.fc.sigma.data
        del self.fc
        self.fc = fc

    def transition_to_incremental_stage(self):
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
                if hasattr(self.extractor, "extract_vector"):
                    return self.extractor.extract_vector(x)
                else:
                    return self.extractor(x)["features"]

        if self.stability_encoder is None:
            self.stability_encoder = FrozenFeatureExtractor(self.convnet)
            # After the first incremental stage, the old base convnet is kept only
            # as the frozen stability branch. It should not appear trainable.
            for p in self.convnet.parameters():
                p.requires_grad = False
            self.convnet.eval()
        else:
            class FusedFeatureExtractor(nn.Module):
                def __init__(self, stability, plasticity, gate):
                    super().__init__()
                    self.stability = copy.deepcopy(stability)
                    self.plasticity = copy.deepcopy(plasticity)
                    self.gate = copy.deepcopy(gate)
                    for p in self.parameters():
                        p.requires_grad = False
                    self.eval()
                def extract_vector(self, x):
                    phi_x = self.stability.extract_vector(x)
                    a_x = self.plasticity.extract_vector(x)
                    g = self.gate(phi_x, a_x)
                    return g * phi_x + (1.0 - g) * a_x
                def forward(self, x):
                    return {"features": self.extract_vector(x)}

            self.stability_encoder = FusedFeatureExtractor(self.stability_encoder, self.plasticity_adapter, self.gate)
        
        class BottleneckFeatureAdapter(nn.Module):
            def __init__(self, feature_source, feature_dim, bottleneck_dim):
                super().__init__()
                self.frozen_source = copy.deepcopy(feature_source)
                for p in self.frozen_source.parameters():
                    p.requires_grad = False
                self.frozen_source.eval()
                self.adapter = nn.Sequential(
                    nn.Linear(feature_dim, bottleneck_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(bottleneck_dim, feature_dim),
                )

            def extract_vector(self, x):
                self.frozen_source.eval()
                with torch.no_grad():
                    base = self.frozen_source.extract_vector(x)
                return base + self.adapter(base)

            def forward(self, x):
                return {"features": self.extract_vector(x)}

        bottleneck_dim = int(self.args.get("adapter_bottleneck", max(8, self.feature_dim // 4)))
        self.plasticity_adapter = BottleneckFeatureAdapter(self.stability_encoder, self.feature_dim, bottleneck_dim)
        self.gate = VectorGate(self.feature_dim)
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
                proto_norm = proto / (torch.norm(proto, p=2) + 1e-8)
                self.fc.weight.data[cid] = proto_norm

    def get_trainable_incremental_params(self):
        params = []
        if self.plasticity_adapter is not None:
            params.extend(self.plasticity_adapter.parameters())
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
            if "plasticity_adapter.frozen_source" in k:
                continue
            if any(sub in k for sub in ["plasticity_adapter.adapter", "gate", "fc"]):
                state[k] = v
        return state

    def load_incremental_state_dict(self, state_dict):
        """Load only incremental parameters from a state dict."""
        own_sd = self.state_dict()
        for k, v in state_dict.items():
            if k in own_sd:
                own_sd[k] = v
        self.load_state_dict(own_sd)
