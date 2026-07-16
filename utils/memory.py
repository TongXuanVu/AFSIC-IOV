import numpy as np
import torch

class LocalExemplarMemory:
    def __init__(self, memory_ratio=0.01, memory_per_class=None):
        self.memory_ratio = memory_ratio
        self.memory_per_class = memory_per_class
        # Mapping from class_id -> numpy array of samples
        self.data_memory = {}
        # Mapping from class_id -> numpy array of targets
        self.targets_memory = {}

    def get_memory(self):
        if not self.data_memory:
            return None
        all_data = []
        all_targets = []
        for cid in sorted(self.data_memory.keys()):
            all_data.append(self.data_memory[cid])
            all_targets.append(self.targets_memory[cid])
        return np.concatenate(all_data, axis=0), np.concatenate(all_targets, axis=0)

    def construct_exemplars(self, class_id, data, targets, features,
                            features_normalized=False, device=None):
        """Herding selection, tính theo chunk (GPU nếu có) để không tạo bản copy
        float32/float64 của toàn bộ ma trận feature — với lớp ~29M mẫu bản cũ
        cần thêm ~15GB RAM chỉ riêng bước chuẩn hóa.

        features: ma trận [N, d] (numpy hoặc tensor), float16 được khuyến khích.
        features_normalized: True nếu caller đã chuẩn hóa L2 từng hàng.
        """
        num_samples = len(data)
        if num_samples == 0:
            return

        if self.memory_per_class is not None:
            m = min(int(self.memory_per_class), num_samples)
        else:
            m = int(np.ceil(float(self.memory_ratio) * num_samples))
            m = max(1, min(m, num_samples))

        if isinstance(features, np.ndarray):
            feats = torch.from_numpy(features)
        else:
            feats = features.detach().cpu()

        use_device = torch.device("cpu")
        if device is not None and str(device) != "cpu" and torch.cuda.is_available():
            use_device = device
        feats = feats.to(use_device)

        n, d = feats.shape
        chunk = 4_000_000

        if not features_normalized:
            for s0 in range(0, n, chunk):
                block = feats[s0:s0 + chunk].float()
                block = block / (block.norm(dim=1, keepdim=True) + 1e-8)
                feats[s0:s0 + chunk] = block.to(feats.dtype)

        # class_mean: cộng dồn float64 theo chunk để ổn định số học
        acc = torch.zeros(d, dtype=torch.float64, device=use_device)
        for s0 in range(0, n, chunk):
            acc += feats[s0:s0 + chunk].double().sum(dim=0)
        class_mean = acc / n
        class_mean = (class_mean / (class_mean.norm() + 1e-8)).float()

        selected_indices = []
        S = torch.zeros(d, dtype=torch.float32, device=use_device)
        mask = torch.zeros(n, dtype=torch.bool, device=use_device)
        scores = torch.empty(n, dtype=torch.float32, device=use_device)

        for k in range(1, m + 1):
            # argmax f·(k·mean − S) tương đương argmax f·(mean − S/k) (chia k>0)
            target_vector = class_mean - S / k
            for s0 in range(0, n, chunk):
                scores[s0:s0 + chunk] = feats[s0:s0 + chunk].float() @ target_vector
            scores[mask] = -float("inf")

            i = int(torch.argmax(scores).item())
            mask[i] = True
            selected_indices.append(i)
            S += feats[i].float()

        self.data_memory[class_id] = np.array([data[idx] for idx in selected_indices])
        self.targets_memory[class_id] = np.array([targets[idx] for idx in selected_indices])


class GlobalPrototypeMemory:
    def __init__(self):
        # class_id -> { "prototype": tensor, "count": int, "dispersion": float, "quality": float }
        self.memory = {}

    def update_prototype(self, class_id, prototype, count, dispersion, quality=1.0):
        if isinstance(prototype, np.ndarray):
            prototype = torch.from_numpy(prototype).float()
        
        # Ensure L2 normalized
        norm_proto = prototype / (torch.norm(prototype, p=2) + 1e-8)
        
        self.memory[class_id] = {
            "prototype": norm_proto,
            "count": count,
            "dispersion": dispersion,
            "quality": quality
        }

    def get_prototype(self, class_id):
        if class_id in self.memory:
            return self.memory[class_id]["prototype"]
        return None

    def get_all_prototypes(self):
        # returns class_id -> prototype tensor
        return {cid: info["prototype"] for cid, info in self.memory.items()}

    def has_class(self, class_id):
        return class_id in self.memory

    def get(self, class_id):
        if class_id in self.memory:
            return self.memory[class_id]
        return None

    def get_all(self):
        return dict(self.memory)
