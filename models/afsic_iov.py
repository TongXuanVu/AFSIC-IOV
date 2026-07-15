"""
AFSIC-IoV: Adaptive Personalized Federated Few-Shot Class-Incremental Learning
cho Internet of Vehicles Intrusion Detection.

Mở rộng AFSIC-IDS với các cơ chế Personalized FL:

1. Personalized prototype mixing:
       p̃_{i,c} = ρ_{i,c} · p_{i,c}^local + (1 − ρ_{i,c}) · p_c^global
   với ρ_{i,c} thích ứng theo số mẫu cục bộ: ρ = n / (n + m)
   (client nhiều dữ liệu cho lớp c → tin prototype cục bộ hơn;
    client ít dữ liệu → dựa vào tri thức toàn cục).
   p̃ được dùng cho cả loss FSP/proto lẫn calibration classifier.

2. Personalized adapter + gate: khi bật config "personalized_adapter",
   plasticity adapter và vector gate KHÔNG được aggregate về server
   (xử lý ở utils/aggregation.py + trainer.py), mỗi client giữ bản riêng.

Config liên quan:
    proto_rho_adaptive (bool, mặc định true) — ρ thích ứng theo n_{i,c}
    proto_rho          (float, mặc định 0.5) — ρ cố định khi không thích ứng
    proto_rho_m        (float, mặc định 20)  — hằng số bão hòa n/(n+m)
"""
import torch

from models.afsic_ids import AFSIC_IDS


class AFSIC_IoV(AFSIC_IDS):
    def __init__(self, args):
        super().__init__(args)
        # Prototype cục bộ mới nhất của chính client này: {class_id: {"prototype", "count", "dispersion"}}
        self.local_protos = {}

    def compute_local_prototypes(self, data_manager, class_ids=None, max_samples_per_class=None, seed=0):
        protos = super().compute_local_prototypes(
            data_manager, class_ids=class_ids,
            max_samples_per_class=max_samples_per_class, seed=seed,
        )
        self.local_protos.update(protos)
        return protos

    def _personalization_rho(self, class_id):
        if not self.args.get("proto_rho_adaptive", True):
            return float(self.args.get("proto_rho", 0.5))
        n = int(self.local_protos.get(class_id, {}).get("count", 0))
        if n <= 0:
            return 0.0
        m = float(self.args.get("proto_rho_m", 20.0))
        return n / (n + m)

    def get_personalized_prototype(self, class_id):
        """p̃_{i,c} = ρ·p_local + (1−ρ)·p_global, chuẩn hóa L2."""
        global_p = self.global_proto_memory.get_prototype(class_id)
        local_p = self.local_protos.get(class_id, {}).get("prototype")
        if local_p is None:
            return global_p
        if global_p is None:
            return local_p
        rho = self._personalization_rho(class_id)
        mixed = rho * local_p + (1.0 - rho) * global_p.to(local_p.dtype)
        return mixed / (torch.norm(mixed, p=2) + 1e-8)

    def _get_reference_prototype(self, class_id):
        # Loss FSP/proto và calibration dùng prototype cá nhân hóa
        return self.get_personalized_prototype(class_id)

    def get_calibration_prototypes(self):
        """Dict prototype dùng để khởi tạo trọng số classifier.

        Với global model (không có local_protos) kết quả trùng với
        global prototype — hành vi AFSIC-IDS gốc.
        """
        protos = {}
        for c in range(self._total_classes):
            p = self.get_personalized_prototype(c)
            if p is not None:
                protos[c] = p
        return protos
