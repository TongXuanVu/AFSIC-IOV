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
import math

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
        """Trọng số trộn giữa prototype cục bộ và prototype toàn cục.

        Chế độ chọn qua config "proto_rho_mode":

            "share"  — ρ = n_{i,c} / n_c(toàn cục). Tỉ lệ dữ liệu lớp c mà
                       client i nắm giữ. Không cần siêu tham số, đúng thang đo
                       ở mọi kích cỡ lớp. KHUYẾN NGHỊ.
            "log"    — ρ = log(1+n) / (log(1+n) + m), với m ≈ 5–10.
                       Giữ dạng bão hòa nhưng nén được nhiều bậc độ lớn.
            "linear" — ρ = n/(n+m). Hành vi gốc; CHỈ đúng khi n cùng bậc với m.
            "fixed"  — ρ = proto_rho, không phụ thuộc dữ liệu.

        Vì sao bỏ mặc định "linear" với m=20: dữ liệu CAN-IoV trải từ 10 tới
        29 triệu mẫu (6,5 bậc độ lớn), trong khi n/(n+m) chỉ có vùng chuyển
        tiếp rộng khoảng 2 bậc quanh m. Với m=20, MỌI lớp thật đều cho ρ≈1
        (Benign 29.190.414 mẫu → 1.0000; speed-accessory 687 mẫu → 0.9717),
        nên thành phần toàn cục bị triệt tiêu và cơ chế mất hoàn toàn khả năng
        phân biệt: client giữ 3,8% dữ liệu và client giữ 30% nhận cùng ρ.
        """
        mode = self.args.get("proto_rho_mode", "linear")

        if mode == "fixed" or not self.args.get("proto_rho_adaptive", True):
            return float(self.args.get("proto_rho", 0.5))

        n = int(self.local_protos.get(class_id, {}).get("count", 0))
        if n <= 0:
            return 0.0

        if mode == "share":
            info = self.global_proto_memory.get(class_id)
            n_total = int(info.get("count", 0)) if info else 0
            # Vòng đầu của một task, server chưa tổng hợp count cho lớp mới →
            # chưa có cơ sở so sánh, tạm dựa hẳn vào tri thức toàn cục.
            if n_total <= 0:
                return 0.0
            return min(1.0, n / float(n_total))

        if mode == "log":
            m = float(self.args.get("proto_rho_m", 10.0))
            log_n = math.log1p(n)
            return log_n / (log_n + m)

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
