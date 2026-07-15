# AFSIC-IoV: Adaptive Personalized Federated Few-Shot Class-Incremental Learning for IoV Intrusion Detection

Phiên bản **Personalized Federated Learning cho Internet of Vehicles** của AFSIC-IDS.
Nhiều client IoV (RSU / MEC server / fleet gateway) cùng học một hệ thống IDS trên
dữ liệu CAN-bus mà không chia sẻ dữ liệu thô, trong khi:

- dữ liệu **non-IID** (label-skew) giữa các client;
- lớp tấn công mới xuất hiện **theo thời gian** (class-incremental, 5 stage);
- mỗi lớp mới chỉ có **rất ít mẫu có nhãn** (few-shot, `kshot`);
- client **tham gia động** — không phải client nào cũng có mặt ở mọi stage.

---

## 🆚 Khác biệt so với AFSIC-IDS

| Thành phần | AFSIC-IDS | AFSIC-IoV |
|---|---|---|
| Dataset | CIC-IoT23 (34 lớp, 6 task) | CAN-bus IoV (13 lớp, 5 task `[3,3,3,2,2]`) |
| Adapter + Gate | Client nạp đè bản aggregate mỗi round | **Cá nhân hóa**: client giữ bản cục bộ, không nạp đè (server vẫn giữ bản trung bình cho global model) (`personalized_adapter`) |
| Prototype dùng cho loss/calibration | Global prototype | **Personalized**: `p̃ᵢ,c = ρ·p_local + (1−ρ)·p_global`, ρ = n/(n+m) thích ứng theo số mẫu |
| Prototype aggregation | Trọng số count × quality (per-client) | **Per-class reliability**: `r_ic = β_n·log(1+n) − β_σ·σ + β_q·q − β_drift·drift − β_upd·‖Δθ‖`, `α_ic = softmax(r_ic/τ)` |
| Đánh giá | Chỉ global model | Global + **per-client F1 và personalization gain** (`metrics_per_client.csv`) |
| Tham gia client | Đầy đủ | **Động** — client thiếu file task tự động bị bỏ qua round |

Các thành phần kế thừa: frozen stability encoder + plasticity adapter + vector gate,
prototype-assisted cosine classifier, herding exemplar memory (1%), loss đa mục tiêu
(CE + KD + FSP + proto + RS + prox), quality-score aggregation với lọc MAD z-score.

---

## 📂 Dữ liệu

Đặt tại `data/`:

```
data/
├── federated_data/client_{c}_task_{t}.pt   # t = 1..5, dict {"x": (N,31) float16, "y": (N,) int64}
├── global_test_data.pt                     # test set chung
├── class_mapping.json                      # 13 lớp: Benign, DoS, double, force-neutral, ...
├── task_mapping.json                       # nhóm lớp theo 5 task
└── allocation_plan.csv                     # phân bổ mẫu client × lớp (label-skew)
```

Client không tham gia stage `t` thì không có file `client_{c}_task_{t}.pt` — trainer
tự bỏ qua. Chạy trên Kaggle sẽ tự quét `/kaggle/input`.

**RAM**: dữ liệu được giữ float16 (~6 GB cho toàn bộ train + ~3 GB test). Trên máy
yếu, đặt `max_samples_per_class` / `test_max_samples_per_class` trong config để cắt
bớt lớp đa số (Benign chiếm 99%).

---

## 🚀 Chạy

```bash
# Debug nhanh trên CPU (dữ liệu đã cắt, 2 round/task)
python main.py --config configs/exps/can_iov_debug.json

# Thực nghiệm đầy đủ (GPU, toàn bộ dữ liệu)
python main.py --config configs/exps/can_iov_afsic.json

# Test lại từ checkpoint
python main.py --config configs/exps/can_iov_afsic.json --mode test --test_checkpoint_dir logs/afsic-iov_federated/can_iov/<run_dir>
```

Kết quả mỗi run nằm trong `logs/afsic-iov_federated/can_iov/<timestamp>/`:

- `metrics_round_by_round.csv` — accuracy/precision/recall/F1 (micro/macro/weighted), loss, forgetting của global model;
- `metrics_per_client.csv` — Acc/F1 của model cá nhân hóa từng client + **personalization gain** so với global;
- `confusion_matrix_task_XX.png`, `metrics_plot.png`, checkpoint mỗi round.

---

## ⚙️ Config PerFL chính (`configs/exps/can_iov_afsic.json`)

| Key | Ý nghĩa |
|---|---|
| `personalized_adapter` | Giữ adapter + gate cục bộ tại client (không aggregate) |
| `proto_rho_adaptive`, `proto_rho_m` | ρ = n/(n+m) — nhiều mẫu → tin prototype cục bộ |
| `proto_beta_n/sigma/q/drift/update`, `tau_proto_aggregation` | Per-class reliability aggregation |
| `fewshot_enabled`, `kshot` | Số mẫu có nhãn mỗi lớp attack mới (few-shot) |
| `per_client_eval`, `per_client_eval_every` | Đánh giá per-client (0 = chỉ round cuối task) |
| `max_samples_per_class` | Cắt lớp đa số mỗi client (null = dùng toàn bộ) |
