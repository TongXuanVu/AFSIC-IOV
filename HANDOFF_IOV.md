# AFSIC-IoV — Tài liệu bàn giao (handoff)

> Mục đích: mọi thứ cần biết để tiếp tục chạy AFSIC-IoV trên bộ dữ liệu CAN-bus 10 client mà không phải dò lại. Viết ngày 2026-07-31.

---

## 1. Bối cảnh

Chạy AFSIC-IoV (Federated Class-Incremental Learning có personalized adapter) trên dữ liệu **CAN-bus IoV**, 10 client, 5 task, 13 lớp. Đây là phần tiếp nối sau khi đã hoàn thành nhánh IoT (CIC-IoT23, 100 client) — kết quả IoT nằm ở `C:\FederatedLearning\Tổng hợp kết quả\iot100\`.

Chạy trên **Kaggle** (T4, giới hạn 12h/session, 20 GB đĩa), code đồng bộ qua **GitHub**, resume bằng checkpoint.

- Repo: `https://github.com/TongXuanVu/AFSIC-IOV`
- Kaggle dataset: `/kaggle/input/datasets/tongxuanvu/fcil-iov/data/`
- Thư mục local: `C:\FederatedLearning\AFSIC-IOV`

---

## 2. Dữ liệu

`C:\FederatedLearning\AFSIC-IOV\data\` (bản Kaggle: dataset `fcil-iov`)

```
data/
├── global_test_data.pt          # test set toàn cục, 42,048,683 mẫu × 31 feature
├── federated_data/
│   └── client_{0..9}_task_{1..5}.pt
├── class_mapping.json
├── task_mapping.json
├── allocation_plan.csv
└── summary.json
```

**Thông số:** 98,113,593 mẫu train · 31 feature · lưu `float16` · 13 lớp · 10 client.

**Nhãn đã tuần tự sẵn (0–12) — KHÔNG cần remap.** Đây là khác biệt lớn so với IoT 100-client (bên đó thứ tự task bị xáo nên phải dùng LUT `task_mapping_label_ids.json`).

### Ánh xạ lớp và task

| Task | Lớp (id) | Số mẫu train | Số client có data |
|---|---|---|---|
| 0 | Benign (0), DoS (1), double (2) | 97,668,978 | 5 |
| 1 | force-neutral (3), fuzzing (4), interval (5) | 265,083 | 5 |
| 2 | rpm (6), rpm-accessory (7), speed (8) | 36,540 | 6 |
| 3 | speed-accessory (9), standstill (10) | 18,658 | 8 |
| 4 | systematic (11), triple (12) | 124,334 | 5 |

`task_increments: [3, 3, 3, 2, 2]` — **không đều nhau**, đừng giả định 3 lớp/task.

### Mất cân bằng cực đoan

Benign = 97,301,380 mẫu = **99.17%** toàn bộ train. Tỉ lệ Benign : lớp hiếm nhất (speed-accessory, 5,267 mẫu) là **18,500 : 1**.

Hệ quả trực tiếp: **accuracy gần như vô nghĩa**, luôn phải đọc kèm macro-F1 và macro-recall. Một model chỉ đoán Benign vẫn đạt ~99% accuracy.

### Client không đồng đều

Từ `allocation_plan.csv`: client 0–2 mỗi client ~29 triệu mẫu; client 6 chỉ 7,208 mẫu; client 7 chỉ 4,414. Chênh **6,600 lần**. Nhiều client không có file cho một số task — code tự bỏ qua và in `Client N không có file ... Tự động bỏ qua.` Đây là hành vi bình thường, không phải lỗi.

Chỉ **client 0** có test data; 9 client còn lại `test=(0, 31)`. Đánh giá toàn cục dùng `global_test_data.pt`.

---

## 3. Hai kịch bản — CẠM BẪY QUAN TRỌNG NHẤT

Repo có sẵn **hai config**, khác nhau đúng **một tham số**:

| File | `fewshot_enabled` | Kịch bản |
|---|---|---|
| `configs/exps/can_iov_afsic.json` | `true` | **10-shot** |
| `configs/exps/can_iov_afsic_full.json` | `false` | **full data** |

**Config gốc mặc định là 10-shot, không phải full data.** Khác với AFSIC-IDS bên IoT (`fewshot_enabled: false`). Đã có một lần chạy nhầm vì tưởng mặc định là full.

Cơ chế (`trainer.py:559`):

```python
args.get("kshot", 10) if task > 0 and args.get("fewshot_enabled", True) else None
```

→ **Task 0 luôn dùng full data** bất kể cờ này. Few-shot chỉ áp từ task 1. Nghĩa là **checkpoint task 0 dùng chung được cho cả hai kịch bản** — không cần chạy lại task 0 khi đổi sang full.

---

## 4. Trạng thái hiện tại (2026-07-31)

### Task 0 — XONG (dùng chung cho mọi kịch bản)

Round 30/30: **acc 99.96 · macro-F1 72.75 · macro-recall 70.40 · loss 0.003047**

Artifact: `resume_checkpoints/iov10/ckpt_round0030_task00_r030_acc100.0.pth` + `metrics_r1-30.csv` (đã trên GitHub).

Log đầy đủ + 10 file memory: `logs/afsic-iov_federated/can_iov/10client/full/17-07-26_23-18_seed42_cnn1d_clients10/`

### Kịch bản 10-shot — dở dang, task 1 round 23/30

`logs/.../10client/10shot/30-07-26_06-43_seed42_cnn1d_clients10/`

Diễn biến task 1:

| Round | acc | prec_wei | f1_mac |
|---|---|---|---|
| 1 | 95.35 | 99.53 | 39.95 |
| 2 | 92.88 | 99.52 | 38.87 |
| 3 | **2.48** | 98.86 | 26.10 |
| 4–23 | ~1.24 | 97.89 | 24.33 |

**Đây không phải lỗi code.** Accuracy sập nhưng `prec_wei` vẫn 97.89% — dấu hiệu điển hình của việc model thôi không dự đoán Benign nữa. Với `kshot=10`, mỗi client chỉ có 30 mẫu cho 3 lớp mới, trong khi task 0 có 97 triệu mẫu Benign. Model lệch hẳn sang lớp mới → quên sạch lớp đa số → accuracy = tỉ lệ lớp đa số bị mất. Đây là kết quả thật của chế độ 10-shot trên dữ liệu mất cân bằng 18.500:1.

### Kịch bản full data — đang chạy

Bắt đầu 2026-07-31 06:40, đang ở bước xây Rehearsal Memory cuối task 0 (~1 giờ). Còn 120/150 round.

### Kịch bản 1% — chưa chạy

Hiện `kshot` **chỉ nhận số mẫu tuyệt đối**, chưa hỗ trợ tỉ lệ %. Muốn có kịch bản 1% (để đối xứng với bộ IoT) thì phải sửa `trainer.py:68–87` cho phép `kshot` là float < 1 hiểu theo tỉ lệ.

---

## 5. Chạy trên Kaggle

### Cell full data

```python
!rm -rf /kaggle/working/AFSIC-IOV
!git clone https://github.com/TongXuanVu/AFSIC-IOV.git /kaggle/working/AFSIC-IOV
%cd /kaggle/working/AFSIC-IOV

!python main.py --config configs/exps/can_iov_afsic_full.json \
    --resume resume_checkpoints/iov10/ckpt_round0030_task00_r030_acc100.0.pth
```

### Cell 10-shot

Giống hệt, chỉ đổi `--config configs/exps/can_iov_afsic.json`.

### Bắt buộc: chỉ attach ĐÚNG dataset `fcil-iov`

`utils/data_can_iov.py:36–46` quét `/kaggle/input/**/global_test_data.pt` rồi **lấy kết quả đầu tiên**. Nếu notebook attach cả `iot100client` thì nó vớ nhầm dữ liệu IoT và crash:

```
Auto-detected Test File: .../iot100client/100client/global_test_data.pt
Client 0: train=(0, 31), test=(14002687, 33)
IndexError: index 31 is out of bounds for axis 0 with size 13
```

Nhận diện nhanh: **IoV có 31 feature, IoT có 33**. Thấy `test=(..., 33)` là sai dataset. Thấy `train=(0, 31)` là không tìm ra data train.

Log đúng phải là:

```
Auto-detected Test File: .../fcil-iov/data/global_test_data.pt
Client 0: train=(29304512, 31), test=(42048683, 31)
```

---

## 6. Checkpoint và resume

### Đặt tên

- Mỗi round: `ckpt_round{global+1:04d}_task{task:02d}_r{round+1:03d}_acc{top1:.1f}.pth`
- Cuối mỗi task: `ckpt_task{task:02d}_memory_client{c:02d}.pth` (10 file, một file/client)

### Cơ chế resume (`trainer.py:365–377`)

Đọc `checkpoint['task']` và `checkpoint['round']` → `start_round = round + 1`. Nếu checkpoint có `is_memory_phase=True` thì nhảy thẳng sang task kế.

Khi `--resume` được truyền, CSV mở chế độ **append** (`"a"`) và **không ghi lại header**. Nên các file `metrics_round_by_round.csv` từ nhiều lần resume phải nối thủ công khi tổng hợp.

### Bỏ qua 1 giờ herding

Bước "Xây dựng Rehearsal Memory cho các Clients tại cuối Task 0" mất **~1 giờ** (herding 1% của 29 triệu mẫu Benign). Nếu copy sẵn 10 file `ckpt_task00_memory_client*.pth` vào `run/` thì bỏ qua được.

10 file đó đang ở: `logs/.../10client/full/17-07-26_23-18_*/checkpoints/` — tổng **623 MB**. Quá lớn cho git; nên upload thành Kaggle dataset riêng.

### ⚠️ Rủi ro đầy đĩa

| | Kích thước/checkpoint |
|---|---|
| Task 0 | 107 KB |
| Task 1+ | **72 MB** (gấp 670 lần) |

Vì từ task 1 checkpoint bao gồm cả exemplar memory. 120 round còn lại → **~8.6 GB**, cộng memory checkpoint cuối mỗi task (623 MB × 4) → có thể chạm **11 GB**. Kaggle cho 20 GB nên vẫn lọt, nhưng sát.

**Chưa có cơ chế giữ N checkpoint gần nhất** — HFIN đã có (`KEEP_LAST_CKPTS = 5` trong `HFIN/IDPS/main.py`), AFSIC-IoV thì chưa. Nên port sang. Bài học: hồi chạy HFIN đã từng đầy đĩa làm **hỏng cả hai checkpoint round 11** (file zip không hợp lệ), phải lùi về round 10.

---

## 7. Cấu trúc log

```
logs/afsic-iov_federated/can_iov/
├── 10client/
│   ├── full/    ← task 0 (dùng chung) + kịch bản full
│   ├── 10shot/
│   └── 1%/      (rỗng)
└── 100client/   (rỗng — chưa có kịch bản 100 client cho IoV)
```

Mỗi run: `{DD-MM-YY_HH-MM}_seed42_cnn1d_clients10/` chứa `training.log`, `metrics_round_by_round.csv`, `metrics_per_client.csv`, `checkpoints/`.

Lưu ý: Kaggle ghi log vào `logs/afsic-iov_federated/can_iov/{timestamp}/` (**không có** cấp `10client/full/`). Khi tải về phải tự xếp vào đúng thư mục kịch bản.

### Schema CSV

```
task,round,global_round,method,acc,prec_mic,prec_mac,prec_wei,rec_mic,rec_mac,rec_wei,f1_mic,f1_mac,f1_wei,loss,avg_acc
```

Trùng schema với HFIN bên IoT, chỉ thêm `method` và `avg_acc`. Khi tổng hợp cuối cùng, quy ước dùng bên IoT là:

```
task_id round_in_task global_round acc prec_mic prec_mac prec_wei rec_mic rec_mac rec_wei f1_mic f1_mac f1_wei loss
```

metric = **%** làm tròn **2 chữ số**, loss = **4 chữ số**.

---

## 8. Cạm bẫy đã gặp

| Vấn đề | Nguyên nhân | Cách tránh |
|---|---|---|
| `IndexError: index 31 out of bounds for size 13` | Attach nhầm dataset IoT, glob vớ phải file đầu tiên | Chỉ attach `fcil-iov`. Kiểm tra dòng `Auto-detected Test File` |
| Chạy nhầm 10-shot khi tưởng full | Config gốc mặc định `fewshot_enabled: true` | Dùng `can_iov_afsic_full.json` |
| Mất 1 giờ herding lại | Push thiếu 10 file `ckpt_task00_memory_client*.pth` | Copy sẵn memory vào `run/` |
| Log task 0 nằm lẫn trong thư mục 10-shot | Task 0 dùng chung nhưng run 30-07 build lại memory | Memory task 0 để ở `full/`, xác thực bằng `md5sum` (bản build lại tất định, giống hệt) |
| Accuracy 1.24% ở 10-shot | Không phải bug — model quên lớp đa số | Đọc `prec_wei` cao + `acc` thấp là nhận ra ngay |

### Hai sai lầm phân tích cần tránh lặp lại

**Đừng kết luận từ quá ít round.** Đã từng tuyên bố F2SCIL là baseline hỏng chỉ sau 7 round; sau khi tối ưu tốc độ nó chạy 60 round và đạt 99.40%/78.03% ở task 0. Đường cong FL nhiễu mạnh ở giai đoạn đầu.

**Đừng suy diễn dữ liệu khi thiếu file.** Đã từng tự bịa thứ tự lớp của kịch bản 10-client rồi kết luận sai rằng hai kịch bản dùng thứ tự khác nhau. Thiếu dữ liệu thì đi tìm hoặc hỏi, không suy đoán.

---

## 9. Việc còn lại

1. **Chạy tiếp full data** — 120/150 round (task 1→4)
2. **Chạy tiếp 10-shot** — task 1 còn 7 round, rồi task 2–4 (127 round)
3. **Port cơ chế giữ 5 checkpoint gần nhất** từ HFIN sang, trước khi đĩa đầy
4. **Upload memory task 0 (623 MB) thành Kaggle dataset** để khỏi herding lại mỗi lần
5. Cân nhắc kịch bản **1%** — cần sửa code cho `kshot` nhận tỉ lệ
6. Khi xong: tổng hợp CSV theo quy ước ở mục 7, điền vào `Thống kê tổng hợp.xlsx`

---

## 10. Tham chiếu chéo

Quy ước tổng hợp kết quả, script và các cạm bẫy chung của cả dự án nằm ở:

- `C:\FederatedLearning\Tổng hợp kết quả\iot100\README.md`
- `C:\FederatedLearning\Tổng hợp kết quả\iot100\aggregate.py`
- `C:\FederatedLearning\Tổng hợp kết quả\iot100\fill_excel.py`

Các repo GitHub liên quan: `AFSIC-IOV`, `AFSIC-IDS`, `HFIN`, `LCwoF-FL`, `MalCL-FL`, `FFSCIL`, `F2SCIL-SDD` (tất cả dưới tài khoản `TongXuanVu`).
