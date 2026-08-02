# AFSIC-IoV — Kế hoạch chạy lại

> Viết lại ngày 2026-08-02, sau khi phát hiện lỗi resume làm hỏng toàn bộ thí nghiệm từ 31-07.
> Phạm vi: **chỉ IoV**, federated class-incremental, 10 client, 3 kịch bản (full / 1% / 10-shot).
>
> **Trạng thái: đã vá** (commit `5bcc567`). Xem mục 1 và 3.

---

## 1. Lỗi đã tìm ra — và bản vá

### Gốc rễ

Checkpoint chỉ lưu `model_state_dict` của **global**, còn với client thì chỉ lưu memory:

```python
local_models[c]._data_memory    = c_state.get('data_memory')
local_models[c]._targets_memory = c_state.get('targets_memory')
local_models[c].local_memory    = c_state['local_memory']
```

Trọng số mạng riêng của client không được lưu. Bình thường không sao, vì vòng lặp
build memory cuối task 0 gọi `_load_global_into_client(..., task=0, ...)`, và ở task 0
hàm này nạp **toàn bộ** trạng thái global vào mọi client.

Nhưng có hai đường làm vòng lặp đó không chạy:

1. Resume từ `ckpt_task00_memory_client09.pth` — `last_client_done=9` →
   `current_client_start=10` → vòng lặp bỏ qua hoàn toàn.
2. Resume từ **bất kỳ checkpoint nào ở task ≥ 1** — `if task < start_task: continue`
   bỏ qua mọi task trước, kể cả vòng lặp build memory của task 0.

Trong khi đó `transition_to_incremental_stage()` vẫn chạy cho mọi task:

```python
if self.stability_encoder is None:
    self.stability_encoder = FrozenFeatureExtractor(self.convnet)   # copy convnet HIỆN TẠI
```

Nếu `convnet` của client chưa từng được nạp từ global thì `stability_encoder` sinh ra
từ trọng số ngẫu nhiên. Và `_PERSONALIZED_KEY_MARKERS = ("stability_encoder",
"plasticity_adapter", "gate")` khiến ba thứ đó **không bao giờ** được nạp đè ở task ≥ 1.

### Bằng chứng

| Resume từ | Loss round 1 task 1 | macro-F1 task 1 (10-shot) |
|---|---|---|
| `ckpt_round0030` | **1.601** | **24.33** |
| `ckpt_task00_memory_client09` | 3.273 | 5.95 |

Cùng kịch bản, cùng seed 42, cùng memory (`Exemplar size` trùng khít), chỉ khác điểm resume.

### Bản vá (commit `5bcc567`)

Lưu thêm trọng số mạng của từng client vào `client_states`, và nạp lại khi resume:

```python
# khi lưu (cả checkpoint mỗi round lẫn checkpoint memory)
'net': {k: v.cpu() for k, v in local_models[c]._network.state_dict().items()},

# khi resume
if c_state.get('net') is not None:
    local_models[c]._network.load_state_dict(c_state['net'], strict=False)
    local_models[c]._network.to(args["device"][0])
```

Chi phí: **+2,95 MB mỗi checkpoint** (72 → 75 MB, tăng 4,1%).

Checkpoint tạo bởi code cũ vẫn nạp được, nhưng sẽ in cảnh báo `CHECKPOINT CŨ —
không có trọng số client`. Thấy dòng đó nghĩa là checkpoint không dùng được, phải
quay về `ckpt_round0030`.

### Phạm vi thiệt hại

| Run | Trạng thái |
|---|---|
| `full/17-07-26_23-18` — task 0 | **hợp lệ** (không resume) |
| `10shot/30-07-26_06-43` — task 1 tới round 23 | **hợp lệ** |
| `full/31-07-26_09-31` | hỏng — đã gỡ checkpoint |
| `01-08-26_05-22`, `01-08-26_06-01` (debug) | hỏng — đã gỡ checkpoint |
| `1%/01-08-26_07-20`, `10shot/01-08-26_07-20` | hỏng — đã gỡ checkpoint |

**Chưa từng có run full data hợp lệ.** Mọi kết luận trước đây về "full data sụp về lớp
mới", "cả hai cấu hình đều hỏng" đều dựa trên run dính lỗi này và phải bỏ.

---

## 2. Đã dọn dẹp (2026-08-02)

**Local** — gỡ thư mục `checkpoints/` của 5 run hỏng, đổi tên thư mục thành hậu tố
`_INVALID`, giữ lại `training.log` + `metrics_*.csv` + confusion matrix để tra cứu.
Giải phóng **12,7 GB** (15 GB → 2,3 GB).

**GitHub** — gỡ `ckpt_round0075_task02_r015_acc0.0.pth`, `ckpt_task01_memory_client09.pth`,
`metrics_full_r31-75.csv`. Thêm `resume_checkpoints/iov10/README.md` cảnh báo cách resume.

---

## 3. Quy tắc resume sau khi vá

**Lần đầu mỗi kịch bản** — resume từ `ckpt_round0030_task00_r030_acc100.0.pth`.
Đây là session duy nhất phải chịu ~1 giờ herding.

**Các session sau** — resume từ checkpoint round mới nhất
(`ckpt_round{NNNN}_task{TT}_r{RRR}_acc*.pth`). Không herding lại, và nhờ bản vá thì
nhánh cá nhân hoá của client được phục hồi đúng.

**Tuyệt đối không** resume từ `ckpt_task00_memory_client09.pth`. Nó vẫn nằm trong repo
để tham khảo memory đã build, nhưng dùng để resume sẽ dính lại đúng lỗi cũ (xem
`resume_checkpoints/iov10/README.md`).

**Mọi checkpoint tạo trước commit `5bcc567` đều bỏ.** Chúng không có trường `net`.

---

## 4. Nguyên tắc đọc kết quả

Tập test lệch 18.500:1 (Benign chiếm 99,2%), nên **accuracy gần như vô nghĩa**.
Mọi phán quyết dựa trên **macro-F1 so với ngưỡng sụp** — tức macro-F1 mà một mô hình
chỉ đoán Benign sẽ đạt được:

| Task | Số lớp | Ngưỡng sụp (macro-F1) | Ngưỡng sụp (macro-recall) |
|---|---|---|---|
| 0 | 3 | 33.27% | 33.33% |
| 1 | 6 | 16.61% | 16.67% |
| 2 | 9 | 11.07% | 11.11% |
| 3 | 11 | 9.06% | 9.09% |
| 4 | 13 | 7.66% | 7.69% |

Cách đọc: **trên ngưỡng** = mô hình thực sự phân biệt được lớp; **bằng ngưỡng** = sụp
về lớp đa số; **dưới ngưỡng** = sụp về lớp hiếm, tệ nhất.

Bảng kết quả từ giờ phải có cột ngưỡng bên cạnh macro-F1.

---

## 5. Số liệu hợp lệ hiện có

| Kịch bản | Task | acc | macro-F1 | ngưỡng | |
|---|---|---|---|---|---|
| tất cả (chung task 0) | 0 | 99.96 | **72.75** | 33.27 | học thật, gấp 2,2× ngưỡng |
| 10-shot | 1 (r23/30) | 1.24 | **24.33** | 16.61 | học thật |

Task 0 dùng chung cho cả ba kịch bản vì `fewshot_enabled` chỉ tác động từ task 1.

---

## 6. Việc phải chạy

Mỗi kịch bản: 4 task × 30 round = **120 round**, resume từ `ckpt_round0030`.

| # | Kịch bản | Config | Ghi chú |
|---|---|---|---|
| 1 | 10-shot | `can_iov_afsic.json` | có run hợp lệ tới task 1 r23; chạy lại từ đầu task 1 cho liền mạch |
| 2 | 1% | `can_iov_afsic_fewshot1.json` | `kshot: 0.01` |
| 3 | full | `can_iov_afsic_full.json` | `fewshot_enabled: false` |

Ba kịch bản độc lập, chạy song song trên ba tài khoản Kaggle.

### Cell chuẩn

```python
!rm -rf /kaggle/working/AFSIC-IOV
!git clone https://github.com/TongXuanVu/AFSIC-IOV.git /kaggle/working/AFSIC-IOV
%cd /kaggle/working/AFSIC-IOV

!python main.py --config configs/exps/<CONFIG>.json \
    --resume resume_checkpoints/iov10/<CHECKPOINT>.pth
```

`<CHECKPOINT>`: session đầu dùng `ckpt_round0030_task00_r030_acc100.0.pth`;
các session sau dùng checkpoint round mới nhất đã push.

### Lịch dự kiến mỗi kịch bản

| Session | Nội dung | Thời gian |
|---|---|---|
| 1 | herding + 30 round task 1 (12,1 ph/round) | ~7,0 h |
| 2 | 30 round task 2 (21,2 ph/round) | ~10,6 h |
| 3 | 30 round task 3 (~30 ph/round, ước) | ~15 h → 2 session |
| 4 | 30 round task 4 (~33 ph/round, ước) | ~16,5 h → 2 session |

Khoảng **5 session** mỗi kịch bản. Session Kaggle tối đa 12 giờ — **push checkpoint
quanh giờ thứ 11**, nếu không mất toàn bộ tiến độ session đó.

### Kiểm tra ngay khi log ra

1. `Auto-detected Test File` phải trỏ tới `fcil-iov` — **không** phải `iot100client`
   (IoV có 31 feature, IoT có 33). Thấy `test=(..., 33)` là sai dataset.
2. `Exemplar size` phải là `292410 / 293047 / 293393 / 60142 / 37705 / 0 / 0 / 0 / 0 / 0`.
3. **Loss round 1 task 1 phải ~1.6**, không phải ~3.3.
4. Từ session 2 trở đi: phải thấy `Đã phục hồi trọng số riêng cho 10/10 client`.
   Nếu thay vào đó hiện `CHECKPOINT CŨ` thì checkpoint tạo bởi code trước bản vá —
   phải chạy lại từ `ckpt_round0030`.

Điểm 3 và 4 là chốt chặn cho đúng lỗi đã gặp — sai là dừng ngay, đừng chạy tiếp 12 giờ.

---

## 7. Chi phí và rủi ro

**Thời gian:** 21 phút/round ở task 2, tăng dần theo task. Nguyên nhân: mỗi round,
mỗi client, `compute_local_prototypes` duyệt **toàn bộ** dữ liệu lớp cũ — client 0 có
29,2 triệu mẫu Benign, tức 3.565 batch, trong khi bước train thật chỉ ~40 batch.
Ước tính phân bổ ở task 2 (7 client active): prototype lớp cũ ~24.955 batch (83%),
đánh giá test ~10.000 batch, train ~280 batch (dưới 1%).

Có thể rút xuống 3–6 phút/round bằng cách lấy mẫu con để ước lượng vector trung bình
(sai số ~0,7% với n = 20.000). **Nhưng phải cẩn thận**: `count` báo về server đi vào
`r_ic = β_n·log1p(n_ic)` rồi qua softmax thành trọng số tổng hợp prototype. Nếu để
việc cắt mẫu ghi đè `count`, trọng số bị san phẳng (log1p(20.000)=9,90 cho mọi client
thay vì 17,19/15,61/15,13) và kết quả đổi hoàn toàn — đã gặp đúng lỗi này khi thêm
`proto_max_samples` lần đầu. Muốn tối ưu thì phải tách hai khái niệm: số mẫu dùng để
tính trung bình, và số mẫu thật báo cáo.

**Đĩa:** checkpoint task 0 chỉ 107 KB nhưng từ task 1 là **75 MB/round** (kèm exemplar
và trọng số client). 120 round ≈ 9,0 GB, cộng memory checkpoint cuối mỗi task.
Dưới 20 GB nhưng sát.

**Chưa có cơ chế giữ N checkpoint gần nhất** như HFIN (`KEEP_LAST_CKPTS = 5`).

---

## 8. Sau khi có kết quả 3 kịch bản

**Baseline** — dựng bằng config, không port repo ngoài. Tài liệu thiết kế
(`AFSIC-IoV.docx`, mục "Giai đoạn 4") yêu cầu FedAvg, FedProx, FedPer, FedProto.
Repo đã có sẵn đủ cờ:

| Baseline | Cách dựng |
|---|---|
| FedAvg | `aggregate_backbone: true`, `personalized_adapter: false`, `calibrate_with_prototypes: false`, mọi `lambda_*: 0`, mọi `beta_*: 0` |
| FedProx | như trên, giữ `lambda_prox: 0.01` |
| FedPer | `personalized_adapter: true`, `lambda_*: 0`, không calibrate |
| FedProto | giữ prototype + calibrate, `personalized_adapter: false` |
| AFSIC-IDS | `proto_rho_mode: fixed`, `proto_rho: 0` |
| Fine-tuning | `memory_ratio: 0` (không replay) |

Cách này kiêm luôn **ablation study** mà mục 9 tài liệu yêu cầu.

**Metric còn thiếu** — tài liệu yêu cầu "per-client F1" và "so sánh global model vs
personalized model". Hiện `per_client_eval_every: 0` nên chỉ ghi một điểm ở cuối mỗi task.
Ngoài ra chỉ client 0 có test data, 9 client còn lại `test=(0, 31)`, khiến
`_build_local_eval_loader` **fallback sang train data** — mà `client_accs` từ đó lại đi
thẳng vào `Q_i` rồi vào trọng số `α`. Muốn báo cáo per-client đúng nghĩa thì phải phân
hoạch tập test theo client trước.

---

## 9. Nghi vấn còn treo

Chưa kết luận được, để lại sau khi có số liệu sạch:

- Full data có thực sự kém hơn 10-shot không (mọi số liệu cũ đều hỏng).
- `_calibrate_classifier_from_prototypes` ghi đè classifier đã train mỗi round.
- Client không có exemplar lớp cũ (client 5, `Exemplar size: 0`) vẫn nhận α ≈ 0.20.
- `client_accs` đo trên tập lệch / train data → `Q_i` và `α` không đáng tin.

Kịch bản 1% sẽ nói nhiều về nghi vấn đầu: số mẫu lớp mới mà client 0 nhận ở task 1 là
20 (10-shot) / 390 (1%) / 39.018 (full), tức tỉ lệ exemplar:mới là 14.620 : 750 : 7,5.
Ba điểm này đủ để thấy hiệu năng suy giảm theo hướng nào.

---

## 10. Bài học ghi lại

Ba lần kết luận sai trong dự án này, đều cùng một dạng — đọc một chỉ số mà không kiểm
tra chỉ số bổ trợ:

1. **F2SCIL** bị đánh giá hỏng sau 7 round; chạy tiếp tới 60 round thì đạt 99,40%.
2. **10-shot IoV** bị gọi là "catastrophic forgetting cực đoan" vì accuracy 1,24%;
   macro-F1 24,33 cho thấy nó vượt ngưỡng sụp và đang hoạt động.
3. **`proto_max_samples`** được gọi là "đột phá" vì `Old Acc` 96,76%; macro-F1 16,55
   khớp chính xác công thức của mô hình sụp về một lớp.

Và một lần nữa với chính lỗi resume: xác nhận checkpoint memory **bỏ qua được bước
herding**, nhưng không kiểm tra nó có **mang đủ trạng thái cần thiết** hay không.
Kiểm tra "cơ chế có chạy đúng như thiết kế" không thay được kiểm tra "kết quả có hợp lý".

Nguyên tắc rút ra: mỗi phép thử phải có tiêu chí phán quyết đặt trước, đổi một biến mỗi
lần, không kết luận dưới 5 round, và luôn đối chiếu macro-F1 với ngưỡng sụp.
