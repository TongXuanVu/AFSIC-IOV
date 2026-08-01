# AFSIC-IoV — Kế hoạch chẩn đoán và sửa

> Lập ngày 2026-08-01. Bối cảnh: full data task 1 chỉ đạt acc 4.19% / macro-F1 1.35%, task 2 sụp về 0.02%, trong khi AFSIC-IDS trên IoT đạt 96.16% / 68.91% với cùng siêu tham số.

---

## 0. Hiện trạng — cái gì đã biết chắc

### Số liệu mốc (task 1, full data)

| Trạng thái mô hình | Acc | Ghi chú |
|---|---|---|
| Client 0 sau train cục bộ, **trước** aggregation | 99.35% | `aggregation.py`, trên test của client 0 |
| Client 0 **sau** khi nạp global về (personalized) | 31.08% | `metrics_per_client.csv` round 30 |
| Mô hình global | 4.19% | `metrics_round_by_round.csv` round 30 |

Chi tiết round 5: `Acc 0.20% | Old Acc 0.04% | New Acc 58.15%` — lớp mới học được, lớp cũ mất sạch.

### Đã loại trừ

**Giả thuyết ρ = n/(n+m) với m=20.** Đã chạy `can_iov_afsic_full_rho0.json` (ρ=0, prototype hoàn toàn global). Kết quả 5 round đầu: 0.04 / 0.10 / 0.27 / 0.19 / 0.20 — trùng khít baseline 0.06 / 0.13 / 0.27 / 0.18 / 0.17. **ρ không phải nguyên nhân.**

**Test set sai.** Code lọc đúng: `learned_classes = np.arange(0, total_classes)`, log xác nhận `filtered to learned classes: 0-5`. Không chấm điểm trên lớp chưa học.

**Memory task 0 hỏng.** `Exemplar size` giống hệt giữa các run (292410 / 293047 / 293393), md5 checkpoint trùng nhau.

### Nghi can còn lại

| # | Nghi can | Bằng chứng |
|---|---|---|
| A | Client không có memory lớp cũ vẫn nhận ~20% trọng số | Client 5: `Exemplar size: 0`, `classes=[3,4,5,6,8,10,11]` (không có 0,1,2), nhưng `alpha_5 = 0.2046` |
| B | `_calibrate_classifier_from_prototypes` ghi đè classifier đã train | 99.35% → 31.08% xảy ra đúng ở bước `_load_global_into_client` + calibrate |
| C | Backbone không được aggregate từ task 1 | `is_aggregated_state_key` chỉ nhận `adapter`, `gate`, `fc` |

Lưu ý: nghi can C tồn tại y hệt trong AFSIC-IDS (cùng hàm, cùng dòng) mà IDS vẫn đạt 96.16%, nên C ít khả năng là nguyên nhân đơn lẻ — nhưng có thể là điều kiện cần khi kết hợp A hoặc B.

### Bối cảnh thiết kế (từ `AFSIC-IoV.docx`)

Tài liệu đặt mục tiêu **"học mô hình cá nhân hóa cho từng client"**, lấy FedPer làm baseline, contribution #2 là "stable encoder + personalized adapter + gated fusion". Nên global model thấp là **đặc tính của họ phương pháp**, không hẳn là lỗi. Mục "Kết quả cần nộp" yêu cầu **per-client F1** và **so sánh global model vs personalized model** — cả hai đều phải có.

Nhưng personalized 31.08% vẫn quá thấp so với 99.35% trước aggregation, nên vẫn còn lỗi thật cần tìm.

---

## Giai đoạn 0 — Làm cho việc thử nghiệm khả thi

**Vấn đề:** hiện 21 phút/round → mỗi phép thử 5 round mất gần 2 giờ. Không thể chẩn đoán với vòng lặp chậm như vậy.

**Nguyên nhân:** mỗi round, mỗi client, `compute_local_prototypes` duyệt **toàn bộ** dữ liệu lớp cũ (client 0 có 29,2 triệu mẫu Benign → 3.565 batch), trong khi bước train thật chỉ ~40 batch. Cộng thêm đánh giá trên test 42 triệu mẫu.

**Việc cần làm** — chỉ áp dụng cho config debug, không đụng config production:

1. Thêm tham số `proto_max_samples` (mặc định `None` = giữ nguyên hành vi cũ), truyền vào lời gọi `old_protos` ở `trainer.py` dòng ~450 và ~550.
2. Tạo `configs/exps/can_iov_debug_diag.json` với:
   - `proto_max_samples: 20000`
   - `test_max_samples_per_class: 50000`
   - `num_rounds: 5`

**Sai số chấp nhận được:** prototype là trung bình của feature đã chuẩn hoá L2; với n=20.000 sai số lấy mẫu ~0,7%. Test 50.000 mẫu/lớp vẫn cho ước lượng accuracy chính xác tới ~0,2%.

**Kết quả mong đợi:** round từ 21 phút xuống 1–2 phút → mỗi phép thử dưới 15 phút.

**Quan trọng:** config debug chỉ dùng để **so sánh tương đối giữa các phép thử**, không dùng để báo cáo.

---

## Giai đoạn 1 — Ba phép thử chẩn đoán

Mỗi phép thử: resume từ `resume_checkpoints/iov10/ckpt_task00_memory_client09.pth`, chạy **5 round task 1**, cùng seed 42.

**Mốc so sánh (baseline hiện tại, 5 round):** `Old Acc` = 0.04 / 0.02 / 0.04 / 0.03 / 0.04

**Tiêu chí phán quyết:** `Old Acc` vượt **5%** ở round 5 → có tín hiệu, đáng theo đuổi. Dưới 1% → loại.

### Thử A — Loại client không có memory lớp cũ

Thêm cờ `require_old_memory: true`. Ở task > 0, client có `_get_memory()` rỗng thì không tham gia aggregation (α = 0), nhưng **vẫn train cục bộ** để giữ mô hình cá nhân hoá của nó.

Ảnh hưởng: client 5 bị loại khỏi aggregation ở task 1. Trọng số 0.2046 được chia lại cho client 0, 1, 2, 3.

Đây là nghi can tôi đặt cao nhất: nó giải thích được vì sao ρ=0 và ρ≈1 cho kết quả như nhau — vấn đề không nằm ở prototype nào được dùng, mà ở việc một phần mô hình đến từ client chưa từng thấy lớp cũ.

### Thử B — Không calibrate lại classifier khi đánh giá personalized

Thêm cờ `skip_calibrate_on_eval: true`. Bỏ `_calibrate_classifier_from_prototypes` ở khối `do_pc_eval` (`trainer.py` ~712), giữ classifier đã train.

Chỉ ảnh hưởng số liệu personalized, không đổi quá trình huấn luyện. Nếu personalized nhảy từ 31% lên gần 99% thì xác nhận calibrate là chỗ mất mát.

### Thử C — Aggregate cả backbone

Đặt `aggregate_backbone: true` (cờ đã có sẵn trong `is_aggregated_state_key`, không cần sửa code).

Khi đó AFSIC-IoV tổng hợp toàn bộ tham số như FedAvg chuẩn. Nếu kết quả bật lên thì vấn đề nằm ở việc chia sẻ chỉ một phần mô hình.

### Bảng ghi kết quả

| Thử | Cấu hình | Old Acc r5 | New Acc r5 | Acc r5 | Kết luận |
|---|---|---|---|---|---|
| Baseline | — | 0.04 | 58.15 | 0.20 | mốc |
| A | `require_old_memory` | | | | |
| B | `skip_calibrate_on_eval` | | | | |
| C | `aggregate_backbone` | | | | |

Nếu cả ba đều dưới 1%, chạy tiếp **thử D**: đặt `num_clients` hiệu dụng = 1 (chỉ client 0 huấn luyện, không aggregation). Đây là phép thử loại trừ — nếu client 0 đơn độc cũng sụp thì lỗi nằm trong bản thân quá trình học tăng dần, không phải ở FL.

---

## Giai đoạn 2 — Sửa theo kết quả

Chỉ triển khai sau khi Giai đoạn 1 chỉ đúng thủ phạm. Định hướng sẵn:

**Nếu A thắng:** đưa `require_old_memory` thành mặc định cho task > 0. Cân nhắc thêm: cho `α` tỉ lệ với kích thước memory lớp cũ thay vì loại hẳn, để không mất hoàn toàn đóng góp của client đó về lớp mới.

**Nếu B thắng:** chỉ calibrate classifier **một lần** ở đầu task (khi lớp mới xuất hiện và chưa có trọng số), không calibrate lại mỗi round/mỗi lần đánh giá.

**Nếu C thắng:** làm rõ trong bài rằng cơ chế chia sẻ một phần cần điều kiện gì để hoạt động; hoặc đổi sang chia sẻ backbone + giữ riêng adapter (đúng tinh thần FedPer hơn).

---

## Giai đoạn 3 — Sửa khâu đo lường

Độc lập với Giai đoạn 1–2, làm được ngay.

1. **`per_client_eval_every: 1`** — hiện là `0` nên personalized metrics chỉ ghi ở round cuối mỗi task, không có đường cong.

2. **Chia tập test theo client.** Hiện chỉ client 0 có test data (42 triệu mẫu = toàn bộ), 9 client còn lại `test=(0, 31)`. Hệ quả: `_build_local_eval_loader` **fallback sang train data** cho client 1–9, nên `client_accs` dùng để tính điểm chất lượng `Q_i` thực chất là **train accuracy** — sai lệch trực tiếp vào trọng số aggregation.

   Cần phân hoạch 42 triệu mẫu test theo đúng tỉ lệ non-IID của `allocation_plan.csv`. Đây là điều kiện bắt buộc để báo cáo per-client F1 như tài liệu yêu cầu.

3. **Bổ sung cột vào bảng kết quả:** global acc/F1 và personalized acc/F1 đặt cạnh nhau, đúng yêu cầu "so sánh global model và personalized model".

---

## Giai đoạn 4 — Chạy lại đầy đủ

Sau khi sửa xong, chạy lại từ `ckpt_task00_memory_client09.pth` (task 0 không bị ảnh hưởng bởi bất kỳ thay đổi nào ở trên).

- 4 task × 30 round = 120 round
- Với tối ưu prototype ở Giai đoạn 0: ước tính 6–9 giờ, một session Kaggle
- Không tối ưu: ~38 giờ, 4 session

Sau đó chạy lại kịch bản 10-shot để hai kịch bản đồng nhất về code.

---

## Thứ tự ưu tiên đề nghị

1. Giai đoạn 0 (bắt buộc, mở đường cho mọi thứ sau)
2. Thử A (nghi can mạnh nhất)
3. Thử B, C nếu A không thắng
4. Giai đoạn 3 mục 1 (một dòng config, làm luôn)
5. Giai đoạn 2 theo kết quả
6. Giai đoạn 3 mục 2 (chia test set — tốn công, làm khi đã ổn định)
7. Giai đoạn 4

---

## Nguyên tắc khi chẩn đoán

Rút từ hai lần kết luận sớm trong dự án này (F2SCIL bị đánh giá hỏng sau 7 round rồi hoá ra tốt ở round 60; và một lần suy diễn thứ tự lớp mà không có dữ liệu):

- Mỗi phép thử phải có **tiêu chí phán quyết định trước**, không đọc kết quả rồi mới diễn giải.
- Đổi **một biến mỗi lần**.
- Không kết luận từ dưới 5 round.
- Ghi lại cả phép thử thất bại — việc ρ bị bác bỏ cũng là kết quả có giá trị, nó thu hẹp không gian tìm kiếm.
