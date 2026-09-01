# AFSIC-IoV — cấu hình chạy hiện tại

> Chốt 01-09-2026, sau 11 session / 20 arm trên bộ IoV 100 client.
> Mọi số trong file này là `[ĐO]`, trích từ
> `logs/afsic-iov_federated/can_iov/debug/{1..11}/`.
>
> **Đọc mục 5 trước khi thử bất cứ ý tưởng nào** — 13 can thiệp đã bị loại bằng
> thực nghiệm, đừng chạy lại.

---

## 1. Công thức thắng

File: `configs/exps/xacnhan_FINAL_5task.json`

Khác **12 khoá** so với `configs/exps/can_iov_full_v2_100client.json`. Chia làm
ba nhóm — chỉ nhóm A ảnh hưởng tới kết quả học.

### A. Bốn khoá tạo ra cải thiện

| khoá | gốc | mới | vì sao |
|---|---|---|---|
| `lr` | 0.001 | **0.0001** | Đã quét đủ 4 mức. `1e-3` phân kỳ; `3e-5` và `1e-5` chỉ thoát ngưỡng 6/30 round. |
| `milestones` | `[10, 20]` | **`[]`** | Bỏ giảm LR. Là **mức** quyết định, không phải lịch — `LR2` giữ 1e-3 mà bỏ mốc giảm vẫn hỏng. |
| `personalized_adapter` | true | **false** | Với `true`, ở task > 0 client **không nhận lại** `adapter`/`gate`/`stability_encoder` từ server. Mô hình toàn cục thành trung bình của 50 adapter chưa từng đồng bộ. |
| `plastic_source_trainable` | *(chưa có)* | **true** | **Đòn bẩy lớn nhất.** Mở đóng băng `frozen_source` để nhánh plasticity học từ **dữ liệu gốc**, thay vì chỉ trộn lại 64 đặc trưng đã đóng băng từ task 0. |
| `adapter_bottleneck` | *(mặc định 16)* | **64** | 16 là giá trị mặc định `max(8, 64//4)`, không ai chọn. 128 thì **quá rộng** (25,94). |

`plastic_source_trainable` và `adapter_bottleneck` **chỉ có tác dụng từ task 1**
(task 0 chưa tạo adapter). Ở task 0 chỉ `lr` và `milestones` có nghĩa.

### B. Bảy khoá cắt chi phí — **KHÔNG dùng cho lượt chính thức**

⚠ **Đính chính.** Tôi từng gọi cả nhóm này là "không đụng dữ liệu". Sai. Chỉ
**dữ liệu huấn luyện** là nguyên vẹn. Ba khoá dưới đây có cắt dữ liệu, chỉ là
cắt ở chỗ khác:

| khoá | cắt cái gì | hậu quả |
|---|---|---|
| `proto_max_samples` | dữ liệu dùng để **tính prototype** | Prototype → `r_ic` → gộp prototype → `fc`. **Đổi quỹ đạo học.** |
| `quality_eval_max_samples` | dữ liệu dùng để **tính `acc_i`** | `acc_i` → `beta_acc` → `Q_i` → `alpha`. **Đổi quỹ đạo học.** |
| `eval_max_per_class` | **tập test** | Chỉ đổi con số báo cáo, không đổi việc học. |

Chỉ hai khoá thực sự vô hại: cache `class_counts` (chính xác tuyệt đối — đại
lượng này không đổi trong một task) và `memory_ckpt_per_client` (đổi cách lưu
checkpoint, không đổi phép tính nào).

⇒ **Lượt chính thức dùng `configs/exps/xacnhan_CHINH_THUC.json`**, ở đó cả ba
khoá trên đều `null`. Xem mục 3.

Bảng dưới là chế độ SÀNG LỌC, chỉ dùng khi so sánh nhiều arm với nhau (mọi arm
cùng thiết lập nên so sánh tương đối vẫn hợp lệ):

| khoá | giá trị | tác dụng |
|---|---|---|
| `proto_max_samples` | 20000 | Prototype tính từ 20k mẫu/lớp thay vì toàn bộ. Sai số ~0,7 %. |
| `quality_eval_max_samples` | 50000 | `acc_i` (đầu vào `beta_acc`) ước lượng từ 50k mẫu. |
| `eval_max_per_class` | 200000 | Tập test đường-cong cắt Benign còn 200k. **Xem mục 4.** |
| `per_client_eval` | false | Bỏ đánh giá 100 model cá nhân hoá mỗi round. |
| `ckpt_every_n_rounds` | 10 | 3 checkpoint/task (~150 MB) thay vì 30. |
| `memory_ckpt_per_client` | false | Pha memory lưu 1 lần thay vì sau mỗi client (O(n²), 2 giờ + 6,6 GB). |
| `skip_memory_phase` | **false** | Phải là `false` cho lượt 5 task — task 1–4 cần replay thật. |

⚠ `proto_max_samples` và `quality_eval_max_samples` **có ảnh hưởng quỹ đạo học**
(chúng đi vào prototype và `Q_i`), không thuần tuý là quan sát. Phải ghi vào
phần thiết lập thí nghiệm của luận văn như một xấp xỉ có chủ ý.

### C. Ràng buộc bất di bất dịch

```jsonc
"max_samples_per_class":      null,   // KHONG gioi han du lieu huan luyen
"test_max_samples_per_class": null
```

Mọi client vẫn train trên **toàn bộ** dữ liệu của mình. Không arm nào từng vi
phạm điều này.

---

## 2. Kết quả đo được

`rec_macro` — bất biến với việc cắt Benign nên so sánh được giữa mọi lượt chạy.

| task | lớp | ngưỡng | AFSIC gốc | **AFSIC mới** | SPCIL-FL | HFIN |
|---|---|---|---|---|---|---|
| 0 | 3 | 33,33 | 33,27 | **42,04** | 39,27 | 49,84 |
| 1 | 6 | 16,67 | 16,61 | **50,96** | 16,74 | 53,72 |
| 2 | 9 | 11,11 | **0,04** | **43,87** \* | 30,34 | 68,68 |
| 3 | 11 | 9,09 | 0,03 | *chưa chạy* | — | 70,71 |
| 4 | 13 | 7,69 | 0,04 | *chưa chạy* | — | 54,34 |

\* đo ở `lr 3e-5` (cấu hình yếu hơn) — lượt full sẽ cho số ở `1e-4`.

**Đường đi của task 1:** 16,61 → 20,74 (`lr`) → 41,02 (`plastic`) → **50,96**
(`bottleneck 64`).

Task 2 từ **0,04** — dưới cả ngưỡng đoán một lớp, tức phân kỳ — lên 43,87 và
ổn định (độ lệch chuẩn 10 round cuối = 3,08).

---

## 3. Cách chạy

### Lượt CHÍNH THỨC — không xấp xỉ gì

```
python main.py --config configs/exps/xacnhan_CHINH_THUC.json
```

`proto_max_samples`, `quality_eval_max_samples`, `eval_max_per_class` đều
`null`. `ckpt_every_n_rounds: 1` — lưu và đánh giá **mỗi round** trên tập test
đầy đủ 41,8 triệu mẫu.

**Chi phí: ~13,7 phút/round → ~34 giờ cho 150 round, cộng ~4 giờ pha memory
≈ 38 giờ.** So với ~20 giờ ở chế độ sàng lọc. Chia 5–6 session.

⚠ `ckpt_every_n_rounds: 1` ở task ≥1 thì mỗi checkpoint chứa cả exemplar memory
của 100 client (~80–200 MB). 30 round/task có thể tới 6 GB — vẫn dưới hạn 20 GB
output của Kaggle **nếu mỗi session chỉ chạy 1–2 task**. Đừng dồn cả 5 task vào
một session.

⚠ Công thức được **tinh chỉnh ở chế độ sàng lọc**. Bỏ hai khoá xấp xỉ đi thì
`Q_i` và prototype được tính chính xác hơn, nên kết quả **có thể khác** con số
42,04 / 50,96 / 43,87. Khác theo hướng nào thì chưa ai đo.

### Lượt sàng lọc (rẻ hơn, để so nhiều arm)

```
python main.py --config configs/exps/xacnhan_FINAL_5task.json
```

Session sau nối bằng `--resume <checkpoint moi nhat>`.

Trên Kaggle: mỗi session dùng **Save Version → Save & Run All (Commit)**, chạy
**một mình**, không nhồi arm khác. Xong thì vào tab Output bấm **New Dataset**
để session sau có checkpoint mà `--resume`.

`--depth 1` khi clone: 29 MB / 2,5 giây thay vì 169 MB / 15 giây (lịch sử repo
còn giữ các `.pth` cũ đã xoá).

Chi phí: ~6,2 phút/round → ~3,1 giờ mỗi task → ~20 giờ cả lượt.

### Lấy số cho luận văn — KHÔNG lấy từ đường cong

```
python main.py --config configs/exps/xacnhan_FINAL_5task.json \
               --mode test --test_checkpoint_dir <thu_muc_run>
```

Nhánh `--mode test` dùng **tập test đầy đủ 41,8 triệu mẫu** và không bao giờ bị
cắt (cố ý viết vậy). Mỗi checkpoint mất 76 giây.

---

## 4. Đọc kết quả

**Nhìn `rec_macro`, KHÔNG nhìn `f1_macro` hay `acc`** trên đường cong từng
round. Vì `eval_max_per_class: 200000` cắt bớt Benign nên tỉ lệ lớp đổi:
`rec_macro` (recall trung bình theo lớp) không đổi, còn `f1_macro` và `acc` thì
đổi và **không so được** với bảng HFIN/SPCIL. Chỉ số từ `--mode test` mới so được.

Ngưỡng đoán-một-lớp theo task: **33,33 · 16,67 · 11,11 · 9,09 · 7,69**.

**Nhìn cả đường cong, đừng nhìn round cuối.** Bài học từ SPCIL-FL: nó ghi 43,35
ở round 30 nhưng nằm ở 33,27 suốt 26/30 round. Dùng hai số:

```
N_thoat = số round có rec_macro > ngưỡng
R_max   = rec_macro cao nhất
```

**Đủ 30 round, không rút ngắn.** HFIN nằm im ở 33,27 suốt 14 round rồi mới
thoát. Chạy 5–10 round thì mọi cấu hình đều ra ngưỡng.

---

## 5. Đã thử và KHÔNG hiệu quả — đừng chạy lại

| can thiệp | kết quả |
|---|---|
| `lr` 3e-5 / 1e-5 ở task 0 | 6/30 round, kết ~33 và ~31 (kém `1e-4`) |
| `lr` 3e-5 ghép với `bottleneck 64` | 42,78 — **kém** `1e-4` (50,96). Hạ lr không ổn định hoá được. |
| `adapter_bottleneck: 128` | 25,94 — quá rộng |
| `beta_acc: 0` + `beta_n: 0` | Sụp về 33,33 — hai số hạng này đang **gánh việc** |
| Gộp FedAvg theo số mẫu | Sụp hoàn toàn 33,33. `alpha` gần đều của `Q_i` là **có ích** |
| `bn_stats_by_count` | Không giúp |
| `calibrate_with_prototypes: false` một mình | Không giúp (33,33) |
| `aggregate_backbone: true` | **Thí nghiệm rỗng** — ở task > 0 backbone không hề được huấn luyện cục bộ nên gộp nó vô nghĩa |
| Tắt cả 5 `lambda_*` | 19/30 round **giống hệt** tới 4 chữ số — nhóm loss vô can |
| `personalized_adapter: false` một mình | 20,69 = mốc cũ, vô dụng nếu thiếu `plastic_source_trainable` |
| `normalize_local_steps` (FedNova) | Chưa từng chạy trên dữ liệu thật |
| Vá lỗi `num_batches_tracked` | Lỗi có thật (R² = 1,000000 → 0,889) nhưng **không nâng điểm** |

Ba nhóm đã đo và trượt, đừng tiêu thêm tài nguyên: **khâu gộp** (`beta_*`,
FedAvg, BN), **nhóm loss** (`lambda_*`), **cấu trúc task ≥1** (`aggregate_backbone`,
`calibrate_with_prototypes`).

---

## 6. Còn bỏ ngỏ

**Task 3 và 4 chưa từng chạy** với công thức mới. AFSIC gốc ở đó là 0,03 và
0,04. Task 2 sống là tín hiệu tốt, không phải bảo đảm. Nếu task 3 hoặc 4 rơi
dưới ngưỡng (9,09 / 7,69) thì dừng và xem lại.

**Chỉ một seed.** Độ lệch chuẩn ở task 1 là 8,00 — con số 50,96 rơi đúng round
cuối; round 29 chỉ có 37,39. Nếu có ngân sách, chạy thêm seed 43 và 44 rồi báo
trung bình ± độ lệch chuẩn.

**`abl_T1_best_replay` chưa chạy.** Replay chia đều 2.000/lớp (chính sách HFIN)
thay vì 1 % mỗi lớp. `[ĐO]` — mở thẳng `ckpt_LR1_task00_FINAL.pth` ra đếm, cuối
task 0 bộ nhớ replay có **976.717 exemplar** nằm ở 50/100 client:
Benign 973.025 | DoS 3.396 | double 296 → **3.287 : 1**. HFIN chia **1 : 1**.
(Con số "72.500 : 1" ở các bản trước là ước lượng và SAI.) Đây là cơ chế cuối của HFIN chưa được thử, và
nó nhắm đúng vào việc quên ở các task sau.

**Vẫn kém HFIN ở mọi task**, rõ nhất ở task 2 (43,87 so với 68,68). Đừng viết
như đã vượt.

**Vì sao HFIN mạnh hơn** (đã đọc code, không phải suy đoán): DER thêm **một
backbone CNN mới mỗi task** — `feat_dim` 64 → 320, tham số 22.369 → 115.041.
AFSIC giữ `feat_dim` 64 vĩnh viễn. Đổi lại, chi phí truyền thông của AFSIC **cố
định** qua mọi task còn HFIN tăng tuyến tính lên 5× ở task 4 — đó là đánh đổi
thiết kế nên nêu trong luận văn.

---

## 7. Ghi chú viết luận văn

Trích cặp `LR1` vs `LR2_khong_giam` khi nói về learning rate (khác đúng một
biến), **đừng** trích `LR1` vs `A_goc` — `A_goc` còn bật `legacy_update_norm`
nên khác hai biến.

Nhãn **"replay 1 %"** ở ba framework **không cùng nghĩa**: AFSIC và SPCIL-FL là
1 % mỗi lớp của riêng client (giữ nguyên độ lệch); HFIN là `memory_size //
total_classes` (chia đều). Phải nói rõ, nếu không reviewer coi là so sánh không
công bằng — theo hướng bất lợi cho chính bạn.

Các kết quả âm ở mục 5 **nên viết vào**, đừng giấu. Riêng ablation FedAvg còn là
bằng chứng ủng hộ `Q_i` — cơ chế trung tâm của AFSIC.
