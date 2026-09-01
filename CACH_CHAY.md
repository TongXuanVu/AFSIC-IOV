# AFSIC-IoV — trạng thái thật sau lượt chính thức 01-09-2026

> **Bản này thay thế toàn bộ bản trước.** Bản trước chấm điểm bằng `rec_macro`.
> Thước đo đó **ngược** — nó cho điểm cao cho đúng trạng thái hỏng nhất. Mọi
> kết luận rút ra từ nó (kể cả "công thức thắng" 42,04 / 50,96) đã được chấm
> lại và một phần bị lật.
>
> Mọi số `[ĐO]`, trích từ `logs/.../debug/{1..12}` và
> `logs/.../100clientiov/full/01-09-26_03-37_seed42_cnn1d_clients100`.

---

## 1. Cách chấm điểm — đọc mục này trước tiên

**Chấm bằng `f1_macro` VÀ `acc` cùng lúc. KHÔNG chấm bằng `rec_macro`.**

Bằng chứng, lấy thẳng từ CSV của lượt chính thức, task 0:

| round | acc | rec_mac | prec_mac | **f1_mac** | thực chất |
|---|---|---|---|---|---|
| 4 | 99,62 | 33,33 | 33,21 | 33,27 | đoán hết Benign |
| 16 | 99,62 | 33,33 | 33,21 | 33,27 | đoán hết Benign |
| 11 | 97,06 | 40,43 | 50,01 | **43,66** | **học thật** |
| 22 | 0,84 | **33,50** | 33,45 | **0,56** | đoán hết một lớp tấn công |
| 30 | 0,52 | **33,50** | 33,34 | **0,35** | đoán hết một lớp tấn công |

Quy tắc cũ (`N_thoat` = số round có `rec_macro > 33,40`) **đánh dấu round 22–30
là "đã thoát"** trong khi f1_macro của chúng là 0,35–0,56, còn round 4 và 16
thì bị chấm "thất bại". `rec_macro` không phân biệt được "học được tấn công"
với "đoán một lớp tấn công cho tất cả", vì cả hai đều cho recall cao ở lớp
được đoán và 0 ở lớp khác.

### Ba mốc để đọc một con số

Với K lớp đã học, tập test có 99,62 % Benign:

```
f1_macro ≈ 99,7/K   ->  đoán hết Benign        (K=3: 33,2 · 6: 16,6 · 9: 11,1 · 11: 9,1 · 13: 7,7)
f1_macro ≈ 0        ->  đoán hết MỘT lớp tấn công
f1_macro > 99,7/K   VÀ acc > 95 %  ->  học thật
```

`acc` một mình cũng mù theo chiều ngược lại: đoán hết Benign cho `acc` 99,62 %.
Phải nhìn cặp.

### Chỉ số nào so sánh được với chỉ số nào

`f1_macro` và `acc` **KHÔNG bất biến** với `eval_max_per_class`. Arm chạy ở chế
độ sàng lọc (`eval_max_per_class: 200000`) chỉ so được **với nhau**, không so
được với HFIN/SPCIL hay với lượt nguyên bản. Số đưa vào luận văn **bắt buộc**
lấy từ lượt có `eval_max_per_class: null`.

`rec_macro` thì bất biến — đã chứng minh bằng đo: run 12 (nguyên bản) và
`T1_plastic_cap64` (sàng lọc) trùng nhau tới 2 chữ số suốt 8 round đầu
(20,70 / 20,68 / 23,29 / 22,85 …). Nhưng bất biến mà mù thì vô dụng.

### Chỉ báo sớm đọc được trong lúc chạy

`[DIAG] classifier cos(w_i,w_j)`:

```
0,38 – 0,62   vùng khoẻ
âm (tới −0,97) đã lật sang "tất cả là tấn công"
→ 0,99        classifier sập, mọi lớp cùng một vector
```

---

## 2. Kết quả thật — cùng tập test 41,8 triệu mẫu, không cắt

`f1_macro / acc`. HFIN và SPCIL-FL: round cuối **bằng** round tốt nhất (đường
cong ổn định). AFSIC: hai số khác nhau rất xa — đó chính là vấn đề, xem mục 4.

| | task 0 | task 1 | task 2 | task 3 | task 4 |
|---|---|---|---|---|---|
| **HFIN** | 55,38 / 99,80 | 58,25 / 99,78 | 68,68 / 99,85 | 70,71 / 99,83 | 54,34 / 99,73 |
| **SPCIL-FL** | 43,35 / 99,69 | 16,76 / 99,35 | 30,34 / 99,34 | 27,39 / 99,32 | 16,17 / 99,20 |
| **AFSIC** tốt nhất | **43,66 / 97,06** (round 11) | *chưa có* | — | — | — |
| **AFSIC** round 30 | 0,35 / 0,52 | | | | |

⚠ Bảng baseline ở bản trước **trộn hai chỉ số**: HFIN task 0/1 ghi 49,84 và
53,72 (đó là `rec_macro`) còn task 2/3/4 ghi 68,68 / 70,71 / 54,34 (đó là
`f1_macro`). **Không dùng lại bảng đó.**

Đọc bảng này: task 0 của AFSIC ở round 11 là kết quả thật, nhỉnh hơn SPCIL-FL
(43,66 so với 43,35) và kém HFIN rõ (55,38). Đổi lại AFSIC là framework duy
nhất phải trả giá bằng accuracy — 97,06 so với 99,7 của hai bên kia.

**Task 1 trở đi chưa từng có số nguyên bản nào dùng được.** Mọi con số task 1
từng báo (20,74 · 41,02 · 50,96) đều là `rec_macro` trên tập test đã cắt. Lần
đo nguyên bản duy nhất cho f1_macro ≈ 5.

---

## 3. Công thức — đã sửa

Chấm lại 20 arm bằng `f1_macro` (trung bình 10 round cuối, đều ở chế độ sàng
lọc nên so được với nhau).

### Task 0 — kết luận cũ đứng vững

| arm | f1 TB10 | f1 max | số round f1 > 30 |
|---|---|---|---|
| **LR1 (`lr` 1e-4, `milestones` [])** | **29,59** | 40,30 | **14** |
| V2_lr3e5 | 26,58 | 35,15 | 6 |
| V2_lr1e5 | 24,33 | 46,78 | 5 |
| A_goc | 19,89 | 25,07 | 0 |

`V2_lr1e5` có f1 max cao nhất (46,78) nhưng đó là một cái gai ở **round 3**,
không phải mức ổn định. `lr = 1e-4` vẫn là lựa chọn đúng cho task 0.

### Task 1 — kết luận cũ SAI

| arm | `lr` | `adapter_bottleneck` | f1 TB10 | số round f1 > 30 |
|---|---|---|---|---|
| **T1_plastic_lr3e5** | **3e-5** | **16 (mặc định)** | **30,65** | **14** |
| T1_plastic_cap64 | 1e-4 | 64 | 25,65 | 3 |
| T1_best | 3e-5 | 64 | 24,77 | 4 |
| T1_sync_plastic | 1e-4 | 16 | 22,90 | 7 |
| T1_cap128 | 3e-5 | 128 | 22,90 | 0 |

Hai thay đổi từng chốt cho task 1 đều đi sai hướng:

- `adapter_bottleneck: 64` **làm giảm** f1 (ở lr 3e-5: 30,65 → 24,77). Nó chỉ
  nâng `rec_macro`, tức đổi precision lấy recall.
- Giữ `lr 1e-4` ở task 1: 3/30 round trên f1 30, so với 14/30 của lr 3e-5.
  Bản trước viết "hạ lr không ổn định hoá được" — **ngược lại**.

### Chốt lại

```
task 0      lr 1e-4 · milestones [] · personalized_adapter false
task 1..4   lr 3e-5 · milestones [] · personalized_adapter false
            plastic_source_trainable true · adapter_bottleneck MẶC ĐỊNH (bỏ khoá này)
mọi task    max_samples_per_class null · test_max_samples_per_class null
            proto_max_samples/quality_eval_max_samples/eval_max_per_class null
```

Chạy từng task bằng `--resume` nên chỉ cần hai file config, không phải sửa code.

`plastic_source_trainable` và `adapter_bottleneck` chỉ có tác dụng từ task 1
(task 0 chưa tạo adapter). Ở task 0 chỉ `lr` và `milestones` có nghĩa.

---

## 4. Vấn đề trung tâm chưa giải quyết: mô hình có hai trạng thái

Đây mới là khiếm khuyết thật, không phải "điểm thấp".

Lượt chính thức, task 0, `acc` trên tập test đầy đủ:

```
round  1– 2    0,35 %
round  3–20   93 – 99,6 %      <- vùng khoẻ, round 11 là đỉnh (f1 43,66)
round 21      0,35 %           <- LẬT
round 22–30   0,3 – 0,8 %      <- không bao giờ hồi
```

Mô hình dao động giữa đúng hai nghiệm suy biến — "đoán hết Benign" và "đoán
hết một lớp tấn công" — và **round 30 rơi vào đâu là may rủi**. Arm LR1 (bản
sàng lọc) lảo đảo đúng chỗ đó (round 20 acc 13,89, round 21 acc 11,39) nhưng
may mắn hồi lại và dừng lúc đang khoẻ. Vì thế "42,04" của LR1 và "0,52" của
lượt này **không phải do bỏ xấp xỉ** — cùng một mô hình bất ổn, khác chỗ đứng
lúc round 30.

Hệ quả dây chuyền đã đo được ở lượt chính thức:

1. Task 0 kết thúc ở trạng thái hỏng (round 30).
2. Pha memory (1h58m) herding bằng feature của mô hình hỏng → exemplar sai.
3. Task 1 round 1: `cos(w_i,w_j)` mean **0,9944** — sáu vector lớp gần trùng
   nhau, classifier không tách được gì. 14 round tiếp theo f1 ≈ 5.

Ma trận nhầm lẫn task 1 xác nhận: **toàn bộ ~40 triệu mẫu Benign rơi vào một ô
duy nhất, cột lớp 4 (fuzzing)**. Arm `T1_plastic_cap64` (cái ra 50,96) cũng
vậy — 145.000/200.000 mẫu Benign vào đúng cột đó.

Nghi phạm chưa kiểm chứng, xếp theo mức đáng thử:

1. `calibrate_with_prototypes` (mặc định **true**) ghi đè `fc.weight` bằng
   prototype **mỗi round**. Bộ phân loại bị reset mỗi vòng theo feature vừa
   tính → có đường để lật. Arm `abl_LR1_cong_E0` (lr 1e-4 + tắt calibrate) đã
   có sẵn trong repo và **chưa từng chạy** — trước đây chỉ chạy calibrate-off ở
   lr 1e-3.
2. Class-balanced CE ở task ≥1: lớp 300k mẫu được cân ngang lớp 97M mẫu, nên
   nghiệm tối ưu bập bênh giữa "giữ Benign" và "bắt lớp hiếm".
3. Chọn round theo `f1_macro` thay vì lấy round cuối. Rẻ, làm được ngay, nhưng
   là vá chứ không phải chữa — và phải khai báo trong luận văn.

---

## 5. Chi phí thật `[ĐO]` từ lượt chính thức

```
task 0        6h26m   (13,3 phút/round, 30 round)
pha memory    1h58m
=> mỗi task ~6,5h + memory ~2h   =>   cả lượt 5 task ≈ 42 giờ, 4–5 session
```

Con số "34 giờ" ở bản trước là thiếu pha memory.

Checkpoint mỗi round: task 0 ~10 MB/file; từ task 1 ~112 MB/file (chứa
976.717 exemplar replay). Một task ≈ 3,4 GB.

Clone bằng `--depth 1`: 29 MB thay vì 169 MB.

---

## 6. Đã thử và KHÔNG hiệu quả — đừng chạy lại

| can thiệp | kết quả |
|---|---|
| `beta_acc: 0` + `beta_n: 0` | Sụp về nghiệm hằng số |
| Gộp FedAvg theo số mẫu | Sụp hoàn toàn. `alpha` gần đều của `Q_i` là **có ích** |
| `bn_stats_by_count` | Không giúp |
| `aggregate_backbone: true` | **Thí nghiệm rỗng** — task > 0 backbone không được huấn luyện cục bộ |
| Tắt cả 5 `lambda_*` | 19/30 round **giống hệt** tới 4 chữ số — nhóm loss vô can |
| `personalized_adapter: false` một mình | Vô dụng nếu thiếu `plastic_source_trainable` |
| `adapter_bottleneck` 64 và 128 | Đều **kém** mặc định 16 theo f1_macro |
| Vá lỗi `num_batches_tracked` | Lỗi có thật (R² 1,000000 → 0,889) nhưng không nâng điểm |

Ba nhóm đã đo và trượt: **khâu gộp** (`beta_*`, FedAvg, BN), **nhóm loss**
(`lambda_*`), **cấu trúc adapter** (`aggregate_backbone`, `adapter_bottleneck`).

Nút thắt nằm ở **tối ưu hoá và tính ổn định**, không ở kiến trúc hay khâu gộp.

---

## 7. Replay `[ĐO]`

Mở thẳng `ckpt_LR1_task00_FINAL.pth` ra đếm, cuối task 0 bộ nhớ replay có
**976.717 exemplar** nằm ở 50/100 client:

```
Benign 973.025  |  DoS 3.396  |  double 296     ->  3.287 : 1
```

HFIN chia **1 : 1**. (Con số "72.500 : 1" ở bản trước là ước lượng và SAI.)

Nhãn **"replay 1 %"** ở ba framework **không cùng nghĩa**: AFSIC và SPCIL-FL là
1 % mỗi lớp của riêng client; HFIN là `memory_size // total_classes` (chia
đều). Phải nói rõ trong luận văn.

---

## 8. Ghi chú viết luận văn

**AFSIC học được thật ở task 0** — f1_macro 43,66 / acc 97,06, hơn SPCIL-FL
(43,35) trên cùng tập test. Đó là kết quả đưa được vào bảng.

**Nhưng nó không ổn định**, và đây là điều phải viết ra chứ không giấu: kết quả
phụ thuộc vào round dừng. HFIN và SPCIL-FL có round cuối = round tốt nhất;
AFSIC thì round 11 cho 43,66 còn round 30 cho 0,35.

**Task 1–4 chưa có số nguyên bản dùng được.** Không được trích các số
20,74 / 41,02 / 50,96 — chúng là `rec_macro` trên tập test đã cắt.

**Vẫn kém HFIN ở mọi task.** Lý do đã đọc trong code, không phải suy đoán: DER
thêm **một backbone CNN mới mỗi task** — `feat_dim` 64 → 320, tham số 22.369 →
115.041. AFSIC giữ `feat_dim` 64 vĩnh viễn. Đổi lại chi phí truyền thông của
AFSIC **cố định** qua mọi task còn HFIN tăng tuyến tính lên 5× ở task 4 — đó là
đánh đổi thiết kế nên nêu.

Các kết quả âm ở mục 6 nên viết vào. Riêng ablation FedAvg là bằng chứng ủng hộ
`Q_i` — cơ chế trung tâm của AFSIC — nên vừa trung thực vừa có lợi.

Trích cặp `LR1` vs `LR2_khong_giam` khi nói về learning rate (khác đúng một
biến), **đừng** trích `LR1` vs `A_goc` — `A_goc` còn bật `legacy_update_norm`.
