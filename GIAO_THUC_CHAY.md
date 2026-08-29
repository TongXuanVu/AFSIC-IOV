# Giao thức chạy — ngân sách 2 tài khoản Kaggle

> Mục tiêu: kết luận được **bước nào gỡ được sụp ở task 0** với chi phí nhỏ nhất.
> Ràng buộc giữ nguyên: **không cắt dữ liệu huấn luyện**
> (`max_samples_per_class: null` ở mọi cấu hình).

---

## 1. Chi phí hiện tại và chỗ cắt được

`[ĐO]` Từ log thật `15-08-26_04-03` (100 client, task 0):

```
13,7 phút/round  ×  30 round  =  6,8 giờ cho MỘT task 0
                 × 150 round  = 34,2 giờ cho một lượt đầy đủ 5 task
```

Với 30 h GPU/tuần mỗi tài khoản, 2 tài khoản = 60 h/tuần ⇒ **8 arm task-0 mỗi
tuần**. Quá ít để sàng lọc.

### Một round đang làm gì

`[SUY]` Ở task 0, mỗi client mỗi round duyệt **bốn lượt** toàn bộ dữ liệu của
mình, mà chỉ **một** lượt là huấn luyện:

| # | việc | có cần mỗi round không |
|---|---|---|
| 1 | đếm `class_counts` cho class-balanced CE (`afsic_ids.py`) | **Không.** Chỉ phụ thuộc dữ liệu ⇒ không đổi trong một task |
| 2 | huấn luyện 1 epoch | Có — đây là thứ duy nhất thực sự cần |
| 3 | `_build_local_eval_loader` → forward toàn bộ để lấy `acc_i` | Chỉ cần một ước lượng |
| 4 | `compute_local_prototypes` lớp mới → forward toàn bộ | Chỉ cần một vector trung bình |

Lượt 1, 3, 4 là **đo**, không phải học. Cắt chúng không đụng tới dữ liệu mà mô
hình được huấn luyện trên.

### Đã sửa

| sửa | loại | rủi ro |
|---|---|---|
| Cache `class_counts` theo `(task, total_classes)` | **Chính xác tuyệt đối** — dữ liệu không đổi trong task | Không |
| `proto_max_samples: 20000` áp cho cả lớp mới (trước chỉ áp lớp cũ) | Lấy mẫu khi đo. Prototype là mean của feature đã chuẩn hoá L2 ⇒ n=20k cho sai số ~0,7 % | Thấp |
| `quality_eval_max_samples: 50000` cho `acc_i` | Lấy mẫu khi đo. Sai số chuẩn < 0,25 % | Thấp |
| `per_client_eval: false` | Bỏ đánh giá 100 model cá nhân hoá mỗi round | Không (không dùng khi sàng lọc) |

Ba bẫy đã tránh trong lúc sửa, ghi lại để không ai vô tình phá:

1. **Không dùng `torch.utils.data.Subset`** — `make_loader` không nhận dạng nó
   nên rơi về `DataLoader` chuẩn, chậm 20–50 lần (`utils/fast_loader.py`), phá
   đúng mục đích tối ưu. Phải lấy mẫu trên `SubDummyDataset` (vốn là view).
2. **Không dùng `get_dataset(ret_data=True)`** để lấy mẫu — nó copy nguyên mảng
   trước khi cắt (lớp Benign 29 triệu mẫu ≈ 1,8 GB mỗi client mỗi round).
3. **`count` trả về từ `compute_local_prototypes` chính là `n_i`** dùng cho
   `size_term` (`beta_n`), `tau_local` (FedNova), `r_ic` (gộp prototype) và
   `n_samples` (gộp BN). Nếu để nó thành số mẫu đã lấy mẫu thì việc tối ưu sẽ
   **ngầm bóp méo đúng những đại lượng đang nghiên cứu**. Đã tách
   `report_full_count`: `True` khi cắt mẫu là để ĐO (`proto_max_samples`),
   `False` khi cắt mẫu là THẬT (`kshot`, `1%` — giữ nguyên hai kịch bản few-shot).

`[GIẢ THUYẾT]` Ước tính sau tối ưu: **~3,5–4 phút/round** ⇒ task 0 còn **~2 giờ**.
Chưa đo trên máy thật — round 1 của lượt đầu tiên sẽ cho biết ngay, và nếu
không đạt thì cần đo lại chỗ nghẽn trước khi chạy tiếp.

---

## 2. Ba quy tắc bắt buộc

**a) Mọi arm dùng CÙNG một chế độ nhanh.** Việc lấy mẫu khi đo có làm lệch kết
quả một chút, nhưng nếu mọi arm lệch giống nhau thì so sánh vẫn hợp lệ. Vì thế
`abl_A_goc` (đối chứng) cũng phải chạy ở chế độ nhanh — **không được** so arm
nhanh với baseline cũ chạy đầy đủ.

**b) Đủ 30 round, không rút ngắn.** HFIN nằm im ở 33,27 suốt 14 round rồi mới
thoát. Chạy 5–10 round thì mọi arm đều ra 33,27.

**c) Chấm điểm bằng cả đường cong, không bằng round cuối.** SPCIL-FL "43,35" là
bài học: nó chỉ chớp lên ở round 24, 26, 30 còn 26/30 round vẫn ở 33,27. Dùng
hai con số:

```
  N_thoat  = số round có rec_macro > 33,40  trong 30 round
  R_max    = rec_macro cao nhất đạt được
```

Đối chiếu: HFIN `N_thoat = 16`, `R_max = 49,84`. SPCIL-FL `N_thoat = 3`,
`R_max = 39,27`. AFSIC gốc `N_thoat = 2` (round 9 và 11), `R_max = 43,93`.

Nhìn `rec_macro` chứ **không** nhìn `f1_macro`: round 9 AFSIC có `rec_mac`
43,93 nhưng `f1_mac` chỉ 33,01, vì mô hình đổi recall của Benign lấy recall của
lớp tấn công và `f1_macro` che mất điều đó trên tập test 99,62 % Benign.

---

## 3. Lịch chạy

### Vòng 1 — sàng lọc (4 arm, ~8 giờ GPU tổng)

| tài khoản | arm | đổi gì |
|---|---|---|
| A, session 1 | `abl_LR1_thap_hang` | `lr=1e-4` cố định |
| A, session 1 | `abl_A_goc` | đối chứng (bắt buộc, cùng chế độ nhanh) |
| B, session 1 | `abl_LR2_khong_giam` | `lr=1e-3` cố định |
| B, session 1 | `abl_E0_tat_calibrate` | không ghi đè classifier bằng prototype |

Mỗi session ~4 giờ, chạy tuần tự 2 arm. **Xong trong một ngày**, tốn ~8/60 giờ
ngân sách tuần.

Round 1 của arm đầu tiên cho biết thời gian thật một round. Nếu vẫn >8 phút thì
**dừng lại báo cho tôi** thay vì chạy tiếp — nghĩa là chỗ nghẽn nằm chỗ khác và
đốt tiếp là lãng phí.

### Vòng 2 — chỉ chạy nếu vòng 1 có arm thắng (2–3 arm)

Tổ hợp arm thắng: `abl_LR1_cong_E0`, `abl_LR3_giam_som`, hoặc
`abl_FULL_cong_don`. ~6 giờ.

### Vòng 3 — xác nhận (1 lượt duy nhất)

Arm thắng chạy **đầy đủ 5 task, 150 round, TẮT chế độ nhanh**
(`proto_max_samples: null`, `quality_eval_max_samples: null`,
`per_client_eval: true`). Đây là con số đưa vào luận văn. Ở tốc độ đã tối ưu
ước tính ~17–20 giờ; chia hai tài khoản bằng `--resume`.

### Nếu vòng 1 không arm nào thoát

Đừng chạy vòng 2. Quay lại với 4 file log — chẩn đoán hiện tại sai và cần đọc
lại từ đầu. Bốn arm thất bại vẫn là thông tin: chúng loại được LR, calibration
và cách gộp ra khỏi danh sách nghi phạm.

---

## 4. Những gì KHÔNG nên tiêu ngân sách vào

- **Chạy đủ 150 round để sàng lọc.** Task 0 quyết định; task 1–4 chỉ có nghĩa
  sau khi task 0 thoát sụp.
- **Nhiều seed ở vòng 1.** Đắt, và `N_thoat` trên một lượt đã bắt được phần lớn
  nhiễu mà round-cuối bỏ sót. Để dành seed cho vòng 3.
- **Kịch bản `1%` và `10-shot`.** Cả ba framework đều hỏng ở đó (SPCIL-FL khớp
  ngưỡng chính xác, HFIN chỉ ~1,9 lần ngưỡng). Sửa task 0 của `full` trước.
- **`abl_FEDAVG`.** Vẫn đáng chạy cho bảng ablation của luận văn, nhưng SPCIL-FL
  gần như đã là thí nghiệm đó rồi và nó ở 33,27 suốt 26/30 round. Xếp sau.
