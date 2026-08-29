# Đối chiếu AFSIC-IoV / HFIN / SPCIL-FL trên **cùng** bộ IoV 100 client

> Viết 29-08-2026, đọc từ log đầy đủ của HFIN (`HFIN/IDPS/logs/100clientiov/`)
> và SPCIL-FL (`SPCIL-FL/logs/100clientiov/`), cả 3 kịch bản.
> Mọi con số là `[ĐO]` — trích trực tiếp từ CSV, không suy luận.
>
> **Tài liệu này đính chính hai kết luận trong `HUONG_SUA_V2.md` và làm mất
> hiệu lực bộ config ablation 5 round.**

---

## 1. Bảng tổng hợp — cuối mỗi task

Ngưỡng = giá trị `f1_macro` của bộ đoán đúng một lớp.

### Kịch bản `full`

| task | lớp | ngưỡng | HFIN | SPCIL-FL | AFSIC-IoV |
|---|---|---|---|---|---|
| 0 | 3 | 33,27 | **55,35** | 43,35 * | **33,27** |
| 1 | 6 | 16,61 | **58,25** | 16,76 | 16,61 |
| 2 | 9 | 11,07 | **68,68** | 30,34 | 0,04 |
| 3 | 11 | 9,06 | **70,71** | 27,39 | 0,03 |
| 4 | 13 | 7,66 | **54,34** | 16,17 | 0,04 |

\* xem mục 2 — con số này gây hiểu nhầm.

### Kịch bản `1%` và `10-shot`

| task | lớp | ngưỡng | HFIN 1% | SPCIL 1% | HFIN 10sh | SPCIL 10sh | AFSIC |
|---|---|---|---|---|---|---|---|
| 1 | 6 | 16,61 | 31,83 | **16,61** | 32,02 | **16,61** | **16,61** |
| 2 | 9 | 11,07 | 21,13 | **11,07** | 21,14 | **11,07** | **11,07** |
| 3 | 11 | 9,06 | 17,25 | — | 17,27 | — | **9,06** |
| 4 | 13 | 7,66 | 14,58 | — | 14,61 | — | **7,66** |

`[ĐO]` **SPCIL-FL trong few-shot khớp ngưỡng tới từng chữ số thập phân** —
đúng hiện tượng mà `TINH_HINH_VA_HUONG_SUA.md` mục 1 mô tả cho AFSIC. HFIN
tuy trên ngưỡng nhưng chỉ khoảng **1,9 lần** ngưỡng ở mọi task.

⇒ Trong hai kịch bản few-shot, **cả ba framework đều hỏng**, chỉ khác mức độ.
Điều này phải nói rõ trong luận văn: đó không phải điểm yếu riêng của AFSIC.

---

## 2. Phát hiện chính — cả ba đều bắt đầu từ nghiệm hằng số

Đây là thứ mà bảng "cuối task" che mất. Đường cong **task 0, từng round**:

| round | HFIN | SPCIL-FL | AFSIC-IoV (100 client) |
|---|---|---|---|
| 1–8 | 33,27 | 33,27 | **0,03–0,35** (dưới ngưỡng = phân kỳ) |
| 9 | 33,27 | 33,27 | 33,01 — `rec_mac` **43,93**, acc 91,84 |
| 10 | 33,27 | 33,27 | 0,35 |
| 11 | 33,27 | 33,27 | 29,13 — `rec_mac` **38,92**, acc 77,31 |
| 12–14 | 33,27 | 33,27 | **33,27** (khoá vĩnh viễn) |
| **15** | **36,44** ← thoát | 33,27 | 33,27 |
| 16–23 | 38,40 → 54,62 | 33,27 | 33,27 |
| 24 | 54,94 | **43,35** ← chớp | 33,27 |
| 25 | 55,04 | 33,27 | 33,27 |
| 26 | 55,15 | **43,35** ← chớp | 33,27 |
| 27–29 | 55,15 → 55,38 | 33,27 | 33,27 |
| 30 | **55,35** | **43,35** ← chớp | 33,27 |

Ba điều đọc được:

**a) Nghiệm hằng số là điểm xuất phát của mọi phương pháp, không phải dấu hiệu
hỏng.** HFIN nằm im ở 33,27 suốt **14 round** rồi mới thoát. Không có gì đặc
biệt sai với AFSIC ở 10 round đầu — mọi framework đều thế.

**b) SPCIL-FL không thực sự học được task 0.** Nó ở 33,27 trong **26/30 round**,
chỉ chớp lên 43,35 ở round 24, 26, 30. Bảng mục 1 ghi 43,35 chỉ vì round cuối
tình cờ là một round chớp. Đây là dao động, không phải hội tụ.

**c) AFSIC là trường hợp NGƯỢC lại, không phải tệ hơn.** Nó **đã thoát** khỏi
nghiệm hằng số ở round 9 (`rec_mac` 43,93 — gần bằng 49,80 của HFIN lúc kết
thúc!) và round 11 (38,92). Nó không thiếu năng lực. Nó không giữ được.

⇒ Vấn đề của AFSIC ở task 0 là **bất ổn định, không phải bất lực**.

---

## 3. Khác biệt cấu hình quan trọng nhất: lịch learning rate

`[ĐO]` Đọc từ code của cả ba:

| | optimizer | learning rate qua 30 round |
|---|---|---|
| HFIN | SGD (`edge_server.py:259`) | **1e-3 cố định** |
| SPCIL-FL | Adam (`models/der.py:106`) | **1e-3 cố định** — `milestones` mặc định `[80,120,150]`, mà `epochs=1`/round nên `MultiStepLR` không bao giờ chạm mốc |
| AFSIC-IoV | Adam | **1e-3 → 1e-4 (round 11) → 1e-5 (round 21)** — `trainer.py:543` tính theo `round_idx` trong task |

**AFSIC là framework duy nhất giảm learning rate theo round.** Và nó giảm 100
lần đúng vào cửa sổ round 15–25, chính là lúc HFIN đang leo ra khỏi nghiệm hằng
số.

### Nhưng bằng chứng ở đây có hai mặt — không được kể một chiều

`[ĐO]` AFSIC-IoV **10 client**, task 0, cùng lịch LR đó:

| round | 1–10 (lr 1e-3) | **11** (lr 1e-4) | 12–20 | **21** (lr 1e-5) | 22–30 |
|---|---|---|---|---|---|
| acc | 6,6 – 63,8 hỗn loạn | **99,78** | ~99,9 | 99,95 | 99,96 |
| `f1_mac` | 4,20 – 26,60 | **61,81** | 63–65 | **69,65** | 72,75 |

Với 10 client, **việc giảm LR chính là thứ làm nó chạy được**. Pha `lr=1e-3`
là hỗn loạn thuần tuý (acc nhảy 6,6 % ↔ 63,8 %), và mô hình tốt xuất hiện ngay
round đầu tiên sau khi hạ LR.

Với 100 client, cùng cú hạ LR đó cho acc 77,31 ở round 11 rồi **tụt lại** ở
round 12 và khoá luôn.

⇒ Cách đọc nhất quán với cả hai: **pha LR cao của AFSIC không "khám phá" mà
phân kỳ** (100 client: acc 0,03 %, loss leo từ 2,0 lên 6,5 — thấp hơn cả hàm
hằng tối ưu, tức đang đoán một lớp *hiếm*). Việc hạ LR chỉ đóng băng bất kỳ thứ
gì mô hình đang đứng. Với 10 client thứ đó tình cờ tốt; với 100 client thì
không.

Nếu cách đọc này đúng thì lời giải là **đừng bao giờ vào pha phân kỳ** — chạy
LR thấp ngay từ round 1 — chứ không phải chỉnh thời điểm hạ.

---

## 4. Đính chính `HUONG_SUA_V2.md`

| chỗ | nội dung cũ | đính chính |
|---|---|---|
| Thứ tự thí nghiệm, **5 round** | "5 round task 0 là đủ kết luận" | **SAI, và sai nguy hiểm.** HFIN cần 15 round mới rời khỏi 33,27. Chạy 5 round thì mọi cấu hình đều ra 33,27 và sẽ bị kết luận nhầm là "không cái nào ăn". Mọi ablation task 0 phải chạy **đủ 30 round**. Đã sửa toàn bộ config. |
| `abl_FEDAVG` là "thí nghiệm phân định" | kỳ vọng FedAvg-theo-`n` sẽ cứu được task 0 | **Hạ kỳ vọng.** SPCIL-FL *chính là* FedAvg có trọng số theo `n_k`, không prototype, không `Q_i` (`fl/trainer_fl.py:426`) — và nó ở 33,27 trong 26/30 round. Thí nghiệm vẫn đáng chạy để tách biến, nhưng đừng trông đợi nó nhảy lên 55. |
| Xếp lịch LR vào "vấn đề phụ" (mục 7 tài liệu gốc) | "không sai, nhưng nên biết khi đọc đường cong" | **Nâng lên ứng viên số 1.** Đây là khác biệt cấu hình lớn nhất giữa AFSIC và hai framework kia, và bằng chứng 10-client cho thấy nó quyết định. |
| Lỗi bộ đếm `num_batches_tracked` | được trình bày như phát hiện trung tâm | Vẫn là lỗi thật (R² = 1,000000 trên log thật), nhưng **không còn là đòn bẩy chính**. SPCIL-FL xử lý đúng chỗ này (`average_weights` tách riêng key không phải float) mà vẫn sụp ở task 0. Sửa nó để các phép đo đọc được, đừng kỳ vọng nó nâng `f1_macro`. |

Hai kết luận cũ **vẫn đứng vững**: bộ dữ liệu học được (HFIN đạt 55,35), và
`Δθ`/`robust_filter` trong code hiện tại đang đo bộ đếm batch.

---

## 5. Thứ tự thí nghiệm mới — **30 round, task 0, mỗi lần một biến**

| # | config | đổi gì | vì sao ưu tiên thế này |
|---|---|---|---|
| 1 | `abl_LR1_thap_hang` | `lr = 1e-4` cố định, bỏ mốc giảm | Bỏ hẳn pha phân kỳ. Bằng chứng mạnh nhất hiện có: 10 client nhảy lên `f1` 61,81 ngay round đầu tiên ở LR này. |
| 2 | `abl_LR2_khong_giam` | `lr = 1e-3` cố định | Đúng cấu hình HFIN/SPCIL-FL. Kiểm xem AFSIC có leo ra như HFIN ở round 15 không, nếu không bị đóng băng. |
| 3 | `abl_E0_tat_calibrate` | tắt ghi đè classifier bằng prototype | Vẫn là giả thuyết độc lập đáng kiểm; đọc `[DIAG] classifier cos(w_i,w_j)` trong log để biết ngay. |
| 4 | `abl_LR3_giam_som` | mốc `[3, 10]` | Nếu 1 và 2 đều hé mở nhưng chưa đủ. |
| 5 | `abl_LR1_cong_E0` | 1 + 3 + BN theo `n_i` | Chỉ chạy sau khi đã biết bước nào có tác dụng. |
| 6 | `abl_FEDAVG`, `abl_E1`…`abl_E5` | tách từng cơ chế | Cho bảng ablation của luận văn, không phải để tìm lời giải. |

**Đọc kết quả — nhìn `rec_macro`, không nhìn `f1_macro`.** Ở round 9 AFSIC có
`rec_mac` 43,93 nhưng `f1_mac` chỉ 33,01: mô hình đang đánh đổi recall của
Benign lấy recall của lớp tấn công, và `f1_macro` che mất điều đó trên tập test
99,62 % Benign. `rec_macro` > 33,33 nghĩa là đã thoát chế độ hằng số.

**Và nhìn cả đường cong, không nhìn round cuối.** SPCIL-FL "43,35" là bài học:
một round chớp không phải là hội tụ.

---

## 6. Cho luận văn

Bảng mục 1 nên vào luận văn gần như nguyên trạng — nó cho thấy **bộ IoV 100
client là bài toán khó với mọi phương pháp**, không riêng AFSIC:

- `full`: HFIN học được; SPCIL-FL dao động quanh nghiệm hằng số; AFSIC phân kỳ.
- `1%` và `10-shot`: HFIN chỉ đạt ~1,9 lần ngưỡng; SPCIL-FL khớp ngưỡng chính
  xác; AFSIC khớp ngưỡng chính xác.

Đây là bối cảnh trung thực để đặt kết quả của AFSIC vào, thay vì trình bày nó
như một thất bại đơn lẻ. Nhưng cũng đừng dùng nó để bào chữa: AFSIC là bản
**duy nhất phân kỳ xuống dưới ngưỡng** ở `full` task 2–4 (`f1` 0,03–0,04), và
đó là hỏng nặng hơn hai bản kia.
