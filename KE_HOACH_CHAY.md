# AFSIC-IoV — Kế hoạch chạy sau khi sửa code

> Lập 2026-08-06. Trả lời câu hỏi: **chạy lại từ đầu, chạy tiếp, hay chạy tiếp từ
> lúc chưa hỏng?**

---

## 1. Đã sửa gì

Bốn chỗ lệch so với đặc tả `AFSIC-IoV.docx`, mỗi chỗ có cờ riêng, **mặc định giữ
nguyên hành vi cũ** nên checkpoint hiện có không bị ảnh hưởng.

| # | Vấn đề | Cờ | Mặc định |
|---|---|---|---|
| 1 | Trọng số tổng hợp `Δθ` thiếu `log(1+n)` | `beta_n` | `0.0` (như cũ) |
| 2 | `Drift` và `UpdateNorm` tính trùng nhau | `drift_mode` | `"vs_global"` (như cũ) |
| 3 | `L_proto` chuẩn hoá `z` trước MSE | `proto_loss_normalize` | `true` (như cũ) |
| 4 | Gated fusion thiếu `Norm` | `gated_fusion_norm` | `false` (như cũ) |

### Kiểm chứng bằng số

**Sửa 1:** `α = softmax(Q/τ)` và `softmax(log n) = n_k/Σn`, nên `beta_n=1` cùng mọi
beta khác `=0` và `τ=1` cho **đúng FedAvg chuẩn** — sai lệch đo được `1,4×10⁻⁸`.

**Sửa 2:** mô phỏng 5 client, trong đó client 4 lệch hẳn khỏi xu hướng chung:

| Client | UpdateNorm | Drift (vs_global) | Drift (vs_mean) |
|---|---|---|---|
| 0 | 0.9389 | 0.9389 | 0.4977 |
| 3 | 0.9713 | 0.9713 | 0.5123 |
| **4** | 2.7889 | 2.7889 | **1.5144** |

Chế độ cũ cho `Drift == UpdateNorm` y hệt nhau; chế độ mới tách được client bất thường.

---

## 2. Ba lựa chọn chạy

### A. Phép thử nhanh — chạy tiếp từ điểm CHƯA hỏng ⭐ làm trước

Resume từ `ckpt_round0090_task02_r030_acc97.7.pth` (hết task 2, macro-F1 **14,44**,
mô hình còn lành), chạy trọn task 3 — đúng đoạn bản gốc sụp từ 14,44 xuống 0,07.

**Không** resume từ round 126: lúc đó mô hình đã hỏng, sửa trọng số ở vài round cuối
không cứu được, nên phép thử sẽ cho âm tính bất kể giả thuyết đúng hay sai.

Đối chứng không cần chạy — dùng luôn số liệu run gốc (cùng config, cùng checkpoint).

| | |
|---|---|
| Config | `can_iov_fewshot1_fedavgw.json` |
| Checkpoint | `ckpt_round0090_task02_r030_acc97.7.pth` |
| Chi phí | 30 round × ~21 phút ≈ **10,5 giờ** (một session) |
| Tiêu chí | macro-F1 cuối task 3 **> 11** (ngưỡng sụp 9,06 + biên) |
| Đối chứng | bản gốc: **0,07** |

Nếu dưới 1 thì giả thuyết sai — dừng, không chạy phương án B.

### B. Chạy lại toàn bộ từ task 0 — sau khi A thắng

Sửa 1 và 2 đều nằm trong `compute_aggregation_weights`, mà hàm này chạy **từ task 0**.
Nên checkpoint task 0 hiện có (`ckpt_round0030`) được tạo bằng trọng số cũ, không dùng
lại được nếu muốn kết quả nhất quán.

| Kịch bản | Config | Round |
|---|---|---|
| full | `can_iov_full_v2.json` | 150 |
| 1% | `can_iov_fewshot1_v2.json` | 150 |
| 10-shot | `can_iov_10shot_v2.json` | 150 |

Task 0 vẫn dùng chung được: chạy **một lần** với `can_iov_full_v2` rồi hai kịch bản kia
resume từ checkpoint task 0 mới (vì `fewshot_enabled` chỉ tác động từ task 1).

**Chi phí ước tính:**

| Phần | Round | Thời gian |
|---|---|---|
| Task 0 (dùng chung) + herding | 30 | ~6 h |
| Mỗi kịch bản, task 1–4 | 120 | ~40 h |
| **Ba kịch bản** | 390 | **~126 h** |

Với quota 30 giờ GPU/tuần/tài khoản: cần khoảng **4–5 tài khoản-tuần**. Chạy song song
ba tài khoản thì gói trong ~2 tuần; sáu tài khoản thì ~1 tuần.

### C. Chạy tiếp từ checkpoint hiện tại — KHÔNG nên

Ba kịch bản đang ở round 118/126/127, tức đã qua điểm sụp. Chạy tiếp chỉ cho thêm số
liệu của một mô hình đã hỏng. Chỉ làm nếu anh cần điền đủ bảng 150 round cho bản gốc
(để so sánh trước/sau khi sửa).

---

## 3. Thứ tự đề nghị

1. **Chạy A** — một session, biết ngay giả thuyết đúng hay sai.
2. Nếu A thắng: **chạy B**, ba kịch bản song song trên ba tài khoản.
3. Nếu A thua: dừng lại phân tích, không tiêu 126 giờ GPU cho một giả thuyết chưa
   được xác nhận.
4. Sửa 3 và 4 xử lý ở khâu **viết bài**, không cần chạy lại:
   - Sửa 3: chọn giữa "sửa công thức trong bài cho khớp code" (khuyến nghị, vì cosine
     đồng nhất với classifier và FSP loss) hoặc "bật `proto_loss_normalize: false` rồi
     dò lại `λ_proto`".
   - Sửa 4: ghi chú rằng chuẩn hoá được hoãn tới nơi sử dụng. Bật `gated_fusion_norm`
     sẽ đổi biểu diễn đặc trưng nên phải chạy lại từ task 0 — không đáng, vì ảnh hưởng
     thực tế gần bằng không.

---

## 4. Cell chạy

### Phép thử A

```python
!rm -rf /kaggle/working/AFSIC-IOV
!git clone https://github.com/TongXuanVu/AFSIC-IOV.git /kaggle/working/AFSIC-IOV
%cd /kaggle/working/AFSIC-IOV
!ls resume_checkpoints/iov10/*/

!python main.py --config configs/exps/can_iov_fewshot1_fedavgw.json \
    --resume resume_checkpoints/iov10/thunghiem/ckpt_round0090_task02_r030_acc97.7.pth
```

### Phương án B — task 0 (chạy trước, dùng chung)

```python
!python main.py --config configs/exps/can_iov_full_v2.json
```

### Phương án B — ba kịch bản (sau khi có checkpoint task 0 mới)

```python
!python main.py --config configs/exps/can_iov_full_v2.json      --resume <ckpt_task0_moi>
!python main.py --config configs/exps/can_iov_fewshot1_v2.json  --resume <ckpt_task0_moi>
!python main.py --config configs/exps/can_iov_10shot_v2.json    --resume <ckpt_task0_moi>
```

---

## 5. Cần thấy gì trong log

**Mọi lần chạy:**

1. `Auto-detected Test File` trỏ `fcil-iov` — không phải `iot100client`
   (IoV 31 feature, IoT 33)
2. `Exemplar size` bắt đầu bằng `292410 / 293047 / 293393`
3. Khi resume: `Đã phục hồi trọng số riêng cho 10/10 client`

**Riêng bản đã sửa** — dòng `Client N => Q_i: ... | n_i: ...` phải cho thấy:

| Client | n_i | α mong đợi | α bản gốc |
|---|---|---|---|
| 0–2 | ~29,3 triệu | ~0,30 mỗi client | 0,11–0,22 |
| 7 | 4.414 | **0,00004** | 0,143 |

Và `Drift` phải **khác** `UpdateNorm` (bản gốc chúng bằng nhau tới từng chữ số, ví dụ
`Drift: 0.0909 | UpdateNorm: 0.0909`).

---

## 6. Nguyên tắc

- Tiêu chí phán quyết **đặt trước**, không đọc kết quả rồi mới diễn giải.
- Đổi **một biến** mỗi lần — phép thử A chỉ đổi trọng số tổng hợp.
- Resume từ điểm mô hình **còn lành**, không từ điểm đã hỏng.
- Luôn đối chiếu macro-F1 với **ngưỡng sụp**, không nhìn accuracy.
- Không kết luận dưới 5 round.
