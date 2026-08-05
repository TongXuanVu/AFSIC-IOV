# AFSIC-IoV — Kế hoạch sửa code cho khớp thiết kế

> Lập ngày 2026-08-06, sau khi rà toàn bộ code với đặc tả trong `AFSIC-IoV.docx`
> (mục 5.1–5.7 và mục 6).
>
> Nguyên tắc chung: **mọi thay đổi đều có cờ config, mặc định giữ nguyên hành vi cũ.**
> Nhờ vậy các run đang chạy và checkpoint hiện có không bị ảnh hưởng, và có thể bật/tắt
> từng thay đổi để đo riêng đóng góp của nó.

---

## 0. Tóm tắt: cái gì đúng, cái gì lệch

Phần lớn code **khớp thiết kế**:

| Mục | Đặc tả | Cài đặt |
|---|---|---|
| 5.2 Personalized adapter | adapter riêng mỗi client | `plasticity_adapter`, không aggregate ✓ |
| 5.4 Prototype classifier | `P(c\|x) ∝ exp(γ·cos(z, p̃_c))` | `CosineLinear` → `σ·cos(z,w)` ✓ |
| 5.6 Local exemplar memory | replay lớp cũ | herding, `memory_ratio` ✓ |
| 5.7 Prototype aggregation | `r_ic = β₁log(1+n) − β₂σ + β₃q − β₄Drift − β₅Upd` | khớp từng số hạng ✓ |
| 6. `L_KD` | `T²·KL(σ(o^{t-1}/T) ‖ σ(o^t/T))` | ✓ |
| 6. `L_FSP` | `log(1+exp((s⁻−s⁺)/T))` | ✓ |
| 6. `L_RS` | `‖A‖₁` | L1 trên adapter + gate ✓ |
| 6. `L_prox` | `‖θ^t − θ^{t-1}‖²` | ✓ |

Mục 5.5 (unknown buffer + energy score) **chưa cài** — nhưng đây là chủ ý, tài liệu mục 7
ghi rõ "chưa cần cài unknown detection ở bản đầu", và config có `use_unknown_discovery: false`.

Bốn chỗ lệch, xếp theo mức ảnh hưởng tới kết quả.

---

## Ưu tiên 1 — Trọng số tổng hợp `Δθ` thiếu thành phần quy mô dữ liệu

### Vấn đề

Mục 5.7 chỉ định nghĩa `α_{i,c}` (per-class) dùng cho prototype. Tài liệu **không nói**
trộn `Δθ` bằng trọng số nào. Code tự thêm trong `utils/aggregation.py`:

```python
Q_i = β_acc·Acc_i + β_proto·ProtoCons_i + β_nov·Novelty_i − β_drift·Drift_i − β_upd·UpdateNorm_i
α_i = softmax(Q_i / τ_agg)
```

Công thức này **không có `log(1+n_i)`**, trong khi `r_{i,c}` của thiết kế thì có.

### Hậu quả đo được (IoV 10 client, task 3)

| Client | mẫu train | % dữ liệu | α | lệch |
|---|---|---|---|---|
| 0 | 29.304.512 | 29,87% | 21,9% | 0,7× |
| 1 | 29.392.930 | 29,96% | 10,8% | 0,4× |
| 2 | 29.423.906 | 29,99% | 10,9% | 0,4× |
| 5 | 87.690 | 0,09% | 10,5% | **117×** |
| 6 | 7.208 | 0,01% | 10,1% | **1.375×** |
| 7 | 4.414 | 0,004% | 14,3% | **3.190×** |
| 9 | 31.400 | 0,03% | 22,6% | **707×** |

Ba client nắm 89,8% dữ liệu chỉ được 43,6% trọng số. Client có 4.414 mẫu chiếm 14,3%
mô hình toàn cục.

**Vì sao IoT không bị:** ở IoT 100 client, top 3 client chỉ chiếm **9,1%** dữ liệu
(trung vị 320.880 mẫu). Chia α gần đều cho 100 client thì mỗi client 1%, khớp tỉ trọng
thật. Ở IoV, top 3 chiếm **89,8%** nên chia đều là bóp méo nghiêm trọng.

### Cách sửa (đã cài, commit `1225e38`)

Thêm `beta_n` vào `Q_i`, mặc định `0.0`:

```python
n_i = sum(int(info.get("count", 0)) for info in client_protos[c].values())
Q_i = beta_n*log1p(n_i) + beta_acc*acc_i + ... 
```

Vì `α = softmax(Q/τ)` và `softmax(log n) = n_k/Σn`, đặt `beta_n=1` cùng mọi beta khác
bằng 0 và `τ=1` sẽ cho **đúng FedAvg chuẩn**. Đã kiểm chứng bằng số: sai lệch 1,4×10⁻⁸.

### Cách kiểm chứng

Resume từ `ckpt_round0090_task02_r030_acc97.7.pth` — **hết task 2, macro-F1 14,44, mô hình
còn lành** — rồi chạy trọn task 3, tức đúng đoạn mà bản gốc sụp từ 14,44 xuống 0,07.

> Không resume từ round 126: lúc đó mô hình đã hỏng, sửa trọng số ở vài round cuối
> không thể cứu được, nên phép thử sẽ cho âm tính bất kể giả thuyết đúng hay sai.

Đối chứng không cần chạy — dùng luôn số liệu run gốc (cùng config, cùng checkpoint,
`beta_n` mặc định 0 nên hành vi không đổi).

**Tiêu chí phán quyết đặt trước:** macro-F1 cuối task 3 phải vượt **11** (ngưỡng sụp
9,06 cộng biên). Dưới 1 thì giả thuyết sai, chuyển sang Ưu tiên 2.

Chi phí: 30 round × ~21 phút ≈ 10,5 giờ, vừa một session Kaggle.

---

## Ưu tiên 2 — `Drift` và `UpdateNorm` bị tính trùng nhau

### Vấn đề

Trong `compute_aggregation_weights`:

```python
for k in local_dict.keys():
    if is_aggregated_state_key(k, task, ...):
        diff = local_dict[k].float() - global_state_round_start[k].float()
        drift_val  += torch.sum(diff ** 2).item()
        update_val += torch.sum(diff ** 2).item()      # ← y hệt dòng trên
        num_params += diff.numel()

drift_i       = np.sqrt(drift_val  / max(1, num_params))
update_norm_i = np.sqrt(update_val / max(1, num_params))
```

`drift_i == update_norm_i` **luôn luôn**. Thiết kế trình bày `β₄·Drift` và `β₅·UpdateNorm`
là hai tiêu chí độc lập; thực tế chỉ là một đại lượng bị trừ hai lần với hệ số `β₄+β₅`.

Điều này xuất hiện ở **cả hai** công thức — `Q_i` và `r_{i,c}`.

### Cách sửa

Phân biệt đúng ngữ nghĩa hai đại lượng:

- **UpdateNorm** — độ lớn update của chính client trong vòng này:
  `‖θ_i^t − θ^{t-1}‖` (so với global **đầu round**). Đây chính là công thức hiện tại.
- **Drift** — client lệch bao xa khỏi *xu hướng chung*: `‖θ_i^t − mean_j(θ_j^t)‖`,
  tức so với trung bình update của các client khác trong cùng round.

Cần một vòng lặp hai lượt: lượt một tính trung bình update, lượt hai tính khoảng cách
tới trung bình đó. Thêm cờ `drift_mode: "vs_global" | "vs_mean"`, mặc định `"vs_global"`
để giữ hành vi cũ.

### Kiểm chứng

So `drift_i` và `update_norm_i` trong log: hiện chúng bằng nhau tới từng chữ số
(ví dụ `Drift: 0.0909 | UpdateNorm: 0.0909`). Sau khi sửa phải khác nhau.

Ảnh hưởng tới kết quả có thể nhỏ vì `β_drift = β_upd = 0.01`, nhưng đây là lỗi **phải
sửa trước khi viết bài** — không thể mô tả hai tiêu chí trong khi cài đặt chỉ có một.

---

## Ưu tiên 3 — `L_proto` lệch thang giá trị

### Vấn đề

Thiết kế mục 6: `L_proto = Σ ‖z_i(x) − p̃_{i,y}‖²` — trên `z` **gốc**.

Code (`losses/proto_loss.py`):

```python
z_norm = F.normalize(features, p=2, dim=1)
loss_proto = F.mse_loss(z_norm, proto_matrix[targets], reduction="mean")
```

Chuẩn hoá `z` trước, nên thực chất tính `2(1 − cos(z, p̃))` chứ không phải khoảng cách
Euclid bình phương. Hướng tối ưu tương đương, nhưng **thang giá trị khác hẳn** — nghĩa là
`λ_proto = 0.5` trong config mang ý nghĩa khác với `λ_proto` trong công thức của bài.

### Cách sửa

Chọn một trong hai, rồi **viết đúng cái đã chọn vào bài**:

- Giữ code, sửa công thức trong tài liệu thành `L_proto = 1 − cos(z, p̃)` (hoặc dạng
  MSE trên vector đã chuẩn hoá).
- Hoặc sửa code bỏ `F.normalize`, và dò lại `λ_proto`.

Tôi nghiêng về cách đầu: cosine hợp với phần còn lại của phương pháp hơn (classifier và
FSP loss đều dùng cosine), và không phải dò lại siêu tham số.

---

## Ưu tiên 4 — Gated fusion thiếu `Norm`

### Vấn đề

Thiết kế mục 5.3: `z = Norm(g ⊙ h_s + (1−g) ⊙ h_a)`

Code (`utils/inc_net.py`):

```python
g = self.gate(phi_x, a_x)
z = g * phi_x + (1.0 - g) * a_x
return z                                # không Norm
```

### Đánh giá

Ảnh hưởng thực tế **nhỏ**, vì mọi nơi dùng `z` đều tự chuẩn hoá:

- `CosineLinear.forward` → `F.normalize(input, p=2, dim=1)`
- `compute_fsp_loss` → `F.normalize(features, ...)`
- `compute_proto_loss` → `F.normalize(features, ...)`
- `compute_local_prototypes` → `F.normalize(feats, p=2, dim=1)`

Nên về mặt kết quả gần như tương đương. Nhưng nếu bài viết ghi `Norm` trong công thức
5.3 thì nên hoặc thêm `F.normalize` vào `extract_vector`, hoặc ghi chú rằng chuẩn hoá
được hoãn tới nơi sử dụng.

**Cảnh báo:** thêm `Norm` vào `extract_vector` sẽ đổi hành vi của **mọi** checkpoint hiện
có. Nếu sửa thì phải chạy lại từ task 0.

---

## Thứ tự thực hiện đề nghị

1. **Chạy phép thử Ưu tiên 1** (đã sẵn code + config, chỉ cần checkpoint r90).
   Đây là thứ duy nhất có khả năng thay đổi kết quả đáng kể.
2. Nếu thắng: đặt `beta_n` thành mặc định, chạy lại cả ba kịch bản từ task 0.
3. Nếu thua: chuyển sang Ưu tiên 2, và xem lại các nghi can khác.
4. **Ưu tiên 2** sửa dù kết quả phép thử thế nào — đây là lỗi cài đặt rõ ràng.
5. **Ưu tiên 3 và 4** xử lý ở khâu viết bài, không cần chạy lại.

---

## Nguyên tắc khi chạy phép thử

Rút từ những lần kết luận sai trong dự án này:

- Tiêu chí phán quyết **đặt trước**, không đọc kết quả rồi mới diễn giải.
- Đổi **một biến** mỗi lần.
- Resume từ điểm mô hình **còn lành**, không từ điểm đã hỏng.
- Luôn đối chiếu macro-F1 với **ngưỡng sụp**, không nhìn accuracy.
- Không kết luận dưới 5 round.
