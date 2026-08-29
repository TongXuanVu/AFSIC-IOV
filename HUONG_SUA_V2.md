# AFSIC-IoV — hướng sửa V2

> Viết 27-08-2026, sau khi đọc `TINH_HINH_VA_HUONG_SUA.md` và đối chiếu lại
> `utils/aggregation.py`, `trainer.py`, `convs/linears.py`, `convs/cnn1d.py`
> với log thật `logs/.../100clientiov/full/15-08-26_04-03_.../training.log`.
>
> Giữ quy ước mức chắc chắn: `[ĐO]` = trích/tính lại từ log thật,
> `[SUY]` = suy luận từ code, `[GIẢ THUYẾT]` = chưa kiểm chứng.

---

## 0. Tóm tắt cho người vội

Tìm được **một lỗi thực thi đo được chắc chắn** mà tài liệu cũ chưa nêu:
`Drift` và `UpdateNorm` — hai đại lượng đang điều khiển cả việc gộp trọng số
lẫn việc gộp prototype — **không đo trọng số. Chúng đo `num_batches_tracked`
của BatchNorm**, tức là đếm số batch cục bộ.

Kéo theo ba hệ quả, trong đó một hệ quả nối thẳng tới hiện tượng sụp ở mục 1
của tài liệu cũ. Vì vậy **thứ tự thử nghiệm ở mục 9 tài liệu cũ nên đổi**:
ba thí nghiệm rẻ nhất bây giờ chỉ cần sửa **config, không sửa code**, và
`normalize_local_steps` nên lùi xuống sau chúng.

---

## 1. Lỗi #1 — `Drift`/`UpdateNorm` đang đo bộ đếm batch, không đo trọng số `[ĐO]`

### Cơ chế

`utils/aggregation.py:85-90` duyệt **mọi key** trong `state_dict` của client:

```python
for k in local_dict.keys():
    if is_aggregated_state_key(k, task, agg_bb):
        d = local_dict[k].float() - global_state_round_start[k].float()
        sq += torch.sum(d ** 2).item()
        n_par += d.numel()
update_norm = sqrt(sq / n_par)
```

`CNN1DFeatureExtractor` có **4 lớp `BatchNorm1d`**, mỗi lớp mang một buffer
`num_batches_tracked` kiểu `int64`. Ở task 0 `is_aggregated_state_key` trả
`True` cho *mọi* key, nên bốn bộ đếm này lọt vào phép tính. Sau một round mỗi
bộ đếm tăng đúng bằng số batch cục bộ `τᵢ`, nên đóng góp vào `sq` là `4·τᵢ²`
— trong khi toàn bộ trọng số thật chỉ đóng góp phần lẻ.

### Bằng chứng số — khớp tới 4 chữ số

`n_par` của state_dict task 0 = 22.369 tham số (đúng con số `Params: 22,369`
trong log) + 384 phần tử `running_mean`/`running_var` + 4 bộ đếm = **22.757**.

Dự đoán `UpdateNormᵢ = sqrt(4·τᵢ² / 22757)` với `τᵢ = ceil(nᵢ/8192)`, đối chiếu
giá trị log ở **round 1 task 0**:

| client | `nᵢ` | `τᵢ` | dự đoán | log thật | tỉ lệ |
|---|---|---|---|---|---|
| 30 | 12.638.022 | 1543 | 20,4568 | **20,4572** | 1,0000 |
| 9 | 11.139.791 | 1360 | 18,0307 | **18,0310** | 1,0000 |
| 10 | 8.242.750 | 1007 | 13,3506 | **13,3510** | 1,0000 |
| 4 | 8.012.344 | 979 | 12,9794 | **12,9798** | 1,0000 |
| 38 | 7.869.131 | 961 | 12,7408 | **12,7412** | 1,0000 |
| 31 | 7.620.607 | 931 | 12,3430 | **12,3435** | 1,0000 |
| 46 | 7.112.780 | 869 | 11,5211 | **11,5214** | 1,0000 |
| 39 | 4.269.318 | 522 | 6,9206 | **6,9212** | 0,9999 |
| 6 | 3.154.644 | 386 | 5,1175 | **5,1183** | 0,9998 |
| 8 | 1.291.433 | 158 | 2,0947 | **2,0969** | 0,9990 |
| 0 | 736.461 | 90 | 1,1932 | **1,1964** | 0,9973 |
| 3 | 448 | 1 | 0,0133 | **0,0159** | 0,834 |

Kiểm chéo bằng `Drift`: 30-odd client nhỏ đều báo **đúng cùng một giá trị
3,1553**. Với `drift_mode: vs_mean`, client nhỏ lệch khỏi trung bình gần đúng
bằng chính trung bình bộ đếm, nên `3,1553 = sqrt(4·τ̄²/22757)` ⇒ `τ̄ = 238,0`
⇒ tổng dữ liệu 50 client active ≈ **97,5 triệu mẫu** — khớp con số ~98M trong
config. Không còn chỗ cho trùng hợp.

⇒ Với mọi client lớn hơn vài batch, **≥ 99,9 % giá trị `UpdateNorm` và
`Drift` là bộ đếm batch**. Chỉ client 448 mẫu (1 batch) mới còn ~17 % là
trọng số thật.

### Hệ quả A — khâu gộp trọng số: nhẹ

`beta_drift = beta_update = 0,01`. Client lớn nhất bị trừ
`0,01·(17,29 + 20,46) ≈ 0,38` khỏi `Q_i`. Có thật nhưng nhỏ, và nó phạt theo
**quy mô dữ liệu**, không theo độ bất thường. Đây là lý do mục 7 tài liệu cũ
nhận xét đúng rằng "drift/update là hai đại lượng duy nhất trong `Q_i` tương
quan với quy mô client" — bây giờ đã biết vì sao: chúng *là* quy mô client.

### Hệ quả B — khâu gộp prototype: nghiêm trọng `[ĐO]`

`trainer.py:203-210` dùng **cùng hai đại lượng đó** với hệ số lớn hơn 50× và
20×:

```
r_ic = 1,0·log(1+n_ic) − 1,0·σ_ic + 1,0·q_i − 0,5·Drift_i − 0,2·UpdateNorm_i
α_ic = softmax_i(r_ic)
```

Thay số round 1, lớp Benign:

| client | `n_ic` | `log(1+n)` | phạt `0,5·Drift + 0,2·Upd` | `r_ic` (≈) |
|---|---|---|---|---|
| 30 (12,6 M mẫu) | 12.638.022 | 16,35 | 0,5·17,29 + 0,2·20,46 = **12,74** | ≈ 3,6 |
| client nhỏ (1.000 mẫu) | 1.000 | 6,91 | 0,5·3,16 + 0,2·0,016 = **1,58** | ≈ 5,3 |

⇒ Client giữ **1.000 mẫu được trọng số prototype cao gấp ~5,5 lần** client
giữ **12,6 triệu mẫu** (`e^{5,3−3,6}`). Và trong 50 client active thì ~30
client thuộc nhóm tí hon. **Prototype toàn cục do nhóm client tí hon quyết
định.**

### Hệ quả C — kết luận cũ về `robust_filter_updates` phải xem lại `[ĐO]`

Mục 5 tài liệu cũ loại phương án bật bộ lọc vì "z-score 556–894 sẽ loại cả 20
client lớn". Nhưng z-score đó tính trên `update_norm_list` — tức là **z-score
của số batch**, không phải của độ lớn update. Kết luận "bộ lọc vứt 99,6 % dữ
liệu" đúng *với code hiện tại*, nhưng nó không nói gì về việc bộ lọc có hoạt
động đúng hay không sau khi sửa. Tương tự, `max_update_norm` hiện là ngưỡng
đặt trên số batch — vô nghĩa.

### Hệ quả D — bộ đếm bị trung bình rồi ép về int `[SUY]`

`trainer.py:702`: `global_dict[k] = val.to(global_dict[k].dtype)` biến trung
bình có trọng số của `num_batches_tracked` thành `int64`. Vô hại về mặt
forward (BatchNorm dùng `momentum=0.1` cố định nên không đọc bộ đếm), nhưng
nó là dấu hiệu rõ rằng buffer đang bị đối xử như tham số.

---

## 2. Lỗi #2 — bộ phân loại toàn cục không phải do huấn luyện `[ĐO/SUY]`

Ba mảnh code ghép lại thành một điều dễ bị bỏ sót:

1. `convs/linears.py:55` — `CosineLinear.forward` là
   `F.linear(normalize(input), normalize(weight))`, **không có bias**.
   Logit = `σ · cos(z, w_c)`.
2. `utils/inc_net.py:187-202` — `init_new_class_weights_from_prototypes` ghi
   `fc.weight[c] = normalize(prototype_c)` cho **mọi** `c` trong
   `range(_total_classes)`, bất kể tên hàm có chữ "new".
3. `trainer.py:704` — `_calibrate_classifier_from_prototypes(global_model)`
   chạy **ngay sau mỗi lần gộp trọng số, mỗi round**.

⇒ Mô hình toàn cục thực chất là **bộ phân loại nearest-prototype theo cosine**.
Mọi thứ `fc` học được trong local training, và mọi thứ khâu gộp trọng số làm
với `fc`, **bị ghi đè mỗi round**. `normalize_local_steps` chỉnh
`α·Δθ` — nhưng phần `Δθ` của `fc` không bao giờ tới được mô hình đánh giá.

Nối với lỗi #1 thì chuỗi nhân quả tới hiện tượng "đoán đúng một lớp" là
thẳng tuột:

```
num_batches_tracked lọt vào Drift/UpdateNorm
   → r_ic sai, ưu ái client tí hon
   → prototype toàn cục sai
   → fc.weight = prototype sai  (ghi đè mỗi round)
   → argmax cos(z, w_c) hằng số
```

**Một điểm nữa `[SUY]`:** prototype `p_ic` được tính bằng **mạng cục bộ đã
train của client i** (`trainer.py:587+`, sau `_train`), rồi đem cắm vào **mạng
đã gộp**. 50 client = 50 không gian đặc trưng khác nhau; trung bình của 50
prototype trong 50 không gian không nằm trong không gian nào cả. Với 10 client
sai lệch này nhỏ hơn nhiều — khớp với việc bộ 10 client học được task 0
(72,75) còn bộ 100 thì không.

**Tin tốt:** đã có sẵn cờ `calibrate_with_prototypes` (`trainer.py:236`, mặc
định `True`). Tắt được bằng **một dòng config, không sửa code**.

---

## 3. Lỗi #3 — thống kê BatchNorm gộp bằng `alpha` gần đều `[SUY]`

`running_mean`/`running_var` là **ước lượng của phân phối dữ liệu**, nhưng
đang được gộp bằng đúng `alpha` dùng cho trọng số — mà `alpha` gần đều
(0,005–0,03 trên 50 client, xem log round 1).

- Client 1 batch trả về `running = 0,9·global_cũ + 0,1·batch_riêng`
  (momentum 0,1) — gần như không cập nhật.
- Client 1.543 batch trả về thống kê hội tụ hẳn về dữ liệu riêng của nó.
- Trung bình gần đều của hai loại đó ⇒ thống kê BN của global **vừa sai phân
  phối, vừa trễ** so với trọng số vốn đã đi rất xa trong cùng round.

`[ĐO]` Khớp với đường cong task 0 của run 15-08:

| round | 1–8 | 9 | 10 | 11 | 12–30 |
|---|---|---|---|---|---|
| `acc` | 0,03–0,35 | **91,84** | 0,35 | 77,31 | 99,62 (đứng yên) |
| `rec_mac` | 33,33 | **43,93** | 33,33 | 38,92 | 33,33 |
| loss | 2,0 → 6,5 | 0,36 | 4,38 | 0,59 | ~0,2 |

Mô hình **có** học được (round 9: `rec_mac` 43,93 — vượt hẳn ngưỡng đoán một
lớp), rồi dao động dữ dội, rồi **đứng yên đúng từ round 12** — tức ngay sau
milestone 10 khi `lr` giảm 10×. Đọc là: khi trọng số ngừng chạy nhanh thì
thống kê BN bắt kịp và hệ ổn định — nhưng điểm nó dừng lại là nghiệm suy
biến. Đây là chữ ký của mất đồng bộ trọng số ↔ thống kê, không phải của "học
kém".

`[ĐO]` HFIN dùng **cùng kiến trúc** (`HFIN/IDPS/models/feature_extractor.py`
cũng là Conv1d → ReLU → BatchNorm1d ×4) nhưng gộp bằng `FedWeightedAvg` theo
số mẫu ⇒ thống kê BN của nó bám phân phối thật. Đây là khác biệt cụ thể thứ
hai giữa HFIN và AFSIC, bên cạnh khác biệt replay đã nêu ở mục 3.5 tài liệu cũ.

---

## 4. Thứ tự thử nghiệm mới

Lý do đổi so với mục 9 tài liệu cũ: `normalize_local_steps` sửa đường
`α·Δθ` trên **trọng số**, nhưng bộ phân loại toàn cục không dùng `fc` đã học
(lỗi #2) và prototype không đi qua đường đó chút nào (lỗi #1B). Nên nó không
thể một mình gỡ được sụp ở task 0. Ba thí nghiệm rẻ hơn nên đi trước.

Vẫn giữ nguyên tắc: **mỗi lần một biến, 30 round task 0 là đủ.**

| # | thay đổi | sửa gì | kỳ vọng / đọc kết quả |
|---|---|---|---|
| **E0** | `"calibrate_with_prototypes": false` | config, 1 dòng | Ablation trực tiếp cho lỗi #2. Nếu `rec_mac` task 0 > 34 ⇒ thủ phạm chính là đường prototype → classifier. **Chạy đầu tiên.** |
| **E1** | `"proto_beta_drift": 0, "proto_beta_update": 0` | config | Gỡ ảnh hưởng bộ đếm khỏi gộp prototype mà chưa cần vá code. So với E0 để biết prototype *sai* hay prototype *về nguyên tắc không dùng được*. |
| **E2** | `"beta_acc": 0, "beta_n": 0` | config | Như bước 2+3 tài liệu cũ. Bỏ phần thưởng cho client sụp và bỏ ưu ái client lớn. |
| **E3** | vá P1 (bỏ buffer khỏi diff) | code | Sau vá, **đo lại** tỉ lệ `α·‖Δθ‖` thật. Con số 295:1 ở mục 3.1 tài liệu cũ phải tính lại — nó đang là tỉ lệ số batch. |
| **E4** | `"normalize_local_steps": true` | đã có config sẵn | Chỉ có nghĩa **sau E3**, vì trước đó không ai biết `‖Δθ‖` thật là bao nhiêu. |
| **E5** | vá P2 (BN gộp theo `n_i`) | code | Kỳ vọng hết dao động round-to-round. |
| **E6** | `"aggregate_backbone": true` + `"fixed_memory": true, "memory_size": 26000` | config | Như bước 4 tài liệu cũ — **chỉ có tác dụng từ task 1**, chỉ chạy sau khi task 0 đã thoát sụp. |

**Bảng đọc `rec_macro` task 0** giữ nguyên như mục 9 tài liệu cũ: đúng 33,33 =
vẫn hằng số; > 34 = đã thoát; ~55 = ngang HFIN; ~70 = ngang bộ 10 client.

Thêm hai thứ nên log để khỏi phải đoán ở lần sau `[GIẢ THUYẾT]`:

- **Ma trận cosine giữa các prototype toàn cục** sau mỗi round. Nếu mọi cặp
  ~0,99 thì classifier không thể tách lớp — đó là bằng chứng trực tiếp,
  không cần chạy lại gì.
- **Histogram lớp được dự đoán** trên tập test toàn cục. Phân biệt ngay
  "đoán lớp đa số" với "đoán lớp hiếm" mà không phải suy từ `f1_macro`.

---

## 5. Ba bản vá đề nghị

Cả ba đều **mặc định tắt** để mọi config cũ giữ nguyên hành vi.

### P1 — bỏ buffer không phải float khỏi phép đo `Δθ` (`utils/aggregation.py:85`)

```python
for k in local_dict.keys():
    if not is_aggregated_state_key(k, task, agg_bb):
        continue
    t = local_dict[k]
    if not t.is_floating_point():          # num_batches_tracked
        continue
    if args.get("exclude_bn_stats_from_norm", True) and \
       ("running_mean" in k or "running_var" in k):
        continue                            # thống kê, không phải tham số
    d = t.float() - global_state_round_start[k].float()
    ...
```

Rủi ro: thấp. Đổi thang đo của `Drift`/`UpdateNorm` nên `beta_drift`,
`beta_update`, `proto_beta_drift`, `proto_beta_update`, `robust_z`,
`max_update_norm` đều phải hiệu chuẩn lại — chạy 1 round rồi đọc log trước
khi đặt số.

### P2 — gộp buffer BN theo số mẫu (`trainer.py:687-702`)

Trong vòng `for k in global_dict.keys()`, tách nhánh:

```python
is_bn_stat = ("running_mean" in k) or ("running_var" in k)
w = n_weights if (is_bn_stat and args.get("bn_stats_by_count", False)) else alpha
```

với `n_weights[i] = n_i / Σ n_j` trên các client được nhận. `num_batches_tracked`
thì lấy `max` thay vì trung bình.

Lý lẽ viết được vào luận văn: **thống kê BN là ước lượng của phân phối dữ
liệu ⇒ ước lượng không chệch cần cân theo số mẫu; trọng số là biến tối ưu ⇒
cân theo điểm chất lượng.** Đây là lập luận độc lập, không phải "bắt chước
HFIN", và có tham chiếu được (FedBN, Li et al., ICLR 2021, cho biến thể mạnh
hơn là không gộp BN chút nào).

### P3 — calibrate ít lại (`trainer.py:233-247`) `[GIẢ THUYẾT]`

Hai biến thể đáng thử, chọn một:

- `calibrate_new_classes_only: true` — chỉ ghi đè `fc.weight` cho các lớp
  thuộc task hiện tại, giữ nguyên phần đã học của lớp cũ.
- `calibrate_once_per_task: true` — chỉ calibrate ở round đầu mỗi task
  (đúng vai trò "khởi tạo"), các round sau để `fc` học và được gộp bình
  thường.

E0 (`calibrate_with_prototypes: false`) là ablation cực đoan của cả hai; nếu
E0 ăn thì P3 là cách giữ lại cơ chế prototype mà không để nó nuốt classifier.

---

## 6. Cần đính chính trong `TINH_HINH_VA_HUONG_SUA.md`

| chỗ | nội dung cũ | vì sao phải sửa |
|---|---|---|
| mục 3.1 | "lực kéo thực tế `alpha × ‖Δθ‖` … **295 : 1**" | `‖Δθ‖` lấy từ `UpdateNorm` đã log = số batch. Con số 295:1 thực chất là tỉ lệ `α·τ`, không phải `α·‖Δθ‖`. Phải đo lại sau P1. |
| mục 3.1 | "kéo theo `‖Δθ‖` chênh 1.295 lần" (comment trong `aggregation.py`) | 1.295 ≈ 1.543 chính là tỉ lệ `τ`. Đây là cùng một đại lượng được trình bày như hai bằng chứng độc lập. |
| mục 5, dòng `robust_filter_updates` | "z-score 556–894 sẽ loại cả 20 client lớn" | z-score của bộ đếm batch. Đúng với code hiện tại, nhưng không kết luận được gì về bộ lọc sau khi vá. |
| mục 7 | "drift/update là hai đại lượng duy nhất trong `Q_i` tương quan với quy mô client" | Nhận xét đúng, nhưng nguyên nhân không phải "hiệu chỉnh chưa hợp lý" — chúng *bằng* quy mô client. |
| mục 9, thứ tự | `normalize_local_steps` ở bước 1 | Không chạm được tới `fc` (bị ghi đè) lẫn prototype. Lùi xuống sau E0–E3. |

Mục 2 (đính chính "dữ liệu quá khó") và mục 10 (cảnh báo hai codebase) vẫn
đứng vững, không cần đổi.

---

## 7. Lưu ý khi viết luận văn

- **Lỗi #1 là bug thực thi, không phải đóng góp.** Sửa nó thì viết trong phần
  thiết lập thí nghiệm, đừng viết thành "cải tiến phương pháp". Nếu số liệu cũ
  đã đưa vào bản thảo thì phải chạy lại — mọi con số `Drift`/`UpdateNorm` đã
  công bố đều là số batch.
- **Lỗi #2 và #3 là lựa chọn thiết kế**, nên chúng viết được thành ablation
  hợp lệ: "ảnh hưởng của prototype-calibration tới bộ phân loại toàn cục" và
  "cách gộp thống kê BatchNorm trong FL không-IID". Cả hai đều có tham chiếu
  (FedBN cho #3).
- Cảnh báo cuối tài liệu cũ vẫn nguyên giá trị và giờ áp cho cả V2:
  **tính tới 27-08-2026 chưa có bản vá nào chạy trên CAN-bus thật.** Toàn bộ
  mục 1–3 ở đây là đọc code + tính lại từ log, không phải kết quả thực nghiệm.
