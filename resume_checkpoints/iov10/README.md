# Checkpoint resume — AFSIC-IoV 10 client

## ⚠️ CẢNH BÁO: cách resume đúng

**Luôn resume từ `ckpt_round0030_task00_r030_acc100.0.pth`.**

**KHÔNG resume từ `ckpt_task00_memory_client09.pth`** dù nó tiết kiệm ~1 giờ herding.

### Lý do

Khi resume, `trainer.py` chỉ nạp lại **memory** cho client:

```python
local_models[c]._data_memory    = c_state.get('data_memory')
local_models[c]._targets_memory = c_state.get('targets_memory')
local_models[c].local_memory    = c_state['local_memory']
```

Trọng số mạng của client **không** nằm trong checkpoint. Bình thường không sao,
vì vòng lặp build memory cuối task 0 có gọi `_load_global_into_client(..., task=0, ...)`
— và ở task 0 hàm này nạp *toàn bộ* trạng thái global vào mọi client.

Nhưng `ckpt_task00_memory_client09.pth` có `last_client_done=9`, nên
`current_client_start = 10` và **vòng lặp bị bỏ qua hoàn toàn**. Client bước vào
task 1 với `adapter`, `gate`, `stability_encoder` còn nguyên khởi tạo ngẫu nhiên.
Đến task 1, `_load_global_into_client` với `personalized_adapter: true` lại cố ý
**không nạp đè** đúng ba thứ đó.

### Dấu hiệu nhận biết

Loss ở round 1 của task 1:

| Resume từ | Loss round 1 | task 1 macro-F1 (10-shot) |
|---|---|---|
| `ckpt_round0030` | **1.601** | **24.33** |
| `ckpt_task00_memory_client09` | 3.273 | 5.95 |

Thấy loss ~3.3 thay vì ~1.6 ở round 1 task 1 là đã dính lỗi.

## File trong thư mục

### Dùng chung (task 0)

| File | Dùng để |
|---|---|
| `ckpt_round0030_task00_r030_acc100.0.pth` | **điểm khởi đầu** cho mọi kịch bản (hết task 0) |
| `ckpt_task00_memory_client09.pth` | chỉ tham khảo memory đã build; KHÔNG resume từ đây |
| `metrics_r1-30.csv` | số liệu task 0 (acc 99.96 / macro-F1 72.75) |

### Theo kịch bản

| Thư mục | Checkpoint mới nhất | Tiến độ |
|---|---|---|
| `full/` | `ckpt_round0083_task02_r023_acc90.8.pth` | task 2 r23/30 (83/150) |
| `1pct/` | `ckpt_round0101_task03_r011_acc97.0.pth` | task 3 r11/30 (101/150) |
| `10shot/` | `ckpt_round0103_task03_r013_acc17.9.pth` | task 3 r13/30 (103/150) |

Kết quả tới thời điểm này (round mới nhất của mỗi task):

| Task | ACC full / 1% / 10shot | macro-F1 full / 1% / 10shot | ngưỡng sụp F1 |
|---|---|---|---|
| 0 | 99.96 / — / — | **72.75** / — / — | 33.27 |
| 1 | 93.77 / 81.83 / 1.29 | 44.57 / 39.24 / 24.07 | 16.61 |
| 2 | 90.82 / 97.72 / 96.38 | 10.63 / 14.44 / **23.82** | 11.07 |
| 3 | — / 97.01 / 17.85 | — / 9.17 / 4.02 | 9.06 |

**Đọc bằng macro-F1, không phải accuracy.** Tập test lệch 18.500:1 (Benign 99,2%)
nên một mô hình chỉ đoán Benign vẫn đạt accuracy 97–99%. Ví dụ 1% ở task 2 có
accuracy 97,72 nhưng macro-F1 14,44 — chỉ nhỉnh hơn ngưỡng sụp 11,07; còn 10-shot ở
task 1 có accuracy 1,29 nhưng macro-F1 24,07, tức thực sự phân biệt được các lớp.

Mọi checkpoint ở đây tạo sau commit `5bcc567` nên **có trường `net`** (78–85 MB).
Khi resume phải thấy `Đã phục hồi trọng số riêng cho 10/10 client`.

## Đã gỡ (2026-08-02)

`ckpt_round0075_task02_r015_acc0.0.pth`, `ckpt_task01_memory_client09.pth`,
`metrics_full_r31-75.csv` — đều sinh ra từ run dính lỗi resume ở trên.
