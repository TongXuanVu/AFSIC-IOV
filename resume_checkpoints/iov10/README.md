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

| Thư mục | Checkpoint | Tiến độ | macro-F1 task 1 |
|---|---|---|---|
| `10shot/` | `ckpt_round0053_task01_r023_acc1.2.pth` | task 1 round 23/30 | 24.33 |
| `1pct/` | `ckpt_round0044_task01_r014_acc97.5.pth` | task 1 round 14/30 | **50.72** |

Ngưỡng sụp task 1 = 16.61, nên cả hai đều đang học thật. Kịch bản 1% giữ được
accuracy 97.5% trong khi 10-shot mất Benign (accuracy 1.24%).

Cả hai checkpoint đều tạo sau commit `5bcc567` nên **có trường `net`** (74 MB thay vì
72 MB). Khi resume phải thấy dòng `Đã phục hồi trọng số riêng cho 10/10 client`.

## Đã gỡ (2026-08-02)

`ckpt_round0075_task02_r015_acc0.0.pth`, `ckpt_task01_memory_client09.pth`,
`metrics_full_r31-75.csv` — đều sinh ra từ run dính lỗi resume ở trên.
