# Checkpoint task 0, round 11 — mô hình AFSIC-IoV tốt nhất từng đo

Trích từ lượt chính thức `01-09-26_03-37_seed42_cnn1d_clients100`
(config `afsic-iov-full.json` lúc commit `ee24caa`: lr 1e-4, milestones [],
personalized_adapter false, plastic_source_trainable true,
adapter_bottleneck 64, ba khoá xấp xỉ đều null).

`[ĐO]` trên tập test đầy đủ 41,8 triệu mẫu, không cắt:

```
acc 97,06 | f1_macro 43,66 | rec_macro 40,43 | prec_macro 50,01
```

Đối chiếu cùng tập test: SPCIL-FL 43,35/99,69 · HFIN 55,38/99,80 ·
mốc "đoán hết Benign" 33,27/99,62.

**Vì sao là round 11 chứ không phải round 30.** Task 0 của lượt đó lật ở
round 21 (acc 99,13 → 0,35) và không bao giờ hồi; round 30 cho f1_macro 0,35.
Xem `CACH_CHAY.md` mục 4.

## Dùng để làm gì

Xem ma trận nhầm lẫn (~2 phút GPU):

```
python tools/xem_confusion.py --config configs/exps/afsic-iov-task0.json \
       --ckpt resume_checkpoints/task0_r11/ckpt_task00_r011_f1_43.66.pth
```

Chạy tiếp task 1 — đây là checkpoint ROUND, chưa có replay memory. Đặt
`num_rounds: 11` để bỏ qua các round task 0 rồi build memory (~2 giờ):

```
python main.py --config <config co num_rounds 11> \
       --resume resume_checkpoints/task0_r11/ckpt_task00_r011_f1_43.66.pth
```

Nếu `load_state_dict` báo không khớp kiến trúc thì thêm `"adapter_bottleneck": 64`
vào config — lượt đó chạy khi khoá này còn bật.
