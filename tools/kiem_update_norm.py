"""Kiểm chứng: Drift/UpdateNorm có còn bám bộ đếm batch không?

Cách dùng:
    python kiem_update_norm.py <duong_dan_training.log>

Nguyên lý. Nếu `num_batches_tracked` của các lớp BatchNorm lọt vào phép tính
`Δθ` thì với mọi client có nhiều hơn vài batch:

    UpdateNorm_i  ==  sqrt(n_bn * tau_i^2 / n_par)         (tau_i = ceil(n_i/batch_size))

tức UpdateNorm tỉ lệ THẲNG với tau_i và tỉ số đo/dự đoán bằng 1,0000.
Sau khi vá, quan hệ đó phải BIẾN MẤT.

Script đọc n_i và UpdateNorm từ log, tự dò n_par và số buffer BN bằng cách khớp
bình phương tối thiểu trên các client lớn, rồi báo hệ số tương quan.

ĐỌC KẾT QUẢ
    R^2 > 0,99  và tỉ số đo/dự đoán ~1,000  ->  VẪN CÒN BUG
    R^2 sụp, tỉ số tản mát                  ->  ĐÃ HẾT (UpdateNorm đo trọng số thật)
"""
import math
import re
import sys

PAT = re.compile(
    r"Client\s+(\d+)\s*=>\s*Q_i:\s*([-\d.]+)\s*\|\s*n_i:\s*(\d+).*?"
    r"Drift:\s*([\d.]+)\s*\|\s*UpdateNorm:\s*([\d.]+)"
)


def doc_round1(path):
    """Lấy các dòng Q_i của ROUND ĐẦU TIÊN trong log."""
    rows, seen = [], set()
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = PAT.search(line)
            if not m:
                continue
            cid = int(m.group(1))
            if cid in seen:          # đã sang round 2 -> dừng
                break
            seen.add(cid)
            rows.append(dict(cid=cid, n=int(m.group(3)),
                             drift=float(m.group(4)), upd=float(m.group(5))))
    return rows


def doc_batch_size(path):
    m = re.search(r"batch_size[\"']?\s*[:=]\s*(\d+)", open(
        path, encoding="utf-8", errors="replace").read()[:200000])
    return int(m.group(1)) if m else None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else doc_batch_size(path)
    if not bs:
        print("Khong doc duoc batch_size trong log; truyen tay: "
              "python kiem_update_norm.py <log> <batch_size>")
        sys.exit(1)

    rows = doc_round1(path)
    if not rows:
        print("Khong tim thay dong 'Client ... UpdateNorm:' nao trong log.")
        sys.exit(1)

    for r in rows:
        r["tau"] = math.ceil(r["n"] / bs)

    # k = n_bn / n_par, uoc luong tu cac client lon (noi bo dem ap dao neu con bug)
    big = sorted(rows, key=lambda r: -r["tau"])[:max(3, len(rows) // 5)]
    k = sum(r["upd"] ** 2 for r in big) / sum(r["tau"] ** 2 for r in big)

    print(f"batch_size = {bs} | {len(rows)} client o round 1")
    print(f"He so khop k = n_bn/n_par = {k:.6e}  "
          f"(vd 4/22757 = {4/22757:.6e} cho CNN1D task 0)\n")
    print(f"{'client':>6} {'n_i':>12} {'tau':>6} {'UpdNorm do':>12} "
          f"{'du doan':>10} {'do/du doan':>11}")
    print("-" * 62)

    # Chi xet client co du batch de bo dem ap dao (tau >= 10). Voi client 1-2
    # batch thi trong so that va bo dem cung bac, ti so khong noi len dieu gi.
    lon = [r for r in sorted(rows, key=lambda r: -r["tau"]) if r["tau"] >= 10][:15]
    if not lon:
        lon = sorted(rows, key=lambda r: -r["tau"])[:15]
    ratios = []
    for r in lon:
        pred = math.sqrt(k) * r["tau"]
        ratio = r["upd"] / pred if pred > 0 else float("nan")
        ratios.append(ratio)
        print(f"{r['cid']:>6} {r['n']:>12,} {r['tau']:>6} {r['upd']:>12.4f} "
              f"{pred:>10.4f} {ratio:>11.4f}")

    # R^2 cua UpdateNorm theo tau
    xs = [r["tau"] for r in rows]
    ys = [r["upd"] for r in rows]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    r2 = (sxy ** 2 / (sxx * syy)) if sxx * syy > 0 else 0.0

    print("\n" + "=" * 62)
    print(f"R^2 cua UpdateNorm theo tau (so batch) = {r2:.6f}")
    spread = max(ratios) - min(ratios)
    print(f"Bien do ti so do/du doan tren {len(ratios)} client nhieu batch nhat "
          f"= {spread:.4f}")
    print("=" * 62)
    if r2 > 0.99 and spread < 0.05:
        print(">>> VAN CON BUG: UpdateNorm gan nhu la ham tuyen tinh cua so batch.")
    elif r2 > 0.9:
        print(">>> NGHI NGO: van con tuong quan manh voi so batch. Kiem lai co "
              "legacy_update_norm va exclude_bn_stats_from_norm.")
    else:
        print(">>> DA HET: UpdateNorm khong con bam so batch — no dang do trong so that.")


if __name__ == "__main__":
    main()
