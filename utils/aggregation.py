import logging
import numpy as np
import torch
import torch.nn.functional as F

def is_aggregated_state_key(key, task, aggregate_backbone=False):
    # Lưu ý PerFL (personalized_adapter): server VẪN aggregate adapter/gate để
    # global model có nhánh plasticity ý nghĩa khi đánh giá; tính cá nhân hóa
    # nằm ở phía client — client KHÔNG nạp đè adapter/gate cục bộ của mình
    # (xem trainer._load_global_into_client).
    if task == 0 or aggregate_backbone:
        return True
    if "plasticity_adapter.frozen_source" in key:
        return False
    return any(sub in key for sub in ["plasticity_adapter.adapter", "gate", "fc"])

def compute_aggregation_weights(
    args,
    global_model,
    client_accs,
    client_protos,
    client_weights,
    global_state_round_start,
    active_client_indices,
    task
):
    Q_list = []
    drift_list = []
    update_norm_list = []
    
    beta_acc = args.get("beta_acc", 1.0)
    beta_proto = args.get("beta_proto", 1.0)
    beta_novelty = args.get("beta_novelty", 0.5)
    beta_drift = args.get("beta_drift", 0.5)
    beta_update = args.get("beta_update", 0.2)
    # beta_n: trọng số theo QUY MÔ DỮ LIỆU của client. Mặc định 0.0 nên không
    # đổi hành vi của mọi config cũ.
    #
    # VÌ SAO CẦN: Q_i gốc chỉ gồm accuracy, độ nhất quán prototype, độ mới,
    # drift và update-norm — KHÔNG có thành phần nào theo n_k. Với dữ liệu IoV
    # 10 client (3 client nắm 89,8% dữ liệu), softmax cho alpha gần đều nhau,
    # khiến client chỉ có 4.414 mẫu (0,004% dữ liệu) vẫn chiếm 14,3% mô hình
    # toàn cục — khuếch đại 3.190 lần so với tỉ trọng thật. Đặc tả mục 5.7 có
    # beta_1*log(1+n) trong r_{i,c}, nhưng Q_i thì thiếu.
    #
    # size_term_mode quyết định cách đưa quy mô vào Q_i:
    #
    #   "norm" (khuyến nghị) — chuẩn hoá log(1+n_i) về [0,1] theo min–max giữa
    #       các client active, rồi nhân beta_n. GIỮ ĐƯỢC các cơ chế
    #       reliability-aware, vì mọi số hạng cùng thang [0,1].
    #
    #   "raw" — cộng thẳng beta_n*log(1+n_i). CẢNH BÁO: log(1+n) đạt 17,19 với
    #       client 29 triệu mẫu, trong khi acc và proto_cons chỉ trong [0,1],
    #       nên số hạng này áp đảo hoàn toàn và Q_i thực chất chỉ còn phụ thuộc
    #       n. Với beta_n=1 + mọi beta khác=0 + tau=1 thì cho đúng FedAvg chuẩn
    #       (softmax(log n) = n_k/sum(n)) — dùng để dựng BASELINE FedAvg, KHÔNG
    #       phải để "sửa" AFSIC-IoV.
    beta_n = args.get("beta_n", 0.0)
    size_term_mode = args.get("size_term_mode", "norm")

    # ── Drift vs UpdateNorm ──────────────────────────────────────────────────
    # Đặc tả mục 5.7 mô tả beta_4·Drift và beta_5·UpdateNorm là HAI tiêu chí
    # độc lập. Bản gốc lại cộng dồn cả hai bằng đúng một biểu thức nên
    # drift_i == update_norm_i luôn luôn — thực chất chỉ là một đại lượng bị
    # trừ hai lần với hệ số (beta_4 + beta_5).
    #
    #   drift_mode = "vs_global" (mặc định)  → giữ nguyên hành vi cũ
    #   drift_mode = "vs_mean"               → tách đúng ngữ nghĩa:
    #        UpdateNorm_i = ||theta_i - theta_global||   (độ lớn update của client)
    #        Drift_i      = ||theta_i - mean_j(theta_j)|| (lệch khỏi xu hướng chung)
    #
    # Chế độ "vs_mean" cần biết trung bình update của mọi client nên phải tính
    # trước vòng lặp chính.
    drift_mode = args.get("drift_mode", "vs_global")
    agg_bb = args.get("aggregate_backbone", False)

    diff_cache = {}          # c_idx -> {key: tensor diff}
    update_norm_cache = {}   # c_idx -> float
    for c_idx, _c in enumerate(active_client_indices):
        local_dict = client_weights[c_idx]
        diffs, sq, n_par = {}, 0.0, 0
        for k in local_dict.keys():
            if is_aggregated_state_key(k, task, agg_bb):
                d = local_dict[k].float() - global_state_round_start[k].float()
                diffs[k] = d
                sq += torch.sum(d ** 2).item()
                n_par += d.numel()
        diff_cache[c_idx] = diffs
        update_norm_cache[c_idx] = (np.sqrt(sq / max(1, n_par)), n_par)

    drift_cache = {}
    if drift_mode == "vs_mean" and len(active_client_indices) > 1:
        keys = set().union(*(set(d) for d in diff_cache.values())) if diff_cache else set()
        mean_diff = {k: torch.stack([diff_cache[i][k] for i in diff_cache if k in diff_cache[i]]).mean(0)
                     for k in keys}
        for c_idx in diff_cache:
            sq, n_par = 0.0, 0
            for k, d in diff_cache[c_idx].items():
                dev = d - mean_diff[k]
                sq += torch.sum(dev ** 2).item()
                n_par += dev.numel()
            drift_cache[c_idx] = np.sqrt(sq / max(1, n_par))
    else:
        for c_idx in diff_cache:
            drift_cache[c_idx] = update_norm_cache[c_idx][0]

    diff_cache.clear()   # giải phóng bộ nhớ

    # Số mẫu mỗi client (từ 'count' của prototype từng lớp) và số hạng quy mô.
    n_cache = {c: sum(int(info.get("count", 0)) for info in client_protos[c].values())
               for c in active_client_indices}
    log_n = {c: float(np.log1p(max(n_cache[c], 0))) for c in active_client_indices}
    if size_term_mode == "norm":
        lo, hi = min(log_n.values()), max(log_n.values())
        size_cache = {c: (log_n[c] - lo) / (hi - lo) if hi > lo else 0.0
                      for c in active_client_indices}
    else:                       # "raw"
        size_cache = dict(log_n)

    for c_idx, c in enumerate(active_client_indices):
        acc_i = client_accs[c]
        
        # Prototype Consistency
        proto_cons_vals = []
        for class_id in range(global_model._total_classes):
            local_p = client_protos[c].get(class_id, {}).get("prototype")
            global_p = global_model.global_proto_memory.get_prototype(class_id)
            if local_p is not None and global_p is not None:
                sim = torch.sum(F.normalize(local_p, p=2, dim=0) * F.normalize(global_p, p=2, dim=0)).item()
                proto_cons_vals.append(sim)
        proto_cons_i = sum(proto_cons_vals) / len(proto_cons_vals) if proto_cons_vals else 1.0
        
        # Novelty
        novelty_vals = []
        new_classes = range(global_model._known_classes, global_model._total_classes)
        old_classes = range(global_model._known_classes)
        if new_classes and old_classes:
            for n_c in new_classes:
                local_p = client_protos[c].get(n_c, {}).get("prototype")
                if local_p is not None:
                    local_p = F.normalize(local_p, p=2, dim=0)
                    min_dist = 1.0
                    for o_c in old_classes:
                        global_p = global_model.global_proto_memory.get_prototype(o_c)
                        if global_p is not None:
                            global_p = F.normalize(global_p, p=2, dim=0)
                            dist = 1.0 - torch.sum(local_p * global_p).item()
                            if dist < min_dist:
                                min_dist = dist
                    novelty_vals.append(min_dist)
        novelty_i = sum(novelty_vals) / len(novelty_vals) if novelty_vals else 0.5
        
        # Drift và UpdateNorm — đã tính sẵn ở hai lượt phía trên
        update_norm_i = update_norm_cache[c_idx][0]
        drift_i = drift_cache[c_idx]
        
        n_i = n_cache[c]
        size_term = beta_n * size_cache[c]

        Q_i = size_term + beta_acc * acc_i + beta_proto * proto_cons_i + beta_novelty * novelty_i - beta_drift * drift_i - beta_update * update_norm_i
        Q_list.append(Q_i)
        drift_list.append(drift_i)
        update_norm_list.append(update_norm_i)

        logging.info(
            f"Client {c} => Q_i: {Q_i:.4f} | n_i: {n_i} | Acc: {acc_i*100:.2f}% | "
            f"ProtoCons: {proto_cons_i:.4f} | Novelty: {novelty_i:.4f} | "
            f"Drift: {drift_i:.4f} | UpdateNorm: {update_norm_i:.4f}"
        )
    
    accepted_positions = list(range(len(Q_list)))
    if args.get("robust_filter_updates", True) and len(Q_list) > 2:
        update_arr = np.array(update_norm_list, dtype=np.float64)
        drift_arr = np.array(drift_list, dtype=np.float64)
        update_med = float(np.median(update_arr))
        drift_med = float(np.median(drift_arr))
        update_mad = float(np.median(np.abs(update_arr - update_med))) + 1e-8
        drift_mad = float(np.median(np.abs(drift_arr - drift_med))) + 1e-8
        z_limit = args.get("robust_z", 3.5)
        max_update_norm = args.get("max_update_norm", None)
        accepted_positions = []
        for pos, c in enumerate(active_client_indices):
            update_ok = (update_arr[pos] - update_med) / update_mad <= z_limit
            drift_ok = (drift_arr[pos] - drift_med) / drift_mad <= z_limit
            norm_ok = max_update_norm is None or update_arr[pos] <= float(max_update_norm)
            if update_ok and drift_ok and norm_ok:
                accepted_positions.append(pos)
            else:
                logging.warning(
                    f"Client {c} rejected by robust filter | "
                    f"Drift: {drift_arr[pos]:.4f} | UpdateNorm: {update_arr[pos]:.4f}"
                )
        if not accepted_positions:
            logging.warning("Robust filter rejected all clients; falling back to all active clients.")
            accepted_positions = list(range(len(Q_list)))

    Q_accepted = [Q_list[pos] for pos in accepted_positions]
    Q_tensor = torch.tensor(Q_accepted, dtype=torch.float32)
    tau_agg = args.get("tau_aggregation", 1.0)
    alpha = torch.softmax(Q_tensor / tau_agg, dim=0).tolist()

    # Stats per-client (theo vị trí trong active_client_indices) cho
    # per-class reliability-aware prototype aggregation ở trainer.
    client_stats = {"drift": drift_list, "update_norm": update_norm_list}

    return alpha, accepted_positions, Q_accepted, client_stats
