import csv, glob, json, os, re, shutil, signal, subprocess, sys, time, zipfile

# CELL KAGGLE: thi nghiem ADAPTER TONG HOP (nhanh exp/adapter-tong-hop, config exp_adapter_tong_hop.json).
# Adapter va gate duoc tong hop o server va giu qua cac task. Chay task 2-5, la_tau = 0.5,
# hieu chinh tau* moi round, kich ban 1% co nhan 1/kshot khi chon tau*. Khong dung gate kiem dinh.
#
# Moi kich ban mat ~14,5 gio (4 task) -> 2 phien Kaggle:
#   Phien 1: JOBS = [("full", "afsictask0")]            chay tu checkpoint sau task 1
#   Phien 2: JOBS = [("full", "ketqua_exp_adapter_tong_hop_full_la05")]
#            (Add Input output/zip cua phien 1; cell tu tim checkpoint task cao nhat)
# Tuong tu cho "10shot" va "1percent". Co the chay 3 kich ban song song tren 3 tai khoan.
JOBS = [("full", "afsictask0")]
BRANCH = "exp/adapter-tong-hop"
LA_TAU = 0.5
EVAL_EVERY = 1
TIME_BUDGET_H = 11.6

REPO = "https://github.com/TongXuanVu/AFSIC-IOV.git"
WORK = "/kaggle/working"
CODE = os.path.join(WORK, "AFSIC-IOV")
CKPT_DIR = os.path.join(WORK, "ckpt_in")
BASE_NAME = "exp_adapter_tong_hop"
T_START = time.time()

assert glob.glob("/kaggle/input/**/federated_data/client_*_task_*.pt", recursive=True) and \
       glob.glob("/kaggle/input/**/global_test_data.pt", recursive=True), "Thieu dataset IoV 100 client"


def _task_of(p):
    return int(re.search(r"ckpt_task(\d+)_memory_client99\.pth$", p).group(1))


def find_ckpt(key):
    """Tim ckpt_taskXX_memory_client99.pth (task cao nhat) co duong dan chua `key`, trong
    /kaggle/input hoac trong cac file zip cua input (tu giai nen vao /kaggle/working/ckpt_in)."""
    def scan():
        c = glob.glob("/kaggle/input/**/ckpt_task*_memory_client99.pth", recursive=True) + \
            glob.glob(f"{CKPT_DIR}/**/ckpt_task*_memory_client99.pth", recursive=True)
        return [p for p in c if key in p]
    cands = scan()
    if not cands:
        for zp in glob.glob("/kaggle/input/**/*.zip", recursive=True):
            with zipfile.ZipFile(zp) as z:
                names = z.namelist()
                if key in os.path.basename(zp) or any(key in n for n in names):
                    dst = os.path.join(CKPT_DIR, os.path.splitext(os.path.basename(zp))[0])
                    z.extractall(dst)
        cands = scan()
    assert cands, f"Khong thay checkpoint nao co '{key}' trong duong dan"
    mdl = sorted(set(cands), key=_task_of)[-1]
    mem = mdl.replace(".pth", "_MEM.pth")
    assert os.path.exists(mem), f"Thieu {mem}"
    return mdl, mem


RES = {sc: find_ckpt(key) for sc, key in JOBS}
for sc, (mdl, _) in RES.items():
    t = _task_of(mdl)
    assert t < 4, f"{sc}: checkpoint da la sau task 5, khong con gi de chay"
    print(f"[resume] {sc}: da xong task {t + 1} -> chay tiep tu task {t + 2} | {mdl}")

shutil.rmtree(CODE, ignore_errors=True)
subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH, REPO, CODE], check=True)
os.chdir(CODE)
COMMIT = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
print("[code]", BRANCH, "@", COMMIT, "|", subprocess.check_output(["git", "log", "--oneline", "-1"]).decode().strip())
_src = "".join(open(f, encoding="utf-8").read() for f in ("trainer.py", "utils/inc_net.py"))
for _f in ("keep_adapter_across_tasks", "_client_tau_selection", "calib_fewshot_ratio_scale"):
    assert _f in _src, f"Code chua co {_f} - sai nhanh hoac chua push?"

for SCENARIO, key in JOBS:
    left_s = TIME_BUDGET_H * 3600 - (time.time() - T_START)
    if left_s < 1800:
        print(f"\n[bo qua] {SCENARIO}: con duoi 30 phut")
        continue
    ratio = SCENARIO == "1percent"
    CONFIG_NAME = f"{BASE_NAME}_{SCENARIO}_la{str(LA_TAU).replace('.', '')}"
    CONFIG = f"configs/exps/{CONFIG_NAME}.json"
    cfg = json.load(open(f"configs/exps/{BASE_NAME}.json", encoding="utf-8"))
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    if SCENARIO == "1percent":
        cfg.update(fewshot_enabled=True, kshot=0.01)
    elif SCENARIO == "10shot":
        cfg.update(fewshot_enabled=True, kshot=10)
    cfg["eval_every_n_rounds"] = int(EVAL_EVERY)
    cfg["la_tau"] = float(LA_TAU)
    cfg["calib_fewshot_ratio_scale"] = bool(ratio)
    cfg["quality_gated_update"] = False
    cfg.pop("max_tasks", None)
    json.dump(cfg, open(CONFIG, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    for k, v in {"personalized_adapter": False, "aggregate_encoder_only": False, "keep_adapter_across_tasks": True,
                 "shared_encoder_trainable": True, "single_shared_encoder": True, "adapter_input": "raw",
                 "decision_calibration": True, "calib_every_n_rounds": 1, "local_loss_mode": "logit_adjusted",
                 "class_weight_power": 0.0, "memory_ratio": 0.01, "lr": 3e-05, "lr_adapter": 0.001,
                 "num_clients": 100, "num_rounds": 30}.items():
        assert cfg.get(k) == v, f"{k}={cfg.get(k)!r}, can {v!r}"
    mdl, mem = RES[SCENARIO]
    print(f"\n========== {CONFIG_NAME} | la_tau={cfg['la_tau']} | ratio_scale={ratio} | fewshot={cfg['fewshot_enabled']} "
          f"kshot={cfg['kshot']} | con {left_s / 3600:.2f} gio ==========")

    before = set(glob.glob("logs/afsic-iov_federated/can_iov/*_clients100"))
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, "main.py", "--config", CONFIG, "--resume", mdl, "--resume_memory", mem])
    stopped = False
    try:
        proc.wait(timeout=max(60, left_s))
    except subprocess.TimeoutExpired:
        stopped = True
        print(f"\n[het gio] dung {SCENARIO} de dong goi ket qua")
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
    print(f"[xong] {SCENARIO}: {(time.time() - t0) / 3600:.2f} gio | {'DUNG GIUA CHUNG' if stopped else 'chay het'}")

    new_runs = sorted(set(glob.glob("logs/afsic-iov_federated/can_iov/*_clients100")) - before, key=os.path.getmtime)
    if not new_runs:
        print("[loi] khong thay thu muc log moi")
        continue
    run = new_runs[-1]
    open(os.path.join(run, "CODE_COMMIT.txt"), "w").write(
        f"{BRANCH} @ {COMMIT}\nconfig {CONFIG_NAME} (la_tau={LA_TAU}, ratio_scale={ratio}, gate=False)\nresume {mdl}\n")
    mpath = os.path.join(run, "metrics_round_by_round.csv")
    if os.path.exists(mpath):
        rows = list(csv.DictReader(open(mpath, encoding="utf-8")))
        print(f"KET QUA {SCENARIO} (sau hieu chinh tau*, round 30 moi task):")
        for r in rows:
            if r["round"] == "30":
                print(f"  task {int(r['task']) + 1}: acc {float(r['acc']):.2f} | f1_mac {float(r['f1_mac']):.2f} | "
                      f"f1_wei {float(r['f1_wei']):.2f} | tau* {r['tau_star']}")
    done = sorted(glob.glob(os.path.join(run, "checkpoints", "ckpt_task*_memory_client99.pth")))
    if done:
        print(f"[chay tiep] checkpoint moi nhat: {os.path.basename(done[-1])}. Phien sau: "
              f"JOBS = [(\"{SCENARIO}\", \"ketqua_{CONFIG_NAME}\")] va Add Input output cua phien nay.")
    zpath = os.path.join(WORK, f"ketqua_{CONFIG_NAME}_resume.zip")
    with zipfile.ZipFile(zpath, "w", allowZip64=True) as z:
        for root, _, files in os.walk(run):
            for fn in sorted(files):
                if fn.endswith(".pth") and not re.match(r"ckpt_task\d+_memory_client99(_MEM)?\.pth$", fn):
                    continue
                full = os.path.join(root, fn)
                arc = os.path.join(f"ketqua_{CONFIG_NAME}", os.path.basename(run), os.path.relpath(full, run))
                z.write(full, arc, compress_type=zipfile.ZIP_STORED if fn.endswith(".pth") else zipfile.ZIP_DEFLATED)
    print(f"[zip] {zpath} ({os.path.getsize(zpath) / 1e9:.2f} GB)")

print(f"\n[het] tong {(time.time() - T_START) / 3600:.2f} gio")
