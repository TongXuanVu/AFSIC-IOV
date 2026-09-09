import json
import argparse
import os
from trainer import train, run_test

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./configs/exps/can_iov_afsic.json',
                        help='Json file of settings.')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help=(
                            'train: chay training day du va luu checkpoint sau moi task. '
                            'test : chi load checkpoint da luu va chay evaluation.'
                        ))
    parser.add_argument('--test_checkpoint_dir', type=str, default='',
                        help='Duong dan thu muc chua checkpoint khi chay --mode test. '
                             'Vi du: ./logs/afsic-iov_federated/can_iov/16-07-26_03-38_seed42_cnn1d_clients10')
    parser.add_argument('--debug', action='store_true',
                        help='Che do debug: giam so epoch xuong 2 de test nhanh.')
    parser.add_argument('--resume', type=str, default='',
                        help='Duong dan den checkpoint (.pth) de tiep tuc training.')
    parser.add_argument('--resume_memory', type=str, default='',
                        help=('Duong dan den file bo nho exemplar rieng (*_MEM.pth). '
                              'Tu ban tach file, checkpoint model khong con chua exemplar. '
                              'Dung lai mot file _MEM cho nhieu lan chay se bo qua han pha '
                              'herding — khau dat nhat cua chuong trinh.'))
    parser.add_argument('--test_rounds', type=str, default=None,
                        help='Chi danh gia cac vong nay khi --mode test, vi du "11,30". '
                             'Bo trong = danh gia toan bo checkpoint trong thu muc.')
    parser.add_argument('--logit_prior_adjust', action='store_const', const=True, default=None,
                        help='Hieu chinh prior o logit luc SUY LUAN: cong tau*log(pi_c) vao '
                             'logit moi lop, pi_c uoc luong tu so mau HUAN LUYEN toan lien '
                             'doan. Mac dinh TAT.')
    parser.add_argument('--logit_prior_tau', type=float, default=None,
                        help='He so tau cho hieu chinh prior (mac dinh 1.0).')
    parser.add_argument('--logit_prior_tau_sweep', type=str, default=None,
                        help='Quet nhieu tau trong MOT lan chay --mode test, vi du '
                             '"0,0.25,0.5,0.75,1.0". Ket qua ghi ra tau_sweep.csv. '
                             'tau=0 chinh la ket qua goc khong hieu chinh.')
    parser.add_argument('--memory_size', type=int, default=None,
                        help='Tong so luong mau luu trong bo nho dem (Exemplar memory).')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size cho training.')
    parser.add_argument('--init_epoch', type=int, default=None,
                        help='So epoch cho task dau tien.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='So epoch cho cac task sau.')
    
    # --- Federated Learning Arguments ---
    parser.add_argument('--num_clients', type=int, default=None,
                        help='So luong client tham gia FL.')
    parser.add_argument('--num_rounds', type=int, default=None,
                        help='So vong aggregation (rounds) moi task.')
    parser.add_argument('--local_epochs', type=int, default=None,
                        help='So epoch huan luyen cuc bo cua moi client.')
    
    return parser

def main():
    args_cli = setup_parser().parse_args()
    param = load_json(args_cli.config)
    
    # Merge logic: JSON config acts as base, CLI args override it
    config = load_json(args_cli.config)
    
    # Remove None values from CLI args so they don't overwrite JSON defaults unnecessarily
    cli_args = {k: v for k, v in vars(args_cli).items() if v is not None}
    
    # Final args: JSON base + CLI overrides
    args = config
    args.update(cli_args)

    # --logit_prior_tau_sweep nhan chuoi "0,0.5,1" -> list float
    _sw = args.get("logit_prior_tau_sweep")
    if isinstance(_sw, str):
        args["logit_prior_tau_sweep"] = [
            float(x) for x in _sw.replace(" ", "").split(",") if x
        ]

    if args.get("debug"):
        print("[DEBUG] Che do debug: set init_epoch=2, epochs=2, local_epochs=1, num_rounds=2")
        args["init_epoch"] = 2
        args["epochs"] = 2
        args["local_epochs"] = 1
        args["num_rounds"] = 2

    if args.get("mode", "train") == "test":
        # Giai doan Test: chi load checkpoint va chay evaluation
        if not args.get("run_dir") and not args.get("test_checkpoint_dir"):
            print("[ERROR] --mode test yeu cau 'run_dir' hoac 'test_checkpoint_dir' trong config.")
            return
        run_test(args)
    else:
        # Giai doan Train
        train(args)

if __name__ == '__main__':
    main()
