import copy
import logging
import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.base import BaseLearner
from utils.fast_loader import make_loader
from utils.inc_net import AFSICIDSNet
from utils.memory import LocalExemplarMemory, GlobalPrototypeMemory
from losses import compute_kd_loss, compute_fsp_loss, compute_proto_loss, compute_sparse_regularization, compute_fedprox_regularization

class AFSIC_IDS(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = AFSICIDSNet(args, False)
        self._old_network = None
        self.local_memory = LocalExemplarMemory(
            memory_ratio=args.get("memory_ratio", 0.01),
            memory_per_class=args.get("memory_per_class", None)
        )
        self.global_proto_memory = GlobalPrototypeMemory()
        self.best_acc_per_task = {}

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager, skip_train=False):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        
        # Update network fc size
        self._network.update_fc(self._total_classes)
        self._network.to(self._device)
        
        logging.info("Learning on classes {}-{}".format(self._known_classes, self._total_classes))
        
        if self._cur_task > 0:
            # Transition to incremental task: freeze stability encoder, initialize new adapter/gate
            self._network.transition_to_incremental_stage()
            self._network.freeze_stability_encoder()
            self._network.unfreeze_adapter()
            
            # _old_network for KD must be the previous global model before expansion.
            # The federated trainer injects it before local training; do not copy
            # the newly expanded adapter model here.
        
        # Setup Test Loader
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = make_loader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
        )

        if skip_train:
            logging.info(f"Skipping training for task {self._cur_task}")
            if not hasattr(self, 'train_loader'):
                self.train_loader = None
            return

        # Setup Train Loader
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = make_loader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        # Optimize only trainable incremental parameters during incremental stages
        if self._cur_task > 0:
            params = self._network.get_trainable_incremental_params()
        else:
            params = self._network.parameters()

        # LR decay theo ROUND liên bang: optimizer/scheduler bị tạo lại mỗi round
        # nên MultiStepLR trong-round không bao giờ chạm milestones (local_epochs=1
        # → lr đứng nguyên suốt run, gây dao động acc giữa các round cuối).
        # Trainer set args["current_round"]; lr hiệu dụng = lr · gamma^(số
        # milestone đã vượt qua). Khi đó milestones trong-round bị vô hiệu để
        # không decay hai lần.
        lr = self.args.get("lr", 0.001)
        gamma = self.args.get("gamma", 0.1)
        milestones = list(self.args.get("milestones", [80, 120, 150]))
        current_round = self.args.get("current_round", None)
        if current_round is not None:
            lr = lr * (gamma ** sum(1 for m in milestones if current_round >= m))
            epoch_milestones = []
        else:
            epoch_milestones = milestones

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, params),
            lr=lr,
            weight_decay=self.args.get("weight_decay", 0.0002),
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=epoch_milestones,
            gamma=gamma
        )

        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        start_round = self.args.get("start_round", 0)
        epochs = self.args.get("epochs", 30)
        
        # 1. Compute class weights for Class-balanced CE
        #
        # TOI UU (chinh xac tuyet doi, khong phai xap xi): class_counts chi phu
        # thuoc DU LIEU nen trong mot task no KHONG DOI qua cac round. Ban cu
        # duyet lai TOAN BO train_loader moi round chi de dem nhan — voi client
        # 12,6 trieu mau day la mot luot doc du lieu day du, ngang chi phi mot
        # epoch huan luyen. Cache theo (_cur_task, _total_classes).
        _ck = getattr(self, "_class_counts_cache", None)
        if _ck is not None and _ck[0] == (self._cur_task, self._total_classes):
            class_counts = _ck[1].to(self._device)
        else:
            class_counts = torch.zeros(self._total_classes).to(self._device)
            for _, _, targets in train_loader:
                targets = targets.to(self._device)
                class_counts += torch.bincount(targets, minlength=self._total_classes)
            self._class_counts_cache = ((self._cur_task, self._total_classes),
                                        class_counts.detach().cpu().clone())
        
        total_samples = class_counts.sum()
        class_weights = torch.zeros(self._total_classes).to(self._device)
        # Lam diu bang CAN BAC HAI roi chuan hoa mean=1 (theo HFIN/SPCIL).
        #
        # Ban cu dung nghich dao tan suat THO: w_c = N/(n_c*K). Ti le w giua
        # Benign (29M mau) va lop hiem (300 mau) khi do dung bang ti le so
        # luong = 96.667:1. Hau qua do duoc: trong mot lo 8192 mau co 8190
        # Benign + 2 mau tan cong, hai mau do ganh ~75% loss cua ca lo -> huong
        # gradient thuc chat la nhieu tu vai mau, va day la nguon cua trang
        # thai luong on dinh (task 0 lat o round 21: acc 99,1 -> 0,3).
        # Can bac hai dua ti le ve 311:1, van chong mat can bang nhung khong
        # con de mot nhum mau lai ca lo.
        _smooth = float(self.args.get("class_weight_power", 0.5))
        for c in range(self._total_classes):
            if class_counts[c] > 0:
                class_weights[c] = (total_samples / (class_counts[c] * self._total_classes)) ** _smooth
            else:
                class_weights[c] = 1.0
        class_weights = class_weights / class_weights.mean().clamp_min(1e-12)

        # Save model params at start of local training round for FedProx
        self.global_model_params_round_start = {
            name: p.clone().detach()
            for name, p in self._network.named_parameters()
            if p.requires_grad
        }

        # Build prototype matrix for FSP and Proto Alignment
        proto_matrix = torch.zeros(self._total_classes, self._network.feature_dim).to(self._device)
        for c in range(self._total_classes):
            proto = self._get_reference_prototype(c)
            if proto is not None:
                proto_matrix[c] = proto.to(self._device)
            else:
                # fallback to normalized classifier weight
                proto_matrix[c] = F.normalize(self._network.fc.weight.data[c], p=2, dim=0)

        for epoch in range(start_round, epochs):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits = outputs["logits"]
                features = outputs["features"]
                
                # a. Class-balanced CE Loss
                loss_ce = F.cross_entropy(logits, targets, weight=class_weights)
                
                # b. KD Loss on old classes
                loss_kd = torch.tensor(0.0).to(self._device)
                if self._cur_task > 0 and self._old_network is not None:
                    self._old_network.eval()
                    with torch.no_grad():
                        old_outputs = self._old_network(inputs)
                        old_logits = old_outputs["logits"][:, :self._known_classes]
                    new_logits = logits[:, :self._known_classes]
                    loss_kd = compute_kd_loss(new_logits, old_logits, T=2.0)
                
                # c. FSP Loss (Few-Shot Sparse Pairwise Loss)
                # CO TRONG SO LOP. Ban cu dung torch.mean tron -> trong lo 8192
                # mau voi 8190 Benign, hai mau tan cong chi dong gop 0,024%.
                # FSP keo z ve prototype dung va day khoi prototype gan nhat;
                # neu 99,6% cap la (Benign, lop-gan-Benign) thi khong bao gio co
                # ap luc tach DoS khoi double -> chinh la goc 18,4 do do duoc.
                # Dung lai class_weights da tinh o tren, khong tinh them gi.
                _w = class_weights[targets]
                _wsum = _w.sum().clamp_min(1e-12)
                loss_fsp = (_w * compute_fsp_loss(
                    features, targets, proto_matrix, T_fsp=0.5, reduction="none"
                )).sum() / _wsum
                
                # d. Prototype Alignment Loss (cung co trong so lop, cung ly do)
                loss_proto = (_w * compute_proto_loss(
                    features, targets, proto_matrix,
                    normalize=self.args.get("proto_loss_normalize", True),
                    reduction="none",
                )).sum() / _wsum
                
                # e. Sparse Regularization Loss (L1 norm on adapter and gate parameters)
                loss_rs = torch.tensor(0.0).to(self._device)
                if self._cur_task > 0:
                    loss_rs = compute_sparse_regularization(self._network, self._device)
                
                # f. FedProx Regularization Loss
                loss_prox = torch.tensor(0.0).to(self._device)
                if self._cur_task > 0:
                    loss_prox = compute_fedprox_regularization(self._network, self.global_model_params_round_start, self._device)
                
                # Hyperparameters. Base stage should use CE only; prototype losses
                # are meaningful after old-class prototypes exist.
                if self._cur_task == 0:
                    lambda_kd = lambda_fsp = lambda_proto = lambda_rs = lambda_prox = 0.0
                else:
                    lambda_kd = self.args.get("lambda_kd", 1.0)
                    lambda_fsp = self.args.get("lambda_fsp", 0.5)
                    lambda_proto = self.args.get("lambda_proto", 0.5)
                    lambda_rs = self.args.get("lambda_rs", 0.0001)
                    lambda_prox = self.args.get("lambda_prox", 0.01)
                
                loss = loss_ce + lambda_kd * loss_kd + lambda_fsp * loss_fsp + lambda_proto * loss_proto + lambda_rs * loss_rs + lambda_prox * loss_prox
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)
                
            scheduler.step()
            
            train_acc = np.around(correct.item() * 100 / total, decimals=2)
            if test_loader is not None:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc, test_acc["total"]
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc
                )
            logging.info(info)

    def _get_reference_prototype(self, class_id):
        """Prototype tham chiếu cho loss FSP/proto và calibration.

        Bản gốc dùng global prototype; AFSIC-IoV override để trộn
        local/global (personalized prototype).
        """
        return self.global_proto_memory.get_prototype(class_id)

    def _get_memory(self):
        return self.local_memory.get_memory()

    @property
    def exemplar_size(self):
        mem = self._get_memory()
        if mem is None:
            return 0
        return len(mem[1])

    def build_rehearsal_memory(self, data_manager, per_class):
        """Trích feature dạng streaming vào ma trận float16 cấp phát sẵn rồi herding.

        Tránh giữ ma trận float32 khổng lồ + spike torch.cat với lớp hàng chục
        triệu mẫu (Benign ~29M) — nguyên nhân OOM trên Kaggle.
        """
        self._network.eval()
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            n = len(data)
            if n == 0:
                continue

            loader = make_loader(dset, batch_size=self.args["batch_size"], shuffle=False)
            features = np.empty((n, self._network.feature_dim), dtype=np.float16)
            pos = 0
            with torch.no_grad():
                for _, inputs, _ in loader:
                    feats = self._network.extract_vector(inputs.to(self._device))
                    feats = F.normalize(feats, p=2, dim=1)
                    features[pos:pos + feats.shape[0]] = feats.cpu().numpy().astype(np.float16)
                    pos += feats.shape[0]

            self.local_memory.construct_exemplars(
                class_idx, data, targets, features,
                features_normalized=True, device=self._device,
            )

    def compute_local_prototypes(self, data_manager, class_ids=None, max_samples_per_class=None, seed=0,
                                 report_full_count=False):
        """Tính prototype dạng streaming 1 lượt — không giữ ma trận feature trong RAM.

        Tương đương chính xác với bản cũ về mặt toán học: với m = mean(feature
        đã chuẩn hóa L2) thì prototype = m/||m|| và
        dispersion = mean(1 − cos(f, prototype)) = 1 − ||m||.
        """
        local_protos = {}
        self._network.eval()

        # Compute prototypes for selected active classes.
        if class_ids is None:
            class_ids = range(self._total_classes)

        rng = np.random.default_rng(seed)
        for class_idx in class_ids:
            if max_samples_per_class is not None:
                # Lay mau tren VIEW (SubDummyDataset), KHONG dung ret_data=True.
                # ret_data=True copy nguyen mang cua lop truoc khi lay mau — voi
                # lop Benign 29 trieu mau day la ~1,8 GB moi client moi round,
                # dat hon chinh phep tinh no dinh thay the.
                from utils.data_manager import SubDummyDataset
                dset = data_manager.get_dataset(
                    np.arange(class_idx, class_idx + 1),
                    source="train",
                    mode="test",
                )
                n_all = len(dset)
                if n_all == 0:
                    continue
                # max_samples_per_class < 1 => hiểu là TỈ LỆ (kịch bản 1%);
                # >= 1 => số mẫu tuyệt đối (10-shot, proto_max_samples).
                _cap = float(max_samples_per_class)
                _cap = max(1, int(round(_cap * n_all))) if _cap < 1.0 else int(_cap)
                if n_all > _cap and isinstance(dset, SubDummyDataset) and dset.append_x is None:
                    selected = np.sort(rng.choice(dset.len_source, size=_cap, replace=False))
                    dset = SubDummyDataset(
                        dset.x_source, dset.y_source,
                        np.asarray(dset.indices)[selected],
                        dset.trsf, dset.use_path, None)
                    n_used = _cap
                else:
                    n_used = n_all
                # "count" tra ve duoc dung lam n_i o khau gop (size_term
                # beta_n, tau_local cua FedNova, r_ic cua gop prototype,
                # n_samples cua bn_stats_by_count). Phan biet HAI truong hop:
                #
                #   report_full_count=True  — cat mau chi de ĐO (proto_max_samples).
                #       Client van train tren toan bo du lieu, nen n_i phai la so
                #       mau THAT. Neu bao so da lay mau thi viec toi uu se ngam
                #       bop meo dung nhung dai luong dang nghien cuu.
                #
                #   report_full_count=False — cat mau la THAT (kshot/1%): client
                #       chi co tung ay mau. Bao dung so da dung. Mac dinh, giu
                #       nguyen hanh vi cu cua hai kich ban few-shot.
                count = n_all if report_full_count else n_used
            else:
                # Không subsample: dùng dataset dạng view, KHÔNG copy dữ liệu lớp
                dset = data_manager.get_dataset(
                    np.arange(class_idx, class_idx + 1),
                    source="train",
                    mode="test",
                )
                count = len(dset)
                n_used = count
                if count == 0:
                    continue

            loader = make_loader(dset, batch_size=self.args["batch_size"], shuffle=False)
            feat_sum = None
            with torch.no_grad():
                for _, inputs, _ in loader:
                    feats = self._network.extract_vector(inputs.to(self._device))
                    feats = F.normalize(feats, p=2, dim=1)
                    part = feats.sum(dim=0).double().cpu()
                    feat_sum = part if feat_sum is None else feat_sum + part

            if feat_sum is None:
                continue

            mean_vec = feat_sum / n_used
            mean_norm = float(torch.norm(mean_vec, p=2))
            prototype = (mean_vec / (mean_norm + 1e-8)).float()
            dispersion = max(0.0, 1.0 - mean_norm)

            local_protos[class_idx] = {
                "prototype": prototype,
                "count": count,
                "dispersion": dispersion
            }
        return local_protos


