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
        class_counts = torch.zeros(self._total_classes).to(self._device)
        for _, _, targets in train_loader:
            targets = targets.to(self._device)
            class_counts += torch.bincount(targets, minlength=self._total_classes)
        
        total_samples = class_counts.sum()
        class_weights = torch.zeros(self._total_classes).to(self._device)
        for c in range(self._total_classes):
            if class_counts[c] > 0:
                class_weights[c] = total_samples / (class_counts[c] * self._total_classes)
            else:
                class_weights[c] = 1.0

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
                loss_fsp = compute_fsp_loss(features, targets, proto_matrix, T_fsp=0.5)
                
                # d. Prototype Alignment Loss
                loss_proto = compute_proto_loss(features, targets, proto_matrix)
                
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

    def compute_local_prototypes(self, data_manager, class_ids=None, max_samples_per_class=None, seed=0):
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
                data, targets, dset = data_manager.get_dataset(
                    np.arange(class_idx, class_idx + 1),
                    source="train",
                    mode="test",
                    ret_data=True,
                )
                if len(data) == 0:
                    continue
                if len(data) > max_samples_per_class:
                    selected = rng.choice(len(data), size=max_samples_per_class, replace=False)
                    data = data[selected] if isinstance(data, np.ndarray) else [data[int(i)] for i in selected]
                    targets = targets[selected] if isinstance(targets, np.ndarray) else [targets[int(i)] for i in selected]
                    from utils.data_manager import DummyDataset
                    from torchvision import transforms
                    trsf = transforms.Compose([*data_manager._test_trsf, *data_manager._common_trsf])
                    dset = DummyDataset(data, targets, trsf, data_manager.use_path)
                count = len(data)
            else:
                # Không subsample: dùng dataset dạng view, KHÔNG copy dữ liệu lớp
                dset = data_manager.get_dataset(
                    np.arange(class_idx, class_idx + 1),
                    source="train",
                    mode="test",
                )
                count = len(dset)
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

            mean_vec = feat_sum / count
            mean_norm = float(torch.norm(mean_vec, p=2))
            prototype = (mean_vec / (mean_norm + 1e-8)).float()
            dispersion = max(0.0, 1.0 - mean_norm)

            local_protos[class_idx] = {
                "prototype": prototype,
                "count": count,
                "dispersion": dispersion
            }
        return local_protos


