import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from models.base import BaseLearner
from utils.inc_net import CosineIncrementalNet, FOSTERNet, IL2ANet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy

EPSILON = 1e-8


class IL2A(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = IL2ANet(args, False)
        self._protos = []
        self._covs = []


    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network,"module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network
        self.save_checkpoint("{}_{}_{}".format(self.args["model_name"],self.args["init_cls"],self.args["increment"]))
    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1

        task_size = self.data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + task_size
        self._network.update_fc(self._known_classes,self._total_classes,int((task_size-1)*task_size/2))
        self._network_module_ptr = self._network
        logging.info(
            'Learning on {}-{}'.format(self._known_classes, self._total_classes))

        
        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(
            count_parameters(self._network, True)))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=0)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module


    def _train(self, train_loader, test_loader):
        
        resume = False
        if self._cur_task in []:
            self._network.load_state_dict(torch.load("{}_{}_{}_{}.pkl".format(self.args["model_name"],self.args["init_cls"],self.args["increment"],self._cur_task))["model_state_dict"])
            resume = True
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        if not resume:
            self._epoch_num = self.args["epochs"]
            optimizer = torch.optim.Adam(self._network.parameters(), lr=self.args["lr"], weight_decay=self.args["weight_decay"])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args["step_size"], gamma=self.args["gamma"])
            self._train_function(train_loader, test_loader, optimizer, scheduler)
        self._build_protos()
            
        
    def _build_protos(self):
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=0)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                self._protos.append(class_mean)
                cov = np.cov(vectors.T)
                self._covs.append(cov)

    def _train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._epoch_num))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            losses_clf, losses_fkd, losses_proto = 0., 0., 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                targets = targets.to(torch.int64)
                inputs, targets = inputs.to(
                    self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                inputs,targets = self._class_aug(inputs,targets)
                logits, loss_clf, loss_fkd, loss_proto = self._compute_il2a_loss(inputs,targets)
                loss = loss_clf + loss_fkd + loss_proto
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_clf += loss_clf.item()
                losses_fkd += loss_fkd.item()
                losses_proto += loss_proto.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct)*100 / total, decimals=2)
            if epoch % 5 != 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fkd {:.3f}, Loss_proto {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), losses_clf/len(train_loader), losses_fkd/len(train_loader), losses_proto/len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fkd {:.3f}, Loss_proto {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), losses_clf/len(train_loader), losses_fkd/len(train_loader), losses_proto/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
            logging.info(info)

    def _compute_il2a_loss(self,inputs, targets):
        logits = self._network(inputs)["logits"]
        loss_clf = F.cross_entropy(logits / self.args["temp"], targets)

        if self._cur_task == 0:
            return logits, loss_clf, torch.tensor(0., device=logits.device), torch.tensor(0., device=logits.device)

        features = self._network_module_ptr.extract_vector(inputs)
        features_old = self.old_network_module_ptr.extract_vector(inputs)
        loss_fkd = self.args["lambda_fkd"] * torch.dist(features, features_old, 2)

        # ---- 用当前 batch 大小，且直接用 torch.randint，避免 numpy->torch 的 dtype/device 问题
        N = inputs.size(0)
        idx = torch.randint(low=0, high=self._known_classes, size=(N,), device=self._device)

        proto_features = torch.as_tensor(np.array(self._protos), dtype=torch.float32, device=self._device)[idx]
        proto_targets = idx.long()  # 关键：int64

        proto_logits = self._network_module_ptr.fc(proto_features)["logits"][:, :self._total_classes]
        proto_logits = self._semantic_aug(proto_logits, proto_targets, self.args["ratio"])

        loss_proto = self.args["lambda_proto"] * F.cross_entropy(proto_logits / self.args["temp"], proto_targets)
        return logits, loss_clf, loss_fkd, loss_proto
        
    
    def _semantic_aug(self,proto_logits,proto_targets,ratio):
        # weight_fc = self._network_module_ptr.fc.weight.data[:self._total_classes] # don't use it ! data is not involved in back propagation
        weight_fc = self._network_module_ptr.fc.weight[:self._total_classes]  # [C,D]
        N = proto_targets.size(0)  # ← 当前 batch 的实际大小
        C, D = weight_fc.shape

        N_weight = weight_fc.expand(N, C, D)  # [N,C,D]
        # 关键：proto_targets 必须是 long，并且扩展到 [N,C,D] 作为 gather 的 index
        index = proto_targets.view(N, 1, 1).expand(N, C, D)  # long
        N_target_weight = torch.gather(N_weight, 1, index)  # [N,C,D]

        N_v = N_weight - N_target_weight
        # 如果 self._covs 仍然是 numpy，转换到 torch 并放到设备上
        N_cov = torch.as_tensor(np.array(self._covs), dtype=torch.float32, device=self._device)[
            proto_targets]  # [N,D,D]

        # [N,C,D] @ [N,D,D] @ [N,D,C] → 对角线得到 [N,C]
        proto_logits = proto_logits + ratio / 2 * torch.diagonal(N_v @ N_cov @ N_v.transpose(1, 2), dim1=1, dim2=2)
        return proto_logits




    def _class_aug(self,inputs,targets,alpha=20., mix_time=4):
        
        mixup_inputs = []
        mixup_targets = []
        for _ in range(mix_time):
            index = torch.randperm(inputs.shape[0])
            perm_inputs = inputs[index]
            perm_targets = targets[index]
            mask = perm_targets!= targets

            select_inputs = inputs[mask]
            select_targets = targets[mask]
            perm_inputs = perm_inputs[mask]
            perm_targets = perm_targets[mask]

            lams = np.random.beta(alpha,alpha,sum(mask))
            lams = np.where((lams<0.4)|(lams>0.6),0.5,lams)
            lams = torch.from_numpy(lams).to(self._device)[:,None,None,None].float()


            mixup_inputs.append(lams*select_inputs+(1-lams)*perm_inputs)
            mixup_targets.append(self._map_targets(select_targets,perm_targets))
        mixup_inputs = torch.cat(mixup_inputs,dim=0)
        mixup_targets = torch.cat(mixup_targets,dim=0)
        
        inputs = torch.cat([inputs,mixup_inputs],dim=0)
        targets = torch.cat([targets,mixup_targets],dim=0)
        return inputs,targets
    
    def _map_targets(self,select_targets,perm_targets):
        assert (select_targets != perm_targets).all()
        large_targets = torch.max(select_targets,perm_targets)-self._known_classes
        small_targets = torch.min(select_targets,perm_targets)-self._known_classes

        mixup_targets = (large_targets*(large_targets-1)/2  + small_targets + self._total_classes).long()
        return mixup_targets
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            targets = targets.to(torch.int64)
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"][:,:self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            targets = targets.to(torch.int64)
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"][:,:self._total_classes]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  
    
    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, '_class_means'):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        elif hasattr(self, '_protos'):
            y_pred, y_true = self._eval_nme(self.test_loader, self._protos/np.linalg.norm(self._protos,axis=1)[:,None])
            nme_accy = self._evaluate(y_pred, y_true)            
        else:
            nme_accy = None

        return cnn_accy, nme_accy
