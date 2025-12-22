import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import torch

class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, aug=1,
                 data_dir=None, train_ratio=0.8, no_transform=True):
        self.dataset_name = dataset_name
        self.aug = aug
        self.data_dir = data_dir
        self.train_ratio = train_ratio
        self.no_transform = no_transform
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]
    
    def get_accumulate_tasksize(self,task):
        return sum(self._increments[:task+1])
    
    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path,self.aug if source == "train" and mode == "train" else 1)
        else:
            return DummyDataset(data, targets, trsf, self.use_path,self.aug if source == "train" and mode == "train" else 1)

        
    def get_finetune_dataset(self,known_classes,total_classes,source,mode,appendent,type="ratio"):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))
        val_data = []
        val_targets = []

        old_num_tot = 0
        appendent_data, appendent_targets = appendent

        for idx in range(0, known_classes):
            append_data, append_targets = self._select(appendent_data, appendent_targets,
                                                       low_range=idx, high_range=idx+1)
            num=len(append_data)
            if num == 0:
                continue
            old_num_tot += num
            val_data.append(append_data)
            val_targets.append(append_targets)
        if type == "ratio":
            new_num_tot = int(old_num_tot*(total_classes-known_classes)/known_classes)
        elif type == "same":
            new_num_tot = old_num_tot
        else:
            assert 0, "not implemented yet"
        new_num_average = int(new_num_tot/(total_classes-known_classes))
        for idx in range(known_classes,total_classes):
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            val_indx = np.random.choice(len(class_data),new_num_average, replace=False)
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
        val_data=np.concatenate(val_data)
        val_targets = np.concatenate(val_targets)
        return DummyDataset(val_data, val_targets, trsf, self.use_path, self.aug if source == "train" and mode == "train" else 1)

    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=True
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path, self.aug
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name)

        if dataset_name.lower() not in ['plaid', 'whited', 'cooll', 'case_2']:
            # 原有数据集路径
            idata.download_data()
            self._train_data, self._train_targets = idata.train_data, idata.train_targets
            self._test_data, self._test_targets = idata.test_data, idata.test_targets
            self.use_path = idata.use_path
            self._train_trsf = idata.train_trsf
            self._test_trsf = idata.test_trsf
            self._common_trsf = idata.common_trsf
            order = [i for i in range(len(np.unique(self._train_targets)))]
            if shuffle:
                np.random.seed(seed)
                order = np.random.permutation(len(order)).tolist()
            else:
                order = idata.class_order
            self._class_order = order
            logging.info(self._class_order)
            self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
            self._test_targets = _map_new_class_index(self._test_targets, self._class_order)
            return

        # === npy 模式 ===
        assert self.data_dir is not None, "For dataset_name='npy', please provide data_dir."

        X = np.load(f"{self.data_dir}/X.npy", allow_pickle=True)
        y = np.load(f"{self.data_dir}/y.npy", allow_pickle=True).ravel()
        enc = LabelEncoder()

        y = enc.fit_transform(y)
        # 如果 X 是 (N, H, W, C) 或 (N, ...)，保持原样；DummyDataset 将直接返回 numpy
        # 切分 train/test（如果你已有独立 test，可自行改为直接加载）
        N = len(y)
        rng = np.random.RandomState(seed)
        idx = np.arange(N)
        rng.shuffle(idx)
        n_tr = int(self.train_ratio * N)
        tr_idx, te_idx = idx[:n_tr], idx[n_tr:]

        self._train_data = X[tr_idx]
        self._train_targets = y[tr_idx]
        self._test_data = X[te_idx]
        self._test_targets = y[te_idx]

        # 不使用路径
        self.use_path = False

        # 禁用一切 transforms
        if self.no_transform:
            self._train_trsf = []
            self._test_trsf = []
            self._common_trsf = []
        else:
            # 如需后续自行加 transform，可在这里配置
            self._train_trsf = []
            self._test_trsf = []
            self._common_trsf = []

        # 类顺序
        classes = np.unique(self._train_targets)
        order = list(classes.astype(int))  # 以出现类的自然顺序
        if shuffle:
            rng = np.random.RandomState(seed)
            order = rng.permutation(order).tolist()
        self._class_order = order
        logging.info(self._class_order)

        # 标签按新顺序重映射到 0..C-1
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        
        if isinstance(x,np.ndarray):
            x_return = x[idxes]
        else:
            x_return = []
            for id in idxes:
                x_return.append(x[id])
        return x_return, y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf=None, use_path=False, aug=1):
        assert len(images) == len(labels), "Data size error!"
        self.aug = aug
        self.images = images
        self.labels = labels
        self.trsf = trsf  # 允许为 None
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def _apply_trsf(self, x):
        if self.trsf is None:
            return x
        return self.trsf(x)

    def __getitem__(self, idx):
        if self.use_path:
            # 路径模式：按需加载为 PIL 后再 trsf（但我们 npy 模式不会走这里）
            from PIL import Image
            img = pil_loader(self.images[idx])
            if self.trsf is not None:
                img = self.trsf(img)
        else:
            # 直接使用 numpy，不做 PIL 转换
            img = self.images[idx]
            if self.trsf is not None:
                img = self.trsf(img)

        label = self.labels[idx]

        if self.aug == 1:
            return idx, img, label
        else:
            # 多视角增广：这里如果 trsf=None，则只是重复同一份 img
            imgs = [self._apply_trsf(img) for _ in range(self.aug)]
            return idx, *imgs, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name in ["plaid", 'whited', 'cooll', 'case_2']:
        return iNPY()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class iNPY(object):

    def __init__(self):
        self.use_path = False
        self.class_order = None

        self.train_data = None
        self.train_targets = None
        self.test_data = None
        self.test_targets = None

        self.train_trsf = []
        self.test_trsf = []
        self.common_trsf = []

    def download_data(self):
        pass
