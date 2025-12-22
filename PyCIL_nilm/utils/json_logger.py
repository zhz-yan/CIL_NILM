# utils/json_logger.py

import json
import time
import platform
import sys
import numpy as np
import torch


def _to_jsonable(x, max_ndarray_elems: int = 1024):
    # 基础类型/None 直接返回
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    # numpy 标量
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    # numpy 数组：小数组转 list；大的只记录形状与 dtype，避免把整块数据写进 json
    if isinstance(x, np.ndarray):
        if x.size <= max_ndarray_elems:
            return x.tolist()
        return {"__ndarray__": True, "shape": list(x.shape), "dtype": str(x.dtype)}
    # torch 相关
    if isinstance(x, torch.device):
        return str(x)  # e.g. 'cuda:0' / 'cpu'
    if isinstance(x, torch.dtype):
        return str(x)  # e.g. 'torch.float32'
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
        return _to_jsonable(x, max_ndarray_elems)
    # 容器类型
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(i, max_ndarray_elems) for i in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v, max_ndarray_elems) for k, v in x.items()}
    # 其它可转字符串的对象（如 Path、Enum、datetime 等）
    try:
        return str(x)
    except Exception:
        return repr(x)

class ResultsLogger:
    def __init__(self, config: dict, save_path: str):
        self.save_path = save_path
        # 这里就先转一次，避免后面 save 出问题
        self.payload = {
            "config": _to_jsonable(config),
            "env": _to_jsonable({
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            }),
            "tasks": [],
            "summary": {},
            "run_wall_time_sec": 0.0,
        }
        self._run_start = None
        self._task_start = None

    def start_run(self, extra: dict = None):
        self._run_start = time.time()
        if extra:
            self.payload.setdefault("run_extra", {})
            self.payload["run_extra"].update(_to_jsonable(extra))

    def start_task(self, task_id: int, known_classes: int, total_classes: int, extra: dict = None):
        self._task_start = time.time()
        self.payload["tasks"].append({
            "task_id": int(task_id),
            "known_classes": int(known_classes),
            "total_classes": int(total_classes),
            "time_sec": 0.0,
            "metrics": {},
            "per_class": {},
            "groups": {},
            "model": {},
        })
        if extra:
            self.payload["tasks"][-1]["extra"] = _to_jsonable(extra)

    def end_task(self, y_true, y_pred, model=None, include_confusion=False):
        import sklearn.metrics as skm
        task = self.payload["tasks"][-1]
        task["time_sec"] = float(time.time() - self._task_start)

        # —— 基础指标
        acc = skm.accuracy_score(y_true, y_pred)
        prc = skm.precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = skm.recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1m = skm.f1_score(y_true, y_pred, average="macro", zero_division=0)
        task["metrics"] = {
            "accuracy": float(acc),
            "precision_macro": float(prc),
            "recall_macro": float(rec),
            "f1_macro": float(f1m),
        }

        # —— 逐类
        rep = skm.classification_report(y_true, y_pred, digits=4, output_dict=True, zero_division=0)
        # 去掉 'accuracy'/'macro avg'/'weighted avg'，只保留数字类
        per_class = {str(k): v for k, v in rep.items() if str(k).isdigit()}
        # 转基础类型
        task["per_class"] = _to_jsonable(per_class)

        # —— old/new 分组（按当前 task 的 known/total）
        known = int(task["known_classes"])
        total = int(task["total_classes"])
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        old_mask = (y_true < known)
        new_mask = (y_true >= known) & (y_true < total)
        def _macro_f1(mask):
            if mask.sum() == 0:
                return float("nan")
            return float(skm.f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0))

        task["groups"] = {
            "old": {
                "support": int(old_mask.sum()),
                "f1_macro": _macro_f1(old_mask),
                "accuracy": float(skm.accuracy_score(y_true[old_mask], y_pred[old_mask]) if old_mask.any() else float("nan")),
            },
            "new": {
                "support": int(new_mask.sum()),
                "f1_macro": _macro_f1(new_mask),
                "accuracy": float(skm.accuracy_score(y_true[new_mask], y_pred[new_mask]) if new_mask.any() else float("nan")),
            },
        }

        if include_confusion:
            cm = skm.confusion_matrix(y_true, y_pred, labels=list(range(total)))
            task["confusion_matrix"] = _to_jsonable(cm)

        # —— 模型大小/参数量
        if model is not None:
            try:
                n_params = sum(p.numel() for p in model.parameters())
                n_bytes = sum(p.element_size() * p.nelement() for p in model.parameters())
                task["model"] = {
                    "num_parameters": int(n_params),
                    "size_mb_params_only": float(n_bytes / (1024 ** 2)),
                }
            except Exception as e:
                task["model"] = {"error": str(e)}

    def finish_run(self):
        self.payload["run_wall_time_sec"] = float(time.time() - self._run_start if self._run_start else 0.0)
        # 可在这里根据 self.payload["tasks"] 计算整体 forgetting / 平均精度等汇总
        # 留空或根据你已有计算结果再 set 也行

    def save(self):
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(self.payload), f, ensure_ascii=False, indent=2)
