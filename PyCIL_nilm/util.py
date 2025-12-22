import torch
from typing import Optional, Iterable

def _tensor_nbytes(t: torch.Tensor) -> int:
    return 0 if t is None else t.numel() * t.element_size()

def model_nbytes(model: torch.nn.Module, include_buffers: bool = True) -> int:
    """参数 + （可选）buffers 的总字节数。"""
    total = 0
    for p in model.parameters(recurse=True):
        total += _tensor_nbytes(p)
    if include_buffers:
        for b in model.buffers(recurse=True):
            total += _tensor_nbytes(b)
    return total

def optimizer_nbytes(optimizer: Optional[torch.optim.Optimizer]) -> int:
    """优化器状态（动量等）的总字节数。SGD 可能较小，Adam 会接近 2~3× 参数量。"""
    if optimizer is None:
        return 0
    total = 0
    for st in optimizer.state.values():
        if isinstance(st, dict):
            for v in st.values():
                if torch.is_tensor(v):
                    total += _tensor_nbytes(v)
    return total

def bytes_to_mb(nbytes: int) -> float:
    return nbytes / (1024.0 ** 2)

def replay_memory_nbytes(features: Optional[torch.Tensor] = None,
                         labels: Optional[torch.Tensor] = None) -> int:
    """用于非 ACIL 方法：重放库占用。ACIL 直接传 None, None -> 0。"""
    total = 0
    if features is not None:
        total += _tensor_nbytes(features)
    if labels is not None:
        total += _tensor_nbytes(labels)
    return total

def print_model_memory_report(tag: str,
                              model: torch.nn.Module,
                              optimizer: Optional[torch.optim.Optimizer] = None,
                              mem_features: Optional[torch.Tensor] = None,
                              mem_labels: Optional[torch.Tensor] = None):
    m_bytes = model_nbytes(model, include_buffers=True)
    o_bytes = optimizer_nbytes(optimizer)
    r_bytes = replay_memory_nbytes(mem_features, mem_labels)
    total_bytes = m_bytes + o_bytes + r_bytes
    print(f"[{tag}] model={bytes_to_mb(m_bytes):.2f} MB, "
          f"optim={bytes_to_mb(o_bytes):.2f} MB, "
          f"replay={bytes_to_mb(r_bytes):.2f} MB, "
          f"total={bytes_to_mb(total_bytes):.2f} MB")