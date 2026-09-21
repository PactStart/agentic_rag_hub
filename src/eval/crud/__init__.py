"""CRUD-RAG 四任务评测轨（对齐官方 Create/Read/Update/Delete + 指标）。"""

from src.eval.crud.dataset import TASK_KEYS, load_task_samples
from src.eval.crud.runner import run_crud_eval

__all__ = ["TASK_KEYS", "load_task_samples", "run_crud_eval"]
