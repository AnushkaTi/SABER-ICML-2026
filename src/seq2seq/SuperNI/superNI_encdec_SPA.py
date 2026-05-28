from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import default_data_collator

from .t5_contiual_BigP import T5ContinualLearner as T5BigPContinualLearner
from .superNI_decoder import load_task_dataset, TASK_TO_LABELS
from .superNI_t5_continual import _SuperNIS2SWrapper


class SuperNI_T5ContinualBigPLearner(T5BigPContinualLearner):
    """
    Encoder-decoder continual learner (BigP) with SuperNI data support.
    - Reuses BigP trainer logic; overrides data loading to support SuperNI JSON.
    """
    def __init__(self, *args, superni_dir: str = "./SuperNI", data_root: str = "./datasets/src/data", **kwargs):
        self._superni_dir = superni_dir
        self._data_root = data_root
        super().__init__(*args, **kwargs)

    def _estimate_target_len(self, ds, default_cap: int = 32, max_cap: int = 64) -> int:
        try:
            n = min(len(ds), 512)
            if n == 0:
                return default_cap
            lens = []
            for i in range(n):
                tgt = ds[i].get("target", "")
                if isinstance(tgt, list):
                    tgt = tgt[0] if len(tgt) > 0 else ""
                ids = self.tokenizer(str(tgt), add_special_tokens=False)["input_ids"]
                lens.append(len(ids))
            if not lens:
                return default_cap
            observed = max(lens)
            return int(min(max(observed, 4), max_cap))
        except Exception:
            return int(default_cap)

    def _wrap_split(self, split_ds, label_texts: Tuple[str, ...], target_len_cap: int) -> Dataset:
        def _add_labels(ex):
            ex["_label_texts"] = list(label_texts)
            return ex
        split_ds = split_ds.map(_add_labels)
        return _SuperNIS2SWrapper(self.tokenizer, split_ds, target_len_cap, self.seq_len)

    def get_tasks_data_dict(self, memory_perc=0):
        tasks_data_dict = {}
        for task in self.task_list:
            tasks_data_dict[task] = {}
            if not task.startswith("superni:"):
                return super().get_tasks_data_dict(memory_perc=memory_perc)
            dataset, label_column, eval_split, canonical = load_task_dataset(
                task, cache_dir=self.cache_dir, data_root=self._data_root, superni_dir=self._superni_dir
            )
            label_texts = TASK_TO_LABELS.get(task, ())
            if not label_texts and label_column is None:
                label_texts = ()
            tgt_cap = self._estimate_target_len(dataset["train"])
            self.task_to_target_len[task] = tgt_cap
            train_wrapped = self._wrap_split(dataset["train"], label_texts, tgt_cap)
            val_name = eval_split
            eval_wrapped = self._wrap_split(dataset[val_name], label_texts, tgt_cap) if val_name in dataset else train_wrapped
            test_wrapped = self._wrap_split(dataset["test"], label_texts, tgt_cap) if "test" in dataset else eval_wrapped
            tasks_data_dict[task]["train"] = DataLoader(
                train_wrapped, batch_size=self.batch_size, shuffle=True, collate_fn=default_data_collator
            )
            if self.get_test_subset:
                tasks_data_dict[task]["val"] = DataLoader(
                    eval_wrapped, batch_size=self.batch_size, shuffle=False, collate_fn=default_data_collator
                )
                tasks_data_dict[task]["test"] = DataLoader(
                    test_wrapped, batch_size=self.batch_size, shuffle=False, collate_fn=default_data_collator
                )
            else:
                tasks_data_dict[task]["val"] = DataLoader(
                    eval_wrapped, batch_size=self.batch_size, shuffle=False, collate_fn=default_data_collator
                )
        return tasks_data_dict

