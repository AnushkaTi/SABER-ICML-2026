import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import default_data_collator

# Reuse encoder-decoder trainer and tokenizer/model setup
from .t5_continual import T5ContinualLearner
# Reuse SuperNI dataset loader and mappings
from .superNI_decoder import load_task_dataset, TASK_TO_LABELS


class _SuperNIS2SWrapper(Dataset):
    def __init__(self, enc_tok, split_ds, target_len_cap: int, max_source_len: int):
        self.tok = enc_tok
        self.ds = split_ds
        self.tgt_cap = int(target_len_cap)
        self.src_cap = int(max_source_len)

    def __len__(self):
        return len(self.ds)

    def _build_source(self, ex: Dict) -> str:
        definition = ex.get("definition", "") or ""
        inp = ex.get("input", "") or ""
        if definition:
            return f"{definition}\n{inp}".strip()
        return inp

    def _build_target(self, ex: Dict, label_texts: Tuple[str, ...]) -> str:
        if "target" in ex:
            t = ex.get("target", "")
            if isinstance(t, list):
                return t[0] if len(t) > 0 else ""
            return t
        # classification: ex["label"] is an int index
        li = int(ex.get("label", 0))
        if 0 <= li < len(label_texts):
            return label_texts[li]
        return str(li)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.ds[idx]
        label_texts = tuple(ex.get("_label_texts", ()))  # injected by wrapper builder
        src = self._build_source(ex)
        tgt = self._build_target(ex, label_texts)
        s = self.tok(src, truncation=True, max_length=self.src_cap, padding=False)
        t = self.tok(tgt, truncation=True, max_length=self.tgt_cap, padding=False)
        return {
            "source_ids": torch.tensor(s["input_ids"], dtype=torch.long),
            "source_mask": torch.tensor(s["attention_mask"], dtype=torch.long),
            "target_ids": torch.tensor(t["input_ids"], dtype=torch.long),
            "target_mask": torch.tensor(t["attention_mask"], dtype=torch.long),
        }


class SuperNI_T5ContinualLearner(T5ContinualLearner):
    """
    Encoder-decoder continual learner with SuperNI data support (no BigP).
    - Reuses T5ContinualLearner training/selection/projection logic.
    - Overrides get_tasks_data_dict to load SuperNI JSON tasks (task name: 'superni:<dir>').
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
        # inject label_texts for classification reconstruction
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
                # fallback to base behavior for non-SuperNI tasks
                return super().get_tasks_data_dict(memory_perc=memory_perc)
            dataset, label_column, eval_split, canonical = load_task_dataset(
                task, cache_dir=self.cache_dir, data_root=self._data_root, superni_dir=self._superni_dir
            )
            # determine label texts for classification; empty tuple for generation
            label_texts = TASK_TO_LABELS.get(task, ())
            if not label_texts and label_column is None:
                label_texts = ()
            # estimate target len cap from train split
            tgt_cap = self._estimate_target_len(dataset["train"])
            self.task_to_target_len[task] = tgt_cap
            # build wrapped datasets
            train_wrapped = self._wrap_split(dataset["train"], label_texts, tgt_cap)
            val_name = eval_split
            eval_wrapped = self._wrap_split(dataset[val_name], label_texts, tgt_cap) if val_name in dataset else train_wrapped
            test_wrapped = self._wrap_split(dataset["test"], label_texts, tgt_cap) if "test" in dataset else eval_wrapped
            # dataloaders
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

