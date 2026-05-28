import os
import math
import random
import json
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import time
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,
)

# --------------------------
# Task and label definitions
# --------------------------

GLUE_DATASETS = [
    "cola", "sst2", "mrpc", "qqp", "stsb", "mnli",
    "mnli_mismatched", "mnli_matched", "qnli", "rte", "wnli", "ax"
]

SUPERGLUE_DATASETS = [
    "copa", "boolq", "wic", "wsc", "cb", "record",
    "multirc", "rte_superglue", "wsc_bool"
]

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),

    "boolq": ("passage", "question"),
    "copa": ("choice1", "choice2", "premise", "question"),
    "wic": ("start1", "end1", "sentence1", "start2", "end2", "sentence2", "word"),
    "wsc": ("span1_text", "span1_index", "span2_text", "span2_index", "text"),
    "wsc_bool": ("span1_text", "span1_index", "span2_text", "span2_index", "text"),
    "cb": ("premise", "hypothesis"),
    "record": ("passage", "query", "entities"),
    "multirc": ("question", "answer", "paragraph"),
    "rte_superglue": ("premise", "hypothesis"),

    "scicite": ("sectionName", "string"),
    "imdb": ("text", None),

    "ag_news": ("text", None),
    "yelp_review_full": ("text", None),
    "yahoo_answers_topics": ("question_content", "best_answer"),
    "dbpedia_14": ("title", "content"),

    "ag": ("content", None),
    "yelp": ("content", None),
    "yahoo": ("content", None),
    "dbpedia": ("content", None),
    "amazon": ("content", None),
}

TASK_TO_LABELS = {
    "cola": ("not_acceptable", "acceptable"),
    "mnli": ("entailment", "neutral", "contradiction"),
    # "mnli": ("e", "n", "c"),
    "mnli-mm": ("entailment", "neutral", "contradiction"),
    "mrpc": ("not_equivalent", "equivalent"),
    "qnli": ("entailment", "not_entailment"),
    "qqp": ("not_duplicate", "duplicate"),
    "rte": ("true", "false"),
    "sst2": ("negative", "positive"),
    "stsb": (),
    "wnli": (),

    "boolq": ("false", "true"),
    "copa": ("choice1", "choice2"),
    "wic": ("false", "true"),
    "wsc_bool": ("false", "true"),
    "cb": ("entailment", "contradiction", "neutral"),
    "multirc": ("false", "true"),
    "rte_superglue": ("entailment", "not_entailment"),

    "scicite": (),
    "imdb": ("negative", "positive"),

    "ag_news": ("world", "sports", "business", "science"),
    "yelp_review_full": ("terrible", "bad", "middle", "good", "wonderful"),
    "yahoo_answers_topics": (
        "society and culture", "science", "health", "education and reference",
        "computers and internet", "sports", "business", "entertainment and music",
        "family and relationships", "politics and government"
    ),
    "dbpedia_14": (
        "company", "educationalinstitution", "artist", "athlete", "officeholder",
        "meanoftransportation", "building", "naturalplace", "village", "animal",
        "plant", "album", "film", "writtenwork"
    ),

    "ag": ("world", "sports", "business", "science"),
    "yelp": ("terrible", "bad", "middle", "good", "wonderful"),
    "yahoo": (
        "society and culture", "science", "health", "education and reference",
        "computers and internet", "sports", "business", "entertainment and music",
        "family and relationships", "politics and government"
    ),
    "dbpedia": (
        "company", "educationalinstitution", "artist", "athlete", "officeholder",
        "meanoftransportation", "building", "naturalplace", "village", "animal",
        "plant", "album", "film", "writtenwork"
    ),
    "amazon": ("terrible", "bad", "middle", "good", "wonderful"),
}

# Map task names to fixed test .npy files (if available)
TEST_FIXED_FILES = {
    "ag_news": "ag_news.npy",
    "amazon": "amazon.npy",
    "dbpedia_14": "dbpedia_14.npy",
    "qqp": "glue_qqp.npy",
    "imdb": "imdb.npy",
    "mnli": "mnli_test.npy",
    "mnli_matched": "mnli_test.npy",
    "mnli_mismatched": "mnli_test.npy",
    "qnli": "qnli_test.npy",  # not in your list, but supported
    "sst2": "sst2.npy",
    "boolq": "super_glue_boolq.npy",
    "wic": "super_glue_wic.npy",
    "multirc": "super_glue_multirc.npy",
    "yahoo_answers_topics": "yahoo_answers_topics.npy",
    "yelp_review_full": "yelp_review_full.npy",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------
# Dataset helpers
# --------------------------

def load_task_dataset(task_name: str,
                      cache_dir: str = None,
                      data_root: str = "./datasets/src/data"):
    """
    Load GLUE / SuperGLUE / other datasets, including special cases like amazon.
    """

    # --------------------------
    # Special non-HF / custom tasks
    # --------------------------
    if task_name == "amazon":
        def _load_amazon_split(split_name: str) -> Dataset:
            csv_path = os.path.join(data_root, "amazon", f"{split_name}.csv")
            df = pd.read_csv(csv_path, header=None)
            df = df.rename(columns={0: "label", 1: "title", 2: "content"})
            df["label"] = df["label"] - 1
            return Dataset.from_pandas(df)

        dataset = DatasetDict({
            "train": _load_amazon_split("train"),
            "validation": _load_amazon_split("test"),
            "test": _load_amazon_split("test"),
        })

        label_column = "label"
        eval_split = "validation"
        canonical_for_prompts = "amazon"
        return dataset, label_column, eval_split, canonical_for_prompts

    # --------------------------
    # GLUE tasks
    # --------------------------
    if task_name in GLUE_DATASETS:
        if task_name == "ax":
            raise ValueError("GLUE 'ax' has no training labels; skip it.")

        if task_name in ["mnli", "mnli_matched", "mnli_mismatched"]:
            train_ds = load_dataset(
                "LysandreJik/glue-mnli-train",
                split="train",
                cache_dir=cache_dir,
            )
            glue_mnli = load_dataset("glue", "mnli", cache_dir=cache_dir)
            dataset = DatasetDict({
                "train": train_ds,
                "validation_matched": glue_mnli["validation_matched"],
                "validation_mismatched": glue_mnli["validation_mismatched"],
            })
            if task_name == "mnli_mismatched":
                eval_split = "validation_mismatched"
            else:
                eval_split = "validation_matched"
            canonical_for_prompts = "mnli"
            label_column = "label"
            return dataset, label_column, eval_split, canonical_for_prompts

        if task_name == "qnli":
            train_ds = load_dataset(
                "SetFit/qnli",
                split="train",
                cache_dir=cache_dir,
            )
            glue_qnli = load_dataset("glue", "qnli", cache_dir=cache_dir)
            dataset = DatasetDict({
                "train": train_ds,
                "validation": glue_qnli["validation"],
            })
            label_column = "label"
            eval_split = "validation"
            canonical_for_prompts = "qnli"
            return dataset, label_column, eval_split, canonical_for_prompts

        subset = task_name
        dataset = load_dataset("glue", subset, cache_dir=cache_dir)
        eval_split = "validation"
        canonical_for_prompts = task_name
        label_column = "label"
        return dataset, label_column, eval_split, canonical_for_prompts

    # --------------------------
    # SuperGLUE tasks
    # --------------------------
    if task_name in SUPERGLUE_DATASETS:
        subset_map = {
            "rte_superglue": "rte",
            "wsc_bool": "wsc",
        }
        subset = subset_map.get(task_name, task_name)

        if task_name == "stsb":
            dataset = load_dataset(
                "stsb_multi_mt",
                name="en",
                cache_dir=cache_dir,
            )
            dataset = DatasetDict({
                "train": dataset["train"],
                "validation": dataset["dev"],
                "test": dataset["test"] if "test" in dataset else dataset["dev"],
            })
            label_column = "label"
            eval_split = "validation"
            canonical_for_prompts = "stsb"
            return dataset, label_column, eval_split, canonical_for_prompts

        dataset = load_dataset("super_glue", subset, cache_dir=cache_dir)
        eval_split = "validation"
        canonical_for_prompts = task_name
        label_column = "label"
        return dataset, label_column, eval_split, canonical_for_prompts

    if task_name == "yahoo_answers_topics":
            dataset = load_dataset(task_name, cache_dir=cache_dir)
            label_column = "topic"
            eval_split = "test" if "test" in dataset else ("validation" if "validation" in dataset else None)
            if eval_split is None:
                raise ValueError("yahoo_answers_topics has no test/validation split to evaluate on.")
            canonical_for_prompts = "yahoo_answers_topics"
            return dataset, label_column, eval_split, canonical_for_prompts

    # --------------------------
    # Other datasets
    # --------------------------
    dataset = load_dataset(task_name, cache_dir=cache_dir)
    if "label" in dataset["train"].column_names:
        label_column = "label"
    else:
        raise ValueError(f"Could not infer label column for task {task_name}")

    eval_split = "test" if "test" in dataset else "validation"
    canonical_for_prompts = task_name
    return dataset, label_column, eval_split, canonical_for_prompts


def balanced_subsample(split: Dataset, label_column: str, k: int, seed: int = 42) -> Dataset:
    random.seed(seed)
    labels = split[label_column]
    unique_labels = sorted(set(labels))
    label_to_indices = {lbl: [] for lbl in unique_labels}
    for idx, lbl in enumerate(labels):
        label_to_indices[lbl].append(idx)
    selected = []
    for lbl, idxs in label_to_indices.items():
        random.shuffle(idxs)
        selected.extend(idxs[:k])
    selected = sorted(selected)
    return split.select(selected)


def build_prompt_from_example(example: Dict, task_name: str, keys: Tuple[str, ...]) -> str:
    if task_name in ["mnli_matched", "mnli_mismatched"]:
        prompt_task = "mnli"
    else:
        prompt_task = task_name
    keys_for_task = TASK_TO_KEYS.get(prompt_task)
    if keys_for_task is None:
        parts = []
        for k, v in example.items():
            if isinstance(v, str):
                parts.append(f"{k}: {v}")
        return "\n".join(parts) + "\nLabel:"
    parts = []
    for k in keys_for_task:
        if k is None:
            continue
        if k in example:
            parts.append(f"{k}: {example[k]}")
    if not parts and "text" in example:
        parts = [f"text: {example['text']}"]
    return "\n".join(parts) + "\nLabel:"


def build_label_token_seqs(label_texts: Tuple[str, ...], tokenizer) -> List[List[int]]:
    seqs = []
    for label in label_texts:
        ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        seqs.append(ids)
    return seqs


class ProgressiveDecoderPromptTrainer:
    def __init__(
        self,
        base_model_name: str,
        task_list: List[str],
        prefix_len: int = 10,
        max_length: int = 64,
        lr: float = 3e-2,
        batch_size: int = 8,
        k_per_class: int = 2000,
        num_epochs: int = 10,
        selection_method: str = "proj_cos",
        seed: int = 42,
        cache_dir: str = None,
        fix_test_data: bool = True,
        test_fixed_dir: str = "./datasets/test_fixed",
        data_root: str = "./datasets/src/data",
    ):
        self.base_model_name = base_model_name
        self.task_list = task_list
        self.prefix_len = prefix_len
        self.max_length = max_length
        self.lr = lr
        self.batch_size = batch_size
        self.k_per_class = k_per_class
        self.num_epochs = num_epochs
        self.selection_method = selection_method
        self.seed = seed
        self.cache_dir = cache_dir
        self.fix_test_data = fix_test_data
        self.test_fixed_dir = test_fixed_dir
        self.data_root = data_root

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(DEVICE)

        self.hidden_size = self.model.get_input_embeddings().embedding_dim
        self.current_prompt: nn.Parameter = None
        self.previous_prompts = torch.empty(0, self.hidden_size, device=DEVICE)

        self.task_meta: Dict[str, Dict] = {}
        self._prepare_task_meta()

        self.task_to_acc: Dict[str, float] = {}
        self.per_epoch_acc = {}

        self.prompt_grad_dirs: Dict[str, np.ndarray] = {}
        self.previous_prompts_updated: torch.Tensor = None
        self.task_to_svd_basis: Dict[str, np.ndarray] = {}
        self.prev_prompt_grad_dirs: Dict[str, Dict[str, Dict]] = {}
        self.reverse_phase_active: bool = False
        self.previous_prompts_param: nn.Parameter = None
        self._reverse_selected_order: List[str] = []
        self._reverse_selected_indices: List[int] = []
        self._reverse_basis_list: List[torch.Tensor] = []

        self.base_losses: Dict[str, List[float]] = {}
        self.self_losses: Dict[str, List[float]] = {}
        self.cross_losses: Dict[Tuple[str, str], List[float]] = {}

        self.cumulative_restrict_basis: Dict[str, np.ndarray] = {}

    def _maybe_load_fixed_test(self, task_name: str):
        if not self.fix_test_data:
            return None
        fname = TEST_FIXED_FILES.get(task_name)
        if fname is None:
            return None
        path = os.path.join(self.test_fixed_dir, fname)
        if not os.path.exists(path):
            return None
        arr = np.load(path, allow_pickle=True)
        data_dict = arr if isinstance(arr, dict) else arr.item()
        return Dataset.from_dict(data_dict)

    def _prepare_task_meta(self):
        for task_name in self.task_list:
            dataset, label_column, eval_split, canonical_for_prompts = load_task_dataset(
                task_name,
                cache_dir=self.cache_dir,
                data_root=self.data_root,
            )
            fixed_ds = self._maybe_load_fixed_test(task_name)
            if fixed_ds is not None:
                dataset = DatasetDict({**dataset, "fixed_test": fixed_ds})
                eval_split = "fixed_test"

            labels_key = task_name
            if labels_key in ["mnli_matched", "mnli_mismatched"]:
                labels_key = "mnli"

            label_texts = TASK_TO_LABELS.get(labels_key, ())
            if not label_texts:
                raise ValueError(f"No label texts defined for task {task_name} (key={labels_key}).")

            keys = TASK_TO_KEYS.get(canonical_for_prompts, ())
            if canonical_for_prompts in ["mnli", "mnli_matched", "mnli_mismatched"]:
                keys = TASK_TO_KEYS.get("mnli", ())

            label_token_seqs = build_label_token_seqs(label_texts, self.tokenizer)

            self.task_meta[task_name] = {
                "dataset": dataset,
                "label_column": label_column,
                "eval_split": eval_split,
                "canonical_for_prompts": canonical_for_prompts,
                "label_texts": label_texts,
                "label_token_seqs": label_token_seqs,
                "keys": keys,
            }

    def _init_prompt_from_text(self, label_texts: Tuple[str, ...]) -> torch.Tensor:
        init_text = f"Classify the text into one of: {', '.join(label_texts)}"
        ids = self.tokenizer(init_text, add_special_tokens=False)["input_ids"]
        ids_tensor = torch.tensor(ids, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            emb = self.model.get_input_embeddings()(ids_tensor)
        emb = emb[0]
        if emb.size(0) >= self.prefix_len:
            prompt = emb[: self.prefix_len]
        else:
            reps = math.ceil(self.prefix_len / emb.size(0))
            prompt = emb.repeat(reps, 1)[: self.prefix_len]
        return prompt.detach().clone()

    def _start_new_task_prompt(self, label_texts: Tuple[str, ...]):
        if self.current_prompt is not None:
            with torch.no_grad():
                prev = self.current_prompt.detach().to(DEVICE)
            prev.requires_grad = False
            if self.previous_prompts.numel() == 0:
                self.previous_prompts = prev
            else:
                self.previous_prompts = torch.cat([prev, self.previous_prompts], dim=0)
        init_prompt = self._init_prompt_from_text(label_texts)
        self.current_prompt = nn.Parameter(init_prompt.to(DEVICE), requires_grad=True)

    def _get_task_prompt(self) -> torch.Tensor:
        if self.reverse_phase_active and self.previous_prompts_param is not None:
            return self.previous_prompts_param
        if self.current_prompt is None:
            return self.previous_prompts
        if self.previous_prompts.numel() == 0:
            return self.current_prompt
        return torch.cat([self.previous_prompts, self.current_prompt], dim=0)

    def get_prompts_numpy(self, include_current: bool = True) -> np.ndarray:
        with torch.no_grad():
            prev = self.previous_prompts.detach().cpu()
            if include_current and self.current_prompt is not None:
                curr = self.current_prompt.detach().cpu()
                all_prompts = curr if prev.numel() == 0 else torch.cat([curr, prev], dim=0)
            else:
                all_prompts = prev
        return all_prompts.numpy()

    def _build_preprocess_fn(
        self,
        label_texts: Tuple[str, ...],
        label_column: str,
        canonical_for_prompts: str,
        keys_for_task: Tuple[str, ...],
        target_max_length: int,
    ):
        max_length = self.max_length
        tokenizer = self.tokenizer
        input_max_length = max_length - (target_max_length + 1)
        if input_max_length <= 0:
            raise ValueError("max_length too small for this label set.")
        def preprocess_function(examples):
            batch_input_ids, batch_attention_masks, batch_labels = [], [], []
            texts = []
            for i in range(len(examples[label_column])):
                ex = {k: examples[k][i] for k in examples.keys()}
                prompt_text = build_prompt_from_example(ex, canonical_for_prompts, keys_for_task)
                texts.append(prompt_text)
            targets = [label_texts[label_id] for label_id in examples[label_column]]
            model_inputs = tokenizer(texts, truncation=True, max_length=input_max_length, padding=False)
            label_encodings = tokenizer(targets, add_special_tokens=False, truncation=True, max_length=target_max_length, padding=False)
            for input_ids, attn_mask, lab_ids in zip(model_inputs["input_ids"], model_inputs["attention_mask"], label_encodings["input_ids"]):
                lab_ids = lab_ids + [tokenizer.pad_token_id]
                full_input_ids = input_ids + lab_ids
                full_attention_mask = attn_mask + [1] * len(lab_ids)
                full_labels = [-100] * len(input_ids) + lab_ids
                full_input_ids = full_input_ids[:max_length]
                full_attention_mask = full_attention_mask[:max_length]
                full_labels = full_labels[:max_length]
                pad_len = max_length - len(full_input_ids)
                if pad_len > 0:
                    full_input_ids += [tokenizer.pad_token_id] * pad_len
                    full_attention_mask += [0] * pad_len
                    full_labels += [-100] * pad_len
                batch_input_ids.append(full_input_ids)
                batch_attention_masks.append(full_attention_mask)
                batch_labels.append(full_labels)
            return {"input_ids": batch_input_ids, "attention_mask": batch_attention_masks, "labels": batch_labels}
        return preprocess_function

    def _train_step(self, batch) -> torch.Tensor:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        prefix_len_total = self._get_task_prompt().size(0)
        total_len = prefix_len_total + input_ids.size(1)
        if total_len > self.model.config.max_position_embeddings:
            raise RuntimeError(f"Train sequence too long: {total_len} > {self.model.config.max_position_embeddings}")
        bsz, seq_len = input_ids.shape
        task_prompt = self._get_task_prompt()
        prefix_len_total = task_prompt.size(0)
        with torch.no_grad():
            token_embeds = self.model.get_input_embeddings()(input_ids)
        prompt_batched = task_prompt.unsqueeze(0).expand(bsz, -1, -1)
        inputs_embeds = torch.cat([prompt_batched, token_embeds], dim=1)
        prefix_mask = torch.ones(bsz, prefix_len_total, device=DEVICE, dtype=attention_mask.dtype)
        attention_mask_ext = torch.cat([prefix_mask, attention_mask], dim=1)
        ignore_prefix = torch.full((bsz, prefix_len_total), -100, device=DEVICE, dtype=labels.dtype)
        labels_ext = torch.cat([ignore_prefix, labels], dim=1)
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask_ext, labels=labels_ext)
        return outputs.loss

    def _flatten_block(self, block: torch.Tensor) -> torch.Tensor:
        return block.reshape(-1)

    def _build_loader_for_task(self, task_name: str, split: str = "train") -> DataLoader:
        meta = self.task_meta[task_name]
        dataset = meta["dataset"]
        label_column = meta["label_column"]
        canonical_for_prompts = meta["canonical_for_prompts"]
        keys_for_task = meta["keys"]
        label_texts = meta["label_texts"]
        target_max_length = max(len(self.tokenizer(lbl, add_special_tokens=False)["input_ids"]) for lbl in label_texts)
        preprocess_fn = self._build_preprocess_fn(label_texts, label_column, canonical_for_prompts, keys_for_task, target_max_length)
        raw_split = dataset[split]
        processed = raw_split.map(preprocess_fn, batched=True, num_proc=1, remove_columns=raw_split.column_names, load_from_cache_file=False, desc=f"Tokenizing {split} for {task_name}")
        return DataLoader(processed, shuffle=False, collate_fn=default_data_collator, batch_size=self.batch_size, pin_memory=True)

    def compute_loss_distribution(self, loader: DataLoader, prompt: Optional[torch.Tensor]) -> List[float]:
        losses: List[float] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                if prompt is None:
                    loss = self._loss_without_prompt(batch)
                else:
                    loss = self._loss_with_explicit_prompt(batch, prompt)
                losses.append(float(loss.detach().cpu().item()))
        return losses

    def compute_loss_based_similarity_decoder(self, current_task: str, prev_order: List[str], train_loader_current: DataLoader, print_results: bool = True) -> Dict[str, Dict]:
        def wdist(a, b):
            try:
                from scipy.stats import wasserstein_distance
                return float(wasserstein_distance(a, b))
            except Exception:
                a_sorted = np.sort(np.array(a))
                b_sorted = np.sort(np.array(b))
                n = max(len(a_sorted), len(b_sorted))
                if n == 0:
                    return 0.0
                a_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(a_sorted)), a_sorted)
                b_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(b_sorted)), b_sorted)
                return float(np.mean(np.abs(np.cumsum(a_interp) - np.cumsum(b_interp))) / n)
        sim: Dict[str, Dict] = {}
        if current_task not in self.base_losses:
            self.base_losses[current_task] = self.compute_loss_distribution(train_loader_current, prompt=None)
        chunks = torch.split(self.previous_prompts.detach().to(DEVICE), self.prefix_len, dim=0) if self.previous_prompts.numel() > 0 else []
        for idx, prev_task in enumerate(prev_order):
            if idx >= len(chunks):
                break
            if prev_task not in self.base_losses or prev_task not in self.self_losses:
                try:
                    loader_prev_train = self._build_loader_for_task(prev_task, split="train")
                except Exception:
                    loader_prev_train = None
                if loader_prev_train is not None:
                    if prev_task not in self.base_losses:
                        self.base_losses[prev_task] = self.compute_loss_distribution(loader_prev_train, prompt=None)
                    if prev_task not in self.self_losses:
                        prompt_prev = chunks[idx].to(DEVICE)
                        self.self_losses[prev_task] = self.compute_loss_distribution(loader_prev_train, prompt=prompt_prev)
            key = (current_task, prev_task)
            if key not in self.cross_losses:
                prompt_prev = chunks[idx].to(DEVICE)
                self.cross_losses[key] = self.compute_loss_distribution(train_loader_current, prompt=prompt_prev)
            L_t_base = self.base_losses.get(current_task, [])
            L_i_base = self.base_losses.get(prev_task, []
            )
            L_i_self = self.self_losses.get(prev_task, [])
            L_t_with_i = self.cross_losses.get(key, [])
            dis_prime = wdist(L_i_base, L_t_base)
            dis = wdist(L_i_self, L_t_with_i)
            is_similar = dis < dis_prime
            sim[prev_task] = {'dis_prime': dis_prime, 'dis': dis, 'similar': is_similar}
            if print_results:
                try:
                    print(f"[DecLossSim] current={current_task} prev={prev_task} dis_self_vs_cross={dis:.6f} dis_base_vs_base={dis_prime:.6f} similar={is_similar}")
                except Exception:
                    pass
        return sim

    def _loss_with_explicit_prompt(self, batch, prompt_block: torch.Tensor) -> torch.Tensor:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        bsz, seq_len = input_ids.shape
        prefix_len_total = prompt_block.size(0)
        total_len = prefix_len_total + seq_len
        if total_len > self.model.config.max_position_embeddings:
            trim = total_len - self.model.config.max_position_embeddings
            if trim >= seq_len:
                raise RuntimeError("Sequence too long even after trimming.")
            input_ids = input_ids[:, trim:]
            attention_mask = attention_mask[:, trim:]
            labels = labels[:, trim:]
            seq_len = input_ids.size(1)
            total_len = prefix_len_total + seq_len
        with torch.no_grad():
            token_embeds = self.model.get_input_embeddings()(input_ids)
        prompt_batched = prompt_block.unsqueeze(0).expand(bsz, -1, -1)
        inputs_embeds = torch.cat([prompt_batched, token_embeds], dim=1)
        prefix_mask = torch.ones(bsz, prefix_len_total, device=DEVICE, dtype=attention_mask.dtype)
        attention_mask_ext = torch.cat([prefix_mask, attention_mask], dim=1)
        ignore_prefix = torch.full((bsz, prefix_len_total), -100, device=DEVICE, dtype=labels.dtype)
        labels_ext = torch.cat([ignore_prefix, labels], dim=1)
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask_ext, labels=labels_ext)
        return outputs.loss

    def _loss_without_prompt(self, batch) -> torch.Tensor:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        with torch.no_grad():
            token_embeds = self.model.get_input_embeddings()(input_ids)
        outputs = self.model(inputs_embeds=token_embeds, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    def compute_prompt_mean_gradient(self, loader: DataLoader, max_batches: int = 30) -> torch.Tensor:
        if self.current_prompt is None:
            return None
        self.model.train()
        grads = []
        seen = 0
        for batch in loader:
            if seen >= max_batches:
                break
            seen += 1
            if self.current_prompt.grad is not None:
                self.current_prompt.grad.zero_()
            loss = self._train_step(batch)
            loss.backward()
            if self.current_prompt.grad is not None:
                g = self._flatten_block(self.current_prompt.grad.detach())
                grads.append(g.cpu())
        if len(grads) == 0:
            return None
        G = torch.stack(grads, dim=0)
        mean_dir = G.mean(dim=0)
        norm = torch.norm(mean_dir)
        if norm > 0:
            mean_dir = mean_dir / norm
        return mean_dir.cpu()

    def compute_prev_prompts_gradients_on_current(self, loader: DataLoader, prev_order: List[str], max_batches: int = 10) -> Dict[str, Dict]:
        results = {}
        if self.previous_prompts.numel() == 0 or self.prefix_len <= 0:
            return results
        chunks = torch.split(self.previous_prompts.detach().to(DEVICE), self.prefix_len, dim=0)
        for idx, prev_task in enumerate(prev_order):
            if idx >= len(chunks):
                break
            block = chunks[idx].detach().clone().to(DEVICE)
            block.requires_grad_(True)
            grads = []
            seen = 0
            for batch in loader:
                if seen >= max_batches:
                    break
                seen += 1
                if block.grad is not None:
                    block.grad.zero_()
                loss = self._loss_with_explicit_prompt(batch, block)
                loss.backward()
                if block.grad is not None:
                    g = self._flatten_block(block.grad.detach())
                    grads.append(g.cpu())
            if len(grads) == 0:
                continue
            G = torch.stack(grads, dim=0)
            mean_dir = G.mean(dim=0)
            norm = torch.norm(mean_dir)
            if norm > 0:
                dir_unit = (mean_dir / norm).cpu().numpy()
            else:
                dir_unit = mean_dir.cpu().numpy()
            results[prev_task] = {'direction': dir_unit, 'norm': float(norm.cpu().item()) if norm is not None else 0.0}
        return results

    def compute_prompt_svd_basis(self, loader: DataLoader, max_batches: int = 30, topk: int = 3) -> np.ndarray:
        if self.current_prompt is None:
            return None
        self.model.train()
        rows = []
        seen = 0
        for batch in loader:
            if seen >= max_batches:
                break
            seen += 1
            if self.current_prompt.grad is not None:
                self.current_prompt.grad.zero_()
            loss = self._train_step(batch)
            loss.backward()
            if self.current_prompt.grad is not None:
                g = self._flatten_block(self.current_prompt.grad.detach()).float()
                ng = torch.norm(g)
                if ng > 0:
                    g = g / ng
                rows.append(g.unsqueeze(0).cpu())
        if len(rows) < 1:
            return None
        M = torch.cat(rows, dim=0)
        try:
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            V = Vh.transpose(0, 1)
            k = min(topk, V.shape[1])
            Vk = V[:, :k].contiguous().cpu().numpy()
            return Vk
        except Exception:
            return None

    @torch.no_grad()
    def _score_label_for_example(
        self,
        example: Dict,
        label_text: str,
        canonical_for_prompts: str,
        keys_for_task: Tuple[str, ...],
    ) -> float:
        self.model.eval()
        prompt_text = build_prompt_from_example(example, canonical_for_prompts, keys_for_task)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        label_ids = self.tokenizer(label_text, add_special_tokens=False)["input_ids"]
        task_prompt = self._get_task_prompt()
        prefix_len_total = task_prompt.size(0)
        max_pos = self.model.config.max_position_embeddings
        max_prompt_len = max_pos - prefix_len_total - len(label_ids)
        if max_prompt_len <= 0:
            return -1e9
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]
        full_ids = prompt_ids + label_ids
        input_ids = torch.tensor(full_ids, device=DEVICE).unsqueeze(0)
        seq_len = input_ids.size(1)
        token_embeds = self.model.get_input_embeddings()(input_ids)
        prompt_batched = task_prompt.unsqueeze(0)
        inputs_embeds = torch.cat([prompt_batched, token_embeds], dim=1)
        attention_mask = torch.ones(1, prefix_len_total + seq_len, device=DEVICE, dtype=torch.long)
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        logprob = 0.0
        base_pos = prefix_len_total + len(prompt_ids)
        for j, tok_id in enumerate(label_ids):
            token_pos = base_pos + j
            logprob += log_probs[0, token_pos - 1, tok_id].item()
        logprob /= max(len(label_ids), 1)
        return float(logprob)

    @torch.no_grad()
    def _predict_label_for_example(
        self,
        example: Dict,
        label_texts: Tuple[str, ...],
        canonical_for_prompts: str,
        keys_for_task: Tuple[str, ...],
    ) -> str:
        scores = []
        for label in label_texts:
            s = self._score_label_for_example(example, label, canonical_for_prompts, keys_for_task)
            scores.append(s)
        best_idx = int(torch.tensor(scores).argmax().item())
        return label_texts[best_idx]

    @torch.no_grad()
    def evaluate_task(
        self,
        task_name: str,
        split: str = "eval",
        max_samples: Optional[int] = None,
    ):
        meta = self.task_meta[task_name]
        dataset = meta["dataset"]
        label_column = meta["label_column"]
        label_texts = meta["label_texts"]
        canonical_for_prompts = meta["canonical_for_prompts"]
        keys_for_task = meta["keys"]
        if split == "eval":
            split_name = meta["eval_split"]
        elif split == "train":
            split_name = "train"
        else:
            split_name = split
        eval_split_raw = dataset[split_name]
        if max_samples is not None:
            n = min(max_samples, len(eval_split_raw))
            eval_split_raw = eval_split_raw.select(range(n))
        indices = range(len(eval_split_raw))
        correct = 0
        total = 0
        for idx in tqdm(indices, desc=f"Eval [{task_name}]"):
            example = eval_split_raw[idx]
            gold_id = example[label_column]
            gold_label = label_texts[gold_id]
            pred_label = self._predict_label_for_example(example, label_texts, canonical_for_prompts, keys_for_task)
            total += 1
            if pred_label == gold_label:
                correct += 1
        acc = correct / total if total > 0 else 0.0
        return acc

    @torch.no_grad()
    def _score_label_for_example_with_prompt(
        self,
        example: Dict,
        label_text: str,
        prompt_block: torch.Tensor,
        canonical_for_prompts: str,
        keys_for_task: Tuple[str, ...],
    ) -> float:
        self.model.eval()
        prompt_text = build_prompt_from_example(example, canonical_for_prompts, keys_for_task)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        label_ids = self.tokenizer(label_text, add_special_tokens=False)["input_ids"]
        P = prompt_block.size(0)
        max_pos = self.model.config.max_position_embeddings
        max_prompt_len = max_pos - P - len(label_ids)
        if max_prompt_len <= 0:
            return -1e9
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]
        full_ids = prompt_ids + label_ids
        input_ids = torch.tensor(full_ids, device=DEVICE).unsqueeze(0)
        seq_len = input_ids.size(1)
        token_embeds = self.model.get_input_embeddings()(input_ids)
        prompt_batched = prompt_block.unsqueeze(0)
        inputs_embeds = torch.cat([prompt_batched, token_embeds], dim=1)
        attention_mask = torch.ones(1, P + seq_len, device=DEVICE, dtype=torch.long)
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        logprob = 0.0
        base_pos = P + len(prompt_ids)
        for j, tok_id in enumerate(label_ids):
            token_pos = base_pos + j
            logprob += log_probs[0, token_pos - 1, tok_id].item()
        logprob /= max(len(label_ids), 1)
        return float(logprob)

    @torch.no_grad()
    def evaluate_task_with_prompt(
        self,
        task_name: str,
        prompt_block: torch.Tensor,
        split: str = "eval",
        max_samples: Optional[int] = None,
    ) -> float:
        meta = self.task_meta[task_name]
        dataset = meta["dataset"]
        label_column = meta["label_column"]
        label_texts = meta["label_texts"]
        canonical_for_prompts = meta["canonical_for_prompts"]
        keys_for_task = meta["keys"]
        if split == "eval":
            split_name = meta["eval_split"]
        elif split in dataset:
            split_name = split
        else:
            split_name = meta["eval_split"]
        eval_split_raw = dataset[split_name]
        if max_samples is not None:
            n = min(max_samples, len(eval_split_raw))
            eval_split_raw = eval_split_raw.select(range(n))
        correct = 0
        total = 0
        for idx in tqdm(range(len(eval_split_raw)), desc=f"Eval [{task_name}] explicit"):
            example = eval_split_raw[idx]
            gold_id = example[label_column]
            gold_label = label_texts[gold_id]
            scores = []
            for lb in label_texts:
                s = self._score_label_for_example_with_prompt(example, lb, prompt_block, canonical_for_prompts, keys_for_task)
                scores.append(s)
            pred_idx = int(torch.tensor(scores).argmax().item())
            pred_label = label_texts[pred_idx]
            total += 1
            if pred_label == gold_label:
                correct += 1
        return (correct / total) if total > 0 else 0.0

    def train_single_task(self, task_name: str) -> float:
        print(f"\n=== Training progressive prompt on task: {task_name} ===")
        meta = self.task_meta[task_name]
        dataset = meta["dataset"]
        label_column = meta["label_column"]
        canonical_for_prompts = meta["canonical_for_prompts"]
        label_texts = meta["label_texts"]
        keys_for_task = meta["keys"]
        self._start_new_task_prompt(label_texts)
        train_split = dataset["train"]
        if self.k_per_class > 0:
            train_split = balanced_subsample(train_split, label_column, self.k_per_class, seed=self.seed)
        print(f"[{task_name}] Train size after balancing: {len(train_split)}")
        target_max_length = max(len(self.tokenizer(lbl, add_special_tokens=False)["input_ids"]) for lbl in label_texts)
        preprocess_fn = self._build_preprocess_fn(label_texts, label_column, canonical_for_prompts, keys_for_task, target_max_length)
        processed = {}
        processed["train"] = train_split.map(preprocess_fn, batched=True, num_proc=1, remove_columns=train_split.column_names, load_from_cache_file=False, desc=f"Tokenizing train for {task_name}")
        eval_split_name = meta["eval_split"]
        eval_split_raw = dataset[eval_split_name]
        processed["eval"] = eval_split_raw.map(preprocess_fn, batched=True, num_proc=1, remove_columns=eval_split_raw.column_names, load_from_cache_file=False, desc=f"Tokenizing eval for {task_name}")
        train_loader = DataLoader(processed["train"], shuffle=True, collate_fn=default_data_collator, batch_size=self.batch_size, pin_memory=True)
        eval_loader = DataLoader(processed["eval"], shuffle=False, collate_fn=default_data_collator, batch_size=self.batch_size, pin_memory=True)
        optimizer = torch.optim.AdamW([self.current_prompt], lr=self.lr)
        self.per_epoch_acc[task_name] = []
        try:
            if task_name in self.task_list:
                curr_idx = self.task_list.index(task_name)
            else:
                curr_idx = len(self.task_list)
            prev_tasks_order_full = list(reversed(self.task_list[:curr_idx]))
            prev_grad_current = self.compute_prev_prompts_gradients_on_current(eval_loader, prev_tasks_order_full, max_batches=10)
            self.prev_prompt_grad_dirs[task_name] = prev_grad_current
            loss_sim = {}
            try:
                loss_sim = self.compute_loss_based_similarity_decoder(task_name, prev_tasks_order_full, train_loader, print_results=True)
            except Exception as e_loss_sim:
                print("Warning: decoder Wasserstein similarity failed:", e_loss_sim)
            try:
                cur_mean_dir = self.compute_prompt_mean_gradient(eval_loader, max_batches=20)
            except Exception:
                cur_mean_dir = None
            cur_mean_dir_t = cur_mean_dir.detach().float() if (cur_mean_dir is not None and torch.is_tensor(cur_mean_dir)) else None
            sims = []
            proj_scores = {}
            for pt in prev_tasks_order_full:
                own = self.prompt_grad_dirs.get(pt, None)
                cur = prev_grad_current.get(pt, None)
                if own is None or cur is None:
                    continue
                own_t = torch.tensor(own).float()
                cur_t = torch.tensor(cur['direction']).float()
                if own_t.numel() != cur_t.numel():
                    continue
                cos = torch.dot(own_t, cur_t) / (torch.norm(own_t) * torch.norm(cur_t) + 1e-8)
                sims.append((pt, float(cos.cpu().item())))
                pscore = 0.0
                try:
                    if cur_mean_dir_t is not None and pt in self.task_to_svd_basis:
                        V_np = self.task_to_svd_basis.get(pt, None)
                        if V_np is not None:
                            V = torch.tensor(V_np, dtype=cur_mean_dir_t.dtype, device=cur_mean_dir_t.device)
                            coeff = V.transpose(0, 1) @ cur_mean_dir_t.view(-1)
                            pscore = float(torch.norm(coeff).cpu().item())
                except Exception:
                    pscore = 0.0
                proj_scores[pt] = pscore
                try:
                    gnorm = float(prev_grad_current[pt]['norm'])
                except Exception:
                    gnorm = float('nan')
                cval = float(cos.cpu().item()) if torch.is_tensor(cos) else float(cos)
                cval = max(min(cval, 1.0), -1.0)
                angle_deg = float(np.degrees(np.arccos(cval)))
                print(f"[DecPrevGradNorm] current={task_name} prev={pt} norm={gnorm:.6f}")
                print(f"[DecDirAlign] current={task_name} prev={pt} cosine={cval:.6f} angle_deg={angle_deg:.2f}")
                try:
                    print(f"[DecProjScore] current={task_name} prev={pt} proj_norm={proj_scores.get(pt, 0.0):.6f}")
                except Exception:
                    pass
            selected_prev = []
            if getattr(self, "selection_method", "proj_cos") == "wasserstein":
                TH = 0.2
                for pt in prev_tasks_order_full:
                    s = loss_sim.get(pt, None)
                    if s is None:
                        continue
                    score = float(s.get('dis_prime', 0.0)) - float(s.get('dis', 0.0))
                    if s.get('similar', False) and score > TH:
                        selected_prev.append(pt)
                if len(selected_prev) == 0:
                    selected_prev = list(prev_tasks_order_full)
                try:
                    print(f"[DecSelectByWass] current={task_name} selected_prev={selected_prev} thresh=0.2")
                except Exception:
                    pass
            else:
                for pt, cosv in sims:
                    if cosv > 0.0 and proj_scores.get(pt, 0.0) > 0.1:
                        selected_prev.append(pt)
                if len(selected_prev) == 0:
                    selected_prev = list(prev_tasks_order_full)
                try:
                    print(f"[DecSelectByProjCos] current={task_name} selected_prev={selected_prev}")
                except Exception:
                    pass
            chunks = torch.split(self.previous_prompts.detach().to(DEVICE), self.prefix_len, dim=0)
            indices = [i for i, t in enumerate(prev_tasks_order_full) if t in set(selected_prev)]
            if len(indices) == 0:
                indices = list(range(len(chunks)))
            try:
                print(f"[DecSelectResolved] current={task_name} selected_prev={[prev_tasks_order_full[i] for i in indices]}")
            except Exception:
                pass
            self._reverse_selected_order = [prev_tasks_order_full[i] for i in indices]
            self._reverse_selected_indices = list(indices)
        except Exception as e_sel:
            print("Warning: selection prep failed:", e_sel)
            self._reverse_selected_order = []
            self._reverse_selected_indices = []

        has_prev = self.previous_prompts.numel() > 0
        normal_phase_epochs = 10
        reverse_phase_epochs = 2 if has_prev else 0
        effective_epochs = normal_phase_epochs + reverse_phase_epochs if has_prev else normal_phase_epochs
        print(f"[DecReversePlan] task={task_name} has_prev={has_prev} plan=normal:{normal_phase_epochs}, reverse:{reverse_phase_epochs}, effective:{effective_epochs}")

        for epoch in range(effective_epochs):
            self.model.train()
            total_loss = 0.0
            in_reverse = has_prev and (epoch >= normal_phase_epochs) and (epoch < normal_phase_epochs + reverse_phase_epochs)
            if epoch == normal_phase_epochs and not has_prev:
                print(f"[DecReversePhaseSkip] epoch={epoch} task={task_name} no previous prompts; reverse phase disabled")
            if epoch == normal_phase_epochs and reverse_phase_epochs == 0:
                print(f"[DecReversePhaseSkip] epoch={epoch} task={task_name} reverse_phase_epochs=0; configured epochs too few for reverse phase")
            if in_reverse and not self.reverse_phase_active:
                try:
                    self.reverse_phase_active = True
                    if self.current_prompt is not None:
                        self.current_prompt.requires_grad_(False)
                        print(f"[DecReversePhaseStart] epoch={epoch} task={task_name} entering reverse phase; updating previous prompts orthogonally and printing per-task updated accuracies")
                    full_chunks = list(torch.split(self.previous_prompts.detach().to(DEVICE), self.prefix_len, dim=0))
                    sel_chunks = [full_chunks[idx] for idx in self._reverse_selected_indices] if len(self._reverse_selected_indices) > 0 else []
                    sel_block = torch.cat(sel_chunks, dim=0).to(DEVICE) if len(sel_chunks) > 0 else torch.empty(0, self.hidden_size, device=DEVICE)
                    self.previous_prompts_param = nn.Parameter(sel_block.clone(), requires_grad=True)
                    optimizer.add_param_group({"params": [self.previous_prompts_param], "lr": self.lr})
                    basis_list = []
                    D = self.prefix_len * self.hidden_size
                    for prev_task in self._reverse_selected_order:
                        parts: List[torch.Tensor] = []
                        V_main_np = self.task_to_svd_basis.get(prev_task, None)
                        if V_main_np is not None:
                            V_main = torch.tensor(V_main_np, device=DEVICE, dtype=self.previous_prompts_param.dtype)
                            parts.append(V_main)
                        V_cum_np = self.cumulative_restrict_basis.get(prev_task, None)
                        if V_cum_np is not None:
                            V_cum = torch.tensor(V_cum_np, device=DEVICE, dtype=self.previous_prompts_param.dtype)
                            parts.append(V_cum)
                        if len(parts) == 0:
                            own_dir = self.prompt_grad_dirs.get(prev_task, None)
                            if own_dir is None:
                                basis_list.append(torch.zeros((D, 0), device=DEVICE, dtype=self.previous_prompts_param.dtype))
                            else:
                                v = torch.tensor(own_dir, device=DEVICE, dtype=self.previous_prompts_param.dtype).reshape(-1)
                                nv = torch.norm(v)
                                if nv > 0:
                                    v = v / nv
                                basis_list.append(v.view(-1, 1))
                        else:
                            Vcat = torch.cat(parts, dim=1)
                            try:
                                Q, _ = torch.linalg.qr(Vcat, mode="reduced")
                                basis_list.append(Q)
                            except Exception:
                                basis_list.append(Vcat)
                    self._reverse_basis_list = basis_list
                except Exception as e_enter:
                    print("Warning: failed to enter reverse phase:", e_enter)
                    self.reverse_phase_active = False
                    self.previous_prompts_param = None

            if self.reverse_phase_active and (not in_reverse):
                try:
                    full_chunks = list(torch.split(self.previous_prompts.detach().to(DEVICE), self.prefix_len, dim=0))
                    upd_chunks = list(torch.split(self.previous_prompts_param.detach().to(DEVICE), self.prefix_len, dim=0)) if self.previous_prompts_param is not None else []
                    try:
                        for j, idx_full in enumerate(self._reverse_selected_indices):
                            if j < len(upd_chunks) and idx_full < len(full_chunks):
                                prev_task = self._reverse_selected_order[j] if j < len(self._reverse_selected_order) else None
                                if prev_task is None:
                                    continue
                                delta = (upd_chunks[j] - full_chunks[idx_full]).detach()
                                dflat = delta.reshape(-1)
                                nrm = torch.norm(dflat)
                                if nrm is not None and float(nrm.item()) > 0.0:
                                    dflat = (dflat / nrm).to(torch.float32).cpu().numpy().reshape(-1, 1)
                                    exist = self.cumulative_restrict_basis.get(prev_task, None)
                                    if exist is None or exist.size == 0:
                                        Vnew = dflat
                                    else:
                                        Vnew = np.concatenate([exist, dflat], axis=1)
                                        Vtorch = torch.tensor(Vnew, dtype=torch.float32)
                                        try:
                                            Q, _ = torch.linalg.qr(Vtorch, mode="reduced")
                                            Vnew = Q.cpu().numpy()
                                        except Exception:
                                            Vnew = Vtorch.cpu().numpy()
                                    if Vnew.shape[1] > 6:
                                        Vnew = Vnew[:, :6]
                                    self.cumulative_restrict_basis[prev_task] = Vnew
                                    try:
                                        print(f"[DecCumRestrict] prev_task={prev_task} added=1 total={Vnew.shape[1]}")
                                    except Exception:
                                        pass
                    except Exception as e_cum:
                        print("Warning: cumulative restrict basis update failed:", e_cum)
                    if len(upd_chunks) == len(full_chunks):
                        full_chunks = upd_chunks
                    else:
                        for j, idx_full in enumerate(self._reverse_selected_indices):
                            if j < len(upd_chunks) and idx_full < len(full_chunks):
                                full_chunks[idx_full] = upd_chunks[j]
                    self.previous_prompts_updated = torch.cat(full_chunks, dim=0).detach().to(DEVICE)
                    self.previous_prompts_updated.requires_grad = False
                except Exception as e_exit:
                    print("Warning: reverse merge failed:", e_exit)
                if self.current_prompt is not None:
                    self.current_prompt.requires_grad_(True)
                self.reverse_phase_active = False
                self.previous_prompts_param = None
                self._reverse_basis_list = []

            for batch in tqdm(train_loader, desc=f"{task_name} | epoch {epoch} [train]"):
                optimizer.zero_grad()
                loss = self._train_step(batch)
                total_loss += loss.detach().float()
                loss.backward()
                if self.reverse_phase_active and self.previous_prompts_param is not None and self.previous_prompts_param.grad is not None:
                    try:
                        g = self.previous_prompts_param.grad
                        chunks_g = torch.split(g, self.prefix_len, dim=0)
                        for idx_block, gi in enumerate(chunks_g):
                            if idx_block >= len(self._reverse_basis_list):
                                break
                            V = self._reverse_basis_list[idx_block]
                            if V is None or V.numel() == 0:
                                continue
                            gi_flat = gi.reshape(-1)
                            proj = V @ (V.transpose(0, 1) @ gi_flat)
                            gi_flat = gi_flat - proj
                            gi.copy_(gi_flat.view_as(gi))
                    except Exception as e_proj:
                        print("Warning: projection failed:", e_proj)
                optimizer.step()

            avg_loss = (total_loss / len(train_loader)).item()

            self.model.eval()
            eval_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(eval_loader, desc=f"{task_name} | epoch {epoch} [eval-lm]"):
                    loss = self._train_step(batch)
                    eval_loss += loss.detach().float()
            eval_loss = (eval_loss / len(eval_loader)).item()
            print(f"[{task_name}] epoch {epoch}: eval_lm_loss={eval_loss:.4f}")

            acc_epoch = self.evaluate_task(task_name)
            self.per_epoch_acc[task_name].append(acc_epoch)
            print(f"[{task_name}] epoch {epoch}: eval_class_acc={acc_epoch:.4f}")

            if in_reverse and self.previous_prompts_param is not None:
                print("In reverse phase")
                try:
                    prev_tasks_order = list(self._reverse_selected_order)
                    chunks_upd = torch.split(self.previous_prompts_param.detach(), self.prefix_len, dim=0)
                    orig_chunks = torch.split(self.previous_prompts.detach(), self.prefix_len, dim=0) if self.previous_prompts.numel() > 0 else []
                    selected_indices = list(self._reverse_selected_indices)
                    for j, prev_task in enumerate(prev_tasks_order):
                        if j >= len(chunks_upd):
                            break
                        prompt_prev_only = chunks_upd[j].to(DEVICE)
                        try:
                            acc_test = self.evaluate_task_with_prompt(prev_task, prompt_prev_only, split="test")
                        except Exception:
                            acc_test = self.evaluate_task_with_prompt(prev_task, prompt_prev_only, split="eval")
                        print(f"[DecReverseEval] epoch={epoch} prev_task={prev_task} acc={acc_test:.4f}")
                        try:
                            if len(orig_chunks) > 0 and j < len(selected_indices):
                                idx_full = selected_indices[j]
                                if idx_full < len(orig_chunks):
                                    if idx_full + 1 < len(orig_chunks):
                                        hist_chunks = [chunks_upd[j]] + list(orig_chunks[idx_full + 1:])
                                    else:
                                        hist_chunks = [chunks_upd[j]]
                                else:
                                    hist_chunks = [chunks_upd[j]]
                            else:
                                hist_chunks = [chunks_upd[j]]
                            prompt_with_queue = torch.cat(hist_chunks, dim=0).to(DEVICE)
                            try:
                                acc_hist_test = self.evaluate_task_with_prompt(prev_task, prompt_with_queue, split="test")
                            except Exception:
                                acc_hist_test = self.evaluate_task_with_prompt(prev_task, prompt_with_queue, split="eval")
                            print(f"[DecReverseEvalQueue] epoch={epoch} prev_task={prev_task} acc={acc_hist_test:.4f}")
                        except Exception as e_hist:
                            print("Warning: decoder reverse queue eval failed:", e_hist)
                except Exception as e_rev:
                    print("Warning: decoder reverse eval failed:", e_rev)

        final_acc = self.per_epoch_acc[task_name][-1]
        self.task_to_acc[task_name] = final_acc
        print(f"[{task_name}] final accuracy: {final_acc:.4f}")

        try:
            mean_dir = self.compute_prompt_mean_gradient(eval_loader, max_batches=30)
            if mean_dir is not None:
                self.prompt_grad_dirs[task_name] = mean_dir.numpy()
                print(f"[DecOwnMean] task={task_name} stored mean gradient dir (D={mean_dir.numel()}).")
        except Exception as e_mean:
            print("Warning: failed to compute/store mean gradient:", e_mean)
        try:
            Vk = self.compute_prompt_svd_basis(eval_loader, max_batches=30, topk=3)
            if Vk is not None:
                self.task_to_svd_basis[task_name] = Vk
                print(f"[DecOwnSVD] task={task_name} stored topk SVD basis with shape={Vk.shape}.")
        except Exception as e_svd:
            print("Warning: failed to compute/store SVD basis:", e_svd)

        return final_acc

    def train_sequence(self, eval_all_tasks: bool = False, eval_seen_only: bool = False, results_path: str = None):
        history = []
        for idx, task_name in enumerate(self.task_list):
            acc = self.train_single_task(task_name)
            if eval_all_tasks:
                results = {}
                tasks_to_eval = self.task_list[: idx + 1] if eval_seen_only else self.task_list
                for t in tasks_to_eval:
                    use_updated = False
                    prompt_block = None
                    if self.previous_prompts_updated is not None:
                        prev_order_full = list(reversed(self.task_list[:idx]))
                        if t in prev_order_full:
                            try:
                                t_idx = prev_order_full.index(t)
                                chunks_upd = torch.split(self.previous_prompts_updated.detach().to(DEVICE), self.prefix_len, dim=0)
                                if t_idx < len(chunks_upd):
                                    prompt_block = chunks_upd[t_idx]
                                    use_updated = True
                            except Exception:
                                use_updated = False
                    if use_updated and prompt_block is not None:
                        try:
                            try:
                                acc_t = self.evaluate_task_with_prompt(t, prompt_block, split="test")
                            except Exception:
                                acc_t = self.evaluate_task_with_prompt(t, prompt_block, split="eval")
                        except Exception:
                            acc_t = self.evaluate_task(t)
                    else:
                        acc_t = self.evaluate_task(t)
                    results[t] = acc_t
                self.task_to_acc = results
                print(results)
                history.append(dict(self.task_to_acc)) 
            else:
                print(dict(self.task_to_acc))
                history.append(dict(self.task_to_acc))
        print("Final task_to_acc:", dict(self.task_to_acc))
        if results_path is not None:
            out = {
                "final_task_acc": dict(self.task_to_acc),
                "history": history,
                "per_epoch_acc": self.per_epoch_acc,  
            }
            with open(results_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved results history to {results_path}")
        return dict(self.task_to_acc), history


