import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import default_data_collator

from .superNI_decoder import ProgressiveDecoderPromptTrainer, DEVICE


class SuperNIDecoderBigPPromptTrainer(ProgressiveDecoderPromptTrainer):
    """
    Decoder-only progressive prompt trainer (SuperNI) with a global shared prompt (Big P).
    - Normal phase (10 epochs): update Big P (scaled LR) + current task prompt.
    - Reverse phase (2 epochs): freeze Big P and current; update SELECTED previous prompts orthogonally.
    - Selection method matches base (projection score + cosine on SVD/dirs).
    - Reverse-phase evals: prompt-only and prompt+BigP-snapshot variants for each updated previous task.
    """

    def __init__(
        self,
        base_model_name: str,
        task_list: List[str],
        prefix_len: int = 10,
        max_length: int = 1024,
        lr: float = 3e-2,
        batch_size: int = 8,
        k_per_class: int = 2000,
        num_epochs: int = 10,
        selection_method: str = "proj_cos",
        seed: int = 42,
        cache_dir: Optional[str] = None,
        fix_test_data: bool = True,
        test_fixed_dir: str = "./datasets/test_fixed",
        data_root: str = "./datasets/src/data",
        superni_dir: str = "./SuperNI",
        pred_mode: str = "logprob",
        gen_max_new_tokens: int = 5,
        # BigP-specific
        global_prefix_len: Optional[int] = None,
        global_prompt_train_scale: float = 0.5,
        global_prompt_lr_scale: float = 0.1,
    ):
        super().__init__(
            base_model_name=base_model_name,
            task_list=task_list,
            prefix_len=prefix_len,
            max_length=max_length,
            lr=lr,
            batch_size=batch_size,
            k_per_class=k_per_class,
            num_epochs=num_epochs,
            seed=seed,
            cache_dir=cache_dir,
            fix_test_data=fix_test_data,
            test_fixed_dir=test_fixed_dir,
            data_root=data_root,
            superni_dir=superni_dir,
            pred_mode=pred_mode,
            gen_max_new_tokens=gen_max_new_tokens,
        )
        # Selection method: "proj_cos" or "wasserstein"
        self.selection_method = selection_method
        # Global shared prompt (Big P)
        self.global_prefix_len = prefix_len if global_prefix_len is None else int(global_prefix_len)
        self.global_prompt_train_scale = float(global_prompt_train_scale)
        self.global_prompt_lr_scale = float(global_prompt_lr_scale)
        self.global_prompt: Optional[nn.Parameter] = None
        if self.global_prefix_len and self.global_prefix_len > 0:
            self.global_prompt = nn.Parameter(
                torch.zeros(self.global_prefix_len, self.hidden_size, device=DEVICE),
                requires_grad=True,
            )
            # initialize from text similar to current prompt init (use first task labels)
            first_task = self.task_list[0] if len(self.task_list) > 0 else None
            if first_task is not None:
                first_labels = tuple(self.task_meta[first_task]["label_texts"]) if self.task_meta.get(first_task) else ()
                try:
                    init_gp = self._init_prompt_from_text(first_labels)[: self.global_prefix_len]
                    with torch.no_grad():
                        self.global_prompt.data.copy_(init_gp.to(DEVICE))
                except Exception:
                    pass
        # Snapshot of Big P when each task is learned
        self.task_to_global_snapshot: Dict[str, torch.Tensor] = {}

        # Loss-distribution caches for Wasserstein similarity
        self.base_losses: Dict[str, List[float]] = {}
        self.self_losses: Dict[str, List[float]] = {}
        self.cross_losses: Dict[Tuple[str, str], List[float]] = {}

    def _get_task_prompt(self) -> torch.Tensor:
        """
        Compose active prefix for decoder:
        - reverse phase: selected previous prompts block only
        - normal phase: [Big P (scaled)] + [current task prompt]
        """
        if self.reverse_phase_active and self.previous_prompts_param is not None:
            return self.previous_prompts_param
        segs: List[torch.Tensor] = []
        if self.global_prompt is not None:
            segs.append(self.global_prompt * self.global_prompt_train_scale)
        if self.current_prompt is not None:
            segs.append(self.current_prompt)
        if not segs:
            return self.previous_prompts  # can be empty
        return torch.cat(segs, dim=0)

    def train_single_task(self, task_name: str) -> float:
        """
        Same structure as the SuperNI base, with additions:
        - add Big P to the optimizer (scaled LR)
        - freeze/unfreeze Big P around reverse phase
        - snapshot Big P per task and use it for reverse-phase eval variants
        - prompt composition uses Big P + current in normal phase; only selected prev in reverse
        """
        print(f"\n=== Training (BigP, SuperNI) on task: {task_name} ===")
        meta = self.task_meta[task_name]
        dataset = meta["dataset"]
        label_column = meta["label_column"]
        canonical = meta["canonical_for_prompts"]
        label_texts = meta["label_texts"]
        keys = meta["keys"]
        is_gen = meta.get("is_generation", False)

        # Start new prompt and snapshot current BigP
        self._start_new_task_prompt(label_texts)
        if self.global_prompt is not None:
            self.task_to_global_snapshot[task_name] = self.global_prompt.detach().clone()

        train_split = dataset["train"]
        if not is_gen and self.k_per_class > 0:
            from .t5_decoder import balanced_subsample  # helper lives in t5_decoder module
            train_split = balanced_subsample(train_split, label_column, self.k_per_class, seed=self.seed)

        if not is_gen:
            target_max = max(len(self.tokenizer(lbl, add_special_tokens=False)["input_ids"]) for lbl in label_texts)
        else:
            sample_size = min(len(train_split), 512)
            if sample_size == 0:
                target_max = max(4, self.max_length // 2)
            else:
                targets = [train_split[i].get("target", "") or "" for i in range(sample_size)]
                lens = [len(self.tokenizer(t, add_special_tokens=False)["input_ids"]) for t in targets]
                observed = max(lens) if lens else 0
                target_max = min(max(4, observed), max(4, self.max_length // 2))

        preprocess = self._build_preprocess_fn(label_texts, label_column, canonical, keys, target_max)
        eval_name = meta["eval_split"]
        eval_raw = dataset[eval_name]

        processed_train = train_split.map(
            preprocess,
            batched=True,
            batch_size=16,
            writer_batch_size=16,
            num_proc=1,
            remove_columns=train_split.column_names,
            load_from_cache_file=False,
            keep_in_memory=False,
            desc=f"Tokenize train {task_name}",
        )
        processed_eval = eval_raw.map(
            preprocess,
            batched=True,
            batch_size=16,
            writer_batch_size=16,
            num_proc=1,
            remove_columns=eval_raw.column_names,
            load_from_cache_file=False,
            keep_in_memory=False,
            desc=f"Tokenize eval {task_name}",
        )

        train_loader = DataLoader(
            processed_train,
            shuffle=True,
            collate_fn=default_data_collator,
            batch_size=self.batch_size,
            pin_memory=True,
        )
        eval_loader = DataLoader(
            processed_eval,
            shuffle=False,
            collate_fn=default_data_collator,
            batch_size=self.batch_size,
            pin_memory=True,
        )

        # Optimizer: current prompt + Big P (scaled LR)
        optimizer = torch.optim.AdamW([self.current_prompt], lr=self.lr)
        if self.global_prompt is not None:
            optimizer.add_param_group({"params": [self.global_prompt], "lr": self.lr * self.global_prompt_lr_scale})
        self.per_epoch_acc[task_name] = []

        # Two-phase schedule: 10 normal + 2 reverse (if previous prompts exist)
        normal_phase_epochs = 10
        reverse_phase_epochs = 2 if self.previous_prompts.numel() > 0 else 0
        effective_epochs = normal_phase_epochs + reverse_phase_epochs if reverse_phase_epochs > 0 else self.num_epochs

        # Prepare selection signals (projection score + cosine)
        prev_tasks_order_full: List[str] = []
        proj_scores: Dict[str, float] = {}
        cos_scores: Dict[str, float] = {}
        loss_sim: Dict[str, Dict] = {}
        if self.previous_prompts.numel() > 0:
            try:
                curr_idx = self.task_list.index(task_name)
            except ValueError:
                curr_idx = len(self.task_list)
            prev_tasks_order_full = list(reversed(self.task_list[:curr_idx]))
            try:
                chunks_prev = torch.split(self.previous_prompts.detach().to(DEVICE), self.prefix_len, dim=0)
                for idx_prev, t_prev in enumerate(prev_tasks_order_full):
                    if idx_prev >= len(chunks_prev):
                        break
                    g_dir = self._compute_prev_slice_grad_dir_on_current(chunks_prev[idx_prev], eval_loader, max_batches=10)
                    V_np = self.task_to_svd_basis.get(t_prev, None)
                    if g_dir is not None and V_np is not None:
                        V = torch.tensor(V_np, dtype=g_dir.dtype, device=g_dir.device)
                        coeff = V.t() @ g_dir
                        proj_scores[t_prev] = float(torch.norm(coeff).item())
                        v1 = V[:, 0]
                        v1 = v1 / (v1.norm() + 1e-12)
                        cos_scores[t_prev] = float(torch.clamp(torch.dot(v1, g_dir), -1.0, 1.0).item())
                    else:
                        proj_scores[t_prev] = 0.0
                        cos_scores[t_prev] = -1.0
            except Exception:
                proj_scores, cos_scores = {}, {}
            # Optional: loss-based similarity (Wasserstein)
            if getattr(self, "selection_method", "proj_cos") == "wasserstein":
                try:
                    loss_sim = self.compute_loss_based_similarity_decoder(task_name, prev_tasks_order_full, train_loader, print_results=True)
                except Exception as e_loss:
                    print("Warning: decoder Wasserstein similarity failed:", e_loss)
                    loss_sim = {}

        for epoch in range(max(effective_epochs, self.num_epochs)):
            self.model.train()
            total_loss = 0.0
            in_reverse = (epoch >= normal_phase_epochs) and (epoch < normal_phase_epochs + reverse_phase_epochs)

            # Enter reverse phase
            if in_reverse and not self.reverse_phase_active and self.previous_prompts.numel() > 0:
                try:
                    self.reverse_phase_active = True
                    if self.current_prompt is not None:
                        self.current_prompt.requires_grad_(False)
                    if self.global_prompt is not None:
                        self.global_prompt.requires_grad_(False)
                    # Selection: projection/cosine or wasserstein
                    selected_set = set()
                    if getattr(self, "selection_method", "proj_cos") == "wasserstein":
                        TH = 0.2
                        for t_prev in prev_tasks_order_full:
                            s = loss_sim.get(t_prev, None)
                            if s is None:
                                continue
                            score = float(s.get("dis_prime", 0.0)) - float(s.get("dis", 0.0))
                            if s.get("similar", False) and score > TH:
                                selected_set.add(t_prev)
                        if len(selected_set) == 0:
                            selected_set = set(prev_tasks_order_full)
                        print(f"[DecSelectByWass] current={task_name} selected_prev={list(selected_set)} thresh=0.2")
                    else:
                        for t_prev in prev_tasks_order_full:
                            if cos_scores.get(t_prev, -1.0) > 0.0 and proj_scores.get(t_prev, 0.0) > 0.1:
                                selected_set.add(t_prev)
                        if len(selected_set) == 0:
                            selected_set = set(prev_tasks_order_full)
                        print(f"[DecSelectByProjCos] current={task_name} selected_prev={list(selected_set)}")
                    chunks_prev = torch.split(self.previous_prompts.detach().to(DEVICE), self.prefix_len, dim=0)
                    sel_indices = [i for i, t in enumerate(prev_tasks_order_full) if t in selected_set]
                    sel_chunks = [chunks_prev[i] for i in sel_indices] if len(sel_indices) > 0 else list(chunks_prev)
                    sel_block = torch.cat(sel_chunks, dim=0) if len(sel_chunks) > 0 else torch.empty(0, self.hidden_size, device=DEVICE)
                    self.previous_prompts_param = nn.Parameter(sel_block.clone(), requires_grad=True)
                    optimizer.add_param_group({"params": [self.previous_prompts_param], "lr": self.lr})
                    self._reverse_selected_order = [prev_tasks_order_full[i] for i in sel_indices] if len(sel_indices) > 0 else list(prev_tasks_order_full)
                    self._reverse_selected_indices = sel_indices if len(sel_indices) > 0 else list(range(len(chunks_prev)))
                    # Build projection bases with cumulative restriction
                    basis_list = []
                    D = self.prefix_len * self.hidden_size
                    for t_prev in self._reverse_selected_order:
                        parts = []
                        V_np = self.task_to_svd_basis.get(t_prev, None)
                        if V_np is not None:
                            parts.append(torch.tensor(V_np, device=DEVICE, dtype=self.previous_prompts_param.dtype))
                        V_cum = self.cumulative_restrict_basis.get(t_prev, None)
                        if V_cum is not None and getattr(V_cum, "size", 0) > 0:
                            parts.append(torch.tensor(V_cum, device=DEVICE, dtype=self.previous_prompts_param.dtype))
                        if len(parts) == 0:
                            v_self = self.prompt_grad_dirs.get(t_prev, None)
                            if v_self is None:
                                basis_list.append(torch.zeros((D, 0), device=DEVICE, dtype=self.previous_prompts_param.dtype))
                            else:
                                v = torch.tensor(v_self, device=DEVICE, dtype=self.previous_prompts_param.dtype).reshape(-1)
                                v = v / (v.norm() + 1e-12)
                                basis_list.append(v.view(-1, 1))
                        else:
                            Vcat = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
                            try:
                                Q, _ = torch.linalg.qr(Vcat, mode="reduced")
                                if Q.shape[1] > 6:
                                    Q = Q[:, :6]
                                basis_list.append(Q)
                            except Exception:
                                basis_list.append(Vcat)
                    self._reverse_basis_list = basis_list
                    print(f"[DecReversePhaseStart(BigP)] epoch={epoch} task={task_name} entering reverse phase")
                except Exception as e_enter:
                    print("Warning: reverse phase enter failed:", e_enter)
                    self.reverse_phase_active = False
                    self.previous_prompts_param = None

            # Exit reverse phase
            if self.reverse_phase_active and (not in_reverse):
                try:
                    full_chunks = list(torch.split(self.previous_prompts.detach().to(DEVICE), self.prefix_len, dim=0))
                    upd_chunks = list(torch.split(self.previous_prompts_param.detach().to(DEVICE), self.prefix_len, dim=0)) if self.previous_prompts_param is not None else []
                    selected_indices = self._reverse_selected_indices if len(self._reverse_selected_indices) > 0 else list(range(len(full_chunks)))
                    # update cumulative restriction with deltas
                    try:
                        for j, idx_full in enumerate(selected_indices):
                            if j < len(upd_chunks) and idx_full < len(full_chunks):
                                prev_task = self._reverse_selected_order[j] if j < len(self._reverse_selected_order) else None
                                if prev_task is None:
                                    continue
                                delta = (upd_chunks[j] - full_chunks[idx_full]).detach()
                                dflat = delta.reshape(-1)
                                nrm = dflat.norm()
                                if float(nrm.item()) > 0.0:
                                    v = (dflat / nrm).to(torch.float32).cpu().numpy().reshape(-1, 1)
                                    exist = self.cumulative_restrict_basis.get(prev_task, None)
                                    if exist is None or getattr(exist, "size", 0) == 0:
                                        Vnew = v
                                    else:
                                        Vnew = np.concatenate([exist, v], axis=1)
                                        Vtorch = torch.tensor(Vnew, dtype=torch.float32)
                                        try:
                                            Q, _ = torch.linalg.qr(Vtorch, mode="reduced")
                                            Vnew = Q.cpu().numpy()
                                        except Exception:
                                            Vnew = Vtorch.cpu().numpy()
                                    if Vnew.shape[1] > 6:
                                        Vnew = Vnew[:, :6]
                                    self.cumulative_restrict_basis[prev_task] = Vnew
                    except Exception as e_cum:
                        print("Warning: cumulative restrict update failed:", e_cum)
                    # merge back
                    for j, idx_full in enumerate(selected_indices):
                        if j < len(upd_chunks) and idx_full < len(full_chunks):
                            full_chunks[idx_full] = upd_chunks[j]
                    self.previous_prompts = torch.cat(full_chunks, dim=0).detach().to(DEVICE)
                except Exception as e_exit:
                    print("Warning: reverse phase exit failed:", e_exit)
                self.reverse_phase_active = False
                if self.current_prompt is not None:
                    self.current_prompt.requires_grad_(True)
                if self.global_prompt is not None:
                    self.global_prompt.requires_grad_(True)
                self.previous_prompts_param = None
                self._reverse_basis_list = []

            # Training loop
            from tqdm import tqdm
            for batch in tqdm(train_loader, desc=f"{task_name} ep{epoch}"):
                optimizer.zero_grad()
                loss = self._train_step(batch)
                total_loss += loss.detach().float()
                loss.backward()
                # Project gradients for previous_prompts_param if in reverse phase
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
                            proj = V @ (V.t() @ gi_flat)
                            gi_flat -= proj
                            gi.copy_(gi_flat.view_as(gi))
                    except Exception as e_proj:
                        print("Warning: reverse projection failed:", e_proj)
                optimizer.step()
            avg_loss = (total_loss / len(train_loader)).item()

            # Per-epoch reporting and eval
            train_acc = self.evaluate_task(task_name, split="train", max_samples=100)
            if is_gen and isinstance(train_acc, dict):
                train_scalar = float(train_acc.get("rougeL", 0.0))
            else:
                train_scalar = float(train_acc)
            print(f"[{task_name}] ep{epoch} loss={avg_loss:.4f} train_acc={train_scalar:.4f}")

            self.model.eval()
            if len(eval_loader) > 0 and not is_gen:
                eval_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(eval_loader, desc=f"{task_name} ep{epoch} eval"):
                        eval_loss += self._train_step(batch).detach().float()
                print(f"[{task_name}] ep{epoch} eval_loss={eval_loss.item() / len(eval_loader):.4f}")

            acc_epoch = self.evaluate_task(task_name, max_samples=100)
            self.per_epoch_acc[task_name].append(acc_epoch)
            if is_gen:
                print(f"[{task_name}] ep{epoch} rouge1={acc_epoch['rouge1']:.4f} rougeL={acc_epoch['rougeL']:.4f} bleu={acc_epoch['bleu']:.2f}")
            else:
                print(f"[{task_name}] ep{epoch} eval_acc={float(acc_epoch):.4f}")

            # Reverse-phase per-previous-task evaluation (prompt-only and prompt+BigP snapshot)
            if self.reverse_phase_active and self.previous_prompts_param is not None and self._reverse_selected_order:
                try:
                    chunks_upd = torch.split(self.previous_prompts_param.detach(), self.prefix_len, dim=0)
                    for j, prev_task in enumerate(self._reverse_selected_order):
                        if j >= len(chunks_upd):
                            break
                        prev_only = chunks_upd[j].to(DEVICE)
                        saved_curr = self.current_prompt
                        saved_prev = self.previous_prompts
                        try:
                            # Prompt-only
                            self.current_prompt = None
                            self.previous_prompts = prev_only
                            acc_prev = self.evaluate_task(prev_task, split="eval", max_samples=100)
                            if isinstance(acc_prev, dict):
                                print(f"[DecReverseEval(BigP)] epoch={epoch} prev_task={prev_task} rougeL={acc_prev.get('rougeL',0.0):.4f} bleu={acc_prev.get('bleu',0.0):.2f}")
                            else:
                                print(f"[DecReverseEval(BigP)] epoch={epoch} prev_task={prev_task} acc={float(acc_prev):.4f}")
                            # Prompt + BigP snapshot
                            gp_snap = self.task_to_global_snapshot.get(prev_task, None)
                            if gp_snap is not None:
                                composite = torch.cat([gp_snap.to(DEVICE), prev_only], dim=0)
                                self.previous_prompts = composite
                                acc_prev_gp = self.evaluate_task(prev_task, split="eval", max_samples=100)
                                if isinstance(acc_prev_gp, dict):
                                    print(f"[DecReverseEvalBigP] epoch={epoch} prev_task={prev_task} rougeL={acc_prev_gp.get('rougeL',0.0):.4f} bleu={acc_prev_gp.get('bleu',0.0):.2f}")
                                else:
                                    print(f"[DecReverseEvalBigP] epoch={epoch} prev_task={prev_task} acc={float(acc_prev_gp):.4f}")
                        except Exception as e_eval:
                            print("Warning: reverse per-task eval (BigP) failed:", e_eval)
                        finally:
                            self.current_prompt = saved_curr
                            self.previous_prompts = saved_prev
                except Exception as e_rev:
                    print("Warning: BigP reverse eval loop failed:", e_rev)

        # Store own-task mean dir and SVD basis (reuse base utilities)
        # Store own-task mean dir and SVD basis (reuse base utilities)
        try:
            mean_dir = self._compute_current_prompt_mean_grad(eval_loader, max_batches=30)
            if mean_dir is not None:
                self.prompt_grad_dirs[task_name] = mean_dir.detach().cpu().numpy()
        except Exception:
            pass
        try:
            Vk = self._compute_task_svd_basis(eval_loader, topk=3, max_batches=30)
            if Vk is not None:
                self.task_to_svd_basis[task_name] = Vk.numpy()
        except Exception:
            pass

        final = self.per_epoch_acc[task_name][-1]
        self.task_to_acc[task_name] = final
        if is_gen and isinstance(final, dict):
            print(f"[{task_name}] final rougeL={final.get('rougeL', 0):.4f} bleu={final.get('bleu', 0):.2f}")
        else:
            print(f"[{task_name}] final acc={float(final):.4f}")
        return final

    # ---------- Loss distribution utilities for Wasserstein similarity ----------
    def _build_loader_for_task(self, task_name: str, split: str = "train") -> DataLoader:
        meta = self.task_meta[task_name]
        dataset = meta["dataset"]
        label_column = meta["label_column"]
        canonical = meta["canonical_for_prompts"]
        keys = meta["keys"]
        label_texts = meta["label_texts"]
        is_gen = meta.get("is_generation", False)
        # estimate target_max similar to train_single_task
        if not is_gen:
            target_max = max(len(self.tokenizer(lbl, add_special_tokens=False)["input_ids"]) for lbl in label_texts)
        else:
            raw_split = dataset[split]
            sample_size = min(len(raw_split), 512)
            if sample_size == 0:
                target_max = max(4, self.max_length // 2)
            else:
                targets = [raw_split[i].get("target", "") or "" for i in range(sample_size)]
                lens = [len(self.tokenizer(t, add_special_tokens=False)["input_ids"]) for t in targets]
                observed = max(lens) if lens else 0
                target_max = min(max(4, observed), max(4, self.max_length // 2))
        preprocess_fn = self._build_preprocess_fn(label_texts, label_column, canonical, keys, target_max)
        raw_split = dataset[split]
        processed = raw_split.map(
            preprocess_fn,
            batched=True,
            num_proc=1,
            remove_columns=raw_split.column_names,
            load_from_cache_file=False,
            desc=f"Tokenizing {split} for {task_name}",
        )
        return DataLoader(
            processed,
            shuffle=False,
            collate_fn=default_data_collator,
            batch_size=self.batch_size,
            pin_memory=True,
        )

    def _loss_with_explicit_prompt(self, batch, prompt_block: torch.Tensor) -> torch.Tensor:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        bsz, seq_len = input_ids.shape
        p_total = prompt_block.size(0)
        total_len = p_total + seq_len
        try:
            max_pos = int(getattr(self.model.config, "max_position_embeddings", 4096))
        except Exception:
            max_pos = 4096
        if total_len > max_pos:
            trim = total_len - max_pos
            if trim >= seq_len:
                raise RuntimeError("Sequence too long even after trimming.")
            input_ids = input_ids[:, trim:]
            attention_mask = attention_mask[:, trim:]
            labels = labels[:, trim:]
            seq_len = input_ids.size(1)
        with torch.no_grad():
            token_embeds = self.model.get_input_embeddings()(input_ids)
        prompt_b = prompt_block.unsqueeze(0).expand(bsz, -1, -1)
        inputs_embeds = torch.cat([prompt_b, token_embeds], dim=1)
        prefix_mask = torch.ones(bsz, p_total, device=DEVICE, dtype=attention_mask.dtype)
        attention_mask_ext = torch.cat([prefix_mask, attention_mask], dim=1)
        ignore = torch.full((bsz, p_total), -100, device=DEVICE, dtype=labels.dtype)
        labels_ext = torch.cat([ignore, labels], dim=1)
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

    def compute_loss_based_similarity_decoder(
        self,
        current_task: str,
        prev_order: List[str],
        train_loader_current: DataLoader,
        print_results: bool = True,
    ) -> Dict[str, Dict]:
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
            L_i_base = self.base_losses.get(prev_task, [])
            L_i_self = self.self_losses.get(prev_task, [])
            L_t_with_i = self.cross_losses.get(key, [])
            dis_prime = wdist(L_i_base, L_t_base)
            dis = wdist(L_i_self, L_t_with_i)
            is_similar = dis < dis_prime
            sim[prev_task] = {"dis_prime": dis_prime, "dis": dis, "similar": is_similar}
            if print_results:
                try:
                    print(f"[DecLossSim] current={current_task} prev={prev_task} dis_self_vs_cross={dis:.6f} dis_base_vs_base={dis_prime:.6f} similar={is_similar}")
                except Exception:
                    pass
        return sim

