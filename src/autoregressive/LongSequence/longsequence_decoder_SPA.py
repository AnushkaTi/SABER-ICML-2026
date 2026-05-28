import os
import math
import json
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import DatasetDict, Dataset
from transformers import default_data_collator

from .t5_decoder import ProgressiveDecoderPromptTrainer, DEVICE


class ProgressiveDecoderBigPPromptTrainer(ProgressiveDecoderPromptTrainer):
    # Decoder-only trainer with a global prompt (Big P)
    def __init__(
        self,
        base_model_name: str,
        task_list: List[str],
        prefix_len: int = 10,
        global_prefix_len: Optional[int] = None,
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
            selection_method=selection_method,
            seed=seed,
            cache_dir=cache_dir,
            fix_test_data=fix_test_data,
            test_fixed_dir=test_fixed_dir,
            data_root=data_root,
        )
        # Global shared prompt (Big P)
        self.global_prefix_len = prefix_len if global_prefix_len is None else int(global_prefix_len)
        self.global_prompt_train_scale = float(global_prompt_train_scale)
        self.global_prompt_lr_scale = float(global_prompt_lr_scale)
        self.global_prompt: Optional[nn.Parameter] = nn.Parameter(
            torch.zeros(self.global_prefix_len, self.hidden_size, device=DEVICE), requires_grad=True
        ) if self.global_prefix_len and self.global_prefix_len > 0 else None
        if self.global_prompt is not None:
            # initialize from text similar to current prompt init
            self.global_prompt.data.copy_(self._init_prompt_from_text(tuple(self.task_meta[self.task_list[0]]["label_texts"])).to(DEVICE)[: self.global_prefix_len])
        # Snapshot of Big P at the time a task is learned
        self.task_to_global_snapshot: Dict[str, torch.Tensor] = {}

    # Build active prefix for current phase
    def _get_task_prompt(self) -> torch.Tensor:
        if self.reverse_phase_active and self.previous_prompts_param is not None:
            return self.previous_prompts_param
        segs = []
        if self.global_prompt is not None:
            segs.append(self.global_prompt * self.global_prompt_train_scale)
        if self.current_prompt is not None:
            segs.append(self.current_prompt)
        if not segs:
            return self.previous_prompts  # can be empty
        return torch.cat(segs, dim=0)

    # Train a single task with Big P behavior
    def train_single_task(self, task_name: str) -> float:
        print(f"\n=== Training (BigP) progressive prompt on task: {task_name} ===")
        meta = self.task_meta[task_name]
        dataset = meta["dataset"]
        label_column = meta["label_column"]
        canonical_for_prompts = meta["canonical_for_prompts"]
        label_texts = meta["label_texts"]
        keys_for_task = meta["keys"]

        # Start new prompt, snapshot Big P that will be used during this task's training
        self._start_new_task_prompt(label_texts)
        if self.global_prompt is not None:
            self.task_to_global_snapshot[task_name] = self.global_prompt.detach().clone()

        # Build loaders (reuse base code)
        train_split = dataset["train"]
        if self.k_per_class > 0:
            from .t5_decoder import balanced_subsample  # reuse helper
            train_split = balanced_subsample(train_split, label_column, self.k_per_class, seed=self.seed)
        print(f"[{task_name}] Train size after balancing: {len(train_split)}")
        target_max_length = max(len(self.tokenizer(lbl, add_special_tokens=False)["input_ids"]) for lbl in label_texts)
        preprocess_fn = self._build_preprocess_fn(label_texts, label_column, canonical_for_prompts, keys_for_task, target_max_length)
        processed = {}
        processed["train"] = train_split.map(preprocess_fn, batched=True, num_proc=1,
                                             remove_columns=train_split.column_names, load_from_cache_file=False,
                                             desc=f"Tokenizing train for {task_name}")
        eval_split_name = meta["eval_split"]
        eval_split_raw = dataset[eval_split_name]
        processed["eval"] = eval_split_raw.map(preprocess_fn, batched=True, num_proc=1,
                                               remove_columns=eval_split_raw.column_names, load_from_cache_file=False,
                                               desc=f"Tokenizing eval for {task_name}")
        train_loader = DataLoader(processed["train"], shuffle=True, collate_fn=default_data_collator, batch_size=self.batch_size, pin_memory=True)
        eval_loader = DataLoader(processed["eval"], shuffle=False, collate_fn=default_data_collator, batch_size=self.batch_size, pin_memory=True)

        # Optimizer: current prompt + Big P (scaled LR)
        optimizer = torch.optim.AdamW([self.current_prompt], lr=self.lr)
        if self.global_prompt is not None:
            optimizer.add_param_group({"params": [self.global_prompt], "lr": self.lr * self.global_prompt_lr_scale})

        # Precompute signals for selection (reuse base)
        try:
            if task_name in self.task_list:
                curr_idx = self.task_list.index(task_name)
            else:
                curr_idx = len(self.task_list)
            prev_tasks_order_full = list(reversed(self.task_list[:curr_idx]))
            prev_grad_current = self.compute_prev_prompts_gradients_on_current(eval_loader, prev_tasks_order_full, max_batches=10)
            self.prev_prompt_grad_dirs[task_name] = prev_grad_current
            # mean dir for projection scores
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
                # projection score
                pscore = 0.0
                try:
                    if cur_mean_dir_t is not None and pt in self.task_to_svd_basis:
                        V_np = self.task_to_svd_basis.get(pt, None)
                        if V_np is not None:
                            V = torch.tensor(V_np, dtype=cur_mean_dir_t.dtype, device=cur_mean_dir_t.device)  # [D,k]
                            coeff = V.transpose(0, 1) @ cur_mean_dir_t.view(-1)
                            pscore = float(torch.norm(coeff).cpu().item())
                except Exception:
                    pscore = 0.0
                proj_scores[pt] = pscore
                # debug prints
                try:
                    gnorm = float(prev_grad_current[pt]['norm'])
                except Exception:
                    gnorm = float('nan')
                cval = max(min(float(cos), 1.0), -1.0)
                import numpy as _np
                angle_deg = float(_np.degrees(_np.arccos(cval)))
                print(f"[DecPrevGradNorm] current={task_name} prev={pt} norm={gnorm:.6f}")
                print(f"[DecDirAlign] current={task_name} prev={pt} cosine={cval:.6f} angle_deg={angle_deg:.2f}")
                print(f"[DecProjScore] current={task_name} prev={pt} proj_norm={proj_scores.get(pt, 0.0):.6f}")
            # Wasserstein similarity
            loss_sim = {}
            try:
                loss_sim = self.compute_loss_based_similarity_decoder(task_name, prev_tasks_order_full, train_loader, print_results=True)
            except Exception as e_loss_sim:
                print("Warning: decoder Wasserstein similarity failed:", e_loss_sim)
            # Selection
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
                print(f"[DecSelectByWass] current={task_name} selected_prev={selected_prev} thresh=0.2")
            else:
                for pt, cosv in sims:
                    if cosv > 0.0 and proj_scores.get(pt, 0.0) > 0.1:
                        selected_prev.append(pt)
                if len(selected_prev) == 0:
                    selected_prev = list(prev_tasks_order_full)
                print(f"[DecSelectByProjCos] current={task_name} selected_prev={selected_prev}")
            chunks = torch.split(self.previous_prompts.detach().to(DEVICE), self.prefix_len, dim=0)
            indices = [i for i, t in enumerate(prev_tasks_order_full) if t in set(selected_prev)]
            if len(indices) == 0:
                indices = list(range(len(chunks)))
            print(f"[DecSelectResolved] current={task_name} selected_prev={[prev_tasks_order_full[i] for i in indices]}")
            self._reverse_selected_order = [prev_tasks_order_full[i] for i in indices]
            self._reverse_selected_indices = list(indices)
        except Exception as e_sel:
            print("Warning: selection prep failed:", e_sel)
            self._reverse_selected_order = []
            self._reverse_selected_indices = []

        # Fixed schedule: 10 + 2
        normal_phase_epochs = 10
        reverse_phase_epochs = 2 if self.previous_prompts.numel() > 0 else 0
        effective_epochs = normal_phase_epochs + reverse_phase_epochs if self.previous_prompts.numel() > 0 else normal_phase_epochs
        print(f"[DecReversePlan(BigP)] task={task_name} has_prev={self.previous_prompts.numel() > 0} plan=normal:{normal_phase_epochs}, reverse:{reverse_phase_epochs}, effective:{effective_epochs}")

        for epoch in range(effective_epochs):
            self.model.train()
            total_loss = 0.0
            in_reverse = (self.previous_prompts.numel() > 0) and (epoch >= normal_phase_epochs) and (epoch < normal_phase_epochs + reverse_phase_epochs)
            if epoch == normal_phase_epochs and reverse_phase_epochs == 0:
                print(f"[DecReversePhaseSkip] epoch={epoch} task={task_name} reverse_phase_epochs=0")
            # Enter reverse phase
            if in_reverse and not self.reverse_phase_active:
                try:
                    self.reverse_phase_active = True
                    if self.current_prompt is not None:
                        self.current_prompt.requires_grad_(False)
                    if self.global_prompt is not None:
                        self.global_prompt.requires_grad_(False)
                    print(f"[DecReversePhaseStart] epoch={epoch} task={task_name} entering reverse phase; updating selected previous prompts orthogonally")
                    full_chunks = list(torch.split(self.previous_prompts.detach().to(DEVICE), self.prefix_len, dim=0))
                    sel_chunks = [full_chunks[idx] for idx in self._reverse_selected_indices] if len(self._reverse_selected_indices) > 0 else []
                    sel_block = torch.cat(sel_chunks, dim=0).to(DEVICE) if len(sel_chunks) > 0 else torch.empty(0, self.hidden_size, device=DEVICE)
                    self.previous_prompts_param = nn.Parameter(sel_block.clone(), requires_grad=True)
                    optimizer.add_param_group({"params": [self.previous_prompts_param], "lr": self.lr})
                    # Build projection bases with cumulative restriction
                    basis_list = []
                    D = self.prefix_len * self.hidden_size
                    for prev_task in self._reverse_selected_order:
                        parts = []
                        V_np = self.task_to_svd_basis.get(prev_task, None)
                        if V_np is not None:
                            parts.append(torch.tensor(V_np, device=DEVICE, dtype=self.previous_prompts_param.dtype))
                        V_cum = self.cumulative_restrict_basis.get(prev_task, None)
                        if V_cum is not None:
                            parts.append(torch.tensor(V_cum, device=DEVICE, dtype=self.previous_prompts_param.dtype))
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
                            Vcat = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
                            try:
                                Q, _ = torch.linalg.qr(Vcat, mode="reduced")
                                if Q.shape[1] > 6:
                                    Q = Q[:, :6]
                                basis_list.append(Q)
                            except Exception:
                                basis_list.append(Vcat)
                    self._reverse_basis_list = basis_list
                except Exception as e_enter:
                    print("Warning: failed to enter reverse phase:", e_enter)
                    self.reverse_phase_active = False
                    self.previous_prompts_param = None

            # Exit reverse phase
            if self.reverse_phase_active and (not in_reverse):
                try:
                    full_chunks = list(torch.split(self.previous_prompts.detach().to(DEVICE), self.prefix_len, dim=0))
                    upd_chunks = list(torch.split(self.previous_prompts_param.detach().to(DEVICE), self.prefix_len, dim=0)) if self.previous_prompts_param is not None else []
                    # update cumulative restriction
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
                                    v = (dflat / nrm).to(torch.float32).cpu().numpy().reshape(-1, 1)
                                    exist = self.cumulative_restrict_basis.get(prev_task, None)
                                    if exist is None or exist.size == 0:
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
                                    try:
                                        print(f"[DecCumRestrict] prev_task={prev_task} added=1 total={Vnew.shape[1]}")
                                    except Exception:
                                        pass
                    except Exception as e_cum:
                        print("Warning: cumulative restrict basis update failed:", e_cum)
                    # merge updated
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
                if self.global_prompt is not None:
                    self.global_prompt.requires_grad_(True)
                self.reverse_phase_active = False
                self.previous_prompts_param = None
                self._reverse_basis_list = []

            # Epoch training
            from tqdm import tqdm
            for batch in tqdm(train_loader, desc=f"{task_name} | epoch {epoch} [train]"):
                optimizer.zero_grad()
                loss = self._train_step(batch)
                total_loss += loss.detach().float()
                loss.backward()
                # projection
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

            # Per-epoch evals
            self.model.eval()
            eval_loss = 0.0
            with torch.no_grad():
                for batch in DataLoader(processed["eval"], shuffle=False, collate_fn=default_data_collator, batch_size=self.batch_size, pin_memory=True):
                    loss = self._train_step(batch)
                    eval_loss += loss.detach().float()
            eval_loss = (eval_loss / len(DataLoader(processed["eval"], shuffle=False, collate_fn=default_data_collator, batch_size=self.batch_size))).item()
            print(f"[{task_name}] epoch {epoch}: eval_lm_loss={eval_loss:.4f}")
            acc_epoch = self.evaluate_task(task_name)
            self.per_epoch_acc[task_name].append(acc_epoch)
            print(f"[{task_name}] epoch {epoch}: eval_class_acc={acc_epoch:.4f}")

            # Reverse-phase per-previous-task evaluations (BigP variants)
            if in_reverse and self.previous_prompts_param is not None:
                try:
                    prev_tasks_order = list(self._reverse_selected_order)
                    chunks_upd = torch.split(self.previous_prompts_param.detach(), self.prefix_len, dim=0)
                    for j, prev_task in enumerate(prev_tasks_order):
                        if j >= len(chunks_upd):
                            break
                        prompt_prev_only = chunks_upd[j].to(DEVICE)
                        # prompt-only
                        try:
                            acc_test = self.evaluate_task_with_prompt(prev_task, prompt_prev_only, split="test")
                        except Exception:
                            acc_test = self.evaluate_task_with_prompt(prev_task, prompt_prev_only, split="eval")
                        print(f"[DecReverseEval] epoch={epoch} prev_task={prev_task} acc={acc_test:.4f}")
                        # prompt + BigP snapshot used when prev_task was trained
                        try:
                            gp_snap = self.task_to_global_snapshot.get(prev_task, None)
                            if gp_snap is not None:
                                prompt_with_gp = torch.cat([gp_snap.to(DEVICE), prompt_prev_only], dim=0)
                                try:
                                    acc_bigp = self.evaluate_task_with_prompt(prev_task, prompt_with_gp, split="test")
                                except Exception:
                                    acc_bigp = self.evaluate_task_with_prompt(prev_task, prompt_with_gp, split="eval")
                                print(f"[DecReverseEvalBigP] epoch={epoch} prev_task={prev_task} acc={acc_bigp:.4f}")
                        except Exception as e_bigp:
                            print("Warning: BigP reverse eval failed:", e_bigp)
                except Exception as e_rev:
                    print("Warning: decoder BigP reverse eval failed:", e_rev)

        # Store own-task mean and SVD (reuse base)
        try:
            mean_dir = self.compute_prompt_mean_gradient(eval_loader, max_batches=30)
            if mean_dir is not None:
                self.prompt_grad_dirs[task_name] = mean_dir.numpy()
                print(f"[DecOwnMean] task={task_name} stored mean gradient dir (D={mean_dir.numel()}).")
        except Exception as e_mean:
            print("Warning: failed to compute/store mean dir:", e_mean)
        try:
            Vk = self.compute_prompt_svd_basis(eval_loader, max_batches=30, topk=3)
            if Vk is not None:
                self.task_to_svd_basis[task_name] = Vk
                print(f"[DecOwnSVD] task={task_name} stored topk SVD basis with shape={Vk.shape}.")
        except Exception as e_svd:
            print("Warning: failed to compute/store SVD basis:", e_svd)

        final_acc = self.per_epoch_acc[task_name][-1]
        self.task_to_acc[task_name] = final_acc
        print(f"[{task_name}] final accuracy: {final_acc:.4f}")
        return final_acc


