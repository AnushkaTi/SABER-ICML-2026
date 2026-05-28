import argparse
import json
import os
import time

import numpy as np
import torch

from superNI import ProgressiveDecoderPromptTrainer


def _to_json(v):
    if isinstance(v, dict):
        return {k: _to_json(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_to_json(x) for x in v]
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    return str(v)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_name", type=str, default="gpt2")
    p.add_argument("--tasks", type=str,
                   help="Comma-separated task list")
    p.add_argument("--prefix_len", type=int, default=10)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--k_per_class", type=int, default=2000)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--data_root", type=str, default="./datasets/src/data")
    p.add_argument("--test_fixed_dir", type=str, default="./datasets/test_fixed")
    p.add_argument("--save_path", type=str, default="./.runs/progressive_example")
    p.add_argument("--eval_all_tasks", action="store_true")
    p.add_argument("--no_eval_all_tasks", dest="eval_all_tasks", action="store_false")
    p.set_defaults(eval_all_tasks=False)
    p.add_argument("--fix_test_data", action="store_true")
    p.add_argument("--no_fix_test_data", dest="fix_test_data", action="store_false")
    p.set_defaults(fix_test_data=True)
    p.add_argument("--pred_mode", type=str, default="logprob", choices=["logprob", "generate"])
    p.add_argument("--gen_max_new_tokens", type=int, default=5)
    p.add_argument("--superni_dir", type=str, default="./SuperNI")
    return p.parse_args()


def main():
    args = parse_args()
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print("Tasks:", task_list)
    print("Base model:", args.base_model_name)
    print("Fixed test data:", args.fix_test_data)

    os.makedirs(args.save_path, exist_ok=True)
    t0 = time.time()

    trainer = ProgressiveDecoderPromptTrainer(
        base_model_name=args.base_model_name,
        task_list=task_list,
        prefix_len=args.prefix_len,
        max_length=args.max_length,
        lr=args.lr,
        batch_size=args.batch_size,
        k_per_class=args.k_per_class,
        num_epochs=args.num_epochs,
        seed=args.seed,
        cache_dir=args.cache_dir,
        fix_test_data=args.fix_test_data,
        test_fixed_dir=args.test_fixed_dir,
        data_root=args.data_root,
        superni_dir=args.superni_dir,
        pred_mode=args.pred_mode,
        gen_max_new_tokens=args.gen_max_new_tokens,
    )

    final_results, history = trainer.train_sequence(
        eval_all_tasks=args.eval_all_tasks,
        results_path=None,
    )

    out = args.save_path
    np.save(os.path.join(out, "results_dict.npy"), final_results)
    np.save(os.path.join(out, "results_history.npy"), history, allow_pickle=True)
    np.save(os.path.join(out, "prompts.npy"), trainer.get_prompts_numpy(include_current=True))
    per_epoch = trainer.per_epoch_acc
    np.save(os.path.join(out, "per_epoch_acc.npy"), per_epoch, allow_pickle=True)

    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({
            "final": _to_json(final_results),
            "history": _to_json(history),
            "per_epoch_acc": _to_json(per_epoch),
        }, f, indent=2)

    print(f"Results saved to {out}")
    print(f"Elapsed: {time.time() - t0:.2f}s")
    if torch.cuda.is_available():
        print(f"GPU allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        print(f"Peak GPU: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")


if __name__ == "__main__":
    main()
