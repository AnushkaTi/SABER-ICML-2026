import argparse
import os
import time

import numpy as np
import torch

from progressive_decoder_cl_final import ProgressiveDecoderPromptTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Decoder-only progressive prompt tuning")

    # Core model / task settings
    parser.add_argument("--base_model_name", type=str, default="gpt2",
                        help="HF model name, e.g. gpt2, bigscience/bloomz-560m, etc.")
    parser.add_argument(
        "--tasks",
        type=str,
        default="sst2,wic",
        help="Comma-separated list of tasks, e.g. 'mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp_review_full,amazon,sst2,dbpedia_14,ag_news,multirc,yahoo_answers_topics'"
    )

    # Prompt / training settings
    parser.add_argument("--prefix_len", type=int, default=10, help="Soft prompt length per task")
    parser.add_argument("--max_length", type=int, default=64, help="Max total sequence length")
    parser.add_argument("--lr", type=float, default=3e-2, help="Learning rate for prompt params")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--k_per_class", type=int, default=2000,
                        help="Max number of examples per class for training (0 = use all)")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs per task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--selection_method",
        type=str,
        default="proj_cos",
        choices=["proj_cos", "wasserstein"],
        help="Criterion for selecting previous prompts: 'proj_cos' or 'wasserstein'.",
    )

    # Data / cache
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HF cache dir (optional, otherwise env/HF defaults).")
    parser.add_argument("--data_root", type=str, default="./datasets/src/data",
                        help="Root for local CSV-based datasets (e.g., amazon).")
    parser.add_argument("--test_fixed_dir", type=str, default="./datasets/test_fixed",
                        help="Directory containing fixed test .npy files.")

    # Continual learning / eval behavior
    parser.add_argument("--save_path", type=str, default="./.runs/progressive_example",
                        help="Directory to save results and prompts.")
    parser.add_argument(
        "--eval_all_tasks",
        action="store_true",
        help="If set, evaluate on ALL tasks after each task training stage.",
    )
    parser.add_argument(
        "--eval_seen_only",
        action="store_true",
        help="When --eval_all_tasks is set, evaluate only on tasks seen so far (previous + current).",
    )    
    parser.add_argument(
        "--no_eval_all_tasks",
        dest="eval_all_tasks",
        action="store_false",
    )
    
    parser.set_defaults(eval_all_tasks=False)

    parser.add_argument(
        "--fix_test_data",
        action="store_true",
        help="Use pre-saved fixed test sets from test_fixed dir when available.",
    )
    parser.add_argument(
        "--no_fix_test_data",
        dest="fix_test_data",
        action="store_false",
    )
    parser.set_defaults(fix_test_data=True)

    return parser.parse_args()


def main():
    args = parse_args()

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print("Tasks:", task_list)
    print("Base model:", args.base_model_name)
    print("Using fixed test data:", args.fix_test_data)

    os.makedirs(args.save_path, exist_ok=True)

    start_time = time.time()

    trainer = ProgressiveDecoderPromptTrainer(
        base_model_name=args.base_model_name,
        task_list=task_list,
        prefix_len=args.prefix_len,
        max_length=args.max_length,
        lr=args.lr,
        batch_size=args.batch_size,
        k_per_class=args.k_per_class,
        num_epochs=args.num_epochs,
        selection_method=args.selection_method,
        seed=args.seed,
        cache_dir=args.cache_dir,
        fix_test_data=args.fix_test_data,
        test_fixed_dir=args.test_fixed_dir,
        data_root=args.data_root,
    )

    final_results, history = trainer.train_sequence(
        eval_all_tasks=args.eval_all_tasks,
        eval_seen_only=args.eval_seen_only,   # <-- add this
        results_path=None,  # we are saving npy files instead
    )

    # Save results
    np.save(os.path.join(args.save_path, "results_dict.npy"), final_results)
    np.save(os.path.join(args.save_path, "results_history.npy"), history, allow_pickle=True)

    prompts_np = trainer.get_prompts_numpy(include_current=True)
    np.save(os.path.join(args.save_path, "prompts.npy"), prompts_np)

    per_epoch_acc = trainer.per_epoch_acc
    np.save(os.path.join(args.save_path, "per_epoch_acc.npy"), per_epoch_acc, allow_pickle=True)

    print(f"Results saved to {args.save_path}")

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")

    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")


if __name__ == "__main__":
    main()