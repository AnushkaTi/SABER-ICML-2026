# Turning Back Without Forgetting: Selective Backward Refinement for Parameter-Efficient Continual Learning

## Abstract

While prompt-based parameter-efficient continual learning mitigates catastrophic forgetting by isolating task-specific prompts, this isolation also limits later tasks from improving earlier ones, leaving backward knowledge transfer underexplored. We address this limitation by proposing **S**elective b**A**ckward refinement for positive **B**ackward knowledge transf**ER** (**SABER**), a replay-free framework that enables controlled backward transfer in prompt-based continual learning. SABER determines *when* backward refinement is beneficial using complementary task-correlation criteria based on prompt-gradient geometry and loss-distribution similarity, and *how* to perform refinement safely by restricting updates to non-interfering directions in the prompt parameter space. Extensive experiments across multiple continual learning benchmarks and diverse pretrained backbones, including T5-Large, LLaMA, and Qwen, demonstrate that SABER consistently achieves positive backward transfer while maintaining strong overall average performance.

## Repository Structure

```text
.
├── seq2seq/
│   ├── LongSequence/        # Encoder-decoder experiments on Long Sequence benchmark
│   └── SuperNI/             # Encoder-decoder experiments on SuperNI benchmark
├── autoregressive/
│   ├── LongSequence/        # Decoder-only experiments on Long Sequence benchmark
│   └── SuperNI/             # Decoder-only experiments on SuperNI benchmark
└── data/
    ├── train/               # Sample training data
    └── test/                # Sample test data
```

## Main Components

The repository supports two model families:

1. **Seq2Seq models**, such as T5-Large.
2. **Autoregressive decoder-only models**, such as LLaMA and Qwen.

SABER supports two complementary task-correlation criteria:

* `proj_cos`: prompt-gradient-geometry-based criterion;
* `wasserstein`: loss-distribution-similarity-based criterion.

## Setup

Create and activate a conda environment:

```bash
conda create -n saber python=3.10
conda activate saber
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Data

Sample data for training and evaluation is provided under:

```text
data/train/
data/test/
```

For full experiments, prepare the corresponding benchmark datasets following the same format as the provided sample files.

## Running Seq2Seq Experiments

For Long Sequence experiments with an encoder-decoder model such as T5-Large:

```bash
cd seq2seq/LongSequence

python train.py \
  --save_name saber_t5_longsequence \
  --task_list mnli cb wic copa \
  --select_k_per_class 1000 \
  --batch_size 32 \
  --lr 0.3 \
  --num_epochs 10 \
  --freeze_weights \
  --prefix_len 10 \
  --model_name t5-large \
  --early_stopping \
  --test_eval_after_every_task \
  --selection_method proj_cos
```

To use the loss-distribution-similarity criterion, replace the last argument with:

```bash
--selection_method wasserstein
```

## Running Decoder-Only Experiments

For Long Sequence experiments with a decoder-only model:

```bash
cd autoregressive/LongSequence

python main.py \
  --base_model_name meta-llama/Llama-2-7b-hf \
  --tasks "yelp_review_full,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia_14,ag_news,yahoo_answers_topics,multirc,boolq,wic" \
  --prefix_len 10 \
  --max_length 256 \
  --lr 0.03 \
  --batch_size 16 \
  --k_per_class 1000 \
  --num_epochs 2 \
  --seed 42 \
  --save_path "./runs/saber_llama_longsequence" \
  --eval_all_tasks \
  --eval_seen_only \
  --fix_test_data \
  --test_fixed_dir "../../data/test" \
  --data_root "../../data" \
  --selection_method proj_cos
```

To use the loss-distribution-similarity criterion:

```bash
--selection_method wasserstein
```

## SuperNI Experiments

The repository also includes SuperNI implementations under:

```text
seq2seq/SuperNI/
autoregressive/SuperNI/
```

For decoder-only SuperNI experiments, use:

```bash
cd autoregressive/SuperNI

python caller.py \
  --base_model_name meta-llama/Llama-2-7b-hf \
  --tasks "task1,task2,task3" \
  --prefix_len 10 \
  --max_length 1024 \
  --lr 0.03 \
  --batch_size 8 \
  --k_per_class 2000 \
  --num_epochs 10 \
  --seed 42 \
  --save_path "./runs/saber_superni" \
  --eval_all_tasks \
  --fix_test_data
```

Please replace `"task1,task2,task3"` with the desired SuperNI task sequence.

## Outputs

The scripts save learned prompts, evaluation results, and intermediate logs to the specified output directory. Typical output files include:

```text
results_dict.npy
results_history.npy
prompts.npy
per_epoch_acc.npy
results.json
```

