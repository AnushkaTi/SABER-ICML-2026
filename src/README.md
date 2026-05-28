# Turning Back Without Forgetting: Selective Backward Refinement for Parameter-Efficient Continual Learning


## Abstract

While prompt-based parameter-efficient continual learning mitigates catastrophic forgetting by isolating task-specific prompts, this isolation also limits later tasks from improving earlier ones, leaving backward knowledge transfer underexplored. We address this limitation by proposing Selective bAckward refinement for positive Backward knowledge transfER (SABER), a replay-free framework that enables controlled backward transfer in prompt-based continual learning. SABER explicitly determines when backward refinement is beneficial using complementary task-correlation criteria based on prompt-gradient geometry and loss-distribution similarity, and how to perform refinement safely by restricting updates to non-interfering directions in the prompt parameter space. We provide theoretical guarantees showing that SABER’s backward refinements are interference-free and induce non-increasing loss under mild conditions. Extensive experiments across multiple continual learning benchmarks with different task orders, and diverse pretrained backbones, including T5-Large, LLaMA, and Qwen, demonstrate that SABER consistently achieves positive backward transfer while maintaining strong overall average performance.

The codebase provided contains:
1. seq2seq: Encoder Decoder Architecture
2. autoregressive: Decoder only Architecture
3. data: Sample Data used for training and evaluation