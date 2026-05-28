# SABER-ICML-2026

## Abstract

While prompt-based parameter-efficient continual learning mitigates catastrophic forgetting by isolating task-specific prompts, this isolation also limits later tasks from improving earlier ones, leaving backward knowledge transfer underexplored. We address this limitation by proposing **S**elective b**A**ckward refinement for positive **B**ackward knowledge transf**ER** (**SABER**), a replay-free framework that enables controlled backward transfer in prompt-based continual learning.

SABER determines *when* backward refinement is beneficial using complementary task-correlation criteria based on prompt-gradient geometry and loss-distribution similarity, and *how* to perform refinement safely by restricting updates to non-interfering directions in the prompt parameter space.

Extensive experiments across multiple continual learning benchmarks, and diverse pretrained backbones, including T5-Large, LLaMA, and Qwen, demonstrate that SABER consistently achieves positive backward transfer while maintaining strong overall average performance.
