# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 87 of 100

## 目录 (Table of Contents)

1. [R-Sparse: Rank-Aware Activation Sparsity for Efficient LLM Inference](#R-Sparse-Rank-Aware-Activation-Sparsity-for-Efficient-LLM-Inference)
2. [Differentiable and Learnable Wireless Simulation with Geometric Transformers](#Differentiable-and-Learnable-Wireless-Simulation-with-Geometric-Transformers)
3. [Meta Flow Matching: Integrating Vector Fields on the Wasserstein Manifold](#Meta-Flow-Matching-Integrating-Vector-Fields-on-the-Wasserstein-Manifold)
4. [Multi-Reward as Condition for Instruction-based Image Editing](#Multi-Reward-as-Condition-for-Instruction-based-Image-Editing)
5. [Montessori-Instruct: Generate Influential Training Data Tailored for Student Learning](#Montessori-Instruct-Generate-Influential-Training-Data-Tailored-for-Student-Learning)
6. [Infilling Score: A Pretraining Data Detection Algorithm for Large Language Models](#Infilling-Score-A-Pretraining-Data-Detection-Algorithm-for-Large-Language-Models)
7. [DiffusionGuard: A Robust Defense Against Malicious Diffusion-based Image Editing](#DiffusionGuard-A-Robust-Defense-Against-Malicious-Diffusion-based-Image-Editing)
8. [LLMOPT: Learning to Define and Solve General Optimization Problems from Scratch](#LLMOPT-Learning-to-Define-and-Solve-General-Optimization-Problems-from-Scratch)
9. [Controllable Unlearning for Image-to-Image Generative Models via $\epsilon$-Constrained Optimization](#Controllable-Unlearning-for-Image-to-Image-Generative-Models-via-epsilon-Constrained-Optimization)
10. [IRIS: LLM-Assisted Static Analysis for Detecting Security Vulnerabilities](#IRIS-LLM-Assisted-Static-Analysis-for-Detecting-Security-Vulnerabilities)
11. [Towards Optimal Multi-draft Speculative Decoding](#Towards-Optimal-Multi-draft-Speculative-Decoding)
12. [ALBAR: Adversarial Learning approach to mitigate Biases in Action Recognition](#ALBAR-Adversarial-Learning-approach-to-mitigate-Biases-in-Action-Recognition)
13. [An Optimal Discriminator Weighted Imitation Perspective for Reinforcement Learning](#An-Optimal-Discriminator-Weighted-Imitation-Perspective-for-Reinforcement-Learning)
14. [Visual Haystacks: A Vision-Centric Needle-In-A-Haystack Benchmark](#Visual-Haystacks-A-Vision-Centric-Needle-In-A-Haystack-Benchmark)
15. [Reward Learning from Multiple Feedback Types](#Reward-Learning-from-Multiple-Feedback-Types)
16. [3D-Properties: Identifying Challenges in DPO and Charting a Path Forward](#3D-Properties-Identifying-Challenges-in-DPO-and-Charting-a-Path-Forward)
17. [Temporal Flexibility in Spiking Neural Networks: Towards Generalization Across Time Steps and Deployment Friendliness](#Temporal-Flexibility-in-Spiking-Neural-Networks-Towards-Generalization-Across-Time-Steps-and-Deployment-Friendliness)
18. [OmniPhysGS: 3D Constitutive Gaussians for General Physics-Based Dynamics Generation](#OmniPhysGS-3D-Constitutive-Gaussians-for-General-Physics-Based-Dynamics-Generation)
19. [SqueezeAttention: 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget](#SqueezeAttention-2D-Management-of-KV-Cache-in-LLM-Inference-via-Layer-wise-Optimal-Budget)
20. [Hyper-Connections](#Hyper-Connections)
21. [PRDP: Progressively Refined Differentiable Physics](#PRDP-Progressively-Refined-Differentiable-Physics)
22. [Inverse Constitutional AI: Compressing Preferences into Principles](#Inverse-Constitutional-AI-Compressing-Preferences-into-Principles)
23. [An Efficient Framework for Crediting Data Contributors of Diffusion Models](#An-Efficient-Framework-for-Crediting-Data-Contributors-of-Diffusion-Models)
24. [Optimization by Parallel Quasi-Quantum Annealing with Gradient-Based Sampling](#Optimization-by-Parallel-Quasi-Quantum-Annealing-with-Gradient-Based-Sampling)
25. [VideoPhy: Evaluating Physical Commonsense for Video Generation](#VideoPhy-Evaluating-Physical-Commonsense-for-Video-Generation)
26. [Ada-K Routing: Boosting the Efficiency of MoE-based LLMs](#Ada-K-Routing-Boosting-the-Efficiency-of-MoE-based-LLMs)
27. [Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures](#Enhancing-Zeroth-order-Fine-tuning-for-Language-Models-with-Low-rank-Structures)
28. [DebGCD: Debiased Learning with Distribution Guidance for Generalized Category Discovery](#DebGCD-Debiased-Learning-with-Distribution-Guidance-for-Generalized-Category-Discovery)
29. [LANTERN: Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding](#LANTERN-Accelerating-Visual-Autoregressive-Models-with-Relaxed-Speculative-Decoding)
30. [Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency](#Semi-Supervised-CLIP-Adaptation-by-Enforcing-Semantic-and-Trapezoidal-Consistency)
31. [SiMHand: Mining Similar Hands for Large-Scale 3D Hand Pose Pre-training](#SiMHand-Mining-Similar-Hands-for-Large-Scale-3D-Hand-Pose-Pre-training)
32. [Causal Identification for Complex Functional Longitudinal Studies](#Causal-Identification-for-Complex-Functional-Longitudinal-Studies)
33. [Denoising Task Difficulty-based Curriculum for Training Diffusion Models](#Denoising-Task-Difficulty-based-Curriculum-for-Training-Diffusion-Models)
34. [Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations](#Interpreting-and-Editing-Vision-Language-Representations-to-Mitigate-Hallucinations)
35. [Data Pruning by Information Maximization](#Data-Pruning-by-Information-Maximization)
36. [Edge Prompt Tuning for Graph Neural Networks](#Edge-Prompt-Tuning-for-Graph-Neural-Networks)
37. [Joint Fine-tuning and Conversion of Pretrained Speech and Language Models towards Linear Complexity](#Joint-Fine-tuning-and-Conversion-of-Pretrained-Speech-and-Language-Models-towards-Linear-Complexity)

---


## R-Sparse: Rank-Aware Activation Sparsity for Efficient LLM Inference

### Images

![09e1135673f8a9adf4669a0d737492da014f29b1f992d1d62d37b505443d7fb5.jpg](../iclr_results/3197_Mixture of Parrots_ Experts improve memorization more than reasoning/images/09e1135673f8a9adf4669a0d737492da014f29b1f992d1d62d37b505443d7fb5.jpg)

![1cf4fdb517dad54cbb6e2769ea31dc18fbcbf5e8182af9c71fd98cbaf9cce144.jpg](../iclr_results/3197_Mixture of Parrots_ Experts improve memorization more than reasoning/images/1cf4fdb517dad54cbb6e2769ea31dc18fbcbf5e8182af9c71fd98cbaf9cce144.jpg)

![2271639568950da649b7a07d73d2765d0e25a5e09f018c970584e7d5ae390519.jpg](../iclr_results/3197_Mixture of Parrots_ Experts improve memorization more than reasoning/images/2271639568950da649b7a07d73d2765d0e25a5e09f018c970584e7d5ae390519.jpg)

![3d1ba18e6c9f582ff77f44de030b04b2710d5e4b131b7d0ecbb218cd5051d35c.jpg](../iclr_results/3197_Mixture of Parrots_ Experts improve memorization more than reasoning/images/3d1ba18e6c9f582ff77f44de030b04b2710d5e4b131b7d0ecbb218cd5051d35c.jpg)

![462a62352074eafa6899d2f40b3b6e74fc5375eb8cedc1e3aa0ad266af481bd5.jpg](../iclr_results/3197_Mixture of Parrots_ Experts improve memorization more than reasoning/images/462a62352074eafa6899d2f40b3b6e74fc5375eb8cedc1e3aa0ad266af481bd5.jpg)

![76d23ac32ae064d95a4511791ca5ce8f8c337c58b3731b160a473d5feecdbf2e.jpg](../iclr_results/3197_Mixture of Parrots_ Experts improve memorization more than reasoning/images/76d23ac32ae064d95a4511791ca5ce8f8c337c58b3731b160a473d5feecdbf2e.jpg)

![7de8b0ba9d24e0ad58da5d4ad83199c1b5dac312754496e88b2ad86180905a5f.jpg](../iclr_results/3197_Mixture of Parrots_ Experts improve memorization more than reasoning/images/7de8b0ba9d24e0ad58da5d4ad83199c1b5dac312754496e88b2ad86180905a5f.jpg)

![ba127868fe2b7a70d6f7e9261a65f7878e249fc262c5626d5917b7f13f84219d.jpg](../iclr_results/3197_Mixture of Parrots_ Experts improve memorization more than reasoning/images/ba127868fe2b7a70d6f7e9261a65f7878e249fc262c5626d5917b7f13f84219d.jpg)

![ce1563621be5c763a3a033c71ca11ac2bb912f375fbcda36204710c53e5eba1b.jpg](../iclr_results/3197_Mixture of Parrots_ Experts improve memorization more than reasoning/images/ce1563621be5c763a3a033c71ca11ac2bb912f375fbcda36204710c53e5eba1b.jpg)

![e1d00f7e97b7c07cdc8c0947b39bd0ae5136070c148b1e8c97a6b9454a8ffcce.jpg](../iclr_results/3197_Mixture of Parrots_ Experts improve memorization more than reasoning/images/e1d00f7e97b7c07cdc8c0947b39bd0ae5136070c148b1e8c97a6b9454a8ffcce.jpg)

![e482929fd07a7d29e594ec9d43badc44a2ff49cfb58e422a70a8d29142266b34.jpg](../iclr_results/3197_Mixture of Parrots_ Experts improve memorization more than reasoning/images/e482929fd07a7d29e594ec9d43badc44a2ff49cfb58e422a70a8d29142266b34.jpg)

![e680b81f5354d1a1f43eab8f9050004cc21ec7dbc4fdb715ae9fe10a2e2a7f49.jpg](../iclr_results/3197_Mixture of Parrots_ Experts improve memorization more than reasoning/images/e680b81f5354d1a1f43eab8f9050004cc21ec7dbc4fdb715ae9fe10a2e2a7f49.jpg)

## R-Sparse: Rank-Aware Activation Sparsity for Efficient LLM Inference


### Images

![2e3838eaf35d5552787c9a4f29c3f530434fbd7c29dcc5a1ec620dffe282cf82.jpg](../iclr_results/3198_R-Sparse_ Rank-Aware Activation Sparsity for Efficient LLM Inference/images/2e3838eaf35d5552787c9a4f29c3f530434fbd7c29dcc5a1ec620dffe282cf82.jpg)

![378a2c2c1c6cc10a68fe31cd9b69cb8d63cdfdcb8d8b3b1f6ea82c680aff3075.jpg](../iclr_results/3198_R-Sparse_ Rank-Aware Activation Sparsity for Efficient LLM Inference/images/378a2c2c1c6cc10a68fe31cd9b69cb8d63cdfdcb8d8b3b1f6ea82c680aff3075.jpg)

![3bc3a4639e863abe2cf1b6d726847f578231e829dffea4db5a88e6251a3d53d7.jpg](../iclr_results/3198_R-Sparse_ Rank-Aware Activation Sparsity for Efficient LLM Inference/images/3bc3a4639e863abe2cf1b6d726847f578231e829dffea4db5a88e6251a3d53d7.jpg)

![4c259398814d0db336e2d11dc904ef258920855b5451d625f7de50391b2f4c58.jpg](../iclr_results/3198_R-Sparse_ Rank-Aware Activation Sparsity for Efficient LLM Inference/images/4c259398814d0db336e2d11dc904ef258920855b5451d625f7de50391b2f4c58.jpg)

![56b775bd9f7ee0eec79f1a82ae6df46af88c631481d8f880fce4e615e2f7a2c0.jpg](../iclr_results/3198_R-Sparse_ Rank-Aware Activation Sparsity for Efficient LLM Inference/images/56b775bd9f7ee0eec79f1a82ae6df46af88c631481d8f880fce4e615e2f7a2c0.jpg)

![670ebc1f20a552463e783a30180bb42a8fbbe294a3f741a3849867864e68d5dc.jpg](../iclr_results/3198_R-Sparse_ Rank-Aware Activation Sparsity for Efficient LLM Inference/images/670ebc1f20a552463e783a30180bb42a8fbbe294a3f741a3849867864e68d5dc.jpg)

![8011172650d4e702262f9baed7238a3479abd014c3a2d59dddf785cf989e7605.jpg](../iclr_results/3198_R-Sparse_ Rank-Aware Activation Sparsity for Efficient LLM Inference/images/8011172650d4e702262f9baed7238a3479abd014c3a2d59dddf785cf989e7605.jpg)

![da83d05488575a39fc19121e9b54ef21df014fcb5b960aea10c955323a319556.jpg](../iclr_results/3198_R-Sparse_ Rank-Aware Activation Sparsity for Efficient LLM Inference/images/da83d05488575a39fc19121e9b54ef21df014fcb5b960aea10c955323a319556.jpg)

![f167cd2649875b893ca340edb9c9d7c74b2b5bfa6dae8875241c0cc2a327186b.jpg](../iclr_results/3198_R-Sparse_ Rank-Aware Activation Sparsity for Efficient LLM Inference/images/f167cd2649875b893ca340edb9c9d7c74b2b5bfa6dae8875241c0cc2a327186b.jpg)

### Tables

![4b3435bbe92626f6889654131b9792a49a9948012b6a56ce8727e2f02ea66f6d.jpg](../iclr_results/3198_R-Sparse_ Rank-Aware Activation Sparsity for Efficient LLM Inference/tables/4b3435bbe92626f6889654131b9792a49a9948012b6a56ce8727e2f02ea66f6d.jpg)

![53f6555fcc1180bff567334796eaaeb583839dbe210bc43472b011be8593d46e.jpg](../iclr_results/3198_R-Sparse_ Rank-Aware Activation Sparsity for Efficient LLM Inference/tables/53f6555fcc1180bff567334796eaaeb583839dbe210bc43472b011be8593d46e.jpg)

![578968e7e72f7ea0ae8c3a1c70c01f4660ff28529eb6af8d53d9d846e250088f.jpg](../iclr_results/3198_R-Sparse_ Rank-Aware Activation Sparsity for Efficient LLM Inference/tables/578968e7e72f7ea0ae8c3a1c70c01f4660ff28529eb6af8d53d9d846e250088f.jpg)

![734dacc52d59d7f7bf897f922f6d7daeb3e3154759b7a0d48fb12b972dbc6c01.jpg](../iclr_results/3198_R-Sparse_ Rank-Aware Activation Sparsity for Efficient LLM Inference/tables/734dacc52d59d7f7bf897f922f6d7daeb3e3154759b7a0d48fb12b972dbc6c01.jpg)

## Differentiable and Learnable Wireless Simulation with Geometric Transformers


### Images

![01178f1b6b99e02fa86319070e666c83fa62e07171c8977cc87689de581aeab0.jpg](../iclr_results/3199_Differentiable and Learnable Wireless Simulation with Geometric Transformers/images/01178f1b6b99e02fa86319070e666c83fa62e07171c8977cc87689de581aeab0.jpg)

![3085eb89ae20b27614755fabc3b2f9faabef04219bad56c3413a23cae72f96ab.jpg](../iclr_results/3199_Differentiable and Learnable Wireless Simulation with Geometric Transformers/images/3085eb89ae20b27614755fabc3b2f9faabef04219bad56c3413a23cae72f96ab.jpg)

![3da68fb594521150433c0ce3bb82a1078b50550eb611b54a16252eff4e2cc265.jpg](../iclr_results/3199_Differentiable and Learnable Wireless Simulation with Geometric Transformers/images/3da68fb594521150433c0ce3bb82a1078b50550eb611b54a16252eff4e2cc265.jpg)

![529ac83633d5b239586a835e6abce27ce22f1fe27dc82e1100d40ad465f32787.jpg](../iclr_results/3199_Differentiable and Learnable Wireless Simulation with Geometric Transformers/images/529ac83633d5b239586a835e6abce27ce22f1fe27dc82e1100d40ad465f32787.jpg)

![68a362c0068b37f1779ef6d08046f734602dbf01fca59d7ff85e660390842dd3.jpg](../iclr_results/3199_Differentiable and Learnable Wireless Simulation with Geometric Transformers/images/68a362c0068b37f1779ef6d08046f734602dbf01fca59d7ff85e660390842dd3.jpg)

![ae90f3514a6a6a5f6a21cc56ab221d78f30966fba13a8e0a15402d2c811e8fc2.jpg](../iclr_results/3199_Differentiable and Learnable Wireless Simulation with Geometric Transformers/images/ae90f3514a6a6a5f6a21cc56ab221d78f30966fba13a8e0a15402d2c811e8fc2.jpg)

![b0889fa5011edbfeaea1b48e979c718d7f5e06da14cb0b48855b02d4346d6e6f.jpg](../iclr_results/3199_Differentiable and Learnable Wireless Simulation with Geometric Transformers/images/b0889fa5011edbfeaea1b48e979c718d7f5e06da14cb0b48855b02d4346d6e6f.jpg)

![c7caed012127be3048b0cd85240cb6d7b4c40459b2fa6d5e210c2cc85dc51942.jpg](../iclr_results/3199_Differentiable and Learnable Wireless Simulation with Geometric Transformers/images/c7caed012127be3048b0cd85240cb6d7b4c40459b2fa6d5e210c2cc85dc51942.jpg)

![d36e161d2dab97cecf2425262b02ef3be2ce72ade485e4463667a465c99557ee.jpg](../iclr_results/3199_Differentiable and Learnable Wireless Simulation with Geometric Transformers/images/d36e161d2dab97cecf2425262b02ef3be2ce72ade485e4463667a465c99557ee.jpg)

![f35b32fe32be6daa1df8154a35dcfa7dd804a89898e65d69f10ce64e3433a8a4.jpg](../iclr_results/3199_Differentiable and Learnable Wireless Simulation with Geometric Transformers/images/f35b32fe32be6daa1df8154a35dcfa7dd804a89898e65d69f10ce64e3433a8a4.jpg)

### Tables

![010a4b5034892916fae39d5656101ed0afc3c948b7d785a0ab0b30adce6131c9.jpg](../iclr_results/3199_Differentiable and Learnable Wireless Simulation with Geometric Transformers/tables/010a4b5034892916fae39d5656101ed0afc3c948b7d785a0ab0b30adce6131c9.jpg)

![0c82345c6c06ab2d3412a3ec38feaa11f8daa17691c15d5fd03b82d98d0d9544.jpg](../iclr_results/3199_Differentiable and Learnable Wireless Simulation with Geometric Transformers/tables/0c82345c6c06ab2d3412a3ec38feaa11f8daa17691c15d5fd03b82d98d0d9544.jpg)

![4acd14dfd16a4360ab8dfe74cfb69e3f811730f4ab7a2186614df56b29721701.jpg](../iclr_results/3199_Differentiable and Learnable Wireless Simulation with Geometric Transformers/tables/4acd14dfd16a4360ab8dfe74cfb69e3f811730f4ab7a2186614df56b29721701.jpg)

![6d0d80341fe9576653415a7dac1b4f71f7880baef097e27ced2b0ac47138f443.jpg](../iclr_results/3199_Differentiable and Learnable Wireless Simulation with Geometric Transformers/tables/6d0d80341fe9576653415a7dac1b4f71f7880baef097e27ced2b0ac47138f443.jpg)

![9de6eb5095ed8311aa077617d569b0fcf2b5a7ca68a926768adb1002087ff304.jpg](../iclr_results/3199_Differentiable and Learnable Wireless Simulation with Geometric Transformers/tables/9de6eb5095ed8311aa077617d569b0fcf2b5a7ca68a926768adb1002087ff304.jpg)

## Meta Flow Matching: Integrating Vector Fields on the Wasserstein Manifold


### Images

![31522f88a7752d85da3e3a54a21434f2ee2f20800f697b85d56eea758a231dc4.jpg](../iclr_results/3200_Meta Flow Matching_ Integrating Vector Fields on the Wasserstein Manifold/images/31522f88a7752d85da3e3a54a21434f2ee2f20800f697b85d56eea758a231dc4.jpg)

![543e5e7650db59cd1f7c696797be90b9dec9b0f554e682bbaad73367a402f203.jpg](../iclr_results/3200_Meta Flow Matching_ Integrating Vector Fields on the Wasserstein Manifold/images/543e5e7650db59cd1f7c696797be90b9dec9b0f554e682bbaad73367a402f203.jpg)

![72f4dd142030e667ad62c854ef36cca3ea9e4852cbd45641fa05439bf1331401.jpg](../iclr_results/3200_Meta Flow Matching_ Integrating Vector Fields on the Wasserstein Manifold/images/72f4dd142030e667ad62c854ef36cca3ea9e4852cbd45641fa05439bf1331401.jpg)

![82a801859afc4aa28ae404aa5de65667710b41c24537d9c52622773921231711.jpg](../iclr_results/3200_Meta Flow Matching_ Integrating Vector Fields on the Wasserstein Manifold/images/82a801859afc4aa28ae404aa5de65667710b41c24537d9c52622773921231711.jpg)

![8f957f7def40d9fddc978e259cfe6de56f49cfaf5c2a9971069c4c090568a814.jpg](../iclr_results/3200_Meta Flow Matching_ Integrating Vector Fields on the Wasserstein Manifold/images/8f957f7def40d9fddc978e259cfe6de56f49cfaf5c2a9971069c4c090568a814.jpg)

![b234ed0fa922dc174e160c80b0ad55458d5ad538296c913985e6271c3aaf5192.jpg](../iclr_results/3200_Meta Flow Matching_ Integrating Vector Fields on the Wasserstein Manifold/images/b234ed0fa922dc174e160c80b0ad55458d5ad538296c913985e6271c3aaf5192.jpg)

![d06e54d0ba4ccc94957fe9049f5ad71cd34c57b635d445f21afb60cae125e34c.jpg](../iclr_results/3200_Meta Flow Matching_ Integrating Vector Fields on the Wasserstein Manifold/images/d06e54d0ba4ccc94957fe9049f5ad71cd34c57b635d445f21afb60cae125e34c.jpg)

![d0b062510120384159533e2fff5878349545b551d47747a9169ef566ba208193.jpg](../iclr_results/3200_Meta Flow Matching_ Integrating Vector Fields on the Wasserstein Manifold/images/d0b062510120384159533e2fff5878349545b551d47747a9169ef566ba208193.jpg)

### Tables

![22876dcdf18dc0bbbe23f948949f596cc5f81e7d7d89c7b53f39a13da43f1e62.jpg](../iclr_results/3200_Meta Flow Matching_ Integrating Vector Fields on the Wasserstein Manifold/tables/22876dcdf18dc0bbbe23f948949f596cc5f81e7d7d89c7b53f39a13da43f1e62.jpg)

![333cff41ee5532fe68c6c1da1589f5aef5cc8cf238e52937982d58748620b940.jpg](../iclr_results/3200_Meta Flow Matching_ Integrating Vector Fields on the Wasserstein Manifold/tables/333cff41ee5532fe68c6c1da1589f5aef5cc8cf238e52937982d58748620b940.jpg)

![52f43a0380044fa425612ba00cac06907029fb9d7b015f126657f52b8121ba12.jpg](../iclr_results/3200_Meta Flow Matching_ Integrating Vector Fields on the Wasserstein Manifold/tables/52f43a0380044fa425612ba00cac06907029fb9d7b015f126657f52b8121ba12.jpg)

![837d17e8ce205b4d2a25721736fb357eb59dad327c506f73204bb5030209128f.jpg](../iclr_results/3200_Meta Flow Matching_ Integrating Vector Fields on the Wasserstein Manifold/tables/837d17e8ce205b4d2a25721736fb357eb59dad327c506f73204bb5030209128f.jpg)

![ae49485e9c80e22c85e4fb31eea72c2a98933f3032c306ff70c2b384c2f0946c.jpg](../iclr_results/3200_Meta Flow Matching_ Integrating Vector Fields on the Wasserstein Manifold/tables/ae49485e9c80e22c85e4fb31eea72c2a98933f3032c306ff70c2b384c2f0946c.jpg)

![ae8d0138fd1b843d7f1fe8288f2fbd3241b17611de8aa1663550a26a69ddf7d7.jpg](../iclr_results/3200_Meta Flow Matching_ Integrating Vector Fields on the Wasserstein Manifold/tables/ae8d0138fd1b843d7f1fe8288f2fbd3241b17611de8aa1663550a26a69ddf7d7.jpg)

![c04bf4b26b6e789637c9d3a2fc46a5a53d554487509efcdd83b7721a39fe5cc8.jpg](../iclr_results/3200_Meta Flow Matching_ Integrating Vector Fields on the Wasserstein Manifold/tables/c04bf4b26b6e789637c9d3a2fc46a5a53d554487509efcdd83b7721a39fe5cc8.jpg)

![de5c076196350b3ead5ccdd8d8e3e6432a3c3ec22bfddf3ded31bec66b02759e.jpg](../iclr_results/3200_Meta Flow Matching_ Integrating Vector Fields on the Wasserstein Manifold/tables/de5c076196350b3ead5ccdd8d8e3e6432a3c3ec22bfddf3ded31bec66b02759e.jpg)

![e465de9160d7bdab71fd7a868dff0a63487f515fd6b58b926280fc43d0bc580e.jpg](../iclr_results/3200_Meta Flow Matching_ Integrating Vector Fields on the Wasserstein Manifold/tables/e465de9160d7bdab71fd7a868dff0a63487f515fd6b58b926280fc43d0bc580e.jpg)

## Multi-Reward as Condition for Instruction-based Image Editing


### Images

![1a7d7c289ee0671bf605d3f0903855f40d08de8a651aba49f3de795b7d1f5a09.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/images/1a7d7c289ee0671bf605d3f0903855f40d08de8a651aba49f3de795b7d1f5a09.jpg)

![31879a4c633bc3e4f62f080a3e3e30cf66b6f8cc5b6f8b24bbd9d623af938c3f.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/images/31879a4c633bc3e4f62f080a3e3e30cf66b6f8cc5b6f8b24bbd9d623af938c3f.jpg)

![47271044e034f921879a831a53ec69420866a8bce795f0a6554bc278ed8f00e9.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/images/47271044e034f921879a831a53ec69420866a8bce795f0a6554bc278ed8f00e9.jpg)

![6325eb390f8377a163d3b1e59e8d54e72060e0a125ee337aea1813108cfd69fe.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/images/6325eb390f8377a163d3b1e59e8d54e72060e0a125ee337aea1813108cfd69fe.jpg)

![6f0547693c035b8271cf0ef69653813ac0feed78b27d405eac72b46f0520539d.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/images/6f0547693c035b8271cf0ef69653813ac0feed78b27d405eac72b46f0520539d.jpg)

![882548f6b41066499661840e6189eef40900c3e4a6e2e4c6b569427ddcd35867.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/images/882548f6b41066499661840e6189eef40900c3e4a6e2e4c6b569427ddcd35867.jpg)

![8b3dc7f194f2ba7a5d9f702173a5e3946658e2afbf52804eabd9bcbeaf9a3a4b.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/images/8b3dc7f194f2ba7a5d9f702173a5e3946658e2afbf52804eabd9bcbeaf9a3a4b.jpg)

![8d63a6cf24039de1c6b2a8b6722e3b79c71f0cd769c16cba6de3d1058812ed44.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/images/8d63a6cf24039de1c6b2a8b6722e3b79c71f0cd769c16cba6de3d1058812ed44.jpg)

![9805a1759198a3691a2f4484cb5e5cfdb4468b09d9df7e29161ae8a1222731cb.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/images/9805a1759198a3691a2f4484cb5e5cfdb4468b09d9df7e29161ae8a1222731cb.jpg)

![a6b658f8a64efbc1022b3e2a54c57dd43e5d22a19b9f0ff50e4a5dcebb1caef8.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/images/a6b658f8a64efbc1022b3e2a54c57dd43e5d22a19b9f0ff50e4a5dcebb1caef8.jpg)

![b4e8758c78e8ca22ebfde4e12e42149f7cc874026b9e6e69c85d65c0eefab66f.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/images/b4e8758c78e8ca22ebfde4e12e42149f7cc874026b9e6e69c85d65c0eefab66f.jpg)

![f6149fe1c22cd9d592a3a2a4ccc7fd63c697a2398af8fbacd07fca4fb3ebabf0.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/images/f6149fe1c22cd9d592a3a2a4ccc7fd63c697a2398af8fbacd07fca4fb3ebabf0.jpg)

![fc051ac7976d5e3d90e7b869260c7c14c261a0f7cb4462610f52c383d667716e.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/images/fc051ac7976d5e3d90e7b869260c7c14c261a0f7cb4462610f52c383d667716e.jpg)

### Tables

![087d75751147ec0f16bc3409e9736bfddf8861e0e074dfb0da53d25b0763aa9b.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/tables/087d75751147ec0f16bc3409e9736bfddf8861e0e074dfb0da53d25b0763aa9b.jpg)

![14cb228dd3380a8d98a70723962bdd4cca5654b1fcc02b475f90072999089123.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/tables/14cb228dd3380a8d98a70723962bdd4cca5654b1fcc02b475f90072999089123.jpg)

![1d2e2d4d6c203f85d76c84bc0aa494c79e7ce05f3b9e0fdc793cfd3c7dddfde0.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/tables/1d2e2d4d6c203f85d76c84bc0aa494c79e7ce05f3b9e0fdc793cfd3c7dddfde0.jpg)

![1e244e1c01c32acd6b16ef0416b5b6cbe7990f874cf8c136341565a35f9c8dd7.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/tables/1e244e1c01c32acd6b16ef0416b5b6cbe7990f874cf8c136341565a35f9c8dd7.jpg)

![71db34c23ff360ab6eaa4aba64dd8900eef5ba0860d681edf769cadf02e9909e.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/tables/71db34c23ff360ab6eaa4aba64dd8900eef5ba0860d681edf769cadf02e9909e.jpg)

![84da1fe7f3e42531307991571072fd4af363bc3286c968d24a7f989e548cd7e2.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/tables/84da1fe7f3e42531307991571072fd4af363bc3286c968d24a7f989e548cd7e2.jpg)

![8a3a68e64805590b3969245f04af2074dffaabdee91a64bc3c27b41e9a8a2972.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/tables/8a3a68e64805590b3969245f04af2074dffaabdee91a64bc3c27b41e9a8a2972.jpg)

![cc88b937425923b94d185b5b0377360d5d33793f7b342135aa239f1adf94050e.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/tables/cc88b937425923b94d185b5b0377360d5d33793f7b342135aa239f1adf94050e.jpg)

![e89c87839971f2cf7ee2d82b2cec8957a6baae40c69fc3a42e1c812619d0b28a.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/tables/e89c87839971f2cf7ee2d82b2cec8957a6baae40c69fc3a42e1c812619d0b28a.jpg)

![ef45362bf25a07d500b834cc22782e3fdb9f8339bb95a0e9d5c667f51509545e.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/tables/ef45362bf25a07d500b834cc22782e3fdb9f8339bb95a0e9d5c667f51509545e.jpg)

![f20af6baa8e770a35a31eabb4ddb30bd5b138f509cd64bb64008f8a60b024f67.jpg](../iclr_results/3201_Multi-Reward as Condition for Instruction-based Image Editing/tables/f20af6baa8e770a35a31eabb4ddb30bd5b138f509cd64bb64008f8a60b024f67.jpg)

## Montessori-Instruct: Generate Influential Training Data Tailored for Student Learning


### Images

![086333f35d8faf84c5e8106a6c802a586c61707104062d7b1721f026bcf795cf.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/images/086333f35d8faf84c5e8106a6c802a586c61707104062d7b1721f026bcf795cf.jpg)

![2833ab4118c7649269e670af86c5ff187360334504f80df91890c13fa626d04a.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/images/2833ab4118c7649269e670af86c5ff187360334504f80df91890c13fa626d04a.jpg)

![3c4c983c89945d1e217039c6ed21f9108a8f2bca9462085444e58feb34d316c3.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/images/3c4c983c89945d1e217039c6ed21f9108a8f2bca9462085444e58feb34d316c3.jpg)

![5457e56615e8dd52a1f9944a961c60df8d05cb92ad3895a6c2dde9f596c13b1e.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/images/5457e56615e8dd52a1f9944a961c60df8d05cb92ad3895a6c2dde9f596c13b1e.jpg)

![6c3bc76746cf962465349a58099a43d2e205ccd3d2da2b9f777906b4f897b90d.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/images/6c3bc76746cf962465349a58099a43d2e205ccd3d2da2b9f777906b4f897b90d.jpg)

![6c99d79bed2f78cdcc9ed8be7f1807b2a15a3b7fdff508d66e7f5f55acb2a13f.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/images/6c99d79bed2f78cdcc9ed8be7f1807b2a15a3b7fdff508d66e7f5f55acb2a13f.jpg)

![6f6ee45b4b1c0e8096a317b10e64bc88e28c9a60c6a44429adc6c3e768afef3a.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/images/6f6ee45b4b1c0e8096a317b10e64bc88e28c9a60c6a44429adc6c3e768afef3a.jpg)

![811bfee1816bb86597c653fca1b7de38036c2eea32c8099338081cc4ae9c56ec.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/images/811bfee1816bb86597c653fca1b7de38036c2eea32c8099338081cc4ae9c56ec.jpg)

![8c2a6cfb5e243057e62cbfbf5f6317674568330caf81bee7eb54c8b26f932fbf.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/images/8c2a6cfb5e243057e62cbfbf5f6317674568330caf81bee7eb54c8b26f932fbf.jpg)

![905f11ee928abbbba33d07a09945d7774293fefe8ac63bccaa34172b1fb64862.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/images/905f11ee928abbbba33d07a09945d7774293fefe8ac63bccaa34172b1fb64862.jpg)

![a1c77d76094468a6d6ac2b0f04c4a88fbe0a0f4cb69202e71bc5d444976a3a78.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/images/a1c77d76094468a6d6ac2b0f04c4a88fbe0a0f4cb69202e71bc5d444976a3a78.jpg)

![a2e55adcf22f696ee429c8cb16ec1d2639cada3c03e5055995ff98da56aae80f.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/images/a2e55adcf22f696ee429c8cb16ec1d2639cada3c03e5055995ff98da56aae80f.jpg)

![bedf13fb5c61d36a95e37a810b27de457e26dfd5ae7704594960fb5c842a0995.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/images/bedf13fb5c61d36a95e37a810b27de457e26dfd5ae7704594960fb5c842a0995.jpg)

![c65e09f583c047cdd159a787a9c2eb105fb33003a80e731aa754f07089342632.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/images/c65e09f583c047cdd159a787a9c2eb105fb33003a80e731aa754f07089342632.jpg)

![f64259b31f2df47c806c88db861779bc02095865480bf6e673f21a842870dcde.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/images/f64259b31f2df47c806c88db861779bc02095865480bf6e673f21a842870dcde.jpg)

![fbc40725dffef8e98e8d112b02899d427502bf52c90b570a8cf2eaae2581c798.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/images/fbc40725dffef8e98e8d112b02899d427502bf52c90b570a8cf2eaae2581c798.jpg)

### Tables

![043af343cb50775d89bbad0f0fd4876ff617d196ad2f5d717edb6382a0a1a558.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/tables/043af343cb50775d89bbad0f0fd4876ff617d196ad2f5d717edb6382a0a1a558.jpg)

![150d1df513e6b7579996e4f711bed9a5fcf50351da0a3a4c946330b6e94aa06f.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/tables/150d1df513e6b7579996e4f711bed9a5fcf50351da0a3a4c946330b6e94aa06f.jpg)

![24007e2cdf8afe40729f6e6a67a1771d9726ecd1bf27dcbdc739705acc6f624f.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/tables/24007e2cdf8afe40729f6e6a67a1771d9726ecd1bf27dcbdc739705acc6f624f.jpg)

![25ae437ca7b2c9be9706cf7d192a5b84d60d2f7a743175742272708b7ab8627f.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/tables/25ae437ca7b2c9be9706cf7d192a5b84d60d2f7a743175742272708b7ab8627f.jpg)

![3ad07acac29e90f1cb260819efe49eafdb50a96c333ef5b274af6f2f7a0d6cb5.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/tables/3ad07acac29e90f1cb260819efe49eafdb50a96c333ef5b274af6f2f7a0d6cb5.jpg)

![90de705855dabdb95eaea1712bac5cab4ce5fc1b4efb0c14d0258a61cb0e5334.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/tables/90de705855dabdb95eaea1712bac5cab4ce5fc1b4efb0c14d0258a61cb0e5334.jpg)

![9a470354c7aadff2e1d88ff2cda3c76cb8165437c5603b6f179d4ef16e496909.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/tables/9a470354c7aadff2e1d88ff2cda3c76cb8165437c5603b6f179d4ef16e496909.jpg)

![b35f44db0a71e7790dc3a671fd89f595f69d0f7a8cbaa4c80e17502addb37cc7.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/tables/b35f44db0a71e7790dc3a671fd89f595f69d0f7a8cbaa4c80e17502addb37cc7.jpg)

![d75b07067193d6efecce65c6ff302e874d2dd987d51dfbe3b0329142cfe4c9ef.jpg](../iclr_results/3202_Montessori-Instruct_ Generate Influential Training Data Tailored for Student Learning/tables/d75b07067193d6efecce65c6ff302e874d2dd987d51dfbe3b0329142cfe4c9ef.jpg)

## Infilling Score: A Pretraining Data Detection Algorithm for Large Language Models


### Images

![0c875e82f04515a7a909fbc6a587bd0804128e46f08c185c5a28314a8324923e.jpg](../iclr_results/3203_Infilling Score_ A Pretraining Data Detection Algorithm for Large Language Models/images/0c875e82f04515a7a909fbc6a587bd0804128e46f08c185c5a28314a8324923e.jpg)

![c1db3adba3a73092718255a86a6be3c4ef92e71206e58c11d3f4f43d3a2208df.jpg](../iclr_results/3203_Infilling Score_ A Pretraining Data Detection Algorithm for Large Language Models/images/c1db3adba3a73092718255a86a6be3c4ef92e71206e58c11d3f4f43d3a2208df.jpg)

### Tables

![0e3dd0e1f0ac14e529e966a507cdc8c9930e95bb46ec46bf857786813e79d1a1.jpg](../iclr_results/3203_Infilling Score_ A Pretraining Data Detection Algorithm for Large Language Models/tables/0e3dd0e1f0ac14e529e966a507cdc8c9930e95bb46ec46bf857786813e79d1a1.jpg)

![49b0be48a494f54979d3689183220428ffa6687f77eccc906562240919789149.jpg](../iclr_results/3203_Infilling Score_ A Pretraining Data Detection Algorithm for Large Language Models/tables/49b0be48a494f54979d3689183220428ffa6687f77eccc906562240919789149.jpg)

![4d680cb2d9f58f42eb1f3fe3c8e643899ab6901e26b1dbb2d72e4e4a17c2111f.jpg](../iclr_results/3203_Infilling Score_ A Pretraining Data Detection Algorithm for Large Language Models/tables/4d680cb2d9f58f42eb1f3fe3c8e643899ab6901e26b1dbb2d72e4e4a17c2111f.jpg)

![7dda35bbf56e4a86ef94253986c42e150a0914e4a67bd2b6b6d42c9bad3d1fa5.jpg](../iclr_results/3203_Infilling Score_ A Pretraining Data Detection Algorithm for Large Language Models/tables/7dda35bbf56e4a86ef94253986c42e150a0914e4a67bd2b6b6d42c9bad3d1fa5.jpg)

![87895b85fce8628a3d15ccf3aa79ffa242e8d12258dac7c27870704bec4a4a87.jpg](../iclr_results/3203_Infilling Score_ A Pretraining Data Detection Algorithm for Large Language Models/tables/87895b85fce8628a3d15ccf3aa79ffa242e8d12258dac7c27870704bec4a4a87.jpg)

![93ef9a76171dcbb58e6f556c9f65ff0dcf8d9fd07dd4eaa3fe8b52c9f7090f74.jpg](../iclr_results/3203_Infilling Score_ A Pretraining Data Detection Algorithm for Large Language Models/tables/93ef9a76171dcbb58e6f556c9f65ff0dcf8d9fd07dd4eaa3fe8b52c9f7090f74.jpg)

![9c84d961ae8db80a4f687b2f8ccd97ed2bd22759c52f92e63541638647e32346.jpg](../iclr_results/3203_Infilling Score_ A Pretraining Data Detection Algorithm for Large Language Models/tables/9c84d961ae8db80a4f687b2f8ccd97ed2bd22759c52f92e63541638647e32346.jpg)

![b7708c286f270a80f5645a8268a8062fc7214b5f056689994a8bd2475d79911c.jpg](../iclr_results/3203_Infilling Score_ A Pretraining Data Detection Algorithm for Large Language Models/tables/b7708c286f270a80f5645a8268a8062fc7214b5f056689994a8bd2475d79911c.jpg)

![ec6d3fed3ffd8df86e34d8c8ec02e463eb8da22e2d54ab0ea37ebd0a7e36ab6b.jpg](../iclr_results/3203_Infilling Score_ A Pretraining Data Detection Algorithm for Large Language Models/tables/ec6d3fed3ffd8df86e34d8c8ec02e463eb8da22e2d54ab0ea37ebd0a7e36ab6b.jpg)

![f106b23587d4b25f4cc933bcdfacb60f7e466374d64ad57f51e87b6251392f3e.jpg](../iclr_results/3203_Infilling Score_ A Pretraining Data Detection Algorithm for Large Language Models/tables/f106b23587d4b25f4cc933bcdfacb60f7e466374d64ad57f51e87b6251392f3e.jpg)

![f656a02bd9e3f7c78ac872db611843fedc55e6610d27e1bd34142d432b313ac8.jpg](../iclr_results/3203_Infilling Score_ A Pretraining Data Detection Algorithm for Large Language Models/tables/f656a02bd9e3f7c78ac872db611843fedc55e6610d27e1bd34142d432b313ac8.jpg)

![fb740181f76349ee3b21241929b0a92cda2caef6b7604547a744c81dbd05c4d1.jpg](../iclr_results/3203_Infilling Score_ A Pretraining Data Detection Algorithm for Large Language Models/tables/fb740181f76349ee3b21241929b0a92cda2caef6b7604547a744c81dbd05c4d1.jpg)

## DiffusionGuard: A Robust Defense Against Malicious Diffusion-based Image Editing


### Images

![0cc410001bf4fd226e853f115b0a030212106bc0c4847a15fc997f540e8e1611.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/0cc410001bf4fd226e853f115b0a030212106bc0c4847a15fc997f540e8e1611.jpg)

![14a9ff1a506ae9793f92780cd3d67929c46d517a42aaa4bc422e5d074ad601dd.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/14a9ff1a506ae9793f92780cd3d67929c46d517a42aaa4bc422e5d074ad601dd.jpg)

![22647faa95d1352f26baae4c84b8335f2e086492758ae7a8b9409afa15811f04.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/22647faa95d1352f26baae4c84b8335f2e086492758ae7a8b9409afa15811f04.jpg)

![2a210b1c3c39b030aa163092db260bfd44059e68c6fd407059565a407b37cc48.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/2a210b1c3c39b030aa163092db260bfd44059e68c6fd407059565a407b37cc48.jpg)

![2e6bed9524a861b82453c07efe9f324eff247e10e48033a3e0edd0a36d4c8000.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/2e6bed9524a861b82453c07efe9f324eff247e10e48033a3e0edd0a36d4c8000.jpg)

![2fc47710609c7fe56e0b577ee28eb8532e2bfeb605a9e2f8ca14e44142a6c19d.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/2fc47710609c7fe56e0b577ee28eb8532e2bfeb605a9e2f8ca14e44142a6c19d.jpg)

![3102c3e06bcbf96a0a7dd61af322bcf9ff827f6db75f488c785937a4bb768fb5.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/3102c3e06bcbf96a0a7dd61af322bcf9ff827f6db75f488c785937a4bb768fb5.jpg)

![3857c6b998ef3e52c92f4ec84e752db3e1becc3b3924b9aa24d36ed2d896bc3a.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/3857c6b998ef3e52c92f4ec84e752db3e1becc3b3924b9aa24d36ed2d896bc3a.jpg)

![3f4fb1c218553cfd4dd7b040fc76754159db594d758c4b7e54c458ab407e63d0.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/3f4fb1c218553cfd4dd7b040fc76754159db594d758c4b7e54c458ab407e63d0.jpg)

![4563a1dbfa95dbdeffa9c166b0a3dcb1e5b22a5283d2730000828358ef16c35d.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/4563a1dbfa95dbdeffa9c166b0a3dcb1e5b22a5283d2730000828358ef16c35d.jpg)

![4c7595b609111abdac82761706b23c342d7ecf9c3a9e8a1e1cb6d816a45b7fe2.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/4c7595b609111abdac82761706b23c342d7ecf9c3a9e8a1e1cb6d816a45b7fe2.jpg)

![591849b0bdaaa06cbc8ad42cb803cbc96bddaf283a10b8ba07d9d4da12e2937f.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/591849b0bdaaa06cbc8ad42cb803cbc96bddaf283a10b8ba07d9d4da12e2937f.jpg)

![655675314a62b694304a8a6b294cc7c7be1c7c39df21504a46d7b385c26a7e32.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/655675314a62b694304a8a6b294cc7c7be1c7c39df21504a46d7b385c26a7e32.jpg)

![71f2df9618ac50e2b8d42c8ebbbd443fec7ad3618d3b5e1b1b5e4b8057f83618.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/71f2df9618ac50e2b8d42c8ebbbd443fec7ad3618d3b5e1b1b5e4b8057f83618.jpg)

![727c028c159805ffe812dc2bde152c6c3d7e32fa8023f0f28b3d988fb4597b08.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/727c028c159805ffe812dc2bde152c6c3d7e32fa8023f0f28b3d988fb4597b08.jpg)

![8b9d47f2f273e49dedf933e911f80c3a4c5ca55d5e0f2205eddc2ff3a8c31fcf.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/8b9d47f2f273e49dedf933e911f80c3a4c5ca55d5e0f2205eddc2ff3a8c31fcf.jpg)

![8fe1c0b661c8ab7b83874d6e1d70d2a8e8a776086104228bbf2a4d4e4426947a.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/8fe1c0b661c8ab7b83874d6e1d70d2a8e8a776086104228bbf2a4d4e4426947a.jpg)

![90a11970affd8c1e3f50e0b60c3efc181454752bd57a1e47e33176aca3066c1d.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/90a11970affd8c1e3f50e0b60c3efc181454752bd57a1e47e33176aca3066c1d.jpg)

![91c0a847e7e729def85575fbdc8357c104cb3c3a416f90fb84ba86cb7429a94c.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/91c0a847e7e729def85575fbdc8357c104cb3c3a416f90fb84ba86cb7429a94c.jpg)

![992bea1c03eee6adb56f0a94be8045bb5b67eced5fb74fc2337f71760494cc67.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/992bea1c03eee6adb56f0a94be8045bb5b67eced5fb74fc2337f71760494cc67.jpg)

![aa1c6224b40f32a918585cade1288589df6e90b1289e45853311e58836e7749f.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/aa1c6224b40f32a918585cade1288589df6e90b1289e45853311e58836e7749f.jpg)

![ad3a2931a1f43fc068896db2d369dc56c358acce91b5ddadf33e14d8f9f97c7e.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/ad3a2931a1f43fc068896db2d369dc56c358acce91b5ddadf33e14d8f9f97c7e.jpg)

![c51f6152c2526bb9ad57d7085bcc256d960172144c3fc28a8d33d4a96ec18e02.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/c51f6152c2526bb9ad57d7085bcc256d960172144c3fc28a8d33d4a96ec18e02.jpg)

![c970b5b1bd23cff1beeba6a7cd9e40e5ac9b536290ff9793c562251ba30a2f56.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/c970b5b1bd23cff1beeba6a7cd9e40e5ac9b536290ff9793c562251ba30a2f56.jpg)

![cda5a52290a93471546c9ddd28272740582a89b39014123285196a61af6c02a7.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/cda5a52290a93471546c9ddd28272740582a89b39014123285196a61af6c02a7.jpg)

![dcda9abf13e81b9c4a8a720438b306d0bf363678e6033eec5fbf3c55a6d0641f.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/dcda9abf13e81b9c4a8a720438b306d0bf363678e6033eec5fbf3c55a6d0641f.jpg)

![e0c110311bf4bfd0e0394f1db6cfe758b0a3f8a9a0ed36f3f415cb3d9ffedadd.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/e0c110311bf4bfd0e0394f1db6cfe758b0a3f8a9a0ed36f3f415cb3d9ffedadd.jpg)

![e68caccf6e3034900eebcb0ea9731b486e13eb2502f12270aa790f4d8084fca4.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/e68caccf6e3034900eebcb0ea9731b486e13eb2502f12270aa790f4d8084fca4.jpg)

![e97cec58a185d5a9c4f68d46273a8f732624a81738eef329ff7b2bf5551d8699.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/e97cec58a185d5a9c4f68d46273a8f732624a81738eef329ff7b2bf5551d8699.jpg)

![ebde33c2a445476ddda3fec6d8b0a8994a78d31d0006bade1850290c41300658.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/ebde33c2a445476ddda3fec6d8b0a8994a78d31d0006bade1850290c41300658.jpg)

![f21e9b2585a2333b96356f8063a202816a27959de0b8764593d353bd3d200f33.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/f21e9b2585a2333b96356f8063a202816a27959de0b8764593d353bd3d200f33.jpg)

![f44e84d1309d0394dca1e8197050e0c4013c6471c946c5cc6bb6e3210eaeb78c.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/f44e84d1309d0394dca1e8197050e0c4013c6471c946c5cc6bb6e3210eaeb78c.jpg)

![fffcd0a02d2e5c38993034e46145143ccf8f0fe32a4f6564e8d1216a888bd310.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/images/fffcd0a02d2e5c38993034e46145143ccf8f0fe32a4f6564e8d1216a888bd310.jpg)

### Tables

![0fd2b26ced35c5f1156e1aa8e92e52ae26a1c92c22961ebdc6529cbdee547e84.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/tables/0fd2b26ced35c5f1156e1aa8e92e52ae26a1c92c22961ebdc6529cbdee547e84.jpg)

![251aefaaf37b84c246b1a473b02bab7a37d0f6307f8303399dad6cad7d863c6c.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/tables/251aefaaf37b84c246b1a473b02bab7a37d0f6307f8303399dad6cad7d863c6c.jpg)

![2d68c9e96aa20f1948dd291adf3b590ebde3199c19757a3dad5a7047d39cd1e5.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/tables/2d68c9e96aa20f1948dd291adf3b590ebde3199c19757a3dad5a7047d39cd1e5.jpg)

![49828f79717c5d2cd3b2baa9c8799354aca9eb3e47fc803fe25c493eac79f049.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/tables/49828f79717c5d2cd3b2baa9c8799354aca9eb3e47fc803fe25c493eac79f049.jpg)

![6b1925eb0a28a7ae92de07cc62e015d340838eb651722da5f2393439fe756442.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/tables/6b1925eb0a28a7ae92de07cc62e015d340838eb651722da5f2393439fe756442.jpg)

![8e246df7c3d87308261caa74ebc2b9ee4d3f6e54c348bb920029eb80df909fe0.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/tables/8e246df7c3d87308261caa74ebc2b9ee4d3f6e54c348bb920029eb80df909fe0.jpg)

![9037d3cd4459eee7d417922d0a7510dce6fee51b978065c931d84dbfc07d0f9c.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/tables/9037d3cd4459eee7d417922d0a7510dce6fee51b978065c931d84dbfc07d0f9c.jpg)

![95c26b8cd85d33d4e49b77cb881e74aeea7027b7655d489d59532d2b0e69637e.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/tables/95c26b8cd85d33d4e49b77cb881e74aeea7027b7655d489d59532d2b0e69637e.jpg)

![a439bc2a5b0f4f43934a5b45906acf90efae0f6aed1e01cb8b95ed84a3076b81.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/tables/a439bc2a5b0f4f43934a5b45906acf90efae0f6aed1e01cb8b95ed84a3076b81.jpg)

![be8e28da7db134955f2407e5c6bab054406e102bcb05194db43a372422669751.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/tables/be8e28da7db134955f2407e5c6bab054406e102bcb05194db43a372422669751.jpg)

![e43e00601a494a2c9f11085d2462a457b52e319ee3ae6d309f52cc17bb5aaab7.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/tables/e43e00601a494a2c9f11085d2462a457b52e319ee3ae6d309f52cc17bb5aaab7.jpg)

![f5816f31debced44d162e3621d24c83e9bdd4bd34aa6fe5c032e102d756f9538.jpg](../iclr_results/3204_DiffusionGuard_ A Robust Defense Against Malicious Diffusion-based Image Editing/tables/f5816f31debced44d162e3621d24c83e9bdd4bd34aa6fe5c032e102d756f9538.jpg)

## LLMOPT: Learning to Define and Solve General Optimization Problems from Scratch


### Images

![0041aefffc78a39d000a956b52829f29d2a74e5a69ec4473bd681dd25f01b1dd.jpg](../iclr_results/3205_LLMOPT_ Learning to Define and Solve General Optimization Problems from Scratch/images/0041aefffc78a39d000a956b52829f29d2a74e5a69ec4473bd681dd25f01b1dd.jpg)

![027c1df557fe5c56926674f57decdd9bc4c4be6d260643e774d62832db498435.jpg](../iclr_results/3205_LLMOPT_ Learning to Define and Solve General Optimization Problems from Scratch/images/027c1df557fe5c56926674f57decdd9bc4c4be6d260643e774d62832db498435.jpg)

![2ff3971569d9dbb195c43cf1d9d894d99e3cadd15e64ca6cf8998eff36802ce4.jpg](../iclr_results/3205_LLMOPT_ Learning to Define and Solve General Optimization Problems from Scratch/images/2ff3971569d9dbb195c43cf1d9d894d99e3cadd15e64ca6cf8998eff36802ce4.jpg)

![62015ffec364a908b3bcecffc8ea8071db42e3db18fbdbfc88220891542da1bd.jpg](../iclr_results/3205_LLMOPT_ Learning to Define and Solve General Optimization Problems from Scratch/images/62015ffec364a908b3bcecffc8ea8071db42e3db18fbdbfc88220891542da1bd.jpg)

![7b24f210ce5dfe6dfa8f8ac09fa233c6ba9f6cfd096f65c9a80f97e4c9f52ca3.jpg](../iclr_results/3205_LLMOPT_ Learning to Define and Solve General Optimization Problems from Scratch/images/7b24f210ce5dfe6dfa8f8ac09fa233c6ba9f6cfd096f65c9a80f97e4c9f52ca3.jpg)

![c8447cd0693c2da570908f1430b25f4a399946d2cd9accc8555f984552b08b5d.jpg](../iclr_results/3205_LLMOPT_ Learning to Define and Solve General Optimization Problems from Scratch/images/c8447cd0693c2da570908f1430b25f4a399946d2cd9accc8555f984552b08b5d.jpg)

![f449f236f3f43b16f88b848c7f2be2b97c1b557b46e148a7b1d133a04d24ee48.jpg](../iclr_results/3205_LLMOPT_ Learning to Define and Solve General Optimization Problems from Scratch/images/f449f236f3f43b16f88b848c7f2be2b97c1b557b46e148a7b1d133a04d24ee48.jpg)

![fbadef2317bc9deb5711484e956fb8d1bdd7ed0d8c2c5085d8f9ea78df089a96.jpg](../iclr_results/3205_LLMOPT_ Learning to Define and Solve General Optimization Problems from Scratch/images/fbadef2317bc9deb5711484e956fb8d1bdd7ed0d8c2c5085d8f9ea78df089a96.jpg)

### Tables

![00a3f3a900dc89dd4626ce5ec205e6799d934d39eacb32856fa9aee3c1c69de2.jpg](../iclr_results/3205_LLMOPT_ Learning to Define and Solve General Optimization Problems from Scratch/tables/00a3f3a900dc89dd4626ce5ec205e6799d934d39eacb32856fa9aee3c1c69de2.jpg)

![10bd587e13f0b610da64865b94207fb00703c37f7fd45ccb3cbfcfd4498fe21e.jpg](../iclr_results/3205_LLMOPT_ Learning to Define and Solve General Optimization Problems from Scratch/tables/10bd587e13f0b610da64865b94207fb00703c37f7fd45ccb3cbfcfd4498fe21e.jpg)

![27b4b4f678ebd43fa2cf897b1e508521002e025440975555485416a22ac541e3.jpg](../iclr_results/3205_LLMOPT_ Learning to Define and Solve General Optimization Problems from Scratch/tables/27b4b4f678ebd43fa2cf897b1e508521002e025440975555485416a22ac541e3.jpg)

![29394672f71515211caa3323c468c067aed6cd7d56e566744399ae75cd63b655.jpg](../iclr_results/3205_LLMOPT_ Learning to Define and Solve General Optimization Problems from Scratch/tables/29394672f71515211caa3323c468c067aed6cd7d56e566744399ae75cd63b655.jpg)

![2e9c523aaefb1ec27bc83ad355b22f82e648f8eca73efc6f6f9543c5fdff06de.jpg](../iclr_results/3205_LLMOPT_ Learning to Define and Solve General Optimization Problems from Scratch/tables/2e9c523aaefb1ec27bc83ad355b22f82e648f8eca73efc6f6f9543c5fdff06de.jpg)

![3015b676d7c3279b83ae4ca621787c3a6edcf6d93afc82654aeaefcdbc4bde71.jpg](../iclr_results/3205_LLMOPT_ Learning to Define and Solve General Optimization Problems from Scratch/tables/3015b676d7c3279b83ae4ca621787c3a6edcf6d93afc82654aeaefcdbc4bde71.jpg)

![33938475b903ff4383c16494ad47f03db529e74d7d90f3b5b08f3b2e72cf550b.jpg](../iclr_results/3205_LLMOPT_ Learning to Define and Solve General Optimization Problems from Scratch/tables/33938475b903ff4383c16494ad47f03db529e74d7d90f3b5b08f3b2e72cf550b.jpg)

![8173166d5fd9266d1333d527120273f89b6503d8a56448ff868a13c3c8da022c.jpg](../iclr_results/3205_LLMOPT_ Learning to Define and Solve General Optimization Problems from Scratch/tables/8173166d5fd9266d1333d527120273f89b6503d8a56448ff868a13c3c8da022c.jpg)

![88e8156670ad383ba020e51955992e790d33b06ef3681bb385a4364813301191.jpg](../iclr_results/3205_LLMOPT_ Learning to Define and Solve General Optimization Problems from Scratch/tables/88e8156670ad383ba020e51955992e790d33b06ef3681bb385a4364813301191.jpg)

![ba1a7d1c727ad287007e6c9cd3845fd9b9d6cd595b9a495c9ab057426c952221.jpg](../iclr_results/3205_LLMOPT_ Learning to Define and Solve General Optimization Problems from Scratch/tables/ba1a7d1c727ad287007e6c9cd3845fd9b9d6cd595b9a495c9ab057426c952221.jpg)

![f2b8a3cf789d3d49b48ce8c88c1b8cf1353df67bb6bcaf970b282a91625efc55.jpg](../iclr_results/3205_LLMOPT_ Learning to Define and Solve General Optimization Problems from Scratch/tables/f2b8a3cf789d3d49b48ce8c88c1b8cf1353df67bb6bcaf970b282a91625efc55.jpg)

## Controllable Unlearning for Image-to-Image Generative Models via $\epsilon$-Constrained Optimization


### Images

![0961b5c051cbb5087bea557ea35d15518070f1ee656019a92fcfc12b94f800d9.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/0961b5c051cbb5087bea557ea35d15518070f1ee656019a92fcfc12b94f800d9.jpg)

![141b488a0d4cbaf6f460687d238e2b3ad66b7116e2d00f61fcfa6ef73cefd18c.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/141b488a0d4cbaf6f460687d238e2b3ad66b7116e2d00f61fcfa6ef73cefd18c.jpg)

![161b50801dd5abd023c5f876208a99ceac239f2161cc6c45b9cb84814bb5101d.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/161b50801dd5abd023c5f876208a99ceac239f2161cc6c45b9cb84814bb5101d.jpg)

![19c40e1b587f72720f58d7fa4485990a0b2b700bdc006b3a6673ebc51e0908f3.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/19c40e1b587f72720f58d7fa4485990a0b2b700bdc006b3a6673ebc51e0908f3.jpg)

![1d59baf0fa7fec560686dcd15defb6a12f773a9ee90930cb9beed9e6e77227f3.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/1d59baf0fa7fec560686dcd15defb6a12f773a9ee90930cb9beed9e6e77227f3.jpg)

![1e607980a53fd56cefb35d0beb9ef0fbd726e60b1a86d3fbf76c91283b84bd69.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/1e607980a53fd56cefb35d0beb9ef0fbd726e60b1a86d3fbf76c91283b84bd69.jpg)

![1e6ae8aa0c568c70cccb6e2fc0e9c72c57d6bdf0434dbc16523c53dc0fe1f4cd.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/1e6ae8aa0c568c70cccb6e2fc0e9c72c57d6bdf0434dbc16523c53dc0fe1f4cd.jpg)

![21a9d7ae9b2b0cd0b051ba037bd159cced7a126294815765e380c99c5eb9ac1f.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/21a9d7ae9b2b0cd0b051ba037bd159cced7a126294815765e380c99c5eb9ac1f.jpg)

![2bd16c1039162e4c39ecfa85677958073e43d1fe76d9014963064244c47b2f2f.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/2bd16c1039162e4c39ecfa85677958073e43d1fe76d9014963064244c47b2f2f.jpg)

![2e4bb68baff6b02a32884caae9fd577c9c74bccfba0d9289f96404a7c2682c1e.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/2e4bb68baff6b02a32884caae9fd577c9c74bccfba0d9289f96404a7c2682c1e.jpg)

![2f2b09ee505e289859a17702b35fd7216f55384c2c82cd473771324c6dd99255.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/2f2b09ee505e289859a17702b35fd7216f55384c2c82cd473771324c6dd99255.jpg)

![3f815d855fe8b6afdc915aaa857e3ccf2fb7f625cfa40663bca27c2c9413a6a4.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/3f815d855fe8b6afdc915aaa857e3ccf2fb7f625cfa40663bca27c2c9413a6a4.jpg)

![48f08aeffe4a4c62964b4d1852e57f7fe72afd0af32acd59cb728af1113d2eb4.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/48f08aeffe4a4c62964b4d1852e57f7fe72afd0af32acd59cb728af1113d2eb4.jpg)

![60aad4561a5eebead3dc9436d21c832c8ce0397c357fc961a9e4294ce9c2a1c1.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/60aad4561a5eebead3dc9436d21c832c8ce0397c357fc961a9e4294ce9c2a1c1.jpg)

![67230be96d6e471c7cbbef0553f531808917705424c20b1012d849045d4655fa.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/67230be96d6e471c7cbbef0553f531808917705424c20b1012d849045d4655fa.jpg)

![723ab45c86f4d87457db04258d7108d5fbdc181f2281b928dfef57647fe81e7d.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/723ab45c86f4d87457db04258d7108d5fbdc181f2281b928dfef57647fe81e7d.jpg)

![78109a4ed43f7a081821edf55f1ecda906aa353976266ee3eb31112704689123.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/78109a4ed43f7a081821edf55f1ecda906aa353976266ee3eb31112704689123.jpg)

![8851c444778ef3895678024e19972a35989871f0b742e4a705a0f082f407af18.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/8851c444778ef3895678024e19972a35989871f0b742e4a705a0f082f407af18.jpg)

![95950caa4ed2be5f379d59b2dc1da0a6ae0c0d3f119311b4fd719e8a88b1f390.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/95950caa4ed2be5f379d59b2dc1da0a6ae0c0d3f119311b4fd719e8a88b1f390.jpg)

![a102fc37c6921270dd352ecf7f8cb9e41591ae42992b8421f2a6b36888580f9b.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/a102fc37c6921270dd352ecf7f8cb9e41591ae42992b8421f2a6b36888580f9b.jpg)

![b49403378419c3190d1c67b921a735cf93f897c5d12923cd39c3e2c236bc6677.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/b49403378419c3190d1c67b921a735cf93f897c5d12923cd39c3e2c236bc6677.jpg)

![bd6009a5e196fa3ac0d67f7fb55798ef946242c7f41c35cf13281dbea3531281.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/bd6009a5e196fa3ac0d67f7fb55798ef946242c7f41c35cf13281dbea3531281.jpg)

![d4363760aad47d6f7e5fbcb3f542a1609fbfabfe02c110386215458fb7be0b31.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/d4363760aad47d6f7e5fbcb3f542a1609fbfabfe02c110386215458fb7be0b31.jpg)

![f9138f63c3e7a385c8a96068425e82f7ac87991a4f0e410fba63fc42791874c5.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/images/f9138f63c3e7a385c8a96068425e82f7ac87991a4f0e410fba63fc42791874c5.jpg)

### Tables

![270add4b641a70e737d283d92e6dab6e3f2cd719c99abc8a91742482ef011614.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/tables/270add4b641a70e737d283d92e6dab6e3f2cd719c99abc8a91742482ef011614.jpg)

![3d6345b7298270a25b93c14c71c3e581f64218643535092ff56954795c0cf887.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/tables/3d6345b7298270a25b93c14c71c3e581f64218643535092ff56954795c0cf887.jpg)

![d03cfd4a2672cc03b3c66a3e8b462bad4afe3e952f1730013fbec716dea9ee87.jpg](../iclr_results/3206_Controllable Unlearning for Image-to-Image Generative Models via $_epsilon$-Constrained Optimization/tables/d03cfd4a2672cc03b3c66a3e8b462bad4afe3e952f1730013fbec716dea9ee87.jpg)

## IRIS: LLM-Assisted Static Analysis for Detecting Security Vulnerabilities


### Images

![02559caf3ea62cee9437844c709dcdfc28eb8bc65f0976318df9813aa0d309c3.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/images/02559caf3ea62cee9437844c709dcdfc28eb8bc65f0976318df9813aa0d309c3.jpg)

![13b9e816d224aff58d8218efb0b3ea1686d974c7d016ccb105315a837bf0559e.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/images/13b9e816d224aff58d8218efb0b3ea1686d974c7d016ccb105315a837bf0559e.jpg)

![2bcee8048a2117859771fcf9518ea4d6ed688cc558c1089cf2ba7994400a7c47.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/images/2bcee8048a2117859771fcf9518ea4d6ed688cc558c1089cf2ba7994400a7c47.jpg)

![5fdcd5d38100b039b1bfd11fe00520b923c1c2c4f79f22f8dd9d1ffa8329e664.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/images/5fdcd5d38100b039b1bfd11fe00520b923c1c2c4f79f22f8dd9d1ffa8329e664.jpg)

![77d7970962b45f4460dfe6a9ef87917524172f8f9a361c1f2114fd52963b0766.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/images/77d7970962b45f4460dfe6a9ef87917524172f8f9a361c1f2114fd52963b0766.jpg)

![84c6eb4188b3d0df26d7807c3abb5e6e63f244013d2cdcb86de90112a845db1e.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/images/84c6eb4188b3d0df26d7807c3abb5e6e63f244013d2cdcb86de90112a845db1e.jpg)

![adcce285889a71e1f2d69cf6814878aae8a96df3fef9c99a19c5a6b86c67549b.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/images/adcce285889a71e1f2d69cf6814878aae8a96df3fef9c99a19c5a6b86c67549b.jpg)

![b91c02bbf35f376e845999c06ca04955ea3676a538872ec62634ced9c0765073.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/images/b91c02bbf35f376e845999c06ca04955ea3676a538872ec62634ced9c0765073.jpg)

![bc08362eb3fb5e05e8fb67c27a1f95fc1f559d96fd2bb3db66fa82b9d5855841.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/images/bc08362eb3fb5e05e8fb67c27a1f95fc1f559d96fd2bb3db66fa82b9d5855841.jpg)

![c860a1f5cb0ff484f4c6872ee2b8093167bd3bbb27232da532a86fea6bbade8f.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/images/c860a1f5cb0ff484f4c6872ee2b8093167bd3bbb27232da532a86fea6bbade8f.jpg)

![cbd8646747fe5cef628ad2a20451d1775242fc070975ab14b33168e1ac4ef2cc.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/images/cbd8646747fe5cef628ad2a20451d1775242fc070975ab14b33168e1ac4ef2cc.jpg)

![d325a67717f54f1ddd5f90b8c707c9154c49567be3d37d5a7e440c5e33aff79a.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/images/d325a67717f54f1ddd5f90b8c707c9154c49567be3d37d5a7e440c5e33aff79a.jpg)

![f5e669d7db41634851d7ef630640c57d203bd941401d95469988b1c76ed7a6f9.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/images/f5e669d7db41634851d7ef630640c57d203bd941401d95469988b1c76ed7a6f9.jpg)

### Tables

![00ddf6e106faa32484dfb0a516bca538896b173efe1afc64db3d63d22425db2f.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/tables/00ddf6e106faa32484dfb0a516bca538896b173efe1afc64db3d63d22425db2f.jpg)

![0591664807bb7a86ca5c7cf54eda9012b8799eb662c4dfe9504d52b332ec3526.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/tables/0591664807bb7a86ca5c7cf54eda9012b8799eb662c4dfe9504d52b332ec3526.jpg)

![0e1e1571514f8ef1f960a86f05c8374d48d9d108f24b70ae0268952497e80452.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/tables/0e1e1571514f8ef1f960a86f05c8374d48d9d108f24b70ae0268952497e80452.jpg)

![4c9eb3044a9a91c787eef1a1c64a43f48242db0aec03742bab6901eaaee0644c.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/tables/4c9eb3044a9a91c787eef1a1c64a43f48242db0aec03742bab6901eaaee0644c.jpg)

![571aa57578e410f72eec924f59c760023e3d007cc7934a7eb45c492bafaf715c.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/tables/571aa57578e410f72eec924f59c760023e3d007cc7934a7eb45c492bafaf715c.jpg)

![789cf041c94a02b99a5477903ff145a91cc11a665c3ca6810595040851ad74b1.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/tables/789cf041c94a02b99a5477903ff145a91cc11a665c3ca6810595040851ad74b1.jpg)

![896c7b759720cbccee47ef5d5884e67ae7f5053275821ba65c8ad411733a75ab.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/tables/896c7b759720cbccee47ef5d5884e67ae7f5053275821ba65c8ad411733a75ab.jpg)

![b34452d136f2f26839f5970069c4df4403a8c5c1d8aed5b09d5d33c71ba21506.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/tables/b34452d136f2f26839f5970069c4df4403a8c5c1d8aed5b09d5d33c71ba21506.jpg)

![e1c833336e07680c033320349ca330ef3d75085e8cebda2301bee56d5acede25.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/tables/e1c833336e07680c033320349ca330ef3d75085e8cebda2301bee56d5acede25.jpg)

![f738cf83699e4cba3cef4cd9341af7a02cd38a60be689baaa227e3710373fe71.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/tables/f738cf83699e4cba3cef4cd9341af7a02cd38a60be689baaa227e3710373fe71.jpg)

![f92312612a9b9dcefa33dfa6ea0f446b9e7f37fd837d66bf1e3d0517b30a9993.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/tables/f92312612a9b9dcefa33dfa6ea0f446b9e7f37fd837d66bf1e3d0517b30a9993.jpg)

![fa9eb349e438db263317251f8f170b4bda911eb05a23713c5a7ec8a042bb5920.jpg](../iclr_results/3207_IRIS_ LLM-Assisted Static Analysis for Detecting Security Vulnerabilities/tables/fa9eb349e438db263317251f8f170b4bda911eb05a23713c5a7ec8a042bb5920.jpg)

## Towards Optimal Multi-draft Speculative Decoding


### Images

![040f9ca6712d114d2fd14ba2ef86051644fb8765a6b6c88f01173c899a455418.jpg](../iclr_results/3208_Towards Optimal Multi-draft Speculative Decoding/images/040f9ca6712d114d2fd14ba2ef86051644fb8765a6b6c88f01173c899a455418.jpg)

![5e60ca81685e4a19fc7197d97467527733416b0b68a537ce809df5308a8f229c.jpg](../iclr_results/3208_Towards Optimal Multi-draft Speculative Decoding/images/5e60ca81685e4a19fc7197d97467527733416b0b68a537ce809df5308a8f229c.jpg)

![7c8cca66e21d8b2f7d04d237cd987cacf0c161a0b683d6f8b3574de6653a2721.jpg](../iclr_results/3208_Towards Optimal Multi-draft Speculative Decoding/images/7c8cca66e21d8b2f7d04d237cd987cacf0c161a0b683d6f8b3574de6653a2721.jpg)

![b93ff7252719a6494753d5939f6e9cea0f0b542058b18351366c942302ee6513.jpg](../iclr_results/3208_Towards Optimal Multi-draft Speculative Decoding/images/b93ff7252719a6494753d5939f6e9cea0f0b542058b18351366c942302ee6513.jpg)

### Tables

![1971a1e00c87a3c43399da16630551ed30c3ed388ae0da16ca83827e82664381.jpg](../iclr_results/3208_Towards Optimal Multi-draft Speculative Decoding/tables/1971a1e00c87a3c43399da16630551ed30c3ed388ae0da16ca83827e82664381.jpg)

![2385b1795ac684254734674fb700fb0fb44edcb78d03979a32c879aa0f8e93cb.jpg](../iclr_results/3208_Towards Optimal Multi-draft Speculative Decoding/tables/2385b1795ac684254734674fb700fb0fb44edcb78d03979a32c879aa0f8e93cb.jpg)

![e3f557c2fcb7ef1a3e4d646a838847c7e839f20184e290972d4c66c9ec5bd119.jpg](../iclr_results/3208_Towards Optimal Multi-draft Speculative Decoding/tables/e3f557c2fcb7ef1a3e4d646a838847c7e839f20184e290972d4c66c9ec5bd119.jpg)

## ALBAR: Adversarial Learning approach to mitigate Biases in Action Recognition


### Images

![53d6829cd0c801db395f20163a85a43789cf285f5dda692638628f60f7edf607.jpg](../iclr_results/3209_ALBAR_ Adversarial Learning approach to mitigate Biases in Action Recognition/images/53d6829cd0c801db395f20163a85a43789cf285f5dda692638628f60f7edf607.jpg)

![70243227ca390140e1c3ff4cf96ca34c285ee25015e79d16fcfcced87b380b8e.jpg](../iclr_results/3209_ALBAR_ Adversarial Learning approach to mitigate Biases in Action Recognition/images/70243227ca390140e1c3ff4cf96ca34c285ee25015e79d16fcfcced87b380b8e.jpg)

![9d9aff4d90d59676adac3082174be64d87b524ad60bc9f0d47207209c210f27b.jpg](../iclr_results/3209_ALBAR_ Adversarial Learning approach to mitigate Biases in Action Recognition/images/9d9aff4d90d59676adac3082174be64d87b524ad60bc9f0d47207209c210f27b.jpg)

![bd044ba2679abbef59685b2458d7427246ef88ae948c8c99e64d2b03329e574d.jpg](../iclr_results/3209_ALBAR_ Adversarial Learning approach to mitigate Biases in Action Recognition/images/bd044ba2679abbef59685b2458d7427246ef88ae948c8c99e64d2b03329e574d.jpg)

![d039eda323e76ac7609838910b7b77831338b5354dbddfd0d8c05b1c168bf74d.jpg](../iclr_results/3209_ALBAR_ Adversarial Learning approach to mitigate Biases in Action Recognition/images/d039eda323e76ac7609838910b7b77831338b5354dbddfd0d8c05b1c168bf74d.jpg)

![d2fdf1e3d6c32640a43a63155a2f531e7cb3d264a188c46d5735ac51cc154aa1.jpg](../iclr_results/3209_ALBAR_ Adversarial Learning approach to mitigate Biases in Action Recognition/images/d2fdf1e3d6c32640a43a63155a2f531e7cb3d264a188c46d5735ac51cc154aa1.jpg)

### Tables

![20659820775ba9b1e6450fde6da76e19ea4a0224cb3ccd3d591dccfb2bdc0718.jpg](../iclr_results/3209_ALBAR_ Adversarial Learning approach to mitigate Biases in Action Recognition/tables/20659820775ba9b1e6450fde6da76e19ea4a0224cb3ccd3d591dccfb2bdc0718.jpg)

![4530932f0d2805585680d85da67e03b772d33577a4a12a444ddb2dc11f4e0ab0.jpg](../iclr_results/3209_ALBAR_ Adversarial Learning approach to mitigate Biases in Action Recognition/tables/4530932f0d2805585680d85da67e03b772d33577a4a12a444ddb2dc11f4e0ab0.jpg)

![4d20b39c59709fa8cd78f2754cae0f4866d5426613df0e9e57f06da13ba3f701.jpg](../iclr_results/3209_ALBAR_ Adversarial Learning approach to mitigate Biases in Action Recognition/tables/4d20b39c59709fa8cd78f2754cae0f4866d5426613df0e9e57f06da13ba3f701.jpg)

![4f5d21f8fef0086ef68c5cb739330469fa95491b7aaaed75a30b7fae190decbd.jpg](../iclr_results/3209_ALBAR_ Adversarial Learning approach to mitigate Biases in Action Recognition/tables/4f5d21f8fef0086ef68c5cb739330469fa95491b7aaaed75a30b7fae190decbd.jpg)

![5d12176515bebd5d8ed92995a27a4a6a6d183a64bcb3e2a323a94e9023dd3875.jpg](../iclr_results/3209_ALBAR_ Adversarial Learning approach to mitigate Biases in Action Recognition/tables/5d12176515bebd5d8ed92995a27a4a6a6d183a64bcb3e2a323a94e9023dd3875.jpg)

![787ff9c83e24c9675dbcad83ed4da5293d54e79fbe9aba4a0499dbff3677e47e.jpg](../iclr_results/3209_ALBAR_ Adversarial Learning approach to mitigate Biases in Action Recognition/tables/787ff9c83e24c9675dbcad83ed4da5293d54e79fbe9aba4a0499dbff3677e47e.jpg)

![984e9a83b2a644ad9ecf3014cfbe5ac1cada6db86fda97606db4e1e8c8f1a713.jpg](../iclr_results/3209_ALBAR_ Adversarial Learning approach to mitigate Biases in Action Recognition/tables/984e9a83b2a644ad9ecf3014cfbe5ac1cada6db86fda97606db4e1e8c8f1a713.jpg)

![9c3514ad41cffc118f4db3c17451f4f5288178ceec18272af621a01172d05249.jpg](../iclr_results/3209_ALBAR_ Adversarial Learning approach to mitigate Biases in Action Recognition/tables/9c3514ad41cffc118f4db3c17451f4f5288178ceec18272af621a01172d05249.jpg)

![b9090e8b483d7a1f1fb063486105c61e733d76d1e18f5e6f57cd39c530e79593.jpg](../iclr_results/3209_ALBAR_ Adversarial Learning approach to mitigate Biases in Action Recognition/tables/b9090e8b483d7a1f1fb063486105c61e733d76d1e18f5e6f57cd39c530e79593.jpg)

![cb9ee6d5cb7d09031950eaddecb9af90f43c53a600ba3f9288ed425a5a1d5220.jpg](../iclr_results/3209_ALBAR_ Adversarial Learning approach to mitigate Biases in Action Recognition/tables/cb9ee6d5cb7d09031950eaddecb9af90f43c53a600ba3f9288ed425a5a1d5220.jpg)

![d7a54a2a6c81426fee23862ce58f9a4fd3e499a73925503f198b733c9f5949b0.jpg](../iclr_results/3209_ALBAR_ Adversarial Learning approach to mitigate Biases in Action Recognition/tables/d7a54a2a6c81426fee23862ce58f9a4fd3e499a73925503f198b733c9f5949b0.jpg)

![fae345e9698b0619d4d41c27622f9d18680ac5187db31a46675363fce8d45235.jpg](../iclr_results/3209_ALBAR_ Adversarial Learning approach to mitigate Biases in Action Recognition/tables/fae345e9698b0619d4d41c27622f9d18680ac5187db31a46675363fce8d45235.jpg)

## An Optimal Discriminator Weighted Imitation Perspective for Reinforcement Learning


### Images

![42f5606c7950cce82f92c3acbdd3d460d1f34d1ac35bb5e6a0071cc33add7ff5.jpg](../iclr_results/3210_An Optimal Discriminator Weighted Imitation Perspective for Reinforcement Learning/images/42f5606c7950cce82f92c3acbdd3d460d1f34d1ac35bb5e6a0071cc33add7ff5.jpg)

![735516bc86ed036439e133d9c9002ec0ab0dd2d40311edbf69af3c67ec182273.jpg](../iclr_results/3210_An Optimal Discriminator Weighted Imitation Perspective for Reinforcement Learning/images/735516bc86ed036439e133d9c9002ec0ab0dd2d40311edbf69af3c67ec182273.jpg)

![ad472841ef5a4f18335f2f345b3b2396821fac4eb29c8b32f7f2fb661ab183aa.jpg](../iclr_results/3210_An Optimal Discriminator Weighted Imitation Perspective for Reinforcement Learning/images/ad472841ef5a4f18335f2f345b3b2396821fac4eb29c8b32f7f2fb661ab183aa.jpg)

![e20c222e9c34258fc7c90056acff3cfe26b0d294706c0741812451e3d5e694ae.jpg](../iclr_results/3210_An Optimal Discriminator Weighted Imitation Perspective for Reinforcement Learning/images/e20c222e9c34258fc7c90056acff3cfe26b0d294706c0741812451e3d5e694ae.jpg)

![eb046cc1113d487a493b71a93d87619b7cdbfec72a2ed34cb48f6d13d9d9be9f.jpg](../iclr_results/3210_An Optimal Discriminator Weighted Imitation Perspective for Reinforcement Learning/images/eb046cc1113d487a493b71a93d87619b7cdbfec72a2ed34cb48f6d13d9d9be9f.jpg)

### Tables

![1bb41a07c3d9a5b123f4b999e43f571aeed98d6146af548d44bd829f16a057e2.jpg](../iclr_results/3210_An Optimal Discriminator Weighted Imitation Perspective for Reinforcement Learning/tables/1bb41a07c3d9a5b123f4b999e43f571aeed98d6146af548d44bd829f16a057e2.jpg)

![585cd256f40b8f5dcf032036dbab859c69be163cb8b7d798d8826ea623d375bd.jpg](../iclr_results/3210_An Optimal Discriminator Weighted Imitation Perspective for Reinforcement Learning/tables/585cd256f40b8f5dcf032036dbab859c69be163cb8b7d798d8826ea623d375bd.jpg)

![89ecc9dcc50117f42481e9dc55dc2b50c871e82401361e669577ed83d2e3ea12.jpg](../iclr_results/3210_An Optimal Discriminator Weighted Imitation Perspective for Reinforcement Learning/tables/89ecc9dcc50117f42481e9dc55dc2b50c871e82401361e669577ed83d2e3ea12.jpg)

![9f33826788f1460b42ddade1cc40a6c4ba9813a3d658dde49897a786b612ae12.jpg](../iclr_results/3210_An Optimal Discriminator Weighted Imitation Perspective for Reinforcement Learning/tables/9f33826788f1460b42ddade1cc40a6c4ba9813a3d658dde49897a786b612ae12.jpg)

## Visual Haystacks: A Vision-Centric Needle-In-A-Haystack Benchmark


### Images

![18d30fb8fce3944a1ba7d12ae0a218eeca6ec7e2c573895cee75111059805670.jpg](../iclr_results/3211_Visual Haystacks_ A Vision-Centric Needle-In-A-Haystack Benchmark/images/18d30fb8fce3944a1ba7d12ae0a218eeca6ec7e2c573895cee75111059805670.jpg)

![33f258b73ee6fa9674f928e032ee8e317e86558828816517d955a3946cc65f2c.jpg](../iclr_results/3211_Visual Haystacks_ A Vision-Centric Needle-In-A-Haystack Benchmark/images/33f258b73ee6fa9674f928e032ee8e317e86558828816517d955a3946cc65f2c.jpg)

![38021283dc8b29b044fe75bf87a74e9245e71fb387bdf63454c7bd05d9fd5381.jpg](../iclr_results/3211_Visual Haystacks_ A Vision-Centric Needle-In-A-Haystack Benchmark/images/38021283dc8b29b044fe75bf87a74e9245e71fb387bdf63454c7bd05d9fd5381.jpg)

![4b6518c271f2c1a77b04c1a71724763428e622be26476dc05e283d95945bf8e0.jpg](../iclr_results/3211_Visual Haystacks_ A Vision-Centric Needle-In-A-Haystack Benchmark/images/4b6518c271f2c1a77b04c1a71724763428e622be26476dc05e283d95945bf8e0.jpg)

![4e4fb21c2bc91ffb5430703155142bc24c2a0c387ad4359e7ab92f77cd819f63.jpg](../iclr_results/3211_Visual Haystacks_ A Vision-Centric Needle-In-A-Haystack Benchmark/images/4e4fb21c2bc91ffb5430703155142bc24c2a0c387ad4359e7ab92f77cd819f63.jpg)

![5e19a551be94eae1c9d91adbb4bd1157725fcea86c7deb738c3345705f88a56b.jpg](../iclr_results/3211_Visual Haystacks_ A Vision-Centric Needle-In-A-Haystack Benchmark/images/5e19a551be94eae1c9d91adbb4bd1157725fcea86c7deb738c3345705f88a56b.jpg)

![aa9518b6df411aca6f089464c401118876d0c5231f953a67a805aee561004070.jpg](../iclr_results/3211_Visual Haystacks_ A Vision-Centric Needle-In-A-Haystack Benchmark/images/aa9518b6df411aca6f089464c401118876d0c5231f953a67a805aee561004070.jpg)

![ae4672fa84dbbfef1602d77fecade73c52f0d6c2614ac671ed6c4e7b1095f889.jpg](../iclr_results/3211_Visual Haystacks_ A Vision-Centric Needle-In-A-Haystack Benchmark/images/ae4672fa84dbbfef1602d77fecade73c52f0d6c2614ac671ed6c4e7b1095f889.jpg)

![c51dc55cff824aaf028d113c3f0549fdb307c1a45260a31dcdb7b5168f6e73b8.jpg](../iclr_results/3211_Visual Haystacks_ A Vision-Centric Needle-In-A-Haystack Benchmark/images/c51dc55cff824aaf028d113c3f0549fdb307c1a45260a31dcdb7b5168f6e73b8.jpg)

![c89817966d37ccd503af204415f9efc159d23b3b1966a8eda33f7a0a6afbda8d.jpg](../iclr_results/3211_Visual Haystacks_ A Vision-Centric Needle-In-A-Haystack Benchmark/images/c89817966d37ccd503af204415f9efc159d23b3b1966a8eda33f7a0a6afbda8d.jpg)

![d8b6682ea9b39f69144fb91aee318af2262e5635aa8202e2ed85243cbf124cc4.jpg](../iclr_results/3211_Visual Haystacks_ A Vision-Centric Needle-In-A-Haystack Benchmark/images/d8b6682ea9b39f69144fb91aee318af2262e5635aa8202e2ed85243cbf124cc4.jpg)

![ecad12158d504dff59e615085126757c15d8f1bbf72d981015f758b704766cca.jpg](../iclr_results/3211_Visual Haystacks_ A Vision-Centric Needle-In-A-Haystack Benchmark/images/ecad12158d504dff59e615085126757c15d8f1bbf72d981015f758b704766cca.jpg)

![f06ac5d527d82592af0e18db1e529d09326fa1bba3bf51ad7b79861b57a7b252.jpg](../iclr_results/3211_Visual Haystacks_ A Vision-Centric Needle-In-A-Haystack Benchmark/images/f06ac5d527d82592af0e18db1e529d09326fa1bba3bf51ad7b79861b57a7b252.jpg)

![f2a83907a22745e7dfa4f61f403463cdaaad87cbc61ea311efb425b5d80c19a6.jpg](../iclr_results/3211_Visual Haystacks_ A Vision-Centric Needle-In-A-Haystack Benchmark/images/f2a83907a22745e7dfa4f61f403463cdaaad87cbc61ea311efb425b5d80c19a6.jpg)

![f68027672b0d5586111e43bfe4f96f42633a9fc56aad03f81d372eed2daa5308.jpg](../iclr_results/3211_Visual Haystacks_ A Vision-Centric Needle-In-A-Haystack Benchmark/images/f68027672b0d5586111e43bfe4f96f42633a9fc56aad03f81d372eed2daa5308.jpg)

### Tables

![2a517117f08abaa6c325549751c1cf35541e145b8508f2e5f3bb3d2beffbd291.jpg](../iclr_results/3211_Visual Haystacks_ A Vision-Centric Needle-In-A-Haystack Benchmark/tables/2a517117f08abaa6c325549751c1cf35541e145b8508f2e5f3bb3d2beffbd291.jpg)

![8bea4427e8ccb2c2b42abb980bad5ff2403376a0610c9494df906872c751a9c6.jpg](../iclr_results/3211_Visual Haystacks_ A Vision-Centric Needle-In-A-Haystack Benchmark/tables/8bea4427e8ccb2c2b42abb980bad5ff2403376a0610c9494df906872c751a9c6.jpg)

![99ba70d4fab7e515ff6215181fe49ecb8dc218ccc23ce2f8b12b83fcd449f22c.jpg](../iclr_results/3211_Visual Haystacks_ A Vision-Centric Needle-In-A-Haystack Benchmark/tables/99ba70d4fab7e515ff6215181fe49ecb8dc218ccc23ce2f8b12b83fcd449f22c.jpg)

![9f4ecfad533260205055f1dc96cd34df2426a2dd648885a6969ac21310644482.jpg](../iclr_results/3211_Visual Haystacks_ A Vision-Centric Needle-In-A-Haystack Benchmark/tables/9f4ecfad533260205055f1dc96cd34df2426a2dd648885a6969ac21310644482.jpg)

## Reward Learning from Multiple Feedback Types


### Images

![083c11afc30b347061b9616dfb9a2cb43a67bdbae662a82ff89bd1a500b2cf7f.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/083c11afc30b347061b9616dfb9a2cb43a67bdbae662a82ff89bd1a500b2cf7f.jpg)

![0c542d4dfa75ec8bb9f98ebbccfa132908f1b67e9fca60c146c3312b91ae500e.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/0c542d4dfa75ec8bb9f98ebbccfa132908f1b67e9fca60c146c3312b91ae500e.jpg)

![0d206e19972f44613a0fdddae63c1a36ecc03b46f54990dad935592afd07ef20.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/0d206e19972f44613a0fdddae63c1a36ecc03b46f54990dad935592afd07ef20.jpg)

![0f503c67c865f6855cfa35cd0547d7aefa81c4a34786bb9f32c52e10fcdc84f4.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/0f503c67c865f6855cfa35cd0547d7aefa81c4a34786bb9f32c52e10fcdc84f4.jpg)

![0fcb1671d462d23aaeadf8676273a352751ae6c7292f44bb71d4b670bb066deb.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/0fcb1671d462d23aaeadf8676273a352751ae6c7292f44bb71d4b670bb066deb.jpg)

![12dd7e0ccb8564a150da4a55b182555b1c86768047d3f1876b5df45574b1a789.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/12dd7e0ccb8564a150da4a55b182555b1c86768047d3f1876b5df45574b1a789.jpg)

![1671fb234d27b6a17f3baf40d3fd4f814e841c9f09b41a14461c86df7d7c82a2.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/1671fb234d27b6a17f3baf40d3fd4f814e841c9f09b41a14461c86df7d7c82a2.jpg)

![1a97c673df08c66391b1ff1c9a20e0fdfeab27b4c74527dec4bd76d4324fb639.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/1a97c673df08c66391b1ff1c9a20e0fdfeab27b4c74527dec4bd76d4324fb639.jpg)

![1e66c8224b88edc52117d5b42872c1f363fec966d4c9b30b640f8d8ba441267b.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/1e66c8224b88edc52117d5b42872c1f363fec966d4c9b30b640f8d8ba441267b.jpg)

![1ed5d0da0087ee98360ebaa0f1ca43c38db9afc248f129a8818317bde74999a8.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/1ed5d0da0087ee98360ebaa0f1ca43c38db9afc248f129a8818317bde74999a8.jpg)

![21d0e1b1c886921bf70d4d99e1d5cb66e2038c4547a83ebe3b49c88b5b4f70ef.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/21d0e1b1c886921bf70d4d99e1d5cb66e2038c4547a83ebe3b49c88b5b4f70ef.jpg)

![2e2c12ec428e9932683a164835b6fe342c6aa6a21ee6cc913f3e07df2bfccca5.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/2e2c12ec428e9932683a164835b6fe342c6aa6a21ee6cc913f3e07df2bfccca5.jpg)

![307ee371bcca9d05d8f8b63f8849d3dbc50d3f43488830ecfe75fba6d811bf90.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/307ee371bcca9d05d8f8b63f8849d3dbc50d3f43488830ecfe75fba6d811bf90.jpg)

![3117533619f38d5eb0662bee4e85277c5c40915e493f33118853ba64612434fe.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/3117533619f38d5eb0662bee4e85277c5c40915e493f33118853ba64612434fe.jpg)

![3374e986132f7039da2aa5ff00d4b3a851f39fc876784813caf4824d2ede9a5b.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/3374e986132f7039da2aa5ff00d4b3a851f39fc876784813caf4824d2ede9a5b.jpg)

![397c448b9dd2b85d8de1248d4ba0a8045e431b8da499bbdeb44fb246ac5f2000.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/397c448b9dd2b85d8de1248d4ba0a8045e431b8da499bbdeb44fb246ac5f2000.jpg)

![3ddae965b121f6bdf02e37d3865726f4629b93ca48fbb9230627460fcd484528.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/3ddae965b121f6bdf02e37d3865726f4629b93ca48fbb9230627460fcd484528.jpg)

![48ee0d9bac81c995a19c60002bacaeeb482da12781d76dec35ab30014133c2fc.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/48ee0d9bac81c995a19c60002bacaeeb482da12781d76dec35ab30014133c2fc.jpg)

![4b647e31a51fefbce3052cc9585dce783ce66808880828c6ed22877410401ccc.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/4b647e31a51fefbce3052cc9585dce783ce66808880828c6ed22877410401ccc.jpg)

![4de99bcbda7629bc7bcee6e7fb6da5fd722192f4790e7c3d274362c43ac708be.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/4de99bcbda7629bc7bcee6e7fb6da5fd722192f4790e7c3d274362c43ac708be.jpg)

![52bc6002d93bac674933f303ceb98daeb73e398588dde6757d38416c579f368c.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/52bc6002d93bac674933f303ceb98daeb73e398588dde6757d38416c579f368c.jpg)

![5969d90fac48113d06428e3cc76e91c90cf48d240d6b6bacc4cc96ad3da317eb.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/5969d90fac48113d06428e3cc76e91c90cf48d240d6b6bacc4cc96ad3da317eb.jpg)

![5a34e0fc19399ba3408d2f2f103564464d7da49149185acfebc27ae33eac141b.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/5a34e0fc19399ba3408d2f2f103564464d7da49149185acfebc27ae33eac141b.jpg)

![5a510aad6f8b7cdb11c1ddde5a7f28cce2ce271bd6d6a983a1933f03fff54833.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/5a510aad6f8b7cdb11c1ddde5a7f28cce2ce271bd6d6a983a1933f03fff54833.jpg)

![604fbd256811038d4d4d9e859e276a05bfe435b57baa0f7eb2b4d6e03c78c91d.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/604fbd256811038d4d4d9e859e276a05bfe435b57baa0f7eb2b4d6e03c78c91d.jpg)

![6cec2a35998fb0d989b9c75d2940ee7f563a469596e2321e9d965b194b1f3883.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/6cec2a35998fb0d989b9c75d2940ee7f563a469596e2321e9d965b194b1f3883.jpg)

![6d4404bb3e8f6365ce23951a2930c4178c03f6a37c7af76409b93e8443eff9d8.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/6d4404bb3e8f6365ce23951a2930c4178c03f6a37c7af76409b93e8443eff9d8.jpg)

![746194f34b6a71783f786c1e0cae026a76f20fa3ab975e1bb19dd6f380200004.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/746194f34b6a71783f786c1e0cae026a76f20fa3ab975e1bb19dd6f380200004.jpg)

![783d22f661f49fc3326dd54c573fe1a7460e4f2d0f447b3835da584db570b560.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/783d22f661f49fc3326dd54c573fe1a7460e4f2d0f447b3835da584db570b560.jpg)

![81aacc2710c575f6be35a7f241a009775902d6f34aafad1d3adf79b2e1265565.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/81aacc2710c575f6be35a7f241a009775902d6f34aafad1d3adf79b2e1265565.jpg)

![849943e5003c088b383b50406f8b527b1f272d39443c7c8586b305e774a5e787.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/849943e5003c088b383b50406f8b527b1f272d39443c7c8586b305e774a5e787.jpg)

![8c3cae9edd68a4786e8747f454fc7ff43778a19d1b88f042f5e44dada7bc5e18.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/8c3cae9edd68a4786e8747f454fc7ff43778a19d1b88f042f5e44dada7bc5e18.jpg)

![928d5d939f525d67639eaf4a0931aeef4184bd012465a8527b3c38f5235f12ca.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/928d5d939f525d67639eaf4a0931aeef4184bd012465a8527b3c38f5235f12ca.jpg)

![977db2d8c39efd259fcf0e9434f1adc8beb84664efc5200f223248dc9fbccd3c.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/977db2d8c39efd259fcf0e9434f1adc8beb84664efc5200f223248dc9fbccd3c.jpg)

![9fe14370c5954e83e614b17c845b276f96ada8479b176761f3a34b63617754f9.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/9fe14370c5954e83e614b17c845b276f96ada8479b176761f3a34b63617754f9.jpg)

![a0dc51a8842344e3386358a1985646283b4f5c8e0d4dbb03cb119665914d1151.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/a0dc51a8842344e3386358a1985646283b4f5c8e0d4dbb03cb119665914d1151.jpg)

![a27eeedc2014b558635b5dfbd36df9f355b08406b11c5bab585c65995c0b589f.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/a27eeedc2014b558635b5dfbd36df9f355b08406b11c5bab585c65995c0b589f.jpg)

![a3f96dcd2b666742b11b9e7263eb5c9bcc7c2346b387b543a608b37c584534c4.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/a3f96dcd2b666742b11b9e7263eb5c9bcc7c2346b387b543a608b37c584534c4.jpg)

![a490cf9bc06142b0c045545dccebda75ed6d359a0d83dbce8d95495a78203921.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/a490cf9bc06142b0c045545dccebda75ed6d359a0d83dbce8d95495a78203921.jpg)

![a6233b1ece3e5a7f041443434bb967c3c85e473dfa111caa2d5b1e586a2f840b.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/a6233b1ece3e5a7f041443434bb967c3c85e473dfa111caa2d5b1e586a2f840b.jpg)

![b206317a11e351769a5bdd05751b29c3b1e72f5bf3ae085ee283fb288ec1c516.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/b206317a11e351769a5bdd05751b29c3b1e72f5bf3ae085ee283fb288ec1c516.jpg)

![b514ed76203340c3f2c2eedfaf5ec7fd71672c9bebd124156f9aee873e6181cc.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/b514ed76203340c3f2c2eedfaf5ec7fd71672c9bebd124156f9aee873e6181cc.jpg)

![b74de54b4dd8867236f6313b26968792765c02afc4b64b7ec49fe9d2d5c251ff.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/b74de54b4dd8867236f6313b26968792765c02afc4b64b7ec49fe9d2d5c251ff.jpg)

![bbe1cc3b94d616979eac02c8c9e35524aa746cda8922e749242a0c041283c083.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/bbe1cc3b94d616979eac02c8c9e35524aa746cda8922e749242a0c041283c083.jpg)

![bf9b57a3e81831010f74119228372c2e6f9c5f532acc4513f77b594544ef0c22.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/bf9b57a3e81831010f74119228372c2e6f9c5f532acc4513f77b594544ef0c22.jpg)

![c1c3c3cd5e061ca05eedc3bb0dc928212b72ef5e4d1816ef59ac730e48fa8c97.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/c1c3c3cd5e061ca05eedc3bb0dc928212b72ef5e4d1816ef59ac730e48fa8c97.jpg)

![c38fb3157c5f29069420bea78e5eb01edbf261ffd5d9efb3a01e3bf0b4f247d4.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/c38fb3157c5f29069420bea78e5eb01edbf261ffd5d9efb3a01e3bf0b4f247d4.jpg)

![c44a816ade5bc739a81589c4fa09eeee85e5de3fecbe4b7cfba745a1a11dcaa4.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/c44a816ade5bc739a81589c4fa09eeee85e5de3fecbe4b7cfba745a1a11dcaa4.jpg)

![cb5ae354b1d71382e127589068ee80bb7be47465d92c741ad5caa2f46465a143.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/cb5ae354b1d71382e127589068ee80bb7be47465d92c741ad5caa2f46465a143.jpg)

![cf88ec323b1e96f5e394e3b693694b911888eeb62741b7dc52133ba3ec501c97.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/cf88ec323b1e96f5e394e3b693694b911888eeb62741b7dc52133ba3ec501c97.jpg)

![d32b63379522faba7fd8a90a90ef7a0ae6a4a4f14133ef7a7f53f93447071b94.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/d32b63379522faba7fd8a90a90ef7a0ae6a4a4f14133ef7a7f53f93447071b94.jpg)

![d34e35faa3f816b2c0563b57519ee9f2f5680fa11278dc03893eccf6b1346c11.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/d34e35faa3f816b2c0563b57519ee9f2f5680fa11278dc03893eccf6b1346c11.jpg)

![d9c8547d56f935a42f2055fa7477b23176fb49022a8bee85d23cb4da457836cf.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/d9c8547d56f935a42f2055fa7477b23176fb49022a8bee85d23cb4da457836cf.jpg)

![db34f5870e8f142d5370ddb59fa2253ed3a6fd7012a4675048ffe65464b98178.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/db34f5870e8f142d5370ddb59fa2253ed3a6fd7012a4675048ffe65464b98178.jpg)

![dc074a7f450db1e69e31cf82129aaa3adde7f334544904f8a2e3363309daf0fe.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/dc074a7f450db1e69e31cf82129aaa3adde7f334544904f8a2e3363309daf0fe.jpg)

![dec920d9250e664cde9fea499de3e0dbf3bd89898800f2e68370f9dcc6e23573.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/dec920d9250e664cde9fea499de3e0dbf3bd89898800f2e68370f9dcc6e23573.jpg)

![dff3570eafcbfe4ef07c8dce99c92a2f73994cae9ef4feff378510e6b337624d.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/dff3570eafcbfe4ef07c8dce99c92a2f73994cae9ef4feff378510e6b337624d.jpg)

![e91336348726c494b46f2318871c518fb336ed1676f5b0ee250e0607bf57f434.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/e91336348726c494b46f2318871c518fb336ed1676f5b0ee250e0607bf57f434.jpg)

![eab344b9d27571d67b11a9a8ac297403147788d356624f96256cd7b015014c53.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/eab344b9d27571d67b11a9a8ac297403147788d356624f96256cd7b015014c53.jpg)

![f395876958536242fda89ae0829e854f37567dce6d2a6d87b6af7e76d3517054.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/f395876958536242fda89ae0829e854f37567dce6d2a6d87b6af7e76d3517054.jpg)

![fa35a66c2fffa4d15aedbae39361d01ea496e9cb27dbff22206850966766adbe.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/fa35a66c2fffa4d15aedbae39361d01ea496e9cb27dbff22206850966766adbe.jpg)

![fc77cf1d2fac242dbacce408db5a0085e01875c4dfd5c3ffede3e966eb584f2e.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/images/fc77cf1d2fac242dbacce408db5a0085e01875c4dfd5c3ffede3e966eb584f2e.jpg)

### Tables

![067ae418c356aa1c63aa8ab92b9790de72d80332ae366106249ce92a2148dca7.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/tables/067ae418c356aa1c63aa8ab92b9790de72d80332ae366106249ce92a2148dca7.jpg)

![29a9ba688d94bb931f39d2518db0afd6d6d1d743738c74780c6c77891717abd5.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/tables/29a9ba688d94bb931f39d2518db0afd6d6d1d743738c74780c6c77891717abd5.jpg)

![390b37d71cd942995b490bf0b4063fff9570dde649d07334573a64ac7d2d4f10.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/tables/390b37d71cd942995b490bf0b4063fff9570dde649d07334573a64ac7d2d4f10.jpg)

![8f66da48aed4df91f676930ebc9a1e1b2e6f98dc61b7fd224441cc4c3d07ece8.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/tables/8f66da48aed4df91f676930ebc9a1e1b2e6f98dc61b7fd224441cc4c3d07ece8.jpg)

![b5a99e923a8630607918c38aaa6b87e59996d4b53e58cc800636c537a86ab7e6.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/tables/b5a99e923a8630607918c38aaa6b87e59996d4b53e58cc800636c537a86ab7e6.jpg)

![e2e1ed50e4d8953e5914c639a9a889a54c07775f2ed6ccd514acb9fb2a39479e.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/tables/e2e1ed50e4d8953e5914c639a9a889a54c07775f2ed6ccd514acb9fb2a39479e.jpg)

![e94b5a487e41f39422d3b41b9d75837f97c5a0543e4a1123ec5e5db3dd388427.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/tables/e94b5a487e41f39422d3b41b9d75837f97c5a0543e4a1123ec5e5db3dd388427.jpg)

![f861fcceffdc45712c4f9ec2e2b96cfa71068226a196390f88ea1f1711e76b3c.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/tables/f861fcceffdc45712c4f9ec2e2b96cfa71068226a196390f88ea1f1711e76b3c.jpg)

![fc1a3a85e6f9ce337ef79944fc6e62974887b91485c7f6a25c82e3b87f6d8859.jpg](../iclr_results/3212_Reward Learning from Multiple Feedback Types/tables/fc1a3a85e6f9ce337ef79944fc6e62974887b91485c7f6a25c82e3b87f6d8859.jpg)

## 3D-Properties: Identifying Challenges in DPO and Charting a Path Forward


### Images

![1e56eee6d66e1e4f19cde2b9c0d0f6bd745f424cf05f90e70fdaa1c0e745636c.jpg](../iclr_results/3213_3D-Properties_ Identifying Challenges in DPO and Charting a Path Forward/images/1e56eee6d66e1e4f19cde2b9c0d0f6bd745f424cf05f90e70fdaa1c0e745636c.jpg)

![4c4566061be831e873d182897436a0f098e17f7724700b55d68040182a236ec7.jpg](../iclr_results/3213_3D-Properties_ Identifying Challenges in DPO and Charting a Path Forward/images/4c4566061be831e873d182897436a0f098e17f7724700b55d68040182a236ec7.jpg)

![6322abb48a963492339b9bc52a99d18de929c4b7f53f142af50214627aefd67c.jpg](../iclr_results/3213_3D-Properties_ Identifying Challenges in DPO and Charting a Path Forward/images/6322abb48a963492339b9bc52a99d18de929c4b7f53f142af50214627aefd67c.jpg)

![68e5ab1c7556c67f3c94a7d3cd228e3b8ab402e6da677c62ca42859f615f3300.jpg](../iclr_results/3213_3D-Properties_ Identifying Challenges in DPO and Charting a Path Forward/images/68e5ab1c7556c67f3c94a7d3cd228e3b8ab402e6da677c62ca42859f615f3300.jpg)

![6dc18ad70182de0fcd32f066bb6b5f5d4f6be30fcbfd194d6873ed4ae3792a78.jpg](../iclr_results/3213_3D-Properties_ Identifying Challenges in DPO and Charting a Path Forward/images/6dc18ad70182de0fcd32f066bb6b5f5d4f6be30fcbfd194d6873ed4ae3792a78.jpg)

![872b4bdca8f7228928210ecdf446eb12c9931c432555d6602da72c3e3b97ccd4.jpg](../iclr_results/3213_3D-Properties_ Identifying Challenges in DPO and Charting a Path Forward/images/872b4bdca8f7228928210ecdf446eb12c9931c432555d6602da72c3e3b97ccd4.jpg)

![89bf60072d4a929bc5627f7225edab4eb6a378d96b7932942b43c0e4e6849be0.jpg](../iclr_results/3213_3D-Properties_ Identifying Challenges in DPO and Charting a Path Forward/images/89bf60072d4a929bc5627f7225edab4eb6a378d96b7932942b43c0e4e6849be0.jpg)

![8b63767107939ac847ea56f074a2783a515c3108b0168600dd12e63c99d86a36.jpg](../iclr_results/3213_3D-Properties_ Identifying Challenges in DPO and Charting a Path Forward/images/8b63767107939ac847ea56f074a2783a515c3108b0168600dd12e63c99d86a36.jpg)

![9b0004b4d7534517de947587cbd8816c0a12c102d145e0b181a316b3f90995d4.jpg](../iclr_results/3213_3D-Properties_ Identifying Challenges in DPO and Charting a Path Forward/images/9b0004b4d7534517de947587cbd8816c0a12c102d145e0b181a316b3f90995d4.jpg)

![dda9e4c5846377a079ce664c923d6a3944193de732c357b77b68d7fbf324e9d8.jpg](../iclr_results/3213_3D-Properties_ Identifying Challenges in DPO and Charting a Path Forward/images/dda9e4c5846377a079ce664c923d6a3944193de732c357b77b68d7fbf324e9d8.jpg)

![e3037ba3c5a51ab34ac56d9f38a882efd5a12bf226971068b87038db9a6c739d.jpg](../iclr_results/3213_3D-Properties_ Identifying Challenges in DPO and Charting a Path Forward/images/e3037ba3c5a51ab34ac56d9f38a882efd5a12bf226971068b87038db9a6c739d.jpg)

![e56492a19eef2f44e3b7b3b5c8e5672ed461f3a8a4f3be873c61da1a1765e790.jpg](../iclr_results/3213_3D-Properties_ Identifying Challenges in DPO and Charting a Path Forward/images/e56492a19eef2f44e3b7b3b5c8e5672ed461f3a8a4f3be873c61da1a1765e790.jpg)

### Tables

![147bd8dba2d577b17721a9c50a98afd8f31667118a360e0d07d06f6c9fd9c96a.jpg](../iclr_results/3213_3D-Properties_ Identifying Challenges in DPO and Charting a Path Forward/tables/147bd8dba2d577b17721a9c50a98afd8f31667118a360e0d07d06f6c9fd9c96a.jpg)

![4e8733c175cb8ec06848d50a20e992332ee70d448a969d3d4186b1be8e441ccf.jpg](../iclr_results/3213_3D-Properties_ Identifying Challenges in DPO and Charting a Path Forward/tables/4e8733c175cb8ec06848d50a20e992332ee70d448a969d3d4186b1be8e441ccf.jpg)

![837084577a58d1b2ed43c929cb92f0f2736361138c4707abf78f991f586e2a77.jpg](../iclr_results/3213_3D-Properties_ Identifying Challenges in DPO and Charting a Path Forward/tables/837084577a58d1b2ed43c929cb92f0f2736361138c4707abf78f991f586e2a77.jpg)

![9603c0f3a65f1b9eff54f242b853ad83ff87ace72646c5b632ef395458fd9a1c.jpg](../iclr_results/3213_3D-Properties_ Identifying Challenges in DPO and Charting a Path Forward/tables/9603c0f3a65f1b9eff54f242b853ad83ff87ace72646c5b632ef395458fd9a1c.jpg)

![ab84a1cbd3a0d6ddf4bbdc9ed216310011de3c133e9914c59e5b1cbfe5dfb8b9.jpg](../iclr_results/3213_3D-Properties_ Identifying Challenges in DPO and Charting a Path Forward/tables/ab84a1cbd3a0d6ddf4bbdc9ed216310011de3c133e9914c59e5b1cbfe5dfb8b9.jpg)

![c4eb7d9bcce0601804210c8c80856f4cd7bdd2dd2e7698c66015808bbc7700b2.jpg](../iclr_results/3213_3D-Properties_ Identifying Challenges in DPO and Charting a Path Forward/tables/c4eb7d9bcce0601804210c8c80856f4cd7bdd2dd2e7698c66015808bbc7700b2.jpg)

![d08510bc4904fdc1022f9ef883e20d508f8332d269b660f71bced05ec014e585.jpg](../iclr_results/3213_3D-Properties_ Identifying Challenges in DPO and Charting a Path Forward/tables/d08510bc4904fdc1022f9ef883e20d508f8332d269b660f71bced05ec014e585.jpg)

![e9b7bb079fd52415591e2c3a12fb1fed75d89ad5604511cf6a8452ae14666e3b.jpg](../iclr_results/3213_3D-Properties_ Identifying Challenges in DPO and Charting a Path Forward/tables/e9b7bb079fd52415591e2c3a12fb1fed75d89ad5604511cf6a8452ae14666e3b.jpg)

![fec767e3f8a91c804440b11d8257771584b16274eaa9947e57f9dfe6fa571cf9.jpg](../iclr_results/3213_3D-Properties_ Identifying Challenges in DPO and Charting a Path Forward/tables/fec767e3f8a91c804440b11d8257771584b16274eaa9947e57f9dfe6fa571cf9.jpg)

## Temporal Flexibility in Spiking Neural Networks: Towards Generalization Across Time Steps and Deployment Friendliness


### Images

![0c3b5d8ccb2d745c5e8014158a1817178daabb7225ccf2ed3cae61214fa0fe1e.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/images/0c3b5d8ccb2d745c5e8014158a1817178daabb7225ccf2ed3cae61214fa0fe1e.jpg)

![1a1c663e794c37fa84c5b5fc47055554ee2b9f1efd15eeb0060b7e60c783d127.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/images/1a1c663e794c37fa84c5b5fc47055554ee2b9f1efd15eeb0060b7e60c783d127.jpg)

![2c308f9ae9903f8b5f62fd7c8209ed72b7b70f2efed52f031a4e9ada946868b2.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/images/2c308f9ae9903f8b5f62fd7c8209ed72b7b70f2efed52f031a4e9ada946868b2.jpg)

![5bf9a7f2eca662660c2b5dd980fb5140db63aac1d02776b9a3e2a63e44182027.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/images/5bf9a7f2eca662660c2b5dd980fb5140db63aac1d02776b9a3e2a63e44182027.jpg)

![65641eb9bddbbebe14552bea77f19625ce0fa0e8988d6704be4542f7e63ef8d7.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/images/65641eb9bddbbebe14552bea77f19625ce0fa0e8988d6704be4542f7e63ef8d7.jpg)

![8546196423cadefc131ed83626a9b93eb21e06dd343106580bbf895605895dc1.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/images/8546196423cadefc131ed83626a9b93eb21e06dd343106580bbf895605895dc1.jpg)

![8b4dcce6adb37f1964b7a7d868b2d127ab40e97e3fd2bb27098702f8b213a6fc.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/images/8b4dcce6adb37f1964b7a7d868b2d127ab40e97e3fd2bb27098702f8b213a6fc.jpg)

![a286d5010dcedff58c5ee8899c59a65a2326566cc98a85d522e2457c07ace60c.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/images/a286d5010dcedff58c5ee8899c59a65a2326566cc98a85d522e2457c07ace60c.jpg)

![bcc0ef24d3cbd7f2b06e0e5ae2d31b7ff6badefcbb612fae1b92538ac8f98066.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/images/bcc0ef24d3cbd7f2b06e0e5ae2d31b7ff6badefcbb612fae1b92538ac8f98066.jpg)

### Tables

![12f37d6b4109bd54463b94bece77aeff9d19f4cbff3b19ddc6ffd1a7a4b8d3c9.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/tables/12f37d6b4109bd54463b94bece77aeff9d19f4cbff3b19ddc6ffd1a7a4b8d3c9.jpg)

![17a4613f28ed79178cbf06e802019e0e7402533386da4e8458030715d6efe140.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/tables/17a4613f28ed79178cbf06e802019e0e7402533386da4e8458030715d6efe140.jpg)

![189a7f25750097ba358351152c9305e7cee05a4fe542ab91e2bdad1f0fc014a1.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/tables/189a7f25750097ba358351152c9305e7cee05a4fe542ab91e2bdad1f0fc014a1.jpg)

![1b51d2153500c597f26b99bac8f9ec739ef9aca42fb45b89d1f0d7139b0358a0.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/tables/1b51d2153500c597f26b99bac8f9ec739ef9aca42fb45b89d1f0d7139b0358a0.jpg)

![28239214a96aa6b303b61d63bf63cba6f334f33b2af7f77852e9e0003c4793aa.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/tables/28239214a96aa6b303b61d63bf63cba6f334f33b2af7f77852e9e0003c4793aa.jpg)

![798dd9475ee65b4ce3e4a5eaccf4fb853e3eb032d6cf27fce3d810de6a1c5e16.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/tables/798dd9475ee65b4ce3e4a5eaccf4fb853e3eb032d6cf27fce3d810de6a1c5e16.jpg)

![79dcd922a118722c4ae84001d678a7f695d410967da8a3cf574de50dcb9792b4.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/tables/79dcd922a118722c4ae84001d678a7f695d410967da8a3cf574de50dcb9792b4.jpg)

![8c8cabdf4e26c09f07dfcca62883c370bbb46216d49e56d5cb6e64eac23493b5.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/tables/8c8cabdf4e26c09f07dfcca62883c370bbb46216d49e56d5cb6e64eac23493b5.jpg)

![a255a6dcb3202b4bbcde5bee26af54539a1d44bc045194a43f9e45c1c4f067a7.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/tables/a255a6dcb3202b4bbcde5bee26af54539a1d44bc045194a43f9e45c1c4f067a7.jpg)

![a916cc62251a0b3a90d19bf95c16049c4a2162bb723c695a545b9a4dbb16b20d.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/tables/a916cc62251a0b3a90d19bf95c16049c4a2162bb723c695a545b9a4dbb16b20d.jpg)

![c0f8143273a32019328d62284c4e13939c7ea241099e7b4e4f806e254cd964da.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/tables/c0f8143273a32019328d62284c4e13939c7ea241099e7b4e4f806e254cd964da.jpg)

![c1d37c9f27de49e2e9a6ad804f28db7d3217b73aba1813aba1d97bee8faec75b.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/tables/c1d37c9f27de49e2e9a6ad804f28db7d3217b73aba1813aba1d97bee8faec75b.jpg)

![c97335d61b4837e793ee9648506ebefe40acbe33aa1c035bb299d3c6a4f2f632.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/tables/c97335d61b4837e793ee9648506ebefe40acbe33aa1c035bb299d3c6a4f2f632.jpg)

![d30637776c1f9c9a806c58ae4889753219e0c4d7d27a48dbf0b3bd66cb97fe0f.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/tables/d30637776c1f9c9a806c58ae4889753219e0c4d7d27a48dbf0b3bd66cb97fe0f.jpg)

![d313f741407f7300292e5ba123ddaf4eac228cf522c0ec8890eaf2501b5f702a.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/tables/d313f741407f7300292e5ba123ddaf4eac228cf522c0ec8890eaf2501b5f702a.jpg)

![e470abc019aae0b5aa413c7efba1c90a19f0e04546399b5fbc2b99a0679511a0.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/tables/e470abc019aae0b5aa413c7efba1c90a19f0e04546399b5fbc2b99a0679511a0.jpg)

![efdea449959265d2542b107a625f42de373d8cdb22553a84f048e0b44568bb76.jpg](../iclr_results/3214_Temporal Flexibility in Spiking Neural Networks_ Towards Generalization Across Time Steps and Deploy/tables/efdea449959265d2542b107a625f42de373d8cdb22553a84f048e0b44568bb76.jpg)

## OmniPhysGS: 3D Constitutive Gaussians for General Physics-Based Dynamics Generation


### Images

![06b28df3cddcaa354c7bc9cc691ca2d7864044299ca29401cd70da1290ed69b6.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/images/06b28df3cddcaa354c7bc9cc691ca2d7864044299ca29401cd70da1290ed69b6.jpg)

![2d4e90f6a53d452d19079cbc6ab176ac67215daa1caa9a761d9c64e532224ba7.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/images/2d4e90f6a53d452d19079cbc6ab176ac67215daa1caa9a761d9c64e532224ba7.jpg)

![32f53661672404b063874a74beb0a3e3727feeb38c492c261b78937490e136cb.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/images/32f53661672404b063874a74beb0a3e3727feeb38c492c261b78937490e136cb.jpg)

![3815033a6b965091eb6c7d387ea3005305a2e2b1c2c98b14dc974e3e5074bd47.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/images/3815033a6b965091eb6c7d387ea3005305a2e2b1c2c98b14dc974e3e5074bd47.jpg)

![412cf62404a0f1b317dd53e0e2631aa60ea57792bc40fd76e41dd6d44dd5386a.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/images/412cf62404a0f1b317dd53e0e2631aa60ea57792bc40fd76e41dd6d44dd5386a.jpg)

![42b2a441bef2f0ce8d18c8f08d113a8204aab3e75b9059c42a71ded0d87c3ac6.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/images/42b2a441bef2f0ce8d18c8f08d113a8204aab3e75b9059c42a71ded0d87c3ac6.jpg)

![43a6133e58a81de470eacf074c26a91447d8b1041914b9af34ef1ceb82df235d.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/images/43a6133e58a81de470eacf074c26a91447d8b1041914b9af34ef1ceb82df235d.jpg)

![4adf03ccc1aadf1c785eb6fe7cd73b64170c54fd721beec9d68e1fa56a49b8bd.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/images/4adf03ccc1aadf1c785eb6fe7cd73b64170c54fd721beec9d68e1fa56a49b8bd.jpg)

![4c4ab4fef8cf90f3d1a5ec7175cf610af7c1515cbd23835663ecaa89cc9f0721.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/images/4c4ab4fef8cf90f3d1a5ec7175cf610af7c1515cbd23835663ecaa89cc9f0721.jpg)

![593039fb2a68cd20f640ade6cb24b09df295d8b8ef335da736ef8e8d36924d14.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/images/593039fb2a68cd20f640ade6cb24b09df295d8b8ef335da736ef8e8d36924d14.jpg)

![71b2b9d3ef0c98f8b99c23775fba4259114c647592179040f48d5cad5b23bcd6.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/images/71b2b9d3ef0c98f8b99c23775fba4259114c647592179040f48d5cad5b23bcd6.jpg)

![79f6f32ed13bd72e6838d93f032c14a7caa980eb7f371ff758a087070800fa74.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/images/79f6f32ed13bd72e6838d93f032c14a7caa980eb7f371ff758a087070800fa74.jpg)

![80c197d1af2bba6846e7b417d04de88c3238862a85373b826cc2445a354ad59b.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/images/80c197d1af2bba6846e7b417d04de88c3238862a85373b826cc2445a354ad59b.jpg)

![8cb7f2c891f798d3abf208dc1b9c6efc4aa1e8429daf2ebe8ea9b9a5308116dc.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/images/8cb7f2c891f798d3abf208dc1b9c6efc4aa1e8429daf2ebe8ea9b9a5308116dc.jpg)

![ab5d058707a432a327cdd30f54f346499fe6463112a40f46eeab7e02178c9354.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/images/ab5d058707a432a327cdd30f54f346499fe6463112a40f46eeab7e02178c9354.jpg)

![b39a283fd0362314bbaa240cfdd23bef2404f95514f5b11914f504a8c8f1cf12.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/images/b39a283fd0362314bbaa240cfdd23bef2404f95514f5b11914f504a8c8f1cf12.jpg)

![d355c01d07687d1e2fea932cdb2beefffec8ae57cbbbc57d669457cdde1d20fe.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/images/d355c01d07687d1e2fea932cdb2beefffec8ae57cbbbc57d669457cdde1d20fe.jpg)

![d4242bbd164e81374eb47ff96d9d489d7f396cbd123ecac0d780d9bc0240fbdc.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/images/d4242bbd164e81374eb47ff96d9d489d7f396cbd123ecac0d780d9bc0240fbdc.jpg)

![e272373474a1a8d399fc63c83a0e43d1c73a1bc172fe8661f2aeff3bfda0a348.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/images/e272373474a1a8d399fc63c83a0e43d1c73a1bc172fe8661f2aeff3bfda0a348.jpg)

![ed615dac8954fb45087a37bb51addf0baaccb1b8a1d14f017d0b6f308183b428.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/images/ed615dac8954fb45087a37bb51addf0baaccb1b8a1d14f017d0b6f308183b428.jpg)

![feb3d9362dd705b98193e74ca8574f64ee1b6e30b5540a90ec3cb94e8a28c731.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/images/feb3d9362dd705b98193e74ca8574f64ee1b6e30b5540a90ec3cb94e8a28c731.jpg)

### Tables

![2a1e601059241ea08bdb129c52f8d107b5be53a8805ed9912aa95367a8635bef.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/tables/2a1e601059241ea08bdb129c52f8d107b5be53a8805ed9912aa95367a8635bef.jpg)

![51eba6b8baa7e2d6af7605f723dc0fd231de85d6810ab46dcf74855805f47320.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/tables/51eba6b8baa7e2d6af7605f723dc0fd231de85d6810ab46dcf74855805f47320.jpg)

![758a95549f195316061e337f860982cafa014b4ced7a7eefb563ae28aa06cf46.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/tables/758a95549f195316061e337f860982cafa014b4ced7a7eefb563ae28aa06cf46.jpg)

![8aa1360a21fa8b50d8cbdb3b0940aa33e6fb956872f46f1c570698467e6ef2b0.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/tables/8aa1360a21fa8b50d8cbdb3b0940aa33e6fb956872f46f1c570698467e6ef2b0.jpg)

![c2747b6593df3743dc8c8b6576dfb6502d782ff3e2146bc6081c3f13c3183889.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/tables/c2747b6593df3743dc8c8b6576dfb6502d782ff3e2146bc6081c3f13c3183889.jpg)

![ee3f892a5042a0c17c9a5b67492be90e807e3dc640eededd63f8ce48e46c2a0c.jpg](../iclr_results/3215_OmniPhysGS_ 3D Constitutive Gaussians for General Physics-Based Dynamics Generation/tables/ee3f892a5042a0c17c9a5b67492be90e807e3dc640eededd63f8ce48e46c2a0c.jpg)

## SqueezeAttention: 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget


### Images

![5d3e20f71a95314aea27f7345d3bf071f86d948ba8597ce8d9de28103f8062b2.jpg](../iclr_results/3216_SqueezeAttention_ 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget/images/5d3e20f71a95314aea27f7345d3bf071f86d948ba8597ce8d9de28103f8062b2.jpg)

![86e1fe12b754085633bf624d0a85e927d5140bd0b6c1490dd6231eeda8c02bec.jpg](../iclr_results/3216_SqueezeAttention_ 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget/images/86e1fe12b754085633bf624d0a85e927d5140bd0b6c1490dd6231eeda8c02bec.jpg)

![d22126faed3cac055d52166b6a7e5db88188939fbc7f39feb15160a6b177d09f.jpg](../iclr_results/3216_SqueezeAttention_ 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget/images/d22126faed3cac055d52166b6a7e5db88188939fbc7f39feb15160a6b177d09f.jpg)

![d7ced06c772939962426e9e04092543d16459dabfc700e5e2207bbe0cdcf1b90.jpg](../iclr_results/3216_SqueezeAttention_ 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget/images/d7ced06c772939962426e9e04092543d16459dabfc700e5e2207bbe0cdcf1b90.jpg)

### Tables

![23f2928e60fabe1dd235cf2a5f716abb976c02d8a2900e13c9e578819a63f38b.jpg](../iclr_results/3216_SqueezeAttention_ 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget/tables/23f2928e60fabe1dd235cf2a5f716abb976c02d8a2900e13c9e578819a63f38b.jpg)

![57ab74164f02edaea9b3818d54e5f069f02cf5c802c4e36e8998f69831bdaf34.jpg](../iclr_results/3216_SqueezeAttention_ 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget/tables/57ab74164f02edaea9b3818d54e5f069f02cf5c802c4e36e8998f69831bdaf34.jpg)

![7bd4aa5b7217ad2dc6a52b201d16ed2a02bfe08b7c7b4b72d4844ea4fd74020b.jpg](../iclr_results/3216_SqueezeAttention_ 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget/tables/7bd4aa5b7217ad2dc6a52b201d16ed2a02bfe08b7c7b4b72d4844ea4fd74020b.jpg)

![8ab819ffefc0a3a61df79bf11021dfdac926ba974aeec531f6fabf8f9a85ec93.jpg](../iclr_results/3216_SqueezeAttention_ 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget/tables/8ab819ffefc0a3a61df79bf11021dfdac926ba974aeec531f6fabf8f9a85ec93.jpg)

![96915e7d2d95b1688d7096b9451fa0ac1abba72aae57c1b06fbe17095f25757d.jpg](../iclr_results/3216_SqueezeAttention_ 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget/tables/96915e7d2d95b1688d7096b9451fa0ac1abba72aae57c1b06fbe17095f25757d.jpg)

![a0f3d5ebc8b68c2c0d48307ffb2f2de961cc229cb07b28c5a4f0349931a45574.jpg](../iclr_results/3216_SqueezeAttention_ 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget/tables/a0f3d5ebc8b68c2c0d48307ffb2f2de961cc229cb07b28c5a4f0349931a45574.jpg)

![d35fa1345349a05eb28fc727a1a46c163fc79b889b13b89233e22bd5170f4f6b.jpg](../iclr_results/3216_SqueezeAttention_ 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget/tables/d35fa1345349a05eb28fc727a1a46c163fc79b889b13b89233e22bd5170f4f6b.jpg)

![eddf112cc5266c39583b9f7cf8aa1c31d5f6b0d1caea6340652a5bfdb73ef889.jpg](../iclr_results/3216_SqueezeAttention_ 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget/tables/eddf112cc5266c39583b9f7cf8aa1c31d5f6b0d1caea6340652a5bfdb73ef889.jpg)

![f17d297d6b33dfdc2b0629737461b7d35a2dd021e3ddecfdf1b1028212884e5a.jpg](../iclr_results/3216_SqueezeAttention_ 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget/tables/f17d297d6b33dfdc2b0629737461b7d35a2dd021e3ddecfdf1b1028212884e5a.jpg)

## Hyper-Connections


### Images

![1777f36abb971c5a6cdc14bbefe11c7ebb73ab548ace6d751927bb33435a67f8.jpg](../iclr_results/3217_Hyper-Connections/images/1777f36abb971c5a6cdc14bbefe11c7ebb73ab548ace6d751927bb33435a67f8.jpg)

![1b6bbbf9818673f44d9dd11d239fc71dd8e16b7a218c957625a54af5471bd1e7.jpg](../iclr_results/3217_Hyper-Connections/images/1b6bbbf9818673f44d9dd11d239fc71dd8e16b7a218c957625a54af5471bd1e7.jpg)

![2914548973c5ecbff6ee84fed397f029a42d89f3e2d2529e61a3ff6165af90a4.jpg](../iclr_results/3217_Hyper-Connections/images/2914548973c5ecbff6ee84fed397f029a42d89f3e2d2529e61a3ff6165af90a4.jpg)

![469c21865043af400b8eb6811fb1e93095b9998f4b5965b10ee05a1adf4b9065.jpg](../iclr_results/3217_Hyper-Connections/images/469c21865043af400b8eb6811fb1e93095b9998f4b5965b10ee05a1adf4b9065.jpg)

![498e514aadb82d54559d13e2142e578cd43995fb76e33945cdcc311c640e291b.jpg](../iclr_results/3217_Hyper-Connections/images/498e514aadb82d54559d13e2142e578cd43995fb76e33945cdcc311c640e291b.jpg)

![51a95860d26f2bf2dc7a76c938b1f02a49ccbd35011d7baf766611746df49e99.jpg](../iclr_results/3217_Hyper-Connections/images/51a95860d26f2bf2dc7a76c938b1f02a49ccbd35011d7baf766611746df49e99.jpg)

![61b6388c2b2fea02472de3f46c3d17448ca4f42f70874d1523878efe444d9fea.jpg](../iclr_results/3217_Hyper-Connections/images/61b6388c2b2fea02472de3f46c3d17448ca4f42f70874d1523878efe444d9fea.jpg)

![66304ec3b16efe0d0e373685b84c270e27af9980e03db2de5f32fee95e262ca2.jpg](../iclr_results/3217_Hyper-Connections/images/66304ec3b16efe0d0e373685b84c270e27af9980e03db2de5f32fee95e262ca2.jpg)

![6efb94cc59a44e330b27e4de66cebe6e75853b5c908579e5aadd598845dd4c72.jpg](../iclr_results/3217_Hyper-Connections/images/6efb94cc59a44e330b27e4de66cebe6e75853b5c908579e5aadd598845dd4c72.jpg)

![8df017faa9b76c113f463bf61ec3593100019dbcf37549c4b921a532913ae119.jpg](../iclr_results/3217_Hyper-Connections/images/8df017faa9b76c113f463bf61ec3593100019dbcf37549c4b921a532913ae119.jpg)

![9995b2959606a004a03acd67de6fb9dcc0c24385d52bc8b2889c2c2a64f6a504.jpg](../iclr_results/3217_Hyper-Connections/images/9995b2959606a004a03acd67de6fb9dcc0c24385d52bc8b2889c2c2a64f6a504.jpg)

![9d7239d58d635cbb591a32fc9fdef7120cf9d7b688fc1dcb70a1fa31349577c1.jpg](../iclr_results/3217_Hyper-Connections/images/9d7239d58d635cbb591a32fc9fdef7120cf9d7b688fc1dcb70a1fa31349577c1.jpg)

![aa2113c8ad3974280371a772f05e486893d733eacda63d7a8f005c99c8b3f9eb.jpg](../iclr_results/3217_Hyper-Connections/images/aa2113c8ad3974280371a772f05e486893d733eacda63d7a8f005c99c8b3f9eb.jpg)

![ab0c172a380620939d29a524edd063497ec6f190a1f778abea54e4b59d4a60ee.jpg](../iclr_results/3217_Hyper-Connections/images/ab0c172a380620939d29a524edd063497ec6f190a1f778abea54e4b59d4a60ee.jpg)

![b166cd24f1ea19e41a6e7c56fccea91354246ffb9232966ebf769431dc1fef1d.jpg](../iclr_results/3217_Hyper-Connections/images/b166cd24f1ea19e41a6e7c56fccea91354246ffb9232966ebf769431dc1fef1d.jpg)

![ba5ee78a88d16aa3806b426889d64d647670286bd9538c5d8883441360966658.jpg](../iclr_results/3217_Hyper-Connections/images/ba5ee78a88d16aa3806b426889d64d647670286bd9538c5d8883441360966658.jpg)

![bf345cd726ce8aeb07abd12a8698d4bbe3b2ce2f12196646ff1e4183a8ad315c.jpg](../iclr_results/3217_Hyper-Connections/images/bf345cd726ce8aeb07abd12a8698d4bbe3b2ce2f12196646ff1e4183a8ad315c.jpg)

![c6e4729b27c721ed3f2acac78a0c2222578132d8161e383883a59a0908ab6947.jpg](../iclr_results/3217_Hyper-Connections/images/c6e4729b27c721ed3f2acac78a0c2222578132d8161e383883a59a0908ab6947.jpg)

### Tables

![024a78fc6db800ce4a4a89e42c7a08385304a72729aef8e21a7eaf8c8450356a.jpg](../iclr_results/3217_Hyper-Connections/tables/024a78fc6db800ce4a4a89e42c7a08385304a72729aef8e21a7eaf8c8450356a.jpg)

![070d27ad56ff762eb0b424d817e5cff0c95b2d07ecd0540d36c24cafcc45b58f.jpg](../iclr_results/3217_Hyper-Connections/tables/070d27ad56ff762eb0b424d817e5cff0c95b2d07ecd0540d36c24cafcc45b58f.jpg)

![0e805e8d874ab85ec8ec3aca90015877e9d33b3f07ac9310f8fc72a7adf32b69.jpg](../iclr_results/3217_Hyper-Connections/tables/0e805e8d874ab85ec8ec3aca90015877e9d33b3f07ac9310f8fc72a7adf32b69.jpg)

![13d5dc1e0ddc7f639e32a667245aa8d2acd875207da59497dd93ebe3947cc500.jpg](../iclr_results/3217_Hyper-Connections/tables/13d5dc1e0ddc7f639e32a667245aa8d2acd875207da59497dd93ebe3947cc500.jpg)

![1b9870fb4a9660dca8939c57d7fe5d7f5bee6cabb60f34f2986d7defbd3929ee.jpg](../iclr_results/3217_Hyper-Connections/tables/1b9870fb4a9660dca8939c57d7fe5d7f5bee6cabb60f34f2986d7defbd3929ee.jpg)

![4fd4d697dc777b53203f66e9c256d76f40deb7c21e98885270703f5ef05b7fe4.jpg](../iclr_results/3217_Hyper-Connections/tables/4fd4d697dc777b53203f66e9c256d76f40deb7c21e98885270703f5ef05b7fe4.jpg)

![66c62471340487727ce7ccdc67c088e4e55e57e7b86a6797e132e1031c7f61d7.jpg](../iclr_results/3217_Hyper-Connections/tables/66c62471340487727ce7ccdc67c088e4e55e57e7b86a6797e132e1031c7f61d7.jpg)

![69bc68fe6cf9437384a6c2141de74b689386a251856311d0baeb543311720373.jpg](../iclr_results/3217_Hyper-Connections/tables/69bc68fe6cf9437384a6c2141de74b689386a251856311d0baeb543311720373.jpg)

![72aa5c5e889cf14091fd83332a2bd3326ce0354ef3adc56f1e76638956d2760a.jpg](../iclr_results/3217_Hyper-Connections/tables/72aa5c5e889cf14091fd83332a2bd3326ce0354ef3adc56f1e76638956d2760a.jpg)

![74379d9163c744a3a675f3993b92f5b7e48babf80a0aa3506900435274ed4743.jpg](../iclr_results/3217_Hyper-Connections/tables/74379d9163c744a3a675f3993b92f5b7e48babf80a0aa3506900435274ed4743.jpg)

![7c0656758586e3bde2e38cf4ba9b235b2decd063d1c4a69fc057be602d3da9af.jpg](../iclr_results/3217_Hyper-Connections/tables/7c0656758586e3bde2e38cf4ba9b235b2decd063d1c4a69fc057be602d3da9af.jpg)

![987ce99109e94a371fc4f6eb6de7033094a7e31b2ae56ccc63ceefdc6e9feb3a.jpg](../iclr_results/3217_Hyper-Connections/tables/987ce99109e94a371fc4f6eb6de7033094a7e31b2ae56ccc63ceefdc6e9feb3a.jpg)

![ab85f1f5b61251a8bb585df14ca482b1e461092ebea9dd94c85ece021433ced6.jpg](../iclr_results/3217_Hyper-Connections/tables/ab85f1f5b61251a8bb585df14ca482b1e461092ebea9dd94c85ece021433ced6.jpg)

![c1cc012ec5e5039248e947bce824941dec542c379f7d3b3ee8677e55c4d8fcd2.jpg](../iclr_results/3217_Hyper-Connections/tables/c1cc012ec5e5039248e947bce824941dec542c379f7d3b3ee8677e55c4d8fcd2.jpg)

![c7938fb16cfd55231017abc9bbfb5228db6134fdf6273e5b7c63905c4feade92.jpg](../iclr_results/3217_Hyper-Connections/tables/c7938fb16cfd55231017abc9bbfb5228db6134fdf6273e5b7c63905c4feade92.jpg)

![c8b7f4e0d5161c6921988bac38ab8ca7d588961203536f420423b15d4331a08b.jpg](../iclr_results/3217_Hyper-Connections/tables/c8b7f4e0d5161c6921988bac38ab8ca7d588961203536f420423b15d4331a08b.jpg)

![ce25cb906b0d8048a5586d6b121338d0c86b8247f33e03c31d7973909ec2deb7.jpg](../iclr_results/3217_Hyper-Connections/tables/ce25cb906b0d8048a5586d6b121338d0c86b8247f33e03c31d7973909ec2deb7.jpg)

![e29111f5f0e9e470596294abe48fb23d516fee3586e12685333832cf97fcd865.jpg](../iclr_results/3217_Hyper-Connections/tables/e29111f5f0e9e470596294abe48fb23d516fee3586e12685333832cf97fcd865.jpg)

![e2f2a5b3b27fd01a83035acea46e33739688ca2bf2115537b7e23491806d06c7.jpg](../iclr_results/3217_Hyper-Connections/tables/e2f2a5b3b27fd01a83035acea46e33739688ca2bf2115537b7e23491806d06c7.jpg)

![f43bf56431b95f4fd45db9c3d466128168a62aa5f9cacd5e536094164615bad8.jpg](../iclr_results/3217_Hyper-Connections/tables/f43bf56431b95f4fd45db9c3d466128168a62aa5f9cacd5e536094164615bad8.jpg)

## PRDP: Progressively Refined Differentiable Physics


### Images

![047a931994fffb00b1e6df7347c750d8eac67870fe12f93f6d2e5c3955c9ce5d.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/047a931994fffb00b1e6df7347c750d8eac67870fe12f93f6d2e5c3955c9ce5d.jpg)

![04fdd7dce3540266eae2196b7bb49ea5a6b10a1ee02f5af40aab0c7f9adcf917.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/04fdd7dce3540266eae2196b7bb49ea5a6b10a1ee02f5af40aab0c7f9adcf917.jpg)

![09a31467357b35e2e36f1a9d29011067733f92ff695e6743b25f50f0d9aef428.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/09a31467357b35e2e36f1a9d29011067733f92ff695e6743b25f50f0d9aef428.jpg)

![1332588d02dce6198c13b6e6a90403e20287ce5c1341b1b3f476811660603396.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/1332588d02dce6198c13b6e6a90403e20287ce5c1341b1b3f476811660603396.jpg)

![1a7554e59eb4d6639af946655f2c8611da26f43e06d2babe46fd4dd7dee0e0f4.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/1a7554e59eb4d6639af946655f2c8611da26f43e06d2babe46fd4dd7dee0e0f4.jpg)

![2094d47730d768625b0749981d5d25e73188868d9380f538900ecd360b4cdce2.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/2094d47730d768625b0749981d5d25e73188868d9380f538900ecd360b4cdce2.jpg)

![3c0f8abda1a178e1537244513f934dec55cf770ac5d91c09bbafca466884fc65.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/3c0f8abda1a178e1537244513f934dec55cf770ac5d91c09bbafca466884fc65.jpg)

![4bbbced7d425ecd52118c4b4b3790dd12203e04fd2dd6994f9110b3e5d155689.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/4bbbced7d425ecd52118c4b4b3790dd12203e04fd2dd6994f9110b3e5d155689.jpg)

![56d4ce0c13e12f4e5dbff941ffc456bde64582a8f3ca20538d24d3f1bd71031c.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/56d4ce0c13e12f4e5dbff941ffc456bde64582a8f3ca20538d24d3f1bd71031c.jpg)

![58d4f1dcabad5f88fafe7d5d8ba084a039cbffb7e93c5d366693933441146474.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/58d4f1dcabad5f88fafe7d5d8ba084a039cbffb7e93c5d366693933441146474.jpg)

![64db13c6a1dc1d841c5d2090d7098ca063eaa459e9f3aa3f27cfb7903c1b7a24.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/64db13c6a1dc1d841c5d2090d7098ca063eaa459e9f3aa3f27cfb7903c1b7a24.jpg)

![657af8207872ad0ba70dfd51123049dad23bd7ea0a59ee4f0aa81d9ca85e20c7.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/657af8207872ad0ba70dfd51123049dad23bd7ea0a59ee4f0aa81d9ca85e20c7.jpg)

![65bde681ffa2edce1c332612d68b898c989f5bf3432ff1bcc3bc892f47d3ed52.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/65bde681ffa2edce1c332612d68b898c989f5bf3432ff1bcc3bc892f47d3ed52.jpg)

![6f831aef58e4fac669780e20b08d07033c73a1f858db2bb78e60f43ba6d3cbe0.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/6f831aef58e4fac669780e20b08d07033c73a1f858db2bb78e60f43ba6d3cbe0.jpg)

![72f420f16990ef71bfe1f1dba96376a09f29fe2b1577f38722b050e87d085c3e.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/72f420f16990ef71bfe1f1dba96376a09f29fe2b1577f38722b050e87d085c3e.jpg)

![8216a312f533139507243b7a926f9dc9672375564c73859996a40ca975c684cd.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/8216a312f533139507243b7a926f9dc9672375564c73859996a40ca975c684cd.jpg)

![9175f3626bbd5e53cd9154a768da3d4cbbd62d6ef214da10b6e007e89b5245a0.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/9175f3626bbd5e53cd9154a768da3d4cbbd62d6ef214da10b6e007e89b5245a0.jpg)

![941b07a627d768ebf95cecbcc71b97f3e62dc4c94bdf104475dc34a1423a96a1.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/941b07a627d768ebf95cecbcc71b97f3e62dc4c94bdf104475dc34a1423a96a1.jpg)

![a830185311720a35fe115b000edcce9ee48b688f19f29ec8cbdc8350a0fc7df2.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/a830185311720a35fe115b000edcce9ee48b688f19f29ec8cbdc8350a0fc7df2.jpg)

![b1ea04329fbda10a80b2a91a78f4ef57514462c0c8d86f41b0595ea1d3f39fcc.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/b1ea04329fbda10a80b2a91a78f4ef57514462c0c8d86f41b0595ea1d3f39fcc.jpg)

![bada07fa398281cd8dbe5c1a521c3f7c40b6e51093f09d01957c7940f3f8cfd3.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/bada07fa398281cd8dbe5c1a521c3f7c40b6e51093f09d01957c7940f3f8cfd3.jpg)

![bdb4e3ec01b4bdeecf8360c1742f89c81e7abb870292386fda1134b51f734511.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/bdb4e3ec01b4bdeecf8360c1742f89c81e7abb870292386fda1134b51f734511.jpg)

![c07f877873ddc23450e41ada64b4c27c98ea3069c803e49a939d574f825340e7.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/c07f877873ddc23450e41ada64b4c27c98ea3069c803e49a939d574f825340e7.jpg)

![c0cd37a0aceea2c3d1d46c74e8534a85cd469f98f76d80dd23031bdcd1ed6ecf.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/c0cd37a0aceea2c3d1d46c74e8534a85cd469f98f76d80dd23031bdcd1ed6ecf.jpg)

![ce6c4ba41660b8504665b33f25ff2c903267497978d1e04fe16605773a66e27d.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/ce6c4ba41660b8504665b33f25ff2c903267497978d1e04fe16605773a66e27d.jpg)

![da10fe13f981b0c14c1d5735cc0a04c6bfff3b6e1ea4a99ea0f586614424891c.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/da10fe13f981b0c14c1d5735cc0a04c6bfff3b6e1ea4a99ea0f586614424891c.jpg)

![db04fa280ce95df89efe7c08fab3c399f2b8c4fecc4f0482c5e9ab0b3c586ef2.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/db04fa280ce95df89efe7c08fab3c399f2b8c4fecc4f0482c5e9ab0b3c586ef2.jpg)

![defacb6c7923bea0989ff94e24081cb8c61c6ec62479addf24f08cc8f431def5.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/defacb6c7923bea0989ff94e24081cb8c61c6ec62479addf24f08cc8f431def5.jpg)

![e959e68fc125d88227632bc8609e852a6eaf89e3c6967b02afefc9a1f93d08f9.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/e959e68fc125d88227632bc8609e852a6eaf89e3c6967b02afefc9a1f93d08f9.jpg)

![ea7c7b85fabe23b2a992a586b9377342eb7dcbe9eb4be38512c1d26df52a2637.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/images/ea7c7b85fabe23b2a992a586b9377342eb7dcbe9eb4be38512c1d26df52a2637.jpg)

### Tables

![11a1b3011b9b4d5afc92af9cc322c99c77dd20c329dc1d7a36730579151d5dc1.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/tables/11a1b3011b9b4d5afc92af9cc322c99c77dd20c329dc1d7a36730579151d5dc1.jpg)

![498200074bac76dc88bfb56ba0352135265ff55d7f7b7be7ae12d2f5c74e060c.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/tables/498200074bac76dc88bfb56ba0352135265ff55d7f7b7be7ae12d2f5c74e060c.jpg)

![53a4c35e1d05c6e7e98f9b028f5b949148ea17fe4d99800f5db553bb911162e5.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/tables/53a4c35e1d05c6e7e98f9b028f5b949148ea17fe4d99800f5db553bb911162e5.jpg)

![cb5e7ac65b09840833ca0f29dc05cad097f9d8988d8bd04b2307db1ad09260a4.jpg](../iclr_results/3218_PRDP_ Progressively Refined Differentiable Physics/tables/cb5e7ac65b09840833ca0f29dc05cad097f9d8988d8bd04b2307db1ad09260a4.jpg)

## Inverse Constitutional AI: Compressing Preferences into Principles


### Images

![04e1f2d9610b7e5626386865b89d662de41cdc403e163bc50fe31c3fdef12d40.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/images/04e1f2d9610b7e5626386865b89d662de41cdc403e163bc50fe31c3fdef12d40.jpg)

![260ce6661aa416be2756de33e86db1c92473937e46f294e3d5cf44454ccda9cb.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/images/260ce6661aa416be2756de33e86db1c92473937e46f294e3d5cf44454ccda9cb.jpg)

![3bf21135a809633ae581d039be28d8032338cfbbe2f9bb082f41d35b2a58e9a6.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/images/3bf21135a809633ae581d039be28d8032338cfbbe2f9bb082f41d35b2a58e9a6.jpg)

![4f9879619d09ca5bb5a5fb6acce34a4d4a5ac03b1979e66cf9b2b66991c421dc.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/images/4f9879619d09ca5bb5a5fb6acce34a4d4a5ac03b1979e66cf9b2b66991c421dc.jpg)

![627f1c63eac8a0f3e80a7d1771d6c19fd5a00e95db78c66ccbc89dd0b310c3f0.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/images/627f1c63eac8a0f3e80a7d1771d6c19fd5a00e95db78c66ccbc89dd0b310c3f0.jpg)

![9af19f93122268a3b861e46e5d38cb9ac5729ec4874f42ede40b7ef25d25734a.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/images/9af19f93122268a3b861e46e5d38cb9ac5729ec4874f42ede40b7ef25d25734a.jpg)

![a104d8ebc80100d2171e08d4b552126a4815360a56b1ef8425b58265325b49c2.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/images/a104d8ebc80100d2171e08d4b552126a4815360a56b1ef8425b58265325b49c2.jpg)

![a132e05120e93081c253811d5ce9ce0c3a1f2bb4e6b419f1988e61a4d9ba5c26.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/images/a132e05120e93081c253811d5ce9ce0c3a1f2bb4e6b419f1988e61a4d9ba5c26.jpg)

![bf5d3f8f9164286191ed317bf62d6a00e7aba79dfcebbb121476a559ce0b3529.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/images/bf5d3f8f9164286191ed317bf62d6a00e7aba79dfcebbb121476a559ce0b3529.jpg)

![e3ec656098681ff77d6385592b21331fb61ac53c21db94c5b7f8587ff397e844.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/images/e3ec656098681ff77d6385592b21331fb61ac53c21db94c5b7f8587ff397e844.jpg)

### Tables

![00e0c8feefa4c3ee0626cb186669301fb66f5806b430e87b1d164e9ce0e3b573.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/tables/00e0c8feefa4c3ee0626cb186669301fb66f5806b430e87b1d164e9ce0e3b573.jpg)

![0c209688f38e7014548a2d0ba7b92c602101e1094bce112202db329b1441aaa8.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/tables/0c209688f38e7014548a2d0ba7b92c602101e1094bce112202db329b1441aaa8.jpg)

![0df20c329b2e97213d75770942a5e8a7ac934b876a32b42f85b2c3548707acf9.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/tables/0df20c329b2e97213d75770942a5e8a7ac934b876a32b42f85b2c3548707acf9.jpg)

![116d2048d2baccd261da50525125992f9c1d1814faca7793382f65a0ae130bab.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/tables/116d2048d2baccd261da50525125992f9c1d1814faca7793382f65a0ae130bab.jpg)

![39e31eefa4408be6307e623bf8714db131fb02879a2aab42900d055d90e1f1cd.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/tables/39e31eefa4408be6307e623bf8714db131fb02879a2aab42900d055d90e1f1cd.jpg)

![40e994eb1bd1c2dfe1459a8ddbea496114074e28231d91646392d16ec7a68d93.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/tables/40e994eb1bd1c2dfe1459a8ddbea496114074e28231d91646392d16ec7a68d93.jpg)

![5a0031e82eecad11a3b6e3a25825c420dcf4ea625a14fad3f0e84c14dde409d0.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/tables/5a0031e82eecad11a3b6e3a25825c420dcf4ea625a14fad3f0e84c14dde409d0.jpg)

![673df21ca342da79f4228c1ec2f2d3c7e378ae1ed300974a591ec6f419a047e4.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/tables/673df21ca342da79f4228c1ec2f2d3c7e378ae1ed300974a591ec6f419a047e4.jpg)

![69e6e8b3ce750ca915eaa887a4abfcc45b9cb825e2196c5d5813e2870c2761e8.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/tables/69e6e8b3ce750ca915eaa887a4abfcc45b9cb825e2196c5d5813e2870c2761e8.jpg)

![76df3a0df3736da2c32f51b94607007bd1baea81290724a7b946db6b55ed8233.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/tables/76df3a0df3736da2c32f51b94607007bd1baea81290724a7b946db6b55ed8233.jpg)

![82bb173279660a6ca9aecaed2b60b158c51b21ff33b150081b153c31ec1a2697.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/tables/82bb173279660a6ca9aecaed2b60b158c51b21ff33b150081b153c31ec1a2697.jpg)

![8f07dab7bc094955755e2f9aa92ce2170c1432b4c4a37db2bcf796c91e115d3d.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/tables/8f07dab7bc094955755e2f9aa92ce2170c1432b4c4a37db2bcf796c91e115d3d.jpg)

![8ffbcc97b58ec7bb8f3709d01f205723c0812cbf57142c115993f006aba533a3.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/tables/8ffbcc97b58ec7bb8f3709d01f205723c0812cbf57142c115993f006aba533a3.jpg)

![aa9e6dbb76aca71e7597008969391466e5005c5bb2503ab777e86af6101336df.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/tables/aa9e6dbb76aca71e7597008969391466e5005c5bb2503ab777e86af6101336df.jpg)

![bf9f953b46617e6cf900a40beeeb5496f883ec5fca8b74ee4c051ede00e38442.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/tables/bf9f953b46617e6cf900a40beeeb5496f883ec5fca8b74ee4c051ede00e38442.jpg)

![c9dacb3c764ff31d6210989ba2d195839a645b6b896dfa0c45eb640ddc9191af.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/tables/c9dacb3c764ff31d6210989ba2d195839a645b6b896dfa0c45eb640ddc9191af.jpg)

![ca0c23976423ec0fe8c3511461ebf0c8e036c1ebf2a05872a14ace3b67772106.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/tables/ca0c23976423ec0fe8c3511461ebf0c8e036c1ebf2a05872a14ace3b67772106.jpg)

![cc20a946731ac3dfc8b254c0ab48c8135d57dd879b82f558926e1b0f141e4d1f.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/tables/cc20a946731ac3dfc8b254c0ab48c8135d57dd879b82f558926e1b0f141e4d1f.jpg)

![d95fe30a4c76b2b46c0242dd5c0da84337e7aeb625e064f9e78bb1d783587b00.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/tables/d95fe30a4c76b2b46c0242dd5c0da84337e7aeb625e064f9e78bb1d783587b00.jpg)

![db361a1f952d2c700a1ff5e15cffff3589cf01d8d5fb337b3fe4d124f7b55253.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/tables/db361a1f952d2c700a1ff5e15cffff3589cf01d8d5fb337b3fe4d124f7b55253.jpg)

![df8925b6155d36412b1dd504caf5da78180678320cce3d936c27637c83916b14.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/tables/df8925b6155d36412b1dd504caf5da78180678320cce3d936c27637c83916b14.jpg)

![ebc884a68e0ab3bb6a719b10c07b02e789c82846662fa304fbfce01b297be5be.jpg](../iclr_results/3219_Inverse Constitutional AI_ Compressing Preferences into Principles/tables/ebc884a68e0ab3bb6a719b10c07b02e789c82846662fa304fbfce01b297be5be.jpg)

## An Efficient Framework for Crediting Data Contributors of Diffusion Models


### Images

![0c24fa6a1ee5a7ee9d5e75cedbb66bffb66fc76a0a79aa558e2acbad7fbe5ad8.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/images/0c24fa6a1ee5a7ee9d5e75cedbb66bffb66fc76a0a79aa558e2acbad7fbe5ad8.jpg)

![18c11f5fec6a7b16582023edbcfffa4c23875c2c3a3d72b36a1cb3fff626eafd.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/images/18c11f5fec6a7b16582023edbcfffa4c23875c2c3a3d72b36a1cb3fff626eafd.jpg)

![2c5ef5d5f7a5b730ea9a5f293ec09f819d143aaf3355356b2b5392515d6bc9ce.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/images/2c5ef5d5f7a5b730ea9a5f293ec09f819d143aaf3355356b2b5392515d6bc9ce.jpg)

![305aa2c0c4485714e060b82d3da57472a8756829b190cfc8ac46e0d902efde82.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/images/305aa2c0c4485714e060b82d3da57472a8756829b190cfc8ac46e0d902efde82.jpg)

![3a1e0fea951119b7b003a09f68d00729921034711bf9d9efccac8d92fd6deb68.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/images/3a1e0fea951119b7b003a09f68d00729921034711bf9d9efccac8d92fd6deb68.jpg)

![6893e48dba0310164bd7ad4cc8a22d7518332337b538dddbcfd9f443ae6f4efc.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/images/6893e48dba0310164bd7ad4cc8a22d7518332337b538dddbcfd9f443ae6f4efc.jpg)

![76ba63a82268ed4aaa5bc5b16abcaf74c3d41f52031888af861f4740b3698323.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/images/76ba63a82268ed4aaa5bc5b16abcaf74c3d41f52031888af861f4740b3698323.jpg)

![77b9c059ed1dee5a0cdefd42c78ea14a42a5b21fdd9d671df8c5d65f7a083b82.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/images/77b9c059ed1dee5a0cdefd42c78ea14a42a5b21fdd9d671df8c5d65f7a083b82.jpg)

![8937a16055f344c3a33fbc185bae6d7a012ed067cd673117b522363d4bda112d.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/images/8937a16055f344c3a33fbc185bae6d7a012ed067cd673117b522363d4bda112d.jpg)

![8f029cd12975f53f4f00e8a48222018354cc509681f9545ba85b421f9ddba99f.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/images/8f029cd12975f53f4f00e8a48222018354cc509681f9545ba85b421f9ddba99f.jpg)

![96eb5ef0dfcad672d0cec608db6ff8aed1c56b2c3d1f6811c804df34161b68e2.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/images/96eb5ef0dfcad672d0cec608db6ff8aed1c56b2c3d1f6811c804df34161b68e2.jpg)

![a512fe38d6030eba54bbd8e10789931ed1b79c355805767223ee56de2a92aa08.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/images/a512fe38d6030eba54bbd8e10789931ed1b79c355805767223ee56de2a92aa08.jpg)

![c196b4e2f3ca19d7d92dba696fdf517817e6fa326f84e6e8d969fb60143c8511.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/images/c196b4e2f3ca19d7d92dba696fdf517817e6fa326f84e6e8d969fb60143c8511.jpg)

![c97368dc76c8ade334dfdfeaf62ba774ee401e7da0a7e8196273688c7bf24f45.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/images/c97368dc76c8ade334dfdfeaf62ba774ee401e7da0a7e8196273688c7bf24f45.jpg)

![cf7f76768e9943cd6621c261f8b1a0e16b18ac94d6c59b5679452099fe840154.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/images/cf7f76768e9943cd6621c261f8b1a0e16b18ac94d6c59b5679452099fe840154.jpg)

![edde023be25d4d830104268dfcd7d72158d597c31ee688309da8ca7f7cd54719.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/images/edde023be25d4d830104268dfcd7d72158d597c31ee688309da8ca7f7cd54719.jpg)

![f0477c2614bed00707788ede2fde105b78202ad40fadf912259f198ca30b81d3.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/images/f0477c2614bed00707788ede2fde105b78202ad40fadf912259f198ca30b81d3.jpg)

![fb033914e4728b6e75e897d059be1f46ed9890e07a7b48d90ffcc9b2ccdc4128.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/images/fb033914e4728b6e75e897d059be1f46ed9890e07a7b48d90ffcc9b2ccdc4128.jpg)

![fcbce4783affb0dfd661b9341057cbed8e343c763e9328a55a6919bd6680640a.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/images/fcbce4783affb0dfd661b9341057cbed8e343c763e9328a55a6919bd6680640a.jpg)

### Tables

![217600478d3311062037b8693d4202c2cf548016e45a91a8cb66e41649a78f74.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/tables/217600478d3311062037b8693d4202c2cf548016e45a91a8cb66e41649a78f74.jpg)

![4bc6c85209ad17379c6b7d1e5dc638ef1858e48aa44ef25cf0454dad434e9a7e.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/tables/4bc6c85209ad17379c6b7d1e5dc638ef1858e48aa44ef25cf0454dad434e9a7e.jpg)

![58093c98af3a9a1f12739715356f7a458cbb2c929b0b398fd5c05b569bfe89e7.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/tables/58093c98af3a9a1f12739715356f7a458cbb2c929b0b398fd5c05b569bfe89e7.jpg)

![63bf97d21dfb9d361a48ce0a3ad9db303cc5b4f1abb084ba250617bb9b50e328.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/tables/63bf97d21dfb9d361a48ce0a3ad9db303cc5b4f1abb084ba250617bb9b50e328.jpg)

![7f925c8c488d8ade3e6eccf7775f0def843b726c2e7a672c3ab6a53e3b69e31f.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/tables/7f925c8c488d8ade3e6eccf7775f0def843b726c2e7a672c3ab6a53e3b69e31f.jpg)

![94cadfdfd6504bd911d0d5d7815f2f4d445dba08ce8e7d298b04665c0ddbffd4.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/tables/94cadfdfd6504bd911d0d5d7815f2f4d445dba08ce8e7d298b04665c0ddbffd4.jpg)

![c3c5ef16e3c5a57174bc145a7d0be65cc6ba45abfebff16d1f3eb8db99d2e8e1.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/tables/c3c5ef16e3c5a57174bc145a7d0be65cc6ba45abfebff16d1f3eb8db99d2e8e1.jpg)

![ca6368e8a0cff67e7d29cc10fe9e793b28ec027c91133343d309b6764b8f87f5.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/tables/ca6368e8a0cff67e7d29cc10fe9e793b28ec027c91133343d309b6764b8f87f5.jpg)

![d57338a09965308ad1a21d9ade0d4f438ad14f3d07cb0f8187baa82adc63c4e7.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/tables/d57338a09965308ad1a21d9ade0d4f438ad14f3d07cb0f8187baa82adc63c4e7.jpg)

![dcfceb6bcf104a7e939e4f8286f281ef782a6ea9cf5d012dac77764026c80135.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/tables/dcfceb6bcf104a7e939e4f8286f281ef782a6ea9cf5d012dac77764026c80135.jpg)

![e27ed5af2c90e747171b2e5f4eee6938001750d7487b567d406b2560408c52d4.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/tables/e27ed5af2c90e747171b2e5f4eee6938001750d7487b567d406b2560408c52d4.jpg)

![e79ffd990d884129689a12b94c2d00e236c04f861d0c7cb60ef050ca67f27249.jpg](../iclr_results/3220_An Efficient Framework for Crediting Data Contributors of Diffusion Models/tables/e79ffd990d884129689a12b94c2d00e236c04f861d0c7cb60ef050ca67f27249.jpg)

## Optimization by Parallel Quasi-Quantum Annealing with Gradient-Based Sampling


### Images

![6c9d18f702f672afe74bba12949b2973fc66ca50a07d3f8d34ba28e312433ef7.jpg](../iclr_results/3221_Optimization by Parallel Quasi-Quantum Annealing with Gradient-Based Sampling/images/6c9d18f702f672afe74bba12949b2973fc66ca50a07d3f8d34ba28e312433ef7.jpg)

![87fb6448ecbeb6f0537c37a19e9921b499f65ce08dfcea41ec92be2ad7aaa8c1.jpg](../iclr_results/3221_Optimization by Parallel Quasi-Quantum Annealing with Gradient-Based Sampling/images/87fb6448ecbeb6f0537c37a19e9921b499f65ce08dfcea41ec92be2ad7aaa8c1.jpg)

![b1c85d3ba36f57d492763f39fe3d8fa792aadbcd832d4df09ecfe86719a3f2fc.jpg](../iclr_results/3221_Optimization by Parallel Quasi-Quantum Annealing with Gradient-Based Sampling/images/b1c85d3ba36f57d492763f39fe3d8fa792aadbcd832d4df09ecfe86719a3f2fc.jpg)

### Tables

![1fdfa33d053850cd905d88d952be36fdc06e15c6c47f5cd33008346fde553c51.jpg](../iclr_results/3221_Optimization by Parallel Quasi-Quantum Annealing with Gradient-Based Sampling/tables/1fdfa33d053850cd905d88d952be36fdc06e15c6c47f5cd33008346fde553c51.jpg)

![24de86b200649213ffb6e8f69aed6c9f3825e6d44006839ed2e02a1299adb1a5.jpg](../iclr_results/3221_Optimization by Parallel Quasi-Quantum Annealing with Gradient-Based Sampling/tables/24de86b200649213ffb6e8f69aed6c9f3825e6d44006839ed2e02a1299adb1a5.jpg)

![3fd4e25811d85bdf4b0b5bf30498b9ce85b00582da0f1acf4d1e70725e762384.jpg](../iclr_results/3221_Optimization by Parallel Quasi-Quantum Annealing with Gradient-Based Sampling/tables/3fd4e25811d85bdf4b0b5bf30498b9ce85b00582da0f1acf4d1e70725e762384.jpg)

![7aa4af49213e37a6a9b60e5fc164ece428611bcea85521f93791e995fbed2d64.jpg](../iclr_results/3221_Optimization by Parallel Quasi-Quantum Annealing with Gradient-Based Sampling/tables/7aa4af49213e37a6a9b60e5fc164ece428611bcea85521f93791e995fbed2d64.jpg)

![ba0cdaefb22a2f592b56d19f4e1dd17d613b41aa14443fbe7036f1212d9ee9e8.jpg](../iclr_results/3221_Optimization by Parallel Quasi-Quantum Annealing with Gradient-Based Sampling/tables/ba0cdaefb22a2f592b56d19f4e1dd17d613b41aa14443fbe7036f1212d9ee9e8.jpg)

![c1500a28ddfb6cceb044366612a81900414af56adbff2df0ca3b9d9892cb49aa.jpg](../iclr_results/3221_Optimization by Parallel Quasi-Quantum Annealing with Gradient-Based Sampling/tables/c1500a28ddfb6cceb044366612a81900414af56adbff2df0ca3b9d9892cb49aa.jpg)

![c2a54259680ba7b842971047b1bb370a108a9e270c1c183a204e6964c04dd350.jpg](../iclr_results/3221_Optimization by Parallel Quasi-Quantum Annealing with Gradient-Based Sampling/tables/c2a54259680ba7b842971047b1bb370a108a9e270c1c183a204e6964c04dd350.jpg)

![c890729940188a09dbea55709bafec7c8d1a8c87479249c98478b2bb3ab3b36e.jpg](../iclr_results/3221_Optimization by Parallel Quasi-Quantum Annealing with Gradient-Based Sampling/tables/c890729940188a09dbea55709bafec7c8d1a8c87479249c98478b2bb3ab3b36e.jpg)

![d63ade4406664a11c67b4c3c382ac1a989589095793a384b1490718924ed04c2.jpg](../iclr_results/3221_Optimization by Parallel Quasi-Quantum Annealing with Gradient-Based Sampling/tables/d63ade4406664a11c67b4c3c382ac1a989589095793a384b1490718924ed04c2.jpg)

## VideoPhy: Evaluating Physical Commonsense for Video Generation


### Images

![07e25697c3d92d9e1067b4420d2046997ce73c19493c969d19055508358d1990.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/07e25697c3d92d9e1067b4420d2046997ce73c19493c969d19055508358d1990.jpg)

![0a4eefbf7fb0275264a1ecf3c4655f39a3be237921482d1989cf6f59d8b02431.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/0a4eefbf7fb0275264a1ecf3c4655f39a3be237921482d1989cf6f59d8b02431.jpg)

![0cd8cec8c3d045485b2afca7ad11508f4735a5ad3fee032f55881c6ca6d95c07.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/0cd8cec8c3d045485b2afca7ad11508f4735a5ad3fee032f55881c6ca6d95c07.jpg)

![12ee042e1d6886449553214e02171fba5066067eb3390e6a4e6e697d83dfafad.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/12ee042e1d6886449553214e02171fba5066067eb3390e6a4e6e697d83dfafad.jpg)

![14c69d6dc530dd9e839fe28c2c78ee3d7b6d52fd59a7597af0727223a076f1be.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/14c69d6dc530dd9e839fe28c2c78ee3d7b6d52fd59a7597af0727223a076f1be.jpg)

![1d491318e4fa921257d27d3909cccae89e649081d6cf5630f7ff9cf5936cc61d.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/1d491318e4fa921257d27d3909cccae89e649081d6cf5630f7ff9cf5936cc61d.jpg)

![1d7e7825f40ec04b30feff2b48902467cef81394df68d83e2f3b14dcfda54f22.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/1d7e7825f40ec04b30feff2b48902467cef81394df68d83e2f3b14dcfda54f22.jpg)

![1e18f8137404ec9e897e7f5b9a510a8312928c161451d729440e434ea1aaea36.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/1e18f8137404ec9e897e7f5b9a510a8312928c161451d729440e434ea1aaea36.jpg)

![1f84e23a64f75d7a4d9923a30192fed830010dce4ad528c47c487b738b10c2a0.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/1f84e23a64f75d7a4d9923a30192fed830010dce4ad528c47c487b738b10c2a0.jpg)

![23c36d298903543c97430ec2d15c5bc96da15160e92ba6663cb70b88b6489235.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/23c36d298903543c97430ec2d15c5bc96da15160e92ba6663cb70b88b6489235.jpg)

![2c5ff03256ede4ecea4d91cd7f963a236cc15afbf0a17bc3f9892b464fc9edc1.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/2c5ff03256ede4ecea4d91cd7f963a236cc15afbf0a17bc3f9892b464fc9edc1.jpg)

![37de1445807e0625ff8ad7bc8bdce0cddb5b9ff5e62b151bc3a0cc0c612c26bb.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/37de1445807e0625ff8ad7bc8bdce0cddb5b9ff5e62b151bc3a0cc0c612c26bb.jpg)

![3a7876e5c59cf0cd8472a8fcfd01befef577340e4ea9ac13854f89f37710a5a5.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/3a7876e5c59cf0cd8472a8fcfd01befef577340e4ea9ac13854f89f37710a5a5.jpg)

![4a8809d524592951c68fe32180231c800608953d4a63cb63819e2a3012abb07d.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/4a8809d524592951c68fe32180231c800608953d4a63cb63819e2a3012abb07d.jpg)

![4c95d1c87f2b9484acb875e1435d0b581d94204eb7de601bd1fa14be65059d12.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/4c95d1c87f2b9484acb875e1435d0b581d94204eb7de601bd1fa14be65059d12.jpg)

![50e734a9489229da934be3f7d112770078bfec8bc886fde79d7e1a64d10a8b63.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/50e734a9489229da934be3f7d112770078bfec8bc886fde79d7e1a64d10a8b63.jpg)

![5222747e8a64c8cbe65a4feab77f41cfbf95f48c2e363e74b7629089ba0b35b5.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/5222747e8a64c8cbe65a4feab77f41cfbf95f48c2e363e74b7629089ba0b35b5.jpg)

![52d69a81a602bf3a236c1d8738870fc3a77c32195ecac286a3c3f2eb2bf22631.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/52d69a81a602bf3a236c1d8738870fc3a77c32195ecac286a3c3f2eb2bf22631.jpg)

![5328312dd492938cbc70bc9750d232167ba478eb2e3cdf5d69e8eb40f45324d2.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/5328312dd492938cbc70bc9750d232167ba478eb2e3cdf5d69e8eb40f45324d2.jpg)

![56f2691dda7d961bae66677ad83124fac4ddcf3a81877012ed79319fbc8a4886.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/56f2691dda7d961bae66677ad83124fac4ddcf3a81877012ed79319fbc8a4886.jpg)

![59e6b89695839047d08705131cf42debaf2f942d03ebac5763375cfb792eb9bf.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/59e6b89695839047d08705131cf42debaf2f942d03ebac5763375cfb792eb9bf.jpg)

![5bd61e4f86c859935c4e51241de2aa5970d11406747bbfc6bb556366cf502db3.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/5bd61e4f86c859935c4e51241de2aa5970d11406747bbfc6bb556366cf502db3.jpg)

![638984e5a8879ce2b01fac55ad9502589537eabb0bc68b95970a942b79a91351.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/638984e5a8879ce2b01fac55ad9502589537eabb0bc68b95970a942b79a91351.jpg)

![66a09ec0c303808b1cab5b7d216c17429cba686d7d5f922b95ed7c8b64c88d78.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/66a09ec0c303808b1cab5b7d216c17429cba686d7d5f922b95ed7c8b64c88d78.jpg)

![6aead1c73be844421ef88af6ed48f4d8432f0924f4f8c0d4af52124c28c96d26.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/6aead1c73be844421ef88af6ed48f4d8432f0924f4f8c0d4af52124c28c96d26.jpg)

![6e011591b8ec3a8549ad1c10788c78996ccb8086177ca5f73d75512b53df6a46.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/6e011591b8ec3a8549ad1c10788c78996ccb8086177ca5f73d75512b53df6a46.jpg)

![6e321789911844d5954e4a9d8f82befed6c10b51c0b0a41333a5503edf3bb1ec.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/6e321789911844d5954e4a9d8f82befed6c10b51c0b0a41333a5503edf3bb1ec.jpg)

![6fcd8f2a02e7814aa290bd847f4ca4d0f884ca2ebb4699f17d3aaa2d2ed39997.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/6fcd8f2a02e7814aa290bd847f4ca4d0f884ca2ebb4699f17d3aaa2d2ed39997.jpg)

![7a0c7007cfe9ee04dfb53b8e93add09cc473ab657162432128c3f0aa0b647e5f.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/7a0c7007cfe9ee04dfb53b8e93add09cc473ab657162432128c3f0aa0b647e5f.jpg)

![7c0af9fefe4e328a34491722e5be2d415d2ba4cd1cba551bdf0ac861a56b2d05.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/7c0af9fefe4e328a34491722e5be2d415d2ba4cd1cba551bdf0ac861a56b2d05.jpg)

![7c8c4239347afd7698e46806bfe103770a8583b24b5daabbcaaf4968db6cf716.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/7c8c4239347afd7698e46806bfe103770a8583b24b5daabbcaaf4968db6cf716.jpg)

![847f92f8518d1fc4e3c909c7f36a7d6e969f01dff045045c6e4f250cd96ee9a6.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/847f92f8518d1fc4e3c909c7f36a7d6e969f01dff045045c6e4f250cd96ee9a6.jpg)

![87a4a11e25ecf889c205147a296138bdc27c588caf4047d0e322ff983ecf4bd7.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/87a4a11e25ecf889c205147a296138bdc27c588caf4047d0e322ff983ecf4bd7.jpg)

![8ad1085d2680fa553a5ce8ce2d3a5a640d51d8556452faa45b509beb7ed9dd69.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/8ad1085d2680fa553a5ce8ce2d3a5a640d51d8556452faa45b509beb7ed9dd69.jpg)

![8e9bbd8042dfb6c6c9f372154e6637d09a602baafa29d7cdc587c1eb0faebb6c.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/8e9bbd8042dfb6c6c9f372154e6637d09a602baafa29d7cdc587c1eb0faebb6c.jpg)

![92bf063fbad2dbe8c6359df6dba84c9943075c0876db618c542db5b5ca4cdae2.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/92bf063fbad2dbe8c6359df6dba84c9943075c0876db618c542db5b5ca4cdae2.jpg)

![97936da30d04acb98e0c2688c042e1ce88f75ee8c116d3f23e13977437eef0cb.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/97936da30d04acb98e0c2688c042e1ce88f75ee8c116d3f23e13977437eef0cb.jpg)

![9af8f3d543cdb3500d99d3ed94f5ee222e49e8d7e33b533d6ac66bd4790b6c46.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/9af8f3d543cdb3500d99d3ed94f5ee222e49e8d7e33b533d6ac66bd4790b6c46.jpg)

![9af9a1efd69ed3b9eb0ab6be93c8f8e6fe7cdc7e7beff2fdb5a7301df1da0239.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/9af9a1efd69ed3b9eb0ab6be93c8f8e6fe7cdc7e7beff2fdb5a7301df1da0239.jpg)

![9dcef44dea56f614d4442b955e369c9e304f4290db341c7fa733b686bce994bd.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/9dcef44dea56f614d4442b955e369c9e304f4290db341c7fa733b686bce994bd.jpg)

![9deb5b4d9e57e450538fdb463498842da5dd8230d42c67ec626fe8a927a7fbf3.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/9deb5b4d9e57e450538fdb463498842da5dd8230d42c67ec626fe8a927a7fbf3.jpg)

![9fac9dd5ece621d62695a63cbda251317fc0d3dd1154c2170cf1c1d01d5928fe.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/9fac9dd5ece621d62695a63cbda251317fc0d3dd1154c2170cf1c1d01d5928fe.jpg)

![a259146bdece9b9d23bc6f636a05fa3b615e59d4a8505f5b1305cd9955693b8c.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/a259146bdece9b9d23bc6f636a05fa3b615e59d4a8505f5b1305cd9955693b8c.jpg)

![a43f141ea67a0e65a40e5080675dff84092c776c7d6f493ad1c30f1c585c0063.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/a43f141ea67a0e65a40e5080675dff84092c776c7d6f493ad1c30f1c585c0063.jpg)

![a4c9479cf7a6811f3d8406e3e90e23cb554f17d8fa13b30f1e01ede26977ac66.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/a4c9479cf7a6811f3d8406e3e90e23cb554f17d8fa13b30f1e01ede26977ac66.jpg)

![a7818c5f5d0408d6c94693aef22073259b8fd4a425bbb1b0a9244442b3e038c7.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/a7818c5f5d0408d6c94693aef22073259b8fd4a425bbb1b0a9244442b3e038c7.jpg)

![a98c0ba8f3bdebadd9f88605b48f915afbd49669969164991849b56015c381af.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/a98c0ba8f3bdebadd9f88605b48f915afbd49669969164991849b56015c381af.jpg)

![abbaa4a08b06e94ae36048bc4ac1ef7ca9b3c797135b4aa9ade28c356273453f.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/abbaa4a08b06e94ae36048bc4ac1ef7ca9b3c797135b4aa9ade28c356273453f.jpg)

![acbef8d37c77b4761030677f41ddaccf3affe2819081775ac67efbc5a5b6290f.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/acbef8d37c77b4761030677f41ddaccf3affe2819081775ac67efbc5a5b6290f.jpg)

![af5eb8d8f9ae57949d549cfc7d4606ec368954564106d8c4f5f54b26631e1292.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/af5eb8d8f9ae57949d549cfc7d4606ec368954564106d8c4f5f54b26631e1292.jpg)

![b2c5ec00293e045cdbfb0a35615d4f995ea67e3eb841af67c9fb3abc8e9f4e55.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/b2c5ec00293e045cdbfb0a35615d4f995ea67e3eb841af67c9fb3abc8e9f4e55.jpg)

![b7f820cf48d7abfd9f401c9bd88502cdfa2e0dcb2a8e2aa409ed220b1fc53637.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/b7f820cf48d7abfd9f401c9bd88502cdfa2e0dcb2a8e2aa409ed220b1fc53637.jpg)

![bb3d34582fbf9159dccf524a8cd5df871791745dde6dfca481ce2d4c79b2a9df.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/bb3d34582fbf9159dccf524a8cd5df871791745dde6dfca481ce2d4c79b2a9df.jpg)

![bc3522d221aaa0584e0fab806aaf59d260a0449c9ba6601d3441c57c2c9b110b.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/bc3522d221aaa0584e0fab806aaf59d260a0449c9ba6601d3441c57c2c9b110b.jpg)

![bf6858c7c2053927c901b23b8ba45c8e35792e080074cd8a116dd1694898ac37.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/bf6858c7c2053927c901b23b8ba45c8e35792e080074cd8a116dd1694898ac37.jpg)

![c471c9129ce507e830cab7233cb2c8d649f39b74fc4b8040bc856aa24cc5ac4d.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/c471c9129ce507e830cab7233cb2c8d649f39b74fc4b8040bc856aa24cc5ac4d.jpg)

![c6b09b668d9419107dd5fa62a97e4d19ed50b87d7564d8e8d3d97bad0aab23d7.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/c6b09b668d9419107dd5fa62a97e4d19ed50b87d7564d8e8d3d97bad0aab23d7.jpg)

![d467f8774e39a2bf31bb9d13aa157f3e07ca82e75cbb2b81ca130f1df8801862.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/d467f8774e39a2bf31bb9d13aa157f3e07ca82e75cbb2b81ca130f1df8801862.jpg)

![d5d50efcad634c0fc0a9b5d3664a4b519f6899bc9ff03f841b4028c6043b2c7b.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/d5d50efcad634c0fc0a9b5d3664a4b519f6899bc9ff03f841b4028c6043b2c7b.jpg)

![da9c385fe5c784611c9bb9812e527560f776c8aeb7ea3bad8e4ea8622fd28f2a.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/da9c385fe5c784611c9bb9812e527560f776c8aeb7ea3bad8e4ea8622fd28f2a.jpg)

![daf581a736275e9a0a17198d37be190098bdcad8de26785eba5b2f40b4ba0cb4.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/daf581a736275e9a0a17198d37be190098bdcad8de26785eba5b2f40b4ba0cb4.jpg)

![dbdf2b1189b6393865d02d405bc321c63a7ff9bfb2ab9b157f83c45831e4c4fa.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/dbdf2b1189b6393865d02d405bc321c63a7ff9bfb2ab9b157f83c45831e4c4fa.jpg)

![dde94c7cdcad3eee93cc841d74d72280c832c40b64738be03b7b0dcd3ccdc8a3.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/dde94c7cdcad3eee93cc841d74d72280c832c40b64738be03b7b0dcd3ccdc8a3.jpg)

![de51dcace5f976af86cccabb4a6915fb668ebbaaa9574ac3f59f008c72fea0dc.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/de51dcace5f976af86cccabb4a6915fb668ebbaaa9574ac3f59f008c72fea0dc.jpg)

![dfe2ffb005d42dcd247bf16e96ffd119aa639baabbb6f6d7833d67975d1b19c0.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/dfe2ffb005d42dcd247bf16e96ffd119aa639baabbb6f6d7833d67975d1b19c0.jpg)

![e4f25724910c452e1b05e026fb04905d660e35ac812d6a12bdd35243e614ff85.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/e4f25724910c452e1b05e026fb04905d660e35ac812d6a12bdd35243e614ff85.jpg)

![e6a26548dcfd020c22929844b9d8895dc7cf161067791fa56a2cbcf3d21790bc.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/e6a26548dcfd020c22929844b9d8895dc7cf161067791fa56a2cbcf3d21790bc.jpg)

![e78d26e4da70bdaa3ca589b4b27965a8c84077b1d94fecb4f0c25b76ed719fbe.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/e78d26e4da70bdaa3ca589b4b27965a8c84077b1d94fecb4f0c25b76ed719fbe.jpg)

![eddede139cd1b21d3a58877172f756db2ffd4f0d54b1db0f1bd4cc9974f289f0.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/eddede139cd1b21d3a58877172f756db2ffd4f0d54b1db0f1bd4cc9974f289f0.jpg)

![f1d7bb7d68729c2c1339cc4154849927f14bac9f0008ad58831984029af8f92f.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/f1d7bb7d68729c2c1339cc4154849927f14bac9f0008ad58831984029af8f92f.jpg)

![f3799809dc831e644bd4934447e300f554eaecbedd117b37d732ddbc19000206.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/f3799809dc831e644bd4934447e300f554eaecbedd117b37d732ddbc19000206.jpg)

![f4a0cc24b201dec6ca7b877d5f91fd724b244a620b7e2e2154d2818981b25f1f.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/f4a0cc24b201dec6ca7b877d5f91fd724b244a620b7e2e2154d2818981b25f1f.jpg)

![f85a4db770564bfc0ea7f67148259805e9ef1e230e24cb11004f566f29e1c59a.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/f85a4db770564bfc0ea7f67148259805e9ef1e230e24cb11004f566f29e1c59a.jpg)

![fbf14b30effba27ba4d6459920983ae75ba3345892ad28001478f6e430d8beed.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/images/fbf14b30effba27ba4d6459920983ae75ba3345892ad28001478f6e430d8beed.jpg)

### Tables

![0fb0ebb25b610edee88fecada1c7557ced34f60a6052827c0e4824dc35299c4b.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/tables/0fb0ebb25b610edee88fecada1c7557ced34f60a6052827c0e4824dc35299c4b.jpg)

![1e7919d6b7bfe5e0f11a23a257cc323ef6af1d18ca36418fa0ad3e425b43a2a0.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/tables/1e7919d6b7bfe5e0f11a23a257cc323ef6af1d18ca36418fa0ad3e425b43a2a0.jpg)

![2f756d8687d9c02e841b54a536937d0c7611616438925470982f51e3e0a30691.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/tables/2f756d8687d9c02e841b54a536937d0c7611616438925470982f51e3e0a30691.jpg)

![4abe5d6c07960d78f0ccaade8636cb3944587eeeca66a4cd955a6b4778178bc3.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/tables/4abe5d6c07960d78f0ccaade8636cb3944587eeeca66a4cd955a6b4778178bc3.jpg)

![4f5f2f50d5a3b7a1f7c75752d988e69f7119e625d55359f3ce23f9b0737ff576.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/tables/4f5f2f50d5a3b7a1f7c75752d988e69f7119e625d55359f3ce23f9b0737ff576.jpg)

![6071528dfc79a4ce70b7b94b0edb8ade11c69c43f8dd988e3e1370143e6d4acc.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/tables/6071528dfc79a4ce70b7b94b0edb8ade11c69c43f8dd988e3e1370143e6d4acc.jpg)

![7c5c87479bf548823314ae0e7c1093de479f69fec5a17a7a2fbdb19bddc8ad20.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/tables/7c5c87479bf548823314ae0e7c1093de479f69fec5a17a7a2fbdb19bddc8ad20.jpg)

![82d147ce6509188c6d0ad05d0b7deca4e311f54cf1475b172bddfd845fa7398c.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/tables/82d147ce6509188c6d0ad05d0b7deca4e311f54cf1475b172bddfd845fa7398c.jpg)

![8716acaf5c7855adbfab8994f5f419c620ef976c978ea82ee214998b9efc4a78.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/tables/8716acaf5c7855adbfab8994f5f419c620ef976c978ea82ee214998b9efc4a78.jpg)

![bbd6980d745753dd5bb094b272833d4d810c1f73511463cc1dbf2cc33f57bae6.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/tables/bbd6980d745753dd5bb094b272833d4d810c1f73511463cc1dbf2cc33f57bae6.jpg)

![cbb0f09b01af48b9589b0809abb42121338f94d0f6eccba5e9bebaef2dd2b316.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/tables/cbb0f09b01af48b9589b0809abb42121338f94d0f6eccba5e9bebaef2dd2b316.jpg)

![d682b17a69c5b5bab165803731e0479203eb6298b4834d3922ba2e4a5221c90e.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/tables/d682b17a69c5b5bab165803731e0479203eb6298b4834d3922ba2e4a5221c90e.jpg)

![dd13158df518eeb8ea18538ab44523506cc32bb992490be1f32de996d8813db4.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/tables/dd13158df518eeb8ea18538ab44523506cc32bb992490be1f32de996d8813db4.jpg)

![ee4babcecae11bdbf94ebd86073a8c3831f84d56ad2285394c2ac611c1505616.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/tables/ee4babcecae11bdbf94ebd86073a8c3831f84d56ad2285394c2ac611c1505616.jpg)

![f5d50722269671fd18fd68d539579246297e4c294cb1e22a14aa6ac7b43422b4.jpg](../iclr_results/3222_VideoPhy_ Evaluating Physical Commonsense for Video Generation/tables/f5d50722269671fd18fd68d539579246297e4c294cb1e22a14aa6ac7b43422b4.jpg)

## Ada-K Routing: Boosting the Efficiency of MoE-based LLMs


### Images

![0097b81113e6179aa0dc9f999c24ec9d85f1dca3c39256cf094a9ce872699832.jpg](../iclr_results/3223_Ada-K Routing_ Boosting the Efficiency of MoE-based LLMs/images/0097b81113e6179aa0dc9f999c24ec9d85f1dca3c39256cf094a9ce872699832.jpg)

![1ffcd3f33368beda7ccde5d702dcc804624fbf0176ef73a631e09527c9975ad2.jpg](../iclr_results/3223_Ada-K Routing_ Boosting the Efficiency of MoE-based LLMs/images/1ffcd3f33368beda7ccde5d702dcc804624fbf0176ef73a631e09527c9975ad2.jpg)

![58d43153854b56e1d8e3cbe5c06c7beb957bb79380ac07a735850ae5e56d814c.jpg](../iclr_results/3223_Ada-K Routing_ Boosting the Efficiency of MoE-based LLMs/images/58d43153854b56e1d8e3cbe5c06c7beb957bb79380ac07a735850ae5e56d814c.jpg)

![78b5b44c30c8d37f7306fd809f7b210d21189b1cf809ec23f40ad227bb649c3f.jpg](../iclr_results/3223_Ada-K Routing_ Boosting the Efficiency of MoE-based LLMs/images/78b5b44c30c8d37f7306fd809f7b210d21189b1cf809ec23f40ad227bb649c3f.jpg)

![986ac871f65feaadaa9c6ed0ada0b48566f4356774d648af02636b1849d56b12.jpg](../iclr_results/3223_Ada-K Routing_ Boosting the Efficiency of MoE-based LLMs/images/986ac871f65feaadaa9c6ed0ada0b48566f4356774d648af02636b1849d56b12.jpg)

![be26aeb1a77f6d04e57c87bd10ebac0f6dda5813610dfedd5d5db36ae8a279e1.jpg](../iclr_results/3223_Ada-K Routing_ Boosting the Efficiency of MoE-based LLMs/images/be26aeb1a77f6d04e57c87bd10ebac0f6dda5813610dfedd5d5db36ae8a279e1.jpg)

![db5e87e1350992c94d7dd772a95f4d4c21b916c40172afd6c749730054335360.jpg](../iclr_results/3223_Ada-K Routing_ Boosting the Efficiency of MoE-based LLMs/images/db5e87e1350992c94d7dd772a95f4d4c21b916c40172afd6c749730054335360.jpg)

![f75f61ae08c50636b00ec31bfdcec2471022c76d6ddf56423b819f2c5b98aebc.jpg](../iclr_results/3223_Ada-K Routing_ Boosting the Efficiency of MoE-based LLMs/images/f75f61ae08c50636b00ec31bfdcec2471022c76d6ddf56423b819f2c5b98aebc.jpg)

### Tables

![07f1e25638a8f66c4b65e5b05b058a7a73e35083fa8b943f7de6a2a43319d452.jpg](../iclr_results/3223_Ada-K Routing_ Boosting the Efficiency of MoE-based LLMs/tables/07f1e25638a8f66c4b65e5b05b058a7a73e35083fa8b943f7de6a2a43319d452.jpg)

![1f11bb82ff963e8e884fdabb05aa50fed0a2354ad5ba0fa3dbf9112c2f53eaff.jpg](../iclr_results/3223_Ada-K Routing_ Boosting the Efficiency of MoE-based LLMs/tables/1f11bb82ff963e8e884fdabb05aa50fed0a2354ad5ba0fa3dbf9112c2f53eaff.jpg)

![2171fd028ba732b56322cc60b1c218a6600f2d6b494d7a359b78d7929b1497c5.jpg](../iclr_results/3223_Ada-K Routing_ Boosting the Efficiency of MoE-based LLMs/tables/2171fd028ba732b56322cc60b1c218a6600f2d6b494d7a359b78d7929b1497c5.jpg)

![4382235a3f46fb04ba67f02a9dc1939d49cc8234f549a621366568461b534bdd.jpg](../iclr_results/3223_Ada-K Routing_ Boosting the Efficiency of MoE-based LLMs/tables/4382235a3f46fb04ba67f02a9dc1939d49cc8234f549a621366568461b534bdd.jpg)

![45178f38a91d4c9568dbdfa677e01211f64e9f282ace59bb2e8b58a4cb993d39.jpg](../iclr_results/3223_Ada-K Routing_ Boosting the Efficiency of MoE-based LLMs/tables/45178f38a91d4c9568dbdfa677e01211f64e9f282ace59bb2e8b58a4cb993d39.jpg)

![5d3734d1afca28ec60a1e6a399aeba13e1066360bb0b7d3a8c5be0aab9f78bba.jpg](../iclr_results/3223_Ada-K Routing_ Boosting the Efficiency of MoE-based LLMs/tables/5d3734d1afca28ec60a1e6a399aeba13e1066360bb0b7d3a8c5be0aab9f78bba.jpg)

![7f681a8bd75d6a241052d142f2cfbd765b87103717570c84197e710fbb7af025.jpg](../iclr_results/3223_Ada-K Routing_ Boosting the Efficiency of MoE-based LLMs/tables/7f681a8bd75d6a241052d142f2cfbd765b87103717570c84197e710fbb7af025.jpg)

![cd8d4709216e1b1ad07955e1095ad0fce9962d7a82c61879a281b253c58bd799.jpg](../iclr_results/3223_Ada-K Routing_ Boosting the Efficiency of MoE-based LLMs/tables/cd8d4709216e1b1ad07955e1095ad0fce9962d7a82c61879a281b253c58bd799.jpg)

![d789a1dfe45c3d9541056a8e624bd42327ec6d339ce0c6480e6560f80c5c21fe.jpg](../iclr_results/3223_Ada-K Routing_ Boosting the Efficiency of MoE-based LLMs/tables/d789a1dfe45c3d9541056a8e624bd42327ec6d339ce0c6480e6560f80c5c21fe.jpg)

![e08df84193de833cd54665ce823c4b9d40597440626422740747bbdaf5b09ca6.jpg](../iclr_results/3223_Ada-K Routing_ Boosting the Efficiency of MoE-based LLMs/tables/e08df84193de833cd54665ce823c4b9d40597440626422740747bbdaf5b09ca6.jpg)

![e813f58040d6de284157de0c14aea7e6cf1de5fc33221b348976bbef3afe602b.jpg](../iclr_results/3223_Ada-K Routing_ Boosting the Efficiency of MoE-based LLMs/tables/e813f58040d6de284157de0c14aea7e6cf1de5fc33221b348976bbef3afe602b.jpg)

![f2df487fd908a5d5c8e2f3c712ac655c7e6a80479e18bc24334e4b3ee9b30ae8.jpg](../iclr_results/3223_Ada-K Routing_ Boosting the Efficiency of MoE-based LLMs/tables/f2df487fd908a5d5c8e2f3c712ac655c7e6a80479e18bc24334e4b3ee9b30ae8.jpg)

## Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures


### Images

![165e03a3e0ee7d3473075a389c74615cb52e4a2f7ba2a754ebe7fe3f5a34a81b.jpg](../iclr_results/3224_Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures/images/165e03a3e0ee7d3473075a389c74615cb52e4a2f7ba2a754ebe7fe3f5a34a81b.jpg)

![20998e4d45b747eeb728a3e8cc90e26e4cbf991eeb9ae27076c0985140c465ea.jpg](../iclr_results/3224_Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures/images/20998e4d45b747eeb728a3e8cc90e26e4cbf991eeb9ae27076c0985140c465ea.jpg)

![30bfcf001e60c66104bf5cb6ba3553ad880e7c02e25a4b226ee28e11221236eb.jpg](../iclr_results/3224_Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures/images/30bfcf001e60c66104bf5cb6ba3553ad880e7c02e25a4b226ee28e11221236eb.jpg)

![7700b89cc4ce1b52dbac27b2988b3326352403b7c367b5e4a109981bd3e26ef9.jpg](../iclr_results/3224_Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures/images/7700b89cc4ce1b52dbac27b2988b3326352403b7c367b5e4a109981bd3e26ef9.jpg)

![89802f77f8166dbcf4e7d5579795b638c691f191d916c2812c686a1021936139.jpg](../iclr_results/3224_Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures/images/89802f77f8166dbcf4e7d5579795b638c691f191d916c2812c686a1021936139.jpg)

### Tables

![019a1b6e2890b97da8198c6bf9a30200842493bb851a723a7849f364c0068e04.jpg](../iclr_results/3224_Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures/tables/019a1b6e2890b97da8198c6bf9a30200842493bb851a723a7849f364c0068e04.jpg)

![1a454ca99b2254a17af996743e7141e7c86d85907a86defa91cabac0b0b1849c.jpg](../iclr_results/3224_Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures/tables/1a454ca99b2254a17af996743e7141e7c86d85907a86defa91cabac0b0b1849c.jpg)

![23cf03702bbfef17c913f990e4f5d9f9480f56ef1bcb5224d247dda420c2c853.jpg](../iclr_results/3224_Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures/tables/23cf03702bbfef17c913f990e4f5d9f9480f56ef1bcb5224d247dda420c2c853.jpg)

![3914a70b2169435df319c8c2e695317b7228da82043be75c4e6b38be86a0a461.jpg](../iclr_results/3224_Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures/tables/3914a70b2169435df319c8c2e695317b7228da82043be75c4e6b38be86a0a461.jpg)

![53a4d360df543c1c13fe8076d3c163f637a422b467db42ac16b4f310d80a4ae4.jpg](../iclr_results/3224_Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures/tables/53a4d360df543c1c13fe8076d3c163f637a422b467db42ac16b4f310d80a4ae4.jpg)

![575cc1d5c318da84a0ac8c61c95394a74045ed2f9f4766217198929c1a2be5cf.jpg](../iclr_results/3224_Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures/tables/575cc1d5c318da84a0ac8c61c95394a74045ed2f9f4766217198929c1a2be5cf.jpg)

![6c63d8978cb2a71696cc33ef62a19bfa2caa079b3928c063ea5c161be575e240.jpg](../iclr_results/3224_Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures/tables/6c63d8978cb2a71696cc33ef62a19bfa2caa079b3928c063ea5c161be575e240.jpg)

![919449ad62bbd87c44337e7637e24a1692440c9cf766ec18cd9f64188f6244d5.jpg](../iclr_results/3224_Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures/tables/919449ad62bbd87c44337e7637e24a1692440c9cf766ec18cd9f64188f6244d5.jpg)

![a02387013b513f23a113df6eff1448f1d09bc71eed71e39cbf817d32077d61c7.jpg](../iclr_results/3224_Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures/tables/a02387013b513f23a113df6eff1448f1d09bc71eed71e39cbf817d32077d61c7.jpg)

![ab3f92fb87d2bbf581dd3bb805be12669ea029980d8a05afb0ab8a3ad30adb8a.jpg](../iclr_results/3224_Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures/tables/ab3f92fb87d2bbf581dd3bb805be12669ea029980d8a05afb0ab8a3ad30adb8a.jpg)

![c9b53a87bccde2165ba65b32783707b12b7794fa4724153b888a9a40c16a24d9.jpg](../iclr_results/3224_Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures/tables/c9b53a87bccde2165ba65b32783707b12b7794fa4724153b888a9a40c16a24d9.jpg)

![fe57eef3262ea0d262ccd542c79dd3b67596aed737a1769fcb6627e1e7d24189.jpg](../iclr_results/3224_Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures/tables/fe57eef3262ea0d262ccd542c79dd3b67596aed737a1769fcb6627e1e7d24189.jpg)

## DebGCD: Debiased Learning with Distribution Guidance for Generalized Category Discovery


### Images

![549d1a1ae1b23c7a66de533003b8a915d15aa0c412e69a7753249096eeb70f29.jpg](../iclr_results/3225_DebGCD_ Debiased Learning with Distribution Guidance for Generalized Category Discovery/images/549d1a1ae1b23c7a66de533003b8a915d15aa0c412e69a7753249096eeb70f29.jpg)

![ab60f6ccba44915ad55fb7d7543d4bee31f70b86a58004e02902869954fcf612.jpg](../iclr_results/3225_DebGCD_ Debiased Learning with Distribution Guidance for Generalized Category Discovery/images/ab60f6ccba44915ad55fb7d7543d4bee31f70b86a58004e02902869954fcf612.jpg)

### Tables

![1f7053521d5f0dbefd1125e696726ec7e326a4ec54859ef2cec71d680af95796.jpg](../iclr_results/3225_DebGCD_ Debiased Learning with Distribution Guidance for Generalized Category Discovery/tables/1f7053521d5f0dbefd1125e696726ec7e326a4ec54859ef2cec71d680af95796.jpg)

![28056758925cb05a492fde22bb8c5773e69ac3c605557aa66bf9bc2ec8c1861a.jpg](../iclr_results/3225_DebGCD_ Debiased Learning with Distribution Guidance for Generalized Category Discovery/tables/28056758925cb05a492fde22bb8c5773e69ac3c605557aa66bf9bc2ec8c1861a.jpg)

![7b6c6798e21e1ff88419a7f62189075f1ad0ce8382812f320374e2de16336823.jpg](../iclr_results/3225_DebGCD_ Debiased Learning with Distribution Guidance for Generalized Category Discovery/tables/7b6c6798e21e1ff88419a7f62189075f1ad0ce8382812f320374e2de16336823.jpg)

![85ef79d76dddd62ff433892bd0ed93631da0a1a6bea1960453b14f9538b5b840.jpg](../iclr_results/3225_DebGCD_ Debiased Learning with Distribution Guidance for Generalized Category Discovery/tables/85ef79d76dddd62ff433892bd0ed93631da0a1a6bea1960453b14f9538b5b840.jpg)

![b2265d111b60d266a467d862723844cc140a06367bee08da9d70c0d560916fca.jpg](../iclr_results/3225_DebGCD_ Debiased Learning with Distribution Guidance for Generalized Category Discovery/tables/b2265d111b60d266a467d862723844cc140a06367bee08da9d70c0d560916fca.jpg)

![b5ac5307337efdd3e8a051410e4bfd2e5b15ad3607538e16b27d0ab229458b36.jpg](../iclr_results/3225_DebGCD_ Debiased Learning with Distribution Guidance for Generalized Category Discovery/tables/b5ac5307337efdd3e8a051410e4bfd2e5b15ad3607538e16b27d0ab229458b36.jpg)

![f579d6bccb71fceacb030679fe394e181e885eebae773e0b208d80553208b325.jpg](../iclr_results/3225_DebGCD_ Debiased Learning with Distribution Guidance for Generalized Category Discovery/tables/f579d6bccb71fceacb030679fe394e181e885eebae773e0b208d80553208b325.jpg)

## LANTERN: Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding


### Images

![182d2e3033d65d26fb56d6551a4fcbf7fb024213f4efdc8e092318e3078db2b8.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/images/182d2e3033d65d26fb56d6551a4fcbf7fb024213f4efdc8e092318e3078db2b8.jpg)

![189adf452e1fb82c77d0251f15ef3a57162ecf3e79fb8dfa3232cef19643ee1f.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/images/189adf452e1fb82c77d0251f15ef3a57162ecf3e79fb8dfa3232cef19643ee1f.jpg)

![37bb4b67d9977b9fa8f3a922f93eaa958087ff8dea0bbcca7bec36236f874064.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/images/37bb4b67d9977b9fa8f3a922f93eaa958087ff8dea0bbcca7bec36236f874064.jpg)

![7a35f0307da1fd68b351bbe46de39dace76c157f66c33c72cfe57ca806031605.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/images/7a35f0307da1fd68b351bbe46de39dace76c157f66c33c72cfe57ca806031605.jpg)

![7d6bb84e448b1d4cfeeaac519eaa78013a6711df66b2d53b75d072ca501fccf7.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/images/7d6bb84e448b1d4cfeeaac519eaa78013a6711df66b2d53b75d072ca501fccf7.jpg)

![7d79320fbc12d5b3ddb0148f2f1cf97c2a8598a4f3822f0ad571c3354f0c3547.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/images/7d79320fbc12d5b3ddb0148f2f1cf97c2a8598a4f3822f0ad571c3354f0c3547.jpg)

![86a3816288e04553ae9af4d69eafe459404e09a140c5ba4becc16016221575b6.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/images/86a3816288e04553ae9af4d69eafe459404e09a140c5ba4becc16016221575b6.jpg)

![b0884b8ff37888fa970d817c85a52293ccc29cb480107c295bb139139147229d.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/images/b0884b8ff37888fa970d817c85a52293ccc29cb480107c295bb139139147229d.jpg)

![b5fe35a8a5c233b10fc4ae0e8a8e295c89a6e9e7f0b6af915bc29e3316c0f543.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/images/b5fe35a8a5c233b10fc4ae0e8a8e295c89a6e9e7f0b6af915bc29e3316c0f543.jpg)

![c90656283d40faa75d1ee83f3703dcfaa237b1b7eaac7df10a0ab6fdb100c214.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/images/c90656283d40faa75d1ee83f3703dcfaa237b1b7eaac7df10a0ab6fdb100c214.jpg)

![cc3b02adbacb6b74da5ecc6ec6727c3d2bb07b6dc2b7bc2850394104b09423b0.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/images/cc3b02adbacb6b74da5ecc6ec6727c3d2bb07b6dc2b7bc2850394104b09423b0.jpg)

![cf6a5915e615da4129eb89e430b487cfbc99be695bd0d53cdcbd0051096fb8b0.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/images/cf6a5915e615da4129eb89e430b487cfbc99be695bd0d53cdcbd0051096fb8b0.jpg)

![d55979028f9d8fd9f80b244448e013d39419da640cf763bdd1080bb10e3adf59.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/images/d55979028f9d8fd9f80b244448e013d39419da640cf763bdd1080bb10e3adf59.jpg)

![dc18170cd88492311b46452258acd46f6d2e200f185205cc768bd8b71acc7ee7.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/images/dc18170cd88492311b46452258acd46f6d2e200f185205cc768bd8b71acc7ee7.jpg)

![de975ee3a3fda767929a657ce7d955a302fde341a0353abda35b05f4ccd71093.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/images/de975ee3a3fda767929a657ce7d955a302fde341a0353abda35b05f4ccd71093.jpg)

![fc9c78b0f99995cd9dd7586b9e7f890d5833e4f5d63f6a5becc25807a6086576.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/images/fc9c78b0f99995cd9dd7586b9e7f890d5833e4f5d63f6a5becc25807a6086576.jpg)

### Tables

![3b6c3f41ac18b1e6f818ff36ad5ce03847381893e60c970ea8d7ea9773ac12f0.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/tables/3b6c3f41ac18b1e6f818ff36ad5ce03847381893e60c970ea8d7ea9773ac12f0.jpg)

![41eaa64e4a766e8459e998599b60d2a99c997f0d3f928bd0d416d0c4b967b00e.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/tables/41eaa64e4a766e8459e998599b60d2a99c997f0d3f928bd0d416d0c4b967b00e.jpg)

![6017b8e3ea108eff15498b56e9840af9cff9fab18f25e33b3da81598306c0856.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/tables/6017b8e3ea108eff15498b56e9840af9cff9fab18f25e33b3da81598306c0856.jpg)

![63cdc405af003c77e2d74c19ad5c226e405d94feb6d7f692071b0ade64607417.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/tables/63cdc405af003c77e2d74c19ad5c226e405d94feb6d7f692071b0ade64607417.jpg)

![6f84a0dba9255f4298ac84546bb54d0f592c2f957801bb48de490ba0e39529e9.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/tables/6f84a0dba9255f4298ac84546bb54d0f592c2f957801bb48de490ba0e39529e9.jpg)

![86a32e07c20bc45fd1cae2cac3345e031510fe9ef9298b905ddb4895f773c7f2.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/tables/86a32e07c20bc45fd1cae2cac3345e031510fe9ef9298b905ddb4895f773c7f2.jpg)

![92482ec793c290c063373148343c056e9576b4fdf4ca05dd21524310f832d327.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/tables/92482ec793c290c063373148343c056e9576b4fdf4ca05dd21524310f832d327.jpg)

![abe744df8f8c369686c56c7ae9f4e8e7f23a5404c6186ea080640a39f8c5c967.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/tables/abe744df8f8c369686c56c7ae9f4e8e7f23a5404c6186ea080640a39f8c5c967.jpg)

![b532cd18edb6c01d675d584e6429b995fb6e9a0480392c98867ac801dc16cbf7.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/tables/b532cd18edb6c01d675d584e6429b995fb6e9a0480392c98867ac801dc16cbf7.jpg)

![b9089c65f06ac56c4bd7ad3cd8e12af6ece54b8ab233d39e80b0027f47d5cd33.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/tables/b9089c65f06ac56c4bd7ad3cd8e12af6ece54b8ab233d39e80b0027f47d5cd33.jpg)

![e00197439ff209d566646bab32831ac4a0a135665fa7f65246ccf6616f174458.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/tables/e00197439ff209d566646bab32831ac4a0a135665fa7f65246ccf6616f174458.jpg)

![f31fe20dbdb091dff2d8e85c3bb19d09141373bcd94fe9924da5bb5c03ee50d2.jpg](../iclr_results/3226_LANTERN_ Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding/tables/f31fe20dbdb091dff2d8e85c3bb19d09141373bcd94fe9924da5bb5c03ee50d2.jpg)

## Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency


### Images

![1d90f8661c5bcd2dd9498373c157413ad3268a596ba34b65130a7843805cfdc9.jpg](../iclr_results/3227_Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency/images/1d90f8661c5bcd2dd9498373c157413ad3268a596ba34b65130a7843805cfdc9.jpg)

![2783c6c1a201328f4484679fd5ba1593ceb45d75c6798de7cb85183c075327b6.jpg](../iclr_results/3227_Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency/images/2783c6c1a201328f4484679fd5ba1593ceb45d75c6798de7cb85183c075327b6.jpg)

![4bff4bea5c5756d92e060616b5f7fd4a59be995a4b7e2896460f8f968a4ada55.jpg](../iclr_results/3227_Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency/images/4bff4bea5c5756d92e060616b5f7fd4a59be995a4b7e2896460f8f968a4ada55.jpg)

![9a45385428a2dc2d73c2b85d2e50ad04374ff115e6c5a9c1dd0ab56339b1b992.jpg](../iclr_results/3227_Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency/images/9a45385428a2dc2d73c2b85d2e50ad04374ff115e6c5a9c1dd0ab56339b1b992.jpg)

![d0061069c05d2468ddb7b33c50909468ac8be6385350d8b89e354e25d896fb97.jpg](../iclr_results/3227_Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency/images/d0061069c05d2468ddb7b33c50909468ac8be6385350d8b89e354e25d896fb97.jpg)

![d792430d0e4df059af43443eb27f2be067099eb3847ccb16aed6f7a72ceaa28b.jpg](../iclr_results/3227_Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency/images/d792430d0e4df059af43443eb27f2be067099eb3847ccb16aed6f7a72ceaa28b.jpg)

### Tables

![02344fa6984527ef1434570afc78e3bf47bb13184f0bfe208c626cb489c4e559.jpg](../iclr_results/3227_Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency/tables/02344fa6984527ef1434570afc78e3bf47bb13184f0bfe208c626cb489c4e559.jpg)

![0ba98eb30d2d579007dfdfe2b9bd0c23e1a28458a2e9d18fee6cbeb53e11b96b.jpg](../iclr_results/3227_Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency/tables/0ba98eb30d2d579007dfdfe2b9bd0c23e1a28458a2e9d18fee6cbeb53e11b96b.jpg)

![0dbe9b2426d5eaaaee83d299219bd05c016594fcff86d81e684461027f014a95.jpg](../iclr_results/3227_Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency/tables/0dbe9b2426d5eaaaee83d299219bd05c016594fcff86d81e684461027f014a95.jpg)

![0e6df71c3dbf9cd2126fbd0eb0a9577459ddd5aac04a17d3c73694468ccc8c6f.jpg](../iclr_results/3227_Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency/tables/0e6df71c3dbf9cd2126fbd0eb0a9577459ddd5aac04a17d3c73694468ccc8c6f.jpg)

![26d79973d104ba928bde1cbb7b9c08562ceda33c3e6c9214dafc0dbb98750a00.jpg](../iclr_results/3227_Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency/tables/26d79973d104ba928bde1cbb7b9c08562ceda33c3e6c9214dafc0dbb98750a00.jpg)

![279ff2826db5bffd5536c98a87ff8007afe94088112a11206c28f5546d9ffbd3.jpg](../iclr_results/3227_Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency/tables/279ff2826db5bffd5536c98a87ff8007afe94088112a11206c28f5546d9ffbd3.jpg)

![29c5d35d5eb9eb69599039f51d62466288d24e81f20bb10277f6068202022bfa.jpg](../iclr_results/3227_Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency/tables/29c5d35d5eb9eb69599039f51d62466288d24e81f20bb10277f6068202022bfa.jpg)

![30674d247b97d80bae6266496b93fb834ced38c571c79591dff28a9aab6cecb6.jpg](../iclr_results/3227_Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency/tables/30674d247b97d80bae6266496b93fb834ced38c571c79591dff28a9aab6cecb6.jpg)

![3d6d5a28a21422f6e47f445f7d7798135c8a8e21f106688e230e8c6c998f681b.jpg](../iclr_results/3227_Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency/tables/3d6d5a28a21422f6e47f445f7d7798135c8a8e21f106688e230e8c6c998f681b.jpg)

![5ed936acdfc8dde6e81147d9f1b071124d81989b1e476d7d2e4b439b78eb5854.jpg](../iclr_results/3227_Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency/tables/5ed936acdfc8dde6e81147d9f1b071124d81989b1e476d7d2e4b439b78eb5854.jpg)

![7fe92807b9555c474d6134ff32f38a5b48905562aa6bf3d716e3335b503d7a90.jpg](../iclr_results/3227_Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency/tables/7fe92807b9555c474d6134ff32f38a5b48905562aa6bf3d716e3335b503d7a90.jpg)

![90429202611a2ff538d4f0a764cee492061aaddc49fd9d65cfc8bd28edb69957.jpg](../iclr_results/3227_Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency/tables/90429202611a2ff538d4f0a764cee492061aaddc49fd9d65cfc8bd28edb69957.jpg)

![bf3b3e679d99d7db89178824c36bff809d57696f6eb26c2e820d8fec9a2eee0f.jpg](../iclr_results/3227_Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency/tables/bf3b3e679d99d7db89178824c36bff809d57696f6eb26c2e820d8fec9a2eee0f.jpg)

![cadfb9122435d87c9787bda64387b62871481e27e812cf7e7262d988d4ad99ac.jpg](../iclr_results/3227_Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency/tables/cadfb9122435d87c9787bda64387b62871481e27e812cf7e7262d988d4ad99ac.jpg)

![dd0b3e3276fff2c016401f110931681449a0fc72b5eb952cbf5c4358a28399dc.jpg](../iclr_results/3227_Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency/tables/dd0b3e3276fff2c016401f110931681449a0fc72b5eb952cbf5c4358a28399dc.jpg)

![df77dbd752c294ca95385c27be050f06eb1a9d698ee75f89b118055c3207432d.jpg](../iclr_results/3227_Semi-Supervised CLIP Adaptation by Enforcing Semantic and Trapezoidal Consistency/tables/df77dbd752c294ca95385c27be050f06eb1a9d698ee75f89b118055c3207432d.jpg)

## SiMHand: Mining Similar Hands for Large-Scale 3D Hand Pose Pre-training


### Images

![4cc7b3668ce9cf4022d9d50cb5b48f46f977861c33720ee59608ec1e06086775.jpg](../iclr_results/3228_SiMHand_ Mining Similar Hands for Large-Scale 3D Hand Pose Pre-training/images/4cc7b3668ce9cf4022d9d50cb5b48f46f977861c33720ee59608ec1e06086775.jpg)

![522d5d26ac272534a7af9258132290e3e4ba4b78e239dc7876875e8772477a4d.jpg](../iclr_results/3228_SiMHand_ Mining Similar Hands for Large-Scale 3D Hand Pose Pre-training/images/522d5d26ac272534a7af9258132290e3e4ba4b78e239dc7876875e8772477a4d.jpg)

![572342c1901a4f40474482d6c5f8b5908acaad48ccb8b437c76f42956543a241.jpg](../iclr_results/3228_SiMHand_ Mining Similar Hands for Large-Scale 3D Hand Pose Pre-training/images/572342c1901a4f40474482d6c5f8b5908acaad48ccb8b437c76f42956543a241.jpg)

![5a5d1cb095e104e01cfe780b0da3c8f2ab256888faeb53eb56b88b011c6c6e7b.jpg](../iclr_results/3228_SiMHand_ Mining Similar Hands for Large-Scale 3D Hand Pose Pre-training/images/5a5d1cb095e104e01cfe780b0da3c8f2ab256888faeb53eb56b88b011c6c6e7b.jpg)

![933577b3bad91bfa2e8a84727ba90e20fff6e34227b7c9cc939f940ecd878639.jpg](../iclr_results/3228_SiMHand_ Mining Similar Hands for Large-Scale 3D Hand Pose Pre-training/images/933577b3bad91bfa2e8a84727ba90e20fff6e34227b7c9cc939f940ecd878639.jpg)

![9f6ce18b445bd2d7033256fa3b3d71fa0fdffb4c84d45fc557fa3b7097c577e0.jpg](../iclr_results/3228_SiMHand_ Mining Similar Hands for Large-Scale 3D Hand Pose Pre-training/images/9f6ce18b445bd2d7033256fa3b3d71fa0fdffb4c84d45fc557fa3b7097c577e0.jpg)

![b41cfdaa78b8b8b00fa192f5d966fb2420a0bd9ef04dd43e39f4d32f84a70e9a.jpg](../iclr_results/3228_SiMHand_ Mining Similar Hands for Large-Scale 3D Hand Pose Pre-training/images/b41cfdaa78b8b8b00fa192f5d966fb2420a0bd9ef04dd43e39f4d32f84a70e9a.jpg)

![d304ca4a024e43f36a5397da2312bc8bc2f25c7bff998d8771838953ee5d5888.jpg](../iclr_results/3228_SiMHand_ Mining Similar Hands for Large-Scale 3D Hand Pose Pre-training/images/d304ca4a024e43f36a5397da2312bc8bc2f25c7bff998d8771838953ee5d5888.jpg)

### Tables

![0c715dcc74a7ce749e26e07dbf6bf79412fa672b26e9c330ff433935ce2b9629.jpg](../iclr_results/3228_SiMHand_ Mining Similar Hands for Large-Scale 3D Hand Pose Pre-training/tables/0c715dcc74a7ce749e26e07dbf6bf79412fa672b26e9c330ff433935ce2b9629.jpg)

![19f3617f17fff3186188c238a9b56d77c674b027bfe1594127946e7b6dee08e1.jpg](../iclr_results/3228_SiMHand_ Mining Similar Hands for Large-Scale 3D Hand Pose Pre-training/tables/19f3617f17fff3186188c238a9b56d77c674b027bfe1594127946e7b6dee08e1.jpg)

![3de734ee40f5507edc4aca63c61f9f952b1895f1179f35864f48bd862fef0501.jpg](../iclr_results/3228_SiMHand_ Mining Similar Hands for Large-Scale 3D Hand Pose Pre-training/tables/3de734ee40f5507edc4aca63c61f9f952b1895f1179f35864f48bd862fef0501.jpg)

![57ade0b0f409bcda9563175d67caed69670b41ec602f8d05c8226d3dcd1420cc.jpg](../iclr_results/3228_SiMHand_ Mining Similar Hands for Large-Scale 3D Hand Pose Pre-training/tables/57ade0b0f409bcda9563175d67caed69670b41ec602f8d05c8226d3dcd1420cc.jpg)

![6a47ba27107736bdb507c07ec0057f53a96685ddd3128ba6ba283874dbe0c9cb.jpg](../iclr_results/3228_SiMHand_ Mining Similar Hands for Large-Scale 3D Hand Pose Pre-training/tables/6a47ba27107736bdb507c07ec0057f53a96685ddd3128ba6ba283874dbe0c9cb.jpg)

![7f26df866cfcce85dc28699b6336cc9c040774f82ae0473540a557951a30b968.jpg](../iclr_results/3228_SiMHand_ Mining Similar Hands for Large-Scale 3D Hand Pose Pre-training/tables/7f26df866cfcce85dc28699b6336cc9c040774f82ae0473540a557951a30b968.jpg)

![b451b815e540c8b22d9e969fdfe461af5f6f9c4b747e48db7dd993fb10972373.jpg](../iclr_results/3228_SiMHand_ Mining Similar Hands for Large-Scale 3D Hand Pose Pre-training/tables/b451b815e540c8b22d9e969fdfe461af5f6f9c4b747e48db7dd993fb10972373.jpg)

![ca5ee685d9b3eaeddc28fc4d31bb4bac259146702d14a7b5d2675be7fb2e74f7.jpg](../iclr_results/3228_SiMHand_ Mining Similar Hands for Large-Scale 3D Hand Pose Pre-training/tables/ca5ee685d9b3eaeddc28fc4d31bb4bac259146702d14a7b5d2675be7fb2e74f7.jpg)

## Causal Identification for Complex Functional Longitudinal Studies


### Images

![96b4fdbfc110bc80e68207fbece2a6ef0ae13f1069fa5624197585355cf95340.jpg](../iclr_results/3229_Causal Identification for Complex Functional Longitudinal Studies/images/96b4fdbfc110bc80e68207fbece2a6ef0ae13f1069fa5624197585355cf95340.jpg)

![b0d8731bd12f388cfbcff1d97cadc21777b09e319774fd2abef1fd5de65eb0bb.jpg](../iclr_results/3229_Causal Identification for Complex Functional Longitudinal Studies/images/b0d8731bd12f388cfbcff1d97cadc21777b09e319774fd2abef1fd5de65eb0bb.jpg)

## Denoising Task Difficulty-based Curriculum for Training Diffusion Models


### Images

![0287f819752475eb472fab8c64965002c74576701464ff7dfb4133560359c2eb.jpg](../iclr_results/3230_Denoising Task Difficulty-based Curriculum for Training Diffusion Models/images/0287f819752475eb472fab8c64965002c74576701464ff7dfb4133560359c2eb.jpg)

![17373f729a812d0c4dffbbc5f8a7c59ddb87dc769b18b2c16fb53bb6b4b0fd3d.jpg](../iclr_results/3230_Denoising Task Difficulty-based Curriculum for Training Diffusion Models/images/17373f729a812d0c4dffbbc5f8a7c59ddb87dc769b18b2c16fb53bb6b4b0fd3d.jpg)

![6d0db5abe6064feff8c52e1a5e3d39ec88a360478d0463c0b8caa9dfdf84779c.jpg](../iclr_results/3230_Denoising Task Difficulty-based Curriculum for Training Diffusion Models/images/6d0db5abe6064feff8c52e1a5e3d39ec88a360478d0463c0b8caa9dfdf84779c.jpg)

![c1a3c1132b4a4684ab0105783ef8c2f20e540d7ca427cde971bd9f045ce7ebf5.jpg](../iclr_results/3230_Denoising Task Difficulty-based Curriculum for Training Diffusion Models/images/c1a3c1132b4a4684ab0105783ef8c2f20e540d7ca427cde971bd9f045ce7ebf5.jpg)

![d479ebe80451eb68b7bb93392fdf31f3dfabd47d1fe0e716629ba44d34ecd0d9.jpg](../iclr_results/3230_Denoising Task Difficulty-based Curriculum for Training Diffusion Models/images/d479ebe80451eb68b7bb93392fdf31f3dfabd47d1fe0e716629ba44d34ecd0d9.jpg)

![d6ce3e5d6f2e7b9ae36157678347d87b14fe3f064c1a06632b6acd6e89b39551.jpg](../iclr_results/3230_Denoising Task Difficulty-based Curriculum for Training Diffusion Models/images/d6ce3e5d6f2e7b9ae36157678347d87b14fe3f064c1a06632b6acd6e89b39551.jpg)

### Tables

![01c667270a6d04355916bc273354834e7b1e059162d941321fb1134fd2fe512d.jpg](../iclr_results/3230_Denoising Task Difficulty-based Curriculum for Training Diffusion Models/tables/01c667270a6d04355916bc273354834e7b1e059162d941321fb1134fd2fe512d.jpg)

![402e949efa86ad2eb26c06fe11bba4ec0a1f20c191600bf83da188157a4039ac.jpg](../iclr_results/3230_Denoising Task Difficulty-based Curriculum for Training Diffusion Models/tables/402e949efa86ad2eb26c06fe11bba4ec0a1f20c191600bf83da188157a4039ac.jpg)

![5f05b9116deb54dff557689488bb63ccda5f919a36e08e68ce7b3c367a6480e3.jpg](../iclr_results/3230_Denoising Task Difficulty-based Curriculum for Training Diffusion Models/tables/5f05b9116deb54dff557689488bb63ccda5f919a36e08e68ce7b3c367a6480e3.jpg)

![895262c6143b4156687161b59512824522d44cb6de4ef94450e503b3fe39fe4b.jpg](../iclr_results/3230_Denoising Task Difficulty-based Curriculum for Training Diffusion Models/tables/895262c6143b4156687161b59512824522d44cb6de4ef94450e503b3fe39fe4b.jpg)

![a13211cc0150ba4de60ceb22b39685a023ac4ab87ba21ea67ff2b64ea105b700.jpg](../iclr_results/3230_Denoising Task Difficulty-based Curriculum for Training Diffusion Models/tables/a13211cc0150ba4de60ceb22b39685a023ac4ab87ba21ea67ff2b64ea105b700.jpg)

## Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations


### Images

![00397c186a6ef9d247d93940e8a64d6429c5913fa35424029e6e32c80abae78f.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/00397c186a6ef9d247d93940e8a64d6429c5913fa35424029e6e32c80abae78f.jpg)

![0d6e267e9367e96764fa91a36d03830e58c27b6492c8b546cffcdce9d53b817e.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/0d6e267e9367e96764fa91a36d03830e58c27b6492c8b546cffcdce9d53b817e.jpg)

![0de08cc7736c291097970801717a0be360da03cb2dfcb3dbb19e5985dc7c4d7b.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/0de08cc7736c291097970801717a0be360da03cb2dfcb3dbb19e5985dc7c4d7b.jpg)

![1acd5e5706c30cf4322f93c76efdb864677a1909ca9283da993f95b57829d5d5.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/1acd5e5706c30cf4322f93c76efdb864677a1909ca9283da993f95b57829d5d5.jpg)

![1c1dcb00cbacc4409181297043d01c0636838eb126823c0c545d3c86dd874e4e.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/1c1dcb00cbacc4409181297043d01c0636838eb126823c0c545d3c86dd874e4e.jpg)

![210ee434782c161b82e50fa39fe85f4198bd5e26c317536c5dec4fef60fceeb5.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/210ee434782c161b82e50fa39fe85f4198bd5e26c317536c5dec4fef60fceeb5.jpg)

![2b13dec21bd4ea1d8dd67d6fbb385760903c2ea380805a665c23870f3bb300ce.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/2b13dec21bd4ea1d8dd67d6fbb385760903c2ea380805a665c23870f3bb300ce.jpg)

![36c0890b4f5a5240f9ea08f7a9df5d08a3b249f063986e2b72b73bd95f4f7330.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/36c0890b4f5a5240f9ea08f7a9df5d08a3b249f063986e2b72b73bd95f4f7330.jpg)

![38b2bbf56a6e95dc2426b4a5d05d8fc2262ac56ea3baa5aec98cd3a5608369c3.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/38b2bbf56a6e95dc2426b4a5d05d8fc2262ac56ea3baa5aec98cd3a5608369c3.jpg)

![3a0cef3399e79a97cf3ecae25025ccf2480539a8f6cc2f72cd9b2421a0271b6f.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/3a0cef3399e79a97cf3ecae25025ccf2480539a8f6cc2f72cd9b2421a0271b6f.jpg)

![473421a03be3db6fd9be428163785acbcd59e6b110c00620cf72d10f9e992407.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/473421a03be3db6fd9be428163785acbcd59e6b110c00620cf72d10f9e992407.jpg)

![4ccc6a86f8104d459028274f05676d9849bdf74a51a5ca81064856a557f33518.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/4ccc6a86f8104d459028274f05676d9849bdf74a51a5ca81064856a557f33518.jpg)

![50a618ba77b21cf225296974ff6db86b8ecdeb4d920b345eec16509ae9e96166.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/50a618ba77b21cf225296974ff6db86b8ecdeb4d920b345eec16509ae9e96166.jpg)

![50add790587ddb59177a317463a02daa8ebd53bcd9c1c66e61b59d0a1fb05e2b.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/50add790587ddb59177a317463a02daa8ebd53bcd9c1c66e61b59d0a1fb05e2b.jpg)

![50bea395dc44bff98479e96055dd33c07cb2d8842a74596877abbe9b396a65a9.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/50bea395dc44bff98479e96055dd33c07cb2d8842a74596877abbe9b396a65a9.jpg)

![539cd15ad1f6c4180a8d66bd6f3f599542a1d9f2227411ef29d6578ac6de44fc.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/539cd15ad1f6c4180a8d66bd6f3f599542a1d9f2227411ef29d6578ac6de44fc.jpg)

![6a5e519b7e450b9c69e46f785a7f7ed7811ccad1795a257f61c847c2e97d8434.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/6a5e519b7e450b9c69e46f785a7f7ed7811ccad1795a257f61c847c2e97d8434.jpg)

![6c59370696f0cb627082ab1059d61ebb53b0984808186e271cfbd3ed7aa4e79b.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/6c59370696f0cb627082ab1059d61ebb53b0984808186e271cfbd3ed7aa4e79b.jpg)

![879bfc2f890102b2bcfd7feced1beb6a6702386d2801d34270378fea12be9c06.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/879bfc2f890102b2bcfd7feced1beb6a6702386d2801d34270378fea12be9c06.jpg)

![8e1be0523974a867a820f90ad453b10fa4041d306f054bcafb92dd7b58440d47.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/8e1be0523974a867a820f90ad453b10fa4041d306f054bcafb92dd7b58440d47.jpg)

![9fd535906844758645f349862e941000fdb4e853261b16abb41c170eb7c87fca.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/9fd535906844758645f349862e941000fdb4e853261b16abb41c170eb7c87fca.jpg)

![a3098c19e018ec72a9a47f3b32fd5dc9f31de11ed672599c642ad9dd7ba057a5.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/a3098c19e018ec72a9a47f3b32fd5dc9f31de11ed672599c642ad9dd7ba057a5.jpg)

![a508145a1bd59220178a7f0a23dc228a86bd33cfd32ade2eee50d1da7300faf5.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/a508145a1bd59220178a7f0a23dc228a86bd33cfd32ade2eee50d1da7300faf5.jpg)

![a76bb194f5d96ef1cf8541221e27781d03f2f62655781082fe0a484596abf5e3.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/a76bb194f5d96ef1cf8541221e27781d03f2f62655781082fe0a484596abf5e3.jpg)

![abcf751a2368b9f13f7d4899089ee1621461eff1969ab59ff854346faf69a645.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/abcf751a2368b9f13f7d4899089ee1621461eff1969ab59ff854346faf69a645.jpg)

![b00c45d47c98412a554d79ec98c7430018a3c499193c2fdcea5f27fce5186453.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/b00c45d47c98412a554d79ec98c7430018a3c499193c2fdcea5f27fce5186453.jpg)

![b1429bcb3a0ffc659738aae1be14a69c9e5843085209bd64beb0744dcfb391db.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/b1429bcb3a0ffc659738aae1be14a69c9e5843085209bd64beb0744dcfb391db.jpg)

![b6d85b01026a712a65ee39b22b8de8a152345bead7604b5dc7a9300d952d71af.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/b6d85b01026a712a65ee39b22b8de8a152345bead7604b5dc7a9300d952d71af.jpg)

![bac8a52d7e8b368bcbc2142119c15f1046ed7c8f4b7f44679b4bc5b0b137a966.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/bac8a52d7e8b368bcbc2142119c15f1046ed7c8f4b7f44679b4bc5b0b137a966.jpg)

![bf637d2bf1df4963e349c7aa9c6434ed1bc41a6f533bf579fec11f8c0415e05c.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/bf637d2bf1df4963e349c7aa9c6434ed1bc41a6f533bf579fec11f8c0415e05c.jpg)

![c73d8f10bfb40ed3dba6bd5ca233a211fc439bce08de5b0327bd21768e8fca5a.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/c73d8f10bfb40ed3dba6bd5ca233a211fc439bce08de5b0327bd21768e8fca5a.jpg)

![c912a4efaf6fe593a673eaac3bab81df63523279630fb5adc361bf40df1fd2f1.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/c912a4efaf6fe593a673eaac3bab81df63523279630fb5adc361bf40df1fd2f1.jpg)

![cf4c4bd99f20ff1fc884f46ff97f995c16966bd18fc50bf2a7ba41541620ca13.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/cf4c4bd99f20ff1fc884f46ff97f995c16966bd18fc50bf2a7ba41541620ca13.jpg)

![d051d548537e6228daa36cabfe74614097787666f5198ac47ca046f4deae17f8.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/d051d548537e6228daa36cabfe74614097787666f5198ac47ca046f4deae17f8.jpg)

![d2ff04b42e650877d8089965df8217127f604666bd14a9c7e3ce95d5e7c93b4e.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/d2ff04b42e650877d8089965df8217127f604666bd14a9c7e3ce95d5e7c93b4e.jpg)

![e4325a868efa803614974431e1cfdd158f8d5ebd1195fe9422f32daa0b9816d1.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/e4325a868efa803614974431e1cfdd158f8d5ebd1195fe9422f32daa0b9816d1.jpg)

![e6cb8d836c675845701ecd69e95711848972eee36f540defe6a69562f21b9d78.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/e6cb8d836c675845701ecd69e95711848972eee36f540defe6a69562f21b9d78.jpg)

![e7a89deb634b245abfbb3a1c5a7bd09dec651478cbc40a65dbb9d7a0711e723f.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/e7a89deb634b245abfbb3a1c5a7bd09dec651478cbc40a65dbb9d7a0711e723f.jpg)

![f536335c80bf4c3bc6f9b481893f462b282011089d7134263e357b443d2d53c8.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/f536335c80bf4c3bc6f9b481893f462b282011089d7134263e357b443d2d53c8.jpg)

![f7619a9ce82c0d68f50bef43140fddafaf48610492d4131aca9545776b3bad5b.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/f7619a9ce82c0d68f50bef43140fddafaf48610492d4131aca9545776b3bad5b.jpg)

![f8c869115761ffc56c0190a66d62a8f1b830b59ba76d482ab11d6c5444c90ff4.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/f8c869115761ffc56c0190a66d62a8f1b830b59ba76d482ab11d6c5444c90ff4.jpg)

![fdef8fabbe96e39abae0a8279952dd11fa0845fce9b60580986ddf63b0aadabb.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/images/fdef8fabbe96e39abae0a8279952dd11fa0845fce9b60580986ddf63b0aadabb.jpg)

### Tables

![2d73db377585d4c159108ef4a2f53494f24eb11d43fdf902ad7564b490f3c1b5.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/tables/2d73db377585d4c159108ef4a2f53494f24eb11d43fdf902ad7564b490f3c1b5.jpg)

![2f11f8361afdb58c424dfc04c4c36611d44338a450d55b7fb31bbaf85b94a45d.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/tables/2f11f8361afdb58c424dfc04c4c36611d44338a450d55b7fb31bbaf85b94a45d.jpg)

![591a07f48e1caa05cb85026589bdfc5b1a089470628e73542d97de365c55a6b7.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/tables/591a07f48e1caa05cb85026589bdfc5b1a089470628e73542d97de365c55a6b7.jpg)

![752cbe3b42bf3802e73656877c0d5ffc4dfe7ab5a7c0301927d573738c25a9b8.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/tables/752cbe3b42bf3802e73656877c0d5ffc4dfe7ab5a7c0301927d573738c25a9b8.jpg)

![8cbce24f83dd228fd1b75b22680f8c16fa104629d4e5e062286772bdaaad0d52.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/tables/8cbce24f83dd228fd1b75b22680f8c16fa104629d4e5e062286772bdaaad0d52.jpg)

![98aae958d01fe9603e15013319c202a34a776b6fc0440d32b8af36b222bf8785.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/tables/98aae958d01fe9603e15013319c202a34a776b6fc0440d32b8af36b222bf8785.jpg)

![99a575ed2a6eee91365e9df1bed8a5486b1df50b4d286cbb39d9506cb3c69f34.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/tables/99a575ed2a6eee91365e9df1bed8a5486b1df50b4d286cbb39d9506cb3c69f34.jpg)

![bc2ceff408e0d8bce79fef53420bcb2c9f3a8dd9c666bb41c717c98067fc6066.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/tables/bc2ceff408e0d8bce79fef53420bcb2c9f3a8dd9c666bb41c717c98067fc6066.jpg)

![ef6cc29e4cabdc18d970263d3584d15a02c705d9a79287718fa35a7eed13bf3d.jpg](../iclr_results/3231_Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations/tables/ef6cc29e4cabdc18d970263d3584d15a02c705d9a79287718fa35a7eed13bf3d.jpg)

## Data Pruning by Information Maximization


### Images

![0329b2732e0c88a1a2d7fbc2649134373a935ac5371da3e7d9d0781868f6426d.jpg](../iclr_results/3232_Data Pruning by Information Maximization/images/0329b2732e0c88a1a2d7fbc2649134373a935ac5371da3e7d9d0781868f6426d.jpg)

![043edf27b22e02933335fed0c668bce8fe1efdf36980810da6171c7873acd128.jpg](../iclr_results/3232_Data Pruning by Information Maximization/images/043edf27b22e02933335fed0c668bce8fe1efdf36980810da6171c7873acd128.jpg)

![2fc773bc19649b9283cfe2b6f883edb7925432d035a8358938300283e89c788c.jpg](../iclr_results/3232_Data Pruning by Information Maximization/images/2fc773bc19649b9283cfe2b6f883edb7925432d035a8358938300283e89c788c.jpg)

![635e2923c3724463bd6cf85ccd06623f8213680ebf474d22b3081f147f60bd6d.jpg](../iclr_results/3232_Data Pruning by Information Maximization/images/635e2923c3724463bd6cf85ccd06623f8213680ebf474d22b3081f147f60bd6d.jpg)

![bb7337e742561a8dfb0fe696a613d2e3cb1614e914ac602171878b4d936c1761.jpg](../iclr_results/3232_Data Pruning by Information Maximization/images/bb7337e742561a8dfb0fe696a613d2e3cb1614e914ac602171878b4d936c1761.jpg)

![e5a7b105de82ced830bc91b7d0a1d96a68bc52c8a47e2ab2981b232604a87f12.jpg](../iclr_results/3232_Data Pruning by Information Maximization/images/e5a7b105de82ced830bc91b7d0a1d96a68bc52c8a47e2ab2981b232604a87f12.jpg)

![ea617b31a672695b8a0514500c23d30e1a5de3f6919c121ad7d55c2ec6e3dcfd.jpg](../iclr_results/3232_Data Pruning by Information Maximization/images/ea617b31a672695b8a0514500c23d30e1a5de3f6919c121ad7d55c2ec6e3dcfd.jpg)

### Tables

![168739213a4ade6a64ece7e949861fc904f7087b0ed26553dd46c477c9651b08.jpg](../iclr_results/3232_Data Pruning by Information Maximization/tables/168739213a4ade6a64ece7e949861fc904f7087b0ed26553dd46c477c9651b08.jpg)

![30204d27a9fab50c324ced4a0f1becacce81a687272f3d03d77728202d99460f.jpg](../iclr_results/3232_Data Pruning by Information Maximization/tables/30204d27a9fab50c324ced4a0f1becacce81a687272f3d03d77728202d99460f.jpg)

![91f39510e155ab2c745d7e69bfd9be305d9879e4b7252094208456601e39b02f.jpg](../iclr_results/3232_Data Pruning by Information Maximization/tables/91f39510e155ab2c745d7e69bfd9be305d9879e4b7252094208456601e39b02f.jpg)

![bbd5647038e8610cf6690ef210b040e70d03d8403b497402c279684088f705e7.jpg](../iclr_results/3232_Data Pruning by Information Maximization/tables/bbd5647038e8610cf6690ef210b040e70d03d8403b497402c279684088f705e7.jpg)

![c3a202c8a68bf500bc22414c02f05c3849b53863b2ab2a96280da98ab747879d.jpg](../iclr_results/3232_Data Pruning by Information Maximization/tables/c3a202c8a68bf500bc22414c02f05c3849b53863b2ab2a96280da98ab747879d.jpg)

## Edge Prompt Tuning for Graph Neural Networks


### Images

![14a800d6b2c14baf5bb13d7ef171c9c6c392609d99a40feb0254d0400b0ca08a.jpg](../iclr_results/3233_Edge Prompt Tuning for Graph Neural Networks/images/14a800d6b2c14baf5bb13d7ef171c9c6c392609d99a40feb0254d0400b0ca08a.jpg)

![1e8afd9b469779690d7e99ccb00756a7eafcada87c07a97fe55945dee0be41e4.jpg](../iclr_results/3233_Edge Prompt Tuning for Graph Neural Networks/images/1e8afd9b469779690d7e99ccb00756a7eafcada87c07a97fe55945dee0be41e4.jpg)

![b3364287a59439108604b1a1d516a4ea4d275bfa2bf06ed5863875be4493840e.jpg](../iclr_results/3233_Edge Prompt Tuning for Graph Neural Networks/images/b3364287a59439108604b1a1d516a4ea4d275bfa2bf06ed5863875be4493840e.jpg)

![c37e017942ef6256f888e1eb355db9792d1ba73ed926947c003dbee2fa1d58cb.jpg](../iclr_results/3233_Edge Prompt Tuning for Graph Neural Networks/images/c37e017942ef6256f888e1eb355db9792d1ba73ed926947c003dbee2fa1d58cb.jpg)

![d1af719c1fef027ed3c86163a5e5ec615264e6daf53b69454a16c60097d65dbf.jpg](../iclr_results/3233_Edge Prompt Tuning for Graph Neural Networks/images/d1af719c1fef027ed3c86163a5e5ec615264e6daf53b69454a16c60097d65dbf.jpg)

### Tables

![0ab75ae0e25fc1c13da3fbe36bc718dbd482bbafa0aa0186c175348a841b5e50.jpg](../iclr_results/3233_Edge Prompt Tuning for Graph Neural Networks/tables/0ab75ae0e25fc1c13da3fbe36bc718dbd482bbafa0aa0186c175348a841b5e50.jpg)

![0cc5c90f3e16648c6955dd17fe27b2abad1c3fc3005e6db7a071b4da26ab7aa5.jpg](../iclr_results/3233_Edge Prompt Tuning for Graph Neural Networks/tables/0cc5c90f3e16648c6955dd17fe27b2abad1c3fc3005e6db7a071b4da26ab7aa5.jpg)

![10b600952458d175359bc459363185e44a7538e25fa826a36b9e3cd0a2a1003d.jpg](../iclr_results/3233_Edge Prompt Tuning for Graph Neural Networks/tables/10b600952458d175359bc459363185e44a7538e25fa826a36b9e3cd0a2a1003d.jpg)

![1214a90c2807438432ec70d1a1083e9c865d5853931b99f563a34098e2957abc.jpg](../iclr_results/3233_Edge Prompt Tuning for Graph Neural Networks/tables/1214a90c2807438432ec70d1a1083e9c865d5853931b99f563a34098e2957abc.jpg)

![15aef143fef8f4add0b13bed80206b2a4a4816654cbbddf797dfab5dbf8f50c9.jpg](../iclr_results/3233_Edge Prompt Tuning for Graph Neural Networks/tables/15aef143fef8f4add0b13bed80206b2a4a4816654cbbddf797dfab5dbf8f50c9.jpg)

![304c3d6a6e51a205f62c2e2dc24d56f1dc9a32ea172b32b5880180ef281260bc.jpg](../iclr_results/3233_Edge Prompt Tuning for Graph Neural Networks/tables/304c3d6a6e51a205f62c2e2dc24d56f1dc9a32ea172b32b5880180ef281260bc.jpg)

![36037e47356f01d5924d47a8c98b0d76432707b8ec7400016c6abd258d0ea577.jpg](../iclr_results/3233_Edge Prompt Tuning for Graph Neural Networks/tables/36037e47356f01d5924d47a8c98b0d76432707b8ec7400016c6abd258d0ea577.jpg)

![3711672e3227948f9a7630bd3217a54058c90260f6ee5de305ef8a6d3b4f09f0.jpg](../iclr_results/3233_Edge Prompt Tuning for Graph Neural Networks/tables/3711672e3227948f9a7630bd3217a54058c90260f6ee5de305ef8a6d3b4f09f0.jpg)

![78c7258afba89efe4931c21ef5cdfd523f6b0e66b92cc3bbbb5ad400479288da.jpg](../iclr_results/3233_Edge Prompt Tuning for Graph Neural Networks/tables/78c7258afba89efe4931c21ef5cdfd523f6b0e66b92cc3bbbb5ad400479288da.jpg)

![b2ccabc2c828b691893315bba9f2aea61fad04481d4ec8ed7c5f6c7145892033.jpg](../iclr_results/3233_Edge Prompt Tuning for Graph Neural Networks/tables/b2ccabc2c828b691893315bba9f2aea61fad04481d4ec8ed7c5f6c7145892033.jpg)

![d68b25b0a0205311606c755fd2bdf9f94ff4c6fbdf358e5d5c241c693b46571a.jpg](../iclr_results/3233_Edge Prompt Tuning for Graph Neural Networks/tables/d68b25b0a0205311606c755fd2bdf9f94ff4c6fbdf358e5d5c241c693b46571a.jpg)

## Joint Fine-tuning and Conversion of Pretrained Speech and Language Models towards Linear Complexity

### Images

![1ef0c701ceb81e61995a8fa31696d0916b531a652b4232c8797ee200373d9879.jpg](../iclr_results/3234_Joint Fine-tuning and Conversion of Pretrained Speech and Language Models towards Linear Complexity/images/1ef0c701ceb81e61995a8fa31696d0916b531a652b4232c8797ee200373d9879.jpg)

![2fcdedabf35a2e93f6310fb3ab0d57753567a9ea910798f4509eea90a32991ea.jpg](../iclr_results/3234_Joint Fine-tuning and Conversion of Pretrained Speech and Language Models towards Linear Complexity/images/2fcdedabf35a2e93f6310fb3ab0d57753567a9ea910798f4509eea90a32991ea.jpg)

![316bf406a0d502e1ee56fec117e0d9e9c2c5e3970c2d463d4b3274024c302e13.jpg](../iclr_results/3234_Joint Fine-tuning and Conversion of Pretrained Speech and Language Models towards Linear Complexity/images/316bf406a0d502e1ee56fec117e0d9e9c2c5e3970c2d463d4b3274024c302e13.jpg)

![8187a02851ba145cb416d706395fac598cd82b72504cf017859076e90cf16ba8.jpg](../iclr_results/3234_Joint Fine-tuning and Conversion of Pretrained Speech and Language Models towards Linear Complexity/images/8187a02851ba145cb416d706395fac598cd82b72504cf017859076e90cf16ba8.jpg)

![8b4763bd8dc30ed088b9eecb216ee9a5b292dd0114eb1b3ab0368eb9b07c8197.jpg](../iclr_results/3234_Joint Fine-tuning and Conversion of Pretrained Speech and Language Models towards Linear Complexity/images/8b4763bd8dc30ed088b9eecb216ee9a5b292dd0114eb1b3ab0368eb9b07c8197.jpg)

### Tables

![33fc0ccff0a728116939589a2f559439e4fcccd6aabcca213af89b7ba6f11bc7.jpg](../iclr_results/3234_Joint Fine-tuning and Conversion of Pretrained Speech and Language Models towards Linear Complexity/tables/33fc0ccff0a728116939589a2f559439e4fcccd6aabcca213af89b7ba6f11bc7.jpg)

![cd40897753bf46f2c5be9fdbc9abfd4663ad9275558c207d9ba8c55f3c8105da.jpg](../iclr_results/3234_Joint Fine-tuning and Conversion of Pretrained Speech and Language Models towards Linear Complexity/tables/cd40897753bf46f2c5be9fdbc9abfd4663ad9275558c207d9ba8c55f3c8105da.jpg)

![ec70231fe36d69d7d14975749fdd9b1f4a908ef89dee15026c3ea840ef1c6593.jpg](../iclr_results/3234_Joint Fine-tuning and Conversion of Pretrained Speech and Language Models towards Linear Complexity/tables/ec70231fe36d69d7d14975749fdd9b1f4a908ef89dee15026c3ea840ef1c6593.jpg)

![faf08783b035450d563d7cb6146a68f5bc2dc5b4419778052dd73bfdb1e56ac5.jpg](../iclr_results/3234_Joint Fine-tuning and Conversion of Pretrained Speech and Language Models towards Linear Complexity/tables/faf08783b035450d563d7cb6146a68f5bc2dc5b4419778052dd73bfdb1e56ac5.jpg)
