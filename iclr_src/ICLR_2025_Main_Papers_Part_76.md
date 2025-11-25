# ICLR 2025 Main Conference Papers

**Summary:** 36 papers with extracted content:
- 📊 Total images: 44031
- 📋 Total tables: 33468
- 📄 Total files: 77499

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 76 of 100

## 目录 (Table of Contents)

1. [Remove Symmetries to Control Model Expressivity and Improve Optimization](#Remove-Symmetries-to-Control-Model-Expressivity-and-Improve-Optimization)
2. [Round and Round We Go! What makes Rotary Positional Encodings useful?](#Round-and-Round-We-Go-What-makes-Rotary-Positional-Encodings-useful)
3. [Advancing Mathematical Reasoning in Language Models: The Impact of Problem-Solving Data, Data Synthesis Methods, and Training Stages](#Advancing-Mathematical-Reasoning-in-Language-Models-The-Impact-of-Problem-Solving-Data-Data-Synthesis-Methods-and-Training-Stages)
4. [Learning Gain Map for Inverse Tone Mapping](#Learning-Gain-Map-for-Inverse-Tone-Mapping)
5. [Robust Simulation-Based Inference under Missing Data via Neural Processes](#Robust-Simulation-Based-Inference-under-Missing-Data-via-Neural-Processes)
6. [Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings](#Adversarially-Robust-Out-of-Distribution-Detection-Using-Lyapunov-Stabilized-Embeddings)
7. [MuseGNN: Forming Scalable, Convergent GNN Layers that Minimize a Sampling-Based Energy](#MuseGNN-Forming-Scalable-Convergent-GNN-Layers-that-Minimize-a-Sampling-Based-Energy)
8. [The Crystal Ball Hypothesis in diffusion models: Anticipating object positions from initial noise](#The-Crystal-Ball-Hypothesis-in-diffusion-models-Anticipating-object-positions-from-initial-noise)
9. [Towards counterfactual fairness through auxiliary variables](#Towards-counterfactual-fairness-through-auxiliary-variables)
10. [Geometry Image Diffusion: Fast and Data-Efficient Text-to-3D with Image-Based Surface Representation](#Geometry-Image-Diffusion-Fast-and-Data-Efficient-Text-to-3D-with-Image-Based-Surface-Representation)
11. [Multi-agent cooperation through learning-aware policy gradients](#Multi-agent-cooperation-through-learning-aware-policy-gradients)
12. [StructRAG: Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Structurization](#StructRAG-Boosting-Knowledge-Intensive-Reasoning-of-LLMs-via-Inference-time-Hybrid-Information-Structurization)
13. [Unhackable Temporal Reward for Scalable Video MLLMs](#Unhackable-Temporal-Reward-for-Scalable-Video-MLLMs)
14. [Taming Transformer Without Using Learning Rate Warmup](#Taming-Transformer-Without-Using-Learning-Rate-Warmup)
15. [How Learnable Grids Recover Fine Detail in Low Dimensions: A Neural Tangent Kernel Analysis of Multigrid Parametric Encodings](#How-Learnable-Grids-Recover-Fine-Detail-in-Low-Dimensions-A-Neural-Tangent-Kernel-Analysis-of-Multigrid-Parametric-Encodings)
16. [NeurFlow: Interpreting Neural Networks through Neuron Groups and Functional Interactions](#NeurFlow-Interpreting-Neural-Networks-through-Neuron-Groups-and-Functional-Interactions)
17. [RaSA: Rank-Sharing Low-Rank Adaptation](#RaSA-Rank-Sharing-Low-Rank-Adaptation)
18. [Rethinking Diffusion Posterior Sampling: From Conditional Score Estimator to Maximizing a Posterior](#Rethinking-Diffusion-Posterior-Sampling-From-Conditional-Score-Estimator-to-Maximizing-a-Posterior)
19. [Adversarial Mixup Unlearning](#Adversarial-Mixup-Unlearning)
20. [SMT: Fine-Tuning Large Language Models with Sparse Matrices](#SMT-Fine-Tuning-Large-Language-Models-with-Sparse-Matrices)
21. [3D-AffordanceLLM: Harnessing Large Language Models for Open-Vocabulary Affordance Detection in 3D Worlds](#3D-AffordanceLLM-Harnessing-Large-Language-Models-for-Open-Vocabulary-Affordance-Detection-in-3D-Worlds)
22. [Mutual Effort for Efficiency: A Similarity-based Token Pruning for Vision Transformers in Self-Supervised Learning](#Mutual-Effort-for-Efficiency-A-Similarity-based-Token-Pruning-for-Vision-Transformers-in-Self-Supervised-Learning)
23. [Bridging Compressed Image Latents and Multimodal Large Language Models](#Bridging-Compressed-Image-Latents-and-Multimodal-Large-Language-Models)
24. [MAPS: Advancing Multi-Modal Reasoning in Expert-Level Physical Science](#MAPS-Advancing-Multi-Modal-Reasoning-in-Expert-Level-Physical-Science)
25. [Weakly Supervised Video Scene Graph Generation via Natural Language Supervision](#Weakly-Supervised-Video-Scene-Graph-Generation-via-Natural-Language-Supervision)
26. [Optimizing $(L_0, L_1)$-Smooth Functions by Gradient Methods](#Optimizing-L_0-L_1-Smooth-Functions-by-Gradient-Methods)
27. [Interpreting the Second-Order Effects of Neurons in CLIP](#Interpreting-the-Second-Order-Effects-of-Neurons-in-CLIP)
28. [SaLoRA: Safety-Alignment Preserved Low-Rank Adaptation](#SaLoRA-Safety-Alignment-Preserved-Low-Rank-Adaptation)
29. [Rethinking Neural Multi-Objective Combinatorial Optimization via Neat Weight Embedding](#Rethinking-Neural-Multi-Objective-Combinatorial-Optimization-via-Neat-Weight-Embedding)
30. [Skill Expansion and Composition in Parameter Space](#Skill-Expansion-and-Composition-in-Parameter-Space)
31. [ADMM for Nonconvex Optimization under Minimal Continuity Assumption](#ADMM-for-Nonconvex-Optimization-under-Minimal-Continuity-Assumption)
32. [TFG-Flow: Training-free Guidance in Multimodal Generative Flow](#TFG-Flow-Training-free-Guidance-in-Multimodal-Generative-Flow)
33. [Efficient Active Imitation Learning with Random Network Distillation](#Efficient-Active-Imitation-Learning-with-Random-Network-Distillation)
34. [Neural Functions for Learning Periodic Signal](#Neural-Functions-for-Learning-Periodic-Signal)
35. [ExACT: Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning](#ExACT-Teaching-AI-Agents-to-Explore-with-Reflective-MCTS-and-Exploratory-Learning)
36. [Singular Subspace Perturbation Bounds via Rectangular Random Matrix Diffusions](#Singular-Subspace-Perturbation-Bounds-via-Rectangular-Random-Matrix-Diffusions)

---


## Remove Symmetries to Control Model Expressivity and Improve Optimization

### Images

![25ca3cb29340bba170dc63f6a34e63835de005f13feaa4b556f484659b885f71.jpg](../iclr_results/2819_3DTrajMaster_ Mastering 3D Trajectory for Multi-Entity Motion in Video Generation/images/25ca3cb29340bba170dc63f6a34e63835de005f13feaa4b556f484659b885f71.jpg)

![380330a6176b9a43967cad8e32dc1637d16583ad028dbd73d3d9404dbd75dccf.jpg](../iclr_results/2819_3DTrajMaster_ Mastering 3D Trajectory for Multi-Entity Motion in Video Generation/images/380330a6176b9a43967cad8e32dc1637d16583ad028dbd73d3d9404dbd75dccf.jpg)

![57bdabc76dce1f716bc1880b9196c93528950dec9d537ee4f64762f369cd9f1e.jpg](../iclr_results/2819_3DTrajMaster_ Mastering 3D Trajectory for Multi-Entity Motion in Video Generation/images/57bdabc76dce1f716bc1880b9196c93528950dec9d537ee4f64762f369cd9f1e.jpg)

![b7187098b8315b20e9a7026d9b391cdd03c939d3b378cb57a044954f5214c095.jpg](../iclr_results/2819_3DTrajMaster_ Mastering 3D Trajectory for Multi-Entity Motion in Video Generation/images/b7187098b8315b20e9a7026d9b391cdd03c939d3b378cb57a044954f5214c095.jpg)

![c43753edf6e1e47d38c9dc9bd675ae23163e5a2b1751000b0396fd9d9251f5b0.jpg](../iclr_results/2819_3DTrajMaster_ Mastering 3D Trajectory for Multi-Entity Motion in Video Generation/images/c43753edf6e1e47d38c9dc9bd675ae23163e5a2b1751000b0396fd9d9251f5b0.jpg)

![c5bbceefb50ecbb881251450f8689d18381d4d1da03c7a75763312a0da49258c.jpg](../iclr_results/2819_3DTrajMaster_ Mastering 3D Trajectory for Multi-Entity Motion in Video Generation/images/c5bbceefb50ecbb881251450f8689d18381d4d1da03c7a75763312a0da49258c.jpg)

![e31947f2ae3311b4780026d9ecdf101246e8b4a1e4b24beefb8d8992561b4cf6.jpg](../iclr_results/2819_3DTrajMaster_ Mastering 3D Trajectory for Multi-Entity Motion in Video Generation/images/e31947f2ae3311b4780026d9ecdf101246e8b4a1e4b24beefb8d8992561b4cf6.jpg)

### Tables

![3b0b002a20b5852e537cb52776aacffef9ec717e7b506ccd61f9bb89b78e93ff.jpg](../iclr_results/2819_3DTrajMaster_ Mastering 3D Trajectory for Multi-Entity Motion in Video Generation/tables/3b0b002a20b5852e537cb52776aacffef9ec717e7b506ccd61f9bb89b78e93ff.jpg)

![4b0ed4706f831c5942404741dd1470897f59548ea82b0b00abdfb3dd389354ec.jpg](../iclr_results/2819_3DTrajMaster_ Mastering 3D Trajectory for Multi-Entity Motion in Video Generation/tables/4b0ed4706f831c5942404741dd1470897f59548ea82b0b00abdfb3dd389354ec.jpg)

![c813e566906ebb779ecb176901d9547c822d201a2b812a47bd42fd868c48bd15.jpg](../iclr_results/2819_3DTrajMaster_ Mastering 3D Trajectory for Multi-Entity Motion in Video Generation/tables/c813e566906ebb779ecb176901d9547c822d201a2b812a47bd42fd868c48bd15.jpg)

## Remove Symmetries to Control Model Expressivity and Improve Optimization


### Images

![16034970228ae0e36a2ba79af8418cf18113135c208fc3c44658632705bdf6b4.jpg](../iclr_results/2820_Remove Symmetries to Control Model Expressivity and Improve Optimization/images/16034970228ae0e36a2ba79af8418cf18113135c208fc3c44658632705bdf6b4.jpg)

![1def79106b44e5b4315d20971f3b194bf7b3f145a8059b701c395c29700e9a77.jpg](../iclr_results/2820_Remove Symmetries to Control Model Expressivity and Improve Optimization/images/1def79106b44e5b4315d20971f3b194bf7b3f145a8059b701c395c29700e9a77.jpg)

![23d069ba952a96c1cf7c9a6d384a60c0813051746c141016a2de413ba6dad868.jpg](../iclr_results/2820_Remove Symmetries to Control Model Expressivity and Improve Optimization/images/23d069ba952a96c1cf7c9a6d384a60c0813051746c141016a2de413ba6dad868.jpg)

![39c39d7dce1291d5b01ba833aa0747989561c9a12c279802b5788f29004ef7c7.jpg](../iclr_results/2820_Remove Symmetries to Control Model Expressivity and Improve Optimization/images/39c39d7dce1291d5b01ba833aa0747989561c9a12c279802b5788f29004ef7c7.jpg)

![3b8721e99eb9f3ad82064ac020f4210973e78557fcb2de32d5eef8501feefc4b.jpg](../iclr_results/2820_Remove Symmetries to Control Model Expressivity and Improve Optimization/images/3b8721e99eb9f3ad82064ac020f4210973e78557fcb2de32d5eef8501feefc4b.jpg)

![60c8fbc7afddd0e7d320ef121f138c5da4636a0d5c1b82fcc26dfbdbdb5f5d29.jpg](../iclr_results/2820_Remove Symmetries to Control Model Expressivity and Improve Optimization/images/60c8fbc7afddd0e7d320ef121f138c5da4636a0d5c1b82fcc26dfbdbdb5f5d29.jpg)

![67b17a11d9adf16d703d59615fec430725318762830c200ef65044b023f1d7d2.jpg](../iclr_results/2820_Remove Symmetries to Control Model Expressivity and Improve Optimization/images/67b17a11d9adf16d703d59615fec430725318762830c200ef65044b023f1d7d2.jpg)

![6f56056d069b899c3e28b0c46e450a3002548ebdbbc0eddced92ca0168775c33.jpg](../iclr_results/2820_Remove Symmetries to Control Model Expressivity and Improve Optimization/images/6f56056d069b899c3e28b0c46e450a3002548ebdbbc0eddced92ca0168775c33.jpg)

![81d5f8df547af422bc8d8f8ac57abf51cabce082c495bd543cc904fbbf42a7ed.jpg](../iclr_results/2820_Remove Symmetries to Control Model Expressivity and Improve Optimization/images/81d5f8df547af422bc8d8f8ac57abf51cabce082c495bd543cc904fbbf42a7ed.jpg)

![88d2d293e9a7b0e8f993202a2788b24ca5386e4831f177768168bf9fc6fbf299.jpg](../iclr_results/2820_Remove Symmetries to Control Model Expressivity and Improve Optimization/images/88d2d293e9a7b0e8f993202a2788b24ca5386e4831f177768168bf9fc6fbf299.jpg)

![a05f8bc4741ddd29c237b35f7035959a163f88072984cd5135c4fcafe9c00380.jpg](../iclr_results/2820_Remove Symmetries to Control Model Expressivity and Improve Optimization/images/a05f8bc4741ddd29c237b35f7035959a163f88072984cd5135c4fcafe9c00380.jpg)

![b0ccc44cc6fe8e66934ecb1000d6e673693ef62d43d626ac19bf31d58865d5bb.jpg](../iclr_results/2820_Remove Symmetries to Control Model Expressivity and Improve Optimization/images/b0ccc44cc6fe8e66934ecb1000d6e673693ef62d43d626ac19bf31d58865d5bb.jpg)

![f5695567cad28bf03370b170adf48afaa6f4f451045a674f5e0f5886155b626b.jpg](../iclr_results/2820_Remove Symmetries to Control Model Expressivity and Improve Optimization/images/f5695567cad28bf03370b170adf48afaa6f4f451045a674f5e0f5886155b626b.jpg)

![f8e3f12e9dc150daaf45d5201db5ecd041803fd0dc9aea818d752fcca357c7e6.jpg](../iclr_results/2820_Remove Symmetries to Control Model Expressivity and Improve Optimization/images/f8e3f12e9dc150daaf45d5201db5ecd041803fd0dc9aea818d752fcca357c7e6.jpg)

### Tables

![01174bcfe956cee055989a7f6cbc8a5f9bd6ca16d4f5d00e0820558f06a4d934.jpg](../iclr_results/2820_Remove Symmetries to Control Model Expressivity and Improve Optimization/tables/01174bcfe956cee055989a7f6cbc8a5f9bd6ca16d4f5d00e0820558f06a4d934.jpg)

![2b57394636248f402d0499f77493e70902238cd01791c50e83879afd736844a4.jpg](../iclr_results/2820_Remove Symmetries to Control Model Expressivity and Improve Optimization/tables/2b57394636248f402d0499f77493e70902238cd01791c50e83879afd736844a4.jpg)

![7472c0ba2188376bc2d478cdde04adfd2006fc06a90f76fde2d32b62470feca4.jpg](../iclr_results/2820_Remove Symmetries to Control Model Expressivity and Improve Optimization/tables/7472c0ba2188376bc2d478cdde04adfd2006fc06a90f76fde2d32b62470feca4.jpg)

## Round and Round We Go! What makes Rotary Positional Encodings useful?


### Images

![207d08cc95d41382dad9b471ea5c354d5ff1e74fa4988f1d58c63b7be72d86b6.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/207d08cc95d41382dad9b471ea5c354d5ff1e74fa4988f1d58c63b7be72d86b6.jpg)

![3437570b9f90bfd3efe7b762512561963095c47ed2397cb1c607e20a7c2f9d29.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/3437570b9f90bfd3efe7b762512561963095c47ed2397cb1c607e20a7c2f9d29.jpg)

![3548552bc38c930b6efbc13f48396a2964cea257332a77a865074d38423242ab.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/3548552bc38c930b6efbc13f48396a2964cea257332a77a865074d38423242ab.jpg)

![58b7f70984f9fc4fd8c9c95ce1089a423d267cb8c20c40b0b250f15dee66223c.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/58b7f70984f9fc4fd8c9c95ce1089a423d267cb8c20c40b0b250f15dee66223c.jpg)

![5dafa788f938a8aebbefb705dfdb3987cb8a982bd35be3e9108eb022fd89a85e.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/5dafa788f938a8aebbefb705dfdb3987cb8a982bd35be3e9108eb022fd89a85e.jpg)

![5ffd4ad18410f0cd6da2983e83e47fc2b1f0b7413966b61d77e7f9d1629e5bf7.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/5ffd4ad18410f0cd6da2983e83e47fc2b1f0b7413966b61d77e7f9d1629e5bf7.jpg)

![661e3b7071906173dc7f6300a63f943726a4774e8f7d506f4cd82f39b492a758.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/661e3b7071906173dc7f6300a63f943726a4774e8f7d506f4cd82f39b492a758.jpg)

![683b9b831b58607a076ec1a74114f004bd2e5cabb59b18cd20c3656c3fe9196b.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/683b9b831b58607a076ec1a74114f004bd2e5cabb59b18cd20c3656c3fe9196b.jpg)

![72612cb2662fefb795fe315cb6b2066fc411c87486dd86ccc2d5e6db7f924840.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/72612cb2662fefb795fe315cb6b2066fc411c87486dd86ccc2d5e6db7f924840.jpg)

![84360e5f36420f9a395bcca9894c44f1ba0d3e4894e35665784e0e58720e2a04.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/84360e5f36420f9a395bcca9894c44f1ba0d3e4894e35665784e0e58720e2a04.jpg)

![868a56e80ba6f0b8f8a782f47a991a14417441cf7913751d61f383215ddb9d75.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/868a56e80ba6f0b8f8a782f47a991a14417441cf7913751d61f383215ddb9d75.jpg)

![8e006b8f516c6d52bf214ee90325dcb29b6fdcb39e90ee72fcdf33698fd946a0.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/8e006b8f516c6d52bf214ee90325dcb29b6fdcb39e90ee72fcdf33698fd946a0.jpg)

![91afe2f52ab4889dc0ce3098e807bd40bc086774935162b1b07e1b48feaf0a43.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/91afe2f52ab4889dc0ce3098e807bd40bc086774935162b1b07e1b48feaf0a43.jpg)

![9a0a9194721a049557422cdf5bf821a0f6fef36485cbdfc785f56cadfa100a51.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/9a0a9194721a049557422cdf5bf821a0f6fef36485cbdfc785f56cadfa100a51.jpg)

![a17deceec161af1353ab9212272adc76c22e46a878d73c75d1ff7d35121f42d7.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/a17deceec161af1353ab9212272adc76c22e46a878d73c75d1ff7d35121f42d7.jpg)

![ae7705677240867019605b9223d44132f6303ebfab3c4f79098f748f2d4ea871.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/ae7705677240867019605b9223d44132f6303ebfab3c4f79098f748f2d4ea871.jpg)

![b41e4f3f1fdcecfcc16b95dee17ccc36016c7194a782724db499f120ac9573c6.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/b41e4f3f1fdcecfcc16b95dee17ccc36016c7194a782724db499f120ac9573c6.jpg)

![baaf6dad4531a62e842c739293d6ed28775f52f8603881d109ec464e1b84d7cd.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/baaf6dad4531a62e842c739293d6ed28775f52f8603881d109ec464e1b84d7cd.jpg)

![bb998496bcaa1ce8acabac491d05b42402ae132081378a2222d23019df3f0f9a.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/bb998496bcaa1ce8acabac491d05b42402ae132081378a2222d23019df3f0f9a.jpg)

![bc6f356dd0a97fd8fe3874c01b4ed76d09f5f89d5660bd38f7092e4e7421dd92.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/bc6f356dd0a97fd8fe3874c01b4ed76d09f5f89d5660bd38f7092e4e7421dd92.jpg)

![c0aa81b2035d40d9639bb438b88757bcdeb7362f2fb23a2c9fb7e90578625514.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/c0aa81b2035d40d9639bb438b88757bcdeb7362f2fb23a2c9fb7e90578625514.jpg)

![c43d34ee086c655e75bb999ea29d6f61dfbce493cb9eedc7f0b97589e2a9bd69.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/c43d34ee086c655e75bb999ea29d6f61dfbce493cb9eedc7f0b97589e2a9bd69.jpg)

![da6c4acdd5e33a9dcf839212ea40815c02a1a6b1a2c02b0480fa2f6a6bddedd1.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/da6c4acdd5e33a9dcf839212ea40815c02a1a6b1a2c02b0480fa2f6a6bddedd1.jpg)

![db7b5e827ce28e2be1fcba6083559ac27553fb72d66862434b6505099f2a6111.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/db7b5e827ce28e2be1fcba6083559ac27553fb72d66862434b6505099f2a6111.jpg)

![f25d5ba67d000e29defcb3baa4373cbfccc8fd8d3e0a405fff1bddd8b5d3fc0d.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/f25d5ba67d000e29defcb3baa4373cbfccc8fd8d3e0a405fff1bddd8b5d3fc0d.jpg)

![f47c37f96971233a0d711be672a314fe6736b7bc1599ae6d0a9bc691df70ff89.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/images/f47c37f96971233a0d711be672a314fe6736b7bc1599ae6d0a9bc691df70ff89.jpg)

### Tables

![247d810450a437c7ee5d16ed24cf16c5f6f8af3c606075655b44d642a5aead0a.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/tables/247d810450a437c7ee5d16ed24cf16c5f6f8af3c606075655b44d642a5aead0a.jpg)

![30e12a8810dec728f7afdf4ca1c2081a00219deac9f50e430ae521d049eecf9c.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/tables/30e12a8810dec728f7afdf4ca1c2081a00219deac9f50e430ae521d049eecf9c.jpg)

![6d4875621827fbecf4e9b1ca9fb87d141f1e108a8228096c69f9b4b0bb4c2bd9.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/tables/6d4875621827fbecf4e9b1ca9fb87d141f1e108a8228096c69f9b4b0bb4c2bd9.jpg)

![7b0f8b1e54ce395ab62b21dfb79106d092193052b24a963c5a55c6d88126af57.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/tables/7b0f8b1e54ce395ab62b21dfb79106d092193052b24a963c5a55c6d88126af57.jpg)

![ab00c7b336331bd930fe34d62a033e50e3c7ba26bc1056b9a345725315c33477.jpg](../iclr_results/2821_Round and Round We Go! What makes Rotary Positional Encodings useful_/tables/ab00c7b336331bd930fe34d62a033e50e3c7ba26bc1056b9a345725315c33477.jpg)

## Advancing Mathematical Reasoning in Language Models: The Impact of Problem-Solving Data, Data Synthesis Methods, and Training Stages


### Images

![05103a08ee667a72bff14ea1cd8dc715c6b080ad8f72ac33ec78c386c50a2a6e.jpg](../iclr_results/2822_Advancing Mathematical Reasoning in Language Models_ The Impact of Problem-Solving Data, Data Synthe/images/05103a08ee667a72bff14ea1cd8dc715c6b080ad8f72ac33ec78c386c50a2a6e.jpg)

![12378f3f5eff6c630374dddd5e60cc648799ff33986e37fe40878a5ae71f8d33.jpg](../iclr_results/2822_Advancing Mathematical Reasoning in Language Models_ The Impact of Problem-Solving Data, Data Synthe/images/12378f3f5eff6c630374dddd5e60cc648799ff33986e37fe40878a5ae71f8d33.jpg)

![4e24df852a8ddf27968cf16d5ac6d4841a85514548f78dd2ae712cdaddbd0a12.jpg](../iclr_results/2822_Advancing Mathematical Reasoning in Language Models_ The Impact of Problem-Solving Data, Data Synthe/images/4e24df852a8ddf27968cf16d5ac6d4841a85514548f78dd2ae712cdaddbd0a12.jpg)

![66fdc8096bd24e0e7240eb42a734e2e607b8b37c19850a799bd234b59d0c55ed.jpg](../iclr_results/2822_Advancing Mathematical Reasoning in Language Models_ The Impact of Problem-Solving Data, Data Synthe/images/66fdc8096bd24e0e7240eb42a734e2e607b8b37c19850a799bd234b59d0c55ed.jpg)

![dbfdc731cd0f5d6baef99611fd7a5a05e89d0cd2c516a063ef0244473c2c062f.jpg](../iclr_results/2822_Advancing Mathematical Reasoning in Language Models_ The Impact of Problem-Solving Data, Data Synthe/images/dbfdc731cd0f5d6baef99611fd7a5a05e89d0cd2c516a063ef0244473c2c062f.jpg)

![f96ed427ec41f61d45726b49e2fd96d848f742285117d8f0d5b875948ccdd32f.jpg](../iclr_results/2822_Advancing Mathematical Reasoning in Language Models_ The Impact of Problem-Solving Data, Data Synthe/images/f96ed427ec41f61d45726b49e2fd96d848f742285117d8f0d5b875948ccdd32f.jpg)

### Tables

![0192992b691d9462087e086ed8f1b9b5a9cb49cdfb20ce3906b16f8c21a489ce.jpg](../iclr_results/2822_Advancing Mathematical Reasoning in Language Models_ The Impact of Problem-Solving Data, Data Synthe/tables/0192992b691d9462087e086ed8f1b9b5a9cb49cdfb20ce3906b16f8c21a489ce.jpg)

![11d0e2b837a8192bc08db66deaa532abdc9bc397726e048787fbb91d871c2031.jpg](../iclr_results/2822_Advancing Mathematical Reasoning in Language Models_ The Impact of Problem-Solving Data, Data Synthe/tables/11d0e2b837a8192bc08db66deaa532abdc9bc397726e048787fbb91d871c2031.jpg)

![2bfd31f82a826f82086ab5abd01039f4f8b1afa83f1472472c299e68cde89935.jpg](../iclr_results/2822_Advancing Mathematical Reasoning in Language Models_ The Impact of Problem-Solving Data, Data Synthe/tables/2bfd31f82a826f82086ab5abd01039f4f8b1afa83f1472472c299e68cde89935.jpg)

![31e761a6d2bb679f61d0f237f1abb93ad817f3b133a766c69705a1a8bf406be5.jpg](../iclr_results/2822_Advancing Mathematical Reasoning in Language Models_ The Impact of Problem-Solving Data, Data Synthe/tables/31e761a6d2bb679f61d0f237f1abb93ad817f3b133a766c69705a1a8bf406be5.jpg)

![b85c87f7fc02c41f0c1be8faf7d22b8555221e4b4666a8c13b1f807223616b56.jpg](../iclr_results/2822_Advancing Mathematical Reasoning in Language Models_ The Impact of Problem-Solving Data, Data Synthe/tables/b85c87f7fc02c41f0c1be8faf7d22b8555221e4b4666a8c13b1f807223616b56.jpg)

![bc8e3f815449f1601276e4f6a00b9e663b7fd52191665b678e25eb74c9874b66.jpg](../iclr_results/2822_Advancing Mathematical Reasoning in Language Models_ The Impact of Problem-Solving Data, Data Synthe/tables/bc8e3f815449f1601276e4f6a00b9e663b7fd52191665b678e25eb74c9874b66.jpg)

![dbfa88d3e663023ba0eee678b9c3830022490fc30562730f299554812f0ea15d.jpg](../iclr_results/2822_Advancing Mathematical Reasoning in Language Models_ The Impact of Problem-Solving Data, Data Synthe/tables/dbfa88d3e663023ba0eee678b9c3830022490fc30562730f299554812f0ea15d.jpg)

![ec4388a5ab702cf78a32427d42fe160c4a2b29de3c76d2a7f2fa11e5189733dc.jpg](../iclr_results/2822_Advancing Mathematical Reasoning in Language Models_ The Impact of Problem-Solving Data, Data Synthe/tables/ec4388a5ab702cf78a32427d42fe160c4a2b29de3c76d2a7f2fa11e5189733dc.jpg)

## Learning Gain Map for Inverse Tone Mapping


### Images

![022eef2dbacde6089c58911339bfcb8ec5083f567f6176a2978b3b255cb406ed.jpg](../iclr_results/2823_Learning Gain Map for Inverse Tone Mapping/images/022eef2dbacde6089c58911339bfcb8ec5083f567f6176a2978b3b255cb406ed.jpg)

![0a6d55db02a8f1029283b389a2a661ab9aa9a6809e9822a3801271bf3ec71d01.jpg](../iclr_results/2823_Learning Gain Map for Inverse Tone Mapping/images/0a6d55db02a8f1029283b389a2a661ab9aa9a6809e9822a3801271bf3ec71d01.jpg)

![100dece005c91f5890038f28297cd46abb91e6531dff809281963ec2a3ef3b70.jpg](../iclr_results/2823_Learning Gain Map for Inverse Tone Mapping/images/100dece005c91f5890038f28297cd46abb91e6531dff809281963ec2a3ef3b70.jpg)

![43ec908db3eb236f4a02265de39c1d2b2b4c71a9fd9340874913fa617ca49b1b.jpg](../iclr_results/2823_Learning Gain Map for Inverse Tone Mapping/images/43ec908db3eb236f4a02265de39c1d2b2b4c71a9fd9340874913fa617ca49b1b.jpg)

![61ab69350cb543045327b9fd865852113829c3e98a60d085e0b9d9dc05674c0d.jpg](../iclr_results/2823_Learning Gain Map for Inverse Tone Mapping/images/61ab69350cb543045327b9fd865852113829c3e98a60d085e0b9d9dc05674c0d.jpg)

![7fba3e0c744cc6b66c01be3bc7ef2db58e6cdc634b409084f5b94d7f8843043d.jpg](../iclr_results/2823_Learning Gain Map for Inverse Tone Mapping/images/7fba3e0c744cc6b66c01be3bc7ef2db58e6cdc634b409084f5b94d7f8843043d.jpg)

![9194236b716207c593a1cbbd3b714f4bdaf3194182ab0581fa933f2872dd82b1.jpg](../iclr_results/2823_Learning Gain Map for Inverse Tone Mapping/images/9194236b716207c593a1cbbd3b714f4bdaf3194182ab0581fa933f2872dd82b1.jpg)

![91b51974a75ec6db664c02f403d9143f56124017201bcaabb8c498f229fdd877.jpg](../iclr_results/2823_Learning Gain Map for Inverse Tone Mapping/images/91b51974a75ec6db664c02f403d9143f56124017201bcaabb8c498f229fdd877.jpg)

![9b4cb0f395728311ff4964e9a2fd798086ccc13a1047ae635a048c347088d287.jpg](../iclr_results/2823_Learning Gain Map for Inverse Tone Mapping/images/9b4cb0f395728311ff4964e9a2fd798086ccc13a1047ae635a048c347088d287.jpg)

![a6219fc4ce62ef778ebfd9cfb4b6b5480a4fecc65cc036e89fcaca06ce7b581f.jpg](../iclr_results/2823_Learning Gain Map for Inverse Tone Mapping/images/a6219fc4ce62ef778ebfd9cfb4b6b5480a4fecc65cc036e89fcaca06ce7b581f.jpg)

![a7ce9ed2888931d4d122d4d2f1ddecab2ffca3159eac35ca3c5129872509149e.jpg](../iclr_results/2823_Learning Gain Map for Inverse Tone Mapping/images/a7ce9ed2888931d4d122d4d2f1ddecab2ffca3159eac35ca3c5129872509149e.jpg)

![a8d644897b2f2b69a6495ccacac61c561f28500a99be81bed9e5bc119034169a.jpg](../iclr_results/2823_Learning Gain Map for Inverse Tone Mapping/images/a8d644897b2f2b69a6495ccacac61c561f28500a99be81bed9e5bc119034169a.jpg)

![c8ab4365c9122c9fb833862bfc19c9d0831809fa693aa237d6ae564ecb2ddb3f.jpg](../iclr_results/2823_Learning Gain Map for Inverse Tone Mapping/images/c8ab4365c9122c9fb833862bfc19c9d0831809fa693aa237d6ae564ecb2ddb3f.jpg)

![d06a3b01f26458ba00df3260353a547fe8ce578bef53dacbd27ed820247092e2.jpg](../iclr_results/2823_Learning Gain Map for Inverse Tone Mapping/images/d06a3b01f26458ba00df3260353a547fe8ce578bef53dacbd27ed820247092e2.jpg)

![d3de66143376644a7a599bb7705dc1270e68c84431891ac659786cf37c072585.jpg](../iclr_results/2823_Learning Gain Map for Inverse Tone Mapping/images/d3de66143376644a7a599bb7705dc1270e68c84431891ac659786cf37c072585.jpg)

### Tables

![3bac0a71bb8a00c0305ccdf13413e10279fec97f73f486de6b9169698d7d2dbf.jpg](../iclr_results/2823_Learning Gain Map for Inverse Tone Mapping/tables/3bac0a71bb8a00c0305ccdf13413e10279fec97f73f486de6b9169698d7d2dbf.jpg)

![603e65edba1b13a86e16b6fadd381e94c42a6a5e2dd8f8f616551d0bb953c353.jpg](../iclr_results/2823_Learning Gain Map for Inverse Tone Mapping/tables/603e65edba1b13a86e16b6fadd381e94c42a6a5e2dd8f8f616551d0bb953c353.jpg)

![7b75e55742570e5cbf9f680644bc8ff52584d445269cd7116d71543db1b9e713.jpg](../iclr_results/2823_Learning Gain Map for Inverse Tone Mapping/tables/7b75e55742570e5cbf9f680644bc8ff52584d445269cd7116d71543db1b9e713.jpg)

![8e87d296c594df1ca96c159b3008ce482973c8a4668f6d75dee0519abd42dfeb.jpg](../iclr_results/2823_Learning Gain Map for Inverse Tone Mapping/tables/8e87d296c594df1ca96c159b3008ce482973c8a4668f6d75dee0519abd42dfeb.jpg)

![9d78a0b42785a32379c0106c1ac90df707a2823dd3e862a509fded33c9fcd0d2.jpg](../iclr_results/2823_Learning Gain Map for Inverse Tone Mapping/tables/9d78a0b42785a32379c0106c1ac90df707a2823dd3e862a509fded33c9fcd0d2.jpg)

![d7a36de5bdf068e255fe515a440d94b2e6109762b6ae2980d983392771c949d0.jpg](../iclr_results/2823_Learning Gain Map for Inverse Tone Mapping/tables/d7a36de5bdf068e255fe515a440d94b2e6109762b6ae2980d983392771c949d0.jpg)

![e5e1041014d5bcd6dd3844c1700d440af45fe24f591b5525ef118bac8dfa2fe4.jpg](../iclr_results/2823_Learning Gain Map for Inverse Tone Mapping/tables/e5e1041014d5bcd6dd3844c1700d440af45fe24f591b5525ef118bac8dfa2fe4.jpg)

![fbaef55a321edb3c88cc47fb65abeae8ca9f958c35a2af838ed54c2dc5100d98.jpg](../iclr_results/2823_Learning Gain Map for Inverse Tone Mapping/tables/fbaef55a321edb3c88cc47fb65abeae8ca9f958c35a2af838ed54c2dc5100d98.jpg)

## Robust Simulation-Based Inference under Missing Data via Neural Processes


### Images

![1a2dff5605fe88fd7ed3878b0bdde1c6e5ca9c4042571d1959093b0745fbd440.jpg](../iclr_results/2824_Robust Simulation-Based Inference under Missing Data via Neural Processes/images/1a2dff5605fe88fd7ed3878b0bdde1c6e5ca9c4042571d1959093b0745fbd440.jpg)

![29b351b329e1700942f6868f74e85424ebaaa111f9be962f2f54cc6a8bcbc393.jpg](../iclr_results/2824_Robust Simulation-Based Inference under Missing Data via Neural Processes/images/29b351b329e1700942f6868f74e85424ebaaa111f9be962f2f54cc6a8bcbc393.jpg)

![2d9afe1488edf193ffc2ee3d0734b50d3da7acfadef78d75c2467b7ec197263b.jpg](../iclr_results/2824_Robust Simulation-Based Inference under Missing Data via Neural Processes/images/2d9afe1488edf193ffc2ee3d0734b50d3da7acfadef78d75c2467b7ec197263b.jpg)

![39e72c59fa958b2ee616210aa72f9cff90c8548c5b1cb85b1fee9ee1715282c4.jpg](../iclr_results/2824_Robust Simulation-Based Inference under Missing Data via Neural Processes/images/39e72c59fa958b2ee616210aa72f9cff90c8548c5b1cb85b1fee9ee1715282c4.jpg)

![41b2e2d7130c708e6a83058aa2fa1770fd73ed9bfa3b4adb60d2517c036f97d3.jpg](../iclr_results/2824_Robust Simulation-Based Inference under Missing Data via Neural Processes/images/41b2e2d7130c708e6a83058aa2fa1770fd73ed9bfa3b4adb60d2517c036f97d3.jpg)

![61b20f003a82fb4cd3b13b76e72829a6a15e77b7fc2a5bb1005f0216e1a4bb13.jpg](../iclr_results/2824_Robust Simulation-Based Inference under Missing Data via Neural Processes/images/61b20f003a82fb4cd3b13b76e72829a6a15e77b7fc2a5bb1005f0216e1a4bb13.jpg)

![da99721fc3a5fa06a485fa870526b03d4390eab5cd4d13c06682c773e39fbd75.jpg](../iclr_results/2824_Robust Simulation-Based Inference under Missing Data via Neural Processes/images/da99721fc3a5fa06a485fa870526b03d4390eab5cd4d13c06682c773e39fbd75.jpg)

![f26354530c2dfa48f217b038d9da5aad381272493570e30d95fca71c4d7819a7.jpg](../iclr_results/2824_Robust Simulation-Based Inference under Missing Data via Neural Processes/images/f26354530c2dfa48f217b038d9da5aad381272493570e30d95fca71c4d7819a7.jpg)

### Tables

![0cdf2fea95eee29f406f893c79d9724245398bfb465c70301f8a047cf350ae1f.jpg](../iclr_results/2824_Robust Simulation-Based Inference under Missing Data via Neural Processes/tables/0cdf2fea95eee29f406f893c79d9724245398bfb465c70301f8a047cf350ae1f.jpg)

![3e2c66951736a3eaeafe1cd9c7a00448a8afb04ff3ecc724bff3cb9af1f9efae.jpg](../iclr_results/2824_Robust Simulation-Based Inference under Missing Data via Neural Processes/tables/3e2c66951736a3eaeafe1cd9c7a00448a8afb04ff3ecc724bff3cb9af1f9efae.jpg)

![41a8a11bc22a589af97abba9aa89e815d61d7c4753f58ae1b1f04b363030837a.jpg](../iclr_results/2824_Robust Simulation-Based Inference under Missing Data via Neural Processes/tables/41a8a11bc22a589af97abba9aa89e815d61d7c4753f58ae1b1f04b363030837a.jpg)

![44f3fc8e4153beebabb0db774e483dce8ba5fe930d1519feac0537db9c08ecd4.jpg](../iclr_results/2824_Robust Simulation-Based Inference under Missing Data via Neural Processes/tables/44f3fc8e4153beebabb0db774e483dce8ba5fe930d1519feac0537db9c08ecd4.jpg)

![4a57f3ff071df0d6bc42ceae963f345c0ae24666e8836d4e588b5534e9ed4016.jpg](../iclr_results/2824_Robust Simulation-Based Inference under Missing Data via Neural Processes/tables/4a57f3ff071df0d6bc42ceae963f345c0ae24666e8836d4e588b5534e9ed4016.jpg)

![58ac79f75cf344eae338f06c5305115c877efd53870b27f65c56531a15c4d5d4.jpg](../iclr_results/2824_Robust Simulation-Based Inference under Missing Data via Neural Processes/tables/58ac79f75cf344eae338f06c5305115c877efd53870b27f65c56531a15c4d5d4.jpg)

![594726333dc048f408021232b9eed599890c0c37dfbdcaef4fed43b4cbd81d5f.jpg](../iclr_results/2824_Robust Simulation-Based Inference under Missing Data via Neural Processes/tables/594726333dc048f408021232b9eed599890c0c37dfbdcaef4fed43b4cbd81d5f.jpg)

![85dfe1491dcf0317a3f51245c623769b6308de36a220326a06dee6943cc9cfe8.jpg](../iclr_results/2824_Robust Simulation-Based Inference under Missing Data via Neural Processes/tables/85dfe1491dcf0317a3f51245c623769b6308de36a220326a06dee6943cc9cfe8.jpg)

![a76bfc031ddeff185a82a7235d5b66016d106bce5878e273d405e2e1f435bdaf.jpg](../iclr_results/2824_Robust Simulation-Based Inference under Missing Data via Neural Processes/tables/a76bfc031ddeff185a82a7235d5b66016d106bce5878e273d405e2e1f435bdaf.jpg)

![cf2b810dc6284ec9ed3c8020b0a2aed769a18571ccd5f1a80ce0d4e899d94e8d.jpg](../iclr_results/2824_Robust Simulation-Based Inference under Missing Data via Neural Processes/tables/cf2b810dc6284ec9ed3c8020b0a2aed769a18571ccd5f1a80ce0d4e899d94e8d.jpg)

## Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings


### Images

![6963d0997ef186fa4cad533b2ff91fe95783e7d3e7dcbd3c30154c570097b519.jpg](../iclr_results/2825_Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings/images/6963d0997ef186fa4cad533b2ff91fe95783e7d3e7dcbd3c30154c570097b519.jpg)

![6bbb101647fe46e08861a5a6a47c8f2de35c26804d4c4fcb3d73181c07e51814.jpg](../iclr_results/2825_Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings/images/6bbb101647fe46e08861a5a6a47c8f2de35c26804d4c4fcb3d73181c07e51814.jpg)

### Tables

![26332a19ea23532acedfffab7d12d5e1ac9da8944b7f34e5951ef37f060853a2.jpg](../iclr_results/2825_Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings/tables/26332a19ea23532acedfffab7d12d5e1ac9da8944b7f34e5951ef37f060853a2.jpg)

![3f1c51e1122ce87f8a7f29af5a7fe7b76327eadb6ad58ee23f5903e6df78143e.jpg](../iclr_results/2825_Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings/tables/3f1c51e1122ce87f8a7f29af5a7fe7b76327eadb6ad58ee23f5903e6df78143e.jpg)

![5ed8626b801653ae275f7816b7d206408439df6c5f7378cbaef0f56febfa4ec6.jpg](../iclr_results/2825_Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings/tables/5ed8626b801653ae275f7816b7d206408439df6c5f7378cbaef0f56febfa4ec6.jpg)

![8155a8d51f23aed51ce3a4db9d4cbdce7618410eccb4bcbd36568194a15c8468.jpg](../iclr_results/2825_Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings/tables/8155a8d51f23aed51ce3a4db9d4cbdce7618410eccb4bcbd36568194a15c8468.jpg)

![bb39068c93bc5ab79ad1e7dcb18ef33125df09f86be9249faf63ab076197ab70.jpg](../iclr_results/2825_Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings/tables/bb39068c93bc5ab79ad1e7dcb18ef33125df09f86be9249faf63ab076197ab70.jpg)

![d8bc7082d81c95e6f585682d8e86a9889b7a22b25108399b1147dad484177b55.jpg](../iclr_results/2825_Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings/tables/d8bc7082d81c95e6f585682d8e86a9889b7a22b25108399b1147dad484177b55.jpg)

![f41eb9e12a9c0c30449305baef3688cb6a6ba0692837438e798b662265f19138.jpg](../iclr_results/2825_Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings/tables/f41eb9e12a9c0c30449305baef3688cb6a6ba0692837438e798b662265f19138.jpg)

## MuseGNN: Forming Scalable, Convergent GNN Layers that Minimize a Sampling-Based Energy


### Images

![0f95ed8d330013271220358e758f18527aa17bf313dbc39d33be31c59b630f21.jpg](../iclr_results/2826_MuseGNN_ Forming Scalable, Convergent GNN Layers that Minimize a Sampling-Based Energy/images/0f95ed8d330013271220358e758f18527aa17bf313dbc39d33be31c59b630f21.jpg)

![17b8d993548e90995bb33f43cbb93c88dff04b587b18ae08d29051c6974fde35.jpg](../iclr_results/2826_MuseGNN_ Forming Scalable, Convergent GNN Layers that Minimize a Sampling-Based Energy/images/17b8d993548e90995bb33f43cbb93c88dff04b587b18ae08d29051c6974fde35.jpg)

![7e85f81ef0a39dbb5f7a1d4c357f030c4d5ec3f9d38236daae453f5c1878ac10.jpg](../iclr_results/2826_MuseGNN_ Forming Scalable, Convergent GNN Layers that Minimize a Sampling-Based Energy/images/7e85f81ef0a39dbb5f7a1d4c357f030c4d5ec3f9d38236daae453f5c1878ac10.jpg)

### Tables

![09691f540a25343409e696f4e4db0b6c3f30e53be10dc6d35dcda5b373da73b5.jpg](../iclr_results/2826_MuseGNN_ Forming Scalable, Convergent GNN Layers that Minimize a Sampling-Based Energy/tables/09691f540a25343409e696f4e4db0b6c3f30e53be10dc6d35dcda5b373da73b5.jpg)

![3a566ccd0ef5116a559318e06e5d6a92701e3e05ded4b830322e54ea0bc480c4.jpg](../iclr_results/2826_MuseGNN_ Forming Scalable, Convergent GNN Layers that Minimize a Sampling-Based Energy/tables/3a566ccd0ef5116a559318e06e5d6a92701e3e05ded4b830322e54ea0bc480c4.jpg)

![42708837e49614896f0f9319f109cffc183f7fb40b16b8e759840acbf0f4cb1f.jpg](../iclr_results/2826_MuseGNN_ Forming Scalable, Convergent GNN Layers that Minimize a Sampling-Based Energy/tables/42708837e49614896f0f9319f109cffc183f7fb40b16b8e759840acbf0f4cb1f.jpg)

![73ac6b8f26849a675f8cb066add3e1ca5e42c68782ea5307ec2b8b1e4886c2a4.jpg](../iclr_results/2826_MuseGNN_ Forming Scalable, Convergent GNN Layers that Minimize a Sampling-Based Energy/tables/73ac6b8f26849a675f8cb066add3e1ca5e42c68782ea5307ec2b8b1e4886c2a4.jpg)

![a6b941f3bb5e6888c2cc9f8a9172f9303e5fc61abf2497f02cb1fb810cd1f1ac.jpg](../iclr_results/2826_MuseGNN_ Forming Scalable, Convergent GNN Layers that Minimize a Sampling-Based Energy/tables/a6b941f3bb5e6888c2cc9f8a9172f9303e5fc61abf2497f02cb1fb810cd1f1ac.jpg)

![e37e93ff1625c7c52e185579b81062c10d684bd06e219d0c7db2ae09643a3ce8.jpg](../iclr_results/2826_MuseGNN_ Forming Scalable, Convergent GNN Layers that Minimize a Sampling-Based Energy/tables/e37e93ff1625c7c52e185579b81062c10d684bd06e219d0c7db2ae09643a3ce8.jpg)

![fffe414378735875d59b14cce13bf44b100669642b5705e17fbda4f6b36dd821.jpg](../iclr_results/2826_MuseGNN_ Forming Scalable, Convergent GNN Layers that Minimize a Sampling-Based Energy/tables/fffe414378735875d59b14cce13bf44b100669642b5705e17fbda4f6b36dd821.jpg)

## The Crystal Ball Hypothesis in diffusion models: Anticipating object positions from initial noise


### Images

![107b58688104d9553352267dd3925892bfddc70e7f5cb044c447f4fa6ba7a4ce.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/images/107b58688104d9553352267dd3925892bfddc70e7f5cb044c447f4fa6ba7a4ce.jpg)

![50f81cf7119adeccbc47ed542befcacfba0e126ef1be2bf3a0b7f36b94252af4.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/images/50f81cf7119adeccbc47ed542befcacfba0e126ef1be2bf3a0b7f36b94252af4.jpg)

![5d23264942b94007710acfacc12bd41b3f1d3cbb78a888b5c7670b2892dddd04.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/images/5d23264942b94007710acfacc12bd41b3f1d3cbb78a888b5c7670b2892dddd04.jpg)

![730122315e176fb1ce4a678c85f4efb1c3dba861e1c85dea9a2e94d5670612f3.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/images/730122315e176fb1ce4a678c85f4efb1c3dba861e1c85dea9a2e94d5670612f3.jpg)

![73ef3108ac7edce0f14ee08b35413c8c42b5574b6e946510f89c2ed3d0de209c.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/images/73ef3108ac7edce0f14ee08b35413c8c42b5574b6e946510f89c2ed3d0de209c.jpg)

![7cae5ffa526159a8724a23e0759a68c0216a0d84d79d05fc2cc8635a4c274a84.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/images/7cae5ffa526159a8724a23e0759a68c0216a0d84d79d05fc2cc8635a4c274a84.jpg)

![7ccb7b694c7b900228f5ccfd59bb5259def2baa0fadadaa1d6730a56cb951068.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/images/7ccb7b694c7b900228f5ccfd59bb5259def2baa0fadadaa1d6730a56cb951068.jpg)

![87f5385f0533cb50248de68502124dd202e42120eecd5271832d6125ff28e3d7.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/images/87f5385f0533cb50248de68502124dd202e42120eecd5271832d6125ff28e3d7.jpg)

![99aa907f574998c840c6a0e346571b46e91bc456e3029ae5802baaca8fe1d23f.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/images/99aa907f574998c840c6a0e346571b46e91bc456e3029ae5802baaca8fe1d23f.jpg)

![a2027ed6ea6bfef0791531a4da2b284a0466c432e80adbe323ebdabaa72ed653.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/images/a2027ed6ea6bfef0791531a4da2b284a0466c432e80adbe323ebdabaa72ed653.jpg)

![be50794a41f3c502ee4cf7e93e53d08a25ba03c40a67674d66cd3c7352954f5f.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/images/be50794a41f3c502ee4cf7e93e53d08a25ba03c40a67674d66cd3c7352954f5f.jpg)

![cb480f38c06302d3abc7962c415c3b68811de2386902fd7b66f85cca20766c44.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/images/cb480f38c06302d3abc7962c415c3b68811de2386902fd7b66f85cca20766c44.jpg)

![d92b1ad3f1156e8e02d7ed4e69d5934f12dab106b6e3c2a5bc037ed433d596fc.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/images/d92b1ad3f1156e8e02d7ed4e69d5934f12dab106b6e3c2a5bc037ed433d596fc.jpg)

![e36e3f108457f1543febdf703e3acd0df972ca8c38e4908f470addebe6307611.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/images/e36e3f108457f1543febdf703e3acd0df972ca8c38e4908f470addebe6307611.jpg)

### Tables

![1bf60afa8c6c1775960c2a63473ec74f08bb15d6ee866259767113c8a24a8498.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/tables/1bf60afa8c6c1775960c2a63473ec74f08bb15d6ee866259767113c8a24a8498.jpg)

![1f33f44c85a3f1ae22b762f58d64108f9a825c232f088ebcc160b0781cb4dac9.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/tables/1f33f44c85a3f1ae22b762f58d64108f9a825c232f088ebcc160b0781cb4dac9.jpg)

![38a776f2b45009f02df591f4d25e8c72ac2d1e33a4309acca3101d29a776003d.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/tables/38a776f2b45009f02df591f4d25e8c72ac2d1e33a4309acca3101d29a776003d.jpg)

![3a93bc658f114c6919b468ccc6dd4b3239d7b85a78cd7c3483afc5c88c205cf0.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/tables/3a93bc658f114c6919b468ccc6dd4b3239d7b85a78cd7c3483afc5c88c205cf0.jpg)

![3c20157257c5eff84e6f1414cf5144518c860a4a38e6b704d7e5b7f826a0255d.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/tables/3c20157257c5eff84e6f1414cf5144518c860a4a38e6b704d7e5b7f826a0255d.jpg)

![3db078a6f7642947b5e94aeab4b9d10f0c972461dde93d3c43010ffb6c58927c.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/tables/3db078a6f7642947b5e94aeab4b9d10f0c972461dde93d3c43010ffb6c58927c.jpg)

![4c1e56b32210cd62b3da062d91ecf60fd012ef5d6789853ce6c9a97fc4978d12.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/tables/4c1e56b32210cd62b3da062d91ecf60fd012ef5d6789853ce6c9a97fc4978d12.jpg)

![65b1d152cc1ac002e1d890065edaf303782d2083cb807a2564f33b08884cd40d.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/tables/65b1d152cc1ac002e1d890065edaf303782d2083cb807a2564f33b08884cd40d.jpg)

![97c35eeb1332259591df1f66b58c34d9672008bccbdb658dc52c7603ffd0ddda.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/tables/97c35eeb1332259591df1f66b58c34d9672008bccbdb658dc52c7603ffd0ddda.jpg)

![abc1056d7315edfd7e3dc805b7db09fcf6cc4e883d7923751991c77df8eeb9b7.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/tables/abc1056d7315edfd7e3dc805b7db09fcf6cc4e883d7923751991c77df8eeb9b7.jpg)

![ac759f386b31720b07181c9c5496e44fb96a48535c21c27fa2339d99c9bf98f4.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/tables/ac759f386b31720b07181c9c5496e44fb96a48535c21c27fa2339d99c9bf98f4.jpg)

![fb9ca5736962011139d39858d892c3f2a2bae67e0e5985be96916171f873c033.jpg](../iclr_results/2827_The Crystal Ball Hypothesis in diffusion models_ Anticipating object positions from initial noise/tables/fb9ca5736962011139d39858d892c3f2a2bae67e0e5985be96916171f873c033.jpg)

## Towards counterfactual fairness through auxiliary variables


### Images

![3c8b752553b25582f6f7490cb78fb6c3a0e4339600ead7b8b3ef3812402ce6ad.jpg](../iclr_results/2828_Towards counterfactual fairness through auxiliary variables/images/3c8b752553b25582f6f7490cb78fb6c3a0e4339600ead7b8b3ef3812402ce6ad.jpg)

![7154e6e17e41b2427bc010561ee038f9f68b11c10b5faafd017e0a15743bd12b.jpg](../iclr_results/2828_Towards counterfactual fairness through auxiliary variables/images/7154e6e17e41b2427bc010561ee038f9f68b11c10b5faafd017e0a15743bd12b.jpg)

![d50f0a4ba5f2aa75c6921f3684c6b66dd98b4bfa3743fcfcae786b5dfd37777d.jpg](../iclr_results/2828_Towards counterfactual fairness through auxiliary variables/images/d50f0a4ba5f2aa75c6921f3684c6b66dd98b4bfa3743fcfcae786b5dfd37777d.jpg)

### Tables

![0a1cc7b4ef8d1921b04d18a64cbe1ed648e2021a2023eb5b1b643abbfee5798e.jpg](../iclr_results/2828_Towards counterfactual fairness through auxiliary variables/tables/0a1cc7b4ef8d1921b04d18a64cbe1ed648e2021a2023eb5b1b643abbfee5798e.jpg)

![4472f89d7a4a6ed8743b8eeb467d57a20632fcfec56dba9b28149445a9ee5f09.jpg](../iclr_results/2828_Towards counterfactual fairness through auxiliary variables/tables/4472f89d7a4a6ed8743b8eeb467d57a20632fcfec56dba9b28149445a9ee5f09.jpg)

![fcb105b9a282f5fbf257ea399e790ed5174cfa2fc97b36f6e59e4357da0b988f.jpg](../iclr_results/2828_Towards counterfactual fairness through auxiliary variables/tables/fcb105b9a282f5fbf257ea399e790ed5174cfa2fc97b36f6e59e4357da0b988f.jpg)

## Geometry Image Diffusion: Fast and Data-Efficient Text-to-3D with Image-Based Surface Representation


### Images

![0fc36fc9b015e58c0e5b18e7153915f2bc3a6793b347940f9c309a21747c0953.jpg](../iclr_results/2829_Geometry Image Diffusion_ Fast and Data-Efficient Text-to-3D with Image-Based Surface Representation/images/0fc36fc9b015e58c0e5b18e7153915f2bc3a6793b347940f9c309a21747c0953.jpg)

![1a554c5bc8149cf700141c8a4e2ac9542ddc525bdee12a27825e883a520fe3f4.jpg](../iclr_results/2829_Geometry Image Diffusion_ Fast and Data-Efficient Text-to-3D with Image-Based Surface Representation/images/1a554c5bc8149cf700141c8a4e2ac9542ddc525bdee12a27825e883a520fe3f4.jpg)

![239e046968c3419de11dd7d9bd471f484485a9173a197c25907e1e50cdeb5720.jpg](../iclr_results/2829_Geometry Image Diffusion_ Fast and Data-Efficient Text-to-3D with Image-Based Surface Representation/images/239e046968c3419de11dd7d9bd471f484485a9173a197c25907e1e50cdeb5720.jpg)

![2a444071cf4d5f91b8be50c52b9c2938be8d5f47bedaae2ebb02ed96a1378a3b.jpg](../iclr_results/2829_Geometry Image Diffusion_ Fast and Data-Efficient Text-to-3D with Image-Based Surface Representation/images/2a444071cf4d5f91b8be50c52b9c2938be8d5f47bedaae2ebb02ed96a1378a3b.jpg)

![301e24e8cebd47bd49e5cbc86820615f07097c5c89d4a94ff3c44dd6bd1babfe.jpg](../iclr_results/2829_Geometry Image Diffusion_ Fast and Data-Efficient Text-to-3D with Image-Based Surface Representation/images/301e24e8cebd47bd49e5cbc86820615f07097c5c89d4a94ff3c44dd6bd1babfe.jpg)

![4888ac37743b8f999c20d4436ded17ee6a8e1a6246e70f4666a4a5a25c6fcacd.jpg](../iclr_results/2829_Geometry Image Diffusion_ Fast and Data-Efficient Text-to-3D with Image-Based Surface Representation/images/4888ac37743b8f999c20d4436ded17ee6a8e1a6246e70f4666a4a5a25c6fcacd.jpg)

![521b3b912b9d1d454cdef1f5315f929b0a21d0ebe0c6729848d8286ddb7b212c.jpg](../iclr_results/2829_Geometry Image Diffusion_ Fast and Data-Efficient Text-to-3D with Image-Based Surface Representation/images/521b3b912b9d1d454cdef1f5315f929b0a21d0ebe0c6729848d8286ddb7b212c.jpg)

![7e76ce8f7f80114be4d7279b601e42658cf1fa33f6c6acf765b029f9eefa0c3f.jpg](../iclr_results/2829_Geometry Image Diffusion_ Fast and Data-Efficient Text-to-3D with Image-Based Surface Representation/images/7e76ce8f7f80114be4d7279b601e42658cf1fa33f6c6acf765b029f9eefa0c3f.jpg)

![87f33989b6b64c3428fef2bfad852730205899dedf562db9d718fab8f6d6c9ec.jpg](../iclr_results/2829_Geometry Image Diffusion_ Fast and Data-Efficient Text-to-3D with Image-Based Surface Representation/images/87f33989b6b64c3428fef2bfad852730205899dedf562db9d718fab8f6d6c9ec.jpg)

![8f6177f2bad65eae011d83ce567a77d3673374ae80d3e4472dd84555ce565a04.jpg](../iclr_results/2829_Geometry Image Diffusion_ Fast and Data-Efficient Text-to-3D with Image-Based Surface Representation/images/8f6177f2bad65eae011d83ce567a77d3673374ae80d3e4472dd84555ce565a04.jpg)

![94559082febb7fa5f8eeeac4bd1b8f7892c02dc7214acc2ab9a9cea427260804.jpg](../iclr_results/2829_Geometry Image Diffusion_ Fast and Data-Efficient Text-to-3D with Image-Based Surface Representation/images/94559082febb7fa5f8eeeac4bd1b8f7892c02dc7214acc2ab9a9cea427260804.jpg)

![b7f89f1aa1aaa774197683b2efdcb4f88ffaafb30229aa540944d9fe061d63d5.jpg](../iclr_results/2829_Geometry Image Diffusion_ Fast and Data-Efficient Text-to-3D with Image-Based Surface Representation/images/b7f89f1aa1aaa774197683b2efdcb4f88ffaafb30229aa540944d9fe061d63d5.jpg)

![b9826c27b88fd46a3e8ded77e30e2f005519e255070e3315ee72d1daae689d4b.jpg](../iclr_results/2829_Geometry Image Diffusion_ Fast and Data-Efficient Text-to-3D with Image-Based Surface Representation/images/b9826c27b88fd46a3e8ded77e30e2f005519e255070e3315ee72d1daae689d4b.jpg)

![d7871c155e3faad9addc6faf5dd9b5f2beda1b82b663fba448b54dc948bbd817.jpg](../iclr_results/2829_Geometry Image Diffusion_ Fast and Data-Efficient Text-to-3D with Image-Based Surface Representation/images/d7871c155e3faad9addc6faf5dd9b5f2beda1b82b663fba448b54dc948bbd817.jpg)

![dda41771e9bc88be811253aa2f36cc918ad948c5127bc10e0b3b0800ab38f3b9.jpg](../iclr_results/2829_Geometry Image Diffusion_ Fast and Data-Efficient Text-to-3D with Image-Based Surface Representation/images/dda41771e9bc88be811253aa2f36cc918ad948c5127bc10e0b3b0800ab38f3b9.jpg)

![ee4a6f6fee23f8d426b1fc914fd3cc27ea8e87be194e76459e09fbc484783fde.jpg](../iclr_results/2829_Geometry Image Diffusion_ Fast and Data-Efficient Text-to-3D with Image-Based Surface Representation/images/ee4a6f6fee23f8d426b1fc914fd3cc27ea8e87be194e76459e09fbc484783fde.jpg)

### Tables

![abba256a910dca338f52a725dca6dd406077a16ae0f38f31e13141a0cf193bcf.jpg](../iclr_results/2829_Geometry Image Diffusion_ Fast and Data-Efficient Text-to-3D with Image-Based Surface Representation/tables/abba256a910dca338f52a725dca6dd406077a16ae0f38f31e13141a0cf193bcf.jpg)

![b0d3702acc189c0212bde1b0b63cfa846f11d78c1cdcf7fa2b2206872002abe4.jpg](../iclr_results/2829_Geometry Image Diffusion_ Fast and Data-Efficient Text-to-3D with Image-Based Surface Representation/tables/b0d3702acc189c0212bde1b0b63cfa846f11d78c1cdcf7fa2b2206872002abe4.jpg)

## Multi-agent cooperation through learning-aware policy gradients


### Images

![15aff67b95ec0b649b5e4c24b126db2020af877e4bc2f644f1abb95d6854bf09.jpg](../iclr_results/2831_Multi-agent cooperation through learning-aware policy gradients/images/15aff67b95ec0b649b5e4c24b126db2020af877e4bc2f644f1abb95d6854bf09.jpg)

![168ae8c71d1d8f94eb7436b5f3cf2bb6070232ff04ab498fc589fb56cd1c9982.jpg](../iclr_results/2831_Multi-agent cooperation through learning-aware policy gradients/images/168ae8c71d1d8f94eb7436b5f3cf2bb6070232ff04ab498fc589fb56cd1c9982.jpg)

![1ee0889b56831375099b6dfe9212b8b489e2bf5a73e23ccc8244d715a7e59ef6.jpg](../iclr_results/2831_Multi-agent cooperation through learning-aware policy gradients/images/1ee0889b56831375099b6dfe9212b8b489e2bf5a73e23ccc8244d715a7e59ef6.jpg)

![2aca250303ce839075c8e2a1d5809911fd2ee2d4bfd24780e033d6a7b5bef3d6.jpg](../iclr_results/2831_Multi-agent cooperation through learning-aware policy gradients/images/2aca250303ce839075c8e2a1d5809911fd2ee2d4bfd24780e033d6a7b5bef3d6.jpg)

![3a0829173941812e3727e184b149605ec546d293428991e66d1578c5a57836d1.jpg](../iclr_results/2831_Multi-agent cooperation through learning-aware policy gradients/images/3a0829173941812e3727e184b149605ec546d293428991e66d1578c5a57836d1.jpg)

![4d767555487648bddfd35c0022f27efd7e7d7095a2bce679682c64bc89d20f39.jpg](../iclr_results/2831_Multi-agent cooperation through learning-aware policy gradients/images/4d767555487648bddfd35c0022f27efd7e7d7095a2bce679682c64bc89d20f39.jpg)

![59102669020e35152239ebd16aec4d8da74e0439b4a231baf3ebfb3d381225e5.jpg](../iclr_results/2831_Multi-agent cooperation through learning-aware policy gradients/images/59102669020e35152239ebd16aec4d8da74e0439b4a231baf3ebfb3d381225e5.jpg)

![9e3bccf7ff67c38b9a80a91261d7635d75bb09723cb7a1721700d4066a0856ce.jpg](../iclr_results/2831_Multi-agent cooperation through learning-aware policy gradients/images/9e3bccf7ff67c38b9a80a91261d7635d75bb09723cb7a1721700d4066a0856ce.jpg)

![b443d4721d60e8d8c2c1abcdfad579f76b1a4934c9e624f85394707c1f649f4b.jpg](../iclr_results/2831_Multi-agent cooperation through learning-aware policy gradients/images/b443d4721d60e8d8c2c1abcdfad579f76b1a4934c9e624f85394707c1f649f4b.jpg)

![c2fcb640453b564f7ba4509540e1b488bb98c0f113a2a225db39c46da860947e.jpg](../iclr_results/2831_Multi-agent cooperation through learning-aware policy gradients/images/c2fcb640453b564f7ba4509540e1b488bb98c0f113a2a225db39c46da860947e.jpg)

![c74bb495a7f7803ffd85316b3ebe007deb777c990dc7e5adeb717acb03ab2f24.jpg](../iclr_results/2831_Multi-agent cooperation through learning-aware policy gradients/images/c74bb495a7f7803ffd85316b3ebe007deb777c990dc7e5adeb717acb03ab2f24.jpg)

![eeffd7ba2917d04f3701bc62bf09ccdf92148ac9569fbc82202e13619619d501.jpg](../iclr_results/2831_Multi-agent cooperation through learning-aware policy gradients/images/eeffd7ba2917d04f3701bc62bf09ccdf92148ac9569fbc82202e13619619d501.jpg)

![efc1bf8898ae8b00eb9d826be4ac26281a7ebec40e6066d6ec54802ee24280a0.jpg](../iclr_results/2831_Multi-agent cooperation through learning-aware policy gradients/images/efc1bf8898ae8b00eb9d826be4ac26281a7ebec40e6066d6ec54802ee24280a0.jpg)

![f36f89fcde3cff23706e254fd750af08a4f2cb699360a607c6344064c82acc5e.jpg](../iclr_results/2831_Multi-agent cooperation through learning-aware policy gradients/images/f36f89fcde3cff23706e254fd750af08a4f2cb699360a607c6344064c82acc5e.jpg)

![ff422268d88438068841553aac092b7e630008eb04644658cbfed7d2899aed2d.jpg](../iclr_results/2831_Multi-agent cooperation through learning-aware policy gradients/images/ff422268d88438068841553aac092b7e630008eb04644658cbfed7d2899aed2d.jpg)

### Tables

![0c4b74d1c7f9eb493f866fa07a40f7702fa58211b3886f3b885bb281724b2abe.jpg](../iclr_results/2831_Multi-agent cooperation through learning-aware policy gradients/tables/0c4b74d1c7f9eb493f866fa07a40f7702fa58211b3886f3b885bb281724b2abe.jpg)

![0f95fec26201ffaddea3322cf4fc26032892e57d6d173925f78574cb6c97d249.jpg](../iclr_results/2831_Multi-agent cooperation through learning-aware policy gradients/tables/0f95fec26201ffaddea3322cf4fc26032892e57d6d173925f78574cb6c97d249.jpg)

![12b198e8b935630709b551b91554e221924dda55001550010400449a739723f0.jpg](../iclr_results/2831_Multi-agent cooperation through learning-aware policy gradients/tables/12b198e8b935630709b551b91554e221924dda55001550010400449a739723f0.jpg)

![49d3c91ff9d9425032a7cf21ad719817f4d8c7063cf5149f69bd194a8d65b4b0.jpg](../iclr_results/2831_Multi-agent cooperation through learning-aware policy gradients/tables/49d3c91ff9d9425032a7cf21ad719817f4d8c7063cf5149f69bd194a8d65b4b0.jpg)

![521f552a39de34427658409d9754a56b28fb96b5f51f33a377f9b09355cc150c.jpg](../iclr_results/2831_Multi-agent cooperation through learning-aware policy gradients/tables/521f552a39de34427658409d9754a56b28fb96b5f51f33a377f9b09355cc150c.jpg)

![729b9416d368904e77ad16901613aab0fe34618341461a822cabae4d1185697f.jpg](../iclr_results/2831_Multi-agent cooperation through learning-aware policy gradients/tables/729b9416d368904e77ad16901613aab0fe34618341461a822cabae4d1185697f.jpg)

![a4f9db8dbebeb61b1fc53d71639e86e526e30d6a91991510a5e537632fda3a60.jpg](../iclr_results/2831_Multi-agent cooperation through learning-aware policy gradients/tables/a4f9db8dbebeb61b1fc53d71639e86e526e30d6a91991510a5e537632fda3a60.jpg)

![da42df96432c80900d9fc2ad360ac7d203fbc35a8ff544ec9f45d1e620b76a65.jpg](../iclr_results/2831_Multi-agent cooperation through learning-aware policy gradients/tables/da42df96432c80900d9fc2ad360ac7d203fbc35a8ff544ec9f45d1e620b76a65.jpg)

## StructRAG: Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Structurization


### Images

![0a73d4bfbf09e4c7f9b5f2030632a411679260adcc286959063a922772a610bc.jpg](../iclr_results/2832_StructRAG_ Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Stru/images/0a73d4bfbf09e4c7f9b5f2030632a411679260adcc286959063a922772a610bc.jpg)

![231a4410e796d3fc415a5fa79965b2ae15568c998a5697ed2ab069265aab638b.jpg](../iclr_results/2832_StructRAG_ Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Stru/images/231a4410e796d3fc415a5fa79965b2ae15568c998a5697ed2ab069265aab638b.jpg)

![57310b155daa7136c882dd44772c4b7fbd6f146f5e1f86b55c21bf61aea328d1.jpg](../iclr_results/2832_StructRAG_ Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Stru/images/57310b155daa7136c882dd44772c4b7fbd6f146f5e1f86b55c21bf61aea328d1.jpg)

![a984aec63b388aa40c08ecd9a13025a844f90dd3d0dab00e514b559a263978d8.jpg](../iclr_results/2832_StructRAG_ Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Stru/images/a984aec63b388aa40c08ecd9a13025a844f90dd3d0dab00e514b559a263978d8.jpg)

![ad065d9ef1a4ba1ca2f7ba0489a8df72d880d61babd59f1228b239bfdb2653ba.jpg](../iclr_results/2832_StructRAG_ Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Stru/images/ad065d9ef1a4ba1ca2f7ba0489a8df72d880d61babd59f1228b239bfdb2653ba.jpg)

![c90388ce906afa7d7cab02d6a94977ba6c40b8e2af73ab8e6ee2c27ec779c8fe.jpg](../iclr_results/2832_StructRAG_ Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Stru/images/c90388ce906afa7d7cab02d6a94977ba6c40b8e2af73ab8e6ee2c27ec779c8fe.jpg)

### Tables

![749185a6495f53f4752ffde8b5cecf62a62caf0dfc56b5257a46ced67ae52ff8.jpg](../iclr_results/2832_StructRAG_ Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Stru/tables/749185a6495f53f4752ffde8b5cecf62a62caf0dfc56b5257a46ced67ae52ff8.jpg)

![802022b7d578e806ad4ba67a11eb004144fd391e253fa6d80c9c84587e5228fe.jpg](../iclr_results/2832_StructRAG_ Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Stru/tables/802022b7d578e806ad4ba67a11eb004144fd391e253fa6d80c9c84587e5228fe.jpg)

![a59b81ce2f93a402226d08c9424484e8088406756a4c227b19c42d36a3c04a45.jpg](../iclr_results/2832_StructRAG_ Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Stru/tables/a59b81ce2f93a402226d08c9424484e8088406756a4c227b19c42d36a3c04a45.jpg)

![a787c8c7dacd1901be39e54c3dec71595a3fff89d714c0e736427074f7b0d80c.jpg](../iclr_results/2832_StructRAG_ Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Stru/tables/a787c8c7dacd1901be39e54c3dec71595a3fff89d714c0e736427074f7b0d80c.jpg)

![ad8cd9c994473a0e53f61644eb9dbaee02f0f8d1541a17797c38a580f06c691f.jpg](../iclr_results/2832_StructRAG_ Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Stru/tables/ad8cd9c994473a0e53f61644eb9dbaee02f0f8d1541a17797c38a580f06c691f.jpg)

![d6237fe2a9d8a35921fe0f0b142937678d74717bd7abc5b30d5552c0422f2cd4.jpg](../iclr_results/2832_StructRAG_ Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Stru/tables/d6237fe2a9d8a35921fe0f0b142937678d74717bd7abc5b30d5552c0422f2cd4.jpg)

![eb6bce8fe760ad2d604e78226044e4b37b4e99b1f4c077ce3559e950007c5082.jpg](../iclr_results/2832_StructRAG_ Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Stru/tables/eb6bce8fe760ad2d604e78226044e4b37b4e99b1f4c077ce3559e950007c5082.jpg)

## Unhackable Temporal Reward for Scalable Video MLLMs


### Images

![02305faebdeaaa92d01be94cc057bf0989c1e7c5f9fdcd046b5cf3967dd5958b.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/images/02305faebdeaaa92d01be94cc057bf0989c1e7c5f9fdcd046b5cf3967dd5958b.jpg)

![31f198f3908f2e4026cc4895ce49bc35e0ed8bd863469bfaac8cff0b5b6e6a2e.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/images/31f198f3908f2e4026cc4895ce49bc35e0ed8bd863469bfaac8cff0b5b6e6a2e.jpg)

![48d6c47781b2da1db3387211bc8050884cd3b6539d437c8ccbeee6ed885f2c03.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/images/48d6c47781b2da1db3387211bc8050884cd3b6539d437c8ccbeee6ed885f2c03.jpg)

![5da388f9dfd7e4b82d55f39d10fdc529e2e03753d5986f35b7b298d784ecb6ec.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/images/5da388f9dfd7e4b82d55f39d10fdc529e2e03753d5986f35b7b298d784ecb6ec.jpg)

![75922b6ae720dc1254fcec87d622bbbdda1313545b2ebaf5e49819f480a1176e.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/images/75922b6ae720dc1254fcec87d622bbbdda1313545b2ebaf5e49819f480a1176e.jpg)

![7619fdaf68966b59c763186c70790740044869b372802d08e2f25808a73f84e6.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/images/7619fdaf68966b59c763186c70790740044869b372802d08e2f25808a73f84e6.jpg)

![7a6cf713832e54e6e1093d2fab0046cacba362337339ef9687d0d9896cc45eb0.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/images/7a6cf713832e54e6e1093d2fab0046cacba362337339ef9687d0d9896cc45eb0.jpg)

![8346f6b572cbf233bf60e2750587d863705679de9512e1e6788a4c33426de854.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/images/8346f6b572cbf233bf60e2750587d863705679de9512e1e6788a4c33426de854.jpg)

![9ba1e583bdd7ad49cde3bb209653730553f007d2610c7361cf17509b2587823c.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/images/9ba1e583bdd7ad49cde3bb209653730553f007d2610c7361cf17509b2587823c.jpg)

![afc76c38f7a5607bdf841d5bdf274256e8da4c243ed0bf82b8600ad4bb4b46c1.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/images/afc76c38f7a5607bdf841d5bdf274256e8da4c243ed0bf82b8600ad4bb4b46c1.jpg)

![bf2ce267ba5190dd454bcae709156b7b91136cf6e1ed4e57ef970d7bf8155280.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/images/bf2ce267ba5190dd454bcae709156b7b91136cf6e1ed4e57ef970d7bf8155280.jpg)

![d91617356694437b4922ce37435fcbcfb82c8fe73e3968ca7769d751a7d8860c.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/images/d91617356694437b4922ce37435fcbcfb82c8fe73e3968ca7769d751a7d8860c.jpg)

### Tables

![0d6872baaf1df13251b6bc583baa230f7de9f5aed715c1b245e5e7b5ca30ec7a.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/tables/0d6872baaf1df13251b6bc583baa230f7de9f5aed715c1b245e5e7b5ca30ec7a.jpg)

![254c33ef1569e214dd63c9f667da761865c2f06a7be5a6b2d8b0845b82b39cc8.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/tables/254c33ef1569e214dd63c9f667da761865c2f06a7be5a6b2d8b0845b82b39cc8.jpg)

![260cf835faff510ba2478655149e72c94da17ae6b166dad0716c0c231dea9f13.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/tables/260cf835faff510ba2478655149e72c94da17ae6b166dad0716c0c231dea9f13.jpg)

![340c601115d5f276e7c64f9cd2122310a9b61d684240fc4f9e550745d73f87d5.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/tables/340c601115d5f276e7c64f9cd2122310a9b61d684240fc4f9e550745d73f87d5.jpg)

![66db0e214788bbe481b0a026b4af683a837c2298f18e8418c97107d6d9700110.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/tables/66db0e214788bbe481b0a026b4af683a837c2298f18e8418c97107d6d9700110.jpg)

![9a6c1664b563548774556b8bdd34bba58ca73b1bdcfbe34958061b5346ee2ffc.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/tables/9a6c1664b563548774556b8bdd34bba58ca73b1bdcfbe34958061b5346ee2ffc.jpg)

![bad0741c3a62ad123e064fad0b9623ebe9e0b25548ca9bbad8cdfc5765604195.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/tables/bad0741c3a62ad123e064fad0b9623ebe9e0b25548ca9bbad8cdfc5765604195.jpg)

![c55cc6b11edfd7c17234de63e07b9e7cd8d596bb6e1cb19b9f349fc675e70a2b.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/tables/c55cc6b11edfd7c17234de63e07b9e7cd8d596bb6e1cb19b9f349fc675e70a2b.jpg)

![d02d3c46692d9e9f8a42422e5e59ca03a47512eb5d472ac01d367173e1566b30.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/tables/d02d3c46692d9e9f8a42422e5e59ca03a47512eb5d472ac01d367173e1566b30.jpg)

![e881ff372dc0eae6b4f10f0b411766518fa76b1c3badd31383bcd8d978d91b6d.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/tables/e881ff372dc0eae6b4f10f0b411766518fa76b1c3badd31383bcd8d978d91b6d.jpg)

![f25c38c7288d26fa9185f32963747dfe1b80bab1178f1d45b47907373a9bfb93.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/tables/f25c38c7288d26fa9185f32963747dfe1b80bab1178f1d45b47907373a9bfb93.jpg)

![f57643693995df00b01f8b2f937ce13fc1495eb795c2c09d13c8871057e9156f.jpg](../iclr_results/2834_Unhackable Temporal Reward for Scalable Video MLLMs/tables/f57643693995df00b01f8b2f937ce13fc1495eb795c2c09d13c8871057e9156f.jpg)

## Taming Transformer Without Using Learning Rate Warmup


### Images

![0445e5511cb0d09c6b551689522b9e96847f7b949791182acdc8b21c6f07b215.jpg](../iclr_results/2835_Taming Transformer Without Using Learning Rate Warmup/images/0445e5511cb0d09c6b551689522b9e96847f7b949791182acdc8b21c6f07b215.jpg)

![0bd6875290055c3092747253fd41e0f32ceb85202804ad279c89dc10dc23dadd.jpg](../iclr_results/2835_Taming Transformer Without Using Learning Rate Warmup/images/0bd6875290055c3092747253fd41e0f32ceb85202804ad279c89dc10dc23dadd.jpg)

![0c727af152a6f46e0a3ab0a557b834944c765d1a88ebbff2d6361b4059f1c46d.jpg](../iclr_results/2835_Taming Transformer Without Using Learning Rate Warmup/images/0c727af152a6f46e0a3ab0a557b834944c765d1a88ebbff2d6361b4059f1c46d.jpg)

![0c8bad1f7098bc144d336cbd3b6763972b81800343c7bb7358e1eaead0e3308b.jpg](../iclr_results/2835_Taming Transformer Without Using Learning Rate Warmup/images/0c8bad1f7098bc144d336cbd3b6763972b81800343c7bb7358e1eaead0e3308b.jpg)

![1142ec61ae8df0d5afc7dd68eb81918bcc2c755f36c83a665673a7369db5dc2c.jpg](../iclr_results/2835_Taming Transformer Without Using Learning Rate Warmup/images/1142ec61ae8df0d5afc7dd68eb81918bcc2c755f36c83a665673a7369db5dc2c.jpg)

![4b38c7f4cc7c7e059f031d09cfe54cac4a4be13eb8a3f9d3a4038494190f1c8b.jpg](../iclr_results/2835_Taming Transformer Without Using Learning Rate Warmup/images/4b38c7f4cc7c7e059f031d09cfe54cac4a4be13eb8a3f9d3a4038494190f1c8b.jpg)

![5ee868eb4ca0f6e52fd46e30398f945cf3a0cdcdce44579db5ecb3fe6f576253.jpg](../iclr_results/2835_Taming Transformer Without Using Learning Rate Warmup/images/5ee868eb4ca0f6e52fd46e30398f945cf3a0cdcdce44579db5ecb3fe6f576253.jpg)

![65c62300a4922686e8989d1a8797b16095869c8ca6b0b86eb1a2a98b8a4c8fb8.jpg](../iclr_results/2835_Taming Transformer Without Using Learning Rate Warmup/images/65c62300a4922686e8989d1a8797b16095869c8ca6b0b86eb1a2a98b8a4c8fb8.jpg)

![7207754409cea127e0df01101ba1442595de7d293d19e90f02ace81d2cbb42d3.jpg](../iclr_results/2835_Taming Transformer Without Using Learning Rate Warmup/images/7207754409cea127e0df01101ba1442595de7d293d19e90f02ace81d2cbb42d3.jpg)

![9097f4fed2b97068cf7ba3dcc3ea57f980499c019d10c332f9bd528d893fe7c9.jpg](../iclr_results/2835_Taming Transformer Without Using Learning Rate Warmup/images/9097f4fed2b97068cf7ba3dcc3ea57f980499c019d10c332f9bd528d893fe7c9.jpg)

![e65f290d71bdaa8f4e293ec2c7578c993ff0222b6c357c99f22c2a6d003061f5.jpg](../iclr_results/2835_Taming Transformer Without Using Learning Rate Warmup/images/e65f290d71bdaa8f4e293ec2c7578c993ff0222b6c357c99f22c2a6d003061f5.jpg)

![eb13667fa85fcb031184d91da476058c4375da312fb42c6ec92bc67f2748efe2.jpg](../iclr_results/2835_Taming Transformer Without Using Learning Rate Warmup/images/eb13667fa85fcb031184d91da476058c4375da312fb42c6ec92bc67f2748efe2.jpg)

### Tables

![623053aa9d763e1b34e0b2318ae7002e7de9d4ef5d141bc63d98023cbeeb17f6.jpg](../iclr_results/2835_Taming Transformer Without Using Learning Rate Warmup/tables/623053aa9d763e1b34e0b2318ae7002e7de9d4ef5d141bc63d98023cbeeb17f6.jpg)

![b707372e105e680239efed3b8a86fd728bade06690eac842bf3c067c749266fe.jpg](../iclr_results/2835_Taming Transformer Without Using Learning Rate Warmup/tables/b707372e105e680239efed3b8a86fd728bade06690eac842bf3c067c749266fe.jpg)

![beb89c6634011ef16fc36e9f7befd92522fe511aaaae15c130b73191c4e64db9.jpg](../iclr_results/2835_Taming Transformer Without Using Learning Rate Warmup/tables/beb89c6634011ef16fc36e9f7befd92522fe511aaaae15c130b73191c4e64db9.jpg)

![bf4f16bd1c1fdb7d9d9f46246c2b34fb1bf66ccd8c30c111f49dfb561e8b74b3.jpg](../iclr_results/2835_Taming Transformer Without Using Learning Rate Warmup/tables/bf4f16bd1c1fdb7d9d9f46246c2b34fb1bf66ccd8c30c111f49dfb561e8b74b3.jpg)

![d2f754f1a0ce9cd9a6da0beb519e9dcab427d970e553267d0ce5134f8164419b.jpg](../iclr_results/2835_Taming Transformer Without Using Learning Rate Warmup/tables/d2f754f1a0ce9cd9a6da0beb519e9dcab427d970e553267d0ce5134f8164419b.jpg)

![e79d5527a524953309a577cfae7351dc606f23c29bfc5952f67015cab576a410.jpg](../iclr_results/2835_Taming Transformer Without Using Learning Rate Warmup/tables/e79d5527a524953309a577cfae7351dc606f23c29bfc5952f67015cab576a410.jpg)

## How Learnable Grids Recover Fine Detail in Low Dimensions: A Neural Tangent Kernel Analysis of Multigrid Parametric Encodings


### Images

![1509b0bded8c7dc5ebe07292d97e34939da08e578c0be27db12199145e86219f.jpg](../iclr_results/2836_How Learnable Grids Recover Fine Detail in Low Dimensions_ A Neural Tangent Kernel Analysis of Multi/images/1509b0bded8c7dc5ebe07292d97e34939da08e578c0be27db12199145e86219f.jpg)

![30b152a7d1cee8cf6263ee1e803d8f8c1eb2bdb20ca426bf4f8a2698f52635f4.jpg](../iclr_results/2836_How Learnable Grids Recover Fine Detail in Low Dimensions_ A Neural Tangent Kernel Analysis of Multi/images/30b152a7d1cee8cf6263ee1e803d8f8c1eb2bdb20ca426bf4f8a2698f52635f4.jpg)

![357a0a943304a0c74f5a3b5202df9f480dbe2e6e2ebfda69a2620a978484d612.jpg](../iclr_results/2836_How Learnable Grids Recover Fine Detail in Low Dimensions_ A Neural Tangent Kernel Analysis of Multi/images/357a0a943304a0c74f5a3b5202df9f480dbe2e6e2ebfda69a2620a978484d612.jpg)

![44ee11cf5173c67d0550f734f9fe4ea3e146c303e46070b1a07e184bced82107.jpg](../iclr_results/2836_How Learnable Grids Recover Fine Detail in Low Dimensions_ A Neural Tangent Kernel Analysis of Multi/images/44ee11cf5173c67d0550f734f9fe4ea3e146c303e46070b1a07e184bced82107.jpg)

![5fff7e6b3ac22a6f64ef9a1200ad230eea7e27f41ea5d3e9d78467c468931952.jpg](../iclr_results/2836_How Learnable Grids Recover Fine Detail in Low Dimensions_ A Neural Tangent Kernel Analysis of Multi/images/5fff7e6b3ac22a6f64ef9a1200ad230eea7e27f41ea5d3e9d78467c468931952.jpg)

![722a07a1d4610934c2947f80469fcccc4694c5c284d726cdceb158d83bffcd8b.jpg](../iclr_results/2836_How Learnable Grids Recover Fine Detail in Low Dimensions_ A Neural Tangent Kernel Analysis of Multi/images/722a07a1d4610934c2947f80469fcccc4694c5c284d726cdceb158d83bffcd8b.jpg)

![87d8aa5e38dc53b333773ef18ec8fb361a60709f48a816f5d9783278cc55117d.jpg](../iclr_results/2836_How Learnable Grids Recover Fine Detail in Low Dimensions_ A Neural Tangent Kernel Analysis of Multi/images/87d8aa5e38dc53b333773ef18ec8fb361a60709f48a816f5d9783278cc55117d.jpg)

![903f9ee231ff0fc65c7a70c4a51226ddaa09f43898f01c2a11facf746e51a35c.jpg](../iclr_results/2836_How Learnable Grids Recover Fine Detail in Low Dimensions_ A Neural Tangent Kernel Analysis of Multi/images/903f9ee231ff0fc65c7a70c4a51226ddaa09f43898f01c2a11facf746e51a35c.jpg)

![99035d671429407f9b940d0647aa38ba3feff0255a9d93ed447b361a39466a58.jpg](../iclr_results/2836_How Learnable Grids Recover Fine Detail in Low Dimensions_ A Neural Tangent Kernel Analysis of Multi/images/99035d671429407f9b940d0647aa38ba3feff0255a9d93ed447b361a39466a58.jpg)

![991b5566008d45dc904e8f80d58f5698ac7855b23750af0ba0bdd98ba41e3845.jpg](../iclr_results/2836_How Learnable Grids Recover Fine Detail in Low Dimensions_ A Neural Tangent Kernel Analysis of Multi/images/991b5566008d45dc904e8f80d58f5698ac7855b23750af0ba0bdd98ba41e3845.jpg)

![a19859602b2c6dfc65c497317717a9269a7ed67973a81d3b9e64fd6eb7b8b0ee.jpg](../iclr_results/2836_How Learnable Grids Recover Fine Detail in Low Dimensions_ A Neural Tangent Kernel Analysis of Multi/images/a19859602b2c6dfc65c497317717a9269a7ed67973a81d3b9e64fd6eb7b8b0ee.jpg)

![a774444d4e42e62536587566f100344606a848dbe15ef96a7851ca25b318f61f.jpg](../iclr_results/2836_How Learnable Grids Recover Fine Detail in Low Dimensions_ A Neural Tangent Kernel Analysis of Multi/images/a774444d4e42e62536587566f100344606a848dbe15ef96a7851ca25b318f61f.jpg)

![c918cfcc60a641ab3f84396dd8e4017f320df5d030936c9fd45256dd7ddc28fc.jpg](../iclr_results/2836_How Learnable Grids Recover Fine Detail in Low Dimensions_ A Neural Tangent Kernel Analysis of Multi/images/c918cfcc60a641ab3f84396dd8e4017f320df5d030936c9fd45256dd7ddc28fc.jpg)

![dd0af47945a83c05e0db0b1fe51057d2948a3580a13c982d8228a67f4692ea85.jpg](../iclr_results/2836_How Learnable Grids Recover Fine Detail in Low Dimensions_ A Neural Tangent Kernel Analysis of Multi/images/dd0af47945a83c05e0db0b1fe51057d2948a3580a13c982d8228a67f4692ea85.jpg)

### Tables

![19d1c8e95028b5e2fd4b7ff62c817c666b7570a95de458883067aab6811a22d3.jpg](../iclr_results/2836_How Learnable Grids Recover Fine Detail in Low Dimensions_ A Neural Tangent Kernel Analysis of Multi/tables/19d1c8e95028b5e2fd4b7ff62c817c666b7570a95de458883067aab6811a22d3.jpg)

![2c156ecb1f5febe6304243b033339255c6770cc2e2b0e7a972b27bbf138eac07.jpg](../iclr_results/2836_How Learnable Grids Recover Fine Detail in Low Dimensions_ A Neural Tangent Kernel Analysis of Multi/tables/2c156ecb1f5febe6304243b033339255c6770cc2e2b0e7a972b27bbf138eac07.jpg)

![7b9e31e80fbe0f02445ce91cf01f46e13f58bc0a25699ed7ead64fc48187d51e.jpg](../iclr_results/2836_How Learnable Grids Recover Fine Detail in Low Dimensions_ A Neural Tangent Kernel Analysis of Multi/tables/7b9e31e80fbe0f02445ce91cf01f46e13f58bc0a25699ed7ead64fc48187d51e.jpg)

![9628a471119eaea7e3231ba38ce238cbb8ed1a65c663a689fc28f5b00cbafeba.jpg](../iclr_results/2836_How Learnable Grids Recover Fine Detail in Low Dimensions_ A Neural Tangent Kernel Analysis of Multi/tables/9628a471119eaea7e3231ba38ce238cbb8ed1a65c663a689fc28f5b00cbafeba.jpg)

## NeurFlow: Interpreting Neural Networks through Neuron Groups and Functional Interactions


### Images

![0ad57b962de5f8bebca4150a0ed719ae666b6007daa8a90eceec4023ad3190e2.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/images/0ad57b962de5f8bebca4150a0ed719ae666b6007daa8a90eceec4023ad3190e2.jpg)

![1164c3fea7f8407e9ea24642498ef7b87fffea51f1f3950d65f4652f838a4a65.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/images/1164c3fea7f8407e9ea24642498ef7b87fffea51f1f3950d65f4652f838a4a65.jpg)

![13c2a768077685b4003d632d6f3ea38ef394c6dc5b7562954848897b14c03003.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/images/13c2a768077685b4003d632d6f3ea38ef394c6dc5b7562954848897b14c03003.jpg)

![14584d9912b531d36ce34716c3b5cd13d8285a8fe17e89ea841dbd81d204103e.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/images/14584d9912b531d36ce34716c3b5cd13d8285a8fe17e89ea841dbd81d204103e.jpg)

![19501d4de6f0ba59811d4ff7e2a62a00f202b34892cf964fcff02c65ccba77c9.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/images/19501d4de6f0ba59811d4ff7e2a62a00f202b34892cf964fcff02c65ccba77c9.jpg)

![1dcff224ddc3ffc45b9e504a04dc4238bede52d379c50995da3f2574f32a229c.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/images/1dcff224ddc3ffc45b9e504a04dc4238bede52d379c50995da3f2574f32a229c.jpg)

![27702724b623ea377d7d89f77098bdee99cf67169a9686f64a3df9134676e5a8.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/images/27702724b623ea377d7d89f77098bdee99cf67169a9686f64a3df9134676e5a8.jpg)

![28032c85e32b54b2d6e1333020425e1b584c31b26b2b1e7dc2bff529a9f890b2.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/images/28032c85e32b54b2d6e1333020425e1b584c31b26b2b1e7dc2bff529a9f890b2.jpg)

![28584bbe4401946a9b410218ca597640e161a292571f583d0d2f575e425cb656.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/images/28584bbe4401946a9b410218ca597640e161a292571f583d0d2f575e425cb656.jpg)

![2b5d3d23e28c81424378c8b0db281cd3c98eab00d6f9283d390db3b06502bc3f.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/images/2b5d3d23e28c81424378c8b0db281cd3c98eab00d6f9283d390db3b06502bc3f.jpg)

![2f7323b3b23dfbab578a86fe617b43adeecb4ae53683ef49b6c1ef40c1aee344.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/images/2f7323b3b23dfbab578a86fe617b43adeecb4ae53683ef49b6c1ef40c1aee344.jpg)

![42eb8aa879928680ed0df6ffac90503d17a7f4fb22ccd0c1850ede6ec88b1380.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/images/42eb8aa879928680ed0df6ffac90503d17a7f4fb22ccd0c1850ede6ec88b1380.jpg)

![451c8972ecceec36ada1e4a1fc4ffaaacf24dc995225dd174a52a7b420d1576f.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/images/451c8972ecceec36ada1e4a1fc4ffaaacf24dc995225dd174a52a7b420d1576f.jpg)

![741d527a54eb3d9571475bf9b870769968b5b761e4eec91150950f53c4e74827.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/images/741d527a54eb3d9571475bf9b870769968b5b761e4eec91150950f53c4e74827.jpg)

![ad49f5eaf352b4a17b4420d8211039b91778d096e6b8e3fabc21e4e6c6e735ec.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/images/ad49f5eaf352b4a17b4420d8211039b91778d096e6b8e3fabc21e4e6c6e735ec.jpg)

![b6376496a2b7f64d261efc78e8ccc1172e7f507efd3f88cc039e648f62106dc0.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/images/b6376496a2b7f64d261efc78e8ccc1172e7f507efd3f88cc039e648f62106dc0.jpg)

![d13975d661e6e9f88ad95972537b7011bfb7da1d3965d6114c2fd8c41fc8599f.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/images/d13975d661e6e9f88ad95972537b7011bfb7da1d3965d6114c2fd8c41fc8599f.jpg)

![e21e6faae16146860b9539803be1b309a943e40f6049d5f85a8b91565f2b6c6c.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/images/e21e6faae16146860b9539803be1b309a943e40f6049d5f85a8b91565f2b6c6c.jpg)

### Tables

![5f62a2302e74a884b4f389cc942ad9d1b33410a95eeec7e9b1e4b8cf3990e7f0.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/tables/5f62a2302e74a884b4f389cc942ad9d1b33410a95eeec7e9b1e4b8cf3990e7f0.jpg)

![5f91b354b1eddfdfbaf4ec2b08f9d7c277ca37ad296b94a1cb4338bdb41f065e.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/tables/5f91b354b1eddfdfbaf4ec2b08f9d7c277ca37ad296b94a1cb4338bdb41f065e.jpg)

![7236bc11c30fb2b85923f6829063d80ccd2cf7ce2c92063a57ddb90d7c8c9f16.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/tables/7236bc11c30fb2b85923f6829063d80ccd2cf7ce2c92063a57ddb90d7c8c9f16.jpg)

![8687704f071b636aac07f5f40bdba676c3f666bfc8d44f66ff1aebc707f48f57.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/tables/8687704f071b636aac07f5f40bdba676c3f666bfc8d44f66ff1aebc707f48f57.jpg)

![8cf79b2c293ec3fc9abd26e3892019f5866836105c09ac3799db017b7500bbdd.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/tables/8cf79b2c293ec3fc9abd26e3892019f5866836105c09ac3799db017b7500bbdd.jpg)

![a8cff4d8cca19787ce50ffafc9b2446b6705834a3d8d90cf9f3625d302943f29.jpg](../iclr_results/2837_NeurFlow_ Interpreting Neural Networks through Neuron Groups and Functional Interactions/tables/a8cff4d8cca19787ce50ffafc9b2446b6705834a3d8d90cf9f3625d302943f29.jpg)

## RaSA: Rank-Sharing Low-Rank Adaptation


### Images

![1828ff07e2ed8695ad27f78eb944c6ce2b37f70b54b729f37865c67f7d526d63.jpg](../iclr_results/2838_RaSA_ Rank-Sharing Low-Rank Adaptation/images/1828ff07e2ed8695ad27f78eb944c6ce2b37f70b54b729f37865c67f7d526d63.jpg)

![3e48792f8303d2ccf8671c2264ca5c0bdac673fe58e5d5c0c39b3d3357e2660e.jpg](../iclr_results/2838_RaSA_ Rank-Sharing Low-Rank Adaptation/images/3e48792f8303d2ccf8671c2264ca5c0bdac673fe58e5d5c0c39b3d3357e2660e.jpg)

![5b310700270c4525b787ddb73d4cbf900e5d6d67ea1c2764d8bb8076b81bec46.jpg](../iclr_results/2838_RaSA_ Rank-Sharing Low-Rank Adaptation/images/5b310700270c4525b787ddb73d4cbf900e5d6d67ea1c2764d8bb8076b81bec46.jpg)

![690709c451a79c66a4e74bdf6a608b714c8efed38e2cb5651d1a4e4dbc9682d6.jpg](../iclr_results/2838_RaSA_ Rank-Sharing Low-Rank Adaptation/images/690709c451a79c66a4e74bdf6a608b714c8efed38e2cb5651d1a4e4dbc9682d6.jpg)

![72cda4d0d9350b3bcb5ea4c844fffaf8cd5e32f53af328d2ae3fb9d01cbca037.jpg](../iclr_results/2838_RaSA_ Rank-Sharing Low-Rank Adaptation/images/72cda4d0d9350b3bcb5ea4c844fffaf8cd5e32f53af328d2ae3fb9d01cbca037.jpg)

![75185dd9111770851676463861064037ab402a57592ef11db279ed8448ee2448.jpg](../iclr_results/2838_RaSA_ Rank-Sharing Low-Rank Adaptation/images/75185dd9111770851676463861064037ab402a57592ef11db279ed8448ee2448.jpg)

![b9de41483e04c1a723f635550d7714e349a738b22d5d183d6f67dfdb4d40840c.jpg](../iclr_results/2838_RaSA_ Rank-Sharing Low-Rank Adaptation/images/b9de41483e04c1a723f635550d7714e349a738b22d5d183d6f67dfdb4d40840c.jpg)

### Tables

![244e0a61765fe19d2d59eb19bbd652736cb29aa09df8ca72def55c391aabd251.jpg](../iclr_results/2838_RaSA_ Rank-Sharing Low-Rank Adaptation/tables/244e0a61765fe19d2d59eb19bbd652736cb29aa09df8ca72def55c391aabd251.jpg)

![f0259d196b8c19f5f0859c0d5e32cb6889e7701b77fe4a838a905d6588e89cf5.jpg](../iclr_results/2838_RaSA_ Rank-Sharing Low-Rank Adaptation/tables/f0259d196b8c19f5f0859c0d5e32cb6889e7701b77fe4a838a905d6588e89cf5.jpg)

## Rethinking Diffusion Posterior Sampling: From Conditional Score Estimator to Maximizing a Posterior


### Images

![03ba72584820a0014491b2885b160281002ab2ec4f7dae6b56f32653c226d29a.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/images/03ba72584820a0014491b2885b160281002ab2ec4f7dae6b56f32653c226d29a.jpg)

![31970e94fab16ef6c1328b2448f22a7812570ba1758cf88fbd0887333d4e3a9e.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/images/31970e94fab16ef6c1328b2448f22a7812570ba1758cf88fbd0887333d4e3a9e.jpg)

![46104e6abb38d2a65b636e8f91c31136e11c86e9a02c87937baca46709a8e9ed.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/images/46104e6abb38d2a65b636e8f91c31136e11c86e9a02c87937baca46709a8e9ed.jpg)

![7547535f2b0e5859bdca1539d12cf5bba306d80ed017d4ccda571879edc4d3ae.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/images/7547535f2b0e5859bdca1539d12cf5bba306d80ed017d4ccda571879edc4d3ae.jpg)

![79d25ac28b46f53df7083f3cda738d290a9abbaff8d717b708dae177fa6ca5ad.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/images/79d25ac28b46f53df7083f3cda738d290a9abbaff8d717b708dae177fa6ca5ad.jpg)

![7b592446cb369a628287bc354558bb11f5e97987559731c6c7eac827fb2e3516.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/images/7b592446cb369a628287bc354558bb11f5e97987559731c6c7eac827fb2e3516.jpg)

![ad074f5e791c81d683ffaa9069779be672157525bc64055fe8fe9300e533a3ac.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/images/ad074f5e791c81d683ffaa9069779be672157525bc64055fe8fe9300e533a3ac.jpg)

![ae3a131303fe6e54fbfbd5b9a1bc1f4f3ab03b20721b1d88007a1baf380fd588.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/images/ae3a131303fe6e54fbfbd5b9a1bc1f4f3ab03b20721b1d88007a1baf380fd588.jpg)

![b9bc8dea7ce21a91d1c4aee75ea28f7274586450abff77e167afc539439aaeda.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/images/b9bc8dea7ce21a91d1c4aee75ea28f7274586450abff77e167afc539439aaeda.jpg)

![c582b0bd22d4cfda342bea2e099bd243a574d9e20c3ff12c486aae15598b6337.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/images/c582b0bd22d4cfda342bea2e099bd243a574d9e20c3ff12c486aae15598b6337.jpg)

![cabf5208c88ca83c681f965e516e4682f2be592e1df2b751e6061575cf05a83d.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/images/cabf5208c88ca83c681f965e516e4682f2be592e1df2b751e6061575cf05a83d.jpg)

![db4b8cd88ae1d51e46b8bb73062dffcb5a5d034cb4961ab28a56ef8b8bcdd775.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/images/db4b8cd88ae1d51e46b8bb73062dffcb5a5d034cb4961ab28a56ef8b8bcdd775.jpg)

### Tables

![34f0cc9fea3415e3b0a9d2d07044b82cb2d1d9a2461fe72bceaa2e26b1a5ec8a.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/tables/34f0cc9fea3415e3b0a9d2d07044b82cb2d1d9a2461fe72bceaa2e26b1a5ec8a.jpg)

![40449806fd9afdc8047e47ce9874f51be774a52b8d893ae46748a5fde34484b3.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/tables/40449806fd9afdc8047e47ce9874f51be774a52b8d893ae46748a5fde34484b3.jpg)

![4cc10c0e983536b10a79a734628e71e9c880db4af94273eb8fe1a357c5e98b15.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/tables/4cc10c0e983536b10a79a734628e71e9c880db4af94273eb8fe1a357c5e98b15.jpg)

![4d41265db65d3be80685e15ced4958dca5a8a730908ebfa389e15312a333e979.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/tables/4d41265db65d3be80685e15ced4958dca5a8a730908ebfa389e15312a333e979.jpg)

![5b6036971bdc22f7553da87435a86548474dd486e91f2ece134c07e3751d200e.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/tables/5b6036971bdc22f7553da87435a86548474dd486e91f2ece134c07e3751d200e.jpg)

![7149a9097ab1b6bca7ee0bf128d41332cded95e6af1041f9d06499223843837e.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/tables/7149a9097ab1b6bca7ee0bf128d41332cded95e6af1041f9d06499223843837e.jpg)

![7b01b7ace4cc3c84ae95de94e20063b6f6cc761edba765fc4460af507bc545e8.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/tables/7b01b7ace4cc3c84ae95de94e20063b6f6cc761edba765fc4460af507bc545e8.jpg)

![8da6f2421749d40049ce74e2ed539555d2bd4bb915e39fd9071cebb5f19f84ee.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/tables/8da6f2421749d40049ce74e2ed539555d2bd4bb915e39fd9071cebb5f19f84ee.jpg)

![c71c716d7ec26a3366e2c011d961ab41e1d37086686d926e0cd1ca554a5b42f1.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/tables/c71c716d7ec26a3366e2c011d961ab41e1d37086686d926e0cd1ca554a5b42f1.jpg)

![e8958cdb21a4ed6275dfc633aa84adbcfba1240d394a45978d3d21b2c0b254ec.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/tables/e8958cdb21a4ed6275dfc633aa84adbcfba1240d394a45978d3d21b2c0b254ec.jpg)

![ea0cea851273c62c8540fc12753430ec1b667fafa3ee817a16fd75c4c3b67dfd.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/tables/ea0cea851273c62c8540fc12753430ec1b667fafa3ee817a16fd75c4c3b67dfd.jpg)

![efe7446765c28150948aa5533389089b9f4c2635c4c2a3acca26f703b7ae4384.jpg](../iclr_results/2839_Rethinking Diffusion Posterior Sampling_ From Conditional Score Estimator to Maximizing a Posterior/tables/efe7446765c28150948aa5533389089b9f4c2635c4c2a3acca26f703b7ae4384.jpg)

## Adversarial Mixup Unlearning


### Images

![06336166b55199aaade96f3dd4ef9f6a948a7d4db38aed99fd221803aef4fa4b.jpg](../iclr_results/2840_Adversarial Mixup Unlearning/images/06336166b55199aaade96f3dd4ef9f6a948a7d4db38aed99fd221803aef4fa4b.jpg)

![1c8b318e4751ddb2181a3747ec76a4d1a834d0084bee14341863c46971e36dca.jpg](../iclr_results/2840_Adversarial Mixup Unlearning/images/1c8b318e4751ddb2181a3747ec76a4d1a834d0084bee14341863c46971e36dca.jpg)

![250d4080e62d8fa3a68bdd888fda72ff6cbf3a62ccad84876a390dfe9df6d3f7.jpg](../iclr_results/2840_Adversarial Mixup Unlearning/images/250d4080e62d8fa3a68bdd888fda72ff6cbf3a62ccad84876a390dfe9df6d3f7.jpg)

![27aeb5b76dfe5a6ea2b86236e55f8485d7cac28e32149595cb47dd71d59615a6.jpg](../iclr_results/2840_Adversarial Mixup Unlearning/images/27aeb5b76dfe5a6ea2b86236e55f8485d7cac28e32149595cb47dd71d59615a6.jpg)

![4082b0220cdca07f9a558f24c170008e69fc06e0145161f4734754be87205903.jpg](../iclr_results/2840_Adversarial Mixup Unlearning/images/4082b0220cdca07f9a558f24c170008e69fc06e0145161f4734754be87205903.jpg)

![4106ddb3464fecc5ec5057d20754f68054055652f02ac1a8e72508968ba049dd.jpg](../iclr_results/2840_Adversarial Mixup Unlearning/images/4106ddb3464fecc5ec5057d20754f68054055652f02ac1a8e72508968ba049dd.jpg)

![4f02c520af8aa7b67537f7ac1259a0ad752a0cb05e3e2322568b8bba9a0a0ad7.jpg](../iclr_results/2840_Adversarial Mixup Unlearning/images/4f02c520af8aa7b67537f7ac1259a0ad752a0cb05e3e2322568b8bba9a0a0ad7.jpg)

![5587b286764cfc329002c6164ab89a55ddf5b78a4d093bc5bad5de0d06c4dab9.jpg](../iclr_results/2840_Adversarial Mixup Unlearning/images/5587b286764cfc329002c6164ab89a55ddf5b78a4d093bc5bad5de0d06c4dab9.jpg)

![715583b8088b03bb64a124c3565a9aaa956ed5f7d00850bb13cb8ede040f5333.jpg](../iclr_results/2840_Adversarial Mixup Unlearning/images/715583b8088b03bb64a124c3565a9aaa956ed5f7d00850bb13cb8ede040f5333.jpg)

![d8c51ba3d15a8c857245116dc9f8dda2c4e40b38b7c0033634fa0378feaff0bd.jpg](../iclr_results/2840_Adversarial Mixup Unlearning/images/d8c51ba3d15a8c857245116dc9f8dda2c4e40b38b7c0033634fa0378feaff0bd.jpg)

![fc6df487843e51946ae6e9d978bca7317e076221b8aa055ca5e87782e3889e63.jpg](../iclr_results/2840_Adversarial Mixup Unlearning/images/fc6df487843e51946ae6e9d978bca7317e076221b8aa055ca5e87782e3889e63.jpg)

### Tables

![06f1dd1e1a4e1da0383363208581f4b2b97ab46276255bbada2c262b85fa3094.jpg](../iclr_results/2840_Adversarial Mixup Unlearning/tables/06f1dd1e1a4e1da0383363208581f4b2b97ab46276255bbada2c262b85fa3094.jpg)

![340063bccbc739da8a8e500001bb6edb5230d506ebe4eb367f0f8236690ea23c.jpg](../iclr_results/2840_Adversarial Mixup Unlearning/tables/340063bccbc739da8a8e500001bb6edb5230d506ebe4eb367f0f8236690ea23c.jpg)

![410a87c481e7f7b5703ed498595bbc7f1a67ff50e93f7205c5af4f28b9865ca8.jpg](../iclr_results/2840_Adversarial Mixup Unlearning/tables/410a87c481e7f7b5703ed498595bbc7f1a67ff50e93f7205c5af4f28b9865ca8.jpg)

![6d7aafefe961bc67526be48489a27b80489bc6bd5964619ae514af13d89fa4ea.jpg](../iclr_results/2840_Adversarial Mixup Unlearning/tables/6d7aafefe961bc67526be48489a27b80489bc6bd5964619ae514af13d89fa4ea.jpg)

![7d163569871059a42d18ea42386094c76eaa3cf5842dcfb60e67f9b4163168d3.jpg](../iclr_results/2840_Adversarial Mixup Unlearning/tables/7d163569871059a42d18ea42386094c76eaa3cf5842dcfb60e67f9b4163168d3.jpg)

![875344986514f823d3064b151ce8ef3f33f3ca5c587009b3097f831b482cfad6.jpg](../iclr_results/2840_Adversarial Mixup Unlearning/tables/875344986514f823d3064b151ce8ef3f33f3ca5c587009b3097f831b482cfad6.jpg)

![deaef1c2534d398b508e3ad5c0d2fffa25f4bb4c4daf19949c89caca7637b5ef.jpg](../iclr_results/2840_Adversarial Mixup Unlearning/tables/deaef1c2534d398b508e3ad5c0d2fffa25f4bb4c4daf19949c89caca7637b5ef.jpg)

## SMT: Fine-Tuning Large Language Models with Sparse Matrices


### Images

![096ca40ffb8b8f23b638dbe6a1c2b3009f506d7a266477a3be4a969b59c33b29.jpg](../iclr_results/2841_SMT_ Fine-Tuning Large Language Models with Sparse Matrices/images/096ca40ffb8b8f23b638dbe6a1c2b3009f506d7a266477a3be4a969b59c33b29.jpg)

![4d81745139e953f20feb4c22519ce85b1b9ca42becc7fe2341754eb7c1f89c5d.jpg](../iclr_results/2841_SMT_ Fine-Tuning Large Language Models with Sparse Matrices/images/4d81745139e953f20feb4c22519ce85b1b9ca42becc7fe2341754eb7c1f89c5d.jpg)

![5068be275e49a7b5199d64a5d7c4c7a9eb1d8e4798094a557899103fc605893d.jpg](../iclr_results/2841_SMT_ Fine-Tuning Large Language Models with Sparse Matrices/images/5068be275e49a7b5199d64a5d7c4c7a9eb1d8e4798094a557899103fc605893d.jpg)

![52d8706fdbcf6513baed260ee45c1084735033c4265ee5257f00b381a20961dc.jpg](../iclr_results/2841_SMT_ Fine-Tuning Large Language Models with Sparse Matrices/images/52d8706fdbcf6513baed260ee45c1084735033c4265ee5257f00b381a20961dc.jpg)

![54a574663470442171acf659130c5640c9afa1257225d50a3ea5170378a15d3f.jpg](../iclr_results/2841_SMT_ Fine-Tuning Large Language Models with Sparse Matrices/images/54a574663470442171acf659130c5640c9afa1257225d50a3ea5170378a15d3f.jpg)

![6c6caebc918f0fd921e722bcab522c9c737aa35e0609b46727e7a761ecc85088.jpg](../iclr_results/2841_SMT_ Fine-Tuning Large Language Models with Sparse Matrices/images/6c6caebc918f0fd921e722bcab522c9c737aa35e0609b46727e7a761ecc85088.jpg)

![9e5bff049f09c78ed99ef41fc62d57e182c8f494b2c4688ae22a53b0bb16407f.jpg](../iclr_results/2841_SMT_ Fine-Tuning Large Language Models with Sparse Matrices/images/9e5bff049f09c78ed99ef41fc62d57e182c8f494b2c4688ae22a53b0bb16407f.jpg)

![b8d7b517c61eefb1512bdd234bb29f069fb9986fa727517b3ff6cb25d3dad5b5.jpg](../iclr_results/2841_SMT_ Fine-Tuning Large Language Models with Sparse Matrices/images/b8d7b517c61eefb1512bdd234bb29f069fb9986fa727517b3ff6cb25d3dad5b5.jpg)

![d9b40948ea4f91bfef96dbb7b3c40231dc5c70a9e636926d68cee3536c41715f.jpg](../iclr_results/2841_SMT_ Fine-Tuning Large Language Models with Sparse Matrices/images/d9b40948ea4f91bfef96dbb7b3c40231dc5c70a9e636926d68cee3536c41715f.jpg)

![eeae7267b5bca4054da0e327598369e8da9c1fe10843e4303f2c7a149f136f12.jpg](../iclr_results/2841_SMT_ Fine-Tuning Large Language Models with Sparse Matrices/images/eeae7267b5bca4054da0e327598369e8da9c1fe10843e4303f2c7a149f136f12.jpg)

![f97d9c38e7ae9fa430213934606e6b49dec7cb46b45d3690e6fba69a7b556cae.jpg](../iclr_results/2841_SMT_ Fine-Tuning Large Language Models with Sparse Matrices/images/f97d9c38e7ae9fa430213934606e6b49dec7cb46b45d3690e6fba69a7b556cae.jpg)

### Tables

![0ac2e9652a35cc0aa2156196e83c9b02396f21fed02b25f6ae51445a8dc9edb4.jpg](../iclr_results/2841_SMT_ Fine-Tuning Large Language Models with Sparse Matrices/tables/0ac2e9652a35cc0aa2156196e83c9b02396f21fed02b25f6ae51445a8dc9edb4.jpg)

![16c526648661f7beb0aed38d736919cdc90378442e546a4d8e06c6633b3e52c4.jpg](../iclr_results/2841_SMT_ Fine-Tuning Large Language Models with Sparse Matrices/tables/16c526648661f7beb0aed38d736919cdc90378442e546a4d8e06c6633b3e52c4.jpg)

![43bbb6f39bd036305b6b89023a4333652c9d295beaf3c470f3083bcc6ff0ce6c.jpg](../iclr_results/2841_SMT_ Fine-Tuning Large Language Models with Sparse Matrices/tables/43bbb6f39bd036305b6b89023a4333652c9d295beaf3c470f3083bcc6ff0ce6c.jpg)

![4e13bcc90e7b0d3f86a7c4c8bcd3d4b4f389a897f94f0b7e2796e45ee6824861.jpg](../iclr_results/2841_SMT_ Fine-Tuning Large Language Models with Sparse Matrices/tables/4e13bcc90e7b0d3f86a7c4c8bcd3d4b4f389a897f94f0b7e2796e45ee6824861.jpg)

![509067661333f99745bc906565b046384eac2986ce89e6b96b57044c53538341.jpg](../iclr_results/2841_SMT_ Fine-Tuning Large Language Models with Sparse Matrices/tables/509067661333f99745bc906565b046384eac2986ce89e6b96b57044c53538341.jpg)

![61d625cf2d22fa9d989d6fe8cc39d8933e58f1d2bbf58c6fa5aad4050f5b97ea.jpg](../iclr_results/2841_SMT_ Fine-Tuning Large Language Models with Sparse Matrices/tables/61d625cf2d22fa9d989d6fe8cc39d8933e58f1d2bbf58c6fa5aad4050f5b97ea.jpg)

![b384035cd830dfb13a5460794e4b496f5067cb23ea45eeb031d14853b7878936.jpg](../iclr_results/2841_SMT_ Fine-Tuning Large Language Models with Sparse Matrices/tables/b384035cd830dfb13a5460794e4b496f5067cb23ea45eeb031d14853b7878936.jpg)

![b68fe11acccdfdc72eb3cc8b65232cae73c05dcb716f4036e1d43916b3e77b9b.jpg](../iclr_results/2841_SMT_ Fine-Tuning Large Language Models with Sparse Matrices/tables/b68fe11acccdfdc72eb3cc8b65232cae73c05dcb716f4036e1d43916b3e77b9b.jpg)

![b80185f9ffd0ee643a16fe12fadaf150dd2e5fb937a7dbde41962a171c33ca79.jpg](../iclr_results/2841_SMT_ Fine-Tuning Large Language Models with Sparse Matrices/tables/b80185f9ffd0ee643a16fe12fadaf150dd2e5fb937a7dbde41962a171c33ca79.jpg)

![c6685cbabcc7b9b2feb829736d7ec91fe069368eb417b5c2a4e61d988c08b74e.jpg](../iclr_results/2841_SMT_ Fine-Tuning Large Language Models with Sparse Matrices/tables/c6685cbabcc7b9b2feb829736d7ec91fe069368eb417b5c2a4e61d988c08b74e.jpg)

![cab5929d8ab9a7c046f62912aaad6746ff72f0bea1a718ee9837b32435862c59.jpg](../iclr_results/2841_SMT_ Fine-Tuning Large Language Models with Sparse Matrices/tables/cab5929d8ab9a7c046f62912aaad6746ff72f0bea1a718ee9837b32435862c59.jpg)

![d5329155a0dbf10646b4bd3eb76856bb40a2967142a4cab4ec777dc6e74f700f.jpg](../iclr_results/2841_SMT_ Fine-Tuning Large Language Models with Sparse Matrices/tables/d5329155a0dbf10646b4bd3eb76856bb40a2967142a4cab4ec777dc6e74f700f.jpg)

## 3D-AffordanceLLM: Harnessing Large Language Models for Open-Vocabulary Affordance Detection in 3D Worlds


### Images

![305b234f21c961d2907344630a753881035f00187392069572044bae0b1245ce.jpg](../iclr_results/2842_3D-AffordanceLLM_ Harnessing Large Language Models for Open-Vocabulary Affordance Detection in 3D Wo/images/305b234f21c961d2907344630a753881035f00187392069572044bae0b1245ce.jpg)

![344d9aeee161c87afb7904785e9c2dd4e943e0cce8d2f909c1065ff4c9b633d0.jpg](../iclr_results/2842_3D-AffordanceLLM_ Harnessing Large Language Models for Open-Vocabulary Affordance Detection in 3D Wo/images/344d9aeee161c87afb7904785e9c2dd4e943e0cce8d2f909c1065ff4c9b633d0.jpg)

![57ebaa776caecc12376104506dc5fd5906d6b6a081336de08bc08b4b056dd431.jpg](../iclr_results/2842_3D-AffordanceLLM_ Harnessing Large Language Models for Open-Vocabulary Affordance Detection in 3D Wo/images/57ebaa776caecc12376104506dc5fd5906d6b6a081336de08bc08b4b056dd431.jpg)

![71493f670dd72bc858ed09878b39c5a27539bce1df39ce1155e8a8d9e9735181.jpg](../iclr_results/2842_3D-AffordanceLLM_ Harnessing Large Language Models for Open-Vocabulary Affordance Detection in 3D Wo/images/71493f670dd72bc858ed09878b39c5a27539bce1df39ce1155e8a8d9e9735181.jpg)

![749c3836f4f44d8e120ba80a9cadfd723e1da7f503f42db8dd3a69d591754a27.jpg](../iclr_results/2842_3D-AffordanceLLM_ Harnessing Large Language Models for Open-Vocabulary Affordance Detection in 3D Wo/images/749c3836f4f44d8e120ba80a9cadfd723e1da7f503f42db8dd3a69d591754a27.jpg)

![94f8f2f7eccf97e06b980474f6a74b60a2f5ab61b040613085c8a08932b41154.jpg](../iclr_results/2842_3D-AffordanceLLM_ Harnessing Large Language Models for Open-Vocabulary Affordance Detection in 3D Wo/images/94f8f2f7eccf97e06b980474f6a74b60a2f5ab61b040613085c8a08932b41154.jpg)

![9d273285f3e51a9938efc06a3e122cbc26aebf8159046ff0560de860d190e5be.jpg](../iclr_results/2842_3D-AffordanceLLM_ Harnessing Large Language Models for Open-Vocabulary Affordance Detection in 3D Wo/images/9d273285f3e51a9938efc06a3e122cbc26aebf8159046ff0560de860d190e5be.jpg)

### Tables

![2816523746014c01cbeb97714c3ed40568910338d8fec1634424b88c4006e32b.jpg](../iclr_results/2842_3D-AffordanceLLM_ Harnessing Large Language Models for Open-Vocabulary Affordance Detection in 3D Wo/tables/2816523746014c01cbeb97714c3ed40568910338d8fec1634424b88c4006e32b.jpg)

![50a5ad5c1a540548d13414650725f3a1994e7488d2ba592d61509c6549ee1220.jpg](../iclr_results/2842_3D-AffordanceLLM_ Harnessing Large Language Models for Open-Vocabulary Affordance Detection in 3D Wo/tables/50a5ad5c1a540548d13414650725f3a1994e7488d2ba592d61509c6549ee1220.jpg)

![6df4b83e4ad77d06d8cfe0e4478153b324287442ad6db3396ebb0aa2310c8353.jpg](../iclr_results/2842_3D-AffordanceLLM_ Harnessing Large Language Models for Open-Vocabulary Affordance Detection in 3D Wo/tables/6df4b83e4ad77d06d8cfe0e4478153b324287442ad6db3396ebb0aa2310c8353.jpg)

![8003e292ca33511e4efdf4bee74bdda222ad3af8ecfa05673ee017157ba4611c.jpg](../iclr_results/2842_3D-AffordanceLLM_ Harnessing Large Language Models for Open-Vocabulary Affordance Detection in 3D Wo/tables/8003e292ca33511e4efdf4bee74bdda222ad3af8ecfa05673ee017157ba4611c.jpg)

![80f8fea1001b83cfcf7a89e33f9f94a5d9741a5798b3dedf900ea60458948ec4.jpg](../iclr_results/2842_3D-AffordanceLLM_ Harnessing Large Language Models for Open-Vocabulary Affordance Detection in 3D Wo/tables/80f8fea1001b83cfcf7a89e33f9f94a5d9741a5798b3dedf900ea60458948ec4.jpg)

![98c2f60b1df20328fb20238c6a194e1e9633befd022c8810506949d85230e6f6.jpg](../iclr_results/2842_3D-AffordanceLLM_ Harnessing Large Language Models for Open-Vocabulary Affordance Detection in 3D Wo/tables/98c2f60b1df20328fb20238c6a194e1e9633befd022c8810506949d85230e6f6.jpg)

![bad816791d06b8ac1036bcbc0c1fbf6b57bf4ded0467baacbc36958107416fb4.jpg](../iclr_results/2842_3D-AffordanceLLM_ Harnessing Large Language Models for Open-Vocabulary Affordance Detection in 3D Wo/tables/bad816791d06b8ac1036bcbc0c1fbf6b57bf4ded0467baacbc36958107416fb4.jpg)

![df0fe433aa3f33985a1c1bd8c73369be3c74bc08df128c10f43fce999eb313aa.jpg](../iclr_results/2842_3D-AffordanceLLM_ Harnessing Large Language Models for Open-Vocabulary Affordance Detection in 3D Wo/tables/df0fe433aa3f33985a1c1bd8c73369be3c74bc08df128c10f43fce999eb313aa.jpg)

## Mutual Effort for Efficiency: A Similarity-based Token Pruning for Vision Transformers in Self-Supervised Learning


### Images

![61c4a894bffd0ca218415cc3027653b3bdfd0591160ca3f86b9ed0b4a2816373.jpg](../iclr_results/2843_Mutual Effort for Efficiency_ A Similarity-based Token Pruning for Vision Transformers in Self-Super/images/61c4a894bffd0ca218415cc3027653b3bdfd0591160ca3f86b9ed0b4a2816373.jpg)

![666be7a058785b5f09a808fc2a691ca9ea9c29d637f8c4997e155fe053249ae3.jpg](../iclr_results/2843_Mutual Effort for Efficiency_ A Similarity-based Token Pruning for Vision Transformers in Self-Super/images/666be7a058785b5f09a808fc2a691ca9ea9c29d637f8c4997e155fe053249ae3.jpg)

![8c2295fc0748d9c5fb1479ab50c7182bae810bc2e2fe62c38cf62ac71ab547b5.jpg](../iclr_results/2843_Mutual Effort for Efficiency_ A Similarity-based Token Pruning for Vision Transformers in Self-Super/images/8c2295fc0748d9c5fb1479ab50c7182bae810bc2e2fe62c38cf62ac71ab547b5.jpg)

![ed2fadf5162e149f24b3ea0ba351b6b50be4c619b35e5c31bc820812f925ead5.jpg](../iclr_results/2843_Mutual Effort for Efficiency_ A Similarity-based Token Pruning for Vision Transformers in Self-Super/images/ed2fadf5162e149f24b3ea0ba351b6b50be4c619b35e5c31bc820812f925ead5.jpg)

### Tables

![35a66e8c486be9335292bc5d19dcb10d68329fca05bc7eaf68b5aafd2b5cf542.jpg](../iclr_results/2843_Mutual Effort for Efficiency_ A Similarity-based Token Pruning for Vision Transformers in Self-Super/tables/35a66e8c486be9335292bc5d19dcb10d68329fca05bc7eaf68b5aafd2b5cf542.jpg)

![386a33774229671a58ac4a543b45ac91d13b091e6204dfe7e1f67368c2a9bc44.jpg](../iclr_results/2843_Mutual Effort for Efficiency_ A Similarity-based Token Pruning for Vision Transformers in Self-Super/tables/386a33774229671a58ac4a543b45ac91d13b091e6204dfe7e1f67368c2a9bc44.jpg)

![4d067ad15a3d5525237c3526004d317d75a6c6aa7af1c1c0356965cc9fe410a6.jpg](../iclr_results/2843_Mutual Effort for Efficiency_ A Similarity-based Token Pruning for Vision Transformers in Self-Super/tables/4d067ad15a3d5525237c3526004d317d75a6c6aa7af1c1c0356965cc9fe410a6.jpg)

![56a1fdf328ed62ad1f5383b66e18a2b69d9d5568b06b3d7f3cf521891e01fd7a.jpg](../iclr_results/2843_Mutual Effort for Efficiency_ A Similarity-based Token Pruning for Vision Transformers in Self-Super/tables/56a1fdf328ed62ad1f5383b66e18a2b69d9d5568b06b3d7f3cf521891e01fd7a.jpg)

![7c1f422ddf2d3f95e3963a9892e1ae286654ef5919a9292ced5cf443eca43f94.jpg](../iclr_results/2843_Mutual Effort for Efficiency_ A Similarity-based Token Pruning for Vision Transformers in Self-Super/tables/7c1f422ddf2d3f95e3963a9892e1ae286654ef5919a9292ced5cf443eca43f94.jpg)

![a30451fca0755ef694a7850fedd877f04a8abdc612e554830985f722e1ddab2a.jpg](../iclr_results/2843_Mutual Effort for Efficiency_ A Similarity-based Token Pruning for Vision Transformers in Self-Super/tables/a30451fca0755ef694a7850fedd877f04a8abdc612e554830985f722e1ddab2a.jpg)

![a4e9647c9ceefc7412ff2dbb7084161a72d8f24aeb32d582f7c5d7b6e60ac824.jpg](../iclr_results/2843_Mutual Effort for Efficiency_ A Similarity-based Token Pruning for Vision Transformers in Self-Super/tables/a4e9647c9ceefc7412ff2dbb7084161a72d8f24aeb32d582f7c5d7b6e60ac824.jpg)

![b887c5ddd76112593740a56f4815d70db0e6494967a86ff44f0dd54e89873afe.jpg](../iclr_results/2843_Mutual Effort for Efficiency_ A Similarity-based Token Pruning for Vision Transformers in Self-Super/tables/b887c5ddd76112593740a56f4815d70db0e6494967a86ff44f0dd54e89873afe.jpg)

![bc11cd2cd74e0fb320e46e4b3a618b1ed4a092db7802659946bf25d88fd08010.jpg](../iclr_results/2843_Mutual Effort for Efficiency_ A Similarity-based Token Pruning for Vision Transformers in Self-Super/tables/bc11cd2cd74e0fb320e46e4b3a618b1ed4a092db7802659946bf25d88fd08010.jpg)

![c1919f99620fe00cab70a57c1720ae53d8885c65395de939ce386c4b79bb87fa.jpg](../iclr_results/2843_Mutual Effort for Efficiency_ A Similarity-based Token Pruning for Vision Transformers in Self-Super/tables/c1919f99620fe00cab70a57c1720ae53d8885c65395de939ce386c4b79bb87fa.jpg)

![c5ad27196caac897953d09981e0a3bf1a1e2919a36acc6f3fb2b44e58a834885.jpg](../iclr_results/2843_Mutual Effort for Efficiency_ A Similarity-based Token Pruning for Vision Transformers in Self-Super/tables/c5ad27196caac897953d09981e0a3bf1a1e2919a36acc6f3fb2b44e58a834885.jpg)

![e5690f308919bf2ec24c98863c9e1a2afb1dcb8dc1381e7c5c4f3f5b95796e7b.jpg](../iclr_results/2843_Mutual Effort for Efficiency_ A Similarity-based Token Pruning for Vision Transformers in Self-Super/tables/e5690f308919bf2ec24c98863c9e1a2afb1dcb8dc1381e7c5c4f3f5b95796e7b.jpg)

## Bridging Compressed Image Latents and Multimodal Large Language Models


### Images

![050dc14836c0f20c93e50333a3ae670d1c5ea3e1693f8b93e65106fc5e2cc5a5.jpg](../iclr_results/2844_Bridging Compressed Image Latents and Multimodal Large Language Models/images/050dc14836c0f20c93e50333a3ae670d1c5ea3e1693f8b93e65106fc5e2cc5a5.jpg)

![2e31de9d5bc6cef8b0900b28b8fa5d0b0a3a8c5ddc368a148f92c858146d4025.jpg](../iclr_results/2844_Bridging Compressed Image Latents and Multimodal Large Language Models/images/2e31de9d5bc6cef8b0900b28b8fa5d0b0a3a8c5ddc368a148f92c858146d4025.jpg)

![3c4142f44f548b23851a29fae0ca1a5667913dd2b6aa3d3f5784e3a2c1f37f32.jpg](../iclr_results/2844_Bridging Compressed Image Latents and Multimodal Large Language Models/images/3c4142f44f548b23851a29fae0ca1a5667913dd2b6aa3d3f5784e3a2c1f37f32.jpg)

![4c3d0b582005a44b0489d0197ae420bb575e1651595fbb31f0193ab51e82ff8f.jpg](../iclr_results/2844_Bridging Compressed Image Latents and Multimodal Large Language Models/images/4c3d0b582005a44b0489d0197ae420bb575e1651595fbb31f0193ab51e82ff8f.jpg)

![5740871bbd8703a51737c2576a6e77ed6909f6ef2c166d15d992159fed6cfd1a.jpg](../iclr_results/2844_Bridging Compressed Image Latents and Multimodal Large Language Models/images/5740871bbd8703a51737c2576a6e77ed6909f6ef2c166d15d992159fed6cfd1a.jpg)

![5c1df1ef4d1bc6e79b51f288f9d6ace468a14ed6eb7aca57bad89e1773c2e369.jpg](../iclr_results/2844_Bridging Compressed Image Latents and Multimodal Large Language Models/images/5c1df1ef4d1bc6e79b51f288f9d6ace468a14ed6eb7aca57bad89e1773c2e369.jpg)

![6c20bcdd1729e936ee506da2a6e2d5a166ebe8f999a6a539f9241f2aef0cd6a9.jpg](../iclr_results/2844_Bridging Compressed Image Latents and Multimodal Large Language Models/images/6c20bcdd1729e936ee506da2a6e2d5a166ebe8f999a6a539f9241f2aef0cd6a9.jpg)

![79cb8fbfaf01555f8effca88169d97400b3184764c3ef63a89afacb981049dd1.jpg](../iclr_results/2844_Bridging Compressed Image Latents and Multimodal Large Language Models/images/79cb8fbfaf01555f8effca88169d97400b3184764c3ef63a89afacb981049dd1.jpg)

![871a75c44f86419649cdf164a7fef4a6bf90374b29319ae65f9190b0708cf02a.jpg](../iclr_results/2844_Bridging Compressed Image Latents and Multimodal Large Language Models/images/871a75c44f86419649cdf164a7fef4a6bf90374b29319ae65f9190b0708cf02a.jpg)

![96a3545cc9fca699a4c64c217a1fd8e68004b7dc472fc42759255b0045868438.jpg](../iclr_results/2844_Bridging Compressed Image Latents and Multimodal Large Language Models/images/96a3545cc9fca699a4c64c217a1fd8e68004b7dc472fc42759255b0045868438.jpg)

![9b54628be0b9b5d8044360a187b3534aabeb2245cbca81f0a32030f64ef23be0.jpg](../iclr_results/2844_Bridging Compressed Image Latents and Multimodal Large Language Models/images/9b54628be0b9b5d8044360a187b3534aabeb2245cbca81f0a32030f64ef23be0.jpg)

![a5f3bf7011a3502ce11c5e9393f8e1efe631a24390c00fb3563330c5e3fc7681.jpg](../iclr_results/2844_Bridging Compressed Image Latents and Multimodal Large Language Models/images/a5f3bf7011a3502ce11c5e9393f8e1efe631a24390c00fb3563330c5e3fc7681.jpg)

![b7f98533e20e0ca4bb48d1c3747af7646af523cb0e9e3c16b12c3512a5d212d1.jpg](../iclr_results/2844_Bridging Compressed Image Latents and Multimodal Large Language Models/images/b7f98533e20e0ca4bb48d1c3747af7646af523cb0e9e3c16b12c3512a5d212d1.jpg)

![be161abfbd82da60ed16cb2c7de95b6d11cf293148d241113a7a0c9fc8331ed3.jpg](../iclr_results/2844_Bridging Compressed Image Latents and Multimodal Large Language Models/images/be161abfbd82da60ed16cb2c7de95b6d11cf293148d241113a7a0c9fc8331ed3.jpg)

![ee4edd7540834a88fd472976e750c1e5dca1b80cd152afa88850602530d1a396.jpg](../iclr_results/2844_Bridging Compressed Image Latents and Multimodal Large Language Models/images/ee4edd7540834a88fd472976e750c1e5dca1b80cd152afa88850602530d1a396.jpg)

![fed455f44a846f5e4081ca8fb6e9092aa9323b3e441e31ea979481cc85d89397.jpg](../iclr_results/2844_Bridging Compressed Image Latents and Multimodal Large Language Models/images/fed455f44a846f5e4081ca8fb6e9092aa9323b3e441e31ea979481cc85d89397.jpg)

### Tables

![440438249d016fd2e6c49a3e398d823c39886f12e7f60a48a91b11fff2ecba1f.jpg](../iclr_results/2844_Bridging Compressed Image Latents and Multimodal Large Language Models/tables/440438249d016fd2e6c49a3e398d823c39886f12e7f60a48a91b11fff2ecba1f.jpg)

![bf2e3ccdfcccb449bc09da7ad2d2a257f29e7fb52b9ba79abaa5de9f2453c4af.jpg](../iclr_results/2844_Bridging Compressed Image Latents and Multimodal Large Language Models/tables/bf2e3ccdfcccb449bc09da7ad2d2a257f29e7fb52b9ba79abaa5de9f2453c4af.jpg)

![c52f2c9c5882e8dcdbdf982a17259f87a8dac56b0f46687b51033e7cb6140dbd.jpg](../iclr_results/2844_Bridging Compressed Image Latents and Multimodal Large Language Models/tables/c52f2c9c5882e8dcdbdf982a17259f87a8dac56b0f46687b51033e7cb6140dbd.jpg)

![db98500fdf1eb4980cdbcdac8875e6a310a4bff577bc705cc6c0074102ba5134.jpg](../iclr_results/2844_Bridging Compressed Image Latents and Multimodal Large Language Models/tables/db98500fdf1eb4980cdbcdac8875e6a310a4bff577bc705cc6c0074102ba5134.jpg)

## MAPS: Advancing Multi-Modal Reasoning in Expert-Level Physical Science


### Images

![0251e52c6af6c5c44d8933c3cd91cb2cf5bc6a4822d39269bc4cca5434199332.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/images/0251e52c6af6c5c44d8933c3cd91cb2cf5bc6a4822d39269bc4cca5434199332.jpg)

![096d73cf7dde4ebc6bd4fd0275f6382703374e4e6a310f0e12601ccf8026976d.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/images/096d73cf7dde4ebc6bd4fd0275f6382703374e4e6a310f0e12601ccf8026976d.jpg)

![0aa17ab249684106c01c66f9064260ba402859d89200f043873bd20f9e35915b.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/images/0aa17ab249684106c01c66f9064260ba402859d89200f043873bd20f9e35915b.jpg)

![22b0c3b115cbb3bac136994ada07ae5986e13c7e7e4dd0f964a47997c713f275.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/images/22b0c3b115cbb3bac136994ada07ae5986e13c7e7e4dd0f964a47997c713f275.jpg)

![3bd78b99459c6338f7953a8cc388b9a7f19a38728bb80a35d206e64eaa42fdff.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/images/3bd78b99459c6338f7953a8cc388b9a7f19a38728bb80a35d206e64eaa42fdff.jpg)

![49f42ff72aaeede1972ab032fe0df97b57c04dd2b2da5c5941a492c6d23c77e9.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/images/49f42ff72aaeede1972ab032fe0df97b57c04dd2b2da5c5941a492c6d23c77e9.jpg)

![4f8130c0b290df849e6e7a5959a42339d5a4f9a8f77248cc87096736583c1c12.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/images/4f8130c0b290df849e6e7a5959a42339d5a4f9a8f77248cc87096736583c1c12.jpg)

![53b9e4db82794692440b68b40e373b927bf5dbe0d43de03d063030890904e034.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/images/53b9e4db82794692440b68b40e373b927bf5dbe0d43de03d063030890904e034.jpg)

![5bdca788c31ec2354f9ec6b64cc9d311e1f3e27a106fbe34229e8fc1e47ee385.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/images/5bdca788c31ec2354f9ec6b64cc9d311e1f3e27a106fbe34229e8fc1e47ee385.jpg)

![711b58d964159ce57db3dbeb5c3ebe864a91986b51335bcc4a1c026600cc8297.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/images/711b58d964159ce57db3dbeb5c3ebe864a91986b51335bcc4a1c026600cc8297.jpg)

![739b3772f8034836a326c850fb3e25366afca523239b93bffc0bdcd87c958cdf.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/images/739b3772f8034836a326c850fb3e25366afca523239b93bffc0bdcd87c958cdf.jpg)

![789f38e48e4f2a1ad7c60619825f9f5f56decc1517e030324f453d3deee73a31.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/images/789f38e48e4f2a1ad7c60619825f9f5f56decc1517e030324f453d3deee73a31.jpg)

![78ca7198e9a567eab355d26cd2e5849cd001a0045182dc82acd7228edd0d9412.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/images/78ca7198e9a567eab355d26cd2e5849cd001a0045182dc82acd7228edd0d9412.jpg)

![84f599b41878ee1f35d4807fd0bf1ca93b61f49f5c0fddb82ee0d65fdc306d6b.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/images/84f599b41878ee1f35d4807fd0bf1ca93b61f49f5c0fddb82ee0d65fdc306d6b.jpg)

![85b0b469c05ed6069a0cd68ae847ab03fb070e5de5f79b4d6dc7a628b12afb2f.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/images/85b0b469c05ed6069a0cd68ae847ab03fb070e5de5f79b4d6dc7a628b12afb2f.jpg)

![97b3e9803481258bfd51e2a866c72cedd92df33b7dff2c40cb5b7a4cbdb65660.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/images/97b3e9803481258bfd51e2a866c72cedd92df33b7dff2c40cb5b7a4cbdb65660.jpg)

![b4cee2235a45044438f99905fb8ac9e3db1c23572f928d2ebbc9f87e4899a0c1.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/images/b4cee2235a45044438f99905fb8ac9e3db1c23572f928d2ebbc9f87e4899a0c1.jpg)

![c78a1e517a92db2ac9f9aa7803e317990fbc742388372bffd913cfc06bd8cce2.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/images/c78a1e517a92db2ac9f9aa7803e317990fbc742388372bffd913cfc06bd8cce2.jpg)

![e7c2e813a6e7f9f2d06c74f5a16f6f425a90fb48a1502cd3fb55de0ecfcf8786.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/images/e7c2e813a6e7f9f2d06c74f5a16f6f425a90fb48a1502cd3fb55de0ecfcf8786.jpg)

![ef77ef31d593db5ca75d0beae7d59393447d1aa2b93bb78adc3522f152dd0d9e.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/images/ef77ef31d593db5ca75d0beae7d59393447d1aa2b93bb78adc3522f152dd0d9e.jpg)

### Tables

![134d4e587e23f99421bc3536f368da7a8f9aea69746ff05c1951169d7193563b.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/tables/134d4e587e23f99421bc3536f368da7a8f9aea69746ff05c1951169d7193563b.jpg)

![2fa7fc1512617e237ed5de6ec50a392c1eb44ebf4dd25cf3ec87d3d8d3228ac5.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/tables/2fa7fc1512617e237ed5de6ec50a392c1eb44ebf4dd25cf3ec87d3d8d3228ac5.jpg)

![33fc208f626b1afc69ce88e8cca1c2f66c7acc079b02826b7580ebaccd9812c6.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/tables/33fc208f626b1afc69ce88e8cca1c2f66c7acc079b02826b7580ebaccd9812c6.jpg)

![779d5a4a6945c3407702266e521c5ec537c7fc4ef5b9d0f1d58ad9a7acb45325.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/tables/779d5a4a6945c3407702266e521c5ec537c7fc4ef5b9d0f1d58ad9a7acb45325.jpg)

![8d8b5da8e1908e53d252db909ffefa52cfa94cdebee37f2c20d98cb3c329c63d.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/tables/8d8b5da8e1908e53d252db909ffefa52cfa94cdebee37f2c20d98cb3c329c63d.jpg)

![ad92cbf3a1f10808251aaa21cc06c30a3ee11ae7355930f627ccf2fb88660400.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/tables/ad92cbf3a1f10808251aaa21cc06c30a3ee11ae7355930f627ccf2fb88660400.jpg)

![c8b470987899e55a44aaec510d3917c8944b300b4e8a2cfbedd3242917a281ab.jpg](../iclr_results/2845_MAPS_ Advancing Multi-Modal Reasoning in Expert-Level Physical Science/tables/c8b470987899e55a44aaec510d3917c8944b300b4e8a2cfbedd3242917a281ab.jpg)

## Weakly Supervised Video Scene Graph Generation via Natural Language Supervision


### Images

![0a5926e49bf5a3b032b21c1553cd8702ef1ded2b98fcdb94fce63cd6fed996ac.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/images/0a5926e49bf5a3b032b21c1553cd8702ef1ded2b98fcdb94fce63cd6fed996ac.jpg)

![0ca9800754c0e4d75c0a9a96cb7ed1dafa7f24932cecbf90ddd694a717372ccb.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/images/0ca9800754c0e4d75c0a9a96cb7ed1dafa7f24932cecbf90ddd694a717372ccb.jpg)

![2a9d9f00bd1545313d3ca333222032fb81b609c816ae9bd215ba41f0902a1a1f.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/images/2a9d9f00bd1545313d3ca333222032fb81b609c816ae9bd215ba41f0902a1a1f.jpg)

![3f80a8bfc99a48cf25d6cd737235af9582fe0abfb0e5d2d529a29a9e3672ab33.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/images/3f80a8bfc99a48cf25d6cd737235af9582fe0abfb0e5d2d529a29a9e3672ab33.jpg)

![546a07964220db83ab888ac205c05cd15339cc95d5b03d8e8def28581ad8dab4.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/images/546a07964220db83ab888ac205c05cd15339cc95d5b03d8e8def28581ad8dab4.jpg)

![6df18dca2a4295dded66aa3b886111e12732f41c221780890bf482d47e055fc9.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/images/6df18dca2a4295dded66aa3b886111e12732f41c221780890bf482d47e055fc9.jpg)

![b0e4f9e97dc970e0e109e1ab47daf6337da61dfa581c69ae5c4f2f00b3128c04.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/images/b0e4f9e97dc970e0e109e1ab47daf6337da61dfa581c69ae5c4f2f00b3128c04.jpg)

![d200a7448a8c6673b353d742ea0b11727737af67305a523e7f2a5dbf4995f4a2.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/images/d200a7448a8c6673b353d742ea0b11727737af67305a523e7f2a5dbf4995f4a2.jpg)

![dc7c888ba8cf5ae329106ef2e37f14e719f42c8df76ec3ae5cacac4f0873022d.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/images/dc7c888ba8cf5ae329106ef2e37f14e719f42c8df76ec3ae5cacac4f0873022d.jpg)

![e87a7e3a30a636748767cfbc0f254e8ab9efe11b8788803b27ba2242a0d429dc.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/images/e87a7e3a30a636748767cfbc0f254e8ab9efe11b8788803b27ba2242a0d429dc.jpg)

![f2f668c3f534e6a6e6fa143515bbb0b267beb1f32d3844e1e05862b12014884c.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/images/f2f668c3f534e6a6e6fa143515bbb0b267beb1f32d3844e1e05862b12014884c.jpg)

![fa058210e6dd28b43fc8bbd3eb51e446d3f4bde284d0b9b45a4a2d6d0e138739.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/images/fa058210e6dd28b43fc8bbd3eb51e446d3f4bde284d0b9b45a4a2d6d0e138739.jpg)

### Tables

![19ba16c706cd6a6b9857d69a5c86a8d59084a243cf680d32b725b3b87d03d6d5.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/tables/19ba16c706cd6a6b9857d69a5c86a8d59084a243cf680d32b725b3b87d03d6d5.jpg)

![30c79a1befe271e94a5e2f4a9511630f6e2ec931265d5a50301c2eb426a4bd85.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/tables/30c79a1befe271e94a5e2f4a9511630f6e2ec931265d5a50301c2eb426a4bd85.jpg)

![399dbc7067a5f22c5aee24ff9c94a7d8afbfdd5835e9479cd8864a348aa6d8c2.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/tables/399dbc7067a5f22c5aee24ff9c94a7d8afbfdd5835e9479cd8864a348aa6d8c2.jpg)

![5e1a96cecb852652746767a9e7b0437ed3d4e0065188bc2f22ba6e65462ea3d9.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/tables/5e1a96cecb852652746767a9e7b0437ed3d4e0065188bc2f22ba6e65462ea3d9.jpg)

![6338aff3ee35ee0a86e3921bbf5fd912b48d8ae761b89c7e10e5242cba71a877.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/tables/6338aff3ee35ee0a86e3921bbf5fd912b48d8ae761b89c7e10e5242cba71a877.jpg)

![71ecc737038cb7a7be87ae7c441ff31c45b3ed702ba1c3e4bf3d9a7abbdee718.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/tables/71ecc737038cb7a7be87ae7c441ff31c45b3ed702ba1c3e4bf3d9a7abbdee718.jpg)

![871d7d2eb1fa4b867cd3271189c39dc62237c9cf9b03ff2c769fa63832444f47.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/tables/871d7d2eb1fa4b867cd3271189c39dc62237c9cf9b03ff2c769fa63832444f47.jpg)

![8d61ee2401bf7f700dde8e608bbcf2e9a312833826e84501406d85d6bfe2b75a.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/tables/8d61ee2401bf7f700dde8e608bbcf2e9a312833826e84501406d85d6bfe2b75a.jpg)

![8e34f30dc884b328967a5d09ac448babe0340b5f3f44f5878e21f053944c9a7c.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/tables/8e34f30dc884b328967a5d09ac448babe0340b5f3f44f5878e21f053944c9a7c.jpg)

![b97928d0cdce47d45ac8c0487b4d7f2dda302b693c19956607c17fcad46a4876.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/tables/b97928d0cdce47d45ac8c0487b4d7f2dda302b693c19956607c17fcad46a4876.jpg)

![baa8ca34a80ae65252aa6324b5983326e7a7592fc01d481e826f9bce1bd3a08f.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/tables/baa8ca34a80ae65252aa6324b5983326e7a7592fc01d481e826f9bce1bd3a08f.jpg)

![c2f0848febe120bf16ec60114e69e34b464d4718032058dbfb35431a0e625902.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/tables/c2f0848febe120bf16ec60114e69e34b464d4718032058dbfb35431a0e625902.jpg)

![eed105466766febcd11b65192816168567a440d512ed8d656fb568b3f2d5c3af.jpg](../iclr_results/2846_Weakly Supervised Video Scene Graph Generation via Natural Language Supervision/tables/eed105466766febcd11b65192816168567a440d512ed8d656fb568b3f2d5c3af.jpg)

## Optimizing $(L_0, L_1)$-Smooth Functions by Gradient Methods


### Images

![279e8dd638d5754c383c18ce31a2b5d165073b78b79b2c91940cef6703bf367e.jpg](../iclr_results/2847_Optimizing $(L_0, L_1)$-Smooth Functions by Gradient Methods/images/279e8dd638d5754c383c18ce31a2b5d165073b78b79b2c91940cef6703bf367e.jpg)

![e74627e3e14b5ffa8ea322b606262a35f16ff44dd40f80cc09c5d91df0d355e3.jpg](../iclr_results/2847_Optimizing $(L_0, L_1)$-Smooth Functions by Gradient Methods/images/e74627e3e14b5ffa8ea322b606262a35f16ff44dd40f80cc09c5d91df0d355e3.jpg)

![f2e2fda6718750c32427d22b660d108a47080e8b5df2272bf082c4f0b4af80ef.jpg](../iclr_results/2847_Optimizing $(L_0, L_1)$-Smooth Functions by Gradient Methods/images/f2e2fda6718750c32427d22b660d108a47080e8b5df2272bf082c4f0b4af80ef.jpg)

### Tables

![839d102460b025f093c09ca187a1d98102e8e9f5f2818fba370204517f6877ac.jpg](../iclr_results/2847_Optimizing $(L_0, L_1)$-Smooth Functions by Gradient Methods/tables/839d102460b025f093c09ca187a1d98102e8e9f5f2818fba370204517f6877ac.jpg)

## Interpreting the Second-Order Effects of Neurons in CLIP


### Images

![0753d5e3d89c45c58d0abeea86689a34301cf69d16b3e42efae34fd428a38f05.jpg](../iclr_results/2848_Interpreting the Second-Order Effects of Neurons in CLIP/images/0753d5e3d89c45c58d0abeea86689a34301cf69d16b3e42efae34fd428a38f05.jpg)

![10abc8d2ff4c786755318529fc9a2196a7486edfe4f1acfde24c9fb2a9e7f5b4.jpg](../iclr_results/2848_Interpreting the Second-Order Effects of Neurons in CLIP/images/10abc8d2ff4c786755318529fc9a2196a7486edfe4f1acfde24c9fb2a9e7f5b4.jpg)

![16884216d9d44e5a87454f54a54e3ecb90495ce1ea03e1a32c9fc2660cb2cf60.jpg](../iclr_results/2848_Interpreting the Second-Order Effects of Neurons in CLIP/images/16884216d9d44e5a87454f54a54e3ecb90495ce1ea03e1a32c9fc2660cb2cf60.jpg)

![2d7e601fd9d7376afc1bc16f546742764e589c947a1b4b97d3236f9711df11fe.jpg](../iclr_results/2848_Interpreting the Second-Order Effects of Neurons in CLIP/images/2d7e601fd9d7376afc1bc16f546742764e589c947a1b4b97d3236f9711df11fe.jpg)

![61cc45fc66d1f6ac86a889c63a30fd0cd768d9e4f1905822d3c375b42c8f02a4.jpg](../iclr_results/2848_Interpreting the Second-Order Effects of Neurons in CLIP/images/61cc45fc66d1f6ac86a889c63a30fd0cd768d9e4f1905822d3c375b42c8f02a4.jpg)

![6580984da49a92adb60e0784db09e68079bb90c18270e14c3060a7b66ecf513f.jpg](../iclr_results/2848_Interpreting the Second-Order Effects of Neurons in CLIP/images/6580984da49a92adb60e0784db09e68079bb90c18270e14c3060a7b66ecf513f.jpg)

![7b182327283491c4b52dac9717651c08314ddac501f89c13b74d6e3334f66e94.jpg](../iclr_results/2848_Interpreting the Second-Order Effects of Neurons in CLIP/images/7b182327283491c4b52dac9717651c08314ddac501f89c13b74d6e3334f66e94.jpg)

![88af2e7ddb494088e7a9378e6ad3260ec1e9f7564ee1f02a3adb9f2695453d08.jpg](../iclr_results/2848_Interpreting the Second-Order Effects of Neurons in CLIP/images/88af2e7ddb494088e7a9378e6ad3260ec1e9f7564ee1f02a3adb9f2695453d08.jpg)

![a363cba8944b0f8cae782fb21fecbcd185215041d16a6c4b0a6a465183b7375a.jpg](../iclr_results/2848_Interpreting the Second-Order Effects of Neurons in CLIP/images/a363cba8944b0f8cae782fb21fecbcd185215041d16a6c4b0a6a465183b7375a.jpg)

![bd977ff6e3f12f706cab05f7788dbea8e40a5a439c608f939c00d76a05c45ee1.jpg](../iclr_results/2848_Interpreting the Second-Order Effects of Neurons in CLIP/images/bd977ff6e3f12f706cab05f7788dbea8e40a5a439c608f939c00d76a05c45ee1.jpg)

![beba03f35e54f41e3a420b765a09efe313159090f8cbed7631d249c483228ab3.jpg](../iclr_results/2848_Interpreting the Second-Order Effects of Neurons in CLIP/images/beba03f35e54f41e3a420b765a09efe313159090f8cbed7631d249c483228ab3.jpg)

![c96c333f5b370c08e2eee0a59d5dfc74686d030cb7f44cfe04827f07c6334dac.jpg](../iclr_results/2848_Interpreting the Second-Order Effects of Neurons in CLIP/images/c96c333f5b370c08e2eee0a59d5dfc74686d030cb7f44cfe04827f07c6334dac.jpg)

![e29ce4bceb8c8e537737284146404d6c331d89775ddb8c39dd7c94278322891d.jpg](../iclr_results/2848_Interpreting the Second-Order Effects of Neurons in CLIP/images/e29ce4bceb8c8e537737284146404d6c331d89775ddb8c39dd7c94278322891d.jpg)

![ff26a5102ac563531e066799ed952ecb85bb862aa8cb0f08460ff88c54a9a212.jpg](../iclr_results/2848_Interpreting the Second-Order Effects of Neurons in CLIP/images/ff26a5102ac563531e066799ed952ecb85bb862aa8cb0f08460ff88c54a9a212.jpg)

### Tables

![399b1bb8c9976bd22239a4a5b3f52179ad47938701df40345e362ee1791f45bd.jpg](../iclr_results/2848_Interpreting the Second-Order Effects of Neurons in CLIP/tables/399b1bb8c9976bd22239a4a5b3f52179ad47938701df40345e362ee1791f45bd.jpg)

![5897819d2da8c97aac3e9e906c04984048097b39b000d7c7877589fca56f858a.jpg](../iclr_results/2848_Interpreting the Second-Order Effects of Neurons in CLIP/tables/5897819d2da8c97aac3e9e906c04984048097b39b000d7c7877589fca56f858a.jpg)

![7bb68f39454b62af03eb14246f00af50430b88c254ca0a3bd9dfdfed4b9e92e8.jpg](../iclr_results/2848_Interpreting the Second-Order Effects of Neurons in CLIP/tables/7bb68f39454b62af03eb14246f00af50430b88c254ca0a3bd9dfdfed4b9e92e8.jpg)

![b0c6a7b61e408531209cfe820593fae9f0958ee79f1e13edf9bf470803ee508f.jpg](../iclr_results/2848_Interpreting the Second-Order Effects of Neurons in CLIP/tables/b0c6a7b61e408531209cfe820593fae9f0958ee79f1e13edf9bf470803ee508f.jpg)

![eb0471b1fa08d36c55a406dcbe18969eefe63e4c676dfa1d6fa1421467a87ce4.jpg](../iclr_results/2848_Interpreting the Second-Order Effects of Neurons in CLIP/tables/eb0471b1fa08d36c55a406dcbe18969eefe63e4c676dfa1d6fa1421467a87ce4.jpg)

![f5b8a564edbbc809cb2323821d03224dec7432761fe734051b81d90c4feb9f51.jpg](../iclr_results/2848_Interpreting the Second-Order Effects of Neurons in CLIP/tables/f5b8a564edbbc809cb2323821d03224dec7432761fe734051b81d90c4feb9f51.jpg)

## SaLoRA: Safety-Alignment Preserved Low-Rank Adaptation


### Images

![0cbebb741c32391335d990dd5ad0430b7f0d01af794eb7b55ed2cab5ae29fdc8.jpg](../iclr_results/2849_SaLoRA_ Safety-Alignment Preserved Low-Rank Adaptation/images/0cbebb741c32391335d990dd5ad0430b7f0d01af794eb7b55ed2cab5ae29fdc8.jpg)

![34b242fd21ad3895029aa5b5eee985fc9c7cc2a0edf420bc809809b506ed0e45.jpg](../iclr_results/2849_SaLoRA_ Safety-Alignment Preserved Low-Rank Adaptation/images/34b242fd21ad3895029aa5b5eee985fc9c7cc2a0edf420bc809809b506ed0e45.jpg)

![56c6a36c6f2670d9c048e3f9b153c4c03dddcdcf913b6d1ff71365ddb46f4a5f.jpg](../iclr_results/2849_SaLoRA_ Safety-Alignment Preserved Low-Rank Adaptation/images/56c6a36c6f2670d9c048e3f9b153c4c03dddcdcf913b6d1ff71365ddb46f4a5f.jpg)

![79739786b4a5d5df7d6d8139735ecaf35ddaa20425b03ec77221b3b1b4ad5d55.jpg](../iclr_results/2849_SaLoRA_ Safety-Alignment Preserved Low-Rank Adaptation/images/79739786b4a5d5df7d6d8139735ecaf35ddaa20425b03ec77221b3b1b4ad5d55.jpg)

![a815551d17230d09f2b934979407678053727b04e8da508fc7100f46e4bf3669.jpg](../iclr_results/2849_SaLoRA_ Safety-Alignment Preserved Low-Rank Adaptation/images/a815551d17230d09f2b934979407678053727b04e8da508fc7100f46e4bf3669.jpg)

![aca2bc0634733ae3574a078886ae97e925b54b2204c9fbd7923d927b9e5ef5c7.jpg](../iclr_results/2849_SaLoRA_ Safety-Alignment Preserved Low-Rank Adaptation/images/aca2bc0634733ae3574a078886ae97e925b54b2204c9fbd7923d927b9e5ef5c7.jpg)

![b685833f70349f1a928c3f3395bf18c25e1298a7ee255b6dcd7f3f4e3a492bdd.jpg](../iclr_results/2849_SaLoRA_ Safety-Alignment Preserved Low-Rank Adaptation/images/b685833f70349f1a928c3f3395bf18c25e1298a7ee255b6dcd7f3f4e3a492bdd.jpg)

### Tables

![101bcd5472e4999651e86947553c928816e3a1729a0f3926aaf767f33ec0e628.jpg](../iclr_results/2849_SaLoRA_ Safety-Alignment Preserved Low-Rank Adaptation/tables/101bcd5472e4999651e86947553c928816e3a1729a0f3926aaf767f33ec0e628.jpg)

![3f3ba292f637dad93e76926506d9c9afb8c927e3edf960cd9833603f48191d8d.jpg](../iclr_results/2849_SaLoRA_ Safety-Alignment Preserved Low-Rank Adaptation/tables/3f3ba292f637dad93e76926506d9c9afb8c927e3edf960cd9833603f48191d8d.jpg)

![893e43470407fbbffa994d1e6bc57f613f3403e20047c6ce83c6648cc1645311.jpg](../iclr_results/2849_SaLoRA_ Safety-Alignment Preserved Low-Rank Adaptation/tables/893e43470407fbbffa994d1e6bc57f613f3403e20047c6ce83c6648cc1645311.jpg)

![a85b39e41f3422578ed22d24125c073ddedb82cd9f13b42cddb1dcf14ae90a71.jpg](../iclr_results/2849_SaLoRA_ Safety-Alignment Preserved Low-Rank Adaptation/tables/a85b39e41f3422578ed22d24125c073ddedb82cd9f13b42cddb1dcf14ae90a71.jpg)

![e19d749a64d7cff8997223b592a312a474f140f48502ef88282e321bd781c6f8.jpg](../iclr_results/2849_SaLoRA_ Safety-Alignment Preserved Low-Rank Adaptation/tables/e19d749a64d7cff8997223b592a312a474f140f48502ef88282e321bd781c6f8.jpg)

![e67bd76457b1dc47986acebba971908433152365499fcc13180bb65445688cfd.jpg](../iclr_results/2849_SaLoRA_ Safety-Alignment Preserved Low-Rank Adaptation/tables/e67bd76457b1dc47986acebba971908433152365499fcc13180bb65445688cfd.jpg)

## Rethinking Neural Multi-Objective Combinatorial Optimization via Neat Weight Embedding


### Images

![57b09bccb8d16270b87118c9993457e0b2ed69ea54bb77aaeb22abe0c70f04a1.jpg](../iclr_results/2850_Rethinking Neural Multi-Objective Combinatorial Optimization via Neat Weight Embedding/images/57b09bccb8d16270b87118c9993457e0b2ed69ea54bb77aaeb22abe0c70f04a1.jpg)

![73b5aa436a0101cb18ee54f0c38c1b6ad4fb1714af37da0beb8a63c1ef3a44c5.jpg](../iclr_results/2850_Rethinking Neural Multi-Objective Combinatorial Optimization via Neat Weight Embedding/images/73b5aa436a0101cb18ee54f0c38c1b6ad4fb1714af37da0beb8a63c1ef3a44c5.jpg)

![74b27cc929701747709523d0acfb2b72f0d46a03458904bd4afd7940cabc747a.jpg](../iclr_results/2850_Rethinking Neural Multi-Objective Combinatorial Optimization via Neat Weight Embedding/images/74b27cc929701747709523d0acfb2b72f0d46a03458904bd4afd7940cabc747a.jpg)

![8a087a58d9849312f82fda6de28a6a6ce460a3c7e2a3e3b7b64489bec53b1ea7.jpg](../iclr_results/2850_Rethinking Neural Multi-Objective Combinatorial Optimization via Neat Weight Embedding/images/8a087a58d9849312f82fda6de28a6a6ce460a3c7e2a3e3b7b64489bec53b1ea7.jpg)

![9ff1dd3a2b77b322d912f9cf8d4e2751858ee5024261c56294e37e20185e751d.jpg](../iclr_results/2850_Rethinking Neural Multi-Objective Combinatorial Optimization via Neat Weight Embedding/images/9ff1dd3a2b77b322d912f9cf8d4e2751858ee5024261c56294e37e20185e751d.jpg)

![c4b56f76f3d712300897a7f4f1ff43673417d04bca7814f8c52d1f49532515ce.jpg](../iclr_results/2850_Rethinking Neural Multi-Objective Combinatorial Optimization via Neat Weight Embedding/images/c4b56f76f3d712300897a7f4f1ff43673417d04bca7814f8c52d1f49532515ce.jpg)

![cb1b00c87ff2318f94fa370a8e5ed6acaf54e2041bcd78b68d7d7252b3cbbd3d.jpg](../iclr_results/2850_Rethinking Neural Multi-Objective Combinatorial Optimization via Neat Weight Embedding/images/cb1b00c87ff2318f94fa370a8e5ed6acaf54e2041bcd78b68d7d7252b3cbbd3d.jpg)

![fba17758e0480c81a7ea60acb5917857e47ada07a845b0ae8cc7bb45db130e62.jpg](../iclr_results/2850_Rethinking Neural Multi-Objective Combinatorial Optimization via Neat Weight Embedding/images/fba17758e0480c81a7ea60acb5917857e47ada07a845b0ae8cc7bb45db130e62.jpg)

### Tables

![2dd4dee2ef34b67215aab6b8e671c397c69b1dea42b0a81b042556d5d52e18c0.jpg](../iclr_results/2850_Rethinking Neural Multi-Objective Combinatorial Optimization via Neat Weight Embedding/tables/2dd4dee2ef34b67215aab6b8e671c397c69b1dea42b0a81b042556d5d52e18c0.jpg)

![4405ba149a5a7df745357a2dbb48bb44c35f266146f7508ce013399b2de42132.jpg](../iclr_results/2850_Rethinking Neural Multi-Objective Combinatorial Optimization via Neat Weight Embedding/tables/4405ba149a5a7df745357a2dbb48bb44c35f266146f7508ce013399b2de42132.jpg)

![50bc6e789f41b7db4ba58e8bc27b3fd67bd9633e9d5c8a1263512a9fbb188cbb.jpg](../iclr_results/2850_Rethinking Neural Multi-Objective Combinatorial Optimization via Neat Weight Embedding/tables/50bc6e789f41b7db4ba58e8bc27b3fd67bd9633e9d5c8a1263512a9fbb188cbb.jpg)

![51c6ad16817c880cd658249756277cc9c4f37336d25317e4138f851660afa7e6.jpg](../iclr_results/2850_Rethinking Neural Multi-Objective Combinatorial Optimization via Neat Weight Embedding/tables/51c6ad16817c880cd658249756277cc9c4f37336d25317e4138f851660afa7e6.jpg)

![71d0d7e63c7a01fd535904c15a36adeba654425a4893049a6b5f40ab3973cfbe.jpg](../iclr_results/2850_Rethinking Neural Multi-Objective Combinatorial Optimization via Neat Weight Embedding/tables/71d0d7e63c7a01fd535904c15a36adeba654425a4893049a6b5f40ab3973cfbe.jpg)

![855cc75792cd9219465ce86c2e3384a090e2f7a4818cd382df4539f2627edbcc.jpg](../iclr_results/2850_Rethinking Neural Multi-Objective Combinatorial Optimization via Neat Weight Embedding/tables/855cc75792cd9219465ce86c2e3384a090e2f7a4818cd382df4539f2627edbcc.jpg)

![8634f876719ee1c7cc38a50286aa8b5940e4c830a5f6d5b11992be417a7cd979.jpg](../iclr_results/2850_Rethinking Neural Multi-Objective Combinatorial Optimization via Neat Weight Embedding/tables/8634f876719ee1c7cc38a50286aa8b5940e4c830a5f6d5b11992be417a7cd979.jpg)

![98a0b68dfc9e87cdb94863c0c1a1eaab2094281d5fdf4913c413561c62c113fe.jpg](../iclr_results/2850_Rethinking Neural Multi-Objective Combinatorial Optimization via Neat Weight Embedding/tables/98a0b68dfc9e87cdb94863c0c1a1eaab2094281d5fdf4913c413561c62c113fe.jpg)

![d1cfa91cc4938b59a51fd1ac1caccb6273b5cabfe0982918a9e6028b63d20469.jpg](../iclr_results/2850_Rethinking Neural Multi-Objective Combinatorial Optimization via Neat Weight Embedding/tables/d1cfa91cc4938b59a51fd1ac1caccb6273b5cabfe0982918a9e6028b63d20469.jpg)

![d9744a0e055269303534c14775aec2f15fc45978c30b216ac6c5b8d20213e676.jpg](../iclr_results/2850_Rethinking Neural Multi-Objective Combinatorial Optimization via Neat Weight Embedding/tables/d9744a0e055269303534c14775aec2f15fc45978c30b216ac6c5b8d20213e676.jpg)

## Skill Expansion and Composition in Parameter Space


### Images

![06cd8e3009b1796cbe6a6e4e8af569a4e9c501c6f9c4e5e753e394eae7c5d23d.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/images/06cd8e3009b1796cbe6a6e4e8af569a4e9c501c6f9c4e5e753e394eae7c5d23d.jpg)

![1079bb3e537d7f935d3155b043047eb10a67bb18c5755f2cf1eff2601068d995.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/images/1079bb3e537d7f935d3155b043047eb10a67bb18c5755f2cf1eff2601068d995.jpg)

![186ca03f275040ef4ae7dee67a853959c51e9048c6b2eb01b47aceaa0b6b3f6f.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/images/186ca03f275040ef4ae7dee67a853959c51e9048c6b2eb01b47aceaa0b6b3f6f.jpg)

![1b13cfabf6ea48641fa8bb91f761810e6c2664583d77bdacf3d8b8a12d09026b.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/images/1b13cfabf6ea48641fa8bb91f761810e6c2664583d77bdacf3d8b8a12d09026b.jpg)

![27e594141b490bf29192fd0c1d9e217730ac8516bb1559e12ba112bbf7aa464f.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/images/27e594141b490bf29192fd0c1d9e217730ac8516bb1559e12ba112bbf7aa464f.jpg)

![2f079326e99cfbeaa931d2cd61185dfdf19bb96324a789ead85db5cb74c7d5ca.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/images/2f079326e99cfbeaa931d2cd61185dfdf19bb96324a789ead85db5cb74c7d5ca.jpg)

![53b8adc9517db395292a9d0996fd1a96356ecaad8677d67c8993ebfb20d835b6.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/images/53b8adc9517db395292a9d0996fd1a96356ecaad8677d67c8993ebfb20d835b6.jpg)

![6002fd8943b99191fcd50e8bc81f3259e754bf2f6d2982de2adc3979a7119bad.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/images/6002fd8943b99191fcd50e8bc81f3259e754bf2f6d2982de2adc3979a7119bad.jpg)

![a44396ef18c23ac33099fca2f8a417f53c3f6c7900fd113b87d37ced353a19f3.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/images/a44396ef18c23ac33099fca2f8a417f53c3f6c7900fd113b87d37ced353a19f3.jpg)

![a7a3ba645a6c6354dde043adc1ce59f8b335b7d56122a66175e04572c2e6bc56.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/images/a7a3ba645a6c6354dde043adc1ce59f8b335b7d56122a66175e04572c2e6bc56.jpg)

![ae465fe134d5017c5e7ba254ece4672ed77410ed5b6c8e6354a29cc5e276e95e.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/images/ae465fe134d5017c5e7ba254ece4672ed77410ed5b6c8e6354a29cc5e276e95e.jpg)

![c6690162a2021b068b008d5733cad13679210650e40f1a15ded145af4c0b465f.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/images/c6690162a2021b068b008d5733cad13679210650e40f1a15ded145af4c0b465f.jpg)

![d5a7901ca2ab2b8d1a207474e031ff63c6bbe190d556aadbe90c696f0015cd36.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/images/d5a7901ca2ab2b8d1a207474e031ff63c6bbe190d556aadbe90c696f0015cd36.jpg)

![d6eb12b4f4fbab1596a6bfb37ac5aef0ad556c334be5e854fd98dc0e0127c811.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/images/d6eb12b4f4fbab1596a6bfb37ac5aef0ad556c334be5e854fd98dc0e0127c811.jpg)

![e0937b7ab7303b0d83f64e09e47b0f174d05ca35853994d3a6490ddac7a6dd39.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/images/e0937b7ab7303b0d83f64e09e47b0f174d05ca35853994d3a6490ddac7a6dd39.jpg)

![e6d9a63728b35adf5551b5b761902ae0e5afafe5099658fb846a5f2c6280eb1a.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/images/e6d9a63728b35adf5551b5b761902ae0e5afafe5099658fb846a5f2c6280eb1a.jpg)

![e6ef9bd70ec538c7c2c1932a4d744226835b7f55da3517f8ba89bf6ac3f3c2a1.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/images/e6ef9bd70ec538c7c2c1932a4d744226835b7f55da3517f8ba89bf6ac3f3c2a1.jpg)

![f20128a5b37efcdf653b9e32d37593ba0f0eb2f8900c4cf07d65a048487afb8e.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/images/f20128a5b37efcdf653b9e32d37593ba0f0eb2f8900c4cf07d65a048487afb8e.jpg)

![f3def10baefdc9fde1fc0151d79ce7687ace76c7ca88e415864c3aaa1fd1a64b.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/images/f3def10baefdc9fde1fc0151d79ce7687ace76c7ca88e415864c3aaa1fd1a64b.jpg)

![f4f7b601ba739d70239bd5b047e9143bec8fb8823e68c4fa4e0e72f37059b6b6.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/images/f4f7b601ba739d70239bd5b047e9143bec8fb8823e68c4fa4e0e72f37059b6b6.jpg)

### Tables

![041921d849d7808b3659d37729c32c614b02e105d8904c1a1cbb1f4a45bd2f1d.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/tables/041921d849d7808b3659d37729c32c614b02e105d8904c1a1cbb1f4a45bd2f1d.jpg)

![6c2f3404ed6c68ff95f4ba4f5f8ce2fca99ec27f5fe9d8730fb854b1bca95dff.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/tables/6c2f3404ed6c68ff95f4ba4f5f8ce2fca99ec27f5fe9d8730fb854b1bca95dff.jpg)

![712bc96a10c8a6488ec3c8cf5d5f2db88fa61a05fc26d2e4195550b4744c57a4.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/tables/712bc96a10c8a6488ec3c8cf5d5f2db88fa61a05fc26d2e4195550b4744c57a4.jpg)

![753c199d6458ac3a50a674e1e103b552b879acf36b2df643cd23befdd12f52cd.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/tables/753c199d6458ac3a50a674e1e103b552b879acf36b2df643cd23befdd12f52cd.jpg)

![9e35ae567f24eb7913b8bf35db4a425428a9c067b98801ef9b36228132301afe.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/tables/9e35ae567f24eb7913b8bf35db4a425428a9c067b98801ef9b36228132301afe.jpg)

![a200b06c8e499130de978654b3a6d1895837eeb9996c7610186fe0b44273821c.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/tables/a200b06c8e499130de978654b3a6d1895837eeb9996c7610186fe0b44273821c.jpg)

![ac9c2933092a4c5f4f96080b93ed7a23f25504854952a33464a495e83538a8de.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/tables/ac9c2933092a4c5f4f96080b93ed7a23f25504854952a33464a495e83538a8de.jpg)

![b52da12eced117c3cc6fefa426b150ca2ea8748b26e33ea00558af990cc4b9b8.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/tables/b52da12eced117c3cc6fefa426b150ca2ea8748b26e33ea00558af990cc4b9b8.jpg)

![babc8264953c533ec37fceff4948cd8cb6065a39f70bb9fed40951ab7e4f33e6.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/tables/babc8264953c533ec37fceff4948cd8cb6065a39f70bb9fed40951ab7e4f33e6.jpg)

![bf0c29cce35b6f6ecda81d3423f574379784f8a4955b45f0b23e6698999e0a9e.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/tables/bf0c29cce35b6f6ecda81d3423f574379784f8a4955b45f0b23e6698999e0a9e.jpg)

![c96af399f853891c16bb2adf47f6310909d81ba72e5c1d4f6faed60473322c76.jpg](../iclr_results/2851_Skill Expansion and Composition in Parameter Space/tables/c96af399f853891c16bb2adf47f6310909d81ba72e5c1d4f6faed60473322c76.jpg)

## ADMM for Nonconvex Optimization under Minimal Continuity Assumption


### Images

![211ec44ca11536497772a8e421ea624f01cbb937b2f54b4824cf9c5ad0203f4b.jpg](../iclr_results/2852_ADMM for Nonconvex Optimization under Minimal Continuity Assumption/images/211ec44ca11536497772a8e421ea624f01cbb937b2f54b4824cf9c5ad0203f4b.jpg)

![3978edda696e3bb9335594552e007fa37b6f29ed78fb62512b48d979aeb4e412.jpg](../iclr_results/2852_ADMM for Nonconvex Optimization under Minimal Continuity Assumption/images/3978edda696e3bb9335594552e007fa37b6f29ed78fb62512b48d979aeb4e412.jpg)

![5e98ee4fb90183c2fc2c6808f6769c3cd06b78863c66713ab5bcabf1bd3865f6.jpg](../iclr_results/2852_ADMM for Nonconvex Optimization under Minimal Continuity Assumption/images/5e98ee4fb90183c2fc2c6808f6769c3cd06b78863c66713ab5bcabf1bd3865f6.jpg)

![5ea843c513e0f52bc83b3edd9d4af14616a51067e570ab9b5e722b40711855d5.jpg](../iclr_results/2852_ADMM for Nonconvex Optimization under Minimal Continuity Assumption/images/5ea843c513e0f52bc83b3edd9d4af14616a51067e570ab9b5e722b40711855d5.jpg)

![69997dee9160fc2133530df008c50ee2c40d2c0e8c152832b2ecf3b82267ee53.jpg](../iclr_results/2852_ADMM for Nonconvex Optimization under Minimal Continuity Assumption/images/69997dee9160fc2133530df008c50ee2c40d2c0e8c152832b2ecf3b82267ee53.jpg)

![6b089362d3e0565c16504425c34b00bb631715b18545ecf7f73a4c9bd90b941d.jpg](../iclr_results/2852_ADMM for Nonconvex Optimization under Minimal Continuity Assumption/images/6b089362d3e0565c16504425c34b00bb631715b18545ecf7f73a4c9bd90b941d.jpg)

![6bc3456c32b424946e1a5064e47099a08d83b461b2e27073bd6d09c369df3000.jpg](../iclr_results/2852_ADMM for Nonconvex Optimization under Minimal Continuity Assumption/images/6bc3456c32b424946e1a5064e47099a08d83b461b2e27073bd6d09c369df3000.jpg)

![6f08db20ea29bbf706ace97153d3ed345294e612a949f0cda9e518abbc29ac8c.jpg](../iclr_results/2852_ADMM for Nonconvex Optimization under Minimal Continuity Assumption/images/6f08db20ea29bbf706ace97153d3ed345294e612a949f0cda9e518abbc29ac8c.jpg)

![710b1743a275ea052c377003267cfaedc765755c7959305f87558f929870c057.jpg](../iclr_results/2852_ADMM for Nonconvex Optimization under Minimal Continuity Assumption/images/710b1743a275ea052c377003267cfaedc765755c7959305f87558f929870c057.jpg)

![98a77ddeefe861f5a2bea76d22b48342a7bf0c82c3930b6ac53fd306017a3848.jpg](../iclr_results/2852_ADMM for Nonconvex Optimization under Minimal Continuity Assumption/images/98a77ddeefe861f5a2bea76d22b48342a7bf0c82c3930b6ac53fd306017a3848.jpg)

![c6b37b39e01774ffb2e7077c06bd87099764076e193b236cdbe281e420521308.jpg](../iclr_results/2852_ADMM for Nonconvex Optimization under Minimal Continuity Assumption/images/c6b37b39e01774ffb2e7077c06bd87099764076e193b236cdbe281e420521308.jpg)

![dbf101e4efce9cabeee6a8707ac68b2e6b003b0c983a55265de1688a6d63442f.jpg](../iclr_results/2852_ADMM for Nonconvex Optimization under Minimal Continuity Assumption/images/dbf101e4efce9cabeee6a8707ac68b2e6b003b0c983a55265de1688a6d63442f.jpg)

![e01f7a17969680761be4853002c6e9c6932403ad24bc0de5b940c7a23f46b9a6.jpg](../iclr_results/2852_ADMM for Nonconvex Optimization under Minimal Continuity Assumption/images/e01f7a17969680761be4853002c6e9c6932403ad24bc0de5b940c7a23f46b9a6.jpg)

![e4f3d4038199c647df2ff577c38d38b9a8597cb7016b8a2f8b4266287e55c9dc.jpg](../iclr_results/2852_ADMM for Nonconvex Optimization under Minimal Continuity Assumption/images/e4f3d4038199c647df2ff577c38d38b9a8597cb7016b8a2f8b4266287e55c9dc.jpg)

![f2934bdc18f525a1ca4fb8d9b10ae30364f60f78d919eef70955dd172afed34f.jpg](../iclr_results/2852_ADMM for Nonconvex Optimization under Minimal Continuity Assumption/images/f2934bdc18f525a1ca4fb8d9b10ae30364f60f78d919eef70955dd172afed34f.jpg)

![f9edee4bb5bf7852d51e488a46a697d6111557a5e6ebdf01c5f1547c7ea54b45.jpg](../iclr_results/2852_ADMM for Nonconvex Optimization under Minimal Continuity Assumption/images/f9edee4bb5bf7852d51e488a46a697d6111557a5e6ebdf01c5f1547c7ea54b45.jpg)

![facdfe972669a13dd6de72159f57b313a53890afbc2b149d6a90e3834dba05c3.jpg](../iclr_results/2852_ADMM for Nonconvex Optimization under Minimal Continuity Assumption/images/facdfe972669a13dd6de72159f57b313a53890afbc2b149d6a90e3834dba05c3.jpg)

### Tables

![0799c7e653c8e7b765e7f79300f9b40191f56fcc00bbb4fe5c1af537fe748e5d.jpg](../iclr_results/2852_ADMM for Nonconvex Optimization under Minimal Continuity Assumption/tables/0799c7e653c8e7b765e7f79300f9b40191f56fcc00bbb4fe5c1af537fe748e5d.jpg)

![ab0ae857f7471d8d9ff2bbd1b40591a186031796f01e2492307a8707615a425a.jpg](../iclr_results/2852_ADMM for Nonconvex Optimization under Minimal Continuity Assumption/tables/ab0ae857f7471d8d9ff2bbd1b40591a186031796f01e2492307a8707615a425a.jpg)

![d4c90938c7f4f32cd2f8c206162c0f41f7ec98f550a61c9c73734ac84138eff0.jpg](../iclr_results/2852_ADMM for Nonconvex Optimization under Minimal Continuity Assumption/tables/d4c90938c7f4f32cd2f8c206162c0f41f7ec98f550a61c9c73734ac84138eff0.jpg)

## TFG-Flow: Training-free Guidance in Multimodal Generative Flow


### Images

![4ca081637e824ab0e05ad617f36ed4c1e5ed940dcd1622ca3e2a2bdb6e4be741.jpg](../iclr_results/2853_TFG-Flow_ Training-free Guidance in Multimodal Generative Flow/images/4ca081637e824ab0e05ad617f36ed4c1e5ed940dcd1622ca3e2a2bdb6e4be741.jpg)

![f695d44c20abb8315b34079a017040eaa3b9e4593b147b4b04e526dcc526962f.jpg](../iclr_results/2853_TFG-Flow_ Training-free Guidance in Multimodal Generative Flow/images/f695d44c20abb8315b34079a017040eaa3b9e4593b147b4b04e526dcc526962f.jpg)

### Tables

![0314ccf572a88790cf4633de37754b459fc0c4cd3450ea7577b1c4908bd2d02c.jpg](../iclr_results/2853_TFG-Flow_ Training-free Guidance in Multimodal Generative Flow/tables/0314ccf572a88790cf4633de37754b459fc0c4cd3450ea7577b1c4908bd2d02c.jpg)

![1407b5a69dbbe2cbbce3dc326fd6b47eed0ad82e8f35c9a7a7b84c85eb8bcc7f.jpg](../iclr_results/2853_TFG-Flow_ Training-free Guidance in Multimodal Generative Flow/tables/1407b5a69dbbe2cbbce3dc326fd6b47eed0ad82e8f35c9a7a7b84c85eb8bcc7f.jpg)

![2364a7ef110e87fd69b0053c12e0abfb7e58483946e384a448ce5aebc57c8777.jpg](../iclr_results/2853_TFG-Flow_ Training-free Guidance in Multimodal Generative Flow/tables/2364a7ef110e87fd69b0053c12e0abfb7e58483946e384a448ce5aebc57c8777.jpg)

![3da496319fe924a461305bc8025ebb62a33491c86e9606409f4e7592dbb2a1db.jpg](../iclr_results/2853_TFG-Flow_ Training-free Guidance in Multimodal Generative Flow/tables/3da496319fe924a461305bc8025ebb62a33491c86e9606409f4e7592dbb2a1db.jpg)

![5c009c8c526c332827ac1deb2b922d5b69a579aa7e349e3585bbda6bfdb9b334.jpg](../iclr_results/2853_TFG-Flow_ Training-free Guidance in Multimodal Generative Flow/tables/5c009c8c526c332827ac1deb2b922d5b69a579aa7e349e3585bbda6bfdb9b334.jpg)

![6346f4c88d0f1655ce4f8df675ef86d8702db8d453aa5eb60ef64c60f4b860f6.jpg](../iclr_results/2853_TFG-Flow_ Training-free Guidance in Multimodal Generative Flow/tables/6346f4c88d0f1655ce4f8df675ef86d8702db8d453aa5eb60ef64c60f4b860f6.jpg)

![b61d56dd2e368f4dd2d7da2078757612f7cd70b0846e32889e875ccc4d0dfaf5.jpg](../iclr_results/2853_TFG-Flow_ Training-free Guidance in Multimodal Generative Flow/tables/b61d56dd2e368f4dd2d7da2078757612f7cd70b0846e32889e875ccc4d0dfaf5.jpg)

![c389a8cebf8abfd1ade1ec2744694594f09426f28f272e092e2067b6ab99727b.jpg](../iclr_results/2853_TFG-Flow_ Training-free Guidance in Multimodal Generative Flow/tables/c389a8cebf8abfd1ade1ec2744694594f09426f28f272e092e2067b6ab99727b.jpg)

![e364dd17347e58b115a8ff0b5446e56f6908756b859537cbf0a571788ffe9b6b.jpg](../iclr_results/2853_TFG-Flow_ Training-free Guidance in Multimodal Generative Flow/tables/e364dd17347e58b115a8ff0b5446e56f6908756b859537cbf0a571788ffe9b6b.jpg)

## Efficient Active Imitation Learning with Random Network Distillation


### Images

![23766f632eab521985746ca4acee74cf6600f38716784de36a40d587c13766ef.jpg](../iclr_results/2855_Efficient Active Imitation Learning with Random Network Distillation/images/23766f632eab521985746ca4acee74cf6600f38716784de36a40d587c13766ef.jpg)

![502ec04d5c9b9920f9003963377a14312d74f836b062c7bb70de6ce8a30eb1f1.jpg](../iclr_results/2855_Efficient Active Imitation Learning with Random Network Distillation/images/502ec04d5c9b9920f9003963377a14312d74f836b062c7bb70de6ce8a30eb1f1.jpg)

![7192247d957f77150a937acbfd3a52a67da78907cbbfe5cdbca4cf7b8e98dafc.jpg](../iclr_results/2855_Efficient Active Imitation Learning with Random Network Distillation/images/7192247d957f77150a937acbfd3a52a67da78907cbbfe5cdbca4cf7b8e98dafc.jpg)

![7d31bd3b6ee204396a75a150961f145d86ba162ab88b8f5dda855aba9709a9a4.jpg](../iclr_results/2855_Efficient Active Imitation Learning with Random Network Distillation/images/7d31bd3b6ee204396a75a150961f145d86ba162ab88b8f5dda855aba9709a9a4.jpg)

![7f65da4e30603e836435b98758af6f8e8156fe37cb8f55cf049b9acc539c5f2b.jpg](../iclr_results/2855_Efficient Active Imitation Learning with Random Network Distillation/images/7f65da4e30603e836435b98758af6f8e8156fe37cb8f55cf049b9acc539c5f2b.jpg)

![a7dc85d943200e527a506111a99f2ad59efa1d45aacd8f08d8adf46d41365acc.jpg](../iclr_results/2855_Efficient Active Imitation Learning with Random Network Distillation/images/a7dc85d943200e527a506111a99f2ad59efa1d45aacd8f08d8adf46d41365acc.jpg)

![ab140f2fd3320cd686829eef45fc61701a5a09973fe7bfd291ed8556b0e96231.jpg](../iclr_results/2855_Efficient Active Imitation Learning with Random Network Distillation/images/ab140f2fd3320cd686829eef45fc61701a5a09973fe7bfd291ed8556b0e96231.jpg)

![aca3b5b336c4577b82450ea2126e61b268b5372ab4d4ebdf8635486631fabf6c.jpg](../iclr_results/2855_Efficient Active Imitation Learning with Random Network Distillation/images/aca3b5b336c4577b82450ea2126e61b268b5372ab4d4ebdf8635486631fabf6c.jpg)

![b70e9afb4ea5720fc251e33a54cfc8790ce2d7e9368010084643987b9bb75632.jpg](../iclr_results/2855_Efficient Active Imitation Learning with Random Network Distillation/images/b70e9afb4ea5720fc251e33a54cfc8790ce2d7e9368010084643987b9bb75632.jpg)

![c1a54b4d1e0cc8b1134190c4151133fb36d691629f1548685e3aec32065a6f78.jpg](../iclr_results/2855_Efficient Active Imitation Learning with Random Network Distillation/images/c1a54b4d1e0cc8b1134190c4151133fb36d691629f1548685e3aec32065a6f78.jpg)

![edde98d8606b39ff652706a524db470fc08236f93901225c9168ed0ebfd01630.jpg](../iclr_results/2855_Efficient Active Imitation Learning with Random Network Distillation/images/edde98d8606b39ff652706a524db470fc08236f93901225c9168ed0ebfd01630.jpg)

![f403dbb7b72bba117576c98d257e1f4bfa01474d7795599bc204d7a07a09187b.jpg](../iclr_results/2855_Efficient Active Imitation Learning with Random Network Distillation/images/f403dbb7b72bba117576c98d257e1f4bfa01474d7795599bc204d7a07a09187b.jpg)

### Tables

![25cb7974328928c7c79627e525acb6eb00c8c8e4f96ed0e0b921ab13d92d4084.jpg](../iclr_results/2855_Efficient Active Imitation Learning with Random Network Distillation/tables/25cb7974328928c7c79627e525acb6eb00c8c8e4f96ed0e0b921ab13d92d4084.jpg)

![5ff10d49c6aa4891439de9b03909d5da02477bc673cd913fe695e040dc3893a0.jpg](../iclr_results/2855_Efficient Active Imitation Learning with Random Network Distillation/tables/5ff10d49c6aa4891439de9b03909d5da02477bc673cd913fe695e040dc3893a0.jpg)

![74625ccea7cd96c48dfec509ac2dc338a5c62499754cc17d4269b312deff11ef.jpg](../iclr_results/2855_Efficient Active Imitation Learning with Random Network Distillation/tables/74625ccea7cd96c48dfec509ac2dc338a5c62499754cc17d4269b312deff11ef.jpg)

![76f6f1f598d09c8923b3ccb02f465b34f676277373a74bbed637f5f3ce30f66d.jpg](../iclr_results/2855_Efficient Active Imitation Learning with Random Network Distillation/tables/76f6f1f598d09c8923b3ccb02f465b34f676277373a74bbed637f5f3ce30f66d.jpg)

![80c1da4784902d7c69f192757ab2c4e6a7f5c2e769417d07c1ddf6a01aeff52c.jpg](../iclr_results/2855_Efficient Active Imitation Learning with Random Network Distillation/tables/80c1da4784902d7c69f192757ab2c4e6a7f5c2e769417d07c1ddf6a01aeff52c.jpg)

## Neural Functions for Learning Periodic Signal


### Images

![022a60ed2e0e94d8aac4245264699b4c86caba91fe3cd9cac7668f0b74c797ae.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/022a60ed2e0e94d8aac4245264699b4c86caba91fe3cd9cac7668f0b74c797ae.jpg)

![034cdc67759915c7fbe56a94ebdf088334f9bec7b3308e75a17b6caa37bc00b5.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/034cdc67759915c7fbe56a94ebdf088334f9bec7b3308e75a17b6caa37bc00b5.jpg)

![08908495b5f101e788af18f57b1291cde0d770a77bea70acd29f032751e2c2fb.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/08908495b5f101e788af18f57b1291cde0d770a77bea70acd29f032751e2c2fb.jpg)

![0c78e68a4d06c2a653f1dd2f0629b56aa8f52224b116b5c31935373932c80f1d.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/0c78e68a4d06c2a653f1dd2f0629b56aa8f52224b116b5c31935373932c80f1d.jpg)

![0d68c228f6f46d2b60112759b0aff3b72710d8098de07b425f94708ce464b9e1.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/0d68c228f6f46d2b60112759b0aff3b72710d8098de07b425f94708ce464b9e1.jpg)

![1ad079810881ea9249dfba558a35e3dbfe0d39fa31ae27a7aecd24506d810acf.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/1ad079810881ea9249dfba558a35e3dbfe0d39fa31ae27a7aecd24506d810acf.jpg)

![1fc04c754d8181d348738945ddbb57cda39cb65822cf541dec8acdfd22e91a1f.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/1fc04c754d8181d348738945ddbb57cda39cb65822cf541dec8acdfd22e91a1f.jpg)

![2cedcdf76b3f0cd8bdfaf533d5e6fa8a3d0e7598c5fc36a58c7a31b72960dcb0.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/2cedcdf76b3f0cd8bdfaf533d5e6fa8a3d0e7598c5fc36a58c7a31b72960dcb0.jpg)

![4f23f0bdd22ebb38f7571917385e1a3a8d53e3fbb20a93b161453fa1c4d744ea.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/4f23f0bdd22ebb38f7571917385e1a3a8d53e3fbb20a93b161453fa1c4d744ea.jpg)

![50843b2f5dc677ad9b527aa54d81a9039b5cc9d71e63d3135516599a71fda676.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/50843b2f5dc677ad9b527aa54d81a9039b5cc9d71e63d3135516599a71fda676.jpg)

![554a516d85f6993ce92ed4f56be29575604fc0198ce36c133921976807010bb4.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/554a516d85f6993ce92ed4f56be29575604fc0198ce36c133921976807010bb4.jpg)

![602aff55e0fba1556761e9709422754af88bffe76a2bc1e2508599bd41f6bbd6.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/602aff55e0fba1556761e9709422754af88bffe76a2bc1e2508599bd41f6bbd6.jpg)

![6b23906fb3919d9b0dda94e6f95d5ff74357e498890ba9604b4371f106bbb31a.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/6b23906fb3919d9b0dda94e6f95d5ff74357e498890ba9604b4371f106bbb31a.jpg)

![7103e01beec304f95403af3b70bcd790d49cdbc5af92a80f7ba88a81a77b39ec.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/7103e01beec304f95403af3b70bcd790d49cdbc5af92a80f7ba88a81a77b39ec.jpg)

![71748f06ee89d7db4582edc35e89d00f3a28cb65e4c15643d6d948b6437a3dd5.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/71748f06ee89d7db4582edc35e89d00f3a28cb65e4c15643d6d948b6437a3dd5.jpg)

![7c5d414b6e9200a2c0fc9c91fbba0c9a342a4e3ee973e9783d87ee63e032d58b.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/7c5d414b6e9200a2c0fc9c91fbba0c9a342a4e3ee973e9783d87ee63e032d58b.jpg)

![85df014431e791ba54a708df0dcee22a81598a0de7382c9b612166b3bdba4392.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/85df014431e791ba54a708df0dcee22a81598a0de7382c9b612166b3bdba4392.jpg)

![963374e7b0354027d450c613c2ffe383a7bbc4efa6cf6b6ee619a06dfcd2fe17.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/963374e7b0354027d450c613c2ffe383a7bbc4efa6cf6b6ee619a06dfcd2fe17.jpg)

![98da21017c9a189eacace497ff8ec50fc64422473d34a3ae3022e355e9eb452b.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/98da21017c9a189eacace497ff8ec50fc64422473d34a3ae3022e355e9eb452b.jpg)

![9930daf9fa5c1a437416b858e7449076c6213de1aa6e97e48e2cad4cdd124c01.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/9930daf9fa5c1a437416b858e7449076c6213de1aa6e97e48e2cad4cdd124c01.jpg)

![9dd6c2f9ca4bc832c610a935fb7abd15946fa7aec6e6093b1af30b11bdca40ad.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/9dd6c2f9ca4bc832c610a935fb7abd15946fa7aec6e6093b1af30b11bdca40ad.jpg)

![bb46afb33d24f1aeb71f340af262b3ebccdf985faa4b117c79c4c01d7e2f9385.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/bb46afb33d24f1aeb71f340af262b3ebccdf985faa4b117c79c4c01d7e2f9385.jpg)

![c36be98b5a1b5ee49b9b7e310995175da0d2d5f524edd5322b279290b6ca0618.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/c36be98b5a1b5ee49b9b7e310995175da0d2d5f524edd5322b279290b6ca0618.jpg)

![d1821c67fcf6c5369567cbbff1115c0febae144278a2cfed645116ff351065c1.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/d1821c67fcf6c5369567cbbff1115c0febae144278a2cfed645116ff351065c1.jpg)

![da1d5981e44bb5bdecbfa1fa76abed24bf80f5660a14260e5a7b2f992822b190.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/da1d5981e44bb5bdecbfa1fa76abed24bf80f5660a14260e5a7b2f992822b190.jpg)

![db906081b8bb4acd71f46b36dd3760185058d4ea3a87ab7762015e67a825493c.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/db906081b8bb4acd71f46b36dd3760185058d4ea3a87ab7762015e67a825493c.jpg)

![dd8f210edb15cf388f891326ade19713fc92d5745d0ff1868b97bb32b80aab7e.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/dd8f210edb15cf388f891326ade19713fc92d5745d0ff1868b97bb32b80aab7e.jpg)

![e1544e6042b875032b493242ed0cebef3cfe2438783a260c25d6f60cdffc0fcd.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/e1544e6042b875032b493242ed0cebef3cfe2438783a260c25d6f60cdffc0fcd.jpg)

![e3bf7f028d49745856af563df1e7d2130f2e339c801737ec6cc50ed726a3d48b.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/e3bf7f028d49745856af563df1e7d2130f2e339c801737ec6cc50ed726a3d48b.jpg)

![e3e063fbfeb75b58f7b8c152e1d6f248c1f1e6cc1863a722aa3295ccb55ce7e0.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/e3e063fbfeb75b58f7b8c152e1d6f248c1f1e6cc1863a722aa3295ccb55ce7e0.jpg)

![e6d5341a57e7cd41bdec87a7cc22016204f35e5d539568d7f2be6801b9ddff7a.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/e6d5341a57e7cd41bdec87a7cc22016204f35e5d539568d7f2be6801b9ddff7a.jpg)

![edfb0f6b370dcb1087c5de2a5b17c2532ee068062b3a42ee0c14cd7b959eac61.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/edfb0f6b370dcb1087c5de2a5b17c2532ee068062b3a42ee0c14cd7b959eac61.jpg)

![ef54b2bfd50d77c3c0e823d4bbc89126ccdbebe039b6da899b75194bd321924c.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/ef54b2bfd50d77c3c0e823d4bbc89126ccdbebe039b6da899b75194bd321924c.jpg)

![f252e6a96b960ede6d03ebd520184c6cab1408f3c42520836182744d6919ef79.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/f252e6a96b960ede6d03ebd520184c6cab1408f3c42520836182744d6919ef79.jpg)

![f94cae2ac9bddcbec554faace5ed7a01aae7ce2210e3f027435a7bb48d0ea319.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/images/f94cae2ac9bddcbec554faace5ed7a01aae7ce2210e3f027435a7bb48d0ea319.jpg)

### Tables

![16e0277c4c9f6de06889ff260e60f241f69048277ca5ba8ff8b8561bd89c2075.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/tables/16e0277c4c9f6de06889ff260e60f241f69048277ca5ba8ff8b8561bd89c2075.jpg)

![1bbdaad6b8a04fdce774836b46ae1946615437bc0bc4bcc1e08083eaed985df9.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/tables/1bbdaad6b8a04fdce774836b46ae1946615437bc0bc4bcc1e08083eaed985df9.jpg)

![2d33b809386440b740c51b77beeafa34f38e042bcf7df4b860b7d62f5443f860.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/tables/2d33b809386440b740c51b77beeafa34f38e042bcf7df4b860b7d62f5443f860.jpg)

![3960e760f0b38e14ccb34dfd8489b611e65381d209f5582494465c14cfcbbd23.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/tables/3960e760f0b38e14ccb34dfd8489b611e65381d209f5582494465c14cfcbbd23.jpg)

![4b9d58a63de1f6ee1fc9b3ae55011984f76e697a462f4145e0b1473ae8d1360c.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/tables/4b9d58a63de1f6ee1fc9b3ae55011984f76e697a462f4145e0b1473ae8d1360c.jpg)

![52eb81cabca65bad8b808c0a07e2fff5fc04ca340791e1af282af9a15956233c.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/tables/52eb81cabca65bad8b808c0a07e2fff5fc04ca340791e1af282af9a15956233c.jpg)

![534600932f0212a6e0022c4eea44abc4f5af0d6a8b81441b960ac09dd915dc7e.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/tables/534600932f0212a6e0022c4eea44abc4f5af0d6a8b81441b960ac09dd915dc7e.jpg)

![6d23f7ee9d97996be4dab2391a6303572fbbfd614a1f2752cc77f24bf1e6fbe1.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/tables/6d23f7ee9d97996be4dab2391a6303572fbbfd614a1f2752cc77f24bf1e6fbe1.jpg)

![7a54c5dfd805e7a398abab1535e424ee1460382213228e2ba919f4351ad74c6c.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/tables/7a54c5dfd805e7a398abab1535e424ee1460382213228e2ba919f4351ad74c6c.jpg)

![828766647635714064d9b391921a0b8ded33678f30050d46ab97dafdd85d80e4.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/tables/828766647635714064d9b391921a0b8ded33678f30050d46ab97dafdd85d80e4.jpg)

![86d08341c3e1bb416de7829912962f75fb8046ee1a9e754d31cdeebbe7f1e4dd.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/tables/86d08341c3e1bb416de7829912962f75fb8046ee1a9e754d31cdeebbe7f1e4dd.jpg)

![bcec0039f8c5f0e228c8f48b27c75639768c48dd7a272337524a293db3d77cdb.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/tables/bcec0039f8c5f0e228c8f48b27c75639768c48dd7a272337524a293db3d77cdb.jpg)

![c5af3cdc95feeaf02eb7b6712112f4380642761c18f81362fcb56d99e12e6834.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/tables/c5af3cdc95feeaf02eb7b6712112f4380642761c18f81362fcb56d99e12e6834.jpg)

![c6dca7522da332cf84490ebe145216c9f88520553458d5ca67940ac726d85aa2.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/tables/c6dca7522da332cf84490ebe145216c9f88520553458d5ca67940ac726d85aa2.jpg)

![d55ccb141494048b4fa26ab9e3dd8bdf6094000a52be27af6fc0fbe2bbe52cdb.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/tables/d55ccb141494048b4fa26ab9e3dd8bdf6094000a52be27af6fc0fbe2bbe52cdb.jpg)

![d8d372aa243c54f1f3a6fc09e18e52f796cd69191fd983894ec7b013d13a7524.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/tables/d8d372aa243c54f1f3a6fc09e18e52f796cd69191fd983894ec7b013d13a7524.jpg)

![e3c4a68d230cb1252c4f1b996fa5c334cd7a490f1be5e48392d9a4b642d8c15c.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/tables/e3c4a68d230cb1252c4f1b996fa5c334cd7a490f1be5e48392d9a4b642d8c15c.jpg)

![e438495f8857d4132385a624cac8202a0c5e8e90e1264e794989c8d8395dab44.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/tables/e438495f8857d4132385a624cac8202a0c5e8e90e1264e794989c8d8395dab44.jpg)

![f5b0ef1dab9d88dda8d9e77a68cfda80301facd0dc951037ef18b758b1cafac0.jpg](../iclr_results/2858_Neural Functions for Learning Periodic Signal/tables/f5b0ef1dab9d88dda8d9e77a68cfda80301facd0dc951037ef18b758b1cafac0.jpg)

## ExACT: Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning


### Images

![04ea3acd31945375ffaa3cc1b7556e7e96f1295757a378ad67e645b62ba669af.jpg](../iclr_results/2860_ExACT_ Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning/images/04ea3acd31945375ffaa3cc1b7556e7e96f1295757a378ad67e645b62ba669af.jpg)

![378c3342fdebd04d18c1ec1a3c999074d5f9cc7d207fa01a74b47ef58c26e230.jpg](../iclr_results/2860_ExACT_ Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning/images/378c3342fdebd04d18c1ec1a3c999074d5f9cc7d207fa01a74b47ef58c26e230.jpg)

![389d857522ecfe436d6aa9ad22394271682559435118254c247f02f68089eadd.jpg](../iclr_results/2860_ExACT_ Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning/images/389d857522ecfe436d6aa9ad22394271682559435118254c247f02f68089eadd.jpg)

![4726387e640a827d08bbc3a9135f2382c2c5be68321f5077540681039e02faca.jpg](../iclr_results/2860_ExACT_ Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning/images/4726387e640a827d08bbc3a9135f2382c2c5be68321f5077540681039e02faca.jpg)

![4c390332729990223d90811f483bb99c3b8da9edbc9f0f2a385d5ce91c902cbe.jpg](../iclr_results/2860_ExACT_ Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning/images/4c390332729990223d90811f483bb99c3b8da9edbc9f0f2a385d5ce91c902cbe.jpg)

![d5e6230d3361561a79c6dd4001b1778c3a324445c815173cd3e32f1d102e3857.jpg](../iclr_results/2860_ExACT_ Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning/images/d5e6230d3361561a79c6dd4001b1778c3a324445c815173cd3e32f1d102e3857.jpg)

![f3ff7489eb952d072d19a3c52d9fd5d3477b3c26409c10b167edd2598c25703b.jpg](../iclr_results/2860_ExACT_ Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning/images/f3ff7489eb952d072d19a3c52d9fd5d3477b3c26409c10b167edd2598c25703b.jpg)

### Tables

![0c833934151cbf5c49378f8e7ff2fe604f63c11639150bd003052cd877360fb2.jpg](../iclr_results/2860_ExACT_ Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning/tables/0c833934151cbf5c49378f8e7ff2fe604f63c11639150bd003052cd877360fb2.jpg)

![1673786315146dc859e503dc901b9f359de72555507427bff25f03ab151930f6.jpg](../iclr_results/2860_ExACT_ Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning/tables/1673786315146dc859e503dc901b9f359de72555507427bff25f03ab151930f6.jpg)

![29d5c71fcf15ef9f5dff96317620d49e38c78ef41b2358933f5781248eed94dd.jpg](../iclr_results/2860_ExACT_ Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning/tables/29d5c71fcf15ef9f5dff96317620d49e38c78ef41b2358933f5781248eed94dd.jpg)

![2ffffacf565cf84d906b25d18a16d934101d6fe425195430b11f5c45e119d0cf.jpg](../iclr_results/2860_ExACT_ Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning/tables/2ffffacf565cf84d906b25d18a16d934101d6fe425195430b11f5c45e119d0cf.jpg)

![55ba816154fee660a92166e760f97fd20ee53d81108ddc87b68b22fd8e5675bb.jpg](../iclr_results/2860_ExACT_ Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning/tables/55ba816154fee660a92166e760f97fd20ee53d81108ddc87b68b22fd8e5675bb.jpg)

![66713d2eb6384b8389c34c428454c9b69a133f61177f4c520674a55ee107e72c.jpg](../iclr_results/2860_ExACT_ Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning/tables/66713d2eb6384b8389c34c428454c9b69a133f61177f4c520674a55ee107e72c.jpg)

![7d5e1a8dcc930f6422bb23a52e947c3225c01850fd76559bac527c803717f5b9.jpg](../iclr_results/2860_ExACT_ Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning/tables/7d5e1a8dcc930f6422bb23a52e947c3225c01850fd76559bac527c803717f5b9.jpg)

![a76101bde2397cf55ad616f51cd152bcd0548280efd6e0239a7a11e0e78496d6.jpg](../iclr_results/2860_ExACT_ Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning/tables/a76101bde2397cf55ad616f51cd152bcd0548280efd6e0239a7a11e0e78496d6.jpg)

![bb6c1086a6c6727fdd0bc805d923eb25168f6b107978e3fd9305d02a40a847b2.jpg](../iclr_results/2860_ExACT_ Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning/tables/bb6c1086a6c6727fdd0bc805d923eb25168f6b107978e3fd9305d02a40a847b2.jpg)

![bf9d9e06116ca0f957afac592231f08cc3340c36896caf236d99704206cce2e6.jpg](../iclr_results/2860_ExACT_ Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning/tables/bf9d9e06116ca0f957afac592231f08cc3340c36896caf236d99704206cce2e6.jpg)

![d95d715cd1c45e376284062dd23940503a5357e3eb92206e3ee59240ce50c566.jpg](../iclr_results/2860_ExACT_ Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning/tables/d95d715cd1c45e376284062dd23940503a5357e3eb92206e3ee59240ce50c566.jpg)

![db049f247cec3a3328e0a9f03ba171c20d5c18dbf6fe3222a7dfdde9d91b09e0.jpg](../iclr_results/2860_ExACT_ Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning/tables/db049f247cec3a3328e0a9f03ba171c20d5c18dbf6fe3222a7dfdde9d91b09e0.jpg)

![e496c52d95dd7cfcb72b2ff43ebfbd51d165a502ccb91b26b6a579ad3f7c56df.jpg](../iclr_results/2860_ExACT_ Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning/tables/e496c52d95dd7cfcb72b2ff43ebfbd51d165a502ccb91b26b6a579ad3f7c56df.jpg)

![e688742a3869da38ea789224b3ab116df5fee42e139306ee10e567b20b931325.jpg](../iclr_results/2860_ExACT_ Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning/tables/e688742a3869da38ea789224b3ab116df5fee42e139306ee10e567b20b931325.jpg)

![f929f11957305520764c50b011324ae98a3b4a494270d629ecc3ac2dfaa3f442.jpg](../iclr_results/2860_ExACT_ Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning/tables/f929f11957305520764c50b011324ae98a3b4a494270d629ecc3ac2dfaa3f442.jpg)

## Singular Subspace Perturbation Bounds via Rectangular Random Matrix Diffusions

### Images

![52daf720ce8a033ce8637539bb26d8fbdc8e10f9af4c17cb684a2ce94073a128.jpg](../iclr_results/2861_Singular Subspace Perturbation Bounds via Rectangular Random Matrix Diffusions/images/52daf720ce8a033ce8637539bb26d8fbdc8e10f9af4c17cb684a2ce94073a128.jpg)

![bdfb44ccfd019c5afb57d4af5b35c53295491660a4e0800a989091dc7d3da5e5.jpg](../iclr_results/2861_Singular Subspace Perturbation Bounds via Rectangular Random Matrix Diffusions/images/bdfb44ccfd019c5afb57d4af5b35c53295491660a4e0800a989091dc7d3da5e5.jpg)

![cb122ae6040cf987a6f6153de7c8262cd0936a11a3add0689cb9d147ee5d61e6.jpg](../iclr_results/2861_Singular Subspace Perturbation Bounds via Rectangular Random Matrix Diffusions/images/cb122ae6040cf987a6f6153de7c8262cd0936a11a3add0689cb9d147ee5d61e6.jpg)

![cb8b7df5e32e1c9e287724e79bc184a9871a11b7c46a68c8e86ac29ea8a4d557.jpg](../iclr_results/2861_Singular Subspace Perturbation Bounds via Rectangular Random Matrix Diffusions/images/cb8b7df5e32e1c9e287724e79bc184a9871a11b7c46a68c8e86ac29ea8a4d557.jpg)

![e54f28ebabfcd76e89b237c69582dc0b428c8efe6b6c56b885bd9a0c0054ae9a.jpg](../iclr_results/2861_Singular Subspace Perturbation Bounds via Rectangular Random Matrix Diffusions/images/e54f28ebabfcd76e89b237c69582dc0b428c8efe6b6c56b885bd9a0c0054ae9a.jpg)

![f83e56adce3585953d14052d78e60cb9419b15129ccebcbca7e80a23edec6c31.jpg](../iclr_results/2861_Singular Subspace Perturbation Bounds via Rectangular Random Matrix Diffusions/images/f83e56adce3585953d14052d78e60cb9419b15129ccebcbca7e80a23edec6c31.jpg)
