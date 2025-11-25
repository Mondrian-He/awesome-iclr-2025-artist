# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 74 of 100

## 目录 (Table of Contents)

1. [Discretization-invariance? On the Discretization Mismatch Errors in Neural Operators](#Discretization-invariance-On-the-Discretization-Mismatch-Errors-in-Neural-Operators)
2. [PostEdit: Posterior Sampling for Efficient Zero-Shot Image Editing](#PostEdit-Posterior-Sampling-for-Efficient-Zero-Shot-Image-Editing)
3. [Exploiting Hidden Symmetry to Improve Objective Perturbation for DP Linear Learners with a Nonsmooth L1-Norm](#Exploiting-Hidden-Symmetry-to-Improve-Objective-Perturbation-for-DP-Linear-Learners-with-a-Nonsmooth-L1-Norm)
4. [RetroInText: A Multimodal Large Language Model Enhanced Framework for Retrosynthetic Planning via In-Context Representation Learning](#RetroInText-A-Multimodal-Large-Language-Model-Enhanced-Framework-for-Retrosynthetic-Planning-via-In-Context-Representation-Learning)
5. [LiNeS: Post-training Layer Scaling Prevents Forgetting and Enhances Model Merging](#LiNeS-Post-training-Layer-Scaling-Prevents-Forgetting-and-Enhances-Model-Merging)
6. [An Empirical Analysis of Uncertainty in Large Language Model Evaluations](#An-Empirical-Analysis-of-Uncertainty-in-Large-Language-Model-Evaluations)
7. [Learning Dynamics of Deep Matrix Factorization Beyond the Edge of Stability](#Learning-Dynamics-of-Deep-Matrix-Factorization-Beyond-the-Edge-of-Stability)
8. [RECAST: Reparameterized, Compact weight Adaptation for Sequential Tasks](#RECAST-Reparameterized-Compact-weight-Adaptation-for-Sequential-Tasks)
9. [MMSearch: Unveiling the Potential of Large Models as Multi-modal Search Engines](#MMSearch-Unveiling-the-Potential-of-Large-Models-as-Multi-modal-Search-Engines)
10. [ToolDial: Multi-turn Dialogue Generation Method for Tool-Augmented Language Models](#ToolDial-Multi-turn-Dialogue-Generation-Method-for-Tool-Augmented-Language-Models)
11. [GenARM: Reward Guided Generation with Autoregressive Reward Model for Test-Time Alignment](#GenARM-Reward-Guided-Generation-with-Autoregressive-Reward-Model-for-Test-Time-Alignment)
12. [Learning to Steer Markovian Agents under Model Uncertainty](#Learning-to-Steer-Markovian-Agents-under-Model-Uncertainty)
13. [What Makes a Maze Look Like a Maze?](#What-Makes-a-Maze-Look-Like-a-Maze)
14. [A Simple yet Effective $\Delta\Delta G$ Predictor is An Unsupervised Antibody Optimizer and Explainer](#A-Simple-yet-Effective-DeltaDelta-G-Predictor-is-An-Unsupervised-Antibody-Optimizer-and-Explainer)
15. [MedTrinity-25M: A Large-scale Multimodal Dataset with Multigranular Annotations for Medicine](#MedTrinity-25M-A-Large-scale-Multimodal-Dataset-with-Multigranular-Annotations-for-Medicine)
16. [Post-hoc Reward Calibration: A Case Study on Length Bias](#Post-hoc-Reward-Calibration-A-Case-Study-on-Length-Bias)
17. [PseDet: Revisiting the Power of Pseudo Label in Incremental Object Detection](#PseDet-Revisiting-the-Power-of-Pseudo-Label-in-Incremental-Object-Detection)
18. [Improving Reasoning Performance in Large Language Models via Representation Engineering](#Improving-Reasoning-Performance-in-Large-Language-Models-via-Representation-Engineering)
19. [Conditional Testing based on Localized Conformal $p$-values](#Conditional-Testing-based-on-Localized-Conformal-p-values)
20. [NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals](#NeuroLM-A-Universal-Multi-task-Foundation-Model-for-Bridging-the-Gap-between-Language-and-EEG-Signals)
21. [IV-mixed Sampler: Leveraging Image Diffusion Models for Enhanced Video Synthesis](#IV-mixed-Sampler-Leveraging-Image-Diffusion-Models-for-Enhanced-Video-Synthesis)
22. [Watermark Anything With Localized Messages](#Watermark-Anything-With-Localized-Messages)
23. [From Attention to Activation: Unraveling the Enigmas of Large Language Models](#From-Attention-to-Activation-Unraveling-the-Enigmas-of-Large-Language-Models)
24. [Bio-xLSTM: Generative modeling, representation and in-context learning of biological and chemical sequences](#Bio-xLSTM-Generative-modeling-representation-and-in-context-learning-of-biological-and-chemical-sequences)
25. [The Hyperfitting Phenomenon: Sharpening and Stabilizing LLMs for Open-Ended Text Generation](#The-Hyperfitting-Phenomenon-Sharpening-and-Stabilizing-LLMs-for-Open-Ended-Text-Generation)
26. [Memory Mosaics](#Memory-Mosaics)
27. [Start Smart: Leveraging Gradients For Enhancing Mask-based XAI Methods](#Start-Smart-Leveraging-Gradients-For-Enhancing-Mask-based-XAI-Methods)
28. [Controllable Context Sensitivity and the Knob Behind It](#Controllable-Context-Sensitivity-and-the-Knob-Behind-It)
29. [Intelligence at the Edge of Chaos](#Intelligence-at-the-Edge-of-Chaos)
30. [DynAlign: Unsupervised Dynamic Taxonomy Alignment for Cross-Domain Segmentation](#DynAlign-Unsupervised-Dynamic-Taxonomy-Alignment-for-Cross-Domain-Segmentation)
31. [Optimizing 4D Gaussians for Dynamic Scene Video from Single Landscape Images](#Optimizing-4D-Gaussians-for-Dynamic-Scene-Video-from-Single-Landscape-Images)
32. [Improving Language Model Distillation through Hidden State Matching](#Improving-Language-Model-Distillation-through-Hidden-State-Matching)
33. [Efficient Model-Based Reinforcement Learning Through Optimistic Thompson Sampling](#Efficient-Model-Based-Reinforcement-Learning-Through-Optimistic-Thompson-Sampling)
34. [What's New in My Data? Novelty Exploration via Contrastive Generation](#Whats-New-in-My-Data-Novelty-Exploration-via-Contrastive-Generation)
35. [Learning Color Equivariant Representations](#Learning-Color-Equivariant-Representations)
36. [LeanQuant: Accurate and Scalable Large Language Model Quantization with Loss-error-aware Grid](#LeanQuant-Accurate-and-Scalable-Large-Language-Model-Quantization-with-Loss-error-aware-Grid)
37. [Generalization v.s. Memorization: Tracing Language Models’ Capabilities Back to Pretraining Data](#Generalization-vs-Memorization-Tracing-Language-Models-Capabilities-Back-to-Pretraining-Data)

---


## Discretization-invariance? On the Discretization Mismatch Errors in Neural Operators

### Images

![0b202235b7866033f2aa8fde0139206c470ae2fb7137b166e3b39fdcf54d5375.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/images/0b202235b7866033f2aa8fde0139206c470ae2fb7137b166e3b39fdcf54d5375.jpg)

![0b337fa267377099c9f9d74743ead5cb04ffb93d75ed3f4b738e448f93138826.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/images/0b337fa267377099c9f9d74743ead5cb04ffb93d75ed3f4b738e448f93138826.jpg)

![1699105a24c986393d109859ebac2e6d625e3b910fbe4f421603a7235b201b13.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/images/1699105a24c986393d109859ebac2e6d625e3b910fbe4f421603a7235b201b13.jpg)

![18bfb4e4d0452fa6fbc0f59edeaaa4a0880509a7dfc8edc98a74b58a045aa80b.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/images/18bfb4e4d0452fa6fbc0f59edeaaa4a0880509a7dfc8edc98a74b58a045aa80b.jpg)

![204daca7415271b95a6a2fd29b581ad24eb7fd33ddcc181037e0e78bbb463e4c.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/images/204daca7415271b95a6a2fd29b581ad24eb7fd33ddcc181037e0e78bbb463e4c.jpg)

![429c48fc17c2e9328f496c7716696cff9e27c7e286eb5cb744c149133ed433bd.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/images/429c48fc17c2e9328f496c7716696cff9e27c7e286eb5cb744c149133ed433bd.jpg)

![4e1d7926fbc47db21b9efd7fa94c3db8b5fb06b852a5114b0f5632b89a1fdbc7.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/images/4e1d7926fbc47db21b9efd7fa94c3db8b5fb06b852a5114b0f5632b89a1fdbc7.jpg)

![85b3ae91c3f81a185729a83b79b6d98773bcaef0a4ef248ee2be48cbb9ef4f75.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/images/85b3ae91c3f81a185729a83b79b6d98773bcaef0a4ef248ee2be48cbb9ef4f75.jpg)

![91e3cf1b136658779e77fbf2ab68d7237704baf52a653665ec27a106c35264af.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/images/91e3cf1b136658779e77fbf2ab68d7237704baf52a653665ec27a106c35264af.jpg)

![9947f871485df5e5cf5ec7a686f3e61a71505986a233bc6db4aa94a4370b0bbc.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/images/9947f871485df5e5cf5ec7a686f3e61a71505986a233bc6db4aa94a4370b0bbc.jpg)

![c14e4e8207921cf536d91606084b06b120010f360722b7c79abaf6906baa7be0.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/images/c14e4e8207921cf536d91606084b06b120010f360722b7c79abaf6906baa7be0.jpg)

![c834cd6dae7adec8f49057b739ef1b09db3553c243bef2d81f78a38e4ad20a2e.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/images/c834cd6dae7adec8f49057b739ef1b09db3553c243bef2d81f78a38e4ad20a2e.jpg)

![ce174fe2fe4a4a8bdd852946979ce408c9c818a419494faf436df6118ac81dff.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/images/ce174fe2fe4a4a8bdd852946979ce408c9c818a419494faf436df6118ac81dff.jpg)

![d15cf922f4be503248a9937d2e652e18edab74be1331a0dd3262fc66371877e5.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/images/d15cf922f4be503248a9937d2e652e18edab74be1331a0dd3262fc66371877e5.jpg)

![db294a8cbc399e3fb1c580469f729c2d0dbff59a008342dec2835672f50a699c.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/images/db294a8cbc399e3fb1c580469f729c2d0dbff59a008342dec2835672f50a699c.jpg)

![e9c495aa70485d578f15445adb8eb47d57357f87c3fe26b28105c5512a4f7360.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/images/e9c495aa70485d578f15445adb8eb47d57357f87c3fe26b28105c5512a4f7360.jpg)

![f8c0b826884ec8d7d758068585129a6ed967ea0f8b8dea8f2d29435760a01bf2.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/images/f8c0b826884ec8d7d758068585129a6ed967ea0f8b8dea8f2d29435760a01bf2.jpg)

![ffe8373ec4b22982ea02d66d7dc7925be24a35da1b4cafcda2bcbab8cbdf75a5.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/images/ffe8373ec4b22982ea02d66d7dc7925be24a35da1b4cafcda2bcbab8cbdf75a5.jpg)

### Tables

![08220f57df12294923ac22bf61dbe9604f9625e564844691a4b1e35ad6542c59.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/tables/08220f57df12294923ac22bf61dbe9604f9625e564844691a4b1e35ad6542c59.jpg)

![0a6fadc8b3a12035ba73f7705fd0574260f60dbc60bab46aa7834c7dc9bb5b56.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/tables/0a6fadc8b3a12035ba73f7705fd0574260f60dbc60bab46aa7834c7dc9bb5b56.jpg)

![17698856f536bd8309b7500898c7e993b41857ee95eebd45d563d1883c97473f.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/tables/17698856f536bd8309b7500898c7e993b41857ee95eebd45d563d1883c97473f.jpg)

![2285b747c96a5eb8b73ea9e17fcb2cb4854bb054339c9f2414467cf55a87583c.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/tables/2285b747c96a5eb8b73ea9e17fcb2cb4854bb054339c9f2414467cf55a87583c.jpg)

![506fc5739d8bb9c40e0860b26784170f6de49dfae5221c8193a0cb881b7a5d66.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/tables/506fc5739d8bb9c40e0860b26784170f6de49dfae5221c8193a0cb881b7a5d66.jpg)

![5ed254bdf10f5292e127a73760bf3b9fd22e1d0bd95bdecea639ab330c7bd757.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/tables/5ed254bdf10f5292e127a73760bf3b9fd22e1d0bd95bdecea639ab330c7bd757.jpg)

![6fc02b92bac8ced05c7fcbb65de241dc6b4dec77c5dff04ecdc9f549fa1c083b.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/tables/6fc02b92bac8ced05c7fcbb65de241dc6b4dec77c5dff04ecdc9f549fa1c083b.jpg)

![7f8e95d4f071a416ba588c8797c25557ea8b06e5180b561fbbaee334ce7a3cdf.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/tables/7f8e95d4f071a416ba588c8797c25557ea8b06e5180b561fbbaee334ce7a3cdf.jpg)

![7fd1671fe61800b6447122965ae24e45d58d0d7e9c88b73bdac402b96378536c.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/tables/7fd1671fe61800b6447122965ae24e45d58d0d7e9c88b73bdac402b96378536c.jpg)

![b40edd01042ada76ba55b220da54411aa78328cc4a5176d777e4b1dfd02b4648.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/tables/b40edd01042ada76ba55b220da54411aa78328cc4a5176d777e4b1dfd02b4648.jpg)

![b4277e30e4a41e0a5535bf37632880a62d37e559fda55e8e52f9cddd071cb0c8.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/tables/b4277e30e4a41e0a5535bf37632880a62d37e559fda55e8e52f9cddd071cb0c8.jpg)

![b44349f6c7ab567dd50e8495e7c93fcf60f3384bae7552dbd5be6e69c94d7cbf.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/tables/b44349f6c7ab567dd50e8495e7c93fcf60f3384bae7552dbd5be6e69c94d7cbf.jpg)

![b4a5b9a1a3105c2517e7fa773f5171f68dbdd938e3ab13264f28c960d6061b78.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/tables/b4a5b9a1a3105c2517e7fa773f5171f68dbdd938e3ab13264f28c960d6061b78.jpg)

![b6f03f46239f607913bd43099753930eca2b1e1dbadbb1197640a541c76d2f6f.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/tables/b6f03f46239f607913bd43099753930eca2b1e1dbadbb1197640a541c76d2f6f.jpg)

![bc49f9be939f9e4fe0a5ff662b1a5b3ae5def8b17e10a0e7192fd12af5fae623.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/tables/bc49f9be939f9e4fe0a5ff662b1a5b3ae5def8b17e10a0e7192fd12af5fae623.jpg)

![c16b5a7f550be5866c354213a992535b438e3b006f62efbfd33feea171587c77.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/tables/c16b5a7f550be5866c354213a992535b438e3b006f62efbfd33feea171587c77.jpg)

![c536b451706fcc6042aa4504e83df3a305808bfe4884855cf24373884ab71a4b.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/tables/c536b451706fcc6042aa4504e83df3a305808bfe4884855cf24373884ab71a4b.jpg)

![de177e1b3575ce107feaa3509a0114d776f190788129860f78f138c8badeedbd.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/tables/de177e1b3575ce107feaa3509a0114d776f190788129860f78f138c8badeedbd.jpg)

![e70fac455f3bcfef0e8114329746b6377319867fc2560d0c57b7d8dc0c595839.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/tables/e70fac455f3bcfef0e8114329746b6377319867fc2560d0c57b7d8dc0c595839.jpg)

![ea982b20cd2e123c19de1d198ac618d410901e579f4cd39c2afd486a00c77c54.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/tables/ea982b20cd2e123c19de1d198ac618d410901e579f4cd39c2afd486a00c77c54.jpg)

![f1d994b4cc8db0dfa1226104b58a44c2a5162066c85c072c99395b0840415705.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/tables/f1d994b4cc8db0dfa1226104b58a44c2a5162066c85c072c99395b0840415705.jpg)

![f7553be5301561977df5c2c9518fbc4d688ac7f1fa35d2c088a1ad0620e3b02b.jpg](../iclr_results/2714_Words%20in%20Motion_%20Extracting%20Interpretable%20Control%20Vectors%20for%20Motion%20Transformers/tables/f7553be5301561977df5c2c9518fbc4d688ac7f1fa35d2c088a1ad0620e3b02b.jpg)

## Discretization-invariance? On the Discretization Mismatch Errors in Neural Operators


### Images

![63f11301409e0c6f965a33da337abc9652e4795a15082e39d7172000546bf36f.jpg](../iclr_results/2715_Discretization-invariance_%20On%20the%20Discretization%20Mismatch%20Errors%20in%20Neural%20Operators/images/63f11301409e0c6f965a33da337abc9652e4795a15082e39d7172000546bf36f.jpg)

![b544d27a7bbafd6685dee1ad6b90e5ef17726373275dd576d64ee969354ce622.jpg](../iclr_results/2715_Discretization-invariance_%20On%20the%20Discretization%20Mismatch%20Errors%20in%20Neural%20Operators/images/b544d27a7bbafd6685dee1ad6b90e5ef17726373275dd576d64ee969354ce622.jpg)

![c1d058532a8f7155849c7e76584943d6377e6d6213c9c1999cbc36aa0eea5b08.jpg](../iclr_results/2715_Discretization-invariance_%20On%20the%20Discretization%20Mismatch%20Errors%20in%20Neural%20Operators/images/c1d058532a8f7155849c7e76584943d6377e6d6213c9c1999cbc36aa0eea5b08.jpg)

![cb9ef285e3ce41370dd4724c571cdf5b9ddeb8d937e476edeb43a5e5264b4a2e.jpg](../iclr_results/2715_Discretization-invariance_%20On%20the%20Discretization%20Mismatch%20Errors%20in%20Neural%20Operators/images/cb9ef285e3ce41370dd4724c571cdf5b9ddeb8d937e476edeb43a5e5264b4a2e.jpg)

![f4eedaeee3dceac85c78e5e8355ff1a929198f6ba1fdd0997c21c07f768e6d06.jpg](../iclr_results/2715_Discretization-invariance_%20On%20the%20Discretization%20Mismatch%20Errors%20in%20Neural%20Operators/images/f4eedaeee3dceac85c78e5e8355ff1a929198f6ba1fdd0997c21c07f768e6d06.jpg)

### Tables

![140af756fcef8550231434d0eb5bb6f541aabce84a0de696962c1682a35466e9.jpg](../iclr_results/2715_Discretization-invariance_%20On%20the%20Discretization%20Mismatch%20Errors%20in%20Neural%20Operators/tables/140af756fcef8550231434d0eb5bb6f541aabce84a0de696962c1682a35466e9.jpg)

![324c4e0f670cd39e337623e5141aca4c228b2e172c6f5191a8716b115331ae82.jpg](../iclr_results/2715_Discretization-invariance_%20On%20the%20Discretization%20Mismatch%20Errors%20in%20Neural%20Operators/tables/324c4e0f670cd39e337623e5141aca4c228b2e172c6f5191a8716b115331ae82.jpg)

![36dbef55dc80fbbdf3e586be9a159d76bec9af351f406f323ae957fbc861960a.jpg](../iclr_results/2715_Discretization-invariance_%20On%20the%20Discretization%20Mismatch%20Errors%20in%20Neural%20Operators/tables/36dbef55dc80fbbdf3e586be9a159d76bec9af351f406f323ae957fbc861960a.jpg)

![575e072969ddd62379712478ae9991e7bcf6713946aeb5ea2e91cabff94dcc95.jpg](../iclr_results/2715_Discretization-invariance_%20On%20the%20Discretization%20Mismatch%20Errors%20in%20Neural%20Operators/tables/575e072969ddd62379712478ae9991e7bcf6713946aeb5ea2e91cabff94dcc95.jpg)

![57bc5df2f99bce33c3cbe55b86f57a91963c38c9cb91aa31714fa7152016ef71.jpg](../iclr_results/2715_Discretization-invariance_%20On%20the%20Discretization%20Mismatch%20Errors%20in%20Neural%20Operators/tables/57bc5df2f99bce33c3cbe55b86f57a91963c38c9cb91aa31714fa7152016ef71.jpg)

![85addb7434d1d76ffac44b1dad49e08afa4d53493aaf350e92b829114973ab7a.jpg](../iclr_results/2715_Discretization-invariance_%20On%20the%20Discretization%20Mismatch%20Errors%20in%20Neural%20Operators/tables/85addb7434d1d76ffac44b1dad49e08afa4d53493aaf350e92b829114973ab7a.jpg)

![a1a58efc9d86e67b6002f438c420978fe672ad819849f8c1cdb944128c0847a6.jpg](../iclr_results/2715_Discretization-invariance_%20On%20the%20Discretization%20Mismatch%20Errors%20in%20Neural%20Operators/tables/a1a58efc9d86e67b6002f438c420978fe672ad819849f8c1cdb944128c0847a6.jpg)

![b652abef9e8a8fcefc44500f5c0ca172c878f97cdfd9ccec6ce8e6378e7a5b71.jpg](../iclr_results/2715_Discretization-invariance_%20On%20the%20Discretization%20Mismatch%20Errors%20in%20Neural%20Operators/tables/b652abef9e8a8fcefc44500f5c0ca172c878f97cdfd9ccec6ce8e6378e7a5b71.jpg)

![f9a24d25141949d8035fb83a0f1c1c60af68210c41e0c22097dc537b2a97ca0f.jpg](../iclr_results/2715_Discretization-invariance_%20On%20the%20Discretization%20Mismatch%20Errors%20in%20Neural%20Operators/tables/f9a24d25141949d8035fb83a0f1c1c60af68210c41e0c22097dc537b2a97ca0f.jpg)

## PostEdit: Posterior Sampling for Efficient Zero-Shot Image Editing


### Images

![0341fad2b1ac046ef4580fa29b461558788035ffb4e82f000958ac722e223b75.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/images/0341fad2b1ac046ef4580fa29b461558788035ffb4e82f000958ac722e223b75.jpg)

![15d15acf4d39dd7df9364d67e610f3a62cdd1493012db38f4b435e1cda44a3c0.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/images/15d15acf4d39dd7df9364d67e610f3a62cdd1493012db38f4b435e1cda44a3c0.jpg)

![213b32745b674aebbd196d267581aca84a0a605e0b48860115a23688c94158be.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/images/213b32745b674aebbd196d267581aca84a0a605e0b48860115a23688c94158be.jpg)

![2485d7911dcedd53670bd512b2249f1e0e824cabb49ab80bdea662bd371596e8.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/images/2485d7911dcedd53670bd512b2249f1e0e824cabb49ab80bdea662bd371596e8.jpg)

![307028310a0240445343f4ae7d0216764fa1f21371274b2b0939cffa1e25f231.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/images/307028310a0240445343f4ae7d0216764fa1f21371274b2b0939cffa1e25f231.jpg)

![3434681e88ffe015b43d3e94145c6308a3e60df9143d25ea8b6f2e9c9bb5636b.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/images/3434681e88ffe015b43d3e94145c6308a3e60df9143d25ea8b6f2e9c9bb5636b.jpg)

![521806fcdb2a44af8aca07d26fa867df3e9c302a73931cc1731d5ef78538f4c5.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/images/521806fcdb2a44af8aca07d26fa867df3e9c302a73931cc1731d5ef78538f4c5.jpg)

![5708cd9d1b162799829156528d8e5ca16938414b36b32ee31e69d36a125e93ad.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/images/5708cd9d1b162799829156528d8e5ca16938414b36b32ee31e69d36a125e93ad.jpg)

![5c7a29b6b279c7b00a0070d6522bc6dbdc7ee60e51f65255bef627ae515fc0bf.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/images/5c7a29b6b279c7b00a0070d6522bc6dbdc7ee60e51f65255bef627ae515fc0bf.jpg)

![617781b4837dbfdb529b01040cb6b91b69f46d6c0ee05923b350caf17b87c3ab.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/images/617781b4837dbfdb529b01040cb6b91b69f46d6c0ee05923b350caf17b87c3ab.jpg)

![913fcccc2802c31a968e32020fbec8b348f6ece440b7f49ef85bd51072e014c4.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/images/913fcccc2802c31a968e32020fbec8b348f6ece440b7f49ef85bd51072e014c4.jpg)

![96978bccbc3184ee5b965a4a24394b7f8c22580e70000f2a98424c11bfaa5541.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/images/96978bccbc3184ee5b965a4a24394b7f8c22580e70000f2a98424c11bfaa5541.jpg)

![a50041d2471c05f6d4fb5914b558f6468db897738293e336c64a564e4d809ce5.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/images/a50041d2471c05f6d4fb5914b558f6468db897738293e336c64a564e4d809ce5.jpg)

![b11c2acb0e1b34638cb3e4340336f6e3c2bdc0c6c35db8934172899d7cc26a6e.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/images/b11c2acb0e1b34638cb3e4340336f6e3c2bdc0c6c35db8934172899d7cc26a6e.jpg)

![bd8a7f3631c166835c5c9bb912737076c4b4c3bd251750e410b6fd8edf34793c.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/images/bd8a7f3631c166835c5c9bb912737076c4b4c3bd251750e410b6fd8edf34793c.jpg)

![bf8edb71697e6408b22e6f6668d9b067985471b133a66ad6dc7a4a8813a824e7.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/images/bf8edb71697e6408b22e6f6668d9b067985471b133a66ad6dc7a4a8813a824e7.jpg)

![d0419aa431d53a14f402f573a39811fcdf5bac6a31917809656fb63d3f10a3a4.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/images/d0419aa431d53a14f402f573a39811fcdf5bac6a31917809656fb63d3f10a3a4.jpg)

![db578c45b77cb5a2045400158134e79d2f235356798abeffaa3877295aa82bb9.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/images/db578c45b77cb5a2045400158134e79d2f235356798abeffaa3877295aa82bb9.jpg)

![e64ca8930cd2759121c25605e94727476ad10ee8b274fa26f6fe3352a37be4b0.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/images/e64ca8930cd2759121c25605e94727476ad10ee8b274fa26f6fe3352a37be4b0.jpg)

![e7562f4f791c8b12ecc1b57a06a92d696837349b96ec6ca680e84a80f17419b0.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/images/e7562f4f791c8b12ecc1b57a06a92d696837349b96ec6ca680e84a80f17419b0.jpg)

![e8c1d159ca2cb82ac4a9265be402692472f00ec07cc53d41738a03cd5ba20a29.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/images/e8c1d159ca2cb82ac4a9265be402692472f00ec07cc53d41738a03cd5ba20a29.jpg)

![e9cc8e9130e9857d9ae5da3b353c6c96c8f9fc060fffe752e42513db743c111b.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/images/e9cc8e9130e9857d9ae5da3b353c6c96c8f9fc060fffe752e42513db743c111b.jpg)

### Tables

![0e76a3aa77b2b1a5cd4b29a34162bcf0cef44e61972315170febfc6454cba815.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/tables/0e76a3aa77b2b1a5cd4b29a34162bcf0cef44e61972315170febfc6454cba815.jpg)

![33c249f401a3e912985265c4a2b372fd622ed15e6cb7b722426cefead3c6160c.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/tables/33c249f401a3e912985265c4a2b372fd622ed15e6cb7b722426cefead3c6160c.jpg)

![469030483dfbd86ba56e6653743619406b8442d507dacd7bf0c743b30fe466e9.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/tables/469030483dfbd86ba56e6653743619406b8442d507dacd7bf0c743b30fe466e9.jpg)

![961871b0314b28e382c83a801400adb20ad2aa37467a02dbeebc044d4b5642dc.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/tables/961871b0314b28e382c83a801400adb20ad2aa37467a02dbeebc044d4b5642dc.jpg)

![aa30889d9afc70f51207e5d1e2e95f021e7e2bcb507465035e613406c7e0e32d.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/tables/aa30889d9afc70f51207e5d1e2e95f021e7e2bcb507465035e613406c7e0e32d.jpg)

![ae043ef3cda5de4424d8902f92ce0bfcbe296071b2912f85cec0e263cd01d400.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/tables/ae043ef3cda5de4424d8902f92ce0bfcbe296071b2912f85cec0e263cd01d400.jpg)

![b27416d971dd6b3a129d4deafb5ca539dba82a6670a2535cbe73e9590389288c.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/tables/b27416d971dd6b3a129d4deafb5ca539dba82a6670a2535cbe73e9590389288c.jpg)

![bac23208708550bbf0aee842794a4c641b415436a6932b943f1b403a5a45e9c4.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/tables/bac23208708550bbf0aee842794a4c641b415436a6932b943f1b403a5a45e9c4.jpg)

![bbf97fd96336bbf3dd3b70b4169d820a75b2940c419776c66f70148f6a9c142b.jpg](../iclr_results/2716_PostEdit_%20Posterior%20Sampling%20for%20Efficient%20Zero-Shot%20Image%20Editing/tables/bbf97fd96336bbf3dd3b70b4169d820a75b2940c419776c66f70148f6a9c142b.jpg)

## Exploiting Hidden Symmetry to Improve Objective Perturbation for DP Linear Learners with a Nonsmooth L1-Norm


### Images

![00e8bb4f0db07a8eb01559288bbd9ad3e47e08ba18640953426105dc865ae227.jpg](../iclr_results/2717_Exploiting%20Hidden%20Symmetry%20to%20Improve%20Objective%20Perturbation%20for%20DP%20Linear%20Learners%20with%20a%20Nonsmooth/images/00e8bb4f0db07a8eb01559288bbd9ad3e47e08ba18640953426105dc865ae227.jpg)

![3ebbf02b41de2c3e8ff44a7be59aef44eb4f3fcfc5b9165dfb2a8aca7db4548b.jpg](../iclr_results/2717_Exploiting%20Hidden%20Symmetry%20to%20Improve%20Objective%20Perturbation%20for%20DP%20Linear%20Learners%20with%20a%20Nonsmooth/images/3ebbf02b41de2c3e8ff44a7be59aef44eb4f3fcfc5b9165dfb2a8aca7db4548b.jpg)

![49c158675f02c30417e123069fb4c813551aa51eb1706d9ff788a6ed9ee00f43.jpg](../iclr_results/2717_Exploiting%20Hidden%20Symmetry%20to%20Improve%20Objective%20Perturbation%20for%20DP%20Linear%20Learners%20with%20a%20Nonsmooth/images/49c158675f02c30417e123069fb4c813551aa51eb1706d9ff788a6ed9ee00f43.jpg)

![740f9832d616ba3784be5747e1ee6d8cf48d9d38d8c056e9e9cc8783aec68961.jpg](../iclr_results/2717_Exploiting%20Hidden%20Symmetry%20to%20Improve%20Objective%20Perturbation%20for%20DP%20Linear%20Learners%20with%20a%20Nonsmooth/images/740f9832d616ba3784be5747e1ee6d8cf48d9d38d8c056e9e9cc8783aec68961.jpg)

![af21381544037ceceb84f277b457495a225ac8ec00c5cd5464a7f07add7e0ebb.jpg](../iclr_results/2717_Exploiting%20Hidden%20Symmetry%20to%20Improve%20Objective%20Perturbation%20for%20DP%20Linear%20Learners%20with%20a%20Nonsmooth/images/af21381544037ceceb84f277b457495a225ac8ec00c5cd5464a7f07add7e0ebb.jpg)

![e7825def85b6bf2edae49f88e72c9ad7c0978104b0cbb3095c3595990d73f439.jpg](../iclr_results/2717_Exploiting%20Hidden%20Symmetry%20to%20Improve%20Objective%20Perturbation%20for%20DP%20Linear%20Learners%20with%20a%20Nonsmooth/images/e7825def85b6bf2edae49f88e72c9ad7c0978104b0cbb3095c3595990d73f439.jpg)

![f8c5ee3a6092ff5777e5cdb5ba4b3f9f0809279e40c45a8ea82995bc01ad1284.jpg](../iclr_results/2717_Exploiting%20Hidden%20Symmetry%20to%20Improve%20Objective%20Perturbation%20for%20DP%20Linear%20Learners%20with%20a%20Nonsmooth/images/f8c5ee3a6092ff5777e5cdb5ba4b3f9f0809279e40c45a8ea82995bc01ad1284.jpg)

![fa2087d6cf0b9c0dfd7dc55c9f060a57d803913ebccfcb01a0bb051ab5a98840.jpg](../iclr_results/2717_Exploiting%20Hidden%20Symmetry%20to%20Improve%20Objective%20Perturbation%20for%20DP%20Linear%20Learners%20with%20a%20Nonsmooth/images/fa2087d6cf0b9c0dfd7dc55c9f060a57d803913ebccfcb01a0bb051ab5a98840.jpg)

### Tables

![1441255de37e1f99450d1b0469a1cc923bc9c4f606f7330d5e78d143e954ffa5.jpg](../iclr_results/2717_Exploiting%20Hidden%20Symmetry%20to%20Improve%20Objective%20Perturbation%20for%20DP%20Linear%20Learners%20with%20a%20Nonsmooth/tables/1441255de37e1f99450d1b0469a1cc923bc9c4f606f7330d5e78d143e954ffa5.jpg)

![9c2332107e88bfd8c4752a074443defbae5e428d776a5819fa56b63e5187f327.jpg](../iclr_results/2717_Exploiting%20Hidden%20Symmetry%20to%20Improve%20Objective%20Perturbation%20for%20DP%20Linear%20Learners%20with%20a%20Nonsmooth/tables/9c2332107e88bfd8c4752a074443defbae5e428d776a5819fa56b63e5187f327.jpg)

![a35bc9efc390c2dbfb897fc6c38d77c638664421909de7a67984d1de39428507.jpg](../iclr_results/2717_Exploiting%20Hidden%20Symmetry%20to%20Improve%20Objective%20Perturbation%20for%20DP%20Linear%20Learners%20with%20a%20Nonsmooth/tables/a35bc9efc390c2dbfb897fc6c38d77c638664421909de7a67984d1de39428507.jpg)

## RetroInText: A Multimodal Large Language Model Enhanced Framework for Retrosynthetic Planning via In-Context Representation Learning


### Images

![4421231e8a313b43106704e827beccc5ff8ad9465ec44b35e8c20d3afd033f82.jpg](../iclr_results/2718_RetroInText_%20A%20Multimodal%20Large%20Language%20Model%20Enhanced%20Framework%20for%20Retrosynthetic%20Planning%20via%20In/images/4421231e8a313b43106704e827beccc5ff8ad9465ec44b35e8c20d3afd033f82.jpg)

![eb1df07f762c402a98481f5dbfc78e317ea8d29bf542e48781b243ddf3b347a0.jpg](../iclr_results/2718_RetroInText_%20A%20Multimodal%20Large%20Language%20Model%20Enhanced%20Framework%20for%20Retrosynthetic%20Planning%20via%20In/images/eb1df07f762c402a98481f5dbfc78e317ea8d29bf542e48781b243ddf3b347a0.jpg)

### Tables

![0975721d0e4113a16794d633eeb2d8f8de32b3525db79af49efe87631743ae46.jpg](../iclr_results/2718_RetroInText_%20A%20Multimodal%20Large%20Language%20Model%20Enhanced%20Framework%20for%20Retrosynthetic%20Planning%20via%20In/tables/0975721d0e4113a16794d633eeb2d8f8de32b3525db79af49efe87631743ae46.jpg)

![173635408ff9888c7f958cf71abbf8d48056fac39ab47e5597c5def20afb591d.jpg](../iclr_results/2718_RetroInText_%20A%20Multimodal%20Large%20Language%20Model%20Enhanced%20Framework%20for%20Retrosynthetic%20Planning%20via%20In/tables/173635408ff9888c7f958cf71abbf8d48056fac39ab47e5597c5def20afb591d.jpg)

![1a7f182d34fca1488d37a00f1a07ea63286bfa6cd89c7ccdca1dd3d998ac0d0a.jpg](../iclr_results/2718_RetroInText_%20A%20Multimodal%20Large%20Language%20Model%20Enhanced%20Framework%20for%20Retrosynthetic%20Planning%20via%20In/tables/1a7f182d34fca1488d37a00f1a07ea63286bfa6cd89c7ccdca1dd3d998ac0d0a.jpg)

![225fcbf08f8484343c4e2252eaa6dc33d0736c5c29f9868ed2720b5e80616b21.jpg](../iclr_results/2718_RetroInText_%20A%20Multimodal%20Large%20Language%20Model%20Enhanced%20Framework%20for%20Retrosynthetic%20Planning%20via%20In/tables/225fcbf08f8484343c4e2252eaa6dc33d0736c5c29f9868ed2720b5e80616b21.jpg)

![653b94cbc62d01ba93a7cadb37a09ac138a6a92a5255aeaf5e40b564ccc97363.jpg](../iclr_results/2718_RetroInText_%20A%20Multimodal%20Large%20Language%20Model%20Enhanced%20Framework%20for%20Retrosynthetic%20Planning%20via%20In/tables/653b94cbc62d01ba93a7cadb37a09ac138a6a92a5255aeaf5e40b564ccc97363.jpg)

![6add7d54f03f74cb2333a705025d6afc64254b380124955e12a93d36fa8ba0e0.jpg](../iclr_results/2718_RetroInText_%20A%20Multimodal%20Large%20Language%20Model%20Enhanced%20Framework%20for%20Retrosynthetic%20Planning%20via%20In/tables/6add7d54f03f74cb2333a705025d6afc64254b380124955e12a93d36fa8ba0e0.jpg)

![6f7cd02a033595fdf9cf6c7d5073c0f23fd6d5b83d83d8bb03c9490d61a0b352.jpg](../iclr_results/2718_RetroInText_%20A%20Multimodal%20Large%20Language%20Model%20Enhanced%20Framework%20for%20Retrosynthetic%20Planning%20via%20In/tables/6f7cd02a033595fdf9cf6c7d5073c0f23fd6d5b83d83d8bb03c9490d61a0b352.jpg)

![7203c7fc05cbe4e0f00e7d4c95cd224643146d324cd6c95c1d448423d6e57e85.jpg](../iclr_results/2718_RetroInText_%20A%20Multimodal%20Large%20Language%20Model%20Enhanced%20Framework%20for%20Retrosynthetic%20Planning%20via%20In/tables/7203c7fc05cbe4e0f00e7d4c95cd224643146d324cd6c95c1d448423d6e57e85.jpg)

![e872eb9e724c51d9f228f31ef3fded3b55c56f36762be44205b90c293a9e16a9.jpg](../iclr_results/2718_RetroInText_%20A%20Multimodal%20Large%20Language%20Model%20Enhanced%20Framework%20for%20Retrosynthetic%20Planning%20via%20In/tables/e872eb9e724c51d9f228f31ef3fded3b55c56f36762be44205b90c293a9e16a9.jpg)

![eebfdcf29042f358a702fbd4d58a269b049562c8087a9d50cedd1b98b57986cd.jpg](../iclr_results/2718_RetroInText_%20A%20Multimodal%20Large%20Language%20Model%20Enhanced%20Framework%20for%20Retrosynthetic%20Planning%20via%20In/tables/eebfdcf29042f358a702fbd4d58a269b049562c8087a9d50cedd1b98b57986cd.jpg)

## LiNeS: Post-training Layer Scaling Prevents Forgetting and Enhances Model Merging


### Images

![06763ed0f16333a41fb2b52e55cd6f997e7a5569959b1d0aa1038eb5e42db936.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/images/06763ed0f16333a41fb2b52e55cd6f997e7a5569959b1d0aa1038eb5e42db936.jpg)

![176baf5e568a1a1a4718e936f1438bdf396c4855699933d9293aa4707f5a7b1a.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/images/176baf5e568a1a1a4718e936f1438bdf396c4855699933d9293aa4707f5a7b1a.jpg)

![1b35a0cedf8090beb28329a767a3cbfa300657fa54398aa0c7e2faa6a2e00cf3.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/images/1b35a0cedf8090beb28329a767a3cbfa300657fa54398aa0c7e2faa6a2e00cf3.jpg)

![1c5cd858bdd67d597d58b52737a3559603b4aa417cc96ea5f0a90ed614a20118.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/images/1c5cd858bdd67d597d58b52737a3559603b4aa417cc96ea5f0a90ed614a20118.jpg)

![227ffacdb53a35f454bec1e1336e35f51c24d24259e1f70c62190ab0b7032b7e.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/images/227ffacdb53a35f454bec1e1336e35f51c24d24259e1f70c62190ab0b7032b7e.jpg)

![2deede5aa117d54da3ad7035742bbe19e07f1b627ea84473b7b6305c2769f199.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/images/2deede5aa117d54da3ad7035742bbe19e07f1b627ea84473b7b6305c2769f199.jpg)

![71f96e5dfbf47469dc3f8d7de73ab9b392bb9ab11ac82bde1a42e012b1702a96.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/images/71f96e5dfbf47469dc3f8d7de73ab9b392bb9ab11ac82bde1a42e012b1702a96.jpg)

![7908f5ffd2b9c3ee52bb587ea4ef56dc142ed5bf0ddf822f8107a8c5158cf649.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/images/7908f5ffd2b9c3ee52bb587ea4ef56dc142ed5bf0ddf822f8107a8c5158cf649.jpg)

![891556ce1b54682f9bceb6fd007559f34beee29b781ee7a67a0f5d23bad38590.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/images/891556ce1b54682f9bceb6fd007559f34beee29b781ee7a67a0f5d23bad38590.jpg)

![8e42e92cbd53c6ce6dfd2bbfb3bad03d633ad52bee7d5c20827d1caf38e43ef6.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/images/8e42e92cbd53c6ce6dfd2bbfb3bad03d633ad52bee7d5c20827d1caf38e43ef6.jpg)

![958493034cfba5eb69d0b4eee01369a150471b9bf9be4644a4bbe0bb4c5115b9.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/images/958493034cfba5eb69d0b4eee01369a150471b9bf9be4644a4bbe0bb4c5115b9.jpg)

![a8261894ff1cd6a38d18d6476e2f2c3440ce5552436f339b664608e1cafd3c8d.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/images/a8261894ff1cd6a38d18d6476e2f2c3440ce5552436f339b664608e1cafd3c8d.jpg)

![ab115d3ab4df5064fc1ec4c03ecc59b5ce0869f258f76853a73296e55b7b6b5a.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/images/ab115d3ab4df5064fc1ec4c03ecc59b5ce0869f258f76853a73296e55b7b6b5a.jpg)

![d0d35af2768718a2b0d3c9b6d46b3eb47f1cb51a71e84cc6224c5352d06f23a8.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/images/d0d35af2768718a2b0d3c9b6d46b3eb47f1cb51a71e84cc6224c5352d06f23a8.jpg)

![dcd4dac1c46228b5e9023f4f7d65ccff9a7f61231f32e95444cc534ffb7fe83b.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/images/dcd4dac1c46228b5e9023f4f7d65ccff9a7f61231f32e95444cc534ffb7fe83b.jpg)

![fd4fc863daf68ca54727dab55b1161524283f3026c6341fc944ae5cfc37b8e76.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/images/fd4fc863daf68ca54727dab55b1161524283f3026c6341fc944ae5cfc37b8e76.jpg)

### Tables

![0cc12c89200a9ce6d464510c2f3aa8a9236ff7686cb2537bb06a0b58f5464145.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/tables/0cc12c89200a9ce6d464510c2f3aa8a9236ff7686cb2537bb06a0b58f5464145.jpg)

![1abc23a512e77aca9503bb0faa10d7c99d6da0be7faca439b828d4062b56ff23.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/tables/1abc23a512e77aca9503bb0faa10d7c99d6da0be7faca439b828d4062b56ff23.jpg)

![511469be8b74438f74e6c28c06d66bd212f1f8e3a67a49392360a52d8ff59563.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/tables/511469be8b74438f74e6c28c06d66bd212f1f8e3a67a49392360a52d8ff59563.jpg)

![583b54d73a868e81d70ed4659972b4bc72f13fa205eb426d0ac6f3ddebbf5ad2.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/tables/583b54d73a868e81d70ed4659972b4bc72f13fa205eb426d0ac6f3ddebbf5ad2.jpg)

![594b0f84da2656b6ce444323aaca3ae4ac76c72852500649ea675d7ed4d0dcef.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/tables/594b0f84da2656b6ce444323aaca3ae4ac76c72852500649ea675d7ed4d0dcef.jpg)

![63d1666cf4f9c855f00f4b41b40d3f8db3044b01ab01bbd69adedc40f172131d.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/tables/63d1666cf4f9c855f00f4b41b40d3f8db3044b01ab01bbd69adedc40f172131d.jpg)

![6f8da657f0be224193e0e9d3fc59769c7ccad93573db1890a61177ca9cdcf3fd.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/tables/6f8da657f0be224193e0e9d3fc59769c7ccad93573db1890a61177ca9cdcf3fd.jpg)

![9780873be180fdc88c11628450d525f2f46e2180620b6cc34cf0934118fe6ce3.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/tables/9780873be180fdc88c11628450d525f2f46e2180620b6cc34cf0934118fe6ce3.jpg)

![9c2c32fc225a90f0692992503b6a1f3062226d0c845183bae59401ab9f4cd861.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/tables/9c2c32fc225a90f0692992503b6a1f3062226d0c845183bae59401ab9f4cd861.jpg)

![a46ec1e29d13027667476ad95bde02de0270f31d2101e507a40843df775dbbe0.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/tables/a46ec1e29d13027667476ad95bde02de0270f31d2101e507a40843df775dbbe0.jpg)

![ab2aff06b0cef0689d4ec2d885f2bc53bc54f66cc4f8a991eeca116509aec25f.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/tables/ab2aff06b0cef0689d4ec2d885f2bc53bc54f66cc4f8a991eeca116509aec25f.jpg)

![e0446bf830e6b1f88e10deff6458be888a100fbae762ef8ec894e2774a595044.jpg](../iclr_results/2719_LiNeS_%20Post-training%20Layer%20Scaling%20Prevents%20Forgetting%20and%20Enhances%20Model%20Merging/tables/e0446bf830e6b1f88e10deff6458be888a100fbae762ef8ec894e2774a595044.jpg)

## An Empirical Analysis of Uncertainty in Large Language Model Evaluations


### Images

![0bd76850f17f0908deeba7ad2739a60eb71617ee629f854c801dc08994a82973.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/images/0bd76850f17f0908deeba7ad2739a60eb71617ee629f854c801dc08994a82973.jpg)

![3ee66d99940d16ca4ccaba9ec3ddedb0a5badda90fa78d442e201f45ae8243be.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/images/3ee66d99940d16ca4ccaba9ec3ddedb0a5badda90fa78d442e201f45ae8243be.jpg)

![464b6a91c7d2549b722d71e0cf92479b70b6eb84458d4c5839fb7a7bb6ea61ff.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/images/464b6a91c7d2549b722d71e0cf92479b70b6eb84458d4c5839fb7a7bb6ea61ff.jpg)

![496affbf30f5321b2c237bbc72d30a0078e350a5b830099fce22156ec5ff91e5.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/images/496affbf30f5321b2c237bbc72d30a0078e350a5b830099fce22156ec5ff91e5.jpg)

![526e9cc2db9f523a9fd18d44b3ce8663ad1c1d8cbbf61e926001dc54b7179d35.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/images/526e9cc2db9f523a9fd18d44b3ce8663ad1c1d8cbbf61e926001dc54b7179d35.jpg)

![7aa8a5b624a3992ad33cc9ee3c09ef18d52d1c5377a371d34e1bdc137ce9dc48.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/images/7aa8a5b624a3992ad33cc9ee3c09ef18d52d1c5377a371d34e1bdc137ce9dc48.jpg)

![81ad86a6b9a9ed83c97ced2875ccf6f12e77c5b7c07798b83965ff5d3fdd0aee.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/images/81ad86a6b9a9ed83c97ced2875ccf6f12e77c5b7c07798b83965ff5d3fdd0aee.jpg)

![882b580bbe4e6af05b8b6077934e69b977311d82438ccf83536c26bee7eb7b2a.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/images/882b580bbe4e6af05b8b6077934e69b977311d82438ccf83536c26bee7eb7b2a.jpg)

![b96b2abadcb61086cfa65dc500a19e30967f855bebcbf6ebb2ee962971de48bb.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/images/b96b2abadcb61086cfa65dc500a19e30967f855bebcbf6ebb2ee962971de48bb.jpg)

![f55be218d6606404c4680cebe14fa9fe2918d5f3642d542d9de6cfaaae8e550f.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/images/f55be218d6606404c4680cebe14fa9fe2918d5f3642d542d9de6cfaaae8e550f.jpg)

### Tables

![0db779b24588e3fde22c4472934b050f02b81dcb315563d6803f2ee0229b9938.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/0db779b24588e3fde22c4472934b050f02b81dcb315563d6803f2ee0229b9938.jpg)

![10c4997d764d04658cc94c84acbed7e832b24cf5a7861eccc785342b30af607d.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/10c4997d764d04658cc94c84acbed7e832b24cf5a7861eccc785342b30af607d.jpg)

![12d7f8b9114cbc245dff178574ec3acb943f65b20a13fc53afa2e3d66a987bcd.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/12d7f8b9114cbc245dff178574ec3acb943f65b20a13fc53afa2e3d66a987bcd.jpg)

![1413a4e6cad2ede182f58c390e4f258c11284ebeff1e913cc76507897dd75d97.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/1413a4e6cad2ede182f58c390e4f258c11284ebeff1e913cc76507897dd75d97.jpg)

![143700a84a89b56f7ef613b6af2fe0daa40b0a2a51c7ed791c0d34bce76a4534.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/143700a84a89b56f7ef613b6af2fe0daa40b0a2a51c7ed791c0d34bce76a4534.jpg)

![1fd602db0fcf43da7a1a94daad2530cd2a5297b142472b2f43a81f94fce0d957.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/1fd602db0fcf43da7a1a94daad2530cd2a5297b142472b2f43a81f94fce0d957.jpg)

![2e94ab45bbabc88f47519b2e4a7e3a080e95f6d26469204ce374fd505137b3cd.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/2e94ab45bbabc88f47519b2e4a7e3a080e95f6d26469204ce374fd505137b3cd.jpg)

![414a1919f85ecafb4f26f30443199f1ba46d4ea952dbb93cd36f697d592e41fe.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/414a1919f85ecafb4f26f30443199f1ba46d4ea952dbb93cd36f697d592e41fe.jpg)

![4ca89a21a47a599cb87e1b953203c992c807fcc622a1d00f6dda63dfbcb14501.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/4ca89a21a47a599cb87e1b953203c992c807fcc622a1d00f6dda63dfbcb14501.jpg)

![4dfbaec23e19763e973bb5b22905a402f7e36eb56c2d9ba2a75b19876f0b27d2.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/4dfbaec23e19763e973bb5b22905a402f7e36eb56c2d9ba2a75b19876f0b27d2.jpg)

![5034839cc4ead6ae30f4c75fb2c104007b937051ec1b16cd805aecbfcf7e94a2.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/5034839cc4ead6ae30f4c75fb2c104007b937051ec1b16cd805aecbfcf7e94a2.jpg)

![515c45589bd5d5305d23fb7af6c7b835858612fbf44d3185fb2e1d5ab7f64544.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/515c45589bd5d5305d23fb7af6c7b835858612fbf44d3185fb2e1d5ab7f64544.jpg)

![5be5daa9c276b5f76e7c0e39c1d2b7d8fc850f373b7480b35a2f6b853ff38381.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/5be5daa9c276b5f76e7c0e39c1d2b7d8fc850f373b7480b35a2f6b853ff38381.jpg)

![6edffbe935c23f496b226a234ef3df52aa321782888c05f492959bf497202c6d.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/6edffbe935c23f496b226a234ef3df52aa321782888c05f492959bf497202c6d.jpg)

![702cca4781fcc0848f42e60907ae875ace4926dd6c36cd9c90ba4e5668bcc182.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/702cca4781fcc0848f42e60907ae875ace4926dd6c36cd9c90ba4e5668bcc182.jpg)

![7c9d6f90d34797203ad854bb2d8eca247fc033479c88392ccd1dc090ea84a839.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/7c9d6f90d34797203ad854bb2d8eca247fc033479c88392ccd1dc090ea84a839.jpg)

![8ddd3d82b7ea0beeea337725935b4f4883f3c6371ee93d930e88cc90c4cafbea.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/8ddd3d82b7ea0beeea337725935b4f4883f3c6371ee93d930e88cc90c4cafbea.jpg)

![93a2ad8e51cd0f18a36458bd16f542eb7233e82bacb0f1d6decc0416eed24a8c.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/93a2ad8e51cd0f18a36458bd16f542eb7233e82bacb0f1d6decc0416eed24a8c.jpg)

![9f7245a0a3541a380bd9efbb2d550983297b15f3e32f3908b9ed1111dd2e2054.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/9f7245a0a3541a380bd9efbb2d550983297b15f3e32f3908b9ed1111dd2e2054.jpg)

![c2b23046b420999b50a1318bee44f9d87b28b35ab5e9fc7520470214cf90081b.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/c2b23046b420999b50a1318bee44f9d87b28b35ab5e9fc7520470214cf90081b.jpg)

![d17f5baf6993a983395e49a42cb71a514d542a49fd81abd21951bd4109e7ccaf.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/d17f5baf6993a983395e49a42cb71a514d542a49fd81abd21951bd4109e7ccaf.jpg)

![f2c274bd44937637f39937df9c90356cf6e999473204e8e13d9d3d806a2f0bc0.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/f2c274bd44937637f39937df9c90356cf6e999473204e8e13d9d3d806a2f0bc0.jpg)

![f8fc0dc1b1d20c49bddb1ff3b7d5200f3d6b3f627cd2cf121acdb15ca0f8cd4d.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/f8fc0dc1b1d20c49bddb1ff3b7d5200f3d6b3f627cd2cf121acdb15ca0f8cd4d.jpg)

![ffa7c0bcb2630b94dac555427c124d928a3420e264e35e81882e0a10c388563e.jpg](../iclr_results/2720_An%20Empirical%20Analysis%20of%20Uncertainty%20in%20Large%20Language%20Model%20Evaluations/tables/ffa7c0bcb2630b94dac555427c124d928a3420e264e35e81882e0a10c388563e.jpg)

## Learning Dynamics of Deep Matrix Factorization Beyond the Edge of Stability


### Images

![08071b98ac987baf1882cf5df7a080dd7b7b38042b4c28ec8f1cdf37aeefe7cf.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/08071b98ac987baf1882cf5df7a080dd7b7b38042b4c28ec8f1cdf37aeefe7cf.jpg)

![0ae1f51676b1da2d7485eb6feeb77cb8d1e1f70d24d8f0efafbfb9b50d64ee43.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/0ae1f51676b1da2d7485eb6feeb77cb8d1e1f70d24d8f0efafbfb9b50d64ee43.jpg)

![19ec5fb5f6c53cba361d872e9054a125f443ce39f93d1980ee996c18b945698a.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/19ec5fb5f6c53cba361d872e9054a125f443ce39f93d1980ee996c18b945698a.jpg)

![33112a5fe3a93db8100154f68205d0fefd832f27c4ef32bae9b223f4a9a6631c.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/33112a5fe3a93db8100154f68205d0fefd832f27c4ef32bae9b223f4a9a6631c.jpg)

![3a70a3fbd8d7e357a48a8994bdd8d609aeefe5b214e3baa7f4172aaf60526659.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/3a70a3fbd8d7e357a48a8994bdd8d609aeefe5b214e3baa7f4172aaf60526659.jpg)

![4ef8bc69ebd4bcb9fe1313cb0abf8ef3e1c9611ac050a3b8baae3706ade3a9b6.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/4ef8bc69ebd4bcb9fe1313cb0abf8ef3e1c9611ac050a3b8baae3706ade3a9b6.jpg)

![5f66ff03816943bf78de5ba72e430b98db0386bb6ef8890cc387edd41a154fd3.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/5f66ff03816943bf78de5ba72e430b98db0386bb6ef8890cc387edd41a154fd3.jpg)

![659841221707eb76297d15dc1a0c1e630f7a86ead3539019cb8df5157744caf7.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/659841221707eb76297d15dc1a0c1e630f7a86ead3539019cb8df5157744caf7.jpg)

![74bb3433ad31c236dbb582ac4d58769276a059754909dfdc03006cf998a13c23.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/74bb3433ad31c236dbb582ac4d58769276a059754909dfdc03006cf998a13c23.jpg)

![84a45debe64c0f86c3314e5d819e7fc3b41cd26fdf936ab9562486d49da37107.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/84a45debe64c0f86c3314e5d819e7fc3b41cd26fdf936ab9562486d49da37107.jpg)

![9534e8c44fdb36af9cd17d73ce069d469f06f70ab996c3c7b108d0e81727f36e.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/9534e8c44fdb36af9cd17d73ce069d469f06f70ab996c3c7b108d0e81727f36e.jpg)

![a480da17e47c45a7bf9e760e8c5fdc9667f52813bfe3e3b009bac2370e7fe332.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/a480da17e47c45a7bf9e760e8c5fdc9667f52813bfe3e3b009bac2370e7fe332.jpg)

![ae3fd12dd95a8662d14ab0afd8847fbcdbbd7576d8dbda6c14cfb6f869f739ed.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/ae3fd12dd95a8662d14ab0afd8847fbcdbbd7576d8dbda6c14cfb6f869f739ed.jpg)

![b3500cb68b63a2558ea860601f4b7bcb51577697fce5847effdbdc52418cf3c0.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/b3500cb68b63a2558ea860601f4b7bcb51577697fce5847effdbdc52418cf3c0.jpg)

![bb61a07ef7c788e36bd926319afb9bec09c8827ef0f7335495ad556c02f357f6.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/bb61a07ef7c788e36bd926319afb9bec09c8827ef0f7335495ad556c02f357f6.jpg)

![c1bccbb68e8dfaa6349c068be6a03be194c5069b3362f3b28cfb59276495da57.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/c1bccbb68e8dfaa6349c068be6a03be194c5069b3362f3b28cfb59276495da57.jpg)

![c3fe1adba5e156ec43e3fd5e147837438b5c9cadd41e870c045fc982d605fb6a.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/c3fe1adba5e156ec43e3fd5e147837438b5c9cadd41e870c045fc982d605fb6a.jpg)

![cd53a5e8dabbbe0a8b7a6379e419d0b5abcbf0b9c64b4d9edaa494e96a752fbc.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/cd53a5e8dabbbe0a8b7a6379e419d0b5abcbf0b9c64b4d9edaa494e96a752fbc.jpg)

![d00a7ea800176413e3024d24ec622ec4f3af008936804db0cc697686f1a95948.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/d00a7ea800176413e3024d24ec622ec4f3af008936804db0cc697686f1a95948.jpg)

![d28b2aa5e4d6904009e664642078735b53b42062cffa69a71e5cf4d1d8950862.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/d28b2aa5e4d6904009e664642078735b53b42062cffa69a71e5cf4d1d8950862.jpg)

![d46d59619df03936333cabee4bc1e5eb6d8df8f6d3bb9b23d2c6a78301c601a3.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/d46d59619df03936333cabee4bc1e5eb6d8df8f6d3bb9b23d2c6a78301c601a3.jpg)

![d8946849bac6ca2c3b9f8a657a79ea89923c9f3995a0f70f3037182184cf0d86.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/d8946849bac6ca2c3b9f8a657a79ea89923c9f3995a0f70f3037182184cf0d86.jpg)

![db61b2baabcc22e59fd10b16de39a85ce233b6f3803c77f91a0da79d2796958f.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/db61b2baabcc22e59fd10b16de39a85ce233b6f3803c77f91a0da79d2796958f.jpg)

![f4a34120d72253db220583c9d05ceb71fc7789012920257e76535b33cbdd8f94.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/f4a34120d72253db220583c9d05ceb71fc7789012920257e76535b33cbdd8f94.jpg)

![fe45c9deca7c7339d2f8efc44d1097c5cb25c62856745b7ff7799bdbf232ba59.jpg](../iclr_results/2721_Learning%20Dynamics%20of%20Deep%20Matrix%20Factorization%20Beyond%20the%20Edge%20of%20Stability/images/fe45c9deca7c7339d2f8efc44d1097c5cb25c62856745b7ff7799bdbf232ba59.jpg)

## RECAST: Reparameterized, Compact weight Adaptation for Sequential Tasks


### Images

![1ca57787c56b7b1bb10567ef1aef1b63a8608b780ebd2cd08eb8dea74485d16a.jpg](../iclr_results/2722_RECAST_%20Reparameterized%2C%20Compact%20weight%20Adaptation%20for%20Sequential%20Tasks/images/1ca57787c56b7b1bb10567ef1aef1b63a8608b780ebd2cd08eb8dea74485d16a.jpg)

![312d18b0904428a6ce17adb8bcff69dae518dcb6883888ddc0e1537487c9919c.jpg](../iclr_results/2722_RECAST_%20Reparameterized%2C%20Compact%20weight%20Adaptation%20for%20Sequential%20Tasks/images/312d18b0904428a6ce17adb8bcff69dae518dcb6883888ddc0e1537487c9919c.jpg)

![3a88b2aba219abd4cd63843303b5b1e2dd621eec2cf6958de60aea6ca1244f98.jpg](../iclr_results/2722_RECAST_%20Reparameterized%2C%20Compact%20weight%20Adaptation%20for%20Sequential%20Tasks/images/3a88b2aba219abd4cd63843303b5b1e2dd621eec2cf6958de60aea6ca1244f98.jpg)

![59e07ad64e1367a122ba4f69aa53177b5adeb3b21581566976ef4a7053273ddf.jpg](../iclr_results/2722_RECAST_%20Reparameterized%2C%20Compact%20weight%20Adaptation%20for%20Sequential%20Tasks/images/59e07ad64e1367a122ba4f69aa53177b5adeb3b21581566976ef4a7053273ddf.jpg)

![7648a83633bc3f4fa4425f7f4a5003b85930598e9317e95dabce916b783d3dd1.jpg](../iclr_results/2722_RECAST_%20Reparameterized%2C%20Compact%20weight%20Adaptation%20for%20Sequential%20Tasks/images/7648a83633bc3f4fa4425f7f4a5003b85930598e9317e95dabce916b783d3dd1.jpg)

![83c5f610169542c7db60511a16fa0ea513f09ca3baeb6c7c517518340bc91dcb.jpg](../iclr_results/2722_RECAST_%20Reparameterized%2C%20Compact%20weight%20Adaptation%20for%20Sequential%20Tasks/images/83c5f610169542c7db60511a16fa0ea513f09ca3baeb6c7c517518340bc91dcb.jpg)

![999b77df5a3e3e7bcc443532b1616efeeb18b6a5a707319c9f009398d567d101.jpg](../iclr_results/2722_RECAST_%20Reparameterized%2C%20Compact%20weight%20Adaptation%20for%20Sequential%20Tasks/images/999b77df5a3e3e7bcc443532b1616efeeb18b6a5a707319c9f009398d567d101.jpg)

![d53119801681f1ce6fd6f8d826dc4fe93b2088331198e041451b9c33f2f1763d.jpg](../iclr_results/2722_RECAST_%20Reparameterized%2C%20Compact%20weight%20Adaptation%20for%20Sequential%20Tasks/images/d53119801681f1ce6fd6f8d826dc4fe93b2088331198e041451b9c33f2f1763d.jpg)

![f566b5e9d4b834ee8ada4175f15d4551ad4a18371766af2df2bd817f546d826a.jpg](../iclr_results/2722_RECAST_%20Reparameterized%2C%20Compact%20weight%20Adaptation%20for%20Sequential%20Tasks/images/f566b5e9d4b834ee8ada4175f15d4551ad4a18371766af2df2bd817f546d826a.jpg)

![f8730b8ba8fe1b51006b13c90a4d6d677b16c312e98b46c8dd50b3fc7ccf2b31.jpg](../iclr_results/2722_RECAST_%20Reparameterized%2C%20Compact%20weight%20Adaptation%20for%20Sequential%20Tasks/images/f8730b8ba8fe1b51006b13c90a4d6d677b16c312e98b46c8dd50b3fc7ccf2b31.jpg)

### Tables

![57a9d02882d306dbdb704250d5e1bd1e30c290eb85e67b575d71e7606a6c4c37.jpg](../iclr_results/2722_RECAST_%20Reparameterized%2C%20Compact%20weight%20Adaptation%20for%20Sequential%20Tasks/tables/57a9d02882d306dbdb704250d5e1bd1e30c290eb85e67b575d71e7606a6c4c37.jpg)

![5a71d957d182e344287b7c57215824c1044e23198c0704cb37e9c381b1903371.jpg](../iclr_results/2722_RECAST_%20Reparameterized%2C%20Compact%20weight%20Adaptation%20for%20Sequential%20Tasks/tables/5a71d957d182e344287b7c57215824c1044e23198c0704cb37e9c381b1903371.jpg)

![6a9afde2443cb4bf58b744f297f5786dc4d4121863f67138346963de9ce266df.jpg](../iclr_results/2722_RECAST_%20Reparameterized%2C%20Compact%20weight%20Adaptation%20for%20Sequential%20Tasks/tables/6a9afde2443cb4bf58b744f297f5786dc4d4121863f67138346963de9ce266df.jpg)

![70e0d9ee24885ee8340f04e6f293e2f30f6794afd2c6a2f0a905759065f36f24.jpg](../iclr_results/2722_RECAST_%20Reparameterized%2C%20Compact%20weight%20Adaptation%20for%20Sequential%20Tasks/tables/70e0d9ee24885ee8340f04e6f293e2f30f6794afd2c6a2f0a905759065f36f24.jpg)

![aafaf1b6e9ddc332605d1acec9ddf7c7ff455ecb197102e34806d40a7e49df08.jpg](../iclr_results/2722_RECAST_%20Reparameterized%2C%20Compact%20weight%20Adaptation%20for%20Sequential%20Tasks/tables/aafaf1b6e9ddc332605d1acec9ddf7c7ff455ecb197102e34806d40a7e49df08.jpg)

![bdf7ebf526ea729815e16080caba7340afb6a9f36b299cec64a302b6e47e92fb.jpg](../iclr_results/2722_RECAST_%20Reparameterized%2C%20Compact%20weight%20Adaptation%20for%20Sequential%20Tasks/tables/bdf7ebf526ea729815e16080caba7340afb6a9f36b299cec64a302b6e47e92fb.jpg)

## MMSearch: Unveiling the Potential of Large Models as Multi-modal Search Engines


### Images

![035a229d1f3d5b5dd9ce0dfbc3218ad7ca203b403d6d6d634abdfe98993bffd4.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/035a229d1f3d5b5dd9ce0dfbc3218ad7ca203b403d6d6d634abdfe98993bffd4.jpg)

![037860c870e1c5ff1933076795c9fa2b1a22583a2e524ed920f96b956dcbd7e7.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/037860c870e1c5ff1933076795c9fa2b1a22583a2e524ed920f96b956dcbd7e7.jpg)

![03de2664b0bd73f64ba75e8c6b888b330afd97c4bd07d4859d402587d46182b1.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/03de2664b0bd73f64ba75e8c6b888b330afd97c4bd07d4859d402587d46182b1.jpg)

![0811d837f4a41df9586b98dd1a9a31ccf3b62434b83f6413782a85645c250651.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/0811d837f4a41df9586b98dd1a9a31ccf3b62434b83f6413782a85645c250651.jpg)

![0a4ccb516b5847913eb30cd1c8b45811c456a25991ef07babbccd7b4bbcbf90f.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/0a4ccb516b5847913eb30cd1c8b45811c456a25991ef07babbccd7b4bbcbf90f.jpg)

![0ac09d5ecc6ccad89d3e0a520e478eeadf0ded5e1e58a35cacf62e3439acb138.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/0ac09d5ecc6ccad89d3e0a520e478eeadf0ded5e1e58a35cacf62e3439acb138.jpg)

![0b6e452b69629e9f5ccebce8b4e8f248a30d12299c4655960fab3163057d8768.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/0b6e452b69629e9f5ccebce8b4e8f248a30d12299c4655960fab3163057d8768.jpg)

![0be7a704b34f40743848ab9f9a05f7d008d6c9e84cb8d9217f6e7159eee1a036.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/0be7a704b34f40743848ab9f9a05f7d008d6c9e84cb8d9217f6e7159eee1a036.jpg)

![0dbfc52a9ec755732dd8d12421c5dc60d34bef063c0e0234826ee33afd71c33b.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/0dbfc52a9ec755732dd8d12421c5dc60d34bef063c0e0234826ee33afd71c33b.jpg)

![0f9aca3846c096eeb934aa36885f2760c0b45e1258df04ed9a4105c49c3db7c6.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/0f9aca3846c096eeb934aa36885f2760c0b45e1258df04ed9a4105c49c3db7c6.jpg)

![1020b680080df2434f64440ab6b4c29c90ba758846ddd312d84f5acc63240220.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/1020b680080df2434f64440ab6b4c29c90ba758846ddd312d84f5acc63240220.jpg)

![1273b28b31fa106354cff0dc1178b31aca0e921ac536db52d8068e339709a09e.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/1273b28b31fa106354cff0dc1178b31aca0e921ac536db52d8068e339709a09e.jpg)

![1291f0766ec1a53b3f5f315b48bbb2b1329202d94b4e1ea82cf11fc68123109a.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/1291f0766ec1a53b3f5f315b48bbb2b1329202d94b4e1ea82cf11fc68123109a.jpg)

![1bbf535840b8becd5a031408730f94dde2cacf8c5674853a674893247484c9ca.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/1bbf535840b8becd5a031408730f94dde2cacf8c5674853a674893247484c9ca.jpg)

![1c870d9e707ad2ec2398ff2378d9ee69295d3df127c65de3999ca3d21d0626de.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/1c870d9e707ad2ec2398ff2378d9ee69295d3df127c65de3999ca3d21d0626de.jpg)

![1ff3f59fe0a0aa69eb893d1d56cacde3ed299cd631ed2f43a82c01a8f341119e.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/1ff3f59fe0a0aa69eb893d1d56cacde3ed299cd631ed2f43a82c01a8f341119e.jpg)

![22e685e61fcaf1f8b2a5e7532181e3fceedb8340e0d0c1bc5b1242a680d0997b.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/22e685e61fcaf1f8b2a5e7532181e3fceedb8340e0d0c1bc5b1242a680d0997b.jpg)

![2af4eb2b313fea3fa881599b7bf551ee92371cd5b249618f6be865ec594e35ea.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/2af4eb2b313fea3fa881599b7bf551ee92371cd5b249618f6be865ec594e35ea.jpg)

![2bfef0d4419767be1e2ddda2dc21a593390a939ee7c41e206ed69d0736459731.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/2bfef0d4419767be1e2ddda2dc21a593390a939ee7c41e206ed69d0736459731.jpg)

![2e485afb5bade2d26ba340453eefe29134ccaceaa80b200a6faf478127ffebbf.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/2e485afb5bade2d26ba340453eefe29134ccaceaa80b200a6faf478127ffebbf.jpg)

![33908a6fb8feada91bf595e3cf2f4532c6433b973dc6a4eaeabf8480d5a3d497.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/33908a6fb8feada91bf595e3cf2f4532c6433b973dc6a4eaeabf8480d5a3d497.jpg)

![39eb383ca100a18f36e896cfe29a5028e41a1a25bcd114386a694dff521088af.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/39eb383ca100a18f36e896cfe29a5028e41a1a25bcd114386a694dff521088af.jpg)

![3c521623ba671fdb97fdf29f843d5d53198600dfd0a474e5a84a0246ee484c7f.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/3c521623ba671fdb97fdf29f843d5d53198600dfd0a474e5a84a0246ee484c7f.jpg)

![3ef518df3c932a9097e82dde3a7b88ab67303c2f482c41069ed1f68f0347f19b.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/3ef518df3c932a9097e82dde3a7b88ab67303c2f482c41069ed1f68f0347f19b.jpg)

![4e40090469aaf26a4cf4f68494f9a0e346697e8ffe3ddf1794cfcb1dc017305e.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/4e40090469aaf26a4cf4f68494f9a0e346697e8ffe3ddf1794cfcb1dc017305e.jpg)

![521dbf8bb41ddb26806c0a964cdd3764043077f65789ad05cc0082cd103c45a7.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/521dbf8bb41ddb26806c0a964cdd3764043077f65789ad05cc0082cd103c45a7.jpg)

![54eb05c28a42a22ab7901fc210a8a30888f584c2d2e8a46283ba1501f9ee998f.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/54eb05c28a42a22ab7901fc210a8a30888f584c2d2e8a46283ba1501f9ee998f.jpg)

![58f39e3c27327cce947dc54b7d44403da1f35a405941822f36579a85b5bd67c7.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/58f39e3c27327cce947dc54b7d44403da1f35a405941822f36579a85b5bd67c7.jpg)

![598f595f34e8f6d56b5994bfe84717c5e2ec43ffe18f0133563644b00a451294.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/598f595f34e8f6d56b5994bfe84717c5e2ec43ffe18f0133563644b00a451294.jpg)

![5c19ebc6da6df2ca6229a09943d647f342cc8bd48d6176e85b5e5465bb412462.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/5c19ebc6da6df2ca6229a09943d647f342cc8bd48d6176e85b5e5465bb412462.jpg)

![5d9462a63ff00e18812c4e8fb5cb9f438a00ea93e397932e8ffc65767ab05d2e.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/5d9462a63ff00e18812c4e8fb5cb9f438a00ea93e397932e8ffc65767ab05d2e.jpg)

![64805712b401662d924eac4183ba10709317286ee458c0449509757b2009d369.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/64805712b401662d924eac4183ba10709317286ee458c0449509757b2009d369.jpg)

![67093fc226dba5d377439fc222a3491f6e4a88fb2f229f7d2c2cf4fc67afe096.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/67093fc226dba5d377439fc222a3491f6e4a88fb2f229f7d2c2cf4fc67afe096.jpg)

![6cebeb05f280fbb53c6ebd50d4544864934cf91e04242dca7b04a7cf183921a0.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/6cebeb05f280fbb53c6ebd50d4544864934cf91e04242dca7b04a7cf183921a0.jpg)

![6df4def8e917dd4bb2c2e00192f9e0debba5bb4635f84d99b05943ac75aa4d38.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/6df4def8e917dd4bb2c2e00192f9e0debba5bb4635f84d99b05943ac75aa4d38.jpg)

![749440d1263a30bfbd0644881d40ba89036f152c6770e6a32186ae2a9c395e06.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/749440d1263a30bfbd0644881d40ba89036f152c6770e6a32186ae2a9c395e06.jpg)

![7d7c4cc284e3ebc6496d21b213c3b95eff35baec9f9eadacba4cad2cf5d25728.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/7d7c4cc284e3ebc6496d21b213c3b95eff35baec9f9eadacba4cad2cf5d25728.jpg)

![84838b0fb8c290254204cae8d1bc39d8ee2073543de4502205592bd86b8cb486.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/84838b0fb8c290254204cae8d1bc39d8ee2073543de4502205592bd86b8cb486.jpg)

![9581f0697841a7f4809166c0d27f9e64855492852392a6dc73a387dffa8e73e7.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/9581f0697841a7f4809166c0d27f9e64855492852392a6dc73a387dffa8e73e7.jpg)

![961858aaed9bb269a7cefefb70ccfa1eb10b34702864c268cd8647867633192d.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/961858aaed9bb269a7cefefb70ccfa1eb10b34702864c268cd8647867633192d.jpg)

![a0d40bb028c9a444ac047a847ca812ac8d0f9b7d1c2345540910b80f07dbc48c.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/a0d40bb028c9a444ac047a847ca812ac8d0f9b7d1c2345540910b80f07dbc48c.jpg)

![a0eb1202f93f585437d862f32092aa56c7ada12264dd1cbdfafc395955294bb4.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/a0eb1202f93f585437d862f32092aa56c7ada12264dd1cbdfafc395955294bb4.jpg)

![bbc2ef369788adf136dbc5992f1b6537258571826a28d92739dd824dbb6cc2f1.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/bbc2ef369788adf136dbc5992f1b6537258571826a28d92739dd824dbb6cc2f1.jpg)

![be06ca37b9cbca1098e07fc0a0b0207527b7e20ae282829336823964c02abde3.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/be06ca37b9cbca1098e07fc0a0b0207527b7e20ae282829336823964c02abde3.jpg)

![c0b82d1c86be8fd30e84a1d1a53a934a7aaaef445b2a3647b5db67d97d45e0da.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/c0b82d1c86be8fd30e84a1d1a53a934a7aaaef445b2a3647b5db67d97d45e0da.jpg)

![c1eab380bc3cb3a3d8fd65a85093e52865e49fb558d4263911ed2bff9f8c43e4.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/c1eab380bc3cb3a3d8fd65a85093e52865e49fb558d4263911ed2bff9f8c43e4.jpg)

![c6b52a0c9f2f51c68c7ee533e9e0ae5aa10d27b91a427e2827c7d20def436c58.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/c6b52a0c9f2f51c68c7ee533e9e0ae5aa10d27b91a427e2827c7d20def436c58.jpg)

![c958c59d4efb63f6ec4b902f190a999ebd0b4ca4f126ca127dbe56a065f5a5fa.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/c958c59d4efb63f6ec4b902f190a999ebd0b4ca4f126ca127dbe56a065f5a5fa.jpg)

![c9f903e38298f930c00668d7cda8978bdcca06155271782a60a696c95f1e573d.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/c9f903e38298f930c00668d7cda8978bdcca06155271782a60a696c95f1e573d.jpg)

![cc8f3b9fc23c1cfbd0d71b9588d75e437dec6c115db37aa014f5dab4a351e88a.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/cc8f3b9fc23c1cfbd0d71b9588d75e437dec6c115db37aa014f5dab4a351e88a.jpg)

![cda9f7001b4f2fc7b0fbe60f14c55bdecd703b17a0db8768bf3363b93d18da7d.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/cda9f7001b4f2fc7b0fbe60f14c55bdecd703b17a0db8768bf3363b93d18da7d.jpg)

![ce638d3b8f6a3ab88d42cb89b378245b4024cace583939a9298c2dd3e2a2983c.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/ce638d3b8f6a3ab88d42cb89b378245b4024cace583939a9298c2dd3e2a2983c.jpg)

![d011e0d5398aff0fd0c6fe0ccb472841615347ae74fefb51ed2c33ded5248ccd.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/d011e0d5398aff0fd0c6fe0ccb472841615347ae74fefb51ed2c33ded5248ccd.jpg)

![d31208ad588698d80d61e3a80d8b2017ce4c8aaaa371c3c4a5ae068e07f41af4.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/d31208ad588698d80d61e3a80d8b2017ce4c8aaaa371c3c4a5ae068e07f41af4.jpg)

![d312cc19049a7ab1fa5486d707572706f0476ea66f139d325b157334f5fe7535.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/d312cc19049a7ab1fa5486d707572706f0476ea66f139d325b157334f5fe7535.jpg)

![d62c7449d1f348257566f0fbd2e4ccfc141cdf7109ded367d6dc20d40335042d.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/d62c7449d1f348257566f0fbd2e4ccfc141cdf7109ded367d6dc20d40335042d.jpg)

![e037b3df614c37c781d764b20325cb2885023213420f1cc198407d35ad2191cd.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/e037b3df614c37c781d764b20325cb2885023213420f1cc198407d35ad2191cd.jpg)

![e19f86e391e4dbb5271bf3b1d173ed3b32ebb74eb77b233db6c024e1de483b43.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/e19f86e391e4dbb5271bf3b1d173ed3b32ebb74eb77b233db6c024e1de483b43.jpg)

![e4dddcbcf856f8a6a99617ea64c1ec5e9d51684df30518721f806affedcfa769.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/e4dddcbcf856f8a6a99617ea64c1ec5e9d51684df30518721f806affedcfa769.jpg)

![f4d8d6732422d9d36cea46af354c5a35998d0492a5d7677e24c95b375ee6c8a5.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/f4d8d6732422d9d36cea46af354c5a35998d0492a5d7677e24c95b375ee6c8a5.jpg)

![f4df259e8e497df0f36e3266e62e2231170e3c59c40e0446b97b6f0d9d7f7f2d.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/f4df259e8e497df0f36e3266e62e2231170e3c59c40e0446b97b6f0d9d7f7f2d.jpg)

![fe209cd1c30c3ed98cf6e31828db87e5ce0b518dd61d44c30472dd29c229787d.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/images/fe209cd1c30c3ed98cf6e31828db87e5ce0b518dd61d44c30472dd29c229787d.jpg)

### Tables

![14e0f57dfb6007989e1206723ea69ef6b7fbaa3a676129a33e43dfcd139ffcf4.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/tables/14e0f57dfb6007989e1206723ea69ef6b7fbaa3a676129a33e43dfcd139ffcf4.jpg)

![776d567d4204cdf43118de2960d0c269d1dfaece8905d48474d77558c0e477ab.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/tables/776d567d4204cdf43118de2960d0c269d1dfaece8905d48474d77558c0e477ab.jpg)

![8fe1f1821fba36fd033ca93f92550b06a7f634bb1c2a42c8afcbd24b5a58acd3.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/tables/8fe1f1821fba36fd033ca93f92550b06a7f634bb1c2a42c8afcbd24b5a58acd3.jpg)

![a3c94e492e8f545865c9f1d7b180ca8bb7de588ec717c4f52f666cf76e533826.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/tables/a3c94e492e8f545865c9f1d7b180ca8bb7de588ec717c4f52f666cf76e533826.jpg)

![b96dc1ac6007fb9d2ee9d8417ff41c79a157b3665f6dec33c211aad7482d47f2.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/tables/b96dc1ac6007fb9d2ee9d8417ff41c79a157b3665f6dec33c211aad7482d47f2.jpg)

![e102fa9038a5a6cec732b7adc8e4bd2bc707a445259a1410c09e398fe8be0725.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/tables/e102fa9038a5a6cec732b7adc8e4bd2bc707a445259a1410c09e398fe8be0725.jpg)

![ed154d965aec6ba6ddf974a0cc2703f8ac449d837f250e4cfe45b4e7f302fc69.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/tables/ed154d965aec6ba6ddf974a0cc2703f8ac449d837f250e4cfe45b4e7f302fc69.jpg)

![eea05e76b8d629173cff700e204a9db2dd4ddc20011156b410a213c131d9567e.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/tables/eea05e76b8d629173cff700e204a9db2dd4ddc20011156b410a213c131d9567e.jpg)

![f61c336104e4f0c82ecb61e7a0a8abdc447ab84ccafd5aff2891b4b263cd13f9.jpg](../iclr_results/2723_MMSearch_%20Unveiling%20the%20Potential%20of%20Large%20Models%20as%20Multi-modal%20Search%20Engines/tables/f61c336104e4f0c82ecb61e7a0a8abdc447ab84ccafd5aff2891b4b263cd13f9.jpg)

## ToolDial: Multi-turn Dialogue Generation Method for Tool-Augmented Language Models


### Images

![5a8fcf797c6b38bac3cab43b7425627244716d4a1efabb63abf55572105bf826.jpg](../iclr_results/2724_ToolDial_%20Multi-turn%20Dialogue%20Generation%20Method%20for%20Tool-Augmented%20Language%20Models/images/5a8fcf797c6b38bac3cab43b7425627244716d4a1efabb63abf55572105bf826.jpg)

![7350cc64a0bbe6b03ede95d7fb37def9e476c87f3213270336d02406f3700c74.jpg](../iclr_results/2724_ToolDial_%20Multi-turn%20Dialogue%20Generation%20Method%20for%20Tool-Augmented%20Language%20Models/images/7350cc64a0bbe6b03ede95d7fb37def9e476c87f3213270336d02406f3700c74.jpg)

![7c84605c561be3d72f3d616c9002b3513a01da5e8a60018c8ec684dd30781c47.jpg](../iclr_results/2724_ToolDial_%20Multi-turn%20Dialogue%20Generation%20Method%20for%20Tool-Augmented%20Language%20Models/images/7c84605c561be3d72f3d616c9002b3513a01da5e8a60018c8ec684dd30781c47.jpg)

![bfe6697039341efce741fa91b04ea20fb32b5d08a1da8118e478316215ccffb6.jpg](../iclr_results/2724_ToolDial_%20Multi-turn%20Dialogue%20Generation%20Method%20for%20Tool-Augmented%20Language%20Models/images/bfe6697039341efce741fa91b04ea20fb32b5d08a1da8118e478316215ccffb6.jpg)

![eba2e767a51c9fa5d32af0ffedf06f26f1109c2d3ca7a63702c24595f773f83b.jpg](../iclr_results/2724_ToolDial_%20Multi-turn%20Dialogue%20Generation%20Method%20for%20Tool-Augmented%20Language%20Models/images/eba2e767a51c9fa5d32af0ffedf06f26f1109c2d3ca7a63702c24595f773f83b.jpg)

### Tables

![093aae869e01b10b3ed12e782ce7b8e0347f20dd962ceca3bb38af399875713a.jpg](../iclr_results/2724_ToolDial_%20Multi-turn%20Dialogue%20Generation%20Method%20for%20Tool-Augmented%20Language%20Models/tables/093aae869e01b10b3ed12e782ce7b8e0347f20dd962ceca3bb38af399875713a.jpg)

![0b76ff22854e9f4c6801ad2ae21fbcb74c62d27c9fece6ff76ca33e001d5222c.jpg](../iclr_results/2724_ToolDial_%20Multi-turn%20Dialogue%20Generation%20Method%20for%20Tool-Augmented%20Language%20Models/tables/0b76ff22854e9f4c6801ad2ae21fbcb74c62d27c9fece6ff76ca33e001d5222c.jpg)

![1167b2364a38a691f672c31d271f727d2c520b1177a1ca2a4ee648e73eded25b.jpg](../iclr_results/2724_ToolDial_%20Multi-turn%20Dialogue%20Generation%20Method%20for%20Tool-Augmented%20Language%20Models/tables/1167b2364a38a691f672c31d271f727d2c520b1177a1ca2a4ee648e73eded25b.jpg)

![3c8e0d486db6f41b86f036d4cc1546cf93f1544117e60745bbd6230bca00e609.jpg](../iclr_results/2724_ToolDial_%20Multi-turn%20Dialogue%20Generation%20Method%20for%20Tool-Augmented%20Language%20Models/tables/3c8e0d486db6f41b86f036d4cc1546cf93f1544117e60745bbd6230bca00e609.jpg)

![4e9e7d2214711e2b1bc1db5b6b8419d7d89041008b65d6e8a3d4074b0ad0f934.jpg](../iclr_results/2724_ToolDial_%20Multi-turn%20Dialogue%20Generation%20Method%20for%20Tool-Augmented%20Language%20Models/tables/4e9e7d2214711e2b1bc1db5b6b8419d7d89041008b65d6e8a3d4074b0ad0f934.jpg)

![5e9c0d4a6e7d5c79d9e2a4efcc0a03b0c4b9817a4bbb105cf960a62d5d0b1f19.jpg](../iclr_results/2724_ToolDial_%20Multi-turn%20Dialogue%20Generation%20Method%20for%20Tool-Augmented%20Language%20Models/tables/5e9c0d4a6e7d5c79d9e2a4efcc0a03b0c4b9817a4bbb105cf960a62d5d0b1f19.jpg)

![77cdb0423a84e993208c2814114c2c9f8e9693fae64f955767d5f4a0158ef5be.jpg](../iclr_results/2724_ToolDial_%20Multi-turn%20Dialogue%20Generation%20Method%20for%20Tool-Augmented%20Language%20Models/tables/77cdb0423a84e993208c2814114c2c9f8e9693fae64f955767d5f4a0158ef5be.jpg)

![7b36b2041fb58b0345b5250cb6c330ba0e2aa12e42ee2af1a4f674dfeebeca61.jpg](../iclr_results/2724_ToolDial_%20Multi-turn%20Dialogue%20Generation%20Method%20for%20Tool-Augmented%20Language%20Models/tables/7b36b2041fb58b0345b5250cb6c330ba0e2aa12e42ee2af1a4f674dfeebeca61.jpg)

![819ea42440c07ec778c2f477679ba9fe89be2b7cdcb44df20d1ea2a0f737c55d.jpg](../iclr_results/2724_ToolDial_%20Multi-turn%20Dialogue%20Generation%20Method%20for%20Tool-Augmented%20Language%20Models/tables/819ea42440c07ec778c2f477679ba9fe89be2b7cdcb44df20d1ea2a0f737c55d.jpg)

![8691e7a6835b9bbc9abf06e23c6efbcf14765d34cbf022ff3a97e77d87612f08.jpg](../iclr_results/2724_ToolDial_%20Multi-turn%20Dialogue%20Generation%20Method%20for%20Tool-Augmented%20Language%20Models/tables/8691e7a6835b9bbc9abf06e23c6efbcf14765d34cbf022ff3a97e77d87612f08.jpg)

![97c01bdf067017b644348373b2975c903adc41f1ec7df42676244585e9f5897c.jpg](../iclr_results/2724_ToolDial_%20Multi-turn%20Dialogue%20Generation%20Method%20for%20Tool-Augmented%20Language%20Models/tables/97c01bdf067017b644348373b2975c903adc41f1ec7df42676244585e9f5897c.jpg)

![bf4bf1e30c0377c220c9d9b1b6a9fe7a904337230cb4c895a9223839ecd2cf4f.jpg](../iclr_results/2724_ToolDial_%20Multi-turn%20Dialogue%20Generation%20Method%20for%20Tool-Augmented%20Language%20Models/tables/bf4bf1e30c0377c220c9d9b1b6a9fe7a904337230cb4c895a9223839ecd2cf4f.jpg)

![ccc50d7b9bab1aa46d30d1da3d5cc7eec9b9f7aa9b7db758acc86e3e4d4ac4de.jpg](../iclr_results/2724_ToolDial_%20Multi-turn%20Dialogue%20Generation%20Method%20for%20Tool-Augmented%20Language%20Models/tables/ccc50d7b9bab1aa46d30d1da3d5cc7eec9b9f7aa9b7db758acc86e3e4d4ac4de.jpg)

![e3b15ecd9785c956becf52ae93170a83a524ba08dfac43d7a0e4ce5409d773ae.jpg](../iclr_results/2724_ToolDial_%20Multi-turn%20Dialogue%20Generation%20Method%20for%20Tool-Augmented%20Language%20Models/tables/e3b15ecd9785c956becf52ae93170a83a524ba08dfac43d7a0e4ce5409d773ae.jpg)

![f00d85a5041d2548ab1551d8de70a4631967ce832a3e3551bea12e5093bcc82a.jpg](../iclr_results/2724_ToolDial_%20Multi-turn%20Dialogue%20Generation%20Method%20for%20Tool-Augmented%20Language%20Models/tables/f00d85a5041d2548ab1551d8de70a4631967ce832a3e3551bea12e5093bcc82a.jpg)

![f6808f34267aa93c99d2082340501ac5f71c05c0460a5cf5e3834d1ece7985d3.jpg](../iclr_results/2724_ToolDial_%20Multi-turn%20Dialogue%20Generation%20Method%20for%20Tool-Augmented%20Language%20Models/tables/f6808f34267aa93c99d2082340501ac5f71c05c0460a5cf5e3834d1ece7985d3.jpg)

![f8800950c47597c98913aff1b9eb8ad8997fb5e3b92505ec3e93a48c21337736.jpg](../iclr_results/2724_ToolDial_%20Multi-turn%20Dialogue%20Generation%20Method%20for%20Tool-Augmented%20Language%20Models/tables/f8800950c47597c98913aff1b9eb8ad8997fb5e3b92505ec3e93a48c21337736.jpg)

![fb6af45bda4bc9d032909cc55d4fd2d3eef8bacd4bb0b1a1915e74c3577be01f.jpg](../iclr_results/2724_ToolDial_%20Multi-turn%20Dialogue%20Generation%20Method%20for%20Tool-Augmented%20Language%20Models/tables/fb6af45bda4bc9d032909cc55d4fd2d3eef8bacd4bb0b1a1915e74c3577be01f.jpg)

## GenARM: Reward Guided Generation with Autoregressive Reward Model for Test-Time Alignment


### Images

![17b91fd2555c31314ce7a0a2d769fd3d83ac5ea5e9e94fa5e52364aacc3a9c52.jpg](../iclr_results/2725_GenARM_%20Reward%20Guided%20Generation%20with%20Autoregressive%20Reward%20Model%20for%20Test-Time%20Alignment/images/17b91fd2555c31314ce7a0a2d769fd3d83ac5ea5e9e94fa5e52364aacc3a9c52.jpg)

![21499fc0857dd63f85b8c40b7ccd44074e4987268f006be780805d71ec4f808e.jpg](../iclr_results/2725_GenARM_%20Reward%20Guided%20Generation%20with%20Autoregressive%20Reward%20Model%20for%20Test-Time%20Alignment/images/21499fc0857dd63f85b8c40b7ccd44074e4987268f006be780805d71ec4f808e.jpg)

![52b46aa8a0d37895681d8461f483af3932e91e1de548d42db00f765d4932a205.jpg](../iclr_results/2725_GenARM_%20Reward%20Guided%20Generation%20with%20Autoregressive%20Reward%20Model%20for%20Test-Time%20Alignment/images/52b46aa8a0d37895681d8461f483af3932e91e1de548d42db00f765d4932a205.jpg)

![6310c76637bd9c52939afff899aab188cca3888e255b03a93defcf17d14e00da.jpg](../iclr_results/2725_GenARM_%20Reward%20Guided%20Generation%20with%20Autoregressive%20Reward%20Model%20for%20Test-Time%20Alignment/images/6310c76637bd9c52939afff899aab188cca3888e255b03a93defcf17d14e00da.jpg)

![894147fe10ba73a6092e4ff773cbd88f231af0c9903ae5a3e66d3ceb3fb7f606.jpg](../iclr_results/2725_GenARM_%20Reward%20Guided%20Generation%20with%20Autoregressive%20Reward%20Model%20for%20Test-Time%20Alignment/images/894147fe10ba73a6092e4ff773cbd88f231af0c9903ae5a3e66d3ceb3fb7f606.jpg)

![900f187f72fefdb9ab94d36d21adbc4e15542be73a1d7ed8d8733b0b2338803f.jpg](../iclr_results/2725_GenARM_%20Reward%20Guided%20Generation%20with%20Autoregressive%20Reward%20Model%20for%20Test-Time%20Alignment/images/900f187f72fefdb9ab94d36d21adbc4e15542be73a1d7ed8d8733b0b2338803f.jpg)

![c34948a817fbb6267de215f74726ada54a5618714e9066e217602a641582a883.jpg](../iclr_results/2725_GenARM_%20Reward%20Guided%20Generation%20with%20Autoregressive%20Reward%20Model%20for%20Test-Time%20Alignment/images/c34948a817fbb6267de215f74726ada54a5618714e9066e217602a641582a883.jpg)

![e7c6ea3a9587a70de1e012fe883ab6aa70bbf285aa3d8df4c81aeeaa6f07ef44.jpg](../iclr_results/2725_GenARM_%20Reward%20Guided%20Generation%20with%20Autoregressive%20Reward%20Model%20for%20Test-Time%20Alignment/images/e7c6ea3a9587a70de1e012fe883ab6aa70bbf285aa3d8df4c81aeeaa6f07ef44.jpg)

### Tables

![061e4c809fa6c069499e4a573aeca0a86ee66481845671ba924193bd4d04b645.jpg](../iclr_results/2725_GenARM_%20Reward%20Guided%20Generation%20with%20Autoregressive%20Reward%20Model%20for%20Test-Time%20Alignment/tables/061e4c809fa6c069499e4a573aeca0a86ee66481845671ba924193bd4d04b645.jpg)

![69b5f860215ed0c07bd0a5f057bb163c9f26ae0b7238dfc818f2d6019eeb8e2d.jpg](../iclr_results/2725_GenARM_%20Reward%20Guided%20Generation%20with%20Autoregressive%20Reward%20Model%20for%20Test-Time%20Alignment/tables/69b5f860215ed0c07bd0a5f057bb163c9f26ae0b7238dfc818f2d6019eeb8e2d.jpg)

![d62ee263dbaa75b58c49b8464ee6237646d6cef936e009ae6cedffc15d97cf5e.jpg](../iclr_results/2725_GenARM_%20Reward%20Guided%20Generation%20with%20Autoregressive%20Reward%20Model%20for%20Test-Time%20Alignment/tables/d62ee263dbaa75b58c49b8464ee6237646d6cef936e009ae6cedffc15d97cf5e.jpg)

## Learning to Steer Markovian Agents under Model Uncertainty


### Images

![79366d09b1bf4b006fdfcd8f551b2f62944a3c202e1af6b104fa07b879ceb4b4.jpg](../iclr_results/2726_Learning%20to%20Steer%20Markovian%20Agents%20under%20Model%20Uncertainty/images/79366d09b1bf4b006fdfcd8f551b2f62944a3c202e1af6b104fa07b879ceb4b4.jpg)

![899fafafb55fd0115ea4323c94b9c28471bb39106ad556e315f0dabf3fd6d02e.jpg](../iclr_results/2726_Learning%20to%20Steer%20Markovian%20Agents%20under%20Model%20Uncertainty/images/899fafafb55fd0115ea4323c94b9c28471bb39106ad556e315f0dabf3fd6d02e.jpg)

![8c0832e78f49bbdda9325767dd4af8e0053eca8816f1b43284c8a7ad7d518079.jpg](../iclr_results/2726_Learning%20to%20Steer%20Markovian%20Agents%20under%20Model%20Uncertainty/images/8c0832e78f49bbdda9325767dd4af8e0053eca8816f1b43284c8a7ad7d518079.jpg)

![9083464362341c7b332c3b62a65dfa671eabac28a5f5b0e743403c782b7b0a40.jpg](../iclr_results/2726_Learning%20to%20Steer%20Markovian%20Agents%20under%20Model%20Uncertainty/images/9083464362341c7b332c3b62a65dfa671eabac28a5f5b0e743403c782b7b0a40.jpg)

![a125db0d9df10026c725e8169c03d646e6196aad8f8778cc96adcccdb8198363.jpg](../iclr_results/2726_Learning%20to%20Steer%20Markovian%20Agents%20under%20Model%20Uncertainty/images/a125db0d9df10026c725e8169c03d646e6196aad8f8778cc96adcccdb8198363.jpg)

![a163d2be2acd2b7d233e039b8646bdf4c8a1aea3c001f7183cff3af477dcb17e.jpg](../iclr_results/2726_Learning%20to%20Steer%20Markovian%20Agents%20under%20Model%20Uncertainty/images/a163d2be2acd2b7d233e039b8646bdf4c8a1aea3c001f7183cff3af477dcb17e.jpg)

### Tables

![134e0fe905b20e23fd44b3ec9923a6b33b512d3a28764f4d62d850d1f5d1669c.jpg](../iclr_results/2726_Learning%20to%20Steer%20Markovian%20Agents%20under%20Model%20Uncertainty/tables/134e0fe905b20e23fd44b3ec9923a6b33b512d3a28764f4d62d850d1f5d1669c.jpg)

![4f6fa977fa71f6afeb47e0148cfc28834489a4163de997428ef68a13c046f1ec.jpg](../iclr_results/2726_Learning%20to%20Steer%20Markovian%20Agents%20under%20Model%20Uncertainty/tables/4f6fa977fa71f6afeb47e0148cfc28834489a4163de997428ef68a13c046f1ec.jpg)

![5ccd06d7ca34f1dc1dad138079e0d00d36d2489ee866321db3f93fa348aba26f.jpg](../iclr_results/2726_Learning%20to%20Steer%20Markovian%20Agents%20under%20Model%20Uncertainty/tables/5ccd06d7ca34f1dc1dad138079e0d00d36d2489ee866321db3f93fa348aba26f.jpg)

![7840eb84186e86b51afb7a0c135506698fa1ec7a27c39d49f82dc9fe816e725a.jpg](../iclr_results/2726_Learning%20to%20Steer%20Markovian%20Agents%20under%20Model%20Uncertainty/tables/7840eb84186e86b51afb7a0c135506698fa1ec7a27c39d49f82dc9fe816e725a.jpg)

![97b6259ca224fffa1baf637097d497912b21ace301d0a724b54be4fb9668e821.jpg](../iclr_results/2726_Learning%20to%20Steer%20Markovian%20Agents%20under%20Model%20Uncertainty/tables/97b6259ca224fffa1baf637097d497912b21ace301d0a724b54be4fb9668e821.jpg)

## What Makes a Maze Look Like a Maze?


### Images

![06e9d8ed90821337ab22c15b6b6c6d42b83ccb495113d73cb20e85cdcba16dea.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/06e9d8ed90821337ab22c15b6b6c6d42b83ccb495113d73cb20e85cdcba16dea.jpg)

![0cd67377f63cf6ccdfec088db03fbc4cb1af07ac1a019ed8def037366834bb96.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/0cd67377f63cf6ccdfec088db03fbc4cb1af07ac1a019ed8def037366834bb96.jpg)

![155d68622e27a62360499248b0f42388b0a11be38c1fa1c20ce1f03ad1f27ad0.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/155d68622e27a62360499248b0f42388b0a11be38c1fa1c20ce1f03ad1f27ad0.jpg)

![1a8720ecf4abd29d8afd60ecda2b172827c739139cb5ec6ffb5c3f2379ee3497.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/1a8720ecf4abd29d8afd60ecda2b172827c739139cb5ec6ffb5c3f2379ee3497.jpg)

![1d7bf796dd6de3011f37630fc6023ad793e44043efd0632632f7afff6558b3d1.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/1d7bf796dd6de3011f37630fc6023ad793e44043efd0632632f7afff6558b3d1.jpg)

![1e0dc19c4188a6e47783e899f266259d826896c17fe7ccfb5d72f24be0d44e36.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/1e0dc19c4188a6e47783e899f266259d826896c17fe7ccfb5d72f24be0d44e36.jpg)

![237cad08d95d327108d2baf2fa538c311e3c0a92f7e53998c78d8d308aa7c79f.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/237cad08d95d327108d2baf2fa538c311e3c0a92f7e53998c78d8d308aa7c79f.jpg)

![28774917bd2ddbf72a15f1f313b04eb7956e1b834553dafde8756fbf3d489681.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/28774917bd2ddbf72a15f1f313b04eb7956e1b834553dafde8756fbf3d489681.jpg)

![2d46b18efc0a1676910f3e3d6bfad7217853777772fd33009845f1d14232ac58.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/2d46b18efc0a1676910f3e3d6bfad7217853777772fd33009845f1d14232ac58.jpg)

![3c72acfc610424212536c19aa16cde9e8148b3a62dbf6c9eb304e85817a0a56b.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/3c72acfc610424212536c19aa16cde9e8148b3a62dbf6c9eb304e85817a0a56b.jpg)

![4f891f6de33d1377520b74682b33f1c4a9f3eb97646bc609f7d7d9120def241e.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/4f891f6de33d1377520b74682b33f1c4a9f3eb97646bc609f7d7d9120def241e.jpg)

![4fc09144a479c8c905004abacc20f22867379deee90b803ee2b03ac2885562de.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/4fc09144a479c8c905004abacc20f22867379deee90b803ee2b03ac2885562de.jpg)

![4fe7d7c14d33b88733b2278277107ce10fb1578f9c047ed492b5bc69f35333a6.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/4fe7d7c14d33b88733b2278277107ce10fb1578f9c047ed492b5bc69f35333a6.jpg)

![5127d93dcd1128afa40e548abe174f9ca09a9d3dd49b918524b4278d3cb46a08.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/5127d93dcd1128afa40e548abe174f9ca09a9d3dd49b918524b4278d3cb46a08.jpg)

![5302c443208d359863c0b2d92d7e62415da6d9f1deb23db3aa55b1138bdd9f1f.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/5302c443208d359863c0b2d92d7e62415da6d9f1deb23db3aa55b1138bdd9f1f.jpg)

![5643a325ea6f5808e549855b1a3e50b52a97ca4b577ad0c892730b780593fc3f.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/5643a325ea6f5808e549855b1a3e50b52a97ca4b577ad0c892730b780593fc3f.jpg)

![5ce42abdd16ffab6acd72c01d43b3be28f8c15f535e4d040bc33f261ccaf8119.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/5ce42abdd16ffab6acd72c01d43b3be28f8c15f535e4d040bc33f261ccaf8119.jpg)

![6261df7cf85040536199980d6b64abce203b8e14d8e8a263cf42bc688a9d56e6.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/6261df7cf85040536199980d6b64abce203b8e14d8e8a263cf42bc688a9d56e6.jpg)

![62d09890938bac9f0ca06929491dc5689b2e44131e419cf44f0bb507da0351f3.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/62d09890938bac9f0ca06929491dc5689b2e44131e419cf44f0bb507da0351f3.jpg)

![65fd7fdf1903c1afa61c4d4577f6d63579849a12e48b94f8485e0254b354865e.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/65fd7fdf1903c1afa61c4d4577f6d63579849a12e48b94f8485e0254b354865e.jpg)

![7156a1f640b2fe43b555c54f41adddd614a3b3c815cf17d7eb9cbbe93930baed.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/7156a1f640b2fe43b555c54f41adddd614a3b3c815cf17d7eb9cbbe93930baed.jpg)

![788b4bb5a38481b3e4706bfe625e053bd6768ae1539c6d539612c3f96127a8a3.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/788b4bb5a38481b3e4706bfe625e053bd6768ae1539c6d539612c3f96127a8a3.jpg)

![78bb9bc225c1030f7910439df2798ce3911e8fbd452fc039191f4b21463e8fe2.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/78bb9bc225c1030f7910439df2798ce3911e8fbd452fc039191f4b21463e8fe2.jpg)

![83a9adda5559adb56125f4d50c0adbf8c5897d232944ca82ac9077e88b1dc334.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/83a9adda5559adb56125f4d50c0adbf8c5897d232944ca82ac9077e88b1dc334.jpg)

![8936938ed206924d07d1c1127c544583f48bd7c5d364b7a9ddba594db6143757.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/8936938ed206924d07d1c1127c544583f48bd7c5d364b7a9ddba594db6143757.jpg)

![9851c02adbc9dd30579c792579d7a4cd22c45285ef05a6e036f1780e0f301416.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/9851c02adbc9dd30579c792579d7a4cd22c45285ef05a6e036f1780e0f301416.jpg)

![9bc82352a07f872bb48cec71763a82c1f9c8db6f675047f8c12ecaa780944ea8.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/9bc82352a07f872bb48cec71763a82c1f9c8db6f675047f8c12ecaa780944ea8.jpg)

![a0f807473a5164efd141cdcbc500ab3cd6cd24396f50055cb32b5bc65acc3c7c.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/a0f807473a5164efd141cdcbc500ab3cd6cd24396f50055cb32b5bc65acc3c7c.jpg)

![ab121329facb9f629e0f632b764d40922b354a08122cc82a52bbfac24722ad0f.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/ab121329facb9f629e0f632b764d40922b354a08122cc82a52bbfac24722ad0f.jpg)

![ad77633119e6275306738107e01e14a568cffb9b813d71e805ada182b6fe7655.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/ad77633119e6275306738107e01e14a568cffb9b813d71e805ada182b6fe7655.jpg)

![ae6dfcfe468d82bf0c875368f14ed0f42ebafe20bd0616577aa762573262dcdc.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/ae6dfcfe468d82bf0c875368f14ed0f42ebafe20bd0616577aa762573262dcdc.jpg)

![b2525b4f023019d83437d9568683883d05dea0d98ecad7095e39cf61f5812269.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/b2525b4f023019d83437d9568683883d05dea0d98ecad7095e39cf61f5812269.jpg)

![b964a5e8080a43491fb78c7f88c6399ff591fe9afd012c9387dfd0e9b210e6c1.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/b964a5e8080a43491fb78c7f88c6399ff591fe9afd012c9387dfd0e9b210e6c1.jpg)

![c1ce4d127d2a1c1f52ca5e60fe30025f4e9d9aed1cc96fce87b3ddf365ffecb8.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/c1ce4d127d2a1c1f52ca5e60fe30025f4e9d9aed1cc96fce87b3ddf365ffecb8.jpg)

![c2392171b85e3e898de94bc3d6a920ba92c35eee4c9734d762628478b37725ca.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/c2392171b85e3e898de94bc3d6a920ba92c35eee4c9734d762628478b37725ca.jpg)

![c24cea62e0c980b0b3d72d4a6bef6c188bba43f197aad30f11ea13f9bb2db074.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/c24cea62e0c980b0b3d72d4a6bef6c188bba43f197aad30f11ea13f9bb2db074.jpg)

![cb6fe0c622f6679ffd409d496a415699376b5a1eec1fdec1899ed595f9c0b169.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/cb6fe0c622f6679ffd409d496a415699376b5a1eec1fdec1899ed595f9c0b169.jpg)

![cc93213ee24213842a0c1bd096e7d324529915c695db278e8dc3e40925f41be9.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/cc93213ee24213842a0c1bd096e7d324529915c695db278e8dc3e40925f41be9.jpg)

![cda5a09a3b0d7144dc26b3ed3d1dd9dff5e709962d38974ccef6d5676ce8a029.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/cda5a09a3b0d7144dc26b3ed3d1dd9dff5e709962d38974ccef6d5676ce8a029.jpg)

![ce2aa072a315ef7315a37e9112b191c882e7df9d7017c9d2ad7447f123c067d7.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/ce2aa072a315ef7315a37e9112b191c882e7df9d7017c9d2ad7447f123c067d7.jpg)

![d88c6068180758449e610d4eba947fc88e68d8d7f95ea044b39aa33f8d895f66.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/d88c6068180758449e610d4eba947fc88e68d8d7f95ea044b39aa33f8d895f66.jpg)

![dacbe21d4160a54dd45878268bfd5812df1ee2791a5ec640309f08e071b3b9b2.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/dacbe21d4160a54dd45878268bfd5812df1ee2791a5ec640309f08e071b3b9b2.jpg)

![e328e684814081b4ca343e8f1fbd145ed7645caeb26e0088642767bd184c7ee5.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/e328e684814081b4ca343e8f1fbd145ed7645caeb26e0088642767bd184c7ee5.jpg)

![e564a1a06a0df617a61ae188d9e7d25d2842e2db801b1bbb887700c655b1fb06.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/e564a1a06a0df617a61ae188d9e7d25d2842e2db801b1bbb887700c655b1fb06.jpg)

![eb0942a700a94c881275e5416af83e45940cd8dca0f6f72ab1c7ef36734485e6.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/eb0942a700a94c881275e5416af83e45940cd8dca0f6f72ab1c7ef36734485e6.jpg)

![f1a6b97a799e7ab0c30b5b7b74ef39fc94f7176bcc418e3ccf55f665718679a7.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/f1a6b97a799e7ab0c30b5b7b74ef39fc94f7176bcc418e3ccf55f665718679a7.jpg)

![f1ed24b20000056340d55cb02bbab6aa3b7d754f5d64724c05180c437c08c323.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/f1ed24b20000056340d55cb02bbab6aa3b7d754f5d64724c05180c437c08c323.jpg)

![f2383c94e8b100063542c0d3b816d74d842379cadc51d0c404f7f543d8c72837.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/f2383c94e8b100063542c0d3b816d74d842379cadc51d0c404f7f543d8c72837.jpg)

![f34cf744c611378f7c16e4a1e975bfd7489826bd601128b1c57b669e91a45cc7.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/f34cf744c611378f7c16e4a1e975bfd7489826bd601128b1c57b669e91a45cc7.jpg)

![f8ee7aa81d88a66c310c85416d5b3cbe7fd55a660d3136db81c226c5dab3c9e7.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/f8ee7aa81d88a66c310c85416d5b3cbe7fd55a660d3136db81c226c5dab3c9e7.jpg)

![f9ce7f31b4e37a31e1151d1e6fc5535e74102e06a476832654f578fb1889b594.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/f9ce7f31b4e37a31e1151d1e6fc5535e74102e06a476832654f578fb1889b594.jpg)

![fda1eea7c4d74f2090b0dc68c1c2dbe5566ea95c54df63559f2d711c6c826fb7.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/fda1eea7c4d74f2090b0dc68c1c2dbe5566ea95c54df63559f2d711c6c826fb7.jpg)

![fdac9cfa8571e1c520f2c5e6e2e4454f40e6ccb3ac238c64c27e3d6ad83b0689.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/images/fdac9cfa8571e1c520f2c5e6e2e4454f40e6ccb3ac238c64c27e3d6ad83b0689.jpg)

### Tables

![158fea64150a73b0881e0083340ccbc09a8581e3e5ce5f75abb52a479a1da832.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/tables/158fea64150a73b0881e0083340ccbc09a8581e3e5ce5f75abb52a479a1da832.jpg)

![1fc1b1ed709b5a6abbb719b74f110b9e7aa4d0e109015d57a51636786ffdef45.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/tables/1fc1b1ed709b5a6abbb719b74f110b9e7aa4d0e109015d57a51636786ffdef45.jpg)

![4274684e54bc3889d552afe8baa7051da729a0866b8495b968a2a70e9152b9a2.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/tables/4274684e54bc3889d552afe8baa7051da729a0866b8495b968a2a70e9152b9a2.jpg)

![499e698e912b92b5b3d8282fc4c4512875ab684fb12b8d879e509ae344865f02.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/tables/499e698e912b92b5b3d8282fc4c4512875ab684fb12b8d879e509ae344865f02.jpg)

![67058317fea1f817bdc85756f698b7cfa332043d3a2f856ab2debe6e60aeaaad.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/tables/67058317fea1f817bdc85756f698b7cfa332043d3a2f856ab2debe6e60aeaaad.jpg)

![6ab25e0870eb87f0adb5fd0e73f3b00ad62c13902f84ded5a7fcf65a845b6baf.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/tables/6ab25e0870eb87f0adb5fd0e73f3b00ad62c13902f84ded5a7fcf65a845b6baf.jpg)

![6dff94573b484f92be24ce30f6bc139ec82803acf68ca0bcf0bd54e50b4f0e3d.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/tables/6dff94573b484f92be24ce30f6bc139ec82803acf68ca0bcf0bd54e50b4f0e3d.jpg)

![82a79e1ebcd728182c014137bf33d6c554fd0b78d891e0a97e028b59b8265d4e.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/tables/82a79e1ebcd728182c014137bf33d6c554fd0b78d891e0a97e028b59b8265d4e.jpg)

![ac232b69b8fd20b8662806174e2e7246f1e37e9dcbbf5afa7a122b5d64a69b72.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/tables/ac232b69b8fd20b8662806174e2e7246f1e37e9dcbbf5afa7a122b5d64a69b72.jpg)

![ac9e700e788910977eb9e6f7ed56fe596274c3a8b19da18c61e856bdf43564f8.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/tables/ac9e700e788910977eb9e6f7ed56fe596274c3a8b19da18c61e856bdf43564f8.jpg)

![b2c7307bc127938b30c00667418d4998d96560db4fe6a14fdc55fdbc7f08d0aa.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/tables/b2c7307bc127938b30c00667418d4998d96560db4fe6a14fdc55fdbc7f08d0aa.jpg)

![b52474bcd506b954c4cde8a033f7160487a4339e7e98846f5c99418380039eb8.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/tables/b52474bcd506b954c4cde8a033f7160487a4339e7e98846f5c99418380039eb8.jpg)

![ccde8a5abf5e976e0673a5b553022c33ed03dbfcc20ff1c93899b05d07c19cc8.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/tables/ccde8a5abf5e976e0673a5b553022c33ed03dbfcc20ff1c93899b05d07c19cc8.jpg)

![ff8622412967440336f4664fead734f664380d059bc541949a67bc2609311e92.jpg](../iclr_results/2727_What%20Makes%20a%20Maze%20Look%20Like%20a%20Maze_/tables/ff8622412967440336f4664fead734f664380d059bc541949a67bc2609311e92.jpg)

## A Simple yet Effective $\Delta\Delta G$ Predictor is An Unsupervised Antibody Optimizer and Explainer


### Images

![1f2170a7a49740b143398b965b2e91c21ef379b75b0d8e3c3540427a991907d0.jpg](../iclr_results/2728_A%20Simple%20yet%20Effective%20%24_Delta_Delta%20G%24%20Predictor%20is%20An%20Unsupervised%20Antibody%20Optimizer%20and%20Explaine/images/1f2170a7a49740b143398b965b2e91c21ef379b75b0d8e3c3540427a991907d0.jpg)

![68c11ce05820580ac1c3311e51f1236cef6cfc8112e1f1fa20dc0767aec78606.jpg](../iclr_results/2728_A%20Simple%20yet%20Effective%20%24_Delta_Delta%20G%24%20Predictor%20is%20An%20Unsupervised%20Antibody%20Optimizer%20and%20Explaine/images/68c11ce05820580ac1c3311e51f1236cef6cfc8112e1f1fa20dc0767aec78606.jpg)

![71cc1a7de85877a3603512a1c29863f367f9af905e4b6c4546504fd6e3d14ff4.jpg](../iclr_results/2728_A%20Simple%20yet%20Effective%20%24_Delta_Delta%20G%24%20Predictor%20is%20An%20Unsupervised%20Antibody%20Optimizer%20and%20Explaine/images/71cc1a7de85877a3603512a1c29863f367f9af905e4b6c4546504fd6e3d14ff4.jpg)

![8fa6c56c0b1619edd18e7ec7a3b481c353c45ab41f993deb57026d9747d674c1.jpg](../iclr_results/2728_A%20Simple%20yet%20Effective%20%24_Delta_Delta%20G%24%20Predictor%20is%20An%20Unsupervised%20Antibody%20Optimizer%20and%20Explaine/images/8fa6c56c0b1619edd18e7ec7a3b481c353c45ab41f993deb57026d9747d674c1.jpg)

![9fdf5d70f65a5ea771e068f37659b56bf32663aa6437ad01263d5a8730595f85.jpg](../iclr_results/2728_A%20Simple%20yet%20Effective%20%24_Delta_Delta%20G%24%20Predictor%20is%20An%20Unsupervised%20Antibody%20Optimizer%20and%20Explaine/images/9fdf5d70f65a5ea771e068f37659b56bf32663aa6437ad01263d5a8730595f85.jpg)

![c05cc7b810330e6f4e2687ab140f4136e8919a18bd45b2ee4f7add2060e5cf84.jpg](../iclr_results/2728_A%20Simple%20yet%20Effective%20%24_Delta_Delta%20G%24%20Predictor%20is%20An%20Unsupervised%20Antibody%20Optimizer%20and%20Explaine/images/c05cc7b810330e6f4e2687ab140f4136e8919a18bd45b2ee4f7add2060e5cf84.jpg)

### Tables

![092d537e52ee62d5f6f37be2435fd970a1b947bb5d74305b04dc1f8194761fb3.jpg](../iclr_results/2728_A%20Simple%20yet%20Effective%20%24_Delta_Delta%20G%24%20Predictor%20is%20An%20Unsupervised%20Antibody%20Optimizer%20and%20Explaine/tables/092d537e52ee62d5f6f37be2435fd970a1b947bb5d74305b04dc1f8194761fb3.jpg)

![2d6a8c7ad1f44c1688d49dba35e65f90f65b6ce794985415f508461af910e419.jpg](../iclr_results/2728_A%20Simple%20yet%20Effective%20%24_Delta_Delta%20G%24%20Predictor%20is%20An%20Unsupervised%20Antibody%20Optimizer%20and%20Explaine/tables/2d6a8c7ad1f44c1688d49dba35e65f90f65b6ce794985415f508461af910e419.jpg)

![57092398d91e8b614f009c3089756135432a16ee202df15a8115d8404458dd09.jpg](../iclr_results/2728_A%20Simple%20yet%20Effective%20%24_Delta_Delta%20G%24%20Predictor%20is%20An%20Unsupervised%20Antibody%20Optimizer%20and%20Explaine/tables/57092398d91e8b614f009c3089756135432a16ee202df15a8115d8404458dd09.jpg)

![7621004e0a88a225f765701fc7428e8924737d98fe993c2e7d3e2b7caccdd0d8.jpg](../iclr_results/2728_A%20Simple%20yet%20Effective%20%24_Delta_Delta%20G%24%20Predictor%20is%20An%20Unsupervised%20Antibody%20Optimizer%20and%20Explaine/tables/7621004e0a88a225f765701fc7428e8924737d98fe993c2e7d3e2b7caccdd0d8.jpg)

![988f9bfeaae9c98479aadf2729caa960b2907b220246bd6900e39f7ad8f547b8.jpg](../iclr_results/2728_A%20Simple%20yet%20Effective%20%24_Delta_Delta%20G%24%20Predictor%20is%20An%20Unsupervised%20Antibody%20Optimizer%20and%20Explaine/tables/988f9bfeaae9c98479aadf2729caa960b2907b220246bd6900e39f7ad8f547b8.jpg)

![ab26d285f3504e7b36489b06a5ada9ac31cb8014c84369157b34323ecf64ee33.jpg](../iclr_results/2728_A%20Simple%20yet%20Effective%20%24_Delta_Delta%20G%24%20Predictor%20is%20An%20Unsupervised%20Antibody%20Optimizer%20and%20Explaine/tables/ab26d285f3504e7b36489b06a5ada9ac31cb8014c84369157b34323ecf64ee33.jpg)

![b93d043c6dfac4a8477a63b6ad95febb4f2e0f748f16a15de67ee4623269370a.jpg](../iclr_results/2728_A%20Simple%20yet%20Effective%20%24_Delta_Delta%20G%24%20Predictor%20is%20An%20Unsupervised%20Antibody%20Optimizer%20and%20Explaine/tables/b93d043c6dfac4a8477a63b6ad95febb4f2e0f748f16a15de67ee4623269370a.jpg)

## MedTrinity-25M: A Large-scale Multimodal Dataset with Multigranular Annotations for Medicine


### Images

![11e65274abc8c420712559f4b84de10941a19423dc62c0b41f4b6a4c222f72c9.jpg](../iclr_results/2729_MedTrinity-25M_%20A%20Large-scale%20Multimodal%20Dataset%20with%20Multigranular%20Annotations%20for%20Medicine/images/11e65274abc8c420712559f4b84de10941a19423dc62c0b41f4b6a4c222f72c9.jpg)

![1c5a58dd7c7ce5d4a7001a30df84165d22257731d39dc383af2a166e40713c5d.jpg](../iclr_results/2729_MedTrinity-25M_%20A%20Large-scale%20Multimodal%20Dataset%20with%20Multigranular%20Annotations%20for%20Medicine/images/1c5a58dd7c7ce5d4a7001a30df84165d22257731d39dc383af2a166e40713c5d.jpg)

![280a8cc3093e7e64211139e4ccb1b97f100fbe205ee5dc38c15e66fac6c033ef.jpg](../iclr_results/2729_MedTrinity-25M_%20A%20Large-scale%20Multimodal%20Dataset%20with%20Multigranular%20Annotations%20for%20Medicine/images/280a8cc3093e7e64211139e4ccb1b97f100fbe205ee5dc38c15e66fac6c033ef.jpg)

![28ceab02ee2a01d6eec994188f8ffdb39dfc7e85ffac99bb3dc85807f8dc93bf.jpg](../iclr_results/2729_MedTrinity-25M_%20A%20Large-scale%20Multimodal%20Dataset%20with%20Multigranular%20Annotations%20for%20Medicine/images/28ceab02ee2a01d6eec994188f8ffdb39dfc7e85ffac99bb3dc85807f8dc93bf.jpg)

![542d4a5b74c622ff3646df335be9fbbe309c515eb6d3693d0cfd48dde7dbf54e.jpg](../iclr_results/2729_MedTrinity-25M_%20A%20Large-scale%20Multimodal%20Dataset%20with%20Multigranular%20Annotations%20for%20Medicine/images/542d4a5b74c622ff3646df335be9fbbe309c515eb6d3693d0cfd48dde7dbf54e.jpg)

![5fa84ed24539607b3abf743051a7cfd8fd4d12b78ddccf70c5f4706d5d4baeb6.jpg](../iclr_results/2729_MedTrinity-25M_%20A%20Large-scale%20Multimodal%20Dataset%20with%20Multigranular%20Annotations%20for%20Medicine/images/5fa84ed24539607b3abf743051a7cfd8fd4d12b78ddccf70c5f4706d5d4baeb6.jpg)

![6f685e0fef6fa718f64187cad3aefb68f98f232b32ffadce258024ee70a9832d.jpg](../iclr_results/2729_MedTrinity-25M_%20A%20Large-scale%20Multimodal%20Dataset%20with%20Multigranular%20Annotations%20for%20Medicine/images/6f685e0fef6fa718f64187cad3aefb68f98f232b32ffadce258024ee70a9832d.jpg)

![8d1ce890826333de3246af0c498f300cd0838ab6171098bf7f09aaebf7726954.jpg](../iclr_results/2729_MedTrinity-25M_%20A%20Large-scale%20Multimodal%20Dataset%20with%20Multigranular%20Annotations%20for%20Medicine/images/8d1ce890826333de3246af0c498f300cd0838ab6171098bf7f09aaebf7726954.jpg)

![9cf2d4d524432dc0bae328ea2bfb4007bb898d75b1d04e118494b8033e006831.jpg](../iclr_results/2729_MedTrinity-25M_%20A%20Large-scale%20Multimodal%20Dataset%20with%20Multigranular%20Annotations%20for%20Medicine/images/9cf2d4d524432dc0bae328ea2bfb4007bb898d75b1d04e118494b8033e006831.jpg)

![b1ffb61348e9067f13fed5cd72dd9b121759acdad407cae128b9b7d09b8c5239.jpg](../iclr_results/2729_MedTrinity-25M_%20A%20Large-scale%20Multimodal%20Dataset%20with%20Multigranular%20Annotations%20for%20Medicine/images/b1ffb61348e9067f13fed5cd72dd9b121759acdad407cae128b9b7d09b8c5239.jpg)

![b7178c4bcde56647ca2910785d95f9b3068e79a495c1c230b6645760a21aee92.jpg](../iclr_results/2729_MedTrinity-25M_%20A%20Large-scale%20Multimodal%20Dataset%20with%20Multigranular%20Annotations%20for%20Medicine/images/b7178c4bcde56647ca2910785d95f9b3068e79a495c1c230b6645760a21aee92.jpg)

![becad348861a11992cd13a5dd6a218c93c390e54cd89d4a525b81bd55beb686a.jpg](../iclr_results/2729_MedTrinity-25M_%20A%20Large-scale%20Multimodal%20Dataset%20with%20Multigranular%20Annotations%20for%20Medicine/images/becad348861a11992cd13a5dd6a218c93c390e54cd89d4a525b81bd55beb686a.jpg)

![d0c944272c4598f2e5df531eab45584e453b7f2e7acf3dbd7b3ccb18bbc30810.jpg](../iclr_results/2729_MedTrinity-25M_%20A%20Large-scale%20Multimodal%20Dataset%20with%20Multigranular%20Annotations%20for%20Medicine/images/d0c944272c4598f2e5df531eab45584e453b7f2e7acf3dbd7b3ccb18bbc30810.jpg)

![d772e56cc8792bd6aec2447fa741cd74c06b463106e9f4ce15c5069bf3f75fac.jpg](../iclr_results/2729_MedTrinity-25M_%20A%20Large-scale%20Multimodal%20Dataset%20with%20Multigranular%20Annotations%20for%20Medicine/images/d772e56cc8792bd6aec2447fa741cd74c06b463106e9f4ce15c5069bf3f75fac.jpg)

![f05dfc4a8a9c8c7790e447d88abdaf800b481f39589ab07fb665c31baad252c4.jpg](../iclr_results/2729_MedTrinity-25M_%20A%20Large-scale%20Multimodal%20Dataset%20with%20Multigranular%20Annotations%20for%20Medicine/images/f05dfc4a8a9c8c7790e447d88abdaf800b481f39589ab07fb665c31baad252c4.jpg)

### Tables

![2f82f89654c33e942a0fd7b1c36ca92f2e1841a4f8558c50494e0fa18cfa478b.jpg](../iclr_results/2729_MedTrinity-25M_%20A%20Large-scale%20Multimodal%20Dataset%20with%20Multigranular%20Annotations%20for%20Medicine/tables/2f82f89654c33e942a0fd7b1c36ca92f2e1841a4f8558c50494e0fa18cfa478b.jpg)

![333b4581d0dc585e76635121b94c4dd7b12c92e8e0d11d76de466926da2a5524.jpg](../iclr_results/2729_MedTrinity-25M_%20A%20Large-scale%20Multimodal%20Dataset%20with%20Multigranular%20Annotations%20for%20Medicine/tables/333b4581d0dc585e76635121b94c4dd7b12c92e8e0d11d76de466926da2a5524.jpg)

![4e8cba4348d27a80da6c29df7c4219ac3b8acc62353402c8e672ab57385d48dc.jpg](../iclr_results/2729_MedTrinity-25M_%20A%20Large-scale%20Multimodal%20Dataset%20with%20Multigranular%20Annotations%20for%20Medicine/tables/4e8cba4348d27a80da6c29df7c4219ac3b8acc62353402c8e672ab57385d48dc.jpg)

![522fe63577bac00daab8e414117d3bc48f1ad171f6f975bf6f763ad777d81f60.jpg](../iclr_results/2729_MedTrinity-25M_%20A%20Large-scale%20Multimodal%20Dataset%20with%20Multigranular%20Annotations%20for%20Medicine/tables/522fe63577bac00daab8e414117d3bc48f1ad171f6f975bf6f763ad777d81f60.jpg)

![678883de507b2171d463ec371cfb7f394b629b296dd42737835aa2e99dd864bb.jpg](../iclr_results/2729_MedTrinity-25M_%20A%20Large-scale%20Multimodal%20Dataset%20with%20Multigranular%20Annotations%20for%20Medicine/tables/678883de507b2171d463ec371cfb7f394b629b296dd42737835aa2e99dd864bb.jpg)

![83221dc02a9209755b10adcbe0a41995591ef75dcb4c6d9e5331c59fd1b0c6c3.jpg](../iclr_results/2729_MedTrinity-25M_%20A%20Large-scale%20Multimodal%20Dataset%20with%20Multigranular%20Annotations%20for%20Medicine/tables/83221dc02a9209755b10adcbe0a41995591ef75dcb4c6d9e5331c59fd1b0c6c3.jpg)

![97ecd7d4ecebc0deba79ed584ccb2b496644b28c5b3e49a277bb89f28dd6bde4.jpg](../iclr_results/2729_MedTrinity-25M_%20A%20Large-scale%20Multimodal%20Dataset%20with%20Multigranular%20Annotations%20for%20Medicine/tables/97ecd7d4ecebc0deba79ed584ccb2b496644b28c5b3e49a277bb89f28dd6bde4.jpg)

![a94509f7df7afab1942b199a12e993b1669e599f4f680f41cce2a515398ed3e7.jpg](../iclr_results/2729_MedTrinity-25M_%20A%20Large-scale%20Multimodal%20Dataset%20with%20Multigranular%20Annotations%20for%20Medicine/tables/a94509f7df7afab1942b199a12e993b1669e599f4f680f41cce2a515398ed3e7.jpg)

## Post-hoc Reward Calibration: A Case Study on Length Bias


### Images

![0379b888589b8b96b69557b5dfb9ca05c35175fa1fc3c08269a043ab5175c4fa.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/0379b888589b8b96b69557b5dfb9ca05c35175fa1fc3c08269a043ab5175c4fa.jpg)

![0e94c9d0b32b4ae78e146aafca924f1d1773c10a1296af8783f4f966870ec7f6.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/0e94c9d0b32b4ae78e146aafca924f1d1773c10a1296af8783f4f966870ec7f6.jpg)

![10a07d838d8d526f4bed03948a62ecac939f9ee408410074dbfcda35b43c29de.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/10a07d838d8d526f4bed03948a62ecac939f9ee408410074dbfcda35b43c29de.jpg)

![1d6c08d83b0323d460b50027e34fdb6d60d4205cb56915b5185130150356a196.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/1d6c08d83b0323d460b50027e34fdb6d60d4205cb56915b5185130150356a196.jpg)

![1fbcbcadcfce3af64d4480d8e6d4e5a4bd4cb49cc6c0bfc6c8b547fd06130c67.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/1fbcbcadcfce3af64d4480d8e6d4e5a4bd4cb49cc6c0bfc6c8b547fd06130c67.jpg)

![277138ff2219cf627edcd9b511fa57199f4afa9fdd97f0b7f14beb15691079d5.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/277138ff2219cf627edcd9b511fa57199f4afa9fdd97f0b7f14beb15691079d5.jpg)

![50e32ec2a7e2bea4d6ff3d953a2311355a83a35b558feda7b687d55f85ba3035.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/50e32ec2a7e2bea4d6ff3d953a2311355a83a35b558feda7b687d55f85ba3035.jpg)

![5102ce0157d0bddf2ea56a12ff66f1207c3e38aa5952162a0e7e0fd8cacfdaf7.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/5102ce0157d0bddf2ea56a12ff66f1207c3e38aa5952162a0e7e0fd8cacfdaf7.jpg)

![6c5fea09361b2c21ee12eec45c2d3ea792a521b29a512700e956c47daf14b6c8.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/6c5fea09361b2c21ee12eec45c2d3ea792a521b29a512700e956c47daf14b6c8.jpg)

![6dee60aed3ee5f02d73d548e5ac586862806376d3d2f80c50c6db17d240e0ecb.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/6dee60aed3ee5f02d73d548e5ac586862806376d3d2f80c50c6db17d240e0ecb.jpg)

![7597c81537952d7bbe605eb299af0322188a56a1296f345cd49292878baaa2f2.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/7597c81537952d7bbe605eb299af0322188a56a1296f345cd49292878baaa2f2.jpg)

![836d56a3509dee0680cff676580cef3905cfdfea87a16de272b6142cb297d3bb.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/836d56a3509dee0680cff676580cef3905cfdfea87a16de272b6142cb297d3bb.jpg)

![842e90d08caa505370a3f366c1b74a907b9ec88ce1e0a854582103031404f33a.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/842e90d08caa505370a3f366c1b74a907b9ec88ce1e0a854582103031404f33a.jpg)

![849827329c4080732f08d0ab1df9d0d2728f59d4de111cc23ce3d38b818cb0d1.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/849827329c4080732f08d0ab1df9d0d2728f59d4de111cc23ce3d38b818cb0d1.jpg)

![8a753b5c7153f7d300c4c0e8f40ad1188cef3a52e02e95a7fff1c1bb24d4b7e1.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/8a753b5c7153f7d300c4c0e8f40ad1188cef3a52e02e95a7fff1c1bb24d4b7e1.jpg)

![95f7aa8fcaff1d2372757437bd7315453200e558ec2632edf34fcd3ba0fce757.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/95f7aa8fcaff1d2372757437bd7315453200e558ec2632edf34fcd3ba0fce757.jpg)

![9c1cf5ec5446f4a2c49e2593731595b54c38cfbf6f7dc79ad7d274410c9dae61.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/9c1cf5ec5446f4a2c49e2593731595b54c38cfbf6f7dc79ad7d274410c9dae61.jpg)

![9ea9d7e4cdd86e112537a52c8b74152df6b5b454d04816a51c1038937dd37225.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/9ea9d7e4cdd86e112537a52c8b74152df6b5b454d04816a51c1038937dd37225.jpg)

![a8ef9d4227df9da5fb5a0ce3d4506b7227174d83248e4ba7260837b1dfe18f9e.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/a8ef9d4227df9da5fb5a0ce3d4506b7227174d83248e4ba7260837b1dfe18f9e.jpg)

![ae34f96a12ec144df6cb08fa6b608171e0895ad519c0e1659ab27b96fcfaed4c.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/ae34f96a12ec144df6cb08fa6b608171e0895ad519c0e1659ab27b96fcfaed4c.jpg)

![b3e9957232c1999b34b903c71acc67c38ce8d8c8a9f18e1fafb19eaf0d2e3f5b.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/b3e9957232c1999b34b903c71acc67c38ce8d8c8a9f18e1fafb19eaf0d2e3f5b.jpg)

![c12c92beae9549e43ba930ea1f9643c540e8c9509275097e1c1e65f00cc98227.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/c12c92beae9549e43ba930ea1f9643c540e8c9509275097e1c1e65f00cc98227.jpg)

![d62d6da8cbce6a627666b1efd96be0c5a0b8e06892b1f3169106710bce8fc863.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/d62d6da8cbce6a627666b1efd96be0c5a0b8e06892b1f3169106710bce8fc863.jpg)

![dcb868ebc666aa02a0eb0f56b13e2c2f474057d8f473f0ba5d87d799e785ad88.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/dcb868ebc666aa02a0eb0f56b13e2c2f474057d8f473f0ba5d87d799e785ad88.jpg)

![ed28ae8877019594b56cdc8820d53abb3fc9d0287d0ef6a0bd29648649fb0c82.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/ed28ae8877019594b56cdc8820d53abb3fc9d0287d0ef6a0bd29648649fb0c82.jpg)

![f4ee5e4630fd6c9820c1eafaeef702df426680f4cac34ee88cecce78ffb33e08.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/images/f4ee5e4630fd6c9820c1eafaeef702df426680f4cac34ee88cecce78ffb33e08.jpg)

### Tables

![58f2791e07f83ff4c01880ad8b37250b6da2f0a3e79e400409609144d6e02a09.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/tables/58f2791e07f83ff4c01880ad8b37250b6da2f0a3e79e400409609144d6e02a09.jpg)

![842a95f4ea31af65eabd82eb9b3fd6524fd735d4330519f7761bfd925457a91a.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/tables/842a95f4ea31af65eabd82eb9b3fd6524fd735d4330519f7761bfd925457a91a.jpg)

![bb212ba52da0ad426601139f1329782ecb9ed7fa59c670ffa5e073b0fb43c457.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/tables/bb212ba52da0ad426601139f1329782ecb9ed7fa59c670ffa5e073b0fb43c457.jpg)

![d8f9116633574fcbfdacb1342e424a50708cfd470848011e4bc81cbd2c1b3399.jpg](../iclr_results/2730_Post-hoc%20Reward%20Calibration_%20A%20Case%20Study%20on%20Length%20Bias/tables/d8f9116633574fcbfdacb1342e424a50708cfd470848011e4bc81cbd2c1b3399.jpg)

## PseDet: Revisiting the Power of Pseudo Label in Incremental Object Detection


### Images

![030ce368931ba8da3e209ad98f6befe44f4c07e0248e8882f99b0d75c2958f3f.jpg](../iclr_results/2731_PseDet_%20Revisiting%20the%20Power%20of%20Pseudo%20Label%20in%20Incremental%20Object%20Detection/images/030ce368931ba8da3e209ad98f6befe44f4c07e0248e8882f99b0d75c2958f3f.jpg)

![08dc0473bd1c9aefcefc923e957bc2880156fd7b88771f775edb604fb84eb06a.jpg](../iclr_results/2731_PseDet_%20Revisiting%20the%20Power%20of%20Pseudo%20Label%20in%20Incremental%20Object%20Detection/images/08dc0473bd1c9aefcefc923e957bc2880156fd7b88771f775edb604fb84eb06a.jpg)

![2c680447c4cec9f6a0fc07f9f27b67d6b63b6de31ce4e8388326dc44fbe6d914.jpg](../iclr_results/2731_PseDet_%20Revisiting%20the%20Power%20of%20Pseudo%20Label%20in%20Incremental%20Object%20Detection/images/2c680447c4cec9f6a0fc07f9f27b67d6b63b6de31ce4e8388326dc44fbe6d914.jpg)

![4ee65bd98c77100db9e77f5b8f24b767e61788e87b86fd09a289207a87750929.jpg](../iclr_results/2731_PseDet_%20Revisiting%20the%20Power%20of%20Pseudo%20Label%20in%20Incremental%20Object%20Detection/images/4ee65bd98c77100db9e77f5b8f24b767e61788e87b86fd09a289207a87750929.jpg)

![577b6ef96e2aa422dc8f900e34edf7e77cea2f3eeec767df3f36543fb5685327.jpg](../iclr_results/2731_PseDet_%20Revisiting%20the%20Power%20of%20Pseudo%20Label%20in%20Incremental%20Object%20Detection/images/577b6ef96e2aa422dc8f900e34edf7e77cea2f3eeec767df3f36543fb5685327.jpg)

![71028b32ba1d891501901542b535305b8029d7b09281be8b953e575a0e68cff4.jpg](../iclr_results/2731_PseDet_%20Revisiting%20the%20Power%20of%20Pseudo%20Label%20in%20Incremental%20Object%20Detection/images/71028b32ba1d891501901542b535305b8029d7b09281be8b953e575a0e68cff4.jpg)

![a2c2a056134c9cf1fb3b8f27150882ff7dc2c8b276acbe062871e8cfde9b40bb.jpg](../iclr_results/2731_PseDet_%20Revisiting%20the%20Power%20of%20Pseudo%20Label%20in%20Incremental%20Object%20Detection/images/a2c2a056134c9cf1fb3b8f27150882ff7dc2c8b276acbe062871e8cfde9b40bb.jpg)

### Tables

![0881d3e5cbb27df9efef748fe04b2f1493f10627a0684cd38574d5e07cbeb139.jpg](../iclr_results/2731_PseDet_%20Revisiting%20the%20Power%20of%20Pseudo%20Label%20in%20Incremental%20Object%20Detection/tables/0881d3e5cbb27df9efef748fe04b2f1493f10627a0684cd38574d5e07cbeb139.jpg)

![0f667827b5d09fe81d514cc5606fc586cae3d30907dcf75dfb1a847780493736.jpg](../iclr_results/2731_PseDet_%20Revisiting%20the%20Power%20of%20Pseudo%20Label%20in%20Incremental%20Object%20Detection/tables/0f667827b5d09fe81d514cc5606fc586cae3d30907dcf75dfb1a847780493736.jpg)

![1ee747fd5bbc6ae8586382460535d4a906a49b78373eb9df25e807047fd95f5a.jpg](../iclr_results/2731_PseDet_%20Revisiting%20the%20Power%20of%20Pseudo%20Label%20in%20Incremental%20Object%20Detection/tables/1ee747fd5bbc6ae8586382460535d4a906a49b78373eb9df25e807047fd95f5a.jpg)

![577329f1fb48ff1145a8520c21c3110e7e7b52a14f0d5c454de4dddeb4b75265.jpg](../iclr_results/2731_PseDet_%20Revisiting%20the%20Power%20of%20Pseudo%20Label%20in%20Incremental%20Object%20Detection/tables/577329f1fb48ff1145a8520c21c3110e7e7b52a14f0d5c454de4dddeb4b75265.jpg)

![63380736d450ad4587e804fa7c134a316429413a0ac4999e3cfbb7749133c60a.jpg](../iclr_results/2731_PseDet_%20Revisiting%20the%20Power%20of%20Pseudo%20Label%20in%20Incremental%20Object%20Detection/tables/63380736d450ad4587e804fa7c134a316429413a0ac4999e3cfbb7749133c60a.jpg)

![7c3662dbf9cbb09f6e7b4030a722556993cce6cb636b692c602e5ebffe75e94f.jpg](../iclr_results/2731_PseDet_%20Revisiting%20the%20Power%20of%20Pseudo%20Label%20in%20Incremental%20Object%20Detection/tables/7c3662dbf9cbb09f6e7b4030a722556993cce6cb636b692c602e5ebffe75e94f.jpg)

![86194f042c215fab204d7e5051d71dc76734311b1d417055c8d342601666ed43.jpg](../iclr_results/2731_PseDet_%20Revisiting%20the%20Power%20of%20Pseudo%20Label%20in%20Incremental%20Object%20Detection/tables/86194f042c215fab204d7e5051d71dc76734311b1d417055c8d342601666ed43.jpg)

![86e68d66cf45d2671bd39ede155860bdf3328f8073ff45994acc68d442df60ca.jpg](../iclr_results/2731_PseDet_%20Revisiting%20the%20Power%20of%20Pseudo%20Label%20in%20Incremental%20Object%20Detection/tables/86e68d66cf45d2671bd39ede155860bdf3328f8073ff45994acc68d442df60ca.jpg)

![f047e6c74a20059b8d27ea80aa43b6b7223757d8c8c0b89d4c8211966f527480.jpg](../iclr_results/2731_PseDet_%20Revisiting%20the%20Power%20of%20Pseudo%20Label%20in%20Incremental%20Object%20Detection/tables/f047e6c74a20059b8d27ea80aa43b6b7223757d8c8c0b89d4c8211966f527480.jpg)

## Improving Reasoning Performance in Large Language Models via Representation Engineering


### Images

![21580ad3e9c0f7da79fd0d7d90cf4fab9f39cabe938c7f4e7c95735207875aa8.jpg](../iclr_results/2732_Improving%20Reasoning%20Performance%20in%20Large%20Language%20Models%20via%20Representation%20Engineering/images/21580ad3e9c0f7da79fd0d7d90cf4fab9f39cabe938c7f4e7c95735207875aa8.jpg)

![22d4a095e613cda60ddfe1dd42bfd17361cb9a9d3b16298f7e12e4f8a9583673.jpg](../iclr_results/2732_Improving%20Reasoning%20Performance%20in%20Large%20Language%20Models%20via%20Representation%20Engineering/images/22d4a095e613cda60ddfe1dd42bfd17361cb9a9d3b16298f7e12e4f8a9583673.jpg)

![510be5f37b95e8a91e663da97d4e8bc2665c05fc3514612fd6666732ae8aba60.jpg](../iclr_results/2732_Improving%20Reasoning%20Performance%20in%20Large%20Language%20Models%20via%20Representation%20Engineering/images/510be5f37b95e8a91e663da97d4e8bc2665c05fc3514612fd6666732ae8aba60.jpg)

![61f1de13aac38df54472edb0c37a320676738f76934bb620314d6eecb8e2ba2f.jpg](../iclr_results/2732_Improving%20Reasoning%20Performance%20in%20Large%20Language%20Models%20via%20Representation%20Engineering/images/61f1de13aac38df54472edb0c37a320676738f76934bb620314d6eecb8e2ba2f.jpg)

![650a6d61bd347eb727727aa3293de7eaf410748486f0d0cacdab60bc3d548f1b.jpg](../iclr_results/2732_Improving%20Reasoning%20Performance%20in%20Large%20Language%20Models%20via%20Representation%20Engineering/images/650a6d61bd347eb727727aa3293de7eaf410748486f0d0cacdab60bc3d548f1b.jpg)

![846f1621fdc2bb86eabdf773550f5b6b07b7f94bf49f79734b9f29fc43bd2b58.jpg](../iclr_results/2732_Improving%20Reasoning%20Performance%20in%20Large%20Language%20Models%20via%20Representation%20Engineering/images/846f1621fdc2bb86eabdf773550f5b6b07b7f94bf49f79734b9f29fc43bd2b58.jpg)

![913685b025200929fa16ec51accf82571228c7c99ad7f5f961a4b1e85c80a58a.jpg](../iclr_results/2732_Improving%20Reasoning%20Performance%20in%20Large%20Language%20Models%20via%20Representation%20Engineering/images/913685b025200929fa16ec51accf82571228c7c99ad7f5f961a4b1e85c80a58a.jpg)

![a2b377f20d95a876290f8c3206f6aafabbb29f7cd478c557ca3358447d0e9937.jpg](../iclr_results/2732_Improving%20Reasoning%20Performance%20in%20Large%20Language%20Models%20via%20Representation%20Engineering/images/a2b377f20d95a876290f8c3206f6aafabbb29f7cd478c557ca3358447d0e9937.jpg)

![bcadab85ea3ff84676bae07d710b5c2cca7956e28917fc58489602269e5f7330.jpg](../iclr_results/2732_Improving%20Reasoning%20Performance%20in%20Large%20Language%20Models%20via%20Representation%20Engineering/images/bcadab85ea3ff84676bae07d710b5c2cca7956e28917fc58489602269e5f7330.jpg)

![d1023a4ec113d4ea018e3a6c39a52bff8a1951ff6904330bcdd8bf4327cff766.jpg](../iclr_results/2732_Improving%20Reasoning%20Performance%20in%20Large%20Language%20Models%20via%20Representation%20Engineering/images/d1023a4ec113d4ea018e3a6c39a52bff8a1951ff6904330bcdd8bf4327cff766.jpg)

![f751366aedf66ed8a6f7fdeef8899f5aa447f2592cc43f5e930163fb4ef1c2b4.jpg](../iclr_results/2732_Improving%20Reasoning%20Performance%20in%20Large%20Language%20Models%20via%20Representation%20Engineering/images/f751366aedf66ed8a6f7fdeef8899f5aa447f2592cc43f5e930163fb4ef1c2b4.jpg)

## Conditional Testing based on Localized Conformal $p$-values


### Images

![0618d606487ffa87f7655c09c87de8be44b3f40bca8c7f9b8ac984dbb1445a85.jpg](../iclr_results/2733_Conditional%20Testing%20based%20on%20Localized%20Conformal%20%24p%24-values/images/0618d606487ffa87f7655c09c87de8be44b3f40bca8c7f9b8ac984dbb1445a85.jpg)

![16f59a647fa3d9daa6de185ed429e3e869a05b4cdf4cb7934bfd4fbf1d3fc3cc.jpg](../iclr_results/2733_Conditional%20Testing%20based%20on%20Localized%20Conformal%20%24p%24-values/images/16f59a647fa3d9daa6de185ed429e3e869a05b4cdf4cb7934bfd4fbf1d3fc3cc.jpg)

![296ccd81f013672463a09bb61ce7ad92a15c2321a7e906720282039ca403ad48.jpg](../iclr_results/2733_Conditional%20Testing%20based%20on%20Localized%20Conformal%20%24p%24-values/images/296ccd81f013672463a09bb61ce7ad92a15c2321a7e906720282039ca403ad48.jpg)

![3cc05e9ef5dc7b5219cde5ce5f9440e1e69c30ed07b916f4369c6ee4837f2a56.jpg](../iclr_results/2733_Conditional%20Testing%20based%20on%20Localized%20Conformal%20%24p%24-values/images/3cc05e9ef5dc7b5219cde5ce5f9440e1e69c30ed07b916f4369c6ee4837f2a56.jpg)

![98c9ca8c9f3c859636376de443a2660b2076ef8e3f7bfe837bc5336943526f0d.jpg](../iclr_results/2733_Conditional%20Testing%20based%20on%20Localized%20Conformal%20%24p%24-values/images/98c9ca8c9f3c859636376de443a2660b2076ef8e3f7bfe837bc5336943526f0d.jpg)

![b28957001d35d03449e489d5c5afb6c78c61d6a1add838cac359aa1cfde88d3c.jpg](../iclr_results/2733_Conditional%20Testing%20based%20on%20Localized%20Conformal%20%24p%24-values/images/b28957001d35d03449e489d5c5afb6c78c61d6a1add838cac359aa1cfde88d3c.jpg)

![e59758cebc9e8aa13157557f66a415a8d9329d6c93adf81a1bbbb447367c2f80.jpg](../iclr_results/2733_Conditional%20Testing%20based%20on%20Localized%20Conformal%20%24p%24-values/images/e59758cebc9e8aa13157557f66a415a8d9329d6c93adf81a1bbbb447367c2f80.jpg)

### Tables

![23cfd586ced16dbb7cafd4b18e66720b0de4555ca6efa43f614b7e875184d83d.jpg](../iclr_results/2733_Conditional%20Testing%20based%20on%20Localized%20Conformal%20%24p%24-values/tables/23cfd586ced16dbb7cafd4b18e66720b0de4555ca6efa43f614b7e875184d83d.jpg)

![71e59d2b8989bce4768192c38812cf0de3d835c2d31331ba5783d7d02ffe0640.jpg](../iclr_results/2733_Conditional%20Testing%20based%20on%20Localized%20Conformal%20%24p%24-values/tables/71e59d2b8989bce4768192c38812cf0de3d835c2d31331ba5783d7d02ffe0640.jpg)

![8eb24b8bd7c997d01995d297db34b16357e03d142d13780897d05023c2282bc7.jpg](../iclr_results/2733_Conditional%20Testing%20based%20on%20Localized%20Conformal%20%24p%24-values/tables/8eb24b8bd7c997d01995d297db34b16357e03d142d13780897d05023c2282bc7.jpg)

![a73313e4bbe7cfb4ac8cdd4cb26bd2cf193412d29741e0fa77b9efbd9d4b82be.jpg](../iclr_results/2733_Conditional%20Testing%20based%20on%20Localized%20Conformal%20%24p%24-values/tables/a73313e4bbe7cfb4ac8cdd4cb26bd2cf193412d29741e0fa77b9efbd9d4b82be.jpg)

![e1c48e63e6a4c68635e45a65fca4551db08725c8bb5c90776b6dee13d7ebee6a.jpg](../iclr_results/2733_Conditional%20Testing%20based%20on%20Localized%20Conformal%20%24p%24-values/tables/e1c48e63e6a4c68635e45a65fca4551db08725c8bb5c90776b6dee13d7ebee6a.jpg)

## NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals


### Images

![166895a69f1ae641cfd709539cf9f6415a17aa031fd845430d70cc2ae0164bfe.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/images/166895a69f1ae641cfd709539cf9f6415a17aa031fd845430d70cc2ae0164bfe.jpg)

![25c4bd8f24745cb8b8dff5b1871e901e007d06b83bc7b11383b45da9e1ae43f6.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/images/25c4bd8f24745cb8b8dff5b1871e901e007d06b83bc7b11383b45da9e1ae43f6.jpg)

![273c160ecacf5d8cdda8ef350e51be4582877bfe4a0f42961f8a840f3b486744.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/images/273c160ecacf5d8cdda8ef350e51be4582877bfe4a0f42961f8a840f3b486744.jpg)

![4b16a6f91af77fa9b98eb112e18117ddbc4eb2bacf37bff113f35b1a58119a8c.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/images/4b16a6f91af77fa9b98eb112e18117ddbc4eb2bacf37bff113f35b1a58119a8c.jpg)

![58c2149058391b15bedd55e9c57f72621ec3c3b9912eb63ef83dfe007fa7d0e4.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/images/58c2149058391b15bedd55e9c57f72621ec3c3b9912eb63ef83dfe007fa7d0e4.jpg)

![5e5549927d378bb4ce9a056cbefffb7aebb6e35c774f2f342b917e46599c2496.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/images/5e5549927d378bb4ce9a056cbefffb7aebb6e35c774f2f342b917e46599c2496.jpg)

![81e4c134cf6c55ab36eaee9785ed310967c98b4389824e1c8e3307b4247c4b44.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/images/81e4c134cf6c55ab36eaee9785ed310967c98b4389824e1c8e3307b4247c4b44.jpg)

![9ee141a7150a307fc9458583466b366170c6e9c71f60800af775a2788d4d8d89.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/images/9ee141a7150a307fc9458583466b366170c6e9c71f60800af775a2788d4d8d89.jpg)

![cbbdaab27e781eeae75ba16a8ba6f4334af0c8fa4c4a5c5d2d874c83f0eb9a36.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/images/cbbdaab27e781eeae75ba16a8ba6f4334af0c8fa4c4a5c5d2d874c83f0eb9a36.jpg)

![deae90d170d65641db5a851b95efee111f26f1bdaa6f5746e7992db21a7b5682.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/images/deae90d170d65641db5a851b95efee111f26f1bdaa6f5746e7992db21a7b5682.jpg)

![e9086649a8b89aaf0512c911c1a4ea6aa53df3824b961abd66a150f082f19941.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/images/e9086649a8b89aaf0512c911c1a4ea6aa53df3824b961abd66a150f082f19941.jpg)

![ffe58a125c83999debb6f3e58e86b3646c54f704fc380059e714edff281a62c5.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/images/ffe58a125c83999debb6f3e58e86b3646c54f704fc380059e714edff281a62c5.jpg)

### Tables

![2ae25f73d268b51749a4938489ead284093181817d5bec0f25d5ee7bdc2b6b05.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/tables/2ae25f73d268b51749a4938489ead284093181817d5bec0f25d5ee7bdc2b6b05.jpg)

![5ec6959cfc5a0845eb93473dcb4f3cf02f7f6f2cb96a7de5ab9659a889993a48.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/tables/5ec6959cfc5a0845eb93473dcb4f3cf02f7f6f2cb96a7de5ab9659a889993a48.jpg)

![af032a3a7794849e2701fc38d76c7d2b5a0f98c1f6b5d2389a848edfccdffff3.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/tables/af032a3a7794849e2701fc38d76c7d2b5a0f98c1f6b5d2389a848edfccdffff3.jpg)

![bcecf39171e096ea01751fa7a3957239a244b0f275fbfe378075787ff08ed7bb.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/tables/bcecf39171e096ea01751fa7a3957239a244b0f275fbfe378075787ff08ed7bb.jpg)

![d6bfd3ce8094be534b1dfca844ae4bb4f4f93cfb3cc56c9935b693f474ead892.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/tables/d6bfd3ce8094be534b1dfca844ae4bb4f4f93cfb3cc56c9935b693f474ead892.jpg)

![d88ebfd1863299bd1ab99ac98975b4929d43fe061bd6485d7feaacfb563bbcca.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/tables/d88ebfd1863299bd1ab99ac98975b4929d43fe061bd6485d7feaacfb563bbcca.jpg)

![e15abc96f1d5239ced5d83b542dcdd1b37353dcf9ba7ce6b8493349ae941a2ab.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/tables/e15abc96f1d5239ced5d83b542dcdd1b37353dcf9ba7ce6b8493349ae941a2ab.jpg)

![e6d67e2bfc9b73e2cbaa190cb674e45a8ca7ef6667fb9eac5136721c246117d1.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/tables/e6d67e2bfc9b73e2cbaa190cb674e45a8ca7ef6667fb9eac5136721c246117d1.jpg)

![e8b44306deb3f0b814891aa634b7ae6f7470c758209e3aca1884a94789a4df8c.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/tables/e8b44306deb3f0b814891aa634b7ae6f7470c758209e3aca1884a94789a4df8c.jpg)

![f50efca13a197ea5352d3a4c33b1555391c19e67fb14b1812b7469a044d0ee56.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/tables/f50efca13a197ea5352d3a4c33b1555391c19e67fb14b1812b7469a044d0ee56.jpg)

![fa34dff6bf56e7ebefd71e2a7626e33266f6efe0658bdc20d822004a58da0a78.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/tables/fa34dff6bf56e7ebefd71e2a7626e33266f6efe0658bdc20d822004a58da0a78.jpg)

![fd4620b9018b73cd80b9849e880d257eedcd23058c0b4aec50e062490c283554.jpg](../iclr_results/2734_NeuroLM_%20A%20Universal%20Multi-task%20Foundation%20Model%20for%20Bridging%20the%20Gap%20between%20Language%20and%20EEG%20Signa/tables/fd4620b9018b73cd80b9849e880d257eedcd23058c0b4aec50e062490c283554.jpg)

## IV-mixed Sampler: Leveraging Image Diffusion Models for Enhanced Video Synthesis


### Images

![00f02fc9326a90ebc22208712e44cdae847f689ceb42b14e227dddf1d50eb640.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/images/00f02fc9326a90ebc22208712e44cdae847f689ceb42b14e227dddf1d50eb640.jpg)

![0a8ba0296cc1331ec4c9fb933f5652a59047dd90194b8df7053453049cd88b4f.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/images/0a8ba0296cc1331ec4c9fb933f5652a59047dd90194b8df7053453049cd88b4f.jpg)

![12133ee3a627b97bc25720658ba15c2f3bf492a45f889c8db6327682754149c9.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/images/12133ee3a627b97bc25720658ba15c2f3bf492a45f889c8db6327682754149c9.jpg)

![1682ff119258a06ed586cbf3e95c8fe8eec303d0b9ad7f6a05b442dd72462889.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/images/1682ff119258a06ed586cbf3e95c8fe8eec303d0b9ad7f6a05b442dd72462889.jpg)

![2a2b9db351cb434612a31e533623eaeefb95ddc44deeecab7ba416e24cffd904.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/images/2a2b9db351cb434612a31e533623eaeefb95ddc44deeecab7ba416e24cffd904.jpg)

![399d41e566423a3e6ea476aabe980ad65e7cf2001497ff79576e203e18583b3d.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/images/399d41e566423a3e6ea476aabe980ad65e7cf2001497ff79576e203e18583b3d.jpg)

![468a0129c93f2c7a93efc0bdab1048e7a21de8fa691de3b7478ea8bb5b1a5a09.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/images/468a0129c93f2c7a93efc0bdab1048e7a21de8fa691de3b7478ea8bb5b1a5a09.jpg)

![75836b5b12288a5c39292400b9a50931c988c746536531c3a4142b40263f3979.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/images/75836b5b12288a5c39292400b9a50931c988c746536531c3a4142b40263f3979.jpg)

![7754f9c45a3d53971503e226440b530c7e890f3206406979d74f74229e9869a5.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/images/7754f9c45a3d53971503e226440b530c7e890f3206406979d74f74229e9869a5.jpg)

![77875470c8cb56f185de85a17db29adff5845c2f819dae065a8bb9a70473b30b.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/images/77875470c8cb56f185de85a17db29adff5845c2f819dae065a8bb9a70473b30b.jpg)

![7adccb417e2f63643b486cd5d107406f8d476db8b6a9c4f2325d2d8a4353a85f.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/images/7adccb417e2f63643b486cd5d107406f8d476db8b6a9c4f2325d2d8a4353a85f.jpg)

![d9d0a7ffdb13ec19cdb743621ef8fbe421963cbe8b138b8922f3b8cf5c55edce.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/images/d9d0a7ffdb13ec19cdb743621ef8fbe421963cbe8b138b8922f3b8cf5c55edce.jpg)

![dc0335eeacd9d6e91e22ac5f49d3ae0e08ccf4c787429e7996f0fbcb898891e5.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/images/dc0335eeacd9d6e91e22ac5f49d3ae0e08ccf4c787429e7996f0fbcb898891e5.jpg)

![dd78a5ce74a6e54311dac2159958ab7d8fadafdb52a3dbc73343e2d09d6b37cd.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/images/dd78a5ce74a6e54311dac2159958ab7d8fadafdb52a3dbc73343e2d09d6b37cd.jpg)

![dfdf3b346a86a3c9c3cbe0c297b835707ec3d9c3c2aa560a4b7bf9e3614b3b64.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/images/dfdf3b346a86a3c9c3cbe0c297b835707ec3d9c3c2aa560a4b7bf9e3614b3b64.jpg)

![ea10a71d25a7c8eb80c4651b9fdb09ebd8b7a159af691e8e268aa6cb4c9e0af6.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/images/ea10a71d25a7c8eb80c4651b9fdb09ebd8b7a159af691e8e268aa6cb4c9e0af6.jpg)

![ecff29ca17d4f294fe4f57ccb37fead68a6346048ea557f7089a77fcc7e53a29.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/images/ecff29ca17d4f294fe4f57ccb37fead68a6346048ea557f7089a77fcc7e53a29.jpg)

![ef8262125726d5126a760745e505e1e194d7fe9650e3d4b75e27ddc49814e894.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/images/ef8262125726d5126a760745e505e1e194d7fe9650e3d4b75e27ddc49814e894.jpg)

![f2a67c9a6e2993f703c653c1b1ff54b197f1bce5add7389f215651326cf7a07b.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/images/f2a67c9a6e2993f703c653c1b1ff54b197f1bce5add7389f215651326cf7a07b.jpg)

### Tables

![30567b71ed38d3460c8dd2516b7f13ec4c293785657006ce84ce775b23f80670.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/tables/30567b71ed38d3460c8dd2516b7f13ec4c293785657006ce84ce775b23f80670.jpg)

![4a38a2a4b07b91c4882cd3630889f6c2610135547eb0977f55e55d9f1b1926d8.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/tables/4a38a2a4b07b91c4882cd3630889f6c2610135547eb0977f55e55d9f1b1926d8.jpg)

![4f74c536843822d0c640bc27bf60280600a1179c8d77d4f374127c7b99698933.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/tables/4f74c536843822d0c640bc27bf60280600a1179c8d77d4f374127c7b99698933.jpg)

![7538a2d14aa2b234edda5475486587377d01291596cc21b358c578a52921e946.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/tables/7538a2d14aa2b234edda5475486587377d01291596cc21b358c578a52921e946.jpg)

![7ab69148ae678c8a3de61a4b63cc3c8afa4035a61dc714b6326d920e6a080ae8.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/tables/7ab69148ae678c8a3de61a4b63cc3c8afa4035a61dc714b6326d920e6a080ae8.jpg)

![a387d4e7adb88eec8b654a21a6a44a815f1c0521d185a1309e480acbbee792e5.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/tables/a387d4e7adb88eec8b654a21a6a44a815f1c0521d185a1309e480acbbee792e5.jpg)

![c3cd9244f5f174e36d509de03bec68e4ecb02bc8cb23af4eb423c54411333ebe.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/tables/c3cd9244f5f174e36d509de03bec68e4ecb02bc8cb23af4eb423c54411333ebe.jpg)

![ed1a3992ac1c80c7e3b5201a0fd34606b12d0158a083e9b15979628da767901d.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/tables/ed1a3992ac1c80c7e3b5201a0fd34606b12d0158a083e9b15979628da767901d.jpg)

![fcfa3b8d4de39b99fe413c89b4bd9e5d9bec119c6624954e320a0352c51059a4.jpg](../iclr_results/2735_IV-mixed%20Sampler_%20Leveraging%20Image%20Diffusion%20Models%20for%20Enhanced%20Video%20Synthesis/tables/fcfa3b8d4de39b99fe413c89b4bd9e5d9bec119c6624954e320a0352c51059a4.jpg)

## Watermark Anything With Localized Messages


### Images

![3318d06bc5ae749c3f7ba590d39fdb785e34182beb00c91625af1359b29ec57b.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/images/3318d06bc5ae749c3f7ba590d39fdb785e34182beb00c91625af1359b29ec57b.jpg)

![4adc3250f17a48bc09f7f0e4d2f9279f424f6aa6a20b5c7dead474945d0e3c63.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/images/4adc3250f17a48bc09f7f0e4d2f9279f424f6aa6a20b5c7dead474945d0e3c63.jpg)

![4d46dd9ce58dd4233ea0b05d36c55d02f397d1154be923852cf3baf05967cbaa.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/images/4d46dd9ce58dd4233ea0b05d36c55d02f397d1154be923852cf3baf05967cbaa.jpg)

![5102c3da7a600b1906e9a5ca499f54e03b1c3a4afdb824d95b9f403c9928113a.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/images/5102c3da7a600b1906e9a5ca499f54e03b1c3a4afdb824d95b9f403c9928113a.jpg)

![542029cf69709cafe35ce9b74111abeadd5547a8333f98c5ab3ce4254f1a7444.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/images/542029cf69709cafe35ce9b74111abeadd5547a8333f98c5ab3ce4254f1a7444.jpg)

![58bff3decff20230f858dde7cb926032786df351a2e426f794073ab5299cb19f.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/images/58bff3decff20230f858dde7cb926032786df351a2e426f794073ab5299cb19f.jpg)

![5ac4e2737613593bc2dfd5aa1cd760e4abcf1fc2c5fc118f8660dce4ef756c5d.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/images/5ac4e2737613593bc2dfd5aa1cd760e4abcf1fc2c5fc118f8660dce4ef756c5d.jpg)

![6552660c5733e04f4f6b12b2b144f049322f89a23f90b5b375ad4bc8b61a9204.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/images/6552660c5733e04f4f6b12b2b144f049322f89a23f90b5b375ad4bc8b61a9204.jpg)

![65d4d7a43faf517987419feea5fdaf1b3915d0488556475c2194714c553c4f0e.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/images/65d4d7a43faf517987419feea5fdaf1b3915d0488556475c2194714c553c4f0e.jpg)

![69077952fa916bece69fc4e7063e06479765957d34a8acd1638cf964b5d39781.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/images/69077952fa916bece69fc4e7063e06479765957d34a8acd1638cf964b5d39781.jpg)

![7972da4d2ab1fdf3076681ffb691235cc2ac8ff3ce8f504121816aba935df83e.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/images/7972da4d2ab1fdf3076681ffb691235cc2ac8ff3ce8f504121816aba935df83e.jpg)

![a1c8d3057be77370d2a1d15e5d4b3308341eeaecd980cd6e75d7ede37e4ce7f5.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/images/a1c8d3057be77370d2a1d15e5d4b3308341eeaecd980cd6e75d7ede37e4ce7f5.jpg)

![d794620a008c7d503a02128a84135714e56b84cd68817be60ccb72fc8381729e.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/images/d794620a008c7d503a02128a84135714e56b84cd68817be60ccb72fc8381729e.jpg)

![ed02404e083a361e73108197b038eda0b54c526de707e4a1520af6128fb00af1.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/images/ed02404e083a361e73108197b038eda0b54c526de707e4a1520af6128fb00af1.jpg)

### Tables

![141b8a34cb8fc23e3794be0dff3b1b409d8863c56854c9e2156229689cdda216.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/tables/141b8a34cb8fc23e3794be0dff3b1b409d8863c56854c9e2156229689cdda216.jpg)

![1e1468dd2799aed2cbc08ffff52995f5c7f1ca92f76e10d8b26d6f8878804b02.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/tables/1e1468dd2799aed2cbc08ffff52995f5c7f1ca92f76e10d8b26d6f8878804b02.jpg)

![236f5fa9e50f1b018ae95c148d46b4c357eb526c9ac0c8f3773785fcdad40b1b.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/tables/236f5fa9e50f1b018ae95c148d46b4c357eb526c9ac0c8f3773785fcdad40b1b.jpg)

![30a33a7a9921e23a6d99460bb99067a00ea48dbd1cd7892ce0d61eb19263145d.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/tables/30a33a7a9921e23a6d99460bb99067a00ea48dbd1cd7892ce0d61eb19263145d.jpg)

![3255ea8686360e156f40475ac874783381af5b2293e34f09070415beedde122c.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/tables/3255ea8686360e156f40475ac874783381af5b2293e34f09070415beedde122c.jpg)

![5c5f259680ba9968508b7d9730bd88e73bae089bdaf983818b2691755c37aa23.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/tables/5c5f259680ba9968508b7d9730bd88e73bae089bdaf983818b2691755c37aa23.jpg)

![7589cd7a21d583b856b6ea2a8ffb921db309f6f1f3021eff9a31c51bd63b2ba1.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/tables/7589cd7a21d583b856b6ea2a8ffb921db309f6f1f3021eff9a31c51bd63b2ba1.jpg)

![8bfaa4e88caedd633573562f6bf824648ccc1d5b1b03e0210383071b307732e4.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/tables/8bfaa4e88caedd633573562f6bf824648ccc1d5b1b03e0210383071b307732e4.jpg)

![a058237b128331ce78031f802b788c9a40e03ae8905b14f176f8e0ebe48a422d.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/tables/a058237b128331ce78031f802b788c9a40e03ae8905b14f176f8e0ebe48a422d.jpg)

![a4667da928afb43dc2b7d4232cc6b3e7524dc1ba672ad84b83d83428deb68c15.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/tables/a4667da928afb43dc2b7d4232cc6b3e7524dc1ba672ad84b83d83428deb68c15.jpg)

![cb7f89c65505a987135fdb91172e40d24612931aec793cfce1369cf318718fb4.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/tables/cb7f89c65505a987135fdb91172e40d24612931aec793cfce1369cf318718fb4.jpg)

![fcafcf59bff64e62cc2ef93cfa58dcf5673348bbd1a0cbc6d25658ad48dfccc3.jpg](../iclr_results/2736_Watermark%20Anything%20With%20Localized%20Messages/tables/fcafcf59bff64e62cc2ef93cfa58dcf5673348bbd1a0cbc6d25658ad48dfccc3.jpg)

## From Attention to Activation: Unraveling the Enigmas of Large Language Models


### Images

![0ff069acd88aff50e1c9ba08c30ebd4ecd6a1e505144b77b541bf141e9d09dea.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/0ff069acd88aff50e1c9ba08c30ebd4ecd6a1e505144b77b541bf141e9d09dea.jpg)

![119c5a4ad8b54f4addb6e7851c292736bd99852bdc5394d82ce497b645750fc2.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/119c5a4ad8b54f4addb6e7851c292736bd99852bdc5394d82ce497b645750fc2.jpg)

![11dd15773c24c0c35e5704ae1061714f3a24b2040c4d9ab946a0c127b237ed4c.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/11dd15773c24c0c35e5704ae1061714f3a24b2040c4d9ab946a0c127b237ed4c.jpg)

![1dab7ab61c98d666f626ce43c84770226b67a6aef220a0d775333be38d7c7ca3.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/1dab7ab61c98d666f626ce43c84770226b67a6aef220a0d775333be38d7c7ca3.jpg)

![1ebac64d6f84fa9c1cf1732bd91cbd182e11bbe9afc0232d68304e5bab88c505.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/1ebac64d6f84fa9c1cf1732bd91cbd182e11bbe9afc0232d68304e5bab88c505.jpg)

![1fd0fccfbf3fcbf45edf39651768ad42807214ecc7bc208704a2f29f5c4433ac.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/1fd0fccfbf3fcbf45edf39651768ad42807214ecc7bc208704a2f29f5c4433ac.jpg)

![1fea8c2dec3681ee9414533e4aba056c2752a0dea7f4dad1c95ebebb8e3c30ff.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/1fea8c2dec3681ee9414533e4aba056c2752a0dea7f4dad1c95ebebb8e3c30ff.jpg)

![2e896dc62fcb67105581be6049cb0b45a6cc4c1dab1e9c29f9d4b2756c5bbab1.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/2e896dc62fcb67105581be6049cb0b45a6cc4c1dab1e9c29f9d4b2756c5bbab1.jpg)

![38f4027ad311606730046bfbe2ad66d1889fca32256b28bb697d75f05cbf0eb2.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/38f4027ad311606730046bfbe2ad66d1889fca32256b28bb697d75f05cbf0eb2.jpg)

![3952be87d5930fe5f122cd8a0feac2115517a031733a402d500f2a4856c7e2dc.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/3952be87d5930fe5f122cd8a0feac2115517a031733a402d500f2a4856c7e2dc.jpg)

![3f373a1665bbb9ffb303f771a9e3807edb26f14887a3f012c1cb6d990d535132.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/3f373a1665bbb9ffb303f771a9e3807edb26f14887a3f012c1cb6d990d535132.jpg)

![49a0c23d9f1549b253753a385ed83582a3283d960ec351330caae639ac25bc15.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/49a0c23d9f1549b253753a385ed83582a3283d960ec351330caae639ac25bc15.jpg)

![5c379af03487d59ad74ec3457030ba1012090d751dd848be710989ca23dc22d4.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/5c379af03487d59ad74ec3457030ba1012090d751dd848be710989ca23dc22d4.jpg)

![6815494ac18859b50a46c17d977e4a19f5a512c9aef14efa91129b903b3acdff.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/6815494ac18859b50a46c17d977e4a19f5a512c9aef14efa91129b903b3acdff.jpg)

![68ff07bb15cb189aedad3cb09b26f6d361c7aa535a6235cdeb978d4dcf2e7a81.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/68ff07bb15cb189aedad3cb09b26f6d361c7aa535a6235cdeb978d4dcf2e7a81.jpg)

![6c5265747ed4138d54a25db166b31f4693440e7cc4ef2fac3e59f2b61bc4a4c5.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/6c5265747ed4138d54a25db166b31f4693440e7cc4ef2fac3e59f2b61bc4a4c5.jpg)

![6c6e7c3b6e0eadb5f7d15fc305f1161400f67e6d960fc87e7e9636294a51a0ba.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/6c6e7c3b6e0eadb5f7d15fc305f1161400f67e6d960fc87e7e9636294a51a0ba.jpg)

![6ef26d1e71e49cefefcf545837140a9dd2761f782b84b30a4eb13bc7bd6daefa.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/6ef26d1e71e49cefefcf545837140a9dd2761f782b84b30a4eb13bc7bd6daefa.jpg)

![709bc761152259860fce678c52d900bbc636e03d3d5a997b628ed47a2f6378b8.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/709bc761152259860fce678c52d900bbc636e03d3d5a997b628ed47a2f6378b8.jpg)

![736464ab358a08a16a77f0010bce72018efdad0a6bbe920df922f74a7a6a50df.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/736464ab358a08a16a77f0010bce72018efdad0a6bbe920df922f74a7a6a50df.jpg)

![7de31cd81ebb6f35ef026b705437038e16badf855a6195883b33cc72ded215d2.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/7de31cd81ebb6f35ef026b705437038e16badf855a6195883b33cc72ded215d2.jpg)

![7f885567efcf6f3f1929a327fa4776a68016fb312e96165a117c700362f42fe8.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/7f885567efcf6f3f1929a327fa4776a68016fb312e96165a117c700362f42fe8.jpg)

![883202dc4386f0d7ef92402a975570341ed2a2b35bd5640dce76c74b998866dc.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/883202dc4386f0d7ef92402a975570341ed2a2b35bd5640dce76c74b998866dc.jpg)

![8b81fff405fdc5f4a2b0522ae1350024c1a193ad13df4a55df4785464a21e60c.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/8b81fff405fdc5f4a2b0522ae1350024c1a193ad13df4a55df4785464a21e60c.jpg)

![8bdc7d115d7902c7eca0ddbc60337793b7d530025a411112376201af4294d6b8.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/8bdc7d115d7902c7eca0ddbc60337793b7d530025a411112376201af4294d6b8.jpg)

![90a40e506a3dcc133a386b0131891c0fb6daacf3ea1b8e569dd090e7806c7c84.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/90a40e506a3dcc133a386b0131891c0fb6daacf3ea1b8e569dd090e7806c7c84.jpg)

![926153017b9aee15557f1648690b3d8ffcd8f16f359d8e324db2f566114e5505.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/926153017b9aee15557f1648690b3d8ffcd8f16f359d8e324db2f566114e5505.jpg)

![992c33db63ac8ad7d2c61cb4f42778596fff06abd2f1793f3673415df65b0918.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/992c33db63ac8ad7d2c61cb4f42778596fff06abd2f1793f3673415df65b0918.jpg)

![9bb7ccd13c21647e0fce591f43512435291c6a6d555c275236293eac4086c20f.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/9bb7ccd13c21647e0fce591f43512435291c6a6d555c275236293eac4086c20f.jpg)

![a780c36659eb50b694fec3f7253063f9f613328125879ccaf00f5c3ea0d7eccc.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/a780c36659eb50b694fec3f7253063f9f613328125879ccaf00f5c3ea0d7eccc.jpg)

![b10f92ebc7e3efe68cdb48be639b3b31528489161c56319e688671e6c47fb1a1.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/b10f92ebc7e3efe68cdb48be639b3b31528489161c56319e688671e6c47fb1a1.jpg)

![b5ace86442db256d9c29533e128a796e464344ec2fb7193d8ae4d90af67d3e10.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/b5ace86442db256d9c29533e128a796e464344ec2fb7193d8ae4d90af67d3e10.jpg)

![bde4c493506fe78bd7343c85a5c8721b37025d124cc8b9e080c374de9a7ed8f4.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/bde4c493506fe78bd7343c85a5c8721b37025d124cc8b9e080c374de9a7ed8f4.jpg)

![c56141216187b119411d09a7419a8e1db32c8ad0e73c79a339c20c2ddedfd71c.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/c56141216187b119411d09a7419a8e1db32c8ad0e73c79a339c20c2ddedfd71c.jpg)

![c811699ca6600e3fe5b65ae3c63f2c56607762971cf426afaff7b796da0ab9f8.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/c811699ca6600e3fe5b65ae3c63f2c56607762971cf426afaff7b796da0ab9f8.jpg)

![cb26a61a47ee4327e2ef7df3298dc31b6359cfb1dd4ef5029fdea97a9f10e110.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/cb26a61a47ee4327e2ef7df3298dc31b6359cfb1dd4ef5029fdea97a9f10e110.jpg)

![d5c66ff3a972d4b47cf1cf12525c1f41efda0a86715051cdf35bb4046913e5fd.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/d5c66ff3a972d4b47cf1cf12525c1f41efda0a86715051cdf35bb4046913e5fd.jpg)

![da4a24c0519a5ef7ebce8d2d0f6982cd41176c5cbf27785f7b17508153fbfb22.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/da4a24c0519a5ef7ebce8d2d0f6982cd41176c5cbf27785f7b17508153fbfb22.jpg)

![dffbfd7325d60eca83d95648a53206cf119ec55dfb8bb17cc173dc9f57aec389.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/dffbfd7325d60eca83d95648a53206cf119ec55dfb8bb17cc173dc9f57aec389.jpg)

![ef09276f4de498ddd2b2887abc7d3add491d613f94e804d91e8dc812d8a9f9c2.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/ef09276f4de498ddd2b2887abc7d3add491d613f94e804d91e8dc812d8a9f9c2.jpg)

![efd9bb413f1b9795fa5b6e0a19c8311a3595b1a68584da2588a643478542f5ab.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/efd9bb413f1b9795fa5b6e0a19c8311a3595b1a68584da2588a643478542f5ab.jpg)

![f12dcaf876434d5da4ff9a0a9bfdf354827274a687792cc2e74f21f3635e61a8.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/f12dcaf876434d5da4ff9a0a9bfdf354827274a687792cc2e74f21f3635e61a8.jpg)

![f2fa24eaac690b2ba2f139a987060976aa3bc975b7350dea0a66f934cb6498c3.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/f2fa24eaac690b2ba2f139a987060976aa3bc975b7350dea0a66f934cb6498c3.jpg)

![f3140fdf9751a162b87fc8fbb4cdae333f69b0c50e35f0cdcd4a32f340fc5887.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/f3140fdf9751a162b87fc8fbb4cdae333f69b0c50e35f0cdcd4a32f340fc5887.jpg)

![fb404c41c44ca1e0783bc1051ff2f61115471ad151776ec2f52e1e617f9521d2.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/images/fb404c41c44ca1e0783bc1051ff2f61115471ad151776ec2f52e1e617f9521d2.jpg)

### Tables

![1921a106b151446df14ebffb2e83eab7d8f938d21648a525dd420f9ca9730b57.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/tables/1921a106b151446df14ebffb2e83eab7d8f938d21648a525dd420f9ca9730b57.jpg)

![1d4b19872f8867df04d6c0799d5efab081677c6d48fc09ded2f712e1f5aa1bcc.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/tables/1d4b19872f8867df04d6c0799d5efab081677c6d48fc09ded2f712e1f5aa1bcc.jpg)

![2e5685b018e303fbec5fa4711debbbb4cc390054ca2a8f4a6843a10f1ec962f5.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/tables/2e5685b018e303fbec5fa4711debbbb4cc390054ca2a8f4a6843a10f1ec962f5.jpg)

![4112249e6292d9551518f56f573ef18bfa53ee1038aa1045e059a11255df1321.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/tables/4112249e6292d9551518f56f573ef18bfa53ee1038aa1045e059a11255df1321.jpg)

![6d85704b3b2511b0ab72b4c1ab3702bc747bc2442c72f25e4375e5aa9c1cf88c.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/tables/6d85704b3b2511b0ab72b4c1ab3702bc747bc2442c72f25e4375e5aa9c1cf88c.jpg)

![6f5d98230140aeeea24b108af6e30160e4b9eb5803075f8a6db9b6603ba5fe0c.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/tables/6f5d98230140aeeea24b108af6e30160e4b9eb5803075f8a6db9b6603ba5fe0c.jpg)

![7795c9016b7ff50cfe96f7c0e171cfcb1321b71881cd4ca1c36f4ed9b67d37b3.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/tables/7795c9016b7ff50cfe96f7c0e171cfcb1321b71881cd4ca1c36f4ed9b67d37b3.jpg)

![8237f5216979b8c4028a6a4ec7baaf5280bf74dc5a1159e9197e0f8fe938be82.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/tables/8237f5216979b8c4028a6a4ec7baaf5280bf74dc5a1159e9197e0f8fe938be82.jpg)

![bd583fb1190efa89a4dca169b992bc6c05988c2b23563ed5fb34883c71fc2581.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/tables/bd583fb1190efa89a4dca169b992bc6c05988c2b23563ed5fb34883c71fc2581.jpg)

![cbab562e095e0230865a8f5c52bff6821ae080b5e1a71e9bf69d252764f6d3d7.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/tables/cbab562e095e0230865a8f5c52bff6821ae080b5e1a71e9bf69d252764f6d3d7.jpg)

![fb0d7613d623ad66086ad841b9a04986f287a78e4486cf230b1b7998704e4d64.jpg](../iclr_results/2737_From%20Attention%20to%20Activation_%20Unraveling%20the%20Enigmas%20of%20Large%20Language%20Models/tables/fb0d7613d623ad66086ad841b9a04986f287a78e4486cf230b1b7998704e4d64.jpg)

## Bio-xLSTM: Generative modeling, representation and in-context learning of biological and chemical sequences


### Images

![0682963ecf5729ee89cdc5ee55a0339dea5deebed2f02ced6554724740ccea04.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/images/0682963ecf5729ee89cdc5ee55a0339dea5deebed2f02ced6554724740ccea04.jpg)

![1b378f091eec66368e35ea42a81388759f70044f5f36629110c37218c84e3804.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/images/1b378f091eec66368e35ea42a81388759f70044f5f36629110c37218c84e3804.jpg)

![21cabc8a3fac90123741c5332a477ec99cdd65a1a7e88294dc4fe1ceee36f02d.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/images/21cabc8a3fac90123741c5332a477ec99cdd65a1a7e88294dc4fe1ceee36f02d.jpg)

![392d1d273832ddf30d33515e0ec33c5f97b63d027dbc7cf327a69b161f03eba2.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/images/392d1d273832ddf30d33515e0ec33c5f97b63d027dbc7cf327a69b161f03eba2.jpg)

![5420f702b19c735c6e3aee3bd17f4b48e17c51272d73b0832e187d8e6e5c2e90.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/images/5420f702b19c735c6e3aee3bd17f4b48e17c51272d73b0832e187d8e6e5c2e90.jpg)

![5645e484096b7b63deaa4aee897e4d1c7e094f00c823287f12124d83adb52ebf.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/images/5645e484096b7b63deaa4aee897e4d1c7e094f00c823287f12124d83adb52ebf.jpg)

![70c36679b35028949fc563984b9be78cc4c10372e97ba3b66565c99fe53bd40d.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/images/70c36679b35028949fc563984b9be78cc4c10372e97ba3b66565c99fe53bd40d.jpg)

![88ede2a7f4307086e4ff024c510e17b90aa78caf543322202a7434555cfbdba6.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/images/88ede2a7f4307086e4ff024c510e17b90aa78caf543322202a7434555cfbdba6.jpg)

![b9d01fa7d81033edd050417347451d68c32ab6c9cafa1d4e7f6aaf8d4223db47.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/images/b9d01fa7d81033edd050417347451d68c32ab6c9cafa1d4e7f6aaf8d4223db47.jpg)

![c0e3216426a3afb10a6c6ea2edc9ff77a93909401ca5d3e89f40744cb56f2d2a.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/images/c0e3216426a3afb10a6c6ea2edc9ff77a93909401ca5d3e89f40744cb56f2d2a.jpg)

![d79e8cea574a4e9ff6d5904a165ed471f89b3d9ebde615f2c7d8bd65b5d5e622.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/images/d79e8cea574a4e9ff6d5904a165ed471f89b3d9ebde615f2c7d8bd65b5d5e622.jpg)

![f028646de3a7948393f1dccbf5a5cf2625e742ea50c61efef98cf34e3e6f9886.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/images/f028646de3a7948393f1dccbf5a5cf2625e742ea50c61efef98cf34e3e6f9886.jpg)

### Tables

![1a49311ba34cfe5799ff41471a287951072913821431da28f909b9672dc6dbb0.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/tables/1a49311ba34cfe5799ff41471a287951072913821431da28f909b9672dc6dbb0.jpg)

![24f295644fee905e680ae85b5e5cd168dcedc655b538432ac5378312b4ed6858.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/tables/24f295644fee905e680ae85b5e5cd168dcedc655b538432ac5378312b4ed6858.jpg)

![381b0456ca54684074ba48577d0d99e4e95ffe2b033a4b757148c0eb19e31265.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/tables/381b0456ca54684074ba48577d0d99e4e95ffe2b033a4b757148c0eb19e31265.jpg)

![3a4fa7333254d732da8dc5652a1f2146daec0d65aabfbfca95bc25c23cf7a660.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/tables/3a4fa7333254d732da8dc5652a1f2146daec0d65aabfbfca95bc25c23cf7a660.jpg)

![3fc842d52d78c5294a791c78290ea4582c365844ad812770f16b394e485d9a5b.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/tables/3fc842d52d78c5294a791c78290ea4582c365844ad812770f16b394e485d9a5b.jpg)

![43030ba03a1481d43a9c2516321db46c3dd326c93b9dd7562b5019b971b03f66.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/tables/43030ba03a1481d43a9c2516321db46c3dd326c93b9dd7562b5019b971b03f66.jpg)

![57f0b818cb31dcf90a3675f69610986ed77fa925c033b6f108b2aa5223bca0b6.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/tables/57f0b818cb31dcf90a3675f69610986ed77fa925c033b6f108b2aa5223bca0b6.jpg)

![663838980e70a801b0214d1635b6f2b75dfd3f976849b8366ba54429120add4f.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/tables/663838980e70a801b0214d1635b6f2b75dfd3f976849b8366ba54429120add4f.jpg)

![6fa5ceb61be19487216c88565eaa468c68dd590907b547b880d1c95119575482.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/tables/6fa5ceb61be19487216c88565eaa468c68dd590907b547b880d1c95119575482.jpg)

![78078dfc3d9a8aa7a97bc3b23e05a93495717d463f3ca30f19ff0a897f29d112.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/tables/78078dfc3d9a8aa7a97bc3b23e05a93495717d463f3ca30f19ff0a897f29d112.jpg)

![7efb509ee1f481319f7e30a6eafe0364c4154d42058e84f9885d6232a106afe9.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/tables/7efb509ee1f481319f7e30a6eafe0364c4154d42058e84f9885d6232a106afe9.jpg)

![83ef159a670471de3c04b5236792ab74f7c8b2003f2faeaaba250f1a264e7451.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/tables/83ef159a670471de3c04b5236792ab74f7c8b2003f2faeaaba250f1a264e7451.jpg)

![8ff3f3e1390ad6f3cdc0eb086ad13de7b83fd39a199af72d025964f1d17cd915.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/tables/8ff3f3e1390ad6f3cdc0eb086ad13de7b83fd39a199af72d025964f1d17cd915.jpg)

![9e2f99e926fe310b136d7a55111258aa8231771db0dd814407cf2e7a9f310785.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/tables/9e2f99e926fe310b136d7a55111258aa8231771db0dd814407cf2e7a9f310785.jpg)

![a92c9877c879ed36bbb5a0f1b5abfd3f23def2456cd21e62cdc7fbb3f78ea0be.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/tables/a92c9877c879ed36bbb5a0f1b5abfd3f23def2456cd21e62cdc7fbb3f78ea0be.jpg)

![d7afe2204c386b02a1efaf5b8c737b6248969456b7c643c26a37f170d42aa62d.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/tables/d7afe2204c386b02a1efaf5b8c737b6248969456b7c643c26a37f170d42aa62d.jpg)

![d89578f5cd7884b6ec044c024e1a0f9ef87ac8d9cd6c5f27e13cab525e7b0074.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/tables/d89578f5cd7884b6ec044c024e1a0f9ef87ac8d9cd6c5f27e13cab525e7b0074.jpg)

![f247f3181111f5311d26e80c91394a6fec8f234153204ff6080d9b58220560c5.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/tables/f247f3181111f5311d26e80c91394a6fec8f234153204ff6080d9b58220560c5.jpg)

![f9378a9e1823a0270f5e66e5eea09fd0b5781251f482c135c8fd1a7154768a9d.jpg](../iclr_results/2738_Bio-xLSTM_%20Generative%20modeling%2C%20representation%20and%20in-context%20learning%20of%20biological%20and%20chemical%20se/tables/f9378a9e1823a0270f5e66e5eea09fd0b5781251f482c135c8fd1a7154768a9d.jpg)

## The Hyperfitting Phenomenon: Sharpening and Stabilizing LLMs for Open-Ended Text Generation


### Images

![0b63243e19a80aebcdbe3ad84bab8c9764331e4416e64d7b2a1951db9ea6fa37.jpg](../iclr_results/2739_The%20Hyperfitting%20Phenomenon_%20Sharpening%20and%20Stabilizing%20LLMs%20for%20Open-Ended%20Text%20Generation/images/0b63243e19a80aebcdbe3ad84bab8c9764331e4416e64d7b2a1951db9ea6fa37.jpg)

![20772d6236c8c3b167f8b4d3d5838fc3fb550c001fa842f7d92be0816cfe795d.jpg](../iclr_results/2739_The%20Hyperfitting%20Phenomenon_%20Sharpening%20and%20Stabilizing%20LLMs%20for%20Open-Ended%20Text%20Generation/images/20772d6236c8c3b167f8b4d3d5838fc3fb550c001fa842f7d92be0816cfe795d.jpg)

![20b2e504e5c21bec71eb72cbf101b0c72f9dacf52fec3f1c846ea95946c3bbe5.jpg](../iclr_results/2739_The%20Hyperfitting%20Phenomenon_%20Sharpening%20and%20Stabilizing%20LLMs%20for%20Open-Ended%20Text%20Generation/images/20b2e504e5c21bec71eb72cbf101b0c72f9dacf52fec3f1c846ea95946c3bbe5.jpg)

![2713ecc0acbca976006d8d1b4f83e0d663d992a7146c7fda45511b74dffff64c.jpg](../iclr_results/2739_The%20Hyperfitting%20Phenomenon_%20Sharpening%20and%20Stabilizing%20LLMs%20for%20Open-Ended%20Text%20Generation/images/2713ecc0acbca976006d8d1b4f83e0d663d992a7146c7fda45511b74dffff64c.jpg)

![27998592308c08a22b5ca2ffa3f2666dd5c32de9c438809b73dafc5a1e959b2f.jpg](../iclr_results/2739_The%20Hyperfitting%20Phenomenon_%20Sharpening%20and%20Stabilizing%20LLMs%20for%20Open-Ended%20Text%20Generation/images/27998592308c08a22b5ca2ffa3f2666dd5c32de9c438809b73dafc5a1e959b2f.jpg)

![51604706b0580831e0d378746c7bb6787fc656cb5ff81a3873f8b58f8d7fb082.jpg](../iclr_results/2739_The%20Hyperfitting%20Phenomenon_%20Sharpening%20and%20Stabilizing%20LLMs%20for%20Open-Ended%20Text%20Generation/images/51604706b0580831e0d378746c7bb6787fc656cb5ff81a3873f8b58f8d7fb082.jpg)

![c99ac4711b7bd3cd245d66850c3872527311d887eab89dfab4167275514bb4e8.jpg](../iclr_results/2739_The%20Hyperfitting%20Phenomenon_%20Sharpening%20and%20Stabilizing%20LLMs%20for%20Open-Ended%20Text%20Generation/images/c99ac4711b7bd3cd245d66850c3872527311d887eab89dfab4167275514bb4e8.jpg)

![eb01e61363ffc938fcbe8838b5c74a2271ba1c6ce0722b78bda4abbd1810ed84.jpg](../iclr_results/2739_The%20Hyperfitting%20Phenomenon_%20Sharpening%20and%20Stabilizing%20LLMs%20for%20Open-Ended%20Text%20Generation/images/eb01e61363ffc938fcbe8838b5c74a2271ba1c6ce0722b78bda4abbd1810ed84.jpg)

![f35d9066757d67ed56491ec0f63f9db6ce0c3488c29b958c9b8b7d54181bec00.jpg](../iclr_results/2739_The%20Hyperfitting%20Phenomenon_%20Sharpening%20and%20Stabilizing%20LLMs%20for%20Open-Ended%20Text%20Generation/images/f35d9066757d67ed56491ec0f63f9db6ce0c3488c29b958c9b8b7d54181bec00.jpg)

### Tables

![3028ab1966608d0638db179c78a7f7f00b2a2941aa12db3528b5b0b3aae62c2a.jpg](../iclr_results/2739_The%20Hyperfitting%20Phenomenon_%20Sharpening%20and%20Stabilizing%20LLMs%20for%20Open-Ended%20Text%20Generation/tables/3028ab1966608d0638db179c78a7f7f00b2a2941aa12db3528b5b0b3aae62c2a.jpg)

![3589c32c52b021ee1d3af2ea8d74aae2834dcda8bfd61e245422a7d408f5017e.jpg](../iclr_results/2739_The%20Hyperfitting%20Phenomenon_%20Sharpening%20and%20Stabilizing%20LLMs%20for%20Open-Ended%20Text%20Generation/tables/3589c32c52b021ee1d3af2ea8d74aae2834dcda8bfd61e245422a7d408f5017e.jpg)

![47bd1f6f22a87132615f0c9ade5afd62c547ced70cb390454667decef4a2421a.jpg](../iclr_results/2739_The%20Hyperfitting%20Phenomenon_%20Sharpening%20and%20Stabilizing%20LLMs%20for%20Open-Ended%20Text%20Generation/tables/47bd1f6f22a87132615f0c9ade5afd62c547ced70cb390454667decef4a2421a.jpg)

![5e44561a376ca3cba5581c994f4420d7c890991c7677c01e2015214c39739ef5.jpg](../iclr_results/2739_The%20Hyperfitting%20Phenomenon_%20Sharpening%20and%20Stabilizing%20LLMs%20for%20Open-Ended%20Text%20Generation/tables/5e44561a376ca3cba5581c994f4420d7c890991c7677c01e2015214c39739ef5.jpg)

![5ecebb581d78c057149d739763074e259a1b453f9770ae86e5ced285ef93210e.jpg](../iclr_results/2739_The%20Hyperfitting%20Phenomenon_%20Sharpening%20and%20Stabilizing%20LLMs%20for%20Open-Ended%20Text%20Generation/tables/5ecebb581d78c057149d739763074e259a1b453f9770ae86e5ced285ef93210e.jpg)

![5fff4bf1cfbfaad3e63344d1edec13972f6216fbbf17ca8f7852634d0aaa0986.jpg](../iclr_results/2739_The%20Hyperfitting%20Phenomenon_%20Sharpening%20and%20Stabilizing%20LLMs%20for%20Open-Ended%20Text%20Generation/tables/5fff4bf1cfbfaad3e63344d1edec13972f6216fbbf17ca8f7852634d0aaa0986.jpg)

![61b980a78420f108c34ff98be8e0a0406809739e0f97f9795a64e5291c334c90.jpg](../iclr_results/2739_The%20Hyperfitting%20Phenomenon_%20Sharpening%20and%20Stabilizing%20LLMs%20for%20Open-Ended%20Text%20Generation/tables/61b980a78420f108c34ff98be8e0a0406809739e0f97f9795a64e5291c334c90.jpg)

![bc80fac37865af73b2905f03a3cf61f324d8187757bbba8be5d2700ea5ad9a67.jpg](../iclr_results/2739_The%20Hyperfitting%20Phenomenon_%20Sharpening%20and%20Stabilizing%20LLMs%20for%20Open-Ended%20Text%20Generation/tables/bc80fac37865af73b2905f03a3cf61f324d8187757bbba8be5d2700ea5ad9a67.jpg)

![d7dba3dc6db46be568f67512470fc583ac239f442ee93bcc047fd79127c58c2c.jpg](../iclr_results/2739_The%20Hyperfitting%20Phenomenon_%20Sharpening%20and%20Stabilizing%20LLMs%20for%20Open-Ended%20Text%20Generation/tables/d7dba3dc6db46be568f67512470fc583ac239f442ee93bcc047fd79127c58c2c.jpg)

![f5baf270ae2d4b590d5bdbecde67e3fa09491e9e5be8e601bd45542bb6bdd26f.jpg](../iclr_results/2739_The%20Hyperfitting%20Phenomenon_%20Sharpening%20and%20Stabilizing%20LLMs%20for%20Open-Ended%20Text%20Generation/tables/f5baf270ae2d4b590d5bdbecde67e3fa09491e9e5be8e601bd45542bb6bdd26f.jpg)

## Memory Mosaics


### Images

![1b56efede090775e2cd80c9fa0fe5379ed6e1e97a5415d037ee19a091dfe8a32.jpg](../iclr_results/2740_Memory%20Mosaics/images/1b56efede090775e2cd80c9fa0fe5379ed6e1e97a5415d037ee19a091dfe8a32.jpg)

![23c847084a352a8b0a79216142479481e2bf7be454e4cbdb57c3b74b59998b34.jpg](../iclr_results/2740_Memory%20Mosaics/images/23c847084a352a8b0a79216142479481e2bf7be454e4cbdb57c3b74b59998b34.jpg)

![2d972cb295330810e390bb90c7e5ebd414211f61f8a95ee35317e276ffe93fae.jpg](../iclr_results/2740_Memory%20Mosaics/images/2d972cb295330810e390bb90c7e5ebd414211f61f8a95ee35317e276ffe93fae.jpg)

![359ea2dd7cfa2c5e3c798c98c7e48b20bd1d67de0b72429e5eff72bf38b7ebfe.jpg](../iclr_results/2740_Memory%20Mosaics/images/359ea2dd7cfa2c5e3c798c98c7e48b20bd1d67de0b72429e5eff72bf38b7ebfe.jpg)

![5ed0528a262409ca097ab8b1ecc72868c2bbaa29330b2c41852c4f21a31de25d.jpg](../iclr_results/2740_Memory%20Mosaics/images/5ed0528a262409ca097ab8b1ecc72868c2bbaa29330b2c41852c4f21a31de25d.jpg)

![8fd80d8cd8293bddf4e8263ba84ede63e9f3711f7ed023d9b41802c9304bd74d.jpg](../iclr_results/2740_Memory%20Mosaics/images/8fd80d8cd8293bddf4e8263ba84ede63e9f3711f7ed023d9b41802c9304bd74d.jpg)

![9f56eee9517d580f097342d9b4138ef21722ef91e04279e7994371481998316d.jpg](../iclr_results/2740_Memory%20Mosaics/images/9f56eee9517d580f097342d9b4138ef21722ef91e04279e7994371481998316d.jpg)

![be15ad9dcdf4ef11f491d93cf734137a9a60ac83c900e5578e7abd509fe4f96b.jpg](../iclr_results/2740_Memory%20Mosaics/images/be15ad9dcdf4ef11f491d93cf734137a9a60ac83c900e5578e7abd509fe4f96b.jpg)

![c26b466c1670b41259f13a8f2f9430bcbb843e63536a996b9d6196ae0a60dcac.jpg](../iclr_results/2740_Memory%20Mosaics/images/c26b466c1670b41259f13a8f2f9430bcbb843e63536a996b9d6196ae0a60dcac.jpg)

![c3ad7fd63396c9aadd61c7b7e3752bd7fdca57a05aed99e6386b7199fb081301.jpg](../iclr_results/2740_Memory%20Mosaics/images/c3ad7fd63396c9aadd61c7b7e3752bd7fdca57a05aed99e6386b7199fb081301.jpg)

![d691b2d7bce0ee4aa9f90d233e7ef2a88e376e1f9aa9efe1993a4224af1db3da.jpg](../iclr_results/2740_Memory%20Mosaics/images/d691b2d7bce0ee4aa9f90d233e7ef2a88e376e1f9aa9efe1993a4224af1db3da.jpg)

![deed64164eb5318c6d9f8ebb9d79fe57d0ca29fa37e5cecafb2452fa899581f3.jpg](../iclr_results/2740_Memory%20Mosaics/images/deed64164eb5318c6d9f8ebb9d79fe57d0ca29fa37e5cecafb2452fa899581f3.jpg)

![e163bd7632e69a59ea438e6646ddbe6c5110869333b783cff2cdf859416b20c0.jpg](../iclr_results/2740_Memory%20Mosaics/images/e163bd7632e69a59ea438e6646ddbe6c5110869333b783cff2cdf859416b20c0.jpg)

![f1901eb85abb27a5b966351a2195ba8ad542a68f332ed560871d1b7159c8707e.jpg](../iclr_results/2740_Memory%20Mosaics/images/f1901eb85abb27a5b966351a2195ba8ad542a68f332ed560871d1b7159c8707e.jpg)

![f2d75c6c8c71d515a26413a27bda84c0f9bbb35c04c62ee5168b3332204fa123.jpg](../iclr_results/2740_Memory%20Mosaics/images/f2d75c6c8c71d515a26413a27bda84c0f9bbb35c04c62ee5168b3332204fa123.jpg)

![fb89b8cffdd6c5c3eff862111ec5ab31f725a7a4b1814df821c0ba7d11044439.jpg](../iclr_results/2740_Memory%20Mosaics/images/fb89b8cffdd6c5c3eff862111ec5ab31f725a7a4b1814df821c0ba7d11044439.jpg)

### Tables

![11f8c2c7617ad3f7f6e9646a84329c7b2885af59387624d67f29ee3cf3512791.jpg](../iclr_results/2740_Memory%20Mosaics/tables/11f8c2c7617ad3f7f6e9646a84329c7b2885af59387624d67f29ee3cf3512791.jpg)

![20dddbcdac7b3cd456725748d3b2c377e7933d3797db089781ed24550efc57f1.jpg](../iclr_results/2740_Memory%20Mosaics/tables/20dddbcdac7b3cd456725748d3b2c377e7933d3797db089781ed24550efc57f1.jpg)

![76aae20f8b78aebd85413689258a268b57a0a42a649caf2d98aec3a0af7161b9.jpg](../iclr_results/2740_Memory%20Mosaics/tables/76aae20f8b78aebd85413689258a268b57a0a42a649caf2d98aec3a0af7161b9.jpg)

![9e66209fa8ed88995a330c0e16e02d6a67e90347d38619661e13cd396e37cdd5.jpg](../iclr_results/2740_Memory%20Mosaics/tables/9e66209fa8ed88995a330c0e16e02d6a67e90347d38619661e13cd396e37cdd5.jpg)

![a5e59779e53884f8e3e0cd33e8eb677f6f19689010f152040bf67db532487098.jpg](../iclr_results/2740_Memory%20Mosaics/tables/a5e59779e53884f8e3e0cd33e8eb677f6f19689010f152040bf67db532487098.jpg)

## Start Smart: Leveraging Gradients For Enhancing Mask-based XAI Methods


### Images

![0164b28d87415e8d3efb2fefa64ce037eccf7c0542f4f4c1036528799b013913.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/0164b28d87415e8d3efb2fefa64ce037eccf7c0542f4f4c1036528799b013913.jpg)

![01b85a9acfdabdd71dcf1443e7508d7b828e3404b85c9ca2b3eb14d5d0013f2f.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/01b85a9acfdabdd71dcf1443e7508d7b828e3404b85c9ca2b3eb14d5d0013f2f.jpg)

![14b1f6c82db41c0ed9c4b2b2d714ac80e8720b55e1154470f6c597d714c7a438.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/14b1f6c82db41c0ed9c4b2b2d714ac80e8720b55e1154470f6c597d714c7a438.jpg)

![199eedbf0a596f3414066822e998f110a62451e232a7386f03f01bfac46b77f4.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/199eedbf0a596f3414066822e998f110a62451e232a7386f03f01bfac46b77f4.jpg)

![2a92c678eaecccf202034f2ad35d6331653309f90fa9ea770309e5cf04fb0362.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/2a92c678eaecccf202034f2ad35d6331653309f90fa9ea770309e5cf04fb0362.jpg)

![2bbe8c665c87aca4ee9d935e422235ee3e952953d009895854f892101c5e1e88.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/2bbe8c665c87aca4ee9d935e422235ee3e952953d009895854f892101c5e1e88.jpg)

![2f2bbd1398355b0d81ac9924aa3d8058ee112922e2844057d4f58004590a0c06.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/2f2bbd1398355b0d81ac9924aa3d8058ee112922e2844057d4f58004590a0c06.jpg)

![3107e3d978ffb4725d6183b15bd6f285f781b2287aba7144aff14e3a145becc5.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/3107e3d978ffb4725d6183b15bd6f285f781b2287aba7144aff14e3a145becc5.jpg)

![31089c76a0572d322eceaea16ede5fa62634397d17d2e86760addd0ffd46f474.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/31089c76a0572d322eceaea16ede5fa62634397d17d2e86760addd0ffd46f474.jpg)

![3d2afabb258b34e96ddb7d56f3d6bd8faa9c988ac4780ea08f30dfb445b308e3.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/3d2afabb258b34e96ddb7d56f3d6bd8faa9c988ac4780ea08f30dfb445b308e3.jpg)

![463f4fb3a8b14519f3997b6b450f2ec540eb2e12a2ce8dbd18feede226638891.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/463f4fb3a8b14519f3997b6b450f2ec540eb2e12a2ce8dbd18feede226638891.jpg)

![485506522321b40debce27710a1ad5a9a2d781094ea2c2988669c7d6f2be7d59.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/485506522321b40debce27710a1ad5a9a2d781094ea2c2988669c7d6f2be7d59.jpg)

![4db2c661685f3121106b4371175e00b982a0deea0df3eb5d8970dcaf4d96734e.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/4db2c661685f3121106b4371175e00b982a0deea0df3eb5d8970dcaf4d96734e.jpg)

![4f99bb43da759489294bb06d79f273e68945d7b3b5e0a1be792c6f3efdf00186.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/4f99bb43da759489294bb06d79f273e68945d7b3b5e0a1be792c6f3efdf00186.jpg)

![51ea1351b3fc526dd8cecfc8bb44310bf8f1fe926fa70fafe1623e0fdad08800.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/51ea1351b3fc526dd8cecfc8bb44310bf8f1fe926fa70fafe1623e0fdad08800.jpg)

![5673898b9550d6adb904e02cb86cb44e1a0293d3b9562ace110f0d748651ef4d.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/5673898b9550d6adb904e02cb86cb44e1a0293d3b9562ace110f0d748651ef4d.jpg)

![6157c437d09cc461ba5ef3510a228b5c620da9202151fd94d1ec280934ae7ceb.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/6157c437d09cc461ba5ef3510a228b5c620da9202151fd94d1ec280934ae7ceb.jpg)

![622aeef9a2c1726e438b5021aa60d9de157bc82f941c72494f6ecad7f3e1a04e.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/622aeef9a2c1726e438b5021aa60d9de157bc82f941c72494f6ecad7f3e1a04e.jpg)

![638d4a0dc261e7689f3a2b6616f1740b0cf8140c61b9f63b43ec39cb020d1d35.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/638d4a0dc261e7689f3a2b6616f1740b0cf8140c61b9f63b43ec39cb020d1d35.jpg)

![6c76527ea288872ea8062679a2efcd79febc0e73a6fa666e2b7482f21c768bb1.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/6c76527ea288872ea8062679a2efcd79febc0e73a6fa666e2b7482f21c768bb1.jpg)

![6fbae0cc883cdb36306dd3b4085dc7cfff2d58aff4749b792a585724d218c234.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/6fbae0cc883cdb36306dd3b4085dc7cfff2d58aff4749b792a585724d218c234.jpg)

![7a6c2843a9deb49c3343c12914630ede65ef063896b31ff1b9c3c36000edb60e.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/7a6c2843a9deb49c3343c12914630ede65ef063896b31ff1b9c3c36000edb60e.jpg)

![7b8a6013d7e610c9e18485fbb9ca0b0145f446bfe89f3061c73de6a87c684e4c.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/7b8a6013d7e610c9e18485fbb9ca0b0145f446bfe89f3061c73de6a87c684e4c.jpg)

![8c274e3c65def006e085b065322431112b52891c45000ad6ef58bd5401cd58c5.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/8c274e3c65def006e085b065322431112b52891c45000ad6ef58bd5401cd58c5.jpg)

![9c23976cee557638ec4934379fd65bb5cd4411e8ab5841261336bd53cc23186c.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/9c23976cee557638ec4934379fd65bb5cd4411e8ab5841261336bd53cc23186c.jpg)

![a6d9c38b76cc5e249d386d672d31688724e9780ea3d83e97d0884941d402d5d7.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/a6d9c38b76cc5e249d386d672d31688724e9780ea3d83e97d0884941d402d5d7.jpg)

![b133fbe2004c93c751cfcb8b0174a3ca5a20d969efbac95f7665d7640b737670.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/b133fbe2004c93c751cfcb8b0174a3ca5a20d969efbac95f7665d7640b737670.jpg)

![b7b505ab9f0cce11db38170f3994b7e05926c02d258488454fc3d2d6981e550d.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/b7b505ab9f0cce11db38170f3994b7e05926c02d258488454fc3d2d6981e550d.jpg)

![ba3f798720c4d8d2be253529966b0abf7e08cce1580bcdcda42f90340bb6f921.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/ba3f798720c4d8d2be253529966b0abf7e08cce1580bcdcda42f90340bb6f921.jpg)

![c7d1727fb85792d89ead34f711d58b3a3ec57dc506f01d7a19daeef238da6a22.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/c7d1727fb85792d89ead34f711d58b3a3ec57dc506f01d7a19daeef238da6a22.jpg)

![cc55567f3b68530a8a7395e5a4da1f4f1ce211b83069df134c8477e231181085.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/cc55567f3b68530a8a7395e5a4da1f4f1ce211b83069df134c8477e231181085.jpg)

![cf2b8e82023acc655197c4ac591550ec6c785e80f33e5d345c5598f38ece538f.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/cf2b8e82023acc655197c4ac591550ec6c785e80f33e5d345c5598f38ece538f.jpg)

![d9ac386d72b45339023d690aafd76674931126fc44977561f0d3e52f2b270d94.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/d9ac386d72b45339023d690aafd76674931126fc44977561f0d3e52f2b270d94.jpg)

![d9b062323df4bdd84cf4b58997ad884d4db859be3736bb2b402c7b378dc45801.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/d9b062323df4bdd84cf4b58997ad884d4db859be3736bb2b402c7b378dc45801.jpg)

![eff136fb935062362c2fc9503c700858c8182fe31a045352734391e300852617.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/eff136fb935062362c2fc9503c700858c8182fe31a045352734391e300852617.jpg)

![f05816ab738536c56c9b02da5ed1994fcb124f774c77866c0b03ff768a254d94.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/images/f05816ab738536c56c9b02da5ed1994fcb124f774c77866c0b03ff768a254d94.jpg)

### Tables

![00234fc1cb22f1c223d7c0e777fc8b92de91050dc0427d5b15c322ae844f6726.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/tables/00234fc1cb22f1c223d7c0e777fc8b92de91050dc0427d5b15c322ae844f6726.jpg)

![048b27a0f64447b9f02c66d9c909633ed3f7954da6aa946e2093a30cfd4e79ce.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/tables/048b27a0f64447b9f02c66d9c909633ed3f7954da6aa946e2093a30cfd4e79ce.jpg)

![24f373e966b4c7f23967b64b7a811f5f3dbe8e3a909dc253e9e1748fd5b83c48.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/tables/24f373e966b4c7f23967b64b7a811f5f3dbe8e3a909dc253e9e1748fd5b83c48.jpg)

![31a2b67029ffffd786cbf2bfa7958103cff2e11a413521fbc2f9df6777e1194e.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/tables/31a2b67029ffffd786cbf2bfa7958103cff2e11a413521fbc2f9df6777e1194e.jpg)

![346c8e518530cb40332ef838293dadc65a1585a7629444a19df5b8ad1f9bf2cc.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/tables/346c8e518530cb40332ef838293dadc65a1585a7629444a19df5b8ad1f9bf2cc.jpg)

![4d328ba2dddee1b39d119d86eda9d0f46322bb229dc209ef09f4f7e210a4cbc0.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/tables/4d328ba2dddee1b39d119d86eda9d0f46322bb229dc209ef09f4f7e210a4cbc0.jpg)

![5999edf889a5f52e2a632dc919b3bab3557315caeb3975e293e02d23ed8b288f.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/tables/5999edf889a5f52e2a632dc919b3bab3557315caeb3975e293e02d23ed8b288f.jpg)

![5b5de84a229a140894e1306222cd937cb69ce6d958b8310f1dd727233c27d0ff.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/tables/5b5de84a229a140894e1306222cd937cb69ce6d958b8310f1dd727233c27d0ff.jpg)

![75bb2bec6f4a03fec3cafa636a35c6070a2f82d8a83d3a69e6c3787bcdad222c.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/tables/75bb2bec6f4a03fec3cafa636a35c6070a2f82d8a83d3a69e6c3787bcdad222c.jpg)

![7f848b1d7bccabd566a9b822e30c60ad64b43818cbea31481e5726e88d5eb30e.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/tables/7f848b1d7bccabd566a9b822e30c60ad64b43818cbea31481e5726e88d5eb30e.jpg)

![b56ebf628604e65cde2897f5b318967e48bc9bbd42c6fcb61b047b73f6f2a41a.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/tables/b56ebf628604e65cde2897f5b318967e48bc9bbd42c6fcb61b047b73f6f2a41a.jpg)

![b75612c6a67049f21cc9e1a3fc793fe6ea2ea6b60fbb5987fc15aec6ba3390ae.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/tables/b75612c6a67049f21cc9e1a3fc793fe6ea2ea6b60fbb5987fc15aec6ba3390ae.jpg)

![bc722538054637472877e0af917971763365fd9a1305cc60c03aa971f1ab037c.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/tables/bc722538054637472877e0af917971763365fd9a1305cc60c03aa971f1ab037c.jpg)

![d5a1141986c75a0db4d505033334403bac39ee1a1b84e20cd933a7c586aadb66.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/tables/d5a1141986c75a0db4d505033334403bac39ee1a1b84e20cd933a7c586aadb66.jpg)

![f6e65461d2ccfca7393eafea8e2f4d5deed82b3af1842eece4ebc9392a1fef59.jpg](../iclr_results/2741_Start%20Smart_%20Leveraging%20Gradients%20For%20Enhancing%20Mask-based%20XAI%20Methods/tables/f6e65461d2ccfca7393eafea8e2f4d5deed82b3af1842eece4ebc9392a1fef59.jpg)

## Controllable Context Sensitivity and the Knob Behind It


### Images

![057195532bdaaef67d17987b7f36d2cf526cf7ab1a61ca1b1fdfe933577a60ea.jpg](../iclr_results/2742_Controllable%20Context%20Sensitivity%20and%20the%20Knob%20Behind%20It/images/057195532bdaaef67d17987b7f36d2cf526cf7ab1a61ca1b1fdfe933577a60ea.jpg)

![2b255c54195722ed9d762e94ced50a71a98d4a4eaeaf12837888dd905c23245c.jpg](../iclr_results/2742_Controllable%20Context%20Sensitivity%20and%20the%20Knob%20Behind%20It/images/2b255c54195722ed9d762e94ced50a71a98d4a4eaeaf12837888dd905c23245c.jpg)

![3efbf15b2088ee5d9feccfa9986f0760ab6dcf8bbdbf1d7170eabaaf4e90d584.jpg](../iclr_results/2742_Controllable%20Context%20Sensitivity%20and%20the%20Knob%20Behind%20It/images/3efbf15b2088ee5d9feccfa9986f0760ab6dcf8bbdbf1d7170eabaaf4e90d584.jpg)

![443fe76efe312890d477944e6d24dc5ccde9483c60e9dc24185253a24600da3b.jpg](../iclr_results/2742_Controllable%20Context%20Sensitivity%20and%20the%20Knob%20Behind%20It/images/443fe76efe312890d477944e6d24dc5ccde9483c60e9dc24185253a24600da3b.jpg)

![4a7c805890852f4959a66bdd54eddded23c8ff359aa94276fc852aa441fdd58f.jpg](../iclr_results/2742_Controllable%20Context%20Sensitivity%20and%20the%20Knob%20Behind%20It/images/4a7c805890852f4959a66bdd54eddded23c8ff359aa94276fc852aa441fdd58f.jpg)

![5a69e6866680d53ad42b6b9d343ebc28e255564dd1d852a8759ac68b1b07e31f.jpg](../iclr_results/2742_Controllable%20Context%20Sensitivity%20and%20the%20Knob%20Behind%20It/images/5a69e6866680d53ad42b6b9d343ebc28e255564dd1d852a8759ac68b1b07e31f.jpg)

![605dd85d78efe1e28f541ccca347cd99b37d6b3c63d46371a13ffb090ca66240.jpg](../iclr_results/2742_Controllable%20Context%20Sensitivity%20and%20the%20Knob%20Behind%20It/images/605dd85d78efe1e28f541ccca347cd99b37d6b3c63d46371a13ffb090ca66240.jpg)

![82b7b239319ca0f2c789c21a2857bd5603b2227f3855a057a14a352a5c003a6f.jpg](../iclr_results/2742_Controllable%20Context%20Sensitivity%20and%20the%20Knob%20Behind%20It/images/82b7b239319ca0f2c789c21a2857bd5603b2227f3855a057a14a352a5c003a6f.jpg)

![8c72d74b8c50ef74cd7d8dea0660ec8c72bae363d8881efec8c7952482c5d887.jpg](../iclr_results/2742_Controllable%20Context%20Sensitivity%20and%20the%20Knob%20Behind%20It/images/8c72d74b8c50ef74cd7d8dea0660ec8c72bae363d8881efec8c7952482c5d887.jpg)

![905bce99d3a90e4daa5c904fe0107c34d69d1bd2344a6cd2d91b1cb3d2784159.jpg](../iclr_results/2742_Controllable%20Context%20Sensitivity%20and%20the%20Knob%20Behind%20It/images/905bce99d3a90e4daa5c904fe0107c34d69d1bd2344a6cd2d91b1cb3d2784159.jpg)

![9b45f1990b47e5663311ea27ad54a989dc3bf06a37c1e4ebc69ee53f34fc0fbc.jpg](../iclr_results/2742_Controllable%20Context%20Sensitivity%20and%20the%20Knob%20Behind%20It/images/9b45f1990b47e5663311ea27ad54a989dc3bf06a37c1e4ebc69ee53f34fc0fbc.jpg)

![aa8f537699e462a3bb335e6ba345cae85da60a68d292cf1625106313ab034c46.jpg](../iclr_results/2742_Controllable%20Context%20Sensitivity%20and%20the%20Knob%20Behind%20It/images/aa8f537699e462a3bb335e6ba345cae85da60a68d292cf1625106313ab034c46.jpg)

![bbde28ca650688783836a2e0aed340f6e873fd2b7649a4f67f589286e8f18a2d.jpg](../iclr_results/2742_Controllable%20Context%20Sensitivity%20and%20the%20Knob%20Behind%20It/images/bbde28ca650688783836a2e0aed340f6e873fd2b7649a4f67f589286e8f18a2d.jpg)

![d53fb75759be419bfacb883662824ba80bb46601e24c707279a66ffe415651c8.jpg](../iclr_results/2742_Controllable%20Context%20Sensitivity%20and%20the%20Knob%20Behind%20It/images/d53fb75759be419bfacb883662824ba80bb46601e24c707279a66ffe415651c8.jpg)

![db1f8ac1541133d2917f584cc61c849baa7485db480c3fa01f1c95300384183c.jpg](../iclr_results/2742_Controllable%20Context%20Sensitivity%20and%20the%20Knob%20Behind%20It/images/db1f8ac1541133d2917f584cc61c849baa7485db480c3fa01f1c95300384183c.jpg)

![de6b3a17a4ba8f5e9fc62ec2b7bd6d7addb10f8fd604f3a8e499373907c74729.jpg](../iclr_results/2742_Controllable%20Context%20Sensitivity%20and%20the%20Knob%20Behind%20It/images/de6b3a17a4ba8f5e9fc62ec2b7bd6d7addb10f8fd604f3a8e499373907c74729.jpg)

![e2ef64faf8625d8126cb7d622da65643995528a4eaf8749e1b8b459e930e7e69.jpg](../iclr_results/2742_Controllable%20Context%20Sensitivity%20and%20the%20Knob%20Behind%20It/images/e2ef64faf8625d8126cb7d622da65643995528a4eaf8749e1b8b459e930e7e69.jpg)

![f8bc1ab9698187fefdc51648e8dbda1352ed588657422f158bb0ccef0e20793a.jpg](../iclr_results/2742_Controllable%20Context%20Sensitivity%20and%20the%20Knob%20Behind%20It/images/f8bc1ab9698187fefdc51648e8dbda1352ed588657422f158bb0ccef0e20793a.jpg)

### Tables

![7a8f10ef5dbc57d3482188aa99ebf672393691b7db04400115dcae028aba8a40.jpg](../iclr_results/2742_Controllable%20Context%20Sensitivity%20and%20the%20Knob%20Behind%20It/tables/7a8f10ef5dbc57d3482188aa99ebf672393691b7db04400115dcae028aba8a40.jpg)

![a1432812e0f933c54461c813a0c10a4d6a62f76532ee3cb5b3c57f5ac6d36517.jpg](../iclr_results/2742_Controllable%20Context%20Sensitivity%20and%20the%20Knob%20Behind%20It/tables/a1432812e0f933c54461c813a0c10a4d6a62f76532ee3cb5b3c57f5ac6d36517.jpg)

![b67f0ae9ab8bf2710bcce8b3c1e90b190885e593e3c92f139e6e3afb2743933d.jpg](../iclr_results/2742_Controllable%20Context%20Sensitivity%20and%20the%20Knob%20Behind%20It/tables/b67f0ae9ab8bf2710bcce8b3c1e90b190885e593e3c92f139e6e3afb2743933d.jpg)

## Intelligence at the Edge of Chaos


### Images

![0ea757e5cc342619b1eddf083d5428d51ef556eb901b2d7ceff6f98d0b28631c.jpg](../iclr_results/2743_Intelligence%20at%20the%20Edge%20of%20Chaos/images/0ea757e5cc342619b1eddf083d5428d51ef556eb901b2d7ceff6f98d0b28631c.jpg)

![80926ead401745a41c6c34dc406b64dc6e3ad4f49a4d536d0ee0ccea7f4be671.jpg](../iclr_results/2743_Intelligence%20at%20the%20Edge%20of%20Chaos/images/80926ead401745a41c6c34dc406b64dc6e3ad4f49a4d536d0ee0ccea7f4be671.jpg)

![9ce545ddb8882716318c818240b801ddfb3cf68c198d38e7daf938e6f134de21.jpg](../iclr_results/2743_Intelligence%20at%20the%20Edge%20of%20Chaos/images/9ce545ddb8882716318c818240b801ddfb3cf68c198d38e7daf938e6f134de21.jpg)

![a62e77589d5042c36ce4884220141a4958c9a77b4df20bc06e1865bd4cfb5ada.jpg](../iclr_results/2743_Intelligence%20at%20the%20Edge%20of%20Chaos/images/a62e77589d5042c36ce4884220141a4958c9a77b4df20bc06e1865bd4cfb5ada.jpg)

![ad25f4c0140e345caec2603da3d38838aab275f620d90d59542a617c44a7c035.jpg](../iclr_results/2743_Intelligence%20at%20the%20Edge%20of%20Chaos/images/ad25f4c0140e345caec2603da3d38838aab275f620d90d59542a617c44a7c035.jpg)

![c630a9b8cf360911a5db77b37e9704453dafb25e7a05b6883bf814b798c11762.jpg](../iclr_results/2743_Intelligence%20at%20the%20Edge%20of%20Chaos/images/c630a9b8cf360911a5db77b37e9704453dafb25e7a05b6883bf814b798c11762.jpg)

![cef4f83a49deb4013a61b70416f2e2d7d44be26751ab42c7cd8b7a85091f2527.jpg](../iclr_results/2743_Intelligence%20at%20the%20Edge%20of%20Chaos/images/cef4f83a49deb4013a61b70416f2e2d7d44be26751ab42c7cd8b7a85091f2527.jpg)

![ed13bc7a837a83bea258c9bca2c9bfe998dff7f635c24d025cfd01ce1f85924a.jpg](../iclr_results/2743_Intelligence%20at%20the%20Edge%20of%20Chaos/images/ed13bc7a837a83bea258c9bca2c9bfe998dff7f635c24d025cfd01ce1f85924a.jpg)

### Tables

![81c523e62e98f7c23aadfa48eacb2cb2fed7cea8b8e30b3e53078fca34b7418e.jpg](../iclr_results/2743_Intelligence%20at%20the%20Edge%20of%20Chaos/tables/81c523e62e98f7c23aadfa48eacb2cb2fed7cea8b8e30b3e53078fca34b7418e.jpg)

## DynAlign: Unsupervised Dynamic Taxonomy Alignment for Cross-Domain Segmentation


### Images

![57422703987316da9adf530506d380584ec87efc75d90127cba44a22b53cdcce.jpg](../iclr_results/2744_DynAlign_%20Unsupervised%20Dynamic%20Taxonomy%20Alignment%20for%20Cross-Domain%20Segmentation/images/57422703987316da9adf530506d380584ec87efc75d90127cba44a22b53cdcce.jpg)

![62f9fef71bc02ec954ae68be4af63721d767ef96ef8b76417dfc47e42efa3b95.jpg](../iclr_results/2744_DynAlign_%20Unsupervised%20Dynamic%20Taxonomy%20Alignment%20for%20Cross-Domain%20Segmentation/images/62f9fef71bc02ec954ae68be4af63721d767ef96ef8b76417dfc47e42efa3b95.jpg)

![b4c3b648180c5040ec954800a1334ea63601ee160c8a4ab1e28b185c249e0e25.jpg](../iclr_results/2744_DynAlign_%20Unsupervised%20Dynamic%20Taxonomy%20Alignment%20for%20Cross-Domain%20Segmentation/images/b4c3b648180c5040ec954800a1334ea63601ee160c8a4ab1e28b185c249e0e25.jpg)

![b6003f6d32172fd9fa105a4bf04aabe8be16e5b7b4983ca829a2c41fcfa974be.jpg](../iclr_results/2744_DynAlign_%20Unsupervised%20Dynamic%20Taxonomy%20Alignment%20for%20Cross-Domain%20Segmentation/images/b6003f6d32172fd9fa105a4bf04aabe8be16e5b7b4983ca829a2c41fcfa974be.jpg)

![beb043a87633a72f864b7edab6c559e2d7d725605ce0f66e0a3c7ad34cc40e31.jpg](../iclr_results/2744_DynAlign_%20Unsupervised%20Dynamic%20Taxonomy%20Alignment%20for%20Cross-Domain%20Segmentation/images/beb043a87633a72f864b7edab6c559e2d7d725605ce0f66e0a3c7ad34cc40e31.jpg)

![c7bcfa620a051999e561649b0113f330939d94fe31ec4b5f57c5c836cf22d329.jpg](../iclr_results/2744_DynAlign_%20Unsupervised%20Dynamic%20Taxonomy%20Alignment%20for%20Cross-Domain%20Segmentation/images/c7bcfa620a051999e561649b0113f330939d94fe31ec4b5f57c5c836cf22d329.jpg)

### Tables

![0484d5396516b6edbc505350a6b7590a5decef9b2020cba90c1112b6631b140f.jpg](../iclr_results/2744_DynAlign_%20Unsupervised%20Dynamic%20Taxonomy%20Alignment%20for%20Cross-Domain%20Segmentation/tables/0484d5396516b6edbc505350a6b7590a5decef9b2020cba90c1112b6631b140f.jpg)

![21dbac3bee2b6b6ea44da6acdb8d76117cd82c4e27b4989b7237f4a33d9c4f77.jpg](../iclr_results/2744_DynAlign_%20Unsupervised%20Dynamic%20Taxonomy%20Alignment%20for%20Cross-Domain%20Segmentation/tables/21dbac3bee2b6b6ea44da6acdb8d76117cd82c4e27b4989b7237f4a33d9c4f77.jpg)

![30d30aee75045b655bd022d65ade13b0b9270765a3655e71857c83b462e884e4.jpg](../iclr_results/2744_DynAlign_%20Unsupervised%20Dynamic%20Taxonomy%20Alignment%20for%20Cross-Domain%20Segmentation/tables/30d30aee75045b655bd022d65ade13b0b9270765a3655e71857c83b462e884e4.jpg)

![82ab3d88183d25f081f50aca45fa0a1be223cdccebab6c79beb34273a2f75475.jpg](../iclr_results/2744_DynAlign_%20Unsupervised%20Dynamic%20Taxonomy%20Alignment%20for%20Cross-Domain%20Segmentation/tables/82ab3d88183d25f081f50aca45fa0a1be223cdccebab6c79beb34273a2f75475.jpg)

![9ea8d38d29831fd904de3eeaa0024172c9b7d6178d66f1773c346eb759c9cb71.jpg](../iclr_results/2744_DynAlign_%20Unsupervised%20Dynamic%20Taxonomy%20Alignment%20for%20Cross-Domain%20Segmentation/tables/9ea8d38d29831fd904de3eeaa0024172c9b7d6178d66f1773c346eb759c9cb71.jpg)

![a2a7a395b18b1f070a22163db68a8c7fdba2225b7228d519980c1f8e7c800bf9.jpg](../iclr_results/2744_DynAlign_%20Unsupervised%20Dynamic%20Taxonomy%20Alignment%20for%20Cross-Domain%20Segmentation/tables/a2a7a395b18b1f070a22163db68a8c7fdba2225b7228d519980c1f8e7c800bf9.jpg)

![ab99210a384202c098eab0db91060e70859664f34340ea8c07b5212ab628df75.jpg](../iclr_results/2744_DynAlign_%20Unsupervised%20Dynamic%20Taxonomy%20Alignment%20for%20Cross-Domain%20Segmentation/tables/ab99210a384202c098eab0db91060e70859664f34340ea8c07b5212ab628df75.jpg)

![abde461a5eaa230a45958dc14631d319d0bf6b2e9e2d49bbb11477c6aa4ef04e.jpg](../iclr_results/2744_DynAlign_%20Unsupervised%20Dynamic%20Taxonomy%20Alignment%20for%20Cross-Domain%20Segmentation/tables/abde461a5eaa230a45958dc14631d319d0bf6b2e9e2d49bbb11477c6aa4ef04e.jpg)

![ade8b9a714b9ded06970376bbd163a36bd6250d1bf15cd37aa06682972ef3e89.jpg](../iclr_results/2744_DynAlign_%20Unsupervised%20Dynamic%20Taxonomy%20Alignment%20for%20Cross-Domain%20Segmentation/tables/ade8b9a714b9ded06970376bbd163a36bd6250d1bf15cd37aa06682972ef3e89.jpg)

![af9031b6200b77af9aad00dfe6e5ac46b18e2375e2ba1358f7c5efaaed906e49.jpg](../iclr_results/2744_DynAlign_%20Unsupervised%20Dynamic%20Taxonomy%20Alignment%20for%20Cross-Domain%20Segmentation/tables/af9031b6200b77af9aad00dfe6e5ac46b18e2375e2ba1358f7c5efaaed906e49.jpg)

![c424af82aa061e644145ad92a531f736b43a4c3323fbad22f372dd3136b3d188.jpg](../iclr_results/2744_DynAlign_%20Unsupervised%20Dynamic%20Taxonomy%20Alignment%20for%20Cross-Domain%20Segmentation/tables/c424af82aa061e644145ad92a531f736b43a4c3323fbad22f372dd3136b3d188.jpg)

![c5935933ab19d2bbd235d8de7852f72dc24c233b83a2511f84e2d140c0298b28.jpg](../iclr_results/2744_DynAlign_%20Unsupervised%20Dynamic%20Taxonomy%20Alignment%20for%20Cross-Domain%20Segmentation/tables/c5935933ab19d2bbd235d8de7852f72dc24c233b83a2511f84e2d140c0298b28.jpg)

![d3f16e01249e1c48f3f8343a4eb1d9c98d1207abd0f86fa56fc0dfe5f34d17c3.jpg](../iclr_results/2744_DynAlign_%20Unsupervised%20Dynamic%20Taxonomy%20Alignment%20for%20Cross-Domain%20Segmentation/tables/d3f16e01249e1c48f3f8343a4eb1d9c98d1207abd0f86fa56fc0dfe5f34d17c3.jpg)

![d46ffb1ee9ec02e9143723f165f17012ff7bd82abe52d54ba263df20ebf888de.jpg](../iclr_results/2744_DynAlign_%20Unsupervised%20Dynamic%20Taxonomy%20Alignment%20for%20Cross-Domain%20Segmentation/tables/d46ffb1ee9ec02e9143723f165f17012ff7bd82abe52d54ba263df20ebf888de.jpg)

![e0dac5aeec629860dee417e4856f5ff41863c37be0ce5d7b39a195ed7d29d3bc.jpg](../iclr_results/2744_DynAlign_%20Unsupervised%20Dynamic%20Taxonomy%20Alignment%20for%20Cross-Domain%20Segmentation/tables/e0dac5aeec629860dee417e4856f5ff41863c37be0ce5d7b39a195ed7d29d3bc.jpg)

## Optimizing 4D Gaussians for Dynamic Scene Video from Single Landscape Images


### Images

![1f1dde5fe4ef209602030229e0257d846306aa354dd2a2e6337e08fd729d1b40.jpg](../iclr_results/2745_Optimizing%204D%20Gaussians%20for%20Dynamic%20Scene%20Video%20from%20Single%20Landscape%20Images/images/1f1dde5fe4ef209602030229e0257d846306aa354dd2a2e6337e08fd729d1b40.jpg)

![32e1397e0bb4e66d229c7b093fea0dd8aa545c8c6113d515b6a0802f35b96ba8.jpg](../iclr_results/2745_Optimizing%204D%20Gaussians%20for%20Dynamic%20Scene%20Video%20from%20Single%20Landscape%20Images/images/32e1397e0bb4e66d229c7b093fea0dd8aa545c8c6113d515b6a0802f35b96ba8.jpg)

![567abeb14a7bc3febca2de6f37d6eb213de021287d52d2d13533417b897a1b6c.jpg](../iclr_results/2745_Optimizing%204D%20Gaussians%20for%20Dynamic%20Scene%20Video%20from%20Single%20Landscape%20Images/images/567abeb14a7bc3febca2de6f37d6eb213de021287d52d2d13533417b897a1b6c.jpg)

![5708e4310b756ea97c48c0d8a3f506f3ea158504b2b738977a2f1211df40ccfb.jpg](../iclr_results/2745_Optimizing%204D%20Gaussians%20for%20Dynamic%20Scene%20Video%20from%20Single%20Landscape%20Images/images/5708e4310b756ea97c48c0d8a3f506f3ea158504b2b738977a2f1211df40ccfb.jpg)

![699cd962dfb4bd68258138f732ac609e870e35548b54fb3755e811f3177c4a05.jpg](../iclr_results/2745_Optimizing%204D%20Gaussians%20for%20Dynamic%20Scene%20Video%20from%20Single%20Landscape%20Images/images/699cd962dfb4bd68258138f732ac609e870e35548b54fb3755e811f3177c4a05.jpg)

![6abc648370886c61c223eaa77daaa142a6cc1ae9d9de3ebe01f15a927a5e9aca.jpg](../iclr_results/2745_Optimizing%204D%20Gaussians%20for%20Dynamic%20Scene%20Video%20from%20Single%20Landscape%20Images/images/6abc648370886c61c223eaa77daaa142a6cc1ae9d9de3ebe01f15a927a5e9aca.jpg)

![7c21a3efb189c14dc8063954bcb65f99d8a665158f1aa464345fae6e048bbbeb.jpg](../iclr_results/2745_Optimizing%204D%20Gaussians%20for%20Dynamic%20Scene%20Video%20from%20Single%20Landscape%20Images/images/7c21a3efb189c14dc8063954bcb65f99d8a665158f1aa464345fae6e048bbbeb.jpg)

![8101c3a06627ddc9eea1b822ce503ccbe266562eb3919ab66e1c9a1e8e86e99c.jpg](../iclr_results/2745_Optimizing%204D%20Gaussians%20for%20Dynamic%20Scene%20Video%20from%20Single%20Landscape%20Images/images/8101c3a06627ddc9eea1b822ce503ccbe266562eb3919ab66e1c9a1e8e86e99c.jpg)

![86af501a883cac15d79fdf0033c8bfeb94a7a9604d8e877a358d0ef150c64969.jpg](../iclr_results/2745_Optimizing%204D%20Gaussians%20for%20Dynamic%20Scene%20Video%20from%20Single%20Landscape%20Images/images/86af501a883cac15d79fdf0033c8bfeb94a7a9604d8e877a358d0ef150c64969.jpg)

![9f1c64f5ed50a32733110b9eaba3c7c12e931c948578d017d38af1f2add3de68.jpg](../iclr_results/2745_Optimizing%204D%20Gaussians%20for%20Dynamic%20Scene%20Video%20from%20Single%20Landscape%20Images/images/9f1c64f5ed50a32733110b9eaba3c7c12e931c948578d017d38af1f2add3de68.jpg)

![9f81912f9aec4aac40afc1115254baba823fca931b9c94e89f255758b77f1be3.jpg](../iclr_results/2745_Optimizing%204D%20Gaussians%20for%20Dynamic%20Scene%20Video%20from%20Single%20Landscape%20Images/images/9f81912f9aec4aac40afc1115254baba823fca931b9c94e89f255758b77f1be3.jpg)

![a5bd208a4326211a4637e3093ce23c5a1dd6b9d41fa34fdee31fdba234504a27.jpg](../iclr_results/2745_Optimizing%204D%20Gaussians%20for%20Dynamic%20Scene%20Video%20from%20Single%20Landscape%20Images/images/a5bd208a4326211a4637e3093ce23c5a1dd6b9d41fa34fdee31fdba234504a27.jpg)

![b2c8f5ec527d4fc703ad08fe117a101b96be5fe77fca7b9b6c80842e77b5fab0.jpg](../iclr_results/2745_Optimizing%204D%20Gaussians%20for%20Dynamic%20Scene%20Video%20from%20Single%20Landscape%20Images/images/b2c8f5ec527d4fc703ad08fe117a101b96be5fe77fca7b9b6c80842e77b5fab0.jpg)

![eef582b236e4f9472bde006b7d37d8e4f49db449021127e9dbe293f3a6159d88.jpg](../iclr_results/2745_Optimizing%204D%20Gaussians%20for%20Dynamic%20Scene%20Video%20from%20Single%20Landscape%20Images/images/eef582b236e4f9472bde006b7d37d8e4f49db449021127e9dbe293f3a6159d88.jpg)

### Tables

![42681c09daf05bb490b1c441ab95c83cd13ef9608bc5bc610f2042884a987703.jpg](../iclr_results/2745_Optimizing%204D%20Gaussians%20for%20Dynamic%20Scene%20Video%20from%20Single%20Landscape%20Images/tables/42681c09daf05bb490b1c441ab95c83cd13ef9608bc5bc610f2042884a987703.jpg)

![e4650f21066108be55f6716782834d845b11010314ce2e69c7bded7d6991749f.jpg](../iclr_results/2745_Optimizing%204D%20Gaussians%20for%20Dynamic%20Scene%20Video%20from%20Single%20Landscape%20Images/tables/e4650f21066108be55f6716782834d845b11010314ce2e69c7bded7d6991749f.jpg)

## Improving Language Model Distillation through Hidden State Matching


### Images

![433e6ef6b2fc75acb641de89398bffa54880b3ae65cece28aede6e48291f7f5b.jpg](../iclr_results/2746_Improving%20Language%20Model%20Distillation%20through%20Hidden%20State%20Matching/images/433e6ef6b2fc75acb641de89398bffa54880b3ae65cece28aede6e48291f7f5b.jpg)

![a5223863792ff852e44627d43de3343e67531b91df1451fdf909f00f24bd75dd.jpg](../iclr_results/2746_Improving%20Language%20Model%20Distillation%20through%20Hidden%20State%20Matching/images/a5223863792ff852e44627d43de3343e67531b91df1451fdf909f00f24bd75dd.jpg)

![e89cf4d06b5d67a83d947ec63a5461e03b4405ae7dcb85adefc787a8909173cf.jpg](../iclr_results/2746_Improving%20Language%20Model%20Distillation%20through%20Hidden%20State%20Matching/images/e89cf4d06b5d67a83d947ec63a5461e03b4405ae7dcb85adefc787a8909173cf.jpg)

![ec1e32564c607a9e498e468122f54261aae0c9b76627093cfdf13f364c4f7eaf.jpg](../iclr_results/2746_Improving%20Language%20Model%20Distillation%20through%20Hidden%20State%20Matching/images/ec1e32564c607a9e498e468122f54261aae0c9b76627093cfdf13f364c4f7eaf.jpg)

### Tables

![07fe4e4935487177102ac2cab9347d640781a478be6d9ede404b13011e3c80fe.jpg](../iclr_results/2746_Improving%20Language%20Model%20Distillation%20through%20Hidden%20State%20Matching/tables/07fe4e4935487177102ac2cab9347d640781a478be6d9ede404b13011e3c80fe.jpg)

![277aeb520013da0a59de29924e39b04d5378573f46e2bee05551cbaf77ee0735.jpg](../iclr_results/2746_Improving%20Language%20Model%20Distillation%20through%20Hidden%20State%20Matching/tables/277aeb520013da0a59de29924e39b04d5378573f46e2bee05551cbaf77ee0735.jpg)

![9d178a05897c95204297c690dbca944d4d420c5f11ff528bfc787a0a290ab5b5.jpg](../iclr_results/2746_Improving%20Language%20Model%20Distillation%20through%20Hidden%20State%20Matching/tables/9d178a05897c95204297c690dbca944d4d420c5f11ff528bfc787a0a290ab5b5.jpg)

![bac795fa0be62f444a766a3009f378529b918c7bbbfd6e8ee4cd6ca954f57c39.jpg](../iclr_results/2746_Improving%20Language%20Model%20Distillation%20through%20Hidden%20State%20Matching/tables/bac795fa0be62f444a766a3009f378529b918c7bbbfd6e8ee4cd6ca954f57c39.jpg)

![bc8040ddb5d456f8ae20ad744eaa56f7ad7d64758aef93495971e1d6a14b1931.jpg](../iclr_results/2746_Improving%20Language%20Model%20Distillation%20through%20Hidden%20State%20Matching/tables/bc8040ddb5d456f8ae20ad744eaa56f7ad7d64758aef93495971e1d6a14b1931.jpg)

![e7ac6c51830daa330daa330d6d9d6af9830917a94065a76524d4d1c04f9192c1.jpg](../iclr_results/2746_Improving%20Language%20Model%20Distillation%20through%20Hidden%20State%20Matching/tables/e7ac6c51830daa330daa330d6d9d6af9830917a94065a76524d4d1c04f9192c1.jpg)

## Efficient Model-Based Reinforcement Learning Through Optimistic Thompson Sampling


### Images

![1654306ab6fcc41668d021d16d4b05759c43eca0a58554f7f0d7ae8c6441d150.jpg](../iclr_results/2747_Efficient%20Model-Based%20Reinforcement%20Learning%20Through%20Optimistic%20Thompson%20Sampling/images/1654306ab6fcc41668d021d16d4b05759c43eca0a58554f7f0d7ae8c6441d150.jpg)

![3ebdd87902c814fa19d2bfc76eb85222af738c4791788cbb6ab3cd8d0259937b.jpg](../iclr_results/2747_Efficient%20Model-Based%20Reinforcement%20Learning%20Through%20Optimistic%20Thompson%20Sampling/images/3ebdd87902c814fa19d2bfc76eb85222af738c4791788cbb6ab3cd8d0259937b.jpg)

![4e9276ecbac92856757dd5b33417cccd449be7d8cfde8feb998e375bb598b603.jpg](../iclr_results/2747_Efficient%20Model-Based%20Reinforcement%20Learning%20Through%20Optimistic%20Thompson%20Sampling/images/4e9276ecbac92856757dd5b33417cccd449be7d8cfde8feb998e375bb598b603.jpg)

![546fd09f6d2dfa61d5a3cd879b2fc5266d9bcc035a25a97d3fc28f8c5d00d4b2.jpg](../iclr_results/2747_Efficient%20Model-Based%20Reinforcement%20Learning%20Through%20Optimistic%20Thompson%20Sampling/images/546fd09f6d2dfa61d5a3cd879b2fc5266d9bcc035a25a97d3fc28f8c5d00d4b2.jpg)

![899d88dbecece0c76658a079dd24885bfc629abf0595f95a5d07279cb00f8744.jpg](../iclr_results/2747_Efficient%20Model-Based%20Reinforcement%20Learning%20Through%20Optimistic%20Thompson%20Sampling/images/899d88dbecece0c76658a079dd24885bfc629abf0595f95a5d07279cb00f8744.jpg)

![8c3df63f56525f6494c2c41728734bf146cdaedd5e19f967fb33992f4cd372c7.jpg](../iclr_results/2747_Efficient%20Model-Based%20Reinforcement%20Learning%20Through%20Optimistic%20Thompson%20Sampling/images/8c3df63f56525f6494c2c41728734bf146cdaedd5e19f967fb33992f4cd372c7.jpg)

![c08cf6613c84ba9df49609b3b16546f009761459a263dd8f53a864a676082d8b.jpg](../iclr_results/2747_Efficient%20Model-Based%20Reinforcement%20Learning%20Through%20Optimistic%20Thompson%20Sampling/images/c08cf6613c84ba9df49609b3b16546f009761459a263dd8f53a864a676082d8b.jpg)

### Tables

![ac41820d55c3b89f8f74266d2a82c3991a98f055f429f8510b4ed23a40b91cf3.jpg](../iclr_results/2747_Efficient%20Model-Based%20Reinforcement%20Learning%20Through%20Optimistic%20Thompson%20Sampling/tables/ac41820d55c3b89f8f74266d2a82c3991a98f055f429f8510b4ed23a40b91cf3.jpg)

![e38a0d34f8123646c598bd92e92c9c94f1aac7e9056ebc16df07fe62252216c3.jpg](../iclr_results/2747_Efficient%20Model-Based%20Reinforcement%20Learning%20Through%20Optimistic%20Thompson%20Sampling/tables/e38a0d34f8123646c598bd92e92c9c94f1aac7e9056ebc16df07fe62252216c3.jpg)

## What's New in My Data? Novelty Exploration via Contrastive Generation


### Images

![287423af1ed9de12d434f477513a7e0353c7bf899d8dc3ae1a700690ce93598d.jpg](../iclr_results/2748_What%27s%20New%20in%20My%20Data_%20Novelty%20Exploration%20via%20Contrastive%20Generation/images/287423af1ed9de12d434f477513a7e0353c7bf899d8dc3ae1a700690ce93598d.jpg)

![8b2299d62d0601b0be64e06139cb44d29e01497a82a4ee578b35e87fb9c07390.jpg](../iclr_results/2748_What%27s%20New%20in%20My%20Data_%20Novelty%20Exploration%20via%20Contrastive%20Generation/images/8b2299d62d0601b0be64e06139cb44d29e01497a82a4ee578b35e87fb9c07390.jpg)

![9bde22b9cbd498dd2bd4fe131bbc6d7d609975a06076e3a4773a4ca006b71197.jpg](../iclr_results/2748_What%27s%20New%20in%20My%20Data_%20Novelty%20Exploration%20via%20Contrastive%20Generation/images/9bde22b9cbd498dd2bd4fe131bbc6d7d609975a06076e3a4773a4ca006b71197.jpg)

### Tables

![0563938519ba9ecbebd5ffdd18219c7d808dd29493827ca1ccb91f08823c9f7d.jpg](../iclr_results/2748_What%27s%20New%20in%20My%20Data_%20Novelty%20Exploration%20via%20Contrastive%20Generation/tables/0563938519ba9ecbebd5ffdd18219c7d808dd29493827ca1ccb91f08823c9f7d.jpg)

![0cec43d68bdf5189fc0f3e7a77f5aff9213eef7e992cd957feebc7ad6459818d.jpg](../iclr_results/2748_What%27s%20New%20in%20My%20Data_%20Novelty%20Exploration%20via%20Contrastive%20Generation/tables/0cec43d68bdf5189fc0f3e7a77f5aff9213eef7e992cd957feebc7ad6459818d.jpg)

![191ffd5b8b61c092639de04d44517280f7c8cd5a47d82c7ddf5982f540e2f740.jpg](../iclr_results/2748_What%27s%20New%20in%20My%20Data_%20Novelty%20Exploration%20via%20Contrastive%20Generation/tables/191ffd5b8b61c092639de04d44517280f7c8cd5a47d82c7ddf5982f540e2f740.jpg)

![1eb193fbe558b76c98673b9c45ce31b53a6fef19b0dce99f1d135964d3228e77.jpg](../iclr_results/2748_What%27s%20New%20in%20My%20Data_%20Novelty%20Exploration%20via%20Contrastive%20Generation/tables/1eb193fbe558b76c98673b9c45ce31b53a6fef19b0dce99f1d135964d3228e77.jpg)

![79190639ff60f6a5868ba9ee93a8c5f290598a943b87da78e63d8945c970631e.jpg](../iclr_results/2748_What%27s%20New%20in%20My%20Data_%20Novelty%20Exploration%20via%20Contrastive%20Generation/tables/79190639ff60f6a5868ba9ee93a8c5f290598a943b87da78e63d8945c970631e.jpg)

![9b5e0d0a480df2776196dbf95a31bf1c4a59ee639b0d6105a31a1d81fac92192.jpg](../iclr_results/2748_What%27s%20New%20in%20My%20Data_%20Novelty%20Exploration%20via%20Contrastive%20Generation/tables/9b5e0d0a480df2776196dbf95a31bf1c4a59ee639b0d6105a31a1d81fac92192.jpg)

![bca060bf85aba859b7507417d0189d38e05302152bb7e32da0602d4fde6adbc0.jpg](../iclr_results/2748_What%27s%20New%20in%20My%20Data_%20Novelty%20Exploration%20via%20Contrastive%20Generation/tables/bca060bf85aba859b7507417d0189d38e05302152bb7e32da0602d4fde6adbc0.jpg)

![f2f5e2b1fd678630b48df0ffae9a496f47e746416a6f827b3fd49d8187a9b682.jpg](../iclr_results/2748_What%27s%20New%20in%20My%20Data_%20Novelty%20Exploration%20via%20Contrastive%20Generation/tables/f2f5e2b1fd678630b48df0ffae9a496f47e746416a6f827b3fd49d8187a9b682.jpg)

## Learning Color Equivariant Representations


### Images

![00868f2f4b0b1a8277d3bdddcc1c50f8d3f28746fb09a950daf4bb1c515aabad.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/images/00868f2f4b0b1a8277d3bdddcc1c50f8d3f28746fb09a950daf4bb1c515aabad.jpg)

![15b5c3a7ce65a2d9b71d6ca968299598321a0e61a10665682d42fb30eee41b79.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/images/15b5c3a7ce65a2d9b71d6ca968299598321a0e61a10665682d42fb30eee41b79.jpg)

![1cf39a827c53a2e21d418b96110be8e0870de4088428d584d1cbf943d465330a.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/images/1cf39a827c53a2e21d418b96110be8e0870de4088428d584d1cbf943d465330a.jpg)

![4d734f83b9df6abe0a85f6caa27a8de0460513e6e9d36c44134006ecea27ab86.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/images/4d734f83b9df6abe0a85f6caa27a8de0460513e6e9d36c44134006ecea27ab86.jpg)

![539e5f02a2f00545843092bb868869e2c0bcb3a5666427c22464e83fb5b16c80.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/images/539e5f02a2f00545843092bb868869e2c0bcb3a5666427c22464e83fb5b16c80.jpg)

![59c08dcd29a844ac160593d8ee87863e4b2aa03078e27cafe7244fca6e20343f.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/images/59c08dcd29a844ac160593d8ee87863e4b2aa03078e27cafe7244fca6e20343f.jpg)

![5dadeef78bb7cfb850be5189f2d953b3b945717656e04b842f2b5c9b79f6f341.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/images/5dadeef78bb7cfb850be5189f2d953b3b945717656e04b842f2b5c9b79f6f341.jpg)

![614bc3d7fb907de14cb834f7c2b8f34f483237a069f571feaa8a9ef1706ff6ce.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/images/614bc3d7fb907de14cb834f7c2b8f34f483237a069f571feaa8a9ef1706ff6ce.jpg)

![684d492ba5a130068f6491ebbfe7fe271fe42d75d857e4f68d31ad5ca46de2cd.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/images/684d492ba5a130068f6491ebbfe7fe271fe42d75d857e4f68d31ad5ca46de2cd.jpg)

![72e473e1ea9231d712b0773848e795a80f43f6928e89230b42aa801a3e9d6740.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/images/72e473e1ea9231d712b0773848e795a80f43f6928e89230b42aa801a3e9d6740.jpg)

![75275b339610dc1f052b5086fcba1118f75356c2a5d99d1ef36f4c99c5b66b21.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/images/75275b339610dc1f052b5086fcba1118f75356c2a5d99d1ef36f4c99c5b66b21.jpg)

![78d685142a5a97ad8630a3787b64e6626afecece8e7273a0fe4195a67f0002ce.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/images/78d685142a5a97ad8630a3787b64e6626afecece8e7273a0fe4195a67f0002ce.jpg)

![7d0480265d72f9e166e5d0c5c2fe0dfe503daf46a1bf2ffa106992ac7994ad21.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/images/7d0480265d72f9e166e5d0c5c2fe0dfe503daf46a1bf2ffa106992ac7994ad21.jpg)

![8c10de9a62e37928446a4a86737c225dcf9cbd1b7b02aa29da717dbc18d127b9.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/images/8c10de9a62e37928446a4a86737c225dcf9cbd1b7b02aa29da717dbc18d127b9.jpg)

![93437dcfd92aad8ad0827293eb7c71d16478ec2d7dd3c215bdc4fd2238b29ead.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/images/93437dcfd92aad8ad0827293eb7c71d16478ec2d7dd3c215bdc4fd2238b29ead.jpg)

![a2b871ce24c8fe7d356aab465aeea8be0910657144dab2ff63f6d6ab43326cc3.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/images/a2b871ce24c8fe7d356aab465aeea8be0910657144dab2ff63f6d6ab43326cc3.jpg)

![a715a5b56d93465a35df5588c70741e0edd4ac425c716dde48cb3546b875656a.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/images/a715a5b56d93465a35df5588c70741e0edd4ac425c716dde48cb3546b875656a.jpg)

![b4bd3a6f99a1553c38718e3aa146ea55fcc5604cd4286affe1190a738729e981.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/images/b4bd3a6f99a1553c38718e3aa146ea55fcc5604cd4286affe1190a738729e981.jpg)

![c25082fe9ca41179edd2c59d270171d7b5d3e653dfac76389ce577cc5df88910.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/images/c25082fe9ca41179edd2c59d270171d7b5d3e653dfac76389ce577cc5df88910.jpg)

![e364fca7e126bbb79d116624aaa77b87fa6b5921a86685c458905b573090ee34.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/images/e364fca7e126bbb79d116624aaa77b87fa6b5921a86685c458905b573090ee34.jpg)

### Tables

![0bf547186b3a576bb1ee77d8eb86386b24064f93b19e1503f8e4e991c38d82f7.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/tables/0bf547186b3a576bb1ee77d8eb86386b24064f93b19e1503f8e4e991c38d82f7.jpg)

![0f25f64b1ff757ea1e1fcdd559f3d0f7d6a1c1234609abac60f8f098e02b632a.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/tables/0f25f64b1ff757ea1e1fcdd559f3d0f7d6a1c1234609abac60f8f098e02b632a.jpg)

![2a0342cb595f41536c1ae0ac9f732c7ddf88f575ef0fc5fdeb218e33d23d5fb9.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/tables/2a0342cb595f41536c1ae0ac9f732c7ddf88f575ef0fc5fdeb218e33d23d5fb9.jpg)

![395a19ffdcf6268a5eeddc45dc67adfbd028d85fa74fb28a13aa6678710ab1c5.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/tables/395a19ffdcf6268a5eeddc45dc67adfbd028d85fa74fb28a13aa6678710ab1c5.jpg)

![3dc18e27ca986d79c51b965982338b32504849d5c4a5ad7917158586bab06776.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/tables/3dc18e27ca986d79c51b965982338b32504849d5c4a5ad7917158586bab06776.jpg)

![ac476780346e17f50aecbc536bd0ee33d4e819e7b5320edbc30acc611a9409d4.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/tables/ac476780346e17f50aecbc536bd0ee33d4e819e7b5320edbc30acc611a9409d4.jpg)

![d9d49f5795c667d6b649a90ca2d7cd887a3683d62c05cab55a2930c18a54b93f.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/tables/d9d49f5795c667d6b649a90ca2d7cd887a3683d62c05cab55a2930c18a54b93f.jpg)

![f847f98ac66dbd684d4d878cb68f0b25c6e2881f6be2403cd011a1abad303a4c.jpg](../iclr_results/2749_Learning%20Color%20Equivariant%20Representations/tables/f847f98ac66dbd684d4d878cb68f0b25c6e2881f6be2403cd011a1abad303a4c.jpg)

## LeanQuant: Accurate and Scalable Large Language Model Quantization with Loss-error-aware Grid


### Images

![02a45084a8c5972ad96a39cfe749792c578977814dee39854cb3c7d9f41d5320.jpg](../iclr_results/2750_LeanQuant_%20Accurate%20and%20Scalable%20Large%20Language%20Model%20Quantization%20with%20Loss-error-aware%20Grid/images/02a45084a8c5972ad96a39cfe749792c578977814dee39854cb3c7d9f41d5320.jpg)

![10c02fabf5ac09239513f11538114098d2651cc97fa9fcda5cfdb5ec85f6bed0.jpg](../iclr_results/2750_LeanQuant_%20Accurate%20and%20Scalable%20Large%20Language%20Model%20Quantization%20with%20Loss-error-aware%20Grid/images/10c02fabf5ac09239513f11538114098d2651cc97fa9fcda5cfdb5ec85f6bed0.jpg)

![2624bcd5559b5169df6da58c1f1066ef95064d9375b06261a16ad3513ca75b82.jpg](../iclr_results/2750_LeanQuant_%20Accurate%20and%20Scalable%20Large%20Language%20Model%20Quantization%20with%20Loss-error-aware%20Grid/images/2624bcd5559b5169df6da58c1f1066ef95064d9375b06261a16ad3513ca75b82.jpg)

![37b83930e007f88050e70319093aa6c954659c8c8e097ff9f67f109a5de068d3.jpg](../iclr_results/2750_LeanQuant_%20Accurate%20and%20Scalable%20Large%20Language%20Model%20Quantization%20with%20Loss-error-aware%20Grid/images/37b83930e007f88050e70319093aa6c954659c8c8e097ff9f67f109a5de068d3.jpg)

![e8871ceb6b7e6100cfc201143c0aabd6ff16d8da99cd918a6a33a907e53931d0.jpg](../iclr_results/2750_LeanQuant_%20Accurate%20and%20Scalable%20Large%20Language%20Model%20Quantization%20with%20Loss-error-aware%20Grid/images/e8871ceb6b7e6100cfc201143c0aabd6ff16d8da99cd918a6a33a907e53931d0.jpg)

### Tables

![3ea490098b6332fd8d672a6396f55e184a681911908fabcb3642d57b6d3218ae.jpg](../iclr_results/2750_LeanQuant_%20Accurate%20and%20Scalable%20Large%20Language%20Model%20Quantization%20with%20Loss-error-aware%20Grid/tables/3ea490098b6332fd8d672a6396f55e184a681911908fabcb3642d57b6d3218ae.jpg)

![4350e87bb74acc78d53b2b72c760a13ae1adc9375e775ece9daccde09e5974c6.jpg](../iclr_results/2750_LeanQuant_%20Accurate%20and%20Scalable%20Large%20Language%20Model%20Quantization%20with%20Loss-error-aware%20Grid/tables/4350e87bb74acc78d53b2b72c760a13ae1adc9375e775ece9daccde09e5974c6.jpg)

![458e727676e603524a070319b1a8272ffe68817ba3a7b20b36c9b69a54520b6f.jpg](../iclr_results/2750_LeanQuant_%20Accurate%20and%20Scalable%20Large%20Language%20Model%20Quantization%20with%20Loss-error-aware%20Grid/tables/458e727676e603524a070319b1a8272ffe68817ba3a7b20b36c9b69a54520b6f.jpg)

![4bd23cff845ee003fe3927ecca2fe356f54368dcd475b51ec9509eb77d17297e.jpg](../iclr_results/2750_LeanQuant_%20Accurate%20and%20Scalable%20Large%20Language%20Model%20Quantization%20with%20Loss-error-aware%20Grid/tables/4bd23cff845ee003fe3927ecca2fe356f54368dcd475b51ec9509eb77d17297e.jpg)

![4c34dee169a61184bb5cdd793523591e24656dcff97ef8a792b4bfd60ffdbeed.jpg](../iclr_results/2750_LeanQuant_%20Accurate%20and%20Scalable%20Large%20Language%20Model%20Quantization%20with%20Loss-error-aware%20Grid/tables/4c34dee169a61184bb5cdd793523591e24656dcff97ef8a792b4bfd60ffdbeed.jpg)

![6768d310de213dd982759c40cb2d9ac71da8229a7d0ad7352cea3f46319eaba0.jpg](../iclr_results/2750_LeanQuant_%20Accurate%20and%20Scalable%20Large%20Language%20Model%20Quantization%20with%20Loss-error-aware%20Grid/tables/6768d310de213dd982759c40cb2d9ac71da8229a7d0ad7352cea3f46319eaba0.jpg)

![6ac51006ff1955f620f2e900ccb7b66ee34f268c141f5258cff59c9a86934b67.jpg](../iclr_results/2750_LeanQuant_%20Accurate%20and%20Scalable%20Large%20Language%20Model%20Quantization%20with%20Loss-error-aware%20Grid/tables/6ac51006ff1955f620f2e900ccb7b66ee34f268c141f5258cff59c9a86934b67.jpg)

![8a98ac49a3fdec74cbc5e47d5b0a92580d4b89f3cb679c1f0613fc07dfa6c171.jpg](../iclr_results/2750_LeanQuant_%20Accurate%20and%20Scalable%20Large%20Language%20Model%20Quantization%20with%20Loss-error-aware%20Grid/tables/8a98ac49a3fdec74cbc5e47d5b0a92580d4b89f3cb679c1f0613fc07dfa6c171.jpg)

![94677608effdba8f07d29dcfe11d697deb654cf9b40d832cd98a446baaaff60d.jpg](../iclr_results/2750_LeanQuant_%20Accurate%20and%20Scalable%20Large%20Language%20Model%20Quantization%20with%20Loss-error-aware%20Grid/tables/94677608effdba8f07d29dcfe11d697deb654cf9b40d832cd98a446baaaff60d.jpg)

![9ada469334837fdd7b3c6ff3da4d16fa8d50501b77b1a4024f64f8afc7d22c34.jpg](../iclr_results/2750_LeanQuant_%20Accurate%20and%20Scalable%20Large%20Language%20Model%20Quantization%20with%20Loss-error-aware%20Grid/tables/9ada469334837fdd7b3c6ff3da4d16fa8d50501b77b1a4024f64f8afc7d22c34.jpg)

![a2ad385cf136ca17efd1bfac9cd0ae4927242e86084fbcc93ae0b0b67f4e716a.jpg](../iclr_results/2750_LeanQuant_%20Accurate%20and%20Scalable%20Large%20Language%20Model%20Quantization%20with%20Loss-error-aware%20Grid/tables/a2ad385cf136ca17efd1bfac9cd0ae4927242e86084fbcc93ae0b0b67f4e716a.jpg)

![bb3b5bf7614669b0c618d2292a9a9366f8f402348d46ec897d744888a8b75569.jpg](../iclr_results/2750_LeanQuant_%20Accurate%20and%20Scalable%20Large%20Language%20Model%20Quantization%20with%20Loss-error-aware%20Grid/tables/bb3b5bf7614669b0c618d2292a9a9366f8f402348d46ec897d744888a8b75569.jpg)

![c3343b129af62c225739c19878a62cad6104d24d5d77a42d1cba301d31e60408.jpg](../iclr_results/2750_LeanQuant_%20Accurate%20and%20Scalable%20Large%20Language%20Model%20Quantization%20with%20Loss-error-aware%20Grid/tables/c3343b129af62c225739c19878a62cad6104d24d5d77a42d1cba301d31e60408.jpg)

![c3c76f6fa931c4174423547c3505f5b82b045d6a3a6db0a5d05d370bf13571f6.jpg](../iclr_results/2750_LeanQuant_%20Accurate%20and%20Scalable%20Large%20Language%20Model%20Quantization%20with%20Loss-error-aware%20Grid/tables/c3c76f6fa931c4174423547c3505f5b82b045d6a3a6db0a5d05d370bf13571f6.jpg)

![d031162bdf3ff341d9d296b49773c90861662a762cfa02b990e102e583165b11.jpg](../iclr_results/2750_LeanQuant_%20Accurate%20and%20Scalable%20Large%20Language%20Model%20Quantization%20with%20Loss-error-aware%20Grid/tables/d031162bdf3ff341d9d296b49773c90861662a762cfa02b990e102e583165b11.jpg)

![e9391db4623cef585854ca1943c164dc0681e94b168a0ea4c1505e9cf16d70ff.jpg](../iclr_results/2750_LeanQuant_%20Accurate%20and%20Scalable%20Large%20Language%20Model%20Quantization%20with%20Loss-error-aware%20Grid/tables/e9391db4623cef585854ca1943c164dc0681e94b168a0ea4c1505e9cf16d70ff.jpg)

## Generalization v.s. Memorization: Tracing Language Models’ Capabilities Back to Pretraining Data

### Images

![2ba98663aeeee1de756ec1c9183349bafcce2090a7d09491fa9ae0a58e54ada3.jpg](../iclr_results/2751_Generalization%20v.s.%20Memorization_%20Tracing%20Language%20Models%E2%80%99%20Capabilities%20Back%20to%20Pretraining%20Data/images/2ba98663aeeee1de756ec1c9183349bafcce2090a7d09491fa9ae0a58e54ada3.jpg)

![2ec6bd20ac5ce254624c152af416c8ad9d641639f7af9e688e2dd70e80102992.jpg](../iclr_results/2751_Generalization%20v.s.%20Memorization_%20Tracing%20Language%20Models%E2%80%99%20Capabilities%20Back%20to%20Pretraining%20Data/images/2ec6bd20ac5ce254624c152af416c8ad9d641639f7af9e688e2dd70e80102992.jpg)

![480fcaa0b7a50f6cdb1bc34692088dc79acd8d27f709971ac6b14bb34b7afd89.jpg](../iclr_results/2751_Generalization%20v.s.%20Memorization_%20Tracing%20Language%20Models%E2%80%99%20Capabilities%20Back%20to%20Pretraining%20Data/images/480fcaa0b7a50f6cdb1bc34692088dc79acd8d27f709971ac6b14bb34b7afd89.jpg)

![8bc7c0347c9cbdeb38c6492035a902baf21e300ba56fe976dd5a04985d761c1f.jpg](../iclr_results/2751_Generalization%20v.s.%20Memorization_%20Tracing%20Language%20Models%E2%80%99%20Capabilities%20Back%20to%20Pretraining%20Data/images/8bc7c0347c9cbdeb38c6492035a902baf21e300ba56fe976dd5a04985d761c1f.jpg)

![94cdbe6f4ff27e58d3b1935d1402bd7ee0c7aad8ee218a5efecf7565e8cd7c56.jpg](../iclr_results/2751_Generalization%20v.s.%20Memorization_%20Tracing%20Language%20Models%E2%80%99%20Capabilities%20Back%20to%20Pretraining%20Data/images/94cdbe6f4ff27e58d3b1935d1402bd7ee0c7aad8ee218a5efecf7565e8cd7c56.jpg)

![a677bd91c45c3c0476b76bce99f4cb345284b35b03a24f6e3225ea72f3fde935.jpg](../iclr_results/2751_Generalization%20v.s.%20Memorization_%20Tracing%20Language%20Models%E2%80%99%20Capabilities%20Back%20to%20Pretraining%20Data/images/a677bd91c45c3c0476b76bce99f4cb345284b35b03a24f6e3225ea72f3fde935.jpg)

![c539decdef42c92474f230ce0d38876e16387eca0408d6f3cacf7a0623f35aa8.jpg](../iclr_results/2751_Generalization%20v.s.%20Memorization_%20Tracing%20Language%20Models%E2%80%99%20Capabilities%20Back%20to%20Pretraining%20Data/images/c539decdef42c92474f230ce0d38876e16387eca0408d6f3cacf7a0623f35aa8.jpg)

![d1adadb3304a6c1c0cb779c992672f21fb9c254608ef3fa8c1fba35acb33c845.jpg](../iclr_results/2751_Generalization%20v.s.%20Memorization_%20Tracing%20Language%20Models%E2%80%99%20Capabilities%20Back%20to%20Pretraining%20Data/images/d1adadb3304a6c1c0cb779c992672f21fb9c254608ef3fa8c1fba35acb33c845.jpg)

### Tables

![13214cda1cb5a4fb38f0f3b6809bbda23daf728019efe7bbf58c7a30390aa493.jpg](../iclr_results/2751_Generalization%20v.s.%20Memorization_%20Tracing%20Language%20Models%E2%80%99%20Capabilities%20Back%20to%20Pretraining%20Data/tables/13214cda1cb5a4fb38f0f3b6809bbda23daf728019efe7bbf58c7a30390aa493.jpg)

![14df88178fe16ed5e1abd4cd7aa0006f48f9e51e72899068f68b443e863eceb7.jpg](../iclr_results/2751_Generalization%20v.s.%20Memorization_%20Tracing%20Language%20Models%E2%80%99%20Capabilities%20Back%20to%20Pretraining%20Data/tables/14df88178fe16ed5e1abd4cd7aa0006f48f9e51e72899068f68b443e863eceb7.jpg)

![409f708d1562401a6529db4275ff21060c8b6c8301790e663cd2544052b51ca0.jpg](../iclr_results/2751_Generalization%20v.s.%20Memorization_%20Tracing%20Language%20Models%E2%80%99%20Capabilities%20Back%20to%20Pretraining%20Data/tables/409f708d1562401a6529db4275ff21060c8b6c8301790e663cd2544052b51ca0.jpg)

![58c7887d908519837a8c546c941a9100dd1874fa64838fc5e8d2003a5c9b0b87.jpg](../iclr_results/2751_Generalization%20v.s.%20Memorization_%20Tracing%20Language%20Models%E2%80%99%20Capabilities%20Back%20to%20Pretraining%20Data/tables/58c7887d908519837a8c546c941a9100dd1874fa64838fc5e8d2003a5c9b0b87.jpg)

![76e83b6799fc886a71c857fad258dd98cb2051c628add4d21e0b0b3e0b892974.jpg](../iclr_results/2751_Generalization%20v.s.%20Memorization_%20Tracing%20Language%20Models%E2%80%99%20Capabilities%20Back%20to%20Pretraining%20Data/tables/76e83b6799fc886a71c857fad258dd98cb2051c628add4d21e0b0b3e0b892974.jpg)

![d454ed21520ee604a0dde6edad045b4a36cc48d83bb3e536c483f95cd838b0fc.jpg](../iclr_results/2751_Generalization%20v.s.%20Memorization_%20Tracing%20Language%20Models%E2%80%99%20Capabilities%20Back%20to%20Pretraining%20Data/tables/d454ed21520ee604a0dde6edad045b4a36cc48d83bb3e536c483f95cd838b0fc.jpg)

![f58c4e17d2eede1d92f71a2c5201e5d026a18b4b149d217abad0939700fcb9b6.jpg](../iclr_results/2751_Generalization%20v.s.%20Memorization_%20Tracing%20Language%20Models%E2%80%99%20Capabilities%20Back%20to%20Pretraining%20Data/tables/f58c4e17d2eede1d92f71a2c5201e5d026a18b4b149d217abad0939700fcb9b6.jpg)
