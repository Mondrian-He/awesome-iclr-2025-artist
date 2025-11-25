# ICLR 2025 Main Conference Papers

**Summary:** 36 papers with extracted content:
- 📊 Total images: 44031
- 📋 Total tables: 33468
- 📄 Total files: 77499

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 57 of 100

## 目录 (Table of Contents)

1. [Differentially Private Federated Learning with Time-Adaptive Privacy Spending](#Differentially-Private-Federated-Learning-with-Time-Adaptive-Privacy-Spending)
2. [TimeInf: Time Series Data Contribution via Influence Functions](#TimeInf-Time-Series-Data-Contribution-via-Influence-Functions)
3. [Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model](#Solving-Token-Gradient-Conflict-in-Mixture-of-Experts-for-Large-Vision-Language-Model)
4. [HaDeMiF: Hallucination Detection and Mitigation in Large Language Models](#HaDeMiF-Hallucination-Detection-and-Mitigation-in-Large-Language-Models)
5. [Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Planning Agent](#Benchmarking-Multimodal-Retrieval-Augmented-Generation-with-Dynamic-VQA-Dataset-and-Self-adaptive-Planning-Agent)
6. [Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving](#Navigation-Guided-Sparse-Scene-Representation-for-End-to-End-Autonomous-Driving)
7. [OSCAR: Operating System Control via State-Aware Reasoning and Re-Planning](#OSCAR-Operating-System-Control-via-State-Aware-Reasoning-and-Re-Planning)
8. [GoodDrag: Towards Good Practices for Drag Editing with Diffusion Models](#GoodDrag-Towards-Good-Practices-for-Drag-Editing-with-Diffusion-Models)
9. [Fengbo: a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics](#Fengbo-a-Clifford-Neural-Operator-pipeline-for-3D-PDEs-in-Computational-Fluid-Dynamics)
10. [Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data](#Learning-Interpretable-Hierarchical-Dynamical-Systems-Models-from-Time-Series-Data)
11. [NeSyC: A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains](#NeSyC-A-Neuro-symbolic-Continual-Learner-For-Complex-Embodied-Tasks-In-Open-Domains)
12. [An Effective Theory of Bias Amplification](#An-Effective-Theory-of-Bias-Amplification)
13. [K-HALU: Multiple Answer Korean Hallucination Benchmark for Large Language Models](#K-HALU-Multiple-Answer-Korean-Hallucination-Benchmark-for-Large-Language-Models)
14. [Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning](#Frequency-Guided-Masking-for-Enhanced-Vision-Self-Supervised-Learning)
15. [Correlation and Navigation in the Vocabulary Key Representation Space of Language Models](#Correlation-and-Navigation-in-the-Vocabulary-Key-Representation-Space-of-Language-Models)
16. [CREAM: Consistency Regularized Self-Rewarding Language Models](#CREAM-Consistency-Regularized-Self-Rewarding-Language-Models)
17. [Universal Sharpness Dynamics in Neural Network Training: Fixed Point Analysis, Edge of Stability, and Route to Chaos](#Universal-Sharpness-Dynamics-in-Neural-Network-Training-Fixed-Point-Analysis-Edge-of-Stability-and-Route-to-Chaos)
18. [ObscuraCoder: Powering Efficient Code LM Pre-Training Via Obfuscation Grounding](#ObscuraCoder-Powering-Efficient-Code-LM-Pre-Training-Via-Obfuscation-Grounding)
19. [MrT5: Dynamic Token Merging for Efficient Byte-level Language Models](#MrT5-Dynamic-Token-Merging-for-Efficient-Byte-level-Language-Models)
20. [Latent Action Pretraining from Videos](#Latent-Action-Pretraining-from-Videos)
21. [Cross the Gap:  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion](#Cross-the-Gap-Exposing-the-Intra-modal-Misalignment-in-CLIP-via-Modality-Inversion)
22. [Transformer Encoder Satisfiability: Complexity and Impact on Formal Reasoning](#Transformer-Encoder-Satisfiability-Complexity-and-Impact-on-Formal-Reasoning)
23. [CARTS: Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree Search](#CARTS-Advancing-Neural-Theorem-Proving-with-Diversified-Tactic-Calibration-and-Bias-Resistant-Tree-Search)
24. [Bayesian Regularization of Latent Representation](#Bayesian-Regularization-of-Latent-Representation)
25. [DynaMath: A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Language Models](#DynaMath-A-Dynamic-Visual-Benchmark-for-Evaluating-Mathematical-Reasoning-Robustness-of-Vision-Language-Models)
26. [Convergence of Distributed Adaptive Optimization with Local Updates](#Convergence-of-Distributed-Adaptive-Optimization-with-Local-Updates)
27. [Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving](#Inference-Scaling-Laws-An-Empirical-Analysis-of-Compute-Optimal-Inference-for-LLM-Problem-Solving)
28. [Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks](#Activation-Gradient-based-Poisoned-Sample-Detection-Against-Backdoor-Attacks)
29. [Designing Mechanical Meta-Materials by Learning Equivariant Flows](#Designing-Mechanical-Meta-Materials-by-Learning-Equivariant-Flows)
30. [MCNC: Manifold-Constrained Reparameterization for Neural Compression](#MCNC-Manifold-Constrained-Reparameterization-for-Neural-Compression)
31. [TypedThinker: Diversify Large Language Model Reasoning with Typed Thinking](#TypedThinker-Diversify-Large-Language-Model-Reasoning-with-Typed-Thinking)
32. [SEAL: Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection](#SEAL-Safety-enhanced-Aligned-LLM-Fine-tuning-via-Bilevel-Data-Selection)
33. [Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport](#Accelerating-3D-Molecule-Generation-via-Jointly-Geometric-Optimal-Transport)
34. [Neural Dueling Bandits: Preference-Based Optimization with Human Feedback](#Neural-Dueling-Bandits-Preference-Based-Optimization-with-Human-Feedback)
35. [Divergence of Neural Tangent Kernel in Classification Problems](#Divergence-of-Neural-Tangent-Kernel-in-Classification-Problems)
36. [Regret Bounds for Episodic Risk-Sensitive Linear Quadratic Regulator](#Regret-Bounds-for-Episodic-Risk-Sensitive-Linear-Quadratic-Regulator)

---


## Differentially Private Federated Learning with Time-Adaptive Privacy Spending

### Images

![045c1e5ed2c9d5ce862a83f83391fc469c6495e463a5961fba7d84392e4d3cba.jpg](../iclr_results/2090_Forget the Data and Fine-Tuning! Just Fold the Network to Compress/images/045c1e5ed2c9d5ce862a83f83391fc469c6495e463a5961fba7d84392e4d3cba.jpg)

![154da777c5fcafde90a84560f6b5047504f3826d9b40bba5ee19288730895c8a.jpg](../iclr_results/2090_Forget the Data and Fine-Tuning! Just Fold the Network to Compress/images/154da777c5fcafde90a84560f6b5047504f3826d9b40bba5ee19288730895c8a.jpg)

![2b5529043d30671d94a395b62df7d29820b931c4878d292899ddf8bf5f75ad20.jpg](../iclr_results/2090_Forget the Data and Fine-Tuning! Just Fold the Network to Compress/images/2b5529043d30671d94a395b62df7d29820b931c4878d292899ddf8bf5f75ad20.jpg)

![34d68b22a379d9e31b92ef1c6390413419bee9a149371c6bb4c3a3b85310d4cd.jpg](../iclr_results/2090_Forget the Data and Fine-Tuning! Just Fold the Network to Compress/images/34d68b22a379d9e31b92ef1c6390413419bee9a149371c6bb4c3a3b85310d4cd.jpg)

![38e676c2cecf88b83628b80abf3450ba8e91ed11238245bdf6d20d9c832788b2.jpg](../iclr_results/2090_Forget the Data and Fine-Tuning! Just Fold the Network to Compress/images/38e676c2cecf88b83628b80abf3450ba8e91ed11238245bdf6d20d9c832788b2.jpg)

![52673d38702234aeb7e9fa2ffa74ad9b0df69414dd59636953a74e9036312e6c.jpg](../iclr_results/2090_Forget the Data and Fine-Tuning! Just Fold the Network to Compress/images/52673d38702234aeb7e9fa2ffa74ad9b0df69414dd59636953a74e9036312e6c.jpg)

![5e7ab9be8d46bb796968440cab2b01d676415f0ea08cd66efdd7252010e26503.jpg](../iclr_results/2090_Forget the Data and Fine-Tuning! Just Fold the Network to Compress/images/5e7ab9be8d46bb796968440cab2b01d676415f0ea08cd66efdd7252010e26503.jpg)

![661c310015a8503df36eb7db04234012361c09e6a7b734bf02c144ec5939c076.jpg](../iclr_results/2090_Forget the Data and Fine-Tuning! Just Fold the Network to Compress/images/661c310015a8503df36eb7db04234012361c09e6a7b734bf02c144ec5939c076.jpg)

![a183f10f92bdbc5d4e508677168b2f1879984a9acdbc0ccee2d135c5fdc0f2ec.jpg](../iclr_results/2090_Forget the Data and Fine-Tuning! Just Fold the Network to Compress/images/a183f10f92bdbc5d4e508677168b2f1879984a9acdbc0ccee2d135c5fdc0f2ec.jpg)

![bb838d7fa8feca4886f80545fd5db41d74f64a7dbb6cfb65159059d97292b737.jpg](../iclr_results/2090_Forget the Data and Fine-Tuning! Just Fold the Network to Compress/images/bb838d7fa8feca4886f80545fd5db41d74f64a7dbb6cfb65159059d97292b737.jpg)

![dc99f5bbeb358fe41f7cba9e5fb497d205ac3426e48959e95aad7daa85d483cb.jpg](../iclr_results/2090_Forget the Data and Fine-Tuning! Just Fold the Network to Compress/images/dc99f5bbeb358fe41f7cba9e5fb497d205ac3426e48959e95aad7daa85d483cb.jpg)

![de8d5f7eccd1b7ce47c44b15cc02961d12f9e133d52691367393d7a5588cf131.jpg](../iclr_results/2090_Forget the Data and Fine-Tuning! Just Fold the Network to Compress/images/de8d5f7eccd1b7ce47c44b15cc02961d12f9e133d52691367393d7a5588cf131.jpg)

![e3e076b8ea6dc3042410b44e06d2ddecee9993089df0128329563042996e6e49.jpg](../iclr_results/2090_Forget the Data and Fine-Tuning! Just Fold the Network to Compress/images/e3e076b8ea6dc3042410b44e06d2ddecee9993089df0128329563042996e6e49.jpg)

### Tables

![00afafb636045f8b8ffb04155b77323d82dd37081e31bf85c989a93329cfa4ef.jpg](../iclr_results/2090_Forget the Data and Fine-Tuning! Just Fold the Network to Compress/tables/00afafb636045f8b8ffb04155b77323d82dd37081e31bf85c989a93329cfa4ef.jpg)

![08ae35ccefc322562d42e280bb4f0f73b588483ed5be90be05e1c3d3b3a463a5.jpg](../iclr_results/2090_Forget the Data and Fine-Tuning! Just Fold the Network to Compress/tables/08ae35ccefc322562d42e280bb4f0f73b588483ed5be90be05e1c3d3b3a463a5.jpg)

![513acf300953d9245799baab222326d17d7204135b1abe7eee74844e3fd1dffb.jpg](../iclr_results/2090_Forget the Data and Fine-Tuning! Just Fold the Network to Compress/tables/513acf300953d9245799baab222326d17d7204135b1abe7eee74844e3fd1dffb.jpg)

![8e9f3cf8a502c1a5513a2cde1c844c01603fa1e732e63d800f25945aa72b7213.jpg](../iclr_results/2090_Forget the Data and Fine-Tuning! Just Fold the Network to Compress/tables/8e9f3cf8a502c1a5513a2cde1c844c01603fa1e732e63d800f25945aa72b7213.jpg)

![b86c902db242daa99aebedeb527f9befa7512bbb5b943df7710836831fc2fc42.jpg](../iclr_results/2090_Forget the Data and Fine-Tuning! Just Fold the Network to Compress/tables/b86c902db242daa99aebedeb527f9befa7512bbb5b943df7710836831fc2fc42.jpg)

![e6aff69f615986008649a9a90917dcf9885b68d90a4982dc411993fd87a98c40.jpg](../iclr_results/2090_Forget the Data and Fine-Tuning! Just Fold the Network to Compress/tables/e6aff69f615986008649a9a90917dcf9885b68d90a4982dc411993fd87a98c40.jpg)

![e9c235546f2f6011c4fbf901b31820ca9e3545611b122156e43417e60ec3b096.jpg](../iclr_results/2090_Forget the Data and Fine-Tuning! Just Fold the Network to Compress/tables/e9c235546f2f6011c4fbf901b31820ca9e3545611b122156e43417e60ec3b096.jpg)

## Differentially Private Federated Learning with Time-Adaptive Privacy Spending


### Images

![08f9b93e4339961869400694d640551f5dc129b31323c1408081c1a8c1a39116.jpg](../iclr_results/2091_Differentially Private Federated Learning with Time-Adaptive Privacy Spending/images/08f9b93e4339961869400694d640551f5dc129b31323c1408081c1a8c1a39116.jpg)

![4eda14aeb7ec289434c1e3a1330f01cabe8ceb52c31d41a076ff54f87c4a9a30.jpg](../iclr_results/2091_Differentially Private Federated Learning with Time-Adaptive Privacy Spending/images/4eda14aeb7ec289434c1e3a1330f01cabe8ceb52c31d41a076ff54f87c4a9a30.jpg)

![c27308f3c44ac94e6dbd043ad16ff8bb2bd7330cd2376a5d4062e6e31f4334f9.jpg](../iclr_results/2091_Differentially Private Federated Learning with Time-Adaptive Privacy Spending/images/c27308f3c44ac94e6dbd043ad16ff8bb2bd7330cd2376a5d4062e6e31f4334f9.jpg)

![e4e983ae23883a17712a8d30327fef1697ac80999e78595cf5e12580b9e5f023.jpg](../iclr_results/2091_Differentially Private Federated Learning with Time-Adaptive Privacy Spending/images/e4e983ae23883a17712a8d30327fef1697ac80999e78595cf5e12580b9e5f023.jpg)

![f015d9cfb38cce8a5e26bde21bcbf0133e7d5f797bcb145774152267c0358c4d.jpg](../iclr_results/2091_Differentially Private Federated Learning with Time-Adaptive Privacy Spending/images/f015d9cfb38cce8a5e26bde21bcbf0133e7d5f797bcb145774152267c0358c4d.jpg)

### Tables

![2c4594fc271c2f98abe29f2fb34e9ca00d95221dc1d25d88042ac42564dd180d.jpg](../iclr_results/2091_Differentially Private Federated Learning with Time-Adaptive Privacy Spending/tables/2c4594fc271c2f98abe29f2fb34e9ca00d95221dc1d25d88042ac42564dd180d.jpg)

![4bea8f64105b8ffda2adf16038a9440478027bce3dead7d865566765348bac61.jpg](../iclr_results/2091_Differentially Private Federated Learning with Time-Adaptive Privacy Spending/tables/4bea8f64105b8ffda2adf16038a9440478027bce3dead7d865566765348bac61.jpg)

![4c5533c175e503ec02883cedaaf3d6a5eea76624212422f76f0c30a73460665c.jpg](../iclr_results/2091_Differentially Private Federated Learning with Time-Adaptive Privacy Spending/tables/4c5533c175e503ec02883cedaaf3d6a5eea76624212422f76f0c30a73460665c.jpg)

![5fa7a09b9af442a48fbdc23140bb9d3c835fb9019c0e0cb2dd5288cd41a1d0c0.jpg](../iclr_results/2091_Differentially Private Federated Learning with Time-Adaptive Privacy Spending/tables/5fa7a09b9af442a48fbdc23140bb9d3c835fb9019c0e0cb2dd5288cd41a1d0c0.jpg)

![66c01ca5bab4f2f76e913d843f070a3696ecd12a1ea74ba9acfad869de46e6da.jpg](../iclr_results/2091_Differentially Private Federated Learning with Time-Adaptive Privacy Spending/tables/66c01ca5bab4f2f76e913d843f070a3696ecd12a1ea74ba9acfad869de46e6da.jpg)

![70216dacd702b12f4dc5f7fb9048b3dbe8af00f67923942d366e6d7287cf2eb7.jpg](../iclr_results/2091_Differentially Private Federated Learning with Time-Adaptive Privacy Spending/tables/70216dacd702b12f4dc5f7fb9048b3dbe8af00f67923942d366e6d7287cf2eb7.jpg)

![7aad7f011af894c9666d2d9cb6316ec3d545e8c1043845cdccbb454d05bdf5b1.jpg](../iclr_results/2091_Differentially Private Federated Learning with Time-Adaptive Privacy Spending/tables/7aad7f011af894c9666d2d9cb6316ec3d545e8c1043845cdccbb454d05bdf5b1.jpg)

![851ec6936c8e2c9e2ad41c18511b8722937de81fae78132f6813e84656c75f89.jpg](../iclr_results/2091_Differentially Private Federated Learning with Time-Adaptive Privacy Spending/tables/851ec6936c8e2c9e2ad41c18511b8722937de81fae78132f6813e84656c75f89.jpg)

![b2af47080090ca32c870cb95d362b75716bc83d1a5998f9f4ec4e179017eb524.jpg](../iclr_results/2091_Differentially Private Federated Learning with Time-Adaptive Privacy Spending/tables/b2af47080090ca32c870cb95d362b75716bc83d1a5998f9f4ec4e179017eb524.jpg)

![cc5fe837e0a8a3a516a947c56262a691a18812dff96829e7e1cd8adbd73163d7.jpg](../iclr_results/2091_Differentially Private Federated Learning with Time-Adaptive Privacy Spending/tables/cc5fe837e0a8a3a516a947c56262a691a18812dff96829e7e1cd8adbd73163d7.jpg)

![fa8579fe8fde9f357d5d6f313175a67b445e4dc2c8b103c02cf0e4b181f17f28.jpg](../iclr_results/2091_Differentially Private Federated Learning with Time-Adaptive Privacy Spending/tables/fa8579fe8fde9f357d5d6f313175a67b445e4dc2c8b103c02cf0e4b181f17f28.jpg)

![fc832db834ba96a34e1c4d0f323d35b8ab1e20f51b4ad344b149298c8572e5c1.jpg](../iclr_results/2091_Differentially Private Federated Learning with Time-Adaptive Privacy Spending/tables/fc832db834ba96a34e1c4d0f323d35b8ab1e20f51b4ad344b149298c8572e5c1.jpg)

## TimeInf: Time Series Data Contribution via Influence Functions


### Images

![08cace287e07529b1b91aeaa26d2a6670102cc0bccf352e15f994a3216574c99.jpg](../iclr_results/2092_TimeInf_ Time Series Data Contribution via Influence Functions/images/08cace287e07529b1b91aeaa26d2a6670102cc0bccf352e15f994a3216574c99.jpg)

![0ca86060438d83b4ee715ff764d61f830569146f64a4cdad5698adfe6d69781d.jpg](../iclr_results/2092_TimeInf_ Time Series Data Contribution via Influence Functions/images/0ca86060438d83b4ee715ff764d61f830569146f64a4cdad5698adfe6d69781d.jpg)

![1b184ff69b41cb83e83414d006fb8c9b2dde100c361c66e8fcd0608dc1e69347.jpg](../iclr_results/2092_TimeInf_ Time Series Data Contribution via Influence Functions/images/1b184ff69b41cb83e83414d006fb8c9b2dde100c361c66e8fcd0608dc1e69347.jpg)

![27690097efee5c40920f37a73bb98804b84a7214d21abd60e28b0ca2a35671a8.jpg](../iclr_results/2092_TimeInf_ Time Series Data Contribution via Influence Functions/images/27690097efee5c40920f37a73bb98804b84a7214d21abd60e28b0ca2a35671a8.jpg)

![50f20d7f16633b70c49c621e56abd8d172bd7b38a1181045143c844713a16a06.jpg](../iclr_results/2092_TimeInf_ Time Series Data Contribution via Influence Functions/images/50f20d7f16633b70c49c621e56abd8d172bd7b38a1181045143c844713a16a06.jpg)

![70604011864c2e9ae9be1d1e22a9e36c2540a47b94a186526981ddfaf945670a.jpg](../iclr_results/2092_TimeInf_ Time Series Data Contribution via Influence Functions/images/70604011864c2e9ae9be1d1e22a9e36c2540a47b94a186526981ddfaf945670a.jpg)

![987ee504672ab512f5f863dc47ee2d2412198090b96f5c347508c59086d65b61.jpg](../iclr_results/2092_TimeInf_ Time Series Data Contribution via Influence Functions/images/987ee504672ab512f5f863dc47ee2d2412198090b96f5c347508c59086d65b61.jpg)

![a6777421784982b8965881a03cb9d37b6e9e09ac806f3c2140c5849415dfeb28.jpg](../iclr_results/2092_TimeInf_ Time Series Data Contribution via Influence Functions/images/a6777421784982b8965881a03cb9d37b6e9e09ac806f3c2140c5849415dfeb28.jpg)

![c3f6a4cec6722199b579efedfbf04d914be2739f4d926a25714ac16401c77b74.jpg](../iclr_results/2092_TimeInf_ Time Series Data Contribution via Influence Functions/images/c3f6a4cec6722199b579efedfbf04d914be2739f4d926a25714ac16401c77b74.jpg)

![c5c1cd55ad4ea8e5945cf4f1b2bd4b810998a9697e0b1baf2fea12316910bbf9.jpg](../iclr_results/2092_TimeInf_ Time Series Data Contribution via Influence Functions/images/c5c1cd55ad4ea8e5945cf4f1b2bd4b810998a9697e0b1baf2fea12316910bbf9.jpg)

![d5042861ab2e0001fecc2873d6beaf99479587cb2e06d6de0a5f6f629084986f.jpg](../iclr_results/2092_TimeInf_ Time Series Data Contribution via Influence Functions/images/d5042861ab2e0001fecc2873d6beaf99479587cb2e06d6de0a5f6f629084986f.jpg)

![f1ea22c2e894e91c804e74d8a0f384ad3975a054341ffa37025a5cfb9d838704.jpg](../iclr_results/2092_TimeInf_ Time Series Data Contribution via Influence Functions/images/f1ea22c2e894e91c804e74d8a0f384ad3975a054341ffa37025a5cfb9d838704.jpg)

### Tables

![0b31f564b556b419e09a4d8f4a8508a9b51aa055e1230dd105d71031da755390.jpg](../iclr_results/2092_TimeInf_ Time Series Data Contribution via Influence Functions/tables/0b31f564b556b419e09a4d8f4a8508a9b51aa055e1230dd105d71031da755390.jpg)

![3698c1f9c606078994e75f470beca2746300f74a535882e8b6fc94a1c0bf73b3.jpg](../iclr_results/2092_TimeInf_ Time Series Data Contribution via Influence Functions/tables/3698c1f9c606078994e75f470beca2746300f74a535882e8b6fc94a1c0bf73b3.jpg)

![430234e98fc82af56929a3733dc3de99044ba4ce084bcf08ae1107b1f4aef6fe.jpg](../iclr_results/2092_TimeInf_ Time Series Data Contribution via Influence Functions/tables/430234e98fc82af56929a3733dc3de99044ba4ce084bcf08ae1107b1f4aef6fe.jpg)

![74b5da1000de33a1dd1db0fd1fb12745d79c4ae28fb6a3d6239470e444ebb127.jpg](../iclr_results/2092_TimeInf_ Time Series Data Contribution via Influence Functions/tables/74b5da1000de33a1dd1db0fd1fb12745d79c4ae28fb6a3d6239470e444ebb127.jpg)

![c9c6c939f870b932b9ea5094120025690124b227b0aeff2d38f7d516b5e91eba.jpg](../iclr_results/2092_TimeInf_ Time Series Data Contribution via Influence Functions/tables/c9c6c939f870b932b9ea5094120025690124b227b0aeff2d38f7d516b5e91eba.jpg)

## Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model


### Images

![34c65710a9d869e4b5fcc0908df1bebc433c99fba44df2242d1aaf33981985d4.jpg](../iclr_results/2093_Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model/images/34c65710a9d869e4b5fcc0908df1bebc433c99fba44df2242d1aaf33981985d4.jpg)

![3c28a21f411a72b6bb867a80abff9a35d35b08c849bbf9551bae97f218f76957.jpg](../iclr_results/2093_Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model/images/3c28a21f411a72b6bb867a80abff9a35d35b08c849bbf9551bae97f218f76957.jpg)

![3f4df11acb3c78f5ef9e82db1f73e8621826a151750f23157cea2bab3a2a6b32.jpg](../iclr_results/2093_Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model/images/3f4df11acb3c78f5ef9e82db1f73e8621826a151750f23157cea2bab3a2a6b32.jpg)

![715fde100d539dd9fc0900597bc18f0b8366200c1d798e357454f80b276d5724.jpg](../iclr_results/2093_Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model/images/715fde100d539dd9fc0900597bc18f0b8366200c1d798e357454f80b276d5724.jpg)

![869593adc2c386ae814d36d8d4ed42a5e0df874b2a47eee19202f683b882ed50.jpg](../iclr_results/2093_Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model/images/869593adc2c386ae814d36d8d4ed42a5e0df874b2a47eee19202f683b882ed50.jpg)

![8c7256cc535ffe43b5aa6482a36f3a2cce2b5209455e1a8e57a2ef32f85df67d.jpg](../iclr_results/2093_Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model/images/8c7256cc535ffe43b5aa6482a36f3a2cce2b5209455e1a8e57a2ef32f85df67d.jpg)

![a219ecbac4797f9702cc17a1dac46f092a89299ee226ce33f6e6112ee9ef1527.jpg](../iclr_results/2093_Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model/images/a219ecbac4797f9702cc17a1dac46f092a89299ee226ce33f6e6112ee9ef1527.jpg)

![bbfbbc7a72b80b9dd451fad9a1b1984300c53b406858437d0c8d709530d20fd8.jpg](../iclr_results/2093_Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model/images/bbfbbc7a72b80b9dd451fad9a1b1984300c53b406858437d0c8d709530d20fd8.jpg)

![c91090ce4d08119c1ad0c0571eb15b9769f7b692ef0e0215a3f0ce8a89dd2496.jpg](../iclr_results/2093_Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model/images/c91090ce4d08119c1ad0c0571eb15b9769f7b692ef0e0215a3f0ce8a89dd2496.jpg)

### Tables

![0300af4499b2d82287637e88008c8aa5f3de0d196356a1814225f57ce6468edc.jpg](../iclr_results/2093_Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model/tables/0300af4499b2d82287637e88008c8aa5f3de0d196356a1814225f57ce6468edc.jpg)

![19252c0eb022e6ba8d044b125e807e89c177e2951957deecf87e4b88ad9176a0.jpg](../iclr_results/2093_Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model/tables/19252c0eb022e6ba8d044b125e807e89c177e2951957deecf87e4b88ad9176a0.jpg)

![1b833ad24e5eaf133868c78f59a235b4a630e57a9f5a45c13b08373f86453080.jpg](../iclr_results/2093_Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model/tables/1b833ad24e5eaf133868c78f59a235b4a630e57a9f5a45c13b08373f86453080.jpg)

![53aa071b0d5ec455aaaf5ba82b9693d4f768cd695d1a37b0274b2636b22f48a2.jpg](../iclr_results/2093_Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model/tables/53aa071b0d5ec455aaaf5ba82b9693d4f768cd695d1a37b0274b2636b22f48a2.jpg)

![5f77a3ce78759587c5744c738b3a13369197289ffdf2164a0afd76bfdb575008.jpg](../iclr_results/2093_Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model/tables/5f77a3ce78759587c5744c738b3a13369197289ffdf2164a0afd76bfdb575008.jpg)

![728c07d22b9d803524c9940cab0e6dd92e6cae176f9945ef56f8cd7f162074e8.jpg](../iclr_results/2093_Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model/tables/728c07d22b9d803524c9940cab0e6dd92e6cae176f9945ef56f8cd7f162074e8.jpg)

![8c3dc736377b831e5f8ba028f5575e36d85a27523ab069b7671563bed4de6ae4.jpg](../iclr_results/2093_Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model/tables/8c3dc736377b831e5f8ba028f5575e36d85a27523ab069b7671563bed4de6ae4.jpg)

![8f2e942ca82c66650def66f4b0629d334018c819f6971f2e14953cfa5c90227d.jpg](../iclr_results/2093_Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model/tables/8f2e942ca82c66650def66f4b0629d334018c819f6971f2e14953cfa5c90227d.jpg)

![a4128af9a8c8f6fff36017e45ef7d888f0972d3f290da8d8dbc3f9857cfe2b10.jpg](../iclr_results/2093_Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model/tables/a4128af9a8c8f6fff36017e45ef7d888f0972d3f290da8d8dbc3f9857cfe2b10.jpg)

![ab9ad133424de1545b0e0737fa564e4c8cb1ff57e771eff65092aaecff7970ca.jpg](../iclr_results/2093_Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model/tables/ab9ad133424de1545b0e0737fa564e4c8cb1ff57e771eff65092aaecff7970ca.jpg)

![b4162b1473d5fdddab5e2cad505356ce0957f18bc8807b06f0a317b44da38424.jpg](../iclr_results/2093_Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model/tables/b4162b1473d5fdddab5e2cad505356ce0957f18bc8807b06f0a317b44da38424.jpg)

![b4b1b4343fb5b62461058029d30e89cb25621fa2552ed7ef2cdf7c6172e2f0aa.jpg](../iclr_results/2093_Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model/tables/b4b1b4343fb5b62461058029d30e89cb25621fa2552ed7ef2cdf7c6172e2f0aa.jpg)

![c1c1716de713a17043d7a65598253795fc0e1544e7bdf4e70e4650d664c8c278.jpg](../iclr_results/2093_Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model/tables/c1c1716de713a17043d7a65598253795fc0e1544e7bdf4e70e4650d664c8c278.jpg)

![fccc19e1a1858bc616c2252ab66aa45037949d2ae55573187c8679810fd6c1c4.jpg](../iclr_results/2093_Solving Token Gradient Conflict in Mixture-of-Experts for Large Vision-Language Model/tables/fccc19e1a1858bc616c2252ab66aa45037949d2ae55573187c8679810fd6c1c4.jpg)

## HaDeMiF: Hallucination Detection and Mitigation in Large Language Models


### Images

![0619ec65ec3e1e93ac474b5b91422680e4e11bd2d2e1c57258519e6d20508e05.jpg](../iclr_results/2094_HaDeMiF_ Hallucination Detection and Mitigation in Large Language Models/images/0619ec65ec3e1e93ac474b5b91422680e4e11bd2d2e1c57258519e6d20508e05.jpg)

![28282b061494fb0710cc4a8f870d852ca16857501566465d03eaa3306de31297.jpg](../iclr_results/2094_HaDeMiF_ Hallucination Detection and Mitigation in Large Language Models/images/28282b061494fb0710cc4a8f870d852ca16857501566465d03eaa3306de31297.jpg)

![42fae8e8357234cdab9a76e20b4c6d1344e59e8bb26ba717120b7bb73a82f805.jpg](../iclr_results/2094_HaDeMiF_ Hallucination Detection and Mitigation in Large Language Models/images/42fae8e8357234cdab9a76e20b4c6d1344e59e8bb26ba717120b7bb73a82f805.jpg)

![5a59f10b705c73e01a3c65afcc1b4e630f58d46f863f1a80b22d13bfdfc26566.jpg](../iclr_results/2094_HaDeMiF_ Hallucination Detection and Mitigation in Large Language Models/images/5a59f10b705c73e01a3c65afcc1b4e630f58d46f863f1a80b22d13bfdfc26566.jpg)

![7a50f81e38e30aca01133e51742af2009e3823cae3404b8809ccfa0d08c0921b.jpg](../iclr_results/2094_HaDeMiF_ Hallucination Detection and Mitigation in Large Language Models/images/7a50f81e38e30aca01133e51742af2009e3823cae3404b8809ccfa0d08c0921b.jpg)

![f32c09d5ee9447820f3f57e861d430b0f7431b60886d52e946f799fc0e5c84e5.jpg](../iclr_results/2094_HaDeMiF_ Hallucination Detection and Mitigation in Large Language Models/images/f32c09d5ee9447820f3f57e861d430b0f7431b60886d52e946f799fc0e5c84e5.jpg)

### Tables

![0713751e5e2565bc6c3400c4157fc0942d9b9bb09430b085946a71a22e5507bd.jpg](../iclr_results/2094_HaDeMiF_ Hallucination Detection and Mitigation in Large Language Models/tables/0713751e5e2565bc6c3400c4157fc0942d9b9bb09430b085946a71a22e5507bd.jpg)

![08106ee15063540506b89c848ce3bc87cb8ee8259091356d122612d090e2ba70.jpg](../iclr_results/2094_HaDeMiF_ Hallucination Detection and Mitigation in Large Language Models/tables/08106ee15063540506b89c848ce3bc87cb8ee8259091356d122612d090e2ba70.jpg)

![1d5b35d8597202cc4fe402ad035bff5910ba77418e1a5b5c4117f52bcdb8c831.jpg](../iclr_results/2094_HaDeMiF_ Hallucination Detection and Mitigation in Large Language Models/tables/1d5b35d8597202cc4fe402ad035bff5910ba77418e1a5b5c4117f52bcdb8c831.jpg)

![5132e4d9783fb9f5f0a101bf15e998f6a41b54ded11309882f6bdcb85a8c2e65.jpg](../iclr_results/2094_HaDeMiF_ Hallucination Detection and Mitigation in Large Language Models/tables/5132e4d9783fb9f5f0a101bf15e998f6a41b54ded11309882f6bdcb85a8c2e65.jpg)

![5a288eafc8caae18c80faf555375cab8179f3225db6d6f797a1400b3183b96b6.jpg](../iclr_results/2094_HaDeMiF_ Hallucination Detection and Mitigation in Large Language Models/tables/5a288eafc8caae18c80faf555375cab8179f3225db6d6f797a1400b3183b96b6.jpg)

![8a94c65ee2fadbf7505d3f70d4f715f211857c012eae0acc553ba7f84398688f.jpg](../iclr_results/2094_HaDeMiF_ Hallucination Detection and Mitigation in Large Language Models/tables/8a94c65ee2fadbf7505d3f70d4f715f211857c012eae0acc553ba7f84398688f.jpg)

![9476b97cb0491035b5b463b3641f893053d5bf4afa4e3bf7fca14e86ade606e9.jpg](../iclr_results/2094_HaDeMiF_ Hallucination Detection and Mitigation in Large Language Models/tables/9476b97cb0491035b5b463b3641f893053d5bf4afa4e3bf7fca14e86ade606e9.jpg)

![a4dd28e4b24ed3707832fcf854ec561df36b7e29bc03ed7edcca7bfea2057505.jpg](../iclr_results/2094_HaDeMiF_ Hallucination Detection and Mitigation in Large Language Models/tables/a4dd28e4b24ed3707832fcf854ec561df36b7e29bc03ed7edcca7bfea2057505.jpg)

![aeaa2f93d81390e65f0d0990db1c77681c1d923eb71627acf5a686426b1f379a.jpg](../iclr_results/2094_HaDeMiF_ Hallucination Detection and Mitigation in Large Language Models/tables/aeaa2f93d81390e65f0d0990db1c77681c1d923eb71627acf5a686426b1f379a.jpg)

![b99cc43013d41d4032d19b0a81800c745bffb1c1ceac23d661db1ac5f20a50f3.jpg](../iclr_results/2094_HaDeMiF_ Hallucination Detection and Mitigation in Large Language Models/tables/b99cc43013d41d4032d19b0a81800c745bffb1c1ceac23d661db1ac5f20a50f3.jpg)

![bd3e58aee0a11d29dfa29dbdc0882360156a927623b76b7a6878590c3ebbf007.jpg](../iclr_results/2094_HaDeMiF_ Hallucination Detection and Mitigation in Large Language Models/tables/bd3e58aee0a11d29dfa29dbdc0882360156a927623b76b7a6878590c3ebbf007.jpg)

![e7c23f8a3b3e944ee61105b2b7a35e886352fca0339de6fdf1bd217996b549bf.jpg](../iclr_results/2094_HaDeMiF_ Hallucination Detection and Mitigation in Large Language Models/tables/e7c23f8a3b3e944ee61105b2b7a35e886352fca0339de6fdf1bd217996b549bf.jpg)

## Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Planning Agent


### Images

![0f3180196bb718e5ccef56360859ea53a90b74231dea6e8463fea07a8178271f.jpg](../iclr_results/2095_Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Pl/images/0f3180196bb718e5ccef56360859ea53a90b74231dea6e8463fea07a8178271f.jpg)

![1ca5c42990c643758780f90c5de29602a6f7b750c88a5d3c9ff02174acfc8a0e.jpg](../iclr_results/2095_Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Pl/images/1ca5c42990c643758780f90c5de29602a6f7b750c88a5d3c9ff02174acfc8a0e.jpg)

![227b521c03ff847d41e476271d2562d935c213a2e54e1995468887898af6a7a3.jpg](../iclr_results/2095_Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Pl/images/227b521c03ff847d41e476271d2562d935c213a2e54e1995468887898af6a7a3.jpg)

![38ac129bd55e02b73a570afddaeb3242e7d2ec90ce4ab64302b0fd1ed76a6ed2.jpg](../iclr_results/2095_Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Pl/images/38ac129bd55e02b73a570afddaeb3242e7d2ec90ce4ab64302b0fd1ed76a6ed2.jpg)

![52c4f4a7d8b0498657874663444f32bc1e1e226800506b24590fa90e0f7c2a77.jpg](../iclr_results/2095_Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Pl/images/52c4f4a7d8b0498657874663444f32bc1e1e226800506b24590fa90e0f7c2a77.jpg)

![7af92c33d584e0fe42b449739974ca5528dbd582af4160d04094ddf55459be45.jpg](../iclr_results/2095_Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Pl/images/7af92c33d584e0fe42b449739974ca5528dbd582af4160d04094ddf55459be45.jpg)

![bf2b76553944f3c7ac9ef32e93b24f6bf2057b665f250c96875b76739a9b871b.jpg](../iclr_results/2095_Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Pl/images/bf2b76553944f3c7ac9ef32e93b24f6bf2057b665f250c96875b76739a9b871b.jpg)

![eeb10ee6f94d0ed614bb91177d6c11492a15e0967faa5193b2eb34f493a8c920.jpg](../iclr_results/2095_Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Pl/images/eeb10ee6f94d0ed614bb91177d6c11492a15e0967faa5193b2eb34f493a8c920.jpg)

### Tables

![13544e148369585aedb85279613fe5a261768f949a7bc852d006a79db6e894a1.jpg](../iclr_results/2095_Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Pl/tables/13544e148369585aedb85279613fe5a261768f949a7bc852d006a79db6e894a1.jpg)

![179105d90b9d8da6127bb4b95880fef788a91862a4e872fe9fb115fde3046205.jpg](../iclr_results/2095_Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Pl/tables/179105d90b9d8da6127bb4b95880fef788a91862a4e872fe9fb115fde3046205.jpg)

![3b7f9301435f347f2b98843fdf7371c330249298b3dd1f2bcab3e251d8884108.jpg](../iclr_results/2095_Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Pl/tables/3b7f9301435f347f2b98843fdf7371c330249298b3dd1f2bcab3e251d8884108.jpg)

![4e0dc94276e1829bab2d20aedb9a007de68ed9ae22a21ec6433ad1f13572ee0f.jpg](../iclr_results/2095_Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Pl/tables/4e0dc94276e1829bab2d20aedb9a007de68ed9ae22a21ec6433ad1f13572ee0f.jpg)

![5773f723e3cddbd5b9fe2575826f92916d703dce9612264cc855d67bb6ef5edb.jpg](../iclr_results/2095_Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Pl/tables/5773f723e3cddbd5b9fe2575826f92916d703dce9612264cc855d67bb6ef5edb.jpg)

![709bc11b3c60935e7da060838365297e38b354e9af27db889f2fa304e8fd98b1.jpg](../iclr_results/2095_Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Pl/tables/709bc11b3c60935e7da060838365297e38b354e9af27db889f2fa304e8fd98b1.jpg)

![818f9be45281c5738fd35fe23337a716314b0e8d24ee23bfc15e5bf033fdd86a.jpg](../iclr_results/2095_Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Pl/tables/818f9be45281c5738fd35fe23337a716314b0e8d24ee23bfc15e5bf033fdd86a.jpg)

![a5621e0b022bfc304775b7447205badaf47e35bf1545fa1e2352d24229d75748.jpg](../iclr_results/2095_Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Pl/tables/a5621e0b022bfc304775b7447205badaf47e35bf1545fa1e2352d24229d75748.jpg)

![a99f231c59c666e4015ef68a81d5726717b4e9db278eb5b79bcb6cba30af0073.jpg](../iclr_results/2095_Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Pl/tables/a99f231c59c666e4015ef68a81d5726717b4e9db278eb5b79bcb6cba30af0073.jpg)

![af3cbc8a38e78477591ba8c5dcc24cb86c436819ec390937759acbb2928ee251.jpg](../iclr_results/2095_Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Pl/tables/af3cbc8a38e78477591ba8c5dcc24cb86c436819ec390937759acbb2928ee251.jpg)

![b944305776da01452b52a987305317bcdff751796468f2ac8b9ccf5dfb6d3880.jpg](../iclr_results/2095_Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Pl/tables/b944305776da01452b52a987305317bcdff751796468f2ac8b9ccf5dfb6d3880.jpg)

![bba9e0ea7622e39e1fa51049540f4f438c76c21a5b8a43b78ed027a032ab0ad1.jpg](../iclr_results/2095_Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Pl/tables/bba9e0ea7622e39e1fa51049540f4f438c76c21a5b8a43b78ed027a032ab0ad1.jpg)

![c65fffc85144e80c84be1ddea20d89f7d9127deff98c099eae6aae796a1a28e0.jpg](../iclr_results/2095_Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Pl/tables/c65fffc85144e80c84be1ddea20d89f7d9127deff98c099eae6aae796a1a28e0.jpg)

![d416e1a25aa1ead09cd3b61b4eb8055733c81a1d724ecf1f77745338389229f5.jpg](../iclr_results/2095_Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Pl/tables/d416e1a25aa1ead09cd3b61b4eb8055733c81a1d724ecf1f77745338389229f5.jpg)

![f5f8541a9f537b74f82df8774340c6af571adedcf48316f40aafe7d065dc36d0.jpg](../iclr_results/2095_Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Pl/tables/f5f8541a9f537b74f82df8774340c6af571adedcf48316f40aafe7d065dc36d0.jpg)

## Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving


### Images

![08eb3608397f7b7070e3aa8bb40eda28d0b34e9403cc6b75db4055426d053cb1.jpg](../iclr_results/2096_Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving/images/08eb3608397f7b7070e3aa8bb40eda28d0b34e9403cc6b75db4055426d053cb1.jpg)

![1487c3c4aa7ed0dfc6ced68cf11a6315cf4adbcd3045cbf16ac09f34d6df56ac.jpg](../iclr_results/2096_Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving/images/1487c3c4aa7ed0dfc6ced68cf11a6315cf4adbcd3045cbf16ac09f34d6df56ac.jpg)

![2af813754286dd8619be3342811ad86e191c3e23c271b84ccc74c7cd83178f39.jpg](../iclr_results/2096_Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving/images/2af813754286dd8619be3342811ad86e191c3e23c271b84ccc74c7cd83178f39.jpg)

![35c07a3a017dc68b939af564d93ba5bbd31726fbfc5d94aa1a3655a9e5d3c7c0.jpg](../iclr_results/2096_Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving/images/35c07a3a017dc68b939af564d93ba5bbd31726fbfc5d94aa1a3655a9e5d3c7c0.jpg)

![48f22f1d2d5172e3182915678c95057bff956cb65c90e0182accdda167b803a7.jpg](../iclr_results/2096_Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving/images/48f22f1d2d5172e3182915678c95057bff956cb65c90e0182accdda167b803a7.jpg)

![4bcef40d58136694d8be404f7dd68f3c446ebcf7ff5c17afef90312235263b68.jpg](../iclr_results/2096_Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving/images/4bcef40d58136694d8be404f7dd68f3c446ebcf7ff5c17afef90312235263b68.jpg)

![735fdc397002a9f37722bc4455c0d897451f9d404f00d37158141d541b6f4ea5.jpg](../iclr_results/2096_Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving/images/735fdc397002a9f37722bc4455c0d897451f9d404f00d37158141d541b6f4ea5.jpg)

![7496506a0704edcdacf46bcf4f922e6f3b7e5dbcf3e03365c6955d3a0c2d1e3c.jpg](../iclr_results/2096_Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving/images/7496506a0704edcdacf46bcf4f922e6f3b7e5dbcf3e03365c6955d3a0c2d1e3c.jpg)

![81747ea21312e62c7a66148c6ba967b10e4ebb2ba81304cc8a4a2c6629c53008.jpg](../iclr_results/2096_Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving/images/81747ea21312e62c7a66148c6ba967b10e4ebb2ba81304cc8a4a2c6629c53008.jpg)

![9caf3a44e4531ddaa07a0c2ccc6bf69bf8b27089b1ec1cbef5347f2eb4e6534c.jpg](../iclr_results/2096_Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving/images/9caf3a44e4531ddaa07a0c2ccc6bf69bf8b27089b1ec1cbef5347f2eb4e6534c.jpg)

![c1d1f7c65fcfa96382584fdd253709ea096c66d0b350a87faa0bbd82591865c2.jpg](../iclr_results/2096_Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving/images/c1d1f7c65fcfa96382584fdd253709ea096c66d0b350a87faa0bbd82591865c2.jpg)

![c9fda45d93eaf2e660bb1fd1b51b2d70c708bcfeffbf61d4f4d28d44004d3f3d.jpg](../iclr_results/2096_Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving/images/c9fda45d93eaf2e660bb1fd1b51b2d70c708bcfeffbf61d4f4d28d44004d3f3d.jpg)

![ca04527597058291a219f25ac61b9b0d6d12f144644cb3d60e0da75c179cc1a7.jpg](../iclr_results/2096_Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving/images/ca04527597058291a219f25ac61b9b0d6d12f144644cb3d60e0da75c179cc1a7.jpg)

### Tables

![0a8c0edfaebbf6795209ebb50f9abd9c35f51cb066c90fe95c81545b5ae38b63.jpg](../iclr_results/2096_Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving/tables/0a8c0edfaebbf6795209ebb50f9abd9c35f51cb066c90fe95c81545b5ae38b63.jpg)

![280ca11e32c52b3297b58ca899ceecdb6a0c40078aceb4e51c1731e5fbd15386.jpg](../iclr_results/2096_Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving/tables/280ca11e32c52b3297b58ca899ceecdb6a0c40078aceb4e51c1731e5fbd15386.jpg)

![29a1aa3b1d9de88f9a3796f204eb43d854599b55148f4249098718e92aa239cb.jpg](../iclr_results/2096_Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving/tables/29a1aa3b1d9de88f9a3796f204eb43d854599b55148f4249098718e92aa239cb.jpg)

![44ffd32c4e565beb0764fcac3355d73e8b5df3586240cd74f9c3035fb49e35b4.jpg](../iclr_results/2096_Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving/tables/44ffd32c4e565beb0764fcac3355d73e8b5df3586240cd74f9c3035fb49e35b4.jpg)

![648a0a61b4e0a72b031fc74f9b915c85e3d27068a8a15ec67780241284f90c84.jpg](../iclr_results/2096_Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving/tables/648a0a61b4e0a72b031fc74f9b915c85e3d27068a8a15ec67780241284f90c84.jpg)

![8049941274f526a96f0cda84ad43e95d869089d38d94827cc194f80f2a588c9d.jpg](../iclr_results/2096_Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving/tables/8049941274f526a96f0cda84ad43e95d869089d38d94827cc194f80f2a588c9d.jpg)

![8cdad2a4273a4e303201189cab31aea16845e3d6bde6413c4a7327c46b3cff54.jpg](../iclr_results/2096_Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving/tables/8cdad2a4273a4e303201189cab31aea16845e3d6bde6413c4a7327c46b3cff54.jpg)

![d6d3d47b544efab664a88a4e2821d3d313bf686fc5c46f2f10e81f8aee32fc4d.jpg](../iclr_results/2096_Navigation-Guided Sparse Scene Representation for End-to-End Autonomous Driving/tables/d6d3d47b544efab664a88a4e2821d3d313bf686fc5c46f2f10e81f8aee32fc4d.jpg)

## OSCAR: Operating System Control via State-Aware Reasoning and Re-Planning


### Images

![0d811d9db7476d2e89bd0fe8d9e82f6b32128161b2b45e05b055f7f1782f9b46.jpg](../iclr_results/2097_OSCAR_ Operating System Control via State-Aware Reasoning and Re-Planning/images/0d811d9db7476d2e89bd0fe8d9e82f6b32128161b2b45e05b055f7f1782f9b46.jpg)

![0dd93c11261550a15b401da135aed884f6d5520f955b25c34054ca0feedc2b42.jpg](../iclr_results/2097_OSCAR_ Operating System Control via State-Aware Reasoning and Re-Planning/images/0dd93c11261550a15b401da135aed884f6d5520f955b25c34054ca0feedc2b42.jpg)

![19934e7e6bf245b051a294e63618bc9b1546f36ec5fd795d2b88ef4f5d9b7e80.jpg](../iclr_results/2097_OSCAR_ Operating System Control via State-Aware Reasoning and Re-Planning/images/19934e7e6bf245b051a294e63618bc9b1546f36ec5fd795d2b88ef4f5d9b7e80.jpg)

![866e221069271e1a4960d34d2d200b8021552b870f972de6c6c7b73cfa0da657.jpg](../iclr_results/2097_OSCAR_ Operating System Control via State-Aware Reasoning and Re-Planning/images/866e221069271e1a4960d34d2d200b8021552b870f972de6c6c7b73cfa0da657.jpg)

![9585bef0db3e7356f70583fe8471acd654d9320adbe0838d3a700b490613a9c2.jpg](../iclr_results/2097_OSCAR_ Operating System Control via State-Aware Reasoning and Re-Planning/images/9585bef0db3e7356f70583fe8471acd654d9320adbe0838d3a700b490613a9c2.jpg)

![9c27fbf488870f5dc16bca891ff37a0e1e632b6257efb0f1684750b25c09937c.jpg](../iclr_results/2097_OSCAR_ Operating System Control via State-Aware Reasoning and Re-Planning/images/9c27fbf488870f5dc16bca891ff37a0e1e632b6257efb0f1684750b25c09937c.jpg)

![b599b82224133b01f838346039c2014e1bc44849d4bbf6e698db8c55b91f082c.jpg](../iclr_results/2097_OSCAR_ Operating System Control via State-Aware Reasoning and Re-Planning/images/b599b82224133b01f838346039c2014e1bc44849d4bbf6e698db8c55b91f082c.jpg)

![d32fb503230bbdbdc903c6645019725ac6c21d8a53ce6aef3a0ac28667ff53ed.jpg](../iclr_results/2097_OSCAR_ Operating System Control via State-Aware Reasoning and Re-Planning/images/d32fb503230bbdbdc903c6645019725ac6c21d8a53ce6aef3a0ac28667ff53ed.jpg)

![e20187f0c7de958fb045634000bda7fffaff20f0c5b9060e45d3433fe4cbd36e.jpg](../iclr_results/2097_OSCAR_ Operating System Control via State-Aware Reasoning and Re-Planning/images/e20187f0c7de958fb045634000bda7fffaff20f0c5b9060e45d3433fe4cbd36e.jpg)

### Tables

![0273f2bee43ec773981ac905b74ad27da5ec59451b1a13c4e80fafb803a821f7.jpg](../iclr_results/2097_OSCAR_ Operating System Control via State-Aware Reasoning and Re-Planning/tables/0273f2bee43ec773981ac905b74ad27da5ec59451b1a13c4e80fafb803a821f7.jpg)

![42e619073c990b816f943e80c872864439cda23f84deeef18ef9b721b6ad27da.jpg](../iclr_results/2097_OSCAR_ Operating System Control via State-Aware Reasoning and Re-Planning/tables/42e619073c990b816f943e80c872864439cda23f84deeef18ef9b721b6ad27da.jpg)

![45fc5e0073bfbdea5f0f9032459c43521b356ac4df6b003d77de7da8bdd6251a.jpg](../iclr_results/2097_OSCAR_ Operating System Control via State-Aware Reasoning and Re-Planning/tables/45fc5e0073bfbdea5f0f9032459c43521b356ac4df6b003d77de7da8bdd6251a.jpg)

![6ec3ca1806fb8149bc2395d804ae20acc49ce471f397f2dfe4ff7ea7848dbbd8.jpg](../iclr_results/2097_OSCAR_ Operating System Control via State-Aware Reasoning and Re-Planning/tables/6ec3ca1806fb8149bc2395d804ae20acc49ce471f397f2dfe4ff7ea7848dbbd8.jpg)

![8b9b4165e515acc00fb100827aa618ddb402566b4ee1e400952d35879611d16e.jpg](../iclr_results/2097_OSCAR_ Operating System Control via State-Aware Reasoning and Re-Planning/tables/8b9b4165e515acc00fb100827aa618ddb402566b4ee1e400952d35879611d16e.jpg)

![8e2ccd7f472d71e00f01639ea141b4fdba7e38756858fe3de64c6349347a4b35.jpg](../iclr_results/2097_OSCAR_ Operating System Control via State-Aware Reasoning and Re-Planning/tables/8e2ccd7f472d71e00f01639ea141b4fdba7e38756858fe3de64c6349347a4b35.jpg)

![d93906d09cc71fcb6346a7771d6954964115c09d51888e16f21a125d15fd4cbd.jpg](../iclr_results/2097_OSCAR_ Operating System Control via State-Aware Reasoning and Re-Planning/tables/d93906d09cc71fcb6346a7771d6954964115c09d51888e16f21a125d15fd4cbd.jpg)

![ebe5f2bc592132ca07b69cf7b4bd7a11aecddd3f71fc84b9fda2d40e4181d5b8.jpg](../iclr_results/2097_OSCAR_ Operating System Control via State-Aware Reasoning and Re-Planning/tables/ebe5f2bc592132ca07b69cf7b4bd7a11aecddd3f71fc84b9fda2d40e4181d5b8.jpg)

![fd974f49c703c4f3da1f0b76cee3a44ae61dd9aac37ca2a2c7d1aba07f7dac47.jpg](../iclr_results/2097_OSCAR_ Operating System Control via State-Aware Reasoning and Re-Planning/tables/fd974f49c703c4f3da1f0b76cee3a44ae61dd9aac37ca2a2c7d1aba07f7dac47.jpg)

## GoodDrag: Towards Good Practices for Drag Editing with Diffusion Models


### Images

![1c298b10c72e0ba75d9a5bbcaaa921a2cb670e6afd15a180425afd5d1d402940.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/images/1c298b10c72e0ba75d9a5bbcaaa921a2cb670e6afd15a180425afd5d1d402940.jpg)

![227233a0174ce7fe42b7bf537153e52e3a113a22c76943aac5e8fb222593efc5.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/images/227233a0174ce7fe42b7bf537153e52e3a113a22c76943aac5e8fb222593efc5.jpg)

![325f475099c353fc1b177ce3c72778e2903fbb8c6f8e138595ecc0a220001f78.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/images/325f475099c353fc1b177ce3c72778e2903fbb8c6f8e138595ecc0a220001f78.jpg)

![3ad5a360ab4c6e54887b4eefeb3559313ce9ff1c1232e54b6aa594cefdd5174c.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/images/3ad5a360ab4c6e54887b4eefeb3559313ce9ff1c1232e54b6aa594cefdd5174c.jpg)

![3b0657b1191ac5d12e6ecbb52aaf37651074831784e8ab4ff9d9c42bfceb5138.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/images/3b0657b1191ac5d12e6ecbb52aaf37651074831784e8ab4ff9d9c42bfceb5138.jpg)

![4bc7004f409210d40659caaa34d50d29733c79d3bd1fbaa5e7dbdd56d24a42ec.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/images/4bc7004f409210d40659caaa34d50d29733c79d3bd1fbaa5e7dbdd56d24a42ec.jpg)

![697a9f3b23e80a7f4b1ddd268a7d19075e86603067097ec0b94667944162b195.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/images/697a9f3b23e80a7f4b1ddd268a7d19075e86603067097ec0b94667944162b195.jpg)

![79a12a6bb2362489e5cfa1e260508302ef46aa6409fbf91784209e52b47949ed.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/images/79a12a6bb2362489e5cfa1e260508302ef46aa6409fbf91784209e52b47949ed.jpg)

![7d64710e8ff64ed9b8a374a865d6d9445a4ace5209b11df766161c4afa1f5c4b.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/images/7d64710e8ff64ed9b8a374a865d6d9445a4ace5209b11df766161c4afa1f5c4b.jpg)

![a0bdf729f3a4efedfc101e66369a6cc39721aecbc491a2f74a7a817c44f394d4.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/images/a0bdf729f3a4efedfc101e66369a6cc39721aecbc491a2f74a7a817c44f394d4.jpg)

![a4e9f6f1819231a8c1ff3d0d228c2035b4b7523803bd7ed63239b08c8000b82c.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/images/a4e9f6f1819231a8c1ff3d0d228c2035b4b7523803bd7ed63239b08c8000b82c.jpg)

![a53e0caa59731160c512b3ddf217b0cbb3cea6c68f08c3cb310ffd3d0685a507.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/images/a53e0caa59731160c512b3ddf217b0cbb3cea6c68f08c3cb310ffd3d0685a507.jpg)

![b5b250150e45553646143ec1598cc963bd993d73d4f979af043acc28c2150209.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/images/b5b250150e45553646143ec1598cc963bd993d73d4f979af043acc28c2150209.jpg)

![bcaf47dae372d5574b2191a84b1f574db89a40fbbb6612a336d999da841fc7c1.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/images/bcaf47dae372d5574b2191a84b1f574db89a40fbbb6612a336d999da841fc7c1.jpg)

![c773ad9dfea2c9bea2ffc6f30b62953ca6a6fa4fc2009f940b7a7c12907f7cdb.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/images/c773ad9dfea2c9bea2ffc6f30b62953ca6a6fa4fc2009f940b7a7c12907f7cdb.jpg)

![c7800b03f84fca9513f77ed979f4b5fd3a0771ac369d5c51642b4f11ff198c15.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/images/c7800b03f84fca9513f77ed979f4b5fd3a0771ac369d5c51642b4f11ff198c15.jpg)

![d306bd34c80745685f588b76ff397612c25cece2d1df3152216f9d583fe8184d.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/images/d306bd34c80745685f588b76ff397612c25cece2d1df3152216f9d583fe8184d.jpg)

![d4416f61ff253faa550530aa212fa9caf3011d6b6d9e26e63fb6f3f16ae4694e.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/images/d4416f61ff253faa550530aa212fa9caf3011d6b6d9e26e63fb6f3f16ae4694e.jpg)

### Tables

![0e7c965331d78526ff6906df4e9b942949916b09db351807aa794e1cf4ef18dd.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/tables/0e7c965331d78526ff6906df4e9b942949916b09db351807aa794e1cf4ef18dd.jpg)

![156788e51b3be92b14facdbadeea03c51f57b9a763d61eaa97bfc9953552f608.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/tables/156788e51b3be92b14facdbadeea03c51f57b9a763d61eaa97bfc9953552f608.jpg)

![1efbcd8cebfc8c1a1ab8843c27dd299043203bf35439cb18a48d39a1a5d8bba6.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/tables/1efbcd8cebfc8c1a1ab8843c27dd299043203bf35439cb18a48d39a1a5d8bba6.jpg)

![1f1fb97defb9aba923848d855822b37a2dd5accc6c72b6ca4b9db632137d7601.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/tables/1f1fb97defb9aba923848d855822b37a2dd5accc6c72b6ca4b9db632137d7601.jpg)

![2fa61d93f4bb98822654e78bf35e16b1b671f6aa705f2fc595db8e7648c50455.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/tables/2fa61d93f4bb98822654e78bf35e16b1b671f6aa705f2fc595db8e7648c50455.jpg)

![3106ca98f8e03faa57c539a1fba8d2e8bf352e2ae69ca4783c2ffccd6230f4e0.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/tables/3106ca98f8e03faa57c539a1fba8d2e8bf352e2ae69ca4783c2ffccd6230f4e0.jpg)

![58ee5dedef30c22818299afc1803f51c528a3f6194b0488ad919a3c1bcc73411.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/tables/58ee5dedef30c22818299afc1803f51c528a3f6194b0488ad919a3c1bcc73411.jpg)

![6e94e8a0a823d41ecb4ac3d0bbc9a1d80ccd115e3a39dadf7925979302ef8ce6.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/tables/6e94e8a0a823d41ecb4ac3d0bbc9a1d80ccd115e3a39dadf7925979302ef8ce6.jpg)

![7bbac9bdd1d4267e20c7353bb9b1d1d0539f800744166e84a1e325bcf4d54415.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/tables/7bbac9bdd1d4267e20c7353bb9b1d1d0539f800744166e84a1e325bcf4d54415.jpg)

![909f9d48f2e49c87b9fb1d677f55d61d291ab529753b1acaa7600f61abc61aa5.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/tables/909f9d48f2e49c87b9fb1d677f55d61d291ab529753b1acaa7600f61abc61aa5.jpg)

![c11e41b0c64ac03fba707b83b8acc7f736ea22de1206d6100665db44093fafe3.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/tables/c11e41b0c64ac03fba707b83b8acc7f736ea22de1206d6100665db44093fafe3.jpg)

![f3e4c41ad2c75ea90c0c15a5dcbb20d740d03f3f86a9438d57516f3b9e58cf48.jpg](../iclr_results/2098_GoodDrag_ Towards Good Practices for Drag Editing with Diffusion Models/tables/f3e4c41ad2c75ea90c0c15a5dcbb20d740d03f3f86a9438d57516f3b9e58cf48.jpg)

## Fengbo: a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics


### Images

![2114e48f36318a7ec02e4c9645b72c32cf4f74221b3e42e0b64e11de22be8fa2.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/images/2114e48f36318a7ec02e4c9645b72c32cf4f74221b3e42e0b64e11de22be8fa2.jpg)

![2623a4beb422cb0d0b08198cbf25022cf5af837f7bf03138efac3f9757155054.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/images/2623a4beb422cb0d0b08198cbf25022cf5af837f7bf03138efac3f9757155054.jpg)

![2a29feafba02aa3512950b00d26d6b4aa2552af72b1bdc3f6dbfc70badcc47de.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/images/2a29feafba02aa3512950b00d26d6b4aa2552af72b1bdc3f6dbfc70badcc47de.jpg)

![2e650126278b6138f081207202c4f3f7d3f7eac4c0f000a3e07f2e26de52bc41.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/images/2e650126278b6138f081207202c4f3f7d3f7eac4c0f000a3e07f2e26de52bc41.jpg)

![48963272cec397d0b83ee25ac5a0ec963336e3988e4652369c946407f0a627e5.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/images/48963272cec397d0b83ee25ac5a0ec963336e3988e4652369c946407f0a627e5.jpg)

![5062396606e8363447e5fea64ab9d2842ce469bf5cafa7b0eb628af20f59e9f0.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/images/5062396606e8363447e5fea64ab9d2842ce469bf5cafa7b0eb628af20f59e9f0.jpg)

![5964c3e2585ad8a0010aac2b753f41a6e5bf98884f4efbbaf16f11d87a8269bd.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/images/5964c3e2585ad8a0010aac2b753f41a6e5bf98884f4efbbaf16f11d87a8269bd.jpg)

![69735ad06262b01c182c6aa0982f85e5aa8673027c75f02d0f19a34537b59913.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/images/69735ad06262b01c182c6aa0982f85e5aa8673027c75f02d0f19a34537b59913.jpg)

![70aa6950b1c850072c7e8461d4552f8941984eae3c5288bf30c1d09e9e7fc819.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/images/70aa6950b1c850072c7e8461d4552f8941984eae3c5288bf30c1d09e9e7fc819.jpg)

![75eea582d0e0543dfb2020a8ff71cbd326bb1a160b077d9eb72513c629917d5d.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/images/75eea582d0e0543dfb2020a8ff71cbd326bb1a160b077d9eb72513c629917d5d.jpg)

![8baf9698c6906dc0122a336861f57c1f2bb2935be8f3c289ee9942d9e6ec9e9a.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/images/8baf9698c6906dc0122a336861f57c1f2bb2935be8f3c289ee9942d9e6ec9e9a.jpg)

![8e2d8aa65086daadb741364e416e355e5ca781eed0db4c94d43ca44cddc73a27.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/images/8e2d8aa65086daadb741364e416e355e5ca781eed0db4c94d43ca44cddc73a27.jpg)

![aca243c069589ef26ebf57771b8766099d80629aff2769519b657280226fe9fc.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/images/aca243c069589ef26ebf57771b8766099d80629aff2769519b657280226fe9fc.jpg)

![b3afed1da61a6ded80ad85118a2be9de1f7341e62598c3e9a2e3796a3598a036.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/images/b3afed1da61a6ded80ad85118a2be9de1f7341e62598c3e9a2e3796a3598a036.jpg)

![baa8f57b6110233ca16f32b3ade47a40073ca85ba191c1845c8607f5973fca34.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/images/baa8f57b6110233ca16f32b3ade47a40073ca85ba191c1845c8607f5973fca34.jpg)

![c434f5ff7a1e3dfa1fc0485dc47f59cc327b2552e0a7ff30267be06b65837326.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/images/c434f5ff7a1e3dfa1fc0485dc47f59cc327b2552e0a7ff30267be06b65837326.jpg)

![d388f66200fdf4dd4e623557a5ad11668bf14a09672bf7dea415f1e3dff50c1f.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/images/d388f66200fdf4dd4e623557a5ad11668bf14a09672bf7dea415f1e3dff50c1f.jpg)

![ddd0f0840f3c90109aeacfa6c9acbdf7933d01219dfad86783b3cd95247a544b.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/images/ddd0f0840f3c90109aeacfa6c9acbdf7933d01219dfad86783b3cd95247a544b.jpg)

![e4c5c967cb9ccd24218ce2d0c61317352029798a0bfb83a1f183d9388bf6d708.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/images/e4c5c967cb9ccd24218ce2d0c61317352029798a0bfb83a1f183d9388bf6d708.jpg)

![e7cbdc6bcff2aead9500cfa308d4ea769c6c24bfe513e2ddea84081b5cbd9e15.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/images/e7cbdc6bcff2aead9500cfa308d4ea769c6c24bfe513e2ddea84081b5cbd9e15.jpg)

![e968814c9d86375e30072e333831d6df3d77c6c15d2d08b645dd2e375500505b.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/images/e968814c9d86375e30072e333831d6df3d77c6c15d2d08b645dd2e375500505b.jpg)

### Tables

![08fd071247da09a626e40bfb969762a0657a56e153321a4b098411cd23f25d5c.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/tables/08fd071247da09a626e40bfb969762a0657a56e153321a4b098411cd23f25d5c.jpg)

![463f32a5b4749a7efe9fc2a2691db7ad6fe3a915780faad106e2575d7b97f7ef.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/tables/463f32a5b4749a7efe9fc2a2691db7ad6fe3a915780faad106e2575d7b97f7ef.jpg)

![5dbeafaffb18a73ebce544f15be8289c29a15b5de1c4e0da98b6f34ce6922f10.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/tables/5dbeafaffb18a73ebce544f15be8289c29a15b5de1c4e0da98b6f34ce6922f10.jpg)

![612b56cd83177998ccb0836b5efbd90c88a40974849ba17d9d67955df3ce8e89.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/tables/612b56cd83177998ccb0836b5efbd90c88a40974849ba17d9d67955df3ce8e89.jpg)

![63b2d68d3c4189f85358a739ae5368f0fd994754f863be989b596a0965eff932.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/tables/63b2d68d3c4189f85358a739ae5368f0fd994754f863be989b596a0965eff932.jpg)

![643e9ac42ede1ccba8a3aa5c10d5e5978467c357cabccf484cdf587e2eec7cc3.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/tables/643e9ac42ede1ccba8a3aa5c10d5e5978467c357cabccf484cdf587e2eec7cc3.jpg)

![691fe78f0d0b71d8d5334cb486bfdef1aa1baf269471eab612f50ed12b22e1f0.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/tables/691fe78f0d0b71d8d5334cb486bfdef1aa1baf269471eab612f50ed12b22e1f0.jpg)

![8c30bb032f4b415e9dfd49ea049b9f43e74940ad27ee39bba24dee803a0954e4.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/tables/8c30bb032f4b415e9dfd49ea049b9f43e74940ad27ee39bba24dee803a0954e4.jpg)

![a8f97908a9cfd41a61ae00f7ac045386761a4b25cb6741235c311cc8eef71dfe.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/tables/a8f97908a9cfd41a61ae00f7ac045386761a4b25cb6741235c311cc8eef71dfe.jpg)

![d6e00ce66932e5643a064918f4e86f4120a8c99ca172ee8ba597b387a53ae50f.jpg](../iclr_results/2099_Fengbo_ a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics/tables/d6e00ce66932e5643a064918f4e86f4120a8c99ca172ee8ba597b387a53ae50f.jpg)

## Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data


### Images

![02eb42655282f5d9c1d8594ae5318892a4a0904d4544128450880020b9c6056c.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/02eb42655282f5d9c1d8594ae5318892a4a0904d4544128450880020b9c6056c.jpg)

![0ee20ecc651f9f6738b7119643fb70c86e0ba41f81f9ade8a73a28f6216f33cb.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/0ee20ecc651f9f6738b7119643fb70c86e0ba41f81f9ade8a73a28f6216f33cb.jpg)

![0f04bec76c0d7c5ebbd3bb16d51be8c05eb1319c5ed50f246ed4bafb8833c153.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/0f04bec76c0d7c5ebbd3bb16d51be8c05eb1319c5ed50f246ed4bafb8833c153.jpg)

![10253702f68e29b95db4bffb85b5f2161acb1dad9e0975fe93737fd5a366634a.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/10253702f68e29b95db4bffb85b5f2161acb1dad9e0975fe93737fd5a366634a.jpg)

![12a516e8b750d9c4bd8ccebb7731b18a9afa80fd43d43df9f8da3726c47e335b.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/12a516e8b750d9c4bd8ccebb7731b18a9afa80fd43d43df9f8da3726c47e335b.jpg)

![20de5d706e18cf87afa1219c0831f38f44372a6707ae82f207418d7e2c1203d6.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/20de5d706e18cf87afa1219c0831f38f44372a6707ae82f207418d7e2c1203d6.jpg)

![3651ecb4c71bcbf4a9a55f317071080ef5b01402818dc8f65f9c7a2738f1de9b.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/3651ecb4c71bcbf4a9a55f317071080ef5b01402818dc8f65f9c7a2738f1de9b.jpg)

![36d3752511a3b62c277ddd3676f328742f5bec68c9ce85ec51911c40c974ca21.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/36d3752511a3b62c277ddd3676f328742f5bec68c9ce85ec51911c40c974ca21.jpg)

![3b38ec54f30bd435ac465b3979f7a3b80e295350d88d71e25c011754a650ac64.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/3b38ec54f30bd435ac465b3979f7a3b80e295350d88d71e25c011754a650ac64.jpg)

![442988d0b6ba493b973594ce89f6075748248eca195bdf7e7a9d619cb22fb937.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/442988d0b6ba493b973594ce89f6075748248eca195bdf7e7a9d619cb22fb937.jpg)

![4a69da78aaba3ad5bcab8545bdd07bafe7ce2001084ad55f2521d0c7a743845c.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/4a69da78aaba3ad5bcab8545bdd07bafe7ce2001084ad55f2521d0c7a743845c.jpg)

![4f2d066f32ad4f5f671f89d46a96b4e49557c48750daf7b0733ab478566f5b9d.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/4f2d066f32ad4f5f671f89d46a96b4e49557c48750daf7b0733ab478566f5b9d.jpg)

![5e9345969b5f2cd051b568fcc405a80f8b2ba7081587501b94912fd628c1491e.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/5e9345969b5f2cd051b568fcc405a80f8b2ba7081587501b94912fd628c1491e.jpg)

![5f0400d32132c183bee6fe1307c36a691716f013b8a3db6f3220b27c66cc53d6.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/5f0400d32132c183bee6fe1307c36a691716f013b8a3db6f3220b27c66cc53d6.jpg)

![6196c76fbed4bebd1b27898dbf6f9f1b3068cad4a681f90f203e6debb13061e4.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/6196c76fbed4bebd1b27898dbf6f9f1b3068cad4a681f90f203e6debb13061e4.jpg)

![650548c0da892988a86be0d352acb7051d5b2e4c044475e7084a2d5aeca41ab6.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/650548c0da892988a86be0d352acb7051d5b2e4c044475e7084a2d5aeca41ab6.jpg)

![727d42a8a438b6c5a7c725805af58ae1137ca6aa21b85e2dd9c231ee18861ca4.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/727d42a8a438b6c5a7c725805af58ae1137ca6aa21b85e2dd9c231ee18861ca4.jpg)

![8b71fd29ee9168aac8d31a0464d808811ff344fbb5f26dd613b8fc985cf1fda0.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/8b71fd29ee9168aac8d31a0464d808811ff344fbb5f26dd613b8fc985cf1fda0.jpg)

![8cdfcbdc8cc5f5d39eccb002bd010149f6a36c47263a10c15d5886f9dbe3a311.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/8cdfcbdc8cc5f5d39eccb002bd010149f6a36c47263a10c15d5886f9dbe3a311.jpg)

![8e09fbd633d83256e031122e92f5f56bc003153f2b6d17176e546bd34ca82126.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/8e09fbd633d83256e031122e92f5f56bc003153f2b6d17176e546bd34ca82126.jpg)

![8e8aeafda9e44aad6d66269979010bf13f20cc35bfe003d0f59fdafd6872eab5.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/8e8aeafda9e44aad6d66269979010bf13f20cc35bfe003d0f59fdafd6872eab5.jpg)

![a2cee2906ed71ac19a2f8cc90d720c574895547d6ece9f7b8a930ad05c10b6c1.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/a2cee2906ed71ac19a2f8cc90d720c574895547d6ece9f7b8a930ad05c10b6c1.jpg)

![a34b4770ec6aede738d0f2ea3f0e99617115c1199f9bc61354def2b2dc06c448.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/a34b4770ec6aede738d0f2ea3f0e99617115c1199f9bc61354def2b2dc06c448.jpg)

![a65377db7c78e99dc6f485b34edb0ba8569662fe4e9907c76aa6ac0b8a1ac28a.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/a65377db7c78e99dc6f485b34edb0ba8569662fe4e9907c76aa6ac0b8a1ac28a.jpg)

![bfa4020af19f8e2517386bf35dc46bd075163f1da8437bacf79d419a54cd5a45.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/bfa4020af19f8e2517386bf35dc46bd075163f1da8437bacf79d419a54cd5a45.jpg)

![c0e8d287324dd8a83132d41dbe3fb14353919d89aa7753fd8cbac10377db9def.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/c0e8d287324dd8a83132d41dbe3fb14353919d89aa7753fd8cbac10377db9def.jpg)

![cf0efb3cbe0e424305a25653517ded4663fab847578305fdab4389ee828e0277.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/cf0efb3cbe0e424305a25653517ded4663fab847578305fdab4389ee828e0277.jpg)

![d15722c7743ca152aa7fd2a34cf7d7762976032e9c112b2dc289a29a3b3ddbc5.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/d15722c7743ca152aa7fd2a34cf7d7762976032e9c112b2dc289a29a3b3ddbc5.jpg)

![d2bc0cd1a6cfcfd01f19316055377a134dc81d2f499f92f3cc3f5896e4ae3a74.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/d2bc0cd1a6cfcfd01f19316055377a134dc81d2f499f92f3cc3f5896e4ae3a74.jpg)

![d718666b78bdc6b51e74966614fa4bdbdb85f05e46a761843fbb83e8b9440f10.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/images/d718666b78bdc6b51e74966614fa4bdbdb85f05e46a761843fbb83e8b9440f10.jpg)

### Tables

![64f2b069b3ad54eb86a7196bd648bbb0f33b8782027c8632a7f0fd08c81175eb.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/tables/64f2b069b3ad54eb86a7196bd648bbb0f33b8782027c8632a7f0fd08c81175eb.jpg)

![671e309c83b6669b540f40d892dbd349fe1233ddd2e8388799d816569fa14dbd.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/tables/671e309c83b6669b540f40d892dbd349fe1233ddd2e8388799d816569fa14dbd.jpg)

![b8d45ab384b9e4878e3c0dc94a81460675fefe9e0e28e039253798d4f56473e5.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/tables/b8d45ab384b9e4878e3c0dc94a81460675fefe9e0e28e039253798d4f56473e5.jpg)

![bfddca95ea49bf7133afc4e510cfefe5694145172cfb363863b889fdcdb9257f.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/tables/bfddca95ea49bf7133afc4e510cfefe5694145172cfb363863b889fdcdb9257f.jpg)

![d29a9f22e3425562b91ebcd81a0318868c4a8d29e5321a9f8e1e993a76a30e2a.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/tables/d29a9f22e3425562b91ebcd81a0318868c4a8d29e5321a9f8e1e993a76a30e2a.jpg)

![d947e438516e703ddbc2faef7147efef510d277a1c68d16586db11e9470daf0b.jpg](../iclr_results/2100_Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data/tables/d947e438516e703ddbc2faef7147efef510d277a1c68d16586db11e9470daf0b.jpg)

## NeSyC: A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains


### Images

![209ee797f6ae0d3f5aafd1ff33875b92978ddfca5a5ac20fb377bf0d7aeec891.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/images/209ee797f6ae0d3f5aafd1ff33875b92978ddfca5a5ac20fb377bf0d7aeec891.jpg)

![259b591dbb18a2f7ecffbe769e83807a92db57686b037da96fb1378e60f947eb.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/images/259b591dbb18a2f7ecffbe769e83807a92db57686b037da96fb1378e60f947eb.jpg)

![29c7bd4facd9487bdf2f2ffc9f21746bd10a3caba002b3b5072d860f81d020ac.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/images/29c7bd4facd9487bdf2f2ffc9f21746bd10a3caba002b3b5072d860f81d020ac.jpg)

![2a8bcf7eaa59fad303f29a33ce160f12a90e07862533a03ff5d7ba3ab60c5a70.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/images/2a8bcf7eaa59fad303f29a33ce160f12a90e07862533a03ff5d7ba3ab60c5a70.jpg)

![2a942ff247c0cfc14939a4f7f49ea9cc077e5ab71e209aaa6f69aa9183a3fffb.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/images/2a942ff247c0cfc14939a4f7f49ea9cc077e5ab71e209aaa6f69aa9183a3fffb.jpg)

![2acdc14ae8bbe1faa3bbffe3f3c98ad42636de6f21881244ab9cc857eac21360.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/images/2acdc14ae8bbe1faa3bbffe3f3c98ad42636de6f21881244ab9cc857eac21360.jpg)

![2df2359948a286c816c7f1212eb4efb63f6480f3b5cd40251029e288e112d633.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/images/2df2359948a286c816c7f1212eb4efb63f6480f3b5cd40251029e288e112d633.jpg)

![381ba3f40d3458c3eaca3a73844312fac3efc99dcf747be412230b69675d1860.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/images/381ba3f40d3458c3eaca3a73844312fac3efc99dcf747be412230b69675d1860.jpg)

![5cc7c5836b8d23c5257fd670338291e7b673737a19fb6e2a5d8471eef7adf745.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/images/5cc7c5836b8d23c5257fd670338291e7b673737a19fb6e2a5d8471eef7adf745.jpg)

![5f3158eba14f449d240e40c45e24e38558389df241e0eae07733e7888b5870b6.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/images/5f3158eba14f449d240e40c45e24e38558389df241e0eae07733e7888b5870b6.jpg)

![6e5a277ccf70d41384256cbbe828d9ac59ff7afc5de6e72bc216ac022523c08d.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/images/6e5a277ccf70d41384256cbbe828d9ac59ff7afc5de6e72bc216ac022523c08d.jpg)

![791b0b6fe94d72eede90f2b48d967141508fa360b5a663c11c50f49201df049a.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/images/791b0b6fe94d72eede90f2b48d967141508fa360b5a663c11c50f49201df049a.jpg)

![86692aaff4ccf7c7370dcc514fcaadf18e2379f6d9538da0bdeb60cfdc96349e.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/images/86692aaff4ccf7c7370dcc514fcaadf18e2379f6d9538da0bdeb60cfdc96349e.jpg)

![987d281b3964e18bb03d83ea2147ccb8089dfa1b4d6fd93547c7bc78aa3fc90d.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/images/987d281b3964e18bb03d83ea2147ccb8089dfa1b4d6fd93547c7bc78aa3fc90d.jpg)

![a20842051ec66d64adc8cbea6c3db003890a7b10c97a0832a4397ec7e480842b.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/images/a20842051ec66d64adc8cbea6c3db003890a7b10c97a0832a4397ec7e480842b.jpg)

![d7fa6426aa51e2feb4ae7562fff9f8c2143f6a4681d428cc98656ca0058ab469.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/images/d7fa6426aa51e2feb4ae7562fff9f8c2143f6a4681d428cc98656ca0058ab469.jpg)

![e90abbe6b0a3371df5fe79ad20d99319aa59ef0f7886ff041cd1ed831e7d7528.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/images/e90abbe6b0a3371df5fe79ad20d99319aa59ef0f7886ff041cd1ed831e7d7528.jpg)

![e98f2ba72225e3048b345e20b22474e0c0addda2bd204a7cccde38f9e3e7a55d.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/images/e98f2ba72225e3048b345e20b22474e0c0addda2bd204a7cccde38f9e3e7a55d.jpg)

![fd08d53384ce257c28b67ecd5666f0a84f3c40b352d1870da031a2a5fa015153.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/images/fd08d53384ce257c28b67ecd5666f0a84f3c40b352d1870da031a2a5fa015153.jpg)

### Tables

![0b9473dbf707a2b2bf071608c86ed1199c058fff21d58a02d0c57ced3948d466.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/0b9473dbf707a2b2bf071608c86ed1199c058fff21d58a02d0c57ced3948d466.jpg)

![17b0e0aff6294dc9cfe28fb1c23e6e5edd769e28f329517f026f8609fc4919ee.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/17b0e0aff6294dc9cfe28fb1c23e6e5edd769e28f329517f026f8609fc4919ee.jpg)

![22915ffc86894f5414879498b68fe5cf86b192ef9cefa6928ce4355e7b84165b.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/22915ffc86894f5414879498b68fe5cf86b192ef9cefa6928ce4355e7b84165b.jpg)

![249875ccf557839e04e97cb2dcee275adb209522f7afc2716b3db5d69df9e14c.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/249875ccf557839e04e97cb2dcee275adb209522f7afc2716b3db5d69df9e14c.jpg)

![263565fbecd5a9b2a5b64fbd95c19baa91b366ca1b79093162797df74feaf340.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/263565fbecd5a9b2a5b64fbd95c19baa91b366ca1b79093162797df74feaf340.jpg)

![280d6b92143fc82c1c5248fd3e53c343039a78894bb1a3a8e6fb8ad859062654.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/280d6b92143fc82c1c5248fd3e53c343039a78894bb1a3a8e6fb8ad859062654.jpg)

![372fe9e042a731f0e570fc7e4510b3d9ded5f3a828f2630fe9d8f70bd1e6eb4f.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/372fe9e042a731f0e570fc7e4510b3d9ded5f3a828f2630fe9d8f70bd1e6eb4f.jpg)

![3a75827ed5e908db98e577ee62429dc0c1e299a8c1627a0598291e234e3c3d84.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/3a75827ed5e908db98e577ee62429dc0c1e299a8c1627a0598291e234e3c3d84.jpg)

![4482a63e09b86b076c8cbfb3bc1deaf08878f09bd9e9f1eeeb1676d35c5dd293.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/4482a63e09b86b076c8cbfb3bc1deaf08878f09bd9e9f1eeeb1676d35c5dd293.jpg)

![532e16727d379f9f9e7b9c0eb7e27388d4b0f304ed2d419717061b112087e584.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/532e16727d379f9f9e7b9c0eb7e27388d4b0f304ed2d419717061b112087e584.jpg)

![57912a799837f9fb1c39584a544a07b673a9a3591ae290920c25601029ab6332.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/57912a799837f9fb1c39584a544a07b673a9a3591ae290920c25601029ab6332.jpg)

![65d3cd44a2d2180bb32fa483423efff1e105a4cad421a1cfff7bddb8c6fda56d.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/65d3cd44a2d2180bb32fa483423efff1e105a4cad421a1cfff7bddb8c6fda56d.jpg)

![6f84f6bc0f88e042ff179851acf75ef5e3bb370c43a767e85195f0327db83539.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/6f84f6bc0f88e042ff179851acf75ef5e3bb370c43a767e85195f0327db83539.jpg)

![780a8713334609733d8df79e99624bb8b742f6f940395fb13688a3f0beb661da.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/780a8713334609733d8df79e99624bb8b742f6f940395fb13688a3f0beb661da.jpg)

![79554019340bbbf5a7cb46567248f826307d4b374c6c67a063d02ed9e10e9279.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/79554019340bbbf5a7cb46567248f826307d4b374c6c67a063d02ed9e10e9279.jpg)

![7d0982b7ee5b084461b51249962e9e670d0c143c925a18549aa92c1068ff470b.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/7d0982b7ee5b084461b51249962e9e670d0c143c925a18549aa92c1068ff470b.jpg)

![8e2ee48bfe99ee9f45d7665405bcf0c5144efb380e66ba3c531fb74e405c53c2.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/8e2ee48bfe99ee9f45d7665405bcf0c5144efb380e66ba3c531fb74e405c53c2.jpg)

![9a4c7502d6e72d864d1eb71a2977ab36c76ff82b94bbe23975a2ddf90e64bf7e.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/9a4c7502d6e72d864d1eb71a2977ab36c76ff82b94bbe23975a2ddf90e64bf7e.jpg)

![9dd846ba2102ef2786dcd2da29ff512647db7b42e1e3081a5db56d25705fd5e8.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/9dd846ba2102ef2786dcd2da29ff512647db7b42e1e3081a5db56d25705fd5e8.jpg)

![b27ead9bed93a1713cc629e45cf949d17e220df9e9ebbefe28b79e960679440b.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/b27ead9bed93a1713cc629e45cf949d17e220df9e9ebbefe28b79e960679440b.jpg)

![c34836d81753b551d93a690ab3c7cec8c7d6456a6b2c6350e2db9d057291b812.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/c34836d81753b551d93a690ab3c7cec8c7d6456a6b2c6350e2db9d057291b812.jpg)

![c7d73ad6b5a205166dc4f5c1608ddfb6fa142e5899b2537cc07e605760ed160a.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/c7d73ad6b5a205166dc4f5c1608ddfb6fa142e5899b2537cc07e605760ed160a.jpg)

![d87b65d0dea32909894ef62fa03e3b1c5d72ebbd7aa29a3d631c8d2707e71e27.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/d87b65d0dea32909894ef62fa03e3b1c5d72ebbd7aa29a3d631c8d2707e71e27.jpg)

![eb9fcf670a41a335d42156a86c872246861b15b0b18d16024ff47a8c3586dabc.jpg](../iclr_results/2101_NeSyC_ A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains/tables/eb9fcf670a41a335d42156a86c872246861b15b0b18d16024ff47a8c3586dabc.jpg)

## An Effective Theory of Bias Amplification


### Images

![05d2cefe2e85454f42eabd1054b933cc1e31575d861e8e93b0826079402eb012.jpg](../iclr_results/2102_An Effective Theory of Bias Amplification/images/05d2cefe2e85454f42eabd1054b933cc1e31575d861e8e93b0826079402eb012.jpg)

![05d4949420170fe19336e3d1c8a1a3ca8a19407ffdc36468455b29a94b3f118a.jpg](../iclr_results/2102_An Effective Theory of Bias Amplification/images/05d4949420170fe19336e3d1c8a1a3ca8a19407ffdc36468455b29a94b3f118a.jpg)

![1403dcda7423a07ae74e720b27fd4f7de343a14ebdbae1a9223118d7327c3520.jpg](../iclr_results/2102_An Effective Theory of Bias Amplification/images/1403dcda7423a07ae74e720b27fd4f7de343a14ebdbae1a9223118d7327c3520.jpg)

![1d9e480b557621da33a9259b6298cc9e5b20de8b26d951e6519d340f9a2f6b65.jpg](../iclr_results/2102_An Effective Theory of Bias Amplification/images/1d9e480b557621da33a9259b6298cc9e5b20de8b26d951e6519d340f9a2f6b65.jpg)

![50a50877acdefcc82e0c2c7ece3092daf639fc23dc1c9bf01a1768356cfa12c2.jpg](../iclr_results/2102_An Effective Theory of Bias Amplification/images/50a50877acdefcc82e0c2c7ece3092daf639fc23dc1c9bf01a1768356cfa12c2.jpg)

![6c20ce693daeb2941b5bfa0184f6a00564a23c4a5388589ddf2326a2e3bd7037.jpg](../iclr_results/2102_An Effective Theory of Bias Amplification/images/6c20ce693daeb2941b5bfa0184f6a00564a23c4a5388589ddf2326a2e3bd7037.jpg)

![72c0379ee97d43153809898594f651775dcbb9fed9ffae50b59f3a1450e4cf99.jpg](../iclr_results/2102_An Effective Theory of Bias Amplification/images/72c0379ee97d43153809898594f651775dcbb9fed9ffae50b59f3a1450e4cf99.jpg)

![7c40545576b5e638ae7fda3b1231f59b2bfafb630d2a424b5cd16169f4bdb1d8.jpg](../iclr_results/2102_An Effective Theory of Bias Amplification/images/7c40545576b5e638ae7fda3b1231f59b2bfafb630d2a424b5cd16169f4bdb1d8.jpg)

![8a35bd3bb8e6e17d194786e98582294b7fd1bce2f4e3a34a3adee1ae857f16bb.jpg](../iclr_results/2102_An Effective Theory of Bias Amplification/images/8a35bd3bb8e6e17d194786e98582294b7fd1bce2f4e3a34a3adee1ae857f16bb.jpg)

![a1ee3e7c5f7fc1a7d807c377ba55d1bdadca38f38b632ec5301d51060cc3c5af.jpg](../iclr_results/2102_An Effective Theory of Bias Amplification/images/a1ee3e7c5f7fc1a7d807c377ba55d1bdadca38f38b632ec5301d51060cc3c5af.jpg)

![daaf4bc834d517048fd1272dd5c4047f52877137b00d145e7a430cd421700ae5.jpg](../iclr_results/2102_An Effective Theory of Bias Amplification/images/daaf4bc834d517048fd1272dd5c4047f52877137b00d145e7a430cd421700ae5.jpg)

![e949b05749d0a50de5da412cb498bcc72cc261d97a025ba414d4a1b3d0174e6e.jpg](../iclr_results/2102_An Effective Theory of Bias Amplification/images/e949b05749d0a50de5da412cb498bcc72cc261d97a025ba414d4a1b3d0174e6e.jpg)

![f304b9434ed8091429b0daf714e9b1a97bf1e903976287ed3dd06809b631efed.jpg](../iclr_results/2102_An Effective Theory of Bias Amplification/images/f304b9434ed8091429b0daf714e9b1a97bf1e903976287ed3dd06809b631efed.jpg)

![f51e5637e43473cb1d44c8802f252bf1417c1307ccb7d5ee41c8e72d268cb6b5.jpg](../iclr_results/2102_An Effective Theory of Bias Amplification/images/f51e5637e43473cb1d44c8802f252bf1417c1307ccb7d5ee41c8e72d268cb6b5.jpg)

![f5a61e836684b6eaa72354354073e334bbf345b99a014c522cfbe728f2661d3e.jpg](../iclr_results/2102_An Effective Theory of Bias Amplification/images/f5a61e836684b6eaa72354354073e334bbf345b99a014c522cfbe728f2661d3e.jpg)

## K-HALU: Multiple Answer Korean Hallucination Benchmark for Large Language Models


### Images

![06eb8f07ffb6331cd5b160b09974944b33ae45d48ee6d57734736f445407a02b.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/images/06eb8f07ffb6331cd5b160b09974944b33ae45d48ee6d57734736f445407a02b.jpg)

![201dc9088d7e93186a7f50c858cd24a4019fed54cfa887d0a9db20ec08c181f4.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/images/201dc9088d7e93186a7f50c858cd24a4019fed54cfa887d0a9db20ec08c181f4.jpg)

![27c758d7f9c5781ebcb76788cc173696f34b65ab1c1a11f5e9b6581c250a3455.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/images/27c758d7f9c5781ebcb76788cc173696f34b65ab1c1a11f5e9b6581c250a3455.jpg)

![4f37640a57c498ad51a2594cb6bcfeb13d129ed6853c2303ed562b76e911d43b.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/images/4f37640a57c498ad51a2594cb6bcfeb13d129ed6853c2303ed562b76e911d43b.jpg)

![b238b0fb3b3870cee8bd0174b23ddd8cbbc18b7accf002f2c11cbc23e9bdea1d.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/images/b238b0fb3b3870cee8bd0174b23ddd8cbbc18b7accf002f2c11cbc23e9bdea1d.jpg)

![c653d6d2ccf2564083f4a9b976fbea884d123211af2b451048456bf662a61cae.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/images/c653d6d2ccf2564083f4a9b976fbea884d123211af2b451048456bf662a61cae.jpg)

![ca70296ff400dd3000d69057847d19e66d470f278a52564300206ae688a21377.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/images/ca70296ff400dd3000d69057847d19e66d470f278a52564300206ae688a21377.jpg)

![cbcd6b5a35144cbdbac9762531ae31d542ed7bf42ce273f4d436fd7a47db0b1a.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/images/cbcd6b5a35144cbdbac9762531ae31d542ed7bf42ce273f4d436fd7a47db0b1a.jpg)

### Tables

![12cfc725ed2b8278e921203aae613d67f8cead28f1cf3f96de0621b86977435e.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/tables/12cfc725ed2b8278e921203aae613d67f8cead28f1cf3f96de0621b86977435e.jpg)

![2d1aa393a4acd8d85b6533c58365fa05c0f2baca71b69a3b2e7f5c001d6afc20.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/tables/2d1aa393a4acd8d85b6533c58365fa05c0f2baca71b69a3b2e7f5c001d6afc20.jpg)

![38f01e07397d26ca4257d911655e5d078f950f7baba6857131a2c57450aae1ef.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/tables/38f01e07397d26ca4257d911655e5d078f950f7baba6857131a2c57450aae1ef.jpg)

![407aa4ce4705309b0bf639b8b6fb4f255a266b47e60880b41de795a40cdd1cd9.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/tables/407aa4ce4705309b0bf639b8b6fb4f255a266b47e60880b41de795a40cdd1cd9.jpg)

![42375a7ea4831742d88122dfbb56f8d5b0dce51f540a61229101fb684b2dfb50.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/tables/42375a7ea4831742d88122dfbb56f8d5b0dce51f540a61229101fb684b2dfb50.jpg)

![5565148b8b07895cae67cd64801a47d7a8ef6ad7ce17acd2ec1eca70123257c1.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/tables/5565148b8b07895cae67cd64801a47d7a8ef6ad7ce17acd2ec1eca70123257c1.jpg)

![772a4dbeecb6146caa0bcc2aaefebda44701056a4a7871ad87672985ae149f71.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/tables/772a4dbeecb6146caa0bcc2aaefebda44701056a4a7871ad87672985ae149f71.jpg)

![7a20598f0b4e75dfdd5b1cd6059cedd4b8cd66d328c3f9986c7bac5c55afa93a.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/tables/7a20598f0b4e75dfdd5b1cd6059cedd4b8cd66d328c3f9986c7bac5c55afa93a.jpg)

![b4629477a36a3657c9e1929a805e6a390570f88824efaeed65aa62439b8de315.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/tables/b4629477a36a3657c9e1929a805e6a390570f88824efaeed65aa62439b8de315.jpg)

![b7bfe55fe445516d0242d1c3d7debe6fbb02e2e5127c0bfd1cd5f28e3c668649.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/tables/b7bfe55fe445516d0242d1c3d7debe6fbb02e2e5127c0bfd1cd5f28e3c668649.jpg)

![b86d2acc2778cdace258328c10edc71214bc5773e16efbaf1eb2293e1de22283.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/tables/b86d2acc2778cdace258328c10edc71214bc5773e16efbaf1eb2293e1de22283.jpg)

![dd95058a04f0f95309155a44eb1f997d8191c1e4fbbe0da8ce7662ec1b197465.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/tables/dd95058a04f0f95309155a44eb1f997d8191c1e4fbbe0da8ce7662ec1b197465.jpg)

![e15ccde0115ab85a1c184c80ae5f95a51235f2b24385131f2cc2b44efbe283e4.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/tables/e15ccde0115ab85a1c184c80ae5f95a51235f2b24385131f2cc2b44efbe283e4.jpg)

![e7b2331915d388a2862d685b0a0c9717709a41b6c5506a0eab112cfd8c46be24.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/tables/e7b2331915d388a2862d685b0a0c9717709a41b6c5506a0eab112cfd8c46be24.jpg)

![e87516a021b7ea05ed7d12087bc723d400340c800e3e397d9e7a3317a00a9bc2.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/tables/e87516a021b7ea05ed7d12087bc723d400340c800e3e397d9e7a3317a00a9bc2.jpg)

![e8aaf9be38446cde9d7c72466446ddb1169d36d34223cbf724b7d3795b22a478.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/tables/e8aaf9be38446cde9d7c72466446ddb1169d36d34223cbf724b7d3795b22a478.jpg)

![f2f95ec4eb5ebcd966e23820a5e4cc336a4ae5e7d2fec195092b2fb6b30ef75d.jpg](../iclr_results/2103_K-HALU_ Multiple Answer Korean Hallucination Benchmark for Large Language Models/tables/f2f95ec4eb5ebcd966e23820a5e4cc336a4ae5e7d2fec195092b2fb6b30ef75d.jpg)

## Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning


### Images

![3c8c2e39d077eb44df8248ce585d91b605516602f13764542fd452ef49cfd55b.jpg](../iclr_results/2104_Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning/images/3c8c2e39d077eb44df8248ce585d91b605516602f13764542fd452ef49cfd55b.jpg)

![4385e05f8648284ffad27af6a1451ffd5b9c18dc6d6180abb8c05f0514292237.jpg](../iclr_results/2104_Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning/images/4385e05f8648284ffad27af6a1451ffd5b9c18dc6d6180abb8c05f0514292237.jpg)

![4630477f902fa2246f0742faf39bf5f9314221aab6774adbd1efce1ee21024d6.jpg](../iclr_results/2104_Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning/images/4630477f902fa2246f0742faf39bf5f9314221aab6774adbd1efce1ee21024d6.jpg)

![49e4fdbb3463b234842a28101da5dd91f0d6971d36156a6ada75748c817c2a52.jpg](../iclr_results/2104_Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning/images/49e4fdbb3463b234842a28101da5dd91f0d6971d36156a6ada75748c817c2a52.jpg)

![668654afcba28d2f8e9b12afa11ea12be8fe90a8c017fff876305f3ca776bb7a.jpg](../iclr_results/2104_Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning/images/668654afcba28d2f8e9b12afa11ea12be8fe90a8c017fff876305f3ca776bb7a.jpg)

![ad48d4562542c70450158167e0a6b13bd4abf8e7fa7d548ecc0cfbcef6e7da88.jpg](../iclr_results/2104_Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning/images/ad48d4562542c70450158167e0a6b13bd4abf8e7fa7d548ecc0cfbcef6e7da88.jpg)

![e7ce3c937c597281916cc6d8a5118f984ff1348686d983646a8c0d215d6a3b13.jpg](../iclr_results/2104_Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning/images/e7ce3c937c597281916cc6d8a5118f984ff1348686d983646a8c0d215d6a3b13.jpg)

![f69f1e8408720c44832c5c7d79ecd1d1e6f34f9e775fcfa612c005ada7f0fe6e.jpg](../iclr_results/2104_Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning/images/f69f1e8408720c44832c5c7d79ecd1d1e6f34f9e775fcfa612c005ada7f0fe6e.jpg)

### Tables

![06be75f71c8b1b2df2377776395578bc3d9b06a118d6282edb72cb5b63745ac6.jpg](../iclr_results/2104_Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning/tables/06be75f71c8b1b2df2377776395578bc3d9b06a118d6282edb72cb5b63745ac6.jpg)

![182bf6c0ba4d0d4a78835264f73c996dced602a3d4f15b7abcaa802c8223e4c8.jpg](../iclr_results/2104_Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning/tables/182bf6c0ba4d0d4a78835264f73c996dced602a3d4f15b7abcaa802c8223e4c8.jpg)

![1f0c1c698cb28cd9e15517fd8ae811c32e5f73070dd6c19f4df7f17348cda1c1.jpg](../iclr_results/2104_Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning/tables/1f0c1c698cb28cd9e15517fd8ae811c32e5f73070dd6c19f4df7f17348cda1c1.jpg)

![24e53ef31a6d512164a814ef05cec459008570600b2274aee9b7fc1f88a7f60b.jpg](../iclr_results/2104_Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning/tables/24e53ef31a6d512164a814ef05cec459008570600b2274aee9b7fc1f88a7f60b.jpg)

![413358f4f9dd0888ab00754e4312501027544c119fc06186bb97693884f651c1.jpg](../iclr_results/2104_Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning/tables/413358f4f9dd0888ab00754e4312501027544c119fc06186bb97693884f651c1.jpg)

![575e9e3ea4d5c2719d3fcf03eae0672d1dd0b6715386dec0c5b5d14ece460509.jpg](../iclr_results/2104_Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning/tables/575e9e3ea4d5c2719d3fcf03eae0672d1dd0b6715386dec0c5b5d14ece460509.jpg)

![991f9686c02c787745cf549edc6b818b4a267600ef1e6390474470800a7260cb.jpg](../iclr_results/2104_Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning/tables/991f9686c02c787745cf549edc6b818b4a267600ef1e6390474470800a7260cb.jpg)

![a15b763469857d131be477e8df3b039a61a2db82f27407e72d9b49c46221779a.jpg](../iclr_results/2104_Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning/tables/a15b763469857d131be477e8df3b039a61a2db82f27407e72d9b49c46221779a.jpg)

![afb1a6b5c892e8621000b0a2d26e1a7a4e419b2aa8e9442a165ceb8e43a6ece1.jpg](../iclr_results/2104_Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning/tables/afb1a6b5c892e8621000b0a2d26e1a7a4e419b2aa8e9442a165ceb8e43a6ece1.jpg)

![d0bc9d69b918b1666a4c4621ba19e34b59857a4a5d6f19a71d0b447722c18fbe.jpg](../iclr_results/2104_Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning/tables/d0bc9d69b918b1666a4c4621ba19e34b59857a4a5d6f19a71d0b447722c18fbe.jpg)

![d71702752a578594a1fbd1a8eb6e75b699096560903d63be906190acaa913627.jpg](../iclr_results/2104_Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning/tables/d71702752a578594a1fbd1a8eb6e75b699096560903d63be906190acaa913627.jpg)

![da107de198aa1ade6c4a1da8b07c67a910b134a82a7fc67562196c5469501e1b.jpg](../iclr_results/2104_Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning/tables/da107de198aa1ade6c4a1da8b07c67a910b134a82a7fc67562196c5469501e1b.jpg)

![ee3b0ba55c2e51059d7944b336b39c08e8a83e30d79d892f17b43530db82dc90.jpg](../iclr_results/2104_Frequency-Guided Masking for Enhanced Vision Self-Supervised Learning/tables/ee3b0ba55c2e51059d7944b336b39c08e8a83e30d79d892f17b43530db82dc90.jpg)

## Correlation and Navigation in the Vocabulary Key Representation Space of Language Models


### Images

![06529462ce2f25f467d3422f1ac5b39d6f17ea7d27d0fc010484b76614463285.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/images/06529462ce2f25f467d3422f1ac5b39d6f17ea7d27d0fc010484b76614463285.jpg)

![0805380e4c8438b0d761a9187544e33f36f80dd0eb2894bc243a6eefc79a7cd0.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/images/0805380e4c8438b0d761a9187544e33f36f80dd0eb2894bc243a6eefc79a7cd0.jpg)

![57176d22b20d9046bdc5e4c0284c0f768d56b8386eb4a6ac1e2ce3b53f35355f.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/images/57176d22b20d9046bdc5e4c0284c0f768d56b8386eb4a6ac1e2ce3b53f35355f.jpg)

![6d9893ef3704878e4da07da84bbd5404894522997678b33c3d7ad7850d44ab77.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/images/6d9893ef3704878e4da07da84bbd5404894522997678b33c3d7ad7850d44ab77.jpg)

![70ac080be13732d1d7b04ad87f6dc1609ac991aa972b2cb4abae460824aec904.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/images/70ac080be13732d1d7b04ad87f6dc1609ac991aa972b2cb4abae460824aec904.jpg)

![75a98fedbb832e641b3d6826a482d3f80b0ea30ab9b0e50112a5c469dc7772b0.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/images/75a98fedbb832e641b3d6826a482d3f80b0ea30ab9b0e50112a5c469dc7772b0.jpg)

![9d9fd7e21f9bd342cc9860f7b35da861c1464ea46888047be5caf6485102b111.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/images/9d9fd7e21f9bd342cc9860f7b35da861c1464ea46888047be5caf6485102b111.jpg)

![a180ddcb55c3fb422ba8c3f981c8927a85049837bf7d647d34cf9d6013c80019.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/images/a180ddcb55c3fb422ba8c3f981c8927a85049837bf7d647d34cf9d6013c80019.jpg)

![c1794cd5ad39fdd0c2a2e5087217d905520f2b091e3af50e4f3534eccdddb128.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/images/c1794cd5ad39fdd0c2a2e5087217d905520f2b091e3af50e4f3534eccdddb128.jpg)

![c81a1fc0531db6e490cb363f8207c5a4dd1e0883d041a7e372df6749b351d058.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/images/c81a1fc0531db6e490cb363f8207c5a4dd1e0883d041a7e372df6749b351d058.jpg)

![c8d4c9edfe129cce73ce92c158cd4b0002a7fd93bb72aae0be5c3853e93566be.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/images/c8d4c9edfe129cce73ce92c158cd4b0002a7fd93bb72aae0be5c3853e93566be.jpg)

![ea1fcc30029e3224ddb3cde27f913a699d981a926075dcf5608d12aab7594fe4.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/images/ea1fcc30029e3224ddb3cde27f913a699d981a926075dcf5608d12aab7594fe4.jpg)

### Tables

![10fdc2e11c767da116cd2bcc7b53063014c92fa4f130e145ad636de24a3a3eee.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/tables/10fdc2e11c767da116cd2bcc7b53063014c92fa4f130e145ad636de24a3a3eee.jpg)

![231942bd0d8ed4973064be46689ad5f7c4558f2687b3595f5ec14738dc363a8b.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/tables/231942bd0d8ed4973064be46689ad5f7c4558f2687b3595f5ec14738dc363a8b.jpg)

![2fc2eb1f097a5ed14292da447a1173c796b19fb0951922c9fe1e435e53ffad27.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/tables/2fc2eb1f097a5ed14292da447a1173c796b19fb0951922c9fe1e435e53ffad27.jpg)

![375d288663f4e61477ab5d0ad8c580b25ec869e3d1bae7e8f86ba6ec0e2cdb82.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/tables/375d288663f4e61477ab5d0ad8c580b25ec869e3d1bae7e8f86ba6ec0e2cdb82.jpg)

![3887403c069a9445e13f1ded13fc21c2e068d4590bb56b80522f7b2cb5c34f93.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/tables/3887403c069a9445e13f1ded13fc21c2e068d4590bb56b80522f7b2cb5c34f93.jpg)

![4b6dc0a410f97e89cb87d24afe5e6b86430f915ab1c5d03df36956109d304ffa.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/tables/4b6dc0a410f97e89cb87d24afe5e6b86430f915ab1c5d03df36956109d304ffa.jpg)

![583de4bd52ca22c9803fc4c74604eda7bfa123821c96c7fe82cfb7dfbb9f6d1c.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/tables/583de4bd52ca22c9803fc4c74604eda7bfa123821c96c7fe82cfb7dfbb9f6d1c.jpg)

![62550c6ab11895be6a84bbda8a72b9a7d55eab93617d3aa428b17c8f8d8f9b08.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/tables/62550c6ab11895be6a84bbda8a72b9a7d55eab93617d3aa428b17c8f8d8f9b08.jpg)

![6d2cf2f822322536c0b080625fe9dfa9874ca9544a493658ac3cae49ab0c7113.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/tables/6d2cf2f822322536c0b080625fe9dfa9874ca9544a493658ac3cae49ab0c7113.jpg)

![87802ab23c7941a14aa6a1435ce843ae54b3cdbd58c85d29ed85ebdb769475cf.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/tables/87802ab23c7941a14aa6a1435ce843ae54b3cdbd58c85d29ed85ebdb769475cf.jpg)

![8c9bb64ee62782fc268441bce4e8eedf613d9fd0fa8fe578c194df647f100f02.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/tables/8c9bb64ee62782fc268441bce4e8eedf613d9fd0fa8fe578c194df647f100f02.jpg)

![a3ff2df147830dfe2465aa2217d733122f1ee6f75031cfd4f32ce98dd262a26c.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/tables/a3ff2df147830dfe2465aa2217d733122f1ee6f75031cfd4f32ce98dd262a26c.jpg)

![b5bdb764c52883a30c1e8bc3b40700e2cf8c431e88458241eb6b31bf9756cf0e.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/tables/b5bdb764c52883a30c1e8bc3b40700e2cf8c431e88458241eb6b31bf9756cf0e.jpg)

![b974bc4afa10b0adbac618d343f58c36cce7946acaadff9b458fa79e5d11e21b.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/tables/b974bc4afa10b0adbac618d343f58c36cce7946acaadff9b458fa79e5d11e21b.jpg)

![c8a7133ffbba46c01f12dfc8ab3733ddd546d96fc98dba92fde5f869b3661e49.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/tables/c8a7133ffbba46c01f12dfc8ab3733ddd546d96fc98dba92fde5f869b3661e49.jpg)

![d83893da24dffdb9a01cadcaf31f5eea5a6daf40142629ddbb01d1c9c96c4f19.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/tables/d83893da24dffdb9a01cadcaf31f5eea5a6daf40142629ddbb01d1c9c96c4f19.jpg)

![edcfa82b5196fe4a79d0abe85c1be063ccb4b51f4b6130a3d65d7787bc49408a.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/tables/edcfa82b5196fe4a79d0abe85c1be063ccb4b51f4b6130a3d65d7787bc49408a.jpg)

![f80a9d42ea77d454a47071cc3928ab889cffe783d2e8fbfe98e22fc127fdea37.jpg](../iclr_results/2105_Correlation and Navigation in the Vocabulary Key Representation Space of Language Models/tables/f80a9d42ea77d454a47071cc3928ab889cffe783d2e8fbfe98e22fc127fdea37.jpg)

## CREAM: Consistency Regularized Self-Rewarding Language Models


### Images

![0d4a40d2c1eed235663362d87d2a0afbe95baa4e8a7cff295d38205b9e5c1485.jpg](../iclr_results/2106_CREAM_ Consistency Regularized Self-Rewarding Language Models/images/0d4a40d2c1eed235663362d87d2a0afbe95baa4e8a7cff295d38205b9e5c1485.jpg)

![3dea8315fc29759b1b40b9b3511c8cf2e4981f6de79ea3527d16c1b11d67198d.jpg](../iclr_results/2106_CREAM_ Consistency Regularized Self-Rewarding Language Models/images/3dea8315fc29759b1b40b9b3511c8cf2e4981f6de79ea3527d16c1b11d67198d.jpg)

![693ebe4f9c0b64699ac54577e049c998209f127b84595669cd45dd577c5e9433.jpg](../iclr_results/2106_CREAM_ Consistency Regularized Self-Rewarding Language Models/images/693ebe4f9c0b64699ac54577e049c998209f127b84595669cd45dd577c5e9433.jpg)

![78d1da9db2dd3285b9f7589e97a9f8f580061727376696fe90a3863782b9d61d.jpg](../iclr_results/2106_CREAM_ Consistency Regularized Self-Rewarding Language Models/images/78d1da9db2dd3285b9f7589e97a9f8f580061727376696fe90a3863782b9d61d.jpg)

![b99b38be3a8d65ac835574e762507c17fa46094920c33caddfb45cbbfb4c7e9c.jpg](../iclr_results/2106_CREAM_ Consistency Regularized Self-Rewarding Language Models/images/b99b38be3a8d65ac835574e762507c17fa46094920c33caddfb45cbbfb4c7e9c.jpg)

### Tables

![0cad54e179c546cb30288b715c1b76e4bbee906bd8d1605a02e58af06f2e24c8.jpg](../iclr_results/2106_CREAM_ Consistency Regularized Self-Rewarding Language Models/tables/0cad54e179c546cb30288b715c1b76e4bbee906bd8d1605a02e58af06f2e24c8.jpg)

![10052af662aac70525618be670cf36334f808ff3adfce59716df78be3a2578b4.jpg](../iclr_results/2106_CREAM_ Consistency Regularized Self-Rewarding Language Models/tables/10052af662aac70525618be670cf36334f808ff3adfce59716df78be3a2578b4.jpg)

![2dafbaa6fc46cb12db42a770677f2c5c7b971968b147a411cf91367766a0be98.jpg](../iclr_results/2106_CREAM_ Consistency Regularized Self-Rewarding Language Models/tables/2dafbaa6fc46cb12db42a770677f2c5c7b971968b147a411cf91367766a0be98.jpg)

![31144efd2ebae168822da40d55d4ef883dc42d25549a49ebf2fc4014e5403e65.jpg](../iclr_results/2106_CREAM_ Consistency Regularized Self-Rewarding Language Models/tables/31144efd2ebae168822da40d55d4ef883dc42d25549a49ebf2fc4014e5403e65.jpg)

![4c735a0155d7375f50670c05fc3962209e6f58bcaf0272faf8cf9bb0e444f2b7.jpg](../iclr_results/2106_CREAM_ Consistency Regularized Self-Rewarding Language Models/tables/4c735a0155d7375f50670c05fc3962209e6f58bcaf0272faf8cf9bb0e444f2b7.jpg)

![597fecff91ad6eca675d3e702d776968762d62b72a33ec2b5ca5587646b753c3.jpg](../iclr_results/2106_CREAM_ Consistency Regularized Self-Rewarding Language Models/tables/597fecff91ad6eca675d3e702d776968762d62b72a33ec2b5ca5587646b753c3.jpg)

![82305e25f830f578e2d8516a2a6bbe9b37e74ec24a5205cd7b48ad345198ca3a.jpg](../iclr_results/2106_CREAM_ Consistency Regularized Self-Rewarding Language Models/tables/82305e25f830f578e2d8516a2a6bbe9b37e74ec24a5205cd7b48ad345198ca3a.jpg)

![82b619890f4db809c2b39469386aa09a386282e85dc9a4ca4fa0ba6391e21187.jpg](../iclr_results/2106_CREAM_ Consistency Regularized Self-Rewarding Language Models/tables/82b619890f4db809c2b39469386aa09a386282e85dc9a4ca4fa0ba6391e21187.jpg)

![972514ee67619265189ade21914d754b47ff1549c7bebf88eb77ab8a94542f44.jpg](../iclr_results/2106_CREAM_ Consistency Regularized Self-Rewarding Language Models/tables/972514ee67619265189ade21914d754b47ff1549c7bebf88eb77ab8a94542f44.jpg)

![ad893da4ce666c926fb2bd83efbc8e04646ce05b63acee8cee74da46fffe6cbf.jpg](../iclr_results/2106_CREAM_ Consistency Regularized Self-Rewarding Language Models/tables/ad893da4ce666c926fb2bd83efbc8e04646ce05b63acee8cee74da46fffe6cbf.jpg)

![b0772db298d4fd9f80cc6cd4c76b52a729519345104d1295c059b3821e56d248.jpg](../iclr_results/2106_CREAM_ Consistency Regularized Self-Rewarding Language Models/tables/b0772db298d4fd9f80cc6cd4c76b52a729519345104d1295c059b3821e56d248.jpg)

![bbb26908d5b62e8ab2387b05ab708b8b02cbf5943bda226dbd0030f8b785b9f6.jpg](../iclr_results/2106_CREAM_ Consistency Regularized Self-Rewarding Language Models/tables/bbb26908d5b62e8ab2387b05ab708b8b02cbf5943bda226dbd0030f8b785b9f6.jpg)

![fe7037b70f807b832985a7cc88ca86761d9e85cc3b29ce52a2d9a929437ca787.jpg](../iclr_results/2106_CREAM_ Consistency Regularized Self-Rewarding Language Models/tables/fe7037b70f807b832985a7cc88ca86761d9e85cc3b29ce52a2d9a929437ca787.jpg)

## Universal Sharpness Dynamics in Neural Network Training: Fixed Point Analysis, Edge of Stability, and Route to Chaos


### Images

![02b11e417d60fa5328ca96aa0cf7829b7b58c037493b4730e2c76348a64de4d7.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/02b11e417d60fa5328ca96aa0cf7829b7b58c037493b4730e2c76348a64de4d7.jpg)

![04124ea4c6af0d050e26745e6863a895a758e4541135ac4652227dc64ce0d158.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/04124ea4c6af0d050e26745e6863a895a758e4541135ac4652227dc64ce0d158.jpg)

![0a267db768783731aa42c167247b909527e84d6df200c05becfa3d5c1200db27.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/0a267db768783731aa42c167247b909527e84d6df200c05becfa3d5c1200db27.jpg)

![157df855ca77dca81117dfcd2ca91e91a9a66451ef46fe81cbcae81ff5ab7e9e.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/157df855ca77dca81117dfcd2ca91e91a9a66451ef46fe81cbcae81ff5ab7e9e.jpg)

![1dfc0cb9ab9b1ae636fbb6a60948cf4359ff2bf252a545dc6a6d0cc5f76b399f.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/1dfc0cb9ab9b1ae636fbb6a60948cf4359ff2bf252a545dc6a6d0cc5f76b399f.jpg)

![247db7693d94ffe1b465dddf918145c951ec486c459eff9bad69df545665caee.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/247db7693d94ffe1b465dddf918145c951ec486c459eff9bad69df545665caee.jpg)

![2edf18912c00e11b4ffff2f0a8b84fbc9bab8f088ec05baa8a2457c1a4610cf2.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/2edf18912c00e11b4ffff2f0a8b84fbc9bab8f088ec05baa8a2457c1a4610cf2.jpg)

![2fb82b27733d073f08c8115bcab3423760539e294d6b282e0421bdf94e63e143.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/2fb82b27733d073f08c8115bcab3423760539e294d6b282e0421bdf94e63e143.jpg)

![324f85d0c4731d86a316a6d34d3688392e67c99b75d2f4672199fa2ad42cc73b.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/324f85d0c4731d86a316a6d34d3688392e67c99b75d2f4672199fa2ad42cc73b.jpg)

![3f9ba23dd9b5aa2a7941f9a1f14d6791b97e6e8a9378e34d5022da97fa8496c6.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/3f9ba23dd9b5aa2a7941f9a1f14d6791b97e6e8a9378e34d5022da97fa8496c6.jpg)

![4c79dd7c78d8f24ee21c5ed22631b05648e9aac44d6e2d8807bdc2660bf9295e.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/4c79dd7c78d8f24ee21c5ed22631b05648e9aac44d6e2d8807bdc2660bf9295e.jpg)

![5649e2750bbdd26eb564f0192086cd6f72b2f6922ded3200d3339ac24305d2b3.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/5649e2750bbdd26eb564f0192086cd6f72b2f6922ded3200d3339ac24305d2b3.jpg)

![5988da94dfcbf39b09adba32a0ec849598644ab04ad6842575cea9c48d89c40e.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/5988da94dfcbf39b09adba32a0ec849598644ab04ad6842575cea9c48d89c40e.jpg)

![68d7271424f1df31b4b2087ca920e902bab6a118a73a4d35181f6c9a89c5fdf3.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/68d7271424f1df31b4b2087ca920e902bab6a118a73a4d35181f6c9a89c5fdf3.jpg)

![75f9562667475d6a7a82eea814a94bfa76ea6bf634e718c8ce6c69912de7306a.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/75f9562667475d6a7a82eea814a94bfa76ea6bf634e718c8ce6c69912de7306a.jpg)

![7fc6848b3185fa795f1cdbdd2439f28dec7be48c6259f3c3e09aeff0c9e7e737.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/7fc6848b3185fa795f1cdbdd2439f28dec7be48c6259f3c3e09aeff0c9e7e737.jpg)

![87857bd4a78212492b6734d8aa73247b576dbc6918bad95f4bca035464669ac5.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/87857bd4a78212492b6734d8aa73247b576dbc6918bad95f4bca035464669ac5.jpg)

![8f8406a655231fc879a1ac7cfa438641e489432857d50152a3c6c065b9535129.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/8f8406a655231fc879a1ac7cfa438641e489432857d50152a3c6c065b9535129.jpg)

![978a040c0c49d7902b68525e839247f913236891b6b85bac62e0f2375dd9c1b5.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/978a040c0c49d7902b68525e839247f913236891b6b85bac62e0f2375dd9c1b5.jpg)

![9b9294b9c7a1b618aae346ba61d03edcf35d498ba3f3b31a679539cf581aeb3f.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/9b9294b9c7a1b618aae346ba61d03edcf35d498ba3f3b31a679539cf581aeb3f.jpg)

![9dbbd88e68206c693c143025b1b5967d3b68528808c05a45bbe8e2a8cb9b2dff.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/9dbbd88e68206c693c143025b1b5967d3b68528808c05a45bbe8e2a8cb9b2dff.jpg)

![a6770f62757cc2ee4edc07f0afd0dd0e391e2683c27ad6ac9ac32d0c08300f16.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/a6770f62757cc2ee4edc07f0afd0dd0e391e2683c27ad6ac9ac32d0c08300f16.jpg)

![a8d43677665c363e1df3a86cdc0352fb6b2a6805f91d98d4b60472408020af36.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/a8d43677665c363e1df3a86cdc0352fb6b2a6805f91d98d4b60472408020af36.jpg)

![b65223cc415f6ec436747ea096857fe33ff8214d879b54ed38ee236a1f21f851.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/b65223cc415f6ec436747ea096857fe33ff8214d879b54ed38ee236a1f21f851.jpg)

![b6c8c3e2652ee00ce08c4f8e876fa965c049766f829986d2baa13d982f652e37.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/b6c8c3e2652ee00ce08c4f8e876fa965c049766f829986d2baa13d982f652e37.jpg)

![c72a3719e708a30e2f63380efffc2adc5a18d90d079df64a970e5933b5d36abe.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/c72a3719e708a30e2f63380efffc2adc5a18d90d079df64a970e5933b5d36abe.jpg)

![ca5c85aeffb6515f1f11a5e4ad32b015e4e0b2911c8e5d7bf50a4532ec840f90.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/ca5c85aeffb6515f1f11a5e4ad32b015e4e0b2911c8e5d7bf50a4532ec840f90.jpg)

![dd297793d6cc1af48075890f6151f335c3cf107f610fe274dc1fc7106b306a5c.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/dd297793d6cc1af48075890f6151f335c3cf107f610fe274dc1fc7106b306a5c.jpg)

![e09ef02cd18d944b82ad65101b2f48a9b01ae8b098c9f2ad508169fe3ece2fa6.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/e09ef02cd18d944b82ad65101b2f48a9b01ae8b098c9f2ad508169fe3ece2fa6.jpg)

![e0fde929ebebe06290d2598d28e483acc74eb9c6f40923fc97a0d34d926a5718.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/e0fde929ebebe06290d2598d28e483acc74eb9c6f40923fc97a0d34d926a5718.jpg)

![eee384eda4c9b3e1b3073a6a55570bae3ee0aced7c130eaf0b6c3d7c282895b5.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/eee384eda4c9b3e1b3073a6a55570bae3ee0aced7c130eaf0b6c3d7c282895b5.jpg)

![fa9d950cc6a98b81af68b14530228e457f4f29c9e3a0fa8d2dd6b776aceaba8e.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/images/fa9d950cc6a98b81af68b14530228e457f4f29c9e3a0fa8d2dd6b776aceaba8e.jpg)

### Tables

![f305ff8cbda703ca4298571aef9c6070be09d392182907fce3a588f54c562fdb.jpg](../iclr_results/2107_Universal Sharpness Dynamics in Neural Network Training_ Fixed Point Analysis, Edge of Stability, an/tables/f305ff8cbda703ca4298571aef9c6070be09d392182907fce3a588f54c562fdb.jpg)

## ObscuraCoder: Powering Efficient Code LM Pre-Training Via Obfuscation Grounding


### Images

![219c9d37324b3f06ef478659098c9227420364ea2b20796b85abf4aa852e1bf7.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/images/219c9d37324b3f06ef478659098c9227420364ea2b20796b85abf4aa852e1bf7.jpg)

![2a1d4c4acf071270b2f601b5bfb71c2c9d36b095b7ee6d4e8e7ff9a302d86961.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/images/2a1d4c4acf071270b2f601b5bfb71c2c9d36b095b7ee6d4e8e7ff9a302d86961.jpg)

![411e9ccce6de1d7ea9103a268e8b62f594007ed3246cbb47023709b7c6c243f3.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/images/411e9ccce6de1d7ea9103a268e8b62f594007ed3246cbb47023709b7c6c243f3.jpg)

![638a01b61cf79fb61d48f47d1aa4f23c358b5a6c776a5e13278f6f5d39bff9fe.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/images/638a01b61cf79fb61d48f47d1aa4f23c358b5a6c776a5e13278f6f5d39bff9fe.jpg)

![93729562602af69fc33ecf8a9f6ecbc428fe91dacdb9a40ee94dfeb40b879e5b.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/images/93729562602af69fc33ecf8a9f6ecbc428fe91dacdb9a40ee94dfeb40b879e5b.jpg)

![a60d0f61ad493d154ea4c83abc8f1cfa024d7b20132d61a7e8c406125ec9103b.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/images/a60d0f61ad493d154ea4c83abc8f1cfa024d7b20132d61a7e8c406125ec9103b.jpg)

![a914569c66ffe7d13219046ca9cc707d388a71087bc26b65b809f53c9260f8dd.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/images/a914569c66ffe7d13219046ca9cc707d388a71087bc26b65b809f53c9260f8dd.jpg)

![dc3f01a5c2d34c7167dc9e1df30a9cf74a11e8924599c3ab1723565c616dc9e6.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/images/dc3f01a5c2d34c7167dc9e1df30a9cf74a11e8924599c3ab1723565c616dc9e6.jpg)

![e58774e135a4373ecd76b7bb70d800742e64fe931d1fcfaaf387c1c2025f4c43.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/images/e58774e135a4373ecd76b7bb70d800742e64fe931d1fcfaaf387c1c2025f4c43.jpg)

### Tables

![09030c8cf6ba27e76a4d51b6ec82e57d32c58ce2107068ff71b5cb65c0d418cf.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/tables/09030c8cf6ba27e76a4d51b6ec82e57d32c58ce2107068ff71b5cb65c0d418cf.jpg)

![0baa0d77e884f24aa129950cab9c89da9caf9836d6eba5fed77c8175a6e6b809.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/tables/0baa0d77e884f24aa129950cab9c89da9caf9836d6eba5fed77c8175a6e6b809.jpg)

![33f030e651b6f1270005a57b6bbae9013fc5a3037140aff048d246b328be3a01.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/tables/33f030e651b6f1270005a57b6bbae9013fc5a3037140aff048d246b328be3a01.jpg)

![6c1e233876a613cfb0a56a96b32d8bb0fa866404955a35ac933fb78802ea0fab.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/tables/6c1e233876a613cfb0a56a96b32d8bb0fa866404955a35ac933fb78802ea0fab.jpg)

![705060115f1a8eb072c38cb6e28e2eef80d829739a8008d10eadc2492d383458.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/tables/705060115f1a8eb072c38cb6e28e2eef80d829739a8008d10eadc2492d383458.jpg)

![78c4f7c62ecf33d3badc134f019f4943d84ce713dca0835adab31928bebd34f7.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/tables/78c4f7c62ecf33d3badc134f019f4943d84ce713dca0835adab31928bebd34f7.jpg)

![8d70f9a8b6585ad984e7b9f680ca66b5bfbb4bbcf41ef173d773103657f9b632.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/tables/8d70f9a8b6585ad984e7b9f680ca66b5bfbb4bbcf41ef173d773103657f9b632.jpg)

![a9aa2dac3b106b0589e5b9914f64168f896344c3e2ca8dc26422ad50658188ea.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/tables/a9aa2dac3b106b0589e5b9914f64168f896344c3e2ca8dc26422ad50658188ea.jpg)

![afd342a7f435d07ba7f874d8f5e6b250ddf6a414bb383aeab9ddc150e6b37697.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/tables/afd342a7f435d07ba7f874d8f5e6b250ddf6a414bb383aeab9ddc150e6b37697.jpg)

![b429a9a48e43ffae0fb29f472d1a6ac5a351acec370832dea08fa4a4e655688e.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/tables/b429a9a48e43ffae0fb29f472d1a6ac5a351acec370832dea08fa4a4e655688e.jpg)

![d1820b46ca0ddbedef43b08b34461cb2d31476ba9afecea7bf274303b9b4a545.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/tables/d1820b46ca0ddbedef43b08b34461cb2d31476ba9afecea7bf274303b9b4a545.jpg)

![d79a31b10bfb71e6d3b881a98bd3b464b246cb3d4c8f71eb6bb6488f32198776.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/tables/d79a31b10bfb71e6d3b881a98bd3b464b246cb3d4c8f71eb6bb6488f32198776.jpg)

![d985085258bae011f6e2a50812f5a4e5309e199e9054c835c8dc9869aac7fb4b.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/tables/d985085258bae011f6e2a50812f5a4e5309e199e9054c835c8dc9869aac7fb4b.jpg)

![e5b65f8b047b1815b1e3aacd44fc5c7052f35dfaf761bf6cb01ae5fd3f8fc4ef.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/tables/e5b65f8b047b1815b1e3aacd44fc5c7052f35dfaf761bf6cb01ae5fd3f8fc4ef.jpg)

![ed69d2c992f6f85c39d2ae5f9a0313ef40ae50147d129d7488fc75348cddc85a.jpg](../iclr_results/2108_ObscuraCoder_ Powering Efficient Code LM Pre-Training Via Obfuscation Grounding/tables/ed69d2c992f6f85c39d2ae5f9a0313ef40ae50147d129d7488fc75348cddc85a.jpg)

## MrT5: Dynamic Token Merging for Efficient Byte-level Language Models


### Images

![08f58ad91052440ecf426b98420090bcd9dc4eea7ff08cd17fa4affe7110a447.jpg](../iclr_results/2109_MrT5_ Dynamic Token Merging for Efficient Byte-level Language Models/images/08f58ad91052440ecf426b98420090bcd9dc4eea7ff08cd17fa4affe7110a447.jpg)

![6997dc7d60eec634c140b3c8be65b6fe7e2df5591a41a89ba3a9a98fcbd11e34.jpg](../iclr_results/2109_MrT5_ Dynamic Token Merging for Efficient Byte-level Language Models/images/6997dc7d60eec634c140b3c8be65b6fe7e2df5591a41a89ba3a9a98fcbd11e34.jpg)

![736131753830dab37fb03a45d2aa922a7d6699483d3822523f1d6f4fa51e144b.jpg](../iclr_results/2109_MrT5_ Dynamic Token Merging for Efficient Byte-level Language Models/images/736131753830dab37fb03a45d2aa922a7d6699483d3822523f1d6f4fa51e144b.jpg)

![86cd8385c9f813188b27b08a4445b891a9648d1738bc4016712df03cac03a18d.jpg](../iclr_results/2109_MrT5_ Dynamic Token Merging for Efficient Byte-level Language Models/images/86cd8385c9f813188b27b08a4445b891a9648d1738bc4016712df03cac03a18d.jpg)

![9840d8e82a0ae7327c522751f3c4876fc064c55aadd80f801b37adc933ee8995.jpg](../iclr_results/2109_MrT5_ Dynamic Token Merging for Efficient Byte-level Language Models/images/9840d8e82a0ae7327c522751f3c4876fc064c55aadd80f801b37adc933ee8995.jpg)

![b2bffe1d6f3e09525a477ea5f6f3b0465dc2d5732277fdad7fa6f1174d0e7587.jpg](../iclr_results/2109_MrT5_ Dynamic Token Merging for Efficient Byte-level Language Models/images/b2bffe1d6f3e09525a477ea5f6f3b0465dc2d5732277fdad7fa6f1174d0e7587.jpg)

![b723737c33245adfa8f3f64cadda8471a5aa061bb2ef38ae2a82b47c5c9708e8.jpg](../iclr_results/2109_MrT5_ Dynamic Token Merging for Efficient Byte-level Language Models/images/b723737c33245adfa8f3f64cadda8471a5aa061bb2ef38ae2a82b47c5c9708e8.jpg)

![f0c50eb1d2ce44962e260764add148c8eb88944df228c67f18eef690b1e3246e.jpg](../iclr_results/2109_MrT5_ Dynamic Token Merging for Efficient Byte-level Language Models/images/f0c50eb1d2ce44962e260764add148c8eb88944df228c67f18eef690b1e3246e.jpg)

### Tables

![03040a430f8a6dd6d0e91558904177388ad8f2dc26428737573237a986821c23.jpg](../iclr_results/2109_MrT5_ Dynamic Token Merging for Efficient Byte-level Language Models/tables/03040a430f8a6dd6d0e91558904177388ad8f2dc26428737573237a986821c23.jpg)

![108b9e795dda376654b024a9a7e4097e7cfeb55b894d9b07aa2727f869e588e3.jpg](../iclr_results/2109_MrT5_ Dynamic Token Merging for Efficient Byte-level Language Models/tables/108b9e795dda376654b024a9a7e4097e7cfeb55b894d9b07aa2727f869e588e3.jpg)

![1f4f32081d4a98e0441d304e14bf11cb5930fda8e9a00a0a75059506600a8241.jpg](../iclr_results/2109_MrT5_ Dynamic Token Merging for Efficient Byte-level Language Models/tables/1f4f32081d4a98e0441d304e14bf11cb5930fda8e9a00a0a75059506600a8241.jpg)

![262d5db633e9270e66793cb2ebcd2621e824788d774f057bf9906f6293201dd6.jpg](../iclr_results/2109_MrT5_ Dynamic Token Merging for Efficient Byte-level Language Models/tables/262d5db633e9270e66793cb2ebcd2621e824788d774f057bf9906f6293201dd6.jpg)

![4f47895a8184cbec27cfffaf326c10d240c7f094ca3cc24f920bac202b48c84a.jpg](../iclr_results/2109_MrT5_ Dynamic Token Merging for Efficient Byte-level Language Models/tables/4f47895a8184cbec27cfffaf326c10d240c7f094ca3cc24f920bac202b48c84a.jpg)

![5c10bb5207804ea98309cf11d50246df7c11e0924942924f1d7262409882094e.jpg](../iclr_results/2109_MrT5_ Dynamic Token Merging for Efficient Byte-level Language Models/tables/5c10bb5207804ea98309cf11d50246df7c11e0924942924f1d7262409882094e.jpg)

![5ca68bb4981192ab237d1bcb5758aa3681d97d7bc9f76a94c6df445aa566f8d9.jpg](../iclr_results/2109_MrT5_ Dynamic Token Merging for Efficient Byte-level Language Models/tables/5ca68bb4981192ab237d1bcb5758aa3681d97d7bc9f76a94c6df445aa566f8d9.jpg)

![6572f75d4cf9a99ad5cbd111072fc21752ce5a73fcd585cc6109098bb9179f4e.jpg](../iclr_results/2109_MrT5_ Dynamic Token Merging for Efficient Byte-level Language Models/tables/6572f75d4cf9a99ad5cbd111072fc21752ce5a73fcd585cc6109098bb9179f4e.jpg)

![a8527388889e6ad9533b671e76db69e2de9f6a0393b3d1e5ed93f78eb069dfb8.jpg](../iclr_results/2109_MrT5_ Dynamic Token Merging for Efficient Byte-level Language Models/tables/a8527388889e6ad9533b671e76db69e2de9f6a0393b3d1e5ed93f78eb069dfb8.jpg)

![f1e72fdd449de4bc9c3981ac44fad82b0778527d9c44f3efd1cf1040d2f9ecd6.jpg](../iclr_results/2109_MrT5_ Dynamic Token Merging for Efficient Byte-level Language Models/tables/f1e72fdd449de4bc9c3981ac44fad82b0778527d9c44f3efd1cf1040d2f9ecd6.jpg)

![f8e87b76f406922015a2266467f87d6001522069924e6b473b954bbc37d0460a.jpg](../iclr_results/2109_MrT5_ Dynamic Token Merging for Efficient Byte-level Language Models/tables/f8e87b76f406922015a2266467f87d6001522069924e6b473b954bbc37d0460a.jpg)

## Latent Action Pretraining from Videos


### Images

![090df216837797272ecb6126e8630bf6077b491de321eefb85f2cbe4802b63cb.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/images/090df216837797272ecb6126e8630bf6077b491de321eefb85f2cbe4802b63cb.jpg)

![11ff7b9fc922d8fb5043a3ff978b0034c40a3a8135db72da5aba244dd92dbc1d.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/images/11ff7b9fc922d8fb5043a3ff978b0034c40a3a8135db72da5aba244dd92dbc1d.jpg)

![64e42cc0fed6445b028b382dcb62176bf99a9392261187cf77b2fbad35344539.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/images/64e42cc0fed6445b028b382dcb62176bf99a9392261187cf77b2fbad35344539.jpg)

![86149dbefcc478dab171e4893083f84a8e25b59102e2db8e2d8f37516820e6fb.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/images/86149dbefcc478dab171e4893083f84a8e25b59102e2db8e2d8f37516820e6fb.jpg)

![8822f7b901c7e8a971b27db2fdccef21b8aacbf31a995b81514731d61b7bb237.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/images/8822f7b901c7e8a971b27db2fdccef21b8aacbf31a995b81514731d61b7bb237.jpg)

![8a72a3f7af9275dd8934f224f18fe840890b21dd9f708bf8e6a34d32315e6319.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/images/8a72a3f7af9275dd8934f224f18fe840890b21dd9f708bf8e6a34d32315e6319.jpg)

![8cc052b9b6942e2cc52113701c5643ef168573db344c822f2532cfaa11c7e0d2.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/images/8cc052b9b6942e2cc52113701c5643ef168573db344c822f2532cfaa11c7e0d2.jpg)

![962268ad1540bffa360d09c8fc7607657acba148f6b61de240982f622acbbeed.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/images/962268ad1540bffa360d09c8fc7607657acba148f6b61de240982f622acbbeed.jpg)

![96e79bc7094b7fb2f6c4ffc8b3e2d2860e7b6522415da85897bacd3932542582.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/images/96e79bc7094b7fb2f6c4ffc8b3e2d2860e7b6522415da85897bacd3932542582.jpg)

![9a8c1688015c41e4bddb3dcf56b5a33140a3f8924eb37cb4c49a2a0ed14df9e5.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/images/9a8c1688015c41e4bddb3dcf56b5a33140a3f8924eb37cb4c49a2a0ed14df9e5.jpg)

![9b04ccbd3a516e2d1c0b0aa9a038606759894f9f2c8973bf5be5ebb73e044062.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/images/9b04ccbd3a516e2d1c0b0aa9a038606759894f9f2c8973bf5be5ebb73e044062.jpg)

![9c3d7a4b9057cc224be9be5ea216726b4b3aeb400d42bfce2b2f49bb2555dd94.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/images/9c3d7a4b9057cc224be9be5ea216726b4b3aeb400d42bfce2b2f49bb2555dd94.jpg)

![a3877613eff2fb5b50a7b94c0d9299bef18624ed3ecd180ed3d525362bf31ed3.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/images/a3877613eff2fb5b50a7b94c0d9299bef18624ed3ecd180ed3d525362bf31ed3.jpg)

![a5640a88399571a6a3a1f24dff65ee99d17690baa785971c9920a913e4cede47.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/images/a5640a88399571a6a3a1f24dff65ee99d17690baa785971c9920a913e4cede47.jpg)

![bd0432bd813daa282d9636cf6d2a5ff4533c5f03a8083a6ec03f4962f3ebd6a0.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/images/bd0432bd813daa282d9636cf6d2a5ff4533c5f03a8083a6ec03f4962f3ebd6a0.jpg)

![c5ec17b5814753612787f379588d1e6bc1c9fbed7e5b00d1e3cefc80d3b3b695.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/images/c5ec17b5814753612787f379588d1e6bc1c9fbed7e5b00d1e3cefc80d3b3b695.jpg)

![eadcf8e0b485a734c179e8b0bb9caf5fd13749765461850d106d5ee32657873f.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/images/eadcf8e0b485a734c179e8b0bb9caf5fd13749765461850d106d5ee32657873f.jpg)

### Tables

![02179f6ce1fccaadbb82b43ebcd154e41509d448254ab0028c49f7f18ec3d6b5.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/tables/02179f6ce1fccaadbb82b43ebcd154e41509d448254ab0028c49f7f18ec3d6b5.jpg)

![0334d442184b4082f6674541bc9e5f55bb47cb3f210ceed34daea35c542ca6bf.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/tables/0334d442184b4082f6674541bc9e5f55bb47cb3f210ceed34daea35c542ca6bf.jpg)

![13bcd5fbb2d0195cdce752ba8faa1feab6b40daf9ddd22b2ca562869b3c82925.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/tables/13bcd5fbb2d0195cdce752ba8faa1feab6b40daf9ddd22b2ca562869b3c82925.jpg)

![272f3497eb728c9d9048a4716d1b309ca3298bbc10d52b4969091f97fbabadcb.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/tables/272f3497eb728c9d9048a4716d1b309ca3298bbc10d52b4969091f97fbabadcb.jpg)

![309d97a48d308ec3e6f5cf61cd1a6e4f47a4cf62f3111022edd0c37d5041a2ec.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/tables/309d97a48d308ec3e6f5cf61cd1a6e4f47a4cf62f3111022edd0c37d5041a2ec.jpg)

![4438be32408bea9d9999ff9c1a7e50dafd5d7b1f637f753b095da4876b5de498.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/tables/4438be32408bea9d9999ff9c1a7e50dafd5d7b1f637f753b095da4876b5de498.jpg)

![64f2b437c5f4bd4b61717f98194ed103eeb3bc3b8647192542574e86eb5c8c9b.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/tables/64f2b437c5f4bd4b61717f98194ed103eeb3bc3b8647192542574e86eb5c8c9b.jpg)

![67e17404bb3180411c46353c4fa518343afdc63ae3f2493e3e518a32bc4a11b2.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/tables/67e17404bb3180411c46353c4fa518343afdc63ae3f2493e3e518a32bc4a11b2.jpg)

![7cfc61e7730596da0d227a15b1b1bd62d103a314e944615053e601a9bf2e6b2d.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/tables/7cfc61e7730596da0d227a15b1b1bd62d103a314e944615053e601a9bf2e6b2d.jpg)

![9031a78aff6f7af9b435865b9f023fa874adc0956c7fd76e65c49a4a8fffdd94.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/tables/9031a78aff6f7af9b435865b9f023fa874adc0956c7fd76e65c49a4a8fffdd94.jpg)

![92be582eb8726c9797e64affce9abec6a63c7b90831c468a3e5673945ef01697.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/tables/92be582eb8726c9797e64affce9abec6a63c7b90831c468a3e5673945ef01697.jpg)

![abdf2f1499a58001ec3d17efb4b8dddfb9f01ed9752a017dce1d0ea96292f66f.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/tables/abdf2f1499a58001ec3d17efb4b8dddfb9f01ed9752a017dce1d0ea96292f66f.jpg)

![ba5e83d296b92d5329372878af891e469fc4247d1bf91a326032b1e54d41e0c3.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/tables/ba5e83d296b92d5329372878af891e469fc4247d1bf91a326032b1e54d41e0c3.jpg)

![cee04fca3921f21bf89edc78f96b0d8f731ddcb826cbe26c7bfda13bfe48246d.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/tables/cee04fca3921f21bf89edc78f96b0d8f731ddcb826cbe26c7bfda13bfe48246d.jpg)

![e553b326884476c08744b67d7ee46cd458eaffd5fe3e5a1f267ca6f677a8e7b3.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/tables/e553b326884476c08744b67d7ee46cd458eaffd5fe3e5a1f267ca6f677a8e7b3.jpg)

![f0215e978c660329e508d150cd754df45492cfccd12634e0df32a19d05197886.jpg](../iclr_results/2110_Latent Action Pretraining from Videos/tables/f0215e978c660329e508d150cd754df45492cfccd12634e0df32a19d05197886.jpg)

## Cross the Gap:  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion


### Images

![1a3aec42704192c02189b7cf90e5e55998361e7106948fb03080b90fd9aa062e.jpg](../iclr_results/2111_Cross the Gap_  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion/images/1a3aec42704192c02189b7cf90e5e55998361e7106948fb03080b90fd9aa062e.jpg)

![c16243a25dda2a607c1349ce9c63547c75c3a034e4efc3ae4ab7304e74b0b8b5.jpg](../iclr_results/2111_Cross the Gap_  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion/images/c16243a25dda2a607c1349ce9c63547c75c3a034e4efc3ae4ab7304e74b0b8b5.jpg)

![e126a8491286c7f1df932933af8b69a5acd43471384f444701d3ebb3b4aad3b0.jpg](../iclr_results/2111_Cross the Gap_  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion/images/e126a8491286c7f1df932933af8b69a5acd43471384f444701d3ebb3b4aad3b0.jpg)

![fe3527fcc4f5ecafba52c61f28f02a6f3a0bf3e6253f9eb2c6885889fac93997.jpg](../iclr_results/2111_Cross the Gap_  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion/images/fe3527fcc4f5ecafba52c61f28f02a6f3a0bf3e6253f9eb2c6885889fac93997.jpg)

### Tables

![02466522ea490d8004d04de6ec6d2c2edc13bb6e8bb1e9141a608696c710c352.jpg](../iclr_results/2111_Cross the Gap_  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion/tables/02466522ea490d8004d04de6ec6d2c2edc13bb6e8bb1e9141a608696c710c352.jpg)

![19bc87e7f3c74ebb95fb3599172739689b5c91136b85e903ca50b8858e55a1ff.jpg](../iclr_results/2111_Cross the Gap_  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion/tables/19bc87e7f3c74ebb95fb3599172739689b5c91136b85e903ca50b8858e55a1ff.jpg)

![3fd2361ada26f3a60e4cdadb9605cbe53dfabd91aebe9229cf31a40a5598bb3b.jpg](../iclr_results/2111_Cross the Gap_  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion/tables/3fd2361ada26f3a60e4cdadb9605cbe53dfabd91aebe9229cf31a40a5598bb3b.jpg)

![604608d8fd217771353afeed1c7b2a90cd56f98f99c5b0740edf80eef14013b5.jpg](../iclr_results/2111_Cross the Gap_  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion/tables/604608d8fd217771353afeed1c7b2a90cd56f98f99c5b0740edf80eef14013b5.jpg)

![7f8bcabec22e872aecffa0946a6337810f264342df66104006176ddc11811e62.jpg](../iclr_results/2111_Cross the Gap_  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion/tables/7f8bcabec22e872aecffa0946a6337810f264342df66104006176ddc11811e62.jpg)

![8ece05561ae0555515a6398ad6d13bc4bff64c636fd95ebf018b14cd0a87f9c4.jpg](../iclr_results/2111_Cross the Gap_  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion/tables/8ece05561ae0555515a6398ad6d13bc4bff64c636fd95ebf018b14cd0a87f9c4.jpg)

![a7a23f9b3cdbf070e694311ec8ae4b4d7e3d96c79977c804c6ebfc7a737f32d9.jpg](../iclr_results/2111_Cross the Gap_  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion/tables/a7a23f9b3cdbf070e694311ec8ae4b4d7e3d96c79977c804c6ebfc7a737f32d9.jpg)

![aa2fdb2b81bed0a34f6c06ec1a6fc0a1729a28c7103947a56db1fbc0b988f3b6.jpg](../iclr_results/2111_Cross the Gap_  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion/tables/aa2fdb2b81bed0a34f6c06ec1a6fc0a1729a28c7103947a56db1fbc0b988f3b6.jpg)

![caa548014043021c8e6787e061e43626a6aac85e8a8bfcf189c9678b9b899cc6.jpg](../iclr_results/2111_Cross the Gap_  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion/tables/caa548014043021c8e6787e061e43626a6aac85e8a8bfcf189c9678b9b899cc6.jpg)

![cc58bac1658709b25c6ebc8768435a6d558f7fd10596d0f863e269900999cf43.jpg](../iclr_results/2111_Cross the Gap_  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion/tables/cc58bac1658709b25c6ebc8768435a6d558f7fd10596d0f863e269900999cf43.jpg)

![e2b8ddff8e83a3775f16be73a2baba51aafe1ad4fb96456d31abf8a7e16e1e6f.jpg](../iclr_results/2111_Cross the Gap_  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion/tables/e2b8ddff8e83a3775f16be73a2baba51aafe1ad4fb96456d31abf8a7e16e1e6f.jpg)

![e90337f0ddc04f7d28fc0aa1d1f5dccae54cdbf8ca8b1b69b37d35c5087e3f8a.jpg](../iclr_results/2111_Cross the Gap_  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion/tables/e90337f0ddc04f7d28fc0aa1d1f5dccae54cdbf8ca8b1b69b37d35c5087e3f8a.jpg)

![edaf6aac6421fbc784f4ca5dae47754b9a2215c3d468133f5d98c14a0069b10a.jpg](../iclr_results/2111_Cross the Gap_  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion/tables/edaf6aac6421fbc784f4ca5dae47754b9a2215c3d468133f5d98c14a0069b10a.jpg)

![ee65ccc119861bd4b6641214176071b5753e4b82bb99d330eb91a757e7316c39.jpg](../iclr_results/2111_Cross the Gap_  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion/tables/ee65ccc119861bd4b6641214176071b5753e4b82bb99d330eb91a757e7316c39.jpg)

![efb7f3f7d75b1b43363373c0fb6853747be89e18f24a6211017ab3a9ed1042a4.jpg](../iclr_results/2111_Cross the Gap_  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion/tables/efb7f3f7d75b1b43363373c0fb6853747be89e18f24a6211017ab3a9ed1042a4.jpg)

![f9e1d17963a6d72c4c473f43ab12af2d6729597e17518c0cf0ff6c986e0c92fe.jpg](../iclr_results/2111_Cross the Gap_  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion/tables/f9e1d17963a6d72c4c473f43ab12af2d6729597e17518c0cf0ff6c986e0c92fe.jpg)

![fbccb314ad162d9651fb2424f4912e185194a9c0b342ed72138fea5d745c522d.jpg](../iclr_results/2111_Cross the Gap_  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion/tables/fbccb314ad162d9651fb2424f4912e185194a9c0b342ed72138fea5d745c522d.jpg)

![fd17bf4aa9bb0d53d44bd2f14037adf5eee39874abc618478240499c6b7e6c9d.jpg](../iclr_results/2111_Cross the Gap_  Exposing the Intra-modal Misalignment in CLIP via Modality Inversion/tables/fd17bf4aa9bb0d53d44bd2f14037adf5eee39874abc618478240499c6b7e6c9d.jpg)

## Transformer Encoder Satisfiability: Complexity and Impact on Formal Reasoning


### Images

![0464babf785aa5ebf8f1cf44250500822c8a3390fe50755b3e5272b6d095798c.jpg](../iclr_results/2112_Transformer Encoder Satisfiability_ Complexity and Impact on Formal Reasoning/images/0464babf785aa5ebf8f1cf44250500822c8a3390fe50755b3e5272b6d095798c.jpg)

![4b3039cf44e66081111db3fecd1c60978e53a48d1d3582079e2e1d8fb0f177a1.jpg](../iclr_results/2112_Transformer Encoder Satisfiability_ Complexity and Impact on Formal Reasoning/images/4b3039cf44e66081111db3fecd1c60978e53a48d1d3582079e2e1d8fb0f177a1.jpg)

![84337a6a019a83e448d3bbfa81a946037866a91b525b56c91cda2bafc2819240.jpg](../iclr_results/2112_Transformer Encoder Satisfiability_ Complexity and Impact on Formal Reasoning/images/84337a6a019a83e448d3bbfa81a946037866a91b525b56c91cda2bafc2819240.jpg)

## CARTS: Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree Search


### Images

![20ad974e607ac167f9272cc5084204290f7f6a7dfccd4bbe9f0f06a2c4ca8f2b.jpg](../iclr_results/2113_CARTS_ Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree /images/20ad974e607ac167f9272cc5084204290f7f6a7dfccd4bbe9f0f06a2c4ca8f2b.jpg)

![34e8760d866bfb010773720556a1192f39db3abcee21cc3bf1dbb457a150e8fe.jpg](../iclr_results/2113_CARTS_ Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree /images/34e8760d866bfb010773720556a1192f39db3abcee21cc3bf1dbb457a150e8fe.jpg)

![523a67da7a8de63e58bea43a057523a9a11e17145be50766689c7ec0c10cd21a.jpg](../iclr_results/2113_CARTS_ Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree /images/523a67da7a8de63e58bea43a057523a9a11e17145be50766689c7ec0c10cd21a.jpg)

![56c9caff41d23c82289de63a7820728521e183761b5eb49f2bf3cd7eb1575c9b.jpg](../iclr_results/2113_CARTS_ Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree /images/56c9caff41d23c82289de63a7820728521e183761b5eb49f2bf3cd7eb1575c9b.jpg)

![5fa13fa921c3e8b45555b75b2a38a8705a4e1ef2455a11b3c405be8176635c3e.jpg](../iclr_results/2113_CARTS_ Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree /images/5fa13fa921c3e8b45555b75b2a38a8705a4e1ef2455a11b3c405be8176635c3e.jpg)

![5fde79ebd24b27ddf5dd99ecdac076862df89905fbf6a598cb256c72bfe0639b.jpg](../iclr_results/2113_CARTS_ Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree /images/5fde79ebd24b27ddf5dd99ecdac076862df89905fbf6a598cb256c72bfe0639b.jpg)

![7d68678af8dc901174956eb7d3a7516f31859c6e22fd1aa7856140c054fb53ac.jpg](../iclr_results/2113_CARTS_ Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree /images/7d68678af8dc901174956eb7d3a7516f31859c6e22fd1aa7856140c054fb53ac.jpg)

![9fdf944f5a4817e5e5a313e4e3b6199ab668cd17f2cbbbcc4a8d1299438ac7f7.jpg](../iclr_results/2113_CARTS_ Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree /images/9fdf944f5a4817e5e5a313e4e3b6199ab668cd17f2cbbbcc4a8d1299438ac7f7.jpg)

![b5401f01e3abe255d8553e008c71807f7518a810ef46052b9d361ed3a00aba17.jpg](../iclr_results/2113_CARTS_ Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree /images/b5401f01e3abe255d8553e008c71807f7518a810ef46052b9d361ed3a00aba17.jpg)

![eb7758a92732b37a1ca523315f87c85dc0ed654d9e3bb3df4c5824607ba873ba.jpg](../iclr_results/2113_CARTS_ Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree /images/eb7758a92732b37a1ca523315f87c85dc0ed654d9e3bb3df4c5824607ba873ba.jpg)

![efc0fa83ab068018e2854818291d60927641b5256b3e7df31b945d19f8749e44.jpg](../iclr_results/2113_CARTS_ Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree /images/efc0fa83ab068018e2854818291d60927641b5256b3e7df31b945d19f8749e44.jpg)

![f30e1ab6386c4865fbd75c7bd74479c6780ad4744729d48abb225fd044ed96f9.jpg](../iclr_results/2113_CARTS_ Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree /images/f30e1ab6386c4865fbd75c7bd74479c6780ad4744729d48abb225fd044ed96f9.jpg)

### Tables

![4b3d2f743c441b66e888f76fc06146f74ff0e3c783c0d9723239db5b74707ff0.jpg](../iclr_results/2113_CARTS_ Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree /tables/4b3d2f743c441b66e888f76fc06146f74ff0e3c783c0d9723239db5b74707ff0.jpg)

![9a2ec1310412e4943ece8092e4077b56783a717674e117f21d8de8d24ec6d4f7.jpg](../iclr_results/2113_CARTS_ Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree /tables/9a2ec1310412e4943ece8092e4077b56783a717674e117f21d8de8d24ec6d4f7.jpg)

![bbfb5b4388f2571fef1166969933e6b06262c0937a881f91384836eafcd785d1.jpg](../iclr_results/2113_CARTS_ Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree /tables/bbfb5b4388f2571fef1166969933e6b06262c0937a881f91384836eafcd785d1.jpg)

![cb6654dcc39ac20d99731d4984e99595a22ced0ef8e49f05067989e0c2bfdef0.jpg](../iclr_results/2113_CARTS_ Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree /tables/cb6654dcc39ac20d99731d4984e99595a22ced0ef8e49f05067989e0c2bfdef0.jpg)

![cd72e06bc995a6dcf5c81a534a6f0890ec9e6a7cd372056ffec0bef9b844563d.jpg](../iclr_results/2113_CARTS_ Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree /tables/cd72e06bc995a6dcf5c81a534a6f0890ec9e6a7cd372056ffec0bef9b844563d.jpg)

![ff45df0bcf7d3ed9ba6145be00c6b9c60abe47e5df4ca692bc98a82ae4ca17a7.jpg](../iclr_results/2113_CARTS_ Advancing Neural Theorem Proving with Diversified Tactic Calibration and Bias-Resistant Tree /tables/ff45df0bcf7d3ed9ba6145be00c6b9c60abe47e5df4ca692bc98a82ae4ca17a7.jpg)

## Bayesian Regularization of Latent Representation


### Images

![271ff7b19410717b8d67936fb8e5df82b291901631d9b10f1547727c202593f0.jpg](../iclr_results/2114_Bayesian Regularization of Latent Representation/images/271ff7b19410717b8d67936fb8e5df82b291901631d9b10f1547727c202593f0.jpg)

![2f7cb888a942044e1c59a02632b26317d9bf9b31b5e0e3da249941208df2f378.jpg](../iclr_results/2114_Bayesian Regularization of Latent Representation/images/2f7cb888a942044e1c59a02632b26317d9bf9b31b5e0e3da249941208df2f378.jpg)

![3f019d5d00897bb9bd27284cb896e5c7a7ced114fb7adf0cd3a711cd1cbaf6cb.jpg](../iclr_results/2114_Bayesian Regularization of Latent Representation/images/3f019d5d00897bb9bd27284cb896e5c7a7ced114fb7adf0cd3a711cd1cbaf6cb.jpg)

![492d2fd9fd80b4d075c3dfabc030e0178ccb4230fccafe57199a1563f6f2ec1e.jpg](../iclr_results/2114_Bayesian Regularization of Latent Representation/images/492d2fd9fd80b4d075c3dfabc030e0178ccb4230fccafe57199a1563f6f2ec1e.jpg)

![67de72c3e3f40f14949968a33f185c08b36c897e88fe0e9d2690d0f4e8be3c7b.jpg](../iclr_results/2114_Bayesian Regularization of Latent Representation/images/67de72c3e3f40f14949968a33f185c08b36c897e88fe0e9d2690d0f4e8be3c7b.jpg)

![6bd28d51badebabedab29ad2eeb108fbee12df9f259607913cdd7cb65cc40868.jpg](../iclr_results/2114_Bayesian Regularization of Latent Representation/images/6bd28d51badebabedab29ad2eeb108fbee12df9f259607913cdd7cb65cc40868.jpg)

![748117c4acc6e83fa025093502fe7c9c3a079b5ea426803e6ab40cca0d2419df.jpg](../iclr_results/2114_Bayesian Regularization of Latent Representation/images/748117c4acc6e83fa025093502fe7c9c3a079b5ea426803e6ab40cca0d2419df.jpg)

![b297665b28842cfa45f6e78d8c52d9bfaba05b06724d95a7d4ab45b5c9d02cac.jpg](../iclr_results/2114_Bayesian Regularization of Latent Representation/images/b297665b28842cfa45f6e78d8c52d9bfaba05b06724d95a7d4ab45b5c9d02cac.jpg)

![b49a42efbe3af9e566b83094ecacf54c60736dc3a1caf16a92c4081f3665c678.jpg](../iclr_results/2114_Bayesian Regularization of Latent Representation/images/b49a42efbe3af9e566b83094ecacf54c60736dc3a1caf16a92c4081f3665c678.jpg)

![b5481e5c8aa8df3ac428ef8dd58b4aaab30ee30096bf51d5e11f9cd4a17bad1c.jpg](../iclr_results/2114_Bayesian Regularization of Latent Representation/images/b5481e5c8aa8df3ac428ef8dd58b4aaab30ee30096bf51d5e11f9cd4a17bad1c.jpg)

![be987824ae2d27761231025d64db232b7cd15650e0f2098e0d0bbf95c4bea556.jpg](../iclr_results/2114_Bayesian Regularization of Latent Representation/images/be987824ae2d27761231025d64db232b7cd15650e0f2098e0d0bbf95c4bea556.jpg)

![cabeb30bb3d2d0ab6409b00dd27cf7db56ef8057a0445831f1676ef629237ecb.jpg](../iclr_results/2114_Bayesian Regularization of Latent Representation/images/cabeb30bb3d2d0ab6409b00dd27cf7db56ef8057a0445831f1676ef629237ecb.jpg)

![cd6a72aa29f3f495bd0c5b93402404cbddc1220c05a981453fc62ff4f57aca8e.jpg](../iclr_results/2114_Bayesian Regularization of Latent Representation/images/cd6a72aa29f3f495bd0c5b93402404cbddc1220c05a981453fc62ff4f57aca8e.jpg)

![ec83b8983cdea8651b6790ab57ace1b42389707ca10574ae2f09d2c0ef3ee835.jpg](../iclr_results/2114_Bayesian Regularization of Latent Representation/images/ec83b8983cdea8651b6790ab57ace1b42389707ca10574ae2f09d2c0ef3ee835.jpg)

### Tables

![d0b22cd0370f663e78cbb1f1cbe667cafaa809f09f08bcb5b9c3b172705f752e.jpg](../iclr_results/2114_Bayesian Regularization of Latent Representation/tables/d0b22cd0370f663e78cbb1f1cbe667cafaa809f09f08bcb5b9c3b172705f752e.jpg)

![e1aad0a05389a10c81846ab049b174707a818a4960d4e18585a28f333a6fb220.jpg](../iclr_results/2114_Bayesian Regularization of Latent Representation/tables/e1aad0a05389a10c81846ab049b174707a818a4960d4e18585a28f333a6fb220.jpg)

## DynaMath: A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Language Models


### Images

![0059f62bf975e7852897cf79a0380150e2fa285d948425ca78bb314e8aa67867.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/0059f62bf975e7852897cf79a0380150e2fa285d948425ca78bb314e8aa67867.jpg)

![01ff6b425b40af8794613019e5adf7df9c9962c6549b3f7ac80806c683e9ce73.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/01ff6b425b40af8794613019e5adf7df9c9962c6549b3f7ac80806c683e9ce73.jpg)

![03e01a1743b869edafdd050e544132009c44f1c0582dbade4c88f671cc5c1578.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/03e01a1743b869edafdd050e544132009c44f1c0582dbade4c88f671cc5c1578.jpg)

![04c09bee9a8b2f45c89a1c3b9f5525f754b4d9eafc421e78437585309cb212ca.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/04c09bee9a8b2f45c89a1c3b9f5525f754b4d9eafc421e78437585309cb212ca.jpg)

![06e62835313e6e5efa50a84945c4880e37fa39186e36bf70e9a87ecd35bffe9c.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/06e62835313e6e5efa50a84945c4880e37fa39186e36bf70e9a87ecd35bffe9c.jpg)

![089ca110b9e1e885752ec1b8ba2b3469f49bbaa984ed74ee3d961295f81089ce.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/089ca110b9e1e885752ec1b8ba2b3469f49bbaa984ed74ee3d961295f81089ce.jpg)

![0b5f7d7ee36b13003bcbf6878171e86c60ed676749477ee7399d79784db88aad.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/0b5f7d7ee36b13003bcbf6878171e86c60ed676749477ee7399d79784db88aad.jpg)

![0b946510f2870398c86a695d2eedb9701ddee05e490ef7fa4c598e2ea972eed7.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/0b946510f2870398c86a695d2eedb9701ddee05e490ef7fa4c598e2ea972eed7.jpg)

![0c61258638c1e1139a5f60456fe0547fe22431d86552e65d0d3a620f0928f339.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/0c61258638c1e1139a5f60456fe0547fe22431d86552e65d0d3a620f0928f339.jpg)

![0e2eace81a42c21df9ceef9911b31e9971c0814c4ab3562bd184d9cfd116ac7d.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/0e2eace81a42c21df9ceef9911b31e9971c0814c4ab3562bd184d9cfd116ac7d.jpg)

![19b453d577e0bf255477c13c4df0c2e24c8676e76771f450f55239682e714ccd.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/19b453d577e0bf255477c13c4df0c2e24c8676e76771f450f55239682e714ccd.jpg)

![1ac4f19de3968a46f78fe1403185afb3532fb8ccbf28218e6f4045486455aefb.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/1ac4f19de3968a46f78fe1403185afb3532fb8ccbf28218e6f4045486455aefb.jpg)

![1ec84ca95595a284d91b6de937e312ba974ff7596e74856f64613d63da2cfcd6.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/1ec84ca95595a284d91b6de937e312ba974ff7596e74856f64613d63da2cfcd6.jpg)

![1f92195d76265172a496d5ed41536e057f75ba58763d5cc4684b866d27b9ab82.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/1f92195d76265172a496d5ed41536e057f75ba58763d5cc4684b866d27b9ab82.jpg)

![244ba3145f7b9bd654af7b2c82781df5da7f05297400f43daa899179ec9e2691.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/244ba3145f7b9bd654af7b2c82781df5da7f05297400f43daa899179ec9e2691.jpg)

![26bca7d625f49d3a4f1939849931869cd93422947c45e435258f5dcdf728d889.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/26bca7d625f49d3a4f1939849931869cd93422947c45e435258f5dcdf728d889.jpg)

![28bcffb2fc70986048482613f0750ee08a9ef8308d8c5c5d22389ec9b025fa60.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/28bcffb2fc70986048482613f0750ee08a9ef8308d8c5c5d22389ec9b025fa60.jpg)

![2d8167a1979aa8e0a8db337f0c0e5dcef5fb58f84fe7699481b274582386e0f2.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/2d8167a1979aa8e0a8db337f0c0e5dcef5fb58f84fe7699481b274582386e0f2.jpg)

![2e58bf5104e4ae80695a9f6859ea6a9e725a26087e6c0f3bef790ad4561151da.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/2e58bf5104e4ae80695a9f6859ea6a9e725a26087e6c0f3bef790ad4561151da.jpg)

![2fcf6a121348cd61f5a78d3db3ba1b90d67febceab9236ae1aa4e24ea1fac2a5.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/2fcf6a121348cd61f5a78d3db3ba1b90d67febceab9236ae1aa4e24ea1fac2a5.jpg)

![3e28edbd3e64106fc2315051e7350f1fe02b5849288a4aa5659db78c30536884.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/3e28edbd3e64106fc2315051e7350f1fe02b5849288a4aa5659db78c30536884.jpg)

![41c70a06218a533e1389e6dee0aedc6a8eadb1e4bf4d56b90fe2aacce81dba6c.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/41c70a06218a533e1389e6dee0aedc6a8eadb1e4bf4d56b90fe2aacce81dba6c.jpg)

![4536df4cef2e70a03adc189f61f6b50f4c7a81946db03413749eecbf66c1367c.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/4536df4cef2e70a03adc189f61f6b50f4c7a81946db03413749eecbf66c1367c.jpg)

![456766a0dea997e6cc8be24aca9751a27b2354c256b4f83f3aa88fbc000800fb.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/456766a0dea997e6cc8be24aca9751a27b2354c256b4f83f3aa88fbc000800fb.jpg)

![470a87cd4ad7d60d07c3ae39e5a170c7d227eb4a3860b715e770860e5bd44231.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/470a87cd4ad7d60d07c3ae39e5a170c7d227eb4a3860b715e770860e5bd44231.jpg)

![4cc27f776bdac6b19d411788c5a31eaebc9f801e61637e2b17794b33b86e04e2.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/4cc27f776bdac6b19d411788c5a31eaebc9f801e61637e2b17794b33b86e04e2.jpg)

![4d8156b1625e337a2273ff5837e2c992b7359113d61f0755dec5313e82fe57ff.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/4d8156b1625e337a2273ff5837e2c992b7359113d61f0755dec5313e82fe57ff.jpg)

![51166ba9f6b4923b8e4af521da1cd2dfc31bc6ded9fff2363137d88790a3d387.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/51166ba9f6b4923b8e4af521da1cd2dfc31bc6ded9fff2363137d88790a3d387.jpg)

![51c85866ae7da76ac91e3581f1e4734e30d5987020f3f61cafbe6fe0d3ca30ff.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/51c85866ae7da76ac91e3581f1e4734e30d5987020f3f61cafbe6fe0d3ca30ff.jpg)

![52dd7d89582ea12ba13b64e8b1d4341bab6701f4178174767d5698c624eae628.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/52dd7d89582ea12ba13b64e8b1d4341bab6701f4178174767d5698c624eae628.jpg)

![5878426c9b39152f129a05639df21b158180faca5357ada432ad19462ce30466.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/5878426c9b39152f129a05639df21b158180faca5357ada432ad19462ce30466.jpg)

![5d24766fe2f460026d35b8d0b5d0495bc0d3eadee55a56fd77957bae33e516c4.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/5d24766fe2f460026d35b8d0b5d0495bc0d3eadee55a56fd77957bae33e516c4.jpg)

![5f01e8496cceee4dbe6a7d6bfea3d7ff7b8f7895fc2505efec5c9e4c9d0b63cc.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/5f01e8496cceee4dbe6a7d6bfea3d7ff7b8f7895fc2505efec5c9e4c9d0b63cc.jpg)

![61d29bcddabd074a394eec7dadb596d65ef64bbc2d855f1539bdc77a01866da2.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/61d29bcddabd074a394eec7dadb596d65ef64bbc2d855f1539bdc77a01866da2.jpg)

![623bce3c64054d42caca2082e3a1b6b9175391ac46b46c42233e701b61222776.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/623bce3c64054d42caca2082e3a1b6b9175391ac46b46c42233e701b61222776.jpg)

![632950db063f895f0935b9032121388eb80710019f0c2d8c85cec5d54a012ebc.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/632950db063f895f0935b9032121388eb80710019f0c2d8c85cec5d54a012ebc.jpg)

![64e4a588df44796ce5eee02fde3f8e062b22239c2d83dc6705d768a5b43af7c4.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/64e4a588df44796ce5eee02fde3f8e062b22239c2d83dc6705d768a5b43af7c4.jpg)

![66a82388204a3aa3d0f6e032f5110e869d291df0f5a2b2966fb33f1cbf58721c.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/66a82388204a3aa3d0f6e032f5110e869d291df0f5a2b2966fb33f1cbf58721c.jpg)

![69e5be4fd87fc15c45a7526d249dd8e399b5eab67a356c8f44832af8d5a90508.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/69e5be4fd87fc15c45a7526d249dd8e399b5eab67a356c8f44832af8d5a90508.jpg)

![6cd35556bfc2326ab28fd6adf0ab7cd123305377c9b92a99cb07f9ee11c0c321.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/6cd35556bfc2326ab28fd6adf0ab7cd123305377c9b92a99cb07f9ee11c0c321.jpg)

![6e10570ba47d3f3c30c9ad820202b35cc8f2dfb58a82fd668063834c0e029050.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/6e10570ba47d3f3c30c9ad820202b35cc8f2dfb58a82fd668063834c0e029050.jpg)

![6f5fb3eabad30a8a874445f362adf6e7e03e408fd4c69013adb38ecc928876e6.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/6f5fb3eabad30a8a874445f362adf6e7e03e408fd4c69013adb38ecc928876e6.jpg)

![70b5b9321da2f47289d0ad1b4a4135394c55435f276eae91d8f48a5cd241fead.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/70b5b9321da2f47289d0ad1b4a4135394c55435f276eae91d8f48a5cd241fead.jpg)

![7366e550ae8701ea409237273cbff5da6e65acfdcfb27e446ab38b8f9ca80c2b.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/7366e550ae8701ea409237273cbff5da6e65acfdcfb27e446ab38b8f9ca80c2b.jpg)

![764b236c479b1c953e3b8a2fe612552ce5487723143533becf514f187ae7d9dd.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/764b236c479b1c953e3b8a2fe612552ce5487723143533becf514f187ae7d9dd.jpg)

![7d73f93c75dcf3977257eb46784d617cba85e4a4f7812cb5eeaf8b1ff4de32df.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/7d73f93c75dcf3977257eb46784d617cba85e4a4f7812cb5eeaf8b1ff4de32df.jpg)

![7e050acefab8302ac80d7068e9fa9f1c50e2d8086569789b1bf36272269b0414.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/7e050acefab8302ac80d7068e9fa9f1c50e2d8086569789b1bf36272269b0414.jpg)

![8492bbe5f1d89ca5bfbbb249ce018e76fe39277fa0d0dd3476d8c5e1661713ac.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/8492bbe5f1d89ca5bfbbb249ce018e76fe39277fa0d0dd3476d8c5e1661713ac.jpg)

![895332ecc5b6b3f9b127cd5f34d7f1ea566a81c6b6e0c557d0fbe84e15da7213.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/895332ecc5b6b3f9b127cd5f34d7f1ea566a81c6b6e0c557d0fbe84e15da7213.jpg)

![89bb9c3dc867f0cb1439405c7d826917b428fb20c70d3a5d47f2d5b956ec512a.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/89bb9c3dc867f0cb1439405c7d826917b428fb20c70d3a5d47f2d5b956ec512a.jpg)

![8f86275ff4e4c2c9acc4296001f5833681afa44cbda8f26cb7cc9e13cc879dc7.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/8f86275ff4e4c2c9acc4296001f5833681afa44cbda8f26cb7cc9e13cc879dc7.jpg)

![941c3b806af4578c7e2b3deeb3f8f27a3c6a3ddcb83142d833d292c04213269e.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/941c3b806af4578c7e2b3deeb3f8f27a3c6a3ddcb83142d833d292c04213269e.jpg)

![95602b312168296bd273d5c9dfa5902c82d519ef7e884bceb3827d0ea322f2c7.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/95602b312168296bd273d5c9dfa5902c82d519ef7e884bceb3827d0ea322f2c7.jpg)

![959a8b6f347e2ffb015f9ca83625a9ad9ee0d38ea3067549afa19bab5f644e20.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/959a8b6f347e2ffb015f9ca83625a9ad9ee0d38ea3067549afa19bab5f644e20.jpg)

![989b1092a0105ac0efd2c886bd7a075f9b5fad5d63ea5df9f6f9477081f24cdb.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/989b1092a0105ac0efd2c886bd7a075f9b5fad5d63ea5df9f6f9477081f24cdb.jpg)

![9d62f713af220960c33921dd256ee8326e61104f669925fb6dbdf88b0d901f39.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/9d62f713af220960c33921dd256ee8326e61104f669925fb6dbdf88b0d901f39.jpg)

![9f73d970c059f2cf22fdd8139501e6ba4e596a381901158663cc9112934d23d1.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/9f73d970c059f2cf22fdd8139501e6ba4e596a381901158663cc9112934d23d1.jpg)

![a108b08a54f93baf63c62e93398acca4f907d0f12c2529a7db1d58744da335d9.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/a108b08a54f93baf63c62e93398acca4f907d0f12c2529a7db1d58744da335d9.jpg)

![a2ca20f990c3662cad731374929ee61d876a374aeb8324fa9ef9f5442e621e0e.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/a2ca20f990c3662cad731374929ee61d876a374aeb8324fa9ef9f5442e621e0e.jpg)

![a2d4107164917477daa87a825801c4307dd0301ed5821b22c06f33329dd78c1e.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/a2d4107164917477daa87a825801c4307dd0301ed5821b22c06f33329dd78c1e.jpg)

![a3f7f92c7ca7e26ea4aa141192d01143ed30c363a203c04510f2f328e2a19681.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/a3f7f92c7ca7e26ea4aa141192d01143ed30c363a203c04510f2f328e2a19681.jpg)

![a56df99e9eb690b9e100c2dd0bdb2042bc6d038df115fa1bf22f7dbe95c504d6.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/a56df99e9eb690b9e100c2dd0bdb2042bc6d038df115fa1bf22f7dbe95c504d6.jpg)

![a5e96fd6ac1bc27cef17e2bc00bd0fcffbfbf360d1c21ea0cabde45f2f633f5c.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/a5e96fd6ac1bc27cef17e2bc00bd0fcffbfbf360d1c21ea0cabde45f2f633f5c.jpg)

![a823b9270a0b6046a4e1310112a31f6a633cb355af274ecb07e0487339ab3c79.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/a823b9270a0b6046a4e1310112a31f6a633cb355af274ecb07e0487339ab3c79.jpg)

![aaed79c2efed8e093d4d95da00fdc472b64b336db1b284a25973c10e0aa21360.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/aaed79c2efed8e093d4d95da00fdc472b64b336db1b284a25973c10e0aa21360.jpg)

![ad8aa3130ade575644d0602b47f0170cca7526057f938dbca6c2bc80766a90e2.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/ad8aa3130ade575644d0602b47f0170cca7526057f938dbca6c2bc80766a90e2.jpg)

![ae644f1788b871d9a80b78f23c17c5a0d18bae1920d5a12608e48cab6f0c19a0.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/ae644f1788b871d9a80b78f23c17c5a0d18bae1920d5a12608e48cab6f0c19a0.jpg)

![b1c4c7f534e0b2c577daeaa50e1f2652042bcbc4d7c37cb73a7a50ca94371041.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/b1c4c7f534e0b2c577daeaa50e1f2652042bcbc4d7c37cb73a7a50ca94371041.jpg)

![b825fff9f2821a027c884f6ac821fc89b202088b93660eb9fdf0bf0f18b52f96.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/b825fff9f2821a027c884f6ac821fc89b202088b93660eb9fdf0bf0f18b52f96.jpg)

![bacd82f9233fc37684756d6e0a9a286b590625385821c84eee01d53fce1c3409.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/bacd82f9233fc37684756d6e0a9a286b590625385821c84eee01d53fce1c3409.jpg)

![c1f4fda0fa11f3e31806055dc2cf035c33f9455d39c7d4ec75efc73e4c225806.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/c1f4fda0fa11f3e31806055dc2cf035c33f9455d39c7d4ec75efc73e4c225806.jpg)

![c8f396118a7c2f293a38d1a80c335164df221ef2ee4633796f4f5f731ab300e8.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/c8f396118a7c2f293a38d1a80c335164df221ef2ee4633796f4f5f731ab300e8.jpg)

![cbd3474b5cf3927a666bb150a4fac98f1f90a8036f5c9c0b20a1318f2635585c.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/cbd3474b5cf3927a666bb150a4fac98f1f90a8036f5c9c0b20a1318f2635585c.jpg)

![cc46a089de5fb0bc13904aa3d344613ccd9246f5603e43c8dc6e3dbfc3df282d.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/cc46a089de5fb0bc13904aa3d344613ccd9246f5603e43c8dc6e3dbfc3df282d.jpg)

![cee5aba037837134d05ab810cf77954413ad90de4e3053b916a309ffebc5c8fa.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/cee5aba037837134d05ab810cf77954413ad90de4e3053b916a309ffebc5c8fa.jpg)

![d0a6e9886bf1aec1c63074c2b594de21b7c0472358a63c2da4dfb2f862a73c94.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/d0a6e9886bf1aec1c63074c2b594de21b7c0472358a63c2da4dfb2f862a73c94.jpg)

![d0e8e5dd2099a1f00134b06d3f47aca36102c43cf41f890bfa5d1cfc399ba95d.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/d0e8e5dd2099a1f00134b06d3f47aca36102c43cf41f890bfa5d1cfc399ba95d.jpg)

![d202af09c2c9df369c8e6bfc1029422f6200bf9d8d71c1c247bcf656d1388313.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/d202af09c2c9df369c8e6bfc1029422f6200bf9d8d71c1c247bcf656d1388313.jpg)

![d6641857fb4956caaeff053abbb02116efafbc94fef2715c521d963267239832.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/d6641857fb4956caaeff053abbb02116efafbc94fef2715c521d963267239832.jpg)

![d769703fc775ce18fb273bbdeafdcec0f0e87176505c4f704f5cd5d210e1b7fd.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/d769703fc775ce18fb273bbdeafdcec0f0e87176505c4f704f5cd5d210e1b7fd.jpg)

![dc0bbb3fbf74345af4cb4148e72f1c23af3dbd9502a93409fbd045ac52e4f971.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/dc0bbb3fbf74345af4cb4148e72f1c23af3dbd9502a93409fbd045ac52e4f971.jpg)

![e0e72b086442dfd9039e7e96b7ebaa19dcffd75251ab5369047e4aee2b09b377.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/e0e72b086442dfd9039e7e96b7ebaa19dcffd75251ab5369047e4aee2b09b377.jpg)

![e274ee9e22e16ff5a0672fb5c9937d7b05f0fdc1d973f1e58b87764601c80819.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/e274ee9e22e16ff5a0672fb5c9937d7b05f0fdc1d973f1e58b87764601c80819.jpg)

![e3d9c5b202d931fc9184f9b07b889d52b343a080c5b562f1f607c4a9aac79ed0.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/e3d9c5b202d931fc9184f9b07b889d52b343a080c5b562f1f607c4a9aac79ed0.jpg)

![ead7bd18f7e5cb9786f4e3b2a2ca1bd7993f30827c02850e86adb4156834a074.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/ead7bd18f7e5cb9786f4e3b2a2ca1bd7993f30827c02850e86adb4156834a074.jpg)

![efbc9737bb279d4c78439afd882e8cd3c64350d6d850d1fd94dfe1983f6dd41f.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/efbc9737bb279d4c78439afd882e8cd3c64350d6d850d1fd94dfe1983f6dd41f.jpg)

![f0f9af3a162b545c59ce6cfe3a1d239ea2aebfc0d1f1cd2cd3ad6a07612c44ec.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/f0f9af3a162b545c59ce6cfe3a1d239ea2aebfc0d1f1cd2cd3ad6a07612c44ec.jpg)

![f143c656d11415a2e6ddac7317c470e13abff33d11e74134e47c8f9b20843fac.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/f143c656d11415a2e6ddac7317c470e13abff33d11e74134e47c8f9b20843fac.jpg)

![f277fe4e45438e9a3150019b0e96a06199a5218aecc9ca7b619036cc3811153a.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/f277fe4e45438e9a3150019b0e96a06199a5218aecc9ca7b619036cc3811153a.jpg)

![f3a3c73f15fca6b144fabe1e8ddb14bfa08a135bdd910d429f6800382e950eaf.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/f3a3c73f15fca6b144fabe1e8ddb14bfa08a135bdd910d429f6800382e950eaf.jpg)

![f747f62c5ebfdb6f074cf5e928e9bf75107718d6b93a7b447892adff525cb23a.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/f747f62c5ebfdb6f074cf5e928e9bf75107718d6b93a7b447892adff525cb23a.jpg)

![f7bff73d1a8b12a6910784e4aaf4a3e158d98d8ef7c7643124aa29756cf8ab8d.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/f7bff73d1a8b12a6910784e4aaf4a3e158d98d8ef7c7643124aa29756cf8ab8d.jpg)

![f896c6f5542000a56b6571aaa9c983bc77987b56ba9c31465affef9297014a0c.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/images/f896c6f5542000a56b6571aaa9c983bc77987b56ba9c31465affef9297014a0c.jpg)

### Tables

![1dffe3faa58777be9bae9f1ed19fdfece972f5b0dd26c396f398ca799352ea98.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/tables/1dffe3faa58777be9bae9f1ed19fdfece972f5b0dd26c396f398ca799352ea98.jpg)

![1f7e1e7b16110bca81a0a051285d6770e51dec5fa95cc7a767d94f12c892f9a0.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/tables/1f7e1e7b16110bca81a0a051285d6770e51dec5fa95cc7a767d94f12c892f9a0.jpg)

![3548bb9a0d9a729e5df4c6bccf7155d4b65746cf014bbb87584b021893c3b191.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/tables/3548bb9a0d9a729e5df4c6bccf7155d4b65746cf014bbb87584b021893c3b191.jpg)

![3f268b0a49d9e133f1085373bac98ffedc613b7726092fbf5ece163731544012.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/tables/3f268b0a49d9e133f1085373bac98ffedc613b7726092fbf5ece163731544012.jpg)

![3f6282cb3fa4c6cc7e090106d9a07e883e6c22b0c20b17d584b9165aee15ac31.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/tables/3f6282cb3fa4c6cc7e090106d9a07e883e6c22b0c20b17d584b9165aee15ac31.jpg)

![3f801f4dea6bf5b997353de0705c5cfa5db7942bf172a656187ce7e30748e2a0.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/tables/3f801f4dea6bf5b997353de0705c5cfa5db7942bf172a656187ce7e30748e2a0.jpg)

![704adbdaab0556020f0dbc4cc8d344b808f5a7413bc989e3ad4c680bc714dc80.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/tables/704adbdaab0556020f0dbc4cc8d344b808f5a7413bc989e3ad4c680bc714dc80.jpg)

![90e6edac3139627de4ca02ec816daf5571e0612fa328eefd6bcd3bb93fcf4d3e.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/tables/90e6edac3139627de4ca02ec816daf5571e0612fa328eefd6bcd3bb93fcf4d3e.jpg)

![94a436ddaa320056c5e8ef47c6428998269f8d67cbe311523d9f4509be334aba.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/tables/94a436ddaa320056c5e8ef47c6428998269f8d67cbe311523d9f4509be334aba.jpg)

![97817a4d114b7e3a4c821f238514042b182bd65a3a7c20e1d283635b355a5248.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/tables/97817a4d114b7e3a4c821f238514042b182bd65a3a7c20e1d283635b355a5248.jpg)

![9d1bc43c80e00370fcfbb4610ac7664f41880e34d216254076a9e0d133852795.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/tables/9d1bc43c80e00370fcfbb4610ac7664f41880e34d216254076a9e0d133852795.jpg)

![abcef7c59d6fc1f958a135690161a4a5b669c6a5d09bdc6daaa58effca351ff8.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/tables/abcef7c59d6fc1f958a135690161a4a5b669c6a5d09bdc6daaa58effca351ff8.jpg)

![b8134b035b5de261e51b235611879eea089fedb290f4d66c6003d2ee991c809a.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/tables/b8134b035b5de261e51b235611879eea089fedb290f4d66c6003d2ee991c809a.jpg)

![c01ff7e65665b25fd004f10d387e22a6c59eaa394180a31b1af542f3a9ec94ca.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/tables/c01ff7e65665b25fd004f10d387e22a6c59eaa394180a31b1af542f3a9ec94ca.jpg)

![d02fd1024bc80d390fffd077ff93bfb80659e51dcba8f7422093678f8c7e88e4.jpg](../iclr_results/2115_DynaMath_ A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Lang/tables/d02fd1024bc80d390fffd077ff93bfb80659e51dcba8f7422093678f8c7e88e4.jpg)

## Convergence of Distributed Adaptive Optimization with Local Updates


### Images

![a1196e00f275ebc4b6bbf65c1c4391e790757e51a3ccd52c6e65de6d85509b0a.jpg](../iclr_results/2116_Convergence of Distributed Adaptive Optimization with Local Updates/images/a1196e00f275ebc4b6bbf65c1c4391e790757e51a3ccd52c6e65de6d85509b0a.jpg)

### Tables

![7709fd04c02e166f50f5d78da89533c008636811e07e90944e132457cfdc80b9.jpg](../iclr_results/2116_Convergence of Distributed Adaptive Optimization with Local Updates/tables/7709fd04c02e166f50f5d78da89533c008636811e07e90944e132457cfdc80b9.jpg)

![e339934a3eacf94c314d83c2cfa0337611b594b9a28183fe5874ef70a819f9ae.jpg](../iclr_results/2116_Convergence of Distributed Adaptive Optimization with Local Updates/tables/e339934a3eacf94c314d83c2cfa0337611b594b9a28183fe5874ef70a819f9ae.jpg)

## Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving


### Images

![1ba42e95508a1ea38214be996f6fad6abc456034bd2c3a76d4c150fc6c1d8f62.jpg](../iclr_results/2117_Inference Scaling Laws_ An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving/images/1ba42e95508a1ea38214be996f6fad6abc456034bd2c3a76d4c150fc6c1d8f62.jpg)

![4875435048f964e138d6103e475926c03f7ba5f93bdc30eb13f123c01e47b188.jpg](../iclr_results/2117_Inference Scaling Laws_ An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving/images/4875435048f964e138d6103e475926c03f7ba5f93bdc30eb13f123c01e47b188.jpg)

![52fe92c508ef91c8aef7c0b93a7779751f505a040cb359c996ad8e932e49ee4b.jpg](../iclr_results/2117_Inference Scaling Laws_ An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving/images/52fe92c508ef91c8aef7c0b93a7779751f505a040cb359c996ad8e932e49ee4b.jpg)

![5b4ce464097ac4d98552ceb862cc47d7f6c8b573091f8d4d719f8995cd66a465.jpg](../iclr_results/2117_Inference Scaling Laws_ An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving/images/5b4ce464097ac4d98552ceb862cc47d7f6c8b573091f8d4d719f8995cd66a465.jpg)

![5bfe698b9e25372f18432326bc19075d49b44c2aa76c3f3cd160684f50b510e0.jpg](../iclr_results/2117_Inference Scaling Laws_ An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving/images/5bfe698b9e25372f18432326bc19075d49b44c2aa76c3f3cd160684f50b510e0.jpg)

![68496da98fa319ab8a02c446854a1b1a9f57eb7b71fd72248e584dacc54fc80f.jpg](../iclr_results/2117_Inference Scaling Laws_ An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving/images/68496da98fa319ab8a02c446854a1b1a9f57eb7b71fd72248e584dacc54fc80f.jpg)

![8b9f1a9fef279db2fc3811f3efb828338f58e9445388db7c3691b516b0d61307.jpg](../iclr_results/2117_Inference Scaling Laws_ An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving/images/8b9f1a9fef279db2fc3811f3efb828338f58e9445388db7c3691b516b0d61307.jpg)

![b32953453824ced7bd378101ada71acff8e389fa02f5b792b3cdfc5207863505.jpg](../iclr_results/2117_Inference Scaling Laws_ An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving/images/b32953453824ced7bd378101ada71acff8e389fa02f5b792b3cdfc5207863505.jpg)

![b4363578e34c462d5ad295c84dff96b2800693dad6008ef54e8dc01f47b00646.jpg](../iclr_results/2117_Inference Scaling Laws_ An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving/images/b4363578e34c462d5ad295c84dff96b2800693dad6008ef54e8dc01f47b00646.jpg)

![bb58d0a7f74fb450b9519118c566214c5e0008d4c7b55603374756fc91c9ef16.jpg](../iclr_results/2117_Inference Scaling Laws_ An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving/images/bb58d0a7f74fb450b9519118c566214c5e0008d4c7b55603374756fc91c9ef16.jpg)

![f8651f9d1b42669dad755b270f1032a6f4fb63bddb535845bddccd74741f90e9.jpg](../iclr_results/2117_Inference Scaling Laws_ An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving/images/f8651f9d1b42669dad755b270f1032a6f4fb63bddb535845bddccd74741f90e9.jpg)

![f8bbc67626b8e51ef3edd7a31a63d1ec610eb44f562b2d6f8dd52dcd0f053bc0.jpg](../iclr_results/2117_Inference Scaling Laws_ An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving/images/f8bbc67626b8e51ef3edd7a31a63d1ec610eb44f562b2d6f8dd52dcd0f053bc0.jpg)

![fb2253ce09e023010c32c7ad48250ce44f3901a2eec8dbc2e72f2550a396a7b9.jpg](../iclr_results/2117_Inference Scaling Laws_ An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving/images/fb2253ce09e023010c32c7ad48250ce44f3901a2eec8dbc2e72f2550a396a7b9.jpg)

### Tables

![88789ca421e45f7423a59df199b0edadc055a6bf2de1170fffeecde56f60eca7.jpg](../iclr_results/2117_Inference Scaling Laws_ An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving/tables/88789ca421e45f7423a59df199b0edadc055a6bf2de1170fffeecde56f60eca7.jpg)

![cecac33faafba8ba2974e59fa8f69378f4c59f8e41b4a33f33617584233b83d1.jpg](../iclr_results/2117_Inference Scaling Laws_ An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving/tables/cecac33faafba8ba2974e59fa8f69378f4c59f8e41b4a33f33617584233b83d1.jpg)

![f96c89fa62a473dc827b6cfc599aa2f1cecc9782164cac35a13d1a6c422c4ca9.jpg](../iclr_results/2117_Inference Scaling Laws_ An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving/tables/f96c89fa62a473dc827b6cfc599aa2f1cecc9782164cac35a13d1a6c422c4ca9.jpg)

![fcca578c2f7e00eec4b1724868ac3d73b6ac95753cfff7c55aac151fdeda4f5a.jpg](../iclr_results/2117_Inference Scaling Laws_ An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving/tables/fcca578c2f7e00eec4b1724868ac3d73b6ac95753cfff7c55aac151fdeda4f5a.jpg)

## Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks


### Images

![0538a9b4a45623e57fbeeb92c4c40f0fb8558b30bbc4525faf5b8075bc206723.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/images/0538a9b4a45623e57fbeeb92c4c40f0fb8558b30bbc4525faf5b8075bc206723.jpg)

![154e4f677d0f34d062e8813579f571c4115c8a8465185ba6141876c4d16db430.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/images/154e4f677d0f34d062e8813579f571c4115c8a8465185ba6141876c4d16db430.jpg)

![2a61e552e9206337f32f3329814219aaaa214b944a610a83b680b656c1fff464.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/images/2a61e552e9206337f32f3329814219aaaa214b944a610a83b680b656c1fff464.jpg)

![2a74313b16bf38e17030a964b06a47de11d21a7d37ed33c5b7bd437ec2a79d87.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/images/2a74313b16bf38e17030a964b06a47de11d21a7d37ed33c5b7bd437ec2a79d87.jpg)

![410089c0f3ef6898681050d6751c4f2acf39f0ef0268b8e26d686cdb06f8a5c1.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/images/410089c0f3ef6898681050d6751c4f2acf39f0ef0268b8e26d686cdb06f8a5c1.jpg)

![4138c23f78c8a65ce3fa61eecd8e9043f81ba1033404bb6154e1534f2d98212b.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/images/4138c23f78c8a65ce3fa61eecd8e9043f81ba1033404bb6154e1534f2d98212b.jpg)

![43ae3beebbb5006a4c88f22b6bb9e23f8d5e1ff9f1cdf5a17cdc16e4c01b9175.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/images/43ae3beebbb5006a4c88f22b6bb9e23f8d5e1ff9f1cdf5a17cdc16e4c01b9175.jpg)

![5188decf81bf93cb7bd0b3dba4367f175c7b498b88fa5567764faac2cee62742.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/images/5188decf81bf93cb7bd0b3dba4367f175c7b498b88fa5567764faac2cee62742.jpg)

![54ca68d102bba73da0022eb663ce88d9d704d6bfe4870b1090402b5a8fd1a3e2.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/images/54ca68d102bba73da0022eb663ce88d9d704d6bfe4870b1090402b5a8fd1a3e2.jpg)

![59aae98c7256b7c10839789eba6f5da87ebf8b9cddc0fc8f3b27b254599b4010.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/images/59aae98c7256b7c10839789eba6f5da87ebf8b9cddc0fc8f3b27b254599b4010.jpg)

![5b34af5bd685848592406b3b85852e8ab786b5ad69e0977865ec0333a4c4b23d.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/images/5b34af5bd685848592406b3b85852e8ab786b5ad69e0977865ec0333a4c4b23d.jpg)

![5d72e1bc42f9b3c817609bd2a34d813e305ed63ff4ed1da5b2f00e08e14ed5b6.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/images/5d72e1bc42f9b3c817609bd2a34d813e305ed63ff4ed1da5b2f00e08e14ed5b6.jpg)

![6aace2867c024faac856ba67537530c12e9570c633c8a8062f2f538ce4869b2c.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/images/6aace2867c024faac856ba67537530c12e9570c633c8a8062f2f538ce4869b2c.jpg)

![af6dfdb9ab6c859ad53a5fbe1dc7ccba2429b8cf4ba7ef611c08566b4e016e12.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/images/af6dfdb9ab6c859ad53a5fbe1dc7ccba2429b8cf4ba7ef611c08566b4e016e12.jpg)

![b8e1884c5c3cc3a08013bf722f037548a92994568b3c0bea70690e24433c6490.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/images/b8e1884c5c3cc3a08013bf722f037548a92994568b3c0bea70690e24433c6490.jpg)

![b9b59c7d5f1be5cf00d23943a6c0b9dfb48ffb3b3012ffe70199fd1caa327d8c.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/images/b9b59c7d5f1be5cf00d23943a6c0b9dfb48ffb3b3012ffe70199fd1caa327d8c.jpg)

![cb078896a10e27e79b50d81dda886173415df3f7e91d6f74ac7e926a713b7ff2.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/images/cb078896a10e27e79b50d81dda886173415df3f7e91d6f74ac7e926a713b7ff2.jpg)

![d1b98b7e304bfa7d7e8dea41b3b4d516a199aee2dab5a10b64211819ab59ef5f.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/images/d1b98b7e304bfa7d7e8dea41b3b4d516a199aee2dab5a10b64211819ab59ef5f.jpg)

![dd13918b0a332e2a1bea523b8caf40770398104c6ae27b531435a26e4184abea.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/images/dd13918b0a332e2a1bea523b8caf40770398104c6ae27b531435a26e4184abea.jpg)

![f4731590ee7f7ab08ccdb61e0764337978b4a2644d894cc06f697b2e82c40207.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/images/f4731590ee7f7ab08ccdb61e0764337978b4a2644d894cc06f697b2e82c40207.jpg)

### Tables

![0b32148651e2820b8a440dc4a3da35b04a641a44aff95e3dbbc562610fba63e9.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/tables/0b32148651e2820b8a440dc4a3da35b04a641a44aff95e3dbbc562610fba63e9.jpg)

![29557cc051e9864876956f2020f4da458fa05526e73a3762771a7147f8f363bd.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/tables/29557cc051e9864876956f2020f4da458fa05526e73a3762771a7147f8f363bd.jpg)

![4258024c5ba716b2e5235d6c8fe1e7fedb63af9250934fa1e3dcc8459c24c16a.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/tables/4258024c5ba716b2e5235d6c8fe1e7fedb63af9250934fa1e3dcc8459c24c16a.jpg)

![462daed79d5ff032ce4ed1f9608df05fc940b9787d88b1f3f0e2cbb4f6b8cb47.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/tables/462daed79d5ff032ce4ed1f9608df05fc940b9787d88b1f3f0e2cbb4f6b8cb47.jpg)

![4ceedcc4560ca09030858874f14d2e645434e0db2c1eb0e500a946a48d9421dc.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/tables/4ceedcc4560ca09030858874f14d2e645434e0db2c1eb0e500a946a48d9421dc.jpg)

![504ba3fad122c36a9aa693ace6c860fd87c9c748dfec068a6c10cd241b009dcf.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/tables/504ba3fad122c36a9aa693ace6c860fd87c9c748dfec068a6c10cd241b009dcf.jpg)

![545cc121265259db63feb3afca755126d2b0eddd790c754efd9ee70d5d6f8625.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/tables/545cc121265259db63feb3afca755126d2b0eddd790c754efd9ee70d5d6f8625.jpg)

![5f04ecf6f23639362cb41c00185110618866d670eca20c1407d2cdd472a76bc8.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/tables/5f04ecf6f23639362cb41c00185110618866d670eca20c1407d2cdd472a76bc8.jpg)

![6a787f9e736ca01dc73e9154c2bb746b47e1190624e9d3b245a06322b0bccf90.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/tables/6a787f9e736ca01dc73e9154c2bb746b47e1190624e9d3b245a06322b0bccf90.jpg)

![832602b1e046d7492d8fd2e8239abbde5ee878194b1704bef7e2b6dd7fcd22a2.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/tables/832602b1e046d7492d8fd2e8239abbde5ee878194b1704bef7e2b6dd7fcd22a2.jpg)

![8f7cf17d5b0a75ebb51216bb36bc73e41930dbd91ca8f33cf1b761d43507a009.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/tables/8f7cf17d5b0a75ebb51216bb36bc73e41930dbd91ca8f33cf1b761d43507a009.jpg)

![938448b10482b2c1a7c6a6d93abb1560da7ab38a826829d4246dce58ffed07b7.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/tables/938448b10482b2c1a7c6a6d93abb1560da7ab38a826829d4246dce58ffed07b7.jpg)

![952674331fbee6ac2bc83843c1a18f1d3ddd3be35da0d9a7193b7eef71e55b50.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/tables/952674331fbee6ac2bc83843c1a18f1d3ddd3be35da0d9a7193b7eef71e55b50.jpg)

![9605780ed4215610beb064afab45766a5d24ac1be42a3b6ac294c76b5e110188.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/tables/9605780ed4215610beb064afab45766a5d24ac1be42a3b6ac294c76b5e110188.jpg)

![a7427373c6c9a4695e13a69c2b90ecb3e871be639539163407790927df01bca3.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/tables/a7427373c6c9a4695e13a69c2b90ecb3e871be639539163407790927df01bca3.jpg)

![bffb6590577067b67d0e4a437f9810c2eb68ef7f31c5f06a1cdeec5334ab4b57.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/tables/bffb6590577067b67d0e4a437f9810c2eb68ef7f31c5f06a1cdeec5334ab4b57.jpg)

![c3805af1ba922389d7883c1aced4bfd0df9f187b77cbeaf661106c02e9d4a99a.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/tables/c3805af1ba922389d7883c1aced4bfd0df9f187b77cbeaf661106c02e9d4a99a.jpg)

![eb2439b131b35eab17cc7bf05028fc4838ab135b683367e6316a693a09e5f44a.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/tables/eb2439b131b35eab17cc7bf05028fc4838ab135b683367e6316a693a09e5f44a.jpg)

![ebf86c559dcca7b2c847ab46396fbf330f93dde82e5a24413babba97cbbb8b3d.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/tables/ebf86c559dcca7b2c847ab46396fbf330f93dde82e5a24413babba97cbbb8b3d.jpg)

![ee9f56b829c74100ec8dff8cd0e4e600795b2ad3eb947e14b9e2a6859d3fd729.jpg](../iclr_results/2118_Activation Gradient based Poisoned Sample Detection Against Backdoor Attacks/tables/ee9f56b829c74100ec8dff8cd0e4e600795b2ad3eb947e14b9e2a6859d3fd729.jpg)

## Designing Mechanical Meta-Materials by Learning Equivariant Flows


### Images

![463296d8f4ba39e950a9baef2e4f2df06d52534a9f830610ce834cbbf56fc8ba.jpg](../iclr_results/2119_Designing Mechanical Meta-Materials by Learning Equivariant Flows/images/463296d8f4ba39e950a9baef2e4f2df06d52534a9f830610ce834cbbf56fc8ba.jpg)

![4caf84bc22ffc3039274ec0135c7141c790407b9db2b52a8fd579821ff289881.jpg](../iclr_results/2119_Designing Mechanical Meta-Materials by Learning Equivariant Flows/images/4caf84bc22ffc3039274ec0135c7141c790407b9db2b52a8fd579821ff289881.jpg)

![4e2c4394cc67e27bc2365410da57843b812e57e16abd7188a515436089b5599e.jpg](../iclr_results/2119_Designing Mechanical Meta-Materials by Learning Equivariant Flows/images/4e2c4394cc67e27bc2365410da57843b812e57e16abd7188a515436089b5599e.jpg)

![50ee8d375caff7e021c19e54636369e413163b0fb5c7ccc6df7750e66d9287c4.jpg](../iclr_results/2119_Designing Mechanical Meta-Materials by Learning Equivariant Flows/images/50ee8d375caff7e021c19e54636369e413163b0fb5c7ccc6df7750e66d9287c4.jpg)

![555d3a0567b403a37c0fa0cbae61b5a03eda3a5f606e15f19dbf2a71e18a1a55.jpg](../iclr_results/2119_Designing Mechanical Meta-Materials by Learning Equivariant Flows/images/555d3a0567b403a37c0fa0cbae61b5a03eda3a5f606e15f19dbf2a71e18a1a55.jpg)

![b828a81c253f0bf0f5db1f9c8c019f75b26270d05de10474012ae89a147ad7d6.jpg](../iclr_results/2119_Designing Mechanical Meta-Materials by Learning Equivariant Flows/images/b828a81c253f0bf0f5db1f9c8c019f75b26270d05de10474012ae89a147ad7d6.jpg)

![bc8b0c7f6e3ce62ca185664d9a74548d4bda0bebf1c4e43a9bbdebb019a3ed8f.jpg](../iclr_results/2119_Designing Mechanical Meta-Materials by Learning Equivariant Flows/images/bc8b0c7f6e3ce62ca185664d9a74548d4bda0bebf1c4e43a9bbdebb019a3ed8f.jpg)

![c5c7cd2891cc6c6ac5fbd6abd23367a73f1de3a20c6ba9fd56ea812a5bc50c0f.jpg](../iclr_results/2119_Designing Mechanical Meta-Materials by Learning Equivariant Flows/images/c5c7cd2891cc6c6ac5fbd6abd23367a73f1de3a20c6ba9fd56ea812a5bc50c0f.jpg)

![e1b3cf15730ec0fd8155f97505f87bc206d54163e0778b3fc4a071ef3b608400.jpg](../iclr_results/2119_Designing Mechanical Meta-Materials by Learning Equivariant Flows/images/e1b3cf15730ec0fd8155f97505f87bc206d54163e0778b3fc4a071ef3b608400.jpg)

![f0cbdbece7d59a82490d23f251c7a666b5c55930887ec35324afec0682a1ffd6.jpg](../iclr_results/2119_Designing Mechanical Meta-Materials by Learning Equivariant Flows/images/f0cbdbece7d59a82490d23f251c7a666b5c55930887ec35324afec0682a1ffd6.jpg)

![f50f0148d958c3eb97db49807ddc3d6dc3b19fca2cd63d2fdb522660e88b0a2e.jpg](../iclr_results/2119_Designing Mechanical Meta-Materials by Learning Equivariant Flows/images/f50f0148d958c3eb97db49807ddc3d6dc3b19fca2cd63d2fdb522660e88b0a2e.jpg)

![fae10cd15437ef97c507552970371067f51b4c4b45db8c9f41ebe1b9a42bb83e.jpg](../iclr_results/2119_Designing Mechanical Meta-Materials by Learning Equivariant Flows/images/fae10cd15437ef97c507552970371067f51b4c4b45db8c9f41ebe1b9a42bb83e.jpg)

### Tables

![c833e28833105b3fcc8bf67861566172b11da9f4a01354a6999cda3f9b38322a.jpg](../iclr_results/2119_Designing Mechanical Meta-Materials by Learning Equivariant Flows/tables/c833e28833105b3fcc8bf67861566172b11da9f4a01354a6999cda3f9b38322a.jpg)

![cad5f39ec307851dd3c553bd363fdfe088cd45f58233eec04d0227e708b0c453.jpg](../iclr_results/2119_Designing Mechanical Meta-Materials by Learning Equivariant Flows/tables/cad5f39ec307851dd3c553bd363fdfe088cd45f58233eec04d0227e708b0c453.jpg)

## MCNC: Manifold-Constrained Reparameterization for Neural Compression


### Images

![93afb58e70e19df645ce2bd34b1837375245f309f35d78507e18d9d141cdb5ed.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/images/93afb58e70e19df645ce2bd34b1837375245f309f35d78507e18d9d141cdb5ed.jpg)

![a4a94441be833f0980976f960fefa61d3aefdeebc5d25ba7f55cae085342579d.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/images/a4a94441be833f0980976f960fefa61d3aefdeebc5d25ba7f55cae085342579d.jpg)

### Tables

![11ba5b9705dce1cd7084c66e4cdee58980113c5afb664af0fdd4bc72417594cc.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/11ba5b9705dce1cd7084c66e4cdee58980113c5afb664af0fdd4bc72417594cc.jpg)

![29cb8bd14bf97c21db3b242e99d08e3912939c3cb174eb2bb6ec78d274aae079.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/29cb8bd14bf97c21db3b242e99d08e3912939c3cb174eb2bb6ec78d274aae079.jpg)

![2d46363897404ebee86c1f7fecbc6602f30e3156005d6ef5bf434d45f20fc7f7.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/2d46363897404ebee86c1f7fecbc6602f30e3156005d6ef5bf434d45f20fc7f7.jpg)

![2d4669fbe7e43a401a8806dd1b45171195950c56efe1edac58a7b197c03fe0ff.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/2d4669fbe7e43a401a8806dd1b45171195950c56efe1edac58a7b197c03fe0ff.jpg)

![3ee3b071c48591cb401abbdc54e5bdb9ed045c3fe5dc3354027e5484d0ff7651.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/3ee3b071c48591cb401abbdc54e5bdb9ed045c3fe5dc3354027e5484d0ff7651.jpg)

![6409ac9ebf7d373c8159918e0995f02c594ce81019c7bbbfa3034fa636b7757f.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/6409ac9ebf7d373c8159918e0995f02c594ce81019c7bbbfa3034fa636b7757f.jpg)

![6c1fd9abe0da280adecd9ec85c931ef1ccb9b4ea631aef782b25371e41c365cb.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/6c1fd9abe0da280adecd9ec85c931ef1ccb9b4ea631aef782b25371e41c365cb.jpg)

![79e936261e8e2e144421c1b7a077ca2b6bf9f02907fb08722f36826ed3a17c4d.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/79e936261e8e2e144421c1b7a077ca2b6bf9f02907fb08722f36826ed3a17c4d.jpg)

![7da0337723860eff1b03f0309d72f6515a157d0703cfce180e18c17ddcfc255a.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/7da0337723860eff1b03f0309d72f6515a157d0703cfce180e18c17ddcfc255a.jpg)

![7e588f5d39e6a9f374da2e7e39f0f7ca4d92dd9aaf0a7fc5e6ccd9a0b6f72186.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/7e588f5d39e6a9f374da2e7e39f0f7ca4d92dd9aaf0a7fc5e6ccd9a0b6f72186.jpg)

![878191f45108b29d41e5cb88d8254d7f42ec19f0be4239a3e90b7829196e4fc5.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/878191f45108b29d41e5cb88d8254d7f42ec19f0be4239a3e90b7829196e4fc5.jpg)

![8c04dd367951baf37f2ca7e0380899589f0ff2f3f132dc8e2c5445d0d7fded4b.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/8c04dd367951baf37f2ca7e0380899589f0ff2f3f132dc8e2c5445d0d7fded4b.jpg)

![8cbd8ca365465943e7e1b35577546a7231b58e8c2dc88065ca3ccd2a7d629c2e.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/8cbd8ca365465943e7e1b35577546a7231b58e8c2dc88065ca3ccd2a7d629c2e.jpg)

![9102ca7566a850f6b7e8700082d4290091e2c6e083afee1752ed45ac7d6b40eb.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/9102ca7566a850f6b7e8700082d4290091e2c6e083afee1752ed45ac7d6b40eb.jpg)

![97e0ccd9354a3e27ed7bd24874a1f668faaaea20a4ce98179bbdd69c5af3ff9f.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/97e0ccd9354a3e27ed7bd24874a1f668faaaea20a4ce98179bbdd69c5af3ff9f.jpg)

![d09ab8664fec8deb8b706138c6190bb41c39823f3144fe0862a752447f17e453.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/d09ab8664fec8deb8b706138c6190bb41c39823f3144fe0862a752447f17e453.jpg)

![d3aeab703cfa444622f0d693ce17aebea0f27fb9f59fffb7a0335a505de34647.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/d3aeab703cfa444622f0d693ce17aebea0f27fb9f59fffb7a0335a505de34647.jpg)

![ed429b394fba2dff18c2b6ac00f7e3e49d4cd0002dff02c904f5ceb6118f0044.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/ed429b394fba2dff18c2b6ac00f7e3e49d4cd0002dff02c904f5ceb6118f0044.jpg)

## TypedThinker: Diversify Large Language Model Reasoning with Typed Thinking


### Images

![1f190da7250a45f860a672a4500ddf1ba2286058e6d7c7fc4cd3a7382720c15f.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/images/1f190da7250a45f860a672a4500ddf1ba2286058e6d7c7fc4cd3a7382720c15f.jpg)

![5439223283ff60790ec4c0a6825b5ad99b045f122e749d001da37691819439bb.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/images/5439223283ff60790ec4c0a6825b5ad99b045f122e749d001da37691819439bb.jpg)

![6eb9bb1910d17cc3cc831a0064d08a721468f3df543be8cc8965e5b208042978.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/images/6eb9bb1910d17cc3cc831a0064d08a721468f3df543be8cc8965e5b208042978.jpg)

![7f02543f27b5558097956bb35f231c3fce5629a0fc27867106634b04edc00640.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/images/7f02543f27b5558097956bb35f231c3fce5629a0fc27867106634b04edc00640.jpg)

![b78a7a3a5fe9d1dba9777327a1a0961a6b7e855a27b2aefa93f92e677b6fa92d.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/images/b78a7a3a5fe9d1dba9777327a1a0961a6b7e855a27b2aefa93f92e677b6fa92d.jpg)

![d8c6434a9006d83cff7f896672ae18f38f02dc94736c812a82479b8067c12654.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/images/d8c6434a9006d83cff7f896672ae18f38f02dc94736c812a82479b8067c12654.jpg)

### Tables

![176e27c01fb4c16f76584f86f1cde7a74d2b40798b654bf49fa9ea3b1d92bf56.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/176e27c01fb4c16f76584f86f1cde7a74d2b40798b654bf49fa9ea3b1d92bf56.jpg)

![1f2cd394df85463853bdf08d60d9f3fe9bbab572cb62f696109ca149da0d497a.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/1f2cd394df85463853bdf08d60d9f3fe9bbab572cb62f696109ca149da0d497a.jpg)

![20027d48fc465ffd831e22e11932110222acb4caeb501487a6c02c0f3626c9bb.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/20027d48fc465ffd831e22e11932110222acb4caeb501487a6c02c0f3626c9bb.jpg)

![25cf1aaa2dfe588e540db6268dd018173a724815904fecca195677b41a3ab616.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/25cf1aaa2dfe588e540db6268dd018173a724815904fecca195677b41a3ab616.jpg)

![27f15e3a05cdb1ef89a606f9baa0fc5e40a1579bafb0571d7f49772cca5b6e33.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/27f15e3a05cdb1ef89a606f9baa0fc5e40a1579bafb0571d7f49772cca5b6e33.jpg)

![3285f67317f2012481a3a4c05522e7184fd18f96532312eb7e23555914abc1ee.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/3285f67317f2012481a3a4c05522e7184fd18f96532312eb7e23555914abc1ee.jpg)

![41faaa5a96b9a2130bfc6ce5781e93f36388934948e72a520f9672bb12f28b1f.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/41faaa5a96b9a2130bfc6ce5781e93f36388934948e72a520f9672bb12f28b1f.jpg)

![5ac2a6be0387286b2209c50b7f2200dc19bfd40c29413fc7c5ba398073059125.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/5ac2a6be0387286b2209c50b7f2200dc19bfd40c29413fc7c5ba398073059125.jpg)

![694e7fd591d64ed27c7944ebbd3e8e3a7e3581f0aadc4faecd95a4bcffb6bc58.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/694e7fd591d64ed27c7944ebbd3e8e3a7e3581f0aadc4faecd95a4bcffb6bc58.jpg)

![7efbf059505b898fc24986637a997b2acddb6924169b074cc011fff11725540b.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/7efbf059505b898fc24986637a997b2acddb6924169b074cc011fff11725540b.jpg)

![a7a807d0eeba60f5d7244edd804486d679863887e3472700c47639ba25e6a1fb.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/a7a807d0eeba60f5d7244edd804486d679863887e3472700c47639ba25e6a1fb.jpg)

![aa5850810bca84eb4ac511a72a657592bfc343a14a60a8451960f6a4998a3ca9.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/aa5850810bca84eb4ac511a72a657592bfc343a14a60a8451960f6a4998a3ca9.jpg)

![b1e155cc8bef5796ac14709b203dfb306a4b08b644810dd3985daeea81d4f4c6.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/b1e155cc8bef5796ac14709b203dfb306a4b08b644810dd3985daeea81d4f4c6.jpg)

![cd49e0252223a577b3974b3dc57ef5f839fbd1159d9c7611a6e94880a4b5951b.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/cd49e0252223a577b3974b3dc57ef5f839fbd1159d9c7611a6e94880a4b5951b.jpg)

![cfbb80c9bee1c44643f83628da2affa56f37b44f41560e98a8060f8e03a821ad.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/cfbb80c9bee1c44643f83628da2affa56f37b44f41560e98a8060f8e03a821ad.jpg)

![dd5ea29150bddc3583d3f835222da5280fba71e5083bc6b3bfae13b1e568b8a3.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/dd5ea29150bddc3583d3f835222da5280fba71e5083bc6b3bfae13b1e568b8a3.jpg)

## SEAL: Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection


### Images

![078e37192de3a13e8a4940142ba4ec162efce4f930f8c80cae1770d8b9d5329c.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/images/078e37192de3a13e8a4940142ba4ec162efce4f930f8c80cae1770d8b9d5329c.jpg)

![184fa5435de66d22882430742ed74750061aea64f596cd8eb05826488cf29c39.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/images/184fa5435de66d22882430742ed74750061aea64f596cd8eb05826488cf29c39.jpg)

![243ea7bcc044ce03e2982df0757184ef702974aa1f8627c43294799a2b0b5a57.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/images/243ea7bcc044ce03e2982df0757184ef702974aa1f8627c43294799a2b0b5a57.jpg)

![3baaaff459daea8556ad4d53df4cf3bdde51215fa3b13f88893c52167d12ca78.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/images/3baaaff459daea8556ad4d53df4cf3bdde51215fa3b13f88893c52167d12ca78.jpg)

![5f7baf735a4c2326fd4a362a90ff5ed15351cfbc381de89ee4bcf582a1a79938.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/images/5f7baf735a4c2326fd4a362a90ff5ed15351cfbc381de89ee4bcf582a1a79938.jpg)

![6bb2b97932b5d21e1d19fd91ac364b719afdf9946a48e4cf4b1fc7b36d5b994a.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/images/6bb2b97932b5d21e1d19fd91ac364b719afdf9946a48e4cf4b1fc7b36d5b994a.jpg)

![6cf968073d75ae86febbb5af542eeac40ebe7efa7957e0945ca6c80944a21c79.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/images/6cf968073d75ae86febbb5af542eeac40ebe7efa7957e0945ca6c80944a21c79.jpg)

![80e66421eb1424f8622d29a5521c40eedc4339d271c33be902fb003fbeaa6290.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/images/80e66421eb1424f8622d29a5521c40eedc4339d271c33be902fb003fbeaa6290.jpg)

![ed00fb0e915b1102914b3b79d08f964dfb7e9e3cbd0d804781b3110fe8a003cd.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/images/ed00fb0e915b1102914b3b79d08f964dfb7e9e3cbd0d804781b3110fe8a003cd.jpg)

### Tables

![1a9b3b528215cdd20fdc602190faa78a19f476e192a33ac55e2450e70efc3818.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/tables/1a9b3b528215cdd20fdc602190faa78a19f476e192a33ac55e2450e70efc3818.jpg)

![2e7c4c809544ec80e10482ef70fc881680875d1cdc3dac1544fe762b54cde3e5.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/tables/2e7c4c809544ec80e10482ef70fc881680875d1cdc3dac1544fe762b54cde3e5.jpg)

![406a16b00e68732d88a3f01942cb4d76e474dce5ebaba99de858f87943132e0d.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/tables/406a16b00e68732d88a3f01942cb4d76e474dce5ebaba99de858f87943132e0d.jpg)

![55f8b03c3151e28b517093fae14654df19a64adc364df8bc0fd4b052f5ec24cd.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/tables/55f8b03c3151e28b517093fae14654df19a64adc364df8bc0fd4b052f5ec24cd.jpg)

![5a6708878d7008648598d17cf13b79c723bcbedb7dee0ba95bd7932388d08e0c.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/tables/5a6708878d7008648598d17cf13b79c723bcbedb7dee0ba95bd7932388d08e0c.jpg)

![5aa2e163619754f5c0e7bf444b9970eca88eeec6ee2b9fa7c0d8f39e2ec93448.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/tables/5aa2e163619754f5c0e7bf444b9970eca88eeec6ee2b9fa7c0d8f39e2ec93448.jpg)

![71e36ca59f1bfe1aa91a8342c7be9dd2ddf9311352119121fe3cfc79c430269c.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/tables/71e36ca59f1bfe1aa91a8342c7be9dd2ddf9311352119121fe3cfc79c430269c.jpg)

![9cb98e9627f87d6f76cf03739279e0d2364c36dbc17b314ab7c9b53ac33c2802.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/tables/9cb98e9627f87d6f76cf03739279e0d2364c36dbc17b314ab7c9b53ac33c2802.jpg)

![a698fa12977bea1d8f3a91750d58baba865b21e6916bf0f732a1b07ff17b1385.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/tables/a698fa12977bea1d8f3a91750d58baba865b21e6916bf0f732a1b07ff17b1385.jpg)

![ab078c94188ca9dce14a66f292bbabf6ab348e1ba94b4d3a637897c9fd5ebbd6.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/tables/ab078c94188ca9dce14a66f292bbabf6ab348e1ba94b4d3a637897c9fd5ebbd6.jpg)

## Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport


### Images

![01ed98587b943a272919f5a638fba66379c7d10fb53b423f4d9e767ce6a196d0.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/images/01ed98587b943a272919f5a638fba66379c7d10fb53b423f4d9e767ce6a196d0.jpg)

![66a543967df0d2013c4a24182b17e398d6e0459a34f472b7df240089038d9519.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/images/66a543967df0d2013c4a24182b17e398d6e0459a34f472b7df240089038d9519.jpg)

![b9918abc3704483363dbcea56d64c6ba2e532bfabbbe29a6e67ec1bc72577e59.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/images/b9918abc3704483363dbcea56d64c6ba2e532bfabbbe29a6e67ec1bc72577e59.jpg)

![db890d970e89c5feecd0e9297ff1d8963d4d5ab2aedff8bb0e026bd034c61f28.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/images/db890d970e89c5feecd0e9297ff1d8963d4d5ab2aedff8bb0e026bd034c61f28.jpg)

![e3c495af33a3a200dd3715a15d99181678e360a3991b7f55e4f08df430671224.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/images/e3c495af33a3a200dd3715a15d99181678e360a3991b7f55e4f08df430671224.jpg)

![fdf57da2750d81445c3100e621e7fbcdc0762018433f57e4d1167875c734b2cf.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/images/fdf57da2750d81445c3100e621e7fbcdc0762018433f57e4d1167875c734b2cf.jpg)

### Tables

![1951265d88e2a96995cff428314b202550cc2b8bc98a7e359b35bcb5987affff.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/tables/1951265d88e2a96995cff428314b202550cc2b8bc98a7e359b35bcb5987affff.jpg)

![291c818f439ab9450d9c642594512d010c5c789b8d6e1cc3429503cb549331e5.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/tables/291c818f439ab9450d9c642594512d010c5c789b8d6e1cc3429503cb549331e5.jpg)

![4abd0298d34fbedb6c8b4769b72e77a320f58ec9a0c7c6f176f61ea01ed3da05.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/tables/4abd0298d34fbedb6c8b4769b72e77a320f58ec9a0c7c6f176f61ea01ed3da05.jpg)

![677eba64b0b85f1e0df50e761a73bfacac72a673bf2af4f185bbccf0ab7c10e7.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/tables/677eba64b0b85f1e0df50e761a73bfacac72a673bf2af4f185bbccf0ab7c10e7.jpg)

![850b1b976479d8f39d989361e24b9ba40b7504b1b7f1e32fbac5be7267d320d7.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/tables/850b1b976479d8f39d989361e24b9ba40b7504b1b7f1e32fbac5be7267d320d7.jpg)

![d9bc8641d1717dbd29d4732c3630ff9b9bea21075292f254ff8c420b57a336ef.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/tables/d9bc8641d1717dbd29d4732c3630ff9b9bea21075292f254ff8c420b57a336ef.jpg)

![f590e847c9f080b4e323a297f26333a0a8dfb69bd0bc821e0cb7d802bf273197.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/tables/f590e847c9f080b4e323a297f26333a0a8dfb69bd0bc821e0cb7d802bf273197.jpg)

## Neural Dueling Bandits: Preference-Based Optimization with Human Feedback


### Images

![0c99e0fbbb70e1d04ec1d2c02d0c3d3f986222cf16fb195fa12290da455e40f3.jpg](../iclr_results/2124_Neural Dueling Bandits_ Preference-Based Optimization with Human Feedback/images/0c99e0fbbb70e1d04ec1d2c02d0c3d3f986222cf16fb195fa12290da455e40f3.jpg)

![15654db7f0ef8cc5c4470a74ea601c9adb9172c397472eb400b0dec8c1473c93.jpg](../iclr_results/2124_Neural Dueling Bandits_ Preference-Based Optimization with Human Feedback/images/15654db7f0ef8cc5c4470a74ea601c9adb9172c397472eb400b0dec8c1473c93.jpg)

![27444cf1199e3baa648aa66215022309de92f2a8f0ff01e949a4a7d98cc939c0.jpg](../iclr_results/2124_Neural Dueling Bandits_ Preference-Based Optimization with Human Feedback/images/27444cf1199e3baa648aa66215022309de92f2a8f0ff01e949a4a7d98cc939c0.jpg)

![52d909aab8c08e099660c9f61510b0b6703c5a6b8c2df57153764e605bf0af41.jpg](../iclr_results/2124_Neural Dueling Bandits_ Preference-Based Optimization with Human Feedback/images/52d909aab8c08e099660c9f61510b0b6703c5a6b8c2df57153764e605bf0af41.jpg)

![6343216ed37a86f5065375e819436d920b7c39431249ce79ac18fcb21848039c.jpg](../iclr_results/2124_Neural Dueling Bandits_ Preference-Based Optimization with Human Feedback/images/6343216ed37a86f5065375e819436d920b7c39431249ce79ac18fcb21848039c.jpg)

![6ff5d815be635b45431bb912836344b99d75f0f1e2d507533e7ed2b5f16eb567.jpg](../iclr_results/2124_Neural Dueling Bandits_ Preference-Based Optimization with Human Feedback/images/6ff5d815be635b45431bb912836344b99d75f0f1e2d507533e7ed2b5f16eb567.jpg)

![7ebbcfa9f0cbd15cd9fa1134a835779a74c1e329fe060968a8988e3ba7f8a108.jpg](../iclr_results/2124_Neural Dueling Bandits_ Preference-Based Optimization with Human Feedback/images/7ebbcfa9f0cbd15cd9fa1134a835779a74c1e329fe060968a8988e3ba7f8a108.jpg)

![88833834c53a1af3be38d35385897861864b878bf33cdfd769c3ad84d8f9f43f.jpg](../iclr_results/2124_Neural Dueling Bandits_ Preference-Based Optimization with Human Feedback/images/88833834c53a1af3be38d35385897861864b878bf33cdfd769c3ad84d8f9f43f.jpg)

![de4479d9682f3b06ff68ae40e42f42d2c311545d316c02094a44a44e770cb05d.jpg](../iclr_results/2124_Neural Dueling Bandits_ Preference-Based Optimization with Human Feedback/images/de4479d9682f3b06ff68ae40e42f42d2c311545d316c02094a44a44e770cb05d.jpg)

## Divergence of Neural Tangent Kernel in Classification Problems


### Images

![347f66e26e76b04d7f9e120b9ce64c3b577f399db36c586eea3741f0aacf9680.jpg](../iclr_results/2125_Divergence of Neural Tangent Kernel in Classification Problems/images/347f66e26e76b04d7f9e120b9ce64c3b577f399db36c586eea3741f0aacf9680.jpg)

![70b7c5a09fe4185e1703713e5a338aab38ad301ddbef248129263d3043556e7d.jpg](../iclr_results/2125_Divergence of Neural Tangent Kernel in Classification Problems/images/70b7c5a09fe4185e1703713e5a338aab38ad301ddbef248129263d3043556e7d.jpg)

![8a2af81ab9c8ab3ccd6e5d2d11982ae077f9e576e0adeee00bd01a44045afe89.jpg](../iclr_results/2125_Divergence of Neural Tangent Kernel in Classification Problems/images/8a2af81ab9c8ab3ccd6e5d2d11982ae077f9e576e0adeee00bd01a44045afe89.jpg)

![d876d7d2ecad6ef8211f858d370234a4072e779917eda2e4a465beb1a19bf9b3.jpg](../iclr_results/2125_Divergence of Neural Tangent Kernel in Classification Problems/images/d876d7d2ecad6ef8211f858d370234a4072e779917eda2e4a465beb1a19bf9b3.jpg)

## Regret Bounds for Episodic Risk-Sensitive Linear Quadratic Regulator

### Images

![006632cb6fcd5a4dbfca67c3ad765d2d7563ee0a2b8542a3dbbc051e12a4d19c.jpg](../iclr_results/2126_Regret Bounds for Episodic Risk-Sensitive Linear Quadratic Regulator/images/006632cb6fcd5a4dbfca67c3ad765d2d7563ee0a2b8542a3dbbc051e12a4d19c.jpg)

![049b0f9c7abe37d212d9478934d0cbf179cc5efdc0fad73599481ff8f4f9405a.jpg](../iclr_results/2126_Regret Bounds for Episodic Risk-Sensitive Linear Quadratic Regulator/images/049b0f9c7abe37d212d9478934d0cbf179cc5efdc0fad73599481ff8f4f9405a.jpg)

![42c2833ee8527d5e50fed26dc00af4c0382b77262717aef492bb057bdb29e0d7.jpg](../iclr_results/2126_Regret Bounds for Episodic Risk-Sensitive Linear Quadratic Regulator/images/42c2833ee8527d5e50fed26dc00af4c0382b77262717aef492bb057bdb29e0d7.jpg)
