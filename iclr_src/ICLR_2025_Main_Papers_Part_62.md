# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 62 of 100

## 目录 (Table of Contents)

1. [Conservative Contextual Bandits: Beyond Linear Representations](#Conservative-Contextual-Bandits-Beyond-Linear-Representations)
2. [Is In-Context Learning Sufficient for Instruction Following in LLMs?](#Is-In-Context-Learning-Sufficient-for-Instruction-Following-in-LLMs)
3. [VideoGrain: Modulating Space-Time Attention for Multi-Grained Video Editing](#VideoGrain-Modulating-Space-Time-Attention-for-Multi-Grained-Video-Editing)
4. [3D Vision-Language Gaussian Splatting](#3D-Vision-Language-Gaussian-Splatting)
5. [Fat-to-Thin Policy Optimization: Offline Reinforcement Learning with Sparse Policies](#Fat-to-Thin-Policy-Optimization-Offline-Reinforcement-Learning-with-Sparse-Policies)
6. [Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probability Ratio Test](#Learning-the-Optimal-Stopping-for-Early-Classification-within-Finite-Horizons-via-Sequential-Probability-Ratio-Test)
7. [Value-Incentivized Preference Optimization: A Unified Approach to Online and Offline RLHF](#Value-Incentivized-Preference-Optimization-A-Unified-Approach-to-Online-and-Offline-RLHF)
8. [Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion](#Learnable-Expansion-of-Graph-Operators-for-Multi-Modal-Feature-Fusion)
9. [Language Models Are Implicitly Continuous](#Language-Models-Are-Implicitly-Continuous)
10. [Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias](#Towards-Understanding-Text-Hallucination-of-Diffusion-Models-via-Local-Generation-Bias)
11. [High-Dimensional Bayesian Optimisation with Gaussian Process Prior Variational Autoencoders](#High-Dimensional-Bayesian-Optimisation-with-Gaussian-Process-Prior-Variational-Autoencoders)
12. [Learning to Clarify: Multi-turn Conversations with Action-Based Contrastive Self-Training](#Learning-to-Clarify-Multi-turn-Conversations-with-Action-Based-Contrastive-Self-Training)
13. [NExUME: Adaptive Training and Inference for DNNs under Intermittent Power Environments](#NExUME-Adaptive-Training-and-Inference-for-DNNs-under-Intermittent-Power-Environments)
14. [TorchTitan: One-stop PyTorch native solution for production ready LLM pretraining](#TorchTitan-One-stop-PyTorch-native-solution-for-production-ready-LLM-pretraining)
15. [Surgical, Cheap, and Flexible: Mitigating False Refusal in Language Models via Single Vector Ablation](#Surgical-Cheap-and-Flexible-Mitigating-False-Refusal-in-Language-Models-via-Single-Vector-Ablation)
16. [HiSplat: Hierarchical 3D Gaussian Splatting for Generalizable Sparse-View Reconstruction](#HiSplat-Hierarchical-3D-Gaussian-Splatting-for-Generalizable-Sparse-View-Reconstruction)
17. [GROOT-2: Weakly Supervised Multimodal Instruction Following Agents](#GROOT-2-Weakly-Supervised-Multimodal-Instruction-Following-Agents)
18. [Redefining the task of Bioactivity Prediction](#Redefining-the-task-of-Bioactivity-Prediction)
19. [Cafe-Talk: Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control](#Cafe-Talk-Generating-3D-Talking-Face-Animation-with-Multimodal-Coarse-and-Fine-grained-Control)
20. [Can We Trust Embodied Agents? Exploring Backdoor Attacks against Embodied LLM-Based Decision-Making Systems](#Can-We-Trust-Embodied-Agents-Exploring-Backdoor-Attacks-against-Embodied-LLM-Based-Decision-Making-Systems)
21. [Concept-ROT: Poisoning Concepts in Large Language Models with Model Editing](#Concept-ROT-Poisoning-Concepts-in-Large-Language-Models-with-Model-Editing)
22. [Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned Datasets](#Medium-Difficulty-Samples-Constitute-Smoothed-Decision-Boundary-for-Knowledge-Distillation-on-Pruned-Datasets)
23. [Mixture of Attentions For Speculative Decoding](#Mixture-of-Attentions-For-Speculative-Decoding)
24. [Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs](#Reframing-Structure-Based-Drug-Design-Model-Evaluation-via-Metrics-Correlated-to-Practical-Needs)
25. [Vec2Face: Scaling Face Dataset Generation with Loosely Constrained Vectors](#Vec2Face-Scaling-Face-Dataset-Generation-with-Loosely-Constrained-Vectors)
26. [A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics](#A-Riemannian-Framework-for-Learning-Reduced-order-Lagrangian-Dynamics)
27. [CLDyB: Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models](#CLDyB-Towards-Dynamic-Benchmarking-for-Continual-Learning-with-Pre-trained-Models)
28. [Sensor-Invariant Tactile Representation](#Sensor-Invariant-Tactile-Representation)
29. [Diffusion Models as Cartoonists: The Curious Case of High Density Regions](#Diffusion-Models-as-Cartoonists-The-Curious-Case-of-High-Density-Regions)
30. [3DMolFormer: A Dual-channel Framework for Structure-based Drug Discovery](#3DMolFormer-A-Dual-channel-Framework-for-Structure-based-Drug-Discovery)
31. [Optimality and Adaptivity of Deep Neural Features for Instrumental Variable Regression](#Optimality-and-Adaptivity-of-Deep-Neural-Features-for-Instrumental-Variable-Regression)
32. [Shape as Line Segments: Accurate and Flexible Implicit Surface Representation](#Shape-as-Line-Segments-Accurate-and-Flexible-Implicit-Surface-Representation)
33. [HOPE for a Robust Parameterization of Long-memory State Space Models](#HOPE-for-a-Robust-Parameterization-of-Long-memory-State-Space-Models)
34. [Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning](#Unleashing-the-Power-of-Task-Specific-Directions-in-Parameter-Efficient-Fine-tuning)
35. [Kronecker Mask and Interpretive Prompts are Language-Action Video Learners](#Kronecker-Mask-and-Interpretive-Prompts-are-Language-Action-Video-Learners)
36. [Large Language Models Often Say One Thing and Do Another](#Large-Language-Models-Often-Say-One-Thing-and-Do-Another)
37. [A Theory of Initialisation's Impact on Specialisation](#A-Theory-of-Initialisations-Impact-on-Specialisation)

---


## Conservative Contextual Bandits: Beyond Linear Representations

### Images

![206c02c8ba1b76a3f5f91aecd3cb700dd7a5c25a970ec5576ef5ebc618f33120.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/images/206c02c8ba1b76a3f5f91aecd3cb700dd7a5c25a970ec5576ef5ebc618f33120.jpg)

![2baca0cc2378fef51b2b35c5bbdd976b19a61b8b017a47125703d419cca525fa.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/images/2baca0cc2378fef51b2b35c5bbdd976b19a61b8b017a47125703d419cca525fa.jpg)

![3b86eefcef05c22575d48793dc81a5200399fb8fdcbc18f47e184c13a7f4b471.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/images/3b86eefcef05c22575d48793dc81a5200399fb8fdcbc18f47e184c13a7f4b471.jpg)

![4ee09ec76eadb6fa69828963e96465f9f6f2a3a51d75a7d5ba15f3f457f5e4b4.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/images/4ee09ec76eadb6fa69828963e96465f9f6f2a3a51d75a7d5ba15f3f457f5e4b4.jpg)

![5bc7cdc3225426795987a5931e302e86c89f856498534b049d9c85a11af1c81f.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/images/5bc7cdc3225426795987a5931e302e86c89f856498534b049d9c85a11af1c81f.jpg)

![718e24531b309ec6eac3df57299cab61c15dce4d4c1300d80f20c6a2aa3c8792.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/images/718e24531b309ec6eac3df57299cab61c15dce4d4c1300d80f20c6a2aa3c8792.jpg)

![7c181c977e1df1d99200e1fa4721ec7e7918f5468f304aa352f1f515b339bbb1.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/images/7c181c977e1df1d99200e1fa4721ec7e7918f5468f304aa352f1f515b339bbb1.jpg)

![7f3098edabaed60524645bb86de0f70dbd02ba76b21f0c2c733321fa917ecedd.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/images/7f3098edabaed60524645bb86de0f70dbd02ba76b21f0c2c733321fa917ecedd.jpg)

![88b2930d4be3d9800a26346607899bc83a33284e201025c2091ba931dce8353a.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/images/88b2930d4be3d9800a26346607899bc83a33284e201025c2091ba931dce8353a.jpg)

![9d349e860d86e9661b6e851f568673af9fa42023f146891f33f504fb83102a9f.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/images/9d349e860d86e9661b6e851f568673af9fa42023f146891f33f504fb83102a9f.jpg)

![ac9801965c63027d735c07502f62d2b1f50bb9e7332b593640f84658ff416d2f.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/images/ac9801965c63027d735c07502f62d2b1f50bb9e7332b593640f84658ff416d2f.jpg)

![acc986c12a7e65f5dee9c6c262a31baacc04e760a4ac678de6ef885e2ddada8a.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/images/acc986c12a7e65f5dee9c6c262a31baacc04e760a4ac678de6ef885e2ddada8a.jpg)

![d266384e267e37a635055c5508a4b34a0b4b59908c6a66a0ae5a5c71c37fa6cb.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/images/d266384e267e37a635055c5508a4b34a0b4b59908c6a66a0ae5a5c71c37fa6cb.jpg)

![df28c0c6c44611d793ce209841d0745666af0896a194d106d344298809ccbe51.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/images/df28c0c6c44611d793ce209841d0745666af0896a194d106d344298809ccbe51.jpg)

![fb8f950a6ee839ed7062ab6e27b6bd24e50cb9ce37b7c8e7466997b90573fac7.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/images/fb8f950a6ee839ed7062ab6e27b6bd24e50cb9ce37b7c8e7466997b90573fac7.jpg)

![ff10860e78bab15128c439056473a307621c4d35744c3fc139e99c41466800b9.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/images/ff10860e78bab15128c439056473a307621c4d35744c3fc139e99c41466800b9.jpg)

### Tables

![0388daac2159ae04c721bc3dceea94d65c241a91ae71b0d60bf8228207ed7cca.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/tables/0388daac2159ae04c721bc3dceea94d65c241a91ae71b0d60bf8228207ed7cca.jpg)

![08d0e5c09e770aa75246ba4b8ad7a5aaef36153869d4eb79eed1b29332cda2b7.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/tables/08d0e5c09e770aa75246ba4b8ad7a5aaef36153869d4eb79eed1b29332cda2b7.jpg)

![1d22cedb6edff5210cfa183e23415f56af363de6c6af4aba3a250ee1d45b613e.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/tables/1d22cedb6edff5210cfa183e23415f56af363de6c6af4aba3a250ee1d45b613e.jpg)

![2b1c941f2fe468aa02c020a3dc02e7efce1c4a2cb1b84aaa1780c5e66103a2e8.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/tables/2b1c941f2fe468aa02c020a3dc02e7efce1c4a2cb1b84aaa1780c5e66103a2e8.jpg)

![2c93fe62f15a9646067ef0be3fdc240364bd6b5cfbebe189a1c58ac10ed229f3.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/tables/2c93fe62f15a9646067ef0be3fdc240364bd6b5cfbebe189a1c58ac10ed229f3.jpg)

![538df1cf9cf4c94d11e916f77788dd14a4aef71b4ada96a413431ca1ea1f3055.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/tables/538df1cf9cf4c94d11e916f77788dd14a4aef71b4ada96a413431ca1ea1f3055.jpg)

![6732f274e529ba1f5a51554182b1d49bf7dd6b179826a2ac9234312e8d2b8599.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/tables/6732f274e529ba1f5a51554182b1d49bf7dd6b179826a2ac9234312e8d2b8599.jpg)

![678974d3aa3d90dfb89414d07323d01171801b2a844a915ce8c7e1718c374222.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/tables/678974d3aa3d90dfb89414d07323d01171801b2a844a915ce8c7e1718c374222.jpg)

![6c74b0123e071b3e06b1de3b5c5c85438b787f9333e9b6836aaca114ee355342.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/tables/6c74b0123e071b3e06b1de3b5c5c85438b787f9333e9b6836aaca114ee355342.jpg)

![6cb564effc8b443494e4f9412e76426935ab192abd3ef0ba18d32d70072d4537.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/tables/6cb564effc8b443494e4f9412e76426935ab192abd3ef0ba18d32d70072d4537.jpg)

![89276749868c7f6de3b02a0285e18792ffeadcd38ce893bb24abb959d2b095db.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/tables/89276749868c7f6de3b02a0285e18792ffeadcd38ce893bb24abb959d2b095db.jpg)

![94bcbdd92858378f0ddd149c85c0777cad61bd0783364bc6c27a8fea2ff77a44.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/tables/94bcbdd92858378f0ddd149c85c0777cad61bd0783364bc6c27a8fea2ff77a44.jpg)

![a3de88bc6f43a4517a15ef6fcd09aa6cddaf87e751fc366c232ccaae8006b850.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/tables/a3de88bc6f43a4517a15ef6fcd09aa6cddaf87e751fc366c232ccaae8006b850.jpg)

![acba747bc85da779237fbfa68323395ab57fad3d6d696827c10fcbf3a4537b34.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/tables/acba747bc85da779237fbfa68323395ab57fad3d6d696827c10fcbf3a4537b34.jpg)

![af57ea98016db590626dbe680972a856cc3dc25bf785cc5c2dd70ce1273cc185.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/tables/af57ea98016db590626dbe680972a856cc3dc25bf785cc5c2dd70ce1273cc185.jpg)

![b4c2d7cf1e03bfb42ca21dc2c477398302972d6fe120010563c9e5becaf049cb.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/tables/b4c2d7cf1e03bfb42ca21dc2c477398302972d6fe120010563c9e5becaf049cb.jpg)

![bd6e87bbec1f7d9c1a5e8e1d284b33ad89cd2c49b1731a0a7f8daf23ad839b9c.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/tables/bd6e87bbec1f7d9c1a5e8e1d284b33ad89cd2c49b1731a0a7f8daf23ad839b9c.jpg)

![cfea73b252c04b3eb61908704823a7b7ca4c02894877bf398729cf09d1c06071.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/tables/cfea73b252c04b3eb61908704823a7b7ca4c02894877bf398729cf09d1c06071.jpg)

![e5a493a5fb993281ab3296c668bee382fb81c65339112233d76c1a541a348635.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/tables/e5a493a5fb993281ab3296c668bee382fb81c65339112233d76c1a541a348635.jpg)

![f3319b636677e09793003b9d8abd49322e02268622d0e60331577f1c765a57f7.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/tables/f3319b636677e09793003b9d8abd49322e02268622d0e60331577f1c765a57f7.jpg)

![f63f9e78248493b78e957d28d19530ab1b7b39bda3fdc72529e6a7504a87ec85.jpg](../iclr_results/2269_KOR-Bench_ Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks/tables/f63f9e78248493b78e957d28d19530ab1b7b39bda3fdc72529e6a7504a87ec85.jpg)

## Conservative Contextual Bandits: Beyond Linear Representations


### Images

![08991b179b0cdc56cfc9a7cb4ecb06eb99400241e11fc77d6b556f1afdbaf6c0.jpg](../iclr_results/2270_Conservative Contextual Bandits_ Beyond Linear Representations/images/08991b179b0cdc56cfc9a7cb4ecb06eb99400241e11fc77d6b556f1afdbaf6c0.jpg)

![350f4c284a28f051b31e5c6f9957c880fa23f918f7acfe5f91863a94cae5d630.jpg](../iclr_results/2270_Conservative Contextual Bandits_ Beyond Linear Representations/images/350f4c284a28f051b31e5c6f9957c880fa23f918f7acfe5f91863a94cae5d630.jpg)

![510f5598b2567a57f736996c87d627dc93ee33aef993cb28d90826a2157fe949.jpg](../iclr_results/2270_Conservative Contextual Bandits_ Beyond Linear Representations/images/510f5598b2567a57f736996c87d627dc93ee33aef993cb28d90826a2157fe949.jpg)

![67a629ed9efe80ad12a8a8881a920295a9377ca065af46fe7683f27e7cb74f59.jpg](../iclr_results/2270_Conservative Contextual Bandits_ Beyond Linear Representations/images/67a629ed9efe80ad12a8a8881a920295a9377ca065af46fe7683f27e7cb74f59.jpg)

![d2f3bbf15f2f4978b14171fa25c1c7d1efa233961916f586024f8f5918c8764a.jpg](../iclr_results/2270_Conservative Contextual Bandits_ Beyond Linear Representations/images/d2f3bbf15f2f4978b14171fa25c1c7d1efa233961916f586024f8f5918c8764a.jpg)

## Is In-Context Learning Sufficient for Instruction Following in LLMs?


### Images

![0ed5de1d527bcbd889f0689716f9d60adeb1c40394db881f56bd314f742ad13a.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/images/0ed5de1d527bcbd889f0689716f9d60adeb1c40394db881f56bd314f742ad13a.jpg)

![1078706e8a8c1a38e1e6c4844281f77d58b9aec6c32a26ab7fe4e2d3599f8c37.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/images/1078706e8a8c1a38e1e6c4844281f77d58b9aec6c32a26ab7fe4e2d3599f8c37.jpg)

![2d9276d98574ec88f1f9234649d74d8042c51c9f679aaed9da7e142b193a8603.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/images/2d9276d98574ec88f1f9234649d74d8042c51c9f679aaed9da7e142b193a8603.jpg)

![36d6c5ed26057fdc28900b7c913692ba2c6ca1c26b8dad4de6cc6ed4f631a649.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/images/36d6c5ed26057fdc28900b7c913692ba2c6ca1c26b8dad4de6cc6ed4f631a649.jpg)

![4faffbd090124f7427d2509988fc398ad2458f125923d5c1e47ac9bb2fdb30a5.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/images/4faffbd090124f7427d2509988fc398ad2458f125923d5c1e47ac9bb2fdb30a5.jpg)

![575a3da3ff30c6104e8119cbbd9cf916d7e066da2eacde0b051503e25f37baf8.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/images/575a3da3ff30c6104e8119cbbd9cf916d7e066da2eacde0b051503e25f37baf8.jpg)

![58e5f27750a83e7c97c0b1a415735ae6642b7a499006b652d2698dece10662bf.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/images/58e5f27750a83e7c97c0b1a415735ae6642b7a499006b652d2698dece10662bf.jpg)

![6edd42cd4f58de3562fade140927ac9f8090d54f45ffad0413ede93a98e1685b.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/images/6edd42cd4f58de3562fade140927ac9f8090d54f45ffad0413ede93a98e1685b.jpg)

![9c1d1cbba18ff249c5f9aee7859dfc86582db74b709d63c010f9273d073b38ca.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/images/9c1d1cbba18ff249c5f9aee7859dfc86582db74b709d63c010f9273d073b38ca.jpg)

![b09cf8757c147f63f1b3bed1537779b83d63e2a16845c85efa8672e2603100da.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/images/b09cf8757c147f63f1b3bed1537779b83d63e2a16845c85efa8672e2603100da.jpg)

![c0e1a7b9ae13417187768aab39e4c8677bf942c86e997f6330cbad24b69deeb7.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/images/c0e1a7b9ae13417187768aab39e4c8677bf942c86e997f6330cbad24b69deeb7.jpg)

![d14fc147e1bcee6e1f8f408730eb712049abf063918b27edf8427863eee360c0.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/images/d14fc147e1bcee6e1f8f408730eb712049abf063918b27edf8427863eee360c0.jpg)

![f13588c4e4af0c0139b03259799b7575299c08e027ffd39f7daf1c91d3c56d5f.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/images/f13588c4e4af0c0139b03259799b7575299c08e027ffd39f7daf1c91d3c56d5f.jpg)

![fb80f72f5a1aa10adad66922a0c5d981c1319785d7fae81c4d4c06b82293c270.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/images/fb80f72f5a1aa10adad66922a0c5d981c1319785d7fae81c4d4c06b82293c270.jpg)

### Tables

![53ee4d5b84c15cd7b9bd8a80f744eba36df7674fdc0e7f485ff66eb657b3858e.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/tables/53ee4d5b84c15cd7b9bd8a80f744eba36df7674fdc0e7f485ff66eb657b3858e.jpg)

![5a6c5d65811178856608df4766518397020c88ed74bae287e5d9b24bab5e5756.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/tables/5a6c5d65811178856608df4766518397020c88ed74bae287e5d9b24bab5e5756.jpg)

![947667353b936a204ecee41d6ab605a18f1d013cabd763be5e0235d84c6caaa2.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/tables/947667353b936a204ecee41d6ab605a18f1d013cabd763be5e0235d84c6caaa2.jpg)

![99ac236d4ef066adf1a337e631b2c29123acaa6f31932c2756e0ca0575e2ed59.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/tables/99ac236d4ef066adf1a337e631b2c29123acaa6f31932c2756e0ca0575e2ed59.jpg)

![acf382977ace3d7a41bbb7d284fe02a9e416de72c482d71f044f0a7dfeccba60.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/tables/acf382977ace3d7a41bbb7d284fe02a9e416de72c482d71f044f0a7dfeccba60.jpg)

![c9363b5191fe39a97072c9a59112be601491477d84a9127a308ebe6f8a5cbd8c.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/tables/c9363b5191fe39a97072c9a59112be601491477d84a9127a308ebe6f8a5cbd8c.jpg)

![ce6867e42ac7384220662123b59a82f6f71b8e50e43baea1089ac5910d1e8327.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/tables/ce6867e42ac7384220662123b59a82f6f71b8e50e43baea1089ac5910d1e8327.jpg)

![ebb45117ea8b5cd22984a3352dfb5af0d608e47e68dac0297f50b7b77af12e44.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/tables/ebb45117ea8b5cd22984a3352dfb5af0d608e47e68dac0297f50b7b77af12e44.jpg)

![f0cdc552655f962dad0dc73dd53638e11b1bd40f4df7d0370d67925b64e26b50.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/tables/f0cdc552655f962dad0dc73dd53638e11b1bd40f4df7d0370d67925b64e26b50.jpg)

![f5b7200b848ed8b89e817a65ceb46c761ccd6d8938366b35e59b54f97adbb133.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/tables/f5b7200b848ed8b89e817a65ceb46c761ccd6d8938366b35e59b54f97adbb133.jpg)

![fd2db04ea27bbf2ab844f7018e9ad364ecc2887f66e21fa28131ba0ff919b843.jpg](../iclr_results/2271_Is In-Context Learning Sufficient for Instruction Following in LLMs_/tables/fd2db04ea27bbf2ab844f7018e9ad364ecc2887f66e21fa28131ba0ff919b843.jpg)

## VideoGrain: Modulating Space-Time Attention for Multi-Grained Video Editing


### Images

![039b848006ef7e145bdf26b1c8bc6a3e09fe79d998757f7622693911aec9f31f.jpg](../iclr_results/2272_VideoGrain_ Modulating Space-Time Attention for Multi-Grained Video Editing/images/039b848006ef7e145bdf26b1c8bc6a3e09fe79d998757f7622693911aec9f31f.jpg)

![0eb3570ed790911c574176b5b723fb1bbb35104fe36eff1bc378e49c6894c085.jpg](../iclr_results/2272_VideoGrain_ Modulating Space-Time Attention for Multi-Grained Video Editing/images/0eb3570ed790911c574176b5b723fb1bbb35104fe36eff1bc378e49c6894c085.jpg)

![186412a09f384835e3e1ecc31a4d489bc71a3276e32787ba44396aeaa722e51a.jpg](../iclr_results/2272_VideoGrain_ Modulating Space-Time Attention for Multi-Grained Video Editing/images/186412a09f384835e3e1ecc31a4d489bc71a3276e32787ba44396aeaa722e51a.jpg)

![291402adc738c8c08cd08a7adfeee50eb702d766f0a759417594f78cfe10d328.jpg](../iclr_results/2272_VideoGrain_ Modulating Space-Time Attention for Multi-Grained Video Editing/images/291402adc738c8c08cd08a7adfeee50eb702d766f0a759417594f78cfe10d328.jpg)

![2fd999257a2e85ac117a637fd120534e5b57ad045776f9d6a8224632948fe403.jpg](../iclr_results/2272_VideoGrain_ Modulating Space-Time Attention for Multi-Grained Video Editing/images/2fd999257a2e85ac117a637fd120534e5b57ad045776f9d6a8224632948fe403.jpg)

![334e96122fd7582d242472211d56d17fee9cce48c0dbef8e7a90c1caf6d4a819.jpg](../iclr_results/2272_VideoGrain_ Modulating Space-Time Attention for Multi-Grained Video Editing/images/334e96122fd7582d242472211d56d17fee9cce48c0dbef8e7a90c1caf6d4a819.jpg)

![39baa02ddb6ee26d1e5da45398f0ef472a21d9a97861f143588182b13851c525.jpg](../iclr_results/2272_VideoGrain_ Modulating Space-Time Attention for Multi-Grained Video Editing/images/39baa02ddb6ee26d1e5da45398f0ef472a21d9a97861f143588182b13851c525.jpg)

![3b3bc3b6b87c38adf6ce3c6d2945a8090537468fd0db6000a5172c7ac3af3c5f.jpg](../iclr_results/2272_VideoGrain_ Modulating Space-Time Attention for Multi-Grained Video Editing/images/3b3bc3b6b87c38adf6ce3c6d2945a8090537468fd0db6000a5172c7ac3af3c5f.jpg)

![5bd0c30a9b047f0d6ed0968ac76e1e29ed7bc91478dd1f18a180501b78ad730d.jpg](../iclr_results/2272_VideoGrain_ Modulating Space-Time Attention for Multi-Grained Video Editing/images/5bd0c30a9b047f0d6ed0968ac76e1e29ed7bc91478dd1f18a180501b78ad730d.jpg)

![712dddd506e96798d78750280bd02f3aa68dc76260d612741fbc34ffae6eb47a.jpg](../iclr_results/2272_VideoGrain_ Modulating Space-Time Attention for Multi-Grained Video Editing/images/712dddd506e96798d78750280bd02f3aa68dc76260d612741fbc34ffae6eb47a.jpg)

![8e6bdd99d0387bf73d77be609d85949e601a0117aefc9b14a248e9c8cbc63d2c.jpg](../iclr_results/2272_VideoGrain_ Modulating Space-Time Attention for Multi-Grained Video Editing/images/8e6bdd99d0387bf73d77be609d85949e601a0117aefc9b14a248e9c8cbc63d2c.jpg)

![a1416414f601655eec6bb659938e0f874f130fa4eebb6b378984ee94f2ee3972.jpg](../iclr_results/2272_VideoGrain_ Modulating Space-Time Attention for Multi-Grained Video Editing/images/a1416414f601655eec6bb659938e0f874f130fa4eebb6b378984ee94f2ee3972.jpg)

![b8784b5e67f15ed93edab294cefc0aa6b488644d6b7630983d32627bf17dc878.jpg](../iclr_results/2272_VideoGrain_ Modulating Space-Time Attention for Multi-Grained Video Editing/images/b8784b5e67f15ed93edab294cefc0aa6b488644d6b7630983d32627bf17dc878.jpg)

![b882b5f751d884c4d009eef91408aee121c506ff2d7d0af40ffb6fb5dd43e6d6.jpg](../iclr_results/2272_VideoGrain_ Modulating Space-Time Attention for Multi-Grained Video Editing/images/b882b5f751d884c4d009eef91408aee121c506ff2d7d0af40ffb6fb5dd43e6d6.jpg)

![d37b1600199b30897408bd0ded5592ec1e7eb6b649eb6496f7e1cf0a877c4c6e.jpg](../iclr_results/2272_VideoGrain_ Modulating Space-Time Attention for Multi-Grained Video Editing/images/d37b1600199b30897408bd0ded5592ec1e7eb6b649eb6496f7e1cf0a877c4c6e.jpg)

![f43007e8a90fc3ab183df6181f252b4e6f3dbfaa52ababbd00ac29de86659aaa.jpg](../iclr_results/2272_VideoGrain_ Modulating Space-Time Attention for Multi-Grained Video Editing/images/f43007e8a90fc3ab183df6181f252b4e6f3dbfaa52ababbd00ac29de86659aaa.jpg)

![f440011abbe7d63f0d9ff496288fcacd8a6f76d223f68039798918beb610469f.jpg](../iclr_results/2272_VideoGrain_ Modulating Space-Time Attention for Multi-Grained Video Editing/images/f440011abbe7d63f0d9ff496288fcacd8a6f76d223f68039798918beb610469f.jpg)

### Tables

![d4c8126b914f9404e8a571793207d9934f1ee66ed3403b345ae93169af3287a5.jpg](../iclr_results/2272_VideoGrain_ Modulating Space-Time Attention for Multi-Grained Video Editing/tables/d4c8126b914f9404e8a571793207d9934f1ee66ed3403b345ae93169af3287a5.jpg)

![df3ef0ed5c2c14cf7cce259f1e3f61af4cd77339be68b434df70d2f77130a5e6.jpg](../iclr_results/2272_VideoGrain_ Modulating Space-Time Attention for Multi-Grained Video Editing/tables/df3ef0ed5c2c14cf7cce259f1e3f61af4cd77339be68b434df70d2f77130a5e6.jpg)

![fdec012015005c07a17261a0d569a01550f58e50f3e0fa462baba284b3a65c55.jpg](../iclr_results/2272_VideoGrain_ Modulating Space-Time Attention for Multi-Grained Video Editing/tables/fdec012015005c07a17261a0d569a01550f58e50f3e0fa462baba284b3a65c55.jpg)

## 3D Vision-Language Gaussian Splatting


### Images

![30fa61ac96bab8eb45e5706c874ed7a579f5a470f8c04b3532e3485f4a40a20b.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/images/30fa61ac96bab8eb45e5706c874ed7a579f5a470f8c04b3532e3485f4a40a20b.jpg)

![4b5afde156e9d800dda6ef46d78c30e8305cee63819c03fea8b630860d155dbe.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/images/4b5afde156e9d800dda6ef46d78c30e8305cee63819c03fea8b630860d155dbe.jpg)

![4bdc12e508ee31fb7562ae47b77d9318afafaf9015d6b5e83c6af48e180a369f.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/images/4bdc12e508ee31fb7562ae47b77d9318afafaf9015d6b5e83c6af48e180a369f.jpg)

![5b313513a8a69ba9508c177ba35747b911871876cf13c1b49fd0e0a174db59f3.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/images/5b313513a8a69ba9508c177ba35747b911871876cf13c1b49fd0e0a174db59f3.jpg)

![7b2ddbf9a3b812d5620d5db90aa215b6451a1ea357f1bdaf21f9019cb8c177a9.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/images/7b2ddbf9a3b812d5620d5db90aa215b6451a1ea357f1bdaf21f9019cb8c177a9.jpg)

![910ec2a4fa1553633caf1237dd5ed0a70430ae9d1f0e7706ddc0a8f9dce29c08.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/images/910ec2a4fa1553633caf1237dd5ed0a70430ae9d1f0e7706ddc0a8f9dce29c08.jpg)

![a0636b1332cf7cd126f5979fb1bf9f156558b9e6c5f143454a65c902b2436870.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/images/a0636b1332cf7cd126f5979fb1bf9f156558b9e6c5f143454a65c902b2436870.jpg)

![a08e9fd6aa1dd5e9e3602a4c6da578626420f5ba7db4dea1b36862f36afffb14.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/images/a08e9fd6aa1dd5e9e3602a4c6da578626420f5ba7db4dea1b36862f36afffb14.jpg)

![a7b6feaafe7fdc29a3c900f896f6be265055a9d6ec2526db814dfaa509992a29.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/images/a7b6feaafe7fdc29a3c900f896f6be265055a9d6ec2526db814dfaa509992a29.jpg)

![c4c2355e6aa81ef2515705fbcb8895aaf405270930220c0af91e2178cfa37d49.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/images/c4c2355e6aa81ef2515705fbcb8895aaf405270930220c0af91e2178cfa37d49.jpg)

![c6cb37a95a1288d36ebf2ce1002535ff33cdea2ace9bd004340b157f411144f0.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/images/c6cb37a95a1288d36ebf2ce1002535ff33cdea2ace9bd004340b157f411144f0.jpg)

![cddd832ceb070f92c6425549317a583c8b4634e7051ad5bc11ec9bc414ebe2eb.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/images/cddd832ceb070f92c6425549317a583c8b4634e7051ad5bc11ec9bc414ebe2eb.jpg)

![e229c431a884c0c371618682e59ad215c13aa06199e9d27fa97ec32b8c0cb7c5.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/images/e229c431a884c0c371618682e59ad215c13aa06199e9d27fa97ec32b8c0cb7c5.jpg)

![e3944f42571f021f68a0b1a07e5256fc66e3a6ab897998b0f0aafcd2d042e6de.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/images/e3944f42571f021f68a0b1a07e5256fc66e3a6ab897998b0f0aafcd2d042e6de.jpg)

![e42ebcbf173d2b0e98a83641e55c26b086aaaeb69f4276d78ee0e5d16cc991da.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/images/e42ebcbf173d2b0e98a83641e55c26b086aaaeb69f4276d78ee0e5d16cc991da.jpg)

![fb249e3df9cfec5d78a725a50a607dfd6c318a2fc900c0cd35f8bc7400632b68.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/images/fb249e3df9cfec5d78a725a50a607dfd6c318a2fc900c0cd35f8bc7400632b68.jpg)

### Tables

![061658f0a0226ea52c2d888efe4dd31a936aeae195ae7ba894fac162b0575114.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/061658f0a0226ea52c2d888efe4dd31a936aeae195ae7ba894fac162b0575114.jpg)

![07c22fa1f70a20ae5106be76d55c82f9e994fa13374fc3decfbf103517ed5f44.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/07c22fa1f70a20ae5106be76d55c82f9e994fa13374fc3decfbf103517ed5f44.jpg)

![133ae6a2b20ec3ebee44c37d04b70372e18718b3248fab89e4f1eeecf9679440.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/133ae6a2b20ec3ebee44c37d04b70372e18718b3248fab89e4f1eeecf9679440.jpg)

![1351feba9839102960241120eb17fa56f27ef153bf72eaa995f2d09ffafbadf4.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/1351feba9839102960241120eb17fa56f27ef153bf72eaa995f2d09ffafbadf4.jpg)

![25f5eb1d615439974fdd094663230900e3515ff39a0b0cd475492b930cd6ac5a.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/25f5eb1d615439974fdd094663230900e3515ff39a0b0cd475492b930cd6ac5a.jpg)

![36061f422e8a3d7976c5be102295bef7cf7491e46da2e6cd8fbbe6d3f4d5f46e.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/36061f422e8a3d7976c5be102295bef7cf7491e46da2e6cd8fbbe6d3f4d5f46e.jpg)

![39d59e8de80c63b20fbf7ffde78fd22414297bb29dc676776a2627e627a8cdb7.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/39d59e8de80c63b20fbf7ffde78fd22414297bb29dc676776a2627e627a8cdb7.jpg)

![3bbffc08eae7d151372879a401dd58d5eecc0a7924bff70ae92fd73f5344dc37.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/3bbffc08eae7d151372879a401dd58d5eecc0a7924bff70ae92fd73f5344dc37.jpg)

![3d4e0e8d9176676c97f32249bff4dcc12740f000effc603f28be044d89cac2ad.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/3d4e0e8d9176676c97f32249bff4dcc12740f000effc603f28be044d89cac2ad.jpg)

![4d42b3af0488b0e1b9332a9f86bcd0e5f5d0c5595b09771f89dc8106138881d1.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/4d42b3af0488b0e1b9332a9f86bcd0e5f5d0c5595b09771f89dc8106138881d1.jpg)

![4fbe3a4db190d407294c5a7ee8b45dc4963e0a4054fa5023f4f13f8588207056.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/4fbe3a4db190d407294c5a7ee8b45dc4963e0a4054fa5023f4f13f8588207056.jpg)

![50c3a6e996ffba52ca40a842b4f3e74e483d770e4e6b761efd0ad4f2bfab9244.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/50c3a6e996ffba52ca40a842b4f3e74e483d770e4e6b761efd0ad4f2bfab9244.jpg)

![65164f64e52f54846e6ca7a3a89e8f168165eca3d20a59d1dd5ba647ae4fecbc.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/65164f64e52f54846e6ca7a3a89e8f168165eca3d20a59d1dd5ba647ae4fecbc.jpg)

![7893f22af3da1279d29d2173882f0838650103adedc6e2358e6a24d816f70dfd.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/7893f22af3da1279d29d2173882f0838650103adedc6e2358e6a24d816f70dfd.jpg)

![84db794d88692fdf4b6f0985489e2309ab1f84562e077903be9a7a9df3ec2af2.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/84db794d88692fdf4b6f0985489e2309ab1f84562e077903be9a7a9df3ec2af2.jpg)

![872a7569bf25a01676cb4114e666004d0d1b55b367a77ac460df681ba3fb6af1.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/872a7569bf25a01676cb4114e666004d0d1b55b367a77ac460df681ba3fb6af1.jpg)

![980373b7fefba5a294710f06a8b70bec7554046cd9eee51c09e19d499e8b29e1.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/980373b7fefba5a294710f06a8b70bec7554046cd9eee51c09e19d499e8b29e1.jpg)

![9ea58734bfa10d7b79190ffc18afe5abedbe7fecf3681e0669bf77531f2e9d95.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/9ea58734bfa10d7b79190ffc18afe5abedbe7fecf3681e0669bf77531f2e9d95.jpg)

![ae984f087401e280c6ccf3f83dd8f11cccbb98a2342c5993b9d2181c373d9e58.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/ae984f087401e280c6ccf3f83dd8f11cccbb98a2342c5993b9d2181c373d9e58.jpg)

![bb94a20d28ef6be120031f29e8328c0568aea36369d0c21c14660997b2c4cd4c.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/bb94a20d28ef6be120031f29e8328c0568aea36369d0c21c14660997b2c4cd4c.jpg)

![c2191266631e7fccc94b6d0a135e4362bae02b3deb68c650e9914e41be926652.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/c2191266631e7fccc94b6d0a135e4362bae02b3deb68c650e9914e41be926652.jpg)

![c44803fe780274f22db9ed71c8270c2008cd6a1349fe76b37ab18870fa973e77.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/c44803fe780274f22db9ed71c8270c2008cd6a1349fe76b37ab18870fa973e77.jpg)

![e60bc0e06f61f808bc48939ec43048f2925ffad7be01e44773dee3608282af7a.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/e60bc0e06f61f808bc48939ec43048f2925ffad7be01e44773dee3608282af7a.jpg)

![e7e0308c5e324ca65a0e8032bc26bd65cff50d70ce6dc77cd115628b2e2d0185.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/e7e0308c5e324ca65a0e8032bc26bd65cff50d70ce6dc77cd115628b2e2d0185.jpg)

![ec30dd716e21335819ab72da88af947cdfc0e496342be265a174b824d82c1af9.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/ec30dd716e21335819ab72da88af947cdfc0e496342be265a174b824d82c1af9.jpg)

![ef2aaa13174c0a71473b325d9bf9aa4938026f3c50d7bc1641cc65a371afb04d.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/ef2aaa13174c0a71473b325d9bf9aa4938026f3c50d7bc1641cc65a371afb04d.jpg)

![efbcd21a2671531630f60d5a840b9b53e82e0ba707a0096cb757190de17f7696.jpg](../iclr_results/2273_3D Vision-Language Gaussian Splatting/tables/efbcd21a2671531630f60d5a840b9b53e82e0ba707a0096cb757190de17f7696.jpg)

## Fat-to-Thin Policy Optimization: Offline Reinforcement Learning with Sparse Policies


### Images

![13c40553bafc427ae686a711aa0294b60f34f3a636e3caca956f48e312d03ee3.jpg](../iclr_results/2274_Fat-to-Thin Policy Optimization_ Offline Reinforcement Learning with Sparse Policies/images/13c40553bafc427ae686a711aa0294b60f34f3a636e3caca956f48e312d03ee3.jpg)

![147ebfbeb2433697722e5277cd061b04148df26103a11a9a7a608969301786f6.jpg](../iclr_results/2274_Fat-to-Thin Policy Optimization_ Offline Reinforcement Learning with Sparse Policies/images/147ebfbeb2433697722e5277cd061b04148df26103a11a9a7a608969301786f6.jpg)

![30785ee876f81de7cf6e01857f45debd3c0f380f71c33a5d610102d04bee729f.jpg](../iclr_results/2274_Fat-to-Thin Policy Optimization_ Offline Reinforcement Learning with Sparse Policies/images/30785ee876f81de7cf6e01857f45debd3c0f380f71c33a5d610102d04bee729f.jpg)

![3af58d9edbc3c3f58b9cac02ab050bc11009187c55acb27bf1ce1b389f57ae9b.jpg](../iclr_results/2274_Fat-to-Thin Policy Optimization_ Offline Reinforcement Learning with Sparse Policies/images/3af58d9edbc3c3f58b9cac02ab050bc11009187c55acb27bf1ce1b389f57ae9b.jpg)

![4200a8cacea7e6f5a43ccea1b7c08b7bf54a8ebabb03a57369608f90b67ea2d1.jpg](../iclr_results/2274_Fat-to-Thin Policy Optimization_ Offline Reinforcement Learning with Sparse Policies/images/4200a8cacea7e6f5a43ccea1b7c08b7bf54a8ebabb03a57369608f90b67ea2d1.jpg)

![5aedf183f213c0e2e9520e691ed60ba4245faea9bbc6bf82a8455c57fe699fae.jpg](../iclr_results/2274_Fat-to-Thin Policy Optimization_ Offline Reinforcement Learning with Sparse Policies/images/5aedf183f213c0e2e9520e691ed60ba4245faea9bbc6bf82a8455c57fe699fae.jpg)

![87e3893e3048aeb00e34f8028aea785c8a0403ca6cf1fdd8b989ddf891196599.jpg](../iclr_results/2274_Fat-to-Thin Policy Optimization_ Offline Reinforcement Learning with Sparse Policies/images/87e3893e3048aeb00e34f8028aea785c8a0403ca6cf1fdd8b989ddf891196599.jpg)

![8dac39591ec0a09f1354f30147fff0ea6a4d9ddc3a2a8d0b5c640897d7637651.jpg](../iclr_results/2274_Fat-to-Thin Policy Optimization_ Offline Reinforcement Learning with Sparse Policies/images/8dac39591ec0a09f1354f30147fff0ea6a4d9ddc3a2a8d0b5c640897d7637651.jpg)

![aea634337c2d4533cf2b0e136c43ef831fbddace275e84aeee318dc56f498cac.jpg](../iclr_results/2274_Fat-to-Thin Policy Optimization_ Offline Reinforcement Learning with Sparse Policies/images/aea634337c2d4533cf2b0e136c43ef831fbddace275e84aeee318dc56f498cac.jpg)

![b8598c4081dc4e444386dfa031c2258e798ffd61985385674e40cb8e2f751dbd.jpg](../iclr_results/2274_Fat-to-Thin Policy Optimization_ Offline Reinforcement Learning with Sparse Policies/images/b8598c4081dc4e444386dfa031c2258e798ffd61985385674e40cb8e2f751dbd.jpg)

![f55fcbc85e0109a48f0e7ef53ca4ea7019c7291c0292430ec5f4a965cbbbe8ad.jpg](../iclr_results/2274_Fat-to-Thin Policy Optimization_ Offline Reinforcement Learning with Sparse Policies/images/f55fcbc85e0109a48f0e7ef53ca4ea7019c7291c0292430ec5f4a965cbbbe8ad.jpg)

### Tables

![b63814b021412e3306c8f97e0b80d1e24c0d8f1ea9732ed62cea4b0e2400a0af.jpg](../iclr_results/2274_Fat-to-Thin Policy Optimization_ Offline Reinforcement Learning with Sparse Policies/tables/b63814b021412e3306c8f97e0b80d1e24c0d8f1ea9732ed62cea4b0e2400a0af.jpg)

![e2bbb9a4d4951b6c744557608f909da3c8dae699609366c5a0750b229509e6ab.jpg](../iclr_results/2274_Fat-to-Thin Policy Optimization_ Offline Reinforcement Learning with Sparse Policies/tables/e2bbb9a4d4951b6c744557608f909da3c8dae699609366c5a0750b229509e6ab.jpg)

![f50e353038e44ba4b697423710a87cc25a69494f0ec80bf60c546d0b7e040a1e.jpg](../iclr_results/2274_Fat-to-Thin Policy Optimization_ Offline Reinforcement Learning with Sparse Policies/tables/f50e353038e44ba4b697423710a87cc25a69494f0ec80bf60c546d0b7e040a1e.jpg)

![f5b8ca357ac4e352f548d6089d4b86c406ce787ccbbf652bb9bccfe65d5fbf4b.jpg](../iclr_results/2274_Fat-to-Thin Policy Optimization_ Offline Reinforcement Learning with Sparse Policies/tables/f5b8ca357ac4e352f548d6089d4b86c406ce787ccbbf652bb9bccfe65d5fbf4b.jpg)

## Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probability Ratio Test


### Images

![053cdc07e90c353025e5641ea7e3a3e9e0d731a6b7c9d60843586aabff737cab.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/images/053cdc07e90c353025e5641ea7e3a3e9e0d731a6b7c9d60843586aabff737cab.jpg)

![075ede6de20301d197098377fb5b1c7c2263987d7c96ee6c7df0cd92a9e20e02.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/images/075ede6de20301d197098377fb5b1c7c2263987d7c96ee6c7df0cd92a9e20e02.jpg)

![10c9f915eb73dc687ac1dc56f6e395a866be8751b54179314876c42139acd78e.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/images/10c9f915eb73dc687ac1dc56f6e395a866be8751b54179314876c42139acd78e.jpg)

![239c88e206e2b4818c38bc4967a789c193b4690ac15f17d40609dfb7d13d5db1.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/images/239c88e206e2b4818c38bc4967a789c193b4690ac15f17d40609dfb7d13d5db1.jpg)

![5a42686a747ec2b6678e7cf7f0b694aa28a8303802b6aa26ad4b9c046d072e6b.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/images/5a42686a747ec2b6678e7cf7f0b694aa28a8303802b6aa26ad4b9c046d072e6b.jpg)

![60810495928ed3bbc930cb74ad09e45661456fe91bcf8c41bb5ea63b5ebd2d6e.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/images/60810495928ed3bbc930cb74ad09e45661456fe91bcf8c41bb5ea63b5ebd2d6e.jpg)

![7f1c09c8f2bf4e7d299fcd55339b6b2742f4338f862e1031cc5d55df2a3f8434.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/images/7f1c09c8f2bf4e7d299fcd55339b6b2742f4338f862e1031cc5d55df2a3f8434.jpg)

![88518f3cb569a8f036c909fb5a5371cb83401937b21b663a2667b9f245e9bf3d.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/images/88518f3cb569a8f036c909fb5a5371cb83401937b21b663a2667b9f245e9bf3d.jpg)

![8c7a09888200b3f10d6ce0989f3655f669378974e6ff46d6d99f147bb8358763.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/images/8c7a09888200b3f10d6ce0989f3655f669378974e6ff46d6d99f147bb8358763.jpg)

![9a17d341a9d9f4b11e35cce2eb7d26e0059f35b894c1fa61ab0d16459d12cdc6.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/images/9a17d341a9d9f4b11e35cce2eb7d26e0059f35b894c1fa61ab0d16459d12cdc6.jpg)

![a0f4514cf645e5fbcea241a48b57ff3f72db6422e2f74a43e6b118d02e01e0bb.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/images/a0f4514cf645e5fbcea241a48b57ff3f72db6422e2f74a43e6b118d02e01e0bb.jpg)

![b047ad5173d97b98de1ff72ce5c17cc120a6576f785025f4ae47c8f1b92ded3c.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/images/b047ad5173d97b98de1ff72ce5c17cc120a6576f785025f4ae47c8f1b92ded3c.jpg)

![c71d264cb846bfffbf9cce528a0d58e2e4d5260875292dae035947873375fefe.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/images/c71d264cb846bfffbf9cce528a0d58e2e4d5260875292dae035947873375fefe.jpg)

![d04a650f141ae62343bedd66872d163dd2904b2f78a4dc5fc932182b6be91189.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/images/d04a650f141ae62343bedd66872d163dd2904b2f78a4dc5fc932182b6be91189.jpg)

![e80b9155678c3ef51e21afc0efa6abac86489129c3a195a2dd0050c43d9fa7ed.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/images/e80b9155678c3ef51e21afc0efa6abac86489129c3a195a2dd0050c43d9fa7ed.jpg)

![f4d1fd542a8cfc9bf2a83442770bd152317c489c819c3c377365a9444522ec64.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/images/f4d1fd542a8cfc9bf2a83442770bd152317c489c819c3c377365a9444522ec64.jpg)

![fdc562c6e0c9335ad3d235db512d07325ba94f196b79eafdcb51cb06ae8934e5.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/images/fdc562c6e0c9335ad3d235db512d07325ba94f196b79eafdcb51cb06ae8934e5.jpg)

### Tables

![07b2ab523d28d6001bb34b68c1f9cfbd236ba3935abad107684f672aa616cc82.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/07b2ab523d28d6001bb34b68c1f9cfbd236ba3935abad107684f672aa616cc82.jpg)

![1286a9c6f5c6be8f76143ffb50d84e4a21ab6e080e03fb1f75ed04afe27a4947.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/1286a9c6f5c6be8f76143ffb50d84e4a21ab6e080e03fb1f75ed04afe27a4947.jpg)

![1dbe9a1abb0fa2a02fa30f6c604a37b91eac3bf3164f633f0d20f3617b9ad72a.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/1dbe9a1abb0fa2a02fa30f6c604a37b91eac3bf3164f633f0d20f3617b9ad72a.jpg)

![1f96e668fa6091a33cfe5d0427bc3cb2a7bdd9e17321f61cfb63a32a1305f221.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/1f96e668fa6091a33cfe5d0427bc3cb2a7bdd9e17321f61cfb63a32a1305f221.jpg)

![23d2a03f00c8424e52b80cbd826eaf173599ee73183fe9fd3bb0fafd6df1dff1.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/23d2a03f00c8424e52b80cbd826eaf173599ee73183fe9fd3bb0fafd6df1dff1.jpg)

![28801f70c3532403661ccede2ffe58f4fe0ef355787a5453a095a377f6bb755a.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/28801f70c3532403661ccede2ffe58f4fe0ef355787a5453a095a377f6bb755a.jpg)

![2c49216e8680db5b59efa7e186999c5ba02ffe95d262f6dff0ae47f9e4af5192.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/2c49216e8680db5b59efa7e186999c5ba02ffe95d262f6dff0ae47f9e4af5192.jpg)

![42353a0e19528461a1bd3ed28a22eb6a6bfa1819d05b5bab40e2d9ea8a819677.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/42353a0e19528461a1bd3ed28a22eb6a6bfa1819d05b5bab40e2d9ea8a819677.jpg)

![4875119600970c4ca2b86849d4c97510355e29b073aeca5541992a29bcce0362.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/4875119600970c4ca2b86849d4c97510355e29b073aeca5541992a29bcce0362.jpg)

![5088db410e006bdb9befa44795e720dfb51638fbf9fec97df9c856d8aae7945b.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/5088db410e006bdb9befa44795e720dfb51638fbf9fec97df9c856d8aae7945b.jpg)

![5dbc157ea9e2ffdf12f69412066d6cbb035de89e247f4e2c11c7247d7615e505.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/5dbc157ea9e2ffdf12f69412066d6cbb035de89e247f4e2c11c7247d7615e505.jpg)

![5e140981edb9f64557c1a7011785b5248e0c80ab9e4610f53aa2be25d71d63bf.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/5e140981edb9f64557c1a7011785b5248e0c80ab9e4610f53aa2be25d71d63bf.jpg)

![7e142159e83c1b25b6c72ea3314d5dc1f75c3832d85a6bdf3e17173dcefebad4.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/7e142159e83c1b25b6c72ea3314d5dc1f75c3832d85a6bdf3e17173dcefebad4.jpg)

![87d109a8966acf719775e5e03ed0bf96946c7bf91b125eaeb0ec97d4b4214556.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/87d109a8966acf719775e5e03ed0bf96946c7bf91b125eaeb0ec97d4b4214556.jpg)

![88045809b40884335df29a3007171dcbc95ac6acacf5846178e0420d6498c05e.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/88045809b40884335df29a3007171dcbc95ac6acacf5846178e0420d6498c05e.jpg)

![937e5abd92f3fb976138aeb1d6660a7f795791903d74a69cc5b16138242e0f49.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/937e5abd92f3fb976138aeb1d6660a7f795791903d74a69cc5b16138242e0f49.jpg)

![a6056cced3ed0f1bb56c6beb7d4f6086d45187ff36b4b90fa506de2b0dcdf68b.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/a6056cced3ed0f1bb56c6beb7d4f6086d45187ff36b4b90fa506de2b0dcdf68b.jpg)

![ac2989c3732fca8a19c47a962d5ccedf888d58f24ba3809288ee40f91ffeefdc.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/ac2989c3732fca8a19c47a962d5ccedf888d58f24ba3809288ee40f91ffeefdc.jpg)

![b3895412f4f49f0202a39327b11082f38e12c1cae78abbc228e9c6d07676331e.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/b3895412f4f49f0202a39327b11082f38e12c1cae78abbc228e9c6d07676331e.jpg)

![c335d3f8ee8df88c262862271ebd4d446186b3638b9ab2fcc2648b7d954561da.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/c335d3f8ee8df88c262862271ebd4d446186b3638b9ab2fcc2648b7d954561da.jpg)

![ca6f3e03cd30230a84e0f0b2fd05356be42d3fad708aaf46ae289990bd031db7.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/ca6f3e03cd30230a84e0f0b2fd05356be42d3fad708aaf46ae289990bd031db7.jpg)

![d12869832618ae1ce4e893c98e492f7725526b9bbc7309baf8bf17cd0892e51d.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/d12869832618ae1ce4e893c98e492f7725526b9bbc7309baf8bf17cd0892e51d.jpg)

![de38e241bc6b23fa5a40ff2502c3bd12bf342576241cfe7a80820991b3958f28.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/de38e241bc6b23fa5a40ff2502c3bd12bf342576241cfe7a80820991b3958f28.jpg)

![e864867d156c6a730b90d67cdfb5b3e938d3b8d3d980dbc8c7083babfd7e1f71.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/e864867d156c6a730b90d67cdfb5b3e938d3b8d3d980dbc8c7083babfd7e1f71.jpg)

![f678769930f9fdd53941adddd361433f9f78b2ca22648f73d67e76d2aa9701a2.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/f678769930f9fdd53941adddd361433f9f78b2ca22648f73d67e76d2aa9701a2.jpg)

![f8dc82f2a58796962c766b9791eba72a9245cf06cd6b6fa87028cf27aee40997.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/f8dc82f2a58796962c766b9791eba72a9245cf06cd6b6fa87028cf27aee40997.jpg)

![f9cc1c4a03696ec9a41a9ae32134b3754be448e52fd3f31d8125c47fb0e6454d.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/f9cc1c4a03696ec9a41a9ae32134b3754be448e52fd3f31d8125c47fb0e6454d.jpg)

![fd10684c090e1804417b3197b15ce3f2cc4841dadb0a4c8480706bf679736b83.jpg](../iclr_results/2275_Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probabi/tables/fd10684c090e1804417b3197b15ce3f2cc4841dadb0a4c8480706bf679736b83.jpg)

## Value-Incentivized Preference Optimization: A Unified Approach to Online and Offline RLHF


### Images

![24046126532ab38663d71d3180cd8169ccdfed68286eb7ac23c4993731a57ab3.jpg](../iclr_results/2276_Value-Incentivized Preference Optimization_ A Unified Approach to Online and Offline RLHF/images/24046126532ab38663d71d3180cd8169ccdfed68286eb7ac23c4993731a57ab3.jpg)

![3eeacbcc86f390b28603d9f78126692ac695148b1b57c1e0af6adf4a59982a3a.jpg](../iclr_results/2276_Value-Incentivized Preference Optimization_ A Unified Approach to Online and Offline RLHF/images/3eeacbcc86f390b28603d9f78126692ac695148b1b57c1e0af6adf4a59982a3a.jpg)

![59b0d5772b388c2ee75da88ce31a896e8c538a60663f9234fc3b8861c97d4944.jpg](../iclr_results/2276_Value-Incentivized Preference Optimization_ A Unified Approach to Online and Offline RLHF/images/59b0d5772b388c2ee75da88ce31a896e8c538a60663f9234fc3b8861c97d4944.jpg)

![80ca25fa3c4b8ef7338b6bf5d71c0cbe5b7884eb6aa317723407df62dfaaea04.jpg](../iclr_results/2276_Value-Incentivized Preference Optimization_ A Unified Approach to Online and Offline RLHF/images/80ca25fa3c4b8ef7338b6bf5d71c0cbe5b7884eb6aa317723407df62dfaaea04.jpg)

![c7a0734f56c9bb9512d8c1d8b210741c429551da161cf07639de7e65ad07a5f3.jpg](../iclr_results/2276_Value-Incentivized Preference Optimization_ A Unified Approach to Online and Offline RLHF/images/c7a0734f56c9bb9512d8c1d8b210741c429551da161cf07639de7e65ad07a5f3.jpg)

![f718513fefc763b8b62a06440571689c76348666b872f31147908ba8cfba4f1a.jpg](../iclr_results/2276_Value-Incentivized Preference Optimization_ A Unified Approach to Online and Offline RLHF/images/f718513fefc763b8b62a06440571689c76348666b872f31147908ba8cfba4f1a.jpg)

### Tables

![a61e00122d1240fdc00a3b8d1e1cd44bea21bd47a056a2e62915f5454cb33963.jpg](../iclr_results/2276_Value-Incentivized Preference Optimization_ A Unified Approach to Online and Offline RLHF/tables/a61e00122d1240fdc00a3b8d1e1cd44bea21bd47a056a2e62915f5454cb33963.jpg)

## Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion


### Images

![1c158f640656a70f77753c4048a4b17db353beeaf2d1e36dba1ea05d2e50b795.jpg](../iclr_results/2277_Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion/images/1c158f640656a70f77753c4048a4b17db353beeaf2d1e36dba1ea05d2e50b795.jpg)

![3ed1b91b1d9132d1dc0de9ad9b84f0b34d87ebf8c8d18984d008a201a5e18874.jpg](../iclr_results/2277_Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion/images/3ed1b91b1d9132d1dc0de9ad9b84f0b34d87ebf8c8d18984d008a201a5e18874.jpg)

![5f2e146fcd5e35ba7376b33008181bee1f0f56c6994b8c724e2af321e9585333.jpg](../iclr_results/2277_Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion/images/5f2e146fcd5e35ba7376b33008181bee1f0f56c6994b8c724e2af321e9585333.jpg)

![69539a73bdfd79b8eb9fc1abc29dd2e40d568e3c8384bf6c0c0caf3b131d3303.jpg](../iclr_results/2277_Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion/images/69539a73bdfd79b8eb9fc1abc29dd2e40d568e3c8384bf6c0c0caf3b131d3303.jpg)

![726d4f625ae54fb0ce8ae5a4562b09a9852b2912e521a9821359784fff97d226.jpg](../iclr_results/2277_Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion/images/726d4f625ae54fb0ce8ae5a4562b09a9852b2912e521a9821359784fff97d226.jpg)

![cb4bd28d50823c4f08545e7780b654c52e23544672416f80ab0805f00b329842.jpg](../iclr_results/2277_Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion/images/cb4bd28d50823c4f08545e7780b654c52e23544672416f80ab0805f00b329842.jpg)

![fdd3a78ae6777018827859204ae77ebed7fa82195a40eeb4d8faaeacb7652784.jpg](../iclr_results/2277_Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion/images/fdd3a78ae6777018827859204ae77ebed7fa82195a40eeb4d8faaeacb7652784.jpg)

### Tables

![05284879202a99fe707ddf4849de53c544eeedd38b77bce85cc0988f4eeea8e5.jpg](../iclr_results/2277_Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion/tables/05284879202a99fe707ddf4849de53c544eeedd38b77bce85cc0988f4eeea8e5.jpg)

![1d0d97623dfd1a40f80efb800e5966692456fc1b5223cb6cf824fe58b835cad2.jpg](../iclr_results/2277_Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion/tables/1d0d97623dfd1a40f80efb800e5966692456fc1b5223cb6cf824fe58b835cad2.jpg)

![35a4ebb1e6893ef6644527453ad443c98608a2ba926e68b01531290461d0ec8a.jpg](../iclr_results/2277_Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion/tables/35a4ebb1e6893ef6644527453ad443c98608a2ba926e68b01531290461d0ec8a.jpg)

![4f141d8a74969af82ea06bca72514559916914b4c88ebb1cde60f4acbb1308e3.jpg](../iclr_results/2277_Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion/tables/4f141d8a74969af82ea06bca72514559916914b4c88ebb1cde60f4acbb1308e3.jpg)

![7910cb34e25df04410ced8c98b94516c5c39c35aad05527cdcc412860977f4d5.jpg](../iclr_results/2277_Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion/tables/7910cb34e25df04410ced8c98b94516c5c39c35aad05527cdcc412860977f4d5.jpg)

![7b86c692bc80a3f3f28e68999a583626f6e62ff387e14b8748374fe908ab0367.jpg](../iclr_results/2277_Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion/tables/7b86c692bc80a3f3f28e68999a583626f6e62ff387e14b8748374fe908ab0367.jpg)

![8709fe656c6b5fee6a08461ae6260f93def4bd816e6dc64e7b1673f4fbd6ddde.jpg](../iclr_results/2277_Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion/tables/8709fe656c6b5fee6a08461ae6260f93def4bd816e6dc64e7b1673f4fbd6ddde.jpg)

![9162e69f0e383337379313a5bd360168a7a950de0f3e679849837f85b048ec80.jpg](../iclr_results/2277_Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion/tables/9162e69f0e383337379313a5bd360168a7a950de0f3e679849837f85b048ec80.jpg)

![9c9bd5f16f616fe8ce75c7dd51266ae584fc989e41458f214fa00ca827b71b51.jpg](../iclr_results/2277_Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion/tables/9c9bd5f16f616fe8ce75c7dd51266ae584fc989e41458f214fa00ca827b71b51.jpg)

![aa9e231d3bb246526a455d50f4e0d7ae6ade04f47e3213794c4dd7046f5f7b04.jpg](../iclr_results/2277_Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion/tables/aa9e231d3bb246526a455d50f4e0d7ae6ade04f47e3213794c4dd7046f5f7b04.jpg)

![cdf564fe2993ef6def07f1998043ae29ead23561147b72a97ce630d66bfdd021.jpg](../iclr_results/2277_Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion/tables/cdf564fe2993ef6def07f1998043ae29ead23561147b72a97ce630d66bfdd021.jpg)

## Language Models Are Implicitly Continuous


### Images

![006a01cdc5a1c0b0e1b46ad53fabf48ab570152111426f38ab74e019ba9ee48d.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/006a01cdc5a1c0b0e1b46ad53fabf48ab570152111426f38ab74e019ba9ee48d.jpg)

![00fc6e5379181117edd685142a5dc2848a2ff959fbbdf1dd91619cc0a77ca558.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/00fc6e5379181117edd685142a5dc2848a2ff959fbbdf1dd91619cc0a77ca558.jpg)

![0bb561c6653f2af0c8d7a61b1ec9c77c4f5a72a235cd64fb587235747b0dd840.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/0bb561c6653f2af0c8d7a61b1ec9c77c4f5a72a235cd64fb587235747b0dd840.jpg)

![1a1fabe1974452725b5aa03708eab068a385378cd6c096013b8096f88a5c7d8f.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/1a1fabe1974452725b5aa03708eab068a385378cd6c096013b8096f88a5c7d8f.jpg)

![1aa5f6a481bdf45d7f78b0cfa3550e73995a720a0e0846169b6831894f535e90.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/1aa5f6a481bdf45d7f78b0cfa3550e73995a720a0e0846169b6831894f535e90.jpg)

![2fd84579f139984a7e2f4ccc5a7e0a3d4e7c070c02ea5f21e39e024c95ef4386.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/2fd84579f139984a7e2f4ccc5a7e0a3d4e7c070c02ea5f21e39e024c95ef4386.jpg)

![330ac03ce00c1062ec1408531268229e5fad3a730281e2917d0c60cedf3c29d1.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/330ac03ce00c1062ec1408531268229e5fad3a730281e2917d0c60cedf3c29d1.jpg)

![3405ee81da0de3e555253a9effd9cbf2c9b88d9722c19fcace1e4fa3a5f28ab2.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/3405ee81da0de3e555253a9effd9cbf2c9b88d9722c19fcace1e4fa3a5f28ab2.jpg)

![34964b6feb5836824c1d72b9c14cb98b764957e985a115b61b147fe7b0a9e217.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/34964b6feb5836824c1d72b9c14cb98b764957e985a115b61b147fe7b0a9e217.jpg)

![354f9e3d33e975dac9725c4faa8a7d36132b1506f3d7d1fd9044aa8665a94d2b.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/354f9e3d33e975dac9725c4faa8a7d36132b1506f3d7d1fd9044aa8665a94d2b.jpg)

![445b5eef1ad1208d2fce02426e255bff6a0a3b65219f1a39c24637edeba40c8e.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/445b5eef1ad1208d2fce02426e255bff6a0a3b65219f1a39c24637edeba40c8e.jpg)

![4766f52a394547ce27dbe990061a21f0c2cf347d39778448666a5e46d00da2eb.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/4766f52a394547ce27dbe990061a21f0c2cf347d39778448666a5e46d00da2eb.jpg)

![47df90d03f49e1206641b42f01dbef68703450f05b45d875a1ce45f83f3df0c5.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/47df90d03f49e1206641b42f01dbef68703450f05b45d875a1ce45f83f3df0c5.jpg)

![4a10d85242c568cbb207fbb61c74ef471ae16ab7e2a549ed5f4cf6a07463b1ff.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/4a10d85242c568cbb207fbb61c74ef471ae16ab7e2a549ed5f4cf6a07463b1ff.jpg)

![51c5a00259976b4ca315b803d7ef2fd0bc0de2c9fcebd51598331bbc48a87d4e.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/51c5a00259976b4ca315b803d7ef2fd0bc0de2c9fcebd51598331bbc48a87d4e.jpg)

![527d313164264ad0085627fd3332c12cfd529b53f0830fbbbee47bc26ff6c08e.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/527d313164264ad0085627fd3332c12cfd529b53f0830fbbbee47bc26ff6c08e.jpg)

![536b9103bd851c0c3d087eb1155779e2e950bd0dd9d9b12a8a951ceeb56bffef.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/536b9103bd851c0c3d087eb1155779e2e950bd0dd9d9b12a8a951ceeb56bffef.jpg)

![578bd8bb07605226986da6c44e5f960c4a7b823be9fa32fb41f34c930b1e4d09.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/578bd8bb07605226986da6c44e5f960c4a7b823be9fa32fb41f34c930b1e4d09.jpg)

![61c4537ea0db3265bae05dc55b5cb20016a4f1a082e0113a8d7a25ed372fa142.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/61c4537ea0db3265bae05dc55b5cb20016a4f1a082e0113a8d7a25ed372fa142.jpg)

![64bdc615ce424ad630da9283cb8a8f2d90dd159d2b7329a33cf8a190a7e27a39.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/64bdc615ce424ad630da9283cb8a8f2d90dd159d2b7329a33cf8a190a7e27a39.jpg)

![6555a051b709e468dde5697f5d17d91b8c1bcd1d1b477af6c78c071ffda7ff65.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/6555a051b709e468dde5697f5d17d91b8c1bcd1d1b477af6c78c071ffda7ff65.jpg)

![66fc2ee23c3b589fa9e30ef6a67e75f5c09e80948b009b76076fc5c968a61707.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/66fc2ee23c3b589fa9e30ef6a67e75f5c09e80948b009b76076fc5c968a61707.jpg)

![674fc0802609186722974da21efcadd96048b1eb2bdfd61cff4897b0a999063a.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/674fc0802609186722974da21efcadd96048b1eb2bdfd61cff4897b0a999063a.jpg)

![6836181f25e5d7033965b1b38513d97f531c5d48b7e047615399cc3aa76530a7.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/6836181f25e5d7033965b1b38513d97f531c5d48b7e047615399cc3aa76530a7.jpg)

![6836d1645b05b807cb053f8a4734e92cb34d1ea36e1bdef3dc15d82ec82576e7.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/6836d1645b05b807cb053f8a4734e92cb34d1ea36e1bdef3dc15d82ec82576e7.jpg)

![6bc99c1880b19f27d41b582a04281ccc5dde4437fdb47b7c2e44611651fe54a4.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/6bc99c1880b19f27d41b582a04281ccc5dde4437fdb47b7c2e44611651fe54a4.jpg)

![6ffd2b0a93a0ffc3dcb22aaa944973fb98a403acbfcc01c8942f297a48b14a57.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/6ffd2b0a93a0ffc3dcb22aaa944973fb98a403acbfcc01c8942f297a48b14a57.jpg)

![70e88e1e7593c7fccc33fa3ce6b7502047e05ed4c4f602e92807f5f9e8480aa5.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/70e88e1e7593c7fccc33fa3ce6b7502047e05ed4c4f602e92807f5f9e8480aa5.jpg)

![7a8e52c7319333b13900333b0724b7b0b7c2b9387df415c5f17be410f536696a.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/7a8e52c7319333b13900333b0724b7b0b7c2b9387df415c5f17be410f536696a.jpg)

![7bc4197a5304fc8e9e6bfe385bfe7a84788e7d84e4ba7c1f0486b90b4e68f794.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/7bc4197a5304fc8e9e6bfe385bfe7a84788e7d84e4ba7c1f0486b90b4e68f794.jpg)

![7d2fb5c4d712c26faf9c39b6abf5c42853e4ae6e99e7c3053654adea215e0718.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/7d2fb5c4d712c26faf9c39b6abf5c42853e4ae6e99e7c3053654adea215e0718.jpg)

![9a6df4f5d0eead1efd62c2a1250c6ad8ca231d47c3c66b9f97d522212f135ce7.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/9a6df4f5d0eead1efd62c2a1250c6ad8ca231d47c3c66b9f97d522212f135ce7.jpg)

![9dfed0674d5923e02318db92a317b76a83131de9922432d353ea7e7ee131cc59.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/9dfed0674d5923e02318db92a317b76a83131de9922432d353ea7e7ee131cc59.jpg)

![9e1e1db0051d0949abf51d78cfd817787226f75f65f185d3a572787d048f0470.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/9e1e1db0051d0949abf51d78cfd817787226f75f65f185d3a572787d048f0470.jpg)

![9f04d41ee8f94c9f93f0add4ddf31f44180346c6a03bf09d5d2a2053ede788db.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/9f04d41ee8f94c9f93f0add4ddf31f44180346c6a03bf09d5d2a2053ede788db.jpg)

![a20ed287a6fd384b145765aa339e61076d5d438385d3b2bdcf5041dad983c7be.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/a20ed287a6fd384b145765aa339e61076d5d438385d3b2bdcf5041dad983c7be.jpg)

![b0013a12603ae1940533282419b288220b6c115f03cd84b150cc1320ac9cd32d.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/b0013a12603ae1940533282419b288220b6c115f03cd84b150cc1320ac9cd32d.jpg)

![b6c3f685f0e634cabefaa039c9c59744fad718e3151e4d32ee0e822bb0d504a7.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/b6c3f685f0e634cabefaa039c9c59744fad718e3151e4d32ee0e822bb0d504a7.jpg)

![b9eedb21839a5ea563c83a956af3c0902887aa9b2b91747018774be62fd20a99.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/b9eedb21839a5ea563c83a956af3c0902887aa9b2b91747018774be62fd20a99.jpg)

![bf0bcd44b99b4a6d4be40e2695183d1442aa9348f57ed99daf841ab7da4fce7c.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/bf0bcd44b99b4a6d4be40e2695183d1442aa9348f57ed99daf841ab7da4fce7c.jpg)

![c197c55934022882ee34a6954ee9506d1b02be03bb4e6c40111f1a2436212f53.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/c197c55934022882ee34a6954ee9506d1b02be03bb4e6c40111f1a2436212f53.jpg)

![cfc29064bd074b172ebaf5e3907d5389bd0d89d70e3dfbda1336ebab22bf0f10.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/cfc29064bd074b172ebaf5e3907d5389bd0d89d70e3dfbda1336ebab22bf0f10.jpg)

![d36e1af49e36a6dcbcb993b32d63c478ef2ea25c9a032d87b09ddd8369ff357f.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/d36e1af49e36a6dcbcb993b32d63c478ef2ea25c9a032d87b09ddd8369ff357f.jpg)

![e023e704b932adc775b13737265bc77421183219de276dabea0f4dd9d64adbc8.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/e023e704b932adc775b13737265bc77421183219de276dabea0f4dd9d64adbc8.jpg)

![e03b3d65982c0f2be4a2c790f9b00be5ee51842a1f2076b80e2259c79081dd2b.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/e03b3d65982c0f2be4a2c790f9b00be5ee51842a1f2076b80e2259c79081dd2b.jpg)

![e0766ba74dce3c4516521e4cbf579089833f00d9d81f1aa6af8f5d42117306f1.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/e0766ba74dce3c4516521e4cbf579089833f00d9d81f1aa6af8f5d42117306f1.jpg)

![e2e20dbad89a6314b3c2356b98582c76c86084c437cb7994174049c5f8e36f50.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/e2e20dbad89a6314b3c2356b98582c76c86084c437cb7994174049c5f8e36f50.jpg)

![e82200e4afdc8c1a0b5c33a9dee11e213e6f270e7813affa6275bb0df15a0216.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/e82200e4afdc8c1a0b5c33a9dee11e213e6f270e7813affa6275bb0df15a0216.jpg)

![ee18cc5f2114f6b1fdc971a1a12502fe28c30981d3f35e0d6ae7cb42c8e72982.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/ee18cc5f2114f6b1fdc971a1a12502fe28c30981d3f35e0d6ae7cb42c8e72982.jpg)

![f54e1ab2d2af28d184aab24e0d294f0271a77d8c39db2e25b34fb561bad228ad.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/f54e1ab2d2af28d184aab24e0d294f0271a77d8c39db2e25b34fb561bad228ad.jpg)

![f9a59ca51d9aa6a81861af0ece54e2e72d164f3f254847ed534f0873dafd4e88.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/f9a59ca51d9aa6a81861af0ece54e2e72d164f3f254847ed534f0873dafd4e88.jpg)

![fb1fbdc40098896debb016ad968d2c2d1a82edc18a1c5b2f1b20a9667137bb42.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/images/fb1fbdc40098896debb016ad968d2c2d1a82edc18a1c5b2f1b20a9667137bb42.jpg)

### Tables

![0902e6a059e89239568fbe22e903345a77eea48053adf77074a212f2918be0f9.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/tables/0902e6a059e89239568fbe22e903345a77eea48053adf77074a212f2918be0f9.jpg)

![35ac8ce3810be70581876c05b6e0ec836e9a1df30177b219afe2e29ed8cf8edf.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/tables/35ac8ce3810be70581876c05b6e0ec836e9a1df30177b219afe2e29ed8cf8edf.jpg)

![3b6f97c4ab2b3ce8fd1683ab1058f52b5d8e64adac4f12399dfb22191b6f9145.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/tables/3b6f97c4ab2b3ce8fd1683ab1058f52b5d8e64adac4f12399dfb22191b6f9145.jpg)

![47f3f243490b8ff62aea0d05a0da7ca4578cd2559aaa3da3b8b16b526510439b.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/tables/47f3f243490b8ff62aea0d05a0da7ca4578cd2559aaa3da3b8b16b526510439b.jpg)

![63f383d907f01b114f4d104a7435405d187ec5e6574c198c64c030f43b81eec5.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/tables/63f383d907f01b114f4d104a7435405d187ec5e6574c198c64c030f43b81eec5.jpg)

![701acdc765d0078a7fb5e12c31a37f76af7f1a9e8389550c145942f2872de142.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/tables/701acdc765d0078a7fb5e12c31a37f76af7f1a9e8389550c145942f2872de142.jpg)

![7206d390228b924ea72a52d2ded77ea343e288180228831ce65f7b3ab6c65abe.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/tables/7206d390228b924ea72a52d2ded77ea343e288180228831ce65f7b3ab6c65abe.jpg)

![81316dab12b50035420d5a354b3464e398423883f039de6314f71192b3bd6585.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/tables/81316dab12b50035420d5a354b3464e398423883f039de6314f71192b3bd6585.jpg)

![e75e917ac149245df14ff8807b03d6b9bf3295614c2e2febb247a1a765528901.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/tables/e75e917ac149245df14ff8807b03d6b9bf3295614c2e2febb247a1a765528901.jpg)

![faa4ccaeb410b0e6b7a26e93769d180001705bad430d59e0832a806a738293de.jpg](../iclr_results/2278_Language Models Are Implicitly Continuous/tables/faa4ccaeb410b0e6b7a26e93769d180001705bad430d59e0832a806a738293de.jpg)

## Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias


### Images

![027934f3981209ac32c0d523941d189b0ecd9e9d4b0e2e177f783a1a7de802f9.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/027934f3981209ac32c0d523941d189b0ecd9e9d4b0e2e177f783a1a7de802f9.jpg)

![08a27b858560bf05a83e19c7b26eab50b7a066f20e0cac6db2d02733bd2f359d.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/08a27b858560bf05a83e19c7b26eab50b7a066f20e0cac6db2d02733bd2f359d.jpg)

![0c731ebcc37f99739df194abc3cab3eb1f1d3d762ddace002d1cdce749979276.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/0c731ebcc37f99739df194abc3cab3eb1f1d3d762ddace002d1cdce749979276.jpg)

![0ea04b7e0a33a8825e41ebc1847a33c7b3c4b344c50fef4577e6a578ed718913.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/0ea04b7e0a33a8825e41ebc1847a33c7b3c4b344c50fef4577e6a578ed718913.jpg)

![0ff804bb480ba7a0e52743c96f6f380997c72e30f76adc611128c7da02d59b8f.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/0ff804bb480ba7a0e52743c96f6f380997c72e30f76adc611128c7da02d59b8f.jpg)

![1f5e8c7a9257402f1b00441dc91c55cc11f4c4993b3b3b8f4f96290f3d93b014.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/1f5e8c7a9257402f1b00441dc91c55cc11f4c4993b3b3b8f4f96290f3d93b014.jpg)

![2297369e1e144b4b77779b3c3f8ff15b05aa6adc607b542683f269ae01802114.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/2297369e1e144b4b77779b3c3f8ff15b05aa6adc607b542683f269ae01802114.jpg)

![254616f2fa6be4cbcaa10e01560096ae3833a1a4a1b54888c6c7a7cc8101e4cd.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/254616f2fa6be4cbcaa10e01560096ae3833a1a4a1b54888c6c7a7cc8101e4cd.jpg)

![29cf0561435952086e0c88c29872c7b4d9b61abc680288ffe24856d8c42b9147.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/29cf0561435952086e0c88c29872c7b4d9b61abc680288ffe24856d8c42b9147.jpg)

![375474b049dac242cec03028928bf3f4fff73ff4af61aec20e8c68223b27b5ec.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/375474b049dac242cec03028928bf3f4fff73ff4af61aec20e8c68223b27b5ec.jpg)

![37e0f8d064d40ebeed037ac764b3af266291b68ca4103d18bbe1ff96c83b0602.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/37e0f8d064d40ebeed037ac764b3af266291b68ca4103d18bbe1ff96c83b0602.jpg)

![508f6580c21b114117963d8f3b4e1e7eb01a369ff156bb64f92fabef57e20360.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/508f6580c21b114117963d8f3b4e1e7eb01a369ff156bb64f92fabef57e20360.jpg)

![53334c1bd9d7bc658af716fe641980d94d82b63d54819761ee6c4d783a1a7180.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/53334c1bd9d7bc658af716fe641980d94d82b63d54819761ee6c4d783a1a7180.jpg)

![67845faa9b501cf2eda1ef90b906624862005cc9e79d287c4a2f11e36af1a342.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/67845faa9b501cf2eda1ef90b906624862005cc9e79d287c4a2f11e36af1a342.jpg)

![6a810f54f813eaf93f50d6ddddecc8c8fcd886bdc726156ca6a0449e7e1c65ba.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/6a810f54f813eaf93f50d6ddddecc8c8fcd886bdc726156ca6a0449e7e1c65ba.jpg)

![710b0d61b6078f2c4e05f61223d91be5122a3e312b7e72cca090062122e93e93.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/710b0d61b6078f2c4e05f61223d91be5122a3e312b7e72cca090062122e93e93.jpg)

![8281b35b52ac9fb82475f3e3a42622d47f3b5512fa716b496ceb424b9104b05c.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/8281b35b52ac9fb82475f3e3a42622d47f3b5512fa716b496ceb424b9104b05c.jpg)

![8c62965c1c8b5d2643b05bc2c51fe22428b7d75291bb0bed845894ed30c78a8b.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/8c62965c1c8b5d2643b05bc2c51fe22428b7d75291bb0bed845894ed30c78a8b.jpg)

![9deecc4af7b5742f6398b3925d04c869970acb948570ffdb82fabd562c773678.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/9deecc4af7b5742f6398b3925d04c869970acb948570ffdb82fabd562c773678.jpg)

![a1d561b40ebbf622f25397745e10416ff4730e48242c8b64a5d7748c6e464edd.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/a1d561b40ebbf622f25397745e10416ff4730e48242c8b64a5d7748c6e464edd.jpg)

![a82c96fe789b47c1a4857c58da421c2e6c27325d2aabf54389cff90de7453312.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/a82c96fe789b47c1a4857c58da421c2e6c27325d2aabf54389cff90de7453312.jpg)

![c0147b7bc956bec72b2f69504fdcfdbcf4b0c006dd644b087fa1b911c92923be.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/c0147b7bc956bec72b2f69504fdcfdbcf4b0c006dd644b087fa1b911c92923be.jpg)

![d0829e2a8c2fe4db8229c4e0f1132f9eb731643ecc46ab239d85ef8b1f6e6540.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/d0829e2a8c2fe4db8229c4e0f1132f9eb731643ecc46ab239d85ef8b1f6e6540.jpg)

![d1189a98c6058418d6c58f537153af168d9bc538070e2e297b21a54a477ce52d.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/d1189a98c6058418d6c58f537153af168d9bc538070e2e297b21a54a477ce52d.jpg)

![e13fd51c25319d802cb1317b07794d3dd8f5871546dca95007ea59ed2245973f.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/e13fd51c25319d802cb1317b07794d3dd8f5871546dca95007ea59ed2245973f.jpg)

![e17061add1f1fc6c5dadeb6f5afc4ca3a118c7f1e6d80a015176565968211154.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/e17061add1f1fc6c5dadeb6f5afc4ca3a118c7f1e6d80a015176565968211154.jpg)

![e33a631ef183d94b053cda876779abf88b868e8a591cc07df289b5833f074726.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/e33a631ef183d94b053cda876779abf88b868e8a591cc07df289b5833f074726.jpg)

![e674d9244a9779518c733be7b3444fdf15991d3cdb3102ca3cdf56c1770c915b.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/e674d9244a9779518c733be7b3444fdf15991d3cdb3102ca3cdf56c1770c915b.jpg)

![f2715035b32952674ed6edc8b836dd0e174a155acbc29f4f2c9a5cc0cc50a8fa.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/images/f2715035b32952674ed6edc8b836dd0e174a155acbc29f4f2c9a5cc0cc50a8fa.jpg)

### Tables

![5f8c57a43eb221ebbe8a61ed9c01d251a8f7c7bffb892e1bf2e85c9f75c3d00f.jpg](../iclr_results/2279_Towards Understanding Text Hallucination of Diffusion Models via Local Generation Bias/tables/5f8c57a43eb221ebbe8a61ed9c01d251a8f7c7bffb892e1bf2e85c9f75c3d00f.jpg)

## High-Dimensional Bayesian Optimisation with Gaussian Process Prior Variational Autoencoders


### Images

![01c808b5d55fd4c460268091ba9eda395ca0ce0e9a4095d3ac760fb2cdb8384e.jpg](../iclr_results/2280_High-Dimensional Bayesian Optimisation with Gaussian Process Prior Variational Autoencoders/images/01c808b5d55fd4c460268091ba9eda395ca0ce0e9a4095d3ac760fb2cdb8384e.jpg)

![052dcd2da12c544d2a2fdd900434c6df038634911a5c62fe3885e399aa48e999.jpg](../iclr_results/2280_High-Dimensional Bayesian Optimisation with Gaussian Process Prior Variational Autoencoders/images/052dcd2da12c544d2a2fdd900434c6df038634911a5c62fe3885e399aa48e999.jpg)

![1c231ef8844face0bf6a872f9fcb432b7e8c610ddef8b64a24af2809c76a4cbf.jpg](../iclr_results/2280_High-Dimensional Bayesian Optimisation with Gaussian Process Prior Variational Autoencoders/images/1c231ef8844face0bf6a872f9fcb432b7e8c610ddef8b64a24af2809c76a4cbf.jpg)

![280d129d1a637d582511537290e787b0ee6cf7281670fedf7e92b5ebfd73a557.jpg](../iclr_results/2280_High-Dimensional Bayesian Optimisation with Gaussian Process Prior Variational Autoencoders/images/280d129d1a637d582511537290e787b0ee6cf7281670fedf7e92b5ebfd73a557.jpg)

![29aeeff2d95e60c9707b075d85c3843ad45b3d79cd2099bb9dde0da461b4caee.jpg](../iclr_results/2280_High-Dimensional Bayesian Optimisation with Gaussian Process Prior Variational Autoencoders/images/29aeeff2d95e60c9707b075d85c3843ad45b3d79cd2099bb9dde0da461b4caee.jpg)

![3498fc486e369ac1fca9bbe83d8adda46fa098008366c99bf59433566c481755.jpg](../iclr_results/2280_High-Dimensional Bayesian Optimisation with Gaussian Process Prior Variational Autoencoders/images/3498fc486e369ac1fca9bbe83d8adda46fa098008366c99bf59433566c481755.jpg)

![4a5ef2d35a3a1e8012337732e468debdf38b811a4480cc49fdbd3adfc517e736.jpg](../iclr_results/2280_High-Dimensional Bayesian Optimisation with Gaussian Process Prior Variational Autoencoders/images/4a5ef2d35a3a1e8012337732e468debdf38b811a4480cc49fdbd3adfc517e736.jpg)

![790014a0e51f5861a5d1777b7e3be1abb3bf36e6f6baecc19847f60f3523466e.jpg](../iclr_results/2280_High-Dimensional Bayesian Optimisation with Gaussian Process Prior Variational Autoencoders/images/790014a0e51f5861a5d1777b7e3be1abb3bf36e6f6baecc19847f60f3523466e.jpg)

![8c813a9b745f1e4b555493157d9edd0ead0d441ba2c378fa15618dc7bc317386.jpg](../iclr_results/2280_High-Dimensional Bayesian Optimisation with Gaussian Process Prior Variational Autoencoders/images/8c813a9b745f1e4b555493157d9edd0ead0d441ba2c378fa15618dc7bc317386.jpg)

![abe2d379ae42e31569c88ea680c3b46acb228105007730845532bf917eeb670e.jpg](../iclr_results/2280_High-Dimensional Bayesian Optimisation with Gaussian Process Prior Variational Autoencoders/images/abe2d379ae42e31569c88ea680c3b46acb228105007730845532bf917eeb670e.jpg)

![b8548f956ae08f0c78404385d925753cf44f63fd481979bafe6aacc8b4bea650.jpg](../iclr_results/2280_High-Dimensional Bayesian Optimisation with Gaussian Process Prior Variational Autoencoders/images/b8548f956ae08f0c78404385d925753cf44f63fd481979bafe6aacc8b4bea650.jpg)

![ba108dfe4805ef8bdfbb13ec11bfcaeb25c2afacbaf4510136075c8b81b893fa.jpg](../iclr_results/2280_High-Dimensional Bayesian Optimisation with Gaussian Process Prior Variational Autoencoders/images/ba108dfe4805ef8bdfbb13ec11bfcaeb25c2afacbaf4510136075c8b81b893fa.jpg)

![d637cc24ebfa0962f466cb9a0db0f89fcedf1284cda6e3f567897ac85f77e26a.jpg](../iclr_results/2280_High-Dimensional Bayesian Optimisation with Gaussian Process Prior Variational Autoencoders/images/d637cc24ebfa0962f466cb9a0db0f89fcedf1284cda6e3f567897ac85f77e26a.jpg)

![f84e3a24220529043820ac9094d6201420cc4fe01dc9a1a9f08dae4174992a8d.jpg](../iclr_results/2280_High-Dimensional Bayesian Optimisation with Gaussian Process Prior Variational Autoencoders/images/f84e3a24220529043820ac9094d6201420cc4fe01dc9a1a9f08dae4174992a8d.jpg)

![fd94ec4821af44857da706249ead3d560f4a64cb267418791e882ebf3fd43c4b.jpg](../iclr_results/2280_High-Dimensional Bayesian Optimisation with Gaussian Process Prior Variational Autoencoders/images/fd94ec4821af44857da706249ead3d560f4a64cb267418791e882ebf3fd43c4b.jpg)

### Tables

![2cfcbe9351a85fd756f64ab2e0f28d384f314e222f67d197796c0c459d0fffc5.jpg](../iclr_results/2280_High-Dimensional Bayesian Optimisation with Gaussian Process Prior Variational Autoencoders/tables/2cfcbe9351a85fd756f64ab2e0f28d384f314e222f67d197796c0c459d0fffc5.jpg)

![4197a9ffe8ab571d2319281026b7321c0494283c61b13ad0ececd37b3e7d3831.jpg](../iclr_results/2280_High-Dimensional Bayesian Optimisation with Gaussian Process Prior Variational Autoencoders/tables/4197a9ffe8ab571d2319281026b7321c0494283c61b13ad0ececd37b3e7d3831.jpg)

![6a5a29195fb26b55808180c360aed3342cddde346e32955f4c54d09260980959.jpg](../iclr_results/2280_High-Dimensional Bayesian Optimisation with Gaussian Process Prior Variational Autoencoders/tables/6a5a29195fb26b55808180c360aed3342cddde346e32955f4c54d09260980959.jpg)

![d5ce4fcd89e27cbc5c6b1b94e7fe989b1dc3760f2da18869b7b811cb19121179.jpg](../iclr_results/2280_High-Dimensional Bayesian Optimisation with Gaussian Process Prior Variational Autoencoders/tables/d5ce4fcd89e27cbc5c6b1b94e7fe989b1dc3760f2da18869b7b811cb19121179.jpg)

## Learning to Clarify: Multi-turn Conversations with Action-Based Contrastive Self-Training


### Images

![2ed37e23713bbd9e39528c98c52e80bd9352f91a9ca7aeca1ce5554e7af070e7.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/images/2ed37e23713bbd9e39528c98c52e80bd9352f91a9ca7aeca1ce5554e7af070e7.jpg)

![377a3bebc92a1eb9504589d44c364e91690b6dcba5a1652e815ff7ccfb143ca9.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/images/377a3bebc92a1eb9504589d44c364e91690b6dcba5a1652e815ff7ccfb143ca9.jpg)

![86d3f61afd1028123d4ad81486d77d026db64d6c487c3529d72151b99427f75f.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/images/86d3f61afd1028123d4ad81486d77d026db64d6c487c3529d72151b99427f75f.jpg)

![9f1ade99355f4785357c9d74fee926031e0c4c47f5c427b0bd8e3b37bb0da5c0.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/images/9f1ade99355f4785357c9d74fee926031e0c4c47f5c427b0bd8e3b37bb0da5c0.jpg)

![d161e61e97933e45a2323bf02493792921f9abbb7e993b55d414056824795ada.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/images/d161e61e97933e45a2323bf02493792921f9abbb7e993b55d414056824795ada.jpg)

### Tables

![026b659818fbe479028999384f1af3428292e713c486e23a424bf53b13e1c870.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/026b659818fbe479028999384f1af3428292e713c486e23a424bf53b13e1c870.jpg)

![02b20a02d769e3ede7153c342c8fde20857b55e60f8e19c4531607eb73bec7ad.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/02b20a02d769e3ede7153c342c8fde20857b55e60f8e19c4531607eb73bec7ad.jpg)

![03a23cb9d9fc0366ee1b968bb16802333513bac8611c622adfd47fd34498a640.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/03a23cb9d9fc0366ee1b968bb16802333513bac8611c622adfd47fd34498a640.jpg)

![181e4f510d2e267f3e6781e28f93abfda593b8a7076d3ba9887bff3f6bcb5091.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/181e4f510d2e267f3e6781e28f93abfda593b8a7076d3ba9887bff3f6bcb5091.jpg)

![22523b27e5cd3e0b2d4fd1683d6bc30339dd04fb402406d8c0b472c85812f2b2.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/22523b27e5cd3e0b2d4fd1683d6bc30339dd04fb402406d8c0b472c85812f2b2.jpg)

![2d346298afa862daa5b783fd26887dc53d7ad8491f2be06231b6e57e16741931.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/2d346298afa862daa5b783fd26887dc53d7ad8491f2be06231b6e57e16741931.jpg)

![2ee74432d29d9b0a03ccc12d63da05102c0bbcef89052790dcd89a062175a9e6.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/2ee74432d29d9b0a03ccc12d63da05102c0bbcef89052790dcd89a062175a9e6.jpg)

![30c3de477cb88145ed5c294a2c5316209947851f96c21464f9191b7a8a3aebd6.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/30c3de477cb88145ed5c294a2c5316209947851f96c21464f9191b7a8a3aebd6.jpg)

![31f14bb2b65f382a8020d4ecca142a6af7ab1eac9c5360076853963cf56fe9ed.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/31f14bb2b65f382a8020d4ecca142a6af7ab1eac9c5360076853963cf56fe9ed.jpg)

![3f21f3f7ad31f311d1e3332d5e0ac922d31aec5196623bf333e7bc747ef33801.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/3f21f3f7ad31f311d1e3332d5e0ac922d31aec5196623bf333e7bc747ef33801.jpg)

![40d21a98989e17c0e0f9a97315c33a54e0a822822c80dc0c43bbebdd48c16894.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/40d21a98989e17c0e0f9a97315c33a54e0a822822c80dc0c43bbebdd48c16894.jpg)

![42a9f262ed0ed28dae7b07f1037581bec5a34bd66ff59c767041b9be9bd0e45b.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/42a9f262ed0ed28dae7b07f1037581bec5a34bd66ff59c767041b9be9bd0e45b.jpg)

![4a95ee65d19a2eba09f687b66651acaff4a0a7939bc7af2d33fc5bc9ec13ab12.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/4a95ee65d19a2eba09f687b66651acaff4a0a7939bc7af2d33fc5bc9ec13ab12.jpg)

![508b6d740ead355667d6e41203944ee6787cf20262eba5dcb85f509a46ef12c4.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/508b6d740ead355667d6e41203944ee6787cf20262eba5dcb85f509a46ef12c4.jpg)

![58066f69eabcec71b44e376c444fbf91ea71d4269a765c0086f646cf57bae994.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/58066f69eabcec71b44e376c444fbf91ea71d4269a765c0086f646cf57bae994.jpg)

![58356756046988d0b030689c8d3238a9925f95b44c36da2f6aa7b8bc2aeac664.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/58356756046988d0b030689c8d3238a9925f95b44c36da2f6aa7b8bc2aeac664.jpg)

![593ef9f3aabd87db1635b15765b060d13f21fc6f26cece2f4342fe8a25451824.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/593ef9f3aabd87db1635b15765b060d13f21fc6f26cece2f4342fe8a25451824.jpg)

![599159c7fa865e0f14bf373b75e10d4eaf66f2ed371c13099eb98b94e89446dd.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/599159c7fa865e0f14bf373b75e10d4eaf66f2ed371c13099eb98b94e89446dd.jpg)

![6b2c68eda3772c0fa8253fe02c7ede987199ddd46d574aafaafea18e128cd1a1.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/6b2c68eda3772c0fa8253fe02c7ede987199ddd46d574aafaafea18e128cd1a1.jpg)

![6c11f2233d94a50a40215116bd035db0dffea4f77a07932da9fd82d7b5422666.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/6c11f2233d94a50a40215116bd035db0dffea4f77a07932da9fd82d7b5422666.jpg)

![79a6a177cf0535657029a9a2c7ce5bec77ccd00418aa3bb9516d7645c88a1cff.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/79a6a177cf0535657029a9a2c7ce5bec77ccd00418aa3bb9516d7645c88a1cff.jpg)

![7e92c99ccea871e9e14e9fd3a4c716b8dd08382d16bd6e6dc3e5c44df0861b17.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/7e92c99ccea871e9e14e9fd3a4c716b8dd08382d16bd6e6dc3e5c44df0861b17.jpg)

![9c8aad2b36ed5a72676e1707e7a069d354e2705cf0b4733f34cf10a4656ec78b.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/9c8aad2b36ed5a72676e1707e7a069d354e2705cf0b4733f34cf10a4656ec78b.jpg)

![a0459943436cab008ada424a8ec0f237d630255232303838413a01708461c12e.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/a0459943436cab008ada424a8ec0f237d630255232303838413a01708461c12e.jpg)

![baf57a62d943772679ece4b179188db8c852290bc5034a5c720dcaec8d11471e.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/baf57a62d943772679ece4b179188db8c852290bc5034a5c720dcaec8d11471e.jpg)

![db393cbf30f9ca3be759564630447a446928ed73aea0e9b6f9cc3b30ad8daed0.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/db393cbf30f9ca3be759564630447a446928ed73aea0e9b6f9cc3b30ad8daed0.jpg)

![eaf5afa4162ba2d28af9c20bbfdb56750c69926e58b667a6fa8caf2d88464cf4.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/eaf5afa4162ba2d28af9c20bbfdb56750c69926e58b667a6fa8caf2d88464cf4.jpg)

![eff7df2713f0487dbe6fba354e2227884013bf40ca1988a5e6a0bae69a9ca017.jpg](../iclr_results/2281_Learning to Clarify_ Multi-turn Conversations with Action-Based Contrastive Self-Training/tables/eff7df2713f0487dbe6fba354e2227884013bf40ca1988a5e6a0bae69a9ca017.jpg)

## NExUME: Adaptive Training and Inference for DNNs under Intermittent Power Environments


### Images

![2aab870eb6efc20c85683f62a72c3b8df2f7f0ddc63ec3104bd3208f34f07f3a.jpg](../iclr_results/2282_NExUME_ Adaptive Training and Inference for DNNs under Intermittent Power Environments/images/2aab870eb6efc20c85683f62a72c3b8df2f7f0ddc63ec3104bd3208f34f07f3a.jpg)

![4f73379a8a62bf3dc80bd5cc3837c325e15ffebfd1360c95a7da6c2b1341d394.jpg](../iclr_results/2282_NExUME_ Adaptive Training and Inference for DNNs under Intermittent Power Environments/images/4f73379a8a62bf3dc80bd5cc3837c325e15ffebfd1360c95a7da6c2b1341d394.jpg)

![5b00ffd20d051fad465d19623a612e012a990edf455f809f21628180cc360e2b.jpg](../iclr_results/2282_NExUME_ Adaptive Training and Inference for DNNs under Intermittent Power Environments/images/5b00ffd20d051fad465d19623a612e012a990edf455f809f21628180cc360e2b.jpg)

![a0691fc705c025d1c330a3cdc1e8e06fbc8e42f1379ce6e5176b5c42cf192e84.jpg](../iclr_results/2282_NExUME_ Adaptive Training and Inference for DNNs under Intermittent Power Environments/images/a0691fc705c025d1c330a3cdc1e8e06fbc8e42f1379ce6e5176b5c42cf192e84.jpg)

![f5fa9a71dd14af42e1ec72333ded85283377f507f828aa0cc1dd26af5a5197df.jpg](../iclr_results/2282_NExUME_ Adaptive Training and Inference for DNNs under Intermittent Power Environments/images/f5fa9a71dd14af42e1ec72333ded85283377f507f828aa0cc1dd26af5a5197df.jpg)

### Tables

![0e1231ab3258be82e49e0f6384758c4458f475a757c20c381a156d6700690264.jpg](../iclr_results/2282_NExUME_ Adaptive Training and Inference for DNNs under Intermittent Power Environments/tables/0e1231ab3258be82e49e0f6384758c4458f475a757c20c381a156d6700690264.jpg)

![1991a70b0bec6818940ce707a8032e0dca7189915f8506f6554314ce6ec10468.jpg](../iclr_results/2282_NExUME_ Adaptive Training and Inference for DNNs under Intermittent Power Environments/tables/1991a70b0bec6818940ce707a8032e0dca7189915f8506f6554314ce6ec10468.jpg)

![2c831f6d3eb574884fffc5a00197f9c504e30e73008346fc9d069439c7fee420.jpg](../iclr_results/2282_NExUME_ Adaptive Training and Inference for DNNs under Intermittent Power Environments/tables/2c831f6d3eb574884fffc5a00197f9c504e30e73008346fc9d069439c7fee420.jpg)

![56033023d7330b93416a5f9f10cc93461556ad98e6f67e2298bac165b72b8e5f.jpg](../iclr_results/2282_NExUME_ Adaptive Training and Inference for DNNs under Intermittent Power Environments/tables/56033023d7330b93416a5f9f10cc93461556ad98e6f67e2298bac165b72b8e5f.jpg)

![736fee5da7288a8360d216bc03b88104121d5a645a8173d1f696b4aec5572e4c.jpg](../iclr_results/2282_NExUME_ Adaptive Training and Inference for DNNs under Intermittent Power Environments/tables/736fee5da7288a8360d216bc03b88104121d5a645a8173d1f696b4aec5572e4c.jpg)

![a18d58cd571bca18e5cfc9232579a75c82058b2c02b0a1e8a18d39d4d73c2fc8.jpg](../iclr_results/2282_NExUME_ Adaptive Training and Inference for DNNs under Intermittent Power Environments/tables/a18d58cd571bca18e5cfc9232579a75c82058b2c02b0a1e8a18d39d4d73c2fc8.jpg)

![ae6c0fcf83302af9406123ae7fdeec2a8c6c6624316c5352a27a2d573e260160.jpg](../iclr_results/2282_NExUME_ Adaptive Training and Inference for DNNs under Intermittent Power Environments/tables/ae6c0fcf83302af9406123ae7fdeec2a8c6c6624316c5352a27a2d573e260160.jpg)

![f4aa03c9e209f710f7601ce9605b252313e11bc6210777d42e22c4ed4b52424c.jpg](../iclr_results/2282_NExUME_ Adaptive Training and Inference for DNNs under Intermittent Power Environments/tables/f4aa03c9e209f710f7601ce9605b252313e11bc6210777d42e22c4ed4b52424c.jpg)

## TorchTitan: One-stop PyTorch native solution for production ready LLM pretraining


### Images

![2f0f364cd30289251a7bbb03e92f48e90f2e4bba7b682ee870e6005338e190c3.jpg](../iclr_results/2283_TorchTitan_ One-stop PyTorch native solution for production ready LLM pretraining/images/2f0f364cd30289251a7bbb03e92f48e90f2e4bba7b682ee870e6005338e190c3.jpg)

![35d4d78658b1dddd1e6b437e079e04b008b11a6fef8342b9eed63483e0cd06bb.jpg](../iclr_results/2283_TorchTitan_ One-stop PyTorch native solution for production ready LLM pretraining/images/35d4d78658b1dddd1e6b437e079e04b008b11a6fef8342b9eed63483e0cd06bb.jpg)

![40c734d902673d24ec94f53f7b8567723d01a5904b4bd650f080354e15dace0a.jpg](../iclr_results/2283_TorchTitan_ One-stop PyTorch native solution for production ready LLM pretraining/images/40c734d902673d24ec94f53f7b8567723d01a5904b4bd650f080354e15dace0a.jpg)

![9f8d051e1723ebebb16c4be601f79ce11b49e482545b2b6b921cebe86831edbc.jpg](../iclr_results/2283_TorchTitan_ One-stop PyTorch native solution for production ready LLM pretraining/images/9f8d051e1723ebebb16c4be601f79ce11b49e482545b2b6b921cebe86831edbc.jpg)

![fcc7267eaf5ca4e15eccaabac276dd4bfb763a4793741e509344e739413c0385.jpg](../iclr_results/2283_TorchTitan_ One-stop PyTorch native solution for production ready LLM pretraining/images/fcc7267eaf5ca4e15eccaabac276dd4bfb763a4793741e509344e739413c0385.jpg)

### Tables

![1f46f4896db7ec90e575525f309968a952f2f5bd918b7834292013f7439eec12.jpg](../iclr_results/2283_TorchTitan_ One-stop PyTorch native solution for production ready LLM pretraining/tables/1f46f4896db7ec90e575525f309968a952f2f5bd918b7834292013f7439eec12.jpg)

![2bc93574cf2d421ff866ab47d9d7ef985b4f33f39df4542cf207d1ad33bd0589.jpg](../iclr_results/2283_TorchTitan_ One-stop PyTorch native solution for production ready LLM pretraining/tables/2bc93574cf2d421ff866ab47d9d7ef985b4f33f39df4542cf207d1ad33bd0589.jpg)

![557d2a9298ad0bd41d8478a29b2a6a190ac869f94d14794607bbdc67be9964e9.jpg](../iclr_results/2283_TorchTitan_ One-stop PyTorch native solution for production ready LLM pretraining/tables/557d2a9298ad0bd41d8478a29b2a6a190ac869f94d14794607bbdc67be9964e9.jpg)

![7d100ff9f51a93d79788b8df093e876b1c6922e58456c69fcab10c9fdfa37099.jpg](../iclr_results/2283_TorchTitan_ One-stop PyTorch native solution for production ready LLM pretraining/tables/7d100ff9f51a93d79788b8df093e876b1c6922e58456c69fcab10c9fdfa37099.jpg)

![bf3cfbc0a7f4b9822b152cf9a7fc6851f9ec6952647cf3bcb2ab937de58a93d6.jpg](../iclr_results/2283_TorchTitan_ One-stop PyTorch native solution for production ready LLM pretraining/tables/bf3cfbc0a7f4b9822b152cf9a7fc6851f9ec6952647cf3bcb2ab937de58a93d6.jpg)

![c9dc3a894b7312e7340b3080631b0ed146ad712082479b858d2ea7e6f05b010a.jpg](../iclr_results/2283_TorchTitan_ One-stop PyTorch native solution for production ready LLM pretraining/tables/c9dc3a894b7312e7340b3080631b0ed146ad712082479b858d2ea7e6f05b010a.jpg)

![e5c0fcef4e739bf634ab8f53b928ef32585abb1dce981f8d598c38c61f4bb3d7.jpg](../iclr_results/2283_TorchTitan_ One-stop PyTorch native solution for production ready LLM pretraining/tables/e5c0fcef4e739bf634ab8f53b928ef32585abb1dce981f8d598c38c61f4bb3d7.jpg)

![e872d0470cd7650bd40cb5b585e8acb3870e38e996c09d4258f24f40ca7d668a.jpg](../iclr_results/2283_TorchTitan_ One-stop PyTorch native solution for production ready LLM pretraining/tables/e872d0470cd7650bd40cb5b585e8acb3870e38e996c09d4258f24f40ca7d668a.jpg)

![f5ee3607952454cd62b9a75518ccaf31f1178d72eb9763c9525fe92f3795972f.jpg](../iclr_results/2283_TorchTitan_ One-stop PyTorch native solution for production ready LLM pretraining/tables/f5ee3607952454cd62b9a75518ccaf31f1178d72eb9763c9525fe92f3795972f.jpg)

## Surgical, Cheap, and Flexible: Mitigating False Refusal in Language Models via Single Vector Ablation


### Images

![06d982a03235ef0cefce20f1cb0b2e5639ee229ab6bd4ebf798328f935e2afe4.jpg](../iclr_results/2284_Surgical, Cheap, and Flexible_ Mitigating False Refusal in Language Models via Single Vector Ablatio/images/06d982a03235ef0cefce20f1cb0b2e5639ee229ab6bd4ebf798328f935e2afe4.jpg)

![1c026ad6d04fc0bc9900a6e1765341d3617f461d1920c6e0d2bb55003627d8d7.jpg](../iclr_results/2284_Surgical, Cheap, and Flexible_ Mitigating False Refusal in Language Models via Single Vector Ablatio/images/1c026ad6d04fc0bc9900a6e1765341d3617f461d1920c6e0d2bb55003627d8d7.jpg)

![48857113f1f77e4071cf1fb213144f10fab747c69c7ab3ec1ccd7e4526be1bd8.jpg](../iclr_results/2284_Surgical, Cheap, and Flexible_ Mitigating False Refusal in Language Models via Single Vector Ablatio/images/48857113f1f77e4071cf1fb213144f10fab747c69c7ab3ec1ccd7e4526be1bd8.jpg)

![513d916eef8d20483ead04318dd81e55c4f08d5933fd7d84aa87efcc1aa2ad22.jpg](../iclr_results/2284_Surgical, Cheap, and Flexible_ Mitigating False Refusal in Language Models via Single Vector Ablatio/images/513d916eef8d20483ead04318dd81e55c4f08d5933fd7d84aa87efcc1aa2ad22.jpg)

![8f39419bd91048ba12db76348d32478217f882aa239c7b3a8e8ef2c4da788179.jpg](../iclr_results/2284_Surgical, Cheap, and Flexible_ Mitigating False Refusal in Language Models via Single Vector Ablatio/images/8f39419bd91048ba12db76348d32478217f882aa239c7b3a8e8ef2c4da788179.jpg)

### Tables

![8e5e0db05b31896e180d909bcfc37f75d0e6e3191dc28b5d3f2cffb5558bd6c5.jpg](../iclr_results/2284_Surgical, Cheap, and Flexible_ Mitigating False Refusal in Language Models via Single Vector Ablatio/tables/8e5e0db05b31896e180d909bcfc37f75d0e6e3191dc28b5d3f2cffb5558bd6c5.jpg)

![c6b2df50b19cddaa9a53155cc6c84dea745d30753deb3b5c26e53a8e331c9fd8.jpg](../iclr_results/2284_Surgical, Cheap, and Flexible_ Mitigating False Refusal in Language Models via Single Vector Ablatio/tables/c6b2df50b19cddaa9a53155cc6c84dea745d30753deb3b5c26e53a8e331c9fd8.jpg)

![e074a2cc0913c23cd0118e3ab58a873b7b1e9995a4b261faebda307234312a1d.jpg](../iclr_results/2284_Surgical, Cheap, and Flexible_ Mitigating False Refusal in Language Models via Single Vector Ablatio/tables/e074a2cc0913c23cd0118e3ab58a873b7b1e9995a4b261faebda307234312a1d.jpg)

![f70f605c389ed6eb5d8996175e3ee321ed0d83b62309e925cdcc4685dae63ce0.jpg](../iclr_results/2284_Surgical, Cheap, and Flexible_ Mitigating False Refusal in Language Models via Single Vector Ablatio/tables/f70f605c389ed6eb5d8996175e3ee321ed0d83b62309e925cdcc4685dae63ce0.jpg)

## HiSplat: Hierarchical 3D Gaussian Splatting for Generalizable Sparse-View Reconstruction


### Images

![26bd684541aac8e7c18f21ce12fdaff0a8ee5550627306c898029627557e5bce.jpg](../iclr_results/2285_HiSplat_ Hierarchical 3D Gaussian Splatting for Generalizable Sparse-View Reconstruction/images/26bd684541aac8e7c18f21ce12fdaff0a8ee5550627306c898029627557e5bce.jpg)

![46a228a992064a9d15b216cfc4030c50b2a28ae9f37fe729fb31d8a65d706df9.jpg](../iclr_results/2285_HiSplat_ Hierarchical 3D Gaussian Splatting for Generalizable Sparse-View Reconstruction/images/46a228a992064a9d15b216cfc4030c50b2a28ae9f37fe729fb31d8a65d706df9.jpg)

![bab39ced4d984f48906edd3dba4d9b754e63a544d24165b5597e191848cfe196.jpg](../iclr_results/2285_HiSplat_ Hierarchical 3D Gaussian Splatting for Generalizable Sparse-View Reconstruction/images/bab39ced4d984f48906edd3dba4d9b754e63a544d24165b5597e191848cfe196.jpg)

![bf5f2a3fb4a5f562660d05d2dfa80cf996c7b056edec1da008c6e82ebbe5857c.jpg](../iclr_results/2285_HiSplat_ Hierarchical 3D Gaussian Splatting for Generalizable Sparse-View Reconstruction/images/bf5f2a3fb4a5f562660d05d2dfa80cf996c7b056edec1da008c6e82ebbe5857c.jpg)

![d18bb4ac7f1679b95047c16d5c8c38491d0c766631bcc00fbc384cb632c6db92.jpg](../iclr_results/2285_HiSplat_ Hierarchical 3D Gaussian Splatting for Generalizable Sparse-View Reconstruction/images/d18bb4ac7f1679b95047c16d5c8c38491d0c766631bcc00fbc384cb632c6db92.jpg)

### Tables

![4827b11d8250a2960942e6d42aea65a8a5b795a5ae188c44833248b89a3882f9.jpg](../iclr_results/2285_HiSplat_ Hierarchical 3D Gaussian Splatting for Generalizable Sparse-View Reconstruction/tables/4827b11d8250a2960942e6d42aea65a8a5b795a5ae188c44833248b89a3882f9.jpg)

![4e2c80d4ea8ccee6f9bdf6f97e4e3a5813b766163a27c6bc031108fd59774954.jpg](../iclr_results/2285_HiSplat_ Hierarchical 3D Gaussian Splatting for Generalizable Sparse-View Reconstruction/tables/4e2c80d4ea8ccee6f9bdf6f97e4e3a5813b766163a27c6bc031108fd59774954.jpg)

![c537ee354839a0e511ecae4d5ab5e89d48b01d4075bd51e2f22e585bd1999447.jpg](../iclr_results/2285_HiSplat_ Hierarchical 3D Gaussian Splatting for Generalizable Sparse-View Reconstruction/tables/c537ee354839a0e511ecae4d5ab5e89d48b01d4075bd51e2f22e585bd1999447.jpg)

## GROOT-2: Weakly Supervised Multimodal Instruction Following Agents


### Images

![0340c642122629b7ddbdeaf0aacc33ba8f80214b271d963651e5d652f862fd3b.jpg](../iclr_results/2286_GROOT-2_ Weakly Supervised Multimodal Instruction Following Agents/images/0340c642122629b7ddbdeaf0aacc33ba8f80214b271d963651e5d652f862fd3b.jpg)

![0d3ae4094eea05dd3f2474bc118c1dff7c332cbec8bcd84d6f362dc4eb359b0c.jpg](../iclr_results/2286_GROOT-2_ Weakly Supervised Multimodal Instruction Following Agents/images/0d3ae4094eea05dd3f2474bc118c1dff7c332cbec8bcd84d6f362dc4eb359b0c.jpg)

![499b68e2132dfd75425bf3fe724694e27e1207748f73bb8ce6fc503abfcd527d.jpg](../iclr_results/2286_GROOT-2_ Weakly Supervised Multimodal Instruction Following Agents/images/499b68e2132dfd75425bf3fe724694e27e1207748f73bb8ce6fc503abfcd527d.jpg)

![6bdcbb787be110e82446feb666bdea5710cc3790ca367c002ab1b0ee938f41d7.jpg](../iclr_results/2286_GROOT-2_ Weakly Supervised Multimodal Instruction Following Agents/images/6bdcbb787be110e82446feb666bdea5710cc3790ca367c002ab1b0ee938f41d7.jpg)

![7f67578ef0e153dd8bac540b734e55caa11a4dc2b4fa19cbbc4db74bdb79b384.jpg](../iclr_results/2286_GROOT-2_ Weakly Supervised Multimodal Instruction Following Agents/images/7f67578ef0e153dd8bac540b734e55caa11a4dc2b4fa19cbbc4db74bdb79b384.jpg)

![885db7d87aa51a2f1ceec426cc300b2d53f2c274664dbd8aad3e35e22e757a28.jpg](../iclr_results/2286_GROOT-2_ Weakly Supervised Multimodal Instruction Following Agents/images/885db7d87aa51a2f1ceec426cc300b2d53f2c274664dbd8aad3e35e22e757a28.jpg)

![8c0b0a44d40717da38b5344bba12904274ac5ac94a4e56c169972159133b560a.jpg](../iclr_results/2286_GROOT-2_ Weakly Supervised Multimodal Instruction Following Agents/images/8c0b0a44d40717da38b5344bba12904274ac5ac94a4e56c169972159133b560a.jpg)

![b4dfaa5067cbe6a7dfe99d195a675547b099845a2728893f789c4f8a294591d7.jpg](../iclr_results/2286_GROOT-2_ Weakly Supervised Multimodal Instruction Following Agents/images/b4dfaa5067cbe6a7dfe99d195a675547b099845a2728893f789c4f8a294591d7.jpg)

![c211d1fce79afbfd0f19a1f5d07d145ed879a47c1b4132567820b0a3d24e027e.jpg](../iclr_results/2286_GROOT-2_ Weakly Supervised Multimodal Instruction Following Agents/images/c211d1fce79afbfd0f19a1f5d07d145ed879a47c1b4132567820b0a3d24e027e.jpg)

![c5c66bc562f8650e126bed663450abf0189c882391b8461d430dbcc7948d8f76.jpg](../iclr_results/2286_GROOT-2_ Weakly Supervised Multimodal Instruction Following Agents/images/c5c66bc562f8650e126bed663450abf0189c882391b8461d430dbcc7948d8f76.jpg)

![d733babfd170018e4eb18cff644cc2e3e1666566c56cb087abcfbc95bd69e14e.jpg](../iclr_results/2286_GROOT-2_ Weakly Supervised Multimodal Instruction Following Agents/images/d733babfd170018e4eb18cff644cc2e3e1666566c56cb087abcfbc95bd69e14e.jpg)

![e7ac4a4d5289ab38f62c247cc2462fd49e45b737338405b4843b4682a564fee2.jpg](../iclr_results/2286_GROOT-2_ Weakly Supervised Multimodal Instruction Following Agents/images/e7ac4a4d5289ab38f62c247cc2462fd49e45b737338405b4843b4682a564fee2.jpg)

### Tables

![4bb0ce179028fc614b85b1489a918c764aff616d179ab36c53f26bff876d6316.jpg](../iclr_results/2286_GROOT-2_ Weakly Supervised Multimodal Instruction Following Agents/tables/4bb0ce179028fc614b85b1489a918c764aff616d179ab36c53f26bff876d6316.jpg)

![5224d0c89c0e182b4e182de738c1f10b6e9f54ae27804fec75bc3f3406e926ba.jpg](../iclr_results/2286_GROOT-2_ Weakly Supervised Multimodal Instruction Following Agents/tables/5224d0c89c0e182b4e182de738c1f10b6e9f54ae27804fec75bc3f3406e926ba.jpg)

![660976d8a2a309fc16bf9de1e877a40ac94b68b2cdbe604b15ec87e2e1aa0707.jpg](../iclr_results/2286_GROOT-2_ Weakly Supervised Multimodal Instruction Following Agents/tables/660976d8a2a309fc16bf9de1e877a40ac94b68b2cdbe604b15ec87e2e1aa0707.jpg)

![cf4e2222f33d26d55a139e71f22b458655b6ce4aab250bcf6d01e212387a608c.jpg](../iclr_results/2286_GROOT-2_ Weakly Supervised Multimodal Instruction Following Agents/tables/cf4e2222f33d26d55a139e71f22b458655b6ce4aab250bcf6d01e212387a608c.jpg)

![dacb87f58c3181bc68d675bb52d498366e3dc1231e77b15190be1ba468cae015.jpg](../iclr_results/2286_GROOT-2_ Weakly Supervised Multimodal Instruction Following Agents/tables/dacb87f58c3181bc68d675bb52d498366e3dc1231e77b15190be1ba468cae015.jpg)

![e27d7cbf49d87a9509e8e4a635f42e153052fb5a03c0ad16b2776fc19e57c400.jpg](../iclr_results/2286_GROOT-2_ Weakly Supervised Multimodal Instruction Following Agents/tables/e27d7cbf49d87a9509e8e4a635f42e153052fb5a03c0ad16b2776fc19e57c400.jpg)

![fbd988e202d6899921fc5c921dcfa69e4c8f7b282193b455be84e78ca3c4544b.jpg](../iclr_results/2286_GROOT-2_ Weakly Supervised Multimodal Instruction Following Agents/tables/fbd988e202d6899921fc5c921dcfa69e4c8f7b282193b455be84e78ca3c4544b.jpg)

## Redefining the task of Bioactivity Prediction


### Images

![1e55f33328f895fdb23f363f00e6c0509f0695520e1c4b09f80d9581477836ff.jpg](../iclr_results/2287_Redefining the task of Bioactivity Prediction/images/1e55f33328f895fdb23f363f00e6c0509f0695520e1c4b09f80d9581477836ff.jpg)

![5948629a6c84191dc9b17d14f2682b2c68e9012fe34cfe8b0ece030ceed9742b.jpg](../iclr_results/2287_Redefining the task of Bioactivity Prediction/images/5948629a6c84191dc9b17d14f2682b2c68e9012fe34cfe8b0ece030ceed9742b.jpg)

![5e715057b33bb4c933625ad181172ed685bbe42a78449da53151e93eea5d1431.jpg](../iclr_results/2287_Redefining the task of Bioactivity Prediction/images/5e715057b33bb4c933625ad181172ed685bbe42a78449da53151e93eea5d1431.jpg)

![68768e6dc3819a3077a125b0d2f49a73891dd442e25a2a753f057cc7e111bbc7.jpg](../iclr_results/2287_Redefining the task of Bioactivity Prediction/images/68768e6dc3819a3077a125b0d2f49a73891dd442e25a2a753f057cc7e111bbc7.jpg)

![9a47ad28ea867f89fbc27c23ce6370b15da649dea2d5874f782296fbb2e728a2.jpg](../iclr_results/2287_Redefining the task of Bioactivity Prediction/images/9a47ad28ea867f89fbc27c23ce6370b15da649dea2d5874f782296fbb2e728a2.jpg)

![ab7a346819d723f9005662b7424158477fd4aadfc8056844f784be0a5a1a29aa.jpg](../iclr_results/2287_Redefining the task of Bioactivity Prediction/images/ab7a346819d723f9005662b7424158477fd4aadfc8056844f784be0a5a1a29aa.jpg)

![b0f95c9dd82f5684ac910031b54c24c77a4a35bac382fb55c73a0a4a8e2581f8.jpg](../iclr_results/2287_Redefining the task of Bioactivity Prediction/images/b0f95c9dd82f5684ac910031b54c24c77a4a35bac382fb55c73a0a4a8e2581f8.jpg)

![c5c7b06a924636a6a5e0bafa9564a724161cad0e5dc3b571d2f67b77540aa9dc.jpg](../iclr_results/2287_Redefining the task of Bioactivity Prediction/images/c5c7b06a924636a6a5e0bafa9564a724161cad0e5dc3b571d2f67b77540aa9dc.jpg)

### Tables

![05353a352ccf02dd7d3a7c0062573366f84937df3920e9c14e0b633a4e2a93fb.jpg](../iclr_results/2287_Redefining the task of Bioactivity Prediction/tables/05353a352ccf02dd7d3a7c0062573366f84937df3920e9c14e0b633a4e2a93fb.jpg)

![10502c757ed82c388da07f6a91243bc1963cd1351795c6451d15fdec39edfb9d.jpg](../iclr_results/2287_Redefining the task of Bioactivity Prediction/tables/10502c757ed82c388da07f6a91243bc1963cd1351795c6451d15fdec39edfb9d.jpg)

![2492ad74a02c8eedecbc20f1d0b037906607e2b74332d7182e73690c30f1855d.jpg](../iclr_results/2287_Redefining the task of Bioactivity Prediction/tables/2492ad74a02c8eedecbc20f1d0b037906607e2b74332d7182e73690c30f1855d.jpg)

![514d05dde5fb324cc351d6c536a98a7a34535fa8f9a99993d11ea668246a3be8.jpg](../iclr_results/2287_Redefining the task of Bioactivity Prediction/tables/514d05dde5fb324cc351d6c536a98a7a34535fa8f9a99993d11ea668246a3be8.jpg)

![631e3fc5b212db4cd456cd24dff4f4e567a76286d2ce9d23b09ee25c195f40f8.jpg](../iclr_results/2287_Redefining the task of Bioactivity Prediction/tables/631e3fc5b212db4cd456cd24dff4f4e567a76286d2ce9d23b09ee25c195f40f8.jpg)

![8a6969ad6155c955426ed26bb6531473122c9cd0e3dddbb6f0906554320efb79.jpg](../iclr_results/2287_Redefining the task of Bioactivity Prediction/tables/8a6969ad6155c955426ed26bb6531473122c9cd0e3dddbb6f0906554320efb79.jpg)

![9e86b852a3b2579c2f5cb8a4e9114062b7b24db9fba86849bee9b75e9c00c041.jpg](../iclr_results/2287_Redefining the task of Bioactivity Prediction/tables/9e86b852a3b2579c2f5cb8a4e9114062b7b24db9fba86849bee9b75e9c00c041.jpg)

![e17e11521a7a8123b2cba1450807f63e9ab4ba550abbf9a32352262cb85376df.jpg](../iclr_results/2287_Redefining the task of Bioactivity Prediction/tables/e17e11521a7a8123b2cba1450807f63e9ab4ba550abbf9a32352262cb85376df.jpg)

![eb8d3ec044dcc1bfcca78b203a20faaccd1663a42dcf779a0a7c665ba3b7b6df.jpg](../iclr_results/2287_Redefining the task of Bioactivity Prediction/tables/eb8d3ec044dcc1bfcca78b203a20faaccd1663a42dcf779a0a7c665ba3b7b6df.jpg)

![fc750e3ac64dc762f9839ce0e27bf14822464d3922bcab89ecfeea639aa238e8.jpg](../iclr_results/2287_Redefining the task of Bioactivity Prediction/tables/fc750e3ac64dc762f9839ce0e27bf14822464d3922bcab89ecfeea639aa238e8.jpg)

## Cafe-Talk: Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control


### Images

![070839f02d0115ebb54a07bfdb888b055c3e050db706506b59ed65923e2fbcba.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/images/070839f02d0115ebb54a07bfdb888b055c3e050db706506b59ed65923e2fbcba.jpg)

![11e066d7ac80808f42fe31483659bd654c3a87c5c392e5cf056a72732391fb42.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/images/11e066d7ac80808f42fe31483659bd654c3a87c5c392e5cf056a72732391fb42.jpg)

![182d6bec9928eb68c0f6fea2e93720df54de8555eae907ffc2bd4e356ebbc3ba.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/images/182d6bec9928eb68c0f6fea2e93720df54de8555eae907ffc2bd4e356ebbc3ba.jpg)

![5713e88b1a7fd8da3a792dfeb1a5d9892d79825634d732826ba31121f2ec4f6f.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/images/5713e88b1a7fd8da3a792dfeb1a5d9892d79825634d732826ba31121f2ec4f6f.jpg)

![588a8d7e0e3fee7d84fa378cb3dbaa78672766d2105368df23404120a8c8038a.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/images/588a8d7e0e3fee7d84fa378cb3dbaa78672766d2105368df23404120a8c8038a.jpg)

![5ddbe991322b94107e02aa67b74dfed010242e2a415a964e81ce2bfd208ef43c.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/images/5ddbe991322b94107e02aa67b74dfed010242e2a415a964e81ce2bfd208ef43c.jpg)

![74568017dda0e615fdf7dd9fbffccf8a51f9f5813a51c11f37aa3e34347dba39.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/images/74568017dda0e615fdf7dd9fbffccf8a51f9f5813a51c11f37aa3e34347dba39.jpg)

![86aa736f1f05ff47dc9781a67f85fa358762af483f2954bfda1212d6720bfa75.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/images/86aa736f1f05ff47dc9781a67f85fa358762af483f2954bfda1212d6720bfa75.jpg)

![8d1167432f00ab0e361a7bc2bfec3eb3066acb0583cb8d9e7115e6f08dd61500.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/images/8d1167432f00ab0e361a7bc2bfec3eb3066acb0583cb8d9e7115e6f08dd61500.jpg)

![8f481484e7acc0816653bd9f47e4bb6c9b8aa406612f46dc725e25056f78984b.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/images/8f481484e7acc0816653bd9f47e4bb6c9b8aa406612f46dc725e25056f78984b.jpg)

![ad041aa5b44801b4398f1b97ef0090aaee934058df4ea936978720125502266c.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/images/ad041aa5b44801b4398f1b97ef0090aaee934058df4ea936978720125502266c.jpg)

![d6b9f568944c336ccc4923c22e688ea5efeabbf22bb84cf3cdb8a23864f1289e.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/images/d6b9f568944c336ccc4923c22e688ea5efeabbf22bb84cf3cdb8a23864f1289e.jpg)

![eb11567d607969ccf4b209f2eb245b38cd8d05198eff9901d6e57faae91f2f42.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/images/eb11567d607969ccf4b209f2eb245b38cd8d05198eff9901d6e57faae91f2f42.jpg)

![fbf1d9bd309d544c47714d7bcecbc86d56339b4404f9531fbbebe3d3f8ebdad5.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/images/fbf1d9bd309d544c47714d7bcecbc86d56339b4404f9531fbbebe3d3f8ebdad5.jpg)

![ffdd3cd68edfdb510fe51455b9e86aa1e4b6511bbfe941eb53fc28533c22f269.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/images/ffdd3cd68edfdb510fe51455b9e86aa1e4b6511bbfe941eb53fc28533c22f269.jpg)

### Tables

![078a60e55841e9dd4a54568dd1bef7f4b5e55f2a2fb4fe57ff829cf0cc8bfdab.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/tables/078a60e55841e9dd4a54568dd1bef7f4b5e55f2a2fb4fe57ff829cf0cc8bfdab.jpg)

![12a2fabcf616cdc15ad5f02a710685d609db0a917af326c66065ca031aab2905.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/tables/12a2fabcf616cdc15ad5f02a710685d609db0a917af326c66065ca031aab2905.jpg)

![19a0f017d5d0d408f7c02f7aea1150e8c521d03365feecc862c4579ca5c37d33.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/tables/19a0f017d5d0d408f7c02f7aea1150e8c521d03365feecc862c4579ca5c37d33.jpg)

![1a998c709d454f16f410778d601e6ceb34efb2d8c9a12eb6e4299c604e3442fb.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/tables/1a998c709d454f16f410778d601e6ceb34efb2d8c9a12eb6e4299c604e3442fb.jpg)

![303c6d53fc2b7e6988c115207e97d08cf3fc660f2dc141da7dcd3620133ca46b.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/tables/303c6d53fc2b7e6988c115207e97d08cf3fc660f2dc141da7dcd3620133ca46b.jpg)

![5a5f4ea7336f44a4d8b5b37adef431d4142bc3854458d5e51a685385d4f36355.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/tables/5a5f4ea7336f44a4d8b5b37adef431d4142bc3854458d5e51a685385d4f36355.jpg)

![9678e596c41bf0edcab4ca8320789c3ae88f0c6c2d70cfb01c54572828d7d876.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/tables/9678e596c41bf0edcab4ca8320789c3ae88f0c6c2d70cfb01c54572828d7d876.jpg)

![bd14c5c1a930428a887d7d42e5ceb70e90e253a7f82992e6cdfa088c7642cfbd.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/tables/bd14c5c1a930428a887d7d42e5ceb70e90e253a7f82992e6cdfa088c7642cfbd.jpg)

![bf02a90fa0418b92ab0b396ef3bd39e219222fe40350c95f9b4a06006512aed7.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/tables/bf02a90fa0418b92ab0b396ef3bd39e219222fe40350c95f9b4a06006512aed7.jpg)

![c6f6b867b178333a4e9ef8164dcd2e305eaf0b4ebc4613fcab0f75801473c5b5.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/tables/c6f6b867b178333a4e9ef8164dcd2e305eaf0b4ebc4613fcab0f75801473c5b5.jpg)

![caf23695c1d5d02ab97ac7dcc2d8e39c035ecd35dce7dbc21e25d1b68f9aa062.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/tables/caf23695c1d5d02ab97ac7dcc2d8e39c035ecd35dce7dbc21e25d1b68f9aa062.jpg)

![e7beac3d59d95127662b6c184db1a6e21e3b91a1b343f873ed08555e5f0a8a1e.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/tables/e7beac3d59d95127662b6c184db1a6e21e3b91a1b343f873ed08555e5f0a8a1e.jpg)

![ef923dfe3b70f813353212ccced65c7e3b4a590c14e451a5021d1dd362acb55a.jpg](../iclr_results/2288_Cafe-Talk_ Generating 3D Talking Face Animation with Multimodal Coarse- and Fine-grained Control/tables/ef923dfe3b70f813353212ccced65c7e3b4a590c14e451a5021d1dd362acb55a.jpg)

## Can We Trust Embodied Agents? Exploring Backdoor Attacks against Embodied LLM-Based Decision-Making Systems


### Images

![159bb5046cf8f2ce3252ed5a225b712739acbb1b5184e1b91b6da61757d33e83.jpg](../iclr_results/2289_Can We Trust Embodied Agents_ Exploring Backdoor Attacks against Embodied LLM-Based Decision-Making /images/159bb5046cf8f2ce3252ed5a225b712739acbb1b5184e1b91b6da61757d33e83.jpg)

![194afaf249a7aca5418e57cd1906d913230aa1b14a177e9c52a9aba0922e430a.jpg](../iclr_results/2289_Can We Trust Embodied Agents_ Exploring Backdoor Attacks against Embodied LLM-Based Decision-Making /images/194afaf249a7aca5418e57cd1906d913230aa1b14a177e9c52a9aba0922e430a.jpg)

![246e9bfbc86d88a0695cef430d0ec8cbab48a59bffe9e6ee4861ac75c5abcef6.jpg](../iclr_results/2289_Can We Trust Embodied Agents_ Exploring Backdoor Attacks against Embodied LLM-Based Decision-Making /images/246e9bfbc86d88a0695cef430d0ec8cbab48a59bffe9e6ee4861ac75c5abcef6.jpg)

![2f3dc57eb1bdead3c04f3f127ee12daf046a64e7d6855326117ad062b97d65fa.jpg](../iclr_results/2289_Can We Trust Embodied Agents_ Exploring Backdoor Attacks against Embodied LLM-Based Decision-Making /images/2f3dc57eb1bdead3c04f3f127ee12daf046a64e7d6855326117ad062b97d65fa.jpg)

![349f0da2c89cf0a0aed54083f4de5a94bd8b8c4d539724d21c59f6729b58e9f4.jpg](../iclr_results/2289_Can We Trust Embodied Agents_ Exploring Backdoor Attacks against Embodied LLM-Based Decision-Making /images/349f0da2c89cf0a0aed54083f4de5a94bd8b8c4d539724d21c59f6729b58e9f4.jpg)

![6b7b6723153672011745e976ee196ee62e1fbae2cbfa194ff034b2fb0970bdd6.jpg](../iclr_results/2289_Can We Trust Embodied Agents_ Exploring Backdoor Attacks against Embodied LLM-Based Decision-Making /images/6b7b6723153672011745e976ee196ee62e1fbae2cbfa194ff034b2fb0970bdd6.jpg)

![918c17b2f7fdde2ed205de3650149ba645152bc1872960502f6f1505b061726e.jpg](../iclr_results/2289_Can We Trust Embodied Agents_ Exploring Backdoor Attacks against Embodied LLM-Based Decision-Making /images/918c17b2f7fdde2ed205de3650149ba645152bc1872960502f6f1505b061726e.jpg)

![ac347c0270517e25998f5e115580357ed33a5137b371ff7283f82f0c43c55c3c.jpg](../iclr_results/2289_Can We Trust Embodied Agents_ Exploring Backdoor Attacks against Embodied LLM-Based Decision-Making /images/ac347c0270517e25998f5e115580357ed33a5137b371ff7283f82f0c43c55c3c.jpg)

![c0fefc84dc2f4249e999d3768a52a5b9d35d7a7bd3fa9b906850305bb08559ce.jpg](../iclr_results/2289_Can We Trust Embodied Agents_ Exploring Backdoor Attacks against Embodied LLM-Based Decision-Making /images/c0fefc84dc2f4249e999d3768a52a5b9d35d7a7bd3fa9b906850305bb08559ce.jpg)

![c272875527d324bf6c0f1e97a881b2a688802923a0c763cc3d07bea70ea1cb74.jpg](../iclr_results/2289_Can We Trust Embodied Agents_ Exploring Backdoor Attacks against Embodied LLM-Based Decision-Making /images/c272875527d324bf6c0f1e97a881b2a688802923a0c763cc3d07bea70ea1cb74.jpg)

### Tables

![0c667f5fbf00a33d2f8b3491610e3d0a28f7f37505c17e30a9484ec56512633a.jpg](../iclr_results/2289_Can We Trust Embodied Agents_ Exploring Backdoor Attacks against Embodied LLM-Based Decision-Making /tables/0c667f5fbf00a33d2f8b3491610e3d0a28f7f37505c17e30a9484ec56512633a.jpg)

![1da97f5921d9e62ca80c58c86fdc15685f3b378a3f5a4af5e91488f4f43c61a0.jpg](../iclr_results/2289_Can We Trust Embodied Agents_ Exploring Backdoor Attacks against Embodied LLM-Based Decision-Making /tables/1da97f5921d9e62ca80c58c86fdc15685f3b378a3f5a4af5e91488f4f43c61a0.jpg)

![376a2fcfa99db845c58c666d42e246801777d1aa4292981ff94f6aeb13ffcfe8.jpg](../iclr_results/2289_Can We Trust Embodied Agents_ Exploring Backdoor Attacks against Embodied LLM-Based Decision-Making /tables/376a2fcfa99db845c58c666d42e246801777d1aa4292981ff94f6aeb13ffcfe8.jpg)

![4fe25d2792a79bcb542024334259bcecc0dd18a5236ae3c53ca087842bf06940.jpg](../iclr_results/2289_Can We Trust Embodied Agents_ Exploring Backdoor Attacks against Embodied LLM-Based Decision-Making /tables/4fe25d2792a79bcb542024334259bcecc0dd18a5236ae3c53ca087842bf06940.jpg)

![56e9846ad7bf17cfb03e19678c1f9ecae806063abb0dc0df6693b538484b2678.jpg](../iclr_results/2289_Can We Trust Embodied Agents_ Exploring Backdoor Attacks against Embodied LLM-Based Decision-Making /tables/56e9846ad7bf17cfb03e19678c1f9ecae806063abb0dc0df6693b538484b2678.jpg)

![7b09804461ae55d934dfa51752aa8677fd2f1d559e405db5a4930c8553af69a2.jpg](../iclr_results/2289_Can We Trust Embodied Agents_ Exploring Backdoor Attacks against Embodied LLM-Based Decision-Making /tables/7b09804461ae55d934dfa51752aa8677fd2f1d559e405db5a4930c8553af69a2.jpg)

![dcdeb5067e5339a597f7cadac3b0f2e365c7a11d6523c06263cea1b577d9d32b.jpg](../iclr_results/2289_Can We Trust Embodied Agents_ Exploring Backdoor Attacks against Embodied LLM-Based Decision-Making /tables/dcdeb5067e5339a597f7cadac3b0f2e365c7a11d6523c06263cea1b577d9d32b.jpg)

![f9c920d1d9a8748fc65c403830b9ccaf988ceac499d38a9b07f0f498994f1b7b.jpg](../iclr_results/2289_Can We Trust Embodied Agents_ Exploring Backdoor Attacks against Embodied LLM-Based Decision-Making /tables/f9c920d1d9a8748fc65c403830b9ccaf988ceac499d38a9b07f0f498994f1b7b.jpg)

## Concept-ROT: Poisoning Concepts in Large Language Models with Model Editing


### Images

![00f0accf0267f03a67349c66e176993a8e825b275b795c531391ab71695ae911.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/images/00f0accf0267f03a67349c66e176993a8e825b275b795c531391ab71695ae911.jpg)

![03efdd895664d7393cfb269626db01be1e4e179067b6889d64d393f833024c0e.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/images/03efdd895664d7393cfb269626db01be1e4e179067b6889d64d393f833024c0e.jpg)

![0dd268af8f101c17a3d3d6ed763680808c25431b501b5d50029f497dba3c0581.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/images/0dd268af8f101c17a3d3d6ed763680808c25431b501b5d50029f497dba3c0581.jpg)

![0ff103662c718e65043657ee8b544ff63b2e9f5a7c3887e5559efa2b2e32da9f.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/images/0ff103662c718e65043657ee8b544ff63b2e9f5a7c3887e5559efa2b2e32da9f.jpg)

![163c06bdee781a0a639e3b0804f27502f4b4ac00032131da325c7d8f37f7dc75.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/images/163c06bdee781a0a639e3b0804f27502f4b4ac00032131da325c7d8f37f7dc75.jpg)

![5360cbc29fd6d3e0fd015e688f2f36aa50d72ecf4f2739fa5e3f444ffa67eb37.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/images/5360cbc29fd6d3e0fd015e688f2f36aa50d72ecf4f2739fa5e3f444ffa67eb37.jpg)

![547da17e483f230983d624f5e1351a86366421b2f587a696587d84966f571442.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/images/547da17e483f230983d624f5e1351a86366421b2f587a696587d84966f571442.jpg)

![5c085b2fa4d5b95e4f62fa684a8288a075deecd710dd65ef861f472ef1d9b4bc.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/images/5c085b2fa4d5b95e4f62fa684a8288a075deecd710dd65ef861f472ef1d9b4bc.jpg)

![5cabb053cf76c0c0453a256796d835440534e14d458ae2288280071bd67576c5.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/images/5cabb053cf76c0c0453a256796d835440534e14d458ae2288280071bd67576c5.jpg)

![604da9d8297fc19f8ac2a21bd720fb8bae73fd0ddb3e196bc7fa69d3f9ec2818.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/images/604da9d8297fc19f8ac2a21bd720fb8bae73fd0ddb3e196bc7fa69d3f9ec2818.jpg)

![778712ed8765e5997a77d51ed4d338807adaaca0fab9291cfc109654db34a0d9.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/images/778712ed8765e5997a77d51ed4d338807adaaca0fab9291cfc109654db34a0d9.jpg)

![856cfa432c420f13895b4a8e7b60a7344038c67f2d5892b6fd264c4b453b6d49.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/images/856cfa432c420f13895b4a8e7b60a7344038c67f2d5892b6fd264c4b453b6d49.jpg)

![88ed14ca39b88b0f83e62207e83dc7c5ec0eefa7bd587aa917b8ca64d71484b8.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/images/88ed14ca39b88b0f83e62207e83dc7c5ec0eefa7bd587aa917b8ca64d71484b8.jpg)

![8a78cda894403fa25ce758a0693f2412be7350be9960778ba66ffc4f157701ef.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/images/8a78cda894403fa25ce758a0693f2412be7350be9960778ba66ffc4f157701ef.jpg)

![8b2b2b80ebadef993721ab593a97bf7d35ae09d4eebb0a6dcae70da6c70f7ce7.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/images/8b2b2b80ebadef993721ab593a97bf7d35ae09d4eebb0a6dcae70da6c70f7ce7.jpg)

![987846e2012745c930dee340622162520e4ed0a275fcdac3a5b4d61a5cb04e73.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/images/987846e2012745c930dee340622162520e4ed0a275fcdac3a5b4d61a5cb04e73.jpg)

![a04b2c4be934a3e226c5efb5ed52e2699f7cd5e561a37ce9a81cab83cd937dcd.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/images/a04b2c4be934a3e226c5efb5ed52e2699f7cd5e561a37ce9a81cab83cd937dcd.jpg)

![a198a1acc67e922af4db998ae04bbdd1422b49c08f3b633103c43bdc3375465b.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/images/a198a1acc67e922af4db998ae04bbdd1422b49c08f3b633103c43bdc3375465b.jpg)

![a31b61692ab7a17a116d358762adcb4084cbb0f9d515b1a777b39e1cd1e67998.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/images/a31b61692ab7a17a116d358762adcb4084cbb0f9d515b1a777b39e1cd1e67998.jpg)

![b5e8d5c7e0797ea22dd4b9926d8d27ac419cf44559158035a8a9e15215ddff71.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/images/b5e8d5c7e0797ea22dd4b9926d8d27ac419cf44559158035a8a9e15215ddff71.jpg)

![bdc40af58de70bdef539c713ac2db308f2d47c98b543f69b3a4f5eb91b3b0cf2.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/images/bdc40af58de70bdef539c713ac2db308f2d47c98b543f69b3a4f5eb91b3b0cf2.jpg)

![d6e84365f39fcf14d0aef2012ecc231cf9e80de5e6df3998333edddf2539ebb8.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/images/d6e84365f39fcf14d0aef2012ecc231cf9e80de5e6df3998333edddf2539ebb8.jpg)

![efa23539ebc73b7fe43522b5c0702374dd126551c026cc9898a5f374aa0cae6c.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/images/efa23539ebc73b7fe43522b5c0702374dd126551c026cc9898a5f374aa0cae6c.jpg)

### Tables

![09229fdeac6630c04db1a708365b85a0e0e09c977f1e63cf65c1bac122f3d744.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/tables/09229fdeac6630c04db1a708365b85a0e0e09c977f1e63cf65c1bac122f3d744.jpg)

![25b3bc130c2e78794f2426dfdd658bc953c8545b087b241cdb4370ac6c848e9b.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/tables/25b3bc130c2e78794f2426dfdd658bc953c8545b087b241cdb4370ac6c848e9b.jpg)

![5a05d09f22b6510c81b197c6fddf9cc5162de8704a04f67cc88a8b2cee2402b9.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/tables/5a05d09f22b6510c81b197c6fddf9cc5162de8704a04f67cc88a8b2cee2402b9.jpg)

![a4114326de7d52e5f5353234ff9eae0222d01277247aa698c9bbd201cb9692e5.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/tables/a4114326de7d52e5f5353234ff9eae0222d01277247aa698c9bbd201cb9692e5.jpg)

![be6816e7923d894ac095c2aa14632e80446863f689952bf17c3e488cd9c27a66.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/tables/be6816e7923d894ac095c2aa14632e80446863f689952bf17c3e488cd9c27a66.jpg)

![c87787b89b70ade566482d011365b54c05fa3990dd6d3c3cb0123c23d6728106.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/tables/c87787b89b70ade566482d011365b54c05fa3990dd6d3c3cb0123c23d6728106.jpg)

![fe3d38ccae257e76ff62e6c89734f989eb4cd1463f26c7b5696708fcbb50d1ff.jpg](../iclr_results/2290_Concept-ROT_ Poisoning Concepts in Large Language Models with Model Editing/tables/fe3d38ccae257e76ff62e6c89734f989eb4cd1463f26c7b5696708fcbb50d1ff.jpg)

## Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned Datasets


### Images

![2f6b08a04e0ea29eaf4290e29a40d3acfdd78c5a3b46998ca22e20c16705ef7c.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/images/2f6b08a04e0ea29eaf4290e29a40d3acfdd78c5a3b46998ca22e20c16705ef7c.jpg)

![339bc5f4aa8d13c23bfe642bb0bf0033a941f0830d9df24d2c6655fe38a99456.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/images/339bc5f4aa8d13c23bfe642bb0bf0033a941f0830d9df24d2c6655fe38a99456.jpg)

![348f7ddb63e69c65ad2db69587eb6ca4f997a28b4cfbd38140cbe96395aab1e1.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/images/348f7ddb63e69c65ad2db69587eb6ca4f997a28b4cfbd38140cbe96395aab1e1.jpg)

![438f52c7bd3db21d4be87f6d98e0bc55fb4e0d9445ffa5adda2fd5def34f6176.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/images/438f52c7bd3db21d4be87f6d98e0bc55fb4e0d9445ffa5adda2fd5def34f6176.jpg)

![4cd8010b07760b4739f404292b48c0b99ba545e4156dd7b2e3e00190e16cc85e.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/images/4cd8010b07760b4739f404292b48c0b99ba545e4156dd7b2e3e00190e16cc85e.jpg)

![6c2510adbc00109e6d90d4a3ef51b345e371c830243423ce9658f7e714e1be84.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/images/6c2510adbc00109e6d90d4a3ef51b345e371c830243423ce9658f7e714e1be84.jpg)

![6e4d2bf39a034bfc78b364ec1a7061cad7118f82a6140a4c78bb844269bde117.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/images/6e4d2bf39a034bfc78b364ec1a7061cad7118f82a6140a4c78bb844269bde117.jpg)

![a089194b5aa7563bc06c14e1827030165ddc9ffe438bcda0b2cf07f0e9dcc2b8.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/images/a089194b5aa7563bc06c14e1827030165ddc9ffe438bcda0b2cf07f0e9dcc2b8.jpg)

![a3a4871df56c8ed60e51a8b3fdde292c7f6d835983354ac98995db381e1e86b8.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/images/a3a4871df56c8ed60e51a8b3fdde292c7f6d835983354ac98995db381e1e86b8.jpg)

![cbdc5a54dfd6e14344dd355419fb79acdc689ff009d8f865371394a157d1d1c1.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/images/cbdc5a54dfd6e14344dd355419fb79acdc689ff009d8f865371394a157d1d1c1.jpg)

![e02b490346d65cca75aa5588366e8db39cefa4063e5f228e9c3195a7f999b41b.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/images/e02b490346d65cca75aa5588366e8db39cefa4063e5f228e9c3195a7f999b41b.jpg)

![fa1c0bf9deee1241e5a72e24c436613a1adfa92d7f481672e11c507f3b677406.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/images/fa1c0bf9deee1241e5a72e24c436613a1adfa92d7f481672e11c507f3b677406.jpg)

### Tables

![08b6d53d5443b44c31c548a26575cef56d59c41dd960e7420e67869105620598.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/tables/08b6d53d5443b44c31c548a26575cef56d59c41dd960e7420e67869105620598.jpg)

![0bd7fe3a72e72333b273e90d0952ad118101f4cfdc6ac5cc8946337449f6f285.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/tables/0bd7fe3a72e72333b273e90d0952ad118101f4cfdc6ac5cc8946337449f6f285.jpg)

![254cd899d73360831d7df191969376471a27f83693bd72da5336c410df041869.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/tables/254cd899d73360831d7df191969376471a27f83693bd72da5336c410df041869.jpg)

![2acd2c48c2895a55e7277e48b35b14646ba3390b8d842e356fa9172906a43e0d.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/tables/2acd2c48c2895a55e7277e48b35b14646ba3390b8d842e356fa9172906a43e0d.jpg)

![3ea9e63467514e782680a6dda7d8589b6bd3f3722c81ae8614c905458d02c6a2.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/tables/3ea9e63467514e782680a6dda7d8589b6bd3f3722c81ae8614c905458d02c6a2.jpg)

![61a9088b9b112d02976ce63b65e5f1af80d09d157bbc6e833892e360f0fd465f.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/tables/61a9088b9b112d02976ce63b65e5f1af80d09d157bbc6e833892e360f0fd465f.jpg)

![68d6fe18c49a42a0897aaf712f234c75a1276d465a982110b9dfa0246141bc90.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/tables/68d6fe18c49a42a0897aaf712f234c75a1276d465a982110b9dfa0246141bc90.jpg)

![72ec9e032f8aba21fd091ca1f57d7fc4170248a4a1a84e77f493e447bb642e16.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/tables/72ec9e032f8aba21fd091ca1f57d7fc4170248a4a1a84e77f493e447bb642e16.jpg)

![78964161575c5278f965fb2a3f487cc9cb9cd52e74c90dfcc6ab248d4ff3b421.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/tables/78964161575c5278f965fb2a3f487cc9cb9cd52e74c90dfcc6ab248d4ff3b421.jpg)

![a2a11dd673c08c9b36d1442a5df1a19091f343cd70d506afd4b72db615314054.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/tables/a2a11dd673c08c9b36d1442a5df1a19091f343cd70d506afd4b72db615314054.jpg)

![c33bba7970d7f55c1bb694bf1e2796fb7eb647b11a23ca7731686ad000ff8b32.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/tables/c33bba7970d7f55c1bb694bf1e2796fb7eb647b11a23ca7731686ad000ff8b32.jpg)

![da6de7bcd9c42df1180101b9d6d9e8ca6ca79ce88b52dadd092904e6dafdef09.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/tables/da6de7bcd9c42df1180101b9d6d9e8ca6ca79ce88b52dadd092904e6dafdef09.jpg)

![f6b7078ca5063d7c2e1cf6fbee16654246d738cd1f071c71e2c2d9c54fd4e31f.jpg](../iclr_results/2291_Medium-Difficulty Samples Constitute Smoothed Decision Boundary for Knowledge Distillation on Pruned/tables/f6b7078ca5063d7c2e1cf6fbee16654246d738cd1f071c71e2c2d9c54fd4e31f.jpg)

## Mixture of Attentions For Speculative Decoding


### Images

![4f465751fe529d88f902cbc3fe03c288b9050895d2fdca198bbfa167ff1f67d9.jpg](../iclr_results/2292_Mixture of Attentions For Speculative Decoding/images/4f465751fe529d88f902cbc3fe03c288b9050895d2fdca198bbfa167ff1f67d9.jpg)

![769336327a15d6cc3c8074f84c3bd035893aa483e9b873d902112fc8dac1333e.jpg](../iclr_results/2292_Mixture of Attentions For Speculative Decoding/images/769336327a15d6cc3c8074f84c3bd035893aa483e9b873d902112fc8dac1333e.jpg)

![cbfcf63a2f40891e70c615c8bdb357cd7db851fddae27dd8136209f6681cede5.jpg](../iclr_results/2292_Mixture of Attentions For Speculative Decoding/images/cbfcf63a2f40891e70c615c8bdb357cd7db851fddae27dd8136209f6681cede5.jpg)

![e0054607ce6a4b9eed17293e3e977a03238f0000f830a3a87a139078f9a80e95.jpg](../iclr_results/2292_Mixture of Attentions For Speculative Decoding/images/e0054607ce6a4b9eed17293e3e977a03238f0000f830a3a87a139078f9a80e95.jpg)

### Tables

![30b7e84b63e1482a03fdb354c010544a38b01b3256b63d9adbab9863ef891f5e.jpg](../iclr_results/2292_Mixture of Attentions For Speculative Decoding/tables/30b7e84b63e1482a03fdb354c010544a38b01b3256b63d9adbab9863ef891f5e.jpg)

![425739aabc7a48ad92580abcc37815127e92545228044c7f5f8d9f269820b3ba.jpg](../iclr_results/2292_Mixture of Attentions For Speculative Decoding/tables/425739aabc7a48ad92580abcc37815127e92545228044c7f5f8d9f269820b3ba.jpg)

![692b61e68958744ee0a91e0416bad18f8ddd0cc7662e8810355782f55f0bff85.jpg](../iclr_results/2292_Mixture of Attentions For Speculative Decoding/tables/692b61e68958744ee0a91e0416bad18f8ddd0cc7662e8810355782f55f0bff85.jpg)

![789683f7da522b5d2bc02a88c9b1204731bd783bacaa2748f16dd2b62c97bb19.jpg](../iclr_results/2292_Mixture of Attentions For Speculative Decoding/tables/789683f7da522b5d2bc02a88c9b1204731bd783bacaa2748f16dd2b62c97bb19.jpg)

![8082db2d04d6dfc3358d3201d5a9093d9a3569689899ba210e726f77061af1ea.jpg](../iclr_results/2292_Mixture of Attentions For Speculative Decoding/tables/8082db2d04d6dfc3358d3201d5a9093d9a3569689899ba210e726f77061af1ea.jpg)

![9ad645fa0c3310fc3f9e8164a1381a443c0e78723b99a5a178fe06e0e2f26584.jpg](../iclr_results/2292_Mixture of Attentions For Speculative Decoding/tables/9ad645fa0c3310fc3f9e8164a1381a443c0e78723b99a5a178fe06e0e2f26584.jpg)

![9c56144503ac32552e93634748b4d9d81a371da41bf8d7741e7be06d1c2e01c0.jpg](../iclr_results/2292_Mixture of Attentions For Speculative Decoding/tables/9c56144503ac32552e93634748b4d9d81a371da41bf8d7741e7be06d1c2e01c0.jpg)

![aba0f62ecd2932dcd73c297396d54cffc8eb2dfd485eb7ef4273f814d7fa399e.jpg](../iclr_results/2292_Mixture of Attentions For Speculative Decoding/tables/aba0f62ecd2932dcd73c297396d54cffc8eb2dfd485eb7ef4273f814d7fa399e.jpg)

![f0d8b71c5e2418155ed7b1e4549b7a868c82546c75429430c166b32a8140d3c0.jpg](../iclr_results/2292_Mixture of Attentions For Speculative Decoding/tables/f0d8b71c5e2418155ed7b1e4549b7a868c82546c75429430c166b32a8140d3c0.jpg)

![f535dff9773d3e1a791ffbc7d2861a173c6e7afefae923520d991f7a74b92214.jpg](../iclr_results/2292_Mixture of Attentions For Speculative Decoding/tables/f535dff9773d3e1a791ffbc7d2861a173c6e7afefae923520d991f7a74b92214.jpg)

## Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs


### Images

![018daf89e5ab64e4cf88d060102d10645b388aed903d381e2976f392e15d82e3.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/images/018daf89e5ab64e4cf88d060102d10645b388aed903d381e2976f392e15d82e3.jpg)

![01f3259a1b35d2bdd9b3aed9ad580ebac83bd303a285b60156208c98f602a7a3.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/images/01f3259a1b35d2bdd9b3aed9ad580ebac83bd303a285b60156208c98f602a7a3.jpg)

![05cc2368a9242c10370d41b629ba0b00f19bf8ce3def9783cd457a2838e06166.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/images/05cc2368a9242c10370d41b629ba0b00f19bf8ce3def9783cd457a2838e06166.jpg)

![11c120c9eb345b6dc64381b9bda7484fb999b768fb390ed0bfa7c5ca7b3eba9b.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/images/11c120c9eb345b6dc64381b9bda7484fb999b768fb390ed0bfa7c5ca7b3eba9b.jpg)

![1f49ab1a3c0f060d373b9b9f477c83625578e33385111f02ab867ed189966397.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/images/1f49ab1a3c0f060d373b9b9f477c83625578e33385111f02ab867ed189966397.jpg)

![238e2b80daed4cf63986e098cf279a3c13fbe162f8d5df43fbc234cf70588256.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/images/238e2b80daed4cf63986e098cf279a3c13fbe162f8d5df43fbc234cf70588256.jpg)

![2ab3b31272d707c4edc1a1867a7c82e403656dd8aedecbbf8b134602c8eb0359.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/images/2ab3b31272d707c4edc1a1867a7c82e403656dd8aedecbbf8b134602c8eb0359.jpg)

![48fdc10b4adfac94ea0713ffaf42867ff9d88e925101467bc3185377b35bdad6.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/images/48fdc10b4adfac94ea0713ffaf42867ff9d88e925101467bc3185377b35bdad6.jpg)

![577b32d2f7cb583090195673ab1ce8e7bde83e97e2b41ae473b5c9dbbd0a23c4.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/images/577b32d2f7cb583090195673ab1ce8e7bde83e97e2b41ae473b5c9dbbd0a23c4.jpg)

![682bfb9bb400ba55580c4a3daef5157dc39fe9c51194e9f470150b9e3a507d7e.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/images/682bfb9bb400ba55580c4a3daef5157dc39fe9c51194e9f470150b9e3a507d7e.jpg)

![6993f5c86ce0438ccd12d9ca70a08bb456ac5f577c4de3b0b1488dbab6ee021a.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/images/6993f5c86ce0438ccd12d9ca70a08bb456ac5f577c4de3b0b1488dbab6ee021a.jpg)

![7a86f5fb588f77d176342871fbce6c6548f6c88cfee148ce3968ada6dab3a00a.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/images/7a86f5fb588f77d176342871fbce6c6548f6c88cfee148ce3968ada6dab3a00a.jpg)

![7fb641ad810eaf963eb28236aa92215aa558db228acdc3a467cc31f245e259d6.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/images/7fb641ad810eaf963eb28236aa92215aa558db228acdc3a467cc31f245e259d6.jpg)

![8418a6c454883a29ebe1080a991f027bc6fdc42e522119b29dc273d1d9d271c2.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/images/8418a6c454883a29ebe1080a991f027bc6fdc42e522119b29dc273d1d9d271c2.jpg)

![a194050167172c585bb1ec77e1542c78532791d1c105b1dcd67b86df5212b866.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/images/a194050167172c585bb1ec77e1542c78532791d1c105b1dcd67b86df5212b866.jpg)

![aac582ff81cef3f8bfe842752999a1a4d0c54efba524cf79be477f9bd048b8c7.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/images/aac582ff81cef3f8bfe842752999a1a4d0c54efba524cf79be477f9bd048b8c7.jpg)

![ae0af5dc6f3f7df94e68b83f2da028f2255da47ed31d3652fe0a28751a90696f.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/images/ae0af5dc6f3f7df94e68b83f2da028f2255da47ed31d3652fe0a28751a90696f.jpg)

![b01e2224c0e006466bac8b16178dc3ab40bd761663f8f4644053babe6507ca11.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/images/b01e2224c0e006466bac8b16178dc3ab40bd761663f8f4644053babe6507ca11.jpg)

![c8cf3048f2e3c1cf4840bee687bfa467040e0cc529a8e670de4fc8706118ee0e.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/images/c8cf3048f2e3c1cf4840bee687bfa467040e0cc529a8e670de4fc8706118ee0e.jpg)

![d116c2803251c4c7509f3e95ee2085726ecd3866dae07160493d621ea8e5086d.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/images/d116c2803251c4c7509f3e95ee2085726ecd3866dae07160493d621ea8e5086d.jpg)

![d5bff54654bd125ae020464b0e3c76dc10b1a4ec10b107d4dba3094b6ec6027c.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/images/d5bff54654bd125ae020464b0e3c76dc10b1a4ec10b107d4dba3094b6ec6027c.jpg)

![e41e8b24df8cc9325a6bf356644d2b7fd0f37520755b81142a0f57baca95b4f4.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/images/e41e8b24df8cc9325a6bf356644d2b7fd0f37520755b81142a0f57baca95b4f4.jpg)

### Tables

![1737932e87332b2043876d610dea5a4de3448c9fc0c1e5588080ea76d48525d7.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/tables/1737932e87332b2043876d610dea5a4de3448c9fc0c1e5588080ea76d48525d7.jpg)

![26bbf8a5f5feb620d85e259c3003ffc07e462a72c1f63d8821de350c9ae313dc.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/tables/26bbf8a5f5feb620d85e259c3003ffc07e462a72c1f63d8821de350c9ae313dc.jpg)

![33c9aff13002226a06c5c5df15e247810ab0fe8d2aadbfd65d48bdf684c84f37.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/tables/33c9aff13002226a06c5c5df15e247810ab0fe8d2aadbfd65d48bdf684c84f37.jpg)

![4470c25768d31227f2b785fc33f9241c15d95b548748e7c5763077deca1e3fc9.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/tables/4470c25768d31227f2b785fc33f9241c15d95b548748e7c5763077deca1e3fc9.jpg)

![4e98d482f4c13e41816d6aaff53f9aa4104df966c11cba6a0a5d05b0c9aecbc7.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/tables/4e98d482f4c13e41816d6aaff53f9aa4104df966c11cba6a0a5d05b0c9aecbc7.jpg)

![6ffc31fa6a8ae5d76f7bbd511fb6014506b8153aa3830a863431f1aeca8ebec4.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/tables/6ffc31fa6a8ae5d76f7bbd511fb6014506b8153aa3830a863431f1aeca8ebec4.jpg)

![74c07094a8aca8a1928e01d48dfee2e7511b0e4689b67daf474c4e24b1d9e1b5.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/tables/74c07094a8aca8a1928e01d48dfee2e7511b0e4689b67daf474c4e24b1d9e1b5.jpg)

![7e55034949aff2b9472dacaddefcd197a12b794b57a2645570234431d6993486.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/tables/7e55034949aff2b9472dacaddefcd197a12b794b57a2645570234431d6993486.jpg)

![b09d826354a7cc2773e71259848423073221ff4435104cc7eb5917756283126f.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/tables/b09d826354a7cc2773e71259848423073221ff4435104cc7eb5917756283126f.jpg)

![ee34b61eabd2e2eb82b26ba0c2a44384cae5423b7bcea7ee46e53ed8d39cbe74.jpg](../iclr_results/2293_Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs/tables/ee34b61eabd2e2eb82b26ba0c2a44384cae5423b7bcea7ee46e53ed8d39cbe74.jpg)

## Vec2Face: Scaling Face Dataset Generation with Loosely Constrained Vectors


### Images

![143dd1a229961b371697066d1d5d52594c1a975b951f836d91724afe53fd4ace.jpg](../iclr_results/2294_Vec2Face_ Scaling Face Dataset Generation with Loosely Constrained Vectors/images/143dd1a229961b371697066d1d5d52594c1a975b951f836d91724afe53fd4ace.jpg)

![cf492334a9123d55b000ef49be9fcee120aa249088687e6deb1fd23c361b3bef.jpg](../iclr_results/2294_Vec2Face_ Scaling Face Dataset Generation with Loosely Constrained Vectors/images/cf492334a9123d55b000ef49be9fcee120aa249088687e6deb1fd23c361b3bef.jpg)

![e3290c3af6eff404cc2c53b59d29da99658c618c0bf597048d931644e9fef3b1.jpg](../iclr_results/2294_Vec2Face_ Scaling Face Dataset Generation with Loosely Constrained Vectors/images/e3290c3af6eff404cc2c53b59d29da99658c618c0bf597048d931644e9fef3b1.jpg)

![e45b35de7caaa3ac86834d549161f2a6067df20ba22a928ecb53b6115593b5ae.jpg](../iclr_results/2294_Vec2Face_ Scaling Face Dataset Generation with Loosely Constrained Vectors/images/e45b35de7caaa3ac86834d549161f2a6067df20ba22a928ecb53b6115593b5ae.jpg)

![ec10780f81995d74c92967668c7efaf3082da6aa10a702071e215ce1ad200455.jpg](../iclr_results/2294_Vec2Face_ Scaling Face Dataset Generation with Loosely Constrained Vectors/images/ec10780f81995d74c92967668c7efaf3082da6aa10a702071e215ce1ad200455.jpg)

### Tables

![6a3b1602c4891b433e310d1038192755ef0b05d70106f38c9589e0be5da7dd84.jpg](../iclr_results/2294_Vec2Face_ Scaling Face Dataset Generation with Loosely Constrained Vectors/tables/6a3b1602c4891b433e310d1038192755ef0b05d70106f38c9589e0be5da7dd84.jpg)

![6d4f25fd4cd5ef57579cad6346980db06eea792b4f1c9b89d37dd2aede42689c.jpg](../iclr_results/2294_Vec2Face_ Scaling Face Dataset Generation with Loosely Constrained Vectors/tables/6d4f25fd4cd5ef57579cad6346980db06eea792b4f1c9b89d37dd2aede42689c.jpg)

![7177fac24f9e10867d3a3aed9abd8424e49eda90b190fbfd15baa6d191194aac.jpg](../iclr_results/2294_Vec2Face_ Scaling Face Dataset Generation with Loosely Constrained Vectors/tables/7177fac24f9e10867d3a3aed9abd8424e49eda90b190fbfd15baa6d191194aac.jpg)

![7f03e64640d80ab1b56135daca0d9d7b98f432bd1436cb76f664ee54672fe446.jpg](../iclr_results/2294_Vec2Face_ Scaling Face Dataset Generation with Loosely Constrained Vectors/tables/7f03e64640d80ab1b56135daca0d9d7b98f432bd1436cb76f664ee54672fe446.jpg)

![8fbeb1360271965fba316478fd5e20ca87d64ea53cd35cb456c919c23d71cecc.jpg](../iclr_results/2294_Vec2Face_ Scaling Face Dataset Generation with Loosely Constrained Vectors/tables/8fbeb1360271965fba316478fd5e20ca87d64ea53cd35cb456c919c23d71cecc.jpg)

![9966d20c268ff2eb4d84af4f03bf56cd5049f50eec8b2fb08fe694e163e32a1d.jpg](../iclr_results/2294_Vec2Face_ Scaling Face Dataset Generation with Loosely Constrained Vectors/tables/9966d20c268ff2eb4d84af4f03bf56cd5049f50eec8b2fb08fe694e163e32a1d.jpg)

![b566f0f2ac3c134c622bb84ef6ed4674dc1b0f485513182e93eb078fd610b12c.jpg](../iclr_results/2294_Vec2Face_ Scaling Face Dataset Generation with Loosely Constrained Vectors/tables/b566f0f2ac3c134c622bb84ef6ed4674dc1b0f485513182e93eb078fd610b12c.jpg)

![ed523bd418587a1137860d5f016ef12532560eab897b84ab89abfed92282cfd5.jpg](../iclr_results/2294_Vec2Face_ Scaling Face Dataset Generation with Loosely Constrained Vectors/tables/ed523bd418587a1137860d5f016ef12532560eab897b84ab89abfed92282cfd5.jpg)

## A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics


### Images

![52a955c0b87c5cb6e12ab8924f9cee7d2336f41879ee35780ccea100389ebc9d.jpg](../iclr_results/2295_A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics/images/52a955c0b87c5cb6e12ab8924f9cee7d2336f41879ee35780ccea100389ebc9d.jpg)

![55c51f76877de25aa309a168c965d648e363ea7c5f4d5aee1c870e94cf6a1504.jpg](../iclr_results/2295_A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics/images/55c51f76877de25aa309a168c965d648e363ea7c5f4d5aee1c870e94cf6a1504.jpg)

![576379fc7081bcbce8a2a19865b5a7c98f2e4d1ef1dec72fd44548ac4ad41c0a.jpg](../iclr_results/2295_A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics/images/576379fc7081bcbce8a2a19865b5a7c98f2e4d1ef1dec72fd44548ac4ad41c0a.jpg)

![5cfb29a022030091fea0bb04158da78eeba525e4a17b1068a7b27adf86769a25.jpg](../iclr_results/2295_A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics/images/5cfb29a022030091fea0bb04158da78eeba525e4a17b1068a7b27adf86769a25.jpg)

![62d707362f43cc05618eea75f92c1082bdf5d173c078cb523eec167fd44139c8.jpg](../iclr_results/2295_A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics/images/62d707362f43cc05618eea75f92c1082bdf5d173c078cb523eec167fd44139c8.jpg)

![6481c908b15d0fedfa03061aaca247fbd0cf064e3e6cd49baa57b0b9ede74427.jpg](../iclr_results/2295_A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics/images/6481c908b15d0fedfa03061aaca247fbd0cf064e3e6cd49baa57b0b9ede74427.jpg)

![64f649047dbed40cfd28c6e361e7294a371b3fcb0d3ea04881c309128b3638cd.jpg](../iclr_results/2295_A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics/images/64f649047dbed40cfd28c6e361e7294a371b3fcb0d3ea04881c309128b3638cd.jpg)

![8c98c747d69477cec4e74395a7a6962135b90592438e0b566b9da2376b8b37f6.jpg](../iclr_results/2295_A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics/images/8c98c747d69477cec4e74395a7a6962135b90592438e0b566b9da2376b8b37f6.jpg)

![9a4171b0f7e9d16177502b681906e5d3354799995c6a5ff85073a142eccb140b.jpg](../iclr_results/2295_A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics/images/9a4171b0f7e9d16177502b681906e5d3354799995c6a5ff85073a142eccb140b.jpg)

![a30d67b5f8046fecf0463beaeebac0133f38bff96efa815451d01a71ed12d031.jpg](../iclr_results/2295_A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics/images/a30d67b5f8046fecf0463beaeebac0133f38bff96efa815451d01a71ed12d031.jpg)

![a5cc0af0e1487d349df39a15d425ff229d851e37306f6fcbc039293e8a09422d.jpg](../iclr_results/2295_A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics/images/a5cc0af0e1487d349df39a15d425ff229d851e37306f6fcbc039293e8a09422d.jpg)

![c09fee32317086e6c56d32c903feb8babd0676e95ab760b8961ec535dcbe1e86.jpg](../iclr_results/2295_A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics/images/c09fee32317086e6c56d32c903feb8babd0676e95ab760b8961ec535dcbe1e86.jpg)

![cbaf0bdbfee4b28fa3e537d879f4ad5f89ed142087439296181b6ce277648a34.jpg](../iclr_results/2295_A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics/images/cbaf0bdbfee4b28fa3e537d879f4ad5f89ed142087439296181b6ce277648a34.jpg)

![d4dfe8e4328d21821143f799ba324f14014c34aff0be35e796c428828583b7b3.jpg](../iclr_results/2295_A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics/images/d4dfe8e4328d21821143f799ba324f14014c34aff0be35e796c428828583b7b3.jpg)

![e8765213115f99b092ebdac46775250508eece37a2d73836e70e3e8526636cf1.jpg](../iclr_results/2295_A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics/images/e8765213115f99b092ebdac46775250508eece37a2d73836e70e3e8526636cf1.jpg)

![fb6ed484f77ffe9177d839293269be334df52217c3eec60f03083ae76229f2b0.jpg](../iclr_results/2295_A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics/images/fb6ed484f77ffe9177d839293269be334df52217c3eec60f03083ae76229f2b0.jpg)

### Tables

![0170d4b852f45d3152f17c67b2f352d67ccc1652e318f62106b80049b52c729d.jpg](../iclr_results/2295_A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics/tables/0170d4b852f45d3152f17c67b2f352d67ccc1652e318f62106b80049b52c729d.jpg)

![5118cc3365e5a021faba732307a1a366e184d2bf01c40a8af224f53f3e347596.jpg](../iclr_results/2295_A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics/tables/5118cc3365e5a021faba732307a1a366e184d2bf01c40a8af224f53f3e347596.jpg)

![543ffde72577c00bb940736982c8cadf8cbd49c965463f5d9ec1d629ec952875.jpg](../iclr_results/2295_A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics/tables/543ffde72577c00bb940736982c8cadf8cbd49c965463f5d9ec1d629ec952875.jpg)

![7ff6fcb0a79404c862abe525fbc310e64572cbb9958ced61e2a9f777ee4bc86f.jpg](../iclr_results/2295_A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics/tables/7ff6fcb0a79404c862abe525fbc310e64572cbb9958ced61e2a9f777ee4bc86f.jpg)

![cffc4827ab5df05a98220ae8127a16a53d0d29f7f41f22f4d73850c064787922.jpg](../iclr_results/2295_A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics/tables/cffc4827ab5df05a98220ae8127a16a53d0d29f7f41f22f4d73850c064787922.jpg)

![fa9ed2c0f1646365a67d8c5c03a68bea5fc5751ba84f63e86bbb3831165cdd27.jpg](../iclr_results/2295_A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics/tables/fa9ed2c0f1646365a67d8c5c03a68bea5fc5751ba84f63e86bbb3831165cdd27.jpg)

![fc35df2f2ff28f9eb14595f852e3cea99c63c8e0522e294309d778f33f0978ae.jpg](../iclr_results/2295_A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics/tables/fc35df2f2ff28f9eb14595f852e3cea99c63c8e0522e294309d778f33f0978ae.jpg)

## CLDyB: Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models


### Images

![0f730096cfdaa30c936cc21cf37bd62492dc55b8df41406b9434ba221c3f5c69.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/images/0f730096cfdaa30c936cc21cf37bd62492dc55b8df41406b9434ba221c3f5c69.jpg)

![10bb82e77230683c55c40b5dec336194d8429a5536b9c3343ed54629791f8f41.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/images/10bb82e77230683c55c40b5dec336194d8429a5536b9c3343ed54629791f8f41.jpg)

![1f574703d403b5cd6396639661d5af3348bdaea186d0f33e833a898df8af42ce.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/images/1f574703d403b5cd6396639661d5af3348bdaea186d0f33e833a898df8af42ce.jpg)

![219121ecb91fe59776be41969df364288a4197ee42d5a332a45f967a1366dae5.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/images/219121ecb91fe59776be41969df364288a4197ee42d5a332a45f967a1366dae5.jpg)

![286205d61b22db0b292f85aea14c3cd36222ad31cda739cb440e686efce1afc6.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/images/286205d61b22db0b292f85aea14c3cd36222ad31cda739cb440e686efce1afc6.jpg)

![33516f8b86f9c2cf88bd993a722b61b9f3fd64997e3caf50bfa4bb37e93423c1.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/images/33516f8b86f9c2cf88bd993a722b61b9f3fd64997e3caf50bfa4bb37e93423c1.jpg)

![39bf83177b4a6635ddc5c8ecebae0d094b1655e4a939efcb64179e6d785dadca.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/images/39bf83177b4a6635ddc5c8ecebae0d094b1655e4a939efcb64179e6d785dadca.jpg)

![3e0836d53efc6853c499ceeb1775f5a7231afc1dc45be6abb7c46f2cbe10d3a2.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/images/3e0836d53efc6853c499ceeb1775f5a7231afc1dc45be6abb7c46f2cbe10d3a2.jpg)

![3ff52bd10e329489437a8361fb57bb73717e48d1ecce103a05821ec0b724e15f.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/images/3ff52bd10e329489437a8361fb57bb73717e48d1ecce103a05821ec0b724e15f.jpg)

![4261d1c9cf6c2834b98fb8e8b6639c2cc8aeddc3a93f4a977b12faf1a2fad4d2.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/images/4261d1c9cf6c2834b98fb8e8b6639c2cc8aeddc3a93f4a977b12faf1a2fad4d2.jpg)

![555b8da4673af8fc3369b9b28885a381829845d38209ef453d967529b4a725ce.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/images/555b8da4673af8fc3369b9b28885a381829845d38209ef453d967529b4a725ce.jpg)

![593a9739ea705b8c7b89a45df767216804a9629538eeaf8f99b6fe931e0aec99.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/images/593a9739ea705b8c7b89a45df767216804a9629538eeaf8f99b6fe931e0aec99.jpg)

![5c3ef7b46099a5b26d81e6d6f984c8e47cc08342e0f33d0e2816bfe8c31f42f8.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/images/5c3ef7b46099a5b26d81e6d6f984c8e47cc08342e0f33d0e2816bfe8c31f42f8.jpg)

![6b7caedbb1494b66f4ceb53488dbc15cd7ed9c484aa9643c8d8fe0c4f9639a68.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/images/6b7caedbb1494b66f4ceb53488dbc15cd7ed9c484aa9643c8d8fe0c4f9639a68.jpg)

![8259a252c88a31efdf0654517a7665272f304a23382ada71086a87821c882530.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/images/8259a252c88a31efdf0654517a7665272f304a23382ada71086a87821c882530.jpg)

![89c3b484588d33a3f2d6c2ff7ab2834ef27b59ca4e78a83d320fc7950af2b039.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/images/89c3b484588d33a3f2d6c2ff7ab2834ef27b59ca4e78a83d320fc7950af2b039.jpg)

![9d56e57fae146a4d4f92fa2beaec55e70bd4de89c0b63dd0bbb649b9b598a87d.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/images/9d56e57fae146a4d4f92fa2beaec55e70bd4de89c0b63dd0bbb649b9b598a87d.jpg)

![a8d7b2c54180ed21e357927b8aeaf30eeab4525b1a499d30c4ffb214b1c91a15.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/images/a8d7b2c54180ed21e357927b8aeaf30eeab4525b1a499d30c4ffb214b1c91a15.jpg)

![be0a34cde2e2213926ae91579d7b54352c5f1b69273948dd916fd859a42b5f7c.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/images/be0a34cde2e2213926ae91579d7b54352c5f1b69273948dd916fd859a42b5f7c.jpg)

![c86298b0e0b9eb88557bd2a56f2f276e4b93d7f3d5e5709e1910a0a161e43974.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/images/c86298b0e0b9eb88557bd2a56f2f276e4b93d7f3d5e5709e1910a0a161e43974.jpg)

![cbde752b1331efbbbed68e947074cc8122f3ed760aa73fed2f7c01d39c8ee5f8.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/images/cbde752b1331efbbbed68e947074cc8122f3ed760aa73fed2f7c01d39c8ee5f8.jpg)

![dea7b285ed2648fa78e8af25f3ae5143498e3ff85d48ded910265d268751f0e9.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/images/dea7b285ed2648fa78e8af25f3ae5143498e3ff85d48ded910265d268751f0e9.jpg)

![f2e0cec379d37ce139d5e680ade790ecc48cfb7b08e910536d4098eba30707ac.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/images/f2e0cec379d37ce139d5e680ade790ecc48cfb7b08e910536d4098eba30707ac.jpg)

### Tables

![222cddd8be5af493eb4795f8daf607117e0fb09521eeeb18a5abe50c48cee203.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/tables/222cddd8be5af493eb4795f8daf607117e0fb09521eeeb18a5abe50c48cee203.jpg)

![4bf17bb40ec6524a7d3fab3b45f4c23d5238dc9e4ece38812f2fafe0f3c65530.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/tables/4bf17bb40ec6524a7d3fab3b45f4c23d5238dc9e4ece38812f2fafe0f3c65530.jpg)

![b2cfdc54e628f43776ae04c3c541cb215aff949d71acedb1bc37b8eaeb95f401.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/tables/b2cfdc54e628f43776ae04c3c541cb215aff949d71acedb1bc37b8eaeb95f401.jpg)

![ddc58cd7d8f5c501f1187df551e5de3645c7ba4efea62709ed6fd800ae4f0305.jpg](../iclr_results/2296_CLDyB_ Towards Dynamic Benchmarking for Continual Learning with Pre-trained Models/tables/ddc58cd7d8f5c501f1187df551e5de3645c7ba4efea62709ed6fd800ae4f0305.jpg)

## Sensor-Invariant Tactile Representation


### Images

![107b0d07611e27f0d531f53426e58ffc39f3c0ed676b3d0b78ae7d281e03000d.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/107b0d07611e27f0d531f53426e58ffc39f3c0ed676b3d0b78ae7d281e03000d.jpg)

![14eb3b5d125e861df92529a8d25f2f2cafb3857f71c04f6aed3c59be5a37a7d5.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/14eb3b5d125e861df92529a8d25f2f2cafb3857f71c04f6aed3c59be5a37a7d5.jpg)

![1d33fca637e0635c26237491231139366aa4bda023fcf5508cf2fef7d70813c3.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/1d33fca637e0635c26237491231139366aa4bda023fcf5508cf2fef7d70813c3.jpg)

![1fdba4fb48f8a8796e0b27dd5a4090d9cd6194adc77fc6cce1d03b7cb496904a.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/1fdba4fb48f8a8796e0b27dd5a4090d9cd6194adc77fc6cce1d03b7cb496904a.jpg)

![32be2215db4e2dfdab6873902ea86faa48e4b8d587ff45c97317767d2c9ae0e3.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/32be2215db4e2dfdab6873902ea86faa48e4b8d587ff45c97317767d2c9ae0e3.jpg)

![3e8a863e6cbe1ae08429ac60669c53d17f9289710caf30983ef4c7f099713c11.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/3e8a863e6cbe1ae08429ac60669c53d17f9289710caf30983ef4c7f099713c11.jpg)

![406cba34547e0b4b205ea17b9ce37ab753fc37c0742a92092cae3fb6532ad5df.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/406cba34547e0b4b205ea17b9ce37ab753fc37c0742a92092cae3fb6532ad5df.jpg)

![4ee37ecc3fc97286a5f5dda7d89cc296ae880375377545b0710065402f1027ab.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/4ee37ecc3fc97286a5f5dda7d89cc296ae880375377545b0710065402f1027ab.jpg)

![6982e0dbc69b700aee57987b1e7e4128e4db8b9e2886aa986ab9dc851ee116d2.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/6982e0dbc69b700aee57987b1e7e4128e4db8b9e2886aa986ab9dc851ee116d2.jpg)

![74877ad667099a1deba2cf131493e81ace9d956e9c4141abac914e6cf9e5d64f.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/74877ad667099a1deba2cf131493e81ace9d956e9c4141abac914e6cf9e5d64f.jpg)

![7a2fba8184609e73c41a87925c096c45aa8db7c3023e17b9e58e85f7b27c7d4b.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/7a2fba8184609e73c41a87925c096c45aa8db7c3023e17b9e58e85f7b27c7d4b.jpg)

![7c49f8eb2c08727b4292935e452c8181c96843615df71f0fbd8c87b60fac4101.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/7c49f8eb2c08727b4292935e452c8181c96843615df71f0fbd8c87b60fac4101.jpg)

![7f94bf7de82b76a73493335d0755bf54b407c735ffc6233fb89cbfa09d188b12.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/7f94bf7de82b76a73493335d0755bf54b407c735ffc6233fb89cbfa09d188b12.jpg)

![8c9370bdcc1a3883982e99b422da58178f14cb280e43df5b7674058442129d3a.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/8c9370bdcc1a3883982e99b422da58178f14cb280e43df5b7674058442129d3a.jpg)

![96c63e696a8ee531bc92c2d3d14d2a9abf27dcde32ed9f2adeda600161a5c1cf.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/96c63e696a8ee531bc92c2d3d14d2a9abf27dcde32ed9f2adeda600161a5c1cf.jpg)

![98676e5392e545c481a06d5531a09a3575c598c69e3fbdffddd7818e5d68c148.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/98676e5392e545c481a06d5531a09a3575c598c69e3fbdffddd7818e5d68c148.jpg)

![9fc6d5096b393aa51df822dfd1741645eb2bae8246ce95ce0a9767c98a1b94e0.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/9fc6d5096b393aa51df822dfd1741645eb2bae8246ce95ce0a9767c98a1b94e0.jpg)

![a166457184c004f97fe44325951029862d784276f831de49f25110c786a7ef67.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/a166457184c004f97fe44325951029862d784276f831de49f25110c786a7ef67.jpg)

![a36b858041b5e9ab7bafa67fdd4fe14d62ed0e5ac2aa15c31cbb2af856238355.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/a36b858041b5e9ab7bafa67fdd4fe14d62ed0e5ac2aa15c31cbb2af856238355.jpg)

![b1ddee7cbdbd51c0fd66b382d79c3a64c41fd0f945eae3ee521eafffb76fba27.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/b1ddee7cbdbd51c0fd66b382d79c3a64c41fd0f945eae3ee521eafffb76fba27.jpg)

![b9590bb9d4d31720c54a72c0b33859a8a4f16bcb423e3c73a2ddc8c63bd11ec6.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/b9590bb9d4d31720c54a72c0b33859a8a4f16bcb423e3c73a2ddc8c63bd11ec6.jpg)

![ec9305004eb618ccb592569b0f31f6d144d7ea0acf16fce6e38ee7abb3129fc0.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/ec9305004eb618ccb592569b0f31f6d144d7ea0acf16fce6e38ee7abb3129fc0.jpg)

![fce57611486c3a049e417baed88a4b3b736dfda94e07f979717a8550835f7e8d.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/fce57611486c3a049e417baed88a4b3b736dfda94e07f979717a8550835f7e8d.jpg)

![fd95e92e9e088f1fd5298f9837bd86d6b9c9f73c492599a9bf0166487dce7929.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/fd95e92e9e088f1fd5298f9837bd86d6b9c9f73c492599a9bf0166487dce7929.jpg)

![fe6bc8481a5655bd8752a8227c404d63e3f8fbb0b940a9dfbde3222248c4899b.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/fe6bc8481a5655bd8752a8227c404d63e3f8fbb0b940a9dfbde3222248c4899b.jpg)

![ff06bd480d2224faab4e524b4d8ae70b3b247b60760639f2a03138f856d3de53.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/images/ff06bd480d2224faab4e524b4d8ae70b3b247b60760639f2a03138f856d3de53.jpg)

### Tables

![2130839bbcfcacd2b5bfe00def7cf9e6b1d0ad6079eafe142d2aabf4f4f7c6fa.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/tables/2130839bbcfcacd2b5bfe00def7cf9e6b1d0ad6079eafe142d2aabf4f4f7c6fa.jpg)

![22d97e36841af1984df3a9be3bf2af7232d4c4233fca122788513b7ad02478f3.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/tables/22d97e36841af1984df3a9be3bf2af7232d4c4233fca122788513b7ad02478f3.jpg)

![67bd4b92c849924251beee20dff53e320463883572b54e026b697630d5b59263.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/tables/67bd4b92c849924251beee20dff53e320463883572b54e026b697630d5b59263.jpg)

![71c0ce08d15f7e80bf4be9a4efb162a3ce7f3be69bd48810fc1bfd9cb3b8b354.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/tables/71c0ce08d15f7e80bf4be9a4efb162a3ce7f3be69bd48810fc1bfd9cb3b8b354.jpg)

![a5df2d7814701fe7ee616f23e9ac4d0042edbed97a843878236102f7030d4f4d.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/tables/a5df2d7814701fe7ee616f23e9ac4d0042edbed97a843878236102f7030d4f4d.jpg)

![bc634dcfb40b77321a36bb6348885df020791bc1ac1b040f735144fdd59e8cc6.jpg](../iclr_results/2297_Sensor-Invariant Tactile Representation/tables/bc634dcfb40b77321a36bb6348885df020791bc1ac1b040f735144fdd59e8cc6.jpg)

## Diffusion Models as Cartoonists: The Curious Case of High Density Regions


### Images

![123dbc0b9e464b033b809c90a87abf3818c51e4ad80af55abdb14a58916cc641.jpg](../iclr_results/2298_Diffusion Models as Cartoonists_ The Curious Case of High Density Regions/images/123dbc0b9e464b033b809c90a87abf3818c51e4ad80af55abdb14a58916cc641.jpg)

![320397487469eb979db1a69aff8040ac421260b033afca70035416acedf46b40.jpg](../iclr_results/2298_Diffusion Models as Cartoonists_ The Curious Case of High Density Regions/images/320397487469eb979db1a69aff8040ac421260b033afca70035416acedf46b40.jpg)

![467d6e870d0d035ef444d26dfb38d6590668d27e6392c52d5ea2587daea216bf.jpg](../iclr_results/2298_Diffusion Models as Cartoonists_ The Curious Case of High Density Regions/images/467d6e870d0d035ef444d26dfb38d6590668d27e6392c52d5ea2587daea216bf.jpg)

![567a02598157fc85fe5db7e18b77e03a768955444336632de854e108a9a4d55a.jpg](../iclr_results/2298_Diffusion Models as Cartoonists_ The Curious Case of High Density Regions/images/567a02598157fc85fe5db7e18b77e03a768955444336632de854e108a9a4d55a.jpg)

![64a3357ef05c1e433bc7ce3f90db547ee8f4982ec0cff1a422df8a29bd9927a8.jpg](../iclr_results/2298_Diffusion Models as Cartoonists_ The Curious Case of High Density Regions/images/64a3357ef05c1e433bc7ce3f90db547ee8f4982ec0cff1a422df8a29bd9927a8.jpg)

![6b6889351128cee56f23b103ad2165ba2f21ce4f32955bec3705967f4dd30163.jpg](../iclr_results/2298_Diffusion Models as Cartoonists_ The Curious Case of High Density Regions/images/6b6889351128cee56f23b103ad2165ba2f21ce4f32955bec3705967f4dd30163.jpg)

![742cbe29e5e692fb1c3250d831aeaf0ff25204afce34fd1d93049e71bcf58fc3.jpg](../iclr_results/2298_Diffusion Models as Cartoonists_ The Curious Case of High Density Regions/images/742cbe29e5e692fb1c3250d831aeaf0ff25204afce34fd1d93049e71bcf58fc3.jpg)

![749ce8b5e86075af0ada876fa1bf55077cf3b830046c3995a711b9b9de3b9927.jpg](../iclr_results/2298_Diffusion Models as Cartoonists_ The Curious Case of High Density Regions/images/749ce8b5e86075af0ada876fa1bf55077cf3b830046c3995a711b9b9de3b9927.jpg)

![7f5246da1080265792461a87b94688374cfcf5d11038ccc8aca707f0d8793a6b.jpg](../iclr_results/2298_Diffusion Models as Cartoonists_ The Curious Case of High Density Regions/images/7f5246da1080265792461a87b94688374cfcf5d11038ccc8aca707f0d8793a6b.jpg)

![9951d01114adc21f98b6e06a177ee164360498307f5d2d4cec4ba5edb6d2a4fd.jpg](../iclr_results/2298_Diffusion Models as Cartoonists_ The Curious Case of High Density Regions/images/9951d01114adc21f98b6e06a177ee164360498307f5d2d4cec4ba5edb6d2a4fd.jpg)

![e8b84765206367e7314bad651770b6ff7e1bb54fb880e098b10fbcfa813abecb.jpg](../iclr_results/2298_Diffusion Models as Cartoonists_ The Curious Case of High Density Regions/images/e8b84765206367e7314bad651770b6ff7e1bb54fb880e098b10fbcfa813abecb.jpg)

![ee785aee97e67287d0dcb0d0fe4abc5928da3fe0c85e20fb9fdb621b77d80705.jpg](../iclr_results/2298_Diffusion Models as Cartoonists_ The Curious Case of High Density Regions/images/ee785aee97e67287d0dcb0d0fe4abc5928da3fe0c85e20fb9fdb621b77d80705.jpg)

### Tables

![3eb07da1bdeb3ce3456f42339490cd89b97087b943cb8a021fd7ec3d19ec5eb7.jpg](../iclr_results/2298_Diffusion Models as Cartoonists_ The Curious Case of High Density Regions/tables/3eb07da1bdeb3ce3456f42339490cd89b97087b943cb8a021fd7ec3d19ec5eb7.jpg)

![6b86233b7819d85d368e26d7301c15d26781290ce6632afb41e82db19e0e9b11.jpg](../iclr_results/2298_Diffusion Models as Cartoonists_ The Curious Case of High Density Regions/tables/6b86233b7819d85d368e26d7301c15d26781290ce6632afb41e82db19e0e9b11.jpg)

![d1aac31967303baad1d24fa874e3d00bfa6e676fe30b92b4c30d98ea4cf49dfe.jpg](../iclr_results/2298_Diffusion Models as Cartoonists_ The Curious Case of High Density Regions/tables/d1aac31967303baad1d24fa874e3d00bfa6e676fe30b92b4c30d98ea4cf49dfe.jpg)

## 3DMolFormer: A Dual-channel Framework for Structure-based Drug Discovery


### Images

![1985eb719fc21aac4fb049ce302c83095ea68b69d68a835ab2ca3777fe501ab0.jpg](../iclr_results/2299_3DMolFormer_ A Dual-channel Framework for Structure-based Drug Discovery/images/1985eb719fc21aac4fb049ce302c83095ea68b69d68a835ab2ca3777fe501ab0.jpg)

![1e0dba96a949f36c348b69ef2c246ebf13dab6ccf8f5e9d0083caf64b631ad32.jpg](../iclr_results/2299_3DMolFormer_ A Dual-channel Framework for Structure-based Drug Discovery/images/1e0dba96a949f36c348b69ef2c246ebf13dab6ccf8f5e9d0083caf64b631ad32.jpg)

![7270355ef24bd02c396a87b4789ffeb5cf3ad56b1917fb18a011d2355354bcfc.jpg](../iclr_results/2299_3DMolFormer_ A Dual-channel Framework for Structure-based Drug Discovery/images/7270355ef24bd02c396a87b4789ffeb5cf3ad56b1917fb18a011d2355354bcfc.jpg)

![d8fa46b1453e337a3697c00ad623152d21f72700dbd5b8c3bbc00f990acaa58d.jpg](../iclr_results/2299_3DMolFormer_ A Dual-channel Framework for Structure-based Drug Discovery/images/d8fa46b1453e337a3697c00ad623152d21f72700dbd5b8c3bbc00f990acaa58d.jpg)

![debb95c8fbd5a57f700cfb3c646747e588acdbdf202ce6c6c3ba703b0d925a34.jpg](../iclr_results/2299_3DMolFormer_ A Dual-channel Framework for Structure-based Drug Discovery/images/debb95c8fbd5a57f700cfb3c646747e588acdbdf202ce6c6c3ba703b0d925a34.jpg)

### Tables

![293c253786b487f05298088f16f26592987c6c130ef8709fd457a4ab2fe0acc2.jpg](../iclr_results/2299_3DMolFormer_ A Dual-channel Framework for Structure-based Drug Discovery/tables/293c253786b487f05298088f16f26592987c6c130ef8709fd457a4ab2fe0acc2.jpg)

![3f0508662e4d419568bf1ff50e6ef7fc26704cd63fb795c08681290408683cd4.jpg](../iclr_results/2299_3DMolFormer_ A Dual-channel Framework for Structure-based Drug Discovery/tables/3f0508662e4d419568bf1ff50e6ef7fc26704cd63fb795c08681290408683cd4.jpg)

![55565fc2d49d0fec78a9f888b80b019722b46d571e426652b38688dd36f28509.jpg](../iclr_results/2299_3DMolFormer_ A Dual-channel Framework for Structure-based Drug Discovery/tables/55565fc2d49d0fec78a9f888b80b019722b46d571e426652b38688dd36f28509.jpg)

![55c200c8c5d198fb4b689c3a34dce8639f4f144e55554eecf260dd3845298132.jpg](../iclr_results/2299_3DMolFormer_ A Dual-channel Framework for Structure-based Drug Discovery/tables/55c200c8c5d198fb4b689c3a34dce8639f4f144e55554eecf260dd3845298132.jpg)

![63de71c13d4c2e7fe138532652736cea2ea98e0145477997d40e1cd01df027c7.jpg](../iclr_results/2299_3DMolFormer_ A Dual-channel Framework for Structure-based Drug Discovery/tables/63de71c13d4c2e7fe138532652736cea2ea98e0145477997d40e1cd01df027c7.jpg)

![91703dd1723756f9af3c83964c5f85e78905f8e489f385909c425392284049ab.jpg](../iclr_results/2299_3DMolFormer_ A Dual-channel Framework for Structure-based Drug Discovery/tables/91703dd1723756f9af3c83964c5f85e78905f8e489f385909c425392284049ab.jpg)

![9bbf208d96e096559db2b6144b4fa216bbb38a1962e823f05d47bce01cbc4988.jpg](../iclr_results/2299_3DMolFormer_ A Dual-channel Framework for Structure-based Drug Discovery/tables/9bbf208d96e096559db2b6144b4fa216bbb38a1962e823f05d47bce01cbc4988.jpg)

![9df1ee95dfa1023de6351650838372f3c6604f86da8aab302341d07c42e2704f.jpg](../iclr_results/2299_3DMolFormer_ A Dual-channel Framework for Structure-based Drug Discovery/tables/9df1ee95dfa1023de6351650838372f3c6604f86da8aab302341d07c42e2704f.jpg)

![a72a54d2cece92cc5b5c63adb941cd69cc832622ae4cdf266a1b0689e84c4185.jpg](../iclr_results/2299_3DMolFormer_ A Dual-channel Framework for Structure-based Drug Discovery/tables/a72a54d2cece92cc5b5c63adb941cd69cc832622ae4cdf266a1b0689e84c4185.jpg)

![ae5864e36c6f27d59aac935083fab1df61c01fe2432d9a9ab430df949f5ffd0d.jpg](../iclr_results/2299_3DMolFormer_ A Dual-channel Framework for Structure-based Drug Discovery/tables/ae5864e36c6f27d59aac935083fab1df61c01fe2432d9a9ab430df949f5ffd0d.jpg)

![ceb8e08059434d24b485ff399da86503ed6d0a59cb65b0bb8b2d2e320f897d2e.jpg](../iclr_results/2299_3DMolFormer_ A Dual-channel Framework for Structure-based Drug Discovery/tables/ceb8e08059434d24b485ff399da86503ed6d0a59cb65b0bb8b2d2e320f897d2e.jpg)

## Optimality and Adaptivity of Deep Neural Features for Instrumental Variable Regression


### Images

![fd7827eec8dc6da4ed098e79116f2621a12972d8a2daa22ed82005e4dded2792.jpg](../iclr_results/2300_Optimality and Adaptivity of Deep Neural Features for Instrumental Variable Regression/images/fd7827eec8dc6da4ed098e79116f2621a12972d8a2daa22ed82005e4dded2792.jpg)

### Tables

![6192f339b97a68e41ca3bb55754ef01844de719724cef4f00f834fd7b016698f.jpg](../iclr_results/2300_Optimality and Adaptivity of Deep Neural Features for Instrumental Variable Regression/tables/6192f339b97a68e41ca3bb55754ef01844de719724cef4f00f834fd7b016698f.jpg)

![6b0ddc396e972416c8ff277bd8d4f51ce6172cc59d835903713947264ea73a01.jpg](../iclr_results/2300_Optimality and Adaptivity of Deep Neural Features for Instrumental Variable Regression/tables/6b0ddc396e972416c8ff277bd8d4f51ce6172cc59d835903713947264ea73a01.jpg)

## Shape as Line Segments: Accurate and Flexible Implicit Surface Representation


### Images

![206c52faf4b22b2311a7e7f68b9b2137fc09bbcbb904d8a3b2b8b1beb4fa8472.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/images/206c52faf4b22b2311a7e7f68b9b2137fc09bbcbb904d8a3b2b8b1beb4fa8472.jpg)

![329b89de1d1e5e9498be05f112d005638b4d7124d62c56f45e619f65886ecde3.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/images/329b89de1d1e5e9498be05f112d005638b4d7124d62c56f45e619f65886ecde3.jpg)

![4225f5c822f5e247af0ada579a5cb1067423524c20589e9c34f77c934ab29ac9.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/images/4225f5c822f5e247af0ada579a5cb1067423524c20589e9c34f77c934ab29ac9.jpg)

![44dceb0de8374dc2d537906d57680d137b080e38cbb3cb34911609877f6671ed.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/images/44dceb0de8374dc2d537906d57680d137b080e38cbb3cb34911609877f6671ed.jpg)

![65abac419e2d4349d8d548a8cb1e93490397d046c133f03266d260f095a09d67.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/images/65abac419e2d4349d8d548a8cb1e93490397d046c133f03266d260f095a09d67.jpg)

![7001f57d96b7f4b7ff75d988f223e5f51732fcca89d86d616573563351f61741.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/images/7001f57d96b7f4b7ff75d988f223e5f51732fcca89d86d616573563351f61741.jpg)

![942c53fc43212456136b90da708044b05589110a29be8d0c1bb64326b7ad10f1.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/images/942c53fc43212456136b90da708044b05589110a29be8d0c1bb64326b7ad10f1.jpg)

![9e7565f37ef93f8a8e9d1d8b64bf018fbe727f9dd9c6629a67611b32b197965e.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/images/9e7565f37ef93f8a8e9d1d8b64bf018fbe727f9dd9c6629a67611b32b197965e.jpg)

![a29fd54fe9b38ec5990f77f7009401623379b8e6683c88b48bf3de5b0ceb1847.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/images/a29fd54fe9b38ec5990f77f7009401623379b8e6683c88b48bf3de5b0ceb1847.jpg)

![a79e7869522ba39b935cfbca9cb51a44930228581563b537b1614cd6c994b887.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/images/a79e7869522ba39b935cfbca9cb51a44930228581563b537b1614cd6c994b887.jpg)

![b8c7cb8057b54b8f3e2cab8c343016da955857455ff3280df701323061010190.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/images/b8c7cb8057b54b8f3e2cab8c343016da955857455ff3280df701323061010190.jpg)

![c88be14419b5452306aaa084ca40e8b76f66e8ce888eb9683399972ee1c4d086.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/images/c88be14419b5452306aaa084ca40e8b76f66e8ce888eb9683399972ee1c4d086.jpg)

![ce56b8a6554b4fa8c487d69d89037c30b960d83e6ef4284be21412209f7b8b4b.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/images/ce56b8a6554b4fa8c487d69d89037c30b960d83e6ef4284be21412209f7b8b4b.jpg)

![d75b53eedcfdb35962a7d49d7e62f5263146e893a5f2a1904cf53613cae14250.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/images/d75b53eedcfdb35962a7d49d7e62f5263146e893a5f2a1904cf53613cae14250.jpg)

![d8df68305669018ca940b3f90f9de9d1e84e6021a2c8db5f0ef745f7d39a6390.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/images/d8df68305669018ca940b3f90f9de9d1e84e6021a2c8db5f0ef745f7d39a6390.jpg)

![d9f58e5494f96d08b235907e76533dc0f9c43213d283cc56ce4c6a4079119291.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/images/d9f58e5494f96d08b235907e76533dc0f9c43213d283cc56ce4c6a4079119291.jpg)

![e4ab2f067372b887c07d028bf4bfe68bf67ca15757c5c53ce0105bd2f1b92f26.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/images/e4ab2f067372b887c07d028bf4bfe68bf67ca15757c5c53ce0105bd2f1b92f26.jpg)

![e7d5bd196f732d4f5d7218047ceb7a195f552ddea86c5a6af0c9a7841b8dd0f1.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/images/e7d5bd196f732d4f5d7218047ceb7a195f552ddea86c5a6af0c9a7841b8dd0f1.jpg)

![eae500740eaf22e3873dc63043abfd3b23bc35f347634061edb4f41478cd312b.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/images/eae500740eaf22e3873dc63043abfd3b23bc35f347634061edb4f41478cd312b.jpg)

![f7758460149f8df2b9a566c23da8730161bbef3e15ae47b499b016135163c2c0.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/images/f7758460149f8df2b9a566c23da8730161bbef3e15ae47b499b016135163c2c0.jpg)

### Tables

![07bb420f6ebc9b6ef644beb358b986205412b550efac480f3de1cf9f33841d3d.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/tables/07bb420f6ebc9b6ef644beb358b986205412b550efac480f3de1cf9f33841d3d.jpg)

![18489c315ce1a9f607b02b147c3279181a02cf7956e4221f801f170bfae8c13e.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/tables/18489c315ce1a9f607b02b147c3279181a02cf7956e4221f801f170bfae8c13e.jpg)

![2c3682bb0e7ac597ddb7455aff1c8aa43ffb3cbcf433a9e3475de6d408ad784e.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/tables/2c3682bb0e7ac597ddb7455aff1c8aa43ffb3cbcf433a9e3475de6d408ad784e.jpg)

![56d5d000d434f7acb81606875377499dd772853a38a82f8c6c97b2511c9e3734.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/tables/56d5d000d434f7acb81606875377499dd772853a38a82f8c6c97b2511c9e3734.jpg)

![6846448b92262a2dbd1e7e0b62fe6f7f8ac7bd9a0f8da15f9487a543142b56d5.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/tables/6846448b92262a2dbd1e7e0b62fe6f7f8ac7bd9a0f8da15f9487a543142b56d5.jpg)

![83b50d5659d4ae12b27690a3b649436a2b2416979eaeb62a83681a8ca2e93528.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/tables/83b50d5659d4ae12b27690a3b649436a2b2416979eaeb62a83681a8ca2e93528.jpg)

![8f8e6b3805ec2f1287ab2d5c56fc2a354b9a0ff0d212ba0740a5b9c827c6e1fc.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/tables/8f8e6b3805ec2f1287ab2d5c56fc2a354b9a0ff0d212ba0740a5b9c827c6e1fc.jpg)

![9dbc7724be6c75b17ce4ddc2317fd3baee7430df5e7dd033aa59a7489eba7810.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/tables/9dbc7724be6c75b17ce4ddc2317fd3baee7430df5e7dd033aa59a7489eba7810.jpg)

![a0eccb52f47cca1a2b8dc9825dd41d366c4a7367baefda8858f7ea1046e930f4.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/tables/a0eccb52f47cca1a2b8dc9825dd41d366c4a7367baefda8858f7ea1046e930f4.jpg)

![a1e6b2b547f4244d561f9e157be011bd70eb0a585884d5f9335583636b6477e1.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/tables/a1e6b2b547f4244d561f9e157be011bd70eb0a585884d5f9335583636b6477e1.jpg)

![ae2e11c71637b55605956e0af3614cb8beed8954b26fc3d5d39a0349bccd98c9.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/tables/ae2e11c71637b55605956e0af3614cb8beed8954b26fc3d5d39a0349bccd98c9.jpg)

![f1cdd2fcf576956dccb149a9ddd8a0580de26b2a9d29f80e1dadea5c8d5a1e66.jpg](../iclr_results/2301_Shape as Line Segments_ Accurate and Flexible Implicit Surface Representation/tables/f1cdd2fcf576956dccb149a9ddd8a0580de26b2a9d29f80e1dadea5c8d5a1e66.jpg)

## HOPE for a Robust Parameterization of Long-memory State Space Models


### Images

![152f67253a26bb7009fb667973d2cdc01e8b68e299d5d33f3ceac695edb80f66.jpg](../iclr_results/2302_HOPE for a Robust Parameterization of Long-memory State Space Models/images/152f67253a26bb7009fb667973d2cdc01e8b68e299d5d33f3ceac695edb80f66.jpg)

![15b4e793b9b80409932832687903b5bc47e9192a25f51242a04fa293308beef8.jpg](../iclr_results/2302_HOPE for a Robust Parameterization of Long-memory State Space Models/images/15b4e793b9b80409932832687903b5bc47e9192a25f51242a04fa293308beef8.jpg)

![2d0f452d0e828b46ddca76be76ee477bb988985a2e47f00a8d201800370155cf.jpg](../iclr_results/2302_HOPE for a Robust Parameterization of Long-memory State Space Models/images/2d0f452d0e828b46ddca76be76ee477bb988985a2e47f00a8d201800370155cf.jpg)

![3e77300176c7eb764dc251ec028203632191a9ef7f9f9285e1c9001a6eb6dcd1.jpg](../iclr_results/2302_HOPE for a Robust Parameterization of Long-memory State Space Models/images/3e77300176c7eb764dc251ec028203632191a9ef7f9f9285e1c9001a6eb6dcd1.jpg)

![6480acbf0d982b2ae84e5c126873f8ea402393b762181fad65576a2b87179ce2.jpg](../iclr_results/2302_HOPE for a Robust Parameterization of Long-memory State Space Models/images/6480acbf0d982b2ae84e5c126873f8ea402393b762181fad65576a2b87179ce2.jpg)

![80c9ac6d05f13adfe84e5f77f2eb4c9c4764f0ca715c4ba0d82a1be5714b4a24.jpg](../iclr_results/2302_HOPE for a Robust Parameterization of Long-memory State Space Models/images/80c9ac6d05f13adfe84e5f77f2eb4c9c4764f0ca715c4ba0d82a1be5714b4a24.jpg)

![848430abf10b8a776d1bdbc182efefad668612358c42e3c2b6e36849ba268189.jpg](../iclr_results/2302_HOPE for a Robust Parameterization of Long-memory State Space Models/images/848430abf10b8a776d1bdbc182efefad668612358c42e3c2b6e36849ba268189.jpg)

![8c7c79491ff37d9ad2c8b915f273cc3c1ae998674d342b0ba8e3efd549f03d30.jpg](../iclr_results/2302_HOPE for a Robust Parameterization of Long-memory State Space Models/images/8c7c79491ff37d9ad2c8b915f273cc3c1ae998674d342b0ba8e3efd549f03d30.jpg)

![93358e49c39f207231eb897bae182b6c0fae9e66f4adf1145799ff8259f06d05.jpg](../iclr_results/2302_HOPE for a Robust Parameterization of Long-memory State Space Models/images/93358e49c39f207231eb897bae182b6c0fae9e66f4adf1145799ff8259f06d05.jpg)

![cdf942b30f69fdf89b868b621b6965c36f60cf164f173cda8b8c9179ebcad83b.jpg](../iclr_results/2302_HOPE for a Robust Parameterization of Long-memory State Space Models/images/cdf942b30f69fdf89b868b621b6965c36f60cf164f173cda8b8c9179ebcad83b.jpg)

![e63d451d138b8e67b3beb3704e5d8955476f62f5ea11ca7b40d7633ce3fb0203.jpg](../iclr_results/2302_HOPE for a Robust Parameterization of Long-memory State Space Models/images/e63d451d138b8e67b3beb3704e5d8955476f62f5ea11ca7b40d7633ce3fb0203.jpg)

![f0afa124570064a072cb96469d782fbfa6df926dceec9509948a6b62dc9df0b7.jpg](../iclr_results/2302_HOPE for a Robust Parameterization of Long-memory State Space Models/images/f0afa124570064a072cb96469d782fbfa6df926dceec9509948a6b62dc9df0b7.jpg)

### Tables

![19e5eb88bcb1b8b0eceb0ba28ed4bad78a062a2cee445f4875f790cd1123b542.jpg](../iclr_results/2302_HOPE for a Robust Parameterization of Long-memory State Space Models/tables/19e5eb88bcb1b8b0eceb0ba28ed4bad78a062a2cee445f4875f790cd1123b542.jpg)

![1fd042583dfa3203e0803ca197ac6da17357689d43c903178c6399fa4d4b0ef5.jpg](../iclr_results/2302_HOPE for a Robust Parameterization of Long-memory State Space Models/tables/1fd042583dfa3203e0803ca197ac6da17357689d43c903178c6399fa4d4b0ef5.jpg)

![8d8d4cd55fa78b0196096844369b7a14199829a1e848fc761af81630752d1322.jpg](../iclr_results/2302_HOPE for a Robust Parameterization of Long-memory State Space Models/tables/8d8d4cd55fa78b0196096844369b7a14199829a1e848fc761af81630752d1322.jpg)

![9c079ccb5b2e2df80d027066f028951d26d5a95160afdde617ec39b62f448753.jpg](../iclr_results/2302_HOPE for a Robust Parameterization of Long-memory State Space Models/tables/9c079ccb5b2e2df80d027066f028951d26d5a95160afdde617ec39b62f448753.jpg)

## Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning


### Images

![1e237f14cf1d15a9db8be0ab5daaab579e3ef557684cc4e1f878e50d48fe3f9b.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/images/1e237f14cf1d15a9db8be0ab5daaab579e3ef557684cc4e1f878e50d48fe3f9b.jpg)

![38cd91434b6970a2149622df387acfc666356577581b41fd36b17b5b0f814314.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/images/38cd91434b6970a2149622df387acfc666356577581b41fd36b17b5b0f814314.jpg)

![3af1513f20857a779f7ea768cbabec15ade976740be016ff1921426e2a75592d.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/images/3af1513f20857a779f7ea768cbabec15ade976740be016ff1921426e2a75592d.jpg)

![50aeeb5bb7071ff1127f2555858745012b18883d55a362a9b84cd56164e7652a.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/images/50aeeb5bb7071ff1127f2555858745012b18883d55a362a9b84cd56164e7652a.jpg)

![9d599b543b48bf2c49eaec273b5236d0c447f8e68a6b30183dfa67dcfd308cc1.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/images/9d599b543b48bf2c49eaec273b5236d0c447f8e68a6b30183dfa67dcfd308cc1.jpg)

![a22b7c69cb8089ccab7c4ac132dafeb0107af27531956385ce55f0282971f270.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/images/a22b7c69cb8089ccab7c4ac132dafeb0107af27531956385ce55f0282971f270.jpg)

![a33413a8e607f3c674c20595966db4ed478d3b2b606cf61e456e57b0571a2ead.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/images/a33413a8e607f3c674c20595966db4ed478d3b2b606cf61e456e57b0571a2ead.jpg)

![a836c443e2a6908e9febd75355632effc8a0504a1cf2cec420301a762c353c1b.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/images/a836c443e2a6908e9febd75355632effc8a0504a1cf2cec420301a762c353c1b.jpg)

![b2a4dda820b2c263ac64a41a37d4ce735e381512699929fa4fa04af5909f17e4.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/images/b2a4dda820b2c263ac64a41a37d4ce735e381512699929fa4fa04af5909f17e4.jpg)

![cb8c8f66bbd58537189a833928f3379ccbc45a9b53fc4394d294fe01c68e7c1b.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/images/cb8c8f66bbd58537189a833928f3379ccbc45a9b53fc4394d294fe01c68e7c1b.jpg)

![eeffe35d5d803b70424ac1145b4059f90519576e4efa7dbf850e37dfff063828.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/images/eeffe35d5d803b70424ac1145b4059f90519576e4efa7dbf850e37dfff063828.jpg)

![f808306911baf6e6f986ef5b0df31e3bbcd56cdccc12ef72a30fe8a742e5246f.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/images/f808306911baf6e6f986ef5b0df31e3bbcd56cdccc12ef72a30fe8a742e5246f.jpg)

### Tables

![02a8abff0ab36ddaa3a4ffb67f7d43b1a854f8148b61356fd74474c176a88d59.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/02a8abff0ab36ddaa3a4ffb67f7d43b1a854f8148b61356fd74474c176a88d59.jpg)

![1d1dec18a73e46a2a4519b2b726e83c2ae7dd8d62c182f1ef7d81019c19eaf8a.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/1d1dec18a73e46a2a4519b2b726e83c2ae7dd8d62c182f1ef7d81019c19eaf8a.jpg)

![1e046fc314210cb603c655e43d6699ad92c81b45264fdb44b2394f010d067a93.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/1e046fc314210cb603c655e43d6699ad92c81b45264fdb44b2394f010d067a93.jpg)

![1e5550954be6d9dd14661e1d0a8af8ad3c7f9d21143fa94952654baa0ab083ab.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/1e5550954be6d9dd14661e1d0a8af8ad3c7f9d21143fa94952654baa0ab083ab.jpg)

![43f95761ed42c03d195ec3cdba10e78fc8450ec39d11e76252e1753e705cc948.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/43f95761ed42c03d195ec3cdba10e78fc8450ec39d11e76252e1753e705cc948.jpg)

![50cbc1f67fe7530b3afd1113543514dfd70332fb423f80680a6fb0fae60bba9a.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/50cbc1f67fe7530b3afd1113543514dfd70332fb423f80680a6fb0fae60bba9a.jpg)

![5eba3667f7e76d3cee1c7b9fe6c555222ffbd2fb7bd95aadca112c9ed75719db.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/5eba3667f7e76d3cee1c7b9fe6c555222ffbd2fb7bd95aadca112c9ed75719db.jpg)

![602a7913fae1b4b2e853cee2da110a8a9856238582b781797fd766103a94bd3b.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/602a7913fae1b4b2e853cee2da110a8a9856238582b781797fd766103a94bd3b.jpg)

![8f62c17830f8b9ec278a017de8cd9f36436567f7e3d32f074c23f4ee84fbffd2.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/8f62c17830f8b9ec278a017de8cd9f36436567f7e3d32f074c23f4ee84fbffd2.jpg)

![915c8dcb32217e5ba11fca9942479231dcbff668fc269e52e416186e1c6c039d.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/915c8dcb32217e5ba11fca9942479231dcbff668fc269e52e416186e1c6c039d.jpg)

![9270a8d05481522cfb3e5e8c32c262c02325bcdcc9c2f50d5e422d905ebe9623.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/9270a8d05481522cfb3e5e8c32c262c02325bcdcc9c2f50d5e422d905ebe9623.jpg)

![a851fb462877fe87dd48951897a40a01872a614fdf219dc6e76034482bf65738.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/a851fb462877fe87dd48951897a40a01872a614fdf219dc6e76034482bf65738.jpg)

![b0b7c4e04c4a03920959273b9a9e3b456d7f44a74b9f46759c98ae9ff01216a5.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/b0b7c4e04c4a03920959273b9a9e3b456d7f44a74b9f46759c98ae9ff01216a5.jpg)

![be1262154782cdd354dcf4a5ade84efde77c9654f7007992d08c2051afb54b69.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/be1262154782cdd354dcf4a5ade84efde77c9654f7007992d08c2051afb54b69.jpg)

![be7bf6e54e962dbbd62ea069cad3167c69eb900ab96f37ba996a7b0e04b245d2.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/be7bf6e54e962dbbd62ea069cad3167c69eb900ab96f37ba996a7b0e04b245d2.jpg)

![cea160d1795cd3cde2afb6f4e46336eeba01f08e4325cbdceb033f11926fec4c.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/cea160d1795cd3cde2afb6f4e46336eeba01f08e4325cbdceb033f11926fec4c.jpg)

![d1f6e93623dab6f978af2e9c253179dee013f692e04d867fe9990baf8ff87b43.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/d1f6e93623dab6f978af2e9c253179dee013f692e04d867fe9990baf8ff87b43.jpg)

![db673873840ce9ba12bcd472d3d983c7a7427c0f7f3c5fb8a1ed17f44c3dafd2.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/db673873840ce9ba12bcd472d3d983c7a7427c0f7f3c5fb8a1ed17f44c3dafd2.jpg)

![dd707d637236eed1d5d87fbbe121ba553700c68a8d8c3143948953da08179720.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/dd707d637236eed1d5d87fbbe121ba553700c68a8d8c3143948953da08179720.jpg)

![df6eac01b052a6c0e9e65bacc631c81daf48a46021521eaa59cd2a18c9604d00.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/df6eac01b052a6c0e9e65bacc631c81daf48a46021521eaa59cd2a18c9604d00.jpg)

![efbefa95ae203905f60eea1e6a6a93daa6c48f9b9397c59971b7c3ab790945aa.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/efbefa95ae203905f60eea1e6a6a93daa6c48f9b9397c59971b7c3ab790945aa.jpg)

![f339e7c6bbb5c395d092c17bdb4916cf5288f37173a068dc5f00dd71496fede8.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/f339e7c6bbb5c395d092c17bdb4916cf5288f37173a068dc5f00dd71496fede8.jpg)

![f7addb18545af241f8f798f5a93d862ab87d1e4f5cea7dc78228101df7966a7f.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/f7addb18545af241f8f798f5a93d862ab87d1e4f5cea7dc78228101df7966a7f.jpg)

![fbbcc33674ce1cb35fe63c247af290c71b6da12ad8ef71942352d7c224f07b8d.jpg](../iclr_results/2303_Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning/tables/fbbcc33674ce1cb35fe63c247af290c71b6da12ad8ef71942352d7c224f07b8d.jpg)

## Kronecker Mask and Interpretive Prompts are Language-Action Video Learners


### Images

![0a2dbf5fc86907277ca55dca898f6aa2bd96a78d29e90b293f293dc76cc0b5e4.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/images/0a2dbf5fc86907277ca55dca898f6aa2bd96a78d29e90b293f293dc76cc0b5e4.jpg)

![13d2f788884dabbcdf28aaec2ce7ea6e409c4b461114e0db4be5ed2e4a6d3243.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/images/13d2f788884dabbcdf28aaec2ce7ea6e409c4b461114e0db4be5ed2e4a6d3243.jpg)

![14955875b60300e038b9de19290bb84a4620dd5cd88fbf64eb114f99d4683942.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/images/14955875b60300e038b9de19290bb84a4620dd5cd88fbf64eb114f99d4683942.jpg)

![39411015ecc548dd9450cd2c0b98316d3076d4667f710678d8d4fa058fcf0afc.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/images/39411015ecc548dd9450cd2c0b98316d3076d4667f710678d8d4fa058fcf0afc.jpg)

![3d9babe96978dafa34ffdd9a4cd396f2d7ae163fcb458f9a648edc62e83b8f39.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/images/3d9babe96978dafa34ffdd9a4cd396f2d7ae163fcb458f9a648edc62e83b8f39.jpg)

![66c8d5ccdc6dc2046f7713e2f81887f8e49a8cf8de89a4d9341be4222a466dfe.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/images/66c8d5ccdc6dc2046f7713e2f81887f8e49a8cf8de89a4d9341be4222a466dfe.jpg)

![6f2eb05b1509b147543cf635cf3b23a8bb8f7967b1515e5f24b6b854e0ccb0a0.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/images/6f2eb05b1509b147543cf635cf3b23a8bb8f7967b1515e5f24b6b854e0ccb0a0.jpg)

![7c33114f67ec6ed1cdfea378d7ca6328789140312b3eeeaa267376940d063cb0.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/images/7c33114f67ec6ed1cdfea378d7ca6328789140312b3eeeaa267376940d063cb0.jpg)

![9f25d66abe75a2eddea91077efb22d877e1b2324791c5c2444ae60c119c6cc65.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/images/9f25d66abe75a2eddea91077efb22d877e1b2324791c5c2444ae60c119c6cc65.jpg)

![ba7140bb71235545c2c0bada4f926d9a88b3637b1dcb0e8f02eefcc5d257065a.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/images/ba7140bb71235545c2c0bada4f926d9a88b3637b1dcb0e8f02eefcc5d257065a.jpg)

![c20b240f2dc8f2511487c7def355e5f0601629f1855ad7a711dba2b568cb428c.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/images/c20b240f2dc8f2511487c7def355e5f0601629f1855ad7a711dba2b568cb428c.jpg)

![c66ec8d24b79c6f97da2beb92f8cb4a29d09f4cead72bd79fa26cb118ced219e.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/images/c66ec8d24b79c6f97da2beb92f8cb4a29d09f4cead72bd79fa26cb118ced219e.jpg)

![cca71af4a524dd0072f6818cf2a3d39c2e2f8c9f43ed081574690e23f6bd441d.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/images/cca71af4a524dd0072f6818cf2a3d39c2e2f8c9f43ed081574690e23f6bd441d.jpg)

![dceea55c1da45bab941fee4af0ff138c85c956a048bd10b19fef6fd23e3070ff.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/images/dceea55c1da45bab941fee4af0ff138c85c956a048bd10b19fef6fd23e3070ff.jpg)

![fb6cb984cbdc415290f04f45fea7d8c39cd0091379b59e989a847ba9314eeb83.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/images/fb6cb984cbdc415290f04f45fea7d8c39cd0091379b59e989a847ba9314eeb83.jpg)

### Tables

![08dfd9273137b93ec18d9f5c798151d9290a74a6ed0c41dc23b544c4ec5c6cf0.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/tables/08dfd9273137b93ec18d9f5c798151d9290a74a6ed0c41dc23b544c4ec5c6cf0.jpg)

![0aa7aaccbcca83d14cc292e604ddb7bf8b31b03b0115fef941efbef3df32e220.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/tables/0aa7aaccbcca83d14cc292e604ddb7bf8b31b03b0115fef941efbef3df32e220.jpg)

![0c22d03cc4dbcf33eb354ed35f6487cb532bc2692048e66660e0594940f55fdc.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/tables/0c22d03cc4dbcf33eb354ed35f6487cb532bc2692048e66660e0594940f55fdc.jpg)

![296bd98d45a4112b119363b0abfd2d2c5133f2678c84136e82d446d5b2304c1e.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/tables/296bd98d45a4112b119363b0abfd2d2c5133f2678c84136e82d446d5b2304c1e.jpg)

![2e797333d2433b3a8300cde4526bf00ea7684484a812628c889c70df112c4e26.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/tables/2e797333d2433b3a8300cde4526bf00ea7684484a812628c889c70df112c4e26.jpg)

![304a388e21f934c4f2d1ec4e166b7103481d476b1cf0699fbb3e3c935015d2f0.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/tables/304a388e21f934c4f2d1ec4e166b7103481d476b1cf0699fbb3e3c935015d2f0.jpg)

![488bb0fd2188cc7ce5797ed1cf061fc0ff38d73ca71286db833c39fb2e32ae98.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/tables/488bb0fd2188cc7ce5797ed1cf061fc0ff38d73ca71286db833c39fb2e32ae98.jpg)

![825887c85041e841718806f86244f7e9e178edf44403c033c0928e7c5760657c.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/tables/825887c85041e841718806f86244f7e9e178edf44403c033c0928e7c5760657c.jpg)

![914e0a7ac441fd81440b9ff8839ec6a78de096e94fd5eae3990b0281a4ce9c1b.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/tables/914e0a7ac441fd81440b9ff8839ec6a78de096e94fd5eae3990b0281a4ce9c1b.jpg)

![bcea747d3f3e6cecdc974e7282271bb2c343115839a9a47effbfcb6014a99686.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/tables/bcea747d3f3e6cecdc974e7282271bb2c343115839a9a47effbfcb6014a99686.jpg)

![c303954a3724d35f13c1decfc8531f27a8f32bb541cbfeb5127a872689f135ca.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/tables/c303954a3724d35f13c1decfc8531f27a8f32bb541cbfeb5127a872689f135ca.jpg)

![c66036b4afb90b078b6585e7a531f26ad01bca03ef0f8577f5f7f04fb9b5a24f.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/tables/c66036b4afb90b078b6585e7a531f26ad01bca03ef0f8577f5f7f04fb9b5a24f.jpg)

![d38a3fc009006703a11deb150d35a50e51a6ace21503e5af4ed33296ed6f5f24.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/tables/d38a3fc009006703a11deb150d35a50e51a6ace21503e5af4ed33296ed6f5f24.jpg)

![f5f465b502eb5be637b55ee8ad12ab4d9be18bc1dad5e59e14ab075f9064cf02.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/tables/f5f465b502eb5be637b55ee8ad12ab4d9be18bc1dad5e59e14ab075f9064cf02.jpg)

![ff1198dc7a654e163d6c974419c0c22c92c76a16442736da1008d0c573e9d6d4.jpg](../iclr_results/2304_Kronecker Mask and Interpretive Prompts are Language-Action Video Learners/tables/ff1198dc7a654e163d6c974419c0c22c92c76a16442736da1008d0c573e9d6d4.jpg)

## Large Language Models Often Say One Thing and Do Another


### Images

![41c4964d63c87ddc2ba5c1f8b9ac44d433602c710a8efe43b46aa58fbbf85e3f.jpg](../iclr_results/2305_Large Language Models Often Say One Thing and Do Another/images/41c4964d63c87ddc2ba5c1f8b9ac44d433602c710a8efe43b46aa58fbbf85e3f.jpg)

![4414a10bff4a663623b8e6cd54e3926330d1285224bfe237dccb0e620a01b331.jpg](../iclr_results/2305_Large Language Models Often Say One Thing and Do Another/images/4414a10bff4a663623b8e6cd54e3926330d1285224bfe237dccb0e620a01b331.jpg)

![571ab018aed852e800093503950058648f3fbb625ff07669139c9380fd40a73c.jpg](../iclr_results/2305_Large Language Models Often Say One Thing and Do Another/images/571ab018aed852e800093503950058648f3fbb625ff07669139c9380fd40a73c.jpg)

![61f2febdb5ed4eed730a107327af890bdd6984402428690606b4ae3ee394f033.jpg](../iclr_results/2305_Large Language Models Often Say One Thing and Do Another/images/61f2febdb5ed4eed730a107327af890bdd6984402428690606b4ae3ee394f033.jpg)

![64384676ab8f5bbad850f3ca897dfe7f4b3d159cc4338bd38cbc13f5f0be9743.jpg](../iclr_results/2305_Large Language Models Often Say One Thing and Do Another/images/64384676ab8f5bbad850f3ca897dfe7f4b3d159cc4338bd38cbc13f5f0be9743.jpg)

![ce16694b22ee11071155cb0a92eac8f47ccb4cc3ce04faaa8c0691ef72fc93bd.jpg](../iclr_results/2305_Large Language Models Often Say One Thing and Do Another/images/ce16694b22ee11071155cb0a92eac8f47ccb4cc3ce04faaa8c0691ef72fc93bd.jpg)

![d357ec06dc3579f59c6f1de13b1d499ca6698c6c47b21faa160d04cef8562a44.jpg](../iclr_results/2305_Large Language Models Often Say One Thing and Do Another/images/d357ec06dc3579f59c6f1de13b1d499ca6698c6c47b21faa160d04cef8562a44.jpg)

![d9301e51f67052b2469756712111de59afd1bdc27df6fb8a0788fe627684b459.jpg](../iclr_results/2305_Large Language Models Often Say One Thing and Do Another/images/d9301e51f67052b2469756712111de59afd1bdc27df6fb8a0788fe627684b459.jpg)

![e253d2ed5af7fd29a33023be92a31550b9dee29de133a796a45fd91a9bbf1162.jpg](../iclr_results/2305_Large Language Models Often Say One Thing and Do Another/images/e253d2ed5af7fd29a33023be92a31550b9dee29de133a796a45fd91a9bbf1162.jpg)

![e29b9182cb6001e2ddf3e0c033215b0da81ba030928dfe293b4ee4df3103b161.jpg](../iclr_results/2305_Large Language Models Often Say One Thing and Do Another/images/e29b9182cb6001e2ddf3e0c033215b0da81ba030928dfe293b4ee4df3103b161.jpg)

![ec8056e2cc79160c5831569c30162fd3038dc59be84cd1e8a5d00c7f84412cc5.jpg](../iclr_results/2305_Large Language Models Often Say One Thing and Do Another/images/ec8056e2cc79160c5831569c30162fd3038dc59be84cd1e8a5d00c7f84412cc5.jpg)

### Tables

![0a070c6e16c6da679b0a072b00296ad02124b58945b10964f9b3408df993b24c.jpg](../iclr_results/2305_Large Language Models Often Say One Thing and Do Another/tables/0a070c6e16c6da679b0a072b00296ad02124b58945b10964f9b3408df993b24c.jpg)

![478edc8074fa9f4a445ad78eae0ad38f9ee1db31c2a5422d2cecb48ceeb514a4.jpg](../iclr_results/2305_Large Language Models Often Say One Thing and Do Another/tables/478edc8074fa9f4a445ad78eae0ad38f9ee1db31c2a5422d2cecb48ceeb514a4.jpg)

![4a2260399b67298b156fb143affe23033c778b6f68c44842e59e4a24a0d7bddf.jpg](../iclr_results/2305_Large Language Models Often Say One Thing and Do Another/tables/4a2260399b67298b156fb143affe23033c778b6f68c44842e59e4a24a0d7bddf.jpg)

![6799f5f9111362f265ee30549d737de2ea293356c523c69e2752d58eb05c6ec8.jpg](../iclr_results/2305_Large Language Models Often Say One Thing and Do Another/tables/6799f5f9111362f265ee30549d737de2ea293356c523c69e2752d58eb05c6ec8.jpg)

![6a5b0b63bdeb83e15e80977b7057c87ce7e7dfc17e4b9bd8888d32a19d3b1ced.jpg](../iclr_results/2305_Large Language Models Often Say One Thing and Do Another/tables/6a5b0b63bdeb83e15e80977b7057c87ce7e7dfc17e4b9bd8888d32a19d3b1ced.jpg)

![820ce1acc85698f44b570a9f9065eca5930931801a879671fe1151d448f716d1.jpg](../iclr_results/2305_Large Language Models Often Say One Thing and Do Another/tables/820ce1acc85698f44b570a9f9065eca5930931801a879671fe1151d448f716d1.jpg)

![ad609d3a59a8c46c5a09dbb87d9f8c5394f77cccb48ef141c25653241e99040d.jpg](../iclr_results/2305_Large Language Models Often Say One Thing and Do Another/tables/ad609d3a59a8c46c5a09dbb87d9f8c5394f77cccb48ef141c25653241e99040d.jpg)

![be043a2f32da0dfecf6c45717cc7060c4fc04081c8f50a0543dbdf328bbd4148.jpg](../iclr_results/2305_Large Language Models Often Say One Thing and Do Another/tables/be043a2f32da0dfecf6c45717cc7060c4fc04081c8f50a0543dbdf328bbd4148.jpg)

## A Theory of Initialisation's Impact on Specialisation

### Images

![37c552fada599dc39be32105c83f91c5b8c118f30dc242c0d711c0f78cddd066.jpg](../iclr_results/2306_A Theory of Initialisation's Impact on Specialisation/images/37c552fada599dc39be32105c83f91c5b8c118f30dc242c0d711c0f78cddd066.jpg)

![4372def90f4bda01aed489ae20ef0df26605e4f60901eb224fb0b67925178f40.jpg](../iclr_results/2306_A Theory of Initialisation's Impact on Specialisation/images/4372def90f4bda01aed489ae20ef0df26605e4f60901eb224fb0b67925178f40.jpg)

![4c004ff8dfc6fb131b4586a93913393174bf6386fe1bac23bfa77068826a7afa.jpg](../iclr_results/2306_A Theory of Initialisation's Impact on Specialisation/images/4c004ff8dfc6fb131b4586a93913393174bf6386fe1bac23bfa77068826a7afa.jpg)

![54384625589463879bde4f09b18fca9c1919a542ca40541ac9771a272ac23f21.jpg](../iclr_results/2306_A Theory of Initialisation's Impact on Specialisation/images/54384625589463879bde4f09b18fca9c1919a542ca40541ac9771a272ac23f21.jpg)

![8b23ec6a9a166652a3e92c140dbbd8e461ae810d1febe162bd1d8fb63f93f444.jpg](../iclr_results/2306_A Theory of Initialisation's Impact on Specialisation/images/8b23ec6a9a166652a3e92c140dbbd8e461ae810d1febe162bd1d8fb63f93f444.jpg)

![9502d8f6f0e36dfda7c192a64e50766c5498f49d0cf3b30459503cd3841058e0.jpg](../iclr_results/2306_A Theory of Initialisation's Impact on Specialisation/images/9502d8f6f0e36dfda7c192a64e50766c5498f49d0cf3b30459503cd3841058e0.jpg)

![97dff5de6695a5851116654f49778f03f6f525315cfad640856dd26ff8263e00.jpg](../iclr_results/2306_A Theory of Initialisation's Impact on Specialisation/images/97dff5de6695a5851116654f49778f03f6f525315cfad640856dd26ff8263e00.jpg)

![9e89f2183984e56f1f5710579874f48c7df3bbe667d5cb3e0d8a176599e5b4ec.jpg](../iclr_results/2306_A Theory of Initialisation's Impact on Specialisation/images/9e89f2183984e56f1f5710579874f48c7df3bbe667d5cb3e0d8a176599e5b4ec.jpg)

![c39c0868ce62c4af58e1d94f32d56a5f6299630b5da91056830c9730736263fb.jpg](../iclr_results/2306_A Theory of Initialisation's Impact on Specialisation/images/c39c0868ce62c4af58e1d94f32d56a5f6299630b5da91056830c9730736263fb.jpg)

![ccb614df45a84053261303fa0378ed2d474061aaa9296f53da6cbbf62c5e999a.jpg](../iclr_results/2306_A Theory of Initialisation's Impact on Specialisation/images/ccb614df45a84053261303fa0378ed2d474061aaa9296f53da6cbbf62c5e999a.jpg)

![cd573cbe9325876cc778a23e4b8f7219476c380b199a720f6331dc0994039735.jpg](../iclr_results/2306_A Theory of Initialisation's Impact on Specialisation/images/cd573cbe9325876cc778a23e4b8f7219476c380b199a720f6331dc0994039735.jpg)

![e17689eb892eea0dbe415fc9e0107a264295cacd852104dc093d2999158c20ce.jpg](../iclr_results/2306_A Theory of Initialisation's Impact on Specialisation/images/e17689eb892eea0dbe415fc9e0107a264295cacd852104dc093d2999158c20ce.jpg)

![f01c219b63847472325941d084e31e3d967c74fda8defed34d5e7fa2b7198b0e.jpg](../iclr_results/2306_A Theory of Initialisation's Impact on Specialisation/images/f01c219b63847472325941d084e31e3d967c74fda8defed34d5e7fa2b7198b0e.jpg)
