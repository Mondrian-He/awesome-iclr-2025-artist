# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 4 of 100

## 目录 (Table of Contents)

1. [Learning Randomized Algorithms with Transformers](#Learning-Randomized-Algorithms-with-Transformers)
2. [Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement](#Trust-or-Escalate-LLM-Judges-with-Provable-Guarantees-for-Human-Agreement)
3. [HiRA: Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models](#HiRA-Parameter-Efficient-Hadamard-High-Rank-Adaptation-for-Large-Language-Models)
4. [Proteina: Scaling Flow-based Protein Structure Generative Models](#Proteina-Scaling-Flow-based-Protein-Structure-Generative-Models)
5. [REEF: Representation Encoding Fingerprints for Large Language Models](#REEF-Representation-Encoding-Fingerprints-for-Large-Language-Models)
6. [A Decade's Battle on Dataset Bias: Are We There Yet?](#A-Decades-Battle-on-Dataset-Bias-Are-We-There-Yet)
7. [Context-Parametric Inversion: Why Instruction Finetuning May Not Actually Improve Context Reliance](#Context-Parametric-Inversion-Why-Instruction-Finetuning-May-Not-Actually-Improve-Context-Reliance)
8. [Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](#Transfusion-Predict-the-Next-Token-and-Diffuse-Images-with-One-Multi-Modal-Model)
9. [ECD: A Machine Learning Benchmark for Predicting Enhanced-Precision Electronic Charge Density in Crystalline Inorganic Materials](#ECD-A-Machine-Learning-Benchmark-for-Predicting-Enhanced-Precision-Electronic-Charge-Density-in-Crystalline-Inorganic-Materials)
10. [Generator Matching: Generative modeling with arbitrary Markov processes](#Generator-Matching-Generative-modeling-with-arbitrary-Markov-processes)
11. [Brain Bandit: A Biologically Grounded Neural Network for Efficient Control of Exploration](#Brain-Bandit-A-Biologically-Grounded-Neural-Network-for-Efficient-Control-of-Exploration)
12. [Do LLMs Recognize Your Preferences? Evaluating Personalized Preference Following in LLMs](#Do-LLMs-Recognize-Your-Preferences-Evaluating-Personalized-Preference-Following-in-LLMs)
13. [LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias](#LVSM-A-Large-View-Synthesis-Model-with-Minimal-3D-Inductive-Bias)
14. [From Exploration to Mastery: Enabling LLMs to Master Tools via Self-Driven Interactions](#From-Exploration-to-Mastery-Enabling-LLMs-to-Master-Tools-via-Self-Driven-Interactions)
15. [Advantage Alignment Algorithms](#Advantage-Alignment-Algorithms)
16. [RM-Bench: Benchmarking Reward Models of Language Models with Subtlety and Style](#RM-Bench-Benchmarking-Reward-Models-of-Language-Models-with-Subtlety-and-Style)
17. [PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding](#PhysBench-Benchmarking-and-Enhancing-Vision-Language-Models-for-Physical-World-Understanding)
18. [Iterative Nash Policy Optimization: Aligning LLMs with General Preferences via No-Regret Learning](#Iterative-Nash-Policy-Optimization-Aligning-LLMs-with-General-Preferences-via-No-Regret-Learning)
19. [Steering Protein Family Design through Profile Bayesian Flow](#Steering-Protein-Family-Design-through-Profile-Bayesian-Flow)
20. [Unlearning-based Neural Interpretations](#Unlearning-based-Neural-Interpretations)
21. [On the Hölder Stability of Multiset and Graph Neural Networks](#On-the-Hölder-Stability-of-Multiset-and-Graph-Neural-Networks)
22. [No Pose, No Problem: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images](#No-Pose-No-Problem-Surprisingly-Simple-3D-Gaussian-Splats-from-Sparse-Unposed-Images)
23. [KAN: Kolmogorov–Arnold Networks](#KAN-KolmogorovArnold-Networks)
24. [Differential Transformer](#Differential-Transformer)
25. [One Step Diffusion via Shortcut Models](#One-Step-Diffusion-via-Shortcut-Models)
26. [FlexPrefill: A Context-Aware Sparse Attention Mechanism for Efficient Long-Sequence Inference](#FlexPrefill-A-Context-Aware-Sparse-Attention-Mechanism-for-Efficient-Long-Sequence-Inference)
27. [A Theoretically-Principled Sparse, Connected, and Rigid Graph Representation of Molecules](#A-Theoretically-Principled-Sparse-Connected-and-Rigid-Graph-Representation-of-Molecules)
28. [REGENT: A Retrieval-Augmented Generalist Agent That Can Act In-Context in New Environments](#REGENT-A-Retrieval-Augmented-Generalist-Agent-That-Can-Act-In-Context-in-New-Environments)
29. [Limits to scalable evaluation at the frontier: LLM as judge won’t beat twice the data](#Limits-to-scalable-evaluation-at-the-frontier-LLM-as-judge-wont-beat-twice-the-data)
30. [MAP: Multi-Human-Value Alignment Palette](#MAP-Multi-Human-Value-Alignment-Palette)
31. [SANA: Efficient High-Resolution Text-to-Image Synthesis with Linear Diffusion Transformers](#SANA-Efficient-High-Resolution-Text-to-Image-Synthesis-with-Linear-Diffusion-Transformers)
32. [Learning to Discover Regulatory Elements for Gene Expression Prediction](#Learning-to-Discover-Regulatory-Elements-for-Gene-Expression-Prediction)
33. [Simplifying, Stabilizing and Scaling Continuous-time Consistency Models](#Simplifying-Stabilizing-and-Scaling-Continuous-time-Consistency-Models)
34. [TANGO: Co-Speech Gesture Video Reenactment with Hierarchical Audio Motion Embedding and Diffusion Interpolation](#TANGO-Co-Speech-Gesture-Video-Reenactment-with-Hierarchical-Audio-Motion-Embedding-and-Diffusion-Interpolation)
35. [ShEPhERD: Diffusing shape, electrostatics, and pharmacophores for bioisosteric drug design](#ShEPhERD-Diffusing-shape-electrostatics-and-pharmacophores-for-bioisosteric-drug-design)
36. [miniCTX: Neural Theorem Proving with (Long-)Contexts](#miniCTX-Neural-Theorem-Proving-with-Long-Contexts)
37. [Residual Deep Gaussian Processes on Manifolds](#Residual-Deep-Gaussian-Processes-on-Manifolds)

---


## Learning Randomized Algorithms with Transformers

### Images

![112f8bf9f0e85bce72252f5b5ff5d51eaae3ba00be95257eb1db6268e9c32768.jpg](../iclr_results/113_Unlocking%20State-Tracking%20in%20Linear%20RNNs%20Through%20Negative%20Eigenvalues/images/112f8bf9f0e85bce72252f5b5ff5d51eaae3ba00be95257eb1db6268e9c32768.jpg)

![2caf76a620294290c784c2ab9f3ee789d4e11f857ba770af3fd74e224dfef8f7.jpg](../iclr_results/113_Unlocking%20State-Tracking%20in%20Linear%20RNNs%20Through%20Negative%20Eigenvalues/images/2caf76a620294290c784c2ab9f3ee789d4e11f857ba770af3fd74e224dfef8f7.jpg)

![50eeac28266ed27c428a7856a97276267f221156cce41545c201c16227c96db8.jpg](../iclr_results/113_Unlocking%20State-Tracking%20in%20Linear%20RNNs%20Through%20Negative%20Eigenvalues/images/50eeac28266ed27c428a7856a97276267f221156cce41545c201c16227c96db8.jpg)

![52f5b27b0069b6dde6cfdf685a3b4e144cdd6abab5c4ddcc91015efc49d46fee.jpg](../iclr_results/113_Unlocking%20State-Tracking%20in%20Linear%20RNNs%20Through%20Negative%20Eigenvalues/images/52f5b27b0069b6dde6cfdf685a3b4e144cdd6abab5c4ddcc91015efc49d46fee.jpg)

![641ca3f815911ab1316f2240a3a4b67d2c9784845447d10171a604b70913165e.jpg](../iclr_results/113_Unlocking%20State-Tracking%20in%20Linear%20RNNs%20Through%20Negative%20Eigenvalues/images/641ca3f815911ab1316f2240a3a4b67d2c9784845447d10171a604b70913165e.jpg)

![76c5cf6b354b0f69184681c5b89453dd5e80636a8b6960a8f84ff4a4e107c2a0.jpg](../iclr_results/113_Unlocking%20State-Tracking%20in%20Linear%20RNNs%20Through%20Negative%20Eigenvalues/images/76c5cf6b354b0f69184681c5b89453dd5e80636a8b6960a8f84ff4a4e107c2a0.jpg)

![7e6e5c6fa5b9a6043c90c0b5af8af4d6d30a1eefbc001d7fb457e31a54117508.jpg](../iclr_results/113_Unlocking%20State-Tracking%20in%20Linear%20RNNs%20Through%20Negative%20Eigenvalues/images/7e6e5c6fa5b9a6043c90c0b5af8af4d6d30a1eefbc001d7fb457e31a54117508.jpg)

![8eb2ac98242f807525c15618df8745eb01c0467bf3e8e0818cfe692e57a1f0c1.jpg](../iclr_results/113_Unlocking%20State-Tracking%20in%20Linear%20RNNs%20Through%20Negative%20Eigenvalues/images/8eb2ac98242f807525c15618df8745eb01c0467bf3e8e0818cfe692e57a1f0c1.jpg)

![abfea7bcd78bb74ae962a2f7c70f1dd02f6f9749185bf74d6fc9089865a7bf37.jpg](../iclr_results/113_Unlocking%20State-Tracking%20in%20Linear%20RNNs%20Through%20Negative%20Eigenvalues/images/abfea7bcd78bb74ae962a2f7c70f1dd02f6f9749185bf74d6fc9089865a7bf37.jpg)

![ba30db46e3ee2b25dd4fe62996720ffe8cce2026d85be5bf4104362bf808da07.jpg](../iclr_results/113_Unlocking%20State-Tracking%20in%20Linear%20RNNs%20Through%20Negative%20Eigenvalues/images/ba30db46e3ee2b25dd4fe62996720ffe8cce2026d85be5bf4104362bf808da07.jpg)

![ce424b27539aedd5ef391f7532a1598ce1451b29b65d8ae733bc1e4b689ef923.jpg](../iclr_results/113_Unlocking%20State-Tracking%20in%20Linear%20RNNs%20Through%20Negative%20Eigenvalues/images/ce424b27539aedd5ef391f7532a1598ce1451b29b65d8ae733bc1e4b689ef923.jpg)

![d8e9dd7db02985a53e9ce658aabb9baa645a79e194755bd72a9ecfc2b30f7037.jpg](../iclr_results/113_Unlocking%20State-Tracking%20in%20Linear%20RNNs%20Through%20Negative%20Eigenvalues/images/d8e9dd7db02985a53e9ce658aabb9baa645a79e194755bd72a9ecfc2b30f7037.jpg)

![eac3fac010e652339760ecfda9a3e649dd9cf81c2b6fde9cd5d5995a2eaff7fd.jpg](../iclr_results/113_Unlocking%20State-Tracking%20in%20Linear%20RNNs%20Through%20Negative%20Eigenvalues/images/eac3fac010e652339760ecfda9a3e649dd9cf81c2b6fde9cd5d5995a2eaff7fd.jpg)

### Tables

![9d0194109beb170b03abca38ccc89760acfc4cb8fa49f4706ee8b5d193cac9b4.jpg](../iclr_results/113_Unlocking%20State-Tracking%20in%20Linear%20RNNs%20Through%20Negative%20Eigenvalues/tables/9d0194109beb170b03abca38ccc89760acfc4cb8fa49f4706ee8b5d193cac9b4.jpg)

![b0da2967ef48dfe21be89bcd91d2176c28e19eba954aa750cf73b0ef325ddfa1.jpg](../iclr_results/113_Unlocking%20State-Tracking%20in%20Linear%20RNNs%20Through%20Negative%20Eigenvalues/tables/b0da2967ef48dfe21be89bcd91d2176c28e19eba954aa750cf73b0ef325ddfa1.jpg)

![b7b40e14a7a50bf21be9705b8de94b51deac07036b57cca892457214902dce55.jpg](../iclr_results/113_Unlocking%20State-Tracking%20in%20Linear%20RNNs%20Through%20Negative%20Eigenvalues/tables/b7b40e14a7a50bf21be9705b8de94b51deac07036b57cca892457214902dce55.jpg)

![b82bc4671517620910b0f1f05ce77031a527a584a43a76b6c5953ba2a02e827e.jpg](../iclr_results/113_Unlocking%20State-Tracking%20in%20Linear%20RNNs%20Through%20Negative%20Eigenvalues/tables/b82bc4671517620910b0f1f05ce77031a527a584a43a76b6c5953ba2a02e827e.jpg)

![ef56a91ca21cc357cff4591b938f3388559e0594dacc7a8e7bf53dad36c1be76.jpg](../iclr_results/113_Unlocking%20State-Tracking%20in%20Linear%20RNNs%20Through%20Negative%20Eigenvalues/tables/ef56a91ca21cc357cff4591b938f3388559e0594dacc7a8e7bf53dad36c1be76.jpg)

## Learning Randomized Algorithms with Transformers


### Images

![1737221bdd23e456cabaf9f119b8255b53f3086a5af3a1bb22f2bdc50444d479.jpg](../iclr_results/114_Learning%20Randomized%20Algorithms%20with%20Transformers/images/1737221bdd23e456cabaf9f119b8255b53f3086a5af3a1bb22f2bdc50444d479.jpg)

![3e849264019817c0ea38271d49963f5c68343702eefcf191d19a7ab5cd3899b9.jpg](../iclr_results/114_Learning%20Randomized%20Algorithms%20with%20Transformers/images/3e849264019817c0ea38271d49963f5c68343702eefcf191d19a7ab5cd3899b9.jpg)

![537a148a3bf3781fc59a5b82b6706bb16b1ecf7e76fc53053cacdd08e16e78f4.jpg](../iclr_results/114_Learning%20Randomized%20Algorithms%20with%20Transformers/images/537a148a3bf3781fc59a5b82b6706bb16b1ecf7e76fc53053cacdd08e16e78f4.jpg)

![66fca5b9e0345db9e4f2cfa2e0002bffc5f15c4261a6c404a3d700321a833a7a.jpg](../iclr_results/114_Learning%20Randomized%20Algorithms%20with%20Transformers/images/66fca5b9e0345db9e4f2cfa2e0002bffc5f15c4261a6c404a3d700321a833a7a.jpg)

![6dee92f4a13c94a3aae9b394364572f89f58b3e6f536e59ad88bda9cf85279e4.jpg](../iclr_results/114_Learning%20Randomized%20Algorithms%20with%20Transformers/images/6dee92f4a13c94a3aae9b394364572f89f58b3e6f536e59ad88bda9cf85279e4.jpg)

![78391803567d7f8fc830aff80d2904f7ead69a44d4d49fe60ff2f5385539fdb9.jpg](../iclr_results/114_Learning%20Randomized%20Algorithms%20with%20Transformers/images/78391803567d7f8fc830aff80d2904f7ead69a44d4d49fe60ff2f5385539fdb9.jpg)

![7e59dadf267f8c0d7a2c913aeb3c47fdf1256d6b9f54c90d37c2a86653e263bc.jpg](../iclr_results/114_Learning%20Randomized%20Algorithms%20with%20Transformers/images/7e59dadf267f8c0d7a2c913aeb3c47fdf1256d6b9f54c90d37c2a86653e263bc.jpg)

![89c5029dd042d4f84ac4ea571d94674be894fe8dea34bb4844a1ae715a7c2785.jpg](../iclr_results/114_Learning%20Randomized%20Algorithms%20with%20Transformers/images/89c5029dd042d4f84ac4ea571d94674be894fe8dea34bb4844a1ae715a7c2785.jpg)

![d85f4cf113e3742b44a72f6d1969e18831e13ae5071ebeed17594a54d2c1d067.jpg](../iclr_results/114_Learning%20Randomized%20Algorithms%20with%20Transformers/images/d85f4cf113e3742b44a72f6d1969e18831e13ae5071ebeed17594a54d2c1d067.jpg)

![dea510726abc2cf7660195d6409b7f45ae36f359f9d1a9a62fba5ecf8dd2ab37.jpg](../iclr_results/114_Learning%20Randomized%20Algorithms%20with%20Transformers/images/dea510726abc2cf7660195d6409b7f45ae36f359f9d1a9a62fba5ecf8dd2ab37.jpg)

### Tables

![0cfeb991bab79f22ca4b0ef18d800dc8736414a2ef8b695b7d29cc5f5aca8c99.jpg](../iclr_results/114_Learning%20Randomized%20Algorithms%20with%20Transformers/tables/0cfeb991bab79f22ca4b0ef18d800dc8736414a2ef8b695b7d29cc5f5aca8c99.jpg)

![425608e59de352f81486c0f54b71dbcc85ad9bb9963ea2cf35402c6957e0e9d7.jpg](../iclr_results/114_Learning%20Randomized%20Algorithms%20with%20Transformers/tables/425608e59de352f81486c0f54b71dbcc85ad9bb9963ea2cf35402c6957e0e9d7.jpg)

![cbe2c291ae3ef19e5a02f951a7259e10a20a565a609d9d6e46e140fe1a38c2a4.jpg](../iclr_results/114_Learning%20Randomized%20Algorithms%20with%20Transformers/tables/cbe2c291ae3ef19e5a02f951a7259e10a20a565a609d9d6e46e140fe1a38c2a4.jpg)

## Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement


### Images

![0e1114daffdc3c87522e72697b2baf1b8ca56d9466c9504a0dfa9a0b9608783b.jpg](../iclr_results/115_Trust%20or%20Escalate_%20LLM%20Judges%20with%20Provable%20Guarantees%20for%20Human%20Agreement/images/0e1114daffdc3c87522e72697b2baf1b8ca56d9466c9504a0dfa9a0b9608783b.jpg)

![0ef53991d4f542b214873b4b6623a4cf2d2e7d3198d9c4d4507e147830364c40.jpg](../iclr_results/115_Trust%20or%20Escalate_%20LLM%20Judges%20with%20Provable%20Guarantees%20for%20Human%20Agreement/images/0ef53991d4f542b214873b4b6623a4cf2d2e7d3198d9c4d4507e147830364c40.jpg)

![2b83d9517a937d13325bc14b05888279cb374363167b78dea0c022d2d06127c8.jpg](../iclr_results/115_Trust%20or%20Escalate_%20LLM%20Judges%20with%20Provable%20Guarantees%20for%20Human%20Agreement/images/2b83d9517a937d13325bc14b05888279cb374363167b78dea0c022d2d06127c8.jpg)

![bb100cc2e61fd557af5fd4786835ab4e84b13a1fbc39f0cadd34555f2aeef266.jpg](../iclr_results/115_Trust%20or%20Escalate_%20LLM%20Judges%20with%20Provable%20Guarantees%20for%20Human%20Agreement/images/bb100cc2e61fd557af5fd4786835ab4e84b13a1fbc39f0cadd34555f2aeef266.jpg)

![c66383fbcc423e2fc250852674aed7567aaf00298e00c6e1617c768e181494cd.jpg](../iclr_results/115_Trust%20or%20Escalate_%20LLM%20Judges%20with%20Provable%20Guarantees%20for%20Human%20Agreement/images/c66383fbcc423e2fc250852674aed7567aaf00298e00c6e1617c768e181494cd.jpg)

### Tables

![0c3ee4a892650874c004582c0f1d81980f5392604d235c1902b70ef738385328.jpg](../iclr_results/115_Trust%20or%20Escalate_%20LLM%20Judges%20with%20Provable%20Guarantees%20for%20Human%20Agreement/tables/0c3ee4a892650874c004582c0f1d81980f5392604d235c1902b70ef738385328.jpg)

![0d10a17be6c9b9c7725bc5b6f048d7b81ad4831c5bbaa8d05f65f03208142bab.jpg](../iclr_results/115_Trust%20or%20Escalate_%20LLM%20Judges%20with%20Provable%20Guarantees%20for%20Human%20Agreement/tables/0d10a17be6c9b9c7725bc5b6f048d7b81ad4831c5bbaa8d05f65f03208142bab.jpg)

![14341b620d6fdb18b167bdaf860637ffa805aabf7769c9266feec0d9f6e5461a.jpg](../iclr_results/115_Trust%20or%20Escalate_%20LLM%20Judges%20with%20Provable%20Guarantees%20for%20Human%20Agreement/tables/14341b620d6fdb18b167bdaf860637ffa805aabf7769c9266feec0d9f6e5461a.jpg)

![2032f00de2595402ec3919ee3f36da227002f3ecb4bd07645fe8687349670c9b.jpg](../iclr_results/115_Trust%20or%20Escalate_%20LLM%20Judges%20with%20Provable%20Guarantees%20for%20Human%20Agreement/tables/2032f00de2595402ec3919ee3f36da227002f3ecb4bd07645fe8687349670c9b.jpg)

![24c816f3286e2e079d52f1a8b62fc411685b67a311f6ebafe0d9d32b61acb9f9.jpg](../iclr_results/115_Trust%20or%20Escalate_%20LLM%20Judges%20with%20Provable%20Guarantees%20for%20Human%20Agreement/tables/24c816f3286e2e079d52f1a8b62fc411685b67a311f6ebafe0d9d32b61acb9f9.jpg)

![3f49cdc6d3d903dba071f5ae4da1b21784c42163d47b313f2834ad9c8e35a0f4.jpg](../iclr_results/115_Trust%20or%20Escalate_%20LLM%20Judges%20with%20Provable%20Guarantees%20for%20Human%20Agreement/tables/3f49cdc6d3d903dba071f5ae4da1b21784c42163d47b313f2834ad9c8e35a0f4.jpg)

![7a13c91ee3f26aa410e378f28ba06adaed3f9393fbd4f3dd6baa5aaa7dea6e85.jpg](../iclr_results/115_Trust%20or%20Escalate_%20LLM%20Judges%20with%20Provable%20Guarantees%20for%20Human%20Agreement/tables/7a13c91ee3f26aa410e378f28ba06adaed3f9393fbd4f3dd6baa5aaa7dea6e85.jpg)

![905db7c28a4f4a61ee614b5edbf08dce497e1cef67596bdefaf68928ac969559.jpg](../iclr_results/115_Trust%20or%20Escalate_%20LLM%20Judges%20with%20Provable%20Guarantees%20for%20Human%20Agreement/tables/905db7c28a4f4a61ee614b5edbf08dce497e1cef67596bdefaf68928ac969559.jpg)

![b620ae9d742c624adaa683e4a4a0c874dd3679ee0250c14dcaf426f3f0e5538c.jpg](../iclr_results/115_Trust%20or%20Escalate_%20LLM%20Judges%20with%20Provable%20Guarantees%20for%20Human%20Agreement/tables/b620ae9d742c624adaa683e4a4a0c874dd3679ee0250c14dcaf426f3f0e5538c.jpg)

![ccfbab025ccf854935e0b884d663200ded015bc3698c3927e126e98d6df8a977.jpg](../iclr_results/115_Trust%20or%20Escalate_%20LLM%20Judges%20with%20Provable%20Guarantees%20for%20Human%20Agreement/tables/ccfbab025ccf854935e0b884d663200ded015bc3698c3927e126e98d6df8a977.jpg)

![d14e6bc2d410ec6d6292ca7a2a5d7f7a9908b960855c8953441a922376b5a70a.jpg](../iclr_results/115_Trust%20or%20Escalate_%20LLM%20Judges%20with%20Provable%20Guarantees%20for%20Human%20Agreement/tables/d14e6bc2d410ec6d6292ca7a2a5d7f7a9908b960855c8953441a922376b5a70a.jpg)

## HiRA: Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models


### Images

![112a808d1c081e6c20f67cf74d147e6ae56f31fbfeec29a8d88a9c9be9a7589b.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/images/112a808d1c081e6c20f67cf74d147e6ae56f31fbfeec29a8d88a9c9be9a7589b.jpg)

![137169e82e066989b9641ad1bbe9dea8e79e3f576a3604d491d280cdddd7ef1d.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/images/137169e82e066989b9641ad1bbe9dea8e79e3f576a3604d491d280cdddd7ef1d.jpg)

![164524c3162b0588d8658af1300b73c57510505ebcd889ba76ee13d0e7ba3f3e.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/images/164524c3162b0588d8658af1300b73c57510505ebcd889ba76ee13d0e7ba3f3e.jpg)

![2e8bc835f443eb27559af27b55b6ad1d53e5834d2145f0fa75f4120788499edc.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/images/2e8bc835f443eb27559af27b55b6ad1d53e5834d2145f0fa75f4120788499edc.jpg)

![68bc0b42a8caab61ef96a1b413670ed4c1010569ba26105622fdaba3582bac94.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/images/68bc0b42a8caab61ef96a1b413670ed4c1010569ba26105622fdaba3582bac94.jpg)

![6be0a9102ded4a8768f1ce74a7588ddc0871bc995b617153210e1e09a5912859.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/images/6be0a9102ded4a8768f1ce74a7588ddc0871bc995b617153210e1e09a5912859.jpg)

![6f01b5b2f471b06183b4edc9215bf4fffceaf9fb84382011316dcf0123918625.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/images/6f01b5b2f471b06183b4edc9215bf4fffceaf9fb84382011316dcf0123918625.jpg)

![7399c4eaac70c530232cff49386a0e1bf133f538b10009ad9b8cc2ed387b183d.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/images/7399c4eaac70c530232cff49386a0e1bf133f538b10009ad9b8cc2ed387b183d.jpg)

![93c4f009d22dc2093e0bf864690e47bd40dc83eaf05c4ca7b51ecaacae63bee7.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/images/93c4f009d22dc2093e0bf864690e47bd40dc83eaf05c4ca7b51ecaacae63bee7.jpg)

![a767b7b1fb0b2c0e5122d7dbe20b699ef3fd01e00726e8d5a85a6ccbf37abe60.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/images/a767b7b1fb0b2c0e5122d7dbe20b699ef3fd01e00726e8d5a85a6ccbf37abe60.jpg)

![fd77e42fa9595d7fda8f569c93602a9ff324cbf89d8dc34060a1822097679ef1.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/images/fd77e42fa9595d7fda8f569c93602a9ff324cbf89d8dc34060a1822097679ef1.jpg)

### Tables

![04033a70a0f56d75700ec0805952d0a2eeafc856fe5e5d0d8fe9b9acbb84c49e.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/tables/04033a70a0f56d75700ec0805952d0a2eeafc856fe5e5d0d8fe9b9acbb84c49e.jpg)

![10bd5f1cb57cc810fa22859e5b637ad5795cbb9a6909e9ca879b37bd9cb5b758.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/tables/10bd5f1cb57cc810fa22859e5b637ad5795cbb9a6909e9ca879b37bd9cb5b758.jpg)

![259b5a873d81dcf11b1d3709ab9e5131d7f86874aad4d27894213c8718dc5e46.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/tables/259b5a873d81dcf11b1d3709ab9e5131d7f86874aad4d27894213c8718dc5e46.jpg)

![285c7718d09706e805b958caffa6e1bbbf520b65c72f20143afd4ca236a13498.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/tables/285c7718d09706e805b958caffa6e1bbbf520b65c72f20143afd4ca236a13498.jpg)

![640371e9670d85dce9b01f97b1efd14aad3365311594f09db2a33daf5bdc3682.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/tables/640371e9670d85dce9b01f97b1efd14aad3365311594f09db2a33daf5bdc3682.jpg)

![641df8ccf521d0027c205d39a9d9d849303e9017329772db72f27806d03238a4.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/tables/641df8ccf521d0027c205d39a9d9d849303e9017329772db72f27806d03238a4.jpg)

![7b06c7f45b3949ed5090766538d88c1d3e512aef71ede78d45a4854f2403c2b2.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/tables/7b06c7f45b3949ed5090766538d88c1d3e512aef71ede78d45a4854f2403c2b2.jpg)

![83e0cb433f1a2fcb6632bd8ca2cd94eb084ac08f8898bfcf7ec9ccfcf74dabd4.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/tables/83e0cb433f1a2fcb6632bd8ca2cd94eb084ac08f8898bfcf7ec9ccfcf74dabd4.jpg)

![846ff2fb83760df69478ed26cd015ab0b612f4a9f8ffc13d68fbf2a1dabab53f.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/tables/846ff2fb83760df69478ed26cd015ab0b612f4a9f8ffc13d68fbf2a1dabab53f.jpg)

![bacd81633ed41b5f76755e5723469fab3cf0d3e2decca2149c1fc7854bd9bba8.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/tables/bacd81633ed41b5f76755e5723469fab3cf0d3e2decca2149c1fc7854bd9bba8.jpg)

![cae53408ddbbb7a898e4fc8dd8710fc94fc8f1095cbcfaf77764a0c97f2a3813.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/tables/cae53408ddbbb7a898e4fc8dd8710fc94fc8f1095cbcfaf77764a0c97f2a3813.jpg)

![d3ccf11b1352c00c20854506954a419daf81582368ed3a1c9398f78d851e7f7f.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/tables/d3ccf11b1352c00c20854506954a419daf81582368ed3a1c9398f78d851e7f7f.jpg)

![e7c41d79e487ddbfd7536389aaa056ce2e6ff51dc550411478785d2b906bb176.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/tables/e7c41d79e487ddbfd7536389aaa056ce2e6ff51dc550411478785d2b906bb176.jpg)

![ea63962d293f137fe8416d032fa998cd6da6dc0994b8735666ad8f5fe79a33e9.jpg](../iclr_results/116_HiRA_%20Parameter-Efficient%20Hadamard%20High-Rank%20Adaptation%20for%20Large%20Language%20Models/tables/ea63962d293f137fe8416d032fa998cd6da6dc0994b8735666ad8f5fe79a33e9.jpg)

## Proteina: Scaling Flow-based Protein Structure Generative Models


### Images

![05338d893e63e1380b8b8c092c5b2e57cbc65ff2d52758db9d9ad40ff1b397c4.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/05338d893e63e1380b8b8c092c5b2e57cbc65ff2d52758db9d9ad40ff1b397c4.jpg)

![1b7277c16206ece76d0ef1c45972c484badb0211ffeef4917be4ccc54c16239a.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/1b7277c16206ece76d0ef1c45972c484badb0211ffeef4917be4ccc54c16239a.jpg)

![21104b66dfed6ea10e95879850786eea74ea0670c05815bcf6e7dc111e7477b1.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/21104b66dfed6ea10e95879850786eea74ea0670c05815bcf6e7dc111e7477b1.jpg)

![25ead3da68caefb0b09be25319880b18470d441f2023874bee8983db157c2488.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/25ead3da68caefb0b09be25319880b18470d441f2023874bee8983db157c2488.jpg)

![2c99fa1f511abaa2508a77c9cd5d12481808eef5c168915947956c80be870dee.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/2c99fa1f511abaa2508a77c9cd5d12481808eef5c168915947956c80be870dee.jpg)

![3d04c7975ac1d11bf81edaeb776b2123a3bca04b25569d48f52fd19bebc729d1.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/3d04c7975ac1d11bf81edaeb776b2123a3bca04b25569d48f52fd19bebc729d1.jpg)

![4118e2db35dddb50150f4351decb4107f441184cde64d2eb250cd1caf5f396c0.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/4118e2db35dddb50150f4351decb4107f441184cde64d2eb250cd1caf5f396c0.jpg)

![41205599648703870afa58c0e0b37b460070d71b7a488b6fa9964549327adaa6.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/41205599648703870afa58c0e0b37b460070d71b7a488b6fa9964549327adaa6.jpg)

![44584a4ecd474166e9e945fbc9796ddb3069b2b7e4a2cad4316feb8f81e8f69e.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/44584a4ecd474166e9e945fbc9796ddb3069b2b7e4a2cad4316feb8f81e8f69e.jpg)

![44c71c064e9ef3c67406053a4236ec7f3b27daa0ffb688e3dcff35fec603108f.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/44c71c064e9ef3c67406053a4236ec7f3b27daa0ffb688e3dcff35fec603108f.jpg)

![50efd7c7ebfaf310d13f0cdedc4d01f451b4729c1c863f6877186b1fdd833ea7.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/50efd7c7ebfaf310d13f0cdedc4d01f451b4729c1c863f6877186b1fdd833ea7.jpg)

![51c10417f74e483a75a38946857eb9fef23557bb8483b66bf851b9d9a4b47d25.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/51c10417f74e483a75a38946857eb9fef23557bb8483b66bf851b9d9a4b47d25.jpg)

![6d8296a75145efd3db72fa1dfcde667e1984cff9d3676a32648f313185335442.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/6d8296a75145efd3db72fa1dfcde667e1984cff9d3676a32648f313185335442.jpg)

![6e3e0398b82a5dd877209696b29c3414d7b83d83cb78fea578ba1c74369cd517.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/6e3e0398b82a5dd877209696b29c3414d7b83d83cb78fea578ba1c74369cd517.jpg)

![72cc7fe5eaed65368b1262a4c5a408326842da71e1bf06da4a89001a161d0002.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/72cc7fe5eaed65368b1262a4c5a408326842da71e1bf06da4a89001a161d0002.jpg)

![79df876d2079788b29fd25ced386bd75d5e5c5d364423c0beb2d36ceaa01bdc9.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/79df876d2079788b29fd25ced386bd75d5e5c5d364423c0beb2d36ceaa01bdc9.jpg)

![833ad2e9b8f9ebd0ac5e5ae980678f3bbbc35ce950939ccc6e6276573c5b8648.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/833ad2e9b8f9ebd0ac5e5ae980678f3bbbc35ce950939ccc6e6276573c5b8648.jpg)

![a19da17464276d053d541106d0bffc6bf61045b54a0341fbd4bb416321b65cfd.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/a19da17464276d053d541106d0bffc6bf61045b54a0341fbd4bb416321b65cfd.jpg)

![ad517d9a9e4dd774e7e943a09853e95dc3f92b8aa4691e56738063dee0cfb921.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/ad517d9a9e4dd774e7e943a09853e95dc3f92b8aa4691e56738063dee0cfb921.jpg)

![b22a6e52ba38f27301c319cdd9a05936931bf39a4656eba481701177b5a80d4e.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/b22a6e52ba38f27301c319cdd9a05936931bf39a4656eba481701177b5a80d4e.jpg)

![bd720245ee33267b3e504307b65b9067a7638f9ccc3eec69b406bce6139132f9.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/bd720245ee33267b3e504307b65b9067a7638f9ccc3eec69b406bce6139132f9.jpg)

![c85f96199fef8b11cd542ec73b5ab2dbf2c6a78744ac73c2ec83cb6f411b93b1.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/c85f96199fef8b11cd542ec73b5ab2dbf2c6a78744ac73c2ec83cb6f411b93b1.jpg)

![cc4b881f4d6b3115c0efa562340470dd10e84c05127afba60dff30dc71e312cc.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/cc4b881f4d6b3115c0efa562340470dd10e84c05127afba60dff30dc71e312cc.jpg)

![d7176bb148e61007646e47e7cc949e7748e7ec0f8a2538426be533fe8facb6ec.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/d7176bb148e61007646e47e7cc949e7748e7ec0f8a2538426be533fe8facb6ec.jpg)

![e907df560448dc0b6a48c7bd5cfb0f0b39566c72a9a19b70d694a8a38ce02eee.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/images/e907df560448dc0b6a48c7bd5cfb0f0b39566c72a9a19b70d694a8a38ce02eee.jpg)

### Tables

![1622e3372c245fdb6abbaf139ed2ef99b394c4b08670d6e1994f6f7371096075.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/tables/1622e3372c245fdb6abbaf139ed2ef99b394c4b08670d6e1994f6f7371096075.jpg)

![1df06ccbdb25e064d3e9a40adfff82ceb00b56b2ccfcba99334f7f6ffb4479c9.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/tables/1df06ccbdb25e064d3e9a40adfff82ceb00b56b2ccfcba99334f7f6ffb4479c9.jpg)

![34937d1d7c070769e1c6d4b99ee201e6af78fcca44706e165946894f24b78653.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/tables/34937d1d7c070769e1c6d4b99ee201e6af78fcca44706e165946894f24b78653.jpg)

![46934d390b18c48d3d6f1c77a8621179d1642cc8dfcbe9e159a79b0eef43451e.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/tables/46934d390b18c48d3d6f1c77a8621179d1642cc8dfcbe9e159a79b0eef43451e.jpg)

![7316f886d1095b8fb867a13341b40ab77ae5a1f421a665ccb9aa56dc0ef39eb1.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/tables/7316f886d1095b8fb867a13341b40ab77ae5a1f421a665ccb9aa56dc0ef39eb1.jpg)

![7df2c2190e3c56a52bbd360c341d5c0a557ad6b57d6006edf48c46e6c597edec.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/tables/7df2c2190e3c56a52bbd360c341d5c0a557ad6b57d6006edf48c46e6c597edec.jpg)

![7e2eec5c7a12ce01deb06a587bc3f277f0ef02ad8feec9238b6a13d4e3bbad1b.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/tables/7e2eec5c7a12ce01deb06a587bc3f277f0ef02ad8feec9238b6a13d4e3bbad1b.jpg)

![86f8f91d30b10f67a27f8c84502ed5f04689da856bc67210b4d946d24ffd5b7c.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/tables/86f8f91d30b10f67a27f8c84502ed5f04689da856bc67210b4d946d24ffd5b7c.jpg)

![8b599610fe0d7671de2682dbe7a5193d0942fa66632d95a7bc12919de429eefa.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/tables/8b599610fe0d7671de2682dbe7a5193d0942fa66632d95a7bc12919de429eefa.jpg)

![9b89e00bdd79de6d30b6d67f01db81b685327ffcc4bd118850c917467d3d9d36.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/tables/9b89e00bdd79de6d30b6d67f01db81b685327ffcc4bd118850c917467d3d9d36.jpg)

![9e3a0913a9c67de85b2821ee8ab80ea6f0492fddeaea1d68e56562f14128859c.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/tables/9e3a0913a9c67de85b2821ee8ab80ea6f0492fddeaea1d68e56562f14128859c.jpg)

![a89e0ac12a9998e8b5c9da2bbd46b879b2ec6cca8e59ee3e11c33c3e07c8dafb.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/tables/a89e0ac12a9998e8b5c9da2bbd46b879b2ec6cca8e59ee3e11c33c3e07c8dafb.jpg)

![a98c99164b01b0e1d0b65b50594d647a32a1126709afa6e50bbf0b1d8095c536.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/tables/a98c99164b01b0e1d0b65b50594d647a32a1126709afa6e50bbf0b1d8095c536.jpg)

![c0349b26f6815edf51cdbebaf705b68f73e272465854925c4ab3ff3e31cbf54c.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/tables/c0349b26f6815edf51cdbebaf705b68f73e272465854925c4ab3ff3e31cbf54c.jpg)

![f2bfd518746c6cb322aa77a8d41f189b2bd4b9fdb26cb43d135b490dc056b133.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/tables/f2bfd518746c6cb322aa77a8d41f189b2bd4b9fdb26cb43d135b490dc056b133.jpg)

![fd7e6f233057f49f1c19c6d9a81525df5403bbc5ebb3e2ea05131d3969c00a56.jpg](../iclr_results/117_Proteina_%20Scaling%20Flow-based%20Protein%20Structure%20Generative%20Models/tables/fd7e6f233057f49f1c19c6d9a81525df5403bbc5ebb3e2ea05131d3969c00a56.jpg)

## REEF: Representation Encoding Fingerprints for Large Language Models


### Images

![13268f7c77dcdb82549910c02197ad5097736b533de4d022043458f75c62ba32.jpg](../iclr_results/118_REEF_%20Representation%20Encoding%20Fingerprints%20for%20Large%20Language%20Models/images/13268f7c77dcdb82549910c02197ad5097736b533de4d022043458f75c62ba32.jpg)

![609692949135009fbc389ce91b1c3e5aca9d7eea8b6b6098f0fadb8deda4125c.jpg](../iclr_results/118_REEF_%20Representation%20Encoding%20Fingerprints%20for%20Large%20Language%20Models/images/609692949135009fbc389ce91b1c3e5aca9d7eea8b6b6098f0fadb8deda4125c.jpg)

![7736360cb1c708ef4e8490c6086f94f7c75583d6000fdf7fc7184e4dac9e45d8.jpg](../iclr_results/118_REEF_%20Representation%20Encoding%20Fingerprints%20for%20Large%20Language%20Models/images/7736360cb1c708ef4e8490c6086f94f7c75583d6000fdf7fc7184e4dac9e45d8.jpg)

![8ab7ec422b7cea167f4019645e77288192546b8187dbcf07f66526f372c57502.jpg](../iclr_results/118_REEF_%20Representation%20Encoding%20Fingerprints%20for%20Large%20Language%20Models/images/8ab7ec422b7cea167f4019645e77288192546b8187dbcf07f66526f372c57502.jpg)

![a229663a328cb78e228cd9aac6d3400e393e8d272c932c83559ca7be15a73353.jpg](../iclr_results/118_REEF_%20Representation%20Encoding%20Fingerprints%20for%20Large%20Language%20Models/images/a229663a328cb78e228cd9aac6d3400e393e8d272c932c83559ca7be15a73353.jpg)

![c4bafdcb0b3687826ce63a3c5356e1cfbb615b2c1bc1c4ce58b1e722cdaf89a3.jpg](../iclr_results/118_REEF_%20Representation%20Encoding%20Fingerprints%20for%20Large%20Language%20Models/images/c4bafdcb0b3687826ce63a3c5356e1cfbb615b2c1bc1c4ce58b1e722cdaf89a3.jpg)

![d0d00449cdb7bb906a924a2296cd60b7ce8cf65e20ef160b9f4983d2f7c108dc.jpg](../iclr_results/118_REEF_%20Representation%20Encoding%20Fingerprints%20for%20Large%20Language%20Models/images/d0d00449cdb7bb906a924a2296cd60b7ce8cf65e20ef160b9f4983d2f7c108dc.jpg)

![d5e0387431254483dc59b9825364fe6098da8a9f2181d37e2d1ba50f845c260c.jpg](../iclr_results/118_REEF_%20Representation%20Encoding%20Fingerprints%20for%20Large%20Language%20Models/images/d5e0387431254483dc59b9825364fe6098da8a9f2181d37e2d1ba50f845c260c.jpg)

![f878810324df05d1970738c693fe2b091d1c53580dfd885a69639d27825aae24.jpg](../iclr_results/118_REEF_%20Representation%20Encoding%20Fingerprints%20for%20Large%20Language%20Models/images/f878810324df05d1970738c693fe2b091d1c53580dfd885a69639d27825aae24.jpg)

### Tables

![3bc37f9191923bc624e0cc647eeab4a0afa16b0d3b70f8751225a09d4f0bbd9c.jpg](../iclr_results/118_REEF_%20Representation%20Encoding%20Fingerprints%20for%20Large%20Language%20Models/tables/3bc37f9191923bc624e0cc647eeab4a0afa16b0d3b70f8751225a09d4f0bbd9c.jpg)

![3dc048b8c83fa3c359c11ce0afde354a7a60882bed6099005e4c040a629066ba.jpg](../iclr_results/118_REEF_%20Representation%20Encoding%20Fingerprints%20for%20Large%20Language%20Models/tables/3dc048b8c83fa3c359c11ce0afde354a7a60882bed6099005e4c040a629066ba.jpg)

![458526ac3e93d72f37a3e37ec2a6f97ac68463344d351b3afba34c0bd40d6743.jpg](../iclr_results/118_REEF_%20Representation%20Encoding%20Fingerprints%20for%20Large%20Language%20Models/tables/458526ac3e93d72f37a3e37ec2a6f97ac68463344d351b3afba34c0bd40d6743.jpg)

![574520ffd76d9d87cb865ad7cf5ed30f417ea361f34f7c19bcf737f23b70e195.jpg](../iclr_results/118_REEF_%20Representation%20Encoding%20Fingerprints%20for%20Large%20Language%20Models/tables/574520ffd76d9d87cb865ad7cf5ed30f417ea361f34f7c19bcf737f23b70e195.jpg)

![d6097b8f1da51c83532e76965c9415f654e2a5fe92240a6c68c99f70b5659451.jpg](../iclr_results/118_REEF_%20Representation%20Encoding%20Fingerprints%20for%20Large%20Language%20Models/tables/d6097b8f1da51c83532e76965c9415f654e2a5fe92240a6c68c99f70b5659451.jpg)

![ed703c8ce2b3511e9583e059f1864cb7f65783c62cb0f435261f13e784761742.jpg](../iclr_results/118_REEF_%20Representation%20Encoding%20Fingerprints%20for%20Large%20Language%20Models/tables/ed703c8ce2b3511e9583e059f1864cb7f65783c62cb0f435261f13e784761742.jpg)

![ef6fdc0764782c0eeeeaf2a90dbc8f3dcb57e246009a0e171a1a1ff26683cb39.jpg](../iclr_results/118_REEF_%20Representation%20Encoding%20Fingerprints%20for%20Large%20Language%20Models/tables/ef6fdc0764782c0eeeeaf2a90dbc8f3dcb57e246009a0e171a1a1ff26683cb39.jpg)

## A Decade's Battle on Dataset Bias: Are We There Yet?


### Images

![11dc30a0a82bf1d327df0c218dbeb12329575b05d55463fea936ffd52bca4bf3.jpg](../iclr_results/119_A%20Decade%27s%20Battle%20on%20Dataset%20Bias_%20Are%20We%20There%20Yet_/images/11dc30a0a82bf1d327df0c218dbeb12329575b05d55463fea936ffd52bca4bf3.jpg)

![5d4f0ab433cc260e175a0eedb9b05353ebc5d8d38a120540db80c7841dcd044e.jpg](../iclr_results/119_A%20Decade%27s%20Battle%20on%20Dataset%20Bias_%20Are%20We%20There%20Yet_/images/5d4f0ab433cc260e175a0eedb9b05353ebc5d8d38a120540db80c7841dcd044e.jpg)

![61c286cefb05962265973753e4f6aab47d820bb1db89cdc966ee0445f66a077d.jpg](../iclr_results/119_A%20Decade%27s%20Battle%20on%20Dataset%20Bias_%20Are%20We%20There%20Yet_/images/61c286cefb05962265973753e4f6aab47d820bb1db89cdc966ee0445f66a077d.jpg)

![c332c6f6630a7e463c0a36eb64231d45156620253558d7694f8153ff7efeb6ce.jpg](../iclr_results/119_A%20Decade%27s%20Battle%20on%20Dataset%20Bias_%20Are%20We%20There%20Yet_/images/c332c6f6630a7e463c0a36eb64231d45156620253558d7694f8153ff7efeb6ce.jpg)

![c8fff5177e4603c761ae2f5e3983c85f8112c57429dc06a274d74281ead6d06d.jpg](../iclr_results/119_A%20Decade%27s%20Battle%20on%20Dataset%20Bias_%20Are%20We%20There%20Yet_/images/c8fff5177e4603c761ae2f5e3983c85f8112c57429dc06a274d74281ead6d06d.jpg)

![d54cbc3b6240278797b76955c9a44c28409b8c38ec6b6c46c5fa4bc65ea09a05.jpg](../iclr_results/119_A%20Decade%27s%20Battle%20on%20Dataset%20Bias_%20Are%20We%20There%20Yet_/images/d54cbc3b6240278797b76955c9a44c28409b8c38ec6b6c46c5fa4bc65ea09a05.jpg)

![dcf70c24344bcd243757cc6d51f5bedc36e817fc20d0e4d149bbe5280a1e1eb9.jpg](../iclr_results/119_A%20Decade%27s%20Battle%20on%20Dataset%20Bias_%20Are%20We%20There%20Yet_/images/dcf70c24344bcd243757cc6d51f5bedc36e817fc20d0e4d149bbe5280a1e1eb9.jpg)

![e8fe0fb762d920f76ce0626419266f3d3c28e5746953e730bf68db09f6818dcc.jpg](../iclr_results/119_A%20Decade%27s%20Battle%20on%20Dataset%20Bias_%20Are%20We%20There%20Yet_/images/e8fe0fb762d920f76ce0626419266f3d3c28e5746953e730bf68db09f6818dcc.jpg)

### Tables

![117dde26c77e745ae7fd404c6830d88f69f62ba80257d4962201d02c05d89af1.jpg](../iclr_results/119_A%20Decade%27s%20Battle%20on%20Dataset%20Bias_%20Are%20We%20There%20Yet_/tables/117dde26c77e745ae7fd404c6830d88f69f62ba80257d4962201d02c05d89af1.jpg)

![1493abf85f045c22b0e6d912455602588e021d609e46ebb8b3b24932be1ddc6e.jpg](../iclr_results/119_A%20Decade%27s%20Battle%20on%20Dataset%20Bias_%20Are%20We%20There%20Yet_/tables/1493abf85f045c22b0e6d912455602588e021d609e46ebb8b3b24932be1ddc6e.jpg)

![62d7cd68f959bfa8ebc8bdba06fb2d7452b86e4e0b07bd2a544935a7f0539799.jpg](../iclr_results/119_A%20Decade%27s%20Battle%20on%20Dataset%20Bias_%20Are%20We%20There%20Yet_/tables/62d7cd68f959bfa8ebc8bdba06fb2d7452b86e4e0b07bd2a544935a7f0539799.jpg)

![748460629a280ece2523ec6df10c93154a7bf62f444347ca743253e5c333b6e8.jpg](../iclr_results/119_A%20Decade%27s%20Battle%20on%20Dataset%20Bias_%20Are%20We%20There%20Yet_/tables/748460629a280ece2523ec6df10c93154a7bf62f444347ca743253e5c333b6e8.jpg)

![937d5c712097a71c1f9110c9a9d89136a5ff323a14daeb7692bc914fa72981f4.jpg](../iclr_results/119_A%20Decade%27s%20Battle%20on%20Dataset%20Bias_%20Are%20We%20There%20Yet_/tables/937d5c712097a71c1f9110c9a9d89136a5ff323a14daeb7692bc914fa72981f4.jpg)

![984deb3ce5b7fc87541294c050f73dc670b78dc2e074916f0ea80432a428702c.jpg](../iclr_results/119_A%20Decade%27s%20Battle%20on%20Dataset%20Bias_%20Are%20We%20There%20Yet_/tables/984deb3ce5b7fc87541294c050f73dc670b78dc2e074916f0ea80432a428702c.jpg)

![bb5474239c9667caafe448849254d5aeac5adb27689021e19ee6c60ddf570607.jpg](../iclr_results/119_A%20Decade%27s%20Battle%20on%20Dataset%20Bias_%20Are%20We%20There%20Yet_/tables/bb5474239c9667caafe448849254d5aeac5adb27689021e19ee6c60ddf570607.jpg)

![d605f0c79b1ea8e9063150785fc435768f30ab49db97a74ee33c72ffcb7b038c.jpg](../iclr_results/119_A%20Decade%27s%20Battle%20on%20Dataset%20Bias_%20Are%20We%20There%20Yet_/tables/d605f0c79b1ea8e9063150785fc435768f30ab49db97a74ee33c72ffcb7b038c.jpg)

![f19f55ebd896a944d06a6b756229cfa62981fc2a29579deb883f9d0e68301f55.jpg](../iclr_results/119_A%20Decade%27s%20Battle%20on%20Dataset%20Bias_%20Are%20We%20There%20Yet_/tables/f19f55ebd896a944d06a6b756229cfa62981fc2a29579deb883f9d0e68301f55.jpg)

![f2df136a221243d6be02e05a0ba2c8ae02d01e7b8c73437cf0474551167610b5.jpg](../iclr_results/119_A%20Decade%27s%20Battle%20on%20Dataset%20Bias_%20Are%20We%20There%20Yet_/tables/f2df136a221243d6be02e05a0ba2c8ae02d01e7b8c73437cf0474551167610b5.jpg)

## Context-Parametric Inversion: Why Instruction Finetuning May Not Actually Improve Context Reliance


### Images

![138102ff1a409af893bad10b4eac5e070930e7edc91f235ecec34f375b707b54.jpg](../iclr_results/120_Context-Parametric%20Inversion_%20Why%20Instruction%20Finetuning%20May%20Not%20Actually%20Improve%20Context%20Reliance/images/138102ff1a409af893bad10b4eac5e070930e7edc91f235ecec34f375b707b54.jpg)

![1510696bdd1065a8c24854f808451f0dbf009510ea0a911109c7d75f0d3b463c.jpg](../iclr_results/120_Context-Parametric%20Inversion_%20Why%20Instruction%20Finetuning%20May%20Not%20Actually%20Improve%20Context%20Reliance/images/1510696bdd1065a8c24854f808451f0dbf009510ea0a911109c7d75f0d3b463c.jpg)

![1583590e02ebd7100bcf6cc09f1d2135a14ae3b96929f710eb4ca29cfc5c8a07.jpg](../iclr_results/120_Context-Parametric%20Inversion_%20Why%20Instruction%20Finetuning%20May%20Not%20Actually%20Improve%20Context%20Reliance/images/1583590e02ebd7100bcf6cc09f1d2135a14ae3b96929f710eb4ca29cfc5c8a07.jpg)

![1ca630b3715f54b76c7fbb80f9915ff74b86cfd3d024055da9765c93222cc1e7.jpg](../iclr_results/120_Context-Parametric%20Inversion_%20Why%20Instruction%20Finetuning%20May%20Not%20Actually%20Improve%20Context%20Reliance/images/1ca630b3715f54b76c7fbb80f9915ff74b86cfd3d024055da9765c93222cc1e7.jpg)

![39d10889d9d2e9404d1563416a441f9ebdbc3680b2522cddc588652f01ac7843.jpg](../iclr_results/120_Context-Parametric%20Inversion_%20Why%20Instruction%20Finetuning%20May%20Not%20Actually%20Improve%20Context%20Reliance/images/39d10889d9d2e9404d1563416a441f9ebdbc3680b2522cddc588652f01ac7843.jpg)

![3ba264f29d1eeb59a60c39b50f8400bd8a8d743681dbfd6e18a55573778d85e6.jpg](../iclr_results/120_Context-Parametric%20Inversion_%20Why%20Instruction%20Finetuning%20May%20Not%20Actually%20Improve%20Context%20Reliance/images/3ba264f29d1eeb59a60c39b50f8400bd8a8d743681dbfd6e18a55573778d85e6.jpg)

![50574966d30d2273510783c97f6e70ec3e4a4a500ac23922bf1f7f0a2ff548bf.jpg](../iclr_results/120_Context-Parametric%20Inversion_%20Why%20Instruction%20Finetuning%20May%20Not%20Actually%20Improve%20Context%20Reliance/images/50574966d30d2273510783c97f6e70ec3e4a4a500ac23922bf1f7f0a2ff548bf.jpg)

![5486163afc9a1215487a938df5e13a9ee976eb15daae7d4ebaf8106838e45a34.jpg](../iclr_results/120_Context-Parametric%20Inversion_%20Why%20Instruction%20Finetuning%20May%20Not%20Actually%20Improve%20Context%20Reliance/images/5486163afc9a1215487a938df5e13a9ee976eb15daae7d4ebaf8106838e45a34.jpg)

![5a6db5592a8a3bd1f8283ede5cfbf60f3d4ccf5e8ac241ec43ae9b2df765fc6c.jpg](../iclr_results/120_Context-Parametric%20Inversion_%20Why%20Instruction%20Finetuning%20May%20Not%20Actually%20Improve%20Context%20Reliance/images/5a6db5592a8a3bd1f8283ede5cfbf60f3d4ccf5e8ac241ec43ae9b2df765fc6c.jpg)

![9253b23429b082f85536ac5bc46f3cf265a61418f3f4986b366bd1639e583999.jpg](../iclr_results/120_Context-Parametric%20Inversion_%20Why%20Instruction%20Finetuning%20May%20Not%20Actually%20Improve%20Context%20Reliance/images/9253b23429b082f85536ac5bc46f3cf265a61418f3f4986b366bd1639e583999.jpg)

![99965af7b464423e592ca9ac31ad40bb63736aeb97db1c783cc4d24373b758e2.jpg](../iclr_results/120_Context-Parametric%20Inversion_%20Why%20Instruction%20Finetuning%20May%20Not%20Actually%20Improve%20Context%20Reliance/images/99965af7b464423e592ca9ac31ad40bb63736aeb97db1c783cc4d24373b758e2.jpg)

![b0b96e796937a135ebf0ec4cae5bb5c00e4f765c022c0fe9cc583de1e3b1c4d3.jpg](../iclr_results/120_Context-Parametric%20Inversion_%20Why%20Instruction%20Finetuning%20May%20Not%20Actually%20Improve%20Context%20Reliance/images/b0b96e796937a135ebf0ec4cae5bb5c00e4f765c022c0fe9cc583de1e3b1c4d3.jpg)

![c698bbad16380422c986f7b6ae640bdde4be1363d4856031b29d9f4ca9422207.jpg](../iclr_results/120_Context-Parametric%20Inversion_%20Why%20Instruction%20Finetuning%20May%20Not%20Actually%20Improve%20Context%20Reliance/images/c698bbad16380422c986f7b6ae640bdde4be1363d4856031b29d9f4ca9422207.jpg)

![d6b259451d87d2170505cb95b408a1a270e5dae1ac0952ee479ed61c360c24a6.jpg](../iclr_results/120_Context-Parametric%20Inversion_%20Why%20Instruction%20Finetuning%20May%20Not%20Actually%20Improve%20Context%20Reliance/images/d6b259451d87d2170505cb95b408a1a270e5dae1ac0952ee479ed61c360c24a6.jpg)

![fc33bb5d1f9f708fb0af1d1ca91959e391467abb1d4f1680b70766c42c1376c8.jpg](../iclr_results/120_Context-Parametric%20Inversion_%20Why%20Instruction%20Finetuning%20May%20Not%20Actually%20Improve%20Context%20Reliance/images/fc33bb5d1f9f708fb0af1d1ca91959e391467abb1d4f1680b70766c42c1376c8.jpg)

## Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model


### Images

![01812ba6db964fb99326003d848609c23adaa5a57522d6ca6bb07db131cfd370.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/01812ba6db964fb99326003d848609c23adaa5a57522d6ca6bb07db131cfd370.jpg)

![02e2bbd4906b8bf25026b3e4e85aaff49452bf4faf6f2563e70fb76da77ccd7a.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/02e2bbd4906b8bf25026b3e4e85aaff49452bf4faf6f2563e70fb76da77ccd7a.jpg)

![02e50b6826cf30786b71ae92cff883bc4dbfbcc3e9fb3bbf134ce2d32efbaaa2.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/02e50b6826cf30786b71ae92cff883bc4dbfbcc3e9fb3bbf134ce2d32efbaaa2.jpg)

![0987a78b1624d6a63754a8e7be1dbe28dc8f98b2f6269eaae4cc40bfd1c1a7e4.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/0987a78b1624d6a63754a8e7be1dbe28dc8f98b2f6269eaae4cc40bfd1c1a7e4.jpg)

![09bbbbf77c76d5072281a1708e962659a24502b043621dd714b6b6cfbba5a5cd.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/09bbbbf77c76d5072281a1708e962659a24502b043621dd714b6b6cfbba5a5cd.jpg)

![0c0f1039127dd9ffbc5cd3833a7c66edae732536cce3ccdd217fa38e2554c0cb.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/0c0f1039127dd9ffbc5cd3833a7c66edae732536cce3ccdd217fa38e2554c0cb.jpg)

![0daefe6c38986c50dacb4bb9902488fd234e918ba19e602b575b9d978dad1a2a.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/0daefe6c38986c50dacb4bb9902488fd234e918ba19e602b575b9d978dad1a2a.jpg)

![16bcab982b80172eac796ece10ef72bb4cb34fa84c4b8be33bd7888fbe667a5e.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/16bcab982b80172eac796ece10ef72bb4cb34fa84c4b8be33bd7888fbe667a5e.jpg)

![1ece28cc676eab362a01f8abe9d3d129419d2ffb80b1cd77d272d65051831cba.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/1ece28cc676eab362a01f8abe9d3d129419d2ffb80b1cd77d272d65051831cba.jpg)

![2876b09a1c6b1751fb795114c64e985624e064e28753fda8435c6bf762ab87af.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/2876b09a1c6b1751fb795114c64e985624e064e28753fda8435c6bf762ab87af.jpg)

![2a21db66d19b20c0a1f4265255c7a3f22191c01f38cceb6e7da28761ca57457f.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/2a21db66d19b20c0a1f4265255c7a3f22191c01f38cceb6e7da28761ca57457f.jpg)

![369fe01b0f5755abe42450c4e1f7e97c21ee53b0dae75acd194ca0c11ab7ae85.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/369fe01b0f5755abe42450c4e1f7e97c21ee53b0dae75acd194ca0c11ab7ae85.jpg)

![36a5effa7e3563c2eaa0235480fa49c67d1919fd66bbe0794aff9a2d1f09edb6.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/36a5effa7e3563c2eaa0235480fa49c67d1919fd66bbe0794aff9a2d1f09edb6.jpg)

![37471a6932a5600cbf8f45b012c4b17fed49e5009e9e4e2dfd870eb4e6e4dae1.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/37471a6932a5600cbf8f45b012c4b17fed49e5009e9e4e2dfd870eb4e6e4dae1.jpg)

![39bf4b9b6928e7d2d3de1f8b8e65d314d8ae9353276f7b93ab78c046d0fc15ad.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/39bf4b9b6928e7d2d3de1f8b8e65d314d8ae9353276f7b93ab78c046d0fc15ad.jpg)

![39ea55fca031310441236f8e9d5c2342532564dab0613e9e5879933e7998072b.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/39ea55fca031310441236f8e9d5c2342532564dab0613e9e5879933e7998072b.jpg)

![39f3eb0a4da3c0267218456af526d35e4ea612a586ba6763aecb2d489931df3c.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/39f3eb0a4da3c0267218456af526d35e4ea612a586ba6763aecb2d489931df3c.jpg)

![3f228246cd5185b2b65d5ff2466e6fd1580bc83f62150035ea193712561459d2.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/3f228246cd5185b2b65d5ff2466e6fd1580bc83f62150035ea193712561459d2.jpg)

![41c7ad2ca4c52a06379f93dc527910a65296034cddff46ade8899fa3a4b0582d.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/41c7ad2ca4c52a06379f93dc527910a65296034cddff46ade8899fa3a4b0582d.jpg)

![42c85d8b98eee853188e846ba22f5409f39b0c7ce18cf73cb64990b4f34b43c3.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/42c85d8b98eee853188e846ba22f5409f39b0c7ce18cf73cb64990b4f34b43c3.jpg)

![4ae604ef2d689d5e6be294d582ccfea5d76bef2c207dbeb0dc638f2c57f7b6fc.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/4ae604ef2d689d5e6be294d582ccfea5d76bef2c207dbeb0dc638f2c57f7b6fc.jpg)

![50e13ef6d45617e9e9647d2bebff1b132c5ff6a5e068419efb8eb7a335742831.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/50e13ef6d45617e9e9647d2bebff1b132c5ff6a5e068419efb8eb7a335742831.jpg)

![529f00c0fdcf41a6b9e76bb9a1f3e0c7642d921801d07c4b2244d996eb335d45.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/529f00c0fdcf41a6b9e76bb9a1f3e0c7642d921801d07c4b2244d996eb335d45.jpg)

![54552dfea8f0ef159f1389309d41af5a9d822a6be5cb90da7815a9ddc5ccbeb8.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/54552dfea8f0ef159f1389309d41af5a9d822a6be5cb90da7815a9ddc5ccbeb8.jpg)

![54fd6eafa8b64b21845700e83e9f1dd90e764106eef8e05d4435e376d51142ff.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/54fd6eafa8b64b21845700e83e9f1dd90e764106eef8e05d4435e376d51142ff.jpg)

![55fbf9f651adf5a3ea57bd48877dee2af3970d7cc38b8e69ec1d082348425dd0.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/55fbf9f651adf5a3ea57bd48877dee2af3970d7cc38b8e69ec1d082348425dd0.jpg)

![5e718fa2071331a6b50b00120d7ef209cd9701cfde8901e01b305f8734b61a12.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/5e718fa2071331a6b50b00120d7ef209cd9701cfde8901e01b305f8734b61a12.jpg)

![6216d7a0f87d2a035f4e051204342b08fc6b592c0e5a10b3ef813f1cb632b095.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/6216d7a0f87d2a035f4e051204342b08fc6b592c0e5a10b3ef813f1cb632b095.jpg)

![6263b9acda93858562d5a38cae091e21612974923750da7852e928d4ce7e45e2.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/6263b9acda93858562d5a38cae091e21612974923750da7852e928d4ce7e45e2.jpg)

![6c1d0a0660766e6aa412adc8cac688cc1808c7c8aebd7546e148f461e784c4d7.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/6c1d0a0660766e6aa412adc8cac688cc1808c7c8aebd7546e148f461e784c4d7.jpg)

![6d87469b6e0b97d36a826b0dbd820528a1e18a6f26490740385a0a8b75307892.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/6d87469b6e0b97d36a826b0dbd820528a1e18a6f26490740385a0a8b75307892.jpg)

![6e9b0d05b972483b1355397a9a675c628ac1bbba0ce710312e876e5edccf7d5c.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/6e9b0d05b972483b1355397a9a675c628ac1bbba0ce710312e876e5edccf7d5c.jpg)

![7058d5908aff0eff9fd06f6562f47e12ba6da4877e39abd470987f1d6fa9d1d4.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/7058d5908aff0eff9fd06f6562f47e12ba6da4877e39abd470987f1d6fa9d1d4.jpg)

![70c00b06b7cac67b6c8b997ff57a68b69fc6298ba88620db1e39da0760f15e2b.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/70c00b06b7cac67b6c8b997ff57a68b69fc6298ba88620db1e39da0760f15e2b.jpg)

![71c41428d358231eaac44434e4dadcef170524120dd6bdb4d0ae729c4183fd2a.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/71c41428d358231eaac44434e4dadcef170524120dd6bdb4d0ae729c4183fd2a.jpg)

![74b2b962ff96cd993dfcd967e0d5637ec4a93d3dcc640d1a4277a5bf09f14405.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/74b2b962ff96cd993dfcd967e0d5637ec4a93d3dcc640d1a4277a5bf09f14405.jpg)

![7618fce750303719d7dfbb50a7b78766d6be2f3ea56147dec07dbfa24d54ca6d.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/7618fce750303719d7dfbb50a7b78766d6be2f3ea56147dec07dbfa24d54ca6d.jpg)

![7721cb907728f1720b6f4e35908cec9f140e8e10dba4b8b4fe4014d0a867befe.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/7721cb907728f1720b6f4e35908cec9f140e8e10dba4b8b4fe4014d0a867befe.jpg)

![7de0bdf07a284749db8f19d5bafb7f47edf26c78a85ecdd5dbaad9d90f315bf9.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/7de0bdf07a284749db8f19d5bafb7f47edf26c78a85ecdd5dbaad9d90f315bf9.jpg)

![7e9c8699007f640f19a501e756d75b1c295f2e103f682b708963d113162c4bd1.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/7e9c8699007f640f19a501e756d75b1c295f2e103f682b708963d113162c4bd1.jpg)

![804dbcacaa884e9524b7c33cc1d25bd2613936fbb9909a0b6a99b7cf32b0a0a2.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/804dbcacaa884e9524b7c33cc1d25bd2613936fbb9909a0b6a99b7cf32b0a0a2.jpg)

![83ba87c54378e184b6206f20496982dd27592d127359129b28b0b5517519f623.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/83ba87c54378e184b6206f20496982dd27592d127359129b28b0b5517519f623.jpg)

![8461b3252a4ded7cc7698adf88a5ad334a9d10884edb0b316fb749bcf01f58ca.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/8461b3252a4ded7cc7698adf88a5ad334a9d10884edb0b316fb749bcf01f58ca.jpg)

![8502e09800a6d272fbb6dd471c52a89f4dfdc5bb07ccc09ecef509c24374393b.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/8502e09800a6d272fbb6dd471c52a89f4dfdc5bb07ccc09ecef509c24374393b.jpg)

![8543a8a3ccabb1d4afc52c4573049ee778126b0153c711613f2d055c75a5e291.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/8543a8a3ccabb1d4afc52c4573049ee778126b0153c711613f2d055c75a5e291.jpg)

![924f4d0bb2cde66f63ed79ce33379907b736eca63717b17f6a20a8939f11f993.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/924f4d0bb2cde66f63ed79ce33379907b736eca63717b17f6a20a8939f11f993.jpg)

![a32e84876215869a79f6dda4e37d7eea0067e04ed6677a609197061cc81a99a0.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/a32e84876215869a79f6dda4e37d7eea0067e04ed6677a609197061cc81a99a0.jpg)

![a43f08c044927a70e0c33d13bf6014923fec6ba3e39f79ff56ef79f2c642b841.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/a43f08c044927a70e0c33d13bf6014923fec6ba3e39f79ff56ef79f2c642b841.jpg)

![bce26fe19283d601714b6a54951babec7eeed35cdc05ca0c7aa784561e1ee86a.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/bce26fe19283d601714b6a54951babec7eeed35cdc05ca0c7aa784561e1ee86a.jpg)

![c2967a5ea6927aa8a93aa8bdc911545784e3e0069f302456aab5f7bf2c536703.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/c2967a5ea6927aa8a93aa8bdc911545784e3e0069f302456aab5f7bf2c536703.jpg)

![c6e7a3d1d7cf47d610db13b2deb07424c770157c704f7889f678442d140d643d.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/c6e7a3d1d7cf47d610db13b2deb07424c770157c704f7889f678442d140d643d.jpg)

![c80f592981e26d73af03d832876fade9f674ff20b4e03a3d27517d33648b40d2.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/c80f592981e26d73af03d832876fade9f674ff20b4e03a3d27517d33648b40d2.jpg)

![cef8c2d36f831e07edc958fdb14187e1323f4de0380f0b3d6204aa07b978cf9c.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/cef8c2d36f831e07edc958fdb14187e1323f4de0380f0b3d6204aa07b978cf9c.jpg)

![cf9f09d38109c92d6a768d00d181532483c09265e6383169e9e92e431bf75882.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/cf9f09d38109c92d6a768d00d181532483c09265e6383169e9e92e431bf75882.jpg)

![da24a6c1484b0ab6f8b96e2de995794e797ef110ca5f59f716635d8fc192a0d5.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/da24a6c1484b0ab6f8b96e2de995794e797ef110ca5f59f716635d8fc192a0d5.jpg)

![dbd208c3ade1b2f0ea47fa4b6835ab5933c7380ee460630870aec9dfc0e5fdcd.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/dbd208c3ade1b2f0ea47fa4b6835ab5933c7380ee460630870aec9dfc0e5fdcd.jpg)

![de03067f3a9e715017ed0100b283b9d088b605bc6246ae0f5bcc869f74700e92.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/de03067f3a9e715017ed0100b283b9d088b605bc6246ae0f5bcc869f74700e92.jpg)

![de22ac4fc910fe2c6e1c2ae49ba2f05423c182ff9228632dadb0db0a1e7e8900.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/de22ac4fc910fe2c6e1c2ae49ba2f05423c182ff9228632dadb0db0a1e7e8900.jpg)

![e96d7f62f08d3d480e1ed49e6ab2ff03bbaa5bfff4db6bfd5a519d374fd39375.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/e96d7f62f08d3d480e1ed49e6ab2ff03bbaa5bfff4db6bfd5a519d374fd39375.jpg)

![f231ad00f49865b86a0d1a611e659c4e305a566f7c0db0725ffa45d531f58d56.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/f231ad00f49865b86a0d1a611e659c4e305a566f7c0db0725ffa45d531f58d56.jpg)

![fce9632faf1c87b58ffdb77052de9b212cdd708db2c5ac5d23a6e6f748d413e7.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/fce9632faf1c87b58ffdb77052de9b212cdd708db2c5ac5d23a6e6f748d413e7.jpg)

![fdb8f9bf0ecd558e89099e191b0a0709dc83bbfc6268f0a8878ed5da33aecb83.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/images/fdb8f9bf0ecd558e89099e191b0a0709dc83bbfc6268f0a8878ed5da33aecb83.jpg)

### Tables

![23a4309a3667a594808c325598a71d06d92110793bd344ad55958eaf14d0c555.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/tables/23a4309a3667a594808c325598a71d06d92110793bd344ad55958eaf14d0c555.jpg)

![3117c7f3e6848dd7a8ae9e96ea480bf9ce236c725d3d7b12d771b7e704a59d5a.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/tables/3117c7f3e6848dd7a8ae9e96ea480bf9ce236c725d3d7b12d771b7e704a59d5a.jpg)

![99687135a808d9eaa94cb9527a6d193a86749b8a626028c012d4fb334cf664c8.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/tables/99687135a808d9eaa94cb9527a6d193a86749b8a626028c012d4fb334cf664c8.jpg)

![ab9509ea0899a08a3e4caa28c7938d13b784c0d6d7e46a982800f2daa5660d49.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/tables/ab9509ea0899a08a3e4caa28c7938d13b784c0d6d7e46a982800f2daa5660d49.jpg)

![b881c3b8b41fbfb6243efd4e3f23aa08677a17990e23de5ba5cb2e6af101b901.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/tables/b881c3b8b41fbfb6243efd4e3f23aa08677a17990e23de5ba5cb2e6af101b901.jpg)

![bce550ae3cfecf7e73d00373e92449349f00a7f25ab447ad5ee8142fc39fee24.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/tables/bce550ae3cfecf7e73d00373e92449349f00a7f25ab447ad5ee8142fc39fee24.jpg)

![cbecf1040dcc6b59d6a1b8fc6df4fd48784a2e067eb3c8f9b9f80eb6aede8b56.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/tables/cbecf1040dcc6b59d6a1b8fc6df4fd48784a2e067eb3c8f9b9f80eb6aede8b56.jpg)

![db357f28a336afd1900566e35f90b8af7401f6ad30be96149119f88fcdff9c64.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/tables/db357f28a336afd1900566e35f90b8af7401f6ad30be96149119f88fcdff9c64.jpg)

![de669f8874a96a1d23b49ccbe4621e27f3e65c4c6c3bb719ec9e4963ecd17b50.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/tables/de669f8874a96a1d23b49ccbe4621e27f3e65c4c6c3bb719ec9e4963ecd17b50.jpg)

![f3c8cecdf03f256505d12007cbb06f2ae296da1364c510a5aeecc7771abf02d9.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/tables/f3c8cecdf03f256505d12007cbb06f2ae296da1364c510a5aeecc7771abf02d9.jpg)

![f562042acb531ba2d8efd123e72231ed13fef0d30dc009f13bbf704017525592.jpg](../iclr_results/121_Transfusion_%20Predict%20the%20Next%20Token%20and%20Diffuse%20Images%20with%20One%20Multi-Modal%20Model/tables/f562042acb531ba2d8efd123e72231ed13fef0d30dc009f13bbf704017525592.jpg)

## ECD: A Machine Learning Benchmark for Predicting Enhanced-Precision Electronic Charge Density in Crystalline Inorganic Materials


### Images

![0fd207f467a184d078d1ab54cd35b0cddac8557583d8308f37fdc41f6e6ff4dd.jpg](../iclr_results/122_ECD_%20A%20Machine%20Learning%20Benchmark%20for%20Predicting%20Enhanced-Precision%20Electronic%20Charge%20Density%20in%20Cry/images/0fd207f467a184d078d1ab54cd35b0cddac8557583d8308f37fdc41f6e6ff4dd.jpg)

![32421d2425fd03f19ffdb6f5a00a95eac6debd0c14081e729d5d86cd265f1e22.jpg](../iclr_results/122_ECD_%20A%20Machine%20Learning%20Benchmark%20for%20Predicting%20Enhanced-Precision%20Electronic%20Charge%20Density%20in%20Cry/images/32421d2425fd03f19ffdb6f5a00a95eac6debd0c14081e729d5d86cd265f1e22.jpg)

![58fe1595f1e5dace035baddb5ff373c4224b0c1c646ea3f4dfa097dd9bce22db.jpg](../iclr_results/122_ECD_%20A%20Machine%20Learning%20Benchmark%20for%20Predicting%20Enhanced-Precision%20Electronic%20Charge%20Density%20in%20Cry/images/58fe1595f1e5dace035baddb5ff373c4224b0c1c646ea3f4dfa097dd9bce22db.jpg)

![6fdd52400cd0edb4a7301bbcac8dd76aa8cfd516edf675868df084ee953e6270.jpg](../iclr_results/122_ECD_%20A%20Machine%20Learning%20Benchmark%20for%20Predicting%20Enhanced-Precision%20Electronic%20Charge%20Density%20in%20Cry/images/6fdd52400cd0edb4a7301bbcac8dd76aa8cfd516edf675868df084ee953e6270.jpg)

![87ef6645c5c61c96f881e5c982ebf2466da84267845489aa31760dbd0b58830d.jpg](../iclr_results/122_ECD_%20A%20Machine%20Learning%20Benchmark%20for%20Predicting%20Enhanced-Precision%20Electronic%20Charge%20Density%20in%20Cry/images/87ef6645c5c61c96f881e5c982ebf2466da84267845489aa31760dbd0b58830d.jpg)

![939c1d97f94bbf2824ecdfcb2367069542010d35f39118839bac7fa704e72418.jpg](../iclr_results/122_ECD_%20A%20Machine%20Learning%20Benchmark%20for%20Predicting%20Enhanced-Precision%20Electronic%20Charge%20Density%20in%20Cry/images/939c1d97f94bbf2824ecdfcb2367069542010d35f39118839bac7fa704e72418.jpg)

![eb365d2fd2b16c4cfb6ccdc9a1296767443ebe0a849ff4596677a2aadc016d42.jpg](../iclr_results/122_ECD_%20A%20Machine%20Learning%20Benchmark%20for%20Predicting%20Enhanced-Precision%20Electronic%20Charge%20Density%20in%20Cry/images/eb365d2fd2b16c4cfb6ccdc9a1296767443ebe0a849ff4596677a2aadc016d42.jpg)

![ff6bdc54684246dfdd28395ede276a76279c580f7deeb5be247fa4dbe069d9ce.jpg](../iclr_results/122_ECD_%20A%20Machine%20Learning%20Benchmark%20for%20Predicting%20Enhanced-Precision%20Electronic%20Charge%20Density%20in%20Cry/images/ff6bdc54684246dfdd28395ede276a76279c580f7deeb5be247fa4dbe069d9ce.jpg)

### Tables

![223f8c42800e2c2051c3b1eaee6a077eb5410d73575705d2a2ed537fb33abd26.jpg](../iclr_results/122_ECD_%20A%20Machine%20Learning%20Benchmark%20for%20Predicting%20Enhanced-Precision%20Electronic%20Charge%20Density%20in%20Cry/tables/223f8c42800e2c2051c3b1eaee6a077eb5410d73575705d2a2ed537fb33abd26.jpg)

![a9c76ba59adad1970d21daa043f8bdaaa9dcadbade010db469f9d2f3bb813cbe.jpg](../iclr_results/122_ECD_%20A%20Machine%20Learning%20Benchmark%20for%20Predicting%20Enhanced-Precision%20Electronic%20Charge%20Density%20in%20Cry/tables/a9c76ba59adad1970d21daa043f8bdaaa9dcadbade010db469f9d2f3bb813cbe.jpg)

![b011b4447e39ae855edcf83f8fc3fab8919a50290315cc3c1eb4dd2fb1f7df51.jpg](../iclr_results/122_ECD_%20A%20Machine%20Learning%20Benchmark%20for%20Predicting%20Enhanced-Precision%20Electronic%20Charge%20Density%20in%20Cry/tables/b011b4447e39ae855edcf83f8fc3fab8919a50290315cc3c1eb4dd2fb1f7df51.jpg)

![ba8e4ce857ee2288167014610c57af20b0f22c114703a8447a170a3013b1e41b.jpg](../iclr_results/122_ECD_%20A%20Machine%20Learning%20Benchmark%20for%20Predicting%20Enhanced-Precision%20Electronic%20Charge%20Density%20in%20Cry/tables/ba8e4ce857ee2288167014610c57af20b0f22c114703a8447a170a3013b1e41b.jpg)

![d055ebf9b67e9a838c6a0ce2a07b4c4cf157ef0dc17d4ba4d3d8dead00ed277c.jpg](../iclr_results/122_ECD_%20A%20Machine%20Learning%20Benchmark%20for%20Predicting%20Enhanced-Precision%20Electronic%20Charge%20Density%20in%20Cry/tables/d055ebf9b67e9a838c6a0ce2a07b4c4cf157ef0dc17d4ba4d3d8dead00ed277c.jpg)

## Generator Matching: Generative modeling with arbitrary Markov processes


### Images

![016f7a376531ba3f5bc52c49e404c9a0629b9594b2c1fae29c1ecb9bbf72f1f4.jpg](../iclr_results/123_Generator%20Matching_%20Generative%20modeling%20with%20arbitrary%20Markov%20processes/images/016f7a376531ba3f5bc52c49e404c9a0629b9594b2c1fae29c1ecb9bbf72f1f4.jpg)

![072390c4c124d9a45875d942e4d08c592285028f0906921a160295ec12cc822c.jpg](../iclr_results/123_Generator%20Matching_%20Generative%20modeling%20with%20arbitrary%20Markov%20processes/images/072390c4c124d9a45875d942e4d08c592285028f0906921a160295ec12cc822c.jpg)

![55ffc091ffd90bc8e98cd0ee2e04aa48497ce886ecbcd0afc9c94ab26e70ced4.jpg](../iclr_results/123_Generator%20Matching_%20Generative%20modeling%20with%20arbitrary%20Markov%20processes/images/55ffc091ffd90bc8e98cd0ee2e04aa48497ce886ecbcd0afc9c94ab26e70ced4.jpg)

![6eb251288c95da25a6c8521ec6807798b1c9bad13decd11f19f2797e8af5c6c5.jpg](../iclr_results/123_Generator%20Matching_%20Generative%20modeling%20with%20arbitrary%20Markov%20processes/images/6eb251288c95da25a6c8521ec6807798b1c9bad13decd11f19f2797e8af5c6c5.jpg)

![7855599d1e2e8ac7afc59a995d9f0ee28eca19a0ed2df574a70b8da002bdcadd.jpg](../iclr_results/123_Generator%20Matching_%20Generative%20modeling%20with%20arbitrary%20Markov%20processes/images/7855599d1e2e8ac7afc59a995d9f0ee28eca19a0ed2df574a70b8da002bdcadd.jpg)

![ae52605edbfc21d013791b6c6a91d47eb32ae2a22585dbdc74af3e42e28f829b.jpg](../iclr_results/123_Generator%20Matching_%20Generative%20modeling%20with%20arbitrary%20Markov%20processes/images/ae52605edbfc21d013791b6c6a91d47eb32ae2a22585dbdc74af3e42e28f829b.jpg)

![b392e9539ecd6e3735396b24a5564e9ee9604fcec69cee3e8118ed3a2159084f.jpg](../iclr_results/123_Generator%20Matching_%20Generative%20modeling%20with%20arbitrary%20Markov%20processes/images/b392e9539ecd6e3735396b24a5564e9ee9604fcec69cee3e8118ed3a2159084f.jpg)

![b811c9ef1f7583c0ee5ce9329f31968e78b09e327731f538eed9bb290b758791.jpg](../iclr_results/123_Generator%20Matching_%20Generative%20modeling%20with%20arbitrary%20Markov%20processes/images/b811c9ef1f7583c0ee5ce9329f31968e78b09e327731f538eed9bb290b758791.jpg)

![c2bd8b0dca82dcd45e22f526326b4112ceb992b2950e6c8c1064e17fd823b7a4.jpg](../iclr_results/123_Generator%20Matching_%20Generative%20modeling%20with%20arbitrary%20Markov%20processes/images/c2bd8b0dca82dcd45e22f526326b4112ceb992b2950e6c8c1064e17fd823b7a4.jpg)

![ebd0819095162c181e389ce86e1ba969e68cf410ad18127325161d55a39a2e21.jpg](../iclr_results/123_Generator%20Matching_%20Generative%20modeling%20with%20arbitrary%20Markov%20processes/images/ebd0819095162c181e389ce86e1ba969e68cf410ad18127325161d55a39a2e21.jpg)

### Tables

![2f30c886a1873ee05dceac8ba0024410e07c564bd8c513ef64e6ad7645875bc7.jpg](../iclr_results/123_Generator%20Matching_%20Generative%20modeling%20with%20arbitrary%20Markov%20processes/tables/2f30c886a1873ee05dceac8ba0024410e07c564bd8c513ef64e6ad7645875bc7.jpg)

![572fb11d2f357804c9c1420379b090d28b1b437e0e9cbd7addb8dc6aebbb6583.jpg](../iclr_results/123_Generator%20Matching_%20Generative%20modeling%20with%20arbitrary%20Markov%20processes/tables/572fb11d2f357804c9c1420379b090d28b1b437e0e9cbd7addb8dc6aebbb6583.jpg)

![bd151aa71653c2b0d3b59561d0a5555adeca19e47c867b4887ab46596a55f6d8.jpg](../iclr_results/123_Generator%20Matching_%20Generative%20modeling%20with%20arbitrary%20Markov%20processes/tables/bd151aa71653c2b0d3b59561d0a5555adeca19e47c867b4887ab46596a55f6d8.jpg)

![c14da62aef1f550bfbddd2b8131975c6082481802eeb6a3f51a31e4410d0281a.jpg](../iclr_results/123_Generator%20Matching_%20Generative%20modeling%20with%20arbitrary%20Markov%20processes/tables/c14da62aef1f550bfbddd2b8131975c6082481802eeb6a3f51a31e4410d0281a.jpg)

![d065a373fb03ef121d3f6fa26283f04be228c72dc81fb9c412704b49b1c8491d.jpg](../iclr_results/123_Generator%20Matching_%20Generative%20modeling%20with%20arbitrary%20Markov%20processes/tables/d065a373fb03ef121d3f6fa26283f04be228c72dc81fb9c412704b49b1c8491d.jpg)

![db2160147d88a48a54778fd4e8a71cc941ed260113d00400a3693d6ea8465b6a.jpg](../iclr_results/123_Generator%20Matching_%20Generative%20modeling%20with%20arbitrary%20Markov%20processes/tables/db2160147d88a48a54778fd4e8a71cc941ed260113d00400a3693d6ea8465b6a.jpg)

## Brain Bandit: A Biologically Grounded Neural Network for Efficient Control of Exploration


### Images

![0921dd9afc4e680fe95cfeeb8c24565bb9fd5b43a4e1a5e11c4e5c4a6571aec4.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/0921dd9afc4e680fe95cfeeb8c24565bb9fd5b43a4e1a5e11c4e5c4a6571aec4.jpg)

![241df4abd3f08f4a96eb79ba39bd06dce4b1fb6ecb4098a3f5800935157d432e.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/241df4abd3f08f4a96eb79ba39bd06dce4b1fb6ecb4098a3f5800935157d432e.jpg)

![2bfe8629db8434ce23f5c8b711aa3f9fa7c3b29c54971acd850443580f03600f.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/2bfe8629db8434ce23f5c8b711aa3f9fa7c3b29c54971acd850443580f03600f.jpg)

![327bbd45a82499c2934aa80f1338b1375ea7843f313dfaa124bb9fcd06dc307b.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/327bbd45a82499c2934aa80f1338b1375ea7843f313dfaa124bb9fcd06dc307b.jpg)

![3b532979bccd49a6860596a16a6483135b882220bd1860af6b5725cc911f3c4d.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/3b532979bccd49a6860596a16a6483135b882220bd1860af6b5725cc911f3c4d.jpg)

![5211494861ce4468838b31fce828330303d4e28fcfd3361be42d4d704dfe038b.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/5211494861ce4468838b31fce828330303d4e28fcfd3361be42d4d704dfe038b.jpg)

![54294865fe41ebdd71216212153fa07dc831a0c74fe5ddb22e3857f458d4baf1.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/54294865fe41ebdd71216212153fa07dc831a0c74fe5ddb22e3857f458d4baf1.jpg)

![74e237c46e5def4103b83e6d11ba8622cf211660b622480dfb1db4a71ca27ee3.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/74e237c46e5def4103b83e6d11ba8622cf211660b622480dfb1db4a71ca27ee3.jpg)

![82c6710c854b4d216c2292395ed23856a10c2ffea22cf7ffdcb33cc90d62d5cd.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/82c6710c854b4d216c2292395ed23856a10c2ffea22cf7ffdcb33cc90d62d5cd.jpg)

![8cd4f8beabd966842687f00f7cf2e1dc09604a1f6d081b19f5696d6ec3004935.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/8cd4f8beabd966842687f00f7cf2e1dc09604a1f6d081b19f5696d6ec3004935.jpg)

![8f095d2d1e112d4dadc5af5dc254b7a779103c4f4890d05ef2f61f707579d1de.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/8f095d2d1e112d4dadc5af5dc254b7a779103c4f4890d05ef2f61f707579d1de.jpg)

![9e0d0781624e25c1d5a31cbac152e771b7c51978db7a409581d58d54eb94790d.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/9e0d0781624e25c1d5a31cbac152e771b7c51978db7a409581d58d54eb94790d.jpg)

![a0519cc6abce02a76a59204e5b8e1e8a60ace2e34c5ac45cf79bd8a2b587dc9f.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/a0519cc6abce02a76a59204e5b8e1e8a60ace2e34c5ac45cf79bd8a2b587dc9f.jpg)

![a28d1c3e489ce07cc2b4806372e90babbe998d9d2416045d1bd5306b252ed721.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/a28d1c3e489ce07cc2b4806372e90babbe998d9d2416045d1bd5306b252ed721.jpg)

![a80e4a56f8ce2f5e1e5e426b8f249c0b7f2ef6ca0a50e0b6028ad9c6549e6bee.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/a80e4a56f8ce2f5e1e5e426b8f249c0b7f2ef6ca0a50e0b6028ad9c6549e6bee.jpg)

![ae6e11203e7e3d59fe68e617e1eef634de09e3cae1489725fd7035b201e152ff.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/ae6e11203e7e3d59fe68e617e1eef634de09e3cae1489725fd7035b201e152ff.jpg)

![b34daa38f438a08d198e1266784643ae11b7e767e3515e58f8396040644a2c41.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/b34daa38f438a08d198e1266784643ae11b7e767e3515e58f8396040644a2c41.jpg)

![b8702f572bddf4196fe0557a1ba30f2ea9533d919bb5366e7b750282288c919b.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/b8702f572bddf4196fe0557a1ba30f2ea9533d919bb5366e7b750282288c919b.jpg)

![b88ef20827503465fe2545258365267952d3eb3a5fe5d92234ce1c1fc855ad1a.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/b88ef20827503465fe2545258365267952d3eb3a5fe5d92234ce1c1fc855ad1a.jpg)

![be98cd1aae75abef0022f9384ee6b288b1eb298c737c1f5f3d215a59b457fedc.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/be98cd1aae75abef0022f9384ee6b288b1eb298c737c1f5f3d215a59b457fedc.jpg)

![c69bc66fb9d603c9c1ae432cee5ed060539957b708a891b47c333908751cb0c9.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/c69bc66fb9d603c9c1ae432cee5ed060539957b708a891b47c333908751cb0c9.jpg)

![cb74195aff8474b40726e39fad82cad19ad1de1387f0f4aecfb2b09f9140dd57.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/cb74195aff8474b40726e39fad82cad19ad1de1387f0f4aecfb2b09f9140dd57.jpg)

![ea78f684cb576130d32a01be18173b3a3a6c794f33354f597fdc1ab90fe2596f.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/ea78f684cb576130d32a01be18173b3a3a6c794f33354f597fdc1ab90fe2596f.jpg)

![ef8388bc8c8fa23891a493545c075655713fccb19231c6576525dacb7a5325ae.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/ef8388bc8c8fa23891a493545c075655713fccb19231c6576525dacb7a5325ae.jpg)

![f7059e421a03cd5c69624dcb606151aa016179d94cd6657bc22a924e551178a1.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/f7059e421a03cd5c69624dcb606151aa016179d94cd6657bc22a924e551178a1.jpg)

![f96612fa5b55dc6fdf29cd93df54d1bd6480643b109bb0772b9b50e0e030d214.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/images/f96612fa5b55dc6fdf29cd93df54d1bd6480643b109bb0772b9b50e0e030d214.jpg)

### Tables

![5a7d86e1b5ab133577b872448c0c6bf87f3f0522d8b98718e15393d365d37298.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/tables/5a7d86e1b5ab133577b872448c0c6bf87f3f0522d8b98718e15393d365d37298.jpg)

![5d12f664eab65b9fa2dca4c5260b1c6319bfe3ed38369591d001482662e269ca.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/tables/5d12f664eab65b9fa2dca4c5260b1c6319bfe3ed38369591d001482662e269ca.jpg)

![cac17764658f37d43c2203481b71ff995e0456665865086cfba462cb4e1f76dd.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/tables/cac17764658f37d43c2203481b71ff995e0456665865086cfba462cb4e1f76dd.jpg)

![d3f858ecac1a5fd6a9d757212ae32b7548551abecd4aba686bf37b46c1231a1d.jpg](../iclr_results/124_Brain%20Bandit_%20A%20Biologically%20Grounded%20Neural%20Network%20for%20Efficient%20Control%20of%20Exploration/tables/d3f858ecac1a5fd6a9d757212ae32b7548551abecd4aba686bf37b46c1231a1d.jpg)

## Do LLMs Recognize Your Preferences? Evaluating Personalized Preference Following in LLMs


### Images

![1092a1d059b5b0afc876ffad42846b56a9ce812871bf5eb00e3260d7e13584fa.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/images/1092a1d059b5b0afc876ffad42846b56a9ce812871bf5eb00e3260d7e13584fa.jpg)

![10d12aa57a2851108308c71e1591f37588313aee6d68c1c423f8b5195d023c7e.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/images/10d12aa57a2851108308c71e1591f37588313aee6d68c1c423f8b5195d023c7e.jpg)

![2643fd1d8823552e2e5559d82b0f57e805b8a72e39b6dd21d2350be9b5eb14ba.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/images/2643fd1d8823552e2e5559d82b0f57e805b8a72e39b6dd21d2350be9b5eb14ba.jpg)

![388c74a829d44db3a71970ef2333013bd8390907604a78557bac861e957db9da.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/images/388c74a829d44db3a71970ef2333013bd8390907604a78557bac861e957db9da.jpg)

![45cec32eb484a9d0893b0a6c674a2289e623f29a9b19c917097e3f0e979b1276.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/images/45cec32eb484a9d0893b0a6c674a2289e623f29a9b19c917097e3f0e979b1276.jpg)

![57c28a36475903ca2e55e63d72e87e87aa15d19ad034df26bfe2e6e9be63d5af.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/images/57c28a36475903ca2e55e63d72e87e87aa15d19ad034df26bfe2e6e9be63d5af.jpg)

![5b64f9237ae59f1d5262f8dbd241df1c5ec7b489d2c3e2b2f5c4faf5bfb7c350.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/images/5b64f9237ae59f1d5262f8dbd241df1c5ec7b489d2c3e2b2f5c4faf5bfb7c350.jpg)

![62dc6095b20cca2b6c1caca3b67ba2b4f6ddf82fda6616c590932d74bd1d9abc.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/images/62dc6095b20cca2b6c1caca3b67ba2b4f6ddf82fda6616c590932d74bd1d9abc.jpg)

![6cce4f03e79e71ce96fc13a028bdf8cbd6a8dd37cac63284f3747d276ab8a4e0.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/images/6cce4f03e79e71ce96fc13a028bdf8cbd6a8dd37cac63284f3747d276ab8a4e0.jpg)

![7398a1fbaa3374b2b99eeb6d9f5ffc695d60cc2107ac1c34eb9653ddfc05de10.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/images/7398a1fbaa3374b2b99eeb6d9f5ffc695d60cc2107ac1c34eb9653ddfc05de10.jpg)

![8400d7b2f66707cf41b19886b47f31ea00ac56c96c9e667ac4c22805d8108301.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/images/8400d7b2f66707cf41b19886b47f31ea00ac56c96c9e667ac4c22805d8108301.jpg)

![8dd4dace66ddb9283e95c4bba7f770c4a9974ddde4c6df2933f4eef58dd8b5e9.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/images/8dd4dace66ddb9283e95c4bba7f770c4a9974ddde4c6df2933f4eef58dd8b5e9.jpg)

![8f3dc5361ec6222c09ce4f7b91f817d822f80aa220445b7b82bf38cf330c16f4.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/images/8f3dc5361ec6222c09ce4f7b91f817d822f80aa220445b7b82bf38cf330c16f4.jpg)

![90a3341431b96f1049c897fe75bd629d29259c9b71effa565892e4056345db38.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/images/90a3341431b96f1049c897fe75bd629d29259c9b71effa565892e4056345db38.jpg)

![a4559a5b5062d01a1b6ec6d4cc6d3e5c5c4f059fac3001108db133d3787ea479.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/images/a4559a5b5062d01a1b6ec6d4cc6d3e5c5c4f059fac3001108db133d3787ea479.jpg)

![a8f2a3ca576b7c31942b129d4364fe14c0a8f33be376714a760977de6b0e4712.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/images/a8f2a3ca576b7c31942b129d4364fe14c0a8f33be376714a760977de6b0e4712.jpg)

![ab172286445835f9b169a5d6ac3e816be52e7a9633d12c1ba5a29f4a76740781.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/images/ab172286445835f9b169a5d6ac3e816be52e7a9633d12c1ba5a29f4a76740781.jpg)

![b0f46f45b1f747249b36791b4979d363715629267e39181d2fac978777cf0628.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/images/b0f46f45b1f747249b36791b4979d363715629267e39181d2fac978777cf0628.jpg)

![c9855012044c6871287581a9302e75e7e629ec95eb1a6b882e8ed2b6cf386601.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/images/c9855012044c6871287581a9302e75e7e629ec95eb1a6b882e8ed2b6cf386601.jpg)

![d41902339458b92641ee72c0778f0609fca574d14a834aa9f88563974f61d2f2.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/images/d41902339458b92641ee72c0778f0609fca574d14a834aa9f88563974f61d2f2.jpg)

![e6374bc250ad8c4d910fa5059b8daa9d992bd06b2b64f39156edd9b1c331ad30.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/images/e6374bc250ad8c4d910fa5059b8daa9d992bd06b2b64f39156edd9b1c331ad30.jpg)

![f29e4c2d3d0d85af22408fa6f7327ade5dbc4368ce5c9741e516351232bb9ed5.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/images/f29e4c2d3d0d85af22408fa6f7327ade5dbc4368ce5c9741e516351232bb9ed5.jpg)

### Tables

![041b1c77dbbf3dd076094d892b15780387c9d4aa5b2556f68e9a6c1d630de049.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/tables/041b1c77dbbf3dd076094d892b15780387c9d4aa5b2556f68e9a6c1d630de049.jpg)

![095d1222d8f2e3517c68aab5bda42f6936951274b322100d59d502b9e29ce7d3.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/tables/095d1222d8f2e3517c68aab5bda42f6936951274b322100d59d502b9e29ce7d3.jpg)

![10cfb450270fdd36af211993dee0a1a4e803e347aae973de75724fc8e75ef214.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/tables/10cfb450270fdd36af211993dee0a1a4e803e347aae973de75724fc8e75ef214.jpg)

![124796e66fe256229b2fcc22b6eca407418b69171181774cc86f7b3fb3f601d3.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/tables/124796e66fe256229b2fcc22b6eca407418b69171181774cc86f7b3fb3f601d3.jpg)

![140f2bccee0c44846d40592e51f394bdf29385fa3e2c457c61009bdb0381ffdc.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/tables/140f2bccee0c44846d40592e51f394bdf29385fa3e2c457c61009bdb0381ffdc.jpg)

![1f99e1d47a768f0fd0b706c0764571a2913ceea0fdfbfc6e3a7d3b3de47b83ef.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/tables/1f99e1d47a768f0fd0b706c0764571a2913ceea0fdfbfc6e3a7d3b3de47b83ef.jpg)

![2a6d3d8c530b0f849bd53c83b1cac8647574d957dbebf64dd6f6ddb86ddf721e.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/tables/2a6d3d8c530b0f849bd53c83b1cac8647574d957dbebf64dd6f6ddb86ddf721e.jpg)

![3775c58bc3225d43a2541247dc091900fc6a19363d839be079f03cd6028faa6d.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/tables/3775c58bc3225d43a2541247dc091900fc6a19363d839be079f03cd6028faa6d.jpg)

![58216a5112734f8a28eeeb34de6d31f17d6e56f901e4f5f64958a10fa68f878c.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/tables/58216a5112734f8a28eeeb34de6d31f17d6e56f901e4f5f64958a10fa68f878c.jpg)

![6a9d48306ecc6798467407c0f8a88fdfd31687d605574a14410d3b623edc4629.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/tables/6a9d48306ecc6798467407c0f8a88fdfd31687d605574a14410d3b623edc4629.jpg)

![76e42eea218ea305c5d4ec2002584f448b09cda441f6141be7fb09ee98f8d38e.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/tables/76e42eea218ea305c5d4ec2002584f448b09cda441f6141be7fb09ee98f8d38e.jpg)

![858e2761f3ca27d0b5994fefa7b8dc522d354942514b119b077f2cdab40e2e78.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/tables/858e2761f3ca27d0b5994fefa7b8dc522d354942514b119b077f2cdab40e2e78.jpg)

![974c99fa0aafd72ea9cb0a4cfd70356067444ac9272dded387324b3229f501e2.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/tables/974c99fa0aafd72ea9cb0a4cfd70356067444ac9272dded387324b3229f501e2.jpg)

![9c42b9b8efb2dc71eeab60bf1f09755c5aeea2264fcde995c86a7d054c434272.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/tables/9c42b9b8efb2dc71eeab60bf1f09755c5aeea2264fcde995c86a7d054c434272.jpg)

![a2fa82e68ba96040697b987e2ae6f7438c967fa50b027aa600baf06236dae7ac.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/tables/a2fa82e68ba96040697b987e2ae6f7438c967fa50b027aa600baf06236dae7ac.jpg)

![ac359e373d54c06f83cb306e03ef4935cc76d7874b8dc46fcb3b2f19ff9e37e1.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/tables/ac359e373d54c06f83cb306e03ef4935cc76d7874b8dc46fcb3b2f19ff9e37e1.jpg)

![b66b8cd11a021ecc7bb582b68fc14e28fcea6cf0dba2f451317359aabb680986.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/tables/b66b8cd11a021ecc7bb582b68fc14e28fcea6cf0dba2f451317359aabb680986.jpg)

![c1304b53fa4723c0d316d973c9bcdbbd055a7f3d61e17c9dd2c6d54c2a62c17a.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/tables/c1304b53fa4723c0d316d973c9bcdbbd055a7f3d61e17c9dd2c6d54c2a62c17a.jpg)

![d27d240a1f29c0357f56bd246ee2465ffc5b77a4c9d3d79db51f8a168d99e727.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/tables/d27d240a1f29c0357f56bd246ee2465ffc5b77a4c9d3d79db51f8a168d99e727.jpg)

![e88a43c67cc44059e4a732d276d870c8071db05deeac7ce009392ba2c876c9d4.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/tables/e88a43c67cc44059e4a732d276d870c8071db05deeac7ce009392ba2c876c9d4.jpg)

![efb0c4f31096ef27f19ab415e95fd85b6d8a5d93c90833cbf54e01e29e46babf.jpg](../iclr_results/125_Do%20LLMs%20Recognize%20Your%20Preferences_%20Evaluating%20Personalized%20Preference%20Following%20in%20LLMs/tables/efb0c4f31096ef27f19ab415e95fd85b6d8a5d93c90833cbf54e01e29e46babf.jpg)

## LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias


### Images

![2d64f92a1ceaf80c3e600f898ff5bb0eb3dadfe04944c31b397c2a64603b4ee1.jpg](../iclr_results/126_LVSM_%20A%20Large%20View%20Synthesis%20Model%20with%20Minimal%203D%20Inductive%20Bias/images/2d64f92a1ceaf80c3e600f898ff5bb0eb3dadfe04944c31b397c2a64603b4ee1.jpg)

![2ed996585406e05cb6b05138d8a7acf348bea269b96b7ba1e7cd6f9171ed6b4a.jpg](../iclr_results/126_LVSM_%20A%20Large%20View%20Synthesis%20Model%20with%20Minimal%203D%20Inductive%20Bias/images/2ed996585406e05cb6b05138d8a7acf348bea269b96b7ba1e7cd6f9171ed6b4a.jpg)

![3d1030aff61050b472cdeb551c38013f71555e3e25f0a2a58279f6716b3ba264.jpg](../iclr_results/126_LVSM_%20A%20Large%20View%20Synthesis%20Model%20with%20Minimal%203D%20Inductive%20Bias/images/3d1030aff61050b472cdeb551c38013f71555e3e25f0a2a58279f6716b3ba264.jpg)

![5bae16fb580e25dc235cc272a30ab387a5938575419b68e5c8634e184bac304f.jpg](../iclr_results/126_LVSM_%20A%20Large%20View%20Synthesis%20Model%20with%20Minimal%203D%20Inductive%20Bias/images/5bae16fb580e25dc235cc272a30ab387a5938575419b68e5c8634e184bac304f.jpg)

![7fdadd4df74e180dd46fd89ddbbe00e7c8b39fdee349b5ed0e9e3279887c0fe0.jpg](../iclr_results/126_LVSM_%20A%20Large%20View%20Synthesis%20Model%20with%20Minimal%203D%20Inductive%20Bias/images/7fdadd4df74e180dd46fd89ddbbe00e7c8b39fdee349b5ed0e9e3279887c0fe0.jpg)

![80c5faa2f01beb292ba75376fc61da715fbf367d33faabc873ead7b5472836cb.jpg](../iclr_results/126_LVSM_%20A%20Large%20View%20Synthesis%20Model%20with%20Minimal%203D%20Inductive%20Bias/images/80c5faa2f01beb292ba75376fc61da715fbf367d33faabc873ead7b5472836cb.jpg)

![81a01b64d249093318e6b9af9f7ddc56b9ff08eb55d311c94a7d32aca58faae0.jpg](../iclr_results/126_LVSM_%20A%20Large%20View%20Synthesis%20Model%20with%20Minimal%203D%20Inductive%20Bias/images/81a01b64d249093318e6b9af9f7ddc56b9ff08eb55d311c94a7d32aca58faae0.jpg)

![aa65253e30070ced947bf2dfb0a93815cab720f5a49d8df79f6a11e08a5d0f7a.jpg](../iclr_results/126_LVSM_%20A%20Large%20View%20Synthesis%20Model%20with%20Minimal%203D%20Inductive%20Bias/images/aa65253e30070ced947bf2dfb0a93815cab720f5a49d8df79f6a11e08a5d0f7a.jpg)

![e402adba90aaf5c4238b2ab5a8d97ed2a5e80c7a93f2ea8c89714fca55c2e843.jpg](../iclr_results/126_LVSM_%20A%20Large%20View%20Synthesis%20Model%20with%20Minimal%203D%20Inductive%20Bias/images/e402adba90aaf5c4238b2ab5a8d97ed2a5e80c7a93f2ea8c89714fca55c2e843.jpg)

### Tables

![55089b9b3f87b1a7aca0c30aacabd6d001fef7d507bc25d5651cf0c26e7e925a.jpg](../iclr_results/126_LVSM_%20A%20Large%20View%20Synthesis%20Model%20with%20Minimal%203D%20Inductive%20Bias/tables/55089b9b3f87b1a7aca0c30aacabd6d001fef7d507bc25d5651cf0c26e7e925a.jpg)

![630997a03b38f98197bbc6581c7c8e8ec3d258683281e9fde61270b62493b9e5.jpg](../iclr_results/126_LVSM_%20A%20Large%20View%20Synthesis%20Model%20with%20Minimal%203D%20Inductive%20Bias/tables/630997a03b38f98197bbc6581c7c8e8ec3d258683281e9fde61270b62493b9e5.jpg)

![a16a1d82048ccd62de0e04813494302ae55149941611d0a6afc2dad0f3f94d73.jpg](../iclr_results/126_LVSM_%20A%20Large%20View%20Synthesis%20Model%20with%20Minimal%203D%20Inductive%20Bias/tables/a16a1d82048ccd62de0e04813494302ae55149941611d0a6afc2dad0f3f94d73.jpg)

![d8e5992b7390b1ae1837ede2b51b996f99b03baac14328f781c60c672c63ed9d.jpg](../iclr_results/126_LVSM_%20A%20Large%20View%20Synthesis%20Model%20with%20Minimal%203D%20Inductive%20Bias/tables/d8e5992b7390b1ae1837ede2b51b996f99b03baac14328f781c60c672c63ed9d.jpg)

## From Exploration to Mastery: Enabling LLMs to Master Tools via Self-Driven Interactions


### Images

![00673069a8a211320be7b9370f71c3c9dba07b96064bb39390d3b5851bf3ab04.jpg](../iclr_results/127_From%20Exploration%20to%20Mastery_%20Enabling%20LLMs%20to%20Master%20Tools%20via%20Self-Driven%20Interactions/images/00673069a8a211320be7b9370f71c3c9dba07b96064bb39390d3b5851bf3ab04.jpg)

![21cd3055b4bbec22d9a86614a5f7160d3e77f04939a297b7055b01f2e8c2992e.jpg](../iclr_results/127_From%20Exploration%20to%20Mastery_%20Enabling%20LLMs%20to%20Master%20Tools%20via%20Self-Driven%20Interactions/images/21cd3055b4bbec22d9a86614a5f7160d3e77f04939a297b7055b01f2e8c2992e.jpg)

![22659aa99787e6397aea18bc20f30a92581b6ec624250d6ee808274e6fc30ec6.jpg](../iclr_results/127_From%20Exploration%20to%20Mastery_%20Enabling%20LLMs%20to%20Master%20Tools%20via%20Self-Driven%20Interactions/images/22659aa99787e6397aea18bc20f30a92581b6ec624250d6ee808274e6fc30ec6.jpg)

![6c80893eb150fc8941364cea299a71d424543abc7e17e0f1871784365aec0d27.jpg](../iclr_results/127_From%20Exploration%20to%20Mastery_%20Enabling%20LLMs%20to%20Master%20Tools%20via%20Self-Driven%20Interactions/images/6c80893eb150fc8941364cea299a71d424543abc7e17e0f1871784365aec0d27.jpg)

![810f12d56d0f40ede7acce62980854a10ba6bfe42e3825d626b17c8e4aea72d5.jpg](../iclr_results/127_From%20Exploration%20to%20Mastery_%20Enabling%20LLMs%20to%20Master%20Tools%20via%20Self-Driven%20Interactions/images/810f12d56d0f40ede7acce62980854a10ba6bfe42e3825d626b17c8e4aea72d5.jpg)

![b7eadb137424a3459e3725e97b500161bd21d50a5a321ceffcd8c24d7fb46008.jpg](../iclr_results/127_From%20Exploration%20to%20Mastery_%20Enabling%20LLMs%20to%20Master%20Tools%20via%20Self-Driven%20Interactions/images/b7eadb137424a3459e3725e97b500161bd21d50a5a321ceffcd8c24d7fb46008.jpg)

![f87c63853f90a3028774776204a77b4a7154f36c35b32a79ed3c5d4ddaf7a92d.jpg](../iclr_results/127_From%20Exploration%20to%20Mastery_%20Enabling%20LLMs%20to%20Master%20Tools%20via%20Self-Driven%20Interactions/images/f87c63853f90a3028774776204a77b4a7154f36c35b32a79ed3c5d4ddaf7a92d.jpg)

### Tables

![07c719ada4a16ee445a4dbffc3ca6890b6ad7f691a3b6a6d413518dd3d1a6fba.jpg](../iclr_results/127_From%20Exploration%20to%20Mastery_%20Enabling%20LLMs%20to%20Master%20Tools%20via%20Self-Driven%20Interactions/tables/07c719ada4a16ee445a4dbffc3ca6890b6ad7f691a3b6a6d413518dd3d1a6fba.jpg)

![199b2515638b63b1a5f6f62ac4aee930c14e8f2e221338b5d7ddd15f3cceeb38.jpg](../iclr_results/127_From%20Exploration%20to%20Mastery_%20Enabling%20LLMs%20to%20Master%20Tools%20via%20Self-Driven%20Interactions/tables/199b2515638b63b1a5f6f62ac4aee930c14e8f2e221338b5d7ddd15f3cceeb38.jpg)

![499a13af92f8db261a3bbfcf9f429f8a0662a4ffab1208858b2fd46826742da9.jpg](../iclr_results/127_From%20Exploration%20to%20Mastery_%20Enabling%20LLMs%20to%20Master%20Tools%20via%20Self-Driven%20Interactions/tables/499a13af92f8db261a3bbfcf9f429f8a0662a4ffab1208858b2fd46826742da9.jpg)

![625c30fb9b7a152456669cba161670856ff5a18051dc48185fb230e0b28f1f4d.jpg](../iclr_results/127_From%20Exploration%20to%20Mastery_%20Enabling%20LLMs%20to%20Master%20Tools%20via%20Self-Driven%20Interactions/tables/625c30fb9b7a152456669cba161670856ff5a18051dc48185fb230e0b28f1f4d.jpg)

![6db7c585a1aa46d6a0d0a48e332113a4df9448a5a75e9bc63f573e83b0ebc0f3.jpg](../iclr_results/127_From%20Exploration%20to%20Mastery_%20Enabling%20LLMs%20to%20Master%20Tools%20via%20Self-Driven%20Interactions/tables/6db7c585a1aa46d6a0d0a48e332113a4df9448a5a75e9bc63f573e83b0ebc0f3.jpg)

![6f68fe1018092bd2d35a3672145bf41a4707ea0d85fbbc49a0e69185d70a4a6a.jpg](../iclr_results/127_From%20Exploration%20to%20Mastery_%20Enabling%20LLMs%20to%20Master%20Tools%20via%20Self-Driven%20Interactions/tables/6f68fe1018092bd2d35a3672145bf41a4707ea0d85fbbc49a0e69185d70a4a6a.jpg)

![847c9cd5fdf9da1c3b86804d5cc555ce030feff7e371cb13cc01a2def8dc476c.jpg](../iclr_results/127_From%20Exploration%20to%20Mastery_%20Enabling%20LLMs%20to%20Master%20Tools%20via%20Self-Driven%20Interactions/tables/847c9cd5fdf9da1c3b86804d5cc555ce030feff7e371cb13cc01a2def8dc476c.jpg)

![99ee194bc97c1bb2048f45b9ebcbac94bf21add01b863dccc826040df762e0b0.jpg](../iclr_results/127_From%20Exploration%20to%20Mastery_%20Enabling%20LLMs%20to%20Master%20Tools%20via%20Self-Driven%20Interactions/tables/99ee194bc97c1bb2048f45b9ebcbac94bf21add01b863dccc826040df762e0b0.jpg)

![b88f6a6bae9aa8c59b365b7872ced68b41ee6b69947c9caa627459b665fdb500.jpg](../iclr_results/127_From%20Exploration%20to%20Mastery_%20Enabling%20LLMs%20to%20Master%20Tools%20via%20Self-Driven%20Interactions/tables/b88f6a6bae9aa8c59b365b7872ced68b41ee6b69947c9caa627459b665fdb500.jpg)

![d48df7cfc49791230e8146a6c5a1d67793e3b0f9e628c2cb0bac8234265aca46.jpg](../iclr_results/127_From%20Exploration%20to%20Mastery_%20Enabling%20LLMs%20to%20Master%20Tools%20via%20Self-Driven%20Interactions/tables/d48df7cfc49791230e8146a6c5a1d67793e3b0f9e628c2cb0bac8234265aca46.jpg)

![f099f7c05686417ab85065ee4b640747a4ef961ffbd725f1034e89baf22b65ec.jpg](../iclr_results/127_From%20Exploration%20to%20Mastery_%20Enabling%20LLMs%20to%20Master%20Tools%20via%20Self-Driven%20Interactions/tables/f099f7c05686417ab85065ee4b640747a4ef961ffbd725f1034e89baf22b65ec.jpg)

## Advantage Alignment Algorithms


### Images

![59be94ec9723bae4bb3808125d28872c06d1006eb04ede5327500867814925c7.jpg](../iclr_results/128_Advantage%20Alignment%20Algorithms/images/59be94ec9723bae4bb3808125d28872c06d1006eb04ede5327500867814925c7.jpg)

![6ae46462b5fd6e0928db039f3be94588956437960e53f1a659147b23a234ef32.jpg](../iclr_results/128_Advantage%20Alignment%20Algorithms/images/6ae46462b5fd6e0928db039f3be94588956437960e53f1a659147b23a234ef32.jpg)

![7653500c27a0c839a883961e228a4e0c78dbf842b9c0fd484f8356a831e86b31.jpg](../iclr_results/128_Advantage%20Alignment%20Algorithms/images/7653500c27a0c839a883961e228a4e0c78dbf842b9c0fd484f8356a831e86b31.jpg)

![9f370344ad26590e53bfa50325672c59b42219d3be2d38b4b2b66ab6a62a6e7c.jpg](../iclr_results/128_Advantage%20Alignment%20Algorithms/images/9f370344ad26590e53bfa50325672c59b42219d3be2d38b4b2b66ab6a62a6e7c.jpg)

![b11e004506b9e2708f02174c58dc5b33dd9a725eabb7a9e07dbbbcca95efc8a9.jpg](../iclr_results/128_Advantage%20Alignment%20Algorithms/images/b11e004506b9e2708f02174c58dc5b33dd9a725eabb7a9e07dbbbcca95efc8a9.jpg)

![c1cb07c2640b7b85d548e82f8bcb5d5ba6cf9aa03e2a4a176cbc8f56c2c5c8ac.jpg](../iclr_results/128_Advantage%20Alignment%20Algorithms/images/c1cb07c2640b7b85d548e82f8bcb5d5ba6cf9aa03e2a4a176cbc8f56c2c5c8ac.jpg)

![d6929b6f4df40f728d4796e363bb7e970867fcb6ac66fc4d6ba98306278af0ce.jpg](../iclr_results/128_Advantage%20Alignment%20Algorithms/images/d6929b6f4df40f728d4796e363bb7e970867fcb6ac66fc4d6ba98306278af0ce.jpg)

![f168bbb7701499372ad90565ba3206a04698104a47bc350ef1f468b64ab81e35.jpg](../iclr_results/128_Advantage%20Alignment%20Algorithms/images/f168bbb7701499372ad90565ba3206a04698104a47bc350ef1f468b64ab81e35.jpg)

### Tables

![2066ddfc398f7109df14e1aab153d81f04f212e6587e43f684778e812db8f89e.jpg](../iclr_results/128_Advantage%20Alignment%20Algorithms/tables/2066ddfc398f7109df14e1aab153d81f04f212e6587e43f684778e812db8f89e.jpg)

![3b496274b04f317e7c61bb66b8f0f18d299834ee96fff1168c4f2cc91abc72b6.jpg](../iclr_results/128_Advantage%20Alignment%20Algorithms/tables/3b496274b04f317e7c61bb66b8f0f18d299834ee96fff1168c4f2cc91abc72b6.jpg)

![708f3d8290daaecfd74a64d2a4a97d5559162c058445aa2a1bedabb73e111d5a.jpg](../iclr_results/128_Advantage%20Alignment%20Algorithms/tables/708f3d8290daaecfd74a64d2a4a97d5559162c058445aa2a1bedabb73e111d5a.jpg)

![c175fa7cf1d2c98456c2576b2b20c896fe780e5f41aaec927eceae0e97a9ca59.jpg](../iclr_results/128_Advantage%20Alignment%20Algorithms/tables/c175fa7cf1d2c98456c2576b2b20c896fe780e5f41aaec927eceae0e97a9ca59.jpg)

![ccbdbf6de8a1d49596b0f2e144cc2a6bc2cc15331c4a017a10c4f5d5bc56bd84.jpg](../iclr_results/128_Advantage%20Alignment%20Algorithms/tables/ccbdbf6de8a1d49596b0f2e144cc2a6bc2cc15331c4a017a10c4f5d5bc56bd84.jpg)

![f915244cb4af3ba3a9726e0b107b8f7d3472bbc0aa8199a05e022b4d5dd4bb5a.jpg](../iclr_results/128_Advantage%20Alignment%20Algorithms/tables/f915244cb4af3ba3a9726e0b107b8f7d3472bbc0aa8199a05e022b4d5dd4bb5a.jpg)

## RM-Bench: Benchmarking Reward Models of Language Models with Subtlety and Style


### Images

![021c9233dc2263dbd8822a2b43ad1ec4483b8c0cd431f9f2cb48f7a2a0abe7dc.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/images/021c9233dc2263dbd8822a2b43ad1ec4483b8c0cd431f9f2cb48f7a2a0abe7dc.jpg)

![086e919828597e81b4dbd3a235afafecdd315a21a4a95e99c369bd71c0698f4d.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/images/086e919828597e81b4dbd3a235afafecdd315a21a4a95e99c369bd71c0698f4d.jpg)

![19abd726c2ef41a9fa2c85b4343c714730b98cfaba7a55d35c372ef0928fcfd5.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/images/19abd726c2ef41a9fa2c85b4343c714730b98cfaba7a55d35c372ef0928fcfd5.jpg)

![20ec8a9f69cdeea12d886af5196de43fd75bbb09c5ee8eb3055415cc10e08842.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/images/20ec8a9f69cdeea12d886af5196de43fd75bbb09c5ee8eb3055415cc10e08842.jpg)

![2a47e6ce9f3760badb11504f61e4188eed77f516e9662d75d0191947318040f2.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/images/2a47e6ce9f3760badb11504f61e4188eed77f516e9662d75d0191947318040f2.jpg)

![3678105e78e1f3d6000daa25dc37c29b8bd393643fced7df9d04972402eb4f5a.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/images/3678105e78e1f3d6000daa25dc37c29b8bd393643fced7df9d04972402eb4f5a.jpg)

![51370d40489a5d45dea8f90600e3a261ccb048e3ba22ff9623a3c68990c41597.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/images/51370d40489a5d45dea8f90600e3a261ccb048e3ba22ff9623a3c68990c41597.jpg)

![7523857638386f1826a97a7d2d37edd723bc904133920b466b025941dfc98275.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/images/7523857638386f1826a97a7d2d37edd723bc904133920b466b025941dfc98275.jpg)

![79aca2a3cef5758772f94137d15e52e6d35c9aa9e67e7d675e2d0aa80217e2b1.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/images/79aca2a3cef5758772f94137d15e52e6d35c9aa9e67e7d675e2d0aa80217e2b1.jpg)

![a6ac04c9b4076df528cb580d1c21767b5a258e34e8284341e9693d11dabf2f4a.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/images/a6ac04c9b4076df528cb580d1c21767b5a258e34e8284341e9693d11dabf2f4a.jpg)

![b5d51c766686d72853bb67f4d728ae0b86d54a26292dd8d013e08ad0e9cace80.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/images/b5d51c766686d72853bb67f4d728ae0b86d54a26292dd8d013e08ad0e9cace80.jpg)

![caca8af65723951d38e8cb8af03f7a720a6715b86297c548c31114cae55c289e.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/images/caca8af65723951d38e8cb8af03f7a720a6715b86297c548c31114cae55c289e.jpg)

### Tables

![01f13724e23b8aac90662826c45323d5da0c238da716a3d82730993edebdf470.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/tables/01f13724e23b8aac90662826c45323d5da0c238da716a3d82730993edebdf470.jpg)

![2091f6381624fe8299584b1ecef7635a82b170022b87a13d7ed4d86037bde73c.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/tables/2091f6381624fe8299584b1ecef7635a82b170022b87a13d7ed4d86037bde73c.jpg)

![331be5cd66707473c88fa6eec72420863bc2217821591974d060fa14b577953e.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/tables/331be5cd66707473c88fa6eec72420863bc2217821591974d060fa14b577953e.jpg)

![42deb47d94e5fda44e92945ca1f5bc8885edc0f919979b91825dd12bc8eb98b3.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/tables/42deb47d94e5fda44e92945ca1f5bc8885edc0f919979b91825dd12bc8eb98b3.jpg)

![49baef56773e023d75fbd5d37f9d97cd6d83b9fa1c03f60cb5c55d125d9fa492.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/tables/49baef56773e023d75fbd5d37f9d97cd6d83b9fa1c03f60cb5c55d125d9fa492.jpg)

![6d4b2cba6f44aa52e99a685ac33a6c2cbb2bab7c515763bf703edf815dbacb94.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/tables/6d4b2cba6f44aa52e99a685ac33a6c2cbb2bab7c515763bf703edf815dbacb94.jpg)

![7105138ada0a9f40c789ae6cddaa8cfafb404d2c0dd393725da37823f7e3f535.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/tables/7105138ada0a9f40c789ae6cddaa8cfafb404d2c0dd393725da37823f7e3f535.jpg)

![7915fdeb866802d493b9d34f19348aa0b9e7530c179971d1d8b19392013a3c83.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/tables/7915fdeb866802d493b9d34f19348aa0b9e7530c179971d1d8b19392013a3c83.jpg)

![8bae785f140e91c857b9a2c1b267e35ef6ce0225cba8ca066a18f33ee940f249.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/tables/8bae785f140e91c857b9a2c1b267e35ef6ce0225cba8ca066a18f33ee940f249.jpg)

![939d08ade169f18b0b308d264f1505e03c67fc0aa968113abd0f3f851ee8189b.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/tables/939d08ade169f18b0b308d264f1505e03c67fc0aa968113abd0f3f851ee8189b.jpg)

![9c562689fa278dc8a3e520335031274fea5114229ce701c5f9a03a7efacb603e.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/tables/9c562689fa278dc8a3e520335031274fea5114229ce701c5f9a03a7efacb603e.jpg)

![9fb59e99a2f396eabe490ea57229f312103e67cd465c10122fb4804810b06df5.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/tables/9fb59e99a2f396eabe490ea57229f312103e67cd465c10122fb4804810b06df5.jpg)

![a9bb4b8c361f52366f4ddc7d574c8cf910c6312802aece4cabf679109df255f7.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/tables/a9bb4b8c361f52366f4ddc7d574c8cf910c6312802aece4cabf679109df255f7.jpg)

![c98042373b9c6063c3fbb53c5ddfcb065cc0c4926be67c6c77a2705b6a03fd43.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/tables/c98042373b9c6063c3fbb53c5ddfcb065cc0c4926be67c6c77a2705b6a03fd43.jpg)

![d7b26f342f9ec47f24e8e765a962dde1cd9840ff8e96aecc6c76f6fad1150503.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/tables/d7b26f342f9ec47f24e8e765a962dde1cd9840ff8e96aecc6c76f6fad1150503.jpg)

![ddd25d5c1dc62d7cfe6af195b77580fecdc312732c3eccfac87e06a5e0584442.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/tables/ddd25d5c1dc62d7cfe6af195b77580fecdc312732c3eccfac87e06a5e0584442.jpg)

![e2bb1cb04eebc01dde68e1d236c45fdbc3c640950113e7297d36f61156b63de3.jpg](../iclr_results/129_RM-Bench_%20Benchmarking%20Reward%20Models%20of%20Language%20Models%20with%20Subtlety%20and%20Style/tables/e2bb1cb04eebc01dde68e1d236c45fdbc3c640950113e7297d36f61156b63de3.jpg)

## PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding


### Images

![03b6c7591c072823b2fe392fbd7de48f1efaa7cdab6a19d9bc7a26f5bc7dacae.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/03b6c7591c072823b2fe392fbd7de48f1efaa7cdab6a19d9bc7a26f5bc7dacae.jpg)

![03ca473451bc12a64218dcb0686dadf86a32f2b2b7bc6e11e2b67301aaaec2c4.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/03ca473451bc12a64218dcb0686dadf86a32f2b2b7bc6e11e2b67301aaaec2c4.jpg)

![05dd54266abca44849fd179c0d285e3c541b9f68870d036158ca7c4f10f1e272.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/05dd54266abca44849fd179c0d285e3c541b9f68870d036158ca7c4f10f1e272.jpg)

![089676752a8bbfa208774606db7206d22cdff9226c6474589ddcdcdfdc9699bb.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/089676752a8bbfa208774606db7206d22cdff9226c6474589ddcdcdfdc9699bb.jpg)

![0a12af2e033322256c459aec7fc9ca594a13548135929d778174506746770165.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/0a12af2e033322256c459aec7fc9ca594a13548135929d778174506746770165.jpg)

![0a1dddec398d036479978837f24cb40559fb9d9c4c6c15d4c0493da95bb60545.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/0a1dddec398d036479978837f24cb40559fb9d9c4c6c15d4c0493da95bb60545.jpg)

![0c05f3b30519b6302de62a525c37f576e2596acf76931fb6e900449c5e9b4381.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/0c05f3b30519b6302de62a525c37f576e2596acf76931fb6e900449c5e9b4381.jpg)

![0f2852ddfee3e321653a8eab3f7895413ad5eaa29c0ddfef30172d12fc8efd7c.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/0f2852ddfee3e321653a8eab3f7895413ad5eaa29c0ddfef30172d12fc8efd7c.jpg)

![0f3a274aadc9b08148646c3704a15fcda9d2df49830c66ce893d6e863468ab7d.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/0f3a274aadc9b08148646c3704a15fcda9d2df49830c66ce893d6e863468ab7d.jpg)

![1020ea3238441a8e70e8042ea4c872b3c14392be30fc7f7e00b0634bcc1b8ba6.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/1020ea3238441a8e70e8042ea4c872b3c14392be30fc7f7e00b0634bcc1b8ba6.jpg)

![11677cad10777a80c307678326c43c85d6143a02973938c9120aa06ba8d0db9f.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/11677cad10777a80c307678326c43c85d6143a02973938c9120aa06ba8d0db9f.jpg)

![1220ecd755915a1821bc34f514f9ced3084f8d23ae4228a89c70fd167efba514.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/1220ecd755915a1821bc34f514f9ced3084f8d23ae4228a89c70fd167efba514.jpg)

![134f0b3bd2194f5fee86542423d4183c9cbdf86a91d436959cd676510be8d447.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/134f0b3bd2194f5fee86542423d4183c9cbdf86a91d436959cd676510be8d447.jpg)

![14e8e2af3eec44cc965d4ae19426ff5b845bbe2aaaeef328b8e42b2e84edc74d.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/14e8e2af3eec44cc965d4ae19426ff5b845bbe2aaaeef328b8e42b2e84edc74d.jpg)

![15715dd1425f78b560280c3eb1d43d4ed27bf6d13309c97043595e9d932d2ea0.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/15715dd1425f78b560280c3eb1d43d4ed27bf6d13309c97043595e9d932d2ea0.jpg)

![1607ba03a130f9a7d8a625307a0072b38a8c2e7b6a7cc1c6bc13cb7b373daa5e.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/1607ba03a130f9a7d8a625307a0072b38a8c2e7b6a7cc1c6bc13cb7b373daa5e.jpg)

![16d3e823736c7fdf9ae7f7c6031498bfd324cda27b590c09f9af369953e5d439.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/16d3e823736c7fdf9ae7f7c6031498bfd324cda27b590c09f9af369953e5d439.jpg)

![184a52680407be874db458c1959f940b4bcc24e977afa715711324d41461ac55.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/184a52680407be874db458c1959f940b4bcc24e977afa715711324d41461ac55.jpg)

![194286364cb460f57780f2fe289dad02f4d5b0a5abf7f77a239b8a4ac2a5081b.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/194286364cb460f57780f2fe289dad02f4d5b0a5abf7f77a239b8a4ac2a5081b.jpg)

![1b1e6e12c83686bb805016a69104ccaf75a91a74dc02c30109a700e83ff7245a.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/1b1e6e12c83686bb805016a69104ccaf75a91a74dc02c30109a700e83ff7245a.jpg)

![1c570f9da239bb34d13f41755c43507fb1124fad02f6ca92d8c4ae4085a60691.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/1c570f9da239bb34d13f41755c43507fb1124fad02f6ca92d8c4ae4085a60691.jpg)

![1e485513efcb39bba0a8328df9400e3136ecc47d5bf5f917052eab4902acb9ca.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/1e485513efcb39bba0a8328df9400e3136ecc47d5bf5f917052eab4902acb9ca.jpg)

![217ce1dc440bb46b45ee2df7db46603b3d69485be43b6cd8c8fab9f0c882f4ab.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/217ce1dc440bb46b45ee2df7db46603b3d69485be43b6cd8c8fab9f0c882f4ab.jpg)

![23d3d6b8950af9143c67d3d3493ec80d8da6bc854c4bfeda46cb9fd5dc0abb74.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/23d3d6b8950af9143c67d3d3493ec80d8da6bc854c4bfeda46cb9fd5dc0abb74.jpg)

![276322dab2791ed07553868e8c53576ea7d2b479799e92c7c80e661a4886f7df.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/276322dab2791ed07553868e8c53576ea7d2b479799e92c7c80e661a4886f7df.jpg)

![27e599a51c58eca211fdc53b408ac225f11899ed039289344d0cb44faa62186f.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/27e599a51c58eca211fdc53b408ac225f11899ed039289344d0cb44faa62186f.jpg)

![28029b7ff87e2bd964a0ab3223ac4c51cc205267bc777a8243476b7c4e320932.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/28029b7ff87e2bd964a0ab3223ac4c51cc205267bc777a8243476b7c4e320932.jpg)

![2a03776988c9d81d1429dfd17393fc10d6a3a14296aa73a4d457160ce5455edb.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/2a03776988c9d81d1429dfd17393fc10d6a3a14296aa73a4d457160ce5455edb.jpg)

![2a547ddb829793d883eda13c70cb3a3db008c9a4b9a571ebb3aee788c766df54.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/2a547ddb829793d883eda13c70cb3a3db008c9a4b9a571ebb3aee788c766df54.jpg)

![2e4074964177b8838da6995616d0a0f60e5a895889dea089d9d076ac74c35140.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/2e4074964177b8838da6995616d0a0f60e5a895889dea089d9d076ac74c35140.jpg)

![2e642d31dc48a63364c30295deb994d738f1d55fc62c9dc6d05925dde304713b.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/2e642d31dc48a63364c30295deb994d738f1d55fc62c9dc6d05925dde304713b.jpg)

![300b233e6d0ae184e46ae19fc419f51fca0c1155d1403b4ba0195932d8f8da73.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/300b233e6d0ae184e46ae19fc419f51fca0c1155d1403b4ba0195932d8f8da73.jpg)

![30859deab3dfd16e2dfaf0a6c67c6896cc1f858bb8c8088d8f8234e8f7a08faa.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/30859deab3dfd16e2dfaf0a6c67c6896cc1f858bb8c8088d8f8234e8f7a08faa.jpg)

![30964c2abfa438b26262d7e84c65c3673dd73cfe4329953680c3a8bf7eca7616.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/30964c2abfa438b26262d7e84c65c3673dd73cfe4329953680c3a8bf7eca7616.jpg)

![312a7d43b26e44355aabbc6495be535bf5c6b20dd5dc7a5eb52a723d6e1c38c2.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/312a7d43b26e44355aabbc6495be535bf5c6b20dd5dc7a5eb52a723d6e1c38c2.jpg)

![316f04024bb2b6a95b48a3c6402d16f75789de4a3ad22e6bd24cc302c33a3878.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/316f04024bb2b6a95b48a3c6402d16f75789de4a3ad22e6bd24cc302c33a3878.jpg)

![317abac409ced27114be9b91744fc7ee10a44c1872fedf427caa0db9edacf68c.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/317abac409ced27114be9b91744fc7ee10a44c1872fedf427caa0db9edacf68c.jpg)

![328a4698ffe144255ad2da082f704975f1c5baf7a712f091bd08931e6392b96c.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/328a4698ffe144255ad2da082f704975f1c5baf7a712f091bd08931e6392b96c.jpg)

![3551fb6ff76eec6a62a29d32c9348c2d2c3b71312275c5f59a3819ad2ff77ede.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/3551fb6ff76eec6a62a29d32c9348c2d2c3b71312275c5f59a3819ad2ff77ede.jpg)

![35dea7d973f2276fc2583f418054c0388f9d5cef0232194a42685e6740e8586d.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/35dea7d973f2276fc2583f418054c0388f9d5cef0232194a42685e6740e8586d.jpg)

![36d7e504fde4fada67fb47a9fccbf75958e71feb23f8b9970b85aadade935c4e.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/36d7e504fde4fada67fb47a9fccbf75958e71feb23f8b9970b85aadade935c4e.jpg)

![37443e309e0557a138a8b88eac2f08f7b60aaedb132d735edf85e0d008970cbf.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/37443e309e0557a138a8b88eac2f08f7b60aaedb132d735edf85e0d008970cbf.jpg)

![38bbd0162a77a1a9b81a51afbb2476029591a9553c11d43dc56c926eb56868ff.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/38bbd0162a77a1a9b81a51afbb2476029591a9553c11d43dc56c926eb56868ff.jpg)

![38f8046868efc110e2054ba833db59538e7cea933d60813b77951b5955d5b080.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/38f8046868efc110e2054ba833db59538e7cea933d60813b77951b5955d5b080.jpg)

![3a0ee2126c72ea5f5119f46612f1aee9b08605902364a250a5ad386036cd0fca.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/3a0ee2126c72ea5f5119f46612f1aee9b08605902364a250a5ad386036cd0fca.jpg)

![3ba22ffd666418d4357ee92f54cb162c6702b8d31fc9c78eccec98c521ec0130.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/3ba22ffd666418d4357ee92f54cb162c6702b8d31fc9c78eccec98c521ec0130.jpg)

![3c79b12df91d0b0e4289579caac54287714db3a5a68c173530f10a6a638c116a.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/3c79b12df91d0b0e4289579caac54287714db3a5a68c173530f10a6a638c116a.jpg)

![3da60392875ab2326cfd0d9a407865fa8382d2e0e1ef975a09117f57d2a4c2e6.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/3da60392875ab2326cfd0d9a407865fa8382d2e0e1ef975a09117f57d2a4c2e6.jpg)

![41e47c6ec8b0144b4c2c2e147fc565d357a145787544ba8372dea1e9757198e4.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/41e47c6ec8b0144b4c2c2e147fc565d357a145787544ba8372dea1e9757198e4.jpg)

![444c3c5a8ed5831f113d4b22f8749aa2908f0c3aa0a2addda9ff4e77e11752a2.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/444c3c5a8ed5831f113d4b22f8749aa2908f0c3aa0a2addda9ff4e77e11752a2.jpg)

![4b223ad99150b3bea55bf9e0f7923d4e6ac4d46cc017d1b192528cbdd9216a33.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/4b223ad99150b3bea55bf9e0f7923d4e6ac4d46cc017d1b192528cbdd9216a33.jpg)

![4c02199b2783878c32359f0ea91b2ced949e3d349916f360ce2c936bea881689.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/4c02199b2783878c32359f0ea91b2ced949e3d349916f360ce2c936bea881689.jpg)

![4de717eeb3ffbffb0fce753c17e843b4e88c41fd7098933c8a248818a3b427bb.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/4de717eeb3ffbffb0fce753c17e843b4e88c41fd7098933c8a248818a3b427bb.jpg)

![4ef0869fc7bf1e2c02f8276bd8f18aa9d1721ef9fd48d406e2595bf5aaef4c99.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/4ef0869fc7bf1e2c02f8276bd8f18aa9d1721ef9fd48d406e2595bf5aaef4c99.jpg)

![529a297c1033a1a5d061704935774d774a1a8f2d24c2e8b2834fd4dd03d5d64f.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/529a297c1033a1a5d061704935774d774a1a8f2d24c2e8b2834fd4dd03d5d64f.jpg)

![53a95071532cae565f63b72fe65adf9f09862eeaae1fca0dc7936e50cc211983.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/53a95071532cae565f63b72fe65adf9f09862eeaae1fca0dc7936e50cc211983.jpg)

![5407fc9f609372025951462386104891fef45304737e29d11365a59d7650bf7e.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/5407fc9f609372025951462386104891fef45304737e29d11365a59d7650bf7e.jpg)

![54646ab7e859fce3fb7ba0237e3866d1ca5e261dec2803be0b09b2b94a6f7b05.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/54646ab7e859fce3fb7ba0237e3866d1ca5e261dec2803be0b09b2b94a6f7b05.jpg)

![56144c6b94b3714d0b1ec14e12bcdabc0bded5bbd70967dd9b9d32e4f00ce140.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/56144c6b94b3714d0b1ec14e12bcdabc0bded5bbd70967dd9b9d32e4f00ce140.jpg)

![569aa486f65564f71175bb4716760a7e56a0f6444bad01ef7003c460591e5177.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/569aa486f65564f71175bb4716760a7e56a0f6444bad01ef7003c460591e5177.jpg)

![56ce437e758c97a174d4b997425ca7c3e2e75fc004973aa0877f3c18c3bd0545.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/56ce437e758c97a174d4b997425ca7c3e2e75fc004973aa0877f3c18c3bd0545.jpg)

![56d94ca87178774d72f02b7221a2bfd008a3c177992f2b6e9e3850a2f46d9c9c.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/56d94ca87178774d72f02b7221a2bfd008a3c177992f2b6e9e3850a2f46d9c9c.jpg)

![56ede855b5fa90ccd311f309b9d494f6508d8c680a26bab297cfd439cddbbef4.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/56ede855b5fa90ccd311f309b9d494f6508d8c680a26bab297cfd439cddbbef4.jpg)

![575ec775f3584300f5f12a22f78b54119ec5b65167e42bf3b22146b58b550370.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/575ec775f3584300f5f12a22f78b54119ec5b65167e42bf3b22146b58b550370.jpg)

![5798506f87efe8396c6110a8552ebc424a39fc88d7bf751464ac14f4f7178dc7.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/5798506f87efe8396c6110a8552ebc424a39fc88d7bf751464ac14f4f7178dc7.jpg)

![5970f489bc7ec7a4519276580d58a3b9b2348785d87b161ff07abdd9d7b2bf35.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/5970f489bc7ec7a4519276580d58a3b9b2348785d87b161ff07abdd9d7b2bf35.jpg)

![5c5e606d4ebbdfb8b31b661e6e9b0e55be7b3e3afed9f3affb4270834e6606ac.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/5c5e606d4ebbdfb8b31b661e6e9b0e55be7b3e3afed9f3affb4270834e6606ac.jpg)

![5d8a9ab28b97a34ea83cfefad2b4cb0b9ade4cb8d205c55433101f35f8576238.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/5d8a9ab28b97a34ea83cfefad2b4cb0b9ade4cb8d205c55433101f35f8576238.jpg)

![5ed824e3974d4a1066858bbe9f0aec486f60b557cd5c5fbdfa51151a5e324444.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/5ed824e3974d4a1066858bbe9f0aec486f60b557cd5c5fbdfa51151a5e324444.jpg)

![5ff823f8328e642d8507d3ea1fa915125f184c9301741e8c02fcaa29ce2423fd.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/5ff823f8328e642d8507d3ea1fa915125f184c9301741e8c02fcaa29ce2423fd.jpg)

![609b406ddc9961748558eea5cf0214dfad5274048589169719413a381c89ade1.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/609b406ddc9961748558eea5cf0214dfad5274048589169719413a381c89ade1.jpg)

![6576eaaa57918f6a9390eed058fa331bdcfa2de0d6838bec55c0dee27068cf2f.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/6576eaaa57918f6a9390eed058fa331bdcfa2de0d6838bec55c0dee27068cf2f.jpg)

![66772cf8d1ec84d5238db67e2da869466374dc272eea0ccb6136bb054b88ee5b.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/66772cf8d1ec84d5238db67e2da869466374dc272eea0ccb6136bb054b88ee5b.jpg)

![66ae59812c4cdca309570e62b7006659e408ad5ae56a87654ae8e89158086f75.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/66ae59812c4cdca309570e62b7006659e408ad5ae56a87654ae8e89158086f75.jpg)

![675fd492dc113e22e2a6e5cb390819397ab6d57b6d1fb3558d5a920dc5fb6594.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/675fd492dc113e22e2a6e5cb390819397ab6d57b6d1fb3558d5a920dc5fb6594.jpg)

![677bb42cb65e1640a9c78fa1d37c40ba2684a3c220aa7a6bf8fff88244c2d867.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/677bb42cb65e1640a9c78fa1d37c40ba2684a3c220aa7a6bf8fff88244c2d867.jpg)

![67902ce6d0a3304c2dbbb3317cc67549ad4b084c08128c824584cf020f8e72b9.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/67902ce6d0a3304c2dbbb3317cc67549ad4b084c08128c824584cf020f8e72b9.jpg)

![681cf50b4c4e4ce416d72218f606b462b86ede3a87af95df1fbf80bcd2825663.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/681cf50b4c4e4ce416d72218f606b462b86ede3a87af95df1fbf80bcd2825663.jpg)

![68d028196d856e229b25109adfd9599bc0bc0f94729abad400a7ae06b2b95ab6.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/68d028196d856e229b25109adfd9599bc0bc0f94729abad400a7ae06b2b95ab6.jpg)

![68eb2bafcadff077a594c06650b7a738b9c4aca30ccbb2ec71cf05a85ae34a04.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/68eb2bafcadff077a594c06650b7a738b9c4aca30ccbb2ec71cf05a85ae34a04.jpg)

![68eda2dc00f3f671da2a8d424b5de2b162fa5ef6e54460c5d55ec1b4301116d5.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/68eda2dc00f3f671da2a8d424b5de2b162fa5ef6e54460c5d55ec1b4301116d5.jpg)

![6aa8f4a4c31a50b4b7582fe9cc017ab586a7754542480c5d1375b6b1753c9126.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/6aa8f4a4c31a50b4b7582fe9cc017ab586a7754542480c5d1375b6b1753c9126.jpg)

![6ff6adebe1b380243bed083ef43e377df3876ff0161b2c9f134b6143aefb51c4.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/6ff6adebe1b380243bed083ef43e377df3876ff0161b2c9f134b6143aefb51c4.jpg)

![717833f9552b67ceddc89999eaa86028d1c4c33fcf795909f16dcb0c8ca3931d.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/717833f9552b67ceddc89999eaa86028d1c4c33fcf795909f16dcb0c8ca3931d.jpg)

![75569a938bed73ad5f0877f3683f2ede0ffc8c9f906b9d10f9370adcffacf174.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/75569a938bed73ad5f0877f3683f2ede0ffc8c9f906b9d10f9370adcffacf174.jpg)

![7651b3a3a8a89c390524bed0bcd306543a74c78d0c4328a3d15c2727680257db.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/7651b3a3a8a89c390524bed0bcd306543a74c78d0c4328a3d15c2727680257db.jpg)

![77420494e27530b8893ab966f0294bf9567e51bf7e9c4acc6f119115f4a4686e.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/77420494e27530b8893ab966f0294bf9567e51bf7e9c4acc6f119115f4a4686e.jpg)

![77618e98f684913acc9fd9a01e7aad02d1c29d2e73e2c1bd6ff800401e3382ea.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/77618e98f684913acc9fd9a01e7aad02d1c29d2e73e2c1bd6ff800401e3382ea.jpg)

![779454586482f22cd72504ba7dc39c60195a73541a555b57a33f7a3ee0a457f6.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/779454586482f22cd72504ba7dc39c60195a73541a555b57a33f7a3ee0a457f6.jpg)

![78bf089f7f57596d66272a2677bad45354f2d46e71a24453d0a4d8dba0d4a6de.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/78bf089f7f57596d66272a2677bad45354f2d46e71a24453d0a4d8dba0d4a6de.jpg)

![791fc8b664921c5b9ca836acb895c57724fd2813bbec319f9bfe9f4a528f6899.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/791fc8b664921c5b9ca836acb895c57724fd2813bbec319f9bfe9f4a528f6899.jpg)

![79a75c8bc9052af3f66a658c35190c7da2a91a24383902f323021f6d16d2703c.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/79a75c8bc9052af3f66a658c35190c7da2a91a24383902f323021f6d16d2703c.jpg)

![7a52ae1a3cb83415e70dc1399d7056aedaf90af52a98953d3246f2c4854e3654.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/7a52ae1a3cb83415e70dc1399d7056aedaf90af52a98953d3246f2c4854e3654.jpg)

![7b5cfd64f29aef9f60c860ef31c31bd2ca6b86bfea66a647b30a0ad0dec821dc.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/7b5cfd64f29aef9f60c860ef31c31bd2ca6b86bfea66a647b30a0ad0dec821dc.jpg)

![7dc7fc64dd94f616f6fdc69671818845889fe407864ccb765ad567b9ea44f6f3.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/7dc7fc64dd94f616f6fdc69671818845889fe407864ccb765ad567b9ea44f6f3.jpg)

![7f811ff0f66c410dc8bab5b3f0b35aa1217c78987cb1facadbfbd8720a2a9cf9.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/7f811ff0f66c410dc8bab5b3f0b35aa1217c78987cb1facadbfbd8720a2a9cf9.jpg)

![7fe1bad49580b2ce53b40294b51a09423ccb42ccba9a735dfa1df0ad10533634.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/7fe1bad49580b2ce53b40294b51a09423ccb42ccba9a735dfa1df0ad10533634.jpg)

![823093c3a234d707da5e5736d6ca80314c94bb5fd42d749d4846704a5a8ad86e.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/823093c3a234d707da5e5736d6ca80314c94bb5fd42d749d4846704a5a8ad86e.jpg)

![829732d2407f571b44c5084a3fbca63df365f2fb14e6e81e4d81d16fb8a1a7db.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/829732d2407f571b44c5084a3fbca63df365f2fb14e6e81e4d81d16fb8a1a7db.jpg)

![82de5181c1ae9ad316a9847f8e76f948b95d24569ffa53cdf2fb33656ab12976.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/82de5181c1ae9ad316a9847f8e76f948b95d24569ffa53cdf2fb33656ab12976.jpg)

![8441e6ed1eabecffab3ce90d50a40f332293f7be49ff25f20d48d49838737bf6.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/8441e6ed1eabecffab3ce90d50a40f332293f7be49ff25f20d48d49838737bf6.jpg)

![856d634032372dcf200f9833e6184ca70f04205562d5167e65f5bc57348521fb.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/856d634032372dcf200f9833e6184ca70f04205562d5167e65f5bc57348521fb.jpg)

![8600169942f77f10dc0194465dbce580505c77e2611e1d1c9ab2e7b0df84d3ce.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/8600169942f77f10dc0194465dbce580505c77e2611e1d1c9ab2e7b0df84d3ce.jpg)

![8833b81ff65cdbe79f0bc5087585d0e4cc9feab445f5ee82906b69d249a71dd1.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/8833b81ff65cdbe79f0bc5087585d0e4cc9feab445f5ee82906b69d249a71dd1.jpg)

![885da3afceb0e61fcbcbf1017a32d562d9e08574575b2156516efc72fc7a27e4.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/885da3afceb0e61fcbcbf1017a32d562d9e08574575b2156516efc72fc7a27e4.jpg)

![8ba93e36435ab19002e6884a2bcbefc5a1d8022542230bfcdf2006fc49511915.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/8ba93e36435ab19002e6884a2bcbefc5a1d8022542230bfcdf2006fc49511915.jpg)

![90039a0efb4d0f82aad4c5f01e8d0e3f3b401686cad6b7d33d3565e5b7c0d18a.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/90039a0efb4d0f82aad4c5f01e8d0e3f3b401686cad6b7d33d3565e5b7c0d18a.jpg)

![903505d68b86ccf5a57c77d098169edf3938e2f6cc7cd68f5df2aa1771677de3.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/903505d68b86ccf5a57c77d098169edf3938e2f6cc7cd68f5df2aa1771677de3.jpg)

![905e5a8cc335347e367b662b71f0e69d514bfc116dcb7292abb8b03a99f86528.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/905e5a8cc335347e367b662b71f0e69d514bfc116dcb7292abb8b03a99f86528.jpg)

![932dc64abbb2f5a692425d62f579cbf0b8ca97a2470361eb6f97c5c06ce817d5.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/932dc64abbb2f5a692425d62f579cbf0b8ca97a2470361eb6f97c5c06ce817d5.jpg)

![949997dbe17dfe75a756eddee5489ade85f660ac7b7c9a637b05f09269022d22.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/949997dbe17dfe75a756eddee5489ade85f660ac7b7c9a637b05f09269022d22.jpg)

![96e9ef9f2db1215d13bcef9cd662c8054057e1430175b140e212b8b964be234d.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/96e9ef9f2db1215d13bcef9cd662c8054057e1430175b140e212b8b964be234d.jpg)

![96ec91edfd3153a251327ec62b53db2745caf4b222d6d73ec4b099fd7bb0fd95.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/96ec91edfd3153a251327ec62b53db2745caf4b222d6d73ec4b099fd7bb0fd95.jpg)

![970db0faaf8cfe96091e25024c8c0e9478571665a7b5e6876caaa2009cd3710e.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/970db0faaf8cfe96091e25024c8c0e9478571665a7b5e6876caaa2009cd3710e.jpg)

![985d27608951d32178fd1e231d5204aa5ec57ef07df11fd77ec21e1d8fcb6046.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/985d27608951d32178fd1e231d5204aa5ec57ef07df11fd77ec21e1d8fcb6046.jpg)

![99ba19b56342d4f7c8d389d59d5f18f444ca5f45436304f3c06280a5917caa74.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/99ba19b56342d4f7c8d389d59d5f18f444ca5f45436304f3c06280a5917caa74.jpg)

![9d1c638a3e2ddc20daa56479507246f73f4a2acff28220562a9bcf9d74488c10.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/9d1c638a3e2ddc20daa56479507246f73f4a2acff28220562a9bcf9d74488c10.jpg)

![9d3b074345fd7fb70a34f8bac3a4846945d6954148dc4b5a1b59c6038703b6fa.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/9d3b074345fd7fb70a34f8bac3a4846945d6954148dc4b5a1b59c6038703b6fa.jpg)

![9d798b021fe57967c1fc759d51917f6a3fc81315df7c944958dece4ff5a4bb01.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/9d798b021fe57967c1fc759d51917f6a3fc81315df7c944958dece4ff5a4bb01.jpg)

![9ea7be4ce253c28c8d1f2bc6282b427fa6e8eb9904b790b7ba936d4225d6cd14.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/9ea7be4ce253c28c8d1f2bc6282b427fa6e8eb9904b790b7ba936d4225d6cd14.jpg)

![9fd78667eca69ad0e00a137848d48e8198f28b7a7529c1b5dea16a99d039c587.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/9fd78667eca69ad0e00a137848d48e8198f28b7a7529c1b5dea16a99d039c587.jpg)

![a309118ff21724692827305aa00287bae173ce05fd649e1c26545fe72689d72d.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/a309118ff21724692827305aa00287bae173ce05fd649e1c26545fe72689d72d.jpg)

![a358ca68540b4b8967f68241e7d258b94744ace11d67ddad7c009f13ec5d9d61.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/a358ca68540b4b8967f68241e7d258b94744ace11d67ddad7c009f13ec5d9d61.jpg)

![a6f1dcd04cb7a1533d60cf94b4391dbc4431ac5164cbbe8e0c08f6f0a08bec6b.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/a6f1dcd04cb7a1533d60cf94b4391dbc4431ac5164cbbe8e0c08f6f0a08bec6b.jpg)

![a7e3a9f0455ddf966c41bb740c4c5fea39c2de54a064dd65a63f5e7b7a782ffc.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/a7e3a9f0455ddf966c41bb740c4c5fea39c2de54a064dd65a63f5e7b7a782ffc.jpg)

![a9cd3ed3862541323584243941bd5fe334bc4bbfda7e1c36fa97d7ac7c8f1ecf.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/a9cd3ed3862541323584243941bd5fe334bc4bbfda7e1c36fa97d7ac7c8f1ecf.jpg)

![aa5020d8da6dc09ad10f81d842b247875d215683c3955d1aef063d9da27474f8.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/aa5020d8da6dc09ad10f81d842b247875d215683c3955d1aef063d9da27474f8.jpg)

![ae1beb2108c7b7f3a29b276d7b3b1c4efac9bd83e69fd003d1930569d4dce103.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/ae1beb2108c7b7f3a29b276d7b3b1c4efac9bd83e69fd003d1930569d4dce103.jpg)

![b0e204e3f1949f79809dd7094869381d85aaf96cc48dfc0781c4adb2cddac069.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/b0e204e3f1949f79809dd7094869381d85aaf96cc48dfc0781c4adb2cddac069.jpg)

![b306a8d3d6a5a70719ba1c57f518c4e8cd0e12781cbee26f69b032f940db1986.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/b306a8d3d6a5a70719ba1c57f518c4e8cd0e12781cbee26f69b032f940db1986.jpg)

![b3f7e46e7793e82562c85eed084e6b73f14b9aaba8537edbe5d80d1f6cc054c0.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/b3f7e46e7793e82562c85eed084e6b73f14b9aaba8537edbe5d80d1f6cc054c0.jpg)

![b4a30cc55b8ae603484fe39ccc0842463180b096600772e150da24ce4378ae40.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/b4a30cc55b8ae603484fe39ccc0842463180b096600772e150da24ce4378ae40.jpg)

![b4b6dae735eb68471cbfdf6f1371965c7ae8cd81c73a8cac9cdb4c488e67ab6a.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/b4b6dae735eb68471cbfdf6f1371965c7ae8cd81c73a8cac9cdb4c488e67ab6a.jpg)

![b682673b098694164edf750a79cd129dfd8eaac7adb23206355210199143aee2.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/b682673b098694164edf750a79cd129dfd8eaac7adb23206355210199143aee2.jpg)

![b8bc108ba74968c168f1808ca8add8e1f5eec47eb1d04d68cd016019f23bc41b.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/b8bc108ba74968c168f1808ca8add8e1f5eec47eb1d04d68cd016019f23bc41b.jpg)

![b92cf21a61470b78a2da60f4f52d9271bd3df359442a5e3d468a8d3796e3dddf.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/b92cf21a61470b78a2da60f4f52d9271bd3df359442a5e3d468a8d3796e3dddf.jpg)

![b9d26a80515d45ac817c57e8d7db34f0108beb1e231756ac920664ccb0a45833.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/b9d26a80515d45ac817c57e8d7db34f0108beb1e231756ac920664ccb0a45833.jpg)

![b9e4453a2f44e1fc663f8cc2b5f9a015e9a8649afb7327799990e9f649b0fe7d.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/b9e4453a2f44e1fc663f8cc2b5f9a015e9a8649afb7327799990e9f649b0fe7d.jpg)

![bac97537290e8f17c01f9be1ce568d8de1cb916096eaf1def92e9d19f7b00c60.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/bac97537290e8f17c01f9be1ce568d8de1cb916096eaf1def92e9d19f7b00c60.jpg)

![be33da0ae64ee9d770899bb9cd9ed781aa6e529bff619442d51345778acd2d2c.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/be33da0ae64ee9d770899bb9cd9ed781aa6e529bff619442d51345778acd2d2c.jpg)

![bf25c46543610df6372d2c2a00e9a40d4954c3e2565c29e166ed2f7b2e48273e.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/bf25c46543610df6372d2c2a00e9a40d4954c3e2565c29e166ed2f7b2e48273e.jpg)

![c157dca30954e1ed2b4ffaed01a2f94172f635df9d9d8d9c74b61f7042586760.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/c157dca30954e1ed2b4ffaed01a2f94172f635df9d9d8d9c74b61f7042586760.jpg)

![c316c044965a0452ce0c3fbef76f7d7f089497fab885abcebd60033106873b25.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/c316c044965a0452ce0c3fbef76f7d7f089497fab885abcebd60033106873b25.jpg)

![c343b00cb05f3021c1c9dd556da94cd9d65de8a4ec86f45e91f0b36a28b47433.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/c343b00cb05f3021c1c9dd556da94cd9d65de8a4ec86f45e91f0b36a28b47433.jpg)

![cd5f7ee1e9e1b135a0e2859d167d5c68624fb7652ff412fc824c435400a01543.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/cd5f7ee1e9e1b135a0e2859d167d5c68624fb7652ff412fc824c435400a01543.jpg)

![ce07f2685fba17ab3af1c89c995e37d30c8890cbf988ffbfaa72e208bdaff711.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/ce07f2685fba17ab3af1c89c995e37d30c8890cbf988ffbfaa72e208bdaff711.jpg)

![cfb6270bc52b01ae0c7a85e74927e448f73177a493e8ffdbd80a1e2beb84842b.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/cfb6270bc52b01ae0c7a85e74927e448f73177a493e8ffdbd80a1e2beb84842b.jpg)

![d174c3d668d5109249ac66f79bdb958ab305ce431905c7fd3bec6ccf92f3c8bc.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/d174c3d668d5109249ac66f79bdb958ab305ce431905c7fd3bec6ccf92f3c8bc.jpg)

![d33288fe04a2e5522ac81c29c2cbf43b7b4bf72ba28227821e0c23b2d988dc43.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/d33288fe04a2e5522ac81c29c2cbf43b7b4bf72ba28227821e0c23b2d988dc43.jpg)

![d4eac1bb7b12bf7abf45c32d53f392cabc79420453c5f41f91790c92a1b04a46.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/d4eac1bb7b12bf7abf45c32d53f392cabc79420453c5f41f91790c92a1b04a46.jpg)

![d52c3f1fc0f427941481b5c80aa547e985346644d65ceb3868ff1270dc8a94c9.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/d52c3f1fc0f427941481b5c80aa547e985346644d65ceb3868ff1270dc8a94c9.jpg)

![d6e3f29cabb7fd01ea26d0b32bde5b7dc6e747b171663e4fabf08c936526692b.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/d6e3f29cabb7fd01ea26d0b32bde5b7dc6e747b171663e4fabf08c936526692b.jpg)

![d702c8cbe85169a787448c28cb2a4eeee044eca27cd01b20a7517f03a3c68ec8.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/d702c8cbe85169a787448c28cb2a4eeee044eca27cd01b20a7517f03a3c68ec8.jpg)

![d982f5fe5bba25483b534af2fc8ebe3fdf680074772014a337f65c4cdf3b2594.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/d982f5fe5bba25483b534af2fc8ebe3fdf680074772014a337f65c4cdf3b2594.jpg)

![dbaf8dddf2e2335174f4d3b48b0101c25bcfb8aa824de6cfbea7c79f12a63310.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/dbaf8dddf2e2335174f4d3b48b0101c25bcfb8aa824de6cfbea7c79f12a63310.jpg)

![dcb528b003b65bf5b3e28a3832e0ae404725494e410c842aaa6b012f674453c9.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/dcb528b003b65bf5b3e28a3832e0ae404725494e410c842aaa6b012f674453c9.jpg)

![dcfc088961d3b5ee10a5e6d0026d4ff0aebbe4716714bf8d82aefc0d33cff603.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/dcfc088961d3b5ee10a5e6d0026d4ff0aebbe4716714bf8d82aefc0d33cff603.jpg)

![de44dcea3cef8a5dd032206ae4209d7a15552e7f842d1e08b33e770ff924182c.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/de44dcea3cef8a5dd032206ae4209d7a15552e7f842d1e08b33e770ff924182c.jpg)

![df0c975778beaf7df991cfaf7840ae1da58d541e042507cfa83d12fcb9a45040.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/df0c975778beaf7df991cfaf7840ae1da58d541e042507cfa83d12fcb9a45040.jpg)

![dfe9dd41321d65b0afee887b8b5257c47436654dff8c462ad46dae28949003ed.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/dfe9dd41321d65b0afee887b8b5257c47436654dff8c462ad46dae28949003ed.jpg)

![e4971c8943ac9ff4359484f2c2a3bb74d43c02d057b5d9dd13cd256d56e144d0.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/e4971c8943ac9ff4359484f2c2a3bb74d43c02d057b5d9dd13cd256d56e144d0.jpg)

![e54b8a07c95f4ad4fde00541aa5fab4ed0476c832d2cd3214e89e46813aafd73.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/e54b8a07c95f4ad4fde00541aa5fab4ed0476c832d2cd3214e89e46813aafd73.jpg)

![e8adaac58f99a9ce682f20163757a47b181e81aee10f0a02061f0412070076ce.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/e8adaac58f99a9ce682f20163757a47b181e81aee10f0a02061f0412070076ce.jpg)

![ea477b58f112539b30e44430c57f64175b854af21184c20a2ccae36ba0101144.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/ea477b58f112539b30e44430c57f64175b854af21184c20a2ccae36ba0101144.jpg)

![eae0d6903d86eb00912874d16d7c5044786b65fc66dc1e165ceb4bf16feb8e89.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/eae0d6903d86eb00912874d16d7c5044786b65fc66dc1e165ceb4bf16feb8e89.jpg)

![ecaf765249feb978406e547fe56a2b4fd9063d1012b3e0a77489fba8287907a1.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/ecaf765249feb978406e547fe56a2b4fd9063d1012b3e0a77489fba8287907a1.jpg)

![f0376a4ede57c0c4f281f997fff4f1deed1fc7666156ea8a9c5bd956c29f07b4.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/f0376a4ede57c0c4f281f997fff4f1deed1fc7666156ea8a9c5bd956c29f07b4.jpg)

![f10c6e34a0e727df8078ff74c8f6a7215d43a210d302b5f18f929f51d51533bb.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/f10c6e34a0e727df8078ff74c8f6a7215d43a210d302b5f18f929f51d51533bb.jpg)

![f2a3afab0ebcc68fa1106b9892499ba3c2ec860e1d145d09be405f2bc1d54de5.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/f2a3afab0ebcc68fa1106b9892499ba3c2ec860e1d145d09be405f2bc1d54de5.jpg)

![f2d65267ad38405bf7c8b8c04c9e94a10082120b4bc3f58dc04dcb7ab8a02ab1.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/f2d65267ad38405bf7c8b8c04c9e94a10082120b4bc3f58dc04dcb7ab8a02ab1.jpg)

![f486e25882e823f22728b0a8e3fcc3c8241f69b24138f15766e5981146be594c.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/f486e25882e823f22728b0a8e3fcc3c8241f69b24138f15766e5981146be594c.jpg)

![fb900f786d94d5c58f0c5fb918f836bbb7ef2a0231e43c36feee3e4b54ed7969.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/fb900f786d94d5c58f0c5fb918f836bbb7ef2a0231e43c36feee3e4b54ed7969.jpg)

![fc50811eb07e62a3e6912ade744b98b546604f43b1657468bd5e7833469ba033.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/fc50811eb07e62a3e6912ade744b98b546604f43b1657468bd5e7833469ba033.jpg)

![fda91ab30910afed979d19997c394c5ef270d0f6cfe475d001eecbf159e15331.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/fda91ab30910afed979d19997c394c5ef270d0f6cfe475d001eecbf159e15331.jpg)

![fe585f174edbfbd8fc111138e4a644529090866de02ee5237f2f0aaafdf1d904.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/fe585f174edbfbd8fc111138e4a644529090866de02ee5237f2f0aaafdf1d904.jpg)

![fef3ec195d41e0a55e5eee76a199e12b6d15d9d1c674769b6b7ee754d01c048d.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/fef3ec195d41e0a55e5eee76a199e12b6d15d9d1c674769b6b7ee754d01c048d.jpg)

![ff3e1156842f412e76b4f2c2d8e0eeaa9f9d1e76b92e6e0894815845a9dd8937.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/images/ff3e1156842f412e76b4f2c2d8e0eeaa9f9d1e76b92e6e0894815845a9dd8937.jpg)

### Tables

![08af85688bc9bfbed17f7deeef26203481940d0945a40f2f625551128d173329.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/08af85688bc9bfbed17f7deeef26203481940d0945a40f2f625551128d173329.jpg)

![0acb900efc57e86126b1c0166479ecc9a3b8733dc90ad9a1ffdce7eb2cb27f75.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/0acb900efc57e86126b1c0166479ecc9a3b8733dc90ad9a1ffdce7eb2cb27f75.jpg)

![1d27a0b8feecfa05fab028dc27d6d04449782c5e4b560b88e064e90b4ae23da4.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/1d27a0b8feecfa05fab028dc27d6d04449782c5e4b560b88e064e90b4ae23da4.jpg)

![22a83133fc2dccda677b401145917043cef72ce76f4ec4ae9bedfefbe9462a8a.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/22a83133fc2dccda677b401145917043cef72ce76f4ec4ae9bedfefbe9462a8a.jpg)

![3143009653c1526e9223146e55266b770c2407af5602c067be0f482aaf30b5d2.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/3143009653c1526e9223146e55266b770c2407af5602c067be0f482aaf30b5d2.jpg)

![4216df4ead93f4c77836f066acc70ff57b0db81f0e533c3c5f6b1bf9089ad896.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/4216df4ead93f4c77836f066acc70ff57b0db81f0e533c3c5f6b1bf9089ad896.jpg)

![513e68ce69898c2434eee0289f3dbe1dd6d2b4dbfd7d00522d6448bc1fc5eafd.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/513e68ce69898c2434eee0289f3dbe1dd6d2b4dbfd7d00522d6448bc1fc5eafd.jpg)

![51dd804edd9b3575e98ae40b333c2ae9e311a9cdbe4dde3fe39f8232cfb05b6a.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/51dd804edd9b3575e98ae40b333c2ae9e311a9cdbe4dde3fe39f8232cfb05b6a.jpg)

![531d7a1d7e73a196b4d442cdd5039e33856e3984a522cd1de3e7256db78bb9b0.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/531d7a1d7e73a196b4d442cdd5039e33856e3984a522cd1de3e7256db78bb9b0.jpg)

![553ce8ffbbe8ed22ef04b89430595ee9064fbe9dea63989d2c4fae328d065b94.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/553ce8ffbbe8ed22ef04b89430595ee9064fbe9dea63989d2c4fae328d065b94.jpg)

![61ad1f8ed0780108474960dee8a8ee28361d07b9c60325211040d37c4f23a1d8.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/61ad1f8ed0780108474960dee8a8ee28361d07b9c60325211040d37c4f23a1d8.jpg)

![6590cff5370152344f2140a0e32cb512d407335f3d7252cb4cc481f70f81a809.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/6590cff5370152344f2140a0e32cb512d407335f3d7252cb4cc481f70f81a809.jpg)

![74ee3d2677d01eda1a122af4615904499461972df24d3aec3cb6af5214fa964a.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/74ee3d2677d01eda1a122af4615904499461972df24d3aec3cb6af5214fa964a.jpg)

![7ccde6f3aaa7e1333f0f14ccfb21c91f36b55f3e8e60223e8d8322870649f4ee.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/7ccde6f3aaa7e1333f0f14ccfb21c91f36b55f3e8e60223e8d8322870649f4ee.jpg)

![7da5f1020939b575a2dc7af5c79c18f17840f603cb7c87a952eb7b1c095edc5b.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/7da5f1020939b575a2dc7af5c79c18f17840f603cb7c87a952eb7b1c095edc5b.jpg)

![81ae27b5d011cceed01c1443520f140bd5ddf7139cbcd40fb99da935f3b60bfb.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/81ae27b5d011cceed01c1443520f140bd5ddf7139cbcd40fb99da935f3b60bfb.jpg)

![87a2ecc6eec6e070e487a68e579e74251276fffe89fcb40a8221e5f8ccad674c.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/87a2ecc6eec6e070e487a68e579e74251276fffe89fcb40a8221e5f8ccad674c.jpg)

![93a04b3fff5e95e39bfc29dc06235784bfdff9c2a67ca5165a91b084bca953be.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/93a04b3fff5e95e39bfc29dc06235784bfdff9c2a67ca5165a91b084bca953be.jpg)

![95680b1a550f3027ccf8d6db0747f65c5fe58d020beb7ed39f54748fea8d350f.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/95680b1a550f3027ccf8d6db0747f65c5fe58d020beb7ed39f54748fea8d350f.jpg)

![a15459e07ea6488d2b37a3c7d3bf2e06ac4dacaad9073af90e50659e6466287a.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/a15459e07ea6488d2b37a3c7d3bf2e06ac4dacaad9073af90e50659e6466287a.jpg)

![a502326eb4c9649dc731f7dedee7ce642faa81d5869376bc4e352f90ea1987bd.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/a502326eb4c9649dc731f7dedee7ce642faa81d5869376bc4e352f90ea1987bd.jpg)

![a5e7e14322dfe45e00d9658544f934fc49b3126f9df161e41dd64adb0357528a.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/a5e7e14322dfe45e00d9658544f934fc49b3126f9df161e41dd64adb0357528a.jpg)

![d717bda29c20b772495c0d9f84de6945fdb55fab682b0d2dcbccc0377aa8d563.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/d717bda29c20b772495c0d9f84de6945fdb55fab682b0d2dcbccc0377aa8d563.jpg)

![ded25947d862d4be8557b9a9aabaec0e83f91ddedf096454a1a4b72528a74b0b.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/ded25947d862d4be8557b9a9aabaec0e83f91ddedf096454a1a4b72528a74b0b.jpg)

![df654887fbecbf83e23d0cffd3c18ea0621822e4924aeef39d7aa3117cb07a44.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/df654887fbecbf83e23d0cffd3c18ea0621822e4924aeef39d7aa3117cb07a44.jpg)

![e153649032b5fffed9c505128367bf7fa19a699de832cc68f66bba7f238169e1.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/e153649032b5fffed9c505128367bf7fa19a699de832cc68f66bba7f238169e1.jpg)

![eac7a18d557d54ad16b424581e8fb61364af700a4cc9520938bea982b550d654.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/eac7a18d557d54ad16b424581e8fb61364af700a4cc9520938bea982b550d654.jpg)

![f2673c1437f1fe8b29ece6eecf0fdcc6b7b6174a98591b930a8e4a633d16f2e1.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/f2673c1437f1fe8b29ece6eecf0fdcc6b7b6174a98591b930a8e4a633d16f2e1.jpg)

![f54c8a1890afb74fcdc6d3924f4d31238c9bc17d4a5438d6981dc560e4a2e646.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/f54c8a1890afb74fcdc6d3924f4d31238c9bc17d4a5438d6981dc560e4a2e646.jpg)

![f671132cffe36144d44101d720763245f7974df1940952eea0ccb56e294b0819.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/f671132cffe36144d44101d720763245f7974df1940952eea0ccb56e294b0819.jpg)

![f8bf283cc9ebfed61870f2d16256f3d7683daacc0c6acc7eb3484455e35f73b7.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/f8bf283cc9ebfed61870f2d16256f3d7683daacc0c6acc7eb3484455e35f73b7.jpg)

![fc0861d11b01e3ae377494acbbc05006712f68a4279c69093eefa44811d7f633.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/fc0861d11b01e3ae377494acbbc05006712f68a4279c69093eefa44811d7f633.jpg)

![fc278d2b81a4dae3b97ee624346aa43dbdfd647414ae3c8f6c0486b2b573f038.jpg](../iclr_results/130_PhysBench_%20Benchmarking%20and%20Enhancing%20Vision-Language%20Models%20for%20Physical%20World%20Understanding/tables/fc278d2b81a4dae3b97ee624346aa43dbdfd647414ae3c8f6c0486b2b573f038.jpg)

## Iterative Nash Policy Optimization: Aligning LLMs with General Preferences via No-Regret Learning


### Tables

![4cd6775b3f26e4a227aac327ee172f06d11a667e4a1dd214c2d3c1f39ddea91c.jpg](../iclr_results/131_Iterative%20Nash%20Policy%20Optimization_%20Aligning%20LLMs%20with%20General%20Preferences%20via%20No-Regret%20Learning/tables/4cd6775b3f26e4a227aac327ee172f06d11a667e4a1dd214c2d3c1f39ddea91c.jpg)

![4edcbd4c7f075edd3ca0b51c0e8c94d8c5975f8b4454cd3cfbd290e583bc8f80.jpg](../iclr_results/131_Iterative%20Nash%20Policy%20Optimization_%20Aligning%20LLMs%20with%20General%20Preferences%20via%20No-Regret%20Learning/tables/4edcbd4c7f075edd3ca0b51c0e8c94d8c5975f8b4454cd3cfbd290e583bc8f80.jpg)

![857f5370196868c048401b35bfa728023b09cf2f30cacbcb729f9caa0bd5883e.jpg](../iclr_results/131_Iterative%20Nash%20Policy%20Optimization_%20Aligning%20LLMs%20with%20General%20Preferences%20via%20No-Regret%20Learning/tables/857f5370196868c048401b35bfa728023b09cf2f30cacbcb729f9caa0bd5883e.jpg)

![d0f55f1d0ab5132d2cd6b0170f19f75d07eb325cae5253ee9dacbb0fa640ce16.jpg](../iclr_results/131_Iterative%20Nash%20Policy%20Optimization_%20Aligning%20LLMs%20with%20General%20Preferences%20via%20No-Regret%20Learning/tables/d0f55f1d0ab5132d2cd6b0170f19f75d07eb325cae5253ee9dacbb0fa640ce16.jpg)

![e92eca658064816a4c20a7dff27298557105338df449a8b77dc38f622a84866b.jpg](../iclr_results/131_Iterative%20Nash%20Policy%20Optimization_%20Aligning%20LLMs%20with%20General%20Preferences%20via%20No-Regret%20Learning/tables/e92eca658064816a4c20a7dff27298557105338df449a8b77dc38f622a84866b.jpg)

## Steering Protein Family Design through Profile Bayesian Flow


### Images

![13857685e647553c88674784acd23754ed410f76017e4baded142a413345300d.jpg](../iclr_results/132_Steering%20Protein%20Family%20Design%20through%20Profile%20Bayesian%20Flow/images/13857685e647553c88674784acd23754ed410f76017e4baded142a413345300d.jpg)

![3c4e7c5bae0df66fbd2248d0f07aa91258d8494ba8b1eda91cad7a6047ca9ce5.jpg](../iclr_results/132_Steering%20Protein%20Family%20Design%20through%20Profile%20Bayesian%20Flow/images/3c4e7c5bae0df66fbd2248d0f07aa91258d8494ba8b1eda91cad7a6047ca9ce5.jpg)

![4a2e332f8d4fe8d276d0607adebc1cc620aa6b63d6fb83317454a965a4dabc51.jpg](../iclr_results/132_Steering%20Protein%20Family%20Design%20through%20Profile%20Bayesian%20Flow/images/4a2e332f8d4fe8d276d0607adebc1cc620aa6b63d6fb83317454a965a4dabc51.jpg)

![5a81ed5f8f45fec33c06915d724d85fdd2b88646996f32ba4efff18054e164da.jpg](../iclr_results/132_Steering%20Protein%20Family%20Design%20through%20Profile%20Bayesian%20Flow/images/5a81ed5f8f45fec33c06915d724d85fdd2b88646996f32ba4efff18054e164da.jpg)

![70314bf40dce7c81e5d0ae7a315d0886615f87eac53dd6fe6cbb81b535bb55f4.jpg](../iclr_results/132_Steering%20Protein%20Family%20Design%20through%20Profile%20Bayesian%20Flow/images/70314bf40dce7c81e5d0ae7a315d0886615f87eac53dd6fe6cbb81b535bb55f4.jpg)

![7c830688d558d05b31948dbfc39c4f5cf49b6951f4cc8ae63ebf640352328580.jpg](../iclr_results/132_Steering%20Protein%20Family%20Design%20through%20Profile%20Bayesian%20Flow/images/7c830688d558d05b31948dbfc39c4f5cf49b6951f4cc8ae63ebf640352328580.jpg)

![8c079a5180e00ffb6fdefc69dd4989df97597112a3132f42ace5b013b41df2a7.jpg](../iclr_results/132_Steering%20Protein%20Family%20Design%20through%20Profile%20Bayesian%20Flow/images/8c079a5180e00ffb6fdefc69dd4989df97597112a3132f42ace5b013b41df2a7.jpg)

![a479126a00c7209699a637c05387ba9d2dc6210c52db4f2ed863b3ef2b47f127.jpg](../iclr_results/132_Steering%20Protein%20Family%20Design%20through%20Profile%20Bayesian%20Flow/images/a479126a00c7209699a637c05387ba9d2dc6210c52db4f2ed863b3ef2b47f127.jpg)

![ba9c1f32d5e1026abe9b1d2b16cdf595d31347a065e20a8f5a687f2252e70bce.jpg](../iclr_results/132_Steering%20Protein%20Family%20Design%20through%20Profile%20Bayesian%20Flow/images/ba9c1f32d5e1026abe9b1d2b16cdf595d31347a065e20a8f5a687f2252e70bce.jpg)

![bbf505adfbff2af836787c31dcdb954caa4cfc4c083f124458a8c84cddc58910.jpg](../iclr_results/132_Steering%20Protein%20Family%20Design%20through%20Profile%20Bayesian%20Flow/images/bbf505adfbff2af836787c31dcdb954caa4cfc4c083f124458a8c84cddc58910.jpg)

![c557b6b2dd030cff37e524955751061ca506ee9976f54ccc7bebce40b9333ace.jpg](../iclr_results/132_Steering%20Protein%20Family%20Design%20through%20Profile%20Bayesian%20Flow/images/c557b6b2dd030cff37e524955751061ca506ee9976f54ccc7bebce40b9333ace.jpg)

![cd1de619cc349ff17808cfe8ed0a8b7046ca693e1124c620f1851921430b94e7.jpg](../iclr_results/132_Steering%20Protein%20Family%20Design%20through%20Profile%20Bayesian%20Flow/images/cd1de619cc349ff17808cfe8ed0a8b7046ca693e1124c620f1851921430b94e7.jpg)

### Tables

![5d377333e62cd7c872df74892b50c73fb342792d4a28a6b0dbd7f38c5028efd7.jpg](../iclr_results/132_Steering%20Protein%20Family%20Design%20through%20Profile%20Bayesian%20Flow/tables/5d377333e62cd7c872df74892b50c73fb342792d4a28a6b0dbd7f38c5028efd7.jpg)

![7e8222542b524889dc87e4bdafcae4748c3e045a98c3763ea795d714f26c6249.jpg](../iclr_results/132_Steering%20Protein%20Family%20Design%20through%20Profile%20Bayesian%20Flow/tables/7e8222542b524889dc87e4bdafcae4748c3e045a98c3763ea795d714f26c6249.jpg)

![95827e4c3a31090e8b4e5b270970583168338ec4c3aabb6b1d936f51e9f7a2df.jpg](../iclr_results/132_Steering%20Protein%20Family%20Design%20through%20Profile%20Bayesian%20Flow/tables/95827e4c3a31090e8b4e5b270970583168338ec4c3aabb6b1d936f51e9f7a2df.jpg)

![c5f96be61cfd6139b26f3adf24dc62e6e5b082d4862fdbb8c36080a57df87ff6.jpg](../iclr_results/132_Steering%20Protein%20Family%20Design%20through%20Profile%20Bayesian%20Flow/tables/c5f96be61cfd6139b26f3adf24dc62e6e5b082d4862fdbb8c36080a57df87ff6.jpg)

![c91d7af75c19bd0b808b4c5ff63044f306b74bbd00c30f5968ce58cd4a457bd2.jpg](../iclr_results/132_Steering%20Protein%20Family%20Design%20through%20Profile%20Bayesian%20Flow/tables/c91d7af75c19bd0b808b4c5ff63044f306b74bbd00c30f5968ce58cd4a457bd2.jpg)

![e699a41d6030d582b9b107d98ee5cf65ed615f2d239ca8ef670d4e9e6b3e21e6.jpg](../iclr_results/132_Steering%20Protein%20Family%20Design%20through%20Profile%20Bayesian%20Flow/tables/e699a41d6030d582b9b107d98ee5cf65ed615f2d239ca8ef670d4e9e6b3e21e6.jpg)

![ed059a9f4a797933123a73562686cda9ee8d1ce75aaa6c132cd98393fa4830a1.jpg](../iclr_results/132_Steering%20Protein%20Family%20Design%20through%20Profile%20Bayesian%20Flow/tables/ed059a9f4a797933123a73562686cda9ee8d1ce75aaa6c132cd98393fa4830a1.jpg)

![fe0451a34e7ebfa9f0c0a2028e5fadce1800099112277578ae394e8aaaa13da2.jpg](../iclr_results/132_Steering%20Protein%20Family%20Design%20through%20Profile%20Bayesian%20Flow/tables/fe0451a34e7ebfa9f0c0a2028e5fadce1800099112277578ae394e8aaaa13da2.jpg)

## Unlearning-based Neural Interpretations


### Images

![0d3fc5e183e1de06e65ba6b464b32e8457e91a6c55b89495dfba084235975cb2.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/0d3fc5e183e1de06e65ba6b464b32e8457e91a6c55b89495dfba084235975cb2.jpg)

![119278d95de5fa1bf4e35ebcab6845be88b86070eaa280e85a6b2625d7908d0e.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/119278d95de5fa1bf4e35ebcab6845be88b86070eaa280e85a6b2625d7908d0e.jpg)

![1d700c203ee5d6f9bfb7f05146940ba9d461833df74b3a45b0311c315a9d0e43.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/1d700c203ee5d6f9bfb7f05146940ba9d461833df74b3a45b0311c315a9d0e43.jpg)

![235275519dbbfe9e2e8740857f57f5c51671bfde64ff9d35302fce0b43fbe681.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/235275519dbbfe9e2e8740857f57f5c51671bfde64ff9d35302fce0b43fbe681.jpg)

![2541856019d4a3f0a9f4e3bad847331100815701ada4c5a059815b92f7d64e1d.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/2541856019d4a3f0a9f4e3bad847331100815701ada4c5a059815b92f7d64e1d.jpg)

![32807deb6d85d0de96231a555b1d7605b67fcb91a7eb7f50f1ad293b567e0c5b.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/32807deb6d85d0de96231a555b1d7605b67fcb91a7eb7f50f1ad293b567e0c5b.jpg)

![360d1464168355ff8637be0b560692c28598c43dcb392c2067144d59d550e27a.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/360d1464168355ff8637be0b560692c28598c43dcb392c2067144d59d550e27a.jpg)

![3e57d9ab92c1da65cf3b7441a9749c2a7edae3051ed0b3c9b64b63dda3140d60.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/3e57d9ab92c1da65cf3b7441a9749c2a7edae3051ed0b3c9b64b63dda3140d60.jpg)

![4426ba9d133c9d02b094f22c79fb1b1cd62e71a64ce06207648720129f31db61.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/4426ba9d133c9d02b094f22c79fb1b1cd62e71a64ce06207648720129f31db61.jpg)

![4defcb33c3cc14715b16f9d34bcc51662976e00421778702396874eacc51b95b.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/4defcb33c3cc14715b16f9d34bcc51662976e00421778702396874eacc51b95b.jpg)

![5233ec1c779cbe8aabc857a903dbda790e3183c2710fbc2685c8a431fa28ffa9.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/5233ec1c779cbe8aabc857a903dbda790e3183c2710fbc2685c8a431fa28ffa9.jpg)

![56ed8ea192ea891d7e4e1680cd3c39bb9281c0b19f0b4f9e6feca956699b417a.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/56ed8ea192ea891d7e4e1680cd3c39bb9281c0b19f0b4f9e6feca956699b417a.jpg)

![57c99ad13bf6de2ae88144158349411d1b82720611bd0f050d0384c201cbb159.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/57c99ad13bf6de2ae88144158349411d1b82720611bd0f050d0384c201cbb159.jpg)

![58344fbfc123a1fe78a440ce83b126c6a7e5f2a2c5935a3597708bb859b8ab4a.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/58344fbfc123a1fe78a440ce83b126c6a7e5f2a2c5935a3597708bb859b8ab4a.jpg)

![590e45d12ed43bff8a7f9c945575501d138d3db59dcb61c74a15206925607497.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/590e45d12ed43bff8a7f9c945575501d138d3db59dcb61c74a15206925607497.jpg)

![5b820c66952f1e83e7069bcac96aed9758e08c2d06979b003518553fbbda7f6e.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/5b820c66952f1e83e7069bcac96aed9758e08c2d06979b003518553fbbda7f6e.jpg)

![6fbf75438d01e0e716ee5a47e446b0b8f7b48d06c4fb138c7e29c6909185f05d.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/6fbf75438d01e0e716ee5a47e446b0b8f7b48d06c4fb138c7e29c6909185f05d.jpg)

![9b386bf832d7918bd21224b2b7a85e51fab7647787a133df6ae8432614b6d53c.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/9b386bf832d7918bd21224b2b7a85e51fab7647787a133df6ae8432614b6d53c.jpg)

![ae157e4ec95a59e76b0af5b6e6b5ab22a7e1bd00cb7e1b5f766fd1b11a722d6a.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/ae157e4ec95a59e76b0af5b6e6b5ab22a7e1bd00cb7e1b5f766fd1b11a722d6a.jpg)

![b2081be33d7dea603528fef4ae99c2858127fb704f15adebeb97991423d8d659.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/b2081be33d7dea603528fef4ae99c2858127fb704f15adebeb97991423d8d659.jpg)

![d721d4f7674e12ea32c750bf4bc53b92ac2ea65167368e3f8ebb8941e1dea87b.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/d721d4f7674e12ea32c750bf4bc53b92ac2ea65167368e3f8ebb8941e1dea87b.jpg)

![dc958f5d3e94dc3a9daf62b792d26417a38cf18814a14673a6c6d0e5a3425c89.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/dc958f5d3e94dc3a9daf62b792d26417a38cf18814a14673a6c6d0e5a3425c89.jpg)

![e3d5170e72924d9e7481c1193ddd14f98ad9848a6ef89f46607c7271c894aa82.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/e3d5170e72924d9e7481c1193ddd14f98ad9848a6ef89f46607c7271c894aa82.jpg)

![e4b49cdd69b1b0226610a15ef675c928e492f2e41d4b96d52faa607f4131162c.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/e4b49cdd69b1b0226610a15ef675c928e492f2e41d4b96d52faa607f4131162c.jpg)

![f20cded934bd151450da25e75718947322b4a4fc066e37b3360154709be12c52.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/images/f20cded934bd151450da25e75718947322b4a4fc066e37b3360154709be12c52.jpg)

### Tables

![2809241ba6ca1718240fab917e4fa6a1ee22b8301a1dd913647fdfeee78d6ff2.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/tables/2809241ba6ca1718240fab917e4fa6a1ee22b8301a1dd913647fdfeee78d6ff2.jpg)

![54a9a536f605cc4563dab60e2037583e6d7462446f8eee4f44f7fa0ca093962c.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/tables/54a9a536f605cc4563dab60e2037583e6d7462446f8eee4f44f7fa0ca093962c.jpg)

![5e33057939b977d36d52048d382fdc9f04e5e88c7c05e86a953b504dea1d7377.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/tables/5e33057939b977d36d52048d382fdc9f04e5e88c7c05e86a953b504dea1d7377.jpg)

![71b0f381b7667c08137d5268a0706f88192b4db4ac0ce14339bd0c12b2b6894f.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/tables/71b0f381b7667c08137d5268a0706f88192b4db4ac0ce14339bd0c12b2b6894f.jpg)

![7619a8d6b9bc93fac67ab847a9ec44d68b62c8d731b4029554b778c292e8a19c.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/tables/7619a8d6b9bc93fac67ab847a9ec44d68b62c8d731b4029554b778c292e8a19c.jpg)

![861cfce7c6090552cdb76fdbe60feb808e11e4ac388023212e762f56a7615dd6.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/tables/861cfce7c6090552cdb76fdbe60feb808e11e4ac388023212e762f56a7615dd6.jpg)

![9643f1896adc3820e36a946d2e250f1a841f1a1f8651e37a501b855ec38deceb.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/tables/9643f1896adc3820e36a946d2e250f1a841f1a1f8651e37a501b855ec38deceb.jpg)

![97e03c06290b5606fd32398625fd392a6574c1bd571f7c916eb298ce23b52667.jpg](../iclr_results/133_Unlearning-based%20Neural%20Interpretations/tables/97e03c06290b5606fd32398625fd392a6574c1bd571f7c916eb298ce23b52667.jpg)

## On the Hölder Stability of Multiset and Graph Neural Networks


### Images

![24a9622caace5963628108b93631dce7a687b0d3d4470f71372b80f5c530871e.jpg](../iclr_results/134_On%20the%20H%C3%B6lder%20Stability%20of%20Multiset%20and%20Graph%20Neural%20Networks/images/24a9622caace5963628108b93631dce7a687b0d3d4470f71372b80f5c530871e.jpg)

![29278a1ef380a220300576d8e0c0837d61420c0497bbbbe94c0fd7376ce2d632.jpg](../iclr_results/134_On%20the%20H%C3%B6lder%20Stability%20of%20Multiset%20and%20Graph%20Neural%20Networks/images/29278a1ef380a220300576d8e0c0837d61420c0497bbbbe94c0fd7376ce2d632.jpg)

![46692f7f798a926f21ad5a71a6e225108014645f76a590171cefa6f944a3160a.jpg](../iclr_results/134_On%20the%20H%C3%B6lder%20Stability%20of%20Multiset%20and%20Graph%20Neural%20Networks/images/46692f7f798a926f21ad5a71a6e225108014645f76a590171cefa6f944a3160a.jpg)

![6798d167593618fb6dea6d9818124dd2351a6335ac68e9d87296b9ff6de81c5f.jpg](../iclr_results/134_On%20the%20H%C3%B6lder%20Stability%20of%20Multiset%20and%20Graph%20Neural%20Networks/images/6798d167593618fb6dea6d9818124dd2351a6335ac68e9d87296b9ff6de81c5f.jpg)

![abfd77b6faa75462ee6298212998daeb3265b1e32969728eae522dcc76c09db4.jpg](../iclr_results/134_On%20the%20H%C3%B6lder%20Stability%20of%20Multiset%20and%20Graph%20Neural%20Networks/images/abfd77b6faa75462ee6298212998daeb3265b1e32969728eae522dcc76c09db4.jpg)

![b035399168b4e9212dd8f42af8a293b113c21c7711400e801452610093d9605d.jpg](../iclr_results/134_On%20the%20H%C3%B6lder%20Stability%20of%20Multiset%20and%20Graph%20Neural%20Networks/images/b035399168b4e9212dd8f42af8a293b113c21c7711400e801452610093d9605d.jpg)

![c0cc20b91ad0c4526ca093e4a58ea06d5d99d5c5be35a5a32cf108c88d7e3e6f.jpg](../iclr_results/134_On%20the%20H%C3%B6lder%20Stability%20of%20Multiset%20and%20Graph%20Neural%20Networks/images/c0cc20b91ad0c4526ca093e4a58ea06d5d99d5c5be35a5a32cf108c88d7e3e6f.jpg)

![fdf27102ff82005337991446cc84c239efb39eb9bb515a523fa53fd40b255792.jpg](../iclr_results/134_On%20the%20H%C3%B6lder%20Stability%20of%20Multiset%20and%20Graph%20Neural%20Networks/images/fdf27102ff82005337991446cc84c239efb39eb9bb515a523fa53fd40b255792.jpg)

### Tables

![051a9b2fa033ddd23ed66a7ec2dd04b072fe3ef58dee58f20d4147cc09adb57b.jpg](../iclr_results/134_On%20the%20H%C3%B6lder%20Stability%20of%20Multiset%20and%20Graph%20Neural%20Networks/tables/051a9b2fa033ddd23ed66a7ec2dd04b072fe3ef58dee58f20d4147cc09adb57b.jpg)

![2224d8899359fd35691843a943a69dc98ef1aae1a57365db8234c9640cee317f.jpg](../iclr_results/134_On%20the%20H%C3%B6lder%20Stability%20of%20Multiset%20and%20Graph%20Neural%20Networks/tables/2224d8899359fd35691843a943a69dc98ef1aae1a57365db8234c9640cee317f.jpg)

![598b845b16a5b318b0a2f98cbf417deafde145ff9094c34c80ab30188750baaa.jpg](../iclr_results/134_On%20the%20H%C3%B6lder%20Stability%20of%20Multiset%20and%20Graph%20Neural%20Networks/tables/598b845b16a5b318b0a2f98cbf417deafde145ff9094c34c80ab30188750baaa.jpg)

![94132865f9f81037494fa7fafe4aa9c73eb2f86fe4fe477404d73689360ee5a4.jpg](../iclr_results/134_On%20the%20H%C3%B6lder%20Stability%20of%20Multiset%20and%20Graph%20Neural%20Networks/tables/94132865f9f81037494fa7fafe4aa9c73eb2f86fe4fe477404d73689360ee5a4.jpg)

![9a7e8fe48f7837ae4fc46eaaf03fae9d3170e5b7e98999530057fdf25ff69b10.jpg](../iclr_results/134_On%20the%20H%C3%B6lder%20Stability%20of%20Multiset%20and%20Graph%20Neural%20Networks/tables/9a7e8fe48f7837ae4fc46eaaf03fae9d3170e5b7e98999530057fdf25ff69b10.jpg)

![dd99f75a2c3a4d571a49ef88b133e67d842685306378eab60b656c542f96b3b4.jpg](../iclr_results/134_On%20the%20H%C3%B6lder%20Stability%20of%20Multiset%20and%20Graph%20Neural%20Networks/tables/dd99f75a2c3a4d571a49ef88b133e67d842685306378eab60b656c542f96b3b4.jpg)

![f1a3771f43886a839948635282e1342b9109c21e40591e6e90f1c6da2af7ca74.jpg](../iclr_results/134_On%20the%20H%C3%B6lder%20Stability%20of%20Multiset%20and%20Graph%20Neural%20Networks/tables/f1a3771f43886a839948635282e1342b9109c21e40591e6e90f1c6da2af7ca74.jpg)

## No Pose, No Problem: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images


### Images

![05f08d48c7e9372a374b69b922eb2495d8a2a4d3925246a332ffac49a20adcd1.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/images/05f08d48c7e9372a374b69b922eb2495d8a2a4d3925246a332ffac49a20adcd1.jpg)

![06c0c57c169670e39ea5347acd62f3a3cb9bfc89cf7c7060f02f069666b33077.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/images/06c0c57c169670e39ea5347acd62f3a3cb9bfc89cf7c7060f02f069666b33077.jpg)

![10f82af242e8d2477d91bb8b93b2442dd1c3fc3a6cb3e2cad913fbda47c467a9.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/images/10f82af242e8d2477d91bb8b93b2442dd1c3fc3a6cb3e2cad913fbda47c467a9.jpg)

![1befff73c73fa9fd5fc62ee6dae9d3ff5ddd361dfef3da95a22e50a222870f12.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/images/1befff73c73fa9fd5fc62ee6dae9d3ff5ddd361dfef3da95a22e50a222870f12.jpg)

![3618d1fe16ddb762b49d6ea35fa5b014444f426393d763f4458e12e3d9e2c491.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/images/3618d1fe16ddb762b49d6ea35fa5b014444f426393d763f4458e12e3d9e2c491.jpg)

![36f57cab4bbb5cf37a5aa94e0d080dbec3e07d2020a0709ba3a0be71b5466022.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/images/36f57cab4bbb5cf37a5aa94e0d080dbec3e07d2020a0709ba3a0be71b5466022.jpg)

![605595793163689bf6630c9239238ad369128c0bc75bb771a7fa88d7b2adfbaa.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/images/605595793163689bf6630c9239238ad369128c0bc75bb771a7fa88d7b2adfbaa.jpg)

![8eff552bc2e3bb6d4b5c4a33c2cf0e6e2f83f684710c8193d2ea42752d3c6f81.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/images/8eff552bc2e3bb6d4b5c4a33c2cf0e6e2f83f684710c8193d2ea42752d3c6f81.jpg)

![9325e0fa23aca0cc7991e0cd4e1b8d1116eb8882160283a1cb3b876a1593eaa5.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/images/9325e0fa23aca0cc7991e0cd4e1b8d1116eb8882160283a1cb3b876a1593eaa5.jpg)

![a41eb3f63242fd737b55bbd4ccc4255c267ed4061cc065a6ba323058cb3b9763.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/images/a41eb3f63242fd737b55bbd4ccc4255c267ed4061cc065a6ba323058cb3b9763.jpg)

![aff760be49bb25a5f37c59a2ba02af374211f9b4e02655add3318fcc43cb89aa.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/images/aff760be49bb25a5f37c59a2ba02af374211f9b4e02655add3318fcc43cb89aa.jpg)

![b12bdbe380daffdb1e44c23e8e1c782c65978687c9345f5df4ba2316112b48bf.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/images/b12bdbe380daffdb1e44c23e8e1c782c65978687c9345f5df4ba2316112b48bf.jpg)

![bd577049bf496b22f3f1d92e5d10f92ec6911bd0f906b7097b5be81254c70ac0.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/images/bd577049bf496b22f3f1d92e5d10f92ec6911bd0f906b7097b5be81254c70ac0.jpg)

![c0b53115bae309ab3774d2b55987fa88c17dd99661529d8e1b46c8590d053dcb.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/images/c0b53115bae309ab3774d2b55987fa88c17dd99661529d8e1b46c8590d053dcb.jpg)

![debbb6a4d48ffe2507eeb91d95e730fb79fb5b8252815300ee8c00a12b00cd17.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/images/debbb6a4d48ffe2507eeb91d95e730fb79fb5b8252815300ee8c00a12b00cd17.jpg)

![e532e48ffb77995bec480b33aee87c7c6ba05fed8e3c92a7dd94be8a13404697.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/images/e532e48ffb77995bec480b33aee87c7c6ba05fed8e3c92a7dd94be8a13404697.jpg)

![e8897b03e3a555cc87e5c468a8da832cd1ea37ca3b31ea8db3f4e923ba4b70ee.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/images/e8897b03e3a555cc87e5c468a8da832cd1ea37ca3b31ea8db3f4e923ba4b70ee.jpg)

![fe40dfce19d4b5211f4377e52565055451326518043e935cda9e42ccea7f5444.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/images/fe40dfce19d4b5211f4377e52565055451326518043e935cda9e42ccea7f5444.jpg)

### Tables

![1e7cade2749ceb85ab910c8730b51f6e6b392297f3e1035a4b3f2a1315b56fd5.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/tables/1e7cade2749ceb85ab910c8730b51f6e6b392297f3e1035a4b3f2a1315b56fd5.jpg)

![2de4a758583c63b4085806c0e5f38ccbdec1da8a77fb11e8bcd11ed2c2acce97.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/tables/2de4a758583c63b4085806c0e5f38ccbdec1da8a77fb11e8bcd11ed2c2acce97.jpg)

![5542651d5281f4e01a5d9d78e2db2198a92f09a9c3aadaf4eefde1b46aa875b4.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/tables/5542651d5281f4e01a5d9d78e2db2198a92f09a9c3aadaf4eefde1b46aa875b4.jpg)

![8becbf42150acdaa4798368386114eb382b4cb621508e27eb779539189e876bf.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/tables/8becbf42150acdaa4798368386114eb382b4cb621508e27eb779539189e876bf.jpg)

![99e790d1d7ef494bc6384c93837137d0b0e4f48beb364af6493d0e0ce9227275.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/tables/99e790d1d7ef494bc6384c93837137d0b0e4f48beb364af6493d0e0ce9227275.jpg)

![bc5207be8903c36d13910cd8acede5cd2f1d19167df4c52497ebdd3106a8dacb.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/tables/bc5207be8903c36d13910cd8acede5cd2f1d19167df4c52497ebdd3106a8dacb.jpg)

![cba86b7d4133547fd7df28a0dbe219a3bcfbd12253f5a00ab92ad2b7d24a03b9.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/tables/cba86b7d4133547fd7df28a0dbe219a3bcfbd12253f5a00ab92ad2b7d24a03b9.jpg)

![e8152a7ae390c38ec2d6b192a1771e01ebe0f30ee3c5a8ea9ec62a30c8b6dcd9.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/tables/e8152a7ae390c38ec2d6b192a1771e01ebe0f30ee3c5a8ea9ec62a30c8b6dcd9.jpg)

![ecce3761dbe742a67b861a251f352ff702b35d2e30a6647c7a61b7ac7d10140d.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/tables/ecce3761dbe742a67b861a251f352ff702b35d2e30a6647c7a61b7ac7d10140d.jpg)

![fcfd117ef8dec1a7676b666d1010a4c2cb4eacc655bed0066bf19e806929dde4.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/tables/fcfd117ef8dec1a7676b666d1010a4c2cb4eacc655bed0066bf19e806929dde4.jpg)

![ff96a3e8e56efd6eaebb8e7df18bb0668b2476057768bebebcbed01ea3d3443f.jpg](../iclr_results/135_No%20Pose%2C%20No%20Problem_%20Surprisingly%20Simple%203D%20Gaussian%20Splats%20from%20Sparse%20Unposed%20Images/tables/ff96a3e8e56efd6eaebb8e7df18bb0668b2476057768bebebcbed01ea3d3443f.jpg)

## KAN: Kolmogorov–Arnold Networks


### Images

![0bd3bfabcbf6296cb3e96f7eb3eff7ff6a35fa50aa17e57455e53387aa29836c.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/0bd3bfabcbf6296cb3e96f7eb3eff7ff6a35fa50aa17e57455e53387aa29836c.jpg)

![169cd10736cd900322ae91d37423efee9710b0fefc6dc9487b4858519b4af6a7.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/169cd10736cd900322ae91d37423efee9710b0fefc6dc9487b4858519b4af6a7.jpg)

![17965ec6841d308e76cb9ffb7fdb1f5bd25dc7fbdef06d5ae98e617ea33d6312.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/17965ec6841d308e76cb9ffb7fdb1f5bd25dc7fbdef06d5ae98e617ea33d6312.jpg)

![1e981b3c1066082ee3a697374bc7b92ef154cd52f750d949a4344ea757af2d00.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/1e981b3c1066082ee3a697374bc7b92ef154cd52f750d949a4344ea757af2d00.jpg)

![29cf3d88f2f92de222e7f4442ac246076d9592125c069807042259d4a5efe41f.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/29cf3d88f2f92de222e7f4442ac246076d9592125c069807042259d4a5efe41f.jpg)

![2de828e7d1b071cd4d7eb7aee34cc430aa69b08367422bd2d83a217179438a6c.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/2de828e7d1b071cd4d7eb7aee34cc430aa69b08367422bd2d83a217179438a6c.jpg)

![32797138f5d11b34115d7f8e3f0604df8f3a37de6a3969c3a035c6473be12399.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/32797138f5d11b34115d7f8e3f0604df8f3a37de6a3969c3a035c6473be12399.jpg)

![36fde790968271e38d686b9ab23ea30e104cae3cdb999bf177012142a26c0635.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/36fde790968271e38d686b9ab23ea30e104cae3cdb999bf177012142a26c0635.jpg)

![3c16db126112ca3160c658adfd7bd3901cce169651a775ddf4616fa4bdaf1f62.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/3c16db126112ca3160c658adfd7bd3901cce169651a775ddf4616fa4bdaf1f62.jpg)

![5521058b5caa37a7512e8ad9d8a7d615adc139c1f4f84cff63e53e67e1e4ffb0.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/5521058b5caa37a7512e8ad9d8a7d615adc139c1f4f84cff63e53e67e1e4ffb0.jpg)

![55ace8b097433431f28d95f32de059db683e281cded1ccad1f137727174156dd.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/55ace8b097433431f28d95f32de059db683e281cded1ccad1f137727174156dd.jpg)

![608c242736eaa27451973009944f8b11a416512110f8cb7c4702767533e9b606.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/608c242736eaa27451973009944f8b11a416512110f8cb7c4702767533e9b606.jpg)

![646df44915dbdbe249e5f28ae0219cfae8cb5a96e0f5989f39fa38a0a9a3253e.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/646df44915dbdbe249e5f28ae0219cfae8cb5a96e0f5989f39fa38a0a9a3253e.jpg)

![67bbe2e7bd1044c70a2caa9ee460648b8144b437b0383f8445ff758fa0c91a82.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/67bbe2e7bd1044c70a2caa9ee460648b8144b437b0383f8445ff758fa0c91a82.jpg)

![83ea30cc2b271d4c263acac4a7f35b3c3d879d991f4fb7e8d1d8fc47e061e953.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/83ea30cc2b271d4c263acac4a7f35b3c3d879d991f4fb7e8d1d8fc47e061e953.jpg)

![8eebd553d89867cac05716b9e83fea3581af5d5f745ab834543dddcf580b455d.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/8eebd553d89867cac05716b9e83fea3581af5d5f745ab834543dddcf580b455d.jpg)

![9193fbbdefcf905aaa678c55facf78bc08bc0e5484efeb31ef82950b9a75d526.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/9193fbbdefcf905aaa678c55facf78bc08bc0e5484efeb31ef82950b9a75d526.jpg)

![a929e74668e4c60d8b0aca671f2c32dd833122ac62fd718cf7ec9af5c05a1476.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/a929e74668e4c60d8b0aca671f2c32dd833122ac62fd718cf7ec9af5c05a1476.jpg)

![b40392c185188902eb9acfff072887c84793fa534cf2a662be8b59e451d1d473.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/b40392c185188902eb9acfff072887c84793fa534cf2a662be8b59e451d1d473.jpg)

![c55c6beaa0e167c9202a21d7cb9e210a07e00e3636609f63fd5523ac823ffcb5.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/c55c6beaa0e167c9202a21d7cb9e210a07e00e3636609f63fd5523ac823ffcb5.jpg)

![d117fdb2b7a3a4d02c994ee8738bd1a71d696db20299b0a3309de6f8fd29cdbc.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/d117fdb2b7a3a4d02c994ee8738bd1a71d696db20299b0a3309de6f8fd29cdbc.jpg)

![d3819d1bf558eca48110c567b3db3c05ae46daeb084fd68bf2c5238ed5735067.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/d3819d1bf558eca48110c567b3db3c05ae46daeb084fd68bf2c5238ed5735067.jpg)

![deea15877362956d9b7ccb0076379d4e230595a671afb784c0affd43be05924c.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/deea15877362956d9b7ccb0076379d4e230595a671afb784c0affd43be05924c.jpg)

![ecb6e84b1dc526397716ae0329e76a1b00d1abc4b3bd030e80ea9a719fe2ec8a.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/ecb6e84b1dc526397716ae0329e76a1b00d1abc4b3bd030e80ea9a719fe2ec8a.jpg)

![f2d45c730919c55541595a047023e99588a60b77a3e75935e7051149e80c9434.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/f2d45c730919c55541595a047023e99588a60b77a3e75935e7051149e80c9434.jpg)

![f9bb51bfc96c82468f6337cb71e77a18aae81b362c58ea47ff98695cf2bf461f.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/f9bb51bfc96c82468f6337cb71e77a18aae81b362c58ea47ff98695cf2bf461f.jpg)

![ff9d315845d165f844665be344ecc89338c3e906bfe8891c535ceb019327771e.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/images/ff9d315845d165f844665be344ecc89338c3e906bfe8891c535ceb019327771e.jpg)

### Tables

![0e6e103f636485967201e51ec1797408c6083fc14331b07b10fefb7776199c93.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/tables/0e6e103f636485967201e51ec1797408c6083fc14331b07b10fefb7776199c93.jpg)

![34ab43a22bac33dc71afe59ef2189bd4d64bb0d5453d65fde7ebf8054425ab0d.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/tables/34ab43a22bac33dc71afe59ef2189bd4d64bb0d5453d65fde7ebf8054425ab0d.jpg)

![37f58a4171fc8ca605120cdac14d597d1d0fc45cc8d231a3cf0db40ff6aa5581.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/tables/37f58a4171fc8ca605120cdac14d597d1d0fc45cc8d231a3cf0db40ff6aa5581.jpg)

![57279e4d17786b30ae2189ee003d01924e9f7692210c2d8e1b4eb5374d636b26.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/tables/57279e4d17786b30ae2189ee003d01924e9f7692210c2d8e1b4eb5374d636b26.jpg)

![7bdcb5f5822e0cc3cbfa62ebbf1157b20630729044078fcd93bf150c011a74d1.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/tables/7bdcb5f5822e0cc3cbfa62ebbf1157b20630729044078fcd93bf150c011a74d1.jpg)

![7c3e52086a669afdc0958e09b960d41cb8e1ebbef5c5dd80ef5d971e3e0175d0.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/tables/7c3e52086a669afdc0958e09b960d41cb8e1ebbef5c5dd80ef5d971e3e0175d0.jpg)

![803937e43ebeda52b1f5130082046b23fc44969b21d6cb877a75e64df71302a7.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/tables/803937e43ebeda52b1f5130082046b23fc44969b21d6cb877a75e64df71302a7.jpg)

![9987defb888ab83058519cf75da3ea8eb3d5f2c2f5f70e021c7eadb8faab20d2.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/tables/9987defb888ab83058519cf75da3ea8eb3d5f2c2f5f70e021c7eadb8faab20d2.jpg)

![c74a459c948cb8735aa4fe305fa14c05d1458383eecccce482e4cc92dc02abb1.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/tables/c74a459c948cb8735aa4fe305fa14c05d1458383eecccce482e4cc92dc02abb1.jpg)

![ccbc2a7599617ca062cc6b6634ff93c881c3f5384e33b4a8c969f132f6fc2986.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/tables/ccbc2a7599617ca062cc6b6634ff93c881c3f5384e33b4a8c969f132f6fc2986.jpg)

![debc2cc8bf7e7160e51965b36a427fbdf15a29b113b8385d32a7ff824d4efbec.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/tables/debc2cc8bf7e7160e51965b36a427fbdf15a29b113b8385d32a7ff824d4efbec.jpg)

![e1307708820df6247eca9f94c21bf5c8f717252c475ac8a407ee45db9b0e0b51.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/tables/e1307708820df6247eca9f94c21bf5c8f717252c475ac8a407ee45db9b0e0b51.jpg)

![f9a967ad1a58950aba91707ca12f9f4cfbcac19541f3cb39b855a5861571ce80.jpg](../iclr_results/136_KAN_%20Kolmogorov%E2%80%93Arnold%20Networks/tables/f9a967ad1a58950aba91707ca12f9f4cfbcac19541f3cb39b855a5861571ce80.jpg)

## Differential Transformer


### Images

![10a9a8a16e545e7596c0e1d5353ac0b25b3966e026fd03487f77fb77aca8f400.jpg](../iclr_results/137_Differential%20Transformer/images/10a9a8a16e545e7596c0e1d5353ac0b25b3966e026fd03487f77fb77aca8f400.jpg)

![33eb936bb4b2b2374db92268d9b0d4eebb2af14b522e2349a98c1095a79a12b8.jpg](../iclr_results/137_Differential%20Transformer/images/33eb936bb4b2b2374db92268d9b0d4eebb2af14b522e2349a98c1095a79a12b8.jpg)

![37a3da4f80832c5b2e65552556828dbc4685a7683f3972603e8e566c6e02b5da.jpg](../iclr_results/137_Differential%20Transformer/images/37a3da4f80832c5b2e65552556828dbc4685a7683f3972603e8e566c6e02b5da.jpg)

![3a32f2cac254a30b70a94976d33f28fb8efbb7cea34308426efd5ca829193782.jpg](../iclr_results/137_Differential%20Transformer/images/3a32f2cac254a30b70a94976d33f28fb8efbb7cea34308426efd5ca829193782.jpg)

![5fa6a6caceba0607d38f56b7ccecfecc6dbd8c15e317f2a068f0126ffe48570f.jpg](../iclr_results/137_Differential%20Transformer/images/5fa6a6caceba0607d38f56b7ccecfecc6dbd8c15e317f2a068f0126ffe48570f.jpg)

![80b3f093f09d0f011ca420ffba7fcf0beacf0fb62fc5a15915bcf65422a0b0ee.jpg](../iclr_results/137_Differential%20Transformer/images/80b3f093f09d0f011ca420ffba7fcf0beacf0fb62fc5a15915bcf65422a0b0ee.jpg)

![8b8e1defb73282f113b656236ac59f2ccd9b18a0a1c589b2af8fe814f5e8a3fa.jpg](../iclr_results/137_Differential%20Transformer/images/8b8e1defb73282f113b656236ac59f2ccd9b18a0a1c589b2af8fe814f5e8a3fa.jpg)

![8e267c59437cc28fc3580a9e5e003140746811c36ca3b56948ee5dcd862b2e46.jpg](../iclr_results/137_Differential%20Transformer/images/8e267c59437cc28fc3580a9e5e003140746811c36ca3b56948ee5dcd862b2e46.jpg)

![a3b1c2d2836a78f238edfa3b41d8b8063f53488aa10cf965538bc707e0212a99.jpg](../iclr_results/137_Differential%20Transformer/images/a3b1c2d2836a78f238edfa3b41d8b8063f53488aa10cf965538bc707e0212a99.jpg)

![a7e2a45e15d4d12af2ec17ca4cbfa71e08c92416cce6651d8fd962be3e855263.jpg](../iclr_results/137_Differential%20Transformer/images/a7e2a45e15d4d12af2ec17ca4cbfa71e08c92416cce6651d8fd962be3e855263.jpg)

![bff42f1694881234d3aff28036b799bd87c3e3822d62dad4c71c9fc967c9ee9c.jpg](../iclr_results/137_Differential%20Transformer/images/bff42f1694881234d3aff28036b799bd87c3e3822d62dad4c71c9fc967c9ee9c.jpg)

![d7bbb2f368901729f18cd85addb2434f3374d4dd9ff3d87eeb0c90bc2d0a6867.jpg](../iclr_results/137_Differential%20Transformer/images/d7bbb2f368901729f18cd85addb2434f3374d4dd9ff3d87eeb0c90bc2d0a6867.jpg)

![e1f994538c4d90c5aa17297f9b5a157055a457860705b8b50e478bf07a813997.jpg](../iclr_results/137_Differential%20Transformer/images/e1f994538c4d90c5aa17297f9b5a157055a457860705b8b50e478bf07a813997.jpg)

### Tables

![04ae7db097f58bfae75ffd37d87db82e9bfff0d358fce99a268491f1460c702b.jpg](../iclr_results/137_Differential%20Transformer/tables/04ae7db097f58bfae75ffd37d87db82e9bfff0d358fce99a268491f1460c702b.jpg)

![232962e2d806cd3519cc1ee43141b46db2d2e5598ce43a117bb84ac41c62f438.jpg](../iclr_results/137_Differential%20Transformer/tables/232962e2d806cd3519cc1ee43141b46db2d2e5598ce43a117bb84ac41c62f438.jpg)

![2ab0d66d713a16e6cef8e0f4480f99c69705aaa4ac81ba042226461f52f5255b.jpg](../iclr_results/137_Differential%20Transformer/tables/2ab0d66d713a16e6cef8e0f4480f99c69705aaa4ac81ba042226461f52f5255b.jpg)

![43302e1c79ca39a19f892bf99c2e780e9d9f38e0431810973544610f7ce315c5.jpg](../iclr_results/137_Differential%20Transformer/tables/43302e1c79ca39a19f892bf99c2e780e9d9f38e0431810973544610f7ce315c5.jpg)

![4920d700dc76e9a5f5326c398a60e4f10a0be32d0130dba845c8f6b177f048e0.jpg](../iclr_results/137_Differential%20Transformer/tables/4920d700dc76e9a5f5326c398a60e4f10a0be32d0130dba845c8f6b177f048e0.jpg)

![6faa8aafbbbc61eb135e9003b782ab08d34d6ffcdb952b434f110525b293d4a7.jpg](../iclr_results/137_Differential%20Transformer/tables/6faa8aafbbbc61eb135e9003b782ab08d34d6ffcdb952b434f110525b293d4a7.jpg)

![7340c3ef5db9574b2029dc097cb85885c71203a296cddf95607fd159ba403ced.jpg](../iclr_results/137_Differential%20Transformer/tables/7340c3ef5db9574b2029dc097cb85885c71203a296cddf95607fd159ba403ced.jpg)

![87f951bd1d9bb36cd82653bf4ebacbc416c161ee930f1b5e1f1886c6f6af24ed.jpg](../iclr_results/137_Differential%20Transformer/tables/87f951bd1d9bb36cd82653bf4ebacbc416c161ee930f1b5e1f1886c6f6af24ed.jpg)

![9359028405c9b9c0c4acb2d48af1da691c7b25a70f79a3576f29918f8163023c.jpg](../iclr_results/137_Differential%20Transformer/tables/9359028405c9b9c0c4acb2d48af1da691c7b25a70f79a3576f29918f8163023c.jpg)

![aed591c6a6a1ffd72c14e2c6387bd500297dc03a3858f8114fa1f01438e0d8f9.jpg](../iclr_results/137_Differential%20Transformer/tables/aed591c6a6a1ffd72c14e2c6387bd500297dc03a3858f8114fa1f01438e0d8f9.jpg)

![caeaca5a60c8674c1d1388737bc742f40715de01d5c3205308c46021bc5a1685.jpg](../iclr_results/137_Differential%20Transformer/tables/caeaca5a60c8674c1d1388737bc742f40715de01d5c3205308c46021bc5a1685.jpg)

![d70992c8adf2b8ca28b186a63a127679cdd43e6cb4afc46b65e931ed5cee2ade.jpg](../iclr_results/137_Differential%20Transformer/tables/d70992c8adf2b8ca28b186a63a127679cdd43e6cb4afc46b65e931ed5cee2ade.jpg)

![ed57ce87fd808f4ade76046f2190e0b1ad4c436f126fdc0205140d57141d2cb2.jpg](../iclr_results/137_Differential%20Transformer/tables/ed57ce87fd808f4ade76046f2190e0b1ad4c436f126fdc0205140d57141d2cb2.jpg)

![f8c3735c42a990ff8a07716f15a920ae865e27a217db4bc567cbbdcdc791c3e4.jpg](../iclr_results/137_Differential%20Transformer/tables/f8c3735c42a990ff8a07716f15a920ae865e27a217db4bc567cbbdcdc791c3e4.jpg)

![ffdcb14d047f092ef4a37d5d9ee899d9a9899b02733ed83a1f20bc10b79a9791.jpg](../iclr_results/137_Differential%20Transformer/tables/ffdcb14d047f092ef4a37d5d9ee899d9a9899b02733ed83a1f20bc10b79a9791.jpg)

## One Step Diffusion via Shortcut Models


### Images

![30497da77cddceacb7860691417fa604bfe32f8e7343f873c2efa7134449e101.jpg](../iclr_results/138_One%20Step%20Diffusion%20via%20Shortcut%20Models/images/30497da77cddceacb7860691417fa604bfe32f8e7343f873c2efa7134449e101.jpg)

![3a6e204752279eacfc8e17d26a26e2e9a01cea1181b8ab5ad5a907b906e14730.jpg](../iclr_results/138_One%20Step%20Diffusion%20via%20Shortcut%20Models/images/3a6e204752279eacfc8e17d26a26e2e9a01cea1181b8ab5ad5a907b906e14730.jpg)

![44881d85ed3f461c4b49f3e4c9494023468fdce4044ce1fb57e2b1372b8eb754.jpg](../iclr_results/138_One%20Step%20Diffusion%20via%20Shortcut%20Models/images/44881d85ed3f461c4b49f3e4c9494023468fdce4044ce1fb57e2b1372b8eb754.jpg)

![4d377e757045b9cd7ed68507d5244308ee8fcdba6454c2e3c702114f9f2c5b79.jpg](../iclr_results/138_One%20Step%20Diffusion%20via%20Shortcut%20Models/images/4d377e757045b9cd7ed68507d5244308ee8fcdba6454c2e3c702114f9f2c5b79.jpg)

![4ee92d0581cf3c187e06bbdce3d45ecac9ba0264812423e24596e388a62d977d.jpg](../iclr_results/138_One%20Step%20Diffusion%20via%20Shortcut%20Models/images/4ee92d0581cf3c187e06bbdce3d45ecac9ba0264812423e24596e388a62d977d.jpg)

![993033d326aefedab20452c32ede228d3754bd4fd49f3e0657b1cef075073604.jpg](../iclr_results/138_One%20Step%20Diffusion%20via%20Shortcut%20Models/images/993033d326aefedab20452c32ede228d3754bd4fd49f3e0657b1cef075073604.jpg)

![a527d34dee96f12fbb4d45175f2e8c66129968ffc8590e21b2d42a09bd6cb8a6.jpg](../iclr_results/138_One%20Step%20Diffusion%20via%20Shortcut%20Models/images/a527d34dee96f12fbb4d45175f2e8c66129968ffc8590e21b2d42a09bd6cb8a6.jpg)

![b53f672372f0dfa5a42874c465ed66878feeea5974a741522a653b3ac3ac9b1a.jpg](../iclr_results/138_One%20Step%20Diffusion%20via%20Shortcut%20Models/images/b53f672372f0dfa5a42874c465ed66878feeea5974a741522a653b3ac3ac9b1a.jpg)

![ea402b1881ddefef2ab91df5934bc85c81d7a0431b9034cb80292440cd9759f7.jpg](../iclr_results/138_One%20Step%20Diffusion%20via%20Shortcut%20Models/images/ea402b1881ddefef2ab91df5934bc85c81d7a0431b9034cb80292440cd9759f7.jpg)

### Tables

![09991e38888b49c351fa36c67d60833edfb3d906ec13654f41a3cb0613649f46.jpg](../iclr_results/138_One%20Step%20Diffusion%20via%20Shortcut%20Models/tables/09991e38888b49c351fa36c67d60833edfb3d906ec13654f41a3cb0613649f46.jpg)

![7ca83f69465f209b908c62b533bc40d7d29f1321c4b4d721e488aaafc84f5bf0.jpg](../iclr_results/138_One%20Step%20Diffusion%20via%20Shortcut%20Models/tables/7ca83f69465f209b908c62b533bc40d7d29f1321c4b4d721e488aaafc84f5bf0.jpg)

![9f9bc2254826ccc7304148e02437d46454736caf55ef34c8abaa6ca18a0b15bd.jpg](../iclr_results/138_One%20Step%20Diffusion%20via%20Shortcut%20Models/tables/9f9bc2254826ccc7304148e02437d46454736caf55ef34c8abaa6ca18a0b15bd.jpg)

![9fc4ac8079e7739e0e27cf30a2dfedfebb9e1a9a4d8945f8532909faec580ff2.jpg](../iclr_results/138_One%20Step%20Diffusion%20via%20Shortcut%20Models/tables/9fc4ac8079e7739e0e27cf30a2dfedfebb9e1a9a4d8945f8532909faec580ff2.jpg)

![da98e33b87411287d4bb8cdbd657bf77a5c84cde0533c2c920c6579244a71dd3.jpg](../iclr_results/138_One%20Step%20Diffusion%20via%20Shortcut%20Models/tables/da98e33b87411287d4bb8cdbd657bf77a5c84cde0533c2c920c6579244a71dd3.jpg)

## FlexPrefill: A Context-Aware Sparse Attention Mechanism for Efficient Long-Sequence Inference


### Images

![29066c474778d14895f555b0b478118bb6c3c15229e44f8aa82a92738276bcc9.jpg](../iclr_results/139_FlexPrefill_%20A%20Context-Aware%20Sparse%20Attention%20Mechanism%20for%20Efficient%20Long-Sequence%20Inference/images/29066c474778d14895f555b0b478118bb6c3c15229e44f8aa82a92738276bcc9.jpg)

![460ed22c557fcff4323fa7fb6d34f8a90b17b09dadd146fb75e33abe81b39698.jpg](../iclr_results/139_FlexPrefill_%20A%20Context-Aware%20Sparse%20Attention%20Mechanism%20for%20Efficient%20Long-Sequence%20Inference/images/460ed22c557fcff4323fa7fb6d34f8a90b17b09dadd146fb75e33abe81b39698.jpg)

![474605b423deb3625a62a724268f91345850f306e66d215a08e255f361044304.jpg](../iclr_results/139_FlexPrefill_%20A%20Context-Aware%20Sparse%20Attention%20Mechanism%20for%20Efficient%20Long-Sequence%20Inference/images/474605b423deb3625a62a724268f91345850f306e66d215a08e255f361044304.jpg)

![5c374a6ba4fda51ddd4af4300c767d00b8ba7370593052bcd160d14f485aa9a9.jpg](../iclr_results/139_FlexPrefill_%20A%20Context-Aware%20Sparse%20Attention%20Mechanism%20for%20Efficient%20Long-Sequence%20Inference/images/5c374a6ba4fda51ddd4af4300c767d00b8ba7370593052bcd160d14f485aa9a9.jpg)

![80bf55166bad40862ff4b858cc113ef0a6efcd442d9feb2ee058497f46ceed15.jpg](../iclr_results/139_FlexPrefill_%20A%20Context-Aware%20Sparse%20Attention%20Mechanism%20for%20Efficient%20Long-Sequence%20Inference/images/80bf55166bad40862ff4b858cc113ef0a6efcd442d9feb2ee058497f46ceed15.jpg)

![818d835b158bdafefe006a1b36def9b3c6c381afa4a8dd8ea5f05eeac87e8668.jpg](../iclr_results/139_FlexPrefill_%20A%20Context-Aware%20Sparse%20Attention%20Mechanism%20for%20Efficient%20Long-Sequence%20Inference/images/818d835b158bdafefe006a1b36def9b3c6c381afa4a8dd8ea5f05eeac87e8668.jpg)

![c9500155a94ca725907148bbf6a642b036db125e57aee0dcf63972f4cdd6aee1.jpg](../iclr_results/139_FlexPrefill_%20A%20Context-Aware%20Sparse%20Attention%20Mechanism%20for%20Efficient%20Long-Sequence%20Inference/images/c9500155a94ca725907148bbf6a642b036db125e57aee0dcf63972f4cdd6aee1.jpg)

![d4fd92b90163016c90e7cf413fe14ca437d7d5274b616d02ef0dfb71b9d5f5b2.jpg](../iclr_results/139_FlexPrefill_%20A%20Context-Aware%20Sparse%20Attention%20Mechanism%20for%20Efficient%20Long-Sequence%20Inference/images/d4fd92b90163016c90e7cf413fe14ca437d7d5274b616d02ef0dfb71b9d5f5b2.jpg)

![e2523b4a23d2e53baf4fb28e0ba33b9a62db5891ef98ad32e0e735e5b2c426a8.jpg](../iclr_results/139_FlexPrefill_%20A%20Context-Aware%20Sparse%20Attention%20Mechanism%20for%20Efficient%20Long-Sequence%20Inference/images/e2523b4a23d2e53baf4fb28e0ba33b9a62db5891ef98ad32e0e735e5b2c426a8.jpg)

![e3c529bdf43bd229a798c0ce857ce524eb2fc43e3e709143e9bf24251859d944.jpg](../iclr_results/139_FlexPrefill_%20A%20Context-Aware%20Sparse%20Attention%20Mechanism%20for%20Efficient%20Long-Sequence%20Inference/images/e3c529bdf43bd229a798c0ce857ce524eb2fc43e3e709143e9bf24251859d944.jpg)

![e8dfcb2a377b30f25927410bb297e27238b32df06199f669d1e7dad5780e76bd.jpg](../iclr_results/139_FlexPrefill_%20A%20Context-Aware%20Sparse%20Attention%20Mechanism%20for%20Efficient%20Long-Sequence%20Inference/images/e8dfcb2a377b30f25927410bb297e27238b32df06199f669d1e7dad5780e76bd.jpg)

### Tables

![02169b9f33a081984ab75b2f553d15771099de20cc6f58bb699b83ff7c4cdd46.jpg](../iclr_results/139_FlexPrefill_%20A%20Context-Aware%20Sparse%20Attention%20Mechanism%20for%20Efficient%20Long-Sequence%20Inference/tables/02169b9f33a081984ab75b2f553d15771099de20cc6f58bb699b83ff7c4cdd46.jpg)

![0bc6a03a55d2d3c1aa1b4ea32ef8acca26c591caee808214bfd2919472040ba2.jpg](../iclr_results/139_FlexPrefill_%20A%20Context-Aware%20Sparse%20Attention%20Mechanism%20for%20Efficient%20Long-Sequence%20Inference/tables/0bc6a03a55d2d3c1aa1b4ea32ef8acca26c591caee808214bfd2919472040ba2.jpg)

![2438a5fd1eb9b74cb6e5f898799ccffb8fdb84e111a55765a69154cb4d2189dd.jpg](../iclr_results/139_FlexPrefill_%20A%20Context-Aware%20Sparse%20Attention%20Mechanism%20for%20Efficient%20Long-Sequence%20Inference/tables/2438a5fd1eb9b74cb6e5f898799ccffb8fdb84e111a55765a69154cb4d2189dd.jpg)

![346d2654059ca01bd041a21d23ee11fa4a2a4662a354c21943da6e188838d161.jpg](../iclr_results/139_FlexPrefill_%20A%20Context-Aware%20Sparse%20Attention%20Mechanism%20for%20Efficient%20Long-Sequence%20Inference/tables/346d2654059ca01bd041a21d23ee11fa4a2a4662a354c21943da6e188838d161.jpg)

![43774f3b9103e55c346a50a16bcab643220ff4f4c8aaefdce8bff8b7585a96cb.jpg](../iclr_results/139_FlexPrefill_%20A%20Context-Aware%20Sparse%20Attention%20Mechanism%20for%20Efficient%20Long-Sequence%20Inference/tables/43774f3b9103e55c346a50a16bcab643220ff4f4c8aaefdce8bff8b7585a96cb.jpg)

![93a18528b74554f0caa922bbba225142fa76d04001c12cbf530da671b3c94913.jpg](../iclr_results/139_FlexPrefill_%20A%20Context-Aware%20Sparse%20Attention%20Mechanism%20for%20Efficient%20Long-Sequence%20Inference/tables/93a18528b74554f0caa922bbba225142fa76d04001c12cbf530da671b3c94913.jpg)

![9cd8952041e1033b9b4b5a4434dbe2d82d79ae4781231873a6f70370fe3a14eb.jpg](../iclr_results/139_FlexPrefill_%20A%20Context-Aware%20Sparse%20Attention%20Mechanism%20for%20Efficient%20Long-Sequence%20Inference/tables/9cd8952041e1033b9b4b5a4434dbe2d82d79ae4781231873a6f70370fe3a14eb.jpg)

![a0ebdcdf4a512035cd6e5ee58cf399c273417227b7788467f3f301d1dc3e5ef7.jpg](../iclr_results/139_FlexPrefill_%20A%20Context-Aware%20Sparse%20Attention%20Mechanism%20for%20Efficient%20Long-Sequence%20Inference/tables/a0ebdcdf4a512035cd6e5ee58cf399c273417227b7788467f3f301d1dc3e5ef7.jpg)

![aa918ac5382b9c81d95fed9ce375bd6f2fd4ba2491ce9f73738ec7b3de76d9c7.jpg](../iclr_results/139_FlexPrefill_%20A%20Context-Aware%20Sparse%20Attention%20Mechanism%20for%20Efficient%20Long-Sequence%20Inference/tables/aa918ac5382b9c81d95fed9ce375bd6f2fd4ba2491ce9f73738ec7b3de76d9c7.jpg)

![e9fbc87aa8982bcf152a218a33395065a18d04864459233c0a67d66584ad1764.jpg](../iclr_results/139_FlexPrefill_%20A%20Context-Aware%20Sparse%20Attention%20Mechanism%20for%20Efficient%20Long-Sequence%20Inference/tables/e9fbc87aa8982bcf152a218a33395065a18d04864459233c0a67d66584ad1764.jpg)

![eee94d9c4ca09830cb632f20ab2e79fcb3693c894863b2af2c4b483382d58ee4.jpg](../iclr_results/139_FlexPrefill_%20A%20Context-Aware%20Sparse%20Attention%20Mechanism%20for%20Efficient%20Long-Sequence%20Inference/tables/eee94d9c4ca09830cb632f20ab2e79fcb3693c894863b2af2c4b483382d58ee4.jpg)

## A Theoretically-Principled Sparse, Connected, and Rigid Graph Representation of Molecules


### Images

![0076d742e676cc692a10462d852a1d2fbfa3044dfbab4288e5df8999befe42ca.jpg](../iclr_results/140_A%20Theoretically-Principled%20Sparse%2C%20Connected%2C%20and%20Rigid%20Graph%20Representation%20of%20Molecules/images/0076d742e676cc692a10462d852a1d2fbfa3044dfbab4288e5df8999befe42ca.jpg)

![2b289a2e4b7fd8004fd4426e925a36dfc468505cb118b1bb116d068a861f41d3.jpg](../iclr_results/140_A%20Theoretically-Principled%20Sparse%2C%20Connected%2C%20and%20Rigid%20Graph%20Representation%20of%20Molecules/images/2b289a2e4b7fd8004fd4426e925a36dfc468505cb118b1bb116d068a861f41d3.jpg)

![3cba9a670be608e560bad8dbb1adf05341dd7a462c71ee2afc677b3bedb6175f.jpg](../iclr_results/140_A%20Theoretically-Principled%20Sparse%2C%20Connected%2C%20and%20Rigid%20Graph%20Representation%20of%20Molecules/images/3cba9a670be608e560bad8dbb1adf05341dd7a462c71ee2afc677b3bedb6175f.jpg)

![66078e05de98aa15c931813ef48e3bc258c7f2a1f04e3cfd0553701f6de4f2e9.jpg](../iclr_results/140_A%20Theoretically-Principled%20Sparse%2C%20Connected%2C%20and%20Rigid%20Graph%20Representation%20of%20Molecules/images/66078e05de98aa15c931813ef48e3bc258c7f2a1f04e3cfd0553701f6de4f2e9.jpg)

![86beaf2f868f29250eb59e3c9f62909acef2e2049e414559eff5133ada9af752.jpg](../iclr_results/140_A%20Theoretically-Principled%20Sparse%2C%20Connected%2C%20and%20Rigid%20Graph%20Representation%20of%20Molecules/images/86beaf2f868f29250eb59e3c9f62909acef2e2049e414559eff5133ada9af752.jpg)

![9751495f476b20f371d38f209b512d874a5f4344064475024d0e3adc2f9273ce.jpg](../iclr_results/140_A%20Theoretically-Principled%20Sparse%2C%20Connected%2C%20and%20Rigid%20Graph%20Representation%20of%20Molecules/images/9751495f476b20f371d38f209b512d874a5f4344064475024d0e3adc2f9273ce.jpg)

![b574e6eb57c2690e108583eb5546c733aa6eb19bfea26025bb7ab859cf388151.jpg](../iclr_results/140_A%20Theoretically-Principled%20Sparse%2C%20Connected%2C%20and%20Rigid%20Graph%20Representation%20of%20Molecules/images/b574e6eb57c2690e108583eb5546c733aa6eb19bfea26025bb7ab859cf388151.jpg)

![b5ec31f44bbfe341a7e93c01b21d891d580f4bceae7cfc065ebc563f2a84cb6e.jpg](../iclr_results/140_A%20Theoretically-Principled%20Sparse%2C%20Connected%2C%20and%20Rigid%20Graph%20Representation%20of%20Molecules/images/b5ec31f44bbfe341a7e93c01b21d891d580f4bceae7cfc065ebc563f2a84cb6e.jpg)

![b67617af1b9bae7685881951b234a4b1617b9814be303d18721418fa21d7cd40.jpg](../iclr_results/140_A%20Theoretically-Principled%20Sparse%2C%20Connected%2C%20and%20Rigid%20Graph%20Representation%20of%20Molecules/images/b67617af1b9bae7685881951b234a4b1617b9814be303d18721418fa21d7cd40.jpg)

![c7e2838852cf02adcb6abced7e05ac96416dafc1265cff6ed65b997b49a5326d.jpg](../iclr_results/140_A%20Theoretically-Principled%20Sparse%2C%20Connected%2C%20and%20Rigid%20Graph%20Representation%20of%20Molecules/images/c7e2838852cf02adcb6abced7e05ac96416dafc1265cff6ed65b997b49a5326d.jpg)

![d451e0f8722c55c076b5596dfefad9f00fb115781338972f1f28f70dc0add26c.jpg](../iclr_results/140_A%20Theoretically-Principled%20Sparse%2C%20Connected%2C%20and%20Rigid%20Graph%20Representation%20of%20Molecules/images/d451e0f8722c55c076b5596dfefad9f00fb115781338972f1f28f70dc0add26c.jpg)

### Tables

![05f805c2c4ceef224a2314614b2486f1a35c316e5050a8b1451013b549918824.jpg](../iclr_results/140_A%20Theoretically-Principled%20Sparse%2C%20Connected%2C%20and%20Rigid%20Graph%20Representation%20of%20Molecules/tables/05f805c2c4ceef224a2314614b2486f1a35c316e5050a8b1451013b549918824.jpg)

![0800781a27eef390f96af4ce4aa549c0b079bf817e1179a7f7c771d6f5e23daa.jpg](../iclr_results/140_A%20Theoretically-Principled%20Sparse%2C%20Connected%2C%20and%20Rigid%20Graph%20Representation%20of%20Molecules/tables/0800781a27eef390f96af4ce4aa549c0b079bf817e1179a7f7c771d6f5e23daa.jpg)

![1cca84b85adddeabd0c541efb2e744a278e4b7f94a3a188736217c39afb4974a.jpg](../iclr_results/140_A%20Theoretically-Principled%20Sparse%2C%20Connected%2C%20and%20Rigid%20Graph%20Representation%20of%20Molecules/tables/1cca84b85adddeabd0c541efb2e744a278e4b7f94a3a188736217c39afb4974a.jpg)

![2639d4e94e7f91a6a1c2450c58cc9db09a320af4e9c8476821f26d93ced69ad8.jpg](../iclr_results/140_A%20Theoretically-Principled%20Sparse%2C%20Connected%2C%20and%20Rigid%20Graph%20Representation%20of%20Molecules/tables/2639d4e94e7f91a6a1c2450c58cc9db09a320af4e9c8476821f26d93ced69ad8.jpg)

![6f8d386920edcede3428be38b6c138d00f1fd32c40fc392d8a9f0db81c2cb77d.jpg](../iclr_results/140_A%20Theoretically-Principled%20Sparse%2C%20Connected%2C%20and%20Rigid%20Graph%20Representation%20of%20Molecules/tables/6f8d386920edcede3428be38b6c138d00f1fd32c40fc392d8a9f0db81c2cb77d.jpg)

![82577429123c057b2cb8621b0373e12b44730d574f87651fcd2ca7a7c9101c31.jpg](../iclr_results/140_A%20Theoretically-Principled%20Sparse%2C%20Connected%2C%20and%20Rigid%20Graph%20Representation%20of%20Molecules/tables/82577429123c057b2cb8621b0373e12b44730d574f87651fcd2ca7a7c9101c31.jpg)

![84fc44c35c765f528c89e7203232dfb0a415892ab7614d2386bf5cf0dc0981ce.jpg](../iclr_results/140_A%20Theoretically-Principled%20Sparse%2C%20Connected%2C%20and%20Rigid%20Graph%20Representation%20of%20Molecules/tables/84fc44c35c765f528c89e7203232dfb0a415892ab7614d2386bf5cf0dc0981ce.jpg)

![8f72f4f985cc31d46ed654a370b7d7696e7434bb297506897f55fd5f2840b2f8.jpg](../iclr_results/140_A%20Theoretically-Principled%20Sparse%2C%20Connected%2C%20and%20Rigid%20Graph%20Representation%20of%20Molecules/tables/8f72f4f985cc31d46ed654a370b7d7696e7434bb297506897f55fd5f2840b2f8.jpg)

![939256cbd5b1f3718b9a10a132831b51d1b0bc8c2f0995faed70dc552b9b5cba.jpg](../iclr_results/140_A%20Theoretically-Principled%20Sparse%2C%20Connected%2C%20and%20Rigid%20Graph%20Representation%20of%20Molecules/tables/939256cbd5b1f3718b9a10a132831b51d1b0bc8c2f0995faed70dc552b9b5cba.jpg)

![e0a085582f3a45749c7c83475872ae14a9bcf51ba5b81aad3899cd39baad4e4a.jpg](../iclr_results/140_A%20Theoretically-Principled%20Sparse%2C%20Connected%2C%20and%20Rigid%20Graph%20Representation%20of%20Molecules/tables/e0a085582f3a45749c7c83475872ae14a9bcf51ba5b81aad3899cd39baad4e4a.jpg)

![ea8ebe4be59c564574830318827eed541def2e3df6b029a1d4eb8d8597dd4045.jpg](../iclr_results/140_A%20Theoretically-Principled%20Sparse%2C%20Connected%2C%20and%20Rigid%20Graph%20Representation%20of%20Molecules/tables/ea8ebe4be59c564574830318827eed541def2e3df6b029a1d4eb8d8597dd4045.jpg)

## REGENT: A Retrieval-Augmented Generalist Agent That Can Act In-Context in New Environments


### Images

![0532e32b40035d740ae77b0dcad3c8fefc9c859ad49519e063847e9a493a99c4.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/images/0532e32b40035d740ae77b0dcad3c8fefc9c859ad49519e063847e9a493a99c4.jpg)

![0a7efbf885f8359f30ca7945e592dfc9dd2c3d95c5d6309e9396cdded2ee3681.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/images/0a7efbf885f8359f30ca7945e592dfc9dd2c3d95c5d6309e9396cdded2ee3681.jpg)

![0c46803ea091656054b9b7381d8ba6cc11e55daa6c77b5278440ca437c84a716.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/images/0c46803ea091656054b9b7381d8ba6cc11e55daa6c77b5278440ca437c84a716.jpg)

![166dba6fb6a47f4f93112393f832062e239447ff855f9dd7e3ea156116f1a6fc.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/images/166dba6fb6a47f4f93112393f832062e239447ff855f9dd7e3ea156116f1a6fc.jpg)

![3663ed789b737f5ddeb30719360300471375c17e24496e6ee9400d44fac82505.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/images/3663ed789b737f5ddeb30719360300471375c17e24496e6ee9400d44fac82505.jpg)

![393e3afba02aaa36850be9fc9c22d80cbc4ebdfa1ef6bb5438c9a64b80c3fd81.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/images/393e3afba02aaa36850be9fc9c22d80cbc4ebdfa1ef6bb5438c9a64b80c3fd81.jpg)

![4c17213f04595b38e6d16ed75f371330500523d34dff16106bc212c2de1f5fd5.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/images/4c17213f04595b38e6d16ed75f371330500523d34dff16106bc212c2de1f5fd5.jpg)

![4ec9259fc04ca9e94593a846ba2712e1db7acb1abf36eb934eda6549d51c418a.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/images/4ec9259fc04ca9e94593a846ba2712e1db7acb1abf36eb934eda6549d51c418a.jpg)

![6af66e6d0e6724d2025ea7a74cf53b0354dea3d48a67caf3eb0fc08fcf6b0941.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/images/6af66e6d0e6724d2025ea7a74cf53b0354dea3d48a67caf3eb0fc08fcf6b0941.jpg)

![7e071b2e8e5192ea60d0dfbaf581e20e4e4a0e67006b126e92e5d2585c8e31c1.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/images/7e071b2e8e5192ea60d0dfbaf581e20e4e4a0e67006b126e92e5d2585c8e31c1.jpg)

![812d1b6a9df1004c2d1bb9efbae4e09c61131986a7c2271abe966803e9539c13.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/images/812d1b6a9df1004c2d1bb9efbae4e09c61131986a7c2271abe966803e9539c13.jpg)

![a2c2edab07eb6a09afa489407773d2515e4f092fcfb2aaeee76f9572d00322ec.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/images/a2c2edab07eb6a09afa489407773d2515e4f092fcfb2aaeee76f9572d00322ec.jpg)

![b2eeb3dc092532b6eb26bc8ea0a4bca9fdcc98a96f606f3510960aaf0fd58b59.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/images/b2eeb3dc092532b6eb26bc8ea0a4bca9fdcc98a96f606f3510960aaf0fd58b59.jpg)

![b44df93109d9f54a6e19a4e2b974d6b2b590adce9e039c77a47d10fdd12f566f.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/images/b44df93109d9f54a6e19a4e2b974d6b2b590adce9e039c77a47d10fdd12f566f.jpg)

![b69ccca9252f44f48dfc0995671e0cce111993ea4e27a1d9b52da665e4f1f7fd.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/images/b69ccca9252f44f48dfc0995671e0cce111993ea4e27a1d9b52da665e4f1f7fd.jpg)

![c5630344b776cb3c4b1abe972c9b9bf4bed47365162d88a8ed5891943ead2b2c.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/images/c5630344b776cb3c4b1abe972c9b9bf4bed47365162d88a8ed5891943ead2b2c.jpg)

![d729a95fc1e55c7cc91dfbedfe83d79851c5d0f9c1cb86e1d2a64cd45c0bb611.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/images/d729a95fc1e55c7cc91dfbedfe83d79851c5d0f9c1cb86e1d2a64cd45c0bb611.jpg)

![e579d58244b4f98dc1a4ce62455b38873b47d81f6044ece720898cc115eac019.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/images/e579d58244b4f98dc1a4ce62455b38873b47d81f6044ece720898cc115eac019.jpg)

![e68815e09185a16df15155119687d71cc978821b5710171bb70fc7b53b18485d.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/images/e68815e09185a16df15155119687d71cc978821b5710171bb70fc7b53b18485d.jpg)

### Tables

![651282b42365e762ccdca49585080cd5656ee7b699a7e7b2f8eeb748a4284d2f.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/tables/651282b42365e762ccdca49585080cd5656ee7b699a7e7b2f8eeb748a4284d2f.jpg)

![83d6cc92950f88e599b068fc8df56752dea63a72f376a1a6498540b8ee536c3a.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/tables/83d6cc92950f88e599b068fc8df56752dea63a72f376a1a6498540b8ee536c3a.jpg)

![a290ed596e7929727bba5cae501de604ad150ea0522c79b281cc15040e3e2641.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/tables/a290ed596e7929727bba5cae501de604ad150ea0522c79b281cc15040e3e2641.jpg)

![a556c0b7fb67306062cb75db38cfa2af080de925f017de780a95b84a6e20919b.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/tables/a556c0b7fb67306062cb75db38cfa2af080de925f017de780a95b84a6e20919b.jpg)

![d4deaabcc063463bc83edf663a3f88c58c755be0ebf89deefd931a85808ea247.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/tables/d4deaabcc063463bc83edf663a3f88c58c755be0ebf89deefd931a85808ea247.jpg)

![e3263e6b563f710b9bcb6a35d178fb1baf4baff77270e17b640046f1d2dd2543.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/tables/e3263e6b563f710b9bcb6a35d178fb1baf4baff77270e17b640046f1d2dd2543.jpg)

![fb20895b1920de8098bf013e676b680f69bc8de0bf9ecac1a05aa40ed0996267.jpg](../iclr_results/141_REGENT_%20A%20Retrieval-Augmented%20Generalist%20Agent%20That%20Can%20Act%20In-Context%20in%20New%20Environments/tables/fb20895b1920de8098bf013e676b680f69bc8de0bf9ecac1a05aa40ed0996267.jpg)

## Limits to scalable evaluation at the frontier: LLM as judge won’t beat twice the data


### Images

![28b6cd666f783ecbbd2d5aec731c3277b2dd5becfff4d483637649b3dd03f0ec.jpg](../iclr_results/142_Limits%20to%20scalable%20evaluation%20at%20the%20frontier_%20LLM%20as%20judge%20won%E2%80%99t%20beat%20twice%20the%20data/images/28b6cd666f783ecbbd2d5aec731c3277b2dd5becfff4d483637649b3dd03f0ec.jpg)

![2d4ad804000ece48f82079914c0481f6ef6d165c947f06d8b6427ea1b74993ee.jpg](../iclr_results/142_Limits%20to%20scalable%20evaluation%20at%20the%20frontier_%20LLM%20as%20judge%20won%E2%80%99t%20beat%20twice%20the%20data/images/2d4ad804000ece48f82079914c0481f6ef6d165c947f06d8b6427ea1b74993ee.jpg)

![3a0d0239e9fcab6cbbc68bdc5b6c480c87ea6a30a8b895b855da634de05a15fd.jpg](../iclr_results/142_Limits%20to%20scalable%20evaluation%20at%20the%20frontier_%20LLM%20as%20judge%20won%E2%80%99t%20beat%20twice%20the%20data/images/3a0d0239e9fcab6cbbc68bdc5b6c480c87ea6a30a8b895b855da634de05a15fd.jpg)

![623fab7a42d427035b573bf6e7dc45dfb246e251a7e44bf393aa6c45d92e5c46.jpg](../iclr_results/142_Limits%20to%20scalable%20evaluation%20at%20the%20frontier_%20LLM%20as%20judge%20won%E2%80%99t%20beat%20twice%20the%20data/images/623fab7a42d427035b573bf6e7dc45dfb246e251a7e44bf393aa6c45d92e5c46.jpg)

![750dc83d425d8a8f91db58a5d5699e1ef7c3ecfccbbcd13c4afb8a3c10ee1ffb.jpg](../iclr_results/142_Limits%20to%20scalable%20evaluation%20at%20the%20frontier_%20LLM%20as%20judge%20won%E2%80%99t%20beat%20twice%20the%20data/images/750dc83d425d8a8f91db58a5d5699e1ef7c3ecfccbbcd13c4afb8a3c10ee1ffb.jpg)

![7d1234a1d2a4f047c88843d1ee51c50243d24aea3361d012720d65433cfbfd69.jpg](../iclr_results/142_Limits%20to%20scalable%20evaluation%20at%20the%20frontier_%20LLM%20as%20judge%20won%E2%80%99t%20beat%20twice%20the%20data/images/7d1234a1d2a4f047c88843d1ee51c50243d24aea3361d012720d65433cfbfd69.jpg)

![a522569635c8d4a9cd0f46664f5939bee3a7f862f7272920ef417b6095985300.jpg](../iclr_results/142_Limits%20to%20scalable%20evaluation%20at%20the%20frontier_%20LLM%20as%20judge%20won%E2%80%99t%20beat%20twice%20the%20data/images/a522569635c8d4a9cd0f46664f5939bee3a7f862f7272920ef417b6095985300.jpg)

![b68669e2a3083d657fbda47e2452dec12dc2bf79c4deeb4f179dedef00a15ae5.jpg](../iclr_results/142_Limits%20to%20scalable%20evaluation%20at%20the%20frontier_%20LLM%20as%20judge%20won%E2%80%99t%20beat%20twice%20the%20data/images/b68669e2a3083d657fbda47e2452dec12dc2bf79c4deeb4f179dedef00a15ae5.jpg)

## MAP: Multi-Human-Value Alignment Palette


### Images

![20c6aac89a3b51e578d2cad3604cf0a1e5b91a8f5ab875522b143b5fbba09b55.jpg](../iclr_results/143_MAP_%20Multi-Human-Value%20Alignment%20Palette/images/20c6aac89a3b51e578d2cad3604cf0a1e5b91a8f5ab875522b143b5fbba09b55.jpg)

![5b073750c8316ccec070b7e02b094f13061b9334e6513189ca94cd622986c29d.jpg](../iclr_results/143_MAP_%20Multi-Human-Value%20Alignment%20Palette/images/5b073750c8316ccec070b7e02b094f13061b9334e6513189ca94cd622986c29d.jpg)

![6530a9df91d2fd609f6ae366de8d28615492fa974fc285a5fe7e3142e841825a.jpg](../iclr_results/143_MAP_%20Multi-Human-Value%20Alignment%20Palette/images/6530a9df91d2fd609f6ae366de8d28615492fa974fc285a5fe7e3142e841825a.jpg)

![7c5fa66bda5c6d24f0a06c8b559c6e6700036f25c414898806c503e79150d719.jpg](../iclr_results/143_MAP_%20Multi-Human-Value%20Alignment%20Palette/images/7c5fa66bda5c6d24f0a06c8b559c6e6700036f25c414898806c503e79150d719.jpg)

![7fd42c47926e1be291e9d00c66f64868eb1a70979292f6a909e480ddfe1ee6c8.jpg](../iclr_results/143_MAP_%20Multi-Human-Value%20Alignment%20Palette/images/7fd42c47926e1be291e9d00c66f64868eb1a70979292f6a909e480ddfe1ee6c8.jpg)

![8f9d4671c4ce19911eb05210c643be388d8da17797b68eb2bcd04a38ff66fcbd.jpg](../iclr_results/143_MAP_%20Multi-Human-Value%20Alignment%20Palette/images/8f9d4671c4ce19911eb05210c643be388d8da17797b68eb2bcd04a38ff66fcbd.jpg)

![a7c886e816e54bfc8a6a0a0d286caee10f637daf4ca92ddf95e527ea3e2a97e5.jpg](../iclr_results/143_MAP_%20Multi-Human-Value%20Alignment%20Palette/images/a7c886e816e54bfc8a6a0a0d286caee10f637daf4ca92ddf95e527ea3e2a97e5.jpg)

![aaeded999e39d6b6e3dc082d0906f8ab955833abe00c6c776fd29462a9bea892.jpg](../iclr_results/143_MAP_%20Multi-Human-Value%20Alignment%20Palette/images/aaeded999e39d6b6e3dc082d0906f8ab955833abe00c6c776fd29462a9bea892.jpg)

![af2aac877567279f82daafca3c84b2850ebc928786d2dc84a5e08eca4f5b2c39.jpg](../iclr_results/143_MAP_%20Multi-Human-Value%20Alignment%20Palette/images/af2aac877567279f82daafca3c84b2850ebc928786d2dc84a5e08eca4f5b2c39.jpg)

![b052ad51d95a6ebd6ea8b1470034ff3fad8714bf044719b659de53bf7077e417.jpg](../iclr_results/143_MAP_%20Multi-Human-Value%20Alignment%20Palette/images/b052ad51d95a6ebd6ea8b1470034ff3fad8714bf044719b659de53bf7077e417.jpg)

![be32371d44977a5e4988f0d05232ebdfe43e769648e8d715d675a11df8d8c166.jpg](../iclr_results/143_MAP_%20Multi-Human-Value%20Alignment%20Palette/images/be32371d44977a5e4988f0d05232ebdfe43e769648e8d715d675a11df8d8c166.jpg)

![e3e130fcf6860b2817a50620682ed283306a56249f88cfe0bc638b8ede294a7c.jpg](../iclr_results/143_MAP_%20Multi-Human-Value%20Alignment%20Palette/images/e3e130fcf6860b2817a50620682ed283306a56249f88cfe0bc638b8ede294a7c.jpg)

### Tables

![5b35dc7324361847c484be8dc6f2a7b6ab3edec69f88f336be4af39dc487b6c6.jpg](../iclr_results/143_MAP_%20Multi-Human-Value%20Alignment%20Palette/tables/5b35dc7324361847c484be8dc6f2a7b6ab3edec69f88f336be4af39dc487b6c6.jpg)

![66ccb3cca0bcc2c148061e0b94eda568e6afe3dd54b939564fe896193a23b307.jpg](../iclr_results/143_MAP_%20Multi-Human-Value%20Alignment%20Palette/tables/66ccb3cca0bcc2c148061e0b94eda568e6afe3dd54b939564fe896193a23b307.jpg)

![8c6cb5f312ec6ff73965ea571593c559f12198eeb3383f4b1f8dd04781b9d40d.jpg](../iclr_results/143_MAP_%20Multi-Human-Value%20Alignment%20Palette/tables/8c6cb5f312ec6ff73965ea571593c559f12198eeb3383f4b1f8dd04781b9d40d.jpg)

![97b5143ebca8a3404dc6ebfe562622c00460598bb5f449f046cc07be3e355bca.jpg](../iclr_results/143_MAP_%20Multi-Human-Value%20Alignment%20Palette/tables/97b5143ebca8a3404dc6ebfe562622c00460598bb5f449f046cc07be3e355bca.jpg)

![deb09f42de997ec4c1d9955f2a0bdc1c69ecdea3248e84b131e88359e9edba0b.jpg](../iclr_results/143_MAP_%20Multi-Human-Value%20Alignment%20Palette/tables/deb09f42de997ec4c1d9955f2a0bdc1c69ecdea3248e84b131e88359e9edba0b.jpg)

![ea96fc4dee6be6d502450b7e37dac2237b197379091ad191c66da44dca57e114.jpg](../iclr_results/143_MAP_%20Multi-Human-Value%20Alignment%20Palette/tables/ea96fc4dee6be6d502450b7e37dac2237b197379091ad191c66da44dca57e114.jpg)

![fa8534ed554e9f2d6874d40da26a5f52e9669b5eafce8d989e07991428b0fbe3.jpg](../iclr_results/143_MAP_%20Multi-Human-Value%20Alignment%20Palette/tables/fa8534ed554e9f2d6874d40da26a5f52e9669b5eafce8d989e07991428b0fbe3.jpg)

## SANA: Efficient High-Resolution Text-to-Image Synthesis with Linear Diffusion Transformers


### Images

![15142bfff9455b64bd79b205b3a8c91476bb07c55082221c34c59107d573bb6c.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/images/15142bfff9455b64bd79b205b3a8c91476bb07c55082221c34c59107d573bb6c.jpg)

![29d2fc3ba06640fc396aeda2a7e1c1ec44297d291d63ea5becf402017583a187.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/images/29d2fc3ba06640fc396aeda2a7e1c1ec44297d291d63ea5becf402017583a187.jpg)

![35da82904e62e4711ba762b865654044eef9093a88515f9a745ee57804f80fa8.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/images/35da82904e62e4711ba762b865654044eef9093a88515f9a745ee57804f80fa8.jpg)

![52bce304f3de3187327d9f39ad22f12998e1b7425c5a209d9c0134862a8c09f2.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/images/52bce304f3de3187327d9f39ad22f12998e1b7425c5a209d9c0134862a8c09f2.jpg)

![5c80483c6037bdd15d3c73e4e658c6df335d6900c94ea5497d9611228ccf541f.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/images/5c80483c6037bdd15d3c73e4e658c6df335d6900c94ea5497d9611228ccf541f.jpg)

![80645b47a80f55ae1abc8fb83132607501f39535bd5a03f6f76a9c0f8860ea07.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/images/80645b47a80f55ae1abc8fb83132607501f39535bd5a03f6f76a9c0f8860ea07.jpg)

![845d656610cccc3ab33686a23706ddf307dbf6a4207803643aafd26f65945563.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/images/845d656610cccc3ab33686a23706ddf307dbf6a4207803643aafd26f65945563.jpg)

![893bc0bffcd36007eb6949af434f77463a9665acfe147e8864eb838cc781b5cd.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/images/893bc0bffcd36007eb6949af434f77463a9665acfe147e8864eb838cc781b5cd.jpg)

![9e4d276e93fc6c3dc977908343bd7520e6f7eae3c77a68dfb4f798cc79476c3d.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/images/9e4d276e93fc6c3dc977908343bd7520e6f7eae3c77a68dfb4f798cc79476c3d.jpg)

![a1cc6c075b16fb7d35611fdc98d413ee9c4089cee9d7d46618e260dbcab00d8c.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/images/a1cc6c075b16fb7d35611fdc98d413ee9c4089cee9d7d46618e260dbcab00d8c.jpg)

![b61ae77ae9e90e7ca79b566e46eec7122ec38206305ca1e4a2715a90d9d4e199.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/images/b61ae77ae9e90e7ca79b566e46eec7122ec38206305ca1e4a2715a90d9d4e199.jpg)

![bfe82a6b918af681881b1272ba0a90cf6ea186ab77699aa18eaf55182e91a50b.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/images/bfe82a6b918af681881b1272ba0a90cf6ea186ab77699aa18eaf55182e91a50b.jpg)

![cedbf4df5871095cedbc8de189a9fc401f91750ecb042f4d1ba0ce50ecb8bdb1.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/images/cedbf4df5871095cedbc8de189a9fc401f91750ecb042f4d1ba0ce50ecb8bdb1.jpg)

![d2a5ffc543f0a001a254938c9307b703760ea93d15de836165d8f60bde1408f2.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/images/d2a5ffc543f0a001a254938c9307b703760ea93d15de836165d8f60bde1408f2.jpg)

![e2ac65fd3718495f8907af7f55309858efdb83e6a912f21e9fe561455a55f928.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/images/e2ac65fd3718495f8907af7f55309858efdb83e6a912f21e9fe561455a55f928.jpg)

![ee189ee7e586e25588d56dcaaf26327065fd41944eeab6b687daf3dcaf03c8e9.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/images/ee189ee7e586e25588d56dcaaf26327065fd41944eeab6b687daf3dcaf03c8e9.jpg)

![f206f3508ea09e201dcf609a22227af71de5815cc44b04e6a5cf3dfcadbae6b1.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/images/f206f3508ea09e201dcf609a22227af71de5815cc44b04e6a5cf3dfcadbae6b1.jpg)

![fccb928a8d9c9de2df82e6f052c513464fa674bd5f00ed3e9de74dab72104862.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/images/fccb928a8d9c9de2df82e6f052c513464fa674bd5f00ed3e9de74dab72104862.jpg)

### Tables

![03dd513ca1af356528f3ac8674626727705edb364d2f3a53ed1897a39c641565.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/tables/03dd513ca1af356528f3ac8674626727705edb364d2f3a53ed1897a39c641565.jpg)

![1bd02fdd74893c855fc3c0e4f77596bc465500fbcdd15d3a711bec8acafb5cf6.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/tables/1bd02fdd74893c855fc3c0e4f77596bc465500fbcdd15d3a711bec8acafb5cf6.jpg)

![25053d96dc454777bf9590ec14abf070ad78f602029fe03ec5eade122f713c6d.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/tables/25053d96dc454777bf9590ec14abf070ad78f602029fe03ec5eade122f713c6d.jpg)

![342d530d35a2919ab3af17df57c1623a419b3ef87dcb1d89b6afe9644f00e85a.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/tables/342d530d35a2919ab3af17df57c1623a419b3ef87dcb1d89b6afe9644f00e85a.jpg)

![3ccc5561391425160efd8fa9bbb411a726e2b01246b803fcb9355c017148427a.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/tables/3ccc5561391425160efd8fa9bbb411a726e2b01246b803fcb9355c017148427a.jpg)

![3d513f8bceeaf3e967799f80f36fc93056f25640e3a147d098aeba4eb175b46b.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/tables/3d513f8bceeaf3e967799f80f36fc93056f25640e3a147d098aeba4eb175b46b.jpg)

![4dffa448de7c38714c3946e86c209624e2448f1667f9af5554abb36042f4a407.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/tables/4dffa448de7c38714c3946e86c209624e2448f1667f9af5554abb36042f4a407.jpg)

![8e9a8f4b5ee47f5182f5b82b752c37464a97f036e5c4ceb5633489ac03c2a568.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/tables/8e9a8f4b5ee47f5182f5b82b752c37464a97f036e5c4ceb5633489ac03c2a568.jpg)

![9a8932f3aa9a47fea13e81fa8c887aec36b59bdca9aad6b94d7e0b6ee7415b39.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/tables/9a8932f3aa9a47fea13e81fa8c887aec36b59bdca9aad6b94d7e0b6ee7415b39.jpg)

![9dfed77171d006ec61d4d0d7e0e6f101f246c586ae5336451329f1229da87a20.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/tables/9dfed77171d006ec61d4d0d7e0e6f101f246c586ae5336451329f1229da87a20.jpg)

![abda4002ec2493898dd8a73ce4ba950372dd19b0ade711207883a44e4e68493c.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/tables/abda4002ec2493898dd8a73ce4ba950372dd19b0ade711207883a44e4e68493c.jpg)

![bce890caa4ef1fbb5a013f4860c0d8725337a9e8a2a1c4b6349f9291f792d424.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/tables/bce890caa4ef1fbb5a013f4860c0d8725337a9e8a2a1c4b6349f9291f792d424.jpg)

![d841ea79aa88b96ed402bcd72189fc43d2143443e1a1db1f79e577c707556e5d.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/tables/d841ea79aa88b96ed402bcd72189fc43d2143443e1a1db1f79e577c707556e5d.jpg)

![fd84ad4436c50759ed631420eed544f54f1f623cbbac402f4d1cee3f72264979.jpg](../iclr_results/144_SANA_%20Efficient%20High-Resolution%20Text-to-Image%20Synthesis%20with%20Linear%20Diffusion%20Transformers/tables/fd84ad4436c50759ed631420eed544f54f1f623cbbac402f4d1cee3f72264979.jpg)

## Learning to Discover Regulatory Elements for Gene Expression Prediction


### Images

![654f3767eb5d6ef0934c0fd3b5386ec049aa3ffa88ad138a92277fccc7011009.jpg](../iclr_results/145_Learning%20to%20Discover%20Regulatory%20Elements%20for%20Gene%20Expression%20Prediction/images/654f3767eb5d6ef0934c0fd3b5386ec049aa3ffa88ad138a92277fccc7011009.jpg)

![e9f3794786d5e9ae4adf3512422b7838dafeebe374fd3ede960e6cb0452cfb80.jpg](../iclr_results/145_Learning%20to%20Discover%20Regulatory%20Elements%20for%20Gene%20Expression%20Prediction/images/e9f3794786d5e9ae4adf3512422b7838dafeebe374fd3ede960e6cb0452cfb80.jpg)

### Tables

![170780fc9820683d10023c78c5338ce060e0f4f37b1782af295951fc1f4c1b19.jpg](../iclr_results/145_Learning%20to%20Discover%20Regulatory%20Elements%20for%20Gene%20Expression%20Prediction/tables/170780fc9820683d10023c78c5338ce060e0f4f37b1782af295951fc1f4c1b19.jpg)

![53af7d45e6d9753e436cf1d94d728136b3b39c87b7aded76bca94452a51e1146.jpg](../iclr_results/145_Learning%20to%20Discover%20Regulatory%20Elements%20for%20Gene%20Expression%20Prediction/tables/53af7d45e6d9753e436cf1d94d728136b3b39c87b7aded76bca94452a51e1146.jpg)

![86a0a043f4f2053cec4d59d05c8c9d0410c5b4153be06b7d6c30009b8d4a4672.jpg](../iclr_results/145_Learning%20to%20Discover%20Regulatory%20Elements%20for%20Gene%20Expression%20Prediction/tables/86a0a043f4f2053cec4d59d05c8c9d0410c5b4153be06b7d6c30009b8d4a4672.jpg)

![a95db82c522e431a0af318a8f27215183274a4005ac8ba1b90ff81fe82caab80.jpg](../iclr_results/145_Learning%20to%20Discover%20Regulatory%20Elements%20for%20Gene%20Expression%20Prediction/tables/a95db82c522e431a0af318a8f27215183274a4005ac8ba1b90ff81fe82caab80.jpg)

![ab82784312d7340b81deb9fdb7d5e1299dc35ccac2f199b2651c20d29d68c29e.jpg](../iclr_results/145_Learning%20to%20Discover%20Regulatory%20Elements%20for%20Gene%20Expression%20Prediction/tables/ab82784312d7340b81deb9fdb7d5e1299dc35ccac2f199b2651c20d29d68c29e.jpg)

## Simplifying, Stabilizing and Scaling Continuous-time Consistency Models


### Images

![0797baa75d6464676fa191e179a412d079ef77f01bad38ba8862126b740d24a5.jpg](../iclr_results/146_Simplifying%2C%20Stabilizing%20and%20Scaling%20Continuous-time%20Consistency%20Models/images/0797baa75d6464676fa191e179a412d079ef77f01bad38ba8862126b740d24a5.jpg)

![0bc90bb10723846ed5d8916b0014a239c748d6db7865a3ac36420936c6bd81df.jpg](../iclr_results/146_Simplifying%2C%20Stabilizing%20and%20Scaling%20Continuous-time%20Consistency%20Models/images/0bc90bb10723846ed5d8916b0014a239c748d6db7865a3ac36420936c6bd81df.jpg)

![4588a0d0ab809e753b1f98dcaff01e050e8469c9ce9466bca20da722010b2e79.jpg](../iclr_results/146_Simplifying%2C%20Stabilizing%20and%20Scaling%20Continuous-time%20Consistency%20Models/images/4588a0d0ab809e753b1f98dcaff01e050e8469c9ce9466bca20da722010b2e79.jpg)

![7898f36cea3b3c8cfa9623abaca01d7876fb15d60a788b97b4c69c8bdced0692.jpg](../iclr_results/146_Simplifying%2C%20Stabilizing%20and%20Scaling%20Continuous-time%20Consistency%20Models/images/7898f36cea3b3c8cfa9623abaca01d7876fb15d60a788b97b4c69c8bdced0692.jpg)

![b8d09002e94183b16c99572f37a255e20f5fb0319deea017d750a203a9576223.jpg](../iclr_results/146_Simplifying%2C%20Stabilizing%20and%20Scaling%20Continuous-time%20Consistency%20Models/images/b8d09002e94183b16c99572f37a255e20f5fb0319deea017d750a203a9576223.jpg)

![c10def0cfc460597b293fbd8b29cc94c28aa48ff23220549778b251fb19eab19.jpg](../iclr_results/146_Simplifying%2C%20Stabilizing%20and%20Scaling%20Continuous-time%20Consistency%20Models/images/c10def0cfc460597b293fbd8b29cc94c28aa48ff23220549778b251fb19eab19.jpg)

![cb07bfb536b098f2bc8ae545d5c164a46a56b17cccbceeb36321703e6a1e8b91.jpg](../iclr_results/146_Simplifying%2C%20Stabilizing%20and%20Scaling%20Continuous-time%20Consistency%20Models/images/cb07bfb536b098f2bc8ae545d5c164a46a56b17cccbceeb36321703e6a1e8b91.jpg)

![cda420ea7bf149d49a8d6c3a4329b525b728aa6838645e54f28995b315e41292.jpg](../iclr_results/146_Simplifying%2C%20Stabilizing%20and%20Scaling%20Continuous-time%20Consistency%20Models/images/cda420ea7bf149d49a8d6c3a4329b525b728aa6838645e54f28995b315e41292.jpg)

![d0f487af37186b64e28d48e2db4ba61557f3fd6a8919f054d451f2dc6cf928ae.jpg](../iclr_results/146_Simplifying%2C%20Stabilizing%20and%20Scaling%20Continuous-time%20Consistency%20Models/images/d0f487af37186b64e28d48e2db4ba61557f3fd6a8919f054d451f2dc6cf928ae.jpg)

![e7816a44bc995a61bae7b0f624c6d2fe796c48a561bd5d09cff17406050d1ab9.jpg](../iclr_results/146_Simplifying%2C%20Stabilizing%20and%20Scaling%20Continuous-time%20Consistency%20Models/images/e7816a44bc995a61bae7b0f624c6d2fe796c48a561bd5d09cff17406050d1ab9.jpg)

![f1124e2b976b7f44a922bf818356ded3eb6bad51f207b826fcd6168a80c8fcc1.jpg](../iclr_results/146_Simplifying%2C%20Stabilizing%20and%20Scaling%20Continuous-time%20Consistency%20Models/images/f1124e2b976b7f44a922bf818356ded3eb6bad51f207b826fcd6168a80c8fcc1.jpg)

![f36fb6708369badda1a522476e970200d95da4bc200692290dabd527d0cba85b.jpg](../iclr_results/146_Simplifying%2C%20Stabilizing%20and%20Scaling%20Continuous-time%20Consistency%20Models/images/f36fb6708369badda1a522476e970200d95da4bc200692290dabd527d0cba85b.jpg)

![f6e9c2f24192494addf4a364a249cd269f8bd3f1d0a4217c47e854f575c45a75.jpg](../iclr_results/146_Simplifying%2C%20Stabilizing%20and%20Scaling%20Continuous-time%20Consistency%20Models/images/f6e9c2f24192494addf4a364a249cd269f8bd3f1d0a4217c47e854f575c45a75.jpg)

### Tables

![2903c500f209974b7bc410fda0da97cbca9e756ec1586fabfce8858f9d76e81e.jpg](../iclr_results/146_Simplifying%2C%20Stabilizing%20and%20Scaling%20Continuous-time%20Consistency%20Models/tables/2903c500f209974b7bc410fda0da97cbca9e756ec1586fabfce8858f9d76e81e.jpg)

![3f110d2dd324fd987e66a0126a1046f982ab0fef92eebdb4c45df4b3f550f5ce.jpg](../iclr_results/146_Simplifying%2C%20Stabilizing%20and%20Scaling%20Continuous-time%20Consistency%20Models/tables/3f110d2dd324fd987e66a0126a1046f982ab0fef92eebdb4c45df4b3f550f5ce.jpg)

![5a4de2dc2daed9db0e0ddbf0e761e3339f386759012992f2b9a4ef694ebe5a84.jpg](../iclr_results/146_Simplifying%2C%20Stabilizing%20and%20Scaling%20Continuous-time%20Consistency%20Models/tables/5a4de2dc2daed9db0e0ddbf0e761e3339f386759012992f2b9a4ef694ebe5a84.jpg)

![6f30853c58efbee17b98444e3ab14f236fbb20c914c0c98f07f06abfa1f86257.jpg](../iclr_results/146_Simplifying%2C%20Stabilizing%20and%20Scaling%20Continuous-time%20Consistency%20Models/tables/6f30853c58efbee17b98444e3ab14f236fbb20c914c0c98f07f06abfa1f86257.jpg)

![70979bec3ae07bca12cb6561c8354b2774d9df2820c7b8f897f4c81ace57eb12.jpg](../iclr_results/146_Simplifying%2C%20Stabilizing%20and%20Scaling%20Continuous-time%20Consistency%20Models/tables/70979bec3ae07bca12cb6561c8354b2774d9df2820c7b8f897f4c81ace57eb12.jpg)

![b8e686c8e64b8976fd47b5746e5e55d2b5ab1c563f35e5e712fae39e5b4190ec.jpg](../iclr_results/146_Simplifying%2C%20Stabilizing%20and%20Scaling%20Continuous-time%20Consistency%20Models/tables/b8e686c8e64b8976fd47b5746e5e55d2b5ab1c563f35e5e712fae39e5b4190ec.jpg)

![bafb9c562863cad3e10084b355eb734fbb149a5695f85e1500c06b10dc24b5d8.jpg](../iclr_results/146_Simplifying%2C%20Stabilizing%20and%20Scaling%20Continuous-time%20Consistency%20Models/tables/bafb9c562863cad3e10084b355eb734fbb149a5695f85e1500c06b10dc24b5d8.jpg)

![be295d9ce050fc2b9fad372317ea56f9729b607ada51c8bbdb755337fd93a8a8.jpg](../iclr_results/146_Simplifying%2C%20Stabilizing%20and%20Scaling%20Continuous-time%20Consistency%20Models/tables/be295d9ce050fc2b9fad372317ea56f9729b607ada51c8bbdb755337fd93a8a8.jpg)

![de5ccb78955f510f4bfd4a96086fd3668949746911632a27b7bd7dbdf6506f12.jpg](../iclr_results/146_Simplifying%2C%20Stabilizing%20and%20Scaling%20Continuous-time%20Consistency%20Models/tables/de5ccb78955f510f4bfd4a96086fd3668949746911632a27b7bd7dbdf6506f12.jpg)

## TANGO: Co-Speech Gesture Video Reenactment with Hierarchical Audio Motion Embedding and Diffusion Interpolation


### Images

![087e9cc7ce0f69042b2aa7acbf1fa0a623a825e70005474ac597093e0a3ffc56.jpg](../iclr_results/147_TANGO_%20Co-Speech%20Gesture%20Video%20Reenactment%20with%20Hierarchical%20Audio%20Motion%20Embedding%20and%20Diffusion%20In/images/087e9cc7ce0f69042b2aa7acbf1fa0a623a825e70005474ac597093e0a3ffc56.jpg)

![3570685ff376efffe1091147213ee33e25af963cd38ac5790922ebffb47d240d.jpg](../iclr_results/147_TANGO_%20Co-Speech%20Gesture%20Video%20Reenactment%20with%20Hierarchical%20Audio%20Motion%20Embedding%20and%20Diffusion%20In/images/3570685ff376efffe1091147213ee33e25af963cd38ac5790922ebffb47d240d.jpg)

![b5f596cb1bcf9e2bdbb911b106ec67e75048bcbdb5a3d734756cd890b3b1cdf7.jpg](../iclr_results/147_TANGO_%20Co-Speech%20Gesture%20Video%20Reenactment%20with%20Hierarchical%20Audio%20Motion%20Embedding%20and%20Diffusion%20In/images/b5f596cb1bcf9e2bdbb911b106ec67e75048bcbdb5a3d734756cd890b3b1cdf7.jpg)

![bee611e897431b4daa223064df0b89d13f2a7e193d40f6f2c75f2f2255164cc6.jpg](../iclr_results/147_TANGO_%20Co-Speech%20Gesture%20Video%20Reenactment%20with%20Hierarchical%20Audio%20Motion%20Embedding%20and%20Diffusion%20In/images/bee611e897431b4daa223064df0b89d13f2a7e193d40f6f2c75f2f2255164cc6.jpg)

![f0d7a50b0ac1aefadb9824a544d28db896da523382e9b560d026c1232626ccda.jpg](../iclr_results/147_TANGO_%20Co-Speech%20Gesture%20Video%20Reenactment%20with%20Hierarchical%20Audio%20Motion%20Embedding%20and%20Diffusion%20In/images/f0d7a50b0ac1aefadb9824a544d28db896da523382e9b560d026c1232626ccda.jpg)

![f990ea17176f0f72ee24e5fed1f0730f86658fab4be08a1fb377e18ef685093d.jpg](../iclr_results/147_TANGO_%20Co-Speech%20Gesture%20Video%20Reenactment%20with%20Hierarchical%20Audio%20Motion%20Embedding%20and%20Diffusion%20In/images/f990ea17176f0f72ee24e5fed1f0730f86658fab4be08a1fb377e18ef685093d.jpg)

![fdcee6fbec67684aa3b2d71d19f88a94f28e7c6681772cb2830c53b7b3eb27bb.jpg](../iclr_results/147_TANGO_%20Co-Speech%20Gesture%20Video%20Reenactment%20with%20Hierarchical%20Audio%20Motion%20Embedding%20and%20Diffusion%20In/images/fdcee6fbec67684aa3b2d71d19f88a94f28e7c6681772cb2830c53b7b3eb27bb.jpg)

### Tables

![2205e9e3151553842e0a14d08583a87917066f604a2e2f9ae1b7b021c9b46a2d.jpg](../iclr_results/147_TANGO_%20Co-Speech%20Gesture%20Video%20Reenactment%20with%20Hierarchical%20Audio%20Motion%20Embedding%20and%20Diffusion%20In/tables/2205e9e3151553842e0a14d08583a87917066f604a2e2f9ae1b7b021c9b46a2d.jpg)

![5cc98cf0fd2ac327165ca3b9498d08e9bd7ad5e9b40fd9f27d68c15c5d433119.jpg](../iclr_results/147_TANGO_%20Co-Speech%20Gesture%20Video%20Reenactment%20with%20Hierarchical%20Audio%20Motion%20Embedding%20and%20Diffusion%20In/tables/5cc98cf0fd2ac327165ca3b9498d08e9bd7ad5e9b40fd9f27d68c15c5d433119.jpg)

![a7f3a949ac0ead699627699e53ba92e6825c4b924288e66c1a5b07f89ea31b6e.jpg](../iclr_results/147_TANGO_%20Co-Speech%20Gesture%20Video%20Reenactment%20with%20Hierarchical%20Audio%20Motion%20Embedding%20and%20Diffusion%20In/tables/a7f3a949ac0ead699627699e53ba92e6825c4b924288e66c1a5b07f89ea31b6e.jpg)

![d57ba9f63e6ed810bf7a9bd6018870958bec81c6899c20b1f0713fa9e8ff09da.jpg](../iclr_results/147_TANGO_%20Co-Speech%20Gesture%20Video%20Reenactment%20with%20Hierarchical%20Audio%20Motion%20Embedding%20and%20Diffusion%20In/tables/d57ba9f63e6ed810bf7a9bd6018870958bec81c6899c20b1f0713fa9e8ff09da.jpg)

![e86a304816f5f1c4be59f24b2f79c25309f58adca0631d5c7c380aae138707b6.jpg](../iclr_results/147_TANGO_%20Co-Speech%20Gesture%20Video%20Reenactment%20with%20Hierarchical%20Audio%20Motion%20Embedding%20and%20Diffusion%20In/tables/e86a304816f5f1c4be59f24b2f79c25309f58adca0631d5c7c380aae138707b6.jpg)

## ShEPhERD: Diffusing shape, electrostatics, and pharmacophores for bioisosteric drug design


### Images

![02e50eca9f658d8933615ab7199710ddfa81f635e4e15258fe9d4ec5e29758cf.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/02e50eca9f658d8933615ab7199710ddfa81f635e4e15258fe9d4ec5e29758cf.jpg)

![09c014835a36f66bb1bd46728c74af566999fe79cf4f65fbecf2f7dff0fa5325.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/09c014835a36f66bb1bd46728c74af566999fe79cf4f65fbecf2f7dff0fa5325.jpg)

![0e9ab84a31a660bd6c885cb22823b2d2c7d280f4ed950f9e4beda7a3dd479f92.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/0e9ab84a31a660bd6c885cb22823b2d2c7d280f4ed950f9e4beda7a3dd479f92.jpg)

![10aaa102eefdc2b1c009e28f5b13a796736bf02e2a64136f14895384d60324b3.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/10aaa102eefdc2b1c009e28f5b13a796736bf02e2a64136f14895384d60324b3.jpg)

![1a2977a1757fcffad9e73cdf3db6d00cd71576b3b74b3d2adcb4f75a706ac450.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/1a2977a1757fcffad9e73cdf3db6d00cd71576b3b74b3d2adcb4f75a706ac450.jpg)

![2031b26ee0affcde81812e7cbebe42e673fd5227f76905e97dd432e12b8397bd.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/2031b26ee0affcde81812e7cbebe42e673fd5227f76905e97dd432e12b8397bd.jpg)

![25c3247ff85f991719ecf682dc03a36666f93a3a4425347da981ec607902eeea.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/25c3247ff85f991719ecf682dc03a36666f93a3a4425347da981ec607902eeea.jpg)

![29e94048ee15197349a6c088d3cfae87248ee6ca03b97b7f15bb7ad77650d62c.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/29e94048ee15197349a6c088d3cfae87248ee6ca03b97b7f15bb7ad77650d62c.jpg)

![311779ba71c28e2d007fc46b2e27260c2ff8fd3ca76dd54f85039005a63a0717.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/311779ba71c28e2d007fc46b2e27260c2ff8fd3ca76dd54f85039005a63a0717.jpg)

![321c4882ccb550686eaa5f5527ba4cbcdb134ab56f8e05249b7897e2f63ea898.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/321c4882ccb550686eaa5f5527ba4cbcdb134ab56f8e05249b7897e2f63ea898.jpg)

![364ae8d7568cac0f65588603c35b7d58a9a6b7f284a2734d697e3ce4111e74a2.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/364ae8d7568cac0f65588603c35b7d58a9a6b7f284a2734d697e3ce4111e74a2.jpg)

![436537f1a546a8c157af82dbcd82270a10696f40575434ed52804df96e449cf9.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/436537f1a546a8c157af82dbcd82270a10696f40575434ed52804df96e449cf9.jpg)

![4676cdccc4e2e102d0689a669659ee97e3e901777a1a69f909022722faa9affd.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/4676cdccc4e2e102d0689a669659ee97e3e901777a1a69f909022722faa9affd.jpg)

![4c1e3bbeea66716dc3cd6b9d90173e185bf530f6eada3f627ef7175cac3def4f.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/4c1e3bbeea66716dc3cd6b9d90173e185bf530f6eada3f627ef7175cac3def4f.jpg)

![53e0d6bda33253e617658319ba602a12bd4b7a4a3b1e574174be8c5a44da928b.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/53e0d6bda33253e617658319ba602a12bd4b7a4a3b1e574174be8c5a44da928b.jpg)

![65e5e3356490c0534ddac2dd034768917aee39c7097995551fd45d2cbef8d40f.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/65e5e3356490c0534ddac2dd034768917aee39c7097995551fd45d2cbef8d40f.jpg)

![6702305182651a4a3056a727302f6a52d306d94c13d747bc0a797e7a1db5d4df.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/6702305182651a4a3056a727302f6a52d306d94c13d747bc0a797e7a1db5d4df.jpg)

![72e0b197e2e1c500ca433a6ec1d441524d88b42e5456f67f4a1a8f240d54cfc4.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/72e0b197e2e1c500ca433a6ec1d441524d88b42e5456f67f4a1a8f240d54cfc4.jpg)

![74ea2c3615add0f26a36efd5c5c425d9615e1a45662287057e4a9c68fb1f2ac7.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/74ea2c3615add0f26a36efd5c5c425d9615e1a45662287057e4a9c68fb1f2ac7.jpg)

![90822146c354ef351f79e7724406ae46274bd531c72380b7f48af4fca7c64642.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/90822146c354ef351f79e7724406ae46274bd531c72380b7f48af4fca7c64642.jpg)

![93f9b36797fd25e3e91d725ed68c36379404a55ba863a9b0541f5347a2a2af12.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/93f9b36797fd25e3e91d725ed68c36379404a55ba863a9b0541f5347a2a2af12.jpg)

![9915f0df19eb3b3b3a842f55c8e0855e1bdf67805805bc702c060dca61493263.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/9915f0df19eb3b3b3a842f55c8e0855e1bdf67805805bc702c060dca61493263.jpg)

![9bbb5ad19dbfb160286643ac95f789b0d9beec275032737776d22539dec1cb34.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/9bbb5ad19dbfb160286643ac95f789b0d9beec275032737776d22539dec1cb34.jpg)

![9d3d429b045ff7bc0c868c777a91ca27a678d24133d9e9e8a5857a29965ef6f1.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/9d3d429b045ff7bc0c868c777a91ca27a678d24133d9e9e8a5857a29965ef6f1.jpg)

![a4c8a001976671512fb8f9f78e0bae3ca6e3d348b946556872f5386067eb3d09.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/a4c8a001976671512fb8f9f78e0bae3ca6e3d348b946556872f5386067eb3d09.jpg)

![b73f985d886a56199da9b0cae355575bdf82397e85a5dc300a7f7e25fc6d7145.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/b73f985d886a56199da9b0cae355575bdf82397e85a5dc300a7f7e25fc6d7145.jpg)

![b7acc1d3f6fdbdbc91b036c1100a413e021232010236166e28fc67cafe792d22.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/b7acc1d3f6fdbdbc91b036c1100a413e021232010236166e28fc67cafe792d22.jpg)

![b8ef79a3733503570f98abe3b9bee7b94b3cb88480aa8116a0c4aedacbf4e05a.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/b8ef79a3733503570f98abe3b9bee7b94b3cb88480aa8116a0c4aedacbf4e05a.jpg)

![cff7d09bcea72d4e5f64b05dc84828975293debfae9ebe2043e8bc475420e6b4.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/cff7d09bcea72d4e5f64b05dc84828975293debfae9ebe2043e8bc475420e6b4.jpg)

![d6df19bbcbdc192394dd56fcc31d20ec4a61aeb981282960811051538f63a57c.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/d6df19bbcbdc192394dd56fcc31d20ec4a61aeb981282960811051538f63a57c.jpg)

![dcfe0691d3e3c6c3518a89e78b87e8fd556f9992b13a90ce62cf4b5564f82c62.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/dcfe0691d3e3c6c3518a89e78b87e8fd556f9992b13a90ce62cf4b5564f82c62.jpg)

![e0493b54a5793a83b5f224cd5a221bf41bd6980b4663c23e1ab7610a31de5408.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/e0493b54a5793a83b5f224cd5a221bf41bd6980b4663c23e1ab7610a31de5408.jpg)

![e515261bfedb3cb0879a32017ddda741e71f670b3371136edceff1dce0a7fe1b.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/e515261bfedb3cb0879a32017ddda741e71f670b3371136edceff1dce0a7fe1b.jpg)

![ee5482ecb05111b7fddd5e0e7c1ea3e5201f766743214476dc78569486fb737d.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/images/ee5482ecb05111b7fddd5e0e7c1ea3e5201f766743214476dc78569486fb737d.jpg)

### Tables

![021acab61786827e6b2d2a1d714238bc234e9ccc1138b862e521d53269e1ea87.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/tables/021acab61786827e6b2d2a1d714238bc234e9ccc1138b862e521d53269e1ea87.jpg)

![29041c3213ff61702abaa9cf85338e03424be9b857bb5291e95693590efa303d.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/tables/29041c3213ff61702abaa9cf85338e03424be9b857bb5291e95693590efa303d.jpg)

![344593c750eb05314e215490a78ebe125fa93fc5e7ac797571a1f49d038958b5.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/tables/344593c750eb05314e215490a78ebe125fa93fc5e7ac797571a1f49d038958b5.jpg)

![414a4b4760866d55df5503bfc4f67a44cb653f2690f27787cdbdf0f052f70ffe.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/tables/414a4b4760866d55df5503bfc4f67a44cb653f2690f27787cdbdf0f052f70ffe.jpg)

![4370c9ff8f90cd968d95a30bf1ccbad8b8ba2d850d080fa34edfce390ffed3f9.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/tables/4370c9ff8f90cd968d95a30bf1ccbad8b8ba2d850d080fa34edfce390ffed3f9.jpg)

![46dc3ea4e7764b464681935e6eb37eb0db31930660ebbb9cd240ffcfeacdbf41.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/tables/46dc3ea4e7764b464681935e6eb37eb0db31930660ebbb9cd240ffcfeacdbf41.jpg)

![4fbb19863e6b874b5c60ec51dc36724193193bf2a70e0a1065918c40539ee008.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/tables/4fbb19863e6b874b5c60ec51dc36724193193bf2a70e0a1065918c40539ee008.jpg)

![530593781daf6085d6a977e809ac41b01536428cd13ba8b5423e95c11d48bf31.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/tables/530593781daf6085d6a977e809ac41b01536428cd13ba8b5423e95c11d48bf31.jpg)

![6435939149ca20ba7adbddc63011801d5026df4a77917cb10f454203023373c8.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/tables/6435939149ca20ba7adbddc63011801d5026df4a77917cb10f454203023373c8.jpg)

![8cbf61c9d8ef2a11b7b54fbe6a1736bfcc080ad02cd875622e6c5dff36657f15.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/tables/8cbf61c9d8ef2a11b7b54fbe6a1736bfcc080ad02cd875622e6c5dff36657f15.jpg)

![9a9846231865bb1c1a4157af66157632c8e143eb6dc96d6f2cc811be8e5b32cb.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/tables/9a9846231865bb1c1a4157af66157632c8e143eb6dc96d6f2cc811be8e5b32cb.jpg)

![a982b45d59acc649b1a50ade946617825cf79715066879ffb4a2723c39372067.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/tables/a982b45d59acc649b1a50ade946617825cf79715066879ffb4a2723c39372067.jpg)

![bdc837e4e856845321a04a6657feec4017f155967feb3c6e2258f5d87a0b88e5.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/tables/bdc837e4e856845321a04a6657feec4017f155967feb3c6e2258f5d87a0b88e5.jpg)

![c259302c3dc6ade19e2694f80e58db050778af70786c2c3ea59d6a94d351db23.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/tables/c259302c3dc6ade19e2694f80e58db050778af70786c2c3ea59d6a94d351db23.jpg)

![db68e4f74bafeca97782083bd262b2637722e10579af94f1560cbab2ba22f632.jpg](../iclr_results/148_ShEPhERD_%20Diffusing%20shape%2C%20electrostatics%2C%20and%20pharmacophores%20for%20bioisosteric%20drug%20design/tables/db68e4f74bafeca97782083bd262b2637722e10579af94f1560cbab2ba22f632.jpg)

## miniCTX: Neural Theorem Proving with (Long-)Contexts


### Images

![3f39a2b50faa3db84e5810ed975b3f9d6a8fb0213172046288303ba23f3d9aca.jpg](../iclr_results/149_miniCTX_%20Neural%20Theorem%20Proving%20with%20%28Long-)Contexts/images/3f39a2b50faa3db84e5810ed975b3f9d6a8fb0213172046288303ba23f3d9aca.jpg)

![435098d1b2e83c904501502fbab9ca3fa5f7bd71efcde6bd61c7e0d3539d8216.jpg](../iclr_results/149_miniCTX_%20Neural%20Theorem%20Proving%20with%20%28Long-)Contexts/images/435098d1b2e83c904501502fbab9ca3fa5f7bd71efcde6bd61c7e0d3539d8216.jpg)

![44b802fda01705a82fdc8b94ac02cf9ec2a4b07468b2e1ac7b2dddc7628f1b34.jpg](../iclr_results/149_miniCTX_%20Neural%20Theorem%20Proving%20with%20%28Long-)Contexts/images/44b802fda01705a82fdc8b94ac02cf9ec2a4b07468b2e1ac7b2dddc7628f1b34.jpg)

![5606f789b38236fd216c1233309c50cee7f1e8c4193faf50770d86d2b60ef722.jpg](../iclr_results/149_miniCTX_%20Neural%20Theorem%20Proving%20with%20%28Long-)Contexts/images/5606f789b38236fd216c1233309c50cee7f1e8c4193faf50770d86d2b60ef722.jpg)

![5b0f5adf6ba50331fe21a7f9083d3dfbd4d7bc0a3a408c69f6c68db909ea497c.jpg](../iclr_results/149_miniCTX_%20Neural%20Theorem%20Proving%20with%20%28Long-)Contexts/images/5b0f5adf6ba50331fe21a7f9083d3dfbd4d7bc0a3a408c69f6c68db909ea497c.jpg)

![ad3786382a34106e0e83168ec2d6fc10ef86c57c4ee90a71f2c3dc84590aa82b.jpg](../iclr_results/149_miniCTX_%20Neural%20Theorem%20Proving%20with%20%28Long-)Contexts/images/ad3786382a34106e0e83168ec2d6fc10ef86c57c4ee90a71f2c3dc84590aa82b.jpg)

![adfbb080e1426d14bfe243ed2090c3395bcdb9ec828c5b92fb63187562ac1126.jpg](../iclr_results/149_miniCTX_%20Neural%20Theorem%20Proving%20with%20%28Long-)Contexts/images/adfbb080e1426d14bfe243ed2090c3395bcdb9ec828c5b92fb63187562ac1126.jpg)

![f09ba81090084732fbd3b1a20ecbe41db96ad5383258e33bfbc1a3e63be85bf1.jpg](../iclr_results/149_miniCTX_%20Neural%20Theorem%20Proving%20with%20%28Long-)Contexts/images/f09ba81090084732fbd3b1a20ecbe41db96ad5383258e33bfbc1a3e63be85bf1.jpg)

![f620f24ce4903b1c21870da3a8c9f469cfd6243831a09f510dcde45bf06b390c.jpg](../iclr_results/149_miniCTX_%20Neural%20Theorem%20Proving%20with%20%28Long-)Contexts/images/f620f24ce4903b1c21870da3a8c9f469cfd6243831a09f510dcde45bf06b390c.jpg)

### Tables

![02ad4491414b812f8fb96b301dbda29a99a43d09438e05d4a1b9b601d7409159.jpg](../iclr_results/149_miniCTX_%20Neural%20Theorem%20Proving%20with%20%28Long-)Contexts/tables/02ad4491414b812f8fb96b301dbda29a99a43d09438e05d4a1b9b601d7409159.jpg)

![14b9f32b30ae35137f080207ebef88ac046e445e85a73d6d0866c1e56498ceea.jpg](../iclr_results/149_miniCTX_%20Neural%20Theorem%20Proving%20with%20%28Long-)Contexts/tables/14b9f32b30ae35137f080207ebef88ac046e445e85a73d6d0866c1e56498ceea.jpg)

![217372e2bca5f7542b5cfc20781a37a84a139697ab92ec02d3990fc5cd1232b6.jpg](../iclr_results/149_miniCTX_%20Neural%20Theorem%20Proving%20with%20%28Long-)Contexts/tables/217372e2bca5f7542b5cfc20781a37a84a139697ab92ec02d3990fc5cd1232b6.jpg)

![2da78dfa345275b0e50dd3600d19b81d396f380db133c1c4b23f459835342258.jpg](../iclr_results/149_miniCTX_%20Neural%20Theorem%20Proving%20with%20%28Long-)Contexts/tables/2da78dfa345275b0e50dd3600d19b81d396f380db133c1c4b23f459835342258.jpg)

![5b688e8b8375ff0df9fbc8acc1c1ae12403d30afa59ad3d5c454dd9ff8211721.jpg](../iclr_results/149_miniCTX_%20Neural%20Theorem%20Proving%20with%20%28Long-)Contexts/tables/5b688e8b8375ff0df9fbc8acc1c1ae12403d30afa59ad3d5c454dd9ff8211721.jpg)

![9f62f1e8ad3a83cfa8e9ab1ad205a20438014cdcecdaa038d5542baa6e928801.jpg](../iclr_results/149_miniCTX_%20Neural%20Theorem%20Proving%20with%20%28Long-)Contexts/tables/9f62f1e8ad3a83cfa8e9ab1ad205a20438014cdcecdaa038d5542baa6e928801.jpg)

![a1658b70920516622042d4e96f6278cc2cf968aae9091464d5841a2edb3c9b43.jpg](../iclr_results/149_miniCTX_%20Neural%20Theorem%20Proving%20with%20%28Long-)Contexts/tables/a1658b70920516622042d4e96f6278cc2cf968aae9091464d5841a2edb3c9b43.jpg)

![c9b961ec4904f47a94af3fe9e0b6d8becdc4e7a12805806cabebe339c3845d41.jpg](../iclr_results/149_miniCTX_%20Neural%20Theorem%20Proving%20with%20%28Long-)Contexts/tables/c9b961ec4904f47a94af3fe9e0b6d8becdc4e7a12805806cabebe339c3845d41.jpg)

![cca21d25e2c59aee4e30fe7be10a74e5dd239903fea3bcf18fa874b2e2528c30.jpg](../iclr_results/149_miniCTX_%20Neural%20Theorem%20Proving%20with%20%28Long-)Contexts/tables/cca21d25e2c59aee4e30fe7be10a74e5dd239903fea3bcf18fa874b2e2528c30.jpg)

![e295bd1ba64a6bc41fd9ad0ebe0bc3f436cfbb89d3e1730172019d3b09156b56.jpg](../iclr_results/149_miniCTX_%20Neural%20Theorem%20Proving%20with%20%28Long-)Contexts/tables/e295bd1ba64a6bc41fd9ad0ebe0bc3f436cfbb89d3e1730172019d3b09156b56.jpg)

## Residual Deep Gaussian Processes on Manifolds

### Images

![126a5608e7981739d0e85ca038d12b113543625a6a27c7712ff098fe80f7e9b2.jpg](../iclr_results/150_Residual%20Deep%20Gaussian%20Processes%20on%20Manifolds/images/126a5608e7981739d0e85ca038d12b113543625a6a27c7712ff098fe80f7e9b2.jpg)

![13fc288fc2d8839e14263e3a39ea71dd3f64e8bac6bd4c319ca8d0d6ad19467f.jpg](../iclr_results/150_Residual%20Deep%20Gaussian%20Processes%20on%20Manifolds/images/13fc288fc2d8839e14263e3a39ea71dd3f64e8bac6bd4c319ca8d0d6ad19467f.jpg)

![250069ded2cde11ff8eb686bd7bfc3833bf605e8860ca8bf3baeec59f0080c6b.jpg](../iclr_results/150_Residual%20Deep%20Gaussian%20Processes%20on%20Manifolds/images/250069ded2cde11ff8eb686bd7bfc3833bf605e8860ca8bf3baeec59f0080c6b.jpg)

![2c649b46e4e6f001a27d4f185ac4e3e8232531bc2ab19913ec7a2177ca172b8d.jpg](../iclr_results/150_Residual%20Deep%20Gaussian%20Processes%20on%20Manifolds/images/2c649b46e4e6f001a27d4f185ac4e3e8232531bc2ab19913ec7a2177ca172b8d.jpg)

![335162ab93dde37553476e9158b1ab3be77bcdfdf531723588687f41fbfd3c87.jpg](../iclr_results/150_Residual%20Deep%20Gaussian%20Processes%20on%20Manifolds/images/335162ab93dde37553476e9158b1ab3be77bcdfdf531723588687f41fbfd3c87.jpg)

![33849bc242e49cd5d641520469ca065a1d0b3e4fad2a6ddfd999ad0e1279b410.jpg](../iclr_results/150_Residual%20Deep%20Gaussian%20Processes%20on%20Manifolds/images/33849bc242e49cd5d641520469ca065a1d0b3e4fad2a6ddfd999ad0e1279b410.jpg)

![376ef941676e4687608e87c5b676c3cc56fb351209a6f901c6126a48223468f6.jpg](../iclr_results/150_Residual%20Deep%20Gaussian%20Processes%20on%20Manifolds/images/376ef941676e4687608e87c5b676c3cc56fb351209a6f901c6126a48223468f6.jpg)

![41ab5fe992e8c82ae782b0970b09395f63e7afe6b969b96bf470812fe9dca663.jpg](../iclr_results/150_Residual%20Deep%20Gaussian%20Processes%20on%20Manifolds/images/41ab5fe992e8c82ae782b0970b09395f63e7afe6b969b96bf470812fe9dca663.jpg)

![4d5524e9e026a098cbc11d62a273aaf5537393143d7f3af1b0d66e3c332a3aac.jpg](../iclr_results/150_Residual%20Deep%20Gaussian%20Processes%20on%20Manifolds/images/4d5524e9e026a098cbc11d62a273aaf5537393143d7f3af1b0d66e3c332a3aac.jpg)

![4fd25481975b16ad8c803822a08cf00cea00335c41d7d4d14289523605eee56a.jpg](../iclr_results/150_Residual%20Deep%20Gaussian%20Processes%20on%20Manifolds/images/4fd25481975b16ad8c803822a08cf00cea00335c41d7d4d14289523605eee56a.jpg)

![5a0be59e397693e80f20bf36e24edee8803429806a7a960c8ecb9e936a401d89.jpg](../iclr_results/150_Residual%20Deep%20Gaussian%20Processes%20on%20Manifolds/images/5a0be59e397693e80f20bf36e24edee8803429806a7a960c8ecb9e936a401d89.jpg)

![651e8631b28b358dd673136b479751db280f6ea4725a24e14fa2143275816b3e.jpg](../iclr_results/150_Residual%20Deep%20Gaussian%20Processes%20on%20Manifolds/images/651e8631b28b358dd673136b479751db280f6ea4725a24e14fa2143275816b3e.jpg)

![654351f153b4790dfc00fd2035542628032b865df2385ddb49f8579779d5c4aa.jpg](../iclr_results/150_Residual%20Deep%20Gaussian%20Processes%20on%20Manifolds/images/654351f153b4790dfc00fd2035542628032b865df2385ddb49f8579779d5c4aa.jpg)

![6c1092120846bf07c457be1bc9f3ab2a155ad2361414efb55b487ccb4640790a.jpg](../iclr_results/150_Residual%20Deep%20Gaussian%20Processes%20on%20Manifolds/images/6c1092120846bf07c457be1bc9f3ab2a155ad2361414efb55b487ccb4640790a.jpg)

![771c2b86526aa1e87689f942a92acf6172ad5056f2e6efa56676a52940fa3cce.jpg](../iclr_results/150_Residual%20Deep%20Gaussian%20Processes%20on%20Manifolds/images/771c2b86526aa1e87689f942a92acf6172ad5056f2e6efa56676a52940fa3cce.jpg)

![9a7a6f4999b8b2d3bd8c2773842b9b04daed38a3e7b0b009e979c8642fe2181c.jpg](../iclr_results/150_Residual%20Deep%20Gaussian%20Processes%20on%20Manifolds/images/9a7a6f4999b8b2d3bd8c2773842b9b04daed38a3e7b0b009e979c8642fe2181c.jpg)

![c8749ccbfede705b063d6de12e91c307580eef5894b611c9c62ca2cad932f97d.jpg](../iclr_results/150_Residual%20Deep%20Gaussian%20Processes%20on%20Manifolds/images/c8749ccbfede705b063d6de12e91c307580eef5894b611c9c62ca2cad932f97d.jpg)

![ce1d645c0bdbeac90fed8a57ebd5d044dd45a25a80eee3d980172e8c3e49feb0.jpg](../iclr_results/150_Residual%20Deep%20Gaussian%20Processes%20on%20Manifolds/images/ce1d645c0bdbeac90fed8a57ebd5d044dd45a25a80eee3d980172e8c3e49feb0.jpg)

![cf53e0ebb4f0987c86c3155950258285fc9b7ef3ff4987d736acff7d6ad012a3.jpg](../iclr_results/150_Residual%20Deep%20Gaussian%20Processes%20on%20Manifolds/images/cf53e0ebb4f0987c86c3155950258285fc9b7ef3ff4987d736acff7d6ad012a3.jpg)

![e49e7c9be7e99efc9513895f62d0e9be6a4f9e5224e5fbeea62fbc6feee1bf04.jpg](../iclr_results/150_Residual%20Deep%20Gaussian%20Processes%20on%20Manifolds/images/e49e7c9be7e99efc9513895f62d0e9be6a4f9e5224e5fbeea62fbc6feee1bf04.jpg)

![fa7beab63146974b925bac24f804147923141d99cc4086a1448b7a964dc4c877.jpg](../iclr_results/150_Residual%20Deep%20Gaussian%20Processes%20on%20Manifolds/images/fa7beab63146974b925bac24f804147923141d99cc4086a1448b7a964dc4c877.jpg)
