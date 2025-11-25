# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 3 of 100

## 目录 (Table of Contents)

1. [Unlocking the Power of Function Vectors for Characterizing and Mitigating Catastrophic Forgetting in Continual Instruction Tuning](#Unlocking-the-Power-of-Function-Vectors-for-Characterizing-and-Mitigating-Catastrophic-Forgetting-in-Continual-Instruction-Tuning)
2. [Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport](#Learning-stochastic-dynamics-from-snapshots-through-regularized-unbalanced-optimal-transport)
3. [Composing Unbalanced Flows for Flexible Docking and Relaxation](#Composing-Unbalanced-Flows-for-Flexible-Docking-and-Relaxation)
4. [A Computational Framework for Modeling Emergence of Color Vision in the Human Brain](#A-Computational-Framework-for-Modeling-Emergence-of-Color-Vision-in-the-Human-Brain)
5. [Improving Probabilistic Diffusion Models With Optimal Diagonal Covariance Matching](#Improving-Probabilistic-Diffusion-Models-With-Optimal-Diagonal-Covariance-Matching)
6. [BIRD: A Trustworthy Bayesian Inference Framework for Large Language Models](#BIRD-A-Trustworthy-Bayesian-Inference-Framework-for-Large-Language-Models)
7. [Combatting Dimensional Collapse in LLM Pre-Training Data via Submodular File Selection](#Combatting-Dimensional-Collapse-in-LLM-Pre-Training-Data-via-Submodular-File-Selection)
8. [Influence Functions for Scalable Data Attribution in Diffusion Models](#Influence-Functions-for-Scalable-Data-Attribution-in-Diffusion-Models)
9. [Language Representations Can be What Recommenders Need: Findings and Potentials](#Language-Representations-Can-be-What-Recommenders-Need-Findings-and-Potentials)
10. [Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition](#Knowledge-Entropy-Decay-during-Language-Model-Pretraining-Hinders-New-Knowledge-Acquisition)
11. [Your Mixture-of-Experts LLM Is Secretly an Embedding Model for Free](#Your-Mixture-of-Experts-LLM-Is-Secretly-an-Embedding-Model-for-Free)
12. [Emergence of meta-stable clustering in mean-field transformer models](#Emergence-of-meta-stable-clustering-in-mean-field-transformer-models)
13. [Data Selection via Optimal Control for Language Models](#Data-Selection-via-Optimal-Control-for-Language-Models)
14. [Feedback Favors the Generalization of Neural ODEs](#Feedback-Favors-the-Generalization-of-Neural-ODEs)
15. [Comparing noisy neural population dynamics using optimal transport distances](#Comparing-noisy-neural-population-dynamics-using-optimal-transport-distances)
16. [Subgraph Federated Learning for Local Generalization](#Subgraph-Federated-Learning-for-Local-Generalization)
17. [RB-Modulation: Training-Free Stylization using Reference-Based Modulation](#RB-Modulation-Training-Free-Stylization-using-Reference-Based-Modulation)
18. [The Geometry of Categorical and Hierarchical Concepts in Large Language Models](#The-Geometry-of-Categorical-and-Hierarchical-Concepts-in-Large-Language-Models)
19. [TopoLM: brain-like spatio-functional organization in a topographic language model](#TopoLM-brain-like-spatio-functional-organization-in-a-topographic-language-model)
20. [Problem-Parameter-Free Federated Learning](#Problem-Parameter-Free-Federated-Learning)
21. [Latent Bayesian Optimization via Autoregressive Normalizing Flows](#Latent-Bayesian-Optimization-via-Autoregressive-Normalizing-Flows)
22. [BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions](#BigCodeBench-Benchmarking-Code-Generation-with-Diverse-Function-Calls-and-Complex-Instructions)
23. [ReGenesis: LLMs can Grow into Reasoning Generalists via Self-Improvement](#ReGenesis-LLMs-can-Grow-into-Reasoning-Generalists-via-Self-Improvement)
24. [LaMPlace: Learning to Optimize Cross-Stage Metrics in Macro Placement](#LaMPlace-Learning-to-Optimize-Cross-Stage-Metrics-in-Macro-Placement)
25. [MOS: Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection](#MOS-Model-Synergy-for-Test-Time-Adaptation-on-LiDAR-Based-3D-Object-Detection)
26. [On Conformal Isometry of Grid Cells: Learning Distance-Preserving Position Embedding](#On-Conformal-Isometry-of-Grid-Cells-Learning-Distance-Preserving-Position-Embedding)
27. [Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows](#Spider-20-Evaluating-Language-Models-on-Real-World-Enterprise-Text-to-SQL-Workflows)
28. [EmbodiedSAM: Online Segment Any 3D Thing in Real Time](#EmbodiedSAM-Online-Segment-Any-3D-Thing-in-Real-Time)
29. [Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping](#Dynamic-Multimodal-Evaluation-with-Flexible-Complexity-by-Vision-Language-Bootstrapping)
30. [LARP: Tokenizing Videos with a Learned Autoregressive Generative Prior](#LARP-Tokenizing-Videos-with-a-Learned-Autoregressive-Generative-Prior)
31. [Knowing Your Target: Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding](#Knowing-Your-Target-Target-Aware-Transformer-Makes-Better-Spatio-Temporal-Video-Grounding)
32. [Self-Improvement in Language Models: The Sharpening Mechanism](#Self-Improvement-in-Language-Models-The-Sharpening-Mechanism)
33. [Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models](#Do-I-Know-This-Entity-Knowledge-Awareness-and-Hallucinations-in-Language-Models)
34. [LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization](#LoRA-Done-RITE-Robust-Invariant-Transformation-Equilibration-for-LoRA-Optimization)
35. [Reasoning Elicitation in Language Models via Counterfactual Feedback](#Reasoning-Elicitation-in-Language-Models-via-Counterfactual-Feedback)
36. [Attention as a Hypernetwork](#Attention-as-a-Hypernetwork)
37. [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](#Unlocking-State-Tracking-in-Linear-RNNs-Through-Negative-Eigenvalues)

---


## Unlocking the Power of Function Vectors for Characterizing and Mitigating Catastrophic Forgetting in Continual Instruction Tuning

### Images

![00b21e7c76b2a2d20030e3d025f29eadb63a77522cf3080b114d57da17837d71.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/images/00b21e7c76b2a2d20030e3d025f29eadb63a77522cf3080b114d57da17837d71.jpg)

![1539c5519c047a64d2cf1ab6238ddba4de32da9542d534b12f1689ef308d2d09.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/images/1539c5519c047a64d2cf1ab6238ddba4de32da9542d534b12f1689ef308d2d09.jpg)

![46e2463df4720a21201d2b8163c0f2229936566d876023576f12d292fbe6612a.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/images/46e2463df4720a21201d2b8163c0f2229936566d876023576f12d292fbe6612a.jpg)

![5a245903f4811ff4b6ef0bf581f9195e295fe30be8ec5c0e36adecfb6388f74e.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/images/5a245903f4811ff4b6ef0bf581f9195e295fe30be8ec5c0e36adecfb6388f74e.jpg)

![6eed9fb580ad1504cd1ab07d622286d5032d3b85c55fce2f4e25d185f9b548c8.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/images/6eed9fb580ad1504cd1ab07d622286d5032d3b85c55fce2f4e25d185f9b548c8.jpg)

![6f3e420b05920861a2074660c2f767740fa227df6157900051553063ce4b6e0d.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/images/6f3e420b05920861a2074660c2f767740fa227df6157900051553063ce4b6e0d.jpg)

![7d1a444309304e5d1d9cfa61bba3baf0fb13789b1a577da3b3505b0df9baa8d3.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/images/7d1a444309304e5d1d9cfa61bba3baf0fb13789b1a577da3b3505b0df9baa8d3.jpg)

![a769db06d15098ed33275293f0b5b3083dfe70aa2422229aaf7ce705a82b1b3e.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/images/a769db06d15098ed33275293f0b5b3083dfe70aa2422229aaf7ce705a82b1b3e.jpg)

![aa0a15bdab0355d003d86009e49959acca5dc664e5c955128c42cab9b67f09c1.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/images/aa0a15bdab0355d003d86009e49959acca5dc664e5c955128c42cab9b67f09c1.jpg)

![ad4f0f76c5bc356927fbe59515f3704c8460bcf245772a1a9b6670f09c1b8b50.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/images/ad4f0f76c5bc356927fbe59515f3704c8460bcf245772a1a9b6670f09c1b8b50.jpg)

![b2f3daf8336f10a976431aa0f47dc24c61ef2290e0390e2c3d0a3f12eebdac38.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/images/b2f3daf8336f10a976431aa0f47dc24c61ef2290e0390e2c3d0a3f12eebdac38.jpg)

![b81660c077ac57acce192239c07166e13ad740b1a4ce3408349f81fd8e9e7e5f.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/images/b81660c077ac57acce192239c07166e13ad740b1a4ce3408349f81fd8e9e7e5f.jpg)

![c17661327b21767bb7315d1bd28ee392d6193e97a15f64a77a15c2e18ec12c28.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/images/c17661327b21767bb7315d1bd28ee392d6193e97a15f64a77a15c2e18ec12c28.jpg)

![c82cf40d60c5275c18e7da83b6c06b392e69db0fe9758d96a86e13ebbc38cd22.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/images/c82cf40d60c5275c18e7da83b6c06b392e69db0fe9758d96a86e13ebbc38cd22.jpg)

![cc918997438d6a7af78f838831d2544e5e48ca96d3ac64344bfa0ab8c10b7833.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/images/cc918997438d6a7af78f838831d2544e5e48ca96d3ac64344bfa0ab8c10b7833.jpg)

![e587dae971a7f338b71a31f1f06260adc14e72b4bb50200752a0129c9ca76377.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/images/e587dae971a7f338b71a31f1f06260adc14e72b4bb50200752a0129c9ca76377.jpg)

![e5d2b9d30513e9cc577bf7d9a377601596b7055d92aadddbe62eafb4c5d8d0c5.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/images/e5d2b9d30513e9cc577bf7d9a377601596b7055d92aadddbe62eafb4c5d8d0c5.jpg)

### Tables

![150aaa3d8588ab14636a34163b413ab7ab9bfaf7e39dc6120def1124120a1ec0.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/tables/150aaa3d8588ab14636a34163b413ab7ab9bfaf7e39dc6120def1124120a1ec0.jpg)

![27f93458df687f8b15575a93a3c56c0199554b6579c64e78d12405a24075fdd1.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/tables/27f93458df687f8b15575a93a3c56c0199554b6579c64e78d12405a24075fdd1.jpg)

![481b7edab8b59a79425c52fccf1237a70bdc76bd83c29047b13b9ad6259313cf.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/tables/481b7edab8b59a79425c52fccf1237a70bdc76bd83c29047b13b9ad6259313cf.jpg)

![970b5c11940bb7b144fe149d647750a7f3a564d63b6152a61eeb5911099cc9fd.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/tables/970b5c11940bb7b144fe149d647750a7f3a564d63b6152a61eeb5911099cc9fd.jpg)

![9a6fb103a11da877564192d2e5668909190426c8f84ff7efaadff18469570804.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/tables/9a6fb103a11da877564192d2e5668909190426c8f84ff7efaadff18469570804.jpg)

![cde2ef5045998c644054502c1329ede1b82ff696d67bccb6eb1e4b4d3d5ee9ee.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/tables/cde2ef5045998c644054502c1329ede1b82ff696d67bccb6eb1e4b4d3d5ee9ee.jpg)

![d9238b6d9b463ccaba451b4be7610510881cfdebabe901710d69b2a76cf682ce.jpg](../iclr_results/75_On%20the%20Role%20of%20Attention%20Heads%20in%20Large%20Language%20Model%20Safety/tables/d9238b6d9b463ccaba451b4be7610510881cfdebabe901710d69b2a76cf682ce.jpg)

## Unlocking the Power of Function Vectors for Characterizing and Mitigating Catastrophic Forgetting in Continual Instruction Tuning


### Images

![0206de552d1a528089732020356ed866843c67f02cda816cf9dc5fdc51c8d98e.jpg](../iclr_results/76_Unlocking%20the%20Power%20of%20Function%20Vectors%20for%20Characterizing%20and%20Mitigating%20Catastrophic%20Forgetting%20in/images/0206de552d1a528089732020356ed866843c67f02cda816cf9dc5fdc51c8d98e.jpg)

![0ad19089c67ef3a6124023353de2ab020604dde64a0dd3b59503e0d255eaad17.jpg](../iclr_results/76_Unlocking%20the%20Power%20of%20Function%20Vectors%20for%20Characterizing%20and%20Mitigating%20Catastrophic%20Forgetting%20in/images/0ad19089c67ef3a6124023353de2ab020604dde64a0dd3b59503e0d255eaad17.jpg)

![440302af2d21228becf457b40bc2a39ee7e23da946c2f8b84bc728ca8c94eb45.jpg](../iclr_results/76_Unlocking%20the%20Power%20of%20Function%20Vectors%20for%20Characterizing%20and%20Mitigating%20Catastrophic%20Forgetting%20in/images/440302af2d21228becf457b40bc2a39ee7e23da946c2f8b84bc728ca8c94eb45.jpg)

![46cf6891f9584946638da27f573439588d6557082127803614798b29f8dae7f7.jpg](../iclr_results/76_Unlocking%20the%20Power%20of%20Function%20Vectors%20for%20Characterizing%20and%20Mitigating%20Catastrophic%20Forgetting%20in/images/46cf6891f9584946638da27f573439588d6557082127803614798b29f8dae7f7.jpg)

![68f07db5a494ab5c94db38232a8e76cfe34b13a3d17f9e8d65a999de55081175.jpg](../iclr_results/76_Unlocking%20the%20Power%20of%20Function%20Vectors%20for%20Characterizing%20and%20Mitigating%20Catastrophic%20Forgetting%20in/images/68f07db5a494ab5c94db38232a8e76cfe34b13a3d17f9e8d65a999de55081175.jpg)

![86fca29ebbe4dad5bd3c2bd1b729e83d750240845bf1f8665167a7da12527686.jpg](../iclr_results/76_Unlocking%20the%20Power%20of%20Function%20Vectors%20for%20Characterizing%20and%20Mitigating%20Catastrophic%20Forgetting%20in/images/86fca29ebbe4dad5bd3c2bd1b729e83d750240845bf1f8665167a7da12527686.jpg)

![b5c13d1cdfb7207628b6404059434804a4925ccba6b77af5133c9ba0c56c0b24.jpg](../iclr_results/76_Unlocking%20the%20Power%20of%20Function%20Vectors%20for%20Characterizing%20and%20Mitigating%20Catastrophic%20Forgetting%20in/images/b5c13d1cdfb7207628b6404059434804a4925ccba6b77af5133c9ba0c56c0b24.jpg)

![cd3f60fa11622540f6337d1450f82390dde7113c4f4129c61ceb26feb16738e5.jpg](../iclr_results/76_Unlocking%20the%20Power%20of%20Function%20Vectors%20for%20Characterizing%20and%20Mitigating%20Catastrophic%20Forgetting%20in/images/cd3f60fa11622540f6337d1450f82390dde7113c4f4129c61ceb26feb16738e5.jpg)

![e7b94e12a69b1e3a65217ecb0fd5fdab8288602e28ce9903a2f99771dcd57af9.jpg](../iclr_results/76_Unlocking%20the%20Power%20of%20Function%20Vectors%20for%20Characterizing%20and%20Mitigating%20Catastrophic%20Forgetting%20in/images/e7b94e12a69b1e3a65217ecb0fd5fdab8288602e28ce9903a2f99771dcd57af9.jpg)

![ec1fe75d79e39d328a3550a23f2e2ee96f350cdc18c3fda710e9883c3b5f49e0.jpg](../iclr_results/76_Unlocking%20the%20Power%20of%20Function%20Vectors%20for%20Characterizing%20and%20Mitigating%20Catastrophic%20Forgetting%20in/images/ec1fe75d79e39d328a3550a23f2e2ee96f350cdc18c3fda710e9883c3b5f49e0.jpg)

![efcfba94a4c52cdb39167b34c17d9d5cf666ed698895975e6b9b91fd7a49d22b.jpg](../iclr_results/76_Unlocking%20the%20Power%20of%20Function%20Vectors%20for%20Characterizing%20and%20Mitigating%20Catastrophic%20Forgetting%20in/images/efcfba94a4c52cdb39167b34c17d9d5cf666ed698895975e6b9b91fd7a49d22b.jpg)

![f022c6d1c4fa6d473f851ecc3109a7b1d117fefe5a4fb76bd06eb763275848e6.jpg](../iclr_results/76_Unlocking%20the%20Power%20of%20Function%20Vectors%20for%20Characterizing%20and%20Mitigating%20Catastrophic%20Forgetting%20in/images/f022c6d1c4fa6d473f851ecc3109a7b1d117fefe5a4fb76bd06eb763275848e6.jpg)

![f88e226cb47ec4e2b0ff76c3b1c56f6f8ee7d15a1005dbcae67384829f886246.jpg](../iclr_results/76_Unlocking%20the%20Power%20of%20Function%20Vectors%20for%20Characterizing%20and%20Mitigating%20Catastrophic%20Forgetting%20in/images/f88e226cb47ec4e2b0ff76c3b1c56f6f8ee7d15a1005dbcae67384829f886246.jpg)

### Tables

![0a8c66e00fa15d8c8e83a7ced7b52ad66d88e39c820e17ceb2e816d76f048aab.jpg](../iclr_results/76_Unlocking%20the%20Power%20of%20Function%20Vectors%20for%20Characterizing%20and%20Mitigating%20Catastrophic%20Forgetting%20in/tables/0a8c66e00fa15d8c8e83a7ced7b52ad66d88e39c820e17ceb2e816d76f048aab.jpg)

![2b1026b7f7be1888049873a24069ddb6e8455647a9be0893f49d91c40c4c3cb8.jpg](../iclr_results/76_Unlocking%20the%20Power%20of%20Function%20Vectors%20for%20Characterizing%20and%20Mitigating%20Catastrophic%20Forgetting%20in/tables/2b1026b7f7be1888049873a24069ddb6e8455647a9be0893f49d91c40c4c3cb8.jpg)

![7816df1e616cc242415ad780b31cf53d18f5d9c06246146b392f2ea46251d5bb.jpg](../iclr_results/76_Unlocking%20the%20Power%20of%20Function%20Vectors%20for%20Characterizing%20and%20Mitigating%20Catastrophic%20Forgetting%20in/tables/7816df1e616cc242415ad780b31cf53d18f5d9c06246146b392f2ea46251d5bb.jpg)

![a580ad70404fa5ff9908cad16fffc1c88e9a07043eafdd01c970ecd8164bc8bb.jpg](../iclr_results/76_Unlocking%20the%20Power%20of%20Function%20Vectors%20for%20Characterizing%20and%20Mitigating%20Catastrophic%20Forgetting%20in/tables/a580ad70404fa5ff9908cad16fffc1c88e9a07043eafdd01c970ecd8164bc8bb.jpg)

![a599f762823e4e84c552f71c5cf8778cdb8a15e2af2683416ba75ba4bfce7c90.jpg](../iclr_results/76_Unlocking%20the%20Power%20of%20Function%20Vectors%20for%20Characterizing%20and%20Mitigating%20Catastrophic%20Forgetting%20in/tables/a599f762823e4e84c552f71c5cf8778cdb8a15e2af2683416ba75ba4bfce7c90.jpg)

![cc0159ceaa160d54b69ffe5a0cdad83fe861821693a3ae0af10390ccea0b62e9.jpg](../iclr_results/76_Unlocking%20the%20Power%20of%20Function%20Vectors%20for%20Characterizing%20and%20Mitigating%20Catastrophic%20Forgetting%20in/tables/cc0159ceaa160d54b69ffe5a0cdad83fe861821693a3ae0af10390ccea0b62e9.jpg)

![d446a5c6a754379fbed1941b0850c5aea2113303d5592a4ad8079355432a1d7c.jpg](../iclr_results/76_Unlocking%20the%20Power%20of%20Function%20Vectors%20for%20Characterizing%20and%20Mitigating%20Catastrophic%20Forgetting%20in/tables/d446a5c6a754379fbed1941b0850c5aea2113303d5592a4ad8079355432a1d7c.jpg)

![e0691dd5259762796beb1d1f8ce55d040902b563902fd263b9dde2f648c22136.jpg](../iclr_results/76_Unlocking%20the%20Power%20of%20Function%20Vectors%20for%20Characterizing%20and%20Mitigating%20Catastrophic%20Forgetting%20in/tables/e0691dd5259762796beb1d1f8ce55d040902b563902fd263b9dde2f648c22136.jpg)

## Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport


### Images

![034d538baaa5d38386fd902cc4c593abdcab19344a0e436f3d1d19610da3de0f.jpg](../iclr_results/77_Learning%20stochastic%20dynamics%20from%20snapshots%20through%20regularized%20unbalanced%20optimal%20transport/images/034d538baaa5d38386fd902cc4c593abdcab19344a0e436f3d1d19610da3de0f.jpg)

![7147c1c733fcad900b37f95fcf7319b8c18192a17165cf99ba782421495e4e7c.jpg](../iclr_results/77_Learning%20stochastic%20dynamics%20from%20snapshots%20through%20regularized%20unbalanced%20optimal%20transport/images/7147c1c733fcad900b37f95fcf7319b8c18192a17165cf99ba782421495e4e7c.jpg)

![72650a1276695f78378799993fec9c42e8c9700f57785e79d10de41059464615.jpg](../iclr_results/77_Learning%20stochastic%20dynamics%20from%20snapshots%20through%20regularized%20unbalanced%20optimal%20transport/images/72650a1276695f78378799993fec9c42e8c9700f57785e79d10de41059464615.jpg)

![7f9dc92d579c8481419a499abf25ce6922301b3cb9a64ac289f5a83b1dc4d187.jpg](../iclr_results/77_Learning%20stochastic%20dynamics%20from%20snapshots%20through%20regularized%20unbalanced%20optimal%20transport/images/7f9dc92d579c8481419a499abf25ce6922301b3cb9a64ac289f5a83b1dc4d187.jpg)

![cba2793266a6cfd32cb7bcb198658a2adb750c10558e49cac6ff0c5315683e83.jpg](../iclr_results/77_Learning%20stochastic%20dynamics%20from%20snapshots%20through%20regularized%20unbalanced%20optimal%20transport/images/cba2793266a6cfd32cb7bcb198658a2adb750c10558e49cac6ff0c5315683e83.jpg)

![cf6f4d126168017bc0e2cff89fd138a8ee7e1b0bd12056c0cbbed9908ca00825.jpg](../iclr_results/77_Learning%20stochastic%20dynamics%20from%20snapshots%20through%20regularized%20unbalanced%20optimal%20transport/images/cf6f4d126168017bc0e2cff89fd138a8ee7e1b0bd12056c0cbbed9908ca00825.jpg)

![e9af9deba9e5d342fa85bb8c6a2e9a3bfb6f09883678bff546ecf8f50495aaf3.jpg](../iclr_results/77_Learning%20stochastic%20dynamics%20from%20snapshots%20through%20regularized%20unbalanced%20optimal%20transport/images/e9af9deba9e5d342fa85bb8c6a2e9a3bfb6f09883678bff546ecf8f50495aaf3.jpg)

![f1be266f0391dd730bf3f5a53d22a67128a6df7be0c2c870b751bd1be8f16aed.jpg](../iclr_results/77_Learning%20stochastic%20dynamics%20from%20snapshots%20through%20regularized%20unbalanced%20optimal%20transport/images/f1be266f0391dd730bf3f5a53d22a67128a6df7be0c2c870b751bd1be8f16aed.jpg)

### Tables

![0aab154a6af7181f53172d8c4157a50849a42b9e449764eeb31866949c90f9fd.jpg](../iclr_results/77_Learning%20stochastic%20dynamics%20from%20snapshots%20through%20regularized%20unbalanced%20optimal%20transport/tables/0aab154a6af7181f53172d8c4157a50849a42b9e449764eeb31866949c90f9fd.jpg)

![1fa025e03098833275a891f8374824aba05a1b24ce44e23466b5c4a5ce15f9ab.jpg](../iclr_results/77_Learning%20stochastic%20dynamics%20from%20snapshots%20through%20regularized%20unbalanced%20optimal%20transport/tables/1fa025e03098833275a891f8374824aba05a1b24ce44e23466b5c4a5ce15f9ab.jpg)

![4a59a1b9348d95e7916b1e031726b01e653c498384a0718f869330a73261f742.jpg](../iclr_results/77_Learning%20stochastic%20dynamics%20from%20snapshots%20through%20regularized%20unbalanced%20optimal%20transport/tables/4a59a1b9348d95e7916b1e031726b01e653c498384a0718f869330a73261f742.jpg)

![839f7c209de3847239ca1bdef0359b43a8dd65c07c0c96c02282cc44cd110d03.jpg](../iclr_results/77_Learning%20stochastic%20dynamics%20from%20snapshots%20through%20regularized%20unbalanced%20optimal%20transport/tables/839f7c209de3847239ca1bdef0359b43a8dd65c07c0c96c02282cc44cd110d03.jpg)

![b262ac7e21eabff80494d266ec4ae4817b43e21aac154ace48272986aa0375ed.jpg](../iclr_results/77_Learning%20stochastic%20dynamics%20from%20snapshots%20through%20regularized%20unbalanced%20optimal%20transport/tables/b262ac7e21eabff80494d266ec4ae4817b43e21aac154ace48272986aa0375ed.jpg)

![b90bc674ce97c108941af6bb62b18eb0e6a95e7d92d6a588547d4ce0a8874ae5.jpg](../iclr_results/77_Learning%20stochastic%20dynamics%20from%20snapshots%20through%20regularized%20unbalanced%20optimal%20transport/tables/b90bc674ce97c108941af6bb62b18eb0e6a95e7d92d6a588547d4ce0a8874ae5.jpg)

![d81dc062c3924bb6964a244a87b3e11075a789697f020a06ea69079579739ad6.jpg](../iclr_results/77_Learning%20stochastic%20dynamics%20from%20snapshots%20through%20regularized%20unbalanced%20optimal%20transport/tables/d81dc062c3924bb6964a244a87b3e11075a789697f020a06ea69079579739ad6.jpg)

## Composing Unbalanced Flows for Flexible Docking and Relaxation


### Images

![003d22a7e9388324751e5f3ef7b488389400173b66d9d160467792437ba497bd.jpg](../iclr_results/78_Composing%20Unbalanced%20Flows%20for%20Flexible%20Docking%20and%20Relaxation/images/003d22a7e9388324751e5f3ef7b488389400173b66d9d160467792437ba497bd.jpg)

![31f88c46475ef8793ee055b2493d621118b61efc26fcb306f1ae16535292b835.jpg](../iclr_results/78_Composing%20Unbalanced%20Flows%20for%20Flexible%20Docking%20and%20Relaxation/images/31f88c46475ef8793ee055b2493d621118b61efc26fcb306f1ae16535292b835.jpg)

![3f59c57b0a45b7fc3d5e5332a503f6da4a560e41ec7ed38a5b2cc5d36ea5c667.jpg](../iclr_results/78_Composing%20Unbalanced%20Flows%20for%20Flexible%20Docking%20and%20Relaxation/images/3f59c57b0a45b7fc3d5e5332a503f6da4a560e41ec7ed38a5b2cc5d36ea5c667.jpg)

![43021dae7462d34341bb3be50d588f470eb09de2a125b761c6ffc90ef7f8c9ab.jpg](../iclr_results/78_Composing%20Unbalanced%20Flows%20for%20Flexible%20Docking%20and%20Relaxation/images/43021dae7462d34341bb3be50d588f470eb09de2a125b761c6ffc90ef7f8c9ab.jpg)

![635106c14d77c464cfc595801a45e4419db7b2fb3171292b969585d71f4ae8b9.jpg](../iclr_results/78_Composing%20Unbalanced%20Flows%20for%20Flexible%20Docking%20and%20Relaxation/images/635106c14d77c464cfc595801a45e4419db7b2fb3171292b969585d71f4ae8b9.jpg)

![71e8f56aa7cacc072f3f42729bd6146ce949a7111e5b476506605205fa43adbd.jpg](../iclr_results/78_Composing%20Unbalanced%20Flows%20for%20Flexible%20Docking%20and%20Relaxation/images/71e8f56aa7cacc072f3f42729bd6146ce949a7111e5b476506605205fa43adbd.jpg)

![79501ebb8c492abf81fc63162085bae605fb3f05c64e49d6c5f1dbf9399df8c3.jpg](../iclr_results/78_Composing%20Unbalanced%20Flows%20for%20Flexible%20Docking%20and%20Relaxation/images/79501ebb8c492abf81fc63162085bae605fb3f05c64e49d6c5f1dbf9399df8c3.jpg)

![8caa693f020fc43fe2fed18c1ce0ccb7449f027e47921ea84f30230e5182bb35.jpg](../iclr_results/78_Composing%20Unbalanced%20Flows%20for%20Flexible%20Docking%20and%20Relaxation/images/8caa693f020fc43fe2fed18c1ce0ccb7449f027e47921ea84f30230e5182bb35.jpg)

![9f80e4fad33e7f25e8aea725434afe35229df972cbf384633f781cdb607981b6.jpg](../iclr_results/78_Composing%20Unbalanced%20Flows%20for%20Flexible%20Docking%20and%20Relaxation/images/9f80e4fad33e7f25e8aea725434afe35229df972cbf384633f781cdb607981b6.jpg)

![b53277a9ffcfd30146995dc0fb2c3807c0964e751495c4884f57674d2ea78e94.jpg](../iclr_results/78_Composing%20Unbalanced%20Flows%20for%20Flexible%20Docking%20and%20Relaxation/images/b53277a9ffcfd30146995dc0fb2c3807c0964e751495c4884f57674d2ea78e94.jpg)

![cdc4f55d5064c10ac5c53acba324302b32c0b5659c7a4bdba190efebc2c2eee8.jpg](../iclr_results/78_Composing%20Unbalanced%20Flows%20for%20Flexible%20Docking%20and%20Relaxation/images/cdc4f55d5064c10ac5c53acba324302b32c0b5659c7a4bdba190efebc2c2eee8.jpg)

![e00bd6b22b563957d6109c8294c630d80ea81a5687a909db7f6ba01906aad6f4.jpg](../iclr_results/78_Composing%20Unbalanced%20Flows%20for%20Flexible%20Docking%20and%20Relaxation/images/e00bd6b22b563957d6109c8294c630d80ea81a5687a909db7f6ba01906aad6f4.jpg)

![e082e25657740d6f96323aef53521902a29c85deb9f71ff4a18992d75cfac6db.jpg](../iclr_results/78_Composing%20Unbalanced%20Flows%20for%20Flexible%20Docking%20and%20Relaxation/images/e082e25657740d6f96323aef53521902a29c85deb9f71ff4a18992d75cfac6db.jpg)

![e1e01294dc9dd3b44592786de5763f41c63135d36666716bcf0fa3fc97e8e092.jpg](../iclr_results/78_Composing%20Unbalanced%20Flows%20for%20Flexible%20Docking%20and%20Relaxation/images/e1e01294dc9dd3b44592786de5763f41c63135d36666716bcf0fa3fc97e8e092.jpg)

![e4d06108e607f72b7fc67b3cbf1438245d8d83048b6b84fe9e3fca22d19bf130.jpg](../iclr_results/78_Composing%20Unbalanced%20Flows%20for%20Flexible%20Docking%20and%20Relaxation/images/e4d06108e607f72b7fc67b3cbf1438245d8d83048b6b84fe9e3fca22d19bf130.jpg)

![ea827a9b904cb643786ccb01c180c3f4e0e5c569038fb6805496289c759c471f.jpg](../iclr_results/78_Composing%20Unbalanced%20Flows%20for%20Flexible%20Docking%20and%20Relaxation/images/ea827a9b904cb643786ccb01c180c3f4e0e5c569038fb6805496289c759c471f.jpg)

### Tables

![51bac87718b6397eda8a30c47291b9b07db67b2a33fbe60a3f1400547a0fd374.jpg](../iclr_results/78_Composing%20Unbalanced%20Flows%20for%20Flexible%20Docking%20and%20Relaxation/tables/51bac87718b6397eda8a30c47291b9b07db67b2a33fbe60a3f1400547a0fd374.jpg)

![6bba3e933f1be03120022aac68cbe387b6b0fb70bfa90820c3dd3c17a3648b0a.jpg](../iclr_results/78_Composing%20Unbalanced%20Flows%20for%20Flexible%20Docking%20and%20Relaxation/tables/6bba3e933f1be03120022aac68cbe387b6b0fb70bfa90820c3dd3c17a3648b0a.jpg)

![82ea7e5ec95759fd2570f3a0b37568ae4369f0d161efea8892e6ebd24b802f51.jpg](../iclr_results/78_Composing%20Unbalanced%20Flows%20for%20Flexible%20Docking%20and%20Relaxation/tables/82ea7e5ec95759fd2570f3a0b37568ae4369f0d161efea8892e6ebd24b802f51.jpg)

![8b8a468106c4f856e0e7b24933a238b3c777f80fb08ee40f309d3719c7f005ed.jpg](../iclr_results/78_Composing%20Unbalanced%20Flows%20for%20Flexible%20Docking%20and%20Relaxation/tables/8b8a468106c4f856e0e7b24933a238b3c777f80fb08ee40f309d3719c7f005ed.jpg)

![df1c2d8650bcb3e370ec7534165e94fd1e00058aa80e1f9ad7a2d8fbeac97e6b.jpg](../iclr_results/78_Composing%20Unbalanced%20Flows%20for%20Flexible%20Docking%20and%20Relaxation/tables/df1c2d8650bcb3e370ec7534165e94fd1e00058aa80e1f9ad7a2d8fbeac97e6b.jpg)

## A Computational Framework for Modeling Emergence of Color Vision in the Human Brain


### Images

![0b0a3e71dbda16e8cd24719f52f477a5a42cb5214ec7ab873be59062e952024e.jpg](../iclr_results/79_A%20Computational%20Framework%20for%20Modeling%20Emergence%20of%20Color%20Vision%20in%20the%20Human%20Brain/images/0b0a3e71dbda16e8cd24719f52f477a5a42cb5214ec7ab873be59062e952024e.jpg)

![2f3ace62d95d1ab8bf09bd70d3abd389badf144894e0b12c6ff3a9be803a80be.jpg](../iclr_results/79_A%20Computational%20Framework%20for%20Modeling%20Emergence%20of%20Color%20Vision%20in%20the%20Human%20Brain/images/2f3ace62d95d1ab8bf09bd70d3abd389badf144894e0b12c6ff3a9be803a80be.jpg)

![37e8fea536be9acc79717ceb591484172d5faabee620607091503fe627f21ec0.jpg](../iclr_results/79_A%20Computational%20Framework%20for%20Modeling%20Emergence%20of%20Color%20Vision%20in%20the%20Human%20Brain/images/37e8fea536be9acc79717ceb591484172d5faabee620607091503fe627f21ec0.jpg)

![384e8eaac01c51b08bfca779cf3ec1af0507761b5fcb84d332a2a34be5e99161.jpg](../iclr_results/79_A%20Computational%20Framework%20for%20Modeling%20Emergence%20of%20Color%20Vision%20in%20the%20Human%20Brain/images/384e8eaac01c51b08bfca779cf3ec1af0507761b5fcb84d332a2a34be5e99161.jpg)

![4cdd36fd02ffeee9e9c5fd22a9665d83612dd72b39d3efc0d3733b177966c62a.jpg](../iclr_results/79_A%20Computational%20Framework%20for%20Modeling%20Emergence%20of%20Color%20Vision%20in%20the%20Human%20Brain/images/4cdd36fd02ffeee9e9c5fd22a9665d83612dd72b39d3efc0d3733b177966c62a.jpg)

![53ded03667a837b7efb81bedee8aed21daf51191338a6d622405fdfe91f9cd75.jpg](../iclr_results/79_A%20Computational%20Framework%20for%20Modeling%20Emergence%20of%20Color%20Vision%20in%20the%20Human%20Brain/images/53ded03667a837b7efb81bedee8aed21daf51191338a6d622405fdfe91f9cd75.jpg)

![816e630d717b6f741a52a8188b7c9c0135d8ed2bc8cb059348bdee74a2cfb587.jpg](../iclr_results/79_A%20Computational%20Framework%20for%20Modeling%20Emergence%20of%20Color%20Vision%20in%20the%20Human%20Brain/images/816e630d717b6f741a52a8188b7c9c0135d8ed2bc8cb059348bdee74a2cfb587.jpg)

![a5eab18900afbecf0271694fab8ade9affe5be9937a2f197adb11c9d37a6dd55.jpg](../iclr_results/79_A%20Computational%20Framework%20for%20Modeling%20Emergence%20of%20Color%20Vision%20in%20the%20Human%20Brain/images/a5eab18900afbecf0271694fab8ade9affe5be9937a2f197adb11c9d37a6dd55.jpg)

![ab390996def87dee279ca8b00c852b06221b7b84d6fed85c0990351be64a81c1.jpg](../iclr_results/79_A%20Computational%20Framework%20for%20Modeling%20Emergence%20of%20Color%20Vision%20in%20the%20Human%20Brain/images/ab390996def87dee279ca8b00c852b06221b7b84d6fed85c0990351be64a81c1.jpg)

![ad7380d5c72520f18802612605a5a9e60709f333ca9ee51be70df410a0074132.jpg](../iclr_results/79_A%20Computational%20Framework%20for%20Modeling%20Emergence%20of%20Color%20Vision%20in%20the%20Human%20Brain/images/ad7380d5c72520f18802612605a5a9e60709f333ca9ee51be70df410a0074132.jpg)

![c198213a4fdc0167957365b2aedcf112c42303d13202780b7a6636971ac53b18.jpg](../iclr_results/79_A%20Computational%20Framework%20for%20Modeling%20Emergence%20of%20Color%20Vision%20in%20the%20Human%20Brain/images/c198213a4fdc0167957365b2aedcf112c42303d13202780b7a6636971ac53b18.jpg)

![ca8bd425f26d3c59928526847054b5c2d729ba5c2a31ed694c1b03b905c5ce2e.jpg](../iclr_results/79_A%20Computational%20Framework%20for%20Modeling%20Emergence%20of%20Color%20Vision%20in%20the%20Human%20Brain/images/ca8bd425f26d3c59928526847054b5c2d729ba5c2a31ed694c1b03b905c5ce2e.jpg)

![d1b955c41d105617a78675e6c26aadfd0048b5b5f837e3b24b19c18c00ee4dab.jpg](../iclr_results/79_A%20Computational%20Framework%20for%20Modeling%20Emergence%20of%20Color%20Vision%20in%20the%20Human%20Brain/images/d1b955c41d105617a78675e6c26aadfd0048b5b5f837e3b24b19c18c00ee4dab.jpg)

![ffee6ba18aec626e1b8c980202f9eaa757497645b9b0225c3647b928039d86c4.jpg](../iclr_results/79_A%20Computational%20Framework%20for%20Modeling%20Emergence%20of%20Color%20Vision%20in%20the%20Human%20Brain/images/ffee6ba18aec626e1b8c980202f9eaa757497645b9b0225c3647b928039d86c4.jpg)

### Tables

![eb2de1c8404934c4eb730f2e70a1427b4268fdda5d59766343a3922d71ec4922.jpg](../iclr_results/79_A%20Computational%20Framework%20for%20Modeling%20Emergence%20of%20Color%20Vision%20in%20the%20Human%20Brain/tables/eb2de1c8404934c4eb730f2e70a1427b4268fdda5d59766343a3922d71ec4922.jpg)

## Improving Probabilistic Diffusion Models With Optimal Diagonal Covariance Matching


### Images

![17f3383ad5e4f8be66a18cb77a6979aee60a16fc37339f1e405ad64214e85c7d.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/images/17f3383ad5e4f8be66a18cb77a6979aee60a16fc37339f1e405ad64214e85c7d.jpg)

![336bae1f4dcdf0abad00e8bad9b09fe284d5736e8d3e47fde0d77cc8fba7d8c6.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/images/336bae1f4dcdf0abad00e8bad9b09fe284d5736e8d3e47fde0d77cc8fba7d8c6.jpg)

![3fc1309526224cb025bf1c2f700cf68493130f151fe096531cef558e1a971bfc.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/images/3fc1309526224cb025bf1c2f700cf68493130f151fe096531cef558e1a971bfc.jpg)

![5522975284094a72fe752519897cf614633039ce0a99c9b2113af8c65967ba25.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/images/5522975284094a72fe752519897cf614633039ce0a99c9b2113af8c65967ba25.jpg)

![5c89c2cbd6c1e54626d22e6d850b548ea5a4f26b64c855595901cc22b557f9c8.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/images/5c89c2cbd6c1e54626d22e6d850b548ea5a4f26b64c855595901cc22b557f9c8.jpg)

![5e6f2b99a6795bc124796387a3eb8deb967861a5cb216cdca11e0859843a8315.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/images/5e6f2b99a6795bc124796387a3eb8deb967861a5cb216cdca11e0859843a8315.jpg)

![70c0bd9b73afd9a9ee2ef050d15b1f357c2a49cf6e52a7a706c0f0330e13750e.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/images/70c0bd9b73afd9a9ee2ef050d15b1f357c2a49cf6e52a7a706c0f0330e13750e.jpg)

![7b5e1e4a32c1676a18aa327c6b33a36e604209c0ed1cd2a3f5153e2661b73941.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/images/7b5e1e4a32c1676a18aa327c6b33a36e604209c0ed1cd2a3f5153e2661b73941.jpg)

![ac35a38c6630c4f56cd1b7f5ede11f39c6c794b2ead87376c9240b33bd525b25.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/images/ac35a38c6630c4f56cd1b7f5ede11f39c6c794b2ead87376c9240b33bd525b25.jpg)

![d6d2305d4cd65ce02ac6aa697a024417db743945317c3d48cce42f1adf5c2884.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/images/d6d2305d4cd65ce02ac6aa697a024417db743945317c3d48cce42f1adf5c2884.jpg)

![df56b703504f1f8374e0112c44018f18d0de7d09b50a8532ca1a18d593296076.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/images/df56b703504f1f8374e0112c44018f18d0de7d09b50a8532ca1a18d593296076.jpg)

![e7b70485409bb4ce51e9b56e41ee282febbcf2ebae0a51cde14fb544a18224ac.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/images/e7b70485409bb4ce51e9b56e41ee282febbcf2ebae0a51cde14fb544a18224ac.jpg)

### Tables

![01945ad36a75f94f36a28ae25fc99f5084df8782fc5f54e04d56dff0c2763515.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/tables/01945ad36a75f94f36a28ae25fc99f5084df8782fc5f54e04d56dff0c2763515.jpg)

![17ca4cd41edc4c696292ed9a166bf0a30ec93693acd181b1fa52d38c459a9841.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/tables/17ca4cd41edc4c696292ed9a166bf0a30ec93693acd181b1fa52d38c459a9841.jpg)

![1f92d81ebcc2e1ceec638e19ecf9784e05ecb2b3a133dae43d04354ddf2677cc.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/tables/1f92d81ebcc2e1ceec638e19ecf9784e05ecb2b3a133dae43d04354ddf2677cc.jpg)

![293a14ff30104cb622890013dc6bfd72a2d7d328746d336584f1c512d8aa0520.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/tables/293a14ff30104cb622890013dc6bfd72a2d7d328746d336584f1c512d8aa0520.jpg)

![2d479c89cf304b6db8dc37c9cb12d5e2b9dd515bc9ca1506408c15a5687d31b7.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/tables/2d479c89cf304b6db8dc37c9cb12d5e2b9dd515bc9ca1506408c15a5687d31b7.jpg)

![3c08fc06f859beb09b6dd099c70774bf0b46e9c39b7e53f7c4a07663773e9de8.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/tables/3c08fc06f859beb09b6dd099c70774bf0b46e9c39b7e53f7c4a07663773e9de8.jpg)

![683466550ffd0d7e6b95f9e7fb2bcc946d9e4866266397abbe2540a5a83ae3d5.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/tables/683466550ffd0d7e6b95f9e7fb2bcc946d9e4866266397abbe2540a5a83ae3d5.jpg)

![686adde63f21c302867cca66e0a481743e3e50a42ed6266b1f269e74b73b6da9.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/tables/686adde63f21c302867cca66e0a481743e3e50a42ed6266b1f269e74b73b6da9.jpg)

![aa053634b261d9b6b3fa0e0ef3e800cf88986cd054fc88cba28c6c17fd0ac528.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/tables/aa053634b261d9b6b3fa0e0ef3e800cf88986cd054fc88cba28c6c17fd0ac528.jpg)

![bb34729424e8c9161725d4a9969970904418a37e3c032f1f00b6d372383c0fef.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/tables/bb34729424e8c9161725d4a9969970904418a37e3c032f1f00b6d372383c0fef.jpg)

![ce5dd983ea7dd1e4b3069d7115265699ec7538ca5a0f496e0076aee062eee35b.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/tables/ce5dd983ea7dd1e4b3069d7115265699ec7538ca5a0f496e0076aee062eee35b.jpg)

![d2e527c2f8b50578971435225a70e1661b608cef216206383f8c03fb98441c95.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/tables/d2e527c2f8b50578971435225a70e1661b608cef216206383f8c03fb98441c95.jpg)

![ef5947e9f2b652799e39e28a56c50288bca5dd6ac7dd28d7a3cbadf075280625.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/tables/ef5947e9f2b652799e39e28a56c50288bca5dd6ac7dd28d7a3cbadf075280625.jpg)

![f6ed3c5f3da39d48897c1820b1155b430d203841273c1bdcb3b63541fb2f7542.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/tables/f6ed3c5f3da39d48897c1820b1155b430d203841273c1bdcb3b63541fb2f7542.jpg)

![f7349ad500a0545bfa8047458b1bdfc98caeabf319cb1d43e560a1ab01f1884a.jpg](../iclr_results/80_Improving%20Probabilistic%20Diffusion%20Models%20With%20Optimal%20Diagonal%20Covariance%20Matching/tables/f7349ad500a0545bfa8047458b1bdfc98caeabf319cb1d43e560a1ab01f1884a.jpg)

## BIRD: A Trustworthy Bayesian Inference Framework for Large Language Models


### Images

![1069c0576157e9e3cf5255b0c9f03d47409764f878e6c2c107f1f3cb147af2f5.jpg](../iclr_results/82_BIRD_%20A%20Trustworthy%20Bayesian%20Inference%20Framework%20for%20Large%20Language%20Models/images/1069c0576157e9e3cf5255b0c9f03d47409764f878e6c2c107f1f3cb147af2f5.jpg)

![1f8cd2cd0927dbfb5f23c65989b674f74254e430eba58da7c2401220a3f3f973.jpg](../iclr_results/82_BIRD_%20A%20Trustworthy%20Bayesian%20Inference%20Framework%20for%20Large%20Language%20Models/images/1f8cd2cd0927dbfb5f23c65989b674f74254e430eba58da7c2401220a3f3f973.jpg)

![6022955a614908f2e0e1d3e200a81abf2e7dd3d4402a6a9b8ab07f5c3681220c.jpg](../iclr_results/82_BIRD_%20A%20Trustworthy%20Bayesian%20Inference%20Framework%20for%20Large%20Language%20Models/images/6022955a614908f2e0e1d3e200a81abf2e7dd3d4402a6a9b8ab07f5c3681220c.jpg)

![f247a6d229e9107c96121550d903a43424f90a712936dbc765cd392b823d7252.jpg](../iclr_results/82_BIRD_%20A%20Trustworthy%20Bayesian%20Inference%20Framework%20for%20Large%20Language%20Models/images/f247a6d229e9107c96121550d903a43424f90a712936dbc765cd392b823d7252.jpg)

### Tables

![095a02b75d8226ad016ed0bd11674c3289bd4f3473f53dc63885cf9176bf5ce6.jpg](../iclr_results/82_BIRD_%20A%20Trustworthy%20Bayesian%20Inference%20Framework%20for%20Large%20Language%20Models/tables/095a02b75d8226ad016ed0bd11674c3289bd4f3473f53dc63885cf9176bf5ce6.jpg)

![10783147128c13165be3dbe33978db73bc62d522501562a270dc8f9221c9278e.jpg](../iclr_results/82_BIRD_%20A%20Trustworthy%20Bayesian%20Inference%20Framework%20for%20Large%20Language%20Models/tables/10783147128c13165be3dbe33978db73bc62d522501562a270dc8f9221c9278e.jpg)

![3cdfb0f4c213b594f29e6bd00d4e297cd9e0d8453967795aae156537b39128da.jpg](../iclr_results/82_BIRD_%20A%20Trustworthy%20Bayesian%20Inference%20Framework%20for%20Large%20Language%20Models/tables/3cdfb0f4c213b594f29e6bd00d4e297cd9e0d8453967795aae156537b39128da.jpg)

![3f8f4dcfaf87d17e4a1e58f9d887456f1a77e9cd83af4b1ed7366866adbfc756.jpg](../iclr_results/82_BIRD_%20A%20Trustworthy%20Bayesian%20Inference%20Framework%20for%20Large%20Language%20Models/tables/3f8f4dcfaf87d17e4a1e58f9d887456f1a77e9cd83af4b1ed7366866adbfc756.jpg)

![61d6911f2479a023e3d5f7fa718b0160357e1a0ef61e4f5cfaa85e2c436675c9.jpg](../iclr_results/82_BIRD_%20A%20Trustworthy%20Bayesian%20Inference%20Framework%20for%20Large%20Language%20Models/tables/61d6911f2479a023e3d5f7fa718b0160357e1a0ef61e4f5cfaa85e2c436675c9.jpg)

![7e37aa65d95a5a7682b8373fd5675dc76e703ba84d5c4e057325347a67da043e.jpg](../iclr_results/82_BIRD_%20A%20Trustworthy%20Bayesian%20Inference%20Framework%20for%20Large%20Language%20Models/tables/7e37aa65d95a5a7682b8373fd5675dc76e703ba84d5c4e057325347a67da043e.jpg)

![9d453d386d169f27dfa3de0cdd1607b685c1c085efb8fafe87ea17ab42617b55.jpg](../iclr_results/82_BIRD_%20A%20Trustworthy%20Bayesian%20Inference%20Framework%20for%20Large%20Language%20Models/tables/9d453d386d169f27dfa3de0cdd1607b685c1c085efb8fafe87ea17ab42617b55.jpg)

![b6a604e6c45e3c3867bb990d28fb4c30abb61429d62949555d93b044762b92ed.jpg](../iclr_results/82_BIRD_%20A%20Trustworthy%20Bayesian%20Inference%20Framework%20for%20Large%20Language%20Models/tables/b6a604e6c45e3c3867bb990d28fb4c30abb61429d62949555d93b044762b92ed.jpg)

## Combatting Dimensional Collapse in LLM Pre-Training Data via Submodular File Selection


### Images

![37f2e4bb19628171af9e7823897439fc7fa9ddf9682a38260b1986703c0a1d37.jpg](../iclr_results/83_Combatting%20Dimensional%20Collapse%20in%20LLM%20Pre-Training%20Data%20via%20Submodular%20File%20Selection/images/37f2e4bb19628171af9e7823897439fc7fa9ddf9682a38260b1986703c0a1d37.jpg)

![56915801b8434809f31bd684f13024a21678a0d67c918aba76840558273fadd2.jpg](../iclr_results/83_Combatting%20Dimensional%20Collapse%20in%20LLM%20Pre-Training%20Data%20via%20Submodular%20File%20Selection/images/56915801b8434809f31bd684f13024a21678a0d67c918aba76840558273fadd2.jpg)

![574f540e675af7c2f54356976890cb41acaf3b5b11dd71e7b0b587e50c2c2f60.jpg](../iclr_results/83_Combatting%20Dimensional%20Collapse%20in%20LLM%20Pre-Training%20Data%20via%20Submodular%20File%20Selection/images/574f540e675af7c2f54356976890cb41acaf3b5b11dd71e7b0b587e50c2c2f60.jpg)

![636d37ef2301a166b1741fd7353c6394cb18f169b391086a95a9dfb919e1e3a7.jpg](../iclr_results/83_Combatting%20Dimensional%20Collapse%20in%20LLM%20Pre-Training%20Data%20via%20Submodular%20File%20Selection/images/636d37ef2301a166b1741fd7353c6394cb18f169b391086a95a9dfb919e1e3a7.jpg)

![67bc309e78474583068fb346e82635d664fa3bc62b2b725da9c7823930aecbd8.jpg](../iclr_results/83_Combatting%20Dimensional%20Collapse%20in%20LLM%20Pre-Training%20Data%20via%20Submodular%20File%20Selection/images/67bc309e78474583068fb346e82635d664fa3bc62b2b725da9c7823930aecbd8.jpg)

![98bbc7ece76c91082fd268a9d03d6ce4866c03082268b4eb5c5cb866656e83ac.jpg](../iclr_results/83_Combatting%20Dimensional%20Collapse%20in%20LLM%20Pre-Training%20Data%20via%20Submodular%20File%20Selection/images/98bbc7ece76c91082fd268a9d03d6ce4866c03082268b4eb5c5cb866656e83ac.jpg)

![bab57a7151adf2f9e75babe4558e146ca1473e5e319b8f313e69fef50e435f40.jpg](../iclr_results/83_Combatting%20Dimensional%20Collapse%20in%20LLM%20Pre-Training%20Data%20via%20Submodular%20File%20Selection/images/bab57a7151adf2f9e75babe4558e146ca1473e5e319b8f313e69fef50e435f40.jpg)

![ceb319d5e9ea503afbaf4633d592dd2d795cc55a346aa527ac19cfd18639d2d1.jpg](../iclr_results/83_Combatting%20Dimensional%20Collapse%20in%20LLM%20Pre-Training%20Data%20via%20Submodular%20File%20Selection/images/ceb319d5e9ea503afbaf4633d592dd2d795cc55a346aa527ac19cfd18639d2d1.jpg)

### Tables

![175c876501699dbc7a7945047df7fc7daeadf183948a13504944aba9337cc9e1.jpg](../iclr_results/83_Combatting%20Dimensional%20Collapse%20in%20LLM%20Pre-Training%20Data%20via%20Submodular%20File%20Selection/tables/175c876501699dbc7a7945047df7fc7daeadf183948a13504944aba9337cc9e1.jpg)

![18d55a94401f3fdee2cf674301648bda6b4753258a87650b922dda82a536c4c7.jpg](../iclr_results/83_Combatting%20Dimensional%20Collapse%20in%20LLM%20Pre-Training%20Data%20via%20Submodular%20File%20Selection/tables/18d55a94401f3fdee2cf674301648bda6b4753258a87650b922dda82a536c4c7.jpg)

![1ac4055602c4772e5175e9e1f330092e67be4916d744e3110ead7095cb5afd9b.jpg](../iclr_results/83_Combatting%20Dimensional%20Collapse%20in%20LLM%20Pre-Training%20Data%20via%20Submodular%20File%20Selection/tables/1ac4055602c4772e5175e9e1f330092e67be4916d744e3110ead7095cb5afd9b.jpg)

![26c3f402f3061818f356d851b6da3320289b6b81e594fc02fa05272bddf55768.jpg](../iclr_results/83_Combatting%20Dimensional%20Collapse%20in%20LLM%20Pre-Training%20Data%20via%20Submodular%20File%20Selection/tables/26c3f402f3061818f356d851b6da3320289b6b81e594fc02fa05272bddf55768.jpg)

![4ad2add96d0ecfd9e735de708d8d2e35469fa99ee729fd09c4f319e236b00506.jpg](../iclr_results/83_Combatting%20Dimensional%20Collapse%20in%20LLM%20Pre-Training%20Data%20via%20Submodular%20File%20Selection/tables/4ad2add96d0ecfd9e735de708d8d2e35469fa99ee729fd09c4f319e236b00506.jpg)

![822f9269bc42ba70ae860cd4e05eae73e5ccdfe0c986eb25d18686e2fb4888fc.jpg](../iclr_results/83_Combatting%20Dimensional%20Collapse%20in%20LLM%20Pre-Training%20Data%20via%20Submodular%20File%20Selection/tables/822f9269bc42ba70ae860cd4e05eae73e5ccdfe0c986eb25d18686e2fb4888fc.jpg)

![960f532531d74f1293ce5f7924271697d313872086b2b9eaa4620b7a349df321.jpg](../iclr_results/83_Combatting%20Dimensional%20Collapse%20in%20LLM%20Pre-Training%20Data%20via%20Submodular%20File%20Selection/tables/960f532531d74f1293ce5f7924271697d313872086b2b9eaa4620b7a349df321.jpg)

![a3cbee6295d75a045a1150fcbd388ce934d7a07dac35116b807f8843f3dfcc48.jpg](../iclr_results/83_Combatting%20Dimensional%20Collapse%20in%20LLM%20Pre-Training%20Data%20via%20Submodular%20File%20Selection/tables/a3cbee6295d75a045a1150fcbd388ce934d7a07dac35116b807f8843f3dfcc48.jpg)

![b3852f74cb0db66a1a60a7de7a27f6795e912359e6818b43995c02ece6decc03.jpg](../iclr_results/83_Combatting%20Dimensional%20Collapse%20in%20LLM%20Pre-Training%20Data%20via%20Submodular%20File%20Selection/tables/b3852f74cb0db66a1a60a7de7a27f6795e912359e6818b43995c02ece6decc03.jpg)

![c2d6dc175fc6ed2fa9a5792c1219e0a5df0135d8f2b0637059800a09c40ee8aa.jpg](../iclr_results/83_Combatting%20Dimensional%20Collapse%20in%20LLM%20Pre-Training%20Data%20via%20Submodular%20File%20Selection/tables/c2d6dc175fc6ed2fa9a5792c1219e0a5df0135d8f2b0637059800a09c40ee8aa.jpg)

![c8736aab5995d480fc13f8b0c76c1275147561b4dc4462dab7f1e7697b98c0e6.jpg](../iclr_results/83_Combatting%20Dimensional%20Collapse%20in%20LLM%20Pre-Training%20Data%20via%20Submodular%20File%20Selection/tables/c8736aab5995d480fc13f8b0c76c1275147561b4dc4462dab7f1e7697b98c0e6.jpg)

![c899d79c809ca7372cd884e37058ba9bb99fc8544f191cd563d38bed6a86c619.jpg](../iclr_results/83_Combatting%20Dimensional%20Collapse%20in%20LLM%20Pre-Training%20Data%20via%20Submodular%20File%20Selection/tables/c899d79c809ca7372cd884e37058ba9bb99fc8544f191cd563d38bed6a86c619.jpg)

![d3126f61c2059a88e9824b249417d494b40c27f650f6a49cf8635be66eaee0d2.jpg](../iclr_results/83_Combatting%20Dimensional%20Collapse%20in%20LLM%20Pre-Training%20Data%20via%20Submodular%20File%20Selection/tables/d3126f61c2059a88e9824b249417d494b40c27f650f6a49cf8635be66eaee0d2.jpg)

![de091e71af591007eaf63051c802a0c84c3c2f2481070a46722436fc823b55a2.jpg](../iclr_results/83_Combatting%20Dimensional%20Collapse%20in%20LLM%20Pre-Training%20Data%20via%20Submodular%20File%20Selection/tables/de091e71af591007eaf63051c802a0c84c3c2f2481070a46722436fc823b55a2.jpg)

![fa1fe9ea447eb027cca67c3bb8cf288fb202cc71fd413f1a6a8b665b1caff0a5.jpg](../iclr_results/83_Combatting%20Dimensional%20Collapse%20in%20LLM%20Pre-Training%20Data%20via%20Submodular%20File%20Selection/tables/fa1fe9ea447eb027cca67c3bb8cf288fb202cc71fd413f1a6a8b665b1caff0a5.jpg)

## Influence Functions for Scalable Data Attribution in Diffusion Models


### Images

![0921887701a084a843b6e9b3e3c3c3f300e6b7c87fd38e09b0b0c44984ce893b.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/0921887701a084a843b6e9b3e3c3c3f300e6b7c87fd38e09b0b0c44984ce893b.jpg)

![0c9e00fa6a9fa5b60bc93e449c0582e491b015b7446c641cae8074dbe9e8bfe3.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/0c9e00fa6a9fa5b60bc93e449c0582e491b015b7446c641cae8074dbe9e8bfe3.jpg)

![26b6bd2333ad45c6a08cb77a27519d8cbd4d947003f7f9c07e49673d1f7cc180.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/26b6bd2333ad45c6a08cb77a27519d8cbd4d947003f7f9c07e49673d1f7cc180.jpg)

![2823141267db4756d2d8a5d7573b1d9bf358eadc7d1d9ecb9ba569f086344bc7.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/2823141267db4756d2d8a5d7573b1d9bf358eadc7d1d9ecb9ba569f086344bc7.jpg)

![32aadc4acdd36c06b3cb28ac0fc97887bf2cdd3d471f5fcc51de7720b23b526b.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/32aadc4acdd36c06b3cb28ac0fc97887bf2cdd3d471f5fcc51de7720b23b526b.jpg)

![3cdd4ac6171da6c7d0122e3f9542be481cf89e8c4e902f8d1627d6a4b9dc3b94.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/3cdd4ac6171da6c7d0122e3f9542be481cf89e8c4e902f8d1627d6a4b9dc3b94.jpg)

![3f9be704fa6c6f68bed4bd30eefd719057d94e58a6b24616f6a0c0fe67748258.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/3f9be704fa6c6f68bed4bd30eefd719057d94e58a6b24616f6a0c0fe67748258.jpg)

![405723417a9b977c66057e06941edfee94102e80c66902794cb6ef8677a52b00.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/405723417a9b977c66057e06941edfee94102e80c66902794cb6ef8677a52b00.jpg)

![581cd6fdf9081a8e3621825d8f86b6aa8926e95213303ffd041ca29d89a81bc3.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/581cd6fdf9081a8e3621825d8f86b6aa8926e95213303ffd041ca29d89a81bc3.jpg)

![76ecea97365863352bce508ffab2b8281a57448bb1904c2b70e1fd9a0b53dc54.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/76ecea97365863352bce508ffab2b8281a57448bb1904c2b70e1fd9a0b53dc54.jpg)

![7a9498e12df4a34ac92e1ee1d190d70917bd07bb98abe33bc9ec32dfbe85fb43.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/7a9498e12df4a34ac92e1ee1d190d70917bd07bb98abe33bc9ec32dfbe85fb43.jpg)

![7f3c56be49437dd7a3afdb28832a9b570a02e72d3b53e1ac4e08acccdd43a472.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/7f3c56be49437dd7a3afdb28832a9b570a02e72d3b53e1ac4e08acccdd43a472.jpg)

![81b74e716f2fc22a76bd6691f3357cd0ce64d345582b3302c361cde63a45a689.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/81b74e716f2fc22a76bd6691f3357cd0ce64d345582b3302c361cde63a45a689.jpg)

![8340daeb8e01ecc7af061f42f42a5b01fe957fe234ba4d3d30f543c5a6039a35.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/8340daeb8e01ecc7af061f42f42a5b01fe957fe234ba4d3d30f543c5a6039a35.jpg)

![84abad53bfc882251ca47ef4c09aeba36d93d6334b620b014b018d2807adbab7.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/84abad53bfc882251ca47ef4c09aeba36d93d6334b620b014b018d2807adbab7.jpg)

![8d218471a6e009b08678033a87faff36984ced2594fdea0f290395085244d69d.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/8d218471a6e009b08678033a87faff36984ced2594fdea0f290395085244d69d.jpg)

![a13f88db045a29b88d38b5749f8fd9f9283f22632331867e7c045fd88c9b827f.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/a13f88db045a29b88d38b5749f8fd9f9283f22632331867e7c045fd88c9b827f.jpg)

![c1e1a9ab62d0d0ebddca5c0943d0d2161d2ea9bda72ff8217da01bf07eee9904.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/c1e1a9ab62d0d0ebddca5c0943d0d2161d2ea9bda72ff8217da01bf07eee9904.jpg)

![c5a878542595ed5ed229027156ef53ac4fdb62658746d747e07cbd647afd7cd7.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/c5a878542595ed5ed229027156ef53ac4fdb62658746d747e07cbd647afd7cd7.jpg)

![c6e0ef400f6cb7e52457c6748bee0cb00595f9e5753415b4c74223e76ab75686.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/c6e0ef400f6cb7e52457c6748bee0cb00595f9e5753415b4c74223e76ab75686.jpg)

![cd0bc71807cfad6d4e9f1048bac74d0b4f25501b085fc1038e6092adca4744ac.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/cd0bc71807cfad6d4e9f1048bac74d0b4f25501b085fc1038e6092adca4744ac.jpg)

![daa0adcdeb30de8d19ef53491971f3f5913d42be3c29e0e52ef590f84992f2e6.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/daa0adcdeb30de8d19ef53491971f3f5913d42be3c29e0e52ef590f84992f2e6.jpg)

![dfd12deed3fc6cad5de4bede07416cfb74ccb2e69398859d5ddaffddab263c1e.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/dfd12deed3fc6cad5de4bede07416cfb74ccb2e69398859d5ddaffddab263c1e.jpg)

![f6f9db0af75387a034fb927837b7e8c0a34ab73363db442beec50401a7e7977f.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/f6f9db0af75387a034fb927837b7e8c0a34ab73363db442beec50401a7e7977f.jpg)

![ff0b70c942d36323f73980ee258651e0cfbf41188ecbcb5d972e1fbda0591e0f.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/images/ff0b70c942d36323f73980ee258651e0cfbf41188ecbcb5d972e1fbda0591e0f.jpg)

### Tables

![871cd9844a1fe03a127bebf3a91c1ebc232561ceb1e432f911a606aff1440f67.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/tables/871cd9844a1fe03a127bebf3a91c1ebc232561ceb1e432f911a606aff1440f67.jpg)

![9ad9d5933093c53a0ebc0fd4b992b0191a2ce8af528d519287ce7c13137268be.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/tables/9ad9d5933093c53a0ebc0fd4b992b0191a2ce8af528d519287ce7c13137268be.jpg)

![ae12fa16e48a819429825cd7af0f26e6ed0c62e23ac33d95a518aa3829904846.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/tables/ae12fa16e48a819429825cd7af0f26e6ed0c62e23ac33d95a518aa3829904846.jpg)

![bcbcc67055048358841a796921c5d69ecb1dd9fd52c216bcd6ca96980d0ae1e3.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/tables/bcbcc67055048358841a796921c5d69ecb1dd9fd52c216bcd6ca96980d0ae1e3.jpg)

![eac85208174ddcbe4dd9cd7134513ae4cbcd3c9aaab46b744c9a739f9f3eee46.jpg](../iclr_results/84_Influence%20Functions%20for%20Scalable%20Data%20Attribution%20in%20Diffusion%20Models/tables/eac85208174ddcbe4dd9cd7134513ae4cbcd3c9aaab46b744c9a739f9f3eee46.jpg)

## Language Representations Can be What Recommenders Need: Findings and Potentials


### Images

![1bb7bcfa376e8fbbd2bc3f06f9449a403be2cda38d26765fe6c01291d5732bb8.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/images/1bb7bcfa376e8fbbd2bc3f06f9449a403be2cda38d26765fe6c01291d5732bb8.jpg)

![35876925e3c05388b5b5a8eaa77370f239c147a807895c5a97e71f3d0310a97f.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/images/35876925e3c05388b5b5a8eaa77370f239c147a807895c5a97e71f3d0310a97f.jpg)

![39e0b7c57b46d3f1b4d115a6276b8156939e50a312ae5d99b76974bf5128a9a3.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/images/39e0b7c57b46d3f1b4d115a6276b8156939e50a312ae5d99b76974bf5128a9a3.jpg)

![50e462733b2fb060ff6ce3ee100c5ad157ea171b2965b784ef9c9a84404c093d.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/images/50e462733b2fb060ff6ce3ee100c5ad157ea171b2965b784ef9c9a84404c093d.jpg)

![6429fea79ac176aa36f67bbcdc845d1bd8630e024638a881434ba5ad1d2f7505.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/images/6429fea79ac176aa36f67bbcdc845d1bd8630e024638a881434ba5ad1d2f7505.jpg)

![6f22861d9c1406e2ec89a312c2111318e210293bf605d69366f9a93975ac7094.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/images/6f22861d9c1406e2ec89a312c2111318e210293bf605d69366f9a93975ac7094.jpg)

![6f98b4f015e71347426f423bc3e1146677c40b35f0a7c251a0f2010da2d1f9fe.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/images/6f98b4f015e71347426f423bc3e1146677c40b35f0a7c251a0f2010da2d1f9fe.jpg)

![94d66f92f4149ba7ab6c229087c6926afff385f081f11fb41337614b5ac26575.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/images/94d66f92f4149ba7ab6c229087c6926afff385f081f11fb41337614b5ac26575.jpg)

![a4ff93172cf4c97e034923c21cfb0b23cc4c2fb9115397f5b36d058cc699dc54.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/images/a4ff93172cf4c97e034923c21cfb0b23cc4c2fb9115397f5b36d058cc699dc54.jpg)

![d316a416a69ca7e71aeb38f0ff215bbbc38138c3377f5cecf06a28a7f4813821.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/images/d316a416a69ca7e71aeb38f0ff215bbbc38138c3377f5cecf06a28a7f4813821.jpg)

### Tables

![114e36e47ed7a8a2137360413ecdc0ab3c298275adafa6fbe6abc793bee368a1.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/tables/114e36e47ed7a8a2137360413ecdc0ab3c298275adafa6fbe6abc793bee368a1.jpg)

![160c96e052b81688d26c1dd39e6d64ef7aa2fb20e871d521d68c9c1bade392f8.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/tables/160c96e052b81688d26c1dd39e6d64ef7aa2fb20e871d521d68c9c1bade392f8.jpg)

![1fb394536e6429b221c6cf1c801d2a43f46146ff74846e5167fd71bf9228aab1.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/tables/1fb394536e6429b221c6cf1c801d2a43f46146ff74846e5167fd71bf9228aab1.jpg)

![24532c787f9d2ef2bca9350114d377d4b6722da9c3c4bc237d967c43ef49f4c0.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/tables/24532c787f9d2ef2bca9350114d377d4b6722da9c3c4bc237d967c43ef49f4c0.jpg)

![2bf0a0e0e04dc73dd3f488e1f4795c732752e3b22581c3228afdb24b7e73ec17.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/tables/2bf0a0e0e04dc73dd3f488e1f4795c732752e3b22581c3228afdb24b7e73ec17.jpg)

![59682f7c2655279e85a6387d5643db0432428c6ee600f4312cb070d0fabcf9e3.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/tables/59682f7c2655279e85a6387d5643db0432428c6ee600f4312cb070d0fabcf9e3.jpg)

![5e2c6f82fdaf39b23f4ad3b8ab1db9da2af48394f9ec721c75932c73ab2b6752.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/tables/5e2c6f82fdaf39b23f4ad3b8ab1db9da2af48394f9ec721c75932c73ab2b6752.jpg)

![6875e28da69db450f5a8ab9e60743ac5d4d1d634b28be81368a32ddb175c8a32.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/tables/6875e28da69db450f5a8ab9e60743ac5d4d1d634b28be81368a32ddb175c8a32.jpg)

![6f21b3bc6ad3cc23975f947e65958763ab801d4bb3e834fa9285194d1b48d932.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/tables/6f21b3bc6ad3cc23975f947e65958763ab801d4bb3e834fa9285194d1b48d932.jpg)

![9a272f7d249f4d1da20f69e2fd97ffd0fcdbb8ee278ea6aeee8807347869b045.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/tables/9a272f7d249f4d1da20f69e2fd97ffd0fcdbb8ee278ea6aeee8807347869b045.jpg)

![aa12e600c19993d05edd4f4bfbdebdb14959fa45049d1a67b95ed2432dd15255.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/tables/aa12e600c19993d05edd4f4bfbdebdb14959fa45049d1a67b95ed2432dd15255.jpg)

![b2b6bee478b2aeb867515a7075f43138fb1ba6cc93bf435a4e6a67d788752b9f.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/tables/b2b6bee478b2aeb867515a7075f43138fb1ba6cc93bf435a4e6a67d788752b9f.jpg)

![c1a553e88289845f1b6314660dc848b1bfbdb20d74e2599672c802928edfa131.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/tables/c1a553e88289845f1b6314660dc848b1bfbdb20d74e2599672c802928edfa131.jpg)

![d4643573df6928897c925ce1ecb48e6bb203847f771367bbd2315892e4e22669.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/tables/d4643573df6928897c925ce1ecb48e6bb203847f771367bbd2315892e4e22669.jpg)

![dc1eef4ed3046999738aa787d95f7e4a29f0bfdc42edaba74dd78e45bdabb180.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/tables/dc1eef4ed3046999738aa787d95f7e4a29f0bfdc42edaba74dd78e45bdabb180.jpg)

![f76d837f2ec494f509e9feef5eba4899b658c998d5c93a4f02f86f3c634ebfc4.jpg](../iclr_results/85_Language%20Representations%20Can%20be%20What%20Recommenders%20Need_%20Findings%20and%20Potentials/tables/f76d837f2ec494f509e9feef5eba4899b658c998d5c93a4f02f86f3c634ebfc4.jpg)

## Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition


### Images

![1aceeca04625b0feb7465181f7922510a7f11fc0596f98407961b63fb37c8429.jpg](../iclr_results/86_Knowledge%20Entropy%20Decay%20during%20Language%20Model%20Pretraining%20Hinders%20New%20Knowledge%20Acquisition/images/1aceeca04625b0feb7465181f7922510a7f11fc0596f98407961b63fb37c8429.jpg)

![26b2cc1a1e4b637bbe9d3bf361ab96f136a5f28888a36b9751fc312b6aab3c75.jpg](../iclr_results/86_Knowledge%20Entropy%20Decay%20during%20Language%20Model%20Pretraining%20Hinders%20New%20Knowledge%20Acquisition/images/26b2cc1a1e4b637bbe9d3bf361ab96f136a5f28888a36b9751fc312b6aab3c75.jpg)

![41962cf15f9789f204eb73ac5014ff78108ada0066309ff7f868f9758c129a22.jpg](../iclr_results/86_Knowledge%20Entropy%20Decay%20during%20Language%20Model%20Pretraining%20Hinders%20New%20Knowledge%20Acquisition/images/41962cf15f9789f204eb73ac5014ff78108ada0066309ff7f868f9758c129a22.jpg)

![437b430ac5b63bd6dd130505837062ecf493f76d4cab1d526b6e8a05da068a97.jpg](../iclr_results/86_Knowledge%20Entropy%20Decay%20during%20Language%20Model%20Pretraining%20Hinders%20New%20Knowledge%20Acquisition/images/437b430ac5b63bd6dd130505837062ecf493f76d4cab1d526b6e8a05da068a97.jpg)

![46e3c760e7561259cc57cae6a338cba25de062204fbea5c85bae25834c14ef95.jpg](../iclr_results/86_Knowledge%20Entropy%20Decay%20during%20Language%20Model%20Pretraining%20Hinders%20New%20Knowledge%20Acquisition/images/46e3c760e7561259cc57cae6a338cba25de062204fbea5c85bae25834c14ef95.jpg)

![4d900e80391652445d8e0c218df9f9fa2a6a54436ed16f313dfef738ceefc974.jpg](../iclr_results/86_Knowledge%20Entropy%20Decay%20during%20Language%20Model%20Pretraining%20Hinders%20New%20Knowledge%20Acquisition/images/4d900e80391652445d8e0c218df9f9fa2a6a54436ed16f313dfef738ceefc974.jpg)

![4dde04a93e8f7d14291240cac270777df096d07a059d14e8c745edf322f62e5b.jpg](../iclr_results/86_Knowledge%20Entropy%20Decay%20during%20Language%20Model%20Pretraining%20Hinders%20New%20Knowledge%20Acquisition/images/4dde04a93e8f7d14291240cac270777df096d07a059d14e8c745edf322f62e5b.jpg)

![65846977f86b5e5081577f48c1ae26a21d69f1bc54271d95ed0a411f47804643.jpg](../iclr_results/86_Knowledge%20Entropy%20Decay%20during%20Language%20Model%20Pretraining%20Hinders%20New%20Knowledge%20Acquisition/images/65846977f86b5e5081577f48c1ae26a21d69f1bc54271d95ed0a411f47804643.jpg)

![97596a9ba656878991c2aec085b0cc59369e0d6387edd76355408af033b6d082.jpg](../iclr_results/86_Knowledge%20Entropy%20Decay%20during%20Language%20Model%20Pretraining%20Hinders%20New%20Knowledge%20Acquisition/images/97596a9ba656878991c2aec085b0cc59369e0d6387edd76355408af033b6d082.jpg)

![9acd63cc498aa303830a04a7dddf653ded0511419bde5e31941dbe2041bc760e.jpg](../iclr_results/86_Knowledge%20Entropy%20Decay%20during%20Language%20Model%20Pretraining%20Hinders%20New%20Knowledge%20Acquisition/images/9acd63cc498aa303830a04a7dddf653ded0511419bde5e31941dbe2041bc760e.jpg)

![afd8afa6fd025e717ae6f2c1fca3e75197649f9e6f5f9cb248a53a3c1581a25a.jpg](../iclr_results/86_Knowledge%20Entropy%20Decay%20during%20Language%20Model%20Pretraining%20Hinders%20New%20Knowledge%20Acquisition/images/afd8afa6fd025e717ae6f2c1fca3e75197649f9e6f5f9cb248a53a3c1581a25a.jpg)

![ba7620ada78eab95cc1a6e2d74bd7f06df15e3f5a5185967ccd8cdb1ac5657a8.jpg](../iclr_results/86_Knowledge%20Entropy%20Decay%20during%20Language%20Model%20Pretraining%20Hinders%20New%20Knowledge%20Acquisition/images/ba7620ada78eab95cc1a6e2d74bd7f06df15e3f5a5185967ccd8cdb1ac5657a8.jpg)

![c768029cdd9673c273d0372a34dfc2e5e8ea61fb60cd004b9866cdcb3050fe4a.jpg](../iclr_results/86_Knowledge%20Entropy%20Decay%20during%20Language%20Model%20Pretraining%20Hinders%20New%20Knowledge%20Acquisition/images/c768029cdd9673c273d0372a34dfc2e5e8ea61fb60cd004b9866cdcb3050fe4a.jpg)

![c99c7ddb3a55d5deed28ce798806bed4cd47100c818d685f62a5c7a249eef6d0.jpg](../iclr_results/86_Knowledge%20Entropy%20Decay%20during%20Language%20Model%20Pretraining%20Hinders%20New%20Knowledge%20Acquisition/images/c99c7ddb3a55d5deed28ce798806bed4cd47100c818d685f62a5c7a249eef6d0.jpg)

![df9969cbe9ca5ca6761dbe0053bc830ce33bbab355e2f11893817df8ca62b481.jpg](../iclr_results/86_Knowledge%20Entropy%20Decay%20during%20Language%20Model%20Pretraining%20Hinders%20New%20Knowledge%20Acquisition/images/df9969cbe9ca5ca6761dbe0053bc830ce33bbab355e2f11893817df8ca62b481.jpg)

![e376e7b2e505ed8a89b048e77abd541552b0ca664888120cc9f91859c951b075.jpg](../iclr_results/86_Knowledge%20Entropy%20Decay%20during%20Language%20Model%20Pretraining%20Hinders%20New%20Knowledge%20Acquisition/images/e376e7b2e505ed8a89b048e77abd541552b0ca664888120cc9f91859c951b075.jpg)

![f40aef26a969c1f56b9917afa2ac9f7957f151100b91df968142c3bc22480663.jpg](../iclr_results/86_Knowledge%20Entropy%20Decay%20during%20Language%20Model%20Pretraining%20Hinders%20New%20Knowledge%20Acquisition/images/f40aef26a969c1f56b9917afa2ac9f7957f151100b91df968142c3bc22480663.jpg)

![f63d9cf3e20994b3159b9619d7c76f82e363fbd54326a34f85419ccbb376223f.jpg](../iclr_results/86_Knowledge%20Entropy%20Decay%20during%20Language%20Model%20Pretraining%20Hinders%20New%20Knowledge%20Acquisition/images/f63d9cf3e20994b3159b9619d7c76f82e363fbd54326a34f85419ccbb376223f.jpg)

### Tables

![7638b2ea5784449a12681f79df511c389904a842ca954f80898e457600bfd668.jpg](../iclr_results/86_Knowledge%20Entropy%20Decay%20during%20Language%20Model%20Pretraining%20Hinders%20New%20Knowledge%20Acquisition/tables/7638b2ea5784449a12681f79df511c389904a842ca954f80898e457600bfd668.jpg)

![c55028e4151c07423bdd10a1162426568932c2d9cd8cc1c115e4d27e28d8b65b.jpg](../iclr_results/86_Knowledge%20Entropy%20Decay%20during%20Language%20Model%20Pretraining%20Hinders%20New%20Knowledge%20Acquisition/tables/c55028e4151c07423bdd10a1162426568932c2d9cd8cc1c115e4d27e28d8b65b.jpg)

## Your Mixture-of-Experts LLM Is Secretly an Embedding Model for Free


### Images

![24279aad7876b3f09bb2b95fed6f69fd524b2b085aceb964e9b3978e941f3879.jpg](../iclr_results/87_Your%20Mixture-of-Experts%20LLM%20Is%20Secretly%20an%20Embedding%20Model%20for%20Free/images/24279aad7876b3f09bb2b95fed6f69fd524b2b085aceb964e9b3978e941f3879.jpg)

![a1410343ea54aa20ffea75ce3506d7259d4755b8aca8410ec70115078d2befa7.jpg](../iclr_results/87_Your%20Mixture-of-Experts%20LLM%20Is%20Secretly%20an%20Embedding%20Model%20for%20Free/images/a1410343ea54aa20ffea75ce3506d7259d4755b8aca8410ec70115078d2befa7.jpg)

![bb88c88a9dd3584b67c1dc5d0cdb469949e87f0b06487ce4fd5cb78afb65c9ce.jpg](../iclr_results/87_Your%20Mixture-of-Experts%20LLM%20Is%20Secretly%20an%20Embedding%20Model%20for%20Free/images/bb88c88a9dd3584b67c1dc5d0cdb469949e87f0b06487ce4fd5cb78afb65c9ce.jpg)

![bff23001853fd5101b33da7b801e8c0c453661bde65bc60d04680929471adc19.jpg](../iclr_results/87_Your%20Mixture-of-Experts%20LLM%20Is%20Secretly%20an%20Embedding%20Model%20for%20Free/images/bff23001853fd5101b33da7b801e8c0c453661bde65bc60d04680929471adc19.jpg)

![e1256d5cc43c494eaf0607db9e144312cf6aeed79fe4f2f6649ad93866adbb51.jpg](../iclr_results/87_Your%20Mixture-of-Experts%20LLM%20Is%20Secretly%20an%20Embedding%20Model%20for%20Free/images/e1256d5cc43c494eaf0607db9e144312cf6aeed79fe4f2f6649ad93866adbb51.jpg)

![f102eb4111a2c218303cd6beae1773f1a0878d60835c6bb29b8a66fe160f6e4a.jpg](../iclr_results/87_Your%20Mixture-of-Experts%20LLM%20Is%20Secretly%20an%20Embedding%20Model%20for%20Free/images/f102eb4111a2c218303cd6beae1773f1a0878d60835c6bb29b8a66fe160f6e4a.jpg)

### Tables

![35ec252ae830a21ce233b5a147a52187c89695df2c402d30526904601c892fc5.jpg](../iclr_results/87_Your%20Mixture-of-Experts%20LLM%20Is%20Secretly%20an%20Embedding%20Model%20for%20Free/tables/35ec252ae830a21ce233b5a147a52187c89695df2c402d30526904601c892fc5.jpg)

![4f238d9a8359b6bf4b205bb0c8124d46fbf8ef3def54e1a3ec64b20fd563b202.jpg](../iclr_results/87_Your%20Mixture-of-Experts%20LLM%20Is%20Secretly%20an%20Embedding%20Model%20for%20Free/tables/4f238d9a8359b6bf4b205bb0c8124d46fbf8ef3def54e1a3ec64b20fd563b202.jpg)

![6c53ccfc61d00d3387bb17cee2ad1e7b91f0fab71793a203fb4e5c690baf35e5.jpg](../iclr_results/87_Your%20Mixture-of-Experts%20LLM%20Is%20Secretly%20an%20Embedding%20Model%20for%20Free/tables/6c53ccfc61d00d3387bb17cee2ad1e7b91f0fab71793a203fb4e5c690baf35e5.jpg)

![849e7dae5cdee4d579ab39491027b6e4169d0bc6034f1927f5db83f1a3276724.jpg](../iclr_results/87_Your%20Mixture-of-Experts%20LLM%20Is%20Secretly%20an%20Embedding%20Model%20for%20Free/tables/849e7dae5cdee4d579ab39491027b6e4169d0bc6034f1927f5db83f1a3276724.jpg)

![9ef77634182f3c4e0e48593f5e3f6a021e284daea5d294a9973b6aba120bb547.jpg](../iclr_results/87_Your%20Mixture-of-Experts%20LLM%20Is%20Secretly%20an%20Embedding%20Model%20for%20Free/tables/9ef77634182f3c4e0e48593f5e3f6a021e284daea5d294a9973b6aba120bb547.jpg)

![aa3cca5d2e1d3dc3c4aadb5c354e7ac9e75d92479c6c7bf381f606720ad9f369.jpg](../iclr_results/87_Your%20Mixture-of-Experts%20LLM%20Is%20Secretly%20an%20Embedding%20Model%20for%20Free/tables/aa3cca5d2e1d3dc3c4aadb5c354e7ac9e75d92479c6c7bf381f606720ad9f369.jpg)

![ab7f610362b069240be0ba141541a9b744d925f5def1252a680d357c2db117eb.jpg](../iclr_results/87_Your%20Mixture-of-Experts%20LLM%20Is%20Secretly%20an%20Embedding%20Model%20for%20Free/tables/ab7f610362b069240be0ba141541a9b744d925f5def1252a680d357c2db117eb.jpg)

![b02491331c3833e1ac33719268aad57be719827b3b7388f2ff5b08247d213430.jpg](../iclr_results/87_Your%20Mixture-of-Experts%20LLM%20Is%20Secretly%20an%20Embedding%20Model%20for%20Free/tables/b02491331c3833e1ac33719268aad57be719827b3b7388f2ff5b08247d213430.jpg)

![b70a4bc8e388c295dc1799ddc110a387bed876e340bae270716a065b6b48faaf.jpg](../iclr_results/87_Your%20Mixture-of-Experts%20LLM%20Is%20Secretly%20an%20Embedding%20Model%20for%20Free/tables/b70a4bc8e388c295dc1799ddc110a387bed876e340bae270716a065b6b48faaf.jpg)

![ee3f13d69c17d902d34e39e4ac29167f4e001ed4599a35673f3313bd30000635.jpg](../iclr_results/87_Your%20Mixture-of-Experts%20LLM%20Is%20Secretly%20an%20Embedding%20Model%20for%20Free/tables/ee3f13d69c17d902d34e39e4ac29167f4e001ed4599a35673f3313bd30000635.jpg)

![ff8c420442b703999ffeb8aea11cfd77fc2f762447a0abcef8c4d76218f1cdb5.jpg](../iclr_results/87_Your%20Mixture-of-Experts%20LLM%20Is%20Secretly%20an%20Embedding%20Model%20for%20Free/tables/ff8c420442b703999ffeb8aea11cfd77fc2f762447a0abcef8c4d76218f1cdb5.jpg)

## Emergence of meta-stable clustering in mean-field transformer models


### Images

![021c044c8ca9ba28f448f28c3448c9f2654a75e6131acc4ff8b0e7571f5c3038.jpg](../iclr_results/88_Emergence%20of%20meta-stable%20clustering%20in%20mean-field%20transformer%20models/images/021c044c8ca9ba28f448f28c3448c9f2654a75e6131acc4ff8b0e7571f5c3038.jpg)

![16b831706ded63ba5041d649e67e7964a58e5d159a18f0762d6455670a068bc3.jpg](../iclr_results/88_Emergence%20of%20meta-stable%20clustering%20in%20mean-field%20transformer%20models/images/16b831706ded63ba5041d649e67e7964a58e5d159a18f0762d6455670a068bc3.jpg)

![1e7d263c331e501ccb96bc7ec31b40b45635b09ab698ae5ba75ba5f4b6ec8fad.jpg](../iclr_results/88_Emergence%20of%20meta-stable%20clustering%20in%20mean-field%20transformer%20models/images/1e7d263c331e501ccb96bc7ec31b40b45635b09ab698ae5ba75ba5f4b6ec8fad.jpg)

![4434929df2ac1081137cd991cd15fdb2309efcd8c400ac8cd83828341939bfdd.jpg](../iclr_results/88_Emergence%20of%20meta-stable%20clustering%20in%20mean-field%20transformer%20models/images/4434929df2ac1081137cd991cd15fdb2309efcd8c400ac8cd83828341939bfdd.jpg)

![46ccf863dd9a35a9351b512dfe04a721490f89f0fe69727a79b65eb67314d2a5.jpg](../iclr_results/88_Emergence%20of%20meta-stable%20clustering%20in%20mean-field%20transformer%20models/images/46ccf863dd9a35a9351b512dfe04a721490f89f0fe69727a79b65eb67314d2a5.jpg)

![f0500425f06c30dd8b125a1a89b42fca99c20cc05686b3e40499f7bee49b7714.jpg](../iclr_results/88_Emergence%20of%20meta-stable%20clustering%20in%20mean-field%20transformer%20models/images/f0500425f06c30dd8b125a1a89b42fca99c20cc05686b3e40499f7bee49b7714.jpg)

## Data Selection via Optimal Control for Language Models


### Images

![19e15d8163c474d086c1eddcbbee093ece5dd58eec112441c8f67e42f6b58008.jpg](../iclr_results/89_Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/images/19e15d8163c474d086c1eddcbbee093ece5dd58eec112441c8f67e42f6b58008.jpg)

![52968c727c9f54c75f551f256136d9b7ccce6cde6b7aea8a9484047ca7cc0912.jpg](../iclr_results/89_Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/images/52968c727c9f54c75f551f256136d9b7ccce6cde6b7aea8a9484047ca7cc0912.jpg)

![678972a2c9872acc2a00228a36603e13bea3c209fad2b96cf57feff74d2114e7.jpg](../iclr_results/89_Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/images/678972a2c9872acc2a00228a36603e13bea3c209fad2b96cf57feff74d2114e7.jpg)

![68f8150387c1143919c8ccbd33206281fb8e40bc4fbdedeb30e141686d376c0b.jpg](../iclr_results/89_Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/images/68f8150387c1143919c8ccbd33206281fb8e40bc4fbdedeb30e141686d376c0b.jpg)

![7fc74ae3933ce62191060414675119f29b53f634f648d9ded75135ed605c167c.jpg](../iclr_results/89_Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/images/7fc74ae3933ce62191060414675119f29b53f634f648d9ded75135ed605c167c.jpg)

![915dfd0f2b496540dbe1e9cf4d1bfbb03f735444da244539f324fa003ae481b2.jpg](../iclr_results/89_Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/images/915dfd0f2b496540dbe1e9cf4d1bfbb03f735444da244539f324fa003ae481b2.jpg)

![b0b5b7fd15b61f05df9dcffd389c12ed22a7ace4069fac5373157809adba043f.jpg](../iclr_results/89_Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/images/b0b5b7fd15b61f05df9dcffd389c12ed22a7ace4069fac5373157809adba043f.jpg)

![ca439ef3b2addfe7a4b843285e1bc863b33e236bdae0161d9adbb3f3e2b2e962.jpg](../iclr_results/89_Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/images/ca439ef3b2addfe7a4b843285e1bc863b33e236bdae0161d9adbb3f3e2b2e962.jpg)

![e365398978ab15f7f4f2e40eae6f5310741b2fd00de258d4c16d71468d54b2b9.jpg](../iclr_results/89_Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/images/e365398978ab15f7f4f2e40eae6f5310741b2fd00de258d4c16d71468d54b2b9.jpg)

![f8eea761982d19f0b8c6848c7d99ea5adf0d6392170b35386e7361de7dbbc573.jpg](../iclr_results/89_Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/images/f8eea761982d19f0b8c6848c7d99ea5adf0d6392170b35386e7361de7dbbc573.jpg)

### Tables

![1cb44c05bd57983bf8ccb9d5becfc34ba29d98dff0291c23422dc129548f5fd8.jpg](../iclr_results/89_Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/tables/1cb44c05bd57983bf8ccb9d5becfc34ba29d98dff0291c23422dc129548f5fd8.jpg)

![1cf6bcaa9c48e3b1a50735575e3da5ed2331604f8011fd629e41cf1cc12429cb.jpg](../iclr_results/89_Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/tables/1cf6bcaa9c48e3b1a50735575e3da5ed2331604f8011fd629e41cf1cc12429cb.jpg)

![1f46929a9f47a25f99bae9f2c8b9f0e7a910867d3c728cafece587de95da2d39.jpg](../iclr_results/89_Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/tables/1f46929a9f47a25f99bae9f2c8b9f0e7a910867d3c728cafece587de95da2d39.jpg)

![38a9a537274a45a25feb18c697f265a4ce05d62149261fb375278ef6469bfe56.jpg](../iclr_results/89_Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/tables/38a9a537274a45a25feb18c697f265a4ce05d62149261fb375278ef6469bfe56.jpg)

![47b76c90d156248e126c416e33583fccf307e618d52a05fd97c0f462ecbd6608.jpg](../iclr_results/89_Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/tables/47b76c90d156248e126c416e33583fccf307e618d52a05fd97c0f462ecbd6608.jpg)

![6dc3ad1c9c9e270eba8b2fb1372023833bb418371699afd2f164d42d4d180f77.jpg](../iclr_results/89_Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/tables/6dc3ad1c9c9e270eba8b2fb1372023833bb418371699afd2f164d42d4d180f77.jpg)

![8160997d352ee3b5d908892ff5105d3fbdd0d5e9045988aac4486fed8df944b0.jpg](../iclr_results/89_Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/tables/8160997d352ee3b5d908892ff5105d3fbdd0d5e9045988aac4486fed8df944b0.jpg)

![952e078e51f182eef2b232a37e54c4300c06d70d77bc6e89c73eb32ba50bdcc2.jpg](../iclr_results/89_Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/tables/952e078e51f182eef2b232a37e54c4300c06d70d77bc6e89c73eb32ba50bdcc2.jpg)

![bb2962d40353f213231d6f5a4a131fabbc64f1091a24906dde91b30fc6dfefcb.jpg](../iclr_results/89_Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/tables/bb2962d40353f213231d6f5a4a131fabbc64f1091a24906dde91b30fc6dfefcb.jpg)

![bd2b09a460f0b619ca840d7a2004b377da3eb47cdb749a0cf727c2280bfcf1e5.jpg](../iclr_results/89_Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/tables/bd2b09a460f0b619ca840d7a2004b377da3eb47cdb749a0cf727c2280bfcf1e5.jpg)

![cb1a757c04e14e758c675864570ff3bd96b1e7ec8f231cb25f835ebdc654f3f2.jpg](../iclr_results/89_Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/tables/cb1a757c04e14e758c675864570ff3bd96b1e7ec8f231cb25f835ebdc654f3f2.jpg)

![f2e031b012cb66fc27579cbeaa4d26f4d7c30edc92b3a1a840b7d1e6fb9eba73.jpg](../iclr_results/89_Data%20Selection%20via%20Optimal%20Control%20for%20Language%20Models/tables/f2e031b012cb66fc27579cbeaa4d26f4d7c30edc92b3a1a840b7d1e6fb9eba73.jpg)

## Feedback Favors the Generalization of Neural ODEs


### Images

![141416cd7baf5b741077a2624a6abce57a8280a0f14a9adf595e560a93d753a1.jpg](../iclr_results/90_Feedback%20Favors%20the%20Generalization%20of%20Neural%20ODEs/images/141416cd7baf5b741077a2624a6abce57a8280a0f14a9adf595e560a93d753a1.jpg)

![2b1caf934132116960fec3ae14a1ad7f5b9c86bd5668b9d35b071bc2321aad0b.jpg](../iclr_results/90_Feedback%20Favors%20the%20Generalization%20of%20Neural%20ODEs/images/2b1caf934132116960fec3ae14a1ad7f5b9c86bd5668b9d35b071bc2321aad0b.jpg)

![353ac68c745e29c8df3c9ed93b1086429cedf8cbd7374ecf4b7bbe13c7f04be0.jpg](../iclr_results/90_Feedback%20Favors%20the%20Generalization%20of%20Neural%20ODEs/images/353ac68c745e29c8df3c9ed93b1086429cedf8cbd7374ecf4b7bbe13c7f04be0.jpg)

![3de59be99c06ea5d47326c07100b7261aaf83a040f9618f104978a05eb9490d6.jpg](../iclr_results/90_Feedback%20Favors%20the%20Generalization%20of%20Neural%20ODEs/images/3de59be99c06ea5d47326c07100b7261aaf83a040f9618f104978a05eb9490d6.jpg)

![4b3cafb602f9f9d16a5074d3c3bfcd31b4d599d416c3558e77788181bd4c4597.jpg](../iclr_results/90_Feedback%20Favors%20the%20Generalization%20of%20Neural%20ODEs/images/4b3cafb602f9f9d16a5074d3c3bfcd31b4d599d416c3558e77788181bd4c4597.jpg)

![563b1f9ff0f4ac903cbee97d8bb00a3a3adb0b8f42d3a9cb300fc822be4a095f.jpg](../iclr_results/90_Feedback%20Favors%20the%20Generalization%20of%20Neural%20ODEs/images/563b1f9ff0f4ac903cbee97d8bb00a3a3adb0b8f42d3a9cb300fc822be4a095f.jpg)

![5e0c6f8ff9bed926bfe815dc0eaf3a5706f2b17ff7073f8d5361a1bc34db6ee0.jpg](../iclr_results/90_Feedback%20Favors%20the%20Generalization%20of%20Neural%20ODEs/images/5e0c6f8ff9bed926bfe815dc0eaf3a5706f2b17ff7073f8d5361a1bc34db6ee0.jpg)

![66cd7448bd9a5846e010ec9741875dcfb2b3ede32a30a32742da29708fad15d0.jpg](../iclr_results/90_Feedback%20Favors%20the%20Generalization%20of%20Neural%20ODEs/images/66cd7448bd9a5846e010ec9741875dcfb2b3ede32a30a32742da29708fad15d0.jpg)

![73057e15c7d6240fb375453672f3c7a3c72d7d67666dfd18d55d8a8fad36754f.jpg](../iclr_results/90_Feedback%20Favors%20the%20Generalization%20of%20Neural%20ODEs/images/73057e15c7d6240fb375453672f3c7a3c72d7d67666dfd18d55d8a8fad36754f.jpg)

![7c6602888c5991a53c523bbdce15cad10f468eaec94170eaec7c1f29b62e2a99.jpg](../iclr_results/90_Feedback%20Favors%20the%20Generalization%20of%20Neural%20ODEs/images/7c6602888c5991a53c523bbdce15cad10f468eaec94170eaec7c1f29b62e2a99.jpg)

![873afabe083b67be9490abd50f648a1777ab33de1b00a7815342e19fdcf4d459.jpg](../iclr_results/90_Feedback%20Favors%20the%20Generalization%20of%20Neural%20ODEs/images/873afabe083b67be9490abd50f648a1777ab33de1b00a7815342e19fdcf4d459.jpg)

![94149050745ca74d514f1f211687ddfc9b7536d05426e8ede37ec91abf1bf3ce.jpg](../iclr_results/90_Feedback%20Favors%20the%20Generalization%20of%20Neural%20ODEs/images/94149050745ca74d514f1f211687ddfc9b7536d05426e8ede37ec91abf1bf3ce.jpg)

![958236bc4b648cb1daa73028d64769a08c538dd16388f6823708a791b68168e5.jpg](../iclr_results/90_Feedback%20Favors%20the%20Generalization%20of%20Neural%20ODEs/images/958236bc4b648cb1daa73028d64769a08c538dd16388f6823708a791b68168e5.jpg)

![9c6862a73f971c2843be2f301c2bda2760d1546205d99d9be040cd8b716fdaad.jpg](../iclr_results/90_Feedback%20Favors%20the%20Generalization%20of%20Neural%20ODEs/images/9c6862a73f971c2843be2f301c2bda2760d1546205d99d9be040cd8b716fdaad.jpg)

![a04da881bd128791ec4681b4dd1b65ef3c6c46a4f343183ba394a4647990d588.jpg](../iclr_results/90_Feedback%20Favors%20the%20Generalization%20of%20Neural%20ODEs/images/a04da881bd128791ec4681b4dd1b65ef3c6c46a4f343183ba394a4647990d588.jpg)

![a0f350d331eea514dd4a00d464644cf65b915eda16ce623934a6580469c53eb4.jpg](../iclr_results/90_Feedback%20Favors%20the%20Generalization%20of%20Neural%20ODEs/images/a0f350d331eea514dd4a00d464644cf65b915eda16ce623934a6580469c53eb4.jpg)

![a33d05f7f09172c3e7f5b80cecedbe228b47d0141e644169cf52e47d9f86a076.jpg](../iclr_results/90_Feedback%20Favors%20the%20Generalization%20of%20Neural%20ODEs/images/a33d05f7f09172c3e7f5b80cecedbe228b47d0141e644169cf52e47d9f86a076.jpg)

![b214ca96b4692749289830ac77299d39b65da121e7de4eadc34f8ee151430275.jpg](../iclr_results/90_Feedback%20Favors%20the%20Generalization%20of%20Neural%20ODEs/images/b214ca96b4692749289830ac77299d39b65da121e7de4eadc34f8ee151430275.jpg)

![cbeb87cdf15a1bef0809b89420e88d238775237221a37225781fc475748f2120.jpg](../iclr_results/90_Feedback%20Favors%20the%20Generalization%20of%20Neural%20ODEs/images/cbeb87cdf15a1bef0809b89420e88d238775237221a37225781fc475748f2120.jpg)

![d6c61d33f3965cd9d74ccf830cf3d43e283714db9a698640e2cbf09c37fb5b1f.jpg](../iclr_results/90_Feedback%20Favors%20the%20Generalization%20of%20Neural%20ODEs/images/d6c61d33f3965cd9d74ccf830cf3d43e283714db9a698640e2cbf09c37fb5b1f.jpg)

![e8a299d26e51b5052bc6737cc36168e0d4a4df95728fc5bf488d603842c0332f.jpg](../iclr_results/90_Feedback%20Favors%20the%20Generalization%20of%20Neural%20ODEs/images/e8a299d26e51b5052bc6737cc36168e0d4a4df95728fc5bf488d603842c0332f.jpg)

![ee529f847eb4e5da1888a3dd824cbfdcc6e35d0d74c4aa0c84907916d68ef44a.jpg](../iclr_results/90_Feedback%20Favors%20the%20Generalization%20of%20Neural%20ODEs/images/ee529f847eb4e5da1888a3dd824cbfdcc6e35d0d74c4aa0c84907916d68ef44a.jpg)

![eec732852fc43f391aabef82d9f56c6024b4b461fc87e7d90a11e6ac30458ddc.jpg](../iclr_results/90_Feedback%20Favors%20the%20Generalization%20of%20Neural%20ODEs/images/eec732852fc43f391aabef82d9f56c6024b4b461fc87e7d90a11e6ac30458ddc.jpg)

## Comparing noisy neural population dynamics using optimal transport distances


### Images

![0b0fa8675e373f8e8ecd398dca1ca484dd3d350645105661d565044835433e1e.jpg](../iclr_results/91_Comparing%20noisy%20neural%20population%20dynamics%20using%20optimal%20transport%20distances/images/0b0fa8675e373f8e8ecd398dca1ca484dd3d350645105661d565044835433e1e.jpg)

![14776444e7fe4e53c76ff6bfb38805169f0064826c824f54b90a63a1e7f4fff2.jpg](../iclr_results/91_Comparing%20noisy%20neural%20population%20dynamics%20using%20optimal%20transport%20distances/images/14776444e7fe4e53c76ff6bfb38805169f0064826c824f54b90a63a1e7f4fff2.jpg)

![24bf6bfdfcb1aebd9c4b4b3a3c9b34322a69a8b10e779d3e3ad5eff1f6980771.jpg](../iclr_results/91_Comparing%20noisy%20neural%20population%20dynamics%20using%20optimal%20transport%20distances/images/24bf6bfdfcb1aebd9c4b4b3a3c9b34322a69a8b10e779d3e3ad5eff1f6980771.jpg)

![299824d688045e03971c02c653d8d84dbb3757c006c9e57cde0dffb45f633dbd.jpg](../iclr_results/91_Comparing%20noisy%20neural%20population%20dynamics%20using%20optimal%20transport%20distances/images/299824d688045e03971c02c653d8d84dbb3757c006c9e57cde0dffb45f633dbd.jpg)

![311e8a3e8c701ef26bddfd5079f696b24240a3b574ed37dc859c6b8d0caff70e.jpg](../iclr_results/91_Comparing%20noisy%20neural%20population%20dynamics%20using%20optimal%20transport%20distances/images/311e8a3e8c701ef26bddfd5079f696b24240a3b574ed37dc859c6b8d0caff70e.jpg)

![515ec4e6bc7b879ecf901cae425a066af30bfc68290da5efc195b1012af3765d.jpg](../iclr_results/91_Comparing%20noisy%20neural%20population%20dynamics%20using%20optimal%20transport%20distances/images/515ec4e6bc7b879ecf901cae425a066af30bfc68290da5efc195b1012af3765d.jpg)

![56ad95c96336dc8895128d4e3627e0bb66031536ce7352b24238f29b75f9066f.jpg](../iclr_results/91_Comparing%20noisy%20neural%20population%20dynamics%20using%20optimal%20transport%20distances/images/56ad95c96336dc8895128d4e3627e0bb66031536ce7352b24238f29b75f9066f.jpg)

![625463b28eab6e356e8857d5b68859ac61ef5e43c60016e08d75d512eefd7a65.jpg](../iclr_results/91_Comparing%20noisy%20neural%20population%20dynamics%20using%20optimal%20transport%20distances/images/625463b28eab6e356e8857d5b68859ac61ef5e43c60016e08d75d512eefd7a65.jpg)

![694a0d995bc534741b31672985c3345e91bf482988f255e62d91efb36be1b94b.jpg](../iclr_results/91_Comparing%20noisy%20neural%20population%20dynamics%20using%20optimal%20transport%20distances/images/694a0d995bc534741b31672985c3345e91bf482988f255e62d91efb36be1b94b.jpg)

![71c496305b3b050ffe9cd5f6ee7a4db41225322ce8139a5aa0bf134c74af3f3a.jpg](../iclr_results/91_Comparing%20noisy%20neural%20population%20dynamics%20using%20optimal%20transport%20distances/images/71c496305b3b050ffe9cd5f6ee7a4db41225322ce8139a5aa0bf134c74af3f3a.jpg)

![a15e453840650ffe72b795e8c3e2146324b78f2c28fa3cdc8e66a030722d6bc2.jpg](../iclr_results/91_Comparing%20noisy%20neural%20population%20dynamics%20using%20optimal%20transport%20distances/images/a15e453840650ffe72b795e8c3e2146324b78f2c28fa3cdc8e66a030722d6bc2.jpg)

![e95c7107a99c703c091b42bfae35079d24bd0e54ddb979466476d3684101d033.jpg](../iclr_results/91_Comparing%20noisy%20neural%20population%20dynamics%20using%20optimal%20transport%20distances/images/e95c7107a99c703c091b42bfae35079d24bd0e54ddb979466476d3684101d033.jpg)

![eb6868f60da92b79b48bca213d1d0ca177438f09022270e7f123f105d0917a59.jpg](../iclr_results/91_Comparing%20noisy%20neural%20population%20dynamics%20using%20optimal%20transport%20distances/images/eb6868f60da92b79b48bca213d1d0ca177438f09022270e7f123f105d0917a59.jpg)

## Subgraph Federated Learning for Local Generalization


### Images

![01856a481d4cdb9b20be866e02a6a9313bd87a11ee1efe7f74b157c35768cf02.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/images/01856a481d4cdb9b20be866e02a6a9313bd87a11ee1efe7f74b157c35768cf02.jpg)

![0de2457d2132a54f5d04d3c6c688c1ee1019104c3852a102f8b2489b7c4e9a7e.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/images/0de2457d2132a54f5d04d3c6c688c1ee1019104c3852a102f8b2489b7c4e9a7e.jpg)

![298e1fe3c752b4e07d86b3de62316e44a58fd074c0641c8097bb416bc65183f3.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/images/298e1fe3c752b4e07d86b3de62316e44a58fd074c0641c8097bb416bc65183f3.jpg)

![485459937304e4dac12c8248e12bfbdbd7366205ea12d5fbd81b62188c0c7a53.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/images/485459937304e4dac12c8248e12bfbdbd7366205ea12d5fbd81b62188c0c7a53.jpg)

![4afb6e73f8f0f8b7ac6ff938b971325adb4adc3c10d8075c5ecfe50b24626ed7.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/images/4afb6e73f8f0f8b7ac6ff938b971325adb4adc3c10d8075c5ecfe50b24626ed7.jpg)

![5cb726291cd5ca2d31ea63556f2a951f3b4d1b8fde9045125ddcfc9f2c00527f.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/images/5cb726291cd5ca2d31ea63556f2a951f3b4d1b8fde9045125ddcfc9f2c00527f.jpg)

![7379ba11beb17fffb4d2b4a24f6ec17a30f72a99738ac0b4d60941c6e5a4e050.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/images/7379ba11beb17fffb4d2b4a24f6ec17a30f72a99738ac0b4d60941c6e5a4e050.jpg)

![96dcab18c470840ff61162023456482782b4a5ecdf1a42bd4aec549968bfe3e7.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/images/96dcab18c470840ff61162023456482782b4a5ecdf1a42bd4aec549968bfe3e7.jpg)

![9e928cc1be10ce326b779b9ee54dd988b6a0c2d60bc27b4bea83b4936e754e4f.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/images/9e928cc1be10ce326b779b9ee54dd988b6a0c2d60bc27b4bea83b4936e754e4f.jpg)

### Tables

![13daace9bcb8b4bda7c12d54f6eb9eca8693105a953a95e8457d9a330a8bc036.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/tables/13daace9bcb8b4bda7c12d54f6eb9eca8693105a953a95e8457d9a330a8bc036.jpg)

![209436e48f064ca993f8e8cf90d9b706e6cffb7964addbfb69a6326f4c9c5484.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/tables/209436e48f064ca993f8e8cf90d9b706e6cffb7964addbfb69a6326f4c9c5484.jpg)

![20d3e451021b98b5d650e93ab4a178578a6b3ad57c158c0331b6d4f4d8fd20cd.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/tables/20d3e451021b98b5d650e93ab4a178578a6b3ad57c158c0331b6d4f4d8fd20cd.jpg)

![333f2ac6faa439b5c911a5b2c965bcf358ec1e097486771b7164c44141d4fac0.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/tables/333f2ac6faa439b5c911a5b2c965bcf358ec1e097486771b7164c44141d4fac0.jpg)

![3d9c74b65bc3107cfd87c3192ee25698248d9e22a382db461cbbef4a01b702a7.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/tables/3d9c74b65bc3107cfd87c3192ee25698248d9e22a382db461cbbef4a01b702a7.jpg)

![44c8a8c726c20e217a478a9b9024142d302dd1e1b8412bc5bbed830d234732df.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/tables/44c8a8c726c20e217a478a9b9024142d302dd1e1b8412bc5bbed830d234732df.jpg)

![56ee95189d0df63f0163b8a6dfc2c7546db688d100daac2c5071641380be5b16.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/tables/56ee95189d0df63f0163b8a6dfc2c7546db688d100daac2c5071641380be5b16.jpg)

![5e007961a9aeda07522f8fd544ab70598fbcb5bc921574b0422384b65df01b7d.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/tables/5e007961a9aeda07522f8fd544ab70598fbcb5bc921574b0422384b65df01b7d.jpg)

![5e12e1781c80edf276c63f6035bb0d589d9ff5578525a4d0225fe86753af9f6b.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/tables/5e12e1781c80edf276c63f6035bb0d589d9ff5578525a4d0225fe86753af9f6b.jpg)

![64ed3a08e6727d4022b2efa2882d151f42c307bd5ff6e50681f114695d735bac.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/tables/64ed3a08e6727d4022b2efa2882d151f42c307bd5ff6e50681f114695d735bac.jpg)

![6e683de10a44e6d7749832b7960b93b44da7458c19c57418a669557722463dee.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/tables/6e683de10a44e6d7749832b7960b93b44da7458c19c57418a669557722463dee.jpg)

![7d94dd2b7ba9ee25710fa7eb0c60b0506bee51b548113768165a0b4ac9d1288b.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/tables/7d94dd2b7ba9ee25710fa7eb0c60b0506bee51b548113768165a0b4ac9d1288b.jpg)

![8fdcb9055358cd35450e0d8c35fda34e43a16269f5146bd0311d6410e52b37a3.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/tables/8fdcb9055358cd35450e0d8c35fda34e43a16269f5146bd0311d6410e52b37a3.jpg)

![99faf39e368307f4978a2641a4c9015a708145c7fbbf4e6701f67c555d0629c0.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/tables/99faf39e368307f4978a2641a4c9015a708145c7fbbf4e6701f67c555d0629c0.jpg)

![b93ebc521fceae74ad61e35dce3db469ece38bf4ecc54c3ee688c11b6b76d66a.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/tables/b93ebc521fceae74ad61e35dce3db469ece38bf4ecc54c3ee688c11b6b76d66a.jpg)

![bc4f2bb4200e8ef246c05fb04408ae88961c6ac3f06de5380aa72dcf0cf19243.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/tables/bc4f2bb4200e8ef246c05fb04408ae88961c6ac3f06de5380aa72dcf0cf19243.jpg)

![c9d6f6f2898428c759dae6ae4cdc67fa546b37815ff58dc2058efec194a30ce0.jpg](../iclr_results/92_Subgraph%20Federated%20Learning%20for%20Local%20Generalization/tables/c9d6f6f2898428c759dae6ae4cdc67fa546b37815ff58dc2058efec194a30ce0.jpg)

## RB-Modulation: Training-Free Stylization using Reference-Based Modulation


### Images

![09474e7897d5a07fa3c0d6eef21092247f2dfad40ad718063be1f96d9149b869.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/09474e7897d5a07fa3c0d6eef21092247f2dfad40ad718063be1f96d9149b869.jpg)

![23744c9bd52daf3b30efbbbe2b8c5443fbed1b4ca18708bc9ad371ce449aa9f5.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/23744c9bd52daf3b30efbbbe2b8c5443fbed1b4ca18708bc9ad371ce449aa9f5.jpg)

![2fe2222dd18483f97037bff17e6827853b57e597b2789960c6ce31a93816a8d6.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/2fe2222dd18483f97037bff17e6827853b57e597b2789960c6ce31a93816a8d6.jpg)

![395bbdfa8f39218d9898bb8f874101baeb433e4435f7431f18d07497e844bf12.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/395bbdfa8f39218d9898bb8f874101baeb433e4435f7431f18d07497e844bf12.jpg)

![44f46e2df5b6c45e657ea805cd307d5e44caffc14f614d00607153fa4fc1d16a.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/44f46e2df5b6c45e657ea805cd307d5e44caffc14f614d00607153fa4fc1d16a.jpg)

![4698ddee0e5906bf64261d01473c212c449cb6641479738ded3882724f42e4fb.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/4698ddee0e5906bf64261d01473c212c449cb6641479738ded3882724f42e4fb.jpg)

![499a6873c11b0357038a081706fc025c2c23a8e8a22d88010cd577e22b695522.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/499a6873c11b0357038a081706fc025c2c23a8e8a22d88010cd577e22b695522.jpg)

![4c3e58434ab53d18476463e785901cabbc93eb053214428dd144e13477eb8ecf.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/4c3e58434ab53d18476463e785901cabbc93eb053214428dd144e13477eb8ecf.jpg)

![53dc9fa1c7f528b6430c4abbf68cf907a614213dda95da5d4cddc91adad71226.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/53dc9fa1c7f528b6430c4abbf68cf907a614213dda95da5d4cddc91adad71226.jpg)

![61b1a8623d810fb8ecabd0e1c5175dfd4807a41f35083d3c46b2af765f9f6115.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/61b1a8623d810fb8ecabd0e1c5175dfd4807a41f35083d3c46b2af765f9f6115.jpg)

![871acfe88f138e0f1eb0030bf69a0a23a1b30dba8c4c6b34d3fc5baa112bcca6.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/871acfe88f138e0f1eb0030bf69a0a23a1b30dba8c4c6b34d3fc5baa112bcca6.jpg)

![96df21f5548dbbf7fa72df5efba9b8898b2791bb6bb7d49ac25211cd7e0eba2e.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/96df21f5548dbbf7fa72df5efba9b8898b2791bb6bb7d49ac25211cd7e0eba2e.jpg)

![9ed59e1573e2d63cf7a183d38b9b58f847dd18c395cb08a17b062fbf7d00425f.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/9ed59e1573e2d63cf7a183d38b9b58f847dd18c395cb08a17b062fbf7d00425f.jpg)

![a3196d0863c12f6a3709b44b3622fca01142ebc81bb20507439e507c698c2c40.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/a3196d0863c12f6a3709b44b3622fca01142ebc81bb20507439e507c698c2c40.jpg)

![b63e732dbed3683dac039b745ae3026b28f617735e80a91a1aab69d5ef4ca547.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/b63e732dbed3683dac039b745ae3026b28f617735e80a91a1aab69d5ef4ca547.jpg)

![b653f332a2c9e09a2aae25f4f5e3f1b192cbe682097bfeb8e244fafe1967e32c.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/b653f332a2c9e09a2aae25f4f5e3f1b192cbe682097bfeb8e244fafe1967e32c.jpg)

![b9880c98afc7f2e6c5470ab59251fdd95839ebb4ae4bbbfeefda523061c3dfe9.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/b9880c98afc7f2e6c5470ab59251fdd95839ebb4ae4bbbfeefda523061c3dfe9.jpg)

![bf515cbcec1755d01e546427644585b21a81039afafa02bafe1983e5a742c4fb.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/bf515cbcec1755d01e546427644585b21a81039afafa02bafe1983e5a742c4fb.jpg)

![c03743dc43b102a0bfdd2fa2da889ad85418f4a03626cd586370bb5a0c63f3ad.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/c03743dc43b102a0bfdd2fa2da889ad85418f4a03626cd586370bb5a0c63f3ad.jpg)

![c8cc2a70a8dca97862bceef85420c68af53c0e6850bd1544617ec7abc3313c1c.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/c8cc2a70a8dca97862bceef85420c68af53c0e6850bd1544617ec7abc3313c1c.jpg)

![cbff2246d511d458339daacfa8aa1ac27c36986ac74cd1d7cbea67879b36f17a.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/cbff2246d511d458339daacfa8aa1ac27c36986ac74cd1d7cbea67879b36f17a.jpg)

![cc2d47ee559b7edb6f7393c651315bd94e200ebe2063e0625f22663888fbd4bf.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/cc2d47ee559b7edb6f7393c651315bd94e200ebe2063e0625f22663888fbd4bf.jpg)

![cc7b364c41cc37ca1990e2abc979611d6f529c6c0bbf0bea58cc396725178d1d.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/cc7b364c41cc37ca1990e2abc979611d6f529c6c0bbf0bea58cc396725178d1d.jpg)

![d52b8c1d663ce63be875efaa10a4d4b579bf97ab4b7d440b3f3046788e0e804a.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/images/d52b8c1d663ce63be875efaa10a4d4b579bf97ab4b7d440b3f3046788e0e804a.jpg)

### Tables

![0d5acb6ead85191fb6783a2a6e1e70c2b3cfd3202a06cfe2e9104b8079cf6bcd.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/tables/0d5acb6ead85191fb6783a2a6e1e70c2b3cfd3202a06cfe2e9104b8079cf6bcd.jpg)

![437f2c77db17cb30adf1824a8c9afe0da60d3f831c03fd833fb52877fe8e7ccc.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/tables/437f2c77db17cb30adf1824a8c9afe0da60d3f831c03fd833fb52877fe8e7ccc.jpg)

![9e4985d2ddfa22dd098071f850822f30c42ed3218bbbf5db25f5eca962e6f136.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/tables/9e4985d2ddfa22dd098071f850822f30c42ed3218bbbf5db25f5eca962e6f136.jpg)

![e825b40759ba8f6589b93419cc17b40828f6ea2f7fbb4790735a08e3a7445d98.jpg](../iclr_results/93_RB-Modulation_%20Training-Free%20Stylization%20using%20Reference-Based%20Modulation/tables/e825b40759ba8f6589b93419cc17b40828f6ea2f7fbb4790735a08e3a7445d98.jpg)

## The Geometry of Categorical and Hierarchical Concepts in Large Language Models


### Images

![08ad11ea00d315ce2e6477a9733a35c91c164a39dbbe2d221bb9801583c64059.jpg](../iclr_results/94_The%20Geometry%20of%20Categorical%20and%20Hierarchical%20Concepts%20in%20Large%20Language%20Models/images/08ad11ea00d315ce2e6477a9733a35c91c164a39dbbe2d221bb9801583c64059.jpg)

![28872fc54ecf3ca78b2c9af49fa6cb39579f56fbff2f0ca33b266c8a1ce9285e.jpg](../iclr_results/94_The%20Geometry%20of%20Categorical%20and%20Hierarchical%20Concepts%20in%20Large%20Language%20Models/images/28872fc54ecf3ca78b2c9af49fa6cb39579f56fbff2f0ca33b266c8a1ce9285e.jpg)

![33e5d43152c09e789cda6a129abeac8fb04b03ce7f9a28f8f624c8b4bae54871.jpg](../iclr_results/94_The%20Geometry%20of%20Categorical%20and%20Hierarchical%20Concepts%20in%20Large%20Language%20Models/images/33e5d43152c09e789cda6a129abeac8fb04b03ce7f9a28f8f624c8b4bae54871.jpg)

![49a529b455953520441c9c941bdb91856e45ee9c1c11f45631f20684a6156d80.jpg](../iclr_results/94_The%20Geometry%20of%20Categorical%20and%20Hierarchical%20Concepts%20in%20Large%20Language%20Models/images/49a529b455953520441c9c941bdb91856e45ee9c1c11f45631f20684a6156d80.jpg)

![4fb50750efa929714d9dbf489dbef369a85faafafa1ca1e2930876b09375a2af.jpg](../iclr_results/94_The%20Geometry%20of%20Categorical%20and%20Hierarchical%20Concepts%20in%20Large%20Language%20Models/images/4fb50750efa929714d9dbf489dbef369a85faafafa1ca1e2930876b09375a2af.jpg)

![583ebe81484fb6f390f33b70f5cc3b333256616adb1abebf168bc91d0fe597cc.jpg](../iclr_results/94_The%20Geometry%20of%20Categorical%20and%20Hierarchical%20Concepts%20in%20Large%20Language%20Models/images/583ebe81484fb6f390f33b70f5cc3b333256616adb1abebf168bc91d0fe597cc.jpg)

![8054ee7f25064dc729277ef504194954525e84353ec76c705b5de69c2b96aefc.jpg](../iclr_results/94_The%20Geometry%20of%20Categorical%20and%20Hierarchical%20Concepts%20in%20Large%20Language%20Models/images/8054ee7f25064dc729277ef504194954525e84353ec76c705b5de69c2b96aefc.jpg)

![895056dba34b43e071d064d365fc90932903a1a708482bb2babbee968dca0f1e.jpg](../iclr_results/94_The%20Geometry%20of%20Categorical%20and%20Hierarchical%20Concepts%20in%20Large%20Language%20Models/images/895056dba34b43e071d064d365fc90932903a1a708482bb2babbee968dca0f1e.jpg)

![8ffa966c33b2c55a68db5a453ad37f05dcb998b438add1519ecaf5ee9ad4d795.jpg](../iclr_results/94_The%20Geometry%20of%20Categorical%20and%20Hierarchical%20Concepts%20in%20Large%20Language%20Models/images/8ffa966c33b2c55a68db5a453ad37f05dcb998b438add1519ecaf5ee9ad4d795.jpg)

![91c0308a764fa8ac78bfa7ef964465e2c5a1de3a8735bd2947b30368f19c7a53.jpg](../iclr_results/94_The%20Geometry%20of%20Categorical%20and%20Hierarchical%20Concepts%20in%20Large%20Language%20Models/images/91c0308a764fa8ac78bfa7ef964465e2c5a1de3a8735bd2947b30368f19c7a53.jpg)

![9dffdaa642a32dfa5e8e82bbe537ca8cbe1ccdc50fb39021653849ada9ee038d.jpg](../iclr_results/94_The%20Geometry%20of%20Categorical%20and%20Hierarchical%20Concepts%20in%20Large%20Language%20Models/images/9dffdaa642a32dfa5e8e82bbe537ca8cbe1ccdc50fb39021653849ada9ee038d.jpg)

![a837829aebef6a17c412febfcf3120e2cedbfeb4bd907a1642c754a24cb17983.jpg](../iclr_results/94_The%20Geometry%20of%20Categorical%20and%20Hierarchical%20Concepts%20in%20Large%20Language%20Models/images/a837829aebef6a17c412febfcf3120e2cedbfeb4bd907a1642c754a24cb17983.jpg)

![ad318244d1baf661c1a388c29944eab48061b28c2616de34a434088a02bc8682.jpg](../iclr_results/94_The%20Geometry%20of%20Categorical%20and%20Hierarchical%20Concepts%20in%20Large%20Language%20Models/images/ad318244d1baf661c1a388c29944eab48061b28c2616de34a434088a02bc8682.jpg)

![b6867ea2e5f80a3ee70c64a06f2ba1956ae5e9f79477ae14eae903598c26da53.jpg](../iclr_results/94_The%20Geometry%20of%20Categorical%20and%20Hierarchical%20Concepts%20in%20Large%20Language%20Models/images/b6867ea2e5f80a3ee70c64a06f2ba1956ae5e9f79477ae14eae903598c26da53.jpg)

![d31215fc18b6d82d368409b2bd996fbecc82831b9e686bf99b835ad57e58a9ff.jpg](../iclr_results/94_The%20Geometry%20of%20Categorical%20and%20Hierarchical%20Concepts%20in%20Large%20Language%20Models/images/d31215fc18b6d82d368409b2bd996fbecc82831b9e686bf99b835ad57e58a9ff.jpg)

![da8e502a2f13cbb56e091a8cae6d6a2fe4463c5dac8ee36f81573a34c4464279.jpg](../iclr_results/94_The%20Geometry%20of%20Categorical%20and%20Hierarchical%20Concepts%20in%20Large%20Language%20Models/images/da8e502a2f13cbb56e091a8cae6d6a2fe4463c5dac8ee36f81573a34c4464279.jpg)

![dbf57d6c5f3026fc5ca2f24079894e22d0a3ab3f830c8cef085f21ff91a82682.jpg](../iclr_results/94_The%20Geometry%20of%20Categorical%20and%20Hierarchical%20Concepts%20in%20Large%20Language%20Models/images/dbf57d6c5f3026fc5ca2f24079894e22d0a3ab3f830c8cef085f21ff91a82682.jpg)

![e4e398876479fd4fdf0e17aee314ccfb0c6c7f1f99b8966f10fd212d9d61ff84.jpg](../iclr_results/94_The%20Geometry%20of%20Categorical%20and%20Hierarchical%20Concepts%20in%20Large%20Language%20Models/images/e4e398876479fd4fdf0e17aee314ccfb0c6c7f1f99b8966f10fd212d9d61ff84.jpg)

![eeaae67c17c274a3a257ec91c148c540db3cd285611ee076d1af0ff78d6f6d88.jpg](../iclr_results/94_The%20Geometry%20of%20Categorical%20and%20Hierarchical%20Concepts%20in%20Large%20Language%20Models/images/eeaae67c17c274a3a257ec91c148c540db3cd285611ee076d1af0ff78d6f6d88.jpg)

![f12e8d8c36f10f60b7c6efc7b40f0660c635ddbac5ff5d3141e9286b4fa467b9.jpg](../iclr_results/94_The%20Geometry%20of%20Categorical%20and%20Hierarchical%20Concepts%20in%20Large%20Language%20Models/images/f12e8d8c36f10f60b7c6efc7b40f0660c635ddbac5ff5d3141e9286b4fa467b9.jpg)

### Tables

![b23104b4708ff18f666d2ab2a4a3ff6eb5bc364d60cef26c60dc3b99dd78371d.jpg](../iclr_results/94_The%20Geometry%20of%20Categorical%20and%20Hierarchical%20Concepts%20in%20Large%20Language%20Models/tables/b23104b4708ff18f666d2ab2a4a3ff6eb5bc364d60cef26c60dc3b99dd78371d.jpg)

## TopoLM: brain-like spatio-functional organization in a topographic language model


### Images

![18a5884f8935ac84473f88267846098de373f5b60ba74853546198df2b6fbbf8.jpg](../iclr_results/95_TopoLM_%20brain-like%20spatio-functional%20organization%20in%20a%20topographic%20language%20model/images/18a5884f8935ac84473f88267846098de373f5b60ba74853546198df2b6fbbf8.jpg)

![269e9185907a1c04a2f90bea5b6cf27cf15456f71d289086bd4648078c90f18b.jpg](../iclr_results/95_TopoLM_%20brain-like%20spatio-functional%20organization%20in%20a%20topographic%20language%20model/images/269e9185907a1c04a2f90bea5b6cf27cf15456f71d289086bd4648078c90f18b.jpg)

![2d16fab705f65fa92d62202302fefdf718ed2d280fd387730db811e117339641.jpg](../iclr_results/95_TopoLM_%20brain-like%20spatio-functional%20organization%20in%20a%20topographic%20language%20model/images/2d16fab705f65fa92d62202302fefdf718ed2d280fd387730db811e117339641.jpg)

![31fd565d0399a2f9f099d83b76275f1496fec78c3738989fead37fb0ee7b1861.jpg](../iclr_results/95_TopoLM_%20brain-like%20spatio-functional%20organization%20in%20a%20topographic%20language%20model/images/31fd565d0399a2f9f099d83b76275f1496fec78c3738989fead37fb0ee7b1861.jpg)

![3583ce3f068f1b81eaed06e94fb697349290ca1599e77946b0017b546d4cd3da.jpg](../iclr_results/95_TopoLM_%20brain-like%20spatio-functional%20organization%20in%20a%20topographic%20language%20model/images/3583ce3f068f1b81eaed06e94fb697349290ca1599e77946b0017b546d4cd3da.jpg)

![3de2a30987637b88d0086997681bfb7748542b5938e44ba45891c291700c1b74.jpg](../iclr_results/95_TopoLM_%20brain-like%20spatio-functional%20organization%20in%20a%20topographic%20language%20model/images/3de2a30987637b88d0086997681bfb7748542b5938e44ba45891c291700c1b74.jpg)

![5234dde00aa98752ac78f93920d1da581c4f73380ac7aa69dd4db0829d1c80b3.jpg](../iclr_results/95_TopoLM_%20brain-like%20spatio-functional%20organization%20in%20a%20topographic%20language%20model/images/5234dde00aa98752ac78f93920d1da581c4f73380ac7aa69dd4db0829d1c80b3.jpg)

![68ea9d20e7a8fad1036540eb6f5cb4e01eb0fdf40abd987ed4b1604969e9eeb3.jpg](../iclr_results/95_TopoLM_%20brain-like%20spatio-functional%20organization%20in%20a%20topographic%20language%20model/images/68ea9d20e7a8fad1036540eb6f5cb4e01eb0fdf40abd987ed4b1604969e9eeb3.jpg)

![7d3e7ada586f0d9540ff7183a782bc1ba9de93e73b8b51575fe496f1880964a2.jpg](../iclr_results/95_TopoLM_%20brain-like%20spatio-functional%20organization%20in%20a%20topographic%20language%20model/images/7d3e7ada586f0d9540ff7183a782bc1ba9de93e73b8b51575fe496f1880964a2.jpg)

![85522b2f16878e37ac6f261b1fae2ab82a34db70cb81336c34129fd7f3ad041c.jpg](../iclr_results/95_TopoLM_%20brain-like%20spatio-functional%20organization%20in%20a%20topographic%20language%20model/images/85522b2f16878e37ac6f261b1fae2ab82a34db70cb81336c34129fd7f3ad041c.jpg)

![95b6acd0d22d846fab6a8e4382935932f6e079b87cebba8fff2e22a03c473b1b.jpg](../iclr_results/95_TopoLM_%20brain-like%20spatio-functional%20organization%20in%20a%20topographic%20language%20model/images/95b6acd0d22d846fab6a8e4382935932f6e079b87cebba8fff2e22a03c473b1b.jpg)

![a5422115bd35f14046e3f1eebabbaf0bc75407fc4803b458e5077b6b23fb4663.jpg](../iclr_results/95_TopoLM_%20brain-like%20spatio-functional%20organization%20in%20a%20topographic%20language%20model/images/a5422115bd35f14046e3f1eebabbaf0bc75407fc4803b458e5077b6b23fb4663.jpg)

![c1cf71224a963f6011d3ac564617b4732aa460cdd1f2a4130d7a0f025e69c4fa.jpg](../iclr_results/95_TopoLM_%20brain-like%20spatio-functional%20organization%20in%20a%20topographic%20language%20model/images/c1cf71224a963f6011d3ac564617b4732aa460cdd1f2a4130d7a0f025e69c4fa.jpg)

![c72c06ab5a23cbb16fd350cd86f0540ac3d64bb8fe5ab1e590fafa02016646a4.jpg](../iclr_results/95_TopoLM_%20brain-like%20spatio-functional%20organization%20in%20a%20topographic%20language%20model/images/c72c06ab5a23cbb16fd350cd86f0540ac3d64bb8fe5ab1e590fafa02016646a4.jpg)

![e81369dd7a1e002237130e41e255511505f1265f254c590e872a7c62663c7fd0.jpg](../iclr_results/95_TopoLM_%20brain-like%20spatio-functional%20organization%20in%20a%20topographic%20language%20model/images/e81369dd7a1e002237130e41e255511505f1265f254c590e872a7c62663c7fd0.jpg)

![e8850584438b99f99ae421fbaa5263f7fdcd25c0cce9f4b0b5a2d7f61f7f419d.jpg](../iclr_results/95_TopoLM_%20brain-like%20spatio-functional%20organization%20in%20a%20topographic%20language%20model/images/e8850584438b99f99ae421fbaa5263f7fdcd25c0cce9f4b0b5a2d7f61f7f419d.jpg)

![fc82cc30988541be7c82b2e8ba5305e05560cd56c447a5de1812476d2f5aa6f5.jpg](../iclr_results/95_TopoLM_%20brain-like%20spatio-functional%20organization%20in%20a%20topographic%20language%20model/images/fc82cc30988541be7c82b2e8ba5305e05560cd56c447a5de1812476d2f5aa6f5.jpg)

### Tables

![78e8c2dba063d5676b6fcf56dde85e156373923dcef5276e68436e69b6a403fa.jpg](../iclr_results/95_TopoLM_%20brain-like%20spatio-functional%20organization%20in%20a%20topographic%20language%20model/tables/78e8c2dba063d5676b6fcf56dde85e156373923dcef5276e68436e69b6a403fa.jpg)

## Problem-Parameter-Free Federated Learning


### Images

![53eeb9bd9592b8b3f11458fad709e65c9ebb09c4dc51916c5423d611378bcf47.jpg](../iclr_results/96_Problem-Parameter-Free%20Federated%20Learning/images/53eeb9bd9592b8b3f11458fad709e65c9ebb09c4dc51916c5423d611378bcf47.jpg)

![745482e9753d696529ef74279f261ff8fceca6cd737f299e909c404e0a6932f1.jpg](../iclr_results/96_Problem-Parameter-Free%20Federated%20Learning/images/745482e9753d696529ef74279f261ff8fceca6cd737f299e909c404e0a6932f1.jpg)

![7e5651e469502be83a2bb5d807ecfbd28cc7f8fcc041d46f3f0e94a8402a0c9e.jpg](../iclr_results/96_Problem-Parameter-Free%20Federated%20Learning/images/7e5651e469502be83a2bb5d807ecfbd28cc7f8fcc041d46f3f0e94a8402a0c9e.jpg)

![96f4577ddae4d315c7d9b715bc2e959e2e59c6a7431fc8f52ba6e3063c0a78f3.jpg](../iclr_results/96_Problem-Parameter-Free%20Federated%20Learning/images/96f4577ddae4d315c7d9b715bc2e959e2e59c6a7431fc8f52ba6e3063c0a78f3.jpg)

![eaacde52e44bc2294ad1798a1520fbf29afa20e51e6ef8ef95c86ea3124f60b2.jpg](../iclr_results/96_Problem-Parameter-Free%20Federated%20Learning/images/eaacde52e44bc2294ad1798a1520fbf29afa20e51e6ef8ef95c86ea3124f60b2.jpg)

### Tables

![775118e3f449029f8912243711fbd0ce51a28b50a890f3e2969f2084f3de8a62.jpg](../iclr_results/96_Problem-Parameter-Free%20Federated%20Learning/tables/775118e3f449029f8912243711fbd0ce51a28b50a890f3e2969f2084f3de8a62.jpg)

## Latent Bayesian Optimization via Autoregressive Normalizing Flows


### Images

![126c430d332c036193c1c651a0e0f04523c5630563427f42be87c3bae335fcec.jpg](../iclr_results/97_Latent%20Bayesian%20Optimization%20via%20Autoregressive%20Normalizing%20Flows/images/126c430d332c036193c1c651a0e0f04523c5630563427f42be87c3bae335fcec.jpg)

![359bcdc682c3e3f0ae841f0f5e26afa1c77aac11f8be4cf55aa5bd7de34b6062.jpg](../iclr_results/97_Latent%20Bayesian%20Optimization%20via%20Autoregressive%20Normalizing%20Flows/images/359bcdc682c3e3f0ae841f0f5e26afa1c77aac11f8be4cf55aa5bd7de34b6062.jpg)

![37f909f7f2cc736a9af8a7f6c4366df1d93afb7e7048d13e07b17776bdddac37.jpg](../iclr_results/97_Latent%20Bayesian%20Optimization%20via%20Autoregressive%20Normalizing%20Flows/images/37f909f7f2cc736a9af8a7f6c4366df1d93afb7e7048d13e07b17776bdddac37.jpg)

![5b4b124f0f90a52b006d60b5121cbf106e022978ad971ee5b80ac3a6c2dbebef.jpg](../iclr_results/97_Latent%20Bayesian%20Optimization%20via%20Autoregressive%20Normalizing%20Flows/images/5b4b124f0f90a52b006d60b5121cbf106e022978ad971ee5b80ac3a6c2dbebef.jpg)

![62c617f621c1f3d54c0efe0bd0d14dbcf0dea978232614418e7c355e48a473c5.jpg](../iclr_results/97_Latent%20Bayesian%20Optimization%20via%20Autoregressive%20Normalizing%20Flows/images/62c617f621c1f3d54c0efe0bd0d14dbcf0dea978232614418e7c355e48a473c5.jpg)

![6e7594f8028804a142c82016132fc951a5fb50f839ca2a3f48058998a4259597.jpg](../iclr_results/97_Latent%20Bayesian%20Optimization%20via%20Autoregressive%20Normalizing%20Flows/images/6e7594f8028804a142c82016132fc951a5fb50f839ca2a3f48058998a4259597.jpg)

![71bbddebaf48e0ff4e868a56f7825a87aaf3aa1280044ebd958299ddf48f70a7.jpg](../iclr_results/97_Latent%20Bayesian%20Optimization%20via%20Autoregressive%20Normalizing%20Flows/images/71bbddebaf48e0ff4e868a56f7825a87aaf3aa1280044ebd958299ddf48f70a7.jpg)

![a8518dbe1d44edac4d16a8e6339d73aef3b5b2c4d3d0a299653455622e1be82f.jpg](../iclr_results/97_Latent%20Bayesian%20Optimization%20via%20Autoregressive%20Normalizing%20Flows/images/a8518dbe1d44edac4d16a8e6339d73aef3b5b2c4d3d0a299653455622e1be82f.jpg)

![ba9a53123119ad014bdad749392a7ba918a8fa9710609e201fd0f4031ba36dd8.jpg](../iclr_results/97_Latent%20Bayesian%20Optimization%20via%20Autoregressive%20Normalizing%20Flows/images/ba9a53123119ad014bdad749392a7ba918a8fa9710609e201fd0f4031ba36dd8.jpg)

![c1828673545d26172795867d40d2f780b01621aa0d80358af706af7969e54715.jpg](../iclr_results/97_Latent%20Bayesian%20Optimization%20via%20Autoregressive%20Normalizing%20Flows/images/c1828673545d26172795867d40d2f780b01621aa0d80358af706af7969e54715.jpg)

![c23cb4ca6de011d0639d3324f0ca9831697b0cdde69ff43759dfac9867dff8c9.jpg](../iclr_results/97_Latent%20Bayesian%20Optimization%20via%20Autoregressive%20Normalizing%20Flows/images/c23cb4ca6de011d0639d3324f0ca9831697b0cdde69ff43759dfac9867dff8c9.jpg)

![d373e4327729cf4b47f7f0a85eaad0c12b56cdec7e5cc4e27c78fb5c7b822391.jpg](../iclr_results/97_Latent%20Bayesian%20Optimization%20via%20Autoregressive%20Normalizing%20Flows/images/d373e4327729cf4b47f7f0a85eaad0c12b56cdec7e5cc4e27c78fb5c7b822391.jpg)

![d45a014affdff3bec6867819ee267fc6ca2e5fe3ef72293964e7e0997165e740.jpg](../iclr_results/97_Latent%20Bayesian%20Optimization%20via%20Autoregressive%20Normalizing%20Flows/images/d45a014affdff3bec6867819ee267fc6ca2e5fe3ef72293964e7e0997165e740.jpg)

![fc266a74ae1db415d50585237d3a8300d45e8e7395dc245c7dbeabb8b663016d.jpg](../iclr_results/97_Latent%20Bayesian%20Optimization%20via%20Autoregressive%20Normalizing%20Flows/images/fc266a74ae1db415d50585237d3a8300d45e8e7395dc245c7dbeabb8b663016d.jpg)

### Tables

![16e64dc0c8e3119e7cd98a477b629799a13a7cd74cb866dd9f9fcd72a5b01840.jpg](../iclr_results/97_Latent%20Bayesian%20Optimization%20via%20Autoregressive%20Normalizing%20Flows/tables/16e64dc0c8e3119e7cd98a477b629799a13a7cd74cb866dd9f9fcd72a5b01840.jpg)

![28a90e12ca12a8d3f5bb7dd1b2314fddc5c19af948bfa6609515f628c08802bf.jpg](../iclr_results/97_Latent%20Bayesian%20Optimization%20via%20Autoregressive%20Normalizing%20Flows/tables/28a90e12ca12a8d3f5bb7dd1b2314fddc5c19af948bfa6609515f628c08802bf.jpg)

![2b41eb41cdd74b24a0fe9127d991b2c2172efceedec071a54386b525e15c5b3a.jpg](../iclr_results/97_Latent%20Bayesian%20Optimization%20via%20Autoregressive%20Normalizing%20Flows/tables/2b41eb41cdd74b24a0fe9127d991b2c2172efceedec071a54386b525e15c5b3a.jpg)

![32e430ce680676f4a7f462affbccd96fe2f162e059a3fac1ef0cd03cb1bc4415.jpg](../iclr_results/97_Latent%20Bayesian%20Optimization%20via%20Autoregressive%20Normalizing%20Flows/tables/32e430ce680676f4a7f462affbccd96fe2f162e059a3fac1ef0cd03cb1bc4415.jpg)

![33554761a1d034e47ce074878e0eea37d00948b0647a9702c007dd9b01abe589.jpg](../iclr_results/97_Latent%20Bayesian%20Optimization%20via%20Autoregressive%20Normalizing%20Flows/tables/33554761a1d034e47ce074878e0eea37d00948b0647a9702c007dd9b01abe589.jpg)

![5021c133730876a726f8cd587ecdbaadd6fa8e8dc080e6aef02937ae6bb910e9.jpg](../iclr_results/97_Latent%20Bayesian%20Optimization%20via%20Autoregressive%20Normalizing%20Flows/tables/5021c133730876a726f8cd587ecdbaadd6fa8e8dc080e6aef02937ae6bb910e9.jpg)

![6fae10cf80dff11a33b7249b048dc79398ea3af89481399570733fef0c4913ce.jpg](../iclr_results/97_Latent%20Bayesian%20Optimization%20via%20Autoregressive%20Normalizing%20Flows/tables/6fae10cf80dff11a33b7249b048dc79398ea3af89481399570733fef0c4913ce.jpg)

![8bcd731dfdb0969b5c709d3eb926ff83ad403040528905cdae626a66fb5e1ec7.jpg](../iclr_results/97_Latent%20Bayesian%20Optimization%20via%20Autoregressive%20Normalizing%20Flows/tables/8bcd731dfdb0969b5c709d3eb926ff83ad403040528905cdae626a66fb5e1ec7.jpg)

![aa59ba804ee0aecdb5836ba9818ce886fe611110cce4cac425a609f202c5dafe.jpg](../iclr_results/97_Latent%20Bayesian%20Optimization%20via%20Autoregressive%20Normalizing%20Flows/tables/aa59ba804ee0aecdb5836ba9818ce886fe611110cce4cac425a609f202c5dafe.jpg)

## BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions


### Images

![070d2b4fc03922c72ff3f68c9eac244e53d2a198ec5bd0a03d570105813312c9.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/070d2b4fc03922c72ff3f68c9eac244e53d2a198ec5bd0a03d570105813312c9.jpg)

![0fe30d92c15b0d42be5a965f4807c9b40be967ed212e1f5180e33fb4eb3d2179.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/0fe30d92c15b0d42be5a965f4807c9b40be967ed212e1f5180e33fb4eb3d2179.jpg)

![1e84d9fe4dfa2cf96bd5cce2c2b19136aefe12e5be23dfbdad72d12e0f4cf194.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/1e84d9fe4dfa2cf96bd5cce2c2b19136aefe12e5be23dfbdad72d12e0f4cf194.jpg)

![243625518fb67b5a049b479e5e684923d5bf5ac9bf67e34bca89a50e9f6e0eb2.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/243625518fb67b5a049b479e5e684923d5bf5ac9bf67e34bca89a50e9f6e0eb2.jpg)

![2f9b17437e0f8750faf0a88b32da337ecd9723af2c1882947e38321390d34468.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/2f9b17437e0f8750faf0a88b32da337ecd9723af2c1882947e38321390d34468.jpg)

![35a4b80fb6a2bdbace05ec97a81cfc92fa8d7e2a4b6bd907fab227389122d810.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/35a4b80fb6a2bdbace05ec97a81cfc92fa8d7e2a4b6bd907fab227389122d810.jpg)

![3ee10509f7a518f7bbcbd278ac22134dc403f30834b2f8b379e6b375aa5b202e.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/3ee10509f7a518f7bbcbd278ac22134dc403f30834b2f8b379e6b375aa5b202e.jpg)

![3f2383ebce9df5d8aefe7cc20d657bcfe922513dc2a4a5f07d277c71495008a5.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/3f2383ebce9df5d8aefe7cc20d657bcfe922513dc2a4a5f07d277c71495008a5.jpg)

![4091280c27c43fca7eb360a509524d62de82b1d0d6e3e6f37fb0bc0041491558.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/4091280c27c43fca7eb360a509524d62de82b1d0d6e3e6f37fb0bc0041491558.jpg)

![47e46e6b762114c70d87711412091fb7b74a50d5710cc521def2e2d6eea87a2e.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/47e46e6b762114c70d87711412091fb7b74a50d5710cc521def2e2d6eea87a2e.jpg)

![4d861061719f34443e14ec7eb7eeb651b98325b654706aa55da8e512ad365eb0.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/4d861061719f34443e14ec7eb7eeb651b98325b654706aa55da8e512ad365eb0.jpg)

![60d073231fa3c3dd7f72b7e5aded223cc9cff364133c774a627e8c9ac1b764ad.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/60d073231fa3c3dd7f72b7e5aded223cc9cff364133c774a627e8c9ac1b764ad.jpg)

![689b1ed2a9334e2fe38f4692af35421af17a27318b2ec0959db5a0bc46e98a7a.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/689b1ed2a9334e2fe38f4692af35421af17a27318b2ec0959db5a0bc46e98a7a.jpg)

![72cd2e23fb39e7c3317f6d61905e022f6a1c850c72527abf9b169f11a7e86252.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/72cd2e23fb39e7c3317f6d61905e022f6a1c850c72527abf9b169f11a7e86252.jpg)

![7dc29537054d1f574c4a69c65df877ac81ef94c00633afe1fff41946a44d1027.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/7dc29537054d1f574c4a69c65df877ac81ef94c00633afe1fff41946a44d1027.jpg)

![88d5a01b03ffd4b8edbc34d421475cd0ab526b80c9d68db87f03067296b209bd.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/88d5a01b03ffd4b8edbc34d421475cd0ab526b80c9d68db87f03067296b209bd.jpg)

![8a02ddd96949c670259913c4b8ab529db3c54b29de0365a7e9f2c034d42ec768.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/8a02ddd96949c670259913c4b8ab529db3c54b29de0365a7e9f2c034d42ec768.jpg)

![8ecbaafdfef69d43d4b2cf3289d2e2e2b9081441a88a5159010b1332b2b9882b.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/8ecbaafdfef69d43d4b2cf3289d2e2e2b9081441a88a5159010b1332b2b9882b.jpg)

![91fcb31124157f70b168caf4788cfa24416965d0a43ad75698ff3af42d9385f1.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/91fcb31124157f70b168caf4788cfa24416965d0a43ad75698ff3af42d9385f1.jpg)

![93e40f6e3985ea672f29b393e6ccc49042ad336af1506c2b764e0d80f324bccc.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/93e40f6e3985ea672f29b393e6ccc49042ad336af1506c2b764e0d80f324bccc.jpg)

![a34d84a0ec1a7b835447c919ba431ad8bcf2ba9ec197d6f249f63403a846d5e1.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/a34d84a0ec1a7b835447c919ba431ad8bcf2ba9ec197d6f249f63403a846d5e1.jpg)

![a38d4ea8df0fc08b80e2de4c34ca8abb62ca8c785a4411dd740011be03095fba.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/a38d4ea8df0fc08b80e2de4c34ca8abb62ca8c785a4411dd740011be03095fba.jpg)

![b22ece808264c02e7db743a7b47a4fe3f7bacf7a523ba3977f5ef87269b3d206.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/b22ece808264c02e7db743a7b47a4fe3f7bacf7a523ba3977f5ef87269b3d206.jpg)

![b400f60222a36fe3fb64f6982e225fc6fc3f430d3f15c2c515ba110c2b3ea9c2.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/b400f60222a36fe3fb64f6982e225fc6fc3f430d3f15c2c515ba110c2b3ea9c2.jpg)

![b4d156d87a17fac976991ee3959a4441da44822eba6fa56b10a7dfd18569be39.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/b4d156d87a17fac976991ee3959a4441da44822eba6fa56b10a7dfd18569be39.jpg)

![c6c6863d1b8413b58cea27928fea2b7dad13e2ab3fa58d0f1678bff496c52797.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/c6c6863d1b8413b58cea27928fea2b7dad13e2ab3fa58d0f1678bff496c52797.jpg)

![cbd837a55c90f90b00d72bc3d1317afaebfbf9fdff7b9d7d1552860c08e25a4e.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/cbd837a55c90f90b00d72bc3d1317afaebfbf9fdff7b9d7d1552860c08e25a4e.jpg)

![d4953b32e68a9491cde1f94806bb5a63d8a39dc524dfa8a3d13cace9f8b3a078.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/d4953b32e68a9491cde1f94806bb5a63d8a39dc524dfa8a3d13cace9f8b3a078.jpg)

![d7a33f22cc6b0f7f67dcec3bceacf9eebbaf1d5301eea090739d9e654498e16f.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/d7a33f22cc6b0f7f67dcec3bceacf9eebbaf1d5301eea090739d9e654498e16f.jpg)

![e21b8be3ef263d9224e646e19a458405d86439b0cfd078f22500c1376603c6fa.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/e21b8be3ef263d9224e646e19a458405d86439b0cfd078f22500c1376603c6fa.jpg)

![e3536c2fd466cd00bd910b1111f3f2ec85a090e6349d81deb53e7ffcdb6afc20.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/e3536c2fd466cd00bd910b1111f3f2ec85a090e6349d81deb53e7ffcdb6afc20.jpg)

![edc50634dbe455b9a6e4b84665fe5811d44638639c5490aaa7aba6a9df7db67b.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/edc50634dbe455b9a6e4b84665fe5811d44638639c5490aaa7aba6a9df7db67b.jpg)

![f4e6b345ea2c2f61e5044824688b3536d37f8fa6f4df460beb9ac95e538b4e84.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/f4e6b345ea2c2f61e5044824688b3536d37f8fa6f4df460beb9ac95e538b4e84.jpg)

![f6dce287c56bb177d2fa20af6def9544dcbbd38962fa116ced00e709675dc180.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/f6dce287c56bb177d2fa20af6def9544dcbbd38962fa116ced00e709675dc180.jpg)

![f745991d578b2babeccee95a10450a4e9ea0f895f052bfe15defa2530bfd5d28.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/images/f745991d578b2babeccee95a10450a4e9ea0f895f052bfe15defa2530bfd5d28.jpg)

### Tables

![0b61a9360bc716b25bba6f78fceec34bfc9fba8e3e641b05a6a4318171db96e2.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/tables/0b61a9360bc716b25bba6f78fceec34bfc9fba8e3e641b05a6a4318171db96e2.jpg)

![117f3734cf78786af99ccda4a40e4399bd80ed4308e050cf2a52ba182bfd417b.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/tables/117f3734cf78786af99ccda4a40e4399bd80ed4308e050cf2a52ba182bfd417b.jpg)

![1d74fa8e63a57f462ee9b1b516b79234b81f079c498cb3a4708bf7486c78cc72.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/tables/1d74fa8e63a57f462ee9b1b516b79234b81f079c498cb3a4708bf7486c78cc72.jpg)

![21b3bc47689a97b63a101d85f27b0bdf2860f517350b7acb7ed14d6a1b796305.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/tables/21b3bc47689a97b63a101d85f27b0bdf2860f517350b7acb7ed14d6a1b796305.jpg)

![25a92f872126ccbab75b4c592b7ddcd95b2b29ed205ce522b50d82d891b3374e.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/tables/25a92f872126ccbab75b4c592b7ddcd95b2b29ed205ce522b50d82d891b3374e.jpg)

![32240e7169bdf4a735ee74f79d89d7e0307f8f982853b972a33a6c71fac67fd3.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/tables/32240e7169bdf4a735ee74f79d89d7e0307f8f982853b972a33a6c71fac67fd3.jpg)

![48b38942e34649c8e516a25f890ce0f3d091ae5846d7e69ef443fce0c49c24ab.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/tables/48b38942e34649c8e516a25f890ce0f3d091ae5846d7e69ef443fce0c49c24ab.jpg)

![5cc676523b61ae3b0aba9c85fb9ff331d074ff91393c129348e7268d42d038f8.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/tables/5cc676523b61ae3b0aba9c85fb9ff331d074ff91393c129348e7268d42d038f8.jpg)

![5d8a53e1fa105471d32ff76ea5fb4d598b49d12284fecd89dc7b07ea41c20663.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/tables/5d8a53e1fa105471d32ff76ea5fb4d598b49d12284fecd89dc7b07ea41c20663.jpg)

![82a07741dbc712cfce7ee3d7106ae3cf2e26b346094d5ea9a784643a8f071463.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/tables/82a07741dbc712cfce7ee3d7106ae3cf2e26b346094d5ea9a784643a8f071463.jpg)

![bf0fb93748ea7f947e0e7ea159927121aa22bea73c012ba29d630d880b1d4353.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/tables/bf0fb93748ea7f947e0e7ea159927121aa22bea73c012ba29d630d880b1d4353.jpg)

![c6be5ba63eb21e3bfa908721d83de5dfb27a260d0645a47d9b02afd7966839c4.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/tables/c6be5ba63eb21e3bfa908721d83de5dfb27a260d0645a47d9b02afd7966839c4.jpg)

![f2bb31911f247891f4740c4f3aa276b51738fa91bbd64113596f2c928d84dd49.jpg](../iclr_results/98_BigCodeBench_%20Benchmarking%20Code%20Generation%20with%20Diverse%20Function%20Calls%20and%20Complex%20Instructions/tables/f2bb31911f247891f4740c4f3aa276b51738fa91bbd64113596f2c928d84dd49.jpg)

## ReGenesis: LLMs can Grow into Reasoning Generalists via Self-Improvement


### Images

![2f01f33b7146fc324812683c8e0131c21bacb4f9c4594022e4c1fe912f2c763b.jpg](../iclr_results/99_ReGenesis_%20LLMs%20can%20Grow%20into%20Reasoning%20Generalists%20via%20Self-Improvement/images/2f01f33b7146fc324812683c8e0131c21bacb4f9c4594022e4c1fe912f2c763b.jpg)

![48f022e33dc568c818417914d89e8bfb611e136bf5e434d9d6f0b8333cfbe37e.jpg](../iclr_results/99_ReGenesis_%20LLMs%20can%20Grow%20into%20Reasoning%20Generalists%20via%20Self-Improvement/images/48f022e33dc568c818417914d89e8bfb611e136bf5e434d9d6f0b8333cfbe37e.jpg)

![49d7d6a372b1689c683495b66f8e4f4fd2baad6d3b6e6b207974bdb5d781bdb2.jpg](../iclr_results/99_ReGenesis_%20LLMs%20can%20Grow%20into%20Reasoning%20Generalists%20via%20Self-Improvement/images/49d7d6a372b1689c683495b66f8e4f4fd2baad6d3b6e6b207974bdb5d781bdb2.jpg)

![8823befcc3b5a47ddb927963938a3458a3e6de0c9db00df6f3d523d6355f7b01.jpg](../iclr_results/99_ReGenesis_%20LLMs%20can%20Grow%20into%20Reasoning%20Generalists%20via%20Self-Improvement/images/8823befcc3b5a47ddb927963938a3458a3e6de0c9db00df6f3d523d6355f7b01.jpg)

### Tables

![0e27685c865ea823009b0c9b0827c58f8af29a91d3d1f33e96864d0f7d40c1a9.jpg](../iclr_results/99_ReGenesis_%20LLMs%20can%20Grow%20into%20Reasoning%20Generalists%20via%20Self-Improvement/tables/0e27685c865ea823009b0c9b0827c58f8af29a91d3d1f33e96864d0f7d40c1a9.jpg)

![38b9dcd76d9c2bbfcc6b2ff44fff73aa66798c7c3bcfe88a94b1e6730147e7ed.jpg](../iclr_results/99_ReGenesis_%20LLMs%20can%20Grow%20into%20Reasoning%20Generalists%20via%20Self-Improvement/tables/38b9dcd76d9c2bbfcc6b2ff44fff73aa66798c7c3bcfe88a94b1e6730147e7ed.jpg)

![47079b7bdca932b91fe300097d1e705d926a7fff089b491a7ed03b76a4caf9f8.jpg](../iclr_results/99_ReGenesis_%20LLMs%20can%20Grow%20into%20Reasoning%20Generalists%20via%20Self-Improvement/tables/47079b7bdca932b91fe300097d1e705d926a7fff089b491a7ed03b76a4caf9f8.jpg)

![532dd51a823c71aa22c572076da0ccc629b437b73c5b75229eb1664495b714c1.jpg](../iclr_results/99_ReGenesis_%20LLMs%20can%20Grow%20into%20Reasoning%20Generalists%20via%20Self-Improvement/tables/532dd51a823c71aa22c572076da0ccc629b437b73c5b75229eb1664495b714c1.jpg)

![5c7e8d62eadaa74eba371f63c134ddf91181866308ae63e350c4c7849efe703b.jpg](../iclr_results/99_ReGenesis_%20LLMs%20can%20Grow%20into%20Reasoning%20Generalists%20via%20Self-Improvement/tables/5c7e8d62eadaa74eba371f63c134ddf91181866308ae63e350c4c7849efe703b.jpg)

![6d8a40f5193ee0ffb14729bbdd34b1ce33143967a38b4447519d9ade604e7a0f.jpg](../iclr_results/99_ReGenesis_%20LLMs%20can%20Grow%20into%20Reasoning%20Generalists%20via%20Self-Improvement/tables/6d8a40f5193ee0ffb14729bbdd34b1ce33143967a38b4447519d9ade604e7a0f.jpg)

![7d630679ef1afedd920b3e332f640ca89b958e96e2aec7c9808f26f10fa442e1.jpg](../iclr_results/99_ReGenesis_%20LLMs%20can%20Grow%20into%20Reasoning%20Generalists%20via%20Self-Improvement/tables/7d630679ef1afedd920b3e332f640ca89b958e96e2aec7c9808f26f10fa442e1.jpg)

![9c70ad5120c490f3a112f7edc39b2a625c9ae698dfcdbbe6f4c338f8cda4edc7.jpg](../iclr_results/99_ReGenesis_%20LLMs%20can%20Grow%20into%20Reasoning%20Generalists%20via%20Self-Improvement/tables/9c70ad5120c490f3a112f7edc39b2a625c9ae698dfcdbbe6f4c338f8cda4edc7.jpg)

![a3c7a6090d3e0deed3b3c432dd0afbb209c918f402ec4007b5a97c8f225e59f2.jpg](../iclr_results/99_ReGenesis_%20LLMs%20can%20Grow%20into%20Reasoning%20Generalists%20via%20Self-Improvement/tables/a3c7a6090d3e0deed3b3c432dd0afbb209c918f402ec4007b5a97c8f225e59f2.jpg)

![bd60391e7856aed83eeddb5eacfb2756c4ffdcfc4f2a12d25b19dc24df4e57bc.jpg](../iclr_results/99_ReGenesis_%20LLMs%20can%20Grow%20into%20Reasoning%20Generalists%20via%20Self-Improvement/tables/bd60391e7856aed83eeddb5eacfb2756c4ffdcfc4f2a12d25b19dc24df4e57bc.jpg)

![bf8190f2ace9c1f1e0c63b0cabedf25697df851c9401deca503d888c52e9949d.jpg](../iclr_results/99_ReGenesis_%20LLMs%20can%20Grow%20into%20Reasoning%20Generalists%20via%20Self-Improvement/tables/bf8190f2ace9c1f1e0c63b0cabedf25697df851c9401deca503d888c52e9949d.jpg)

![eff8ebbab219404ea7acf8097311e7a6ba633c15cf5db0b6baac27a6c706b392.jpg](../iclr_results/99_ReGenesis_%20LLMs%20can%20Grow%20into%20Reasoning%20Generalists%20via%20Self-Improvement/tables/eff8ebbab219404ea7acf8097311e7a6ba633c15cf5db0b6baac27a6c706b392.jpg)

## LaMPlace: Learning to Optimize Cross-Stage Metrics in Macro Placement


### Images

![1e4e0841d969b2f4dd091bf47289c8863e677eb49783e5f087ab964ef101f58a.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/images/1e4e0841d969b2f4dd091bf47289c8863e677eb49783e5f087ab964ef101f58a.jpg)

![335af537744701041586899b5dff93426b6ee96a3e43236c7c620301806dcee2.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/images/335af537744701041586899b5dff93426b6ee96a3e43236c7c620301806dcee2.jpg)

![4e1f36994a7eaae3dc53b77c5c3f68c5810bd159e919beeb60b56016e3e465f9.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/images/4e1f36994a7eaae3dc53b77c5c3f68c5810bd159e919beeb60b56016e3e465f9.jpg)

![5eb32780afb1eb50f513a85081fde820c76f5ec35ad5dbfbbdbfdd1c6f6fe10c.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/images/5eb32780afb1eb50f513a85081fde820c76f5ec35ad5dbfbbdbfdd1c6f6fe10c.jpg)

![6cecc2b876469d54f51efd084c9a198f4cca60513942400b99a624aea21d458c.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/images/6cecc2b876469d54f51efd084c9a198f4cca60513942400b99a624aea21d458c.jpg)

![743be71ba69dd42490b2d1ccceb85672174fc9f4f37264f92054550ba50a59c6.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/images/743be71ba69dd42490b2d1ccceb85672174fc9f4f37264f92054550ba50a59c6.jpg)

![794a4ee534f3f51300f4b5a09eec63887fe7bb5b90031d21767e9dacfec0a9bb.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/images/794a4ee534f3f51300f4b5a09eec63887fe7bb5b90031d21767e9dacfec0a9bb.jpg)

![8245d68a1ed10d97303b07c5f22eea8c5203279b5a9fa7d517c8e063652f5895.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/images/8245d68a1ed10d97303b07c5f22eea8c5203279b5a9fa7d517c8e063652f5895.jpg)

![8ba3f90dd46cf7aab48826774312c4888bd73e1303d0b45ccd0aa60140dcb07e.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/images/8ba3f90dd46cf7aab48826774312c4888bd73e1303d0b45ccd0aa60140dcb07e.jpg)

![8d81b13f07cc5938f57c0c348321bb44709de18dba3bd435b7ff1f7878f14ab2.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/images/8d81b13f07cc5938f57c0c348321bb44709de18dba3bd435b7ff1f7878f14ab2.jpg)

![905ef3f561b28c73c63f577d3038e9d548a805940e21db3b6d2956a52954e180.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/images/905ef3f561b28c73c63f577d3038e9d548a805940e21db3b6d2956a52954e180.jpg)

![9d7af818b485fc07a7511b22d99ddb96c4b18f3305973a8fe4034ffcc78ed19c.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/images/9d7af818b485fc07a7511b22d99ddb96c4b18f3305973a8fe4034ffcc78ed19c.jpg)

![9ee91fe5a4107ea337e8260fbb781be5693782a4a5e3f3ec66b6cc840f4c08fc.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/images/9ee91fe5a4107ea337e8260fbb781be5693782a4a5e3f3ec66b6cc840f4c08fc.jpg)

![a18dd5adb2d89f21e714c146835e521d1162c65a213cbcae8fc4e576bc622936.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/images/a18dd5adb2d89f21e714c146835e521d1162c65a213cbcae8fc4e576bc622936.jpg)

![a75571030bd804461fe33bf16e70dc7f71d5b35a2b3dbbe9ae062bca9896bc6f.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/images/a75571030bd804461fe33bf16e70dc7f71d5b35a2b3dbbe9ae062bca9896bc6f.jpg)

![af27aadd54701b5965665e0e2ad90d6835303adac59a2e2a9ec757c2e6724d62.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/images/af27aadd54701b5965665e0e2ad90d6835303adac59a2e2a9ec757c2e6724d62.jpg)

![c649560f8dcb248515bcf18d5a82eb824aa30c4d72b290d9ae348ca03be34f3f.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/images/c649560f8dcb248515bcf18d5a82eb824aa30c4d72b290d9ae348ca03be34f3f.jpg)

![dd47ccbcf3857fe6116ff84978bf0d286bae8259a4bd3e1795e225d982b2b007.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/images/dd47ccbcf3857fe6116ff84978bf0d286bae8259a4bd3e1795e225d982b2b007.jpg)

![f3ea0f03a9450c835af8c2cc456f9a29463ec781b66cd97358fa5df06fb3a727.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/images/f3ea0f03a9450c835af8c2cc456f9a29463ec781b66cd97358fa5df06fb3a727.jpg)

### Tables

![1c7b0c16093b8638b5069fc21f0897206ff79ba6a0f1be2f91f0654032326526.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/tables/1c7b0c16093b8638b5069fc21f0897206ff79ba6a0f1be2f91f0654032326526.jpg)

![415a4c2763879f2a7f85f097058af8a5584b3847b5245f16631c254b06b326e6.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/tables/415a4c2763879f2a7f85f097058af8a5584b3847b5245f16631c254b06b326e6.jpg)

![557b6bc6cdf455d6fb111eae6316591fbb07cb946914887da5abf51535b925b0.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/tables/557b6bc6cdf455d6fb111eae6316591fbb07cb946914887da5abf51535b925b0.jpg)

![5a22e19f6afb5315115bedfd78a2c62467fbd21e16a0d1aa8ac40f15db03552e.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/tables/5a22e19f6afb5315115bedfd78a2c62467fbd21e16a0d1aa8ac40f15db03552e.jpg)

![643750239145e0f9021a71bfe4123c2f1dc62486267c3ff708a11e8d4358a007.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/tables/643750239145e0f9021a71bfe4123c2f1dc62486267c3ff708a11e8d4358a007.jpg)

![85ebe18cd6d08ddbfd131cab9718a14843fed97836cba49672fc9486279e8ca5.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/tables/85ebe18cd6d08ddbfd131cab9718a14843fed97836cba49672fc9486279e8ca5.jpg)

![aefd66b6eb0bb68cc95031f738f8dbf08ccf2687681e57884199188be3c5e9ca.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/tables/aefd66b6eb0bb68cc95031f738f8dbf08ccf2687681e57884199188be3c5e9ca.jpg)

![b81cf60fdc4d91df3c805d2df5f6936a07f8f00aa5ff9d861a8829f721375fa4.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/tables/b81cf60fdc4d91df3c805d2df5f6936a07f8f00aa5ff9d861a8829f721375fa4.jpg)

![bd9c62a3404de0aa96168e0d4198e1a16a56eb50b40f6cbccb66da4a298fe754.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/tables/bd9c62a3404de0aa96168e0d4198e1a16a56eb50b40f6cbccb66da4a298fe754.jpg)

![ce4da8603947fc1be3ddd7d6e893b2d2bf2405f6d79a4f2009cd7dad123baf13.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/tables/ce4da8603947fc1be3ddd7d6e893b2d2bf2405f6d79a4f2009cd7dad123baf13.jpg)

![da69ea88649769b8ca845dcf21752e79781600bb1bcb6d08af5eb2f03efb64e3.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/tables/da69ea88649769b8ca845dcf21752e79781600bb1bcb6d08af5eb2f03efb64e3.jpg)

![e0746e0b0fd2b086748e77f4e16a4bfe286d0f81aa4182d853a7a6a0ac213ec1.jpg](../iclr_results/100_LaMPlace_%20Learning%20to%20Optimize%20Cross-Stage%20Metrics%20in%20Macro%20Placement/tables/e0746e0b0fd2b086748e77f4e16a4bfe286d0f81aa4182d853a7a6a0ac213ec1.jpg)

## MOS: Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection


### Images

![285b0addba6ee32d0564594578147a9bbf7887ede77e18f6b12a139dff6b40f4.jpg](../iclr_results/101_MOS_%20Model%20Synergy%20for%20Test-Time%20Adaptation%20on%20LiDAR-Based%203D%20Object%20Detection/images/285b0addba6ee32d0564594578147a9bbf7887ede77e18f6b12a139dff6b40f4.jpg)

![402962f9df0df4751ef7c911baa6de21bbbdbe4e980dccb21b070ef5ba28642b.jpg](../iclr_results/101_MOS_%20Model%20Synergy%20for%20Test-Time%20Adaptation%20on%20LiDAR-Based%203D%20Object%20Detection/images/402962f9df0df4751ef7c911baa6de21bbbdbe4e980dccb21b070ef5ba28642b.jpg)

![512dbbaa327d8e9cd3e892c084b41aeaf2fdf475df1f470ddac01c6b2c138293.jpg](../iclr_results/101_MOS_%20Model%20Synergy%20for%20Test-Time%20Adaptation%20on%20LiDAR-Based%203D%20Object%20Detection/images/512dbbaa327d8e9cd3e892c084b41aeaf2fdf475df1f470ddac01c6b2c138293.jpg)

![67a358a643f52ffa288821eb5bf30e869509579344f5b10f8cec8949f143a97b.jpg](../iclr_results/101_MOS_%20Model%20Synergy%20for%20Test-Time%20Adaptation%20on%20LiDAR-Based%203D%20Object%20Detection/images/67a358a643f52ffa288821eb5bf30e869509579344f5b10f8cec8949f143a97b.jpg)

![7b7be5d769f5526f4189d45a6f5ad005b456a2ab30449eae1a506d4b092c2f66.jpg](../iclr_results/101_MOS_%20Model%20Synergy%20for%20Test-Time%20Adaptation%20on%20LiDAR-Based%203D%20Object%20Detection/images/7b7be5d769f5526f4189d45a6f5ad005b456a2ab30449eae1a506d4b092c2f66.jpg)

![8156a1674ebebc0fd6d2e83c78905bf9fec45296259602dc6aad393cfd0eafc0.jpg](../iclr_results/101_MOS_%20Model%20Synergy%20for%20Test-Time%20Adaptation%20on%20LiDAR-Based%203D%20Object%20Detection/images/8156a1674ebebc0fd6d2e83c78905bf9fec45296259602dc6aad393cfd0eafc0.jpg)

![cdf407f2232db8ad80f0bde6989445a5948a55f491295fb4b3b727462b79eea9.jpg](../iclr_results/101_MOS_%20Model%20Synergy%20for%20Test-Time%20Adaptation%20on%20LiDAR-Based%203D%20Object%20Detection/images/cdf407f2232db8ad80f0bde6989445a5948a55f491295fb4b3b727462b79eea9.jpg)

![d7630a527c47fc1fb3a14d475b8327cebae16d903cc5bff1f5291c261c4cd717.jpg](../iclr_results/101_MOS_%20Model%20Synergy%20for%20Test-Time%20Adaptation%20on%20LiDAR-Based%203D%20Object%20Detection/images/d7630a527c47fc1fb3a14d475b8327cebae16d903cc5bff1f5291c261c4cd717.jpg)

![eb2d2cdf18a6663bb53b27174645a3d10e4d90344c5112dc1590b903878c5b3b.jpg](../iclr_results/101_MOS_%20Model%20Synergy%20for%20Test-Time%20Adaptation%20on%20LiDAR-Based%203D%20Object%20Detection/images/eb2d2cdf18a6663bb53b27174645a3d10e4d90344c5112dc1590b903878c5b3b.jpg)

### Tables

![015e84de1d78fabb56f3c1c229dfe2b9d7eaa004c876fdd92d0468492e6f7178.jpg](../iclr_results/101_MOS_%20Model%20Synergy%20for%20Test-Time%20Adaptation%20on%20LiDAR-Based%203D%20Object%20Detection/tables/015e84de1d78fabb56f3c1c229dfe2b9d7eaa004c876fdd92d0468492e6f7178.jpg)

![17809eccd331fe73271cb9586a1db52766ffb46696f72905104b2615156f27cf.jpg](../iclr_results/101_MOS_%20Model%20Synergy%20for%20Test-Time%20Adaptation%20on%20LiDAR-Based%203D%20Object%20Detection/tables/17809eccd331fe73271cb9586a1db52766ffb46696f72905104b2615156f27cf.jpg)

![1f08690124203626f5d7704998f3c2ce8888e54b378b7ebb566e296a1a663543.jpg](../iclr_results/101_MOS_%20Model%20Synergy%20for%20Test-Time%20Adaptation%20on%20LiDAR-Based%203D%20Object%20Detection/tables/1f08690124203626f5d7704998f3c2ce8888e54b378b7ebb566e296a1a663543.jpg)

![223c5260c32b8c5a263674c94d2676e7a2100c3eeab17f7d2bf2eaa7008faa4c.jpg](../iclr_results/101_MOS_%20Model%20Synergy%20for%20Test-Time%20Adaptation%20on%20LiDAR-Based%203D%20Object%20Detection/tables/223c5260c32b8c5a263674c94d2676e7a2100c3eeab17f7d2bf2eaa7008faa4c.jpg)

![60eb49efbaf51618c76bfb55eacd95e81965d866c49adaa8262ea5fad8ede784.jpg](../iclr_results/101_MOS_%20Model%20Synergy%20for%20Test-Time%20Adaptation%20on%20LiDAR-Based%203D%20Object%20Detection/tables/60eb49efbaf51618c76bfb55eacd95e81965d866c49adaa8262ea5fad8ede784.jpg)

![8ec4ef848f99a843ec9f3ff4888a5bb99c4a94901de6ec0836038eef71e65881.jpg](../iclr_results/101_MOS_%20Model%20Synergy%20for%20Test-Time%20Adaptation%20on%20LiDAR-Based%203D%20Object%20Detection/tables/8ec4ef848f99a843ec9f3ff4888a5bb99c4a94901de6ec0836038eef71e65881.jpg)

![923fdb90139dd2ef41ddecf7fe4897fbf6d055b93735f52601acba52c1041853.jpg](../iclr_results/101_MOS_%20Model%20Synergy%20for%20Test-Time%20Adaptation%20on%20LiDAR-Based%203D%20Object%20Detection/tables/923fdb90139dd2ef41ddecf7fe4897fbf6d055b93735f52601acba52c1041853.jpg)

![964b05a2630000b97bc1e7733a2c32541cf7cf585b5ed8c648fc671197eac5b7.jpg](../iclr_results/101_MOS_%20Model%20Synergy%20for%20Test-Time%20Adaptation%20on%20LiDAR-Based%203D%20Object%20Detection/tables/964b05a2630000b97bc1e7733a2c32541cf7cf585b5ed8c648fc671197eac5b7.jpg)

![ba9ae7de130ffd636775ae8035e0daab5921ec1fb0cf3b33554be8e5a016c335.jpg](../iclr_results/101_MOS_%20Model%20Synergy%20for%20Test-Time%20Adaptation%20on%20LiDAR-Based%203D%20Object%20Detection/tables/ba9ae7de130ffd636775ae8035e0daab5921ec1fb0cf3b33554be8e5a016c335.jpg)

![bde8a7ddc1ab84d6c82350d5b8dd9616a0c1e8e74a79523d360e30dd56586d0f.jpg](../iclr_results/101_MOS_%20Model%20Synergy%20for%20Test-Time%20Adaptation%20on%20LiDAR-Based%203D%20Object%20Detection/tables/bde8a7ddc1ab84d6c82350d5b8dd9616a0c1e8e74a79523d360e30dd56586d0f.jpg)

![e0be8b1c462efdf6d622d5e077191e5ff22466218f50afc9904fcfbefa6f18d7.jpg](../iclr_results/101_MOS_%20Model%20Synergy%20for%20Test-Time%20Adaptation%20on%20LiDAR-Based%203D%20Object%20Detection/tables/e0be8b1c462efdf6d622d5e077191e5ff22466218f50afc9904fcfbefa6f18d7.jpg)

## On Conformal Isometry of Grid Cells: Learning Distance-Preserving Position Embedding


### Images

![0715f89dda10db7d1fd6cecadf21e34756f05ebb304023685a02f3bad75e5f7f.jpg](../iclr_results/102_On%20Conformal%20Isometry%20of%20Grid%20Cells_%20Learning%20Distance-Preserving%20Position%20Embedding/images/0715f89dda10db7d1fd6cecadf21e34756f05ebb304023685a02f3bad75e5f7f.jpg)

![19b64c8e23acdf71ccefd3436e85464fa2b68a73606a257dd4fda3243fa9399f.jpg](../iclr_results/102_On%20Conformal%20Isometry%20of%20Grid%20Cells_%20Learning%20Distance-Preserving%20Position%20Embedding/images/19b64c8e23acdf71ccefd3436e85464fa2b68a73606a257dd4fda3243fa9399f.jpg)

![1afc3c49d11aa69871630e03ffbb21fd0f76a200a19608bfb86ccd10238d4d5e.jpg](../iclr_results/102_On%20Conformal%20Isometry%20of%20Grid%20Cells_%20Learning%20Distance-Preserving%20Position%20Embedding/images/1afc3c49d11aa69871630e03ffbb21fd0f76a200a19608bfb86ccd10238d4d5e.jpg)

![2ebae99cc4fc713e87af609476135dac5a216fc0cd4936954aedd3ae20cee1c1.jpg](../iclr_results/102_On%20Conformal%20Isometry%20of%20Grid%20Cells_%20Learning%20Distance-Preserving%20Position%20Embedding/images/2ebae99cc4fc713e87af609476135dac5a216fc0cd4936954aedd3ae20cee1c1.jpg)

![318f96bb08b45d7eb03e275303ec4427aaf0f25975944f278310f6c7c9891f0c.jpg](../iclr_results/102_On%20Conformal%20Isometry%20of%20Grid%20Cells_%20Learning%20Distance-Preserving%20Position%20Embedding/images/318f96bb08b45d7eb03e275303ec4427aaf0f25975944f278310f6c7c9891f0c.jpg)

![4a52ae14349f42ae77bf998cd45deb948d0df9a9cab8b9be2b032f9aca6c607a.jpg](../iclr_results/102_On%20Conformal%20Isometry%20of%20Grid%20Cells_%20Learning%20Distance-Preserving%20Position%20Embedding/images/4a52ae14349f42ae77bf998cd45deb948d0df9a9cab8b9be2b032f9aca6c607a.jpg)

![4e51153a04429ea97c94e90eaa7b9e7f8de173571ba60c93cf5af5723f133a78.jpg](../iclr_results/102_On%20Conformal%20Isometry%20of%20Grid%20Cells_%20Learning%20Distance-Preserving%20Position%20Embedding/images/4e51153a04429ea97c94e90eaa7b9e7f8de173571ba60c93cf5af5723f133a78.jpg)

![82974c4203f9cfa74f27ddf5067f8becf3c4e83fb2a8dcb46257067034767f4a.jpg](../iclr_results/102_On%20Conformal%20Isometry%20of%20Grid%20Cells_%20Learning%20Distance-Preserving%20Position%20Embedding/images/82974c4203f9cfa74f27ddf5067f8becf3c4e83fb2a8dcb46257067034767f4a.jpg)

![87bf7b8af37c82911593220457bb2442db1d0ed33b7537a4a300f85946aaed09.jpg](../iclr_results/102_On%20Conformal%20Isometry%20of%20Grid%20Cells_%20Learning%20Distance-Preserving%20Position%20Embedding/images/87bf7b8af37c82911593220457bb2442db1d0ed33b7537a4a300f85946aaed09.jpg)

![9ccd69fb23122fb4c9ba9dd860da701dde1ea8fac6719ed0dc6696ca3ec1982e.jpg](../iclr_results/102_On%20Conformal%20Isometry%20of%20Grid%20Cells_%20Learning%20Distance-Preserving%20Position%20Embedding/images/9ccd69fb23122fb4c9ba9dd860da701dde1ea8fac6719ed0dc6696ca3ec1982e.jpg)

![a658ddacdc4ddf34931121a6f385787b17a1cba2e7a066bd084ca6f461bc9e68.jpg](../iclr_results/102_On%20Conformal%20Isometry%20of%20Grid%20Cells_%20Learning%20Distance-Preserving%20Position%20Embedding/images/a658ddacdc4ddf34931121a6f385787b17a1cba2e7a066bd084ca6f461bc9e68.jpg)

![bc821acbcab4679b2d3b6260d825f3cec9aa8d10ebbbf491f4a622a5596267c1.jpg](../iclr_results/102_On%20Conformal%20Isometry%20of%20Grid%20Cells_%20Learning%20Distance-Preserving%20Position%20Embedding/images/bc821acbcab4679b2d3b6260d825f3cec9aa8d10ebbbf491f4a622a5596267c1.jpg)

![be6ec8daafd38c89eb5df3f969df10a4e3c9521d61f6b96b2483ad29f7b69c72.jpg](../iclr_results/102_On%20Conformal%20Isometry%20of%20Grid%20Cells_%20Learning%20Distance-Preserving%20Position%20Embedding/images/be6ec8daafd38c89eb5df3f969df10a4e3c9521d61f6b96b2483ad29f7b69c72.jpg)

![d8ef0a861ed6a57f92257fef3451195fff37590bdfe2bc1eadb4485566593855.jpg](../iclr_results/102_On%20Conformal%20Isometry%20of%20Grid%20Cells_%20Learning%20Distance-Preserving%20Position%20Embedding/images/d8ef0a861ed6a57f92257fef3451195fff37590bdfe2bc1eadb4485566593855.jpg)

![f1fc8a7c56de80b4da4dcb60efdfdf106084225fe623cbe0b609dc7de21b5847.jpg](../iclr_results/102_On%20Conformal%20Isometry%20of%20Grid%20Cells_%20Learning%20Distance-Preserving%20Position%20Embedding/images/f1fc8a7c56de80b4da4dcb60efdfdf106084225fe623cbe0b609dc7de21b5847.jpg)

### Tables

![2e9e6b6ce0ac44dac444101273da19856f53fcb971aee1e8c60dd184f2a9c6d6.jpg](../iclr_results/102_On%20Conformal%20Isometry%20of%20Grid%20Cells_%20Learning%20Distance-Preserving%20Position%20Embedding/tables/2e9e6b6ce0ac44dac444101273da19856f53fcb971aee1e8c60dd184f2a9c6d6.jpg)

![b03a5534051728ee3b8035814b344df2e5175c7a0ede0c2f4a4d33af099236ea.jpg](../iclr_results/102_On%20Conformal%20Isometry%20of%20Grid%20Cells_%20Learning%20Distance-Preserving%20Position%20Embedding/tables/b03a5534051728ee3b8035814b344df2e5175c7a0ede0c2f4a4d33af099236ea.jpg)

## Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows


### Images

![0b64e93c7ff505ee39593dd90ba541dff695dd5042e948ee236ae54f45966278.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/0b64e93c7ff505ee39593dd90ba541dff695dd5042e948ee236ae54f45966278.jpg)

![0fa8bc0a15b15a91c5c418a39d77f4fc2f5d391e22211e55ec26d34c1c614d95.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/0fa8bc0a15b15a91c5c418a39d77f4fc2f5d391e22211e55ec26d34c1c614d95.jpg)

![1cc2f8b43438d312d98d4d8db9700329212ca51911f000ba5aa05306a701eff7.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/1cc2f8b43438d312d98d4d8db9700329212ca51911f000ba5aa05306a701eff7.jpg)

![27b4a5b3c5411cf961ff87342e74464ceb2ee42b8fa5f25da36cb94f4255b8f6.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/27b4a5b3c5411cf961ff87342e74464ceb2ee42b8fa5f25da36cb94f4255b8f6.jpg)

![4a67b22b87aec2a1c08873b4b7846c257e13fa9e531a25f233a0354531da9648.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/4a67b22b87aec2a1c08873b4b7846c257e13fa9e531a25f233a0354531da9648.jpg)

![59b4f20ce8b7c337b3bfc33dfddadc6aa3dc119ba2e77afea59bf8d921a6e430.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/59b4f20ce8b7c337b3bfc33dfddadc6aa3dc119ba2e77afea59bf8d921a6e430.jpg)

![63bab73432b98dfcfd4024b78ea2409e41a5d1011d262841ae7441d0f2316ee7.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/63bab73432b98dfcfd4024b78ea2409e41a5d1011d262841ae7441d0f2316ee7.jpg)

![640185c8ed3c704675b9b8c09d48dd1b804915771befa3aef0824383bb3221ed.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/640185c8ed3c704675b9b8c09d48dd1b804915771befa3aef0824383bb3221ed.jpg)

![664fbee75c615e0477ae5837654b247f9e5a5e94b3d36e3acdb15e436b93c108.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/664fbee75c615e0477ae5837654b247f9e5a5e94b3d36e3acdb15e436b93c108.jpg)

![6909c98ac3a02b56528f216ad4660f9731f66f6d8462abcde8f3dc596e6a26fe.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/6909c98ac3a02b56528f216ad4660f9731f66f6d8462abcde8f3dc596e6a26fe.jpg)

![73b0dcd9189680d6dc6eae6ad89dce18d07c266713200f426ef7e312a362a900.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/73b0dcd9189680d6dc6eae6ad89dce18d07c266713200f426ef7e312a362a900.jpg)

![74df60501f13668462eb0b020696dcb3e89296471a0e63e1a0742155ddf3cddd.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/74df60501f13668462eb0b020696dcb3e89296471a0e63e1a0742155ddf3cddd.jpg)

![8d5bcc44eeb8ab52e8ec57cb7a7a447aeb6a841a329599205b0a38684f65a7bb.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/8d5bcc44eeb8ab52e8ec57cb7a7a447aeb6a841a329599205b0a38684f65a7bb.jpg)

![9b3e8e76b0e6fad10569166a52dd49471844c2e2f4fd618c50069d202894a7c7.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/9b3e8e76b0e6fad10569166a52dd49471844c2e2f4fd618c50069d202894a7c7.jpg)

![a0557aa43057486fbfb5677f67c45fddd213d8c1dbce348d096548217c28a94c.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/a0557aa43057486fbfb5677f67c45fddd213d8c1dbce348d096548217c28a94c.jpg)

![a0886c486fd2109eb7eeb221ea2a1ef36af73520d8dea2c617e152b06384f47b.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/a0886c486fd2109eb7eeb221ea2a1ef36af73520d8dea2c617e152b06384f47b.jpg)

![a385cc2ad146b7a3bfb1edc118110c6142efb0b1c27a788f782d23dd53c540bc.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/a385cc2ad146b7a3bfb1edc118110c6142efb0b1c27a788f782d23dd53c540bc.jpg)

![a757596e4bf14bd8081b887ccf60d2116b7880201ef318ee43e98408d986b51e.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/a757596e4bf14bd8081b887ccf60d2116b7880201ef318ee43e98408d986b51e.jpg)

![ba0fed8a5f153b4ee49c6b117fc5ae92eb2222c553828d45af43026fef5b6495.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/ba0fed8a5f153b4ee49c6b117fc5ae92eb2222c553828d45af43026fef5b6495.jpg)

![cbeeabf246ad687f9670205cfefed77079805916d0d146244818f65fdf3652dd.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/cbeeabf246ad687f9670205cfefed77079805916d0d146244818f65fdf3652dd.jpg)

![e5464edc0bd301de51c4b64b608d793bd70a80046c2bcd7b220432dfffcf013f.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/e5464edc0bd301de51c4b64b608d793bd70a80046c2bcd7b220432dfffcf013f.jpg)

![eee8fd2388a3dca179018302c67be07809823e237858378207e7739daf94e9cf.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/eee8fd2388a3dca179018302c67be07809823e237858378207e7739daf94e9cf.jpg)

![f1001e30fef00476b1da40f751ba05a745a80a43fd0aa18331c85a3c9e430d33.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/f1001e30fef00476b1da40f751ba05a745a80a43fd0aa18331c85a3c9e430d33.jpg)

![fc4b3bbf7bb12686263e6470cec2b2e09aaef7fc484e49882ab0b0d10f48c6d8.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/images/fc4b3bbf7bb12686263e6470cec2b2e09aaef7fc484e49882ab0b0d10f48c6d8.jpg)

### Tables

![0dd0994684ad0168688f952512f9b59c63ec853368487f248a76f3b51a13704b.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/tables/0dd0994684ad0168688f952512f9b59c63ec853368487f248a76f3b51a13704b.jpg)

![1000a743483e3ddfad13a3c446f427445b6f341ba07864be386e3053b2314b21.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/tables/1000a743483e3ddfad13a3c446f427445b6f341ba07864be386e3053b2314b21.jpg)

![17714a906f5819cf5abe95dfdd1d7b2c2f8d90b27ccd5c77183144b70f12ec8f.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/tables/17714a906f5819cf5abe95dfdd1d7b2c2f8d90b27ccd5c77183144b70f12ec8f.jpg)

![1be87204787de7fd5905d65a09dc2020d6f6000f8bf2b1d29d6f8f8243dea1df.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/tables/1be87204787de7fd5905d65a09dc2020d6f6000f8bf2b1d29d6f8f8243dea1df.jpg)

![2ea14148197eeb6f4919aa737738af30c8aef602fac943fb941f8ba52cfce3fa.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/tables/2ea14148197eeb6f4919aa737738af30c8aef602fac943fb941f8ba52cfce3fa.jpg)

![4ac6b2763fa87fa899513f901762009cd6735e095a06a8e364666928822a8d14.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/tables/4ac6b2763fa87fa899513f901762009cd6735e095a06a8e364666928822a8d14.jpg)

![5070f6baa5f79af774f13548a32fb9e06e19e355442ca7dab3a45bae9b506ebc.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/tables/5070f6baa5f79af774f13548a32fb9e06e19e355442ca7dab3a45bae9b506ebc.jpg)

![53f633e25daea9dd787068934cbb4edf27efb5fe810b2a2dd3f0531f2eb237b8.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/tables/53f633e25daea9dd787068934cbb4edf27efb5fe810b2a2dd3f0531f2eb237b8.jpg)

![690b30bc483f72367ffae3c32bf11cb5419f4854900bdaa7a15523cc23a689b9.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/tables/690b30bc483f72367ffae3c32bf11cb5419f4854900bdaa7a15523cc23a689b9.jpg)

![6bf32b0564380e0483f0daf4bc415bb26cdbd742b502913485ad975a31e05d5e.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/tables/6bf32b0564380e0483f0daf4bc415bb26cdbd742b502913485ad975a31e05d5e.jpg)

![6e640e56df4f2b8bad96315cfb960d6f99198f0c0c67fbee4dcf6e6f31436aeb.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/tables/6e640e56df4f2b8bad96315cfb960d6f99198f0c0c67fbee4dcf6e6f31436aeb.jpg)

![7299378de5edb41095498a1803c7686bf9fa6b4818d9f4ebfaa1d521ae03d9cb.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/tables/7299378de5edb41095498a1803c7686bf9fa6b4818d9f4ebfaa1d521ae03d9cb.jpg)

![7dd5955e124992a821ec84526e3e4507f2f9603c70db0d25ae2d1ceba2276d7d.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/tables/7dd5955e124992a821ec84526e3e4507f2f9603c70db0d25ae2d1ceba2276d7d.jpg)

![a57c6ecd530cd12dca639494305f1898e5cf1451cf5b618b297849a4cff68528.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/tables/a57c6ecd530cd12dca639494305f1898e5cf1451cf5b618b297849a4cff68528.jpg)

![a8d4c3fcb82a7f2bd0ba3244056fea26d4d6589ff2cbe7c8779772c9f489a2f5.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/tables/a8d4c3fcb82a7f2bd0ba3244056fea26d4d6589ff2cbe7c8779772c9f489a2f5.jpg)

![b123c721020b71ac421be45bbc95dc3d760b5395ddab53244ac78b5a3f300680.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/tables/b123c721020b71ac421be45bbc95dc3d760b5395ddab53244ac78b5a3f300680.jpg)

![bea604da6c31fac761100b592baa281087609e077910c624e8a96791c9492800.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/tables/bea604da6c31fac761100b592baa281087609e077910c624e8a96791c9492800.jpg)

![d47ab3d1a9d6024c6e86ac9e29f1bd6aa9fef77c32c29b57639bc388f276f597.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/tables/d47ab3d1a9d6024c6e86ac9e29f1bd6aa9fef77c32c29b57639bc388f276f597.jpg)

![d9077c9dbcfc1e0fd488dce2fb2daa3c2725370d09286df82a5d0353bf05c7af.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/tables/d9077c9dbcfc1e0fd488dce2fb2daa3c2725370d09286df82a5d0353bf05c7af.jpg)

![e8dc4956a07e29816fff6ea2996cfd658a76f5344b3c09f5bf726aad84ac90ec.jpg](../iclr_results/103_Spider%202.0_%20Evaluating%20Language%20Models%20on%20Real-World%20Enterprise%20Text-to-SQL%20Workflows/tables/e8dc4956a07e29816fff6ea2996cfd658a76f5344b3c09f5bf726aad84ac90ec.jpg)

## EmbodiedSAM: Online Segment Any 3D Thing in Real Time


### Images

![18cceb255d5fea0c5be38a8c33684864f270c06b129e4c341d5d56f07611d417.jpg](../iclr_results/104_EmbodiedSAM_%20Online%20Segment%20Any%203D%20Thing%20in%20Real%20Time/images/18cceb255d5fea0c5be38a8c33684864f270c06b129e4c341d5d56f07611d417.jpg)

![25514905f4db6c72d10827103316321d0c3b488b5b28505842f9e6e595d6cd7b.jpg](../iclr_results/104_EmbodiedSAM_%20Online%20Segment%20Any%203D%20Thing%20in%20Real%20Time/images/25514905f4db6c72d10827103316321d0c3b488b5b28505842f9e6e595d6cd7b.jpg)

![4ae8f3e4bed3ffad4a83467c97aa6f82e9785cada8b052a1f67f6a9c3df16082.jpg](../iclr_results/104_EmbodiedSAM_%20Online%20Segment%20Any%203D%20Thing%20in%20Real%20Time/images/4ae8f3e4bed3ffad4a83467c97aa6f82e9785cada8b052a1f67f6a9c3df16082.jpg)

![5c4bce71517dc335203a56aa9624f0e0df033d8e228ea9aed541a9464b0b2e8b.jpg](../iclr_results/104_EmbodiedSAM_%20Online%20Segment%20Any%203D%20Thing%20in%20Real%20Time/images/5c4bce71517dc335203a56aa9624f0e0df033d8e228ea9aed541a9464b0b2e8b.jpg)

![5e356b4768d01d984e0f90a194e3833debdd1efbbe452b05c13cb78f3b28b693.jpg](../iclr_results/104_EmbodiedSAM_%20Online%20Segment%20Any%203D%20Thing%20in%20Real%20Time/images/5e356b4768d01d984e0f90a194e3833debdd1efbbe452b05c13cb78f3b28b693.jpg)

![88bad5f28a6e2fac0019564407bebf779c1e12861310bd7c70d55e9e79a36c46.jpg](../iclr_results/104_EmbodiedSAM_%20Online%20Segment%20Any%203D%20Thing%20in%20Real%20Time/images/88bad5f28a6e2fac0019564407bebf779c1e12861310bd7c70d55e9e79a36c46.jpg)

![8a23f301c980b8bb152dbe28702dcb9ca4f534181836c46809f8404d915dd763.jpg](../iclr_results/104_EmbodiedSAM_%20Online%20Segment%20Any%203D%20Thing%20in%20Real%20Time/images/8a23f301c980b8bb152dbe28702dcb9ca4f534181836c46809f8404d915dd763.jpg)

![fa3d3a32aec3dedf5d16880bdc85e89c2881a7b2b40b419e75a9e3972d10591e.jpg](../iclr_results/104_EmbodiedSAM_%20Online%20Segment%20Any%203D%20Thing%20in%20Real%20Time/images/fa3d3a32aec3dedf5d16880bdc85e89c2881a7b2b40b419e75a9e3972d10591e.jpg)

### Tables

![20dbb9db9f1ea0d41c3a0fa6484e2dcf32a1370b48d7994d85e98e13705966c2.jpg](../iclr_results/104_EmbodiedSAM_%20Online%20Segment%20Any%203D%20Thing%20in%20Real%20Time/tables/20dbb9db9f1ea0d41c3a0fa6484e2dcf32a1370b48d7994d85e98e13705966c2.jpg)

![2a0bb680e15b95d81fe5223cdf29e9170c592742d56b77901340b361ebc90a79.jpg](../iclr_results/104_EmbodiedSAM_%20Online%20Segment%20Any%203D%20Thing%20in%20Real%20Time/tables/2a0bb680e15b95d81fe5223cdf29e9170c592742d56b77901340b361ebc90a79.jpg)

![4afd7a338ec4796a8be816dc0e5252150f733b360b19e3ed68424fd8a19b2091.jpg](../iclr_results/104_EmbodiedSAM_%20Online%20Segment%20Any%203D%20Thing%20in%20Real%20Time/tables/4afd7a338ec4796a8be816dc0e5252150f733b360b19e3ed68424fd8a19b2091.jpg)

![65802e00e6ffce7bc9faff9879c26f13840493c168a2e178403475aa0667323e.jpg](../iclr_results/104_EmbodiedSAM_%20Online%20Segment%20Any%203D%20Thing%20in%20Real%20Time/tables/65802e00e6ffce7bc9faff9879c26f13840493c168a2e178403475aa0667323e.jpg)

![716090caef0d1c1979a9466ae1e92bf266fb9eb1670c4814527caa63b32908a8.jpg](../iclr_results/104_EmbodiedSAM_%20Online%20Segment%20Any%203D%20Thing%20in%20Real%20Time/tables/716090caef0d1c1979a9466ae1e92bf266fb9eb1670c4814527caa63b32908a8.jpg)

![77dcd7376696aaab8f9fab9f844d7a7e038ffe8312f5571c03209012b3ac277f.jpg](../iclr_results/104_EmbodiedSAM_%20Online%20Segment%20Any%203D%20Thing%20in%20Real%20Time/tables/77dcd7376696aaab8f9fab9f844d7a7e038ffe8312f5571c03209012b3ac277f.jpg)

![cc5cb490871e583323c713218054031f7e8d1e8ebd584dbde0355990b11b0bc4.jpg](../iclr_results/104_EmbodiedSAM_%20Online%20Segment%20Any%203D%20Thing%20in%20Real%20Time/tables/cc5cb490871e583323c713218054031f7e8d1e8ebd584dbde0355990b11b0bc4.jpg)

![db73aa39469eb4ce31fd8e274a615e8accb2bb22c15636fbbdef9f63fef93204.jpg](../iclr_results/104_EmbodiedSAM_%20Online%20Segment%20Any%203D%20Thing%20in%20Real%20Time/tables/db73aa39469eb4ce31fd8e274a615e8accb2bb22c15636fbbdef9f63fef93204.jpg)

![e20451998ef6ed954d6be86188dc9ecaca61e70604a3a3f3b71eab74a6d3f433.jpg](../iclr_results/104_EmbodiedSAM_%20Online%20Segment%20Any%203D%20Thing%20in%20Real%20Time/tables/e20451998ef6ed954d6be86188dc9ecaca61e70604a3a3f3b71eab74a6d3f433.jpg)

## Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping


### Images

![1e0cb395927d1424ed4d7acb7013ccc3148c9bbbcd8327483cf33841d9eb37e2.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/images/1e0cb395927d1424ed4d7acb7013ccc3148c9bbbcd8327483cf33841d9eb37e2.jpg)

![2c00a38434eee660a79549c9121a43dd732b563a05d30296172774367ac78c60.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/images/2c00a38434eee660a79549c9121a43dd732b563a05d30296172774367ac78c60.jpg)

![32293f002341d3ca1159ed19854e123d8ee125daa1447acf5c6fd2df19943fa2.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/images/32293f002341d3ca1159ed19854e123d8ee125daa1447acf5c6fd2df19943fa2.jpg)

![496f8dbc0ef3eaaa1b3832b0e484980eb8111a82fdc81b17f3dc65d6f39e7839.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/images/496f8dbc0ef3eaaa1b3832b0e484980eb8111a82fdc81b17f3dc65d6f39e7839.jpg)

![50d0ef4aefb212f1d13e922d29430945e2b38e77be3e67900a0ddc2cb889e7a6.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/images/50d0ef4aefb212f1d13e922d29430945e2b38e77be3e67900a0ddc2cb889e7a6.jpg)

![602add0e7be4a410052e0b2f08b3aecd10f0035163ac6ef3172fc8b3fc64bbf3.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/images/602add0e7be4a410052e0b2f08b3aecd10f0035163ac6ef3172fc8b3fc64bbf3.jpg)

![9980e826dfc6768fff2b5484868561f51d5c4ca3f62f343e507190b549fb7407.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/images/9980e826dfc6768fff2b5484868561f51d5c4ca3f62f343e507190b549fb7407.jpg)

![ab6e0d8a6e67b1abf9e88401f71f04caafb6f6660da4277502aa6b882fa336bf.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/images/ab6e0d8a6e67b1abf9e88401f71f04caafb6f6660da4277502aa6b882fa336bf.jpg)

![c8a8f93da1583f058496e8eb488b984f70142e88fb4ee07822c4715f4d7abad8.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/images/c8a8f93da1583f058496e8eb488b984f70142e88fb4ee07822c4715f4d7abad8.jpg)

![cb3f37108809f4031806a6df81424a9408eceee7d79bbd72927bb19acb2f6e3d.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/images/cb3f37108809f4031806a6df81424a9408eceee7d79bbd72927bb19acb2f6e3d.jpg)

![d4dbdbe9eb675aeafdc5ce7b1b36b300c4e3b4dcdfcf250a1a9784c69dd2fdf4.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/images/d4dbdbe9eb675aeafdc5ce7b1b36b300c4e3b4dcdfcf250a1a9784c69dd2fdf4.jpg)

![e4727778ba6e88c26a5917579f86c3592653201ed7b7f7311d827f01fc70c60a.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/images/e4727778ba6e88c26a5917579f86c3592653201ed7b7f7311d827f01fc70c60a.jpg)

![eb0b0e9dbcfa5edfb76aa202f26560f9aec1fa90cda1f23f0dae1eebf1e3e833.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/images/eb0b0e9dbcfa5edfb76aa202f26560f9aec1fa90cda1f23f0dae1eebf1e3e833.jpg)

![ef9771c9776ca985a0dded512112560d9c615bdae0e1fec4b3b3fcef4f128fa1.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/images/ef9771c9776ca985a0dded512112560d9c615bdae0e1fec4b3b3fcef4f128fa1.jpg)

![f7f42f716cf917d302b23cbf2382ee36314604397293cb450c8dc517f3b424f0.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/images/f7f42f716cf917d302b23cbf2382ee36314604397293cb450c8dc517f3b424f0.jpg)

### Tables

![19122ef7175aa7cb539a38b35d034d6f09ed67869ebc1aadd4a3e4ed0791a619.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/tables/19122ef7175aa7cb539a38b35d034d6f09ed67869ebc1aadd4a3e4ed0791a619.jpg)

![1bbace4cc14594766966febda52063606f7f5674c258fd5d402b97b7704e5060.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/tables/1bbace4cc14594766966febda52063606f7f5674c258fd5d402b97b7704e5060.jpg)

![32249ef32eabcc800902cd3620ccf7dd8c1d85386ebdad25eae0305021c860c6.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/tables/32249ef32eabcc800902cd3620ccf7dd8c1d85386ebdad25eae0305021c860c6.jpg)

![4901a4d1447fd4265426cac30b02f9603c6fdfed77584e4ecdc7d9e2555b9265.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/tables/4901a4d1447fd4265426cac30b02f9603c6fdfed77584e4ecdc7d9e2555b9265.jpg)

![59a3a8ffe0deed67f212b1ef36ed85ef4787569cff6a8ec5ca299bded3baa5fb.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/tables/59a3a8ffe0deed67f212b1ef36ed85ef4787569cff6a8ec5ca299bded3baa5fb.jpg)

![66bfdc0e96d3e44193e52540f994938adcd43e4dffdbf967a35726b37290cd6b.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/tables/66bfdc0e96d3e44193e52540f994938adcd43e4dffdbf967a35726b37290cd6b.jpg)

![93ca568701b61071f222981e114507179a75de06e4415c9ecd9037518b64aa62.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/tables/93ca568701b61071f222981e114507179a75de06e4415c9ecd9037518b64aa62.jpg)

![9e5ca0b5c038da03868c75b24834ad47400109b507b87d326a5a0f1ae427328d.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/tables/9e5ca0b5c038da03868c75b24834ad47400109b507b87d326a5a0f1ae427328d.jpg)

![b6ae2c115b4cecdd9f66dbe463b9b6649c004a59f183bbce814ca51a5876f22c.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/tables/b6ae2c115b4cecdd9f66dbe463b9b6649c004a59f183bbce814ca51a5876f22c.jpg)

![c296304c20e308bd4f748fd2666895a1c29d4249a89b30da25af27a7f703daf3.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/tables/c296304c20e308bd4f748fd2666895a1c29d4249a89b30da25af27a7f703daf3.jpg)

![ce64bd87a58da157b30a4b6adc4e521bf8ae495b6c724f370f9e4183e9a9cf51.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/tables/ce64bd87a58da157b30a4b6adc4e521bf8ae495b6c724f370f9e4183e9a9cf51.jpg)

![d368f01d43a2e89e99d62162b6f1d226fb43448b0d50f49affae149ca98dc860.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/tables/d368f01d43a2e89e99d62162b6f1d226fb43448b0d50f49affae149ca98dc860.jpg)

![dafb20b6c24150290f21422b7881f5a9bfddbb425146253379ac6760af0a26f0.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/tables/dafb20b6c24150290f21422b7881f5a9bfddbb425146253379ac6760af0a26f0.jpg)

![e224f835c2ab111a8207a9959dc400ab46d38f391b9b9cbd7167e804f0dd9a57.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/tables/e224f835c2ab111a8207a9959dc400ab46d38f391b9b9cbd7167e804f0dd9a57.jpg)

![faeda35782e99c8d13d89769a4b482087a8af286827612d13c33b34f230886f3.jpg](../iclr_results/105_Dynamic%20Multimodal%20Evaluation%20with%20Flexible%20Complexity%20by%20Vision-Language%20Bootstrapping/tables/faeda35782e99c8d13d89769a4b482087a8af286827612d13c33b34f230886f3.jpg)

## LARP: Tokenizing Videos with a Learned Autoregressive Generative Prior


### Images

![00bf0f290b2ed9dea612da4955aae86bd9943b9a93576d34e26107935ee53d20.jpg](../iclr_results/106_LARP_%20Tokenizing%20Videos%20with%20a%20Learned%20Autoregressive%20Generative%20Prior/images/00bf0f290b2ed9dea612da4955aae86bd9943b9a93576d34e26107935ee53d20.jpg)

![18a5fbe50a0f488ab4bd9e6b46000cc9baee027154ce7d8907e05fbf9e07eb80.jpg](../iclr_results/106_LARP_%20Tokenizing%20Videos%20with%20a%20Learned%20Autoregressive%20Generative%20Prior/images/18a5fbe50a0f488ab4bd9e6b46000cc9baee027154ce7d8907e05fbf9e07eb80.jpg)

![285ad148803b92e5ca4a7256ff503a71f769cc86e783cab76138029c3c97b51c.jpg](../iclr_results/106_LARP_%20Tokenizing%20Videos%20with%20a%20Learned%20Autoregressive%20Generative%20Prior/images/285ad148803b92e5ca4a7256ff503a71f769cc86e783cab76138029c3c97b51c.jpg)

![53fa1fac97c2d216bdc4006c6ebdeadae7f0a0375d4eb7eac244e450598b4f47.jpg](../iclr_results/106_LARP_%20Tokenizing%20Videos%20with%20a%20Learned%20Autoregressive%20Generative%20Prior/images/53fa1fac97c2d216bdc4006c6ebdeadae7f0a0375d4eb7eac244e450598b4f47.jpg)

![5bc0a0f6b1ce41eb1bacbd6b19237c377c55689feae09fa73c7e56994a6a4475.jpg](../iclr_results/106_LARP_%20Tokenizing%20Videos%20with%20a%20Learned%20Autoregressive%20Generative%20Prior/images/5bc0a0f6b1ce41eb1bacbd6b19237c377c55689feae09fa73c7e56994a6a4475.jpg)

![6a285e8aa5a3e6cc9d351dbf364b209fe6dd4004e35e7857e6376b52584540e6.jpg](../iclr_results/106_LARP_%20Tokenizing%20Videos%20with%20a%20Learned%20Autoregressive%20Generative%20Prior/images/6a285e8aa5a3e6cc9d351dbf364b209fe6dd4004e35e7857e6376b52584540e6.jpg)

![6e079da0bf75f68939171c768c704ebafdb137a3a51dec64e44c0c9c5e6d2b01.jpg](../iclr_results/106_LARP_%20Tokenizing%20Videos%20with%20a%20Learned%20Autoregressive%20Generative%20Prior/images/6e079da0bf75f68939171c768c704ebafdb137a3a51dec64e44c0c9c5e6d2b01.jpg)

![7643b5f29197044022a6dd51e82cdaf59a70d051a7143b8c89f809b550439c2a.jpg](../iclr_results/106_LARP_%20Tokenizing%20Videos%20with%20a%20Learned%20Autoregressive%20Generative%20Prior/images/7643b5f29197044022a6dd51e82cdaf59a70d051a7143b8c89f809b550439c2a.jpg)

![7dc1f1cb2255be73e4af973a4d53d72940306dc6395dd28fea987284473aff42.jpg](../iclr_results/106_LARP_%20Tokenizing%20Videos%20with%20a%20Learned%20Autoregressive%20Generative%20Prior/images/7dc1f1cb2255be73e4af973a4d53d72940306dc6395dd28fea987284473aff42.jpg)

![8afb11b47103dd957228f1178dd7896fc84d7f55cd0632dd6650d8b550120e31.jpg](../iclr_results/106_LARP_%20Tokenizing%20Videos%20with%20a%20Learned%20Autoregressive%20Generative%20Prior/images/8afb11b47103dd957228f1178dd7896fc84d7f55cd0632dd6650d8b550120e31.jpg)

![9dded8e1b99db170ec0a955fc880bbe11f40961a44201f09e4533f66b17417e0.jpg](../iclr_results/106_LARP_%20Tokenizing%20Videos%20with%20a%20Learned%20Autoregressive%20Generative%20Prior/images/9dded8e1b99db170ec0a955fc880bbe11f40961a44201f09e4533f66b17417e0.jpg)

![d5b8de1b8c59abf031031ea2ad59637954b0f38966efbb9683fde70f12b11887.jpg](../iclr_results/106_LARP_%20Tokenizing%20Videos%20with%20a%20Learned%20Autoregressive%20Generative%20Prior/images/d5b8de1b8c59abf031031ea2ad59637954b0f38966efbb9683fde70f12b11887.jpg)

![e27275c678a25a70821fc8c76aea759d3cfdca02c15460f489fd5afbe01d2d6d.jpg](../iclr_results/106_LARP_%20Tokenizing%20Videos%20with%20a%20Learned%20Autoregressive%20Generative%20Prior/images/e27275c678a25a70821fc8c76aea759d3cfdca02c15460f489fd5afbe01d2d6d.jpg)

![e5fd921149d71da94f9b40e29dd156753f301ca56245533ab33c90eb83ffb8c0.jpg](../iclr_results/106_LARP_%20Tokenizing%20Videos%20with%20a%20Learned%20Autoregressive%20Generative%20Prior/images/e5fd921149d71da94f9b40e29dd156753f301ca56245533ab33c90eb83ffb8c0.jpg)

![efdd122612d5151cd77ba057a94e359241232c8dc073246fb4bc0edb2c78db5f.jpg](../iclr_results/106_LARP_%20Tokenizing%20Videos%20with%20a%20Learned%20Autoregressive%20Generative%20Prior/images/efdd122612d5151cd77ba057a94e359241232c8dc073246fb4bc0edb2c78db5f.jpg)

![f61b9f3e5307b4626f253f9b43309f998aae67431bcdb08379d78a06e805d928.jpg](../iclr_results/106_LARP_%20Tokenizing%20Videos%20with%20a%20Learned%20Autoregressive%20Generative%20Prior/images/f61b9f3e5307b4626f253f9b43309f998aae67431bcdb08379d78a06e805d928.jpg)

![f74d95c9ff56479fc36a98bde413b138aacbad31c08a2b200cdae85114005f5a.jpg](../iclr_results/106_LARP_%20Tokenizing%20Videos%20with%20a%20Learned%20Autoregressive%20Generative%20Prior/images/f74d95c9ff56479fc36a98bde413b138aacbad31c08a2b200cdae85114005f5a.jpg)

![fe1440aa913c1f06ff85b2ec1b0c2df34f8126d31d00b4211b3d76532cdd2ae7.jpg](../iclr_results/106_LARP_%20Tokenizing%20Videos%20with%20a%20Learned%20Autoregressive%20Generative%20Prior/images/fe1440aa913c1f06ff85b2ec1b0c2df34f8126d31d00b4211b3d76532cdd2ae7.jpg)

### Tables

![1c2a0707a81c47d9dd63d8b9d9b850ff7dea23866652aa14b0e9e844ec15b28a.jpg](../iclr_results/106_LARP_%20Tokenizing%20Videos%20with%20a%20Learned%20Autoregressive%20Generative%20Prior/tables/1c2a0707a81c47d9dd63d8b9d9b850ff7dea23866652aa14b0e9e844ec15b28a.jpg)

![283e1641b06633c7c24b34d6b0b68456d8d41a266c70283bb375347c7e3e9f1f.jpg](../iclr_results/106_LARP_%20Tokenizing%20Videos%20with%20a%20Learned%20Autoregressive%20Generative%20Prior/tables/283e1641b06633c7c24b34d6b0b68456d8d41a266c70283bb375347c7e3e9f1f.jpg)

## Knowing Your Target: Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding


### Images

![06bd34fd5ec722841da7a3983ec090de20bc5b35f18667b11864f074e6e04611.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/images/06bd34fd5ec722841da7a3983ec090de20bc5b35f18667b11864f074e6e04611.jpg)

![203a9b5f750ba705d714f4ac0c2a43dfab60b28e29636c0a179301ec2ffeb9ce.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/images/203a9b5f750ba705d714f4ac0c2a43dfab60b28e29636c0a179301ec2ffeb9ce.jpg)

![2a9e296fc0efc9e48dfa6d518011517080495053ff98b2d87c42b40092089e52.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/images/2a9e296fc0efc9e48dfa6d518011517080495053ff98b2d87c42b40092089e52.jpg)

![34177ff1b654484924d24c31cabc77459638e6440d88e8ca1fe36f182b001284.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/images/34177ff1b654484924d24c31cabc77459638e6440d88e8ca1fe36f182b001284.jpg)

![3630f908fd391454ec9580106256cfdf8c55f1498edeb423b18035038312ae81.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/images/3630f908fd391454ec9580106256cfdf8c55f1498edeb423b18035038312ae81.jpg)

![4e5367c43f40561931cdbe3e0b17338047b97a15e7d85eeabcc12d2ff73797ce.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/images/4e5367c43f40561931cdbe3e0b17338047b97a15e7d85eeabcc12d2ff73797ce.jpg)

![4eec1b8347715405f1f4e1a5d575babeaa4ad7c35165af1cb60569a6f0e9b3d4.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/images/4eec1b8347715405f1f4e1a5d575babeaa4ad7c35165af1cb60569a6f0e9b3d4.jpg)

![5cbbce0a178e164ad615681da3aac23c28ac478fec10159d758fae9ceeec0da6.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/images/5cbbce0a178e164ad615681da3aac23c28ac478fec10159d758fae9ceeec0da6.jpg)

![750e31541e8926e0ca5b793ee4c5e5c9f9fb8b1823b98d150d3555b92ad5925a.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/images/750e31541e8926e0ca5b793ee4c5e5c9f9fb8b1823b98d150d3555b92ad5925a.jpg)

![94f091754252a6f54f56869c0f974006ee0ecf6d30ce947fc4ecaae9808a922c.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/images/94f091754252a6f54f56869c0f974006ee0ecf6d30ce947fc4ecaae9808a922c.jpg)

![b38cccb9a7868abcff27aa3d5be584fe652d83b0a87b8f5aea94cba46090e92c.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/images/b38cccb9a7868abcff27aa3d5be584fe652d83b0a87b8f5aea94cba46090e92c.jpg)

![d69089f398699939ecca79dd72687d1e5aea177d84000e2e947a52f079d5a590.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/images/d69089f398699939ecca79dd72687d1e5aea177d84000e2e947a52f079d5a590.jpg)

![e07bd5af04d5e5397cf1625c648e15c3b776e0267428b3867770d9af4fd6b4cd.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/images/e07bd5af04d5e5397cf1625c648e15c3b776e0267428b3867770d9af4fd6b4cd.jpg)

![e3ef00ed674b0723562a19cf8e15274b312bd5341ccb9c0e9d1cddab1fd37728.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/images/e3ef00ed674b0723562a19cf8e15274b312bd5341ccb9c0e9d1cddab1fd37728.jpg)

![f7d4c514e16ecac0a3c903793ca92eaa57a47684a7f26f3043c9a4bd9a6c98e7.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/images/f7d4c514e16ecac0a3c903793ca92eaa57a47684a7f26f3043c9a4bd9a6c98e7.jpg)

![ff915b1be5a0f67d770132ec3b4c2c882c0584bb943adb4bce50ee62032a788f.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/images/ff915b1be5a0f67d770132ec3b4c2c882c0584bb943adb4bce50ee62032a788f.jpg)

### Tables

![0889e8e55e0255d37566ee75a3d41db0cce678f0cd418e42fa3e8af6f5df4b49.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/tables/0889e8e55e0255d37566ee75a3d41db0cce678f0cd418e42fa3e8af6f5df4b49.jpg)

![15fd2a92c5b4f5ac5307c43bff06a8778b4e3a5be6023db46c83244bc48ee33a.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/tables/15fd2a92c5b4f5ac5307c43bff06a8778b4e3a5be6023db46c83244bc48ee33a.jpg)

![387813a20936219993b1f8a26f279b03c622e47eba4d971ecc3948a79efbeae6.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/tables/387813a20936219993b1f8a26f279b03c622e47eba4d971ecc3948a79efbeae6.jpg)

![75a49d90017da46b980788cd2668f839dbb6fbc4b679a9ecb3b8f68f52f4409b.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/tables/75a49d90017da46b980788cd2668f839dbb6fbc4b679a9ecb3b8f68f52f4409b.jpg)

![7fddc45804a565783bfbd7915f0e5fe15960e94293e41eabdb321e92b8cd4a8f.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/tables/7fddc45804a565783bfbd7915f0e5fe15960e94293e41eabdb321e92b8cd4a8f.jpg)

![9de336e59e97e47217cec5625735ef571fdd5f7451c00f4d9d54dbe1459bceef.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/tables/9de336e59e97e47217cec5625735ef571fdd5f7451c00f4d9d54dbe1459bceef.jpg)

![a5fc8cefc907b98e6a0d93c9c2f939d9d8b92e1bd5c0b06b6d5b20b9c0a140e5.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/tables/a5fc8cefc907b98e6a0d93c9c2f939d9d8b92e1bd5c0b06b6d5b20b9c0a140e5.jpg)

![c8ef632b4c64cb6f2eecd8e55c207b6cd1ea6e99fcd0f9fdb897404d4e46e67d.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/tables/c8ef632b4c64cb6f2eecd8e55c207b6cd1ea6e99fcd0f9fdb897404d4e46e67d.jpg)

![d0334a30ff894c8444075d76e4568449c418873deb863ec9ec146967a0f8f4d3.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/tables/d0334a30ff894c8444075d76e4568449c418873deb863ec9ec146967a0f8f4d3.jpg)

![e7aa30459178dab0d33ea8eeb9c91ab44e30f9325a60fe7697496bca3e9d8634.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/tables/e7aa30459178dab0d33ea8eeb9c91ab44e30f9325a60fe7697496bca3e9d8634.jpg)

![e8b33de14866d2d83ff64d17e49d31d69133c9635632b78d2c1c3aed137efbfa.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/tables/e8b33de14866d2d83ff64d17e49d31d69133c9635632b78d2c1c3aed137efbfa.jpg)

![ef59c65ea525f6e026384de850453ae168f605ad5fcee44bf1be7bbf385d37cd.jpg](../iclr_results/107_Knowing%20Your%20Target_%20Target-Aware%20Transformer%20Makes%20Better%20Spatio-Temporal%20Video%20Grounding/tables/ef59c65ea525f6e026384de850453ae168f605ad5fcee44bf1be7bbf385d37cd.jpg)

## Self-Improvement in Language Models: The Sharpening Mechanism


### Images

![18e5275141b3a9987c588d44eec073e445462b49af8e9a9452c3235cc95a053d.jpg](../iclr_results/108_Self-Improvement%20in%20Language%20Models_%20The%20Sharpening%20Mechanism/images/18e5275141b3a9987c588d44eec073e445462b49af8e9a9452c3235cc95a053d.jpg)

![1e782a3941613684040252887e23c2c0bd1c92907ef0b246a7381438f93162a2.jpg](../iclr_results/108_Self-Improvement%20in%20Language%20Models_%20The%20Sharpening%20Mechanism/images/1e782a3941613684040252887e23c2c0bd1c92907ef0b246a7381438f93162a2.jpg)

![48042c408d0975fca47b6a3d4afb458472408d5009038532a947691e695e9a0d.jpg](../iclr_results/108_Self-Improvement%20in%20Language%20Models_%20The%20Sharpening%20Mechanism/images/48042c408d0975fca47b6a3d4afb458472408d5009038532a947691e695e9a0d.jpg)

![81767971a8881301e4392036a4636b5450b9059f19db9f75b2601142d2ac5b99.jpg](../iclr_results/108_Self-Improvement%20in%20Language%20Models_%20The%20Sharpening%20Mechanism/images/81767971a8881301e4392036a4636b5450b9059f19db9f75b2601142d2ac5b99.jpg)

![9b8c0fcf1b422a3f92f46f21d25501fd5e2150b727ccc4b001e3b0c0cd376cc0.jpg](../iclr_results/108_Self-Improvement%20in%20Language%20Models_%20The%20Sharpening%20Mechanism/images/9b8c0fcf1b422a3f92f46f21d25501fd5e2150b727ccc4b001e3b0c0cd376cc0.jpg)

![bb43dafafd507f6d9bb9bc4ffd695584ca6304f6680d81d2b9b7d0148a9d5cb7.jpg](../iclr_results/108_Self-Improvement%20in%20Language%20Models_%20The%20Sharpening%20Mechanism/images/bb43dafafd507f6d9bb9bc4ffd695584ca6304f6680d81d2b9b7d0148a9d5cb7.jpg)

![c68a7a098255db705aca784b3a700c887c048a9adb18ffbf1034846451258a26.jpg](../iclr_results/108_Self-Improvement%20in%20Language%20Models_%20The%20Sharpening%20Mechanism/images/c68a7a098255db705aca784b3a700c887c048a9adb18ffbf1034846451258a26.jpg)

![cb746cceba9c3318b422e6d1da775cdf3e75c0bf96fd5944811da47180f03bc0.jpg](../iclr_results/108_Self-Improvement%20in%20Language%20Models_%20The%20Sharpening%20Mechanism/images/cb746cceba9c3318b422e6d1da775cdf3e75c0bf96fd5944811da47180f03bc0.jpg)

![e621dbf140e5c4bc62e6b8939c6d4e53ce2db08df491f77a9f6ecfc8ce26b35b.jpg](../iclr_results/108_Self-Improvement%20in%20Language%20Models_%20The%20Sharpening%20Mechanism/images/e621dbf140e5c4bc62e6b8939c6d4e53ce2db08df491f77a9f6ecfc8ce26b35b.jpg)

### Tables

![83bb04729799bc2bd3e4297898e8aeee92b27bf71e9a3950605460b4ebe08f6c.jpg](../iclr_results/108_Self-Improvement%20in%20Language%20Models_%20The%20Sharpening%20Mechanism/tables/83bb04729799bc2bd3e4297898e8aeee92b27bf71e9a3950605460b4ebe08f6c.jpg)

![9aee0a5ed51645615b4a1a8c80016029292b5722c85e3858bcafaf5d99e0eb26.jpg](../iclr_results/108_Self-Improvement%20in%20Language%20Models_%20The%20Sharpening%20Mechanism/tables/9aee0a5ed51645615b4a1a8c80016029292b5722c85e3858bcafaf5d99e0eb26.jpg)

## Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models


### Images

![08ba150e4bcf8b8d0bfab02176c80eb54fe9e731c6ec8742956cfa168efaa276.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/08ba150e4bcf8b8d0bfab02176c80eb54fe9e731c6ec8742956cfa168efaa276.jpg)

![1b3004afc89a90e23145f04b52dfb918890259e3bab76e588c8699f70fa03c27.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/1b3004afc89a90e23145f04b52dfb918890259e3bab76e588c8699f70fa03c27.jpg)

![2624c7ffd69149bec0622eb9f569861a0496520a48cdd99a0c06040f11ab4f33.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/2624c7ffd69149bec0622eb9f569861a0496520a48cdd99a0c06040f11ab4f33.jpg)

![2d9be8ca83e45fe5a861b89299f48237a3e90f18c90ff2bf0bc015e0659922fa.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/2d9be8ca83e45fe5a861b89299f48237a3e90f18c90ff2bf0bc015e0659922fa.jpg)

![2e967aed8952d0a75c830bced5f6412effba0eb7c8c3b997138726f925c5c332.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/2e967aed8952d0a75c830bced5f6412effba0eb7c8c3b997138726f925c5c332.jpg)

![323fafe1429b9674c90e919991e87103b3f49012c0382fadcc8a851d7bebcba2.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/323fafe1429b9674c90e919991e87103b3f49012c0382fadcc8a851d7bebcba2.jpg)

![3e2e0e0dfccc80429e22ad5ffc9503498fd5243af29a3222563ac28a3bffefb0.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/3e2e0e0dfccc80429e22ad5ffc9503498fd5243af29a3222563ac28a3bffefb0.jpg)

![5153109b2253c1ff891b70798ffe576b8fc40de4f57f72ec66a27aaf9c7a0796.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/5153109b2253c1ff891b70798ffe576b8fc40de4f57f72ec66a27aaf9c7a0796.jpg)

![5930d6a9895579041edf08f70d727d68f746f5d02cc216ca633697edebbf567e.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/5930d6a9895579041edf08f70d727d68f746f5d02cc216ca633697edebbf567e.jpg)

![5aa6cfe976ec0c6281dc835b159f7630c78a266c29fbeda9a9e0e5a7fcf6a5b1.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/5aa6cfe976ec0c6281dc835b159f7630c78a266c29fbeda9a9e0e5a7fcf6a5b1.jpg)

![65d25a948df3e6dda49a595e64c60b46e1473b05e2f79c9c7de80e21f8984543.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/65d25a948df3e6dda49a595e64c60b46e1473b05e2f79c9c7de80e21f8984543.jpg)

![77adf8ab90522c3b47f80936b0fe828731e6e1bcbb5e7a585090e5ee68932162.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/77adf8ab90522c3b47f80936b0fe828731e6e1bcbb5e7a585090e5ee68932162.jpg)

![7ad444bf7a0eecba9068f9b607a4fbab3d22260ef199e1e3e293a954f27fb7f7.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/7ad444bf7a0eecba9068f9b607a4fbab3d22260ef199e1e3e293a954f27fb7f7.jpg)

![8375908bdcacec5a55a074d6cd8892289e41d6eee1ddd26457016f8c7b13d0f5.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/8375908bdcacec5a55a074d6cd8892289e41d6eee1ddd26457016f8c7b13d0f5.jpg)

![851ec2972017eaa758ec49e81a92125fef3dc1a71478e96631164976d9d59731.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/851ec2972017eaa758ec49e81a92125fef3dc1a71478e96631164976d9d59731.jpg)

![88b9047ce415f0f04a22f5a4c9e0acd67536ec2a6244391a4abbdcd12b6c4dac.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/88b9047ce415f0f04a22f5a4c9e0acd67536ec2a6244391a4abbdcd12b6c4dac.jpg)

![949e310b8472f94b32478a6c4c67d7446b6d2e6a7454cc635d9e88d3fbcab01a.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/949e310b8472f94b32478a6c4c67d7446b6d2e6a7454cc635d9e88d3fbcab01a.jpg)

![955ac6536628388355c082a09f9bb50a223103736c2261d1e259e15a630c7433.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/955ac6536628388355c082a09f9bb50a223103736c2261d1e259e15a630c7433.jpg)

![95ef2fdb891789bddd64cbb04a5dca7e04eaa81949099d14f7ded1eb18d85233.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/95ef2fdb891789bddd64cbb04a5dca7e04eaa81949099d14f7ded1eb18d85233.jpg)

![97f053b172003585e85ee6359c63074219a57032ca8773173939ccd03b9ee2c8.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/97f053b172003585e85ee6359c63074219a57032ca8773173939ccd03b9ee2c8.jpg)

![a99b1d4bdbe1c1bb957915243ffd88d545d6a864a255ea38430a8823d1991f7a.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/a99b1d4bdbe1c1bb957915243ffd88d545d6a864a255ea38430a8823d1991f7a.jpg)

![ab5452b00c0d4c313ddf3b333507020a51b593f7a28182429447bf78b74473e2.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/ab5452b00c0d4c313ddf3b333507020a51b593f7a28182429447bf78b74473e2.jpg)

![accc65fb0033a610f3015fc954b89eeddfd75b4330c972f54bb00be7f1e66b65.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/accc65fb0033a610f3015fc954b89eeddfd75b4330c972f54bb00be7f1e66b65.jpg)

![b40ec9ab04931f32a0f97603e5573b926d367ccf81ebbe36af48923590715537.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/b40ec9ab04931f32a0f97603e5573b926d367ccf81ebbe36af48923590715537.jpg)

![b4310a449bede353d5f9acd9cf33d7aebcbba717e4ffbd3c80369b8e9b458379.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/b4310a449bede353d5f9acd9cf33d7aebcbba717e4ffbd3c80369b8e9b458379.jpg)

![b59761365ff0e3eb7679184c462da6178d02bcef486550e94c8ae1cfe3235ad6.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/b59761365ff0e3eb7679184c462da6178d02bcef486550e94c8ae1cfe3235ad6.jpg)

![bae7aac41d9364b21686e1d79464126b8225c248d60a53c8b3840c55eab523ff.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/bae7aac41d9364b21686e1d79464126b8225c248d60a53c8b3840c55eab523ff.jpg)

![bb3de4143ca07468a1ec2f502177d3b2d32327f94f6b927c606033feb04fd5ae.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/bb3de4143ca07468a1ec2f502177d3b2d32327f94f6b927c606033feb04fd5ae.jpg)

![c020fbd73ea3d7557d282f670b9d75201d2157f372dbaa26d3d45b0a76995c5f.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/c020fbd73ea3d7557d282f670b9d75201d2157f372dbaa26d3d45b0a76995c5f.jpg)

![c705315c1203850ada67ad30c753d34046ce956f0b15182a388fd7cc92cd0cac.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/c705315c1203850ada67ad30c753d34046ce956f0b15182a388fd7cc92cd0cac.jpg)

![cbb8b40a34db747aedafb78c00e81d58e76e5310c99962de6beb5e932fe19503.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/cbb8b40a34db747aedafb78c00e81d58e76e5310c99962de6beb5e932fe19503.jpg)

![cdc13dd23762c8d0623af5b5467a534992b1a32759a6b27cbf8addc36c00e7e5.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/cdc13dd23762c8d0623af5b5467a534992b1a32759a6b27cbf8addc36c00e7e5.jpg)

![d269a863dfb1f1c63c7bdcce2f8dd5760a3a73a7337428e9108fc0efd4048ac5.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/d269a863dfb1f1c63c7bdcce2f8dd5760a3a73a7337428e9108fc0efd4048ac5.jpg)

![f330c9f0ccb6abf449fd0b36d4edbb4953809ab3c9ae0b698b7a9d111c608c92.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/f330c9f0ccb6abf449fd0b36d4edbb4953809ab3c9ae0b698b7a9d111c608c92.jpg)

![f8d53a800806af2545ea100c5689bf7ae86b70da3e4c35296c7a6a79f8645998.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/images/f8d53a800806af2545ea100c5689bf7ae86b70da3e4c35296c7a6a79f8645998.jpg)

### Tables

![6e51d656d475d66ba88cdd83d355f1329ed9ec9e38e55de8e1f987b3820d7070.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/tables/6e51d656d475d66ba88cdd83d355f1329ed9ec9e38e55de8e1f987b3820d7070.jpg)

![780b428615a8d3ae80a4186a53961c779802464c2c50886ddd82c7042e9f62a9.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/tables/780b428615a8d3ae80a4186a53961c779802464c2c50886ddd82c7042e9f62a9.jpg)

![83e8a8d2036fa5e48914337af0e1201823696fbc5365e73de915a398f8323b01.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/tables/83e8a8d2036fa5e48914337af0e1201823696fbc5365e73de915a398f8323b01.jpg)

![9367b6ae1547fa27b2b756a420973e85706076c27a1560008e1b9ffbf47e54f5.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/tables/9367b6ae1547fa27b2b756a420973e85706076c27a1560008e1b9ffbf47e54f5.jpg)

![9e405ee814211dbd676e40297fa966c577fe95726acc59f8be4d1a49e7accdae.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/tables/9e405ee814211dbd676e40297fa966c577fe95726acc59f8be4d1a49e7accdae.jpg)

![bcc51eb9742de44fab312189286b086f86866dfd2878675f462f33636d4c96c2.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/tables/bcc51eb9742de44fab312189286b086f86866dfd2878675f462f33636d4c96c2.jpg)

![c710a5c2b0047db813ff4596867d425be89b43330a690f9be1d5185a69861156.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/tables/c710a5c2b0047db813ff4596867d425be89b43330a690f9be1d5185a69861156.jpg)

![d08422f0359445acc138c6a384c85c37bcc64d32bb443fe5c73dee366bf1398f.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/tables/d08422f0359445acc138c6a384c85c37bcc64d32bb443fe5c73dee366bf1398f.jpg)

![f1103d5ede6b038c46e34715c0013daa1935cabb444c476a8f9ca708cce17b9d.jpg](../iclr_results/109_Do%20I%20Know%20This%20Entity_%20Knowledge%20Awareness%20and%20Hallucinations%20in%20Language%20Models/tables/f1103d5ede6b038c46e34715c0013daa1935cabb444c476a8f9ca708cce17b9d.jpg)

## LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization


### Images

![2541e95401589ac4a1e6bb548cd2cb5cab269c582b03fe54b8b09256472b29b8.jpg](../iclr_results/110_LoRA%20Done%20RITE_%20Robust%20Invariant%20Transformation%20Equilibration%20for%20LoRA%20Optimization/images/2541e95401589ac4a1e6bb548cd2cb5cab269c582b03fe54b8b09256472b29b8.jpg)

![47ac106272a87309ab28f6e7aef8c04855991faf11df20c03611242b2146c13f.jpg](../iclr_results/110_LoRA%20Done%20RITE_%20Robust%20Invariant%20Transformation%20Equilibration%20for%20LoRA%20Optimization/images/47ac106272a87309ab28f6e7aef8c04855991faf11df20c03611242b2146c13f.jpg)

![4d5e8b7d82331ad7085c0c3c61eb5078f735bf4cbc44c8b8f777de41519e8f0c.jpg](../iclr_results/110_LoRA%20Done%20RITE_%20Robust%20Invariant%20Transformation%20Equilibration%20for%20LoRA%20Optimization/images/4d5e8b7d82331ad7085c0c3c61eb5078f735bf4cbc44c8b8f777de41519e8f0c.jpg)

![5cb217a9dd951e2ec31a042e81295c0f109e04f711d24e5d19ee9099d1ced297.jpg](../iclr_results/110_LoRA%20Done%20RITE_%20Robust%20Invariant%20Transformation%20Equilibration%20for%20LoRA%20Optimization/images/5cb217a9dd951e2ec31a042e81295c0f109e04f711d24e5d19ee9099d1ced297.jpg)

### Tables

![0b9af55f208f1fd8f3125c3e65e178eed0ece1591252e13456c927fb831ad2f7.jpg](../iclr_results/110_LoRA%20Done%20RITE_%20Robust%20Invariant%20Transformation%20Equilibration%20for%20LoRA%20Optimization/tables/0b9af55f208f1fd8f3125c3e65e178eed0ece1591252e13456c927fb831ad2f7.jpg)

![3636b60d74b3143945d635cc19b0cce334b4eada557d38b75265096f8b44ebdc.jpg](../iclr_results/110_LoRA%20Done%20RITE_%20Robust%20Invariant%20Transformation%20Equilibration%20for%20LoRA%20Optimization/tables/3636b60d74b3143945d635cc19b0cce334b4eada557d38b75265096f8b44ebdc.jpg)

![455e57c6980736fd6495b98c66b0672df00dce8e38cf8280ead88ed80cd66f03.jpg](../iclr_results/110_LoRA%20Done%20RITE_%20Robust%20Invariant%20Transformation%20Equilibration%20for%20LoRA%20Optimization/tables/455e57c6980736fd6495b98c66b0672df00dce8e38cf8280ead88ed80cd66f03.jpg)

![56b7832ae3520292238ac789b3aca19bd8d89476d60301c9766d3777481572d8.jpg](../iclr_results/110_LoRA%20Done%20RITE_%20Robust%20Invariant%20Transformation%20Equilibration%20for%20LoRA%20Optimization/tables/56b7832ae3520292238ac789b3aca19bd8d89476d60301c9766d3777481572d8.jpg)

![5d253ad4887f5a751ccbb55dfd9d4a6ce6d95a539498423b3dc694fcbfc4db84.jpg](../iclr_results/110_LoRA%20Done%20RITE_%20Robust%20Invariant%20Transformation%20Equilibration%20for%20LoRA%20Optimization/tables/5d253ad4887f5a751ccbb55dfd9d4a6ce6d95a539498423b3dc694fcbfc4db84.jpg)

![c39ea2905472b51d5428d488c1e44e5efffeaf8e5e5a28e411c640b93ac7818b.jpg](../iclr_results/110_LoRA%20Done%20RITE_%20Robust%20Invariant%20Transformation%20Equilibration%20for%20LoRA%20Optimization/tables/c39ea2905472b51d5428d488c1e44e5efffeaf8e5e5a28e411c640b93ac7818b.jpg)

![c98b5e9e057ccc1e6165fb5e4af7a46ce8893006dfd1137e47586f9e93b0ec70.jpg](../iclr_results/110_LoRA%20Done%20RITE_%20Robust%20Invariant%20Transformation%20Equilibration%20for%20LoRA%20Optimization/tables/c98b5e9e057ccc1e6165fb5e4af7a46ce8893006dfd1137e47586f9e93b0ec70.jpg)

![dc8908f7d7e9ab0605175053ec4eebbafb7fd1dccf6fb5c4e3b1c5b4f9755a53.jpg](../iclr_results/110_LoRA%20Done%20RITE_%20Robust%20Invariant%20Transformation%20Equilibration%20for%20LoRA%20Optimization/tables/dc8908f7d7e9ab0605175053ec4eebbafb7fd1dccf6fb5c4e3b1c5b4f9755a53.jpg)

![ff45f170ba834e8dd1b92a37ec6f6284aab4f7b791211208876a12b98314fd3c.jpg](../iclr_results/110_LoRA%20Done%20RITE_%20Robust%20Invariant%20Transformation%20Equilibration%20for%20LoRA%20Optimization/tables/ff45f170ba834e8dd1b92a37ec6f6284aab4f7b791211208876a12b98314fd3c.jpg)

## Reasoning Elicitation in Language Models via Counterfactual Feedback


### Images

![0df4969376316cbd68e751552166e0b72d45d318a1e2cd239f2203b358e9650e.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/images/0df4969376316cbd68e751552166e0b72d45d318a1e2cd239f2203b358e9650e.jpg)

![13bdceee6554b96b5a8a760727172915633dcb2b9e1f8c5464d039bed9dd4930.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/images/13bdceee6554b96b5a8a760727172915633dcb2b9e1f8c5464d039bed9dd4930.jpg)

![15a75e9b8f2eaab2d68bf02f00d2845e320468a4a7018b5eb3a130ab4e778a54.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/images/15a75e9b8f2eaab2d68bf02f00d2845e320468a4a7018b5eb3a130ab4e778a54.jpg)

![2db37dfab76f8872c7791f8e3a9d6cd37824293ba17cdc8292d5e4208fe6a7d8.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/images/2db37dfab76f8872c7791f8e3a9d6cd37824293ba17cdc8292d5e4208fe6a7d8.jpg)

![389fbcc18d2615550c391103e3149b97209fb41df9c801dd79b9b32d46585e03.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/images/389fbcc18d2615550c391103e3149b97209fb41df9c801dd79b9b32d46585e03.jpg)

![38dc5ccf16e96f50b4dbb43c4102c7ef28ed514347e4c1729a5fb2a9b4099e88.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/images/38dc5ccf16e96f50b4dbb43c4102c7ef28ed514347e4c1729a5fb2a9b4099e88.jpg)

![3e0fb63e02abd63d64555ecb86955451623e7ce71d43c68501ff9a5dbf8467f3.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/images/3e0fb63e02abd63d64555ecb86955451623e7ce71d43c68501ff9a5dbf8467f3.jpg)

![489fe7e5850025cc4f945aa8f89982e7925b7d0684dcbc151fe6f4de413bb3e4.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/images/489fe7e5850025cc4f945aa8f89982e7925b7d0684dcbc151fe6f4de413bb3e4.jpg)

![71f19c3983fb66e7d4bdd96c8e4c81999af82896ea15ce8aded34475f2578cbb.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/images/71f19c3983fb66e7d4bdd96c8e4c81999af82896ea15ce8aded34475f2578cbb.jpg)

![74742c43eb537c0b93846fb8be38fee4b66032dd292a0e7d0eae8dd45fca790b.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/images/74742c43eb537c0b93846fb8be38fee4b66032dd292a0e7d0eae8dd45fca790b.jpg)

![9cd322a30243f399a956da24a485d8eea3934b9da4edad437d27ddfc897b2e7e.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/images/9cd322a30243f399a956da24a485d8eea3934b9da4edad437d27ddfc897b2e7e.jpg)

![a257fe2576391ac22d3fd8ef15137efb321fc826c8b63816e965e67c1fc16590.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/images/a257fe2576391ac22d3fd8ef15137efb321fc826c8b63816e965e67c1fc16590.jpg)

![ae4f8e27a495b226d72d2ef4511feb4b9fa1da81b7523f3c914c8b624880f1f6.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/images/ae4f8e27a495b226d72d2ef4511feb4b9fa1da81b7523f3c914c8b624880f1f6.jpg)

![b814b40d7475035b21379f4aa8142dad2020c27d683971d48cba3da75a80a1d7.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/images/b814b40d7475035b21379f4aa8142dad2020c27d683971d48cba3da75a80a1d7.jpg)

![b825694cc6da5afea4db8f6c6134d4ea71c53e71ca94d29168b515149731b84c.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/images/b825694cc6da5afea4db8f6c6134d4ea71c53e71ca94d29168b515149731b84c.jpg)

![ccbaf8c12ec9f207abced9547941c95f6b09b6d93c70ec207847a5f159c38fdd.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/images/ccbaf8c12ec9f207abced9547941c95f6b09b6d93c70ec207847a5f159c38fdd.jpg)

![cd9f152da5f825fcbeaf01803eea86dab4db7c28b423c26a45b2f755367c4d92.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/images/cd9f152da5f825fcbeaf01803eea86dab4db7c28b423c26a45b2f755367c4d92.jpg)

![d172c63a22bf089b0182850dfa5aee27de6760a98af2befcf34074ea6731bc1c.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/images/d172c63a22bf089b0182850dfa5aee27de6760a98af2befcf34074ea6731bc1c.jpg)

![d45f4cb85f597a33e9547a081e9b767d2e9f67d00329f6935d1f887c18d5d9f9.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/images/d45f4cb85f597a33e9547a081e9b767d2e9f67d00329f6935d1f887c18d5d9f9.jpg)

![d6a865d5cb41dbb42aa477f00d6eb4534b12f18f3585f32c603a32a80dfbcd81.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/images/d6a865d5cb41dbb42aa477f00d6eb4534b12f18f3585f32c603a32a80dfbcd81.jpg)

![db2c694a9ac91732d5d10e80d762b3f4fa81cc0efa3824cfafd664dd0dbc674c.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/images/db2c694a9ac91732d5d10e80d762b3f4fa81cc0efa3824cfafd664dd0dbc674c.jpg)

![f3d6caa046f8bd802fab9c9b85b0c44c07a09b6abb7d8e0879e5d6b9b603bdab.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/images/f3d6caa046f8bd802fab9c9b85b0c44c07a09b6abb7d8e0879e5d6b9b603bdab.jpg)

![fe2402ce678684538bfe721120f37159ccc854f8c6f6fec179ff6d3f54ead0ac.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/images/fe2402ce678684538bfe721120f37159ccc854f8c6f6fec179ff6d3f54ead0ac.jpg)

### Tables

![022b4d8c8f0a1ea6cc92bd20e65b78c292a3a99c355ab05a2374f68ef4281064.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/tables/022b4d8c8f0a1ea6cc92bd20e65b78c292a3a99c355ab05a2374f68ef4281064.jpg)

![2aea60b009defe24ae21bc97ebbacd650c988c724bb302b6ed20c10392c92bf7.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/tables/2aea60b009defe24ae21bc97ebbacd650c988c724bb302b6ed20c10392c92bf7.jpg)

![36f43be02355d984b8dc92f23b6216b6be8a9bc8c02008da8391c8aa7ab4d1f1.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/tables/36f43be02355d984b8dc92f23b6216b6be8a9bc8c02008da8391c8aa7ab4d1f1.jpg)

![4ca6823140bf0eb0dd026653103fb2173959fb33ae13ec6e73daf19bd8c424f4.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/tables/4ca6823140bf0eb0dd026653103fb2173959fb33ae13ec6e73daf19bd8c424f4.jpg)

![56f8a1caac27b395c09b254229ae535c56e6773d0dab627234919f3bf5f5f7d5.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/tables/56f8a1caac27b395c09b254229ae535c56e6773d0dab627234919f3bf5f5f7d5.jpg)

![6d10adf6de080d38f84bf58c4304000d6b23bdf75274f699f9e663497b5e8df9.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/tables/6d10adf6de080d38f84bf58c4304000d6b23bdf75274f699f9e663497b5e8df9.jpg)

![9b39eee9256197a2995aebeba516f1fb15beacaa1690d7911edc9858e4645802.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/tables/9b39eee9256197a2995aebeba516f1fb15beacaa1690d7911edc9858e4645802.jpg)

![9ec2282e2701a390d1e1858d93b7c23df2aa5f96072627c0da922cad210fba36.jpg](../iclr_results/111_Reasoning%20Elicitation%20in%20Language%20Models%20via%20Counterfactual%20Feedback/tables/9ec2282e2701a390d1e1858d93b7c23df2aa5f96072627c0da922cad210fba36.jpg)

## Attention as a Hypernetwork


### Images

![2032eac2e95b556f4e629e0486ea54521e5a610dc33fcd9be5f4c7c05669efd1.jpg](../iclr_results/112_Attention%20as%20a%20Hypernetwork/images/2032eac2e95b556f4e629e0486ea54521e5a610dc33fcd9be5f4c7c05669efd1.jpg)

![410d4b3a4f22d8a6b1421b58138f9c9232b1edc9ac1b054490e0239dc1f79b8d.jpg](../iclr_results/112_Attention%20as%20a%20Hypernetwork/images/410d4b3a4f22d8a6b1421b58138f9c9232b1edc9ac1b054490e0239dc1f79b8d.jpg)

![4b938dd9a69d8391d6aa085bdb07dba7ce8c48cb15b7ca84d8ada1630f99411a.jpg](../iclr_results/112_Attention%20as%20a%20Hypernetwork/images/4b938dd9a69d8391d6aa085bdb07dba7ce8c48cb15b7ca84d8ada1630f99411a.jpg)

![508a6fb0709ac00fe462503ca2ce2e8749434f0c30df16cbbfea012667977836.jpg](../iclr_results/112_Attention%20as%20a%20Hypernetwork/images/508a6fb0709ac00fe462503ca2ce2e8749434f0c30df16cbbfea012667977836.jpg)

![54ecc08b1ce40153ac608cf6c9a2a78bd8988c767f41b67dd213696141cc8ebc.jpg](../iclr_results/112_Attention%20as%20a%20Hypernetwork/images/54ecc08b1ce40153ac608cf6c9a2a78bd8988c767f41b67dd213696141cc8ebc.jpg)

![5d662740c2f3e2d72549f1ec1af8019714a93aace103788e163bf5938924ea78.jpg](../iclr_results/112_Attention%20as%20a%20Hypernetwork/images/5d662740c2f3e2d72549f1ec1af8019714a93aace103788e163bf5938924ea78.jpg)

![601c0c8d470d30250557382d9c53738423744682c1bb893f7af6061a93dd836c.jpg](../iclr_results/112_Attention%20as%20a%20Hypernetwork/images/601c0c8d470d30250557382d9c53738423744682c1bb893f7af6061a93dd836c.jpg)

![8165512b29de4ffb16784b91ef454ef1c81ff22589fb1d47a2f6078dfdc883b0.jpg](../iclr_results/112_Attention%20as%20a%20Hypernetwork/images/8165512b29de4ffb16784b91ef454ef1c81ff22589fb1d47a2f6078dfdc883b0.jpg)

![a341987aa8653223242fd4570ca4479edca037c7629f8ce799beec085a2e0330.jpg](../iclr_results/112_Attention%20as%20a%20Hypernetwork/images/a341987aa8653223242fd4570ca4479edca037c7629f8ce799beec085a2e0330.jpg)

![aa5540ed80f3523cb3bd472303d1d81242eae6364cd70fd0a2014d4335e98010.jpg](../iclr_results/112_Attention%20as%20a%20Hypernetwork/images/aa5540ed80f3523cb3bd472303d1d81242eae6364cd70fd0a2014d4335e98010.jpg)

![ab2e82c0c335f75f65d5db390dc4c77d0ae877bf70b03abf59b634ba12c6f4ae.jpg](../iclr_results/112_Attention%20as%20a%20Hypernetwork/images/ab2e82c0c335f75f65d5db390dc4c77d0ae877bf70b03abf59b634ba12c6f4ae.jpg)

![b3032cb9f24f5d9026f38fe6c8208962e1c7e96ca4c68b4b99fdceb590f90168.jpg](../iclr_results/112_Attention%20as%20a%20Hypernetwork/images/b3032cb9f24f5d9026f38fe6c8208962e1c7e96ca4c68b4b99fdceb590f90168.jpg)

![c0803e244ac071ee0ea180e589c374e30870583c6c9e496b5786f190e54526f1.jpg](../iclr_results/112_Attention%20as%20a%20Hypernetwork/images/c0803e244ac071ee0ea180e589c374e30870583c6c9e496b5786f190e54526f1.jpg)

![c0bb56ef51e2ba54f5006c783b47cbc63c4f5fdd5ba935ada31b25355c970288.jpg](../iclr_results/112_Attention%20as%20a%20Hypernetwork/images/c0bb56ef51e2ba54f5006c783b47cbc63c4f5fdd5ba935ada31b25355c970288.jpg)

![c649ed2a426d264ce6f0da9ee1132e860364f725132f8de18cc2a6ae92ca2cd8.jpg](../iclr_results/112_Attention%20as%20a%20Hypernetwork/images/c649ed2a426d264ce6f0da9ee1132e860364f725132f8de18cc2a6ae92ca2cd8.jpg)

![d5dca0f60240e4ff9252adcb8e9c26da754d1d27e42508eb26a3dad53247655b.jpg](../iclr_results/112_Attention%20as%20a%20Hypernetwork/images/d5dca0f60240e4ff9252adcb8e9c26da754d1d27e42508eb26a3dad53247655b.jpg)

### Tables

![28ebd87711fabbf07ab541065dd88c91b4fc5d469b57ac9efd80d1bdfb4dd9f6.jpg](../iclr_results/112_Attention%20as%20a%20Hypernetwork/tables/28ebd87711fabbf07ab541065dd88c91b4fc5d469b57ac9efd80d1bdfb4dd9f6.jpg)

![b3f69554760dfe3a6224f8d00309c0532fb0c09253128168eeca6f70a697bc90.jpg](../iclr_results/112_Attention%20as%20a%20Hypernetwork/tables/b3f69554760dfe3a6224f8d00309c0532fb0c09253128168eeca6f70a697bc90.jpg)

![b6e098de157f9a8bedd8c0b97c074c802dcdb0cdc61a263f795f9a0e3b656260.jpg](../iclr_results/112_Attention%20as%20a%20Hypernetwork/tables/b6e098de157f9a8bedd8c0b97c074c802dcdb0cdc61a263f795f9a0e3b656260.jpg)

## Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues

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
