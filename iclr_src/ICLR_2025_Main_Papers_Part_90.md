# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 90 of 100

## 目录 (Table of Contents)

1. [Do WGANs succeed because they minimize the Wasserstein Distance? Lessons from Discrete Generators](#Do-WGANs-succeed-because-they-minimize-the-Wasserstein-Distance-Lessons-from-Discrete-Generators)
2. [Can Reinforcement Learning Solve Asymmetric Combinatorial-Continuous Zero-Sum Games?](#Can-Reinforcement-Learning-Solve-Asymmetric-Combinatorial-Continuous-Zero-Sum-Games)
3. [Forte : Finding Outliers with Representation Typicality Estimation](#Forte-Finding-Outliers-with-Representation-Typicality-Estimation)
4. [Drama: Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient](#Drama-Mamba-Enabled-Model-Based-Reinforcement-Learning-Is-Sample-and-Parameter-Efficient)
5. [NextBestPath: Efficient 3D Mapping of Unseen Environments](#NextBestPath-Efficient-3D-Mapping-of-Unseen-Environments)
6. [AnoLLM: Large Language Models for Tabular Anomaly Detection](#AnoLLM-Large-Language-Models-for-Tabular-Anomaly-Detection)
7. [xFinder: Large Language Models as Automated Evaluators for Reliable Evaluation](#xFinder-Large-Language-Models-as-Automated-Evaluators-for-Reliable-Evaluation)
8. [Modality-Specialized Synergizers for Interleaved Vision-Language Generalists](#Modality-Specialized-Synergizers-for-Interleaved-Vision-Language-Generalists)
9. [Entropy-based Activation Function Optimization: A Method on Searching Better Activation Functions](#Entropy-based-Activation-Function-Optimization-A-Method-on-Searching-Better-Activation-Functions)
10. [AI Sandbagging: Language Models can Strategically Underperform on Evaluations](#AI-Sandbagging-Language-Models-can-Strategically-Underperform-on-Evaluations)
11. [Lie Algebra Canonicalization: Equivariant Neural Operators under arbitrary Lie Groups](#Lie-Algebra-Canonicalization-Equivariant-Neural-Operators-under-arbitrary-Lie-Groups)
12. [Logically Consistent Language Models via Neuro-Symbolic Integration](#Logically-Consistent-Language-Models-via-Neuro-Symbolic-Integration)
13. [Alchemy: Amplifying Theorem-Proving Capability Through Symbolic Mutation](#Alchemy-Amplifying-Theorem-Proving-Capability-Through-Symbolic-Mutation)
14. [Examining Alignment of Large Language Models through Representative Heuristics: the case of political stereotypes](#Examining-Alignment-of-Large-Language-Models-through-Representative-Heuristics-the-case-of-political-stereotypes)
15. [DON’T STOP ME NOW: EMBEDDING BASED SCHEDULING FOR LLMS](#DONT-STOP-ME-NOW-EMBEDDING-BASED-SCHEDULING-FOR-LLMS)
16. [3DGS-Drag: Dragging Gaussians for Intuitive Point-Based 3D Editing](#3DGS-Drag-Dragging-Gaussians-for-Intuitive-Point-Based-3D-Editing)
17. [Visually Consistent Hierarchical Image Classification](#Visually-Consistent-Hierarchical-Image-Classification)
18. [Prompting Fairness: Integrating Causality to Debias Large Language Models](#Prompting-Fairness-Integrating-Causality-to-Debias-Large-Language-Models)
19. [WardropNet: Traffic Flow Predictions via Equilibrium-Augmented Learning](#WardropNet-Traffic-Flow-Predictions-via-Equilibrium-Augmented-Learning)
20. [Lawma: The Power of Specialization for Legal Annotation](#Lawma-The-Power-of-Specialization-for-Legal-Annotation)
21. [MIA-Bench: Towards Better Instruction Following Evaluation of Multimodal LLMs](#MIA-Bench-Towards-Better-Instruction-Following-Evaluation-of-Multimodal-LLMs)
22. [Bandit Learning in Matching Markets with Indifference](#Bandit-Learning-in-Matching-Markets-with-Indifference)
23. [PuzzleFusion++: Auto-agglomerative 3D Fracture Assembly by Denoise and Verify](#PuzzleFusion-Auto-agglomerative-3D-Fracture-Assembly-by-Denoise-and-Verify)
24. [Training One-Dimensional Graph Neural Networks is NP-Hard](#Training-One-Dimensional-Graph-Neural-Networks-is-NP-Hard)
25. [Regretful Decisions under Label Noise](#Regretful-Decisions-under-Label-Noise)
26. [Bad-PFL: Exploiting Backdoor Attacks against Personalized Federated Learning](#Bad-PFL-Exploiting-Backdoor-Attacks-against-Personalized-Federated-Learning)
27. [Small-to-Large Generalization: Training Data Influences Models Consistently Across Scale](#Small-to-Large-Generalization-Training-Data-Influences-Models-Consistently-Across-Scale)
28. [MADGEN: Mass-Spec attends to De Novo Molecular generation](#MADGEN-Mass-Spec-attends-to-De-Novo-Molecular-generation)
29. [On the Adversarial Risk of Test Time Adaptation: An Investigation into Realistic Test-Time Data Poisoning](#On-the-Adversarial-Risk-of-Test-Time-Adaptation-An-Investigation-into-Realistic-Test-Time-Data-Poisoning)
30. [Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models](#Inference-Aware-Fine-Tuning-for-Best-of-N-Sampling-in-Large-Language-Models)
31. [Multi-Label Test-Time Adaptation with Bound Entropy Minimization](#Multi-Label-Test-Time-Adaptation-with-Bound-Entropy-Minimization)
32. [Wayward Concepts In Multimodal Models](#Wayward-Concepts-In-Multimodal-Models)
33. [LaGeM: A Large Geometry Model for 3D Representation Learning and Diffusion](#LaGeM-A-Large-Geometry-Model-for-3D-Representation-Learning-and-Diffusion)
34. [Mini-Monkey: Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image Pyramid](#Mini-Monkey-Alleviating-the-Semantic-Sawtooth-Effect-for-Lightweight-MLLMs-via-Complementary-Image-Pyramid)
35. [ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery](#ScienceAgentBench-Toward-Rigorous-Assessment-of-Language-Agents-for-Data-Driven-Scientific-Discovery)
36. [Episodic Memories Generation and Evaluation Benchmark for Large Language Models](#Episodic-Memories-Generation-and-Evaluation-Benchmark-for-Large-Language-Models)
37. [Towards Foundation Models for Mixed Integer Linear Programming](#Towards-Foundation-Models-for-Mixed-Integer-Linear-Programming)

---


## Do WGANs succeed because they minimize the Wasserstein Distance? Lessons from Discrete Generators

### Images

![4a5811a83d3d58f1ff5dc5c47acb9d5b7f4bd6acd11ce3610becf076f5b3e3a6.jpg](../iclr_results/3309_Graph Neural Ricci Flow_ Evolving Feature from a Curvature Perspective/images/4a5811a83d3d58f1ff5dc5c47acb9d5b7f4bd6acd11ce3610becf076f5b3e3a6.jpg)

![5feaa1b7e2016c7b984bcfd4633ac10eae104099f4b1e388092074dd352eaacf.jpg](../iclr_results/3309_Graph Neural Ricci Flow_ Evolving Feature from a Curvature Perspective/images/5feaa1b7e2016c7b984bcfd4633ac10eae104099f4b1e388092074dd352eaacf.jpg)

![a68b8ca264b240a934fbd87234122d06e7d5518a04d36b399b1de05968e97380.jpg](../iclr_results/3309_Graph Neural Ricci Flow_ Evolving Feature from a Curvature Perspective/images/a68b8ca264b240a934fbd87234122d06e7d5518a04d36b399b1de05968e97380.jpg)

![dc020ba28147c86b729c24f844425bac5b61c8c34adee84668d2c0f3d08fc9ba.jpg](../iclr_results/3309_Graph Neural Ricci Flow_ Evolving Feature from a Curvature Perspective/images/dc020ba28147c86b729c24f844425bac5b61c8c34adee84668d2c0f3d08fc9ba.jpg)

![e62567cf36f7e78ac9bc781de442b8a3b7bbd606515b5c2cf7d19e2a2bbd452e.jpg](../iclr_results/3309_Graph Neural Ricci Flow_ Evolving Feature from a Curvature Perspective/images/e62567cf36f7e78ac9bc781de442b8a3b7bbd606515b5c2cf7d19e2a2bbd452e.jpg)

### Tables

![1da34caba4410278fc7da6677ddc5e07f6eb70176af5dce0d30872c66fefcf01.jpg](../iclr_results/3309_Graph Neural Ricci Flow_ Evolving Feature from a Curvature Perspective/tables/1da34caba4410278fc7da6677ddc5e07f6eb70176af5dce0d30872c66fefcf01.jpg)

![1eada136919065e9e4e63b7f15f8c232f9cf38ff5cdd5297eedb19aa639d042e.jpg](../iclr_results/3309_Graph Neural Ricci Flow_ Evolving Feature from a Curvature Perspective/tables/1eada136919065e9e4e63b7f15f8c232f9cf38ff5cdd5297eedb19aa639d042e.jpg)

![3166ee89eac23b83e23f6c8542914ecf1874f8071cd9c98d897bf77cbae6c17c.jpg](../iclr_results/3309_Graph Neural Ricci Flow_ Evolving Feature from a Curvature Perspective/tables/3166ee89eac23b83e23f6c8542914ecf1874f8071cd9c98d897bf77cbae6c17c.jpg)

![897183cbecb21449393fecf8ec0abae4ee46b9c954876ea53609cd351d785ac7.jpg](../iclr_results/3309_Graph Neural Ricci Flow_ Evolving Feature from a Curvature Perspective/tables/897183cbecb21449393fecf8ec0abae4ee46b9c954876ea53609cd351d785ac7.jpg)

![b4df5cec7206985f2052d21a3b8cb18915ed6ffb38fa703dce81a6e4da7a67eb.jpg](../iclr_results/3309_Graph Neural Ricci Flow_ Evolving Feature from a Curvature Perspective/tables/b4df5cec7206985f2052d21a3b8cb18915ed6ffb38fa703dce81a6e4da7a67eb.jpg)

![bc97c2c8cf0f8b4451c14247de47ad1efd2fbbbb421153a9f695b22d2e6a4210.jpg](../iclr_results/3309_Graph Neural Ricci Flow_ Evolving Feature from a Curvature Perspective/tables/bc97c2c8cf0f8b4451c14247de47ad1efd2fbbbb421153a9f695b22d2e6a4210.jpg)

![e019ed5a54004470b7e3a8481e5eef8110e98902d921a9230932f3fd3f4b2c23.jpg](../iclr_results/3309_Graph Neural Ricci Flow_ Evolving Feature from a Curvature Perspective/tables/e019ed5a54004470b7e3a8481e5eef8110e98902d921a9230932f3fd3f4b2c23.jpg)

![f000fa8a39c96cc78f0838e2169a7df3e6a57faa3d88371c6b6a0bb4800694fe.jpg](../iclr_results/3309_Graph Neural Ricci Flow_ Evolving Feature from a Curvature Perspective/tables/f000fa8a39c96cc78f0838e2169a7df3e6a57faa3d88371c6b6a0bb4800694fe.jpg)

## Do WGANs succeed because they minimize the Wasserstein Distance? Lessons from Discrete Generators


### Images

![0febf67bec70139477443aa3cb5a25568b7931e1c7c27290c0f651557e776622.jpg](../iclr_results/3310_Do WGANs succeed because they minimize the Wasserstein Distance_ Lessons from Discrete Generators/images/0febf67bec70139477443aa3cb5a25568b7931e1c7c27290c0f651557e776622.jpg)

![106204ba13457eeebb414ffba53196e4b750282cc9d346d7febebebb6992c230.jpg](../iclr_results/3310_Do WGANs succeed because they minimize the Wasserstein Distance_ Lessons from Discrete Generators/images/106204ba13457eeebb414ffba53196e4b750282cc9d346d7febebebb6992c230.jpg)

![11f55670773ae53820f3fe8fd39f0b775bdb09306653ec9af0623901c2d4027f.jpg](../iclr_results/3310_Do WGANs succeed because they minimize the Wasserstein Distance_ Lessons from Discrete Generators/images/11f55670773ae53820f3fe8fd39f0b775bdb09306653ec9af0623901c2d4027f.jpg)

![289c313d38aa20f23df4c80827b759936200ea5484c4f07665f67d85c9ee655b.jpg](../iclr_results/3310_Do WGANs succeed because they minimize the Wasserstein Distance_ Lessons from Discrete Generators/images/289c313d38aa20f23df4c80827b759936200ea5484c4f07665f67d85c9ee655b.jpg)

![42f6d14b1afbcc8276f56b581c3029c480d2a7f1bc792451953613a0a39aede0.jpg](../iclr_results/3310_Do WGANs succeed because they minimize the Wasserstein Distance_ Lessons from Discrete Generators/images/42f6d14b1afbcc8276f56b581c3029c480d2a7f1bc792451953613a0a39aede0.jpg)

![5bb69580a416efc266e984e3ecf9dd5bd378ca8c75dba932994a8d58f65d155f.jpg](../iclr_results/3310_Do WGANs succeed because they minimize the Wasserstein Distance_ Lessons from Discrete Generators/images/5bb69580a416efc266e984e3ecf9dd5bd378ca8c75dba932994a8d58f65d155f.jpg)

![60cd5b1e384484a986c0a32f1f2823196a49ac005aae62b8e50059d5d06f3a3d.jpg](../iclr_results/3310_Do WGANs succeed because they minimize the Wasserstein Distance_ Lessons from Discrete Generators/images/60cd5b1e384484a986c0a32f1f2823196a49ac005aae62b8e50059d5d06f3a3d.jpg)

![6471a38628a4f00d187d5616fa8dfbc5a2388090690fbfbd2ee9aaa098ddff03.jpg](../iclr_results/3310_Do WGANs succeed because they minimize the Wasserstein Distance_ Lessons from Discrete Generators/images/6471a38628a4f00d187d5616fa8dfbc5a2388090690fbfbd2ee9aaa098ddff03.jpg)

![64e14fa7cce51bfc8c80eba4a8788c82ec60ec71515d6449bf550e0df594e502.jpg](../iclr_results/3310_Do WGANs succeed because they minimize the Wasserstein Distance_ Lessons from Discrete Generators/images/64e14fa7cce51bfc8c80eba4a8788c82ec60ec71515d6449bf550e0df594e502.jpg)

![6c05140cb5379aa4eeca339c1ec3741d035b5d973ca7a16dd587227fa14ee193.jpg](../iclr_results/3310_Do WGANs succeed because they minimize the Wasserstein Distance_ Lessons from Discrete Generators/images/6c05140cb5379aa4eeca339c1ec3741d035b5d973ca7a16dd587227fa14ee193.jpg)

![8b499ceb14fb0fa8a102bea60cb4440cc92e9b087ea8b64d03109851fc2943f6.jpg](../iclr_results/3310_Do WGANs succeed because they minimize the Wasserstein Distance_ Lessons from Discrete Generators/images/8b499ceb14fb0fa8a102bea60cb4440cc92e9b087ea8b64d03109851fc2943f6.jpg)

![90e2cd541cea8d2fa423954939bb3dd545d0d64db307778dc7f988ba2b7310fc.jpg](../iclr_results/3310_Do WGANs succeed because they minimize the Wasserstein Distance_ Lessons from Discrete Generators/images/90e2cd541cea8d2fa423954939bb3dd545d0d64db307778dc7f988ba2b7310fc.jpg)

![b390ba9d99139a93a69c7867f7ea3552bd0f558913955a03318857d858c9d7eb.jpg](../iclr_results/3310_Do WGANs succeed because they minimize the Wasserstein Distance_ Lessons from Discrete Generators/images/b390ba9d99139a93a69c7867f7ea3552bd0f558913955a03318857d858c9d7eb.jpg)

![ca017a601996598513ee33d6b8f644b06c071b1beba93f362c0fdf2597e4cecd.jpg](../iclr_results/3310_Do WGANs succeed because they minimize the Wasserstein Distance_ Lessons from Discrete Generators/images/ca017a601996598513ee33d6b8f644b06c071b1beba93f362c0fdf2597e4cecd.jpg)

![d9fa2637e87c2645e7e75fdf2436c511ff4330c6212f87923d19ead1c44f90ad.jpg](../iclr_results/3310_Do WGANs succeed because they minimize the Wasserstein Distance_ Lessons from Discrete Generators/images/d9fa2637e87c2645e7e75fdf2436c511ff4330c6212f87923d19ead1c44f90ad.jpg)

![e2a22699a53b7af1da9965acf37093ada486351b0dbc681cdc3de83ebe65f447.jpg](../iclr_results/3310_Do WGANs succeed because they minimize the Wasserstein Distance_ Lessons from Discrete Generators/images/e2a22699a53b7af1da9965acf37093ada486351b0dbc681cdc3de83ebe65f447.jpg)

![f0a723dc510442c61ff90588b6f9e30d764efa4fb7b905a675df50f01b6e703c.jpg](../iclr_results/3310_Do WGANs succeed because they minimize the Wasserstein Distance_ Lessons from Discrete Generators/images/f0a723dc510442c61ff90588b6f9e30d764efa4fb7b905a675df50f01b6e703c.jpg)

![f2d2dc7078b3a7fe4703037264c8699e376c036523ecf43f1f365f16c6a78d58.jpg](../iclr_results/3310_Do WGANs succeed because they minimize the Wasserstein Distance_ Lessons from Discrete Generators/images/f2d2dc7078b3a7fe4703037264c8699e376c036523ecf43f1f365f16c6a78d58.jpg)

![f5811d5d9e039b0d3b494a8e8abbe96eedc30b9343256f21b6e2a5757af13050.jpg](../iclr_results/3310_Do WGANs succeed because they minimize the Wasserstein Distance_ Lessons from Discrete Generators/images/f5811d5d9e039b0d3b494a8e8abbe96eedc30b9343256f21b6e2a5757af13050.jpg)

### Tables

![7659c4c1314dbf837fbe349a01bfbd8bb5fa2c912e62b48b8c5b8562f94fb604.jpg](../iclr_results/3310_Do WGANs succeed because they minimize the Wasserstein Distance_ Lessons from Discrete Generators/tables/7659c4c1314dbf837fbe349a01bfbd8bb5fa2c912e62b48b8c5b8562f94fb604.jpg)

## Can Reinforcement Learning Solve Asymmetric Combinatorial-Continuous Zero-Sum Games?


### Images

![668be2238a3aec78618c6fd173973d89fbfeb5799cc6732174afb08a6db5ea24.jpg](../iclr_results/3311_Can Reinforcement Learning Solve Asymmetric Combinatorial-Continuous Zero-Sum Games_/images/668be2238a3aec78618c6fd173973d89fbfeb5799cc6732174afb08a6db5ea24.jpg)

![f1c8beba5f5ea6d16a951fb293ff1bded0a114da35bfd74287ad438e14b121d7.jpg](../iclr_results/3311_Can Reinforcement Learning Solve Asymmetric Combinatorial-Continuous Zero-Sum Games_/images/f1c8beba5f5ea6d16a951fb293ff1bded0a114da35bfd74287ad438e14b121d7.jpg)

### Tables

![52268752105356eb76806507a45a8c71fb17d8f8d6ab64f41aad4b1210b00cfa.jpg](../iclr_results/3311_Can Reinforcement Learning Solve Asymmetric Combinatorial-Continuous Zero-Sum Games_/tables/52268752105356eb76806507a45a8c71fb17d8f8d6ab64f41aad4b1210b00cfa.jpg)

![5c50d7134776bd4b649cd11f5172d9085f0e81643ae051faafd6c8bbbdbd0713.jpg](../iclr_results/3311_Can Reinforcement Learning Solve Asymmetric Combinatorial-Continuous Zero-Sum Games_/tables/5c50d7134776bd4b649cd11f5172d9085f0e81643ae051faafd6c8bbbdbd0713.jpg)

![74b4bf169a1dd61ff12e80ab18add0ef3f87991f948abaac06cf9c78c41ccad6.jpg](../iclr_results/3311_Can Reinforcement Learning Solve Asymmetric Combinatorial-Continuous Zero-Sum Games_/tables/74b4bf169a1dd61ff12e80ab18add0ef3f87991f948abaac06cf9c78c41ccad6.jpg)

![7a47fcb134886f2fb6fd54a49388d5cec31d0f6b70086592054a38e6a083bc0d.jpg](../iclr_results/3311_Can Reinforcement Learning Solve Asymmetric Combinatorial-Continuous Zero-Sum Games_/tables/7a47fcb134886f2fb6fd54a49388d5cec31d0f6b70086592054a38e6a083bc0d.jpg)

![7c0dd32b1e0f1e088ce5b6347cae4212bb693a82685e6c7f072b40eb99a8ea56.jpg](../iclr_results/3311_Can Reinforcement Learning Solve Asymmetric Combinatorial-Continuous Zero-Sum Games_/tables/7c0dd32b1e0f1e088ce5b6347cae4212bb693a82685e6c7f072b40eb99a8ea56.jpg)

![91ed368f56526ea504a7b5b493fe0360a66cce4fa6a00bca5658eff2dffcfff9.jpg](../iclr_results/3311_Can Reinforcement Learning Solve Asymmetric Combinatorial-Continuous Zero-Sum Games_/tables/91ed368f56526ea504a7b5b493fe0360a66cce4fa6a00bca5658eff2dffcfff9.jpg)

![a5abcd2cc3db7fa0ca4acbce55dfd645166efd092098985060efa73d62e65bbc.jpg](../iclr_results/3311_Can Reinforcement Learning Solve Asymmetric Combinatorial-Continuous Zero-Sum Games_/tables/a5abcd2cc3db7fa0ca4acbce55dfd645166efd092098985060efa73d62e65bbc.jpg)

![b0992e814ff937cd0ca4353d9887c7d816c29dc6f0f9cc5196df957397fd47b9.jpg](../iclr_results/3311_Can Reinforcement Learning Solve Asymmetric Combinatorial-Continuous Zero-Sum Games_/tables/b0992e814ff937cd0ca4353d9887c7d816c29dc6f0f9cc5196df957397fd47b9.jpg)

![cb54a6fa125799fef687fe08a0cd04665a17ee6ea326972ed596d37b27767244.jpg](../iclr_results/3311_Can Reinforcement Learning Solve Asymmetric Combinatorial-Continuous Zero-Sum Games_/tables/cb54a6fa125799fef687fe08a0cd04665a17ee6ea326972ed596d37b27767244.jpg)

![ea73b906606c62301ad00dd7bad00b6e8e6b05614bd458cb14a50d4ea596ab0b.jpg](../iclr_results/3311_Can Reinforcement Learning Solve Asymmetric Combinatorial-Continuous Zero-Sum Games_/tables/ea73b906606c62301ad00dd7bad00b6e8e6b05614bd458cb14a50d4ea596ab0b.jpg)

## Forte : Finding Outliers with Representation Typicality Estimation


### Images

![03d873d9b7e7394f33f5566c999902e3dcdbe27556b1c55ce73d91cfd2b5aff2.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/images/03d873d9b7e7394f33f5566c999902e3dcdbe27556b1c55ce73d91cfd2b5aff2.jpg)

![120d853a5c0b539832aa9c140e853fda28f6b250d2f0e23c7bdac33eed4c7f2b.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/images/120d853a5c0b539832aa9c140e853fda28f6b250d2f0e23c7bdac33eed4c7f2b.jpg)

![2e0545d0025568f07fb42fc146348bfae414779d201239d4c3136d0a41aac62f.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/images/2e0545d0025568f07fb42fc146348bfae414779d201239d4c3136d0a41aac62f.jpg)

![3c3a57c9c3e07d3beb65f27339a504ddab11bd2db1b979a549b4cb60dd0ab77e.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/images/3c3a57c9c3e07d3beb65f27339a504ddab11bd2db1b979a549b4cb60dd0ab77e.jpg)

![3fa116c20ab84c6d8c687b6e9ff0f8e9c05f4f28cc90800e7bef3c6fb62c02cc.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/images/3fa116c20ab84c6d8c687b6e9ff0f8e9c05f4f28cc90800e7bef3c6fb62c02cc.jpg)

![4b5289a87aff385b472687a2d819278e32ea01770518a980f572eaeb860fde09.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/images/4b5289a87aff385b472687a2d819278e32ea01770518a980f572eaeb860fde09.jpg)

![535692d85f02a38e1af48764cbc41dcfecf87c4914572fbb84bc7ead8d073cdb.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/images/535692d85f02a38e1af48764cbc41dcfecf87c4914572fbb84bc7ead8d073cdb.jpg)

![54fda3f8695237a0660c526f630e9904f4fb08573897466ca621b1af53f22abb.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/images/54fda3f8695237a0660c526f630e9904f4fb08573897466ca621b1af53f22abb.jpg)

![61f3b9895e830754bfb24be81be0d2c1d3534ebea88fc5407d226cd7a3227564.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/images/61f3b9895e830754bfb24be81be0d2c1d3534ebea88fc5407d226cd7a3227564.jpg)

![72ab4d45e88e8dff8f0243dd5ebdc2351910401f927915e2095f44cd5dfdffc5.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/images/72ab4d45e88e8dff8f0243dd5ebdc2351910401f927915e2095f44cd5dfdffc5.jpg)

![7ae63707a92e264849b2bc689924ae0ec7cc77c577b52800cbea4e029c3604aa.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/images/7ae63707a92e264849b2bc689924ae0ec7cc77c577b52800cbea4e029c3604aa.jpg)

![8d040ac13edc36750e623e0d04605fbabb93164ff5925ae142785a8fc0f1166f.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/images/8d040ac13edc36750e623e0d04605fbabb93164ff5925ae142785a8fc0f1166f.jpg)

![90606e3af9a506f67bc8f81a046920abdd8eeade85b558e44d7187af2e5e8983.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/images/90606e3af9a506f67bc8f81a046920abdd8eeade85b558e44d7187af2e5e8983.jpg)

![9df78fea9ff43e4e0104491133d58847ccb62bd37a0f740b27e4f1d5b2b6a995.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/images/9df78fea9ff43e4e0104491133d58847ccb62bd37a0f740b27e4f1d5b2b6a995.jpg)

![9fc3811f51195e3b975771ef17429fcfd57bdcda7a6a5b9fefe48c2488846fb2.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/images/9fc3811f51195e3b975771ef17429fcfd57bdcda7a6a5b9fefe48c2488846fb2.jpg)

![b6e120c8e5bf6c9ed6227ec1f04b05fda6d25a45f06609e069a50056e56b506a.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/images/b6e120c8e5bf6c9ed6227ec1f04b05fda6d25a45f06609e069a50056e56b506a.jpg)

![c5437ab55a5ab314c61b27546ec489330fcc9ee4b7c022fc98b0d95cafb018a1.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/images/c5437ab55a5ab314c61b27546ec489330fcc9ee4b7c022fc98b0d95cafb018a1.jpg)

![c8cfde03ad730b2a1fdafc0d38c228890521a616d08dbd281023176c50ce688c.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/images/c8cfde03ad730b2a1fdafc0d38c228890521a616d08dbd281023176c50ce688c.jpg)

![e40bca8217855bbaff508a1a3f5795a58a15d7ce2927ac821e61518910998023.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/images/e40bca8217855bbaff508a1a3f5795a58a15d7ce2927ac821e61518910998023.jpg)

### Tables

![0eeea0e22a1ad971431fc0821b13fbdfebbde30e9a2f84ed735cde694391f30b.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/tables/0eeea0e22a1ad971431fc0821b13fbdfebbde30e9a2f84ed735cde694391f30b.jpg)

![0fe9294f19a1046dab4d5d8a3d2787c37db41b9169592465bea9e008f786c344.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/tables/0fe9294f19a1046dab4d5d8a3d2787c37db41b9169592465bea9e008f786c344.jpg)

![324e9662da077480a56ec38928eb8e7b8ec2e4ad39a0698ddb363884016a6a27.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/tables/324e9662da077480a56ec38928eb8e7b8ec2e4ad39a0698ddb363884016a6a27.jpg)

![3f3c3147cb3fd8f54d9d9de6f934eaeba725f12c0852265a68e9e69792f0f4b9.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/tables/3f3c3147cb3fd8f54d9d9de6f934eaeba725f12c0852265a68e9e69792f0f4b9.jpg)

![4379083937b565bce4b8b0159e06dae004726b142071099a85c4c8a8423aa593.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/tables/4379083937b565bce4b8b0159e06dae004726b142071099a85c4c8a8423aa593.jpg)

![50e1581c25302a6acddd1692fae5ac8074ad23d7996be4f5e79aeff8aaeabd92.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/tables/50e1581c25302a6acddd1692fae5ac8074ad23d7996be4f5e79aeff8aaeabd92.jpg)

![56506c011c77515697120c9e229ce4d7429d8fca94321c49e5cec870e20a079b.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/tables/56506c011c77515697120c9e229ce4d7429d8fca94321c49e5cec870e20a079b.jpg)

![db8a1fb59f3abc7b0053bff0a405958245bc46c40a9ace6e55b174e58df37e89.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/tables/db8a1fb59f3abc7b0053bff0a405958245bc46c40a9ace6e55b174e58df37e89.jpg)

![f49a11fda7c6a9d1cee745591863f3d6b658e05be6175cd0503f9b020305a2d9.jpg](../iclr_results/3312_Forte _ Finding Outliers with Representation Typicality Estimation/tables/f49a11fda7c6a9d1cee745591863f3d6b658e05be6175cd0503f9b020305a2d9.jpg)

## Drama: Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient


### Images

![481ddb17eee76d054f2b92c5158a8c4f373bd4e243b05c2e72ca5d1f87f0bc25.jpg](../iclr_results/3313_Drama_ Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient/images/481ddb17eee76d054f2b92c5158a8c4f373bd4e243b05c2e72ca5d1f87f0bc25.jpg)

![8328cbc549a5f3e84c508a0be066db8fb55819d42d69817dd2bf0b8e0d099408.jpg](../iclr_results/3313_Drama_ Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient/images/8328cbc549a5f3e84c508a0be066db8fb55819d42d69817dd2bf0b8e0d099408.jpg)

![8446618808db765a14467edcd3eb66842fcbbdd6628b0dbd30c899f493e3503e.jpg](../iclr_results/3313_Drama_ Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient/images/8446618808db765a14467edcd3eb66842fcbbdd6628b0dbd30c899f493e3503e.jpg)

![85c1cc8fb99c060d684ad586e1f98bec766453b1efba2cb26b1f47ecc7218ade.jpg](../iclr_results/3313_Drama_ Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient/images/85c1cc8fb99c060d684ad586e1f98bec766453b1efba2cb26b1f47ecc7218ade.jpg)

![a5a8f9339c0f608eb013c0dcee47f1bbcdcd50b5f72469ea9364d92691316c9c.jpg](../iclr_results/3313_Drama_ Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient/images/a5a8f9339c0f608eb013c0dcee47f1bbcdcd50b5f72469ea9364d92691316c9c.jpg)

![a911d47dfc573d96f1ea9dc900ab606b73baf94e9288d22b01f1ff2c3091aa0d.jpg](../iclr_results/3313_Drama_ Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient/images/a911d47dfc573d96f1ea9dc900ab606b73baf94e9288d22b01f1ff2c3091aa0d.jpg)

![c85db861acd8dd975a68ae72f02cfe40c2bd8aadab8318a1501acc3b3eb1ba41.jpg](../iclr_results/3313_Drama_ Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient/images/c85db861acd8dd975a68ae72f02cfe40c2bd8aadab8318a1501acc3b3eb1ba41.jpg)

![d85f1c18590d1255362923f8892fa7457d92696380d35a77758ad8616ae2b588.jpg](../iclr_results/3313_Drama_ Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient/images/d85f1c18590d1255362923f8892fa7457d92696380d35a77758ad8616ae2b588.jpg)

![ecd691790e9ceee0f7b9ec3381d6b5dd8fdb6a28ee3d1f39c0f84ed8b0fe11ca.jpg](../iclr_results/3313_Drama_ Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient/images/ecd691790e9ceee0f7b9ec3381d6b5dd8fdb6a28ee3d1f39c0f84ed8b0fe11ca.jpg)

![fddd9de5446a7dff1e324140ed6d50bcb43013fd4df6803825a7a558d6104a99.jpg](../iclr_results/3313_Drama_ Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient/images/fddd9de5446a7dff1e324140ed6d50bcb43013fd4df6803825a7a558d6104a99.jpg)

### Tables

![292f131bcaa064beaa065d4f5d5b87e2d6c1080ef7918c17850158736db0095d.jpg](../iclr_results/3313_Drama_ Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient/tables/292f131bcaa064beaa065d4f5d5b87e2d6c1080ef7918c17850158736db0095d.jpg)

![3cedda08105724f26564d0047e766bc73ce757155f319e466eccbfa5e965da57.jpg](../iclr_results/3313_Drama_ Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient/tables/3cedda08105724f26564d0047e766bc73ce757155f319e466eccbfa5e965da57.jpg)

![47733cd07fb0fc53a230fb67d9b8efd5840c31d271e24a99e4271126796c9481.jpg](../iclr_results/3313_Drama_ Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient/tables/47733cd07fb0fc53a230fb67d9b8efd5840c31d271e24a99e4271126796c9481.jpg)

![5905117b6401daabbb0395f80f76a0e6638ee79daaeff6a92a6e5abbcbeb3b6d.jpg](../iclr_results/3313_Drama_ Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient/tables/5905117b6401daabbb0395f80f76a0e6638ee79daaeff6a92a6e5abbcbeb3b6d.jpg)

![6935fcb57966fed1ab5c10cd217cac50ad523240719364e391ab5ee72aa107fc.jpg](../iclr_results/3313_Drama_ Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient/tables/6935fcb57966fed1ab5c10cd217cac50ad523240719364e391ab5ee72aa107fc.jpg)

![7a51bc6e6f36ba52b7606173e41dba67d1d2d63d7364f48628b76a036061eb28.jpg](../iclr_results/3313_Drama_ Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient/tables/7a51bc6e6f36ba52b7606173e41dba67d1d2d63d7364f48628b76a036061eb28.jpg)

![95964cd6be6b54ffd82849cae223b02c459aedc83d34dcfea3988a9e74c1ac90.jpg](../iclr_results/3313_Drama_ Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient/tables/95964cd6be6b54ffd82849cae223b02c459aedc83d34dcfea3988a9e74c1ac90.jpg)

![a7ea2f6455d6544cd525c7787fd73944daa0eac5f48a041fac6a933e30568fa0.jpg](../iclr_results/3313_Drama_ Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient/tables/a7ea2f6455d6544cd525c7787fd73944daa0eac5f48a041fac6a933e30568fa0.jpg)

![b2db4032e53849058458ec46fb4cc43d570a880e9817e6b0efead39e6be360a0.jpg](../iclr_results/3313_Drama_ Mamba-Enabled Model-Based Reinforcement Learning Is Sample and Parameter Efficient/tables/b2db4032e53849058458ec46fb4cc43d570a880e9817e6b0efead39e6be360a0.jpg)

## NextBestPath: Efficient 3D Mapping of Unseen Environments


### Images

![04b30d77ad332a964ead81c3652a812ccb86e14f3f5730978767beccdd024e67.jpg](../iclr_results/3314_NextBestPath_ Efficient 3D Mapping of Unseen Environments/images/04b30d77ad332a964ead81c3652a812ccb86e14f3f5730978767beccdd024e67.jpg)

![25f4d9aefbdcc434bd3c3b021e6fe51acedc996911da66592d1cbd1d026b23b8.jpg](../iclr_results/3314_NextBestPath_ Efficient 3D Mapping of Unseen Environments/images/25f4d9aefbdcc434bd3c3b021e6fe51acedc996911da66592d1cbd1d026b23b8.jpg)

![339627189e93c28fec44b2c182bf834b6fa65939ca337946ca3a890a6da00749.jpg](../iclr_results/3314_NextBestPath_ Efficient 3D Mapping of Unseen Environments/images/339627189e93c28fec44b2c182bf834b6fa65939ca337946ca3a890a6da00749.jpg)

![397ee20b4926b439ae8ded3af16cd4d448ee84545ea4d01f7460e2be2c17fb8a.jpg](../iclr_results/3314_NextBestPath_ Efficient 3D Mapping of Unseen Environments/images/397ee20b4926b439ae8ded3af16cd4d448ee84545ea4d01f7460e2be2c17fb8a.jpg)

![550a409ce40ac411b79aa99796d2858b40f614036fe6dbd0c2a96d1e2cfd1fa7.jpg](../iclr_results/3314_NextBestPath_ Efficient 3D Mapping of Unseen Environments/images/550a409ce40ac411b79aa99796d2858b40f614036fe6dbd0c2a96d1e2cfd1fa7.jpg)

![c4a5a209c78ff24926750d119d95df174fa2914a1c4b06ff2351d208b3b360ef.jpg](../iclr_results/3314_NextBestPath_ Efficient 3D Mapping of Unseen Environments/images/c4a5a209c78ff24926750d119d95df174fa2914a1c4b06ff2351d208b3b360ef.jpg)

![d43874d02f5afb1b44910c9c051282ed067b7ecc5727c605a480643580f71a7f.jpg](../iclr_results/3314_NextBestPath_ Efficient 3D Mapping of Unseen Environments/images/d43874d02f5afb1b44910c9c051282ed067b7ecc5727c605a480643580f71a7f.jpg)

![e0d4d476f77282f0c13b67f8ea3db68870835b77ff3aff28898aed223d9f7c5a.jpg](../iclr_results/3314_NextBestPath_ Efficient 3D Mapping of Unseen Environments/images/e0d4d476f77282f0c13b67f8ea3db68870835b77ff3aff28898aed223d9f7c5a.jpg)

![f511006fc84bfd8cc1e07c917c4acdfcfce811d0360aeee469e90d7f6d493a55.jpg](../iclr_results/3314_NextBestPath_ Efficient 3D Mapping of Unseen Environments/images/f511006fc84bfd8cc1e07c917c4acdfcfce811d0360aeee469e90d7f6d493a55.jpg)

![f6b50623c01c3e6044f4874db3978c6adca5662500ba5034028d6af4b77bc4eb.jpg](../iclr_results/3314_NextBestPath_ Efficient 3D Mapping of Unseen Environments/images/f6b50623c01c3e6044f4874db3978c6adca5662500ba5034028d6af4b77bc4eb.jpg)

### Tables

![1a34a44a10ab1b630e38e289a9375c090a8eb9b9dd8b2c0503aa00d6826cea98.jpg](../iclr_results/3314_NextBestPath_ Efficient 3D Mapping of Unseen Environments/tables/1a34a44a10ab1b630e38e289a9375c090a8eb9b9dd8b2c0503aa00d6826cea98.jpg)

![6594517521a169526903f6115e0fbaef309c9c3058da2a46177b732766ab8308.jpg](../iclr_results/3314_NextBestPath_ Efficient 3D Mapping of Unseen Environments/tables/6594517521a169526903f6115e0fbaef309c9c3058da2a46177b732766ab8308.jpg)

![6a46adde26848bd1ca02cbd36306b88f4a44999b735782c8d81d748d7e8fa8db.jpg](../iclr_results/3314_NextBestPath_ Efficient 3D Mapping of Unseen Environments/tables/6a46adde26848bd1ca02cbd36306b88f4a44999b735782c8d81d748d7e8fa8db.jpg)

![6a4dbe17c04f095190e43f619467f865ef5aca97744b106cb8560a5b494ef9db.jpg](../iclr_results/3314_NextBestPath_ Efficient 3D Mapping of Unseen Environments/tables/6a4dbe17c04f095190e43f619467f865ef5aca97744b106cb8560a5b494ef9db.jpg)

![6f4193b161d4ab2cb552e3715829d2bd364f960a25b2d5835ff0965e46fca9c0.jpg](../iclr_results/3314_NextBestPath_ Efficient 3D Mapping of Unseen Environments/tables/6f4193b161d4ab2cb552e3715829d2bd364f960a25b2d5835ff0965e46fca9c0.jpg)

![9bc9ad561964b6e34d6e43a0a5bc8cbd69cb206bbdb00fe1da1f68a349f06284.jpg](../iclr_results/3314_NextBestPath_ Efficient 3D Mapping of Unseen Environments/tables/9bc9ad561964b6e34d6e43a0a5bc8cbd69cb206bbdb00fe1da1f68a349f06284.jpg)

![c041dfb09e09ce75d8d6d71a01dd4036c7cd700ff4f4081278def85fcdec44fc.jpg](../iclr_results/3314_NextBestPath_ Efficient 3D Mapping of Unseen Environments/tables/c041dfb09e09ce75d8d6d71a01dd4036c7cd700ff4f4081278def85fcdec44fc.jpg)

![cbac40eb3d6c72f5f191ed6b01013654e06ead7f9b8c21f4944980254dca8d42.jpg](../iclr_results/3314_NextBestPath_ Efficient 3D Mapping of Unseen Environments/tables/cbac40eb3d6c72f5f191ed6b01013654e06ead7f9b8c21f4944980254dca8d42.jpg)

![ccc85ed21b6cfeb301bf1d395a872e096f12fea4ff3cca581d86ff7cddf30126.jpg](../iclr_results/3314_NextBestPath_ Efficient 3D Mapping of Unseen Environments/tables/ccc85ed21b6cfeb301bf1d395a872e096f12fea4ff3cca581d86ff7cddf30126.jpg)

![f84bf31287b07a957d01c9e46ad70fc9eaf8398ab2b28a7b88abb4a445e66d73.jpg](../iclr_results/3314_NextBestPath_ Efficient 3D Mapping of Unseen Environments/tables/f84bf31287b07a957d01c9e46ad70fc9eaf8398ab2b28a7b88abb4a445e66d73.jpg)

## AnoLLM: Large Language Models for Tabular Anomaly Detection


### Images

![03cea5d8d079716823a8c7c7247c141d40caca0cdaa6e728311fb26868cc6a13.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/images/03cea5d8d079716823a8c7c7247c141d40caca0cdaa6e728311fb26868cc6a13.jpg)

![438c031ca90287abf81c9d0fd47f439bf18d15fda961b53e4ea609e7b200e8be.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/images/438c031ca90287abf81c9d0fd47f439bf18d15fda961b53e4ea609e7b200e8be.jpg)

![526a12623653981120ac3303e6e4f24b37d667ee6b420bf766870379a23f04b7.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/images/526a12623653981120ac3303e6e4f24b37d667ee6b420bf766870379a23f04b7.jpg)

![c6545c19bca22b9894c098849188f7db1c331ac88767a33962c71638ea4d45aa.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/images/c6545c19bca22b9894c098849188f7db1c331ac88767a33962c71638ea4d45aa.jpg)

![cc265f57aa6e8baec00f4ab76e607f6091dca3de212d577aeee5e65b30c38492.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/images/cc265f57aa6e8baec00f4ab76e607f6091dca3de212d577aeee5e65b30c38492.jpg)

![ee9ecdb36de3e8ce57d35879950afa1d3552b17a5b2919616a8dac7d59bb4f4b.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/images/ee9ecdb36de3e8ce57d35879950afa1d3552b17a5b2919616a8dac7d59bb4f4b.jpg)

![f24a01e3222c420adbffd7791086e37d36958eb1c0ce663dcd495f544b6fee78.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/images/f24a01e3222c420adbffd7791086e37d36958eb1c0ce663dcd495f544b6fee78.jpg)

### Tables

![10a7dfc29ecc0d416ae872849c5e973f879fd41e5539dbc71922659501ef3dde.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/tables/10a7dfc29ecc0d416ae872849c5e973f879fd41e5539dbc71922659501ef3dde.jpg)

![11c71b6d9d8c99b3b8c4975b38e648975dfdcd845ab5660d38ba7b1074dcef13.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/tables/11c71b6d9d8c99b3b8c4975b38e648975dfdcd845ab5660d38ba7b1074dcef13.jpg)

![277c6080e1b6e2ea5fb2b8344cba432e0e941285a277033e06fa8fb96a47ed19.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/tables/277c6080e1b6e2ea5fb2b8344cba432e0e941285a277033e06fa8fb96a47ed19.jpg)

![339ae3a33c6d7979939731c0bc45d331b2ea1a84af3d598d66ac195c1af5180c.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/tables/339ae3a33c6d7979939731c0bc45d331b2ea1a84af3d598d66ac195c1af5180c.jpg)

![3629fe7eb67a4602a9d13db3d6bd2efa2419885ce8c04c1d0dcd6561631edc8d.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/tables/3629fe7eb67a4602a9d13db3d6bd2efa2419885ce8c04c1d0dcd6561631edc8d.jpg)

![4dce6a1abc0130dbdbabe54c2d0717a58a3548f94e74eef845bc9cdd08f467fd.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/tables/4dce6a1abc0130dbdbabe54c2d0717a58a3548f94e74eef845bc9cdd08f467fd.jpg)

![4e86759c83d1e9e3d88ef74281e47779a9d86373663000741be79b42264abfbe.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/tables/4e86759c83d1e9e3d88ef74281e47779a9d86373663000741be79b42264abfbe.jpg)

![50d03df2584ef889d5f2aaea7b1dc9effefa4037b6682f2fa35dbb88f6a1df64.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/tables/50d03df2584ef889d5f2aaea7b1dc9effefa4037b6682f2fa35dbb88f6a1df64.jpg)

![59e003bc90f41c4c565eb530ab24add466d47fdd2b4571c786becf5b727b3be0.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/tables/59e003bc90f41c4c565eb530ab24add466d47fdd2b4571c786becf5b727b3be0.jpg)

![5e1c56de28655eca85424f6325d64e71a7777b01c7b45ae5f8be7ce23f30f78d.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/tables/5e1c56de28655eca85424f6325d64e71a7777b01c7b45ae5f8be7ce23f30f78d.jpg)

![69e79a1902ad177b2e9f8c1acffeba3d700a356c50d851fbffc8cf718ba8bbc4.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/tables/69e79a1902ad177b2e9f8c1acffeba3d700a356c50d851fbffc8cf718ba8bbc4.jpg)

![7a7432948e2ac3b7092059b38088fcac448edcddb481eedf07e125124f146e90.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/tables/7a7432948e2ac3b7092059b38088fcac448edcddb481eedf07e125124f146e90.jpg)

![83a164cc1e1444e1dce705d6a2792c7f7926eccdaf219fd3375e73c10a951316.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/tables/83a164cc1e1444e1dce705d6a2792c7f7926eccdaf219fd3375e73c10a951316.jpg)

![8910b2e92e54db47431c3662835fdc6c291eb1533169fbc63e41b292d6e57f9b.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/tables/8910b2e92e54db47431c3662835fdc6c291eb1533169fbc63e41b292d6e57f9b.jpg)

![9b7d70ec58898d3e9f1de53ef76b1304d3b46a627a4ae19e3c686eb3d108edc7.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/tables/9b7d70ec58898d3e9f1de53ef76b1304d3b46a627a4ae19e3c686eb3d108edc7.jpg)

![a3e1e6e932fc8c900e264536097c48031e613fa8417205c1789c2cfa583acf95.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/tables/a3e1e6e932fc8c900e264536097c48031e613fa8417205c1789c2cfa583acf95.jpg)

![e828daeaf0e4bd114a2d479fc6773bae052aeec8459775322d93a0111e99876a.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/tables/e828daeaf0e4bd114a2d479fc6773bae052aeec8459775322d93a0111e99876a.jpg)

![f19f81595263c2db85b196d7f5d9c3a5a316982b0d6ebbcf258ce3d6f4d86ee5.jpg](../iclr_results/3315_AnoLLM_ Large Language Models for Tabular Anomaly Detection/tables/f19f81595263c2db85b196d7f5d9c3a5a316982b0d6ebbcf258ce3d6f4d86ee5.jpg)

## xFinder: Large Language Models as Automated Evaluators for Reliable Evaluation


### Images

![004fae7600dfe32394e7c1b5852623787c5e140e48b72ef24e929076223e3667.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/004fae7600dfe32394e7c1b5852623787c5e140e48b72ef24e929076223e3667.jpg)

![1b1bb2525f1f8ec70d121b432270366138886bc243829d3be1c3b54dd5e39539.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/1b1bb2525f1f8ec70d121b432270366138886bc243829d3be1c3b54dd5e39539.jpg)

![2056388aa2aeb67bf9315c0a237f4c197f0c157b4931d1aea4c516d8473bb495.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/2056388aa2aeb67bf9315c0a237f4c197f0c157b4931d1aea4c516d8473bb495.jpg)

![35e7f96fb675c01bedcc47fa1d1b46e207a625f3ffccb1790d662c40f39a6e75.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/35e7f96fb675c01bedcc47fa1d1b46e207a625f3ffccb1790d662c40f39a6e75.jpg)

![3cd0d6e3ea3885b0a87e1f0bd30f3fbf2f6b15bec60a57b13f8db1d5db57e394.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/3cd0d6e3ea3885b0a87e1f0bd30f3fbf2f6b15bec60a57b13f8db1d5db57e394.jpg)

![3e599752e300049ef02a12658a0dc9a5b045b94fdc29894d46f4d305db75aa01.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/3e599752e300049ef02a12658a0dc9a5b045b94fdc29894d46f4d305db75aa01.jpg)

![451787a59d0fc121967d87232cb6a96c75a5ec25b9999e033992585618e363eb.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/451787a59d0fc121967d87232cb6a96c75a5ec25b9999e033992585618e363eb.jpg)

![45cf20416e23323f51fe1ace2f158214d903cb5dfb0d25c7d0cd2535cf960aee.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/45cf20416e23323f51fe1ace2f158214d903cb5dfb0d25c7d0cd2535cf960aee.jpg)

![4860a59a69d9300be10f18db574086cefcd90f02cc6396556fb3a268a8d667e1.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/4860a59a69d9300be10f18db574086cefcd90f02cc6396556fb3a268a8d667e1.jpg)

![5ecc5ba64b4cbf1983e29f3019d3342f1126a7409d317155d030a95a0fd07194.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/5ecc5ba64b4cbf1983e29f3019d3342f1126a7409d317155d030a95a0fd07194.jpg)

![704e069af58e1d1047d2da983abb498757037e347b158afec1dc657e8f3d61c1.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/704e069af58e1d1047d2da983abb498757037e347b158afec1dc657e8f3d61c1.jpg)

![712b60e2a616234ca4928cb07c51e9857e744075bd5552fa5fc52b0c5d8a4040.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/712b60e2a616234ca4928cb07c51e9857e744075bd5552fa5fc52b0c5d8a4040.jpg)

![7a3e3e84be3049269df556db75e990c838de7016924e14af5def71c45df65e02.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/7a3e3e84be3049269df556db75e990c838de7016924e14af5def71c45df65e02.jpg)

![7bf70e656ef3c4e4739bd2175efed7a2b2f27259a9072b5af8199ef26d13ed50.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/7bf70e656ef3c4e4739bd2175efed7a2b2f27259a9072b5af8199ef26d13ed50.jpg)

![7bf7ee943548f13d6b64fb4fbe1c9263c91d08286c38679ccc58954e125c1e7b.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/7bf7ee943548f13d6b64fb4fbe1c9263c91d08286c38679ccc58954e125c1e7b.jpg)

![7e6fdb59ea2908cf85204de7b2f3fb16cdc850c7def1da0ba8d5d0b5e37104aa.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/7e6fdb59ea2908cf85204de7b2f3fb16cdc850c7def1da0ba8d5d0b5e37104aa.jpg)

![80fa64724e3819795f1fe9a7f4452d6633895856eb89e77de1ba06ceb894439b.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/80fa64724e3819795f1fe9a7f4452d6633895856eb89e77de1ba06ceb894439b.jpg)

![90cecb9aa52cfe3e518cbf328e6ef3d1a677e2f7a47024c759ef2b724ebcfe2c.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/90cecb9aa52cfe3e518cbf328e6ef3d1a677e2f7a47024c759ef2b724ebcfe2c.jpg)

![96095826ee2a24120003839a76e5efc015a9a3fe477e0128121e2e7d647147c7.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/96095826ee2a24120003839a76e5efc015a9a3fe477e0128121e2e7d647147c7.jpg)

![9ddcf27d4bd88269331cfcdd46151dad64b09bd3b9c578085c38f1207cf86339.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/9ddcf27d4bd88269331cfcdd46151dad64b09bd3b9c578085c38f1207cf86339.jpg)

![a47452e9200bdbd5ed3446f8fea7310ccc2a40d74c8e8157f4563f698c144570.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/a47452e9200bdbd5ed3446f8fea7310ccc2a40d74c8e8157f4563f698c144570.jpg)

![ab625a044c852749aca6c755cee8ba649344f824b814e5cc3f5b93efbd9cf15a.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/ab625a044c852749aca6c755cee8ba649344f824b814e5cc3f5b93efbd9cf15a.jpg)

![b4f87300d6421170747d5385736ead79ceeebd4e14327c06f0fa0ff40ce44c0f.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/b4f87300d6421170747d5385736ead79ceeebd4e14327c06f0fa0ff40ce44c0f.jpg)

![c1a12afadd186b1d9fda6b5723e116b61c1594d527b9f9c72fa69aa4173c5df7.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/c1a12afadd186b1d9fda6b5723e116b61c1594d527b9f9c72fa69aa4173c5df7.jpg)

![c7e3d70f7ecb0f7b383e06f8a5e41924400f12b1eae74a1e9e0506ef5ea7aaa1.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/c7e3d70f7ecb0f7b383e06f8a5e41924400f12b1eae74a1e9e0506ef5ea7aaa1.jpg)

![ce4dae5cef4626da727c6b7a402b2585adc00a1e3abae833c7467b8c6e14fd30.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/ce4dae5cef4626da727c6b7a402b2585adc00a1e3abae833c7467b8c6e14fd30.jpg)

![eabd6f68017d95b0027a12516d890263beb9dcb07b19fa24782f9b2e70a1fe40.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/eabd6f68017d95b0027a12516d890263beb9dcb07b19fa24782f9b2e70a1fe40.jpg)

![eb986aa15f39071c1d34bca502c4b9cce569e2aa73cdf95d844676b2f636ae8a.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/eb986aa15f39071c1d34bca502c4b9cce569e2aa73cdf95d844676b2f636ae8a.jpg)

![f67562054b5be363e0852ba2d49eb159de94efefeafcfa23518f10eade7c3be8.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/f67562054b5be363e0852ba2d49eb159de94efefeafcfa23518f10eade7c3be8.jpg)

![f85ed0aacf11b1fa93ea650f7cb616d5d5b4d8cb57350cf4cd8405ec070e7e16.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/f85ed0aacf11b1fa93ea650f7cb616d5d5b4d8cb57350cf4cd8405ec070e7e16.jpg)

![ff70bfb892d8bb952a92d8e3fd1695a67f231b869a14065319214a6fb567c34a.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/images/ff70bfb892d8bb952a92d8e3fd1695a67f231b869a14065319214a6fb567c34a.jpg)

### Tables

![03015cf76bb54bcfba8d4e92a96c39b1a68ba20461cb6038e106ed7c8caa1086.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/03015cf76bb54bcfba8d4e92a96c39b1a68ba20461cb6038e106ed7c8caa1086.jpg)

![0436bdf0dafedc3c6960eb44ecc41a9b29908ec2926df477eb091bb1a71bba44.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/0436bdf0dafedc3c6960eb44ecc41a9b29908ec2926df477eb091bb1a71bba44.jpg)

![08c6757a930dcb45130cb13cfb82e843f70014cbcecb65ec7a267013c11cb269.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/08c6757a930dcb45130cb13cfb82e843f70014cbcecb65ec7a267013c11cb269.jpg)

![0c4c984c08c908bb20e1c4e8d457b2a4b5edac3fb2d5c78d27febfdb0847e88d.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/0c4c984c08c908bb20e1c4e8d457b2a4b5edac3fb2d5c78d27febfdb0847e88d.jpg)

![0fefebddd7b157f1ef229a09857707332991323e5a7563aec3f22823d6af71ea.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/0fefebddd7b157f1ef229a09857707332991323e5a7563aec3f22823d6af71ea.jpg)

![17390527fdad60840eb294489fa1600de855a069177c7c2faa7ddde751bbae0e.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/17390527fdad60840eb294489fa1600de855a069177c7c2faa7ddde751bbae0e.jpg)

![1a66665c8d5faaac217054d2642770c6fc5459b29de711e5050ebcdf59bfa8a2.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/1a66665c8d5faaac217054d2642770c6fc5459b29de711e5050ebcdf59bfa8a2.jpg)

![303b3a40d7c58548d75d1e39724b56db369aee91a500a2099d285434d3ebb7a2.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/303b3a40d7c58548d75d1e39724b56db369aee91a500a2099d285434d3ebb7a2.jpg)

![322737f19b3f71305c840e9186e018c393dbdc40229bf14a7acfa567e139f441.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/322737f19b3f71305c840e9186e018c393dbdc40229bf14a7acfa567e139f441.jpg)

![39eda9a01ef19cea3748f4d49409bf3094861a2b818e94167a2041ad69071b84.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/39eda9a01ef19cea3748f4d49409bf3094861a2b818e94167a2041ad69071b84.jpg)

![3ab79c1c06a4aef86f2fc65ee502fb270e14de6ef494bbc57b9a2121f899665e.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/3ab79c1c06a4aef86f2fc65ee502fb270e14de6ef494bbc57b9a2121f899665e.jpg)

![450eb50d2dda4b1980176b6682b4fc51d3c6e79a3a25e28e2babe1d527dc6098.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/450eb50d2dda4b1980176b6682b4fc51d3c6e79a3a25e28e2babe1d527dc6098.jpg)

![45d609156224f965376ddaf39f465349aae82e2947286087716fa1a896e0b175.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/45d609156224f965376ddaf39f465349aae82e2947286087716fa1a896e0b175.jpg)

![4e3295d15e4415c14e40afb82a39d11c80c596c603d346c7e1969db5ac90bf44.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/4e3295d15e4415c14e40afb82a39d11c80c596c603d346c7e1969db5ac90bf44.jpg)

![51ce134e763975a0d510e0923681472eb2334e94f76a7420bf5f7d77977ebf8f.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/51ce134e763975a0d510e0923681472eb2334e94f76a7420bf5f7d77977ebf8f.jpg)

![5c85090f0e7757cdd79febdbed47c0589ff214f211947d8ab9f54ce007fddb94.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/5c85090f0e7757cdd79febdbed47c0589ff214f211947d8ab9f54ce007fddb94.jpg)

![5d720eab67df9ccfaf277a3af12c7ca644cd7b8edb3a39be8c64822c3419cee2.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/5d720eab67df9ccfaf277a3af12c7ca644cd7b8edb3a39be8c64822c3419cee2.jpg)

![61e2db4ffe83132078e7d53342a3df24ca5e67903ae357654719b8ccb69ecdc4.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/61e2db4ffe83132078e7d53342a3df24ca5e67903ae357654719b8ccb69ecdc4.jpg)

![6fa14a34cf7447096a00ec8943e067269b7d39765dced491f43b60f5ae894f6c.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/6fa14a34cf7447096a00ec8943e067269b7d39765dced491f43b60f5ae894f6c.jpg)

![7ad611080c5decba8b11c49f32f4c1fb1436e2046b4ed1e851266272ff05c4ab.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/7ad611080c5decba8b11c49f32f4c1fb1436e2046b4ed1e851266272ff05c4ab.jpg)

![8a857c12380b07dc6e5d7881bd08bc50dfa3ac4f6b33c739b470d37587b06226.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/8a857c12380b07dc6e5d7881bd08bc50dfa3ac4f6b33c739b470d37587b06226.jpg)

![8fbb754fa59597b5f041973802f512f26ad74cdc26c1a45fa80c46de0e17dfe5.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/8fbb754fa59597b5f041973802f512f26ad74cdc26c1a45fa80c46de0e17dfe5.jpg)

![98bf97aa93aff09208c0e1be97501904be411489407c95400d54bce7655d93b1.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/98bf97aa93aff09208c0e1be97501904be411489407c95400d54bce7655d93b1.jpg)

![9d5ce80d2304633531852d6748cd0f9d3e3cf691c323e6f2e32baeecd3ca6490.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/9d5ce80d2304633531852d6748cd0f9d3e3cf691c323e6f2e32baeecd3ca6490.jpg)

![9e999c7b85fdaab16f77580acc6174e2287082ff70c0a88dd7bdec0224e3f594.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/9e999c7b85fdaab16f77580acc6174e2287082ff70c0a88dd7bdec0224e3f594.jpg)

![a33a7390b71f5c621985d748c14d45fdbf1dfb7af3a144f93ea8b68766e56ee6.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/a33a7390b71f5c621985d748c14d45fdbf1dfb7af3a144f93ea8b68766e56ee6.jpg)

![a3e99dd8f29ae11eaf2d2dd93bd76612f8cf695924038d8138d160bf2a502bc0.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/a3e99dd8f29ae11eaf2d2dd93bd76612f8cf695924038d8138d160bf2a502bc0.jpg)

![b0cd7fbb443d36b9245f973df8e18063120b2fcba63e3186f8841e25c0818648.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/b0cd7fbb443d36b9245f973df8e18063120b2fcba63e3186f8841e25c0818648.jpg)

![b41c5ce23a67d50ab2b2bff68b7fef863adf957d57b89bfe53384ea0850f7b86.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/b41c5ce23a67d50ab2b2bff68b7fef863adf957d57b89bfe53384ea0850f7b86.jpg)

![b66366c82002c3d1b7820a6ded1144c5e3f4a7ba8a97f85dcefa9c81be2e13ab.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/b66366c82002c3d1b7820a6ded1144c5e3f4a7ba8a97f85dcefa9c81be2e13ab.jpg)

![bb7d9eff79c5ee178b8768e9c25f649f10923a1629101032d6ccc93bb0602bb6.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/bb7d9eff79c5ee178b8768e9c25f649f10923a1629101032d6ccc93bb0602bb6.jpg)

![be5452ffacc839abd45c70d2dac2aee8ef7af9bb4f0df4ec55509eac87416d7b.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/be5452ffacc839abd45c70d2dac2aee8ef7af9bb4f0df4ec55509eac87416d7b.jpg)

![c8d58a9ae4926646416254559e604b1c257b0b5ae658c5bb3196d00bec241752.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/c8d58a9ae4926646416254559e604b1c257b0b5ae658c5bb3196d00bec241752.jpg)

![d9427f3fe7374aa577026e380deadae8ec3bc483b45b631c03ce60a96ccdadab.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/d9427f3fe7374aa577026e380deadae8ec3bc483b45b631c03ce60a96ccdadab.jpg)

![d9825026c5188023f0a138eee11e6c104374d7a5e6452241486519569aead45e.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/d9825026c5188023f0a138eee11e6c104374d7a5e6452241486519569aead45e.jpg)

![d9b4affeab7c0f789b0609fe6152a0f6e3deb0c4202202fff567d1bb70ad3efd.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/d9b4affeab7c0f789b0609fe6152a0f6e3deb0c4202202fff567d1bb70ad3efd.jpg)

![db7efc8b17548a9f77364f7945fd84de3bd06fd01354d1067f522318275d3219.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/db7efc8b17548a9f77364f7945fd84de3bd06fd01354d1067f522318275d3219.jpg)

![de3d40db7560005f299da7bc8a18015439b6c72673861023e796afa2e3ce92a3.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/de3d40db7560005f299da7bc8a18015439b6c72673861023e796afa2e3ce92a3.jpg)

![e24cde69a121ccf5ca1c59b1d959e5b715562f051543490366f824bf9e8f4634.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/e24cde69a121ccf5ca1c59b1d959e5b715562f051543490366f824bf9e8f4634.jpg)

![eaf5a496ea0add41e1edf246058c30847c3decf62e6baec2337e548d0e5a8d53.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/eaf5a496ea0add41e1edf246058c30847c3decf62e6baec2337e548d0e5a8d53.jpg)

![f554b40149a5c8b0005808d1f032af1b47423703e2ce249d2d8bc16f317fe66b.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/f554b40149a5c8b0005808d1f032af1b47423703e2ce249d2d8bc16f317fe66b.jpg)

![f5d21d1a5d95884cd2bfc02aabaccb079ffaf3a1e7553d85b6a26647a329a4a6.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/f5d21d1a5d95884cd2bfc02aabaccb079ffaf3a1e7553d85b6a26647a329a4a6.jpg)

![fcd53b1a049a076fac185f8ac36e2f8eda299b023838a2cbe784a0bb726498eb.jpg](../iclr_results/3316_xFinder_ Large Language Models as Automated Evaluators for Reliable Evaluation/tables/fcd53b1a049a076fac185f8ac36e2f8eda299b023838a2cbe784a0bb726498eb.jpg)

## Modality-Specialized Synergizers for Interleaved Vision-Language Generalists


### Images

![18d55083d46dbd469ced6d956fdd3deadbdef9cebd7835ca7ef44b121b8937ce.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/images/18d55083d46dbd469ced6d956fdd3deadbdef9cebd7835ca7ef44b121b8937ce.jpg)

![1c45ef442d06caa716a3444e274a0b1111f5533784f05f9d05fe0bb4f2ea636d.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/images/1c45ef442d06caa716a3444e274a0b1111f5533784f05f9d05fe0bb4f2ea636d.jpg)

![3aa975e329950559cd8ef8b9c202ff1b6df9153e53e8a207edc64582a0813400.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/images/3aa975e329950559cd8ef8b9c202ff1b6df9153e53e8a207edc64582a0813400.jpg)

![4e6da0808a01236ee70aafd831761b17303712fd69177b4ba1afcf86c0423d7b.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/images/4e6da0808a01236ee70aafd831761b17303712fd69177b4ba1afcf86c0423d7b.jpg)

![7861c05c8d6289f8d806df64d3c492cb2af2bbf30fda6aecd1e6bc18af5f9300.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/images/7861c05c8d6289f8d806df64d3c492cb2af2bbf30fda6aecd1e6bc18af5f9300.jpg)

![7994b113cc6460712895616cdc2857306f49808feded6947788011e46df5b87b.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/images/7994b113cc6460712895616cdc2857306f49808feded6947788011e46df5b87b.jpg)

![84ff47fab5a4dd98d584975a3c785c99bd3294bdc01e1e2729ee8d4541ffa20d.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/images/84ff47fab5a4dd98d584975a3c785c99bd3294bdc01e1e2729ee8d4541ffa20d.jpg)

![8fd408382f04e05a1fd220b85a2549fdcabaaa8b8c13b61d31fb0363a84a56ff.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/images/8fd408382f04e05a1fd220b85a2549fdcabaaa8b8c13b61d31fb0363a84a56ff.jpg)

![9a9f2a069cdfb321e520d6e2506dd0bef8917021a8355a8048585169f7dea6d7.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/images/9a9f2a069cdfb321e520d6e2506dd0bef8917021a8355a8048585169f7dea6d7.jpg)

![9b97f70136bdfaf784854645caa5584b0b5b0983f8caa522ace3e9f74eda6e31.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/images/9b97f70136bdfaf784854645caa5584b0b5b0983f8caa522ace3e9f74eda6e31.jpg)

![b2295898dafdadd19dd988b1ac5b70e158448df7ad179b301d6523dcd94f96cc.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/images/b2295898dafdadd19dd988b1ac5b70e158448df7ad179b301d6523dcd94f96cc.jpg)

![e54f56215a8f4435982121e0360bce9c47fd33ecfc0c90c1a42d05c53c5b7f86.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/images/e54f56215a8f4435982121e0360bce9c47fd33ecfc0c90c1a42d05c53c5b7f86.jpg)

![f659d69d4d22f4db1ed021e3593568401822c4f2cc0f6ad348141045d9319f81.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/images/f659d69d4d22f4db1ed021e3593568401822c4f2cc0f6ad348141045d9319f81.jpg)

### Tables

![022c52f3861c1966f6263652018d63a0149702fb91cb7e63cc12dae211361995.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/tables/022c52f3861c1966f6263652018d63a0149702fb91cb7e63cc12dae211361995.jpg)

![10815a7d47bca490d81bd5bab97bb4461b549b67dd6a04771ebe94d3c6e32b41.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/tables/10815a7d47bca490d81bd5bab97bb4461b549b67dd6a04771ebe94d3c6e32b41.jpg)

![12922564ebb1ecde732ef38aee31bcfaa0ae34835f2943997931576fe000767b.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/tables/12922564ebb1ecde732ef38aee31bcfaa0ae34835f2943997931576fe000767b.jpg)

![1c9cc4d0a8c2cb697577a02715ede961626eb82968bd4d5965005ce0beeff5d1.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/tables/1c9cc4d0a8c2cb697577a02715ede961626eb82968bd4d5965005ce0beeff5d1.jpg)

![252459275744a9f31f69df538fe76ed9d132d6baf35e7319c0729f9ce5e99ad0.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/tables/252459275744a9f31f69df538fe76ed9d132d6baf35e7319c0729f9ce5e99ad0.jpg)

![2c9246999d1d21c2bdfdf4d3991e4a5d24da13ef8404971fc2c6425805e5a41d.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/tables/2c9246999d1d21c2bdfdf4d3991e4a5d24da13ef8404971fc2c6425805e5a41d.jpg)

![4136de141d0eba2106c31e0033f66931e46b12e231b64a42a3ed80a7d1df980c.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/tables/4136de141d0eba2106c31e0033f66931e46b12e231b64a42a3ed80a7d1df980c.jpg)

![5ce14f04b036ec88efd96004172f3186393426235f639322db4180f1d51eca52.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/tables/5ce14f04b036ec88efd96004172f3186393426235f639322db4180f1d51eca52.jpg)

![7f7648a571b0f812661692e1e876d3dad97bde65e72ff7018b603078e72b0360.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/tables/7f7648a571b0f812661692e1e876d3dad97bde65e72ff7018b603078e72b0360.jpg)

![9809bc4abbefa2150e5a6c4760bf91ccf59a955f014f77cc1096d60561a6cebe.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/tables/9809bc4abbefa2150e5a6c4760bf91ccf59a955f014f77cc1096d60561a6cebe.jpg)

![ac6db4033c02828598458d5e0bfc5b083027167ba8e97e82ed3b116cb306459a.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/tables/ac6db4033c02828598458d5e0bfc5b083027167ba8e97e82ed3b116cb306459a.jpg)

![b15bb3da3095109314e397a0d97100a975b5c3c739e4dddd26c7badf8c0cb3a1.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/tables/b15bb3da3095109314e397a0d97100a975b5c3c739e4dddd26c7badf8c0cb3a1.jpg)

![e19eb1301a2603a4cb21a3f1a6fd2cd96e1e780710105d15c5171c53f7029f07.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/tables/e19eb1301a2603a4cb21a3f1a6fd2cd96e1e780710105d15c5171c53f7029f07.jpg)

![eea0a66e91b3c4cf89ca6da17a40dd115e3ffadffe60e330208ca4d2615f43dd.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/tables/eea0a66e91b3c4cf89ca6da17a40dd115e3ffadffe60e330208ca4d2615f43dd.jpg)

![fd9051b0ee2a31f75a067e9814cd23bac9893661b737210d86ef159153662aa8.jpg](../iclr_results/3317_Modality-Specialized Synergizers for Interleaved Vision-Language Generalists/tables/fd9051b0ee2a31f75a067e9814cd23bac9893661b737210d86ef159153662aa8.jpg)

## Entropy-based Activation Function Optimization: A Method on Searching Better Activation Functions


### Images

![205535ee060bbc5f00dc897787288dd2ef9e64c543167c95ddb975e10550f145.jpg](../iclr_results/3318_Entropy-based Activation Function Optimization_ A Method on Searching Better Activation Functions/images/205535ee060bbc5f00dc897787288dd2ef9e64c543167c95ddb975e10550f145.jpg)

![4c079fee5a906d9d75a5541a1f2f8e688eefcca013fc46f43e72c1c7f01c2a47.jpg](../iclr_results/3318_Entropy-based Activation Function Optimization_ A Method on Searching Better Activation Functions/images/4c079fee5a906d9d75a5541a1f2f8e688eefcca013fc46f43e72c1c7f01c2a47.jpg)

![c373a843c828c12d3cd44499a6aef15dd0fe9c8fbc47dec7ad3aad645f7695ae.jpg](../iclr_results/3318_Entropy-based Activation Function Optimization_ A Method on Searching Better Activation Functions/images/c373a843c828c12d3cd44499a6aef15dd0fe9c8fbc47dec7ad3aad645f7695ae.jpg)

![c554815a8fc6ee3f0059b61e5ccbdc2b5aa9128d52bc7248d5c84a346a1a627a.jpg](../iclr_results/3318_Entropy-based Activation Function Optimization_ A Method on Searching Better Activation Functions/images/c554815a8fc6ee3f0059b61e5ccbdc2b5aa9128d52bc7248d5c84a346a1a627a.jpg)

![c6ce651cf35c488a23f6a856e3b9a6b287560fe0ceea7935c2cc19e3f505f8ab.jpg](../iclr_results/3318_Entropy-based Activation Function Optimization_ A Method on Searching Better Activation Functions/images/c6ce651cf35c488a23f6a856e3b9a6b287560fe0ceea7935c2cc19e3f505f8ab.jpg)

### Tables

![1c2dd56a139f3e79fd5d2ed0984a31cbf37e09ffc087b7e5d933a9e5208ebd11.jpg](../iclr_results/3318_Entropy-based Activation Function Optimization_ A Method on Searching Better Activation Functions/tables/1c2dd56a139f3e79fd5d2ed0984a31cbf37e09ffc087b7e5d933a9e5208ebd11.jpg)

![2660c7b8f2f0de34303d35b9880229f9fe2b09d03dc521a9c686e5447628f960.jpg](../iclr_results/3318_Entropy-based Activation Function Optimization_ A Method on Searching Better Activation Functions/tables/2660c7b8f2f0de34303d35b9880229f9fe2b09d03dc521a9c686e5447628f960.jpg)

![2b65afe3e3b63c7914c78eaaa1f6317bfc314dd721859a52b953a257b7acd80c.jpg](../iclr_results/3318_Entropy-based Activation Function Optimization_ A Method on Searching Better Activation Functions/tables/2b65afe3e3b63c7914c78eaaa1f6317bfc314dd721859a52b953a257b7acd80c.jpg)

![4b9585b663b7e7223b537afbca5a5fa5da77d20a89600df88fa54a34af498ab5.jpg](../iclr_results/3318_Entropy-based Activation Function Optimization_ A Method on Searching Better Activation Functions/tables/4b9585b663b7e7223b537afbca5a5fa5da77d20a89600df88fa54a34af498ab5.jpg)

![6583d1437684366c8f117c932222daa1622d61e3045d7ffae504408d37df2fa0.jpg](../iclr_results/3318_Entropy-based Activation Function Optimization_ A Method on Searching Better Activation Functions/tables/6583d1437684366c8f117c932222daa1622d61e3045d7ffae504408d37df2fa0.jpg)

![663da15ea8c537b6c94479cfc7037c37f3e883b04d8ab1f4a25501e2b8d62010.jpg](../iclr_results/3318_Entropy-based Activation Function Optimization_ A Method on Searching Better Activation Functions/tables/663da15ea8c537b6c94479cfc7037c37f3e883b04d8ab1f4a25501e2b8d62010.jpg)

![6f7b86170b200b00b25ad96631273af2566355cd9c8d5a2365766f88af263a15.jpg](../iclr_results/3318_Entropy-based Activation Function Optimization_ A Method on Searching Better Activation Functions/tables/6f7b86170b200b00b25ad96631273af2566355cd9c8d5a2365766f88af263a15.jpg)

![7b0e4ed8caf92e64e80419c647da0a142984d7cc11a18f01b6b315dea08077ce.jpg](../iclr_results/3318_Entropy-based Activation Function Optimization_ A Method on Searching Better Activation Functions/tables/7b0e4ed8caf92e64e80419c647da0a142984d7cc11a18f01b6b315dea08077ce.jpg)

![a17e405912dc10a278df2ee8e6def9deafbcd239f60e9e4c0daed2eb9012e2c2.jpg](../iclr_results/3318_Entropy-based Activation Function Optimization_ A Method on Searching Better Activation Functions/tables/a17e405912dc10a278df2ee8e6def9deafbcd239f60e9e4c0daed2eb9012e2c2.jpg)

![afd22940f87f942c837def292de9ce3f71bc490004c94812c97479fdb8339625.jpg](../iclr_results/3318_Entropy-based Activation Function Optimization_ A Method on Searching Better Activation Functions/tables/afd22940f87f942c837def292de9ce3f71bc490004c94812c97479fdb8339625.jpg)

![b40a5fb5aba100ac18f355275687228181003c5ac936619151548ba8dfed3fd9.jpg](../iclr_results/3318_Entropy-based Activation Function Optimization_ A Method on Searching Better Activation Functions/tables/b40a5fb5aba100ac18f355275687228181003c5ac936619151548ba8dfed3fd9.jpg)

![fa5aa3b6600d6275e52ffeb08b5459734c930a14f64824562b8852c51122ac2e.jpg](../iclr_results/3318_Entropy-based Activation Function Optimization_ A Method on Searching Better Activation Functions/tables/fa5aa3b6600d6275e52ffeb08b5459734c930a14f64824562b8852c51122ac2e.jpg)

![ff51437371d92c16c65f652d23d51bcf11c7ab633c8852246c3ad91e4985eb80.jpg](../iclr_results/3318_Entropy-based Activation Function Optimization_ A Method on Searching Better Activation Functions/tables/ff51437371d92c16c65f652d23d51bcf11c7ab633c8852246c3ad91e4985eb80.jpg)

## AI Sandbagging: Language Models can Strategically Underperform on Evaluations


### Images

![06ec1876722c84f29d237af14a235043c272f07e8d04752ca8d03b429326056d.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/images/06ec1876722c84f29d237af14a235043c272f07e8d04752ca8d03b429326056d.jpg)

![0ff052f8c29fcc93debdacdd9ee091949a80d74b47603ac8bb2e49fba449cde3.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/images/0ff052f8c29fcc93debdacdd9ee091949a80d74b47603ac8bb2e49fba449cde3.jpg)

![2156ac10060220f4a39fe09b90a4ac4ea41fb40a6ca664d81b71af2a30adfde3.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/images/2156ac10060220f4a39fe09b90a4ac4ea41fb40a6ca664d81b71af2a30adfde3.jpg)

![23ae335fcb59273267bfeb2d66aa83af3d8c7978bda04dd915bf120d4f31ab11.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/images/23ae335fcb59273267bfeb2d66aa83af3d8c7978bda04dd915bf120d4f31ab11.jpg)

![3318c507067b5d5dc4f71038db7e88c95b730b3743987c6900bb371858f536f6.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/images/3318c507067b5d5dc4f71038db7e88c95b730b3743987c6900bb371858f536f6.jpg)

![5680cc2313d8b7f98383877f58d4647fd48520f5f31d763f4df4718ce5e16b5a.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/images/5680cc2313d8b7f98383877f58d4647fd48520f5f31d763f4df4718ce5e16b5a.jpg)

![614c23b1af4f6419367965905a3c681d7a62b57f82f1c2e56576aa69d518922f.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/images/614c23b1af4f6419367965905a3c681d7a62b57f82f1c2e56576aa69d518922f.jpg)

![7f31bd37fbb5bdedd14f618ef42af575a06972a191ce6266d67305551e2fc0f2.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/images/7f31bd37fbb5bdedd14f618ef42af575a06972a191ce6266d67305551e2fc0f2.jpg)

![801f2582e44cd6011eb8b94f253b8080b639d7a1c66ff2574213aa76b63e5926.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/images/801f2582e44cd6011eb8b94f253b8080b639d7a1c66ff2574213aa76b63e5926.jpg)

![81258ffbb97eaa0b0a3f8087c132e2dc5d31d5f589cac6bde99570550bd5241d.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/images/81258ffbb97eaa0b0a3f8087c132e2dc5d31d5f589cac6bde99570550bd5241d.jpg)

![8d0320a57b9f5378580818c24996abe23279f9be80de95d3792c45c42b688c2e.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/images/8d0320a57b9f5378580818c24996abe23279f9be80de95d3792c45c42b688c2e.jpg)

![903dc733342462b556a1cb17a78c42f2b3e4866a5904bd17807218ea4054d600.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/images/903dc733342462b556a1cb17a78c42f2b3e4866a5904bd17807218ea4054d600.jpg)

![a8d3f9bf7fee0489c1445b3bb4766e0dfde23b131d7f25292775280516091817.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/images/a8d3f9bf7fee0489c1445b3bb4766e0dfde23b131d7f25292775280516091817.jpg)

![a9151b33771e6d71dfa1c344e1bf217d41b7a9a3e527c7e59777f8b40b6c8ae6.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/images/a9151b33771e6d71dfa1c344e1bf217d41b7a9a3e527c7e59777f8b40b6c8ae6.jpg)

![ab83fdde33b07a068b922bafc9b0758eaf0a5eb3f273ccf51901ecea85eb2930.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/images/ab83fdde33b07a068b922bafc9b0758eaf0a5eb3f273ccf51901ecea85eb2930.jpg)

![ac69ed2088112cd2263dea3888a2fbab9daad9dc82307edc226b0dc2ed965c3f.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/images/ac69ed2088112cd2263dea3888a2fbab9daad9dc82307edc226b0dc2ed965c3f.jpg)

![c5f5706941da80c4f8dff852efe6bc68592c36156cab65d256174b81e744008a.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/images/c5f5706941da80c4f8dff852efe6bc68592c36156cab65d256174b81e744008a.jpg)

![c853facc73fa0d92c9de5a061e0acd187d4f86bf12fb789e6251faf21dd484a7.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/images/c853facc73fa0d92c9de5a061e0acd187d4f86bf12fb789e6251faf21dd484a7.jpg)

![cea038d02031a25bc85f76f577c5ef6c0b7d90f911a9b6c08ca313b5b1baa90f.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/images/cea038d02031a25bc85f76f577c5ef6c0b7d90f911a9b6c08ca313b5b1baa90f.jpg)

![ee16e13ecae18d9f8323c9acfd7521a5140b6b51371c8a4526158b45c03f9170.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/images/ee16e13ecae18d9f8323c9acfd7521a5140b6b51371c8a4526158b45c03f9170.jpg)

![efd26a944b7be02f02f52e2af3a84f17cb4b9a0d206fb9e7655e6aa9034106b3.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/images/efd26a944b7be02f02f52e2af3a84f17cb4b9a0d206fb9e7655e6aa9034106b3.jpg)

### Tables

![215525d00b588a23542c22c68637b956cde148e312b5fe16d714c34606926a3b.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/tables/215525d00b588a23542c22c68637b956cde148e312b5fe16d714c34606926a3b.jpg)

![50dcb788fa5608fb59178e6a57613c2a087376485d37830017f87a428c8cde03.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/tables/50dcb788fa5608fb59178e6a57613c2a087376485d37830017f87a428c8cde03.jpg)

![892bb1673cd36fdfe31933e1052cdb73c0f1bbdc87a6ba023beb16c3c676f070.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/tables/892bb1673cd36fdfe31933e1052cdb73c0f1bbdc87a6ba023beb16c3c676f070.jpg)

![909e47c0fd1cdbfdae0d2b749bc7a2ec1b2e07a6b270ba401f56b847f51e5117.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/tables/909e47c0fd1cdbfdae0d2b749bc7a2ec1b2e07a6b270ba401f56b847f51e5117.jpg)

![94c1b83b5234f9540c576ed7c0ce409a5793078456e582e218f09821e41bd319.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/tables/94c1b83b5234f9540c576ed7c0ce409a5793078456e582e218f09821e41bd319.jpg)

![981f716ab4061b98e4728e62011031522d0178a290a23d55fa11e28055dcce50.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/tables/981f716ab4061b98e4728e62011031522d0178a290a23d55fa11e28055dcce50.jpg)

![b1571366a9f7cef81d7a0b6578ecc2bd71bb24e355dba0dc3fdd5fd6f6ec3517.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/tables/b1571366a9f7cef81d7a0b6578ecc2bd71bb24e355dba0dc3fdd5fd6f6ec3517.jpg)

![b15b82c0a596c34c2cf504a9c5dad11d152446d317166e363de5a8f6e13ef822.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/tables/b15b82c0a596c34c2cf504a9c5dad11d152446d317166e363de5a8f6e13ef822.jpg)

![ce5565a7514c048f8eb34b60c3727fe3ae3aa72d757c8faaa9b4ca2f493f03a0.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/tables/ce5565a7514c048f8eb34b60c3727fe3ae3aa72d757c8faaa9b4ca2f493f03a0.jpg)

![dc2188ebf25706f5958424c6d34653755a0c3162f15a34026d3722cbe0dc2350.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/tables/dc2188ebf25706f5958424c6d34653755a0c3162f15a34026d3722cbe0dc2350.jpg)

![e3465f2e8921a344b0f69c4c3fb5601c20aac2281312387ad6378beeea9ba33b.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/tables/e3465f2e8921a344b0f69c4c3fb5601c20aac2281312387ad6378beeea9ba33b.jpg)

![ece87f4e4f21fb148584f601e6f4d80371ae0ba6bee79b7f80e6e9b1f232f79e.jpg](../iclr_results/3319_AI Sandbagging_ Language Models can Strategically Underperform on Evaluations/tables/ece87f4e4f21fb148584f601e6f4d80371ae0ba6bee79b7f80e6e9b1f232f79e.jpg)

## Lie Algebra Canonicalization: Equivariant Neural Operators under arbitrary Lie Groups


### Images

![05dbb97d86699687f29271bad31f214399eeb6dbdfab934e70f1b9a90c2ed425.jpg](../iclr_results/3320_Lie Algebra Canonicalization_ Equivariant Neural Operators under arbitrary Lie Groups/images/05dbb97d86699687f29271bad31f214399eeb6dbdfab934e70f1b9a90c2ed425.jpg)

![35c4ddb2a57ea1dbcd2beb8795bf7b8b5c9db5b09b0b9948eac8ac583e25931e.jpg](../iclr_results/3320_Lie Algebra Canonicalization_ Equivariant Neural Operators under arbitrary Lie Groups/images/35c4ddb2a57ea1dbcd2beb8795bf7b8b5c9db5b09b0b9948eac8ac583e25931e.jpg)

![6e2d71eb6ca697d187ee5dc5af50b763fbfd18fb0d680a0bb8a14db7e1f50505.jpg](../iclr_results/3320_Lie Algebra Canonicalization_ Equivariant Neural Operators under arbitrary Lie Groups/images/6e2d71eb6ca697d187ee5dc5af50b763fbfd18fb0d680a0bb8a14db7e1f50505.jpg)

![6eeabd493a3c45d8237e7e059e61d0dce5b19ba87e0e96a4dfd4b1940a0e7b85.jpg](../iclr_results/3320_Lie Algebra Canonicalization_ Equivariant Neural Operators under arbitrary Lie Groups/images/6eeabd493a3c45d8237e7e059e61d0dce5b19ba87e0e96a4dfd4b1940a0e7b85.jpg)

![7cc735ea70105f522518e4fbdfa60344adfe23a8fa89b63054afd4b92bbd8765.jpg](../iclr_results/3320_Lie Algebra Canonicalization_ Equivariant Neural Operators under arbitrary Lie Groups/images/7cc735ea70105f522518e4fbdfa60344adfe23a8fa89b63054afd4b92bbd8765.jpg)

![8333bfeeab65133e0f82abbb7b55f69a45690fa48f40f79f17550f7b4dd4e4b0.jpg](../iclr_results/3320_Lie Algebra Canonicalization_ Equivariant Neural Operators under arbitrary Lie Groups/images/8333bfeeab65133e0f82abbb7b55f69a45690fa48f40f79f17550f7b4dd4e4b0.jpg)

![9ccdbdb038ba4f50350725e9e428c05632083c69de2ca54eb79975fece84bf9c.jpg](../iclr_results/3320_Lie Algebra Canonicalization_ Equivariant Neural Operators under arbitrary Lie Groups/images/9ccdbdb038ba4f50350725e9e428c05632083c69de2ca54eb79975fece84bf9c.jpg)

![a0254d8701a6720f57fa322df7bcfeca98b8f92af9b7fd3b88259431cd8eb253.jpg](../iclr_results/3320_Lie Algebra Canonicalization_ Equivariant Neural Operators under arbitrary Lie Groups/images/a0254d8701a6720f57fa322df7bcfeca98b8f92af9b7fd3b88259431cd8eb253.jpg)

![ac0a2fd2f521923c302b02452698246ba939fbbf144bc8be77aff6d27d3100b1.jpg](../iclr_results/3320_Lie Algebra Canonicalization_ Equivariant Neural Operators under arbitrary Lie Groups/images/ac0a2fd2f521923c302b02452698246ba939fbbf144bc8be77aff6d27d3100b1.jpg)

![adc368ed839d92f759a87c39667750f4797e7a59be475c658e8c632eca639bde.jpg](../iclr_results/3320_Lie Algebra Canonicalization_ Equivariant Neural Operators under arbitrary Lie Groups/images/adc368ed839d92f759a87c39667750f4797e7a59be475c658e8c632eca639bde.jpg)

![b39a9b16a2e18b7fb099a63faae1ecd39b8cbaf4f00b7304adb189e694a895c1.jpg](../iclr_results/3320_Lie Algebra Canonicalization_ Equivariant Neural Operators under arbitrary Lie Groups/images/b39a9b16a2e18b7fb099a63faae1ecd39b8cbaf4f00b7304adb189e694a895c1.jpg)

![c0cb822593fb2ef884398068e2aaff1839280336d5a01c96e04cc6bf295df093.jpg](../iclr_results/3320_Lie Algebra Canonicalization_ Equivariant Neural Operators under arbitrary Lie Groups/images/c0cb822593fb2ef884398068e2aaff1839280336d5a01c96e04cc6bf295df093.jpg)

![c3da5062bdfc7f45e2aed51bdd1a304bedcc45214cef3764aee1bfb3e6b68cdf.jpg](../iclr_results/3320_Lie Algebra Canonicalization_ Equivariant Neural Operators under arbitrary Lie Groups/images/c3da5062bdfc7f45e2aed51bdd1a304bedcc45214cef3764aee1bfb3e6b68cdf.jpg)

![e9d15ac4afd27c23becd90fe8ad215bb2adbc73f3d558b7481d86ebdf6253aca.jpg](../iclr_results/3320_Lie Algebra Canonicalization_ Equivariant Neural Operators under arbitrary Lie Groups/images/e9d15ac4afd27c23becd90fe8ad215bb2adbc73f3d558b7481d86ebdf6253aca.jpg)

### Tables

![1302dd7dcfd3a33b3334d8cd0fdfcce938683b90d650d0fb54695e810588ef66.jpg](../iclr_results/3320_Lie Algebra Canonicalization_ Equivariant Neural Operators under arbitrary Lie Groups/tables/1302dd7dcfd3a33b3334d8cd0fdfcce938683b90d650d0fb54695e810588ef66.jpg)

![4c218735e8e0dd90e561836dd378e81d8b1d57ac4e012b6b1e523fc66a1ba02b.jpg](../iclr_results/3320_Lie Algebra Canonicalization_ Equivariant Neural Operators under arbitrary Lie Groups/tables/4c218735e8e0dd90e561836dd378e81d8b1d57ac4e012b6b1e523fc66a1ba02b.jpg)

![9db1f6b61c3271ba523badf4331e0c270314ea54a2142700033233d5109c971b.jpg](../iclr_results/3320_Lie Algebra Canonicalization_ Equivariant Neural Operators under arbitrary Lie Groups/tables/9db1f6b61c3271ba523badf4331e0c270314ea54a2142700033233d5109c971b.jpg)

![c517d0b38d9f403a045f878d2dd6951a2a96d99b10a796280fadf45bab36eb44.jpg](../iclr_results/3320_Lie Algebra Canonicalization_ Equivariant Neural Operators under arbitrary Lie Groups/tables/c517d0b38d9f403a045f878d2dd6951a2a96d99b10a796280fadf45bab36eb44.jpg)

## Logically Consistent Language Models via Neuro-Symbolic Integration


### Images

![1a9b8ab2510b5d28f2caec07ce0721eeba2f791406f4a2644aea5506a6695864.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/images/1a9b8ab2510b5d28f2caec07ce0721eeba2f791406f4a2644aea5506a6695864.jpg)

![228ec098a1372b0ae48301ae03096bc32d5505d7be2b01afc7656314b3e015e8.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/images/228ec098a1372b0ae48301ae03096bc32d5505d7be2b01afc7656314b3e015e8.jpg)

![427fa64269bbcdfd53278ebdafc61358d608058309bc682ed41d0f8ccab83325.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/images/427fa64269bbcdfd53278ebdafc61358d608058309bc682ed41d0f8ccab83325.jpg)

![8acf58ee648d6096f4190a953853d11a068662775602b241cfe8a24aac499d7e.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/images/8acf58ee648d6096f4190a953853d11a068662775602b241cfe8a24aac499d7e.jpg)

![8c81c70e78d7664eecb084cfc5c2ad6fb147fa4d4e910fbb3b6bbdf784203358.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/images/8c81c70e78d7664eecb084cfc5c2ad6fb147fa4d4e910fbb3b6bbdf784203358.jpg)

![8e3031ce3e33bc2b9db0f8c9b6b1be135bcad8382b05dfb6302c2fc3e8179617.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/images/8e3031ce3e33bc2b9db0f8c9b6b1be135bcad8382b05dfb6302c2fc3e8179617.jpg)

![a527a1977da57e0a9b98f7485d7ece754713a08d54ee15dc73bde127916c5d61.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/images/a527a1977da57e0a9b98f7485d7ece754713a08d54ee15dc73bde127916c5d61.jpg)

![b51e59f687ad42927350f98c3850de7d71f1b799bb9a524fbfd9b8df575e0845.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/images/b51e59f687ad42927350f98c3850de7d71f1b799bb9a524fbfd9b8df575e0845.jpg)

![cf1614adbc91cc7fff469e99ea1cba79d28acfdd3b3bc0565e40bbd3fc8c5226.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/images/cf1614adbc91cc7fff469e99ea1cba79d28acfdd3b3bc0565e40bbd3fc8c5226.jpg)

![f0e14f8b9fe528687220b3811a870948b5a42d840ccaaf1a030b0b758678226f.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/images/f0e14f8b9fe528687220b3811a870948b5a42d840ccaaf1a030b0b758678226f.jpg)

### Tables

![1ce3f3a5e7174905cdc767ab43aaf37aa8ad7b7a36837674520c200bb1451bc9.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/tables/1ce3f3a5e7174905cdc767ab43aaf37aa8ad7b7a36837674520c200bb1451bc9.jpg)

![41d53bd18420d6550e7941b7c30455bbd8cf275443ae60e406dc83ba7021fde4.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/tables/41d53bd18420d6550e7941b7c30455bbd8cf275443ae60e406dc83ba7021fde4.jpg)

![4a7c0592ebf77f36f345bf1865beae6cc35410190eae2a5d6035c0b6311103c9.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/tables/4a7c0592ebf77f36f345bf1865beae6cc35410190eae2a5d6035c0b6311103c9.jpg)

![51f36bbfb2486a18162b06124e5e8d8ebbd6d58a40f8d1ec9c206180ae789f27.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/tables/51f36bbfb2486a18162b06124e5e8d8ebbd6d58a40f8d1ec9c206180ae789f27.jpg)

![5c159bad5c44bd4b5f10bb9c1663619e8682a136b3bc3facbe35294b5f043712.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/tables/5c159bad5c44bd4b5f10bb9c1663619e8682a136b3bc3facbe35294b5f043712.jpg)

![5cf17f202a711ace6c3d6fc4dd2ca44155dbfb1fc4f6e1c6dd2a6012df2d0cbb.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/tables/5cf17f202a711ace6c3d6fc4dd2ca44155dbfb1fc4f6e1c6dd2a6012df2d0cbb.jpg)

![6bf9dbe6437284692f36821f695058b5e02fa17da38cddff0a4b4a08ff4ee6bd.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/tables/6bf9dbe6437284692f36821f695058b5e02fa17da38cddff0a4b4a08ff4ee6bd.jpg)

![751f5ca87a4477cf4ac875abf655cd1493f84a3b0e97abf58b3d698fe94180b6.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/tables/751f5ca87a4477cf4ac875abf655cd1493f84a3b0e97abf58b3d698fe94180b6.jpg)

![7a56b74cf402e0383d1acce67a49d2f383f2bc8261d5c632df9e2fea5de28b39.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/tables/7a56b74cf402e0383d1acce67a49d2f383f2bc8261d5c632df9e2fea5de28b39.jpg)

![7d385df0e75c9bd8459f5008ab6c5da6ab113e0f27791db043b160b442286c93.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/tables/7d385df0e75c9bd8459f5008ab6c5da6ab113e0f27791db043b160b442286c93.jpg)

![9824444c57e0fe7c7d7588f1c6a7b07e284cf059876bd705bf54d3f0711eb44d.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/tables/9824444c57e0fe7c7d7588f1c6a7b07e284cf059876bd705bf54d3f0711eb44d.jpg)

![d265d23b1cd5b258db46271e132a29893a9214a96dfb09b18b4d34d1396f6507.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/tables/d265d23b1cd5b258db46271e132a29893a9214a96dfb09b18b4d34d1396f6507.jpg)

![d35e8806479ce5f7024f20c95d2ee67830543b1437437583bf4738c031d015c6.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/tables/d35e8806479ce5f7024f20c95d2ee67830543b1437437583bf4738c031d015c6.jpg)

![d6e82695207cc2094dde0f94b0439ded40ed86927d91502063b5c40f18bcfbf8.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/tables/d6e82695207cc2094dde0f94b0439ded40ed86927d91502063b5c40f18bcfbf8.jpg)

![db3c3ad3b0c61d1febabb2b3267baaabeecef975b20f7aab49640759b3a0363e.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/tables/db3c3ad3b0c61d1febabb2b3267baaabeecef975b20f7aab49640759b3a0363e.jpg)

![dbd4eb2ef27ad9c2b75ff87311b85ad7ea71caac6de4e02deb8dd3f1b282e293.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/tables/dbd4eb2ef27ad9c2b75ff87311b85ad7ea71caac6de4e02deb8dd3f1b282e293.jpg)

![e9d0ae01f863f86adabbf5c044a913284d6e63e33e840a6a0403e4f74c93f23d.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/tables/e9d0ae01f863f86adabbf5c044a913284d6e63e33e840a6a0403e4f74c93f23d.jpg)

![f1823342f8379a9a52e0d7a82445e6e90d4ba63007915794e78062b6295ed304.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/tables/f1823342f8379a9a52e0d7a82445e6e90d4ba63007915794e78062b6295ed304.jpg)

![f417183f24d00aadea8b6eded9ea73103ab2ec5362260a3287ca5d920eb44363.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/tables/f417183f24d00aadea8b6eded9ea73103ab2ec5362260a3287ca5d920eb44363.jpg)

![f6756780869d3bf50b70575b88a05f14ae07708f4bd83bbaba2878d7d77b1514.jpg](../iclr_results/3321_Logically Consistent Language Models via Neuro-Symbolic Integration/tables/f6756780869d3bf50b70575b88a05f14ae07708f4bd83bbaba2878d7d77b1514.jpg)

## Alchemy: Amplifying Theorem-Proving Capability Through Symbolic Mutation


### Images

![0213a3c3a200a5845f715640a8911aabfc6945eb2c34c4fa8187f818de9050ee.jpg](../iclr_results/3322_Alchemy_ Amplifying Theorem-Proving Capability Through Symbolic Mutation/images/0213a3c3a200a5845f715640a8911aabfc6945eb2c34c4fa8187f818de9050ee.jpg)

![073ca21e59313f11dd58a8695c9f57ee1a760ed568a5efc44ee173652d1da282.jpg](../iclr_results/3322_Alchemy_ Amplifying Theorem-Proving Capability Through Symbolic Mutation/images/073ca21e59313f11dd58a8695c9f57ee1a760ed568a5efc44ee173652d1da282.jpg)

![5157bf6b389a83e6ac624f18d21b5576de2a4c7703894f01fd1dd4ff5828a183.jpg](../iclr_results/3322_Alchemy_ Amplifying Theorem-Proving Capability Through Symbolic Mutation/images/5157bf6b389a83e6ac624f18d21b5576de2a4c7703894f01fd1dd4ff5828a183.jpg)

![57a31df0a24ad7b71ace28f11bf88b5a549048bd6ed3c59a5b963aaef1d8f662.jpg](../iclr_results/3322_Alchemy_ Amplifying Theorem-Proving Capability Through Symbolic Mutation/images/57a31df0a24ad7b71ace28f11bf88b5a549048bd6ed3c59a5b963aaef1d8f662.jpg)

![5f36665fc73e0302a09f0bfa78fbb567ab8687dffdf31d8716ca60542523f935.jpg](../iclr_results/3322_Alchemy_ Amplifying Theorem-Proving Capability Through Symbolic Mutation/images/5f36665fc73e0302a09f0bfa78fbb567ab8687dffdf31d8716ca60542523f935.jpg)

![6f5df6f9ba316f19a3e2bd5b035f180fdc65c9f3b42d66ca3e437bf87403791e.jpg](../iclr_results/3322_Alchemy_ Amplifying Theorem-Proving Capability Through Symbolic Mutation/images/6f5df6f9ba316f19a3e2bd5b035f180fdc65c9f3b42d66ca3e437bf87403791e.jpg)

![77951a5e7f99a6be7e6406a76ebded71e5b26b535e67ff97f1f6663d393451e2.jpg](../iclr_results/3322_Alchemy_ Amplifying Theorem-Proving Capability Through Symbolic Mutation/images/77951a5e7f99a6be7e6406a76ebded71e5b26b535e67ff97f1f6663d393451e2.jpg)

![840e995a45c46a6a77dbddec9b3e96a303104011fd591a7785f941d4533badd0.jpg](../iclr_results/3322_Alchemy_ Amplifying Theorem-Proving Capability Through Symbolic Mutation/images/840e995a45c46a6a77dbddec9b3e96a303104011fd591a7785f941d4533badd0.jpg)

![86227e26ec4af91742a4eb411c2b4686b7cba7248d8a8a34f420af46529708af.jpg](../iclr_results/3322_Alchemy_ Amplifying Theorem-Proving Capability Through Symbolic Mutation/images/86227e26ec4af91742a4eb411c2b4686b7cba7248d8a8a34f420af46529708af.jpg)

![c62116e674ea1d40864e9fe1338188bbbcbae819f35b1d6f4efa53deeea430c5.jpg](../iclr_results/3322_Alchemy_ Amplifying Theorem-Proving Capability Through Symbolic Mutation/images/c62116e674ea1d40864e9fe1338188bbbcbae819f35b1d6f4efa53deeea430c5.jpg)

![f3dbce095fedfe1b0e6253e5983fb648af9cd3fbb330d2b5b47e1b887dc393b2.jpg](../iclr_results/3322_Alchemy_ Amplifying Theorem-Proving Capability Through Symbolic Mutation/images/f3dbce095fedfe1b0e6253e5983fb648af9cd3fbb330d2b5b47e1b887dc393b2.jpg)

### Tables

![02717855f417a3c0a002588297a1b16759e9ec8616ee2975befa61fcd86e5990.jpg](../iclr_results/3322_Alchemy_ Amplifying Theorem-Proving Capability Through Symbolic Mutation/tables/02717855f417a3c0a002588297a1b16759e9ec8616ee2975befa61fcd86e5990.jpg)

![0f6850a64ef1faa7671624906663a0e61613f9a46601983a7988963982b9844f.jpg](../iclr_results/3322_Alchemy_ Amplifying Theorem-Proving Capability Through Symbolic Mutation/tables/0f6850a64ef1faa7671624906663a0e61613f9a46601983a7988963982b9844f.jpg)

![32e83cab451c1743c7af9baef138a827a37c2bd3ef1a6e8162a4ea876cb1e621.jpg](../iclr_results/3322_Alchemy_ Amplifying Theorem-Proving Capability Through Symbolic Mutation/tables/32e83cab451c1743c7af9baef138a827a37c2bd3ef1a6e8162a4ea876cb1e621.jpg)

![3def3085d84c4e90019fd6435cd2d373b43411a40c81b179a37fc011160545dc.jpg](../iclr_results/3322_Alchemy_ Amplifying Theorem-Proving Capability Through Symbolic Mutation/tables/3def3085d84c4e90019fd6435cd2d373b43411a40c81b179a37fc011160545dc.jpg)

![468cd8c309559b3b5d658301832f9acc99ce10baad40866a7145c5eeed019ee3.jpg](../iclr_results/3322_Alchemy_ Amplifying Theorem-Proving Capability Through Symbolic Mutation/tables/468cd8c309559b3b5d658301832f9acc99ce10baad40866a7145c5eeed019ee3.jpg)

![52a660762901d73a080e5b532edf9c05cf01cb78c8f8a58924b8204a00d94922.jpg](../iclr_results/3322_Alchemy_ Amplifying Theorem-Proving Capability Through Symbolic Mutation/tables/52a660762901d73a080e5b532edf9c05cf01cb78c8f8a58924b8204a00d94922.jpg)

![6748e146bf14145609969b48e4c8e2c3b8ecc3ca6e4d4f4453ed47f671981ab1.jpg](../iclr_results/3322_Alchemy_ Amplifying Theorem-Proving Capability Through Symbolic Mutation/tables/6748e146bf14145609969b48e4c8e2c3b8ecc3ca6e4d4f4453ed47f671981ab1.jpg)

![aa85de12d45aa708f58691fbd101b77727337a2d4d7374b9c5462a696e6a704f.jpg](../iclr_results/3322_Alchemy_ Amplifying Theorem-Proving Capability Through Symbolic Mutation/tables/aa85de12d45aa708f58691fbd101b77727337a2d4d7374b9c5462a696e6a704f.jpg)

![cf0b3c3672b7e1d37c7ceb28df3e4f8b2d07f76c6908bcd5288dbf0334e2822e.jpg](../iclr_results/3322_Alchemy_ Amplifying Theorem-Proving Capability Through Symbolic Mutation/tables/cf0b3c3672b7e1d37c7ceb28df3e4f8b2d07f76c6908bcd5288dbf0334e2822e.jpg)

![cfa0dbd89af980440947ed0231f17b19d2f969c125784958078bd35c536c0cac.jpg](../iclr_results/3322_Alchemy_ Amplifying Theorem-Proving Capability Through Symbolic Mutation/tables/cfa0dbd89af980440947ed0231f17b19d2f969c125784958078bd35c536c0cac.jpg)

![eac01fc7e200334035ac2d8829bff3c9b595a0052c9847c8d43077b6fcaa6e4d.jpg](../iclr_results/3322_Alchemy_ Amplifying Theorem-Proving Capability Through Symbolic Mutation/tables/eac01fc7e200334035ac2d8829bff3c9b595a0052c9847c8d43077b6fcaa6e4d.jpg)

## Examining Alignment of Large Language Models through Representative Heuristics: the case of political stereotypes


### Images

![4846e955797d29a16ccdf4ae75a707058cd8ce842c024bf348aa718b7bc47237.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/images/4846e955797d29a16ccdf4ae75a707058cd8ce842c024bf348aa718b7bc47237.jpg)

![54f26fd882770e707177609bb8737ffa76ab4ed5c879f8b1d37f861dd65b9be3.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/images/54f26fd882770e707177609bb8737ffa76ab4ed5c879f8b1d37f861dd65b9be3.jpg)

![634fc7fcb8805a691d513c984485c8190535d23ae4deb84d87aaeb22f7f72a65.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/images/634fc7fcb8805a691d513c984485c8190535d23ae4deb84d87aaeb22f7f72a65.jpg)

![674209c64ce36a233375a0708863a62d6b7347bca95943ec06624afd69935344.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/images/674209c64ce36a233375a0708863a62d6b7347bca95943ec06624afd69935344.jpg)

![b60b0d66542d506814a35b0972f8f207ef8e28f61eb097d396240b241dabbac2.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/images/b60b0d66542d506814a35b0972f8f207ef8e28f61eb097d396240b241dabbac2.jpg)

![db7f6dde5952d79f48028d43f21e92c89ac3fa364ede09c1dca8530fb9f6379c.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/images/db7f6dde5952d79f48028d43f21e92c89ac3fa364ede09c1dca8530fb9f6379c.jpg)

### Tables

![3702b9ba896418e1366ec9757760f0988d6ed1fdcfb25c5f4fe72357b1b358da.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/tables/3702b9ba896418e1366ec9757760f0988d6ed1fdcfb25c5f4fe72357b1b358da.jpg)

![40c45cb10e689a7e8ea5f3cbc5957008299e7ccf08065c79f52f77da72002455.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/tables/40c45cb10e689a7e8ea5f3cbc5957008299e7ccf08065c79f52f77da72002455.jpg)

![42e6c333d0e3b225e22545f5e5766036f6ccb7240f99718131ae022fc151d0ba.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/tables/42e6c333d0e3b225e22545f5e5766036f6ccb7240f99718131ae022fc151d0ba.jpg)

![69cac316a30802d3efdab774e2044d9d7b1de1a8fedaa9d7634c64dfbbf17bc3.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/tables/69cac316a30802d3efdab774e2044d9d7b1de1a8fedaa9d7634c64dfbbf17bc3.jpg)

![84cb6ccf86c71ac38892664eba26da681dbe6bcee7fff81e089766c52d0cfdc3.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/tables/84cb6ccf86c71ac38892664eba26da681dbe6bcee7fff81e089766c52d0cfdc3.jpg)

![8b099910d3860f8174dccfad70a652f99ee166efe42011a9798612253cef8b0b.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/tables/8b099910d3860f8174dccfad70a652f99ee166efe42011a9798612253cef8b0b.jpg)

![9019ba6acb9a164c0001c9cc83586e763f6d9af3ba7a3e47d1e0a04298c235a0.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/tables/9019ba6acb9a164c0001c9cc83586e763f6d9af3ba7a3e47d1e0a04298c235a0.jpg)

![9260999fdc9aeebca0085c797f0aa961b12ed26c652b884787e5a4886ef7c9ce.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/tables/9260999fdc9aeebca0085c797f0aa961b12ed26c652b884787e5a4886ef7c9ce.jpg)

![96fae9ad46cf6aa35f9720f28814d9c25a789e7469e2f34b2b8ad46d4db9857f.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/tables/96fae9ad46cf6aa35f9720f28814d9c25a789e7469e2f34b2b8ad46d4db9857f.jpg)

![a265fb4291742b4fbab5101635ead88623e35ff43e46f7ed90a84fcbb8d8b78b.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/tables/a265fb4291742b4fbab5101635ead88623e35ff43e46f7ed90a84fcbb8d8b78b.jpg)

![b27c8e21dcd08eac1dd8068fa9aa1ea3513365108aca90c4d3a1e05be0eb670c.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/tables/b27c8e21dcd08eac1dd8068fa9aa1ea3513365108aca90c4d3a1e05be0eb670c.jpg)

![b2e66fd7545e3a1c8005fe204d832e479ef9311c7d5e5096b925501f3762416e.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/tables/b2e66fd7545e3a1c8005fe204d832e479ef9311c7d5e5096b925501f3762416e.jpg)

![d447fe05642d31028d106ca9862b68f71bc2caf8dd720c5f3cdeaab02dba8062.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/tables/d447fe05642d31028d106ca9862b68f71bc2caf8dd720c5f3cdeaab02dba8062.jpg)

![d8542c7e6e98953d44ef0ae38831d4025c4c66b1db595b0861fbbc4678cac520.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/tables/d8542c7e6e98953d44ef0ae38831d4025c4c66b1db595b0861fbbc4678cac520.jpg)

![dce927f484e69035ffbe4164c9dec9bbadd90d842104b9a1d047012687ec272a.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/tables/dce927f484e69035ffbe4164c9dec9bbadd90d842104b9a1d047012687ec272a.jpg)

![eb093f0e94937299d9e4dcf0c1538e55409e2f1c7ae653dde493e124e5e80a54.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/tables/eb093f0e94937299d9e4dcf0c1538e55409e2f1c7ae653dde493e124e5e80a54.jpg)

![f412ef54372360aadec126835443c44773bd430a29d64a830ac2dc4d7f771168.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/tables/f412ef54372360aadec126835443c44773bd430a29d64a830ac2dc4d7f771168.jpg)

![fd1b5fae8ce26464142182592608a7c5631447af97cad45c3f93ae17bd6b4f67.jpg](../iclr_results/3323_Examining Alignment of Large Language Models through Representative Heuristics_ the case of politica/tables/fd1b5fae8ce26464142182592608a7c5631447af97cad45c3f93ae17bd6b4f67.jpg)

## DON’T STOP ME NOW: EMBEDDING BASED SCHEDULING FOR LLMS


### Images

![18f4ddda2531f6c1bcf62e805114a2365865f74b91ef8429bb959038b4b651a3.jpg](../iclr_results/3324_DON’T STOP ME NOW_ EMBEDDING BASED SCHEDULING FOR LLMS/images/18f4ddda2531f6c1bcf62e805114a2365865f74b91ef8429bb959038b4b651a3.jpg)

![2131cbe2272ebb44a9e5945b7939c7919082a21080a6e9d6f531275ee95efcac.jpg](../iclr_results/3324_DON’T STOP ME NOW_ EMBEDDING BASED SCHEDULING FOR LLMS/images/2131cbe2272ebb44a9e5945b7939c7919082a21080a6e9d6f531275ee95efcac.jpg)

![27242845d425d8bbd0d1baa2bbe248070d6e64352b87e196d19b377772623f81.jpg](../iclr_results/3324_DON’T STOP ME NOW_ EMBEDDING BASED SCHEDULING FOR LLMS/images/27242845d425d8bbd0d1baa2bbe248070d6e64352b87e196d19b377772623f81.jpg)

![36d22d5175f4832c486056a3ce300bfd851abfec9475e3e1742d3c66b13e5308.jpg](../iclr_results/3324_DON’T STOP ME NOW_ EMBEDDING BASED SCHEDULING FOR LLMS/images/36d22d5175f4832c486056a3ce300bfd851abfec9475e3e1742d3c66b13e5308.jpg)

![47db8d2f0bba92e115623a126c17de9f128a527c82c14e9a22e6eb302d0f3e58.jpg](../iclr_results/3324_DON’T STOP ME NOW_ EMBEDDING BASED SCHEDULING FOR LLMS/images/47db8d2f0bba92e115623a126c17de9f128a527c82c14e9a22e6eb302d0f3e58.jpg)

![4b26afb9f0f2424b149c5eda39c51612278afac7779181b3d646a5d29aaba4a3.jpg](../iclr_results/3324_DON’T STOP ME NOW_ EMBEDDING BASED SCHEDULING FOR LLMS/images/4b26afb9f0f2424b149c5eda39c51612278afac7779181b3d646a5d29aaba4a3.jpg)

![783633fa34bf9cc0462ba51c7e7124b7af17564faa8652c60be4563b9b6ee5f1.jpg](../iclr_results/3324_DON’T STOP ME NOW_ EMBEDDING BASED SCHEDULING FOR LLMS/images/783633fa34bf9cc0462ba51c7e7124b7af17564faa8652c60be4563b9b6ee5f1.jpg)

![79f532038ba08857b44b8e5a4ab5f96cda1834a36da7aaa0cdcb37a99f34f2bd.jpg](../iclr_results/3324_DON’T STOP ME NOW_ EMBEDDING BASED SCHEDULING FOR LLMS/images/79f532038ba08857b44b8e5a4ab5f96cda1834a36da7aaa0cdcb37a99f34f2bd.jpg)

![806d0c794074102618803be28e9e0016c1a4a696b24399f5564e02ee6e81bddf.jpg](../iclr_results/3324_DON’T STOP ME NOW_ EMBEDDING BASED SCHEDULING FOR LLMS/images/806d0c794074102618803be28e9e0016c1a4a696b24399f5564e02ee6e81bddf.jpg)

![9e4e674c9af3def33ad3cccc33c43a801399fb1695b95aff6ff7dcb24d717bde.jpg](../iclr_results/3324_DON’T STOP ME NOW_ EMBEDDING BASED SCHEDULING FOR LLMS/images/9e4e674c9af3def33ad3cccc33c43a801399fb1695b95aff6ff7dcb24d717bde.jpg)

![a05e7d9fec730d4a8f1e2ede30ecd86af04471af2f2c48a0aa8a2d3cd44f3690.jpg](../iclr_results/3324_DON’T STOP ME NOW_ EMBEDDING BASED SCHEDULING FOR LLMS/images/a05e7d9fec730d4a8f1e2ede30ecd86af04471af2f2c48a0aa8a2d3cd44f3690.jpg)

![ad556b0061a572a590421a220627ea84777bf839e72d7bd7b501de2f4084e1a5.jpg](../iclr_results/3324_DON’T STOP ME NOW_ EMBEDDING BASED SCHEDULING FOR LLMS/images/ad556b0061a572a590421a220627ea84777bf839e72d7bd7b501de2f4084e1a5.jpg)

![b4fd58e873153504e3aa5a359eba2ae98943f61fe41c7916d57757ba304b9acb.jpg](../iclr_results/3324_DON’T STOP ME NOW_ EMBEDDING BASED SCHEDULING FOR LLMS/images/b4fd58e873153504e3aa5a359eba2ae98943f61fe41c7916d57757ba304b9acb.jpg)

![ef75ffb4de34690189da53f0a133fdad7795915496940ebd0fe319efced46ea5.jpg](../iclr_results/3324_DON’T STOP ME NOW_ EMBEDDING BASED SCHEDULING FOR LLMS/images/ef75ffb4de34690189da53f0a133fdad7795915496940ebd0fe319efced46ea5.jpg)

### Tables

![81892b2c706c137bc0accd47ce256418966b2b46fa448e6c45be36f96ef79dcb.jpg](../iclr_results/3324_DON’T STOP ME NOW_ EMBEDDING BASED SCHEDULING FOR LLMS/tables/81892b2c706c137bc0accd47ce256418966b2b46fa448e6c45be36f96ef79dcb.jpg)

## 3DGS-Drag: Dragging Gaussians for Intuitive Point-Based 3D Editing


### Images

![3fe1d5f896dd188de8dff9b23e759497958d1e39ad7039a0d5dba0463c605d93.jpg](../iclr_results/3325_3DGS-Drag_ Dragging Gaussians for Intuitive Point-Based 3D Editing/images/3fe1d5f896dd188de8dff9b23e759497958d1e39ad7039a0d5dba0463c605d93.jpg)

![54d4c903b795498a6e64c8ff1789d8aa509fd3381dfa0bcf7577a75ee22965bb.jpg](../iclr_results/3325_3DGS-Drag_ Dragging Gaussians for Intuitive Point-Based 3D Editing/images/54d4c903b795498a6e64c8ff1789d8aa509fd3381dfa0bcf7577a75ee22965bb.jpg)

![6cd540ed2aae26873e849eee9be39eea46376a507d977d95e81230bc6140a281.jpg](../iclr_results/3325_3DGS-Drag_ Dragging Gaussians for Intuitive Point-Based 3D Editing/images/6cd540ed2aae26873e849eee9be39eea46376a507d977d95e81230bc6140a281.jpg)

![6ed5344bf14eaeb57e59c66987e119f7f9ae61002686c936d64075ca4143c534.jpg](../iclr_results/3325_3DGS-Drag_ Dragging Gaussians for Intuitive Point-Based 3D Editing/images/6ed5344bf14eaeb57e59c66987e119f7f9ae61002686c936d64075ca4143c534.jpg)

![7c79a7e58d1c4a0285436368fdca5b4ddb30c5e472a07caec5b833fdfc78d140.jpg](../iclr_results/3325_3DGS-Drag_ Dragging Gaussians for Intuitive Point-Based 3D Editing/images/7c79a7e58d1c4a0285436368fdca5b4ddb30c5e472a07caec5b833fdfc78d140.jpg)

![9f02e8e815e4156da3af9b60c1a4f2f91c2484c08359050a7f6195746d36aaec.jpg](../iclr_results/3325_3DGS-Drag_ Dragging Gaussians for Intuitive Point-Based 3D Editing/images/9f02e8e815e4156da3af9b60c1a4f2f91c2484c08359050a7f6195746d36aaec.jpg)

![a51e91b9c59192219e4adbfc0b7dde4dae5d7918883281ffc2d7600ce61e824a.jpg](../iclr_results/3325_3DGS-Drag_ Dragging Gaussians for Intuitive Point-Based 3D Editing/images/a51e91b9c59192219e4adbfc0b7dde4dae5d7918883281ffc2d7600ce61e824a.jpg)

![acfe0ec68a0d0e5712186bc891e1b5123edeffc92eac4df6b73838c898bafc0c.jpg](../iclr_results/3325_3DGS-Drag_ Dragging Gaussians for Intuitive Point-Based 3D Editing/images/acfe0ec68a0d0e5712186bc891e1b5123edeffc92eac4df6b73838c898bafc0c.jpg)

![af573e4e0c2b3d0f755cf4dc22b844c333c1f9f2fc33325590899510b6e1a6ee.jpg](../iclr_results/3325_3DGS-Drag_ Dragging Gaussians for Intuitive Point-Based 3D Editing/images/af573e4e0c2b3d0f755cf4dc22b844c333c1f9f2fc33325590899510b6e1a6ee.jpg)

![b68ca86d4a1e38c85671909232aafad9c04edf3c7da80d348b74e8ed3662b8c6.jpg](../iclr_results/3325_3DGS-Drag_ Dragging Gaussians for Intuitive Point-Based 3D Editing/images/b68ca86d4a1e38c85671909232aafad9c04edf3c7da80d348b74e8ed3662b8c6.jpg)

![c273027439677afd2c9616c481977058908b099c65c961d406ed5203af73956b.jpg](../iclr_results/3325_3DGS-Drag_ Dragging Gaussians for Intuitive Point-Based 3D Editing/images/c273027439677afd2c9616c481977058908b099c65c961d406ed5203af73956b.jpg)

![de3137361773a5f880c68d1d763572a39c7ec66450b3c7025f688f00a0348cc8.jpg](../iclr_results/3325_3DGS-Drag_ Dragging Gaussians for Intuitive Point-Based 3D Editing/images/de3137361773a5f880c68d1d763572a39c7ec66450b3c7025f688f00a0348cc8.jpg)

![de32af6bf81ace015d289fdf77442e28e9e8f6d35f250b50af0e117c857a169e.jpg](../iclr_results/3325_3DGS-Drag_ Dragging Gaussians for Intuitive Point-Based 3D Editing/images/de32af6bf81ace015d289fdf77442e28e9e8f6d35f250b50af0e117c857a169e.jpg)

![e06df1d9f785608a000edba0b1bf1df6dd0043ef1e65df36baf575ab32d95fa0.jpg](../iclr_results/3325_3DGS-Drag_ Dragging Gaussians for Intuitive Point-Based 3D Editing/images/e06df1d9f785608a000edba0b1bf1df6dd0043ef1e65df36baf575ab32d95fa0.jpg)

### Tables

![d109d41c7d1d91f9781cf89b35557f1868189376d6c12a0ff09c11a4909527e2.jpg](../iclr_results/3325_3DGS-Drag_ Dragging Gaussians for Intuitive Point-Based 3D Editing/tables/d109d41c7d1d91f9781cf89b35557f1868189376d6c12a0ff09c11a4909527e2.jpg)

## Visually Consistent Hierarchical Image Classification


### Images

![0314163a2e4cbfd2f0f34531b652ecb3f25f4bf72ebd849d64c09f8ec72f6e69.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/images/0314163a2e4cbfd2f0f34531b652ecb3f25f4bf72ebd849d64c09f8ec72f6e69.jpg)

![0ea2805f12fae278bde6744f068ff0da90d2a034fe7f8e6c98cd9f17e05b55d6.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/images/0ea2805f12fae278bde6744f068ff0da90d2a034fe7f8e6c98cd9f17e05b55d6.jpg)

![2432e93e5837b4f14d22ad315220a77b5ed3bc82c9ed57d5fa2477f3805f917c.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/images/2432e93e5837b4f14d22ad315220a77b5ed3bc82c9ed57d5fa2477f3805f917c.jpg)

![2c1201cb27b152751401f8c1faf3242bbf2aa8c64de025b0af0c49f3907b4e65.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/images/2c1201cb27b152751401f8c1faf3242bbf2aa8c64de025b0af0c49f3907b4e65.jpg)

![34444fdea036fc2b483875e8df006aaf1b527b23a4cf629041b8493e2452173f.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/images/34444fdea036fc2b483875e8df006aaf1b527b23a4cf629041b8493e2452173f.jpg)

![709687ef7669a5502f7a9696228384d47707841e7d3d5fac0cc3b5493ee479c1.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/images/709687ef7669a5502f7a9696228384d47707841e7d3d5fac0cc3b5493ee479c1.jpg)

![9bbd4f4a767d419c4e48a2e14f0fac7f105ec38362f690a55e032793b20f4a10.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/images/9bbd4f4a767d419c4e48a2e14f0fac7f105ec38362f690a55e032793b20f4a10.jpg)

![a0eecc7e74f04811d6b3a75e4170f35ac89c27f73b10fc28d8b043412462fb43.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/images/a0eecc7e74f04811d6b3a75e4170f35ac89c27f73b10fc28d8b043412462fb43.jpg)

![de3020bc0eedc81d1c382582cbcdd565c2be72d8ec574a867dd0324223e8c03c.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/images/de3020bc0eedc81d1c382582cbcdd565c2be72d8ec574a867dd0324223e8c03c.jpg)

![f9db2269b0e261f382d624b491e4434505d45b55adba0dc3da81439f6f248779.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/images/f9db2269b0e261f382d624b491e4434505d45b55adba0dc3da81439f6f248779.jpg)

![fed423a8fcc02341eb4d4f65f21bd0570fa5d322399c3562c8f280fc3c853e85.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/images/fed423a8fcc02341eb4d4f65f21bd0570fa5d322399c3562c8f280fc3c853e85.jpg)

### Tables

![0ee32c114653b59aac76120e7bb02c57b523591f7997835133f7379877f43310.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/tables/0ee32c114653b59aac76120e7bb02c57b523591f7997835133f7379877f43310.jpg)

![2156689e3f610b3cbe607d7e8fabd1e0fd64166e3af3f58c3ad0cd21e11f8da9.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/tables/2156689e3f610b3cbe607d7e8fabd1e0fd64166e3af3f58c3ad0cd21e11f8da9.jpg)

![4128c522738d2cc24c77205c4f311f3ebdc5bd4f6c1998816d056549a8d4fe2a.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/tables/4128c522738d2cc24c77205c4f311f3ebdc5bd4f6c1998816d056549a8d4fe2a.jpg)

![471c6ac8081895ae520aea4ad66a14157553a9363ebc1ec5d1d3192d7eec415a.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/tables/471c6ac8081895ae520aea4ad66a14157553a9363ebc1ec5d1d3192d7eec415a.jpg)

![47e18bfaf6a976d9ff9b56e8b9f824c14e282ed03f5ef87a463bf0a199e071e1.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/tables/47e18bfaf6a976d9ff9b56e8b9f824c14e282ed03f5ef87a463bf0a199e071e1.jpg)

![8265383ab12f8aa42cdd6037448f50ec7d93e7ead5e130a1e5165ddbcac8e592.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/tables/8265383ab12f8aa42cdd6037448f50ec7d93e7ead5e130a1e5165ddbcac8e592.jpg)

![8c5c577173a032a93fa36b0dbb79bb866c6c836fd4ef1a45de9d32c6fab688b9.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/tables/8c5c577173a032a93fa36b0dbb79bb866c6c836fd4ef1a45de9d32c6fab688b9.jpg)

![af7cf0320d3befb5882a3b0ce953187ec6979e1ebee768bccfad66fca4466a86.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/tables/af7cf0320d3befb5882a3b0ce953187ec6979e1ebee768bccfad66fca4466a86.jpg)

![b5e590cb2cd4fefe3c74222fb847c7516a9ed3bc3f8a16fdeee8aeb1734acbd3.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/tables/b5e590cb2cd4fefe3c74222fb847c7516a9ed3bc3f8a16fdeee8aeb1734acbd3.jpg)

![bb69fe0a277aa70e569c91b2e3ea18a0a1f385c9ce3a31acf737f8c585889896.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/tables/bb69fe0a277aa70e569c91b2e3ea18a0a1f385c9ce3a31acf737f8c585889896.jpg)

![bc6f08932ed5837e6d078c1d3de0872d09944d57257914a9a96513aeab8580c7.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/tables/bc6f08932ed5837e6d078c1d3de0872d09944d57257914a9a96513aeab8580c7.jpg)

![bf5060e9aaf9b5a9e03bc68de764562dfe02541e1d4b7748913ccd368130bd05.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/tables/bf5060e9aaf9b5a9e03bc68de764562dfe02541e1d4b7748913ccd368130bd05.jpg)

![ce1e528d02c859cdcf57853cae2975e6390a009fae2ef55a3a93e6f53d573053.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/tables/ce1e528d02c859cdcf57853cae2975e6390a009fae2ef55a3a93e6f53d573053.jpg)

![d6293177379693c4d33f6ec4eba4cbe811daff8d7d6744411649761d9dd86a2c.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/tables/d6293177379693c4d33f6ec4eba4cbe811daff8d7d6744411649761d9dd86a2c.jpg)

![fa7663bd128d14cfff794d7c65005abc84030aee22a1acc07540923f88bf7874.jpg](../iclr_results/3326_Visually Consistent Hierarchical Image Classification/tables/fa7663bd128d14cfff794d7c65005abc84030aee22a1acc07540923f88bf7874.jpg)

## Prompting Fairness: Integrating Causality to Debias Large Language Models


### Images

![3ae564ae7c8d2145b6ab87daac3ef85920a09a9760685d0d1402280c784c8bcf.jpg](../iclr_results/3327_Prompting Fairness_ Integrating Causality to Debias Large Language Models/images/3ae564ae7c8d2145b6ab87daac3ef85920a09a9760685d0d1402280c784c8bcf.jpg)

![50e981f2ad13c21d957c875e7c2add0913ef442db47d16e3bd8d3ca18a52fa93.jpg](../iclr_results/3327_Prompting Fairness_ Integrating Causality to Debias Large Language Models/images/50e981f2ad13c21d957c875e7c2add0913ef442db47d16e3bd8d3ca18a52fa93.jpg)

![5b36d9f6a787655f425528030bc1f9cb60a66ed9d3d170c8ef7837a30ddab59f.jpg](../iclr_results/3327_Prompting Fairness_ Integrating Causality to Debias Large Language Models/images/5b36d9f6a787655f425528030bc1f9cb60a66ed9d3d170c8ef7837a30ddab59f.jpg)

![6fc74d26a4278723fcc6bebd7563201631d0164467a02637192290844c299a24.jpg](../iclr_results/3327_Prompting Fairness_ Integrating Causality to Debias Large Language Models/images/6fc74d26a4278723fcc6bebd7563201631d0164467a02637192290844c299a24.jpg)

![79a512270dba46658972adda0743c1d5b7b0a1226588fc60450a8ecce0f7f6c2.jpg](../iclr_results/3327_Prompting Fairness_ Integrating Causality to Debias Large Language Models/images/79a512270dba46658972adda0743c1d5b7b0a1226588fc60450a8ecce0f7f6c2.jpg)

![8ac31e2b80266fe2d93f936ea803bb3975c69a583183c369a40d025f3566ac60.jpg](../iclr_results/3327_Prompting Fairness_ Integrating Causality to Debias Large Language Models/images/8ac31e2b80266fe2d93f936ea803bb3975c69a583183c369a40d025f3566ac60.jpg)

![8ce3061e5774bc1f205a29a037f9e8b9b5807f2f3e013be17ddb906209e72569.jpg](../iclr_results/3327_Prompting Fairness_ Integrating Causality to Debias Large Language Models/images/8ce3061e5774bc1f205a29a037f9e8b9b5807f2f3e013be17ddb906209e72569.jpg)

![a461a8a469b242afbd954aa77d06491faba6162fbe046504fc11218aa3e81a44.jpg](../iclr_results/3327_Prompting Fairness_ Integrating Causality to Debias Large Language Models/images/a461a8a469b242afbd954aa77d06491faba6162fbe046504fc11218aa3e81a44.jpg)

![b650ee68a51dc95d92be2059cd52d7e850978f9d61fbb73c299177a5c8f1c06b.jpg](../iclr_results/3327_Prompting Fairness_ Integrating Causality to Debias Large Language Models/images/b650ee68a51dc95d92be2059cd52d7e850978f9d61fbb73c299177a5c8f1c06b.jpg)

![f825cd9dda431127dec1c1395c827648e9eb013dbd48a2705945a3c8d90caae6.jpg](../iclr_results/3327_Prompting Fairness_ Integrating Causality to Debias Large Language Models/images/f825cd9dda431127dec1c1395c827648e9eb013dbd48a2705945a3c8d90caae6.jpg)

### Tables

![28fc711d7bc49bc5711cc1c0d7e8fc9f77d961f015fd0d410253f7be6961f721.jpg](../iclr_results/3327_Prompting Fairness_ Integrating Causality to Debias Large Language Models/tables/28fc711d7bc49bc5711cc1c0d7e8fc9f77d961f015fd0d410253f7be6961f721.jpg)

![29afe8cbb07cc075659cded0682cbaa474a609058f81f04dd80154c41b9c4727.jpg](../iclr_results/3327_Prompting Fairness_ Integrating Causality to Debias Large Language Models/tables/29afe8cbb07cc075659cded0682cbaa474a609058f81f04dd80154c41b9c4727.jpg)

![6dbdb83b1b7e1a78b29c41508e723943ed89bc74e328ea92b4629a399925d4b6.jpg](../iclr_results/3327_Prompting Fairness_ Integrating Causality to Debias Large Language Models/tables/6dbdb83b1b7e1a78b29c41508e723943ed89bc74e328ea92b4629a399925d4b6.jpg)

![718cb4682cd95c3738a8524fa3d80985c2b6e1a88fa58d8129e438aebb729b7d.jpg](../iclr_results/3327_Prompting Fairness_ Integrating Causality to Debias Large Language Models/tables/718cb4682cd95c3738a8524fa3d80985c2b6e1a88fa58d8129e438aebb729b7d.jpg)

![755ebf92b16ab160d85e99a29b95f5ae167d4677b1c26547e5d6860559a9081a.jpg](../iclr_results/3327_Prompting Fairness_ Integrating Causality to Debias Large Language Models/tables/755ebf92b16ab160d85e99a29b95f5ae167d4677b1c26547e5d6860559a9081a.jpg)

![994bebafa6e25fe13eebf591ca74be518a563c48e339a36e0110f1dc250c8f5d.jpg](../iclr_results/3327_Prompting Fairness_ Integrating Causality to Debias Large Language Models/tables/994bebafa6e25fe13eebf591ca74be518a563c48e339a36e0110f1dc250c8f5d.jpg)

![e019f0b61461fa0b257235d8be530a46c78cb0b98664989d5efaccdd755f1656.jpg](../iclr_results/3327_Prompting Fairness_ Integrating Causality to Debias Large Language Models/tables/e019f0b61461fa0b257235d8be530a46c78cb0b98664989d5efaccdd755f1656.jpg)

## WardropNet: Traffic Flow Predictions via Equilibrium-Augmented Learning


### Images

![05c66c7373c1aa62457be43e41e7b048f4298f1610835e07e2fdebba7eb29493.jpg](../iclr_results/3328_WardropNet_ Traffic Flow Predictions via Equilibrium-Augmented Learning/images/05c66c7373c1aa62457be43e41e7b048f4298f1610835e07e2fdebba7eb29493.jpg)

![09b1e4406ff55a9a943f3287234f9004d20b5b507efb9fcebe40df1e8a7b6ca9.jpg](../iclr_results/3328_WardropNet_ Traffic Flow Predictions via Equilibrium-Augmented Learning/images/09b1e4406ff55a9a943f3287234f9004d20b5b507efb9fcebe40df1e8a7b6ca9.jpg)

![23911f983af853fc96f112343b6fd2156152a0198f68958c14ca89d8c208639d.jpg](../iclr_results/3328_WardropNet_ Traffic Flow Predictions via Equilibrium-Augmented Learning/images/23911f983af853fc96f112343b6fd2156152a0198f68958c14ca89d8c208639d.jpg)

![3490fc72b139abb0ef76b1c25659a6a3bd2184a271fd4d9c42e90e5063de616c.jpg](../iclr_results/3328_WardropNet_ Traffic Flow Predictions via Equilibrium-Augmented Learning/images/3490fc72b139abb0ef76b1c25659a6a3bd2184a271fd4d9c42e90e5063de616c.jpg)

![3b494c32e43bf7ad6087ebcc66a7c5e7eea84f15972732d2cc0db6aa62ae9db6.jpg](../iclr_results/3328_WardropNet_ Traffic Flow Predictions via Equilibrium-Augmented Learning/images/3b494c32e43bf7ad6087ebcc66a7c5e7eea84f15972732d2cc0db6aa62ae9db6.jpg)

![4ab618e0e543ade1b748aee387c6ad56a66abd7412b383244fdfffb75ec507f6.jpg](../iclr_results/3328_WardropNet_ Traffic Flow Predictions via Equilibrium-Augmented Learning/images/4ab618e0e543ade1b748aee387c6ad56a66abd7412b383244fdfffb75ec507f6.jpg)

![4dd40b121873784652c2af78215e28cb53cfa11c26f020f985643240dee426a0.jpg](../iclr_results/3328_WardropNet_ Traffic Flow Predictions via Equilibrium-Augmented Learning/images/4dd40b121873784652c2af78215e28cb53cfa11c26f020f985643240dee426a0.jpg)

![5cab546d560bed2df2b0e9716c616dcaf1bca2297fa4f1eecab647f83d4d7a98.jpg](../iclr_results/3328_WardropNet_ Traffic Flow Predictions via Equilibrium-Augmented Learning/images/5cab546d560bed2df2b0e9716c616dcaf1bca2297fa4f1eecab647f83d4d7a98.jpg)

![66275ae9d1e2f3b7d1cf2a0fe0c8a9fe0633f4c8535ffe35fbeb1221c2f1112d.jpg](../iclr_results/3328_WardropNet_ Traffic Flow Predictions via Equilibrium-Augmented Learning/images/66275ae9d1e2f3b7d1cf2a0fe0c8a9fe0633f4c8535ffe35fbeb1221c2f1112d.jpg)

![91f2e7ec70c6b3c78a8c055f6dc5383e61aba80d307789e1ef84729b4346cfda.jpg](../iclr_results/3328_WardropNet_ Traffic Flow Predictions via Equilibrium-Augmented Learning/images/91f2e7ec70c6b3c78a8c055f6dc5383e61aba80d307789e1ef84729b4346cfda.jpg)

![94d5bed989b49a5dcf60cb5807c5e24ff26a08b879d146bb6132511b5a9223c7.jpg](../iclr_results/3328_WardropNet_ Traffic Flow Predictions via Equilibrium-Augmented Learning/images/94d5bed989b49a5dcf60cb5807c5e24ff26a08b879d146bb6132511b5a9223c7.jpg)

![a0984b1a5fe5879823907d2ee64324ce7fb88a314d19b765aaabc62fc053908d.jpg](../iclr_results/3328_WardropNet_ Traffic Flow Predictions via Equilibrium-Augmented Learning/images/a0984b1a5fe5879823907d2ee64324ce7fb88a314d19b765aaabc62fc053908d.jpg)

![c814dabadb413fda1c229628ae7e1bf07e8890f4b03422633ad051eefa42dfba.jpg](../iclr_results/3328_WardropNet_ Traffic Flow Predictions via Equilibrium-Augmented Learning/images/c814dabadb413fda1c229628ae7e1bf07e8890f4b03422633ad051eefa42dfba.jpg)

![cb0625dfd526a63564abf62edeccc298b8566de7077f64c2dc379e026104ae4f.jpg](../iclr_results/3328_WardropNet_ Traffic Flow Predictions via Equilibrium-Augmented Learning/images/cb0625dfd526a63564abf62edeccc298b8566de7077f64c2dc379e026104ae4f.jpg)

![dc0f5579f836e401c0244e5ac601fb983c5bddf6cee21f442fb862611d4e0183.jpg](../iclr_results/3328_WardropNet_ Traffic Flow Predictions via Equilibrium-Augmented Learning/images/dc0f5579f836e401c0244e5ac601fb983c5bddf6cee21f442fb862611d4e0183.jpg)

### Tables

![021c2746fd45b33011d12a47998964f8c3cd6c176d1d152800e30fda758b7e52.jpg](../iclr_results/3328_WardropNet_ Traffic Flow Predictions via Equilibrium-Augmented Learning/tables/021c2746fd45b33011d12a47998964f8c3cd6c176d1d152800e30fda758b7e52.jpg)

![24e6cd2250f4143c4fe80847f7dbe828f4ca494b08191b3c522f7c4adba39b22.jpg](../iclr_results/3328_WardropNet_ Traffic Flow Predictions via Equilibrium-Augmented Learning/tables/24e6cd2250f4143c4fe80847f7dbe828f4ca494b08191b3c522f7c4adba39b22.jpg)

![29503421aad00f1e2f3dd7a1d96d939f647f42f257a2943420f594fb9411ec82.jpg](../iclr_results/3328_WardropNet_ Traffic Flow Predictions via Equilibrium-Augmented Learning/tables/29503421aad00f1e2f3dd7a1d96d939f647f42f257a2943420f594fb9411ec82.jpg)

![92023ab3a3ac60a04cc2df8dc392f7c15bb4fd319d5776ce96ae02e2cbf684b6.jpg](../iclr_results/3328_WardropNet_ Traffic Flow Predictions via Equilibrium-Augmented Learning/tables/92023ab3a3ac60a04cc2df8dc392f7c15bb4fd319d5776ce96ae02e2cbf684b6.jpg)

![cb6b444908492d69af89c2104ff00d7eec60ac85b579b9b22619b914d5cdc1fa.jpg](../iclr_results/3328_WardropNet_ Traffic Flow Predictions via Equilibrium-Augmented Learning/tables/cb6b444908492d69af89c2104ff00d7eec60ac85b579b9b22619b914d5cdc1fa.jpg)

![f82985d86711329a10c3bb4ae86bdb4a107edb5b99aa0f6150afe5e93731510c.jpg](../iclr_results/3328_WardropNet_ Traffic Flow Predictions via Equilibrium-Augmented Learning/tables/f82985d86711329a10c3bb4ae86bdb4a107edb5b99aa0f6150afe5e93731510c.jpg)

## Lawma: The Power of Specialization for Legal Annotation


### Images

![03daef1dff9319ca60e9de89ad4d1c468b7f27f783e31313694a89a28f2a8d93.jpg](../iclr_results/3329_Lawma_ The Power of Specialization for Legal Annotation/images/03daef1dff9319ca60e9de89ad4d1c468b7f27f783e31313694a89a28f2a8d93.jpg)

![157b42b9fff8a4e58b4b231e4faaf9d5756828e9fce175ad9e3adae14be685a2.jpg](../iclr_results/3329_Lawma_ The Power of Specialization for Legal Annotation/images/157b42b9fff8a4e58b4b231e4faaf9d5756828e9fce175ad9e3adae14be685a2.jpg)

![261c0cf9fd00f32e6b8ed15c01b047a2d5b6e349da3256831b33b2266568fd27.jpg](../iclr_results/3329_Lawma_ The Power of Specialization for Legal Annotation/images/261c0cf9fd00f32e6b8ed15c01b047a2d5b6e349da3256831b33b2266568fd27.jpg)

![2833fd9a7bddd6d136adebcfe0169c8a38294e76e0ecc3acc431ba33fab93d63.jpg](../iclr_results/3329_Lawma_ The Power of Specialization for Legal Annotation/images/2833fd9a7bddd6d136adebcfe0169c8a38294e76e0ecc3acc431ba33fab93d63.jpg)

![2b8e8a8e315143e3fcfeb37472506a8a3e881237cd100b6433972ca156eaad04.jpg](../iclr_results/3329_Lawma_ The Power of Specialization for Legal Annotation/images/2b8e8a8e315143e3fcfeb37472506a8a3e881237cd100b6433972ca156eaad04.jpg)

![2f9bf43e8197f150a576a63130c06cd7bc12bf2aa5b4258cf110e80bfd9ede3c.jpg](../iclr_results/3329_Lawma_ The Power of Specialization for Legal Annotation/images/2f9bf43e8197f150a576a63130c06cd7bc12bf2aa5b4258cf110e80bfd9ede3c.jpg)

![2fe6d0752ae9b7d4ee17d2961855dbce9629efaa7f0101c560398f9c2c093468.jpg](../iclr_results/3329_Lawma_ The Power of Specialization for Legal Annotation/images/2fe6d0752ae9b7d4ee17d2961855dbce9629efaa7f0101c560398f9c2c093468.jpg)

![4586733bf83e01ccefb8662cb1b4fc7d95f1dcf105d1aeb261c0aff65097690c.jpg](../iclr_results/3329_Lawma_ The Power of Specialization for Legal Annotation/images/4586733bf83e01ccefb8662cb1b4fc7d95f1dcf105d1aeb261c0aff65097690c.jpg)

![49d7c8564b62e768a1e500c4c08ad1443b06ea8b3af96e871d06fb358a09a3d9.jpg](../iclr_results/3329_Lawma_ The Power of Specialization for Legal Annotation/images/49d7c8564b62e768a1e500c4c08ad1443b06ea8b3af96e871d06fb358a09a3d9.jpg)

![5b4ee66fd787a806dfc58ff56785e7fcab6ab6dc998b2df8ea78f8ffcbcd585d.jpg](../iclr_results/3329_Lawma_ The Power of Specialization for Legal Annotation/images/5b4ee66fd787a806dfc58ff56785e7fcab6ab6dc998b2df8ea78f8ffcbcd585d.jpg)

![73a549f8beb59d17d1302eccf0688ecef5cdea5bf1cfed2c4f3e5122b8312749.jpg](../iclr_results/3329_Lawma_ The Power of Specialization for Legal Annotation/images/73a549f8beb59d17d1302eccf0688ecef5cdea5bf1cfed2c4f3e5122b8312749.jpg)

![74359ded7f0cd4ae7932160024dd9dd30c0f53bcfe8f475f5048eb28bd5566cf.jpg](../iclr_results/3329_Lawma_ The Power of Specialization for Legal Annotation/images/74359ded7f0cd4ae7932160024dd9dd30c0f53bcfe8f475f5048eb28bd5566cf.jpg)

![7fded596087a37a069eda3386a48f8869b797e0ea8115adbace7aa9b21fd8df6.jpg](../iclr_results/3329_Lawma_ The Power of Specialization for Legal Annotation/images/7fded596087a37a069eda3386a48f8869b797e0ea8115adbace7aa9b21fd8df6.jpg)

![9c02e30249b69f5343688b06e6f71c9c3000199bf8117d53201633acc5832d8f.jpg](../iclr_results/3329_Lawma_ The Power of Specialization for Legal Annotation/images/9c02e30249b69f5343688b06e6f71c9c3000199bf8117d53201633acc5832d8f.jpg)

![acc71f3b453110938ce036878edf4e3f68561fb79f0f30fbdec8aefb845fbe47.jpg](../iclr_results/3329_Lawma_ The Power of Specialization for Legal Annotation/images/acc71f3b453110938ce036878edf4e3f68561fb79f0f30fbdec8aefb845fbe47.jpg)

![d1921d1c64ac6df3da3536d9f5a274a2e1813d37cd4301ba66084b763bfa1115.jpg](../iclr_results/3329_Lawma_ The Power of Specialization for Legal Annotation/images/d1921d1c64ac6df3da3536d9f5a274a2e1813d37cd4301ba66084b763bfa1115.jpg)

![e595c92496aaa11fdd164c94b66f1db4466579f6416614b7402f07557533e46f.jpg](../iclr_results/3329_Lawma_ The Power of Specialization for Legal Annotation/images/e595c92496aaa11fdd164c94b66f1db4466579f6416614b7402f07557533e46f.jpg)

![f5eb4bb38a06c270d9b6e75a9b4d377d2b8c4d22323faad517e3716baf386c6c.jpg](../iclr_results/3329_Lawma_ The Power of Specialization for Legal Annotation/images/f5eb4bb38a06c270d9b6e75a9b4d377d2b8c4d22323faad517e3716baf386c6c.jpg)

### Tables

![e18df17fe653fa39f2e9d7e452eb90d99c7f4d29677b50d4e6218a5ebd382fcb.jpg](../iclr_results/3329_Lawma_ The Power of Specialization for Legal Annotation/tables/e18df17fe653fa39f2e9d7e452eb90d99c7f4d29677b50d4e6218a5ebd382fcb.jpg)

![ed254b1dbe73b8ca4ebb7691bb02ce985804465261fe7857f3ab89622002a59a.jpg](../iclr_results/3329_Lawma_ The Power of Specialization for Legal Annotation/tables/ed254b1dbe73b8ca4ebb7691bb02ce985804465261fe7857f3ab89622002a59a.jpg)

## MIA-Bench: Towards Better Instruction Following Evaluation of Multimodal LLMs


### Images

![02b9f6c26752b71680221a35ec28900d4d7d878bf91d65eb42dea07a87f6657b.jpg](../iclr_results/3330_MIA-Bench_ Towards Better Instruction Following Evaluation of Multimodal LLMs/images/02b9f6c26752b71680221a35ec28900d4d7d878bf91d65eb42dea07a87f6657b.jpg)

![192ceb13a4bca49ad1401eeb1ac3f7992b304ec39c12b2bb6b6b5437bbfd8714.jpg](../iclr_results/3330_MIA-Bench_ Towards Better Instruction Following Evaluation of Multimodal LLMs/images/192ceb13a4bca49ad1401eeb1ac3f7992b304ec39c12b2bb6b6b5437bbfd8714.jpg)

![2fd9b205f1473969d385b3048262d68e250c9d10a8fdbfafba0508ef64e51719.jpg](../iclr_results/3330_MIA-Bench_ Towards Better Instruction Following Evaluation of Multimodal LLMs/images/2fd9b205f1473969d385b3048262d68e250c9d10a8fdbfafba0508ef64e51719.jpg)

![31b59bbdd172ed94a9c07cf9576f2d1b090e7549b2968ec8a2e9f02f1ed3d391.jpg](../iclr_results/3330_MIA-Bench_ Towards Better Instruction Following Evaluation of Multimodal LLMs/images/31b59bbdd172ed94a9c07cf9576f2d1b090e7549b2968ec8a2e9f02f1ed3d391.jpg)

![3772408b75ad8138de526bd84e32f60b92464265bfc6ba681cf5dc864574da3b.jpg](../iclr_results/3330_MIA-Bench_ Towards Better Instruction Following Evaluation of Multimodal LLMs/images/3772408b75ad8138de526bd84e32f60b92464265bfc6ba681cf5dc864574da3b.jpg)

![43c6e161aee249e3c367d97ff2df427b8d12c372e294ef267191ef7c58aca240.jpg](../iclr_results/3330_MIA-Bench_ Towards Better Instruction Following Evaluation of Multimodal LLMs/images/43c6e161aee249e3c367d97ff2df427b8d12c372e294ef267191ef7c58aca240.jpg)

![5695274872566c5dea8eb2880b49d2b552d913a848279f12df0434fdde1ab72d.jpg](../iclr_results/3330_MIA-Bench_ Towards Better Instruction Following Evaluation of Multimodal LLMs/images/5695274872566c5dea8eb2880b49d2b552d913a848279f12df0434fdde1ab72d.jpg)

![9d27a7a12535add4f38641d883513a496869c4335a8d891d17cb9ac067e243fe.jpg](../iclr_results/3330_MIA-Bench_ Towards Better Instruction Following Evaluation of Multimodal LLMs/images/9d27a7a12535add4f38641d883513a496869c4335a8d891d17cb9ac067e243fe.jpg)

![a0acd10514483ee43e14384fa710addd6bf912eb8c6816735182dca7520a7504.jpg](../iclr_results/3330_MIA-Bench_ Towards Better Instruction Following Evaluation of Multimodal LLMs/images/a0acd10514483ee43e14384fa710addd6bf912eb8c6816735182dca7520a7504.jpg)

![b8604a912ca66f0b0aef02f284500bde838ad20a3490a9eae5a640fa28982106.jpg](../iclr_results/3330_MIA-Bench_ Towards Better Instruction Following Evaluation of Multimodal LLMs/images/b8604a912ca66f0b0aef02f284500bde838ad20a3490a9eae5a640fa28982106.jpg)

![c83054fcada5997784d2d7294bcd4630ea56351c387377eb620072003eaad02f.jpg](../iclr_results/3330_MIA-Bench_ Towards Better Instruction Following Evaluation of Multimodal LLMs/images/c83054fcada5997784d2d7294bcd4630ea56351c387377eb620072003eaad02f.jpg)

![ff08bc6417acb3857643519f6aec5389239c7f7cefc5908345b93adbde0da70e.jpg](../iclr_results/3330_MIA-Bench_ Towards Better Instruction Following Evaluation of Multimodal LLMs/images/ff08bc6417acb3857643519f6aec5389239c7f7cefc5908345b93adbde0da70e.jpg)

### Tables

![178a037cf0dbc3be44880971c862ec62df6e0712aebe18bd76a92078cb0cfe9a.jpg](../iclr_results/3330_MIA-Bench_ Towards Better Instruction Following Evaluation of Multimodal LLMs/tables/178a037cf0dbc3be44880971c862ec62df6e0712aebe18bd76a92078cb0cfe9a.jpg)

![3bd4d865c8f53d4d579bb7a093ecc52a49a3bd557efccad6e410720b94d27f12.jpg](../iclr_results/3330_MIA-Bench_ Towards Better Instruction Following Evaluation of Multimodal LLMs/tables/3bd4d865c8f53d4d579bb7a093ecc52a49a3bd557efccad6e410720b94d27f12.jpg)

![56637b033669f328b60290d9a49565c783006d43d92f4cb51a4553762c6a9504.jpg](../iclr_results/3330_MIA-Bench_ Towards Better Instruction Following Evaluation of Multimodal LLMs/tables/56637b033669f328b60290d9a49565c783006d43d92f4cb51a4553762c6a9504.jpg)

![7b090d110822e36078a74c1e718a59eac01bbebc1a2a5584476bd53c64813188.jpg](../iclr_results/3330_MIA-Bench_ Towards Better Instruction Following Evaluation of Multimodal LLMs/tables/7b090d110822e36078a74c1e718a59eac01bbebc1a2a5584476bd53c64813188.jpg)

![815516aefb104878ff9c3644db308f2f9ee2a48508d1dd0c80278f90208c95a8.jpg](../iclr_results/3330_MIA-Bench_ Towards Better Instruction Following Evaluation of Multimodal LLMs/tables/815516aefb104878ff9c3644db308f2f9ee2a48508d1dd0c80278f90208c95a8.jpg)

![8df5b38362f849d92f25677ecc038a89f5a4b66f8af012bacb923225267a85ad.jpg](../iclr_results/3330_MIA-Bench_ Towards Better Instruction Following Evaluation of Multimodal LLMs/tables/8df5b38362f849d92f25677ecc038a89f5a4b66f8af012bacb923225267a85ad.jpg)

![97c150de4439747829459c70bbe052b14351d6a60ff6377dc55a2f9bfd50cdaf.jpg](../iclr_results/3330_MIA-Bench_ Towards Better Instruction Following Evaluation of Multimodal LLMs/tables/97c150de4439747829459c70bbe052b14351d6a60ff6377dc55a2f9bfd50cdaf.jpg)

![c3f790756034f91f75a6193690f780d53639e9f117cb6562c629271d4e179521.jpg](../iclr_results/3330_MIA-Bench_ Towards Better Instruction Following Evaluation of Multimodal LLMs/tables/c3f790756034f91f75a6193690f780d53639e9f117cb6562c629271d4e179521.jpg)

![ffbcc46e566973d655e6bf06f59f1d5e5713ff1892e2a0c9ba6bbb4bd0a65052.jpg](../iclr_results/3330_MIA-Bench_ Towards Better Instruction Following Evaluation of Multimodal LLMs/tables/ffbcc46e566973d655e6bf06f59f1d5e5713ff1892e2a0c9ba6bbb4bd0a65052.jpg)

## Bandit Learning in Matching Markets with Indifference


### Images

![a3b4ee914f19cfed23fe05c1b95069b0eb4cdbde622230d96b82d974aa729203.jpg](../iclr_results/3331_Bandit Learning in Matching Markets with Indifference/images/a3b4ee914f19cfed23fe05c1b95069b0eb4cdbde622230d96b82d974aa729203.jpg)

![cefe1ca4edc601ff8fe9ff3b23313ccc039f3aec5ae1faaaf2b2a1e961f9d58b.jpg](../iclr_results/3331_Bandit Learning in Matching Markets with Indifference/images/cefe1ca4edc601ff8fe9ff3b23313ccc039f3aec5ae1faaaf2b2a1e961f9d58b.jpg)

### Tables

![8db9c43b9248fcd398fccae07c13d6678158192e24cb433467b70a6e3dfd7b20.jpg](../iclr_results/3331_Bandit Learning in Matching Markets with Indifference/tables/8db9c43b9248fcd398fccae07c13d6678158192e24cb433467b70a6e3dfd7b20.jpg)

![a89a0e4cca398623d34a43b29a47d06fc1d7b5746747fa18f1f420fdc7f969d2.jpg](../iclr_results/3331_Bandit Learning in Matching Markets with Indifference/tables/a89a0e4cca398623d34a43b29a47d06fc1d7b5746747fa18f1f420fdc7f969d2.jpg)

![b327b0a7facb52428cca30a8e29c592eb7f39b63670ed3509e88d4136262837c.jpg](../iclr_results/3331_Bandit Learning in Matching Markets with Indifference/tables/b327b0a7facb52428cca30a8e29c592eb7f39b63670ed3509e88d4136262837c.jpg)

## PuzzleFusion++: Auto-agglomerative 3D Fracture Assembly by Denoise and Verify


### Images

![0bcd342a341a4c2a6a6fc9f60fe85beceb52722925885d43f50fe3130a5a59ed.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/images/0bcd342a341a4c2a6a6fc9f60fe85beceb52722925885d43f50fe3130a5a59ed.jpg)

![192478cd28ce059aac1f9cc292753b9016c6607c7fb73b4c9f729437e5c8d68c.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/images/192478cd28ce059aac1f9cc292753b9016c6607c7fb73b4c9f729437e5c8d68c.jpg)

![2022e5f6c04450ce470c95d8a55ddfec02138e44cc145c5c3f9d4e95eeff7b8a.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/images/2022e5f6c04450ce470c95d8a55ddfec02138e44cc145c5c3f9d4e95eeff7b8a.jpg)

![32fd02b6ef89bef5247bd6b95de116cf4b5cbb7865a81e85ad48f75f9295f968.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/images/32fd02b6ef89bef5247bd6b95de116cf4b5cbb7865a81e85ad48f75f9295f968.jpg)

![39d8f2f773f0fad14e33569ad5dc5eb1a562a06d9c3dbeb7775d8d349b7294e9.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/images/39d8f2f773f0fad14e33569ad5dc5eb1a562a06d9c3dbeb7775d8d349b7294e9.jpg)

![4baad2fb535ad62357dc8ac34e7d74267fb98b9f3513b8d4ee516d2cf61c2154.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/images/4baad2fb535ad62357dc8ac34e7d74267fb98b9f3513b8d4ee516d2cf61c2154.jpg)

![76b454a9948215fc745840fd70c61ff42560c68fc9a9dfe1fed694effcf78135.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/images/76b454a9948215fc745840fd70c61ff42560c68fc9a9dfe1fed694effcf78135.jpg)

![7a6a7264a9c44d0d5fb29e84fdc4be82803656df91d5fabc0d4a7871f0338542.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/images/7a6a7264a9c44d0d5fb29e84fdc4be82803656df91d5fabc0d4a7871f0338542.jpg)

![7b9be11c6dd08ceaa9887dfcc6188178e3451ff627a6f42afadad874ec137d5f.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/images/7b9be11c6dd08ceaa9887dfcc6188178e3451ff627a6f42afadad874ec137d5f.jpg)

![8df9987b8eeacc9bd78e59bf2532ab0206c29362e4c01135926515e46d8c86dd.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/images/8df9987b8eeacc9bd78e59bf2532ab0206c29362e4c01135926515e46d8c86dd.jpg)

![9ccddc1ec21d84fabb0bb51da1c0e0558ff3d64595924e5d44ad8cb500b44521.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/images/9ccddc1ec21d84fabb0bb51da1c0e0558ff3d64595924e5d44ad8cb500b44521.jpg)

![9e179d5da0a93f961d6fc2e94fd2a820138442ffc85fe9025f6c5ef388cee5d3.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/images/9e179d5da0a93f961d6fc2e94fd2a820138442ffc85fe9025f6c5ef388cee5d3.jpg)

![a68c4228fa1b1fbe34cad117c3ac47b5c9e6ef7651985848e01ce80e5530589e.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/images/a68c4228fa1b1fbe34cad117c3ac47b5c9e6ef7651985848e01ce80e5530589e.jpg)

![c12596cc6e95b4b7c25a7ed57c3eb74e00ca11bc612e59b28ed7d1bdb61e05c6.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/images/c12596cc6e95b4b7c25a7ed57c3eb74e00ca11bc612e59b28ed7d1bdb61e05c6.jpg)

![c24dcaf96c05326013799453fa26500b7ae05e7ef6db196d9ff4c9c29872b8d9.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/images/c24dcaf96c05326013799453fa26500b7ae05e7ef6db196d9ff4c9c29872b8d9.jpg)

![de9d4caa4e284d445fa28b6629b8a5a82f95c1c466b02333c0b7ccbeac9da543.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/images/de9d4caa4e284d445fa28b6629b8a5a82f95c1c466b02333c0b7ccbeac9da543.jpg)

![e53f65dbc1c9ae739cb7fc6bf27913b92a63f949952bcceb2e344c8bcbf38beb.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/images/e53f65dbc1c9ae739cb7fc6bf27913b92a63f949952bcceb2e344c8bcbf38beb.jpg)

### Tables

![172010d831d8ea19cdc82b1bfb4065936917da3169dd494382a8add1be63ed89.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/tables/172010d831d8ea19cdc82b1bfb4065936917da3169dd494382a8add1be63ed89.jpg)

![47f9143e56acf8f8e58b6aa12ee99f94dafe7a01b3fc6fc860f8a18729e0e689.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/tables/47f9143e56acf8f8e58b6aa12ee99f94dafe7a01b3fc6fc860f8a18729e0e689.jpg)

![64d3eb6c1656921b07128e1ecd9a663a2ddcb4d55290ce71a33c4fc300ec9bfc.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/tables/64d3eb6c1656921b07128e1ecd9a663a2ddcb4d55290ce71a33c4fc300ec9bfc.jpg)

![669b4d5da9356b606a3b2e45e5d44224aa36f89511cc5294577721e8bbcce792.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/tables/669b4d5da9356b606a3b2e45e5d44224aa36f89511cc5294577721e8bbcce792.jpg)

![6d0c4c9a614b482670eb4c25e4b7b88ce653413e2fb073f5e678a8bc58628d04.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/tables/6d0c4c9a614b482670eb4c25e4b7b88ce653413e2fb073f5e678a8bc58628d04.jpg)

![719254f5069e97afb6c650934848dda652ce8858421165d828c59c566488b94e.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/tables/719254f5069e97afb6c650934848dda652ce8858421165d828c59c566488b94e.jpg)

![9624a3cd78677b1a76e72a570cdbb8964356baae3ef01bac251c6e7313280fa3.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/tables/9624a3cd78677b1a76e72a570cdbb8964356baae3ef01bac251c6e7313280fa3.jpg)

![9816f4b25f288ef632303d9d7e56ae23fcf3270b57d4bf8a19d61dbab372f5b0.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/tables/9816f4b25f288ef632303d9d7e56ae23fcf3270b57d4bf8a19d61dbab372f5b0.jpg)

![a7a297cae8ab0d5f877c3b86365601f3a9f910d96e1a4aff3e84fc313d9dd37b.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/tables/a7a297cae8ab0d5f877c3b86365601f3a9f910d96e1a4aff3e84fc313d9dd37b.jpg)

![b53f3aa1137820c5dfa361b74573a4d86697801b0bad3ca3434b487e3d1b594d.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/tables/b53f3aa1137820c5dfa361b74573a4d86697801b0bad3ca3434b487e3d1b594d.jpg)

![baff36ae8a68ed7da3959de93d0779387620f36d6481af423636bcff5ffc23c4.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/tables/baff36ae8a68ed7da3959de93d0779387620f36d6481af423636bcff5ffc23c4.jpg)

![d85a19e1357778d6ec94bf62b24be49f7fc6af12ae3c624a2b75e12b3719067e.jpg](../iclr_results/3332_PuzzleFusion++_ Auto-agglomerative 3D Fracture Assembly by Denoise and Verify/tables/d85a19e1357778d6ec94bf62b24be49f7fc6af12ae3c624a2b75e12b3719067e.jpg)

## Training One-Dimensional Graph Neural Networks is NP-Hard


### Images

![1299bf3d9620e32bf4202362d9fab887c9a48b628905cfbc5816145ac730fec4.jpg](../iclr_results/3333_Training One-Dimensional Graph Neural Networks is NP-Hard/images/1299bf3d9620e32bf4202362d9fab887c9a48b628905cfbc5816145ac730fec4.jpg)

![68753e524c2cf83a5422e17dad9b28c716f2a8118975b4131ed76f0f91846f68.jpg](../iclr_results/3333_Training One-Dimensional Graph Neural Networks is NP-Hard/images/68753e524c2cf83a5422e17dad9b28c716f2a8118975b4131ed76f0f91846f68.jpg)

![8112f4875507163cc228931115bab12e272a3254eb7cf172f388c2360e1accd1.jpg](../iclr_results/3333_Training One-Dimensional Graph Neural Networks is NP-Hard/images/8112f4875507163cc228931115bab12e272a3254eb7cf172f388c2360e1accd1.jpg)

![fd3e34a889e21de23ec43967614d016d012fa321c6ae4ff5bc06595382940867.jpg](../iclr_results/3333_Training One-Dimensional Graph Neural Networks is NP-Hard/images/fd3e34a889e21de23ec43967614d016d012fa321c6ae4ff5bc06595382940867.jpg)

### Tables

![7462d251f6d9f87ebb9236cf3908783505b18edbe5008fa1529322e4fc12307e.jpg](../iclr_results/3333_Training One-Dimensional Graph Neural Networks is NP-Hard/tables/7462d251f6d9f87ebb9236cf3908783505b18edbe5008fa1529322e4fc12307e.jpg)

## Regretful Decisions under Label Noise


### Images

![0eb8f96cbc83855d9dbaeb0c216f3756c30c29d931b85ae16a268a0b25b56040.jpg](../iclr_results/3334_Regretful Decisions under Label Noise/images/0eb8f96cbc83855d9dbaeb0c216f3756c30c29d931b85ae16a268a0b25b56040.jpg)

![5251f6261d5f087595163ec61ba5d66bfb6cfdc9d4c191d54117d47438749c60.jpg](../iclr_results/3334_Regretful Decisions under Label Noise/images/5251f6261d5f087595163ec61ba5d66bfb6cfdc9d4c191d54117d47438749c60.jpg)

![a9376a8ea1cd3c2164a81ff10282606e5c3577fffbb2ff13b37ef951f201f170.jpg](../iclr_results/3334_Regretful Decisions under Label Noise/images/a9376a8ea1cd3c2164a81ff10282606e5c3577fffbb2ff13b37ef951f201f170.jpg)

![bdd7b20f557f17c03e2cc1abf3f5f88106f5a4461d44920ffa8394780b329bb0.jpg](../iclr_results/3334_Regretful Decisions under Label Noise/images/bdd7b20f557f17c03e2cc1abf3f5f88106f5a4461d44920ffa8394780b329bb0.jpg)

### Tables

![07b366ac71c3fc50e114af517578e613ff2baa377890186f559555fb8a3e7c9b.jpg](../iclr_results/3334_Regretful Decisions under Label Noise/tables/07b366ac71c3fc50e114af517578e613ff2baa377890186f559555fb8a3e7c9b.jpg)

![243c2933c6407981da05172f1e917d9367fc2d8c2335d1efc071f983a2153209.jpg](../iclr_results/3334_Regretful Decisions under Label Noise/tables/243c2933c6407981da05172f1e917d9367fc2d8c2335d1efc071f983a2153209.jpg)

![4511298a007176f1d4a34a6dae38aaede2c937e1a36f6efe13f46e360a557702.jpg](../iclr_results/3334_Regretful Decisions under Label Noise/tables/4511298a007176f1d4a34a6dae38aaede2c937e1a36f6efe13f46e360a557702.jpg)

![66db73a592e3a4c8002ca02806e80104fc41e295a6bc91859f4f8d2bd09f85ad.jpg](../iclr_results/3334_Regretful Decisions under Label Noise/tables/66db73a592e3a4c8002ca02806e80104fc41e295a6bc91859f4f8d2bd09f85ad.jpg)

## Bad-PFL: Exploiting Backdoor Attacks against Personalized Federated Learning


### Images

![195b13b2b66eb4ba86e8c0dd6fff7821ae4ce033795651d143d6bcfda5d1f2b7.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/images/195b13b2b66eb4ba86e8c0dd6fff7821ae4ce033795651d143d6bcfda5d1f2b7.jpg)

![2f4c8834f7ba0a2215de7ba88c0d81f31041b92a14ce5309d9cb9d403c0eb5c3.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/images/2f4c8834f7ba0a2215de7ba88c0d81f31041b92a14ce5309d9cb9d403c0eb5c3.jpg)

![5715c9577df0c302231a8b63ce6ec007e17ac17dda20322b441b47e19735bcf5.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/images/5715c9577df0c302231a8b63ce6ec007e17ac17dda20322b441b47e19735bcf5.jpg)

![5c838c480660cdf5c8d8ed43c44cdeb3326d799e8c64aede9dca47662ac83c7b.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/images/5c838c480660cdf5c8d8ed43c44cdeb3326d799e8c64aede9dca47662ac83c7b.jpg)

![7a752087c4a0f9ff3e7bdc12a447f5b0af7d845cb6e3258c495673274c43abfb.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/images/7a752087c4a0f9ff3e7bdc12a447f5b0af7d845cb6e3258c495673274c43abfb.jpg)

![849dc2b221b901a3bb197c35b1b35d71f28d177f96467b053722131c4096905c.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/images/849dc2b221b901a3bb197c35b1b35d71f28d177f96467b053722131c4096905c.jpg)

![84cc144374e05b2344e1d601c4ddd13d482445841eb3073133e8fbff94952d1a.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/images/84cc144374e05b2344e1d601c4ddd13d482445841eb3073133e8fbff94952d1a.jpg)

![869e33d1dfb33da5f15c7e4fa69d2528c467cb5385a99855d78d379120da3263.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/images/869e33d1dfb33da5f15c7e4fa69d2528c467cb5385a99855d78d379120da3263.jpg)

![acbd9b538185bc222cb639474535ca395672104166f81f57a6be0fbc953922fd.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/images/acbd9b538185bc222cb639474535ca395672104166f81f57a6be0fbc953922fd.jpg)

![b6dd85497cb3600ca53f9da96560e98e04bd83b4d0c6a345a6535cb27758f068.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/images/b6dd85497cb3600ca53f9da96560e98e04bd83b4d0c6a345a6535cb27758f068.jpg)

![d0564ff183d4e007f08d745c2e7af3e4a4d0b0931cc5dce963c4b12ae94c9d56.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/images/d0564ff183d4e007f08d745c2e7af3e4a4d0b0931cc5dce963c4b12ae94c9d56.jpg)

![d36f99689412bd77b6a2b3fa96394fcb8f617905cfbd7af1e4130f61ddb3fee3.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/images/d36f99689412bd77b6a2b3fa96394fcb8f617905cfbd7af1e4130f61ddb3fee3.jpg)

![d7a103f8c18044bb9c8d227030da36f94c28581cc01c233d9182c70cb74f4c2f.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/images/d7a103f8c18044bb9c8d227030da36f94c28581cc01c233d9182c70cb74f4c2f.jpg)

![d8bbd8618d53a94241e9bf2a190af113b57795a3564539e4c258a5158beb6dc8.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/images/d8bbd8618d53a94241e9bf2a190af113b57795a3564539e4c258a5158beb6dc8.jpg)

![ddf98ee79b99fb7d8b854934362f416f8e1ef9ae5d06f8d7b6e879c5905f61a7.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/images/ddf98ee79b99fb7d8b854934362f416f8e1ef9ae5d06f8d7b6e879c5905f61a7.jpg)

![ec4845ae69bd2f129b4e3bde4c5bbcdcb77c762d9304482a2c2cf51028514250.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/images/ec4845ae69bd2f129b4e3bde4c5bbcdcb77c762d9304482a2c2cf51028514250.jpg)

![fb99c1cc3eac2c5a8cf5e5efa87e7c45b732d9ce7e58c94f6334d88f0849ba2e.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/images/fb99c1cc3eac2c5a8cf5e5efa87e7c45b732d9ce7e58c94f6334d88f0849ba2e.jpg)

![feb76f3d79fe5663ac5ab2fd8485f5c5c95d4f702c38101e124d51283654c82b.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/images/feb76f3d79fe5663ac5ab2fd8485f5c5c95d4f702c38101e124d51283654c82b.jpg)

### Tables

![06618b9930d483a440b05a5eccc3c7b985a53fc8606b51ce058a659267a75f16.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/06618b9930d483a440b05a5eccc3c7b985a53fc8606b51ce058a659267a75f16.jpg)

![0713bf3fd2bee86eb00c025fa9b6c9522cdf979ed271e28fa335a6ff234ea578.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/0713bf3fd2bee86eb00c025fa9b6c9522cdf979ed271e28fa335a6ff234ea578.jpg)

![138723ddbcfbcd17e8dff7b8c3c212007f61dbe03060755da687ae57f13c59de.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/138723ddbcfbcd17e8dff7b8c3c212007f61dbe03060755da687ae57f13c59de.jpg)

![14f6f0d66505a2a964a7ec7c86bd03012c3cb3e2f9d97dff7ada7396c54cfffb.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/14f6f0d66505a2a964a7ec7c86bd03012c3cb3e2f9d97dff7ada7396c54cfffb.jpg)

![1a01830731d112d289c3566a18792a85365884427e3349699cbfd0e5d75c8248.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/1a01830731d112d289c3566a18792a85365884427e3349699cbfd0e5d75c8248.jpg)

![20fa377d9464e25bcf977815fab727c323ada3b83f8bac32a4b8a6f6c850aaa5.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/20fa377d9464e25bcf977815fab727c323ada3b83f8bac32a4b8a6f6c850aaa5.jpg)

![2d9ffb403e1ae6075916342431a288d398db92eca60e5f8e82ca0fc6e8442da7.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/2d9ffb403e1ae6075916342431a288d398db92eca60e5f8e82ca0fc6e8442da7.jpg)

![36c6a490b7802b979ec02a797b47b3d6a5fc670dbcae9501cb06651ad9f73fee.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/36c6a490b7802b979ec02a797b47b3d6a5fc670dbcae9501cb06651ad9f73fee.jpg)

![439e1583dc6c45a5ded15eaf0e02c60924cf813a63fe353cd0372e14e8a7e7e7.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/439e1583dc6c45a5ded15eaf0e02c60924cf813a63fe353cd0372e14e8a7e7e7.jpg)

![4d632105aea3312fff3a016c4abe05b0118be4fb3fdad063eeda6501d4ba7040.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/4d632105aea3312fff3a016c4abe05b0118be4fb3fdad063eeda6501d4ba7040.jpg)

![5a9bb0da1b32ab46b539bf87cc00d1f66640a21f7e7107c20bd84afb1ccee550.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/5a9bb0da1b32ab46b539bf87cc00d1f66640a21f7e7107c20bd84afb1ccee550.jpg)

![5b16a4efdc1aefddb32ab6232f4db2bc3d29128298d882a145a6a4353c49915f.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/5b16a4efdc1aefddb32ab6232f4db2bc3d29128298d882a145a6a4353c49915f.jpg)

![6a658fe9124f36d9667ca4deca56a5ae5922a7df51c04708a207646be9d21b78.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/6a658fe9124f36d9667ca4deca56a5ae5922a7df51c04708a207646be9d21b78.jpg)

![7dc593e4327657efaccf5edbe71f90d750c21fbf8f2201dd1e0624fb8660ffe0.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/7dc593e4327657efaccf5edbe71f90d750c21fbf8f2201dd1e0624fb8660ffe0.jpg)

![7e64edd65387917f8523af62a027a883139d7a83e07aae4185d853b385917f75.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/7e64edd65387917f8523af62a027a883139d7a83e07aae4185d853b385917f75.jpg)

![9e09775e9e8701afbea760c1fa7e561e39eb13432d08f880757022fa0e287e2e.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/9e09775e9e8701afbea760c1fa7e561e39eb13432d08f880757022fa0e287e2e.jpg)

![a209efd14dfa682ec34c4a73a61f8c8a7979288c71d07be5d57a929df11f14db.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/a209efd14dfa682ec34c4a73a61f8c8a7979288c71d07be5d57a929df11f14db.jpg)

![b43e09be9c7b386147f0b2ca5c68e9578144b9daf1d6aaf87902ea42f0a833c7.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/b43e09be9c7b386147f0b2ca5c68e9578144b9daf1d6aaf87902ea42f0a833c7.jpg)

![beab89a3265371de01544cdac9ecf0497310ebad060b349613f04b5ff2ed1dbb.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/beab89a3265371de01544cdac9ecf0497310ebad060b349613f04b5ff2ed1dbb.jpg)

![d8526ea22687b7a046ef84a2be1fec27b64eba6b925afef6d873db0e7b5dd26a.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/d8526ea22687b7a046ef84a2be1fec27b64eba6b925afef6d873db0e7b5dd26a.jpg)

![debdae274cbbea604d6335265707e0a16262ecf8aa8de73f03c44d40bed8b2a4.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/debdae274cbbea604d6335265707e0a16262ecf8aa8de73f03c44d40bed8b2a4.jpg)

![ee3d787e51d2cf281aadf751e41ba038655e91c0ed2dc34f0c20804acc0bdf34.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/ee3d787e51d2cf281aadf751e41ba038655e91c0ed2dc34f0c20804acc0bdf34.jpg)

![faf758ac8b028b1e2be302fb4e1d76c5b291eb37aa16070ed30d1905dcec2f39.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/faf758ac8b028b1e2be302fb4e1d76c5b291eb37aa16070ed30d1905dcec2f39.jpg)

![fe14343c0b6ab6390ae70bc72d07f54e5723b558ffd8ce5b869c9fdf65f6fd96.jpg](../iclr_results/3335_Bad-PFL_ Exploiting Backdoor Attacks against Personalized Federated Learning/tables/fe14343c0b6ab6390ae70bc72d07f54e5723b558ffd8ce5b869c9fdf65f6fd96.jpg)

## Small-to-Large Generalization: Training Data Influences Models Consistently Across Scale


### Images

![0977b83fa9711019a2a023372555221f400950e7ec6f10294793607d2b02bf8c.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/images/0977b83fa9711019a2a023372555221f400950e7ec6f10294793607d2b02bf8c.jpg)

![0b6c022757c3ecef50fe804f5a97c2711126016023adacbf2ff2c649dcdcfcbe.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/images/0b6c022757c3ecef50fe804f5a97c2711126016023adacbf2ff2c649dcdcfcbe.jpg)

![0b7a74212696329892866c858ee4c665308d8960f696ba300da2cb8dab073b07.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/images/0b7a74212696329892866c858ee4c665308d8960f696ba300da2cb8dab073b07.jpg)

![2954b85a2d1f6f9ce741df8968e6d5c25f072304eceb749cfef42c64677ae199.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/images/2954b85a2d1f6f9ce741df8968e6d5c25f072304eceb749cfef42c64677ae199.jpg)

![2b727763581740795e66f0bfa06de4e45ed6d34e54a9409f96ccd10cf91b657d.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/images/2b727763581740795e66f0bfa06de4e45ed6d34e54a9409f96ccd10cf91b657d.jpg)

![40ed70a49aa834ba343c0f71b2410805f3772395a5406739abf339a8dba2ec2b.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/images/40ed70a49aa834ba343c0f71b2410805f3772395a5406739abf339a8dba2ec2b.jpg)

![482585752db2a6865e646b0f1a2ad9f84486045c04518daa361fa77a69a89425.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/images/482585752db2a6865e646b0f1a2ad9f84486045c04518daa361fa77a69a89425.jpg)

![54a699620049b65e9360f555e6192dd93ce553b5ecb82d789b6e9c029b67363c.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/images/54a699620049b65e9360f555e6192dd93ce553b5ecb82d789b6e9c029b67363c.jpg)

![67ed4f1b7c7242572ca4012b655548659988017cb127e363ba5a21ea82122076.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/images/67ed4f1b7c7242572ca4012b655548659988017cb127e363ba5a21ea82122076.jpg)

![74bda37542a438531d4fdb157d895cfa3bbf53fc1cbeb0139a819838820a9a9a.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/images/74bda37542a438531d4fdb157d895cfa3bbf53fc1cbeb0139a819838820a9a9a.jpg)

![8454d496bf1573e418f06f8199454c2eee851377ad12c59da637dcbd6871e12c.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/images/8454d496bf1573e418f06f8199454c2eee851377ad12c59da637dcbd6871e12c.jpg)

![96d7b3659c48c0383a74b2b0ce7e04d6a5fcc9ca1fc48a004097affab140cbce.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/images/96d7b3659c48c0383a74b2b0ce7e04d6a5fcc9ca1fc48a004097affab140cbce.jpg)

![a661ee6a5326aa3a07dcd80095d74e95c32fd7fc3de6afa604ba0886a7f4c720.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/images/a661ee6a5326aa3a07dcd80095d74e95c32fd7fc3de6afa604ba0886a7f4c720.jpg)

![b0afaa468664080029a4273d7e2059c7f609e768868e975d6058de831e259f42.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/images/b0afaa468664080029a4273d7e2059c7f609e768868e975d6058de831e259f42.jpg)

![b19a748eab92c660d7989da2afbe0b03fee454e225a0d746f51d3f82076df616.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/images/b19a748eab92c660d7989da2afbe0b03fee454e225a0d746f51d3f82076df616.jpg)

![c00b0520ca8a0e8db3dffa2fc7a9548e38dd2da9baa0779d67edb586658c86af.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/images/c00b0520ca8a0e8db3dffa2fc7a9548e38dd2da9baa0779d67edb586658c86af.jpg)

![c0dba640d9625090f36164854da221107cfc95fa3783a21531080fc847691d8a.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/images/c0dba640d9625090f36164854da221107cfc95fa3783a21531080fc847691d8a.jpg)

![d162fc74ed9b0412d00c264a07d4b7a718bf048ae2b6b937a2641468c05ba1ff.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/images/d162fc74ed9b0412d00c264a07d4b7a718bf048ae2b6b937a2641468c05ba1ff.jpg)

### Tables

![71c686bec1f01b8d90d327768ef902e0f73e83c9fe0afe935897b1028432e74e.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/tables/71c686bec1f01b8d90d327768ef902e0f73e83c9fe0afe935897b1028432e74e.jpg)

![80ad5878f8910887599f9801a84bfca40bc16a9f2a3b5ffeb49a709e16e46ddc.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/tables/80ad5878f8910887599f9801a84bfca40bc16a9f2a3b5ffeb49a709e16e46ddc.jpg)

![bc48f0679d1fbebea85d884f9b3844b670fec298ef39c38c76fa4eeb84875c96.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/tables/bc48f0679d1fbebea85d884f9b3844b670fec298ef39c38c76fa4eeb84875c96.jpg)

![c430680b030dbc9acad8b43a2555f807906ece0d8bf4089f14f5237fe81bea70.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/tables/c430680b030dbc9acad8b43a2555f807906ece0d8bf4089f14f5237fe81bea70.jpg)

![ed86233f03b6dc2dd43e5024445069b5d96ca2e96c569f5d0694c20d3d276a04.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/tables/ed86233f03b6dc2dd43e5024445069b5d96ca2e96c569f5d0694c20d3d276a04.jpg)

![f93d647048edff843d7eb047e5778fc9d386d7c4c897c323f6c9a5f43e53f8b2.jpg](../iclr_results/3336_Small-to-Large Generalization_ Training Data Influences Models Consistently Across Scale/tables/f93d647048edff843d7eb047e5778fc9d386d7c4c897c323f6c9a5f43e53f8b2.jpg)

## MADGEN: Mass-Spec attends to De Novo Molecular generation


### Images

![50a6a0e08dd9c364086712c0a0e659cafd95f8c7bb04542272408f5a8ebb5676.jpg](../iclr_results/3337_MADGEN_ Mass-Spec attends to De Novo Molecular generation/images/50a6a0e08dd9c364086712c0a0e659cafd95f8c7bb04542272408f5a8ebb5676.jpg)

![7021baf4cdad7e0c209ce23589df4cf01adf3981af8010e401bfe767cf5301ea.jpg](../iclr_results/3337_MADGEN_ Mass-Spec attends to De Novo Molecular generation/images/7021baf4cdad7e0c209ce23589df4cf01adf3981af8010e401bfe767cf5301ea.jpg)

![ce22befa07a4e46e4f982b50595185980fa02fcf0db7b7dd1f33874c2d407163.jpg](../iclr_results/3337_MADGEN_ Mass-Spec attends to De Novo Molecular generation/images/ce22befa07a4e46e4f982b50595185980fa02fcf0db7b7dd1f33874c2d407163.jpg)

### Tables

![22dfeeae54ceb501adb8b07dac665f8e520075514a50c7897b16a8429fda4aae.jpg](../iclr_results/3337_MADGEN_ Mass-Spec attends to De Novo Molecular generation/tables/22dfeeae54ceb501adb8b07dac665f8e520075514a50c7897b16a8429fda4aae.jpg)

![82f83f97b7c213e1d10c1f9ab1ef0841d34163025499cc90b741613dc5a6e1be.jpg](../iclr_results/3337_MADGEN_ Mass-Spec attends to De Novo Molecular generation/tables/82f83f97b7c213e1d10c1f9ab1ef0841d34163025499cc90b741613dc5a6e1be.jpg)

![c08676e2719ca2b950344c4e7eabca0c78aa1f7e54c542b68878a3669d1ce170.jpg](../iclr_results/3337_MADGEN_ Mass-Spec attends to De Novo Molecular generation/tables/c08676e2719ca2b950344c4e7eabca0c78aa1f7e54c542b68878a3669d1ce170.jpg)

## On the Adversarial Risk of Test Time Adaptation: An Investigation into Realistic Test-Time Data Poisoning


### Images

![1bf2f92c62b6d4db5191257fcdd7746888ee168ea6696b90588252b3c8fefff0.jpg](../iclr_results/3338_On the Adversarial Risk of Test Time Adaptation_ An Investigation into Realistic Test-Time Data Pois/images/1bf2f92c62b6d4db5191257fcdd7746888ee168ea6696b90588252b3c8fefff0.jpg)

![3435abaa70615c96a2be21b64734ed3182ba178d863ccbf8f5690e3b0ad7c566.jpg](../iclr_results/3338_On the Adversarial Risk of Test Time Adaptation_ An Investigation into Realistic Test-Time Data Pois/images/3435abaa70615c96a2be21b64734ed3182ba178d863ccbf8f5690e3b0ad7c566.jpg)

![80698af050d7573a79fd8199923a2c7b2b4bff06e9eed9fd14e3f1d5ddd80c52.jpg](../iclr_results/3338_On the Adversarial Risk of Test Time Adaptation_ An Investigation into Realistic Test-Time Data Pois/images/80698af050d7573a79fd8199923a2c7b2b4bff06e9eed9fd14e3f1d5ddd80c52.jpg)

![f83af8ea7d843a7598dd78861ec558303acba5c126dc0bc3d800de95304a9d0d.jpg](../iclr_results/3338_On the Adversarial Risk of Test Time Adaptation_ An Investigation into Realistic Test-Time Data Pois/images/f83af8ea7d843a7598dd78861ec558303acba5c126dc0bc3d800de95304a9d0d.jpg)

### Tables

![05f2b8577fe8809849cab32b981d67899c344f64ccc9a116f83b8b31b1e3a605.jpg](../iclr_results/3338_On the Adversarial Risk of Test Time Adaptation_ An Investigation into Realistic Test-Time Data Pois/tables/05f2b8577fe8809849cab32b981d67899c344f64ccc9a116f83b8b31b1e3a605.jpg)

![3b451dd6137aaf7dee724080ba490b211693a182504a0844ab894a602f7a6835.jpg](../iclr_results/3338_On the Adversarial Risk of Test Time Adaptation_ An Investigation into Realistic Test-Time Data Pois/tables/3b451dd6137aaf7dee724080ba490b211693a182504a0844ab894a602f7a6835.jpg)

![42c7ffb28cf03677204af4c7fcdafceea2cfb087b675c5f1a81766a3d7a6843b.jpg](../iclr_results/3338_On the Adversarial Risk of Test Time Adaptation_ An Investigation into Realistic Test-Time Data Pois/tables/42c7ffb28cf03677204af4c7fcdafceea2cfb087b675c5f1a81766a3d7a6843b.jpg)

![5da61d88526f04b2a28e3d50b31b132d84dcc4ec86a56fcd89b25bfc204f701e.jpg](../iclr_results/3338_On the Adversarial Risk of Test Time Adaptation_ An Investigation into Realistic Test-Time Data Pois/tables/5da61d88526f04b2a28e3d50b31b132d84dcc4ec86a56fcd89b25bfc204f701e.jpg)

![6157c84d9cd506201ca9704d2fa1657d63d0d83ca662cd7c774a1c5efa277eb5.jpg](../iclr_results/3338_On the Adversarial Risk of Test Time Adaptation_ An Investigation into Realistic Test-Time Data Pois/tables/6157c84d9cd506201ca9704d2fa1657d63d0d83ca662cd7c774a1c5efa277eb5.jpg)

![6e151e3d3e32b46521fbf53a3cfc6ba904e89f75ee08dcf4412877f2eba2f369.jpg](../iclr_results/3338_On the Adversarial Risk of Test Time Adaptation_ An Investigation into Realistic Test-Time Data Pois/tables/6e151e3d3e32b46521fbf53a3cfc6ba904e89f75ee08dcf4412877f2eba2f369.jpg)

![862059ed3454a5fbab9874327dfc24eb2630ea4ad782417f12bc694278f3ec52.jpg](../iclr_results/3338_On the Adversarial Risk of Test Time Adaptation_ An Investigation into Realistic Test-Time Data Pois/tables/862059ed3454a5fbab9874327dfc24eb2630ea4ad782417f12bc694278f3ec52.jpg)

![93788a7359b7eff3057a6bb4d3a7644fa125963c8ec65ea9cabda9ded1869524.jpg](../iclr_results/3338_On the Adversarial Risk of Test Time Adaptation_ An Investigation into Realistic Test-Time Data Pois/tables/93788a7359b7eff3057a6bb4d3a7644fa125963c8ec65ea9cabda9ded1869524.jpg)

![d290d0668c6be2c3900c93effd15b708010542a92a9564e9afa5d7d460df7fd2.jpg](../iclr_results/3338_On the Adversarial Risk of Test Time Adaptation_ An Investigation into Realistic Test-Time Data Pois/tables/d290d0668c6be2c3900c93effd15b708010542a92a9564e9afa5d7d460df7fd2.jpg)

![dce9ddeb57acb48c139880ebfdae41c42edcae66194f35bcc493bc84528706f8.jpg](../iclr_results/3338_On the Adversarial Risk of Test Time Adaptation_ An Investigation into Realistic Test-Time Data Pois/tables/dce9ddeb57acb48c139880ebfdae41c42edcae66194f35bcc493bc84528706f8.jpg)

![f2d93cf0f5abe832dc5d5df34ba8fce3fb60bbd47b7dae1f2b8e854dc5bad81a.jpg](../iclr_results/3338_On the Adversarial Risk of Test Time Adaptation_ An Investigation into Realistic Test-Time Data Pois/tables/f2d93cf0f5abe832dc5d5df34ba8fce3fb60bbd47b7dae1f2b8e854dc5bad81a.jpg)

![f73774869e1d2448aedd024194e77aeec5c82a2b93d9eb66bf8e9269125cc1ab.jpg](../iclr_results/3338_On the Adversarial Risk of Test Time Adaptation_ An Investigation into Realistic Test-Time Data Pois/tables/f73774869e1d2448aedd024194e77aeec5c82a2b93d9eb66bf8e9269125cc1ab.jpg)

## Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models


### Images

![142109c047247f6e8b1428061e5a89b59b09a898b475edc4cdc9bb405fb92b54.jpg](../iclr_results/3339_Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models/images/142109c047247f6e8b1428061e5a89b59b09a898b475edc4cdc9bb405fb92b54.jpg)

![1c5f1fee219b0e9e6bd257866e3d91805656a6931514b6edb1f18cab75241903.jpg](../iclr_results/3339_Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models/images/1c5f1fee219b0e9e6bd257866e3d91805656a6931514b6edb1f18cab75241903.jpg)

![222770b1fdd05b7b43a92085ca6a31cf0ceb9907bbb5dbb84803b28f33ab4ced.jpg](../iclr_results/3339_Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models/images/222770b1fdd05b7b43a92085ca6a31cf0ceb9907bbb5dbb84803b28f33ab4ced.jpg)

![3a8490fe97f6b19a6d9ee4da4894e7cc283093573e0b7d8edbd48a05b893ca52.jpg](../iclr_results/3339_Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models/images/3a8490fe97f6b19a6d9ee4da4894e7cc283093573e0b7d8edbd48a05b893ca52.jpg)

![45071e649e2280304be154b51ea3ac44aad2818cf0c5111469b4f230be1c5a2b.jpg](../iclr_results/3339_Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models/images/45071e649e2280304be154b51ea3ac44aad2818cf0c5111469b4f230be1c5a2b.jpg)

![610461dabdbdbca77b0bcb95cf46fbe30c246af7a003853ccd1b7e6445c48fde.jpg](../iclr_results/3339_Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models/images/610461dabdbdbca77b0bcb95cf46fbe30c246af7a003853ccd1b7e6445c48fde.jpg)

![669704a7426784ac00385dd653c91c39874c2cfcf84393cc0a9446cf130dcd0f.jpg](../iclr_results/3339_Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models/images/669704a7426784ac00385dd653c91c39874c2cfcf84393cc0a9446cf130dcd0f.jpg)

![8416925e7865967045bac063b82735215eeb6308ff367e414703dbe01b6d7fc1.jpg](../iclr_results/3339_Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models/images/8416925e7865967045bac063b82735215eeb6308ff367e414703dbe01b6d7fc1.jpg)

![9ccc5bbc212d3a1bf81dfd6ce49670c2134e36ea4a641221c21b2991bc0e64d5.jpg](../iclr_results/3339_Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models/images/9ccc5bbc212d3a1bf81dfd6ce49670c2134e36ea4a641221c21b2991bc0e64d5.jpg)

![9fcc0f48ef891f34f96995b58c2e3b3a6c007d8a50ab19217516d652ab861c5b.jpg](../iclr_results/3339_Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models/images/9fcc0f48ef891f34f96995b58c2e3b3a6c007d8a50ab19217516d652ab861c5b.jpg)

![a13c00b6c3f49cb8a0ec29c81935960b50ac1365d65838281bc1756c128c5e1e.jpg](../iclr_results/3339_Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models/images/a13c00b6c3f49cb8a0ec29c81935960b50ac1365d65838281bc1756c128c5e1e.jpg)

![bfd12aaf667c99ebc4670f249bbc70c6b2ec075a4ff2b15362c84dbd573bfe96.jpg](../iclr_results/3339_Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models/images/bfd12aaf667c99ebc4670f249bbc70c6b2ec075a4ff2b15362c84dbd573bfe96.jpg)

![c7cab039317d3dd6c48257f4ef86148deeb993c4aacb43a626f00536339d1251.jpg](../iclr_results/3339_Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models/images/c7cab039317d3dd6c48257f4ef86148deeb993c4aacb43a626f00536339d1251.jpg)

![decbaabd99c3c76c6423b57ccf7d9fda8e31c07417279812d8a35c25df443b2d.jpg](../iclr_results/3339_Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models/images/decbaabd99c3c76c6423b57ccf7d9fda8e31c07417279812d8a35c25df443b2d.jpg)

![f17c9f02dcebd144bc2a5de03d871d0e4cacbbad1286b329351230923b976685.jpg](../iclr_results/3339_Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models/images/f17c9f02dcebd144bc2a5de03d871d0e4cacbbad1286b329351230923b976685.jpg)

### Tables

![1359d83b129d71acce4cf3bb45be4692a28ef730ce426d8846cecff92f684694.jpg](../iclr_results/3339_Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models/tables/1359d83b129d71acce4cf3bb45be4692a28ef730ce426d8846cecff92f684694.jpg)

![14c2ec06477aea14d1d446538f381f0141d42d8a1b5e9e91b0b9ad6bff5efb95.jpg](../iclr_results/3339_Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models/tables/14c2ec06477aea14d1d446538f381f0141d42d8a1b5e9e91b0b9ad6bff5efb95.jpg)

![84e6af62dc886f64b7bdfa3a4799db82779c176665827f5fda26b7b356a36528.jpg](../iclr_results/3339_Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models/tables/84e6af62dc886f64b7bdfa3a4799db82779c176665827f5fda26b7b356a36528.jpg)

## Multi-Label Test-Time Adaptation with Bound Entropy Minimization


### Images

![364689d60cf433a69227d6fdac676659167b6925ec60e8aa8a9f6bce522fe4ef.jpg](../iclr_results/3340_Multi-Label Test-Time Adaptation with Bound Entropy Minimization/images/364689d60cf433a69227d6fdac676659167b6925ec60e8aa8a9f6bce522fe4ef.jpg)

![d2e2738f187eecdff202d6a8d38023948c9973747ef199ddbc38e06acc94a6b5.jpg](../iclr_results/3340_Multi-Label Test-Time Adaptation with Bound Entropy Minimization/images/d2e2738f187eecdff202d6a8d38023948c9973747ef199ddbc38e06acc94a6b5.jpg)

![efd61f857592659c15f9ee9d77ad3823823056db8f1f793b7192f93a0d712b79.jpg](../iclr_results/3340_Multi-Label Test-Time Adaptation with Bound Entropy Minimization/images/efd61f857592659c15f9ee9d77ad3823823056db8f1f793b7192f93a0d712b79.jpg)

![f431a1f0f0482f65ecf2a18e201ce08258eea5fc78e10c3fd8799fef69d1e6d3.jpg](../iclr_results/3340_Multi-Label Test-Time Adaptation with Bound Entropy Minimization/images/f431a1f0f0482f65ecf2a18e201ce08258eea5fc78e10c3fd8799fef69d1e6d3.jpg)

### Tables

![4120f496e2a2e02a24c901398ae7650f204e22fdcfe3a4c521335947cc91fb6c.jpg](../iclr_results/3340_Multi-Label Test-Time Adaptation with Bound Entropy Minimization/tables/4120f496e2a2e02a24c901398ae7650f204e22fdcfe3a4c521335947cc91fb6c.jpg)

![52fdaae124ff5f32372a7b27d4fd0c2a2a1490d4cf0180c437df475bee8dc920.jpg](../iclr_results/3340_Multi-Label Test-Time Adaptation with Bound Entropy Minimization/tables/52fdaae124ff5f32372a7b27d4fd0c2a2a1490d4cf0180c437df475bee8dc920.jpg)

![8581f0472805bd7dddb84caba12b5b892131c012159af5a748e7adba956550b0.jpg](../iclr_results/3340_Multi-Label Test-Time Adaptation with Bound Entropy Minimization/tables/8581f0472805bd7dddb84caba12b5b892131c012159af5a748e7adba956550b0.jpg)

![86aa17f3c570a7c441cc8f88627031b2b61f65fa80c0b8133654487ad6cc3aa0.jpg](../iclr_results/3340_Multi-Label Test-Time Adaptation with Bound Entropy Minimization/tables/86aa17f3c570a7c441cc8f88627031b2b61f65fa80c0b8133654487ad6cc3aa0.jpg)

![b8c9283bb11dcacd2a89e7fd3442d3f788e355cc7a7f6de299a8dcf81c7af978.jpg](../iclr_results/3340_Multi-Label Test-Time Adaptation with Bound Entropy Minimization/tables/b8c9283bb11dcacd2a89e7fd3442d3f788e355cc7a7f6de299a8dcf81c7af978.jpg)

![e855665f9a84786f4e88e70452be37138b5c31bbc469692cfe6330e1936e5158.jpg](../iclr_results/3340_Multi-Label Test-Time Adaptation with Bound Entropy Minimization/tables/e855665f9a84786f4e88e70452be37138b5c31bbc469692cfe6330e1936e5158.jpg)

![f27eadec8367e39df6f441f39ed11f7832097e9e8dfdfd9c57699d811723d575.jpg](../iclr_results/3340_Multi-Label Test-Time Adaptation with Bound Entropy Minimization/tables/f27eadec8367e39df6f441f39ed11f7832097e9e8dfdfd9c57699d811723d575.jpg)

## Wayward Concepts In Multimodal Models


### Images

![06865883176111d12c2064982ba39e63fcc8c0a646c4348a75bae1c44339c892.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/images/06865883176111d12c2064982ba39e63fcc8c0a646c4348a75bae1c44339c892.jpg)

![17dcedb3c673ab68ec701f2bf38d87c3577df07401d88bdd9ab2497b572301fa.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/images/17dcedb3c673ab68ec701f2bf38d87c3577df07401d88bdd9ab2497b572301fa.jpg)

![2a22761d8a7e9e52cc6536df175c167136513eb27cf8c72df8c0bd153ea6790a.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/images/2a22761d8a7e9e52cc6536df175c167136513eb27cf8c72df8c0bd153ea6790a.jpg)

![4c02d93595e946efca40e7ac48f30deaf8049df54d49518a9ee312baca267aad.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/images/4c02d93595e946efca40e7ac48f30deaf8049df54d49518a9ee312baca267aad.jpg)

![606d36b5528c54125702d13c6189884728314c450693b00f51e0201d914482ca.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/images/606d36b5528c54125702d13c6189884728314c450693b00f51e0201d914482ca.jpg)

![6aa05f2e216971c5e5146fcf6138b4b5fa487af8e4545ab62ee863f9a8c4eafb.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/images/6aa05f2e216971c5e5146fcf6138b4b5fa487af8e4545ab62ee863f9a8c4eafb.jpg)

![6c52133d5e5a31e79fbb5badfe31e37875cfe6f85f94a50f4ee01ce5ffe3fdf1.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/images/6c52133d5e5a31e79fbb5badfe31e37875cfe6f85f94a50f4ee01ce5ffe3fdf1.jpg)

![6cc43aae206cfcfbe25137b83b5f97b4c239f72acd5ef7f5bd76fd82deedd3a7.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/images/6cc43aae206cfcfbe25137b83b5f97b4c239f72acd5ef7f5bd76fd82deedd3a7.jpg)

![6f087d94b6604517622d8f5ca299a1b6f05bbc07e8c37a97a90bae61aa6a3b69.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/images/6f087d94b6604517622d8f5ca299a1b6f05bbc07e8c37a97a90bae61aa6a3b69.jpg)

![71b325a2fb9b564aa36beba017b3f309ecc7ab6dfc12b37ce14adad1eed513da.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/images/71b325a2fb9b564aa36beba017b3f309ecc7ab6dfc12b37ce14adad1eed513da.jpg)

![788828db40c16d83e1b48dbc4f79977fcad9094612a8303fb4ec9a1f254f41b3.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/images/788828db40c16d83e1b48dbc4f79977fcad9094612a8303fb4ec9a1f254f41b3.jpg)

![837dc1335121e79ba052759fbe4df4d4c0ff464a26c20623a45d1382452d10b3.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/images/837dc1335121e79ba052759fbe4df4d4c0ff464a26c20623a45d1382452d10b3.jpg)

![83e0155dca8c010759935a0c0843645aac0beba20da9bb145e93e2bc0fa45a53.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/images/83e0155dca8c010759935a0c0843645aac0beba20da9bb145e93e2bc0fa45a53.jpg)

![859c9434e7b32fea4f02749d0e8fe6311109123cdd9c9c89c0780eb98f8f61e6.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/images/859c9434e7b32fea4f02749d0e8fe6311109123cdd9c9c89c0780eb98f8f61e6.jpg)

![982e2c3471bd9dc2828f792c0e42506ca59d05742aeec2391b56d7a69aa24416.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/images/982e2c3471bd9dc2828f792c0e42506ca59d05742aeec2391b56d7a69aa24416.jpg)

![ba13530c9ef7c3430013866516b4ab8d2fc55b7e34506d92a03acda94ac9d4df.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/images/ba13530c9ef7c3430013866516b4ab8d2fc55b7e34506d92a03acda94ac9d4df.jpg)

![c22581ae104cfc6f3d0ca55477835e32c404bae319f25f6374418731ced3fbcc.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/images/c22581ae104cfc6f3d0ca55477835e32c404bae319f25f6374418731ced3fbcc.jpg)

![c82cb6a98ed9cd7fc60b90dfc56b6e350a80936d4ac4bbd5bcdc6e0c429b0414.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/images/c82cb6a98ed9cd7fc60b90dfc56b6e350a80936d4ac4bbd5bcdc6e0c429b0414.jpg)

![d0fb22e80c241cb2ba564ab21437c70b189484e46fa016e858368f302e0d2319.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/images/d0fb22e80c241cb2ba564ab21437c70b189484e46fa016e858368f302e0d2319.jpg)

![e363f0b3fb5bf0b4395bf9c98470355c4b294a13873867c9dd6b4f212579840d.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/images/e363f0b3fb5bf0b4395bf9c98470355c4b294a13873867c9dd6b4f212579840d.jpg)

![e482cff2f5924115e53989d48dd5f88df3edb95bdbe64312c39986d4f90faa74.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/images/e482cff2f5924115e53989d48dd5f88df3edb95bdbe64312c39986d4f90faa74.jpg)

![ed6a0366e93d4a9e4c9b9ca6cda0f71813aea1c3b062faebd317e2070b8d991c.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/images/ed6a0366e93d4a9e4c9b9ca6cda0f71813aea1c3b062faebd317e2070b8d991c.jpg)

### Tables

![79dae74d0df1ae39fba6c0f093436aede95619f2acb09be3e44c5a4350046d04.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/tables/79dae74d0df1ae39fba6c0f093436aede95619f2acb09be3e44c5a4350046d04.jpg)

![7f5da4b633d03692444eb469b55fed406354d7a58275664e5823b03d3f65760c.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/tables/7f5da4b633d03692444eb469b55fed406354d7a58275664e5823b03d3f65760c.jpg)

![b6cd33b20c46c046b9d571634d810ffe292e869f86b0c75e04f0a1918d40a4d7.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/tables/b6cd33b20c46c046b9d571634d810ffe292e869f86b0c75e04f0a1918d40a4d7.jpg)

![c0cd1d05f1a912452605af7d25a30d4f27d7881ffe0f78a3846ff125963d7a3f.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/tables/c0cd1d05f1a912452605af7d25a30d4f27d7881ffe0f78a3846ff125963d7a3f.jpg)

![d60dc316222c1ffd46f172ec98afcccc8643934a3aaf96f0ae4e0ecfd38200c0.jpg](../iclr_results/3341_Wayward Concepts In Multimodal Models/tables/d60dc316222c1ffd46f172ec98afcccc8643934a3aaf96f0ae4e0ecfd38200c0.jpg)

## LaGeM: A Large Geometry Model for 3D Representation Learning and Diffusion


### Images

![05f253d7f6840d3cf403fafd21e997cffce0cf66507fc92db17345fb300ce59b.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/images/05f253d7f6840d3cf403fafd21e997cffce0cf66507fc92db17345fb300ce59b.jpg)

![2f12d294463288da6d25aa4dc58941d132a287935a7389a2d3dcc697765951ff.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/images/2f12d294463288da6d25aa4dc58941d132a287935a7389a2d3dcc697765951ff.jpg)

![37f740f417156e0eae998ce3cf9648afe4bc578f940799dd85e28f23a25393a7.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/images/37f740f417156e0eae998ce3cf9648afe4bc578f940799dd85e28f23a25393a7.jpg)

![5a1e493e2f2ebc42c7942fc325da2bbfe8a1c77f32ec4848cdc94888ff7fee97.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/images/5a1e493e2f2ebc42c7942fc325da2bbfe8a1c77f32ec4848cdc94888ff7fee97.jpg)

![5bb2ccd5fb33ff4fe310ac8a614be7eda1a7fc0e9c7e5c0b352de13a486bd22e.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/images/5bb2ccd5fb33ff4fe310ac8a614be7eda1a7fc0e9c7e5c0b352de13a486bd22e.jpg)

![5bd3e9e03a65b0d3c7908e0041701d5cd5890b0978ce2225418a9294e253ec6d.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/images/5bd3e9e03a65b0d3c7908e0041701d5cd5890b0978ce2225418a9294e253ec6d.jpg)

![626989b14a3eb422083a41f53c41a80998c8db07fec9615767d01abfc0d42baa.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/images/626989b14a3eb422083a41f53c41a80998c8db07fec9615767d01abfc0d42baa.jpg)

![676c70eb46c575e6f3de9896d458211732b0d9a52c3580c4818140649ea607b0.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/images/676c70eb46c575e6f3de9896d458211732b0d9a52c3580c4818140649ea607b0.jpg)

![84909a6b94f3d1f0f15dbc8a6ba6be22ecdaa2cdaee688d3053c9704f4ac8363.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/images/84909a6b94f3d1f0f15dbc8a6ba6be22ecdaa2cdaee688d3053c9704f4ac8363.jpg)

![a22a8e2076a9dc90c669840e25de821b23afeee63c47e4578e8c7f73f62e0607.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/images/a22a8e2076a9dc90c669840e25de821b23afeee63c47e4578e8c7f73f62e0607.jpg)

![a2d12393d6563bc09dae009178dfd39e9e6905c7313d0e5d99cf3ddf572c289a.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/images/a2d12393d6563bc09dae009178dfd39e9e6905c7313d0e5d99cf3ddf572c289a.jpg)

![b96ab18f830c58f9e1995ed5a58862890bd217c661fb081623e1d924d13952f3.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/images/b96ab18f830c58f9e1995ed5a58862890bd217c661fb081623e1d924d13952f3.jpg)

![cbf16ae3a5d21567df1a93f45497ad32cfb5014400c58163a207c2ca346bd3f5.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/images/cbf16ae3a5d21567df1a93f45497ad32cfb5014400c58163a207c2ca346bd3f5.jpg)

![ce5de42c1a0f4e07dce20630f365ec7ab67ca410ad7fd873f4ff4467340077de.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/images/ce5de42c1a0f4e07dce20630f365ec7ab67ca410ad7fd873f4ff4467340077de.jpg)

![e23b0f9cfbd009b646b52f1741df57f86219cdda87bb1711ae8be720282a3ec7.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/images/e23b0f9cfbd009b646b52f1741df57f86219cdda87bb1711ae8be720282a3ec7.jpg)

![f0db66b36c5b0af5b1be2e9a0079ba74f900e378011266d9907d8b61c0d79e80.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/images/f0db66b36c5b0af5b1be2e9a0079ba74f900e378011266d9907d8b61c0d79e80.jpg)

### Tables

![1aa7ffafef6fcdbd8e2ccbfe37c3191dc7ea4c6f2fb410081e8d7331fd09d98f.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/tables/1aa7ffafef6fcdbd8e2ccbfe37c3191dc7ea4c6f2fb410081e8d7331fd09d98f.jpg)

![23440ab53cd9484de1d2a943452fe778f3119a7f45d9dc35cb39f2bd3290c896.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/tables/23440ab53cd9484de1d2a943452fe778f3119a7f45d9dc35cb39f2bd3290c896.jpg)

![3c43a4472fb25d23fb4240dcddd455c58fec6a492260277372aba7605300a79d.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/tables/3c43a4472fb25d23fb4240dcddd455c58fec6a492260277372aba7605300a79d.jpg)

![4a5791e42f86d8467ed639090595475ec8692136de2982e03d92af4905450fbb.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/tables/4a5791e42f86d8467ed639090595475ec8692136de2982e03d92af4905450fbb.jpg)

![5bf14476545111a276f70c23d6f1275b958dff2a20c32cb2c4ef5afbc530cc75.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/tables/5bf14476545111a276f70c23d6f1275b958dff2a20c32cb2c4ef5afbc530cc75.jpg)

![5eb6876fb26f2110cf8ada7f046a8446998296e4c8b276778b45d1cc9b00874f.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/tables/5eb6876fb26f2110cf8ada7f046a8446998296e4c8b276778b45d1cc9b00874f.jpg)

![857079549aa21ae3dfece6c479d7e9af8468ebcf7a2d9a4d702700ba4fe1bb85.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/tables/857079549aa21ae3dfece6c479d7e9af8468ebcf7a2d9a4d702700ba4fe1bb85.jpg)

![b33c2667c16682f0273eb36366ef9d636301e4a7afb7f2f31d6d2f762c824f0e.jpg](../iclr_results/3342_LaGeM_ A Large Geometry Model for 3D Representation Learning and Diffusion/tables/b33c2667c16682f0273eb36366ef9d636301e4a7afb7f2f31d6d2f762c824f0e.jpg)

## Mini-Monkey: Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image Pyramid


### Images

![2249dc2f64bcbad9be09d72034a699cbfb3d38a6e4b73eaa6e77eb659ecaf52c.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /images/2249dc2f64bcbad9be09d72034a699cbfb3d38a6e4b73eaa6e77eb659ecaf52c.jpg)

![289ac6e42db80493301b2b9ec621d9458e354716697e8d5f424a1bc8bc7e6387.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /images/289ac6e42db80493301b2b9ec621d9458e354716697e8d5f424a1bc8bc7e6387.jpg)

![41ec0cd71490539b433fcbebf169bcf69a6ff10fdcbbb5f2ea77963c57551582.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /images/41ec0cd71490539b433fcbebf169bcf69a6ff10fdcbbb5f2ea77963c57551582.jpg)

![443996ecbbc8ed5a3248d7f70941cd0bd206d369767f8e7274caab52d0c7dda6.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /images/443996ecbbc8ed5a3248d7f70941cd0bd206d369767f8e7274caab52d0c7dda6.jpg)

![447d5f91cb8cdf8c433788ba663a876656c8a47c9673ca5c3bffc95747fc210d.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /images/447d5f91cb8cdf8c433788ba663a876656c8a47c9673ca5c3bffc95747fc210d.jpg)

![70adb88c060d2e8c799a2eac61c4f565a009fc8708af74492efca1ac0b35c819.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /images/70adb88c060d2e8c799a2eac61c4f565a009fc8708af74492efca1ac0b35c819.jpg)

![821098cefa21c03e9b44e7665877d4964e475f4d7a2447f7dfbf2ab6facc2d28.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /images/821098cefa21c03e9b44e7665877d4964e475f4d7a2447f7dfbf2ab6facc2d28.jpg)

![9b58ab0c35db4a48d7c0a2a56e570ae2e5ba0a712b1c7df0bf9206ac4d6525d4.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /images/9b58ab0c35db4a48d7c0a2a56e570ae2e5ba0a712b1c7df0bf9206ac4d6525d4.jpg)

![a7f916bd18af10394e8cd4f409c746dd7a8f48bbae144383d11d11a16bf0e792.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /images/a7f916bd18af10394e8cd4f409c746dd7a8f48bbae144383d11d11a16bf0e792.jpg)

### Tables

![2a87b6a614086741d8cf65b3dd3f7b112c26884033eb5c24f0d75b2d9a4b9bc0.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /tables/2a87b6a614086741d8cf65b3dd3f7b112c26884033eb5c24f0d75b2d9a4b9bc0.jpg)

![2babbc742fd18e96190b24e732a3216c6c22ac638f2885c4b3d33d8d581d0386.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /tables/2babbc742fd18e96190b24e732a3216c6c22ac638f2885c4b3d33d8d581d0386.jpg)

![3bf8c7716f70079bca16a8deb5b97d3e754f73f633c8ccd124f6c42d5f0c267e.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /tables/3bf8c7716f70079bca16a8deb5b97d3e754f73f633c8ccd124f6c42d5f0c267e.jpg)

![55ff5c23cfe0e9bfabe7d8657efc54330169c69baeae4156e92a3928e2a1615d.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /tables/55ff5c23cfe0e9bfabe7d8657efc54330169c69baeae4156e92a3928e2a1615d.jpg)

![689454f940057b52f0e959f3f93589fac5ed5e8a494be3793c065c82ba076863.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /tables/689454f940057b52f0e959f3f93589fac5ed5e8a494be3793c065c82ba076863.jpg)

![6eed79ef60c953d42135c4e6fc0a4f936a7bd3d2fbb9c786647a86246df14ccb.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /tables/6eed79ef60c953d42135c4e6fc0a4f936a7bd3d2fbb9c786647a86246df14ccb.jpg)

![7da5746c43fc7ef0a7890a93899c07f4f14359940c35f5d224d4af20faa4dee2.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /tables/7da5746c43fc7ef0a7890a93899c07f4f14359940c35f5d224d4af20faa4dee2.jpg)

![85414beb72b7ac41a7feef20ca5695aed0f5d78937605af969a982e635752a96.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /tables/85414beb72b7ac41a7feef20ca5695aed0f5d78937605af969a982e635752a96.jpg)

![8e556a1b6e68ecbe725857a067b2326158038a8bfd5cf5292e852760225d5442.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /tables/8e556a1b6e68ecbe725857a067b2326158038a8bfd5cf5292e852760225d5442.jpg)

![9ca2c8bcdfd5e0ef0e0b77b485e5492a840820ab6e271ae0b5e9160f72fb2855.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /tables/9ca2c8bcdfd5e0ef0e0b77b485e5492a840820ab6e271ae0b5e9160f72fb2855.jpg)

![a83cdfc6202800025d1b07a42786c154a3544ef2b101fb7cd5a44fbd2441c975.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /tables/a83cdfc6202800025d1b07a42786c154a3544ef2b101fb7cd5a44fbd2441c975.jpg)

![bd491cad48266df1a02bac9315c3e659f412cd27ae0b4f90b6c2b5d5c71f4c89.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /tables/bd491cad48266df1a02bac9315c3e659f412cd27ae0b4f90b6c2b5d5c71f4c89.jpg)

![c6194e01f74192e0b78e69e31cb86f7f8dd306831ccec6173b68be66160348a9.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /tables/c6194e01f74192e0b78e69e31cb86f7f8dd306831ccec6173b68be66160348a9.jpg)

![e32211df82a1c5cdc4b6f1de0c82c521525931eff029d6132955f4c111d13733.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /tables/e32211df82a1c5cdc4b6f1de0c82c521525931eff029d6132955f4c111d13733.jpg)

![e459a717e1f51018c403d5d46c8046f02f3271443fc3e737ecc56b9fa01b73b8.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /tables/e459a717e1f51018c403d5d46c8046f02f3271443fc3e737ecc56b9fa01b73b8.jpg)

![effd96c24743687a6fd8a05b90bb60464590fc26f880da548ea796ec118eddd6.jpg](../iclr_results/3343_Mini-Monkey_ Alleviating the Semantic Sawtooth Effect for Lightweight MLLMs via Complementary Image /tables/effd96c24743687a6fd8a05b90bb60464590fc26f880da548ea796ec118eddd6.jpg)

## ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery


### Images

![0d267a1267a842aa8e109b873d68f3209e912b84e72f27667dda8739fb5018e9.jpg](../iclr_results/3344_ScienceAgentBench_ Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discover/images/0d267a1267a842aa8e109b873d68f3209e912b84e72f27667dda8739fb5018e9.jpg)

![3a4fa654ac9cf0b99346c4a02724460b520173a72db66b322590d2852d0cec8f.jpg](../iclr_results/3344_ScienceAgentBench_ Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discover/images/3a4fa654ac9cf0b99346c4a02724460b520173a72db66b322590d2852d0cec8f.jpg)

![528cf34163d8f15822e737072fa7e42370d925aa5464f5f580871a6f0702a961.jpg](../iclr_results/3344_ScienceAgentBench_ Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discover/images/528cf34163d8f15822e737072fa7e42370d925aa5464f5f580871a6f0702a961.jpg)

![ec99e87fe8cf99aa387eae6e0b34f896e8268d591d505d96959daf3c3ff8fa4f.jpg](../iclr_results/3344_ScienceAgentBench_ Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discover/images/ec99e87fe8cf99aa387eae6e0b34f896e8268d591d505d96959daf3c3ff8fa4f.jpg)

![f07802add7222aca70e5708d3e1a5ad1ca624918e0ae5d2865c55810fad58c7f.jpg](../iclr_results/3344_ScienceAgentBench_ Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discover/images/f07802add7222aca70e5708d3e1a5ad1ca624918e0ae5d2865c55810fad58c7f.jpg)

### Tables

![03b69143f31f1c2904972e9a1a5e8718379846dd0bfbc71933a5cc17d6ba9ffe.jpg](../iclr_results/3344_ScienceAgentBench_ Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discover/tables/03b69143f31f1c2904972e9a1a5e8718379846dd0bfbc71933a5cc17d6ba9ffe.jpg)

![14c640c9aea36130ffcc79f8465bc106c5f54857abd9062434015a99580a11fa.jpg](../iclr_results/3344_ScienceAgentBench_ Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discover/tables/14c640c9aea36130ffcc79f8465bc106c5f54857abd9062434015a99580a11fa.jpg)

![18d0e71eec915841209f5eda8d84d49435cc88b4abbd78ba24eaaae513c1ad3d.jpg](../iclr_results/3344_ScienceAgentBench_ Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discover/tables/18d0e71eec915841209f5eda8d84d49435cc88b4abbd78ba24eaaae513c1ad3d.jpg)

![251b8b4b8521b87404d90ef69cf08d268d93adc1eb1f0bd62e2f7d4287169859.jpg](../iclr_results/3344_ScienceAgentBench_ Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discover/tables/251b8b4b8521b87404d90ef69cf08d268d93adc1eb1f0bd62e2f7d4287169859.jpg)

![4136573d14fc199e3fe2b9dad3db382f4c1d7e3fced86cfb9166df8863554ebb.jpg](../iclr_results/3344_ScienceAgentBench_ Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discover/tables/4136573d14fc199e3fe2b9dad3db382f4c1d7e3fced86cfb9166df8863554ebb.jpg)

![44d86d03e943307fc219ed04e355713b03c4caf789a4336e63d3dc0631ec6028.jpg](../iclr_results/3344_ScienceAgentBench_ Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discover/tables/44d86d03e943307fc219ed04e355713b03c4caf789a4336e63d3dc0631ec6028.jpg)

![5444151755bde8905897f16f368530fcff45effe0e373b5fe72f482795fe8951.jpg](../iclr_results/3344_ScienceAgentBench_ Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discover/tables/5444151755bde8905897f16f368530fcff45effe0e373b5fe72f482795fe8951.jpg)

![5c5c1324ced0676ad012c0aae8b5dcb52931920d990b2b96037020bf2dd3c87f.jpg](../iclr_results/3344_ScienceAgentBench_ Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discover/tables/5c5c1324ced0676ad012c0aae8b5dcb52931920d990b2b96037020bf2dd3c87f.jpg)

![6412fceb5b62345991aaf058dfd70c589df088a86645328e7de4dd9c303b9343.jpg](../iclr_results/3344_ScienceAgentBench_ Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discover/tables/6412fceb5b62345991aaf058dfd70c589df088a86645328e7de4dd9c303b9343.jpg)

![64b1ae4ed17731241dd6be9a9c77dd209f4b72738253332db9bacd348696be98.jpg](../iclr_results/3344_ScienceAgentBench_ Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discover/tables/64b1ae4ed17731241dd6be9a9c77dd209f4b72738253332db9bacd348696be98.jpg)

![6ad467a89915bc8e3498e0f86d2003eca103c0524144c881665942f523916930.jpg](../iclr_results/3344_ScienceAgentBench_ Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discover/tables/6ad467a89915bc8e3498e0f86d2003eca103c0524144c881665942f523916930.jpg)

![6f78376911a68be48d55e8d641b46c36302a35ea08313238fc7606f3c7bbd78a.jpg](../iclr_results/3344_ScienceAgentBench_ Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discover/tables/6f78376911a68be48d55e8d641b46c36302a35ea08313238fc7606f3c7bbd78a.jpg)

![a60e561ccac5a16f3336f18cbeabefa99f0e5eb750ee61a558ec4067d9f50738.jpg](../iclr_results/3344_ScienceAgentBench_ Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discover/tables/a60e561ccac5a16f3336f18cbeabefa99f0e5eb750ee61a558ec4067d9f50738.jpg)

![c4cbb850e0accb4a53e5e2a2364d0b9eafa3dcb4a62b07d795fbcc2931004b45.jpg](../iclr_results/3344_ScienceAgentBench_ Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discover/tables/c4cbb850e0accb4a53e5e2a2364d0b9eafa3dcb4a62b07d795fbcc2931004b45.jpg)

![d32b18c66160e6a7fc9f9e09ecb4b013d3674b5f56eb8c4ec3c05d2155193ff2.jpg](../iclr_results/3344_ScienceAgentBench_ Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discover/tables/d32b18c66160e6a7fc9f9e09ecb4b013d3674b5f56eb8c4ec3c05d2155193ff2.jpg)

## Episodic Memories Generation and Evaluation Benchmark for Large Language Models


### Images

![0e132ec35cf5a3d2287fd3d51135ff0ccce0519e985022924a85511bbae534e3.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/images/0e132ec35cf5a3d2287fd3d51135ff0ccce0519e985022924a85511bbae534e3.jpg)

![4a5f034bdcf956e4a64dbf6b7dca43f5db5e14c7c849b461a0ffb40f95db886c.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/images/4a5f034bdcf956e4a64dbf6b7dca43f5db5e14c7c849b461a0ffb40f95db886c.jpg)

![5778c30853761f61993c90e58c79ce9a929ad07e95d5a48c20daa432f2db1078.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/images/5778c30853761f61993c90e58c79ce9a929ad07e95d5a48c20daa432f2db1078.jpg)

![82aa76c655a87fbf50b070d849dc59c1c940c7610773df603a640c701385afa6.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/images/82aa76c655a87fbf50b070d849dc59c1c940c7610773df603a640c701385afa6.jpg)

![abe93e7b0dced1551207ce0c813414f79b7a70337ab75b7f9d7277a82d6b749b.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/images/abe93e7b0dced1551207ce0c813414f79b7a70337ab75b7f9d7277a82d6b749b.jpg)

![c45d424cfd1c29b68ecd337e84c7592bb51bb1ad21b4684f020096f4e3ba9382.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/images/c45d424cfd1c29b68ecd337e84c7592bb51bb1ad21b4684f020096f4e3ba9382.jpg)

![eccc19e75cd50f6e15514c3ac2b9ebcd1cb4e91ed981f2e3a605efedded63cfc.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/images/eccc19e75cd50f6e15514c3ac2b9ebcd1cb4e91ed981f2e3a605efedded63cfc.jpg)

![f0ae70bb1ebc4c8f1cb487251cabe5cdfb5f4381b4691fe6fb09558560c8daea.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/images/f0ae70bb1ebc4c8f1cb487251cabe5cdfb5f4381b4691fe6fb09558560c8daea.jpg)

![f49a35803594e2e912793091d98fb68903ad65279800414d6f51343c310a4ed9.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/images/f49a35803594e2e912793091d98fb68903ad65279800414d6f51343c310a4ed9.jpg)

### Tables

![08d47fc9078672c29f878dfee9aea819be275d5016d570687083198a67d9f646.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/08d47fc9078672c29f878dfee9aea819be275d5016d570687083198a67d9f646.jpg)

![1072a38b4edca7d4cabd6c261bd498ade6a4dbd6d1fb0ee344eb07949b771da8.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/1072a38b4edca7d4cabd6c261bd498ade6a4dbd6d1fb0ee344eb07949b771da8.jpg)

![109af753871cd40a6bed18552b2bc5b03abdccbfcf3c19ea020c669e1c146d1d.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/109af753871cd40a6bed18552b2bc5b03abdccbfcf3c19ea020c669e1c146d1d.jpg)

![321720b6c5d421bd21f3459d4b99651658c8ebe58a3c1c95d8e309b202dc69ee.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/321720b6c5d421bd21f3459d4b99651658c8ebe58a3c1c95d8e309b202dc69ee.jpg)

![3a841fbf12dcce583322e23e1c6759721d2763dc03c5f3bf62452af240bbb017.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/3a841fbf12dcce583322e23e1c6759721d2763dc03c5f3bf62452af240bbb017.jpg)

![56af8a9baca06af8456cc0562d87c4256904904a48aaf314c2167cf7a7851980.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/56af8a9baca06af8456cc0562d87c4256904904a48aaf314c2167cf7a7851980.jpg)

![5eac574fed57814ba4ede5b97ddeeb757d3a2d403c7cd955ada083ee90833ec9.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/5eac574fed57814ba4ede5b97ddeeb757d3a2d403c7cd955ada083ee90833ec9.jpg)

![607feb6b3b8283198383e8cce28412748773e509be9b6e1aab637d64c572ef3b.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/607feb6b3b8283198383e8cce28412748773e509be9b6e1aab637d64c572ef3b.jpg)

![61a02b2632550cec919582829b9ad6241002f75d0bd762d52f3ff7939edd0886.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/61a02b2632550cec919582829b9ad6241002f75d0bd762d52f3ff7939edd0886.jpg)

![6b0ca8fab26069aa746fd7152ee2df5bce7dbc699a2f8949a3ec5d9a9bdcb269.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/6b0ca8fab26069aa746fd7152ee2df5bce7dbc699a2f8949a3ec5d9a9bdcb269.jpg)

![722ec33f2575bc3ec226d48d53e99b6292804bcd5c00d0e24a5ef4a16fb581e3.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/722ec33f2575bc3ec226d48d53e99b6292804bcd5c00d0e24a5ef4a16fb581e3.jpg)

![78ba85a1d7536f8b4b61bd02c6f54c33e18c2ec91e349627eecefc171b4a4c7e.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/78ba85a1d7536f8b4b61bd02c6f54c33e18c2ec91e349627eecefc171b4a4c7e.jpg)

![7ab159223299238ae87b91d2fe6659c5568a2be5d7a86e70ff152f2a72bf91f2.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/7ab159223299238ae87b91d2fe6659c5568a2be5d7a86e70ff152f2a72bf91f2.jpg)

![81c76c3dee0ce1247e61d78d45985b839a87763aaac6f4e6fb2e14d041ae9ead.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/81c76c3dee0ce1247e61d78d45985b839a87763aaac6f4e6fb2e14d041ae9ead.jpg)

![8c56de38862b241e317a0542b05beb7306c0f33690dc577be0d9e821ba3e89b9.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/8c56de38862b241e317a0542b05beb7306c0f33690dc577be0d9e821ba3e89b9.jpg)

![8db80ac5321d19c51ad5bf99c9259124308df34d8e7233d27eca1d6c449f7b0a.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/8db80ac5321d19c51ad5bf99c9259124308df34d8e7233d27eca1d6c449f7b0a.jpg)

![9169ca168967b897cf8f8e7a8fae9dbdb56967c47d4068da9c1c27f3caa07edd.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/9169ca168967b897cf8f8e7a8fae9dbdb56967c47d4068da9c1c27f3caa07edd.jpg)

![9969931f1aefc9b94a1d0e80a10cbfca06306d10b93e279df5cb89dc9e1eb1cf.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/9969931f1aefc9b94a1d0e80a10cbfca06306d10b93e279df5cb89dc9e1eb1cf.jpg)

![a193be56ef21dc4be678d191e6d04a29a98ca8f9b107d70aa335a86c095082d7.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/a193be56ef21dc4be678d191e6d04a29a98ca8f9b107d70aa335a86c095082d7.jpg)

![a28c8b3e7be0ede246a28f8db60405ec2599032ef22e89082c68adadb9a5b7ce.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/a28c8b3e7be0ede246a28f8db60405ec2599032ef22e89082c68adadb9a5b7ce.jpg)

![b0c45ea31dc48c55af8dc1365c520eb166f417c5fae2867857ee0ec6e138bbff.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/b0c45ea31dc48c55af8dc1365c520eb166f417c5fae2867857ee0ec6e138bbff.jpg)

![b61734cdada147e22879484a961acb71f6d0e04f4f74e1f8e0a887ff03fbcdc8.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/b61734cdada147e22879484a961acb71f6d0e04f4f74e1f8e0a887ff03fbcdc8.jpg)

![b77a8286209063d806a962de12eb0b3d16aa5603c4ba02a3a68af2af324c5470.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/b77a8286209063d806a962de12eb0b3d16aa5603c4ba02a3a68af2af324c5470.jpg)

![c32849af84da3e14ad910d0efe3e7bbfdbdc7a40a1d58119075c4d3081e4ce61.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/c32849af84da3e14ad910d0efe3e7bbfdbdc7a40a1d58119075c4d3081e4ce61.jpg)

![c856923be028c1801333bfee5e99e47ab745c7eb21e56995bc96526b2a7aa282.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/c856923be028c1801333bfee5e99e47ab745c7eb21e56995bc96526b2a7aa282.jpg)

![ce6704ebb5b76e72e9dfd32da2c4c87b7b3cea92aae559b148f106d7b6eeac6b.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/ce6704ebb5b76e72e9dfd32da2c4c87b7b3cea92aae559b148f106d7b6eeac6b.jpg)

![e0a46918b23bb2895ac3099bd097d0c3e88f92d056eca28c2b15c95204931f54.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/e0a46918b23bb2895ac3099bd097d0c3e88f92d056eca28c2b15c95204931f54.jpg)

![e39c430631ab83f7addefbcdb25b98ac38f6b7d8fd5f5b7faa132f9e37b81895.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/e39c430631ab83f7addefbcdb25b98ac38f6b7d8fd5f5b7faa132f9e37b81895.jpg)

![e5de100baa07ec3ae9f78c6a8b0e1ca315313b60677e9ddbeeda5d817afb7b35.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/e5de100baa07ec3ae9f78c6a8b0e1ca315313b60677e9ddbeeda5d817afb7b35.jpg)

![eaf992c8fb3d3743110153cb7e48af0cfec935ae0fdbd8dcb0e5d13bb14109fd.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/eaf992c8fb3d3743110153cb7e48af0cfec935ae0fdbd8dcb0e5d13bb14109fd.jpg)

![f6e6cfc771cf2cd53dc6f45e29805ce7c6906b42096902459f80ca538553d17c.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/f6e6cfc771cf2cd53dc6f45e29805ce7c6906b42096902459f80ca538553d17c.jpg)

![fff2f7aad44b312e3312aff1aebd47d4ff7e28a6cb664e2d296c95578c481550.jpg](../iclr_results/3345_Episodic Memories Generation and Evaluation Benchmark for Large Language Models/tables/fff2f7aad44b312e3312aff1aebd47d4ff7e28a6cb664e2d296c95578c481550.jpg)

## Towards Foundation Models for Mixed Integer Linear Programming

### Images

![1e468b37495ab4bcc6222a9cc6013ced863c22334a492e9c398118619267fe4b.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/images/1e468b37495ab4bcc6222a9cc6013ced863c22334a492e9c398118619267fe4b.jpg)

![3611569a63d8352cead87a5fecde132bdd4eab039a7cd9bbd0d98ecde277eb80.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/images/3611569a63d8352cead87a5fecde132bdd4eab039a7cd9bbd0d98ecde277eb80.jpg)

![3bbe6fd8092d8c07f3f7acdcb61f3b6d0714b07fb95b09667e7675b7cc476a1e.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/images/3bbe6fd8092d8c07f3f7acdcb61f3b6d0714b07fb95b09667e7675b7cc476a1e.jpg)

![42c6148eb424200611b681ce809d73e314b849b445d04db0c7444b52621f8d72.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/images/42c6148eb424200611b681ce809d73e314b849b445d04db0c7444b52621f8d72.jpg)

![447eb5e9954a9793e66a3a7dd5e51cd923f997555afe5ebba11e2908d36305c0.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/images/447eb5e9954a9793e66a3a7dd5e51cd923f997555afe5ebba11e2908d36305c0.jpg)

![5a42682e78c0b81f04b952a0ab605f0fc3b71a9eca860a7fcec23016ee38543a.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/images/5a42682e78c0b81f04b952a0ab605f0fc3b71a9eca860a7fcec23016ee38543a.jpg)

![69dd461af66ffa71c0c74b65db31d4baa24386425771d1dc90aaef7d4dd77df4.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/images/69dd461af66ffa71c0c74b65db31d4baa24386425771d1dc90aaef7d4dd77df4.jpg)

![88338dde163773b71c928e195f8bbed23fe15e62d7139358fd67fc9452057b97.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/images/88338dde163773b71c928e195f8bbed23fe15e62d7139358fd67fc9452057b97.jpg)

![a27a64c76f4398b1d46b856285dd890d39d5c67edb3995e05e8441c74af75aac.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/images/a27a64c76f4398b1d46b856285dd890d39d5c67edb3995e05e8441c74af75aac.jpg)

![a34308363e4061363aeed2f35b115ecce5ef77a661b4487b1d81a69a78c0e104.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/images/a34308363e4061363aeed2f35b115ecce5ef77a661b4487b1d81a69a78c0e104.jpg)

![abb14f50f295bbc569d960ce76a6981f774f42b240bbcdd8f9c421477455ccf3.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/images/abb14f50f295bbc569d960ce76a6981f774f42b240bbcdd8f9c421477455ccf3.jpg)

![bc9c036754fedab5524f2fded69af48d81cc4fa93c255372ef16b09e930e9f4d.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/images/bc9c036754fedab5524f2fded69af48d81cc4fa93c255372ef16b09e930e9f4d.jpg)

![dae3768d524090e2463d72738cf5bccb677e16d78944beb97f2251b45a1d83d4.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/images/dae3768d524090e2463d72738cf5bccb677e16d78944beb97f2251b45a1d83d4.jpg)

![dfd54e41a1432b7f3648cba43c1cff93fd067416fcefac4049a443eb10dea43b.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/images/dfd54e41a1432b7f3648cba43c1cff93fd067416fcefac4049a443eb10dea43b.jpg)

### Tables

![0751fff5feb17296f3367da4eb2a5b915207817b4b7556672eba69997451d9c2.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/tables/0751fff5feb17296f3367da4eb2a5b915207817b4b7556672eba69997451d9c2.jpg)

![07eaaeba3635dc55c78421f583cafafde25cf1ed268767c0a62e7f7ef5b0713d.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/tables/07eaaeba3635dc55c78421f583cafafde25cf1ed268767c0a62e7f7ef5b0713d.jpg)

![11ff824d5ce5c971f39dd85a759c89f1edd33fb349fb6c70ff61b9466a09f5dc.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/tables/11ff824d5ce5c971f39dd85a759c89f1edd33fb349fb6c70ff61b9466a09f5dc.jpg)

![3d3020015c6537800e9536e4cf55d9ea8632f73960aad42c615bf59539bcc3c1.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/tables/3d3020015c6537800e9536e4cf55d9ea8632f73960aad42c615bf59539bcc3c1.jpg)

![47c4c45597b4673d94450e76c9def848284720ef82bf0c0ce8129862a3ffd9c5.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/tables/47c4c45597b4673d94450e76c9def848284720ef82bf0c0ce8129862a3ffd9c5.jpg)

![562187da1671146176f1c0ba44055a467a2e695c66d2d5efa43350684ff9bf83.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/tables/562187da1671146176f1c0ba44055a467a2e695c66d2d5efa43350684ff9bf83.jpg)

![5c67f2cfeff5636ea779c4030d9dce63ffe83d7be02533a8575252db390ae72b.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/tables/5c67f2cfeff5636ea779c4030d9dce63ffe83d7be02533a8575252db390ae72b.jpg)

![64d723148033ad43f08e2908cb0e80b1b9a8370b51a7b28d9349eb930d9d9106.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/tables/64d723148033ad43f08e2908cb0e80b1b9a8370b51a7b28d9349eb930d9d9106.jpg)

![8a43e3f17275977f683d21e25e8633c7e4ecffcbf52041a33e2242a9ab1f2f21.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/tables/8a43e3f17275977f683d21e25e8633c7e4ecffcbf52041a33e2242a9ab1f2f21.jpg)

![8d1109f0459fc32b091fefaa1131d76f05e9441f1bf7d259930c5dd80ff532bd.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/tables/8d1109f0459fc32b091fefaa1131d76f05e9441f1bf7d259930c5dd80ff532bd.jpg)

![9094960714839b32cd73591fe7748a916bd11ad99cbdd3362be2d4ea66695039.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/tables/9094960714839b32cd73591fe7748a916bd11ad99cbdd3362be2d4ea66695039.jpg)

![a3a65913c40c46e710dc8085be24315a9630dabf5eadabd3dfb81b8f4ed6ba63.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/tables/a3a65913c40c46e710dc8085be24315a9630dabf5eadabd3dfb81b8f4ed6ba63.jpg)

![aea4feaa14e2089686deb4cc76b9ad1546fb4f197147f16ca2db1421b225053f.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/tables/aea4feaa14e2089686deb4cc76b9ad1546fb4f197147f16ca2db1421b225053f.jpg)

![b80ff457e101c607652a4276bbbb403a3723401ec71f5d9e7ebcae98dc1623c6.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/tables/b80ff457e101c607652a4276bbbb403a3723401ec71f5d9e7ebcae98dc1623c6.jpg)

![d364b0b09d5a3076b18d9bd9aab92c4b0577d99bba07a2c8b34cef0148da701e.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/tables/d364b0b09d5a3076b18d9bd9aab92c4b0577d99bba07a2c8b34cef0148da701e.jpg)

![facd0eac2046919d60c7f19c7a85a5cfd4791f1d2e89b3204d0efd29dcbd921b.jpg](../iclr_results/3346_Towards Foundation Models for Mixed Integer Linear Programming/tables/facd0eac2046919d60c7f19c7a85a5cfd4791f1d2e89b3204d0efd29dcbd921b.jpg)
