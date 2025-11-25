# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 77 of 100

## 目录 (Table of Contents)

1. [MuseGNN: Forming Scalable, Convergent GNN Layers that Minimize a Sampling-Based Energy](#MuseGNN-Forming-Scalable-Convergent-GNN-Layers-that-Minimize-a-Sampling-Based-Energy)
2. [The Crystal Ball Hypothesis in diffusion models: Anticipating object positions from initial noise](#The-Crystal-Ball-Hypothesis-in-diffusion-models-Anticipating-object-positions-from-initial-noise)
3. [Towards counterfactual fairness through auxiliary variables](#Towards-counterfactual-fairness-through-auxiliary-variables)
4. [Geometry Image Diffusion: Fast and Data-Efficient Text-to-3D with Image-Based Surface Representation](#Geometry-Image-Diffusion-Fast-and-Data-Efficient-Text-to-3D-with-Image-Based-Surface-Representation)
5. [QPM: Discrete Optimization for Globally Interpretable Image Classification](#QPM-Discrete-Optimization-for-Globally-Interpretable-Image-Classification)
6. [Multi-agent cooperation through learning-aware policy gradients](#Multi-agent-cooperation-through-learning-aware-policy-gradients)
7. [StructRAG: Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Structurization](#StructRAG-Boosting-Knowledge-Intensive-Reasoning-of-LLMs-via-Inference-time-Hybrid-Information-Structurization)
8. [Unhackable Temporal Reward for Scalable Video MLLMs](#Unhackable-Temporal-Reward-for-Scalable-Video-MLLMs)
9. [Taming Transformer Without Using Learning Rate Warmup](#Taming-Transformer-Without-Using-Learning-Rate-Warmup)
10. [How Learnable Grids Recover Fine Detail in Low Dimensions: A Neural Tangent Kernel Analysis of Multigrid Parametric Encodings](#How-Learnable-Grids-Recover-Fine-Detail-in-Low-Dimensions-A-Neural-Tangent-Kernel-Analysis-of-Multigrid-Parametric-Encodings)
11. [NeurFlow: Interpreting Neural Networks through Neuron Groups and Functional Interactions](#NeurFlow-Interpreting-Neural-Networks-through-Neuron-Groups-and-Functional-Interactions)
12. [RaSA: Rank-Sharing Low-Rank Adaptation](#RaSA-Rank-Sharing-Low-Rank-Adaptation)
13. [Rethinking Diffusion Posterior Sampling: From Conditional Score Estimator to Maximizing a Posterior](#Rethinking-Diffusion-Posterior-Sampling-From-Conditional-Score-Estimator-to-Maximizing-a-Posterior)
14. [Adversarial Mixup Unlearning](#Adversarial-Mixup-Unlearning)
15. [SMT: Fine-Tuning Large Language Models with Sparse Matrices](#SMT-Fine-Tuning-Large-Language-Models-with-Sparse-Matrices)
16. [3D-AffordanceLLM: Harnessing Large Language Models for Open-Vocabulary Affordance Detection in 3D Worlds](#3D-AffordanceLLM-Harnessing-Large-Language-Models-for-Open-Vocabulary-Affordance-Detection-in-3D-Worlds)
17. [Mutual Effort for Efficiency: A Similarity-based Token Pruning for Vision Transformers in Self-Supervised Learning](#Mutual-Effort-for-Efficiency-A-Similarity-based-Token-Pruning-for-Vision-Transformers-in-Self-Supervised-Learning)
18. [Bridging Compressed Image Latents and Multimodal Large Language Models](#Bridging-Compressed-Image-Latents-and-Multimodal-Large-Language-Models)
19. [MAPS: Advancing Multi-Modal Reasoning in Expert-Level Physical Science](#MAPS-Advancing-Multi-Modal-Reasoning-in-Expert-Level-Physical-Science)
20. [Weakly Supervised Video Scene Graph Generation via Natural Language Supervision](#Weakly-Supervised-Video-Scene-Graph-Generation-via-Natural-Language-Supervision)
21. [Optimizing $(L_0, L_1)$-Smooth Functions by Gradient Methods](#Optimizing-L_0-L_1-Smooth-Functions-by-Gradient-Methods)
22. [Interpreting the Second-Order Effects of Neurons in CLIP](#Interpreting-the-Second-Order-Effects-of-Neurons-in-CLIP)
23. [SaLoRA: Safety-Alignment Preserved Low-Rank Adaptation](#SaLoRA-Safety-Alignment-Preserved-Low-Rank-Adaptation)
24. [Rethinking Neural Multi-Objective Combinatorial Optimization via Neat Weight Embedding](#Rethinking-Neural-Multi-Objective-Combinatorial-Optimization-via-Neat-Weight-Embedding)
25. [Skill Expansion and Composition in Parameter Space](#Skill-Expansion-and-Composition-in-Parameter-Space)
26. [ADMM for Nonconvex Optimization under Minimal Continuity Assumption](#ADMM-for-Nonconvex-Optimization-under-Minimal-Continuity-Assumption)
27. [TFG-Flow: Training-free Guidance in Multimodal Generative Flow](#TFG-Flow-Training-free-Guidance-in-Multimodal-Generative-Flow)
28. [Meissonic: Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image Synthesis](#Meissonic-Revitalizing-Masked-Generative-Transformers-for-Efficient-High-Resolution-Text-to-Image-Synthesis)
29. [Efficient Active Imitation Learning with Random Network Distillation](#Efficient-Active-Imitation-Learning-with-Random-Network-Distillation)
30. [Rational Decision-Making Agent with Learning Internal Utility Judgment](#Rational-Decision-Making-Agent-with-Learning-Internal-Utility-Judgment)
31. [OVTR: End-to-End Open-Vocabulary Multiple Object Tracking with Transformer](#OVTR-End-to-End-Open-Vocabulary-Multiple-Object-Tracking-with-Transformer)
32. [Neural Functions for Learning Periodic Signal](#Neural-Functions-for-Learning-Periodic-Signal)
33. [Ferret-UI 2: Mastering Universal User Interface Understanding Across Platforms](#Ferret-UI-2-Mastering-Universal-User-Interface-Understanding-Across-Platforms)
34. [ExACT: Teaching AI Agents to Explore with Reflective-MCTS and Exploratory Learning](#ExACT-Teaching-AI-Agents-to-Explore-with-Reflective-MCTS-and-Exploratory-Learning)
35. [Singular Subspace Perturbation Bounds via Rectangular Random Matrix Diffusions](#Singular-Subspace-Perturbation-Bounds-via-Rectangular-Random-Matrix-Diffusions)
36. [Self-Normalized Resets for Plasticity in Continual Learning](#Self-Normalized-Resets-for-Plasticity-in-Continual-Learning)
37. [Implicit In-context Learning](#Implicit-In-context-Learning)

---


## MuseGNN: Forming Scalable, Convergent GNN Layers that Minimize a Sampling-Based Energy

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

## QPM: Discrete Optimization for Globally Interpretable Image Classification


### Images

![31edf6f0df8fa588e5d46a71f6247a6e436eb4dd309910f302d583a3266dfac5.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/31edf6f0df8fa588e5d46a71f6247a6e436eb4dd309910f302d583a3266dfac5.jpg)

![3525ec98131c12a6780fd17af43d10998733786b89247b6887b1bb58795f7d20.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/3525ec98131c12a6780fd17af43d10998733786b89247b6887b1bb58795f7d20.jpg)

![365c7f095a0fe2e60782425f8cd8d94d790d3e12d8544266df34f798c709905c.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/365c7f095a0fe2e60782425f8cd8d94d790d3e12d8544266df34f798c709905c.jpg)

![3684525661fc46e31d193b436a965e985e613da1614994a64378d1ca7d7c75a7.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/3684525661fc46e31d193b436a965e985e613da1614994a64378d1ca7d7c75a7.jpg)

![372d863f8b09ca35e3b614041aa4d887ae78bddf13d4515976a11f1fe2777551.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/372d863f8b09ca35e3b614041aa4d887ae78bddf13d4515976a11f1fe2777551.jpg)

![3b08b4c2074af89d4b28f6089807fc7a147a4add58b11f6d66d44dcaa62db91a.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/3b08b4c2074af89d4b28f6089807fc7a147a4add58b11f6d66d44dcaa62db91a.jpg)

![3de9ef72397d5a97868eca7134c50ff269f28536773184848aa563723a26da45.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/3de9ef72397d5a97868eca7134c50ff269f28536773184848aa563723a26da45.jpg)

![413ce366342c917fe8e6b6f5c15dc5f02a93e0ba70eafe62f67c60f611ba3648.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/413ce366342c917fe8e6b6f5c15dc5f02a93e0ba70eafe62f67c60f611ba3648.jpg)

![4160f1da531d718078aa0a60ab47488ab4dc351a7568c3c78b58241f5c4265f1.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/4160f1da531d718078aa0a60ab47488ab4dc351a7568c3c78b58241f5c4265f1.jpg)

![50b6b580e32282dcb59875c17eef36983510ee18d00a09439b7c22868edd486d.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/50b6b580e32282dcb59875c17eef36983510ee18d00a09439b7c22868edd486d.jpg)

![5314886ad65cd46a93e7304485d65b15fc40bce7a68e3624847bf7eee99fcb2e.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/5314886ad65cd46a93e7304485d65b15fc40bce7a68e3624847bf7eee99fcb2e.jpg)

![6a3100b73abf7049b12099863ca8158e25935866e024ecd615dcd60128e27d91.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/6a3100b73abf7049b12099863ca8158e25935866e024ecd615dcd60128e27d91.jpg)

![71f015a67e69d99660ca8bd2c0a63105b5e2148b903a79360f6e41299ff51431.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/71f015a67e69d99660ca8bd2c0a63105b5e2148b903a79360f6e41299ff51431.jpg)

![75a63e8f00e755dd930e6a24b65a6a541580c9f233bf8b956d596b8c8938413e.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/75a63e8f00e755dd930e6a24b65a6a541580c9f233bf8b956d596b8c8938413e.jpg)

![836f832df24a2afb7ef7147976a18d7432cea3854ff41a39e56d0cfa25c6599a.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/836f832df24a2afb7ef7147976a18d7432cea3854ff41a39e56d0cfa25c6599a.jpg)

![88a0d79d28ff02c2bca01db558faa600f4818560c47d714c8afccb37988d7e06.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/88a0d79d28ff02c2bca01db558faa600f4818560c47d714c8afccb37988d7e06.jpg)

![930ddddcbc55ecfce983c4bc081a78267a5eff16e8763e55424763bab4bbc06c.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/930ddddcbc55ecfce983c4bc081a78267a5eff16e8763e55424763bab4bbc06c.jpg)

![94eeef9bee1356999c95d5be2971388415cd5c3cb0ffad50976c4055d00c1891.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/94eeef9bee1356999c95d5be2971388415cd5c3cb0ffad50976c4055d00c1891.jpg)

![b09eef977883b86e3e8f4c0bef68a0c47e87d8d9caff6694b91b083e0dc4ae01.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/b09eef977883b86e3e8f4c0bef68a0c47e87d8d9caff6694b91b083e0dc4ae01.jpg)

![be1e9915331c1e9336e24a03b466beae137560a706a67ccff82e1a946d790ce1.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/be1e9915331c1e9336e24a03b466beae137560a706a67ccff82e1a946d790ce1.jpg)

![c3e5c321129a81c5227537edebd530c2fffb2367bdab215af50c7f43270292d4.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/c3e5c321129a81c5227537edebd530c2fffb2367bdab215af50c7f43270292d4.jpg)

![c6232c87053a6f001a6593c04ee79d725b7a5c11e67221bc50595632cc20e5b3.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/c6232c87053a6f001a6593c04ee79d725b7a5c11e67221bc50595632cc20e5b3.jpg)

![ce0d74396fd1d69dfd699b367e2fdd6d634c86f6adac283195c94ca72a41453a.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/ce0d74396fd1d69dfd699b367e2fdd6d634c86f6adac283195c94ca72a41453a.jpg)

![cf2b497fda749e2ed3faad4cf6707fbc353e4284b62981b2a7321c0d7d2e066b.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/cf2b497fda749e2ed3faad4cf6707fbc353e4284b62981b2a7321c0d7d2e066b.jpg)

![d7105e7128694e82f1e5f62fcd61c37838e9094e6e5ce5f381fb1b800c4fa983.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/images/d7105e7128694e82f1e5f62fcd61c37838e9094e6e5ce5f381fb1b800c4fa983.jpg)

### Tables

![05b5296266187125d3b0740e4195208f4e9ff05e0bc579d74e815d0cdf5f12b1.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/tables/05b5296266187125d3b0740e4195208f4e9ff05e0bc579d74e815d0cdf5f12b1.jpg)

![06522644aee682b9cc742e2977df2bf791cf8ad9f9d415635532f422e3a09c73.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/tables/06522644aee682b9cc742e2977df2bf791cf8ad9f9d415635532f422e3a09c73.jpg)

![238389561c6452196d28c20e57206e07a1c852974c51cb0d48f1b285cb56a3cf.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/tables/238389561c6452196d28c20e57206e07a1c852974c51cb0d48f1b285cb56a3cf.jpg)

![33c0ff0f6128a4ded64a674bc2d5fa36f121124501a7cf13ac28f59301db778e.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/tables/33c0ff0f6128a4ded64a674bc2d5fa36f121124501a7cf13ac28f59301db778e.jpg)

![3589ffb82253613b35ac6909d79a00785bb7c868eaea89bf3bbe96ed2b9953ff.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/tables/3589ffb82253613b35ac6909d79a00785bb7c868eaea89bf3bbe96ed2b9953ff.jpg)

![37266e826404c6130f700f3bdd28c3ef2979cbd660d1bdf0d1e9b39e532fc2cc.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/tables/37266e826404c6130f700f3bdd28c3ef2979cbd660d1bdf0d1e9b39e532fc2cc.jpg)

![37c8991147ec7f43af9a789eea658e112f010c2a9d89c99b2a7a7aaa016e0928.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/tables/37c8991147ec7f43af9a789eea658e112f010c2a9d89c99b2a7a7aaa016e0928.jpg)

![3a919b8ccce8abcf1b243febb873f5fd2ce0996b21dae5ca323f496a5af87062.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/tables/3a919b8ccce8abcf1b243febb873f5fd2ce0996b21dae5ca323f496a5af87062.jpg)

![43467a1a102cfe04853a03bc0fc8122c6e60b6ce9732bf3d1d3f78b40b1944cf.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/tables/43467a1a102cfe04853a03bc0fc8122c6e60b6ce9732bf3d1d3f78b40b1944cf.jpg)

![44cec587eff90fa7002684547e27368361eb626287e6ee289ff53f73fb35b652.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/tables/44cec587eff90fa7002684547e27368361eb626287e6ee289ff53f73fb35b652.jpg)

![522aa5a4e09bbc713082f1e48cf1f963d7364a2577ddca4e55239957dd5ab8ea.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/tables/522aa5a4e09bbc713082f1e48cf1f963d7364a2577ddca4e55239957dd5ab8ea.jpg)

![60b96898c9416888b986af7906a54f2a64a45850a89d5db04a630898c5b79756.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/tables/60b96898c9416888b986af7906a54f2a64a45850a89d5db04a630898c5b79756.jpg)

![78e9726d39051b1a8affc236b16ed4086f564bbc58a6b15d9b00022e4ad0119a.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/tables/78e9726d39051b1a8affc236b16ed4086f564bbc58a6b15d9b00022e4ad0119a.jpg)

![7bdd197535e6c965abbb78be74b9b323b241282420e7322124983904d376127e.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/tables/7bdd197535e6c965abbb78be74b9b323b241282420e7322124983904d376127e.jpg)

![a0a5b2168d56b8850968b77f460a5cbbaf664396eccf58f2bbcdfe76b9f64387.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/tables/a0a5b2168d56b8850968b77f460a5cbbaf664396eccf58f2bbcdfe76b9f64387.jpg)

![c014ee4c18eadbbceb69f5b1195bdd18bb114737e4585cb9a09d4d50a410f1f5.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/tables/c014ee4c18eadbbceb69f5b1195bdd18bb114737e4585cb9a09d4d50a410f1f5.jpg)

![c6d71b5b7c734c43a14587025b989f13b37cf5813c395537af3b48325bf27e23.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/tables/c6d71b5b7c734c43a14587025b989f13b37cf5813c395537af3b48325bf27e23.jpg)

![cc00a54210454e6dcde4a8a6c2eeb232561a3d0c3f9704940982914ebf923ab0.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/tables/cc00a54210454e6dcde4a8a6c2eeb232561a3d0c3f9704940982914ebf923ab0.jpg)

![ec2c4c9db44378d0249192d17a2ead1211e5c53f8edd13275a5aff628586e6ef.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/tables/ec2c4c9db44378d0249192d17a2ead1211e5c53f8edd13275a5aff628586e6ef.jpg)

![f19ec9a7c3bfde10efa9e493adedc3a06553bd044f39fef066f9f12552a6b31c.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/tables/f19ec9a7c3bfde10efa9e493adedc3a06553bd044f39fef066f9f12552a6b31c.jpg)

![fa245f60067edc15c677b22dabff59907257995d1afefa7c84f9d327e7f12d3e.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/tables/fa245f60067edc15c677b22dabff59907257995d1afefa7c84f9d327e7f12d3e.jpg)

![fbbb9f4ef2b18b10bf07c321751b5b9c87d2237a0da636deefbb41580f50e81c.jpg](../iclr_results/2830_QPM_ Discrete Optimization for Globally Interpretable Image Classification/tables/fbbb9f4ef2b18b10bf07c321751b5b9c87d2237a0da636deefbb41580f50e81c.jpg)

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

## Meissonic: Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image Synthesis


### Images

![0065c7d5635598c68490996dbc5dc02b3ffc76a8c2a972deadfdea44fef4620f.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/0065c7d5635598c68490996dbc5dc02b3ffc76a8c2a972deadfdea44fef4620f.jpg)

![013195718e3b22b04b8e9f5978a653f2096026ff330c53c7d72e29f2a9351ac9.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/013195718e3b22b04b8e9f5978a653f2096026ff330c53c7d72e29f2a9351ac9.jpg)

![02ba8d77a7bbe4b1c0f50f818f769c6588be6eee2193fab227a1235c7d089fc4.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/02ba8d77a7bbe4b1c0f50f818f769c6588be6eee2193fab227a1235c7d089fc4.jpg)

![0478c5809a095f0b39ec024d7761b19b717530368be9f87a1bdd3b162eb7b29b.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/0478c5809a095f0b39ec024d7761b19b717530368be9f87a1bdd3b162eb7b29b.jpg)

![06088bc6d093e8d377ba6aeba6285815f909cbdee0b4d7e96f686b79aed3b51d.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/06088bc6d093e8d377ba6aeba6285815f909cbdee0b4d7e96f686b79aed3b51d.jpg)

![066651a197129811519d842ed03d84489d1bc235d5c13972c352c01370f4d1e5.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/066651a197129811519d842ed03d84489d1bc235d5c13972c352c01370f4d1e5.jpg)

![08543ca935d1781d2c826c9ebf67cbd32b44dc7e720e13929adce5695b7cd294.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/08543ca935d1781d2c826c9ebf67cbd32b44dc7e720e13929adce5695b7cd294.jpg)

![09af8f72757106254bdc7b829939d0294328ac6fcc0c92a616ecf8058c10e811.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/09af8f72757106254bdc7b829939d0294328ac6fcc0c92a616ecf8058c10e811.jpg)

![0a63bc6414d609fea06d6faef7892d997aaee31c2e60b71c26644396ca0b59bb.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/0a63bc6414d609fea06d6faef7892d997aaee31c2e60b71c26644396ca0b59bb.jpg)

![0c9209e6e93c7ef26850932e0c77db01a1cdc0137c363207f484a7a21e8b714d.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/0c9209e6e93c7ef26850932e0c77db01a1cdc0137c363207f484a7a21e8b714d.jpg)

![0d4a7f94c9d88ac22c4cec247c0b09767fae8963ed80a444334cb73c21fe85c9.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/0d4a7f94c9d88ac22c4cec247c0b09767fae8963ed80a444334cb73c21fe85c9.jpg)

![0ef7c8cca597fc163102095372ddaf1ef954356af8bc38e3fa585a94a1c042be.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/0ef7c8cca597fc163102095372ddaf1ef954356af8bc38e3fa585a94a1c042be.jpg)

![0f1136224ed484c38df8fca3bb4e4091ba367593ea5810c1e500bcb31db6ee42.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/0f1136224ed484c38df8fca3bb4e4091ba367593ea5810c1e500bcb31db6ee42.jpg)

![12479af4e8653838fbe6256fcae1d266d1cb16b06cb3f87ad273f04dbdf4c3b6.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/12479af4e8653838fbe6256fcae1d266d1cb16b06cb3f87ad273f04dbdf4c3b6.jpg)

![12e109cccf957596a39a1265d92d43eb00ac5a12e01722fd7674c07c2774caa7.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/12e109cccf957596a39a1265d92d43eb00ac5a12e01722fd7674c07c2774caa7.jpg)

![131e8647ef502b7815c8bca91bd585c7972668ac9898f804286b9a57c2479826.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/131e8647ef502b7815c8bca91bd585c7972668ac9898f804286b9a57c2479826.jpg)

![13397803ebe2216a73f738fb158b4901b74c6026112321365fd2d1357eed370b.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/13397803ebe2216a73f738fb158b4901b74c6026112321365fd2d1357eed370b.jpg)

![13c0a7f860e2bc445247e3aeb5edd57432d51ad0888705708057d0abd8100fd6.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/13c0a7f860e2bc445247e3aeb5edd57432d51ad0888705708057d0abd8100fd6.jpg)

![15d427302787d7ba26ce66b656c0666cc1e602aedfc3a89778406894f152f140.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/15d427302787d7ba26ce66b656c0666cc1e602aedfc3a89778406894f152f140.jpg)

![16c3a347be68d0c514f790076b2a2d21ee1d95ce3ed7636d06fffec5a77aff4e.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/16c3a347be68d0c514f790076b2a2d21ee1d95ce3ed7636d06fffec5a77aff4e.jpg)

![1793aa69fdb927ff43a7b73e3a5398430cb042779985b656973f55e78bbffbea.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/1793aa69fdb927ff43a7b73e3a5398430cb042779985b656973f55e78bbffbea.jpg)

![17b13537994848f490933fef5730011704c1d10382440e336a280fca2be623b4.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/17b13537994848f490933fef5730011704c1d10382440e336a280fca2be623b4.jpg)

![17fe04992e9569e16e79915474a1ccc5440ed01792925c05b8ead790dcd62d3c.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/17fe04992e9569e16e79915474a1ccc5440ed01792925c05b8ead790dcd62d3c.jpg)

![18f6cbd6dd5b56e705d2c8473bcf37be0ed7c8406fbbbf12bd7d45c07b11ed24.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/18f6cbd6dd5b56e705d2c8473bcf37be0ed7c8406fbbbf12bd7d45c07b11ed24.jpg)

![1a476b940962036301f12d9873d30ffcf681990a81977f7969d587d19694bef2.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/1a476b940962036301f12d9873d30ffcf681990a81977f7969d587d19694bef2.jpg)

![1a5c975290750bb46e3ab20e0bc5ea50b19a8f3cb7d94701332f2d42de77662d.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/1a5c975290750bb46e3ab20e0bc5ea50b19a8f3cb7d94701332f2d42de77662d.jpg)

![1af0c6025b702ef1fa634d1d731de77fa7e853d1b0840c5f36cfe1b26bc590e5.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/1af0c6025b702ef1fa634d1d731de77fa7e853d1b0840c5f36cfe1b26bc590e5.jpg)

![1bf702deab1211d15398bbe61d70873fecf63ccbbce7974b65a8d8e99dbe8d10.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/1bf702deab1211d15398bbe61d70873fecf63ccbbce7974b65a8d8e99dbe8d10.jpg)

![1fe2a33002bdb1320f78681ef5fbe2f5627b21108f3bb0a7a279ef970bba3aae.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/1fe2a33002bdb1320f78681ef5fbe2f5627b21108f3bb0a7a279ef970bba3aae.jpg)

![20859ba5d5c93096b73e46c1a9545fbde73cdbab769fd855afff3425ee0508f1.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/20859ba5d5c93096b73e46c1a9545fbde73cdbab769fd855afff3425ee0508f1.jpg)

![20b0740bff1b888873dec1d5ed71a1941a161a3bcd0fab65f1114be13a3786b3.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/20b0740bff1b888873dec1d5ed71a1941a161a3bcd0fab65f1114be13a3786b3.jpg)

![20c183645b925e6473eef62fdfb0ad21e006bf1c22ea426ed95565a5eb9f0c87.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/20c183645b925e6473eef62fdfb0ad21e006bf1c22ea426ed95565a5eb9f0c87.jpg)

![233d97fa137fbee72493b3dd9e94263a52a22e8d85615f2b6c92eabdc9de5aa3.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/233d97fa137fbee72493b3dd9e94263a52a22e8d85615f2b6c92eabdc9de5aa3.jpg)

![23a106fc286d52f12b85026185c82f3247df5ab983a12cb02f43799a16650308.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/23a106fc286d52f12b85026185c82f3247df5ab983a12cb02f43799a16650308.jpg)

![28ba4729a5362f6a55b0c2b4be3abe0e42dede89463b7018293974173c9889c9.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/28ba4729a5362f6a55b0c2b4be3abe0e42dede89463b7018293974173c9889c9.jpg)

![2bddbe22597aee1f2cb6d5519b2510b6e7dd3ead82092ed8ba035e1ce39bf457.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/2bddbe22597aee1f2cb6d5519b2510b6e7dd3ead82092ed8ba035e1ce39bf457.jpg)

![2c2fee23a3897a1ae133cc8b2f5277fcdec019e15c010f822de3fca521edd4ed.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/2c2fee23a3897a1ae133cc8b2f5277fcdec019e15c010f822de3fca521edd4ed.jpg)

![2c7a324ed507a6ec42a535e03f94fc359b4578095220eab3e601cd5d4316eb5d.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/2c7a324ed507a6ec42a535e03f94fc359b4578095220eab3e601cd5d4316eb5d.jpg)

![2e25e3936de2335bbf8900ee907f51d38671bd2120809eda532403b7f41db6ce.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/2e25e3936de2335bbf8900ee907f51d38671bd2120809eda532403b7f41db6ce.jpg)

![2e5cc78218247296b6664b2b295b028235c9aec8e3a61c0455cf6987f6241da7.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/2e5cc78218247296b6664b2b295b028235c9aec8e3a61c0455cf6987f6241da7.jpg)

![31005996c95ef2ddd6b36004224b6e16cada1fdb38df523cc0e1efcecf5a71c8.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/31005996c95ef2ddd6b36004224b6e16cada1fdb38df523cc0e1efcecf5a71c8.jpg)

![311bdc4133ad351fcfb7ff05809ff84ec6f6f281801834ee2a3751842bffc346.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/311bdc4133ad351fcfb7ff05809ff84ec6f6f281801834ee2a3751842bffc346.jpg)

![338f97e006f6770a5d6d81dfcbde66bd81e2cae481db4c7af0cc213e211383e2.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/338f97e006f6770a5d6d81dfcbde66bd81e2cae481db4c7af0cc213e211383e2.jpg)

![34dc29f729c8e4e384272f0ea546839265cc9ae3b3e43603e65c7ecae5198cba.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/34dc29f729c8e4e384272f0ea546839265cc9ae3b3e43603e65c7ecae5198cba.jpg)

![35dba654067c65ccc5ff63aeb73398895857b88a1eaab94153fa17eb0b933983.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/35dba654067c65ccc5ff63aeb73398895857b88a1eaab94153fa17eb0b933983.jpg)

![3645818d029236110706f27514e2d655a00be715f3a06bcf972c21c9e211b026.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/3645818d029236110706f27514e2d655a00be715f3a06bcf972c21c9e211b026.jpg)

![364c48b83360681f69629b90070f8be1c7fdb621f138daa7260687cbd0e2a8ca.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/364c48b83360681f69629b90070f8be1c7fdb621f138daa7260687cbd0e2a8ca.jpg)

![369c98b1d4230e1e588a13b9f2640c585185acadcbc47d5cf366243f11a64b2f.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/369c98b1d4230e1e588a13b9f2640c585185acadcbc47d5cf366243f11a64b2f.jpg)

![37a3986476a5290acc41b4135abc88c601f81f410377eaf43c85bed283b41ac6.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/37a3986476a5290acc41b4135abc88c601f81f410377eaf43c85bed283b41ac6.jpg)

![37ed848d622e7651f2596cdc88ac8dc513e6894978597c8719d317b78e743396.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/37ed848d622e7651f2596cdc88ac8dc513e6894978597c8719d317b78e743396.jpg)

![38333a9556fb8d1e62e26d714778ab78d7efd45f09df6c4bf1c4efc18f0fb353.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/38333a9556fb8d1e62e26d714778ab78d7efd45f09df6c4bf1c4efc18f0fb353.jpg)

![3a30252073fa8e89cbaa7174a0517a82ea5c06ad1e6f6fd962045c107fb97902.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/3a30252073fa8e89cbaa7174a0517a82ea5c06ad1e6f6fd962045c107fb97902.jpg)

![3aad226e3142a11e1a0e699babe16ff70a8de0ca175e4065666d9c706a35d993.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/3aad226e3142a11e1a0e699babe16ff70a8de0ca175e4065666d9c706a35d993.jpg)

![3d7b6eeac64e1d7601c23652751dc685affa6ba10d4f8354b60644da3810e24f.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/3d7b6eeac64e1d7601c23652751dc685affa6ba10d4f8354b60644da3810e24f.jpg)

![3eb03f71ca1f4a9bb698d50a7508f031f2759a270e7c93c3d3621b402a743578.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/3eb03f71ca1f4a9bb698d50a7508f031f2759a270e7c93c3d3621b402a743578.jpg)

![3ef96f4f92030975d9e22f17e76c4c760db38d27c7eb6ab4026066856e9dc6c6.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/3ef96f4f92030975d9e22f17e76c4c760db38d27c7eb6ab4026066856e9dc6c6.jpg)

![3f6097f53c21b98a39d628baf7eafbd4d1169aa8ce36a82654b8893e4890ccbf.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/3f6097f53c21b98a39d628baf7eafbd4d1169aa8ce36a82654b8893e4890ccbf.jpg)

![4005e8799c7b57accaf77cade9b63e960a35530e31bce6ffbfaf3107b3b4fe8e.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/4005e8799c7b57accaf77cade9b63e960a35530e31bce6ffbfaf3107b3b4fe8e.jpg)

![40e4f068eb0d9ae4f5fd2810e6fba060948821fe30ebc6b5a2fdeb5aaddbf989.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/40e4f068eb0d9ae4f5fd2810e6fba060948821fe30ebc6b5a2fdeb5aaddbf989.jpg)

![40f1361a5d5fc13e67e886f531331e48dfcf350a6c5e142a594783069234165b.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/40f1361a5d5fc13e67e886f531331e48dfcf350a6c5e142a594783069234165b.jpg)

![40fb6414bdcb0aacf7270398d71036155faecbea72a77e26ae077e6763b47ffa.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/40fb6414bdcb0aacf7270398d71036155faecbea72a77e26ae077e6763b47ffa.jpg)

![4246bf57e31ea3c3565e43942998711e0c19c6cdc7792b605afad24ce62fac51.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/4246bf57e31ea3c3565e43942998711e0c19c6cdc7792b605afad24ce62fac51.jpg)

![429efdd4497199c2cfaaaf9dac3c7052b4289c1871fa3477002b6795b340a138.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/429efdd4497199c2cfaaaf9dac3c7052b4289c1871fa3477002b6795b340a138.jpg)

![436f23f049756e073cb1be97ad854922cd19ceddb57b8699f7daab1512eb7119.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/436f23f049756e073cb1be97ad854922cd19ceddb57b8699f7daab1512eb7119.jpg)

![4428c7db398a9eec866217e976c197ed44b6058b80ed765641a521d180536664.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/4428c7db398a9eec866217e976c197ed44b6058b80ed765641a521d180536664.jpg)

![46622300470ba38afc65f4a4a6e2e3a08a3b8887b70d4bed043e227e2a6169d8.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/46622300470ba38afc65f4a4a6e2e3a08a3b8887b70d4bed043e227e2a6169d8.jpg)

![46fb4fb28ae0a77483381ee3dab6ba3bc99a30296c45e01716b498f9c5a1cc15.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/46fb4fb28ae0a77483381ee3dab6ba3bc99a30296c45e01716b498f9c5a1cc15.jpg)

![4ae68d5d1d51734b431219b59fa136f056c501a40e1cfd53531c297d74538704.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/4ae68d5d1d51734b431219b59fa136f056c501a40e1cfd53531c297d74538704.jpg)

![4b71ae988d57a1107e90798133936685b7e1424e27c660ed6df7cdb110d08db3.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/4b71ae988d57a1107e90798133936685b7e1424e27c660ed6df7cdb110d08db3.jpg)

![4fb5668870bb4ea223e8326707185669ffe95109067dbac5d62007f730b4bd52.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/4fb5668870bb4ea223e8326707185669ffe95109067dbac5d62007f730b4bd52.jpg)

![503f9d4033d84af13c28dcead49c1aaacebf4ebdd11a1bafe5c8c93db5e08f0a.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/503f9d4033d84af13c28dcead49c1aaacebf4ebdd11a1bafe5c8c93db5e08f0a.jpg)

![5626310717e25a5dbf5c15eb079cd1f210135da85574945fe4311aa902a0451a.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/5626310717e25a5dbf5c15eb079cd1f210135da85574945fe4311aa902a0451a.jpg)

![57014d5226a6bbe8e203eb4f509550bf7c685059b035e398fcab1e1af8b67f0a.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/57014d5226a6bbe8e203eb4f509550bf7c685059b035e398fcab1e1af8b67f0a.jpg)

![59334436e755e975c3e00491526476b3dfe23e7f737a1924820c619d9cacfb1a.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/59334436e755e975c3e00491526476b3dfe23e7f737a1924820c619d9cacfb1a.jpg)

![5a20805f0b610af84a3045a490b703220f00d99cc6aacb74e53d6f66e08a74f4.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/5a20805f0b610af84a3045a490b703220f00d99cc6aacb74e53d6f66e08a74f4.jpg)

![5c0457c0ae8afd9a0049626b49fc2416b4f7ceb5d4e34661f56810f0fd076d59.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/5c0457c0ae8afd9a0049626b49fc2416b4f7ceb5d4e34661f56810f0fd076d59.jpg)

![5e29ddd6681e0d66a33636fde03348a9532eb30f35e3406e44eee1126ab21ba0.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/5e29ddd6681e0d66a33636fde03348a9532eb30f35e3406e44eee1126ab21ba0.jpg)

![6046a992239ab8dd35916a6d635a3ec4b0086e1701b5fceb5ba64f841450ed46.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/6046a992239ab8dd35916a6d635a3ec4b0086e1701b5fceb5ba64f841450ed46.jpg)

![610d14db801edd3af4a9572a4b5db1c4698674d48c368a5e3aec6d0ea5b30fbf.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/610d14db801edd3af4a9572a4b5db1c4698674d48c368a5e3aec6d0ea5b30fbf.jpg)

![6195c8e665a19ee6eb4a499b296211f85d6d62557e4598db0109eb1aac9f4dd3.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/6195c8e665a19ee6eb4a499b296211f85d6d62557e4598db0109eb1aac9f4dd3.jpg)

![6196f543c1d3e5fc67e51eee6fcebf37349cd2f31bc559e829f37ae6eb138f1a.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/6196f543c1d3e5fc67e51eee6fcebf37349cd2f31bc559e829f37ae6eb138f1a.jpg)

![61f85a0b5dccb44913a17b00ab46c71d751253cf34bf4dfb244be86e6c97ac52.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/61f85a0b5dccb44913a17b00ab46c71d751253cf34bf4dfb244be86e6c97ac52.jpg)

![6622245ee124354ec26fadc54900a3a9ab32adfe375a0ae41584d0bc5f818153.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/6622245ee124354ec26fadc54900a3a9ab32adfe375a0ae41584d0bc5f818153.jpg)

![68017cfee31973773d6abfe219ccae1e7ad4b432e602e5d9e8014a747f8bd2ad.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/68017cfee31973773d6abfe219ccae1e7ad4b432e602e5d9e8014a747f8bd2ad.jpg)

![68dada2d3d2721f7bc8427a7ee648b3db82b84beb465f184b0733e9dd4e34a5c.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/68dada2d3d2721f7bc8427a7ee648b3db82b84beb465f184b0733e9dd4e34a5c.jpg)

![69e78ac0b703c9aa0e73f42f21c7b243c12eada9a7faad0fc41a7cda319dd5f1.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/69e78ac0b703c9aa0e73f42f21c7b243c12eada9a7faad0fc41a7cda319dd5f1.jpg)

![6a70f3207da4cd183cc8e09f89d5899b70aab107803c6a83b7ea02a05d82b6b8.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/6a70f3207da4cd183cc8e09f89d5899b70aab107803c6a83b7ea02a05d82b6b8.jpg)

![6bef7c81dcbcac4598b87c735aaea11897dd41180e58ed16676a3af47af1c3ca.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/6bef7c81dcbcac4598b87c735aaea11897dd41180e58ed16676a3af47af1c3ca.jpg)

![6d802e05e24857c9e23b39f111754ae7cb9c095119970d66bdca1f05ba15c161.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/6d802e05e24857c9e23b39f111754ae7cb9c095119970d66bdca1f05ba15c161.jpg)

![6e6d455cb1491616bf17a498e01d11a8274c9540085b7318bfe2dc7463be809d.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/6e6d455cb1491616bf17a498e01d11a8274c9540085b7318bfe2dc7463be809d.jpg)

![6f70a133a07b277ccb897035ab066ccb54b43066f0a0e723b4de310ec94d13e6.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/6f70a133a07b277ccb897035ab066ccb54b43066f0a0e723b4de310ec94d13e6.jpg)

![70495231b0165995414bfdfb27948cd876a8e5f31197b9e70996eb9c09dc5962.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/70495231b0165995414bfdfb27948cd876a8e5f31197b9e70996eb9c09dc5962.jpg)

![705163005acb8849d33cfcada15e3879c1359b5ddf8282b9aae33be74ddba0b0.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/705163005acb8849d33cfcada15e3879c1359b5ddf8282b9aae33be74ddba0b0.jpg)

![714da32292adc1f88c5c035d16fb87c16ff0ccec0d3567a7152379ac00b0617f.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/714da32292adc1f88c5c035d16fb87c16ff0ccec0d3567a7152379ac00b0617f.jpg)

![721ac5156636a63f444b96290e92e2b15fdaa3f4de8803cedc07098c21dc1456.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/721ac5156636a63f444b96290e92e2b15fdaa3f4de8803cedc07098c21dc1456.jpg)

![73467f9b42a95f89ebc66e960322d2154dc721ff141b78b1872059ecae6ab506.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/73467f9b42a95f89ebc66e960322d2154dc721ff141b78b1872059ecae6ab506.jpg)

![761afbb524a3ce136a72587d74c2e35ede5bce028f644952bd9ce6bf2cbca30a.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/761afbb524a3ce136a72587d74c2e35ede5bce028f644952bd9ce6bf2cbca30a.jpg)

![7731245498e788a56bda73601eccfaaa3b8ade4950b50c190875bc77b6a29fba.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/7731245498e788a56bda73601eccfaaa3b8ade4950b50c190875bc77b6a29fba.jpg)

![79f9878682454151de288780c0094364a3688d038385783a3d71a039262293ab.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/79f9878682454151de288780c0094364a3688d038385783a3d71a039262293ab.jpg)

![7aa019d0b02fb4206e2cbed8b02279809d169fb197a17d60adf8123db43f0025.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/7aa019d0b02fb4206e2cbed8b02279809d169fb197a17d60adf8123db43f0025.jpg)

![7b6f7bf22393c305afac2424d5872ea7d898655d64bf36d7311f5d12c7523821.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/7b6f7bf22393c305afac2424d5872ea7d898655d64bf36d7311f5d12c7523821.jpg)

![7d16be0f476ca0e088dceba33e752f05b4cd70ccc5c7ce15c522eb1482cdf499.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/7d16be0f476ca0e088dceba33e752f05b4cd70ccc5c7ce15c522eb1482cdf499.jpg)

![7f7d3c2985e7a8ce3d2bf962d320874d0271092b8e5d60cc4c1c4971f7bf321a.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/7f7d3c2985e7a8ce3d2bf962d320874d0271092b8e5d60cc4c1c4971f7bf321a.jpg)

![81189298030b3753bc1c014212189711a711a2281174388629d542b5cbb48505.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/81189298030b3753bc1c014212189711a711a2281174388629d542b5cbb48505.jpg)

![818eac2d1ec5e5a9da2ca27b135036c7b55bd4ef183e9dc6cd6e3656fb3bfc8b.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/818eac2d1ec5e5a9da2ca27b135036c7b55bd4ef183e9dc6cd6e3656fb3bfc8b.jpg)

![8271e5185eaf68e6f086d08953fedf759a605f0d786e3116543fb6340d107898.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/8271e5185eaf68e6f086d08953fedf759a605f0d786e3116543fb6340d107898.jpg)

![83f3faaf9e6f368ffcd068389a889849b59aed87e469664eb0064bbd108b1124.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/83f3faaf9e6f368ffcd068389a889849b59aed87e469664eb0064bbd108b1124.jpg)

![84f5f7010a9e34935d8d6ab4f062d593266015d98679a7951a31d2d4efae1c36.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/84f5f7010a9e34935d8d6ab4f062d593266015d98679a7951a31d2d4efae1c36.jpg)

![85e4a47c1c76954dbf4efe678623bc725a187e071ee002921091af4913ec03e7.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/85e4a47c1c76954dbf4efe678623bc725a187e071ee002921091af4913ec03e7.jpg)

![8995d90e92f73e8aad9611d3f3b86990348971cc32401eb1fafffe02cbffa4f8.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/8995d90e92f73e8aad9611d3f3b86990348971cc32401eb1fafffe02cbffa4f8.jpg)

![89becd165c032d85a8cc62d3a51157ef07bf0ab26bb75908ff06957774523ae5.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/89becd165c032d85a8cc62d3a51157ef07bf0ab26bb75908ff06957774523ae5.jpg)

![8b8b3ca8d0b1f9f9b53689e5834c4c2ece3b1bc684c3cc560265fd0eaf22d3c9.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/8b8b3ca8d0b1f9f9b53689e5834c4c2ece3b1bc684c3cc560265fd0eaf22d3c9.jpg)

![8c04c5fd14098cd2a59f1d96fdcde5ae239a3cb32b041188e67cf04051248aee.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/8c04c5fd14098cd2a59f1d96fdcde5ae239a3cb32b041188e67cf04051248aee.jpg)

![8d60b0fac8b78283080fb4c85b622aff92d64e57be93931dc2ecd34f063b42a1.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/8d60b0fac8b78283080fb4c85b622aff92d64e57be93931dc2ecd34f063b42a1.jpg)

![8e1be966eda49c991ccfe738da39101ad4c27837e2ffb13cb0e77723d1274ca0.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/8e1be966eda49c991ccfe738da39101ad4c27837e2ffb13cb0e77723d1274ca0.jpg)

![8e3b1baf1a49fa67ac9fc483e3d7ed29c3021645d45705a6d86706bba9598349.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/8e3b1baf1a49fa67ac9fc483e3d7ed29c3021645d45705a6d86706bba9598349.jpg)

![91aabac6e65cef8e8acab89f7f43fbf121599eb3880d04c57c385899cd267a07.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/91aabac6e65cef8e8acab89f7f43fbf121599eb3880d04c57c385899cd267a07.jpg)

![9299aac00318c9ca9b90d89cfe7879d613b07a64c10fc1555106bb2942d23513.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/9299aac00318c9ca9b90d89cfe7879d613b07a64c10fc1555106bb2942d23513.jpg)

![9461f15082a3a3c3a0e5eb80356f4484ea5724f7603133f8cbe47b2056e589de.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/9461f15082a3a3c3a0e5eb80356f4484ea5724f7603133f8cbe47b2056e589de.jpg)

![95047644d9a2ddf3c4400dc02e2fcfd34bd8d32ef6fcb85a0799b6738bcb93d8.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/95047644d9a2ddf3c4400dc02e2fcfd34bd8d32ef6fcb85a0799b6738bcb93d8.jpg)

![99771653b809c3ec0235debe9cdfa9b641390bb9b74c7b7afade639b2add35e7.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/99771653b809c3ec0235debe9cdfa9b641390bb9b74c7b7afade639b2add35e7.jpg)

![9977a7b1d5a4f87aa080d2ea706b975ca9810e9439ae3285a35993f4ed2990c6.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/9977a7b1d5a4f87aa080d2ea706b975ca9810e9439ae3285a35993f4ed2990c6.jpg)

![99ba70b9d3dbc2af6f25fd2f0aa42b142ba299185a9b1c2f2b4cec1515a1c970.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/99ba70b9d3dbc2af6f25fd2f0aa42b142ba299185a9b1c2f2b4cec1515a1c970.jpg)

![99fe25e64e5f687ec37d8b4003e0b67f0374b654d7548a4998e9c772360b117f.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/99fe25e64e5f687ec37d8b4003e0b67f0374b654d7548a4998e9c772360b117f.jpg)

![9a0733de8004b33d1cfbad5bb8d61ed97e9d26c9c3a68545ae57b615ae48c2ab.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/9a0733de8004b33d1cfbad5bb8d61ed97e9d26c9c3a68545ae57b615ae48c2ab.jpg)

![9ae6f8fc0661b4d9afa0c497b550a009f98f0eda07d33b7a7bc387dda147476c.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/9ae6f8fc0661b4d9afa0c497b550a009f98f0eda07d33b7a7bc387dda147476c.jpg)

![9cb3f8d0a76dae36dc4c7dd0d0f1e5261e64c87b751c723aa04095feea8f8919.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/9cb3f8d0a76dae36dc4c7dd0d0f1e5261e64c87b751c723aa04095feea8f8919.jpg)

![9f7c007c05e97395c076d8c9026afc8c552fd5b2b3988081194c8e3281bae245.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/9f7c007c05e97395c076d8c9026afc8c552fd5b2b3988081194c8e3281bae245.jpg)

![a16e01bd6dd6336b5cf472444988eab27dcbb03e1a78005edcdf26fb97669955.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/a16e01bd6dd6336b5cf472444988eab27dcbb03e1a78005edcdf26fb97669955.jpg)

![a190de53ee5cde129575b72a7aa9b98c08e2d7a50263710a66b3e722dede3ee1.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/a190de53ee5cde129575b72a7aa9b98c08e2d7a50263710a66b3e722dede3ee1.jpg)

![a365294a1bf9165eef2452bb689b18925ffbd4f2d27f008367b74c7d18cfb391.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/a365294a1bf9165eef2452bb689b18925ffbd4f2d27f008367b74c7d18cfb391.jpg)

![a3d94a43736370411aca233e003ba4b930b7e3350310cabb76dff7bbbaac1e75.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/a3d94a43736370411aca233e003ba4b930b7e3350310cabb76dff7bbbaac1e75.jpg)

![a3e93de3a78dec95465a8545050def926fa1f26eb5e1e6eb6d15145b4e6aa367.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/a3e93de3a78dec95465a8545050def926fa1f26eb5e1e6eb6d15145b4e6aa367.jpg)

![a56201ec07cc0626b121559cbeff7ebc43d73136dd2ae738e89dc9121bce6365.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/a56201ec07cc0626b121559cbeff7ebc43d73136dd2ae738e89dc9121bce6365.jpg)

![a6e204cbc2eaa522722ade86156b80543c9c35cb27fc0047d3897a7678ffe2a6.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/a6e204cbc2eaa522722ade86156b80543c9c35cb27fc0047d3897a7678ffe2a6.jpg)

![a71ab6bc360865129382ca9a48c723786fbc5b2c91799f25d7f9a568cc2d547d.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/a71ab6bc360865129382ca9a48c723786fbc5b2c91799f25d7f9a568cc2d547d.jpg)

![a72444cfd56d05b929c9c9dc1757383f392b2430827c645b4dcdfe3318c0b1e5.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/a72444cfd56d05b929c9c9dc1757383f392b2430827c645b4dcdfe3318c0b1e5.jpg)

![aa02bcd230ab7bcc27bf7e1c8703cf1f679838e9d5d214d0e60a9a7fb04ac358.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/aa02bcd230ab7bcc27bf7e1c8703cf1f679838e9d5d214d0e60a9a7fb04ac358.jpg)

![aa182c92d0a2550f05cacd3b3373cd95fef68f9bd4c899a46f11113d9c262d76.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/aa182c92d0a2550f05cacd3b3373cd95fef68f9bd4c899a46f11113d9c262d76.jpg)

![ab71f4e095960066bbdccdb82d0e5d25833e7f058a5020b6ffad305c6f542a12.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/ab71f4e095960066bbdccdb82d0e5d25833e7f058a5020b6ffad305c6f542a12.jpg)

![adfe02b9941c39ad7d51c215c3843f0f6e2596472e99215d7fab9dd751efa115.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/adfe02b9941c39ad7d51c215c3843f0f6e2596472e99215d7fab9dd751efa115.jpg)

![ae4f6b1a4513d6ab33d2abe34aeb7aafa48f314d1b4b598e4fe0f907bfcdd97f.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/ae4f6b1a4513d6ab33d2abe34aeb7aafa48f314d1b4b598e4fe0f907bfcdd97f.jpg)

![af52a564f08ee3d302e4938d742a235e278926142378db704715586ec6a16d00.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/af52a564f08ee3d302e4938d742a235e278926142378db704715586ec6a16d00.jpg)

![af8bb01ba3e0a4aa08734ae67c230ea096bdc081690bb62d3d627eb471fb46cf.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/af8bb01ba3e0a4aa08734ae67c230ea096bdc081690bb62d3d627eb471fb46cf.jpg)

![b0b2c480e248398dccee80dede8f79cc18f2193885de6806b423bcb9c4c8a3df.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/b0b2c480e248398dccee80dede8f79cc18f2193885de6806b423bcb9c4c8a3df.jpg)

![b12baf77f0e6e8e6d509ef1dd1461fb11a8fef827f1821550ceee8e1661ce3db.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/b12baf77f0e6e8e6d509ef1dd1461fb11a8fef827f1821550ceee8e1661ce3db.jpg)

![b2e7a6cc0570de009bca43bbce1a4eb75ed4c84821efe7e65d406e3d9d3d61da.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/b2e7a6cc0570de009bca43bbce1a4eb75ed4c84821efe7e65d406e3d9d3d61da.jpg)

![b44784c5749cd94029954605d261ad3eeea9a2faa439d3b56474c789796aafa5.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/b44784c5749cd94029954605d261ad3eeea9a2faa439d3b56474c789796aafa5.jpg)

![b59d0943f5c5ad41179624acd833e517ca52245e2c9ccc27c6420e60db99dfaf.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/b59d0943f5c5ad41179624acd833e517ca52245e2c9ccc27c6420e60db99dfaf.jpg)

![b6c264cdd41f3fdfa874a472fe125b8c05f6556146dc679f821b6e7b7bae1cbf.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/b6c264cdd41f3fdfa874a472fe125b8c05f6556146dc679f821b6e7b7bae1cbf.jpg)

![b8347daf9467d93f9910a27cf64f20489f0535fb6b1d9b43c68fa438be218371.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/b8347daf9467d93f9910a27cf64f20489f0535fb6b1d9b43c68fa438be218371.jpg)

![b8d5d2852f8142f86561a09f4ca29f971ddf3e2124aa3f7936bf9e463e6623e5.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/b8d5d2852f8142f86561a09f4ca29f971ddf3e2124aa3f7936bf9e463e6623e5.jpg)

![b8f981ae695304d53004b2507d6c8545ebdf3882e18601b7d69473cff3e8e85e.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/b8f981ae695304d53004b2507d6c8545ebdf3882e18601b7d69473cff3e8e85e.jpg)

![b9395f19535670fe96280d884e7849fdae3dfae627d6fcc081d42790dcf29199.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/b9395f19535670fe96280d884e7849fdae3dfae627d6fcc081d42790dcf29199.jpg)

![ba78eda90a874fc736bdb67483db16ef5818ef5bd9818dc73ff41e310456cedf.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/ba78eda90a874fc736bdb67483db16ef5818ef5bd9818dc73ff41e310456cedf.jpg)

![babfb3195fa5c4ab9ff68a9494a051425d9847c1ef2d8ef7be7ac014c3c7f8dd.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/babfb3195fa5c4ab9ff68a9494a051425d9847c1ef2d8ef7be7ac014c3c7f8dd.jpg)

![bb205768de09de853fca40084bbf7935327fdc52cc604d2572c13f3c8c851b3b.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/bb205768de09de853fca40084bbf7935327fdc52cc604d2572c13f3c8c851b3b.jpg)

![bed5c5c3217ad40618cb2fa1531e7bd600a34284f8010753918ac892750aff8c.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/bed5c5c3217ad40618cb2fa1531e7bd600a34284f8010753918ac892750aff8c.jpg)

![bf6c6346a33b1e2c9d92002e753298b0a1deea4c58d4ddc99c486d0a7a3ba55b.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/bf6c6346a33b1e2c9d92002e753298b0a1deea4c58d4ddc99c486d0a7a3ba55b.jpg)

![c05065054b6bcdcff75e3edb249ab4023bd96631b633f42f86fe00f02f0a38ee.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/c05065054b6bcdcff75e3edb249ab4023bd96631b633f42f86fe00f02f0a38ee.jpg)

![c0fafeee2003eff818a314e6e1775860133c0dfa524bfb56f793ffc36a37bd8e.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/c0fafeee2003eff818a314e6e1775860133c0dfa524bfb56f793ffc36a37bd8e.jpg)

![c2fbc92c06017f46d992456e80c3c090b23f189da7c8054235715459909c0471.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/c2fbc92c06017f46d992456e80c3c090b23f189da7c8054235715459909c0471.jpg)

![c39cae02954c3af7e5d914e4ac5eb8df49e2c42754b8a7ea76724c4fbcfd25c4.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/c39cae02954c3af7e5d914e4ac5eb8df49e2c42754b8a7ea76724c4fbcfd25c4.jpg)

![c3a59c828d689177af29a459324abb48bcde8e689beb31e2964e69bd3693d191.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/c3a59c828d689177af29a459324abb48bcde8e689beb31e2964e69bd3693d191.jpg)

![c7f29cca8f2fb50920cb7a59bae70580bc1c13a41a8119c532f012dba179f9d0.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/c7f29cca8f2fb50920cb7a59bae70580bc1c13a41a8119c532f012dba179f9d0.jpg)

![c99097d181e9c81041e4e29366aecec8675abdf2f6fb62d47643647fbcdfa08c.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/c99097d181e9c81041e4e29366aecec8675abdf2f6fb62d47643647fbcdfa08c.jpg)

![c9d9ff6a542a94d5d8c18d31cf551d5f15cbf44af7558ef883b41518659889f2.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/c9d9ff6a542a94d5d8c18d31cf551d5f15cbf44af7558ef883b41518659889f2.jpg)

![cbb96826b1e5aa910b9d798316f967e818f3dd24f73b51260dbd987ce654458c.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/cbb96826b1e5aa910b9d798316f967e818f3dd24f73b51260dbd987ce654458c.jpg)

![ccd9638ed2659c798ab7be605acf705c5aa6671fb01a9fed406109000a6f59f3.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/ccd9638ed2659c798ab7be605acf705c5aa6671fb01a9fed406109000a6f59f3.jpg)

![cd05d12e0689d8603d9b4d3f2f7f1f002ad606bde4504e9a3398ee565feded52.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/cd05d12e0689d8603d9b4d3f2f7f1f002ad606bde4504e9a3398ee565feded52.jpg)

![cfcc70cb4a0a47ac2346946d93e49198aee40efa523140f3a5fbdb64dbff2e01.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/cfcc70cb4a0a47ac2346946d93e49198aee40efa523140f3a5fbdb64dbff2e01.jpg)

![d09c4b5d5ba8ffe29886fd193971f0b6de01cb37309866a138016aecd62ced87.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/d09c4b5d5ba8ffe29886fd193971f0b6de01cb37309866a138016aecd62ced87.jpg)

![d1d70117f876fe7954c3e8b12399d655afbe1f7605e154e505b4efe20f59d546.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/d1d70117f876fe7954c3e8b12399d655afbe1f7605e154e505b4efe20f59d546.jpg)

![d242d4fbe53fab39b57583ddaafd19e0ebfe1a8afc427873d229394f6e225454.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/d242d4fbe53fab39b57583ddaafd19e0ebfe1a8afc427873d229394f6e225454.jpg)

![d26dad665c690009bac9c81929c2de437dc3e1d0a9330100ffc7b5201cd59477.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/d26dad665c690009bac9c81929c2de437dc3e1d0a9330100ffc7b5201cd59477.jpg)

![d2eacbdee85cc342ede3e4be9f31c468a492b569da1b84e3a145d94431fe81ec.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/d2eacbdee85cc342ede3e4be9f31c468a492b569da1b84e3a145d94431fe81ec.jpg)

![d4016e9a279862983f5bb6ec7b94aca7d43d1f0034c1e5672f57fbeeaa2dbce8.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/d4016e9a279862983f5bb6ec7b94aca7d43d1f0034c1e5672f57fbeeaa2dbce8.jpg)

![d4876f1cb384ae8caf08c44777f66528f32cacff0eba300943ca8a7280ff595b.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/d4876f1cb384ae8caf08c44777f66528f32cacff0eba300943ca8a7280ff595b.jpg)

![d58215c805f198bb088b5ddb87ad654a28309d266905268fe4eab0ac93a2118d.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/d58215c805f198bb088b5ddb87ad654a28309d266905268fe4eab0ac93a2118d.jpg)

![d80b46f915a234525df6fe522315158da1f709c4446eecba2a736867c5695b90.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/d80b46f915a234525df6fe522315158da1f709c4446eecba2a736867c5695b90.jpg)

![d80b5841c0e2382a0364d9b8c22bfdc5fab29d967fc911052c8a3c383dd81266.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/d80b5841c0e2382a0364d9b8c22bfdc5fab29d967fc911052c8a3c383dd81266.jpg)

![d8c1344a08d70b8bf139d4bcacca849efb7f7d26890be9a1f0872678b0d8e217.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/d8c1344a08d70b8bf139d4bcacca849efb7f7d26890be9a1f0872678b0d8e217.jpg)

![d9697c7faedaff3347fba1f5e8cd1832d248804339f53bc33dad4b80a6a987fe.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/d9697c7faedaff3347fba1f5e8cd1832d248804339f53bc33dad4b80a6a987fe.jpg)

![da220699bf68c260e133f375cb2463aa13e6effe2f871335ad62e5175c94351e.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/da220699bf68c260e133f375cb2463aa13e6effe2f871335ad62e5175c94351e.jpg)

![da6f8318709afb5892ed38a3e84921440a8b20573d123b1791a1e550e9985025.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/da6f8318709afb5892ed38a3e84921440a8b20573d123b1791a1e550e9985025.jpg)

![dc0353c87d8872a8281ed9d263f72154732d3f368f8f89ba87831e33b7de9d42.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/dc0353c87d8872a8281ed9d263f72154732d3f368f8f89ba87831e33b7de9d42.jpg)

![dc0af78a7f8d2fe5502c03280403831f6a48daf6938605a67b22c8a57ad20ffc.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/dc0af78a7f8d2fe5502c03280403831f6a48daf6938605a67b22c8a57ad20ffc.jpg)

![deeba8efdb2ed2e6200792a3c2eadd14e5a5e5f5c2633701c15da8d11c9c798c.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/deeba8efdb2ed2e6200792a3c2eadd14e5a5e5f5c2633701c15da8d11c9c798c.jpg)

![dfa478b01fe7e5d7a406b1f581e56bffbffe4dee5564d0edf3732eace5f54404.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/dfa478b01fe7e5d7a406b1f581e56bffbffe4dee5564d0edf3732eace5f54404.jpg)

![e19ce32a8b393b92650c32ef63254a2d359ac6567460ed2ec48c6cbc271e9b68.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/e19ce32a8b393b92650c32ef63254a2d359ac6567460ed2ec48c6cbc271e9b68.jpg)

![e1a19de5346fda8b4cb8e944f6449b1073de79f9b72f667d983f41c6dde22433.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/e1a19de5346fda8b4cb8e944f6449b1073de79f9b72f667d983f41c6dde22433.jpg)

![e510f834531edade8767a731217e3c1c45b97ae75291e576397c715531c10e27.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/e510f834531edade8767a731217e3c1c45b97ae75291e576397c715531c10e27.jpg)

![e8a32aaf55c6915e290e2e33b32f4f99466c292ed48d5ec01e57fae0ba7b32f1.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/e8a32aaf55c6915e290e2e33b32f4f99466c292ed48d5ec01e57fae0ba7b32f1.jpg)

![ea05253f2af29de1763e53c750e0c0f98a3aa2719fcee9757fc6c9eb43c5b656.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/ea05253f2af29de1763e53c750e0c0f98a3aa2719fcee9757fc6c9eb43c5b656.jpg)

![eadf69613f29e4d57b12f8e1e10608d5ffc473ecb1743bb6196b64c0cb94c3a0.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/eadf69613f29e4d57b12f8e1e10608d5ffc473ecb1743bb6196b64c0cb94c3a0.jpg)

![f1bd38d4b39dc7269de9d3d96b5fa0d3b4ec051604941ea406d85a81ddbf7d01.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/f1bd38d4b39dc7269de9d3d96b5fa0d3b4ec051604941ea406d85a81ddbf7d01.jpg)

![f1d629dfff1d9e5d001776025d2609a5e1808abea0a34f76f925f8d2395fd089.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/f1d629dfff1d9e5d001776025d2609a5e1808abea0a34f76f925f8d2395fd089.jpg)

![f25f8dcb7dd8d15324fb9159e5788eff9de4c567728c372b17409141c08f103a.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/f25f8dcb7dd8d15324fb9159e5788eff9de4c567728c372b17409141c08f103a.jpg)

![f5f0c78b401ca6ccb9193c76ba9e8982dad1f605ab3ef56d9f5e3a721ccd2400.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/f5f0c78b401ca6ccb9193c76ba9e8982dad1f605ab3ef56d9f5e3a721ccd2400.jpg)

![f71e5056e93ffa6a2aedc94a78345bb28908ae408af9b3e0bfd0093dd7a3f6e0.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/f71e5056e93ffa6a2aedc94a78345bb28908ae408af9b3e0bfd0093dd7a3f6e0.jpg)

![f75fed7a723782054f4fef4bcdd07433182a1ada257cf18c6390c6af44d1f373.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/f75fed7a723782054f4fef4bcdd07433182a1ada257cf18c6390c6af44d1f373.jpg)

![f7a0bfda90085034f952325ae03e8236decb313f3c61bb76b6fdc3ffafbc9c19.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/f7a0bfda90085034f952325ae03e8236decb313f3c61bb76b6fdc3ffafbc9c19.jpg)

![f867d13cc10aa7f94b2338eb4f4b8d07a04ef25985a7b2934fffeff11b934d32.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/f867d13cc10aa7f94b2338eb4f4b8d07a04ef25985a7b2934fffeff11b934d32.jpg)

![f8e3089d2f1034ac224e1b1d22122b793a887179a3df7d6dc8dc836fe2975a79.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/f8e3089d2f1034ac224e1b1d22122b793a887179a3df7d6dc8dc836fe2975a79.jpg)

![f98db6f6073fdae8fc335cbaa7c05f1685e36151c3751b6aa92d9204ba80f082.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/f98db6f6073fdae8fc335cbaa7c05f1685e36151c3751b6aa92d9204ba80f082.jpg)

![f994d915e9b0c7cde8e972c200c2f911c7bd9b4f7eb9d7e097ac1eeffde8ce8e.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/f994d915e9b0c7cde8e972c200c2f911c7bd9b4f7eb9d7e097ac1eeffde8ce8e.jpg)

![fe1d2d04e97a130f1ac060d32505e6e79926f651183e7014e3185d3393091d66.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/fe1d2d04e97a130f1ac060d32505e6e79926f651183e7014e3185d3393091d66.jpg)

![ff368502aa5ef9b4cfa73a731f50bc72141a6b262d6ec6b5872a6be338c5ae2f.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/images/ff368502aa5ef9b4cfa73a731f50bc72141a6b262d6ec6b5872a6be338c5ae2f.jpg)

### Tables

![05daca755d820173c2cdf98ecd8ee0731bf7a5c71532f8e0819243f70330fe48.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/tables/05daca755d820173c2cdf98ecd8ee0731bf7a5c71532f8e0819243f70330fe48.jpg)

![1f0d702725f575282a733c9bcc469a2bcaa6f4dcbcaa3762dc70c1ba589a0ec3.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/tables/1f0d702725f575282a733c9bcc469a2bcaa6f4dcbcaa3762dc70c1ba589a0ec3.jpg)

![40c7c92c2a2edbaa74ca0ea3888235613fd8dfe8b476b942d15e651c7c944ad3.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/tables/40c7c92c2a2edbaa74ca0ea3888235613fd8dfe8b476b942d15e651c7c944ad3.jpg)

![8989293aac18d8f08015a0793688fa9d1b228a94a8641508fe3da2e97a5a3934.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/tables/8989293aac18d8f08015a0793688fa9d1b228a94a8641508fe3da2e97a5a3934.jpg)

![af241fed039639b4580b63d33859cad9b0b9f8f03f05750c948ce1410f6206f0.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/tables/af241fed039639b4580b63d33859cad9b0b9f8f03f05750c948ce1410f6206f0.jpg)

![cf9b5f7a0922a29632db1a9133d8e1942c19b308d04f34350ad74b7743ca9c15.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/tables/cf9b5f7a0922a29632db1a9133d8e1942c19b308d04f34350ad74b7743ca9c15.jpg)

![e7d9828f1ce4412d4b64c872c369906dd831fc562ba42ddaf39c264a40a3fa51.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/tables/e7d9828f1ce4412d4b64c872c369906dd831fc562ba42ddaf39c264a40a3fa51.jpg)

![f4a2e117ac3a7856a965a9f6d992f20fabb43d1a3ddd9493fb491db49953ef40.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/tables/f4a2e117ac3a7856a965a9f6d992f20fabb43d1a3ddd9493fb491db49953ef40.jpg)

![f874cf0cc5c1b8ef635aaae8c07bd5c6bf0a5b7e3fc8efe50a5c6474c0e33083.jpg](../iclr_results/2854_Meissonic_ Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image S/tables/f874cf0cc5c1b8ef635aaae8c07bd5c6bf0a5b7e3fc8efe50a5c6474c0e33083.jpg)

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

## Rational Decision-Making Agent with Learning Internal Utility Judgment


### Images

![78e4d0f5a3c36a026c8d47de30fd89e32678b15a4f1303781a637ce32dcfa47c.jpg](../iclr_results/2856_Rational Decision-Making Agent with Learning Internal Utility Judgment/images/78e4d0f5a3c36a026c8d47de30fd89e32678b15a4f1303781a637ce32dcfa47c.jpg)

![e85aae9eb791cec9b8c930450f8309f1573f196c1ccdbc8002129f571aa30910.jpg](../iclr_results/2856_Rational Decision-Making Agent with Learning Internal Utility Judgment/images/e85aae9eb791cec9b8c930450f8309f1573f196c1ccdbc8002129f571aa30910.jpg)

### Tables

![03e74cead71f84eb72fc3092d7a02112962d82887372102f6c1206ea0ed9be5a.jpg](../iclr_results/2856_Rational Decision-Making Agent with Learning Internal Utility Judgment/tables/03e74cead71f84eb72fc3092d7a02112962d82887372102f6c1206ea0ed9be5a.jpg)

![1a1ff9caad85b478329d81d87ef157d2f9eeb7d2b3e373876a28b0aac352a76f.jpg](../iclr_results/2856_Rational Decision-Making Agent with Learning Internal Utility Judgment/tables/1a1ff9caad85b478329d81d87ef157d2f9eeb7d2b3e373876a28b0aac352a76f.jpg)

![2ce9de20dac5b64da195dfa0e7a63a094fdc8b371e46ba71e2adbe05cc8b6596.jpg](../iclr_results/2856_Rational Decision-Making Agent with Learning Internal Utility Judgment/tables/2ce9de20dac5b64da195dfa0e7a63a094fdc8b371e46ba71e2adbe05cc8b6596.jpg)

![3a2ad42c835b7940906b2ae712fc4c0b1f3bc6d91fa955dc89201a12c9561849.jpg](../iclr_results/2856_Rational Decision-Making Agent with Learning Internal Utility Judgment/tables/3a2ad42c835b7940906b2ae712fc4c0b1f3bc6d91fa955dc89201a12c9561849.jpg)

![511d6e91e5deb045ecaecf7f43e79bdeff09d5f81b071f69c48795c2a78d52ed.jpg](../iclr_results/2856_Rational Decision-Making Agent with Learning Internal Utility Judgment/tables/511d6e91e5deb045ecaecf7f43e79bdeff09d5f81b071f69c48795c2a78d52ed.jpg)

![51f03e40f56938ee9da8f776efa222562935c8d22bfb6cdbe6d892b04bf32dda.jpg](../iclr_results/2856_Rational Decision-Making Agent with Learning Internal Utility Judgment/tables/51f03e40f56938ee9da8f776efa222562935c8d22bfb6cdbe6d892b04bf32dda.jpg)

![611279e4b239d2512e4ecc0feeebbfac37777eeb91f3a15a57901299569482f3.jpg](../iclr_results/2856_Rational Decision-Making Agent with Learning Internal Utility Judgment/tables/611279e4b239d2512e4ecc0feeebbfac37777eeb91f3a15a57901299569482f3.jpg)

![a413c3988bc223aa199ff90dc88547b54ed23db3819aecb4f5501b0a9bc5b8fd.jpg](../iclr_results/2856_Rational Decision-Making Agent with Learning Internal Utility Judgment/tables/a413c3988bc223aa199ff90dc88547b54ed23db3819aecb4f5501b0a9bc5b8fd.jpg)

![ac067c176b194503184b9bff4a666a13d831bdf5c1cd14adfe71a0b8e80a7323.jpg](../iclr_results/2856_Rational Decision-Making Agent with Learning Internal Utility Judgment/tables/ac067c176b194503184b9bff4a666a13d831bdf5c1cd14adfe71a0b8e80a7323.jpg)

![b83a115975af39311d2bba002d74cf5bf819c58957780eaf7909094c18d7561e.jpg](../iclr_results/2856_Rational Decision-Making Agent with Learning Internal Utility Judgment/tables/b83a115975af39311d2bba002d74cf5bf819c58957780eaf7909094c18d7561e.jpg)

![e856806426dc3ff5dff533a7619d73812bb50af232313a13741ed6157550aa25.jpg](../iclr_results/2856_Rational Decision-Making Agent with Learning Internal Utility Judgment/tables/e856806426dc3ff5dff533a7619d73812bb50af232313a13741ed6157550aa25.jpg)

![eb91b1dd9881e0f33d37632af4f2a0982393f03eb1eef3e4d2a94fe237c512f4.jpg](../iclr_results/2856_Rational Decision-Making Agent with Learning Internal Utility Judgment/tables/eb91b1dd9881e0f33d37632af4f2a0982393f03eb1eef3e4d2a94fe237c512f4.jpg)

![eec29aa86e1820b8bd0a215abe484c1efc8dc37c418d289954d1878bb4fd6d2c.jpg](../iclr_results/2856_Rational Decision-Making Agent with Learning Internal Utility Judgment/tables/eec29aa86e1820b8bd0a215abe484c1efc8dc37c418d289954d1878bb4fd6d2c.jpg)

![ffa4663c07dec699ddf8f76cbae478289beb77c7ba256dcb9566ab09f34a8e6c.jpg](../iclr_results/2856_Rational Decision-Making Agent with Learning Internal Utility Judgment/tables/ffa4663c07dec699ddf8f76cbae478289beb77c7ba256dcb9566ab09f34a8e6c.jpg)

## OVTR: End-to-End Open-Vocabulary Multiple Object Tracking with Transformer


### Images

![099cc3dc7612606a49580c52a35ddffa47e2c823e59d3e3fc8216c57fdc5f069.jpg](../iclr_results/2857_OVTR_ End-to-End Open-Vocabulary Multiple Object Tracking with Transformer/images/099cc3dc7612606a49580c52a35ddffa47e2c823e59d3e3fc8216c57fdc5f069.jpg)

![0b4ca8f4a9bd2f72d5d0a88a04f49a45a27417a5657dad028e270463f4c4e1b8.jpg](../iclr_results/2857_OVTR_ End-to-End Open-Vocabulary Multiple Object Tracking with Transformer/images/0b4ca8f4a9bd2f72d5d0a88a04f49a45a27417a5657dad028e270463f4c4e1b8.jpg)

![0f19e5ea2128e6604812b97421d3159e02091cb7edd1b45c901050e2ce0bf629.jpg](../iclr_results/2857_OVTR_ End-to-End Open-Vocabulary Multiple Object Tracking with Transformer/images/0f19e5ea2128e6604812b97421d3159e02091cb7edd1b45c901050e2ce0bf629.jpg)

![2c423cd0ee0bcc696e3eebe6ba5d56bb91b978e657ca4e4f12f6fdc2b8871bba.jpg](../iclr_results/2857_OVTR_ End-to-End Open-Vocabulary Multiple Object Tracking with Transformer/images/2c423cd0ee0bcc696e3eebe6ba5d56bb91b978e657ca4e4f12f6fdc2b8871bba.jpg)

![2cdfb6c097c06c5aa293cdccbd2c20a942fc06146969db268caad81504c52d73.jpg](../iclr_results/2857_OVTR_ End-to-End Open-Vocabulary Multiple Object Tracking with Transformer/images/2cdfb6c097c06c5aa293cdccbd2c20a942fc06146969db268caad81504c52d73.jpg)

![4156d7b2c92e6404ce9bf0dda07d4bef4cddc8b6710e578a881a696d57d958d7.jpg](../iclr_results/2857_OVTR_ End-to-End Open-Vocabulary Multiple Object Tracking with Transformer/images/4156d7b2c92e6404ce9bf0dda07d4bef4cddc8b6710e578a881a696d57d958d7.jpg)

![47f83d871fdfb3483e5dcbaf7521b6cd1c42575d469eeb7b01c198776b455481.jpg](../iclr_results/2857_OVTR_ End-to-End Open-Vocabulary Multiple Object Tracking with Transformer/images/47f83d871fdfb3483e5dcbaf7521b6cd1c42575d469eeb7b01c198776b455481.jpg)

![65c306ccbe8e3e5925a6e5ff62ca757d4d78fa5597259f6d991798d67095a96c.jpg](../iclr_results/2857_OVTR_ End-to-End Open-Vocabulary Multiple Object Tracking with Transformer/images/65c306ccbe8e3e5925a6e5ff62ca757d4d78fa5597259f6d991798d67095a96c.jpg)

![e0d13b108238cf728123c7b1852d704546c0c902a54f6fd3fbc494f576f6484d.jpg](../iclr_results/2857_OVTR_ End-to-End Open-Vocabulary Multiple Object Tracking with Transformer/images/e0d13b108238cf728123c7b1852d704546c0c902a54f6fd3fbc494f576f6484d.jpg)

![fec01756036499626abee241cc4cb56d119036961598c415ff6899f0617b61e6.jpg](../iclr_results/2857_OVTR_ End-to-End Open-Vocabulary Multiple Object Tracking with Transformer/images/fec01756036499626abee241cc4cb56d119036961598c415ff6899f0617b61e6.jpg)

### Tables

![01d63441634e192aadfb3adeae53f727c0621536f18304a04a48aefb45b3c16a.jpg](../iclr_results/2857_OVTR_ End-to-End Open-Vocabulary Multiple Object Tracking with Transformer/tables/01d63441634e192aadfb3adeae53f727c0621536f18304a04a48aefb45b3c16a.jpg)

![08f60153c5134a43b1c06cc7dc8d04ca197e529950d96c89e5e920eb03e31e3e.jpg](../iclr_results/2857_OVTR_ End-to-End Open-Vocabulary Multiple Object Tracking with Transformer/tables/08f60153c5134a43b1c06cc7dc8d04ca197e529950d96c89e5e920eb03e31e3e.jpg)

![2cd9c2c5db64afaedf84c0140255041ef84575436b87b23d3634da0252e4b121.jpg](../iclr_results/2857_OVTR_ End-to-End Open-Vocabulary Multiple Object Tracking with Transformer/tables/2cd9c2c5db64afaedf84c0140255041ef84575436b87b23d3634da0252e4b121.jpg)

![3adcf8783dd0e794b6b982fd96b2d7fb93d80acd6c1901ea53d539485ba216d8.jpg](../iclr_results/2857_OVTR_ End-to-End Open-Vocabulary Multiple Object Tracking with Transformer/tables/3adcf8783dd0e794b6b982fd96b2d7fb93d80acd6c1901ea53d539485ba216d8.jpg)

![4585162ced7a9d03146316aa1f9328a44e1a28e59ab3fcc843e00640520d95f6.jpg](../iclr_results/2857_OVTR_ End-to-End Open-Vocabulary Multiple Object Tracking with Transformer/tables/4585162ced7a9d03146316aa1f9328a44e1a28e59ab3fcc843e00640520d95f6.jpg)

![523dd1eacce1da0279c3462db301e5bfe68a43064e98c5c0e9ac43bda14bdfb2.jpg](../iclr_results/2857_OVTR_ End-to-End Open-Vocabulary Multiple Object Tracking with Transformer/tables/523dd1eacce1da0279c3462db301e5bfe68a43064e98c5c0e9ac43bda14bdfb2.jpg)

![6012efac38cc7c14a59c0eb72d159bfd7a3d46d8e6d5a7507da75d88e2a2a827.jpg](../iclr_results/2857_OVTR_ End-to-End Open-Vocabulary Multiple Object Tracking with Transformer/tables/6012efac38cc7c14a59c0eb72d159bfd7a3d46d8e6d5a7507da75d88e2a2a827.jpg)

![673fb54ea57f75c433b3c67d24dc9d6a96eeac34c5499fa093526f8388ee4e19.jpg](../iclr_results/2857_OVTR_ End-to-End Open-Vocabulary Multiple Object Tracking with Transformer/tables/673fb54ea57f75c433b3c67d24dc9d6a96eeac34c5499fa093526f8388ee4e19.jpg)

![6f2b4487ce8aa177286cf8b3992f8753d685ac786c100e6a51f056ad73114146.jpg](../iclr_results/2857_OVTR_ End-to-End Open-Vocabulary Multiple Object Tracking with Transformer/tables/6f2b4487ce8aa177286cf8b3992f8753d685ac786c100e6a51f056ad73114146.jpg)

![97b7df7ff18c5a2066f44b4f1590770dddf34db1009cd3950fd0bef78bf1867a.jpg](../iclr_results/2857_OVTR_ End-to-End Open-Vocabulary Multiple Object Tracking with Transformer/tables/97b7df7ff18c5a2066f44b4f1590770dddf34db1009cd3950fd0bef78bf1867a.jpg)

![a039c39ea9fffa111e2de6c6e9b82a415e2ee3236715df6afb450ef8c4344631.jpg](../iclr_results/2857_OVTR_ End-to-End Open-Vocabulary Multiple Object Tracking with Transformer/tables/a039c39ea9fffa111e2de6c6e9b82a415e2ee3236715df6afb450ef8c4344631.jpg)

![c541faaa73086ff1f6486e8ca0d23634ff8df5ff3969978a34c33bcac5f05f94.jpg](../iclr_results/2857_OVTR_ End-to-End Open-Vocabulary Multiple Object Tracking with Transformer/tables/c541faaa73086ff1f6486e8ca0d23634ff8df5ff3969978a34c33bcac5f05f94.jpg)

![cce18bbedfd2e2991a3b0d3b4921934fe6e77d59eb4e07871df0e960e7495728.jpg](../iclr_results/2857_OVTR_ End-to-End Open-Vocabulary Multiple Object Tracking with Transformer/tables/cce18bbedfd2e2991a3b0d3b4921934fe6e77d59eb4e07871df0e960e7495728.jpg)

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

## Ferret-UI 2: Mastering Universal User Interface Understanding Across Platforms


### Images

![0156774af046155d6b0790c53a7f85b478c821ca501398b84f48d2ac39081011.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/images/0156774af046155d6b0790c53a7f85b478c821ca501398b84f48d2ac39081011.jpg)

![0c2d8879b6c83e90306866b11f61b78d03add0e53e046880e6fa4e72c3bdfab4.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/images/0c2d8879b6c83e90306866b11f61b78d03add0e53e046880e6fa4e72c3bdfab4.jpg)

![256087f7884db67968f62698792fdef76a4cc27bdac43c273d614e45bb87926a.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/images/256087f7884db67968f62698792fdef76a4cc27bdac43c273d614e45bb87926a.jpg)

![5fbd3753f84fcd4905ebb357ca7202339905982cb212f2d8d49b7222c3191901.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/images/5fbd3753f84fcd4905ebb357ca7202339905982cb212f2d8d49b7222c3191901.jpg)

![6fa0adfdd893a8de6fd5dcb4338615231ef283265b55783ef85500e69fde2574.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/images/6fa0adfdd893a8de6fd5dcb4338615231ef283265b55783ef85500e69fde2574.jpg)

![777ebd025657cab345033ec49a106280c965ac7c9f0f68d8de35476173d3aa7c.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/images/777ebd025657cab345033ec49a106280c965ac7c9f0f68d8de35476173d3aa7c.jpg)

![81b783149764ab309ffa6899430c06d9252e6ea0115bb23914665cadb140dfed.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/images/81b783149764ab309ffa6899430c06d9252e6ea0115bb23914665cadb140dfed.jpg)

![867b40ce8bd5d17348d3b101b16e9aad8275dc623120757d21d1148d08f8c369.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/images/867b40ce8bd5d17348d3b101b16e9aad8275dc623120757d21d1148d08f8c369.jpg)

![96eb579c53ec7ffb49ca6332f63a73ed1086fed31effb35b2684521233f4b3d6.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/images/96eb579c53ec7ffb49ca6332f63a73ed1086fed31effb35b2684521233f4b3d6.jpg)

![979c19af1a7c60c2768a5b47036919b89fa9a352e6f07f3e66ab50a22f630efb.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/images/979c19af1a7c60c2768a5b47036919b89fa9a352e6f07f3e66ab50a22f630efb.jpg)

![9cbe252a627942950c13250d0f0a5a7db46bafd83a56ce13f5d158a21cd424b2.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/images/9cbe252a627942950c13250d0f0a5a7db46bafd83a56ce13f5d158a21cd424b2.jpg)

![a4a3217670aeab39206f220797286aa5eeb20f67f0e2dea863940dd00ffff4a1.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/images/a4a3217670aeab39206f220797286aa5eeb20f67f0e2dea863940dd00ffff4a1.jpg)

![ae10b7a109cc940b7d01ef2b352d68387fe9f8f9b3fc3332174ef767eb03a091.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/images/ae10b7a109cc940b7d01ef2b352d68387fe9f8f9b3fc3332174ef767eb03a091.jpg)

![c6826e9e609db94756d011a168fb59d5d0e3590ae46c7ed895c453cfda776a7a.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/images/c6826e9e609db94756d011a168fb59d5d0e3590ae46c7ed895c453cfda776a7a.jpg)

![df5468d023bcbf28565251343e9461371c00bc6aa68f002c4aa9ea9a3e4a6a3c.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/images/df5468d023bcbf28565251343e9461371c00bc6aa68f002c4aa9ea9a3e4a6a3c.jpg)

### Tables

![1d99fa7bb3ce7019f768c8d280080b6439c9ba2fa1c17c74310cdf7cff0530d1.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/tables/1d99fa7bb3ce7019f768c8d280080b6439c9ba2fa1c17c74310cdf7cff0530d1.jpg)

![3612ad7f3b6e43a6af882f427ba5d6a23988efd2ec78c8c3cb87423276a71afa.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/tables/3612ad7f3b6e43a6af882f427ba5d6a23988efd2ec78c8c3cb87423276a71afa.jpg)

![827b79e055d6b9d2a320182571512ac5350bb261997c61a8e73e4f7323878b8a.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/tables/827b79e055d6b9d2a320182571512ac5350bb261997c61a8e73e4f7323878b8a.jpg)

![a9782b98f3a32993a50cbdcf2a4ca147a75cadf613adb8078a3920132a35ec33.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/tables/a9782b98f3a32993a50cbdcf2a4ca147a75cadf613adb8078a3920132a35ec33.jpg)

![b33bc8aee8fbc96eba2dc9a7c353862737b00f2988d0c801512d2e41fc6445fd.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/tables/b33bc8aee8fbc96eba2dc9a7c353862737b00f2988d0c801512d2e41fc6445fd.jpg)

![b937a8b1daea64446bdb4e37df0d8995d0ae7e4b931141bdc875f756123d40a2.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/tables/b937a8b1daea64446bdb4e37df0d8995d0ae7e4b931141bdc875f756123d40a2.jpg)

![d18dea7c32836bc4dd7c59452d120dd868a7e0d5ee56355bf77680f24dab755a.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/tables/d18dea7c32836bc4dd7c59452d120dd868a7e0d5ee56355bf77680f24dab755a.jpg)

![e5a57a9a06f56b8d6b50d3b8bdf44107d97299efe8f94946318b626fdcc8f530.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/tables/e5a57a9a06f56b8d6b50d3b8bdf44107d97299efe8f94946318b626fdcc8f530.jpg)

![e82cf4b4a525aa5a5d7a49337397e2d60bfd17bc65d1151f5b9f3c1c8e899724.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/tables/e82cf4b4a525aa5a5d7a49337397e2d60bfd17bc65d1151f5b9f3c1c8e899724.jpg)

![e8e14b7beea5b314782d08155f6c9d642e0fdb5a46af0f9838125c57ec5592bf.jpg](../iclr_results/2859_Ferret-UI 2_ Mastering Universal User Interface Understanding Across Platforms/tables/e8e14b7beea5b314782d08155f6c9d642e0fdb5a46af0f9838125c57ec5592bf.jpg)

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

## Self-Normalized Resets for Plasticity in Continual Learning


### Images

![1cfc85a9b39333144c9f97e29e6738d3293a30a797d2f1b02ecb1b3757dcb327.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/images/1cfc85a9b39333144c9f97e29e6738d3293a30a797d2f1b02ecb1b3757dcb327.jpg)

![281b0a1fc2ea2279e1a423290cd4809056d85e47941d08b0e6161fa38f29e803.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/images/281b0a1fc2ea2279e1a423290cd4809056d85e47941d08b0e6161fa38f29e803.jpg)

![471ab3cff4605df02cd38c7d3b8d3fb188ccfedb81afdb9eb51b73118f7952c7.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/images/471ab3cff4605df02cd38c7d3b8d3fb188ccfedb81afdb9eb51b73118f7952c7.jpg)

![52469365c8d7a17152d7d0e81b227e484e36eb47bdbbc007c2cfdc38117f5e76.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/images/52469365c8d7a17152d7d0e81b227e484e36eb47bdbbc007c2cfdc38117f5e76.jpg)

![8df5fee23fcd5a95428f29e0d9bf2737ea9a0f464136bc94daa52f3a9227d839.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/images/8df5fee23fcd5a95428f29e0d9bf2737ea9a0f464136bc94daa52f3a9227d839.jpg)

![93f9eab708daf2d00ec2e1bb963fd09e0e64291366e3e1ff525f59ca554ebc22.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/images/93f9eab708daf2d00ec2e1bb963fd09e0e64291366e3e1ff525f59ca554ebc22.jpg)

![a1052d5da2ce929ff20425f0272cc907ef20d3985964e65ae858709deaf6d88b.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/images/a1052d5da2ce929ff20425f0272cc907ef20d3985964e65ae858709deaf6d88b.jpg)

![a777a5cbc270a125964ddb28415c790bc8dbac90f89c6fca3ae8da1018edb7bc.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/images/a777a5cbc270a125964ddb28415c790bc8dbac90f89c6fca3ae8da1018edb7bc.jpg)

![be79a09bc808bc49cbc23a2cd81be8fab2e51132176df22f1e391823ccd446ac.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/images/be79a09bc808bc49cbc23a2cd81be8fab2e51132176df22f1e391823ccd446ac.jpg)

![c719c2e9c3c42cae807f34671ed0e17f184fa91ecc1e12ba1bb83fdc14c10afe.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/images/c719c2e9c3c42cae807f34671ed0e17f184fa91ecc1e12ba1bb83fdc14c10afe.jpg)

![d0e436b5f2ae660d94ca04de51dfb00647f441858a75ae421c0967593a3f829b.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/images/d0e436b5f2ae660d94ca04de51dfb00647f441858a75ae421c0967593a3f829b.jpg)

![e90f272e8f71a092c5e5f617a5d7b93e0fdab98a273d051f05694954430cdafc.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/images/e90f272e8f71a092c5e5f617a5d7b93e0fdab98a273d051f05694954430cdafc.jpg)

![ebbc3b401e4d2c9120b1a91736bfc5bfe2c3f5f136bd79758b9e7801bab43d41.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/images/ebbc3b401e4d2c9120b1a91736bfc5bfe2c3f5f136bd79758b9e7801bab43d41.jpg)

![f5726e2f983aebc21f123a7e1e388e61eb6452f0e6a150e089c95fb5309603fb.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/images/f5726e2f983aebc21f123a7e1e388e61eb6452f0e6a150e089c95fb5309603fb.jpg)

### Tables

![05042a1551c4f69ba3fb5ef7bcbad41ea4daf7b8dc177f3d59c05419f8eb25f8.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/tables/05042a1551c4f69ba3fb5ef7bcbad41ea4daf7b8dc177f3d59c05419f8eb25f8.jpg)

![1ae2f9f9fa3518b266449e0a8ca8d5a0d4ce6c2dd9689748330ea54f764f49fc.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/tables/1ae2f9f9fa3518b266449e0a8ca8d5a0d4ce6c2dd9689748330ea54f764f49fc.jpg)

![21562090ce7e1ba7af8a8d635b6a6c66bf6e1dca1b9ad6dca18de7981138321d.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/tables/21562090ce7e1ba7af8a8d635b6a6c66bf6e1dca1b9ad6dca18de7981138321d.jpg)

![2b1006e487f7c41052bc6d6d60a2e255a738920c88e167796e69714263ca13a0.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/tables/2b1006e487f7c41052bc6d6d60a2e255a738920c88e167796e69714263ca13a0.jpg)

![37715738e094eb18b5e66b6a22bfb5c83e726bae913c2b9f41adc52bba051ff7.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/tables/37715738e094eb18b5e66b6a22bfb5c83e726bae913c2b9f41adc52bba051ff7.jpg)

![3f98ad3fe9b443b8661ffb8f2c4a81010b25a2859ae8bfd0d725c4dc6cf519ee.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/tables/3f98ad3fe9b443b8661ffb8f2c4a81010b25a2859ae8bfd0d725c4dc6cf519ee.jpg)

![41d7c25a4912814c7d0a36ae7db212d50eeed502463e6745be15585f0e394465.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/tables/41d7c25a4912814c7d0a36ae7db212d50eeed502463e6745be15585f0e394465.jpg)

![512beaaf1b32dc8fae533699a5e938c4e6183bb27659607f88c5ae5cced9e940.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/tables/512beaaf1b32dc8fae533699a5e938c4e6183bb27659607f88c5ae5cced9e940.jpg)

![550e9de8d26fa1905cbf4b7bc8be19634396a2c78d8239b8419c1090f0e5fc73.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/tables/550e9de8d26fa1905cbf4b7bc8be19634396a2c78d8239b8419c1090f0e5fc73.jpg)

![788a066cdc22a95222ee6ef541082900ee6dc31eb66c1c5431bb7605cb55becb.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/tables/788a066cdc22a95222ee6ef541082900ee6dc31eb66c1c5431bb7605cb55becb.jpg)

![854db4e9c0119fc2edaed04e21273c63b19428ba570602b11f98144b77b944ca.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/tables/854db4e9c0119fc2edaed04e21273c63b19428ba570602b11f98144b77b944ca.jpg)

![8e79255ecca2368f9127a51dc4aea50c052b018e17ae4c2d51180a7df64903f3.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/tables/8e79255ecca2368f9127a51dc4aea50c052b018e17ae4c2d51180a7df64903f3.jpg)

![9850d1338a97dea1f6d92d9e4c03783f292746c9ef112515a71bc2b55c02026e.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/tables/9850d1338a97dea1f6d92d9e4c03783f292746c9ef112515a71bc2b55c02026e.jpg)

![9d4f5215b13cc2e68886cd851dc501fd728856b3c2b565f3d13db0b2effae31c.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/tables/9d4f5215b13cc2e68886cd851dc501fd728856b3c2b565f3d13db0b2effae31c.jpg)

![b369fcbeba3ee6ad2b80f23805c7148cc70c69228a178aa827c271cff16401ee.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/tables/b369fcbeba3ee6ad2b80f23805c7148cc70c69228a178aa827c271cff16401ee.jpg)

![ea0704fc8086f4634854e1c8a795342eb622b95a24a38728cb7d9e271922ab4c.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/tables/ea0704fc8086f4634854e1c8a795342eb622b95a24a38728cb7d9e271922ab4c.jpg)

![ee1ad1f9be69902642bc4dc2e8e5357bc11e61c297c6560c6c84dc135605c65e.jpg](../iclr_results/2862_Self-Normalized Resets for Plasticity in Continual Learning/tables/ee1ad1f9be69902642bc4dc2e8e5357bc11e61c297c6560c6c84dc135605c65e.jpg)

## Implicit In-context Learning

### Images

![142a7ef56165b08a2cd0461ecaec4b79db0c1dc501b514a2233fbb76b315ef34.jpg](../iclr_results/2863_Implicit In-context Learning/images/142a7ef56165b08a2cd0461ecaec4b79db0c1dc501b514a2233fbb76b315ef34.jpg)

![1c250feac08655e9aceec120fd50ca31bfaaeadd67128e3c23c4cb781e1d7e6d.jpg](../iclr_results/2863_Implicit In-context Learning/images/1c250feac08655e9aceec120fd50ca31bfaaeadd67128e3c23c4cb781e1d7e6d.jpg)

![35aec7dedfff0e2098e0afb19330b128f801618c5014005fa710125d773bbe37.jpg](../iclr_results/2863_Implicit In-context Learning/images/35aec7dedfff0e2098e0afb19330b128f801618c5014005fa710125d773bbe37.jpg)

![59d5c17d451d923c4bb8ab21706c3844481f3ff73e1b14ac368e89e3f2b15a92.jpg](../iclr_results/2863_Implicit In-context Learning/images/59d5c17d451d923c4bb8ab21706c3844481f3ff73e1b14ac368e89e3f2b15a92.jpg)

![c5ffd79bdb61b3571a73e8a6f61ed02821b53f76882090bd136e3318693c1073.jpg](../iclr_results/2863_Implicit In-context Learning/images/c5ffd79bdb61b3571a73e8a6f61ed02821b53f76882090bd136e3318693c1073.jpg)

![cc15a253a462f0c8bce562f83a75f0ddc49590596c8b2158618a5c3769c59c64.jpg](../iclr_results/2863_Implicit In-context Learning/images/cc15a253a462f0c8bce562f83a75f0ddc49590596c8b2158618a5c3769c59c64.jpg)

![e84a48785ab887ee27592737abad63ae1d97078207e835b4cef5a1577e1e56f2.jpg](../iclr_results/2863_Implicit In-context Learning/images/e84a48785ab887ee27592737abad63ae1d97078207e835b4cef5a1577e1e56f2.jpg)

### Tables

![205f6913069ebe93e2b87ea30e7703c5dbe9202f3ba2207c96d9c7219b2288eb.jpg](../iclr_results/2863_Implicit In-context Learning/tables/205f6913069ebe93e2b87ea30e7703c5dbe9202f3ba2207c96d9c7219b2288eb.jpg)

![37f09ad4239ecbf95c7cca38389238edcac859ea5e77047c29d14af133f63a2e.jpg](../iclr_results/2863_Implicit In-context Learning/tables/37f09ad4239ecbf95c7cca38389238edcac859ea5e77047c29d14af133f63a2e.jpg)

![4332125ddea7bcb0e94b3e9b77bb97ca122832b14c3d6a2a6ed3aad1bb19ec1c.jpg](../iclr_results/2863_Implicit In-context Learning/tables/4332125ddea7bcb0e94b3e9b77bb97ca122832b14c3d6a2a6ed3aad1bb19ec1c.jpg)

![582b597cf8ceae48b3f2d58e7eff488a94e8733d211382ba5e2f0aa142ebab57.jpg](../iclr_results/2863_Implicit In-context Learning/tables/582b597cf8ceae48b3f2d58e7eff488a94e8733d211382ba5e2f0aa142ebab57.jpg)

![7ba0dec9f66e25a2c422a354e5060c93df630b9d149fa74299a70c5599facdba.jpg](../iclr_results/2863_Implicit In-context Learning/tables/7ba0dec9f66e25a2c422a354e5060c93df630b9d149fa74299a70c5599facdba.jpg)

![7cfe4806ae6cbc6564f8968d8676aa8776148ddb02f95d2d53d0e644e729dd91.jpg](../iclr_results/2863_Implicit In-context Learning/tables/7cfe4806ae6cbc6564f8968d8676aa8776148ddb02f95d2d53d0e644e729dd91.jpg)

![88786ac46fcf80a6ea15bb528955da62c077eefd6d09045cbde74576fc91e83d.jpg](../iclr_results/2863_Implicit In-context Learning/tables/88786ac46fcf80a6ea15bb528955da62c077eefd6d09045cbde74576fc91e83d.jpg)

![a7b281757cedfb2637a7e1d0e3d7e95d293c0a61dfa53531ffd4fcf58a276fe4.jpg](../iclr_results/2863_Implicit In-context Learning/tables/a7b281757cedfb2637a7e1d0e3d7e95d293c0a61dfa53531ffd4fcf58a276fe4.jpg)

![ad5e19df5bfde8f4db446e4db06630a006951cc1a3958f44674ecf99fa2f0fc0.jpg](../iclr_results/2863_Implicit In-context Learning/tables/ad5e19df5bfde8f4db446e4db06630a006951cc1a3958f44674ecf99fa2f0fc0.jpg)

![b8bb4a9761360046e44d8c6f5d34320274811ea9833ce57dcb6d069bd4a1cae2.jpg](../iclr_results/2863_Implicit In-context Learning/tables/b8bb4a9761360046e44d8c6f5d34320274811ea9833ce57dcb6d069bd4a1cae2.jpg)

![be67161b89b7cbf4a559083783e691a4cb6e5dc06f8fb7cf1200aeac7c882794.jpg](../iclr_results/2863_Implicit In-context Learning/tables/be67161b89b7cbf4a559083783e691a4cb6e5dc06f8fb7cf1200aeac7c882794.jpg)

![c22ca0c10372c20494f48d32ebf81cbadb4242eb4b6a67baed0315097007e4cf.jpg](../iclr_results/2863_Implicit In-context Learning/tables/c22ca0c10372c20494f48d32ebf81cbadb4242eb4b6a67baed0315097007e4cf.jpg)

![d3833d307c3c1979e0995fe6ebb3b7d0b349b050aa8b49a7fa432d2f51a1e263.jpg](../iclr_results/2863_Implicit In-context Learning/tables/d3833d307c3c1979e0995fe6ebb3b7d0b349b050aa8b49a7fa432d2f51a1e263.jpg)

![db3ef99169b0fec1c5fe9ffa81e03fcf597def3491805213e4cbd6b3d0ac8245.jpg](../iclr_results/2863_Implicit In-context Learning/tables/db3ef99169b0fec1c5fe9ffa81e03fcf597def3491805213e4cbd6b3d0ac8245.jpg)

![ea1d3f0511e450fe377e10b86ae0d35ec5e92463db406200c547cfe4b1c08de7.jpg](../iclr_results/2863_Implicit In-context Learning/tables/ea1d3f0511e450fe377e10b86ae0d35ec5e92463db406200c547cfe4b1c08de7.jpg)
