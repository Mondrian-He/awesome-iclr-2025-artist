# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 72 of 100

## 目录 (Table of Contents)

1. [Cauchy-Schwarz Regularizers](#Cauchy-Schwarz-Regularizers)
2. [SysBench: Can LLMs Follow System Message?](#SysBench-Can-LLMs-Follow-System-Message)
3. [Distribution-Specific Agnostic Conditional Classification With Halfspaces](#Distribution-Specific-Agnostic-Conditional-Classification-With-Halfspaces)
4. [GALA: Geometry-Aware Local Adaptive Grids for Detailed 3D Generation](#GALA-Geometry-Aware-Local-Adaptive-Grids-for-Detailed-3D-Generation)
5. [Distilling Structural Representations into Protein Sequence Models](#Distilling-Structural-Representations-into-Protein-Sequence-Models)
6. [YOLO-RD: Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary](#YOLO-RD-Introducing-Relevant-and-Compact-Explicit-Knowledge-to-YOLO-by-Retriever-Dictionary)
7. [Computing Circuits Optimization via Model-Based Circuit Genetic Evolution](#Computing-Circuits-Optimization-via-Model-Based-Circuit-Genetic-Evolution)
8. [To Tackle Adversarial Transferability: A Novel Ensemble Training Method with Fourier Transformation](#To-Tackle-Adversarial-Transferability-A-Novel-Ensemble-Training-Method-with-Fourier-Transformation)
9. [SMITE: Segment Me In TimE](#SMITE-Segment-Me-In-TimE)
10. [Federated Granger Causality Learning For Interdependent Clients With State Space Representation](#Federated-Granger-Causality-Learning-For-Interdependent-Clients-With-State-Space-Representation)
11. [Uncovering Latent Memories in Large Language Models](#Uncovering-Latent-Memories-in-Large-Language-Models)
12. [CaPo: Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation](#CaPo-Cooperative-Plan-Optimization-for-Efficient-Embodied-Multi-Agent-Cooperation)
13. [LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations](#LLMs-Know-More-Than-They-Show-On-the-Intrinsic-Representation-of-LLM-Hallucinations)
14. [Can Watermarks be Used to Detect LLM IP Infringement For Free?](#Can-Watermarks-be-Used-to-Detect-LLM-IP-Infringement-For-Free)
15. [Timer-XL: Long-Context Transformers for Unified Time Series Forecasting](#Timer-XL-Long-Context-Transformers-for-Unified-Time-Series-Forecasting)
16. [Improving Semantic Understanding in Speech Language Models via Brain-tuning](#Improving-Semantic-Understanding-in-Speech-Language-Models-via-Brain-tuning)
17. [MambaQuant: Quantizing the Mamba Family with Variance Aligned Rotation Methods](#MambaQuant-Quantizing-the-Mamba-Family-with-Variance-Aligned-Rotation-Methods)
18. [Training-free Camera Control for Video Generation](#Training-free-Camera-Control-for-Video-Generation)
19. [EVA: Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow](#EVA-Geometric-Inverse-Design-for-Fast-Protein-Motif-Scaffolding-with-Coupled-Flow)
20. [MeshAnything: Artist-Created Mesh Generation with Autoregressive Transformers](#MeshAnything-Artist-Created-Mesh-Generation-with-Autoregressive-Transformers)
21. [Inverse Rendering using Multi-Bounce Path Tracing and Reservoir Sampling](#Inverse-Rendering-using-Multi-Bounce-Path-Tracing-and-Reservoir-Sampling)
22. [From GNNs to Trees: Multi-Granular Interpretability for Graph Neural Networks](#From-GNNs-to-Trees-Multi-Granular-Interpretability-for-Graph-Neural-Networks)
23. [ReAttention: Training-Free Infinite Context with Finite Attention Scope](#ReAttention-Training-Free-Infinite-Context-with-Finite-Attention-Scope)
24. [Divergence-Regularized Discounted Aggregation: Equilibrium Finding in Multiplayer Partially Observable Stochastic Games](#Divergence-Regularized-Discounted-Aggregation-Equilibrium-Finding-in-Multiplayer-Partially-Observable-Stochastic-Games)
25. [Prevalence of Negative Transfer in Continual Reinforcement Learning: Analyses and a Simple Baseline](#Prevalence-of-Negative-Transfer-in-Continual-Reinforcement-Learning-Analyses-and-a-Simple-Baseline)
26. [Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images](#Tracking-the-Copyright-of-Large-Vision-Language-Models-through-Parameter-Learning-Adversarial-Images)
27. [MLLM as Retriever: Interactively Learning Multimodal Retrieval for Embodied Agents](#MLLM-as-Retriever-Interactively-Learning-Multimodal-Retrieval-for-Embodied-Agents)
28. [VLAS: Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation](#VLAS-Vision-Language-Action-Model-with-Speech-Instructions-for-Customized-Robot-Manipulation)
29. [Scaling Large Language Model-based Multi-Agent Collaboration](#Scaling-Large-Language-Model-based-Multi-Agent-Collaboration)
30. [LLMs Can Plan Only If We Tell Them](#LLMs-Can-Plan-Only-If-We-Tell-Them)
31. [Gaussian Differentially Private Human Faces Under a Face Radial Curve Representation](#Gaussian-Differentially-Private-Human-Faces-Under-a-Face-Radial-Curve-Representation)
32. [Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later](#Revisiting-Nearest-Neighbor-for-Tabular-Data-A-Deep-Tabular-Baseline-Two-Decades-Later)
33. [Injective flows for star-like manifolds](#Injective-flows-for-star-like-manifolds)
34. [Robustness of Quantum Algorithms for Nonconvex Optimization](#Robustness-of-Quantum-Algorithms-for-Nonconvex-Optimization)
35. [Simple is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation](#Simple-is-Effective-The-Roles-of-Graphs-and-Large-Language-Models-in-Knowledge-Graph-Based-Retrieval-Augmented-Generation)
36. [Towards Calibrated Deep Clustering Network](#Towards-Calibrated-Deep-Clustering-Network)
37. [Multiagent Finetuning: Self Improvement with Diverse Reasoning Chains](#Multiagent-Finetuning-Self-Improvement-with-Diverse-Reasoning-Chains)

---


## Cauchy-Schwarz Regularizers

### Images

![50adee5616abf26a49f821243297a603800acc5d927ea834cd929fddb62645a5.jpg](../iclr_results/2640_Language Imbalance Driven Rewarding for Multilingual Self-improving/images/50adee5616abf26a49f821243297a603800acc5d927ea834cd929fddb62645a5.jpg)

![8e41131a9a2090992f0d7418fbe7c3766fd93142f5687b3d0d4beafa059c87cb.jpg](../iclr_results/2640_Language Imbalance Driven Rewarding for Multilingual Self-improving/images/8e41131a9a2090992f0d7418fbe7c3766fd93142f5687b3d0d4beafa059c87cb.jpg)

![d0c640dab5ecdcb9ee387c9f5cfee0e7fd483154a0688e593fcb298f9732650b.jpg](../iclr_results/2640_Language Imbalance Driven Rewarding for Multilingual Self-improving/images/d0c640dab5ecdcb9ee387c9f5cfee0e7fd483154a0688e593fcb298f9732650b.jpg)

![d8eaf63acc155089d1b5d0d182d322caaa6607f5472d2bcbdf06e565a9cb7b50.jpg](../iclr_results/2640_Language Imbalance Driven Rewarding for Multilingual Self-improving/images/d8eaf63acc155089d1b5d0d182d322caaa6607f5472d2bcbdf06e565a9cb7b50.jpg)

### Tables

![01ad150e1e8a6d71a3987a0849543e6f2065456dff439c52a8d1355c085e1885.jpg](../iclr_results/2640_Language Imbalance Driven Rewarding for Multilingual Self-improving/tables/01ad150e1e8a6d71a3987a0849543e6f2065456dff439c52a8d1355c085e1885.jpg)

![062f1dec2c0db731be2fdca3282862b2809decadc23a83c4d665935ebde4310a.jpg](../iclr_results/2640_Language Imbalance Driven Rewarding for Multilingual Self-improving/tables/062f1dec2c0db731be2fdca3282862b2809decadc23a83c4d665935ebde4310a.jpg)

![1087a7a492c5a54dee2a973d8a820cf4775bbf88817a1c452dde15e05c0357e8.jpg](../iclr_results/2640_Language Imbalance Driven Rewarding for Multilingual Self-improving/tables/1087a7a492c5a54dee2a973d8a820cf4775bbf88817a1c452dde15e05c0357e8.jpg)

![1d7eeb8dab571ec834c6d110c646ecb2bd966ba1e15788a5b7b55f6e7ee56dcc.jpg](../iclr_results/2640_Language Imbalance Driven Rewarding for Multilingual Self-improving/tables/1d7eeb8dab571ec834c6d110c646ecb2bd966ba1e15788a5b7b55f6e7ee56dcc.jpg)

![263cebff66d7a1e6ddb0e53970e349e16e27ee30552d3a93c9a1c2f073156116.jpg](../iclr_results/2640_Language Imbalance Driven Rewarding for Multilingual Self-improving/tables/263cebff66d7a1e6ddb0e53970e349e16e27ee30552d3a93c9a1c2f073156116.jpg)

![2d9c5915652ecff740b13ec8eaf6960b01a30af6373c8f1611842521bfa6968e.jpg](../iclr_results/2640_Language Imbalance Driven Rewarding for Multilingual Self-improving/tables/2d9c5915652ecff740b13ec8eaf6960b01a30af6373c8f1611842521bfa6968e.jpg)

![669eda464b1fb71831f62aad55feca2b41e74aa6dcd627d987624b21fe3d9cb3.jpg](../iclr_results/2640_Language Imbalance Driven Rewarding for Multilingual Self-improving/tables/669eda464b1fb71831f62aad55feca2b41e74aa6dcd627d987624b21fe3d9cb3.jpg)

![6940107084b4fe26825b472d0a7b5229c5efe9b16aaf41eb2d8bbe96b7c25f4b.jpg](../iclr_results/2640_Language Imbalance Driven Rewarding for Multilingual Self-improving/tables/6940107084b4fe26825b472d0a7b5229c5efe9b16aaf41eb2d8bbe96b7c25f4b.jpg)

![7d88d75b2146fa4f8afd6dd500da8f51c1d9a43f2ffd758ab9c0a4e82e6a860c.jpg](../iclr_results/2640_Language Imbalance Driven Rewarding for Multilingual Self-improving/tables/7d88d75b2146fa4f8afd6dd500da8f51c1d9a43f2ffd758ab9c0a4e82e6a860c.jpg)

![84945bdf745f60d27f637cc22c3e27600b08c581ed8b59247274e4085f7f473b.jpg](../iclr_results/2640_Language Imbalance Driven Rewarding for Multilingual Self-improving/tables/84945bdf745f60d27f637cc22c3e27600b08c581ed8b59247274e4085f7f473b.jpg)

![8d3ff24cd8edb019e0b258ea34a795580386e953f627f4c2b25d5563527b2f47.jpg](../iclr_results/2640_Language Imbalance Driven Rewarding for Multilingual Self-improving/tables/8d3ff24cd8edb019e0b258ea34a795580386e953f627f4c2b25d5563527b2f47.jpg)

![9277b79cf1109ca04272f4c421f90b47b5f1dfa4f64ecd3e5bccfb221b292183.jpg](../iclr_results/2640_Language Imbalance Driven Rewarding for Multilingual Self-improving/tables/9277b79cf1109ca04272f4c421f90b47b5f1dfa4f64ecd3e5bccfb221b292183.jpg)

![99ae828d425f8ed6e5e0177159407570e0feca942ead9499967dec0b74f18c34.jpg](../iclr_results/2640_Language Imbalance Driven Rewarding for Multilingual Self-improving/tables/99ae828d425f8ed6e5e0177159407570e0feca942ead9499967dec0b74f18c34.jpg)

![b85e1c4fd1be65921d2082b9f61030c222fbbbd6628a86d1db65148bfa74fa10.jpg](../iclr_results/2640_Language Imbalance Driven Rewarding for Multilingual Self-improving/tables/b85e1c4fd1be65921d2082b9f61030c222fbbbd6628a86d1db65148bfa74fa10.jpg)

![c6d0eeb10e191c6088f9edc8599c8bd5585bb26fde5838e9b94a5bad99b503ec.jpg](../iclr_results/2640_Language Imbalance Driven Rewarding for Multilingual Self-improving/tables/c6d0eeb10e191c6088f9edc8599c8bd5585bb26fde5838e9b94a5bad99b503ec.jpg)

![d8620ce1278e6b59dc01a8f0c60ffdec8144a26a74ec2f79dd1c605b1bab44fc.jpg](../iclr_results/2640_Language Imbalance Driven Rewarding for Multilingual Self-improving/tables/d8620ce1278e6b59dc01a8f0c60ffdec8144a26a74ec2f79dd1c605b1bab44fc.jpg)

![f13d82e2b3f426fe14d1b83ade1634e1027a27283a80f1be4147a7cda02ad6bd.jpg](../iclr_results/2640_Language Imbalance Driven Rewarding for Multilingual Self-improving/tables/f13d82e2b3f426fe14d1b83ade1634e1027a27283a80f1be4147a7cda02ad6bd.jpg)

## Cauchy-Schwarz Regularizers


### Images

![296724b20e617888b8aa438b12c5aad6266e6fb089ff362c1a04e10a4cc01314.jpg](../iclr_results/2641_Cauchy-Schwarz Regularizers/images/296724b20e617888b8aa438b12c5aad6266e6fb089ff362c1a04e10a4cc01314.jpg)

![2b27a397efea18b98e5146416089b74417b89e558f45bc36097930d11845f872.jpg](../iclr_results/2641_Cauchy-Schwarz Regularizers/images/2b27a397efea18b98e5146416089b74417b89e558f45bc36097930d11845f872.jpg)

![774c82922d40560678aca048d6fa9eb7a9845be9f88e2e66c345b03d8ef8f9d1.jpg](../iclr_results/2641_Cauchy-Schwarz Regularizers/images/774c82922d40560678aca048d6fa9eb7a9845be9f88e2e66c345b03d8ef8f9d1.jpg)

![90316518d10696892ed05d5391afddfbd527682b4fd2ed9feb46ef52e91ba898.jpg](../iclr_results/2641_Cauchy-Schwarz Regularizers/images/90316518d10696892ed05d5391afddfbd527682b4fd2ed9feb46ef52e91ba898.jpg)

![93ea2d7baf7d5873965a987072a3e5218dd145bfb2346b5f59c44bbbc4bb601e.jpg](../iclr_results/2641_Cauchy-Schwarz Regularizers/images/93ea2d7baf7d5873965a987072a3e5218dd145bfb2346b5f59c44bbbc4bb601e.jpg)

![b36557abda77029d4f8432c23851b9f16c07fc548225a9ce2320482278451dba.jpg](../iclr_results/2641_Cauchy-Schwarz Regularizers/images/b36557abda77029d4f8432c23851b9f16c07fc548225a9ce2320482278451dba.jpg)

![bee7292f7e97c43aa00314f3184e6f820439a224e34bee2a75c0bd3eedd12a35.jpg](../iclr_results/2641_Cauchy-Schwarz Regularizers/images/bee7292f7e97c43aa00314f3184e6f820439a224e34bee2a75c0bd3eedd12a35.jpg)

![d9a02b85fecb491cce28fb7b8b52f499cbeef4611380a5ff022b395b1d595068.jpg](../iclr_results/2641_Cauchy-Schwarz Regularizers/images/d9a02b85fecb491cce28fb7b8b52f499cbeef4611380a5ff022b395b1d595068.jpg)

### Tables

![4df274f9db3b8e75d8f42afa8f1d14ee8ad1489f6b6e7a4f209e4d68fad59c92.jpg](../iclr_results/2641_Cauchy-Schwarz Regularizers/tables/4df274f9db3b8e75d8f42afa8f1d14ee8ad1489f6b6e7a4f209e4d68fad59c92.jpg)

![54b098743b37a5de13f95df6718b1ca2f5282ce1a1069fa948037910d965927c.jpg](../iclr_results/2641_Cauchy-Schwarz Regularizers/tables/54b098743b37a5de13f95df6718b1ca2f5282ce1a1069fa948037910d965927c.jpg)

![62c47b7f3b5400441a3c9a9754addd99c95db982eaf212240cf87171ad684f76.jpg](../iclr_results/2641_Cauchy-Schwarz Regularizers/tables/62c47b7f3b5400441a3c9a9754addd99c95db982eaf212240cf87171ad684f76.jpg)

![7a4472fe4e81d04119f75cdbbcd7bf404a14e9a84c404929c48e2427ea01f8b2.jpg](../iclr_results/2641_Cauchy-Schwarz Regularizers/tables/7a4472fe4e81d04119f75cdbbcd7bf404a14e9a84c404929c48e2427ea01f8b2.jpg)

![b6d1be5d32053d43569b90246980d300f3aef63565d77566a16884aa9d45936a.jpg](../iclr_results/2641_Cauchy-Schwarz Regularizers/tables/b6d1be5d32053d43569b90246980d300f3aef63565d77566a16884aa9d45936a.jpg)

![ba7b4671ecd625d8133378c38822ca5679d041c916972d179ea6d3a48df105e4.jpg](../iclr_results/2641_Cauchy-Schwarz Regularizers/tables/ba7b4671ecd625d8133378c38822ca5679d041c916972d179ea6d3a48df105e4.jpg)

![bc5abf31b59d72cbf69b31366712c82c5e54c8ed2f8cf4c166cc91ffe80f23e5.jpg](../iclr_results/2641_Cauchy-Schwarz Regularizers/tables/bc5abf31b59d72cbf69b31366712c82c5e54c8ed2f8cf4c166cc91ffe80f23e5.jpg)

![cfcaa03cde43b5e1fbe82c1c88121df4b1cb3cab869d06c261389a654a0e86a9.jpg](../iclr_results/2641_Cauchy-Schwarz Regularizers/tables/cfcaa03cde43b5e1fbe82c1c88121df4b1cb3cab869d06c261389a654a0e86a9.jpg)

![d2a5a491a254b263b34decd14f204a2f87da09746b3f5a5c4d4baa8e18b46d32.jpg](../iclr_results/2641_Cauchy-Schwarz Regularizers/tables/d2a5a491a254b263b34decd14f204a2f87da09746b3f5a5c4d4baa8e18b46d32.jpg)

![da6aafee68ce3146f19b1213f3d669b41e2564ad9841a31a9d845a30b777b272.jpg](../iclr_results/2641_Cauchy-Schwarz Regularizers/tables/da6aafee68ce3146f19b1213f3d669b41e2564ad9841a31a9d845a30b777b272.jpg)

## SysBench: Can LLMs Follow System Message?


### Images

![2382c24c9c1342ea5f9ae5f78afa288f0556fdb291bc152201eed93d87e4ac3e.jpg](../iclr_results/2642_SysBench_ Can LLMs Follow System Message_/images/2382c24c9c1342ea5f9ae5f78afa288f0556fdb291bc152201eed93d87e4ac3e.jpg)

![26dee83b24ca50280e77687b75b92f7c7cbd96fd76fda87fe6587466ed723532.jpg](../iclr_results/2642_SysBench_ Can LLMs Follow System Message_/images/26dee83b24ca50280e77687b75b92f7c7cbd96fd76fda87fe6587466ed723532.jpg)

![3da9495c0bcb4011b530ec6d8f8d84713722e57081e8fff84389d415aaee5e46.jpg](../iclr_results/2642_SysBench_ Can LLMs Follow System Message_/images/3da9495c0bcb4011b530ec6d8f8d84713722e57081e8fff84389d415aaee5e46.jpg)

![52db033ff7ddda6608a6e8c3f7e9b7a983bd48e506e4ef44b319bc4c7d40b14f.jpg](../iclr_results/2642_SysBench_ Can LLMs Follow System Message_/images/52db033ff7ddda6608a6e8c3f7e9b7a983bd48e506e4ef44b319bc4c7d40b14f.jpg)

![66c0902067167921432419386b0d7305f3b0c15052a0b5649341ba9235a2e90b.jpg](../iclr_results/2642_SysBench_ Can LLMs Follow System Message_/images/66c0902067167921432419386b0d7305f3b0c15052a0b5649341ba9235a2e90b.jpg)

![c20ad6dc9ba94d0e46456bbc2ff47b3b90c80e4cee3c7968944a8bf4a32585d3.jpg](../iclr_results/2642_SysBench_ Can LLMs Follow System Message_/images/c20ad6dc9ba94d0e46456bbc2ff47b3b90c80e4cee3c7968944a8bf4a32585d3.jpg)

### Tables

![23f599d070cbac21cd603ffd179ec3fe13dcbf8195991c852529d1cca18bba39.jpg](../iclr_results/2642_SysBench_ Can LLMs Follow System Message_/tables/23f599d070cbac21cd603ffd179ec3fe13dcbf8195991c852529d1cca18bba39.jpg)

![2c11b779a6b1b8a366c3476771886542e92712faea1f0475a8e2d89e1ee5e9e5.jpg](../iclr_results/2642_SysBench_ Can LLMs Follow System Message_/tables/2c11b779a6b1b8a366c3476771886542e92712faea1f0475a8e2d89e1ee5e9e5.jpg)

![4a4c2a92cf19a703176815ae0ff0b8057b9e1b2acd7fe29f5829dd49f4d6ce3d.jpg](../iclr_results/2642_SysBench_ Can LLMs Follow System Message_/tables/4a4c2a92cf19a703176815ae0ff0b8057b9e1b2acd7fe29f5829dd49f4d6ce3d.jpg)

![8df79435738e1904a6707809b8d96734796269a0e4192f0554e50d0eda3a73e9.jpg](../iclr_results/2642_SysBench_ Can LLMs Follow System Message_/tables/8df79435738e1904a6707809b8d96734796269a0e4192f0554e50d0eda3a73e9.jpg)

![bb5a69b9db18b0821783bd63a1cb1391a2f907895f2cc5e6cfa009e5e3ec2a61.jpg](../iclr_results/2642_SysBench_ Can LLMs Follow System Message_/tables/bb5a69b9db18b0821783bd63a1cb1391a2f907895f2cc5e6cfa009e5e3ec2a61.jpg)

![c918d7704ae81049014481658198049e49b8f65e9cd21e4714ce7585e291197a.jpg](../iclr_results/2642_SysBench_ Can LLMs Follow System Message_/tables/c918d7704ae81049014481658198049e49b8f65e9cd21e4714ce7585e291197a.jpg)

![df42d6b8e9ae05412617c04fa98c67a6916c8bd3fb7315e1cea035d19a8fb3d7.jpg](../iclr_results/2642_SysBench_ Can LLMs Follow System Message_/tables/df42d6b8e9ae05412617c04fa98c67a6916c8bd3fb7315e1cea035d19a8fb3d7.jpg)

![f99aa8308fa6c08b299f2279f2a446e3da70a2351296ad6aac1a1a9633bb3332.jpg](../iclr_results/2642_SysBench_ Can LLMs Follow System Message_/tables/f99aa8308fa6c08b299f2279f2a446e3da70a2351296ad6aac1a1a9633bb3332.jpg)

## Distribution-Specific Agnostic Conditional Classification With Halfspaces


### Images

![394a44897515230bc5bf699d6d4df68d38b28df6b9792ac4048357479ba82dc5.jpg](../iclr_results/2643_Distribution-Specific Agnostic Conditional Classification With Halfspaces/images/394a44897515230bc5bf699d6d4df68d38b28df6b9792ac4048357479ba82dc5.jpg)

![cf75014ea83f8d1d8ca9f92b2d955d951bde120ec4df9fc33cd494bdc3fe5fcb.jpg](../iclr_results/2643_Distribution-Specific Agnostic Conditional Classification With Halfspaces/images/cf75014ea83f8d1d8ca9f92b2d955d951bde120ec4df9fc33cd494bdc3fe5fcb.jpg)

![fda91f69a79e143f193e44c1c4af9370d018aeddbd3a9239ee9380889a244635.jpg](../iclr_results/2643_Distribution-Specific Agnostic Conditional Classification With Halfspaces/images/fda91f69a79e143f193e44c1c4af9370d018aeddbd3a9239ee9380889a244635.jpg)

### Tables

![161ea8b5a81fb5429394add1455d632d9988fd08c12a56958ecdc92a459810e6.jpg](../iclr_results/2643_Distribution-Specific Agnostic Conditional Classification With Halfspaces/tables/161ea8b5a81fb5429394add1455d632d9988fd08c12a56958ecdc92a459810e6.jpg)

![5c9985bce7e6ad12929bcd81922872050f0cd528a1c2c6c5830771f15358e8a0.jpg](../iclr_results/2643_Distribution-Specific Agnostic Conditional Classification With Halfspaces/tables/5c9985bce7e6ad12929bcd81922872050f0cd528a1c2c6c5830771f15358e8a0.jpg)

## GALA: Geometry-Aware Local Adaptive Grids for Detailed 3D Generation


### Images

![058c48ae4ae677e72771870a4566fb21078065acaf7cad5bf82c0897e54da147.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/058c48ae4ae677e72771870a4566fb21078065acaf7cad5bf82c0897e54da147.jpg)

![069dd81098f4d998ac9f9c222928bde72818a31374583e8b48ed3010fec1e757.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/069dd81098f4d998ac9f9c222928bde72818a31374583e8b48ed3010fec1e757.jpg)

![19aa0a235029235573fc60ec0551292d7534ba3734f0b5f1bda31a4b07b3e075.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/19aa0a235029235573fc60ec0551292d7534ba3734f0b5f1bda31a4b07b3e075.jpg)

![1d9c6a774ad863f53d5c267eb81f385fbfd445295f46a9d415b615ab52d67d70.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/1d9c6a774ad863f53d5c267eb81f385fbfd445295f46a9d415b615ab52d67d70.jpg)

![21eedf90ea538858d87726c9742fcc32eae7640db861ccbc23d60ec73906c639.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/21eedf90ea538858d87726c9742fcc32eae7640db861ccbc23d60ec73906c639.jpg)

![37e2c96361a392e212e6af3015ff8bd0c0c2ea2437a75efb2ba29ddae321e6cd.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/37e2c96361a392e212e6af3015ff8bd0c0c2ea2437a75efb2ba29ddae321e6cd.jpg)

![4690be8a9d9da71f377914fd478ad6aac3410f060b10022f8adf4e2cb14e16f0.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/4690be8a9d9da71f377914fd478ad6aac3410f060b10022f8adf4e2cb14e16f0.jpg)

![56849ba5a0eadaacbcb88b2de176eac53d84d28800414393343b1a3b7fd20856.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/56849ba5a0eadaacbcb88b2de176eac53d84d28800414393343b1a3b7fd20856.jpg)

![57ad2783f607072f4470e099c770e64734d74a8cf635bef3e6145db9118a0d08.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/57ad2783f607072f4470e099c770e64734d74a8cf635bef3e6145db9118a0d08.jpg)

![6997aeb9dd05397fc5ff34bc11357ddc11417c4e7e293fdf1dfa45947f140ac6.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/6997aeb9dd05397fc5ff34bc11357ddc11417c4e7e293fdf1dfa45947f140ac6.jpg)

![78e92c73739533d3d011a6a7833c0832d5ea64c6cddcc66fecdc0a48883f5180.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/78e92c73739533d3d011a6a7833c0832d5ea64c6cddcc66fecdc0a48883f5180.jpg)

![82dee8740c523087b6fbfa61aef7fe8550150bd749b9da4af40a233a1de53a48.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/82dee8740c523087b6fbfa61aef7fe8550150bd749b9da4af40a233a1de53a48.jpg)

![8b4de7d16aa8b996ee6ba44992a547e4bd6c3a02993a647a0ea3f0ab9b550433.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/8b4de7d16aa8b996ee6ba44992a547e4bd6c3a02993a647a0ea3f0ab9b550433.jpg)

![8dcd252e1a7016a3761e776ecd3813f03fef40f4c35f23c082113548df0f6ec3.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/8dcd252e1a7016a3761e776ecd3813f03fef40f4c35f23c082113548df0f6ec3.jpg)

![93a66d76f2f559ab9090b395e1a56cf3f8266b9c5c8336f8955d93610c2674bf.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/93a66d76f2f559ab9090b395e1a56cf3f8266b9c5c8336f8955d93610c2674bf.jpg)

![9d0cf0321f7b3a303319bd5f2727a6a401b553a3819ce8978ad38d0ab07c5b89.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/9d0cf0321f7b3a303319bd5f2727a6a401b553a3819ce8978ad38d0ab07c5b89.jpg)

![af5494b22f5598d6f5f4fd89a851b83a4b932166f3e98efabd2a533366cdfa0d.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/af5494b22f5598d6f5f4fd89a851b83a4b932166f3e98efabd2a533366cdfa0d.jpg)

![b2897a9eefe78890d61c50af1a653ff16bf34a579bf136b6dfbf59da6447b511.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/b2897a9eefe78890d61c50af1a653ff16bf34a579bf136b6dfbf59da6447b511.jpg)

![b440ae0f94d6642089af2620ffbb99ba41a966344b8672a06c9a01b5c00c0802.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/b440ae0f94d6642089af2620ffbb99ba41a966344b8672a06c9a01b5c00c0802.jpg)

![c122c26887fd6db9e88cacbededdad4d910984cc2847ddf0a9a430aca23a1afc.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/c122c26887fd6db9e88cacbededdad4d910984cc2847ddf0a9a430aca23a1afc.jpg)

![c91d2cdbaddb91183535a65470c3008ccbe00955f7f589a5bb1e9d179514338d.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/c91d2cdbaddb91183535a65470c3008ccbe00955f7f589a5bb1e9d179514338d.jpg)

![c9779f6318ff951a6cddb51cf62a4a83539d2120d646bb41a9c72103d4e3bef9.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/c9779f6318ff951a6cddb51cf62a4a83539d2120d646bb41a9c72103d4e3bef9.jpg)

![ce7ccf4c156be1e2470a8af6b4e8b405ec932ef7cc5c73775850dc1cef67e082.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/ce7ccf4c156be1e2470a8af6b4e8b405ec932ef7cc5c73775850dc1cef67e082.jpg)

![d3ec943c9f8419d55b05065036dfc6d394b57626587de3ed47e97483ea9f09ed.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/d3ec943c9f8419d55b05065036dfc6d394b57626587de3ed47e97483ea9f09ed.jpg)

![e7ac057b66d6420ea4990eb766b05d93c5abe3d3ec5549522e02ec77af662b00.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/e7ac057b66d6420ea4990eb766b05d93c5abe3d3ec5549522e02ec77af662b00.jpg)

![eee010b6328749db3339bbfdbbe9697478d20f2b09df26640fac6d4f7b435bcd.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/eee010b6328749db3339bbfdbbe9697478d20f2b09df26640fac6d4f7b435bcd.jpg)

![f1bcb797049b1de1360f52f250ac2107c4e304e67dcd9933bbb14c92099fffee.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/f1bcb797049b1de1360f52f250ac2107c4e304e67dcd9933bbb14c92099fffee.jpg)

![f86806bafe55d27cdd19c94a1d3a732d2390f749ef5758406f67e2c954a03292.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/images/f86806bafe55d27cdd19c94a1d3a732d2390f749ef5758406f67e2c954a03292.jpg)

### Tables

![009718ddabb9d44e5c7251757303f6c3603fe912e2bd0294e016e316c3d77681.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/tables/009718ddabb9d44e5c7251757303f6c3603fe912e2bd0294e016e316c3d77681.jpg)

![4bec2d27ff0b8ff5e76e33dfe098e8b911b967b21fe908517b5e071b075a2956.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/tables/4bec2d27ff0b8ff5e76e33dfe098e8b911b967b21fe908517b5e071b075a2956.jpg)

![b8778a0cc624ba6f37a59f4e1bdee0a871a3bd0dbf263fbbf43f88ad30df4af0.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/tables/b8778a0cc624ba6f37a59f4e1bdee0a871a3bd0dbf263fbbf43f88ad30df4af0.jpg)

![d73e0c2d592957baff0b88feb00b72f1151465e1c170a4879402eba47f48f37e.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/tables/d73e0c2d592957baff0b88feb00b72f1151465e1c170a4879402eba47f48f37e.jpg)

![eacada0e97a031b2b95afeebee3eb813d7b0b1275d9423e2579f0c46421ee535.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/tables/eacada0e97a031b2b95afeebee3eb813d7b0b1275d9423e2579f0c46421ee535.jpg)

![edec87aeca123b918bfc2168e57e8a2143ac4cd9e8ba18042743374398a8c213.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/tables/edec87aeca123b918bfc2168e57e8a2143ac4cd9e8ba18042743374398a8c213.jpg)

![f82c7b2553e4e2fcc8050de2911502cf2434d1e82f023279a644c2b29cac5e9a.jpg](../iclr_results/2644_GALA_ Geometry-Aware Local Adaptive Grids for Detailed 3D Generation/tables/f82c7b2553e4e2fcc8050de2911502cf2434d1e82f023279a644c2b29cac5e9a.jpg)

## Distilling Structural Representations into Protein Sequence Models


### Images

![31b92bbbc6bb085ec199eb6d7eed2bb96d86fd303d9f9eb6f7b64eed9e8bdc40.jpg](../iclr_results/2645_Distilling Structural Representations into Protein Sequence Models/images/31b92bbbc6bb085ec199eb6d7eed2bb96d86fd303d9f9eb6f7b64eed9e8bdc40.jpg)

![63614ae376452b83ecc6d9b9f28bb1fa7980c43ee7d3d8e54a00907bae3a4bb5.jpg](../iclr_results/2645_Distilling Structural Representations into Protein Sequence Models/images/63614ae376452b83ecc6d9b9f28bb1fa7980c43ee7d3d8e54a00907bae3a4bb5.jpg)

![8aa0ff3debf94c14c10dd2e4bd3bf095e4df38fbd9d1e4d127560e0e558e8436.jpg](../iclr_results/2645_Distilling Structural Representations into Protein Sequence Models/images/8aa0ff3debf94c14c10dd2e4bd3bf095e4df38fbd9d1e4d127560e0e558e8436.jpg)

![b290f2e07042b263856cdffdb2df58345939e9c33f98c2ed6ae4af96c0c24472.jpg](../iclr_results/2645_Distilling Structural Representations into Protein Sequence Models/images/b290f2e07042b263856cdffdb2df58345939e9c33f98c2ed6ae4af96c0c24472.jpg)

### Tables

![21c03c9069f9be7dcc7af89ca42fb0db1f668504028994acf433a217affe75f7.jpg](../iclr_results/2645_Distilling Structural Representations into Protein Sequence Models/tables/21c03c9069f9be7dcc7af89ca42fb0db1f668504028994acf433a217affe75f7.jpg)

![30a4af03b00a91474a4820ec6b2bdad6f567c0de869c36f4f3f9db391351f8fb.jpg](../iclr_results/2645_Distilling Structural Representations into Protein Sequence Models/tables/30a4af03b00a91474a4820ec6b2bdad6f567c0de869c36f4f3f9db391351f8fb.jpg)

![559c59450c1b6b17c81e9b0e14d05059ac944027b29fdbc57dca4311a307439b.jpg](../iclr_results/2645_Distilling Structural Representations into Protein Sequence Models/tables/559c59450c1b6b17c81e9b0e14d05059ac944027b29fdbc57dca4311a307439b.jpg)

![96ef7ab4d24dac97583668c3e9ffffb5b138a98edd47cddcbbae27b7b9ad9ba2.jpg](../iclr_results/2645_Distilling Structural Representations into Protein Sequence Models/tables/96ef7ab4d24dac97583668c3e9ffffb5b138a98edd47cddcbbae27b7b9ad9ba2.jpg)

## YOLO-RD: Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary


### Images

![08646e5a1986d2296f6234a2c626bea8c863ed5efa665f22cc1f76781cd7129d.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/images/08646e5a1986d2296f6234a2c626bea8c863ed5efa665f22cc1f76781cd7129d.jpg)

![11b66a81e08e4c9fda1045fbbae4389585b37ecfddfb2d7fa05fcd62ed96b31c.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/images/11b66a81e08e4c9fda1045fbbae4389585b37ecfddfb2d7fa05fcd62ed96b31c.jpg)

![144c58fa1ed569fc8b63ea26df09ec5ccccb027fb597551e9d2efe7b799f5f9f.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/images/144c58fa1ed569fc8b63ea26df09ec5ccccb027fb597551e9d2efe7b799f5f9f.jpg)

![18bc19eb9f6f428ac706480da43f8b054c9e2649f4c5211de1c50e08d618507f.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/images/18bc19eb9f6f428ac706480da43f8b054c9e2649f4c5211de1c50e08d618507f.jpg)

![5992f6619394eabddbe8912f8ea6f68c12b6c4ca1af9ec81811c09888803c53f.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/images/5992f6619394eabddbe8912f8ea6f68c12b6c4ca1af9ec81811c09888803c53f.jpg)

![7eb0f6562f53a405d43af38e4b23211e3996d4271c5d5de734531abca2687c57.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/images/7eb0f6562f53a405d43af38e4b23211e3996d4271c5d5de734531abca2687c57.jpg)

![8990e8ac20a85dbd58deccd598dac6fc20a699e3d520d928ebe762ec08e0fa1d.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/images/8990e8ac20a85dbd58deccd598dac6fc20a699e3d520d928ebe762ec08e0fa1d.jpg)

![8eb1ae244680f9ea762362b29905bc5956e485b23f0edb00c8764c6369ddf92b.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/images/8eb1ae244680f9ea762362b29905bc5956e485b23f0edb00c8764c6369ddf92b.jpg)

![c3c752d7d2859248d553ba21f362e5fcf0a752c1f7588818bb2d51019f5a644d.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/images/c3c752d7d2859248d553ba21f362e5fcf0a752c1f7588818bb2d51019f5a644d.jpg)

![dcdee1f7e58897486664e1ffad28e78749312a2b2ff083ac1a8d5ec27f52a8f1.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/images/dcdee1f7e58897486664e1ffad28e78749312a2b2ff083ac1a8d5ec27f52a8f1.jpg)

![e65be780090615e14b925655492cced8cc28a610bc0a41eae22966888093ccaf.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/images/e65be780090615e14b925655492cced8cc28a610bc0a41eae22966888093ccaf.jpg)

![ef90d2680a6b8abe08f807c7e38d5f75a25f7ed6818163c1d11cfd13d78742c4.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/images/ef90d2680a6b8abe08f807c7e38d5f75a25f7ed6818163c1d11cfd13d78742c4.jpg)

![f95294a18d16baad6d296a15031ca7e86a792b3cbf0d29d8cd3c703362da602a.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/images/f95294a18d16baad6d296a15031ca7e86a792b3cbf0d29d8cd3c703362da602a.jpg)

### Tables

![005a01f2e0b8e886baff1c732d4ed4c2f7a83a8fbebb21837dc2d520fa8894c5.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/tables/005a01f2e0b8e886baff1c732d4ed4c2f7a83a8fbebb21837dc2d520fa8894c5.jpg)

![34cb37d10b744d5afeb12ffa6a1edeee9d399584990f887f8bee1d5f991d32fc.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/tables/34cb37d10b744d5afeb12ffa6a1edeee9d399584990f887f8bee1d5f991d32fc.jpg)

![5b50ac58c272d952415a44db9f4cfb02a6c6a0b268fe6d796f31e4e0e84b3a69.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/tables/5b50ac58c272d952415a44db9f4cfb02a6c6a0b268fe6d796f31e4e0e84b3a69.jpg)

![7cb5a5b470235ac6b9b6c8f3fe5b7a8a6e94483c9b52f8c9ac9c3a5a5b3d75ed.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/tables/7cb5a5b470235ac6b9b6c8f3fe5b7a8a6e94483c9b52f8c9ac9c3a5a5b3d75ed.jpg)

![8a22dd0758135ca5316ebe39009381888af0ff3142d81f00b04ea60a8a9bd5ce.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/tables/8a22dd0758135ca5316ebe39009381888af0ff3142d81f00b04ea60a8a9bd5ce.jpg)

![b08a38d1d290591bb9ad2c0439ef958511834ed00ccf36ccab573138502ec685.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/tables/b08a38d1d290591bb9ad2c0439ef958511834ed00ccf36ccab573138502ec685.jpg)

![bac88e3ea84ec910e4655bc541b1f9c210533a27930df9735de0c950863bca67.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/tables/bac88e3ea84ec910e4655bc541b1f9c210533a27930df9735de0c950863bca67.jpg)

![bdf2365b9b09691909d4c078cb86ab96a1a3a170907584b9ea71b98201d2de79.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/tables/bdf2365b9b09691909d4c078cb86ab96a1a3a170907584b9ea71b98201d2de79.jpg)

![c43d6ecbb983e945608d0d48633f1933480e91089d3c75e7728458f0b0777e3b.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/tables/c43d6ecbb983e945608d0d48633f1933480e91089d3c75e7728458f0b0777e3b.jpg)

![dbf6b875d57233c8fb3b001c19e71c45846a95e4290a1941337f75841cc9c2c9.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/tables/dbf6b875d57233c8fb3b001c19e71c45846a95e4290a1941337f75841cc9c2c9.jpg)

![fe4b25c98a04d2b3c3dc3640806c6361125b9491b1d5dcb9b986822c760a9ed1.jpg](../iclr_results/2646_YOLO-RD_ Introducing Relevant and Compact Explicit Knowledge to YOLO by Retriever-Dictionary/tables/fe4b25c98a04d2b3c3dc3640806c6361125b9491b1d5dcb9b986822c760a9ed1.jpg)

## Computing Circuits Optimization via Model-Based Circuit Genetic Evolution


### Images

![14d40bf3816aaed0752125425646c7cb2bcae58d0f263465ba519485f4ba5ded.jpg](../iclr_results/2647_Computing Circuits Optimization via Model-Based Circuit Genetic Evolution/images/14d40bf3816aaed0752125425646c7cb2bcae58d0f263465ba519485f4ba5ded.jpg)

![337da4e35dd6f90c38b685838d26090d960be1ec8bb2178a68ce824b16570a56.jpg](../iclr_results/2647_Computing Circuits Optimization via Model-Based Circuit Genetic Evolution/images/337da4e35dd6f90c38b685838d26090d960be1ec8bb2178a68ce824b16570a56.jpg)

![679c654be1e7245f425b65524bf0f75d28dbf35ea9ba9efa69f46da391bf6911.jpg](../iclr_results/2647_Computing Circuits Optimization via Model-Based Circuit Genetic Evolution/images/679c654be1e7245f425b65524bf0f75d28dbf35ea9ba9efa69f46da391bf6911.jpg)

![6c9e4e8a821cbeb3d928fdce7f9866f8033cfe43fc9d4401b45218cec6c4d1c2.jpg](../iclr_results/2647_Computing Circuits Optimization via Model-Based Circuit Genetic Evolution/images/6c9e4e8a821cbeb3d928fdce7f9866f8033cfe43fc9d4401b45218cec6c4d1c2.jpg)

![8a393ee1a4a120bb48da0ce75c7b8f06ae60b27b8c45a98aec4628f694a523cd.jpg](../iclr_results/2647_Computing Circuits Optimization via Model-Based Circuit Genetic Evolution/images/8a393ee1a4a120bb48da0ce75c7b8f06ae60b27b8c45a98aec4628f694a523cd.jpg)

![b56c7d7e045b58c913b011a7ccbf8a10e0c7829034035811eaf5e6b9b066294c.jpg](../iclr_results/2647_Computing Circuits Optimization via Model-Based Circuit Genetic Evolution/images/b56c7d7e045b58c913b011a7ccbf8a10e0c7829034035811eaf5e6b9b066294c.jpg)

![caa6d9f7dd27c52aab7db3a4b411301ac1d788d1c954783f2eadb2657d35153c.jpg](../iclr_results/2647_Computing Circuits Optimization via Model-Based Circuit Genetic Evolution/images/caa6d9f7dd27c52aab7db3a4b411301ac1d788d1c954783f2eadb2657d35153c.jpg)

![cb24096fadf0760c945d7b57a0ce23a068176dc8db93b23fe03e2f03f4124760.jpg](../iclr_results/2647_Computing Circuits Optimization via Model-Based Circuit Genetic Evolution/images/cb24096fadf0760c945d7b57a0ce23a068176dc8db93b23fe03e2f03f4124760.jpg)

![e5edb55722e63e21e4092b5a5c9e6dcaeb39d3ccd92893c43bf0a272ea726d77.jpg](../iclr_results/2647_Computing Circuits Optimization via Model-Based Circuit Genetic Evolution/images/e5edb55722e63e21e4092b5a5c9e6dcaeb39d3ccd92893c43bf0a272ea726d77.jpg)

![fb94104e8d1e18fd83601d87d8dbc7fb48a68e0fddcc64606da1682c614e0420.jpg](../iclr_results/2647_Computing Circuits Optimization via Model-Based Circuit Genetic Evolution/images/fb94104e8d1e18fd83601d87d8dbc7fb48a68e0fddcc64606da1682c614e0420.jpg)

### Tables

![12724b63897ffb6b570cec76497eebc7c3589ec910e451f0164d6f6e5f453653.jpg](../iclr_results/2647_Computing Circuits Optimization via Model-Based Circuit Genetic Evolution/tables/12724b63897ffb6b570cec76497eebc7c3589ec910e451f0164d6f6e5f453653.jpg)

![1f996cf5286b445e8bea871a560b8867387b8b8314901fd21157b0aece581602.jpg](../iclr_results/2647_Computing Circuits Optimization via Model-Based Circuit Genetic Evolution/tables/1f996cf5286b445e8bea871a560b8867387b8b8314901fd21157b0aece581602.jpg)

![219c5c53dd6f1f87373cc65e7bf84c1ae10300574c84650c9657db6301dd0114.jpg](../iclr_results/2647_Computing Circuits Optimization via Model-Based Circuit Genetic Evolution/tables/219c5c53dd6f1f87373cc65e7bf84c1ae10300574c84650c9657db6301dd0114.jpg)

![2cbad823138f8905b6433ea2f8e7282396d117973b1725ee231d23c7810b0e9c.jpg](../iclr_results/2647_Computing Circuits Optimization via Model-Based Circuit Genetic Evolution/tables/2cbad823138f8905b6433ea2f8e7282396d117973b1725ee231d23c7810b0e9c.jpg)

![48c0fa5381c375b87aa9ade1fbc661ac2868729de7db4982478f0704fac5e791.jpg](../iclr_results/2647_Computing Circuits Optimization via Model-Based Circuit Genetic Evolution/tables/48c0fa5381c375b87aa9ade1fbc661ac2868729de7db4982478f0704fac5e791.jpg)

![5059825818bb4781a281f00666151ad61df0e3c3ae0a696f244e0456eec63b08.jpg](../iclr_results/2647_Computing Circuits Optimization via Model-Based Circuit Genetic Evolution/tables/5059825818bb4781a281f00666151ad61df0e3c3ae0a696f244e0456eec63b08.jpg)

![8e81ced88915da00148ff20a8e6351e6bbece8060bd1237392d4a347e877e0a1.jpg](../iclr_results/2647_Computing Circuits Optimization via Model-Based Circuit Genetic Evolution/tables/8e81ced88915da00148ff20a8e6351e6bbece8060bd1237392d4a347e877e0a1.jpg)

![d74aaafa380ca6ea57c8f6c0b37de5ba4ef74016a071d755732ec20d0cf233f4.jpg](../iclr_results/2647_Computing Circuits Optimization via Model-Based Circuit Genetic Evolution/tables/d74aaafa380ca6ea57c8f6c0b37de5ba4ef74016a071d755732ec20d0cf233f4.jpg)

![dda2718f5a03ee242af080affb053c18448595c68c65e51f063af668f7a5801b.jpg](../iclr_results/2647_Computing Circuits Optimization via Model-Based Circuit Genetic Evolution/tables/dda2718f5a03ee242af080affb053c18448595c68c65e51f063af668f7a5801b.jpg)

![e867c911ccca20f0c1681d04357cddf269950355e727e14cc10c9815c07a6a9e.jpg](../iclr_results/2647_Computing Circuits Optimization via Model-Based Circuit Genetic Evolution/tables/e867c911ccca20f0c1681d04357cddf269950355e727e14cc10c9815c07a6a9e.jpg)

## To Tackle Adversarial Transferability: A Novel Ensemble Training Method with Fourier Transformation


### Images

![0601b2e32727fa3cfad22c729d91113c24c688feccda346b9923719827f482a1.jpg](../iclr_results/2648_To Tackle Adversarial Transferability_ A Novel Ensemble Training Method with Fourier Transformation/images/0601b2e32727fa3cfad22c729d91113c24c688feccda346b9923719827f482a1.jpg)

![22196b44d3dd2f8a1bd2f43079f5f58b2afdb0c347ee566e69d244458f4cdf3e.jpg](../iclr_results/2648_To Tackle Adversarial Transferability_ A Novel Ensemble Training Method with Fourier Transformation/images/22196b44d3dd2f8a1bd2f43079f5f58b2afdb0c347ee566e69d244458f4cdf3e.jpg)

![222971f3f4b77a0750fb0060851f5289d23192be33f6f612c84a008a860b46f8.jpg](../iclr_results/2648_To Tackle Adversarial Transferability_ A Novel Ensemble Training Method with Fourier Transformation/images/222971f3f4b77a0750fb0060851f5289d23192be33f6f612c84a008a860b46f8.jpg)

![9e5a2254f87a373d6fea8687b6f9457f19fce5b5097fd906fd27241cae88dd1e.jpg](../iclr_results/2648_To Tackle Adversarial Transferability_ A Novel Ensemble Training Method with Fourier Transformation/images/9e5a2254f87a373d6fea8687b6f9457f19fce5b5097fd906fd27241cae88dd1e.jpg)

![de52c02f983c9e90ef07ad8402d241c4046a58a479a39aaf9f130552987447ec.jpg](../iclr_results/2648_To Tackle Adversarial Transferability_ A Novel Ensemble Training Method with Fourier Transformation/images/de52c02f983c9e90ef07ad8402d241c4046a58a479a39aaf9f130552987447ec.jpg)

![e86047d9b79511dee3fcca3c8aac49a4ab4bee7dc935de4a40b472a404a9e3d1.jpg](../iclr_results/2648_To Tackle Adversarial Transferability_ A Novel Ensemble Training Method with Fourier Transformation/images/e86047d9b79511dee3fcca3c8aac49a4ab4bee7dc935de4a40b472a404a9e3d1.jpg)

![eef2851dae1febdde3564caeb12467d34f14e92e28a5b21496633e729e7429a1.jpg](../iclr_results/2648_To Tackle Adversarial Transferability_ A Novel Ensemble Training Method with Fourier Transformation/images/eef2851dae1febdde3564caeb12467d34f14e92e28a5b21496633e729e7429a1.jpg)

![efc910412cdfd542e75fbd6f6cc47485c232c3e9b4ad5d0e6c7dea2789b5bd40.jpg](../iclr_results/2648_To Tackle Adversarial Transferability_ A Novel Ensemble Training Method with Fourier Transformation/images/efc910412cdfd542e75fbd6f6cc47485c232c3e9b4ad5d0e6c7dea2789b5bd40.jpg)

### Tables

![1d349ee396707c9694d9163ac22867dcd9bdccf903767c35c6d2133eb7782164.jpg](../iclr_results/2648_To Tackle Adversarial Transferability_ A Novel Ensemble Training Method with Fourier Transformation/tables/1d349ee396707c9694d9163ac22867dcd9bdccf903767c35c6d2133eb7782164.jpg)

![229c236fc1f0db8d2b1319a5fcb120c20f1c96eba1c0f330a07fa65fe8a8d57e.jpg](../iclr_results/2648_To Tackle Adversarial Transferability_ A Novel Ensemble Training Method with Fourier Transformation/tables/229c236fc1f0db8d2b1319a5fcb120c20f1c96eba1c0f330a07fa65fe8a8d57e.jpg)

![53715561fba4905b86fe93e31d1e06c2ef14e70b1ecd49c4ba17a206823ecb7e.jpg](../iclr_results/2648_To Tackle Adversarial Transferability_ A Novel Ensemble Training Method with Fourier Transformation/tables/53715561fba4905b86fe93e31d1e06c2ef14e70b1ecd49c4ba17a206823ecb7e.jpg)

![5860ef5edba87eeb11e4986246b121f675080e891f900121c325438af46f534a.jpg](../iclr_results/2648_To Tackle Adversarial Transferability_ A Novel Ensemble Training Method with Fourier Transformation/tables/5860ef5edba87eeb11e4986246b121f675080e891f900121c325438af46f534a.jpg)

![7712e8b85eaa8b63ee55d2c5c9e613063340fc576c2b53e84eb6fcd1bcd675d4.jpg](../iclr_results/2648_To Tackle Adversarial Transferability_ A Novel Ensemble Training Method with Fourier Transformation/tables/7712e8b85eaa8b63ee55d2c5c9e613063340fc576c2b53e84eb6fcd1bcd675d4.jpg)

![868d8401044e346a5baf8f6b2c8e1b8f20f6ab7dffc8fd53afa0554d26043704.jpg](../iclr_results/2648_To Tackle Adversarial Transferability_ A Novel Ensemble Training Method with Fourier Transformation/tables/868d8401044e346a5baf8f6b2c8e1b8f20f6ab7dffc8fd53afa0554d26043704.jpg)

![b1281ee581f110b7051ba2b88477d55e6ea3fed4d4e7dbefcec581a702143e9b.jpg](../iclr_results/2648_To Tackle Adversarial Transferability_ A Novel Ensemble Training Method with Fourier Transformation/tables/b1281ee581f110b7051ba2b88477d55e6ea3fed4d4e7dbefcec581a702143e9b.jpg)

![d7040c1b0f427d171e626172d9198abb4dfd4674caf6e900448109701e9572a3.jpg](../iclr_results/2648_To Tackle Adversarial Transferability_ A Novel Ensemble Training Method with Fourier Transformation/tables/d7040c1b0f427d171e626172d9198abb4dfd4674caf6e900448109701e9572a3.jpg)

![f22076708e40a7772b2da19ed6a422fc3f2a0bd3bf33870dbba8571b02d0265f.jpg](../iclr_results/2648_To Tackle Adversarial Transferability_ A Novel Ensemble Training Method with Fourier Transformation/tables/f22076708e40a7772b2da19ed6a422fc3f2a0bd3bf33870dbba8571b02d0265f.jpg)

![ffc834157a2d238989fbb76968b79fdea9eb602a2a084a775961fba582688512.jpg](../iclr_results/2648_To Tackle Adversarial Transferability_ A Novel Ensemble Training Method with Fourier Transformation/tables/ffc834157a2d238989fbb76968b79fdea9eb602a2a084a775961fba582688512.jpg)

## SMITE: Segment Me In TimE


### Images

![4348ba10052b1d969ba92b8f3fad6fa3c1ccca8a6f13badd75df191750db2694.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/images/4348ba10052b1d969ba92b8f3fad6fa3c1ccca8a6f13badd75df191750db2694.jpg)

![47888b7c7338618fb8bee1a30f2b59b2d3010744700c4c43d3c40880beb2d0be.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/images/47888b7c7338618fb8bee1a30f2b59b2d3010744700c4c43d3c40880beb2d0be.jpg)

![49d586a729756da6295569aacc37f007457c1a988bee1854c2ea84fd17b4ba3f.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/images/49d586a729756da6295569aacc37f007457c1a988bee1854c2ea84fd17b4ba3f.jpg)

![54edbf07440b340b9fbabedd5ffb7ebd8f933d7058f12336b78422ed3b74beec.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/images/54edbf07440b340b9fbabedd5ffb7ebd8f933d7058f12336b78422ed3b74beec.jpg)

![61efc6a9ebcf0ce525af6c968fe1199fd0e31c692944390c22d45fdb721c64ab.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/images/61efc6a9ebcf0ce525af6c968fe1199fd0e31c692944390c22d45fdb721c64ab.jpg)

![68e17b8d48cd2b08e04d1ed5f6d184dab5364e2a6ca481ec08c5be5bf34371f8.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/images/68e17b8d48cd2b08e04d1ed5f6d184dab5364e2a6ca481ec08c5be5bf34371f8.jpg)

![6b53840291e46872dae1007a6b8b8c7a9729d69f60d4c6ecbf6cb10a395a8d63.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/images/6b53840291e46872dae1007a6b8b8c7a9729d69f60d4c6ecbf6cb10a395a8d63.jpg)

![6eac03df09a4ac22be9b0586ab1b3dc0be14f2fc42d8bbb41ee20a70fc7fdf27.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/images/6eac03df09a4ac22be9b0586ab1b3dc0be14f2fc42d8bbb41ee20a70fc7fdf27.jpg)

![9e0ec121f5fc1e3acedb686d0c133f6033a7e628a890bdd61329ce48f12ca1b5.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/images/9e0ec121f5fc1e3acedb686d0c133f6033a7e628a890bdd61329ce48f12ca1b5.jpg)

![a44c89b4fccf21f9c6d4ded2e1e2b07b58a78a92a8b02146c6c5c94efb57e6a3.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/images/a44c89b4fccf21f9c6d4ded2e1e2b07b58a78a92a8b02146c6c5c94efb57e6a3.jpg)

![a967a850519bb162d05eabf128ca00dde0d539a02ee235a13c03dd58c6d5c303.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/images/a967a850519bb162d05eabf128ca00dde0d539a02ee235a13c03dd58c6d5c303.jpg)

![db03d868ec27768d0b4225d5f42aaf19b3e7a7f2694b8a809f6fdb30f8f4f742.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/images/db03d868ec27768d0b4225d5f42aaf19b3e7a7f2694b8a809f6fdb30f8f4f742.jpg)

![dd0500daa59b87fe817592238e80e5589843579638a0ca076890e4410b6b4f1f.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/images/dd0500daa59b87fe817592238e80e5589843579638a0ca076890e4410b6b4f1f.jpg)

![f861f491ec001d7a35f6c8e7a235554decf89f50288bbf3cb6baaed23c0e54d7.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/images/f861f491ec001d7a35f6c8e7a235554decf89f50288bbf3cb6baaed23c0e54d7.jpg)

### Tables

![11199f250d21a761bcc4753b4fd627a36211cd602820bf23f66b95b709073412.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/tables/11199f250d21a761bcc4753b4fd627a36211cd602820bf23f66b95b709073412.jpg)

![336cfc654133a6a78826dd6ddc2ec2dfa30ffef21cb64e267c6e538e32b7ba6e.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/tables/336cfc654133a6a78826dd6ddc2ec2dfa30ffef21cb64e267c6e538e32b7ba6e.jpg)

![43dd48a0d08fcd3df02171ca2808ef893a6309b0355ad8bbebf29a858b741a43.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/tables/43dd48a0d08fcd3df02171ca2808ef893a6309b0355ad8bbebf29a858b741a43.jpg)

![5d86d8a000f964ca01adccc0dd53451abe7e6105e5da306e889079d0eb61cef1.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/tables/5d86d8a000f964ca01adccc0dd53451abe7e6105e5da306e889079d0eb61cef1.jpg)

![986d0990ecbf173d0686ee8c0581fb1a9cb1d601910de31e06ee3a3bb6a15dc9.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/tables/986d0990ecbf173d0686ee8c0581fb1a9cb1d601910de31e06ee3a3bb6a15dc9.jpg)

![9e064b449d0fb26ad4914e0177d3cf10d8ab2d8b8a768be8ad4cc678f6e7e2e6.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/tables/9e064b449d0fb26ad4914e0177d3cf10d8ab2d8b8a768be8ad4cc678f6e7e2e6.jpg)

![a66bfda1136d7458b03c094b69fe4630d04aee875fcd7b707a37423d4215b04c.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/tables/a66bfda1136d7458b03c094b69fe4630d04aee875fcd7b707a37423d4215b04c.jpg)

![b4d24a7ad0e2430cb0ba0b12c46ee4d6a463d777ea75b0de25165dc25ead82b0.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/tables/b4d24a7ad0e2430cb0ba0b12c46ee4d6a463d777ea75b0de25165dc25ead82b0.jpg)

![b7f0203348893eeccb4bfa10dc925e07d238d48225a27f1103abee378c17e5f6.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/tables/b7f0203348893eeccb4bfa10dc925e07d238d48225a27f1103abee378c17e5f6.jpg)

![beef4c522965c4f18c00afd39193fb5691883db4d0cad660fd8efc84500918fe.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/tables/beef4c522965c4f18c00afd39193fb5691883db4d0cad660fd8efc84500918fe.jpg)

![c88e4436fc0b2d03f908fed603c21a4fa2715d74de3056cf99f67a253cd29553.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/tables/c88e4436fc0b2d03f908fed603c21a4fa2715d74de3056cf99f67a253cd29553.jpg)

![ee3c112ac3a0b01482d3b35d30affae4eb0dae4141f1f040520a1f921b6a9def.jpg](../iclr_results/2649_SMITE_ Segment Me In TimE/tables/ee3c112ac3a0b01482d3b35d30affae4eb0dae4141f1f040520a1f921b6a9def.jpg)

## Federated Granger Causality Learning For Interdependent Clients With State Space Representation


### Images

![0b2dd768ca8a3dae708e82bddcc26616ab1f208d2f6fe5d7208ad57bb764cd99.jpg](../iclr_results/2650_Federated Granger Causality Learning For Interdependent Clients With State Space Representation/images/0b2dd768ca8a3dae708e82bddcc26616ab1f208d2f6fe5d7208ad57bb764cd99.jpg)

![53129a697994f47fbf570dac57e895af05ddefdfd58bea647a6afaeece72f48a.jpg](../iclr_results/2650_Federated Granger Causality Learning For Interdependent Clients With State Space Representation/images/53129a697994f47fbf570dac57e895af05ddefdfd58bea647a6afaeece72f48a.jpg)

![c2e00c27688421bac3a5bffcc4135a3fea7e2d2084ea16aa113d7fba8a4c7801.jpg](../iclr_results/2650_Federated Granger Causality Learning For Interdependent Clients With State Space Representation/images/c2e00c27688421bac3a5bffcc4135a3fea7e2d2084ea16aa113d7fba8a4c7801.jpg)

### Tables

![0c3648e3e1cfa39e3a54f6f507535cd1ec27f76b55dec8031ac2850231cc3506.jpg](../iclr_results/2650_Federated Granger Causality Learning For Interdependent Clients With State Space Representation/tables/0c3648e3e1cfa39e3a54f6f507535cd1ec27f76b55dec8031ac2850231cc3506.jpg)

![2378e6927703b3c328a35e939b0c51f311023fbfc39d0a836846e810111585d8.jpg](../iclr_results/2650_Federated Granger Causality Learning For Interdependent Clients With State Space Representation/tables/2378e6927703b3c328a35e939b0c51f311023fbfc39d0a836846e810111585d8.jpg)

![438b7d21a1cf523455578f2422821cb442d882e46ebc417101fac90b1bbeb37a.jpg](../iclr_results/2650_Federated Granger Causality Learning For Interdependent Clients With State Space Representation/tables/438b7d21a1cf523455578f2422821cb442d882e46ebc417101fac90b1bbeb37a.jpg)

![8556bd68a0a867962d40f75c8250b5054e0894cd79a9bcf769768d3fdb77b2d4.jpg](../iclr_results/2650_Federated Granger Causality Learning For Interdependent Clients With State Space Representation/tables/8556bd68a0a867962d40f75c8250b5054e0894cd79a9bcf769768d3fdb77b2d4.jpg)

![99e90bb982fc87073ebfdbffc8c04b10322cf1f0a284944f1afde3a31a3b9114.jpg](../iclr_results/2650_Federated Granger Causality Learning For Interdependent Clients With State Space Representation/tables/99e90bb982fc87073ebfdbffc8c04b10322cf1f0a284944f1afde3a31a3b9114.jpg)

![ad012b348f230c66ebe08ce38d31a76154e96a2ef6b8dfa489ec9eb9cc6a1b4e.jpg](../iclr_results/2650_Federated Granger Causality Learning For Interdependent Clients With State Space Representation/tables/ad012b348f230c66ebe08ce38d31a76154e96a2ef6b8dfa489ec9eb9cc6a1b4e.jpg)

![d0c3654db8ccb8fb869d31f3d461eaef02e2843b685e353c9e665b1e85366afb.jpg](../iclr_results/2650_Federated Granger Causality Learning For Interdependent Clients With State Space Representation/tables/d0c3654db8ccb8fb869d31f3d461eaef02e2843b685e353c9e665b1e85366afb.jpg)

![f97305c9e55eab3d2f20df9b9c929481fdd5d9179fb5bb6c7eb4a0a52dd1bc20.jpg](../iclr_results/2650_Federated Granger Causality Learning For Interdependent Clients With State Space Representation/tables/f97305c9e55eab3d2f20df9b9c929481fdd5d9179fb5bb6c7eb4a0a52dd1bc20.jpg)

## Uncovering Latent Memories in Large Language Models


### Images

![04e112a6c9d244e4e4e96a4463a4af510800e6fa17cabb4b9e087c493df8e051.jpg](../iclr_results/2651_Uncovering Latent Memories in Large Language Models/images/04e112a6c9d244e4e4e96a4463a4af510800e6fa17cabb4b9e087c493df8e051.jpg)

![17e30a4ff9850ba04dd36db543ed1492653da6813d48fcec513b1d8e8b0f8298.jpg](../iclr_results/2651_Uncovering Latent Memories in Large Language Models/images/17e30a4ff9850ba04dd36db543ed1492653da6813d48fcec513b1d8e8b0f8298.jpg)

![4e284064f048393218e69ec286c8beb6db2903c0a395578c81fbb800b75d3ad4.jpg](../iclr_results/2651_Uncovering Latent Memories in Large Language Models/images/4e284064f048393218e69ec286c8beb6db2903c0a395578c81fbb800b75d3ad4.jpg)

![61f1e9554fc414cfb2f8a48f2c2c5f65ce4a41712af154e88140c92e9165750b.jpg](../iclr_results/2651_Uncovering Latent Memories in Large Language Models/images/61f1e9554fc414cfb2f8a48f2c2c5f65ce4a41712af154e88140c92e9165750b.jpg)

![6fa22dc85e58bf9d3c8ec720f80baca774e8fcaf530a44b8b59bed78ec584098.jpg](../iclr_results/2651_Uncovering Latent Memories in Large Language Models/images/6fa22dc85e58bf9d3c8ec720f80baca774e8fcaf530a44b8b59bed78ec584098.jpg)

![7a4ccdd50da912af48d67632962b263b919a7b6bde3ed50cd85464014e7b8ff4.jpg](../iclr_results/2651_Uncovering Latent Memories in Large Language Models/images/7a4ccdd50da912af48d67632962b263b919a7b6bde3ed50cd85464014e7b8ff4.jpg)

![7e48a440d38429f172f6c3908d973e2ec9e092549da1997f832e2d9a188f6226.jpg](../iclr_results/2651_Uncovering Latent Memories in Large Language Models/images/7e48a440d38429f172f6c3908d973e2ec9e092549da1997f832e2d9a188f6226.jpg)

![a316abdb0df04d023a812bd65752f91e00af7ff8ec373e0fdf32ac0fa59cf1fe.jpg](../iclr_results/2651_Uncovering Latent Memories in Large Language Models/images/a316abdb0df04d023a812bd65752f91e00af7ff8ec373e0fdf32ac0fa59cf1fe.jpg)

![a7574374a571996bb17cf6987216f009a004744a3b895bf3796e0f2994e75dbd.jpg](../iclr_results/2651_Uncovering Latent Memories in Large Language Models/images/a7574374a571996bb17cf6987216f009a004744a3b895bf3796e0f2994e75dbd.jpg)

![cf0637947a7bf273f9cebb7e650283e83541e3f9eda48e1ade8dfdb8f870fadb.jpg](../iclr_results/2651_Uncovering Latent Memories in Large Language Models/images/cf0637947a7bf273f9cebb7e650283e83541e3f9eda48e1ade8dfdb8f870fadb.jpg)

![d78cf7ec9eb1dedbef660bd9afaa278c1992f4b0df023307ef60904ac6ba3886.jpg](../iclr_results/2651_Uncovering Latent Memories in Large Language Models/images/d78cf7ec9eb1dedbef660bd9afaa278c1992f4b0df023307ef60904ac6ba3886.jpg)

![edfef668ec4b6ab1e97f4104349685e7834851becf5667c8fa5c680e3f1d5ba9.jpg](../iclr_results/2651_Uncovering Latent Memories in Large Language Models/images/edfef668ec4b6ab1e97f4104349685e7834851becf5667c8fa5c680e3f1d5ba9.jpg)

![f4f1c17e95cd551354b8eeb25baab587fdb2600afd79182c6b5a7b2bc7b1427b.jpg](../iclr_results/2651_Uncovering Latent Memories in Large Language Models/images/f4f1c17e95cd551354b8eeb25baab587fdb2600afd79182c6b5a7b2bc7b1427b.jpg)

### Tables

![39a6350023238639453835d6b7149f6d1b9b9d88a167ad1ba124835b5caef62e.jpg](../iclr_results/2651_Uncovering Latent Memories in Large Language Models/tables/39a6350023238639453835d6b7149f6d1b9b9d88a167ad1ba124835b5caef62e.jpg)

## CaPo: Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation


### Images

![0e1b4cadcd9a6861b7882031ae7fb351b58bc9131a063b1b803bd3edcdd079bb.jpg](../iclr_results/2652_CaPo_ Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation/images/0e1b4cadcd9a6861b7882031ae7fb351b58bc9131a063b1b803bd3edcdd079bb.jpg)

![196643253ccd14f44cfd2cfb3b3228a2b1312f6441db8d72378cdcafac1402a5.jpg](../iclr_results/2652_CaPo_ Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation/images/196643253ccd14f44cfd2cfb3b3228a2b1312f6441db8d72378cdcafac1402a5.jpg)

![2faf735e353b47b79a12bd35cb4b3b3f062ddd8ae01e85905ef466f8fc87d364.jpg](../iclr_results/2652_CaPo_ Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation/images/2faf735e353b47b79a12bd35cb4b3b3f062ddd8ae01e85905ef466f8fc87d364.jpg)

![484a469c4d9bca63d2050f6d2823aade9ca025c19578c4f34177625d4271c516.jpg](../iclr_results/2652_CaPo_ Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation/images/484a469c4d9bca63d2050f6d2823aade9ca025c19578c4f34177625d4271c516.jpg)

![6cbeabb93efb75e5db79d210c4a4362f3f09ac6a6a4bfaedea71b548f8a73a71.jpg](../iclr_results/2652_CaPo_ Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation/images/6cbeabb93efb75e5db79d210c4a4362f3f09ac6a6a4bfaedea71b548f8a73a71.jpg)

![70b2aea9504749eaca93fde0b123c3875794daf0f5d82eed3de40f45e819529e.jpg](../iclr_results/2652_CaPo_ Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation/images/70b2aea9504749eaca93fde0b123c3875794daf0f5d82eed3de40f45e819529e.jpg)

![871f3a2fab9cbabbe4242ad83641f76e04db9919fe7f0d0a3708a2215ef2925f.jpg](../iclr_results/2652_CaPo_ Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation/images/871f3a2fab9cbabbe4242ad83641f76e04db9919fe7f0d0a3708a2215ef2925f.jpg)

![9109b7ac47da7c5692f2d5aa295607c2c4955aa9d8ed9c12681995b79c078ddd.jpg](../iclr_results/2652_CaPo_ Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation/images/9109b7ac47da7c5692f2d5aa295607c2c4955aa9d8ed9c12681995b79c078ddd.jpg)

![a096b229e209538377bd192c49b8b88c1ceb788db55edcf62c428b7ca0ab6de3.jpg](../iclr_results/2652_CaPo_ Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation/images/a096b229e209538377bd192c49b8b88c1ceb788db55edcf62c428b7ca0ab6de3.jpg)

![b59dc79526ae1abcc76f4260c82945253bbad2fcd7c2b2b25c4eac89e1c2a50b.jpg](../iclr_results/2652_CaPo_ Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation/images/b59dc79526ae1abcc76f4260c82945253bbad2fcd7c2b2b25c4eac89e1c2a50b.jpg)

![b92089bc13ca9808aa8ba7391bfb4964f81ccc8c5be953a82771771a4ac68207.jpg](../iclr_results/2652_CaPo_ Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation/images/b92089bc13ca9808aa8ba7391bfb4964f81ccc8c5be953a82771771a4ac68207.jpg)

### Tables

![132d28b8fc359adf49c407d8ecb2d9e260ebb1c22bf26e016209f2b533472f41.jpg](../iclr_results/2652_CaPo_ Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation/tables/132d28b8fc359adf49c407d8ecb2d9e260ebb1c22bf26e016209f2b533472f41.jpg)

![194086f087f23b4acc9da93284dcc1f959b2434b2b79d5c05ce16df4485d356d.jpg](../iclr_results/2652_CaPo_ Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation/tables/194086f087f23b4acc9da93284dcc1f959b2434b2b79d5c05ce16df4485d356d.jpg)

![1b6a7976b6c8d57eddc0e395df6a7557b05a9cf8599a9f075d0c6aed9c59b925.jpg](../iclr_results/2652_CaPo_ Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation/tables/1b6a7976b6c8d57eddc0e395df6a7557b05a9cf8599a9f075d0c6aed9c59b925.jpg)

![2596a706c5414b06a611d021e80c4a5e6e88334c99f9f543ad757a94ddc4ffff.jpg](../iclr_results/2652_CaPo_ Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation/tables/2596a706c5414b06a611d021e80c4a5e6e88334c99f9f543ad757a94ddc4ffff.jpg)

![411e90b6552a335e823146033c9d42227489918374870f32c48bcf489b37139d.jpg](../iclr_results/2652_CaPo_ Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation/tables/411e90b6552a335e823146033c9d42227489918374870f32c48bcf489b37139d.jpg)

![6581ad0b3c2293fc670907dab2c9598cc746aff318c8ce844a91e263f0f05af9.jpg](../iclr_results/2652_CaPo_ Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation/tables/6581ad0b3c2293fc670907dab2c9598cc746aff318c8ce844a91e263f0f05af9.jpg)

![9071a0151149f8093781e578a0321800b5c339e8ee9f5920c8e499e71cefa749.jpg](../iclr_results/2652_CaPo_ Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation/tables/9071a0151149f8093781e578a0321800b5c339e8ee9f5920c8e499e71cefa749.jpg)

![a62ec9929e3e5d3493482f6f11192de03954fedb1381d71aedf4a075154d7174.jpg](../iclr_results/2652_CaPo_ Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation/tables/a62ec9929e3e5d3493482f6f11192de03954fedb1381d71aedf4a075154d7174.jpg)

![aa00f203c817b4d55354076fe7b7208305bf7d6303af14b00cfcda169c6c641e.jpg](../iclr_results/2652_CaPo_ Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation/tables/aa00f203c817b4d55354076fe7b7208305bf7d6303af14b00cfcda169c6c641e.jpg)

![c9e9da131a083457acba331c03f1c895ff355f02cf076d6c74f195d67bc66e43.jpg](../iclr_results/2652_CaPo_ Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation/tables/c9e9da131a083457acba331c03f1c895ff355f02cf076d6c74f195d67bc66e43.jpg)

![d72b0dd7494c5042dfa096fe8fa4f9f249429e2793b656a16ac68fc1a0cfa654.jpg](../iclr_results/2652_CaPo_ Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation/tables/d72b0dd7494c5042dfa096fe8fa4f9f249429e2793b656a16ac68fc1a0cfa654.jpg)

![f8aba36515594bbfe0a1bbe6f2b2d058773755603bfe747ac61ae4de83f08f4d.jpg](../iclr_results/2652_CaPo_ Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation/tables/f8aba36515594bbfe0a1bbe6f2b2d058773755603bfe747ac61ae4de83f08f4d.jpg)

## LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations


### Images

![0ecc2aaf9725fb99380f2ff4ff96d945b61b1bef47a94eee8cc1d2bb78929dad.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/images/0ecc2aaf9725fb99380f2ff4ff96d945b61b1bef47a94eee8cc1d2bb78929dad.jpg)

![2e33b9392eae2847fed535d670eb1e766cc594da0be4acbc91426f2a719d2778.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/images/2e33b9392eae2847fed535d670eb1e766cc594da0be4acbc91426f2a719d2778.jpg)

![31c1290503604d7320afc2275acc11ebe2cd3eae7cd88e6bb0d07c471aa7deee.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/images/31c1290503604d7320afc2275acc11ebe2cd3eae7cd88e6bb0d07c471aa7deee.jpg)

![37c67ed63d44124ae05f34a79bff5d312fc76a658e3e9185189fb0e0182ca99f.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/images/37c67ed63d44124ae05f34a79bff5d312fc76a658e3e9185189fb0e0182ca99f.jpg)

![5af0316df45751a0700f5a5504d2c55710d7abc54df4e924674a7d1a19dc9407.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/images/5af0316df45751a0700f5a5504d2c55710d7abc54df4e924674a7d1a19dc9407.jpg)

![657c9a448f247e6cf840d84e654e498079c9be35780a25c8d97797dec06fdf38.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/images/657c9a448f247e6cf840d84e654e498079c9be35780a25c8d97797dec06fdf38.jpg)

![7d0a7fdaeb35e12c99cdaa5380bdf72deb08385221dc7d8b1c45d71fb657c07e.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/images/7d0a7fdaeb35e12c99cdaa5380bdf72deb08385221dc7d8b1c45d71fb657c07e.jpg)

![84da9fe99161644da135ada97007fb1da9340228b0939e2f23e255a209ff9a7c.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/images/84da9fe99161644da135ada97007fb1da9340228b0939e2f23e255a209ff9a7c.jpg)

![9ce6b86aa4b7c6131bc2b83a3ac26889de08e3f51936ad53643e957b5fa945be.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/images/9ce6b86aa4b7c6131bc2b83a3ac26889de08e3f51936ad53643e957b5fa945be.jpg)

![a4eb8474def39d5f2b996a46d29e1897d16ae55b8d662ec43c909360136244e3.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/images/a4eb8474def39d5f2b996a46d29e1897d16ae55b8d662ec43c909360136244e3.jpg)

![b753822f9f2d21de0d38006dcbc4a1e8d5e5f41ef9e2d92084da9cce07e72a32.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/images/b753822f9f2d21de0d38006dcbc4a1e8d5e5f41ef9e2d92084da9cce07e72a32.jpg)

![bceb38365ecd438735caf44ae2421459da5474523d3c00553a016e8e4b389b2a.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/images/bceb38365ecd438735caf44ae2421459da5474523d3c00553a016e8e4b389b2a.jpg)

![c088bf9915960b1f404a34edb54d770674092e878357108f17693c6383b5d3c7.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/images/c088bf9915960b1f404a34edb54d770674092e878357108f17693c6383b5d3c7.jpg)

![de737029e82f233d6b3f82178b150624d92c4a53c4aa3b73d8f8d3c1d5782b83.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/images/de737029e82f233d6b3f82178b150624d92c4a53c4aa3b73d8f8d3c1d5782b83.jpg)

![f452232b1d9468eb4178f5115654722e824fbeae8c97cbca0842442a83fc9f34.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/images/f452232b1d9468eb4178f5115654722e824fbeae8c97cbca0842442a83fc9f34.jpg)

![fca43b495b01ffeaa5deadfe9c135bb459326a6d2fbd3cdc9cc7784c4c716d58.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/images/fca43b495b01ffeaa5deadfe9c135bb459326a6d2fbd3cdc9cc7784c4c716d58.jpg)

### Tables

![0bed8f1b5422a239962484e1e76e1aaf166a8428c66c24026bbf7c59d8138e50.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/tables/0bed8f1b5422a239962484e1e76e1aaf166a8428c66c24026bbf7c59d8138e50.jpg)

![212cf162065be8d41c6e608eb28de714f65e5b20b9556db8ea87018b3303a3f0.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/tables/212cf162065be8d41c6e608eb28de714f65e5b20b9556db8ea87018b3303a3f0.jpg)

![36b0108d4e2374f53d64b3c2af62914d35707a6aa23d773bf0377f5892eaed0d.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/tables/36b0108d4e2374f53d64b3c2af62914d35707a6aa23d773bf0377f5892eaed0d.jpg)

![3dcaee4cd9844e5d40a54aa48b7c37a8de7271bb768d1b60e39b38743c5e0f4d.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/tables/3dcaee4cd9844e5d40a54aa48b7c37a8de7271bb768d1b60e39b38743c5e0f4d.jpg)

![4b6106f1899c68c4c9de725ab01fb28867c0ec11e90f55262baf33f21d43aeee.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/tables/4b6106f1899c68c4c9de725ab01fb28867c0ec11e90f55262baf33f21d43aeee.jpg)

![4c08ec1997ad8ddb8547ba6d6d65c74f8c077b21883102a88fb2deadf6e1874d.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/tables/4c08ec1997ad8ddb8547ba6d6d65c74f8c077b21883102a88fb2deadf6e1874d.jpg)

![53a24276a853e0acb250ff3787c41a39b14c6e280cbeaf3f7b0de7b6add1d99d.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/tables/53a24276a853e0acb250ff3787c41a39b14c6e280cbeaf3f7b0de7b6add1d99d.jpg)

![6341168e3e1a856faa2855c91a5a8554cff7ce9cd73561e37c6b78e249cb69df.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/tables/6341168e3e1a856faa2855c91a5a8554cff7ce9cd73561e37c6b78e249cb69df.jpg)

![7f7896b5ff38fd06fca500904b48c92d2ffdfccff0092622b82259ca5a0fc1ce.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/tables/7f7896b5ff38fd06fca500904b48c92d2ffdfccff0092622b82259ca5a0fc1ce.jpg)

![8ed57f88c4a12a107f970e0e5bee3cd554bd3aea9f23a798e55ea5862655512b.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/tables/8ed57f88c4a12a107f970e0e5bee3cd554bd3aea9f23a798e55ea5862655512b.jpg)

![8f4ce23af65bb48a792651119a3508e73bc5582a16c4a3414b87d4fab5cd1494.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/tables/8f4ce23af65bb48a792651119a3508e73bc5582a16c4a3414b87d4fab5cd1494.jpg)

![b1a75cfdbf96ed3a9c99175b45ba478f90176c316c967648e642a20b705ab381.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/tables/b1a75cfdbf96ed3a9c99175b45ba478f90176c316c967648e642a20b705ab381.jpg)

![b970be2a0e46d3a481ab8d7fb517bcf4310755b42ffa340987f6326b1410ab23.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/tables/b970be2a0e46d3a481ab8d7fb517bcf4310755b42ffa340987f6326b1410ab23.jpg)

![bb5e5be165bc2e5ee29b2d2b5585a968558298d775b44cdb160b08cf89961f81.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/tables/bb5e5be165bc2e5ee29b2d2b5585a968558298d775b44cdb160b08cf89961f81.jpg)

![d7c6bbde9af4a4877573b1c2693a7981bb8a95332c7de2c2efe75c2bebc56eab.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/tables/d7c6bbde9af4a4877573b1c2693a7981bb8a95332c7de2c2efe75c2bebc56eab.jpg)

![f0278e7962e3a0ac5535e27b95be5cd5916f49f22b5bc3f073547d28de9787bc.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/tables/f0278e7962e3a0ac5535e27b95be5cd5916f49f22b5bc3f073547d28de9787bc.jpg)

![f088b677d62cfcb52dc32e1601737c003223b3a59d8f87f05be2040e702680b1.jpg](../iclr_results/2653_LLMs Know More Than They Show_ On the Intrinsic Representation of LLM Hallucinations/tables/f088b677d62cfcb52dc32e1601737c003223b3a59d8f87f05be2040e702680b1.jpg)

## Can Watermarks be Used to Detect LLM IP Infringement For Free?


### Images

![55f81d248726894a2f9a0afa092f57d3ca64f1720978feadc3a5e41f6a87a340.jpg](../iclr_results/2654_Can Watermarks be Used to Detect LLM IP Infringement For Free_/images/55f81d248726894a2f9a0afa092f57d3ca64f1720978feadc3a5e41f6a87a340.jpg)

![79278ae4de411e878fe6253c9784f476d50328ab407e120c0012269b6b4af4e8.jpg](../iclr_results/2654_Can Watermarks be Used to Detect LLM IP Infringement For Free_/images/79278ae4de411e878fe6253c9784f476d50328ab407e120c0012269b6b4af4e8.jpg)

![ab9bae7487a3b0a5a0e950322204b0433840286af9d3d348ce3ff6f669f6c2b3.jpg](../iclr_results/2654_Can Watermarks be Used to Detect LLM IP Infringement For Free_/images/ab9bae7487a3b0a5a0e950322204b0433840286af9d3d348ce3ff6f669f6c2b3.jpg)

![ae61633b17126562833fc0eb5c6c2fc9655628fa0c8e228061b3a949310a82fc.jpg](../iclr_results/2654_Can Watermarks be Used to Detect LLM IP Infringement For Free_/images/ae61633b17126562833fc0eb5c6c2fc9655628fa0c8e228061b3a949310a82fc.jpg)

![bb415d4586131687f9770274280e521a621cc4494ebb1511013f4303652de858.jpg](../iclr_results/2654_Can Watermarks be Used to Detect LLM IP Infringement For Free_/images/bb415d4586131687f9770274280e521a621cc4494ebb1511013f4303652de858.jpg)

![bf74f5b210cafa5e49e3b5528897f65a0e61f4075505cae84e313e9936f537d4.jpg](../iclr_results/2654_Can Watermarks be Used to Detect LLM IP Infringement For Free_/images/bf74f5b210cafa5e49e3b5528897f65a0e61f4075505cae84e313e9936f537d4.jpg)

![e467e51b421734b02c10488a683bc56c2adcb7a62bf062d441d0b8677ff98872.jpg](../iclr_results/2654_Can Watermarks be Used to Detect LLM IP Infringement For Free_/images/e467e51b421734b02c10488a683bc56c2adcb7a62bf062d441d0b8677ff98872.jpg)

![ea83fe80ca21b2daa3555d02d6a4e8903d1c25d8b112c84a112957e2d172c8bf.jpg](../iclr_results/2654_Can Watermarks be Used to Detect LLM IP Infringement For Free_/images/ea83fe80ca21b2daa3555d02d6a4e8903d1c25d8b112c84a112957e2d172c8bf.jpg)

### Tables

![4d267ad51702bacfa23284e463a287c85c55d8f98a5b5192efa2ce5ce62eb7ae.jpg](../iclr_results/2654_Can Watermarks be Used to Detect LLM IP Infringement For Free_/tables/4d267ad51702bacfa23284e463a287c85c55d8f98a5b5192efa2ce5ce62eb7ae.jpg)

![55e662225880178a3a0e7617984880abf43f989e0453cde37b21efdab9712136.jpg](../iclr_results/2654_Can Watermarks be Used to Detect LLM IP Infringement For Free_/tables/55e662225880178a3a0e7617984880abf43f989e0453cde37b21efdab9712136.jpg)

![680741edcc158f83ade88274e9e59bc04bf45b220b8a0383eca381f64889b962.jpg](../iclr_results/2654_Can Watermarks be Used to Detect LLM IP Infringement For Free_/tables/680741edcc158f83ade88274e9e59bc04bf45b220b8a0383eca381f64889b962.jpg)

![b6fcb44f9b849225ad523f6dd8e357cc3d36fdfc9ea961ababec28fc2cc69717.jpg](../iclr_results/2654_Can Watermarks be Used to Detect LLM IP Infringement For Free_/tables/b6fcb44f9b849225ad523f6dd8e357cc3d36fdfc9ea961ababec28fc2cc69717.jpg)

![bbad1cceb1fdab65816b97b54fd24eb59593b1ef0b0acfaf16577f64334279cf.jpg](../iclr_results/2654_Can Watermarks be Used to Detect LLM IP Infringement For Free_/tables/bbad1cceb1fdab65816b97b54fd24eb59593b1ef0b0acfaf16577f64334279cf.jpg)

![c660012bd423113aaf96abd1861303e81ee01ff4e05cea5c26441722c41e9202.jpg](../iclr_results/2654_Can Watermarks be Used to Detect LLM IP Infringement For Free_/tables/c660012bd423113aaf96abd1861303e81ee01ff4e05cea5c26441722c41e9202.jpg)

![e8608387040e560ec3fa278852914ecf9fc8b9a3c25310dfae601325ecd9ee5f.jpg](../iclr_results/2654_Can Watermarks be Used to Detect LLM IP Infringement For Free_/tables/e8608387040e560ec3fa278852914ecf9fc8b9a3c25310dfae601325ecd9ee5f.jpg)

## Timer-XL: Long-Context Transformers for Unified Time Series Forecasting


### Images

![104d7208468cd334d91369e5581d79d0ccb5aa48955afe2cf059d18a559c937e.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/images/104d7208468cd334d91369e5581d79d0ccb5aa48955afe2cf059d18a559c937e.jpg)

![3f29339240391e91f064678d5abd2476c00d06cfd8daf54b3eadab30a14b0315.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/images/3f29339240391e91f064678d5abd2476c00d06cfd8daf54b3eadab30a14b0315.jpg)

![452bb532d22dd16ebe481c17617c458471b29d982bad55dc21a22e7536eeccaf.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/images/452bb532d22dd16ebe481c17617c458471b29d982bad55dc21a22e7536eeccaf.jpg)

![5f01cdd8675e29d1b4b8dcfc13e859703a2172e5cbbd6cde9d2bca6f1eeef520.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/images/5f01cdd8675e29d1b4b8dcfc13e859703a2172e5cbbd6cde9d2bca6f1eeef520.jpg)

![6c8ef3f5739732de82f1cf3665b3936bf281931e763b148da12db8a567d40b7f.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/images/6c8ef3f5739732de82f1cf3665b3936bf281931e763b148da12db8a567d40b7f.jpg)

![7b122ab37252b27f98ef9b17ddfee78af6d868baf91c4cc7f091caf7abf89653.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/images/7b122ab37252b27f98ef9b17ddfee78af6d868baf91c4cc7f091caf7abf89653.jpg)

![84f134ff7836f3bb43e8d9120424211a0161b95189817684fbc5d22b4ad9a7cd.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/images/84f134ff7836f3bb43e8d9120424211a0161b95189817684fbc5d22b4ad9a7cd.jpg)

![928c1c6089f0d5baff5f5615ff4c2aa2e1cef1a0cc77b87c9ac28af9b4875947.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/images/928c1c6089f0d5baff5f5615ff4c2aa2e1cef1a0cc77b87c9ac28af9b4875947.jpg)

![95ebf265e1818415871d994c1ee34ad06a43011b8dbca51e92e7753cb26c6d17.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/images/95ebf265e1818415871d994c1ee34ad06a43011b8dbca51e92e7753cb26c6d17.jpg)

![96a7ec197248b8f77bf32452e27f8e941db30bf53f81800d155e781fac35d1f7.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/images/96a7ec197248b8f77bf32452e27f8e941db30bf53f81800d155e781fac35d1f7.jpg)

![978c1436d60409a2a554e5990c0d2719ed1504efc332fab75d91f1a9ecb4470d.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/images/978c1436d60409a2a554e5990c0d2719ed1504efc332fab75d91f1a9ecb4470d.jpg)

![c1577f1968ff2679d26eb47643e59057f8488e1e69e03e017d5346259f405ba4.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/images/c1577f1968ff2679d26eb47643e59057f8488e1e69e03e017d5346259f405ba4.jpg)

### Tables

![0e6ed2faf3d59e6489e248aa3ea84af478d44fd33d13b68080aeb33b5a391a8e.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/tables/0e6ed2faf3d59e6489e248aa3ea84af478d44fd33d13b68080aeb33b5a391a8e.jpg)

![0e7ee87aa12222c064754834402d2c69f7875c860ebbcad99424cf96d97b0adb.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/tables/0e7ee87aa12222c064754834402d2c69f7875c860ebbcad99424cf96d97b0adb.jpg)

![1a1fdcad3bd2c4394db2ff26e89cbf8de77d1327818f24369e54c9b83baf730e.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/tables/1a1fdcad3bd2c4394db2ff26e89cbf8de77d1327818f24369e54c9b83baf730e.jpg)

![3d86ace3ac506154c303eeb0f37ac908fea236fa088773e2756b9d2385b12821.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/tables/3d86ace3ac506154c303eeb0f37ac908fea236fa088773e2756b9d2385b12821.jpg)

![4b7b0ddf2641fa2ff2b4d38bd2cb23d1afa70e2b84d15b30cdf173c52da59297.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/tables/4b7b0ddf2641fa2ff2b4d38bd2cb23d1afa70e2b84d15b30cdf173c52da59297.jpg)

![4d4b82bd9b99a0f24bece051458b35e3cbbf0cfbe93ef70a8373d9eabc85d013.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/tables/4d4b82bd9b99a0f24bece051458b35e3cbbf0cfbe93ef70a8373d9eabc85d013.jpg)

![5c901bbd5d7598eefef682644b3349910c1dfaa4710d0bd6bbcc242da5215c76.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/tables/5c901bbd5d7598eefef682644b3349910c1dfaa4710d0bd6bbcc242da5215c76.jpg)

![7a4b400eeabc0a214067143a4f8771b5099948d48e2ae7f026e3e132c0d6c87e.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/tables/7a4b400eeabc0a214067143a4f8771b5099948d48e2ae7f026e3e132c0d6c87e.jpg)

![8b16dee66ac6ccee7993c65d7790f64d3a494f241e4396908ff0c0c609ada7ad.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/tables/8b16dee66ac6ccee7993c65d7790f64d3a494f241e4396908ff0c0c609ada7ad.jpg)

![9d46ee8074d092592819e62749f11b9d8072b0a76c0b84a30b9c3b3f4d80144c.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/tables/9d46ee8074d092592819e62749f11b9d8072b0a76c0b84a30b9c3b3f4d80144c.jpg)

![c7ddd2647c1bfcf83e634040af00fe701714c861f1c2b6a92bb89cbf0a3b89c7.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/tables/c7ddd2647c1bfcf83e634040af00fe701714c861f1c2b6a92bb89cbf0a3b89c7.jpg)

![ca4b3e9c865c3f9aa93401b5bbce122a8c09c8a601076a63aa52406489149a02.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/tables/ca4b3e9c865c3f9aa93401b5bbce122a8c09c8a601076a63aa52406489149a02.jpg)

![d8224321db01be00a73489d01c90e23d284a2baea004bbbdd4b2e1f4b0f41125.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/tables/d8224321db01be00a73489d01c90e23d284a2baea004bbbdd4b2e1f4b0f41125.jpg)

![ef8fe25afe515d49b9c96d1dcce2f1792332e3df01f413c7bc6802d06697cecd.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/tables/ef8fe25afe515d49b9c96d1dcce2f1792332e3df01f413c7bc6802d06697cecd.jpg)

![fc518f1897ded66ae320603f0b2f4677ffdd38f670566e69a42ca61a1f63328f.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/tables/fc518f1897ded66ae320603f0b2f4677ffdd38f670566e69a42ca61a1f63328f.jpg)

![ffeaf6f8fcbae62c1a589e83fcbe642d2fcd212d2801cb68f7fc96588eb43e69.jpg](../iclr_results/2655_Timer-XL_ Long-Context Transformers for Unified Time Series Forecasting/tables/ffeaf6f8fcbae62c1a589e83fcbe642d2fcd212d2801cb68f7fc96588eb43e69.jpg)

## Improving Semantic Understanding in Speech Language Models via Brain-tuning


### Images

![01e4f07928bae4f4f3e40831d60f1836890ed567bcd6bea3747b0f8c2cf37996.jpg](../iclr_results/2656_Improving Semantic Understanding in Speech Language Models via Brain-tuning/images/01e4f07928bae4f4f3e40831d60f1836890ed567bcd6bea3747b0f8c2cf37996.jpg)

![082268885e4d061d5ab0444bcbfaa0c5aa5501e36c0acb8d75ecc0dfe882d5a9.jpg](../iclr_results/2656_Improving Semantic Understanding in Speech Language Models via Brain-tuning/images/082268885e4d061d5ab0444bcbfaa0c5aa5501e36c0acb8d75ecc0dfe882d5a9.jpg)

![296c7cad60f79e36c067f32200a0af7513bba25881828570a4118317bd3c5dd6.jpg](../iclr_results/2656_Improving Semantic Understanding in Speech Language Models via Brain-tuning/images/296c7cad60f79e36c067f32200a0af7513bba25881828570a4118317bd3c5dd6.jpg)

![357e23a5b591d0324f9982bea3925551a852e12574e2edc640a8b1a184c28db6.jpg](../iclr_results/2656_Improving Semantic Understanding in Speech Language Models via Brain-tuning/images/357e23a5b591d0324f9982bea3925551a852e12574e2edc640a8b1a184c28db6.jpg)

![47a541eae40b00bb1d20dd3dc35ffb80ffccf39dc4dc545f92a6ba3373bc39fa.jpg](../iclr_results/2656_Improving Semantic Understanding in Speech Language Models via Brain-tuning/images/47a541eae40b00bb1d20dd3dc35ffb80ffccf39dc4dc545f92a6ba3373bc39fa.jpg)

![4e204490feee8dccc76ea523c7e8b23dec52d8dd7759bb67449e62e8b7e16d76.jpg](../iclr_results/2656_Improving Semantic Understanding in Speech Language Models via Brain-tuning/images/4e204490feee8dccc76ea523c7e8b23dec52d8dd7759bb67449e62e8b7e16d76.jpg)

![51e89212253e09bde3a2983a99223b015ad14b3e6dbbbb1ea08afa6c652a04ba.jpg](../iclr_results/2656_Improving Semantic Understanding in Speech Language Models via Brain-tuning/images/51e89212253e09bde3a2983a99223b015ad14b3e6dbbbb1ea08afa6c652a04ba.jpg)

![583f790c0e6e7b966ec7cd0a7b14f6cc6627062851e51b3f9d2efb07fb6cfa43.jpg](../iclr_results/2656_Improving Semantic Understanding in Speech Language Models via Brain-tuning/images/583f790c0e6e7b966ec7cd0a7b14f6cc6627062851e51b3f9d2efb07fb6cfa43.jpg)

![633ffa88a65fbfc57559c8bcd380aa8760805d8e24fc210010e07c31d841eac2.jpg](../iclr_results/2656_Improving Semantic Understanding in Speech Language Models via Brain-tuning/images/633ffa88a65fbfc57559c8bcd380aa8760805d8e24fc210010e07c31d841eac2.jpg)

![78b1deae0b133874f8200f4f799a046dfecc8a16b19940199bfc87ab465389a8.jpg](../iclr_results/2656_Improving Semantic Understanding in Speech Language Models via Brain-tuning/images/78b1deae0b133874f8200f4f799a046dfecc8a16b19940199bfc87ab465389a8.jpg)

![7a0dbde23ceb4a7469455b600f07bae98da2e62a8637e972a3d4f9a12f1b0f20.jpg](../iclr_results/2656_Improving Semantic Understanding in Speech Language Models via Brain-tuning/images/7a0dbde23ceb4a7469455b600f07bae98da2e62a8637e972a3d4f9a12f1b0f20.jpg)

![8413257533978fe1fc94668645c3c3dbda0f3f3708ed29bb69f3e7a4c5d94524.jpg](../iclr_results/2656_Improving Semantic Understanding in Speech Language Models via Brain-tuning/images/8413257533978fe1fc94668645c3c3dbda0f3f3708ed29bb69f3e7a4c5d94524.jpg)

![85d2d3c95d3fa1274a54b4ae956f521cfc8737d4852a2b44e31c703cc8c9944a.jpg](../iclr_results/2656_Improving Semantic Understanding in Speech Language Models via Brain-tuning/images/85d2d3c95d3fa1274a54b4ae956f521cfc8737d4852a2b44e31c703cc8c9944a.jpg)

![8f37ac24c31c81d206ac763078f463715e93ac77a05c7ff857690dffb97497f0.jpg](../iclr_results/2656_Improving Semantic Understanding in Speech Language Models via Brain-tuning/images/8f37ac24c31c81d206ac763078f463715e93ac77a05c7ff857690dffb97497f0.jpg)

![9591e91648ceeccd9f56a5d4e7fd195375d69905bf109e8ac15957e53a80899a.jpg](../iclr_results/2656_Improving Semantic Understanding in Speech Language Models via Brain-tuning/images/9591e91648ceeccd9f56a5d4e7fd195375d69905bf109e8ac15957e53a80899a.jpg)

![97283d81f68de1192278b9691c45ae3a9690d8d0f8c8f7daec41864c2f02771b.jpg](../iclr_results/2656_Improving Semantic Understanding in Speech Language Models via Brain-tuning/images/97283d81f68de1192278b9691c45ae3a9690d8d0f8c8f7daec41864c2f02771b.jpg)

![a5eebba5b5ee390bf01db50487425fc3103405184d5bdbc294491ce9d1d2bbdd.jpg](../iclr_results/2656_Improving Semantic Understanding in Speech Language Models via Brain-tuning/images/a5eebba5b5ee390bf01db50487425fc3103405184d5bdbc294491ce9d1d2bbdd.jpg)

![a63fc85e546359016c88ef9c5e1cf7badce822c71fd5b960b25e61f12dcac1a0.jpg](../iclr_results/2656_Improving Semantic Understanding in Speech Language Models via Brain-tuning/images/a63fc85e546359016c88ef9c5e1cf7badce822c71fd5b960b25e61f12dcac1a0.jpg)

![d2ff17bac01a0c964bddd056d35c38615fc7c50f1997b888c5bcdf6c975d41fd.jpg](../iclr_results/2656_Improving Semantic Understanding in Speech Language Models via Brain-tuning/images/d2ff17bac01a0c964bddd056d35c38615fc7c50f1997b888c5bcdf6c975d41fd.jpg)

![f43ba38fa08627126d5514bbfbfac5e7576f9dc5b542302ba7d78e5f9ca843f8.jpg](../iclr_results/2656_Improving Semantic Understanding in Speech Language Models via Brain-tuning/images/f43ba38fa08627126d5514bbfbfac5e7576f9dc5b542302ba7d78e5f9ca843f8.jpg)

![fa3919b183afd8356aea835a54663c652f073c8fec3a9311df431bd2ca1085de.jpg](../iclr_results/2656_Improving Semantic Understanding in Speech Language Models via Brain-tuning/images/fa3919b183afd8356aea835a54663c652f073c8fec3a9311df431bd2ca1085de.jpg)

![fb7a172e6901beb99f8a7b876f3fa8e2570562fd1d367a597fef6fb615e989e9.jpg](../iclr_results/2656_Improving Semantic Understanding in Speech Language Models via Brain-tuning/images/fb7a172e6901beb99f8a7b876f3fa8e2570562fd1d367a597fef6fb615e989e9.jpg)

## MambaQuant: Quantizing the Mamba Family with Variance Aligned Rotation Methods


### Images

![22b0c3400fc3021e77e0ee1c046f2c57a674842cb6a88cbcc63c9320b29d3ff7.jpg](../iclr_results/2657_MambaQuant_ Quantizing the Mamba Family with Variance Aligned Rotation Methods/images/22b0c3400fc3021e77e0ee1c046f2c57a674842cb6a88cbcc63c9320b29d3ff7.jpg)

![2a7ea2cb5ecb1b2c422ae9db84eb6da9c07c58ea0faf4816ceca81ee083103c9.jpg](../iclr_results/2657_MambaQuant_ Quantizing the Mamba Family with Variance Aligned Rotation Methods/images/2a7ea2cb5ecb1b2c422ae9db84eb6da9c07c58ea0faf4816ceca81ee083103c9.jpg)

![5ce8285f1b3e46df656e0a061dc35d4d33134c56deb6020f0dfff4c01481417f.jpg](../iclr_results/2657_MambaQuant_ Quantizing the Mamba Family with Variance Aligned Rotation Methods/images/5ce8285f1b3e46df656e0a061dc35d4d33134c56deb6020f0dfff4c01481417f.jpg)

![63e54bd400a9eed67a6b35abcc314a2b83fe9a42863592c0a28529d0d799a0b3.jpg](../iclr_results/2657_MambaQuant_ Quantizing the Mamba Family with Variance Aligned Rotation Methods/images/63e54bd400a9eed67a6b35abcc314a2b83fe9a42863592c0a28529d0d799a0b3.jpg)

![672c8125521cd8ea68e64f08e09d76a6644a2dee5f0641aa7773db249a0e88f9.jpg](../iclr_results/2657_MambaQuant_ Quantizing the Mamba Family with Variance Aligned Rotation Methods/images/672c8125521cd8ea68e64f08e09d76a6644a2dee5f0641aa7773db249a0e88f9.jpg)

![6ad1eb4aced6adf160c147369f50148ab5a4b3eb7c0465e993183a1cc3de7bf8.jpg](../iclr_results/2657_MambaQuant_ Quantizing the Mamba Family with Variance Aligned Rotation Methods/images/6ad1eb4aced6adf160c147369f50148ab5a4b3eb7c0465e993183a1cc3de7bf8.jpg)

![6e3ee6bc7ac217c64f8b332266b27c65103580c834c05530d195d37617770b5b.jpg](../iclr_results/2657_MambaQuant_ Quantizing the Mamba Family with Variance Aligned Rotation Methods/images/6e3ee6bc7ac217c64f8b332266b27c65103580c834c05530d195d37617770b5b.jpg)

![70ae8682984da6b44b9d6b34d3d7b4f7cf5fc1abd2823bbf34c833519f64c804.jpg](../iclr_results/2657_MambaQuant_ Quantizing the Mamba Family with Variance Aligned Rotation Methods/images/70ae8682984da6b44b9d6b34d3d7b4f7cf5fc1abd2823bbf34c833519f64c804.jpg)

![811c1dc98bad334a82addc54dc1346dfc747887f543003a66531cca48c26768b.jpg](../iclr_results/2657_MambaQuant_ Quantizing the Mamba Family with Variance Aligned Rotation Methods/images/811c1dc98bad334a82addc54dc1346dfc747887f543003a66531cca48c26768b.jpg)

![a56140ea8fc781b657148f07fbf68cca286897d2cd5a4a373493ad3ffab241c0.jpg](../iclr_results/2657_MambaQuant_ Quantizing the Mamba Family with Variance Aligned Rotation Methods/images/a56140ea8fc781b657148f07fbf68cca286897d2cd5a4a373493ad3ffab241c0.jpg)

![aa041a4e6bbde3d2e80bb952673686ca7d53d873162c452adbba69fa63920e95.jpg](../iclr_results/2657_MambaQuant_ Quantizing the Mamba Family with Variance Aligned Rotation Methods/images/aa041a4e6bbde3d2e80bb952673686ca7d53d873162c452adbba69fa63920e95.jpg)

![dd498e266e96d133a8a240ff0fd12e7ea8b79f535b53991f4e6adfba3fad37f4.jpg](../iclr_results/2657_MambaQuant_ Quantizing the Mamba Family with Variance Aligned Rotation Methods/images/dd498e266e96d133a8a240ff0fd12e7ea8b79f535b53991f4e6adfba3fad37f4.jpg)

![e996bdbe9753ec590062ca5ac8d3ee68f9236e2bed44d6803ff7228e3d893c8c.jpg](../iclr_results/2657_MambaQuant_ Quantizing the Mamba Family with Variance Aligned Rotation Methods/images/e996bdbe9753ec590062ca5ac8d3ee68f9236e2bed44d6803ff7228e3d893c8c.jpg)

### Tables

![275528182a65d4faf285dd0ff2689984626ddc0c223a205e733574e8f4943b8d.jpg](../iclr_results/2657_MambaQuant_ Quantizing the Mamba Family with Variance Aligned Rotation Methods/tables/275528182a65d4faf285dd0ff2689984626ddc0c223a205e733574e8f4943b8d.jpg)

![68f8aa838cf4dd595353977f52c91cf74fee8329e2652f674f2f2d05750b32e0.jpg](../iclr_results/2657_MambaQuant_ Quantizing the Mamba Family with Variance Aligned Rotation Methods/tables/68f8aa838cf4dd595353977f52c91cf74fee8329e2652f674f2f2d05750b32e0.jpg)

![9f7e73098a37516ce4511967cafe957a8f6b4a2ebc4d0c804775410c8a03d146.jpg](../iclr_results/2657_MambaQuant_ Quantizing the Mamba Family with Variance Aligned Rotation Methods/tables/9f7e73098a37516ce4511967cafe957a8f6b4a2ebc4d0c804775410c8a03d146.jpg)

![a3bdacc1e6819a8d1213262566f1cdfc9be844b8daaa288a1d78da5386c54af2.jpg](../iclr_results/2657_MambaQuant_ Quantizing the Mamba Family with Variance Aligned Rotation Methods/tables/a3bdacc1e6819a8d1213262566f1cdfc9be844b8daaa288a1d78da5386c54af2.jpg)

![f84c59059a3d63418abe8309bf9931828f8576d3ef96b596bd49411980669fd8.jpg](../iclr_results/2657_MambaQuant_ Quantizing the Mamba Family with Variance Aligned Rotation Methods/tables/f84c59059a3d63418abe8309bf9931828f8576d3ef96b596bd49411980669fd8.jpg)

![fb32a5ee853614676f25d692bbd5daed971739a3617bc0c300c169eb2b67eb57.jpg](../iclr_results/2657_MambaQuant_ Quantizing the Mamba Family with Variance Aligned Rotation Methods/tables/fb32a5ee853614676f25d692bbd5daed971739a3617bc0c300c169eb2b67eb57.jpg)

## Training-free Camera Control for Video Generation


### Images

![17811e48f40c3eaef7de80c11483de9f4fde5d0fa2c41a49b4a7cd7d0658a80a.jpg](../iclr_results/2658_Training-free Camera Control for Video Generation/images/17811e48f40c3eaef7de80c11483de9f4fde5d0fa2c41a49b4a7cd7d0658a80a.jpg)

![1b3ec0d3309f4ffd5e0913972b6a16f0163b48d1d405dde3b2a5bc90bf3c15c9.jpg](../iclr_results/2658_Training-free Camera Control for Video Generation/images/1b3ec0d3309f4ffd5e0913972b6a16f0163b48d1d405dde3b2a5bc90bf3c15c9.jpg)

![4810b9b06ae2fdb7e0228f49ac5b9a1bf4e17cdf8525f14126f4d4e2ad806548.jpg](../iclr_results/2658_Training-free Camera Control for Video Generation/images/4810b9b06ae2fdb7e0228f49ac5b9a1bf4e17cdf8525f14126f4d4e2ad806548.jpg)

![64d75561488adbba8b9e519488d97e3c4858d6119bdb8858c05b858b1b497a85.jpg](../iclr_results/2658_Training-free Camera Control for Video Generation/images/64d75561488adbba8b9e519488d97e3c4858d6119bdb8858c05b858b1b497a85.jpg)

![8b7f0bc65ca1d252493fdc62a75ab794d7215c9316bb8b9d351b554f5d261405.jpg](../iclr_results/2658_Training-free Camera Control for Video Generation/images/8b7f0bc65ca1d252493fdc62a75ab794d7215c9316bb8b9d351b554f5d261405.jpg)

![8d60543b8dc1b1cc67650b6e9852647661781486add48fa2bf48d18aca4a150a.jpg](../iclr_results/2658_Training-free Camera Control for Video Generation/images/8d60543b8dc1b1cc67650b6e9852647661781486add48fa2bf48d18aca4a150a.jpg)

![9ca026d491f0c937f52e9082a61fc1e1d213b82ab27706db3e441175de2a3f07.jpg](../iclr_results/2658_Training-free Camera Control for Video Generation/images/9ca026d491f0c937f52e9082a61fc1e1d213b82ab27706db3e441175de2a3f07.jpg)

![ab4f9e71cfb895d8597d5cf33256b0ddceff15976ff648be1620918fb6918027.jpg](../iclr_results/2658_Training-free Camera Control for Video Generation/images/ab4f9e71cfb895d8597d5cf33256b0ddceff15976ff648be1620918fb6918027.jpg)

![ca9735ac46a78260ea336beb5dada4e630fd86484e64bedf33b53d79a7cac2be.jpg](../iclr_results/2658_Training-free Camera Control for Video Generation/images/ca9735ac46a78260ea336beb5dada4e630fd86484e64bedf33b53d79a7cac2be.jpg)

![cd1881f4cdc8b0de6734102b5f1a04575718a545bcfd10484171667dd7309d78.jpg](../iclr_results/2658_Training-free Camera Control for Video Generation/images/cd1881f4cdc8b0de6734102b5f1a04575718a545bcfd10484171667dd7309d78.jpg)

### Tables

![1ed9583d85c1558155a61d586428e749b1f215c991f96b9d245d7283de106a99.jpg](../iclr_results/2658_Training-free Camera Control for Video Generation/tables/1ed9583d85c1558155a61d586428e749b1f215c991f96b9d245d7283de106a99.jpg)

![218e49116bac34ae1090e0388d2d89020226feb6949fc5984d3e60671ae9b7c7.jpg](../iclr_results/2658_Training-free Camera Control for Video Generation/tables/218e49116bac34ae1090e0388d2d89020226feb6949fc5984d3e60671ae9b7c7.jpg)

![bf68fed3d2cf72ac6b2a508d865507644924716a99e5853e6c8388d39174d1c1.jpg](../iclr_results/2658_Training-free Camera Control for Video Generation/tables/bf68fed3d2cf72ac6b2a508d865507644924716a99e5853e6c8388d39174d1c1.jpg)

![c9f09e5a157dd426c6f01d4f457eb1eb2625428b7715296b8656604b0ee1eb7c.jpg](../iclr_results/2658_Training-free Camera Control for Video Generation/tables/c9f09e5a157dd426c6f01d4f457eb1eb2625428b7715296b8656604b0ee1eb7c.jpg)

![e264fdcf4252586d75e8fae620cdb05a6b7cd4bac28cc3beb0f49310f059ca23.jpg](../iclr_results/2658_Training-free Camera Control for Video Generation/tables/e264fdcf4252586d75e8fae620cdb05a6b7cd4bac28cc3beb0f49310f059ca23.jpg)

## EVA: Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow


### Images

![182745c0a44947fadab6d06cfa257412bd8947f0fc0e3da3a5c0921b00f2a47e.jpg](../iclr_results/2659_EVA_ Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow/images/182745c0a44947fadab6d06cfa257412bd8947f0fc0e3da3a5c0921b00f2a47e.jpg)

![2aeae131018b172301a6ffca19e9cf7f196998c90fddf0c6df3422b90ea562b1.jpg](../iclr_results/2659_EVA_ Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow/images/2aeae131018b172301a6ffca19e9cf7f196998c90fddf0c6df3422b90ea562b1.jpg)

![2f766d7a3779a2dfc6a3fc33d40249c4e1855cbd22247cd60ccc7aac063b153b.jpg](../iclr_results/2659_EVA_ Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow/images/2f766d7a3779a2dfc6a3fc33d40249c4e1855cbd22247cd60ccc7aac063b153b.jpg)

![6a17e30dc522d551cbb3b5c38ba54403b9af3ab81125d606e08ce5a199f1fa8c.jpg](../iclr_results/2659_EVA_ Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow/images/6a17e30dc522d551cbb3b5c38ba54403b9af3ab81125d606e08ce5a199f1fa8c.jpg)

![a26448ed44cf1bcb2e41c857fbc13c6840f0be0d196df46361d326f8a66324a0.jpg](../iclr_results/2659_EVA_ Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow/images/a26448ed44cf1bcb2e41c857fbc13c6840f0be0d196df46361d326f8a66324a0.jpg)

![a31555f6050233c7697ed0e6c7e4a4edfc2b276bd0c5a194eb767ac4440feb0f.jpg](../iclr_results/2659_EVA_ Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow/images/a31555f6050233c7697ed0e6c7e4a4edfc2b276bd0c5a194eb767ac4440feb0f.jpg)

![bee2f630ee5c12c6f7377bb97ba500736518cc65e985ed80d7dff00c7833148d.jpg](../iclr_results/2659_EVA_ Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow/images/bee2f630ee5c12c6f7377bb97ba500736518cc65e985ed80d7dff00c7833148d.jpg)

![f4fa73e6aa32364979deb7fb2a6e335ba249d33d41f1264370b63d5a9484473d.jpg](../iclr_results/2659_EVA_ Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow/images/f4fa73e6aa32364979deb7fb2a6e335ba249d33d41f1264370b63d5a9484473d.jpg)

![fcc077a564ba378230c1282b1a24abaa6ff8523030d13c2344dc02dfe4617bd6.jpg](../iclr_results/2659_EVA_ Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow/images/fcc077a564ba378230c1282b1a24abaa6ff8523030d13c2344dc02dfe4617bd6.jpg)

### Tables

![10a90821830d5dd8277faa24486f9c61aac9bf13f0822bbe4a41fd57bc8c3c35.jpg](../iclr_results/2659_EVA_ Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow/tables/10a90821830d5dd8277faa24486f9c61aac9bf13f0822bbe4a41fd57bc8c3c35.jpg)

![2df2d787538699acd3e20801496ffa68c4c1e9073af0714df36bdba1bc63b4d9.jpg](../iclr_results/2659_EVA_ Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow/tables/2df2d787538699acd3e20801496ffa68c4c1e9073af0714df36bdba1bc63b4d9.jpg)

![2e5f5011e2f224df33d5dd38e313535a686dd135d3e61756620814430145f795.jpg](../iclr_results/2659_EVA_ Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow/tables/2e5f5011e2f224df33d5dd38e313535a686dd135d3e61756620814430145f795.jpg)

![36a46904e6bae465b1b722976ae794071d2d9421263f4840f0a7777c7bf92654.jpg](../iclr_results/2659_EVA_ Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow/tables/36a46904e6bae465b1b722976ae794071d2d9421263f4840f0a7777c7bf92654.jpg)

![39c5c76732f1d23d87292a6ac89dbc84c484edd587c8e08cfb8d39aac2275c7e.jpg](../iclr_results/2659_EVA_ Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow/tables/39c5c76732f1d23d87292a6ac89dbc84c484edd587c8e08cfb8d39aac2275c7e.jpg)

![6e491a86ffb79b82c36eacfc79333f278a7d832866534da72dd2bc32fa08b344.jpg](../iclr_results/2659_EVA_ Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow/tables/6e491a86ffb79b82c36eacfc79333f278a7d832866534da72dd2bc32fa08b344.jpg)

![89f9d7361bbe64c30551c14cf7e921b0cc4a258a44eaea5714ac91bb1c5f1370.jpg](../iclr_results/2659_EVA_ Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow/tables/89f9d7361bbe64c30551c14cf7e921b0cc4a258a44eaea5714ac91bb1c5f1370.jpg)

![96c9c43002cf94e7299f657cc8bb92bf9062c130af272ba4dbb8b234ee8a980a.jpg](../iclr_results/2659_EVA_ Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow/tables/96c9c43002cf94e7299f657cc8bb92bf9062c130af272ba4dbb8b234ee8a980a.jpg)

![a2ead0e45b7dca49f288f94a5e2433b21affc0cd8dcad12774c55a917aa71b37.jpg](../iclr_results/2659_EVA_ Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow/tables/a2ead0e45b7dca49f288f94a5e2433b21affc0cd8dcad12774c55a917aa71b37.jpg)

![ac60689d11d67a7eb9e699f55f942f4668d739c5ba8343a300a31a7584460115.jpg](../iclr_results/2659_EVA_ Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow/tables/ac60689d11d67a7eb9e699f55f942f4668d739c5ba8343a300a31a7584460115.jpg)

![cf285f973888f7cfbdf77d99eb05101cfe8778aaa64d6bc7fd0906e0b5901701.jpg](../iclr_results/2659_EVA_ Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow/tables/cf285f973888f7cfbdf77d99eb05101cfe8778aaa64d6bc7fd0906e0b5901701.jpg)

![e391b74d07ad9b034ad8e3bfcf6b0f41726604f6308e2e6cb2698d5d46388029.jpg](../iclr_results/2659_EVA_ Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow/tables/e391b74d07ad9b034ad8e3bfcf6b0f41726604f6308e2e6cb2698d5d46388029.jpg)

![e502fbc62a7001916cfef78b6ab920277821ad97b0bda938e8fcacffcc6139cc.jpg](../iclr_results/2659_EVA_ Geometric Inverse Design for Fast Protein Motif-Scaffolding with Coupled Flow/tables/e502fbc62a7001916cfef78b6ab920277821ad97b0bda938e8fcacffcc6139cc.jpg)

## MeshAnything: Artist-Created Mesh Generation with Autoregressive Transformers


### Images

![087667fdc0c74d975a064e89f80a390239abe8b0c9c2d275515deae3a891f382.jpg](../iclr_results/2660_MeshAnything_ Artist-Created Mesh Generation with Autoregressive Transformers/images/087667fdc0c74d975a064e89f80a390239abe8b0c9c2d275515deae3a891f382.jpg)

![0fb089cac20dd7d8bc239888e5c20ec6e6b4931430ad1ec7d48cd8270ad0af58.jpg](../iclr_results/2660_MeshAnything_ Artist-Created Mesh Generation with Autoregressive Transformers/images/0fb089cac20dd7d8bc239888e5c20ec6e6b4931430ad1ec7d48cd8270ad0af58.jpg)

![516bb3ff3be2560600391beee658d6dd176a4c7eae0f3f9324ed7fdcc61b9f6a.jpg](../iclr_results/2660_MeshAnything_ Artist-Created Mesh Generation with Autoregressive Transformers/images/516bb3ff3be2560600391beee658d6dd176a4c7eae0f3f9324ed7fdcc61b9f6a.jpg)

![5d3c2a46bdc248b9f40e48fe80ab7a79f7fb0cf07fcdd16aacfffc1edfdb0b72.jpg](../iclr_results/2660_MeshAnything_ Artist-Created Mesh Generation with Autoregressive Transformers/images/5d3c2a46bdc248b9f40e48fe80ab7a79f7fb0cf07fcdd16aacfffc1edfdb0b72.jpg)

![5e3a4615c7e9d6c6945dee1a8d0006b84438fd4acaa8f883b7b1e1207e683d2a.jpg](../iclr_results/2660_MeshAnything_ Artist-Created Mesh Generation with Autoregressive Transformers/images/5e3a4615c7e9d6c6945dee1a8d0006b84438fd4acaa8f883b7b1e1207e683d2a.jpg)

![97e0f73c205023a42723ae3d91948db98bbd649b047186d33e73d71e6a424692.jpg](../iclr_results/2660_MeshAnything_ Artist-Created Mesh Generation with Autoregressive Transformers/images/97e0f73c205023a42723ae3d91948db98bbd649b047186d33e73d71e6a424692.jpg)

![d5241254718d01960665937c56f6990b50d6eac629abeed4e72005e4c7ff209a.jpg](../iclr_results/2660_MeshAnything_ Artist-Created Mesh Generation with Autoregressive Transformers/images/d5241254718d01960665937c56f6990b50d6eac629abeed4e72005e4c7ff209a.jpg)

![f83d4443b996cc4d07a61ad56cdedf95d8b01a310b36c47f60660b379856a972.jpg](../iclr_results/2660_MeshAnything_ Artist-Created Mesh Generation with Autoregressive Transformers/images/f83d4443b996cc4d07a61ad56cdedf95d8b01a310b36c47f60660b379856a972.jpg)

![f9e459580e2d250587e4049bded496d7cee3df462a623fb94a34f0966d952f79.jpg](../iclr_results/2660_MeshAnything_ Artist-Created Mesh Generation with Autoregressive Transformers/images/f9e459580e2d250587e4049bded496d7cee3df462a623fb94a34f0966d952f79.jpg)

### Tables

![2fa795a249a0cfa0bab05cf06d7af1550057a80326a4527bf74d31f9dbb98d73.jpg](../iclr_results/2660_MeshAnything_ Artist-Created Mesh Generation with Autoregressive Transformers/tables/2fa795a249a0cfa0bab05cf06d7af1550057a80326a4527bf74d31f9dbb98d73.jpg)

![361411a44ad6c0e6c0d052f57bfb8fede7af931eedb02fb1b4ffb001b70543c9.jpg](../iclr_results/2660_MeshAnything_ Artist-Created Mesh Generation with Autoregressive Transformers/tables/361411a44ad6c0e6c0d052f57bfb8fede7af931eedb02fb1b4ffb001b70543c9.jpg)

![4e7d249dd742e59c1ea64d073961903b40f075c14dc56fb859b3c376d659e2b0.jpg](../iclr_results/2660_MeshAnything_ Artist-Created Mesh Generation with Autoregressive Transformers/tables/4e7d249dd742e59c1ea64d073961903b40f075c14dc56fb859b3c376d659e2b0.jpg)

![5528b63e211fa834e4aa44318bd9625dd2aee18fb85a5850b0970108faf21528.jpg](../iclr_results/2660_MeshAnything_ Artist-Created Mesh Generation with Autoregressive Transformers/tables/5528b63e211fa834e4aa44318bd9625dd2aee18fb85a5850b0970108faf21528.jpg)

![97ca4c16f9749b2fa269a443184a35a483e8fd765fda73385443d04225f2f5e1.jpg](../iclr_results/2660_MeshAnything_ Artist-Created Mesh Generation with Autoregressive Transformers/tables/97ca4c16f9749b2fa269a443184a35a483e8fd765fda73385443d04225f2f5e1.jpg)

![d662c256ca770f232d482e4e533397b821b9a6727661913fb31d8e9ef93ab26b.jpg](../iclr_results/2660_MeshAnything_ Artist-Created Mesh Generation with Autoregressive Transformers/tables/d662c256ca770f232d482e4e533397b821b9a6727661913fb31d8e9ef93ab26b.jpg)

![eeac7c02ae13855fb5e115ae0b0fe4df14fed6e58fdf6afee599760d774c5f80.jpg](../iclr_results/2660_MeshAnything_ Artist-Created Mesh Generation with Autoregressive Transformers/tables/eeac7c02ae13855fb5e115ae0b0fe4df14fed6e58fdf6afee599760d774c5f80.jpg)

![f89cdb1c5505472c3925e3f63e392f1e4dc76b4998ac46e1890d60ac1394754b.jpg](../iclr_results/2660_MeshAnything_ Artist-Created Mesh Generation with Autoregressive Transformers/tables/f89cdb1c5505472c3925e3f63e392f1e4dc76b4998ac46e1890d60ac1394754b.jpg)

## Inverse Rendering using Multi-Bounce Path Tracing and Reservoir Sampling


### Images

![058de6c3e23c3a493b5b3f2796c768b7ddd3654b5254e73fa57e966f2f3d1c40.jpg](../iclr_results/2661_Inverse Rendering using Multi-Bounce Path Tracing and Reservoir Sampling/images/058de6c3e23c3a493b5b3f2796c768b7ddd3654b5254e73fa57e966f2f3d1c40.jpg)

![0b5cef7252b57821f6be54464fe1cb5a866da83bb69a93ad4db626059bef298d.jpg](../iclr_results/2661_Inverse Rendering using Multi-Bounce Path Tracing and Reservoir Sampling/images/0b5cef7252b57821f6be54464fe1cb5a866da83bb69a93ad4db626059bef298d.jpg)

![2a0d41b67666c4e1e85a5cbf441e0ca6c692bde780a7a44e305f11b0f07b8e3d.jpg](../iclr_results/2661_Inverse Rendering using Multi-Bounce Path Tracing and Reservoir Sampling/images/2a0d41b67666c4e1e85a5cbf441e0ca6c692bde780a7a44e305f11b0f07b8e3d.jpg)

![33905192ae6f088250cf0b66e6fb75f3d466ae24a8d6be4722f7be439130474c.jpg](../iclr_results/2661_Inverse Rendering using Multi-Bounce Path Tracing and Reservoir Sampling/images/33905192ae6f088250cf0b66e6fb75f3d466ae24a8d6be4722f7be439130474c.jpg)

![499a158dc690d9cc269bd0b13fe6a428c4649002485dc600b44119614c0322fa.jpg](../iclr_results/2661_Inverse Rendering using Multi-Bounce Path Tracing and Reservoir Sampling/images/499a158dc690d9cc269bd0b13fe6a428c4649002485dc600b44119614c0322fa.jpg)

![7b329092ea02c4be448cb1d97a0c7b16fc528bd7e2d6650e1f34b3ac4fc01e89.jpg](../iclr_results/2661_Inverse Rendering using Multi-Bounce Path Tracing and Reservoir Sampling/images/7b329092ea02c4be448cb1d97a0c7b16fc528bd7e2d6650e1f34b3ac4fc01e89.jpg)

![7f3e1b5d77f6a724d60fa39bcbbcbe9c3493474ea7a73a6ccfd08b6e5e0919cb.jpg](../iclr_results/2661_Inverse Rendering using Multi-Bounce Path Tracing and Reservoir Sampling/images/7f3e1b5d77f6a724d60fa39bcbbcbe9c3493474ea7a73a6ccfd08b6e5e0919cb.jpg)

![8b94379a66f778189de591100fe045f043d68bd0bd90084c3581cf17d1054bed.jpg](../iclr_results/2661_Inverse Rendering using Multi-Bounce Path Tracing and Reservoir Sampling/images/8b94379a66f778189de591100fe045f043d68bd0bd90084c3581cf17d1054bed.jpg)

![8fe354a7c5c99cff164689298bda62ce0fe4ff17aeb23d7375be47bf01e35e32.jpg](../iclr_results/2661_Inverse Rendering using Multi-Bounce Path Tracing and Reservoir Sampling/images/8fe354a7c5c99cff164689298bda62ce0fe4ff17aeb23d7375be47bf01e35e32.jpg)

![bf08e16c30d0f3e26e6dd386a22d8b1e01d3b3a9f54900214a8d213072d49806.jpg](../iclr_results/2661_Inverse Rendering using Multi-Bounce Path Tracing and Reservoir Sampling/images/bf08e16c30d0f3e26e6dd386a22d8b1e01d3b3a9f54900214a8d213072d49806.jpg)

![c762c564314bb71d6cbe7537b2421b5b588dcb1358e47b30c9c550e6276b0140.jpg](../iclr_results/2661_Inverse Rendering using Multi-Bounce Path Tracing and Reservoir Sampling/images/c762c564314bb71d6cbe7537b2421b5b588dcb1358e47b30c9c550e6276b0140.jpg)

![cef5ac8823f85b03b36cb2ba2eaa0828dc28c0a1bd839282e71325e0a31d21ed.jpg](../iclr_results/2661_Inverse Rendering using Multi-Bounce Path Tracing and Reservoir Sampling/images/cef5ac8823f85b03b36cb2ba2eaa0828dc28c0a1bd839282e71325e0a31d21ed.jpg)

![d508b38f2769518db00b0ae8d6b430b02f536b1ad6c91c8917432b3415f6e005.jpg](../iclr_results/2661_Inverse Rendering using Multi-Bounce Path Tracing and Reservoir Sampling/images/d508b38f2769518db00b0ae8d6b430b02f536b1ad6c91c8917432b3415f6e005.jpg)

![efaffef2c9ad7eb276f80d2f1267bbcc96aea4c3a18b5fc860f348c346f92b6d.jpg](../iclr_results/2661_Inverse Rendering using Multi-Bounce Path Tracing and Reservoir Sampling/images/efaffef2c9ad7eb276f80d2f1267bbcc96aea4c3a18b5fc860f348c346f92b6d.jpg)

### Tables

![3b1b633c85d5565d0d8775d750c277ede3de7e0148d97c2992f3c6575a5011c5.jpg](../iclr_results/2661_Inverse Rendering using Multi-Bounce Path Tracing and Reservoir Sampling/tables/3b1b633c85d5565d0d8775d750c277ede3de7e0148d97c2992f3c6575a5011c5.jpg)

![3f4c0d32e8f835a45a83f657257fa2ec79938a2fa664c7f48fb3069ac4360cfb.jpg](../iclr_results/2661_Inverse Rendering using Multi-Bounce Path Tracing and Reservoir Sampling/tables/3f4c0d32e8f835a45a83f657257fa2ec79938a2fa664c7f48fb3069ac4360cfb.jpg)

![91aca92e53630df5c64a168bc15cd908b150ad196c339f064f12ab8e392a4d27.jpg](../iclr_results/2661_Inverse Rendering using Multi-Bounce Path Tracing and Reservoir Sampling/tables/91aca92e53630df5c64a168bc15cd908b150ad196c339f064f12ab8e392a4d27.jpg)

![e472277407cdbc3023173de645676ed7c57c04c792e7706f46b9f1032dbbd115.jpg](../iclr_results/2661_Inverse Rendering using Multi-Bounce Path Tracing and Reservoir Sampling/tables/e472277407cdbc3023173de645676ed7c57c04c792e7706f46b9f1032dbbd115.jpg)

## From GNNs to Trees: Multi-Granular Interpretability for Graph Neural Networks


### Images

![10b30572adebb1cd1be79e0c6b259e074761a496667a96f4c9c67ff9348c5a2e.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/images/10b30572adebb1cd1be79e0c6b259e074761a496667a96f4c9c67ff9348c5a2e.jpg)

![16ec8d315b86f211ccf0fb03d6c6c15d0412daadacd1f033184292600e8025af.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/images/16ec8d315b86f211ccf0fb03d6c6c15d0412daadacd1f033184292600e8025af.jpg)

![28026752cfef8144b9d7c373cc011ac97e1434b5b949572443e3cd0f4dce5f56.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/images/28026752cfef8144b9d7c373cc011ac97e1434b5b949572443e3cd0f4dce5f56.jpg)

![361e45a9ecd2649ed6420438c3c2a9fe687741bb9f99dc491be1b1e4e5674aea.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/images/361e45a9ecd2649ed6420438c3c2a9fe687741bb9f99dc491be1b1e4e5674aea.jpg)

![3aa4101ff86989763bcc07df53d443dd908bce988ce7ed25710f7df01c2b4853.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/images/3aa4101ff86989763bcc07df53d443dd908bce988ce7ed25710f7df01c2b4853.jpg)

![6785cb614edb70ef9e825c4c09a682f1cdc6fd2d922c252208325884b29b6efa.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/images/6785cb614edb70ef9e825c4c09a682f1cdc6fd2d922c252208325884b29b6efa.jpg)

![710eec795dc9896b4050e7e6051f057ae4976f07e4e6fa23c0987cec7559359e.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/images/710eec795dc9896b4050e7e6051f057ae4976f07e4e6fa23c0987cec7559359e.jpg)

![ab857a31fa3deac69dab78a5c1627b11da3379b2eb16f1c6e693c638546d5cbf.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/images/ab857a31fa3deac69dab78a5c1627b11da3379b2eb16f1c6e693c638546d5cbf.jpg)

![bcb58e6b56ee2eee6167788259373d176eacf616bd250127fc84996f2b02f242.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/images/bcb58e6b56ee2eee6167788259373d176eacf616bd250127fc84996f2b02f242.jpg)

![bedd96c9a691c5f7098eb33925c1c27684ce2a8755fc2378d2c333c15b31004c.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/images/bedd96c9a691c5f7098eb33925c1c27684ce2a8755fc2378d2c333c15b31004c.jpg)

![c22c20bc75669e80c02d30f9d148dce593b0fc16c30b4c198665d49b90c15aed.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/images/c22c20bc75669e80c02d30f9d148dce593b0fc16c30b4c198665d49b90c15aed.jpg)

![c90c66ec33ec25b439f2535b0a090aad7a4e57c985a42d0c264a6d2df35fd41d.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/images/c90c66ec33ec25b439f2535b0a090aad7a4e57c985a42d0c264a6d2df35fd41d.jpg)

![d2ab194a4f49ce699b7371c7f4a84c05585ff770e3e44d63c2b195ee132252a2.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/images/d2ab194a4f49ce699b7371c7f4a84c05585ff770e3e44d63c2b195ee132252a2.jpg)

![d95f1a6f65d4d22b027c2439bc72fd60f342f68ea622478087e9191f6dcd8def.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/images/d95f1a6f65d4d22b027c2439bc72fd60f342f68ea622478087e9191f6dcd8def.jpg)

![d989daeba77d6ec7840ad641d5d9872f242afe298a46ff2543d3138ab8aad767.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/images/d989daeba77d6ec7840ad641d5d9872f242afe298a46ff2543d3138ab8aad767.jpg)

![db4772b610bfb9cc4b0eff43b4cab8e4d839d185694bbe846261157193e8fc5f.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/images/db4772b610bfb9cc4b0eff43b4cab8e4d839d185694bbe846261157193e8fc5f.jpg)

### Tables

![21b62fc09a23641480ce7e74c64a895551a7212ce5fa7c03a6b29e2d4207b5b5.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/tables/21b62fc09a23641480ce7e74c64a895551a7212ce5fa7c03a6b29e2d4207b5b5.jpg)

![32c7d5b29a9bb008af09059893517a518a7084b970789b21b7c3004b2396de5f.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/tables/32c7d5b29a9bb008af09059893517a518a7084b970789b21b7c3004b2396de5f.jpg)

![3a10b5bd49f4866b4869dd3631cfd4a7562622747895d26cd95a6d17b06a29f5.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/tables/3a10b5bd49f4866b4869dd3631cfd4a7562622747895d26cd95a6d17b06a29f5.jpg)

![3ced70616ff8e54751a0f54b2cfad036ce494443ce633a84faca18fe78988693.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/tables/3ced70616ff8e54751a0f54b2cfad036ce494443ce633a84faca18fe78988693.jpg)

![3da7026d283e48419dbed5fb7b94b8ca1404429a9268743d8160511b16b8c90c.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/tables/3da7026d283e48419dbed5fb7b94b8ca1404429a9268743d8160511b16b8c90c.jpg)

![6e9c13d629860f2ac36ccd50d8423b2e2daca2e752895b61d3bbecbbc8ab62ab.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/tables/6e9c13d629860f2ac36ccd50d8423b2e2daca2e752895b61d3bbecbbc8ab62ab.jpg)

![871d8409c9b048bae4c2c0af56b16c17de2ff5ce4e8970da955a5f222629f3d3.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/tables/871d8409c9b048bae4c2c0af56b16c17de2ff5ce4e8970da955a5f222629f3d3.jpg)

![8964a33da66cb2e29c64d92b80a7a7d80b63f77d24d5d6fb1c847e716e0d8d53.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/tables/8964a33da66cb2e29c64d92b80a7a7d80b63f77d24d5d6fb1c847e716e0d8d53.jpg)

![9f12e7bb19c57f72821cf6c96d3613b3c5c9fb3d187ac138cc24dcec4a54ae64.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/tables/9f12e7bb19c57f72821cf6c96d3613b3c5c9fb3d187ac138cc24dcec4a54ae64.jpg)

![bcc4948f58749aa48964011f25451036063b180063cab0f6814b21262852fbf5.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/tables/bcc4948f58749aa48964011f25451036063b180063cab0f6814b21262852fbf5.jpg)

![e8437749d2f2b535e5d8b1a1668b96ad729d91879d65931f9c716465f84dfbdb.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/tables/e8437749d2f2b535e5d8b1a1668b96ad729d91879d65931f9c716465f84dfbdb.jpg)

![f0fa34fd25090fc93df464cf264f7773bec8cc4b981847dea62c78354f5ca699.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/tables/f0fa34fd25090fc93df464cf264f7773bec8cc4b981847dea62c78354f5ca699.jpg)

![fb4045bd06a28c169edddd835a46e920c6448c386bc43251b980c42ee095f9de.jpg](../iclr_results/2662_From GNNs to Trees_ Multi-Granular Interpretability for Graph Neural Networks/tables/fb4045bd06a28c169edddd835a46e920c6448c386bc43251b980c42ee095f9de.jpg)

## ReAttention: Training-Free Infinite Context with Finite Attention Scope


### Images

![2b8e0247eba255113ce906ea859b6e90cf1e45e67e62fcb98d7be856cd3df1bc.jpg](../iclr_results/2663_ReAttention_ Training-Free Infinite Context with Finite Attention Scope/images/2b8e0247eba255113ce906ea859b6e90cf1e45e67e62fcb98d7be856cd3df1bc.jpg)

![3041c320ca19adad1c570838aa97d7aab12d715bbe80321f2256ab4fe5e03578.jpg](../iclr_results/2663_ReAttention_ Training-Free Infinite Context with Finite Attention Scope/images/3041c320ca19adad1c570838aa97d7aab12d715bbe80321f2256ab4fe5e03578.jpg)

### Tables

![38604380d95e9e6a219885f7704964638c6301b57d3f1429214adac5b36dc40b.jpg](../iclr_results/2663_ReAttention_ Training-Free Infinite Context with Finite Attention Scope/tables/38604380d95e9e6a219885f7704964638c6301b57d3f1429214adac5b36dc40b.jpg)

![4367965a240ab041b5fc8697d0b7ae243360032365fcf9709c83b6b2975fcfe2.jpg](../iclr_results/2663_ReAttention_ Training-Free Infinite Context with Finite Attention Scope/tables/4367965a240ab041b5fc8697d0b7ae243360032365fcf9709c83b6b2975fcfe2.jpg)

![810ce2ce1a666734ec1d0de56af9ac23b7b2f59753e5dfb5fa8fa57b1fc62110.jpg](../iclr_results/2663_ReAttention_ Training-Free Infinite Context with Finite Attention Scope/tables/810ce2ce1a666734ec1d0de56af9ac23b7b2f59753e5dfb5fa8fa57b1fc62110.jpg)

![872806642dfdb491978ba99c2d465b131a3d76b643d1f24de75cea4c294059d3.jpg](../iclr_results/2663_ReAttention_ Training-Free Infinite Context with Finite Attention Scope/tables/872806642dfdb491978ba99c2d465b131a3d76b643d1f24de75cea4c294059d3.jpg)

![910d849253501a4a36626f24ba36f1874086fdc3cae7d6a1c049aba7fe000797.jpg](../iclr_results/2663_ReAttention_ Training-Free Infinite Context with Finite Attention Scope/tables/910d849253501a4a36626f24ba36f1874086fdc3cae7d6a1c049aba7fe000797.jpg)

![a1c46c845a5db96db52a431a297633a30fcc18f0f57c136d2768e6f3d62689eb.jpg](../iclr_results/2663_ReAttention_ Training-Free Infinite Context with Finite Attention Scope/tables/a1c46c845a5db96db52a431a297633a30fcc18f0f57c136d2768e6f3d62689eb.jpg)

![b934ad9fa553d575b51ca21fe1b2b1237de7e91e712c9aeea56af5888af7ca34.jpg](../iclr_results/2663_ReAttention_ Training-Free Infinite Context with Finite Attention Scope/tables/b934ad9fa553d575b51ca21fe1b2b1237de7e91e712c9aeea56af5888af7ca34.jpg)

## Divergence-Regularized Discounted Aggregation: Equilibrium Finding in Multiplayer Partially Observable Stochastic Games


### Images

![4e0f464fe20250fe65a686743945b98954ed6c6bc167f79d2a0671a919c9efe8.jpg](../iclr_results/2664_Divergence-Regularized Discounted Aggregation_ Equilibrium Finding in Multiplayer Partially Observab/images/4e0f464fe20250fe65a686743945b98954ed6c6bc167f79d2a0671a919c9efe8.jpg)

![8cbe0b92fe73c8bff94ce387993ef92fbe2a7dc6ca112e1e6a99b6e453475dd6.jpg](../iclr_results/2664_Divergence-Regularized Discounted Aggregation_ Equilibrium Finding in Multiplayer Partially Observab/images/8cbe0b92fe73c8bff94ce387993ef92fbe2a7dc6ca112e1e6a99b6e453475dd6.jpg)

![a2e28af4be2cfcae023d33451844d24a742de721e802cd253bb8d7d538afff7e.jpg](../iclr_results/2664_Divergence-Regularized Discounted Aggregation_ Equilibrium Finding in Multiplayer Partially Observab/images/a2e28af4be2cfcae023d33451844d24a742de721e802cd253bb8d7d538afff7e.jpg)

![c284ab16a3dd75749a6fc12b408ac1ec208e07e101eece47a8a693c43d15b02f.jpg](../iclr_results/2664_Divergence-Regularized Discounted Aggregation_ Equilibrium Finding in Multiplayer Partially Observab/images/c284ab16a3dd75749a6fc12b408ac1ec208e07e101eece47a8a693c43d15b02f.jpg)

![eb39e26f519fdd46ea3d257cb8a22bc1a6644f38549bd615f29c7905667b5785.jpg](../iclr_results/2664_Divergence-Regularized Discounted Aggregation_ Equilibrium Finding in Multiplayer Partially Observab/images/eb39e26f519fdd46ea3d257cb8a22bc1a6644f38549bd615f29c7905667b5785.jpg)

### Tables

![55a1b11cfbd39cf8636eaf6051774d096298af64722da9c0da139bd4e9374584.jpg](../iclr_results/2664_Divergence-Regularized Discounted Aggregation_ Equilibrium Finding in Multiplayer Partially Observab/tables/55a1b11cfbd39cf8636eaf6051774d096298af64722da9c0da139bd4e9374584.jpg)

![6061637a0d2a8f72ab9f75b76c46c1a6ddc088e0f89db8cfd27272de5b1a4ff4.jpg](../iclr_results/2664_Divergence-Regularized Discounted Aggregation_ Equilibrium Finding in Multiplayer Partially Observab/tables/6061637a0d2a8f72ab9f75b76c46c1a6ddc088e0f89db8cfd27272de5b1a4ff4.jpg)

![6ca93ad4532fcdd07b3ff0f3a1759c21c31787ac0bf9299237b09712e8351f86.jpg](../iclr_results/2664_Divergence-Regularized Discounted Aggregation_ Equilibrium Finding in Multiplayer Partially Observab/tables/6ca93ad4532fcdd07b3ff0f3a1759c21c31787ac0bf9299237b09712e8351f86.jpg)

![6f65771bda43e536a4d83271dfba78d7572443793705bd733d115f63c1c20021.jpg](../iclr_results/2664_Divergence-Regularized Discounted Aggregation_ Equilibrium Finding in Multiplayer Partially Observab/tables/6f65771bda43e536a4d83271dfba78d7572443793705bd733d115f63c1c20021.jpg)

![89ba0ec693b783a72e40c4a039fd1d7d240421c5ce5c7567a67cb246d143e843.jpg](../iclr_results/2664_Divergence-Regularized Discounted Aggregation_ Equilibrium Finding in Multiplayer Partially Observab/tables/89ba0ec693b783a72e40c4a039fd1d7d240421c5ce5c7567a67cb246d143e843.jpg)

![bf618aad56a6ea2068128af4ad89ef05caed20ddf13d11ef494536123c4e0b08.jpg](../iclr_results/2664_Divergence-Regularized Discounted Aggregation_ Equilibrium Finding in Multiplayer Partially Observab/tables/bf618aad56a6ea2068128af4ad89ef05caed20ddf13d11ef494536123c4e0b08.jpg)

![d1bb92903c7a514086e695f7400260ab38d2f863c32a1ef3b618773822ccebe0.jpg](../iclr_results/2664_Divergence-Regularized Discounted Aggregation_ Equilibrium Finding in Multiplayer Partially Observab/tables/d1bb92903c7a514086e695f7400260ab38d2f863c32a1ef3b618773822ccebe0.jpg)

![dacb101ad07c809d218fe62b597812e112d055c45fa2d2bb002dec33e164be46.jpg](../iclr_results/2664_Divergence-Regularized Discounted Aggregation_ Equilibrium Finding in Multiplayer Partially Observab/tables/dacb101ad07c809d218fe62b597812e112d055c45fa2d2bb002dec33e164be46.jpg)

## Prevalence of Negative Transfer in Continual Reinforcement Learning: Analyses and a Simple Baseline


### Images

![0ed9283e8600a6f63d861e7a7e78e7662bd42fa97177fff1af35add81f06a69e.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/images/0ed9283e8600a6f63d861e7a7e78e7662bd42fa97177fff1af35add81f06a69e.jpg)

![2a082404630dd391d27b74bbf025cab6a6afabb8995956486f4cfcb14eb46252.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/images/2a082404630dd391d27b74bbf025cab6a6afabb8995956486f4cfcb14eb46252.jpg)

![2f0d99c298d5cc6a6151f09f6dc7a7267dac56d14b29c801869d63098bd416e0.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/images/2f0d99c298d5cc6a6151f09f6dc7a7267dac56d14b29c801869d63098bd416e0.jpg)

![5a36566997ebf100ea6e8a034cc72cdd54907897449b00088789fd042836518a.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/images/5a36566997ebf100ea6e8a034cc72cdd54907897449b00088789fd042836518a.jpg)

![5c1733a303910dba7799aaf8ac9e334e42d1c3f08c548740460693be29eeb6e9.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/images/5c1733a303910dba7799aaf8ac9e334e42d1c3f08c548740460693be29eeb6e9.jpg)

![66b643d8a618438b0f9a84374676be67a70f21ca1455e2e5b1a1269d0a187bdf.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/images/66b643d8a618438b0f9a84374676be67a70f21ca1455e2e5b1a1269d0a187bdf.jpg)

![8888e99eb964607ad1d2fdee39d5ae5aa990ca7230a3e3f5eab07619ec3119fa.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/images/8888e99eb964607ad1d2fdee39d5ae5aa990ca7230a3e3f5eab07619ec3119fa.jpg)

![8f506de4337b8a0bae763e040a1104810df7569bd08196e2ea7308a333429cff.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/images/8f506de4337b8a0bae763e040a1104810df7569bd08196e2ea7308a333429cff.jpg)

![923e835d7dc1282a8530d67b19d075e848e0bbb89e4f05e504b5c571b17bc6b0.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/images/923e835d7dc1282a8530d67b19d075e848e0bbb89e4f05e504b5c571b17bc6b0.jpg)

![9df6d89b92118787ac1e6cf45fb7951d415aa68bf82efdbe74146813fd769bef.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/images/9df6d89b92118787ac1e6cf45fb7951d415aa68bf82efdbe74146813fd769bef.jpg)

![b449f304cfffe77ccd19818cee9670ea76a6098ec451898e33d72ed5e929ac3a.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/images/b449f304cfffe77ccd19818cee9670ea76a6098ec451898e33d72ed5e929ac3a.jpg)

![cd1a01fb8210524aaef72e5c1d953033032af5773d9e7f62b34966c37a47787e.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/images/cd1a01fb8210524aaef72e5c1d953033032af5773d9e7f62b34966c37a47787e.jpg)

![ce9c49c03fd3bfb188c6bb4c6d061f8d517a7bf38adfe2fc579726d355cc26bf.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/images/ce9c49c03fd3bfb188c6bb4c6d061f8d517a7bf38adfe2fc579726d355cc26bf.jpg)

![dc5ec6d310f5ff693ac522b0f6d8282eb8a550568d403c5afb5256666bdbcbd9.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/images/dc5ec6d310f5ff693ac522b0f6d8282eb8a550568d403c5afb5256666bdbcbd9.jpg)

![e18a44a5e73d167db5503e6468954e37580ef8a96efaab172c151174b6373837.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/images/e18a44a5e73d167db5503e6468954e37580ef8a96efaab172c151174b6373837.jpg)

![e6dcf60d7970eff59a636edfb5925969ff2d3230e7b277640b39c7e827bf2490.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/images/e6dcf60d7970eff59a636edfb5925969ff2d3230e7b277640b39c7e827bf2490.jpg)

![e87b686184db94611f7abadbe152e7972617ffa1d0e308fd91e4c615d74d0e23.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/images/e87b686184db94611f7abadbe152e7972617ffa1d0e308fd91e4c615d74d0e23.jpg)

![ea6de0be0ef151e5c2e1afea3234a111f66e010620b38c41ab2be4ec38bafc2a.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/images/ea6de0be0ef151e5c2e1afea3234a111f66e010620b38c41ab2be4ec38bafc2a.jpg)

![fceac78fc0f51d7df6d1913a777a9d251e82a60cc683b0fa94d075718fecda8f.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/images/fceac78fc0f51d7df6d1913a777a9d251e82a60cc683b0fa94d075718fecda8f.jpg)

![fd4652f94604b0266a15083debafc953b83ad42d8ad0599fe75c162912bf9a05.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/images/fd4652f94604b0266a15083debafc953b83ad42d8ad0599fe75c162912bf9a05.jpg)

### Tables

![16691f9cf1b57368df3cc2545a17396f740cbb5faed02fb29c4e34ef245b0e51.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/tables/16691f9cf1b57368df3cc2545a17396f740cbb5faed02fb29c4e34ef245b0e51.jpg)

![60fe3c60b990c62147a90cccb636cefd3e34c419b6828ce6ae4309598329bdfe.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/tables/60fe3c60b990c62147a90cccb636cefd3e34c419b6828ce6ae4309598329bdfe.jpg)

![83528217c823136f5de45f02dc8035f64c8f5fc8554ca849b5447234eae1070b.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/tables/83528217c823136f5de45f02dc8035f64c8f5fc8554ca849b5447234eae1070b.jpg)

![f86685b58efdb5e0cf2d31b3dcc5d2ea4622516a9ed44382ddf33cd098942ec5.jpg](../iclr_results/2665_Prevalence of Negative Transfer in Continual Reinforcement Learning_ Analyses and a Simple Baseline/tables/f86685b58efdb5e0cf2d31b3dcc5d2ea4622516a9ed44382ddf33cd098942ec5.jpg)

## Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images


### Images

![14f24a14cfea8310c908eafa71c6436dbff3f9f65cf26a4c398b2360fcb1e0bd.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/images/14f24a14cfea8310c908eafa71c6436dbff3f9f65cf26a4c398b2360fcb1e0bd.jpg)

![1e29580b6e2498ccba5c16f2a27ac14b00f14b5b9ee46ce35baf3612944dfd95.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/images/1e29580b6e2498ccba5c16f2a27ac14b00f14b5b9ee46ce35baf3612944dfd95.jpg)

![2d6ed1b8107a65db040ce94b8cc0eab48101b57d92c327ac39658ab840d3acd0.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/images/2d6ed1b8107a65db040ce94b8cc0eab48101b57d92c327ac39658ab840d3acd0.jpg)

![30a80596830bdc73c0f99870c8eb55b33c9e14aa49e0065c87fcb50f9b82db7e.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/images/30a80596830bdc73c0f99870c8eb55b33c9e14aa49e0065c87fcb50f9b82db7e.jpg)

![381aec129f6ecfc04dbcb72ddbeb5752fc36129f9712f9c3c792826a2605b332.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/images/381aec129f6ecfc04dbcb72ddbeb5752fc36129f9712f9c3c792826a2605b332.jpg)

![3bbdece76fb4afbaebc1d367c0ec17c87feec822783ba292caebd439202ecd40.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/images/3bbdece76fb4afbaebc1d367c0ec17c87feec822783ba292caebd439202ecd40.jpg)

![4f4e3940fb22bf88cfbcebf8861b68876d3996dfa5140a5d701de344f73dac43.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/images/4f4e3940fb22bf88cfbcebf8861b68876d3996dfa5140a5d701de344f73dac43.jpg)

![5078391463fd6c77693bd8581ed9f8c22b17675cf91b25122ff365ecbfe02505.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/images/5078391463fd6c77693bd8581ed9f8c22b17675cf91b25122ff365ecbfe02505.jpg)

![524db1815fd68d1ee64f6ea929324b0495a8af1745b1f54f43a572d7cb400b40.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/images/524db1815fd68d1ee64f6ea929324b0495a8af1745b1f54f43a572d7cb400b40.jpg)

![6af3d8895370899f2d2297e74dc3c1d8aa500c6489e13cbb7ff6a46ad35f6222.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/images/6af3d8895370899f2d2297e74dc3c1d8aa500c6489e13cbb7ff6a46ad35f6222.jpg)

![7c65f5f3aa478cb4c22765ae0d8ef955ebabb34a2b2db27721972f1d429aadf8.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/images/7c65f5f3aa478cb4c22765ae0d8ef955ebabb34a2b2db27721972f1d429aadf8.jpg)

![80bc8bbd011a30bc0e782b70d2aad36faf44d57c5fdcfe6dc728d987e9026546.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/images/80bc8bbd011a30bc0e782b70d2aad36faf44d57c5fdcfe6dc728d987e9026546.jpg)

![831d5b888b6fe4c5febf9c4f65943bde369f985fe5d86df820c82f9eefbedd51.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/images/831d5b888b6fe4c5febf9c4f65943bde369f985fe5d86df820c82f9eefbedd51.jpg)

![9577707fb4199b88de91fb04d1d924c2d1562af7313b3b92add8009939dbc971.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/images/9577707fb4199b88de91fb04d1d924c2d1562af7313b3b92add8009939dbc971.jpg)

![97b9dd2901f3e76e40adea03d64c2fcd6215d3baaf6bc1497d0b0b18ed3d0229.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/images/97b9dd2901f3e76e40adea03d64c2fcd6215d3baaf6bc1497d0b0b18ed3d0229.jpg)

![993a646b14bc7b61416c916a2b3cdea6ee84a840f1791504c3bf6eb79d7744d9.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/images/993a646b14bc7b61416c916a2b3cdea6ee84a840f1791504c3bf6eb79d7744d9.jpg)

![bb66da3b388a369c338a3e0d3a3639f8e408541f4138c7097fed582df8fbec46.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/images/bb66da3b388a369c338a3e0d3a3639f8e408541f4138c7097fed582df8fbec46.jpg)

![c7755262d257d043762e1f0ce84530451f48f9511c747edfc1665575cbd4d520.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/images/c7755262d257d043762e1f0ce84530451f48f9511c747edfc1665575cbd4d520.jpg)

![d5823b3493ce6046829f1d0a6ce586d2ee92000206461a4c14f3b17763db636c.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/images/d5823b3493ce6046829f1d0a6ce586d2ee92000206461a4c14f3b17763db636c.jpg)

![dc7280a1e1ec9c31910320d5e601a7c390ec3cca7e7de559f30ac85d1cea5bb6.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/images/dc7280a1e1ec9c31910320d5e601a7c390ec3cca7e7de559f30ac85d1cea5bb6.jpg)

![e1aaf5579da53e94f3cc58ce97cd812218cd419f805a255e427437cb62ee2d45.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/images/e1aaf5579da53e94f3cc58ce97cd812218cd419f805a255e427437cb62ee2d45.jpg)

![f7635e0c163087c092f78094ba163148889e7d7a1f6c64c9e8374ff99eaa1a59.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/images/f7635e0c163087c092f78094ba163148889e7d7a1f6c64c9e8374ff99eaa1a59.jpg)

![f7973ead9975360016899748339e7464c952333d9d51fa5acfdba5de636ff456.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/images/f7973ead9975360016899748339e7464c952333d9d51fa5acfdba5de636ff456.jpg)

### Tables

![0f09b3651c574e7f42c449858c0013c822765a3920bc248645e59546c4a53794.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/tables/0f09b3651c574e7f42c449858c0013c822765a3920bc248645e59546c4a53794.jpg)

![11f0b52182f9d99c34c12ff4d739a24a7f8e93ed88ebeba469bb796316f4a84e.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/tables/11f0b52182f9d99c34c12ff4d739a24a7f8e93ed88ebeba469bb796316f4a84e.jpg)

![162f53e9bd2ac3896b4a4390e407b87598afb22dca921503a6c7d2ed779c0d29.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/tables/162f53e9bd2ac3896b4a4390e407b87598afb22dca921503a6c7d2ed779c0d29.jpg)

![1bee852c6603c07b0085a86194429f58c91a26d9ffb71270e65e14f399527619.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/tables/1bee852c6603c07b0085a86194429f58c91a26d9ffb71270e65e14f399527619.jpg)

![2717c0a9624cf9ce3d55026b5d22e2ac26d5c491406b2fe87b4414853cee5dfc.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/tables/2717c0a9624cf9ce3d55026b5d22e2ac26d5c491406b2fe87b4414853cee5dfc.jpg)

![407c20e64538aa370a220437b69c9ad5cd71ddb12f43311171d08a8fa39830be.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/tables/407c20e64538aa370a220437b69c9ad5cd71ddb12f43311171d08a8fa39830be.jpg)

![514f1b91593efbdc62694eb43150eafad5e5a3d74d156cd8846387ede2582ee0.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/tables/514f1b91593efbdc62694eb43150eafad5e5a3d74d156cd8846387ede2582ee0.jpg)

![595b0a014ad8ee9c738f6d38722b014028e5e0d96713d20ed0d7f77b04783935.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/tables/595b0a014ad8ee9c738f6d38722b014028e5e0d96713d20ed0d7f77b04783935.jpg)

![9ff2f1384aaee02e9668ae73fc2d8fb519c85bd0d57d053eeb6bb40b54e39fa0.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/tables/9ff2f1384aaee02e9668ae73fc2d8fb519c85bd0d57d053eeb6bb40b54e39fa0.jpg)

![a579c8f3c822b5352aaaadf0dae9162991397e8095c6ae1e044faa45c33d4356.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/tables/a579c8f3c822b5352aaaadf0dae9162991397e8095c6ae1e044faa45c33d4356.jpg)

![f534ec72dd9ff5ee1070baae39cf36ff2db2b6360fd7f1c5dce0f14a82b808c2.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/tables/f534ec72dd9ff5ee1070baae39cf36ff2db2b6360fd7f1c5dce0f14a82b808c2.jpg)

![f6a945798012ce8482cd115c4fe3746a25495ce2e5c25d2926b5e3b181ead24a.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/tables/f6a945798012ce8482cd115c4fe3746a25495ce2e5c25d2926b5e3b181ead24a.jpg)

![f9fe4d8d64a06592a53830f330a8944965e6ff1ea80520b5c0374322fa937607.jpg](../iclr_results/2666_Tracking the Copyright of Large Vision-Language Models through Parameter Learning Adversarial Images/tables/f9fe4d8d64a06592a53830f330a8944965e6ff1ea80520b5c0374322fa937607.jpg)

## MLLM as Retriever: Interactively Learning Multimodal Retrieval for Embodied Agents


### Images

![02828f0ce1735029abbe73232a0afb02ea8f1111472e98f02b041750c280af36.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/02828f0ce1735029abbe73232a0afb02ea8f1111472e98f02b041750c280af36.jpg)

![25a1537028869c4fe746db846adfa1d1f92c375e8b4e81ac477947bc7a4b71e7.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/25a1537028869c4fe746db846adfa1d1f92c375e8b4e81ac477947bc7a4b71e7.jpg)

![2c775460ce1c6d6cd6d64915b7ff7eaa32db4f09e3f1f7fb1c20d3560ce3e9eb.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/2c775460ce1c6d6cd6d64915b7ff7eaa32db4f09e3f1f7fb1c20d3560ce3e9eb.jpg)

![33e5b4451b36abf94e660acceb56a77672b8b0d787aa319f2954ff2ee8b933be.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/33e5b4451b36abf94e660acceb56a77672b8b0d787aa319f2954ff2ee8b933be.jpg)

![353e18a194b50d25e813fe68c3e771f2be4cf13a47ad22ee2093252ff5553fff.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/353e18a194b50d25e813fe68c3e771f2be4cf13a47ad22ee2093252ff5553fff.jpg)

![405c9384d6a78d85a66b20e8ae6b7bba984c1450afa67b5ffc41e95420efc265.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/405c9384d6a78d85a66b20e8ae6b7bba984c1450afa67b5ffc41e95420efc265.jpg)

![427aa62f30a726622562f0d1118ebadf0d9e8aa0544daa49a9017cf15098caa8.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/427aa62f30a726622562f0d1118ebadf0d9e8aa0544daa49a9017cf15098caa8.jpg)

![42d7e0dc7bd846d1e7448db2b1322e9789035ee979313a296407dfec05126743.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/42d7e0dc7bd846d1e7448db2b1322e9789035ee979313a296407dfec05126743.jpg)

![448c468c64672faf5c59e51e0c8af1654b9e1b0d3a89e518e287c530a3861f68.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/448c468c64672faf5c59e51e0c8af1654b9e1b0d3a89e518e287c530a3861f68.jpg)

![4e13caaed2f301954d51d9be1571d3e3977953ab3d50048d1a2b7b5a784dc578.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/4e13caaed2f301954d51d9be1571d3e3977953ab3d50048d1a2b7b5a784dc578.jpg)

![5d28bdc1f21785ff2ea7a97098c5605c578eea747f0f87d2ed51e221e6fad2be.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/5d28bdc1f21785ff2ea7a97098c5605c578eea747f0f87d2ed51e221e6fad2be.jpg)

![69ef0612c26c2b7e6c98b49945671d56fe1ba1fd7c8f2994b0c4858f874b32d0.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/69ef0612c26c2b7e6c98b49945671d56fe1ba1fd7c8f2994b0c4858f874b32d0.jpg)

![76d73919139fba13528a5a473f89a8dc6b149d48e5caba619d6f36ce82091490.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/76d73919139fba13528a5a473f89a8dc6b149d48e5caba619d6f36ce82091490.jpg)

![7cb9e2d969ef7a48abf0e4e36cd353e9473d8bbb60cbc3c8d3341459380b775d.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/7cb9e2d969ef7a48abf0e4e36cd353e9473d8bbb60cbc3c8d3341459380b775d.jpg)

![7dc6e245f75651026e67798b79f9420b64c49e78fadb057d77ea6342220a8ac4.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/7dc6e245f75651026e67798b79f9420b64c49e78fadb057d77ea6342220a8ac4.jpg)

![8fe9ccec3065150ef66fad0ee63bed8d78d474cb4b28e3b01960a7bbdd2c2197.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/8fe9ccec3065150ef66fad0ee63bed8d78d474cb4b28e3b01960a7bbdd2c2197.jpg)

![a8cb9168f7e9c9994e9ac658cc31b53fbe9ace0547e9a922a632c6610e42772e.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/a8cb9168f7e9c9994e9ac658cc31b53fbe9ace0547e9a922a632c6610e42772e.jpg)

![ac3ceb03ca8f6b5c6e2aec8d10c6e38e4ea78c82b5358e3a43f7f8b6959f6fc4.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/ac3ceb03ca8f6b5c6e2aec8d10c6e38e4ea78c82b5358e3a43f7f8b6959f6fc4.jpg)

![b4c3ba1f57c655552cadd0e33eb896049b2fde354aa3842777e43a362113048e.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/b4c3ba1f57c655552cadd0e33eb896049b2fde354aa3842777e43a362113048e.jpg)

![be60f81cd549146040110c8ed01617e1ced6bf361ebc510351edad378b08e3d0.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/be60f81cd549146040110c8ed01617e1ced6bf361ebc510351edad378b08e3d0.jpg)

![ce0b4379ad647e372a2c044f85b178c31113d255a00e23f6b986f2ce3a28870d.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/ce0b4379ad647e372a2c044f85b178c31113d255a00e23f6b986f2ce3a28870d.jpg)

![e89ab7b1e6004bb9b9820b1d05f7c0173de20c2a835bb55eeaf5732953ce7aa5.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/e89ab7b1e6004bb9b9820b1d05f7c0173de20c2a835bb55eeaf5732953ce7aa5.jpg)

![f64c25ec4d1e4c1b57c55444ddf440987d8b7c62d1850c6a2f7f1ec897487e74.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/f64c25ec4d1e4c1b57c55444ddf440987d8b7c62d1850c6a2f7f1ec897487e74.jpg)

![fc72bd6bfe413cae8266b7d4a003f05f833f2145616b418a0809a5f774d00510.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/images/fc72bd6bfe413cae8266b7d4a003f05f833f2145616b418a0809a5f774d00510.jpg)

### Tables

![4e09f09997ddbd630f27aa5a33a6cf9ae9ae8b7fd86b315b5363b8cd2610ba9c.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/tables/4e09f09997ddbd630f27aa5a33a6cf9ae9ae8b7fd86b315b5363b8cd2610ba9c.jpg)

![5899b210608fa2289455cb1aee9c991906a6a5d90a325694659b9294ac8637d6.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/tables/5899b210608fa2289455cb1aee9c991906a6a5d90a325694659b9294ac8637d6.jpg)

![5d4a8f759313c8136f23b16c74ce0b554056bedbbd9a5c29b0c7570f1b614a72.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/tables/5d4a8f759313c8136f23b16c74ce0b554056bedbbd9a5c29b0c7570f1b614a72.jpg)

![64c3dae6cb5776e47c1ed8f0c164466f25cd0851551b8901b7c20a850d73999c.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/tables/64c3dae6cb5776e47c1ed8f0c164466f25cd0851551b8901b7c20a850d73999c.jpg)

![a468b61c50c84acfcea4b87285ce070df44bcaa89fb9ad3b7101a7fe1accea58.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/tables/a468b61c50c84acfcea4b87285ce070df44bcaa89fb9ad3b7101a7fe1accea58.jpg)

![b3c8154ca29696020675c952cc0f7dbb10dfe0c4f7dc35683e739deb7effb929.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/tables/b3c8154ca29696020675c952cc0f7dbb10dfe0c4f7dc35683e739deb7effb929.jpg)

![bb6709206ea97935b367a03bed436294e3b4fe850b87b01cb14389c1138afb09.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/tables/bb6709206ea97935b367a03bed436294e3b4fe850b87b01cb14389c1138afb09.jpg)

![c921996c33701ab2189f6a4390a8dae4238f3df353cbd8731990a61bc2f30651.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/tables/c921996c33701ab2189f6a4390a8dae4238f3df353cbd8731990a61bc2f30651.jpg)

![d60119109a577b17941a4346130d292158a1479d9577b749f8af887b7cf02590.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/tables/d60119109a577b17941a4346130d292158a1479d9577b749f8af887b7cf02590.jpg)

![d7bda75bba41636fea463c37def4acccc6e072c64ffb70aa89ff4d1d9588bb75.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/tables/d7bda75bba41636fea463c37def4acccc6e072c64ffb70aa89ff4d1d9588bb75.jpg)

![e0662e752e42884fb50381502f974f7efdc32b114a661cb116737ad37f3c87b1.jpg](../iclr_results/2667_MLLM as Retriever_ Interactively Learning Multimodal Retrieval for Embodied Agents/tables/e0662e752e42884fb50381502f974f7efdc32b114a661cb116737ad37f3c87b1.jpg)

## VLAS: Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation


### Images

![0e2bd46c80c41957ab7d4f140f83fd8ddc4a16db4ad123a77b7f205d30f6b76b.jpg](../iclr_results/2668_VLAS_ Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation/images/0e2bd46c80c41957ab7d4f140f83fd8ddc4a16db4ad123a77b7f205d30f6b76b.jpg)

![3d064047d68ce45e42ae105662664bf23b76423d006e60a0d651af99c5a296ca.jpg](../iclr_results/2668_VLAS_ Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation/images/3d064047d68ce45e42ae105662664bf23b76423d006e60a0d651af99c5a296ca.jpg)

![7245184e646f5730794c0362dd2ab46a733ee1f7e1de76d1d4a55ce44c01f27a.jpg](../iclr_results/2668_VLAS_ Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation/images/7245184e646f5730794c0362dd2ab46a733ee1f7e1de76d1d4a55ce44c01f27a.jpg)

![7d50624978a70dca5364f51b2343fd6382ccebe614eed96b69ec4802c73c69cf.jpg](../iclr_results/2668_VLAS_ Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation/images/7d50624978a70dca5364f51b2343fd6382ccebe614eed96b69ec4802c73c69cf.jpg)

![9857e7860906e9cf511e079193d308283949af6370b5f5ca511e0fe4915dd67b.jpg](../iclr_results/2668_VLAS_ Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation/images/9857e7860906e9cf511e079193d308283949af6370b5f5ca511e0fe4915dd67b.jpg)

![d7aa5c74e7fb10ae05f4255bd576020403bb1f3213cc3561ac42806779acdfcc.jpg](../iclr_results/2668_VLAS_ Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation/images/d7aa5c74e7fb10ae05f4255bd576020403bb1f3213cc3561ac42806779acdfcc.jpg)

![e109c9ec16625b294e8f24797d3243367e032b2c8b460134fbb012488ed75dc1.jpg](../iclr_results/2668_VLAS_ Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation/images/e109c9ec16625b294e8f24797d3243367e032b2c8b460134fbb012488ed75dc1.jpg)

![f597e37c7c0317dbba8b2a42d771ee29dd13f843c383a5aa4ffde037f39cf900.jpg](../iclr_results/2668_VLAS_ Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation/images/f597e37c7c0317dbba8b2a42d771ee29dd13f843c383a5aa4ffde037f39cf900.jpg)

![fe2734bb3dcddc5e44a6e72c6c355e4ced4b76270e964ed238739daeff21e353.jpg](../iclr_results/2668_VLAS_ Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation/images/fe2734bb3dcddc5e44a6e72c6c355e4ced4b76270e964ed238739daeff21e353.jpg)

### Tables

![030fa38c7ea4cbc4f9fad965752828f0b4b6535c482fe9e8980db967afee3f54.jpg](../iclr_results/2668_VLAS_ Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation/tables/030fa38c7ea4cbc4f9fad965752828f0b4b6535c482fe9e8980db967afee3f54.jpg)

![18c9afc12f3082bd87e4ad6dfbfeb0c013bcf5e8462d300a1f16d1dd5aa89ba2.jpg](../iclr_results/2668_VLAS_ Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation/tables/18c9afc12f3082bd87e4ad6dfbfeb0c013bcf5e8462d300a1f16d1dd5aa89ba2.jpg)

![1ef1f5cfeeafe74f354416a3eb394da7996cb485f270c2458eb9566ccc0f16a6.jpg](../iclr_results/2668_VLAS_ Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation/tables/1ef1f5cfeeafe74f354416a3eb394da7996cb485f270c2458eb9566ccc0f16a6.jpg)

![483918460ff0301ae4fce38d32891ce59d37568d738a7f653b4cff23bdcf85e5.jpg](../iclr_results/2668_VLAS_ Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation/tables/483918460ff0301ae4fce38d32891ce59d37568d738a7f653b4cff23bdcf85e5.jpg)

![5039b3f439fd04f61077e0db3ffd2b462db1e58e0b68e08fd757164eecfbff77.jpg](../iclr_results/2668_VLAS_ Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation/tables/5039b3f439fd04f61077e0db3ffd2b462db1e58e0b68e08fd757164eecfbff77.jpg)

![57eb899bc47931747c58801a41dca6ed97b3a6e238efebf2114b945f4c93175a.jpg](../iclr_results/2668_VLAS_ Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation/tables/57eb899bc47931747c58801a41dca6ed97b3a6e238efebf2114b945f4c93175a.jpg)

![60f5af63c2f2f227ca3a755bbf36f3cb069bcfc8fbf1dd6ba6711f952417de3c.jpg](../iclr_results/2668_VLAS_ Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation/tables/60f5af63c2f2f227ca3a755bbf36f3cb069bcfc8fbf1dd6ba6711f952417de3c.jpg)

![9663c02f6dbde342cc68e91f44b8390ade0a916e93cdf4db7215617ff6452e7d.jpg](../iclr_results/2668_VLAS_ Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation/tables/9663c02f6dbde342cc68e91f44b8390ade0a916e93cdf4db7215617ff6452e7d.jpg)

![a0dbd1281ca34e4c0a7615a7fb64a33c00f92729f7c202325fde03c71260ff7c.jpg](../iclr_results/2668_VLAS_ Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation/tables/a0dbd1281ca34e4c0a7615a7fb64a33c00f92729f7c202325fde03c71260ff7c.jpg)

![d4e40b40117440627f98194fc4d8757dd0403d0b27f4664a78a45f21d0a1b9f1.jpg](../iclr_results/2668_VLAS_ Vision-Language-Action Model with Speech Instructions for Customized Robot Manipulation/tables/d4e40b40117440627f98194fc4d8757dd0403d0b27f4664a78a45f21d0a1b9f1.jpg)

## Scaling Large Language Model-based Multi-Agent Collaboration


### Images

![205d0431229264c2f9295712325dfe09b941fd75a631118adde735731a4e8fce.jpg](../iclr_results/2669_Scaling Large Language Model-based Multi-Agent Collaboration/images/205d0431229264c2f9295712325dfe09b941fd75a631118adde735731a4e8fce.jpg)

![60f2cf92b51637c4fd6b8b6ff74c88179571f3d0b539e2136552cf07e0c507ec.jpg](../iclr_results/2669_Scaling Large Language Model-based Multi-Agent Collaboration/images/60f2cf92b51637c4fd6b8b6ff74c88179571f3d0b539e2136552cf07e0c507ec.jpg)

![642f6dcf4767502fa80b36edafec3f3a8ee8b334ad630e9f07cf2e1f10b0954a.jpg](../iclr_results/2669_Scaling Large Language Model-based Multi-Agent Collaboration/images/642f6dcf4767502fa80b36edafec3f3a8ee8b334ad630e9f07cf2e1f10b0954a.jpg)

![9fe871701f25838b1943330dc4b2bc70aabc11db7efcac840ea012be70564005.jpg](../iclr_results/2669_Scaling Large Language Model-based Multi-Agent Collaboration/images/9fe871701f25838b1943330dc4b2bc70aabc11db7efcac840ea012be70564005.jpg)

![c990f14379f1ad3556358375f75bcec1ad874e732661d781753c7cef3531a01d.jpg](../iclr_results/2669_Scaling Large Language Model-based Multi-Agent Collaboration/images/c990f14379f1ad3556358375f75bcec1ad874e732661d781753c7cef3531a01d.jpg)

![da452a770bf7e15d5c7ba437b2c69942d0adb094f4878c7b2f1dc87605081a07.jpg](../iclr_results/2669_Scaling Large Language Model-based Multi-Agent Collaboration/images/da452a770bf7e15d5c7ba437b2c69942d0adb094f4878c7b2f1dc87605081a07.jpg)

![dcf78f376be248b5b08d6f705b9afbf7b311790835ff3f15afaf92ed27c7e254.jpg](../iclr_results/2669_Scaling Large Language Model-based Multi-Agent Collaboration/images/dcf78f376be248b5b08d6f705b9afbf7b311790835ff3f15afaf92ed27c7e254.jpg)

![ff6c8d63517303c230f6db2627fba0817368a6562259ddf8e1f1900ae6e0b665.jpg](../iclr_results/2669_Scaling Large Language Model-based Multi-Agent Collaboration/images/ff6c8d63517303c230f6db2627fba0817368a6562259ddf8e1f1900ae6e0b665.jpg)

### Tables

![9e9ee2b31f93cd165eed25bc48f2914f91444005e2e90055c6d5aaf68b43b35b.jpg](../iclr_results/2669_Scaling Large Language Model-based Multi-Agent Collaboration/tables/9e9ee2b31f93cd165eed25bc48f2914f91444005e2e90055c6d5aaf68b43b35b.jpg)

## LLMs Can Plan Only If We Tell Them


### Images

![013afc2323ba7825951d0d2c998721630286b7d4997ef46cc580ead4a4ec097f.jpg](../iclr_results/2670_LLMs Can Plan Only If We Tell Them/images/013afc2323ba7825951d0d2c998721630286b7d4997ef46cc580ead4a4ec097f.jpg)

![333fda40c6f6e47c9fa3d8e73fbc0b4b84710869b70584f30d9432b8ed5233fb.jpg](../iclr_results/2670_LLMs Can Plan Only If We Tell Them/images/333fda40c6f6e47c9fa3d8e73fbc0b4b84710869b70584f30d9432b8ed5233fb.jpg)

![6707da172f3ecc781afac58f29ecd30516e2b41980df950ae2dd24eac0da6cb3.jpg](../iclr_results/2670_LLMs Can Plan Only If We Tell Them/images/6707da172f3ecc781afac58f29ecd30516e2b41980df950ae2dd24eac0da6cb3.jpg)

![b77cf6a8c6d655554b5de0b959c782791ee3191e291aef92f3e88fbc26b556bf.jpg](../iclr_results/2670_LLMs Can Plan Only If We Tell Them/images/b77cf6a8c6d655554b5de0b959c782791ee3191e291aef92f3e88fbc26b556bf.jpg)

![e753b53788309907edc63c0672f8a7ba86793e37f7c50fec0a3a135aa663fe09.jpg](../iclr_results/2670_LLMs Can Plan Only If We Tell Them/images/e753b53788309907edc63c0672f8a7ba86793e37f7c50fec0a3a135aa663fe09.jpg)

### Tables

![11e0c84787e3aa2217fb962f9301c341c202ed7efdc8e49973796cc70a4dc143.jpg](../iclr_results/2670_LLMs Can Plan Only If We Tell Them/tables/11e0c84787e3aa2217fb962f9301c341c202ed7efdc8e49973796cc70a4dc143.jpg)

![34486831845044fb7733c4a19f03abb08b3b73f05ad105ce5178b9d0058ee8a5.jpg](../iclr_results/2670_LLMs Can Plan Only If We Tell Them/tables/34486831845044fb7733c4a19f03abb08b3b73f05ad105ce5178b9d0058ee8a5.jpg)

![4eb4457211e5a21745f0648971fa4b110eec008c0b9c338cb0453dabdd33127f.jpg](../iclr_results/2670_LLMs Can Plan Only If We Tell Them/tables/4eb4457211e5a21745f0648971fa4b110eec008c0b9c338cb0453dabdd33127f.jpg)

![50fe14517ef7dfda32450067b9e713846739ce1686c3c69a39f3d9fb13b3fcec.jpg](../iclr_results/2670_LLMs Can Plan Only If We Tell Them/tables/50fe14517ef7dfda32450067b9e713846739ce1686c3c69a39f3d9fb13b3fcec.jpg)

![b739f9c5dd7e3b379aa91201f625a9c78b45e04739151311ff22552386261d6e.jpg](../iclr_results/2670_LLMs Can Plan Only If We Tell Them/tables/b739f9c5dd7e3b379aa91201f625a9c78b45e04739151311ff22552386261d6e.jpg)

![d7e560a59b8fcbb31eb2dd2c28bf08434a1df8658e119c60bb8f3f061eb9ea1e.jpg](../iclr_results/2670_LLMs Can Plan Only If We Tell Them/tables/d7e560a59b8fcbb31eb2dd2c28bf08434a1df8658e119c60bb8f3f061eb9ea1e.jpg)

![fd84cd5150559ce87a44ea1e84563a3d2c8d3a92e7b8268e62085a788b5b1e6f.jpg](../iclr_results/2670_LLMs Can Plan Only If We Tell Them/tables/fd84cd5150559ce87a44ea1e84563a3d2c8d3a92e7b8268e62085a788b5b1e6f.jpg)

## Gaussian Differentially Private Human Faces Under a Face Radial Curve Representation


### Images

![13c9cd5e80065392e44bdde6798baaf0862b5687f03ab08a27fee1d2e357ca7f.jpg](../iclr_results/2671_Gaussian Differentially Private Human Faces Under a Face Radial Curve Representation/images/13c9cd5e80065392e44bdde6798baaf0862b5687f03ab08a27fee1d2e357ca7f.jpg)

![520be3dbbaefdf0117804a3837d6470e862b952bc83cffd82408d71719f7ac92.jpg](../iclr_results/2671_Gaussian Differentially Private Human Faces Under a Face Radial Curve Representation/images/520be3dbbaefdf0117804a3837d6470e862b952bc83cffd82408d71719f7ac92.jpg)

![5746c1143374460d1bf6ba0607ab13da994920376d4e6ab8949fab7fc5e93646.jpg](../iclr_results/2671_Gaussian Differentially Private Human Faces Under a Face Radial Curve Representation/images/5746c1143374460d1bf6ba0607ab13da994920376d4e6ab8949fab7fc5e93646.jpg)

![656d08b3ba0f082cccc13a1f8c1c306555ff2a92fa94fd6099215de3786aee5b.jpg](../iclr_results/2671_Gaussian Differentially Private Human Faces Under a Face Radial Curve Representation/images/656d08b3ba0f082cccc13a1f8c1c306555ff2a92fa94fd6099215de3786aee5b.jpg)

![7189199f6c26ce6fa3ae52a41bcc2da60c8a601da00962dfedddf0d9c5996eca.jpg](../iclr_results/2671_Gaussian Differentially Private Human Faces Under a Face Radial Curve Representation/images/7189199f6c26ce6fa3ae52a41bcc2da60c8a601da00962dfedddf0d9c5996eca.jpg)

![93e05044684eb7d6354341ff9e8caf48b1919de286e1761c5bfa5dd2fba4c031.jpg](../iclr_results/2671_Gaussian Differentially Private Human Faces Under a Face Radial Curve Representation/images/93e05044684eb7d6354341ff9e8caf48b1919de286e1761c5bfa5dd2fba4c031.jpg)

![ba5949c7f198fcbe9b8f93b9c8aed49ad3bd92cc4ffd4dea6be87d227139d9d4.jpg](../iclr_results/2671_Gaussian Differentially Private Human Faces Under a Face Radial Curve Representation/images/ba5949c7f198fcbe9b8f93b9c8aed49ad3bd92cc4ffd4dea6be87d227139d9d4.jpg)

![babe4d3322ba5a20938485584238e205cb6872dadc4db4145c03bafc13f82a18.jpg](../iclr_results/2671_Gaussian Differentially Private Human Faces Under a Face Radial Curve Representation/images/babe4d3322ba5a20938485584238e205cb6872dadc4db4145c03bafc13f82a18.jpg)

![bd06b3d8b77bb31f8db11c489b68a920bcd3016e746ff568b32a553e7939202c.jpg](../iclr_results/2671_Gaussian Differentially Private Human Faces Under a Face Radial Curve Representation/images/bd06b3d8b77bb31f8db11c489b68a920bcd3016e746ff568b32a553e7939202c.jpg)

![e41706bf38d462e412e575b9b6a0f05a021e15e4eaabcfff545fd23b188036ac.jpg](../iclr_results/2671_Gaussian Differentially Private Human Faces Under a Face Radial Curve Representation/images/e41706bf38d462e412e575b9b6a0f05a021e15e4eaabcfff545fd23b188036ac.jpg)

### Tables

![455812477c06fe206e769572b30f2ad58093c1c89209581979dc05ca57f144a7.jpg](../iclr_results/2671_Gaussian Differentially Private Human Faces Under a Face Radial Curve Representation/tables/455812477c06fe206e769572b30f2ad58093c1c89209581979dc05ca57f144a7.jpg)

## Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later


### Images

![5f6c2ca3b97d7e3f5de2c0890e143b666e4b113419639d24af666e477239fd46.jpg](../iclr_results/2672_Revisiting Nearest Neighbor for Tabular Data_ A Deep Tabular Baseline Two Decades Later/images/5f6c2ca3b97d7e3f5de2c0890e143b666e4b113419639d24af666e477239fd46.jpg)

![651f3199194d44a79b67ebd2f8d6f93ee57a15b6e5542ea39986906e4d73194c.jpg](../iclr_results/2672_Revisiting Nearest Neighbor for Tabular Data_ A Deep Tabular Baseline Two Decades Later/images/651f3199194d44a79b67ebd2f8d6f93ee57a15b6e5542ea39986906e4d73194c.jpg)

![6770578ba3654647e3d765fa255cc2e354653bcce148b27d275ebe04d5e68b58.jpg](../iclr_results/2672_Revisiting Nearest Neighbor for Tabular Data_ A Deep Tabular Baseline Two Decades Later/images/6770578ba3654647e3d765fa255cc2e354653bcce148b27d275ebe04d5e68b58.jpg)

![d98994b0edcc5877dd8505addc156dcdafac0f6a21a462d38069ac0c298aa9db.jpg](../iclr_results/2672_Revisiting Nearest Neighbor for Tabular Data_ A Deep Tabular Baseline Two Decades Later/images/d98994b0edcc5877dd8505addc156dcdafac0f6a21a462d38069ac0c298aa9db.jpg)

### Tables

![04c11b0eaaed41107dc9c30f77babdcea1751251a16fc66f756da994a8adb194.jpg](../iclr_results/2672_Revisiting Nearest Neighbor for Tabular Data_ A Deep Tabular Baseline Two Decades Later/tables/04c11b0eaaed41107dc9c30f77babdcea1751251a16fc66f756da994a8adb194.jpg)

![04c9f45d9313d002ba2d83b7cf4f3a50acd2c93a54519905357b1d62fe41453d.jpg](../iclr_results/2672_Revisiting Nearest Neighbor for Tabular Data_ A Deep Tabular Baseline Two Decades Later/tables/04c9f45d9313d002ba2d83b7cf4f3a50acd2c93a54519905357b1d62fe41453d.jpg)

![3cd0f5631490d0f1a1c85a0c7e073e5a22717b439c2d90eaa033648208120899.jpg](../iclr_results/2672_Revisiting Nearest Neighbor for Tabular Data_ A Deep Tabular Baseline Two Decades Later/tables/3cd0f5631490d0f1a1c85a0c7e073e5a22717b439c2d90eaa033648208120899.jpg)

![485d3b0b87f53df98e6acdde0d58acf9acd6b3d3784f06360a279e5e64effab5.jpg](../iclr_results/2672_Revisiting Nearest Neighbor for Tabular Data_ A Deep Tabular Baseline Two Decades Later/tables/485d3b0b87f53df98e6acdde0d58acf9acd6b3d3784f06360a279e5e64effab5.jpg)

![5e63f8579413b17a58e51976aa194719e2480dc71ac9abd4dd60d6d5aa5c2d74.jpg](../iclr_results/2672_Revisiting Nearest Neighbor for Tabular Data_ A Deep Tabular Baseline Two Decades Later/tables/5e63f8579413b17a58e51976aa194719e2480dc71ac9abd4dd60d6d5aa5c2d74.jpg)

![94c969712d888023721bcc936a7c2eb0365204c3851d0f02f48efae83e0b3276.jpg](../iclr_results/2672_Revisiting Nearest Neighbor for Tabular Data_ A Deep Tabular Baseline Two Decades Later/tables/94c969712d888023721bcc936a7c2eb0365204c3851d0f02f48efae83e0b3276.jpg)

![b6f716462dcf7f461594d240b19b08bc177f4220049c63ee3472aea8b62935af.jpg](../iclr_results/2672_Revisiting Nearest Neighbor for Tabular Data_ A Deep Tabular Baseline Two Decades Later/tables/b6f716462dcf7f461594d240b19b08bc177f4220049c63ee3472aea8b62935af.jpg)

![bab4cd37ebc999208a8936f2e2ec0ba2da44b2043c77df69b49b0dc423946528.jpg](../iclr_results/2672_Revisiting Nearest Neighbor for Tabular Data_ A Deep Tabular Baseline Two Decades Later/tables/bab4cd37ebc999208a8936f2e2ec0ba2da44b2043c77df69b49b0dc423946528.jpg)

![de541ce78b7d5962f85d519fa6633d0be3072f67b0beb6ca53beaf435aca8c88.jpg](../iclr_results/2672_Revisiting Nearest Neighbor for Tabular Data_ A Deep Tabular Baseline Two Decades Later/tables/de541ce78b7d5962f85d519fa6633d0be3072f67b0beb6ca53beaf435aca8c88.jpg)

## Injective flows for star-like manifolds


### Images

![0c5228bbf8b10670a5c7944f4fbd2fd83f388aed5a6fbd2f5cb3b36ff79b37b6.jpg](../iclr_results/2673_Injective flows for star-like manifolds/images/0c5228bbf8b10670a5c7944f4fbd2fd83f388aed5a6fbd2f5cb3b36ff79b37b6.jpg)

![13100c92b47edd81760c19fa42e2a8e50f6926108d95b3a431b1d5dc5e0ff294.jpg](../iclr_results/2673_Injective flows for star-like manifolds/images/13100c92b47edd81760c19fa42e2a8e50f6926108d95b3a431b1d5dc5e0ff294.jpg)

![41c80284226b18867fa7a0258322c5d1eda2bfcdc1305c9de3fe48b40716e8e0.jpg](../iclr_results/2673_Injective flows for star-like manifolds/images/41c80284226b18867fa7a0258322c5d1eda2bfcdc1305c9de3fe48b40716e8e0.jpg)

![472ce91d50f392501c5e0195d4fbcbd67ce3227b69050101db0a08856cc511f9.jpg](../iclr_results/2673_Injective flows for star-like manifolds/images/472ce91d50f392501c5e0195d4fbcbd67ce3227b69050101db0a08856cc511f9.jpg)

![549f7547538d89baaeb9854e3e16543286c0774f153d9c8706393f594f21f1db.jpg](../iclr_results/2673_Injective flows for star-like manifolds/images/549f7547538d89baaeb9854e3e16543286c0774f153d9c8706393f594f21f1db.jpg)

![556d058528ce33108efc5cae6651cb82ebb638524bc9fdd0ffab9dce924d58e7.jpg](../iclr_results/2673_Injective flows for star-like manifolds/images/556d058528ce33108efc5cae6651cb82ebb638524bc9fdd0ffab9dce924d58e7.jpg)

![5a9b6b64bbc9d4b0ef37dc4595c14ad5edbf39b70169ee2d636cf53dada9536e.jpg](../iclr_results/2673_Injective flows for star-like manifolds/images/5a9b6b64bbc9d4b0ef37dc4595c14ad5edbf39b70169ee2d636cf53dada9536e.jpg)

![67a196121b154e980d06685a0c67da2fc0e4badcbbfeebc24cf70b84387360e6.jpg](../iclr_results/2673_Injective flows for star-like manifolds/images/67a196121b154e980d06685a0c67da2fc0e4badcbbfeebc24cf70b84387360e6.jpg)

![8a3a317994ef0a3d2416175a83bdc5c5ae7875441a8e6ffab3c283552c5bbb3d.jpg](../iclr_results/2673_Injective flows for star-like manifolds/images/8a3a317994ef0a3d2416175a83bdc5c5ae7875441a8e6ffab3c283552c5bbb3d.jpg)

![8fa4de6e200d6e62f9eddbafff7f06b24425d822a0a729dd45655c8590e1021b.jpg](../iclr_results/2673_Injective flows for star-like manifolds/images/8fa4de6e200d6e62f9eddbafff7f06b24425d822a0a729dd45655c8590e1021b.jpg)

![9b6877ff70a8bd5adb1faaeedbb82d51e1239bea04d3789d6b493d04502e3d09.jpg](../iclr_results/2673_Injective flows for star-like manifolds/images/9b6877ff70a8bd5adb1faaeedbb82d51e1239bea04d3789d6b493d04502e3d09.jpg)

![ac98ceedf4f94c0729ffbece3ec76b511f0e65910485326323cf231b2e0dc505.jpg](../iclr_results/2673_Injective flows for star-like manifolds/images/ac98ceedf4f94c0729ffbece3ec76b511f0e65910485326323cf231b2e0dc505.jpg)

## Robustness of Quantum Algorithms for Nonconvex Optimization


### Images

![8f19112c8274812e5b8a1cdb25a1b454fd32be6c42dc825e3663b659113a37fe.jpg](../iclr_results/2674_Robustness of Quantum Algorithms for Nonconvex Optimization/images/8f19112c8274812e5b8a1cdb25a1b454fd32be6c42dc825e3663b659113a37fe.jpg)

### Tables

![aeb8497e4de9f2891aa61cb0ea882f17fa61fa805fe79337c2adcde8ec258db6.jpg](../iclr_results/2674_Robustness of Quantum Algorithms for Nonconvex Optimization/tables/aeb8497e4de9f2891aa61cb0ea882f17fa61fa805fe79337c2adcde8ec258db6.jpg)

![c0a6a6beed6f3da563495626aadc1d64e99c61c4145689a1176e18999e0d5e52.jpg](../iclr_results/2674_Robustness of Quantum Algorithms for Nonconvex Optimization/tables/c0a6a6beed6f3da563495626aadc1d64e99c61c4145689a1176e18999e0d5e52.jpg)

![de76db2e7f2c9404a92d871ca5ce252dbf9f3a0d098a65452d87bd240d269869.jpg](../iclr_results/2674_Robustness of Quantum Algorithms for Nonconvex Optimization/tables/de76db2e7f2c9404a92d871ca5ce252dbf9f3a0d098a65452d87bd240d269869.jpg)

## Simple is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation


### Images

![5225f84b396b6d7bbb0d12f069abaee9dc6b8a313c3b922464a34084b60fc903.jpg](../iclr_results/2675_Simple is Effective_ The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieva/images/5225f84b396b6d7bbb0d12f069abaee9dc6b8a313c3b922464a34084b60fc903.jpg)

![5bf3165915077bad2d4fcbe0d1e4c8d8662c99b60715a0c08ebce1198fdbf52f.jpg](../iclr_results/2675_Simple is Effective_ The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieva/images/5bf3165915077bad2d4fcbe0d1e4c8d8662c99b60715a0c08ebce1198fdbf52f.jpg)

![91a7c72b46242ebfa0f4897077185ce3122124c506dde0493d849c5fcad668cc.jpg](../iclr_results/2675_Simple is Effective_ The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieva/images/91a7c72b46242ebfa0f4897077185ce3122124c506dde0493d849c5fcad668cc.jpg)

![a38bc36c5f95f186b5874b347f9ef2ec6b1cbea604a6ca6399b99fe8aecdf100.jpg](../iclr_results/2675_Simple is Effective_ The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieva/images/a38bc36c5f95f186b5874b347f9ef2ec6b1cbea604a6ca6399b99fe8aecdf100.jpg)

![a640028e538e91de5c2a0fb404dc043ba8928c82021edb093327b0028957eb01.jpg](../iclr_results/2675_Simple is Effective_ The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieva/images/a640028e538e91de5c2a0fb404dc043ba8928c82021edb093327b0028957eb01.jpg)

![d8ce7e413557f9d3fa83ed7ea75e016806d8db4613e0f260a55b578ad12de4d6.jpg](../iclr_results/2675_Simple is Effective_ The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieva/images/d8ce7e413557f9d3fa83ed7ea75e016806d8db4613e0f260a55b578ad12de4d6.jpg)

![f42f4f9a3f2ee0f62ec673ed423fce85aef0690d5ff261c5292bf9def7d92bc4.jpg](../iclr_results/2675_Simple is Effective_ The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieva/images/f42f4f9a3f2ee0f62ec673ed423fce85aef0690d5ff261c5292bf9def7d92bc4.jpg)

![fd5c18428e9f9d38afb72ea90883bb3bd6ee427a4a9ae171f01e6d111be3a5b7.jpg](../iclr_results/2675_Simple is Effective_ The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieva/images/fd5c18428e9f9d38afb72ea90883bb3bd6ee427a4a9ae171f01e6d111be3a5b7.jpg)

### Tables

![1a2469b874e2d097cf2da55e4b75c42a186e64508b055bb50a7892ea05ef4939.jpg](../iclr_results/2675_Simple is Effective_ The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieva/tables/1a2469b874e2d097cf2da55e4b75c42a186e64508b055bb50a7892ea05ef4939.jpg)

![2a17b8af8833d220ab248d0c893aeac96775bf2b209cdc824bbf3709120cd22a.jpg](../iclr_results/2675_Simple is Effective_ The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieva/tables/2a17b8af8833d220ab248d0c893aeac96775bf2b209cdc824bbf3709120cd22a.jpg)

![2c22ac556edc00b04839c9f9d085f9ee8976a9c8e224caa8f1461714f586751d.jpg](../iclr_results/2675_Simple is Effective_ The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieva/tables/2c22ac556edc00b04839c9f9d085f9ee8976a9c8e224caa8f1461714f586751d.jpg)

![349bbd337a02136bb19d15e952a91cd20b31762b27b17bd5f1d075f01c24d1cc.jpg](../iclr_results/2675_Simple is Effective_ The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieva/tables/349bbd337a02136bb19d15e952a91cd20b31762b27b17bd5f1d075f01c24d1cc.jpg)

![467c668aeaedc2f2c0d60a679c78be4698bc25aaa8839c766707fd31b5dec4af.jpg](../iclr_results/2675_Simple is Effective_ The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieva/tables/467c668aeaedc2f2c0d60a679c78be4698bc25aaa8839c766707fd31b5dec4af.jpg)

![50286a598a1e5b7c28fd616d323ce93d2e325602620b79e94adea7f783c93e1d.jpg](../iclr_results/2675_Simple is Effective_ The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieva/tables/50286a598a1e5b7c28fd616d323ce93d2e325602620b79e94adea7f783c93e1d.jpg)

![5d1997cdbbb04ef149082906f0c0539856fb76f92764f7d2b01a0bd800e780eb.jpg](../iclr_results/2675_Simple is Effective_ The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieva/tables/5d1997cdbbb04ef149082906f0c0539856fb76f92764f7d2b01a0bd800e780eb.jpg)

![6617ea6372d2c9774ded0f894a3566f9b2f670c7b504a6068b5c218a6fc825dc.jpg](../iclr_results/2675_Simple is Effective_ The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieva/tables/6617ea6372d2c9774ded0f894a3566f9b2f670c7b504a6068b5c218a6fc825dc.jpg)

![b119dbd639c0eb8ea49aa14789be07513e1e654c7479b850a7a737826321fee1.jpg](../iclr_results/2675_Simple is Effective_ The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieva/tables/b119dbd639c0eb8ea49aa14789be07513e1e654c7479b850a7a737826321fee1.jpg)

![c74be5fbf7bd76cf04cfe8ae84308eae8ee298c2a03e6f672b8a7bbd8367d83d.jpg](../iclr_results/2675_Simple is Effective_ The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieva/tables/c74be5fbf7bd76cf04cfe8ae84308eae8ee298c2a03e6f672b8a7bbd8367d83d.jpg)

## Towards Calibrated Deep Clustering Network


### Images

![12d69c52ee3f3219d9e7ca6b6ce46e22ca98638b8af1313dfe4bc94bf76ee99a.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/images/12d69c52ee3f3219d9e7ca6b6ce46e22ca98638b8af1313dfe4bc94bf76ee99a.jpg)

![2421267a1a0c65aa171502a9d6f597c1ffef5fc206f761637cda0927166ff979.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/images/2421267a1a0c65aa171502a9d6f597c1ffef5fc206f761637cda0927166ff979.jpg)

![25a8b34fc4d6f0f804689c0c79e336a835fe7b129bbdad613798fdeff7ac8ca8.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/images/25a8b34fc4d6f0f804689c0c79e336a835fe7b129bbdad613798fdeff7ac8ca8.jpg)

![2f35eb9a987a77489bad7c29eba02cb5c14f2e8646ccab7baf54d4580f5dc1a3.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/images/2f35eb9a987a77489bad7c29eba02cb5c14f2e8646ccab7baf54d4580f5dc1a3.jpg)

![42edeec050a58f92dce0f3e4689f26f22414f6ed0ba0a94902751571d13b7927.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/images/42edeec050a58f92dce0f3e4689f26f22414f6ed0ba0a94902751571d13b7927.jpg)

![5d3fd23e722b3961febeefaa4864ee9476da20a3f91746039ab52bef94b5138b.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/images/5d3fd23e722b3961febeefaa4864ee9476da20a3f91746039ab52bef94b5138b.jpg)

![6b99129990a144b25b12bf38e9747f4e9586669a94fdf86c9a8cc0e83cd7aedd.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/images/6b99129990a144b25b12bf38e9747f4e9586669a94fdf86c9a8cc0e83cd7aedd.jpg)

![809f754fddc76f5cbdd7d3cfe7ba0e1623c74cd32764971a04f9ffd6c36e484d.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/images/809f754fddc76f5cbdd7d3cfe7ba0e1623c74cd32764971a04f9ffd6c36e484d.jpg)

![9ae9cacd7fb009e8a5aa6c99f19247e67b5edfaae51350d328b282bb790e1e4e.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/images/9ae9cacd7fb009e8a5aa6c99f19247e67b5edfaae51350d328b282bb790e1e4e.jpg)

![b08b6d94550401f5ed0968ddb11da74bb3aa235ff7dad02ff5efae8f16670526.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/images/b08b6d94550401f5ed0968ddb11da74bb3aa235ff7dad02ff5efae8f16670526.jpg)

![ba843e2a37d6c49be7d1bbaca28f32efd3901735175c449551729fcf035e55dd.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/images/ba843e2a37d6c49be7d1bbaca28f32efd3901735175c449551729fcf035e55dd.jpg)

### Tables

![041aa0ec8c2f5e2aa11824512e2da5a1ec5ce145332294a8cbd8220d1e8e3b40.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/tables/041aa0ec8c2f5e2aa11824512e2da5a1ec5ce145332294a8cbd8220d1e8e3b40.jpg)

![08baa124f09b15b9fe953bea1c4c052191991225cf04d36372c14eb3a02b83c0.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/tables/08baa124f09b15b9fe953bea1c4c052191991225cf04d36372c14eb3a02b83c0.jpg)

![19bc1103a56fc852ce3f8d5e9664e09f241b3e07500614bc24bb2ca013350008.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/tables/19bc1103a56fc852ce3f8d5e9664e09f241b3e07500614bc24bb2ca013350008.jpg)

![408e27e4dc3f3ba752ee45430ce0333cd791b99ec92fea1bef07da7d27bd6b45.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/tables/408e27e4dc3f3ba752ee45430ce0333cd791b99ec92fea1bef07da7d27bd6b45.jpg)

![40b51ea1c6c0301c1bbf10406467ee1fe3cf19e46287670c818af14092886b62.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/tables/40b51ea1c6c0301c1bbf10406467ee1fe3cf19e46287670c818af14092886b62.jpg)

![4157ffc85cbd7a4dd4fa79bad71eede3771c1e280a33345402f980970d65bba2.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/tables/4157ffc85cbd7a4dd4fa79bad71eede3771c1e280a33345402f980970d65bba2.jpg)

![972876f736060e26ebac7dbd35c20999d6d557342532a698609702175120aa1e.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/tables/972876f736060e26ebac7dbd35c20999d6d557342532a698609702175120aa1e.jpg)

![af41f75790e30fb04c756b735d0528cdc67b80b4ef51a8e66bc94f4a01f361a0.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/tables/af41f75790e30fb04c756b735d0528cdc67b80b4ef51a8e66bc94f4a01f361a0.jpg)

![b8b4f4d03d48fbd255e38121df4c73bb7b71d80f3fae1da68b0c845673ae6253.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/tables/b8b4f4d03d48fbd255e38121df4c73bb7b71d80f3fae1da68b0c845673ae6253.jpg)

![bc28d474174194d2991f1d7853316b975d31aaa5ac3ca9bc371b89534979c063.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/tables/bc28d474174194d2991f1d7853316b975d31aaa5ac3ca9bc371b89534979c063.jpg)

![c2b664a7b2dc47e345ca6832f0c7eb6aaca35dbedfbdce54d4577dfd36993bb3.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/tables/c2b664a7b2dc47e345ca6832f0c7eb6aaca35dbedfbdce54d4577dfd36993bb3.jpg)

![c3fc84aa116010a9250f9358d352842de1edefef632f85852677f170e969d039.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/tables/c3fc84aa116010a9250f9358d352842de1edefef632f85852677f170e969d039.jpg)

![c441abd74a02ddd295683aaec8b7be3b81d97d8624e810817da98ab75ca4ac23.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/tables/c441abd74a02ddd295683aaec8b7be3b81d97d8624e810817da98ab75ca4ac23.jpg)

![ceef76a3a857eaed9fc31a9a7dfe238ac226e06c3b4b3bbb749e44698dce59d8.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/tables/ceef76a3a857eaed9fc31a9a7dfe238ac226e06c3b4b3bbb749e44698dce59d8.jpg)

![e86e3adf98a1c2f0148273139607c7421990787f978dd5129f6dfa51c969247e.jpg](../iclr_results/2676_Towards Calibrated Deep Clustering Network/tables/e86e3adf98a1c2f0148273139607c7421990787f978dd5129f6dfa51c969247e.jpg)

## Multiagent Finetuning: Self Improvement with Diverse Reasoning Chains

### Images

![0b28879d6f4daf677881dba08be1b90fef5984645c8f3276dbf639570dd7f4cc.jpg](../iclr_results/2677_Multiagent Finetuning_ Self Improvement with Diverse Reasoning Chains/images/0b28879d6f4daf677881dba08be1b90fef5984645c8f3276dbf639570dd7f4cc.jpg)

![19b378005e34dc56aa9426eb8e24f8ce07fc5e5acdce33fb69c2ef1a4a6d356e.jpg](../iclr_results/2677_Multiagent Finetuning_ Self Improvement with Diverse Reasoning Chains/images/19b378005e34dc56aa9426eb8e24f8ce07fc5e5acdce33fb69c2ef1a4a6d356e.jpg)

![1a558f4c2bb12769ec515109e5401c10d0c56da0f5e7956f80b28f5bd9941903.jpg](../iclr_results/2677_Multiagent Finetuning_ Self Improvement with Diverse Reasoning Chains/images/1a558f4c2bb12769ec515109e5401c10d0c56da0f5e7956f80b28f5bd9941903.jpg)

![453507970542b3949f7bc7c80457879d0727218abbcbfd8a5a595c84fec4fff6.jpg](../iclr_results/2677_Multiagent Finetuning_ Self Improvement with Diverse Reasoning Chains/images/453507970542b3949f7bc7c80457879d0727218abbcbfd8a5a595c84fec4fff6.jpg)

![47da55e7da4a3575b05b8b52ff2d51f40174787cf7a68d9f022a918e44fb86b3.jpg](../iclr_results/2677_Multiagent Finetuning_ Self Improvement with Diverse Reasoning Chains/images/47da55e7da4a3575b05b8b52ff2d51f40174787cf7a68d9f022a918e44fb86b3.jpg)

![50cae0eb6b90da719fef3494519a4081d935458f6788a19ce5d9e1d1376b384c.jpg](../iclr_results/2677_Multiagent Finetuning_ Self Improvement with Diverse Reasoning Chains/images/50cae0eb6b90da719fef3494519a4081d935458f6788a19ce5d9e1d1376b384c.jpg)

![6c63c62c0ddd386efac81cb9da5e21472583c0c9f680ec8477ec6a427a6d35d6.jpg](../iclr_results/2677_Multiagent Finetuning_ Self Improvement with Diverse Reasoning Chains/images/6c63c62c0ddd386efac81cb9da5e21472583c0c9f680ec8477ec6a427a6d35d6.jpg)

![7019644753b2e89cf6686f72bab5cbe30c4609f74d33bbaa157fffe9e34f3ef6.jpg](../iclr_results/2677_Multiagent Finetuning_ Self Improvement with Diverse Reasoning Chains/images/7019644753b2e89cf6686f72bab5cbe30c4609f74d33bbaa157fffe9e34f3ef6.jpg)

![8663b47040436de87cb978aaa465b5574521a1cca45adf71d209c17616889bb3.jpg](../iclr_results/2677_Multiagent Finetuning_ Self Improvement with Diverse Reasoning Chains/images/8663b47040436de87cb978aaa465b5574521a1cca45adf71d209c17616889bb3.jpg)

![ae586f162a290d067200d915e8669bad4dcd45704dd56b25eea215c59d228084.jpg](../iclr_results/2677_Multiagent Finetuning_ Self Improvement with Diverse Reasoning Chains/images/ae586f162a290d067200d915e8669bad4dcd45704dd56b25eea215c59d228084.jpg)

![ca8988b70a39f5fd4a34d2052f0a95fd51622054ff65e08d486c181505935b2b.jpg](../iclr_results/2677_Multiagent Finetuning_ Self Improvement with Diverse Reasoning Chains/images/ca8988b70a39f5fd4a34d2052f0a95fd51622054ff65e08d486c181505935b2b.jpg)

![d3423677d4ee36a1e3554fd0bf16c876718390e85b5baa8289e88f6ec8b78a0f.jpg](../iclr_results/2677_Multiagent Finetuning_ Self Improvement with Diverse Reasoning Chains/images/d3423677d4ee36a1e3554fd0bf16c876718390e85b5baa8289e88f6ec8b78a0f.jpg)

![e60fbffc49799cf0cb8be8dd402adff8db5551def3a1d538081ebc862a209e07.jpg](../iclr_results/2677_Multiagent Finetuning_ Self Improvement with Diverse Reasoning Chains/images/e60fbffc49799cf0cb8be8dd402adff8db5551def3a1d538081ebc862a209e07.jpg)

### Tables

![2abda7e9d90eabfe4e45e3b9d95c24a1e4593cd8538756632d357df066269d5d.jpg](../iclr_results/2677_Multiagent Finetuning_ Self Improvement with Diverse Reasoning Chains/tables/2abda7e9d90eabfe4e45e3b9d95c24a1e4593cd8538756632d357df066269d5d.jpg)

![3afa552af3b63184e2a8edb0b31da843266bce1bc3b803f4e68f437a83a3e06a.jpg](../iclr_results/2677_Multiagent Finetuning_ Self Improvement with Diverse Reasoning Chains/tables/3afa552af3b63184e2a8edb0b31da843266bce1bc3b803f4e68f437a83a3e06a.jpg)

![4840280f3d152ce23c1e47d51ebe62bf7121bb1b8e0c7090d01fe8d75a6f909b.jpg](../iclr_results/2677_Multiagent Finetuning_ Self Improvement with Diverse Reasoning Chains/tables/4840280f3d152ce23c1e47d51ebe62bf7121bb1b8e0c7090d01fe8d75a6f909b.jpg)

![7cf43ccca64e39e1df1410395effc513064926b4170dbed999ff31590bbeb269.jpg](../iclr_results/2677_Multiagent Finetuning_ Self Improvement with Diverse Reasoning Chains/tables/7cf43ccca64e39e1df1410395effc513064926b4170dbed999ff31590bbeb269.jpg)

![8bf26e2d5f692d2a6abbbb380869c6592b974d284d59dc8da298c6007d0721ce.jpg](../iclr_results/2677_Multiagent Finetuning_ Self Improvement with Diverse Reasoning Chains/tables/8bf26e2d5f692d2a6abbbb380869c6592b974d284d59dc8da298c6007d0721ce.jpg)

![953f93c0a920c48e7b1660c0110e4901b29490763ca94f820072048cac423937.jpg](../iclr_results/2677_Multiagent Finetuning_ Self Improvement with Diverse Reasoning Chains/tables/953f93c0a920c48e7b1660c0110e4901b29490763ca94f820072048cac423937.jpg)

![b434224a9663609219fed0afb2733bba6b6b3d35c0fda6e3d24ce5f98dbd29f7.jpg](../iclr_results/2677_Multiagent Finetuning_ Self Improvement with Diverse Reasoning Chains/tables/b434224a9663609219fed0afb2733bba6b6b3d35c0fda6e3d24ce5f98dbd29f7.jpg)

![ef9334e3515b345e77191e09677f3abe19e1b86130a71cf35aefed9e8411625b.jpg](../iclr_results/2677_Multiagent Finetuning_ Self Improvement with Diverse Reasoning Chains/tables/ef9334e3515b345e77191e09677f3abe19e1b86130a71cf35aefed9e8411625b.jpg)
