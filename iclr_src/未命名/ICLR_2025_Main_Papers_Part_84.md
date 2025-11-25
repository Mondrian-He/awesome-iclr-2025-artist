# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 84 of 100

## 目录 (Table of Contents)

1. [Distilling Reinforcement Learning Algorithms for In-Context Model-Based Planning](#Distilling-Reinforcement-Learning-Algorithms-for-In-Context-Model-Based-Planning)
2. [On the Linear Speedup of Personalized Federated Reinforcement Learning with Shared Representations](#On-the-Linear-Speedup-of-Personalized-Federated-Reinforcement-Learning-with-Shared-Representations)
3. [Adaptive teachers for amortized samplers](#Adaptive-teachers-for-amortized-samplers)
4. [Fourier Sliced-Wasserstein Embedding for Multisets and Measures](#Fourier-Sliced-Wasserstein-Embedding-for-Multisets-and-Measures)
5. [Learning Molecular Representation in a Cell](#Learning-Molecular-Representation-in-a-Cell)
6. [T2V-Turbo-v2: Enhancing Video Model Post-Training through Data, Reward, and Conditional Guidance Design](#T2V-Turbo-v2-Enhancing-Video-Model-Post-Training-through-Data-Reward-and-Conditional-Guidance-Design)
7. [ADAM Optimization with Adaptive Batch Selection](#ADAM-Optimization-with-Adaptive-Batch-Selection)
8. [Efficient and Trustworthy Causal Discovery with Latent Variables and Complex Relations](#Efficient-and-Trustworthy-Causal-Discovery-with-Latent-Variables-and-Complex-Relations)
9. [ControlAR: Controllable Image Generation with Autoregressive Models](#ControlAR-Controllable-Image-Generation-with-Autoregressive-Models)
10. [Number Cookbook: Number Understanding of Language Models and How to Improve It](#Number-Cookbook-Number-Understanding-of-Language-Models-and-How-to-Improve-It)
11. [Efficient Residual Learning with Mixture-of-Experts for Universal Dexterous Grasping](#Efficient-Residual-Learning-with-Mixture-of-Experts-for-Universal-Dexterous-Grasping)
12. [qNBO: quasi-Newton Meets Bilevel Optimization](#qNBO-quasi-Newton-Meets-Bilevel-Optimization)
13. [MatryoshkaKV: Adaptive KV Compression via Trainable Orthogonal Projection](#MatryoshkaKV-Adaptive-KV-Compression-via-Trainable-Orthogonal-Projection)
14. [Grounding by Trying: LLMs with Reinforcement Learning-Enhanced Retrieval](#Grounding-by-Trying-LLMs-with-Reinforcement-Learning-Enhanced-Retrieval)
15. [Towards Bridging Generalization and Expressivity of Graph Neural Networks](#Towards-Bridging-Generalization-and-Expressivity-of-Graph-Neural-Networks)
16. [LucidPPN: Unambiguous Prototypical Parts Network for User-centric Interpretable Computer Vision](#LucidPPN-Unambiguous-Prototypical-Parts-Network-for-User-centric-Interpretable-Computer-Vision)
17. [Generating CAD Code with Vision-Language Models for 3D Designs](#Generating-CAD-Code-with-Vision-Language-Models-for-3D-Designs)
18. [CBMA: Improving Conformal Prediction through Bayesian Model Averaging](#CBMA-Improving-Conformal-Prediction-through-Bayesian-Model-Averaging)
19. [Human-inspired Episodic Memory for Infinite Context LLMs](#Human-inspired-Episodic-Memory-for-Infinite-Context-LLMs)
20. [Constructing Confidence Intervals for Average Treatment Effects from Multiple Datasets](#Constructing-Confidence-Intervals-for-Average-Treatment-Effects-from-Multiple-Datasets)
21. [Horizon Generalization in Reinforcement Learning](#Horizon-Generalization-in-Reinforcement-Learning)
22. [Flavors of Margin: Implicit Bias of Steepest Descent in Homogeneous Neural Networks](#Flavors-of-Margin-Implicit-Bias-of-Steepest-Descent-in-Homogeneous-Neural-Networks)
23. [Mix-LN: Unleashing the Power of Deeper Layers by Combining Pre-LN and Post-LN](#Mix-LN-Unleashing-the-Power-of-Deeper-Layers-by-Combining-Pre-LN-and-Post-LN)
24. [Kolmogorov-Arnold Transformer](#Kolmogorov-Arnold-Transformer)
25. [Human Simulacra: Benchmarking the Personification of Large Language Models](#Human-Simulacra-Benchmarking-the-Personification-of-Large-Language-Models)
26. [Generative Representational Instruction Tuning](#Generative-Representational-Instruction-Tuning)
27. [ROUTE: Robust Multitask Tuning and Collaboration for Text-to-SQL](#ROUTE-Robust-Multitask-Tuning-and-Collaboration-for-Text-to-SQL)
28. [MuHBoost: Multi-Label Boosting For Practical Longitudinal Human Behavior Modeling](#MuHBoost-Multi-Label-Boosting-For-Practical-Longitudinal-Human-Behavior-Modeling)
29. [Linear Partial Gromov-Wasserstein Embedding](#Linear-Partial-Gromov-Wasserstein-Embedding)
30. [PRISM: Privacy-Preserving Improved Stochastic Masking for Federated Generative Models](#PRISM-Privacy-Preserving-Improved-Stochastic-Masking-for-Federated-Generative-Models)
31. [Hot-pluggable Federated Learning: Bridging General and Personalized FL via Dynamic Selection](#Hot-pluggable-Federated-Learning-Bridging-General-and-Personalized-FL-via-Dynamic-Selection)
32. [The Foundations of Tokenization: Statistical and Computational Concerns](#The-Foundations-of-Tokenization-Statistical-and-Computational-Concerns)
33. [SleepSMC: Ubiquitous Sleep Staging via Supervised Multimodal Coordination](#SleepSMC-Ubiquitous-Sleep-Staging-via-Supervised-Multimodal-Coordination)
34. [FormalAlign: Automated Alignment Evaluation for Autoformalization](#FormalAlign-Automated-Alignment-Evaluation-for-Autoformalization)
35. [TopoGaussian: Inferring Internal Topology Structures from Visual Clues](#TopoGaussian-Inferring-Internal-Topology-Structures-from-Visual-Clues)
36. [Understanding Constraint Inference in Safety-Critical Inverse Reinforcement Learning](#Understanding-Constraint-Inference-in-Safety-Critical-Inverse-Reinforcement-Learning)
37. [FLIP: Flow-Centric Generative Planning as General-Purpose Manipulation World Model](#FLIP-Flow-Centric-Generative-Planning-as-General-Purpose-Manipulation-World-Model)

---


## Distilling Reinforcement Learning Algorithms for In-Context Model-Based Planning

### Images

![290fb527f90959a25fb754b29baca05e517111d3bb3c02cfc498bb5dd867898a.jpg](../iclr_results/3085_What%20Matters%20When%20Repurposing%20Diffusion%20Models%20for%20General%20Dense%20Perception%20Tasks_/images/290fb527f90959a25fb754b29baca05e517111d3bb3c02cfc498bb5dd867898a.jpg)

![6d9caa0df78ea73c627798cf2c244ce76033f24ec80e44acbab2027e45a24eac.jpg](../iclr_results/3085_What%20Matters%20When%20Repurposing%20Diffusion%20Models%20for%20General%20Dense%20Perception%20Tasks_/images/6d9caa0df78ea73c627798cf2c244ce76033f24ec80e44acbab2027e45a24eac.jpg)

![a53a4bff44f6180b3e6fa055412cf4c201fa544039aa534f8026d814c5935336.jpg](../iclr_results/3085_What%20Matters%20When%20Repurposing%20Diffusion%20Models%20for%20General%20Dense%20Perception%20Tasks_/images/a53a4bff44f6180b3e6fa055412cf4c201fa544039aa534f8026d814c5935336.jpg)

![aee0136464a6bcee5195f7bf28192435f5f2917ce05e4763e4eee94968cd7dce.jpg](../iclr_results/3085_What%20Matters%20When%20Repurposing%20Diffusion%20Models%20for%20General%20Dense%20Perception%20Tasks_/images/aee0136464a6bcee5195f7bf28192435f5f2917ce05e4763e4eee94968cd7dce.jpg)

![e17d8384e8b215c190a7d4bcda63ece01bf9330d6add0238ab5967060c98ba5d.jpg](../iclr_results/3085_What%20Matters%20When%20Repurposing%20Diffusion%20Models%20for%20General%20Dense%20Perception%20Tasks_/images/e17d8384e8b215c190a7d4bcda63ece01bf9330d6add0238ab5967060c98ba5d.jpg)

### Tables

![12254fc28c082a6dd67d8e4e7d7f671a862c52e6b67f13e498d339f56f35a28c.jpg](../iclr_results/3085_What%20Matters%20When%20Repurposing%20Diffusion%20Models%20for%20General%20Dense%20Perception%20Tasks_/tables/12254fc28c082a6dd67d8e4e7d7f671a862c52e6b67f13e498d339f56f35a28c.jpg)

![24a748e09229d1032973cbbe2ac69f3f7e5e8b75b4088d0e197faa74e8a03cf7.jpg](../iclr_results/3085_What%20Matters%20When%20Repurposing%20Diffusion%20Models%20for%20General%20Dense%20Perception%20Tasks_/tables/24a748e09229d1032973cbbe2ac69f3f7e5e8b75b4088d0e197faa74e8a03cf7.jpg)

![2b7a3d25c9645e04237ccdac098f721c2a38d58ee87deda2ce4fb94bce45e19c.jpg](../iclr_results/3085_What%20Matters%20When%20Repurposing%20Diffusion%20Models%20for%20General%20Dense%20Perception%20Tasks_/tables/2b7a3d25c9645e04237ccdac098f721c2a38d58ee87deda2ce4fb94bce45e19c.jpg)

![518955166eac5ab028508f2ca5b6a3d5809ffe35dd68afcbba34aef73e48afe7.jpg](../iclr_results/3085_What%20Matters%20When%20Repurposing%20Diffusion%20Models%20for%20General%20Dense%20Perception%20Tasks_/tables/518955166eac5ab028508f2ca5b6a3d5809ffe35dd68afcbba34aef73e48afe7.jpg)

![917d9d378af08167836a8d3d7bfe4f7ac222050b4b6a65f305dc134e22aa217e.jpg](../iclr_results/3085_What%20Matters%20When%20Repurposing%20Diffusion%20Models%20for%20General%20Dense%20Perception%20Tasks_/tables/917d9d378af08167836a8d3d7bfe4f7ac222050b4b6a65f305dc134e22aa217e.jpg)

![945427abedcc6cf8b595e5aae5ed3214ecb014c08bb36756a6e303f75b8441a2.jpg](../iclr_results/3085_What%20Matters%20When%20Repurposing%20Diffusion%20Models%20for%20General%20Dense%20Perception%20Tasks_/tables/945427abedcc6cf8b595e5aae5ed3214ecb014c08bb36756a6e303f75b8441a2.jpg)

![c35eda08fa7e67ffc1efc8609084692dd20821074046b97644856ebf5a1e59f4.jpg](../iclr_results/3085_What%20Matters%20When%20Repurposing%20Diffusion%20Models%20for%20General%20Dense%20Perception%20Tasks_/tables/c35eda08fa7e67ffc1efc8609084692dd20821074046b97644856ebf5a1e59f4.jpg)

![c9142528cd4444f7d27ab521145209179dd7be27b62e25eceee18485ec025c66.jpg](../iclr_results/3085_What%20Matters%20When%20Repurposing%20Diffusion%20Models%20for%20General%20Dense%20Perception%20Tasks_/tables/c9142528cd4444f7d27ab521145209179dd7be27b62e25eceee18485ec025c66.jpg)

![de0714a45516833fea208187c9415a30c0d609665a1a0c705cac2485b01955c1.jpg](../iclr_results/3085_What%20Matters%20When%20Repurposing%20Diffusion%20Models%20for%20General%20Dense%20Perception%20Tasks_/tables/de0714a45516833fea208187c9415a30c0d609665a1a0c705cac2485b01955c1.jpg)

![eabaad1e52d6055d99e92bb9718ff3ca7634877391f00b4131a63ceaac8e8dee.jpg](../iclr_results/3085_What%20Matters%20When%20Repurposing%20Diffusion%20Models%20for%20General%20Dense%20Perception%20Tasks_/tables/eabaad1e52d6055d99e92bb9718ff3ca7634877391f00b4131a63ceaac8e8dee.jpg)

## Distilling Reinforcement Learning Algorithms for In-Context Model-Based Planning


### Images

![7018b7a77a5699be6136aa3066ef2914d3685e6af9577af3e269c1e2bbcbdac5.jpg](../iclr_results/3086_Distilling%20Reinforcement%20Learning%20Algorithms%20for%20In-Context%20Model-Based%20Planning/images/7018b7a77a5699be6136aa3066ef2914d3685e6af9577af3e269c1e2bbcbdac5.jpg)

![9013f3d291ac50afcc6334ab5cae9b7ea85480cdf9a3cb16f4152a1f2675d30b.jpg](../iclr_results/3086_Distilling%20Reinforcement%20Learning%20Algorithms%20for%20In-Context%20Model-Based%20Planning/images/9013f3d291ac50afcc6334ab5cae9b7ea85480cdf9a3cb16f4152a1f2675d30b.jpg)

![93df556f6252f385f4b1206b7d103414c959cfceaa8ef7b215ecd0472e2c339a.jpg](../iclr_results/3086_Distilling%20Reinforcement%20Learning%20Algorithms%20for%20In-Context%20Model-Based%20Planning/images/93df556f6252f385f4b1206b7d103414c959cfceaa8ef7b215ecd0472e2c339a.jpg)

![9a0d8cd1e66c3dcee7dc68714f82791cfbd7cb42cd1533b02246ccca4ef69006.jpg](../iclr_results/3086_Distilling%20Reinforcement%20Learning%20Algorithms%20for%20In-Context%20Model-Based%20Planning/images/9a0d8cd1e66c3dcee7dc68714f82791cfbd7cb42cd1533b02246ccca4ef69006.jpg)

### Tables

![035f9f8285c91ef4ea40679796c41dbe957eb60d1b4c5ea9daf7dba5315b0ebd.jpg](../iclr_results/3086_Distilling%20Reinforcement%20Learning%20Algorithms%20for%20In-Context%20Model-Based%20Planning/tables/035f9f8285c91ef4ea40679796c41dbe957eb60d1b4c5ea9daf7dba5315b0ebd.jpg)

![27c1a5c11892715eec17b35655e50af3e73a42a42a84ad5b4c37a802bb9e92e7.jpg](../iclr_results/3086_Distilling%20Reinforcement%20Learning%20Algorithms%20for%20In-Context%20Model-Based%20Planning/tables/27c1a5c11892715eec17b35655e50af3e73a42a42a84ad5b4c37a802bb9e92e7.jpg)

![312174d09741d0d45ad9c8ab6f74651a1253511fd220ce568e79ae395dc45caf.jpg](../iclr_results/3086_Distilling%20Reinforcement%20Learning%20Algorithms%20for%20In-Context%20Model-Based%20Planning/tables/312174d09741d0d45ad9c8ab6f74651a1253511fd220ce568e79ae395dc45caf.jpg)

![70241b190f7462a986ae6248c378112cd2499843dbbda3d9442912cf7a39301b.jpg](../iclr_results/3086_Distilling%20Reinforcement%20Learning%20Algorithms%20for%20In-Context%20Model-Based%20Planning/tables/70241b190f7462a986ae6248c378112cd2499843dbbda3d9442912cf7a39301b.jpg)

![7ec851283f42d7d19b48bc4470eb29696a2a02a71b51fe7c9003525dbbe69c11.jpg](../iclr_results/3086_Distilling%20Reinforcement%20Learning%20Algorithms%20for%20In-Context%20Model-Based%20Planning/tables/7ec851283f42d7d19b48bc4470eb29696a2a02a71b51fe7c9003525dbbe69c11.jpg)

![8f236760e07330094e5ca71e9c6519efbd33f139c266d7658442def186331b92.jpg](../iclr_results/3086_Distilling%20Reinforcement%20Learning%20Algorithms%20for%20In-Context%20Model-Based%20Planning/tables/8f236760e07330094e5ca71e9c6519efbd33f139c266d7658442def186331b92.jpg)

![b9e790c30e84dd26373b1e745f4a37496a07d5eff35dc9b88f515445d0755624.jpg](../iclr_results/3086_Distilling%20Reinforcement%20Learning%20Algorithms%20for%20In-Context%20Model-Based%20Planning/tables/b9e790c30e84dd26373b1e745f4a37496a07d5eff35dc9b88f515445d0755624.jpg)

![c0bc4c37ecaafb724b4d0b899886b6cfd5a6b5bdb45474c0c99fb08d5e6ec035.jpg](../iclr_results/3086_Distilling%20Reinforcement%20Learning%20Algorithms%20for%20In-Context%20Model-Based%20Planning/tables/c0bc4c37ecaafb724b4d0b899886b6cfd5a6b5bdb45474c0c99fb08d5e6ec035.jpg)

![d4f19bce720c6216309a45dfa5ca9c311450735b76b642e0ff581b4eb07d89f9.jpg](../iclr_results/3086_Distilling%20Reinforcement%20Learning%20Algorithms%20for%20In-Context%20Model-Based%20Planning/tables/d4f19bce720c6216309a45dfa5ca9c311450735b76b642e0ff581b4eb07d89f9.jpg)

![fb59fc327e37c64a1f70f2aae568872e66563066ba076c7b680e0e2ba3d07d08.jpg](../iclr_results/3086_Distilling%20Reinforcement%20Learning%20Algorithms%20for%20In-Context%20Model-Based%20Planning/tables/fb59fc327e37c64a1f70f2aae568872e66563066ba076c7b680e0e2ba3d07d08.jpg)

## On the Linear Speedup of Personalized Federated Reinforcement Learning with Shared Representations


### Images

![00d28352c2b21c36786018263f0d846397f4c84ab8606f129f63bdb7ff5875e5.jpg](../iclr_results/3087_On%20the%20Linear%20Speedup%20of%20Personalized%20Federated%20Reinforcement%20Learning%20with%20Shared%20Representations/images/00d28352c2b21c36786018263f0d846397f4c84ab8606f129f63bdb7ff5875e5.jpg)

![0d7292e3baf39a3e48cc68f74d5e0a9fe0779153b25ab6d35b1b2829066e8d98.jpg](../iclr_results/3087_On%20the%20Linear%20Speedup%20of%20Personalized%20Federated%20Reinforcement%20Learning%20with%20Shared%20Representations/images/0d7292e3baf39a3e48cc68f74d5e0a9fe0779153b25ab6d35b1b2829066e8d98.jpg)

![18e978b878213cd3f4136f20c01abdb8d7995bf31b70d2d477efd79644d2b52c.jpg](../iclr_results/3087_On%20the%20Linear%20Speedup%20of%20Personalized%20Federated%20Reinforcement%20Learning%20with%20Shared%20Representations/images/18e978b878213cd3f4136f20c01abdb8d7995bf31b70d2d477efd79644d2b52c.jpg)

![2224dfe0d6cb7c3e9856839b4b737e59ff5e311ae20960b0d617f4f306c1bc39.jpg](../iclr_results/3087_On%20the%20Linear%20Speedup%20of%20Personalized%20Federated%20Reinforcement%20Learning%20with%20Shared%20Representations/images/2224dfe0d6cb7c3e9856839b4b737e59ff5e311ae20960b0d617f4f306c1bc39.jpg)

![2e54f6d74d10f658ed6ab70eeb66949c0fb48936f8442eb6543bf63e3755f408.jpg](../iclr_results/3087_On%20the%20Linear%20Speedup%20of%20Personalized%20Federated%20Reinforcement%20Learning%20with%20Shared%20Representations/images/2e54f6d74d10f658ed6ab70eeb66949c0fb48936f8442eb6543bf63e3755f408.jpg)

![2ed4bc61621c0f28ec490d7b2e5fa9727d194dfd346dd6801bc6f4ddbc3def01.jpg](../iclr_results/3087_On%20the%20Linear%20Speedup%20of%20Personalized%20Federated%20Reinforcement%20Learning%20with%20Shared%20Representations/images/2ed4bc61621c0f28ec490d7b2e5fa9727d194dfd346dd6801bc6f4ddbc3def01.jpg)

![333edb38fa21f259fe6c3e75dda525fb363d6b254fd3f5125d2df53b66f3dbdf.jpg](../iclr_results/3087_On%20the%20Linear%20Speedup%20of%20Personalized%20Federated%20Reinforcement%20Learning%20with%20Shared%20Representations/images/333edb38fa21f259fe6c3e75dda525fb363d6b254fd3f5125d2df53b66f3dbdf.jpg)

![3564182184f01a2a5ce3899a8dea627c6da543fd52e5e554acb5912b9fd3934d.jpg](../iclr_results/3087_On%20the%20Linear%20Speedup%20of%20Personalized%20Federated%20Reinforcement%20Learning%20with%20Shared%20Representations/images/3564182184f01a2a5ce3899a8dea627c6da543fd52e5e554acb5912b9fd3934d.jpg)

![3606774b1108ec5ee4ee211de4be9550f472566836bc9802280c59902d651d5f.jpg](../iclr_results/3087_On%20the%20Linear%20Speedup%20of%20Personalized%20Federated%20Reinforcement%20Learning%20with%20Shared%20Representations/images/3606774b1108ec5ee4ee211de4be9550f472566836bc9802280c59902d651d5f.jpg)

![3b19d1821d7fe084c4a8727eeaa95c46178dcaeff1505ca2686d57807f5983ca.jpg](../iclr_results/3087_On%20the%20Linear%20Speedup%20of%20Personalized%20Federated%20Reinforcement%20Learning%20with%20Shared%20Representations/images/3b19d1821d7fe084c4a8727eeaa95c46178dcaeff1505ca2686d57807f5983ca.jpg)

![474fa2ffa94aaa5478431afa850e86c208c7efb7565424db15745e1d9312b264.jpg](../iclr_results/3087_On%20the%20Linear%20Speedup%20of%20Personalized%20Federated%20Reinforcement%20Learning%20with%20Shared%20Representations/images/474fa2ffa94aaa5478431afa850e86c208c7efb7565424db15745e1d9312b264.jpg)

![7351b7b87f5053a7f59a437497ebfc14fc89e280a2fd5de06825aa220e7e221d.jpg](../iclr_results/3087_On%20the%20Linear%20Speedup%20of%20Personalized%20Federated%20Reinforcement%20Learning%20with%20Shared%20Representations/images/7351b7b87f5053a7f59a437497ebfc14fc89e280a2fd5de06825aa220e7e221d.jpg)

![7a3117072e10912aeaf762fdeaa0b762ea5429e95bdf333beb34b1c29c55e31d.jpg](../iclr_results/3087_On%20the%20Linear%20Speedup%20of%20Personalized%20Federated%20Reinforcement%20Learning%20with%20Shared%20Representations/images/7a3117072e10912aeaf762fdeaa0b762ea5429e95bdf333beb34b1c29c55e31d.jpg)

![8d3f4b05941414402302aa4eb1b9f3342f8ca1f026c93d7aef5001ef11372781.jpg](../iclr_results/3087_On%20the%20Linear%20Speedup%20of%20Personalized%20Federated%20Reinforcement%20Learning%20with%20Shared%20Representations/images/8d3f4b05941414402302aa4eb1b9f3342f8ca1f026c93d7aef5001ef11372781.jpg)

![abf2cea88b4cecd3814d04063f13b132bea02c2497fe59da47d753d19be261e7.jpg](../iclr_results/3087_On%20the%20Linear%20Speedup%20of%20Personalized%20Federated%20Reinforcement%20Learning%20with%20Shared%20Representations/images/abf2cea88b4cecd3814d04063f13b132bea02c2497fe59da47d753d19be261e7.jpg)

![b0abe41fea716aa3e8a8f3ae882547ef5ac77c217e4273e745fab27e038f696d.jpg](../iclr_results/3087_On%20the%20Linear%20Speedup%20of%20Personalized%20Federated%20Reinforcement%20Learning%20with%20Shared%20Representations/images/b0abe41fea716aa3e8a8f3ae882547ef5ac77c217e4273e745fab27e038f696d.jpg)

![f9b8b736b98d5bc8dda0f1834971037671f8ddb82987f5fee914a5134625dca7.jpg](../iclr_results/3087_On%20the%20Linear%20Speedup%20of%20Personalized%20Federated%20Reinforcement%20Learning%20with%20Shared%20Representations/images/f9b8b736b98d5bc8dda0f1834971037671f8ddb82987f5fee914a5134625dca7.jpg)

### Tables

![48099917ba0138ff287a3b919288d4e3f5e0a2c90cff6ff046a528daec493c6b.jpg](../iclr_results/3087_On%20the%20Linear%20Speedup%20of%20Personalized%20Federated%20Reinforcement%20Learning%20with%20Shared%20Representations/tables/48099917ba0138ff287a3b919288d4e3f5e0a2c90cff6ff046a528daec493c6b.jpg)

![5a85dbc452fbeb01ab14a84c54ef79ec2a40e2e18103c1d18d79fa075adad463.jpg](../iclr_results/3087_On%20the%20Linear%20Speedup%20of%20Personalized%20Federated%20Reinforcement%20Learning%20with%20Shared%20Representations/tables/5a85dbc452fbeb01ab14a84c54ef79ec2a40e2e18103c1d18d79fa075adad463.jpg)

![698aeeb294593291644985ab34e0d06592e5ec5dfc65f376908e55036d228652.jpg](../iclr_results/3087_On%20the%20Linear%20Speedup%20of%20Personalized%20Federated%20Reinforcement%20Learning%20with%20Shared%20Representations/tables/698aeeb294593291644985ab34e0d06592e5ec5dfc65f376908e55036d228652.jpg)

![9938a9e51a516326169d41d45ceb3f5d5d333d2f352f57a8be52763e057981cd.jpg](../iclr_results/3087_On%20the%20Linear%20Speedup%20of%20Personalized%20Federated%20Reinforcement%20Learning%20with%20Shared%20Representations/tables/9938a9e51a516326169d41d45ceb3f5d5d333d2f352f57a8be52763e057981cd.jpg)

## Adaptive teachers for amortized samplers


### Images

![2800f6e27aeaa48aedaa7ec95af4d0ce4b1374c724b62a33550a3543565f47bc.jpg](../iclr_results/3088_Adaptive%20teachers%20for%20amortized%20samplers/images/2800f6e27aeaa48aedaa7ec95af4d0ce4b1374c724b62a33550a3543565f47bc.jpg)

![580a035e36704428dadf27ed96371b0b11799e440a80af666162e00b94d77d4c.jpg](../iclr_results/3088_Adaptive%20teachers%20for%20amortized%20samplers/images/580a035e36704428dadf27ed96371b0b11799e440a80af666162e00b94d77d4c.jpg)

![5d7e18c7c06ee8b606d2ec5a090a1976d4179ed57e885e61ff2f5911a56138c9.jpg](../iclr_results/3088_Adaptive%20teachers%20for%20amortized%20samplers/images/5d7e18c7c06ee8b606d2ec5a090a1976d4179ed57e885e61ff2f5911a56138c9.jpg)

![90a16b2f236fdc8303ca28eba4e6b08fba631cea1614016dcace89af5bf8ae16.jpg](../iclr_results/3088_Adaptive%20teachers%20for%20amortized%20samplers/images/90a16b2f236fdc8303ca28eba4e6b08fba631cea1614016dcace89af5bf8ae16.jpg)

![a37b1015419bec44139e4b712ca940278448564777d4043ca452bfa86b19cc4f.jpg](../iclr_results/3088_Adaptive%20teachers%20for%20amortized%20samplers/images/a37b1015419bec44139e4b712ca940278448564777d4043ca452bfa86b19cc4f.jpg)

![b2d81de6bd6a89dfb96fb194295ce0b57fa3099e2fbd75cdaef63059565fffe5.jpg](../iclr_results/3088_Adaptive%20teachers%20for%20amortized%20samplers/images/b2d81de6bd6a89dfb96fb194295ce0b57fa3099e2fbd75cdaef63059565fffe5.jpg)

![bb5dd97987552074a8b6e13d58a4657ef28ec82ff9549a81cc37af8f813eb856.jpg](../iclr_results/3088_Adaptive%20teachers%20for%20amortized%20samplers/images/bb5dd97987552074a8b6e13d58a4657ef28ec82ff9549a81cc37af8f813eb856.jpg)

![bb8967a82d1083a3ac3e6188a00b9c2ba58190a4fef90c51c8b3500665fca4bf.jpg](../iclr_results/3088_Adaptive%20teachers%20for%20amortized%20samplers/images/bb8967a82d1083a3ac3e6188a00b9c2ba58190a4fef90c51c8b3500665fca4bf.jpg)

![bc4fef6f4e8a4c15923cf6864092ed3c63971ae2d80d415d35f371b6a4d2b0b4.jpg](../iclr_results/3088_Adaptive%20teachers%20for%20amortized%20samplers/images/bc4fef6f4e8a4c15923cf6864092ed3c63971ae2d80d415d35f371b6a4d2b0b4.jpg)

![c28eff59d2b050ac2f90432eac8f6ea8e9ee220da553852b8262a1695683b995.jpg](../iclr_results/3088_Adaptive%20teachers%20for%20amortized%20samplers/images/c28eff59d2b050ac2f90432eac8f6ea8e9ee220da553852b8262a1695683b995.jpg)

![e4847f3b13a382b626027863adaec6a9fe2cd9849b9a7402766e6671daed494f.jpg](../iclr_results/3088_Adaptive%20teachers%20for%20amortized%20samplers/images/e4847f3b13a382b626027863adaec6a9fe2cd9849b9a7402766e6671daed494f.jpg)

![f2038b80b117883bb589f53dac2439b1806291efd2620f57a474aa37ae3816cd.jpg](../iclr_results/3088_Adaptive%20teachers%20for%20amortized%20samplers/images/f2038b80b117883bb589f53dac2439b1806291efd2620f57a474aa37ae3816cd.jpg)

### Tables

![2335f2f137ac53299aae467024de87fb1bbf6ce5ea1f0d9d6a62755903e1308a.jpg](../iclr_results/3088_Adaptive%20teachers%20for%20amortized%20samplers/tables/2335f2f137ac53299aae467024de87fb1bbf6ce5ea1f0d9d6a62755903e1308a.jpg)

![2e3268f3d3cc88eb453d588618c36d94ca8c268ffe05ca3f42de8eee9eb800f7.jpg](../iclr_results/3088_Adaptive%20teachers%20for%20amortized%20samplers/tables/2e3268f3d3cc88eb453d588618c36d94ca8c268ffe05ca3f42de8eee9eb800f7.jpg)

![30cb81dc49a5b3b323833fb8eab2d8d7bcff9bdf5334c5c20bbf0a9282f09206.jpg](../iclr_results/3088_Adaptive%20teachers%20for%20amortized%20samplers/tables/30cb81dc49a5b3b323833fb8eab2d8d7bcff9bdf5334c5c20bbf0a9282f09206.jpg)

![65c9089a243ce83d0af3570b0d16f461df2a1af28a9939b5105aa07156f0b753.jpg](../iclr_results/3088_Adaptive%20teachers%20for%20amortized%20samplers/tables/65c9089a243ce83d0af3570b0d16f461df2a1af28a9939b5105aa07156f0b753.jpg)

![8334e9c40d6e9ce323ee3dbbaf0897f9867516ba8175df6830f99d476ff10e4b.jpg](../iclr_results/3088_Adaptive%20teachers%20for%20amortized%20samplers/tables/8334e9c40d6e9ce323ee3dbbaf0897f9867516ba8175df6830f99d476ff10e4b.jpg)

![87a4ca078d990afc26851a86742ca6e8e5c103fd9f8485d98caa1daf0efe9630.jpg](../iclr_results/3088_Adaptive%20teachers%20for%20amortized%20samplers/tables/87a4ca078d990afc26851a86742ca6e8e5c103fd9f8485d98caa1daf0efe9630.jpg)

![b67c37fdecec2124565c281e2113dc06e74ee4acd83923eb4ce8caeb26834fc6.jpg](../iclr_results/3088_Adaptive%20teachers%20for%20amortized%20samplers/tables/b67c37fdecec2124565c281e2113dc06e74ee4acd83923eb4ce8caeb26834fc6.jpg)

![d1aa986a2e80026a83298c00491747b552f81b4716886c0da905b0e722ac145e.jpg](../iclr_results/3088_Adaptive%20teachers%20for%20amortized%20samplers/tables/d1aa986a2e80026a83298c00491747b552f81b4716886c0da905b0e722ac145e.jpg)

![d8c5eadbca67d6f99b3fe307d410aa40528d5f1c42de1345426b0a05f041d5aa.jpg](../iclr_results/3088_Adaptive%20teachers%20for%20amortized%20samplers/tables/d8c5eadbca67d6f99b3fe307d410aa40528d5f1c42de1345426b0a05f041d5aa.jpg)

![e4341f2e94aba8017a3d6b3b25ad26c39dc4f6e97bf7be6f751e96b514fdea37.jpg](../iclr_results/3088_Adaptive%20teachers%20for%20amortized%20samplers/tables/e4341f2e94aba8017a3d6b3b25ad26c39dc4f6e97bf7be6f751e96b514fdea37.jpg)

![fad0ebc796ec64b61e29021a4dee6f490fd4cb920eaf770a51c14867a2b76d6d.jpg](../iclr_results/3088_Adaptive%20teachers%20for%20amortized%20samplers/tables/fad0ebc796ec64b61e29021a4dee6f490fd4cb920eaf770a51c14867a2b76d6d.jpg)

## Fourier Sliced-Wasserstein Embedding for Multisets and Measures


### Images

![4e4f92bddff7276e5e37ff1d429eee6bc2e2aede9aaad55e7efcfccd7cee3b79.jpg](../iclr_results/3089_Fourier%20Sliced-Wasserstein%20Embedding%20for%20Multisets%20and%20Measures/images/4e4f92bddff7276e5e37ff1d429eee6bc2e2aede9aaad55e7efcfccd7cee3b79.jpg)

![dae23f5eba802510fe81386e5d56550f6f7bcf1699707f32a0f49ab29f5b3315.jpg](../iclr_results/3089_Fourier%20Sliced-Wasserstein%20Embedding%20for%20Multisets%20and%20Measures/images/dae23f5eba802510fe81386e5d56550f6f7bcf1699707f32a0f49ab29f5b3315.jpg)

### Tables

![a9446d04820735a156b0982a9a5e27233b4ac186fb698c6f78153fa75c0664c8.jpg](../iclr_results/3089_Fourier%20Sliced-Wasserstein%20Embedding%20for%20Multisets%20and%20Measures/tables/a9446d04820735a156b0982a9a5e27233b4ac186fb698c6f78153fa75c0664c8.jpg)

![e6559b6ed4f106ec823a41c79dc5258abbca60dc711279b0b042bb4e236d5b02.jpg](../iclr_results/3089_Fourier%20Sliced-Wasserstein%20Embedding%20for%20Multisets%20and%20Measures/tables/e6559b6ed4f106ec823a41c79dc5258abbca60dc711279b0b042bb4e236d5b02.jpg)

![ed7aa22bdc94f5d2d71a8b9baf1ad051bb064c79861302e5f5c1feecccd591fe.jpg](../iclr_results/3089_Fourier%20Sliced-Wasserstein%20Embedding%20for%20Multisets%20and%20Measures/tables/ed7aa22bdc94f5d2d71a8b9baf1ad051bb064c79861302e5f5c1feecccd591fe.jpg)

## Learning Molecular Representation in a Cell


### Images

![1e541ab8cbccda32996eff775af3d3898331efe352d6ec8ca3cbee24004403a7.jpg](../iclr_results/3090_Learning%20Molecular%20Representation%20in%20a%20Cell/images/1e541ab8cbccda32996eff775af3d3898331efe352d6ec8ca3cbee24004403a7.jpg)

![1f2bfa137ea2fb88f2f6668b42aa80fed6491edf1ef1a421582057091a70e60c.jpg](../iclr_results/3090_Learning%20Molecular%20Representation%20in%20a%20Cell/images/1f2bfa137ea2fb88f2f6668b42aa80fed6491edf1ef1a421582057091a70e60c.jpg)

![245b04f611d1fde18fb488059de5695ed88978f4e4dcf4f56434c2121031aca9.jpg](../iclr_results/3090_Learning%20Molecular%20Representation%20in%20a%20Cell/images/245b04f611d1fde18fb488059de5695ed88978f4e4dcf4f56434c2121031aca9.jpg)

![4d15ecd0ed69458b6d716008ff7e1005b40ad1eb91e307446ebbfce74206b9e0.jpg](../iclr_results/3090_Learning%20Molecular%20Representation%20in%20a%20Cell/images/4d15ecd0ed69458b6d716008ff7e1005b40ad1eb91e307446ebbfce74206b9e0.jpg)

![8d7ee25c61b51d20f09c8bd6a92579190a63e2a0ac04ef95641db61bd2f28bfe.jpg](../iclr_results/3090_Learning%20Molecular%20Representation%20in%20a%20Cell/images/8d7ee25c61b51d20f09c8bd6a92579190a63e2a0ac04ef95641db61bd2f28bfe.jpg)

![973b0cf0faae97d056d646d62c47b02579d98bec07dfdd14a954c07aa18b94a1.jpg](../iclr_results/3090_Learning%20Molecular%20Representation%20in%20a%20Cell/images/973b0cf0faae97d056d646d62c47b02579d98bec07dfdd14a954c07aa18b94a1.jpg)

![ae4423d73207de01783feea76ec974d8cccdfc918305d48f84a25d779ab70452.jpg](../iclr_results/3090_Learning%20Molecular%20Representation%20in%20a%20Cell/images/ae4423d73207de01783feea76ec974d8cccdfc918305d48f84a25d779ab70452.jpg)

![b3e25c58c8dd8549bfa4682f2804f3bc2993b1ef1bd1372a19f196305192d013.jpg](../iclr_results/3090_Learning%20Molecular%20Representation%20in%20a%20Cell/images/b3e25c58c8dd8549bfa4682f2804f3bc2993b1ef1bd1372a19f196305192d013.jpg)

![c27a433bfbc252bb67ef9be1fea65af284d22edcf4452cfa7234229efb568ec6.jpg](../iclr_results/3090_Learning%20Molecular%20Representation%20in%20a%20Cell/images/c27a433bfbc252bb67ef9be1fea65af284d22edcf4452cfa7234229efb568ec6.jpg)

![f1fa9639fd718f1f0e473940a29e54203d5c11db18283fb6846db4a30e598d80.jpg](../iclr_results/3090_Learning%20Molecular%20Representation%20in%20a%20Cell/images/f1fa9639fd718f1f0e473940a29e54203d5c11db18283fb6846db4a30e598d80.jpg)

### Tables

![1f188966504ab04dc11d159b512a315d989a4a115b0a5e78529dc004a8fb34ce.jpg](../iclr_results/3090_Learning%20Molecular%20Representation%20in%20a%20Cell/tables/1f188966504ab04dc11d159b512a315d989a4a115b0a5e78529dc004a8fb34ce.jpg)

![4e9f7c8a8a716d293de1f2b71686e7134538d56ddc18de5f17a0e018a13b5e4a.jpg](../iclr_results/3090_Learning%20Molecular%20Representation%20in%20a%20Cell/tables/4e9f7c8a8a716d293de1f2b71686e7134538d56ddc18de5f17a0e018a13b5e4a.jpg)

![85366a7b6cf7b9a85f9a44002f5d9c69228768046a1425f875d6f85549577c6f.jpg](../iclr_results/3090_Learning%20Molecular%20Representation%20in%20a%20Cell/tables/85366a7b6cf7b9a85f9a44002f5d9c69228768046a1425f875d6f85549577c6f.jpg)

![bca1b72f9220a8177859d2dc78dfb1d0815cdd128361df6a7951405b6c66f77c.jpg](../iclr_results/3090_Learning%20Molecular%20Representation%20in%20a%20Cell/tables/bca1b72f9220a8177859d2dc78dfb1d0815cdd128361df6a7951405b6c66f77c.jpg)

![c41055947793ec525419101a2256581014f4fec748274ee566f8db6ff513be3e.jpg](../iclr_results/3090_Learning%20Molecular%20Representation%20in%20a%20Cell/tables/c41055947793ec525419101a2256581014f4fec748274ee566f8db6ff513be3e.jpg)

![f32d1b6db6b51c816b91d83645928ada6464ee775d3725eeb16b6aa43598388a.jpg](../iclr_results/3090_Learning%20Molecular%20Representation%20in%20a%20Cell/tables/f32d1b6db6b51c816b91d83645928ada6464ee775d3725eeb16b6aa43598388a.jpg)

![fce12e7a08dee9b192ed75089e0e8f2ac5763c07d57b9b8aa152a021aac7511a.jpg](../iclr_results/3090_Learning%20Molecular%20Representation%20in%20a%20Cell/tables/fce12e7a08dee9b192ed75089e0e8f2ac5763c07d57b9b8aa152a021aac7511a.jpg)

## T2V-Turbo-v2: Enhancing Video Model Post-Training through Data, Reward, and Conditional Guidance Design


### Images

![2bab90146c37b60b0ed6e15521e75278bcd713a2aceecd45193999f42f891f3b.jpg](../iclr_results/3091_T2V-Turbo-v2_%20Enhancing%20Video%20Model%20Post-Training%20through%20Data%2C%20Reward%2C%20and%20Conditional%20Guidance%20Des/images/2bab90146c37b60b0ed6e15521e75278bcd713a2aceecd45193999f42f891f3b.jpg)

![30af44c7dd2ca93dac08c60daa6cb0d464e1ce79c7e8353ae8ffe507631817d0.jpg](../iclr_results/3091_T2V-Turbo-v2_%20Enhancing%20Video%20Model%20Post-Training%20through%20Data%2C%20Reward%2C%20and%20Conditional%20Guidance%20Des/images/30af44c7dd2ca93dac08c60daa6cb0d464e1ce79c7e8353ae8ffe507631817d0.jpg)

![6bd0e3698712647445413b11d8140dccea29a2e3ae80793b3005d7d2a29496df.jpg](../iclr_results/3091_T2V-Turbo-v2_%20Enhancing%20Video%20Model%20Post-Training%20through%20Data%2C%20Reward%2C%20and%20Conditional%20Guidance%20Des/images/6bd0e3698712647445413b11d8140dccea29a2e3ae80793b3005d7d2a29496df.jpg)

![7af61f89d38c9c5a9e494105d3c3cae7fe1935361bfdc32840ce9310eba25881.jpg](../iclr_results/3091_T2V-Turbo-v2_%20Enhancing%20Video%20Model%20Post-Training%20through%20Data%2C%20Reward%2C%20and%20Conditional%20Guidance%20Des/images/7af61f89d38c9c5a9e494105d3c3cae7fe1935361bfdc32840ce9310eba25881.jpg)

![b8a5c7056c662e88c1218519b98d7ef9c78a601a6facc8430387a460e3665df2.jpg](../iclr_results/3091_T2V-Turbo-v2_%20Enhancing%20Video%20Model%20Post-Training%20through%20Data%2C%20Reward%2C%20and%20Conditional%20Guidance%20Des/images/b8a5c7056c662e88c1218519b98d7ef9c78a601a6facc8430387a460e3665df2.jpg)

![bd4e377b5d1bab2c089676dfa1294c7af042dc72291a27c2c6c36eb78d6467d0.jpg](../iclr_results/3091_T2V-Turbo-v2_%20Enhancing%20Video%20Model%20Post-Training%20through%20Data%2C%20Reward%2C%20and%20Conditional%20Guidance%20Des/images/bd4e377b5d1bab2c089676dfa1294c7af042dc72291a27c2c6c36eb78d6467d0.jpg)

![d3e56b250390a6974a05e91c59737415a5a6212355a5e401234a4fcc4e9d0898.jpg](../iclr_results/3091_T2V-Turbo-v2_%20Enhancing%20Video%20Model%20Post-Training%20through%20Data%2C%20Reward%2C%20and%20Conditional%20Guidance%20Des/images/d3e56b250390a6974a05e91c59737415a5a6212355a5e401234a4fcc4e9d0898.jpg)

![e0df0cfa80938c1a18fc25271bf77f176a1b541295bc10cacb9e23fbf502e9c4.jpg](../iclr_results/3091_T2V-Turbo-v2_%20Enhancing%20Video%20Model%20Post-Training%20through%20Data%2C%20Reward%2C%20and%20Conditional%20Guidance%20Des/images/e0df0cfa80938c1a18fc25271bf77f176a1b541295bc10cacb9e23fbf502e9c4.jpg)

![f6499fff303af7fab0a601d6320218a14baadf1e9028497b7b9848a0e7181944.jpg](../iclr_results/3091_T2V-Turbo-v2_%20Enhancing%20Video%20Model%20Post-Training%20through%20Data%2C%20Reward%2C%20and%20Conditional%20Guidance%20Des/images/f6499fff303af7fab0a601d6320218a14baadf1e9028497b7b9848a0e7181944.jpg)

### Tables

![00ba9919a66b00f939f5ef25e574d4fced3e6a3338a40f6ec01a478c466681ae.jpg](../iclr_results/3091_T2V-Turbo-v2_%20Enhancing%20Video%20Model%20Post-Training%20through%20Data%2C%20Reward%2C%20and%20Conditional%20Guidance%20Des/tables/00ba9919a66b00f939f5ef25e574d4fced3e6a3338a40f6ec01a478c466681ae.jpg)

![108f1acbffd4740c131eeaa0807364cb6648f11320b2f49833140d9a5290d7c3.jpg](../iclr_results/3091_T2V-Turbo-v2_%20Enhancing%20Video%20Model%20Post-Training%20through%20Data%2C%20Reward%2C%20and%20Conditional%20Guidance%20Des/tables/108f1acbffd4740c131eeaa0807364cb6648f11320b2f49833140d9a5290d7c3.jpg)

![382f0df4b296a24d0c995cf4353e1c7cdf248688147443c3bcaaf19d44de100d.jpg](../iclr_results/3091_T2V-Turbo-v2_%20Enhancing%20Video%20Model%20Post-Training%20through%20Data%2C%20Reward%2C%20and%20Conditional%20Guidance%20Des/tables/382f0df4b296a24d0c995cf4353e1c7cdf248688147443c3bcaaf19d44de100d.jpg)

![435886a67e6570bd1d47036cca08c4b47c41a04c4473a4893e32df8cf4416875.jpg](../iclr_results/3091_T2V-Turbo-v2_%20Enhancing%20Video%20Model%20Post-Training%20through%20Data%2C%20Reward%2C%20and%20Conditional%20Guidance%20Des/tables/435886a67e6570bd1d47036cca08c4b47c41a04c4473a4893e32df8cf4416875.jpg)

![4d9edf8f9a209d5137349ffafaad736a4e2246fc9b62d495ab95a9497d6aff5a.jpg](../iclr_results/3091_T2V-Turbo-v2_%20Enhancing%20Video%20Model%20Post-Training%20through%20Data%2C%20Reward%2C%20and%20Conditional%20Guidance%20Des/tables/4d9edf8f9a209d5137349ffafaad736a4e2246fc9b62d495ab95a9497d6aff5a.jpg)

![4ef00243662c24029a79175780ff2e5daf1c371f24d95b43838b3414f0caa735.jpg](../iclr_results/3091_T2V-Turbo-v2_%20Enhancing%20Video%20Model%20Post-Training%20through%20Data%2C%20Reward%2C%20and%20Conditional%20Guidance%20Des/tables/4ef00243662c24029a79175780ff2e5daf1c371f24d95b43838b3414f0caa735.jpg)

![68a6b44a7a72f0598f11252c208903e63267648bba1a6c7cdc3d3590a262a6e9.jpg](../iclr_results/3091_T2V-Turbo-v2_%20Enhancing%20Video%20Model%20Post-Training%20through%20Data%2C%20Reward%2C%20and%20Conditional%20Guidance%20Des/tables/68a6b44a7a72f0598f11252c208903e63267648bba1a6c7cdc3d3590a262a6e9.jpg)

![947bc501484051f0671f2d888b566e985c0c70713e0b63b3add07389f2c6f2d8.jpg](../iclr_results/3091_T2V-Turbo-v2_%20Enhancing%20Video%20Model%20Post-Training%20through%20Data%2C%20Reward%2C%20and%20Conditional%20Guidance%20Des/tables/947bc501484051f0671f2d888b566e985c0c70713e0b63b3add07389f2c6f2d8.jpg)

![ac351404ce21bbb0b096bf19a248d51c26e66148fa416315c0a9513edabd3a70.jpg](../iclr_results/3091_T2V-Turbo-v2_%20Enhancing%20Video%20Model%20Post-Training%20through%20Data%2C%20Reward%2C%20and%20Conditional%20Guidance%20Des/tables/ac351404ce21bbb0b096bf19a248d51c26e66148fa416315c0a9513edabd3a70.jpg)

![d193125798f02345dca5bc0ab7dff8ffead4ca356b1d06d538734287177defcd.jpg](../iclr_results/3091_T2V-Turbo-v2_%20Enhancing%20Video%20Model%20Post-Training%20through%20Data%2C%20Reward%2C%20and%20Conditional%20Guidance%20Des/tables/d193125798f02345dca5bc0ab7dff8ffead4ca356b1d06d538734287177defcd.jpg)

![e04a51d7b4db0e925b41ed5152078a11db54c8159f1c59131e4d4fe8224499f4.jpg](../iclr_results/3091_T2V-Turbo-v2_%20Enhancing%20Video%20Model%20Post-Training%20through%20Data%2C%20Reward%2C%20and%20Conditional%20Guidance%20Des/tables/e04a51d7b4db0e925b41ed5152078a11db54c8159f1c59131e4d4fe8224499f4.jpg)

![ec2a9a1716b6616f3bdfa62cfce203310e53819177b052eb84176a7f8bd4cde9.jpg](../iclr_results/3091_T2V-Turbo-v2_%20Enhancing%20Video%20Model%20Post-Training%20through%20Data%2C%20Reward%2C%20and%20Conditional%20Guidance%20Des/tables/ec2a9a1716b6616f3bdfa62cfce203310e53819177b052eb84176a7f8bd4cde9.jpg)

## ADAM Optimization with Adaptive Batch Selection


### Images

![0338b53342ee7a6d297a532518dfc5c3588596748c60da3523fbfba58d2b8c1f.jpg](../iclr_results/3092_ADAM%20Optimization%20with%20Adaptive%20Batch%20Selection/images/0338b53342ee7a6d297a532518dfc5c3588596748c60da3523fbfba58d2b8c1f.jpg)

![1828cb32b054e346c3e08c0f2a3862828dbc9622ad8083adf1274c2645f97e7c.jpg](../iclr_results/3092_ADAM%20Optimization%20with%20Adaptive%20Batch%20Selection/images/1828cb32b054e346c3e08c0f2a3862828dbc9622ad8083adf1274c2645f97e7c.jpg)

![2ec8a7b519718294e7105d333ca34ad1e4a6191ec87a892e97bafe43ba190c93.jpg](../iclr_results/3092_ADAM%20Optimization%20with%20Adaptive%20Batch%20Selection/images/2ec8a7b519718294e7105d333ca34ad1e4a6191ec87a892e97bafe43ba190c93.jpg)

![3f1c32ead1688960027aa4239c97afceed5818828de8662af9d89a0d9901c4ae.jpg](../iclr_results/3092_ADAM%20Optimization%20with%20Adaptive%20Batch%20Selection/images/3f1c32ead1688960027aa4239c97afceed5818828de8662af9d89a0d9901c4ae.jpg)

![5e7016784f2512d7fd8683c8427bd3feb88cbe02e4391117bbc86b73b4e7d135.jpg](../iclr_results/3092_ADAM%20Optimization%20with%20Adaptive%20Batch%20Selection/images/5e7016784f2512d7fd8683c8427bd3feb88cbe02e4391117bbc86b73b4e7d135.jpg)

![7d34dd59ad3b9ff6d295395aebebc15dee3bbda33dd21503a9a046da8f87d95c.jpg](../iclr_results/3092_ADAM%20Optimization%20with%20Adaptive%20Batch%20Selection/images/7d34dd59ad3b9ff6d295395aebebc15dee3bbda33dd21503a9a046da8f87d95c.jpg)

![88df99c8fa32c0cbe93e65761b0b8b5cf2deb342afe7638ca92808c6fe3768e0.jpg](../iclr_results/3092_ADAM%20Optimization%20with%20Adaptive%20Batch%20Selection/images/88df99c8fa32c0cbe93e65761b0b8b5cf2deb342afe7638ca92808c6fe3768e0.jpg)

![913e27150af5cc5c8d202c54cad8931283a4faafaf538f160ee7ba546d9f62ce.jpg](../iclr_results/3092_ADAM%20Optimization%20with%20Adaptive%20Batch%20Selection/images/913e27150af5cc5c8d202c54cad8931283a4faafaf538f160ee7ba546d9f62ce.jpg)

![c09b0b18f95e4b3ae51667ab721efb4247663070d516e361cde7782b55141adc.jpg](../iclr_results/3092_ADAM%20Optimization%20with%20Adaptive%20Batch%20Selection/images/c09b0b18f95e4b3ae51667ab721efb4247663070d516e361cde7782b55141adc.jpg)

### Tables

![0e63b71eb4af3d6665eb67a39b04627f31a65b26358999bff40850ae3c873e81.jpg](../iclr_results/3092_ADAM%20Optimization%20with%20Adaptive%20Batch%20Selection/tables/0e63b71eb4af3d6665eb67a39b04627f31a65b26358999bff40850ae3c873e81.jpg)

![5843dc5ca864161b0c6042ca3b42eca4f34008c118fe3623db7b9e2784b2865d.jpg](../iclr_results/3092_ADAM%20Optimization%20with%20Adaptive%20Batch%20Selection/tables/5843dc5ca864161b0c6042ca3b42eca4f34008c118fe3623db7b9e2784b2865d.jpg)

![6941ef3fad4fd3cf2bcfbc4fcd791ac20037c784169276b265a2215704aa7e3c.jpg](../iclr_results/3092_ADAM%20Optimization%20with%20Adaptive%20Batch%20Selection/tables/6941ef3fad4fd3cf2bcfbc4fcd791ac20037c784169276b265a2215704aa7e3c.jpg)

![ae935b0a98380cd47e547b018ccd07a0e9c090c0d6dd2cc84a9529f073df13f7.jpg](../iclr_results/3092_ADAM%20Optimization%20with%20Adaptive%20Batch%20Selection/tables/ae935b0a98380cd47e547b018ccd07a0e9c090c0d6dd2cc84a9529f073df13f7.jpg)

![e364de80f69cb77e9f085463a071d3a4b29e7a831a62ed9e72c7f262add19f15.jpg](../iclr_results/3092_ADAM%20Optimization%20with%20Adaptive%20Batch%20Selection/tables/e364de80f69cb77e9f085463a071d3a4b29e7a831a62ed9e72c7f262add19f15.jpg)

![ec10a784adbca8a0d17182fddfef5e9729a23e353431c36d9fe8fc371788f0d0.jpg](../iclr_results/3092_ADAM%20Optimization%20with%20Adaptive%20Batch%20Selection/tables/ec10a784adbca8a0d17182fddfef5e9729a23e353431c36d9fe8fc371788f0d0.jpg)

## Efficient and Trustworthy Causal Discovery with Latent Variables and Complex Relations


### Images

![1b94927657671c1705f51a5675df989228c734b138a8b4a7288293b23a84879d.jpg](../iclr_results/3093_Efficient%20and%20Trustworthy%20Causal%20Discovery%20with%20Latent%20Variables%20and%20Complex%20Relations/images/1b94927657671c1705f51a5675df989228c734b138a8b4a7288293b23a84879d.jpg)

![21373a72a7c775535321bbdeafc7eb16dc219f223bf92f6105168d26841e06e2.jpg](../iclr_results/3093_Efficient%20and%20Trustworthy%20Causal%20Discovery%20with%20Latent%20Variables%20and%20Complex%20Relations/images/21373a72a7c775535321bbdeafc7eb16dc219f223bf92f6105168d26841e06e2.jpg)

![333b563bd41e0bf3a8f460377cf481dbaf70167983d3d6985a3e49f484806585.jpg](../iclr_results/3093_Efficient%20and%20Trustworthy%20Causal%20Discovery%20with%20Latent%20Variables%20and%20Complex%20Relations/images/333b563bd41e0bf3a8f460377cf481dbaf70167983d3d6985a3e49f484806585.jpg)

![37b645add0035881bb9d196367bf0acfcf2a14c439ee1e5dff924d3d64242e5e.jpg](../iclr_results/3093_Efficient%20and%20Trustworthy%20Causal%20Discovery%20with%20Latent%20Variables%20and%20Complex%20Relations/images/37b645add0035881bb9d196367bf0acfcf2a14c439ee1e5dff924d3d64242e5e.jpg)

![4b23b98a9aeaf8c1cf53cba6b0478078943bfd8dbc4c5b7d71bb1b1206985af9.jpg](../iclr_results/3093_Efficient%20and%20Trustworthy%20Causal%20Discovery%20with%20Latent%20Variables%20and%20Complex%20Relations/images/4b23b98a9aeaf8c1cf53cba6b0478078943bfd8dbc4c5b7d71bb1b1206985af9.jpg)

![52d1e40694cd3710a0dfea590d9536c52641ef76578a20a83b5a65df0190671c.jpg](../iclr_results/3093_Efficient%20and%20Trustworthy%20Causal%20Discovery%20with%20Latent%20Variables%20and%20Complex%20Relations/images/52d1e40694cd3710a0dfea590d9536c52641ef76578a20a83b5a65df0190671c.jpg)

![5e6e5430006bd278375464366b5fdf231d2bfaf5477c6d20088a2d22287e6554.jpg](../iclr_results/3093_Efficient%20and%20Trustworthy%20Causal%20Discovery%20with%20Latent%20Variables%20and%20Complex%20Relations/images/5e6e5430006bd278375464366b5fdf231d2bfaf5477c6d20088a2d22287e6554.jpg)

![727568dff2f384a04457b4c9226e75fcd041fd80f0611e821874390176eef2f2.jpg](../iclr_results/3093_Efficient%20and%20Trustworthy%20Causal%20Discovery%20with%20Latent%20Variables%20and%20Complex%20Relations/images/727568dff2f384a04457b4c9226e75fcd041fd80f0611e821874390176eef2f2.jpg)

![7d161dadbf41fdc77826b1aa04bb268c652204f6901851bb372c1ef90d87226c.jpg](../iclr_results/3093_Efficient%20and%20Trustworthy%20Causal%20Discovery%20with%20Latent%20Variables%20and%20Complex%20Relations/images/7d161dadbf41fdc77826b1aa04bb268c652204f6901851bb372c1ef90d87226c.jpg)

![95f9cc6e9ba232883c32a3ad2d7aea465b9df7f02630f2070f2e2be61d0e9624.jpg](../iclr_results/3093_Efficient%20and%20Trustworthy%20Causal%20Discovery%20with%20Latent%20Variables%20and%20Complex%20Relations/images/95f9cc6e9ba232883c32a3ad2d7aea465b9df7f02630f2070f2e2be61d0e9624.jpg)

![b9b54808ebc3c41d1c6d68e28aa4d76515f994307fadcbf40447b529742c0817.jpg](../iclr_results/3093_Efficient%20and%20Trustworthy%20Causal%20Discovery%20with%20Latent%20Variables%20and%20Complex%20Relations/images/b9b54808ebc3c41d1c6d68e28aa4d76515f994307fadcbf40447b529742c0817.jpg)

![bdb23590eaf8e2473d4bb32063260adb79cf591d3a263ffa7aabc539201544fc.jpg](../iclr_results/3093_Efficient%20and%20Trustworthy%20Causal%20Discovery%20with%20Latent%20Variables%20and%20Complex%20Relations/images/bdb23590eaf8e2473d4bb32063260adb79cf591d3a263ffa7aabc539201544fc.jpg)

![bdffab19f50b23f8e23dcde14b85c3a0d98f0c15c9cb62c229c31031bfd488da.jpg](../iclr_results/3093_Efficient%20and%20Trustworthy%20Causal%20Discovery%20with%20Latent%20Variables%20and%20Complex%20Relations/images/bdffab19f50b23f8e23dcde14b85c3a0d98f0c15c9cb62c229c31031bfd488da.jpg)

![e952a90297c1cd0c87dcb423b44a1ef65e6bb80bab8b916137252ab0798b0af0.jpg](../iclr_results/3093_Efficient%20and%20Trustworthy%20Causal%20Discovery%20with%20Latent%20Variables%20and%20Complex%20Relations/images/e952a90297c1cd0c87dcb423b44a1ef65e6bb80bab8b916137252ab0798b0af0.jpg)

![eba8f28a02d5e82be97ea6f2f07452112bb57445364ed6969e9ff87c9e59390d.jpg](../iclr_results/3093_Efficient%20and%20Trustworthy%20Causal%20Discovery%20with%20Latent%20Variables%20and%20Complex%20Relations/images/eba8f28a02d5e82be97ea6f2f07452112bb57445364ed6969e9ff87c9e59390d.jpg)

### Tables

![12f333b4efd6866d0ea4a6af88c6d360cc16700194209734eb7efe3115e5d5b9.jpg](../iclr_results/3093_Efficient%20and%20Trustworthy%20Causal%20Discovery%20with%20Latent%20Variables%20and%20Complex%20Relations/tables/12f333b4efd6866d0ea4a6af88c6d360cc16700194209734eb7efe3115e5d5b9.jpg)

![d00823a9670759c100e2959d1365c263b0fe0e61467b5a88c9e75ca662ddea92.jpg](../iclr_results/3093_Efficient%20and%20Trustworthy%20Causal%20Discovery%20with%20Latent%20Variables%20and%20Complex%20Relations/tables/d00823a9670759c100e2959d1365c263b0fe0e61467b5a88c9e75ca662ddea92.jpg)

## ControlAR: Controllable Image Generation with Autoregressive Models


### Images

![184d6f05d43c9bb74849774896a75bcb905dc40a5cc36fed8288a0bd08f14ba9.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/images/184d6f05d43c9bb74849774896a75bcb905dc40a5cc36fed8288a0bd08f14ba9.jpg)

![1d470c59d2d8e9a0064cfdc2ab3a7563d3fdf82cf6c0df0de4fb60950b2578e2.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/images/1d470c59d2d8e9a0064cfdc2ab3a7563d3fdf82cf6c0df0de4fb60950b2578e2.jpg)

![23e5037725f515e0e3681c9cda4c98bd840001580a065e44bb3beedb2ecc18bd.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/images/23e5037725f515e0e3681c9cda4c98bd840001580a065e44bb3beedb2ecc18bd.jpg)

![36e91279db6291d8796c8e015d5c60aab089c3ca6e486309cf08fe1af94edd1e.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/images/36e91279db6291d8796c8e015d5c60aab089c3ca6e486309cf08fe1af94edd1e.jpg)

![54115d743149c26037115f8fb0356cd6b95a35dfe929c09651dbbb1b712df850.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/images/54115d743149c26037115f8fb0356cd6b95a35dfe929c09651dbbb1b712df850.jpg)

![5c4d8bfeca1a8acac60ef23def538c804b4ab7512be2ebc591893a03715dba19.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/images/5c4d8bfeca1a8acac60ef23def538c804b4ab7512be2ebc591893a03715dba19.jpg)

![81d91958d8de260701a3f0e5d6b91d36861edab73d03383bb3008c8ca086f9e4.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/images/81d91958d8de260701a3f0e5d6b91d36861edab73d03383bb3008c8ca086f9e4.jpg)

![91e412a6aaaac114db1c390767a8630e413c26789c80cf354321dfa3cbdcce80.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/images/91e412a6aaaac114db1c390767a8630e413c26789c80cf354321dfa3cbdcce80.jpg)

![922ea6f675d5107e8a2db0528ed5f3cd57a6d4bb6838062268b1de06405d0ff8.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/images/922ea6f675d5107e8a2db0528ed5f3cd57a6d4bb6838062268b1de06405d0ff8.jpg)

![9886b684a6b3087107546c9e502e8bf18bc7a2f2405eebe130406f4d69707c28.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/images/9886b684a6b3087107546c9e502e8bf18bc7a2f2405eebe130406f4d69707c28.jpg)

![c6e8a9546410e5d1024ad2be968bef0bd47b5b7973f165971fa42e45fb1c345b.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/images/c6e8a9546410e5d1024ad2be968bef0bd47b5b7973f165971fa42e45fb1c345b.jpg)

![ca7456999c86e9c4fa2e9a7188252af6245493a47f54c22513b071f78e8d483d.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/images/ca7456999c86e9c4fa2e9a7188252af6245493a47f54c22513b071f78e8d483d.jpg)

![dc85049c0a47e7956208d39a5f806087e3dfbf944bb23e28e5ce9b7f322b3c1e.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/images/dc85049c0a47e7956208d39a5f806087e3dfbf944bb23e28e5ce9b7f322b3c1e.jpg)

![e15f67c585882fbc160118ec83b638d49c9db19da7bb85e4fd0b0b987001158b.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/images/e15f67c585882fbc160118ec83b638d49c9db19da7bb85e4fd0b0b987001158b.jpg)

![e54124598e085a456c40a1c82fa0eca2ef930962aeba83a44f1da7c094604464.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/images/e54124598e085a456c40a1c82fa0eca2ef930962aeba83a44f1da7c094604464.jpg)

![f1d9dfa7b8feadd0ab0d608deb5030f29b10afabcc59b1f9091d996839a7a433.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/images/f1d9dfa7b8feadd0ab0d608deb5030f29b10afabcc59b1f9091d996839a7a433.jpg)

### Tables

![0e58742124781bff63bdc6ddf5663c7fbfb0ae1f01564cf6cba00222a60636b5.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/tables/0e58742124781bff63bdc6ddf5663c7fbfb0ae1f01564cf6cba00222a60636b5.jpg)

![0e5daf560b680a94c09a0e4d141850fedb43d368e3fe73b117b96ba3d04f1f3f.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/tables/0e5daf560b680a94c09a0e4d141850fedb43d368e3fe73b117b96ba3d04f1f3f.jpg)

![1410f926dd813387f49da203a3d76bc691e6e149a07c188b0674d5011e65b39b.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/tables/1410f926dd813387f49da203a3d76bc691e6e149a07c188b0674d5011e65b39b.jpg)

![274c3627a21f1745169a9573a45b6512c09ada624e8170f8d6c85cf992b363e9.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/tables/274c3627a21f1745169a9573a45b6512c09ada624e8170f8d6c85cf992b363e9.jpg)

![6e8c25d42910812e6b246b42746c976a2e6e4ef9056a843db3881b04bf7fa901.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/tables/6e8c25d42910812e6b246b42746c976a2e6e4ef9056a843db3881b04bf7fa901.jpg)

![9bfb0e37351c25bd4707ecb1e341753b39f7d39f655cf0aaa472ae7c1827ce97.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/tables/9bfb0e37351c25bd4707ecb1e341753b39f7d39f655cf0aaa472ae7c1827ce97.jpg)

![a167efe7ae81b395872bacac57344058954dd1f11d0a2090f928691ccde9fc57.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/tables/a167efe7ae81b395872bacac57344058954dd1f11d0a2090f928691ccde9fc57.jpg)

![bf2aa8c062f9e21c185eb996cfa6c7f0f214e459daca8e89af874af18d33c225.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/tables/bf2aa8c062f9e21c185eb996cfa6c7f0f214e459daca8e89af874af18d33c225.jpg)

![e4e14fc5aea7b3d11fa42109b51c69cb8acebc9f5778f4be53062782b18a8190.jpg](../iclr_results/3094_ControlAR_%20Controllable%20Image%20Generation%20with%20Autoregressive%20Models/tables/e4e14fc5aea7b3d11fa42109b51c69cb8acebc9f5778f4be53062782b18a8190.jpg)

## Number Cookbook: Number Understanding of Language Models and How to Improve It


### Images

![03660f8db553e128bf5a50a072b951218846e4e94d7be45996cbf195ea6a44b1.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/images/03660f8db553e128bf5a50a072b951218846e4e94d7be45996cbf195ea6a44b1.jpg)

![125255bcaf76d777dd3f2f68417fdffcfa1b428cc2d1fd832f6671b64ba5e53d.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/images/125255bcaf76d777dd3f2f68417fdffcfa1b428cc2d1fd832f6671b64ba5e53d.jpg)

![222ec4706fe4ad39db8eefa6f3a1b777fdcc37f6a747ee08e3ce779837218989.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/images/222ec4706fe4ad39db8eefa6f3a1b777fdcc37f6a747ee08e3ce779837218989.jpg)

![28bc360a932baa079d65e0e566bdfbd914af69994789657a5a1c101ef6344c41.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/images/28bc360a932baa079d65e0e566bdfbd914af69994789657a5a1c101ef6344c41.jpg)

![2c695ebcc8faae759cc8e73fe76e762329d602e677ccb89238305fa3b037987a.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/images/2c695ebcc8faae759cc8e73fe76e762329d602e677ccb89238305fa3b037987a.jpg)

![50516b6b44a7d83275584aa3d5bb38b6fdd68e1d30e4219e19a231347f450c54.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/images/50516b6b44a7d83275584aa3d5bb38b6fdd68e1d30e4219e19a231347f450c54.jpg)

![641746875e36c8e8d5b2d58ebea2d7393d8e2501e9b6e9e0f3467cc42201869f.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/images/641746875e36c8e8d5b2d58ebea2d7393d8e2501e9b6e9e0f3467cc42201869f.jpg)

![66651388ca8e17b6f61b79324fa2433deba57ed161a1d1a48c0cb009ae29f8ea.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/images/66651388ca8e17b6f61b79324fa2433deba57ed161a1d1a48c0cb009ae29f8ea.jpg)

![672c9ed07500fd73a5fb4102924b51fab0ed34ca0733277e23f69ca0046e1f77.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/images/672c9ed07500fd73a5fb4102924b51fab0ed34ca0733277e23f69ca0046e1f77.jpg)

![6ab3c0fc0cf06f3338e0cc37134804d880ba2374f43a40c6a43a0a873d791ede.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/images/6ab3c0fc0cf06f3338e0cc37134804d880ba2374f43a40c6a43a0a873d791ede.jpg)

![711b357b45a4457759ca0c5d3c579fdf79805007927879fb7622c10d3e2cd190.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/images/711b357b45a4457759ca0c5d3c579fdf79805007927879fb7622c10d3e2cd190.jpg)

![778013c2664bdcd519081644d73c0412cd53eb6f3fa83a485191bdec890004e5.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/images/778013c2664bdcd519081644d73c0412cd53eb6f3fa83a485191bdec890004e5.jpg)

![a1c0dad60c413dfe720cc7a2cea26e2e9328b7bcb44812d03e690dea5297a796.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/images/a1c0dad60c413dfe720cc7a2cea26e2e9328b7bcb44812d03e690dea5297a796.jpg)

![b8fe3a0658a707a0ca807634e8ebb8d92992ff8b89ac5977080ebd31fa07baf2.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/images/b8fe3a0658a707a0ca807634e8ebb8d92992ff8b89ac5977080ebd31fa07baf2.jpg)

![b9993f234b356b2046dff435aa33d77ef5fa3c23b8faddfe203bb9fe7faafed4.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/images/b9993f234b356b2046dff435aa33d77ef5fa3c23b8faddfe203bb9fe7faafed4.jpg)

![e74b779ea8390dd3b56c2ef0cbac423c1529237d603cf5da0e86159962f29978.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/images/e74b779ea8390dd3b56c2ef0cbac423c1529237d603cf5da0e86159962f29978.jpg)

![f098098254eb88cb0c20d5324cbd2e2eafe6a0a57773f026dd4998e7844c313d.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/images/f098098254eb88cb0c20d5324cbd2e2eafe6a0a57773f026dd4998e7844c313d.jpg)

![f4553db88df9cd538134dd99dc6d6e1d5b4ed7d2100cf44c2e91046ccbcd4253.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/images/f4553db88df9cd538134dd99dc6d6e1d5b4ed7d2100cf44c2e91046ccbcd4253.jpg)

### Tables

![0060be411707b8a21eb6602bba6d144f54ed8f755a8a5d64ea923d20ca2671a7.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/tables/0060be411707b8a21eb6602bba6d144f54ed8f755a8a5d64ea923d20ca2671a7.jpg)

![0d839eabca46ebcf5fef2ff3b5dfd72d2e52501a0b071562ba3ed959599e132c.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/tables/0d839eabca46ebcf5fef2ff3b5dfd72d2e52501a0b071562ba3ed959599e132c.jpg)

![17efaccaa171232794377fa1356ef03f1dd79189b7504a23ce84db44dac3b945.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/tables/17efaccaa171232794377fa1356ef03f1dd79189b7504a23ce84db44dac3b945.jpg)

![1b18f0aa115800d688ee00cae0787d9e5b1db7474bdb785aa46375426a83d550.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/tables/1b18f0aa115800d688ee00cae0787d9e5b1db7474bdb785aa46375426a83d550.jpg)

![216b873b04e314da9fc8436c7a90c88bd6d4a160375f12274f0e02ff158b7dd7.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/tables/216b873b04e314da9fc8436c7a90c88bd6d4a160375f12274f0e02ff158b7dd7.jpg)

![22f7224b01dca2052129b9481ffd48ab99d571aa531969af6789245f5a11f368.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/tables/22f7224b01dca2052129b9481ffd48ab99d571aa531969af6789245f5a11f368.jpg)

![2460ece22466a1b2601ba8e78500825184c6a26e01b0777579718024537e6280.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/tables/2460ece22466a1b2601ba8e78500825184c6a26e01b0777579718024537e6280.jpg)

![2e8b8dcd7dcabcfc61f5326732ab6712a8eaf23e32b2c47f21b8536ea98598d1.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/tables/2e8b8dcd7dcabcfc61f5326732ab6712a8eaf23e32b2c47f21b8536ea98598d1.jpg)

![45b64ecc49c6d061bdbd81425bf280c6b8f8d21cbd14c88eed09a9c4338d83ac.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/tables/45b64ecc49c6d061bdbd81425bf280c6b8f8d21cbd14c88eed09a9c4338d83ac.jpg)

![49ae6b806776e92801bd8bdaa035b7140bd2e1a0e33b3d5baa62df75549b7bb9.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/tables/49ae6b806776e92801bd8bdaa035b7140bd2e1a0e33b3d5baa62df75549b7bb9.jpg)

![610cbd592e9200269df9c92617e211b5d1baa6cffb065e17fa7ca797de50dc9c.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/tables/610cbd592e9200269df9c92617e211b5d1baa6cffb065e17fa7ca797de50dc9c.jpg)

![8f1a966f3c237db3e41fa76aef9404cd58ec736bb59ebe849c18acdc669e4ff3.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/tables/8f1a966f3c237db3e41fa76aef9404cd58ec736bb59ebe849c18acdc669e4ff3.jpg)

![adfdcdcae2eebb613ae53183f3cd7c885bb870ad029ee9a038009ca1bb62e5f1.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/tables/adfdcdcae2eebb613ae53183f3cd7c885bb870ad029ee9a038009ca1bb62e5f1.jpg)

![b7dfe2d3eae43cacbad85bbe0fbe2a40357410911dc05c45f667619958a85b1d.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/tables/b7dfe2d3eae43cacbad85bbe0fbe2a40357410911dc05c45f667619958a85b1d.jpg)

![b7fe34c06bfd53ce5d6bfc2a21bd1ef597eb97dfe3cd46074b4ddfe068fed041.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/tables/b7fe34c06bfd53ce5d6bfc2a21bd1ef597eb97dfe3cd46074b4ddfe068fed041.jpg)

![c824b631724a680df66553411e7aad6c3ae539225bfd10bab1665aea7944a0e8.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/tables/c824b631724a680df66553411e7aad6c3ae539225bfd10bab1665aea7944a0e8.jpg)

![c8f21afe0355515ac8cba2a82811edff9c9d6a7a367bf676e765ff08e2c87e85.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/tables/c8f21afe0355515ac8cba2a82811edff9c9d6a7a367bf676e765ff08e2c87e85.jpg)

![d3df87943a7b22c40277cd6370799a46a4c6324fb5a8e4c90fa5103288ef82c0.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/tables/d3df87943a7b22c40277cd6370799a46a4c6324fb5a8e4c90fa5103288ef82c0.jpg)

![e59431da518cc0bd73eb872448c7c662281ace955a48fc9dabf40df9bc464a96.jpg](../iclr_results/3095_Number%20Cookbook_%20Number%20Understanding%20of%20Language%20Models%20and%20How%20to%20Improve%20It/tables/e59431da518cc0bd73eb872448c7c662281ace955a48fc9dabf40df9bc464a96.jpg)

## Efficient Residual Learning with Mixture-of-Experts for Universal Dexterous Grasping


### Images

![1b5219011ebace6dd8ddc245bcd2b06d8ac2211a1ab85ef7f099e8e2625f2f63.jpg](../iclr_results/3096_Efficient%20Residual%20Learning%20with%20Mixture-of-Experts%20for%20Universal%20Dexterous%20Grasping/images/1b5219011ebace6dd8ddc245bcd2b06d8ac2211a1ab85ef7f099e8e2625f2f63.jpg)

![6b3d046d0a8783a1bb9edf0ede74b031c12d962f8fbd8198c50e3b6899ebf68f.jpg](../iclr_results/3096_Efficient%20Residual%20Learning%20with%20Mixture-of-Experts%20for%20Universal%20Dexterous%20Grasping/images/6b3d046d0a8783a1bb9edf0ede74b031c12d962f8fbd8198c50e3b6899ebf68f.jpg)

![6fe34781f5b53977b416a62f6affa0e0f2331f2338b1c0bd1cdd34de5fd896ac.jpg](../iclr_results/3096_Efficient%20Residual%20Learning%20with%20Mixture-of-Experts%20for%20Universal%20Dexterous%20Grasping/images/6fe34781f5b53977b416a62f6affa0e0f2331f2338b1c0bd1cdd34de5fd896ac.jpg)

![8615abcb960f89726fe85cbacbdce8595e4dc57768f7247e63a13bfce332682b.jpg](../iclr_results/3096_Efficient%20Residual%20Learning%20with%20Mixture-of-Experts%20for%20Universal%20Dexterous%20Grasping/images/8615abcb960f89726fe85cbacbdce8595e4dc57768f7247e63a13bfce332682b.jpg)

![ad684bac99815b894783979a0d4c0deabdd5d977c91ba287c1596530253f3c73.jpg](../iclr_results/3096_Efficient%20Residual%20Learning%20with%20Mixture-of-Experts%20for%20Universal%20Dexterous%20Grasping/images/ad684bac99815b894783979a0d4c0deabdd5d977c91ba287c1596530253f3c73.jpg)

![d1e2ca1912d665d4e2715467310fb7dad0fe4082e8a4350cd463d7e0438c207d.jpg](../iclr_results/3096_Efficient%20Residual%20Learning%20with%20Mixture-of-Experts%20for%20Universal%20Dexterous%20Grasping/images/d1e2ca1912d665d4e2715467310fb7dad0fe4082e8a4350cd463d7e0438c207d.jpg)

### Tables

![03fcbabd3bd5f66a06fc4923f1cf109631e4c6ffb536e2ef37dd925234166056.jpg](../iclr_results/3096_Efficient%20Residual%20Learning%20with%20Mixture-of-Experts%20for%20Universal%20Dexterous%20Grasping/tables/03fcbabd3bd5f66a06fc4923f1cf109631e4c6ffb536e2ef37dd925234166056.jpg)

![22a779d0a0b55767f81f54a971732c42d23985375ba151e60facba9fe2f6c507.jpg](../iclr_results/3096_Efficient%20Residual%20Learning%20with%20Mixture-of-Experts%20for%20Universal%20Dexterous%20Grasping/tables/22a779d0a0b55767f81f54a971732c42d23985375ba151e60facba9fe2f6c507.jpg)

![242ad29e059d86e6d244736084e7889eec5333f89b740740afb42f9edc19d8da.jpg](../iclr_results/3096_Efficient%20Residual%20Learning%20with%20Mixture-of-Experts%20for%20Universal%20Dexterous%20Grasping/tables/242ad29e059d86e6d244736084e7889eec5333f89b740740afb42f9edc19d8da.jpg)

![49558fd782c2f8a237c1c27c5de535e4df3c6361d391cd0d4004836b952267f8.jpg](../iclr_results/3096_Efficient%20Residual%20Learning%20with%20Mixture-of-Experts%20for%20Universal%20Dexterous%20Grasping/tables/49558fd782c2f8a237c1c27c5de535e4df3c6361d391cd0d4004836b952267f8.jpg)

![73cc425a73d456c3c09cd1e9f669dbbc0926782bee1b8d09b98ccc8a6a1327e8.jpg](../iclr_results/3096_Efficient%20Residual%20Learning%20with%20Mixture-of-Experts%20for%20Universal%20Dexterous%20Grasping/tables/73cc425a73d456c3c09cd1e9f669dbbc0926782bee1b8d09b98ccc8a6a1327e8.jpg)

![8e06178df7dd785b6823fcc71b0821abce5c8cc00e1bdbb907ea8596a7cf73b3.jpg](../iclr_results/3096_Efficient%20Residual%20Learning%20with%20Mixture-of-Experts%20for%20Universal%20Dexterous%20Grasping/tables/8e06178df7dd785b6823fcc71b0821abce5c8cc00e1bdbb907ea8596a7cf73b3.jpg)

![a847a741ee8f396b7d8063f31212633956b6068b908ab6b2183bced99dc34d9a.jpg](../iclr_results/3096_Efficient%20Residual%20Learning%20with%20Mixture-of-Experts%20for%20Universal%20Dexterous%20Grasping/tables/a847a741ee8f396b7d8063f31212633956b6068b908ab6b2183bced99dc34d9a.jpg)

![df3ab7724e1aa6615b877ce937f37f57a4b3919f456edbd6a0039dc98b14e443.jpg](../iclr_results/3096_Efficient%20Residual%20Learning%20with%20Mixture-of-Experts%20for%20Universal%20Dexterous%20Grasping/tables/df3ab7724e1aa6615b877ce937f37f57a4b3919f456edbd6a0039dc98b14e443.jpg)

![e8bb456ee9168299ab74675900841fe018e5e0710386d6c9a388264a02858d47.jpg](../iclr_results/3096_Efficient%20Residual%20Learning%20with%20Mixture-of-Experts%20for%20Universal%20Dexterous%20Grasping/tables/e8bb456ee9168299ab74675900841fe018e5e0710386d6c9a388264a02858d47.jpg)

## qNBO: quasi-Newton Meets Bilevel Optimization


### Images

![00482d740c550d624d87b926938eeb9e7da751cd4b57fd2639b6ec1c33fccb2d.jpg](../iclr_results/3097_qNBO_%20quasi-Newton%20Meets%20Bilevel%20Optimization/images/00482d740c550d624d87b926938eeb9e7da751cd4b57fd2639b6ec1c33fccb2d.jpg)

![08313bdb6b42069df36a38a97f09b2258fcb0e60e61e797fcf1f2babece5cf1e.jpg](../iclr_results/3097_qNBO_%20quasi-Newton%20Meets%20Bilevel%20Optimization/images/08313bdb6b42069df36a38a97f09b2258fcb0e60e61e797fcf1f2babece5cf1e.jpg)

![0bd403a1984fbbb61544ab72d111ddfd36fc062868fd13789891f5ff6706ec10.jpg](../iclr_results/3097_qNBO_%20quasi-Newton%20Meets%20Bilevel%20Optimization/images/0bd403a1984fbbb61544ab72d111ddfd36fc062868fd13789891f5ff6706ec10.jpg)

![1f51b93975674959f40037d54f5308cacf3bdd7efbfc7c83b231177762bae078.jpg](../iclr_results/3097_qNBO_%20quasi-Newton%20Meets%20Bilevel%20Optimization/images/1f51b93975674959f40037d54f5308cacf3bdd7efbfc7c83b231177762bae078.jpg)

![29ea64d2c1b1a65b4e1fe7cf2e640c97e6f6fa5a564a972159f42a39ed829100.jpg](../iclr_results/3097_qNBO_%20quasi-Newton%20Meets%20Bilevel%20Optimization/images/29ea64d2c1b1a65b4e1fe7cf2e640c97e6f6fa5a564a972159f42a39ed829100.jpg)

![7179f9402b52d659c20e54556496354597b36e7ef0aa94942a5715e4a166c6cc.jpg](../iclr_results/3097_qNBO_%20quasi-Newton%20Meets%20Bilevel%20Optimization/images/7179f9402b52d659c20e54556496354597b36e7ef0aa94942a5715e4a166c6cc.jpg)

![9895bc510107d1d739d6e6ed65ae2ba6ffcd2c89aee99c0caf4386b36dfc051f.jpg](../iclr_results/3097_qNBO_%20quasi-Newton%20Meets%20Bilevel%20Optimization/images/9895bc510107d1d739d6e6ed65ae2ba6ffcd2c89aee99c0caf4386b36dfc051f.jpg)

![ccfffd733706a57dff16e86455eb72804a424a772a71e9068f8597bdc5ff6022.jpg](../iclr_results/3097_qNBO_%20quasi-Newton%20Meets%20Bilevel%20Optimization/images/ccfffd733706a57dff16e86455eb72804a424a772a71e9068f8597bdc5ff6022.jpg)

![d6ee63e3e0c5021bb1c2d94be813c55aad94927f187cbf28edbae7ae1ae3088b.jpg](../iclr_results/3097_qNBO_%20quasi-Newton%20Meets%20Bilevel%20Optimization/images/d6ee63e3e0c5021bb1c2d94be813c55aad94927f187cbf28edbae7ae1ae3088b.jpg)

![f5c23430f3bcea9cd109c7b69aa03a55cc2e4160a8dfa619dded5c152189c37c.jpg](../iclr_results/3097_qNBO_%20quasi-Newton%20Meets%20Bilevel%20Optimization/images/f5c23430f3bcea9cd109c7b69aa03a55cc2e4160a8dfa619dded5c152189c37c.jpg)

### Tables

![1d58ac0798280d613965b14032a86922affc9e76e7028a76da7ec4aef62d9ecd.jpg](../iclr_results/3097_qNBO_%20quasi-Newton%20Meets%20Bilevel%20Optimization/tables/1d58ac0798280d613965b14032a86922affc9e76e7028a76da7ec4aef62d9ecd.jpg)

![51ecdcbe3381d65741332b51091c131a27337c99aaae44d34da88c35639f98a9.jpg](../iclr_results/3097_qNBO_%20quasi-Newton%20Meets%20Bilevel%20Optimization/tables/51ecdcbe3381d65741332b51091c131a27337c99aaae44d34da88c35639f98a9.jpg)

![6505d043da88644fe74ae3ca8b71bb0a3c3a94b4ffc0b3b89149a81330d704c3.jpg](../iclr_results/3097_qNBO_%20quasi-Newton%20Meets%20Bilevel%20Optimization/tables/6505d043da88644fe74ae3ca8b71bb0a3c3a94b4ffc0b3b89149a81330d704c3.jpg)

## MatryoshkaKV: Adaptive KV Compression via Trainable Orthogonal Projection


### Images

![0cc85405725108fc0ccd141bd16685fe8d0fdb2b4445739f49a8726eae597307.jpg](../iclr_results/3098_MatryoshkaKV_%20Adaptive%20KV%20Compression%20via%20Trainable%20Orthogonal%20Projection/images/0cc85405725108fc0ccd141bd16685fe8d0fdb2b4445739f49a8726eae597307.jpg)

![4e2e90174dc8292ee82ecfd63142ccd2128e97c26080ef698c962a375f81222d.jpg](../iclr_results/3098_MatryoshkaKV_%20Adaptive%20KV%20Compression%20via%20Trainable%20Orthogonal%20Projection/images/4e2e90174dc8292ee82ecfd63142ccd2128e97c26080ef698c962a375f81222d.jpg)

![72de8198b47d667ec57c6e10e99dba73af3defa1b1e45861e87acb4ff6e50668.jpg](../iclr_results/3098_MatryoshkaKV_%20Adaptive%20KV%20Compression%20via%20Trainable%20Orthogonal%20Projection/images/72de8198b47d667ec57c6e10e99dba73af3defa1b1e45861e87acb4ff6e50668.jpg)

![a8036f167634b9b116beaa1fa4f383cd781592e0a91fc8a498b6bf0f4bb24e18.jpg](../iclr_results/3098_MatryoshkaKV_%20Adaptive%20KV%20Compression%20via%20Trainable%20Orthogonal%20Projection/images/a8036f167634b9b116beaa1fa4f383cd781592e0a91fc8a498b6bf0f4bb24e18.jpg)

![b52e84ed998b2f1b01ad5e5b55a07608510e65c5774ec56c126a56065d4d474d.jpg](../iclr_results/3098_MatryoshkaKV_%20Adaptive%20KV%20Compression%20via%20Trainable%20Orthogonal%20Projection/images/b52e84ed998b2f1b01ad5e5b55a07608510e65c5774ec56c126a56065d4d474d.jpg)

![d19dfdedf64ded3a9b349c7c269f3fa977f910992cb2747a6d09377661f2b4be.jpg](../iclr_results/3098_MatryoshkaKV_%20Adaptive%20KV%20Compression%20via%20Trainable%20Orthogonal%20Projection/images/d19dfdedf64ded3a9b349c7c269f3fa977f910992cb2747a6d09377661f2b4be.jpg)

![e7d04c19905d74ef5e766957467dcaf03723ca1bca2ffe369fb296eed259b89c.jpg](../iclr_results/3098_MatryoshkaKV_%20Adaptive%20KV%20Compression%20via%20Trainable%20Orthogonal%20Projection/images/e7d04c19905d74ef5e766957467dcaf03723ca1bca2ffe369fb296eed259b89c.jpg)

![f9d29e4bb927c215a6127f51e3ecab991aacd7afa5c9fd05fdc34975a599a1bb.jpg](../iclr_results/3098_MatryoshkaKV_%20Adaptive%20KV%20Compression%20via%20Trainable%20Orthogonal%20Projection/images/f9d29e4bb927c215a6127f51e3ecab991aacd7afa5c9fd05fdc34975a599a1bb.jpg)

![fcf8492416305dbecd677123f7cacb0a57d49852c6309c97b9c615abdac22f4f.jpg](../iclr_results/3098_MatryoshkaKV_%20Adaptive%20KV%20Compression%20via%20Trainable%20Orthogonal%20Projection/images/fcf8492416305dbecd677123f7cacb0a57d49852c6309c97b9c615abdac22f4f.jpg)

### Tables

![2b4eb0a10ba525a512aa6668ffa3f7a26877ef545a093b87fd933fd42cd6f632.jpg](../iclr_results/3098_MatryoshkaKV_%20Adaptive%20KV%20Compression%20via%20Trainable%20Orthogonal%20Projection/tables/2b4eb0a10ba525a512aa6668ffa3f7a26877ef545a093b87fd933fd42cd6f632.jpg)

![68f2b2faac5b65f440e40edbdb9a137810fa77d3f1761fc7003b288ad26bfb14.jpg](../iclr_results/3098_MatryoshkaKV_%20Adaptive%20KV%20Compression%20via%20Trainable%20Orthogonal%20Projection/tables/68f2b2faac5b65f440e40edbdb9a137810fa77d3f1761fc7003b288ad26bfb14.jpg)

![6ef669b5cedd549bcfb6c8cb48278659d9e379c269ae41b04fb44157f2df0099.jpg](../iclr_results/3098_MatryoshkaKV_%20Adaptive%20KV%20Compression%20via%20Trainable%20Orthogonal%20Projection/tables/6ef669b5cedd549bcfb6c8cb48278659d9e379c269ae41b04fb44157f2df0099.jpg)

![9755a97be37815bfac350d770734e6365e7578a9e1a20b13e4cd0c84a5f45c8a.jpg](../iclr_results/3098_MatryoshkaKV_%20Adaptive%20KV%20Compression%20via%20Trainable%20Orthogonal%20Projection/tables/9755a97be37815bfac350d770734e6365e7578a9e1a20b13e4cd0c84a5f45c8a.jpg)

![a4905462917e569d8d28cf9988ee53ca055cbe09b3fc3150cd10bdfb39e7cb26.jpg](../iclr_results/3098_MatryoshkaKV_%20Adaptive%20KV%20Compression%20via%20Trainable%20Orthogonal%20Projection/tables/a4905462917e569d8d28cf9988ee53ca055cbe09b3fc3150cd10bdfb39e7cb26.jpg)

![ab70d08835c9be4bfeef16036cd7ed82ddd2a79e28b16b07a3c5b99e86b22e08.jpg](../iclr_results/3098_MatryoshkaKV_%20Adaptive%20KV%20Compression%20via%20Trainable%20Orthogonal%20Projection/tables/ab70d08835c9be4bfeef16036cd7ed82ddd2a79e28b16b07a3c5b99e86b22e08.jpg)

![c127a58baf1a8a78e77545796f212f2022b6ecd9170c1f6066a4a0606ab99d28.jpg](../iclr_results/3098_MatryoshkaKV_%20Adaptive%20KV%20Compression%20via%20Trainable%20Orthogonal%20Projection/tables/c127a58baf1a8a78e77545796f212f2022b6ecd9170c1f6066a4a0606ab99d28.jpg)

![c8b667ff96127c3fab6ddd929eb394c05ab1d230b9ea4cdf733ccb8fa255edbf.jpg](../iclr_results/3098_MatryoshkaKV_%20Adaptive%20KV%20Compression%20via%20Trainable%20Orthogonal%20Projection/tables/c8b667ff96127c3fab6ddd929eb394c05ab1d230b9ea4cdf733ccb8fa255edbf.jpg)

![f778aec653b4d84d560c9e2ff685ebce092ec293753f94fba02b185697991fb5.jpg](../iclr_results/3098_MatryoshkaKV_%20Adaptive%20KV%20Compression%20via%20Trainable%20Orthogonal%20Projection/tables/f778aec653b4d84d560c9e2ff685ebce092ec293753f94fba02b185697991fb5.jpg)

## Grounding by Trying: LLMs with Reinforcement Learning-Enhanced Retrieval


### Images

![26c2395e3accb3eb8e39f929d9fc7416f16d7a18223d5785cc1a5a7295df088b.jpg](../iclr_results/3099_Grounding%20by%20Trying_%20LLMs%20with%20Reinforcement%20Learning-Enhanced%20Retrieval/images/26c2395e3accb3eb8e39f929d9fc7416f16d7a18223d5785cc1a5a7295df088b.jpg)

![273d544600961313d77d4292e2b97315f05422239a79e7b6ecfa1e75f71f9119.jpg](../iclr_results/3099_Grounding%20by%20Trying_%20LLMs%20with%20Reinforcement%20Learning-Enhanced%20Retrieval/images/273d544600961313d77d4292e2b97315f05422239a79e7b6ecfa1e75f71f9119.jpg)

![29bfe01e5b454a9d04c25f5db92f08ef73d18109eb5ba68f138251186312b1a2.jpg](../iclr_results/3099_Grounding%20by%20Trying_%20LLMs%20with%20Reinforcement%20Learning-Enhanced%20Retrieval/images/29bfe01e5b454a9d04c25f5db92f08ef73d18109eb5ba68f138251186312b1a2.jpg)

![f6401eac696bf970e9d03caef6c88ac38e21747df7eba5d18dd4c0bcd3e2aff7.jpg](../iclr_results/3099_Grounding%20by%20Trying_%20LLMs%20with%20Reinforcement%20Learning-Enhanced%20Retrieval/images/f6401eac696bf970e9d03caef6c88ac38e21747df7eba5d18dd4c0bcd3e2aff7.jpg)

### Tables

![2ed24680436970038db2cf641ee969978df4c6bd5ab3e26d20a0240ced5815af.jpg](../iclr_results/3099_Grounding%20by%20Trying_%20LLMs%20with%20Reinforcement%20Learning-Enhanced%20Retrieval/tables/2ed24680436970038db2cf641ee969978df4c6bd5ab3e26d20a0240ced5815af.jpg)

![307f433461e058fc7ded61ca4458bea493f46c32907edede31ae4e2d88692265.jpg](../iclr_results/3099_Grounding%20by%20Trying_%20LLMs%20with%20Reinforcement%20Learning-Enhanced%20Retrieval/tables/307f433461e058fc7ded61ca4458bea493f46c32907edede31ae4e2d88692265.jpg)

![4a5300cccd4d5b5abce2936cea70e42b3f5245eec60def355053f57b37cf9085.jpg](../iclr_results/3099_Grounding%20by%20Trying_%20LLMs%20with%20Reinforcement%20Learning-Enhanced%20Retrieval/tables/4a5300cccd4d5b5abce2936cea70e42b3f5245eec60def355053f57b37cf9085.jpg)

![818488a4ce863ec4048380e6969a83e75928b5c655b8e7fdbd923c3bbd6a7bd9.jpg](../iclr_results/3099_Grounding%20by%20Trying_%20LLMs%20with%20Reinforcement%20Learning-Enhanced%20Retrieval/tables/818488a4ce863ec4048380e6969a83e75928b5c655b8e7fdbd923c3bbd6a7bd9.jpg)

![8964b7b3e359c67b262c7d8b97c2b5b123940ba3b5ca23a794c7a36c0d4896b6.jpg](../iclr_results/3099_Grounding%20by%20Trying_%20LLMs%20with%20Reinforcement%20Learning-Enhanced%20Retrieval/tables/8964b7b3e359c67b262c7d8b97c2b5b123940ba3b5ca23a794c7a36c0d4896b6.jpg)

![9a42e6f0c19e5421c1e23af0752a289b77d6e5e011d07e8223ede2406f63b14c.jpg](../iclr_results/3099_Grounding%20by%20Trying_%20LLMs%20with%20Reinforcement%20Learning-Enhanced%20Retrieval/tables/9a42e6f0c19e5421c1e23af0752a289b77d6e5e011d07e8223ede2406f63b14c.jpg)

![a5ac6a3ee6e7cc60e6536571578c0c36de121a678c6399d90aa8f25a3e7162c9.jpg](../iclr_results/3099_Grounding%20by%20Trying_%20LLMs%20with%20Reinforcement%20Learning-Enhanced%20Retrieval/tables/a5ac6a3ee6e7cc60e6536571578c0c36de121a678c6399d90aa8f25a3e7162c9.jpg)

![a7006079f30a31423acdb747768bce8ae87f12fc9446121291524c0fe61993eb.jpg](../iclr_results/3099_Grounding%20by%20Trying_%20LLMs%20with%20Reinforcement%20Learning-Enhanced%20Retrieval/tables/a7006079f30a31423acdb747768bce8ae87f12fc9446121291524c0fe61993eb.jpg)

![c4fb30668162e2b6590bebef570362026d57d31b76ea2fa47213ca9785755e33.jpg](../iclr_results/3099_Grounding%20by%20Trying_%20LLMs%20with%20Reinforcement%20Learning-Enhanced%20Retrieval/tables/c4fb30668162e2b6590bebef570362026d57d31b76ea2fa47213ca9785755e33.jpg)

![fa487d62ad3dbb40391ff52a30fcd0d79fdf81280b6c7571448db6724454611a.jpg](../iclr_results/3099_Grounding%20by%20Trying_%20LLMs%20with%20Reinforcement%20Learning-Enhanced%20Retrieval/tables/fa487d62ad3dbb40391ff52a30fcd0d79fdf81280b6c7571448db6724454611a.jpg)

## Towards Bridging Generalization and Expressivity of Graph Neural Networks


### Images

![1bfd2577b51f2756fcbaf4c2e2fa12a46a8c2d50a7aa15e67dd0c8a0fceb773d.jpg](../iclr_results/3100_Towards%20Bridging%20Generalization%20and%20Expressivity%20of%20Graph%20Neural%20Networks/images/1bfd2577b51f2756fcbaf4c2e2fa12a46a8c2d50a7aa15e67dd0c8a0fceb773d.jpg)

![4664d711b86a0424624a4cf4fc811fd00fdefc79b21c54792e86ee132408919a.jpg](../iclr_results/3100_Towards%20Bridging%20Generalization%20and%20Expressivity%20of%20Graph%20Neural%20Networks/images/4664d711b86a0424624a4cf4fc811fd00fdefc79b21c54792e86ee132408919a.jpg)

![7d99a9302d50088f4f7182064761aa9cc846a461c95ae7f157873fd47da32c72.jpg](../iclr_results/3100_Towards%20Bridging%20Generalization%20and%20Expressivity%20of%20Graph%20Neural%20Networks/images/7d99a9302d50088f4f7182064761aa9cc846a461c95ae7f157873fd47da32c72.jpg)

![f1438f70d51dd39a9d8d46dcb7a6477d9957650dbd8c7d59fc55df6ea41b44aa.jpg](../iclr_results/3100_Towards%20Bridging%20Generalization%20and%20Expressivity%20of%20Graph%20Neural%20Networks/images/f1438f70d51dd39a9d8d46dcb7a6477d9957650dbd8c7d59fc55df6ea41b44aa.jpg)

![f9fb25f9f7a605b655ed22b12daded1820578fababa5ec206de0c82e21760dd4.jpg](../iclr_results/3100_Towards%20Bridging%20Generalization%20and%20Expressivity%20of%20Graph%20Neural%20Networks/images/f9fb25f9f7a605b655ed22b12daded1820578fababa5ec206de0c82e21760dd4.jpg)

### Tables

![678eb51804c54352dd23540932fcdd9f7dbf7d2b4ba48e42013902d3662dc37c.jpg](../iclr_results/3100_Towards%20Bridging%20Generalization%20and%20Expressivity%20of%20Graph%20Neural%20Networks/tables/678eb51804c54352dd23540932fcdd9f7dbf7d2b4ba48e42013902d3662dc37c.jpg)

![7a8e022165aff6230b950fb76ad8addbadac4270b59cca1bb6449e23609322fc.jpg](../iclr_results/3100_Towards%20Bridging%20Generalization%20and%20Expressivity%20of%20Graph%20Neural%20Networks/tables/7a8e022165aff6230b950fb76ad8addbadac4270b59cca1bb6449e23609322fc.jpg)

![a48e1362da2f032978a8a07c53b57602d4ce2844088cf3b0954b27e3e50a12ea.jpg](../iclr_results/3100_Towards%20Bridging%20Generalization%20and%20Expressivity%20of%20Graph%20Neural%20Networks/tables/a48e1362da2f032978a8a07c53b57602d4ce2844088cf3b0954b27e3e50a12ea.jpg)

## LucidPPN: Unambiguous Prototypical Parts Network for User-centric Interpretable Computer Vision


### Images

![005e5b78a09011fbe60f23b5f69802b487a394af8a4a56f42fa12de66b3d30bc.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/005e5b78a09011fbe60f23b5f69802b487a394af8a4a56f42fa12de66b3d30bc.jpg)

![02ea25f1d8e32605951c3260a5cc356523513df604ff3fb8a71d5dbb58547fcc.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/02ea25f1d8e32605951c3260a5cc356523513df604ff3fb8a71d5dbb58547fcc.jpg)

![0386735971c7e55959e84f741eb7842e1eb00e6d53ae0d067221f226c4020b8c.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/0386735971c7e55959e84f741eb7842e1eb00e6d53ae0d067221f226c4020b8c.jpg)

![0d4635272b437f2845d880fa5cccc63dcb7a7cfae432757f656c17b18cc61e94.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/0d4635272b437f2845d880fa5cccc63dcb7a7cfae432757f656c17b18cc61e94.jpg)

![10af3c2f330e7869450a6a06397b216a370c18fab7cbf1a689cbff9ed75b0b16.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/10af3c2f330e7869450a6a06397b216a370c18fab7cbf1a689cbff9ed75b0b16.jpg)

![129261e63e72e57e4de08cfdc360f6121745fb738378076c8ac4f0db2ccc9b48.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/129261e63e72e57e4de08cfdc360f6121745fb738378076c8ac4f0db2ccc9b48.jpg)

![14b6374b56c1063256afac41bb68615da46b9ec36fc970cc0466f9d146f11f41.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/14b6374b56c1063256afac41bb68615da46b9ec36fc970cc0466f9d146f11f41.jpg)

![17ec5af1d314a814119de3437fd461c928671131fbfc0732780210fce4c55051.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/17ec5af1d314a814119de3437fd461c928671131fbfc0732780210fce4c55051.jpg)

![189157aaf6fadeb2234a383ada403414d3d8472e6a7c6546fa179fb2b843155f.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/189157aaf6fadeb2234a383ada403414d3d8472e6a7c6546fa179fb2b843155f.jpg)

![1b78bc04e5e339a0d0ef12dfe0a69cbb533d7f337f89c631e9fdfc5bc2f91bd4.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/1b78bc04e5e339a0d0ef12dfe0a69cbb533d7f337f89c631e9fdfc5bc2f91bd4.jpg)

![1c2dc03aff4ca11bd60f571b91179cb848e35f920955d99ebc5fbdeecb5172c1.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/1c2dc03aff4ca11bd60f571b91179cb848e35f920955d99ebc5fbdeecb5172c1.jpg)

![1d9e4ef786f554ecbbd091a84969e00a801187c624de24f8f253398d02653f76.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/1d9e4ef786f554ecbbd091a84969e00a801187c624de24f8f253398d02653f76.jpg)

![2113534f6a772d7fa4268e988e74940b3afca44856b6a3ed3370ea9a47dbc317.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/2113534f6a772d7fa4268e988e74940b3afca44856b6a3ed3370ea9a47dbc317.jpg)

![239f3504c71e489213d39cab1651a3c57bdd638336c9796fef8d795d9ca65192.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/239f3504c71e489213d39cab1651a3c57bdd638336c9796fef8d795d9ca65192.jpg)

![2749950464afa2806ea09e7d9df9f71e1d1003ef13318c7ddc58f1594fae3f17.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/2749950464afa2806ea09e7d9df9f71e1d1003ef13318c7ddc58f1594fae3f17.jpg)

![281870cbfcf83cb428c3fa55a2465410b3c7adc2e0a6361514180efbc566cdcc.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/281870cbfcf83cb428c3fa55a2465410b3c7adc2e0a6361514180efbc566cdcc.jpg)

![28288d2b1b0637b233b5bc2e1f8bdd316d9e9f9ccdb7e633d4057e8761c06de1.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/28288d2b1b0637b233b5bc2e1f8bdd316d9e9f9ccdb7e633d4057e8761c06de1.jpg)

![29db667462591c617832728486eb5e8dea406e0fa7bafeb05cb634e0297693c1.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/29db667462591c617832728486eb5e8dea406e0fa7bafeb05cb634e0297693c1.jpg)

![2d125a874c6df305cdb27fbd33793251a8dfc7cc9630ac201c5bcf4c7055bc81.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/2d125a874c6df305cdb27fbd33793251a8dfc7cc9630ac201c5bcf4c7055bc81.jpg)

![2ece03b752d303019358426e88ff53bc0052d9d8071f098262cc6295ee56f7b6.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/2ece03b752d303019358426e88ff53bc0052d9d8071f098262cc6295ee56f7b6.jpg)

![2ed1ce86b825f1eb0e0efaeb414d349517f876621f83ccf3e6aa9383b4cb79a4.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/2ed1ce86b825f1eb0e0efaeb414d349517f876621f83ccf3e6aa9383b4cb79a4.jpg)

![354093122a5cb755117bb20afc8563ea7f1140402532fb3d85102d6b76a206bf.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/354093122a5cb755117bb20afc8563ea7f1140402532fb3d85102d6b76a206bf.jpg)

![354eb4aecf76c06f8f4bebe550fffa45c109fd1aa1875df779849636cb31a771.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/354eb4aecf76c06f8f4bebe550fffa45c109fd1aa1875df779849636cb31a771.jpg)

![35849e38898e2893f23635581ea11bee41fa50a8840bd12b8395fada011e2d7d.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/35849e38898e2893f23635581ea11bee41fa50a8840bd12b8395fada011e2d7d.jpg)

![382da4c899b265587dd0a5232c83bbcbf0e86a11ecf51f8071e1cba0d4316277.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/382da4c899b265587dd0a5232c83bbcbf0e86a11ecf51f8071e1cba0d4316277.jpg)

![38e52e41ad12dc04deb58fc9e3035aee107ca0ce7f0d4fdf8df9b7c2b53947ee.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/38e52e41ad12dc04deb58fc9e3035aee107ca0ce7f0d4fdf8df9b7c2b53947ee.jpg)

![399782d0ecb2aa0afc7c0323e3651067658ccf26f89451d35480c806fa630114.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/399782d0ecb2aa0afc7c0323e3651067658ccf26f89451d35480c806fa630114.jpg)

![3bb72b6e98a6b51d274f53cfcfd3f80ad348775890f65f8d28d9b6b15ba9f667.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/3bb72b6e98a6b51d274f53cfcfd3f80ad348775890f65f8d28d9b6b15ba9f667.jpg)

![3c9b83056da3387070a80f90c20e37325075ca236024f7b038f47131e3f96107.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/3c9b83056da3387070a80f90c20e37325075ca236024f7b038f47131e3f96107.jpg)

![3ce4811a5469fb0bff5304cb341415cfcd81122980d7482ddb5da8d22fe91250.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/3ce4811a5469fb0bff5304cb341415cfcd81122980d7482ddb5da8d22fe91250.jpg)

![465667dcce68cff8e4f5f4c5220baa6e0f06b73a9aba6f50b4b6c12bfa951142.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/465667dcce68cff8e4f5f4c5220baa6e0f06b73a9aba6f50b4b6c12bfa951142.jpg)

![49b39c6201cc0d560dff4193f413e304203b8aca69c3e4c524696885d2c87ace.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/49b39c6201cc0d560dff4193f413e304203b8aca69c3e4c524696885d2c87ace.jpg)

![4a96a1fa5dbfa0761132182a760f588f5ace6046781967f50b7be30be42a31d2.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/4a96a1fa5dbfa0761132182a760f588f5ace6046781967f50b7be30be42a31d2.jpg)

![4d23daf9ed294d528c5fb6a64a89f6d6ac69dfadbb88d9647f4da0864efce850.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/4d23daf9ed294d528c5fb6a64a89f6d6ac69dfadbb88d9647f4da0864efce850.jpg)

![4f9333066a33a773c0067355206ead6cf76daca3ab9c443fbb11c5674a54d192.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/4f9333066a33a773c0067355206ead6cf76daca3ab9c443fbb11c5674a54d192.jpg)

![531c25a178ce42f69b0f6609656bef8acd1388191d76c01aae0fc0a51982c62e.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/531c25a178ce42f69b0f6609656bef8acd1388191d76c01aae0fc0a51982c62e.jpg)

![5730d0586339a19405204ac80b7f9cd3c921835589fe1c554fc34ea8f55c36af.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/5730d0586339a19405204ac80b7f9cd3c921835589fe1c554fc34ea8f55c36af.jpg)

![58b8e1acf740ad8180c180a082769c0907db114c0f419b1afe7ac18d10be90d4.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/58b8e1acf740ad8180c180a082769c0907db114c0f419b1afe7ac18d10be90d4.jpg)

![5930e32f4fb6f602a20a7f27125b33e1059e974feb566bc6960bcb103db860ed.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/5930e32f4fb6f602a20a7f27125b33e1059e974feb566bc6960bcb103db860ed.jpg)

![5e547c1ebf137c2883b316776b299b985bab262f2c5bce3da6ac0aa3aa97dfba.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/5e547c1ebf137c2883b316776b299b985bab262f2c5bce3da6ac0aa3aa97dfba.jpg)

![62889c9a5a8ed468a74713ed8523c471f18abbc03be253b254921c5d2e469d37.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/62889c9a5a8ed468a74713ed8523c471f18abbc03be253b254921c5d2e469d37.jpg)

![66a571729a3b1817c321f32becc8304bd2d64cfc95074af866121ae99b9b51ac.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/66a571729a3b1817c321f32becc8304bd2d64cfc95074af866121ae99b9b51ac.jpg)

![69aac18e544b466c15c3c83996522a56d6edcbac0d59178595365bfd422cf053.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/69aac18e544b466c15c3c83996522a56d6edcbac0d59178595365bfd422cf053.jpg)

![6f1587fe43763997884795dc778a9de8e621a33091f9d3531560e3aeff20d6f8.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/6f1587fe43763997884795dc778a9de8e621a33091f9d3531560e3aeff20d6f8.jpg)

![748a26abb5321538f037c0015e3c8921e5d83a3787183f9f406181b485cad36f.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/748a26abb5321538f037c0015e3c8921e5d83a3787183f9f406181b485cad36f.jpg)

![7ab7b90d1233da3ce848ceebfeaf3978a4958cd646250934930078925b8acd7c.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/7ab7b90d1233da3ce848ceebfeaf3978a4958cd646250934930078925b8acd7c.jpg)

![7d81d59afc8554616e2b2bb5641c648920ed7ea6499e7bb0411c8303410ca4ee.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/7d81d59afc8554616e2b2bb5641c648920ed7ea6499e7bb0411c8303410ca4ee.jpg)

![87c9cfb6072ad7592043c147feb68b5cc48242a4916d8d7c25b22b162c83b768.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/87c9cfb6072ad7592043c147feb68b5cc48242a4916d8d7c25b22b162c83b768.jpg)

![88ecaab4f2443bbc4775a2c763dc4c2a5574fded85d0a0c1460624a4f5a7c728.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/88ecaab4f2443bbc4775a2c763dc4c2a5574fded85d0a0c1460624a4f5a7c728.jpg)

![8baa770bc94a20fbb0b598f14e03723ac128c910933e251256c6e9bca3aead52.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/8baa770bc94a20fbb0b598f14e03723ac128c910933e251256c6e9bca3aead52.jpg)

![8d8c0345074ccd66bf1320caa5d98352f9d9a22c1507601183deb9edbe6867b7.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/8d8c0345074ccd66bf1320caa5d98352f9d9a22c1507601183deb9edbe6867b7.jpg)

![9749539a6c0fa7add0320a344918f50a5c8807fc319d54448e2a9e829dd7cc4b.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/9749539a6c0fa7add0320a344918f50a5c8807fc319d54448e2a9e829dd7cc4b.jpg)

![98a4daaced46ec228a198f0118fef920ef5f4b73e15533e5ae78faed79bf074f.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/98a4daaced46ec228a198f0118fef920ef5f4b73e15533e5ae78faed79bf074f.jpg)

![a17b38175e857f3c05dcc2df9961f1701894a798b704f58a8772f9afc249fc24.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/a17b38175e857f3c05dcc2df9961f1701894a798b704f58a8772f9afc249fc24.jpg)

![a63606603130596f252cd68975210460665e0d56dd0e93e2867bfb1e1456cd07.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/a63606603130596f252cd68975210460665e0d56dd0e93e2867bfb1e1456cd07.jpg)

![b3e472c29bf7ee37c05aa77a7acf578b5a1d6f2335a95c8605cfbc1d0e26bbb2.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/b3e472c29bf7ee37c05aa77a7acf578b5a1d6f2335a95c8605cfbc1d0e26bbb2.jpg)

![b8cd5e18263c34bbf0f170bced8dd8678abb817bbcd2e92fdf28eb3e5d5d13ef.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/b8cd5e18263c34bbf0f170bced8dd8678abb817bbcd2e92fdf28eb3e5d5d13ef.jpg)

![ba8e939830f49ea932ebccdff87dbb20a89a10a6c8d1d58d8cd8596c3b701cc4.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/ba8e939830f49ea932ebccdff87dbb20a89a10a6c8d1d58d8cd8596c3b701cc4.jpg)

![badfc151314d6fbcbdc3cb80ea699e1f8342540a7582af20712aa50d5b67295c.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/badfc151314d6fbcbdc3cb80ea699e1f8342540a7582af20712aa50d5b67295c.jpg)

![beaf511c05f19f504f362fd47fa24c16a895aba84d879f975c6a5e34b5659d65.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/beaf511c05f19f504f362fd47fa24c16a895aba84d879f975c6a5e34b5659d65.jpg)

![bfd32b42287bde5b8b150ed521a384d8c48cfb0a1a6c81122e8483a9b75f429a.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/bfd32b42287bde5b8b150ed521a384d8c48cfb0a1a6c81122e8483a9b75f429a.jpg)

![c21101e6dc479a00f6e92a9e499a168a7b8316395bbc23a8f86871aec038301e.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/c21101e6dc479a00f6e92a9e499a168a7b8316395bbc23a8f86871aec038301e.jpg)

![c62812dfe1ca139d4fab63ac2d5034e5ddb415f4504c27d76efb4127f12fefc9.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/c62812dfe1ca139d4fab63ac2d5034e5ddb415f4504c27d76efb4127f12fefc9.jpg)

![c6363d539a51c59cb833c91cb05e7c6f7c200bec10d4d6bca038b6cf6a8d2934.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/c6363d539a51c59cb833c91cb05e7c6f7c200bec10d4d6bca038b6cf6a8d2934.jpg)

![c94901d8a47e916bd0534ea0ac4041a04df5b72f2d4764064c83f4c91353af7c.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/c94901d8a47e916bd0534ea0ac4041a04df5b72f2d4764064c83f4c91353af7c.jpg)

![cabda54c85fadf33372460d3aef92e5fd83edae0c2fd4fabd79810b6a36ff1c6.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/cabda54c85fadf33372460d3aef92e5fd83edae0c2fd4fabd79810b6a36ff1c6.jpg)

![cbeaf02c6cbae7974ab1a103244dd7c47f35874246bb571eaf6e6ce7fdad1a58.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/cbeaf02c6cbae7974ab1a103244dd7c47f35874246bb571eaf6e6ce7fdad1a58.jpg)

![ccd14b3dbf3db47a4cd130a9c24940034af5fe1f9f4441fdb4da7e6fc30d99da.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/ccd14b3dbf3db47a4cd130a9c24940034af5fe1f9f4441fdb4da7e6fc30d99da.jpg)

![ced76a7d6c4c28c3e04cba36cc6e8b38a7167f595d54dc91a71c71dda6d88c04.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/ced76a7d6c4c28c3e04cba36cc6e8b38a7167f595d54dc91a71c71dda6d88c04.jpg)

![cfa97f6c7e50a58a0ae0385c90b436f16b5880a56be306e34923d0e2973ee9f5.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/cfa97f6c7e50a58a0ae0385c90b436f16b5880a56be306e34923d0e2973ee9f5.jpg)

![d257f73fdbc7cf177a17c2fd722c7e4974acefee0656dc2bb3ac3032de2fc25f.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/d257f73fdbc7cf177a17c2fd722c7e4974acefee0656dc2bb3ac3032de2fc25f.jpg)

![d3accf4d550d8e391c80f75fdfedc271f66d60ab25391078de3e4d7f23a94a34.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/d3accf4d550d8e391c80f75fdfedc271f66d60ab25391078de3e4d7f23a94a34.jpg)

![d3fb2a0df74d47eaa4b3d154cfdbee11228b73d3802d9df0268a164e42e0e323.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/d3fb2a0df74d47eaa4b3d154cfdbee11228b73d3802d9df0268a164e42e0e323.jpg)

![d432f12cb7669d4b74ad91cb9b8411d02825ece3cfede410223193329e5918b1.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/d432f12cb7669d4b74ad91cb9b8411d02825ece3cfede410223193329e5918b1.jpg)

![d4907978f60c3b5c6168c961bcb44d46750386ebab5a62184542234cc7e54309.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/d4907978f60c3b5c6168c961bcb44d46750386ebab5a62184542234cc7e54309.jpg)

![d605b54509b74e33f7110b577f1d2f57f13580abb96b8a44d2aaf32e26653082.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/d605b54509b74e33f7110b577f1d2f57f13580abb96b8a44d2aaf32e26653082.jpg)

![de7e18e3dd6ffa8161950f4f5fdd53d8c765b47796418557491017136148eeb7.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/de7e18e3dd6ffa8161950f4f5fdd53d8c765b47796418557491017136148eeb7.jpg)

![e0f9e88dfe6823a217e08a4643b846aef1209e15c9171f55d2bbec34eb31eb16.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/e0f9e88dfe6823a217e08a4643b846aef1209e15c9171f55d2bbec34eb31eb16.jpg)

![e2ef3823823e9abdb793fa083856c0d6f219977ccbff952f775d960538967490.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/e2ef3823823e9abdb793fa083856c0d6f219977ccbff952f775d960538967490.jpg)

![e7346d7236535483ae149b888f9068a6e41ae6b3a79e79cfd8a29baaf876d95c.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/e7346d7236535483ae149b888f9068a6e41ae6b3a79e79cfd8a29baaf876d95c.jpg)

![e95d757edd49dd772df5cec9283b61a02b597c95e60bbd61007497348ee3fee2.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/e95d757edd49dd772df5cec9283b61a02b597c95e60bbd61007497348ee3fee2.jpg)

![ed292a6c73552d30a693cabe61db59bc357d9d2c10067f231834b68944a0a859.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/ed292a6c73552d30a693cabe61db59bc357d9d2c10067f231834b68944a0a859.jpg)

![ef832944573d7a853a3ff3e5e999d4a8bd0c01001974119a4ceb8d1153acf95e.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/ef832944573d7a853a3ff3e5e999d4a8bd0c01001974119a4ceb8d1153acf95e.jpg)

![f2231a4a9e8ed7031da5aa5805cb8408fec26de1c9b03549316b012245ba577f.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/f2231a4a9e8ed7031da5aa5805cb8408fec26de1c9b03549316b012245ba577f.jpg)

![f448876ccadaafce15fe7275b51601a463276ded7b1d3329e7cdf0eaf3c19d61.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/f448876ccadaafce15fe7275b51601a463276ded7b1d3329e7cdf0eaf3c19d61.jpg)

![f47950e812eed525cb300235a39bcb6abd671e08e93660aa8dd77bfb8c82c7d0.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/f47950e812eed525cb300235a39bcb6abd671e08e93660aa8dd77bfb8c82c7d0.jpg)

![f5496fb058923f70a307136cde4b97f134ba537678f1133d62733b9c1bca28c6.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/f5496fb058923f70a307136cde4b97f134ba537678f1133d62733b9c1bca28c6.jpg)

![fa18e3fdba3983c48aa125ab60db06ba8bcd04f0acb03c6a5d22859428541553.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/fa18e3fdba3983c48aa125ab60db06ba8bcd04f0acb03c6a5d22859428541553.jpg)

![fac590b5243a76f7cb5a2b12d85423dfa97c44f2ae7b1bdf55ff1f151bdfb9ca.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/fac590b5243a76f7cb5a2b12d85423dfa97c44f2ae7b1bdf55ff1f151bdfb9ca.jpg)

![fec48a58d27b76369e3e5f72e112e2f0d2a80b65f05ec3bae0d154dc07ec2e00.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/fec48a58d27b76369e3e5f72e112e2f0d2a80b65f05ec3bae0d154dc07ec2e00.jpg)

![fed482fdca3a08c996c27193f4c352ffa215c1d3888421ca45c77930afa95096.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/fed482fdca3a08c996c27193f4c352ffa215c1d3888421ca45c77930afa95096.jpg)

![fef366fca96d3fbbb2527441fc46f2a5f0cf3f66d0355c283f8f33e2e97d4132.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/images/fef366fca96d3fbbb2527441fc46f2a5f0cf3f66d0355c283f8f33e2e97d4132.jpg)

### Tables

![0110138c735d29b9a350849e0419ef19923198a68493f3820d0d5f5b3687d62f.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/tables/0110138c735d29b9a350849e0419ef19923198a68493f3820d0d5f5b3687d62f.jpg)

![381a5f8f2e12a8c42d2ce2ef96382b656ac8b80bb16e9e96af13c90735fa0a07.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/tables/381a5f8f2e12a8c42d2ce2ef96382b656ac8b80bb16e9e96af13c90735fa0a07.jpg)

![4ec5a8e7d9fd1fc6c82494b70af86fb3ff4705cb9f20521653f512ff34ca9459.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/tables/4ec5a8e7d9fd1fc6c82494b70af86fb3ff4705cb9f20521653f512ff34ca9459.jpg)

![6cc171ffa93b3d04b28523c28cabba6786f96bbba9892fb7e6210c14ae9c63f4.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/tables/6cc171ffa93b3d04b28523c28cabba6786f96bbba9892fb7e6210c14ae9c63f4.jpg)

![79fb43df4bb6515787db86e0b4cef5e76dde683d5fd9d5baa84e2b16e472a200.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/tables/79fb43df4bb6515787db86e0b4cef5e76dde683d5fd9d5baa84e2b16e472a200.jpg)

![b3c4579ac3c164a937af4cb63cac60bf142a949d0519e0a14ea4ab82e07e6bf9.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/tables/b3c4579ac3c164a937af4cb63cac60bf142a949d0519e0a14ea4ab82e07e6bf9.jpg)

![c5192c0488600af2a23688ee40d58f299cf3efadc6bdbef0f2ef3b0cc3743984.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/tables/c5192c0488600af2a23688ee40d58f299cf3efadc6bdbef0f2ef3b0cc3743984.jpg)

![dfd5e671cb285fb1f8a960ccf83e3e32807b03b5d1cd3e171a01bfcc35d6cbe1.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/tables/dfd5e671cb285fb1f8a960ccf83e3e32807b03b5d1cd3e171a01bfcc35d6cbe1.jpg)

![e92001190a5e1a6e11fa36aa409b82707f0b3593aa541b28f84345f10ccf7703.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/tables/e92001190a5e1a6e11fa36aa409b82707f0b3593aa541b28f84345f10ccf7703.jpg)

![ecffe27695a8a1e147fd323d45ce6868160ff2c876f96917927e039c799138bc.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/tables/ecffe27695a8a1e147fd323d45ce6868160ff2c876f96917927e039c799138bc.jpg)

![f0243106669e1d62bf497cdd91f81b9f7c1a41a208db62f3e07daac9750c336a.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/tables/f0243106669e1d62bf497cdd91f81b9f7c1a41a208db62f3e07daac9750c336a.jpg)

![f33d0a740d6901d6bcf509bd804b4d7862d77b58caf724154a2a00902251eb99.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/tables/f33d0a740d6901d6bcf509bd804b4d7862d77b58caf724154a2a00902251eb99.jpg)

![fd8be430346b9f2c37f87567e3c86d0d53a118281cc90c6e67a65e22f5aa744a.jpg](../iclr_results/3101_LucidPPN_%20Unambiguous%20Prototypical%20Parts%20Network%20for%20User-centric%20Interpretable%20Computer%20Vision/tables/fd8be430346b9f2c37f87567e3c86d0d53a118281cc90c6e67a65e22f5aa744a.jpg)

## Generating CAD Code with Vision-Language Models for 3D Designs


### Images

![1205670376ce0024347e8f928dad186d4807d7a55bc40969e19693f8b05bd1b8.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/images/1205670376ce0024347e8f928dad186d4807d7a55bc40969e19693f8b05bd1b8.jpg)

![1793eaa38844aaa08bfd48d07f0df007c3362488777a50f4f03331a59c8d6d4f.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/images/1793eaa38844aaa08bfd48d07f0df007c3362488777a50f4f03331a59c8d6d4f.jpg)

![3f5ccabb5edff74f26b00116485fed5d08389b928e5aa41e1a4675e1fe3e7c8a.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/images/3f5ccabb5edff74f26b00116485fed5d08389b928e5aa41e1a4675e1fe3e7c8a.jpg)

![41cbfd4306fd990e096ae357ebe8c6558e54697f39287c65c3115d537c37ee6c.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/images/41cbfd4306fd990e096ae357ebe8c6558e54697f39287c65c3115d537c37ee6c.jpg)

![54007e0423dd06c9b7caa8f8aff1d06f89067f3d8963293348fca769c18161a9.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/images/54007e0423dd06c9b7caa8f8aff1d06f89067f3d8963293348fca769c18161a9.jpg)

![655d35848e2089d89aa86eb12781da1934b9ba0fde6a03556302c6766bd40788.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/images/655d35848e2089d89aa86eb12781da1934b9ba0fde6a03556302c6766bd40788.jpg)

![6c64084bbd0c1c4b2f186174ca571180f48b8ae8f215b5f2bac12523c0e9a12c.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/images/6c64084bbd0c1c4b2f186174ca571180f48b8ae8f215b5f2bac12523c0e9a12c.jpg)

![782aa26c4f266076d510219a331cc833eff79f3e22cf3c9d9992af773d8c9955.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/images/782aa26c4f266076d510219a331cc833eff79f3e22cf3c9d9992af773d8c9955.jpg)

![7847921f1a400384439036a683e059410de0a04465d9e14cc0a672d7537e005b.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/images/7847921f1a400384439036a683e059410de0a04465d9e14cc0a672d7537e005b.jpg)

![7ab8693ceb52d28d08a6dda163b548600aae1e84a030dbfbec43ed7a483e62cb.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/images/7ab8693ceb52d28d08a6dda163b548600aae1e84a030dbfbec43ed7a483e62cb.jpg)

![91707f44212744692f9e472d0d43d44c6fd5c60d93602dd44fb8f1262ba0f78a.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/images/91707f44212744692f9e472d0d43d44c6fd5c60d93602dd44fb8f1262ba0f78a.jpg)

![9a254f75c20e3d1b88484476090b1b539c07bcd04fefb485685ae41c16fb2d71.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/images/9a254f75c20e3d1b88484476090b1b539c07bcd04fefb485685ae41c16fb2d71.jpg)

![b41f1ddfc396e1a1dd441d30bc8db61d1bd29ddca4f124028111f2f44a52d2c7.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/images/b41f1ddfc396e1a1dd441d30bc8db61d1bd29ddca4f124028111f2f44a52d2c7.jpg)

![c30ae5103689498559b3f5b58e7358e17d692a818f0ef9e340c5e6d9b48b2cdb.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/images/c30ae5103689498559b3f5b58e7358e17d692a818f0ef9e340c5e6d9b48b2cdb.jpg)

![cb8a3cef52de04b8743c6d11876dd796ed03dd9197d16ddd1b53e69be1b063ab.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/images/cb8a3cef52de04b8743c6d11876dd796ed03dd9197d16ddd1b53e69be1b063ab.jpg)

![d1bb02f8695750e6e9259e54d1a6d9db1651af283c531e63bf53bb6b63bf9605.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/images/d1bb02f8695750e6e9259e54d1a6d9db1651af283c531e63bf53bb6b63bf9605.jpg)

![f645f907a3ec914c18c16c714459cd9fab90e7dde8d1dddc847978650c101f83.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/images/f645f907a3ec914c18c16c714459cd9fab90e7dde8d1dddc847978650c101f83.jpg)

### Tables

![02981b0a66d9ef9df454a32b1f64eacbbcc311eaf68e40e73dff3c6b84403030.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/tables/02981b0a66d9ef9df454a32b1f64eacbbcc311eaf68e40e73dff3c6b84403030.jpg)

![255c858a204e5a729915f3874d009e9c6244d3d101ee6144780632d0f30f690d.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/tables/255c858a204e5a729915f3874d009e9c6244d3d101ee6144780632d0f30f690d.jpg)

![3d9468a6294366092999c5e3ffd41d5b29ede894e76f3ac80d04d9f7377255ed.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/tables/3d9468a6294366092999c5e3ffd41d5b29ede894e76f3ac80d04d9f7377255ed.jpg)

![5cfdcc34f3151c7c7d74cc4181b83e974915928cf34217acd6d04d7d6dbe560e.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/tables/5cfdcc34f3151c7c7d74cc4181b83e974915928cf34217acd6d04d7d6dbe560e.jpg)

![8094b5c8c273e7b92a2e23bc3580f0eeeb898c1bde42621c61803bef1247df1c.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/tables/8094b5c8c273e7b92a2e23bc3580f0eeeb898c1bde42621c61803bef1247df1c.jpg)

![8997096d581936e392e6535ec870674eafe9538750b9b648a26db929cfa31557.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/tables/8997096d581936e392e6535ec870674eafe9538750b9b648a26db929cfa31557.jpg)

![bc2f1d2ea4971f4c72457d178c6fdc085fb3bb5a5345cce31ab3e87348ac27c7.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/tables/bc2f1d2ea4971f4c72457d178c6fdc085fb3bb5a5345cce31ab3e87348ac27c7.jpg)

![c1bdc376f6e7ac991ad35bc191508c84fde18a054cee53886b011083d6c0b4f3.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/tables/c1bdc376f6e7ac991ad35bc191508c84fde18a054cee53886b011083d6c0b4f3.jpg)

![dd65ab4264896559d2504e1bb2aef1e992d093f60bd443b800021188a51fec00.jpg](../iclr_results/3102_Generating%20CAD%20Code%20with%20Vision-Language%20Models%20for%203D%20Designs/tables/dd65ab4264896559d2504e1bb2aef1e992d093f60bd443b800021188a51fec00.jpg)

## CBMA: Improving Conformal Prediction through Bayesian Model Averaging


### Images

![243baf9116164b1c83674e7bf58d1d2f64ac5eabc0d6d48cbc65460ef37ba3a1.jpg](../iclr_results/3103_CBMA_%20Improving%20Conformal%20Prediction%20through%20Bayesian%20Model%20Averaging/images/243baf9116164b1c83674e7bf58d1d2f64ac5eabc0d6d48cbc65460ef37ba3a1.jpg)

![2ed6f54b3d9f0e7afb928235bc94baa44f3c8b4123ec38aa29b88c987a24b3fe.jpg](../iclr_results/3103_CBMA_%20Improving%20Conformal%20Prediction%20through%20Bayesian%20Model%20Averaging/images/2ed6f54b3d9f0e7afb928235bc94baa44f3c8b4123ec38aa29b88c987a24b3fe.jpg)

![3bd5c4890ae77e155b8743c19194f772cb0250e5ebe14ba9bdbdff359175054d.jpg](../iclr_results/3103_CBMA_%20Improving%20Conformal%20Prediction%20through%20Bayesian%20Model%20Averaging/images/3bd5c4890ae77e155b8743c19194f772cb0250e5ebe14ba9bdbdff359175054d.jpg)

![4e59f54e78d5df9b54612b49cd481c9381226ea086f948175b6659c806ddc5cc.jpg](../iclr_results/3103_CBMA_%20Improving%20Conformal%20Prediction%20through%20Bayesian%20Model%20Averaging/images/4e59f54e78d5df9b54612b49cd481c9381226ea086f948175b6659c806ddc5cc.jpg)

![5716f9869f82e547d351031da42bcd0eb138f5d06ec5b9867f7830e3e1dbbdb7.jpg](../iclr_results/3103_CBMA_%20Improving%20Conformal%20Prediction%20through%20Bayesian%20Model%20Averaging/images/5716f9869f82e547d351031da42bcd0eb138f5d06ec5b9867f7830e3e1dbbdb7.jpg)

![ed1a6ba1e32a0d7ab259cb20c43f4994f480ac2e1642c62163a08d136b5e5223.jpg](../iclr_results/3103_CBMA_%20Improving%20Conformal%20Prediction%20through%20Bayesian%20Model%20Averaging/images/ed1a6ba1e32a0d7ab259cb20c43f4994f480ac2e1642c62163a08d136b5e5223.jpg)

### Tables

![5edc083c38766b722fb727c245a0721d12700746e8fcd46ae700cb79a8383c3f.jpg](../iclr_results/3103_CBMA_%20Improving%20Conformal%20Prediction%20through%20Bayesian%20Model%20Averaging/tables/5edc083c38766b722fb727c245a0721d12700746e8fcd46ae700cb79a8383c3f.jpg)

![c611dbeafc49d20eb5d07f8cb21e84b6ef5d6f070a08ae01843aa3282ae36929.jpg](../iclr_results/3103_CBMA_%20Improving%20Conformal%20Prediction%20through%20Bayesian%20Model%20Averaging/tables/c611dbeafc49d20eb5d07f8cb21e84b6ef5d6f070a08ae01843aa3282ae36929.jpg)

![c6c5e3443ea47b63208b186aa3cecdf81d1f26e8bfcd83510929d044927e9c56.jpg](../iclr_results/3103_CBMA_%20Improving%20Conformal%20Prediction%20through%20Bayesian%20Model%20Averaging/tables/c6c5e3443ea47b63208b186aa3cecdf81d1f26e8bfcd83510929d044927e9c56.jpg)

![c7285e0bd41568f77ab7339730f32d7a37b5afef6182bd37cd18a575f75ad360.jpg](../iclr_results/3103_CBMA_%20Improving%20Conformal%20Prediction%20through%20Bayesian%20Model%20Averaging/tables/c7285e0bd41568f77ab7339730f32d7a37b5afef6182bd37cd18a575f75ad360.jpg)

![e6dc6e8eaffe4885d0058ac3a59900b107391e3de5373983fc4e04b9a689aa24.jpg](../iclr_results/3103_CBMA_%20Improving%20Conformal%20Prediction%20through%20Bayesian%20Model%20Averaging/tables/e6dc6e8eaffe4885d0058ac3a59900b107391e3de5373983fc4e04b9a689aa24.jpg)

![e81130563637c3318468abbd92391c8dc9664d21f81e5148c909cc6696334a0f.jpg](../iclr_results/3103_CBMA_%20Improving%20Conformal%20Prediction%20through%20Bayesian%20Model%20Averaging/tables/e81130563637c3318468abbd92391c8dc9664d21f81e5148c909cc6696334a0f.jpg)

![ee4b4caf65712c201c90083c5a1c9b1d389410e1ee545c2729f99ea20d0220ac.jpg](../iclr_results/3103_CBMA_%20Improving%20Conformal%20Prediction%20through%20Bayesian%20Model%20Averaging/tables/ee4b4caf65712c201c90083c5a1c9b1d389410e1ee545c2729f99ea20d0220ac.jpg)

## Human-inspired Episodic Memory for Infinite Context LLMs


### Images

![0507128649b0be6372cfbc2e52395bf310a2ff6f0ad076db77e36c76b35d5ca1.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/images/0507128649b0be6372cfbc2e52395bf310a2ff6f0ad076db77e36c76b35d5ca1.jpg)

![0ed3b5a3004933695d6764ffaed8d41a9533b09f57faff3ec4dec9dc3a595912.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/images/0ed3b5a3004933695d6764ffaed8d41a9533b09f57faff3ec4dec9dc3a595912.jpg)

![3e24fbe1bc7dee1a59e9da1640eca7013ace285b2468f374d6aaba74ad7641a1.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/images/3e24fbe1bc7dee1a59e9da1640eca7013ace285b2468f374d6aaba74ad7641a1.jpg)

![4d3da7278da2ff9e3d13390b3ed22783c7f9df7a6b39ad1ebfc31dc17582e445.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/images/4d3da7278da2ff9e3d13390b3ed22783c7f9df7a6b39ad1ebfc31dc17582e445.jpg)

![4e653d5636aabf6b29f7401d1d0aebcc0ce25c333d438505e9e624465a4457e5.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/images/4e653d5636aabf6b29f7401d1d0aebcc0ce25c333d438505e9e624465a4457e5.jpg)

![68b84a227942b05d4475de2faa2139d9032ce2b5e0e8e8ffd0795ba77708de34.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/images/68b84a227942b05d4475de2faa2139d9032ce2b5e0e8e8ffd0795ba77708de34.jpg)

![6f4841234e9c64cf36715ab18de57f342e39f081a81e90daf1f42696cb2bccf2.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/images/6f4841234e9c64cf36715ab18de57f342e39f081a81e90daf1f42696cb2bccf2.jpg)

![ab6bf99aa4cb8c92c9ee5d1290217fe016d35dc53064d09ad28dad40ce5907e4.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/images/ab6bf99aa4cb8c92c9ee5d1290217fe016d35dc53064d09ad28dad40ce5907e4.jpg)

![b1dbbb4bfbb641022a730a81a1a5c6fe7f609ca996f37377752f019d34ef7e51.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/images/b1dbbb4bfbb641022a730a81a1a5c6fe7f609ca996f37377752f019d34ef7e51.jpg)

![c2abd7fbb510c4d45b41ac5f07c3091647c75206e271f4d788049ba04d52277a.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/images/c2abd7fbb510c4d45b41ac5f07c3091647c75206e271f4d788049ba04d52277a.jpg)

![c7cae53af305d03185f037f3f8c53ba37561fb00701c16031aeb8c4d487db96f.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/images/c7cae53af305d03185f037f3f8c53ba37561fb00701c16031aeb8c4d487db96f.jpg)

![da8cfef46c7aee0326e7b37ed65f52d8699ded0488cbd54c2af41f1e09098450.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/images/da8cfef46c7aee0326e7b37ed65f52d8699ded0488cbd54c2af41f1e09098450.jpg)

![f6630dac9a199af69372823357e8064fa89e745859ba8d32f43f06920fcce304.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/images/f6630dac9a199af69372823357e8064fa89e745859ba8d32f43f06920fcce304.jpg)

![fda9c13edd4a3abedbfb28e01abe06ccec50f2e1891ed598b28982ea4aa02a9c.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/images/fda9c13edd4a3abedbfb28e01abe06ccec50f2e1891ed598b28982ea4aa02a9c.jpg)

### Tables

![123eb34761fce9d4449a34d0a37f8773e0156dc94f1311c85440e55337d1db10.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/tables/123eb34761fce9d4449a34d0a37f8773e0156dc94f1311c85440e55337d1db10.jpg)

![265f3e6dfbe30fcae3eaa2029feadf49aa060dc096f78bce77a547eab6048cc2.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/tables/265f3e6dfbe30fcae3eaa2029feadf49aa060dc096f78bce77a547eab6048cc2.jpg)

![37987d5bc7058cce2e62cb264f8c56e14b4678de37da59d7373e660f955ba5db.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/tables/37987d5bc7058cce2e62cb264f8c56e14b4678de37da59d7373e660f955ba5db.jpg)

![46f0aa64bdab67912fedfae9ed4f96cb70e2c33d0831d850925a856cd953bdd3.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/tables/46f0aa64bdab67912fedfae9ed4f96cb70e2c33d0831d850925a856cd953bdd3.jpg)

![6cde88856a751e7298c2b9b3d1ce70068f0fb356a84b169babd353619e423872.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/tables/6cde88856a751e7298c2b9b3d1ce70068f0fb356a84b169babd353619e423872.jpg)

![9eb9ea3a7ac4a3dec0dab7c94b801c9a09a228e28b4e8203b5fcb98660ca19ba.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/tables/9eb9ea3a7ac4a3dec0dab7c94b801c9a09a228e28b4e8203b5fcb98660ca19ba.jpg)

![ba1f23a393e0a794911438d897b94f6a2955885be77ca166e0e9286e8bc3dff6.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/tables/ba1f23a393e0a794911438d897b94f6a2955885be77ca166e0e9286e8bc3dff6.jpg)

![c421f7a6f72da1ee0ba05f4963e4de19318b526f1f6e64bf61e6c1868fa32e3c.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/tables/c421f7a6f72da1ee0ba05f4963e4de19318b526f1f6e64bf61e6c1868fa32e3c.jpg)

![c4ed77f1784137c5d073be0a2b77e21aa3d28388c57fd8fa937fdb01068eb0d6.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/tables/c4ed77f1784137c5d073be0a2b77e21aa3d28388c57fd8fa937fdb01068eb0d6.jpg)

![cbd0c77ecd8e7c20dba6f34909098a1bb0dd6ac0fcdb40d65d073098adb09fc0.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/tables/cbd0c77ecd8e7c20dba6f34909098a1bb0dd6ac0fcdb40d65d073098adb09fc0.jpg)

![cd9f75d4e747a265b997d3be526befc7d09ca00815786e30eb6831f5e16f5213.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/tables/cd9f75d4e747a265b997d3be526befc7d09ca00815786e30eb6831f5e16f5213.jpg)

![d1354e75889e503de01071bc4e8b7459b0c1a2692e045345c7616714792b4b3f.jpg](../iclr_results/3104_Human-inspired%20Episodic%20Memory%20for%20Infinite%20Context%20LLMs/tables/d1354e75889e503de01071bc4e8b7459b0c1a2692e045345c7616714792b4b3f.jpg)

## Constructing Confidence Intervals for Average Treatment Effects from Multiple Datasets


### Images

![0b41fbfda77a1855de2c5a8cd31569b89b33a2b6c7b649685e42d883a17adb49.jpg](../iclr_results/3105_Constructing%20Confidence%20Intervals%20for%20Average%20Treatment%20Effects%20from%20Multiple%20Datasets/images/0b41fbfda77a1855de2c5a8cd31569b89b33a2b6c7b649685e42d883a17adb49.jpg)

![43d532add8114436237f802a801bb9896a98415425d7293e2074c60e5e5c5896.jpg](../iclr_results/3105_Constructing%20Confidence%20Intervals%20for%20Average%20Treatment%20Effects%20from%20Multiple%20Datasets/images/43d532add8114436237f802a801bb9896a98415425d7293e2074c60e5e5c5896.jpg)

![484eebbb185d94a3aaef741f05905cdeb53298c7f353960ab1eb3e8a9a32d67d.jpg](../iclr_results/3105_Constructing%20Confidence%20Intervals%20for%20Average%20Treatment%20Effects%20from%20Multiple%20Datasets/images/484eebbb185d94a3aaef741f05905cdeb53298c7f353960ab1eb3e8a9a32d67d.jpg)

![5378826d274d00830a50925613458c28f094b30f07134815cfde2b6b73466f31.jpg](../iclr_results/3105_Constructing%20Confidence%20Intervals%20for%20Average%20Treatment%20Effects%20from%20Multiple%20Datasets/images/5378826d274d00830a50925613458c28f094b30f07134815cfde2b6b73466f31.jpg)

![5a28d6e8ad8a582c261ef3a65780dc1ab524a1a746c013dca0a953b20168412a.jpg](../iclr_results/3105_Constructing%20Confidence%20Intervals%20for%20Average%20Treatment%20Effects%20from%20Multiple%20Datasets/images/5a28d6e8ad8a582c261ef3a65780dc1ab524a1a746c013dca0a953b20168412a.jpg)

![72d7257fc6cfafc081f56f2f41515362dde8a743fc7064af7fe295c8baf903ce.jpg](../iclr_results/3105_Constructing%20Confidence%20Intervals%20for%20Average%20Treatment%20Effects%20from%20Multiple%20Datasets/images/72d7257fc6cfafc081f56f2f41515362dde8a743fc7064af7fe295c8baf903ce.jpg)

![79a820af637dd90d6fad1965591d316ba957786aee4a4893138843f7336120f7.jpg](../iclr_results/3105_Constructing%20Confidence%20Intervals%20for%20Average%20Treatment%20Effects%20from%20Multiple%20Datasets/images/79a820af637dd90d6fad1965591d316ba957786aee4a4893138843f7336120f7.jpg)

![902d63cc0767c838ca914eaa31ef9e5371f095f354cc886d2f57b4685d5b4914.jpg](../iclr_results/3105_Constructing%20Confidence%20Intervals%20for%20Average%20Treatment%20Effects%20from%20Multiple%20Datasets/images/902d63cc0767c838ca914eaa31ef9e5371f095f354cc886d2f57b4685d5b4914.jpg)

![9778e226d9dcf1864aef0c5e68a1975e8336a927232761b4eda84b22bcd95606.jpg](../iclr_results/3105_Constructing%20Confidence%20Intervals%20for%20Average%20Treatment%20Effects%20from%20Multiple%20Datasets/images/9778e226d9dcf1864aef0c5e68a1975e8336a927232761b4eda84b22bcd95606.jpg)

![a2797f2d392f5f5722a0de2aec95bc97ebb8cf920331eb352343e43a44728c59.jpg](../iclr_results/3105_Constructing%20Confidence%20Intervals%20for%20Average%20Treatment%20Effects%20from%20Multiple%20Datasets/images/a2797f2d392f5f5722a0de2aec95bc97ebb8cf920331eb352343e43a44728c59.jpg)

![a68eac25a45633c2c02b7054220ab6baaeb000966a4ebf9389a283bc5eaf04d3.jpg](../iclr_results/3105_Constructing%20Confidence%20Intervals%20for%20Average%20Treatment%20Effects%20from%20Multiple%20Datasets/images/a68eac25a45633c2c02b7054220ab6baaeb000966a4ebf9389a283bc5eaf04d3.jpg)

![daf0b15e626a6671792d78f7682bb2efcc5467d206e7dc2bc5509763f7fc8120.jpg](../iclr_results/3105_Constructing%20Confidence%20Intervals%20for%20Average%20Treatment%20Effects%20from%20Multiple%20Datasets/images/daf0b15e626a6671792d78f7682bb2efcc5467d206e7dc2bc5509763f7fc8120.jpg)

![df1f764d40d4b2164034bcdab193d1b07b15462067de732b6d5f2442de59671c.jpg](../iclr_results/3105_Constructing%20Confidence%20Intervals%20for%20Average%20Treatment%20Effects%20from%20Multiple%20Datasets/images/df1f764d40d4b2164034bcdab193d1b07b15462067de732b6d5f2442de59671c.jpg)

![f002a4f0021c01dc5028b8814ced1532e0634139f1c0ca45707cd4828b85cf45.jpg](../iclr_results/3105_Constructing%20Confidence%20Intervals%20for%20Average%20Treatment%20Effects%20from%20Multiple%20Datasets/images/f002a4f0021c01dc5028b8814ced1532e0634139f1c0ca45707cd4828b85cf45.jpg)

### Tables

![166c796e22226490e1843d14f0437496fbbb1aea02692f3c38c111c8f9dfaf6f.jpg](../iclr_results/3105_Constructing%20Confidence%20Intervals%20for%20Average%20Treatment%20Effects%20from%20Multiple%20Datasets/tables/166c796e22226490e1843d14f0437496fbbb1aea02692f3c38c111c8f9dfaf6f.jpg)

![584ecb2241ebfd97b9d28c89c4599c40dee8b8efcdcc2aa8bb5175de1b6d0c5f.jpg](../iclr_results/3105_Constructing%20Confidence%20Intervals%20for%20Average%20Treatment%20Effects%20from%20Multiple%20Datasets/tables/584ecb2241ebfd97b9d28c89c4599c40dee8b8efcdcc2aa8bb5175de1b6d0c5f.jpg)

## Horizon Generalization in Reinforcement Learning


### Images

![027b9414c2f9b40180ea9c7e1075b603b9943299ceddfeb8c9186a086a1311f3.jpg](../iclr_results/3106_Horizon%20Generalization%20in%20Reinforcement%20Learning/images/027b9414c2f9b40180ea9c7e1075b603b9943299ceddfeb8c9186a086a1311f3.jpg)

![1795f39470a8509c4e143a3f6aa5e57e18f74bd95e4bf0876b3ae5e4eece6245.jpg](../iclr_results/3106_Horizon%20Generalization%20in%20Reinforcement%20Learning/images/1795f39470a8509c4e143a3f6aa5e57e18f74bd95e4bf0876b3ae5e4eece6245.jpg)

![39438de60f9f285e618083b9c9992b25d257ffd7b7552731cc4a61912574c3c5.jpg](../iclr_results/3106_Horizon%20Generalization%20in%20Reinforcement%20Learning/images/39438de60f9f285e618083b9c9992b25d257ffd7b7552731cc4a61912574c3c5.jpg)

![3c595ee32f29516e5fcd279f6c4928a690eb361f8adf18db83d368ec91eacf44.jpg](../iclr_results/3106_Horizon%20Generalization%20in%20Reinforcement%20Learning/images/3c595ee32f29516e5fcd279f6c4928a690eb361f8adf18db83d368ec91eacf44.jpg)

![4d891c8dacc0ea77e1cfdbb1aaa014afee9c4626ef384ba3ea2ac872fd2c657e.jpg](../iclr_results/3106_Horizon%20Generalization%20in%20Reinforcement%20Learning/images/4d891c8dacc0ea77e1cfdbb1aaa014afee9c4626ef384ba3ea2ac872fd2c657e.jpg)

![4ee77a1363ea4fb2e2b3641b2836eff270efd6ec5f18337ab7ee2a62acdd8f6b.jpg](../iclr_results/3106_Horizon%20Generalization%20in%20Reinforcement%20Learning/images/4ee77a1363ea4fb2e2b3641b2836eff270efd6ec5f18337ab7ee2a62acdd8f6b.jpg)

![83d2c24283378a56d12920e44f3f87622caaa345fa43a40d1eee03da22fcd3a3.jpg](../iclr_results/3106_Horizon%20Generalization%20in%20Reinforcement%20Learning/images/83d2c24283378a56d12920e44f3f87622caaa345fa43a40d1eee03da22fcd3a3.jpg)

![92004f9235a5c828e4f459d0e27b5eb8ce257cb7ab3653e7c4aa3e66b6538330.jpg](../iclr_results/3106_Horizon%20Generalization%20in%20Reinforcement%20Learning/images/92004f9235a5c828e4f459d0e27b5eb8ce257cb7ab3653e7c4aa3e66b6538330.jpg)

![b1419acef54ce5370322190ab71e15932d3d169a42083df857f774323feb947e.jpg](../iclr_results/3106_Horizon%20Generalization%20in%20Reinforcement%20Learning/images/b1419acef54ce5370322190ab71e15932d3d169a42083df857f774323feb947e.jpg)

![b607be052a7e3c15ff9c0ed7647e7edda956626ce64244edb894a8f3854dc980.jpg](../iclr_results/3106_Horizon%20Generalization%20in%20Reinforcement%20Learning/images/b607be052a7e3c15ff9c0ed7647e7edda956626ce64244edb894a8f3854dc980.jpg)

![c8b6bf74ed2a478c64d3459e70d7b28dfa17695baa79d57f82cd490770cbf884.jpg](../iclr_results/3106_Horizon%20Generalization%20in%20Reinforcement%20Learning/images/c8b6bf74ed2a478c64d3459e70d7b28dfa17695baa79d57f82cd490770cbf884.jpg)

![f2f1237c7ce57f42aa7f79896ac052083f3882f3b925ab885db3f7d96b434ef2.jpg](../iclr_results/3106_Horizon%20Generalization%20in%20Reinforcement%20Learning/images/f2f1237c7ce57f42aa7f79896ac052083f3882f3b925ab885db3f7d96b434ef2.jpg)

### Tables

![9505b09bf94c507ff09e4b36b5e9933cb07106a58a11ea27c78ca503cbba46bb.jpg](../iclr_results/3106_Horizon%20Generalization%20in%20Reinforcement%20Learning/tables/9505b09bf94c507ff09e4b36b5e9933cb07106a58a11ea27c78ca503cbba46bb.jpg)

![ce4800db77cb7541ced430ed2ebcfdfa788d978577008cd5ea1770b4909dd850.jpg](../iclr_results/3106_Horizon%20Generalization%20in%20Reinforcement%20Learning/tables/ce4800db77cb7541ced430ed2ebcfdfa788d978577008cd5ea1770b4909dd850.jpg)

## Flavors of Margin: Implicit Bias of Steepest Descent in Homogeneous Neural Networks


### Images

![20a36a32e39d51c803d363267ef7e0516f0f214d037239087110638f2227fab6.jpg](../iclr_results/3107_Flavors%20of%20Margin_%20Implicit%20Bias%20of%20Steepest%20Descent%20in%20Homogeneous%20Neural%20Networks/images/20a36a32e39d51c803d363267ef7e0516f0f214d037239087110638f2227fab6.jpg)

![ae2c41edd60f9504f81deb3f5def1eb6598042f96044e0ebe1561c1a177eb74b.jpg](../iclr_results/3107_Flavors%20of%20Margin_%20Implicit%20Bias%20of%20Steepest%20Descent%20in%20Homogeneous%20Neural%20Networks/images/ae2c41edd60f9504f81deb3f5def1eb6598042f96044e0ebe1561c1a177eb74b.jpg)

![b6d76b189a81d5ec814fa925f2a2f196a0b1c8751f74eda2da6f83c823f154ad.jpg](../iclr_results/3107_Flavors%20of%20Margin_%20Implicit%20Bias%20of%20Steepest%20Descent%20in%20Homogeneous%20Neural%20Networks/images/b6d76b189a81d5ec814fa925f2a2f196a0b1c8751f74eda2da6f83c823f154ad.jpg)

![c7890b6d11c683e0eaa7d2be7162f8b9af498779ba4695b786ef3af98d6b3b5b.jpg](../iclr_results/3107_Flavors%20of%20Margin_%20Implicit%20Bias%20of%20Steepest%20Descent%20in%20Homogeneous%20Neural%20Networks/images/c7890b6d11c683e0eaa7d2be7162f8b9af498779ba4695b786ef3af98d6b3b5b.jpg)

![fff5fbd5b3422473063773eeeecf6cfb8b4d138a9ddeb2f74a5f29eab51ffcf2.jpg](../iclr_results/3107_Flavors%20of%20Margin_%20Implicit%20Bias%20of%20Steepest%20Descent%20in%20Homogeneous%20Neural%20Networks/images/fff5fbd5b3422473063773eeeecf6cfb8b4d138a9ddeb2f74a5f29eab51ffcf2.jpg)

## Mix-LN: Unleashing the Power of Deeper Layers by Combining Pre-LN and Post-LN


### Images

![096caf360863f4ede88c3629a923ede790ecf69fd431f726054f124febee6135.jpg](../iclr_results/3108_Mix-LN_%20Unleashing%20the%20Power%20of%20Deeper%20Layers%20by%20Combining%20Pre-LN%20and%20Post-LN/images/096caf360863f4ede88c3629a923ede790ecf69fd431f726054f124febee6135.jpg)

![3afdc7c165581c9aebded9e6e9b1cdec867361b19ab10c58c300c5f28429b282.jpg](../iclr_results/3108_Mix-LN_%20Unleashing%20the%20Power%20of%20Deeper%20Layers%20by%20Combining%20Pre-LN%20and%20Post-LN/images/3afdc7c165581c9aebded9e6e9b1cdec867361b19ab10c58c300c5f28429b282.jpg)

![464c6c8bd73bad02b6e957e5187c9ec5af215dc5d02d722a2c0cd36b19c4a0ad.jpg](../iclr_results/3108_Mix-LN_%20Unleashing%20the%20Power%20of%20Deeper%20Layers%20by%20Combining%20Pre-LN%20and%20Post-LN/images/464c6c8bd73bad02b6e957e5187c9ec5af215dc5d02d722a2c0cd36b19c4a0ad.jpg)

![72e414847596826bb9144f4ed7223853b9cfc713f187f3a2bbc951689bc2418a.jpg](../iclr_results/3108_Mix-LN_%20Unleashing%20the%20Power%20of%20Deeper%20Layers%20by%20Combining%20Pre-LN%20and%20Post-LN/images/72e414847596826bb9144f4ed7223853b9cfc713f187f3a2bbc951689bc2418a.jpg)

![7a5dd4216da970fae118fe6376a9dc5b878906c6e99d51b306a1f6ee8f6fc4b9.jpg](../iclr_results/3108_Mix-LN_%20Unleashing%20the%20Power%20of%20Deeper%20Layers%20by%20Combining%20Pre-LN%20and%20Post-LN/images/7a5dd4216da970fae118fe6376a9dc5b878906c6e99d51b306a1f6ee8f6fc4b9.jpg)

![96f97c3e2d873396b08d19b94fa4141f09dd71730ae606f3a0030e087db49ce1.jpg](../iclr_results/3108_Mix-LN_%20Unleashing%20the%20Power%20of%20Deeper%20Layers%20by%20Combining%20Pre-LN%20and%20Post-LN/images/96f97c3e2d873396b08d19b94fa4141f09dd71730ae606f3a0030e087db49ce1.jpg)

![e1bbbd1ac1e72ee952ca2f486f7becb4110762e6be52570f3cef63f74c080050.jpg](../iclr_results/3108_Mix-LN_%20Unleashing%20the%20Power%20of%20Deeper%20Layers%20by%20Combining%20Pre-LN%20and%20Post-LN/images/e1bbbd1ac1e72ee952ca2f486f7becb4110762e6be52570f3cef63f74c080050.jpg)

### Tables

![17517b9d0c409e0105491fe4ebcb44adb0dfe183e1928371dbf6a5e639f9f918.jpg](../iclr_results/3108_Mix-LN_%20Unleashing%20the%20Power%20of%20Deeper%20Layers%20by%20Combining%20Pre-LN%20and%20Post-LN/tables/17517b9d0c409e0105491fe4ebcb44adb0dfe183e1928371dbf6a5e639f9f918.jpg)

![1852c4f9a07ac97110dbdcc7fba23cdef2e32fdb077a2aad80d31d18cd934e86.jpg](../iclr_results/3108_Mix-LN_%20Unleashing%20the%20Power%20of%20Deeper%20Layers%20by%20Combining%20Pre-LN%20and%20Post-LN/tables/1852c4f9a07ac97110dbdcc7fba23cdef2e32fdb077a2aad80d31d18cd934e86.jpg)

![30e2419b27d6dc47e9144b50e05407a1bd79f7823f4b0f9c4fc806afa75471d2.jpg](../iclr_results/3108_Mix-LN_%20Unleashing%20the%20Power%20of%20Deeper%20Layers%20by%20Combining%20Pre-LN%20and%20Post-LN/tables/30e2419b27d6dc47e9144b50e05407a1bd79f7823f4b0f9c4fc806afa75471d2.jpg)

![35d168646d3b3992099251f9adf8956e38512def9bb76f71c4a7a68c1ac6d4be.jpg](../iclr_results/3108_Mix-LN_%20Unleashing%20the%20Power%20of%20Deeper%20Layers%20by%20Combining%20Pre-LN%20and%20Post-LN/tables/35d168646d3b3992099251f9adf8956e38512def9bb76f71c4a7a68c1ac6d4be.jpg)

![3e997239bdea44af6f65d93435f4f95c575e654a911ddeaadbeccc5ff06715a7.jpg](../iclr_results/3108_Mix-LN_%20Unleashing%20the%20Power%20of%20Deeper%20Layers%20by%20Combining%20Pre-LN%20and%20Post-LN/tables/3e997239bdea44af6f65d93435f4f95c575e654a911ddeaadbeccc5ff06715a7.jpg)

![634aeddb83ad5ece7fe3afd10f0aed3ab119ab6702ad3195125518ee1134e4d4.jpg](../iclr_results/3108_Mix-LN_%20Unleashing%20the%20Power%20of%20Deeper%20Layers%20by%20Combining%20Pre-LN%20and%20Post-LN/tables/634aeddb83ad5ece7fe3afd10f0aed3ab119ab6702ad3195125518ee1134e4d4.jpg)

![ac4b70eacd467225ee5e0f6ac4d8a7fe3ce5e445bcad943525646ec8b4cda1d7.jpg](../iclr_results/3108_Mix-LN_%20Unleashing%20the%20Power%20of%20Deeper%20Layers%20by%20Combining%20Pre-LN%20and%20Post-LN/tables/ac4b70eacd467225ee5e0f6ac4d8a7fe3ce5e445bcad943525646ec8b4cda1d7.jpg)

![acc542fea50288e814251b92243dfe1ae5665df4af3fdb2cb1b264fb9af3dcc9.jpg](../iclr_results/3108_Mix-LN_%20Unleashing%20the%20Power%20of%20Deeper%20Layers%20by%20Combining%20Pre-LN%20and%20Post-LN/tables/acc542fea50288e814251b92243dfe1ae5665df4af3fdb2cb1b264fb9af3dcc9.jpg)

## Kolmogorov-Arnold Transformer


### Images

![516f3411669c7eb5799f6a4953668f4c72bbf39a257947f11c53c4fc4c258447.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/images/516f3411669c7eb5799f6a4953668f4c72bbf39a257947f11c53c4fc4c258447.jpg)

![52978240e494936ba806aa2b75ca380301bf34f3562db76aecd4cc054cf872bc.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/images/52978240e494936ba806aa2b75ca380301bf34f3562db76aecd4cc054cf872bc.jpg)

![54713e98e76f6f04f7bf0e5291c1e0922e5fd5374cbe5b7b3aa8ffb854d5f765.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/images/54713e98e76f6f04f7bf0e5291c1e0922e5fd5374cbe5b7b3aa8ffb854d5f765.jpg)

![83b7bf28201e74e134ebd1841268e4ec8d77e53ac93ac096abf555a6482f7b06.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/images/83b7bf28201e74e134ebd1841268e4ec8d77e53ac93ac096abf555a6482f7b06.jpg)

![96d3988dde058fdeb494ae34050b766c4ab0777aed88940badd67bda6c9913e6.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/images/96d3988dde058fdeb494ae34050b766c4ab0777aed88940badd67bda6c9913e6.jpg)

![a33cb404c6cbde8aa854af33f4c790931a9ce5eead39495b04c7f020e45b7345.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/images/a33cb404c6cbde8aa854af33f4c790931a9ce5eead39495b04c7f020e45b7345.jpg)

![a351a607870017072c01897641f97fdec086a9878b16f840a88d0878f9281ec6.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/images/a351a607870017072c01897641f97fdec086a9878b16f840a88d0878f9281ec6.jpg)

![ba8be7560d3803ea64f399d31f26b2f52ae00a8cbbf1c1f48043c5628564dd0a.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/images/ba8be7560d3803ea64f399d31f26b2f52ae00a8cbbf1c1f48043c5628564dd0a.jpg)

![f7d07bf5a031f8bb4850fd57d45d3e5fff54852e1ea476ffed464645306d969e.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/images/f7d07bf5a031f8bb4850fd57d45d3e5fff54852e1ea476ffed464645306d969e.jpg)

### Tables

![1b471c40132cb222c96236a90cc9fecb13115f17bf3e8284c0caee90fe76c863.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/tables/1b471c40132cb222c96236a90cc9fecb13115f17bf3e8284c0caee90fe76c863.jpg)

![28448d7f3e57c9cacad51bf571eb4caeaee9465608d2b926fc63aacca4c2155c.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/tables/28448d7f3e57c9cacad51bf571eb4caeaee9465608d2b926fc63aacca4c2155c.jpg)

![2f3de757be1c883ee4cbf0b2edf9a3e9ad8f9eaeca190be2a95baeb9600b3e69.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/tables/2f3de757be1c883ee4cbf0b2edf9a3e9ad8f9eaeca190be2a95baeb9600b3e69.jpg)

![44e120980d1775fd1cd39798d2dff83ebae7348499f0f589b964b1b28f75bbc1.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/tables/44e120980d1775fd1cd39798d2dff83ebae7348499f0f589b964b1b28f75bbc1.jpg)

![46f412bf16714f3ca72b578980579c38ad58253c7c5ab254b9e389bc75729433.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/tables/46f412bf16714f3ca72b578980579c38ad58253c7c5ab254b9e389bc75729433.jpg)

![52c56678c42cac91ebdaef9368b4b3b180b3f4ccbaae8b295d83e31ae6648bca.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/tables/52c56678c42cac91ebdaef9368b4b3b180b3f4ccbaae8b295d83e31ae6648bca.jpg)

![646ca07cb5aa6145452d9e78c7fcec86d9600a672660487f4f6896bdc54cfd41.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/tables/646ca07cb5aa6145452d9e78c7fcec86d9600a672660487f4f6896bdc54cfd41.jpg)

![6f1d9136b01a2fe350a8886310e44c7100930a6c550fcc7234bfdb9405e58f8c.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/tables/6f1d9136b01a2fe350a8886310e44c7100930a6c550fcc7234bfdb9405e58f8c.jpg)

![743223b1f6ed9fb97c321b03cb51b33b0678103498e572d775ec76d6b0b8b609.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/tables/743223b1f6ed9fb97c321b03cb51b33b0678103498e572d775ec76d6b0b8b609.jpg)

![7dafcf83f2c044476e2f28aee8fc3a7f32e3313182eddb9ec43a60ffe95c8965.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/tables/7dafcf83f2c044476e2f28aee8fc3a7f32e3313182eddb9ec43a60ffe95c8965.jpg)

![9188e543aa7ac9771a95a0f629b47e3b16fac971046c38baf516bc85f709769f.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/tables/9188e543aa7ac9771a95a0f629b47e3b16fac971046c38baf516bc85f709769f.jpg)

![aa628a1f8ce6d7ff983fd69722a04572d9e76cd3b9f48b9771e144d960bb6c1a.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/tables/aa628a1f8ce6d7ff983fd69722a04572d9e76cd3b9f48b9771e144d960bb6c1a.jpg)

![ae6b682359cc430cfae7fadfc867e8c834af3f30f294d9fae9da2e559369de55.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/tables/ae6b682359cc430cfae7fadfc867e8c834af3f30f294d9fae9da2e559369de55.jpg)

![cd4b48de226ca5155546caa897ee4b19e8b15c0848129409e2f537bcf8607d5d.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/tables/cd4b48de226ca5155546caa897ee4b19e8b15c0848129409e2f537bcf8607d5d.jpg)

![d42d9cad38ef90fcdb7228753aa2db0d3b06e0d802e08ab73be8224401c0361b.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/tables/d42d9cad38ef90fcdb7228753aa2db0d3b06e0d802e08ab73be8224401c0361b.jpg)

![e1739e6b7ff08176a1fdff0dd76094357533037b19e3a88f2fbae1ace0400937.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/tables/e1739e6b7ff08176a1fdff0dd76094357533037b19e3a88f2fbae1ace0400937.jpg)

![e224001680c29ac06338944c960529d79d768d91e4ba93d0142ac38c782e723e.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/tables/e224001680c29ac06338944c960529d79d768d91e4ba93d0142ac38c782e723e.jpg)

![f3c82c54cdd7adbcda3297b7959a53ffa410be07f5b77f31057f9382342e29f4.jpg](../iclr_results/3109_Kolmogorov-Arnold%20Transformer/tables/f3c82c54cdd7adbcda3297b7959a53ffa410be07f5b77f31057f9382342e29f4.jpg)

## Human Simulacra: Benchmarking the Personification of Large Language Models


### Images

![1322e7f704ed02f2396b782109e149a7e1f625340cb1508342b0598d66c3eb3a.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/images/1322e7f704ed02f2396b782109e149a7e1f625340cb1508342b0598d66c3eb3a.jpg)

![22d762da6b5d1f65b1b118aeaf2c05268273cdd9a5a117d8a68c0c73c1527e96.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/images/22d762da6b5d1f65b1b118aeaf2c05268273cdd9a5a117d8a68c0c73c1527e96.jpg)

![294d17e9f33f54e17875f533b25962823c20fb23cfcb3773028c67d20df50ced.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/images/294d17e9f33f54e17875f533b25962823c20fb23cfcb3773028c67d20df50ced.jpg)

![2aaee88295a533a4f89e35ab63e5859b2a908c0422bc123631881e782bf8ba2a.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/images/2aaee88295a533a4f89e35ab63e5859b2a908c0422bc123631881e782bf8ba2a.jpg)

![353ecb30f0a16959ddcf39e78689eaf8f358403ad4995adff83112a49a10cee4.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/images/353ecb30f0a16959ddcf39e78689eaf8f358403ad4995adff83112a49a10cee4.jpg)

![452e02cd877258d2f6392cb5f195b84fcde9f04f79bc043d6f26281000d73b29.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/images/452e02cd877258d2f6392cb5f195b84fcde9f04f79bc043d6f26281000d73b29.jpg)

![4662a35e04743044039c208835455513159cb07fc076d40438186fdae9a785cc.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/images/4662a35e04743044039c208835455513159cb07fc076d40438186fdae9a785cc.jpg)

![522ce712a278a645f8553d377119070a2ac78d4119c1afc591f6e6255ec6b8ea.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/images/522ce712a278a645f8553d377119070a2ac78d4119c1afc591f6e6255ec6b8ea.jpg)

![5a252f369c45ab6bcc6e26c4a4d6986bfb60f4aeaf403721e45f6a2f3c271f0f.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/images/5a252f369c45ab6bcc6e26c4a4d6986bfb60f4aeaf403721e45f6a2f3c271f0f.jpg)

![686f141faf9f506b2b02d87cc2c4e955aefbb2008d83aeefaf4037ea78fee550.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/images/686f141faf9f506b2b02d87cc2c4e955aefbb2008d83aeefaf4037ea78fee550.jpg)

![697748bf3322eed06ae43a880e426561b1d8b5f33668f9d546728f0be4704c38.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/images/697748bf3322eed06ae43a880e426561b1d8b5f33668f9d546728f0be4704c38.jpg)

![6fb0b0028b1fe08a2476146f15d488eea2f723c0bd4cc715079e1367ac539bad.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/images/6fb0b0028b1fe08a2476146f15d488eea2f723c0bd4cc715079e1367ac539bad.jpg)

![732f81d3b61ccd6cb2fdf6b07b879a99579b1890ed4320d1aef938ef53e9023d.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/images/732f81d3b61ccd6cb2fdf6b07b879a99579b1890ed4320d1aef938ef53e9023d.jpg)

![7ffac96d0db5c4c75eae770e4cc71233859fad467952692f5c3777d39464c1ba.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/images/7ffac96d0db5c4c75eae770e4cc71233859fad467952692f5c3777d39464c1ba.jpg)

![88ef095bb87e73d5ae44dfe781c14641e77d8f2d20570bff6ab96ca2a1980147.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/images/88ef095bb87e73d5ae44dfe781c14641e77d8f2d20570bff6ab96ca2a1980147.jpg)

![8b687520608863dba9e7a19c1e189d68437584529d891865eed911069964c84b.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/images/8b687520608863dba9e7a19c1e189d68437584529d891865eed911069964c84b.jpg)

![9cf9552647a616322ce9f8e0a9c7d79346d2f4e90dbbcb5d27cbdf63a7f0ce04.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/images/9cf9552647a616322ce9f8e0a9c7d79346d2f4e90dbbcb5d27cbdf63a7f0ce04.jpg)

![ad7102fdc2d05bf7c7145d217a58d83cdae85e0acda5671e6f72feac547c1db1.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/images/ad7102fdc2d05bf7c7145d217a58d83cdae85e0acda5671e6f72feac547c1db1.jpg)

![b16aa92d2bdfd3242139d281d6c2647b42fb5c0de21de843bd77c3a85a51af8b.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/images/b16aa92d2bdfd3242139d281d6c2647b42fb5c0de21de843bd77c3a85a51af8b.jpg)

![d1488ebb9d005aa88b5b155f2b3c324af0804576d82ac6e78abc317a65e6591c.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/images/d1488ebb9d005aa88b5b155f2b3c324af0804576d82ac6e78abc317a65e6591c.jpg)

![e09b421a85b823d0689cf9a7b8325917ada770e9ab42f9d8d7d80e6e6710a1e8.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/images/e09b421a85b823d0689cf9a7b8325917ada770e9ab42f9d8d7d80e6e6710a1e8.jpg)

![e1af59230068f5adb2d84732df4214fa0b053706113d9ec851f375b5630f2ea3.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/images/e1af59230068f5adb2d84732df4214fa0b053706113d9ec851f375b5630f2ea3.jpg)

### Tables

![1fe23a6191be957d28dd8e6be24c2efd3e46539cb457b2260fa260964a189c21.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/tables/1fe23a6191be957d28dd8e6be24c2efd3e46539cb457b2260fa260964a189c21.jpg)

![22ba1893142d6dcfdfe1730bcdf088339bb04ade3233b17803f9d31a545adb03.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/tables/22ba1893142d6dcfdfe1730bcdf088339bb04ade3233b17803f9d31a545adb03.jpg)

![3603ee2c95d1b0db299e8353004ef0e975cd2e38644bfe06973049693862e620.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/tables/3603ee2c95d1b0db299e8353004ef0e975cd2e38644bfe06973049693862e620.jpg)

![489cde925cf788252dfb35134aceae2b5d0745bf120f2d3e30ee135c026f7264.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/tables/489cde925cf788252dfb35134aceae2b5d0745bf120f2d3e30ee135c026f7264.jpg)

![4aaf21f82c225e9aabde26a2eb3ee3999a83abf2753e1b6679fbd40fa00dba98.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/tables/4aaf21f82c225e9aabde26a2eb3ee3999a83abf2753e1b6679fbd40fa00dba98.jpg)

![4d22ca5362ace587f2c7ec869c486102bea7a557bafd1dd9b360275a09f30a11.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/tables/4d22ca5362ace587f2c7ec869c486102bea7a557bafd1dd9b360275a09f30a11.jpg)

![5b10fe3820fda9d93e3996ba99d43fe4fb9416e7957610bdf5aea22996b371df.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/tables/5b10fe3820fda9d93e3996ba99d43fe4fb9416e7957610bdf5aea22996b371df.jpg)

![6af3489516ff3515ff55e22847bd8d5100803a9ce3963a48acba92b0f524fe3c.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/tables/6af3489516ff3515ff55e22847bd8d5100803a9ce3963a48acba92b0f524fe3c.jpg)

![7eb0579ccd5b651f283f5b14d4e5ab25352e30be517516cdadba7002a7447665.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/tables/7eb0579ccd5b651f283f5b14d4e5ab25352e30be517516cdadba7002a7447665.jpg)

![8e823ed366e42025dca37f24d20bdc7708847e03f3bef47a66606f590d15c815.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/tables/8e823ed366e42025dca37f24d20bdc7708847e03f3bef47a66606f590d15c815.jpg)

![9cbd10eeddc7a13aa6d31ae2fe9f129fe687baeb7c161800a831fd5147656132.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/tables/9cbd10eeddc7a13aa6d31ae2fe9f129fe687baeb7c161800a831fd5147656132.jpg)

![a149aadcf254f83301c83159879e7f272236926dd104b9d6a77135988ba0f987.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/tables/a149aadcf254f83301c83159879e7f272236926dd104b9d6a77135988ba0f987.jpg)

![ac051bf7c2487ff6c427151b78cc6b5851eacc56e59f3d2a821e749a89b05ab2.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/tables/ac051bf7c2487ff6c427151b78cc6b5851eacc56e59f3d2a821e749a89b05ab2.jpg)

![ac5041ae83627db6ee21d2c7e744ea58561c1ba2d5da51b86794470d7fe7dddf.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/tables/ac5041ae83627db6ee21d2c7e744ea58561c1ba2d5da51b86794470d7fe7dddf.jpg)

![b1238b857126fd95a78a3be37b5e45bc1d0e848fd28e030f0b6dc9f1413fbd62.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/tables/b1238b857126fd95a78a3be37b5e45bc1d0e848fd28e030f0b6dc9f1413fbd62.jpg)

![b29fd41185d7b7b74342a02a2eee288cc7c46ac24a6220e4f2edba27c173c818.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/tables/b29fd41185d7b7b74342a02a2eee288cc7c46ac24a6220e4f2edba27c173c818.jpg)

![ca3a87535baeb029cbdc33bba11348849ace7a6f43a92e6ce018892da07e75a3.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/tables/ca3a87535baeb029cbdc33bba11348849ace7a6f43a92e6ce018892da07e75a3.jpg)

![ef09a37744bdc2022abd8b50602841bedfb5a8c684dae04d32f8f551cd17a211.jpg](../iclr_results/3110_Human%20Simulacra_%20Benchmarking%20the%20Personification%20of%20Large%20Language%20Models/tables/ef09a37744bdc2022abd8b50602841bedfb5a8c684dae04d32f8f551cd17a211.jpg)

## Generative Representational Instruction Tuning


### Images

![0b4a683070a924460d82926ddc4f269c16beb508ae1d096e7176fe09a6797f4f.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/images/0b4a683070a924460d82926ddc4f269c16beb508ae1d096e7176fe09a6797f4f.jpg)

![572c443f0f233e0790c441d51f1eaeb10f87fb569add05fefe1f73adf65eb712.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/images/572c443f0f233e0790c441d51f1eaeb10f87fb569add05fefe1f73adf65eb712.jpg)

![5ea75914ad5331486a0d58f61de2f7cb1834ace73ce3c1c2ecb6040b310986bf.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/images/5ea75914ad5331486a0d58f61de2f7cb1834ace73ce3c1c2ecb6040b310986bf.jpg)

![6522e7599288132c24a6d3c8608ea4797e1a09b687bc6172d008278aa8f1c2ec.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/images/6522e7599288132c24a6d3c8608ea4797e1a09b687bc6172d008278aa8f1c2ec.jpg)

![82895d142ad53d3c86794117ce416e0cb77c37bb3652e1cf9abf443121bd9049.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/images/82895d142ad53d3c86794117ce416e0cb77c37bb3652e1cf9abf443121bd9049.jpg)

![85b6dbee409a945e0953b3b1e8110318f3be6afaba69423c4525e05084344cf7.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/images/85b6dbee409a945e0953b3b1e8110318f3be6afaba69423c4525e05084344cf7.jpg)

![86a6e19ccf13327698021da084959593250d9b05b81ae043a7c456c95e73484f.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/images/86a6e19ccf13327698021da084959593250d9b05b81ae043a7c456c95e73484f.jpg)

![9b6e15fbd91c5b06b3d656e96717ea15ba128c3001d7fff5a1caf87a5de47981.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/images/9b6e15fbd91c5b06b3d656e96717ea15ba128c3001d7fff5a1caf87a5de47981.jpg)

![b9199c6fa419e4711458aac16298c8742b45c5843e5084f1aed2811edb9815f0.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/images/b9199c6fa419e4711458aac16298c8742b45c5843e5084f1aed2811edb9815f0.jpg)

![bd6d6034ca219759213b49465cd76581cc43badb63d465abfc587626432126d6.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/images/bd6d6034ca219759213b49465cd76581cc43badb63d465abfc587626432126d6.jpg)

![e5d95b21fe0db0ffab37257413725c9e73bf1f05850077bd0e20559a9122a7cd.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/images/e5d95b21fe0db0ffab37257413725c9e73bf1f05850077bd0e20559a9122a7cd.jpg)

![fe7a7d34d9792173417f355c0f9900b491eda2195d378bf8191f856174eb0a45.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/images/fe7a7d34d9792173417f355c0f9900b491eda2195d378bf8191f856174eb0a45.jpg)

### Tables

![09683e98eb758d26716d933181d4cb72b4ab85285e129cb0a27fe2f12a0bbf54.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/09683e98eb758d26716d933181d4cb72b4ab85285e129cb0a27fe2f12a0bbf54.jpg)

![0e55f8671df7ce4c8871504b32cf52d973401ac52cb9cbb1f4518f396af438ad.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/0e55f8671df7ce4c8871504b32cf52d973401ac52cb9cbb1f4518f396af438ad.jpg)

![1008dc2412c586dcf5d74c0793db9a5930b12c21ad621f48e15619a1e06c2a04.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/1008dc2412c586dcf5d74c0793db9a5930b12c21ad621f48e15619a1e06c2a04.jpg)

![128b831bc4761c7c4aabd520283ea6f52a1b785bfa6f1df72cbac7c2767f251d.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/128b831bc4761c7c4aabd520283ea6f52a1b785bfa6f1df72cbac7c2767f251d.jpg)

![13b51f658564c86b81fc01e28657b96070854068e1e0f84f4ccc35fd1a2cdbac.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/13b51f658564c86b81fc01e28657b96070854068e1e0f84f4ccc35fd1a2cdbac.jpg)

![1b3555ccc97a9a048465a163959398b0227363d7567f4a01ee073465e8ec6ad1.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/1b3555ccc97a9a048465a163959398b0227363d7567f4a01ee073465e8ec6ad1.jpg)

![1f9d62e800436e65cbc05b235e4e0769f85abd952dc8ca8980004b1927a2ad9e.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/1f9d62e800436e65cbc05b235e4e0769f85abd952dc8ca8980004b1927a2ad9e.jpg)

![2092938f9e43c3092124c0121399b3c6646ec55e09118d095fceb8affb5c9438.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/2092938f9e43c3092124c0121399b3c6646ec55e09118d095fceb8affb5c9438.jpg)

![2e8cb7c6b3a0b6a108b91e88ca2c9ff9199a5cd88a40b9d4ca031a831ab8044f.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/2e8cb7c6b3a0b6a108b91e88ca2c9ff9199a5cd88a40b9d4ca031a831ab8044f.jpg)

![2ed8939137ace68bd155190471b041072fe0a5d93f5e0ac5e39ee97062abda4f.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/2ed8939137ace68bd155190471b041072fe0a5d93f5e0ac5e39ee97062abda4f.jpg)

![361f019beb21cbe9298008e5816beee4349e55f975cf173f1f61f4aed1b3258c.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/361f019beb21cbe9298008e5816beee4349e55f975cf173f1f61f4aed1b3258c.jpg)

![3695d5da6ab199252e2c0e1525710472172fc64cc7868167767f309af6709467.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/3695d5da6ab199252e2c0e1525710472172fc64cc7868167767f309af6709467.jpg)

![39126bfd3dd11e781e8a305fa1d49dfecb7757e727a8ae14785a532d2beddd9f.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/39126bfd3dd11e781e8a305fa1d49dfecb7757e727a8ae14785a532d2beddd9f.jpg)

![3920567e284f7abf30c7cfc0319e4c24f253ee488a088a8a2f72511e4bc5b0dc.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/3920567e284f7abf30c7cfc0319e4c24f253ee488a088a8a2f72511e4bc5b0dc.jpg)

![3b1ab3d1ff1e4614e8f72d45504db472c32943e3bb293848796395b477f91aa6.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/3b1ab3d1ff1e4614e8f72d45504db472c32943e3bb293848796395b477f91aa6.jpg)

![3dccc0b5d63c63ef1fb3be16ab8858cb2ae736e0a0e981b18a464812f990a729.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/3dccc0b5d63c63ef1fb3be16ab8858cb2ae736e0a0e981b18a464812f990a729.jpg)

![3f9d3c1cfc58dab7dc890e9527b26dc67f64c34ba3b627917259cf3870305581.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/3f9d3c1cfc58dab7dc890e9527b26dc67f64c34ba3b627917259cf3870305581.jpg)

![457e692caf379e2390c4a0a6b722d421765eab8d8732d049a17b4500e6ddd1cd.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/457e692caf379e2390c4a0a6b722d421765eab8d8732d049a17b4500e6ddd1cd.jpg)

![4e19015570411a8e50dfe84199a6f14015a9784cb87c0abc8d2b4256277cdaa2.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/4e19015570411a8e50dfe84199a6f14015a9784cb87c0abc8d2b4256277cdaa2.jpg)

![4fd78b207ca2f7ceb52e9dd6be75471f409e09dba8eb6dd77cdb3f986e18c462.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/4fd78b207ca2f7ceb52e9dd6be75471f409e09dba8eb6dd77cdb3f986e18c462.jpg)

![536b32d6417a26db8d258a470c944fb6c8cf81db499735caddc5eb49962cc40c.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/536b32d6417a26db8d258a470c944fb6c8cf81db499735caddc5eb49962cc40c.jpg)

![55578f8575e5a4985ce08d5c8405f8a3415f36dc6a5ff9d8ec6a05afdd7b1790.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/55578f8575e5a4985ce08d5c8405f8a3415f36dc6a5ff9d8ec6a05afdd7b1790.jpg)

![5a2ed89d2aec6d81ee4ef1f6a30ab7a3e055eac97318e2237fdddf4aba0c966d.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/5a2ed89d2aec6d81ee4ef1f6a30ab7a3e055eac97318e2237fdddf4aba0c966d.jpg)

![5b5785b8066c68a87e736d2a89ae666e63502c31bc04c28391648349583628bc.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/5b5785b8066c68a87e736d2a89ae666e63502c31bc04c28391648349583628bc.jpg)

![6716d66cb5582fd123ab90cfc58e7c3831507dc57f1dc5ebcb766bb49636648a.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/6716d66cb5582fd123ab90cfc58e7c3831507dc57f1dc5ebcb766bb49636648a.jpg)

![6cf33921fe74ce215ce1d1984a68130b6288332a7da47ee168720119fbe7703c.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/6cf33921fe74ce215ce1d1984a68130b6288332a7da47ee168720119fbe7703c.jpg)

![6e3cbbb603c15c4a5332778ef176438ee774c2afbbd49b188f6b449a7f33ed02.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/6e3cbbb603c15c4a5332778ef176438ee774c2afbbd49b188f6b449a7f33ed02.jpg)

![79978ddd0c0ee6a3998fba21bc8295d781de51d43eb99cd3b2193344999a6d78.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/79978ddd0c0ee6a3998fba21bc8295d781de51d43eb99cd3b2193344999a6d78.jpg)

![840590673b01339b466618c6e6a4756511fbc873cb94e91dbd42623340507eae.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/840590673b01339b466618c6e6a4756511fbc873cb94e91dbd42623340507eae.jpg)

![87d2a0acd00d977d973d166e1fc18bed3f2219363aadeb5550be1f973a06fad2.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/87d2a0acd00d977d973d166e1fc18bed3f2219363aadeb5550be1f973a06fad2.jpg)

![90a996948c1b2af33c9410672887c1b3159dae528f99bccb8d9d58958d96ec3c.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/90a996948c1b2af33c9410672887c1b3159dae528f99bccb8d9d58958d96ec3c.jpg)

![93069adf1246d2c5bf8c98b69725c20db9c4638ccbda423433d6af80d164dc67.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/93069adf1246d2c5bf8c98b69725c20db9c4638ccbda423433d6af80d164dc67.jpg)

![978f2f9e2f30a87a9ca3745d37912e275da101edc7a1d5803b60c67cb97c9d3c.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/978f2f9e2f30a87a9ca3745d37912e275da101edc7a1d5803b60c67cb97c9d3c.jpg)

![98d48a15df940ffaec9bd722524e58cc6de194acff53596e724f2267268d9bd3.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/98d48a15df940ffaec9bd722524e58cc6de194acff53596e724f2267268d9bd3.jpg)

![9b78f04f67fc9549224bf9a6f81bad79936094bcdb907eb86edee2f00390e088.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/9b78f04f67fc9549224bf9a6f81bad79936094bcdb907eb86edee2f00390e088.jpg)

![9b893c962f74edb4d2c1e0fe7305b07dc27fa740f7fca6f0cbcf4824352011b0.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/9b893c962f74edb4d2c1e0fe7305b07dc27fa740f7fca6f0cbcf4824352011b0.jpg)

![ab8fa5e6b3dcf0df2ac60dea32edebfbe035f4bdec8c325cb9ed5fc46a393199.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/ab8fa5e6b3dcf0df2ac60dea32edebfbe035f4bdec8c325cb9ed5fc46a393199.jpg)

![b4053330fe4c4cd2d44a34ca294f677ffbfb0e202fd87d28243410daa57f3214.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/b4053330fe4c4cd2d44a34ca294f677ffbfb0e202fd87d28243410daa57f3214.jpg)

![bf7f9870adf34af0e65d95df8647045421c89e2c578a6d36fe4da9987149ebdf.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/bf7f9870adf34af0e65d95df8647045421c89e2c578a6d36fe4da9987149ebdf.jpg)

![bfe524dccf5192cc95e0cd01d102f28170622361043ac46db45d4762369b7aa6.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/bfe524dccf5192cc95e0cd01d102f28170622361043ac46db45d4762369b7aa6.jpg)

![c09f7df300742ab353f1c1e12a08e07940986931a8eac88a8ecef7926872d47d.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/c09f7df300742ab353f1c1e12a08e07940986931a8eac88a8ecef7926872d47d.jpg)

![ca3c304670cde3385c755f006529efd7d4e62f5af81ab35433d181dcf2ff49fc.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/ca3c304670cde3385c755f006529efd7d4e62f5af81ab35433d181dcf2ff49fc.jpg)

![d72bb9a99a331b53d985e60f786548c21369e58c045ac45288f69b419715778d.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/d72bb9a99a331b53d985e60f786548c21369e58c045ac45288f69b419715778d.jpg)

![d7581d788a49e156fc0f74ad597c85cf43f3401803a7b14b1edbe773719d0600.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/d7581d788a49e156fc0f74ad597c85cf43f3401803a7b14b1edbe773719d0600.jpg)

![db2a669ba46dc1a7864b741ba76f0d3d31d08781ca3d98f8553cccd65bab9c62.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/db2a669ba46dc1a7864b741ba76f0d3d31d08781ca3d98f8553cccd65bab9c62.jpg)

![dd4d6e17d0801c7925451adb8e858a7634bf3e5ff615090bbf041eedcdca5e3d.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/dd4d6e17d0801c7925451adb8e858a7634bf3e5ff615090bbf041eedcdca5e3d.jpg)

![dfea30f9b6b263245ebd260ea67e1e141cf61b9fb47ebc1bfb6502eac995e9f4.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/dfea30f9b6b263245ebd260ea67e1e141cf61b9fb47ebc1bfb6502eac995e9f4.jpg)

![e0a6bd15b0682835ba7617cbbac560cd4a4fc63b2093c63bd47be8e7c9b58023.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/e0a6bd15b0682835ba7617cbbac560cd4a4fc63b2093c63bd47be8e7c9b58023.jpg)

![e4b9489cc066602ffe2366a306814898c335af6994c5c89ad2190d68742dd21a.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/e4b9489cc066602ffe2366a306814898c335af6994c5c89ad2190d68742dd21a.jpg)

![e5917ce9ac934fade865a5e4940e06486aedf25b43f705e12164e2112088c23f.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/e5917ce9ac934fade865a5e4940e06486aedf25b43f705e12164e2112088c23f.jpg)

![e74ef50b0e23696d8dd13af17233653b21bda03821a08b49da96df8f42ed90fd.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/e74ef50b0e23696d8dd13af17233653b21bda03821a08b49da96df8f42ed90fd.jpg)

![f2fda9d538186a084fd5139b02fb4494150d5bce02c915fe4b3eb5af7edb87d8.jpg](../iclr_results/3111_Generative%20Representational%20Instruction%20Tuning/tables/f2fda9d538186a084fd5139b02fb4494150d5bce02c915fe4b3eb5af7edb87d8.jpg)

## ROUTE: Robust Multitask Tuning and Collaboration for Text-to-SQL


### Images

![16aae4a5e735673bd1f5ff614ca200ab72a79f68339f35b5309111706cc2fed2.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/images/16aae4a5e735673bd1f5ff614ca200ab72a79f68339f35b5309111706cc2fed2.jpg)

![18c84df91d03eeb15ca02e94bfced9e78f3794f2fa5801c84bbf416d321f9fe9.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/images/18c84df91d03eeb15ca02e94bfced9e78f3794f2fa5801c84bbf416d321f9fe9.jpg)

![5334b8a668cc0daaabda8dd1dfabda121c312126abc56120c6e8a32147bb2c7e.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/images/5334b8a668cc0daaabda8dd1dfabda121c312126abc56120c6e8a32147bb2c7e.jpg)

![5b8c0874a6e16d9edacfd3aefb55b16d401188c702e399b5c939fb0770df7fdf.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/images/5b8c0874a6e16d9edacfd3aefb55b16d401188c702e399b5c939fb0770df7fdf.jpg)

![762a3a204454f759bbebfbea1d5bd027af76338add7c5fcdcc2d49b366407ffc.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/images/762a3a204454f759bbebfbea1d5bd027af76338add7c5fcdcc2d49b366407ffc.jpg)

![78330fe544ba6528a56d70e9fdfd73583a92a208f7269bea74c2cd4acc4ec40f.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/images/78330fe544ba6528a56d70e9fdfd73583a92a208f7269bea74c2cd4acc4ec40f.jpg)

![825a02a3f5d446dd1df3ec4dc29d09d91b1222f7805390ef38493a31bb062707.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/images/825a02a3f5d446dd1df3ec4dc29d09d91b1222f7805390ef38493a31bb062707.jpg)

![dd0066dc7b6d77ee5ac301fc50a6f4ae7875fa67ff444369cfa2a36e658512e5.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/images/dd0066dc7b6d77ee5ac301fc50a6f4ae7875fa67ff444369cfa2a36e658512e5.jpg)

![f42c2413134c5c6def2a8f7cc7b9aba8b5b59a897db39e2e74277af0eeec2f61.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/images/f42c2413134c5c6def2a8f7cc7b9aba8b5b59a897db39e2e74277af0eeec2f61.jpg)

![f457e34ad91af135abd591d6eac008e64659e12b6fd5d9f9158b71a546cfe0c6.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/images/f457e34ad91af135abd591d6eac008e64659e12b6fd5d9f9158b71a546cfe0c6.jpg)

![fdfe9859dbab632704b055a788a01022cc641a3537dc4e0e88aa7b4a07c8447e.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/images/fdfe9859dbab632704b055a788a01022cc641a3537dc4e0e88aa7b4a07c8447e.jpg)

### Tables

![052974cc42ff080d8c1f68d688264a22618dc2d6fc61029d4482aee8ff778992.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/tables/052974cc42ff080d8c1f68d688264a22618dc2d6fc61029d4482aee8ff778992.jpg)

![25443cf9a189d579e0dad3d5e16598bab4bebec3ccd243a00b9a1aa6e21b1fa5.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/tables/25443cf9a189d579e0dad3d5e16598bab4bebec3ccd243a00b9a1aa6e21b1fa5.jpg)

![31ccd6c4e0528ce83d703abbd76b84008f3d04c468a7bb5acb96f6c654a1dff4.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/tables/31ccd6c4e0528ce83d703abbd76b84008f3d04c468a7bb5acb96f6c654a1dff4.jpg)

![3b73ecc2960744f49fae69d6564ccfbf85bbdac306fe63e60d3d1ea11cb114e0.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/tables/3b73ecc2960744f49fae69d6564ccfbf85bbdac306fe63e60d3d1ea11cb114e0.jpg)

![47e29749f81db692fa29c7a9325eff4bcb22598311a851795b3319a321c6bad5.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/tables/47e29749f81db692fa29c7a9325eff4bcb22598311a851795b3319a321c6bad5.jpg)

![5350722efd6993594194ad54b19894b394891d7f5abd39ad1566a852570cfbc3.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/tables/5350722efd6993594194ad54b19894b394891d7f5abd39ad1566a852570cfbc3.jpg)

![5455f7ff8d16476418b99449fdeaa935e74b42b256b9fe6e96d65fd3a9501e2b.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/tables/5455f7ff8d16476418b99449fdeaa935e74b42b256b9fe6e96d65fd3a9501e2b.jpg)

![54bc0c081a0ccf8ab6b495c8b69c19cf0fd277e4ff1fe76ac0311cb75e998ba4.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/tables/54bc0c081a0ccf8ab6b495c8b69c19cf0fd277e4ff1fe76ac0311cb75e998ba4.jpg)

![6b68280551d4ed680493315aa0f320c8e09d50511e2116b1209149a5b1f23dc8.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/tables/6b68280551d4ed680493315aa0f320c8e09d50511e2116b1209149a5b1f23dc8.jpg)

![75a9ee4f6fd2641ecb9e1c075430c16ac39d9f548a89b77291b8eda88e0ffcd3.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/tables/75a9ee4f6fd2641ecb9e1c075430c16ac39d9f548a89b77291b8eda88e0ffcd3.jpg)

![7ef3f53472fd6ab9717a3a45ca981df2c402fa7ade1d48b72b96f896e56461b6.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/tables/7ef3f53472fd6ab9717a3a45ca981df2c402fa7ade1d48b72b96f896e56461b6.jpg)

![91a4bcdc2fe0eb01cc6cdb715093e9275578221f9ccfea4da2feddbccfce8e4c.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/tables/91a4bcdc2fe0eb01cc6cdb715093e9275578221f9ccfea4da2feddbccfce8e4c.jpg)

![a566af3c3642576cdf22e31ed4f5b9a804f6ae2dbca6d460a2cb6a8d877880b8.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/tables/a566af3c3642576cdf22e31ed4f5b9a804f6ae2dbca6d460a2cb6a8d877880b8.jpg)

![bf50a800df4c8680e7ba3b3663bfe24537a3a69e2b7ac9db1ee2221f3b72b3d7.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/tables/bf50a800df4c8680e7ba3b3663bfe24537a3a69e2b7ac9db1ee2221f3b72b3d7.jpg)

![efa71979f8325b88e9267290ae30799dac429c2f3a73a3cdb4062052989d42ba.jpg](../iclr_results/3112_ROUTE_%20Robust%20Multitask%20Tuning%20and%20Collaboration%20for%20Text-to-SQL/tables/efa71979f8325b88e9267290ae30799dac429c2f3a73a3cdb4062052989d42ba.jpg)

## MuHBoost: Multi-Label Boosting For Practical Longitudinal Human Behavior Modeling


### Images

![30da96a9bc1a87a5d8adebe3bff4fda6176011b7d15795464c492d872c318f8c.jpg](../iclr_results/3113_MuHBoost_%20Multi-Label%20Boosting%20For%20Practical%20Longitudinal%20Human%20Behavior%20Modeling/images/30da96a9bc1a87a5d8adebe3bff4fda6176011b7d15795464c492d872c318f8c.jpg)

![5ee99645bc593d4366ccf02a9f766a1d4b208d5c2cbbb988517d0b27bd436808.jpg](../iclr_results/3113_MuHBoost_%20Multi-Label%20Boosting%20For%20Practical%20Longitudinal%20Human%20Behavior%20Modeling/images/5ee99645bc593d4366ccf02a9f766a1d4b208d5c2cbbb988517d0b27bd436808.jpg)

![60fc6c0f14bfad4abc9bc0d0452ad5f89f14be5a44f4189a86c29d3f33448a64.jpg](../iclr_results/3113_MuHBoost_%20Multi-Label%20Boosting%20For%20Practical%20Longitudinal%20Human%20Behavior%20Modeling/images/60fc6c0f14bfad4abc9bc0d0452ad5f89f14be5a44f4189a86c29d3f33448a64.jpg)

![dbb934d3b6b9b50a5477b4fe04ac2cd3c6021317e540ee4e5d4f2cb3b48915d9.jpg](../iclr_results/3113_MuHBoost_%20Multi-Label%20Boosting%20For%20Practical%20Longitudinal%20Human%20Behavior%20Modeling/images/dbb934d3b6b9b50a5477b4fe04ac2cd3c6021317e540ee4e5d4f2cb3b48915d9.jpg)

![e0cf0ec045d8e4664afece35ce6f187e72e07fef4a33d6dfbfd201775c1e5797.jpg](../iclr_results/3113_MuHBoost_%20Multi-Label%20Boosting%20For%20Practical%20Longitudinal%20Human%20Behavior%20Modeling/images/e0cf0ec045d8e4664afece35ce6f187e72e07fef4a33d6dfbfd201775c1e5797.jpg)

### Tables

![1979ca34dc8f7f14f0798c4557a021582a2e969fec0fd79b0b8e0f81357804a7.jpg](../iclr_results/3113_MuHBoost_%20Multi-Label%20Boosting%20For%20Practical%20Longitudinal%20Human%20Behavior%20Modeling/tables/1979ca34dc8f7f14f0798c4557a021582a2e969fec0fd79b0b8e0f81357804a7.jpg)

![23040384b809ea0fff6c14a17a893d375f12ddefe4accf4ddc3ee8c1371666c1.jpg](../iclr_results/3113_MuHBoost_%20Multi-Label%20Boosting%20For%20Practical%20Longitudinal%20Human%20Behavior%20Modeling/tables/23040384b809ea0fff6c14a17a893d375f12ddefe4accf4ddc3ee8c1371666c1.jpg)

![4178fe0d456b6885aee503e0998d79d1da2da71c8d5051b9025e2222777fa3e9.jpg](../iclr_results/3113_MuHBoost_%20Multi-Label%20Boosting%20For%20Practical%20Longitudinal%20Human%20Behavior%20Modeling/tables/4178fe0d456b6885aee503e0998d79d1da2da71c8d5051b9025e2222777fa3e9.jpg)

![4a202e2c34326df84b6562f7766e3bcbb2af85c039bfc90a9210264ed431989c.jpg](../iclr_results/3113_MuHBoost_%20Multi-Label%20Boosting%20For%20Practical%20Longitudinal%20Human%20Behavior%20Modeling/tables/4a202e2c34326df84b6562f7766e3bcbb2af85c039bfc90a9210264ed431989c.jpg)

![5bc8ef406683eeb4333028fa0cc1f310558d1b0226c6b879d0f3364709ef2799.jpg](../iclr_results/3113_MuHBoost_%20Multi-Label%20Boosting%20For%20Practical%20Longitudinal%20Human%20Behavior%20Modeling/tables/5bc8ef406683eeb4333028fa0cc1f310558d1b0226c6b879d0f3364709ef2799.jpg)

![6947dd59490730ec5224d75a94f67902aa579ba4f1096a2f04b58aa86d1e16fc.jpg](../iclr_results/3113_MuHBoost_%20Multi-Label%20Boosting%20For%20Practical%20Longitudinal%20Human%20Behavior%20Modeling/tables/6947dd59490730ec5224d75a94f67902aa579ba4f1096a2f04b58aa86d1e16fc.jpg)

![7219a706bb27e9c660f48b2f7f9511c3b2ce964a02554513b7c489c646b17863.jpg](../iclr_results/3113_MuHBoost_%20Multi-Label%20Boosting%20For%20Practical%20Longitudinal%20Human%20Behavior%20Modeling/tables/7219a706bb27e9c660f48b2f7f9511c3b2ce964a02554513b7c489c646b17863.jpg)

![769e3d674767430c3d5abe7e09751d73277bb9877c96be35db382b12778be71a.jpg](../iclr_results/3113_MuHBoost_%20Multi-Label%20Boosting%20For%20Practical%20Longitudinal%20Human%20Behavior%20Modeling/tables/769e3d674767430c3d5abe7e09751d73277bb9877c96be35db382b12778be71a.jpg)

![923512c47ad3805327559ec05b88528660bb0d7ea4d85719a4877dfb91bc4d08.jpg](../iclr_results/3113_MuHBoost_%20Multi-Label%20Boosting%20For%20Practical%20Longitudinal%20Human%20Behavior%20Modeling/tables/923512c47ad3805327559ec05b88528660bb0d7ea4d85719a4877dfb91bc4d08.jpg)

![b2136d5f0fecd5be5bfd4e8869fbb9460a16ffba2af76dd7db57bb71d40b8de7.jpg](../iclr_results/3113_MuHBoost_%20Multi-Label%20Boosting%20For%20Practical%20Longitudinal%20Human%20Behavior%20Modeling/tables/b2136d5f0fecd5be5bfd4e8869fbb9460a16ffba2af76dd7db57bb71d40b8de7.jpg)

![cb2566c4f6402de8db3720e27ed6dabfea5b814f28f343c0fc686894b88f02c2.jpg](../iclr_results/3113_MuHBoost_%20Multi-Label%20Boosting%20For%20Practical%20Longitudinal%20Human%20Behavior%20Modeling/tables/cb2566c4f6402de8db3720e27ed6dabfea5b814f28f343c0fc686894b88f02c2.jpg)

![f2586bebc76e4c291bdb063a64efa638addf0217adda7abc176fe668e5aafbf6.jpg](../iclr_results/3113_MuHBoost_%20Multi-Label%20Boosting%20For%20Practical%20Longitudinal%20Human%20Behavior%20Modeling/tables/f2586bebc76e4c291bdb063a64efa638addf0217adda7abc176fe668e5aafbf6.jpg)

## Linear Partial Gromov-Wasserstein Embedding


### Images

![4822c4e8ee6df7e09845cf359f9233dd0c1eec2cc8bc3ad1d464eb6019a2914b.jpg](../iclr_results/3114_Linear%20Partial%20Gromov-Wasserstein%20Embedding/images/4822c4e8ee6df7e09845cf359f9233dd0c1eec2cc8bc3ad1d464eb6019a2914b.jpg)

![5892ceb6979df83257573419472148d119f8fa9599f74893df97d12353924b9b.jpg](../iclr_results/3114_Linear%20Partial%20Gromov-Wasserstein%20Embedding/images/5892ceb6979df83257573419472148d119f8fa9599f74893df97d12353924b9b.jpg)

![5fead484ed1bbdc68f18f8be0b586b963fc10a6187a67cb232fce88afd05d32f.jpg](../iclr_results/3114_Linear%20Partial%20Gromov-Wasserstein%20Embedding/images/5fead484ed1bbdc68f18f8be0b586b963fc10a6187a67cb232fce88afd05d32f.jpg)

![83e0e3534b72afbcd0ce4d7876fab175b31544f0040a71e3f3b9b5d9525c992b.jpg](../iclr_results/3114_Linear%20Partial%20Gromov-Wasserstein%20Embedding/images/83e0e3534b72afbcd0ce4d7876fab175b31544f0040a71e3f3b9b5d9525c992b.jpg)

![8d2e875513158b585842b67d57a1f39905c746bda35a54df78e8dd4364014b9d.jpg](../iclr_results/3114_Linear%20Partial%20Gromov-Wasserstein%20Embedding/images/8d2e875513158b585842b67d57a1f39905c746bda35a54df78e8dd4364014b9d.jpg)

![a82b2ff6091caf48ca1ce005d82579b21d01f7e9b56874284862c3e8cf676e1d.jpg](../iclr_results/3114_Linear%20Partial%20Gromov-Wasserstein%20Embedding/images/a82b2ff6091caf48ca1ce005d82579b21d01f7e9b56874284862c3e8cf676e1d.jpg)

![cb8ef686f770a042f79421614cf575217757646b40a3a73c332a250fda7eeb99.jpg](../iclr_results/3114_Linear%20Partial%20Gromov-Wasserstein%20Embedding/images/cb8ef686f770a042f79421614cf575217757646b40a3a73c332a250fda7eeb99.jpg)

![d370ac804c830d66dc36fa01006a974cc9af7691b566c8b12da4a300f26c687b.jpg](../iclr_results/3114_Linear%20Partial%20Gromov-Wasserstein%20Embedding/images/d370ac804c830d66dc36fa01006a974cc9af7691b566c8b12da4a300f26c687b.jpg)

![d921ce5fd8292e7b11d15d81e456fbb9a16e3aec9f1201a6681d2ba3082cfc59.jpg](../iclr_results/3114_Linear%20Partial%20Gromov-Wasserstein%20Embedding/images/d921ce5fd8292e7b11d15d81e456fbb9a16e3aec9f1201a6681d2ba3082cfc59.jpg)

### Tables

![2722e5a3a770453bca8ec9331e8ba8f28217683a65a524b984908b81df81ecb1.jpg](../iclr_results/3114_Linear%20Partial%20Gromov-Wasserstein%20Embedding/tables/2722e5a3a770453bca8ec9331e8ba8f28217683a65a524b984908b81df81ecb1.jpg)

![3b6c4bd64630e7f71a0de15fa9dbda4a34a6d7ff983d8711047859fc7abbff26.jpg](../iclr_results/3114_Linear%20Partial%20Gromov-Wasserstein%20Embedding/tables/3b6c4bd64630e7f71a0de15fa9dbda4a34a6d7ff983d8711047859fc7abbff26.jpg)

![4e99b6f8e0572a93ed071155b3bad5307334578b4f78d9e8b284733764ab948a.jpg](../iclr_results/3114_Linear%20Partial%20Gromov-Wasserstein%20Embedding/tables/4e99b6f8e0572a93ed071155b3bad5307334578b4f78d9e8b284733764ab948a.jpg)

![99d4658ff5f510466d0fcf0b65bffaac5a3d2ebb1b929f33e339fa7e4122417f.jpg](../iclr_results/3114_Linear%20Partial%20Gromov-Wasserstein%20Embedding/tables/99d4658ff5f510466d0fcf0b65bffaac5a3d2ebb1b929f33e339fa7e4122417f.jpg)

![a8e226fa8aa2a27356dccb5816da18abb58e8d3d41e60c5f0f37d893201f0832.jpg](../iclr_results/3114_Linear%20Partial%20Gromov-Wasserstein%20Embedding/tables/a8e226fa8aa2a27356dccb5816da18abb58e8d3d41e60c5f0f37d893201f0832.jpg)

![ec468ee9405a839eef30f8c1877e8bc5b65688faf85f59d080cb299fbb251dea.jpg](../iclr_results/3114_Linear%20Partial%20Gromov-Wasserstein%20Embedding/tables/ec468ee9405a839eef30f8c1877e8bc5b65688faf85f59d080cb299fbb251dea.jpg)

## PRISM: Privacy-Preserving Improved Stochastic Masking for Federated Generative Models


### Images

![03baa9b0262660d0c6cf081777c223a94dc2e27f46fc4b38e89e895e5a2df9bc.jpg](../iclr_results/3115_PRISM_%20Privacy-Preserving%20Improved%20Stochastic%20Masking%20for%20Federated%20Generative%20Models/images/03baa9b0262660d0c6cf081777c223a94dc2e27f46fc4b38e89e895e5a2df9bc.jpg)

![13e412ccf3c712463dfe138c3b04a84de48839f4147da74cfffde3b00a2177e1.jpg](../iclr_results/3115_PRISM_%20Privacy-Preserving%20Improved%20Stochastic%20Masking%20for%20Federated%20Generative%20Models/images/13e412ccf3c712463dfe138c3b04a84de48839f4147da74cfffde3b00a2177e1.jpg)

![1a52b204de0ad84fbd25f8f71d33a24b256a8ec4a9e5958d8caf504c383569cb.jpg](../iclr_results/3115_PRISM_%20Privacy-Preserving%20Improved%20Stochastic%20Masking%20for%20Federated%20Generative%20Models/images/1a52b204de0ad84fbd25f8f71d33a24b256a8ec4a9e5958d8caf504c383569cb.jpg)

![1bcfbd48e0e6c0e4ed7102819753a68f01d1084bcb158e119be6ec0173b01a76.jpg](../iclr_results/3115_PRISM_%20Privacy-Preserving%20Improved%20Stochastic%20Masking%20for%20Federated%20Generative%20Models/images/1bcfbd48e0e6c0e4ed7102819753a68f01d1084bcb158e119be6ec0173b01a76.jpg)

![2365ae991e66bee9a178094dbb4cf35b8879ee343cef19e426d17e7785f0a556.jpg](../iclr_results/3115_PRISM_%20Privacy-Preserving%20Improved%20Stochastic%20Masking%20for%20Federated%20Generative%20Models/images/2365ae991e66bee9a178094dbb4cf35b8879ee343cef19e426d17e7785f0a556.jpg)

![9affdc2b492e3f8873362eb8a7a2e0a977a4b485ec0bc2b09c1a8f65aa11350f.jpg](../iclr_results/3115_PRISM_%20Privacy-Preserving%20Improved%20Stochastic%20Masking%20for%20Federated%20Generative%20Models/images/9affdc2b492e3f8873362eb8a7a2e0a977a4b485ec0bc2b09c1a8f65aa11350f.jpg)

![a1e01df1141787eecdcd56702f602d7073071c3b3ae80e51d49cdfc28c14f382.jpg](../iclr_results/3115_PRISM_%20Privacy-Preserving%20Improved%20Stochastic%20Masking%20for%20Federated%20Generative%20Models/images/a1e01df1141787eecdcd56702f602d7073071c3b3ae80e51d49cdfc28c14f382.jpg)

![c4c3c250851e6d30e9399e3bb2f289c2e1161bb06c536d347b67e929d611bb76.jpg](../iclr_results/3115_PRISM_%20Privacy-Preserving%20Improved%20Stochastic%20Masking%20for%20Federated%20Generative%20Models/images/c4c3c250851e6d30e9399e3bb2f289c2e1161bb06c536d347b67e929d611bb76.jpg)

![d3d43d2626afaec534618e5d0ebdee12af0f06ce4edb198154e42ff70fabb992.jpg](../iclr_results/3115_PRISM_%20Privacy-Preserving%20Improved%20Stochastic%20Masking%20for%20Federated%20Generative%20Models/images/d3d43d2626afaec534618e5d0ebdee12af0f06ce4edb198154e42ff70fabb992.jpg)

![ef8d8fff70c7c010a95d54c14cc78a2ce46982d78ccce5b921e404b03a27f207.jpg](../iclr_results/3115_PRISM_%20Privacy-Preserving%20Improved%20Stochastic%20Masking%20for%20Federated%20Generative%20Models/images/ef8d8fff70c7c010a95d54c14cc78a2ce46982d78ccce5b921e404b03a27f207.jpg)

### Tables

![1b9eb7160017c99d17689df89914beea09d9021603f400e1f986a08257e13e43.jpg](../iclr_results/3115_PRISM_%20Privacy-Preserving%20Improved%20Stochastic%20Masking%20for%20Federated%20Generative%20Models/tables/1b9eb7160017c99d17689df89914beea09d9021603f400e1f986a08257e13e43.jpg)

![260db7cc4e6b482c0ed8b8903ecd9632ea0108c2d3ae9c16f49e01715b61459e.jpg](../iclr_results/3115_PRISM_%20Privacy-Preserving%20Improved%20Stochastic%20Masking%20for%20Federated%20Generative%20Models/tables/260db7cc4e6b482c0ed8b8903ecd9632ea0108c2d3ae9c16f49e01715b61459e.jpg)

![44c0670b89fd6418c8057520560493c707882e9870b11ac824a3907652d76d9f.jpg](../iclr_results/3115_PRISM_%20Privacy-Preserving%20Improved%20Stochastic%20Masking%20for%20Federated%20Generative%20Models/tables/44c0670b89fd6418c8057520560493c707882e9870b11ac824a3907652d76d9f.jpg)

![555f5be1137e06b43e4e5bb9e862314d30c74c2d30cc14cc1626d86b2304960a.jpg](../iclr_results/3115_PRISM_%20Privacy-Preserving%20Improved%20Stochastic%20Masking%20for%20Federated%20Generative%20Models/tables/555f5be1137e06b43e4e5bb9e862314d30c74c2d30cc14cc1626d86b2304960a.jpg)

![7f5db5c66e0da970b08bc54094093d0d11be90e519eb65944e1c286107a2e2c5.jpg](../iclr_results/3115_PRISM_%20Privacy-Preserving%20Improved%20Stochastic%20Masking%20for%20Federated%20Generative%20Models/tables/7f5db5c66e0da970b08bc54094093d0d11be90e519eb65944e1c286107a2e2c5.jpg)

![a5413ed76cf2cd0288e9fbc3b640cb4456ce000a1a2c13feae1d8a9d0b32eafc.jpg](../iclr_results/3115_PRISM_%20Privacy-Preserving%20Improved%20Stochastic%20Masking%20for%20Federated%20Generative%20Models/tables/a5413ed76cf2cd0288e9fbc3b640cb4456ce000a1a2c13feae1d8a9d0b32eafc.jpg)

![b02c8f5f0d80669b78e40b79375f771a5744bcf270e0222267520d3912ea38a6.jpg](../iclr_results/3115_PRISM_%20Privacy-Preserving%20Improved%20Stochastic%20Masking%20for%20Federated%20Generative%20Models/tables/b02c8f5f0d80669b78e40b79375f771a5744bcf270e0222267520d3912ea38a6.jpg)

![b1bb9784738079ae843a481d1d9341ee8580ad4ce300f93e9910704e19fe5bf3.jpg](../iclr_results/3115_PRISM_%20Privacy-Preserving%20Improved%20Stochastic%20Masking%20for%20Federated%20Generative%20Models/tables/b1bb9784738079ae843a481d1d9341ee8580ad4ce300f93e9910704e19fe5bf3.jpg)

![c92144cb90c241007c1f1055f5381f5071952c01c8ab31c48f7b508e8a785453.jpg](../iclr_results/3115_PRISM_%20Privacy-Preserving%20Improved%20Stochastic%20Masking%20for%20Federated%20Generative%20Models/tables/c92144cb90c241007c1f1055f5381f5071952c01c8ab31c48f7b508e8a785453.jpg)

![d058e586577533303e38410fe11e50a2f6b4c18c81cfd34501ccb56164c512a7.jpg](../iclr_results/3115_PRISM_%20Privacy-Preserving%20Improved%20Stochastic%20Masking%20for%20Federated%20Generative%20Models/tables/d058e586577533303e38410fe11e50a2f6b4c18c81cfd34501ccb56164c512a7.jpg)

![e158aa591b987219f18f18550fd81055a975871f510b473f32897ae53678925d.jpg](../iclr_results/3115_PRISM_%20Privacy-Preserving%20Improved%20Stochastic%20Masking%20for%20Federated%20Generative%20Models/tables/e158aa591b987219f18f18550fd81055a975871f510b473f32897ae53678925d.jpg)

![e8deca22da363eb9f84bcfd1bf2e4f26d17492367af8b7d30e3412857e1a9aa6.jpg](../iclr_results/3115_PRISM_%20Privacy-Preserving%20Improved%20Stochastic%20Masking%20for%20Federated%20Generative%20Models/tables/e8deca22da363eb9f84bcfd1bf2e4f26d17492367af8b7d30e3412857e1a9aa6.jpg)

![ef6d63da7d85c568f18eefadf9e6f38c8066e9d4246b64b6937b7f0ae58c82cd.jpg](../iclr_results/3115_PRISM_%20Privacy-Preserving%20Improved%20Stochastic%20Masking%20for%20Federated%20Generative%20Models/tables/ef6d63da7d85c568f18eefadf9e6f38c8066e9d4246b64b6937b7f0ae58c82cd.jpg)

## Hot-pluggable Federated Learning: Bridging General and Personalized FL via Dynamic Selection


### Images

![4afdbbd6eb3c351ada270bbc7419c5aa4e94a994c34793d013e7cb02cfa0ba84.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/images/4afdbbd6eb3c351ada270bbc7419c5aa4e94a994c34793d013e7cb02cfa0ba84.jpg)

![539c0fd321a27ccca2cb553cb6bb4aa54681a9792fdc735768f805e77c864815.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/images/539c0fd321a27ccca2cb553cb6bb4aa54681a9792fdc735768f805e77c864815.jpg)

![7d45ec88265d6fc51a7c61fbccafe18fb7b6a23ea26147d7ac35eab02510ea20.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/images/7d45ec88265d6fc51a7c61fbccafe18fb7b6a23ea26147d7ac35eab02510ea20.jpg)

![8e02baf9e9493a1d066938b5380ecc048707a2c8da3b1219f7459f77c71d3b56.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/images/8e02baf9e9493a1d066938b5380ecc048707a2c8da3b1219f7459f77c71d3b56.jpg)

![977a8e4aa1601947a5a4193e033d9d9312bfb4d31080224af0a817c2d83ad594.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/images/977a8e4aa1601947a5a4193e033d9d9312bfb4d31080224af0a817c2d83ad594.jpg)

![aa05aeeead079a71c91e7707e64edf234ecda01e9129a505bf52546623dcba64.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/images/aa05aeeead079a71c91e7707e64edf234ecda01e9129a505bf52546623dcba64.jpg)

![b1498933724af640a8329250b14ce42fe200a9aaf26c0a2537679f45cdf4e828.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/images/b1498933724af640a8329250b14ce42fe200a9aaf26c0a2537679f45cdf4e828.jpg)

![b65e5aec92a114f3eaf864e4f8f7f242750755e0630b6411e89adcb6073a26e6.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/images/b65e5aec92a114f3eaf864e4f8f7f242750755e0630b6411e89adcb6073a26e6.jpg)

![c86f9c8fa85174b7d43af4839efb427544719f64e6779a525ad2d8081bf37382.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/images/c86f9c8fa85174b7d43af4839efb427544719f64e6779a525ad2d8081bf37382.jpg)

![cba63018edbafad378a10eb539e1c1a350eecfdf5ce9c3579e366d0948932a5b.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/images/cba63018edbafad378a10eb539e1c1a350eecfdf5ce9c3579e366d0948932a5b.jpg)

![cd5f17b5dee2e1d12205a919e2ce635bc6a016cfe0e2999aa3adb8faf866e83f.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/images/cd5f17b5dee2e1d12205a919e2ce635bc6a016cfe0e2999aa3adb8faf866e83f.jpg)

![da9faecc57c777a7d54d48a07d430fd2c023d080bb15db98c11f69f5f5b90e87.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/images/da9faecc57c777a7d54d48a07d430fd2c023d080bb15db98c11f69f5f5b90e87.jpg)

![df69a9e92511fb6a5065407ac524d25ede881291a3dd833ca2d9a0ea9943c21f.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/images/df69a9e92511fb6a5065407ac524d25ede881291a3dd833ca2d9a0ea9943c21f.jpg)

![f25edf924bfcc7539234926f2ebe5110d3228028f43dbb3bf4603fde89dacd29.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/images/f25edf924bfcc7539234926f2ebe5110d3228028f43dbb3bf4603fde89dacd29.jpg)

![fcd048e4f6ffbbc1e5fb0f9071ef20994bc34174672a9dcbcfb915840c7d03ed.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/images/fcd048e4f6ffbbc1e5fb0f9071ef20994bc34174672a9dcbcfb915840c7d03ed.jpg)

### Tables

![30292abfe88c6487de8f2b6d5ed817fc10c267be656550491cfe8d1988bfa512.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/tables/30292abfe88c6487de8f2b6d5ed817fc10c267be656550491cfe8d1988bfa512.jpg)

![34fec0ba7c430cf77b57d3c47f34ad34a80764f1fbcd7590b3249276c5fa7fa9.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/tables/34fec0ba7c430cf77b57d3c47f34ad34a80764f1fbcd7590b3249276c5fa7fa9.jpg)

![5ba309e51cc5d35cbd4ff205a13315d1acb88022773ee2221875c84aba936b9d.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/tables/5ba309e51cc5d35cbd4ff205a13315d1acb88022773ee2221875c84aba936b9d.jpg)

![ba5b0857b70a4ea4a0394b2c84fa1d964e2bd4ed8f0cb9feb7de67730684b898.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/tables/ba5b0857b70a4ea4a0394b2c84fa1d964e2bd4ed8f0cb9feb7de67730684b898.jpg)

![bfa83ce2b4a683b94c94b04086135169f335be75c3774201834452499cc2c7e1.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/tables/bfa83ce2b4a683b94c94b04086135169f335be75c3774201834452499cc2c7e1.jpg)

![c3bcb0a319ed9e4710dd7d46c3bbb9493ac7825dc9e9ead7b8b56998ee0d1b0a.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/tables/c3bcb0a319ed9e4710dd7d46c3bbb9493ac7825dc9e9ead7b8b56998ee0d1b0a.jpg)

![c9e95729022366c60f6ff16d85300b9e04d72566b6a361ca88211b89b89b56e0.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/tables/c9e95729022366c60f6ff16d85300b9e04d72566b6a361ca88211b89b89b56e0.jpg)

![cb5d70ec1022710d68c3ef5f335b6e3ede7e7ce1e021aa23eea1f15110a8d6de.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/tables/cb5d70ec1022710d68c3ef5f335b6e3ede7e7ce1e021aa23eea1f15110a8d6de.jpg)

![cc76d3df6e5aed4ecc1cd325937065a839b41ca36547bef8b134937e9b2ff886.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/tables/cc76d3df6e5aed4ecc1cd325937065a839b41ca36547bef8b134937e9b2ff886.jpg)

![d2769876b4043a0450f93f913435e26a088017f479eb7d69b71534bfb1658893.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/tables/d2769876b4043a0450f93f913435e26a088017f479eb7d69b71534bfb1658893.jpg)

![dc8234f660cadc55cb3f862d5a5ac3075daea5c441291fcba21fcd3abf619fd3.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/tables/dc8234f660cadc55cb3f862d5a5ac3075daea5c441291fcba21fcd3abf619fd3.jpg)

![ed69dcfdd777825a0f6bf5979d0a9bbcb236ad52f1203a1fed5ecc3c75328150.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/tables/ed69dcfdd777825a0f6bf5979d0a9bbcb236ad52f1203a1fed5ecc3c75328150.jpg)

![f261f530915cfe1ede1902f96f594975a44df566420c34a04c07846ac6c10deb.jpg](../iclr_results/3116_Hot-pluggable%20Federated%20Learning_%20Bridging%20General%20and%20Personalized%20FL%20via%20Dynamic%20Selection/tables/f261f530915cfe1ede1902f96f594975a44df566420c34a04c07846ac6c10deb.jpg)

## The Foundations of Tokenization: Statistical and Computational Concerns


### Images

![01b4083d40da03af65f5d952822451c8115521f676e3ece23d948dcb08e7855a.jpg](../iclr_results/3117_The%20Foundations%20of%20Tokenization_%20Statistical%20and%20Computational%20Concerns/images/01b4083d40da03af65f5d952822451c8115521f676e3ece23d948dcb08e7855a.jpg)

![08557157c8ddfb2b6a4150b672be45451f2ab72a0ef454364b8498e1e700c010.jpg](../iclr_results/3117_The%20Foundations%20of%20Tokenization_%20Statistical%20and%20Computational%20Concerns/images/08557157c8ddfb2b6a4150b672be45451f2ab72a0ef454364b8498e1e700c010.jpg)

![631dfc88cc91fcbf5ded20788caf9eb3cd6bdcf8f4bb8d49e4fb054dd49b7362.jpg](../iclr_results/3117_The%20Foundations%20of%20Tokenization_%20Statistical%20and%20Computational%20Concerns/images/631dfc88cc91fcbf5ded20788caf9eb3cd6bdcf8f4bb8d49e4fb054dd49b7362.jpg)

![97b8e446e9c84851172fe6a78631ea5a043d49bc4ae0e93fc36555457fbaecb3.jpg](../iclr_results/3117_The%20Foundations%20of%20Tokenization_%20Statistical%20and%20Computational%20Concerns/images/97b8e446e9c84851172fe6a78631ea5a043d49bc4ae0e93fc36555457fbaecb3.jpg)

![e78b1ae7673378eec3229bc780005bf82981ca6208564119ce1c5818e84d22f6.jpg](../iclr_results/3117_The%20Foundations%20of%20Tokenization_%20Statistical%20and%20Computational%20Concerns/images/e78b1ae7673378eec3229bc780005bf82981ca6208564119ce1c5818e84d22f6.jpg)

## SleepSMC: Ubiquitous Sleep Staging via Supervised Multimodal Coordination


### Images

![00fe58a716d0811764de27628ec395bf1089325f1031ad952ce0ba0ec6d6636e.jpg](../iclr_results/3118_SleepSMC_%20Ubiquitous%20Sleep%20Staging%20via%20Supervised%20Multimodal%20Coordination/images/00fe58a716d0811764de27628ec395bf1089325f1031ad952ce0ba0ec6d6636e.jpg)

![65a7163198d7ba718148a14e297136905e51ebc35aceb044339946182b603aaa.jpg](../iclr_results/3118_SleepSMC_%20Ubiquitous%20Sleep%20Staging%20via%20Supervised%20Multimodal%20Coordination/images/65a7163198d7ba718148a14e297136905e51ebc35aceb044339946182b603aaa.jpg)

![6f78579a45f9a76ea3b1d25fb63423e344e4fdca32fb6b9d639d423b7905d7da.jpg](../iclr_results/3118_SleepSMC_%20Ubiquitous%20Sleep%20Staging%20via%20Supervised%20Multimodal%20Coordination/images/6f78579a45f9a76ea3b1d25fb63423e344e4fdca32fb6b9d639d423b7905d7da.jpg)

![d2775cd2e4b8bf0f7a8e82fdc740fc29a7ce13e07675effbd6e4c46e98bfc8a4.jpg](../iclr_results/3118_SleepSMC_%20Ubiquitous%20Sleep%20Staging%20via%20Supervised%20Multimodal%20Coordination/images/d2775cd2e4b8bf0f7a8e82fdc740fc29a7ce13e07675effbd6e4c46e98bfc8a4.jpg)

### Tables

![0f5896e1f9e15a30731890b4be4c66963b51d6443d0e82026b156331e3eb3a41.jpg](../iclr_results/3118_SleepSMC_%20Ubiquitous%20Sleep%20Staging%20via%20Supervised%20Multimodal%20Coordination/tables/0f5896e1f9e15a30731890b4be4c66963b51d6443d0e82026b156331e3eb3a41.jpg)

![20827052061371c0cc1bfef7bebdff1560cff9358cb425864d3fe05f66b9b61b.jpg](../iclr_results/3118_SleepSMC_%20Ubiquitous%20Sleep%20Staging%20via%20Supervised%20Multimodal%20Coordination/tables/20827052061371c0cc1bfef7bebdff1560cff9358cb425864d3fe05f66b9b61b.jpg)

![86a821a09483955840bc9e408f326062a57733c54178171441d5242698ddb83b.jpg](../iclr_results/3118_SleepSMC_%20Ubiquitous%20Sleep%20Staging%20via%20Supervised%20Multimodal%20Coordination/tables/86a821a09483955840bc9e408f326062a57733c54178171441d5242698ddb83b.jpg)

![a165e44bc1943cde1cae4e4d358355f072e1f883e8ea37864ea7c8dc50fd4f16.jpg](../iclr_results/3118_SleepSMC_%20Ubiquitous%20Sleep%20Staging%20via%20Supervised%20Multimodal%20Coordination/tables/a165e44bc1943cde1cae4e4d358355f072e1f883e8ea37864ea7c8dc50fd4f16.jpg)

![d453ae09a4c9b69e3d94637483731804dad85afb4cf51bbf909e23a18abb4b06.jpg](../iclr_results/3118_SleepSMC_%20Ubiquitous%20Sleep%20Staging%20via%20Supervised%20Multimodal%20Coordination/tables/d453ae09a4c9b69e3d94637483731804dad85afb4cf51bbf909e23a18abb4b06.jpg)

![e1140199ed5014a6a762ca477058c2a8119e7c7c741234804b7f2d91176cfc5a.jpg](../iclr_results/3118_SleepSMC_%20Ubiquitous%20Sleep%20Staging%20via%20Supervised%20Multimodal%20Coordination/tables/e1140199ed5014a6a762ca477058c2a8119e7c7c741234804b7f2d91176cfc5a.jpg)

![e36dbc96e0576889bc6572782528daccf35769ed7d56c24a06be338c550a3368.jpg](../iclr_results/3118_SleepSMC_%20Ubiquitous%20Sleep%20Staging%20via%20Supervised%20Multimodal%20Coordination/tables/e36dbc96e0576889bc6572782528daccf35769ed7d56c24a06be338c550a3368.jpg)

![f2cd1e30032b2cdff2be81f4f05064e1fb881cc1c7e2e80956c9d7a3b85097b0.jpg](../iclr_results/3118_SleepSMC_%20Ubiquitous%20Sleep%20Staging%20via%20Supervised%20Multimodal%20Coordination/tables/f2cd1e30032b2cdff2be81f4f05064e1fb881cc1c7e2e80956c9d7a3b85097b0.jpg)

## FormalAlign: Automated Alignment Evaluation for Autoformalization


### Images

![27885945e83c589a782786c7571236aed1be1d799ff82031852a78fa4bc1c540.jpg](../iclr_results/3119_FormalAlign_%20Automated%20Alignment%20Evaluation%20for%20Autoformalization/images/27885945e83c589a782786c7571236aed1be1d799ff82031852a78fa4bc1c540.jpg)

![2a73e89bb8aa7c06bb4057dcd355fc90373f7d0aac2ec5283e8c932c84cb104c.jpg](../iclr_results/3119_FormalAlign_%20Automated%20Alignment%20Evaluation%20for%20Autoformalization/images/2a73e89bb8aa7c06bb4057dcd355fc90373f7d0aac2ec5283e8c932c84cb104c.jpg)

![73d4797cebcd10138324f9be990d90bc0bf2d4bde9813b0b3f37703c74550dc4.jpg](../iclr_results/3119_FormalAlign_%20Automated%20Alignment%20Evaluation%20for%20Autoformalization/images/73d4797cebcd10138324f9be990d90bc0bf2d4bde9813b0b3f37703c74550dc4.jpg)

### Tables

![090e5c7553721909c118a2f5f509629fb69bc910491a4d83ba21cc5122c19213.jpg](../iclr_results/3119_FormalAlign_%20Automated%20Alignment%20Evaluation%20for%20Autoformalization/tables/090e5c7553721909c118a2f5f509629fb69bc910491a4d83ba21cc5122c19213.jpg)

![16b3c9c5325aeed6a526becaad0fd2aaf93001a80d09b59aa1a8a81ec9222f63.jpg](../iclr_results/3119_FormalAlign_%20Automated%20Alignment%20Evaluation%20for%20Autoformalization/tables/16b3c9c5325aeed6a526becaad0fd2aaf93001a80d09b59aa1a8a81ec9222f63.jpg)

![24124b89aa47e3688b2eafacba9b74bfd7fe6e93c5c5274bf06e23ca5c1db363.jpg](../iclr_results/3119_FormalAlign_%20Automated%20Alignment%20Evaluation%20for%20Autoformalization/tables/24124b89aa47e3688b2eafacba9b74bfd7fe6e93c5c5274bf06e23ca5c1db363.jpg)

![31685e71a5b769a14d438e0f7b55c1b31ebb24c543e8de1819dfc65c3d6a2ca3.jpg](../iclr_results/3119_FormalAlign_%20Automated%20Alignment%20Evaluation%20for%20Autoformalization/tables/31685e71a5b769a14d438e0f7b55c1b31ebb24c543e8de1819dfc65c3d6a2ca3.jpg)

![44da5a9fb7e9043df4cfa6a5134e3243260e7931a806fe3e917de269baa1119e.jpg](../iclr_results/3119_FormalAlign_%20Automated%20Alignment%20Evaluation%20for%20Autoformalization/tables/44da5a9fb7e9043df4cfa6a5134e3243260e7931a806fe3e917de269baa1119e.jpg)

![5e5aacde912bc7923289beb7bfbaad2ffa0b7684a40a261bd7cdc265b86af075.jpg](../iclr_results/3119_FormalAlign_%20Automated%20Alignment%20Evaluation%20for%20Autoformalization/tables/5e5aacde912bc7923289beb7bfbaad2ffa0b7684a40a261bd7cdc265b86af075.jpg)

![646b032e076802a90a512da15f45e8aac7b6ee65c4fbb94649a4f7da6fa1b3a9.jpg](../iclr_results/3119_FormalAlign_%20Automated%20Alignment%20Evaluation%20for%20Autoformalization/tables/646b032e076802a90a512da15f45e8aac7b6ee65c4fbb94649a4f7da6fa1b3a9.jpg)

![769c361679fcc206789bd7faba8c5cb663c66cd66e6419b4ebd6423589bc8e28.jpg](../iclr_results/3119_FormalAlign_%20Automated%20Alignment%20Evaluation%20for%20Autoformalization/tables/769c361679fcc206789bd7faba8c5cb663c66cd66e6419b4ebd6423589bc8e28.jpg)

![877a5601a53936c3f7d9f2fda0e8d65cea466273ee96acee983c895cebedb978.jpg](../iclr_results/3119_FormalAlign_%20Automated%20Alignment%20Evaluation%20for%20Autoformalization/tables/877a5601a53936c3f7d9f2fda0e8d65cea466273ee96acee983c895cebedb978.jpg)

![93389d8066843c809447f6714da24a40cbe6c833ee30cf9c2211227811376e8e.jpg](../iclr_results/3119_FormalAlign_%20Automated%20Alignment%20Evaluation%20for%20Autoformalization/tables/93389d8066843c809447f6714da24a40cbe6c833ee30cf9c2211227811376e8e.jpg)

![a4344c8148d116c4f2ec34ec1401616115105bb8b25a285b81a120612fb3d6d4.jpg](../iclr_results/3119_FormalAlign_%20Automated%20Alignment%20Evaluation%20for%20Autoformalization/tables/a4344c8148d116c4f2ec34ec1401616115105bb8b25a285b81a120612fb3d6d4.jpg)

![cfa55ad6ee270821bfeef1649c1d05b22ef9363f685c2d0692d9aad2b586ce23.jpg](../iclr_results/3119_FormalAlign_%20Automated%20Alignment%20Evaluation%20for%20Autoformalization/tables/cfa55ad6ee270821bfeef1649c1d05b22ef9363f685c2d0692d9aad2b586ce23.jpg)

![df8c5e34db68d0c655a4e5e0abdd7493eb58caa5f1e5c2777f15693d232e765b.jpg](../iclr_results/3119_FormalAlign_%20Automated%20Alignment%20Evaluation%20for%20Autoformalization/tables/df8c5e34db68d0c655a4e5e0abdd7493eb58caa5f1e5c2777f15693d232e765b.jpg)

![e2ef798e2af0efb51b55a24e39b03032ba7b83e65a20eb7e8d46270c16b278e9.jpg](../iclr_results/3119_FormalAlign_%20Automated%20Alignment%20Evaluation%20for%20Autoformalization/tables/e2ef798e2af0efb51b55a24e39b03032ba7b83e65a20eb7e8d46270c16b278e9.jpg)

![e7eba64cf1111f72ddca8b262cf2541f720c3c7dec90b1caddef9b4550452136.jpg](../iclr_results/3119_FormalAlign_%20Automated%20Alignment%20Evaluation%20for%20Autoformalization/tables/e7eba64cf1111f72ddca8b262cf2541f720c3c7dec90b1caddef9b4550452136.jpg)

![ebc4ecf8b06e17610512c9aa69eeb99ea559be23f5ea0993d61294acee988d2b.jpg](../iclr_results/3119_FormalAlign_%20Automated%20Alignment%20Evaluation%20for%20Autoformalization/tables/ebc4ecf8b06e17610512c9aa69eeb99ea559be23f5ea0993d61294acee988d2b.jpg)

![f4a5107e5f488d02f75d7b50cb49a8e8c21778d62608ab1485d9d56579a96a9b.jpg](../iclr_results/3119_FormalAlign_%20Automated%20Alignment%20Evaluation%20for%20Autoformalization/tables/f4a5107e5f488d02f75d7b50cb49a8e8c21778d62608ab1485d9d56579a96a9b.jpg)

## TopoGaussian: Inferring Internal Topology Structures from Visual Clues


### Images

![0c7af77e1bf65d67360d1c58f736bdcd9465d4f5dbf4aa551452284c7a4a5d05.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/0c7af77e1bf65d67360d1c58f736bdcd9465d4f5dbf4aa551452284c7a4a5d05.jpg)

![354d1a7df2b381e892f6013d2e19b4b348f0a8bfdc77f80d4d3486589f0da71f.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/354d1a7df2b381e892f6013d2e19b4b348f0a8bfdc77f80d4d3486589f0da71f.jpg)

![3cace69aa7f409f6489ab5f20c06d76c354cc2e5c3d6a26c3183eeeb329cd2a6.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/3cace69aa7f409f6489ab5f20c06d76c354cc2e5c3d6a26c3183eeeb329cd2a6.jpg)

![3df1d05c70082166ceddaa8b0626fe0c73f2140119ffa3b870070c9a00294bff.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/3df1d05c70082166ceddaa8b0626fe0c73f2140119ffa3b870070c9a00294bff.jpg)

![459e7a722fd843db7b533b61c75cc9906aca1cda2e708fe8b189807fc2a35423.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/459e7a722fd843db7b533b61c75cc9906aca1cda2e708fe8b189807fc2a35423.jpg)

![5bf67a01ecb3742caed4e16ae92fae9e172cff85452036efcbb39237d3c4d77f.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/5bf67a01ecb3742caed4e16ae92fae9e172cff85452036efcbb39237d3c4d77f.jpg)

![60212b3b75333310fc3c5b430cb11a5eb50492e84bede1c02ac5f93284dcf3e7.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/60212b3b75333310fc3c5b430cb11a5eb50492e84bede1c02ac5f93284dcf3e7.jpg)

![6163e4776ee346690558fc96308e3901188c434ca208859b26b7ed5ea77d4cfd.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/6163e4776ee346690558fc96308e3901188c434ca208859b26b7ed5ea77d4cfd.jpg)

![618f945e0f0b4990f72cee02b15187e16e480d97750622264fe1b0e69bc394b4.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/618f945e0f0b4990f72cee02b15187e16e480d97750622264fe1b0e69bc394b4.jpg)

![67ab51aab1a04bd9d5d8571769e36df87b634113684ec1a3a817b056391d0efe.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/67ab51aab1a04bd9d5d8571769e36df87b634113684ec1a3a817b056391d0efe.jpg)

![74e88e076f70f35e4f04f9b0f182f0ee11d3f057584ac5577e2e7eff665973bf.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/74e88e076f70f35e4f04f9b0f182f0ee11d3f057584ac5577e2e7eff665973bf.jpg)

![90a2ebdac013259a52cfaa1c983f618778cc9f65b2d496d82df5627e67f5fa01.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/90a2ebdac013259a52cfaa1c983f618778cc9f65b2d496d82df5627e67f5fa01.jpg)

![93bb945cd092509f78944edfbdb57b2762ae2c560716b5aa331374c0124c833d.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/93bb945cd092509f78944edfbdb57b2762ae2c560716b5aa331374c0124c833d.jpg)

![a259d39e8bf4929d7cb6a6dbf9b9a00c98d2096a75011c76305ffb7fd3caadfa.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/a259d39e8bf4929d7cb6a6dbf9b9a00c98d2096a75011c76305ffb7fd3caadfa.jpg)

![a43b53cf15c2a93ddd212ab5597687fb9ea670c3d0e55b1930fdac05c24aa0b2.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/a43b53cf15c2a93ddd212ab5597687fb9ea670c3d0e55b1930fdac05c24aa0b2.jpg)

![a56188e1a16deb463025de92c5a831e02ffbca68aabdc56b6ca02b0eed870068.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/a56188e1a16deb463025de92c5a831e02ffbca68aabdc56b6ca02b0eed870068.jpg)

![a76f3e78857cb2625e8d6aef73d2dcfda00edbf455a4058d1ca736b61488dcac.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/a76f3e78857cb2625e8d6aef73d2dcfda00edbf455a4058d1ca736b61488dcac.jpg)

![b2056884ccfeb1eda39ea70d4509a3605e449a238947e2c80dc75e8f2fce0202.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/b2056884ccfeb1eda39ea70d4509a3605e449a238947e2c80dc75e8f2fce0202.jpg)

![ba2850748923b66ed5f63ff43c86067ebd6d7cdd1fcf30c8261ff0b3901839cc.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/ba2850748923b66ed5f63ff43c86067ebd6d7cdd1fcf30c8261ff0b3901839cc.jpg)

![be7788fac749afb88d672452df7c2ab385f4d0d59a6786fb4a224c17872a311c.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/be7788fac749afb88d672452df7c2ab385f4d0d59a6786fb4a224c17872a311c.jpg)

![c9b1406473039897d0964c9d5b8ba534e5caaa4f5f5720029abca28fb2ff74b4.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/c9b1406473039897d0964c9d5b8ba534e5caaa4f5f5720029abca28fb2ff74b4.jpg)

![cb841502dea2e203d5c4b3d9a6599f0a4412dbde43857f21828261346ab333f2.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/cb841502dea2e203d5c4b3d9a6599f0a4412dbde43857f21828261346ab333f2.jpg)

![ce2012cc71e637b8b520690ad948761d4958c1cc387e9da6c3a7349cfac83d1a.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/ce2012cc71e637b8b520690ad948761d4958c1cc387e9da6c3a7349cfac83d1a.jpg)

![d2be7ddce68ee06d129a245b7b322190621d3aab1ea3c05eee5c296eb9b4a248.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/d2be7ddce68ee06d129a245b7b322190621d3aab1ea3c05eee5c296eb9b4a248.jpg)

![d495220e6dc02b33275334c0e29b7173de3997679e3dde043d4c4740ccf3a053.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/d495220e6dc02b33275334c0e29b7173de3997679e3dde043d4c4740ccf3a053.jpg)

![de84580da193737e5fb564257c02d9095d480eb69eeff8ad91ccd4f0796e6a84.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/de84580da193737e5fb564257c02d9095d480eb69eeff8ad91ccd4f0796e6a84.jpg)

![f7dbf8421c4d25bf411f3c07e65d0f095c350f9cce521a430c2591af51632858.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/images/f7dbf8421c4d25bf411f3c07e65d0f095c350f9cce521a430c2591af51632858.jpg)

### Tables

![55560b2902709afc547460bd5b537c6f8c9b4838fcf991c0e900c14e442ea1c2.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/tables/55560b2902709afc547460bd5b537c6f8c9b4838fcf991c0e900c14e442ea1c2.jpg)

![a572fda9da75c1a77e2cc5696b5853aeb91979d188d4390ac6ca4679b8f13ef5.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/tables/a572fda9da75c1a77e2cc5696b5853aeb91979d188d4390ac6ca4679b8f13ef5.jpg)

![aded30c81136e4fdc27a9d67cf8eb5aac30e24512ca6d278daa6fa18b417d020.jpg](../iclr_results/3120_TopoGaussian_%20Inferring%20Internal%20Topology%20Structures%20from%20Visual%20Clues/tables/aded30c81136e4fdc27a9d67cf8eb5aac30e24512ca6d278daa6fa18b417d020.jpg)

## Understanding Constraint Inference in Safety-Critical Inverse Reinforcement Learning


### Images

![759b330129aa1e2b39e92f7cf700b13ec86b97bdd0c144c85b22a1e4e9d703d3.jpg](../iclr_results/3121_Understanding%20Constraint%20Inference%20in%20Safety-Critical%20Inverse%20Reinforcement%20Learning/images/759b330129aa1e2b39e92f7cf700b13ec86b97bdd0c144c85b22a1e4e9d703d3.jpg)

![78c780aab8ac8b7edbb4c0a0bc7f3339cb35f31f026ce2f9e6efe20f0057c3b6.jpg](../iclr_results/3121_Understanding%20Constraint%20Inference%20in%20Safety-Critical%20Inverse%20Reinforcement%20Learning/images/78c780aab8ac8b7edbb4c0a0bc7f3339cb35f31f026ce2f9e6efe20f0057c3b6.jpg)

![a750088ba3ae94e2c850b486516f49402775c14e3cdc136e79f0c1505305e5a8.jpg](../iclr_results/3121_Understanding%20Constraint%20Inference%20in%20Safety-Critical%20Inverse%20Reinforcement%20Learning/images/a750088ba3ae94e2c850b486516f49402775c14e3cdc136e79f0c1505305e5a8.jpg)

![c218e6b9177967a967aa81346ea0d3ffddfab86f6b18241cbcf0527c4aa3a66c.jpg](../iclr_results/3121_Understanding%20Constraint%20Inference%20in%20Safety-Critical%20Inverse%20Reinforcement%20Learning/images/c218e6b9177967a967aa81346ea0d3ffddfab86f6b18241cbcf0527c4aa3a66c.jpg)

![c2a44a4112394d75367e563be06c09aed03344c7e5519aa2c937cbdf62758e3d.jpg](../iclr_results/3121_Understanding%20Constraint%20Inference%20in%20Safety-Critical%20Inverse%20Reinforcement%20Learning/images/c2a44a4112394d75367e563be06c09aed03344c7e5519aa2c937cbdf62758e3d.jpg)

![e6ab83862fcfb4ddc9bf3fb8efb792091130bd547950f99ba6135dd32a9cff69.jpg](../iclr_results/3121_Understanding%20Constraint%20Inference%20in%20Safety-Critical%20Inverse%20Reinforcement%20Learning/images/e6ab83862fcfb4ddc9bf3fb8efb792091130bd547950f99ba6135dd32a9cff69.jpg)

![f6ee710470876966547375a8b8044bc0d4ceff1ef787d4eab06eb098c6a6e32c.jpg](../iclr_results/3121_Understanding%20Constraint%20Inference%20in%20Safety-Critical%20Inverse%20Reinforcement%20Learning/images/f6ee710470876966547375a8b8044bc0d4ceff1ef787d4eab06eb098c6a6e32c.jpg)

### Tables

![03a47fc2a6f474dcedce0f08b48a62831d85aa0e816d2e7ef50999e8530ee387.jpg](../iclr_results/3121_Understanding%20Constraint%20Inference%20in%20Safety-Critical%20Inverse%20Reinforcement%20Learning/tables/03a47fc2a6f474dcedce0f08b48a62831d85aa0e816d2e7ef50999e8530ee387.jpg)

![05e8bd45be4d56d33ee4f2b2de290f8528b6f5f01de7711851c81a7b95915b30.jpg](../iclr_results/3121_Understanding%20Constraint%20Inference%20in%20Safety-Critical%20Inverse%20Reinforcement%20Learning/tables/05e8bd45be4d56d33ee4f2b2de290f8528b6f5f01de7711851c81a7b95915b30.jpg)

![18d1ab6a9de32bdf0f4ad858bc9beefcbacd62418a1138ac0c9d79d6dd284f25.jpg](../iclr_results/3121_Understanding%20Constraint%20Inference%20in%20Safety-Critical%20Inverse%20Reinforcement%20Learning/tables/18d1ab6a9de32bdf0f4ad858bc9beefcbacd62418a1138ac0c9d79d6dd284f25.jpg)

![6bc13ad3a29c0aabbeed21764264b33f1f3d6b96f4f0ecdc76f58a208cf0a4bd.jpg](../iclr_results/3121_Understanding%20Constraint%20Inference%20in%20Safety-Critical%20Inverse%20Reinforcement%20Learning/tables/6bc13ad3a29c0aabbeed21764264b33f1f3d6b96f4f0ecdc76f58a208cf0a4bd.jpg)

![b767347e63b0fcb9275791984abc411c711d10e4c11b19d9a6ddcced0c118764.jpg](../iclr_results/3121_Understanding%20Constraint%20Inference%20in%20Safety-Critical%20Inverse%20Reinforcement%20Learning/tables/b767347e63b0fcb9275791984abc411c711d10e4c11b19d9a6ddcced0c118764.jpg)

## FLIP: Flow-Centric Generative Planning as General-Purpose Manipulation World Model

### Images

![0203f3bb41c103810645c1a2ffa957dc9d1180c6b85a9eeddc9cf9f903b7d9cd.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/images/0203f3bb41c103810645c1a2ffa957dc9d1180c6b85a9eeddc9cf9f903b7d9cd.jpg)

![09fb14ac4e4d361c68636ac2bf0809e7ff15183768e09601cb6685a004b876b1.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/images/09fb14ac4e4d361c68636ac2bf0809e7ff15183768e09601cb6685a004b876b1.jpg)

![0f827ff3b56e0cf13f020d59119eee5843b3621e3587a1b8728b0348c5767157.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/images/0f827ff3b56e0cf13f020d59119eee5843b3621e3587a1b8728b0348c5767157.jpg)

![17403b1451f8e2ae702443569aa47abf0063a2635a0929bb8461c41a77e56699.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/images/17403b1451f8e2ae702443569aa47abf0063a2635a0929bb8461c41a77e56699.jpg)

![2ccf7ad8113b23ec315b6e13e76e5ad4e00ea0240a2255075b6b52bbd296274f.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/images/2ccf7ad8113b23ec315b6e13e76e5ad4e00ea0240a2255075b6b52bbd296274f.jpg)

![454e822168a68d4aa2a083c6fb02313f88ab412cce5a2490e68629a2734b2fe5.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/images/454e822168a68d4aa2a083c6fb02313f88ab412cce5a2490e68629a2734b2fe5.jpg)

![4fb28bad830274d213610f57c16fc58389a99422fc7858ec49843feeb7cbeeb1.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/images/4fb28bad830274d213610f57c16fc58389a99422fc7858ec49843feeb7cbeeb1.jpg)

![51ef6193193599c74054419b07b6c4f77bedd0b450257a5fc42c70696fa55e6f.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/images/51ef6193193599c74054419b07b6c4f77bedd0b450257a5fc42c70696fa55e6f.jpg)

![54a1100ef7a3e2b326e37920093871a7e98bba85c60a0b0a9bf1d25c23732a5c.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/images/54a1100ef7a3e2b326e37920093871a7e98bba85c60a0b0a9bf1d25c23732a5c.jpg)

![85874a1bc94ba0ad5b24dc56655173580dd90c045e5ae6d4f7187a63b2ec443b.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/images/85874a1bc94ba0ad5b24dc56655173580dd90c045e5ae6d4f7187a63b2ec443b.jpg)

![a006a76a545e71e17a0f2eb96fedccadfb52e1cee58c5543172bc7e67001982a.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/images/a006a76a545e71e17a0f2eb96fedccadfb52e1cee58c5543172bc7e67001982a.jpg)

![bef150f04fe8c4aa729206b431988ba40e2ed9cd61dab8cac22fea59fd92959d.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/images/bef150f04fe8c4aa729206b431988ba40e2ed9cd61dab8cac22fea59fd92959d.jpg)

![c1b705632d2bf5181434e278d60b3aa894d7caf3e05ebdad393dcec97536c482.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/images/c1b705632d2bf5181434e278d60b3aa894d7caf3e05ebdad393dcec97536c482.jpg)

![e2e3fb14be0e9482224b52a5a956022e38ae5aa3272772a095cd185d87fec7ee.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/images/e2e3fb14be0e9482224b52a5a956022e38ae5aa3272772a095cd185d87fec7ee.jpg)

![e890b77271aae7b17dc1305517355745a9d6003f3678a11e14d20e156830b269.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/images/e890b77271aae7b17dc1305517355745a9d6003f3678a11e14d20e156830b269.jpg)

![f84f2562971ebd934c7992285af97cbe79e181de4f9eaef043020bb4c9fd93af.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/images/f84f2562971ebd934c7992285af97cbe79e181de4f9eaef043020bb4c9fd93af.jpg)

![fe66b5a5b36656c3c9830e7e7e89f902504a069cdd46da224d0c6bb88c457028.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/images/fe66b5a5b36656c3c9830e7e7e89f902504a069cdd46da224d0c6bb88c457028.jpg)

### Tables

![244df414c7a5d6a3f208ac2868ed3de057d414116aef23877cd2fc792862becf.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/tables/244df414c7a5d6a3f208ac2868ed3de057d414116aef23877cd2fc792862becf.jpg)

![3f16c8f2f9f1b063ebdb0b4114fa0f7dea21d7e7a4350933104e726c73dbac2b.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/tables/3f16c8f2f9f1b063ebdb0b4114fa0f7dea21d7e7a4350933104e726c73dbac2b.jpg)

![7abf6b07b5d2c9deb50a3977b46dd983b6e9eb89504d2464da0d00d898dec5f6.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/tables/7abf6b07b5d2c9deb50a3977b46dd983b6e9eb89504d2464da0d00d898dec5f6.jpg)

![88002c858a3872baa2bfc6bb2e1baff26cbaffed13f922f07c5c34ec29779b32.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/tables/88002c858a3872baa2bfc6bb2e1baff26cbaffed13f922f07c5c34ec29779b32.jpg)

![ab19a568d2a7fe774936ca9982fedb43b72f630a4066c41b777ec8a9116f1dea.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/tables/ab19a568d2a7fe774936ca9982fedb43b72f630a4066c41b777ec8a9116f1dea.jpg)

![dfe8d29b1c89f978101ea9a1d645292982400ef6efdcdd8fbaea724c083a0767.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/tables/dfe8d29b1c89f978101ea9a1d645292982400ef6efdcdd8fbaea724c083a0767.jpg)

![fc6af1f8bc5b68b0a607c4d735b456401244a81d1eaaf4591bc5cd70f4d06d73.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/tables/fc6af1f8bc5b68b0a607c4d735b456401244a81d1eaaf4591bc5cd70f4d06d73.jpg)

![ff24c796968e957b267b83abff94c6f1beba468d135f078c6a16fa61f42e7f8b.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/tables/ff24c796968e957b267b83abff94c6f1beba468d135f078c6a16fa61f42e7f8b.jpg)

![ffa7d086ba9fc50d3803f1ff188b1010c2bd6f9a508ae35f362d859e2b83d541.jpg](../iclr_results/3122_FLIP_%20Flow-Centric%20Generative%20Planning%20as%20General-Purpose%20Manipulation%20World%20Model/tables/ffa7d086ba9fc50d3803f1ff188b1010c2bd6f9a508ae35f362d859e2b83d541.jpg)
