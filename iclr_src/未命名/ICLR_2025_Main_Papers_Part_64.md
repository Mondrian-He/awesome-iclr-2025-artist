# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 64 of 100

## 目录 (Table of Contents)

1. [PEARL: Parallel Speculative Decoding with Adaptive Draft Length](#PEARL-Parallel-Speculative-Decoding-with-Adaptive-Draft-Length)
2. [Aligning Human Motion Generation with Human Perceptions](#Aligning-Human-Motion-Generation-with-Human-Perceptions)
3. [Microcanonical Langevin Ensembles: Advancing the Sampling of Bayesian Neural Networks](#Microcanonical-Langevin-Ensembles-Advancing-the-Sampling-of-Bayesian-Neural-Networks)
4. [Scale-aware Recognition in Satellite Images under Resource Constraints](#Scale-aware-Recognition-in-Satellite-Images-under-Resource-Constraints)
5. [MovieDreamer: Hierarchical Generation for Coherent Long Visual Sequences](#MovieDreamer-Hierarchical-Generation-for-Coherent-Long-Visual-Sequences)
6. [Model-agnostic meta-learners for estimating heterogeneous treatment effects over time](#Model-agnostic-meta-learners-for-estimating-heterogeneous-treatment-effects-over-time)
7. [Unleashing the Potential of Vision-Language Pre-Training for 3D Zero-Shot Lesion Segmentation via Mask-Attribute Alignment](#Unleashing-the-Potential-of-Vision-Language-Pre-Training-for-3D-Zero-Shot-Lesion-Segmentation-via-Mask-Attribute-Alignment)
8. [State Space Models are Provably Comparable to Transformers in Dynamic Token Selection](#State-Space-Models-are-Provably-Comparable-to-Transformers-in-Dynamic-Token-Selection)
9. [ImageFolder: Autoregressive Image Generation with Folded Tokens](#ImageFolder-Autoregressive-Image-Generation-with-Folded-Tokens)
10. [Model Equality Testing: Which Model is this API Serving?](#Model-Equality-Testing-Which-Model-is-this-API-Serving)
11. [Demystifying Topological Message-Passing with Relational Structures: A Case Study on Oversquashing in Simplicial Message-Passing](#Demystifying-Topological-Message-Passing-with-Relational-Structures-A-Case-Study-on-Oversquashing-in-Simplicial-Message-Passing)
12. [From an LLM Swarm to a PDDL-empowered Hive: Planning Self-executed Instructions in a Multi-modal Jungle](#From-an-LLM-Swarm-to-a-PDDL-empowered-Hive-Planning-Self-executed-Instructions-in-a-Multi-modal-Jungle)
13. [Navigating Neural Space: Revisiting Concept Activation Vectors to Overcome Directional Divergence](#Navigating-Neural-Space-Revisiting-Concept-Activation-Vectors-to-Overcome-Directional-Divergence)
14. [Generalized Behavior Learning from Diverse Demonstrations](#Generalized-Behavior-Learning-from-Diverse-Demonstrations)
15. [SoftMatcha: A Soft and Fast Pattern Matcher for Billion-Scale Corpus Searches](#SoftMatcha-A-Soft-and-Fast-Pattern-Matcher-for-Billion-Scale-Corpus-Searches)
16. [Efficient Source-Free Time-Series Adaptation via Parameter Subspace Disentanglement](#Efficient-Source-Free-Time-Series-Adaptation-via-Parameter-Subspace-Disentanglement)
17. [Do Contemporary Causal Inference Models Capture Real-World Heterogeneity? Findings from a Large-Scale Benchmark](#Do-Contemporary-Causal-Inference-Models-Capture-Real-World-Heterogeneity-Findings-from-a-Large-Scale-Benchmark)
18. [Towards Self-Supervised Covariance Estimation in Deep Heteroscedastic Regression](#Towards-Self-Supervised-Covariance-Estimation-in-Deep-Heteroscedastic-Regression)
19. [Underdamped Diffusion Bridges with Applications to Sampling](#Underdamped-Diffusion-Bridges-with-Applications-to-Sampling)
20. [A Closer Look at Machine Unlearning for Large Language Models](#A-Closer-Look-at-Machine-Unlearning-for-Large-Language-Models)
21. [A Robust Method to Discover Causal or Anticausal Relation](#A-Robust-Method-to-Discover-Causal-or-Anticausal-Relation)
22. [Debiasing Mini-Batch Quadratics for Applications in Deep Learning](#Debiasing-Mini-Batch-Quadratics-for-Applications-in-Deep-Learning)
23. [Scalable Benchmarking and Robust Learning for Noise-Free Ego-Motion and 3D Reconstruction from Noisy Video](#Scalable-Benchmarking-and-Robust-Learning-for-Noise-Free-Ego-Motion-and-3D-Reconstruction-from-Noisy-Video)
24. [SuperCorrect: Advancing Small LLM Reasoning with Thought Template Distillation and Self-Correction](#SuperCorrect-Advancing-Small-LLM-Reasoning-with-Thought-Template-Distillation-and-Self-Correction)
25. [EC-DIT: Scaling Diffusion Transformers with Adaptive Expert-Choice Routing](#EC-DIT-Scaling-Diffusion-Transformers-with-Adaptive-Expert-Choice-Routing)
26. [Predicting the Energy Landscape of Stochastic Dynamical System via  Physics-informed Self-supervised Learning](#Predicting-the-Energy-Landscape-of-Stochastic-Dynamical-System-via-Physics-informed-Self-supervised-Learning)
27. [Beyond Random Masking: When Dropout meets Graph Convolutional Networks](#Beyond-Random-Masking-When-Dropout-meets-Graph-Convolutional-Networks)
28. [Tight Clusters Make Specialized Experts](#Tight-Clusters-Make-Specialized-Experts)
29. [Domain Guidance: A Simple Transfer Approach for a Pre-trained Diffusion Model](#Domain-Guidance-A-Simple-Transfer-Approach-for-a-Pre-trained-Diffusion-Model)
30. [RAG-DDR: Optimizing Retrieval-Augmented Generation Using Differentiable Data Rewards](#RAG-DDR-Optimizing-Retrieval-Augmented-Generation-Using-Differentiable-Data-Rewards)
31. [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](#Magpie-Alignment-Data-Synthesis-from-Scratch-by-Prompting-Aligned-LLMs-with-Nothing)
32. [Gap Preserving Distillation by Building Bidirectional Mappings with A Dynamic Teacher](#Gap-Preserving-Distillation-by-Building-Bidirectional-Mappings-with-A-Dynamic-Teacher)
33. [SegLLM: Multi-round Reasoning Segmentation with Large Language Models](#SegLLM-Multi-round-Reasoning-Segmentation-with-Large-Language-Models)
34. [Should VLMs be Pre-trained with Image Data?](#Should-VLMs-be-Pre-trained-with-Image-Data)
35. [InfoGS: Efficient Structure-Aware 3D Gaussians via Lightweight Information Shaping](#InfoGS-Efficient-Structure-Aware-3D-Gaussians-via-Lightweight-Information-Shaping)
36. [Transformers Can Learn Temporal Difference Methods for In-Context Reinforcement Learning](#Transformers-Can-Learn-Temporal-Difference-Methods-for-In-Context-Reinforcement-Learning)
37. [CL-DiffPhyCon: Closed-loop Diffusion Control of Complex Physical Systems](#CL-DiffPhyCon-Closed-loop-Diffusion-Control-of-Complex-Physical-Systems)

---


## PEARL: Parallel Speculative Decoding with Adaptive Draft Length

### Images

![627c55daa196fa044040e29b8b005b57898c64c21811d1f4d2788a15cefe85d6.jpg](../iclr_results/2343_Highly%20Efficient%20Self-Adaptive%20Reward%20Shaping%20for%20Reinforcement%20Learning/images/627c55daa196fa044040e29b8b005b57898c64c21811d1f4d2788a15cefe85d6.jpg)

![7a66db650a34f31a323071d195c524d3c087a76b9285bba324cb69106286c6ba.jpg](../iclr_results/2343_Highly%20Efficient%20Self-Adaptive%20Reward%20Shaping%20for%20Reinforcement%20Learning/images/7a66db650a34f31a323071d195c524d3c087a76b9285bba324cb69106286c6ba.jpg)

![a106ff5a9b86c74fafce3dea698953bbb72a48e67a0e9b317d3dc1450ef3c427.jpg](../iclr_results/2343_Highly%20Efficient%20Self-Adaptive%20Reward%20Shaping%20for%20Reinforcement%20Learning/images/a106ff5a9b86c74fafce3dea698953bbb72a48e67a0e9b317d3dc1450ef3c427.jpg)

![ae86553dcbbec94bf68687bdaacf128345eed0fd2268874f49434f8564e1bddb.jpg](../iclr_results/2343_Highly%20Efficient%20Self-Adaptive%20Reward%20Shaping%20for%20Reinforcement%20Learning/images/ae86553dcbbec94bf68687bdaacf128345eed0fd2268874f49434f8564e1bddb.jpg)

![dba9089c7c6c824eee05bfa62c333ddb2fb127af8f0e538b12f7e9c585a4c178.jpg](../iclr_results/2343_Highly%20Efficient%20Self-Adaptive%20Reward%20Shaping%20for%20Reinforcement%20Learning/images/dba9089c7c6c824eee05bfa62c333ddb2fb127af8f0e538b12f7e9c585a4c178.jpg)

![e00a80b741b9c5f696ba30b8a3cec706407c62de54a7024a3f7a396e7fa3ec76.jpg](../iclr_results/2343_Highly%20Efficient%20Self-Adaptive%20Reward%20Shaping%20for%20Reinforcement%20Learning/images/e00a80b741b9c5f696ba30b8a3cec706407c62de54a7024a3f7a396e7fa3ec76.jpg)

![f99c66279872430e53c420122303b63784f959f945513802257d083756caec84.jpg](../iclr_results/2343_Highly%20Efficient%20Self-Adaptive%20Reward%20Shaping%20for%20Reinforcement%20Learning/images/f99c66279872430e53c420122303b63784f959f945513802257d083756caec84.jpg)

### Tables

![09a7f1f7e856bbdf8ebf0ac465275d73c2d3a44a60c490ce1aca482f25db2e54.jpg](../iclr_results/2343_Highly%20Efficient%20Self-Adaptive%20Reward%20Shaping%20for%20Reinforcement%20Learning/tables/09a7f1f7e856bbdf8ebf0ac465275d73c2d3a44a60c490ce1aca482f25db2e54.jpg)

![12530f41b4242022b2c382b51e110d47bc3324a73cd73c76c4352df932635c21.jpg](../iclr_results/2343_Highly%20Efficient%20Self-Adaptive%20Reward%20Shaping%20for%20Reinforcement%20Learning/tables/12530f41b4242022b2c382b51e110d47bc3324a73cd73c76c4352df932635c21.jpg)

![24c484a9ec1a3d6e904e588576f21bf00df9823fff433f84be38d4f25e672c84.jpg](../iclr_results/2343_Highly%20Efficient%20Self-Adaptive%20Reward%20Shaping%20for%20Reinforcement%20Learning/tables/24c484a9ec1a3d6e904e588576f21bf00df9823fff433f84be38d4f25e672c84.jpg)

![3b51c77f00af723bdf8cf3d7ce2935cbf6a8627f401d53626918d9b0c0f33b2b.jpg](../iclr_results/2343_Highly%20Efficient%20Self-Adaptive%20Reward%20Shaping%20for%20Reinforcement%20Learning/tables/3b51c77f00af723bdf8cf3d7ce2935cbf6a8627f401d53626918d9b0c0f33b2b.jpg)

![4da82aa0e12367236dac9986515a3c48dc3155e8efd87e3d67049ce19eea5351.jpg](../iclr_results/2343_Highly%20Efficient%20Self-Adaptive%20Reward%20Shaping%20for%20Reinforcement%20Learning/tables/4da82aa0e12367236dac9986515a3c48dc3155e8efd87e3d67049ce19eea5351.jpg)

![74382bfbbecc117363e2723ba5482f1a9d552e4aa10fe977dee5e9a6114c6c3a.jpg](../iclr_results/2343_Highly%20Efficient%20Self-Adaptive%20Reward%20Shaping%20for%20Reinforcement%20Learning/tables/74382bfbbecc117363e2723ba5482f1a9d552e4aa10fe977dee5e9a6114c6c3a.jpg)

![759552a955da334afd64299293dc0e62ba4d9b1b0317294878bc7ad86cb8a657.jpg](../iclr_results/2343_Highly%20Efficient%20Self-Adaptive%20Reward%20Shaping%20for%20Reinforcement%20Learning/tables/759552a955da334afd64299293dc0e62ba4d9b1b0317294878bc7ad86cb8a657.jpg)

![9167c1ac8321e041357106a659bd31607976f8db85c9e4771e6528a2bfa693db.jpg](../iclr_results/2343_Highly%20Efficient%20Self-Adaptive%20Reward%20Shaping%20for%20Reinforcement%20Learning/tables/9167c1ac8321e041357106a659bd31607976f8db85c9e4771e6528a2bfa693db.jpg)

![a8f440a6adc7a083bd122794f5cd52fb09c8e29966badc0ce9eea22f8e1b05dd.jpg](../iclr_results/2343_Highly%20Efficient%20Self-Adaptive%20Reward%20Shaping%20for%20Reinforcement%20Learning/tables/a8f440a6adc7a083bd122794f5cd52fb09c8e29966badc0ce9eea22f8e1b05dd.jpg)

![aee433737085f2e9b114f2ca93e5a9bb8e4400f7b30ff5aaa805b47631e95dd1.jpg](../iclr_results/2343_Highly%20Efficient%20Self-Adaptive%20Reward%20Shaping%20for%20Reinforcement%20Learning/tables/aee433737085f2e9b114f2ca93e5a9bb8e4400f7b30ff5aaa805b47631e95dd1.jpg)

![b6ee8e3a19be37e21771258d3a24d29f8d02f4f8f5b567303c0061f220777479.jpg](../iclr_results/2343_Highly%20Efficient%20Self-Adaptive%20Reward%20Shaping%20for%20Reinforcement%20Learning/tables/b6ee8e3a19be37e21771258d3a24d29f8d02f4f8f5b567303c0061f220777479.jpg)

![cb93967c6a6c3d5c67a06724e703ce40026dd77c6e3b8f4ba5b4e2e8d0f84351.jpg](../iclr_results/2343_Highly%20Efficient%20Self-Adaptive%20Reward%20Shaping%20for%20Reinforcement%20Learning/tables/cb93967c6a6c3d5c67a06724e703ce40026dd77c6e3b8f4ba5b4e2e8d0f84351.jpg)

![dfe434e4037670884847f500cb657aad83cdd6b100f424192f6b8da405777e17.jpg](../iclr_results/2343_Highly%20Efficient%20Self-Adaptive%20Reward%20Shaping%20for%20Reinforcement%20Learning/tables/dfe434e4037670884847f500cb657aad83cdd6b100f424192f6b8da405777e17.jpg)

![f1877ebb70754edc8436e9b62838f9e807b74a8c4a90ac43d1078c5d62599223.jpg](../iclr_results/2343_Highly%20Efficient%20Self-Adaptive%20Reward%20Shaping%20for%20Reinforcement%20Learning/tables/f1877ebb70754edc8436e9b62838f9e807b74a8c4a90ac43d1078c5d62599223.jpg)

## PEARL: Parallel Speculative Decoding with Adaptive Draft Length


### Images

![6d77d0fc92c9392631eb614c0b32b6ca8f73415791132958592a32de34d11316.jpg](../iclr_results/2344_PEARL_%20Parallel%20Speculative%20Decoding%20with%20Adaptive%20Draft%20Length/images/6d77d0fc92c9392631eb614c0b32b6ca8f73415791132958592a32de34d11316.jpg)

![960b7b756fa3bebae7f667a6dd54d2e15fb4fd06be735a4b0d89b26a67236b2e.jpg](../iclr_results/2344_PEARL_%20Parallel%20Speculative%20Decoding%20with%20Adaptive%20Draft%20Length/images/960b7b756fa3bebae7f667a6dd54d2e15fb4fd06be735a4b0d89b26a67236b2e.jpg)

![ba01d1c510d78c0a3f64a5d4281a25d7122d0a671ef4f61c0336272285081a9b.jpg](../iclr_results/2344_PEARL_%20Parallel%20Speculative%20Decoding%20with%20Adaptive%20Draft%20Length/images/ba01d1c510d78c0a3f64a5d4281a25d7122d0a671ef4f61c0336272285081a9b.jpg)

### Tables

![044c2eb45ad921cdbfdb9082426d7de73b2ecdf4174af55156c20937841bc5b9.jpg](../iclr_results/2344_PEARL_%20Parallel%20Speculative%20Decoding%20with%20Adaptive%20Draft%20Length/tables/044c2eb45ad921cdbfdb9082426d7de73b2ecdf4174af55156c20937841bc5b9.jpg)

![059c2652cb95fcca5060439b53c0ee0801a4088da00c23ca829932025a19590a.jpg](../iclr_results/2344_PEARL_%20Parallel%20Speculative%20Decoding%20with%20Adaptive%20Draft%20Length/tables/059c2652cb95fcca5060439b53c0ee0801a4088da00c23ca829932025a19590a.jpg)

![1e56484b75e72ce0b76a55f30bfca3490c7af28ebb652b1b68985defc79ed2db.jpg](../iclr_results/2344_PEARL_%20Parallel%20Speculative%20Decoding%20with%20Adaptive%20Draft%20Length/tables/1e56484b75e72ce0b76a55f30bfca3490c7af28ebb652b1b68985defc79ed2db.jpg)

![3bc2a698800f4247ab74076f03254801337fdd1447a8013c534c16bb3fa34194.jpg](../iclr_results/2344_PEARL_%20Parallel%20Speculative%20Decoding%20with%20Adaptive%20Draft%20Length/tables/3bc2a698800f4247ab74076f03254801337fdd1447a8013c534c16bb3fa34194.jpg)

![480d8d239adb86e5f7c6de3a93b3b45ddc2e2d81cb76eba20703cb7f03443a7c.jpg](../iclr_results/2344_PEARL_%20Parallel%20Speculative%20Decoding%20with%20Adaptive%20Draft%20Length/tables/480d8d239adb86e5f7c6de3a93b3b45ddc2e2d81cb76eba20703cb7f03443a7c.jpg)

![4b437419d27e211fcca396913fe0ed09c1a1408396b4928a7aca65d0d647ce84.jpg](../iclr_results/2344_PEARL_%20Parallel%20Speculative%20Decoding%20with%20Adaptive%20Draft%20Length/tables/4b437419d27e211fcca396913fe0ed09c1a1408396b4928a7aca65d0d647ce84.jpg)

![56433b26f8a6d3697751e3a47db918a31be4039f2aa0e6af3c2bc5669df3be73.jpg](../iclr_results/2344_PEARL_%20Parallel%20Speculative%20Decoding%20with%20Adaptive%20Draft%20Length/tables/56433b26f8a6d3697751e3a47db918a31be4039f2aa0e6af3c2bc5669df3be73.jpg)

![60f711915ef5d8b3c39bca49c6a2dfe786f8c1efdca69fbc6e602bd167e6cf08.jpg](../iclr_results/2344_PEARL_%20Parallel%20Speculative%20Decoding%20with%20Adaptive%20Draft%20Length/tables/60f711915ef5d8b3c39bca49c6a2dfe786f8c1efdca69fbc6e602bd167e6cf08.jpg)

![6244f6f09dfc98a39200fdb297486c3f594591c50b6e404f480a8ebd62c76a53.jpg](../iclr_results/2344_PEARL_%20Parallel%20Speculative%20Decoding%20with%20Adaptive%20Draft%20Length/tables/6244f6f09dfc98a39200fdb297486c3f594591c50b6e404f480a8ebd62c76a53.jpg)

![723db16f6b37f988a07650b37703a7ae963f4fc36239c54fb03e6df504ebac30.jpg](../iclr_results/2344_PEARL_%20Parallel%20Speculative%20Decoding%20with%20Adaptive%20Draft%20Length/tables/723db16f6b37f988a07650b37703a7ae963f4fc36239c54fb03e6df504ebac30.jpg)

![76e08a605e0bc417316e645b5a09e41f31824195c3a42cba9e81d7e3a2211b15.jpg](../iclr_results/2344_PEARL_%20Parallel%20Speculative%20Decoding%20with%20Adaptive%20Draft%20Length/tables/76e08a605e0bc417316e645b5a09e41f31824195c3a42cba9e81d7e3a2211b15.jpg)

![7ffa30b46e1ec944e4ae28282cd7565d2e6113fd03277d29ba46ce28b92c7200.jpg](../iclr_results/2344_PEARL_%20Parallel%20Speculative%20Decoding%20with%20Adaptive%20Draft%20Length/tables/7ffa30b46e1ec944e4ae28282cd7565d2e6113fd03277d29ba46ce28b92c7200.jpg)

![aa42701aff7a13dbc2d6728b80e1ee08ab30483103a2883e6f74c2387f1191d3.jpg](../iclr_results/2344_PEARL_%20Parallel%20Speculative%20Decoding%20with%20Adaptive%20Draft%20Length/tables/aa42701aff7a13dbc2d6728b80e1ee08ab30483103a2883e6f74c2387f1191d3.jpg)

![ac1f799132d37bdaa40618240af594ed1fc7e621df3028a03916220341491061.jpg](../iclr_results/2344_PEARL_%20Parallel%20Speculative%20Decoding%20with%20Adaptive%20Draft%20Length/tables/ac1f799132d37bdaa40618240af594ed1fc7e621df3028a03916220341491061.jpg)

![c98611b1b323c85863965a3edf296eee772bddbc4b9ff122f569b545ade3beb6.jpg](../iclr_results/2344_PEARL_%20Parallel%20Speculative%20Decoding%20with%20Adaptive%20Draft%20Length/tables/c98611b1b323c85863965a3edf296eee772bddbc4b9ff122f569b545ade3beb6.jpg)

![d3003847505fb2095bd61d7f4c68796e1202aa03ffd13f5193a0cc85e292c951.jpg](../iclr_results/2344_PEARL_%20Parallel%20Speculative%20Decoding%20with%20Adaptive%20Draft%20Length/tables/d3003847505fb2095bd61d7f4c68796e1202aa03ffd13f5193a0cc85e292c951.jpg)

![e5c0d9db5fbe2e9b958d7c8a58f49879fa0e26cfec91e904fcd60603c53d6a05.jpg](../iclr_results/2344_PEARL_%20Parallel%20Speculative%20Decoding%20with%20Adaptive%20Draft%20Length/tables/e5c0d9db5fbe2e9b958d7c8a58f49879fa0e26cfec91e904fcd60603c53d6a05.jpg)

![f2c03cd077e38e9164d87eb8f678dfb1b5e42381519b80995973ea7dbf8642f6.jpg](../iclr_results/2344_PEARL_%20Parallel%20Speculative%20Decoding%20with%20Adaptive%20Draft%20Length/tables/f2c03cd077e38e9164d87eb8f678dfb1b5e42381519b80995973ea7dbf8642f6.jpg)

## Aligning Human Motion Generation with Human Perceptions


### Images

![1a823f9b6f7c5f5f84c84cf529b5934093d5ef2d2ee848279a7fd649a22b3ebc.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/images/1a823f9b6f7c5f5f84c84cf529b5934093d5ef2d2ee848279a7fd649a22b3ebc.jpg)

![22cd56b02d4d4a0b58c3be8f4540164d5dc65e4c1bb956b26e6432288c363ca5.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/images/22cd56b02d4d4a0b58c3be8f4540164d5dc65e4c1bb956b26e6432288c363ca5.jpg)

![2705b3d6dfec6e2ce6656f1500c9b1a1e671365b44656eb9e9c95691ea385b7e.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/images/2705b3d6dfec6e2ce6656f1500c9b1a1e671365b44656eb9e9c95691ea385b7e.jpg)

![295668c859cdbad336273e08ec35ac680bdc5aa999d3d4f372329fe2bd66360f.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/images/295668c859cdbad336273e08ec35ac680bdc5aa999d3d4f372329fe2bd66360f.jpg)

![4ab98a7027d79379d8bac8ca23001d90b906e735ed2b04f69d5799628a209c19.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/images/4ab98a7027d79379d8bac8ca23001d90b906e735ed2b04f69d5799628a209c19.jpg)

![5141c914f0f70d834eed4024346b4dee7cb684320e2843d3cfc6f88aef24906c.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/images/5141c914f0f70d834eed4024346b4dee7cb684320e2843d3cfc6f88aef24906c.jpg)

![56e334916c0ac0ecb92c8191d176876260d82925108cbb9aeaa15f674b29a0b4.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/images/56e334916c0ac0ecb92c8191d176876260d82925108cbb9aeaa15f674b29a0b4.jpg)

![5a5edcdedf8ee9b7c89ab17a7c71373d9302db5910a2c0655836c7d17d795fc4.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/images/5a5edcdedf8ee9b7c89ab17a7c71373d9302db5910a2c0655836c7d17d795fc4.jpg)

![7dc744faaa628cb401f5ef54f8dd9bafce89dd453a94b5de1608095cb7197348.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/images/7dc744faaa628cb401f5ef54f8dd9bafce89dd453a94b5de1608095cb7197348.jpg)

![7edd76e4b7ec1592b5949863e7b83faa7bf91b28929582dbc310cce7117a236b.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/images/7edd76e4b7ec1592b5949863e7b83faa7bf91b28929582dbc310cce7117a236b.jpg)

![8a4979c9a397359e76784dbb983385630b657bc6aa50d0ce3af83a8e1c329cf7.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/images/8a4979c9a397359e76784dbb983385630b657bc6aa50d0ce3af83a8e1c329cf7.jpg)

![9421f82176949d6d120cfbb01e049564373cb67e198eb50f9a0b11c1d03e8598.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/images/9421f82176949d6d120cfbb01e049564373cb67e198eb50f9a0b11c1d03e8598.jpg)

![9818c98ea5b78e02b220f0fe91e12e3ccad2b8bbc9968ac3c828aac49ff0c454.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/images/9818c98ea5b78e02b220f0fe91e12e3ccad2b8bbc9968ac3c828aac49ff0c454.jpg)

![9f5c46b448a719b78664000293ae9be3506b6dbf9de4b433067a46fcddc04d18.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/images/9f5c46b448a719b78664000293ae9be3506b6dbf9de4b433067a46fcddc04d18.jpg)

![a21bb2a8ec2b809cc19f627c47139477121f864ecc79bf004ba15366ef2ecce9.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/images/a21bb2a8ec2b809cc19f627c47139477121f864ecc79bf004ba15366ef2ecce9.jpg)

![ab241321b420cec73bfdd753b07320a09995e869a076117feb0fe7318f110c97.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/images/ab241321b420cec73bfdd753b07320a09995e869a076117feb0fe7318f110c97.jpg)

![bdc3b7d289a6a1cbe3ef8fb51bb2ab9be7a838635ed7fef01d5d666ddca27e6e.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/images/bdc3b7d289a6a1cbe3ef8fb51bb2ab9be7a838635ed7fef01d5d666ddca27e6e.jpg)

![d73c44db7548423ae6fdc0647dc5367dea29c2e2b36a03165bf6dd0f7746d3ff.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/images/d73c44db7548423ae6fdc0647dc5367dea29c2e2b36a03165bf6dd0f7746d3ff.jpg)

### Tables

![00d80d3b297081432fcc3a8df51a7729042a3c29fd6001642c76f1158e02d4a3.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/tables/00d80d3b297081432fcc3a8df51a7729042a3c29fd6001642c76f1158e02d4a3.jpg)

![105164e5c15bd957627874c8d60f21f988a5086e4ef2d5786293838c7bbd0125.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/tables/105164e5c15bd957627874c8d60f21f988a5086e4ef2d5786293838c7bbd0125.jpg)

![21f7fa413f66776a5d3f1d79c6ea41b1ac203baa1c91003647ec2bb9e6a09dad.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/tables/21f7fa413f66776a5d3f1d79c6ea41b1ac203baa1c91003647ec2bb9e6a09dad.jpg)

![231e1e01723f35bdd76db3f4a1ab521a8f6f8cf42f50fb9162e64c98f6ef46c2.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/tables/231e1e01723f35bdd76db3f4a1ab521a8f6f8cf42f50fb9162e64c98f6ef46c2.jpg)

![2884c1c455d9eab296a17e56a5182801de5a2fd2acd6a1c85e8ecd65133d1d20.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/tables/2884c1c455d9eab296a17e56a5182801de5a2fd2acd6a1c85e8ecd65133d1d20.jpg)

![3603fa76c8248ce057d7190f24c4cc59063f83b447208a8e4d90dc2c0040353f.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/tables/3603fa76c8248ce057d7190f24c4cc59063f83b447208a8e4d90dc2c0040353f.jpg)

![3ed1fb020ad27d9a66b27aa06fe0931f288d663b2c41b3887c9f1d6ef390a18f.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/tables/3ed1fb020ad27d9a66b27aa06fe0931f288d663b2c41b3887c9f1d6ef390a18f.jpg)

![446836c1ecfe18b553aa69a2d846c4ee2224613e06946321e44a39e648ac281d.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/tables/446836c1ecfe18b553aa69a2d846c4ee2224613e06946321e44a39e648ac281d.jpg)

![67692cb0232c3bb2b6171dff3bbb29516dcba3901e83919664345d0c6014f92c.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/tables/67692cb0232c3bb2b6171dff3bbb29516dcba3901e83919664345d0c6014f92c.jpg)

![685bd7ebbd09a6bc83b51ac2014375fd2bc96a8930ce8227ca34229389c9d433.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/tables/685bd7ebbd09a6bc83b51ac2014375fd2bc96a8930ce8227ca34229389c9d433.jpg)

![7ede26c985450eec2febcc5e5e94b1cdb6e5369903be31b2c0812b123713d688.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/tables/7ede26c985450eec2febcc5e5e94b1cdb6e5369903be31b2c0812b123713d688.jpg)

![920f834c12ed63e9039e70b5987932303ae18cd58de62c6604e2b7fcca823b9f.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/tables/920f834c12ed63e9039e70b5987932303ae18cd58de62c6604e2b7fcca823b9f.jpg)

![a1dcbc1c9cc3ffb5df4af91afd75253033c78c04c194f03da6659d7eb6dd4855.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/tables/a1dcbc1c9cc3ffb5df4af91afd75253033c78c04c194f03da6659d7eb6dd4855.jpg)

![d67a017ae446f0c523281e1ddd9bc20c0ea89178449cda30782d05175557fd9d.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/tables/d67a017ae446f0c523281e1ddd9bc20c0ea89178449cda30782d05175557fd9d.jpg)

![dc4b8766a961484627424f343ee09539a2780bf713e957502a9c17f537309bfb.jpg](../iclr_results/2345_Aligning%20Human%20Motion%20Generation%20with%20Human%20Perceptions/tables/dc4b8766a961484627424f343ee09539a2780bf713e957502a9c17f537309bfb.jpg)

## Microcanonical Langevin Ensembles: Advancing the Sampling of Bayesian Neural Networks


### Images

![3b39a3bb8897a3e386929bb8be03edcebbc60d40fe2342487ac93a1e415b689e.jpg](../iclr_results/2346_Microcanonical%20Langevin%20Ensembles_%20Advancing%20the%20Sampling%20of%20Bayesian%20Neural%20Networks/images/3b39a3bb8897a3e386929bb8be03edcebbc60d40fe2342487ac93a1e415b689e.jpg)

![3bb6bc2388764babc6e560d7c3237ff6b64760ac4cb37c282d72310e9d4d1426.jpg](../iclr_results/2346_Microcanonical%20Langevin%20Ensembles_%20Advancing%20the%20Sampling%20of%20Bayesian%20Neural%20Networks/images/3bb6bc2388764babc6e560d7c3237ff6b64760ac4cb37c282d72310e9d4d1426.jpg)

![4812009e57446763b8e7fc4f050fa2e4c8cd7ae3aa2e8298eb8a87b5c7ffea97.jpg](../iclr_results/2346_Microcanonical%20Langevin%20Ensembles_%20Advancing%20the%20Sampling%20of%20Bayesian%20Neural%20Networks/images/4812009e57446763b8e7fc4f050fa2e4c8cd7ae3aa2e8298eb8a87b5c7ffea97.jpg)

![6aab8cc9456c14ea2b7cdffeb32065502ac73e7b638bed47ff789b9be9ac99bf.jpg](../iclr_results/2346_Microcanonical%20Langevin%20Ensembles_%20Advancing%20the%20Sampling%20of%20Bayesian%20Neural%20Networks/images/6aab8cc9456c14ea2b7cdffeb32065502ac73e7b638bed47ff789b9be9ac99bf.jpg)

![8044987a80bec81c7670ff172d904eb452d905e34a8e92dda968fac31fce2e4f.jpg](../iclr_results/2346_Microcanonical%20Langevin%20Ensembles_%20Advancing%20the%20Sampling%20of%20Bayesian%20Neural%20Networks/images/8044987a80bec81c7670ff172d904eb452d905e34a8e92dda968fac31fce2e4f.jpg)

![8d175e94588604a4e0a11a3beb0ee1309f6d04a8f378e6408fdae44e8c8756f7.jpg](../iclr_results/2346_Microcanonical%20Langevin%20Ensembles_%20Advancing%20the%20Sampling%20of%20Bayesian%20Neural%20Networks/images/8d175e94588604a4e0a11a3beb0ee1309f6d04a8f378e6408fdae44e8c8756f7.jpg)

![db7edf8fdad3042263b5b42a86b6016354c5f094a34f22f33178dc3489e07edb.jpg](../iclr_results/2346_Microcanonical%20Langevin%20Ensembles_%20Advancing%20the%20Sampling%20of%20Bayesian%20Neural%20Networks/images/db7edf8fdad3042263b5b42a86b6016354c5f094a34f22f33178dc3489e07edb.jpg)

### Tables

![17307c9f6a20380d17180bb7ed02c7a5384ed55a91a977a3828e7eaf261d956e.jpg](../iclr_results/2346_Microcanonical%20Langevin%20Ensembles_%20Advancing%20the%20Sampling%20of%20Bayesian%20Neural%20Networks/tables/17307c9f6a20380d17180bb7ed02c7a5384ed55a91a977a3828e7eaf261d956e.jpg)

![1956ca1ed10119848fc765b37b9fdb39a7cc94d4a14fc095b714c27dd23e9eab.jpg](../iclr_results/2346_Microcanonical%20Langevin%20Ensembles_%20Advancing%20the%20Sampling%20of%20Bayesian%20Neural%20Networks/tables/1956ca1ed10119848fc765b37b9fdb39a7cc94d4a14fc095b714c27dd23e9eab.jpg)

![1e3586eb8e70de6f694626e5432fc38e37d9dc9a913547f1574350e42e39bb04.jpg](../iclr_results/2346_Microcanonical%20Langevin%20Ensembles_%20Advancing%20the%20Sampling%20of%20Bayesian%20Neural%20Networks/tables/1e3586eb8e70de6f694626e5432fc38e37d9dc9a913547f1574350e42e39bb04.jpg)

![3c8cb22dd928bf787343783c09708dfd8969fa260addb1cd2ac88f8e28325eee.jpg](../iclr_results/2346_Microcanonical%20Langevin%20Ensembles_%20Advancing%20the%20Sampling%20of%20Bayesian%20Neural%20Networks/tables/3c8cb22dd928bf787343783c09708dfd8969fa260addb1cd2ac88f8e28325eee.jpg)

![47d1924be74878aaa438431ae7f3f317641e1a82e6d6f0d5ac510c927768b9ef.jpg](../iclr_results/2346_Microcanonical%20Langevin%20Ensembles_%20Advancing%20the%20Sampling%20of%20Bayesian%20Neural%20Networks/tables/47d1924be74878aaa438431ae7f3f317641e1a82e6d6f0d5ac510c927768b9ef.jpg)

![847dcfa7e7310eaadf9f58d1266ac67e4effd89792c30e08181997689dab4d07.jpg](../iclr_results/2346_Microcanonical%20Langevin%20Ensembles_%20Advancing%20the%20Sampling%20of%20Bayesian%20Neural%20Networks/tables/847dcfa7e7310eaadf9f58d1266ac67e4effd89792c30e08181997689dab4d07.jpg)

![9145f754502428bfd6f68777cdda488d8a8ff2b6da2c551af9111b644f808ec8.jpg](../iclr_results/2346_Microcanonical%20Langevin%20Ensembles_%20Advancing%20the%20Sampling%20of%20Bayesian%20Neural%20Networks/tables/9145f754502428bfd6f68777cdda488d8a8ff2b6da2c551af9111b644f808ec8.jpg)

![919218d400039354008b80be1f0f11d6c5af78640ebf75fe8255a763c3f83a6c.jpg](../iclr_results/2346_Microcanonical%20Langevin%20Ensembles_%20Advancing%20the%20Sampling%20of%20Bayesian%20Neural%20Networks/tables/919218d400039354008b80be1f0f11d6c5af78640ebf75fe8255a763c3f83a6c.jpg)

![93901f5bc20c540b33a95f53255da91611a5136e09559d40a00fa17fd4a5dbab.jpg](../iclr_results/2346_Microcanonical%20Langevin%20Ensembles_%20Advancing%20the%20Sampling%20of%20Bayesian%20Neural%20Networks/tables/93901f5bc20c540b33a95f53255da91611a5136e09559d40a00fa17fd4a5dbab.jpg)

![efc1f890e643d9c98343390f2dbf4d681d441f8916145088cbf94395ca6f1928.jpg](../iclr_results/2346_Microcanonical%20Langevin%20Ensembles_%20Advancing%20the%20Sampling%20of%20Bayesian%20Neural%20Networks/tables/efc1f890e643d9c98343390f2dbf4d681d441f8916145088cbf94395ca6f1928.jpg)

## Scale-aware Recognition in Satellite Images under Resource Constraints


### Images

![5cf73207e53b30a360665d81248c24094c1f1fe16819a8a9d7eeb1b527d8f797.jpg](../iclr_results/2347_Scale-aware%20Recognition%20in%20Satellite%20Images%20under%20Resource%20Constraints/images/5cf73207e53b30a360665d81248c24094c1f1fe16819a8a9d7eeb1b527d8f797.jpg)

![7e755bcc989fd1b23e682632a28b66b6b86f3ac5cdcd0afb4faaa8aa9225eed1.jpg](../iclr_results/2347_Scale-aware%20Recognition%20in%20Satellite%20Images%20under%20Resource%20Constraints/images/7e755bcc989fd1b23e682632a28b66b6b86f3ac5cdcd0afb4faaa8aa9225eed1.jpg)

![83fa978b53bddcdbcc93a0e9955259487a561273d033efc074cef1f491775a34.jpg](../iclr_results/2347_Scale-aware%20Recognition%20in%20Satellite%20Images%20under%20Resource%20Constraints/images/83fa978b53bddcdbcc93a0e9955259487a561273d033efc074cef1f491775a34.jpg)

![ff23a65b48b10b333896897756ae8c80c1159aa494dccd719689b4d879e9c06e.jpg](../iclr_results/2347_Scale-aware%20Recognition%20in%20Satellite%20Images%20under%20Resource%20Constraints/images/ff23a65b48b10b333896897756ae8c80c1159aa494dccd719689b4d879e9c06e.jpg)

### Tables

![042d2ebf7662499edecb55fe13845b0fa1b951682b6206ad127bc8b34b80633e.jpg](../iclr_results/2347_Scale-aware%20Recognition%20in%20Satellite%20Images%20under%20Resource%20Constraints/tables/042d2ebf7662499edecb55fe13845b0fa1b951682b6206ad127bc8b34b80633e.jpg)

![338184938c03f8d629d4f5f94e061e886691a715c1f6f08ca55e0bc0c50caaea.jpg](../iclr_results/2347_Scale-aware%20Recognition%20in%20Satellite%20Images%20under%20Resource%20Constraints/tables/338184938c03f8d629d4f5f94e061e886691a715c1f6f08ca55e0bc0c50caaea.jpg)

![74b25dddf3d722187c7cf4d18d7ec7c920314098c6adddfc32aa937054b161a6.jpg](../iclr_results/2347_Scale-aware%20Recognition%20in%20Satellite%20Images%20under%20Resource%20Constraints/tables/74b25dddf3d722187c7cf4d18d7ec7c920314098c6adddfc32aa937054b161a6.jpg)

![d481ed800bde59f45399015820f65f48bb514ae53fcaf11ce237f6ee541f5acf.jpg](../iclr_results/2347_Scale-aware%20Recognition%20in%20Satellite%20Images%20under%20Resource%20Constraints/tables/d481ed800bde59f45399015820f65f48bb514ae53fcaf11ce237f6ee541f5acf.jpg)

## MovieDreamer: Hierarchical Generation for Coherent Long Visual Sequences


### Images

![058a642a72e17d51fdad048c734d8b839fe891d21e6381c271e8c5fecf64ec6b.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/058a642a72e17d51fdad048c734d8b839fe891d21e6381c271e8c5fecf64ec6b.jpg)

![0ab3b8cb971938b4623bd46b180f4f9d35e2615f3a625a10c60b91b60bdc97e0.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/0ab3b8cb971938b4623bd46b180f4f9d35e2615f3a625a10c60b91b60bdc97e0.jpg)

![0eca91ed25dc9269cb009cd35cd25f8fc6f5e05442e5df19377ab79936ca74d8.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/0eca91ed25dc9269cb009cd35cd25f8fc6f5e05442e5df19377ab79936ca74d8.jpg)

![114822c5266689c41436a8e0de64fcdcaffd2c72f53996ae51fe2b5c1cba6af9.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/114822c5266689c41436a8e0de64fcdcaffd2c72f53996ae51fe2b5c1cba6af9.jpg)

![118ab874ad813e9d5e516e6bb15fadef47de54bee2029c1829030935aedb5cb8.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/118ab874ad813e9d5e516e6bb15fadef47de54bee2029c1829030935aedb5cb8.jpg)

![1256c93910b56d2b30234ef3fdcfe3864db5f15d6996fd67bcca6df9a87b3b1b.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/1256c93910b56d2b30234ef3fdcfe3864db5f15d6996fd67bcca6df9a87b3b1b.jpg)

![15eb291c1518d845a523adb42539dffd2b911ae429331af064de6ea72fad8353.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/15eb291c1518d845a523adb42539dffd2b911ae429331af064de6ea72fad8353.jpg)

![1af1b4a4f2d0af89de665c600c045aacb624225bc1ec5e01471ebebce75c7e2d.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/1af1b4a4f2d0af89de665c600c045aacb624225bc1ec5e01471ebebce75c7e2d.jpg)

![1cbec1d4935248444f1cf062cc116f32971d8da236db2ec0c1e7cc96db201343.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/1cbec1d4935248444f1cf062cc116f32971d8da236db2ec0c1e7cc96db201343.jpg)

![1f63824d17342b0eaaf167f674dc5081256f64b48dd4ad02f61b947c321ee0a9.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/1f63824d17342b0eaaf167f674dc5081256f64b48dd4ad02f61b947c321ee0a9.jpg)

![205825f0b99be387181c3cfac55e505efc7d2e6b034d2fa70c0766e16050644b.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/205825f0b99be387181c3cfac55e505efc7d2e6b034d2fa70c0766e16050644b.jpg)

![21a13db17302ad284345454be9e71f602ac093a48c36b6a9c5a10f17fd1855ed.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/21a13db17302ad284345454be9e71f602ac093a48c36b6a9c5a10f17fd1855ed.jpg)

![256f2e6fe8d02e470072611582f6cb054d81cfd36fb2b47c185ea3bf451cf83c.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/256f2e6fe8d02e470072611582f6cb054d81cfd36fb2b47c185ea3bf451cf83c.jpg)

![2a25adb9f45d57c305d26b1a6ddc21d5ac63e3cede4ac3497e6bf0dce98696c5.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/2a25adb9f45d57c305d26b1a6ddc21d5ac63e3cede4ac3497e6bf0dce98696c5.jpg)

![2a889ef565b054504c331d7b68cc0ec275d65fe0ad673325ebdc1f3d2ea10ab0.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/2a889ef565b054504c331d7b68cc0ec275d65fe0ad673325ebdc1f3d2ea10ab0.jpg)

![2ad1f74fa6060f16fb70b8d87596e0201c5703574bddb8326cf0d7a9dffce955.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/2ad1f74fa6060f16fb70b8d87596e0201c5703574bddb8326cf0d7a9dffce955.jpg)

![2addb38ae1e59abfe1a5cc1b0b3cac2969fee411b45cbd9ca0f8aa0b9fd34722.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/2addb38ae1e59abfe1a5cc1b0b3cac2969fee411b45cbd9ca0f8aa0b9fd34722.jpg)

![2bd6e2fd76087b0038c2af2b117762c35f732c9f530bf1145384fb71e6a8fbfb.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/2bd6e2fd76087b0038c2af2b117762c35f732c9f530bf1145384fb71e6a8fbfb.jpg)

![2e043ce849e8e436480784b23aeb91e29f2594b586f25dd4041b5b0e2d7386f1.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/2e043ce849e8e436480784b23aeb91e29f2594b586f25dd4041b5b0e2d7386f1.jpg)

![2fc4f289bf13a7e9e9968a7eac59ca4760df38e262f7f32aeffc705838ed7b9e.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/2fc4f289bf13a7e9e9968a7eac59ca4760df38e262f7f32aeffc705838ed7b9e.jpg)

![35a19aa2d90d48569463c1795e47cfb8b3ee225ac0d8c936f57e162f57c51aab.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/35a19aa2d90d48569463c1795e47cfb8b3ee225ac0d8c936f57e162f57c51aab.jpg)

![366c6b9d450cc60068e91e15027cf080ba76b916fad48e9586c98db2dd7b494a.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/366c6b9d450cc60068e91e15027cf080ba76b916fad48e9586c98db2dd7b494a.jpg)

![3cf4714e99dca0c611103cb9e434086e9d8f617b20bce845c8bb0608c3b97402.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/3cf4714e99dca0c611103cb9e434086e9d8f617b20bce845c8bb0608c3b97402.jpg)

![40485ef040733f38bf517e69c657967bfb7a9e192330727130f512f468fe30ad.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/40485ef040733f38bf517e69c657967bfb7a9e192330727130f512f468fe30ad.jpg)

![404abdab7cfdc444c6c5309fd05263fb61c6449b18fecb0d891e88c33ee6a3f7.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/404abdab7cfdc444c6c5309fd05263fb61c6449b18fecb0d891e88c33ee6a3f7.jpg)

![424bd2cabe4b5e4000764d6c033e8c756498b2bd879ae10e354b51ff403e63ef.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/424bd2cabe4b5e4000764d6c033e8c756498b2bd879ae10e354b51ff403e63ef.jpg)

![45cd96f539cf7121eef18bd36819ee1bb9711d4799bba8dff7e24fdebe485f00.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/45cd96f539cf7121eef18bd36819ee1bb9711d4799bba8dff7e24fdebe485f00.jpg)

![462d87bacd8f16d6bc8b5bc25884041bfda4f60bc8d06372ac0a235c03a628b1.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/462d87bacd8f16d6bc8b5bc25884041bfda4f60bc8d06372ac0a235c03a628b1.jpg)

![47991a4e76401f64c7b65e71efa36d9fcc2e86c3168efd0cc3289f7a3083bd5d.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/47991a4e76401f64c7b65e71efa36d9fcc2e86c3168efd0cc3289f7a3083bd5d.jpg)

![4a3381dee62e65c2840dc4004a1e63ee737392e9c9f4d1c4342179557908eabe.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/4a3381dee62e65c2840dc4004a1e63ee737392e9c9f4d1c4342179557908eabe.jpg)

![4b7fd351a1a3f66d3357b0516f9b3b412cc5c00e069fc8b1b8682213fd56962f.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/4b7fd351a1a3f66d3357b0516f9b3b412cc5c00e069fc8b1b8682213fd56962f.jpg)

![4d2f98763559e91a8bdcf0a140a3c4a439fdf044c5ae3ce0b01a784a7086a317.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/4d2f98763559e91a8bdcf0a140a3c4a439fdf044c5ae3ce0b01a784a7086a317.jpg)

![4d85b60c674cfb50a6ee9c911035ec310c464654f796f06b8a2817bc7a9076e7.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/4d85b60c674cfb50a6ee9c911035ec310c464654f796f06b8a2817bc7a9076e7.jpg)

![4e0b9291adbf6ec9a915a948aa1b7ad3b767a6925f020888a939583784c1bdd8.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/4e0b9291adbf6ec9a915a948aa1b7ad3b767a6925f020888a939583784c1bdd8.jpg)

![4e8c49bfe076ba39b711961e8cd35283832930ec04520ecd715564e09d3885b1.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/4e8c49bfe076ba39b711961e8cd35283832930ec04520ecd715564e09d3885b1.jpg)

![519a1088d6d45397ec9b98ddc29f6aa3c547e52444acd47ef0c27d4ce29da844.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/519a1088d6d45397ec9b98ddc29f6aa3c547e52444acd47ef0c27d4ce29da844.jpg)

![52b8b35ec554d16dd104f4a8229bf09387ff6eab8e4c8b2ba9ef0c0d8a6a3422.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/52b8b35ec554d16dd104f4a8229bf09387ff6eab8e4c8b2ba9ef0c0d8a6a3422.jpg)

![56427c55ca4aee0f9776375f1b61a4c6c2e37876e1a3158e3d421496b4e28d71.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/56427c55ca4aee0f9776375f1b61a4c6c2e37876e1a3158e3d421496b4e28d71.jpg)

![5996416e12558434e4008e5c2de4976f43c509e45f87d112065c04b77ad07f92.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/5996416e12558434e4008e5c2de4976f43c509e45f87d112065c04b77ad07f92.jpg)

![5bfcbf4c133c4b0e518469b1f4f9204f045aa809a106b6f0bcb261e734d62db2.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/5bfcbf4c133c4b0e518469b1f4f9204f045aa809a106b6f0bcb261e734d62db2.jpg)

![5c57e8daa55ff36898286391c9f5e4b3a61fdace720c67bb06a9278118c7dcda.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/5c57e8daa55ff36898286391c9f5e4b3a61fdace720c67bb06a9278118c7dcda.jpg)

![5fab356d6b7ccacec6a7df0de3ac295958b89515b91413e37e3a68e62c455f14.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/5fab356d6b7ccacec6a7df0de3ac295958b89515b91413e37e3a68e62c455f14.jpg)

![6149f20bdeca2e860eae60d99134a0e61b2d5522e18dd326d769480324a253a0.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/6149f20bdeca2e860eae60d99134a0e61b2d5522e18dd326d769480324a253a0.jpg)

![65c9fb2bdd537a6f25032f1847d769bfd3b80107830fcda8fd8eea66db429741.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/65c9fb2bdd537a6f25032f1847d769bfd3b80107830fcda8fd8eea66db429741.jpg)

![6b3892919a1ed7d710d52fb9f52f8111b9b5e7efa11d3bc3690313d992f4aa8d.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/6b3892919a1ed7d710d52fb9f52f8111b9b5e7efa11d3bc3690313d992f4aa8d.jpg)

![6c394011425a33bccc2272d917009fb0664087d746ab04c1b9e9e56f2430d540.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/6c394011425a33bccc2272d917009fb0664087d746ab04c1b9e9e56f2430d540.jpg)

![6d6487a5333b6d151b2435198be1cc8f9c7f5bd26c4db3bdbbe88200f7272298.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/6d6487a5333b6d151b2435198be1cc8f9c7f5bd26c4db3bdbbe88200f7272298.jpg)

![6e9f1a233d7bcb67894b0d177ebc606ede2ee060f71244f04f8031d9c53e1ae8.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/6e9f1a233d7bcb67894b0d177ebc606ede2ee060f71244f04f8031d9c53e1ae8.jpg)

![6ed070de7f5cc1e87fdc6f0be185d080ca08f36ec5e9ec5930b922d0c1a27d39.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/6ed070de7f5cc1e87fdc6f0be185d080ca08f36ec5e9ec5930b922d0c1a27d39.jpg)

![75732ba03ccb7f394ebc82bef2b381907886c248bfa936b51153dbf1cfc75a52.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/75732ba03ccb7f394ebc82bef2b381907886c248bfa936b51153dbf1cfc75a52.jpg)

![76f25a52c8e3da747e977420c7c25c080882dd5a2e7f760e2fcc47798f416ec7.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/76f25a52c8e3da747e977420c7c25c080882dd5a2e7f760e2fcc47798f416ec7.jpg)

![7a47f2bb8fce72e7f9dc220e89357d2cbde2e8f968987801f73c323bef29023a.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/7a47f2bb8fce72e7f9dc220e89357d2cbde2e8f968987801f73c323bef29023a.jpg)

![81f05bb4e418c3f041421f7002c4f72db727faed3f7c2bf8b26b898878fc9c70.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/81f05bb4e418c3f041421f7002c4f72db727faed3f7c2bf8b26b898878fc9c70.jpg)

![8315825eddb5d4b0dde903275bd01e773386a9191c5b1b22e9f20a5006e4f1ff.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/8315825eddb5d4b0dde903275bd01e773386a9191c5b1b22e9f20a5006e4f1ff.jpg)

![83a8c6edcc99f6e26ee1df9e6f00d61403786c00e4f420032a6dd5b5732b52ed.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/83a8c6edcc99f6e26ee1df9e6f00d61403786c00e4f420032a6dd5b5732b52ed.jpg)

![83e45a8d4dab620b4accfd2dabdedaa2115debc8de670bd97def45aee358fcf9.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/83e45a8d4dab620b4accfd2dabdedaa2115debc8de670bd97def45aee358fcf9.jpg)

![85a2b923558a0e5893bfabfb6502813fe1c7f2f001a0c30b38327b3c7ae6f919.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/85a2b923558a0e5893bfabfb6502813fe1c7f2f001a0c30b38327b3c7ae6f919.jpg)

![86a2322b6dc41bc4fb86964291e61dc72bf525c3c158363932f1524ea5078951.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/86a2322b6dc41bc4fb86964291e61dc72bf525c3c158363932f1524ea5078951.jpg)

![89a67fe33b995e1d17265aae81d0221f444efc5ed8a216884260d0aeb4a04f40.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/89a67fe33b995e1d17265aae81d0221f444efc5ed8a216884260d0aeb4a04f40.jpg)

![8c1e311b3224345c7579ee1a3b2b768b1016bd150ab382ddd643ec66f738c0d7.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/8c1e311b3224345c7579ee1a3b2b768b1016bd150ab382ddd643ec66f738c0d7.jpg)

![91d4976ded8278bf79027bb40d75210660a214307c8ab9d30922a8ba4ad7e975.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/91d4976ded8278bf79027bb40d75210660a214307c8ab9d30922a8ba4ad7e975.jpg)

![92c40e43a37b9bff376ab3059f7cda4aa2e90cdbe42a07731eb4a6d3b859ac00.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/92c40e43a37b9bff376ab3059f7cda4aa2e90cdbe42a07731eb4a6d3b859ac00.jpg)

![9491e429130a85b0d4fbed57973a458a971feb334e631f33a0c070f44cea3d4a.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/9491e429130a85b0d4fbed57973a458a971feb334e631f33a0c070f44cea3d4a.jpg)

![95b3e4306b45baccaa4251a790462d81a730789cab8c34e277698ef7150a0e94.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/95b3e4306b45baccaa4251a790462d81a730789cab8c34e277698ef7150a0e94.jpg)

![9614b1a6d1c164dc61af719dddb1bb4cc06aa5b87e324b2060ca04aae9a81f27.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/9614b1a6d1c164dc61af719dddb1bb4cc06aa5b87e324b2060ca04aae9a81f27.jpg)

![9a4ad8d572f037c7643a5043e49b47ef58e374b4cc9e8da2c875f71084c0e9dc.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/9a4ad8d572f037c7643a5043e49b47ef58e374b4cc9e8da2c875f71084c0e9dc.jpg)

![9a7fcf8903dfecc2ab2d29dbf0514e2e2b4ea25d2d96b3258746d77f7156f129.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/9a7fcf8903dfecc2ab2d29dbf0514e2e2b4ea25d2d96b3258746d77f7156f129.jpg)

![9ce64a62d4daeb3b31c9dcdcf6ad197b40501ec5d08691424a171d23a2b1b6c4.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/9ce64a62d4daeb3b31c9dcdcf6ad197b40501ec5d08691424a171d23a2b1b6c4.jpg)

![9e9201da83695b0e009b29e04cfc3ef5f04fb38ec059343b73553b14ae8f497d.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/9e9201da83695b0e009b29e04cfc3ef5f04fb38ec059343b73553b14ae8f497d.jpg)

![9f634994d2477bfa10d34675c657f696fe91bb218b7c922cbe2738fd3b95f862.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/9f634994d2477bfa10d34675c657f696fe91bb218b7c922cbe2738fd3b95f862.jpg)

![a4a888b7b66e230d1b3a8311f0b741e3dfbef54ac6579f2cab7856717bd26a63.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/a4a888b7b66e230d1b3a8311f0b741e3dfbef54ac6579f2cab7856717bd26a63.jpg)

![a58426466cac24fb633d5075c5c8ed7e416ed0b1980bff19f5c55b9ec135c536.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/a58426466cac24fb633d5075c5c8ed7e416ed0b1980bff19f5c55b9ec135c536.jpg)

![aeb6b1436ffe1caf9f7efa43c69640a47939b73dc382d71f5d97b48e27f84c73.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/aeb6b1436ffe1caf9f7efa43c69640a47939b73dc382d71f5d97b48e27f84c73.jpg)

![b0f841db5a2dcf49c1d293a0ec1eb37202ccf5e47783102c67e87f18b4a97d45.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/b0f841db5a2dcf49c1d293a0ec1eb37202ccf5e47783102c67e87f18b4a97d45.jpg)

![b12c16034c5cca53be60f8a08ee9bfc09bde9ac0b8eb0c24aacab4d9124e490a.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/b12c16034c5cca53be60f8a08ee9bfc09bde9ac0b8eb0c24aacab4d9124e490a.jpg)

![b82e39302c1f6a28918f312679b3b9c3d82eba5f8ce408e302b920e2ee5f968f.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/b82e39302c1f6a28918f312679b3b9c3d82eba5f8ce408e302b920e2ee5f968f.jpg)

![bb99bb90f3d08c3605a930fafe8b4f7baf27f09c812cfa7f88528d4a406b989c.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/bb99bb90f3d08c3605a930fafe8b4f7baf27f09c812cfa7f88528d4a406b989c.jpg)

![be1c9fa671961a98cba6f41c301965252c72a8389adee5a15df909b491b80391.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/be1c9fa671961a98cba6f41c301965252c72a8389adee5a15df909b491b80391.jpg)

![c42afbd3cd2a18f90abfda7a79013ac0a462aca813c2aba344dbc26c5ba34358.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/c42afbd3cd2a18f90abfda7a79013ac0a462aca813c2aba344dbc26c5ba34358.jpg)

![c5c3e837b3416de902eed1ba5d88a78507c67bc682cf5fab06c132630acb9cac.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/c5c3e837b3416de902eed1ba5d88a78507c67bc682cf5fab06c132630acb9cac.jpg)

![c95f6a270f8879ff9ad4a5c498f8d4704d5c926177e7932965c7f05aecad9add.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/c95f6a270f8879ff9ad4a5c498f8d4704d5c926177e7932965c7f05aecad9add.jpg)

![cd4396d14855bdbc1e251f617933652f3dabdabd10074d96c0c536191b6d7a2a.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/cd4396d14855bdbc1e251f617933652f3dabdabd10074d96c0c536191b6d7a2a.jpg)

![d0dc074b49b3e613993a712ddb0b0de7f10c1074163c2b39610a365da0588617.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/d0dc074b49b3e613993a712ddb0b0de7f10c1074163c2b39610a365da0588617.jpg)

![d5d6113a1f82e1ff94334df31d3c4315db8a502cfb1bf2c089dcf46fd001b829.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/d5d6113a1f82e1ff94334df31d3c4315db8a502cfb1bf2c089dcf46fd001b829.jpg)

![d83fb9f6ae09d8236754ae757dbed3dcb1a9be85088e1932fcfadf8857f0e89e.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/d83fb9f6ae09d8236754ae757dbed3dcb1a9be85088e1932fcfadf8857f0e89e.jpg)

![dba1d6726daa1495521b6b5cc086b06ff59c285a66a1e717da4615e526c0d4f9.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/dba1d6726daa1495521b6b5cc086b06ff59c285a66a1e717da4615e526c0d4f9.jpg)

![de7446d41db3764101baff9b15902ec9a790d2447c4c53772705ba5f445ac250.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/de7446d41db3764101baff9b15902ec9a790d2447c4c53772705ba5f445ac250.jpg)

![e003679d2610d934e208d9870c0ac3735c06b619e10f80bde07de44dadb39b2d.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/e003679d2610d934e208d9870c0ac3735c06b619e10f80bde07de44dadb39b2d.jpg)

![e05f468393c1fccaf7bfbfa0c54f516857d4d5cd38d5e0fc276b80103c35548c.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/e05f468393c1fccaf7bfbfa0c54f516857d4d5cd38d5e0fc276b80103c35548c.jpg)

![e1478ddc531d083d14a37a51c224eab8e917f31d0b9f8c8d5d58ff447e9dd11a.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/e1478ddc531d083d14a37a51c224eab8e917f31d0b9f8c8d5d58ff447e9dd11a.jpg)

![e1fb31d8a89e87f4f3a4eb32cb0a2e73c95da80275f94b778c16c430d089ad55.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/e1fb31d8a89e87f4f3a4eb32cb0a2e73c95da80275f94b778c16c430d089ad55.jpg)

![e82226de5c130dca7f5a7840a123a1d1362216f4b3ad00da3148f8a0a6dcf453.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/e82226de5c130dca7f5a7840a123a1d1362216f4b3ad00da3148f8a0a6dcf453.jpg)

![ec0e388f404c834d4ef4eb10f22deb6df62dcd6bae3b7c657a463d15406d936a.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/ec0e388f404c834d4ef4eb10f22deb6df62dcd6bae3b7c657a463d15406d936a.jpg)

![eea65483bff99e4737bcc7cda685108ac4b83142e174b05079bde1b50c1b5ac4.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/eea65483bff99e4737bcc7cda685108ac4b83142e174b05079bde1b50c1b5ac4.jpg)

![ef01f3a29b6e39a0eb1435b5a4488c1940f39f70fb61171b76ac84b9931f9cb7.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/ef01f3a29b6e39a0eb1435b5a4488c1940f39f70fb61171b76ac84b9931f9cb7.jpg)

![f2875dec623f3c5d141a86fcfbb9012148dc0d04a4a36c90a67ea1e9b1e5d34e.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/f2875dec623f3c5d141a86fcfbb9012148dc0d04a4a36c90a67ea1e9b1e5d34e.jpg)

![f584d32344e23928ec8662c939f440d064d63d9a146f4415a3e131bdd1d640e5.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/f584d32344e23928ec8662c939f440d064d63d9a146f4415a3e131bdd1d640e5.jpg)

![f7f92d171ecdd56fd1d631c9c0f251bd3f7a14a260009da37e61cf0e873191bf.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/f7f92d171ecdd56fd1d631c9c0f251bd3f7a14a260009da37e61cf0e873191bf.jpg)

![fcae6cabca898350a27089dcafcd0880ee9708b8f2ba9170c9c99560780e12cf.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/fcae6cabca898350a27089dcafcd0880ee9708b8f2ba9170c9c99560780e12cf.jpg)

![fcaf4e22d0e78adbf35f04c2f78fe9873f2f268a778f8103594869988038ff0b.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/images/fcaf4e22d0e78adbf35f04c2f78fe9873f2f268a778f8103594869988038ff0b.jpg)

### Tables

![1a748a5092a8d266a55f427c0555d29ddd2935591094a458286a88770e4c59c1.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/tables/1a748a5092a8d266a55f427c0555d29ddd2935591094a458286a88770e4c59c1.jpg)

![5ca40f7fa9a216fcacab10b84ae8889bc998a27dbca80fed04dd05b49f7459b3.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/tables/5ca40f7fa9a216fcacab10b84ae8889bc998a27dbca80fed04dd05b49f7459b3.jpg)

![79cf754429be692999966ed9f2f86b46c1fd0d1c2a9f93b8ac6743e13466c4bb.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/tables/79cf754429be692999966ed9f2f86b46c1fd0d1c2a9f93b8ac6743e13466c4bb.jpg)

![f5bf322a283cf99dde72b441b24cce3cb724336a51189ce96135e4829e10404c.jpg](../iclr_results/2348_MovieDreamer_%20Hierarchical%20Generation%20for%20Coherent%20Long%20Visual%20Sequences/tables/f5bf322a283cf99dde72b441b24cce3cb724336a51189ce96135e4829e10404c.jpg)

## Model-agnostic meta-learners for estimating heterogeneous treatment effects over time


### Images

![0a78b0fb3b98dda7f25498cebb3b2c6ac22e550ee857a20730d1c60053116a74.jpg](../iclr_results/2349_Model-agnostic%20meta-learners%20for%20estimating%20heterogeneous%20treatment%20effects%20over%20time/images/0a78b0fb3b98dda7f25498cebb3b2c6ac22e550ee857a20730d1c60053116a74.jpg)

![8acd757d121b174e363fa71cc0309c63964aeb859e0cd81e5bec3a0b958d1463.jpg](../iclr_results/2349_Model-agnostic%20meta-learners%20for%20estimating%20heterogeneous%20treatment%20effects%20over%20time/images/8acd757d121b174e363fa71cc0309c63964aeb859e0cd81e5bec3a0b958d1463.jpg)

![bdb1d6e36ce314cd1f2520320ac80be77d80bca913839ff28610eac2ccce60f9.jpg](../iclr_results/2349_Model-agnostic%20meta-learners%20for%20estimating%20heterogeneous%20treatment%20effects%20over%20time/images/bdb1d6e36ce314cd1f2520320ac80be77d80bca913839ff28610eac2ccce60f9.jpg)

![ed30dfa836e6a7b35c0c60607a95fca7f8807424e718f91a59d44ede6ba23b21.jpg](../iclr_results/2349_Model-agnostic%20meta-learners%20for%20estimating%20heterogeneous%20treatment%20effects%20over%20time/images/ed30dfa836e6a7b35c0c60607a95fca7f8807424e718f91a59d44ede6ba23b21.jpg)

### Tables

![021e9543aebb9b4a7c916be5e43a7e9e30415ce9c2adb6a9a2635bb1aa06dcff.jpg](../iclr_results/2349_Model-agnostic%20meta-learners%20for%20estimating%20heterogeneous%20treatment%20effects%20over%20time/tables/021e9543aebb9b4a7c916be5e43a7e9e30415ce9c2adb6a9a2635bb1aa06dcff.jpg)

![558343937013f44f01bd38b0bd3539de8790d037b76cba5f0fba4de7cfcf4a49.jpg](../iclr_results/2349_Model-agnostic%20meta-learners%20for%20estimating%20heterogeneous%20treatment%20effects%20over%20time/tables/558343937013f44f01bd38b0bd3539de8790d037b76cba5f0fba4de7cfcf4a49.jpg)

![659571903140bd63629c64817e5cb64b68670df312b9acb0b17274ba015761e7.jpg](../iclr_results/2349_Model-agnostic%20meta-learners%20for%20estimating%20heterogeneous%20treatment%20effects%20over%20time/tables/659571903140bd63629c64817e5cb64b68670df312b9acb0b17274ba015761e7.jpg)

![75a8e13c5139f1f0dee8ae69e2a405d976955cbd344ea0fc633b2762f6e5f562.jpg](../iclr_results/2349_Model-agnostic%20meta-learners%20for%20estimating%20heterogeneous%20treatment%20effects%20over%20time/tables/75a8e13c5139f1f0dee8ae69e2a405d976955cbd344ea0fc633b2762f6e5f562.jpg)

![a35d07e283cb987f07e2e3e44c63153a7286460aaab3371a1796a86180e318f6.jpg](../iclr_results/2349_Model-agnostic%20meta-learners%20for%20estimating%20heterogeneous%20treatment%20effects%20over%20time/tables/a35d07e283cb987f07e2e3e44c63153a7286460aaab3371a1796a86180e318f6.jpg)

![a920051f320d28a36cda8a6a8f20bfaf09b1ddacbf11921ea4dd08b6d9f7e3f9.jpg](../iclr_results/2349_Model-agnostic%20meta-learners%20for%20estimating%20heterogeneous%20treatment%20effects%20over%20time/tables/a920051f320d28a36cda8a6a8f20bfaf09b1ddacbf11921ea4dd08b6d9f7e3f9.jpg)

![ccaa88be97e8d91b127825ea4f7dd8a7fbf2c96fe52baa543b89341a4f68f3ae.jpg](../iclr_results/2349_Model-agnostic%20meta-learners%20for%20estimating%20heterogeneous%20treatment%20effects%20over%20time/tables/ccaa88be97e8d91b127825ea4f7dd8a7fbf2c96fe52baa543b89341a4f68f3ae.jpg)

## Unleashing the Potential of Vision-Language Pre-Training for 3D Zero-Shot Lesion Segmentation via Mask-Attribute Alignment


### Images

![37ee24d0dd0960be5dfe0c2de1345eac78363ae6d9bf26eb195f7abad0071593.jpg](../iclr_results/2350_Unleashing%20the%20Potential%20of%20Vision-Language%20Pre-Training%20for%203D%20Zero-Shot%20Lesion%20Segmentation%20via%20Ma/images/37ee24d0dd0960be5dfe0c2de1345eac78363ae6d9bf26eb195f7abad0071593.jpg)

![5559c9c88215b0cd55a89c382dde48daa27e65a2da72ca671940cd26b82d1e9e.jpg](../iclr_results/2350_Unleashing%20the%20Potential%20of%20Vision-Language%20Pre-Training%20for%203D%20Zero-Shot%20Lesion%20Segmentation%20via%20Ma/images/5559c9c88215b0cd55a89c382dde48daa27e65a2da72ca671940cd26b82d1e9e.jpg)

![753443df543c8447a451942b8807517aafea349678bb440a30cd83a8746381a0.jpg](../iclr_results/2350_Unleashing%20the%20Potential%20of%20Vision-Language%20Pre-Training%20for%203D%20Zero-Shot%20Lesion%20Segmentation%20via%20Ma/images/753443df543c8447a451942b8807517aafea349678bb440a30cd83a8746381a0.jpg)

![7f8911b52d196caf89699c0422b1aad0b59ce50df27e958fce19efdb9f47bfff.jpg](../iclr_results/2350_Unleashing%20the%20Potential%20of%20Vision-Language%20Pre-Training%20for%203D%20Zero-Shot%20Lesion%20Segmentation%20via%20Ma/images/7f8911b52d196caf89699c0422b1aad0b59ce50df27e958fce19efdb9f47bfff.jpg)

![9ca246d943accc1f25cb4a3313ee8edce24ec4052187ecfbacaf909416650060.jpg](../iclr_results/2350_Unleashing%20the%20Potential%20of%20Vision-Language%20Pre-Training%20for%203D%20Zero-Shot%20Lesion%20Segmentation%20via%20Ma/images/9ca246d943accc1f25cb4a3313ee8edce24ec4052187ecfbacaf909416650060.jpg)

![a03b8935b77e6e28516392741132b496d15ead3657d980ddd801411577bad8e5.jpg](../iclr_results/2350_Unleashing%20the%20Potential%20of%20Vision-Language%20Pre-Training%20for%203D%20Zero-Shot%20Lesion%20Segmentation%20via%20Ma/images/a03b8935b77e6e28516392741132b496d15ead3657d980ddd801411577bad8e5.jpg)

![a7980160eeee56aff837dab7d2b2ab45e5dd4bbe7f87e57ec664ed26bc671048.jpg](../iclr_results/2350_Unleashing%20the%20Potential%20of%20Vision-Language%20Pre-Training%20for%203D%20Zero-Shot%20Lesion%20Segmentation%20via%20Ma/images/a7980160eeee56aff837dab7d2b2ab45e5dd4bbe7f87e57ec664ed26bc671048.jpg)

![a863bed0278f6f69cf92de8344783925a0bebb6f1dfa3b6560e0ab30b579e709.jpg](../iclr_results/2350_Unleashing%20the%20Potential%20of%20Vision-Language%20Pre-Training%20for%203D%20Zero-Shot%20Lesion%20Segmentation%20via%20Ma/images/a863bed0278f6f69cf92de8344783925a0bebb6f1dfa3b6560e0ab30b579e709.jpg)

![c657f6a26888825a2335fc1c28f2c7b62aa0cad2d5143007c6bb5c174e160595.jpg](../iclr_results/2350_Unleashing%20the%20Potential%20of%20Vision-Language%20Pre-Training%20for%203D%20Zero-Shot%20Lesion%20Segmentation%20via%20Ma/images/c657f6a26888825a2335fc1c28f2c7b62aa0cad2d5143007c6bb5c174e160595.jpg)

![d3be2fa5a1bddef3e7cab0e97a71d2159496139f7a9985f31f2547af09cef467.jpg](../iclr_results/2350_Unleashing%20the%20Potential%20of%20Vision-Language%20Pre-Training%20for%203D%20Zero-Shot%20Lesion%20Segmentation%20via%20Ma/images/d3be2fa5a1bddef3e7cab0e97a71d2159496139f7a9985f31f2547af09cef467.jpg)

![e641ac38418ae968c0657c480b7658e5cbf40a84b560a75746c98c7fc6edded1.jpg](../iclr_results/2350_Unleashing%20the%20Potential%20of%20Vision-Language%20Pre-Training%20for%203D%20Zero-Shot%20Lesion%20Segmentation%20via%20Ma/images/e641ac38418ae968c0657c480b7658e5cbf40a84b560a75746c98c7fc6edded1.jpg)

![f7afd9862b23ec0d190d68a8d10bf2751c87d7a60d6df3becdf19dfdcf313345.jpg](../iclr_results/2350_Unleashing%20the%20Potential%20of%20Vision-Language%20Pre-Training%20for%203D%20Zero-Shot%20Lesion%20Segmentation%20via%20Ma/images/f7afd9862b23ec0d190d68a8d10bf2751c87d7a60d6df3becdf19dfdcf313345.jpg)

### Tables

![3a9dbf2d9775559b893065091af9656823175d9da9d105b6c1a80fa0595c3ae7.jpg](../iclr_results/2350_Unleashing%20the%20Potential%20of%20Vision-Language%20Pre-Training%20for%203D%20Zero-Shot%20Lesion%20Segmentation%20via%20Ma/tables/3a9dbf2d9775559b893065091af9656823175d9da9d105b6c1a80fa0595c3ae7.jpg)

![46570018a1ef2870208aa8a6404c7b992e30e01cd0643a037f8df207cc2640f4.jpg](../iclr_results/2350_Unleashing%20the%20Potential%20of%20Vision-Language%20Pre-Training%20for%203D%20Zero-Shot%20Lesion%20Segmentation%20via%20Ma/tables/46570018a1ef2870208aa8a6404c7b992e30e01cd0643a037f8df207cc2640f4.jpg)

![4afd67986157fc9eb5fad157b34e7a73649b372f61dd9f61bb4940c9162714d5.jpg](../iclr_results/2350_Unleashing%20the%20Potential%20of%20Vision-Language%20Pre-Training%20for%203D%20Zero-Shot%20Lesion%20Segmentation%20via%20Ma/tables/4afd67986157fc9eb5fad157b34e7a73649b372f61dd9f61bb4940c9162714d5.jpg)

![4ee4ae639196b8fd53b8dfd20901e40b7ee4d51ffa3ad46f593747eaa60e825b.jpg](../iclr_results/2350_Unleashing%20the%20Potential%20of%20Vision-Language%20Pre-Training%20for%203D%20Zero-Shot%20Lesion%20Segmentation%20via%20Ma/tables/4ee4ae639196b8fd53b8dfd20901e40b7ee4d51ffa3ad46f593747eaa60e825b.jpg)

![5c44769433d511b0add1d590ffebdd33df32e05767656b2938f1b01f4647dad5.jpg](../iclr_results/2350_Unleashing%20the%20Potential%20of%20Vision-Language%20Pre-Training%20for%203D%20Zero-Shot%20Lesion%20Segmentation%20via%20Ma/tables/5c44769433d511b0add1d590ffebdd33df32e05767656b2938f1b01f4647dad5.jpg)

![773de23b52dd4dfdf3ffc9849cd1544d43eb334694b968f0b4ee2804c524731a.jpg](../iclr_results/2350_Unleashing%20the%20Potential%20of%20Vision-Language%20Pre-Training%20for%203D%20Zero-Shot%20Lesion%20Segmentation%20via%20Ma/tables/773de23b52dd4dfdf3ffc9849cd1544d43eb334694b968f0b4ee2804c524731a.jpg)

![7ec96618af3235ecb4baa94b06bc702fb4ec7f04563874ee0113abec538507da.jpg](../iclr_results/2350_Unleashing%20the%20Potential%20of%20Vision-Language%20Pre-Training%20for%203D%20Zero-Shot%20Lesion%20Segmentation%20via%20Ma/tables/7ec96618af3235ecb4baa94b06bc702fb4ec7f04563874ee0113abec538507da.jpg)

![8d3dc486664b9806bc43d024b00d05337d0faade77e7ce736709be2a8a5e061c.jpg](../iclr_results/2350_Unleashing%20the%20Potential%20of%20Vision-Language%20Pre-Training%20for%203D%20Zero-Shot%20Lesion%20Segmentation%20via%20Ma/tables/8d3dc486664b9806bc43d024b00d05337d0faade77e7ce736709be2a8a5e061c.jpg)

![b0e0c37d70524ed9079858cf90ae75e867b0bb3973327a5b42f064d8964e38d5.jpg](../iclr_results/2350_Unleashing%20the%20Potential%20of%20Vision-Language%20Pre-Training%20for%203D%20Zero-Shot%20Lesion%20Segmentation%20via%20Ma/tables/b0e0c37d70524ed9079858cf90ae75e867b0bb3973327a5b42f064d8964e38d5.jpg)

![bae9e2689a54791e107ce405fbd5501a0749dafc2733390d044a37e149cbfc17.jpg](../iclr_results/2350_Unleashing%20the%20Potential%20of%20Vision-Language%20Pre-Training%20for%203D%20Zero-Shot%20Lesion%20Segmentation%20via%20Ma/tables/bae9e2689a54791e107ce405fbd5501a0749dafc2733390d044a37e149cbfc17.jpg)

![c13631095a24cbabe2be10bf592fdf7b24eb2cadf5163a7675372faed1bfa92a.jpg](../iclr_results/2350_Unleashing%20the%20Potential%20of%20Vision-Language%20Pre-Training%20for%203D%20Zero-Shot%20Lesion%20Segmentation%20via%20Ma/tables/c13631095a24cbabe2be10bf592fdf7b24eb2cadf5163a7675372faed1bfa92a.jpg)

## State Space Models are Provably Comparable to Transformers in Dynamic Token Selection


### Images

![07ae447252308abaecd7f51d662a68db6405277a050a7ff9d604e94d136d25ab.jpg](../iclr_results/2351_State%20Space%20Models%20are%20Provably%20Comparable%20to%20Transformers%20in%20Dynamic%20Token%20Selection/images/07ae447252308abaecd7f51d662a68db6405277a050a7ff9d604e94d136d25ab.jpg)

![0a5863fbd9dfd3bfd7a2235f048ff95ee7003ee4061a0cf988d690f70ef48fda.jpg](../iclr_results/2351_State%20Space%20Models%20are%20Provably%20Comparable%20to%20Transformers%20in%20Dynamic%20Token%20Selection/images/0a5863fbd9dfd3bfd7a2235f048ff95ee7003ee4061a0cf988d690f70ef48fda.jpg)

![0e1b377407481d494fc29614b50dfe17c70990438cee952540927694c68bb2ae.jpg](../iclr_results/2351_State%20Space%20Models%20are%20Provably%20Comparable%20to%20Transformers%20in%20Dynamic%20Token%20Selection/images/0e1b377407481d494fc29614b50dfe17c70990438cee952540927694c68bb2ae.jpg)

![26d1ed86578485da258eef510a21eec28141270f0971b5a99d3ee08be1d50757.jpg](../iclr_results/2351_State%20Space%20Models%20are%20Provably%20Comparable%20to%20Transformers%20in%20Dynamic%20Token%20Selection/images/26d1ed86578485da258eef510a21eec28141270f0971b5a99d3ee08be1d50757.jpg)

![b4d639ff2607af79f6259e22d98e8b71ff56d2a9cdb65fbe046b89d6d61e9a8e.jpg](../iclr_results/2351_State%20Space%20Models%20are%20Provably%20Comparable%20to%20Transformers%20in%20Dynamic%20Token%20Selection/images/b4d639ff2607af79f6259e22d98e8b71ff56d2a9cdb65fbe046b89d6d61e9a8e.jpg)

![b657b91a4efeaa6d66d3ff311c4980246b6cc17ab89b7fe94a4aac0ab2b7305d.jpg](../iclr_results/2351_State%20Space%20Models%20are%20Provably%20Comparable%20to%20Transformers%20in%20Dynamic%20Token%20Selection/images/b657b91a4efeaa6d66d3ff311c4980246b6cc17ab89b7fe94a4aac0ab2b7305d.jpg)

## ImageFolder: Autoregressive Image Generation with Folded Tokens


### Images

![0d60f4bd6b525ada8619f16692d9e5718046827d9c3852947b3be611470bf6dd.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/images/0d60f4bd6b525ada8619f16692d9e5718046827d9c3852947b3be611470bf6dd.jpg)

![179b9177faef5010f3d2b5143d21eb35713a99a18ecf869c4eae92bc85379bc1.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/images/179b9177faef5010f3d2b5143d21eb35713a99a18ecf869c4eae92bc85379bc1.jpg)

![1a340ca3ef983f9109ad4c8c156c386a1aba93e7698165278e3b422f743fb957.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/images/1a340ca3ef983f9109ad4c8c156c386a1aba93e7698165278e3b422f743fb957.jpg)

![375e9555c4485d9f16a328dde0c81f53e60cddf2a436f8a5207b38b3c955e791.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/images/375e9555c4485d9f16a328dde0c81f53e60cddf2a436f8a5207b38b3c955e791.jpg)

![48db8941a289b7bf8eef569b600a105744f998e0ffceedda36f4a0c450e39c73.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/images/48db8941a289b7bf8eef569b600a105744f998e0ffceedda36f4a0c450e39c73.jpg)

![563c1d4336cbc9617aec9e1541dc8d0731d46ccd456c4a9569fe948284908708.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/images/563c1d4336cbc9617aec9e1541dc8d0731d46ccd456c4a9569fe948284908708.jpg)

![648e461172062f47e5e4e4093ec9864dc667a2f75afeccb06633c11e7c1f7833.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/images/648e461172062f47e5e4e4093ec9864dc667a2f75afeccb06633c11e7c1f7833.jpg)

![a032bad54aed3fe1abb5865004b3dec1f354b31929e9cb8aa91e01e5ec97b366.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/images/a032bad54aed3fe1abb5865004b3dec1f354b31929e9cb8aa91e01e5ec97b366.jpg)

![b8b587ddde3d7e9242ca9bb770530a3cc38b9a22ab6070c389563832be5a3735.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/images/b8b587ddde3d7e9242ca9bb770530a3cc38b9a22ab6070c389563832be5a3735.jpg)

![b9e5f45ee7f5539befde0fed64f2df9e3fca1f9d354f0ae29c45eca84d741d4a.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/images/b9e5f45ee7f5539befde0fed64f2df9e3fca1f9d354f0ae29c45eca84d741d4a.jpg)

![ce4a886c47adf05b30f507baffb96897714264794174dbc2cd0bfeaefd0def4d.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/images/ce4a886c47adf05b30f507baffb96897714264794174dbc2cd0bfeaefd0def4d.jpg)

![eacb4a76537e1598a080792f4cee9bd4ce291846b7d0689f0e6db2f001c70045.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/images/eacb4a76537e1598a080792f4cee9bd4ce291846b7d0689f0e6db2f001c70045.jpg)

![ed81b5aad747a965e6e0f3cd2ff867e30d4ae1248154d36b29458e7ba4f65a7b.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/images/ed81b5aad747a965e6e0f3cd2ff867e30d4ae1248154d36b29458e7ba4f65a7b.jpg)

![f688867bed8437273e18fb12342256acb7d566c79b7384770f61cff708fc53d3.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/images/f688867bed8437273e18fb12342256acb7d566c79b7384770f61cff708fc53d3.jpg)

### Tables

![0f7bb278e2fd3cb697523250b8b9f45cd6b365d2c3bd65a58a01cdf19754b209.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/tables/0f7bb278e2fd3cb697523250b8b9f45cd6b365d2c3bd65a58a01cdf19754b209.jpg)

![322dcc951dc00d8087addc27d0ae02d4bb8ff3844f566c3bcfad770b7d155f1b.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/tables/322dcc951dc00d8087addc27d0ae02d4bb8ff3844f566c3bcfad770b7d155f1b.jpg)

![4612e41c9ef44502ae429b8a12ae3fdf77a2a77a9a2848f64e1b7cdd49f2be07.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/tables/4612e41c9ef44502ae429b8a12ae3fdf77a2a77a9a2848f64e1b7cdd49f2be07.jpg)

![5633b76c2e5a82553c5f922a1ec4237febe8d3b7e7772ab832b30f15a2393437.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/tables/5633b76c2e5a82553c5f922a1ec4237febe8d3b7e7772ab832b30f15a2393437.jpg)

![727f1f4b6d35929df6c1324c3bab160502049f33cc53e97f7f0dc40bd38773b1.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/tables/727f1f4b6d35929df6c1324c3bab160502049f33cc53e97f7f0dc40bd38773b1.jpg)

![8d8c6bfa50cd175c20f740eb27ae79d12f85ba2651a4768c634e02abcc3a1648.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/tables/8d8c6bfa50cd175c20f740eb27ae79d12f85ba2651a4768c634e02abcc3a1648.jpg)

![9545119d17141e30388d27d2a81091596d833fef1a4e9c76da33bdec7669a9c9.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/tables/9545119d17141e30388d27d2a81091596d833fef1a4e9c76da33bdec7669a9c9.jpg)

![ae56a0ac9b4d6eb661c2e5fc706d97607f6c2a4435f4c5c29d751181bb0be248.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/tables/ae56a0ac9b4d6eb661c2e5fc706d97607f6c2a4435f4c5c29d751181bb0be248.jpg)

![c8c8e681ae7b223d16fab049e0b22b49d8aa28e77a2ead1608556b9e305aefa9.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/tables/c8c8e681ae7b223d16fab049e0b22b49d8aa28e77a2ead1608556b9e305aefa9.jpg)

![e7cae283a01e31aa1d1893bd53be609073b3449de521e601f256e4f11a112c9c.jpg](../iclr_results/2352_ImageFolder_%20Autoregressive%20Image%20Generation%20with%20Folded%20Tokens/tables/e7cae283a01e31aa1d1893bd53be609073b3449de521e601f256e4f11a112c9c.jpg)

## Model Equality Testing: Which Model is this API Serving?


### Images

![062d80bcc8620275f64e166893e7748fdb89261a63ced1dbb1d23dba697e02b2.jpg](../iclr_results/2353_Model%20Equality%20Testing_%20Which%20Model%20is%20this%20API%20Serving_/images/062d80bcc8620275f64e166893e7748fdb89261a63ced1dbb1d23dba697e02b2.jpg)

![5f67958e2bdf4031fe8878ca4f1f6f913bdc0f55b4c33bf43cd11ed7f9c07c9e.jpg](../iclr_results/2353_Model%20Equality%20Testing_%20Which%20Model%20is%20this%20API%20Serving_/images/5f67958e2bdf4031fe8878ca4f1f6f913bdc0f55b4c33bf43cd11ed7f9c07c9e.jpg)

![9f43041c7d0fe8f71fd110ef165b0adaae9c531e38914d1290dcdfed89d2b3a8.jpg](../iclr_results/2353_Model%20Equality%20Testing_%20Which%20Model%20is%20this%20API%20Serving_/images/9f43041c7d0fe8f71fd110ef165b0adaae9c531e38914d1290dcdfed89d2b3a8.jpg)

![c3f28df6ffbb2b0132681fa25a8943020c68fbd2a4fe9e3ac7e344745caf5dc6.jpg](../iclr_results/2353_Model%20Equality%20Testing_%20Which%20Model%20is%20this%20API%20Serving_/images/c3f28df6ffbb2b0132681fa25a8943020c68fbd2a4fe9e3ac7e344745caf5dc6.jpg)

![f45a335c369eb924fae876231037f21a2a521b942c1ace84cc352216b411097e.jpg](../iclr_results/2353_Model%20Equality%20Testing_%20Which%20Model%20is%20this%20API%20Serving_/images/f45a335c369eb924fae876231037f21a2a521b942c1ace84cc352216b411097e.jpg)

### Tables

![1d05cad21ea5ccb39524f9379902d41c282433a3779ef2fb173c0115afdc6743.jpg](../iclr_results/2353_Model%20Equality%20Testing_%20Which%20Model%20is%20this%20API%20Serving_/tables/1d05cad21ea5ccb39524f9379902d41c282433a3779ef2fb173c0115afdc6743.jpg)

![1d3b27adb4f943b7723079c6fb500fb9fc4cde7bb2901b3a935bb5b356f54e1f.jpg](../iclr_results/2353_Model%20Equality%20Testing_%20Which%20Model%20is%20this%20API%20Serving_/tables/1d3b27adb4f943b7723079c6fb500fb9fc4cde7bb2901b3a935bb5b356f54e1f.jpg)

![fd1d138f03890223a0e4489cbfc354e5ca59a0854062a88ccde8e6345f4c5c63.jpg](../iclr_results/2353_Model%20Equality%20Testing_%20Which%20Model%20is%20this%20API%20Serving_/tables/fd1d138f03890223a0e4489cbfc354e5ca59a0854062a88ccde8e6345f4c5c63.jpg)

## Demystifying Topological Message-Passing with Relational Structures: A Case Study on Oversquashing in Simplicial Message-Passing


### Images

![3afa7ae84c69cbb868a26f7e81b36eb199db014e8526e809a57a6b15b9e2d92c.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/images/3afa7ae84c69cbb868a26f7e81b36eb199db014e8526e809a57a6b15b9e2d92c.jpg)

![3b95e55635b97a608feca3c6dc263f298edefdf11e72f8f606b1f3ec97bd56b4.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/images/3b95e55635b97a608feca3c6dc263f298edefdf11e72f8f606b1f3ec97bd56b4.jpg)

![50c4dc6a4b7d845d2373c867734b9d49d5bd4442f23e8b92ee54c321bd600adf.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/images/50c4dc6a4b7d845d2373c867734b9d49d5bd4442f23e8b92ee54c321bd600adf.jpg)

![59ab74bc273dbce078254a62011a857b6c8899d3ff0521818dd3eb4bd3b7386e.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/images/59ab74bc273dbce078254a62011a857b6c8899d3ff0521818dd3eb4bd3b7386e.jpg)

![5e4737b48be13025cdd2aec96d1e22124e677b085bffa766f7e72ec6d19c3c73.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/images/5e4737b48be13025cdd2aec96d1e22124e677b085bffa766f7e72ec6d19c3c73.jpg)

![6462f0ff604b12b6f168126d4563265740c4c3b0676b0122bc7217c732d57d6d.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/images/6462f0ff604b12b6f168126d4563265740c4c3b0676b0122bc7217c732d57d6d.jpg)

![72cf456f936710334e535b743a9f2cefb524848aaa2502978c14b847c5d11cad.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/images/72cf456f936710334e535b743a9f2cefb524848aaa2502978c14b847c5d11cad.jpg)

![72f1d5e02b9a53783572b0d5045617097fb34a8543910ee9e85b9823ea89160d.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/images/72f1d5e02b9a53783572b0d5045617097fb34a8543910ee9e85b9823ea89160d.jpg)

![8750b1cdecaa3f14a68be0aa9b00c0908012ae488ba528692ac730d2c9153668.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/images/8750b1cdecaa3f14a68be0aa9b00c0908012ae488ba528692ac730d2c9153668.jpg)

![8f6bcaaa121008cc6e3b0d95bbfce11dd8adab61de46b3844eb57378f4696cc4.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/images/8f6bcaaa121008cc6e3b0d95bbfce11dd8adab61de46b3844eb57378f4696cc4.jpg)

![9b0ca92b2f2777d54918fb3f14ce25f28dc2753b1a54c9d2b8726bd89c03db47.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/images/9b0ca92b2f2777d54918fb3f14ce25f28dc2753b1a54c9d2b8726bd89c03db47.jpg)

![e02f23e85ac440ef9d435cd065bc03345f91b94663eac0a9cf2de786f7d2c9a4.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/images/e02f23e85ac440ef9d435cd065bc03345f91b94663eac0a9cf2de786f7d2c9a4.jpg)

![e179334b90da9dfcd206f1a42d98d2b08f116436118e2544193a31e047e033e0.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/images/e179334b90da9dfcd206f1a42d98d2b08f116436118e2544193a31e047e033e0.jpg)

![f990b3b8ac15ac0c414ec2648ba7f1b70f77b15481e7010f68c72ec29f8d768b.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/images/f990b3b8ac15ac0c414ec2648ba7f1b70f77b15481e7010f68c72ec29f8d768b.jpg)

### Tables

![03a40368c74bae0bb8ef8f7f7957a394b2e1477dca6dc492ca0d77897f66580d.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/tables/03a40368c74bae0bb8ef8f7f7957a394b2e1477dca6dc492ca0d77897f66580d.jpg)

![042b004945a7759296a637f185d606107c346b6696cc43fbf82acc96b5bf08d0.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/tables/042b004945a7759296a637f185d606107c346b6696cc43fbf82acc96b5bf08d0.jpg)

![11d4a5972e6d39b2473a0b6cac5c5a4a280f9667e6f3eba099ddabf4481bf168.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/tables/11d4a5972e6d39b2473a0b6cac5c5a4a280f9667e6f3eba099ddabf4481bf168.jpg)

![12781d1509449ef2374cd87d614e0e6517dfba96582039a2c00643587b3d6517.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/tables/12781d1509449ef2374cd87d614e0e6517dfba96582039a2c00643587b3d6517.jpg)

![201aa96d215a779608022428372a83c9480d8f1a12a39f53f3bbca4bc6ee0174.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/tables/201aa96d215a779608022428372a83c9480d8f1a12a39f53f3bbca4bc6ee0174.jpg)

![24d4c82b7d9af0576118f538f07a4ee76ecd16ae57f4629f38cbae73e7898a66.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/tables/24d4c82b7d9af0576118f538f07a4ee76ecd16ae57f4629f38cbae73e7898a66.jpg)

![3a36f2fc2423f18a77e0504f28b7f59b4d3a0a7635a628d1e25321906ae57e19.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/tables/3a36f2fc2423f18a77e0504f28b7f59b4d3a0a7635a628d1e25321906ae57e19.jpg)

![4130e77751463d2adba1b60e25232f1ad9688b551f7f6901dffc0d1320265dad.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/tables/4130e77751463d2adba1b60e25232f1ad9688b551f7f6901dffc0d1320265dad.jpg)

![643a37aba6583e12dcb8301991669002ed5eb7ae75d6b8bf3e9cbf456c96ce35.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/tables/643a37aba6583e12dcb8301991669002ed5eb7ae75d6b8bf3e9cbf456c96ce35.jpg)

![750e29861e87d82210f28344de374b1cfd89596306649785460c07e3fed7a629.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/tables/750e29861e87d82210f28344de374b1cfd89596306649785460c07e3fed7a629.jpg)

![7c5d6506cd5507eb5657e61b152b505824d23707137fac979532797ab539b06a.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/tables/7c5d6506cd5507eb5657e61b152b505824d23707137fac979532797ab539b06a.jpg)

![7e433f70248b11ff71715fa6b3f1d79ac4db4c44fb756460be9907899373ae21.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/tables/7e433f70248b11ff71715fa6b3f1d79ac4db4c44fb756460be9907899373ae21.jpg)

![7f5320bd47ff12d1da39736a59bd465ff54d04d77246cecc1c90a2a6b61b3395.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/tables/7f5320bd47ff12d1da39736a59bd465ff54d04d77246cecc1c90a2a6b61b3395.jpg)

![8bbd16e9dafa332d45f8c7e0f217858766a65dc8084d872b235df01f849e1924.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/tables/8bbd16e9dafa332d45f8c7e0f217858766a65dc8084d872b235df01f849e1924.jpg)

![9a53f61a0d67edfc3831f5a2cab6217c4eba7c4b5f1a8150a8d0b2fb799e46c2.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/tables/9a53f61a0d67edfc3831f5a2cab6217c4eba7c4b5f1a8150a8d0b2fb799e46c2.jpg)

![a1ca1b8434cb180099b85a955f445c78afb41d5a72c4857b8e033e8b5138c003.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/tables/a1ca1b8434cb180099b85a955f445c78afb41d5a72c4857b8e033e8b5138c003.jpg)

![a6e75e022df389507f3d4edcfa608275e0327fe8214691029dfbd1d36460ebab.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/tables/a6e75e022df389507f3d4edcfa608275e0327fe8214691029dfbd1d36460ebab.jpg)

![fe3857626cc381d0dfa44edbdea77cb790d734756d7ab450c1974369c652cd5e.jpg](../iclr_results/2354_Demystifying%20Topological%20Message-Passing%20with%20Relational%20Structures_%20A%20Case%20Study%20on%20Oversquashing%20i/tables/fe3857626cc381d0dfa44edbdea77cb790d734756d7ab450c1974369c652cd5e.jpg)

## From an LLM Swarm to a PDDL-empowered Hive: Planning Self-executed Instructions in a Multi-modal Jungle


### Images

![1229047333ef8ada646b79286b0e1cac98883426c8dfef9ee0eb1df529ffc848.jpg](../iclr_results/2355_From%20an%20LLM%20Swarm%20to%20a%20PDDL-empowered%20Hive_%20Planning%20Self-executed%20Instructions%20in%20a%20Multi-modal%20Jun/images/1229047333ef8ada646b79286b0e1cac98883426c8dfef9ee0eb1df529ffc848.jpg)

![3ce9b536bc52fab1238fd4cf877fe414e94d63869aa581a9b1b490fcfd3a9b2d.jpg](../iclr_results/2355_From%20an%20LLM%20Swarm%20to%20a%20PDDL-empowered%20Hive_%20Planning%20Self-executed%20Instructions%20in%20a%20Multi-modal%20Jun/images/3ce9b536bc52fab1238fd4cf877fe414e94d63869aa581a9b1b490fcfd3a9b2d.jpg)

![62442c5f4a7ec92f84c511c1053524639ca614728fa7b8257e0bceba5f81b0fd.jpg](../iclr_results/2355_From%20an%20LLM%20Swarm%20to%20a%20PDDL-empowered%20Hive_%20Planning%20Self-executed%20Instructions%20in%20a%20Multi-modal%20Jun/images/62442c5f4a7ec92f84c511c1053524639ca614728fa7b8257e0bceba5f81b0fd.jpg)

![7147a18c3d34a8b82e90c226d2b38b60dcbcc3f1720d3275d3d962297633c840.jpg](../iclr_results/2355_From%20an%20LLM%20Swarm%20to%20a%20PDDL-empowered%20Hive_%20Planning%20Self-executed%20Instructions%20in%20a%20Multi-modal%20Jun/images/7147a18c3d34a8b82e90c226d2b38b60dcbcc3f1720d3275d3d962297633c840.jpg)

![d50981940947708f08f11bd942dc4ed6f39f287529963634804e2c250c5b7285.jpg](../iclr_results/2355_From%20an%20LLM%20Swarm%20to%20a%20PDDL-empowered%20Hive_%20Planning%20Self-executed%20Instructions%20in%20a%20Multi-modal%20Jun/images/d50981940947708f08f11bd942dc4ed6f39f287529963634804e2c250c5b7285.jpg)

![e88463fd2df06e0303c45dbc1493b92613eac75b1e0fbec5b08f4dd51ccd30da.jpg](../iclr_results/2355_From%20an%20LLM%20Swarm%20to%20a%20PDDL-empowered%20Hive_%20Planning%20Self-executed%20Instructions%20in%20a%20Multi-modal%20Jun/images/e88463fd2df06e0303c45dbc1493b92613eac75b1e0fbec5b08f4dd51ccd30da.jpg)

### Tables

![13685d59f2a831854163a99c8782d7a1bfa37b9853d0ba7a0e52083462dd698d.jpg](../iclr_results/2355_From%20an%20LLM%20Swarm%20to%20a%20PDDL-empowered%20Hive_%20Planning%20Self-executed%20Instructions%20in%20a%20Multi-modal%20Jun/tables/13685d59f2a831854163a99c8782d7a1bfa37b9853d0ba7a0e52083462dd698d.jpg)

![282a947b08f64af2b868323397af9fd69eebedc0c20053d38f73cab7d734c928.jpg](../iclr_results/2355_From%20an%20LLM%20Swarm%20to%20a%20PDDL-empowered%20Hive_%20Planning%20Self-executed%20Instructions%20in%20a%20Multi-modal%20Jun/tables/282a947b08f64af2b868323397af9fd69eebedc0c20053d38f73cab7d734c928.jpg)

![3cf25a8b66cd174226ee80d2f965448544b1ab5ae9ea10dc3abb8d07f3aeadaa.jpg](../iclr_results/2355_From%20an%20LLM%20Swarm%20to%20a%20PDDL-empowered%20Hive_%20Planning%20Self-executed%20Instructions%20in%20a%20Multi-modal%20Jun/tables/3cf25a8b66cd174226ee80d2f965448544b1ab5ae9ea10dc3abb8d07f3aeadaa.jpg)

![401691d4e867e819ad53fc83958da370ef9f73563e3187fff748743f86f8a3b9.jpg](../iclr_results/2355_From%20an%20LLM%20Swarm%20to%20a%20PDDL-empowered%20Hive_%20Planning%20Self-executed%20Instructions%20in%20a%20Multi-modal%20Jun/tables/401691d4e867e819ad53fc83958da370ef9f73563e3187fff748743f86f8a3b9.jpg)

![53bd3a59a242af020736e2ceee0d8764c17beb93bbb1431906a7c76664ec0a49.jpg](../iclr_results/2355_From%20an%20LLM%20Swarm%20to%20a%20PDDL-empowered%20Hive_%20Planning%20Self-executed%20Instructions%20in%20a%20Multi-modal%20Jun/tables/53bd3a59a242af020736e2ceee0d8764c17beb93bbb1431906a7c76664ec0a49.jpg)

![92343d55b5d681ac43e03369b925b0f7c912c08b2f9b462360f214cc79f1fdd1.jpg](../iclr_results/2355_From%20an%20LLM%20Swarm%20to%20a%20PDDL-empowered%20Hive_%20Planning%20Self-executed%20Instructions%20in%20a%20Multi-modal%20Jun/tables/92343d55b5d681ac43e03369b925b0f7c912c08b2f9b462360f214cc79f1fdd1.jpg)

![a43ec036813cf9f4104b81dbf2ed64f20b57a72fb6ff5d44a1851317af976be0.jpg](../iclr_results/2355_From%20an%20LLM%20Swarm%20to%20a%20PDDL-empowered%20Hive_%20Planning%20Self-executed%20Instructions%20in%20a%20Multi-modal%20Jun/tables/a43ec036813cf9f4104b81dbf2ed64f20b57a72fb6ff5d44a1851317af976be0.jpg)

![ae05a73aa0dfbc8c2ea9c3e6d7eca4ca6dfc663d1120bead84bc16e74f04e97a.jpg](../iclr_results/2355_From%20an%20LLM%20Swarm%20to%20a%20PDDL-empowered%20Hive_%20Planning%20Self-executed%20Instructions%20in%20a%20Multi-modal%20Jun/tables/ae05a73aa0dfbc8c2ea9c3e6d7eca4ca6dfc663d1120bead84bc16e74f04e97a.jpg)

![b18ad947123754b1b5573ce4f64f5030e39838d0cd234d98b96d12eb01291b92.jpg](../iclr_results/2355_From%20an%20LLM%20Swarm%20to%20a%20PDDL-empowered%20Hive_%20Planning%20Self-executed%20Instructions%20in%20a%20Multi-modal%20Jun/tables/b18ad947123754b1b5573ce4f64f5030e39838d0cd234d98b96d12eb01291b92.jpg)

![c56efe1208dd212547b94805e499bdae55a166f3cc10a742fea040de63962844.jpg](../iclr_results/2355_From%20an%20LLM%20Swarm%20to%20a%20PDDL-empowered%20Hive_%20Planning%20Self-executed%20Instructions%20in%20a%20Multi-modal%20Jun/tables/c56efe1208dd212547b94805e499bdae55a166f3cc10a742fea040de63962844.jpg)

## Navigating Neural Space: Revisiting Concept Activation Vectors to Overcome Directional Divergence


### Images

![09d78b25ddf5eb7a954a407709fba840d2f7ea95457e14f0d85dbeeefcace1c9.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/09d78b25ddf5eb7a954a407709fba840d2f7ea95457e14f0d85dbeeefcace1c9.jpg)

![0c0a8e5a4a5dcfaef73881dc18ff47916d5dfaf13d39b179e820e198ec7b86ef.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/0c0a8e5a4a5dcfaef73881dc18ff47916d5dfaf13d39b179e820e198ec7b86ef.jpg)

![133ce889558cc6568a0406d746e1f33de580be9159ce99d5e50c234a1a4d2f3b.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/133ce889558cc6568a0406d746e1f33de580be9159ce99d5e50c234a1a4d2f3b.jpg)

![14eb9f141cb52fa4ebe3f9648010eccf98b11f75d18a2abaacaca687ba1a8aff.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/14eb9f141cb52fa4ebe3f9648010eccf98b11f75d18a2abaacaca687ba1a8aff.jpg)

![18597d0de45a01cdca7e55783f032a45ac156b51a9af3f6ef59b77c38d2bd1a8.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/18597d0de45a01cdca7e55783f032a45ac156b51a9af3f6ef59b77c38d2bd1a8.jpg)

![1b492767a3dcc785cbb074935859488007c1d4c7a6f1c579893d89507b723e39.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/1b492767a3dcc785cbb074935859488007c1d4c7a6f1c579893d89507b723e39.jpg)

![3be408376b30076e547038ada62494b1320ea07c242ac77bbe8ba5c0bfd51c70.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/3be408376b30076e547038ada62494b1320ea07c242ac77bbe8ba5c0bfd51c70.jpg)

![41ccdc49a8f15b2ca4a36d82b006da2e981848cbd0ae0f91f09207c54efb3015.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/41ccdc49a8f15b2ca4a36d82b006da2e981848cbd0ae0f91f09207c54efb3015.jpg)

![43567fa7dc23a234abac7f64b048acd01704b31df820efe81346d87d0ae60d7e.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/43567fa7dc23a234abac7f64b048acd01704b31df820efe81346d87d0ae60d7e.jpg)

![4c7c969a79f75b824d5944aec34f36d5b2a79aad2cb34dccd2008b2099d01b29.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/4c7c969a79f75b824d5944aec34f36d5b2a79aad2cb34dccd2008b2099d01b29.jpg)

![52ac938aa771cb367dd895aade85486621f77e70fdf26d425d5992423059879d.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/52ac938aa771cb367dd895aade85486621f77e70fdf26d425d5992423059879d.jpg)

![59547b4c77f4867b0874807837aae30959fedf79090947cfa66170f01a4002ef.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/59547b4c77f4867b0874807837aae30959fedf79090947cfa66170f01a4002ef.jpg)

![5bbc26e46872925c74c046b13a685dd7d6244698a075b80d78084bda0bb98f74.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/5bbc26e46872925c74c046b13a685dd7d6244698a075b80d78084bda0bb98f74.jpg)

![60b6e1d026d3377864e19aca368b897a531f7daef11e70a3812a8be32bd6536d.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/60b6e1d026d3377864e19aca368b897a531f7daef11e70a3812a8be32bd6536d.jpg)

![689ed0c80389f73ee9d67929ebd29425ebc8feb6df18c634d0d169cb2df3a5d8.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/689ed0c80389f73ee9d67929ebd29425ebc8feb6df18c634d0d169cb2df3a5d8.jpg)

![721d9a7b388b034a489998073ea5509e527bbdb0a0003e353dbc52fb3e6cd767.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/721d9a7b388b034a489998073ea5509e527bbdb0a0003e353dbc52fb3e6cd767.jpg)

![78251b9de01c03e9f0af8ea314611534b942ba5cccc92d71b5575a86b6d7aa93.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/78251b9de01c03e9f0af8ea314611534b942ba5cccc92d71b5575a86b6d7aa93.jpg)

![785dc3779621b53035c33a676af455e7b4dfb2623863e33f1d96ee8b66b4cbb6.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/785dc3779621b53035c33a676af455e7b4dfb2623863e33f1d96ee8b66b4cbb6.jpg)

![7aa5676e82d340bb8a5ad0ebff6fce36a41855e052e11d7e7a5a91c6e31b3640.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/7aa5676e82d340bb8a5ad0ebff6fce36a41855e052e11d7e7a5a91c6e31b3640.jpg)

![7d4c4902b91352a68684d99eb96ff2a9ba4c95a5438fe6a9c225f66d211aa06a.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/7d4c4902b91352a68684d99eb96ff2a9ba4c95a5438fe6a9c225f66d211aa06a.jpg)

![878c6b95727b9bf1854de96fb269d516cefcf19d22456ba4487f819eef7a3b2b.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/878c6b95727b9bf1854de96fb269d516cefcf19d22456ba4487f819eef7a3b2b.jpg)

![87e9891e402fb76db19dfe9fe30d897c4c948d6e7b97cbbc5444555210c66068.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/87e9891e402fb76db19dfe9fe30d897c4c948d6e7b97cbbc5444555210c66068.jpg)

![90e46a7829f048f4cc0266d6ae77b0659f543f6126d5734d9bcb901f079f39dc.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/90e46a7829f048f4cc0266d6ae77b0659f543f6126d5734d9bcb901f079f39dc.jpg)

![9785f06bb880277a6000114bb1f2769c9a2ac40f081dc77aadd598504501fb1f.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/9785f06bb880277a6000114bb1f2769c9a2ac40f081dc77aadd598504501fb1f.jpg)

![9b557863ee17524dcfe0f80a215953676e6ee84a9f619bad30ee0c56f50c7273.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/9b557863ee17524dcfe0f80a215953676e6ee84a9f619bad30ee0c56f50c7273.jpg)

![9edf469f283ebb961b22dfb28f16d4d2f3d796c78f905487ed9812ff5571b7eb.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/9edf469f283ebb961b22dfb28f16d4d2f3d796c78f905487ed9812ff5571b7eb.jpg)

![af9390f48fd7f9067b51e74083a06521c9a7c45e771bd1f82a9c0c47d2e6134c.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/af9390f48fd7f9067b51e74083a06521c9a7c45e771bd1f82a9c0c47d2e6134c.jpg)

![b1643709a8bfdf4966748545dde6124edc8315d4585efb0ea95243278c034f7c.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/b1643709a8bfdf4966748545dde6124edc8315d4585efb0ea95243278c034f7c.jpg)

![b486359a939793d66a2958e887a7bdcbb28467bcfa6aab226489090fddf146b6.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/b486359a939793d66a2958e887a7bdcbb28467bcfa6aab226489090fddf146b6.jpg)

![bf6ca7e2165226433022e69c29bca3ec2ac3f1e491cb336718ecb050d5e4d247.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/bf6ca7e2165226433022e69c29bca3ec2ac3f1e491cb336718ecb050d5e4d247.jpg)

![c21fd79fefae79990368b7a8e0d4a48f374dea19a7b4f190e6cc70b63cf31c77.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/c21fd79fefae79990368b7a8e0d4a48f374dea19a7b4f190e6cc70b63cf31c77.jpg)

![c887de3d885769d45f0793f851f958e355fab7aebe79510a31362d96f514e9f9.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/c887de3d885769d45f0793f851f958e355fab7aebe79510a31362d96f514e9f9.jpg)

![d079db33dc80f890b7364c63afcc3f24153ee1913b59669e579c2aa8da1bf391.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/d079db33dc80f890b7364c63afcc3f24153ee1913b59669e579c2aa8da1bf391.jpg)

![d5bee6033810550d7b5af14ad5700a750d19150f6cd6e5da408cdb6236cf5620.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/d5bee6033810550d7b5af14ad5700a750d19150f6cd6e5da408cdb6236cf5620.jpg)

![d5e975dbe3e988b17e041c74d2f63f0aaf82c87de65085084926564f2763d50a.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/d5e975dbe3e988b17e041c74d2f63f0aaf82c87de65085084926564f2763d50a.jpg)

![e59d2e90f79a20e9d03d850bac4cdd4ab107afb627322fd8916259798ba18a48.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/e59d2e90f79a20e9d03d850bac4cdd4ab107afb627322fd8916259798ba18a48.jpg)

![e608cde7d059ac6a508cddba8e8a2970dfda702f11c8fa34de2baf0dbb42f306.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/images/e608cde7d059ac6a508cddba8e8a2970dfda702f11c8fa34de2baf0dbb42f306.jpg)

### Tables

![0c0e04a93ef994709646392628ab4a1e415de3f65ed2eed3e54259a841614be7.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/tables/0c0e04a93ef994709646392628ab4a1e415de3f65ed2eed3e54259a841614be7.jpg)

![442b9ef1f5035284d0c4e0d2c20032713dfba1020bb98849c2936a8e16d62b01.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/tables/442b9ef1f5035284d0c4e0d2c20032713dfba1020bb98849c2936a8e16d62b01.jpg)

![612dae00475ba8b6144650ff71d13926c9f37d3932e5d519149d3dacda55b99a.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/tables/612dae00475ba8b6144650ff71d13926c9f37d3932e5d519149d3dacda55b99a.jpg)

![75f5c38449de4ccfbfe009b4df04bb046b597f9351417755b7f13ffb19f4585c.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/tables/75f5c38449de4ccfbfe009b4df04bb046b597f9351417755b7f13ffb19f4585c.jpg)

![77f5c776a3fd965227b7a8d0eb4793382b3f67a83613a59c641e16a1b9cea154.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/tables/77f5c776a3fd965227b7a8d0eb4793382b3f67a83613a59c641e16a1b9cea154.jpg)

![ad65db2076ee01df48b7617be86ce1859c3ef4616e8b6565f16301333187910e.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/tables/ad65db2076ee01df48b7617be86ce1859c3ef4616e8b6565f16301333187910e.jpg)

![b9f1c11bf93a143495b24d273db68fb548a2a608b1f0d7510a43378df4abfa95.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/tables/b9f1c11bf93a143495b24d273db68fb548a2a608b1f0d7510a43378df4abfa95.jpg)

![bc5d01125b41e0f625052f1187cba04d2895c55d8e4cb3bfd820ccadcf8d3804.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/tables/bc5d01125b41e0f625052f1187cba04d2895c55d8e4cb3bfd820ccadcf8d3804.jpg)

![e2044f4c33d713a03a5ae6137fa95c0b960e3e54d768e0f3e75825958f713b2f.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/tables/e2044f4c33d713a03a5ae6137fa95c0b960e3e54d768e0f3e75825958f713b2f.jpg)

![e2e8f9399fdeb44baa2f2d75e1192a7448760edb1a1210e4ad9a64d6c8b71d43.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/tables/e2e8f9399fdeb44baa2f2d75e1192a7448760edb1a1210e4ad9a64d6c8b71d43.jpg)

![ec9902ff4c5df1cdcafef844dddeeff9e7ea0942d83243384d58d0dfaed4784c.jpg](../iclr_results/2356_Navigating%20Neural%20Space_%20Revisiting%20Concept%20Activation%20Vectors%20to%20Overcome%20Directional%20Divergence/tables/ec9902ff4c5df1cdcafef844dddeeff9e7ea0942d83243384d58d0dfaed4784c.jpg)

## Generalized Behavior Learning from Diverse Demonstrations


### Images

![27b6f6079051ed994aac4d831c32a414d4b99a20baa263d625915d62674680e9.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/images/27b6f6079051ed994aac4d831c32a414d4b99a20baa263d625915d62674680e9.jpg)

![3b32d2bcce190c4f4c8e842858d45a6cf7722c094f39e2de4907bac403a56b89.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/images/3b32d2bcce190c4f4c8e842858d45a6cf7722c094f39e2de4907bac403a56b89.jpg)

![4332cff519c3ce084a93c8477f46c475a2d88fb8e728f57d6906a70e656b3007.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/images/4332cff519c3ce084a93c8477f46c475a2d88fb8e728f57d6906a70e656b3007.jpg)

![494445297e6162032ed1bc4f040b9f5b2d67d95dbf205a876208c0d55f297412.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/images/494445297e6162032ed1bc4f040b9f5b2d67d95dbf205a876208c0d55f297412.jpg)

![6996c153c6b7fa8ca70724efeeb1cd4897beb1e7f33a6affd1ebdd11d9f2190d.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/images/6996c153c6b7fa8ca70724efeeb1cd4897beb1e7f33a6affd1ebdd11d9f2190d.jpg)

![7c7ab0d871a2459ebfdaf5117da8d04c4ece4043d37bf6c438201137c5b4be4e.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/images/7c7ab0d871a2459ebfdaf5117da8d04c4ece4043d37bf6c438201137c5b4be4e.jpg)

![96a9d8dc0c05449e28b48192b71e1b26775476a5cf0776fc4a76a25d18f0f663.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/images/96a9d8dc0c05449e28b48192b71e1b26775476a5cf0776fc4a76a25d18f0f663.jpg)

![9d70d69fcb4000ce3ee71646147b5fca9610210bac1d23bcd1a2a978d4118a10.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/images/9d70d69fcb4000ce3ee71646147b5fca9610210bac1d23bcd1a2a978d4118a10.jpg)

![bb8f614403b957def02a517c9a690174d7a49256d9279a63cb7fb8ceb931bf3e.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/images/bb8f614403b957def02a517c9a690174d7a49256d9279a63cb7fb8ceb931bf3e.jpg)

![c1bd629db54d1f4f76751fcd82f6b3873b65658a2b127b6351985cd3f71ea6f2.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/images/c1bd629db54d1f4f76751fcd82f6b3873b65658a2b127b6351985cd3f71ea6f2.jpg)

![c57f4ddc35e104670e926e7a73f2f5334490da5bae247c17875d29866e51fb55.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/images/c57f4ddc35e104670e926e7a73f2f5334490da5bae247c17875d29866e51fb55.jpg)

![df7a733fd4dccf4ec946f9f186f5e1d593c2cb017cb4db6042854c5f836213bc.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/images/df7a733fd4dccf4ec946f9f186f5e1d593c2cb017cb4db6042854c5f836213bc.jpg)

![e929333d512194d88cebcd58dd19d92a16a0b7c51554f032d7c54bdd4e74bb71.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/images/e929333d512194d88cebcd58dd19d92a16a0b7c51554f032d7c54bdd4e74bb71.jpg)

### Tables

![1d2a2def4ca91aac6c63202b19d62cbaa0e9e76b0e0c551647f3f03b782ad034.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/tables/1d2a2def4ca91aac6c63202b19d62cbaa0e9e76b0e0c551647f3f03b782ad034.jpg)

![303545ac73cddfb658ee7e8e7b1d59b24bb231ca9214ab11e43556b1422fa46f.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/tables/303545ac73cddfb658ee7e8e7b1d59b24bb231ca9214ab11e43556b1422fa46f.jpg)

![390c173af259592d5952066215693c5fe87b6b5ae4ece2914506cac8958a2a74.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/tables/390c173af259592d5952066215693c5fe87b6b5ae4ece2914506cac8958a2a74.jpg)

![4df22e03a56ed3f24d0e2d99b17c37fb402a0f0e36edb3e1eb56c3d03d9703c3.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/tables/4df22e03a56ed3f24d0e2d99b17c37fb402a0f0e36edb3e1eb56c3d03d9703c3.jpg)

![6e56ada5ff4d81a2a8441b2028370337686914e32b580a1831eabcd95b98fdcb.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/tables/6e56ada5ff4d81a2a8441b2028370337686914e32b580a1831eabcd95b98fdcb.jpg)

![808dd9990c97e14b4f185b8d94169f98aa2d6479969cf46f76a9f809a609870a.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/tables/808dd9990c97e14b4f185b8d94169f98aa2d6479969cf46f76a9f809a609870a.jpg)

![89e3d75c5b28b15c0d6a017cba08ed83a1fc9fe705fd09ad1013f3ace1d5e1b8.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/tables/89e3d75c5b28b15c0d6a017cba08ed83a1fc9fe705fd09ad1013f3ace1d5e1b8.jpg)

![a7806e1441680a2a68e9c1026c3e2b283fc893c0bbef752a12d8ae59e1681c4f.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/tables/a7806e1441680a2a68e9c1026c3e2b283fc893c0bbef752a12d8ae59e1681c4f.jpg)

![ac37fc5174fb6fa48641e7dfa77a92cc99aee7ecd676e3edd878983fbabe0be1.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/tables/ac37fc5174fb6fa48641e7dfa77a92cc99aee7ecd676e3edd878983fbabe0be1.jpg)

![b43b688d00082a0138ad537ea1967c82056b4437f401477e35846851c2803cd2.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/tables/b43b688d00082a0138ad537ea1967c82056b4437f401477e35846851c2803cd2.jpg)

![c31b3ff530b087b3c5f71470c25fb4151e4d08829ebb44bedcc9fb6e86ffa53a.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/tables/c31b3ff530b087b3c5f71470c25fb4151e4d08829ebb44bedcc9fb6e86ffa53a.jpg)

![ca279b8323ea9a05b8243d0d21e839be83ddeb0fb2123d693848e4080c42a88a.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/tables/ca279b8323ea9a05b8243d0d21e839be83ddeb0fb2123d693848e4080c42a88a.jpg)

![eb43eaab7ad751601c8ab498ac8781cb65c975601ea68431340e1e35929f745f.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/tables/eb43eaab7ad751601c8ab498ac8781cb65c975601ea68431340e1e35929f745f.jpg)

![ed21a6729f6ccc36096c654ae985bda7fa31776db43719486cebeeffcd0cea57.jpg](../iclr_results/2357_Generalized%20Behavior%20Learning%20from%20Diverse%20Demonstrations/tables/ed21a6729f6ccc36096c654ae985bda7fa31776db43719486cebeeffcd0cea57.jpg)

## SoftMatcha: A Soft and Fast Pattern Matcher for Billion-Scale Corpus Searches


### Images

![0b72c845170ef16056321f6999229e92363953b2a71074171fa8e4710ec992ee.jpg](../iclr_results/2358_SoftMatcha_%20A%20Soft%20and%20Fast%20Pattern%20Matcher%20for%20Billion-Scale%20Corpus%20Searches/images/0b72c845170ef16056321f6999229e92363953b2a71074171fa8e4710ec992ee.jpg)

![654fffb5446475f334d4780353b50494ce0d159d7fd714b0142e41119c6267e0.jpg](../iclr_results/2358_SoftMatcha_%20A%20Soft%20and%20Fast%20Pattern%20Matcher%20for%20Billion-Scale%20Corpus%20Searches/images/654fffb5446475f334d4780353b50494ce0d159d7fd714b0142e41119c6267e0.jpg)

![8327ba7aadf52697faedb60efc22d5f2064a5220af0b5e745536389c3ef76fe0.jpg](../iclr_results/2358_SoftMatcha_%20A%20Soft%20and%20Fast%20Pattern%20Matcher%20for%20Billion-Scale%20Corpus%20Searches/images/8327ba7aadf52697faedb60efc22d5f2064a5220af0b5e745536389c3ef76fe0.jpg)

![8aab35f770947e97a531ec61549eedc451314af3ff45bf9951a280083044b2ab.jpg](../iclr_results/2358_SoftMatcha_%20A%20Soft%20and%20Fast%20Pattern%20Matcher%20for%20Billion-Scale%20Corpus%20Searches/images/8aab35f770947e97a531ec61549eedc451314af3ff45bf9951a280083044b2ab.jpg)

![a1a036ea3c72a22335f45c28c5ea0535114b7ce4a4799d77984e5f539746c400.jpg](../iclr_results/2358_SoftMatcha_%20A%20Soft%20and%20Fast%20Pattern%20Matcher%20for%20Billion-Scale%20Corpus%20Searches/images/a1a036ea3c72a22335f45c28c5ea0535114b7ce4a4799d77984e5f539746c400.jpg)

![f533cfb6bb76323c52ac9b01c73c546a50407031d0a04735ad2189b16bf62e2e.jpg](../iclr_results/2358_SoftMatcha_%20A%20Soft%20and%20Fast%20Pattern%20Matcher%20for%20Billion-Scale%20Corpus%20Searches/images/f533cfb6bb76323c52ac9b01c73c546a50407031d0a04735ad2189b16bf62e2e.jpg)

### Tables

![118228471972d459b9fb821e1dd4fde7fc8e76274da8f029e8918b099d3fb227.jpg](../iclr_results/2358_SoftMatcha_%20A%20Soft%20and%20Fast%20Pattern%20Matcher%20for%20Billion-Scale%20Corpus%20Searches/tables/118228471972d459b9fb821e1dd4fde7fc8e76274da8f029e8918b099d3fb227.jpg)

![231559547f78d2b772b2c02b019608d20e59ba9f1a23b011bc4950110b078d07.jpg](../iclr_results/2358_SoftMatcha_%20A%20Soft%20and%20Fast%20Pattern%20Matcher%20for%20Billion-Scale%20Corpus%20Searches/tables/231559547f78d2b772b2c02b019608d20e59ba9f1a23b011bc4950110b078d07.jpg)

![3127fc24e8cca227e90bcab8fd8229300b8470b6de9cfd6e40bf2cc78407ca08.jpg](../iclr_results/2358_SoftMatcha_%20A%20Soft%20and%20Fast%20Pattern%20Matcher%20for%20Billion-Scale%20Corpus%20Searches/tables/3127fc24e8cca227e90bcab8fd8229300b8470b6de9cfd6e40bf2cc78407ca08.jpg)

![38c99ca6b03752c9b61627feee335db51023f17873fed9469059e2bab07b3b91.jpg](../iclr_results/2358_SoftMatcha_%20A%20Soft%20and%20Fast%20Pattern%20Matcher%20for%20Billion-Scale%20Corpus%20Searches/tables/38c99ca6b03752c9b61627feee335db51023f17873fed9469059e2bab07b3b91.jpg)

![3a9b53c688408b5f80707ec60cf94a1c5f69c54751c8377d5aa9df31720ac148.jpg](../iclr_results/2358_SoftMatcha_%20A%20Soft%20and%20Fast%20Pattern%20Matcher%20for%20Billion-Scale%20Corpus%20Searches/tables/3a9b53c688408b5f80707ec60cf94a1c5f69c54751c8377d5aa9df31720ac148.jpg)

![5965b76bb014d3d28319bd24c3cc7224c8594d2721d9405aa7744f3ee53b9293.jpg](../iclr_results/2358_SoftMatcha_%20A%20Soft%20and%20Fast%20Pattern%20Matcher%20for%20Billion-Scale%20Corpus%20Searches/tables/5965b76bb014d3d28319bd24c3cc7224c8594d2721d9405aa7744f3ee53b9293.jpg)

![5d456d0dd47cca99002c06b42c7f3f2a9d7a49e0da9a479cc4e27fb9023af146.jpg](../iclr_results/2358_SoftMatcha_%20A%20Soft%20and%20Fast%20Pattern%20Matcher%20for%20Billion-Scale%20Corpus%20Searches/tables/5d456d0dd47cca99002c06b42c7f3f2a9d7a49e0da9a479cc4e27fb9023af146.jpg)

![77b2e338a1a352a1c894f6f0a790ce37bca1876bc36948a4d2ff5fdd336e8bbf.jpg](../iclr_results/2358_SoftMatcha_%20A%20Soft%20and%20Fast%20Pattern%20Matcher%20for%20Billion-Scale%20Corpus%20Searches/tables/77b2e338a1a352a1c894f6f0a790ce37bca1876bc36948a4d2ff5fdd336e8bbf.jpg)

![d6a0095446c81e68adfb7018600a69fd5d33779d291fd4c5728de5b02d8a8bbd.jpg](../iclr_results/2358_SoftMatcha_%20A%20Soft%20and%20Fast%20Pattern%20Matcher%20for%20Billion-Scale%20Corpus%20Searches/tables/d6a0095446c81e68adfb7018600a69fd5d33779d291fd4c5728de5b02d8a8bbd.jpg)

![d9cc53bdaf1a2d04a4063b2ee37270bb8a77b5cdd5fdbf6d2d869a7b35742500.jpg](../iclr_results/2358_SoftMatcha_%20A%20Soft%20and%20Fast%20Pattern%20Matcher%20for%20Billion-Scale%20Corpus%20Searches/tables/d9cc53bdaf1a2d04a4063b2ee37270bb8a77b5cdd5fdbf6d2d869a7b35742500.jpg)

![e28994bc6980a11295a694f298a83ed57ecf401ee87371844d4dabec4082f807.jpg](../iclr_results/2358_SoftMatcha_%20A%20Soft%20and%20Fast%20Pattern%20Matcher%20for%20Billion-Scale%20Corpus%20Searches/tables/e28994bc6980a11295a694f298a83ed57ecf401ee87371844d4dabec4082f807.jpg)

![e39bd64d927f57aa73cc26b4bb634ab2c982a1f9808421a164b9e16d041e011d.jpg](../iclr_results/2358_SoftMatcha_%20A%20Soft%20and%20Fast%20Pattern%20Matcher%20for%20Billion-Scale%20Corpus%20Searches/tables/e39bd64d927f57aa73cc26b4bb634ab2c982a1f9808421a164b9e16d041e011d.jpg)

![f4134464a70df5fb14447cafac638b23cf41e299c65028266b1ba4ac5c94618e.jpg](../iclr_results/2358_SoftMatcha_%20A%20Soft%20and%20Fast%20Pattern%20Matcher%20for%20Billion-Scale%20Corpus%20Searches/tables/f4134464a70df5fb14447cafac638b23cf41e299c65028266b1ba4ac5c94618e.jpg)

## Efficient Source-Free Time-Series Adaptation via Parameter Subspace Disentanglement


### Images

![1afd6fb6256d1eabd1b46247c12ff640e206e07ac7520a35a71528509185d909.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/images/1afd6fb6256d1eabd1b46247c12ff640e206e07ac7520a35a71528509185d909.jpg)

![3281293c84baa576f529b15aa7029bc4ee254d35526252842541c45ec0aac076.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/images/3281293c84baa576f529b15aa7029bc4ee254d35526252842541c45ec0aac076.jpg)

![424614eb44b56fbc193bc44c4cca172480707c4d8a78c8db5a9c88e2f6212aeb.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/images/424614eb44b56fbc193bc44c4cca172480707c4d8a78c8db5a9c88e2f6212aeb.jpg)

![48d004bc185cdc08d70c42823da45887b55cdd64fc27b95898b7fd82028eea2a.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/images/48d004bc185cdc08d70c42823da45887b55cdd64fc27b95898b7fd82028eea2a.jpg)

![4c5b1ddeeabf426aeae7a7e403dc3382645e1effba3d2a39c7304a93da0cdf0f.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/images/4c5b1ddeeabf426aeae7a7e403dc3382645e1effba3d2a39c7304a93da0cdf0f.jpg)

![540da5c81334fcddf0bca2d8a6fe9171cf1ee71bdcfacc8a490c6d53530fa863.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/images/540da5c81334fcddf0bca2d8a6fe9171cf1ee71bdcfacc8a490c6d53530fa863.jpg)

![586f429c46f3e6844edaeed6683dd61071028620387b33935ed90e983e40c7c9.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/images/586f429c46f3e6844edaeed6683dd61071028620387b33935ed90e983e40c7c9.jpg)

![5899548139e088759ee5f279474cbf69642dfc1cbcff514fc70ad55183edfff4.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/images/5899548139e088759ee5f279474cbf69642dfc1cbcff514fc70ad55183edfff4.jpg)

![8157b80e63d79b933ec521533571b853a7feb2e7a45ffaf2c33a2c5a9890aa81.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/images/8157b80e63d79b933ec521533571b853a7feb2e7a45ffaf2c33a2c5a9890aa81.jpg)

![892458faf09ba0ad0484246cd82c0bd576f580afe5d58a3b95b2af73baac7aa8.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/images/892458faf09ba0ad0484246cd82c0bd576f580afe5d58a3b95b2af73baac7aa8.jpg)

![8ad034259ebbecd11fc7a233d9218fba231c86daf8d3d31dd442b95d01de60ad.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/images/8ad034259ebbecd11fc7a233d9218fba231c86daf8d3d31dd442b95d01de60ad.jpg)

![93f0c66c4c4b2fd3855038c5a90bbac9cfcca05113d1048aa4ec10efc4f37366.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/images/93f0c66c4c4b2fd3855038c5a90bbac9cfcca05113d1048aa4ec10efc4f37366.jpg)

![94d44e64a8bfc22ae3955e818760ded51ab789fac2966b3aabf784b417bb6bee.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/images/94d44e64a8bfc22ae3955e818760ded51ab789fac2966b3aabf784b417bb6bee.jpg)

![963fbb850a0f37b03a288414ef91ef407b6e3f32af95442da94bc368965dfedd.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/images/963fbb850a0f37b03a288414ef91ef407b6e3f32af95442da94bc368965dfedd.jpg)

![9941824aaedb1d8fc635d2b37e5ce8e3581848017066fae1296762a78ca5fc32.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/images/9941824aaedb1d8fc635d2b37e5ce8e3581848017066fae1296762a78ca5fc32.jpg)

![9caf5572e053f59e815699a5bd08f093b7e2fac8dbd30d5fb6e1a0fb2b14a44b.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/images/9caf5572e053f59e815699a5bd08f093b7e2fac8dbd30d5fb6e1a0fb2b14a44b.jpg)

![af7e75416b421552efd30ae46921afe52b86e9a32ecedd9ca27e1c12fc70713e.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/images/af7e75416b421552efd30ae46921afe52b86e9a32ecedd9ca27e1c12fc70713e.jpg)

![bf60aaa8c47c3efe45cb7b829abffe8b51141008c582aca64278697a2be22934.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/images/bf60aaa8c47c3efe45cb7b829abffe8b51141008c582aca64278697a2be22934.jpg)

![d9093ec471dfddd02d79ef8043240b14cad1b379ff60b7a960635deaf2d88a2f.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/images/d9093ec471dfddd02d79ef8043240b14cad1b379ff60b7a960635deaf2d88a2f.jpg)

![e648e4db67fb2b690ba82e2ba1095ce172cc075a57b3507683310f98bf9c95cf.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/images/e648e4db67fb2b690ba82e2ba1095ce172cc075a57b3507683310f98bf9c95cf.jpg)

### Tables

![16fc36eddad3d8a2162c19094e95c06d54960c481c2dac3d9da42a3251d1fd5b.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/tables/16fc36eddad3d8a2162c19094e95c06d54960c481c2dac3d9da42a3251d1fd5b.jpg)

![34efa0dcbf3916eeef84e3a28fce02e080be9e32896444e3c48e861efd98cc26.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/tables/34efa0dcbf3916eeef84e3a28fce02e080be9e32896444e3c48e861efd98cc26.jpg)

![35b2ecef4201a0410ae5b11140e6ce958c30a06f4d6fb0f694770843ee50873a.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/tables/35b2ecef4201a0410ae5b11140e6ce958c30a06f4d6fb0f694770843ee50873a.jpg)

![460af941e88500f2e889aa004789e391f1f49819e560fb9de0fa0b26b428fa99.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/tables/460af941e88500f2e889aa004789e391f1f49819e560fb9de0fa0b26b428fa99.jpg)

![5dcf967714273a73fd86ab56198a691fddc9eb9fc18641035ce99e6b1cce2594.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/tables/5dcf967714273a73fd86ab56198a691fddc9eb9fc18641035ce99e6b1cce2594.jpg)

![742f46dc38ea59b63806d19ef5bae5fae2950cb671737d76bb2003f0d1350d88.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/tables/742f46dc38ea59b63806d19ef5bae5fae2950cb671737d76bb2003f0d1350d88.jpg)

![74f05611ca4caad9fb971ec272b14dec72d56bb9239f456e0cc5934e23a1e045.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/tables/74f05611ca4caad9fb971ec272b14dec72d56bb9239f456e0cc5934e23a1e045.jpg)

![78ceb375c81af0dcf8361d8e2355da8903f79f04a50a49c4e31fc61ee34f54c0.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/tables/78ceb375c81af0dcf8361d8e2355da8903f79f04a50a49c4e31fc61ee34f54c0.jpg)

![7b4b4f3e6abc5abd6751e3219c6be4ea71f782f026289a2788f90500e1d4e1b7.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/tables/7b4b4f3e6abc5abd6751e3219c6be4ea71f782f026289a2788f90500e1d4e1b7.jpg)

![9b0a5b26c8dd3d9374e6408bbfe66f94583ad9b73f404ef5d3ce6d28eace4cc5.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/tables/9b0a5b26c8dd3d9374e6408bbfe66f94583ad9b73f404ef5d3ce6d28eace4cc5.jpg)

![b1bf861706d5f7534b384a1fc96be3e22973f26a276ba58e830e0ca2b26e7545.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/tables/b1bf861706d5f7534b384a1fc96be3e22973f26a276ba58e830e0ca2b26e7545.jpg)

![b939f6d3fbd23e10bec720a40ea7ddd6fbdd65f2f64a49ba35e17a0fceb83cae.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/tables/b939f6d3fbd23e10bec720a40ea7ddd6fbdd65f2f64a49ba35e17a0fceb83cae.jpg)

![dc2a8013bf4bdeac9715858fda2bdd96508b53db52a515a9059ad03aa497b447.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/tables/dc2a8013bf4bdeac9715858fda2bdd96508b53db52a515a9059ad03aa497b447.jpg)

![f20e56b309052f047d8624a2464916f283655eb31b57763fdea4df220b597b17.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/tables/f20e56b309052f047d8624a2464916f283655eb31b57763fdea4df220b597b17.jpg)

![f86941236c6f952aff4d30d1ea8e29e2dfceae5f540f61b33ff93b3d9d432fbb.jpg](../iclr_results/2359_Efficient%20Source-Free%20Time-Series%20Adaptation%20via%20Parameter%20Subspace%20Disentanglement/tables/f86941236c6f952aff4d30d1ea8e29e2dfceae5f540f61b33ff93b3d9d432fbb.jpg)

## Do Contemporary Causal Inference Models Capture Real-World Heterogeneity? Findings from a Large-Scale Benchmark


### Images

![022b608b745fe2a91fa3cc673c82b58524e9189dfea815c7c555af98fd6906e7.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/022b608b745fe2a91fa3cc673c82b58524e9189dfea815c7c555af98fd6906e7.jpg)

![035a16620c0ec5d88584520f80c68f5a5a4d33807a934078342de0a99d85718d.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/035a16620c0ec5d88584520f80c68f5a5a4d33807a934078342de0a99d85718d.jpg)

![0635c2337609287aab4e45e2b112e8d48ba84e563f2a4dbaf2462f64edbf611c.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/0635c2337609287aab4e45e2b112e8d48ba84e563f2a4dbaf2462f64edbf611c.jpg)

![086f67f0d05146a7f34daa87fb323b20a62c2d5b034a6fafdbd3778c2c7360f6.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/086f67f0d05146a7f34daa87fb323b20a62c2d5b034a6fafdbd3778c2c7360f6.jpg)

![11b74a6657f9fbf9852eb6cde559d57a0c054c3502155c05a24f01ea2130663e.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/11b74a6657f9fbf9852eb6cde559d57a0c054c3502155c05a24f01ea2130663e.jpg)

![15238ef754ff70305be70c24132348bd0dea8dec31aee65bf1ceb7fb821004d7.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/15238ef754ff70305be70c24132348bd0dea8dec31aee65bf1ceb7fb821004d7.jpg)

![19005049dc3bf2b5b846425c1d27f512470dfa612bbaf33f7ff6b24efa2b6181.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/19005049dc3bf2b5b846425c1d27f512470dfa612bbaf33f7ff6b24efa2b6181.jpg)

![1ca04202b32a74ddea78020ca7c68a0d6f2c0a181775725d0a73579cd66a3697.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/1ca04202b32a74ddea78020ca7c68a0d6f2c0a181775725d0a73579cd66a3697.jpg)

![1cba05c65a7305ba167024140063fb80b9b13fd1507e6f641f785feafb49fa9c.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/1cba05c65a7305ba167024140063fb80b9b13fd1507e6f641f785feafb49fa9c.jpg)

![1e383203ee8c4215175af15aa673ff87f2ebb54b225ad08072d102506d9b8dce.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/1e383203ee8c4215175af15aa673ff87f2ebb54b225ad08072d102506d9b8dce.jpg)

![1fb9ef2a1c9a218c27adfb6bf783723907dfd43d38d8c4e91fe1050df66f8de6.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/1fb9ef2a1c9a218c27adfb6bf783723907dfd43d38d8c4e91fe1050df66f8de6.jpg)

![26e9ca89dab568182f614b156b2c49ee3c7f52542865270ef890a0724a076c34.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/26e9ca89dab568182f614b156b2c49ee3c7f52542865270ef890a0724a076c34.jpg)

![2959f95997f95b891ff6b1383fdf3a5406af2274cedf4ab7fae6aa58746ab792.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/2959f95997f95b891ff6b1383fdf3a5406af2274cedf4ab7fae6aa58746ab792.jpg)

![38d5bb27a3e733b8b2210c45d5b2e7c2ab553dbc04d8770ebf44cfa0593e8ee9.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/38d5bb27a3e733b8b2210c45d5b2e7c2ab553dbc04d8770ebf44cfa0593e8ee9.jpg)

![4a8eae8b9db619f1e8811c875ca6d0349fd4c32470318350cc44b4001072cc1b.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/4a8eae8b9db619f1e8811c875ca6d0349fd4c32470318350cc44b4001072cc1b.jpg)

![52f148c114007642ee4b242175a0988cace439816cfc31f9fa4bb7d65ae3d57c.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/52f148c114007642ee4b242175a0988cace439816cfc31f9fa4bb7d65ae3d57c.jpg)

![5490e36954525c9ee5d193983fe9c435cf03845fffa8cbe7d1bc754f96907722.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/5490e36954525c9ee5d193983fe9c435cf03845fffa8cbe7d1bc754f96907722.jpg)

![5982bde70b317af1ab1184f5364b9ea200a2c2cdd3ee262c90c410ae3fb5b431.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/5982bde70b317af1ab1184f5364b9ea200a2c2cdd3ee262c90c410ae3fb5b431.jpg)

![5a36e8525a10c175782b203d336a04b8d40220988cb7d240b3cb58707e808aee.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/5a36e8525a10c175782b203d336a04b8d40220988cb7d240b3cb58707e808aee.jpg)

![5f1340e2b733bad08ad6cb859e480847b576c706c869b6abfe4c71def43d8da3.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/5f1340e2b733bad08ad6cb859e480847b576c706c869b6abfe4c71def43d8da3.jpg)

![61eb61b8462237942410a33ebc393b3efbe6481f677f67c64a127f4176e5985d.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/61eb61b8462237942410a33ebc393b3efbe6481f677f67c64a127f4176e5985d.jpg)

![66eb1b8af68c69e0220470c951a535349ef7e5214bb42666e7e5f9c95bef16ab.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/66eb1b8af68c69e0220470c951a535349ef7e5214bb42666e7e5f9c95bef16ab.jpg)

![66fbb9db9303e85569c416a10b9590a83f9c6d58bdc6c893906e5930883c51e9.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/66fbb9db9303e85569c416a10b9590a83f9c6d58bdc6c893906e5930883c51e9.jpg)

![679cb88f80d0ef01ab1871398f6b89863c22a4ddc7fbb78a490906e3780917ce.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/679cb88f80d0ef01ab1871398f6b89863c22a4ddc7fbb78a490906e3780917ce.jpg)

![7290e3805bdb676e6cbcad40aaad7b6773463d5632ad0ae5514084111edab869.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/7290e3805bdb676e6cbcad40aaad7b6773463d5632ad0ae5514084111edab869.jpg)

![7cc1dd3006ca5657f791fc125e6a9e8134649eb13b2788ae96b109776f816e7d.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/7cc1dd3006ca5657f791fc125e6a9e8134649eb13b2788ae96b109776f816e7d.jpg)

![827e87e19ee3e4c482c4760adda992adfc95d99fc15849762c86a09606e39727.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/827e87e19ee3e4c482c4760adda992adfc95d99fc15849762c86a09606e39727.jpg)

![83057ec96a476ea70f77e6fe28e6ec048ebdcbce80104777b25437a0f42e53c4.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/83057ec96a476ea70f77e6fe28e6ec048ebdcbce80104777b25437a0f42e53c4.jpg)

![8d6b9da4bad0fb54239715fe396d5a2273f97bcb5edf4800449ff37458b22a13.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/8d6b9da4bad0fb54239715fe396d5a2273f97bcb5edf4800449ff37458b22a13.jpg)

![8f7b96fe1658ec2bcfada04a22c50ae63eebceccb387ee6e992a7d18a8ae412c.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/8f7b96fe1658ec2bcfada04a22c50ae63eebceccb387ee6e992a7d18a8ae412c.jpg)

![931c1b26f352530ca08df0c803416c4d9b480e800975fc9c56d86965feb818b7.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/931c1b26f352530ca08df0c803416c4d9b480e800975fc9c56d86965feb818b7.jpg)

![a169f6af1f6252e555da10b5faf8b43ad3e7c6adc7e5318598a999cbcbcfca94.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/a169f6af1f6252e555da10b5faf8b43ad3e7c6adc7e5318598a999cbcbcfca94.jpg)

![a9be41ef71b681a786a90166fdd03caaf29aa2f8657ca708a172ed343e183944.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/a9be41ef71b681a786a90166fdd03caaf29aa2f8657ca708a172ed343e183944.jpg)

![ad9136210d92332f1bb28a95abf6f18b76397acfb39543e3e4a5395f765fa84d.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/ad9136210d92332f1bb28a95abf6f18b76397acfb39543e3e4a5395f765fa84d.jpg)

![b215a7b6d2fac904ed835cec65bb27d160f73d7226450060855c7622c26643f5.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/b215a7b6d2fac904ed835cec65bb27d160f73d7226450060855c7622c26643f5.jpg)

![b6d32a52c3a10c6de94d7390b7f738bb9ca3a9f062bcb0de26fe131c536e5b34.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/b6d32a52c3a10c6de94d7390b7f738bb9ca3a9f062bcb0de26fe131c536e5b34.jpg)

![b98f9cc3ddcccf2a4c456702263fa6de02eef20a25163ef21afebcf5a9d0ad4d.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/b98f9cc3ddcccf2a4c456702263fa6de02eef20a25163ef21afebcf5a9d0ad4d.jpg)

![be41a2352f4bfdec56de5ced0d4f524a6d3be4a66ee3e6344de9f86ea23af087.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/be41a2352f4bfdec56de5ced0d4f524a6d3be4a66ee3e6344de9f86ea23af087.jpg)

![be59ca6d3e5fdc0b132e850fd3ec3e6eaaf852d9166004656c4a3dc9ea6e233a.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/be59ca6d3e5fdc0b132e850fd3ec3e6eaaf852d9166004656c4a3dc9ea6e233a.jpg)

![c382c017d1603271cc81a126901000a0cc9061744012499c1b1994062f14dc85.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/c382c017d1603271cc81a126901000a0cc9061744012499c1b1994062f14dc85.jpg)

![c4015af0b52adc8bf24a77ab82ccd08ba6fc9dea55808c4a97a5fff71a369ebf.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/c4015af0b52adc8bf24a77ab82ccd08ba6fc9dea55808c4a97a5fff71a369ebf.jpg)

![c44e618019d6f044194a5f23d59304c13ec74ca6dd1484784d1245e1b3077be6.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/c44e618019d6f044194a5f23d59304c13ec74ca6dd1484784d1245e1b3077be6.jpg)

![c4fbfde1074779d7d3f42d7d2744341078af8fd63cf09e0323cb26c11f6f04f9.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/c4fbfde1074779d7d3f42d7d2744341078af8fd63cf09e0323cb26c11f6f04f9.jpg)

![cc346d5aabd7fce72cbe51ffe5ff762b2ea61eab478dff22275cf4994cf2de35.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/cc346d5aabd7fce72cbe51ffe5ff762b2ea61eab478dff22275cf4994cf2de35.jpg)

![d15bbdf41163bdf20a204fa584e75c5090f761dc998ab023d8e5356b67f3fc22.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/d15bbdf41163bdf20a204fa584e75c5090f761dc998ab023d8e5356b67f3fc22.jpg)

![d799a4288b0d1be2cb8c699305f5ae630ea74dd4bf029930221d72f34b8c9b93.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/d799a4288b0d1be2cb8c699305f5ae630ea74dd4bf029930221d72f34b8c9b93.jpg)

![d8014245ac8b274c73ed2b2aa5632f723028e888f56d86ce072c10dab4b57580.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/d8014245ac8b274c73ed2b2aa5632f723028e888f56d86ce072c10dab4b57580.jpg)

![d9c6632c9b7d9e55b9484e46fd790d29f876339a324ff17fe77cbb4da2483106.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/d9c6632c9b7d9e55b9484e46fd790d29f876339a324ff17fe77cbb4da2483106.jpg)

![dc2efdfd630fc67319c1024d928e7339b31ed0e568ba751b6287f83378843593.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/dc2efdfd630fc67319c1024d928e7339b31ed0e568ba751b6287f83378843593.jpg)

![dda6945e54c8f559041ad88b9953c76a523877ffa32c926df141b7a89abbb36f.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/dda6945e54c8f559041ad88b9953c76a523877ffa32c926df141b7a89abbb36f.jpg)

![e329b0f112814f932bf914fe75bab7f2bd0537d6f9257daa51d14dd46bcbf980.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/e329b0f112814f932bf914fe75bab7f2bd0537d6f9257daa51d14dd46bcbf980.jpg)

![e55a4bdeca7959b31c44c21023a6c9af876d605765df14ebe8bcae527c242294.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/e55a4bdeca7959b31c44c21023a6c9af876d605765df14ebe8bcae527c242294.jpg)

![edd0d3b00102b9e1b0fe499af773751e76425f0b547db4797ef641f2b53180aa.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/edd0d3b00102b9e1b0fe499af773751e76425f0b547db4797ef641f2b53180aa.jpg)

![f0b3165a9835a20b948aefe4250f7c8c9029e5b0ebed98175f0f7e87fc40b6d9.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/f0b3165a9835a20b948aefe4250f7c8c9029e5b0ebed98175f0f7e87fc40b6d9.jpg)

![f5b561faa81aede34ee74d470d0d0c60709535dc32d9d348ea873c904682f637.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/f5b561faa81aede34ee74d470d0d0c60709535dc32d9d348ea873c904682f637.jpg)

![f6b71d6b22327a764b63a5e880e6e053629fea3a117a27a746f3255313be560e.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/f6b71d6b22327a764b63a5e880e6e053629fea3a117a27a746f3255313be560e.jpg)

![f9c61ddd481b62e7005276ab82310efece3a99b63809d354034a6932fa6f852c.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/f9c61ddd481b62e7005276ab82310efece3a99b63809d354034a6932fa6f852c.jpg)

![fa8855b3aca108249c2f8d9d1f8f48b91e54b53377144c6ee73c163d38a36fa4.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/fa8855b3aca108249c2f8d9d1f8f48b91e54b53377144c6ee73c163d38a36fa4.jpg)

![ff4c653125d8ec843f6fae941692579b57fcf45001702fd15a09b40e24c3bf37.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/images/ff4c653125d8ec843f6fae941692579b57fcf45001702fd15a09b40e24c3bf37.jpg)

### Tables

![12d75b2b26108ad1834d1c4cf1b5bfc0f10b8faa58be50267bb23c0a10039cdd.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/tables/12d75b2b26108ad1834d1c4cf1b5bfc0f10b8faa58be50267bb23c0a10039cdd.jpg)

![21be70192c4b9f801241bdf7d20dc00cbafc85d50fb30a7a2d8817742c912805.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/tables/21be70192c4b9f801241bdf7d20dc00cbafc85d50fb30a7a2d8817742c912805.jpg)

![4082477c30b968d75a6b19bad7612193e481d6c2d4f073eabdfe620ff3d9e754.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/tables/4082477c30b968d75a6b19bad7612193e481d6c2d4f073eabdfe620ff3d9e754.jpg)

![71e4bac5f35c76185da38df3e7b8d98367b0a2ef1e3493d7e7f6e7e08ae483ff.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/tables/71e4bac5f35c76185da38df3e7b8d98367b0a2ef1e3493d7e7f6e7e08ae483ff.jpg)

![7307a3bac38603b4c88e542d3c4b9adb15da8a7541f5dda8011105ced1d09733.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/tables/7307a3bac38603b4c88e542d3c4b9adb15da8a7541f5dda8011105ced1d09733.jpg)

![866d774daacbb23e9017ee11ea318cd48cd2358e20f19780500d421a42e58bb2.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/tables/866d774daacbb23e9017ee11ea318cd48cd2358e20f19780500d421a42e58bb2.jpg)

![a8ce26146a6755abcf189acb590c929f34413d5084def39c0e1bb7b6ad583a69.jpg](../iclr_results/2360_Do%20Contemporary%20Causal%20Inference%20Models%20Capture%20Real-World%20Heterogeneity_%20Findings%20from%20a%20Large-Scal/tables/a8ce26146a6755abcf189acb590c929f34413d5084def39c0e1bb7b6ad583a69.jpg)

## Towards Self-Supervised Covariance Estimation in Deep Heteroscedastic Regression


### Images

![074c3470fd5a67aa2577222c630a57b0d5b10fede3f68e0f823dcb1aa75a3b04.jpg](../iclr_results/2361_Towards%20Self-Supervised%20Covariance%20Estimation%20in%20Deep%20Heteroscedastic%20Regression/images/074c3470fd5a67aa2577222c630a57b0d5b10fede3f68e0f823dcb1aa75a3b04.jpg)

![1641c3c7734924f3e7c0f57056bd0f50a3009e44f09de9e6d40ef3eb70596733.jpg](../iclr_results/2361_Towards%20Self-Supervised%20Covariance%20Estimation%20in%20Deep%20Heteroscedastic%20Regression/images/1641c3c7734924f3e7c0f57056bd0f50a3009e44f09de9e6d40ef3eb70596733.jpg)

![1cb67124d915980ded2e6b568f14e44d7812fc5430a69047b5ddb38fc6238be1.jpg](../iclr_results/2361_Towards%20Self-Supervised%20Covariance%20Estimation%20in%20Deep%20Heteroscedastic%20Regression/images/1cb67124d915980ded2e6b568f14e44d7812fc5430a69047b5ddb38fc6238be1.jpg)

![20c18d866a5bd71a55ffaa77fc34416346ba967a63cfaaf1eb6759c7d1294acf.jpg](../iclr_results/2361_Towards%20Self-Supervised%20Covariance%20Estimation%20in%20Deep%20Heteroscedastic%20Regression/images/20c18d866a5bd71a55ffaa77fc34416346ba967a63cfaaf1eb6759c7d1294acf.jpg)

![29fe37cc8f1d39e5a0af2b9706c0468a3967e11e57fb7f98bd37cccc7b395e68.jpg](../iclr_results/2361_Towards%20Self-Supervised%20Covariance%20Estimation%20in%20Deep%20Heteroscedastic%20Regression/images/29fe37cc8f1d39e5a0af2b9706c0468a3967e11e57fb7f98bd37cccc7b395e68.jpg)

![728fad8c6cd158a01b25bf3066f46958cfee6cf5d4c9ccc67ad62376cfec43d0.jpg](../iclr_results/2361_Towards%20Self-Supervised%20Covariance%20Estimation%20in%20Deep%20Heteroscedastic%20Regression/images/728fad8c6cd158a01b25bf3066f46958cfee6cf5d4c9ccc67ad62376cfec43d0.jpg)

![93e48fc4178ceeaea11833238cd9711ccf398a5b979dbd1d7068a6db99861dad.jpg](../iclr_results/2361_Towards%20Self-Supervised%20Covariance%20Estimation%20in%20Deep%20Heteroscedastic%20Regression/images/93e48fc4178ceeaea11833238cd9711ccf398a5b979dbd1d7068a6db99861dad.jpg)

![9e60e9c7fb4ac6ff6648281ea712e53e00770ce6a3ad71e47ea85005b1ea6424.jpg](../iclr_results/2361_Towards%20Self-Supervised%20Covariance%20Estimation%20in%20Deep%20Heteroscedastic%20Regression/images/9e60e9c7fb4ac6ff6648281ea712e53e00770ce6a3ad71e47ea85005b1ea6424.jpg)

![cb48171f871c6bab8f74e302787f9ca445f2af74d296e965000ec145f42884c9.jpg](../iclr_results/2361_Towards%20Self-Supervised%20Covariance%20Estimation%20in%20Deep%20Heteroscedastic%20Regression/images/cb48171f871c6bab8f74e302787f9ca445f2af74d296e965000ec145f42884c9.jpg)

![d107cf4186ce829ccba04861aa1d8c4396b838fa93a5d4effa1bf31035f77bbe.jpg](../iclr_results/2361_Towards%20Self-Supervised%20Covariance%20Estimation%20in%20Deep%20Heteroscedastic%20Regression/images/d107cf4186ce829ccba04861aa1d8c4396b838fa93a5d4effa1bf31035f77bbe.jpg)

![db8d3079754c1e2005351cca301e5b723c40f74f7ecf23f644a79c592ae5bc9f.jpg](../iclr_results/2361_Towards%20Self-Supervised%20Covariance%20Estimation%20in%20Deep%20Heteroscedastic%20Regression/images/db8d3079754c1e2005351cca301e5b723c40f74f7ecf23f644a79c592ae5bc9f.jpg)

![dcea7bf50aaae4b7646d2983ed30cdd4b70a5b4b2036244052fbdb75fd9d0ad4.jpg](../iclr_results/2361_Towards%20Self-Supervised%20Covariance%20Estimation%20in%20Deep%20Heteroscedastic%20Regression/images/dcea7bf50aaae4b7646d2983ed30cdd4b70a5b4b2036244052fbdb75fd9d0ad4.jpg)

![ea926dbfac4e1a55219e9d200283187c9a04d8cd4f6222184f30a9122ff8faba.jpg](../iclr_results/2361_Towards%20Self-Supervised%20Covariance%20Estimation%20in%20Deep%20Heteroscedastic%20Regression/images/ea926dbfac4e1a55219e9d200283187c9a04d8cd4f6222184f30a9122ff8faba.jpg)

![ebf2128b25b7da51870b4ff68474c15083b19a78b0a2c8db70ebaec5fd63579b.jpg](../iclr_results/2361_Towards%20Self-Supervised%20Covariance%20Estimation%20in%20Deep%20Heteroscedastic%20Regression/images/ebf2128b25b7da51870b4ff68474c15083b19a78b0a2c8db70ebaec5fd63579b.jpg)

![f54a3ed76036929892605d7216d8d8eb7dd848fc3276c4a0a2173fd1e1f8621d.jpg](../iclr_results/2361_Towards%20Self-Supervised%20Covariance%20Estimation%20in%20Deep%20Heteroscedastic%20Regression/images/f54a3ed76036929892605d7216d8d8eb7dd848fc3276c4a0a2173fd1e1f8621d.jpg)

![feb4ceee17a0d5c5b936353a9e9ddba0cdc77d7be758e71aa05befa7fda72f45.jpg](../iclr_results/2361_Towards%20Self-Supervised%20Covariance%20Estimation%20in%20Deep%20Heteroscedastic%20Regression/images/feb4ceee17a0d5c5b936353a9e9ddba0cdc77d7be758e71aa05befa7fda72f45.jpg)

### Tables

![2c66d5bc14f4b8e859798c28ecdc26bd4b3c43e485e9701b3a5bb1d5632b0e5c.jpg](../iclr_results/2361_Towards%20Self-Supervised%20Covariance%20Estimation%20in%20Deep%20Heteroscedastic%20Regression/tables/2c66d5bc14f4b8e859798c28ecdc26bd4b3c43e485e9701b3a5bb1d5632b0e5c.jpg)

![4ebdaa30615486c1c27eb69dec456ac64af765097786fe2016f270b37242d064.jpg](../iclr_results/2361_Towards%20Self-Supervised%20Covariance%20Estimation%20in%20Deep%20Heteroscedastic%20Regression/tables/4ebdaa30615486c1c27eb69dec456ac64af765097786fe2016f270b37242d064.jpg)

![79a25046abe76dafdcd9c1009ccf70f4970488970fd3095b69b35a49a65b7bc9.jpg](../iclr_results/2361_Towards%20Self-Supervised%20Covariance%20Estimation%20in%20Deep%20Heteroscedastic%20Regression/tables/79a25046abe76dafdcd9c1009ccf70f4970488970fd3095b69b35a49a65b7bc9.jpg)

![7d33fa3c213afec199a7302f17088b02dc88d46e85b7a8149b1f94a6ed285ad5.jpg](../iclr_results/2361_Towards%20Self-Supervised%20Covariance%20Estimation%20in%20Deep%20Heteroscedastic%20Regression/tables/7d33fa3c213afec199a7302f17088b02dc88d46e85b7a8149b1f94a6ed285ad5.jpg)

![abfb473cf10b68e088c614bb9389ab49844ea7720a36c662125e72f81dde1f7c.jpg](../iclr_results/2361_Towards%20Self-Supervised%20Covariance%20Estimation%20in%20Deep%20Heteroscedastic%20Regression/tables/abfb473cf10b68e088c614bb9389ab49844ea7720a36c662125e72f81dde1f7c.jpg)

![f2a3c47d9757f1537cd7b947dd27559fe7a5721ec2c6f04b212011b6540fce2c.jpg](../iclr_results/2361_Towards%20Self-Supervised%20Covariance%20Estimation%20in%20Deep%20Heteroscedastic%20Regression/tables/f2a3c47d9757f1537cd7b947dd27559fe7a5721ec2c6f04b212011b6540fce2c.jpg)

## Underdamped Diffusion Bridges with Applications to Sampling


### Images

![06e553f6857d37146b1d273f9ede8d19d1e34508c3af21a2f80eaf16fce1c8f3.jpg](../iclr_results/2362_Underdamped%20Diffusion%20Bridges%20with%20Applications%20to%20Sampling/images/06e553f6857d37146b1d273f9ede8d19d1e34508c3af21a2f80eaf16fce1c8f3.jpg)

![31dbb8ecc1ff56de2cab399bcf0e95f8c52871d67b087266636a622a83fd44fe.jpg](../iclr_results/2362_Underdamped%20Diffusion%20Bridges%20with%20Applications%20to%20Sampling/images/31dbb8ecc1ff56de2cab399bcf0e95f8c52871d67b087266636a622a83fd44fe.jpg)

![4367056799bdddb0a0bd52b407a22990d6ca471e1a31c3ce532e3db9b0374484.jpg](../iclr_results/2362_Underdamped%20Diffusion%20Bridges%20with%20Applications%20to%20Sampling/images/4367056799bdddb0a0bd52b407a22990d6ca471e1a31c3ce532e3db9b0374484.jpg)

![568446c67189ea5b3dd185cef8b528b26eb480d438e97e8c05b1b2164abe8292.jpg](../iclr_results/2362_Underdamped%20Diffusion%20Bridges%20with%20Applications%20to%20Sampling/images/568446c67189ea5b3dd185cef8b528b26eb480d438e97e8c05b1b2164abe8292.jpg)

![783c14a82aed1424e840dc046b0c232042690addae5a2e602e5c637c82886bdc.jpg](../iclr_results/2362_Underdamped%20Diffusion%20Bridges%20with%20Applications%20to%20Sampling/images/783c14a82aed1424e840dc046b0c232042690addae5a2e602e5c637c82886bdc.jpg)

![8cdda8edb1fba617716f76d406c2f8c59807f2840610d5f40f6a87b1f77a2155.jpg](../iclr_results/2362_Underdamped%20Diffusion%20Bridges%20with%20Applications%20to%20Sampling/images/8cdda8edb1fba617716f76d406c2f8c59807f2840610d5f40f6a87b1f77a2155.jpg)

![8d0fa318a83e114b73db3e78d82dccc6851315dda526d6c80c226b611f490c8e.jpg](../iclr_results/2362_Underdamped%20Diffusion%20Bridges%20with%20Applications%20to%20Sampling/images/8d0fa318a83e114b73db3e78d82dccc6851315dda526d6c80c226b611f490c8e.jpg)

![953aa7db7cf24939afcabce7312b7aaf17fd3bfa8e2a33f29d84281f11dff1ee.jpg](../iclr_results/2362_Underdamped%20Diffusion%20Bridges%20with%20Applications%20to%20Sampling/images/953aa7db7cf24939afcabce7312b7aaf17fd3bfa8e2a33f29d84281f11dff1ee.jpg)

![a12603a38ae6a8751041bff7c7667c38890d24a8f3966ae6dae6f22be997ec59.jpg](../iclr_results/2362_Underdamped%20Diffusion%20Bridges%20with%20Applications%20to%20Sampling/images/a12603a38ae6a8751041bff7c7667c38890d24a8f3966ae6dae6f22be997ec59.jpg)

![d2e02c6735976799e0a4ab614a661ea53a8ef436c8ba2f07da698426cbfdb02e.jpg](../iclr_results/2362_Underdamped%20Diffusion%20Bridges%20with%20Applications%20to%20Sampling/images/d2e02c6735976799e0a4ab614a661ea53a8ef436c8ba2f07da698426cbfdb02e.jpg)

![dd164a75b18552fc82cc4d590f58152f09b2b5dc2255c189b105245a11ef4649.jpg](../iclr_results/2362_Underdamped%20Diffusion%20Bridges%20with%20Applications%20to%20Sampling/images/dd164a75b18552fc82cc4d590f58152f09b2b5dc2255c189b105245a11ef4649.jpg)

### Tables

![22861b4028dd11fe96ddffc68fe77b2d2331ffdae50598662f2bf04f1cceff3b.jpg](../iclr_results/2362_Underdamped%20Diffusion%20Bridges%20with%20Applications%20to%20Sampling/tables/22861b4028dd11fe96ddffc68fe77b2d2331ffdae50598662f2bf04f1cceff3b.jpg)

![3b2661338a3560b297d93a521df3a64c456601b4aff50274925c62322f8d9cfc.jpg](../iclr_results/2362_Underdamped%20Diffusion%20Bridges%20with%20Applications%20to%20Sampling/tables/3b2661338a3560b297d93a521df3a64c456601b4aff50274925c62322f8d9cfc.jpg)

![4a1995d818ce427a6e132e2cf0281aa04a34c39520915bf337773f40dff1c2cd.jpg](../iclr_results/2362_Underdamped%20Diffusion%20Bridges%20with%20Applications%20to%20Sampling/tables/4a1995d818ce427a6e132e2cf0281aa04a34c39520915bf337773f40dff1c2cd.jpg)

![c58df663ed995fcdbef21980e18d7765077de6424ae0f11fa166810f7cad7f11.jpg](../iclr_results/2362_Underdamped%20Diffusion%20Bridges%20with%20Applications%20to%20Sampling/tables/c58df663ed995fcdbef21980e18d7765077de6424ae0f11fa166810f7cad7f11.jpg)

![f1197a10086c91df93adeea740c151ee33063546b98e1a026ff438737d5715c4.jpg](../iclr_results/2362_Underdamped%20Diffusion%20Bridges%20with%20Applications%20to%20Sampling/tables/f1197a10086c91df93adeea740c151ee33063546b98e1a026ff438737d5715c4.jpg)

## A Closer Look at Machine Unlearning for Large Language Models


### Images

![2684186bb8dc5fc819b424b24a6b5890a9cf324fdc6309ad082225d1db608830.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/images/2684186bb8dc5fc819b424b24a6b5890a9cf324fdc6309ad082225d1db608830.jpg)

![2cd9af06415b38ffaa6dac7e1ef6a43b7d1dfe41eafe10e130cbf20e74418a9b.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/images/2cd9af06415b38ffaa6dac7e1ef6a43b7d1dfe41eafe10e130cbf20e74418a9b.jpg)

![3e3d35c1d80bfcc93963cc37ed6ce4fb632df0561b2c1b46469c5b61dc77a598.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/images/3e3d35c1d80bfcc93963cc37ed6ce4fb632df0561b2c1b46469c5b61dc77a598.jpg)

![4f557099c1462b9809792a87716672b47405ecc734ba74a1d4fa1f08673757c6.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/images/4f557099c1462b9809792a87716672b47405ecc734ba74a1d4fa1f08673757c6.jpg)

![4fce30499d04b22bfea963f65663a060ab964cd9c0ab09befe6b81c9249a28f4.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/images/4fce30499d04b22bfea963f65663a060ab964cd9c0ab09befe6b81c9249a28f4.jpg)

![5d69a1bbf3dfb9be327ae48be86265cbf802078358cdfb78da42ae112377cd9a.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/images/5d69a1bbf3dfb9be327ae48be86265cbf802078358cdfb78da42ae112377cd9a.jpg)

![7719b1d76f22f635c0943cd48a092c98530c86c1d19c7d6955a4dcba8346c96a.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/images/7719b1d76f22f635c0943cd48a092c98530c86c1d19c7d6955a4dcba8346c96a.jpg)

![91b9e1764b8af6fe3c2b69b67db7ecaceadfe465f2e38be843450c06ac8ed634.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/images/91b9e1764b8af6fe3c2b69b67db7ecaceadfe465f2e38be843450c06ac8ed634.jpg)

![af68209efb598afcdd4d9c522d87d0e842b8bcf852408271646090c9feb01f3f.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/images/af68209efb598afcdd4d9c522d87d0e842b8bcf852408271646090c9feb01f3f.jpg)

![ee3227ea5a193665bf106d76c7e625dfc31e61ef3de06d10341dd7aef1bed32b.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/images/ee3227ea5a193665bf106d76c7e625dfc31e61ef3de06d10341dd7aef1bed32b.jpg)

![f8c93847281e4c18c531c95a1967e9242c920a2ed35233049882d76e0d618ca0.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/images/f8c93847281e4c18c531c95a1967e9242c920a2ed35233049882d76e0d618ca0.jpg)

### Tables

![00a14233cbff5b7d42effca08d81020933701e090094f70530f82d0eb6c46a07.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/tables/00a14233cbff5b7d42effca08d81020933701e090094f70530f82d0eb6c46a07.jpg)

![0a529fb098f96291353e8b25ffa936f745b836a9625df2b3e568e99f8a797221.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/tables/0a529fb098f96291353e8b25ffa936f745b836a9625df2b3e568e99f8a797221.jpg)

![0cc987a23fd2d42b9c9b79a27cd085073bb72d10ea189c26e4a3b78466ee248f.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/tables/0cc987a23fd2d42b9c9b79a27cd085073bb72d10ea189c26e4a3b78466ee248f.jpg)

![213b2c7d87c9728fd53bad2337b9f69be663a6cfe542430e0decdf1d7bd9d11c.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/tables/213b2c7d87c9728fd53bad2337b9f69be663a6cfe542430e0decdf1d7bd9d11c.jpg)

![2eb18bac001941f4dd1e162bdb56a1f25ad8548764ab2117333bef563652facd.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/tables/2eb18bac001941f4dd1e162bdb56a1f25ad8548764ab2117333bef563652facd.jpg)

![4ae8a483801df20a9b38d30839ba271e4a1e00a38d91d635f9a75c660cb9c4e5.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/tables/4ae8a483801df20a9b38d30839ba271e4a1e00a38d91d635f9a75c660cb9c4e5.jpg)

![53b1d3ffc82ef3fdde6df1fe0e8a175a5e13c94cd3dc055fb969ea7b7f4c28e2.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/tables/53b1d3ffc82ef3fdde6df1fe0e8a175a5e13c94cd3dc055fb969ea7b7f4c28e2.jpg)

![6231ea6d372c1c80059646373d95a5f7ef8cead906c61febf40beaa59d5a0cf0.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/tables/6231ea6d372c1c80059646373d95a5f7ef8cead906c61febf40beaa59d5a0cf0.jpg)

![77d942ee349d8d7707c41980a6670b87709b85722ef0be072a68e3afb80e0c53.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/tables/77d942ee349d8d7707c41980a6670b87709b85722ef0be072a68e3afb80e0c53.jpg)

![8c0e51282c9ca916361198b4349407683681062a10fabe92aeb47317333b164d.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/tables/8c0e51282c9ca916361198b4349407683681062a10fabe92aeb47317333b164d.jpg)

![ad6188ce067f0bd914dad9005a44e33c6ffd6dc269c1a47151bdc2811e463589.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/tables/ad6188ce067f0bd914dad9005a44e33c6ffd6dc269c1a47151bdc2811e463589.jpg)

![b0eed9f1c616985229b164d2143d065c15aae12e4d5e9f99a22630b845b7eb4d.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/tables/b0eed9f1c616985229b164d2143d065c15aae12e4d5e9f99a22630b845b7eb4d.jpg)

![b9a1643d479bf1226993212366a4634785929f5197346b273b1e54de81de0cfe.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/tables/b9a1643d479bf1226993212366a4634785929f5197346b273b1e54de81de0cfe.jpg)

![d15b8aebd8fa5f0566cefc2958e3afaed2fdc50cb8dfe944b1e01f49207ad2c0.jpg](../iclr_results/2363_A%20Closer%20Look%20at%20Machine%20Unlearning%20for%20Large%20Language%20Models/tables/d15b8aebd8fa5f0566cefc2958e3afaed2fdc50cb8dfe944b1e01f49207ad2c0.jpg)

## A Robust Method to Discover Causal or Anticausal Relation


### Images

![05afe82538b781e6bff9bed2837edbd0a63b7b897005f3212dea810f64f1f5f4.jpg](../iclr_results/2364_A%20Robust%20Method%20to%20Discover%20Causal%20or%20Anticausal%20Relation/images/05afe82538b781e6bff9bed2837edbd0a63b7b897005f3212dea810f64f1f5f4.jpg)

![0a0b51010e4d04c117ae7514ada31c98b64457a75a71f2b4f2944d028577b225.jpg](../iclr_results/2364_A%20Robust%20Method%20to%20Discover%20Causal%20or%20Anticausal%20Relation/images/0a0b51010e4d04c117ae7514ada31c98b64457a75a71f2b4f2944d028577b225.jpg)

![0dd82720772c9dbbf5d2619a4868adfba4e9be1c3d9325295939517ee718d298.jpg](../iclr_results/2364_A%20Robust%20Method%20to%20Discover%20Causal%20or%20Anticausal%20Relation/images/0dd82720772c9dbbf5d2619a4868adfba4e9be1c3d9325295939517ee718d298.jpg)

![4d537b188553ae34e5c88290a6379a6584736b0996201757cdc530a8afac4e11.jpg](../iclr_results/2364_A%20Robust%20Method%20to%20Discover%20Causal%20or%20Anticausal%20Relation/images/4d537b188553ae34e5c88290a6379a6584736b0996201757cdc530a8afac4e11.jpg)

![7e176dd2fcb76c7bc6ab74cd08b614fdd11e654d0d898ad1dec6f8b005e565ac.jpg](../iclr_results/2364_A%20Robust%20Method%20to%20Discover%20Causal%20or%20Anticausal%20Relation/images/7e176dd2fcb76c7bc6ab74cd08b614fdd11e654d0d898ad1dec6f8b005e565ac.jpg)

![c0a61b54332c2a9c8e1e8f20b8667d351cc085312498c30d52507394cf989537.jpg](../iclr_results/2364_A%20Robust%20Method%20to%20Discover%20Causal%20or%20Anticausal%20Relation/images/c0a61b54332c2a9c8e1e8f20b8667d351cc085312498c30d52507394cf989537.jpg)

### Tables

![0a8ebe481ef25de3aab40fed4024620c9718ac9e98fa23e95305a7fd53650e37.jpg](../iclr_results/2364_A%20Robust%20Method%20to%20Discover%20Causal%20or%20Anticausal%20Relation/tables/0a8ebe481ef25de3aab40fed4024620c9718ac9e98fa23e95305a7fd53650e37.jpg)

![2812a152de104215ecd3f96f0c541243d1e6599ab47ef066c98474fc6cb27998.jpg](../iclr_results/2364_A%20Robust%20Method%20to%20Discover%20Causal%20or%20Anticausal%20Relation/tables/2812a152de104215ecd3f96f0c541243d1e6599ab47ef066c98474fc6cb27998.jpg)

![50e9997662e56aabf346f560d20ac38543e18136be38887083c07780d0cc9699.jpg](../iclr_results/2364_A%20Robust%20Method%20to%20Discover%20Causal%20or%20Anticausal%20Relation/tables/50e9997662e56aabf346f560d20ac38543e18136be38887083c07780d0cc9699.jpg)

![5db707294bfd169fe09ee99b576e21850dc0b079e2cfaf92f439d5f599bfb99b.jpg](../iclr_results/2364_A%20Robust%20Method%20to%20Discover%20Causal%20or%20Anticausal%20Relation/tables/5db707294bfd169fe09ee99b576e21850dc0b079e2cfaf92f439d5f599bfb99b.jpg)

![8eae95560153b73026187615523395e0ca057bd74a0e931dbe8fa4c6981f3cfa.jpg](../iclr_results/2364_A%20Robust%20Method%20to%20Discover%20Causal%20or%20Anticausal%20Relation/tables/8eae95560153b73026187615523395e0ca057bd74a0e931dbe8fa4c6981f3cfa.jpg)

![a147a8777e2e66ae0f0ca063ad7be2d02e428900b30dab790e1bf7ecd0d4250b.jpg](../iclr_results/2364_A%20Robust%20Method%20to%20Discover%20Causal%20or%20Anticausal%20Relation/tables/a147a8777e2e66ae0f0ca063ad7be2d02e428900b30dab790e1bf7ecd0d4250b.jpg)

![a8aa424c8fad4a08dff973fbf208ca38ae2dca0df028d8626d2d3ff88c050f94.jpg](../iclr_results/2364_A%20Robust%20Method%20to%20Discover%20Causal%20or%20Anticausal%20Relation/tables/a8aa424c8fad4a08dff973fbf208ca38ae2dca0df028d8626d2d3ff88c050f94.jpg)

![adaf84b6dcdc2f64c7edbc3bc32a12b3d748d6a2d212435e2bd475f8540892d4.jpg](../iclr_results/2364_A%20Robust%20Method%20to%20Discover%20Causal%20or%20Anticausal%20Relation/tables/adaf84b6dcdc2f64c7edbc3bc32a12b3d748d6a2d212435e2bd475f8540892d4.jpg)

## Debiasing Mini-Batch Quadratics for Applications in Deep Learning


### Images

![18253c2b3f089c7b42cb53a0cf24799c3b40275dd42bbb97c032c080e8c8cdfb.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/images/18253c2b3f089c7b42cb53a0cf24799c3b40275dd42bbb97c032c080e8c8cdfb.jpg)

![199d61bff8229e8da4d66113850e707f6590df5de2818ce4f48d0bc5df532bb5.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/images/199d61bff8229e8da4d66113850e707f6590df5de2818ce4f48d0bc5df532bb5.jpg)

![2701ecf9cfcf6740c42c6119d2d49def2bac734ee5a5753cff1f7df08177a68b.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/images/2701ecf9cfcf6740c42c6119d2d49def2bac734ee5a5753cff1f7df08177a68b.jpg)

![2ace9a611230e22b7ef2eef7620bd6321c8ae388e9368d132b6580cf131ab680.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/images/2ace9a611230e22b7ef2eef7620bd6321c8ae388e9368d132b6580cf131ab680.jpg)

![2bf2b9f3b04e91321c38b5b4de338f39d1c6f6ce0a41c0efe02b809da06f34f5.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/images/2bf2b9f3b04e91321c38b5b4de338f39d1c6f6ce0a41c0efe02b809da06f34f5.jpg)

![34a16813cf159703e25961126285ce0bfc4f7290b42c66e62cdeea1ed6fd8a2c.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/images/34a16813cf159703e25961126285ce0bfc4f7290b42c66e62cdeea1ed6fd8a2c.jpg)

![4c317b3d8a079312db9df6846d41ee9604da3f9f49ce9a13e1aa443c27d4cef5.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/images/4c317b3d8a079312db9df6846d41ee9604da3f9f49ce9a13e1aa443c27d4cef5.jpg)

![4e1cf51d7b47db77c2eafee25ff66b525a2e5ef0f1b485a63744a24cfb1e2819.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/images/4e1cf51d7b47db77c2eafee25ff66b525a2e5ef0f1b485a63744a24cfb1e2819.jpg)

![61761c860aaa26c36f6feac211362b26af81a7ebd9d572a49cd132dfacacf971.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/images/61761c860aaa26c36f6feac211362b26af81a7ebd9d572a49cd132dfacacf971.jpg)

![70b317ade02018b0d43ae8ee014038213f2af355efec4d850694b14fa68678f8.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/images/70b317ade02018b0d43ae8ee014038213f2af355efec4d850694b14fa68678f8.jpg)

![75504c968ddf721a068478c574abdfcc5af27ccb7afe7482545947e7731ce9d7.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/images/75504c968ddf721a068478c574abdfcc5af27ccb7afe7482545947e7731ce9d7.jpg)

![91a397360cbf3a8f390846bb277fa984cb364a76cf99012d423be8875e93d097.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/images/91a397360cbf3a8f390846bb277fa984cb364a76cf99012d423be8875e93d097.jpg)

![92477bb02931e73baa5d9e0f76672be5ea75033eb29d2cfc2ba1e8902df5e2ee.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/images/92477bb02931e73baa5d9e0f76672be5ea75033eb29d2cfc2ba1e8902df5e2ee.jpg)

![a1c0e338cdb8ec71d9dd9bc4ad437bad1435684f4df42dd08072cf043e52a607.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/images/a1c0e338cdb8ec71d9dd9bc4ad437bad1435684f4df42dd08072cf043e52a607.jpg)

![a1e154ef94429fafb3c55bd2791ada7ac07f590b2c6292e12fdb8d22dc0c765a.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/images/a1e154ef94429fafb3c55bd2791ada7ac07f590b2c6292e12fdb8d22dc0c765a.jpg)

![a58c35ebeed7a5eb801ca897ef30b69455741d38e9bf68df8b63b64c0453e918.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/images/a58c35ebeed7a5eb801ca897ef30b69455741d38e9bf68df8b63b64c0453e918.jpg)

![b49d5b5a1fcf4d6bf6157fe8f28d25ece61b730136eb8d9fc84cb518f69b3f61.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/images/b49d5b5a1fcf4d6bf6157fe8f28d25ece61b730136eb8d9fc84cb518f69b3f61.jpg)

![bd5636608b70da8901fbef060d2e9c193c8c042a92ec09db94a01dbb403d6dcd.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/images/bd5636608b70da8901fbef060d2e9c193c8c042a92ec09db94a01dbb403d6dcd.jpg)

![c12c237f9a2887fcc26b737cc684b55f0230c02296400d315c57022dae02a22f.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/images/c12c237f9a2887fcc26b737cc684b55f0230c02296400d315c57022dae02a22f.jpg)

![c3ddd7811804d79a184683489c661dba9da5d94994cab4e1ecae20b3f0f28a70.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/images/c3ddd7811804d79a184683489c661dba9da5d94994cab4e1ecae20b3f0f28a70.jpg)

![ead5cafa638c9db786202ad6d62b3bf05f1d64233887f2bc7ea02a54f8d84fe4.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/images/ead5cafa638c9db786202ad6d62b3bf05f1d64233887f2bc7ea02a54f8d84fe4.jpg)

![ebfaa9a1d611c1972f72ef9583c0b47b9a5d166a0ef62bbaaf3efe63e829b3c3.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/images/ebfaa9a1d611c1972f72ef9583c0b47b9a5d166a0ef62bbaaf3efe63e829b3c3.jpg)

![fc8e152a2fb36ee28b783282cb37381afe31e7dfcd0ea174a29bed78d2d89204.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/images/fc8e152a2fb36ee28b783282cb37381afe31e7dfcd0ea174a29bed78d2d89204.jpg)

### Tables

![b71897973e79b388f44c4283529977791ba45ec7791fa1d8bbd02ac2eef04e6d.jpg](../iclr_results/2365_Debiasing%20Mini-Batch%20Quadratics%20for%20Applications%20in%20Deep%20Learning/tables/b71897973e79b388f44c4283529977791ba45ec7791fa1d8bbd02ac2eef04e6d.jpg)

## Scalable Benchmarking and Robust Learning for Noise-Free Ego-Motion and 3D Reconstruction from Noisy Video


### Images

![01a8ac582ceb6aaa58016820af45d5c795819cb65d81e6d676f0ecdf7f04aede.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/01a8ac582ceb6aaa58016820af45d5c795819cb65d81e6d676f0ecdf7f04aede.jpg)

![0a662e7948eccb729ce2f0837fc05f89a6720049574df7aa01fef8c0622a11a3.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/0a662e7948eccb729ce2f0837fc05f89a6720049574df7aa01fef8c0622a11a3.jpg)

![0bec97340cc79bf11f8fbc7ea807fe553611b87505442f473669d8cb2a869ac5.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/0bec97340cc79bf11f8fbc7ea807fe553611b87505442f473669d8cb2a869ac5.jpg)

![12d88c7c79bb0071d89f60bb6d336d04674d57a7f21163d81ae1dc3915ee7ef8.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/12d88c7c79bb0071d89f60bb6d336d04674d57a7f21163d81ae1dc3915ee7ef8.jpg)

![133da05a3edc35715db2d3ea2be80048ec0ee0667c6e3a159d90b79e49e34f50.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/133da05a3edc35715db2d3ea2be80048ec0ee0667c6e3a159d90b79e49e34f50.jpg)

![134557d437f395a89eed169382bf8724937e2e144910f455ebd22ac167696968.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/134557d437f395a89eed169382bf8724937e2e144910f455ebd22ac167696968.jpg)

![16dfe73ebdf7727e3300a8db06582930302a23d5a3136aa7bf6bc2a91eca13be.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/16dfe73ebdf7727e3300a8db06582930302a23d5a3136aa7bf6bc2a91eca13be.jpg)

![181cae7b1fea9364441e0c6a993fbbce92eb300e81956c00efe1794a915e3b88.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/181cae7b1fea9364441e0c6a993fbbce92eb300e81956c00efe1794a915e3b88.jpg)

![1b6d7ed48b7fe442f3ffecc952c3c2a9465e749648f0525c7bb3b9938079c8d8.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/1b6d7ed48b7fe442f3ffecc952c3c2a9465e749648f0525c7bb3b9938079c8d8.jpg)

![1bff650fcf3948c051a600635e5a57be430e735e22c36bb1f53381d6086f8c49.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/1bff650fcf3948c051a600635e5a57be430e735e22c36bb1f53381d6086f8c49.jpg)

![24674586060e000ab2cc0c7871d6aeb6cff17c9c0da1b366f163e9a7e97ea84e.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/24674586060e000ab2cc0c7871d6aeb6cff17c9c0da1b366f163e9a7e97ea84e.jpg)

![2c6be5375ab09812dd08e4eaabca1826237b8bb358e30f890027e88d2d112bff.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/2c6be5375ab09812dd08e4eaabca1826237b8bb358e30f890027e88d2d112bff.jpg)

![2f63d794fe6a9cf0d6387424c39623c33ac25320ac2dfae83bab72914cf7c9a3.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/2f63d794fe6a9cf0d6387424c39623c33ac25320ac2dfae83bab72914cf7c9a3.jpg)

![308b1079e87e04b125fd575318fb0f4a27a6d188f03d6cd00564b4be558fa3a9.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/308b1079e87e04b125fd575318fb0f4a27a6d188f03d6cd00564b4be558fa3a9.jpg)

![322d49b73039524e2b5edb1c3dbbe7deb3b882038364d8c3ee9b5dfa0fd54648.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/322d49b73039524e2b5edb1c3dbbe7deb3b882038364d8c3ee9b5dfa0fd54648.jpg)

![3b016fca53d4d73b93b8de68bba1927875964a218aa9a387d403217417e72d75.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/3b016fca53d4d73b93b8de68bba1927875964a218aa9a387d403217417e72d75.jpg)

![3d40a539e0077f8cade319e318b32d2511fe0cd8f357c3b759942d2b9c179f5f.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/3d40a539e0077f8cade319e318b32d2511fe0cd8f357c3b759942d2b9c179f5f.jpg)

![3e6cd7b42ee60e319c5f0159e0caa6736b0a4e4191ca5eb504cdc44130f5d929.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/3e6cd7b42ee60e319c5f0159e0caa6736b0a4e4191ca5eb504cdc44130f5d929.jpg)

![4845792a4d4fe76a6af7f5637e83366580a3184a691e46e0092c9449866261b2.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/4845792a4d4fe76a6af7f5637e83366580a3184a691e46e0092c9449866261b2.jpg)

![4c50e23c8fa4bba9f4b7147251b3b6359ca72a0e5b995d47f98113184ed9177d.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/4c50e23c8fa4bba9f4b7147251b3b6359ca72a0e5b995d47f98113184ed9177d.jpg)

![545a5c312e97ce23d97a37432f8adb7e67e7b9677e694d3d6bcae1b786e1276b.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/545a5c312e97ce23d97a37432f8adb7e67e7b9677e694d3d6bcae1b786e1276b.jpg)

![577dbb1d14fb97b54e38666e61eb694c7d4db5bc5c1abd902e73e5a639c4c09b.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/577dbb1d14fb97b54e38666e61eb694c7d4db5bc5c1abd902e73e5a639c4c09b.jpg)

![5e9f687ccd7c05d5048e464a12196af90053e0d8a37b72dfa57d26d64f8a9352.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/5e9f687ccd7c05d5048e464a12196af90053e0d8a37b72dfa57d26d64f8a9352.jpg)

![6014cbb02a4598a3cae18d272e772a6c5a60b4d068a855a69d440a6c35d57e21.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/6014cbb02a4598a3cae18d272e772a6c5a60b4d068a855a69d440a6c35d57e21.jpg)

![6526fef0f51b59f72cc119dfdefb738daa4474397402d17121122e12e5dcd109.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/6526fef0f51b59f72cc119dfdefb738daa4474397402d17121122e12e5dcd109.jpg)

![6aa2acfed5c86ce1333582cb988d01e90e21ab9e30f1359ba8b8cb9e9aeb422a.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/6aa2acfed5c86ce1333582cb988d01e90e21ab9e30f1359ba8b8cb9e9aeb422a.jpg)

![6f907f05c14f5c61def3c42423436e3a461c6660dfabc9ce23167c9c8e4a1298.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/6f907f05c14f5c61def3c42423436e3a461c6660dfabc9ce23167c9c8e4a1298.jpg)

![753e98c7bbaf30274159e4e5e171e4e7b1ca34f0d64cf49f1d1942971f2e1c18.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/753e98c7bbaf30274159e4e5e171e4e7b1ca34f0d64cf49f1d1942971f2e1c18.jpg)

![793940a5b7fba6c46b30b22ae7a7b52537e72950ebdc290d76631965aaff24ca.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/793940a5b7fba6c46b30b22ae7a7b52537e72950ebdc290d76631965aaff24ca.jpg)

![812439d2f8f28f71fd4530deab357d7f3e9987554dca2abcb7e01712bb10adde.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/812439d2f8f28f71fd4530deab357d7f3e9987554dca2abcb7e01712bb10adde.jpg)

![840b3fc6b986929a6bec8f686c90df0bae8e358e428b4e04ba2384eb6043c3fe.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/840b3fc6b986929a6bec8f686c90df0bae8e358e428b4e04ba2384eb6043c3fe.jpg)

![876b492b815768062dc327957873365f1e0208029679ce67b6d8d6913a31f36c.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/876b492b815768062dc327957873365f1e0208029679ce67b6d8d6913a31f36c.jpg)

![89b64060f90ca87ee5eba1e3001ee86636101762c6c06e779672ab29b11ea9cc.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/89b64060f90ca87ee5eba1e3001ee86636101762c6c06e779672ab29b11ea9cc.jpg)

![8aefe8869017f8f96d99f204459ed85afa772411773afe45860d38f2f5498e04.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/8aefe8869017f8f96d99f204459ed85afa772411773afe45860d38f2f5498e04.jpg)

![8c4498d08a0a88d502e11e3b7d7b946e714d3aadda3cf319e2ff3c244b6ebe7f.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/8c4498d08a0a88d502e11e3b7d7b946e714d3aadda3cf319e2ff3c244b6ebe7f.jpg)

![90b0dbe601f7e333b715480d87a20e85dc170038d7f98d4f16d894fdd46324ad.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/90b0dbe601f7e333b715480d87a20e85dc170038d7f98d4f16d894fdd46324ad.jpg)

![95891822119b2cfbc8ce84348f3837c97c78298e7503b1e4bc60e1ceb185ad1f.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/95891822119b2cfbc8ce84348f3837c97c78298e7503b1e4bc60e1ceb185ad1f.jpg)

![9686501cdb14be9cfb08bae4ac27593aa30f4162c6e372092542d7584d9f2523.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/9686501cdb14be9cfb08bae4ac27593aa30f4162c6e372092542d7584d9f2523.jpg)

![9b4458f3f31ba992b53ee054e17726efc2d6bdcf0839380f7f647c6875ebc1f3.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/9b4458f3f31ba992b53ee054e17726efc2d6bdcf0839380f7f647c6875ebc1f3.jpg)

![9cf257e8ead6d36ea2f1e5545653c17800adb3cb276257cfb5437c0207607701.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/9cf257e8ead6d36ea2f1e5545653c17800adb3cb276257cfb5437c0207607701.jpg)

![a2b244188dfaeaedfcd27a6b51b914a59a6798c14b207efd1d26070350157388.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/a2b244188dfaeaedfcd27a6b51b914a59a6798c14b207efd1d26070350157388.jpg)

![a6066aa23c3e75a178680931aab6291826b1c4379b96e335ea42f1aff87a1268.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/a6066aa23c3e75a178680931aab6291826b1c4379b96e335ea42f1aff87a1268.jpg)

![a89b4e923a1c820bbaf141fb4359648147600aeb46a57572bd3138fa8777c8f0.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/a89b4e923a1c820bbaf141fb4359648147600aeb46a57572bd3138fa8777c8f0.jpg)

![af7110b56616adb2cf590b1acf8ad8db4819f2adf62b47bbac23d6aec9eb7d98.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/af7110b56616adb2cf590b1acf8ad8db4819f2adf62b47bbac23d6aec9eb7d98.jpg)

![b27f640d98ba19985959d68fe18ba6ebef025b3510501afc20b858a54a8d0df3.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/b27f640d98ba19985959d68fe18ba6ebef025b3510501afc20b858a54a8d0df3.jpg)

![b411ce304bc1f7ef210a977a0f64f2d5d167b216d6a4f36aee2d246a14f3dfaa.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/b411ce304bc1f7ef210a977a0f64f2d5d167b216d6a4f36aee2d246a14f3dfaa.jpg)

![b6d5dda4a1627b61ec62216d5be903571530e3899b8e814675a3c24ad7a42ee1.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/b6d5dda4a1627b61ec62216d5be903571530e3899b8e814675a3c24ad7a42ee1.jpg)

![b6e64f65903bf62603ee2cab9f344a17e00eaa8b0e79cd0982a54b44b0c05c94.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/b6e64f65903bf62603ee2cab9f344a17e00eaa8b0e79cd0982a54b44b0c05c94.jpg)

![b6eca77716022751a217aaf6294075fd293f05ca9caa4cee5f74011b2e7615f2.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/b6eca77716022751a217aaf6294075fd293f05ca9caa4cee5f74011b2e7615f2.jpg)

![c5449c8cca88217283604dd980c0f10354ddb592dafede6b97c2cf81720eee85.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/c5449c8cca88217283604dd980c0f10354ddb592dafede6b97c2cf81720eee85.jpg)

![c7e642ce150194646e584096a7f93d901a71f3c9b4032fafce5774fbcc5e6cd5.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/c7e642ce150194646e584096a7f93d901a71f3c9b4032fafce5774fbcc5e6cd5.jpg)

![cc6d79bdddebbec871c544f9a82f6d735862892a9aa66e7a1190b64c9d41ce7b.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/cc6d79bdddebbec871c544f9a82f6d735862892a9aa66e7a1190b64c9d41ce7b.jpg)

![ce54af2f80b8edf1ec352b99da38e758ee129c97e97be3db5a09f39d223ec1ed.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/ce54af2f80b8edf1ec352b99da38e758ee129c97e97be3db5a09f39d223ec1ed.jpg)

![d3a6bd6b972a9e6b7f40b1a37ee89517bf92321a22c58d089a4c8f74d0bfea50.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/d3a6bd6b972a9e6b7f40b1a37ee89517bf92321a22c58d089a4c8f74d0bfea50.jpg)

![da56c41f2046fe420a28892c7928ac6ba7c9faf0194f00a3c2653e5f77633d53.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/da56c41f2046fe420a28892c7928ac6ba7c9faf0194f00a3c2653e5f77633d53.jpg)

![db1aa16885405ca2e51737672ac7fcc6ec0c4dfcf8d719cc20ade0cdeb19b749.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/db1aa16885405ca2e51737672ac7fcc6ec0c4dfcf8d719cc20ade0cdeb19b749.jpg)

![dba746b8ed3478dc22ff871e01006f75fc1d203a5ed02d20685f291724f8e5da.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/dba746b8ed3478dc22ff871e01006f75fc1d203a5ed02d20685f291724f8e5da.jpg)

![e73aa89f9c4760d49b62b5943fd3057d66d664b91321c19f8ed92d3863081edd.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/e73aa89f9c4760d49b62b5943fd3057d66d664b91321c19f8ed92d3863081edd.jpg)

![e774fcb05c784c4beba78c273956c74769963fbbbb03d58b2df0f2c21c339ea3.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/e774fcb05c784c4beba78c273956c74769963fbbbb03d58b2df0f2c21c339ea3.jpg)

![e96f107f9073715fec3b70ba48b13190f3ba4c4c79f535ae89b2231bf0dba816.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/e96f107f9073715fec3b70ba48b13190f3ba4c4c79f535ae89b2231bf0dba816.jpg)

![efed1f704027e8b1fe8187ee4678b481ce41828ba6cfea36a29797b3700583f8.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/efed1f704027e8b1fe8187ee4678b481ce41828ba6cfea36a29797b3700583f8.jpg)

![fad1b57cd228f93e8bf80a56bf5230f972ecedca9f9a7cf081aa403cfef6ea06.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/images/fad1b57cd228f93e8bf80a56bf5230f972ecedca9f9a7cf081aa403cfef6ea06.jpg)

### Tables

![0d9bbd3aa2463187e44b900b074ca88aa2c7f28f84f0cfc9a0747d8002dd2359.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/0d9bbd3aa2463187e44b900b074ca88aa2c7f28f84f0cfc9a0747d8002dd2359.jpg)

![1d2196d03f425acc29c3d1e9a54dad984efee6cfb6bb4e652702c393de6cb81e.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/1d2196d03f425acc29c3d1e9a54dad984efee6cfb6bb4e652702c393de6cb81e.jpg)

![22708c0f818d03d2333dee27656331beccb7c26e39d8317353566415f304ef62.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/22708c0f818d03d2333dee27656331beccb7c26e39d8317353566415f304ef62.jpg)

![347026b5a6bf4fa700c311bb52c9758ce1d8f7ef82cc5c85e5efe0dd21200462.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/347026b5a6bf4fa700c311bb52c9758ce1d8f7ef82cc5c85e5efe0dd21200462.jpg)

![3d87b666aca61a82e05f3fe2bdaaf1ba3ab89bb3a3075c1ec2e62e0fc15776af.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/3d87b666aca61a82e05f3fe2bdaaf1ba3ab89bb3a3075c1ec2e62e0fc15776af.jpg)

![498790085ca9a6b19943f26d71162b172f5bee23a3488d58402474864e175fb0.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/498790085ca9a6b19943f26d71162b172f5bee23a3488d58402474864e175fb0.jpg)

![4d2fc36a3ddc1cc361802ade12ba0ba751aee3d7cc3491e9f3e88f92bcfecb50.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/4d2fc36a3ddc1cc361802ade12ba0ba751aee3d7cc3491e9f3e88f92bcfecb50.jpg)

![4d3d922e3a49ed2a4f8a51645e92c6f01e869e4e677f2991745e4af02620eef1.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/4d3d922e3a49ed2a4f8a51645e92c6f01e869e4e677f2991745e4af02620eef1.jpg)

![54171d8fd5a3c5ec1c4aef81f0c2442ebb007184d78f30b0ef2df03eb91e9ac0.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/54171d8fd5a3c5ec1c4aef81f0c2442ebb007184d78f30b0ef2df03eb91e9ac0.jpg)

![5a8d643e3bdf2f753e32e02f505396370220ddabc07fc60659989fb253d91b2f.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/5a8d643e3bdf2f753e32e02f505396370220ddabc07fc60659989fb253d91b2f.jpg)

![6485b2d45e4fb191827260c9f7beadaa5e38b0b6389dbd3f04a6a9640c6ff615.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/6485b2d45e4fb191827260c9f7beadaa5e38b0b6389dbd3f04a6a9640c6ff615.jpg)

![7f5d54cdb17b992d7ebe08661e93ec0b14a501154eb7e26911895f55c1474373.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/7f5d54cdb17b992d7ebe08661e93ec0b14a501154eb7e26911895f55c1474373.jpg)

![815a99bc6ccf7f66a6a9f9d686d2bb935a92a3e509141b25b070310e0aa48dc8.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/815a99bc6ccf7f66a6a9f9d686d2bb935a92a3e509141b25b070310e0aa48dc8.jpg)

![8896a6687f906bd78267597f40e4d21f0ee95807fa2b5abc94d2d5b79750d064.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/8896a6687f906bd78267597f40e4d21f0ee95807fa2b5abc94d2d5b79750d064.jpg)

![88a543db8bcb58f992f13cc03d5cb6466b5e890fed2677324813dcef0b869901.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/88a543db8bcb58f992f13cc03d5cb6466b5e890fed2677324813dcef0b869901.jpg)

![8f11893d12b94fdc51b6c56562a9604306c6d24903acd3aea8d7857e67783966.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/8f11893d12b94fdc51b6c56562a9604306c6d24903acd3aea8d7857e67783966.jpg)

![9497739646dfde3b213ef7f4d32858a958e65a3992f88ce1071b95f5ebfc0c08.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/9497739646dfde3b213ef7f4d32858a958e65a3992f88ce1071b95f5ebfc0c08.jpg)

![9aad51d6464afd8c141d9b49a0737add65ae075c352bd01176b4447ae03c978e.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/9aad51d6464afd8c141d9b49a0737add65ae075c352bd01176b4447ae03c978e.jpg)

![b62b1e1def0387669b32cd54b9745d1b0c553727fdb91dc5a140569d2e581ecc.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/b62b1e1def0387669b32cd54b9745d1b0c553727fdb91dc5a140569d2e581ecc.jpg)

![bd2f585902aab78938ce1a5463fca5a29e04f8c680821f87a65229cb61b33e4f.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/bd2f585902aab78938ce1a5463fca5a29e04f8c680821f87a65229cb61b33e4f.jpg)

![bf5a430628151d1482a778e03b651ef566014a9d7c08ca363811f883177136dc.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/bf5a430628151d1482a778e03b651ef566014a9d7c08ca363811f883177136dc.jpg)

![c0d028cc65e9a309be087b8b211794a3f6dfc4d5a8c0e8996a831e40ab783b6b.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/c0d028cc65e9a309be087b8b211794a3f6dfc4d5a8c0e8996a831e40ab783b6b.jpg)

![c5144fe52ca5d78bacfd598ff1e88604695ce9a3c40152af0e93cb7c36ba85dd.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/c5144fe52ca5d78bacfd598ff1e88604695ce9a3c40152af0e93cb7c36ba85dd.jpg)

![cb599d753249f7c68f5d8f677738df81c0754add8d697326746d0a3d8cafb3d2.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/cb599d753249f7c68f5d8f677738df81c0754add8d697326746d0a3d8cafb3d2.jpg)

![cca5222a917e085c8b133a8523f51151f0fa6ae0a16a816c619a3972c708f4f8.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/cca5222a917e085c8b133a8523f51151f0fa6ae0a16a816c619a3972c708f4f8.jpg)

![cd8a34dab4e76eb0370706eb7afb6428d2b1059d77613ba8a9c719559424169d.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/cd8a34dab4e76eb0370706eb7afb6428d2b1059d77613ba8a9c719559424169d.jpg)

![ce8b10581b57e5af0b8d872ba0bbc96aa93f076aa4e1ada887983108faaaa8eb.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/ce8b10581b57e5af0b8d872ba0bbc96aa93f076aa4e1ada887983108faaaa8eb.jpg)

![cf3aeb902f94115e09eba8e87542a0fbefb5953cc3676667098680bce8ad72ac.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/cf3aeb902f94115e09eba8e87542a0fbefb5953cc3676667098680bce8ad72ac.jpg)

![cf5d1b54b0a7224d785d5c93aa0b96b3048a0db55fa9247fbfe7146a0f482324.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/cf5d1b54b0a7224d785d5c93aa0b96b3048a0db55fa9247fbfe7146a0f482324.jpg)

![d28aefdfec081731b3eb2ad523c1342e4f5394725d2ed7bd3d5718a6bde141a3.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/d28aefdfec081731b3eb2ad523c1342e4f5394725d2ed7bd3d5718a6bde141a3.jpg)

![ddb3aa9ea9f1742eb813e5e7cde39475f1149b743fbda4eb765364086bf1b5a2.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/ddb3aa9ea9f1742eb813e5e7cde39475f1149b743fbda4eb765364086bf1b5a2.jpg)

![f705217236bb92850b1bbfbfa07f6bc6f34c1933364e806e8aaf6a5f728f0d14.jpg](../iclr_results/2366_Scalable%20Benchmarking%20and%20Robust%20Learning%20for%20Noise-Free%20Ego-Motion%20and%203D%20Reconstruction%20from%20Noisy/tables/f705217236bb92850b1bbfbfa07f6bc6f34c1933364e806e8aaf6a5f728f0d14.jpg)

## SuperCorrect: Advancing Small LLM Reasoning with Thought Template Distillation and Self-Correction


### Images

![0181194c6a1c5baf1bffa9c01d0c854af3d6094c2ff15a7c54c35491ee3f93e0.jpg](../iclr_results/2367_SuperCorrect_%20Advancing%20Small%20LLM%20Reasoning%20with%20Thought%20Template%20Distillation%20and%20Self-Correction/images/0181194c6a1c5baf1bffa9c01d0c854af3d6094c2ff15a7c54c35491ee3f93e0.jpg)

![14ee8e1f9cb0d42195488fa256d990c38cb667f4bcddc098efb3af921182601f.jpg](../iclr_results/2367_SuperCorrect_%20Advancing%20Small%20LLM%20Reasoning%20with%20Thought%20Template%20Distillation%20and%20Self-Correction/images/14ee8e1f9cb0d42195488fa256d990c38cb667f4bcddc098efb3af921182601f.jpg)

![648c03dfed8487df59fc396e9f67b32cf302e12dfd7b65d200bdb65a43c6fb44.jpg](../iclr_results/2367_SuperCorrect_%20Advancing%20Small%20LLM%20Reasoning%20with%20Thought%20Template%20Distillation%20and%20Self-Correction/images/648c03dfed8487df59fc396e9f67b32cf302e12dfd7b65d200bdb65a43c6fb44.jpg)

![bbe8285d40c29028367a8b575acc9ceee6dd53a4f7d7b0bd95380e4925a931ac.jpg](../iclr_results/2367_SuperCorrect_%20Advancing%20Small%20LLM%20Reasoning%20with%20Thought%20Template%20Distillation%20and%20Self-Correction/images/bbe8285d40c29028367a8b575acc9ceee6dd53a4f7d7b0bd95380e4925a931ac.jpg)

![e15107483e34b444f4f83a177d20d8a0efca3aeb584bac175a03eb74789d6ba8.jpg](../iclr_results/2367_SuperCorrect_%20Advancing%20Small%20LLM%20Reasoning%20with%20Thought%20Template%20Distillation%20and%20Self-Correction/images/e15107483e34b444f4f83a177d20d8a0efca3aeb584bac175a03eb74789d6ba8.jpg)

### Tables

![3bd36afa5f2e0bd9d4f13d6ff91f59f4fa10ee3e19cfed08d709db69c3194a5c.jpg](../iclr_results/2367_SuperCorrect_%20Advancing%20Small%20LLM%20Reasoning%20with%20Thought%20Template%20Distillation%20and%20Self-Correction/tables/3bd36afa5f2e0bd9d4f13d6ff91f59f4fa10ee3e19cfed08d709db69c3194a5c.jpg)

![4c34005dd99afda4e8e98bdc42850f56c1642f5c491121850161f41896bf45c1.jpg](../iclr_results/2367_SuperCorrect_%20Advancing%20Small%20LLM%20Reasoning%20with%20Thought%20Template%20Distillation%20and%20Self-Correction/tables/4c34005dd99afda4e8e98bdc42850f56c1642f5c491121850161f41896bf45c1.jpg)

![5d35d4e7141aeac1a8a3e93120b59b7583df9c647c9344c3261058a85006a251.jpg](../iclr_results/2367_SuperCorrect_%20Advancing%20Small%20LLM%20Reasoning%20with%20Thought%20Template%20Distillation%20and%20Self-Correction/tables/5d35d4e7141aeac1a8a3e93120b59b7583df9c647c9344c3261058a85006a251.jpg)

![713deab1bc250ab00b4c107c5675a000ef28aaaeea587f24420a16454ae406d7.jpg](../iclr_results/2367_SuperCorrect_%20Advancing%20Small%20LLM%20Reasoning%20with%20Thought%20Template%20Distillation%20and%20Self-Correction/tables/713deab1bc250ab00b4c107c5675a000ef28aaaeea587f24420a16454ae406d7.jpg)

![8b59d7359e3063141059013564646716d0638411e5df3a4e3739faa8c1bb06e8.jpg](../iclr_results/2367_SuperCorrect_%20Advancing%20Small%20LLM%20Reasoning%20with%20Thought%20Template%20Distillation%20and%20Self-Correction/tables/8b59d7359e3063141059013564646716d0638411e5df3a4e3739faa8c1bb06e8.jpg)

![a08af97427abaffdb08d7a971df49154beecdb1371870f3e35fe07ba8ce5d750.jpg](../iclr_results/2367_SuperCorrect_%20Advancing%20Small%20LLM%20Reasoning%20with%20Thought%20Template%20Distillation%20and%20Self-Correction/tables/a08af97427abaffdb08d7a971df49154beecdb1371870f3e35fe07ba8ce5d750.jpg)

![a1f031eec7832805c7db7ef4f00c61e4e4b84b660f496328eb63e6e029fe498f.jpg](../iclr_results/2367_SuperCorrect_%20Advancing%20Small%20LLM%20Reasoning%20with%20Thought%20Template%20Distillation%20and%20Self-Correction/tables/a1f031eec7832805c7db7ef4f00c61e4e4b84b660f496328eb63e6e029fe498f.jpg)

![c6e18d78bd228221ecd43151fd87b418cc61626cede2a701620ca54bea491daf.jpg](../iclr_results/2367_SuperCorrect_%20Advancing%20Small%20LLM%20Reasoning%20with%20Thought%20Template%20Distillation%20and%20Self-Correction/tables/c6e18d78bd228221ecd43151fd87b418cc61626cede2a701620ca54bea491daf.jpg)

![c7faf55552feab688a3da57fbfa84034dfd8e6523ba76004d1ca4f8c283bd695.jpg](../iclr_results/2367_SuperCorrect_%20Advancing%20Small%20LLM%20Reasoning%20with%20Thought%20Template%20Distillation%20and%20Self-Correction/tables/c7faf55552feab688a3da57fbfa84034dfd8e6523ba76004d1ca4f8c283bd695.jpg)

![c8afe6f364a80f6cbdcaa122515a5a3f6be5a5826ac6dd17b5f4826a057a4f44.jpg](../iclr_results/2367_SuperCorrect_%20Advancing%20Small%20LLM%20Reasoning%20with%20Thought%20Template%20Distillation%20and%20Self-Correction/tables/c8afe6f364a80f6cbdcaa122515a5a3f6be5a5826ac6dd17b5f4826a057a4f44.jpg)

![ee8ed21ae4151cbd18a4b9734a65bec7a41a656e9166003ec41682e390616ec7.jpg](../iclr_results/2367_SuperCorrect_%20Advancing%20Small%20LLM%20Reasoning%20with%20Thought%20Template%20Distillation%20and%20Self-Correction/tables/ee8ed21ae4151cbd18a4b9734a65bec7a41a656e9166003ec41682e390616ec7.jpg)

![fb20892c867cfa7d02e2f8b758dce73d29a5150c64df1a876e78b76f2ac7097f.jpg](../iclr_results/2367_SuperCorrect_%20Advancing%20Small%20LLM%20Reasoning%20with%20Thought%20Template%20Distillation%20and%20Self-Correction/tables/fb20892c867cfa7d02e2f8b758dce73d29a5150c64df1a876e78b76f2ac7097f.jpg)

## EC-DIT: Scaling Diffusion Transformers with Adaptive Expert-Choice Routing


### Images

![0abf4fd29404a3653e71334850dd491e409b451600d0d47e7d6c6f58e4a4ef00.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/images/0abf4fd29404a3653e71334850dd491e409b451600d0d47e7d6c6f58e4a4ef00.jpg)

![0eb08041aefd68197536c35fd069a97cd66cfddad4bca03b705e31da6a99bfa5.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/images/0eb08041aefd68197536c35fd069a97cd66cfddad4bca03b705e31da6a99bfa5.jpg)

![15b7d2477fba2d3c5bb440a01e8f583f8b8b8548c0f7710df13d50b9a2d23d14.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/images/15b7d2477fba2d3c5bb440a01e8f583f8b8b8548c0f7710df13d50b9a2d23d14.jpg)

![1995c2ef16d25c868b4044119ef9e8d070285a51553259e2cdccad08c05b457c.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/images/1995c2ef16d25c868b4044119ef9e8d070285a51553259e2cdccad08c05b457c.jpg)

![28dd0b4c6b83be4c67441a904cd8fab25c74d035e4a839fa09d8ca5a335bf6a7.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/images/28dd0b4c6b83be4c67441a904cd8fab25c74d035e4a839fa09d8ca5a335bf6a7.jpg)

![44443c40f7f9eb3561d153a363264794eba5e9a1d2b838f4d22590b65fabf26e.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/images/44443c40f7f9eb3561d153a363264794eba5e9a1d2b838f4d22590b65fabf26e.jpg)

![6d690cf9e4c45dc341cb3d1851b94ff2bb049075fa92047b46df0021efcfd2d6.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/images/6d690cf9e4c45dc341cb3d1851b94ff2bb049075fa92047b46df0021efcfd2d6.jpg)

![6ef28a18783268f57d7dfa0263bf42b653cbbb968b32701802287e084d4089ae.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/images/6ef28a18783268f57d7dfa0263bf42b653cbbb968b32701802287e084d4089ae.jpg)

![73cc7298509349f65a05642020bb3f6ea48135e3fa92430e980e4a79ec2f6fb7.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/images/73cc7298509349f65a05642020bb3f6ea48135e3fa92430e980e4a79ec2f6fb7.jpg)

![91a164bd2794cdd4195e2ea576f9cf0b6e8448a97b0e990e8c19ff3c7fef4f4e.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/images/91a164bd2794cdd4195e2ea576f9cf0b6e8448a97b0e990e8c19ff3c7fef4f4e.jpg)

![93c975b2002bfa9e6c84ce7c61150658bc9f833093c3f8545796b7a6b6cf96e2.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/images/93c975b2002bfa9e6c84ce7c61150658bc9f833093c3f8545796b7a6b6cf96e2.jpg)

![9734c3c1973ad123bf33abba0ef9e74fdf2d554dc7f8148b0f8541f1b9f236ea.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/images/9734c3c1973ad123bf33abba0ef9e74fdf2d554dc7f8148b0f8541f1b9f236ea.jpg)

![ab3ba2f7a7f2e358d8faee2f197255b4c29a0a44bfcd181307c4cdbefbcf4d5e.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/images/ab3ba2f7a7f2e358d8faee2f197255b4c29a0a44bfcd181307c4cdbefbcf4d5e.jpg)

![b0766e04303bc36f190902f8b160de19b9ca2697d8098ab4eb9da8e50c6bf15b.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/images/b0766e04303bc36f190902f8b160de19b9ca2697d8098ab4eb9da8e50c6bf15b.jpg)

![c7f4e765bec33cde054ea333e123a5266970ff1cecbd7cd5715efb783223f6be.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/images/c7f4e765bec33cde054ea333e123a5266970ff1cecbd7cd5715efb783223f6be.jpg)

![d5c5dcc059143261b6441578da91da34468d0cd43c0f97ed3c83e01df22ee4f4.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/images/d5c5dcc059143261b6441578da91da34468d0cd43c0f97ed3c83e01df22ee4f4.jpg)

![db7524d82c1dee29aabd3704dfa02105e4ed23f5de18e9dc1cf286bffcc45897.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/images/db7524d82c1dee29aabd3704dfa02105e4ed23f5de18e9dc1cf286bffcc45897.jpg)

![f703bec7f8212bb731313e0f3ec04a19432433b7881943ad3292a07c73d06677.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/images/f703bec7f8212bb731313e0f3ec04a19432433b7881943ad3292a07c73d06677.jpg)

![fd33bb522bcec87734de1d1acce813ff8fa5d4de35619d85e31bafb1a7e67270.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/images/fd33bb522bcec87734de1d1acce813ff8fa5d4de35619d85e31bafb1a7e67270.jpg)

![ff01dadfc5219b654263fcdadf20704839d6276860a2f27f293a3c2e6c3aa902.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/images/ff01dadfc5219b654263fcdadf20704839d6276860a2f27f293a3c2e6c3aa902.jpg)

### Tables

![0618fb01460e5eac8fdb935278380c82a1be5ea1062da62ba3c35346d496b03b.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/tables/0618fb01460e5eac8fdb935278380c82a1be5ea1062da62ba3c35346d496b03b.jpg)

![3fa2c1a496a3ccaeba39d7749a751d2055fbeae391b2bc5ceefa8dd4dab25308.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/tables/3fa2c1a496a3ccaeba39d7749a751d2055fbeae391b2bc5ceefa8dd4dab25308.jpg)

![44a71e0fbc642b9636372f1f192a748e7f802126ff03189f858e1e9ac7a8ff2c.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/tables/44a71e0fbc642b9636372f1f192a748e7f802126ff03189f858e1e9ac7a8ff2c.jpg)

![82af918ef88f4861e2c7c1463f2c1dd55883d3bcf874f727fcaf261e5e763767.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/tables/82af918ef88f4861e2c7c1463f2c1dd55883d3bcf874f727fcaf261e5e763767.jpg)

![add42bf3b6b920312d517f99e2d869108154b0e8824100b596817d9257212faa.jpg](../iclr_results/2368_EC-DIT_%20Scaling%20Diffusion%20Transformers%20with%20Adaptive%20Expert-Choice%20Routing/tables/add42bf3b6b920312d517f99e2d869108154b0e8824100b596817d9257212faa.jpg)

## Predicting the Energy Landscape of Stochastic Dynamical System via  Physics-informed Self-supervised Learning


### Images

![041eddd007e65725aa1e3f979cc273a3adcf3ade9979bfbf57a6cdb5e2274b85.jpg](../iclr_results/2369_Predicting%20the%20Energy%20Landscape%20of%20Stochastic%20Dynamical%20System%20via%20%20Physics-informed%20Self-supervised/images/041eddd007e65725aa1e3f979cc273a3adcf3ade9979bfbf57a6cdb5e2274b85.jpg)

![19317b93ece76134a942e62df4eddcb2190c52f62b2083e6628d440b51ee21ae.jpg](../iclr_results/2369_Predicting%20the%20Energy%20Landscape%20of%20Stochastic%20Dynamical%20System%20via%20%20Physics-informed%20Self-supervised/images/19317b93ece76134a942e62df4eddcb2190c52f62b2083e6628d440b51ee21ae.jpg)

![26fcacf176eff0a60a8c8b63d8122a56a4c4294b1ccc9505f085cea2a5e6780c.jpg](../iclr_results/2369_Predicting%20the%20Energy%20Landscape%20of%20Stochastic%20Dynamical%20System%20via%20%20Physics-informed%20Self-supervised/images/26fcacf176eff0a60a8c8b63d8122a56a4c4294b1ccc9505f085cea2a5e6780c.jpg)

![2f0b258f5bb32c67981711250a718a79e6e9a94be163f5591e83ded9bcad84dc.jpg](../iclr_results/2369_Predicting%20the%20Energy%20Landscape%20of%20Stochastic%20Dynamical%20System%20via%20%20Physics-informed%20Self-supervised/images/2f0b258f5bb32c67981711250a718a79e6e9a94be163f5591e83ded9bcad84dc.jpg)

![3b83014cfe6265d78f77689da40b0bb27be43bed972306dafdd3067ac8f65cd3.jpg](../iclr_results/2369_Predicting%20the%20Energy%20Landscape%20of%20Stochastic%20Dynamical%20System%20via%20%20Physics-informed%20Self-supervised/images/3b83014cfe6265d78f77689da40b0bb27be43bed972306dafdd3067ac8f65cd3.jpg)

![49f1a6675e8d0a7575008901a38b3bbaf6ba444b0ba83ade0bb29c26051a5bbd.jpg](../iclr_results/2369_Predicting%20the%20Energy%20Landscape%20of%20Stochastic%20Dynamical%20System%20via%20%20Physics-informed%20Self-supervised/images/49f1a6675e8d0a7575008901a38b3bbaf6ba444b0ba83ade0bb29c26051a5bbd.jpg)

![7a3c14cb9a17b7be43fc04476ebd32492a23005af6dbac05c4bd0693a981873b.jpg](../iclr_results/2369_Predicting%20the%20Energy%20Landscape%20of%20Stochastic%20Dynamical%20System%20via%20%20Physics-informed%20Self-supervised/images/7a3c14cb9a17b7be43fc04476ebd32492a23005af6dbac05c4bd0693a981873b.jpg)

![b677113baa661678d0c6124626787902ee0f742ede80fec6ddd89d27c2d3a554.jpg](../iclr_results/2369_Predicting%20the%20Energy%20Landscape%20of%20Stochastic%20Dynamical%20System%20via%20%20Physics-informed%20Self-supervised/images/b677113baa661678d0c6124626787902ee0f742ede80fec6ddd89d27c2d3a554.jpg)

![cdae543d1cf261abd58105b0720066acc95b52d25e6bbfede8ac57081aff7a59.jpg](../iclr_results/2369_Predicting%20the%20Energy%20Landscape%20of%20Stochastic%20Dynamical%20System%20via%20%20Physics-informed%20Self-supervised/images/cdae543d1cf261abd58105b0720066acc95b52d25e6bbfede8ac57081aff7a59.jpg)

![ce64d8cd4b8a9f7bd82b5d03840755e54c2e7ede50e44982657a71782b904ae6.jpg](../iclr_results/2369_Predicting%20the%20Energy%20Landscape%20of%20Stochastic%20Dynamical%20System%20via%20%20Physics-informed%20Self-supervised/images/ce64d8cd4b8a9f7bd82b5d03840755e54c2e7ede50e44982657a71782b904ae6.jpg)

![d93312f686bf424344cbeadeb7fcdcb0eb714cd88ac29714cccb79657acb92a9.jpg](../iclr_results/2369_Predicting%20the%20Energy%20Landscape%20of%20Stochastic%20Dynamical%20System%20via%20%20Physics-informed%20Self-supervised/images/d93312f686bf424344cbeadeb7fcdcb0eb714cd88ac29714cccb79657acb92a9.jpg)

![f2384825b92fdae7e6f679e0c381f37ca64f0d7d5c3269027ccf7743f846cce9.jpg](../iclr_results/2369_Predicting%20the%20Energy%20Landscape%20of%20Stochastic%20Dynamical%20System%20via%20%20Physics-informed%20Self-supervised/images/f2384825b92fdae7e6f679e0c381f37ca64f0d7d5c3269027ccf7743f846cce9.jpg)

### Tables

![11f4b31881a6da235e09304ad47bcdcef06e8824b8dc68541a46e1b82b9c3814.jpg](../iclr_results/2369_Predicting%20the%20Energy%20Landscape%20of%20Stochastic%20Dynamical%20System%20via%20%20Physics-informed%20Self-supervised/tables/11f4b31881a6da235e09304ad47bcdcef06e8824b8dc68541a46e1b82b9c3814.jpg)

![2db79d38505f52cb61a69f4000306bae0a80fb3ed78272ef9c160b1da138d4d8.jpg](../iclr_results/2369_Predicting%20the%20Energy%20Landscape%20of%20Stochastic%20Dynamical%20System%20via%20%20Physics-informed%20Self-supervised/tables/2db79d38505f52cb61a69f4000306bae0a80fb3ed78272ef9c160b1da138d4d8.jpg)

![3a68329242bda49665a1c96a47de13a92b6c3a6fe49a169bec90fd1a58129917.jpg](../iclr_results/2369_Predicting%20the%20Energy%20Landscape%20of%20Stochastic%20Dynamical%20System%20via%20%20Physics-informed%20Self-supervised/tables/3a68329242bda49665a1c96a47de13a92b6c3a6fe49a169bec90fd1a58129917.jpg)

![a38a94d2b649b8158168d6bcc5cbccb9de865f01d97b414f15fe7db2fe8954a9.jpg](../iclr_results/2369_Predicting%20the%20Energy%20Landscape%20of%20Stochastic%20Dynamical%20System%20via%20%20Physics-informed%20Self-supervised/tables/a38a94d2b649b8158168d6bcc5cbccb9de865f01d97b414f15fe7db2fe8954a9.jpg)

![aa20f138d15616e9653e18cc2c3815af5f5054d73b82656bbf573303cf9a2361.jpg](../iclr_results/2369_Predicting%20the%20Energy%20Landscape%20of%20Stochastic%20Dynamical%20System%20via%20%20Physics-informed%20Self-supervised/tables/aa20f138d15616e9653e18cc2c3815af5f5054d73b82656bbf573303cf9a2361.jpg)

![aa7e938553c399b88f88489489687b58ca99fd93d8281482b4518cf5998d8259.jpg](../iclr_results/2369_Predicting%20the%20Energy%20Landscape%20of%20Stochastic%20Dynamical%20System%20via%20%20Physics-informed%20Self-supervised/tables/aa7e938553c399b88f88489489687b58ca99fd93d8281482b4518cf5998d8259.jpg)

![c61167f74422e811306d4d0da41567ee0d59fa1d4455031f79d23634e1e8c466.jpg](../iclr_results/2369_Predicting%20the%20Energy%20Landscape%20of%20Stochastic%20Dynamical%20System%20via%20%20Physics-informed%20Self-supervised/tables/c61167f74422e811306d4d0da41567ee0d59fa1d4455031f79d23634e1e8c466.jpg)

![e963f49ef2d411ee3fe1873ae6cd720deca2e911eb1313715f06daf7c2105d4d.jpg](../iclr_results/2369_Predicting%20the%20Energy%20Landscape%20of%20Stochastic%20Dynamical%20System%20via%20%20Physics-informed%20Self-supervised/tables/e963f49ef2d411ee3fe1873ae6cd720deca2e911eb1313715f06daf7c2105d4d.jpg)

## Beyond Random Masking: When Dropout meets Graph Convolutional Networks


### Images

![45c676814417e5ddc9ba6edb2295f143ee7d0a477dd38cb7d996268bd4cd288e.jpg](../iclr_results/2370_Beyond%20Random%20Masking_%20When%20Dropout%20meets%20Graph%20Convolutional%20Networks/images/45c676814417e5ddc9ba6edb2295f143ee7d0a477dd38cb7d996268bd4cd288e.jpg)

![485f2ae19379528880cfd8582f291f720f0dd7bd0c7887482b21f832cf0deea0.jpg](../iclr_results/2370_Beyond%20Random%20Masking_%20When%20Dropout%20meets%20Graph%20Convolutional%20Networks/images/485f2ae19379528880cfd8582f291f720f0dd7bd0c7887482b21f832cf0deea0.jpg)

![66689bffaac252b397a27ab32f26ba67fb0b5676affb2debe1631eed3240ba5c.jpg](../iclr_results/2370_Beyond%20Random%20Masking_%20When%20Dropout%20meets%20Graph%20Convolutional%20Networks/images/66689bffaac252b397a27ab32f26ba67fb0b5676affb2debe1631eed3240ba5c.jpg)

![8e5ca74f65edb72aa4dca95f495ad73df00d2a31a18e73ab69f2d7c07f7f57ad.jpg](../iclr_results/2370_Beyond%20Random%20Masking_%20When%20Dropout%20meets%20Graph%20Convolutional%20Networks/images/8e5ca74f65edb72aa4dca95f495ad73df00d2a31a18e73ab69f2d7c07f7f57ad.jpg)

![aadc67e3b7a0e3281adebcb848755b1af3477ac81981bb3ff693fbae3907ea15.jpg](../iclr_results/2370_Beyond%20Random%20Masking_%20When%20Dropout%20meets%20Graph%20Convolutional%20Networks/images/aadc67e3b7a0e3281adebcb848755b1af3477ac81981bb3ff693fbae3907ea15.jpg)

![b1aa57ba37495a0df107fc656e170afb7aa5f6ac913ec7687ae73da709f330cd.jpg](../iclr_results/2370_Beyond%20Random%20Masking_%20When%20Dropout%20meets%20Graph%20Convolutional%20Networks/images/b1aa57ba37495a0df107fc656e170afb7aa5f6ac913ec7687ae73da709f330cd.jpg)

![bc0e62564e8a6285c4693774bc0e2e8eec7958d18ebd78af9679e3fc765c182b.jpg](../iclr_results/2370_Beyond%20Random%20Masking_%20When%20Dropout%20meets%20Graph%20Convolutional%20Networks/images/bc0e62564e8a6285c4693774bc0e2e8eec7958d18ebd78af9679e3fc765c182b.jpg)

![e3f15131af0b6a76f5a40fbc87a00ceb2e7f126bbb78ee60857c64963cbc139d.jpg](../iclr_results/2370_Beyond%20Random%20Masking_%20When%20Dropout%20meets%20Graph%20Convolutional%20Networks/images/e3f15131af0b6a76f5a40fbc87a00ceb2e7f126bbb78ee60857c64963cbc139d.jpg)

![e7221427e54326a938081bb76967d7fde12987a24e416d2e0b1b061a2ae6bc9e.jpg](../iclr_results/2370_Beyond%20Random%20Masking_%20When%20Dropout%20meets%20Graph%20Convolutional%20Networks/images/e7221427e54326a938081bb76967d7fde12987a24e416d2e0b1b061a2ae6bc9e.jpg)

![ed697bc8b00d4fae19b7d26dadaf628c762af8709f0109fb1a5c0a1408aee9c6.jpg](../iclr_results/2370_Beyond%20Random%20Masking_%20When%20Dropout%20meets%20Graph%20Convolutional%20Networks/images/ed697bc8b00d4fae19b7d26dadaf628c762af8709f0109fb1a5c0a1408aee9c6.jpg)

### Tables

![1e07f057469016f4b0af483ffc5126c710fe89ed843d2548b76af6690511a7f8.jpg](../iclr_results/2370_Beyond%20Random%20Masking_%20When%20Dropout%20meets%20Graph%20Convolutional%20Networks/tables/1e07f057469016f4b0af483ffc5126c710fe89ed843d2548b76af6690511a7f8.jpg)

![6038e1ab0ca191895b6660e824c04e41398afc37dc64ceec4cb931a0c9f02141.jpg](../iclr_results/2370_Beyond%20Random%20Masking_%20When%20Dropout%20meets%20Graph%20Convolutional%20Networks/tables/6038e1ab0ca191895b6660e824c04e41398afc37dc64ceec4cb931a0c9f02141.jpg)

![c7995f4402e5ad63fe3f02f809e50a3d134558f4c9cbd62c5c417edc874bef9c.jpg](../iclr_results/2370_Beyond%20Random%20Masking_%20When%20Dropout%20meets%20Graph%20Convolutional%20Networks/tables/c7995f4402e5ad63fe3f02f809e50a3d134558f4c9cbd62c5c417edc874bef9c.jpg)

![dad12b4ca1dab30d787e713d06ca3d2f7197fc9f2e8ed599c1064be263d3d70b.jpg](../iclr_results/2370_Beyond%20Random%20Masking_%20When%20Dropout%20meets%20Graph%20Convolutional%20Networks/tables/dad12b4ca1dab30d787e713d06ca3d2f7197fc9f2e8ed599c1064be263d3d70b.jpg)

![f3aac6f68482677cd62e05dbf36da4c0e93e5001961166f179267369470943f6.jpg](../iclr_results/2370_Beyond%20Random%20Masking_%20When%20Dropout%20meets%20Graph%20Convolutional%20Networks/tables/f3aac6f68482677cd62e05dbf36da4c0e93e5001961166f179267369470943f6.jpg)

## Tight Clusters Make Specialized Experts


### Images

![247eeed5dcab41ced43e241f381b9fa05d7c470bc430de21986d71d8d09a1357.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/images/247eeed5dcab41ced43e241f381b9fa05d7c470bc430de21986d71d8d09a1357.jpg)

![7faa80df90babfbf23c516560cce08b520595aeeb0a14921699d5f32d6ffa1b6.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/images/7faa80df90babfbf23c516560cce08b520595aeeb0a14921699d5f32d6ffa1b6.jpg)

![a646e7c29181b4d90c63eeb7af20a72c3193d66113615191776d365f99c4af1b.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/images/a646e7c29181b4d90c63eeb7af20a72c3193d66113615191776d365f99c4af1b.jpg)

![ade193fecf2473f6acf371f0d0bee164773498681e5b4f9cb010b6d6ea6d422a.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/images/ade193fecf2473f6acf371f0d0bee164773498681e5b4f9cb010b6d6ea6d422a.jpg)

![d0a6d19745b0d614d9156502cbfdbd98addedcab3c05578888e56bdaaabc7ca6.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/images/d0a6d19745b0d614d9156502cbfdbd98addedcab3c05578888e56bdaaabc7ca6.jpg)

### Tables

![0a8affa43887f0367f7ef7399380c57e64e5055dd5e5d6f1ebf5d7eeb7e03103.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/tables/0a8affa43887f0367f7ef7399380c57e64e5055dd5e5d6f1ebf5d7eeb7e03103.jpg)

![10a62c8f8cbbd82c1e882bc68064bdb2283977126572a21c2d17ea378d37ff8b.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/tables/10a62c8f8cbbd82c1e882bc68064bdb2283977126572a21c2d17ea378d37ff8b.jpg)

![10e500f7768a06e86d82bb439f26db68c5a09e79ce0bdd3fc8f69a503070eec0.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/tables/10e500f7768a06e86d82bb439f26db68c5a09e79ce0bdd3fc8f69a503070eec0.jpg)

![15cad287d751f88a7adcd2672da08fcd08ecf50bc8d0333afa3bb055d78a86fa.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/tables/15cad287d751f88a7adcd2672da08fcd08ecf50bc8d0333afa3bb055d78a86fa.jpg)

![66d14c4e94267f0b79f032b0c7329d974284664cf9c202f8f2860628042f9b0c.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/tables/66d14c4e94267f0b79f032b0c7329d974284664cf9c202f8f2860628042f9b0c.jpg)

![735beaac39140b681ba91c37eb0ef7ef6c54a166aa1e52bb57d948d8a1a936df.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/tables/735beaac39140b681ba91c37eb0ef7ef6c54a166aa1e52bb57d948d8a1a936df.jpg)

![84cd3b6ce47af599177a7ef0e53f7eac4b3a2bdd70669eb2c78634f38d91bebc.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/tables/84cd3b6ce47af599177a7ef0e53f7eac4b3a2bdd70669eb2c78634f38d91bebc.jpg)

![9a0f8fea1a274a5d07dfa5b6bdcf1c86c623e42ea904346441f3c6b707b7d33f.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/tables/9a0f8fea1a274a5d07dfa5b6bdcf1c86c623e42ea904346441f3c6b707b7d33f.jpg)

![a15c9f3808bac838be97be3aa99f12dc7de60d964d4efda6798934b038b09cc6.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/tables/a15c9f3808bac838be97be3aa99f12dc7de60d964d4efda6798934b038b09cc6.jpg)

![a35c203bfb7664ddf96e742cc883e8d042e4471546f7102a9051d3b90d47fb50.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/tables/a35c203bfb7664ddf96e742cc883e8d042e4471546f7102a9051d3b90d47fb50.jpg)

![a6d5ebc01b8f0ebf38f1d82520c0b647a14fb29b4b7d922bbb83faaf6d1e731c.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/tables/a6d5ebc01b8f0ebf38f1d82520c0b647a14fb29b4b7d922bbb83faaf6d1e731c.jpg)

![c6252ed0585167c58aa2961cf2b00af8db31f6c8e6e8f527243be2fc6bdbd999.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/tables/c6252ed0585167c58aa2961cf2b00af8db31f6c8e6e8f527243be2fc6bdbd999.jpg)

![d358e0153aa80c841a47c30f91ec275318bb2beb503de167072d3874502e0d07.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/tables/d358e0153aa80c841a47c30f91ec275318bb2beb503de167072d3874502e0d07.jpg)

![d4879c59514c761936847aaf174267aab0a49161e9b91ea5f92c0603121af0fa.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/tables/d4879c59514c761936847aaf174267aab0a49161e9b91ea5f92c0603121af0fa.jpg)

![d7b316ead4bae074121a4a68dac81d7e56472d8fb0a5c4c9be52ccadcb354c81.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/tables/d7b316ead4bae074121a4a68dac81d7e56472d8fb0a5c4c9be52ccadcb354c81.jpg)

![da303526c155c0f1bf2084f31b14d5196f4eb591d93e3aa1e786dbef9324a061.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/tables/da303526c155c0f1bf2084f31b14d5196f4eb591d93e3aa1e786dbef9324a061.jpg)

![ea0a34d802d28a2d84bcc283598dabc3276d29ee6032b03f1ada4274f2cb5196.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/tables/ea0a34d802d28a2d84bcc283598dabc3276d29ee6032b03f1ada4274f2cb5196.jpg)

![fbd51e3b72eb8ddaeadca12ba819efc1fbf332a99b56f44dc98780d3dff34707.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/tables/fbd51e3b72eb8ddaeadca12ba819efc1fbf332a99b56f44dc98780d3dff34707.jpg)

![fceae9a259ecb9bb6ef134c547276f0e1b26ed15eacb61f3b29e3a8fc517cc02.jpg](../iclr_results/2371_Tight%20Clusters%20Make%20Specialized%20Experts/tables/fceae9a259ecb9bb6ef134c547276f0e1b26ed15eacb61f3b29e3a8fc517cc02.jpg)

## Domain Guidance: A Simple Transfer Approach for a Pre-trained Diffusion Model


### Images

![05a62bbf4b17eed4564b42980bcff40f53d3749bbbeb8c4685c7aa67c448b014.jpg](../iclr_results/2372_Domain%20Guidance_%20A%20Simple%20Transfer%20Approach%20for%20a%20Pre-trained%20Diffusion%20Model/images/05a62bbf4b17eed4564b42980bcff40f53d3749bbbeb8c4685c7aa67c448b014.jpg)

![1b9af3aa87b83fc61d483dd00c8550edfd554f32e5312f7362ce9ec993ba98b7.jpg](../iclr_results/2372_Domain%20Guidance_%20A%20Simple%20Transfer%20Approach%20for%20a%20Pre-trained%20Diffusion%20Model/images/1b9af3aa87b83fc61d483dd00c8550edfd554f32e5312f7362ce9ec993ba98b7.jpg)

![418ffcc15860132a5550dcad8c9511c6f626d7b4819311edcb17e56e593b048e.jpg](../iclr_results/2372_Domain%20Guidance_%20A%20Simple%20Transfer%20Approach%20for%20a%20Pre-trained%20Diffusion%20Model/images/418ffcc15860132a5550dcad8c9511c6f626d7b4819311edcb17e56e593b048e.jpg)

![4d903778774e3423ebba2a42b4ee5583eb731c9102a941ba4f71a6217a8688ad.jpg](../iclr_results/2372_Domain%20Guidance_%20A%20Simple%20Transfer%20Approach%20for%20a%20Pre-trained%20Diffusion%20Model/images/4d903778774e3423ebba2a42b4ee5583eb731c9102a941ba4f71a6217a8688ad.jpg)

![699513bfe4275f70bb6283a4e342eba58f2ed1e91ba2ad35c55dea116f6467fb.jpg](../iclr_results/2372_Domain%20Guidance_%20A%20Simple%20Transfer%20Approach%20for%20a%20Pre-trained%20Diffusion%20Model/images/699513bfe4275f70bb6283a4e342eba58f2ed1e91ba2ad35c55dea116f6467fb.jpg)

![a0e532e714f2725b733ae113d021af9ac3cafb4a975c3f92f2b67cdc157548b2.jpg](../iclr_results/2372_Domain%20Guidance_%20A%20Simple%20Transfer%20Approach%20for%20a%20Pre-trained%20Diffusion%20Model/images/a0e532e714f2725b733ae113d021af9ac3cafb4a975c3f92f2b67cdc157548b2.jpg)

![dd68d14fba95ae6734d8d5d45d2b8179688bd833467bf68c3e1df4b00204b49c.jpg](../iclr_results/2372_Domain%20Guidance_%20A%20Simple%20Transfer%20Approach%20for%20a%20Pre-trained%20Diffusion%20Model/images/dd68d14fba95ae6734d8d5d45d2b8179688bd833467bf68c3e1df4b00204b49c.jpg)

### Tables

![057e3cb64283adf364e0cc3c6dc4d8e8f784e2ffc543b77145ff35a38240bbef.jpg](../iclr_results/2372_Domain%20Guidance_%20A%20Simple%20Transfer%20Approach%20for%20a%20Pre-trained%20Diffusion%20Model/tables/057e3cb64283adf364e0cc3c6dc4d8e8f784e2ffc543b77145ff35a38240bbef.jpg)

![167f3875a771d542d1aa3efe3e2f0acd9120599bf1adc586bf1262e3daadc8da.jpg](../iclr_results/2372_Domain%20Guidance_%20A%20Simple%20Transfer%20Approach%20for%20a%20Pre-trained%20Diffusion%20Model/tables/167f3875a771d542d1aa3efe3e2f0acd9120599bf1adc586bf1262e3daadc8da.jpg)

![3a45fe4e5068ccb67496bbf72596f605ed8c2dfa76089476a98b3bfeb769f5b4.jpg](../iclr_results/2372_Domain%20Guidance_%20A%20Simple%20Transfer%20Approach%20for%20a%20Pre-trained%20Diffusion%20Model/tables/3a45fe4e5068ccb67496bbf72596f605ed8c2dfa76089476a98b3bfeb769f5b4.jpg)

![6bae03cf348e6e72919f322c25587302bed74d88d91a5a8fe2ccaa96899c1e4c.jpg](../iclr_results/2372_Domain%20Guidance_%20A%20Simple%20Transfer%20Approach%20for%20a%20Pre-trained%20Diffusion%20Model/tables/6bae03cf348e6e72919f322c25587302bed74d88d91a5a8fe2ccaa96899c1e4c.jpg)

![74b5386f67e1b8ef1d7659ed1ca111a3ca009c468ac7310477f5bcfb95370019.jpg](../iclr_results/2372_Domain%20Guidance_%20A%20Simple%20Transfer%20Approach%20for%20a%20Pre-trained%20Diffusion%20Model/tables/74b5386f67e1b8ef1d7659ed1ca111a3ca009c468ac7310477f5bcfb95370019.jpg)

![9a3931d9b9073925b11eb860e21b8ceaef53b627f55f57ff0b296c4d07d8015e.jpg](../iclr_results/2372_Domain%20Guidance_%20A%20Simple%20Transfer%20Approach%20for%20a%20Pre-trained%20Diffusion%20Model/tables/9a3931d9b9073925b11eb860e21b8ceaef53b627f55f57ff0b296c4d07d8015e.jpg)

![a6fe90a52877b65920033ad1918dd0ae3c49060929d7015838f7427bd412668f.jpg](../iclr_results/2372_Domain%20Guidance_%20A%20Simple%20Transfer%20Approach%20for%20a%20Pre-trained%20Diffusion%20Model/tables/a6fe90a52877b65920033ad1918dd0ae3c49060929d7015838f7427bd412668f.jpg)

![c025c4acea7b4ee21067e73783ebfa96ce808d68bccdc842f7e7d22764e63c86.jpg](../iclr_results/2372_Domain%20Guidance_%20A%20Simple%20Transfer%20Approach%20for%20a%20Pre-trained%20Diffusion%20Model/tables/c025c4acea7b4ee21067e73783ebfa96ce808d68bccdc842f7e7d22764e63c86.jpg)

![d5b5f059f181681a87065fcdc3be52c1fe372cd84b3ca2f76c6ad123ffc97675.jpg](../iclr_results/2372_Domain%20Guidance_%20A%20Simple%20Transfer%20Approach%20for%20a%20Pre-trained%20Diffusion%20Model/tables/d5b5f059f181681a87065fcdc3be52c1fe372cd84b3ca2f76c6ad123ffc97675.jpg)

![dc2e5ddf587224f9050929a79e1ca6250eb61fd3414965961147b8ce78aff598.jpg](../iclr_results/2372_Domain%20Guidance_%20A%20Simple%20Transfer%20Approach%20for%20a%20Pre-trained%20Diffusion%20Model/tables/dc2e5ddf587224f9050929a79e1ca6250eb61fd3414965961147b8ce78aff598.jpg)

![fa11aec49c5634ecc5a5c7690d5c945f958a8387c7a1ac31eb4f41c983859e2e.jpg](../iclr_results/2372_Domain%20Guidance_%20A%20Simple%20Transfer%20Approach%20for%20a%20Pre-trained%20Diffusion%20Model/tables/fa11aec49c5634ecc5a5c7690d5c945f958a8387c7a1ac31eb4f41c983859e2e.jpg)

## RAG-DDR: Optimizing Retrieval-Augmented Generation Using Differentiable Data Rewards


### Images

![25cf5f5cd06a2897192f0b1034a71d17bb581a0eb8c2f3ae7f7169e3c170cf08.jpg](../iclr_results/2373_RAG-DDR_%20Optimizing%20Retrieval-Augmented%20Generation%20Using%20Differentiable%20Data%20Rewards/images/25cf5f5cd06a2897192f0b1034a71d17bb581a0eb8c2f3ae7f7169e3c170cf08.jpg)

![33f4bd58aeabb3fc0f843e17bc6c27b2ee8aa9e2a9f7e8d82ca9b8281abb538b.jpg](../iclr_results/2373_RAG-DDR_%20Optimizing%20Retrieval-Augmented%20Generation%20Using%20Differentiable%20Data%20Rewards/images/33f4bd58aeabb3fc0f843e17bc6c27b2ee8aa9e2a9f7e8d82ca9b8281abb538b.jpg)

![35f670ef607c5042272ea63a94630d3cc2b44f9111dd727dad9099f47792e025.jpg](../iclr_results/2373_RAG-DDR_%20Optimizing%20Retrieval-Augmented%20Generation%20Using%20Differentiable%20Data%20Rewards/images/35f670ef607c5042272ea63a94630d3cc2b44f9111dd727dad9099f47792e025.jpg)

![481b7d334c666ba1aed5308b9f832eab4e8bf7bc4915676062ee1d7e5b6db7d3.jpg](../iclr_results/2373_RAG-DDR_%20Optimizing%20Retrieval-Augmented%20Generation%20Using%20Differentiable%20Data%20Rewards/images/481b7d334c666ba1aed5308b9f832eab4e8bf7bc4915676062ee1d7e5b6db7d3.jpg)

![7408a4739a5fd69ee4c9eddb02c6827b74c4da574dca348f25c4d644419b5b54.jpg](../iclr_results/2373_RAG-DDR_%20Optimizing%20Retrieval-Augmented%20Generation%20Using%20Differentiable%20Data%20Rewards/images/7408a4739a5fd69ee4c9eddb02c6827b74c4da574dca348f25c4d644419b5b54.jpg)

![aa8e3452ded61718a4f3ac76a88ba71bcc6df0b71fdf224b0fc3ad581e3d87f5.jpg](../iclr_results/2373_RAG-DDR_%20Optimizing%20Retrieval-Augmented%20Generation%20Using%20Differentiable%20Data%20Rewards/images/aa8e3452ded61718a4f3ac76a88ba71bcc6df0b71fdf224b0fc3ad581e3d87f5.jpg)

![ab3b2d689de88915ae8f358cf289738be910f7e38a993571edc6c1bda3b89056.jpg](../iclr_results/2373_RAG-DDR_%20Optimizing%20Retrieval-Augmented%20Generation%20Using%20Differentiable%20Data%20Rewards/images/ab3b2d689de88915ae8f358cf289738be910f7e38a993571edc6c1bda3b89056.jpg)

![d927508d92e252d5d1fbf5aba256c9e6cbda5a87270d06b650462e09bec75cb2.jpg](../iclr_results/2373_RAG-DDR_%20Optimizing%20Retrieval-Augmented%20Generation%20Using%20Differentiable%20Data%20Rewards/images/d927508d92e252d5d1fbf5aba256c9e6cbda5a87270d06b650462e09bec75cb2.jpg)

### Tables

![035f45188bd7bd0effa33db98afc739814872fde5ab3bb277c041b5b8a7a805b.jpg](../iclr_results/2373_RAG-DDR_%20Optimizing%20Retrieval-Augmented%20Generation%20Using%20Differentiable%20Data%20Rewards/tables/035f45188bd7bd0effa33db98afc739814872fde5ab3bb277c041b5b8a7a805b.jpg)

![29b95df58597ad33cee065572a02e1251e3cc868cb49534c765317de392f16e6.jpg](../iclr_results/2373_RAG-DDR_%20Optimizing%20Retrieval-Augmented%20Generation%20Using%20Differentiable%20Data%20Rewards/tables/29b95df58597ad33cee065572a02e1251e3cc868cb49534c765317de392f16e6.jpg)

![496e3274e4ae374ddc25e1775294033e0ac9ce1ded7435ceeb9b776bae783d4e.jpg](../iclr_results/2373_RAG-DDR_%20Optimizing%20Retrieval-Augmented%20Generation%20Using%20Differentiable%20Data%20Rewards/tables/496e3274e4ae374ddc25e1775294033e0ac9ce1ded7435ceeb9b776bae783d4e.jpg)

![4a05023557262fd8c6270183d16bcf6f2a0e986a6bd88c757a14669ad754a5ea.jpg](../iclr_results/2373_RAG-DDR_%20Optimizing%20Retrieval-Augmented%20Generation%20Using%20Differentiable%20Data%20Rewards/tables/4a05023557262fd8c6270183d16bcf6f2a0e986a6bd88c757a14669ad754a5ea.jpg)

![70bb7498fc8ee1e375fd59d4bffcd6c133dd84672b07a16afb244dfcd235ed46.jpg](../iclr_results/2373_RAG-DDR_%20Optimizing%20Retrieval-Augmented%20Generation%20Using%20Differentiable%20Data%20Rewards/tables/70bb7498fc8ee1e375fd59d4bffcd6c133dd84672b07a16afb244dfcd235ed46.jpg)

![7497b93531288980df537f9ef398dfc81d53c52dba18c4c511c14fad1659c235.jpg](../iclr_results/2373_RAG-DDR_%20Optimizing%20Retrieval-Augmented%20Generation%20Using%20Differentiable%20Data%20Rewards/tables/7497b93531288980df537f9ef398dfc81d53c52dba18c4c511c14fad1659c235.jpg)

![adb584cb48d17b8cc05726f770e1e837925a91dc7cea88db0b20d69529527608.jpg](../iclr_results/2373_RAG-DDR_%20Optimizing%20Retrieval-Augmented%20Generation%20Using%20Differentiable%20Data%20Rewards/tables/adb584cb48d17b8cc05726f770e1e837925a91dc7cea88db0b20d69529527608.jpg)

![cd3943cf839c50be66724ae58ff347cee23d47e2ecf428f84a96f70432bc28d3.jpg](../iclr_results/2373_RAG-DDR_%20Optimizing%20Retrieval-Augmented%20Generation%20Using%20Differentiable%20Data%20Rewards/tables/cd3943cf839c50be66724ae58ff347cee23d47e2ecf428f84a96f70432bc28d3.jpg)

![d45445678d069c2369ca29c4f84a6a222ed55502297329b7cbba3b2f8cc51dfb.jpg](../iclr_results/2373_RAG-DDR_%20Optimizing%20Retrieval-Augmented%20Generation%20Using%20Differentiable%20Data%20Rewards/tables/d45445678d069c2369ca29c4f84a6a222ed55502297329b7cbba3b2f8cc51dfb.jpg)

![ec933a04e639d5683462d15cd6d7eb5fb388c463fff413640916928d22bba1dc.jpg](../iclr_results/2373_RAG-DDR_%20Optimizing%20Retrieval-Augmented%20Generation%20Using%20Differentiable%20Data%20Rewards/tables/ec933a04e639d5683462d15cd6d7eb5fb388c463fff413640916928d22bba1dc.jpg)

![fdc671a05f819eec80545de8d4ad3661b63479fcd87158daf0af49b33999afd5.jpg](../iclr_results/2373_RAG-DDR_%20Optimizing%20Retrieval-Augmented%20Generation%20Using%20Differentiable%20Data%20Rewards/tables/fdc671a05f819eec80545de8d4ad3661b63479fcd87158daf0af49b33999afd5.jpg)

## Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing


### Images

![07582110c475d10f3f4b878f89fb228d6c29d7c1bca8de87f7965fc57cd6348c.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/images/07582110c475d10f3f4b878f89fb228d6c29d7c1bca8de87f7965fc57cd6348c.jpg)

![12a8a9ba8dbf053080b1f108f07571c9fae455eb1d115d456fc14a338bf3d94d.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/images/12a8a9ba8dbf053080b1f108f07571c9fae455eb1d115d456fc14a338bf3d94d.jpg)

![159f73ec98b0cda0283fa6444a56426f92f364d57536596c5052ed788b79f0a8.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/images/159f73ec98b0cda0283fa6444a56426f92f364d57536596c5052ed788b79f0a8.jpg)

![448592c9bd4e6511cd79094212b4cc6bd1d1180db237852ca50e3356d6aeea85.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/images/448592c9bd4e6511cd79094212b4cc6bd1d1180db237852ca50e3356d6aeea85.jpg)

![48f4b9b883fd945ffdca5e5def86b374c419dcefd88662bf2273242d5faa2674.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/images/48f4b9b883fd945ffdca5e5def86b374c419dcefd88662bf2273242d5faa2674.jpg)

![4967aaf2a45f37d1f5a57744bec9047b4b1941f72f1863c428ae4ebdc333ecd5.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/images/4967aaf2a45f37d1f5a57744bec9047b4b1941f72f1863c428ae4ebdc333ecd5.jpg)

![51bb1bd163820f1605c321296781bdc6ee630f0569877b9cb5b01936516cd3d2.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/images/51bb1bd163820f1605c321296781bdc6ee630f0569877b9cb5b01936516cd3d2.jpg)

![5f2b97a2e5b1b0d1b326bdacefe1aad27acaf465ca8a54421db48712e959e23b.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/images/5f2b97a2e5b1b0d1b326bdacefe1aad27acaf465ca8a54421db48712e959e23b.jpg)

![6fb97a12a99865c9235fd41c83c77a7da7898e6de24991c614a003d466a0ed37.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/images/6fb97a12a99865c9235fd41c83c77a7da7898e6de24991c614a003d466a0ed37.jpg)

![a11d2cb410fed7d3cd3fb059719b234ef5b0a5e1e45cde7150f35f73377cfc0f.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/images/a11d2cb410fed7d3cd3fb059719b234ef5b0a5e1e45cde7150f35f73377cfc0f.jpg)

![afb57441e78c2ea102635a0b91c134b083fe2ccbd3cc0232ceacd4164e224d5d.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/images/afb57441e78c2ea102635a0b91c134b083fe2ccbd3cc0232ceacd4164e224d5d.jpg)

![b7be3faa18a9507125bfae59d7f4bf68b0aa8e751cb11315d4631ddd77d99752.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/images/b7be3faa18a9507125bfae59d7f4bf68b0aa8e751cb11315d4631ddd77d99752.jpg)

![cca40341f1fdf5d099eae1786d9cf186257295607bb2aadd07a86587ab95eb81.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/images/cca40341f1fdf5d099eae1786d9cf186257295607bb2aadd07a86587ab95eb81.jpg)

![d338de5eaeee5e434202d53bfe159ad81fb058a2378c2d562e89645b3c940ac3.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/images/d338de5eaeee5e434202d53bfe159ad81fb058a2378c2d562e89645b3c940ac3.jpg)

![dedd65f325328cedcff1b18e7391a7732fbd7ac08d43cc9309073cca1927cd58.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/images/dedd65f325328cedcff1b18e7391a7732fbd7ac08d43cc9309073cca1927cd58.jpg)

![e0a356484d93a24377933a01f0f63474a1754844a4935bffb6b1f3b4ab80e47b.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/images/e0a356484d93a24377933a01f0f63474a1754844a4935bffb6b1f3b4ab80e47b.jpg)

### Tables

![0263137380cf560213287d1eb699b5ca35f2fdcfe6964b3e5fcb77749c04bfa9.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/tables/0263137380cf560213287d1eb699b5ca35f2fdcfe6964b3e5fcb77749c04bfa9.jpg)

![18dcc9a663eda1b5e3e00327b17117e227bf343e9596374e93cdb6081e1160d2.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/tables/18dcc9a663eda1b5e3e00327b17117e227bf343e9596374e93cdb6081e1160d2.jpg)

![257cccc465e20e8f8fc9cb40f538a9b98f824946305f58e1571dbfab000aa98a.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/tables/257cccc465e20e8f8fc9cb40f538a9b98f824946305f58e1571dbfab000aa98a.jpg)

![2d76bd182cefbcc16a26e9b960512e4dac5b6d2f144f2aab7c87baa2080243a2.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/tables/2d76bd182cefbcc16a26e9b960512e4dac5b6d2f144f2aab7c87baa2080243a2.jpg)

![3dab42862b9a3bb669d7f6b437e5f727a20097915a88b6f18d5254ed26619bcc.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/tables/3dab42862b9a3bb669d7f6b437e5f727a20097915a88b6f18d5254ed26619bcc.jpg)

![4c5f9cf442a949a89d2ee2b61ec894d2f4200b74c641881fe440907d3d5d8f7e.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/tables/4c5f9cf442a949a89d2ee2b61ec894d2f4200b74c641881fe440907d3d5d8f7e.jpg)

![4f6655e1ad968690bb3c696397a651172e106676c2b22478def55bf716717f0a.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/tables/4f6655e1ad968690bb3c696397a651172e106676c2b22478def55bf716717f0a.jpg)

![5909f7b3b7333a440027b52c93f1f076b076585a6b39564c2a63e0297cee1e34.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/tables/5909f7b3b7333a440027b52c93f1f076b076585a6b39564c2a63e0297cee1e34.jpg)

![791379f20969758873d5c71e338b6051e5ba28d245e822478d1ece423f961532.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/tables/791379f20969758873d5c71e338b6051e5ba28d245e822478d1ece423f961532.jpg)

![884c1bd38c4934fe7587428a468a435a73ba493298ef8d024462761d118b1f5a.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/tables/884c1bd38c4934fe7587428a468a435a73ba493298ef8d024462761d118b1f5a.jpg)

![94990bccf2b09f1bc5478482f382274182bc3b73a8d024ba4dc88455be1963f5.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/tables/94990bccf2b09f1bc5478482f382274182bc3b73a8d024ba4dc88455be1963f5.jpg)

![9a574e3716be12fa73682efe4289aba27c4bc6a7a8fc3f8a47a90e3cf0e58129.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/tables/9a574e3716be12fa73682efe4289aba27c4bc6a7a8fc3f8a47a90e3cf0e58129.jpg)

![a12187906560e3f09542060f97778936fc6310bb28e443f724b2b62a3cda1a78.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/tables/a12187906560e3f09542060f97778936fc6310bb28e443f724b2b62a3cda1a78.jpg)

![a183e1189e25fdfa0da6ddda73c625ad92729f716aa731c4eb6499d52fabcd81.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/tables/a183e1189e25fdfa0da6ddda73c625ad92729f716aa731c4eb6499d52fabcd81.jpg)

![a835c8e788b9eb97f4ef1c27d8a5fb72faeaac800b861a0098958b900ff1535f.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/tables/a835c8e788b9eb97f4ef1c27d8a5fb72faeaac800b861a0098958b900ff1535f.jpg)

![ae070588ef90277c3e887b45f2e3fbf51a7a0e68962d37dfeab24d72c86eb289.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/tables/ae070588ef90277c3e887b45f2e3fbf51a7a0e68962d37dfeab24d72c86eb289.jpg)

![b6fa16546ab3b30cd450f3f9c2d138580f1fdcab002110b32f093886bad3fe77.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/tables/b6fa16546ab3b30cd450f3f9c2d138580f1fdcab002110b32f093886bad3fe77.jpg)

![ba1873556d2da36aa3603d8afcbf3cc580b1867af393652193359aca4c5c0482.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/tables/ba1873556d2da36aa3603d8afcbf3cc580b1867af393652193359aca4c5c0482.jpg)

![cac3212b3d82751272a3145e64942a737c6d46a3a63af73ce48acf874363dd05.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/tables/cac3212b3d82751272a3145e64942a737c6d46a3a63af73ce48acf874363dd05.jpg)

![cf31aa57322087e2942e1add21635e72e77b563015e5c799ed93140b8cef5268.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/tables/cf31aa57322087e2942e1add21635e72e77b563015e5c799ed93140b8cef5268.jpg)

![e5d0007a84ce95a98c8075439673a8b32898ad7ad07b97497d01785e21eb51d0.jpg](../iclr_results/2374_Magpie_%20Alignment%20Data%20Synthesis%20from%20Scratch%20by%20Prompting%20Aligned%20LLMs%20with%20Nothing/tables/e5d0007a84ce95a98c8075439673a8b32898ad7ad07b97497d01785e21eb51d0.jpg)

## Gap Preserving Distillation by Building Bidirectional Mappings with A Dynamic Teacher


### Images

![01479b4af8cf89ba5802e2f2b8f0e71a3bfbf2be52f7566b5711f1fc3e463748.jpg](../iclr_results/2375_Gap%20Preserving%20Distillation%20by%20Building%20Bidirectional%20Mappings%20with%20A%20Dynamic%20Teacher/images/01479b4af8cf89ba5802e2f2b8f0e71a3bfbf2be52f7566b5711f1fc3e463748.jpg)

![022169740e95874b60bd1d70bc14f0edefc1f54054816fa1f795331a9846b8ae.jpg](../iclr_results/2375_Gap%20Preserving%20Distillation%20by%20Building%20Bidirectional%20Mappings%20with%20A%20Dynamic%20Teacher/images/022169740e95874b60bd1d70bc14f0edefc1f54054816fa1f795331a9846b8ae.jpg)

![1d40b735c64a70c197e02e8e764fed14cf4bb98b45db4db0ea80df5bdc2b06e5.jpg](../iclr_results/2375_Gap%20Preserving%20Distillation%20by%20Building%20Bidirectional%20Mappings%20with%20A%20Dynamic%20Teacher/images/1d40b735c64a70c197e02e8e764fed14cf4bb98b45db4db0ea80df5bdc2b06e5.jpg)

![2676f7c775830701485952f7d5f2764e6ce4d96b95649800f2b4b18a0884cccc.jpg](../iclr_results/2375_Gap%20Preserving%20Distillation%20by%20Building%20Bidirectional%20Mappings%20with%20A%20Dynamic%20Teacher/images/2676f7c775830701485952f7d5f2764e6ce4d96b95649800f2b4b18a0884cccc.jpg)

![414514cfddba1fd678ded248506a34a5eba8f675881674ada461ad39160f2e6b.jpg](../iclr_results/2375_Gap%20Preserving%20Distillation%20by%20Building%20Bidirectional%20Mappings%20with%20A%20Dynamic%20Teacher/images/414514cfddba1fd678ded248506a34a5eba8f675881674ada461ad39160f2e6b.jpg)

![558a889a28e4fac6c7e5fa42b72d01915bb6175561396440ec382c8e0aca8d1d.jpg](../iclr_results/2375_Gap%20Preserving%20Distillation%20by%20Building%20Bidirectional%20Mappings%20with%20A%20Dynamic%20Teacher/images/558a889a28e4fac6c7e5fa42b72d01915bb6175561396440ec382c8e0aca8d1d.jpg)

![9f12b27bb8b835ae1d3691ecbb43463e868ee37d01982d1181add976dd8c296a.jpg](../iclr_results/2375_Gap%20Preserving%20Distillation%20by%20Building%20Bidirectional%20Mappings%20with%20A%20Dynamic%20Teacher/images/9f12b27bb8b835ae1d3691ecbb43463e868ee37d01982d1181add976dd8c296a.jpg)

![be5dabe9768da09fdda36dcbbfa90da571063cb527c59aaa16402101a307f539.jpg](../iclr_results/2375_Gap%20Preserving%20Distillation%20by%20Building%20Bidirectional%20Mappings%20with%20A%20Dynamic%20Teacher/images/be5dabe9768da09fdda36dcbbfa90da571063cb527c59aaa16402101a307f539.jpg)

### Tables

![2b06cf67fbc3107da6c790a713ef06bf4d1a911100cd6ccf36aff092b2cda857.jpg](../iclr_results/2375_Gap%20Preserving%20Distillation%20by%20Building%20Bidirectional%20Mappings%20with%20A%20Dynamic%20Teacher/tables/2b06cf67fbc3107da6c790a713ef06bf4d1a911100cd6ccf36aff092b2cda857.jpg)

![4f5c8a562adc355b97039d9a612e232a70a40135c15fdebdd892d0fa1affab4d.jpg](../iclr_results/2375_Gap%20Preserving%20Distillation%20by%20Building%20Bidirectional%20Mappings%20with%20A%20Dynamic%20Teacher/tables/4f5c8a562adc355b97039d9a612e232a70a40135c15fdebdd892d0fa1affab4d.jpg)

![54dae72b1dc8efed4c86f4ac35a90962d68322af06a230397a7a5b615958f8dc.jpg](../iclr_results/2375_Gap%20Preserving%20Distillation%20by%20Building%20Bidirectional%20Mappings%20with%20A%20Dynamic%20Teacher/tables/54dae72b1dc8efed4c86f4ac35a90962d68322af06a230397a7a5b615958f8dc.jpg)

![58a53160b7452809be41a31f04c3af8bf2a922d0b487e2321aa31995ac6c9455.jpg](../iclr_results/2375_Gap%20Preserving%20Distillation%20by%20Building%20Bidirectional%20Mappings%20with%20A%20Dynamic%20Teacher/tables/58a53160b7452809be41a31f04c3af8bf2a922d0b487e2321aa31995ac6c9455.jpg)

![5b8d80d9ba098a90d658b6e4b3e47083b3ff927978a176e1e722be465f09d3f6.jpg](../iclr_results/2375_Gap%20Preserving%20Distillation%20by%20Building%20Bidirectional%20Mappings%20with%20A%20Dynamic%20Teacher/tables/5b8d80d9ba098a90d658b6e4b3e47083b3ff927978a176e1e722be465f09d3f6.jpg)

![70b699f636d7dbe73eec9e0bac8ad0ba6aa2d3fc3a1acfe806794b5c63c39d06.jpg](../iclr_results/2375_Gap%20Preserving%20Distillation%20by%20Building%20Bidirectional%20Mappings%20with%20A%20Dynamic%20Teacher/tables/70b699f636d7dbe73eec9e0bac8ad0ba6aa2d3fc3a1acfe806794b5c63c39d06.jpg)

![850c23c5cd0ff11b6a3424b3f5ae6609ca4c4ee69e2f2d055ecbe8b33757484d.jpg](../iclr_results/2375_Gap%20Preserving%20Distillation%20by%20Building%20Bidirectional%20Mappings%20with%20A%20Dynamic%20Teacher/tables/850c23c5cd0ff11b6a3424b3f5ae6609ca4c4ee69e2f2d055ecbe8b33757484d.jpg)

![cbbe189130758ac673e3fc9dd52855c7d670f78922278a5816887e8353a04db7.jpg](../iclr_results/2375_Gap%20Preserving%20Distillation%20by%20Building%20Bidirectional%20Mappings%20with%20A%20Dynamic%20Teacher/tables/cbbe189130758ac673e3fc9dd52855c7d670f78922278a5816887e8353a04db7.jpg)

![de1aee3ad2122e608d27c4808168add72c8c069b3a226ee759afc380eb008b29.jpg](../iclr_results/2375_Gap%20Preserving%20Distillation%20by%20Building%20Bidirectional%20Mappings%20with%20A%20Dynamic%20Teacher/tables/de1aee3ad2122e608d27c4808168add72c8c069b3a226ee759afc380eb008b29.jpg)

![df308d0a25c483fc8b50dfa8f358fac3887b8fb8ab8d40218d9fa9b3fc8850b5.jpg](../iclr_results/2375_Gap%20Preserving%20Distillation%20by%20Building%20Bidirectional%20Mappings%20with%20A%20Dynamic%20Teacher/tables/df308d0a25c483fc8b50dfa8f358fac3887b8fb8ab8d40218d9fa9b3fc8850b5.jpg)

![e9487cfb5c1360bafc92328332718b3eb09a49537b0ad6da7c742c50983035d5.jpg](../iclr_results/2375_Gap%20Preserving%20Distillation%20by%20Building%20Bidirectional%20Mappings%20with%20A%20Dynamic%20Teacher/tables/e9487cfb5c1360bafc92328332718b3eb09a49537b0ad6da7c742c50983035d5.jpg)

![f4fd703b2001838b15f5bfcc6303927336ed72e5806b964ab95159420725d043.jpg](../iclr_results/2375_Gap%20Preserving%20Distillation%20by%20Building%20Bidirectional%20Mappings%20with%20A%20Dynamic%20Teacher/tables/f4fd703b2001838b15f5bfcc6303927336ed72e5806b964ab95159420725d043.jpg)

## SegLLM: Multi-round Reasoning Segmentation with Large Language Models


### Images

![00ac3ba9bd851eeed404daeb32a28cf95dbc73f407e4f9583a92c39f806501c4.jpg](../iclr_results/2376_SegLLM_%20Multi-round%20Reasoning%20Segmentation%20with%20Large%20Language%20Models/images/00ac3ba9bd851eeed404daeb32a28cf95dbc73f407e4f9583a92c39f806501c4.jpg)

![09ebd585d36e8f3e202730c8095ec503589b3a9b18688b95f80d319fd2cdc1de.jpg](../iclr_results/2376_SegLLM_%20Multi-round%20Reasoning%20Segmentation%20with%20Large%20Language%20Models/images/09ebd585d36e8f3e202730c8095ec503589b3a9b18688b95f80d319fd2cdc1de.jpg)

![1c7542f4fbc06da561bc0abfaa38b032e4501e2f517f6df991b8144d2424e19a.jpg](../iclr_results/2376_SegLLM_%20Multi-round%20Reasoning%20Segmentation%20with%20Large%20Language%20Models/images/1c7542f4fbc06da561bc0abfaa38b032e4501e2f517f6df991b8144d2424e19a.jpg)

![37c9f87efb852eaa02ba48f4d34626df23bdf10bfb7f0aef90bac067d49c77ed.jpg](../iclr_results/2376_SegLLM_%20Multi-round%20Reasoning%20Segmentation%20with%20Large%20Language%20Models/images/37c9f87efb852eaa02ba48f4d34626df23bdf10bfb7f0aef90bac067d49c77ed.jpg)

![3f638d661ff8e30a9465b973b162d976d8ed54f6abbc98f1f0bea8049c09228d.jpg](../iclr_results/2376_SegLLM_%20Multi-round%20Reasoning%20Segmentation%20with%20Large%20Language%20Models/images/3f638d661ff8e30a9465b973b162d976d8ed54f6abbc98f1f0bea8049c09228d.jpg)

![4ef2a61c433e8c1d49a7c8fdf0ec5400c55a320d9227a1ae65a68158a9ffd9c1.jpg](../iclr_results/2376_SegLLM_%20Multi-round%20Reasoning%20Segmentation%20with%20Large%20Language%20Models/images/4ef2a61c433e8c1d49a7c8fdf0ec5400c55a320d9227a1ae65a68158a9ffd9c1.jpg)

![89759b5d2ff9f6f1eeb2fd83051c792cc7e261bca6c4baab5ea684033306d080.jpg](../iclr_results/2376_SegLLM_%20Multi-round%20Reasoning%20Segmentation%20with%20Large%20Language%20Models/images/89759b5d2ff9f6f1eeb2fd83051c792cc7e261bca6c4baab5ea684033306d080.jpg)

![a6405fc86e93fbb33c1273240372bc4f566b833ee76c433cac70956f871f3b22.jpg](../iclr_results/2376_SegLLM_%20Multi-round%20Reasoning%20Segmentation%20with%20Large%20Language%20Models/images/a6405fc86e93fbb33c1273240372bc4f566b833ee76c433cac70956f871f3b22.jpg)

![ae107df354762e148aafe6c37019ee0adfb6e15f64ae8951655c2a216861def4.jpg](../iclr_results/2376_SegLLM_%20Multi-round%20Reasoning%20Segmentation%20with%20Large%20Language%20Models/images/ae107df354762e148aafe6c37019ee0adfb6e15f64ae8951655c2a216861def4.jpg)

![f26c0aae60863fea9b0e0e5cefb8cb1e0a91ad231f52b0a4109fc0428378b42d.jpg](../iclr_results/2376_SegLLM_%20Multi-round%20Reasoning%20Segmentation%20with%20Large%20Language%20Models/images/f26c0aae60863fea9b0e0e5cefb8cb1e0a91ad231f52b0a4109fc0428378b42d.jpg)

### Tables

![037ab91b18d527558d65ea9b489e31d660ea39b2b3de9a2efed669cde1a5d069.jpg](../iclr_results/2376_SegLLM_%20Multi-round%20Reasoning%20Segmentation%20with%20Large%20Language%20Models/tables/037ab91b18d527558d65ea9b489e31d660ea39b2b3de9a2efed669cde1a5d069.jpg)

![0ecad50e2355113042c762513ef9555bcdebd73b9b2bfab2fea8180fbe957483.jpg](../iclr_results/2376_SegLLM_%20Multi-round%20Reasoning%20Segmentation%20with%20Large%20Language%20Models/tables/0ecad50e2355113042c762513ef9555bcdebd73b9b2bfab2fea8180fbe957483.jpg)

![15ef0d254c16853393b62504a4bb74764755dc7ab3bd78a7cfc2625d2bb00ee1.jpg](../iclr_results/2376_SegLLM_%20Multi-round%20Reasoning%20Segmentation%20with%20Large%20Language%20Models/tables/15ef0d254c16853393b62504a4bb74764755dc7ab3bd78a7cfc2625d2bb00ee1.jpg)

![30d3e50b82facc34b0cf520e293d410d3c6f586964d84087f114261109887737.jpg](../iclr_results/2376_SegLLM_%20Multi-round%20Reasoning%20Segmentation%20with%20Large%20Language%20Models/tables/30d3e50b82facc34b0cf520e293d410d3c6f586964d84087f114261109887737.jpg)

![947f1afe7552e8ef5bb5931f957d7b3c04cfd4c7f9cc403f25b7fb4cf5b5c2fd.jpg](../iclr_results/2376_SegLLM_%20Multi-round%20Reasoning%20Segmentation%20with%20Large%20Language%20Models/tables/947f1afe7552e8ef5bb5931f957d7b3c04cfd4c7f9cc403f25b7fb4cf5b5c2fd.jpg)

![9a599bfdf6a8ebbdcc6a149164b7b769ac32bc9bec7a63a0a4ad1b361b9b525b.jpg](../iclr_results/2376_SegLLM_%20Multi-round%20Reasoning%20Segmentation%20with%20Large%20Language%20Models/tables/9a599bfdf6a8ebbdcc6a149164b7b769ac32bc9bec7a63a0a4ad1b361b9b525b.jpg)

![cc5c44d36d9cc627620b5967cfc57b896f5c227474aa7aedde6206c227bd037b.jpg](../iclr_results/2376_SegLLM_%20Multi-round%20Reasoning%20Segmentation%20with%20Large%20Language%20Models/tables/cc5c44d36d9cc627620b5967cfc57b896f5c227474aa7aedde6206c227bd037b.jpg)

![d1d6121731fbbea49be66056f2c05ae90e7d416a677d24d59ab2cfb79f3edf41.jpg](../iclr_results/2376_SegLLM_%20Multi-round%20Reasoning%20Segmentation%20with%20Large%20Language%20Models/tables/d1d6121731fbbea49be66056f2c05ae90e7d416a677d24d59ab2cfb79f3edf41.jpg)

![f15b9d9dda28560eee55bcd9ade33cde692683d355c3c20132fce31ed27d97f5.jpg](../iclr_results/2376_SegLLM_%20Multi-round%20Reasoning%20Segmentation%20with%20Large%20Language%20Models/tables/f15b9d9dda28560eee55bcd9ade33cde692683d355c3c20132fce31ed27d97f5.jpg)

## Should VLMs be Pre-trained with Image Data?


### Images

![095509a3e9f7bd3473cd0e4d90caf08cc2313b2c32df02c00b46555b9be6b4a1.jpg](../iclr_results/2377_Should%20VLMs%20be%20Pre-trained%20with%20Image%20Data_/images/095509a3e9f7bd3473cd0e4d90caf08cc2313b2c32df02c00b46555b9be6b4a1.jpg)

![0ca95a31ff530a55271a80a94cca9c1514e269dfc9945c68c4252f2bcecbcb25.jpg](../iclr_results/2377_Should%20VLMs%20be%20Pre-trained%20with%20Image%20Data_/images/0ca95a31ff530a55271a80a94cca9c1514e269dfc9945c68c4252f2bcecbcb25.jpg)

![149e4d2b670ec18206074207b7181c441166057bcd0a291ff107431ee060e46a.jpg](../iclr_results/2377_Should%20VLMs%20be%20Pre-trained%20with%20Image%20Data_/images/149e4d2b670ec18206074207b7181c441166057bcd0a291ff107431ee060e46a.jpg)

![22abf6be5f14fb815f3f9daa966545b65a5682521bd98cd63ff380c47c50eabd.jpg](../iclr_results/2377_Should%20VLMs%20be%20Pre-trained%20with%20Image%20Data_/images/22abf6be5f14fb815f3f9daa966545b65a5682521bd98cd63ff380c47c50eabd.jpg)

![4e31598c03522d6ac1d94502ba76dadf95f809b0cb91e23832091329bdfc1487.jpg](../iclr_results/2377_Should%20VLMs%20be%20Pre-trained%20with%20Image%20Data_/images/4e31598c03522d6ac1d94502ba76dadf95f809b0cb91e23832091329bdfc1487.jpg)

![4fda9cccbe4ae66ee76a9af5d8e642cce8b3fb03a2f93ff93e72f5dfe6a0cfa7.jpg](../iclr_results/2377_Should%20VLMs%20be%20Pre-trained%20with%20Image%20Data_/images/4fda9cccbe4ae66ee76a9af5d8e642cce8b3fb03a2f93ff93e72f5dfe6a0cfa7.jpg)

![5029946093177b86556a79b9e82da20474bf79da8a26bcb6df5387f44fe8aec8.jpg](../iclr_results/2377_Should%20VLMs%20be%20Pre-trained%20with%20Image%20Data_/images/5029946093177b86556a79b9e82da20474bf79da8a26bcb6df5387f44fe8aec8.jpg)

![6d4a097bb8cef91b4e476b3cd22ead16d815714ff15d28e25618a26db47b6c82.jpg](../iclr_results/2377_Should%20VLMs%20be%20Pre-trained%20with%20Image%20Data_/images/6d4a097bb8cef91b4e476b3cd22ead16d815714ff15d28e25618a26db47b6c82.jpg)

![a758ad9a83349a1c5cafa1d562dbb5e9d56c2eacba979c5985d9e81f61cf9d93.jpg](../iclr_results/2377_Should%20VLMs%20be%20Pre-trained%20with%20Image%20Data_/images/a758ad9a83349a1c5cafa1d562dbb5e9d56c2eacba979c5985d9e81f61cf9d93.jpg)

![ce8f186d1169c92480fa8d60fb2813dbeefdb9a02422686da307d3ac640c841b.jpg](../iclr_results/2377_Should%20VLMs%20be%20Pre-trained%20with%20Image%20Data_/images/ce8f186d1169c92480fa8d60fb2813dbeefdb9a02422686da307d3ac640c841b.jpg)

![d33645220315bd941170985c4c48996fba0c62fb23b1b3d4d7efe5e0d2465b76.jpg](../iclr_results/2377_Should%20VLMs%20be%20Pre-trained%20with%20Image%20Data_/images/d33645220315bd941170985c4c48996fba0c62fb23b1b3d4d7efe5e0d2465b76.jpg)

![d78fceed626dabac71e5f5300bfe96a8835c6c354fdcb0e05103f5e32df14de8.jpg](../iclr_results/2377_Should%20VLMs%20be%20Pre-trained%20with%20Image%20Data_/images/d78fceed626dabac71e5f5300bfe96a8835c6c354fdcb0e05103f5e32df14de8.jpg)

![e4b64af69608864eaacc59444fa02fba66bb358b25441ffa3c97b115e52c91bc.jpg](../iclr_results/2377_Should%20VLMs%20be%20Pre-trained%20with%20Image%20Data_/images/e4b64af69608864eaacc59444fa02fba66bb358b25441ffa3c97b115e52c91bc.jpg)

![ec8d3f80c875a7fdb7fbc677596b75aa533e4088821e9fd2e36b2752049a270d.jpg](../iclr_results/2377_Should%20VLMs%20be%20Pre-trained%20with%20Image%20Data_/images/ec8d3f80c875a7fdb7fbc677596b75aa533e4088821e9fd2e36b2752049a270d.jpg)

### Tables

![3c04da17921bdbcdfba3c50574f7ce45abb89f9616a44f3f0bcaa1eef9ef9ebd.jpg](../iclr_results/2377_Should%20VLMs%20be%20Pre-trained%20with%20Image%20Data_/tables/3c04da17921bdbcdfba3c50574f7ce45abb89f9616a44f3f0bcaa1eef9ef9ebd.jpg)

![5a81fed6ff88f73986dd31e604e26f72610bc4995fd4c3889d08ca4708c8f95d.jpg](../iclr_results/2377_Should%20VLMs%20be%20Pre-trained%20with%20Image%20Data_/tables/5a81fed6ff88f73986dd31e604e26f72610bc4995fd4c3889d08ca4708c8f95d.jpg)

![608bc465d823e7370d1fd94591eed1f8c099e2830901b114d2327865f4c37a4a.jpg](../iclr_results/2377_Should%20VLMs%20be%20Pre-trained%20with%20Image%20Data_/tables/608bc465d823e7370d1fd94591eed1f8c099e2830901b114d2327865f4c37a4a.jpg)

![87befa119cbf83758982c7282cbdd60b524e5e3f4a48f3181f66d8ffac84959c.jpg](../iclr_results/2377_Should%20VLMs%20be%20Pre-trained%20with%20Image%20Data_/tables/87befa119cbf83758982c7282cbdd60b524e5e3f4a48f3181f66d8ffac84959c.jpg)

![9d013d66298d833feafb70ad650123d49325459880e99d69f6c45ee852d74f4c.jpg](../iclr_results/2377_Should%20VLMs%20be%20Pre-trained%20with%20Image%20Data_/tables/9d013d66298d833feafb70ad650123d49325459880e99d69f6c45ee852d74f4c.jpg)

![b5bb66f79914f999b07a6df6d509dca767b73ccd27513287f599c1927c2ebda9.jpg](../iclr_results/2377_Should%20VLMs%20be%20Pre-trained%20with%20Image%20Data_/tables/b5bb66f79914f999b07a6df6d509dca767b73ccd27513287f599c1927c2ebda9.jpg)

![e06a0859d7b46451302613713752634439b742fb4dc3365d588b057a43915d2b.jpg](../iclr_results/2377_Should%20VLMs%20be%20Pre-trained%20with%20Image%20Data_/tables/e06a0859d7b46451302613713752634439b742fb4dc3365d588b057a43915d2b.jpg)

![f91776f2355d27addb77a9fa490bc80254809dbf40cd92b15e59fe0d6bc4c381.jpg](../iclr_results/2377_Should%20VLMs%20be%20Pre-trained%20with%20Image%20Data_/tables/f91776f2355d27addb77a9fa490bc80254809dbf40cd92b15e59fe0d6bc4c381.jpg)

![fb395b44636fbcb83d05d4aeb3e967bda7a784c303ef605f9b6eee75e3e82792.jpg](../iclr_results/2377_Should%20VLMs%20be%20Pre-trained%20with%20Image%20Data_/tables/fb395b44636fbcb83d05d4aeb3e967bda7a784c303ef605f9b6eee75e3e82792.jpg)

## InfoGS: Efficient Structure-Aware 3D Gaussians via Lightweight Information Shaping


### Images

![04cb82cee5794798e33f401083ed33d2a9f3f728f5859cd6eb0780b04bb24536.jpg](../iclr_results/2378_InfoGS_%20Efficient%20Structure-Aware%203D%20Gaussians%20via%20Lightweight%20Information%20Shaping/images/04cb82cee5794798e33f401083ed33d2a9f3f728f5859cd6eb0780b04bb24536.jpg)

![1eb6959e1dbc67d82d5444bc3933cd934ce68e564c32ff9b3fefb3094d2d0092.jpg](../iclr_results/2378_InfoGS_%20Efficient%20Structure-Aware%203D%20Gaussians%20via%20Lightweight%20Information%20Shaping/images/1eb6959e1dbc67d82d5444bc3933cd934ce68e564c32ff9b3fefb3094d2d0092.jpg)

![22cbfc35948d10114017c7dabb82944e485685d966cfd9bab2782cc3cf838dfe.jpg](../iclr_results/2378_InfoGS_%20Efficient%20Structure-Aware%203D%20Gaussians%20via%20Lightweight%20Information%20Shaping/images/22cbfc35948d10114017c7dabb82944e485685d966cfd9bab2782cc3cf838dfe.jpg)

![369cdaeeb269a740c7c42462604a8f100e926b5cf0d10f26b9dde249566720c2.jpg](../iclr_results/2378_InfoGS_%20Efficient%20Structure-Aware%203D%20Gaussians%20via%20Lightweight%20Information%20Shaping/images/369cdaeeb269a740c7c42462604a8f100e926b5cf0d10f26b9dde249566720c2.jpg)

![425c6b71819ecd0b35d0e3cd4f03eb3059f486705f50408bb5aa0fa3b5507004.jpg](../iclr_results/2378_InfoGS_%20Efficient%20Structure-Aware%203D%20Gaussians%20via%20Lightweight%20Information%20Shaping/images/425c6b71819ecd0b35d0e3cd4f03eb3059f486705f50408bb5aa0fa3b5507004.jpg)

![74368fd3e771e8314e7652378cb92a6244f5bff310263b7820b4439ef82902e4.jpg](../iclr_results/2378_InfoGS_%20Efficient%20Structure-Aware%203D%20Gaussians%20via%20Lightweight%20Information%20Shaping/images/74368fd3e771e8314e7652378cb92a6244f5bff310263b7820b4439ef82902e4.jpg)

![7546eae897462be2340ab71af113fee9e443ce93f849622e797d5ec15c1e4a41.jpg](../iclr_results/2378_InfoGS_%20Efficient%20Structure-Aware%203D%20Gaussians%20via%20Lightweight%20Information%20Shaping/images/7546eae897462be2340ab71af113fee9e443ce93f849622e797d5ec15c1e4a41.jpg)

![9b0af6ccb07d47e5c6630292fd659ee911da7b45be4656ca551c55592abb0785.jpg](../iclr_results/2378_InfoGS_%20Efficient%20Structure-Aware%203D%20Gaussians%20via%20Lightweight%20Information%20Shaping/images/9b0af6ccb07d47e5c6630292fd659ee911da7b45be4656ca551c55592abb0785.jpg)

![b43eea50d90f6dbc366609aae950e72b6ae98d47093fc562f084b5863ce3b73a.jpg](../iclr_results/2378_InfoGS_%20Efficient%20Structure-Aware%203D%20Gaussians%20via%20Lightweight%20Information%20Shaping/images/b43eea50d90f6dbc366609aae950e72b6ae98d47093fc562f084b5863ce3b73a.jpg)

![be2a04d277292b9893ba483339065744274b8785fd950bd8a038838c5ed6c370.jpg](../iclr_results/2378_InfoGS_%20Efficient%20Structure-Aware%203D%20Gaussians%20via%20Lightweight%20Information%20Shaping/images/be2a04d277292b9893ba483339065744274b8785fd950bd8a038838c5ed6c370.jpg)

![c200cb0facb3f8a19787460c71de4ddedb1ea7ffe7db4d2cb3ea1747e4379ef5.jpg](../iclr_results/2378_InfoGS_%20Efficient%20Structure-Aware%203D%20Gaussians%20via%20Lightweight%20Information%20Shaping/images/c200cb0facb3f8a19787460c71de4ddedb1ea7ffe7db4d2cb3ea1747e4379ef5.jpg)

![c491d4326d82d5e2326c0a79bb0eee58b16e46d7380b0334a0ccd6ad7b5283d3.jpg](../iclr_results/2378_InfoGS_%20Efficient%20Structure-Aware%203D%20Gaussians%20via%20Lightweight%20Information%20Shaping/images/c491d4326d82d5e2326c0a79bb0eee58b16e46d7380b0334a0ccd6ad7b5283d3.jpg)

![e7d7016c4b6a3ccc16030e0b37d3f94f3767b9986dac2c7eed23d289a5e21e70.jpg](../iclr_results/2378_InfoGS_%20Efficient%20Structure-Aware%203D%20Gaussians%20via%20Lightweight%20Information%20Shaping/images/e7d7016c4b6a3ccc16030e0b37d3f94f3767b9986dac2c7eed23d289a5e21e70.jpg)

![e80bc55cc2d1c65d582ccdaa86f9bc173dc2488f6a7038aa764824d5d6d22ea6.jpg](../iclr_results/2378_InfoGS_%20Efficient%20Structure-Aware%203D%20Gaussians%20via%20Lightweight%20Information%20Shaping/images/e80bc55cc2d1c65d582ccdaa86f9bc173dc2488f6a7038aa764824d5d6d22ea6.jpg)

![fe25bc75e71248fc6c546da811c8209483bee2735d08d79b03714665e1fa34a2.jpg](../iclr_results/2378_InfoGS_%20Efficient%20Structure-Aware%203D%20Gaussians%20via%20Lightweight%20Information%20Shaping/images/fe25bc75e71248fc6c546da811c8209483bee2735d08d79b03714665e1fa34a2.jpg)

### Tables

![2b4a0c4fc2e31113da3e76c61ede0a554da655f31ad7ec3edd5522172a077a53.jpg](../iclr_results/2378_InfoGS_%20Efficient%20Structure-Aware%203D%20Gaussians%20via%20Lightweight%20Information%20Shaping/tables/2b4a0c4fc2e31113da3e76c61ede0a554da655f31ad7ec3edd5522172a077a53.jpg)

![bf07fb8afc9ffdf9fbd607fbc2be34c396c2730eb2e0e1b79baf9516cb4e3ded.jpg](../iclr_results/2378_InfoGS_%20Efficient%20Structure-Aware%203D%20Gaussians%20via%20Lightweight%20Information%20Shaping/tables/bf07fb8afc9ffdf9fbd607fbc2be34c396c2730eb2e0e1b79baf9516cb4e3ded.jpg)

## Transformers Can Learn Temporal Difference Methods for In-Context Reinforcement Learning


### Images

![04bef6acf7959d70fd92c7d9173f4a1030f136bce601fcaa8403d2e83d9bca9d.jpg](../iclr_results/2379_Transformers%20Can%20Learn%20Temporal%20Difference%20Methods%20for%20In-Context%20Reinforcement%20Learning/images/04bef6acf7959d70fd92c7d9173f4a1030f136bce601fcaa8403d2e83d9bca9d.jpg)

![0f4614856ee05979dd3b6f07a8e227cd83db91cd1ba9a23fe3ad0d9b71231de1.jpg](../iclr_results/2379_Transformers%20Can%20Learn%20Temporal%20Difference%20Methods%20for%20In-Context%20Reinforcement%20Learning/images/0f4614856ee05979dd3b6f07a8e227cd83db91cd1ba9a23fe3ad0d9b71231de1.jpg)

![1835bcfd1c3b333853284d1622aacea398673fc6d9f7d52bc11a86b586616a0d.jpg](../iclr_results/2379_Transformers%20Can%20Learn%20Temporal%20Difference%20Methods%20for%20In-Context%20Reinforcement%20Learning/images/1835bcfd1c3b333853284d1622aacea398673fc6d9f7d52bc11a86b586616a0d.jpg)

![19ed11ef2bfe7808d552df5feb9a56c69b28db2d6652c5923b0c2f0dc125e78e.jpg](../iclr_results/2379_Transformers%20Can%20Learn%20Temporal%20Difference%20Methods%20for%20In-Context%20Reinforcement%20Learning/images/19ed11ef2bfe7808d552df5feb9a56c69b28db2d6652c5923b0c2f0dc125e78e.jpg)

![1e0d17433b39de59ab7b655b0f44218b25f04b1723e44beb6d11c3b2be280230.jpg](../iclr_results/2379_Transformers%20Can%20Learn%20Temporal%20Difference%20Methods%20for%20In-Context%20Reinforcement%20Learning/images/1e0d17433b39de59ab7b655b0f44218b25f04b1723e44beb6d11c3b2be280230.jpg)

![20458a1100f03b532c189c3c1706bbbc288f13dd2b9705cb4022f69e5ff21b0f.jpg](../iclr_results/2379_Transformers%20Can%20Learn%20Temporal%20Difference%20Methods%20for%20In-Context%20Reinforcement%20Learning/images/20458a1100f03b532c189c3c1706bbbc288f13dd2b9705cb4022f69e5ff21b0f.jpg)

![41f6cc8a074afb0b861db87bc26b69a3956a6263ee859ef3f7faab67e4d8f123.jpg](../iclr_results/2379_Transformers%20Can%20Learn%20Temporal%20Difference%20Methods%20for%20In-Context%20Reinforcement%20Learning/images/41f6cc8a074afb0b861db87bc26b69a3956a6263ee859ef3f7faab67e4d8f123.jpg)

![60e037c1102fff037e6c818dad1c0d15a85cde568a80a416540c342984e094a0.jpg](../iclr_results/2379_Transformers%20Can%20Learn%20Temporal%20Difference%20Methods%20for%20In-Context%20Reinforcement%20Learning/images/60e037c1102fff037e6c818dad1c0d15a85cde568a80a416540c342984e094a0.jpg)

![8a6a68f6344973fad254072cb570cfe95a1ec4aa514d9e67e4528842aa9f187f.jpg](../iclr_results/2379_Transformers%20Can%20Learn%20Temporal%20Difference%20Methods%20for%20In-Context%20Reinforcement%20Learning/images/8a6a68f6344973fad254072cb570cfe95a1ec4aa514d9e67e4528842aa9f187f.jpg)

![ab8915f4eab53c73e13b3f06778f8137e8c4acd5185af033f4dfbb1bab24da42.jpg](../iclr_results/2379_Transformers%20Can%20Learn%20Temporal%20Difference%20Methods%20for%20In-Context%20Reinforcement%20Learning/images/ab8915f4eab53c73e13b3f06778f8137e8c4acd5185af033f4dfbb1bab24da42.jpg)

![ba3ef8d51059b9f1a21315a54d75fb19d9be104367eaac9bb65f5ea29259dd56.jpg](../iclr_results/2379_Transformers%20Can%20Learn%20Temporal%20Difference%20Methods%20for%20In-Context%20Reinforcement%20Learning/images/ba3ef8d51059b9f1a21315a54d75fb19d9be104367eaac9bb65f5ea29259dd56.jpg)

![cf22c09bcf6b2bf2b2b32b605993013c2e2f2a35be5204abf952ec16b199a988.jpg](../iclr_results/2379_Transformers%20Can%20Learn%20Temporal%20Difference%20Methods%20for%20In-Context%20Reinforcement%20Learning/images/cf22c09bcf6b2bf2b2b32b605993013c2e2f2a35be5204abf952ec16b199a988.jpg)

![eb7145d67f28d09fa1d4df3877e1b5b47219e034eddc5dd17c0e42ce0a986671.jpg](../iclr_results/2379_Transformers%20Can%20Learn%20Temporal%20Difference%20Methods%20for%20In-Context%20Reinforcement%20Learning/images/eb7145d67f28d09fa1d4df3877e1b5b47219e034eddc5dd17c0e42ce0a986671.jpg)

### Tables

![21a0638d4693060c9bc2327f50d5c213c42c17855a37ac388bd1120cf644adde.jpg](../iclr_results/2379_Transformers%20Can%20Learn%20Temporal%20Difference%20Methods%20for%20In-Context%20Reinforcement%20Learning/tables/21a0638d4693060c9bc2327f50d5c213c42c17855a37ac388bd1120cf644adde.jpg)

![2ff214df6e35258d2a69cd5fcae1f37f8c61d27423744ad75d5118f8f07257bd.jpg](../iclr_results/2379_Transformers%20Can%20Learn%20Temporal%20Difference%20Methods%20for%20In-Context%20Reinforcement%20Learning/tables/2ff214df6e35258d2a69cd5fcae1f37f8c61d27423744ad75d5118f8f07257bd.jpg)

## CL-DiffPhyCon: Closed-loop Diffusion Control of Complex Physical Systems

### Images

![1369fe7a5e74a2ccacfcd93e20e4a5470f1201d06373c810b92818f00cdddf5b.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/1369fe7a5e74a2ccacfcd93e20e4a5470f1201d06373c810b92818f00cdddf5b.jpg)

![38ea515101c3ac195ca3aa55b675f3681f2ec7688cc8ee2bd85d2f55974bc4f9.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/38ea515101c3ac195ca3aa55b675f3681f2ec7688cc8ee2bd85d2f55974bc4f9.jpg)

![4c04e832efc575c2e130a0459dcbf82216efb4a2a9a3b457ccdff8ec5a3c399f.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/4c04e832efc575c2e130a0459dcbf82216efb4a2a9a3b457ccdff8ec5a3c399f.jpg)

![5950557a0675af3e454842a00778c6a4f987e57b246db4d2842d374eff48283e.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/5950557a0675af3e454842a00778c6a4f987e57b246db4d2842d374eff48283e.jpg)

![662e5bbd9242bfdedd37e4834267ac6a55ab660e8e63af76850e407b0a61edaf.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/662e5bbd9242bfdedd37e4834267ac6a55ab660e8e63af76850e407b0a61edaf.jpg)

![695ce17a9a9c2bf98cef5a0f21654c0ba25c809749aba4d56ac0ed09c3787c86.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/695ce17a9a9c2bf98cef5a0f21654c0ba25c809749aba4d56ac0ed09c3787c86.jpg)

![8f78cc46e597d73f8b771918f3785e636842a3c12ba7660548d1325f6abc418f.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/8f78cc46e597d73f8b771918f3785e636842a3c12ba7660548d1325f6abc418f.jpg)

![a429e093837b16a91f5a731485ea9d566322e069c101f78bf71d709fdc61e656.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/a429e093837b16a91f5a731485ea9d566322e069c101f78bf71d709fdc61e656.jpg)

![a4dc3a022885894ad76740c959f996ad3291998ef7561e35a405803b3c704762.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/a4dc3a022885894ad76740c959f996ad3291998ef7561e35a405803b3c704762.jpg)

![ad432bbdb36205ded4dc6075a1f6881cf613e9b3ecc91f6912c91ac465fb1443.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/ad432bbdb36205ded4dc6075a1f6881cf613e9b3ecc91f6912c91ac465fb1443.jpg)

![af6d86672ccde700952b3d6316a50571e236553a5558fd527251ab6d55b43368.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/af6d86672ccde700952b3d6316a50571e236553a5558fd527251ab6d55b43368.jpg)

![c635d0f89cddae3596a38cf1c1f757a99605c5736e46b5df7a591ade117bb816.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/c635d0f89cddae3596a38cf1c1f757a99605c5736e46b5df7a591ade117bb816.jpg)

![cbc865e4166b3e09fc7995c8f830182b35389d61d42d746b0230a27f925a419e.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/cbc865e4166b3e09fc7995c8f830182b35389d61d42d746b0230a27f925a419e.jpg)

![d09a79a1edc85715b02fc1b5ad9697b535ad2f7e99db9c94af3e503f58b3b6c4.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/d09a79a1edc85715b02fc1b5ad9697b535ad2f7e99db9c94af3e503f58b3b6c4.jpg)

![e080a85438ff553f877ca789a4f048bcff3e5f7ed8056a9fb8b748cbbc31d160.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/e080a85438ff553f877ca789a4f048bcff3e5f7ed8056a9fb8b748cbbc31d160.jpg)

![f29f317e81f2ad5f88690144c59a36def4b1abf7603bc73e433b752b69681d60.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/images/f29f317e81f2ad5f88690144c59a36def4b1abf7603bc73e433b752b69681d60.jpg)

### Tables

![17c7511bd75f206749e03d431145598d1772ef5d84a426c973f0875eaf679a53.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/tables/17c7511bd75f206749e03d431145598d1772ef5d84a426c973f0875eaf679a53.jpg)

![5ccde043f6240d4b443bfcef6b525f7b05c3756ca542b3170118b57834b8ef99.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/tables/5ccde043f6240d4b443bfcef6b525f7b05c3756ca542b3170118b57834b8ef99.jpg)

![9604c9ff37e127239fd17823e2e110887b358a3de5c547d3a90daecc87a8a10a.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/tables/9604c9ff37e127239fd17823e2e110887b358a3de5c547d3a90daecc87a8a10a.jpg)

![c32cc55bb4abde3e217f02064af8d796df8183f3a943c6cf5e661712df9d1fba.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/tables/c32cc55bb4abde3e217f02064af8d796df8183f3a943c6cf5e661712df9d1fba.jpg)

![c986e326bcde2ace7d97f1471509551d2778f803120c9e55263e998dec566dc7.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/tables/c986e326bcde2ace7d97f1471509551d2778f803120c9e55263e998dec566dc7.jpg)

![dc3e826a7fb91dd5cceeb73f2fa94fa3efde2df5e4e51a4614ca3ee6f16d9430.jpg](../iclr_results/2380_CL-DiffPhyCon_%20Closed-loop%20Diffusion%20Control%20of%20Complex%20Physical%20Systems/tables/dc3e826a7fb91dd5cceeb73f2fa94fa3efde2df5e4e51a4614ca3ee6f16d9430.jpg)
