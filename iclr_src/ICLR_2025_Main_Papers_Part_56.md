# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 56 of 100

## 目录 (Table of Contents)

1. [A Theoretical Perspective: How to Prevent Model Collapse in Self-consuming Training Loops](#A-Theoretical-Perspective-How-to-Prevent-Model-Collapse-in-Self-consuming-Training-Loops)
2. [MMIU: Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models](#MMIU-Multimodal-Multi-image-Understanding-for-Evaluating-Large-Vision-Language-Models)
3. [Contextual Document Embeddings](#Contextual-Document-Embeddings)
4. [Anyprefer: An Agentic Framework for Preference Data Synthesis](#Anyprefer-An-Agentic-Framework-for-Preference-Data-Synthesis)
5. [Small Models are LLM Knowledge Triggers for Medical Tabular Prediction](#Small-Models-are-LLM-Knowledge-Triggers-for-Medical-Tabular-Prediction)
6. [GenDataAgent: On-the-fly Dataset Augmentation with Synthetic Data](#GenDataAgent-On-the-fly-Dataset-Augmentation-with-Synthetic-Data)
7. [GMValuator: Similarity-based Data Valuation for Generative Models](#GMValuator-Similarity-based-Data-Valuation-for-Generative-Models)
8. [Optimal Learning of Kernel Logistic Regression for Complex Classification Scenarios](#Optimal-Learning-of-Kernel-Logistic-Regression-for-Complex-Classification-Scenarios)
9. [Building Math Agents with Multi-Turn Iterative Preference Learning](#Building-Math-Agents-with-Multi-Turn-Iterative-Preference-Learning)
10. [DyCAST: Learning Dynamic Causal Structure from Time Series](#DyCAST-Learning-Dynamic-Causal-Structure-from-Time-Series)
11. [Three Mechanisms of Feature Learning in a Linear Network](#Three-Mechanisms-of-Feature-Learning-in-a-Linear-Network)
12. [Graph Neural Networks Gone Hogwild](#Graph-Neural-Networks-Gone-Hogwild)
13. [Decoding Game: On Minimax Optimality of Heuristic Text Generation Strategies](#Decoding-Game-On-Minimax-Optimality-of-Heuristic-Text-Generation-Strategies)
14. [I-Con: A Unifying Framework for Representation Learning](#I-Con-A-Unifying-Framework-for-Representation-Learning)
15. [On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback](#On-Targeted-Manipulation-and-Deception-when-Optimizing-LLMs-for-User-Feedback)
16. [A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation](#A-Large-scale-Dataset-and-Benchmark-for-Commuting-Origin-Destination-Flow-Generation)
17. [Stable Hadamard Memory: Revitalizing Memory-Augmented Agents for Reinforcement Learning](#Stable-Hadamard-Memory-Revitalizing-Memory-Augmented-Agents-for-Reinforcement-Learning)
18. [Breaking Free from MMI: A New Frontier in Rationalization by Probing Input Utilization](#Breaking-Free-from-MMI-A-New-Frontier-in-Rationalization-by-Probing-Input-Utilization)
19. [Scaling Optimal LR Across Token Horizons](#Scaling-Optimal-LR-Across-Token-Horizons)
20. [MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions](#MA-RLHF-Reinforcement-Learning-from-Human-Feedback-with-Macro-Actions)
21. [Towards Synergistic Path-based Explanations for Knowledge Graph Completion: Exploration and Evaluation](#Towards-Synergistic-Path-based-Explanations-for-Knowledge-Graph-Completion-Exploration-and-Evaluation)
22. [Transition Path Sampling with Improved Off-Policy Training of Diffusion Path Samplers](#Transition-Path-Sampling-with-Improved-Off-Policy-Training-of-Diffusion-Path-Samplers)
23. [Discovering Influential Neuron Path in Vision Transformers](#Discovering-Influential-Neuron-Path-in-Vision-Transformers)
24. [Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model](#Re-Evaluating-the-Impact-of-Unseen-Class-Unlabeled-Data-on-Semi-Supervised-Learning-Model)
25. [Probe Pruning: Accelerating LLMs through Dynamic Pruning via Model-Probing](#Probe-Pruning-Accelerating-LLMs-through-Dynamic-Pruning-via-Model-Probing)
26. [Scaling up Masked Diffusion Models on Text](#Scaling-up-Masked-Diffusion-Models-on-Text)
27. [REBIND: Enhancing Ground-state Molecular Conformation Prediction via Force-Based Graph Rewiring](#REBIND-Enhancing-Ground-state-Molecular-Conformation-Prediction-via-Force-Based-Graph-Rewiring)
28. [Q-Adapter: Customizing Pre-trained LLMs to New Preferences with Forgetting Mitigation](#Q-Adapter-Customizing-Pre-trained-LLMs-to-New-Preferences-with-Forgetting-Mitigation)
29. [Training Robust Ensembles Requires Rethinking Lipschitz Continuity](#Training-Robust-Ensembles-Requires-Rethinking-Lipschitz-Continuity)
30. [Does Spatial Cognition Emerge in Frontier Models?](#Does-Spatial-Cognition-Emerge-in-Frontier-Models)
31. [Learning Multi-Index Models with Neural Networks via Mean-Field Langevin Dynamics](#Learning-Multi-Index-Models-with-Neural-Networks-via-Mean-Field-Langevin-Dynamics)
32. [Rotated Runtime Smooth: Training-Free Activation Smoother for accurate INT4 inference](#Rotated-Runtime-Smooth-Training-Free-Activation-Smoother-for-accurate-INT4-inference)
33. [Multi-Task Corrupted Prediction for Learning Robust Audio-Visual Speech Representation](#Multi-Task-Corrupted-Prediction-for-Learning-Robust-Audio-Visual-Speech-Representation)
34. [Distributional Associations vs In-Context Reasoning: A Study of Feed-forward and Attention Layers](#Distributional-Associations-vs-In-Context-Reasoning-A-Study-of-Feed-forward-and-Attention-Layers)
35. [Salvage: Shapley-distribution Approximation Learning Via Attribution Guided Exploration for Explainable Image Classification](#Salvage-Shapley-distribution-Approximation-Learning-Via-Attribution-Guided-Exploration-for-Explainable-Image-Classification)
36. [PivotMesh: Generic 3D Mesh Generation via Pivot Vertices Guidance](#PivotMesh-Generic-3D-Mesh-Generation-via-Pivot-Vertices-Guidance)
37. [Adaptive Pruning of Pretrained Transformer via Differential Inclusions](#Adaptive-Pruning-of-Pretrained-Transformer-via-Differential-Inclusions)

---


## A Theoretical Perspective: How to Prevent Model Collapse in Self-consuming Training Loops

### Images

![fea9ff357d2efb0dfa31313e8e8a84b09260b775ec9d6ffa64bc06541b9f6330.jpg](../iclr_results/2046_Optimal Non-Asymptotic Rates of Value Iteration for Average-Reward Markov Decision Processes/images/fea9ff357d2efb0dfa31313e8e8a84b09260b775ec9d6ffa64bc06541b9f6330.jpg)

### Tables

![019da2b821fbb46ce12b7e8de73a685fb04b3bba959590c20944b53190c463d5.jpg](../iclr_results/2046_Optimal Non-Asymptotic Rates of Value Iteration for Average-Reward Markov Decision Processes/tables/019da2b821fbb46ce12b7e8de73a685fb04b3bba959590c20944b53190c463d5.jpg)

![721bb270b16529f6f374bb326556b1f18b1022d4fee193435570501aa99b5ef3.jpg](../iclr_results/2046_Optimal Non-Asymptotic Rates of Value Iteration for Average-Reward Markov Decision Processes/tables/721bb270b16529f6f374bb326556b1f18b1022d4fee193435570501aa99b5ef3.jpg)

![7d9e99649b9badef36bf1563b1db2b38b533032ee1e297cac1ff76ea67661e84.jpg](../iclr_results/2046_Optimal Non-Asymptotic Rates of Value Iteration for Average-Reward Markov Decision Processes/tables/7d9e99649b9badef36bf1563b1db2b38b533032ee1e297cac1ff76ea67661e84.jpg)

![8afb5549a7cbf32246d270a6d3499dcc0bac169e7a2ee63b3269d735eee54f91.jpg](../iclr_results/2046_Optimal Non-Asymptotic Rates of Value Iteration for Average-Reward Markov Decision Processes/tables/8afb5549a7cbf32246d270a6d3499dcc0bac169e7a2ee63b3269d735eee54f91.jpg)

![bc0faa1a3edcd0d8a6e6269de6fc84a8a1b5ba8f36f8d14b131db901f4398152.jpg](../iclr_results/2046_Optimal Non-Asymptotic Rates of Value Iteration for Average-Reward Markov Decision Processes/tables/bc0faa1a3edcd0d8a6e6269de6fc84a8a1b5ba8f36f8d14b131db901f4398152.jpg)

![d762632996defd4605f34092d9e8158a7505df93f6ffa05ae405c8fb5e581974.jpg](../iclr_results/2046_Optimal Non-Asymptotic Rates of Value Iteration for Average-Reward Markov Decision Processes/tables/d762632996defd4605f34092d9e8158a7505df93f6ffa05ae405c8fb5e581974.jpg)

![e5e7a20c631b18a0b8805fd8a39abd9f823048b947150ab47610897f0beb4504.jpg](../iclr_results/2046_Optimal Non-Asymptotic Rates of Value Iteration for Average-Reward Markov Decision Processes/tables/e5e7a20c631b18a0b8805fd8a39abd9f823048b947150ab47610897f0beb4504.jpg)

![e82d6bae256884968dc01f5dd0eb7cd2e62437f3d550ffb39fb92d0047596023.jpg](../iclr_results/2046_Optimal Non-Asymptotic Rates of Value Iteration for Average-Reward Markov Decision Processes/tables/e82d6bae256884968dc01f5dd0eb7cd2e62437f3d550ffb39fb92d0047596023.jpg)

## A Theoretical Perspective: How to Prevent Model Collapse in Self-consuming Training Loops


### Images

![3f0b05b4e0397f47632f92d8c84a503341561c63609ae4a5ad47fdfd9f101360.jpg](../iclr_results/2047_A Theoretical Perspective_ How to Prevent Model Collapse in Self-consuming Training Loops/images/3f0b05b4e0397f47632f92d8c84a503341561c63609ae4a5ad47fdfd9f101360.jpg)

### Tables

![04af8409065060c4dbf3c04c222301aa2941347a6b145b19a35284d61afac1b2.jpg](../iclr_results/2047_A Theoretical Perspective_ How to Prevent Model Collapse in Self-consuming Training Loops/tables/04af8409065060c4dbf3c04c222301aa2941347a6b145b19a35284d61afac1b2.jpg)

## MMIU: Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models


### Images

![0286e082ac59e043b840aeac48a73a5dc3dc92118eb94a0c918e5a768a7326df.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/images/0286e082ac59e043b840aeac48a73a5dc3dc92118eb94a0c918e5a768a7326df.jpg)

![038c07f2ceb4a0c5fa5023bbc0497a14d0da734691e9bd7db2a4212b691eaa3d.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/images/038c07f2ceb4a0c5fa5023bbc0497a14d0da734691e9bd7db2a4212b691eaa3d.jpg)

![1121867d621ecd4d419e6595c3231715f11d3f861fb936b5296ca611cc7b8153.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/images/1121867d621ecd4d419e6595c3231715f11d3f861fb936b5296ca611cc7b8153.jpg)

![32cf151ca81a7dd900fe3bc36b6f14f35e242c68b8e5b6edbcc8d7b2c0be48f1.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/images/32cf151ca81a7dd900fe3bc36b6f14f35e242c68b8e5b6edbcc8d7b2c0be48f1.jpg)

![35f46dfee190e60c7fc72143536b3986e2999cafed8b79c4a8d95e20d8ffac40.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/images/35f46dfee190e60c7fc72143536b3986e2999cafed8b79c4a8d95e20d8ffac40.jpg)

![4220fdd8bf989f6cf7486cf7a5cefad4c5bc1f5e492d6ed422ea899588dbc196.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/images/4220fdd8bf989f6cf7486cf7a5cefad4c5bc1f5e492d6ed422ea899588dbc196.jpg)

![4fbca523d1be9924c0de094af09fb0f513e48035d449beca0308cb3479a3ee41.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/images/4fbca523d1be9924c0de094af09fb0f513e48035d449beca0308cb3479a3ee41.jpg)

![5e4a8ce1c41189d8241544620a3a9e4dbcd31df4b07f3331c7273794493d05cf.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/images/5e4a8ce1c41189d8241544620a3a9e4dbcd31df4b07f3331c7273794493d05cf.jpg)

![684f19356f81e7c0a0d9fea4b4e6e5a47a9937e5c83de6303b8e11471efd2b0c.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/images/684f19356f81e7c0a0d9fea4b4e6e5a47a9937e5c83de6303b8e11471efd2b0c.jpg)

![830891827941403b5c8426e2187e9020d43e36550c29cc05e4037a25ede72bb0.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/images/830891827941403b5c8426e2187e9020d43e36550c29cc05e4037a25ede72bb0.jpg)

![8c5a5722890678bae0cf3a19b3326d3f354a2db9e6de40a45ac534795330c1aa.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/images/8c5a5722890678bae0cf3a19b3326d3f354a2db9e6de40a45ac534795330c1aa.jpg)

![9be51ce06b45a833298362eccb97ddc80806b032d88378c2d9faf21c97cc867c.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/images/9be51ce06b45a833298362eccb97ddc80806b032d88378c2d9faf21c97cc867c.jpg)

![c69c5123c3254f6d6aa4b97f1fc5e87ffc1806371e3b26d43b8aa9a7aa8375c6.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/images/c69c5123c3254f6d6aa4b97f1fc5e87ffc1806371e3b26d43b8aa9a7aa8375c6.jpg)

![cf467e63f6642a0520d4878e817e84e8dc53451cc40bfd23efff4c432df09f0a.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/images/cf467e63f6642a0520d4878e817e84e8dc53451cc40bfd23efff4c432df09f0a.jpg)

![cfe98dfe3d23dd69380b03478f021a8021d66871012177cd917ad127a9833e5e.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/images/cfe98dfe3d23dd69380b03478f021a8021d66871012177cd917ad127a9833e5e.jpg)

![e0ccb2cd5972c92d4a07a3c6dafdf1d30a97a238867f1f5210b433e647530816.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/images/e0ccb2cd5972c92d4a07a3c6dafdf1d30a97a238867f1f5210b433e647530816.jpg)

![e7849aa01485bf38076214539a7069dac5ed9a4e291ddab46a8bfbf1a790d68c.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/images/e7849aa01485bf38076214539a7069dac5ed9a4e291ddab46a8bfbf1a790d68c.jpg)

![f678630070865a3003ffe34a22051f9de536464c30f16269aa27152915c4941c.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/images/f678630070865a3003ffe34a22051f9de536464c30f16269aa27152915c4941c.jpg)

### Tables

![0286bde5119fc037b51db7c7bf6f35c899ca6d8f61ae06494ff01ee4c7f74269.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/0286bde5119fc037b51db7c7bf6f35c899ca6d8f61ae06494ff01ee4c7f74269.jpg)

![0306b63e284d87a82c444e5433c13dfbbf5e150e30256f18d89b41ee5cbd4972.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/0306b63e284d87a82c444e5433c13dfbbf5e150e30256f18d89b41ee5cbd4972.jpg)

![03fe05851ae067f6b607a6349403e8eaa644e64c5e2841f6d5c9664651782058.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/03fe05851ae067f6b607a6349403e8eaa644e64c5e2841f6d5c9664651782058.jpg)

![0599862f95d1ae197bba42aebd86d998a2224760c75d1903f16ddb298ec6f3d2.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/0599862f95d1ae197bba42aebd86d998a2224760c75d1903f16ddb298ec6f3d2.jpg)

![078db861b7dfbadde2be50ee1efc78ecbf3a423d36a59b3f28c4bab22a15fc4c.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/078db861b7dfbadde2be50ee1efc78ecbf3a423d36a59b3f28c4bab22a15fc4c.jpg)

![15e98cc37b1abc32274fa69e4f6cc1647978a2d0882d50964791a4862bd5be44.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/15e98cc37b1abc32274fa69e4f6cc1647978a2d0882d50964791a4862bd5be44.jpg)

![1fe13c85815931df2ad06640bac09a84c9e16a8f47153ad9dab0246a64a97f06.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/1fe13c85815931df2ad06640bac09a84c9e16a8f47153ad9dab0246a64a97f06.jpg)

![20557d5416b9f2f6d2375512822a176746e264abf4995cd87939f863f3bcf8b2.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/20557d5416b9f2f6d2375512822a176746e264abf4995cd87939f863f3bcf8b2.jpg)

![225b32b67f833fc34043b3e2cc7bd2c96305a8bdc1678ce4b419fa1ed385f2d7.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/225b32b67f833fc34043b3e2cc7bd2c96305a8bdc1678ce4b419fa1ed385f2d7.jpg)

![23b877351c21a63b72c63b9fed96a93de945054390b2ebe7568e4aefd1fdaba6.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/23b877351c21a63b72c63b9fed96a93de945054390b2ebe7568e4aefd1fdaba6.jpg)

![44d8f2c3fc651b2b65e3b57ef54bc7821010c67155a386c2a93302d814c19f9f.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/44d8f2c3fc651b2b65e3b57ef54bc7821010c67155a386c2a93302d814c19f9f.jpg)

![46c97206a358353f4f1d6eb58f1b3918407865958b7e2355cf8fe1d0cd72fb6b.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/46c97206a358353f4f1d6eb58f1b3918407865958b7e2355cf8fe1d0cd72fb6b.jpg)

![48076bcecfaa9f1ea673d2deb57822ed0db51f5322abc4cb30ac594835e9958d.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/48076bcecfaa9f1ea673d2deb57822ed0db51f5322abc4cb30ac594835e9958d.jpg)

![4bf7727ed407ee95237e7671702be0a0d8b6744ef06374159ea46c4ad9526fdf.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/4bf7727ed407ee95237e7671702be0a0d8b6744ef06374159ea46c4ad9526fdf.jpg)

![5b4b07502916c8ac3c0e6a42697edbb1486069244312f500df1453de25c5069f.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/5b4b07502916c8ac3c0e6a42697edbb1486069244312f500df1453de25c5069f.jpg)

![6cbc7aa1f2335b16983ddf4c93a7403afd302f7e871c02f6bb15fa74e95f0830.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/6cbc7aa1f2335b16983ddf4c93a7403afd302f7e871c02f6bb15fa74e95f0830.jpg)

![7d568fb10cf100e4163ab7ff97a0eaac9bd05b937282925201623f5376691ed2.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/7d568fb10cf100e4163ab7ff97a0eaac9bd05b937282925201623f5376691ed2.jpg)

![807405a1b217ce9c088f14c4a702b434d73cef4b59aaf1b85a9d4bc8071b87a9.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/807405a1b217ce9c088f14c4a702b434d73cef4b59aaf1b85a9d4bc8071b87a9.jpg)

![8085dbe77d6a2f06cdc588fa0380db8cf3f8da0cf6ee58dc3e35fb45b0fedd5e.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/8085dbe77d6a2f06cdc588fa0380db8cf3f8da0cf6ee58dc3e35fb45b0fedd5e.jpg)

![895f5b56fb6d47d4acc0e440a06bc03925434863e9ac4b7d429f2be7e606094a.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/895f5b56fb6d47d4acc0e440a06bc03925434863e9ac4b7d429f2be7e606094a.jpg)

![96d2d6f746f7031be1f8112ac807a1f98e14d780b16442054955c442c8f9634f.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/96d2d6f746f7031be1f8112ac807a1f98e14d780b16442054955c442c8f9634f.jpg)

![a355d30ea3656d579c6e44caa16ef6462ec7743c615d3cad31e1b4ea98012402.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/a355d30ea3656d579c6e44caa16ef6462ec7743c615d3cad31e1b4ea98012402.jpg)

![a83c5528c5c2ac32db79d10bcfe524169754e0380ffa948c5389697675376b02.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/a83c5528c5c2ac32db79d10bcfe524169754e0380ffa948c5389697675376b02.jpg)

![b092263776dbcaabb2a0b48fc30e98ca932832c2126b1c834e13819048c77d9f.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/b092263776dbcaabb2a0b48fc30e98ca932832c2126b1c834e13819048c77d9f.jpg)

![c1b50198b08b7577591d078a996fa304bc756c21b8c26c7df51f6224031b606f.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/c1b50198b08b7577591d078a996fa304bc756c21b8c26c7df51f6224031b606f.jpg)

![c688178fb532b67ee173837eb73a524b61f1b54ad22c8315a0e43ba7f2d0337c.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/c688178fb532b67ee173837eb73a524b61f1b54ad22c8315a0e43ba7f2d0337c.jpg)

![d90eba89aea6b397a3aae0698774dcd3e0c465c59a18c55da8853276bd8beae9.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/d90eba89aea6b397a3aae0698774dcd3e0c465c59a18c55da8853276bd8beae9.jpg)

![da9458add01199cc7f671f91e53215e34039b2317d349384cccf666d182d4234.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/da9458add01199cc7f671f91e53215e34039b2317d349384cccf666d182d4234.jpg)

![f61bf75e1d4e8d5f94da38368667811ef5ed8bbfc4386b782a06635e6c80bc99.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/f61bf75e1d4e8d5f94da38368667811ef5ed8bbfc4386b782a06635e6c80bc99.jpg)

![fb1580c59c5e6aebaf758ba28ff1c73feda3713932796a2f2830442134d52557.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/fb1580c59c5e6aebaf758ba28ff1c73feda3713932796a2f2830442134d52557.jpg)

![ff3bb4ad8a37ad17906ca3fd025188ceeb7e90f378cd2ba0aabcab2ef17fd959.jpg](../iclr_results/2048_MMIU_ Multimodal Multi-image Understanding for Evaluating Large Vision-Language Models/tables/ff3bb4ad8a37ad17906ca3fd025188ceeb7e90f378cd2ba0aabcab2ef17fd959.jpg)

## Contextual Document Embeddings


### Images

![11fcc76cb33241ff627e4802044ce86821a7f8003968785cddee714a5a729278.jpg](../iclr_results/2049_Contextual Document Embeddings/images/11fcc76cb33241ff627e4802044ce86821a7f8003968785cddee714a5a729278.jpg)

### Tables

![49c6b6a74869a5ccde5c1e786134f6975bd80aa70ac7294bd5fd1c3687874572.jpg](../iclr_results/2049_Contextual Document Embeddings/tables/49c6b6a74869a5ccde5c1e786134f6975bd80aa70ac7294bd5fd1c3687874572.jpg)

![a8731dde95701cb474ee079770237cc73af9926c6a74cef8c17fa1cf4cfdfded.jpg](../iclr_results/2049_Contextual Document Embeddings/tables/a8731dde95701cb474ee079770237cc73af9926c6a74cef8c17fa1cf4cfdfded.jpg)

![e7257ab9894f86069278eea9502fd62dcec6c496365165df258f8492fc6cb599.jpg](../iclr_results/2049_Contextual Document Embeddings/tables/e7257ab9894f86069278eea9502fd62dcec6c496365165df258f8492fc6cb599.jpg)

![ef9f343a2d7ce659f269175b615f0a8eae725e3348f6a2e812df25683b1fa9de.jpg](../iclr_results/2049_Contextual Document Embeddings/tables/ef9f343a2d7ce659f269175b615f0a8eae725e3348f6a2e812df25683b1fa9de.jpg)

## Anyprefer: An Agentic Framework for Preference Data Synthesis


### Images

![16f10fb203ab9d7becbc038ebef025446dc04efd4f5fa6fb10e5e68f7e008bde.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/images/16f10fb203ab9d7becbc038ebef025446dc04efd4f5fa6fb10e5e68f7e008bde.jpg)

![243c5249e2c2ca5728481ec3bbcb1a3d4f9c323c1e1176e29c5815004b20ac72.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/images/243c5249e2c2ca5728481ec3bbcb1a3d4f9c323c1e1176e29c5815004b20ac72.jpg)

![288b43ae5999e337667640cc46ae80af02a1244fed4a899d586b7ac50a70321f.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/images/288b43ae5999e337667640cc46ae80af02a1244fed4a899d586b7ac50a70321f.jpg)

![2eacbb0c445d11c99aa4a937e1566309126fa635786f89ddd62f499b4b94176a.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/images/2eacbb0c445d11c99aa4a937e1566309126fa635786f89ddd62f499b4b94176a.jpg)

![5e0ca404cda349ec820af0de13f854b46faf6aec8cf301f3efa9e7ad195379ca.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/images/5e0ca404cda349ec820af0de13f854b46faf6aec8cf301f3efa9e7ad195379ca.jpg)

![86f76c94abc002f9297d3581a4d0f8e192d06e8fcf124841621598d991fe1356.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/images/86f76c94abc002f9297d3581a4d0f8e192d06e8fcf124841621598d991fe1356.jpg)

![94503d941ba0653fc067b8e9575d66984944745cf59d5182c871cb7ee0c16682.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/images/94503d941ba0653fc067b8e9575d66984944745cf59d5182c871cb7ee0c16682.jpg)

![c2e8afd8e5c2fdc739087e9a40dbe2cc4b8a73c0960826e7cb221fa2bd543d6f.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/images/c2e8afd8e5c2fdc739087e9a40dbe2cc4b8a73c0960826e7cb221fa2bd543d6f.jpg)

### Tables

![374bf0559192151fe20aeffcee2671167fa84de220043a01f5343632b4396896.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/tables/374bf0559192151fe20aeffcee2671167fa84de220043a01f5343632b4396896.jpg)

![4b7ccb51111144beb23de3c5dff127f38fc7015cb702d88c53eaa879cc7c72f3.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/tables/4b7ccb51111144beb23de3c5dff127f38fc7015cb702d88c53eaa879cc7c72f3.jpg)

![4c67a67ee6c4159c22a4011ac8d2e918531d90a73d56c6a7a6aad5020465bc28.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/tables/4c67a67ee6c4159c22a4011ac8d2e918531d90a73d56c6a7a6aad5020465bc28.jpg)

![6332a8bfc58814dc948e81c5164c8730828e8eb3c816d89b6747db140e2b8fd6.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/tables/6332a8bfc58814dc948e81c5164c8730828e8eb3c816d89b6747db140e2b8fd6.jpg)

![68f8b6f5b6d758ad740269540f198014801b08e23bd27161bf80a6402d67c8ff.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/tables/68f8b6f5b6d758ad740269540f198014801b08e23bd27161bf80a6402d67c8ff.jpg)

![6b9f4941f31718e6b73a283fc3c75b7ca9ba2e8e107d4ae6e94dfceaadc1735e.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/tables/6b9f4941f31718e6b73a283fc3c75b7ca9ba2e8e107d4ae6e94dfceaadc1735e.jpg)

![759586247b9eb24da08195733139106e9144453f4265f1af7ae402fee514982a.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/tables/759586247b9eb24da08195733139106e9144453f4265f1af7ae402fee514982a.jpg)

![7e15e8dbe57467dc6ac6fbde919f99560063f9d6af9c4cb0ea9694bfa7ca1dbd.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/tables/7e15e8dbe57467dc6ac6fbde919f99560063f9d6af9c4cb0ea9694bfa7ca1dbd.jpg)

![8848659b7a9714af27eee337cc760d24e4c36e44421e315b8b10213135be39cf.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/tables/8848659b7a9714af27eee337cc760d24e4c36e44421e315b8b10213135be39cf.jpg)

![8a33ffbb582f9633b88f3e51eb9599173d624a69d2e0fc3287a1fdd65d9781dc.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/tables/8a33ffbb582f9633b88f3e51eb9599173d624a69d2e0fc3287a1fdd65d9781dc.jpg)

![8e04e853eddb922f9dfc3e58feea8d90265dbe40299877eabcb2643f35d35c44.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/tables/8e04e853eddb922f9dfc3e58feea8d90265dbe40299877eabcb2643f35d35c44.jpg)

![9912a4ba7ede0200613eb9d9c9ab9834ca4f39194a419923548095a150397843.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/tables/9912a4ba7ede0200613eb9d9c9ab9834ca4f39194a419923548095a150397843.jpg)

![a2834fb2fbabd6b36bc225dbb7ffe8f1763ecb24702987084c732c25f63440a5.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/tables/a2834fb2fbabd6b36bc225dbb7ffe8f1763ecb24702987084c732c25f63440a5.jpg)

![bd9f9c28ca0dbb936dc2b5a63436849faa0c9f8f421fb7a2bef74423e14c0b87.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/tables/bd9f9c28ca0dbb936dc2b5a63436849faa0c9f8f421fb7a2bef74423e14c0b87.jpg)

![c577e22eee4a2f77fe0f60d37f760656f37e1b5f96042bccc5f8e23fc808f8ef.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/tables/c577e22eee4a2f77fe0f60d37f760656f37e1b5f96042bccc5f8e23fc808f8ef.jpg)

![c8450ed3046f6d2017e6ad560df351e37db9196c1ed468cc136874b349dc2f39.jpg](../iclr_results/2050_Anyprefer_ An Agentic Framework for Preference Data Synthesis/tables/c8450ed3046f6d2017e6ad560df351e37db9196c1ed468cc136874b349dc2f39.jpg)

## Small Models are LLM Knowledge Triggers for Medical Tabular Prediction


### Images

![5e48f56ed46507c6f4e13ca9179cc5f853f99116d0caed1530d58693df119ec2.jpg](../iclr_results/2051_Small Models are LLM Knowledge Triggers for Medical Tabular Prediction/images/5e48f56ed46507c6f4e13ca9179cc5f853f99116d0caed1530d58693df119ec2.jpg)

![81bd26cc8bfac977bfb21245826f831c438c1538d98758955cffc68f036b7989.jpg](../iclr_results/2051_Small Models are LLM Knowledge Triggers for Medical Tabular Prediction/images/81bd26cc8bfac977bfb21245826f831c438c1538d98758955cffc68f036b7989.jpg)

![87cc881edd56584e0ebab22a73d3da860b31b739e1aa3db6beea9abc98073e38.jpg](../iclr_results/2051_Small Models are LLM Knowledge Triggers for Medical Tabular Prediction/images/87cc881edd56584e0ebab22a73d3da860b31b739e1aa3db6beea9abc98073e38.jpg)

![d2afc461b6859cdfef802b08cdb289d00a77fd0f7b8ed71e47e2608713af7cb4.jpg](../iclr_results/2051_Small Models are LLM Knowledge Triggers for Medical Tabular Prediction/images/d2afc461b6859cdfef802b08cdb289d00a77fd0f7b8ed71e47e2608713af7cb4.jpg)

### Tables

![12fb30eac903ad4ef8b0c8ec126a70d9f1fe23c0c1af7c35364336d8746346fc.jpg](../iclr_results/2051_Small Models are LLM Knowledge Triggers for Medical Tabular Prediction/tables/12fb30eac903ad4ef8b0c8ec126a70d9f1fe23c0c1af7c35364336d8746346fc.jpg)

![56b8cf45a449c4f6ef659db9c36ab99140eb9f5216ac14ed72e006e3bb558f31.jpg](../iclr_results/2051_Small Models are LLM Knowledge Triggers for Medical Tabular Prediction/tables/56b8cf45a449c4f6ef659db9c36ab99140eb9f5216ac14ed72e006e3bb558f31.jpg)

![5ee48152cb70d311ead7ae427fae0dd8d531fe6fee5e2a23a7407f2ab14aec3f.jpg](../iclr_results/2051_Small Models are LLM Knowledge Triggers for Medical Tabular Prediction/tables/5ee48152cb70d311ead7ae427fae0dd8d531fe6fee5e2a23a7407f2ab14aec3f.jpg)

![abc7f67ab6aab95ac24b4a81d7209f37365ab1955a745db97633ef839259c8d1.jpg](../iclr_results/2051_Small Models are LLM Knowledge Triggers for Medical Tabular Prediction/tables/abc7f67ab6aab95ac24b4a81d7209f37365ab1955a745db97633ef839259c8d1.jpg)

![c05afddc9f21f12076b6dd5cc8c2e0e09f2a4ceca36686e5148d437c3acce971.jpg](../iclr_results/2051_Small Models are LLM Knowledge Triggers for Medical Tabular Prediction/tables/c05afddc9f21f12076b6dd5cc8c2e0e09f2a4ceca36686e5148d437c3acce971.jpg)

![c39478b5613d9b4a8bb9b315b5a02de550b6233f3b1e42a1bd37636083ec4058.jpg](../iclr_results/2051_Small Models are LLM Knowledge Triggers for Medical Tabular Prediction/tables/c39478b5613d9b4a8bb9b315b5a02de550b6233f3b1e42a1bd37636083ec4058.jpg)

![eb1f2ec2ce737debf0e9bdf86eac8288474fa496bcec24cf4ad478a4003b07a3.jpg](../iclr_results/2051_Small Models are LLM Knowledge Triggers for Medical Tabular Prediction/tables/eb1f2ec2ce737debf0e9bdf86eac8288474fa496bcec24cf4ad478a4003b07a3.jpg)

![f6e1395f6f59122c948feae71951d880074e1530d4a38eacc682361228bbccf3.jpg](../iclr_results/2051_Small Models are LLM Knowledge Triggers for Medical Tabular Prediction/tables/f6e1395f6f59122c948feae71951d880074e1530d4a38eacc682361228bbccf3.jpg)

## GenDataAgent: On-the-fly Dataset Augmentation with Synthetic Data


### Images

![5e22392d4617b3841ffa5ca581b96784dad14819d879c8b49662bc01cba97ebb.jpg](../iclr_results/2052_GenDataAgent_ On-the-fly Dataset Augmentation with Synthetic Data/images/5e22392d4617b3841ffa5ca581b96784dad14819d879c8b49662bc01cba97ebb.jpg)

![678eb39275a467ba59b844b2d6d564edea9a186d03f2f7f8556f3b8c57abb95c.jpg](../iclr_results/2052_GenDataAgent_ On-the-fly Dataset Augmentation with Synthetic Data/images/678eb39275a467ba59b844b2d6d564edea9a186d03f2f7f8556f3b8c57abb95c.jpg)

![aefc7af90d8e0c6d1da86baa6e99035a048154aeeee128fc84327e00e7d5cea9.jpg](../iclr_results/2052_GenDataAgent_ On-the-fly Dataset Augmentation with Synthetic Data/images/aefc7af90d8e0c6d1da86baa6e99035a048154aeeee128fc84327e00e7d5cea9.jpg)

![b64c78a7b0f364518b01f2a48eb6013f9ed070ad0db0da9772a33284710396fb.jpg](../iclr_results/2052_GenDataAgent_ On-the-fly Dataset Augmentation with Synthetic Data/images/b64c78a7b0f364518b01f2a48eb6013f9ed070ad0db0da9772a33284710396fb.jpg)

![c27345e6fa2e96f4af2f7bdba20ca989e19742a4b41ef7abf885ce167bfb1e16.jpg](../iclr_results/2052_GenDataAgent_ On-the-fly Dataset Augmentation with Synthetic Data/images/c27345e6fa2e96f4af2f7bdba20ca989e19742a4b41ef7abf885ce167bfb1e16.jpg)

![e4bd319ab0d965fc5745d41b06567929c1fd75ee341c532f71d68519b839ce33.jpg](../iclr_results/2052_GenDataAgent_ On-the-fly Dataset Augmentation with Synthetic Data/images/e4bd319ab0d965fc5745d41b06567929c1fd75ee341c532f71d68519b839ce33.jpg)

![eff58effa0a607c1ba617c70e4bc3be7d01a9a4d2c6baec3f746205374825f00.jpg](../iclr_results/2052_GenDataAgent_ On-the-fly Dataset Augmentation with Synthetic Data/images/eff58effa0a607c1ba617c70e4bc3be7d01a9a4d2c6baec3f746205374825f00.jpg)

![fa3f60a6e55b57bc9740c97e967836fd09ffb10269ceb00e149d480a92da33b9.jpg](../iclr_results/2052_GenDataAgent_ On-the-fly Dataset Augmentation with Synthetic Data/images/fa3f60a6e55b57bc9740c97e967836fd09ffb10269ceb00e149d480a92da33b9.jpg)

### Tables

![0e4783a596489c2e04f2f41df24207888baea1c8d894818c5beeba8a2501fd83.jpg](../iclr_results/2052_GenDataAgent_ On-the-fly Dataset Augmentation with Synthetic Data/tables/0e4783a596489c2e04f2f41df24207888baea1c8d894818c5beeba8a2501fd83.jpg)

![18c5cde49e24d48d357924d692d09587cd97b6b07906532b7f997663c19a92d3.jpg](../iclr_results/2052_GenDataAgent_ On-the-fly Dataset Augmentation with Synthetic Data/tables/18c5cde49e24d48d357924d692d09587cd97b6b07906532b7f997663c19a92d3.jpg)

![1d49b46fcdf385b6598b890d78cd86a03b7fd4d4463840b1b3ef2e898958d910.jpg](../iclr_results/2052_GenDataAgent_ On-the-fly Dataset Augmentation with Synthetic Data/tables/1d49b46fcdf385b6598b890d78cd86a03b7fd4d4463840b1b3ef2e898958d910.jpg)

![223d3ddce5d8de48ab0e2210247d5d48d9f1401e6e070f3ad5c8f9989d26ab74.jpg](../iclr_results/2052_GenDataAgent_ On-the-fly Dataset Augmentation with Synthetic Data/tables/223d3ddce5d8de48ab0e2210247d5d48d9f1401e6e070f3ad5c8f9989d26ab74.jpg)

![4730bfe78b7cb2252e610c3c57dd618c0554c4f82a1a49597828ff165746dfe4.jpg](../iclr_results/2052_GenDataAgent_ On-the-fly Dataset Augmentation with Synthetic Data/tables/4730bfe78b7cb2252e610c3c57dd618c0554c4f82a1a49597828ff165746dfe4.jpg)

![4f75b5a77eec433bfa8e7e43dd5caeb700cbe1c4359eedcbae4f88d182f1da4a.jpg](../iclr_results/2052_GenDataAgent_ On-the-fly Dataset Augmentation with Synthetic Data/tables/4f75b5a77eec433bfa8e7e43dd5caeb700cbe1c4359eedcbae4f88d182f1da4a.jpg)

![6c3f4ac77127be2307fa18d4ad76c50a4fc066467399f1e86c4c3a8cb9e8d776.jpg](../iclr_results/2052_GenDataAgent_ On-the-fly Dataset Augmentation with Synthetic Data/tables/6c3f4ac77127be2307fa18d4ad76c50a4fc066467399f1e86c4c3a8cb9e8d776.jpg)

![792f881199d6be9a2b31d410c177f507f62b0ba1eb05eaacf63d90b7aecbcbae.jpg](../iclr_results/2052_GenDataAgent_ On-the-fly Dataset Augmentation with Synthetic Data/tables/792f881199d6be9a2b31d410c177f507f62b0ba1eb05eaacf63d90b7aecbcbae.jpg)

![9f3e150519828640eef147f5a61279f7432ea5f09c73bc45560400737e125051.jpg](../iclr_results/2052_GenDataAgent_ On-the-fly Dataset Augmentation with Synthetic Data/tables/9f3e150519828640eef147f5a61279f7432ea5f09c73bc45560400737e125051.jpg)

![b62810e586d25c7aa6bef4f02dbc3e83b140550079fe7c0cc7747171de629427.jpg](../iclr_results/2052_GenDataAgent_ On-the-fly Dataset Augmentation with Synthetic Data/tables/b62810e586d25c7aa6bef4f02dbc3e83b140550079fe7c0cc7747171de629427.jpg)

![b9fbda45bc66b4bf46dff704c39fb8512748aa0bd997860c0714d52b875473f4.jpg](../iclr_results/2052_GenDataAgent_ On-the-fly Dataset Augmentation with Synthetic Data/tables/b9fbda45bc66b4bf46dff704c39fb8512748aa0bd997860c0714d52b875473f4.jpg)

![cfa8ac1e637127c3f7d7baca22fccfbd03aba11b94d44cc170b02bd653e4e364.jpg](../iclr_results/2052_GenDataAgent_ On-the-fly Dataset Augmentation with Synthetic Data/tables/cfa8ac1e637127c3f7d7baca22fccfbd03aba11b94d44cc170b02bd653e4e364.jpg)

![d1f0238dde0a2d74d0608137b7a35152da70a8902f18612cf02b2bd8148719bb.jpg](../iclr_results/2052_GenDataAgent_ On-the-fly Dataset Augmentation with Synthetic Data/tables/d1f0238dde0a2d74d0608137b7a35152da70a8902f18612cf02b2bd8148719bb.jpg)

![e7afb6d744bde289b9f89dc3a57b234ffefb728a66ef3d7acb6b306720d77185.jpg](../iclr_results/2052_GenDataAgent_ On-the-fly Dataset Augmentation with Synthetic Data/tables/e7afb6d744bde289b9f89dc3a57b234ffefb728a66ef3d7acb6b306720d77185.jpg)

![f732e902765a33007846925246dde8ced29686549b3787f98a7a11499d4c84d8.jpg](../iclr_results/2052_GenDataAgent_ On-the-fly Dataset Augmentation with Synthetic Data/tables/f732e902765a33007846925246dde8ced29686549b3787f98a7a11499d4c84d8.jpg)

## GMValuator: Similarity-based Data Valuation for Generative Models


### Images

![12e25a1843fbf50be5e89baa10f960ab89c29315f8589183fb5800f711ab6268.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/images/12e25a1843fbf50be5e89baa10f960ab89c29315f8589183fb5800f711ab6268.jpg)

![44dd6643c5b17ec795d89607c24349f456616c5f3abf6ddf6fdb57f8f015471f.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/images/44dd6643c5b17ec795d89607c24349f456616c5f3abf6ddf6fdb57f8f015471f.jpg)

![5cda964268fb153b783c4ede6d1086889cacb85b20e6a54dff75bc1412702b86.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/images/5cda964268fb153b783c4ede6d1086889cacb85b20e6a54dff75bc1412702b86.jpg)

![62c518444569a17c1fde913d53cbaa35473c7befdeee6d6e723f5549204d6c08.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/images/62c518444569a17c1fde913d53cbaa35473c7befdeee6d6e723f5549204d6c08.jpg)

![63f7fa2b336f1c8b06b1f0527e439c4b6a53e6f2febfaf0a4d02a80236054913.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/images/63f7fa2b336f1c8b06b1f0527e439c4b6a53e6f2febfaf0a4d02a80236054913.jpg)

![74ed9bcdf7a8a949c20799364aa297abf01e9cf49676ceab66b2522324793b8a.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/images/74ed9bcdf7a8a949c20799364aa297abf01e9cf49676ceab66b2522324793b8a.jpg)

![873cae624449ceca5e8fd4fea0b038fa1879d8a4edc2d2479adbfa15cf552c43.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/images/873cae624449ceca5e8fd4fea0b038fa1879d8a4edc2d2479adbfa15cf552c43.jpg)

![930d07c56f864ae804e68e1fa18e8676a964636dfefa271967f152e6bb0ef758.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/images/930d07c56f864ae804e68e1fa18e8676a964636dfefa271967f152e6bb0ef758.jpg)

![9b028e80bca75d5a70e133a591d227a3d6e98ee132b1419617ee6e9f93e047fd.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/images/9b028e80bca75d5a70e133a591d227a3d6e98ee132b1419617ee6e9f93e047fd.jpg)

![baf5fcb2aacfd8b81638d14d35ece5a13fd0c33c4cbca97e6dc80a826caa33f2.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/images/baf5fcb2aacfd8b81638d14d35ece5a13fd0c33c4cbca97e6dc80a826caa33f2.jpg)

![ed0130d8267b5e1c5e39b1cff757233ae60606c922ed0e5693db9b9417e218ad.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/images/ed0130d8267b5e1c5e39b1cff757233ae60606c922ed0e5693db9b9417e218ad.jpg)

![f25e4fa5b61f42a4531b7e05a009e064d6de389c9d5ec5b315e9359433019560.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/images/f25e4fa5b61f42a4531b7e05a009e064d6de389c9d5ec5b315e9359433019560.jpg)

### Tables

![0186a17a283962558e24c33673cf5617003bc4c881b443058334442d43d3adfb.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/tables/0186a17a283962558e24c33673cf5617003bc4c881b443058334442d43d3adfb.jpg)

![1181b9e86eda5f98c861d7412534f271e9252a4fa9de067f878ecedc432d9978.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/tables/1181b9e86eda5f98c861d7412534f271e9252a4fa9de067f878ecedc432d9978.jpg)

![1aaacca775600866bc90110e840b9c69f28ee1adc34e9dfca22fca7a520a9892.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/tables/1aaacca775600866bc90110e840b9c69f28ee1adc34e9dfca22fca7a520a9892.jpg)

![232b020ad02fd4a4df1fd71793af14a21d2ab3b6ca2343ff87e5c8a8d68cfb95.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/tables/232b020ad02fd4a4df1fd71793af14a21d2ab3b6ca2343ff87e5c8a8d68cfb95.jpg)

![563ac5a6787de4d240aafddf57c9ae63403484d9c0f5ed6f2ac36ae8bc01b412.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/tables/563ac5a6787de4d240aafddf57c9ae63403484d9c0f5ed6f2ac36ae8bc01b412.jpg)

![5de1f4cd3a42a36bfd03725abca83b0785be42059ab29efcc1dfa59e914209f3.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/tables/5de1f4cd3a42a36bfd03725abca83b0785be42059ab29efcc1dfa59e914209f3.jpg)

![61e708e7e4162a976192f59ea132d7af36fbb4e5ea4fa47c6c4b5ade1ec88e90.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/tables/61e708e7e4162a976192f59ea132d7af36fbb4e5ea4fa47c6c4b5ade1ec88e90.jpg)

![64b0751035a81a435bdf4f3fd671c80fcfbe266f9ba2b434fadcec876f5ad531.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/tables/64b0751035a81a435bdf4f3fd671c80fcfbe266f9ba2b434fadcec876f5ad531.jpg)

![7a660d030c690d887521604dd2449707b7101dcd2eb8e8de4cbe164bd7aef87f.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/tables/7a660d030c690d887521604dd2449707b7101dcd2eb8e8de4cbe164bd7aef87f.jpg)

![92b932e39995e8ebf256a50918a89a793a1c8d5bd442691986e21209437ae30e.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/tables/92b932e39995e8ebf256a50918a89a793a1c8d5bd442691986e21209437ae30e.jpg)

![a298a688aae3ea8fc8b7b9a3f43f654d31e08b6386ebcafef04745c4268f46ef.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/tables/a298a688aae3ea8fc8b7b9a3f43f654d31e08b6386ebcafef04745c4268f46ef.jpg)

![bb5e91bfd04da59c53cd8d8737f2e476441fcc6d46cebcc020bc02c16021c73b.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/tables/bb5e91bfd04da59c53cd8d8737f2e476441fcc6d46cebcc020bc02c16021c73b.jpg)

![bc9cbe94803dd59078a48a85445a51b53cd77ba7cacd4636153932e67248a948.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/tables/bc9cbe94803dd59078a48a85445a51b53cd77ba7cacd4636153932e67248a948.jpg)

![d3c2c2796e9a037a9d9d71e9bca2884a9f0b5ec38b606117743b1ab70438d667.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/tables/d3c2c2796e9a037a9d9d71e9bca2884a9f0b5ec38b606117743b1ab70438d667.jpg)

![d869fec0da40922014c15ffe842239ef7d80cd144220dc27eae439de393fb0b7.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/tables/d869fec0da40922014c15ffe842239ef7d80cd144220dc27eae439de393fb0b7.jpg)

![fcc19315de193d76d2f6544eedf53c05f01712a22d68aacf6597fa9c8b357a02.jpg](../iclr_results/2053_GMValuator_ Similarity-based Data Valuation for Generative Models/tables/fcc19315de193d76d2f6544eedf53c05f01712a22d68aacf6597fa9c8b357a02.jpg)

## Optimal Learning of Kernel Logistic Regression for Complex Classification Scenarios


### Images

![a587443fc7e57f91bc9983ff8c8fd19968dba0011a7a1a606fc4eb02a03e623b.jpg](../iclr_results/2054_Optimal Learning of Kernel Logistic Regression for Complex Classification Scenarios/images/a587443fc7e57f91bc9983ff8c8fd19968dba0011a7a1a606fc4eb02a03e623b.jpg)

### Tables

![2b0a8c914034aaf7152ba1bdf12516ec718227456bfbeb7d87b1619d39ac38c4.jpg](../iclr_results/2054_Optimal Learning of Kernel Logistic Regression for Complex Classification Scenarios/tables/2b0a8c914034aaf7152ba1bdf12516ec718227456bfbeb7d87b1619d39ac38c4.jpg)

![94f7fd4b06db9a3e4ae885478abb3fea7524703964578bac643021e40e6c9a84.jpg](../iclr_results/2054_Optimal Learning of Kernel Logistic Regression for Complex Classification Scenarios/tables/94f7fd4b06db9a3e4ae885478abb3fea7524703964578bac643021e40e6c9a84.jpg)

## Building Math Agents with Multi-Turn Iterative Preference Learning


### Images

![1065cdecfd476ee4159a2f527d12bc6554c486781398f271d67ffcbc75d60f86.jpg](../iclr_results/2055_Building Math Agents with Multi-Turn Iterative Preference Learning/images/1065cdecfd476ee4159a2f527d12bc6554c486781398f271d67ffcbc75d60f86.jpg)

![1eac78d7865bdc37adcf5a6e164c4a77373bd4c17f11d4dc36383d4fb0502375.jpg](../iclr_results/2055_Building Math Agents with Multi-Turn Iterative Preference Learning/images/1eac78d7865bdc37adcf5a6e164c4a77373bd4c17f11d4dc36383d4fb0502375.jpg)

![233418f43d0b03874c56ffb67b79adb4792a7108dc09420972e74213b73ffebd.jpg](../iclr_results/2055_Building Math Agents with Multi-Turn Iterative Preference Learning/images/233418f43d0b03874c56ffb67b79adb4792a7108dc09420972e74213b73ffebd.jpg)

![4d24dfdb5fe8241fc2f318b167d3822b0c92a0993ad5b05c0a48cec781a828c8.jpg](../iclr_results/2055_Building Math Agents with Multi-Turn Iterative Preference Learning/images/4d24dfdb5fe8241fc2f318b167d3822b0c92a0993ad5b05c0a48cec781a828c8.jpg)

![7a43869c0c2daaa48b1554de5efcb3654317617bc0e12cdaf69406a46c1359c7.jpg](../iclr_results/2055_Building Math Agents with Multi-Turn Iterative Preference Learning/images/7a43869c0c2daaa48b1554de5efcb3654317617bc0e12cdaf69406a46c1359c7.jpg)

### Tables

![119f1773129aa6931e76b72da4aaa5999f107ed244246941a9feb8bb91910c2f.jpg](../iclr_results/2055_Building Math Agents with Multi-Turn Iterative Preference Learning/tables/119f1773129aa6931e76b72da4aaa5999f107ed244246941a9feb8bb91910c2f.jpg)

![44ec4c5f21e1348533ac481bf596b21f66ae0fc4fa0b1a7de952507888b3d12c.jpg](../iclr_results/2055_Building Math Agents with Multi-Turn Iterative Preference Learning/tables/44ec4c5f21e1348533ac481bf596b21f66ae0fc4fa0b1a7de952507888b3d12c.jpg)

![694150165ecf1cdf8bf9c87f0202d4b16b484a629a60e6ca8b1311f348fdd398.jpg](../iclr_results/2055_Building Math Agents with Multi-Turn Iterative Preference Learning/tables/694150165ecf1cdf8bf9c87f0202d4b16b484a629a60e6ca8b1311f348fdd398.jpg)

![8351a6cbc4c756b578c3a26a97e4847eb174e0e8245756246a08e558b8ccbe1d.jpg](../iclr_results/2055_Building Math Agents with Multi-Turn Iterative Preference Learning/tables/8351a6cbc4c756b578c3a26a97e4847eb174e0e8245756246a08e558b8ccbe1d.jpg)

![934397f19bf74f08dc904c248a4e507ae35771d82cff75de84b3308e7aef0a62.jpg](../iclr_results/2055_Building Math Agents with Multi-Turn Iterative Preference Learning/tables/934397f19bf74f08dc904c248a4e507ae35771d82cff75de84b3308e7aef0a62.jpg)

![cb22676905327d71c085eb9d8546caf1037211382bf2329a2835603d87a57c84.jpg](../iclr_results/2055_Building Math Agents with Multi-Turn Iterative Preference Learning/tables/cb22676905327d71c085eb9d8546caf1037211382bf2329a2835603d87a57c84.jpg)

## DyCAST: Learning Dynamic Causal Structure from Time Series


### Images

![1d28f6ca9abad940696c568fbc954a2c96fdae05af56014f14a5f536e6b7f710.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/images/1d28f6ca9abad940696c568fbc954a2c96fdae05af56014f14a5f536e6b7f710.jpg)

![2843302c22d2b7f0a329d3c3beae5cbecc4b92b0a85c532c0e08ef2c6b1b30e3.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/images/2843302c22d2b7f0a329d3c3beae5cbecc4b92b0a85c532c0e08ef2c6b1b30e3.jpg)

![2a232ce9884e26c8d1a1c0476f9c32816c6cfe905f873a3e36f1f141a75a3658.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/images/2a232ce9884e26c8d1a1c0476f9c32816c6cfe905f873a3e36f1f141a75a3658.jpg)

![2eb8d298c2b5c1a757621fde78657c8b7d60f749ebca9757fc7016a7841827f1.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/images/2eb8d298c2b5c1a757621fde78657c8b7d60f749ebca9757fc7016a7841827f1.jpg)

![32462a44e2a21a42fcf789f4d268ee93bda7a934c27c832f9102949a1eef1fd8.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/images/32462a44e2a21a42fcf789f4d268ee93bda7a934c27c832f9102949a1eef1fd8.jpg)

![362da6f26016cdc5549900f7ec6e17fd0b83b3e3c7d10ac81dd082412a018710.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/images/362da6f26016cdc5549900f7ec6e17fd0b83b3e3c7d10ac81dd082412a018710.jpg)

![4d8a61021358af8bc6566538871fd028b159b3fef167ef97fef99c25d4caedae.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/images/4d8a61021358af8bc6566538871fd028b159b3fef167ef97fef99c25d4caedae.jpg)

![582e59ca5aa5e3e5d2b331de48960a93bf580458dfd1316bfaad08a05a2def40.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/images/582e59ca5aa5e3e5d2b331de48960a93bf580458dfd1316bfaad08a05a2def40.jpg)

![589861a8de687356fbee573bb71efba822ad90462dc7f9359fd5ac91b29859f3.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/images/589861a8de687356fbee573bb71efba822ad90462dc7f9359fd5ac91b29859f3.jpg)

![6404d9a2e9587b5967c8099eccb86592f17f42a4c2c5100ab9d6e5230b71d7a9.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/images/6404d9a2e9587b5967c8099eccb86592f17f42a4c2c5100ab9d6e5230b71d7a9.jpg)

![6665208e654e1053b0c08f5efc5fc752e8cf60e5a6e1645202f433d47a2f4d1f.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/images/6665208e654e1053b0c08f5efc5fc752e8cf60e5a6e1645202f433d47a2f4d1f.jpg)

![69fc8cfc95c29af7989cb8e4ecaefab7e404bf5ffa47a2f45ad8b596c3d7b870.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/images/69fc8cfc95c29af7989cb8e4ecaefab7e404bf5ffa47a2f45ad8b596c3d7b870.jpg)

![76dacd344f7f0c22b4738095918d8d3d74b522b33913dc99e4258ac787df4bd7.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/images/76dacd344f7f0c22b4738095918d8d3d74b522b33913dc99e4258ac787df4bd7.jpg)

![98d1f75ac715db3c9161f6a19568ad17f422e7440de5a4db674e59d6c461422f.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/images/98d1f75ac715db3c9161f6a19568ad17f422e7440de5a4db674e59d6c461422f.jpg)

![9c8d8b7a6ac00cffc061bfad74902a5058ba433870022c2594f62553ac5a6f0a.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/images/9c8d8b7a6ac00cffc061bfad74902a5058ba433870022c2594f62553ac5a6f0a.jpg)

![c323f285fd7c6bd88895f9a36811ac41fe3e6cdcf9011ffbce0944d0fb950ee2.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/images/c323f285fd7c6bd88895f9a36811ac41fe3e6cdcf9011ffbce0944d0fb950ee2.jpg)

![e6f55f4320d3a78c800b488511e2398c22a7903419bf99f5b3974290da52add4.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/images/e6f55f4320d3a78c800b488511e2398c22a7903419bf99f5b3974290da52add4.jpg)

### Tables

![00c77034175d1fb3cb8e7ebb05b78e4ba9c0a9a09919f40259d6617a7a0759d9.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/tables/00c77034175d1fb3cb8e7ebb05b78e4ba9c0a9a09919f40259d6617a7a0759d9.jpg)

![025deca9b2b64ccfd6b55c13857193d481099f458819ea2e0dafe6cf4020f854.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/tables/025deca9b2b64ccfd6b55c13857193d481099f458819ea2e0dafe6cf4020f854.jpg)

![178890970f4116e72a399abfd0169ac6929f2e370ca888feca71df68f286027f.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/tables/178890970f4116e72a399abfd0169ac6929f2e370ca888feca71df68f286027f.jpg)

![1c329b1536761a1b9e0818ce766c798902cb20918670e6df5526cd6e596ea94f.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/tables/1c329b1536761a1b9e0818ce766c798902cb20918670e6df5526cd6e596ea94f.jpg)

![2592f43244db65f9ee7b6159126cc810896bf2469d4f12d80831cb88509bb42c.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/tables/2592f43244db65f9ee7b6159126cc810896bf2469d4f12d80831cb88509bb42c.jpg)

![2f3bf1e64295632654b68da9fe03a3a3ece64296ecb60fa16aa5fe78a9c1a846.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/tables/2f3bf1e64295632654b68da9fe03a3a3ece64296ecb60fa16aa5fe78a9c1a846.jpg)

![598af99a57e19db5f648bcc02399dccc6d1f062b96309249a92d42f454e3a1f2.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/tables/598af99a57e19db5f648bcc02399dccc6d1f062b96309249a92d42f454e3a1f2.jpg)

![68a611025d223674dbf363ef31bc09f5f8320476f04844cc464342a0b456a518.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/tables/68a611025d223674dbf363ef31bc09f5f8320476f04844cc464342a0b456a518.jpg)

![6cee764e514f01cd90b1bf475ddf274d2fdc9ae02db4b0b334b263fae9910ca4.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/tables/6cee764e514f01cd90b1bf475ddf274d2fdc9ae02db4b0b334b263fae9910ca4.jpg)

![b9512c2ccf1cda39a83401a1f18a6bc93afdc0e8f17c26e24c24ba4320b9e560.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/tables/b9512c2ccf1cda39a83401a1f18a6bc93afdc0e8f17c26e24c24ba4320b9e560.jpg)

![bbe38a52f9d8ebbe54f109ee073b9ae8fff90064aee1fc92ee2fe0774c5b0b5d.jpg](../iclr_results/2056_DyCAST_ Learning Dynamic Causal Structure from Time Series/tables/bbe38a52f9d8ebbe54f109ee073b9ae8fff90064aee1fc92ee2fe0774c5b0b5d.jpg)

## Three Mechanisms of Feature Learning in a Linear Network


### Images

![116cb591cc63cc574ed8c8a83b1fed1d1c22d4092fedddf8f4b3c40bcf60b06e.jpg](../iclr_results/2057_Three Mechanisms of Feature Learning in a Linear Network/images/116cb591cc63cc574ed8c8a83b1fed1d1c22d4092fedddf8f4b3c40bcf60b06e.jpg)

![4b060d0883937533eaa0b710debee6af9f0ffa8e1c292206b5dce92951bde59a.jpg](../iclr_results/2057_Three Mechanisms of Feature Learning in a Linear Network/images/4b060d0883937533eaa0b710debee6af9f0ffa8e1c292206b5dce92951bde59a.jpg)

![54fc83fcda3d592dbea2b75afcb615fa7a43ab8b9b7419b65b1d6bb3efb69e49.jpg](../iclr_results/2057_Three Mechanisms of Feature Learning in a Linear Network/images/54fc83fcda3d592dbea2b75afcb615fa7a43ab8b9b7419b65b1d6bb3efb69e49.jpg)

![7721661a875aa96b845a7a117ddbe51dee9feccbca7104b0bbca1ba65e6ae66b.jpg](../iclr_results/2057_Three Mechanisms of Feature Learning in a Linear Network/images/7721661a875aa96b845a7a117ddbe51dee9feccbca7104b0bbca1ba65e6ae66b.jpg)

![9b4a6bb88633aeb236f8486a5e1557d9834bd40d4a0cf35558e7ae5c46d4b397.jpg](../iclr_results/2057_Three Mechanisms of Feature Learning in a Linear Network/images/9b4a6bb88633aeb236f8486a5e1557d9834bd40d4a0cf35558e7ae5c46d4b397.jpg)

![a2f3405d25b0c96119831ada9a7838b988d3d2ce7aa74c50b6c0bd3bfc6bad37.jpg](../iclr_results/2057_Three Mechanisms of Feature Learning in a Linear Network/images/a2f3405d25b0c96119831ada9a7838b988d3d2ce7aa74c50b6c0bd3bfc6bad37.jpg)

![b383020c57c4b641b93e1375cbc2cfa54587a800665600a8fdbbcb6e0b508e29.jpg](../iclr_results/2057_Three Mechanisms of Feature Learning in a Linear Network/images/b383020c57c4b641b93e1375cbc2cfa54587a800665600a8fdbbcb6e0b508e29.jpg)

![b83d79f5227eb4f9824335e8b546c33f2eedbf472bd77bf17cec4d7c1b0fb6bd.jpg](../iclr_results/2057_Three Mechanisms of Feature Learning in a Linear Network/images/b83d79f5227eb4f9824335e8b546c33f2eedbf472bd77bf17cec4d7c1b0fb6bd.jpg)

![da382084c5d489545561c9c18ad93ffb27191bbb4053bc8bb77a652943444ea5.jpg](../iclr_results/2057_Three Mechanisms of Feature Learning in a Linear Network/images/da382084c5d489545561c9c18ad93ffb27191bbb4053bc8bb77a652943444ea5.jpg)

![eee1181708c4562e7f0dfcc9040a7a59c324cbb21648bf75a38580f3ac910cd3.jpg](../iclr_results/2057_Three Mechanisms of Feature Learning in a Linear Network/images/eee1181708c4562e7f0dfcc9040a7a59c324cbb21648bf75a38580f3ac910cd3.jpg)

### Tables

![07589e3702034111814532d963ed409862e425745a59ea5602d8fd5c3a1d0db4.jpg](../iclr_results/2057_Three Mechanisms of Feature Learning in a Linear Network/tables/07589e3702034111814532d963ed409862e425745a59ea5602d8fd5c3a1d0db4.jpg)

![52298239de16d1a0df01a5aa5c0adb30e18366801a1d6265914a22a82b9287bc.jpg](../iclr_results/2057_Three Mechanisms of Feature Learning in a Linear Network/tables/52298239de16d1a0df01a5aa5c0adb30e18366801a1d6265914a22a82b9287bc.jpg)

## Graph Neural Networks Gone Hogwild


### Images

![11ba2b5f5729b7f0192124621e4dae80510cde45aab2e764b0fef31546a45039.jpg](../iclr_results/2058_Graph Neural Networks Gone Hogwild/images/11ba2b5f5729b7f0192124621e4dae80510cde45aab2e764b0fef31546a45039.jpg)

### Tables

![2742ad46309d6e1967284bee143a50e27bbf49a2e7ccac048616a4ef3087b619.jpg](../iclr_results/2058_Graph Neural Networks Gone Hogwild/tables/2742ad46309d6e1967284bee143a50e27bbf49a2e7ccac048616a4ef3087b619.jpg)

![509f16f9cdf13459a225330ee5f04f533f27c388e941c522ca45314e748f4e47.jpg](../iclr_results/2058_Graph Neural Networks Gone Hogwild/tables/509f16f9cdf13459a225330ee5f04f533f27c388e941c522ca45314e748f4e47.jpg)

![52642b5426acecaa153e50bb6df935a59c33d10e9345ec380996ba6bbb81fa38.jpg](../iclr_results/2058_Graph Neural Networks Gone Hogwild/tables/52642b5426acecaa153e50bb6df935a59c33d10e9345ec380996ba6bbb81fa38.jpg)

![a688c5ff19c04064eb3fee9de286991d64553d1e4fd445a7d7c6d49430b8967a.jpg](../iclr_results/2058_Graph Neural Networks Gone Hogwild/tables/a688c5ff19c04064eb3fee9de286991d64553d1e4fd445a7d7c6d49430b8967a.jpg)

![a6aba53a9e5e8ca4c4a9a61d66a4248da9b1b9a54ca876e00dbc5368a9ab6e45.jpg](../iclr_results/2058_Graph Neural Networks Gone Hogwild/tables/a6aba53a9e5e8ca4c4a9a61d66a4248da9b1b9a54ca876e00dbc5368a9ab6e45.jpg)

![c29d78535a6f696b07184660e1bae5e7556a95a4384a80ac606ed6a338cd9d4f.jpg](../iclr_results/2058_Graph Neural Networks Gone Hogwild/tables/c29d78535a6f696b07184660e1bae5e7556a95a4384a80ac606ed6a338cd9d4f.jpg)

## Decoding Game: On Minimax Optimality of Heuristic Text Generation Strategies


### Images

![603d98c3be48686ba6b50bd8d86e596a1d6f336c659373dc0b08769f287c12c2.jpg](../iclr_results/2059_Decoding Game_ On Minimax Optimality of Heuristic Text Generation Strategies/images/603d98c3be48686ba6b50bd8d86e596a1d6f336c659373dc0b08769f287c12c2.jpg)

### Tables

![022e148e3dccb3728fb12772720f7c69f698b7584bf4bc2734c8b095821514ef.jpg](../iclr_results/2059_Decoding Game_ On Minimax Optimality of Heuristic Text Generation Strategies/tables/022e148e3dccb3728fb12772720f7c69f698b7584bf4bc2734c8b095821514ef.jpg)

![1da3fb8286637f781e5de96ad161f4c8ba3be3ae2f588cd634bb30c7fb16df94.jpg](../iclr_results/2059_Decoding Game_ On Minimax Optimality of Heuristic Text Generation Strategies/tables/1da3fb8286637f781e5de96ad161f4c8ba3be3ae2f588cd634bb30c7fb16df94.jpg)

![2d708ef0163313065c901688e5a0532a82c79a19894b0c84c7af3930244b0e4d.jpg](../iclr_results/2059_Decoding Game_ On Minimax Optimality of Heuristic Text Generation Strategies/tables/2d708ef0163313065c901688e5a0532a82c79a19894b0c84c7af3930244b0e4d.jpg)

![32ef5972d33818107a877455af0f9abff8744c11a662e9d324b8d6e4d295d526.jpg](../iclr_results/2059_Decoding Game_ On Minimax Optimality of Heuristic Text Generation Strategies/tables/32ef5972d33818107a877455af0f9abff8744c11a662e9d324b8d6e4d295d526.jpg)

![4f71caeadbfedfbfbd9841d0ea72ff7c48913c5280259413d3e53332087f6292.jpg](../iclr_results/2059_Decoding Game_ On Minimax Optimality of Heuristic Text Generation Strategies/tables/4f71caeadbfedfbfbd9841d0ea72ff7c48913c5280259413d3e53332087f6292.jpg)

![7193520483d38ed3d0b8dcb77fc420f111190b131b3386196ce74059b3aefdb3.jpg](../iclr_results/2059_Decoding Game_ On Minimax Optimality of Heuristic Text Generation Strategies/tables/7193520483d38ed3d0b8dcb77fc420f111190b131b3386196ce74059b3aefdb3.jpg)

![73261b41abd10dc5c24fb2310715da221257e690e39ab785612b28d1a5316e58.jpg](../iclr_results/2059_Decoding Game_ On Minimax Optimality of Heuristic Text Generation Strategies/tables/73261b41abd10dc5c24fb2310715da221257e690e39ab785612b28d1a5316e58.jpg)

![8d926dc27adc466f2c9c3e8f227ed61d0f0722c3ff1e2fcddda85966e59c430b.jpg](../iclr_results/2059_Decoding Game_ On Minimax Optimality of Heuristic Text Generation Strategies/tables/8d926dc27adc466f2c9c3e8f227ed61d0f0722c3ff1e2fcddda85966e59c430b.jpg)

![91afd26a71881ffa70018c19eaf2c5d62b23d21b58c272b38d8ae25b374deb05.jpg](../iclr_results/2059_Decoding Game_ On Minimax Optimality of Heuristic Text Generation Strategies/tables/91afd26a71881ffa70018c19eaf2c5d62b23d21b58c272b38d8ae25b374deb05.jpg)

![93814c24ba2600c28d49a33d2e6ec33301e65745bc57d6c86b73fe817dfd3170.jpg](../iclr_results/2059_Decoding Game_ On Minimax Optimality of Heuristic Text Generation Strategies/tables/93814c24ba2600c28d49a33d2e6ec33301e65745bc57d6c86b73fe817dfd3170.jpg)

![a3136a0074528980bd3db19408805216f741da77e1f34c69e9ca45ca3e66c98d.jpg](../iclr_results/2059_Decoding Game_ On Minimax Optimality of Heuristic Text Generation Strategies/tables/a3136a0074528980bd3db19408805216f741da77e1f34c69e9ca45ca3e66c98d.jpg)

![d062ea82cf14a80271e8ce0f7fc8708b02210b5e93495b88c8fa207c808d9c68.jpg](../iclr_results/2059_Decoding Game_ On Minimax Optimality of Heuristic Text Generation Strategies/tables/d062ea82cf14a80271e8ce0f7fc8708b02210b5e93495b88c8fa207c808d9c68.jpg)

![fb7833b173bc6b8e08ecccbafda7b1af0fa1425b9976ab4a3697cf4c891a86a6.jpg](../iclr_results/2059_Decoding Game_ On Minimax Optimality of Heuristic Text Generation Strategies/tables/fb7833b173bc6b8e08ecccbafda7b1af0fa1425b9976ab4a3697cf4c891a86a6.jpg)

## I-Con: A Unifying Framework for Representation Learning


### Images

![0498a65e906623c4b83ef7987fa375595bacd1fc9fbfc156aa52880beb1d689e.jpg](../iclr_results/2060_I-Con_ A Unifying Framework for Representation Learning/images/0498a65e906623c4b83ef7987fa375595bacd1fc9fbfc156aa52880beb1d689e.jpg)

![1762c59fbf42a1c0588f4b43c6ca33521e22224648fe0bc76332f9de6ef016d5.jpg](../iclr_results/2060_I-Con_ A Unifying Framework for Representation Learning/images/1762c59fbf42a1c0588f4b43c6ca33521e22224648fe0bc76332f9de6ef016d5.jpg)

![625b9e1f95634d140c4cdc96c9555f467f30bcbdcf0c203511282f70e60387c4.jpg](../iclr_results/2060_I-Con_ A Unifying Framework for Representation Learning/images/625b9e1f95634d140c4cdc96c9555f467f30bcbdcf0c203511282f70e60387c4.jpg)

![71ec033e76fc52aeaf14a1845f2c2478e08406249ecf9fe0df8413cb10b3258a.jpg](../iclr_results/2060_I-Con_ A Unifying Framework for Representation Learning/images/71ec033e76fc52aeaf14a1845f2c2478e08406249ecf9fe0df8413cb10b3258a.jpg)

![bcaf43263e0140470589b5bf87ba299271a635d104a7e6dc5b8a3f6faba18ace.jpg](../iclr_results/2060_I-Con_ A Unifying Framework for Representation Learning/images/bcaf43263e0140470589b5bf87ba299271a635d104a7e6dc5b8a3f6faba18ace.jpg)

![c718ab63a4600cb4e4b894bd933545dde33b62190fb4c3c75c0e3309ade094bf.jpg](../iclr_results/2060_I-Con_ A Unifying Framework for Representation Learning/images/c718ab63a4600cb4e4b894bd933545dde33b62190fb4c3c75c0e3309ade094bf.jpg)

![df84214deff4310fb2bc836d659843dccdb581de4c4d380cbf8d18e5be6aea6e.jpg](../iclr_results/2060_I-Con_ A Unifying Framework for Representation Learning/images/df84214deff4310fb2bc836d659843dccdb581de4c4d380cbf8d18e5be6aea6e.jpg)

![ee769daac3a93e5c3e27051a718d4554ce792d5ea6caa2d72b6f023ff3954ea2.jpg](../iclr_results/2060_I-Con_ A Unifying Framework for Representation Learning/images/ee769daac3a93e5c3e27051a718d4554ce792d5ea6caa2d72b6f023ff3954ea2.jpg)

![f0adca6757ad8248d04b07b446458ab0228ecd2ffafe7edaeab6e9173c86247c.jpg](../iclr_results/2060_I-Con_ A Unifying Framework for Representation Learning/images/f0adca6757ad8248d04b07b446458ab0228ecd2ffafe7edaeab6e9173c86247c.jpg)

### Tables

![0f99bc75ac89dd03c5faa124dd8b28176cfcad563cf88771e4707c3c4f378196.jpg](../iclr_results/2060_I-Con_ A Unifying Framework for Representation Learning/tables/0f99bc75ac89dd03c5faa124dd8b28176cfcad563cf88771e4707c3c4f378196.jpg)

![667b96f33c131c67f5e7ef1c3a90bf12e707dc72fd48a6c31bbd2480b5e29c8c.jpg](../iclr_results/2060_I-Con_ A Unifying Framework for Representation Learning/tables/667b96f33c131c67f5e7ef1c3a90bf12e707dc72fd48a6c31bbd2480b5e29c8c.jpg)

![7762fea087a05b3053aeb762db4112107b516b56b8422284fc352679ceb4b68a.jpg](../iclr_results/2060_I-Con_ A Unifying Framework for Representation Learning/tables/7762fea087a05b3053aeb762db4112107b516b56b8422284fc352679ceb4b68a.jpg)

![b858cecaeb49129a3c1c03dfe8c7bf42dd71592248a797ebad3e0af7998c8504.jpg](../iclr_results/2060_I-Con_ A Unifying Framework for Representation Learning/tables/b858cecaeb49129a3c1c03dfe8c7bf42dd71592248a797ebad3e0af7998c8504.jpg)

![bb3161ce01928c4b72a6371b6da89d86d38d796cb3a3e9ffa63ba3a53d4e3453.jpg](../iclr_results/2060_I-Con_ A Unifying Framework for Representation Learning/tables/bb3161ce01928c4b72a6371b6da89d86d38d796cb3a3e9ffa63ba3a53d4e3453.jpg)

![f04417685f86c0231412f4dc57c780c5cbf2d5c416cdaa1a0a70a86852ad580e.jpg](../iclr_results/2060_I-Con_ A Unifying Framework for Representation Learning/tables/f04417685f86c0231412f4dc57c780c5cbf2d5c416cdaa1a0a70a86852ad580e.jpg)

## On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback


### Images

![0b72b8d1f28b4c1157cd8cc5a74a59cd721053f3bd00d09748a7594141e424ad.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/0b72b8d1f28b4c1157cd8cc5a74a59cd721053f3bd00d09748a7594141e424ad.jpg)

![27aae298b7001790b70c8d60117543dfce9a62c613e8ff26b5617bdfdfb9d00a.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/27aae298b7001790b70c8d60117543dfce9a62c613e8ff26b5617bdfdfb9d00a.jpg)

![2925fa3b47feaa704929a7fa600b621fbb54ef81322a0ab48b5e6fbad535ad8c.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/2925fa3b47feaa704929a7fa600b621fbb54ef81322a0ab48b5e6fbad535ad8c.jpg)

![416cfa231ec24cc91ba8cef02eea14bdebb360b00f9f5a04e74acda21ddc136e.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/416cfa231ec24cc91ba8cef02eea14bdebb360b00f9f5a04e74acda21ddc136e.jpg)

![46620e1b83fc3f5d37c153271cbcca3dd8148efeddafd0a3e8eff5bbec2d591d.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/46620e1b83fc3f5d37c153271cbcca3dd8148efeddafd0a3e8eff5bbec2d591d.jpg)

![46a6d810eed7b776aefa2a3ed4be95a53a622ec50c1a6376fe97118b2989e29b.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/46a6d810eed7b776aefa2a3ed4be95a53a622ec50c1a6376fe97118b2989e29b.jpg)

![4c2707cf18c822e27fb9076c789134deed98f0758aa535b6819fdc8aceac1859.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/4c2707cf18c822e27fb9076c789134deed98f0758aa535b6819fdc8aceac1859.jpg)

![4ddfc5a9042032bb593cb36ba4b5ff883dee9e769f5994332613c92e27bf33fe.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/4ddfc5a9042032bb593cb36ba4b5ff883dee9e769f5994332613c92e27bf33fe.jpg)

![525a02d8c9853e2bdda2f453159996f7f3533ab15be6285bbd7cab58fdde897a.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/525a02d8c9853e2bdda2f453159996f7f3533ab15be6285bbd7cab58fdde897a.jpg)

![53fa1ae9762b60079c85d0ffbec92c396f4df4f6272f40199c212f00448e895c.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/53fa1ae9762b60079c85d0ffbec92c396f4df4f6272f40199c212f00448e895c.jpg)

![58f8d4c2e790954619cb4ebea25384955b9b6904265d310251984850f6c5f2b8.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/58f8d4c2e790954619cb4ebea25384955b9b6904265d310251984850f6c5f2b8.jpg)

![60f8ebf1d6ebf6623cf383fa6f7e3060689f4483402c6e3f3d5c8b89d03ed8b0.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/60f8ebf1d6ebf6623cf383fa6f7e3060689f4483402c6e3f3d5c8b89d03ed8b0.jpg)

![7198a28656f98606824c270f52c1db83e8e039d1cd09ccaaad6d69b7db17bd60.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/7198a28656f98606824c270f52c1db83e8e039d1cd09ccaaad6d69b7db17bd60.jpg)

![73e543d7ec309807d34f57041533bc7240339e7a02cb08e85bbd0f1ecc715421.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/73e543d7ec309807d34f57041533bc7240339e7a02cb08e85bbd0f1ecc715421.jpg)

![76edb59bbac3f2128c6570ff8ba9af0107fd94b4a94dbd61065e7c29a285ddd5.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/76edb59bbac3f2128c6570ff8ba9af0107fd94b4a94dbd61065e7c29a285ddd5.jpg)

![7cfe12a7efdcca2f5a49106ac6621ba625f79859f0cbf0bb062d50f428bcd06d.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/7cfe12a7efdcca2f5a49106ac6621ba625f79859f0cbf0bb062d50f428bcd06d.jpg)

![861d239d9514c6fb9682836171df976e822a9b3b568a28a4dc989fdb5ee320de.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/861d239d9514c6fb9682836171df976e822a9b3b568a28a4dc989fdb5ee320de.jpg)

![8cf2c727f8c0cfe004cb4239b8539b93d64a456add5ef6bab11b88903ff95146.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/8cf2c727f8c0cfe004cb4239b8539b93d64a456add5ef6bab11b88903ff95146.jpg)

![ad952158adfdbce34a3d2640fa8b337b396fb8a27664d0480d179eae92841d0b.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/ad952158adfdbce34a3d2640fa8b337b396fb8a27664d0480d179eae92841d0b.jpg)

![b51303d4800ee89b5ec00d99e1264a1ca516a4d793f61290cb3fed16b230c3e9.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/b51303d4800ee89b5ec00d99e1264a1ca516a4d793f61290cb3fed16b230c3e9.jpg)

![b9660d3a61c50a7a6f4e381ce751f5179646dd81ca78d689bdea8796f12fc9de.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/b9660d3a61c50a7a6f4e381ce751f5179646dd81ca78d689bdea8796f12fc9de.jpg)

![bcd6c7f7e23877ee277bf2d0d0077edc20059b52a8d905d25c09f24d16f39471.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/bcd6c7f7e23877ee277bf2d0d0077edc20059b52a8d905d25c09f24d16f39471.jpg)

![beb67332c402d5a2462f9b97dc0414b42380d612082b549a5d69fd96d13b2054.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/beb67332c402d5a2462f9b97dc0414b42380d612082b549a5d69fd96d13b2054.jpg)

![c4b0b0338c8f47b6046713c4b37ad77a761b85f7dde661533ff68e3ad26d4b91.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/c4b0b0338c8f47b6046713c4b37ad77a761b85f7dde661533ff68e3ad26d4b91.jpg)

![e1f56e42e49cf9d762f8dde7d468b5fa954c66a571ea06c8d44ecc23e2a34ca9.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/e1f56e42e49cf9d762f8dde7d468b5fa954c66a571ea06c8d44ecc23e2a34ca9.jpg)

![f09b928800641d36ff7ffdb9341c77ce4d8e43513fe0043a9a1ff2b50ec71836.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/f09b928800641d36ff7ffdb9341c77ce4d8e43513fe0043a9a1ff2b50ec71836.jpg)

![f26c360e4a30a703fb47af1036daf490eedf068c35f2a4d8ce40ca090d8d3160.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/f26c360e4a30a703fb47af1036daf490eedf068c35f2a4d8ce40ca090d8d3160.jpg)

![f3b360b46d6ba0364b9a68aaf1a48488be546d73968b229644c13a5427943f75.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/images/f3b360b46d6ba0364b9a68aaf1a48488be546d73968b229644c13a5427943f75.jpg)

### Tables

![05af9f9592eb2cedb826c3d2f5951c0fdbcf5bbdfbbc71e1a416eb97dc8a20e5.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/tables/05af9f9592eb2cedb826c3d2f5951c0fdbcf5bbdfbbc71e1a416eb97dc8a20e5.jpg)

![2b9f5f4bc192cef9377705281dc24128e8ac7c2af99bca50003a5b0d896c065a.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/tables/2b9f5f4bc192cef9377705281dc24128e8ac7c2af99bca50003a5b0d896c065a.jpg)

![3ebec8c4c4e6e3860ddaae5745747f87d38d6cf3062a913afd9c78f849fb62d3.jpg](../iclr_results/2061_On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback/tables/3ebec8c4c4e6e3860ddaae5745747f87d38d6cf3062a913afd9c78f849fb62d3.jpg)

## A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation


### Images

![161d641940d4c93853252d5fd68043e6dc84e112ec311b293264ebd2a29c4ca6.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/images/161d641940d4c93853252d5fd68043e6dc84e112ec311b293264ebd2a29c4ca6.jpg)

![321c3d1a58e0712f79b848cf2703b27daee648be78e040755e5e11254c91310d.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/images/321c3d1a58e0712f79b848cf2703b27daee648be78e040755e5e11254c91310d.jpg)

![4bb1475ee41991f1e29d44ea66625dad61731b916552a9ac83df058cbceaa079.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/images/4bb1475ee41991f1e29d44ea66625dad61731b916552a9ac83df058cbceaa079.jpg)

![59afea612071af67ef31c3d449df7815de97d80bbf8f7ebc282e1127f0244b40.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/images/59afea612071af67ef31c3d449df7815de97d80bbf8f7ebc282e1127f0244b40.jpg)

![5a7955e01ebc6beaa4c13937bdd413e2128f3bb4d373e93f766642fa3ec4c299.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/images/5a7955e01ebc6beaa4c13937bdd413e2128f3bb4d373e93f766642fa3ec4c299.jpg)

![6e507641737308112868c473f8f62ec4100cf7af3c10966584826ce83ae41d1e.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/images/6e507641737308112868c473f8f62ec4100cf7af3c10966584826ce83ae41d1e.jpg)

![6eac2fb9bd55be4fa9ddc3a42153633c5c6f05a6237a024cb428a8f462e23646.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/images/6eac2fb9bd55be4fa9ddc3a42153633c5c6f05a6237a024cb428a8f462e23646.jpg)

![7446ec0ad04f69d2c32c8eacb2b6e0750b0b515e113b6ac792189c42da45b13d.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/images/7446ec0ad04f69d2c32c8eacb2b6e0750b0b515e113b6ac792189c42da45b13d.jpg)

![7c015d14e8134a8b310c357df26d89ab872ec576de739737ec8438a27d94d09a.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/images/7c015d14e8134a8b310c357df26d89ab872ec576de739737ec8438a27d94d09a.jpg)

![7c0a6a288c5d3e7763bda1950516a1184c426dcc960cbafbfe904157637bbc4f.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/images/7c0a6a288c5d3e7763bda1950516a1184c426dcc960cbafbfe904157637bbc4f.jpg)

![8797de806213e6f073d6ccd358f32e3e98f7a84c9f11529afc914c22972a5068.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/images/8797de806213e6f073d6ccd358f32e3e98f7a84c9f11529afc914c22972a5068.jpg)

![ad3e409179b75731a47a5d8a5880054bb80159f36aad7936418768876199bb8e.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/images/ad3e409179b75731a47a5d8a5880054bb80159f36aad7936418768876199bb8e.jpg)

![b1df3bce073b7e878835cbb38d1ace9efac500aec03b145838b1f67454c02521.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/images/b1df3bce073b7e878835cbb38d1ace9efac500aec03b145838b1f67454c02521.jpg)

![bc69669383eb6aef65d7bb11edb2fafba53e8f0ba0ebf230159d5f80bdd2e55c.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/images/bc69669383eb6aef65d7bb11edb2fafba53e8f0ba0ebf230159d5f80bdd2e55c.jpg)

![ca37dd0bc5e7b2262da4628c6b9bafbe16b5bafb60d04e1c888996e6aed77177.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/images/ca37dd0bc5e7b2262da4628c6b9bafbe16b5bafb60d04e1c888996e6aed77177.jpg)

![d8961069adbdcfe94e1edb5781ffd2f9e4978cc0ba9394f22f6abdc94798efa6.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/images/d8961069adbdcfe94e1edb5781ffd2f9e4978cc0ba9394f22f6abdc94798efa6.jpg)

![deeb16ac6dbb88432ba32e65b9a60a824db1f2643297413c3d161b2b9199aec0.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/images/deeb16ac6dbb88432ba32e65b9a60a824db1f2643297413c3d161b2b9199aec0.jpg)

![e5b41c7f417bc937f5bd531760fd95e27538e8c0e66802ba8cce2a1548a2e50a.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/images/e5b41c7f417bc937f5bd531760fd95e27538e8c0e66802ba8cce2a1548a2e50a.jpg)

![fa845da26308ea9fe64704518cefaa41cfbcc4e947669ebd703302ebea1e63d9.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/images/fa845da26308ea9fe64704518cefaa41cfbcc4e947669ebd703302ebea1e63d9.jpg)

### Tables

![0b3414f33ff98119f16a359a7d9aad98d612b271245a1ef4aeb20b8632c1f38d.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/tables/0b3414f33ff98119f16a359a7d9aad98d612b271245a1ef4aeb20b8632c1f38d.jpg)

![0f97153747b5a03d84810706461f8099b07808833c7c079c2988c04e135b29ba.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/tables/0f97153747b5a03d84810706461f8099b07808833c7c079c2988c04e135b29ba.jpg)

![4350e05277e5a998ecc70242ffba9d481658907c0354946a34148e25b8a83d43.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/tables/4350e05277e5a998ecc70242ffba9d481658907c0354946a34148e25b8a83d43.jpg)

![ab03060b360e713d271e3c13cd5df0c73eb736d78a47d50d0558c65a3bfa3b84.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/tables/ab03060b360e713d271e3c13cd5df0c73eb736d78a47d50d0558c65a3bfa3b84.jpg)

![da241cc4e8de8a354e062887945d5835680b31556a6ceffda21cf986d13e4ab0.jpg](../iclr_results/2062_A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation/tables/da241cc4e8de8a354e062887945d5835680b31556a6ceffda21cf986d13e4ab0.jpg)

## Stable Hadamard Memory: Revitalizing Memory-Augmented Agents for Reinforcement Learning


### Images

![507f0bb818b6453e2af44a40bc934f9d3aa2520406ecccde1e10bb0b7483e70a.jpg](../iclr_results/2063_Stable Hadamard Memory_ Revitalizing Memory-Augmented Agents for Reinforcement Learning/images/507f0bb818b6453e2af44a40bc934f9d3aa2520406ecccde1e10bb0b7483e70a.jpg)

![86a9f7a8a4b87da9149e17938cb9ff6033fb5a8af736f4f5115763f8b173c7aa.jpg](../iclr_results/2063_Stable Hadamard Memory_ Revitalizing Memory-Augmented Agents for Reinforcement Learning/images/86a9f7a8a4b87da9149e17938cb9ff6033fb5a8af736f4f5115763f8b173c7aa.jpg)

![f2a31c2f62b87f9649074e12b55a911443ed87fe1da5eca60b632eb9b5e3d38f.jpg](../iclr_results/2063_Stable Hadamard Memory_ Revitalizing Memory-Augmented Agents for Reinforcement Learning/images/f2a31c2f62b87f9649074e12b55a911443ed87fe1da5eca60b632eb9b5e3d38f.jpg)

![f739e061b018ca3c06416efd81ce7e531e14d678645499e7ce71d87790fa8ddb.jpg](../iclr_results/2063_Stable Hadamard Memory_ Revitalizing Memory-Augmented Agents for Reinforcement Learning/images/f739e061b018ca3c06416efd81ce7e531e14d678645499e7ce71d87790fa8ddb.jpg)

### Tables

![29064907605362a0c1014e1b35481073442f7a8c1f8a07f9810fae1296ba4abf.jpg](../iclr_results/2063_Stable Hadamard Memory_ Revitalizing Memory-Augmented Agents for Reinforcement Learning/tables/29064907605362a0c1014e1b35481073442f7a8c1f8a07f9810fae1296ba4abf.jpg)

![3333e2f25d85e50157952f63549e1fd373c962cb7245b859bbbc9a2cdc2b586d.jpg](../iclr_results/2063_Stable Hadamard Memory_ Revitalizing Memory-Augmented Agents for Reinforcement Learning/tables/3333e2f25d85e50157952f63549e1fd373c962cb7245b859bbbc9a2cdc2b586d.jpg)

![88588e9a813ce8fd33139f27be306f73680bdc9ffd1fb59ecbe8dc0064b86904.jpg](../iclr_results/2063_Stable Hadamard Memory_ Revitalizing Memory-Augmented Agents for Reinforcement Learning/tables/88588e9a813ce8fd33139f27be306f73680bdc9ffd1fb59ecbe8dc0064b86904.jpg)

![9ad621408a03b27bb5ebcbbf903838de7cf44c87e03185d5617382bdf38473bb.jpg](../iclr_results/2063_Stable Hadamard Memory_ Revitalizing Memory-Augmented Agents for Reinforcement Learning/tables/9ad621408a03b27bb5ebcbbf903838de7cf44c87e03185d5617382bdf38473bb.jpg)

![c897363fa5c1e720e218d280b0bba5afdb55c49eb5b28c8ef2daf802773d7380.jpg](../iclr_results/2063_Stable Hadamard Memory_ Revitalizing Memory-Augmented Agents for Reinforcement Learning/tables/c897363fa5c1e720e218d280b0bba5afdb55c49eb5b28c8ef2daf802773d7380.jpg)

![d46ca60b0cd11f0f595f2bf71c3d3de17c9ec0032c6317cfa0df70f23eb56e06.jpg](../iclr_results/2063_Stable Hadamard Memory_ Revitalizing Memory-Augmented Agents for Reinforcement Learning/tables/d46ca60b0cd11f0f595f2bf71c3d3de17c9ec0032c6317cfa0df70f23eb56e06.jpg)

![f3314b97598a374fd39db984e8a617cb7fe356f3eefa46fad4f5418b938b85f0.jpg](../iclr_results/2063_Stable Hadamard Memory_ Revitalizing Memory-Augmented Agents for Reinforcement Learning/tables/f3314b97598a374fd39db984e8a617cb7fe356f3eefa46fad4f5418b938b85f0.jpg)

## Breaking Free from MMI: A New Frontier in Rationalization by Probing Input Utilization


### Images

![6d5c264cdbad7eb002c41f11d9e47ac68d445903f0a37ad7f1f236af6c7a5cb2.jpg](../iclr_results/2064_Breaking Free from MMI_ A New Frontier in Rationalization by Probing Input Utilization/images/6d5c264cdbad7eb002c41f11d9e47ac68d445903f0a37ad7f1f236af6c7a5cb2.jpg)

![9ef8fe00af75355c6331a7d352d1219e3b5f4899699af7189667bacd6518ed1a.jpg](../iclr_results/2064_Breaking Free from MMI_ A New Frontier in Rationalization by Probing Input Utilization/images/9ef8fe00af75355c6331a7d352d1219e3b5f4899699af7189667bacd6518ed1a.jpg)

![a9620e01dda0ae5d47dfc4bb3de015bca093db9eaa93af969f18f7861dca2508.jpg](../iclr_results/2064_Breaking Free from MMI_ A New Frontier in Rationalization by Probing Input Utilization/images/a9620e01dda0ae5d47dfc4bb3de015bca093db9eaa93af969f18f7861dca2508.jpg)

![c7d3ada225df6e2d9f98d0223b04ba6cbfb847799fff80f8454d921b84e3a10a.jpg](../iclr_results/2064_Breaking Free from MMI_ A New Frontier in Rationalization by Probing Input Utilization/images/c7d3ada225df6e2d9f98d0223b04ba6cbfb847799fff80f8454d921b84e3a10a.jpg)

![d27ef17b751beac576c4f69fdc30b5b98de4c7733f9bd436ae5ca94ed69481bc.jpg](../iclr_results/2064_Breaking Free from MMI_ A New Frontier in Rationalization by Probing Input Utilization/images/d27ef17b751beac576c4f69fdc30b5b98de4c7733f9bd436ae5ca94ed69481bc.jpg)

![d69f2a1ce5b7a73677c66ec621ef1f03f2df2d9e85fd238479ee4daf77191221.jpg](../iclr_results/2064_Breaking Free from MMI_ A New Frontier in Rationalization by Probing Input Utilization/images/d69f2a1ce5b7a73677c66ec621ef1f03f2df2d9e85fd238479ee4daf77191221.jpg)

![eb98617d1bf263ff2889988ca30b88c1e3c27390976677bde03310c72691836c.jpg](../iclr_results/2064_Breaking Free from MMI_ A New Frontier in Rationalization by Probing Input Utilization/images/eb98617d1bf263ff2889988ca30b88c1e3c27390976677bde03310c72691836c.jpg)

![ed889500a49f389e8c8e39f1c6676147a2b07717308cd170f974ed51049a3998.jpg](../iclr_results/2064_Breaking Free from MMI_ A New Frontier in Rationalization by Probing Input Utilization/images/ed889500a49f389e8c8e39f1c6676147a2b07717308cd170f974ed51049a3998.jpg)

![f6f0d74bad3d62282547254d941b7fee0c04201db4255f5f949ac2775dae534d.jpg](../iclr_results/2064_Breaking Free from MMI_ A New Frontier in Rationalization by Probing Input Utilization/images/f6f0d74bad3d62282547254d941b7fee0c04201db4255f5f949ac2775dae534d.jpg)

### Tables

![1a293227f6f00ca4a8a5d91588f50d89e9d6638d8b8d8d11d9e1bbcf137e03d9.jpg](../iclr_results/2064_Breaking Free from MMI_ A New Frontier in Rationalization by Probing Input Utilization/tables/1a293227f6f00ca4a8a5d91588f50d89e9d6638d8b8d8d11d9e1bbcf137e03d9.jpg)

![25f3d77a95e5eb03ff67f31fab5b775eaac6a1bfd53437d400bc2143c61ab1d3.jpg](../iclr_results/2064_Breaking Free from MMI_ A New Frontier in Rationalization by Probing Input Utilization/tables/25f3d77a95e5eb03ff67f31fab5b775eaac6a1bfd53437d400bc2143c61ab1d3.jpg)

![ab17891c5bf4a16c19585027c07d11157c83242a7d638840b786ba7ebf119f42.jpg](../iclr_results/2064_Breaking Free from MMI_ A New Frontier in Rationalization by Probing Input Utilization/tables/ab17891c5bf4a16c19585027c07d11157c83242a7d638840b786ba7ebf119f42.jpg)

![b4244dc6a25baf65f0e2cab811350ea632f9efd942ea84d6f29217b00d317832.jpg](../iclr_results/2064_Breaking Free from MMI_ A New Frontier in Rationalization by Probing Input Utilization/tables/b4244dc6a25baf65f0e2cab811350ea632f9efd942ea84d6f29217b00d317832.jpg)

![b92f82a47317f86eca82ed1a8a8508a56ceee98cc9d0630271f422f0b48d5350.jpg](../iclr_results/2064_Breaking Free from MMI_ A New Frontier in Rationalization by Probing Input Utilization/tables/b92f82a47317f86eca82ed1a8a8508a56ceee98cc9d0630271f422f0b48d5350.jpg)

![b9d90d3c38e516de96c8b332a3ae1013d5b436130100c07da3b68467dd2b6a8b.jpg](../iclr_results/2064_Breaking Free from MMI_ A New Frontier in Rationalization by Probing Input Utilization/tables/b9d90d3c38e516de96c8b332a3ae1013d5b436130100c07da3b68467dd2b6a8b.jpg)

![da708e074088fbe20b1c102e4221622d4779f9d64f6d04de830ec3d423bd510f.jpg](../iclr_results/2064_Breaking Free from MMI_ A New Frontier in Rationalization by Probing Input Utilization/tables/da708e074088fbe20b1c102e4221622d4779f9d64f6d04de830ec3d423bd510f.jpg)

![e14b6183dc173252a4ebec47e8de276207da6a3e9b02b4751246b57ba3a9c5e9.jpg](../iclr_results/2064_Breaking Free from MMI_ A New Frontier in Rationalization by Probing Input Utilization/tables/e14b6183dc173252a4ebec47e8de276207da6a3e9b02b4751246b57ba3a9c5e9.jpg)

![f6a60e802521e276a0df0814d676c74a56a013d1042b70fb40f17fabb070253f.jpg](../iclr_results/2064_Breaking Free from MMI_ A New Frontier in Rationalization by Probing Input Utilization/tables/f6a60e802521e276a0df0814d676c74a56a013d1042b70fb40f17fabb070253f.jpg)

![fe5fd9434b72ac4c61e7de94cb105a50d97290dcf3856f83a1d57bfd85c33ca0.jpg](../iclr_results/2064_Breaking Free from MMI_ A New Frontier in Rationalization by Probing Input Utilization/tables/fe5fd9434b72ac4c61e7de94cb105a50d97290dcf3856f83a1d57bfd85c33ca0.jpg)

## Scaling Optimal LR Across Token Horizons


### Images

![06627e93482fe543a05bd2b08fa03419ac1cef0ff12de9e422c9e2a522ad84d3.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/images/06627e93482fe543a05bd2b08fa03419ac1cef0ff12de9e422c9e2a522ad84d3.jpg)

![2debaa1f60461daac40c174389bfb7a53e4d069e7020d4b50a73123cf84c29f8.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/images/2debaa1f60461daac40c174389bfb7a53e4d069e7020d4b50a73123cf84c29f8.jpg)

![35e7945b122d5062334fdc4e11106c808772dd973b2a032834f1dcbe6059e734.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/images/35e7945b122d5062334fdc4e11106c808772dd973b2a032834f1dcbe6059e734.jpg)

![35fc36cb4387a1a3ce4a3732c4c52000a663006821b34b423a38ad9f0b3df77e.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/images/35fc36cb4387a1a3ce4a3732c4c52000a663006821b34b423a38ad9f0b3df77e.jpg)

![479bc3f53af177de25e792f83c58657259e86078de102a28976be48b2c04acf9.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/images/479bc3f53af177de25e792f83c58657259e86078de102a28976be48b2c04acf9.jpg)

![518ceec2e471e6a0001b810c5559c516e4464b7aa7b853ffe3b7c82303df0cac.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/images/518ceec2e471e6a0001b810c5559c516e4464b7aa7b853ffe3b7c82303df0cac.jpg)

![51c29dfacd1532ce511f198b4797a092bfeae7cd24b1b17ffeaa1ff83554472f.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/images/51c29dfacd1532ce511f198b4797a092bfeae7cd24b1b17ffeaa1ff83554472f.jpg)

![7d253cb344d0e1d40c3577974c76b08c973e456ad191340857b55bd9389f4edb.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/images/7d253cb344d0e1d40c3577974c76b08c973e456ad191340857b55bd9389f4edb.jpg)

![ad1d63b4b0d30f1c27418bb0eb87dc2a4cf4cbdd2d035eb5b4a1f1a6444cd727.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/images/ad1d63b4b0d30f1c27418bb0eb87dc2a4cf4cbdd2d035eb5b4a1f1a6444cd727.jpg)

![c3a1fd631e6b0943382dd34613a7bdc56686d0f84c008ed9f533de9bdda673e8.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/images/c3a1fd631e6b0943382dd34613a7bdc56686d0f84c008ed9f533de9bdda673e8.jpg)

![d00dcf10e2cae338056a888dc4befda54d5d963936c33eee1518b0f55e060587.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/images/d00dcf10e2cae338056a888dc4befda54d5d963936c33eee1518b0f55e060587.jpg)

![e54e9700ce75efc1edf754b641c5c62dd1a758ef7da38d3174346c8780bd9ffe.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/images/e54e9700ce75efc1edf754b641c5c62dd1a758ef7da38d3174346c8780bd9ffe.jpg)

![e592745178440990a919a8dd93fe7d4944a99af4aabc20bad90e86598a7bb5b1.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/images/e592745178440990a919a8dd93fe7d4944a99af4aabc20bad90e86598a7bb5b1.jpg)

![ed8f730121d389adc70817ebd1d15163ef5aa0845beafaa42dae4ab49b7817fb.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/images/ed8f730121d389adc70817ebd1d15163ef5aa0845beafaa42dae4ab49b7817fb.jpg)

![fd2105d4873fdf29fe055608ee908bcef893f7569a4f11620b6b1f900c3bcc2c.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/images/fd2105d4873fdf29fe055608ee908bcef893f7569a4f11620b6b1f900c3bcc2c.jpg)

### Tables

![046ee05870f7e56297a108df5c93d6d9b2adb71d550f66e7c3b39e6053766cfc.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/tables/046ee05870f7e56297a108df5c93d6d9b2adb71d550f66e7c3b39e6053766cfc.jpg)

![27d1fbf0341d40312f19d104b0540a22b572f3d37290e7b079f912d223037621.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/tables/27d1fbf0341d40312f19d104b0540a22b572f3d37290e7b079f912d223037621.jpg)

![49bc451a06e2bab44fe459f2e42867b1f78024b373a5a7ca0b11e59c8523fe6b.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/tables/49bc451a06e2bab44fe459f2e42867b1f78024b373a5a7ca0b11e59c8523fe6b.jpg)

![6e90e2e5847e9055ed9a5b633a6f869ec0d40eaede5db7f577c643e11ba835a4.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/tables/6e90e2e5847e9055ed9a5b633a6f869ec0d40eaede5db7f577c643e11ba835a4.jpg)

![91158a36aa259d8e9ef4d26f62f896b63c1c0c4646e253684875508abf14bf74.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/tables/91158a36aa259d8e9ef4d26f62f896b63c1c0c4646e253684875508abf14bf74.jpg)

![bcd528995c319d601976e98dc55bd96ba816ea147d9767f5dd50b6ae481a4ad3.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/tables/bcd528995c319d601976e98dc55bd96ba816ea147d9767f5dd50b6ae481a4ad3.jpg)

![beb27c6b6f2a7de910a9a01577dfe971b88b920094655303337850e8ae74f55d.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/tables/beb27c6b6f2a7de910a9a01577dfe971b88b920094655303337850e8ae74f55d.jpg)

![daa4bd1caae1058a2bef9732213ba8f10ab1a5a72078783ed65e537fb68c7d73.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/tables/daa4bd1caae1058a2bef9732213ba8f10ab1a5a72078783ed65e537fb68c7d73.jpg)

![f40aeaf1617e302eca77127ff2dfcae48875e092c10afe21f2e85584d64eef9b.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/tables/f40aeaf1617e302eca77127ff2dfcae48875e092c10afe21f2e85584d64eef9b.jpg)

![f9be5182ff0b9659056d5308e75f6f6f1e91e304bf6868fdd382cf5e8e7f4bb5.jpg](../iclr_results/2065_Scaling Optimal LR Across Token Horizons/tables/f9be5182ff0b9659056d5308e75f6f6f1e91e304bf6868fdd382cf5e8e7f4bb5.jpg)

## MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions


### Images

![06cd59780f46154e025546d279d9a87651a91241928ed880fe08e9fc3df1c88f.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/06cd59780f46154e025546d279d9a87651a91241928ed880fe08e9fc3df1c88f.jpg)

![0701f24c2912de6a67e543742fae3ab2d008d26f8fbd84e8d49c15e0c1b8cb9d.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/0701f24c2912de6a67e543742fae3ab2d008d26f8fbd84e8d49c15e0c1b8cb9d.jpg)

![10ca43ba66832196cafb9aa2419b35c657dd189663dc4c32123ed46a91b21963.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/10ca43ba66832196cafb9aa2419b35c657dd189663dc4c32123ed46a91b21963.jpg)

![2040d1969e5e66579df6e269f4a242b13441baf0e16c6d2a2f7af7cefc701bd9.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/2040d1969e5e66579df6e269f4a242b13441baf0e16c6d2a2f7af7cefc701bd9.jpg)

![26c96c27713a75845f60b9360098ad66f31c7ecd5817cdcbc195066c4456089a.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/26c96c27713a75845f60b9360098ad66f31c7ecd5817cdcbc195066c4456089a.jpg)

![3606ea313f15d47f6a331d4f6fa53e309153a8ee434de4f3bc78b5e2d529e0b2.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/3606ea313f15d47f6a331d4f6fa53e309153a8ee434de4f3bc78b5e2d529e0b2.jpg)

![438da348876e947db11338aa5eef23648168dca7678a322b667f2e60cd05aaba.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/438da348876e947db11338aa5eef23648168dca7678a322b667f2e60cd05aaba.jpg)

![4e2ed619d1ef1fd3c6897ec724dec0333a6ead864afb554ebb14b87299e22671.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/4e2ed619d1ef1fd3c6897ec724dec0333a6ead864afb554ebb14b87299e22671.jpg)

![526cd6d5d4e1c77056806fc03d3d9ab1a7fb5d86b4c8b4920cd162755a997213.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/526cd6d5d4e1c77056806fc03d3d9ab1a7fb5d86b4c8b4920cd162755a997213.jpg)

![52ef2a93862c54b40553aac4220342329cc8b1af98d7ea4f78cf8cae278ed60f.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/52ef2a93862c54b40553aac4220342329cc8b1af98d7ea4f78cf8cae278ed60f.jpg)

![65380adaa68a3e28d61f53203685d1e9c170b723035d4f28150e3b03e4f14f55.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/65380adaa68a3e28d61f53203685d1e9c170b723035d4f28150e3b03e4f14f55.jpg)

![6774facac33a32370ecb817dad3d1a0e1c93cf2a005ef81e32dc059a036bf871.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/6774facac33a32370ecb817dad3d1a0e1c93cf2a005ef81e32dc059a036bf871.jpg)

![691d8f7ce76757d3e22080e4ba71606c4de2b8a1b6da528b89b4f6e3ebb8c9eb.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/691d8f7ce76757d3e22080e4ba71606c4de2b8a1b6da528b89b4f6e3ebb8c9eb.jpg)

![6da557f0fc6474182d7e0c2e2cbf08626cf4cf520eaca338d70cef63e43e59cd.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/6da557f0fc6474182d7e0c2e2cbf08626cf4cf520eaca338d70cef63e43e59cd.jpg)

![76bd0269b983fec3d9a8b9c320de1ede5ca84cdbfe149e0ad5db0affa742b25c.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/76bd0269b983fec3d9a8b9c320de1ede5ca84cdbfe149e0ad5db0affa742b25c.jpg)

![803e5ee7c7de1dbb62ffba849bb0c9fffcf9a7005df2fa8e63076be1c0b72146.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/803e5ee7c7de1dbb62ffba849bb0c9fffcf9a7005df2fa8e63076be1c0b72146.jpg)

![87b23c330b0b6d05ec84c98e3615b2b8e07a137e75a04e399d14f9562191752e.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/87b23c330b0b6d05ec84c98e3615b2b8e07a137e75a04e399d14f9562191752e.jpg)

![8c38f41b0a28e7616805f56294ec4da06f2dfd6d4a7383db7fd385ee62ea45ad.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/8c38f41b0a28e7616805f56294ec4da06f2dfd6d4a7383db7fd385ee62ea45ad.jpg)

![91e43fce85db88b856fc603ce0f6c20899300791ed6772c093304efdb3247ff9.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/91e43fce85db88b856fc603ce0f6c20899300791ed6772c093304efdb3247ff9.jpg)

![99fec349d9dd382e7eb89e362abf7bc93c3af5a6763688e30c434941f1d28f67.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/99fec349d9dd382e7eb89e362abf7bc93c3af5a6763688e30c434941f1d28f67.jpg)

![abfaa12b025f193e172316500920e82af707e5aa4f978cadaf0c37a2cd133363.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/abfaa12b025f193e172316500920e82af707e5aa4f978cadaf0c37a2cd133363.jpg)

![c78a9f951bcf6c72b953d771c7264c6c35f9e65a9ebdcd870438d6b5bbf1e229.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/c78a9f951bcf6c72b953d771c7264c6c35f9e65a9ebdcd870438d6b5bbf1e229.jpg)

![ca13df19298191986c230e5b8ce8726b05871675c27fc8f36b1438de586e9262.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/ca13df19298191986c230e5b8ce8726b05871675c27fc8f36b1438de586e9262.jpg)

![dbc810a47e611784d2a16ccbdac3739c08769f6cb667d1552fa632dedee37c10.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/dbc810a47e611784d2a16ccbdac3739c08769f6cb667d1552fa632dedee37c10.jpg)

![e1ee622b235f0c0892cd191f3992d4f87fd424e6d071d7aa69c54f9ff77bbc9c.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/images/e1ee622b235f0c0892cd191f3992d4f87fd424e6d071d7aa69c54f9ff77bbc9c.jpg)

### Tables

![0247c7162bf9e374b939f349c2deeaa0e0ce962f77533283a2070c1dae6163c2.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/tables/0247c7162bf9e374b939f349c2deeaa0e0ce962f77533283a2070c1dae6163c2.jpg)

![12c16d87b0f73ed22f2623fb22cdcc0ee8ea66c7f31a79fb71832a48433e1516.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/tables/12c16d87b0f73ed22f2623fb22cdcc0ee8ea66c7f31a79fb71832a48433e1516.jpg)

![280f6c51b1986ec5bb327d2f5713c8e81e65d6590ad9d8a3fbe0562629739308.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/tables/280f6c51b1986ec5bb327d2f5713c8e81e65d6590ad9d8a3fbe0562629739308.jpg)

![330ac4c52aa5d90c278ea0ffd4e795d48a63322d29e2452e21a80bd5a9a91195.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/tables/330ac4c52aa5d90c278ea0ffd4e795d48a63322d29e2452e21a80bd5a9a91195.jpg)

![372dbaa3c35d7bf1d12d80c90c62e1468634e7f2653ec4c129a09530e8cff362.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/tables/372dbaa3c35d7bf1d12d80c90c62e1468634e7f2653ec4c129a09530e8cff362.jpg)

![4cedb8a54342b03db4b22d4e5491ce30ecdb91ef37d639217e6110399bd28898.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/tables/4cedb8a54342b03db4b22d4e5491ce30ecdb91ef37d639217e6110399bd28898.jpg)

![4f7f59817348c4f4b9a482626bab0e38727a7a2c9b053642436768a2e17fb89a.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/tables/4f7f59817348c4f4b9a482626bab0e38727a7a2c9b053642436768a2e17fb89a.jpg)

![6a1d5ba6cea1c7edc048f7413742ce108ea471b0afab0d9f0d23212ab45133ac.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/tables/6a1d5ba6cea1c7edc048f7413742ce108ea471b0afab0d9f0d23212ab45133ac.jpg)

![7c2551f329fca411710833bc8d99f8998d92def853ec0a6550e13b3daf7cee72.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/tables/7c2551f329fca411710833bc8d99f8998d92def853ec0a6550e13b3daf7cee72.jpg)

![857dfb0d0ab1c9f3fb6db30e4e430d7a880218dbd9dff6abd818ed73c90620e5.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/tables/857dfb0d0ab1c9f3fb6db30e4e430d7a880218dbd9dff6abd818ed73c90620e5.jpg)

![c8380b66c4039f762b7a86707a58ac748e60538cf1e85ad4ba1d25797872c368.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/tables/c8380b66c4039f762b7a86707a58ac748e60538cf1e85ad4ba1d25797872c368.jpg)

![d0360b8e136560f4fb0fa3d43dd8fed57afd67029879823641e3b6ef39989858.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/tables/d0360b8e136560f4fb0fa3d43dd8fed57afd67029879823641e3b6ef39989858.jpg)

![d85ac5549d66aa4d6708890741f10d17a3c36da03ce1d284b773ba93e3bf5a63.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/tables/d85ac5549d66aa4d6708890741f10d17a3c36da03ce1d284b773ba93e3bf5a63.jpg)

![dc5a3d02782dd6966fed90425b15c5f7df115128c5093a9d9ab91d59caa1447a.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/tables/dc5a3d02782dd6966fed90425b15c5f7df115128c5093a9d9ab91d59caa1447a.jpg)

![f35b7f7a9244301403de8d890875c39bc7cdc8ee1bbccefaec35e36103424721.jpg](../iclr_results/2066_MA-RLHF_ Reinforcement Learning from Human Feedback with Macro Actions/tables/f35b7f7a9244301403de8d890875c39bc7cdc8ee1bbccefaec35e36103424721.jpg)

## Towards Synergistic Path-based Explanations for Knowledge Graph Completion: Exploration and Evaluation


### Images

![0981835953612d75776fd776bb0399b8287151ffb24da3ffd2034b864f1b8101.jpg](../iclr_results/2067_Towards Synergistic Path-based Explanations for Knowledge Graph Completion_ Exploration and Evaluati/images/0981835953612d75776fd776bb0399b8287151ffb24da3ffd2034b864f1b8101.jpg)

![20751c1a49659c3ee74154b606c2bcfd3e2e810a960126e06ee7e97de21589e0.jpg](../iclr_results/2067_Towards Synergistic Path-based Explanations for Knowledge Graph Completion_ Exploration and Evaluati/images/20751c1a49659c3ee74154b606c2bcfd3e2e810a960126e06ee7e97de21589e0.jpg)

![33d77fb7b15f4b71d2cf43a4c64843a929da751b2299ac196dc7863108bdebaf.jpg](../iclr_results/2067_Towards Synergistic Path-based Explanations for Knowledge Graph Completion_ Exploration and Evaluati/images/33d77fb7b15f4b71d2cf43a4c64843a929da751b2299ac196dc7863108bdebaf.jpg)

![42bf68efd25c31287d241d0c61b8c8db640e919c6afcfb2af4ff337b68c20c11.jpg](../iclr_results/2067_Towards Synergistic Path-based Explanations for Knowledge Graph Completion_ Exploration and Evaluati/images/42bf68efd25c31287d241d0c61b8c8db640e919c6afcfb2af4ff337b68c20c11.jpg)

![95f1ab2db57f6b15a4b36f31f37e7d8b2bfd60e3f1fa2fddefb05c4b82298269.jpg](../iclr_results/2067_Towards Synergistic Path-based Explanations for Knowledge Graph Completion_ Exploration and Evaluati/images/95f1ab2db57f6b15a4b36f31f37e7d8b2bfd60e3f1fa2fddefb05c4b82298269.jpg)

![e4f65297f89a005051b463f1f7e6ed4181bdcf83db24dd42ba1eb2b3cda6af43.jpg](../iclr_results/2067_Towards Synergistic Path-based Explanations for Knowledge Graph Completion_ Exploration and Evaluati/images/e4f65297f89a005051b463f1f7e6ed4181bdcf83db24dd42ba1eb2b3cda6af43.jpg)

### Tables

![11c1df44ea5651566c2438b048910ed8841bf1834da6bb035d660e5c8a6dcedd.jpg](../iclr_results/2067_Towards Synergistic Path-based Explanations for Knowledge Graph Completion_ Exploration and Evaluati/tables/11c1df44ea5651566c2438b048910ed8841bf1834da6bb035d660e5c8a6dcedd.jpg)

![2b2edb8dc4ffc4db2b1ff6661f7eb448a9a5c6f7a5e43df25463713465d5b2f3.jpg](../iclr_results/2067_Towards Synergistic Path-based Explanations for Knowledge Graph Completion_ Exploration and Evaluati/tables/2b2edb8dc4ffc4db2b1ff6661f7eb448a9a5c6f7a5e43df25463713465d5b2f3.jpg)

![36f47d9fc02b7b5840e474d956302225e0143f6945d35b5804885cc365e3b59f.jpg](../iclr_results/2067_Towards Synergistic Path-based Explanations for Knowledge Graph Completion_ Exploration and Evaluati/tables/36f47d9fc02b7b5840e474d956302225e0143f6945d35b5804885cc365e3b59f.jpg)

![3756eae45388b2f30a7681747ccc236ca330f93397ca5e25a7bb0fba93618397.jpg](../iclr_results/2067_Towards Synergistic Path-based Explanations for Knowledge Graph Completion_ Exploration and Evaluati/tables/3756eae45388b2f30a7681747ccc236ca330f93397ca5e25a7bb0fba93618397.jpg)

![4e9f747b2298af2e1ecfc771489abe77b69b6231f1f1b6b6d98e3d5dbb9bd661.jpg](../iclr_results/2067_Towards Synergistic Path-based Explanations for Knowledge Graph Completion_ Exploration and Evaluati/tables/4e9f747b2298af2e1ecfc771489abe77b69b6231f1f1b6b6d98e3d5dbb9bd661.jpg)

![6b20fcee873ca35113921acfaf9a7c1c5cf738132ab0984dddf21bfec7e5d456.jpg](../iclr_results/2067_Towards Synergistic Path-based Explanations for Knowledge Graph Completion_ Exploration and Evaluati/tables/6b20fcee873ca35113921acfaf9a7c1c5cf738132ab0984dddf21bfec7e5d456.jpg)

![7604408b7124fbe4307ea4596f7cd633cebeef32d6809a8a92be2b4be8101445.jpg](../iclr_results/2067_Towards Synergistic Path-based Explanations for Knowledge Graph Completion_ Exploration and Evaluati/tables/7604408b7124fbe4307ea4596f7cd633cebeef32d6809a8a92be2b4be8101445.jpg)

![8bf8dd1fde8f1eb64d4d87518561680c97db1c22b72435132504c1c4335f48b6.jpg](../iclr_results/2067_Towards Synergistic Path-based Explanations for Knowledge Graph Completion_ Exploration and Evaluati/tables/8bf8dd1fde8f1eb64d4d87518561680c97db1c22b72435132504c1c4335f48b6.jpg)

![9c308c8bc7e3a25b24dd72bdf95279fc4be40c8b343ea984d64e174bfc9b5fc3.jpg](../iclr_results/2067_Towards Synergistic Path-based Explanations for Knowledge Graph Completion_ Exploration and Evaluati/tables/9c308c8bc7e3a25b24dd72bdf95279fc4be40c8b343ea984d64e174bfc9b5fc3.jpg)

![a14c232abb39d5aa0a636c10d5bda352ea49fa253bb09431bf19ce6169772d34.jpg](../iclr_results/2067_Towards Synergistic Path-based Explanations for Knowledge Graph Completion_ Exploration and Evaluati/tables/a14c232abb39d5aa0a636c10d5bda352ea49fa253bb09431bf19ce6169772d34.jpg)

![a41e0220755edb34bf0fda8d751c023750658742383d8f33049934d20c13e00b.jpg](../iclr_results/2067_Towards Synergistic Path-based Explanations for Knowledge Graph Completion_ Exploration and Evaluati/tables/a41e0220755edb34bf0fda8d751c023750658742383d8f33049934d20c13e00b.jpg)

![cc6c1aef99d66a5d07a0e13e24fb887d37ea0ba8ea08154b2e535daa800df1b8.jpg](../iclr_results/2067_Towards Synergistic Path-based Explanations for Knowledge Graph Completion_ Exploration and Evaluati/tables/cc6c1aef99d66a5d07a0e13e24fb887d37ea0ba8ea08154b2e535daa800df1b8.jpg)

![f5e55f13aabf9475161b53b0ac599c73fcb519676fc185e836d712d3edb1c691.jpg](../iclr_results/2067_Towards Synergistic Path-based Explanations for Knowledge Graph Completion_ Exploration and Evaluati/tables/f5e55f13aabf9475161b53b0ac599c73fcb519676fc185e836d712d3edb1c691.jpg)

## Transition Path Sampling with Improved Off-Policy Training of Diffusion Path Samplers


### Images

![0f326c51bdf4bdee8de794f7358fdca73fc773f9c2d791dcb1d8052dfc4b690e.jpg](../iclr_results/2068_Transition Path Sampling with Improved Off-Policy Training of Diffusion Path Samplers/images/0f326c51bdf4bdee8de794f7358fdca73fc773f9c2d791dcb1d8052dfc4b690e.jpg)

![279346d4fa26a284c76915db6349aa317d01012febe4e9a071d9ffdc6e818032.jpg](../iclr_results/2068_Transition Path Sampling with Improved Off-Policy Training of Diffusion Path Samplers/images/279346d4fa26a284c76915db6349aa317d01012febe4e9a071d9ffdc6e818032.jpg)

![3cda2f94365c28e24e7352e5e08abaf4ea94200b37351b05e2a84a4b43d4cff3.jpg](../iclr_results/2068_Transition Path Sampling with Improved Off-Policy Training of Diffusion Path Samplers/images/3cda2f94365c28e24e7352e5e08abaf4ea94200b37351b05e2a84a4b43d4cff3.jpg)

![98dc95c319633d29f216e66e6ed11c9ec81ff83ddfee87262524f925e899da46.jpg](../iclr_results/2068_Transition Path Sampling with Improved Off-Policy Training of Diffusion Path Samplers/images/98dc95c319633d29f216e66e6ed11c9ec81ff83ddfee87262524f925e899da46.jpg)

![9b7bf811ea8fb9a3df74f213ae88b69c6147f64754a3035908ba5d11e75749ea.jpg](../iclr_results/2068_Transition Path Sampling with Improved Off-Policy Training of Diffusion Path Samplers/images/9b7bf811ea8fb9a3df74f213ae88b69c6147f64754a3035908ba5d11e75749ea.jpg)

![c5d623287b10ca2b8b694f5689baa76e7dfc4163a4389b6d75b90cada4fb93f8.jpg](../iclr_results/2068_Transition Path Sampling with Improved Off-Policy Training of Diffusion Path Samplers/images/c5d623287b10ca2b8b694f5689baa76e7dfc4163a4389b6d75b90cada4fb93f8.jpg)

![e8b79bd9f0582ac5b8669e59b86852247816bbe1c410a403bb9a185afba5a620.jpg](../iclr_results/2068_Transition Path Sampling with Improved Off-Policy Training of Diffusion Path Samplers/images/e8b79bd9f0582ac5b8669e59b86852247816bbe1c410a403bb9a185afba5a620.jpg)

![f6b3b5cb9a9a6854997582ad5c570d50d5705e5cbb849fc571075673d920337d.jpg](../iclr_results/2068_Transition Path Sampling with Improved Off-Policy Training of Diffusion Path Samplers/images/f6b3b5cb9a9a6854997582ad5c570d50d5705e5cbb849fc571075673d920337d.jpg)

### Tables

![07a967641cebe51618043b441f4d27bb83f65fef2d84e9093ae7ff0121996bd2.jpg](../iclr_results/2068_Transition Path Sampling with Improved Off-Policy Training of Diffusion Path Samplers/tables/07a967641cebe51618043b441f4d27bb83f65fef2d84e9093ae7ff0121996bd2.jpg)

![4f1b935217be1ee177b121af505fcd193abb8797d70648574820dcc1b58e5be3.jpg](../iclr_results/2068_Transition Path Sampling with Improved Off-Policy Training of Diffusion Path Samplers/tables/4f1b935217be1ee177b121af505fcd193abb8797d70648574820dcc1b58e5be3.jpg)

![631f656ea9c5571d1850988bcd9e9cdec653a42e85024d7a6d866797f024ed2a.jpg](../iclr_results/2068_Transition Path Sampling with Improved Off-Policy Training of Diffusion Path Samplers/tables/631f656ea9c5571d1850988bcd9e9cdec653a42e85024d7a6d866797f024ed2a.jpg)

![9343dd937c53d22ce33e5c2d4de7954dc2c11299e246ea317814aff25adb7aa4.jpg](../iclr_results/2068_Transition Path Sampling with Improved Off-Policy Training of Diffusion Path Samplers/tables/9343dd937c53d22ce33e5c2d4de7954dc2c11299e246ea317814aff25adb7aa4.jpg)

![9450456f9c8b7093335c5164de3cb78da2f3e50c3ef14b64b686f1cfc5bba1a9.jpg](../iclr_results/2068_Transition Path Sampling with Improved Off-Policy Training of Diffusion Path Samplers/tables/9450456f9c8b7093335c5164de3cb78da2f3e50c3ef14b64b686f1cfc5bba1a9.jpg)

![a31ab71fb305e09edd0171a6e7c12f22ece2ea7109aa9dffc9124c6e724f4db6.jpg](../iclr_results/2068_Transition Path Sampling with Improved Off-Policy Training of Diffusion Path Samplers/tables/a31ab71fb305e09edd0171a6e7c12f22ece2ea7109aa9dffc9124c6e724f4db6.jpg)

![bce40d820eaa7a53e0698bb43b3ac07f59e0bedcd49b828a15c14c7742b0764c.jpg](../iclr_results/2068_Transition Path Sampling with Improved Off-Policy Training of Diffusion Path Samplers/tables/bce40d820eaa7a53e0698bb43b3ac07f59e0bedcd49b828a15c14c7742b0764c.jpg)

![dd5bbb7572ae03e210fbda3cc902bd86e984b0a5a286ed14995a06a88cad92b7.jpg](../iclr_results/2068_Transition Path Sampling with Improved Off-Policy Training of Diffusion Path Samplers/tables/dd5bbb7572ae03e210fbda3cc902bd86e984b0a5a286ed14995a06a88cad92b7.jpg)

## Discovering Influential Neuron Path in Vision Transformers


### Images

![04cd4d5713de95e0549a706c650089acf53dcf7c0ae55f58c1945dab12349715.jpg](../iclr_results/2069_Discovering Influential Neuron Path in Vision Transformers/images/04cd4d5713de95e0549a706c650089acf53dcf7c0ae55f58c1945dab12349715.jpg)

![09a2a4562d38c9d4244d8b7d4416380f4a7ce551b3c0ca7c7d5df85137d3228d.jpg](../iclr_results/2069_Discovering Influential Neuron Path in Vision Transformers/images/09a2a4562d38c9d4244d8b7d4416380f4a7ce551b3c0ca7c7d5df85137d3228d.jpg)

![0e16901490111be946c4c0cf03b3df18b3f83a87a84401e046b1a9408f421fb0.jpg](../iclr_results/2069_Discovering Influential Neuron Path in Vision Transformers/images/0e16901490111be946c4c0cf03b3df18b3f83a87a84401e046b1a9408f421fb0.jpg)

![16cedf0b181128a8bf316e91296801fe9140322251746d8a07f44d5220f196e8.jpg](../iclr_results/2069_Discovering Influential Neuron Path in Vision Transformers/images/16cedf0b181128a8bf316e91296801fe9140322251746d8a07f44d5220f196e8.jpg)

![1df18734ebb31dda3469bc53ae702aef35a69f3398a7e49204570168f6b3c80d.jpg](../iclr_results/2069_Discovering Influential Neuron Path in Vision Transformers/images/1df18734ebb31dda3469bc53ae702aef35a69f3398a7e49204570168f6b3c80d.jpg)

![358693e99149ff3e1095ca797b562a4da880ec436613f715830e35cc76078133.jpg](../iclr_results/2069_Discovering Influential Neuron Path in Vision Transformers/images/358693e99149ff3e1095ca797b562a4da880ec436613f715830e35cc76078133.jpg)

![4326865d9e034fc13b69146a31457c0d1fd006a6591dd853457b4048aacbdeb7.jpg](../iclr_results/2069_Discovering Influential Neuron Path in Vision Transformers/images/4326865d9e034fc13b69146a31457c0d1fd006a6591dd853457b4048aacbdeb7.jpg)

![5620ae04e6caefad801952414ae5212d073c9156658bd2a883287e412ff01308.jpg](../iclr_results/2069_Discovering Influential Neuron Path in Vision Transformers/images/5620ae04e6caefad801952414ae5212d073c9156658bd2a883287e412ff01308.jpg)

![5bab75d45066aa6cac0f230b49dbde90b03aab9da3bc752060e313e6da2ee82c.jpg](../iclr_results/2069_Discovering Influential Neuron Path in Vision Transformers/images/5bab75d45066aa6cac0f230b49dbde90b03aab9da3bc752060e313e6da2ee82c.jpg)

![69aa78615e1fe39fd6db4fc95a48a57681d478e24c1ae4e27ba1d06b7f5b69b4.jpg](../iclr_results/2069_Discovering Influential Neuron Path in Vision Transformers/images/69aa78615e1fe39fd6db4fc95a48a57681d478e24c1ae4e27ba1d06b7f5b69b4.jpg)

![6c4e1ec1b4699aa0637056e4b6111a34836015b60bf9eb821d226b704186cd14.jpg](../iclr_results/2069_Discovering Influential Neuron Path in Vision Transformers/images/6c4e1ec1b4699aa0637056e4b6111a34836015b60bf9eb821d226b704186cd14.jpg)

![752b652969343104107480fd6f982386e93cce592cf381321e151dc61eda6571.jpg](../iclr_results/2069_Discovering Influential Neuron Path in Vision Transformers/images/752b652969343104107480fd6f982386e93cce592cf381321e151dc61eda6571.jpg)

![8cec45c3796a83265ba61308e893705cf1c2db49b46f70a81b3b11012c98ed1e.jpg](../iclr_results/2069_Discovering Influential Neuron Path in Vision Transformers/images/8cec45c3796a83265ba61308e893705cf1c2db49b46f70a81b3b11012c98ed1e.jpg)

![92e3e1e336062f3e5be8a6f9700b4b5edd9b7c2456ef3f2d155375972900fff9.jpg](../iclr_results/2069_Discovering Influential Neuron Path in Vision Transformers/images/92e3e1e336062f3e5be8a6f9700b4b5edd9b7c2456ef3f2d155375972900fff9.jpg)

![b062e7b6bcf85a2a477de922c886b850ceaee555f3e893b7a2e6ffc24f6094ed.jpg](../iclr_results/2069_Discovering Influential Neuron Path in Vision Transformers/images/b062e7b6bcf85a2a477de922c886b850ceaee555f3e893b7a2e6ffc24f6094ed.jpg)

### Tables

![75aaaa9b44460365fd1a1ccaef8f925cda1f85549958a82e8c0cdf8b950dc04f.jpg](../iclr_results/2069_Discovering Influential Neuron Path in Vision Transformers/tables/75aaaa9b44460365fd1a1ccaef8f925cda1f85549958a82e8c0cdf8b950dc04f.jpg)

![7e2c5c6022aed12f85f78cc943a3766932147c1e0032b67d5fbb61214b85942b.jpg](../iclr_results/2069_Discovering Influential Neuron Path in Vision Transformers/tables/7e2c5c6022aed12f85f78cc943a3766932147c1e0032b67d5fbb61214b85942b.jpg)

![ec23d42b3c1228bc08eceaf0c3b769e1d8c9985f007b3fb40cdcf58dd51458bc.jpg](../iclr_results/2069_Discovering Influential Neuron Path in Vision Transformers/tables/ec23d42b3c1228bc08eceaf0c3b769e1d8c9985f007b3fb40cdcf58dd51458bc.jpg)

## Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model


### Images

![16a7b7ce4963156d9c7176d9bc7aaca4012e853dc78b49f204ae52c0ef814e39.jpg](../iclr_results/2070_Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model/images/16a7b7ce4963156d9c7176d9bc7aaca4012e853dc78b49f204ae52c0ef814e39.jpg)

![76f321b5e6e552f1c5bc0d14fdb6c8aea01f737d4b938ef8fad64d2872461057.jpg](../iclr_results/2070_Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model/images/76f321b5e6e552f1c5bc0d14fdb6c8aea01f737d4b938ef8fad64d2872461057.jpg)

![7a500ebd2237bd8b3e46a52a3fdd00d6ddb1a77bf4150e9cf40dfacbd1d39588.jpg](../iclr_results/2070_Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model/images/7a500ebd2237bd8b3e46a52a3fdd00d6ddb1a77bf4150e9cf40dfacbd1d39588.jpg)

### Tables

![21cb6d5118ee1c02f82eb7b7eea4c06f2f8f43a0dfbbe2ade1a3ac7758d3c5f5.jpg](../iclr_results/2070_Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model/tables/21cb6d5118ee1c02f82eb7b7eea4c06f2f8f43a0dfbbe2ade1a3ac7758d3c5f5.jpg)

![26fa7d30dd8b1f85a8b0e7b8848d0278f54b10e93b0907d1d0987002337ca3be.jpg](../iclr_results/2070_Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model/tables/26fa7d30dd8b1f85a8b0e7b8848d0278f54b10e93b0907d1d0987002337ca3be.jpg)

![3f6b902ebc3f5a092572418ca7541f4efbc9131bde785e5a9ad054659e71349f.jpg](../iclr_results/2070_Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model/tables/3f6b902ebc3f5a092572418ca7541f4efbc9131bde785e5a9ad054659e71349f.jpg)

![45684259307251bb0f7259a1b4d918e1c058aafc7480d03f84553ba2f322235a.jpg](../iclr_results/2070_Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model/tables/45684259307251bb0f7259a1b4d918e1c058aafc7480d03f84553ba2f322235a.jpg)

![46a3d4e238e02ae22f35c82644340eb225ea872d7072594b85223725a94f2ba2.jpg](../iclr_results/2070_Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model/tables/46a3d4e238e02ae22f35c82644340eb225ea872d7072594b85223725a94f2ba2.jpg)

![4b0632d1e4eac50b60f9dee4be04a601efb03a17f1da01592666905e90829e6d.jpg](../iclr_results/2070_Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model/tables/4b0632d1e4eac50b60f9dee4be04a601efb03a17f1da01592666905e90829e6d.jpg)

![5e9c84874ff2789fb71072ca75c22eba9a116cdcbc4e75a6afe6484ffb182f49.jpg](../iclr_results/2070_Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model/tables/5e9c84874ff2789fb71072ca75c22eba9a116cdcbc4e75a6afe6484ffb182f49.jpg)

![68ab3b18a60ffb14e0513c20ced942475192c0d042b52a6b2af156eb5d1b9163.jpg](../iclr_results/2070_Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model/tables/68ab3b18a60ffb14e0513c20ced942475192c0d042b52a6b2af156eb5d1b9163.jpg)

![70d499d6fd3339ee7bd2f79d34ff0d124aea7aa67bcbcc326eb6a3b9f9dd39d8.jpg](../iclr_results/2070_Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model/tables/70d499d6fd3339ee7bd2f79d34ff0d124aea7aa67bcbcc326eb6a3b9f9dd39d8.jpg)

![79743d85fec8261331702cda9b1feeb3a7a50fe026ec213acec039fecbcce7ed.jpg](../iclr_results/2070_Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model/tables/79743d85fec8261331702cda9b1feeb3a7a50fe026ec213acec039fecbcce7ed.jpg)

![87cb6d3e1cfd51e01749791c6145c235b8abb51797c1e058d608fd5d62423cda.jpg](../iclr_results/2070_Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model/tables/87cb6d3e1cfd51e01749791c6145c235b8abb51797c1e058d608fd5d62423cda.jpg)

![a21821e3b1add8cb8945c25fe32b0f775ef2cffc9d605819d191d548bb026af5.jpg](../iclr_results/2070_Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model/tables/a21821e3b1add8cb8945c25fe32b0f775ef2cffc9d605819d191d548bb026af5.jpg)

![ad7bfc80dc1f98021d4fdc2ba5a77d4328ac9105b39a55d8c1c7096efd476a7d.jpg](../iclr_results/2070_Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model/tables/ad7bfc80dc1f98021d4fdc2ba5a77d4328ac9105b39a55d8c1c7096efd476a7d.jpg)

![affa7c11814c7f5df871b64e4a4ab20b70390c7b3cc8a525529bd018f66e358d.jpg](../iclr_results/2070_Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model/tables/affa7c11814c7f5df871b64e4a4ab20b70390c7b3cc8a525529bd018f66e358d.jpg)

![b935c54c5456889a7a83a5b015ae253e660bc1d8fd5c1514ea86b8b367f8c485.jpg](../iclr_results/2070_Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model/tables/b935c54c5456889a7a83a5b015ae253e660bc1d8fd5c1514ea86b8b367f8c485.jpg)

![c2c0d4cc35a471d288d426b7733e752103529ef84c11d014f2199152b6901603.jpg](../iclr_results/2070_Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model/tables/c2c0d4cc35a471d288d426b7733e752103529ef84c11d014f2199152b6901603.jpg)

![c66e77b6ab70a8f8363a8fae1b294bb793348f1adbad5543432c8634ba54b1e7.jpg](../iclr_results/2070_Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model/tables/c66e77b6ab70a8f8363a8fae1b294bb793348f1adbad5543432c8634ba54b1e7.jpg)

![cee1558da59d64f092febb460d3565a01707761beb43b8a7d7efe1e41e1a8fa3.jpg](../iclr_results/2070_Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model/tables/cee1558da59d64f092febb460d3565a01707761beb43b8a7d7efe1e41e1a8fa3.jpg)

![eca7851ff4e522b4f7d066b9efdc975d4dfe8abc3d1a15b870a65defa54ca6e3.jpg](../iclr_results/2070_Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model/tables/eca7851ff4e522b4f7d066b9efdc975d4dfe8abc3d1a15b870a65defa54ca6e3.jpg)

![fb8c180ec789334f0217a129b7c5e562a8d8372accd6a2eb46059d53a006afe2.jpg](../iclr_results/2070_Re-Evaluating the Impact of Unseen-Class Unlabeled Data on Semi-Supervised Learning Model/tables/fb8c180ec789334f0217a129b7c5e562a8d8372accd6a2eb46059d53a006afe2.jpg)

## Probe Pruning: Accelerating LLMs through Dynamic Pruning via Model-Probing


### Images

![10c9c4985bdc2dfc1d26006cfa3459c096f191ad1c83d39499d4015636ca852a.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/images/10c9c4985bdc2dfc1d26006cfa3459c096f191ad1c83d39499d4015636ca852a.jpg)

![2a88e5ce8a9286f8a97abb77cbf5e360d91a7f5b4dc25ced8aa538aa21d1097a.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/images/2a88e5ce8a9286f8a97abb77cbf5e360d91a7f5b4dc25ced8aa538aa21d1097a.jpg)

![6b1174a05379a3bf13a48867d9f43d425243c4a4c7a2bde736e6bd886d465302.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/images/6b1174a05379a3bf13a48867d9f43d425243c4a4c7a2bde736e6bd886d465302.jpg)

![7e020626bc27baa9f1bcd33a3b7c9fc0efe0363e310d44ad2714e8ec6730b493.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/images/7e020626bc27baa9f1bcd33a3b7c9fc0efe0363e310d44ad2714e8ec6730b493.jpg)

![fff800b61b38687079f805eef1f91903c716004055738a29bf9ec967e43fa7fd.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/images/fff800b61b38687079f805eef1f91903c716004055738a29bf9ec967e43fa7fd.jpg)

### Tables

![044a628228109a2b44708144d5422eac4811d517eaee09685c7ba3ded69e8735.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/tables/044a628228109a2b44708144d5422eac4811d517eaee09685c7ba3ded69e8735.jpg)

![07c07fdb0eaa4b73e0a7efcd0419f2786ee6e73b72714788bdf4559b9a95e009.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/tables/07c07fdb0eaa4b73e0a7efcd0419f2786ee6e73b72714788bdf4559b9a95e009.jpg)

![1ccc2993de452eaf0d570afde07ce387de96f515c2bdb7e2e890d511d435d4bd.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/tables/1ccc2993de452eaf0d570afde07ce387de96f515c2bdb7e2e890d511d435d4bd.jpg)

![2fca69e4c94e6ea1667ad0c17910abd4863a5a738794465276b103f2f73df26a.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/tables/2fca69e4c94e6ea1667ad0c17910abd4863a5a738794465276b103f2f73df26a.jpg)

![574fda609a21b240efa3774d45f8d8fb58aee444bbfaf4db34a8b6e1fe688d9c.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/tables/574fda609a21b240efa3774d45f8d8fb58aee444bbfaf4db34a8b6e1fe688d9c.jpg)

![63d686e91210d98c9d04c062e3a5755086fcc0fada930e08d71ce649debb4a31.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/tables/63d686e91210d98c9d04c062e3a5755086fcc0fada930e08d71ce649debb4a31.jpg)

![6d734c38056944b279c805a678eb6a337c013ae37ce7c422f856b5cf2100c29e.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/tables/6d734c38056944b279c805a678eb6a337c013ae37ce7c422f856b5cf2100c29e.jpg)

![7022df0c07580192fea789181ec0c5ff31f586e5a01ba0b41f44dae80f67023d.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/tables/7022df0c07580192fea789181ec0c5ff31f586e5a01ba0b41f44dae80f67023d.jpg)

![73bdebb675660922125d501327b3cb70d0aafe8982ba3668beb07d93524b0395.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/tables/73bdebb675660922125d501327b3cb70d0aafe8982ba3668beb07d93524b0395.jpg)

![74f76f606b3577e4c01c9fb5e24280cba9bcde76bf317fa50456b7bbe81bfb90.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/tables/74f76f606b3577e4c01c9fb5e24280cba9bcde76bf317fa50456b7bbe81bfb90.jpg)

![7f5bffb8fbd8976656d6d508cc141709f1b9fba3ccafd93c837baea1cd6abb96.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/tables/7f5bffb8fbd8976656d6d508cc141709f1b9fba3ccafd93c837baea1cd6abb96.jpg)

![91b79d309ad661665fd04b3ec340dd259608d95ddfd753aef445efe408505cf4.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/tables/91b79d309ad661665fd04b3ec340dd259608d95ddfd753aef445efe408505cf4.jpg)

![a5229add81ccf5ef4895ca736f7da34ca204af1f7fd1414c796691109d942918.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/tables/a5229add81ccf5ef4895ca736f7da34ca204af1f7fd1414c796691109d942918.jpg)

![a8105a6a77b0c4850167d121e2fb5744ba23bbad04991d99cac1c4e1c0ba0ff1.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/tables/a8105a6a77b0c4850167d121e2fb5744ba23bbad04991d99cac1c4e1c0ba0ff1.jpg)

![aa8f6fd6363c129f17f79d764d332ee39ebb9710259719d7c0c2a34218b23bad.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/tables/aa8f6fd6363c129f17f79d764d332ee39ebb9710259719d7c0c2a34218b23bad.jpg)

![c225db9bf65d9215d6f1ddc697496b67d6760dedfac2a8d9ca39ac502f6ba7f1.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/tables/c225db9bf65d9215d6f1ddc697496b67d6760dedfac2a8d9ca39ac502f6ba7f1.jpg)

![ccd66807014d28b7da100fe100fb38bdc56b8f48ee2f59d2d338c880c820da3b.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/tables/ccd66807014d28b7da100fe100fb38bdc56b8f48ee2f59d2d338c880c820da3b.jpg)

![e384a0b160a0dbca3098f554eb6e5f8039ba167e9539721d68168b17d4081f53.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/tables/e384a0b160a0dbca3098f554eb6e5f8039ba167e9539721d68168b17d4081f53.jpg)

![f8423b0e72dcd0e5fda5a8b1c81792262554a016f89e819c67a7bb02205b22aa.jpg](../iclr_results/2071_Probe Pruning_ Accelerating LLMs through Dynamic Pruning via Model-Probing/tables/f8423b0e72dcd0e5fda5a8b1c81792262554a016f89e819c67a7bb02205b22aa.jpg)

## Scaling up Masked Diffusion Models on Text


### Images

![cebc722251e05592369ab1307866fd05acb78820fac954c5f8e3ea5e894eec12.jpg](../iclr_results/2072_Scaling up Masked Diffusion Models on Text/images/cebc722251e05592369ab1307866fd05acb78820fac954c5f8e3ea5e894eec12.jpg)

![d46da6ce9d1589563543b30aead894693e7dbcb3c1aea2ee9cdad78410934de0.jpg](../iclr_results/2072_Scaling up Masked Diffusion Models on Text/images/d46da6ce9d1589563543b30aead894693e7dbcb3c1aea2ee9cdad78410934de0.jpg)

![fb349023ee155215698c5b71f70ddf8982bf5029118fcc52172bce4bfa8d1f25.jpg](../iclr_results/2072_Scaling up Masked Diffusion Models on Text/images/fb349023ee155215698c5b71f70ddf8982bf5029118fcc52172bce4bfa8d1f25.jpg)

### Tables

![064c611a2fae0c756c26d0337b5b9e1469759fe13253f2e7c63f524e5afa5e60.jpg](../iclr_results/2072_Scaling up Masked Diffusion Models on Text/tables/064c611a2fae0c756c26d0337b5b9e1469759fe13253f2e7c63f524e5afa5e60.jpg)

![065e91e736b27f643c07f5e3126110a50581179413598cc9e867b3e91757321e.jpg](../iclr_results/2072_Scaling up Masked Diffusion Models on Text/tables/065e91e736b27f643c07f5e3126110a50581179413598cc9e867b3e91757321e.jpg)

![13249759a100facb5ec860ace0c29f7b5f91bcfac848138d87c9c757bd02340c.jpg](../iclr_results/2072_Scaling up Masked Diffusion Models on Text/tables/13249759a100facb5ec860ace0c29f7b5f91bcfac848138d87c9c757bd02340c.jpg)

![3c54bfe19d5bf4eaef70f1b7e0d200e3c150b03dd7f856ca50e8720ac19d50df.jpg](../iclr_results/2072_Scaling up Masked Diffusion Models on Text/tables/3c54bfe19d5bf4eaef70f1b7e0d200e3c150b03dd7f856ca50e8720ac19d50df.jpg)

![5077c8f40680c690caf04fa29dbed1e6aa90d73ebcfe8a1948512e7656daf2ba.jpg](../iclr_results/2072_Scaling up Masked Diffusion Models on Text/tables/5077c8f40680c690caf04fa29dbed1e6aa90d73ebcfe8a1948512e7656daf2ba.jpg)

![5c6a0a9a69ccefb834cd1fbe0d66056e119ede56fd2f395db4eab46281700c3b.jpg](../iclr_results/2072_Scaling up Masked Diffusion Models on Text/tables/5c6a0a9a69ccefb834cd1fbe0d66056e119ede56fd2f395db4eab46281700c3b.jpg)

![7c6c20ecd973472239929c52cae595795c12e577c96f38a4706b220f69a2ef26.jpg](../iclr_results/2072_Scaling up Masked Diffusion Models on Text/tables/7c6c20ecd973472239929c52cae595795c12e577c96f38a4706b220f69a2ef26.jpg)

![98053fcdb34b14bed01d4fb18bc274ea3efe61f1f62540e667ea49cc2f3d682d.jpg](../iclr_results/2072_Scaling up Masked Diffusion Models on Text/tables/98053fcdb34b14bed01d4fb18bc274ea3efe61f1f62540e667ea49cc2f3d682d.jpg)

![98977d8a957a0d89eaaa48f38270df1e78a68e1beae5e87f6f28fea6bc2cd82d.jpg](../iclr_results/2072_Scaling up Masked Diffusion Models on Text/tables/98977d8a957a0d89eaaa48f38270df1e78a68e1beae5e87f6f28fea6bc2cd82d.jpg)

![ae79eefe5ff1034ad6e6d4bf204eaee5f5ea81b61d228ebd591af28cf26934df.jpg](../iclr_results/2072_Scaling up Masked Diffusion Models on Text/tables/ae79eefe5ff1034ad6e6d4bf204eaee5f5ea81b61d228ebd591af28cf26934df.jpg)

![b0d227bcd92adadc70f14576d9f9870ff00f14d8ece388a54c2f4f17ed440651.jpg](../iclr_results/2072_Scaling up Masked Diffusion Models on Text/tables/b0d227bcd92adadc70f14576d9f9870ff00f14d8ece388a54c2f4f17ed440651.jpg)

![b5c019b0485c91a26649a54038c922d3c27b422f9e88f2a798e377fd2a38a028.jpg](../iclr_results/2072_Scaling up Masked Diffusion Models on Text/tables/b5c019b0485c91a26649a54038c922d3c27b422f9e88f2a798e377fd2a38a028.jpg)

![d017cf0c7b5c734586831d58c6e85ee500d791bd9d4dc68f1073a15c4eddfc80.jpg](../iclr_results/2072_Scaling up Masked Diffusion Models on Text/tables/d017cf0c7b5c734586831d58c6e85ee500d791bd9d4dc68f1073a15c4eddfc80.jpg)

![eff3df3cfbcc6d43d498376c2bb3f8a85cd8d4d02f32835486f43c4246ac2da5.jpg](../iclr_results/2072_Scaling up Masked Diffusion Models on Text/tables/eff3df3cfbcc6d43d498376c2bb3f8a85cd8d4d02f32835486f43c4246ac2da5.jpg)

![f4ed4ed550ff4e772370391b353e7224ee0982215230c0b2157fe7147d293d15.jpg](../iclr_results/2072_Scaling up Masked Diffusion Models on Text/tables/f4ed4ed550ff4e772370391b353e7224ee0982215230c0b2157fe7147d293d15.jpg)

## REBIND: Enhancing Ground-state Molecular Conformation Prediction via Force-Based Graph Rewiring


### Images

![25a55002841f8241285f411d70da493c9d8f60c30c37aa84e8823b406023eeac.jpg](../iclr_results/2073_REBIND_ Enhancing Ground-state Molecular Conformation Prediction via Force-Based Graph Rewiring/images/25a55002841f8241285f411d70da493c9d8f60c30c37aa84e8823b406023eeac.jpg)

![2993eafc8563c97bfc1decb9c97ad29892bfaf536b4e363603d773e1fe8ea8f9.jpg](../iclr_results/2073_REBIND_ Enhancing Ground-state Molecular Conformation Prediction via Force-Based Graph Rewiring/images/2993eafc8563c97bfc1decb9c97ad29892bfaf536b4e363603d773e1fe8ea8f9.jpg)

![4a862b5c20f6da3149ef29b23f8056bbf9839f8e64c89f1b917b66e6f7c6478c.jpg](../iclr_results/2073_REBIND_ Enhancing Ground-state Molecular Conformation Prediction via Force-Based Graph Rewiring/images/4a862b5c20f6da3149ef29b23f8056bbf9839f8e64c89f1b917b66e6f7c6478c.jpg)

![6c3862d3f3d3a15d44607090232b6eb23762af5fe084c4c06319acd0871b3a87.jpg](../iclr_results/2073_REBIND_ Enhancing Ground-state Molecular Conformation Prediction via Force-Based Graph Rewiring/images/6c3862d3f3d3a15d44607090232b6eb23762af5fe084c4c06319acd0871b3a87.jpg)

![9d5ef7145e0547a2ae50a152e5d8307f20ffc1528c6ea49e63079cfb91d38947.jpg](../iclr_results/2073_REBIND_ Enhancing Ground-state Molecular Conformation Prediction via Force-Based Graph Rewiring/images/9d5ef7145e0547a2ae50a152e5d8307f20ffc1528c6ea49e63079cfb91d38947.jpg)

![fb0948449fb8e5e2ca070eae445c4bc11cf7cfa0f3b907c24afb3ce089f8ef38.jpg](../iclr_results/2073_REBIND_ Enhancing Ground-state Molecular Conformation Prediction via Force-Based Graph Rewiring/images/fb0948449fb8e5e2ca070eae445c4bc11cf7cfa0f3b907c24afb3ce089f8ef38.jpg)

### Tables

![2eb477fc518466ba839695d9cf481a15a97bcba4849a972b9d5d10ca79cb8df9.jpg](../iclr_results/2073_REBIND_ Enhancing Ground-state Molecular Conformation Prediction via Force-Based Graph Rewiring/tables/2eb477fc518466ba839695d9cf481a15a97bcba4849a972b9d5d10ca79cb8df9.jpg)

![6b7f6e5da8006489010ade61b9533d5210814d5e86afd7d2c2e7b9495629bf05.jpg](../iclr_results/2073_REBIND_ Enhancing Ground-state Molecular Conformation Prediction via Force-Based Graph Rewiring/tables/6b7f6e5da8006489010ade61b9533d5210814d5e86afd7d2c2e7b9495629bf05.jpg)

![7bbc93a1a78e70d615543044e622c42fdf2a161306fad626e904301778635a38.jpg](../iclr_results/2073_REBIND_ Enhancing Ground-state Molecular Conformation Prediction via Force-Based Graph Rewiring/tables/7bbc93a1a78e70d615543044e622c42fdf2a161306fad626e904301778635a38.jpg)

![7c61913a5e872b62ddcbab11d57f03f78e701fd43035dea5ddbecc47d48c7b06.jpg](../iclr_results/2073_REBIND_ Enhancing Ground-state Molecular Conformation Prediction via Force-Based Graph Rewiring/tables/7c61913a5e872b62ddcbab11d57f03f78e701fd43035dea5ddbecc47d48c7b06.jpg)

![a6b97307513825c53765225be7760360925caacf2ea2ed74e7b611fe0a0fb4dc.jpg](../iclr_results/2073_REBIND_ Enhancing Ground-state Molecular Conformation Prediction via Force-Based Graph Rewiring/tables/a6b97307513825c53765225be7760360925caacf2ea2ed74e7b611fe0a0fb4dc.jpg)

![a70e4abeaf9322e27e4fac4f2f111769bf8b1136a5862f1ec1f44b044c07c737.jpg](../iclr_results/2073_REBIND_ Enhancing Ground-state Molecular Conformation Prediction via Force-Based Graph Rewiring/tables/a70e4abeaf9322e27e4fac4f2f111769bf8b1136a5862f1ec1f44b044c07c737.jpg)

![cd92df62d1ef19b0d1bad49ac111239fec8ef33e115fe4f1351e7723fe91f302.jpg](../iclr_results/2073_REBIND_ Enhancing Ground-state Molecular Conformation Prediction via Force-Based Graph Rewiring/tables/cd92df62d1ef19b0d1bad49ac111239fec8ef33e115fe4f1351e7723fe91f302.jpg)

![edcd346d14de0f2ceb96d1eb5013b183a779805bab933c510c652e95c1e8a904.jpg](../iclr_results/2073_REBIND_ Enhancing Ground-state Molecular Conformation Prediction via Force-Based Graph Rewiring/tables/edcd346d14de0f2ceb96d1eb5013b183a779805bab933c510c652e95c1e8a904.jpg)

## Q-Adapter: Customizing Pre-trained LLMs to New Preferences with Forgetting Mitigation


### Images

![0902ffd6ccec95f8c487ea803583232dac6bab234ca75fb46cb631ce4b00776d.jpg](../iclr_results/2074_Q-Adapter_ Customizing Pre-trained LLMs to New Preferences with Forgetting Mitigation/images/0902ffd6ccec95f8c487ea803583232dac6bab234ca75fb46cb631ce4b00776d.jpg)

![843d5c40a1e761b7dc5f7071f3cea344e2e9239cce28165689aa5414d20a7b55.jpg](../iclr_results/2074_Q-Adapter_ Customizing Pre-trained LLMs to New Preferences with Forgetting Mitigation/images/843d5c40a1e761b7dc5f7071f3cea344e2e9239cce28165689aa5414d20a7b55.jpg)

![b518b84cd19d69c8a22161a9e8d3e291bd0b9668f6de471b71c60e3e9657f3f2.jpg](../iclr_results/2074_Q-Adapter_ Customizing Pre-trained LLMs to New Preferences with Forgetting Mitigation/images/b518b84cd19d69c8a22161a9e8d3e291bd0b9668f6de471b71c60e3e9657f3f2.jpg)

![d105139d1720c7f4502aed718573f1c37731fb338d1eb086d0d5f9789111318a.jpg](../iclr_results/2074_Q-Adapter_ Customizing Pre-trained LLMs to New Preferences with Forgetting Mitigation/images/d105139d1720c7f4502aed718573f1c37731fb338d1eb086d0d5f9789111318a.jpg)

![f554de0cc382937ae5def237235779c8a0a5da566422b5d5c55976692b85beca.jpg](../iclr_results/2074_Q-Adapter_ Customizing Pre-trained LLMs to New Preferences with Forgetting Mitigation/images/f554de0cc382937ae5def237235779c8a0a5da566422b5d5c55976692b85beca.jpg)

![f74ecbc97bb31e40d2ca4ad17a9376259bab06c841932707937ab6fdbaa6fb9a.jpg](../iclr_results/2074_Q-Adapter_ Customizing Pre-trained LLMs to New Preferences with Forgetting Mitigation/images/f74ecbc97bb31e40d2ca4ad17a9376259bab06c841932707937ab6fdbaa6fb9a.jpg)

### Tables

![6be70c92c7bf72a55cc5b68a1b07284d5db9d8b80846c87a62e6aec4c980215e.jpg](../iclr_results/2074_Q-Adapter_ Customizing Pre-trained LLMs to New Preferences with Forgetting Mitigation/tables/6be70c92c7bf72a55cc5b68a1b07284d5db9d8b80846c87a62e6aec4c980215e.jpg)

![830eef94c3908cfa500d93b0f8069f27393fdbf361c21c0ca213f529bfdbd2c3.jpg](../iclr_results/2074_Q-Adapter_ Customizing Pre-trained LLMs to New Preferences with Forgetting Mitigation/tables/830eef94c3908cfa500d93b0f8069f27393fdbf361c21c0ca213f529bfdbd2c3.jpg)

![a5b9612d1ecb02ec2c4a416553abd24dc0afe82332ce35c18a22d137fc228682.jpg](../iclr_results/2074_Q-Adapter_ Customizing Pre-trained LLMs to New Preferences with Forgetting Mitigation/tables/a5b9612d1ecb02ec2c4a416553abd24dc0afe82332ce35c18a22d137fc228682.jpg)

![afe95b01c2ef1908641ec20f0f80b1462194c763db70c845efcaa83ab8bb2c59.jpg](../iclr_results/2074_Q-Adapter_ Customizing Pre-trained LLMs to New Preferences with Forgetting Mitigation/tables/afe95b01c2ef1908641ec20f0f80b1462194c763db70c845efcaa83ab8bb2c59.jpg)

## Training Robust Ensembles Requires Rethinking Lipschitz Continuity


### Images

![1e1a551d3d18bf32e80bc7c37f3eb2d2811facf9556101d6ab1e2318fd812432.jpg](../iclr_results/2075_Training Robust Ensembles Requires Rethinking Lipschitz Continuity/images/1e1a551d3d18bf32e80bc7c37f3eb2d2811facf9556101d6ab1e2318fd812432.jpg)

![2aa08a05e590a83fbef318eded5f73d9c2253d0b93a20e4fcc47448354294cda.jpg](../iclr_results/2075_Training Robust Ensembles Requires Rethinking Lipschitz Continuity/images/2aa08a05e590a83fbef318eded5f73d9c2253d0b93a20e4fcc47448354294cda.jpg)

![3bbde0b73ff510d52d7b44873a1639ab271beb636dadade8a3036ef80036701b.jpg](../iclr_results/2075_Training Robust Ensembles Requires Rethinking Lipschitz Continuity/images/3bbde0b73ff510d52d7b44873a1639ab271beb636dadade8a3036ef80036701b.jpg)

![5e9d00c1f6a120572ee7096c87b7142c602fd60a4ff9bc0aa2b3e837aebec26a.jpg](../iclr_results/2075_Training Robust Ensembles Requires Rethinking Lipschitz Continuity/images/5e9d00c1f6a120572ee7096c87b7142c602fd60a4ff9bc0aa2b3e837aebec26a.jpg)

![7671883bae9eb84e41f68c5cfc1d2b53967b6db474f422a29090db5b225b585b.jpg](../iclr_results/2075_Training Robust Ensembles Requires Rethinking Lipschitz Continuity/images/7671883bae9eb84e41f68c5cfc1d2b53967b6db474f422a29090db5b225b585b.jpg)

![7bedc887a1b8f6b6d295acecc9c35cf3d17ba71d5cc4cf6fe2a4a114a4e1aaae.jpg](../iclr_results/2075_Training Robust Ensembles Requires Rethinking Lipschitz Continuity/images/7bedc887a1b8f6b6d295acecc9c35cf3d17ba71d5cc4cf6fe2a4a114a4e1aaae.jpg)

![8b92cf4ecbed559676c9bb26f5a71c99037bdb256e98db5cde652db1b27f4426.jpg](../iclr_results/2075_Training Robust Ensembles Requires Rethinking Lipschitz Continuity/images/8b92cf4ecbed559676c9bb26f5a71c99037bdb256e98db5cde652db1b27f4426.jpg)

![bd4ae87cad43daf15864b07f93cbe082b87784ed7e43c34a3f86537225131aec.jpg](../iclr_results/2075_Training Robust Ensembles Requires Rethinking Lipschitz Continuity/images/bd4ae87cad43daf15864b07f93cbe082b87784ed7e43c34a3f86537225131aec.jpg)

![d5b468122f7d8153e62fcc8d777406d09dd69d0936448c5e0f35db9ab7d221f6.jpg](../iclr_results/2075_Training Robust Ensembles Requires Rethinking Lipschitz Continuity/images/d5b468122f7d8153e62fcc8d777406d09dd69d0936448c5e0f35db9ab7d221f6.jpg)

### Tables

![5d3f7a5557c93442806ff0a1ba9bab92a8f4eaee25da95e5486d8cf5b939677a.jpg](../iclr_results/2075_Training Robust Ensembles Requires Rethinking Lipschitz Continuity/tables/5d3f7a5557c93442806ff0a1ba9bab92a8f4eaee25da95e5486d8cf5b939677a.jpg)

![6e065cacae3ba248d0a9209e8015f2d84e2ae7537fd22215ca7de034913f71d4.jpg](../iclr_results/2075_Training Robust Ensembles Requires Rethinking Lipschitz Continuity/tables/6e065cacae3ba248d0a9209e8015f2d84e2ae7537fd22215ca7de034913f71d4.jpg)

![8aec8a89b027e4cd28d3540b51f03362deaec18f3d8d9f617e5273a4d688acb2.jpg](../iclr_results/2075_Training Robust Ensembles Requires Rethinking Lipschitz Continuity/tables/8aec8a89b027e4cd28d3540b51f03362deaec18f3d8d9f617e5273a4d688acb2.jpg)

![8b878175cde7f4d621d9494311501dcb2d59ad0625418c6abb2ab10be9bd12df.jpg](../iclr_results/2075_Training Robust Ensembles Requires Rethinking Lipschitz Continuity/tables/8b878175cde7f4d621d9494311501dcb2d59ad0625418c6abb2ab10be9bd12df.jpg)

![8fd5e94d326af9e054a4bc5a37ba8e5f62b57079f4221fc0ee25dc3b2f04ac7e.jpg](../iclr_results/2075_Training Robust Ensembles Requires Rethinking Lipschitz Continuity/tables/8fd5e94d326af9e054a4bc5a37ba8e5f62b57079f4221fc0ee25dc3b2f04ac7e.jpg)

![9ad2e66e70949f2e21440ae5e27b6e75b1a51abe4d935d0fb12718b9f344a873.jpg](../iclr_results/2075_Training Robust Ensembles Requires Rethinking Lipschitz Continuity/tables/9ad2e66e70949f2e21440ae5e27b6e75b1a51abe4d935d0fb12718b9f344a873.jpg)

![a43c5ddfecb9619d5aa9f197369e9cd87848aa9421f806adaaadf19b6ba08c53.jpg](../iclr_results/2075_Training Robust Ensembles Requires Rethinking Lipschitz Continuity/tables/a43c5ddfecb9619d5aa9f197369e9cd87848aa9421f806adaaadf19b6ba08c53.jpg)

![c9a4dd8c3ebedc14c34db404b1bf88b97d912e80646d00237facafbe8f63a6cc.jpg](../iclr_results/2075_Training Robust Ensembles Requires Rethinking Lipschitz Continuity/tables/c9a4dd8c3ebedc14c34db404b1bf88b97d912e80646d00237facafbe8f63a6cc.jpg)

![cd98132fefb083fb307f31d053edce1922a02965b72594d62c831454d16eedd2.jpg](../iclr_results/2075_Training Robust Ensembles Requires Rethinking Lipschitz Continuity/tables/cd98132fefb083fb307f31d053edce1922a02965b72594d62c831454d16eedd2.jpg)

![ce12e87f34143c0a9b6d75ab93e07c2c8b5ec566a8c6b27abd21b221eb4c333e.jpg](../iclr_results/2075_Training Robust Ensembles Requires Rethinking Lipschitz Continuity/tables/ce12e87f34143c0a9b6d75ab93e07c2c8b5ec566a8c6b27abd21b221eb4c333e.jpg)

![d6d0a9afc7213696ab2f7f4288b3b0916dbf19e7c93e6778cb52f6b1022977d7.jpg](../iclr_results/2075_Training Robust Ensembles Requires Rethinking Lipschitz Continuity/tables/d6d0a9afc7213696ab2f7f4288b3b0916dbf19e7c93e6778cb52f6b1022977d7.jpg)

![e9d693bf83622439bf3e617f4be7daba4804305009a106baa0766e7d14ef83ab.jpg](../iclr_results/2075_Training Robust Ensembles Requires Rethinking Lipschitz Continuity/tables/e9d693bf83622439bf3e617f4be7daba4804305009a106baa0766e7d14ef83ab.jpg)

![fe32e96e96fdb8676f63a59010673a6dd2b148ad844a4040ee9acbb519462ed2.jpg](../iclr_results/2075_Training Robust Ensembles Requires Rethinking Lipschitz Continuity/tables/fe32e96e96fdb8676f63a59010673a6dd2b148ad844a4040ee9acbb519462ed2.jpg)

## Does Spatial Cognition Emerge in Frontier Models?


### Images

![0062c75b2607c625b37960f11dd389b123a224b7b4de7114c4f593785fa7c44e.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/0062c75b2607c625b37960f11dd389b123a224b7b4de7114c4f593785fa7c44e.jpg)

![0e793db6d1dd6e6528ba1860bc9f81cebbaacd75d4c21f8b04cc8aec3ce18536.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/0e793db6d1dd6e6528ba1860bc9f81cebbaacd75d4c21f8b04cc8aec3ce18536.jpg)

![1a64ef3a210de1490c0119543008914d2f2f2b33639bb5962ce417879b243643.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/1a64ef3a210de1490c0119543008914d2f2f2b33639bb5962ce417879b243643.jpg)

![26f2c6a8f398cff5cfca45ace9b6f771f18b7fb3032847cdc6c163ac105b4cbf.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/26f2c6a8f398cff5cfca45ace9b6f771f18b7fb3032847cdc6c163ac105b4cbf.jpg)

![382dd5f47b0308f9f186fb535f1326368b61ad6a70868d4beedc718af40b2a61.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/382dd5f47b0308f9f186fb535f1326368b61ad6a70868d4beedc718af40b2a61.jpg)

![57c14f49c4422dddc89ffa87123833c9bf5dbcb3de38f6096748ca2e315f0101.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/57c14f49c4422dddc89ffa87123833c9bf5dbcb3de38f6096748ca2e315f0101.jpg)

![5aab26bc5f95c8f38f0c5933f7ef755a8f76035307422934375dce820224f884.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/5aab26bc5f95c8f38f0c5933f7ef755a8f76035307422934375dce820224f884.jpg)

![5b096113ab8828efb20efe3f489b0a01dce820534cfa7d486a09285cea381ebe.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/5b096113ab8828efb20efe3f489b0a01dce820534cfa7d486a09285cea381ebe.jpg)

![7085a2fc2894bca8a0e4b9a742572863bd6f18a4ad628b04eadd8bdd80cd51bb.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/7085a2fc2894bca8a0e4b9a742572863bd6f18a4ad628b04eadd8bdd80cd51bb.jpg)

![7d8104f0228566fe016039268592fb9dc4f6ea4da2e0113f05d7edd66305bb12.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/7d8104f0228566fe016039268592fb9dc4f6ea4da2e0113f05d7edd66305bb12.jpg)

![8415278901103e0762a5e686c46444939e1abe72c24b385b0d947defa80563fb.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/8415278901103e0762a5e686c46444939e1abe72c24b385b0d947defa80563fb.jpg)

![84e0c562cf02a8a7dd4e9c54aef8c25da50c95bde9cf222f20b33e868991492f.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/84e0c562cf02a8a7dd4e9c54aef8c25da50c95bde9cf222f20b33e868991492f.jpg)

![987f773cfc50065b7e3f2c036696cdafc7f3dce1062ff856495f5963f3e850e7.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/987f773cfc50065b7e3f2c036696cdafc7f3dce1062ff856495f5963f3e850e7.jpg)

![ab2320f066131e7fc76910cdf909d9805dcd25e9a8091f55a23b667ceab6aae2.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/ab2320f066131e7fc76910cdf909d9805dcd25e9a8091f55a23b667ceab6aae2.jpg)

![b0741e5de993526afc91cfe8542ed7f4b1e99a913663c6b46ca2f13216f3449c.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/b0741e5de993526afc91cfe8542ed7f4b1e99a913663c6b46ca2f13216f3449c.jpg)

![c07de14617b8ae7ec3ed2fea8c0f5a82051618b63e3db411e9dc55719cb79fa4.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/c07de14617b8ae7ec3ed2fea8c0f5a82051618b63e3db411e9dc55719cb79fa4.jpg)

![c2422127262c8b791d2190facd9a4c8e28c6560f0f6ce5ef01b072f394894ef1.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/c2422127262c8b791d2190facd9a4c8e28c6560f0f6ce5ef01b072f394894ef1.jpg)

![c818a479c442289416a08a31798ad0a2bbe24d5fb3144b35d1f3b67fa17e688f.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/c818a479c442289416a08a31798ad0a2bbe24d5fb3144b35d1f3b67fa17e688f.jpg)

![cb2f80c00205b9816b0d26f4a07fe4a8cb00c51f4f8512f9f236622583ef5daa.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/cb2f80c00205b9816b0d26f4a07fe4a8cb00c51f4f8512f9f236622583ef5daa.jpg)

![e499cb33677799ad7e06ab0e4e51fb7797c5aa7cf143eb8cffb04feab112e74c.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/e499cb33677799ad7e06ab0e4e51fb7797c5aa7cf143eb8cffb04feab112e74c.jpg)

![e942a09c5443580bce7e30b1f09e121f3789a61e180fddf21dfce69c3b84f4aa.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/e942a09c5443580bce7e30b1f09e121f3789a61e180fddf21dfce69c3b84f4aa.jpg)

![ee7727f77b23ae35fa53bd36f369c8e8de0fe13505e36f5cb9e9e42e2af16e8d.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/ee7727f77b23ae35fa53bd36f369c8e8de0fe13505e36f5cb9e9e42e2af16e8d.jpg)

![fcdac41823d2590504534734926605b74540e3d3ba646a4cc5d033f64fcda150.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/fcdac41823d2590504534734926605b74540e3d3ba646a4cc5d033f64fcda150.jpg)

![fd1e93c694164bc9a78641e4d858a030df01f125337d2cb384001bde1f6cfd00.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/images/fd1e93c694164bc9a78641e4d858a030df01f125337d2cb384001bde1f6cfd00.jpg)

### Tables

![003d1f26fe7b66692343a76137399d35844ba563762a12e68ec33354b50ec269.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/tables/003d1f26fe7b66692343a76137399d35844ba563762a12e68ec33354b50ec269.jpg)

![139b6fb0be6f531b9cd8ba46d11f13947ebf2436e95a336be60b016cbac2b063.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/tables/139b6fb0be6f531b9cd8ba46d11f13947ebf2436e95a336be60b016cbac2b063.jpg)

![2d808ca72c506c44c7a8c7eb5a35126b7cc08d5e2d4ece1c2a62e81c94bdf009.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/tables/2d808ca72c506c44c7a8c7eb5a35126b7cc08d5e2d4ece1c2a62e81c94bdf009.jpg)

![48e1aeaef8659409af177c8ba79d07a96d80544f88f49ed9a14b458f2af3d28d.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/tables/48e1aeaef8659409af177c8ba79d07a96d80544f88f49ed9a14b458f2af3d28d.jpg)

![96e58f6cfe906e0d4ebc483bd1fb44a3b41ad2161a0e0cf24c948fb692ed5973.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/tables/96e58f6cfe906e0d4ebc483bd1fb44a3b41ad2161a0e0cf24c948fb692ed5973.jpg)

![a97fc20c09719f0beada4531807f2635f8b47d9dc2eff4a0a82f2b2d0027e33a.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/tables/a97fc20c09719f0beada4531807f2635f8b47d9dc2eff4a0a82f2b2d0027e33a.jpg)

![a987d28fd1e049f1b3eb9163825b902ab111b985a6937fc97618046eeecf0f56.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/tables/a987d28fd1e049f1b3eb9163825b902ab111b985a6937fc97618046eeecf0f56.jpg)

![b421455d0ba28954d56637df97da98a2239594d19ad9f83560559048f5ee76cf.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/tables/b421455d0ba28954d56637df97da98a2239594d19ad9f83560559048f5ee76cf.jpg)

![fa99f2e4ef377cc970a371aba8db7736c8497226c631ca7feb40fed8e35298a8.jpg](../iclr_results/2076_Does Spatial Cognition Emerge in Frontier Models_/tables/fa99f2e4ef377cc970a371aba8db7736c8497226c631ca7feb40fed8e35298a8.jpg)

## Learning Multi-Index Models with Neural Networks via Mean-Field Langevin Dynamics


### Images

![73529a0015fa004011637d09ccfa0e42e2b6f317c69fb69bacaa0c1f4428c9a9.jpg](../iclr_results/2077_Learning Multi-Index Models with Neural Networks via Mean-Field Langevin Dynamics/images/73529a0015fa004011637d09ccfa0e42e2b6f317c69fb69bacaa0c1f4428c9a9.jpg)

![74f9125fe3af4f8f1a806ea5db99c30985c19a874e7c0877c05640cedf5e4f6f.jpg](../iclr_results/2077_Learning Multi-Index Models with Neural Networks via Mean-Field Langevin Dynamics/images/74f9125fe3af4f8f1a806ea5db99c30985c19a874e7c0877c05640cedf5e4f6f.jpg)

### Tables

![11211cee697565eece3bf24bc0484303291ae85d6f7de6aef0c4a09e32778c69.jpg](../iclr_results/2077_Learning Multi-Index Models with Neural Networks via Mean-Field Langevin Dynamics/tables/11211cee697565eece3bf24bc0484303291ae85d6f7de6aef0c4a09e32778c69.jpg)

## Rotated Runtime Smooth: Training-Free Activation Smoother for accurate INT4 inference


### Images

![0ff41283e078e9c00d55cb4d33562c448dd35ae83520a377f6623577f7997012.jpg](../iclr_results/2078_Rotated Runtime Smooth_ Training-Free Activation Smoother for accurate INT4 inference/images/0ff41283e078e9c00d55cb4d33562c448dd35ae83520a377f6623577f7997012.jpg)

![470f51f83f33991785137b5da825b18ed92e40f0b4e7708164484471ef8dff0b.jpg](../iclr_results/2078_Rotated Runtime Smooth_ Training-Free Activation Smoother for accurate INT4 inference/images/470f51f83f33991785137b5da825b18ed92e40f0b4e7708164484471ef8dff0b.jpg)

![485b0f4cf99bd6ffb2e74473abe5d628e747d8d0783cec580527e7befb720082.jpg](../iclr_results/2078_Rotated Runtime Smooth_ Training-Free Activation Smoother for accurate INT4 inference/images/485b0f4cf99bd6ffb2e74473abe5d628e747d8d0783cec580527e7befb720082.jpg)

![533fb48b55a79d789c4cc772d29f2136e329c4d09d6a985c7add74aba410963a.jpg](../iclr_results/2078_Rotated Runtime Smooth_ Training-Free Activation Smoother for accurate INT4 inference/images/533fb48b55a79d789c4cc772d29f2136e329c4d09d6a985c7add74aba410963a.jpg)

![60524a91f2fe8f666638a94b0508ed005d907be7b97a90e8aea090dda63c52aa.jpg](../iclr_results/2078_Rotated Runtime Smooth_ Training-Free Activation Smoother for accurate INT4 inference/images/60524a91f2fe8f666638a94b0508ed005d907be7b97a90e8aea090dda63c52aa.jpg)

![7b43dca8a6da711a081df2f5ad45231aba7e0926659b1e31e431aef538480d34.jpg](../iclr_results/2078_Rotated Runtime Smooth_ Training-Free Activation Smoother for accurate INT4 inference/images/7b43dca8a6da711a081df2f5ad45231aba7e0926659b1e31e431aef538480d34.jpg)

![98b2b6e4d1153354997224a10f5bba795f001b49009b6eed51990a367cdf0cbd.jpg](../iclr_results/2078_Rotated Runtime Smooth_ Training-Free Activation Smoother for accurate INT4 inference/images/98b2b6e4d1153354997224a10f5bba795f001b49009b6eed51990a367cdf0cbd.jpg)

![c1bcc419ea816265b392c90927c93243c1d2396e9185981c787dfde815f09b44.jpg](../iclr_results/2078_Rotated Runtime Smooth_ Training-Free Activation Smoother for accurate INT4 inference/images/c1bcc419ea816265b392c90927c93243c1d2396e9185981c787dfde815f09b44.jpg)

![c2d96496e8cdc2e4a149ac5a9d57eee08323780de4ec5da1995866e818a75bfa.jpg](../iclr_results/2078_Rotated Runtime Smooth_ Training-Free Activation Smoother for accurate INT4 inference/images/c2d96496e8cdc2e4a149ac5a9d57eee08323780de4ec5da1995866e818a75bfa.jpg)

### Tables

![7e0683f01c2eb126f62ebbad2be9862e28a6215751efbfb2f220e9dc40aa4a77.jpg](../iclr_results/2078_Rotated Runtime Smooth_ Training-Free Activation Smoother for accurate INT4 inference/tables/7e0683f01c2eb126f62ebbad2be9862e28a6215751efbfb2f220e9dc40aa4a77.jpg)

![96a1c53c38d3b37a81ffa65687dffbcca10090d84f34f164dffbf0eb9fb29871.jpg](../iclr_results/2078_Rotated Runtime Smooth_ Training-Free Activation Smoother for accurate INT4 inference/tables/96a1c53c38d3b37a81ffa65687dffbcca10090d84f34f164dffbf0eb9fb29871.jpg)

![a44df5d758d7157622157fe8605f183936de9fccf8bb4efab68bbe5f012dbbcb.jpg](../iclr_results/2078_Rotated Runtime Smooth_ Training-Free Activation Smoother for accurate INT4 inference/tables/a44df5d758d7157622157fe8605f183936de9fccf8bb4efab68bbe5f012dbbcb.jpg)

![ce0ef17aaff961ceb1b123dd3c40f36f257e14dc80954ba7881bbfd82454221e.jpg](../iclr_results/2078_Rotated Runtime Smooth_ Training-Free Activation Smoother for accurate INT4 inference/tables/ce0ef17aaff961ceb1b123dd3c40f36f257e14dc80954ba7881bbfd82454221e.jpg)

![e4b63741a05e54f27e98858ff0f01c493d5f5f6ba59fd7ed2bea9c190c59e038.jpg](../iclr_results/2078_Rotated Runtime Smooth_ Training-Free Activation Smoother for accurate INT4 inference/tables/e4b63741a05e54f27e98858ff0f01c493d5f5f6ba59fd7ed2bea9c190c59e038.jpg)

![ed7ed999e3314d6b6563eb38931234a360692eec7abd489d22307d139e57c06a.jpg](../iclr_results/2078_Rotated Runtime Smooth_ Training-Free Activation Smoother for accurate INT4 inference/tables/ed7ed999e3314d6b6563eb38931234a360692eec7abd489d22307d139e57c06a.jpg)

![ee1e9745e3ff12f36929ab53dbedb1086c30a1fb4fbb4166de1d4947c9b8266a.jpg](../iclr_results/2078_Rotated Runtime Smooth_ Training-Free Activation Smoother for accurate INT4 inference/tables/ee1e9745e3ff12f36929ab53dbedb1086c30a1fb4fbb4166de1d4947c9b8266a.jpg)

## Multi-Task Corrupted Prediction for Learning Robust Audio-Visual Speech Representation


### Images

![01d026d094e9f402f990ad597b914547d05d03a1bdf7231197aa8bfbb5a7cef8.jpg](../iclr_results/2079_Multi-Task Corrupted Prediction for Learning Robust Audio-Visual Speech Representation/images/01d026d094e9f402f990ad597b914547d05d03a1bdf7231197aa8bfbb5a7cef8.jpg)

![2fdb8e9d8a5e8a1e9a53d0a1ecdf95a0050012498c5d1a68f7b6053003ee859a.jpg](../iclr_results/2079_Multi-Task Corrupted Prediction for Learning Robust Audio-Visual Speech Representation/images/2fdb8e9d8a5e8a1e9a53d0a1ecdf95a0050012498c5d1a68f7b6053003ee859a.jpg)

![5d93a5e32c13016b15eea84371a376977dd55d724ac917e8e01a8dce787805ae.jpg](../iclr_results/2079_Multi-Task Corrupted Prediction for Learning Robust Audio-Visual Speech Representation/images/5d93a5e32c13016b15eea84371a376977dd55d724ac917e8e01a8dce787805ae.jpg)

![68b2fb4cf52261591a1932d088f3fbe37fcee170ca4afd780da1bd0bee9ef6e3.jpg](../iclr_results/2079_Multi-Task Corrupted Prediction for Learning Robust Audio-Visual Speech Representation/images/68b2fb4cf52261591a1932d088f3fbe37fcee170ca4afd780da1bd0bee9ef6e3.jpg)

![f7be8e182c89db8da585407a27a0bad392b476441f833f84315b75e22af80792.jpg](../iclr_results/2079_Multi-Task Corrupted Prediction for Learning Robust Audio-Visual Speech Representation/images/f7be8e182c89db8da585407a27a0bad392b476441f833f84315b75e22af80792.jpg)

![ffe0ba326b0348617cff36aab5a87c63e53123514160bc3bb0ee6ac68a616b0e.jpg](../iclr_results/2079_Multi-Task Corrupted Prediction for Learning Robust Audio-Visual Speech Representation/images/ffe0ba326b0348617cff36aab5a87c63e53123514160bc3bb0ee6ac68a616b0e.jpg)

### Tables

![006e698a6ceb410dc8b8bc40ca0f0c71e5ac12dd5eb32e6a1164a58451a9149b.jpg](../iclr_results/2079_Multi-Task Corrupted Prediction for Learning Robust Audio-Visual Speech Representation/tables/006e698a6ceb410dc8b8bc40ca0f0c71e5ac12dd5eb32e6a1164a58451a9149b.jpg)

![01bf1b7ba87a07f2abb45e9090ae373de723a5c308dcfd8edb2617ddbeeae21e.jpg](../iclr_results/2079_Multi-Task Corrupted Prediction for Learning Robust Audio-Visual Speech Representation/tables/01bf1b7ba87a07f2abb45e9090ae373de723a5c308dcfd8edb2617ddbeeae21e.jpg)

![21cb756c23a8cc4a0e2b6004f3ad2e6617720d9f9d07a48b3b92afbda566f815.jpg](../iclr_results/2079_Multi-Task Corrupted Prediction for Learning Robust Audio-Visual Speech Representation/tables/21cb756c23a8cc4a0e2b6004f3ad2e6617720d9f9d07a48b3b92afbda566f815.jpg)

![2363be98999d935fe537ceaf577d4ebecdbedd6e6186a095eee165435eaa0f19.jpg](../iclr_results/2079_Multi-Task Corrupted Prediction for Learning Robust Audio-Visual Speech Representation/tables/2363be98999d935fe537ceaf577d4ebecdbedd6e6186a095eee165435eaa0f19.jpg)

![269891c4e8217e3005600df2554bdaea506079f11f737563929af35693965a78.jpg](../iclr_results/2079_Multi-Task Corrupted Prediction for Learning Robust Audio-Visual Speech Representation/tables/269891c4e8217e3005600df2554bdaea506079f11f737563929af35693965a78.jpg)

![5b8962f21d1c92e5f4e5f447188b7c8ae65bd7254fef3cd7d5b68b041154b74a.jpg](../iclr_results/2079_Multi-Task Corrupted Prediction for Learning Robust Audio-Visual Speech Representation/tables/5b8962f21d1c92e5f4e5f447188b7c8ae65bd7254fef3cd7d5b68b041154b74a.jpg)

![64d748f9aa4639e8e3b75e247bbaf7abe075e6cbaf4a2017209d6a5570484825.jpg](../iclr_results/2079_Multi-Task Corrupted Prediction for Learning Robust Audio-Visual Speech Representation/tables/64d748f9aa4639e8e3b75e247bbaf7abe075e6cbaf4a2017209d6a5570484825.jpg)

![766928524f451cf8252bbab56a63afa68b13afd0c9e563fb984c7d89c63ff27a.jpg](../iclr_results/2079_Multi-Task Corrupted Prediction for Learning Robust Audio-Visual Speech Representation/tables/766928524f451cf8252bbab56a63afa68b13afd0c9e563fb984c7d89c63ff27a.jpg)

![af5b81c550490bac77d6f86a2976d41dc95ec0ef168af320795fb5bbc80e8178.jpg](../iclr_results/2079_Multi-Task Corrupted Prediction for Learning Robust Audio-Visual Speech Representation/tables/af5b81c550490bac77d6f86a2976d41dc95ec0ef168af320795fb5bbc80e8178.jpg)

![b3e1be77c94d03fca00ad1e6eff6e79a2e2877948b8398e2cd293cceaaab0299.jpg](../iclr_results/2079_Multi-Task Corrupted Prediction for Learning Robust Audio-Visual Speech Representation/tables/b3e1be77c94d03fca00ad1e6eff6e79a2e2877948b8398e2cd293cceaaab0299.jpg)

![b8b463802b5db335330307d2f20d883013c698db069c7175136c407633eba1c9.jpg](../iclr_results/2079_Multi-Task Corrupted Prediction for Learning Robust Audio-Visual Speech Representation/tables/b8b463802b5db335330307d2f20d883013c698db069c7175136c407633eba1c9.jpg)

![cc981954e11b03cc8487bcbc85182b3e4df089e574f8167d822b8b0803e9e431.jpg](../iclr_results/2079_Multi-Task Corrupted Prediction for Learning Robust Audio-Visual Speech Representation/tables/cc981954e11b03cc8487bcbc85182b3e4df089e574f8167d822b8b0803e9e431.jpg)

![f4ac18a0e14e70df5653c61f30882460f331f982a1825a426d756e3d67db3065.jpg](../iclr_results/2079_Multi-Task Corrupted Prediction for Learning Robust Audio-Visual Speech Representation/tables/f4ac18a0e14e70df5653c61f30882460f331f982a1825a426d756e3d67db3065.jpg)

![f7f801c470ca68e681d32cab8ad24071c85d704b36622fcf51756ff509d5c70d.jpg](../iclr_results/2079_Multi-Task Corrupted Prediction for Learning Robust Audio-Visual Speech Representation/tables/f7f801c470ca68e681d32cab8ad24071c85d704b36622fcf51756ff509d5c70d.jpg)

## Distributional Associations vs In-Context Reasoning: A Study of Feed-forward and Attention Layers


### Images

![1f3157ce85df158afee1c2075bcf75ee272a0eb26f1e3f2edb884ab01faead97.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/images/1f3157ce85df158afee1c2075bcf75ee272a0eb26f1e3f2edb884ab01faead97.jpg)

![2b5ad663aa6c091958839ba87178ec03cea9639cbfa842bacbe86b4b27e4d48d.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/images/2b5ad663aa6c091958839ba87178ec03cea9639cbfa842bacbe86b4b27e4d48d.jpg)

![31b1f5ebe89d1ca8dbd6acbfc586de6665c546da8c421abd7b10509f90fc161d.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/images/31b1f5ebe89d1ca8dbd6acbfc586de6665c546da8c421abd7b10509f90fc161d.jpg)

![36bd8c28779b4f5db72ce38bb82d8e132446c2f40204af014717d9401f26cdb5.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/images/36bd8c28779b4f5db72ce38bb82d8e132446c2f40204af014717d9401f26cdb5.jpg)

![3d7e2c1ae453e64d5806714349301bf054955c3632ab1a2e400117e6ffb13aa1.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/images/3d7e2c1ae453e64d5806714349301bf054955c3632ab1a2e400117e6ffb13aa1.jpg)

![55c354ae54df6104b27b45491daec1d9f8954441dd061f4fd2556a469a6cb6f0.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/images/55c354ae54df6104b27b45491daec1d9f8954441dd061f4fd2556a469a6cb6f0.jpg)

![5f20c25542a8d58226fc3cf9f12c04b4ec55a49600bae18052d06b79d3456d5b.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/images/5f20c25542a8d58226fc3cf9f12c04b4ec55a49600bae18052d06b79d3456d5b.jpg)

![64ed3be1a82f2f5dea20de094b96d542186adf63b84cfd7ef8a562bf677577d1.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/images/64ed3be1a82f2f5dea20de094b96d542186adf63b84cfd7ef8a562bf677577d1.jpg)

![6bf160713a43b20502f1c452f65f7620f54b45e0b96ece1a34964ad15c11d5d7.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/images/6bf160713a43b20502f1c452f65f7620f54b45e0b96ece1a34964ad15c11d5d7.jpg)

![8e9001ae4c987c734c5b0913567965c0fa923d6122ceeeac00fb82d635a348ed.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/images/8e9001ae4c987c734c5b0913567965c0fa923d6122ceeeac00fb82d635a348ed.jpg)

![b4fa05c80e3ce123c83c6e9ed11a2d042956a613c4c9a6ecbce7a93d27d23b25.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/images/b4fa05c80e3ce123c83c6e9ed11a2d042956a613c4c9a6ecbce7a93d27d23b25.jpg)

![bb69a758285039d3e625dd945ca4071f229dc4bc650f3a2b4503522e0d056a16.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/images/bb69a758285039d3e625dd945ca4071f229dc4bc650f3a2b4503522e0d056a16.jpg)

![be4894c835bd35975766e39eeff69d44ae3faefdf912a9968e1819a6d1894785.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/images/be4894c835bd35975766e39eeff69d44ae3faefdf912a9968e1819a6d1894785.jpg)

![c40ad003886e4d27e9af7ec91971a5cabebe7b3c73f250968ae16a279300338d.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/images/c40ad003886e4d27e9af7ec91971a5cabebe7b3c73f250968ae16a279300338d.jpg)

![cd3f5074f17b492923c6dd7cbfb748adb5dc9a6e241d4092a111cf5645c4a5f0.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/images/cd3f5074f17b492923c6dd7cbfb748adb5dc9a6e241d4092a111cf5645c4a5f0.jpg)

![cd7abeb129e9d01de191523f8cf0a26dfeb63cf967fd51c3c1c17b4c6d5599c9.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/images/cd7abeb129e9d01de191523f8cf0a26dfeb63cf967fd51c3c1c17b4c6d5599c9.jpg)

![dfbeedcca2508b64d2f1d6a1a82232af87bf7a39fb746a3aefa02522d7afc978.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/images/dfbeedcca2508b64d2f1d6a1a82232af87bf7a39fb746a3aefa02522d7afc978.jpg)

![e39ec01846f920016beb7c8d459d5cbf66bf3199bc40438b240da91f89c8fff0.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/images/e39ec01846f920016beb7c8d459d5cbf66bf3199bc40438b240da91f89c8fff0.jpg)

![f4466b3a5e9cd7b47c29b87b7bf4e67dc44b6442d6b5c7b2714253c40256b6ff.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/images/f4466b3a5e9cd7b47c29b87b7bf4e67dc44b6442d6b5c7b2714253c40256b6ff.jpg)

![f7d9d8a81d248ae0aecf75cd186127dd6766a34c98fbd562dc0637a37ba7a309.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/images/f7d9d8a81d248ae0aecf75cd186127dd6766a34c98fbd562dc0637a37ba7a309.jpg)

### Tables

![64bb376d2220d70ef2f9b0be69b54d2d94245a1397d8e07e7963653dce20f70b.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/tables/64bb376d2220d70ef2f9b0be69b54d2d94245a1397d8e07e7963653dce20f70b.jpg)

![6a037ceb7b1bda12f847a50833b2c3fdc656e46878bb0d30c39b7cf9da26dead.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/tables/6a037ceb7b1bda12f847a50833b2c3fdc656e46878bb0d30c39b7cf9da26dead.jpg)

![7b89ef5301e38d71a225e6f0daccd68bcfd5babe4e252e7247a7d300a21a8a8f.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/tables/7b89ef5301e38d71a225e6f0daccd68bcfd5babe4e252e7247a7d300a21a8a8f.jpg)

![886306f69f5fb10ef38ad4f3c62de4d8ab0e5ada5c4cc29621da9044d73c37d5.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/tables/886306f69f5fb10ef38ad4f3c62de4d8ab0e5ada5c4cc29621da9044d73c37d5.jpg)

![b472ab480f0d97c700fd29852f66f7569817fda875a0aa7836bfe80810781a0e.jpg](../iclr_results/2080_Distributional Associations vs In-Context Reasoning_ A Study of Feed-forward and Attention Layers/tables/b472ab480f0d97c700fd29852f66f7569817fda875a0aa7836bfe80810781a0e.jpg)

## Salvage: Shapley-distribution Approximation Learning Via Attribution Guided Exploration for Explainable Image Classification


### Images

![1091f05930f89d68df8de2eef0ae560968b2ce53213c16697eabf2b34a8fc489.jpg](../iclr_results/2081_Salvage_ Shapley-distribution Approximation Learning Via Attribution Guided Exploration for Explaina/images/1091f05930f89d68df8de2eef0ae560968b2ce53213c16697eabf2b34a8fc489.jpg)

![174a8d2a63b39c8cfa596a647043b302fd0d8af9c038f53915ec4a73e55dee0c.jpg](../iclr_results/2081_Salvage_ Shapley-distribution Approximation Learning Via Attribution Guided Exploration for Explaina/images/174a8d2a63b39c8cfa596a647043b302fd0d8af9c038f53915ec4a73e55dee0c.jpg)

![2f700e0317ba9c8ac85e7b35ff59ff1cc25eb509e7aadac4f840b66306464710.jpg](../iclr_results/2081_Salvage_ Shapley-distribution Approximation Learning Via Attribution Guided Exploration for Explaina/images/2f700e0317ba9c8ac85e7b35ff59ff1cc25eb509e7aadac4f840b66306464710.jpg)

![4fb3ea611c142f4e53a6ff963ec5a445bbf27e7f1d94f40e0dae357562b87407.jpg](../iclr_results/2081_Salvage_ Shapley-distribution Approximation Learning Via Attribution Guided Exploration for Explaina/images/4fb3ea611c142f4e53a6ff963ec5a445bbf27e7f1d94f40e0dae357562b87407.jpg)

![69048a05696d69353fe0cfa8aa3a2a30466466fe2a6b97d76c1ecbb65b5d6bbe.jpg](../iclr_results/2081_Salvage_ Shapley-distribution Approximation Learning Via Attribution Guided Exploration for Explaina/images/69048a05696d69353fe0cfa8aa3a2a30466466fe2a6b97d76c1ecbb65b5d6bbe.jpg)

![c8e8a4350a6346c4c713478cb23edef1c12146456764654fd95c69cb5ee92b40.jpg](../iclr_results/2081_Salvage_ Shapley-distribution Approximation Learning Via Attribution Guided Exploration for Explaina/images/c8e8a4350a6346c4c713478cb23edef1c12146456764654fd95c69cb5ee92b40.jpg)

![cc08e015f5fca104afc2014e644a48cd0a96322c002353454608a57049079f6f.jpg](../iclr_results/2081_Salvage_ Shapley-distribution Approximation Learning Via Attribution Guided Exploration for Explaina/images/cc08e015f5fca104afc2014e644a48cd0a96322c002353454608a57049079f6f.jpg)

![de6516b821513d18cf1d4d8f87e57dc73b76d29e6c8d4d455c671228022cd825.jpg](../iclr_results/2081_Salvage_ Shapley-distribution Approximation Learning Via Attribution Guided Exploration for Explaina/images/de6516b821513d18cf1d4d8f87e57dc73b76d29e6c8d4d455c671228022cd825.jpg)

![ff2e64e6b7d7d85b9ea853b024b19e3d7f1e1d687917d95c044f7accfb445e1b.jpg](../iclr_results/2081_Salvage_ Shapley-distribution Approximation Learning Via Attribution Guided Exploration for Explaina/images/ff2e64e6b7d7d85b9ea853b024b19e3d7f1e1d687917d95c044f7accfb445e1b.jpg)

### Tables

![039446126539e0e9d43864ab04b67518cde772606289dffb5b0790c61d2c3d83.jpg](../iclr_results/2081_Salvage_ Shapley-distribution Approximation Learning Via Attribution Guided Exploration for Explaina/tables/039446126539e0e9d43864ab04b67518cde772606289dffb5b0790c61d2c3d83.jpg)

![314e90664fa8a5a516a1e04e429cd421133d2b7ce4415c552a78dbe262335069.jpg](../iclr_results/2081_Salvage_ Shapley-distribution Approximation Learning Via Attribution Guided Exploration for Explaina/tables/314e90664fa8a5a516a1e04e429cd421133d2b7ce4415c552a78dbe262335069.jpg)

![36d8ba5b23bfec04c60f4d6582888d172b25b7f064a60cbadb07023a88b3e4da.jpg](../iclr_results/2081_Salvage_ Shapley-distribution Approximation Learning Via Attribution Guided Exploration for Explaina/tables/36d8ba5b23bfec04c60f4d6582888d172b25b7f064a60cbadb07023a88b3e4da.jpg)

![52272474ad1fb842b4b95bb2f6a78912eda41cb6d8bfbbcb6d07172aa0287577.jpg](../iclr_results/2081_Salvage_ Shapley-distribution Approximation Learning Via Attribution Guided Exploration for Explaina/tables/52272474ad1fb842b4b95bb2f6a78912eda41cb6d8bfbbcb6d07172aa0287577.jpg)

## PivotMesh: Generic 3D Mesh Generation via Pivot Vertices Guidance


### Images

![0b44055aa12d7c1e94446e1870f90cb19c04d96f9e7e4b3474baf48bc2a2bd1b.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/images/0b44055aa12d7c1e94446e1870f90cb19c04d96f9e7e4b3474baf48bc2a2bd1b.jpg)

![16c6d5898c1f38a8bc783898d093bb6e6865b23af273a20a41ed9372eba46215.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/images/16c6d5898c1f38a8bc783898d093bb6e6865b23af273a20a41ed9372eba46215.jpg)

![5759a741b0ec37b6c5ca035a7502d8b353b18ed817b79be2c58e6ca16d210d38.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/images/5759a741b0ec37b6c5ca035a7502d8b353b18ed817b79be2c58e6ca16d210d38.jpg)

![581b7edf3a6f1f3e035ca7cee848b907f8818075e008d05ac31c44d4ee88dd5e.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/images/581b7edf3a6f1f3e035ca7cee848b907f8818075e008d05ac31c44d4ee88dd5e.jpg)

![5fc06b320afd3ba9766156e19491f956429fd14f077d70ba60f9e540314361af.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/images/5fc06b320afd3ba9766156e19491f956429fd14f077d70ba60f9e540314361af.jpg)

![6768af9413589f61aca7f1dcb20e6a70dd2455f64cd7a6bd4f2d24c853a4a0ba.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/images/6768af9413589f61aca7f1dcb20e6a70dd2455f64cd7a6bd4f2d24c853a4a0ba.jpg)

![8b80e7625d746750af65b8ca6131f68783adca0bd66e7e22e5272ea0d3c032bc.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/images/8b80e7625d746750af65b8ca6131f68783adca0bd66e7e22e5272ea0d3c032bc.jpg)

![8dbca4a1c805951798d7477449d1f49ed646860a75898a01f6410727a11bff6b.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/images/8dbca4a1c805951798d7477449d1f49ed646860a75898a01f6410727a11bff6b.jpg)

![965045bb6ac76aab39ff74cb54123c1c743ecc50e8537e449887c108f91dbd04.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/images/965045bb6ac76aab39ff74cb54123c1c743ecc50e8537e449887c108f91dbd04.jpg)

![982dada3c34940567f1856c8f3c060026706b1b503f2cf3ab850c628dfd78e2d.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/images/982dada3c34940567f1856c8f3c060026706b1b503f2cf3ab850c628dfd78e2d.jpg)

![9cff6a05a24ce226f6c7a5ad1c04763e547db61c59084a84e5320eabbee8e94b.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/images/9cff6a05a24ce226f6c7a5ad1c04763e547db61c59084a84e5320eabbee8e94b.jpg)

![a4df22e7edd7bc7f057c75eaabe6ff2c1528969c0d0da798d08837870c0d5634.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/images/a4df22e7edd7bc7f057c75eaabe6ff2c1528969c0d0da798d08837870c0d5634.jpg)

![af365d97676bafbeff7512322db75cd793a33df401fa863fa5fab7803a066369.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/images/af365d97676bafbeff7512322db75cd793a33df401fa863fa5fab7803a066369.jpg)

![b465082ca35502693667a993c436819fa81ca1e6b47eb3303477453c587c91d3.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/images/b465082ca35502693667a993c436819fa81ca1e6b47eb3303477453c587c91d3.jpg)

![b98dd2cc58db9454a3c792a1844bfe61501809eb30aded3e6151ff2eeb6f87dc.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/images/b98dd2cc58db9454a3c792a1844bfe61501809eb30aded3e6151ff2eeb6f87dc.jpg)

![bc8261c55e16c73a1f3fa975aca6db727773ff0c387f429fc035496a55b882e3.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/images/bc8261c55e16c73a1f3fa975aca6db727773ff0c387f429fc035496a55b882e3.jpg)

![d5dff6f2b12e21637f23e84c4111489a32b39d97901b30ce5c7a69b96daf2905.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/images/d5dff6f2b12e21637f23e84c4111489a32b39d97901b30ce5c7a69b96daf2905.jpg)

![da8427558122596c73b84fe63fda0971d70ed8ef60b7f7a9f6ca032b2920b628.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/images/da8427558122596c73b84fe63fda0971d70ed8ef60b7f7a9f6ca032b2920b628.jpg)

![fe9a3bc9bc4e517f0ae05a26b0f63470f54e5dc0f6c558f28e66e570ee8ad99c.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/images/fe9a3bc9bc4e517f0ae05a26b0f63470f54e5dc0f6c558f28e66e570ee8ad99c.jpg)

### Tables

![06dc977d79f169f4b5ffc3d619699226e6138bdfda565c53559e8371e04b90b4.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/tables/06dc977d79f169f4b5ffc3d619699226e6138bdfda565c53559e8371e04b90b4.jpg)

![341db67089803dd2df0f01fb65f6eac909685502818d12b8d8defcca392ab20e.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/tables/341db67089803dd2df0f01fb65f6eac909685502818d12b8d8defcca392ab20e.jpg)

![5e913ab0ee3fe9ab872b65ffbbfae908f7d6bcf8b84cbecf8970d754ce42ac4c.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/tables/5e913ab0ee3fe9ab872b65ffbbfae908f7d6bcf8b84cbecf8970d754ce42ac4c.jpg)

![6e92d9a72143ebdeee6bdfcb60e817a117cacf3218b9bade7226d9e75715b615.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/tables/6e92d9a72143ebdeee6bdfcb60e817a117cacf3218b9bade7226d9e75715b615.jpg)

![b5a24f99e152bfbc8c208c5b0bc4a6c2fdcbf2cfcd9db452849bbbf9721c2299.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/tables/b5a24f99e152bfbc8c208c5b0bc4a6c2fdcbf2cfcd9db452849bbbf9721c2299.jpg)

![d13805e0950aa000f2ff7dcbde161f1b118a704201624441c87347fece7c5f43.jpg](../iclr_results/2082_PivotMesh_ Generic 3D Mesh Generation via Pivot Vertices Guidance/tables/d13805e0950aa000f2ff7dcbde161f1b118a704201624441c87347fece7c5f43.jpg)

## Adaptive Pruning of Pretrained Transformer via Differential Inclusions

### Images

![2304f9aa1819ff447effefd84df108e0a357a2291c18e1489a74317eca7a55a4.jpg](../iclr_results/2083_Adaptive Pruning of Pretrained Transformer via Differential Inclusions/images/2304f9aa1819ff447effefd84df108e0a357a2291c18e1489a74317eca7a55a4.jpg)

![b4ba38c877665ba34dca833dd250b98e24d57d1cecd7fbe1a173c30215c11f43.jpg](../iclr_results/2083_Adaptive Pruning of Pretrained Transformer via Differential Inclusions/images/b4ba38c877665ba34dca833dd250b98e24d57d1cecd7fbe1a173c30215c11f43.jpg)

![f008d66b2a604f8342488e7b24e52a4d9f1f6d9cf79eef2cbd436ab84a28e530.jpg](../iclr_results/2083_Adaptive Pruning of Pretrained Transformer via Differential Inclusions/images/f008d66b2a604f8342488e7b24e52a4d9f1f6d9cf79eef2cbd436ab84a28e530.jpg)

![f01d68fb294f96525423c529f29c9917119755e1e97988f9df19b55c366aa926.jpg](../iclr_results/2083_Adaptive Pruning of Pretrained Transformer via Differential Inclusions/images/f01d68fb294f96525423c529f29c9917119755e1e97988f9df19b55c366aa926.jpg)

### Tables

![1665c8121e5490e42eb5e6e9c6f12df9b970bed5e7ba61eeb429bba8486ccd6c.jpg](../iclr_results/2083_Adaptive Pruning of Pretrained Transformer via Differential Inclusions/tables/1665c8121e5490e42eb5e6e9c6f12df9b970bed5e7ba61eeb429bba8486ccd6c.jpg)

![26db346fec8f6e58fdea2dde23db92b55168b04ed71741e0487d342ce7843611.jpg](../iclr_results/2083_Adaptive Pruning of Pretrained Transformer via Differential Inclusions/tables/26db346fec8f6e58fdea2dde23db92b55168b04ed71741e0487d342ce7843611.jpg)

![2ea1b2c339b40475fd55459fe3c2ff2a178e14a1460d32eea81290f980c9becb.jpg](../iclr_results/2083_Adaptive Pruning of Pretrained Transformer via Differential Inclusions/tables/2ea1b2c339b40475fd55459fe3c2ff2a178e14a1460d32eea81290f980c9becb.jpg)

![365f28f085106b5c66e7269e63c7384a8f7ba4934afcd5312dd09f550d30a61b.jpg](../iclr_results/2083_Adaptive Pruning of Pretrained Transformer via Differential Inclusions/tables/365f28f085106b5c66e7269e63c7384a8f7ba4934afcd5312dd09f550d30a61b.jpg)

![3c6995918b57e5d0a522a72fa05f31bcd8f3eef1f6e96e70ad9a1c42418453ef.jpg](../iclr_results/2083_Adaptive Pruning of Pretrained Transformer via Differential Inclusions/tables/3c6995918b57e5d0a522a72fa05f31bcd8f3eef1f6e96e70ad9a1c42418453ef.jpg)

![614b6357ff98891c51743e1a369b0408084968e0855c52209e0832b212ac21b7.jpg](../iclr_results/2083_Adaptive Pruning of Pretrained Transformer via Differential Inclusions/tables/614b6357ff98891c51743e1a369b0408084968e0855c52209e0832b212ac21b7.jpg)

![63e28c4b2c4565e7cfedca30d5891a4f5e4f842e3f4503f004535bfdcbaae99c.jpg](../iclr_results/2083_Adaptive Pruning of Pretrained Transformer via Differential Inclusions/tables/63e28c4b2c4565e7cfedca30d5891a4f5e4f842e3f4503f004535bfdcbaae99c.jpg)

![755eb01a892df212e2b0604cb07d2a6d74f970c7d466cd421eb090eaccab6f12.jpg](../iclr_results/2083_Adaptive Pruning of Pretrained Transformer via Differential Inclusions/tables/755eb01a892df212e2b0604cb07d2a6d74f970c7d466cd421eb090eaccab6f12.jpg)

![90975f3ebec9c07ae8c48f651d4e0190f2d22fac66243e385d415aa78665cf03.jpg](../iclr_results/2083_Adaptive Pruning of Pretrained Transformer via Differential Inclusions/tables/90975f3ebec9c07ae8c48f651d4e0190f2d22fac66243e385d415aa78665cf03.jpg)

![cf9915c2ce59e4e92cad209432c66dc6cc62e4fde726560b09eafc0501cc313e.jpg](../iclr_results/2083_Adaptive Pruning of Pretrained Transformer via Differential Inclusions/tables/cf9915c2ce59e4e92cad209432c66dc6cc62e4fde726560b09eafc0501cc313e.jpg)
