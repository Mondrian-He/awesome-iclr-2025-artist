# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 46 of 100

## 目录 (Table of Contents)

1. [Diffusion Bridge Implicit Models](#Diffusion-Bridge-Implicit-Models)
2. [Swiss Army Knife: Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learning](#Swiss-Army-Knife-Synergizing-Biases-in-Knowledge-from-Vision-Foundation-Models-for-Multi-Task-Learning)
3. [A Meta-Learning Approach to Bayesian Causal Discovery](#A-Meta-Learning-Approach-to-Bayesian-Causal-Discovery)
4. [TweedieMix: Improving Multi-Concept Fusion for Diffusion-based Image/Video Generation](#TweedieMix-Improving-Multi-Concept-Fusion-for-Diffusion-based-ImageVideo-Generation)
5. [Learning Neural Networks with Distribution Shift: Efficiently Certifiable Guarantees](#Learning-Neural-Networks-with-Distribution-Shift-Efficiently-Certifiable-Guarantees)
6. [Filtered not Mixed: Filtering-Based Online Gating for Mixture of Large Language Models](#Filtered-not-Mixed-Filtering-Based-Online-Gating-for-Mixture-of-Large-Language-Models)
7. [Looking Inward: Language Models Can Learn About Themselves by Introspection](#Looking-Inward-Language-Models-Can-Learn-About-Themselves-by-Introspection)
8. [DiffSplat: Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation](#DiffSplat-Repurposing-Image-Diffusion-Models-for-Scalable-Gaussian-Splat-Generation)
9. [Mitigating Parameter Interference in Model Merging via Sharpness-Aware Fine-Tuning](#Mitigating-Parameter-Interference-in-Model-Merging-via-Sharpness-Aware-Fine-Tuning)
10. [Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood](#Offline-RL-with-Smooth-OOD-Generalization-in-Convex-Hull-and-its-Neighborhood)
11. [How Does Vision-Language Adaptation Impact the Safety of Vision Language Models?](#How-Does-Vision-Language-Adaptation-Impact-the-Safety-of-Vision-Language-Models)
12. [Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection](#Multi-Resolution-Decomposable-Diffusion-Model-for-Non-Stationary-Time-Series-Anomaly-Detection)
13. [Layerwise Recurrent Router for  Mixture-of-Experts](#Layerwise-Recurrent-Router-for-Mixture-of-Experts)
14. [On Minimizing Adversarial Counterfactual Error in Adversarial Reinforcement Learning](#On-Minimizing-Adversarial-Counterfactual-Error-in-Adversarial-Reinforcement-Learning)
15. [GraphRouter: A Graph-based Router for LLM Selections](#GraphRouter-A-Graph-based-Router-for-LLM-Selections)
16. [Rethinking Audio-Visual Adversarial Vulnerability from Temporal and Modality Perspectives](#Rethinking-Audio-Visual-Adversarial-Vulnerability-from-Temporal-and-Modality-Perspectives)
17. [Continuous Ensemble Weather Forecasting with Diffusion models](#Continuous-Ensemble-Weather-Forecasting-with-Diffusion-models)
18. [Generating Physical Dynamics under Priors](#Generating-Physical-Dynamics-under-Priors)
19. [DataMan: Data Manager for Pre-training Large Language Models](#DataMan-Data-Manager-for-Pre-training-Large-Language-Models)
20. [Preserving Deep Representations in One-Shot Pruning: A Hessian-Free Second-Order Optimization Framework](#Preserving-Deep-Representations-in-One-Shot-Pruning-A-Hessian-Free-Second-Order-Optimization-Framework)
21. [UniDetox: Universal Detoxification of Large Language Models via Dataset Distillation](#UniDetox-Universal-Detoxification-of-Large-Language-Models-via-Dataset-Distillation)
22. [Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count](#Arithmetic-Transformers-Can-Length-Generalize-in-Both-Operand-Length-and-Count)
23. [Look Before You Leap: Universal Emergent Mechanism for Retrieval in Language Models](#Look-Before-You-Leap-Universal-Emergent-Mechanism-for-Retrieval-in-Language-Models)
24. [Matrix Product Sketching via Coordinated Sampling](#Matrix-Product-Sketching-via-Coordinated-Sampling)
25. [3D-MolT5: Leveraging Discrete Structural Information for Molecule-Text Modeling](#3D-MolT5-Leveraging-Discrete-Structural-Information-for-Molecule-Text-Modeling)
26. [Unveiling the Secret Recipe: A Guide For Supervised Fine-Tuning Small LLMs](#Unveiling-the-Secret-Recipe-A-Guide-For-Supervised-Fine-Tuning-Small-LLMs)
27. [Learning Clustering-based Prototypes for Compositional Zero-Shot Learning](#Learning-Clustering-based-Prototypes-for-Compositional-Zero-Shot-Learning)
28. [BadJudge: Backdoor Vulnerabilities of LLM-As-A-Judge](#BadJudge-Backdoor-Vulnerabilities-of-LLM-As-A-Judge)
29. [Ctrl-U: Robust Conditional Image Generation via Uncertainty-aware Reward Modeling](#Ctrl-U-Robust-Conditional-Image-Generation-via-Uncertainty-aware-Reward-Modeling)
30. [Pairwise Elimination with Instance-Dependent Guarantees for Bandits with Cost Subsidy](#Pairwise-Elimination-with-Instance-Dependent-Guarantees-for-Bandits-with-Cost-Subsidy)
31. [Improved Techniques for Optimization-Based Jailbreaking on Large Language Models](#Improved-Techniques-for-Optimization-Based-Jailbreaking-on-Large-Language-Models)
32. [Beyond Worst-Case Dimensionality Reduction for Sparse Vectors](#Beyond-Worst-Case-Dimensionality-Reduction-for-Sparse-Vectors)
33. [PAD: Personalized Alignment of LLMs at Decoding-time](#PAD-Personalized-Alignment-of-LLMs-at-Decoding-time)
34. [Dreamweaver: Learning Compositional World Models from Pixels](#Dreamweaver-Learning-Compositional-World-Models-from-Pixels)
35. [Policy Decorator: Model-Agnostic Online Refinement for Large Policy Model](#Policy-Decorator-Model-Agnostic-Online-Refinement-for-Large-Policy-Model)
36. [Ensembling Diffusion Models via Adaptive Feature Aggregation](#Ensembling-Diffusion-Models-via-Adaptive-Feature-Aggregation)
37. [Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models](#Eliminating-Oversaturation-and-Artifacts-of-High-Guidance-Scales-in-Diffusion-Models)

---


## Diffusion Bridge Implicit Models

### Images

![2304f0192e0b9d46b0a16bf4d2a7ba537b639db1d6bf5b3e3f32663f06647a5b.jpg](../iclr_results/1676_Disentangled Representation Learning with the Gromov-Monge Gap/images/2304f0192e0b9d46b0a16bf4d2a7ba537b639db1d6bf5b3e3f32663f06647a5b.jpg)

![3d3879aa3abe7400e4f95dcf61e9695539ba30e3cc8a278505b61008656abfca.jpg](../iclr_results/1676_Disentangled Representation Learning with the Gromov-Monge Gap/images/3d3879aa3abe7400e4f95dcf61e9695539ba30e3cc8a278505b61008656abfca.jpg)

![5ba02de3b9c33d2339ca9eac4738001d8e02724eff082248a047f7e54d597cc3.jpg](../iclr_results/1676_Disentangled Representation Learning with the Gromov-Monge Gap/images/5ba02de3b9c33d2339ca9eac4738001d8e02724eff082248a047f7e54d597cc3.jpg)

![9f0e9f747acd15bd465b9dabc24a42775541aabd092940a8041f35b4547128db.jpg](../iclr_results/1676_Disentangled Representation Learning with the Gromov-Monge Gap/images/9f0e9f747acd15bd465b9dabc24a42775541aabd092940a8041f35b4547128db.jpg)

![b99069c0a6f35e3071f7500ad5a2ed2e7905ac84acb94cc711044250689cd53b.jpg](../iclr_results/1676_Disentangled Representation Learning with the Gromov-Monge Gap/images/b99069c0a6f35e3071f7500ad5a2ed2e7905ac84acb94cc711044250689cd53b.jpg)

![ba830eaed06e181a292ce54ccf3fcc114d03995b1b18d09fd80360a018d59ab5.jpg](../iclr_results/1676_Disentangled Representation Learning with the Gromov-Monge Gap/images/ba830eaed06e181a292ce54ccf3fcc114d03995b1b18d09fd80360a018d59ab5.jpg)

![d82aa432f075b75a2bd31e50612b54c6e7be808d66857fe3260608c1d325d167.jpg](../iclr_results/1676_Disentangled Representation Learning with the Gromov-Monge Gap/images/d82aa432f075b75a2bd31e50612b54c6e7be808d66857fe3260608c1d325d167.jpg)

![e3e7b8702853a91aabd93d37d42bc10e50f1f75018b6baf516dca29eb85776a4.jpg](../iclr_results/1676_Disentangled Representation Learning with the Gromov-Monge Gap/images/e3e7b8702853a91aabd93d37d42bc10e50f1f75018b6baf516dca29eb85776a4.jpg)

### Tables

![0aa95c3badac7a106b3e1aa62571caf1131cc5f5307914f0e7ce3449d3d5b126.jpg](../iclr_results/1676_Disentangled Representation Learning with the Gromov-Monge Gap/tables/0aa95c3badac7a106b3e1aa62571caf1131cc5f5307914f0e7ce3449d3d5b126.jpg)

![2f6e27226ac6a66736bcaa17ecb4253f4830f98582668fc67b918652cac0ea9f.jpg](../iclr_results/1676_Disentangled Representation Learning with the Gromov-Monge Gap/tables/2f6e27226ac6a66736bcaa17ecb4253f4830f98582668fc67b918652cac0ea9f.jpg)

![32b380d8eaf70ef9b227a6dd9840acb085410e6ebb10793453da675ba1ca7664.jpg](../iclr_results/1676_Disentangled Representation Learning with the Gromov-Monge Gap/tables/32b380d8eaf70ef9b227a6dd9840acb085410e6ebb10793453da675ba1ca7664.jpg)

![79204ff7af81a25e0593071fc6e1dfdbfc5aa26453f272bb7399e9324533becd.jpg](../iclr_results/1676_Disentangled Representation Learning with the Gromov-Monge Gap/tables/79204ff7af81a25e0593071fc6e1dfdbfc5aa26453f272bb7399e9324533becd.jpg)

![d1b322fe03dfc9c571ba5d394faab6ec6fc5bbe9d23d10fc92579ca21ca45581.jpg](../iclr_results/1676_Disentangled Representation Learning with the Gromov-Monge Gap/tables/d1b322fe03dfc9c571ba5d394faab6ec6fc5bbe9d23d10fc92579ca21ca45581.jpg)

![d274f2182f97d87dda4eaabbd7014681683dd48023f77bb071596f6981681d41.jpg](../iclr_results/1676_Disentangled Representation Learning with the Gromov-Monge Gap/tables/d274f2182f97d87dda4eaabbd7014681683dd48023f77bb071596f6981681d41.jpg)

![e5e6cce94f1c696dac1a97b5382ae5d055ad4182dd931dcde5c646e701f7e2ab.jpg](../iclr_results/1676_Disentangled Representation Learning with the Gromov-Monge Gap/tables/e5e6cce94f1c696dac1a97b5382ae5d055ad4182dd931dcde5c646e701f7e2ab.jpg)

## Diffusion Bridge Implicit Models


### Images

![2394216739f08636131bf4c8ef29a6adcd6eef8b0bd91f00a4bb33258a248575.jpg](../iclr_results/1677_Diffusion Bridge Implicit Models/images/2394216739f08636131bf4c8ef29a6adcd6eef8b0bd91f00a4bb33258a248575.jpg)

![32125651a7fbcc2e588c0939ccc7c0b77ce038b5f10ed718733240093ca90859.jpg](../iclr_results/1677_Diffusion Bridge Implicit Models/images/32125651a7fbcc2e588c0939ccc7c0b77ce038b5f10ed718733240093ca90859.jpg)

![60b8aa1da2951796de9b53b3369afb2fa09a42c65b35cfa410267d7b894248b9.jpg](../iclr_results/1677_Diffusion Bridge Implicit Models/images/60b8aa1da2951796de9b53b3369afb2fa09a42c65b35cfa410267d7b894248b9.jpg)

![a73ae1883abbbb36400033f0ca9dadb4d9dcde8c73df059ce528b2beeb21567d.jpg](../iclr_results/1677_Diffusion Bridge Implicit Models/images/a73ae1883abbbb36400033f0ca9dadb4d9dcde8c73df059ce528b2beeb21567d.jpg)

![b59aa8d75fe31d667884e9f29bd7cb4f27530b0f511e524fc655780f87989490.jpg](../iclr_results/1677_Diffusion Bridge Implicit Models/images/b59aa8d75fe31d667884e9f29bd7cb4f27530b0f511e524fc655780f87989490.jpg)

![b67147d0fe7914cf1876b27937c9cf87a46edd8fbb716cd487d4b78156d313a8.jpg](../iclr_results/1677_Diffusion Bridge Implicit Models/images/b67147d0fe7914cf1876b27937c9cf87a46edd8fbb716cd487d4b78156d313a8.jpg)

![ef3d96bcfbe0198ca02d2bfe04d99aa6daa6b13fd183cc26fc0206b8237f4357.jpg](../iclr_results/1677_Diffusion Bridge Implicit Models/images/ef3d96bcfbe0198ca02d2bfe04d99aa6daa6b13fd183cc26fc0206b8237f4357.jpg)

![fb320cc4687aef6a3db1c4b8ead9202f63f7325e29d84a87e27cc856e07220c2.jpg](../iclr_results/1677_Diffusion Bridge Implicit Models/images/fb320cc4687aef6a3db1c4b8ead9202f63f7325e29d84a87e27cc856e07220c2.jpg)

### Tables

![1094d86ae17a1617d66b771bc38ab603c6512ff7a6c208d5049f6417c03578b6.jpg](../iclr_results/1677_Diffusion Bridge Implicit Models/tables/1094d86ae17a1617d66b771bc38ab603c6512ff7a6c208d5049f6417c03578b6.jpg)

![464e30ef2764691d65697b9736a595221d5ff9203ade28c1f66b8fa962559d4f.jpg](../iclr_results/1677_Diffusion Bridge Implicit Models/tables/464e30ef2764691d65697b9736a595221d5ff9203ade28c1f66b8fa962559d4f.jpg)

![7173df857000160b9ea78043f277e403ad11809ae36919ffd2e4d0001970c22e.jpg](../iclr_results/1677_Diffusion Bridge Implicit Models/tables/7173df857000160b9ea78043f277e403ad11809ae36919ffd2e4d0001970c22e.jpg)

![89e03669d54c4aaf94d9dfb19d3ecc965cd29d63f1845ed6d2f927c16dd701b2.jpg](../iclr_results/1677_Diffusion Bridge Implicit Models/tables/89e03669d54c4aaf94d9dfb19d3ecc965cd29d63f1845ed6d2f927c16dd701b2.jpg)

![958776444671c87c48294d8f3072d2368f04e857a4db2806d6c34f75e6c5dfdc.jpg](../iclr_results/1677_Diffusion Bridge Implicit Models/tables/958776444671c87c48294d8f3072d2368f04e857a4db2806d6c34f75e6c5dfdc.jpg)

![a632806537f3954579a4c10a06f77c76c136b783d93af83530e918268c07cdf5.jpg](../iclr_results/1677_Diffusion Bridge Implicit Models/tables/a632806537f3954579a4c10a06f77c76c136b783d93af83530e918268c07cdf5.jpg)

![dd3057b0cfcc8a41d4b00fd2b5cf03bd0367ba8934b471cd17b64da4b84829c7.jpg](../iclr_results/1677_Diffusion Bridge Implicit Models/tables/dd3057b0cfcc8a41d4b00fd2b5cf03bd0367ba8934b471cd17b64da4b84829c7.jpg)

![edeae7e045ee06538ad1393c6605526c696da8c170983743d49bbb10706f05a6.jpg](../iclr_results/1677_Diffusion Bridge Implicit Models/tables/edeae7e045ee06538ad1393c6605526c696da8c170983743d49bbb10706f05a6.jpg)

![f4aae120157f1087659287be944abe6bd753d469c3ea630af3825994592f5b12.jpg](../iclr_results/1677_Diffusion Bridge Implicit Models/tables/f4aae120157f1087659287be944abe6bd753d469c3ea630af3825994592f5b12.jpg)

## Swiss Army Knife: Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learning


### Images

![008a0f3321f53930377d8c617c652f356a0acfbb0172df0a9a65ecc0fe04ad7b.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/images/008a0f3321f53930377d8c617c652f356a0acfbb0172df0a9a65ecc0fe04ad7b.jpg)

![30d9ceb716df436c6fcebb1007c5379e5dff61b0bd5676fef3b0f1542644f42f.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/images/30d9ceb716df436c6fcebb1007c5379e5dff61b0bd5676fef3b0f1542644f42f.jpg)

![34a02128531bc86cb3452afa30e9ce3f9980b3a35e085a071b66bbd2fffb4211.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/images/34a02128531bc86cb3452afa30e9ce3f9980b3a35e085a071b66bbd2fffb4211.jpg)

![4daccdc43c7fa8fa28897ea394e4f3e60c0a6479f76b43d72a01ea3922dede15.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/images/4daccdc43c7fa8fa28897ea394e4f3e60c0a6479f76b43d72a01ea3922dede15.jpg)

![4ec4b39f770058e472c85b487b4b8ad5c820a53d21c76865ee0987381581026f.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/images/4ec4b39f770058e472c85b487b4b8ad5c820a53d21c76865ee0987381581026f.jpg)

![5836841c84d59c23d76900a7b9b2f45d9d3fc31b814fdddd3db2fe047ce9fb94.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/images/5836841c84d59c23d76900a7b9b2f45d9d3fc31b814fdddd3db2fe047ce9fb94.jpg)

![6e4304a31ab63b04a05a6fad3714f5d488fdf8952e9b4bb58b7ff32139968bee.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/images/6e4304a31ab63b04a05a6fad3714f5d488fdf8952e9b4bb58b7ff32139968bee.jpg)

![749c29dfe82f82536c2b4dfc4788e0faec2cccca6bd709437dfb88026713affa.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/images/749c29dfe82f82536c2b4dfc4788e0faec2cccca6bd709437dfb88026713affa.jpg)

![79614d95ad20e0433f00d45214e65d294ee71df920c68745eed33e285be31d50.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/images/79614d95ad20e0433f00d45214e65d294ee71df920c68745eed33e285be31d50.jpg)

![f58eafa66c73839f60efecf63c11874e1817d8094d07aec9da111e05b19a5fa6.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/images/f58eafa66c73839f60efecf63c11874e1817d8094d07aec9da111e05b19a5fa6.jpg)

![fabc350aefdc444840119422aa275f127c7b755b02e3168e8173c12e32f3a245.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/images/fabc350aefdc444840119422aa275f127c7b755b02e3168e8173c12e32f3a245.jpg)

### Tables

![054cbd547d22110896d50f01149ee8006954a2151c36966f256550d331e25da7.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/054cbd547d22110896d50f01149ee8006954a2151c36966f256550d331e25da7.jpg)

![07e1fefd47ddd56dae2430b000513149b68cb225609f4dcfe71d375ab31673a0.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/07e1fefd47ddd56dae2430b000513149b68cb225609f4dcfe71d375ab31673a0.jpg)

![0a4e96fb03e15c2980a2f2731188b73c29bd7364af1b6bff200654ac86e42b54.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/0a4e96fb03e15c2980a2f2731188b73c29bd7364af1b6bff200654ac86e42b54.jpg)

![1f23890326b231121e8d4957a0506cd989e08289ef691b5acfe7752a847f5ea4.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/1f23890326b231121e8d4957a0506cd989e08289ef691b5acfe7752a847f5ea4.jpg)

![25d012c276674fd39c6a9e3e0f0872ec5d1c6926c38a2653b6cf4c9a4d0b1cbe.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/25d012c276674fd39c6a9e3e0f0872ec5d1c6926c38a2653b6cf4c9a4d0b1cbe.jpg)

![2d6f9d2618b44e213ee5437df518b41d280977844f68b59da1b1baa802952ea5.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/2d6f9d2618b44e213ee5437df518b41d280977844f68b59da1b1baa802952ea5.jpg)

![2e8b33f27237cfa6aba69432062f85fed99b260ecab5eebb031cd22cbe9a1028.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/2e8b33f27237cfa6aba69432062f85fed99b260ecab5eebb031cd22cbe9a1028.jpg)

![44b88953cd52aa20926d41f5355ad2699530d0b62e0c25c40e1e2650ce737408.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/44b88953cd52aa20926d41f5355ad2699530d0b62e0c25c40e1e2650ce737408.jpg)

![458bf2dc8abf5f2541065af2c37c72de7e2738550f1e0b7d2a98c0e336855333.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/458bf2dc8abf5f2541065af2c37c72de7e2738550f1e0b7d2a98c0e336855333.jpg)

![46b83cce59cb2e92fbb6b2a31355517bf3e2d33bcd1e9b4cca107c8341158549.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/46b83cce59cb2e92fbb6b2a31355517bf3e2d33bcd1e9b4cca107c8341158549.jpg)

![49e43b653294ac7fb1f8367087fe88f617e403f46ce14aafc51c763668996322.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/49e43b653294ac7fb1f8367087fe88f617e403f46ce14aafc51c763668996322.jpg)

![4b225de5efd9071fb077143afecb4bfd97728080fde31af5a43cb4ceb26cfec5.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/4b225de5efd9071fb077143afecb4bfd97728080fde31af5a43cb4ceb26cfec5.jpg)

![5ba39f4165e236dfd16254c1d869b0ae6da5f1c5cf2284c52e8c9d94977d8617.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/5ba39f4165e236dfd16254c1d869b0ae6da5f1c5cf2284c52e8c9d94977d8617.jpg)

![5d7d3d9ef84100b27a2315b40588c9f8972eaa57cf6d3ba7081ab2b0946fbb75.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/5d7d3d9ef84100b27a2315b40588c9f8972eaa57cf6d3ba7081ab2b0946fbb75.jpg)

![6c11405292375d847d6fe13f1f66b60a60fb08557fa40444b4ef2ea727e994dc.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/6c11405292375d847d6fe13f1f66b60a60fb08557fa40444b4ef2ea727e994dc.jpg)

![7432a68201de10174d9669b5785d8f79ec8f288985ea5c5b0c87420c978ab007.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/7432a68201de10174d9669b5785d8f79ec8f288985ea5c5b0c87420c978ab007.jpg)

![74c773bfda446188ed7121df0cd70f687d2aa7dc3de3dd71ce58f4f0d5e5a7b5.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/74c773bfda446188ed7121df0cd70f687d2aa7dc3de3dd71ce58f4f0d5e5a7b5.jpg)

![7fae2bd71e9fe914b20c0d244ae5fc9d6b03d0506219519a3c7fd06b259ca54d.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/7fae2bd71e9fe914b20c0d244ae5fc9d6b03d0506219519a3c7fd06b259ca54d.jpg)

![8b32ff93376a7cb0cba255fe774c99920bd7062946f35c41dce9aa1d192eb5ff.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/8b32ff93376a7cb0cba255fe774c99920bd7062946f35c41dce9aa1d192eb5ff.jpg)

![936834c54daec2016b09fcee321cdf001c1ec39da03d35fe7b990af4016cce8c.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/936834c54daec2016b09fcee321cdf001c1ec39da03d35fe7b990af4016cce8c.jpg)

![954b6752e4654168037cdc9076d5df0b3a39e2050b7696a61c3680a786f211ee.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/954b6752e4654168037cdc9076d5df0b3a39e2050b7696a61c3680a786f211ee.jpg)

![b0e96f9afbc1f01b3876445a3e2caa3c14bbbbb99b98a3df9865715172064f37.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/b0e96f9afbc1f01b3876445a3e2caa3c14bbbbb99b98a3df9865715172064f37.jpg)

![c65ed9a9ba6e7cd2020359c5779284eec08077591d550f18cad57a5f3e98b8b9.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/c65ed9a9ba6e7cd2020359c5779284eec08077591d550f18cad57a5f3e98b8b9.jpg)

![c9cd782abf22554ff01a706d51b0779f91a23471388d2fe2bf3b08a53cb4b894.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/c9cd782abf22554ff01a706d51b0779f91a23471388d2fe2bf3b08a53cb4b894.jpg)

![ce23ba3a1dbe69d1e113900a86c1b1cdb6928b4752b715b9513b5cbe03b9282d.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/ce23ba3a1dbe69d1e113900a86c1b1cdb6928b4752b715b9513b5cbe03b9282d.jpg)

![dc50b9df758d175f6c47fa649104976ac4798fd42767514564274ee970f812f8.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/dc50b9df758d175f6c47fa649104976ac4798fd42767514564274ee970f812f8.jpg)

![e5f1c2d2145ae7f5fbb25e88a0131746d510026eb296a61af9c9405dc4490d8d.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/e5f1c2d2145ae7f5fbb25e88a0131746d510026eb296a61af9c9405dc4490d8d.jpg)

![eeabfa90fe2cccecb4e92e97d680e42d8909aaf91e9292a675e3020a9c594ff7.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/eeabfa90fe2cccecb4e92e97d680e42d8909aaf91e9292a675e3020a9c594ff7.jpg)

![f4438c0a738f3485e3ef938f07279d0fadbfedfd78708ba5b2df39037f30b276.jpg](../iclr_results/1678_Swiss Army Knife_ Synergizing Biases in Knowledge from Vision Foundation Models for Multi-Task Learn/tables/f4438c0a738f3485e3ef938f07279d0fadbfedfd78708ba5b2df39037f30b276.jpg)

## A Meta-Learning Approach to Bayesian Causal Discovery


### Images

![04df11a8bf8d0054652e2097a72d72c59657544ee30f79acff7810609cc5d5b1.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/images/04df11a8bf8d0054652e2097a72d72c59657544ee30f79acff7810609cc5d5b1.jpg)

![b07db81429c401aae8e15942fd07708722602e09fa108fbb691b9dee11ef2127.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/images/b07db81429c401aae8e15942fd07708722602e09fa108fbb691b9dee11ef2127.jpg)

### Tables

![135e09fa01ffdd8f9d40e7fc06b13852d6637b2a33fd920a533ce5b35e738298.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/135e09fa01ffdd8f9d40e7fc06b13852d6637b2a33fd920a533ce5b35e738298.jpg)

![215fd93ed26038bb66f408e5c55d3a9742b8cbae8e22f9208ef05ce39a2915a4.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/215fd93ed26038bb66f408e5c55d3a9742b8cbae8e22f9208ef05ce39a2915a4.jpg)

![219f5cded8a4aa6a8807ebd7d899880ea2ed158d45fb7efac7f0370f6375cf01.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/219f5cded8a4aa6a8807ebd7d899880ea2ed158d45fb7efac7f0370f6375cf01.jpg)

![23a5b738baa87b126151199517eb4078074a51db74ae71a84456ff6ce8ae0fed.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/23a5b738baa87b126151199517eb4078074a51db74ae71a84456ff6ce8ae0fed.jpg)

![314839924ce5dd67a47915e1b9c0f30083871eff81d955e6d045e130d51d611e.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/314839924ce5dd67a47915e1b9c0f30083871eff81d955e6d045e130d51d611e.jpg)

![3dda11313ff38fa6813079e89ee7eb0da78024c18fda1c28edce485d82c6eb39.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/3dda11313ff38fa6813079e89ee7eb0da78024c18fda1c28edce485d82c6eb39.jpg)

![4d1d4f8cc841e614bc34e3ea3e886b44dd07ecf6d89cd2ef2d548c1af0e6d0f8.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/4d1d4f8cc841e614bc34e3ea3e886b44dd07ecf6d89cd2ef2d548c1af0e6d0f8.jpg)

![5073828e002344b6814bcb0c4c2adfb226ad23ca720417fd981d432c6a41b807.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/5073828e002344b6814bcb0c4c2adfb226ad23ca720417fd981d432c6a41b807.jpg)

![7c30606a008b0ac404e556bad60d04407b71c088eab7b9bdff5f719c144ab84f.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/7c30606a008b0ac404e556bad60d04407b71c088eab7b9bdff5f719c144ab84f.jpg)

![83c0cdf5d4a08de481007212bad87da1de2b44a8ac686642fe5905929cbc6af8.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/83c0cdf5d4a08de481007212bad87da1de2b44a8ac686642fe5905929cbc6af8.jpg)

![896e10d5963ffaa8d41d707f200061121a5a6fca0093dce6c8f48984dbaa4da0.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/896e10d5963ffaa8d41d707f200061121a5a6fca0093dce6c8f48984dbaa4da0.jpg)

![8ab525aaab4abffe7ead08ce1615493d04f567e73d889295e3cd3c4578b980f3.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/8ab525aaab4abffe7ead08ce1615493d04f567e73d889295e3cd3c4578b980f3.jpg)

![926b9f65c793d1f04d64f885155c9b2eeac2a4f3aa0123cb87f8785b5ad5ce80.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/926b9f65c793d1f04d64f885155c9b2eeac2a4f3aa0123cb87f8785b5ad5ce80.jpg)

![9fb78126f1bf3fecf486c931c8a06dd0d37b4eb6f7d9619d6c3c90d324865e1c.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/9fb78126f1bf3fecf486c931c8a06dd0d37b4eb6f7d9619d6c3c90d324865e1c.jpg)

![a1e8763ac2e3e8c1c9ee67a0b73f1a9fead0bf3efd21dda89584ce33f1cfd691.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/a1e8763ac2e3e8c1c9ee67a0b73f1a9fead0bf3efd21dda89584ce33f1cfd691.jpg)

![b3cf87ae263eb9119d9c651ae131b46e2e620f1752b968f31e4f34eb917a9f28.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/b3cf87ae263eb9119d9c651ae131b46e2e620f1752b968f31e4f34eb917a9f28.jpg)

![bfb01f0ebf2813bac00498cf943fc029a7a7cb628814b378a76ee992031c0352.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/bfb01f0ebf2813bac00498cf943fc029a7a7cb628814b378a76ee992031c0352.jpg)

![c9488bea0cf969389cb4da97bcaf982cd5dafeafe88e8b1cdbb55e280ec9e50a.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/c9488bea0cf969389cb4da97bcaf982cd5dafeafe88e8b1cdbb55e280ec9e50a.jpg)

![cd6bb01d102f5f1635db4cc2577db80611dda32c82bba458121059806892cec1.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/cd6bb01d102f5f1635db4cc2577db80611dda32c82bba458121059806892cec1.jpg)

![d1bce392283b407b28958446e1432eb5ae874d765ed09764d5af92be5f5fd9b2.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/d1bce392283b407b28958446e1432eb5ae874d765ed09764d5af92be5f5fd9b2.jpg)

![d29bf68ad996208f291f2b7734eeb91a55ee064dcb6a85d1390dff55959e6573.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/d29bf68ad996208f291f2b7734eeb91a55ee064dcb6a85d1390dff55959e6573.jpg)

![e6f356a27eeba0ac57cb0a88146acd77ca99c9fd26146cad645e194726f069a6.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/e6f356a27eeba0ac57cb0a88146acd77ca99c9fd26146cad645e194726f069a6.jpg)

![f3151014905e82955acb77bb348fb158b0d816dd7fe30971df6999045ca7e94e.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/f3151014905e82955acb77bb348fb158b0d816dd7fe30971df6999045ca7e94e.jpg)

![f641278b63a71a529be46d38e489881b2fd20fb2adfc1966c7a5ec47a086cce8.jpg](../iclr_results/1679_A Meta-Learning Approach to Bayesian Causal Discovery/tables/f641278b63a71a529be46d38e489881b2fd20fb2adfc1966c7a5ec47a086cce8.jpg)

## TweedieMix: Improving Multi-Concept Fusion for Diffusion-based Image/Video Generation


### Images

![058aa70ce86556f83e3087c1c2a28b90438150d6caf715e9fa9d4daf99fa71b8.jpg](../iclr_results/1680_TweedieMix_ Improving Multi-Concept Fusion for Diffusion-based Image_Video Generation/images/058aa70ce86556f83e3087c1c2a28b90438150d6caf715e9fa9d4daf99fa71b8.jpg)

![19a1e7b70ef80f9384565942b3a4cb92e54ba2093e1d2b887c1e20e52f8b210e.jpg](../iclr_results/1680_TweedieMix_ Improving Multi-Concept Fusion for Diffusion-based Image_Video Generation/images/19a1e7b70ef80f9384565942b3a4cb92e54ba2093e1d2b887c1e20e52f8b210e.jpg)

![1cf5af69f313d3a890f11953a3c89e4401d5c416fbf5941aa4dcc98e79031a56.jpg](../iclr_results/1680_TweedieMix_ Improving Multi-Concept Fusion for Diffusion-based Image_Video Generation/images/1cf5af69f313d3a890f11953a3c89e4401d5c416fbf5941aa4dcc98e79031a56.jpg)

![336b7a5a90341a10225be3ba2751ca0b5ef96d7ed150a6b380f889697cf3145b.jpg](../iclr_results/1680_TweedieMix_ Improving Multi-Concept Fusion for Diffusion-based Image_Video Generation/images/336b7a5a90341a10225be3ba2751ca0b5ef96d7ed150a6b380f889697cf3145b.jpg)

![3d18bbd48db0073fb66467e78efef6ebd2898cebdc3726872899acea82ff3279.jpg](../iclr_results/1680_TweedieMix_ Improving Multi-Concept Fusion for Diffusion-based Image_Video Generation/images/3d18bbd48db0073fb66467e78efef6ebd2898cebdc3726872899acea82ff3279.jpg)

![3f6821c7e4608a54b31071afd1e109aec863378dd5756ac0643cc55254c5b554.jpg](../iclr_results/1680_TweedieMix_ Improving Multi-Concept Fusion for Diffusion-based Image_Video Generation/images/3f6821c7e4608a54b31071afd1e109aec863378dd5756ac0643cc55254c5b554.jpg)

![53d3c55d0bbdea7298a36b5d748bb633fac024938dfed7aa7a224fc251e3a33f.jpg](../iclr_results/1680_TweedieMix_ Improving Multi-Concept Fusion for Diffusion-based Image_Video Generation/images/53d3c55d0bbdea7298a36b5d748bb633fac024938dfed7aa7a224fc251e3a33f.jpg)

![63499f38d78640a20a2fcafa3e90f132c82cad3fd79b47d831d2f6d2081cee41.jpg](../iclr_results/1680_TweedieMix_ Improving Multi-Concept Fusion for Diffusion-based Image_Video Generation/images/63499f38d78640a20a2fcafa3e90f132c82cad3fd79b47d831d2f6d2081cee41.jpg)

![7014363112fa973c016d7d4b594019b59736a8c2d5ecd9d683bf0590f39ed443.jpg](../iclr_results/1680_TweedieMix_ Improving Multi-Concept Fusion for Diffusion-based Image_Video Generation/images/7014363112fa973c016d7d4b594019b59736a8c2d5ecd9d683bf0590f39ed443.jpg)

![71fb964f2914715b9007bc44deaedb1c9a28ef3cce7bf832b2b787cacbfb5cff.jpg](../iclr_results/1680_TweedieMix_ Improving Multi-Concept Fusion for Diffusion-based Image_Video Generation/images/71fb964f2914715b9007bc44deaedb1c9a28ef3cce7bf832b2b787cacbfb5cff.jpg)

![7ef9f2cc457f6c86d71211011abcdae62d10807aa6a6b830e4dd42732e4412b6.jpg](../iclr_results/1680_TweedieMix_ Improving Multi-Concept Fusion for Diffusion-based Image_Video Generation/images/7ef9f2cc457f6c86d71211011abcdae62d10807aa6a6b830e4dd42732e4412b6.jpg)

![bc6281e98f8ffd6c4ac434ae4348f965d703cd2a730aa9fee2095cb6e8e8d834.jpg](../iclr_results/1680_TweedieMix_ Improving Multi-Concept Fusion for Diffusion-based Image_Video Generation/images/bc6281e98f8ffd6c4ac434ae4348f965d703cd2a730aa9fee2095cb6e8e8d834.jpg)

![cba570d289f02366e37710741b885b8e56a3fb590af12a5f0b6970734e27364a.jpg](../iclr_results/1680_TweedieMix_ Improving Multi-Concept Fusion for Diffusion-based Image_Video Generation/images/cba570d289f02366e37710741b885b8e56a3fb590af12a5f0b6970734e27364a.jpg)

![ce72be9c7c685e073db79c57aa8ee6db42e6a7164f023dab52de41a64b52a8db.jpg](../iclr_results/1680_TweedieMix_ Improving Multi-Concept Fusion for Diffusion-based Image_Video Generation/images/ce72be9c7c685e073db79c57aa8ee6db42e6a7164f023dab52de41a64b52a8db.jpg)

![ea35f088c5e9d4f40cd36cb9219c16778db4cc86a51cd2d22aff9ee21877e4bd.jpg](../iclr_results/1680_TweedieMix_ Improving Multi-Concept Fusion for Diffusion-based Image_Video Generation/images/ea35f088c5e9d4f40cd36cb9219c16778db4cc86a51cd2d22aff9ee21877e4bd.jpg)

![fcb1dafa136d29dad2228a196e7b5666e11b23d63c3248610cb6fec523e73d7b.jpg](../iclr_results/1680_TweedieMix_ Improving Multi-Concept Fusion for Diffusion-based Image_Video Generation/images/fcb1dafa136d29dad2228a196e7b5666e11b23d63c3248610cb6fec523e73d7b.jpg)

![ff175c6622b479fdee5eb017805e2f6d2f463e85c7c0aec2a8d4a9cf50d5bc19.jpg](../iclr_results/1680_TweedieMix_ Improving Multi-Concept Fusion for Diffusion-based Image_Video Generation/images/ff175c6622b479fdee5eb017805e2f6d2f463e85c7c0aec2a8d4a9cf50d5bc19.jpg)

![ff410b351952a1eca9568ea8fe2ed4ed21b426bb2fb23aaaef51a7304c6c43f3.jpg](../iclr_results/1680_TweedieMix_ Improving Multi-Concept Fusion for Diffusion-based Image_Video Generation/images/ff410b351952a1eca9568ea8fe2ed4ed21b426bb2fb23aaaef51a7304c6c43f3.jpg)

### Tables

![7781b0b8168abe87f85254702b7903c5527de50e11aa37d6578536e31eb8aa32.jpg](../iclr_results/1680_TweedieMix_ Improving Multi-Concept Fusion for Diffusion-based Image_Video Generation/tables/7781b0b8168abe87f85254702b7903c5527de50e11aa37d6578536e31eb8aa32.jpg)

![7ea35f0fe184c7647f3ca3421e4e38d08b870d1cf4eae86f096d2105fe13623a.jpg](../iclr_results/1680_TweedieMix_ Improving Multi-Concept Fusion for Diffusion-based Image_Video Generation/tables/7ea35f0fe184c7647f3ca3421e4e38d08b870d1cf4eae86f096d2105fe13623a.jpg)

## Learning Neural Networks with Distribution Shift: Efficiently Certifiable Guarantees


### Tables

![46863c85d1b74288d37ecd0908c454a09942344f42f36c2f5f198e011f5730ba.jpg](../iclr_results/1681_Learning Neural Networks with Distribution Shift_ Efficiently Certifiable Guarantees/tables/46863c85d1b74288d37ecd0908c454a09942344f42f36c2f5f198e011f5730ba.jpg)

![85f1a97d67c8eb2118b5d6a6376ffd7523c87ad0e33b565edc4505309213dbe9.jpg](../iclr_results/1681_Learning Neural Networks with Distribution Shift_ Efficiently Certifiable Guarantees/tables/85f1a97d67c8eb2118b5d6a6376ffd7523c87ad0e33b565edc4505309213dbe9.jpg)

## Filtered not Mixed: Filtering-Based Online Gating for Mixture of Large Language Models


### Images

![046fc0f1eea97b4976454b207c6ce3cd136fe983ac91690a9002eba0b856fc7e.jpg](../iclr_results/1682_Filtered not Mixed_ Filtering-Based Online Gating for Mixture of Large Language Models/images/046fc0f1eea97b4976454b207c6ce3cd136fe983ac91690a9002eba0b856fc7e.jpg)

![3e8f630f0cda031991de33712b59a2ea47b44bc3068355b363f1f6cef3fb788c.jpg](../iclr_results/1682_Filtered not Mixed_ Filtering-Based Online Gating for Mixture of Large Language Models/images/3e8f630f0cda031991de33712b59a2ea47b44bc3068355b363f1f6cef3fb788c.jpg)

![605f26d18a951d9ab990a658b5be1d76d873fa0995670f10a36e8bbd19774b28.jpg](../iclr_results/1682_Filtered not Mixed_ Filtering-Based Online Gating for Mixture of Large Language Models/images/605f26d18a951d9ab990a658b5be1d76d873fa0995670f10a36e8bbd19774b28.jpg)

![8fed955386df3fe82ab8d202fcec51798446a567e893e439f89a23c306584ae2.jpg](../iclr_results/1682_Filtered not Mixed_ Filtering-Based Online Gating for Mixture of Large Language Models/images/8fed955386df3fe82ab8d202fcec51798446a567e893e439f89a23c306584ae2.jpg)

![a35480cf4d10b5e9dafa33c9bb31bd5fe251945e8ebac74e45dfd1189f437f2c.jpg](../iclr_results/1682_Filtered not Mixed_ Filtering-Based Online Gating for Mixture of Large Language Models/images/a35480cf4d10b5e9dafa33c9bb31bd5fe251945e8ebac74e45dfd1189f437f2c.jpg)

![ad1a2a38ab7e3431241bb13522b5ec14cb3868b227fa1289143f2a288bb0efdf.jpg](../iclr_results/1682_Filtered not Mixed_ Filtering-Based Online Gating for Mixture of Large Language Models/images/ad1a2a38ab7e3431241bb13522b5ec14cb3868b227fa1289143f2a288bb0efdf.jpg)

### Tables

![208e23bb9b61f8b6e21a7250f565388a5fe79b403aa51b1d72c71c0bbfd48aa9.jpg](../iclr_results/1682_Filtered not Mixed_ Filtering-Based Online Gating for Mixture of Large Language Models/tables/208e23bb9b61f8b6e21a7250f565388a5fe79b403aa51b1d72c71c0bbfd48aa9.jpg)

![4b04a976310287b2a690205ff4aae0b6362383df2b9f340d06a4092c45c5be38.jpg](../iclr_results/1682_Filtered not Mixed_ Filtering-Based Online Gating for Mixture of Large Language Models/tables/4b04a976310287b2a690205ff4aae0b6362383df2b9f340d06a4092c45c5be38.jpg)

![4b65f7a7592ed0601df79c43e4bf2652796f7d727d8238f0e6dc5932ea59a549.jpg](../iclr_results/1682_Filtered not Mixed_ Filtering-Based Online Gating for Mixture of Large Language Models/tables/4b65f7a7592ed0601df79c43e4bf2652796f7d727d8238f0e6dc5932ea59a549.jpg)

![6e9ef4e7356927be803e32b971265283ae17f76f20e7ec5b042a20c8f0b9113e.jpg](../iclr_results/1682_Filtered not Mixed_ Filtering-Based Online Gating for Mixture of Large Language Models/tables/6e9ef4e7356927be803e32b971265283ae17f76f20e7ec5b042a20c8f0b9113e.jpg)

![76f73878cc7cefc185e2cdacc2de19c7aa1b0cfd9f1d1c68b3eda5d41562e7f7.jpg](../iclr_results/1682_Filtered not Mixed_ Filtering-Based Online Gating for Mixture of Large Language Models/tables/76f73878cc7cefc185e2cdacc2de19c7aa1b0cfd9f1d1c68b3eda5d41562e7f7.jpg)

![8774e7a339f7b8ea1cd9fda058020f8f905846d2625e41207e2ba854de9f82a6.jpg](../iclr_results/1682_Filtered not Mixed_ Filtering-Based Online Gating for Mixture of Large Language Models/tables/8774e7a339f7b8ea1cd9fda058020f8f905846d2625e41207e2ba854de9f82a6.jpg)

![8b6b0634af596c457b4ee08b1d0c6b2265ad7395dbc82270c722f5e00206f319.jpg](../iclr_results/1682_Filtered not Mixed_ Filtering-Based Online Gating for Mixture of Large Language Models/tables/8b6b0634af596c457b4ee08b1d0c6b2265ad7395dbc82270c722f5e00206f319.jpg)

![ae2436478664965b2b9f5a79607fc3868324526f876f5daba2f2db4fa9e703df.jpg](../iclr_results/1682_Filtered not Mixed_ Filtering-Based Online Gating for Mixture of Large Language Models/tables/ae2436478664965b2b9f5a79607fc3868324526f876f5daba2f2db4fa9e703df.jpg)

![db8db335204e48e0837d3eb15930c025cab6da818c7bd5cceb5dada52f72b1e5.jpg](../iclr_results/1682_Filtered not Mixed_ Filtering-Based Online Gating for Mixture of Large Language Models/tables/db8db335204e48e0837d3eb15930c025cab6da818c7bd5cceb5dada52f72b1e5.jpg)

![dc08e987917501fdba0a1625788868c1712ff6b8b0cca02f43db7ae700cab8a5.jpg](../iclr_results/1682_Filtered not Mixed_ Filtering-Based Online Gating for Mixture of Large Language Models/tables/dc08e987917501fdba0a1625788868c1712ff6b8b0cca02f43db7ae700cab8a5.jpg)

## Looking Inward: Language Models Can Learn About Themselves by Introspection


### Images

![0b75fe02079d20f3b590fbc6a5aa74a46e8616738cd03eedfe221a9437058d61.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/0b75fe02079d20f3b590fbc6a5aa74a46e8616738cd03eedfe221a9437058d61.jpg)

![1278faeea017e7240a965120727397c31a86f83b870c6465799baaef21d46133.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/1278faeea017e7240a965120727397c31a86f83b870c6465799baaef21d46133.jpg)

![1a7b284871d6c1c50c6a2a96f5c14bbccfea79e8cf02abdc4c42c2ab56472da6.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/1a7b284871d6c1c50c6a2a96f5c14bbccfea79e8cf02abdc4c42c2ab56472da6.jpg)

![21af4a7c72068f52f3072cc2fc2fba5873ad67f3bef0728201198be7d95e70b7.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/21af4a7c72068f52f3072cc2fc2fba5873ad67f3bef0728201198be7d95e70b7.jpg)

![2a6d703565693ceaed648421bdbc322365466deda9e28fc3dd0718016cb21773.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/2a6d703565693ceaed648421bdbc322365466deda9e28fc3dd0718016cb21773.jpg)

![2baeabd26f1c80040a1f9a48f4f8f537c4719b82928f54f864de2743b5cc8a3b.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/2baeabd26f1c80040a1f9a48f4f8f537c4719b82928f54f864de2743b5cc8a3b.jpg)

![3713319ac3fff9bdb2307d6e36ddfd90b3470ee404874958dd403435362e43ec.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/3713319ac3fff9bdb2307d6e36ddfd90b3470ee404874958dd403435362e43ec.jpg)

![3c0bd09066889f4a0cef8c28e2c4e9ca761e6563e43cc166f59e7ab10f96f08d.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/3c0bd09066889f4a0cef8c28e2c4e9ca761e6563e43cc166f59e7ab10f96f08d.jpg)

![3f7250c9e5fc9ac02f4aefd7e4aef0892fb60dd2ec0ddd10e91c271e9eece90e.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/3f7250c9e5fc9ac02f4aefd7e4aef0892fb60dd2ec0ddd10e91c271e9eece90e.jpg)

![436007cad829a85e149d84b593037ee93baa3270f8f4d4bcd0fe16fa81b6cc71.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/436007cad829a85e149d84b593037ee93baa3270f8f4d4bcd0fe16fa81b6cc71.jpg)

![4b1b441b267600b687b18548c92615e65e4dda667e36d5f309d2c8dd52c98279.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/4b1b441b267600b687b18548c92615e65e4dda667e36d5f309d2c8dd52c98279.jpg)

![53a79b798b1ddff001cd8322738e75bb033ada72e84f647f1be4943fb6118636.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/53a79b798b1ddff001cd8322738e75bb033ada72e84f647f1be4943fb6118636.jpg)

![599c5ee2a9a11a23c222950346ab358aefa12c20aa43e4e7757983f1508d966c.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/599c5ee2a9a11a23c222950346ab358aefa12c20aa43e4e7757983f1508d966c.jpg)

![59d8f3f4db7993e7faf001ba32810223ee80115e078a0851dfd23c1466bbdf45.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/59d8f3f4db7993e7faf001ba32810223ee80115e078a0851dfd23c1466bbdf45.jpg)

![5c94ddef1e5721fd063a3eb0d96fb4f136b4eb67a8923e03a19fbdece3390816.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/5c94ddef1e5721fd063a3eb0d96fb4f136b4eb67a8923e03a19fbdece3390816.jpg)

![66e384847972d1291b69fad994eae77d53b91cc81d69672edc38500b6c8a3b2e.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/66e384847972d1291b69fad994eae77d53b91cc81d69672edc38500b6c8a3b2e.jpg)

![7a1b6a58611960abaa8e074f9bfb8a38360f4d52f4c5e3e7071fb00064dff077.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/7a1b6a58611960abaa8e074f9bfb8a38360f4d52f4c5e3e7071fb00064dff077.jpg)

![7eac53669b176b1b89663a45bd5570f4554c99a2efcb04a3dde19b3b797de3fa.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/7eac53669b176b1b89663a45bd5570f4554c99a2efcb04a3dde19b3b797de3fa.jpg)

![8352661cf744faba87d43b8a04a9414aec29ea0632296fddc8f4ea9a41a97468.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/8352661cf744faba87d43b8a04a9414aec29ea0632296fddc8f4ea9a41a97468.jpg)

![8b032e72b13ee33df98c361b74496dc6b28b1d7c61a18e49033e58e3b0332f55.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/8b032e72b13ee33df98c361b74496dc6b28b1d7c61a18e49033e58e3b0332f55.jpg)

![91d1f72aebf3fff3e1adfea9032ec8f484dd1114b3946e3fe4fd0bd09f7d24f8.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/91d1f72aebf3fff3e1adfea9032ec8f484dd1114b3946e3fe4fd0bd09f7d24f8.jpg)

![9b9045685fe9f9af1b7624a420419c561c82c588f8285398d24371478df21040.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/9b9045685fe9f9af1b7624a420419c561c82c588f8285398d24371478df21040.jpg)

![9d593e141b44c257b652ccb7395eeb994dd93c3a04dbb122b3305f72d61c11db.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/9d593e141b44c257b652ccb7395eeb994dd93c3a04dbb122b3305f72d61c11db.jpg)

![a7375a376e70e7a338b1c9c2aa5efcaf16e5f4d50c36949ccdbdf4314173d744.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/a7375a376e70e7a338b1c9c2aa5efcaf16e5f4d50c36949ccdbdf4314173d744.jpg)

![b4f9b90cff22d66c346988885782063de8ce31d4d8db9bae579be3ae2d76c6cc.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/b4f9b90cff22d66c346988885782063de8ce31d4d8db9bae579be3ae2d76c6cc.jpg)

![bf2e9e2cb99e0ca8bc06cb376a3112f9925d071f49a25c562393ff7ba72bf527.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/bf2e9e2cb99e0ca8bc06cb376a3112f9925d071f49a25c562393ff7ba72bf527.jpg)

![c56be06e9b9a48e3c37a77ff7d052faeee88821c4d61d1546f2b6e1867543f2f.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/c56be06e9b9a48e3c37a77ff7d052faeee88821c4d61d1546f2b6e1867543f2f.jpg)

![c6bf6be1406b9190adcdd75259fb850f31ebad84744f82dc8ca9ec8e35946574.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/c6bf6be1406b9190adcdd75259fb850f31ebad84744f82dc8ca9ec8e35946574.jpg)

![cb869629818ecc987ce0f7b97f1808b5b92e0693ed8c886c4002db131c8b9365.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/cb869629818ecc987ce0f7b97f1808b5b92e0693ed8c886c4002db131c8b9365.jpg)

![d194b6469b1e4d7c2b621274994d2d9e8b6149a18cf803ad1b4730fa23b7e640.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/d194b6469b1e4d7c2b621274994d2d9e8b6149a18cf803ad1b4730fa23b7e640.jpg)

![da778c65f4fc744ebe3447bb3daef4993eb15242bfd93228d09839add1fadc0f.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/da778c65f4fc744ebe3447bb3daef4993eb15242bfd93228d09839add1fadc0f.jpg)

![f5ca515ce28a0544478531372f3d2c96a05e45e63ab7cfa976256a1c982b8f08.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/f5ca515ce28a0544478531372f3d2c96a05e45e63ab7cfa976256a1c982b8f08.jpg)

![f7f190086587c167b1870e16b47bf3674ebd460ee0ad6ed0e717282cbfa841c2.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/f7f190086587c167b1870e16b47bf3674ebd460ee0ad6ed0e717282cbfa841c2.jpg)

![f87df6dd66c79c6c396f6a7fe9ee8391feeb27ceb8e8a235afb73444e0d2c646.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/f87df6dd66c79c6c396f6a7fe9ee8391feeb27ceb8e8a235afb73444e0d2c646.jpg)

![fdf569bc440df890d1afe81d23ca551800eab38eae99c8cc26b9ed83b845593e.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/images/fdf569bc440df890d1afe81d23ca551800eab38eae99c8cc26b9ed83b845593e.jpg)

### Tables

![09935fb1b6d1a8caea2654bc7e6ffded15ea5d8f23348a89b18f6cf27e465836.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/tables/09935fb1b6d1a8caea2654bc7e6ffded15ea5d8f23348a89b18f6cf27e465836.jpg)

![1f9993357d9bc3cef682672277103b39ef69a298e4c489b1ae79c701e17928c2.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/tables/1f9993357d9bc3cef682672277103b39ef69a298e4c489b1ae79c701e17928c2.jpg)

![355e0c3cc3b1dbdb88481ec7355847a6ff22704985ded04d5eb68f7266b2dba6.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/tables/355e0c3cc3b1dbdb88481ec7355847a6ff22704985ded04d5eb68f7266b2dba6.jpg)

![5f083ff0a34a86c1310f134e85cdf5c388ac696633b40eb643c48bdce7773bbb.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/tables/5f083ff0a34a86c1310f134e85cdf5c388ac696633b40eb643c48bdce7773bbb.jpg)

![f76febf046a113848c7e4207c2b8528f84da0e6f2c17cf16253b31ac514bc66a.jpg](../iclr_results/1683_Looking Inward_ Language Models Can Learn About Themselves by Introspection/tables/f76febf046a113848c7e4207c2b8528f84da0e6f2c17cf16253b31ac514bc66a.jpg)

## DiffSplat: Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation


### Images

![105a5d62b33020575e1dcd6dad2d708108be20e8278428602a6f7b68c6668008.jpg](../iclr_results/1684_DiffSplat_ Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation/images/105a5d62b33020575e1dcd6dad2d708108be20e8278428602a6f7b68c6668008.jpg)

![20ae8e5f75b329f6d9498ef25ef056f6c9754a9af93c87082c0e5d753ae1dbdb.jpg](../iclr_results/1684_DiffSplat_ Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation/images/20ae8e5f75b329f6d9498ef25ef056f6c9754a9af93c87082c0e5d753ae1dbdb.jpg)

![2830812c63cf0150cc994405b1aa68729436d4724749abe219de10dd2592b9d3.jpg](../iclr_results/1684_DiffSplat_ Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation/images/2830812c63cf0150cc994405b1aa68729436d4724749abe219de10dd2592b9d3.jpg)

![2ce221f82e2d558eddf09b9a37e9d843eba4d9044fda58f81b3ef2f4b5857950.jpg](../iclr_results/1684_DiffSplat_ Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation/images/2ce221f82e2d558eddf09b9a37e9d843eba4d9044fda58f81b3ef2f4b5857950.jpg)

![39ee6c1e37207991d7ba1e03f406aa0e2554a9760feca684037068dd291fbcfd.jpg](../iclr_results/1684_DiffSplat_ Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation/images/39ee6c1e37207991d7ba1e03f406aa0e2554a9760feca684037068dd291fbcfd.jpg)

![41961b2b2485782bcc6cb88a725ffc517156dc7974529254e0f1ab959aa2d17e.jpg](../iclr_results/1684_DiffSplat_ Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation/images/41961b2b2485782bcc6cb88a725ffc517156dc7974529254e0f1ab959aa2d17e.jpg)

![6aa74c537905fb5e25e15a9784619db60a61ee06abbcbd1b2ebdf9ed9602ae53.jpg](../iclr_results/1684_DiffSplat_ Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation/images/6aa74c537905fb5e25e15a9784619db60a61ee06abbcbd1b2ebdf9ed9602ae53.jpg)

![6c2e26b8a299c8e7ee44a5c0ca5f97c314a76559949172918da47e7203b37cf2.jpg](../iclr_results/1684_DiffSplat_ Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation/images/6c2e26b8a299c8e7ee44a5c0ca5f97c314a76559949172918da47e7203b37cf2.jpg)

![814ec7ad064fe5de668708e2c68ae840ca53aa1d9326366c1f7e34dc75d0d760.jpg](../iclr_results/1684_DiffSplat_ Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation/images/814ec7ad064fe5de668708e2c68ae840ca53aa1d9326366c1f7e34dc75d0d760.jpg)

![8741071b7a57ac66e3bbe5d00dcaca5dc78c4bd2bf5a6a3e767d8c22da5e0121.jpg](../iclr_results/1684_DiffSplat_ Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation/images/8741071b7a57ac66e3bbe5d00dcaca5dc78c4bd2bf5a6a3e767d8c22da5e0121.jpg)

![c24a372281e83286ffee1d467aadd2c2dd2a0aca99caa63d63ffc91533e90cce.jpg](../iclr_results/1684_DiffSplat_ Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation/images/c24a372281e83286ffee1d467aadd2c2dd2a0aca99caa63d63ffc91533e90cce.jpg)

![e1533e887c97bafe876be74873d8cdf6792d73e46d61fd5074dd9d02dfe1dc00.jpg](../iclr_results/1684_DiffSplat_ Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation/images/e1533e887c97bafe876be74873d8cdf6792d73e46d61fd5074dd9d02dfe1dc00.jpg)

![e760629793d75a94d2dbe8de907d5e9a475c3e1dd55e0002e51475e528f0a5c9.jpg](../iclr_results/1684_DiffSplat_ Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation/images/e760629793d75a94d2dbe8de907d5e9a475c3e1dd55e0002e51475e528f0a5c9.jpg)

![f770e20abf09f77e200473651afa6f86010e66b2df14d75a500a3dd675e11dc1.jpg](../iclr_results/1684_DiffSplat_ Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation/images/f770e20abf09f77e200473651afa6f86010e66b2df14d75a500a3dd675e11dc1.jpg)

### Tables

![078db9881b267cddc7eac071d8ec26cf8f5a7cd76f05cfb8d99b21ce4d0c54f4.jpg](../iclr_results/1684_DiffSplat_ Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation/tables/078db9881b267cddc7eac071d8ec26cf8f5a7cd76f05cfb8d99b21ce4d0c54f4.jpg)

![162c5d6bddac68f7669c3756806144d005fe1a85bc461766270d6f9b7a709beb.jpg](../iclr_results/1684_DiffSplat_ Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation/tables/162c5d6bddac68f7669c3756806144d005fe1a85bc461766270d6f9b7a709beb.jpg)

![8ca20dd1b7af4b3aebfd4b407d4b2c9f6c47c1dc9e02031bdc2815a056bc2fb0.jpg](../iclr_results/1684_DiffSplat_ Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation/tables/8ca20dd1b7af4b3aebfd4b407d4b2c9f6c47c1dc9e02031bdc2815a056bc2fb0.jpg)

![b5e1e267a6260e49f4299f67a70873c5aa28c2297216b1099bfd737ce0e202f9.jpg](../iclr_results/1684_DiffSplat_ Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation/tables/b5e1e267a6260e49f4299f67a70873c5aa28c2297216b1099bfd737ce0e202f9.jpg)

![da7491b58083b1d602ca3aa818d8cfd3b74ed34682af4febbd9885a6f17b14b4.jpg](../iclr_results/1684_DiffSplat_ Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation/tables/da7491b58083b1d602ca3aa818d8cfd3b74ed34682af4febbd9885a6f17b14b4.jpg)

## Mitigating Parameter Interference in Model Merging via Sharpness-Aware Fine-Tuning


### Images

![170cc13d2e402e0c35a2a12e5797f9ca4bd2f118c22b53a9aa08eac7f24ec303.jpg](../iclr_results/1685_Mitigating Parameter Interference in Model Merging via Sharpness-Aware Fine-Tuning/images/170cc13d2e402e0c35a2a12e5797f9ca4bd2f118c22b53a9aa08eac7f24ec303.jpg)

![6124597a673adf5c7d583766b0f467882dadc51d379e3cb52ef93c43b39f0437.jpg](../iclr_results/1685_Mitigating Parameter Interference in Model Merging via Sharpness-Aware Fine-Tuning/images/6124597a673adf5c7d583766b0f467882dadc51d379e3cb52ef93c43b39f0437.jpg)

![6d443ca55eef8aa824a4f06562690073f2947f80f55fb7df8b63c7a77234f006.jpg](../iclr_results/1685_Mitigating Parameter Interference in Model Merging via Sharpness-Aware Fine-Tuning/images/6d443ca55eef8aa824a4f06562690073f2947f80f55fb7df8b63c7a77234f006.jpg)

![7f129229bfeb5a33dd7814a907f4b98e522bd0442a3a7d5161e4c7ced225efbe.jpg](../iclr_results/1685_Mitigating Parameter Interference in Model Merging via Sharpness-Aware Fine-Tuning/images/7f129229bfeb5a33dd7814a907f4b98e522bd0442a3a7d5161e4c7ced225efbe.jpg)

![8a731610a6b8e7f257ddf70257d1073737c09580f612aac8ea17b741cedee7ef.jpg](../iclr_results/1685_Mitigating Parameter Interference in Model Merging via Sharpness-Aware Fine-Tuning/images/8a731610a6b8e7f257ddf70257d1073737c09580f612aac8ea17b741cedee7ef.jpg)

![c9dc479b780f247c76177bcf22552224c8524fcbe73e26ba5eb293cff64a75dd.jpg](../iclr_results/1685_Mitigating Parameter Interference in Model Merging via Sharpness-Aware Fine-Tuning/images/c9dc479b780f247c76177bcf22552224c8524fcbe73e26ba5eb293cff64a75dd.jpg)

![c9e7f843473537930d2b1fe9b4769a1d32c1a803898b1f0b5d572fd398012afc.jpg](../iclr_results/1685_Mitigating Parameter Interference in Model Merging via Sharpness-Aware Fine-Tuning/images/c9e7f843473537930d2b1fe9b4769a1d32c1a803898b1f0b5d572fd398012afc.jpg)

![d07c44982d53bb0bfe445b8fd3cd19af5e2b369094c3ff024fcc98369dd87a25.jpg](../iclr_results/1685_Mitigating Parameter Interference in Model Merging via Sharpness-Aware Fine-Tuning/images/d07c44982d53bb0bfe445b8fd3cd19af5e2b369094c3ff024fcc98369dd87a25.jpg)

### Tables

![17c1835b189a8430d086160d75df5188f032b90f629d9df0c03677721f35d45d.jpg](../iclr_results/1685_Mitigating Parameter Interference in Model Merging via Sharpness-Aware Fine-Tuning/tables/17c1835b189a8430d086160d75df5188f032b90f629d9df0c03677721f35d45d.jpg)

![3b0641cfb0e692670145678ace1114066de5ad4b0a1c271bc25a0926dbbeb656.jpg](../iclr_results/1685_Mitigating Parameter Interference in Model Merging via Sharpness-Aware Fine-Tuning/tables/3b0641cfb0e692670145678ace1114066de5ad4b0a1c271bc25a0926dbbeb656.jpg)

![68d36c9f4e519df4392f21b2d894c30be8f0507370407f378f0e38b8cbbc1380.jpg](../iclr_results/1685_Mitigating Parameter Interference in Model Merging via Sharpness-Aware Fine-Tuning/tables/68d36c9f4e519df4392f21b2d894c30be8f0507370407f378f0e38b8cbbc1380.jpg)

![7249b59651e0cd3fce77a2ad1ee8cd0f2d7f701551f1d2fbce714de9fdcea4b5.jpg](../iclr_results/1685_Mitigating Parameter Interference in Model Merging via Sharpness-Aware Fine-Tuning/tables/7249b59651e0cd3fce77a2ad1ee8cd0f2d7f701551f1d2fbce714de9fdcea4b5.jpg)

![7d2f762c7bbba03c8a1d9fc0c8f90bf8af0667ca523fc4f2277b8c989644a5a6.jpg](../iclr_results/1685_Mitigating Parameter Interference in Model Merging via Sharpness-Aware Fine-Tuning/tables/7d2f762c7bbba03c8a1d9fc0c8f90bf8af0667ca523fc4f2277b8c989644a5a6.jpg)

![9430eb7d3b5463a38eec9155fee3a53e654a1fc99bee32ea0a5d9eb113433667.jpg](../iclr_results/1685_Mitigating Parameter Interference in Model Merging via Sharpness-Aware Fine-Tuning/tables/9430eb7d3b5463a38eec9155fee3a53e654a1fc99bee32ea0a5d9eb113433667.jpg)

![b15b6a5a930dcbd9e4c2ead119d109d6a52b489abb5973e1b564ca94094309e1.jpg](../iclr_results/1685_Mitigating Parameter Interference in Model Merging via Sharpness-Aware Fine-Tuning/tables/b15b6a5a930dcbd9e4c2ead119d109d6a52b489abb5973e1b564ca94094309e1.jpg)

![b70a2c7a393765988e1326c43b7b2527abeda503a1c1b4aaae8c57414e9ce2c3.jpg](../iclr_results/1685_Mitigating Parameter Interference in Model Merging via Sharpness-Aware Fine-Tuning/tables/b70a2c7a393765988e1326c43b7b2527abeda503a1c1b4aaae8c57414e9ce2c3.jpg)

![ff31f7467cc89e6f5aa32b05557740d34f698dbc24820951aaa0cc0d51bd1e69.jpg](../iclr_results/1685_Mitigating Parameter Interference in Model Merging via Sharpness-Aware Fine-Tuning/tables/ff31f7467cc89e6f5aa32b05557740d34f698dbc24820951aaa0cc0d51bd1e69.jpg)

## Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood


### Images

![14896d6cc8412fc870a739b5944c00d27b0855ce6f046c052213af86f011ef1c.jpg](../iclr_results/1686_Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood/images/14896d6cc8412fc870a739b5944c00d27b0855ce6f046c052213af86f011ef1c.jpg)

![1aa797c87cac063330656eb9112cfd43e8a9d36be16250e50102b200d30381a7.jpg](../iclr_results/1686_Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood/images/1aa797c87cac063330656eb9112cfd43e8a9d36be16250e50102b200d30381a7.jpg)

![3314185a3698fb72b346df7a4293ba95d97464d62d51dd1d90189e7214b0d513.jpg](../iclr_results/1686_Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood/images/3314185a3698fb72b346df7a4293ba95d97464d62d51dd1d90189e7214b0d513.jpg)

![b1181b8cb65ff8bd3151861b17274fb71ed4ca1321658901798e411b5b4cc1eb.jpg](../iclr_results/1686_Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood/images/b1181b8cb65ff8bd3151861b17274fb71ed4ca1321658901798e411b5b4cc1eb.jpg)

### Tables

![0b7338d31bbc0b5b487a3b7e5e521882ed02f94b135f65543aff44923f4a0d30.jpg](../iclr_results/1686_Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood/tables/0b7338d31bbc0b5b487a3b7e5e521882ed02f94b135f65543aff44923f4a0d30.jpg)

![2c8bd023a2b2bf1f1e8100a7d8a101f064b13997de362f2a90d28ca75520c80a.jpg](../iclr_results/1686_Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood/tables/2c8bd023a2b2bf1f1e8100a7d8a101f064b13997de362f2a90d28ca75520c80a.jpg)

![3e682bab9b921020d192373e3c387c1b0c9a6b9d28343153295fc1f95a940112.jpg](../iclr_results/1686_Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood/tables/3e682bab9b921020d192373e3c387c1b0c9a6b9d28343153295fc1f95a940112.jpg)

![53c150e28852c60af7d817a4aebcc042ef567b94a9e2ced184bdbf37981bf705.jpg](../iclr_results/1686_Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood/tables/53c150e28852c60af7d817a4aebcc042ef567b94a9e2ced184bdbf37981bf705.jpg)

![53e1522358b80deecf94640d6cbc4f87afd76ba3595c0a3064e0ccb43e02ed83.jpg](../iclr_results/1686_Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood/tables/53e1522358b80deecf94640d6cbc4f87afd76ba3595c0a3064e0ccb43e02ed83.jpg)

![70c8bd5dfdea455a9b96c20be8e7ce177736cc688e15bf251903f398df14e6cc.jpg](../iclr_results/1686_Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood/tables/70c8bd5dfdea455a9b96c20be8e7ce177736cc688e15bf251903f398df14e6cc.jpg)

![9c12649eb630bec449fd81934e39e411bf2c44b70de685844984f1fdf85f78c4.jpg](../iclr_results/1686_Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood/tables/9c12649eb630bec449fd81934e39e411bf2c44b70de685844984f1fdf85f78c4.jpg)

![9f7c5c96c7c7a4813796ae57f3ab7ea6065297d224993d5f3e4bbc9387aeeb7b.jpg](../iclr_results/1686_Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood/tables/9f7c5c96c7c7a4813796ae57f3ab7ea6065297d224993d5f3e4bbc9387aeeb7b.jpg)

![f1c1d6e3373a33d41429cdcf467efb5a5659867422d8fab0ed135af1a03289a6.jpg](../iclr_results/1686_Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood/tables/f1c1d6e3373a33d41429cdcf467efb5a5659867422d8fab0ed135af1a03289a6.jpg)

![f27c9ffc4a7b2b085a894a37c65252372ee048757d5486369ed82f3892747e2b.jpg](../iclr_results/1686_Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood/tables/f27c9ffc4a7b2b085a894a37c65252372ee048757d5486369ed82f3892747e2b.jpg)

![f43a6b78a9fce5ffa408ce89dfe4be11ec882f7960f84242f72f6d2bf4f0abf4.jpg](../iclr_results/1686_Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood/tables/f43a6b78a9fce5ffa408ce89dfe4be11ec882f7960f84242f72f6d2bf4f0abf4.jpg)

## How Does Vision-Language Adaptation Impact the Safety of Vision Language Models?


### Images

![26d9b717bef82a325fc2dcf03cad1fa149662a01183e3a041cac6145ba3f64a5.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/images/26d9b717bef82a325fc2dcf03cad1fa149662a01183e3a041cac6145ba3f64a5.jpg)

![4ff32678d70d97aefcca19a54123e4b7111a42f7220d92917fb7ea564f50b11e.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/images/4ff32678d70d97aefcca19a54123e4b7111a42f7220d92917fb7ea564f50b11e.jpg)

![5de6c6de34921e1eb4d20d0040c98af8b88d1100121e974608db6f1881d6d2ea.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/images/5de6c6de34921e1eb4d20d0040c98af8b88d1100121e974608db6f1881d6d2ea.jpg)

![6fb97fd6e14be5a932a4de1d4704af6d22d61b008fb9f24d8c3d6843c744bd77.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/images/6fb97fd6e14be5a932a4de1d4704af6d22d61b008fb9f24d8c3d6843c744bd77.jpg)

![79287db737c05e35f293ad29681650659f6c6f8c201b5ea30a0719489e9473a6.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/images/79287db737c05e35f293ad29681650659f6c6f8c201b5ea30a0719489e9473a6.jpg)

![8aa70e2567e0e608b7ac0f4f711d885ce39de69d4234e5cfa097f7d9e4726636.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/images/8aa70e2567e0e608b7ac0f4f711d885ce39de69d4234e5cfa097f7d9e4726636.jpg)

![a44929f4e70f3d508aea07fd99cb5f6f7ad088e358ec4b89b25a5544166f45c4.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/images/a44929f4e70f3d508aea07fd99cb5f6f7ad088e358ec4b89b25a5544166f45c4.jpg)

![b4dc61f8c04211ce1453cf8e2ee99cfa45c6903bd2aaf33ccb6893e8f29d4c54.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/images/b4dc61f8c04211ce1453cf8e2ee99cfa45c6903bd2aaf33ccb6893e8f29d4c54.jpg)

![c366da2d67337de539bff0ce11e7b8c7498d2ef73fb12c7fef5ae26febcc26a0.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/images/c366da2d67337de539bff0ce11e7b8c7498d2ef73fb12c7fef5ae26febcc26a0.jpg)

![ccb8e7837f8f64690391e535aa21ecbe2173d868060071178a410d2767249f33.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/images/ccb8e7837f8f64690391e535aa21ecbe2173d868060071178a410d2767249f33.jpg)

![d3aa106271084157cbe5a0889648f8c3d041e3382fa908be583f54a6be09acab.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/images/d3aa106271084157cbe5a0889648f8c3d041e3382fa908be583f54a6be09acab.jpg)

![f9c1b273b0368de4f87e3f59db4ae8cd0c22e504bdfe60a42dcbdce0da33c970.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/images/f9c1b273b0368de4f87e3f59db4ae8cd0c22e504bdfe60a42dcbdce0da33c970.jpg)

![fa8abd1a73bc2232b074672718c9e7f561a42a3c63b93ab2cba09b44ad03cc0b.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/images/fa8abd1a73bc2232b074672718c9e7f561a42a3c63b93ab2cba09b44ad03cc0b.jpg)

![fd57ffcb960ec5f3f61ec23856524ee73ae88944ff43bed06551abc5452f8d6b.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/images/fd57ffcb960ec5f3f61ec23856524ee73ae88944ff43bed06551abc5452f8d6b.jpg)

### Tables

![385c95a42eb5750ebd91a56f3d315a72455218ce9643544d29b41a1601a2d708.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/tables/385c95a42eb5750ebd91a56f3d315a72455218ce9643544d29b41a1601a2d708.jpg)

![437e102014123d3c1ed97fe47f30b21824e8d5637043b7b3d33e247de6521ec0.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/tables/437e102014123d3c1ed97fe47f30b21824e8d5637043b7b3d33e247de6521ec0.jpg)

![4a7945239e24150c803b19ed3b80984a14c0993d733eb1adf6d2baa60e9f6c53.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/tables/4a7945239e24150c803b19ed3b80984a14c0993d733eb1adf6d2baa60e9f6c53.jpg)

![4c679bb58ada9f5b59a3cea4aa4b3bfbbbb0bb5ff7643279837e110e12128fae.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/tables/4c679bb58ada9f5b59a3cea4aa4b3bfbbbb0bb5ff7643279837e110e12128fae.jpg)

![59ed149f9142e162b5168ee5e8e0013b3a7e8f4022a793594d71594213593db6.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/tables/59ed149f9142e162b5168ee5e8e0013b3a7e8f4022a793594d71594213593db6.jpg)

![5c0bc60484ffee03afaf319cc522337418c1c0148f94d2fd9ef29e05dd5c4a18.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/tables/5c0bc60484ffee03afaf319cc522337418c1c0148f94d2fd9ef29e05dd5c4a18.jpg)

![5d0edef43be8168056a3f4f1099e26a721abcd25f383ce1383651b63c1d7f8b6.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/tables/5d0edef43be8168056a3f4f1099e26a721abcd25f383ce1383651b63c1d7f8b6.jpg)

![a020b0f37191a8b311bf2a54bd27dc84488fee288cb313cddd4f439398d164a2.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/tables/a020b0f37191a8b311bf2a54bd27dc84488fee288cb313cddd4f439398d164a2.jpg)

![a3b295325cde1ebe76f9bca52d82673383f7001fb4f194d38dc2e2b6c3b6a6d7.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/tables/a3b295325cde1ebe76f9bca52d82673383f7001fb4f194d38dc2e2b6c3b6a6d7.jpg)

![aa3f6224c7a63122826a2d6dc8c33b3d86c854d3946377115c8f3942f202e015.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/tables/aa3f6224c7a63122826a2d6dc8c33b3d86c854d3946377115c8f3942f202e015.jpg)

![c3d12ce3201b4a299c0c0f3349da7a55a84f3737770ca0945507bdb2a3d06ca4.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/tables/c3d12ce3201b4a299c0c0f3349da7a55a84f3737770ca0945507bdb2a3d06ca4.jpg)

![c5e83b9e6f71958ee4d42c551921979217bd0be7ac22207c7b14914bbef49332.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/tables/c5e83b9e6f71958ee4d42c551921979217bd0be7ac22207c7b14914bbef49332.jpg)

![c7bbe409b4bc4917399d49431e7386e8cb2e79090712ec5323fd805aade4a5d7.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/tables/c7bbe409b4bc4917399d49431e7386e8cb2e79090712ec5323fd805aade4a5d7.jpg)

![efaadc60b19a7952e4cdb425de1160d32d15b3ab51cd721612e4e0693dcc2003.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/tables/efaadc60b19a7952e4cdb425de1160d32d15b3ab51cd721612e4e0693dcc2003.jpg)

![f69c392b0f4b5ed56f529cc3fa9c19c6156997fd2fa2ae70194e49b98359bd2d.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/tables/f69c392b0f4b5ed56f529cc3fa9c19c6156997fd2fa2ae70194e49b98359bd2d.jpg)

![ffb76b2df7a905b1b20bac18d4198d68ced331c052b4ce9bff8beaa007b3464c.jpg](../iclr_results/1687_How Does Vision-Language Adaptation Impact the Safety of Vision Language Models_/tables/ffb76b2df7a905b1b20bac18d4198d68ced331c052b4ce9bff8beaa007b3464c.jpg)

## Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection


### Images

![8f3765cce3917be2f882a94cf6151ac14cd45d6a561486bb08c499968b4fc7d7.jpg](../iclr_results/1688_Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection/images/8f3765cce3917be2f882a94cf6151ac14cd45d6a561486bb08c499968b4fc7d7.jpg)

![9366b8d2524bb2a033a646fc2bc6c9d4dd0ad2cbe5527f3fa7b80ac5afc6ddf5.jpg](../iclr_results/1688_Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection/images/9366b8d2524bb2a033a646fc2bc6c9d4dd0ad2cbe5527f3fa7b80ac5afc6ddf5.jpg)

![a9329fa7a6c55814d7793d0338e6fe752b519b5b3f75aac6b204297e01580489.jpg](../iclr_results/1688_Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection/images/a9329fa7a6c55814d7793d0338e6fe752b519b5b3f75aac6b204297e01580489.jpg)

![ca7c42d7d13de242b8a0b0ce2bd934cb6ab397aad3a21ef2a38bbcad596a2457.jpg](../iclr_results/1688_Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection/images/ca7c42d7d13de242b8a0b0ce2bd934cb6ab397aad3a21ef2a38bbcad596a2457.jpg)

![d53e4de6c0c5f05201ee43b5e662c500115d28be8a90fe09ddb0b286eee76eba.jpg](../iclr_results/1688_Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection/images/d53e4de6c0c5f05201ee43b5e662c500115d28be8a90fe09ddb0b286eee76eba.jpg)

![ecb6cb697e08e8ad59f0ebd4a44955b249789cecf8b070d32678d2b2a52dedb8.jpg](../iclr_results/1688_Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection/images/ecb6cb697e08e8ad59f0ebd4a44955b249789cecf8b070d32678d2b2a52dedb8.jpg)

![fe935bbe1764532040b6709eb18abfb0beb5a29235e88a1ee8c71b9660ea2229.jpg](../iclr_results/1688_Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection/images/fe935bbe1764532040b6709eb18abfb0beb5a29235e88a1ee8c71b9660ea2229.jpg)

### Tables

![116b6fb374906a5d4dd7d48ac412eb53ea7cb0da5519f27c0eda96e5cb3e33a2.jpg](../iclr_results/1688_Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection/tables/116b6fb374906a5d4dd7d48ac412eb53ea7cb0da5519f27c0eda96e5cb3e33a2.jpg)

![179d5f916e135c73e6235ee8a6a3738e92e958c2bd3b0b985da7b608b6a3daed.jpg](../iclr_results/1688_Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection/tables/179d5f916e135c73e6235ee8a6a3738e92e958c2bd3b0b985da7b608b6a3daed.jpg)

![189c116b2851be7a503aac0f60c99f9c44101b23ec1a9517f168134d309040f4.jpg](../iclr_results/1688_Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection/tables/189c116b2851be7a503aac0f60c99f9c44101b23ec1a9517f168134d309040f4.jpg)

![2dea3bf172f63053c5675206a5b5c6edc0eaf2845cf9bc26312bb9b6b646024e.jpg](../iclr_results/1688_Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection/tables/2dea3bf172f63053c5675206a5b5c6edc0eaf2845cf9bc26312bb9b6b646024e.jpg)

![51aadb72e6037f353a186f6d5075098b3aef2bedf33ff54a0c44eebf9763c41f.jpg](../iclr_results/1688_Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection/tables/51aadb72e6037f353a186f6d5075098b3aef2bedf33ff54a0c44eebf9763c41f.jpg)

![70b4d46ef8b79ee9321e336b092b496ae290e95627384da2f872d1993e15251a.jpg](../iclr_results/1688_Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection/tables/70b4d46ef8b79ee9321e336b092b496ae290e95627384da2f872d1993e15251a.jpg)

![775ada98fcfa62f86f610327b2d33450478057a730e99433b16173ecdd1a6292.jpg](../iclr_results/1688_Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection/tables/775ada98fcfa62f86f610327b2d33450478057a730e99433b16173ecdd1a6292.jpg)

![88916309214fc0d874c3da81f9041e044a2fb0d3a4f33134f0e172f488292408.jpg](../iclr_results/1688_Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection/tables/88916309214fc0d874c3da81f9041e044a2fb0d3a4f33134f0e172f488292408.jpg)

![8b8d3cd971e914f9420fdb3b2b20078e446bcf25a279f157aba8aa4b7bb45564.jpg](../iclr_results/1688_Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection/tables/8b8d3cd971e914f9420fdb3b2b20078e446bcf25a279f157aba8aa4b7bb45564.jpg)

![90fd9e6dc2d9de708418178da6d97f91a5c8e8bb8a2095db6e20ed48edd55171.jpg](../iclr_results/1688_Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection/tables/90fd9e6dc2d9de708418178da6d97f91a5c8e8bb8a2095db6e20ed48edd55171.jpg)

![984c1f765c19e7ea32a2dbc8465786eccae80931cea2c8e38744638dc13bd807.jpg](../iclr_results/1688_Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection/tables/984c1f765c19e7ea32a2dbc8465786eccae80931cea2c8e38744638dc13bd807.jpg)

![9993c125ac17836df7f14c799391885a734200486b9465f16d1d9a465bd71470.jpg](../iclr_results/1688_Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection/tables/9993c125ac17836df7f14c799391885a734200486b9465f16d1d9a465bd71470.jpg)

![a6cef6fd0c2da2801b3c13ddf2dcb2150ccd3fadaaa1616c6f3fa28082823eea.jpg](../iclr_results/1688_Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection/tables/a6cef6fd0c2da2801b3c13ddf2dcb2150ccd3fadaaa1616c6f3fa28082823eea.jpg)

![b7897f4234cd6a6f65ca130f44cc8d7ff25154483f3bce77e7a1248e9ca6dd6a.jpg](../iclr_results/1688_Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection/tables/b7897f4234cd6a6f65ca130f44cc8d7ff25154483f3bce77e7a1248e9ca6dd6a.jpg)

![bb765a076bbeecd97dd0945b2eb52fddf5cbc205e7874152bfb7fc4032ce7e46.jpg](../iclr_results/1688_Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection/tables/bb765a076bbeecd97dd0945b2eb52fddf5cbc205e7874152bfb7fc4032ce7e46.jpg)

![e470e87c01fe53f39ac163a7302d3172ec1cb42da375c869d98c4180fe04acbf.jpg](../iclr_results/1688_Multi-Resolution Decomposable Diffusion Model for Non-Stationary Time Series Anomaly Detection/tables/e470e87c01fe53f39ac163a7302d3172ec1cb42da375c869d98c4180fe04acbf.jpg)

## Layerwise Recurrent Router for  Mixture-of-Experts


### Images

![2345dbd27b740aaa96e375ed72a968f759889041e70040f83132f47e6dc2cf2e.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/images/2345dbd27b740aaa96e375ed72a968f759889041e70040f83132f47e6dc2cf2e.jpg)

![437ee907b719ce47e0024f7d4fe0a017d51e85103a1a2cc1bb11a054db797898.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/images/437ee907b719ce47e0024f7d4fe0a017d51e85103a1a2cc1bb11a054db797898.jpg)

![4952328debe5a38ef471f94086898746463825247727a9b634feaa2f17ed5e13.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/images/4952328debe5a38ef471f94086898746463825247727a9b634feaa2f17ed5e13.jpg)

![4c6376ed9d0f4e3e10958287e802ed93601a9a5b01fe521d9802a96a97adf152.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/images/4c6376ed9d0f4e3e10958287e802ed93601a9a5b01fe521d9802a96a97adf152.jpg)

![6a33365a2e49e67cc38b70c4eaa299830ef5a9aa8b32fe7789e4ac4def7ba4ad.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/images/6a33365a2e49e67cc38b70c4eaa299830ef5a9aa8b32fe7789e4ac4def7ba4ad.jpg)

![7e940104e70b689d659956a456117a01293819aea5a498f2835c1bc5df6afef8.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/images/7e940104e70b689d659956a456117a01293819aea5a498f2835c1bc5df6afef8.jpg)

![84719e143b1ac4043f877c4336e6ab393ae7e045c244456461faffe06d5fbf99.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/images/84719e143b1ac4043f877c4336e6ab393ae7e045c244456461faffe06d5fbf99.jpg)

![89755abafb7656157f2f440c4ffce6f5d5463f1df053d26596df040238774b43.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/images/89755abafb7656157f2f440c4ffce6f5d5463f1df053d26596df040238774b43.jpg)

![aad88a134f67f162e9560579ba16eea04abb7f7b2e880d24a36f8f9c2b2e067b.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/images/aad88a134f67f162e9560579ba16eea04abb7f7b2e880d24a36f8f9c2b2e067b.jpg)

![be1135d84cb21852bbe328cde06451384c91ce7eca5eafe7677ebc747b2bb386.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/images/be1135d84cb21852bbe328cde06451384c91ce7eca5eafe7677ebc747b2bb386.jpg)

![bed61566738375fef7a2216125595f8768db08f49b334c3d5eb87ef0b4a5d792.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/images/bed61566738375fef7a2216125595f8768db08f49b334c3d5eb87ef0b4a5d792.jpg)

![deb166949d461dd12f840cbc8edaa8fc67ec716509cba4801b600d5ccc8c1c83.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/images/deb166949d461dd12f840cbc8edaa8fc67ec716509cba4801b600d5ccc8c1c83.jpg)

![e1b468d7bf2c04e004f4181b072b73d4d2c17160ef9ee3cc002457d7c1d91505.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/images/e1b468d7bf2c04e004f4181b072b73d4d2c17160ef9ee3cc002457d7c1d91505.jpg)

![e77e4081a2d2c3a44471fefd3d16a14a19fc7a2ab427f31a2aa1c957a26f13df.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/images/e77e4081a2d2c3a44471fefd3d16a14a19fc7a2ab427f31a2aa1c957a26f13df.jpg)

### Tables

![0b25aee10b030b981d9b9b8f48b77b0bd6de44956622e6d3ba413ccabb6975a2.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/tables/0b25aee10b030b981d9b9b8f48b77b0bd6de44956622e6d3ba413ccabb6975a2.jpg)

![3d7c60b0b4214c71a9104cade80d36e524d77a9d2c0d9018db662ce5f58cd5dd.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/tables/3d7c60b0b4214c71a9104cade80d36e524d77a9d2c0d9018db662ce5f58cd5dd.jpg)

![4b3f597cdcb31a7ed37f9943ebd48a5e69add9f3a05694c00da3abc59aca24bf.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/tables/4b3f597cdcb31a7ed37f9943ebd48a5e69add9f3a05694c00da3abc59aca24bf.jpg)

![70d927ebdb060dba54674a282669dde711e8fe7ab5bd0cc06561284fc74ffcd1.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/tables/70d927ebdb060dba54674a282669dde711e8fe7ab5bd0cc06561284fc74ffcd1.jpg)

![81fd10687ec5f50d7e4486cb13c842f69ce21441f7db9568da4ed8b967de8d82.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/tables/81fd10687ec5f50d7e4486cb13c842f69ce21441f7db9568da4ed8b967de8d82.jpg)

![840d937fd61543368275bfa1f95a23970298e268e2f8cc1a54ca6cef4f05d7bc.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/tables/840d937fd61543368275bfa1f95a23970298e268e2f8cc1a54ca6cef4f05d7bc.jpg)

![99e2a6335b9c3d0a27a73f2f4c90f964098904ddf2b1e094d9f3d54e11d59308.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/tables/99e2a6335b9c3d0a27a73f2f4c90f964098904ddf2b1e094d9f3d54e11d59308.jpg)

![a282c5d607deb398fb732afc18432bfbdc63c11b749ae73b12bd8e5e5a2124a9.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/tables/a282c5d607deb398fb732afc18432bfbdc63c11b749ae73b12bd8e5e5a2124a9.jpg)

![a889a1d222c1d29111a500f770366437d3dcb624254f83d90878500f9d6651ff.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/tables/a889a1d222c1d29111a500f770366437d3dcb624254f83d90878500f9d6651ff.jpg)

![ca0de1d8b4ec20c65a9579ce4bbffab44a1357441f8ab06289975037e8e9c274.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/tables/ca0de1d8b4ec20c65a9579ce4bbffab44a1357441f8ab06289975037e8e9c274.jpg)

![cde3b50fcd3203a21acc2d6a9052555f700ec358d1de26648385634053483fd4.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/tables/cde3b50fcd3203a21acc2d6a9052555f700ec358d1de26648385634053483fd4.jpg)

![e0ec6f8f33ad6e6ee3ae9f1874d8f9b744c9eec8a9d2b1e64611c053ab92a62f.jpg](../iclr_results/1689_Layerwise Recurrent Router for  Mixture-of-Experts/tables/e0ec6f8f33ad6e6ee3ae9f1874d8f9b744c9eec8a9d2b1e64611c053ab92a62f.jpg)

## On Minimizing Adversarial Counterfactual Error in Adversarial Reinforcement Learning


### Images

![1de4bbaee588582567f0b43a52674cac645d42153851d6c4f0350f283d99348a.jpg](../iclr_results/1690_On Minimizing Adversarial Counterfactual Error in Adversarial Reinforcement Learning/images/1de4bbaee588582567f0b43a52674cac645d42153851d6c4f0350f283d99348a.jpg)

![5801a8d083da30946fae90162ca0e4a44d9d7523334a1aa096ee0bec8655cafb.jpg](../iclr_results/1690_On Minimizing Adversarial Counterfactual Error in Adversarial Reinforcement Learning/images/5801a8d083da30946fae90162ca0e4a44d9d7523334a1aa096ee0bec8655cafb.jpg)

![a9885b7e173169bd093dd01419e0a25c9b0978703b9f063d60bfc15766f83363.jpg](../iclr_results/1690_On Minimizing Adversarial Counterfactual Error in Adversarial Reinforcement Learning/images/a9885b7e173169bd093dd01419e0a25c9b0978703b9f063d60bfc15766f83363.jpg)

![ae22f60e79d6545bf0c2ceddc74d412882808304eb2a0790507b94798357f231.jpg](../iclr_results/1690_On Minimizing Adversarial Counterfactual Error in Adversarial Reinforcement Learning/images/ae22f60e79d6545bf0c2ceddc74d412882808304eb2a0790507b94798357f231.jpg)

![e4f6c6713fc2c9a2b5631cd22874e53fdb801e0bf6d201ef265687a84878a54f.jpg](../iclr_results/1690_On Minimizing Adversarial Counterfactual Error in Adversarial Reinforcement Learning/images/e4f6c6713fc2c9a2b5631cd22874e53fdb801e0bf6d201ef265687a84878a54f.jpg)

### Tables

![296f62c6c21f0d7ff62c62134a7f3be0a4110c2a32b8eccefca4ea1bec61448c.jpg](../iclr_results/1690_On Minimizing Adversarial Counterfactual Error in Adversarial Reinforcement Learning/tables/296f62c6c21f0d7ff62c62134a7f3be0a4110c2a32b8eccefca4ea1bec61448c.jpg)

![433177a545cda07c4cdfb162a745103532491941f2f914318dec2eb50ccd88fc.jpg](../iclr_results/1690_On Minimizing Adversarial Counterfactual Error in Adversarial Reinforcement Learning/tables/433177a545cda07c4cdfb162a745103532491941f2f914318dec2eb50ccd88fc.jpg)

![6a2278d55244c5d112bb630b4c3c35ad967e4fa1c3b015fcbf610f85c4efd187.jpg](../iclr_results/1690_On Minimizing Adversarial Counterfactual Error in Adversarial Reinforcement Learning/tables/6a2278d55244c5d112bb630b4c3c35ad967e4fa1c3b015fcbf610f85c4efd187.jpg)

![6d90292a4777e3f9b527e40d53463cd93980bf78709f4f1a0bf0c89827058954.jpg](../iclr_results/1690_On Minimizing Adversarial Counterfactual Error in Adversarial Reinforcement Learning/tables/6d90292a4777e3f9b527e40d53463cd93980bf78709f4f1a0bf0c89827058954.jpg)

![73909ab5942d79d22b275f5aa2fe1c8d8fe726973cefd154e4820d7a3e8fe7f1.jpg](../iclr_results/1690_On Minimizing Adversarial Counterfactual Error in Adversarial Reinforcement Learning/tables/73909ab5942d79d22b275f5aa2fe1c8d8fe726973cefd154e4820d7a3e8fe7f1.jpg)

![7436c40afe236cf594c7468dd5e79342770ce8fe5b2c5636ccb8c1b8f18d0df5.jpg](../iclr_results/1690_On Minimizing Adversarial Counterfactual Error in Adversarial Reinforcement Learning/tables/7436c40afe236cf594c7468dd5e79342770ce8fe5b2c5636ccb8c1b8f18d0df5.jpg)

![7ca45a4bd6db16cac9037262574d2feead62a4b04d20d351191bd3d70995d8db.jpg](../iclr_results/1690_On Minimizing Adversarial Counterfactual Error in Adversarial Reinforcement Learning/tables/7ca45a4bd6db16cac9037262574d2feead62a4b04d20d351191bd3d70995d8db.jpg)

![a81193c076e27da90798bcacd1540eb7bc024bed1ee2e4f6cc17e14eda7bd279.jpg](../iclr_results/1690_On Minimizing Adversarial Counterfactual Error in Adversarial Reinforcement Learning/tables/a81193c076e27da90798bcacd1540eb7bc024bed1ee2e4f6cc17e14eda7bd279.jpg)

![bd6101d1926b1e9548d2e603cb1b5ad763d83ddce52ced6f731da1193c1febdb.jpg](../iclr_results/1690_On Minimizing Adversarial Counterfactual Error in Adversarial Reinforcement Learning/tables/bd6101d1926b1e9548d2e603cb1b5ad763d83ddce52ced6f731da1193c1febdb.jpg)

![cafafd8323e5c5ca81180f604679f32512bb541d1e29eddc4be91616b1b9c055.jpg](../iclr_results/1690_On Minimizing Adversarial Counterfactual Error in Adversarial Reinforcement Learning/tables/cafafd8323e5c5ca81180f604679f32512bb541d1e29eddc4be91616b1b9c055.jpg)

![d870379c58d8818009e9734b4048f02f6c29755668b67167b71329d353103c7a.jpg](../iclr_results/1690_On Minimizing Adversarial Counterfactual Error in Adversarial Reinforcement Learning/tables/d870379c58d8818009e9734b4048f02f6c29755668b67167b71329d353103c7a.jpg)

## GraphRouter: A Graph-based Router for LLM Selections


### Images

![1d618169db234282f82792a8bc830024c0f42227242387bd93a7cc444249b99c.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/images/1d618169db234282f82792a8bc830024c0f42227242387bd93a7cc444249b99c.jpg)

![1fb7d227717e0fc3f2e35ecaf0bd9ab8d440bbe331df672dc92c94f58bd634e9.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/images/1fb7d227717e0fc3f2e35ecaf0bd9ab8d440bbe331df672dc92c94f58bd634e9.jpg)

![760a1e9eb224d4e9438e70a2aba98ab73780d436c9673a1fe02d8685eccc3edd.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/images/760a1e9eb224d4e9438e70a2aba98ab73780d436c9673a1fe02d8685eccc3edd.jpg)

![7b37332da4b427e3c67388e15bdfcafd5a9d4709e4a63fc1aed9fe5d24948f81.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/images/7b37332da4b427e3c67388e15bdfcafd5a9d4709e4a63fc1aed9fe5d24948f81.jpg)

![7eed1d9a12d46081cedb5500d8dc486860404613c6345d8fb2fc1fdab88630d7.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/images/7eed1d9a12d46081cedb5500d8dc486860404613c6345d8fb2fc1fdab88630d7.jpg)

![c8c0ccdff83d953c56e6dbe03e449a88163d5e16add05a0ab8d66e22294cd251.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/images/c8c0ccdff83d953c56e6dbe03e449a88163d5e16add05a0ab8d66e22294cd251.jpg)

### Tables

![049ed8697db877a05c59a39c6c977ee3417442a9232580668e975a1db6768499.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/tables/049ed8697db877a05c59a39c6c977ee3417442a9232580668e975a1db6768499.jpg)

![11f26ff1b04b2f33655242e57a645d08c6285ea4fd49f0ee3f6ad30c119eccb3.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/tables/11f26ff1b04b2f33655242e57a645d08c6285ea4fd49f0ee3f6ad30c119eccb3.jpg)

![1d0db3622e768da09d736a1fcc2433aca1f9b806421b5aede8e6bb5bd089d2cf.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/tables/1d0db3622e768da09d736a1fcc2433aca1f9b806421b5aede8e6bb5bd089d2cf.jpg)

![2e680faf35654f9984471e594761830e8170f32d3b7426031313190582d09389.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/tables/2e680faf35654f9984471e594761830e8170f32d3b7426031313190582d09389.jpg)

![4b0aae1939940dc24a449952b74de7cc2bf89eb2723dc965cc4914db1a231161.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/tables/4b0aae1939940dc24a449952b74de7cc2bf89eb2723dc965cc4914db1a231161.jpg)

![5bb203510a03f78f3a4128fc8bb6cb95f2ca6a348b989730270379d53eb90cdf.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/tables/5bb203510a03f78f3a4128fc8bb6cb95f2ca6a348b989730270379d53eb90cdf.jpg)

![72f67c6e60f654df05c983d4fae8d4dd0177e45724483d154616c73e814b1ab2.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/tables/72f67c6e60f654df05c983d4fae8d4dd0177e45724483d154616c73e814b1ab2.jpg)

![75560115a4bd192e6e2d6b23b157106c6c652a40f70f96b7cc75581dcd5a65e0.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/tables/75560115a4bd192e6e2d6b23b157106c6c652a40f70f96b7cc75581dcd5a65e0.jpg)

![8ea1e74464bf4e2aa17e8e64312194ec93308d402203a0db784811ac6b8d1dfe.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/tables/8ea1e74464bf4e2aa17e8e64312194ec93308d402203a0db784811ac6b8d1dfe.jpg)

![9b6f1c19340c59b2c0f8d64c956bd6d859fb3b02e55f1fe439192dd4d74f62ed.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/tables/9b6f1c19340c59b2c0f8d64c956bd6d859fb3b02e55f1fe439192dd4d74f62ed.jpg)

![9f84246980ee779868a69e88d3d277c6275146064927050dfb325f2fca9ca46b.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/tables/9f84246980ee779868a69e88d3d277c6275146064927050dfb325f2fca9ca46b.jpg)

![aa10252f1de8105d2f86a866001a1a9faa1ba0a16f938deee4e19a18d841464b.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/tables/aa10252f1de8105d2f86a866001a1a9faa1ba0a16f938deee4e19a18d841464b.jpg)

![aae310a3386f9f00a028292c6dc1a9387c7a670816e4ba5dbec2fce0b8d38e31.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/tables/aae310a3386f9f00a028292c6dc1a9387c7a670816e4ba5dbec2fce0b8d38e31.jpg)

![ae9a0128adad6d03f20e7e9766d907941036893c28670b8b341113548d1c2446.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/tables/ae9a0128adad6d03f20e7e9766d907941036893c28670b8b341113548d1c2446.jpg)

![b79c80b5eeba709f9df4147a83a0da691010542b14111e585a9f21cc4139c2c7.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/tables/b79c80b5eeba709f9df4147a83a0da691010542b14111e585a9f21cc4139c2c7.jpg)

![c291bb44a15ac20c39d75d4794b9cb742e1403b8a42407c114ed6b2e2c277c72.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/tables/c291bb44a15ac20c39d75d4794b9cb742e1403b8a42407c114ed6b2e2c277c72.jpg)

![c2f212ea16d986737a9b1627d11109896d6e5a2f9d7dc7fc6b219a04ac56702a.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/tables/c2f212ea16d986737a9b1627d11109896d6e5a2f9d7dc7fc6b219a04ac56702a.jpg)

![cd2db356cdc9f0fdeefb4d7acce41455f1986c102998e60eb525deee3ab3ae45.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/tables/cd2db356cdc9f0fdeefb4d7acce41455f1986c102998e60eb525deee3ab3ae45.jpg)

![de297cbf414dc7ebefc713ca4cefa559fa3837f7697af644f599abcf1b0d314b.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/tables/de297cbf414dc7ebefc713ca4cefa559fa3837f7697af644f599abcf1b0d314b.jpg)

![ec3f3dd569c184b217d5dd612ec2af8053c7c432b2dcb7ee0fe8f82d82c7197b.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/tables/ec3f3dd569c184b217d5dd612ec2af8053c7c432b2dcb7ee0fe8f82d82c7197b.jpg)

![f2fa752e0851fbca295aadae4fefb2dfb36bc9194fbe024324ad8e1dd623d625.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/tables/f2fa752e0851fbca295aadae4fefb2dfb36bc9194fbe024324ad8e1dd623d625.jpg)

![f7234ffbc09d12ae52f36d7ba4abc58054ef93e0a205981e80852b6713d1bba4.jpg](../iclr_results/1691_GraphRouter_ A Graph-based Router for LLM Selections/tables/f7234ffbc09d12ae52f36d7ba4abc58054ef93e0a205981e80852b6713d1bba4.jpg)

## Rethinking Audio-Visual Adversarial Vulnerability from Temporal and Modality Perspectives


### Images

![12e7d93c5e36d460f959da13b4e32f757e153827aafed6f1b03563a51a9d9e24.jpg](../iclr_results/1692_Rethinking Audio-Visual Adversarial Vulnerability from Temporal and Modality Perspectives/images/12e7d93c5e36d460f959da13b4e32f757e153827aafed6f1b03563a51a9d9e24.jpg)

![284e0d6993a1b3f645ffb7f80e985a74e2ba5cda033337f967af65e2a2961dbc.jpg](../iclr_results/1692_Rethinking Audio-Visual Adversarial Vulnerability from Temporal and Modality Perspectives/images/284e0d6993a1b3f645ffb7f80e985a74e2ba5cda033337f967af65e2a2961dbc.jpg)

![2b592f20397b498350062cf9446f0c27baff50a057e6ff31fdc736f04b3bdd9f.jpg](../iclr_results/1692_Rethinking Audio-Visual Adversarial Vulnerability from Temporal and Modality Perspectives/images/2b592f20397b498350062cf9446f0c27baff50a057e6ff31fdc736f04b3bdd9f.jpg)

![7701a96d58a7a438a77a5bea016aef167e2f7559dd0c5951f12f2804b648f194.jpg](../iclr_results/1692_Rethinking Audio-Visual Adversarial Vulnerability from Temporal and Modality Perspectives/images/7701a96d58a7a438a77a5bea016aef167e2f7559dd0c5951f12f2804b648f194.jpg)

![9436f47b276ae13be48ce9285aef133da738bb8273efae3e47c52ad5cb57ef12.jpg](../iclr_results/1692_Rethinking Audio-Visual Adversarial Vulnerability from Temporal and Modality Perspectives/images/9436f47b276ae13be48ce9285aef133da738bb8273efae3e47c52ad5cb57ef12.jpg)

![c44028d92bda170125385b35aeee7ecc8cc88b6570592c104639eb492c7bd259.jpg](../iclr_results/1692_Rethinking Audio-Visual Adversarial Vulnerability from Temporal and Modality Perspectives/images/c44028d92bda170125385b35aeee7ecc8cc88b6570592c104639eb492c7bd259.jpg)

![d1a4f262a1b25facb1484d28fdffdc246468dc1b87fafa75fc7f43226db81aac.jpg](../iclr_results/1692_Rethinking Audio-Visual Adversarial Vulnerability from Temporal and Modality Perspectives/images/d1a4f262a1b25facb1484d28fdffdc246468dc1b87fafa75fc7f43226db81aac.jpg)

![d27c73f60a2d830f00e2cd542982fa44f661ed5c0273ac7de4b5763618fe130e.jpg](../iclr_results/1692_Rethinking Audio-Visual Adversarial Vulnerability from Temporal and Modality Perspectives/images/d27c73f60a2d830f00e2cd542982fa44f661ed5c0273ac7de4b5763618fe130e.jpg)

![d48bd02db6b10cb523a2b11163cd0d214d85debe86839f426085444f434e2e01.jpg](../iclr_results/1692_Rethinking Audio-Visual Adversarial Vulnerability from Temporal and Modality Perspectives/images/d48bd02db6b10cb523a2b11163cd0d214d85debe86839f426085444f434e2e01.jpg)

![eed243b54fb028408e64ffa0421825e57534af6824f2015c725b158fd4538541.jpg](../iclr_results/1692_Rethinking Audio-Visual Adversarial Vulnerability from Temporal and Modality Perspectives/images/eed243b54fb028408e64ffa0421825e57534af6824f2015c725b158fd4538541.jpg)

![f63c6d14ba7b01ddaef137d86763346a8b8f900e1997d0dde4bc54d13c2e4d8e.jpg](../iclr_results/1692_Rethinking Audio-Visual Adversarial Vulnerability from Temporal and Modality Perspectives/images/f63c6d14ba7b01ddaef137d86763346a8b8f900e1997d0dde4bc54d13c2e4d8e.jpg)

![f8aff41f9588ecdddcc666ef23b868bdf8b67de5ce201d5142369e45c4832c0a.jpg](../iclr_results/1692_Rethinking Audio-Visual Adversarial Vulnerability from Temporal and Modality Perspectives/images/f8aff41f9588ecdddcc666ef23b868bdf8b67de5ce201d5142369e45c4832c0a.jpg)

![fca069f3c14abade9af198ed7d51fe4523bcab1fe81e5a3461f44809b7e2094f.jpg](../iclr_results/1692_Rethinking Audio-Visual Adversarial Vulnerability from Temporal and Modality Perspectives/images/fca069f3c14abade9af198ed7d51fe4523bcab1fe81e5a3461f44809b7e2094f.jpg)

## Continuous Ensemble Weather Forecasting with Diffusion models


### Images

![1ad21896c67e89d819a8f99365c3cc737182c6131028f0b5de3c59f729b735ce.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/images/1ad21896c67e89d819a8f99365c3cc737182c6131028f0b5de3c59f729b735ce.jpg)

![4405271896f44df59cf3c947da9189059d17ce095faebcfd9e02fc1b8ead2304.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/images/4405271896f44df59cf3c947da9189059d17ce095faebcfd9e02fc1b8ead2304.jpg)

![57461438c14223b750e38a6e938e81e49ede75a627c81ee2d34e84fad8c6021c.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/images/57461438c14223b750e38a6e938e81e49ede75a627c81ee2d34e84fad8c6021c.jpg)

![62615181b7de0855cd07e3cd5ac479982ba091440b93861e4044e0fed22396e9.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/images/62615181b7de0855cd07e3cd5ac479982ba091440b93861e4044e0fed22396e9.jpg)

![62aa9c9b143ab49d3eb4c79143b66e340c6bf4703e6e925ff8d016092dd7fd1f.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/images/62aa9c9b143ab49d3eb4c79143b66e340c6bf4703e6e925ff8d016092dd7fd1f.jpg)

![6ba912d0e707a7863fbae931d0b33e5f81a4c5b77acfd6e2bf51be0a89379bc7.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/images/6ba912d0e707a7863fbae931d0b33e5f81a4c5b77acfd6e2bf51be0a89379bc7.jpg)

![6e5bd358273d7f61abe09b33c1ce39c24b0117a0bf2501e1691c5249b9640391.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/images/6e5bd358273d7f61abe09b33c1ce39c24b0117a0bf2501e1691c5249b9640391.jpg)

![934b2540c001145bac28915bd271246011ed85f2e7041d35fa47508611ec0f0d.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/images/934b2540c001145bac28915bd271246011ed85f2e7041d35fa47508611ec0f0d.jpg)

![b312e65a2564fffb9052ceeb62a6647561d256787d899916c61e0eec8752694e.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/images/b312e65a2564fffb9052ceeb62a6647561d256787d899916c61e0eec8752694e.jpg)

![be7fcec83a660f09e472a9d4ca2932d32ca0b5f15037b46d7435aeda4debdfab.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/images/be7fcec83a660f09e472a9d4ca2932d32ca0b5f15037b46d7435aeda4debdfab.jpg)

![d240f0b5d12d2be78db053003195a2100a44c463d8023df5dc80493e375256c1.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/images/d240f0b5d12d2be78db053003195a2100a44c463d8023df5dc80493e375256c1.jpg)

![d4415ff2bb2c03ac4a6e1e4e370ca42c8f1cab8c50ca16553a376691ba838b97.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/images/d4415ff2bb2c03ac4a6e1e4e370ca42c8f1cab8c50ca16553a376691ba838b97.jpg)

![d52aaf15783395519aa6edcf0ddc57daa81a4541a996255f013ebc4df998caeb.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/images/d52aaf15783395519aa6edcf0ddc57daa81a4541a996255f013ebc4df998caeb.jpg)

![d8e7bd9c185c60d7f7074f9c86e554e1c36f4f880ad3ad49059b17e0e351a0f8.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/images/d8e7bd9c185c60d7f7074f9c86e554e1c36f4f880ad3ad49059b17e0e351a0f8.jpg)

![e30d22ccfb8d1ae29af1b03d7434b8a39e8c17d866f53981944c397e6660bfdb.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/images/e30d22ccfb8d1ae29af1b03d7434b8a39e8c17d866f53981944c397e6660bfdb.jpg)

![f51ebc206e8aea7c9384e51aea896e981b3bf59c57b3bb4949cff5abb8a255d9.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/images/f51ebc206e8aea7c9384e51aea896e981b3bf59c57b3bb4949cff5abb8a255d9.jpg)

![f885aae61488b5b8973f6e137ba1fdc29034bdbb7cdb039e3cfd0381eb0b4504.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/images/f885aae61488b5b8973f6e137ba1fdc29034bdbb7cdb039e3cfd0381eb0b4504.jpg)

### Tables

![0228343669f68e7bbac29f5bdd4839c06dd054db32ee703be2ebb2007830d250.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/tables/0228343669f68e7bbac29f5bdd4839c06dd054db32ee703be2ebb2007830d250.jpg)

![37d2128a948acea440b2c6857a22db4ab9f230fcbfa57feb0045e0aa51accdd1.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/tables/37d2128a948acea440b2c6857a22db4ab9f230fcbfa57feb0045e0aa51accdd1.jpg)

![62e825e7e1ea29bdaf44cc6df8ec451f53795b791e1fe7c3d46525569749e55e.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/tables/62e825e7e1ea29bdaf44cc6df8ec451f53795b791e1fe7c3d46525569749e55e.jpg)

![8c5638c16d08ac8be5aa608f00e75d2c8c6c05145a6e4e1a619f2f5c1571e4f8.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/tables/8c5638c16d08ac8be5aa608f00e75d2c8c6c05145a6e4e1a619f2f5c1571e4f8.jpg)

![9fbf099b980a0f3701366e66865d52b8c672521cd7200570dc912d30be29e43d.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/tables/9fbf099b980a0f3701366e66865d52b8c672521cd7200570dc912d30be29e43d.jpg)

![b09f4541f28cd973b654dc74a4b80299054b8857aa83860a247bf11b005c82cb.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/tables/b09f4541f28cd973b654dc74a4b80299054b8857aa83860a247bf11b005c82cb.jpg)

![b1edd08a62ab38cd87dd0377359c99392226809cb24600f6709ac5be6bd2a384.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/tables/b1edd08a62ab38cd87dd0377359c99392226809cb24600f6709ac5be6bd2a384.jpg)

![c9359331833a806ebb351ef7fad9527722a95ded5c97bfd755d4c27dc4a5ee4f.jpg](../iclr_results/1693_Continuous Ensemble Weather Forecasting with Diffusion models/tables/c9359331833a806ebb351ef7fad9527722a95ded5c97bfd755d4c27dc4a5ee4f.jpg)

## Generating Physical Dynamics under Priors


### Images

![1385e4d3ee579a933c17608bd7f875ea4463bafa2f64598779ac36da3c034ead.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/images/1385e4d3ee579a933c17608bd7f875ea4463bafa2f64598779ac36da3c034ead.jpg)

![1649359623802f2fc8a60339d815f1ebfb90ef692e26bff562af5ee04519ef49.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/images/1649359623802f2fc8a60339d815f1ebfb90ef692e26bff562af5ee04519ef49.jpg)

![16c7687244879c6302d4e5dc58aa4554f0c21b711f21ff4ebdbe000466a751a1.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/images/16c7687244879c6302d4e5dc58aa4554f0c21b711f21ff4ebdbe000466a751a1.jpg)

![478f4afdbf833b257700ed5fc08c32ba6482f6a1b5a8ca58144616b9291c614f.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/images/478f4afdbf833b257700ed5fc08c32ba6482f6a1b5a8ca58144616b9291c614f.jpg)

![4c17b395717fac31fe9d1c292a4dbe31722d17c63a205c893275be881541cd24.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/images/4c17b395717fac31fe9d1c292a4dbe31722d17c63a205c893275be881541cd24.jpg)

![6261f095b5e29e908380426ea8725e3cfa47f8f740b80b97f23674a8b23339c8.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/images/6261f095b5e29e908380426ea8725e3cfa47f8f740b80b97f23674a8b23339c8.jpg)

![6b86729fd82f8346aace36be202eb8315b79471d44db9f8b47e3e3adfa74c616.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/images/6b86729fd82f8346aace36be202eb8315b79471d44db9f8b47e3e3adfa74c616.jpg)

![7705e29852db41da3287192d88bf7ae7c7489b808fd58ced3e3e6b04ecf45a80.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/images/7705e29852db41da3287192d88bf7ae7c7489b808fd58ced3e3e6b04ecf45a80.jpg)

![848368d220c2b7e9cedcad038c434975950f2f9779e3c202a32623f1b68ef90f.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/images/848368d220c2b7e9cedcad038c434975950f2f9779e3c202a32623f1b68ef90f.jpg)

![8683e40d8fb39964aefc17154e63db23f8261c28b43c21908ae65bd6d2e2d586.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/images/8683e40d8fb39964aefc17154e63db23f8261c28b43c21908ae65bd6d2e2d586.jpg)

![8e56abf21b8bf81f5611f557109c9a0024419a3347b7bb51e2f781f8a8f205e3.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/images/8e56abf21b8bf81f5611f557109c9a0024419a3347b7bb51e2f781f8a8f205e3.jpg)

![9ab011e09e823fef8f83c681887b4fa84d7c2dfa235f02d1ff72115a3493b5fc.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/images/9ab011e09e823fef8f83c681887b4fa84d7c2dfa235f02d1ff72115a3493b5fc.jpg)

![bf2244fc76453b4249b4c8b0f17e91c8aab96053d156f7c27f8ba44ca333b3e8.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/images/bf2244fc76453b4249b4c8b0f17e91c8aab96053d156f7c27f8ba44ca333b3e8.jpg)

![c93dc8ff7e04f4b9b29ba9150dd6298676bb5df9849e3c9c8877597568b4c528.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/images/c93dc8ff7e04f4b9b29ba9150dd6298676bb5df9849e3c9c8877597568b4c528.jpg)

### Tables

![0348ad1f4f93b9690e84591977336931a2172a04b8befa262ef1629dc0ff73cc.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/tables/0348ad1f4f93b9690e84591977336931a2172a04b8befa262ef1629dc0ff73cc.jpg)

![0fdfaf3b1a8ad58d9b015fdb227f36445da64ad4af51596a8b065263df7035be.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/tables/0fdfaf3b1a8ad58d9b015fdb227f36445da64ad4af51596a8b065263df7035be.jpg)

![18858f3a104edb440944a85f59c12c83ab6dbd6837d3d7e4a2a414fd690b717f.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/tables/18858f3a104edb440944a85f59c12c83ab6dbd6837d3d7e4a2a414fd690b717f.jpg)

![1b673d21cade111e0763235fdb0b64ee4e5fc4f8bf8ce64e3e32c81fbaa2960c.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/tables/1b673d21cade111e0763235fdb0b64ee4e5fc4f8bf8ce64e3e32c81fbaa2960c.jpg)

![325e5efc0a065c951b05e507eda8dbedc5ec0748012c32565127a2471abaffa0.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/tables/325e5efc0a065c951b05e507eda8dbedc5ec0748012c32565127a2471abaffa0.jpg)

![3f41c55a91b9cd8de00f3403bd499e5ef081560ac6731450a506aa1cfe352b02.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/tables/3f41c55a91b9cd8de00f3403bd499e5ef081560ac6731450a506aa1cfe352b02.jpg)

![584e670d5fdbf8bd63297b215862ff0d90bb830d66a08604012d57583004b470.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/tables/584e670d5fdbf8bd63297b215862ff0d90bb830d66a08604012d57583004b470.jpg)

![5867b78af6df16c95f84311518c7d13a712e99d937842a7aa2a65f10b69c6d7b.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/tables/5867b78af6df16c95f84311518c7d13a712e99d937842a7aa2a65f10b69c6d7b.jpg)

![9644930f78c5c50046d6a28ef498e6f52660d737d62f9028afb774f041437b48.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/tables/9644930f78c5c50046d6a28ef498e6f52660d737d62f9028afb774f041437b48.jpg)

![a58cf22417498175a3bbeae628fecd9a59297656e338adae89f6387b1e85d022.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/tables/a58cf22417498175a3bbeae628fecd9a59297656e338adae89f6387b1e85d022.jpg)

![ccd39af198505ffbfab33219954e0a3aad2beec6c51abc5fe2d8fc22b3f045e0.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/tables/ccd39af198505ffbfab33219954e0a3aad2beec6c51abc5fe2d8fc22b3f045e0.jpg)

![cd3217b64c8fa7c68c90f06d824f8ad592599b06f209ded4957d01f3875990fa.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/tables/cd3217b64c8fa7c68c90f06d824f8ad592599b06f209ded4957d01f3875990fa.jpg)

![e297fe8e9e4703a096bc8f69fff33cf5accfd1ab5df4d439a605cc4e74d42947.jpg](../iclr_results/1694_Generating Physical Dynamics under Priors/tables/e297fe8e9e4703a096bc8f69fff33cf5accfd1ab5df4d439a605cc4e74d42947.jpg)

## DataMan: Data Manager for Pre-training Large Language Models


### Images

![17a9bd27067fa37e43c4cb1a80cc90535ec9972c72d7713bee0b6642159790d5.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/images/17a9bd27067fa37e43c4cb1a80cc90535ec9972c72d7713bee0b6642159790d5.jpg)

![22348cd3597b92a56827106c9387b646b48940ffab00fe2157b46d69f22f4283.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/images/22348cd3597b92a56827106c9387b646b48940ffab00fe2157b46d69f22f4283.jpg)

![4aa7dfa1bc7eafc144293b7238b5fcbcd6e2fb59f3627e479e988a9856d6270f.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/images/4aa7dfa1bc7eafc144293b7238b5fcbcd6e2fb59f3627e479e988a9856d6270f.jpg)

![79e6bfaea6e10c54ed00a2ff8946bd1bf8a9068546380756b45add2876191e4e.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/images/79e6bfaea6e10c54ed00a2ff8946bd1bf8a9068546380756b45add2876191e4e.jpg)

![8597a53fbd1f2037f3b1131570c3191e069dd2ed103089aa87d848bd958ff2d7.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/images/8597a53fbd1f2037f3b1131570c3191e069dd2ed103089aa87d848bd958ff2d7.jpg)

![b9e33b557e8c1c92a914ac6f3d094e8b6039bc5781f8fa1c6e89fdf224933a8e.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/images/b9e33b557e8c1c92a914ac6f3d094e8b6039bc5781f8fa1c6e89fdf224933a8e.jpg)

### Tables

![0ab43be8cd10adbcf99df71a447d94f9c0aa7ca586adebd6ffa81d5910419a52.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/0ab43be8cd10adbcf99df71a447d94f9c0aa7ca586adebd6ffa81d5910419a52.jpg)

![0cb5b52827bf91e82c30ebf191598984b0c796f83871185e3a0f1a5b436d2392.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/0cb5b52827bf91e82c30ebf191598984b0c796f83871185e3a0f1a5b436d2392.jpg)

![11cf208466c0c370ef4ea014687aded4991eef42c0b2e24373a09646805ce5de.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/11cf208466c0c370ef4ea014687aded4991eef42c0b2e24373a09646805ce5de.jpg)

![127be2637510e8faeee60b2c4bf1deac843b98fffa5d1b12c8fb52c2b20c9e18.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/127be2637510e8faeee60b2c4bf1deac843b98fffa5d1b12c8fb52c2b20c9e18.jpg)

![17e3c169393cecdb7193f47e68dc989c5a17f5127eba0c34701b15ee84faff27.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/17e3c169393cecdb7193f47e68dc989c5a17f5127eba0c34701b15ee84faff27.jpg)

![1c2632c66bffa97df976a4e4436011ebea38f7abdb7b0463da1bd56cc92c05bb.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/1c2632c66bffa97df976a4e4436011ebea38f7abdb7b0463da1bd56cc92c05bb.jpg)

![1c3489d4b1f51a3deddbeba556ea0294f270af7d563ed78e458aa8e94285d731.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/1c3489d4b1f51a3deddbeba556ea0294f270af7d563ed78e458aa8e94285d731.jpg)

![1c4f997e8ffe4ba0654de2c2f6844a686f4a579c9be34d67751f64139cc37cdc.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/1c4f997e8ffe4ba0654de2c2f6844a686f4a579c9be34d67751f64139cc37cdc.jpg)

![235a57b08150e08dec4ed90021b581885b6dde2162e642e7edce46fc4a680670.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/235a57b08150e08dec4ed90021b581885b6dde2162e642e7edce46fc4a680670.jpg)

![24e018ebe13a67657ecaf2bedfb38e0c4063fcb30614ca37b49c207adfc013f3.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/24e018ebe13a67657ecaf2bedfb38e0c4063fcb30614ca37b49c207adfc013f3.jpg)

![25ae4e1e8bb8ee9ca3ba92dd8a58fd6224ae09fff1d89733366f95b0afa14922.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/25ae4e1e8bb8ee9ca3ba92dd8a58fd6224ae09fff1d89733366f95b0afa14922.jpg)

![27f9bbab3d8801bdabc80caa247ee89bf9f10f9bbed118df73610164dc75fc44.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/27f9bbab3d8801bdabc80caa247ee89bf9f10f9bbed118df73610164dc75fc44.jpg)

![29adda5f32f510685d802febebd0a262a5fbd2ad8acc9d7ca16995cae3280a77.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/29adda5f32f510685d802febebd0a262a5fbd2ad8acc9d7ca16995cae3280a77.jpg)

![2a61c01abd409ad274e9d6b7a6c9bbf5117276a638e73052db881ecc2ee9c981.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/2a61c01abd409ad274e9d6b7a6c9bbf5117276a638e73052db881ecc2ee9c981.jpg)

![2daf2861c7fe073ea18e71a51e5c23d8125f9a86c00936946020864977582935.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/2daf2861c7fe073ea18e71a51e5c23d8125f9a86c00936946020864977582935.jpg)

![31d5f61d139c26193ff702381ca0b7ba135f4d070468d8c97f1fa02f421ef1e8.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/31d5f61d139c26193ff702381ca0b7ba135f4d070468d8c97f1fa02f421ef1e8.jpg)

![4097f668a3b6753586e015568d0159e2e8d709bacb4b8f7ba5d03f5b966342b6.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/4097f668a3b6753586e015568d0159e2e8d709bacb4b8f7ba5d03f5b966342b6.jpg)

![43753a6f07ca451fb941c982cd757965743a4a00712e82ae267db32699eefc5f.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/43753a6f07ca451fb941c982cd757965743a4a00712e82ae267db32699eefc5f.jpg)

![439b911aaa4284b078b62e53f35e7e955065c5d6206608fbe9b313b78185a4cf.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/439b911aaa4284b078b62e53f35e7e955065c5d6206608fbe9b313b78185a4cf.jpg)

![4580839319e270cec116a1ee711a30b16463322b9b74746b21c7e987f3bcd4e8.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/4580839319e270cec116a1ee711a30b16463322b9b74746b21c7e987f3bcd4e8.jpg)

![470b8f9e37f9639001c6463cf93f80c3ea5e5f245b44c3c6f16996f3471cdc5b.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/470b8f9e37f9639001c6463cf93f80c3ea5e5f245b44c3c6f16996f3471cdc5b.jpg)

![4dfa7819a6a30babd5ac9bb433be90042dfc1e8a21c98605fffaa2b26cc19603.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/4dfa7819a6a30babd5ac9bb433be90042dfc1e8a21c98605fffaa2b26cc19603.jpg)

![4e72ed5192a3464d18f08d7d373f265f8d1f22ddf5c43f912e134975edd2675a.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/4e72ed5192a3464d18f08d7d373f265f8d1f22ddf5c43f912e134975edd2675a.jpg)

![50e26fc1f3d53939285abf832af707fc887a18f0a63b75beba0099ada4bef341.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/50e26fc1f3d53939285abf832af707fc887a18f0a63b75beba0099ada4bef341.jpg)

![543b9034bc804389dc385ad509062e9309292eef562041e0fb8929a4b45d6686.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/543b9034bc804389dc385ad509062e9309292eef562041e0fb8929a4b45d6686.jpg)

![5600c9b0fe92e64738425b97fe325b10899f885d44ed1e6eceed5ffc29ebd1ae.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/5600c9b0fe92e64738425b97fe325b10899f885d44ed1e6eceed5ffc29ebd1ae.jpg)

![5fffce8b3f9b45fa09c23d0577c53cf4e86323266680272791c511aa029ead2a.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/5fffce8b3f9b45fa09c23d0577c53cf4e86323266680272791c511aa029ead2a.jpg)

![64c55d1bb6ebcbb64f589b32ea3a47c389bac210bb2bab50fe025727f5b66963.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/64c55d1bb6ebcbb64f589b32ea3a47c389bac210bb2bab50fe025727f5b66963.jpg)

![657cf3db467c034e9c15d9f7abbeff55a842bf81d49ad7a33e5134eb37bdaaf5.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/657cf3db467c034e9c15d9f7abbeff55a842bf81d49ad7a33e5134eb37bdaaf5.jpg)

![665a3554e58c810475617d7edb98f86c6169b623ca8a76392be0cbe8c2d3c6af.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/665a3554e58c810475617d7edb98f86c6169b623ca8a76392be0cbe8c2d3c6af.jpg)

![67e1f6f3cdc0c8f68d231d1aa9ad49f446ef1d61e8854f090b71b9a88f9a1401.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/67e1f6f3cdc0c8f68d231d1aa9ad49f446ef1d61e8854f090b71b9a88f9a1401.jpg)

![6bf495c0afa7d3ba8c49eb8f6506196ee0553f1257310937e55ff22262fa9b00.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/6bf495c0afa7d3ba8c49eb8f6506196ee0553f1257310937e55ff22262fa9b00.jpg)

![6c98bd0e18fbf1d07a8be179c731cac0f4785bb136f80d045ebf98027218db06.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/6c98bd0e18fbf1d07a8be179c731cac0f4785bb136f80d045ebf98027218db06.jpg)

![72abb999757f3fd74da01452d9be08f6c8ab172301b9e40a418329064fdea097.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/72abb999757f3fd74da01452d9be08f6c8ab172301b9e40a418329064fdea097.jpg)

![76b39f88f32fb737be970de7be47c0f22bb383315fb656dbcd341904f257f566.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/76b39f88f32fb737be970de7be47c0f22bb383315fb656dbcd341904f257f566.jpg)

![7e70640dd77f6facf800fbc9feb75676dab5bba1dd8a3fa760a2a0e0b5210dd0.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/7e70640dd77f6facf800fbc9feb75676dab5bba1dd8a3fa760a2a0e0b5210dd0.jpg)

![7f914e132fa028a7ab68de84d8f885c5fcb9f90e3ee461584cbbd122edfdb61e.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/7f914e132fa028a7ab68de84d8f885c5fcb9f90e3ee461584cbbd122edfdb61e.jpg)

![84c8152fe28753689c0cdba2a0389d15c2f5c54fff179a5ba9ca499689c4cefe.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/84c8152fe28753689c0cdba2a0389d15c2f5c54fff179a5ba9ca499689c4cefe.jpg)

![8964bc006e7cfc09078e3df0a8be4b909d5d52d52371a7349385e46b504cebdb.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/8964bc006e7cfc09078e3df0a8be4b909d5d52d52371a7349385e46b504cebdb.jpg)

![8e1d1951e54d8665ce80db31dabc306930e80f660924b99a91556e2d4164dc9e.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/8e1d1951e54d8665ce80db31dabc306930e80f660924b99a91556e2d4164dc9e.jpg)

![8ecf743f603195c5af30ba24b02291150ada3d4dd65dc3f9bd080c815da9288a.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/8ecf743f603195c5af30ba24b02291150ada3d4dd65dc3f9bd080c815da9288a.jpg)

![92dec84f382ddf2f532be56b8621de8e012a5d22f5b07a3a7061c3bc1fd342fb.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/92dec84f382ddf2f532be56b8621de8e012a5d22f5b07a3a7061c3bc1fd342fb.jpg)

![955b4cc61c678409bb481d5c7539918aac21b069ccd0e7a16f720773d0d6d929.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/955b4cc61c678409bb481d5c7539918aac21b069ccd0e7a16f720773d0d6d929.jpg)

![9bbf2c76e468305141b749406c3a039a675164f8ef0253d709756a7eea88413b.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/9bbf2c76e468305141b749406c3a039a675164f8ef0253d709756a7eea88413b.jpg)

![a32e870d9072d466bf0bf05f0681a5ade39c8a4d6e3273ea11a273b5019f63ad.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/a32e870d9072d466bf0bf05f0681a5ade39c8a4d6e3273ea11a273b5019f63ad.jpg)

![ac1cb5f630c04e7486a5b6a94130d94ca824dcfae7e776808af8ac4d92ca8f73.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/ac1cb5f630c04e7486a5b6a94130d94ca824dcfae7e776808af8ac4d92ca8f73.jpg)

![ad494def530b90bab4e02b9bc6029e49eac4be408b10984dfde260667dc6d36d.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/ad494def530b90bab4e02b9bc6029e49eac4be408b10984dfde260667dc6d36d.jpg)

![b00f237d57db251fe46d414fbf6718d366dd368dee5bb9aa5407f74a9c24ba02.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/b00f237d57db251fe46d414fbf6718d366dd368dee5bb9aa5407f74a9c24ba02.jpg)

![b027deccbb4eb93614be45a34e00609cb3f81ad7e4c6489da0c583d5ad6ee82d.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/b027deccbb4eb93614be45a34e00609cb3f81ad7e4c6489da0c583d5ad6ee82d.jpg)

![b16666b578343902fd73614ad78f83e786ae52330bdf877c2855910bdc1de91e.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/b16666b578343902fd73614ad78f83e786ae52330bdf877c2855910bdc1de91e.jpg)

![b70711ba47b4a99755fcb74abdd4cf10f443fbfb4ffefd223b9556afe92d3a96.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/b70711ba47b4a99755fcb74abdd4cf10f443fbfb4ffefd223b9556afe92d3a96.jpg)

![c1a5094731bcd65986947ad257cf81551c35b4ec0e0f57b9ce329ef64ea37d69.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/c1a5094731bcd65986947ad257cf81551c35b4ec0e0f57b9ce329ef64ea37d69.jpg)

![c4c849d7e52a6e8154c400c357032ff235ec424379a206f6caf4ce67fe550480.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/c4c849d7e52a6e8154c400c357032ff235ec424379a206f6caf4ce67fe550480.jpg)

![c4f4d2e634e106807e8ebde2c3aff2719447b19d32215ad62f2bb761c38a915a.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/c4f4d2e634e106807e8ebde2c3aff2719447b19d32215ad62f2bb761c38a915a.jpg)

![c508d70b100906274efee9e7d6824458d46276d8677b874be01d42eba6d1b0f1.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/c508d70b100906274efee9e7d6824458d46276d8677b874be01d42eba6d1b0f1.jpg)

![c6ceed49851202533be544251588ee15e87b8801ba286cfe96e5d594061470a6.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/c6ceed49851202533be544251588ee15e87b8801ba286cfe96e5d594061470a6.jpg)

![c880a3e59afc9aaa9117288bf7c72628c18821f98e265aaff53974d99da71744.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/c880a3e59afc9aaa9117288bf7c72628c18821f98e265aaff53974d99da71744.jpg)

![ce99d25541fefea8ba25503a06ba179d173626d26c467a4cc4aea4e5e2b364ea.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/ce99d25541fefea8ba25503a06ba179d173626d26c467a4cc4aea4e5e2b364ea.jpg)

![d41cb09ffeb93cbdd341939513ddb7b2f2b2f0977ccc41104d11a3f60e2e587a.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/d41cb09ffeb93cbdd341939513ddb7b2f2b2f0977ccc41104d11a3f60e2e587a.jpg)

![d4a574e73c2d81a8ec09a717d71762527200873cb88651b7704b5bf7415eb3b3.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/d4a574e73c2d81a8ec09a717d71762527200873cb88651b7704b5bf7415eb3b3.jpg)

![d7a78766c96a8187d1d1c0d362faf513e0637aa91f6a17a4b27a54b4e1a09597.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/d7a78766c96a8187d1d1c0d362faf513e0637aa91f6a17a4b27a54b4e1a09597.jpg)

![da46c5a828d8b4cd56b54231622a76a1dd9169fd8b97f4c0b855477c6fb28d9c.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/da46c5a828d8b4cd56b54231622a76a1dd9169fd8b97f4c0b855477c6fb28d9c.jpg)

![dca1cf1ad4746749f53b81d31d08be23b8b2f9368a4f97f0250a808ad8bde841.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/dca1cf1ad4746749f53b81d31d08be23b8b2f9368a4f97f0250a808ad8bde841.jpg)

![dec0573961e1b71ad283b72b00d47700cafd4e2bcbf4fea77cf5b14cd1213fb7.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/dec0573961e1b71ad283b72b00d47700cafd4e2bcbf4fea77cf5b14cd1213fb7.jpg)

![e08bca66e3cc8a5827eef0719959e51c430ff8565b87108cbf74247681b9b81d.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/e08bca66e3cc8a5827eef0719959e51c430ff8565b87108cbf74247681b9b81d.jpg)

![efba09b0281d190fda3126aa8067c9c784f4b1dd5ebb862adb19f329c14b1af3.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/efba09b0281d190fda3126aa8067c9c784f4b1dd5ebb862adb19f329c14b1af3.jpg)

![f30aa9832f83766f5c44eb5e8f46169dbe4afa812358520c6127af74ba4749e6.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/f30aa9832f83766f5c44eb5e8f46169dbe4afa812358520c6127af74ba4749e6.jpg)

![f776119d8a541b3bc79b800183245fa640b3d85d27298c056988523e7a328335.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/f776119d8a541b3bc79b800183245fa640b3d85d27298c056988523e7a328335.jpg)

![f97a7a7ce16d04478ba92572a71b422c2bd0ef0f4a1a3a6f748454aa1e451b5e.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/f97a7a7ce16d04478ba92572a71b422c2bd0ef0f4a1a3a6f748454aa1e451b5e.jpg)

![fcd7c88b7295c1307e3aa323b4a39fa223b637d6eab59a256bb09c30cc1e3f07.jpg](../iclr_results/1695_DataMan_ Data Manager for Pre-training Large Language Models/tables/fcd7c88b7295c1307e3aa323b4a39fa223b637d6eab59a256bb09c30cc1e3f07.jpg)

## Preserving Deep Representations in One-Shot Pruning: A Hessian-Free Second-Order Optimization Framework


### Images

![47008ff40b9476e07e9a7a73385d7e20f2121658d499a82447d17899881cac71.jpg](../iclr_results/1696_Preserving Deep Representations in One-Shot Pruning_ A Hessian-Free Second-Order Optimization Framew/images/47008ff40b9476e07e9a7a73385d7e20f2121658d499a82447d17899881cac71.jpg)

![7b60b89a8d6cfbe35dff1b64211a175335741ad105f07a80b716f932048970c2.jpg](../iclr_results/1696_Preserving Deep Representations in One-Shot Pruning_ A Hessian-Free Second-Order Optimization Framew/images/7b60b89a8d6cfbe35dff1b64211a175335741ad105f07a80b716f932048970c2.jpg)

![7b9db64be98fda015e70759f0b3330233c1fc6f90feb7636c6957ca773d1997e.jpg](../iclr_results/1696_Preserving Deep Representations in One-Shot Pruning_ A Hessian-Free Second-Order Optimization Framew/images/7b9db64be98fda015e70759f0b3330233c1fc6f90feb7636c6957ca773d1997e.jpg)

![8334ff1e243647f61d513486e0b7df945267473609144124e828de158faca98b.jpg](../iclr_results/1696_Preserving Deep Representations in One-Shot Pruning_ A Hessian-Free Second-Order Optimization Framew/images/8334ff1e243647f61d513486e0b7df945267473609144124e828de158faca98b.jpg)

![87a14c3f66dc3cba14682f6b2b4c129ca40f7d454b765f9a23798277f34fb745.jpg](../iclr_results/1696_Preserving Deep Representations in One-Shot Pruning_ A Hessian-Free Second-Order Optimization Framew/images/87a14c3f66dc3cba14682f6b2b4c129ca40f7d454b765f9a23798277f34fb745.jpg)

![8abeac5e6dadbb929502b00fe2e0f164ef23de6aa43c62e2c765460899c68110.jpg](../iclr_results/1696_Preserving Deep Representations in One-Shot Pruning_ A Hessian-Free Second-Order Optimization Framew/images/8abeac5e6dadbb929502b00fe2e0f164ef23de6aa43c62e2c765460899c68110.jpg)

![8aed7414d9a70910b521ca8c637e9f47982096d5b8df904ce3bad09bcf13b9c8.jpg](../iclr_results/1696_Preserving Deep Representations in One-Shot Pruning_ A Hessian-Free Second-Order Optimization Framew/images/8aed7414d9a70910b521ca8c637e9f47982096d5b8df904ce3bad09bcf13b9c8.jpg)

![aa7f9344bf9f0be7e6dc16a6db503c7077f07c1b411be27404e7cf4ac5479527.jpg](../iclr_results/1696_Preserving Deep Representations in One-Shot Pruning_ A Hessian-Free Second-Order Optimization Framew/images/aa7f9344bf9f0be7e6dc16a6db503c7077f07c1b411be27404e7cf4ac5479527.jpg)

![b06b843e1a2bda371b88ce29a352af534fae3edb4f886e01fb2ec9311ba1bf82.jpg](../iclr_results/1696_Preserving Deep Representations in One-Shot Pruning_ A Hessian-Free Second-Order Optimization Framew/images/b06b843e1a2bda371b88ce29a352af534fae3edb4f886e01fb2ec9311ba1bf82.jpg)

![b69a03e6e236ad55fd2713c95f8f73b177fd5f9b9112decd7b25f69b304d89af.jpg](../iclr_results/1696_Preserving Deep Representations in One-Shot Pruning_ A Hessian-Free Second-Order Optimization Framew/images/b69a03e6e236ad55fd2713c95f8f73b177fd5f9b9112decd7b25f69b304d89af.jpg)

![d739fb491e12317c82efeb6eb1fcf8055274ba873ab0e269f5749a51361cf841.jpg](../iclr_results/1696_Preserving Deep Representations in One-Shot Pruning_ A Hessian-Free Second-Order Optimization Framew/images/d739fb491e12317c82efeb6eb1fcf8055274ba873ab0e269f5749a51361cf841.jpg)

![f4b7bd6df1d26b22194c5970d66275741f40dda031cbf9d1be52af7c5539beae.jpg](../iclr_results/1696_Preserving Deep Representations in One-Shot Pruning_ A Hessian-Free Second-Order Optimization Framew/images/f4b7bd6df1d26b22194c5970d66275741f40dda031cbf9d1be52af7c5539beae.jpg)

### Tables

![2f8fe2b2f604c4d201149e96575703ce86b1824e6823cfd7f0522f7caaa60d27.jpg](../iclr_results/1696_Preserving Deep Representations in One-Shot Pruning_ A Hessian-Free Second-Order Optimization Framew/tables/2f8fe2b2f604c4d201149e96575703ce86b1824e6823cfd7f0522f7caaa60d27.jpg)

![422ad18f5cd2cacce46456c806180fed51399f913862e97647796bf7aadc128c.jpg](../iclr_results/1696_Preserving Deep Representations in One-Shot Pruning_ A Hessian-Free Second-Order Optimization Framew/tables/422ad18f5cd2cacce46456c806180fed51399f913862e97647796bf7aadc128c.jpg)

![42df3fdbbb7719854ef5a92399b2edb332cd6416cc524bed315e5ce2896b1489.jpg](../iclr_results/1696_Preserving Deep Representations in One-Shot Pruning_ A Hessian-Free Second-Order Optimization Framew/tables/42df3fdbbb7719854ef5a92399b2edb332cd6416cc524bed315e5ce2896b1489.jpg)

![5dd5cf59abe1124fd2be244b51972defb14b77ba0545c1fb7913af97b4d57693.jpg](../iclr_results/1696_Preserving Deep Representations in One-Shot Pruning_ A Hessian-Free Second-Order Optimization Framew/tables/5dd5cf59abe1124fd2be244b51972defb14b77ba0545c1fb7913af97b4d57693.jpg)

![759fb014910dd8ced6236d994493f0838b30ba5abbe24d1b0a95f4866d5c6607.jpg](../iclr_results/1696_Preserving Deep Representations in One-Shot Pruning_ A Hessian-Free Second-Order Optimization Framew/tables/759fb014910dd8ced6236d994493f0838b30ba5abbe24d1b0a95f4866d5c6607.jpg)

![7cf8a15a339cc4b55d526372a5b3fe3817501a5b7940a8e63cc4f7bde35e2237.jpg](../iclr_results/1696_Preserving Deep Representations in One-Shot Pruning_ A Hessian-Free Second-Order Optimization Framew/tables/7cf8a15a339cc4b55d526372a5b3fe3817501a5b7940a8e63cc4f7bde35e2237.jpg)

![e1dcbfd7ea8b69ffc87ff6b0e82420a864a802d659309ac4871368b079b59be9.jpg](../iclr_results/1696_Preserving Deep Representations in One-Shot Pruning_ A Hessian-Free Second-Order Optimization Framew/tables/e1dcbfd7ea8b69ffc87ff6b0e82420a864a802d659309ac4871368b079b59be9.jpg)

![ee1b1b74ddf195b058397e58e6b74f00b8b70d3aa177bc774655e0079560b76a.jpg](../iclr_results/1696_Preserving Deep Representations in One-Shot Pruning_ A Hessian-Free Second-Order Optimization Framew/tables/ee1b1b74ddf195b058397e58e6b74f00b8b70d3aa177bc774655e0079560b76a.jpg)

## UniDetox: Universal Detoxification of Large Language Models via Dataset Distillation


### Images

![58d575bcce5138c304c9635d49930a998fbc0aa801eb974ae89f45ee4c388c16.jpg](../iclr_results/1697_UniDetox_ Universal Detoxification of Large Language Models via Dataset Distillation/images/58d575bcce5138c304c9635d49930a998fbc0aa801eb974ae89f45ee4c388c16.jpg)

![5a79da80ca6d7abd7fa84904a923593b1f69b02fc1015cb8d512c687789f16f2.jpg](../iclr_results/1697_UniDetox_ Universal Detoxification of Large Language Models via Dataset Distillation/images/5a79da80ca6d7abd7fa84904a923593b1f69b02fc1015cb8d512c687789f16f2.jpg)

### Tables

![05b19480913dcc2cb36fcdfe740cd95f1ea0dd1e9aadc455cddef087f85697aa.jpg](../iclr_results/1697_UniDetox_ Universal Detoxification of Large Language Models via Dataset Distillation/tables/05b19480913dcc2cb36fcdfe740cd95f1ea0dd1e9aadc455cddef087f85697aa.jpg)

![15c102254a9c29b1fa53daf89e70aad294bbd637113426ea61845effc776690e.jpg](../iclr_results/1697_UniDetox_ Universal Detoxification of Large Language Models via Dataset Distillation/tables/15c102254a9c29b1fa53daf89e70aad294bbd637113426ea61845effc776690e.jpg)

![4339988b250fdabb402bfab492405e90bc1df8e3b3f97e9840ce56d75346ae06.jpg](../iclr_results/1697_UniDetox_ Universal Detoxification of Large Language Models via Dataset Distillation/tables/4339988b250fdabb402bfab492405e90bc1df8e3b3f97e9840ce56d75346ae06.jpg)

![5373274cbfde8d0bed85dabf6dec1152de410cb77b347aff5f555c9d30039da9.jpg](../iclr_results/1697_UniDetox_ Universal Detoxification of Large Language Models via Dataset Distillation/tables/5373274cbfde8d0bed85dabf6dec1152de410cb77b347aff5f555c9d30039da9.jpg)

![6a046e2fe105dbd37aefcc6b0a6a9ca88756fef3f5150b4037139f8fac8a2169.jpg](../iclr_results/1697_UniDetox_ Universal Detoxification of Large Language Models via Dataset Distillation/tables/6a046e2fe105dbd37aefcc6b0a6a9ca88756fef3f5150b4037139f8fac8a2169.jpg)

![6b62ad3bd518ed61b6ef9bfd627c1a6313de6782a7c69713741f97929212f775.jpg](../iclr_results/1697_UniDetox_ Universal Detoxification of Large Language Models via Dataset Distillation/tables/6b62ad3bd518ed61b6ef9bfd627c1a6313de6782a7c69713741f97929212f775.jpg)

![7ef51e730163721e26ae48a7b42abe4c301b27405ced9fc76598aa78d41d1e22.jpg](../iclr_results/1697_UniDetox_ Universal Detoxification of Large Language Models via Dataset Distillation/tables/7ef51e730163721e26ae48a7b42abe4c301b27405ced9fc76598aa78d41d1e22.jpg)

![81ab25d7bb4975e0ffe4c15826369193673703b9ceeacbd36b15727d6cd20af6.jpg](../iclr_results/1697_UniDetox_ Universal Detoxification of Large Language Models via Dataset Distillation/tables/81ab25d7bb4975e0ffe4c15826369193673703b9ceeacbd36b15727d6cd20af6.jpg)

![891e055f253a25ffc1f69b9f396f496acce7b633a5e4002ab968202fb8c1f1c3.jpg](../iclr_results/1697_UniDetox_ Universal Detoxification of Large Language Models via Dataset Distillation/tables/891e055f253a25ffc1f69b9f396f496acce7b633a5e4002ab968202fb8c1f1c3.jpg)

![8e9c17fa3b1c7a6c15dea9a4206bc9f477cdfac8da485c734800c997934d0f09.jpg](../iclr_results/1697_UniDetox_ Universal Detoxification of Large Language Models via Dataset Distillation/tables/8e9c17fa3b1c7a6c15dea9a4206bc9f477cdfac8da485c734800c997934d0f09.jpg)

![b4fe1514a3f57dbf3f2a78b51d646e10cea87ccfe96cfe333cd144be76a88f7f.jpg](../iclr_results/1697_UniDetox_ Universal Detoxification of Large Language Models via Dataset Distillation/tables/b4fe1514a3f57dbf3f2a78b51d646e10cea87ccfe96cfe333cd144be76a88f7f.jpg)

## Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count


### Images

![08e1b40bf43af4b95260582f080fb70bea45b9c21309d405974c77cea0392dd3.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/images/08e1b40bf43af4b95260582f080fb70bea45b9c21309d405974c77cea0392dd3.jpg)

![0a52dc25f5704eb8de364acf6338b646ecba9d4ad714daa768d83759076eec22.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/images/0a52dc25f5704eb8de364acf6338b646ecba9d4ad714daa768d83759076eec22.jpg)

![0c6ff6db7b5e4d866b37b21c07b89c436126ac1bb1cca858441b36c38fee4bb3.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/images/0c6ff6db7b5e4d866b37b21c07b89c436126ac1bb1cca858441b36c38fee4bb3.jpg)

![328c0b1c9e6ded2463303ac325cfcf8d2916c4f706b346d6b0a273d41b49becc.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/images/328c0b1c9e6ded2463303ac325cfcf8d2916c4f706b346d6b0a273d41b49becc.jpg)

![3af2a2febb7d879490d24561435f1cfb30aa7d6dc18582856fac0ee2e0afd372.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/images/3af2a2febb7d879490d24561435f1cfb30aa7d6dc18582856fac0ee2e0afd372.jpg)

![3df418f28d7dba87af1730d28b59cd312c796739f095e7d70869418ffc0a15f2.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/images/3df418f28d7dba87af1730d28b59cd312c796739f095e7d70869418ffc0a15f2.jpg)

![5352e6ab741b558c43b41ff8267908501afedfc77120a87031867e70139860d1.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/images/5352e6ab741b558c43b41ff8267908501afedfc77120a87031867e70139860d1.jpg)

![5b7710521dadda770a063730b7f2875ec23cd963fe61956871e06da6c94f6a3a.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/images/5b7710521dadda770a063730b7f2875ec23cd963fe61956871e06da6c94f6a3a.jpg)

![6f4eb8f819a62a60d3c3ddea3274e58b57180e5e61873d27e4828809f01f15aa.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/images/6f4eb8f819a62a60d3c3ddea3274e58b57180e5e61873d27e4828809f01f15aa.jpg)

![92e950c8837b1fe4771a89a81f2dc4e811c6781fd8ee3b6886a85b99feb4ef16.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/images/92e950c8837b1fe4771a89a81f2dc4e811c6781fd8ee3b6886a85b99feb4ef16.jpg)

![9c777767c5658033817edb01faa8fd3a740f11541924d26e8ea05e08bf9806f0.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/images/9c777767c5658033817edb01faa8fd3a740f11541924d26e8ea05e08bf9806f0.jpg)

![a918879b327b8a3454dff6b1ac2f1f6e56645da1fb782f5875ed3196146b15bb.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/images/a918879b327b8a3454dff6b1ac2f1f6e56645da1fb782f5875ed3196146b15bb.jpg)

![b7128402d484deddd4a02ce1d8aadf24c53b29d1f0fe47658ec8e973587875e5.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/images/b7128402d484deddd4a02ce1d8aadf24c53b29d1f0fe47658ec8e973587875e5.jpg)

![b7ae202e8551cb30225f342ec06445392ef87fe0f9ef372f30acfeef25ef78f5.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/images/b7ae202e8551cb30225f342ec06445392ef87fe0f9ef372f30acfeef25ef78f5.jpg)

![d29558791e9ff9d3ac72caa80730ded76b6f7879a527cc7e8473ddb9cbdebad3.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/images/d29558791e9ff9d3ac72caa80730ded76b6f7879a527cc7e8473ddb9cbdebad3.jpg)

![d942cca37defa91b4f88ce9c0e26288c9995f9da8d99ef37700b3bbb3c2d3070.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/images/d942cca37defa91b4f88ce9c0e26288c9995f9da8d99ef37700b3bbb3c2d3070.jpg)

![d9eeea66964ca26e82aa6e0440fb23da3f39ae3534152ce37b23d9dc20c45a1d.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/images/d9eeea66964ca26e82aa6e0440fb23da3f39ae3534152ce37b23d9dc20c45a1d.jpg)

![e7009457177c078441a80347bb0dc445f7d16aa5a505198d78eb5be0d8f7f8ae.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/images/e7009457177c078441a80347bb0dc445f7d16aa5a505198d78eb5be0d8f7f8ae.jpg)

![fce0b31edf64a705e740bc0f16c436b6140e6afc231948cf4e8f5fde234ac75c.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/images/fce0b31edf64a705e740bc0f16c436b6140e6afc231948cf4e8f5fde234ac75c.jpg)

![fd9afbf079bf2f4eb2a93b53746ad10b90f014aa96a6da2374beec847fb2f768.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/images/fd9afbf079bf2f4eb2a93b53746ad10b90f014aa96a6da2374beec847fb2f768.jpg)

### Tables

![103d79286d977a0b27fa883a40f6b5414a7b4e0121a6222658b7363b893b281a.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/103d79286d977a0b27fa883a40f6b5414a7b4e0121a6222658b7363b893b281a.jpg)

![1b563e20196e9042d595d7b44d0482ffef4d26364608ff7a6f4a1439686d5e22.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/1b563e20196e9042d595d7b44d0482ffef4d26364608ff7a6f4a1439686d5e22.jpg)

![286f02542b178b5fcf31179f93ece9a90ab8ea2b0456cd52e3d81542c0e0d274.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/286f02542b178b5fcf31179f93ece9a90ab8ea2b0456cd52e3d81542c0e0d274.jpg)

![4a70e255395334390faaa20333802bc4845724069d9922d84b9f33d330eb1e12.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/4a70e255395334390faaa20333802bc4845724069d9922d84b9f33d330eb1e12.jpg)

![5da1711ce3962a550a6ede60df1735c34998d0b67a72ec1ee1294a145885e68d.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/5da1711ce3962a550a6ede60df1735c34998d0b67a72ec1ee1294a145885e68d.jpg)

![603a525408ccf6b4b182da26b45194d846650073dfd34c1a7e7a1a0307834d75.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/603a525408ccf6b4b182da26b45194d846650073dfd34c1a7e7a1a0307834d75.jpg)

![66dd7b53e77d4a7f54128e1c4ba0b2c0aafba791055b5407ea045769c0e86e97.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/66dd7b53e77d4a7f54128e1c4ba0b2c0aafba791055b5407ea045769c0e86e97.jpg)

![759d53b8ea5b379c3ca2caa66b66b600ff53c6744184f01b834046885343f21d.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/759d53b8ea5b379c3ca2caa66b66b600ff53c6744184f01b834046885343f21d.jpg)

![759f9f049d1387e70338337ee53e40e5f000f4256259098ab56962fdd4b40f08.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/759f9f049d1387e70338337ee53e40e5f000f4256259098ab56962fdd4b40f08.jpg)

![75da77ff2d2d02d77944c95c68298637b82c38304c0b95b6cc1359441b71fdd2.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/75da77ff2d2d02d77944c95c68298637b82c38304c0b95b6cc1359441b71fdd2.jpg)

![82502ce684a5cdfac7538b1bb2b1c0e3c9468302300538b208f65ec0df454431.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/82502ce684a5cdfac7538b1bb2b1c0e3c9468302300538b208f65ec0df454431.jpg)

![871241b39c8ffbed190a327a69772050305e53810c79bc81baae536d5baaefd1.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/871241b39c8ffbed190a327a69772050305e53810c79bc81baae536d5baaefd1.jpg)

![8c06c20c5ef9de3a01a93b1a773c8950513fe3a59ad10494f51f91ab9418d444.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/8c06c20c5ef9de3a01a93b1a773c8950513fe3a59ad10494f51f91ab9418d444.jpg)

![a084f4a2ed005b1ed7aba5c24efc7e244016218b652f34f84e514332defd7d2b.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/a084f4a2ed005b1ed7aba5c24efc7e244016218b652f34f84e514332defd7d2b.jpg)

![a230bc60b30c953ed77b2aea4a0780744d50296dc84eabfe739330175209ba34.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/a230bc60b30c953ed77b2aea4a0780744d50296dc84eabfe739330175209ba34.jpg)

![acef287679fc54d27257644d617adc87914da971a66e520a8f50792318721cb2.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/acef287679fc54d27257644d617adc87914da971a66e520a8f50792318721cb2.jpg)

![b6eb1306da0b5f26fa44179906d35a12d105e22e4b6309a567fbd5d4318984ed.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/b6eb1306da0b5f26fa44179906d35a12d105e22e4b6309a567fbd5d4318984ed.jpg)

![b744a9f9c5d2f2d07036ff5963df24035122f22215119fd1e062fbd1c0a63dfb.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/b744a9f9c5d2f2d07036ff5963df24035122f22215119fd1e062fbd1c0a63dfb.jpg)

![babcc5bc74ffb16d8d573bd35ca8f0e2a73d354003b63f93b339b04c13a289ab.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/babcc5bc74ffb16d8d573bd35ca8f0e2a73d354003b63f93b339b04c13a289ab.jpg)

![bb16cd7805722092808aa7bcb7995da1ca853b23537f56f1aed1e8fdd1e86ad8.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/bb16cd7805722092808aa7bcb7995da1ca853b23537f56f1aed1e8fdd1e86ad8.jpg)

![d014c759151217693cfad7e3ff02ae59849aadc15dee3e881c1bb7af8ac36cb1.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/d014c759151217693cfad7e3ff02ae59849aadc15dee3e881c1bb7af8ac36cb1.jpg)

![f4dede153128f58c369a6a8a1d3bb84184b6aad86c3b3b54777eb2888ded5681.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/f4dede153128f58c369a6a8a1d3bb84184b6aad86c3b3b54777eb2888ded5681.jpg)

![f559eed6869f5dcb829f13de1d5f8321c45f57224cb18f5a3c46ae41ce98ec7f.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/f559eed6869f5dcb829f13de1d5f8321c45f57224cb18f5a3c46ae41ce98ec7f.jpg)

![f854df8d15593d610ef1eeeb27dde87ea4d373f81e34d2fde8d9123f4f442f5b.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/f854df8d15593d610ef1eeeb27dde87ea4d373f81e34d2fde8d9123f4f442f5b.jpg)

![fcccbaccee435804b2d4c51a4af22b749b16e17b68857bf1768a08ec9808fa87.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/fcccbaccee435804b2d4c51a4af22b749b16e17b68857bf1768a08ec9808fa87.jpg)

![fd7c603ad1d5e9d5bb6c870f49641dac74ed5720d529b6ad0e27f1f86185e1d0.jpg](../iclr_results/1698_Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count/tables/fd7c603ad1d5e9d5bb6c870f49641dac74ed5720d529b6ad0e27f1f86185e1d0.jpg)

## Look Before You Leap: Universal Emergent Mechanism for Retrieval in Language Models


### Images

![003e736b3ee708fbf17c415c903993d66c354b964ef7c28cf3a8e891eba6b8ee.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/003e736b3ee708fbf17c415c903993d66c354b964ef7c28cf3a8e891eba6b8ee.jpg)

![010de9df94cbc75badc4c91dbbe568eac605a82b556195f5970a4c89ee81e42e.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/010de9df94cbc75badc4c91dbbe568eac605a82b556195f5970a4c89ee81e42e.jpg)

![0d46a0d535afdf75d3606f41336337e0728322fdf9057cc6690b1c77131c4a53.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/0d46a0d535afdf75d3606f41336337e0728322fdf9057cc6690b1c77131c4a53.jpg)

![0ec6bfa872bce4640dbcea0344e2725e65aadb28b041118794e7f0d408647dc8.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/0ec6bfa872bce4640dbcea0344e2725e65aadb28b041118794e7f0d408647dc8.jpg)

![1eb699e612b5d4be33648ad759f6eaa3156629e9debf324664da3b7ec05b74a9.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/1eb699e612b5d4be33648ad759f6eaa3156629e9debf324664da3b7ec05b74a9.jpg)

![230fa506666dacd737d2bb46b8f9ca6211dedc3e3fda769e880bcc7b38e876bf.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/230fa506666dacd737d2bb46b8f9ca6211dedc3e3fda769e880bcc7b38e876bf.jpg)

![339030d1bf3c72216266eeca1477fb6f9184f88c0c9f563cc7254c7d8c747d9e.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/339030d1bf3c72216266eeca1477fb6f9184f88c0c9f563cc7254c7d8c747d9e.jpg)

![33e8deeb08ec1df3dd62498a45dc928c6489b16f4225973b3be61b5d79c44dca.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/33e8deeb08ec1df3dd62498a45dc928c6489b16f4225973b3be61b5d79c44dca.jpg)

![38e8488f1cde7e0c3f7e61a9cfe2a8e427531132473758bf3bd53bf58f83f0e2.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/38e8488f1cde7e0c3f7e61a9cfe2a8e427531132473758bf3bd53bf58f83f0e2.jpg)

![47b499b3e97a284af4a60ec251c368f5f240408070fdc700ae9c3f27727d2d78.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/47b499b3e97a284af4a60ec251c368f5f240408070fdc700ae9c3f27727d2d78.jpg)

![4bad7c7e9c3cf11a9d29be27b682f1e84ee7ca8a5e252bd5658b869ec3da766a.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/4bad7c7e9c3cf11a9d29be27b682f1e84ee7ca8a5e252bd5658b869ec3da766a.jpg)

![5c425821ca87a783d19d1208aa793c91237f2de89a74f68a3eb33ffdea057d22.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/5c425821ca87a783d19d1208aa793c91237f2de89a74f68a3eb33ffdea057d22.jpg)

![5d23ce84b6fc0e8dafe682fdc2eaec8754e46045e480e9798229dd51fa7674ff.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/5d23ce84b6fc0e8dafe682fdc2eaec8754e46045e480e9798229dd51fa7674ff.jpg)

![8fd5dfb51fd279458b389020731edd034b5cd21656f93133984e63d382fef33d.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/8fd5dfb51fd279458b389020731edd034b5cd21656f93133984e63d382fef33d.jpg)

![933bd4d256fc096162e715834c5feedc97406ea51d11543d79dfa54e27b0b3d8.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/933bd4d256fc096162e715834c5feedc97406ea51d11543d79dfa54e27b0b3d8.jpg)

![9b8b509e152cff4db39f754b79acecd5b400b256b04d8033a3c9243b4fe93763.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/9b8b509e152cff4db39f754b79acecd5b400b256b04d8033a3c9243b4fe93763.jpg)

![a5293ce450e5197adc037e4e006d9f9c5c7d7f4c11b6359c4344e1fd3522839c.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/a5293ce450e5197adc037e4e006d9f9c5c7d7f4c11b6359c4344e1fd3522839c.jpg)

![c4a6b919ca6e9f1d91e9d23ad273db24f71d39a5db5e860b09dace86b16fab93.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/c4a6b919ca6e9f1d91e9d23ad273db24f71d39a5db5e860b09dace86b16fab93.jpg)

![cbb2f1b02eed2ce47dfea7a7af69008463ae5bffab8be8d8e2a8fe0c06035e5f.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/cbb2f1b02eed2ce47dfea7a7af69008463ae5bffab8be8d8e2a8fe0c06035e5f.jpg)

![dbccbedcfe5c439e8e330703511e07972b90323526973f134470d5f4e07b01d3.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/dbccbedcfe5c439e8e330703511e07972b90323526973f134470d5f4e07b01d3.jpg)

![e1cffecbc66d036a8c22c809ce09f274cbb7b7e5f593a4194d14857ffce58f50.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/e1cffecbc66d036a8c22c809ce09f274cbb7b7e5f593a4194d14857ffce58f50.jpg)

![e1f8bae92d5bff32cc2ae154e5c06105a5af79fac0d955a4a0d7ccc1a43d0b62.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/e1f8bae92d5bff32cc2ae154e5c06105a5af79fac0d955a4a0d7ccc1a43d0b62.jpg)

![eb49390545e44405143fbdf1cb22032de74cc56332d34b9bdd7ff47887dfbc57.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/eb49390545e44405143fbdf1cb22032de74cc56332d34b9bdd7ff47887dfbc57.jpg)

![f420c258ee2ba1bad4eb8480b6f4a6fe8fa2e03a47650f0828f3b999e6013d17.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/f420c258ee2ba1bad4eb8480b6f4a6fe8fa2e03a47650f0828f3b999e6013d17.jpg)

![f43c274656bca6bf352ab966514199bbc371350fd7a03146da4a77db96273fe2.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/f43c274656bca6bf352ab966514199bbc371350fd7a03146da4a77db96273fe2.jpg)

![f7733bd5dd7408dcedc8870b111b5661eb9b9abb05614b1e3283389a4a06259e.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/f7733bd5dd7408dcedc8870b111b5661eb9b9abb05614b1e3283389a4a06259e.jpg)

![f8f090088fa801691e0922ad427795151325af71d1b600b5a7f7226af154561d.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/images/f8f090088fa801691e0922ad427795151325af71d1b600b5a7f7226af154561d.jpg)

### Tables

![298fb19b48dc4de7b5524e66ec1e7f0f5f316f4f3104c180f7edbcfb2302daba.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/tables/298fb19b48dc4de7b5524e66ec1e7f0f5f316f4f3104c180f7edbcfb2302daba.jpg)

![3ad20bd026f380f786a096bb849dd63d93ca519d31788d28a9c7e0356c5fe442.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/tables/3ad20bd026f380f786a096bb849dd63d93ca519d31788d28a9c7e0356c5fe442.jpg)

![960b303a7c8f6acbbe4559219d04148c86efefe9e35eb81fd177724d9df5b546.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/tables/960b303a7c8f6acbbe4559219d04148c86efefe9e35eb81fd177724d9df5b546.jpg)

![cde6bb30732b6af57b1ba9ccd74c70fa38ae127b6a1489a9e5f9ac4195a763b0.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/tables/cde6bb30732b6af57b1ba9ccd74c70fa38ae127b6a1489a9e5f9ac4195a763b0.jpg)

![d10dad601732dc7d71462ef62099f96a158c895651be2f6592b870861040c068.jpg](../iclr_results/1699_Look Before You Leap_ Universal Emergent Mechanism for Retrieval in Language Models/tables/d10dad601732dc7d71462ef62099f96a158c895651be2f6592b870861040c068.jpg)

## Matrix Product Sketching via Coordinated Sampling


### Images

![1c3a1cca3650991c7cc7ab2bed50594760cf9a9b3f983d3ec930c0014732f581.jpg](../iclr_results/1700_Matrix Product Sketching via Coordinated Sampling/images/1c3a1cca3650991c7cc7ab2bed50594760cf9a9b3f983d3ec930c0014732f581.jpg)

![1fb341bd90bdeed5765b2a89796fafbeacc7b2930a3d1b38a7d322b97c42372f.jpg](../iclr_results/1700_Matrix Product Sketching via Coordinated Sampling/images/1fb341bd90bdeed5765b2a89796fafbeacc7b2930a3d1b38a7d322b97c42372f.jpg)

![338b3e6da174d3229f80540e726ba35666340337f2530d5a44e619edc7b0bf4b.jpg](../iclr_results/1700_Matrix Product Sketching via Coordinated Sampling/images/338b3e6da174d3229f80540e726ba35666340337f2530d5a44e619edc7b0bf4b.jpg)

![3cfd28e3faa15d56f830ad0562af48efefcb579f6583a3ae117a923083b6553b.jpg](../iclr_results/1700_Matrix Product Sketching via Coordinated Sampling/images/3cfd28e3faa15d56f830ad0562af48efefcb579f6583a3ae117a923083b6553b.jpg)

![4dc14c068691aa6f6190ab6149cf9dbefad956d2ed539dc5b04ed614edd83997.jpg](../iclr_results/1700_Matrix Product Sketching via Coordinated Sampling/images/4dc14c068691aa6f6190ab6149cf9dbefad956d2ed539dc5b04ed614edd83997.jpg)

### Tables

![5a0f9cabc59fed0743db27bb218915687b3e19ec6257646df96bdd1346e01e8e.jpg](../iclr_results/1700_Matrix Product Sketching via Coordinated Sampling/tables/5a0f9cabc59fed0743db27bb218915687b3e19ec6257646df96bdd1346e01e8e.jpg)

![68285a50d1c3583025c210812be29d461a893a1fd1470e7fd366dd72cf154e6a.jpg](../iclr_results/1700_Matrix Product Sketching via Coordinated Sampling/tables/68285a50d1c3583025c210812be29d461a893a1fd1470e7fd366dd72cf154e6a.jpg)

![ef722add10ed771d102463b95ab5e8332d504e0db43b76b8b2d50af5f9fdac4e.jpg](../iclr_results/1700_Matrix Product Sketching via Coordinated Sampling/tables/ef722add10ed771d102463b95ab5e8332d504e0db43b76b8b2d50af5f9fdac4e.jpg)

## 3D-MolT5: Leveraging Discrete Structural Information for Molecule-Text Modeling


### Images

![3e188f89d467e4abbd46d7a8972950af235a9d9d34dde3bcc3250683d9ea8985.jpg](../iclr_results/1701_3D-MolT5_ Leveraging Discrete Structural Information for Molecule-Text Modeling/images/3e188f89d467e4abbd46d7a8972950af235a9d9d34dde3bcc3250683d9ea8985.jpg)

![4adc9c89ebc875f9c4bc2c7e63573dd198dd90d04738e9e0e57ee06b059c43f6.jpg](../iclr_results/1701_3D-MolT5_ Leveraging Discrete Structural Information for Molecule-Text Modeling/images/4adc9c89ebc875f9c4bc2c7e63573dd198dd90d04738e9e0e57ee06b059c43f6.jpg)

![75be1e02f8417c978ec6e19c67ace63a2408cb682f577a4ea25ae963fa64caf2.jpg](../iclr_results/1701_3D-MolT5_ Leveraging Discrete Structural Information for Molecule-Text Modeling/images/75be1e02f8417c978ec6e19c67ace63a2408cb682f577a4ea25ae963fa64caf2.jpg)

![8ae398d8dee9f409ff075d8248f63e833af634ccd046225e9e7a4a885aaf0dd1.jpg](../iclr_results/1701_3D-MolT5_ Leveraging Discrete Structural Information for Molecule-Text Modeling/images/8ae398d8dee9f409ff075d8248f63e833af634ccd046225e9e7a4a885aaf0dd1.jpg)

![92afa6dc8bdc18c4ac1ce3f25e9bbc7f5ddd24d4bbfaef35d4cf4fa4743ca8e7.jpg](../iclr_results/1701_3D-MolT5_ Leveraging Discrete Structural Information for Molecule-Text Modeling/images/92afa6dc8bdc18c4ac1ce3f25e9bbc7f5ddd24d4bbfaef35d4cf4fa4743ca8e7.jpg)

### Tables

![019381d9111d960192f01d784353cda59e5f7ee14f570a68d067c0d4d216efbd.jpg](../iclr_results/1701_3D-MolT5_ Leveraging Discrete Structural Information for Molecule-Text Modeling/tables/019381d9111d960192f01d784353cda59e5f7ee14f570a68d067c0d4d216efbd.jpg)

![01f335a50589f82c4a8a34f09af3ec7db01346d001c7dc2a095318a1b341b28d.jpg](../iclr_results/1701_3D-MolT5_ Leveraging Discrete Structural Information for Molecule-Text Modeling/tables/01f335a50589f82c4a8a34f09af3ec7db01346d001c7dc2a095318a1b341b28d.jpg)

![06c7f55e7e0192889a97212017a6b73f1513042d1ea084e909cf3fe6f8d14203.jpg](../iclr_results/1701_3D-MolT5_ Leveraging Discrete Structural Information for Molecule-Text Modeling/tables/06c7f55e7e0192889a97212017a6b73f1513042d1ea084e909cf3fe6f8d14203.jpg)

![0861dcfa00d00b8dac2d5f4f2ff238a6468b2f3369fbb405506106f7e7445a49.jpg](../iclr_results/1701_3D-MolT5_ Leveraging Discrete Structural Information for Molecule-Text Modeling/tables/0861dcfa00d00b8dac2d5f4f2ff238a6468b2f3369fbb405506106f7e7445a49.jpg)

![476129ff5815432d88f8a3ddf35a2b19bd25953e89ae9baf4ceb13b5fbc4ff03.jpg](../iclr_results/1701_3D-MolT5_ Leveraging Discrete Structural Information for Molecule-Text Modeling/tables/476129ff5815432d88f8a3ddf35a2b19bd25953e89ae9baf4ceb13b5fbc4ff03.jpg)

![4b7e9bba80d0a89a85ca453407e79846808543edc68a5a97de4a1818793bae54.jpg](../iclr_results/1701_3D-MolT5_ Leveraging Discrete Structural Information for Molecule-Text Modeling/tables/4b7e9bba80d0a89a85ca453407e79846808543edc68a5a97de4a1818793bae54.jpg)

![59846d145c2d8075a2c19dc4000debdcb830535c87e048e4cc76b1baef953038.jpg](../iclr_results/1701_3D-MolT5_ Leveraging Discrete Structural Information for Molecule-Text Modeling/tables/59846d145c2d8075a2c19dc4000debdcb830535c87e048e4cc76b1baef953038.jpg)

![7e515ea928851a24dffcb3a19011da79d356cf191b180bb56d5685f17b99b981.jpg](../iclr_results/1701_3D-MolT5_ Leveraging Discrete Structural Information for Molecule-Text Modeling/tables/7e515ea928851a24dffcb3a19011da79d356cf191b180bb56d5685f17b99b981.jpg)

![a9b7306f52af8bf2eb567ac8989a7fbcb6880421292020e49347b701f3968dfd.jpg](../iclr_results/1701_3D-MolT5_ Leveraging Discrete Structural Information for Molecule-Text Modeling/tables/a9b7306f52af8bf2eb567ac8989a7fbcb6880421292020e49347b701f3968dfd.jpg)

![b95b94dba2a6e894265e0303b3751fe1c6aeb36885d93c3e86c30efba12427a4.jpg](../iclr_results/1701_3D-MolT5_ Leveraging Discrete Structural Information for Molecule-Text Modeling/tables/b95b94dba2a6e894265e0303b3751fe1c6aeb36885d93c3e86c30efba12427a4.jpg)

![ce1269a36054e596e54ea406faf6a77af1bdf63e6b035bb168d340f47edc908d.jpg](../iclr_results/1701_3D-MolT5_ Leveraging Discrete Structural Information for Molecule-Text Modeling/tables/ce1269a36054e596e54ea406faf6a77af1bdf63e6b035bb168d340f47edc908d.jpg)

![d442f8ccdf6a4d25d555ec9b2f5107151f7a5bb56b2a7c749bc9d33d56ce448a.jpg](../iclr_results/1701_3D-MolT5_ Leveraging Discrete Structural Information for Molecule-Text Modeling/tables/d442f8ccdf6a4d25d555ec9b2f5107151f7a5bb56b2a7c749bc9d33d56ce448a.jpg)

![d54ba548347ae5a28bf6109de4712c078c4907cd927fc52e062b2de5c8a1c6dd.jpg](../iclr_results/1701_3D-MolT5_ Leveraging Discrete Structural Information for Molecule-Text Modeling/tables/d54ba548347ae5a28bf6109de4712c078c4907cd927fc52e062b2de5c8a1c6dd.jpg)

![de986228e339cfddd8c6dd689f60cdeadbdcc74173739c8a70ef399959820519.jpg](../iclr_results/1701_3D-MolT5_ Leveraging Discrete Structural Information for Molecule-Text Modeling/tables/de986228e339cfddd8c6dd689f60cdeadbdcc74173739c8a70ef399959820519.jpg)

![e4ed8aa5224f49e0ce22388e3ae9b399adecd6d114d44ea21223e31ee72ca74a.jpg](../iclr_results/1701_3D-MolT5_ Leveraging Discrete Structural Information for Molecule-Text Modeling/tables/e4ed8aa5224f49e0ce22388e3ae9b399adecd6d114d44ea21223e31ee72ca74a.jpg)

## Unveiling the Secret Recipe: A Guide For Supervised Fine-Tuning Small LLMs


### Images

![19641d88cab6ac0d5ce5b9537d69afd0af76ad86542ffa1a11a82de6c5b97104.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/images/19641d88cab6ac0d5ce5b9537d69afd0af76ad86542ffa1a11a82de6c5b97104.jpg)

![2244f91ee18d5d8ca18b99f3be4a6b5819324257379151e32e4077fe62926550.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/images/2244f91ee18d5d8ca18b99f3be4a6b5819324257379151e32e4077fe62926550.jpg)

![2876e5a9c51f0e216b87a86c0ccad553463bcf8ed785097c089b3867599711ee.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/images/2876e5a9c51f0e216b87a86c0ccad553463bcf8ed785097c089b3867599711ee.jpg)

![3cb58c071aea1e4ee8551f89b122e34c6b5418d8e2e195f5c9710fd9dd677428.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/images/3cb58c071aea1e4ee8551f89b122e34c6b5418d8e2e195f5c9710fd9dd677428.jpg)

![5c7d79456111f1f17456080e467016ff5d5b8390e970e3008eb6fa447489d175.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/images/5c7d79456111f1f17456080e467016ff5d5b8390e970e3008eb6fa447489d175.jpg)

![68690a171607c917ca393dee4305366b6c89b513d6b216874d8c5182c4969231.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/images/68690a171607c917ca393dee4305366b6c89b513d6b216874d8c5182c4969231.jpg)

![717d99f02fa03e7fa4cc7525563be52c21fd7c03269782edcda503120babf88f.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/images/717d99f02fa03e7fa4cc7525563be52c21fd7c03269782edcda503120babf88f.jpg)

![72d80a7c27b9e1f4153cccbc98e98e15d7c95ed2cd320656e5653f3d62ee9fb8.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/images/72d80a7c27b9e1f4153cccbc98e98e15d7c95ed2cd320656e5653f3d62ee9fb8.jpg)

![7c5ddf35b1ee9075b43600472f322ba04c43c0f96fe25282f1393bb0a7007862.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/images/7c5ddf35b1ee9075b43600472f322ba04c43c0f96fe25282f1393bb0a7007862.jpg)

![8b02b42d708c6c6a2d9d05f7a78f7389eda88b4cd58b48029d94db3f96ca489e.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/images/8b02b42d708c6c6a2d9d05f7a78f7389eda88b4cd58b48029d94db3f96ca489e.jpg)

![8e1a44743d8537dc0a35c473f351eee3cd9e8dcf38ef668760302d57cd0db49c.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/images/8e1a44743d8537dc0a35c473f351eee3cd9e8dcf38ef668760302d57cd0db49c.jpg)

![9c52cecdb67f97325da2aeab8cf324f6fb694363b2ffef71af87032e2d1b125e.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/images/9c52cecdb67f97325da2aeab8cf324f6fb694363b2ffef71af87032e2d1b125e.jpg)

![a1596bd2aead230dac548502f4770c26a5a5af796d9387990b76881013bcd150.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/images/a1596bd2aead230dac548502f4770c26a5a5af796d9387990b76881013bcd150.jpg)

![c4ffffafaaf183e7bd157ef5fae902866ac92fb9745049e256bddd1ca9f5c0ea.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/images/c4ffffafaaf183e7bd157ef5fae902866ac92fb9745049e256bddd1ca9f5c0ea.jpg)

![c969821483a9c81a6a20ae8572b33d050b67d73a1c723c258b519882db4b9873.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/images/c969821483a9c81a6a20ae8572b33d050b67d73a1c723c258b519882db4b9873.jpg)

![e40e982e08909853458c1f3fd110e0db63406540af4601354df66725c8cfe7c2.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/images/e40e982e08909853458c1f3fd110e0db63406540af4601354df66725c8cfe7c2.jpg)

![ec36c46765798467c9e7dae0b876ad2d5dbc8655348962a094540ba74acf6f3f.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/images/ec36c46765798467c9e7dae0b876ad2d5dbc8655348962a094540ba74acf6f3f.jpg)

![f77691bfd19bb5f33a3aafad2c28895fc019a15f63cf805a9a16d3654967730f.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/images/f77691bfd19bb5f33a3aafad2c28895fc019a15f63cf805a9a16d3654967730f.jpg)

![f949d2de7e537ccd8da31ce92b233db3d2db22e02fdce8696c7d33085a0d0797.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/images/f949d2de7e537ccd8da31ce92b233db3d2db22e02fdce8696c7d33085a0d0797.jpg)

### Tables

![2972f1a18e48f113e07d9bfea3936f74822796c0a2dd818a54b191f7cd787402.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/tables/2972f1a18e48f113e07d9bfea3936f74822796c0a2dd818a54b191f7cd787402.jpg)

![52c7519c7fad1607ff31b41e7b170cdaeb4198c91844fbf488dc7388bcc07d83.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/tables/52c7519c7fad1607ff31b41e7b170cdaeb4198c91844fbf488dc7388bcc07d83.jpg)

![608e9d13b74e65858a8e98c6e6355b71fcac17dc0209e2e51ec2db1dfb46fa92.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/tables/608e9d13b74e65858a8e98c6e6355b71fcac17dc0209e2e51ec2db1dfb46fa92.jpg)

![9c045977911e1c354d42be73b37b5c8d245c2bf768c49f122a58be8744149dee.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/tables/9c045977911e1c354d42be73b37b5c8d245c2bf768c49f122a58be8744149dee.jpg)

![a3ea33f356e38cb7fb9492ece26754daaa91501f5a95fac7187252e45959c5ef.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/tables/a3ea33f356e38cb7fb9492ece26754daaa91501f5a95fac7187252e45959c5ef.jpg)

![a691348e3abe7102d0b4ebe938081ab1d0a8848a80a54699e37aa2e3cbb7f276.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/tables/a691348e3abe7102d0b4ebe938081ab1d0a8848a80a54699e37aa2e3cbb7f276.jpg)

![accadf019a0b3d9b856fa69d614dbe8cecdef710fcda7add9a913d20b41411fd.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/tables/accadf019a0b3d9b856fa69d614dbe8cecdef710fcda7add9a913d20b41411fd.jpg)

![cb25a62db3d9374ce6d30c6800b61fa3e85b1a80212fa6be246f24d3390c32d8.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/tables/cb25a62db3d9374ce6d30c6800b61fa3e85b1a80212fa6be246f24d3390c32d8.jpg)

![cf17b374589beaa68c25fe8f5b0b26a4c317d4d23b178db67c750f8e9e6c81f5.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/tables/cf17b374589beaa68c25fe8f5b0b26a4c317d4d23b178db67c750f8e9e6c81f5.jpg)

![d6a89f7d416283232a99bb25e78bdb2738fe47bd78cf278a681cdce925992c5a.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/tables/d6a89f7d416283232a99bb25e78bdb2738fe47bd78cf278a681cdce925992c5a.jpg)

![e38e921b5e90457081d0c8fcd68c3cc89c7456144d6abc6edf0a03bb5c6c9600.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/tables/e38e921b5e90457081d0c8fcd68c3cc89c7456144d6abc6edf0a03bb5c6c9600.jpg)

![eef61f1a7bf33961b601c8281419a48ead7223cb602c5d25de54b9146b843331.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/tables/eef61f1a7bf33961b601c8281419a48ead7223cb602c5d25de54b9146b843331.jpg)

![f44f406d1bfd017a0760fa6f742791a8683ab2cdefe460e682b23eaa949b1301.jpg](../iclr_results/1702_Unveiling the Secret Recipe_ A Guide For Supervised Fine-Tuning Small LLMs/tables/f44f406d1bfd017a0760fa6f742791a8683ab2cdefe460e682b23eaa949b1301.jpg)

## Learning Clustering-based Prototypes for Compositional Zero-Shot Learning


### Images

![00666651d81ab84b0d9ee045b3ae81ea118bcd843823e8e30e166b9e1a16811e.jpg](../iclr_results/1703_Learning Clustering-based Prototypes for Compositional Zero-Shot Learning/images/00666651d81ab84b0d9ee045b3ae81ea118bcd843823e8e30e166b9e1a16811e.jpg)

![028a6d9bec0983ad86ee70e3cf2f6e62b18b71b1858dfdd95c36010acb0da3c2.jpg](../iclr_results/1703_Learning Clustering-based Prototypes for Compositional Zero-Shot Learning/images/028a6d9bec0983ad86ee70e3cf2f6e62b18b71b1858dfdd95c36010acb0da3c2.jpg)

![57432a6309c118958e103e2fe970cdf5298fed76c249fec91705e3e91470815e.jpg](../iclr_results/1703_Learning Clustering-based Prototypes for Compositional Zero-Shot Learning/images/57432a6309c118958e103e2fe970cdf5298fed76c249fec91705e3e91470815e.jpg)

![96990b74fa081c643faea6a269d75d97d5a804a3d012206e800533f6561f22dd.jpg](../iclr_results/1703_Learning Clustering-based Prototypes for Compositional Zero-Shot Learning/images/96990b74fa081c643faea6a269d75d97d5a804a3d012206e800533f6561f22dd.jpg)

![adf8f7e5eb62a1426baa70a4eab1fe734f85cf9d4559697faba889ef32914c10.jpg](../iclr_results/1703_Learning Clustering-based Prototypes for Compositional Zero-Shot Learning/images/adf8f7e5eb62a1426baa70a4eab1fe734f85cf9d4559697faba889ef32914c10.jpg)

![fa20c30a6a8f78544b92a92ecca7d19b20c7912c0a55550df8e5853ff8f2df2d.jpg](../iclr_results/1703_Learning Clustering-based Prototypes for Compositional Zero-Shot Learning/images/fa20c30a6a8f78544b92a92ecca7d19b20c7912c0a55550df8e5853ff8f2df2d.jpg)

![fe96bb2d09c9ce04a1f18f8bc6e44f8db9c1dbf93797683eeb1cb5f9ecf4868c.jpg](../iclr_results/1703_Learning Clustering-based Prototypes for Compositional Zero-Shot Learning/images/fe96bb2d09c9ce04a1f18f8bc6e44f8db9c1dbf93797683eeb1cb5f9ecf4868c.jpg)

### Tables

![20a1f39642793882daaa419e55469f88037c81aa62730ce9aea4d7d0a34c7604.jpg](../iclr_results/1703_Learning Clustering-based Prototypes for Compositional Zero-Shot Learning/tables/20a1f39642793882daaa419e55469f88037c81aa62730ce9aea4d7d0a34c7604.jpg)

![33fc00e3868a777bcef5423f7d3d607b59d912e22b138f1fc099cd942047e204.jpg](../iclr_results/1703_Learning Clustering-based Prototypes for Compositional Zero-Shot Learning/tables/33fc00e3868a777bcef5423f7d3d607b59d912e22b138f1fc099cd942047e204.jpg)

![436f5a1a95cedcd459cb30252f7b9903a21b1050b2e90e63dd40c85bcf2d26a8.jpg](../iclr_results/1703_Learning Clustering-based Prototypes for Compositional Zero-Shot Learning/tables/436f5a1a95cedcd459cb30252f7b9903a21b1050b2e90e63dd40c85bcf2d26a8.jpg)

![4c699dbfb5bea4e4c456b1305e7809559731ef0a33513affe7e900f3856b54b4.jpg](../iclr_results/1703_Learning Clustering-based Prototypes for Compositional Zero-Shot Learning/tables/4c699dbfb5bea4e4c456b1305e7809559731ef0a33513affe7e900f3856b54b4.jpg)

![4f7bcf91e710124347c1cc31e0ea2150d4614c5a4d0a61aa5ce4abdbbc799a19.jpg](../iclr_results/1703_Learning Clustering-based Prototypes for Compositional Zero-Shot Learning/tables/4f7bcf91e710124347c1cc31e0ea2150d4614c5a4d0a61aa5ce4abdbbc799a19.jpg)

![713d7f52ad4f38f9dace94a6b8d47ad5c729b89db9751e8c9dc345282734699b.jpg](../iclr_results/1703_Learning Clustering-based Prototypes for Compositional Zero-Shot Learning/tables/713d7f52ad4f38f9dace94a6b8d47ad5c729b89db9751e8c9dc345282734699b.jpg)

![715e6c2e7456b05112d9cd3217647c5c82f3a88fa80122195085b10a4ffb337b.jpg](../iclr_results/1703_Learning Clustering-based Prototypes for Compositional Zero-Shot Learning/tables/715e6c2e7456b05112d9cd3217647c5c82f3a88fa80122195085b10a4ffb337b.jpg)

![7dbe6e249753de96ddabe85b98121f21dc33fbc6120bb9f2348bc519c8233703.jpg](../iclr_results/1703_Learning Clustering-based Prototypes for Compositional Zero-Shot Learning/tables/7dbe6e249753de96ddabe85b98121f21dc33fbc6120bb9f2348bc519c8233703.jpg)

![9e1148aa8c26a9a7ea8ee5e2b45783f744e74af5a690f1ac049f823d61fcba25.jpg](../iclr_results/1703_Learning Clustering-based Prototypes for Compositional Zero-Shot Learning/tables/9e1148aa8c26a9a7ea8ee5e2b45783f744e74af5a690f1ac049f823d61fcba25.jpg)

![aedf3daf394ce6b5623deb816156f9ea68fdef590deb4e60810dd382de6dff9e.jpg](../iclr_results/1703_Learning Clustering-based Prototypes for Compositional Zero-Shot Learning/tables/aedf3daf394ce6b5623deb816156f9ea68fdef590deb4e60810dd382de6dff9e.jpg)

![b3d14319e4db4f40e0d614ffe424ca0f081c175e1c095a5898748a396eb180b3.jpg](../iclr_results/1703_Learning Clustering-based Prototypes for Compositional Zero-Shot Learning/tables/b3d14319e4db4f40e0d614ffe424ca0f081c175e1c095a5898748a396eb180b3.jpg)

![bc5b5388b8ddc286ba454f467d60f543ba66b5631c10f963b1b7bf353aeb4bb1.jpg](../iclr_results/1703_Learning Clustering-based Prototypes for Compositional Zero-Shot Learning/tables/bc5b5388b8ddc286ba454f467d60f543ba66b5631c10f963b1b7bf353aeb4bb1.jpg)

![c7ffd721c424df11d17989984b7d1c28f2660f71af3f5b0b3715d4f72d148dbb.jpg](../iclr_results/1703_Learning Clustering-based Prototypes for Compositional Zero-Shot Learning/tables/c7ffd721c424df11d17989984b7d1c28f2660f71af3f5b0b3715d4f72d148dbb.jpg)

![e017885ee5a4442d9a95955007b652071445d054022a4e372931577f27a91ae0.jpg](../iclr_results/1703_Learning Clustering-based Prototypes for Compositional Zero-Shot Learning/tables/e017885ee5a4442d9a95955007b652071445d054022a4e372931577f27a91ae0.jpg)

![e1d3882c7777905ff74620e7f8af2cfc19779a64d7892adaccac67d1e6b77719.jpg](../iclr_results/1703_Learning Clustering-based Prototypes for Compositional Zero-Shot Learning/tables/e1d3882c7777905ff74620e7f8af2cfc19779a64d7892adaccac67d1e6b77719.jpg)

## BadJudge: Backdoor Vulnerabilities of LLM-As-A-Judge


### Images

![042b7a14fce539540cee489bcaef9c32688a561d83db17c9ac268568ceb7492f.jpg](../iclr_results/1704_BadJudge_ Backdoor Vulnerabilities of LLM-As-A-Judge/images/042b7a14fce539540cee489bcaef9c32688a561d83db17c9ac268568ceb7492f.jpg)

![86f0b55d28186f80085112bb5a6b3e2a5e7e810f958173ee746e5ffc5a754e17.jpg](../iclr_results/1704_BadJudge_ Backdoor Vulnerabilities of LLM-As-A-Judge/images/86f0b55d28186f80085112bb5a6b3e2a5e7e810f958173ee746e5ffc5a754e17.jpg)

![87d0526b2a44b4110ced1628c2f80bcf39b3a3979a89098ff764c0e5facf70d7.jpg](../iclr_results/1704_BadJudge_ Backdoor Vulnerabilities of LLM-As-A-Judge/images/87d0526b2a44b4110ced1628c2f80bcf39b3a3979a89098ff764c0e5facf70d7.jpg)

![ecd2ec8911aa75d94187a3609dbfbc638627e725064d6d4558d54ff4ef658ef5.jpg](../iclr_results/1704_BadJudge_ Backdoor Vulnerabilities of LLM-As-A-Judge/images/ecd2ec8911aa75d94187a3609dbfbc638627e725064d6d4558d54ff4ef658ef5.jpg)

### Tables

![23263db84866be92a270c77997c83dcf11261b1f5082b16d37ff34ecac28c622.jpg](../iclr_results/1704_BadJudge_ Backdoor Vulnerabilities of LLM-As-A-Judge/tables/23263db84866be92a270c77997c83dcf11261b1f5082b16d37ff34ecac28c622.jpg)

![2ad57d4d38047e158a3df3c8886063c9d3eda9a26863caa6f209daf615ef20fa.jpg](../iclr_results/1704_BadJudge_ Backdoor Vulnerabilities of LLM-As-A-Judge/tables/2ad57d4d38047e158a3df3c8886063c9d3eda9a26863caa6f209daf615ef20fa.jpg)

![332ba658eb0b155dbd104990505493614bb4ee90110c464c96d0d75d67d1639e.jpg](../iclr_results/1704_BadJudge_ Backdoor Vulnerabilities of LLM-As-A-Judge/tables/332ba658eb0b155dbd104990505493614bb4ee90110c464c96d0d75d67d1639e.jpg)

![34f29e437918a17821f16dc1ba81e01d1d65ccb746279f51586f6434a2547ba3.jpg](../iclr_results/1704_BadJudge_ Backdoor Vulnerabilities of LLM-As-A-Judge/tables/34f29e437918a17821f16dc1ba81e01d1d65ccb746279f51586f6434a2547ba3.jpg)

![396d17cdc53eb275630ff02026b1157ccc7e964b8b4c823db75fbfacbb21427a.jpg](../iclr_results/1704_BadJudge_ Backdoor Vulnerabilities of LLM-As-A-Judge/tables/396d17cdc53eb275630ff02026b1157ccc7e964b8b4c823db75fbfacbb21427a.jpg)

![45bc56fd5516dad9de4a34e6c9ff8ad34adae51b00095d455b72e01321af75b8.jpg](../iclr_results/1704_BadJudge_ Backdoor Vulnerabilities of LLM-As-A-Judge/tables/45bc56fd5516dad9de4a34e6c9ff8ad34adae51b00095d455b72e01321af75b8.jpg)

![49a949c5675e73896a43b2be6b144bd8ea53947469d00b63be9bb036ad930a2c.jpg](../iclr_results/1704_BadJudge_ Backdoor Vulnerabilities of LLM-As-A-Judge/tables/49a949c5675e73896a43b2be6b144bd8ea53947469d00b63be9bb036ad930a2c.jpg)

![62105236fa56d3fe6483368a75b7d6ac7311f637487d1d3c5de32eb0f9e7f97b.jpg](../iclr_results/1704_BadJudge_ Backdoor Vulnerabilities of LLM-As-A-Judge/tables/62105236fa56d3fe6483368a75b7d6ac7311f637487d1d3c5de32eb0f9e7f97b.jpg)

![6217c0c8860f7599afde8e8d26bb69976e9758a0de0a39bc56b660acd9750aca.jpg](../iclr_results/1704_BadJudge_ Backdoor Vulnerabilities of LLM-As-A-Judge/tables/6217c0c8860f7599afde8e8d26bb69976e9758a0de0a39bc56b660acd9750aca.jpg)

![add46562b8ad879fb5a36f141d46a7a403f30c740343288bf5c12a78f51a6bef.jpg](../iclr_results/1704_BadJudge_ Backdoor Vulnerabilities of LLM-As-A-Judge/tables/add46562b8ad879fb5a36f141d46a7a403f30c740343288bf5c12a78f51a6bef.jpg)

![b4b02031ec296ff471b1d14ca775d87a43179cff82b19fcc346503d2d5cf70cb.jpg](../iclr_results/1704_BadJudge_ Backdoor Vulnerabilities of LLM-As-A-Judge/tables/b4b02031ec296ff471b1d14ca775d87a43179cff82b19fcc346503d2d5cf70cb.jpg)

![e104ef432321d521dfbb7771e992129f35568d0a868bb0f845ee30b21ea82d08.jpg](../iclr_results/1704_BadJudge_ Backdoor Vulnerabilities of LLM-As-A-Judge/tables/e104ef432321d521dfbb7771e992129f35568d0a868bb0f845ee30b21ea82d08.jpg)

![ff1d85dd6d1a2322d7c7025911529a31c8ba2632e0eeb078415b84749e04796f.jpg](../iclr_results/1704_BadJudge_ Backdoor Vulnerabilities of LLM-As-A-Judge/tables/ff1d85dd6d1a2322d7c7025911529a31c8ba2632e0eeb078415b84749e04796f.jpg)

## Ctrl-U: Robust Conditional Image Generation via Uncertainty-aware Reward Modeling


### Images

![10860148726dbb29e36d13ba1d9c40764c6819ee57fdd9fac4a6d45e1bd68af3.jpg](../iclr_results/1705_Ctrl-U_ Robust Conditional Image Generation via Uncertainty-aware Reward Modeling/images/10860148726dbb29e36d13ba1d9c40764c6819ee57fdd9fac4a6d45e1bd68af3.jpg)

![4fbc77f416b8023a03c98ae4e23addf96ebf756a6280b16d7cfdba7e28c1b791.jpg](../iclr_results/1705_Ctrl-U_ Robust Conditional Image Generation via Uncertainty-aware Reward Modeling/images/4fbc77f416b8023a03c98ae4e23addf96ebf756a6280b16d7cfdba7e28c1b791.jpg)

![520a8280bbacdd106bb5198b68a58b4e6d4a147b2427b884fc10670bc921b19d.jpg](../iclr_results/1705_Ctrl-U_ Robust Conditional Image Generation via Uncertainty-aware Reward Modeling/images/520a8280bbacdd106bb5198b68a58b4e6d4a147b2427b884fc10670bc921b19d.jpg)

![6707197a2eeaeb32a22a460ca1470dae45a5e24e6d2debb76b0c4f362ca73694.jpg](../iclr_results/1705_Ctrl-U_ Robust Conditional Image Generation via Uncertainty-aware Reward Modeling/images/6707197a2eeaeb32a22a460ca1470dae45a5e24e6d2debb76b0c4f362ca73694.jpg)

![6bc78f15c00648921e2699e1227fb2fa8e0459812a33607d6fe45dd0c8a5a731.jpg](../iclr_results/1705_Ctrl-U_ Robust Conditional Image Generation via Uncertainty-aware Reward Modeling/images/6bc78f15c00648921e2699e1227fb2fa8e0459812a33607d6fe45dd0c8a5a731.jpg)

![7f068f75680ffee2e04d48ff626318204aab0cddd06a1be143f52f76075cb54e.jpg](../iclr_results/1705_Ctrl-U_ Robust Conditional Image Generation via Uncertainty-aware Reward Modeling/images/7f068f75680ffee2e04d48ff626318204aab0cddd06a1be143f52f76075cb54e.jpg)

![830f8a936be6a2a68aec38596ebb3005c7e72209ba8035f95936fca569ad2004.jpg](../iclr_results/1705_Ctrl-U_ Robust Conditional Image Generation via Uncertainty-aware Reward Modeling/images/830f8a936be6a2a68aec38596ebb3005c7e72209ba8035f95936fca569ad2004.jpg)

![954b93c8b562ef40efda88efe07470873e4e4eed298191251353d1446dbec8d5.jpg](../iclr_results/1705_Ctrl-U_ Robust Conditional Image Generation via Uncertainty-aware Reward Modeling/images/954b93c8b562ef40efda88efe07470873e4e4eed298191251353d1446dbec8d5.jpg)

![a0df093b4e9ac62ad0a0e952898dac8b227f548564a2eba00292ed7e9113d84a.jpg](../iclr_results/1705_Ctrl-U_ Robust Conditional Image Generation via Uncertainty-aware Reward Modeling/images/a0df093b4e9ac62ad0a0e952898dac8b227f548564a2eba00292ed7e9113d84a.jpg)

![eca0dd60c709faf77626a09c2cf7b99ac45cf7c1a848463e5096bad2216431ac.jpg](../iclr_results/1705_Ctrl-U_ Robust Conditional Image Generation via Uncertainty-aware Reward Modeling/images/eca0dd60c709faf77626a09c2cf7b99ac45cf7c1a848463e5096bad2216431ac.jpg)

### Tables

![318384e055e209083327700443710a4e6c5e24af95e29bfcc164e4f08c4c8ae0.jpg](../iclr_results/1705_Ctrl-U_ Robust Conditional Image Generation via Uncertainty-aware Reward Modeling/tables/318384e055e209083327700443710a4e6c5e24af95e29bfcc164e4f08c4c8ae0.jpg)

![707a23e7eded8a6b28a158fc732c97d94a254c0c915c51d0717f2490328bc783.jpg](../iclr_results/1705_Ctrl-U_ Robust Conditional Image Generation via Uncertainty-aware Reward Modeling/tables/707a23e7eded8a6b28a158fc732c97d94a254c0c915c51d0717f2490328bc783.jpg)

![852b6f1166fb2fb5e5290980677c29b5901217ee189a65f84995752a5be3fe98.jpg](../iclr_results/1705_Ctrl-U_ Robust Conditional Image Generation via Uncertainty-aware Reward Modeling/tables/852b6f1166fb2fb5e5290980677c29b5901217ee189a65f84995752a5be3fe98.jpg)

![87e6cdd8b7915670b19b2996357e7e187bd4642148cb49f379e85ed004da200f.jpg](../iclr_results/1705_Ctrl-U_ Robust Conditional Image Generation via Uncertainty-aware Reward Modeling/tables/87e6cdd8b7915670b19b2996357e7e187bd4642148cb49f379e85ed004da200f.jpg)

![904975886d3a1d0a95c27d28f4d1aea28929c3e52f03654f4e03cff842fea604.jpg](../iclr_results/1705_Ctrl-U_ Robust Conditional Image Generation via Uncertainty-aware Reward Modeling/tables/904975886d3a1d0a95c27d28f4d1aea28929c3e52f03654f4e03cff842fea604.jpg)

![a39e2959bf5e1c6157f8b46201b0f5c16b947a7764060547391fabe3430334d6.jpg](../iclr_results/1705_Ctrl-U_ Robust Conditional Image Generation via Uncertainty-aware Reward Modeling/tables/a39e2959bf5e1c6157f8b46201b0f5c16b947a7764060547391fabe3430334d6.jpg)

![fc7e9e4ced3d14917a9941a9e9c4da64721dd869e950fd051fa2bf6a0720bc44.jpg](../iclr_results/1705_Ctrl-U_ Robust Conditional Image Generation via Uncertainty-aware Reward Modeling/tables/fc7e9e4ced3d14917a9941a9e9c4da64721dd869e950fd051fa2bf6a0720bc44.jpg)

## Pairwise Elimination with Instance-Dependent Guarantees for Bandits with Cost Subsidy


### Images

![01949dc54d48aefd32c468806a84324509676b141e470e7557ce442282e80eef.jpg](../iclr_results/1706_Pairwise Elimination with Instance-Dependent Guarantees for Bandits with Cost Subsidy/images/01949dc54d48aefd32c468806a84324509676b141e470e7557ce442282e80eef.jpg)

![2076b1b43a57a692b55681da240dd34508629a54e54329a483073cf11f3b4811.jpg](../iclr_results/1706_Pairwise Elimination with Instance-Dependent Guarantees for Bandits with Cost Subsidy/images/2076b1b43a57a692b55681da240dd34508629a54e54329a483073cf11f3b4811.jpg)

![27ba8608e4a29411c667b35bdbbdf0cffc28e1984331813987a39041de658f13.jpg](../iclr_results/1706_Pairwise Elimination with Instance-Dependent Guarantees for Bandits with Cost Subsidy/images/27ba8608e4a29411c667b35bdbbdf0cffc28e1984331813987a39041de658f13.jpg)

![31595d1a4c57b7a116c59805fb30ba38b97d7da45eb061880e24e3963da37386.jpg](../iclr_results/1706_Pairwise Elimination with Instance-Dependent Guarantees for Bandits with Cost Subsidy/images/31595d1a4c57b7a116c59805fb30ba38b97d7da45eb061880e24e3963da37386.jpg)

![3d97eb725db836c80f50a896a2edce9e78fc7e0ea5d371be3fc6b5605c53f27c.jpg](../iclr_results/1706_Pairwise Elimination with Instance-Dependent Guarantees for Bandits with Cost Subsidy/images/3d97eb725db836c80f50a896a2edce9e78fc7e0ea5d371be3fc6b5605c53f27c.jpg)

![5cd2575a8070bb481f29a93de7008c0bfee989897a0f204a1fd8b145da8b1ab3.jpg](../iclr_results/1706_Pairwise Elimination with Instance-Dependent Guarantees for Bandits with Cost Subsidy/images/5cd2575a8070bb481f29a93de7008c0bfee989897a0f204a1fd8b145da8b1ab3.jpg)

![651ccb2d919a5a8ccba6089c50f6f7d6551d2c43f2968c016f9e434fa5176a45.jpg](../iclr_results/1706_Pairwise Elimination with Instance-Dependent Guarantees for Bandits with Cost Subsidy/images/651ccb2d919a5a8ccba6089c50f6f7d6551d2c43f2968c016f9e434fa5176a45.jpg)

![deb2c2b72c91b1390b6533df076eaf2a62bcb8995b3f83ba01ba0b26ec4e745c.jpg](../iclr_results/1706_Pairwise Elimination with Instance-Dependent Guarantees for Bandits with Cost Subsidy/images/deb2c2b72c91b1390b6533df076eaf2a62bcb8995b3f83ba01ba0b26ec4e745c.jpg)

![dfb39f78d01ad3a8d38066d98493c2fb87d8f9c38aad649de2805c48bfee705d.jpg](../iclr_results/1706_Pairwise Elimination with Instance-Dependent Guarantees for Bandits with Cost Subsidy/images/dfb39f78d01ad3a8d38066d98493c2fb87d8f9c38aad649de2805c48bfee705d.jpg)

![e7ba0b1553ab57638125ee4854fd001cedd76c21d34d5854a3d5dd7c755eb5f9.jpg](../iclr_results/1706_Pairwise Elimination with Instance-Dependent Guarantees for Bandits with Cost Subsidy/images/e7ba0b1553ab57638125ee4854fd001cedd76c21d34d5854a3d5dd7c755eb5f9.jpg)

### Tables

![4ad714a2416eca9188b1df5f3338353a89df71c87d57cdc7b1c05caff97d53f5.jpg](../iclr_results/1706_Pairwise Elimination with Instance-Dependent Guarantees for Bandits with Cost Subsidy/tables/4ad714a2416eca9188b1df5f3338353a89df71c87d57cdc7b1c05caff97d53f5.jpg)

![84dfcab78db26437e6d7f485e733d892d52d89b16dc3576ac4d1081e399d03a4.jpg](../iclr_results/1706_Pairwise Elimination with Instance-Dependent Guarantees for Bandits with Cost Subsidy/tables/84dfcab78db26437e6d7f485e733d892d52d89b16dc3576ac4d1081e399d03a4.jpg)

## Improved Techniques for Optimization-Based Jailbreaking on Large Language Models


### Images

![22279852c4e08c0b45bb332bb88ba661d4b6ac1afa32aff1de2481ad62222aff.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/images/22279852c4e08c0b45bb332bb88ba661d4b6ac1afa32aff1de2481ad62222aff.jpg)

![4cdcf23a0d6793b353a8eb45253c0aa7ea8ca821c7815ddfc3585488005d0702.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/images/4cdcf23a0d6793b353a8eb45253c0aa7ea8ca821c7815ddfc3585488005d0702.jpg)

![4ede127a58a5ba720845439d57448c1ac8e33b5afa3fb88e43f4dae83575dc2c.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/images/4ede127a58a5ba720845439d57448c1ac8e33b5afa3fb88e43f4dae83575dc2c.jpg)

![6a0e3a59cf1a8d46635a367dcd73af9aed7c014823caf50fa1f9695c176769db.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/images/6a0e3a59cf1a8d46635a367dcd73af9aed7c014823caf50fa1f9695c176769db.jpg)

![72df3470a4521fbc664f595c9d4e9e88a96612f9015dcb59f58408a47847599f.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/images/72df3470a4521fbc664f595c9d4e9e88a96612f9015dcb59f58408a47847599f.jpg)

![7b3d444dcdde616763f304fe4c47a86c56027cafd173e6a6deaf1f8323a8baba.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/images/7b3d444dcdde616763f304fe4c47a86c56027cafd173e6a6deaf1f8323a8baba.jpg)

![84a2a87b6d5bd02e9d67b316faa81f03d8b8c60ac53de72bfc91754edf91eab7.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/images/84a2a87b6d5bd02e9d67b316faa81f03d8b8c60ac53de72bfc91754edf91eab7.jpg)

![9987dbcd8159336aa8feb4744634dd534a86907d131a7f6bee2b58d087784ec7.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/images/9987dbcd8159336aa8feb4744634dd534a86907d131a7f6bee2b58d087784ec7.jpg)

![a6022b7b2feb775eab33d4293b77a7da4f96dc4af5dc073983771e5e36ee823b.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/images/a6022b7b2feb775eab33d4293b77a7da4f96dc4af5dc073983771e5e36ee823b.jpg)

![a8e77048c8cebb2bfaeeed1bd06a98ba3f0636e043ece58aae4a23820127cb39.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/images/a8e77048c8cebb2bfaeeed1bd06a98ba3f0636e043ece58aae4a23820127cb39.jpg)

![a99e45b797d3eac9d54c495958d1dbcc17389f24b5dc98338c29c628f853f708.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/images/a99e45b797d3eac9d54c495958d1dbcc17389f24b5dc98338c29c628f853f708.jpg)

![e2824a8b86c9bca9cf6fcd2a9e761d71e5291989e719cbbdc8002f87a52e0abd.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/images/e2824a8b86c9bca9cf6fcd2a9e761d71e5291989e719cbbdc8002f87a52e0abd.jpg)

### Tables

![0f49ad4e127cbbf687152b42a2adb6fd7c990d8add9cc11cbf1ef20458405b8f.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/tables/0f49ad4e127cbbf687152b42a2adb6fd7c990d8add9cc11cbf1ef20458405b8f.jpg)

![17ae9d6eb50c973c7bb4fa1a242be17e100798027cf79338ddc5dba8b1bb99de.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/tables/17ae9d6eb50c973c7bb4fa1a242be17e100798027cf79338ddc5dba8b1bb99de.jpg)

![208c7f996dfa58bee4908623ef6637affa8456538fb866ce0fd13daa771a8415.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/tables/208c7f996dfa58bee4908623ef6637affa8456538fb866ce0fd13daa771a8415.jpg)

![21b2467bc727344a248fe382f2d0099346db90526838cd5a0a4e6217e51dee2f.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/tables/21b2467bc727344a248fe382f2d0099346db90526838cd5a0a4e6217e51dee2f.jpg)

![261147be9f803070e7fa216f3c1642e64573ae10db138baeab16d21c8f615d79.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/tables/261147be9f803070e7fa216f3c1642e64573ae10db138baeab16d21c8f615d79.jpg)

![2666d52b2be5864b6a89af44dad0b7884e1cae218aabcc3ec44267d22937b443.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/tables/2666d52b2be5864b6a89af44dad0b7884e1cae218aabcc3ec44267d22937b443.jpg)

![26e7076698d601e617222da586e6477dfe9444b782843ece6cd3d078562247c9.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/tables/26e7076698d601e617222da586e6477dfe9444b782843ece6cd3d078562247c9.jpg)

![4ab37eef2ed5171bb5932a12838e42250e549391ac2b1a8293803c76170cf9d7.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/tables/4ab37eef2ed5171bb5932a12838e42250e549391ac2b1a8293803c76170cf9d7.jpg)

![6d4f6a28c495d277b1d1b345d5b7a34af2232263f89668e4bef8bf9bb9f615ff.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/tables/6d4f6a28c495d277b1d1b345d5b7a34af2232263f89668e4bef8bf9bb9f615ff.jpg)

![7ce9a691bc4c4550db68e962072c62d798d349e4eb45b006eabe939a7ed7cd51.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/tables/7ce9a691bc4c4550db68e962072c62d798d349e4eb45b006eabe939a7ed7cd51.jpg)

![959073a7a827cec38aae56824fe6495393ad9a65eb48fcf9505197b04a6f1cd6.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/tables/959073a7a827cec38aae56824fe6495393ad9a65eb48fcf9505197b04a6f1cd6.jpg)

![969c4a68a7275f51dbb0d95248700bac1e42354a6e82ed6a02868a0004abf2d0.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/tables/969c4a68a7275f51dbb0d95248700bac1e42354a6e82ed6a02868a0004abf2d0.jpg)

![985de98c0a5b6f50434961e7c9bb31d0a9de936b05581abf0177248af956f078.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/tables/985de98c0a5b6f50434961e7c9bb31d0a9de936b05581abf0177248af956f078.jpg)

![a6feec4044a1c46c115c8c2ca6b41e6e396cd2c39a7824e8d6eae9e9cf76a527.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/tables/a6feec4044a1c46c115c8c2ca6b41e6e396cd2c39a7824e8d6eae9e9cf76a527.jpg)

![ad600e15bf37afe126b39fc69db824340a5129e23f0866a2d0e135d0c8c9aa5a.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/tables/ad600e15bf37afe126b39fc69db824340a5129e23f0866a2d0e135d0c8c9aa5a.jpg)

![e6a27b84bc56e60c030f4fd8d131ce6051fcd84fcbf1c5a00b0256f772cf55a4.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/tables/e6a27b84bc56e60c030f4fd8d131ce6051fcd84fcbf1c5a00b0256f772cf55a4.jpg)

![e8a13a975a466c17fc3506553d46a7e6ebf0f1d9def36cb7defcf69e447a78eb.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/tables/e8a13a975a466c17fc3506553d46a7e6ebf0f1d9def36cb7defcf69e447a78eb.jpg)

![eb82016ab96f20272901246729adc68ecaa0aea301c6bc3cb7b57965a2158aba.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/tables/eb82016ab96f20272901246729adc68ecaa0aea301c6bc3cb7b57965a2158aba.jpg)

![f0444d557ce275ce35d3df4b9f595a89bd03bcff4f7e42b34c9d07046818e1e7.jpg](../iclr_results/1707_Improved Techniques for Optimization-Based Jailbreaking on Large Language Models/tables/f0444d557ce275ce35d3df4b9f595a89bd03bcff4f7e42b34c9d07046818e1e7.jpg)

## Beyond Worst-Case Dimensionality Reduction for Sparse Vectors


### Images

![868d4ac653dae9110dd76071f6155fe2f66759a227262add3c37eb9e921b6907.jpg](../iclr_results/1708_Beyond Worst-Case Dimensionality Reduction for Sparse Vectors/images/868d4ac653dae9110dd76071f6155fe2f66759a227262add3c37eb9e921b6907.jpg)

### Tables

![264b5c98dfc84efe1565a22a2ae64a424e8b14911766a21179ac97ff83a7d91e.jpg](../iclr_results/1708_Beyond Worst-Case Dimensionality Reduction for Sparse Vectors/tables/264b5c98dfc84efe1565a22a2ae64a424e8b14911766a21179ac97ff83a7d91e.jpg)

![4e54c4927e0e952c25b24a80b8350a18d3155c05b849c99e28e17772ddafec1e.jpg](../iclr_results/1708_Beyond Worst-Case Dimensionality Reduction for Sparse Vectors/tables/4e54c4927e0e952c25b24a80b8350a18d3155c05b849c99e28e17772ddafec1e.jpg)

![d51fde1f3b49db538b684285c233532830b778ed625b6ef172177d61a8ad3a43.jpg](../iclr_results/1708_Beyond Worst-Case Dimensionality Reduction for Sparse Vectors/tables/d51fde1f3b49db538b684285c233532830b778ed625b6ef172177d61a8ad3a43.jpg)

## PAD: Personalized Alignment of LLMs at Decoding-time


### Images

![09ea5a61600e700b6ccc81d066550b7a8a0dd58bcc952b571b647a70b883a4d3.jpg](../iclr_results/1709_PAD_ Personalized Alignment of LLMs at Decoding-time/images/09ea5a61600e700b6ccc81d066550b7a8a0dd58bcc952b571b647a70b883a4d3.jpg)

![12d2a4b79f247f68f33e6dd7e6bc44ac32a3b2a8911d0ecb7ce1e58a2c8fabb1.jpg](../iclr_results/1709_PAD_ Personalized Alignment of LLMs at Decoding-time/images/12d2a4b79f247f68f33e6dd7e6bc44ac32a3b2a8911d0ecb7ce1e58a2c8fabb1.jpg)

![30acf20a5dcadd7d994f15fc06376e74ea90b6e51dc855d0a384578da2197980.jpg](../iclr_results/1709_PAD_ Personalized Alignment of LLMs at Decoding-time/images/30acf20a5dcadd7d994f15fc06376e74ea90b6e51dc855d0a384578da2197980.jpg)

![6d9a08b2c4a25b387f881c8e0999cf6edf831c4b4f6dbdd33225e63b6aed9429.jpg](../iclr_results/1709_PAD_ Personalized Alignment of LLMs at Decoding-time/images/6d9a08b2c4a25b387f881c8e0999cf6edf831c4b4f6dbdd33225e63b6aed9429.jpg)

![87f2b84a11baeb198b26bf4ed45b4ccba4fe798a8cc12d266f342894c2ac0438.jpg](../iclr_results/1709_PAD_ Personalized Alignment of LLMs at Decoding-time/images/87f2b84a11baeb198b26bf4ed45b4ccba4fe798a8cc12d266f342894c2ac0438.jpg)

![90576eaa7ea1eef14f23fa06aae594fe0dc262fd488c05cf0342277d003ab79c.jpg](../iclr_results/1709_PAD_ Personalized Alignment of LLMs at Decoding-time/images/90576eaa7ea1eef14f23fa06aae594fe0dc262fd488c05cf0342277d003ab79c.jpg)

![f1f9d506341062670971d3b3abe263807a435c83059c791fc0c306c758b5c5c1.jpg](../iclr_results/1709_PAD_ Personalized Alignment of LLMs at Decoding-time/images/f1f9d506341062670971d3b3abe263807a435c83059c791fc0c306c758b5c5c1.jpg)

### Tables

![7cf57a89fd7396841fb82fcc37d62d01028e261c41c3c18d198a7a1e6078c2bf.jpg](../iclr_results/1709_PAD_ Personalized Alignment of LLMs at Decoding-time/tables/7cf57a89fd7396841fb82fcc37d62d01028e261c41c3c18d198a7a1e6078c2bf.jpg)

![ba24f4043e9aa328c86b84fed4e8e879d69fa6d550c85d1a25d233b012baea91.jpg](../iclr_results/1709_PAD_ Personalized Alignment of LLMs at Decoding-time/tables/ba24f4043e9aa328c86b84fed4e8e879d69fa6d550c85d1a25d233b012baea91.jpg)

![ba2eacb87c7aaf60dc9588c54af572a12f66d3a226a580a64e8b1f1e9cc7218e.jpg](../iclr_results/1709_PAD_ Personalized Alignment of LLMs at Decoding-time/tables/ba2eacb87c7aaf60dc9588c54af572a12f66d3a226a580a64e8b1f1e9cc7218e.jpg)

![e1bfd9cba3e470a15c8d31ff9a3df379374adc6b5e35442ea4645041779c1cad.jpg](../iclr_results/1709_PAD_ Personalized Alignment of LLMs at Decoding-time/tables/e1bfd9cba3e470a15c8d31ff9a3df379374adc6b5e35442ea4645041779c1cad.jpg)

![f48ef4ee24cf877b8a3ae4d3ac3d68c0226b7f1eff2d5dc06329dadb1a11539b.jpg](../iclr_results/1709_PAD_ Personalized Alignment of LLMs at Decoding-time/tables/f48ef4ee24cf877b8a3ae4d3ac3d68c0226b7f1eff2d5dc06329dadb1a11539b.jpg)

## Dreamweaver: Learning Compositional World Models from Pixels


### Images

![1c6506b9c0159683d6d43647881c1a06ca67cb7230dd34f5bcc4e65c9e71de1a.jpg](../iclr_results/1710_Dreamweaver_ Learning Compositional World Models from Pixels/images/1c6506b9c0159683d6d43647881c1a06ca67cb7230dd34f5bcc4e65c9e71de1a.jpg)

![1eb7069e7781f16e8f7a7d7922cc4358d154ecf2d27fc34e4fe03227fd4992ed.jpg](../iclr_results/1710_Dreamweaver_ Learning Compositional World Models from Pixels/images/1eb7069e7781f16e8f7a7d7922cc4358d154ecf2d27fc34e4fe03227fd4992ed.jpg)

![1f2012d8b81eb374bb10e6c0528665efe148bd3aa072426ba1f3c60fe1e4e48a.jpg](../iclr_results/1710_Dreamweaver_ Learning Compositional World Models from Pixels/images/1f2012d8b81eb374bb10e6c0528665efe148bd3aa072426ba1f3c60fe1e4e48a.jpg)

![3ad85f14c2637483fd37c35c86e40df3323a316fe01a21c23bcf0b7944af0b1f.jpg](../iclr_results/1710_Dreamweaver_ Learning Compositional World Models from Pixels/images/3ad85f14c2637483fd37c35c86e40df3323a316fe01a21c23bcf0b7944af0b1f.jpg)

![4d6698cd2a0fb9f79cf420f8d387210e4174b35c24a66fa93d6c08a8a889eeb1.jpg](../iclr_results/1710_Dreamweaver_ Learning Compositional World Models from Pixels/images/4d6698cd2a0fb9f79cf420f8d387210e4174b35c24a66fa93d6c08a8a889eeb1.jpg)

![4d9e8efc6ac9cf5da64efabf75763bd83bb7bf531cfa3994d3a279569396db41.jpg](../iclr_results/1710_Dreamweaver_ Learning Compositional World Models from Pixels/images/4d9e8efc6ac9cf5da64efabf75763bd83bb7bf531cfa3994d3a279569396db41.jpg)

![52da6ccf8395d32c78386d8f9355fe1062463044c45546e15530f25a240aa7fb.jpg](../iclr_results/1710_Dreamweaver_ Learning Compositional World Models from Pixels/images/52da6ccf8395d32c78386d8f9355fe1062463044c45546e15530f25a240aa7fb.jpg)

![5c486256154be656809155ac7901d80048ddacbc37bcd42d52bbf858d66e8ec4.jpg](../iclr_results/1710_Dreamweaver_ Learning Compositional World Models from Pixels/images/5c486256154be656809155ac7901d80048ddacbc37bcd42d52bbf858d66e8ec4.jpg)

![919c4189fe6198382b6d320dd36f39066a0b884a309858de624cbadbeafae6ba.jpg](../iclr_results/1710_Dreamweaver_ Learning Compositional World Models from Pixels/images/919c4189fe6198382b6d320dd36f39066a0b884a309858de624cbadbeafae6ba.jpg)

![9d168bee8bf9ef59bd92a3c258c1ca5d1c026d138dca3a626b066cd2bf79c03d.jpg](../iclr_results/1710_Dreamweaver_ Learning Compositional World Models from Pixels/images/9d168bee8bf9ef59bd92a3c258c1ca5d1c026d138dca3a626b066cd2bf79c03d.jpg)

![b5b4c4a01a6704bc248e2c5335342107e6a66e6288dac714437f4190d5f24527.jpg](../iclr_results/1710_Dreamweaver_ Learning Compositional World Models from Pixels/images/b5b4c4a01a6704bc248e2c5335342107e6a66e6288dac714437f4190d5f24527.jpg)

![c49c5c1a0725fd279a10829ba7f71b812d559d0f6ac782c2f450986b33acfcd4.jpg](../iclr_results/1710_Dreamweaver_ Learning Compositional World Models from Pixels/images/c49c5c1a0725fd279a10829ba7f71b812d559d0f6ac782c2f450986b33acfcd4.jpg)

![d44cd6371676ad441fb2bbeb06a237fa2f24631d2cea79c7f3e3815fb61125c8.jpg](../iclr_results/1710_Dreamweaver_ Learning Compositional World Models from Pixels/images/d44cd6371676ad441fb2bbeb06a237fa2f24631d2cea79c7f3e3815fb61125c8.jpg)

![df14caa0e3276711cba633a379f54907dd88390ab3cd954d2ddf90ec187c1460.jpg](../iclr_results/1710_Dreamweaver_ Learning Compositional World Models from Pixels/images/df14caa0e3276711cba633a379f54907dd88390ab3cd954d2ddf90ec187c1460.jpg)

![df7ba2e621cdb67ff941b36f917e548545fd5b98438d95d54a5fc708832c1539.jpg](../iclr_results/1710_Dreamweaver_ Learning Compositional World Models from Pixels/images/df7ba2e621cdb67ff941b36f917e548545fd5b98438d95d54a5fc708832c1539.jpg)

### Tables

![167a3667a8592b771ef50e3ad49edeae7de33075b8f7f6d3c771b6679ae66262.jpg](../iclr_results/1710_Dreamweaver_ Learning Compositional World Models from Pixels/tables/167a3667a8592b771ef50e3ad49edeae7de33075b8f7f6d3c771b6679ae66262.jpg)

![4001b23d4bb24a77f77bb5db2897a27e298a4e57a499ba626b2d5df41c45648e.jpg](../iclr_results/1710_Dreamweaver_ Learning Compositional World Models from Pixels/tables/4001b23d4bb24a77f77bb5db2897a27e298a4e57a499ba626b2d5df41c45648e.jpg)

![5b171ea231182a0dc52a3bb3b4d7d005efb704317aa93d3fcdaecb2927a7f2b1.jpg](../iclr_results/1710_Dreamweaver_ Learning Compositional World Models from Pixels/tables/5b171ea231182a0dc52a3bb3b4d7d005efb704317aa93d3fcdaecb2927a7f2b1.jpg)

![5fb7137def00e0870452bc802d478b157d2baeaf580aa9fd51a93f6260e34fd0.jpg](../iclr_results/1710_Dreamweaver_ Learning Compositional World Models from Pixels/tables/5fb7137def00e0870452bc802d478b157d2baeaf580aa9fd51a93f6260e34fd0.jpg)

![8feb91b55206a02b872ef3f9c83d0906ee3e4ec724cb4993b9433ae668951cd2.jpg](../iclr_results/1710_Dreamweaver_ Learning Compositional World Models from Pixels/tables/8feb91b55206a02b872ef3f9c83d0906ee3e4ec724cb4993b9433ae668951cd2.jpg)

![a58b67b63bc5a5fa588c339b5c9f3392df39b899efece39d0772d55d4c0dc749.jpg](../iclr_results/1710_Dreamweaver_ Learning Compositional World Models from Pixels/tables/a58b67b63bc5a5fa588c339b5c9f3392df39b899efece39d0772d55d4c0dc749.jpg)

![adb3e1e8f8d28b42e214c233263fa3b2907a783404dd0c7a0aafa5c11b44800c.jpg](../iclr_results/1710_Dreamweaver_ Learning Compositional World Models from Pixels/tables/adb3e1e8f8d28b42e214c233263fa3b2907a783404dd0c7a0aafa5c11b44800c.jpg)

## Policy Decorator: Model-Agnostic Online Refinement for Large Policy Model


### Images

![078dd554691975175dfc683bd6ae8df7ef18f367c03f9b9dc76b42f7f3aedc32.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/078dd554691975175dfc683bd6ae8df7ef18f367c03f9b9dc76b42f7f3aedc32.jpg)

![1edea5d36596e5e0f82b8fd26a7a094da35cc9bf64cc66fe25811d50589c334c.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/1edea5d36596e5e0f82b8fd26a7a094da35cc9bf64cc66fe25811d50589c334c.jpg)

![23b2a87a6478335dacb40f95fe5f0a4c7c8089b736cc3eee1338d36f29f2f04b.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/23b2a87a6478335dacb40f95fe5f0a4c7c8089b736cc3eee1338d36f29f2f04b.jpg)

![2ca443bb896e4376ae2c80f3e94d05f2fbef9770daaf8acd1a25288122e08aad.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/2ca443bb896e4376ae2c80f3e94d05f2fbef9770daaf8acd1a25288122e08aad.jpg)

![325eac09316a299131c701debe098a2976e01ae8625956fba5643e8bf308e822.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/325eac09316a299131c701debe098a2976e01ae8625956fba5643e8bf308e822.jpg)

![350fb6a04d647d06b4b44e6e7d4f11726bb0992d1a6d8c96aa5cf406a6314ff3.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/350fb6a04d647d06b4b44e6e7d4f11726bb0992d1a6d8c96aa5cf406a6314ff3.jpg)

![3ceddb3626ed057f332e57b4d776ca37a0d2a3517be05c412abaccb6029a870a.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/3ceddb3626ed057f332e57b4d776ca37a0d2a3517be05c412abaccb6029a870a.jpg)

![4d49f0126c420537a3d0447e1a1c2fdede5f23d0cd303172d547e9a396847917.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/4d49f0126c420537a3d0447e1a1c2fdede5f23d0cd303172d547e9a396847917.jpg)

![514a826e3cce309b275ebd994b49c4545348354dea258cd51b3f2b59a7809f59.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/514a826e3cce309b275ebd994b49c4545348354dea258cd51b3f2b59a7809f59.jpg)

![5937304943555e7e56eae60f979127754295c9e1e1008ab17ac6554d91f2d896.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/5937304943555e7e56eae60f979127754295c9e1e1008ab17ac6554d91f2d896.jpg)

![6529b9f7084ffe7df31744c27cef69ca81844daf4f46d62e491409a5cd409188.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/6529b9f7084ffe7df31744c27cef69ca81844daf4f46d62e491409a5cd409188.jpg)

![65d1bfca8e31f2c612cbfc4e21f1447da6e544ab89aca5da1d0489439015da34.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/65d1bfca8e31f2c612cbfc4e21f1447da6e544ab89aca5da1d0489439015da34.jpg)

![68c02406b8a024749a3b2bfcde6051cd6d525ba333a50fa9ea2b7bc34b7558cf.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/68c02406b8a024749a3b2bfcde6051cd6d525ba333a50fa9ea2b7bc34b7558cf.jpg)

![768075f288ddff768641c9880d25fbe0d7f06264a52fd107d6b7d6589b666fab.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/768075f288ddff768641c9880d25fbe0d7f06264a52fd107d6b7d6589b666fab.jpg)

![7b6cbe4669482cb0c53d04c21a86948f38b02812d5b98730bf724dde6829d092.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/7b6cbe4669482cb0c53d04c21a86948f38b02812d5b98730bf724dde6829d092.jpg)

![95e017000e9e64eb649567824acdfaac886575a74a69f3d2abaaa8785ab24021.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/95e017000e9e64eb649567824acdfaac886575a74a69f3d2abaaa8785ab24021.jpg)

![9a630d1de2e6783fd216a821ef6399ef7088f29f4b2066a097baabb9a979e260.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/9a630d1de2e6783fd216a821ef6399ef7088f29f4b2066a097baabb9a979e260.jpg)

![9bbf4e316291b1616fc1328bad164562dbb73155527abf6f8e672c9b8fc29fea.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/9bbf4e316291b1616fc1328bad164562dbb73155527abf6f8e672c9b8fc29fea.jpg)

![a570a8489e9e9075d7a0142cec0da623a7adbc1f91ca0d41c226d5f221a32c52.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/a570a8489e9e9075d7a0142cec0da623a7adbc1f91ca0d41c226d5f221a32c52.jpg)

![a7ec4472e3e54e765dcdfef6bb8c7c191db97c98218ad6035f5da86dedd64aec.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/a7ec4472e3e54e765dcdfef6bb8c7c191db97c98218ad6035f5da86dedd64aec.jpg)

![a90a86d3b0cd9e5028d307bc7eb7f825b012bd8e3681ef82acf587bc3e476b7c.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/a90a86d3b0cd9e5028d307bc7eb7f825b012bd8e3681ef82acf587bc3e476b7c.jpg)

![b3474d649296a5b373fcdd20262f4a02bd736d191062a44716ec225000c7970c.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/b3474d649296a5b373fcdd20262f4a02bd736d191062a44716ec225000c7970c.jpg)

![b7364c6dd99a443fdbac0dc7aa84e903a30e0b0852d5fbf04c8228f4cc00a539.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/b7364c6dd99a443fdbac0dc7aa84e903a30e0b0852d5fbf04c8228f4cc00a539.jpg)

![bae0ab11292f1ff89be6e1c0bb9eb97f8f852726012157106528a7bb2311bb3b.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/bae0ab11292f1ff89be6e1c0bb9eb97f8f852726012157106528a7bb2311bb3b.jpg)

![c855dea23d424f0796521111fe16329aee7c90e52710a508241975a3292620d5.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/c855dea23d424f0796521111fe16329aee7c90e52710a508241975a3292620d5.jpg)

![ca5d32af8ddf4846f3070daac023f3c1128091b1db8601a012bed6757b83fd16.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/ca5d32af8ddf4846f3070daac023f3c1128091b1db8601a012bed6757b83fd16.jpg)

![d2193f78fa990d009eda5715a8b80976d9cc8d1ab6efa93c18e81b2e3ab62ef0.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/d2193f78fa990d009eda5715a8b80976d9cc8d1ab6efa93c18e81b2e3ab62ef0.jpg)

![dc5be19eccf5d7c44eb6b7a29be85bf644e8cba2dfe4fd237c45090f3e480dd5.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/dc5be19eccf5d7c44eb6b7a29be85bf644e8cba2dfe4fd237c45090f3e480dd5.jpg)

![ec2299312298183ae180aaf386c0230e74a9b05b333f94963ba3b7a2b1111f73.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/ec2299312298183ae180aaf386c0230e74a9b05b333f94963ba3b7a2b1111f73.jpg)

![ed3aa9838d4c5ae9f7952785d07852cc4ffd42bc77136cfa1a44f53940bc5b9d.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/images/ed3aa9838d4c5ae9f7952785d07852cc4ffd42bc77136cfa1a44f53940bc5b9d.jpg)

### Tables

![0a4bca2efd489b9c4689dd951bedd12ef70c4f018c365f6b62d39e79284de419.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/tables/0a4bca2efd489b9c4689dd951bedd12ef70c4f018c365f6b62d39e79284de419.jpg)

![0a71f54cef2eaaec0f4bd6fb160afb953835ae521be4df60572bf72ee7f6ef0d.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/tables/0a71f54cef2eaaec0f4bd6fb160afb953835ae521be4df60572bf72ee7f6ef0d.jpg)

![1cc6e7623feca6bfd7d9826bc65df1158379c3c9d6a82dcaf11034975a58017a.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/tables/1cc6e7623feca6bfd7d9826bc65df1158379c3c9d6a82dcaf11034975a58017a.jpg)

![2a2111fc90bafafd4213eba8be387e2de45eef6e28cccbf306227072f0014802.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/tables/2a2111fc90bafafd4213eba8be387e2de45eef6e28cccbf306227072f0014802.jpg)

![2ef014c6e54bad9fd2b81f50b6bcdea4c769b696ccb609948eeb29cd07fe4a6e.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/tables/2ef014c6e54bad9fd2b81f50b6bcdea4c769b696ccb609948eeb29cd07fe4a6e.jpg)

![4a137c831729d6dfe3ea69071d3d9ea31a1d014ba05e758bbed53ddecf4c3e1b.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/tables/4a137c831729d6dfe3ea69071d3d9ea31a1d014ba05e758bbed53ddecf4c3e1b.jpg)

![4aeef149a3728864ab6ecd072cf3020bb4d49ff85d7241a3ad7162189f4b1ca2.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/tables/4aeef149a3728864ab6ecd072cf3020bb4d49ff85d7241a3ad7162189f4b1ca2.jpg)

![68370cb508c18ab5be82495f620d82a3de5cb6dc1379a17c0c96c68d85b32b75.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/tables/68370cb508c18ab5be82495f620d82a3de5cb6dc1379a17c0c96c68d85b32b75.jpg)

![6f68c53405876c62ccd54e9964352a62b17133899e84b61eb145fe8bd95367e0.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/tables/6f68c53405876c62ccd54e9964352a62b17133899e84b61eb145fe8bd95367e0.jpg)

![7b3edaa5151dfcfd1bb9f57b4323f9a73960ad611870603b6cb01091b8661c65.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/tables/7b3edaa5151dfcfd1bb9f57b4323f9a73960ad611870603b6cb01091b8661c65.jpg)

![a23b275749c20d09652494e4221f668de4923782fd160b0c9bbcb8488bc2e325.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/tables/a23b275749c20d09652494e4221f668de4923782fd160b0c9bbcb8488bc2e325.jpg)

![a880218b9b14d1b504e9727c9aa1362a6d0d22bf35c307f13ecc8dafd5b4aa3b.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/tables/a880218b9b14d1b504e9727c9aa1362a6d0d22bf35c307f13ecc8dafd5b4aa3b.jpg)

![d5673831238ef6a9b26de1d22008fb5a8f2e8e0721bf73aec681b36298b89e10.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/tables/d5673831238ef6a9b26de1d22008fb5a8f2e8e0721bf73aec681b36298b89e10.jpg)

![db7354ba8d1010aeabf29b1f96e26d4bdeefc03f743480c56e93af3aac8b9bd3.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/tables/db7354ba8d1010aeabf29b1f96e26d4bdeefc03f743480c56e93af3aac8b9bd3.jpg)

![fd4177bc80e8fbd42f2e5732cda5d8de9371baf9fa65988c7bd8e78a52e6edce.jpg](../iclr_results/1711_Policy Decorator_ Model-Agnostic Online Refinement for Large Policy Model/tables/fd4177bc80e8fbd42f2e5732cda5d8de9371baf9fa65988c7bd8e78a52e6edce.jpg)

## Ensembling Diffusion Models via Adaptive Feature Aggregation


### Images

![09dfaec440ca975a2176976b55f7bc333666f07558a1eda3838ad6bddd5ae160.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/images/09dfaec440ca975a2176976b55f7bc333666f07558a1eda3838ad6bddd5ae160.jpg)

![0ad6080700f1ec9cea756f4ba0354c906963d17522c3c7852a01a310c87191fa.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/images/0ad6080700f1ec9cea756f4ba0354c906963d17522c3c7852a01a310c87191fa.jpg)

![26eaa8f5302006b82e0a87bc46c45ac45852b407d0afd668467ea89a9469d8f1.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/images/26eaa8f5302006b82e0a87bc46c45ac45852b407d0afd668467ea89a9469d8f1.jpg)

![29913e6e32d5aed0bb8f2387d5bc7d3dbd183b77b182ce0248161355ed0964e1.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/images/29913e6e32d5aed0bb8f2387d5bc7d3dbd183b77b182ce0248161355ed0964e1.jpg)

![31cf4b2de97bfb5a45e8e7c308990cf500bbb58ecb59ca997c48971ef97c67e5.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/images/31cf4b2de97bfb5a45e8e7c308990cf500bbb58ecb59ca997c48971ef97c67e5.jpg)

![5a64cf6df8f500633485d539ef370d6cc795ef57f14f2e9d6e72b58a02ef00d7.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/images/5a64cf6df8f500633485d539ef370d6cc795ef57f14f2e9d6e72b58a02ef00d7.jpg)

![5b9006912ad3ddc6d3e4a265eae83ec70ae18c1c75f3d53e3de346529561d238.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/images/5b9006912ad3ddc6d3e4a265eae83ec70ae18c1c75f3d53e3de346529561d238.jpg)

![725e8e0ab9f66782d1be999ae58b404e2e777903c068eea34a658152cc4e76c1.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/images/725e8e0ab9f66782d1be999ae58b404e2e777903c068eea34a658152cc4e76c1.jpg)

![8628be2a25f6a4ebb91fa76a9fc9bf8c101bb09e6044a70e9a2fc83389911f8a.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/images/8628be2a25f6a4ebb91fa76a9fc9bf8c101bb09e6044a70e9a2fc83389911f8a.jpg)

![9b19bf827c2e8a1c431060be47c6913cc22c9ab4d2e6731d409878a46e68b51c.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/images/9b19bf827c2e8a1c431060be47c6913cc22c9ab4d2e6731d409878a46e68b51c.jpg)

![9e03a87891c932a2f0b753d012867929e65469000a1da73f561dec887f6ffd9e.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/images/9e03a87891c932a2f0b753d012867929e65469000a1da73f561dec887f6ffd9e.jpg)

![afffb27fb3d667e253a978802b7461c0b6b9eb65d33a7e820ee0816e0d71bd3e.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/images/afffb27fb3d667e253a978802b7461c0b6b9eb65d33a7e820ee0816e0d71bd3e.jpg)

![c1708c8c73eb88c1a10eecc2037e79cdb1d83fac9fce4532dc632c1f93727210.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/images/c1708c8c73eb88c1a10eecc2037e79cdb1d83fac9fce4532dc632c1f93727210.jpg)

![dea7f00aa90f9d18fdd2f5fd5f3aea7051185e16985c69720c24273586d91443.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/images/dea7f00aa90f9d18fdd2f5fd5f3aea7051185e16985c69720c24273586d91443.jpg)

![e8e784f4f31211d8403875385147ae455e7c5301bc77cfe9820cda5eedcac0c9.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/images/e8e784f4f31211d8403875385147ae455e7c5301bc77cfe9820cda5eedcac0c9.jpg)

![f4b5aa2fad17db722ab40d388eeefad9e4716e57df54309ba71c29dca0722df5.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/images/f4b5aa2fad17db722ab40d388eeefad9e4716e57df54309ba71c29dca0722df5.jpg)

![f6e492e96aec1cee67aed1b6d181db168e21ae4c059fddbfa7bdcf6ccb26ee25.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/images/f6e492e96aec1cee67aed1b6d181db168e21ae4c059fddbfa7bdcf6ccb26ee25.jpg)

![ff5e989ca102c51357f61fb43b20bac1a1d220391be1d04bdf738b25f5f503c9.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/images/ff5e989ca102c51357f61fb43b20bac1a1d220391be1d04bdf738b25f5f503c9.jpg)

### Tables

![0d4157c98dedf3431b5980c5a534e25cfff8e1ac007c4d6c83a0cce4b273284a.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/tables/0d4157c98dedf3431b5980c5a534e25cfff8e1ac007c4d6c83a0cce4b273284a.jpg)

![101fcc53847000ccf3217d81299c15b4137762586379bab9714c37ee52327079.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/tables/101fcc53847000ccf3217d81299c15b4137762586379bab9714c37ee52327079.jpg)

![2fbfca90f5bc8930ed48bcfc88df53fbb20bc960c549a13872ceabe4aa4925b9.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/tables/2fbfca90f5bc8930ed48bcfc88df53fbb20bc960c549a13872ceabe4aa4925b9.jpg)

![32858ce17581d022cdfbd32de72f3a607fa57494201e4c5e76af366a64875637.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/tables/32858ce17581d022cdfbd32de72f3a607fa57494201e4c5e76af366a64875637.jpg)

![385335ef9299b543d27d3be800ce2f5e91670221e31b27854574f9d33719b67d.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/tables/385335ef9299b543d27d3be800ce2f5e91670221e31b27854574f9d33719b67d.jpg)

![48ca0b524a1ab3e74d7f98e67458b088592bf8936dce8dafdcf15882827f96c7.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/tables/48ca0b524a1ab3e74d7f98e67458b088592bf8936dce8dafdcf15882827f96c7.jpg)

![881943604361400cdef90b1bda46b13e8a7071e64ca918ff2c0683af85a4de58.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/tables/881943604361400cdef90b1bda46b13e8a7071e64ca918ff2c0683af85a4de58.jpg)

![8d8b044b9ce2b5db3ff69b6540cbc29d3bb7f6b67c64a69de367eea9f95d4b6e.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/tables/8d8b044b9ce2b5db3ff69b6540cbc29d3bb7f6b67c64a69de367eea9f95d4b6e.jpg)

![94bdc83563818c5cc9798c49ca47c4dafdf7d7eb5f5bcfc5057ac62ed7a17577.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/tables/94bdc83563818c5cc9798c49ca47c4dafdf7d7eb5f5bcfc5057ac62ed7a17577.jpg)

![983565796520a265fe3b9a07e3c822b2f9afacd6583a90d1cdfa9cbc4d774498.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/tables/983565796520a265fe3b9a07e3c822b2f9afacd6583a90d1cdfa9cbc4d774498.jpg)

![a91f684170ea98339fb52f1a8bed936b556c9f101c4166335bdd6cec3aea3c96.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/tables/a91f684170ea98339fb52f1a8bed936b556c9f101c4166335bdd6cec3aea3c96.jpg)

![b9689ea99991182e41b3dbc2336ae79aef42225d3f762b1b831d219c7513a9dd.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/tables/b9689ea99991182e41b3dbc2336ae79aef42225d3f762b1b831d219c7513a9dd.jpg)

![d669f04efa41dad034defa4215aa55655f8ea7889e57e4fe5160ed9a19521996.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/tables/d669f04efa41dad034defa4215aa55655f8ea7889e57e4fe5160ed9a19521996.jpg)

![e183b71e1c0966e3fb9165d9db1cd7ebc3ca0952cf4f7ff64a7bd9580b8d7e1b.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/tables/e183b71e1c0966e3fb9165d9db1cd7ebc3ca0952cf4f7ff64a7bd9580b8d7e1b.jpg)

![e33d4bc003012138e72c82391cda3f5a6e0f1212c1d8965e4d25de3332440e1c.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/tables/e33d4bc003012138e72c82391cda3f5a6e0f1212c1d8965e4d25de3332440e1c.jpg)

![f501d2f40a5261ab2ac43f73014f2173816a02f3bebe5dba12b692abac152fe4.jpg](../iclr_results/1712_Ensembling Diffusion Models via Adaptive Feature Aggregation/tables/f501d2f40a5261ab2ac43f73014f2173816a02f3bebe5dba12b692abac152fe4.jpg)

## Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models

### Images

![121d2fbce3a723c2a738899e7cb82ec0b8805865783db200a2d8a34b57731300.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/images/121d2fbce3a723c2a738899e7cb82ec0b8805865783db200a2d8a34b57731300.jpg)

![128003d42247316b83287e88639a5c7e31dd8eb34fedc045500e3aa092420150.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/images/128003d42247316b83287e88639a5c7e31dd8eb34fedc045500e3aa092420150.jpg)

![1e049f6afe0bdf92aa2c38f5e861d69e86e6aa3e0178eedc3ca1b296aa754958.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/images/1e049f6afe0bdf92aa2c38f5e861d69e86e6aa3e0178eedc3ca1b296aa754958.jpg)

![3cabb96405f856a53f778f62619150125c6fffc339006b593d9eee9829481b1d.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/images/3cabb96405f856a53f778f62619150125c6fffc339006b593d9eee9829481b1d.jpg)

![44ecd525ba370dcb30c334c35b8afdc67444300f876039d065f804b29521c010.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/images/44ecd525ba370dcb30c334c35b8afdc67444300f876039d065f804b29521c010.jpg)

![4f4515f69d3fe79b5da057f532fd3224bfc28f8ca0898faa26c4e7bf4e012e20.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/images/4f4515f69d3fe79b5da057f532fd3224bfc28f8ca0898faa26c4e7bf4e012e20.jpg)

![5b05926a3033dcdadfb5d3866a3b0683aba285fbb09caf79b912fadb60623fb5.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/images/5b05926a3033dcdadfb5d3866a3b0683aba285fbb09caf79b912fadb60623fb5.jpg)

![5fec8e69645d02003e76d794b3fbded9f1d7fe619514dc7484a7abd6b3b1b32c.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/images/5fec8e69645d02003e76d794b3fbded9f1d7fe619514dc7484a7abd6b3b1b32c.jpg)

![75bfef5a8e06d418d0aea2b42a8ce9995e2c9ef993a788b05b0ef9cc0924961d.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/images/75bfef5a8e06d418d0aea2b42a8ce9995e2c9ef993a788b05b0ef9cc0924961d.jpg)

![83422b182bbd220b1263307d8df085b1c514146993564f9fab71e4ded2eb9ae8.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/images/83422b182bbd220b1263307d8df085b1c514146993564f9fab71e4ded2eb9ae8.jpg)

![87520401da94f99fdceac076610fbaecf12bef6060ab80e619d82d752eab17ac.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/images/87520401da94f99fdceac076610fbaecf12bef6060ab80e619d82d752eab17ac.jpg)

![8da03066a59192b3ff802637bb9776057664bed1c2de3fb551afbf425aeaa169.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/images/8da03066a59192b3ff802637bb9776057664bed1c2de3fb551afbf425aeaa169.jpg)

![a4432c1efd71d95fa56daac0c54ab6e8e15e32f9842baa7e0ab9ba2dd6d7532e.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/images/a4432c1efd71d95fa56daac0c54ab6e8e15e32f9842baa7e0ab9ba2dd6d7532e.jpg)

![a5aa23ee4095058da566c684dad1387d37a957d0135a935da34512265aed393e.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/images/a5aa23ee4095058da566c684dad1387d37a957d0135a935da34512265aed393e.jpg)

![abce457db0456202cad07bd006dc710ba75e8fc78e906e30d57ed70b7b7090c9.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/images/abce457db0456202cad07bd006dc710ba75e8fc78e906e30d57ed70b7b7090c9.jpg)

![b97c13184d55f5801038fb8fabe316f7575514d5dac26bb51a58097017026fc9.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/images/b97c13184d55f5801038fb8fabe316f7575514d5dac26bb51a58097017026fc9.jpg)

![bd45e1c396b6ea5871fced8da94b1dcf350ab1c89d5a42aaabbcf4c7adc683d5.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/images/bd45e1c396b6ea5871fced8da94b1dcf350ab1c89d5a42aaabbcf4c7adc683d5.jpg)

![c85e817f346e30f19603abcb24b53c251b67d44a58c8c5a554d130bf6cd3a9b6.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/images/c85e817f346e30f19603abcb24b53c251b67d44a58c8c5a554d130bf6cd3a9b6.jpg)

![d3dc516ddb56d9532277a73610a9f6a29b182f7a7147f1c69eb366f4d02f2e1f.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/images/d3dc516ddb56d9532277a73610a9f6a29b182f7a7147f1c69eb366f4d02f2e1f.jpg)

![d9202cf40a88ed445bda1f7eb52bbef011f15d7598646fe7e96bad37528f1ed5.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/images/d9202cf40a88ed445bda1f7eb52bbef011f15d7598646fe7e96bad37528f1ed5.jpg)

![e66dcefbed6126e0268cc9b6f92e4aebcbe470963f51e7016ccbacaa81cb7449.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/images/e66dcefbed6126e0268cc9b6f92e4aebcbe470963f51e7016ccbacaa81cb7449.jpg)

![eaec4f3209ff543d91d49c000d7c0ebc265d6cb3a4e45b1ee89b7e9699033d93.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/images/eaec4f3209ff543d91d49c000d7c0ebc265d6cb3a4e45b1ee89b7e9699033d93.jpg)

![f884d62b08e8d031519108ab09792eb2797e1bb9ea4a4d24934b804e3fac523a.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/images/f884d62b08e8d031519108ab09792eb2797e1bb9ea4a4d24934b804e3fac523a.jpg)

### Tables

![0aa5db501112ab41aed033f51ec2a5266db9846cdf76693cccd0325985c9000b.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/tables/0aa5db501112ab41aed033f51ec2a5266db9846cdf76693cccd0325985c9000b.jpg)

![16d0eb86f9ead18379b946c325cce0fd9c9e0e87fa45088aae67714418689e99.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/tables/16d0eb86f9ead18379b946c325cce0fd9c9e0e87fa45088aae67714418689e99.jpg)

![18d34c78fd4d295ed8dbec98dd8ec6c0a2744caef32bd5888bfaf7ada2eeb773.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/tables/18d34c78fd4d295ed8dbec98dd8ec6c0a2744caef32bd5888bfaf7ada2eeb773.jpg)

![1ae76880a42dfaae8d52291533b99bfb54a2267f802678d5c8ceaf7c5df4cf30.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/tables/1ae76880a42dfaae8d52291533b99bfb54a2267f802678d5c8ceaf7c5df4cf30.jpg)

![3ec9fa51ce46de131fac1ed58a7fdf93204439aec7622564ec46877b2c1d5e91.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/tables/3ec9fa51ce46de131fac1ed58a7fdf93204439aec7622564ec46877b2c1d5e91.jpg)

![5b6d81d2eb2db3327d79cbb845b5d42e045c01c73a74e008856f542362debbb5.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/tables/5b6d81d2eb2db3327d79cbb845b5d42e045c01c73a74e008856f542362debbb5.jpg)

![6c0becd196c8e72e0656ef3a2a913d0582b01d8e5c9e341440a90a1b0a09b994.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/tables/6c0becd196c8e72e0656ef3a2a913d0582b01d8e5c9e341440a90a1b0a09b994.jpg)

![862db1bdb0efb5d6ca23a0479173aea290f752a591c3b415f81d8345eafd5c1b.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/tables/862db1bdb0efb5d6ca23a0479173aea290f752a591c3b415f81d8345eafd5c1b.jpg)

![8ddd5151953140f32e6cad51fa80d62ea78d6eda25ebc42f49dce6ee869f5f1b.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/tables/8ddd5151953140f32e6cad51fa80d62ea78d6eda25ebc42f49dce6ee869f5f1b.jpg)

![98df5fcd589389ba6739583d0f6c2796dd4b79de601332477ae563c5d01d44f1.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/tables/98df5fcd589389ba6739583d0f6c2796dd4b79de601332477ae563c5d01d44f1.jpg)

![ae258f87feb22a0047c16e03e937e97fa06d1c69206c584cc307a57faa200fe0.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/tables/ae258f87feb22a0047c16e03e937e97fa06d1c69206c584cc307a57faa200fe0.jpg)

![ce2013c5901e369725483f9777a849ed37daa326b8239036af61c8489ccbea88.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/tables/ce2013c5901e369725483f9777a849ed37daa326b8239036af61c8489ccbea88.jpg)

![d278604584eb20622719a47170ab873a9d7ec326261cd094e17b035b4d4c62e8.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/tables/d278604584eb20622719a47170ab873a9d7ec326261cd094e17b035b4d4c62e8.jpg)

![e79e9524129ca187d7d9b325d341db1a4967f8b410169bace240fde669c98cd8.jpg](../iclr_results/1713_Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models/tables/e79e9524129ca187d7d9b325d341db1a4967f8b410169bace240fde669c98cd8.jpg)
