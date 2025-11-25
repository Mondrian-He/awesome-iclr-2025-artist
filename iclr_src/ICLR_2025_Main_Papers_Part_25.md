# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 25 of 100

## 目录 (Table of Contents)

1. [Learning Partial Graph Matching via Optimal Partial Transport](#Learning-Partial-Graph-Matching-via-Optimal-Partial-Transport)
2. [Towards Neural Scaling Laws for Time Series Foundation Models](#Towards-Neural-Scaling-Laws-for-Time-Series-Foundation-Models)
3. [Compositional simulation-based inference for time series](#Compositional-simulation-based-inference-for-time-series)
4. [Jailbreaking as a Reward Misspecification Problem](#Jailbreaking-as-a-Reward-Misspecification-Problem)
5. [Equivariant Neural Functional Networks for Transformers](#Equivariant-Neural-Functional-Networks-for-Transformers)
6. [Adaptive Rank Allocation: Speeding Up Modern Transformers with RaNA Adapters](#Adaptive-Rank-Allocation-Speeding-Up-Modern-Transformers-with-RaNA-Adapters)
7. [Exploring Prosocial Irrationality for LLM Agents: A Social Cognition View](#Exploring-Prosocial-Irrationality-for-LLM-Agents-A-Social-Cognition-View)
8. [Deriving Causal Order from Single-Variable Interventions: Guarantees & Algorithm](#Deriving-Causal-Order-from-Single-Variable-Interventions-Guarantees-Algorithm)
9. [SeedLM: Compressing LLM Weights into Seeds of Pseudo-Random Generators](#SeedLM-Compressing-LLM-Weights-into-Seeds-of-Pseudo-Random-Generators)
10. [Context-aware Dynamic Pruning for Speech Foundation Models](#Context-aware-Dynamic-Pruning-for-Speech-Foundation-Models)
11. [On the Performance Analysis of Momentum Method: A Frequency Domain Perspective](#On-the-Performance-Analysis-of-Momentum-Method-A-Frequency-Domain-Perspective)
12. [PEARL: Towards Permutation-Resilient LLMs](#PEARL-Towards-Permutation-Resilient-LLMs)
13. [RAPID: Retrieval Augmented Training of Differentially Private Diffusion Models](#RAPID-Retrieval-Augmented-Training-of-Differentially-Private-Diffusion-Models)
14. [Near-Exact Privacy Amplification for Matrix Mechanisms](#Near-Exact-Privacy-Amplification-for-Matrix-Mechanisms)
15. [A Coefficient Makes SVRG Effective](#A-Coefficient-Makes-SVRG-Effective)
16. [Cross-Embodiment Dexterous Grasping with Reinforcement Learning](#Cross-Embodiment-Dexterous-Grasping-with-Reinforcement-Learning)
17. [On Generalization Across Environments In Multi-Objective Reinforcement Learning](#On-Generalization-Across-Environments-In-Multi-Objective-Reinforcement-Learning)
18. [Composable Interventions for Language Models](#Composable-Interventions-for-Language-Models)
19. [Bounds on $L_p$ Errors in Density Ratio Estimation via $f$-Divergence Loss Functions](#Bounds-on-L_p-Errors-in-Density-Ratio-Estimation-via-f-Divergence-Loss-Functions)
20. [Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathology](#Interpretable-Vision-Language-Survival-Analysis-with-Ordinal-Inductive-Bias-for-Computational-Pathology)
21. [Accelerating Neural ODEs: A Variational Formulation-based Approach](#Accelerating-Neural-ODEs-A-Variational-Formulation-based-Approach)
22. [RainbowPO: A Unified Framework for Combining Improvements in Preference Optimization](#RainbowPO-A-Unified-Framework-for-Combining-Improvements-in-Preference-Optimization)
23. [Physics-Informed Diffusion Models](#Physics-Informed-Diffusion-Models)
24. [ProAdvPrompter: A Two-Stage Journey to Effective Adversarial Prompting for LLMs](#ProAdvPrompter-A-Two-Stage-Journey-to-Effective-Adversarial-Prompting-for-LLMs)
25. [Improving Large Language Model Planning with Action Sequence Similarity](#Improving-Large-Language-Model-Planning-with-Action-Sequence-Similarity)
26. [Hydra-SGG: Hybrid Relation Assignment for One-stage Scene Graph Generation](#Hydra-SGG-Hybrid-Relation-Assignment-for-One-stage-Scene-Graph-Generation)
27. [Heavy-Tailed Diffusion Models](#Heavy-Tailed-Diffusion-Models)
28. [Rethinking Multiple-Instance Learning From Feature Space to Probability Space](#Rethinking-Multiple-Instance-Learning-From-Feature-Space-to-Probability-Space)
29. [KGARevion: An AI Agent for Knowledge-Intensive Biomedical QA](#KGARevion-An-AI-Agent-for-Knowledge-Intensive-Biomedical-QA)
30. [DOTS: Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search](#DOTS-Learning-to-Reason-Dynamically-in-LLMs-via-Optimal-Reasoning-Trajectories-Search)
31. [Learning the Complexity of Weakly Noisy Quantum States](#Learning-the-Complexity-of-Weakly-Noisy-Quantum-States)
32. [RazorAttention: Efficient KV Cache Compression Through Retrieval Heads](#RazorAttention-Efficient-KV-Cache-Compression-Through-Retrieval-Heads)
33. [An Image is Worth More Than 16x16 Patches: Exploring Transformers on Individual Pixels](#An-Image-is-Worth-More-Than-16x16-Patches-Exploring-Transformers-on-Individual-Pixels)
34. [BAMDP Shaping: a Unified Framework for Intrinsic Motivation and Reward Shaping](#BAMDP-Shaping-a-Unified-Framework-for-Intrinsic-Motivation-and-Reward-Shaping)
35. [Rationalizing and Augmenting Dynamic Graph Neural Networks](#Rationalizing-and-Augmenting-Dynamic-Graph-Neural-Networks)
36. [Training-Free Diffusion Model Alignment with Sampling Demons](#Training-Free-Diffusion-Model-Alignment-with-Sampling-Demons)
37. [Stochastic Semi-Gradient Descent for Learning Mean Field Games with Population-Aware Function Approximation](#Stochastic-Semi-Gradient-Descent-for-Learning-Mean-Field-Games-with-Population-Aware-Function-Approximation)

---


## Learning Partial Graph Matching via Optimal Partial Transport

### Images

![18064f5747416585ebc46f8946e8a5cea1c2c1725813c0fa15e06e9e60f0a63f.jpg](../iclr_results/895_Benchmarking LLMs' Judgments with No Gold Standard/images/18064f5747416585ebc46f8946e8a5cea1c2c1725813c0fa15e06e9e60f0a63f.jpg)

![2666434df8d431acfe4e47d11e1a5624d0fc92d5f076f011a896521a5c4a31d9.jpg](../iclr_results/895_Benchmarking LLMs' Judgments with No Gold Standard/images/2666434df8d431acfe4e47d11e1a5624d0fc92d5f076f011a896521a5c4a31d9.jpg)

![6939d110223f751e654cf5a45348ee46250d79b9ceb76dab5d44799322a5478b.jpg](../iclr_results/895_Benchmarking LLMs' Judgments with No Gold Standard/images/6939d110223f751e654cf5a45348ee46250d79b9ceb76dab5d44799322a5478b.jpg)

![ca11e64562ed7fa6d96652271bc09839d28ca31728effc167b155440cc646e98.jpg](../iclr_results/895_Benchmarking LLMs' Judgments with No Gold Standard/images/ca11e64562ed7fa6d96652271bc09839d28ca31728effc167b155440cc646e98.jpg)

![dbf5a77c359f2fd32e76f1488235a13ba835ed8a7c35805a5fdfa76fd5f2521b.jpg](../iclr_results/895_Benchmarking LLMs' Judgments with No Gold Standard/images/dbf5a77c359f2fd32e76f1488235a13ba835ed8a7c35805a5fdfa76fd5f2521b.jpg)

### Tables

![092c4f0837c4b17ef5f5e86259acabc1a4cfab1a1585862c065598890773ab96.jpg](../iclr_results/895_Benchmarking LLMs' Judgments with No Gold Standard/tables/092c4f0837c4b17ef5f5e86259acabc1a4cfab1a1585862c065598890773ab96.jpg)

![1fda8b5f4d260fb10b08ef59cb9407a0d12cf690aad865de39b2a7f1c5ea5a01.jpg](../iclr_results/895_Benchmarking LLMs' Judgments with No Gold Standard/tables/1fda8b5f4d260fb10b08ef59cb9407a0d12cf690aad865de39b2a7f1c5ea5a01.jpg)

![25c5be71c835e3f6971d575dc53c2fa5767e79fe1ad7cb8f2e25f0cdd8479313.jpg](../iclr_results/895_Benchmarking LLMs' Judgments with No Gold Standard/tables/25c5be71c835e3f6971d575dc53c2fa5767e79fe1ad7cb8f2e25f0cdd8479313.jpg)

![423b169a0d02b3eb8463246baf19eb90dad29863df9344fe897eb784506794db.jpg](../iclr_results/895_Benchmarking LLMs' Judgments with No Gold Standard/tables/423b169a0d02b3eb8463246baf19eb90dad29863df9344fe897eb784506794db.jpg)

![4e9a5ab6e0162922352db0f2e88c724d536a543083620d8fd13028ae94d0e7b4.jpg](../iclr_results/895_Benchmarking LLMs' Judgments with No Gold Standard/tables/4e9a5ab6e0162922352db0f2e88c724d536a543083620d8fd13028ae94d0e7b4.jpg)

![96b65a2a9c0ba4184b5ab633db3429ceca8552f8245d18abb30c3954bb0d8f82.jpg](../iclr_results/895_Benchmarking LLMs' Judgments with No Gold Standard/tables/96b65a2a9c0ba4184b5ab633db3429ceca8552f8245d18abb30c3954bb0d8f82.jpg)

![98510534569986f2decb81f01a1563f9e5c93a99a054cd15550b31581db7fe87.jpg](../iclr_results/895_Benchmarking LLMs' Judgments with No Gold Standard/tables/98510534569986f2decb81f01a1563f9e5c93a99a054cd15550b31581db7fe87.jpg)

![ace6ab8acde034b0e5d9f66a171dfc4018948695a1bf00490f95c70315cdcdca.jpg](../iclr_results/895_Benchmarking LLMs' Judgments with No Gold Standard/tables/ace6ab8acde034b0e5d9f66a171dfc4018948695a1bf00490f95c70315cdcdca.jpg)

![ad1383d37aa6023c4496aacd186e4937f5730e2fe2c72c866e900e8a99b2911c.jpg](../iclr_results/895_Benchmarking LLMs' Judgments with No Gold Standard/tables/ad1383d37aa6023c4496aacd186e4937f5730e2fe2c72c866e900e8a99b2911c.jpg)

![bf4a1f2606a601a13d3979f2bb405bfcb4b0c54229cd7413f5836ea9ad930a56.jpg](../iclr_results/895_Benchmarking LLMs' Judgments with No Gold Standard/tables/bf4a1f2606a601a13d3979f2bb405bfcb4b0c54229cd7413f5836ea9ad930a56.jpg)

![c0f5b8f7f26ebcf467e1e009b491ba16ebe89b3e0d273cb7e4c2e0779220c2ee.jpg](../iclr_results/895_Benchmarking LLMs' Judgments with No Gold Standard/tables/c0f5b8f7f26ebcf467e1e009b491ba16ebe89b3e0d273cb7e4c2e0779220c2ee.jpg)

![e160fec3c9f91bbfe7f20f3c71b243429bd56b43051640e98d0fbfa479fd5566.jpg](../iclr_results/895_Benchmarking LLMs' Judgments with No Gold Standard/tables/e160fec3c9f91bbfe7f20f3c71b243429bd56b43051640e98d0fbfa479fd5566.jpg)

![e190e90d6aada310b77aab6ec7ce9c60951be02eadb52ea3d08ce42092fbfaf3.jpg](../iclr_results/895_Benchmarking LLMs' Judgments with No Gold Standard/tables/e190e90d6aada310b77aab6ec7ce9c60951be02eadb52ea3d08ce42092fbfaf3.jpg)

![fa89f93e269d9ae5faf34378312e7e30890284fe9de766297b6a705a635dd691.jpg](../iclr_results/895_Benchmarking LLMs' Judgments with No Gold Standard/tables/fa89f93e269d9ae5faf34378312e7e30890284fe9de766297b6a705a635dd691.jpg)

![fe2c2dbd7985e17214f055101f0b1e98d1bc5e0634ebe6cd05a526c5a91efb9c.jpg](../iclr_results/895_Benchmarking LLMs' Judgments with No Gold Standard/tables/fe2c2dbd7985e17214f055101f0b1e98d1bc5e0634ebe6cd05a526c5a91efb9c.jpg)

## Learning Partial Graph Matching via Optimal Partial Transport


### Images

![5b135bf52f974b710e1762fd58633956a334e61cf85e394dc448443d7bdbe05b.jpg](../iclr_results/896_Learning Partial Graph Matching via Optimal Partial Transport/images/5b135bf52f974b710e1762fd58633956a334e61cf85e394dc448443d7bdbe05b.jpg)

![b50490f51713a595c680b016f073fee352dc1dcda0e9b8f460702729579f19c3.jpg](../iclr_results/896_Learning Partial Graph Matching via Optimal Partial Transport/images/b50490f51713a595c680b016f073fee352dc1dcda0e9b8f460702729579f19c3.jpg)

![d6b8b8e364e62d5c3295b0d864d9f1e64ef9aaf5871feb3714f01fb8d771d5fc.jpg](../iclr_results/896_Learning Partial Graph Matching via Optimal Partial Transport/images/d6b8b8e364e62d5c3295b0d864d9f1e64ef9aaf5871feb3714f01fb8d771d5fc.jpg)

![f8c0302de7aa7359ad8e338b2b1b5c715e563df8b0b7e901bdb7d03604baa44a.jpg](../iclr_results/896_Learning Partial Graph Matching via Optimal Partial Transport/images/f8c0302de7aa7359ad8e338b2b1b5c715e563df8b0b7e901bdb7d03604baa44a.jpg)

### Tables

![1520a03e5b88c0095dec8fd7d24f9cbe98eab39f46ea8c82e6e4c35fd4d79158.jpg](../iclr_results/896_Learning Partial Graph Matching via Optimal Partial Transport/tables/1520a03e5b88c0095dec8fd7d24f9cbe98eab39f46ea8c82e6e4c35fd4d79158.jpg)

![22369fdc946e1fc31b7ec68610699fb934fef49e629f57695d0197641d52dab7.jpg](../iclr_results/896_Learning Partial Graph Matching via Optimal Partial Transport/tables/22369fdc946e1fc31b7ec68610699fb934fef49e629f57695d0197641d52dab7.jpg)

![76e8159a2fd8c374ba98b209da204bfd2be5527db582c4fc6612380b65ad16a2.jpg](../iclr_results/896_Learning Partial Graph Matching via Optimal Partial Transport/tables/76e8159a2fd8c374ba98b209da204bfd2be5527db582c4fc6612380b65ad16a2.jpg)

![88dd27a9103c939d31a891027ebc032a8a166a979e217f9012a455f1c2d6a541.jpg](../iclr_results/896_Learning Partial Graph Matching via Optimal Partial Transport/tables/88dd27a9103c939d31a891027ebc032a8a166a979e217f9012a455f1c2d6a541.jpg)

## Towards Neural Scaling Laws for Time Series Foundation Models


### Images

![01b886ddc9b024c095111764417fe428d9b497d477b62dfe68a6bcfa813d7bee.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/images/01b886ddc9b024c095111764417fe428d9b497d477b62dfe68a6bcfa813d7bee.jpg)

![08dcecbb0b5ff426357e25fb78b065ded1fae09baf490d9a50e41aa352a1ffa2.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/images/08dcecbb0b5ff426357e25fb78b065ded1fae09baf490d9a50e41aa352a1ffa2.jpg)

![204ab05c2a1ab7596cf075321f0133b02ddc51695327a69a3999c31e80f17a06.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/images/204ab05c2a1ab7596cf075321f0133b02ddc51695327a69a3999c31e80f17a06.jpg)

![249d4ae719abb63233fbcf9f2576363166c0280bece2f04c8dd720d5d055fc01.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/images/249d4ae719abb63233fbcf9f2576363166c0280bece2f04c8dd720d5d055fc01.jpg)

![27e644f9847e3e7a1b7a756502b3939a3abe9a4fc9dbc965fc98e584d6472f79.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/images/27e644f9847e3e7a1b7a756502b3939a3abe9a4fc9dbc965fc98e584d6472f79.jpg)

![2cb2b1d6ac9337def445e0904ee3dfe6db5ea047de87930562f86aec7d302f07.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/images/2cb2b1d6ac9337def445e0904ee3dfe6db5ea047de87930562f86aec7d302f07.jpg)

![325722221df5169a3df6ad341289283f87dcdaeb60f74ee25b75c25d0ce36a80.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/images/325722221df5169a3df6ad341289283f87dcdaeb60f74ee25b75c25d0ce36a80.jpg)

![422f2fd8fe5640b9af1631412d973846c143bf9f2d060853e396b8b75d422b32.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/images/422f2fd8fe5640b9af1631412d973846c143bf9f2d060853e396b8b75d422b32.jpg)

![4dc07bd5f4f997efea26169009fc2ee84f5c43c0cb1014ce334976c08f296804.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/images/4dc07bd5f4f997efea26169009fc2ee84f5c43c0cb1014ce334976c08f296804.jpg)

![56669e25a294c31fd996d0f4cfd3418b76fc1c0e2d6a7af28975e4aaece74a6d.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/images/56669e25a294c31fd996d0f4cfd3418b76fc1c0e2d6a7af28975e4aaece74a6d.jpg)

![59a06981f46b8e583d047ba25fd548308b6f08a4e08d32bbcda4776ecb29cb6c.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/images/59a06981f46b8e583d047ba25fd548308b6f08a4e08d32bbcda4776ecb29cb6c.jpg)

![82900d1340aa7508493e1eeea2cd5c3fa38af2d21f73d7202c87369172e8676d.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/images/82900d1340aa7508493e1eeea2cd5c3fa38af2d21f73d7202c87369172e8676d.jpg)

![953935270601a795bf25969423b40724cf22b8c080e4666db91a2df6b737179b.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/images/953935270601a795bf25969423b40724cf22b8c080e4666db91a2df6b737179b.jpg)

![b04bad5958686391df1dd4a4eea404bded2bafa8a18d68821574d3a63dfa9a1f.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/images/b04bad5958686391df1dd4a4eea404bded2bafa8a18d68821574d3a63dfa9a1f.jpg)

![c95f2fd9fd8efdf6736a3a0f2efca05bc739bf1cb0421f1a363feeb720abef76.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/images/c95f2fd9fd8efdf6736a3a0f2efca05bc739bf1cb0421f1a363feeb720abef76.jpg)

![d2f6bcd21987b478bd275442d5479e2facc993337595d63f959c92cebb256b9b.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/images/d2f6bcd21987b478bd275442d5479e2facc993337595d63f959c92cebb256b9b.jpg)

![dc68640a7a6e62199e1b44b3aed9066d25a9b07ad47a246e2b50c9c79b7cded0.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/images/dc68640a7a6e62199e1b44b3aed9066d25a9b07ad47a246e2b50c9c79b7cded0.jpg)

![e26de6c45c45f5e29cc875a9499bb6ae4f437ebe9354b186611b7f1177e9a69e.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/images/e26de6c45c45f5e29cc875a9499bb6ae4f437ebe9354b186611b7f1177e9a69e.jpg)

![f38cefda8dc466c01d2c4068cb8bf8c6433302b13dbb42c82f59eaae070d7c20.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/images/f38cefda8dc466c01d2c4068cb8bf8c6433302b13dbb42c82f59eaae070d7c20.jpg)

### Tables

![0b08c4add543187f9d55e8cde7599bf76ad1940a87de5d8a19d52a2eb05bbfd5.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/tables/0b08c4add543187f9d55e8cde7599bf76ad1940a87de5d8a19d52a2eb05bbfd5.jpg)

![466368c4ee394edf9a0b7b6ac73873bea4ed60aba5c377301780809fe890dde0.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/tables/466368c4ee394edf9a0b7b6ac73873bea4ed60aba5c377301780809fe890dde0.jpg)

![66838de677b38f6a25abd8ff3f87d08f32a2eec5e46335facea6a33107e68ec7.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/tables/66838de677b38f6a25abd8ff3f87d08f32a2eec5e46335facea6a33107e68ec7.jpg)

![d1c9209423d5a11cb22e57347313c165a3e5ab40aa1cf5996f466def86349957.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/tables/d1c9209423d5a11cb22e57347313c165a3e5ab40aa1cf5996f466def86349957.jpg)

![d52053269991a2951d3c4a7e4dc04009bb07ba94abd9d84e54f5f7e73ae205b5.jpg](../iclr_results/897_Towards Neural Scaling Laws for Time Series Foundation Models/tables/d52053269991a2951d3c4a7e4dc04009bb07ba94abd9d84e54f5f7e73ae205b5.jpg)

## Compositional simulation-based inference for time series


### Images

![1607a3873e097b979d54e0dacfdb605bcee86385d0cce7110ff1c744a603b03e.jpg](../iclr_results/898_Compositional simulation-based inference for time series/images/1607a3873e097b979d54e0dacfdb605bcee86385d0cce7110ff1c744a603b03e.jpg)

![18907d6d67312d1a0e5ca4fa9615cbfe098d4b1897646fb30e9f81570afa3f9b.jpg](../iclr_results/898_Compositional simulation-based inference for time series/images/18907d6d67312d1a0e5ca4fa9615cbfe098d4b1897646fb30e9f81570afa3f9b.jpg)

![23673c7445fbf4689ca45165bb0820f29e63137ea2cd8467c9c28f2668383991.jpg](../iclr_results/898_Compositional simulation-based inference for time series/images/23673c7445fbf4689ca45165bb0820f29e63137ea2cd8467c9c28f2668383991.jpg)

![249697a2cd0084919514ebdb6d6e8ceb9c4258f64d61705d5b4cd693d87cf39a.jpg](../iclr_results/898_Compositional simulation-based inference for time series/images/249697a2cd0084919514ebdb6d6e8ceb9c4258f64d61705d5b4cd693d87cf39a.jpg)

![39f1d7d50a0396c597ddfcd2885ef693177ab80826124e3b9ce1937f8f90d29e.jpg](../iclr_results/898_Compositional simulation-based inference for time series/images/39f1d7d50a0396c597ddfcd2885ef693177ab80826124e3b9ce1937f8f90d29e.jpg)

![4969bbd608863dac5dae1fa5c06ae20724a9cd9b08d555c26ff82050c2937173.jpg](../iclr_results/898_Compositional simulation-based inference for time series/images/4969bbd608863dac5dae1fa5c06ae20724a9cd9b08d555c26ff82050c2937173.jpg)

![4ae77432bebe8068d7a446799e910e91e019c320febf6ea25bcf1484288c7995.jpg](../iclr_results/898_Compositional simulation-based inference for time series/images/4ae77432bebe8068d7a446799e910e91e019c320febf6ea25bcf1484288c7995.jpg)

![55e9abad93dda4a453e35f45b2f5de2cc1eb3ee2bc85711b1e7d6d9ce900d8e9.jpg](../iclr_results/898_Compositional simulation-based inference for time series/images/55e9abad93dda4a453e35f45b2f5de2cc1eb3ee2bc85711b1e7d6d9ce900d8e9.jpg)

![882bb8d5450d2dc361bef0bfad523e5905152385460329e8d4b2ac26dbb8c570.jpg](../iclr_results/898_Compositional simulation-based inference for time series/images/882bb8d5450d2dc361bef0bfad523e5905152385460329e8d4b2ac26dbb8c570.jpg)

![a2251515f3c056f631265e880a136fecc3b6313a7175414d5918c907ccc72a95.jpg](../iclr_results/898_Compositional simulation-based inference for time series/images/a2251515f3c056f631265e880a136fecc3b6313a7175414d5918c907ccc72a95.jpg)

![b4a6d7dba4bc3c9a927d2db608da3a9d538ab0d4e3751ec62708dea834b18f4f.jpg](../iclr_results/898_Compositional simulation-based inference for time series/images/b4a6d7dba4bc3c9a927d2db608da3a9d538ab0d4e3751ec62708dea834b18f4f.jpg)

![e8d3130fbfc1930aab0c92b49758c0661b6f86490d403f550e6140d078ce34e4.jpg](../iclr_results/898_Compositional simulation-based inference for time series/images/e8d3130fbfc1930aab0c92b49758c0661b6f86490d403f550e6140d078ce34e4.jpg)

![ee5d47e49d3c51b224be1fee8549e880ef1c06df001aeda0dee465be2b758e60.jpg](../iclr_results/898_Compositional simulation-based inference for time series/images/ee5d47e49d3c51b224be1fee8549e880ef1c06df001aeda0dee465be2b758e60.jpg)

### Tables

![ad390ef4eb76ce56381efbaa5f9f977217171ecafab0e751dccfca1d6db00456.jpg](../iclr_results/898_Compositional simulation-based inference for time series/tables/ad390ef4eb76ce56381efbaa5f9f977217171ecafab0e751dccfca1d6db00456.jpg)

![c89cf0b33a6b1c836e76446509da6073b21e6ed52350af5d225c7847ce30d86d.jpg](../iclr_results/898_Compositional simulation-based inference for time series/tables/c89cf0b33a6b1c836e76446509da6073b21e6ed52350af5d225c7847ce30d86d.jpg)

![d01044d31d9e44d6db44443ca335968bf0bf89dd517418bddc421210abb54bd0.jpg](../iclr_results/898_Compositional simulation-based inference for time series/tables/d01044d31d9e44d6db44443ca335968bf0bf89dd517418bddc421210abb54bd0.jpg)

![ee0a93b23d45d6ae0f7b3dad9e0ad93fbf8905e0a3668ab571e2b45074ad26b4.jpg](../iclr_results/898_Compositional simulation-based inference for time series/tables/ee0a93b23d45d6ae0f7b3dad9e0ad93fbf8905e0a3668ab571e2b45074ad26b4.jpg)

## Jailbreaking as a Reward Misspecification Problem


### Images

![3553da6b0bf5cc014420d98a5eb3b50dca22473e8973589cfad9cb9cf69893a6.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/images/3553da6b0bf5cc014420d98a5eb3b50dca22473e8973589cfad9cb9cf69893a6.jpg)

![3c7284eccbb6c8398f38971aed3747aad8e7a84616396f187427a0ce0ad578ed.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/images/3c7284eccbb6c8398f38971aed3747aad8e7a84616396f187427a0ce0ad578ed.jpg)

![4769b12372893c01a4f50d994a4356bf9176bb5c93c998dec6ddb7c8ab562d74.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/images/4769b12372893c01a4f50d994a4356bf9176bb5c93c998dec6ddb7c8ab562d74.jpg)

![5988fee3f329ead3703ef84dfaf394b394351111acc8a19d05eb37fedb3bbdb6.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/images/5988fee3f329ead3703ef84dfaf394b394351111acc8a19d05eb37fedb3bbdb6.jpg)

![59d95fcc5d666d24911f120a1ef62f38bc106d977c82e65865e90ab4f4d74756.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/images/59d95fcc5d666d24911f120a1ef62f38bc106d977c82e65865e90ab4f4d74756.jpg)

![824cd0bba649cc2f8db50576f5fadfad105891fff7d383f819185f9eeaea9d2c.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/images/824cd0bba649cc2f8db50576f5fadfad105891fff7d383f819185f9eeaea9d2c.jpg)

![aa54d30ba92d7e6a1c0c8fab663fd1badae2be59404e36aaaf1fbb144a85e691.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/images/aa54d30ba92d7e6a1c0c8fab663fd1badae2be59404e36aaaf1fbb144a85e691.jpg)

![f0470651085406fcc6033aff7159e149395bd7881b892ecb7b3e85e09bd2229e.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/images/f0470651085406fcc6033aff7159e149395bd7881b892ecb7b3e85e09bd2229e.jpg)

![f5de07e04f42d1267905160de59e683ae427b03cd0088c0f81afb647bd7b64ff.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/images/f5de07e04f42d1267905160de59e683ae427b03cd0088c0f81afb647bd7b64ff.jpg)

![fa281f319402a4ec221e6252927e310e2fa3cdbb37bd37980be8d336d5dd5960.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/images/fa281f319402a4ec221e6252927e310e2fa3cdbb37bd37980be8d336d5dd5960.jpg)

### Tables

![0cf41e0b039a20e32abf494a650b9f31db640a4cbd1a94b3b8684249944544ff.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/tables/0cf41e0b039a20e32abf494a650b9f31db640a4cbd1a94b3b8684249944544ff.jpg)

![181acafb3f5430826d75a727b1846f87ec6beb93717d43116de900d722fe8c66.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/tables/181acafb3f5430826d75a727b1846f87ec6beb93717d43116de900d722fe8c66.jpg)

![2512de628cbd61a386b51f8b77382253b2b6d7cf490e601bee1fe0731e03f779.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/tables/2512de628cbd61a386b51f8b77382253b2b6d7cf490e601bee1fe0731e03f779.jpg)

![2829ea1cc7b3fcdcd2582c04d2625cad30e153741110acada81be45c18631f9c.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/tables/2829ea1cc7b3fcdcd2582c04d2625cad30e153741110acada81be45c18631f9c.jpg)

![327d6578894a1294ec2020766da3fde46cc8c77682183095366c2fb5878b154b.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/tables/327d6578894a1294ec2020766da3fde46cc8c77682183095366c2fb5878b154b.jpg)

![55d7a65024620b6f0dffefe1019539e863344b3a398b42ecad02357f83a5f2d0.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/tables/55d7a65024620b6f0dffefe1019539e863344b3a398b42ecad02357f83a5f2d0.jpg)

![5a8f2ab691565beeec66a7f88aced2d82decb47bd7720297d6dcda5f88475790.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/tables/5a8f2ab691565beeec66a7f88aced2d82decb47bd7720297d6dcda5f88475790.jpg)

![5afd76f5469b1c2c06f63de2968a5c5ee949400d984d2aa7c47816715b59bbc6.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/tables/5afd76f5469b1c2c06f63de2968a5c5ee949400d984d2aa7c47816715b59bbc6.jpg)

![68eb7e5add835479c543b34c24c90f9f5fc94bf661a5aa61792d4b265910217c.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/tables/68eb7e5add835479c543b34c24c90f9f5fc94bf661a5aa61792d4b265910217c.jpg)

![87947bb86d2436840767ff8ea5f53a1a4a0ab1f765eed61a5c73966d709448d3.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/tables/87947bb86d2436840767ff8ea5f53a1a4a0ab1f765eed61a5c73966d709448d3.jpg)

![aafd0fc0076053220fc01dc88b81856218b1a56f7ee2db3e0a74c4071b4bb85f.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/tables/aafd0fc0076053220fc01dc88b81856218b1a56f7ee2db3e0a74c4071b4bb85f.jpg)

![c69727f5b8bf8861aaac37b3bc8ffc5757d9ab5a232c47d59657800c1f35697d.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/tables/c69727f5b8bf8861aaac37b3bc8ffc5757d9ab5a232c47d59657800c1f35697d.jpg)

![ce830f612a1b1f548489a247f1ed821ebe2b79fab254fa0ef0df1e4a714dc696.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/tables/ce830f612a1b1f548489a247f1ed821ebe2b79fab254fa0ef0df1e4a714dc696.jpg)

![dd08f96d59cadfd453d3f3ae34e317a59d5e4f9cf419d9a3704ed2a6ec365227.jpg](../iclr_results/899_Jailbreaking as a Reward Misspecification Problem/tables/dd08f96d59cadfd453d3f3ae34e317a59d5e4f9cf419d9a3704ed2a6ec365227.jpg)

## Equivariant Neural Functional Networks for Transformers


### Images

![260008edb785c39eb0950645fbc7bcc8e846d7c93755109e031330b3917ca03b.jpg](../iclr_results/900_Equivariant Neural Functional Networks for Transformers/images/260008edb785c39eb0950645fbc7bcc8e846d7c93755109e031330b3917ca03b.jpg)

![ad7981c887d39018d186945fc02c5664bfab32aa4632b2793fbacdf90f33fe9a.jpg](../iclr_results/900_Equivariant Neural Functional Networks for Transformers/images/ad7981c887d39018d186945fc02c5664bfab32aa4632b2793fbacdf90f33fe9a.jpg)

### Tables

![277f486f76f51b980936a6593450743d6f655ecf80f89724462b32d95c951771.jpg](../iclr_results/900_Equivariant Neural Functional Networks for Transformers/tables/277f486f76f51b980936a6593450743d6f655ecf80f89724462b32d95c951771.jpg)

![5caad8bbc12fa46feca0e7030bbb1ae71cb2cf186c84d9c78afb4cbf9fa91fcc.jpg](../iclr_results/900_Equivariant Neural Functional Networks for Transformers/tables/5caad8bbc12fa46feca0e7030bbb1ae71cb2cf186c84d9c78afb4cbf9fa91fcc.jpg)

![7207a1b9a6ea258433f7d4de2562f3ba6a3c5585a41500cd9bb7427a2ac6e886.jpg](../iclr_results/900_Equivariant Neural Functional Networks for Transformers/tables/7207a1b9a6ea258433f7d4de2562f3ba6a3c5585a41500cd9bb7427a2ac6e886.jpg)

![7330bad28fe33f5567ba4d01f1fbe6e34e6e9b352ff4c72ba83666f97496921e.jpg](../iclr_results/900_Equivariant Neural Functional Networks for Transformers/tables/7330bad28fe33f5567ba4d01f1fbe6e34e6e9b352ff4c72ba83666f97496921e.jpg)

![7908e331afa9279daf2b0f58be6f26b9aec90dd400e8870271f82e477069a66d.jpg](../iclr_results/900_Equivariant Neural Functional Networks for Transformers/tables/7908e331afa9279daf2b0f58be6f26b9aec90dd400e8870271f82e477069a66d.jpg)

![9e380e07b67a4fd0881715e91290704d0e5138842073bc781c65004bc3ba9883.jpg](../iclr_results/900_Equivariant Neural Functional Networks for Transformers/tables/9e380e07b67a4fd0881715e91290704d0e5138842073bc781c65004bc3ba9883.jpg)

![a8f23414f3eba26ad786d96035753db13034a35a298a268a1c3c0d02b704d605.jpg](../iclr_results/900_Equivariant Neural Functional Networks for Transformers/tables/a8f23414f3eba26ad786d96035753db13034a35a298a268a1c3c0d02b704d605.jpg)

![c1de0c3db494d125950156e8ea8eeb55a5c0a5d5851113e876cb2edcf2216ec5.jpg](../iclr_results/900_Equivariant Neural Functional Networks for Transformers/tables/c1de0c3db494d125950156e8ea8eeb55a5c0a5d5851113e876cb2edcf2216ec5.jpg)

![d724fc442cd2301fbd72ac7ac4a2c8784d6ef0c60a1fb3bffbc0384b91226113.jpg](../iclr_results/900_Equivariant Neural Functional Networks for Transformers/tables/d724fc442cd2301fbd72ac7ac4a2c8784d6ef0c60a1fb3bffbc0384b91226113.jpg)

## Adaptive Rank Allocation: Speeding Up Modern Transformers with RaNA Adapters


### Images

![02c91febba8e7bd93d1e3f53049fe19cbcf7cdca637bcbcfee65f90da43d6196.jpg](../iclr_results/901_Adaptive Rank Allocation_ Speeding Up Modern Transformers with RaNA Adapters/images/02c91febba8e7bd93d1e3f53049fe19cbcf7cdca637bcbcfee65f90da43d6196.jpg)

![1eb1e5aa20c628251a3f46862001426272079c15af2da491f35ca8fb44483706.jpg](../iclr_results/901_Adaptive Rank Allocation_ Speeding Up Modern Transformers with RaNA Adapters/images/1eb1e5aa20c628251a3f46862001426272079c15af2da491f35ca8fb44483706.jpg)

![224dececeb7f7b73db5461b1f64fa14998fd19ef1a43428bf0fcd161b7112626.jpg](../iclr_results/901_Adaptive Rank Allocation_ Speeding Up Modern Transformers with RaNA Adapters/images/224dececeb7f7b73db5461b1f64fa14998fd19ef1a43428bf0fcd161b7112626.jpg)

![45af17dc7ef657bb5a5068768373cb8e30218f377710617511ad84f3e0e73b83.jpg](../iclr_results/901_Adaptive Rank Allocation_ Speeding Up Modern Transformers with RaNA Adapters/images/45af17dc7ef657bb5a5068768373cb8e30218f377710617511ad84f3e0e73b83.jpg)

![8cbcc65bdce3974a029328fb30d439b1cd3a8d872ea6f8e0d61c607141749acc.jpg](../iclr_results/901_Adaptive Rank Allocation_ Speeding Up Modern Transformers with RaNA Adapters/images/8cbcc65bdce3974a029328fb30d439b1cd3a8d872ea6f8e0d61c607141749acc.jpg)

### Tables

![2004dba380989bdfcc3d6c940d6fcf778e25fcecfb5cb073b6fbe9794cf197c7.jpg](../iclr_results/901_Adaptive Rank Allocation_ Speeding Up Modern Transformers with RaNA Adapters/tables/2004dba380989bdfcc3d6c940d6fcf778e25fcecfb5cb073b6fbe9794cf197c7.jpg)

![2c1e38ae989c9a01738eb23f5c1bdd690103736938d900431c1a656df9c3e607.jpg](../iclr_results/901_Adaptive Rank Allocation_ Speeding Up Modern Transformers with RaNA Adapters/tables/2c1e38ae989c9a01738eb23f5c1bdd690103736938d900431c1a656df9c3e607.jpg)

![9b4f30e69e6502f24b88d8f8070fa76beedce19c35208cc576e698927a81394d.jpg](../iclr_results/901_Adaptive Rank Allocation_ Speeding Up Modern Transformers with RaNA Adapters/tables/9b4f30e69e6502f24b88d8f8070fa76beedce19c35208cc576e698927a81394d.jpg)

![dab4f899931de50668c4c8c8a5a6e28394aeca2e8d9eee46823ab3131edd3623.jpg](../iclr_results/901_Adaptive Rank Allocation_ Speeding Up Modern Transformers with RaNA Adapters/tables/dab4f899931de50668c4c8c8a5a6e28394aeca2e8d9eee46823ab3131edd3623.jpg)

## Exploring Prosocial Irrationality for LLM Agents: A Social Cognition View


### Images

![0002b094c0813ab35d64dd581e2b7e5fd56e5c09318c2a92112dd334261a25b5.jpg](../iclr_results/902_Exploring Prosocial Irrationality for LLM Agents_ A Social Cognition View/images/0002b094c0813ab35d64dd581e2b7e5fd56e5c09318c2a92112dd334261a25b5.jpg)

![1405da2884a86af072aeee8af33f303123669fd1cab6f447b6ccfa7aa8f0e14c.jpg](../iclr_results/902_Exploring Prosocial Irrationality for LLM Agents_ A Social Cognition View/images/1405da2884a86af072aeee8af33f303123669fd1cab6f447b6ccfa7aa8f0e14c.jpg)

![3ee0c8de6495acfc6f985399c95d8f3f1835a63af92029e509bf241c241434e9.jpg](../iclr_results/902_Exploring Prosocial Irrationality for LLM Agents_ A Social Cognition View/images/3ee0c8de6495acfc6f985399c95d8f3f1835a63af92029e509bf241c241434e9.jpg)

![5db7ff8f1403679e3c7b2fa7792b40101b41973987366029ae2726e849efc82e.jpg](../iclr_results/902_Exploring Prosocial Irrationality for LLM Agents_ A Social Cognition View/images/5db7ff8f1403679e3c7b2fa7792b40101b41973987366029ae2726e849efc82e.jpg)

![69dc0c88665fc0e3324fe6c0e24d57a9630a42d3d24cc213a3e500752144f54c.jpg](../iclr_results/902_Exploring Prosocial Irrationality for LLM Agents_ A Social Cognition View/images/69dc0c88665fc0e3324fe6c0e24d57a9630a42d3d24cc213a3e500752144f54c.jpg)

![7629fe2f033d3ba9113f72d2c06d4abd6a76d154ec80d10be72727a6fdb3df49.jpg](../iclr_results/902_Exploring Prosocial Irrationality for LLM Agents_ A Social Cognition View/images/7629fe2f033d3ba9113f72d2c06d4abd6a76d154ec80d10be72727a6fdb3df49.jpg)

![a9fe203bdd6de49ed6ec7e79a1fa4ba437122ae81863c12abb7bd2fa2aef0278.jpg](../iclr_results/902_Exploring Prosocial Irrationality for LLM Agents_ A Social Cognition View/images/a9fe203bdd6de49ed6ec7e79a1fa4ba437122ae81863c12abb7bd2fa2aef0278.jpg)

![d0a6def5382510cb98ac3126e72f2fe5387ea8ac143e89b30b4befdd03f0a668.jpg](../iclr_results/902_Exploring Prosocial Irrationality for LLM Agents_ A Social Cognition View/images/d0a6def5382510cb98ac3126e72f2fe5387ea8ac143e89b30b4befdd03f0a668.jpg)

![e724bf0451655f8462c177c3ae3a8acafe62cad642ebf93592b8fb3309ba861c.jpg](../iclr_results/902_Exploring Prosocial Irrationality for LLM Agents_ A Social Cognition View/images/e724bf0451655f8462c177c3ae3a8acafe62cad642ebf93592b8fb3309ba861c.jpg)

![e79635e1eb67094140a3e3fc09f164b53c0280fd1a8cd45b3b33f09f02363710.jpg](../iclr_results/902_Exploring Prosocial Irrationality for LLM Agents_ A Social Cognition View/images/e79635e1eb67094140a3e3fc09f164b53c0280fd1a8cd45b3b33f09f02363710.jpg)

### Tables

![2446cb7febed008d1576dfe3d70fce8159ecd1ae507a8f6f81ad1948b9c687fc.jpg](../iclr_results/902_Exploring Prosocial Irrationality for LLM Agents_ A Social Cognition View/tables/2446cb7febed008d1576dfe3d70fce8159ecd1ae507a8f6f81ad1948b9c687fc.jpg)

![27d379678afda37c5e7e4b3c18fba017e392bb0df0e4ee0fdcf2b78a37d29e57.jpg](../iclr_results/902_Exploring Prosocial Irrationality for LLM Agents_ A Social Cognition View/tables/27d379678afda37c5e7e4b3c18fba017e392bb0df0e4ee0fdcf2b78a37d29e57.jpg)

![333e7633761bcedb4662c80b173fa285110bd43092869d652c370610d078cfe8.jpg](../iclr_results/902_Exploring Prosocial Irrationality for LLM Agents_ A Social Cognition View/tables/333e7633761bcedb4662c80b173fa285110bd43092869d652c370610d078cfe8.jpg)

![3f6f245f88b1865d21ff69d09f1f29e91916261408e54f99fc4a95b5450c8932.jpg](../iclr_results/902_Exploring Prosocial Irrationality for LLM Agents_ A Social Cognition View/tables/3f6f245f88b1865d21ff69d09f1f29e91916261408e54f99fc4a95b5450c8932.jpg)

![a42f6d586c849297db02035eb9f77c2965a884a959372f815fe84be20574dba2.jpg](../iclr_results/902_Exploring Prosocial Irrationality for LLM Agents_ A Social Cognition View/tables/a42f6d586c849297db02035eb9f77c2965a884a959372f815fe84be20574dba2.jpg)

![a650ebb67c21e6ce7269d89b798655e621002f252a8fa954721d52992ce7920c.jpg](../iclr_results/902_Exploring Prosocial Irrationality for LLM Agents_ A Social Cognition View/tables/a650ebb67c21e6ce7269d89b798655e621002f252a8fa954721d52992ce7920c.jpg)

![b0cfd4e351762e0f6e987862e7f43a879327fab071bf349ca2ff878213b9590a.jpg](../iclr_results/902_Exploring Prosocial Irrationality for LLM Agents_ A Social Cognition View/tables/b0cfd4e351762e0f6e987862e7f43a879327fab071bf349ca2ff878213b9590a.jpg)

![c6208ef4ea77fe70059d07d649de110caabef6c3e5f8288c737ed2ed7fe66ddf.jpg](../iclr_results/902_Exploring Prosocial Irrationality for LLM Agents_ A Social Cognition View/tables/c6208ef4ea77fe70059d07d649de110caabef6c3e5f8288c737ed2ed7fe66ddf.jpg)

## Deriving Causal Order from Single-Variable Interventions: Guarantees & Algorithm


### Images

![146887f6cb1e27a56c13dc2650d0cc0458d0349baa298367ac267db1caac4937.jpg](../iclr_results/903_Deriving Causal Order from Single-Variable Interventions_ Guarantees & Algorithm/images/146887f6cb1e27a56c13dc2650d0cc0458d0349baa298367ac267db1caac4937.jpg)

![2eb07c525dd17c1a5f19e7b62b890bdd0c887d69d3ea86fb24a01d84c0f9387b.jpg](../iclr_results/903_Deriving Causal Order from Single-Variable Interventions_ Guarantees & Algorithm/images/2eb07c525dd17c1a5f19e7b62b890bdd0c887d69d3ea86fb24a01d84c0f9387b.jpg)

![395a74dd53235d6f8bfb64fd4433a44fb871923d89f0deaea13446033a393165.jpg](../iclr_results/903_Deriving Causal Order from Single-Variable Interventions_ Guarantees & Algorithm/images/395a74dd53235d6f8bfb64fd4433a44fb871923d89f0deaea13446033a393165.jpg)

![428e82aa67d3e75213ec006a6fd06ee428889dcdf8441ef041749c59329d48d4.jpg](../iclr_results/903_Deriving Causal Order from Single-Variable Interventions_ Guarantees & Algorithm/images/428e82aa67d3e75213ec006a6fd06ee428889dcdf8441ef041749c59329d48d4.jpg)

![48e1e09345ac57a8bbc1e13e6e4d6c4fa09856b145ebaa758102b90e083fe50b.jpg](../iclr_results/903_Deriving Causal Order from Single-Variable Interventions_ Guarantees & Algorithm/images/48e1e09345ac57a8bbc1e13e6e4d6c4fa09856b145ebaa758102b90e083fe50b.jpg)

![5f8e1b302a986ac7d0251b85b644c40ecf386585877ec2597a0f369a3e54f697.jpg](../iclr_results/903_Deriving Causal Order from Single-Variable Interventions_ Guarantees & Algorithm/images/5f8e1b302a986ac7d0251b85b644c40ecf386585877ec2597a0f369a3e54f697.jpg)

![9392ca411e8f00fd2da447ba02b1e78002856a4252e238e0bed3a3d8ec4b0c9b.jpg](../iclr_results/903_Deriving Causal Order from Single-Variable Interventions_ Guarantees & Algorithm/images/9392ca411e8f00fd2da447ba02b1e78002856a4252e238e0bed3a3d8ec4b0c9b.jpg)

![9749e0a4a8d2531be5044be527d1971c7ae65e57ebdb771632f005c93a488180.jpg](../iclr_results/903_Deriving Causal Order from Single-Variable Interventions_ Guarantees & Algorithm/images/9749e0a4a8d2531be5044be527d1971c7ae65e57ebdb771632f005c93a488180.jpg)

![ada79facbf7250b6a17644a5d4813823c108a9dc4999696c2f0818bf07344f75.jpg](../iclr_results/903_Deriving Causal Order from Single-Variable Interventions_ Guarantees & Algorithm/images/ada79facbf7250b6a17644a5d4813823c108a9dc4999696c2f0818bf07344f75.jpg)

![da41797106b47aa4c9da8e9fc3dc643ca40cf6e125e1c2d6699f19a31a21cd88.jpg](../iclr_results/903_Deriving Causal Order from Single-Variable Interventions_ Guarantees & Algorithm/images/da41797106b47aa4c9da8e9fc3dc643ca40cf6e125e1c2d6699f19a31a21cd88.jpg)

![dee0275c990159912d06e3142bd310b5eae8c595d7925c7a5ef1910ef4c5ca61.jpg](../iclr_results/903_Deriving Causal Order from Single-Variable Interventions_ Guarantees & Algorithm/images/dee0275c990159912d06e3142bd310b5eae8c595d7925c7a5ef1910ef4c5ca61.jpg)

### Tables

![195ddde4bab18d0c3696fe8af8209f6f41bee2e965ca5154226d133ed7e1c768.jpg](../iclr_results/903_Deriving Causal Order from Single-Variable Interventions_ Guarantees & Algorithm/tables/195ddde4bab18d0c3696fe8af8209f6f41bee2e965ca5154226d133ed7e1c768.jpg)

![e7a3435a2ca91982d5fbe88f853a333a6f07d91c56160b490eb91f804db20490.jpg](../iclr_results/903_Deriving Causal Order from Single-Variable Interventions_ Guarantees & Algorithm/tables/e7a3435a2ca91982d5fbe88f853a333a6f07d91c56160b490eb91f804db20490.jpg)

## SeedLM: Compressing LLM Weights into Seeds of Pseudo-Random Generators


### Images

![5018788694cec86555fc27109decb0032edbe906b33da9c9068ca6c9bbbb1f4a.jpg](../iclr_results/904_SeedLM_ Compressing LLM Weights into Seeds of Pseudo-Random Generators/images/5018788694cec86555fc27109decb0032edbe906b33da9c9068ca6c9bbbb1f4a.jpg)

![77d3f3cbaf662f08156a1028d386f8a44190d3e84a2cf37d3c93b3e058a25678.jpg](../iclr_results/904_SeedLM_ Compressing LLM Weights into Seeds of Pseudo-Random Generators/images/77d3f3cbaf662f08156a1028d386f8a44190d3e84a2cf37d3c93b3e058a25678.jpg)

![80ce55a2b91beca2fccfa7b6d060b903c7292bdfb2dc423da7735fcae59ff18d.jpg](../iclr_results/904_SeedLM_ Compressing LLM Weights into Seeds of Pseudo-Random Generators/images/80ce55a2b91beca2fccfa7b6d060b903c7292bdfb2dc423da7735fcae59ff18d.jpg)

![ae1c406a423a9cdd72026bfe716808a52a53d17c94c9aaf46250deabefd87368.jpg](../iclr_results/904_SeedLM_ Compressing LLM Weights into Seeds of Pseudo-Random Generators/images/ae1c406a423a9cdd72026bfe716808a52a53d17c94c9aaf46250deabefd87368.jpg)

### Tables

![07ca0972ff1801e358a5bacb939bbcaa1087c929634dfa8f0093b94f9c3fa313.jpg](../iclr_results/904_SeedLM_ Compressing LLM Weights into Seeds of Pseudo-Random Generators/tables/07ca0972ff1801e358a5bacb939bbcaa1087c929634dfa8f0093b94f9c3fa313.jpg)

![3adaad666a43f3238f7d4d45613e347234b8722ef2ff6609cfd5f8b75525e3f1.jpg](../iclr_results/904_SeedLM_ Compressing LLM Weights into Seeds of Pseudo-Random Generators/tables/3adaad666a43f3238f7d4d45613e347234b8722ef2ff6609cfd5f8b75525e3f1.jpg)

![525ac41618f0e1f0d44ee4993a5938250269c25b2b42a491481fbbbae00f8622.jpg](../iclr_results/904_SeedLM_ Compressing LLM Weights into Seeds of Pseudo-Random Generators/tables/525ac41618f0e1f0d44ee4993a5938250269c25b2b42a491481fbbbae00f8622.jpg)

![718a54dc5be33a5dd1f714e5db6d504cd5a9256780dcb6f4d69742e1d846440f.jpg](../iclr_results/904_SeedLM_ Compressing LLM Weights into Seeds of Pseudo-Random Generators/tables/718a54dc5be33a5dd1f714e5db6d504cd5a9256780dcb6f4d69742e1d846440f.jpg)

![a4ddc428c3743b403790b1f6150d799403e1b623af3e535cde4b3d55893151c3.jpg](../iclr_results/904_SeedLM_ Compressing LLM Weights into Seeds of Pseudo-Random Generators/tables/a4ddc428c3743b403790b1f6150d799403e1b623af3e535cde4b3d55893151c3.jpg)

![accdbec49e6313a351147bd812ee3f7f407619c4f4353973510b906131408acd.jpg](../iclr_results/904_SeedLM_ Compressing LLM Weights into Seeds of Pseudo-Random Generators/tables/accdbec49e6313a351147bd812ee3f7f407619c4f4353973510b906131408acd.jpg)

![ca2fd70934e5548bf6031900e97df718f4274e7d712057699f4d7b0ccd818d2a.jpg](../iclr_results/904_SeedLM_ Compressing LLM Weights into Seeds of Pseudo-Random Generators/tables/ca2fd70934e5548bf6031900e97df718f4274e7d712057699f4d7b0ccd818d2a.jpg)

## Context-aware Dynamic Pruning for Speech Foundation Models


### Images

![0e32ef701c68a976fd93a6abfd97aa95b73697edf9a50adfe5bc1367e0ad88a4.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/0e32ef701c68a976fd93a6abfd97aa95b73697edf9a50adfe5bc1367e0ad88a4.jpg)

![0eae9677008ca774f4840033dead4a4d5ffef4677e57ed044b2717e120197f0f.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/0eae9677008ca774f4840033dead4a4d5ffef4677e57ed044b2717e120197f0f.jpg)

![1a035f51fb5f21929da22e962af0e7f4a123d62492082ca037ccf1609d026ca1.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/1a035f51fb5f21929da22e962af0e7f4a123d62492082ca037ccf1609d026ca1.jpg)

![2127535ae7c46908b290a3f00a211974a1fe1aa53b5f05e6970b70a79e3f1ca2.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/2127535ae7c46908b290a3f00a211974a1fe1aa53b5f05e6970b70a79e3f1ca2.jpg)

![2948db1f1a7217d8e1403ae6098b06e5d42060822bf3340e9133f7e8fef3cf58.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/2948db1f1a7217d8e1403ae6098b06e5d42060822bf3340e9133f7e8fef3cf58.jpg)

![31316074ddb965296f1d87d6c2041061434d8184be400ad1707f85da1462e62b.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/31316074ddb965296f1d87d6c2041061434d8184be400ad1707f85da1462e62b.jpg)

![336749d156b2fec6bd4f064e0204d933554774d72224b615608eadddd87e236f.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/336749d156b2fec6bd4f064e0204d933554774d72224b615608eadddd87e236f.jpg)

![344d6c42c72c06aea28e3d192d97bc8db274d05dc84e446544097d8c4cca889a.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/344d6c42c72c06aea28e3d192d97bc8db274d05dc84e446544097d8c4cca889a.jpg)

![36704db96c97c4fc9d87dc0a6cbe34433bc8e82423224367ad973c3cf31251e8.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/36704db96c97c4fc9d87dc0a6cbe34433bc8e82423224367ad973c3cf31251e8.jpg)

![46907e9b11780651b4999eae0c4e675811cabbef701633341f1f596d9ac43abc.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/46907e9b11780651b4999eae0c4e675811cabbef701633341f1f596d9ac43abc.jpg)

![4815ffd4ea5d138231f87656b6075864a735a9f5b175539897dc122317d5c7ab.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/4815ffd4ea5d138231f87656b6075864a735a9f5b175539897dc122317d5c7ab.jpg)

![4b4ebd256ee6c90311c81a2aab1fdc75234faa7546d6ce0c8eede00102909fc4.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/4b4ebd256ee6c90311c81a2aab1fdc75234faa7546d6ce0c8eede00102909fc4.jpg)

![4c046150ac0ec1572088f3d0896f6389b204debb93a09eb330309049ad70049d.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/4c046150ac0ec1572088f3d0896f6389b204debb93a09eb330309049ad70049d.jpg)

![523b093f5f2667a003145b791c437bffbb588eb4e301e80f562342ca7622e769.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/523b093f5f2667a003145b791c437bffbb588eb4e301e80f562342ca7622e769.jpg)

![544413c6f7e3d178baf7137872f4c1f39f05aa013b1e898de7c35d8d873c2283.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/544413c6f7e3d178baf7137872f4c1f39f05aa013b1e898de7c35d8d873c2283.jpg)

![6621859e2b196f679b03b7bf7336d16e8aff1d29b3c5dbc83a24bc324658e837.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/6621859e2b196f679b03b7bf7336d16e8aff1d29b3c5dbc83a24bc324658e837.jpg)

![6e9fb64fca648df7e59f173e9e80135c50ff78323243667ed6aebfb693924d24.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/6e9fb64fca648df7e59f173e9e80135c50ff78323243667ed6aebfb693924d24.jpg)

![7697378fbf6c66dff50fa3da42bd4cba4cd997ae5354852d0b036587305b2514.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/7697378fbf6c66dff50fa3da42bd4cba4cd997ae5354852d0b036587305b2514.jpg)

![778f70e846d3b573a8ada280c47d52161eb9707077c0b2d7fda6e4c18e4af4ed.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/778f70e846d3b573a8ada280c47d52161eb9707077c0b2d7fda6e4c18e4af4ed.jpg)

![7c990db6b5d67ce7eb0e1727f948b1c5e7eb5980e2a9df82a167cb56bd17c093.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/7c990db6b5d67ce7eb0e1727f948b1c5e7eb5980e2a9df82a167cb56bd17c093.jpg)

![7d2dd714a2142645155942dcb1c588f910f0582ebb7bcd0bfb12fac0caa8443b.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/7d2dd714a2142645155942dcb1c588f910f0582ebb7bcd0bfb12fac0caa8443b.jpg)

![88e7e7d720c6c1e31ddf5a6319b8b27029838561ddfd384eb290f16cc6d524db.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/88e7e7d720c6c1e31ddf5a6319b8b27029838561ddfd384eb290f16cc6d524db.jpg)

![8c8107a94e58607cd200662950de142aeb7b634c593c8fc31d2f125f26fcef60.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/8c8107a94e58607cd200662950de142aeb7b634c593c8fc31d2f125f26fcef60.jpg)

![abf6beb71740c1c63e933642025564557bac4cfeeefaedd23174c744286c8fff.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/abf6beb71740c1c63e933642025564557bac4cfeeefaedd23174c744286c8fff.jpg)

![b380c58f559fb5eea73fd34f7618f86348f8de444cb1e04519eab00cfd715c0f.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/b380c58f559fb5eea73fd34f7618f86348f8de444cb1e04519eab00cfd715c0f.jpg)

![b38514c55816141641db8624e43a6a43a1175f0b2c1db9cebc1b04c920927360.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/b38514c55816141641db8624e43a6a43a1175f0b2c1db9cebc1b04c920927360.jpg)

![b3919873d77e3ac03772113f9cdb9e765084e81fe3fd821b4e5000aeb758a1dc.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/b3919873d77e3ac03772113f9cdb9e765084e81fe3fd821b4e5000aeb758a1dc.jpg)

![b6415cf8baccdc53e2d43c84e8e60d2c720040048146fa2f003710a55d35b0f8.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/b6415cf8baccdc53e2d43c84e8e60d2c720040048146fa2f003710a55d35b0f8.jpg)

![b88c4e5c634b036138b9f5d6a26ba998cb85b6d2095999f26f714308abe4e160.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/b88c4e5c634b036138b9f5d6a26ba998cb85b6d2095999f26f714308abe4e160.jpg)

![bb3db93be2d640a77767a01371e92ef1c11ff9060be9cda01232dae6f8bf3705.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/bb3db93be2d640a77767a01371e92ef1c11ff9060be9cda01232dae6f8bf3705.jpg)

![bb7b768ceb956bc3b9b9b9997e60005c694937a6c21d32c57ea60e9c296680c2.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/bb7b768ceb956bc3b9b9b9997e60005c694937a6c21d32c57ea60e9c296680c2.jpg)

![bd4871c01d89a2eb6288f4f2cdf27640a6b121381148094d6f7fc412707317f9.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/bd4871c01d89a2eb6288f4f2cdf27640a6b121381148094d6f7fc412707317f9.jpg)

![c3631b6039d9ee29a1f600f1e3ab7ab33a09d8612d92afdbf16ea39b08cb2f49.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/c3631b6039d9ee29a1f600f1e3ab7ab33a09d8612d92afdbf16ea39b08cb2f49.jpg)

![c36fb58a68e947c6b90ea5492df27cc47ab942acf4d6e30b15b5df2bcf4c307b.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/c36fb58a68e947c6b90ea5492df27cc47ab942acf4d6e30b15b5df2bcf4c307b.jpg)

![cc21aa2a6dcea1da198c40c3f14391bfa72ae9276c8121355591b9ba04272419.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/cc21aa2a6dcea1da198c40c3f14391bfa72ae9276c8121355591b9ba04272419.jpg)

![ce6e77ea04197df6b05fcc781178bddc52a5db610bc3eb7a92bd17092f72aeaf.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/ce6e77ea04197df6b05fcc781178bddc52a5db610bc3eb7a92bd17092f72aeaf.jpg)

![d20be5c780d066a83fd53d32db92ddfa1c227bedb5965febf83014879d8b76a2.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/d20be5c780d066a83fd53d32db92ddfa1c227bedb5965febf83014879d8b76a2.jpg)

![d518ff55148de70df8edc0008a7bc9d876eca2aa94a222fdb4180e9c124c0016.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/d518ff55148de70df8edc0008a7bc9d876eca2aa94a222fdb4180e9c124c0016.jpg)

![d52cb0c354250328c79e0ae7ea3cd73ec76fb1ba586052076c56c33e9f88b938.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/d52cb0c354250328c79e0ae7ea3cd73ec76fb1ba586052076c56c33e9f88b938.jpg)

![d5f2b05e4ca21e7d50d5040dfe1028cd506780326e70afff77a8349540b8045b.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/d5f2b05e4ca21e7d50d5040dfe1028cd506780326e70afff77a8349540b8045b.jpg)

![e049e459a03e0372928c5b9a6bff2c074587238f30bd1e3ef00181f2652e3c63.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/e049e459a03e0372928c5b9a6bff2c074587238f30bd1e3ef00181f2652e3c63.jpg)

![e4199f7341a90867feaea05c9861dae0b6a434bf261a5745011bd7fa2ef2155a.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/e4199f7341a90867feaea05c9861dae0b6a434bf261a5745011bd7fa2ef2155a.jpg)

![e49f0a89dec70a317d05205625d9b85b48e076621c8b82f1bf32c0f28403036e.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/e49f0a89dec70a317d05205625d9b85b48e076621c8b82f1bf32c0f28403036e.jpg)

![ebc1875b25db8fc963c8d42741a2cc8f7486db6ccce29c608dbc95f7aa207597.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/ebc1875b25db8fc963c8d42741a2cc8f7486db6ccce29c608dbc95f7aa207597.jpg)

![f774d4a4755c8e7e3dc9e14e289323d8b709941675547acb363a3075ac1f1702.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/f774d4a4755c8e7e3dc9e14e289323d8b709941675547acb363a3075ac1f1702.jpg)

![fb7df9b52a2cb1bbc5f11fd3623e496f25532de5fc96df00ed0ad86d39dc5ecd.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/fb7df9b52a2cb1bbc5f11fd3623e496f25532de5fc96df00ed0ad86d39dc5ecd.jpg)

![fcb92d8c9c8a77ed5096be5fcec50337d69372a9499c989751e8ccc058346d9f.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/images/fcb92d8c9c8a77ed5096be5fcec50337d69372a9499c989751e8ccc058346d9f.jpg)

### Tables

![24772debfad6706bd231533ab7d2a600f0869200c7a6cb79a876ce2826e8deee.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/tables/24772debfad6706bd231533ab7d2a600f0869200c7a6cb79a876ce2826e8deee.jpg)

![53db067375608e4353e56dc983f6102d53fb77b5664d7854824eae305a7bf448.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/tables/53db067375608e4353e56dc983f6102d53fb77b5664d7854824eae305a7bf448.jpg)

![703b741918538bbb5fc7e371257a6d91175b9b17cf332cb397bbf09a7d7fa7cc.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/tables/703b741918538bbb5fc7e371257a6d91175b9b17cf332cb397bbf09a7d7fa7cc.jpg)

![a22ba3ac23f154d0944c5c3863d3c712603c9c08914b78ec81fa7d8c29bfee10.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/tables/a22ba3ac23f154d0944c5c3863d3c712603c9c08914b78ec81fa7d8c29bfee10.jpg)

![a32531e7d2b858cfc5bd3f624b11c774390a1ae282f41ef84b26894abd3f4c0d.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/tables/a32531e7d2b858cfc5bd3f624b11c774390a1ae282f41ef84b26894abd3f4c0d.jpg)

![a8590a221e082c7e0e99a99ebb3b9705d3195c092a13b9e7c2cac1ca500d22ae.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/tables/a8590a221e082c7e0e99a99ebb3b9705d3195c092a13b9e7c2cac1ca500d22ae.jpg)

![b1341b99b4ee9906b2af27e4dcf7ada3cacb9e70ddce62b57963f0dce6f08411.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/tables/b1341b99b4ee9906b2af27e4dcf7ada3cacb9e70ddce62b57963f0dce6f08411.jpg)

![b1fd955cf380c4cc44008e52d7f62562162a3d2f0be32aaa6c8e2921e2ed3003.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/tables/b1fd955cf380c4cc44008e52d7f62562162a3d2f0be32aaa6c8e2921e2ed3003.jpg)

![b9ca877739384a08293b6df0f7a9a0d236e4316005a7677854e55bd0684f13c6.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/tables/b9ca877739384a08293b6df0f7a9a0d236e4316005a7677854e55bd0684f13c6.jpg)

![d2a177271b99029607ae7e2bcf40283e580a33ee81f3f06e2696183f887616f1.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/tables/d2a177271b99029607ae7e2bcf40283e580a33ee81f3f06e2696183f887616f1.jpg)

![d4cf5f170944029be5712cfacbd51da9257c6977332b3100e1450673e16286e8.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/tables/d4cf5f170944029be5712cfacbd51da9257c6977332b3100e1450673e16286e8.jpg)

![f33934e94324250b26d09e4761d48dd2561e8f1982136b47870c0743345460a2.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/tables/f33934e94324250b26d09e4761d48dd2561e8f1982136b47870c0743345460a2.jpg)

![fd3a7fb244d06df8914f14c22bfebe9817e98f612e5b5176639c969b5e992832.jpg](../iclr_results/905_Context-aware Dynamic Pruning for Speech Foundation Models/tables/fd3a7fb244d06df8914f14c22bfebe9817e98f612e5b5176639c969b5e992832.jpg)

## On the Performance Analysis of Momentum Method: A Frequency Domain Perspective


### Images

![004431fc590e46c1a359abba8633c6ed742e58a831838fbc0b1a4504e3c78ebf.jpg](../iclr_results/906_On the Performance Analysis of Momentum Method_ A Frequency Domain Perspective/images/004431fc590e46c1a359abba8633c6ed742e58a831838fbc0b1a4504e3c78ebf.jpg)

![0c28b3d9e389f399cdff1155a27700648395278a0bd4e4bd9e88f604c7634b53.jpg](../iclr_results/906_On the Performance Analysis of Momentum Method_ A Frequency Domain Perspective/images/0c28b3d9e389f399cdff1155a27700648395278a0bd4e4bd9e88f604c7634b53.jpg)

![18f2b11382a0d3a378ee777856edbc6fd7445f958b28b35004453867187a1799.jpg](../iclr_results/906_On the Performance Analysis of Momentum Method_ A Frequency Domain Perspective/images/18f2b11382a0d3a378ee777856edbc6fd7445f958b28b35004453867187a1799.jpg)

![22b22b333417d45eb64ac606e2a3dcf576207f883a9878c520d196a71cf422d0.jpg](../iclr_results/906_On the Performance Analysis of Momentum Method_ A Frequency Domain Perspective/images/22b22b333417d45eb64ac606e2a3dcf576207f883a9878c520d196a71cf422d0.jpg)

![59be2405d2432edae7fa1fd31e14d4f8e0c73296dff5f60ae21b45e9ff04822c.jpg](../iclr_results/906_On the Performance Analysis of Momentum Method_ A Frequency Domain Perspective/images/59be2405d2432edae7fa1fd31e14d4f8e0c73296dff5f60ae21b45e9ff04822c.jpg)

![5adba4e3e2a7a561a7b1f2108d2a433d5d2029b04acd215f96408ff237f1cf3d.jpg](../iclr_results/906_On the Performance Analysis of Momentum Method_ A Frequency Domain Perspective/images/5adba4e3e2a7a561a7b1f2108d2a433d5d2029b04acd215f96408ff237f1cf3d.jpg)

![67d62e657250a191d153ca9da7de80f7373f372e4cc53ef9edac5fb7f5e41280.jpg](../iclr_results/906_On the Performance Analysis of Momentum Method_ A Frequency Domain Perspective/images/67d62e657250a191d153ca9da7de80f7373f372e4cc53ef9edac5fb7f5e41280.jpg)

![810c23a7d3ee347cdbc569b25db11bfdedae940df3fb37c782057bea34c5e59a.jpg](../iclr_results/906_On the Performance Analysis of Momentum Method_ A Frequency Domain Perspective/images/810c23a7d3ee347cdbc569b25db11bfdedae940df3fb37c782057bea34c5e59a.jpg)

![b21ea03caceff7af5e7d449b1cdabd5c073d80d38950f14b9b45f3f5adb20e54.jpg](../iclr_results/906_On the Performance Analysis of Momentum Method_ A Frequency Domain Perspective/images/b21ea03caceff7af5e7d449b1cdabd5c073d80d38950f14b9b45f3f5adb20e54.jpg)

![be30d8bca32a64d287cfc04313606a4f309c4c93af3cca26a06273003d5bc560.jpg](../iclr_results/906_On the Performance Analysis of Momentum Method_ A Frequency Domain Perspective/images/be30d8bca32a64d287cfc04313606a4f309c4c93af3cca26a06273003d5bc560.jpg)

![cf3d307a2d9394b64e6abe6d8dd964b609efce5c4a66b63162b2ffeee7161f69.jpg](../iclr_results/906_On the Performance Analysis of Momentum Method_ A Frequency Domain Perspective/images/cf3d307a2d9394b64e6abe6d8dd964b609efce5c4a66b63162b2ffeee7161f69.jpg)

![e37b118c0623e3158ee0ad054416d9573b45c052b536a40e9f0aa31a9151029e.jpg](../iclr_results/906_On the Performance Analysis of Momentum Method_ A Frequency Domain Perspective/images/e37b118c0623e3158ee0ad054416d9573b45c052b536a40e9f0aa31a9151029e.jpg)

![f28e2fa9758115a9f4ee4229a5221279c2f7432e158d3025cefffc4126206ad1.jpg](../iclr_results/906_On the Performance Analysis of Momentum Method_ A Frequency Domain Perspective/images/f28e2fa9758115a9f4ee4229a5221279c2f7432e158d3025cefffc4126206ad1.jpg)

![fa40423d3723e13157b7dbf455448e7d107225c1aa57ae0d54937bc04f057507.jpg](../iclr_results/906_On the Performance Analysis of Momentum Method_ A Frequency Domain Perspective/images/fa40423d3723e13157b7dbf455448e7d107225c1aa57ae0d54937bc04f057507.jpg)

### Tables

![01e25e14b9a391c9576485dbb7910ea85f2bb532d561e419379091f749058a8f.jpg](../iclr_results/906_On the Performance Analysis of Momentum Method_ A Frequency Domain Perspective/tables/01e25e14b9a391c9576485dbb7910ea85f2bb532d561e419379091f749058a8f.jpg)

![041dc0d88d55a71f3d2f575ed5caed4b2d0d48efbb4f340b3e8f0967b3b703e0.jpg](../iclr_results/906_On the Performance Analysis of Momentum Method_ A Frequency Domain Perspective/tables/041dc0d88d55a71f3d2f575ed5caed4b2d0d48efbb4f340b3e8f0967b3b703e0.jpg)

![19ad8896572fb595e3e2eed7e6968f3a3d9909db99d262faebffb04ffbb232d4.jpg](../iclr_results/906_On the Performance Analysis of Momentum Method_ A Frequency Domain Perspective/tables/19ad8896572fb595e3e2eed7e6968f3a3d9909db99d262faebffb04ffbb232d4.jpg)

![427667f47e79e7b14735ea0a62549b4de05288e41db145dd3e85406a18a326f8.jpg](../iclr_results/906_On the Performance Analysis of Momentum Method_ A Frequency Domain Perspective/tables/427667f47e79e7b14735ea0a62549b4de05288e41db145dd3e85406a18a326f8.jpg)

![a91a1c4e3e84f955b4c589262e1d31a753c4f5c90a115ec309984cba874ad68c.jpg](../iclr_results/906_On the Performance Analysis of Momentum Method_ A Frequency Domain Perspective/tables/a91a1c4e3e84f955b4c589262e1d31a753c4f5c90a115ec309984cba874ad68c.jpg)

![bd77678963af2c24344bf04248b3832a9952074f67517e2dca9bc77083e90123.jpg](../iclr_results/906_On the Performance Analysis of Momentum Method_ A Frequency Domain Perspective/tables/bd77678963af2c24344bf04248b3832a9952074f67517e2dca9bc77083e90123.jpg)

![e0ca10f9b2d819d5767f098a88927fd48b321b2c091414574044f108d3d9d6e3.jpg](../iclr_results/906_On the Performance Analysis of Momentum Method_ A Frequency Domain Perspective/tables/e0ca10f9b2d819d5767f098a88927fd48b321b2c091414574044f108d3d9d6e3.jpg)

![e16cc435361c271e93c55eeb00486105b29f5b5f808d8ff9318b1665a5856984.jpg](../iclr_results/906_On the Performance Analysis of Momentum Method_ A Frequency Domain Perspective/tables/e16cc435361c271e93c55eeb00486105b29f5b5f808d8ff9318b1665a5856984.jpg)

## PEARL: Towards Permutation-Resilient LLMs


### Images

![16b081cd8f57f34a15abeb8dde86e9e9f738d591c3ed355717cbfec18f045bea.jpg](../iclr_results/907_PEARL_ Towards Permutation-Resilient LLMs/images/16b081cd8f57f34a15abeb8dde86e9e9f738d591c3ed355717cbfec18f045bea.jpg)

![20e964d4d57c9abe5f2d051c6ae032d11b601d1fecaa9469e881a277cf980a51.jpg](../iclr_results/907_PEARL_ Towards Permutation-Resilient LLMs/images/20e964d4d57c9abe5f2d051c6ae032d11b601d1fecaa9469e881a277cf980a51.jpg)

![3cc4396245f8ca8b2e94ceade796e00ae74e9c8d04942794033651b729d42603.jpg](../iclr_results/907_PEARL_ Towards Permutation-Resilient LLMs/images/3cc4396245f8ca8b2e94ceade796e00ae74e9c8d04942794033651b729d42603.jpg)

![4aa650a7ca2b0eadc78201236039e9df6a3907e1253b9f64336975478d4e0d3b.jpg](../iclr_results/907_PEARL_ Towards Permutation-Resilient LLMs/images/4aa650a7ca2b0eadc78201236039e9df6a3907e1253b9f64336975478d4e0d3b.jpg)

![8c110b22703c503f9c9ef4025119894b337d71971714bf0735c7a6f9196c4c84.jpg](../iclr_results/907_PEARL_ Towards Permutation-Resilient LLMs/images/8c110b22703c503f9c9ef4025119894b337d71971714bf0735c7a6f9196c4c84.jpg)

![a88b2b59593d040b860bbde3aaac03601b363bcea05d04303b9a0864e4aa0adf.jpg](../iclr_results/907_PEARL_ Towards Permutation-Resilient LLMs/images/a88b2b59593d040b860bbde3aaac03601b363bcea05d04303b9a0864e4aa0adf.jpg)

### Tables

![210abe18753f8452241de5edb07d9fd49795b595918c7c2a595922bb6be2a862.jpg](../iclr_results/907_PEARL_ Towards Permutation-Resilient LLMs/tables/210abe18753f8452241de5edb07d9fd49795b595918c7c2a595922bb6be2a862.jpg)

![236d2b22fa5ce8d9162345dd0a33a4dfcf2916aebe5d659ffcfa2ea0fb01f00c.jpg](../iclr_results/907_PEARL_ Towards Permutation-Resilient LLMs/tables/236d2b22fa5ce8d9162345dd0a33a4dfcf2916aebe5d659ffcfa2ea0fb01f00c.jpg)

![25c6f5b9bdbb4cfaced3a7ae0ede1a822a90e938230a9896b54b2d2f1f71a685.jpg](../iclr_results/907_PEARL_ Towards Permutation-Resilient LLMs/tables/25c6f5b9bdbb4cfaced3a7ae0ede1a822a90e938230a9896b54b2d2f1f71a685.jpg)

![30bdec17a0d7725482bd183c3f6df2f9db88a33270f36b61f3a706014b46da72.jpg](../iclr_results/907_PEARL_ Towards Permutation-Resilient LLMs/tables/30bdec17a0d7725482bd183c3f6df2f9db88a33270f36b61f3a706014b46da72.jpg)

![312a6ae4a2906e8fd968f32c3ecbe51714d70b5394da30775e27c1a13f58fa56.jpg](../iclr_results/907_PEARL_ Towards Permutation-Resilient LLMs/tables/312a6ae4a2906e8fd968f32c3ecbe51714d70b5394da30775e27c1a13f58fa56.jpg)

![38cbe54ceff923dfe44839430ba4b08738878b78575159886ef9bbc1cfdc6060.jpg](../iclr_results/907_PEARL_ Towards Permutation-Resilient LLMs/tables/38cbe54ceff923dfe44839430ba4b08738878b78575159886ef9bbc1cfdc6060.jpg)

![394cec41ff15232e1c6aff4b6875fa4b5348f741834b04c7c266996649794cf3.jpg](../iclr_results/907_PEARL_ Towards Permutation-Resilient LLMs/tables/394cec41ff15232e1c6aff4b6875fa4b5348f741834b04c7c266996649794cf3.jpg)

![4ac92208ef37a5a29db439a7f775e1839a6e5059b076287587ead0249b416943.jpg](../iclr_results/907_PEARL_ Towards Permutation-Resilient LLMs/tables/4ac92208ef37a5a29db439a7f775e1839a6e5059b076287587ead0249b416943.jpg)

![4d85e85bd4b87ccd9d0f2b9bda61f9501ade921d59e7399cbfab459090defae1.jpg](../iclr_results/907_PEARL_ Towards Permutation-Resilient LLMs/tables/4d85e85bd4b87ccd9d0f2b9bda61f9501ade921d59e7399cbfab459090defae1.jpg)

![7607b0e7ec77486e446558986a850c58307b38b8dbca46d5b097b5eddd980829.jpg](../iclr_results/907_PEARL_ Towards Permutation-Resilient LLMs/tables/7607b0e7ec77486e446558986a850c58307b38b8dbca46d5b097b5eddd980829.jpg)

![8eb6e7ed7cbcdb7a02839b04e48785b271e9784b7e28bd74ce56afb0237f7872.jpg](../iclr_results/907_PEARL_ Towards Permutation-Resilient LLMs/tables/8eb6e7ed7cbcdb7a02839b04e48785b271e9784b7e28bd74ce56afb0237f7872.jpg)

![b36bbbf07862bd4885f6d0b084eda93c315f989f46b4a197e0d62adc59e27dec.jpg](../iclr_results/907_PEARL_ Towards Permutation-Resilient LLMs/tables/b36bbbf07862bd4885f6d0b084eda93c315f989f46b4a197e0d62adc59e27dec.jpg)

![bb1dc688522642d543a2b593cc401d96faa9f3f8f7b820aa8d9ffbc3a9b5919b.jpg](../iclr_results/907_PEARL_ Towards Permutation-Resilient LLMs/tables/bb1dc688522642d543a2b593cc401d96faa9f3f8f7b820aa8d9ffbc3a9b5919b.jpg)

![c6932e4d00b3a46826a15580b43a2e3d4190bb3dfcd6ce847a8908b754bc1aff.jpg](../iclr_results/907_PEARL_ Towards Permutation-Resilient LLMs/tables/c6932e4d00b3a46826a15580b43a2e3d4190bb3dfcd6ce847a8908b754bc1aff.jpg)

## RAPID: Retrieval Augmented Training of Differentially Private Diffusion Models


### Images

![15e8fba7903bae74191e3a8021dbe6434778a23dc5686d3ee70e529120e35b64.jpg](../iclr_results/908_RAPID_ Retrieval Augmented Training of Differentially Private Diffusion Models/images/15e8fba7903bae74191e3a8021dbe6434778a23dc5686d3ee70e529120e35b64.jpg)

![3e67763f87315458093e6b0a403d379c030ea3b8b920e77e192ae2563b584d61.jpg](../iclr_results/908_RAPID_ Retrieval Augmented Training of Differentially Private Diffusion Models/images/3e67763f87315458093e6b0a403d379c030ea3b8b920e77e192ae2563b584d61.jpg)

![4754909029d84a142b12e0db31f5af339cf69ae1c41724963b3685695c1c5110.jpg](../iclr_results/908_RAPID_ Retrieval Augmented Training of Differentially Private Diffusion Models/images/4754909029d84a142b12e0db31f5af339cf69ae1c41724963b3685695c1c5110.jpg)

![72756d5cb01aecad91fe0e832bea6166e1e5fc980ffa994553e16394e161dcb6.jpg](../iclr_results/908_RAPID_ Retrieval Augmented Training of Differentially Private Diffusion Models/images/72756d5cb01aecad91fe0e832bea6166e1e5fc980ffa994553e16394e161dcb6.jpg)

![9224fdcdf6dab61ed336c644da1181eb98a67c22780d97eb4b15984ce97abb4a.jpg](../iclr_results/908_RAPID_ Retrieval Augmented Training of Differentially Private Diffusion Models/images/9224fdcdf6dab61ed336c644da1181eb98a67c22780d97eb4b15984ce97abb4a.jpg)

![c68f3428c7e5a5262f7b713c16dabc4080bdf175ddb98183168bd35bb7bbcc7f.jpg](../iclr_results/908_RAPID_ Retrieval Augmented Training of Differentially Private Diffusion Models/images/c68f3428c7e5a5262f7b713c16dabc4080bdf175ddb98183168bd35bb7bbcc7f.jpg)

![c791fd6abb229e47944c1578470f240e0791a7f3f51c39b7c0104aef79d59f74.jpg](../iclr_results/908_RAPID_ Retrieval Augmented Training of Differentially Private Diffusion Models/images/c791fd6abb229e47944c1578470f240e0791a7f3f51c39b7c0104aef79d59f74.jpg)

![cad476880bb802447a4540775091145a792644f2ab9820f404f780e72d0feaf8.jpg](../iclr_results/908_RAPID_ Retrieval Augmented Training of Differentially Private Diffusion Models/images/cad476880bb802447a4540775091145a792644f2ab9820f404f780e72d0feaf8.jpg)

![d9ea4821dcff98714236996b6476318b60a6d61791e111bbb72315055ec3ac34.jpg](../iclr_results/908_RAPID_ Retrieval Augmented Training of Differentially Private Diffusion Models/images/d9ea4821dcff98714236996b6476318b60a6d61791e111bbb72315055ec3ac34.jpg)

![f7caf044d32c2722f3f8437f5f46f58915a42279ab0f8379fe2c97392fb74055.jpg](../iclr_results/908_RAPID_ Retrieval Augmented Training of Differentially Private Diffusion Models/images/f7caf044d32c2722f3f8437f5f46f58915a42279ab0f8379fe2c97392fb74055.jpg)

### Tables

![07362bade16f182fbf651c0ccef4f5a834d27965a9de18c54bf3e1a242ad3029.jpg](../iclr_results/908_RAPID_ Retrieval Augmented Training of Differentially Private Diffusion Models/tables/07362bade16f182fbf651c0ccef4f5a834d27965a9de18c54bf3e1a242ad3029.jpg)

![1f154a6e95989fb1d2af80d94e885a91a12645e54acf0f5e858df301ce7343a8.jpg](../iclr_results/908_RAPID_ Retrieval Augmented Training of Differentially Private Diffusion Models/tables/1f154a6e95989fb1d2af80d94e885a91a12645e54acf0f5e858df301ce7343a8.jpg)

![3ba8be6175f643ee522503fdaa018c38ed8a0b40a88ac6688a930354e592b389.jpg](../iclr_results/908_RAPID_ Retrieval Augmented Training of Differentially Private Diffusion Models/tables/3ba8be6175f643ee522503fdaa018c38ed8a0b40a88ac6688a930354e592b389.jpg)

![3c78ce236baf66377ced95bc22241b451337d05a6708adf9d1d76442105f3a4f.jpg](../iclr_results/908_RAPID_ Retrieval Augmented Training of Differentially Private Diffusion Models/tables/3c78ce236baf66377ced95bc22241b451337d05a6708adf9d1d76442105f3a4f.jpg)

![4a6a1197d9572a7ee6aabda797557138e8b1f013afee45e54558ae0f4da48f87.jpg](../iclr_results/908_RAPID_ Retrieval Augmented Training of Differentially Private Diffusion Models/tables/4a6a1197d9572a7ee6aabda797557138e8b1f013afee45e54558ae0f4da48f87.jpg)

![72f0d39d24c0c650838578df0ad738c03e829703553f371f5dabad707e770318.jpg](../iclr_results/908_RAPID_ Retrieval Augmented Training of Differentially Private Diffusion Models/tables/72f0d39d24c0c650838578df0ad738c03e829703553f371f5dabad707e770318.jpg)

![736ea6d948ba206c158e252f0f3538b2d7eb0defa287a9f014842dce1c7d319c.jpg](../iclr_results/908_RAPID_ Retrieval Augmented Training of Differentially Private Diffusion Models/tables/736ea6d948ba206c158e252f0f3538b2d7eb0defa287a9f014842dce1c7d319c.jpg)

![84249414b24c8c79da4ffe4a2fbf28bd1e6a8b7c3a3e1b7cb5bc264ea2547df8.jpg](../iclr_results/908_RAPID_ Retrieval Augmented Training of Differentially Private Diffusion Models/tables/84249414b24c8c79da4ffe4a2fbf28bd1e6a8b7c3a3e1b7cb5bc264ea2547df8.jpg)

![94a1af9dfb8a3f8b5b30abec629af7aebba0a4aba36db450aa6226ec6bdba83c.jpg](../iclr_results/908_RAPID_ Retrieval Augmented Training of Differentially Private Diffusion Models/tables/94a1af9dfb8a3f8b5b30abec629af7aebba0a4aba36db450aa6226ec6bdba83c.jpg)

![9cb0cb7c8f2a6f7e96c0e205130d9c96298e6ccc9ce16b4494a28832839e0fe7.jpg](../iclr_results/908_RAPID_ Retrieval Augmented Training of Differentially Private Diffusion Models/tables/9cb0cb7c8f2a6f7e96c0e205130d9c96298e6ccc9ce16b4494a28832839e0fe7.jpg)

![a7dac21646c45da02639047e1c140c5c633083a55b5f13fdd1221a10912a4b44.jpg](../iclr_results/908_RAPID_ Retrieval Augmented Training of Differentially Private Diffusion Models/tables/a7dac21646c45da02639047e1c140c5c633083a55b5f13fdd1221a10912a4b44.jpg)

![d1e498ca297e8e7e0f13bdc1bc3a535e6c65a640f8b495b7cea94440ecb6dcf1.jpg](../iclr_results/908_RAPID_ Retrieval Augmented Training of Differentially Private Diffusion Models/tables/d1e498ca297e8e7e0f13bdc1bc3a535e6c65a640f8b495b7cea94440ecb6dcf1.jpg)

## Near-Exact Privacy Amplification for Matrix Mechanisms


### Images

![0b5bfee516d8f4b39f5013fe50d8244ab7938223392162f23b2f6ad5269a8cad.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/0b5bfee516d8f4b39f5013fe50d8244ab7938223392162f23b2f6ad5269a8cad.jpg)

![1e1daf0a926ecf1b5ed2ed4b28e44756fc393ff447ca41b4ea33c9f4e795ad27.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/1e1daf0a926ecf1b5ed2ed4b28e44756fc393ff447ca41b4ea33c9f4e795ad27.jpg)

![25c6a1f937a9874d0b650a38ae26f4fd3c8425f79e018f339b6bcfa70f278ea6.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/25c6a1f937a9874d0b650a38ae26f4fd3c8425f79e018f339b6bcfa70f278ea6.jpg)

![262fb0b0d71beba45e26c4176e6255e51ca1ed7ec860b1775ed604490038befe.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/262fb0b0d71beba45e26c4176e6255e51ca1ed7ec860b1775ed604490038befe.jpg)

![2bd7674284f958120850aeb4c3bc4f535b8b3b2ef613cf172cfe733f9e6bfc98.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/2bd7674284f958120850aeb4c3bc4f535b8b3b2ef613cf172cfe733f9e6bfc98.jpg)

![32b172bf00ae0c5a719079b9c2651ac461c6b57fdd1c0bf9d140f03ca3557af8.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/32b172bf00ae0c5a719079b9c2651ac461c6b57fdd1c0bf9d140f03ca3557af8.jpg)

![3f6a94e56cc2cac33b61d8de8d4c4a1c255a0582e92cac3e07f835851ed60dde.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/3f6a94e56cc2cac33b61d8de8d4c4a1c255a0582e92cac3e07f835851ed60dde.jpg)

![4424124c249b72c56f3b8ff164c6dd116100aa9c386b916b5e8eb7c8de61f952.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/4424124c249b72c56f3b8ff164c6dd116100aa9c386b916b5e8eb7c8de61f952.jpg)

![4b920071ecc25300dcfcc12866b9b7a055ec450a1da3d8405ae2469a7d992d67.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/4b920071ecc25300dcfcc12866b9b7a055ec450a1da3d8405ae2469a7d992d67.jpg)

![51a38764c7dc539994e736cde76c962ef8fc63d4c96d88006d73ece12f297350.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/51a38764c7dc539994e736cde76c962ef8fc63d4c96d88006d73ece12f297350.jpg)

![6c082496c54a102ef54bc1edbb323c9f3f677e186cec5f03a78a1b9947ead437.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/6c082496c54a102ef54bc1edbb323c9f3f677e186cec5f03a78a1b9947ead437.jpg)

![6d0f2665fbef75d5edbaeb49bba02fdeffc91c1c57c109bfb2cc09b1ba5ff53b.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/6d0f2665fbef75d5edbaeb49bba02fdeffc91c1c57c109bfb2cc09b1ba5ff53b.jpg)

![6eb46838f3d4cd1f16d76fb4ae5af3736e3c1215cf506cad8d427a95a8df9e97.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/6eb46838f3d4cd1f16d76fb4ae5af3736e3c1215cf506cad8d427a95a8df9e97.jpg)

![71dbd1ed5ca19388fdfbd54af8e7f951a52315626b2be9331b122ce6eef93889.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/71dbd1ed5ca19388fdfbd54af8e7f951a52315626b2be9331b122ce6eef93889.jpg)

![72c1c0612393af9fc6e9d5b4b3389454d397567f585ebc667a847c9690033124.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/72c1c0612393af9fc6e9d5b4b3389454d397567f585ebc667a847c9690033124.jpg)

![7e996705fcc01ae5f09b14e811771e5465e496aef163c3716617201a07addf90.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/7e996705fcc01ae5f09b14e811771e5465e496aef163c3716617201a07addf90.jpg)

![83908807093e6d794c900e51628522fd75ec21d227d4a85bbc51ab8924037b51.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/83908807093e6d794c900e51628522fd75ec21d227d4a85bbc51ab8924037b51.jpg)

![92f63cf5886e857b6beae6615971b17d0777305f2587a115b67d32f20d6d72f6.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/92f63cf5886e857b6beae6615971b17d0777305f2587a115b67d32f20d6d72f6.jpg)

![a09fb2ee28212e04b9cb857988bb0cccf20d2372ac68981d6188eef93864469d.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/a09fb2ee28212e04b9cb857988bb0cccf20d2372ac68981d6188eef93864469d.jpg)

![a1490c6425274aa01fda98cd89ef8a29d66fe5f560cef36276828d1918f457ab.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/a1490c6425274aa01fda98cd89ef8a29d66fe5f560cef36276828d1918f457ab.jpg)

![a2aea0e447ab1bbbb4f783036076298e0bb2bce9d5c1d6c2e2534854369c2d36.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/a2aea0e447ab1bbbb4f783036076298e0bb2bce9d5c1d6c2e2534854369c2d36.jpg)

![a6c7038f521ef12f96822b866eaae6026bb7f0c84f8da0ac512ff13e7f8d552f.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/a6c7038f521ef12f96822b866eaae6026bb7f0c84f8da0ac512ff13e7f8d552f.jpg)

![a74da7ba19251d8ff2c27b1d8264cb5393d718aaec0952aa04343a87624d5f9e.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/a74da7ba19251d8ff2c27b1d8264cb5393d718aaec0952aa04343a87624d5f9e.jpg)

![b0d79295c873b22afbfe98d810a4cd19a22e6471a8bb8f7921407afe697a138d.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/b0d79295c873b22afbfe98d810a4cd19a22e6471a8bb8f7921407afe697a138d.jpg)

![b4d5cfd4f6b294c3a9e3b3c52d109528cc5746fcc34c9a8442239c139c10fd3d.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/b4d5cfd4f6b294c3a9e3b3c52d109528cc5746fcc34c9a8442239c139c10fd3d.jpg)

![e3d9ca8189dafbd4c1c9bfb8a7c0b6f6971f0f2163016e18915d6a701c9d9033.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/e3d9ca8189dafbd4c1c9bfb8a7c0b6f6971f0f2163016e18915d6a701c9d9033.jpg)

![edca8c625cb437f1e6d8d7892bbe6c02f905969a1a17bd849cf425e15b999361.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/edca8c625cb437f1e6d8d7892bbe6c02f905969a1a17bd849cf425e15b999361.jpg)

![f1e54fd7163ae2db74038a59776506af9349b50a9c74862cfd895ff1878f572c.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/images/f1e54fd7163ae2db74038a59776506af9349b50a9c74862cfd895ff1878f572c.jpg)

### Tables

![2da6ad69780eda46a001eb0a8104bae50f64081851a34e049e339d5bf9baa395.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/tables/2da6ad69780eda46a001eb0a8104bae50f64081851a34e049e339d5bf9baa395.jpg)

![4113904dcd8003a471e62d49e7b15f4a46cc5b459dcfb59bb9a4ecacec40aedd.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/tables/4113904dcd8003a471e62d49e7b15f4a46cc5b459dcfb59bb9a4ecacec40aedd.jpg)

![d237c5a8b1ce21627f47bcc8b3027f78ec3f85f90b5350c0dc9f79b126ed2051.jpg](../iclr_results/909_Near-Exact Privacy Amplification for Matrix Mechanisms/tables/d237c5a8b1ce21627f47bcc8b3027f78ec3f85f90b5350c0dc9f79b126ed2051.jpg)

## A Coefficient Makes SVRG Effective


### Images

![0453ecb6dc6caebfc2250ab93e6323973b6519b0a8af5a4d485e6cb0c6670467.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/images/0453ecb6dc6caebfc2250ab93e6323973b6519b0a8af5a4d485e6cb0c6670467.jpg)

![0987e4822c3ff23afca99f31f94074ada61dde67b83faf3bb8b40fc42d2fca49.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/images/0987e4822c3ff23afca99f31f94074ada61dde67b83faf3bb8b40fc42d2fca49.jpg)

![1f01d90f9791d277fbbdff1b1ba84d7c9f4306b6bccce59531fe2f2a6027a870.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/images/1f01d90f9791d277fbbdff1b1ba84d7c9f4306b6bccce59531fe2f2a6027a870.jpg)

![395065b7d4693c3ad458f31676060fef9618f806319e549dcee3495bb9661eb6.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/images/395065b7d4693c3ad458f31676060fef9618f806319e549dcee3495bb9661eb6.jpg)

![486d4c399b1d154fc64bc9d64e9faf845337cfc5fe67f08a12e0c45ce7f4c662.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/images/486d4c399b1d154fc64bc9d64e9faf845337cfc5fe67f08a12e0c45ce7f4c662.jpg)

![4a3ffbffe0392c15910845376807a7fc06b58de9a8da60ecd4e30b32d32eecd6.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/images/4a3ffbffe0392c15910845376807a7fc06b58de9a8da60ecd4e30b32d32eecd6.jpg)

![56d0ffcf1b34b19e345aac73bc6918cb7170e17ac6576d654d634dbf2626bd09.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/images/56d0ffcf1b34b19e345aac73bc6918cb7170e17ac6576d654d634dbf2626bd09.jpg)

![799a02162d54395bad40dc787d5e763b17a16d66d355dd49bb43f56271ecd2ca.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/images/799a02162d54395bad40dc787d5e763b17a16d66d355dd49bb43f56271ecd2ca.jpg)

![7dd23d6324a6ab672bb179fc706223375276b1ef217dbdbb88fb026a053e9025.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/images/7dd23d6324a6ab672bb179fc706223375276b1ef217dbdbb88fb026a053e9025.jpg)

![824857747517756d99e4c297caf6f959d1730ec763962a658fda1132053b18f2.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/images/824857747517756d99e4c297caf6f959d1730ec763962a658fda1132053b18f2.jpg)

![90d90ae998d2ce2fbc0f4fde1a029bc15bea356bc038b3b17ac076a1dc19369f.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/images/90d90ae998d2ce2fbc0f4fde1a029bc15bea356bc038b3b17ac076a1dc19369f.jpg)

![98c3213e40747cbf4db9548b04ea909661c1148c66a79b40e77b489d4bc82dca.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/images/98c3213e40747cbf4db9548b04ea909661c1148c66a79b40e77b489d4bc82dca.jpg)

![a09b3bea619d11b71c3bf619b4d10103786014a1d96edb16c11296f72d1edb71.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/images/a09b3bea619d11b71c3bf619b4d10103786014a1d96edb16c11296f72d1edb71.jpg)

![a6eedacf0130f70389faf74f5239da51bdb1734715c93ff2f11505e7bfd6cb42.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/images/a6eedacf0130f70389faf74f5239da51bdb1734715c93ff2f11505e7bfd6cb42.jpg)

![b377353058f80e855f068f0a5b43ce9b9086e13bcef48f100d5afa189f48426d.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/images/b377353058f80e855f068f0a5b43ce9b9086e13bcef48f100d5afa189f48426d.jpg)

![d0449c4d757473d770b7392a92b03c4db50f3e2af5a099de7fd819b054f5bc64.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/images/d0449c4d757473d770b7392a92b03c4db50f3e2af5a099de7fd819b054f5bc64.jpg)

![db2385ad682013538c5df867d45506d4c8cd80507e73da99ac86051e69fdcfb6.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/images/db2385ad682013538c5df867d45506d4c8cd80507e73da99ac86051e69fdcfb6.jpg)

![e41599bbc0dd356403e4ac40288307d4a3f8f9a04828e495ed86008a91d071e2.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/images/e41599bbc0dd356403e4ac40288307d4a3f8f9a04828e495ed86008a91d071e2.jpg)

![e53f84968c45d2de1adddfa4e59fb9315c7ea9f53be931589461e9c6d72a520e.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/images/e53f84968c45d2de1adddfa4e59fb9315c7ea9f53be931589461e9c6d72a520e.jpg)

![e9a27831960d3b3c0ea41e72bf5b39f50b869e5b2f745397e5149554c8cd31d0.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/images/e9a27831960d3b3c0ea41e72bf5b39f50b869e5b2f745397e5149554c8cd31d0.jpg)

### Tables

![1c0552b709fb23bdf1276de407501ec77cf83c4055afcd8eb6e56afe26d12c4c.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/tables/1c0552b709fb23bdf1276de407501ec77cf83c4055afcd8eb6e56afe26d12c4c.jpg)

![233dddb8c15fe6f0fc1cd107e4dfab9e6258000d80e9624086989287d1dc2992.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/tables/233dddb8c15fe6f0fc1cd107e4dfab9e6258000d80e9624086989287d1dc2992.jpg)

![44da4ece25ee600ec29009a1e192214165f5c79098ac052edc01be58de0216a4.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/tables/44da4ece25ee600ec29009a1e192214165f5c79098ac052edc01be58de0216a4.jpg)

![6ee4acc75fc663ed9cfa394840bfeb50d79398dfca5ee2ae0f9db0e47b8b15ee.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/tables/6ee4acc75fc663ed9cfa394840bfeb50d79398dfca5ee2ae0f9db0e47b8b15ee.jpg)

![76172a68067ee35c342c9c2607790a85e3c31c4dc1b46c96cec9dfc2a3aa6469.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/tables/76172a68067ee35c342c9c2607790a85e3c31c4dc1b46c96cec9dfc2a3aa6469.jpg)

![7d64ccd4da3d5f5661419763e9b07ce457a548475b3479c046f0dceecfe33777.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/tables/7d64ccd4da3d5f5661419763e9b07ce457a548475b3479c046f0dceecfe33777.jpg)

![913c467126bd2e3718522d6c567397e43cc9353d292c1dbb868556c2c6311c69.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/tables/913c467126bd2e3718522d6c567397e43cc9353d292c1dbb868556c2c6311c69.jpg)

![9f506456e300a02901ca438eaa966594d2f27354997777105759beb96ca50e56.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/tables/9f506456e300a02901ca438eaa966594d2f27354997777105759beb96ca50e56.jpg)

![bd6083f288d25f7d65d1bc93f3e8af675bf9d638d2f132f42b5ffad2466626fe.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/tables/bd6083f288d25f7d65d1bc93f3e8af675bf9d638d2f132f42b5ffad2466626fe.jpg)

![becb9183bddb09d24de71bcf2545e78ac10e21fc9109bfa355459d2abde427ec.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/tables/becb9183bddb09d24de71bcf2545e78ac10e21fc9109bfa355459d2abde427ec.jpg)

![cd96222c1e2b015131705fad720aee2bb088f7e0a24b9dc61703f39c4b75f075.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/tables/cd96222c1e2b015131705fad720aee2bb088f7e0a24b9dc61703f39c4b75f075.jpg)

![deb5ad06c23663b1817ec4cf39852c42a54a2a6582370adc6e4ebadf64a953e0.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/tables/deb5ad06c23663b1817ec4cf39852c42a54a2a6582370adc6e4ebadf64a953e0.jpg)

![e5c7f3cb732dedbfc1a8d5c9c21d7cb74cc19242b36cbcba9c9e0bb31bc02abb.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/tables/e5c7f3cb732dedbfc1a8d5c9c21d7cb74cc19242b36cbcba9c9e0bb31bc02abb.jpg)

![e79060131b4ebe430cb8e5b286a0f30222bf084a1dcdea2b36153d89355f6ee5.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/tables/e79060131b4ebe430cb8e5b286a0f30222bf084a1dcdea2b36153d89355f6ee5.jpg)

![e864724de41ada3cceec68baa087eabebaef92f6e88cd39ec87f3d6036d5b46d.jpg](../iclr_results/911_A Coefficient Makes SVRG Effective/tables/e864724de41ada3cceec68baa087eabebaef92f6e88cd39ec87f3d6036d5b46d.jpg)

## Cross-Embodiment Dexterous Grasping with Reinforcement Learning


### Images

![110d6257306551d914387aebf9969b1091a190674eb11308edae1aee9cf91b0b.jpg](../iclr_results/912_Cross-Embodiment Dexterous Grasping with Reinforcement Learning/images/110d6257306551d914387aebf9969b1091a190674eb11308edae1aee9cf91b0b.jpg)

![282954796c64d1c05c21fdb7c7c4e2a58f57421103c7775ba22aa49b631458f5.jpg](../iclr_results/912_Cross-Embodiment Dexterous Grasping with Reinforcement Learning/images/282954796c64d1c05c21fdb7c7c4e2a58f57421103c7775ba22aa49b631458f5.jpg)

![2c437ea03aa12f4d6c362300f9cdcb9c9d7200b25a007a2e18a2f459c3020de8.jpg](../iclr_results/912_Cross-Embodiment Dexterous Grasping with Reinforcement Learning/images/2c437ea03aa12f4d6c362300f9cdcb9c9d7200b25a007a2e18a2f459c3020de8.jpg)

![4b3327fe742c90a50bf02d91286f27b51e2f4425ad1d56077347e7927b46b609.jpg](../iclr_results/912_Cross-Embodiment Dexterous Grasping with Reinforcement Learning/images/4b3327fe742c90a50bf02d91286f27b51e2f4425ad1d56077347e7927b46b609.jpg)

![4f7fdcf6941eba0087b63453fa1c6b08b4f83b4c88e68a94cec99a978e6a351a.jpg](../iclr_results/912_Cross-Embodiment Dexterous Grasping with Reinforcement Learning/images/4f7fdcf6941eba0087b63453fa1c6b08b4f83b4c88e68a94cec99a978e6a351a.jpg)

![6dca7e1b74e3c45906362e518cb0a1195566085f1c0902b22b5125f437ee5daf.jpg](../iclr_results/912_Cross-Embodiment Dexterous Grasping with Reinforcement Learning/images/6dca7e1b74e3c45906362e518cb0a1195566085f1c0902b22b5125f437ee5daf.jpg)

![85ecb10a7cc054befdd41be34473b2ac1f12117f8bb6756561aa239ca0e4165b.jpg](../iclr_results/912_Cross-Embodiment Dexterous Grasping with Reinforcement Learning/images/85ecb10a7cc054befdd41be34473b2ac1f12117f8bb6756561aa239ca0e4165b.jpg)

![a3c25aea54a6f54cd041058cb3efc5f9b9bd58366d30fa735a46596b562285bd.jpg](../iclr_results/912_Cross-Embodiment Dexterous Grasping with Reinforcement Learning/images/a3c25aea54a6f54cd041058cb3efc5f9b9bd58366d30fa735a46596b562285bd.jpg)

![c7b03b1967288c4af062b789dc4200b11db5bcef2b7bdb9331a51bd3c9f30e7c.jpg](../iclr_results/912_Cross-Embodiment Dexterous Grasping with Reinforcement Learning/images/c7b03b1967288c4af062b789dc4200b11db5bcef2b7bdb9331a51bd3c9f30e7c.jpg)

![db51683428384b4d1cf41553ee638e23818b9fd75dfee2bcdb094b9d3d9d58b3.jpg](../iclr_results/912_Cross-Embodiment Dexterous Grasping with Reinforcement Learning/images/db51683428384b4d1cf41553ee638e23818b9fd75dfee2bcdb094b9d3d9d58b3.jpg)

![e195d76b744a2c36c016a13f3892467057d728e7fd84635360a12b150267d10e.jpg](../iclr_results/912_Cross-Embodiment Dexterous Grasping with Reinforcement Learning/images/e195d76b744a2c36c016a13f3892467057d728e7fd84635360a12b150267d10e.jpg)

### Tables

![0930d5edaa874d173217f2f544eb52a8a2939c94a281341563a0133e1245d50e.jpg](../iclr_results/912_Cross-Embodiment Dexterous Grasping with Reinforcement Learning/tables/0930d5edaa874d173217f2f544eb52a8a2939c94a281341563a0133e1245d50e.jpg)

![1b41aecc7d58fa5256aee4d29e1ca280e3bc38c3488bfefd7e7f46d0676a3e1c.jpg](../iclr_results/912_Cross-Embodiment Dexterous Grasping with Reinforcement Learning/tables/1b41aecc7d58fa5256aee4d29e1ca280e3bc38c3488bfefd7e7f46d0676a3e1c.jpg)

![44c597af1f270389b1c6006096c40cea39e03207583dab17622dcf78e6ce3c53.jpg](../iclr_results/912_Cross-Embodiment Dexterous Grasping with Reinforcement Learning/tables/44c597af1f270389b1c6006096c40cea39e03207583dab17622dcf78e6ce3c53.jpg)

![7fc6a27597ecec89b3e155742faf6dd1a74cba7050ec958cf89e6d39e9991539.jpg](../iclr_results/912_Cross-Embodiment Dexterous Grasping with Reinforcement Learning/tables/7fc6a27597ecec89b3e155742faf6dd1a74cba7050ec958cf89e6d39e9991539.jpg)

![83760d49f5e356775105b542d766194554ac94aac9c5bd5a5f288b107e6a6cb7.jpg](../iclr_results/912_Cross-Embodiment Dexterous Grasping with Reinforcement Learning/tables/83760d49f5e356775105b542d766194554ac94aac9c5bd5a5f288b107e6a6cb7.jpg)

![856868d5b801fc22cec3275dda8b8d7b45ba64721ec20b8d3bb3ece22859a9e6.jpg](../iclr_results/912_Cross-Embodiment Dexterous Grasping with Reinforcement Learning/tables/856868d5b801fc22cec3275dda8b8d7b45ba64721ec20b8d3bb3ece22859a9e6.jpg)

![a763af3f04e38fbf915e8bdd651ded7badb578d49fad9140c066289e7596d713.jpg](../iclr_results/912_Cross-Embodiment Dexterous Grasping with Reinforcement Learning/tables/a763af3f04e38fbf915e8bdd651ded7badb578d49fad9140c066289e7596d713.jpg)

![bc5d905d0eac7ebf0a15c06546744de26e15375557254a6672d39093a85c6853.jpg](../iclr_results/912_Cross-Embodiment Dexterous Grasping with Reinforcement Learning/tables/bc5d905d0eac7ebf0a15c06546744de26e15375557254a6672d39093a85c6853.jpg)

## On Generalization Across Environments In Multi-Objective Reinforcement Learning


### Images

![09e6d2b05b1356208b961dd2a711f41b64308ee4482deccd0643ae6c5a35f32c.jpg](../iclr_results/913_On Generalization Across Environments In Multi-Objective Reinforcement Learning/images/09e6d2b05b1356208b961dd2a711f41b64308ee4482deccd0643ae6c5a35f32c.jpg)

![3aaf2a4bafaba0fb19e5341b64c88f11bc25bc1505d2ad0d2588abbd85320910.jpg](../iclr_results/913_On Generalization Across Environments In Multi-Objective Reinforcement Learning/images/3aaf2a4bafaba0fb19e5341b64c88f11bc25bc1505d2ad0d2588abbd85320910.jpg)

![3d0013fe4d2798d61537524eb6991f12faab17a62f287b7c6a7ef416f5b991b0.jpg](../iclr_results/913_On Generalization Across Environments In Multi-Objective Reinforcement Learning/images/3d0013fe4d2798d61537524eb6991f12faab17a62f287b7c6a7ef416f5b991b0.jpg)

![3e03c1c86ed3df3cd24c8736566d8e1f266b57d7390f2599472ce1f60e719aca.jpg](../iclr_results/913_On Generalization Across Environments In Multi-Objective Reinforcement Learning/images/3e03c1c86ed3df3cd24c8736566d8e1f266b57d7390f2599472ce1f60e719aca.jpg)

![7af05acb715fcb183f987a7582aa920646d764a10b74457937dc3a88b72e7d92.jpg](../iclr_results/913_On Generalization Across Environments In Multi-Objective Reinforcement Learning/images/7af05acb715fcb183f987a7582aa920646d764a10b74457937dc3a88b72e7d92.jpg)

![92adae154359c9a798adf3d69485456ac906a1ea26f9e8abca8f62dea20f4da0.jpg](../iclr_results/913_On Generalization Across Environments In Multi-Objective Reinforcement Learning/images/92adae154359c9a798adf3d69485456ac906a1ea26f9e8abca8f62dea20f4da0.jpg)

![97f031b7ee3c4892150081247b894a7341a3cfea5f0e0fbc9b3e1477cd3c3055.jpg](../iclr_results/913_On Generalization Across Environments In Multi-Objective Reinforcement Learning/images/97f031b7ee3c4892150081247b894a7341a3cfea5f0e0fbc9b3e1477cd3c3055.jpg)

![a4c58cba97401c068a5bedfb7eda73f30a30ca95264a6567b7b4581306ad2977.jpg](../iclr_results/913_On Generalization Across Environments In Multi-Objective Reinforcement Learning/images/a4c58cba97401c068a5bedfb7eda73f30a30ca95264a6567b7b4581306ad2977.jpg)

![aa528835034e94a3a09179a5e1c0ca0aa57be56fe0d1830936277bad19d0b902.jpg](../iclr_results/913_On Generalization Across Environments In Multi-Objective Reinforcement Learning/images/aa528835034e94a3a09179a5e1c0ca0aa57be56fe0d1830936277bad19d0b902.jpg)

![b0798497b9925c235116a126ea775e7d67ad3e2d17cae42ea67042191e0511c2.jpg](../iclr_results/913_On Generalization Across Environments In Multi-Objective Reinforcement Learning/images/b0798497b9925c235116a126ea775e7d67ad3e2d17cae42ea67042191e0511c2.jpg)

![c4b3341e2e38653caf0354e5ca847e29e999ebd603de07a8ea2530c894d1ffac.jpg](../iclr_results/913_On Generalization Across Environments In Multi-Objective Reinforcement Learning/images/c4b3341e2e38653caf0354e5ca847e29e999ebd603de07a8ea2530c894d1ffac.jpg)

![d9aee8b683e21d3530d4788217e2339f7c4feb3246c03fb7b612e9fe5b3d0660.jpg](../iclr_results/913_On Generalization Across Environments In Multi-Objective Reinforcement Learning/images/d9aee8b683e21d3530d4788217e2339f7c4feb3246c03fb7b612e9fe5b3d0660.jpg)

![fc58e21308267a449fceb4edb8ccfdac292203d7671c0c1247c44431d2ddb054.jpg](../iclr_results/913_On Generalization Across Environments In Multi-Objective Reinforcement Learning/images/fc58e21308267a449fceb4edb8ccfdac292203d7671c0c1247c44431d2ddb054.jpg)

### Tables

![3ff8ae0df71d2976008fe44d2dd754c64acd74c55b959264dad3d6575d658db7.jpg](../iclr_results/913_On Generalization Across Environments In Multi-Objective Reinforcement Learning/tables/3ff8ae0df71d2976008fe44d2dd754c64acd74c55b959264dad3d6575d658db7.jpg)

![447d2f87642c83848c8536a18504cc97a70f33ac27bffb885de53ce37337b453.jpg](../iclr_results/913_On Generalization Across Environments In Multi-Objective Reinforcement Learning/tables/447d2f87642c83848c8536a18504cc97a70f33ac27bffb885de53ce37337b453.jpg)

![588598c2e41692df18d1e3fc1590b5e2da250fae02b68ca9fcffb0f010295a96.jpg](../iclr_results/913_On Generalization Across Environments In Multi-Objective Reinforcement Learning/tables/588598c2e41692df18d1e3fc1590b5e2da250fae02b68ca9fcffb0f010295a96.jpg)

![657bc2b4bac8f0b97df9a72ac2d26aab0c3339935af0fd8e5c1dbb5f5c42f44c.jpg](../iclr_results/913_On Generalization Across Environments In Multi-Objective Reinforcement Learning/tables/657bc2b4bac8f0b97df9a72ac2d26aab0c3339935af0fd8e5c1dbb5f5c42f44c.jpg)

![711c478375e49605e4a63742370847890bd3afd197df1159a75bc70b07c66718.jpg](../iclr_results/913_On Generalization Across Environments In Multi-Objective Reinforcement Learning/tables/711c478375e49605e4a63742370847890bd3afd197df1159a75bc70b07c66718.jpg)

![77e9b1b2eec021e52c6ac0affe956a16a9004a58d222c6726e970a9054dc6a35.jpg](../iclr_results/913_On Generalization Across Environments In Multi-Objective Reinforcement Learning/tables/77e9b1b2eec021e52c6ac0affe956a16a9004a58d222c6726e970a9054dc6a35.jpg)

![cca9679902211eff9a9d1b8e604e407e401e58ca7349ec0a4298e29feac0b0af.jpg](../iclr_results/913_On Generalization Across Environments In Multi-Objective Reinforcement Learning/tables/cca9679902211eff9a9d1b8e604e407e401e58ca7349ec0a4298e29feac0b0af.jpg)

![d4dd3f71f2b69666c6766a76c9b06a2169db9b77b6bdc3227c4537951c273751.jpg](../iclr_results/913_On Generalization Across Environments In Multi-Objective Reinforcement Learning/tables/d4dd3f71f2b69666c6766a76c9b06a2169db9b77b6bdc3227c4537951c273751.jpg)

## Composable Interventions for Language Models


### Images

![3f475e0f5357c2a6770517ba5c8fc4249cd88c3bcf22702df0c7eaa94c6564a5.jpg](../iclr_results/914_Composable Interventions for Language Models/images/3f475e0f5357c2a6770517ba5c8fc4249cd88c3bcf22702df0c7eaa94c6564a5.jpg)

![4dd111d6786953983391af4f7fcd45d709858c1cfeb13fa047f12b8962d15254.jpg](../iclr_results/914_Composable Interventions for Language Models/images/4dd111d6786953983391af4f7fcd45d709858c1cfeb13fa047f12b8962d15254.jpg)

![9961a6a68e769ab2d7bc8bc8b80601193ebce814c02a3cd6da72758a04ab52f8.jpg](../iclr_results/914_Composable Interventions for Language Models/images/9961a6a68e769ab2d7bc8bc8b80601193ebce814c02a3cd6da72758a04ab52f8.jpg)

![f16bcf6ed8956774aa226cb0c2b86d4da73a6c62a84c32fc334d420291c40257.jpg](../iclr_results/914_Composable Interventions for Language Models/images/f16bcf6ed8956774aa226cb0c2b86d4da73a6c62a84c32fc334d420291c40257.jpg)

### Tables

![0216741b39784fc592902671f51025311af0360ba1443de9d2bdbe7db15fbb34.jpg](../iclr_results/914_Composable Interventions for Language Models/tables/0216741b39784fc592902671f51025311af0360ba1443de9d2bdbe7db15fbb34.jpg)

![0cf783b0fead475a570050e5123cc48f52143fa25123c608271e965306a82d3e.jpg](../iclr_results/914_Composable Interventions for Language Models/tables/0cf783b0fead475a570050e5123cc48f52143fa25123c608271e965306a82d3e.jpg)

![1fc15d12992be1fab8ea411f801e8904b21376d82a4564b3b73d0b1aa7719ddf.jpg](../iclr_results/914_Composable Interventions for Language Models/tables/1fc15d12992be1fab8ea411f801e8904b21376d82a4564b3b73d0b1aa7719ddf.jpg)

![22e9cb0db6a5adfe8f3825d0cdb1291baf5f66a12cc7349deefa82502eba3c60.jpg](../iclr_results/914_Composable Interventions for Language Models/tables/22e9cb0db6a5adfe8f3825d0cdb1291baf5f66a12cc7349deefa82502eba3c60.jpg)

![290911a7a3120937151f43a10b5cf8a2c356bbd25bf10b0be448fb6e374279c3.jpg](../iclr_results/914_Composable Interventions for Language Models/tables/290911a7a3120937151f43a10b5cf8a2c356bbd25bf10b0be448fb6e374279c3.jpg)

![3f4c19f132b221f5b7fe100432d47a98b4ed858ada7cdfa0b7e06b9d3b3d032b.jpg](../iclr_results/914_Composable Interventions for Language Models/tables/3f4c19f132b221f5b7fe100432d47a98b4ed858ada7cdfa0b7e06b9d3b3d032b.jpg)

![5beaace7447f2d6658fd87cb2d5c5484c1ffabd0442ee645bb4dc2f1df3ed676.jpg](../iclr_results/914_Composable Interventions for Language Models/tables/5beaace7447f2d6658fd87cb2d5c5484c1ffabd0442ee645bb4dc2f1df3ed676.jpg)

![7472c67a286da0d81352635aac2917eda02109382a25b1ab7ac6f3ca7017710a.jpg](../iclr_results/914_Composable Interventions for Language Models/tables/7472c67a286da0d81352635aac2917eda02109382a25b1ab7ac6f3ca7017710a.jpg)

![7ce2b42789bd18fd818a4be1af39b39b948ac6558a05bba290351f4f884cd30e.jpg](../iclr_results/914_Composable Interventions for Language Models/tables/7ce2b42789bd18fd818a4be1af39b39b948ac6558a05bba290351f4f884cd30e.jpg)

![8e6174b95d928b5748e885a2a40be0b50f5f064fd400e6202e4e8082ab5cd3b0.jpg](../iclr_results/914_Composable Interventions for Language Models/tables/8e6174b95d928b5748e885a2a40be0b50f5f064fd400e6202e4e8082ab5cd3b0.jpg)

![928d92a7aeca2ef7f97a9df8aba777d7306c1dda0d9525c40a13bac76c11352d.jpg](../iclr_results/914_Composable Interventions for Language Models/tables/928d92a7aeca2ef7f97a9df8aba777d7306c1dda0d9525c40a13bac76c11352d.jpg)

![9c212d71ae0ecd1c0145a4ee0ad8dc227dc9b4ed5cf5ef8b574d3b424e513184.jpg](../iclr_results/914_Composable Interventions for Language Models/tables/9c212d71ae0ecd1c0145a4ee0ad8dc227dc9b4ed5cf5ef8b574d3b424e513184.jpg)

![9fdc3f526e5e7e2d1f88b318283a91ad41204fd3044539c9beaa053b387a5a60.jpg](../iclr_results/914_Composable Interventions for Language Models/tables/9fdc3f526e5e7e2d1f88b318283a91ad41204fd3044539c9beaa053b387a5a60.jpg)

![bca19f108f945c2e9bc035606047a638c0fe94831bec13da9717ec18a61c4fe4.jpg](../iclr_results/914_Composable Interventions for Language Models/tables/bca19f108f945c2e9bc035606047a638c0fe94831bec13da9717ec18a61c4fe4.jpg)

![c62802b43c1886ab3f2aac91da7371f4a6f4513bc16b2f5b0cb9f99ee69db1e9.jpg](../iclr_results/914_Composable Interventions for Language Models/tables/c62802b43c1886ab3f2aac91da7371f4a6f4513bc16b2f5b0cb9f99ee69db1e9.jpg)

![d2fd2fa78a085cd2271082e6af71d651a93e789d7830db394837048b468b227f.jpg](../iclr_results/914_Composable Interventions for Language Models/tables/d2fd2fa78a085cd2271082e6af71d651a93e789d7830db394837048b468b227f.jpg)

![dd11cce58de5965b8b8ac8d1771b76c97d12e7d3cef7e7be0282a76977e6be57.jpg](../iclr_results/914_Composable Interventions for Language Models/tables/dd11cce58de5965b8b8ac8d1771b76c97d12e7d3cef7e7be0282a76977e6be57.jpg)

![dedea79d01fb87145c1c9bb5fe21ab765965810fd3d2efb4bf2a892a8dbb5007.jpg](../iclr_results/914_Composable Interventions for Language Models/tables/dedea79d01fb87145c1c9bb5fe21ab765965810fd3d2efb4bf2a892a8dbb5007.jpg)

![df2555cc9ef0d969ef3df39cd1a2ce5ae06789cc6925d2a6c7542dd82a38c927.jpg](../iclr_results/914_Composable Interventions for Language Models/tables/df2555cc9ef0d969ef3df39cd1a2ce5ae06789cc6925d2a6c7542dd82a38c927.jpg)

![e995095b9e51e2100a5903f42123f5479607e6b9388238bcfc960ae39bc09355.jpg](../iclr_results/914_Composable Interventions for Language Models/tables/e995095b9e51e2100a5903f42123f5479607e6b9388238bcfc960ae39bc09355.jpg)

![fcf5b4385a83befc4d86a8ff0508583271ffcd7be52e39f868d04971e8720eec.jpg](../iclr_results/914_Composable Interventions for Language Models/tables/fcf5b4385a83befc4d86a8ff0508583271ffcd7be52e39f868d04971e8720eec.jpg)

## Bounds on $L_p$ Errors in Density Ratio Estimation via $f$-Divergence Loss Functions


### Images

![33c07746157c2c96d13a9186a1f9739c1c7259242287984a9aa94a7930fa13c5.jpg](../iclr_results/915_Bounds on $L_p$ Errors in Density Ratio Estimation via $f$-Divergence Loss Functions/images/33c07746157c2c96d13a9186a1f9739c1c7259242287984a9aa94a7930fa13c5.jpg)

![4b4b133f92acc8fe1275b0151cd7cf2812b251134ba4a88315460285ec584e6a.jpg](../iclr_results/915_Bounds on $L_p$ Errors in Density Ratio Estimation via $f$-Divergence Loss Functions/images/4b4b133f92acc8fe1275b0151cd7cf2812b251134ba4a88315460285ec584e6a.jpg)

![5c8d8ab7c5a19fa9492b590b1b14c20636e8ca342cf23de0404a6f0f7d7585c5.jpg](../iclr_results/915_Bounds on $L_p$ Errors in Density Ratio Estimation via $f$-Divergence Loss Functions/images/5c8d8ab7c5a19fa9492b590b1b14c20636e8ca342cf23de0404a6f0f7d7585c5.jpg)

![6f54b98368b6b979105026c1153baf7c6b7713c5b699b3b661492f3be88d0376.jpg](../iclr_results/915_Bounds on $L_p$ Errors in Density Ratio Estimation via $f$-Divergence Loss Functions/images/6f54b98368b6b979105026c1153baf7c6b7713c5b699b3b661492f3be88d0376.jpg)

![88ff3fdeddfb79fa566c8ed2f237e985186bf9ee3cc91fc55d4ad523cee4b7cd.jpg](../iclr_results/915_Bounds on $L_p$ Errors in Density Ratio Estimation via $f$-Divergence Loss Functions/images/88ff3fdeddfb79fa566c8ed2f237e985186bf9ee3cc91fc55d4ad523cee4b7cd.jpg)

### Tables

![399fe66be7e1ab1cf7f8a00a12b5781683f1af83f53b36d2c29952384250884a.jpg](../iclr_results/915_Bounds on $L_p$ Errors in Density Ratio Estimation via $f$-Divergence Loss Functions/tables/399fe66be7e1ab1cf7f8a00a12b5781683f1af83f53b36d2c29952384250884a.jpg)

![d52df5ef5e7f71fc657625bebc2eb5dd58e2bcd9f2bb944755b4f532b3eb4670.jpg](../iclr_results/915_Bounds on $L_p$ Errors in Density Ratio Estimation via $f$-Divergence Loss Functions/tables/d52df5ef5e7f71fc657625bebc2eb5dd58e2bcd9f2bb944755b4f532b3eb4670.jpg)

## Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathology


### Images

![285d3e2d4c08fccf5039d3086449fdd4f2741c24ec62abf2d3a62f0fa98e5ef7.jpg](../iclr_results/916_Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathol/images/285d3e2d4c08fccf5039d3086449fdd4f2741c24ec62abf2d3a62f0fa98e5ef7.jpg)

![3a52e903e15b079106e32eca92be2e614e0e4e30ee511de4031dbd866e96376c.jpg](../iclr_results/916_Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathol/images/3a52e903e15b079106e32eca92be2e614e0e4e30ee511de4031dbd866e96376c.jpg)

![aa8bbf69440fdfd44a28b74daf18f0fab6a5b860b12f251360b38e78effc594f.jpg](../iclr_results/916_Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathol/images/aa8bbf69440fdfd44a28b74daf18f0fab6a5b860b12f251360b38e78effc594f.jpg)

![cb230cd2071b0620e6a57a8177b91c72faf3bc2a73682cc8f43014e2dadff4dc.jpg](../iclr_results/916_Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathol/images/cb230cd2071b0620e6a57a8177b91c72faf3bc2a73682cc8f43014e2dadff4dc.jpg)

![cc62fcbdeaad771f95b66374534e0e1760426c02a45116ac442acf31ad28a1a8.jpg](../iclr_results/916_Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathol/images/cc62fcbdeaad771f95b66374534e0e1760426c02a45116ac442acf31ad28a1a8.jpg)

![cd82f062227a8ac27259d1119b6adf6cb038843290eb5c1445c093bfdb070031.jpg](../iclr_results/916_Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathol/images/cd82f062227a8ac27259d1119b6adf6cb038843290eb5c1445c093bfdb070031.jpg)

![eff82715bd2ec79ff5e60a81a9b1ea5f287ac4c3a91f1624d8fda81b1e5c58fc.jpg](../iclr_results/916_Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathol/images/eff82715bd2ec79ff5e60a81a9b1ea5f287ac4c3a91f1624d8fda81b1e5c58fc.jpg)

### Tables

![1ad3583981bfab76a1ffce07219132319bcc0b50f13959babf77e1d936a1e4c9.jpg](../iclr_results/916_Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathol/tables/1ad3583981bfab76a1ffce07219132319bcc0b50f13959babf77e1d936a1e4c9.jpg)

![4a552798fb9b9a53254ace95ce813d49b080a417c65043fa2e63cff2f4e77948.jpg](../iclr_results/916_Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathol/tables/4a552798fb9b9a53254ace95ce813d49b080a417c65043fa2e63cff2f4e77948.jpg)

![4d3b63a63145e3d83e521451282428935081107ec60992272a7fa4079dfafc96.jpg](../iclr_results/916_Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathol/tables/4d3b63a63145e3d83e521451282428935081107ec60992272a7fa4079dfafc96.jpg)

![5105a24ea2dd74802fe2ac4b4fd3517ad729ed8b2c1c3ed085a7d2a5fd01f093.jpg](../iclr_results/916_Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathol/tables/5105a24ea2dd74802fe2ac4b4fd3517ad729ed8b2c1c3ed085a7d2a5fd01f093.jpg)

![66a490fd50c888ac2b2eda1fc5c2badd40fdae6b0437160b3c8ca08e3ebc2e3b.jpg](../iclr_results/916_Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathol/tables/66a490fd50c888ac2b2eda1fc5c2badd40fdae6b0437160b3c8ca08e3ebc2e3b.jpg)

![7a59dd777e7317de75f76ffbd56ef6c294b960a6df94c35f15bfd3165dc710b6.jpg](../iclr_results/916_Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathol/tables/7a59dd777e7317de75f76ffbd56ef6c294b960a6df94c35f15bfd3165dc710b6.jpg)

![82d64ba60dc3c1f798af4ad96009390a303fb20f8582bea65d3d05011ad1c59b.jpg](../iclr_results/916_Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathol/tables/82d64ba60dc3c1f798af4ad96009390a303fb20f8582bea65d3d05011ad1c59b.jpg)

![8df5ed1d403fec3dc9300ba3d45e7c4e4a9ad72b649b649cfdac2643dd45cd0a.jpg](../iclr_results/916_Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathol/tables/8df5ed1d403fec3dc9300ba3d45e7c4e4a9ad72b649b649cfdac2643dd45cd0a.jpg)

![b50e5816427f19d8d6a141d85367ccc5f4778a36e18937e7f9160722c2f79b1b.jpg](../iclr_results/916_Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathol/tables/b50e5816427f19d8d6a141d85367ccc5f4778a36e18937e7f9160722c2f79b1b.jpg)

![bcca1e252c633bb8cb346d52ab47c3f7f0c764d3c34133fc611895e4ac85f1db.jpg](../iclr_results/916_Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathol/tables/bcca1e252c633bb8cb346d52ab47c3f7f0c764d3c34133fc611895e4ac85f1db.jpg)

![bfc56b94621db3efde4352f5266134de2e3db12b2739302eff390fe79f65210b.jpg](../iclr_results/916_Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathol/tables/bfc56b94621db3efde4352f5266134de2e3db12b2739302eff390fe79f65210b.jpg)

![c772783db0d123048e7f5a21edece1dc29d77410032ceb3c1cfd1a43b17cf3a3.jpg](../iclr_results/916_Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathol/tables/c772783db0d123048e7f5a21edece1dc29d77410032ceb3c1cfd1a43b17cf3a3.jpg)

![d03cccd482106e73ba8b1aac39ada5c8dc904ecedbdbf83cb0118b013854ddf6.jpg](../iclr_results/916_Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathol/tables/d03cccd482106e73ba8b1aac39ada5c8dc904ecedbdbf83cb0118b013854ddf6.jpg)

![d37a25cbdc1b964cef0deea2d6f850611251a1037e7986de45293caa52f74e01.jpg](../iclr_results/916_Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathol/tables/d37a25cbdc1b964cef0deea2d6f850611251a1037e7986de45293caa52f74e01.jpg)

## Accelerating Neural ODEs: A Variational Formulation-based Approach


### Images

![11e97eac8d1c41d8fc85089ba7ada58c8f24ef6eb052e0735e9a7f391fdc5617.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/images/11e97eac8d1c41d8fc85089ba7ada58c8f24ef6eb052e0735e9a7f391fdc5617.jpg)

![171562a3977ff5d84b9e48ca0d1836ff02ae91e8a10a58c4f82ee4b1121d5744.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/images/171562a3977ff5d84b9e48ca0d1836ff02ae91e8a10a58c4f82ee4b1121d5744.jpg)

![3aa7b4609c0fab9becfd65db47d9b38e9d827ef43e24cb7b8deeff34df7143d8.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/images/3aa7b4609c0fab9becfd65db47d9b38e9d827ef43e24cb7b8deeff34df7143d8.jpg)

![403125b08dd02551ee659e7cf0d247fed745cc531a58c26df49e99dd005ec280.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/images/403125b08dd02551ee659e7cf0d247fed745cc531a58c26df49e99dd005ec280.jpg)

![58b26a1a8300bcd5bfaa4944369da58e1bfdaebc6037167bc9deb4a8689ae596.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/images/58b26a1a8300bcd5bfaa4944369da58e1bfdaebc6037167bc9deb4a8689ae596.jpg)

![5b28f1a854541f01b6a5f8ecbe3c9746562e5d428065ce3ba19f20e6aa436ca8.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/images/5b28f1a854541f01b6a5f8ecbe3c9746562e5d428065ce3ba19f20e6aa436ca8.jpg)

![61bd6bae916b001b86db2ae9c7410ed4d28200cf90043427b144f774108c8d68.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/images/61bd6bae916b001b86db2ae9c7410ed4d28200cf90043427b144f774108c8d68.jpg)

![6a99c002cf3b31f876297235b2cd8e34c051fd533ba6f24ab06e0f367c1915fa.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/images/6a99c002cf3b31f876297235b2cd8e34c051fd533ba6f24ab06e0f367c1915fa.jpg)

![70d1433d6f4e15c94d47f0bc4f40a3038f097fe00abbfb25ee245ae3eb3709d6.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/images/70d1433d6f4e15c94d47f0bc4f40a3038f097fe00abbfb25ee245ae3eb3709d6.jpg)

![9f7bcd5b4882dd0d1cd612a5f548515a6edef9920741df9d14868e6142d8bde8.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/images/9f7bcd5b4882dd0d1cd612a5f548515a6edef9920741df9d14868e6142d8bde8.jpg)

![a7e7a551534ba12e183aa4a112d559cd86c144dc3816854ae962c287ce055f2f.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/images/a7e7a551534ba12e183aa4a112d559cd86c144dc3816854ae962c287ce055f2f.jpg)

![b00de3c405b053498c7b592717d431bb36889457ca98335bbdde8ee1c93dbf44.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/images/b00de3c405b053498c7b592717d431bb36889457ca98335bbdde8ee1c93dbf44.jpg)

![bd23bd03535532aa80ac1c5e3d93106ea229a3f1710238a5e7483cae70c6c290.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/images/bd23bd03535532aa80ac1c5e3d93106ea229a3f1710238a5e7483cae70c6c290.jpg)

![c3697f0775fd542b0bdff4346a256bbedc735e36163274640e000bf5e467b761.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/images/c3697f0775fd542b0bdff4346a256bbedc735e36163274640e000bf5e467b761.jpg)

![e91386e2b48f8c08d0dc418de9ad67cc35ede6e10d836e2b4afa73ba1a2bb8f6.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/images/e91386e2b48f8c08d0dc418de9ad67cc35ede6e10d836e2b4afa73ba1a2bb8f6.jpg)

![fef9b27d21d9b47e546cd0cb9b36cc95519442a5bd953546f37b072275fb1950.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/images/fef9b27d21d9b47e546cd0cb9b36cc95519442a5bd953546f37b072275fb1950.jpg)

### Tables

![14b0c1563a605f88a455ff41cbb43c6431dd23dd2fddfec89930b7bb0817745a.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/tables/14b0c1563a605f88a455ff41cbb43c6431dd23dd2fddfec89930b7bb0817745a.jpg)

![446a6134f96d270af5a35915c88b239c8032ad0b6c5d8827e46bf82cf3d3ea04.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/tables/446a6134f96d270af5a35915c88b239c8032ad0b6c5d8827e46bf82cf3d3ea04.jpg)

![461a15c30276a25b562c833f58fb1fed350fd5309a11376dc084d097295892fd.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/tables/461a15c30276a25b562c833f58fb1fed350fd5309a11376dc084d097295892fd.jpg)

![47d16f9e9aed854c2eb1243d478331196cce73a49e6297493b01574695a74fbd.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/tables/47d16f9e9aed854c2eb1243d478331196cce73a49e6297493b01574695a74fbd.jpg)

![6175cc1af02d15df9ba8d46fad1f903de0fccb195e5564e950d1b94db98839d7.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/tables/6175cc1af02d15df9ba8d46fad1f903de0fccb195e5564e950d1b94db98839d7.jpg)

![665945aceffed385597b4a1b66e1ac1ff2e0e96ff76875c49fb22d4c44922af1.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/tables/665945aceffed385597b4a1b66e1ac1ff2e0e96ff76875c49fb22d4c44922af1.jpg)

![67fff378357cc3b112c530ec35fc863ebf19866c0bccdcf5f444e55116f069b0.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/tables/67fff378357cc3b112c530ec35fc863ebf19866c0bccdcf5f444e55116f069b0.jpg)

![7b2441b6f1c1988fd271ae3cfaeca287c8776afc172da7ffbf1a4eaf1713f03e.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/tables/7b2441b6f1c1988fd271ae3cfaeca287c8776afc172da7ffbf1a4eaf1713f03e.jpg)

![7b39e27e33858051302babd0bc313f24382571adb0678c1c85076f950176ab46.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/tables/7b39e27e33858051302babd0bc313f24382571adb0678c1c85076f950176ab46.jpg)

![92610e3907c32ac438ab89f1d0f98910e24359899bc5cdeac518a0f44322e0ae.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/tables/92610e3907c32ac438ab89f1d0f98910e24359899bc5cdeac518a0f44322e0ae.jpg)

![9697afdff28e8a0d589f75643cd9fb713c974b4f815ef5180808b6281f2647b8.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/tables/9697afdff28e8a0d589f75643cd9fb713c974b4f815ef5180808b6281f2647b8.jpg)

![a068510abb0f603a47ec015b3a0b57c66891b06020dc25c1d3a8890693a73c7f.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/tables/a068510abb0f603a47ec015b3a0b57c66891b06020dc25c1d3a8890693a73c7f.jpg)

![d2c7310f9421addcf6d46e5bb7285823c27a88cdcc9ed73c7dcd8a921b60230f.jpg](../iclr_results/917_Accelerating Neural ODEs_ A Variational Formulation-based Approach/tables/d2c7310f9421addcf6d46e5bb7285823c27a88cdcc9ed73c7dcd8a921b60230f.jpg)

## RainbowPO: A Unified Framework for Combining Improvements in Preference Optimization


### Images

![2dadc280765c29a67e682ff3878b684d96781a3f4ff76fa330de3a9a147a0c14.jpg](../iclr_results/918_RainbowPO_ A Unified Framework for Combining Improvements in Preference Optimization/images/2dadc280765c29a67e682ff3878b684d96781a3f4ff76fa330de3a9a147a0c14.jpg)

![81d226b9e926d6b5baf1bd4fad7f7025d7e83144218f7c55e2d3c87837abc74a.jpg](../iclr_results/918_RainbowPO_ A Unified Framework for Combining Improvements in Preference Optimization/images/81d226b9e926d6b5baf1bd4fad7f7025d7e83144218f7c55e2d3c87837abc74a.jpg)

### Tables

![12cba2fd4d0e4a6ccf253f328a859ca1802127418614077e2f752f4e2e1c571b.jpg](../iclr_results/918_RainbowPO_ A Unified Framework for Combining Improvements in Preference Optimization/tables/12cba2fd4d0e4a6ccf253f328a859ca1802127418614077e2f752f4e2e1c571b.jpg)

![1af4b735173fe96d178ea17f39082d751f6e72f882e5447fa775c86fffb833b8.jpg](../iclr_results/918_RainbowPO_ A Unified Framework for Combining Improvements in Preference Optimization/tables/1af4b735173fe96d178ea17f39082d751f6e72f882e5447fa775c86fffb833b8.jpg)

![1c29c6565267ebbe145202b5c5efda590a2e71340b36c5cda0d43df1c135b10c.jpg](../iclr_results/918_RainbowPO_ A Unified Framework for Combining Improvements in Preference Optimization/tables/1c29c6565267ebbe145202b5c5efda590a2e71340b36c5cda0d43df1c135b10c.jpg)

![2059625e3fc971108f1714f40288b70a509be54aeade7ff97c776850c1601aff.jpg](../iclr_results/918_RainbowPO_ A Unified Framework for Combining Improvements in Preference Optimization/tables/2059625e3fc971108f1714f40288b70a509be54aeade7ff97c776850c1601aff.jpg)

![3c1d3f5133032b2f84ea37441072159492c7548e90879d931f091873047dd0d8.jpg](../iclr_results/918_RainbowPO_ A Unified Framework for Combining Improvements in Preference Optimization/tables/3c1d3f5133032b2f84ea37441072159492c7548e90879d931f091873047dd0d8.jpg)

![4d1de46ea79999d296897c24af482cfb34c8b9983c851d287095fa7585fbb036.jpg](../iclr_results/918_RainbowPO_ A Unified Framework for Combining Improvements in Preference Optimization/tables/4d1de46ea79999d296897c24af482cfb34c8b9983c851d287095fa7585fbb036.jpg)

![9363896a042f61a9cda6f0db93f9e0b3363e80cec29b7b19675d85061c3981d9.jpg](../iclr_results/918_RainbowPO_ A Unified Framework for Combining Improvements in Preference Optimization/tables/9363896a042f61a9cda6f0db93f9e0b3363e80cec29b7b19675d85061c3981d9.jpg)

![9ec01e06567fc1de394d258d5c0c16244ac76e6d273a053cfbe49388f7ff721e.jpg](../iclr_results/918_RainbowPO_ A Unified Framework for Combining Improvements in Preference Optimization/tables/9ec01e06567fc1de394d258d5c0c16244ac76e6d273a053cfbe49388f7ff721e.jpg)

![a9efb718162761c91eb47c3c38d623bf48555facc0976d592943eade68a1853e.jpg](../iclr_results/918_RainbowPO_ A Unified Framework for Combining Improvements in Preference Optimization/tables/a9efb718162761c91eb47c3c38d623bf48555facc0976d592943eade68a1853e.jpg)

![aaf190c17f51ebbf3746ebc7c4e850ed6e1e402514b47b6f35e0916248ad5df3.jpg](../iclr_results/918_RainbowPO_ A Unified Framework for Combining Improvements in Preference Optimization/tables/aaf190c17f51ebbf3746ebc7c4e850ed6e1e402514b47b6f35e0916248ad5df3.jpg)

![bc1e01ed69d4f6263f18a7f562475e355b481f0740962c2eae0f7118eb41411a.jpg](../iclr_results/918_RainbowPO_ A Unified Framework for Combining Improvements in Preference Optimization/tables/bc1e01ed69d4f6263f18a7f562475e355b481f0740962c2eae0f7118eb41411a.jpg)

![be65cb90ca47c3f11db8aaff820f79a7202e692d094b62cf74c84762274be4d9.jpg](../iclr_results/918_RainbowPO_ A Unified Framework for Combining Improvements in Preference Optimization/tables/be65cb90ca47c3f11db8aaff820f79a7202e692d094b62cf74c84762274be4d9.jpg)

![cbf93b97e749fe852e38a2fbfde091c98986005e2a9468e09c288543f9502708.jpg](../iclr_results/918_RainbowPO_ A Unified Framework for Combining Improvements in Preference Optimization/tables/cbf93b97e749fe852e38a2fbfde091c98986005e2a9468e09c288543f9502708.jpg)

## Physics-Informed Diffusion Models


### Images

![3bacd82c0383e3f975e78834f10f43e452f6ad55fd5b7d17e92c847f8b6f4e70.jpg](../iclr_results/919_Physics-Informed Diffusion Models/images/3bacd82c0383e3f975e78834f10f43e452f6ad55fd5b7d17e92c847f8b6f4e70.jpg)

![5ab93f7e379a5fc459793415d73f31be433c1e43a855414c1a6bdcb8dd4a68da.jpg](../iclr_results/919_Physics-Informed Diffusion Models/images/5ab93f7e379a5fc459793415d73f31be433c1e43a855414c1a6bdcb8dd4a68da.jpg)

![5feca872eecd77095d49b3614bcf36bd76055ca38b8f00ae402d019a261321dd.jpg](../iclr_results/919_Physics-Informed Diffusion Models/images/5feca872eecd77095d49b3614bcf36bd76055ca38b8f00ae402d019a261321dd.jpg)

![6ba8becb9ac14cd94187dd5d1c751fcfc97de06e53da80f013755e5300592270.jpg](../iclr_results/919_Physics-Informed Diffusion Models/images/6ba8becb9ac14cd94187dd5d1c751fcfc97de06e53da80f013755e5300592270.jpg)

![6e4608a1a7c0bea68b726d660462936d63787e280e051dd2b3b727690dd93f4b.jpg](../iclr_results/919_Physics-Informed Diffusion Models/images/6e4608a1a7c0bea68b726d660462936d63787e280e051dd2b3b727690dd93f4b.jpg)

![7abb9e3605ff90811646b913ca7e7c2eebf0aee8a527f87ac39193025ee9e53d.jpg](../iclr_results/919_Physics-Informed Diffusion Models/images/7abb9e3605ff90811646b913ca7e7c2eebf0aee8a527f87ac39193025ee9e53d.jpg)

![944e5be8c186b8d58355594bd519682059c8fcba1daac082a02e49bc7a07db71.jpg](../iclr_results/919_Physics-Informed Diffusion Models/images/944e5be8c186b8d58355594bd519682059c8fcba1daac082a02e49bc7a07db71.jpg)

![a40d2d56460ccc03eb7837a7bb432a96837825ae6fbc728c6c81f41089f29ad4.jpg](../iclr_results/919_Physics-Informed Diffusion Models/images/a40d2d56460ccc03eb7837a7bb432a96837825ae6fbc728c6c81f41089f29ad4.jpg)

![fb3d5f00af87a0752f4879f70222159af1d8ac18e6e4cd78db332e1ada8c19f1.jpg](../iclr_results/919_Physics-Informed Diffusion Models/images/fb3d5f00af87a0752f4879f70222159af1d8ac18e6e4cd78db332e1ada8c19f1.jpg)

### Tables

![90b6e6b5ed49f84399b12585c8a803b9ff2eb7094fd9a1fd16d58516ac32811a.jpg](../iclr_results/919_Physics-Informed Diffusion Models/tables/90b6e6b5ed49f84399b12585c8a803b9ff2eb7094fd9a1fd16d58516ac32811a.jpg)

![b60912fd59111238cb2a1533c9744883565258dd8d5f649ca25203ab3e6509cf.jpg](../iclr_results/919_Physics-Informed Diffusion Models/tables/b60912fd59111238cb2a1533c9744883565258dd8d5f649ca25203ab3e6509cf.jpg)

![fda2d399a2adca30b21a40ca68f71cdb1576fde3d4771e8a4e5cb9681311a236.jpg](../iclr_results/919_Physics-Informed Diffusion Models/tables/fda2d399a2adca30b21a40ca68f71cdb1576fde3d4771e8a4e5cb9681311a236.jpg)

## ProAdvPrompter: A Two-Stage Journey to Effective Adversarial Prompting for LLMs


### Images

![19de3dfb58886a8490769c51ffe2521187c2705ee6a523cbbe15c39c31d1b4b8.jpg](../iclr_results/920_ProAdvPrompter_ A Two-Stage Journey to Effective Adversarial Prompting for LLMs/images/19de3dfb58886a8490769c51ffe2521187c2705ee6a523cbbe15c39c31d1b4b8.jpg)

![4dd1049721964bcb11cfa52c7bc497598095a9e27f508e60ad913c5b471cfd9c.jpg](../iclr_results/920_ProAdvPrompter_ A Two-Stage Journey to Effective Adversarial Prompting for LLMs/images/4dd1049721964bcb11cfa52c7bc497598095a9e27f508e60ad913c5b471cfd9c.jpg)

![6992a73ababddac9cab642db329243834aeb09e121b325f551891da524ed5cdb.jpg](../iclr_results/920_ProAdvPrompter_ A Two-Stage Journey to Effective Adversarial Prompting for LLMs/images/6992a73ababddac9cab642db329243834aeb09e121b325f551891da524ed5cdb.jpg)

![a42517aa9a420633e370c438c04f1971081961b2e4c93b70fc88974b17e4ea1f.jpg](../iclr_results/920_ProAdvPrompter_ A Two-Stage Journey to Effective Adversarial Prompting for LLMs/images/a42517aa9a420633e370c438c04f1971081961b2e4c93b70fc88974b17e4ea1f.jpg)

![ad2a160005ce75c5a87fa8cf77bc09f0e0d052bd0d0389fd78772ead5d9b6549.jpg](../iclr_results/920_ProAdvPrompter_ A Two-Stage Journey to Effective Adversarial Prompting for LLMs/images/ad2a160005ce75c5a87fa8cf77bc09f0e0d052bd0d0389fd78772ead5d9b6549.jpg)

![bc10966b22bf89a9ffccf5c947193e266906ca186d2f09c27dd469e3ce45d71c.jpg](../iclr_results/920_ProAdvPrompter_ A Two-Stage Journey to Effective Adversarial Prompting for LLMs/images/bc10966b22bf89a9ffccf5c947193e266906ca186d2f09c27dd469e3ce45d71c.jpg)

![ed86762d1abbe028e440e8566f18df8206891abb113d58313977d8b3ab38b7b1.jpg](../iclr_results/920_ProAdvPrompter_ A Two-Stage Journey to Effective Adversarial Prompting for LLMs/images/ed86762d1abbe028e440e8566f18df8206891abb113d58313977d8b3ab38b7b1.jpg)

![f1b1f6ea022ad41e694d3db971d887fc844834b8072a10284e95457b20a051f8.jpg](../iclr_results/920_ProAdvPrompter_ A Two-Stage Journey to Effective Adversarial Prompting for LLMs/images/f1b1f6ea022ad41e694d3db971d887fc844834b8072a10284e95457b20a051f8.jpg)

![f6c0c2048bb6d021fcca2a510570930606f7273826e3c3bacf5730d8340b76df.jpg](../iclr_results/920_ProAdvPrompter_ A Two-Stage Journey to Effective Adversarial Prompting for LLMs/images/f6c0c2048bb6d021fcca2a510570930606f7273826e3c3bacf5730d8340b76df.jpg)

![f9c7bbac6d2c3ea1f6e74a71abfb6c8ec9a8e4f82725fb7bcf9021c6b9ebb48b.jpg](../iclr_results/920_ProAdvPrompter_ A Two-Stage Journey to Effective Adversarial Prompting for LLMs/images/f9c7bbac6d2c3ea1f6e74a71abfb6c8ec9a8e4f82725fb7bcf9021c6b9ebb48b.jpg)

### Tables

![00b374e3b01ac6a6be270a98c9c97c8d4e8027fa80d90e51447dac6e958b48c9.jpg](../iclr_results/920_ProAdvPrompter_ A Two-Stage Journey to Effective Adversarial Prompting for LLMs/tables/00b374e3b01ac6a6be270a98c9c97c8d4e8027fa80d90e51447dac6e958b48c9.jpg)

![2581eee736b160cf35e5c492ac07a7430c3f6fc20beeef1e8bc23b07467cc690.jpg](../iclr_results/920_ProAdvPrompter_ A Two-Stage Journey to Effective Adversarial Prompting for LLMs/tables/2581eee736b160cf35e5c492ac07a7430c3f6fc20beeef1e8bc23b07467cc690.jpg)

![4ab95aa3f03f8d67ab77c44b0656852ecc9ade655b0d90ac2791d028305b0622.jpg](../iclr_results/920_ProAdvPrompter_ A Two-Stage Journey to Effective Adversarial Prompting for LLMs/tables/4ab95aa3f03f8d67ab77c44b0656852ecc9ade655b0d90ac2791d028305b0622.jpg)

![61d43938bcec00b25c676ffaa70b22d90d061839d56480d27eb300dcfcb8b8b7.jpg](../iclr_results/920_ProAdvPrompter_ A Two-Stage Journey to Effective Adversarial Prompting for LLMs/tables/61d43938bcec00b25c676ffaa70b22d90d061839d56480d27eb300dcfcb8b8b7.jpg)

![642eda36a4fbbe6bcf6e4282870a22380f8d20bbfba78b46d6cb00e0cc39fb57.jpg](../iclr_results/920_ProAdvPrompter_ A Two-Stage Journey to Effective Adversarial Prompting for LLMs/tables/642eda36a4fbbe6bcf6e4282870a22380f8d20bbfba78b46d6cb00e0cc39fb57.jpg)

![7060a4af7ad197e5f801620548c2a33a88567752fea592709845ebcbfa1b82f2.jpg](../iclr_results/920_ProAdvPrompter_ A Two-Stage Journey to Effective Adversarial Prompting for LLMs/tables/7060a4af7ad197e5f801620548c2a33a88567752fea592709845ebcbfa1b82f2.jpg)

![7308762a56de32ef5cf2cf21ee0f403b89d4dfcf55efc286129fd0598f7e45ef.jpg](../iclr_results/920_ProAdvPrompter_ A Two-Stage Journey to Effective Adversarial Prompting for LLMs/tables/7308762a56de32ef5cf2cf21ee0f403b89d4dfcf55efc286129fd0598f7e45ef.jpg)

![7648c8f7d95299435a1920027e4bb86b397a0e0847ce269ed5080c0aa3f0d371.jpg](../iclr_results/920_ProAdvPrompter_ A Two-Stage Journey to Effective Adversarial Prompting for LLMs/tables/7648c8f7d95299435a1920027e4bb86b397a0e0847ce269ed5080c0aa3f0d371.jpg)

![948686efa479d23b8cee41df92a022e46b6de3c360ed49e2c6dcdd01fcf94881.jpg](../iclr_results/920_ProAdvPrompter_ A Two-Stage Journey to Effective Adversarial Prompting for LLMs/tables/948686efa479d23b8cee41df92a022e46b6de3c360ed49e2c6dcdd01fcf94881.jpg)

![b23b72d3470b2020f661aef9d60f6d9d8afa372412fd10c9e05c774ee5114c63.jpg](../iclr_results/920_ProAdvPrompter_ A Two-Stage Journey to Effective Adversarial Prompting for LLMs/tables/b23b72d3470b2020f661aef9d60f6d9d8afa372412fd10c9e05c774ee5114c63.jpg)

![d27d02e3359d99df42a09759bd25c7d5701840376590355ade02bd4fa0a0984a.jpg](../iclr_results/920_ProAdvPrompter_ A Two-Stage Journey to Effective Adversarial Prompting for LLMs/tables/d27d02e3359d99df42a09759bd25c7d5701840376590355ade02bd4fa0a0984a.jpg)

![d9febb85d70eb5dd05f2eed810ba7b113b56182329622790071699382545128c.jpg](../iclr_results/920_ProAdvPrompter_ A Two-Stage Journey to Effective Adversarial Prompting for LLMs/tables/d9febb85d70eb5dd05f2eed810ba7b113b56182329622790071699382545128c.jpg)

![ed3ff884bcaab14440bab6cd5df4d67572f0980640df55d34fb3a12797c15410.jpg](../iclr_results/920_ProAdvPrompter_ A Two-Stage Journey to Effective Adversarial Prompting for LLMs/tables/ed3ff884bcaab14440bab6cd5df4d67572f0980640df55d34fb3a12797c15410.jpg)

## Improving Large Language Model Planning with Action Sequence Similarity


### Images

![1fb61c6705ae95a2cdf6e05890b01589b99d2ad4788a059cde9b54e618174921.jpg](../iclr_results/921_Improving Large Language Model Planning with Action Sequence Similarity/images/1fb61c6705ae95a2cdf6e05890b01589b99d2ad4788a059cde9b54e618174921.jpg)

![22a67710623ab6fcbd03711a451371293cab1c9ec94c8c2318f7cb129b3b9690.jpg](../iclr_results/921_Improving Large Language Model Planning with Action Sequence Similarity/images/22a67710623ab6fcbd03711a451371293cab1c9ec94c8c2318f7cb129b3b9690.jpg)

![3154e5e56e2810360ed6f455c9441ac0d3548e045ea7830717241e7966a78f2f.jpg](../iclr_results/921_Improving Large Language Model Planning with Action Sequence Similarity/images/3154e5e56e2810360ed6f455c9441ac0d3548e045ea7830717241e7966a78f2f.jpg)

![4a4accba15642e5ebc55622ae4c36b165154299edd4e247ea65a32816c03fba4.jpg](../iclr_results/921_Improving Large Language Model Planning with Action Sequence Similarity/images/4a4accba15642e5ebc55622ae4c36b165154299edd4e247ea65a32816c03fba4.jpg)

![50fdedfe8ca5681819844360a2538da97766b1c738cdbd40479cee6bc9b8ef37.jpg](../iclr_results/921_Improving Large Language Model Planning with Action Sequence Similarity/images/50fdedfe8ca5681819844360a2538da97766b1c738cdbd40479cee6bc9b8ef37.jpg)

![686dfea11c91e83064bbec59608561760af5c61ef9c59b8e27b4dd6ecf54b909.jpg](../iclr_results/921_Improving Large Language Model Planning with Action Sequence Similarity/images/686dfea11c91e83064bbec59608561760af5c61ef9c59b8e27b4dd6ecf54b909.jpg)

![8ab9cbd8b1cfa839334b2e43c0c8c6f028978fdfc5d076883c5e4195138c953f.jpg](../iclr_results/921_Improving Large Language Model Planning with Action Sequence Similarity/images/8ab9cbd8b1cfa839334b2e43c0c8c6f028978fdfc5d076883c5e4195138c953f.jpg)

![ba3702b40da458ca0f6824e8f864a2f4799a32004e298b0c8c110cad9d9b8331.jpg](../iclr_results/921_Improving Large Language Model Planning with Action Sequence Similarity/images/ba3702b40da458ca0f6824e8f864a2f4799a32004e298b0c8c110cad9d9b8331.jpg)

![e32b3d70e3ef4f1ebf4948dcb50dfc57eb7bd27097015f7dada22bd23267384c.jpg](../iclr_results/921_Improving Large Language Model Planning with Action Sequence Similarity/images/e32b3d70e3ef4f1ebf4948dcb50dfc57eb7bd27097015f7dada22bd23267384c.jpg)

![eb07f3d340bc332a1d14d606035bb7c97c979ff2054f732363b0c312a72ca2d3.jpg](../iclr_results/921_Improving Large Language Model Planning with Action Sequence Similarity/images/eb07f3d340bc332a1d14d606035bb7c97c979ff2054f732363b0c312a72ca2d3.jpg)

![f9a06f7164e9a5e02b2201a5c7f6c867ec4f184de5e0ff2d5f825e8423df6ab4.jpg](../iclr_results/921_Improving Large Language Model Planning with Action Sequence Similarity/images/f9a06f7164e9a5e02b2201a5c7f6c867ec4f184de5e0ff2d5f825e8423df6ab4.jpg)

### Tables

![0cb36256573508ed17ea7725bda92a68420ddce996815f2f92986a1c871d2564.jpg](../iclr_results/921_Improving Large Language Model Planning with Action Sequence Similarity/tables/0cb36256573508ed17ea7725bda92a68420ddce996815f2f92986a1c871d2564.jpg)

![3d406805e3d33963641f1fd14a81520644a77015676af2ad441d2419d2dc1005.jpg](../iclr_results/921_Improving Large Language Model Planning with Action Sequence Similarity/tables/3d406805e3d33963641f1fd14a81520644a77015676af2ad441d2419d2dc1005.jpg)

![54d068dc7b6288b7247a4f15eb504bc1a6bfb602f40e394a20d9067efd6ff1a9.jpg](../iclr_results/921_Improving Large Language Model Planning with Action Sequence Similarity/tables/54d068dc7b6288b7247a4f15eb504bc1a6bfb602f40e394a20d9067efd6ff1a9.jpg)

![7b7a0c5cf1270cc217e6bb7dab9bbb39851efc45c90c051151600dec969555f2.jpg](../iclr_results/921_Improving Large Language Model Planning with Action Sequence Similarity/tables/7b7a0c5cf1270cc217e6bb7dab9bbb39851efc45c90c051151600dec969555f2.jpg)

![bf7106c15350b03b097e7412871982214f0a9cfeb173f611a0750bca9be3effa.jpg](../iclr_results/921_Improving Large Language Model Planning with Action Sequence Similarity/tables/bf7106c15350b03b097e7412871982214f0a9cfeb173f611a0750bca9be3effa.jpg)

![fac9068d9a2f1fcd0171c063efb2c984f6ead0bed8d3a67313332a0f1ea84010.jpg](../iclr_results/921_Improving Large Language Model Planning with Action Sequence Similarity/tables/fac9068d9a2f1fcd0171c063efb2c984f6ead0bed8d3a67313332a0f1ea84010.jpg)

## Hydra-SGG: Hybrid Relation Assignment for One-stage Scene Graph Generation


### Images

![0af5e5f5ce2a438d72d47393cc27f641521c9dc3839116234decaff0facaa318.jpg](../iclr_results/922_Hydra-SGG_ Hybrid Relation Assignment for One-stage Scene Graph Generation/images/0af5e5f5ce2a438d72d47393cc27f641521c9dc3839116234decaff0facaa318.jpg)

![1037901d5cc1f195f6cc89690714d2fb9f473606748b730ff49edb3ed64eb3d8.jpg](../iclr_results/922_Hydra-SGG_ Hybrid Relation Assignment for One-stage Scene Graph Generation/images/1037901d5cc1f195f6cc89690714d2fb9f473606748b730ff49edb3ed64eb3d8.jpg)

![2be1eb4636795c6c56190ae14c29fd183a7ed61157303fc6da012c0a13b8056d.jpg](../iclr_results/922_Hydra-SGG_ Hybrid Relation Assignment for One-stage Scene Graph Generation/images/2be1eb4636795c6c56190ae14c29fd183a7ed61157303fc6da012c0a13b8056d.jpg)

![6ed0f5cae7a209b5f7f51e720f83e0fce02694d05d9c8924c57253f8954c7cad.jpg](../iclr_results/922_Hydra-SGG_ Hybrid Relation Assignment for One-stage Scene Graph Generation/images/6ed0f5cae7a209b5f7f51e720f83e0fce02694d05d9c8924c57253f8954c7cad.jpg)

![705b6e2893a58504425e49976cbdda84095e0eb2722418afbb6ce8e8ebb92b2a.jpg](../iclr_results/922_Hydra-SGG_ Hybrid Relation Assignment for One-stage Scene Graph Generation/images/705b6e2893a58504425e49976cbdda84095e0eb2722418afbb6ce8e8ebb92b2a.jpg)

![ae9198ab3f73a3af3bb1869efa2e3a650ae399644939e9ff0fc68d84b3ae76d2.jpg](../iclr_results/922_Hydra-SGG_ Hybrid Relation Assignment for One-stage Scene Graph Generation/images/ae9198ab3f73a3af3bb1869efa2e3a650ae399644939e9ff0fc68d84b3ae76d2.jpg)

### Tables

![4bd04e3e9e4985bdc9930ce7c5c0da95132b738a50b47aad4216775c6595189a.jpg](../iclr_results/922_Hydra-SGG_ Hybrid Relation Assignment for One-stage Scene Graph Generation/tables/4bd04e3e9e4985bdc9930ce7c5c0da95132b738a50b47aad4216775c6595189a.jpg)

![5aa1f3d15978e2769c0757140dffe7ffd2b02a610a212abf49dc138b5a798f8e.jpg](../iclr_results/922_Hydra-SGG_ Hybrid Relation Assignment for One-stage Scene Graph Generation/tables/5aa1f3d15978e2769c0757140dffe7ffd2b02a610a212abf49dc138b5a798f8e.jpg)

![f1510552f725e982970307ed15e13e3600cdeecc6d212fcdd6597e0c8fc50d93.jpg](../iclr_results/922_Hydra-SGG_ Hybrid Relation Assignment for One-stage Scene Graph Generation/tables/f1510552f725e982970307ed15e13e3600cdeecc6d212fcdd6597e0c8fc50d93.jpg)

![fbc16c535904ea005b49010207fe7840f31e260d6992dcef7c2674b0d24b14cb.jpg](../iclr_results/922_Hydra-SGG_ Hybrid Relation Assignment for One-stage Scene Graph Generation/tables/fbc16c535904ea005b49010207fe7840f31e260d6992dcef7c2674b0d24b14cb.jpg)

## Heavy-Tailed Diffusion Models


### Images

![0948797c5b2c44dd1c4e93d3d0739c3edde73a95e5ec52a47e6c91172b3ebcad.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/images/0948797c5b2c44dd1c4e93d3d0739c3edde73a95e5ec52a47e6c91172b3ebcad.jpg)

![1d7fe3cf774cad518d86513f36a4ba775e4eeea1222199f1e38a3595d934a9e1.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/images/1d7fe3cf774cad518d86513f36a4ba775e4eeea1222199f1e38a3595d934a9e1.jpg)

![54ccbfd6de218e90eecf3870b562b75bbfb8712706236007a90804825f25fdd0.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/images/54ccbfd6de218e90eecf3870b562b75bbfb8712706236007a90804825f25fdd0.jpg)

![7a9555ad4355e9d081be274e3cd6c188608b421a1a113001b2a3fe0668295dc0.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/images/7a9555ad4355e9d081be274e3cd6c188608b421a1a113001b2a3fe0668295dc0.jpg)

![8544cb5d4910f908736b93012fe03b43a518f828875ad2b01d441273f05b2a0e.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/images/8544cb5d4910f908736b93012fe03b43a518f828875ad2b01d441273f05b2a0e.jpg)

![9f128f49eae179e910009e2bbfa8b7c435660dfa7c6d69cfe0711c5f37c0f612.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/images/9f128f49eae179e910009e2bbfa8b7c435660dfa7c6d69cfe0711c5f37c0f612.jpg)

![a7c0bf18a3f0bb763889c94d2f2436fc22e8a842392b6fbf2d37853fbd0b1c6f.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/images/a7c0bf18a3f0bb763889c94d2f2436fc22e8a842392b6fbf2d37853fbd0b1c6f.jpg)

![ce6dc63d4465e3eb4be4e6e0743a1afc650834401e49f42b523fa28e0359ca92.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/images/ce6dc63d4465e3eb4be4e6e0743a1afc650834401e49f42b523fa28e0359ca92.jpg)

![d1fc43c89629afa5a206481edfe0bed4635982dbc6da5a5a9f49199e27797daf.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/images/d1fc43c89629afa5a206481edfe0bed4635982dbc6da5a5a9f49199e27797daf.jpg)

![d52b9cc2ab77adbc1983d9d95e766ce2b02117d8cab851d39b4b24810f93e573.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/images/d52b9cc2ab77adbc1983d9d95e766ce2b02117d8cab851d39b4b24810f93e573.jpg)

![d81c32d85d4a7f92c2b7aba8f8779a442882bab11bdd4ae93028949fd5a90479.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/images/d81c32d85d4a7f92c2b7aba8f8779a442882bab11bdd4ae93028949fd5a90479.jpg)

![eabff4071a1954375d90d6ea30750317cf1a99dd1839a9bc33a76a54fc3c443c.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/images/eabff4071a1954375d90d6ea30750317cf1a99dd1839a9bc33a76a54fc3c443c.jpg)

![ec7edadce42267687354d9fa65e3747059a730ec158692bcc9492ab47e09e99c.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/images/ec7edadce42267687354d9fa65e3747059a730ec158692bcc9492ab47e09e99c.jpg)

![ee1a706bc9889fbf132a10b0107f6c4eb262b0016fdf4ee5289344bf0204e408.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/images/ee1a706bc9889fbf132a10b0107f6c4eb262b0016fdf4ee5289344bf0204e408.jpg)

![f91b43711191de24cb6fc4799f0f8c276a618bf8f27fa6366fcb396b8cb9d969.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/images/f91b43711191de24cb6fc4799f0f8c276a618bf8f27fa6366fcb396b8cb9d969.jpg)

![fc5f2c117c6dc66113b50ea3706df88d87ec096352fef66f94cfcb523385036e.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/images/fc5f2c117c6dc66113b50ea3706df88d87ec096352fef66f94cfcb523385036e.jpg)

### Tables

![0b92c14930c60880f111da7294a18f4904dcebd4f20b4aa2303f6d8d7586dcbf.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/tables/0b92c14930c60880f111da7294a18f4904dcebd4f20b4aa2303f6d8d7586dcbf.jpg)

![27edba3deb35265f61e2cb02e64d4e6be77d9a93bb85c2bb0ae4dcb8ded681db.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/tables/27edba3deb35265f61e2cb02e64d4e6be77d9a93bb85c2bb0ae4dcb8ded681db.jpg)

![37d73df5752066d04f5d4e8d94b8c34d0670e79d32c4e2e87860af0eb722ca2b.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/tables/37d73df5752066d04f5d4e8d94b8c34d0670e79d32c4e2e87860af0eb722ca2b.jpg)

![49eb8d18c0b5ec54180cf87c283e592a48b6dd43f699da1ce5e0266fb72725f9.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/tables/49eb8d18c0b5ec54180cf87c283e592a48b6dd43f699da1ce5e0266fb72725f9.jpg)

![840604c463fbca461a7c3dd8f61a105366bbabb73e26868971d4f4e100148c04.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/tables/840604c463fbca461a7c3dd8f61a105366bbabb73e26868971d4f4e100148c04.jpg)

![b2c59316c6b90336ddd0b615d798997736fe19c3019fcfc1bf2c9211801799ae.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/tables/b2c59316c6b90336ddd0b615d798997736fe19c3019fcfc1bf2c9211801799ae.jpg)

![e139b0de03d3a986ed2716a36c956009cf3236ed34013205100fa0e0c1bbddc5.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/tables/e139b0de03d3a986ed2716a36c956009cf3236ed34013205100fa0e0c1bbddc5.jpg)

![e1f645b875e653d54e498f630128da7ab08360f6ed9b36683108f61535f1e85d.jpg](../iclr_results/923_Heavy-Tailed Diffusion Models/tables/e1f645b875e653d54e498f630128da7ab08360f6ed9b36683108f61535f1e85d.jpg)

## Rethinking Multiple-Instance Learning From Feature Space to Probability Space


### Images

![11e8d438b2ef6e30686bce78729e77cc6309c6961bc595efb3c8792d08b23f20.jpg](../iclr_results/924_Rethinking Multiple-Instance Learning From Feature Space to Probability Space/images/11e8d438b2ef6e30686bce78729e77cc6309c6961bc595efb3c8792d08b23f20.jpg)

![36ff2a3f7fd3b151362878fdb86ca8474247fd6e6d2731ec69f19d3d39f307e6.jpg](../iclr_results/924_Rethinking Multiple-Instance Learning From Feature Space to Probability Space/images/36ff2a3f7fd3b151362878fdb86ca8474247fd6e6d2731ec69f19d3d39f307e6.jpg)

![753934dde5ca7860bbf22c6bd35d94066a28fc0ae10653debfe24d98c9ee051c.jpg](../iclr_results/924_Rethinking Multiple-Instance Learning From Feature Space to Probability Space/images/753934dde5ca7860bbf22c6bd35d94066a28fc0ae10653debfe24d98c9ee051c.jpg)

![7a4e15cf743af775c5d3559af5ca71527aca4b8738641cd31cfbb2e66ccf649b.jpg](../iclr_results/924_Rethinking Multiple-Instance Learning From Feature Space to Probability Space/images/7a4e15cf743af775c5d3559af5ca71527aca4b8738641cd31cfbb2e66ccf649b.jpg)

![8613dc410234efa7876ee8eed0a0f9133a0e07cf9cb88bc36221101a8210a435.jpg](../iclr_results/924_Rethinking Multiple-Instance Learning From Feature Space to Probability Space/images/8613dc410234efa7876ee8eed0a0f9133a0e07cf9cb88bc36221101a8210a435.jpg)

![b8e1936d802d5c639391664e16329562a197985e89cc05519294d7a5ba809a41.jpg](../iclr_results/924_Rethinking Multiple-Instance Learning From Feature Space to Probability Space/images/b8e1936d802d5c639391664e16329562a197985e89cc05519294d7a5ba809a41.jpg)

![bd932094b5c6598d4c60d17184a8830f037a5635cb5f75efa9eb5fe55ad34fc0.jpg](../iclr_results/924_Rethinking Multiple-Instance Learning From Feature Space to Probability Space/images/bd932094b5c6598d4c60d17184a8830f037a5635cb5f75efa9eb5fe55ad34fc0.jpg)

![d2d85761cc4d1d5da1b4acab7ba15345e123b1e93ef0662e676eec2693d150ae.jpg](../iclr_results/924_Rethinking Multiple-Instance Learning From Feature Space to Probability Space/images/d2d85761cc4d1d5da1b4acab7ba15345e123b1e93ef0662e676eec2693d150ae.jpg)

![fba3e91e7084ef26abc2ac73d76836be34d4e7b8aec81ef4ca41b3d552edcc48.jpg](../iclr_results/924_Rethinking Multiple-Instance Learning From Feature Space to Probability Space/images/fba3e91e7084ef26abc2ac73d76836be34d4e7b8aec81ef4ca41b3d552edcc48.jpg)

### Tables

![1b151bc2116c81957ccec0ef35e804be5f18926e46fdc9cc00e2b7ab7e7bd573.jpg](../iclr_results/924_Rethinking Multiple-Instance Learning From Feature Space to Probability Space/tables/1b151bc2116c81957ccec0ef35e804be5f18926e46fdc9cc00e2b7ab7e7bd573.jpg)

![2a984ce7e4e0c30c9be2e6c45ed0a93c3384890cc54b7e318bae8324870d8036.jpg](../iclr_results/924_Rethinking Multiple-Instance Learning From Feature Space to Probability Space/tables/2a984ce7e4e0c30c9be2e6c45ed0a93c3384890cc54b7e318bae8324870d8036.jpg)

![516a574c9aacf406ff9539993bc4a6bbde429a4968a913f5df200dbb35c32a81.jpg](../iclr_results/924_Rethinking Multiple-Instance Learning From Feature Space to Probability Space/tables/516a574c9aacf406ff9539993bc4a6bbde429a4968a913f5df200dbb35c32a81.jpg)

![7109c0db8a5abb21a06e532d1a7cdbc3c207e93e6e837a518b2d6b783162089a.jpg](../iclr_results/924_Rethinking Multiple-Instance Learning From Feature Space to Probability Space/tables/7109c0db8a5abb21a06e532d1a7cdbc3c207e93e6e837a518b2d6b783162089a.jpg)

![81561e6c690d04b5b18eb6b372db87c72bd619df55031436f534bd09e5b65f61.jpg](../iclr_results/924_Rethinking Multiple-Instance Learning From Feature Space to Probability Space/tables/81561e6c690d04b5b18eb6b372db87c72bd619df55031436f534bd09e5b65f61.jpg)

![f3e0a5b26fb85ffc8d7dcdda778f3d94b0a72ce516b8a004c6d7e5e0af083763.jpg](../iclr_results/924_Rethinking Multiple-Instance Learning From Feature Space to Probability Space/tables/f3e0a5b26fb85ffc8d7dcdda778f3d94b0a72ce516b8a004c6d7e5e0af083763.jpg)

![ff0f8b2109be57f6c2278446c3c66fdd233e73df5302fcb0e0a5cbeabe14cde5.jpg](../iclr_results/924_Rethinking Multiple-Instance Learning From Feature Space to Probability Space/tables/ff0f8b2109be57f6c2278446c3c66fdd233e73df5302fcb0e0a5cbeabe14cde5.jpg)

## KGARevion: An AI Agent for Knowledge-Intensive Biomedical QA


### Images

![1471ef2c39d8925d670db1f9d6b4454d648fc332ac2940e3168ba052a0dd89f0.jpg](../iclr_results/925_KGARevion_ An AI Agent for Knowledge-Intensive Biomedical QA/images/1471ef2c39d8925d670db1f9d6b4454d648fc332ac2940e3168ba052a0dd89f0.jpg)

![5a4f8880dfb3454ec621a85a4d8ecc4b4d34cead208dd7983db4c261d3f51acb.jpg](../iclr_results/925_KGARevion_ An AI Agent for Knowledge-Intensive Biomedical QA/images/5a4f8880dfb3454ec621a85a4d8ecc4b4d34cead208dd7983db4c261d3f51acb.jpg)

![85c638d0745594b7d84b21741e00dd4a1a6d8a695b39a9e6bd682c283fd70c6d.jpg](../iclr_results/925_KGARevion_ An AI Agent for Knowledge-Intensive Biomedical QA/images/85c638d0745594b7d84b21741e00dd4a1a6d8a695b39a9e6bd682c283fd70c6d.jpg)

![8cc6b1ae8a53910e9aa1f3807b4449710db61d08a6f5f62559df090f654f887a.jpg](../iclr_results/925_KGARevion_ An AI Agent for Knowledge-Intensive Biomedical QA/images/8cc6b1ae8a53910e9aa1f3807b4449710db61d08a6f5f62559df090f654f887a.jpg)

![9956a023839274f1896b5026e21bb1ebf8514399f07a6eaa507e8283fa6375f1.jpg](../iclr_results/925_KGARevion_ An AI Agent for Knowledge-Intensive Biomedical QA/images/9956a023839274f1896b5026e21bb1ebf8514399f07a6eaa507e8283fa6375f1.jpg)

![c73002bb471d0cc63deeecff557c7e4cd45f93cdc2fd8931afa39e0c6ce68f4f.jpg](../iclr_results/925_KGARevion_ An AI Agent for Knowledge-Intensive Biomedical QA/images/c73002bb471d0cc63deeecff557c7e4cd45f93cdc2fd8931afa39e0c6ce68f4f.jpg)

![cf789fc095a5a664b08167d3731f9cb7de9c32aff7361599ea5c4a3f1ae4b719.jpg](../iclr_results/925_KGARevion_ An AI Agent for Knowledge-Intensive Biomedical QA/images/cf789fc095a5a664b08167d3731f9cb7de9c32aff7361599ea5c4a3f1ae4b719.jpg)

![f449e325ed7e485b5a3ef8efe5c2810e262bcb4a611b8c9c9d9a26360deeedca.jpg](../iclr_results/925_KGARevion_ An AI Agent for Knowledge-Intensive Biomedical QA/images/f449e325ed7e485b5a3ef8efe5c2810e262bcb4a611b8c9c9d9a26360deeedca.jpg)

### Tables

![129ad75c09f858b05563fede0ab72d620f059e7f3399afa7527ccb43bd3f74ac.jpg](../iclr_results/925_KGARevion_ An AI Agent for Knowledge-Intensive Biomedical QA/tables/129ad75c09f858b05563fede0ab72d620f059e7f3399afa7527ccb43bd3f74ac.jpg)

![282ea84b9fd7636db3908c930ed6d3b969b5f7937cc0b4595a4de8563a352516.jpg](../iclr_results/925_KGARevion_ An AI Agent for Knowledge-Intensive Biomedical QA/tables/282ea84b9fd7636db3908c930ed6d3b969b5f7937cc0b4595a4de8563a352516.jpg)

![3e3b0d6ad7ebc0b015aa2fe7f0a0ad9472415ad4d441e8ce5e1599446664d727.jpg](../iclr_results/925_KGARevion_ An AI Agent for Knowledge-Intensive Biomedical QA/tables/3e3b0d6ad7ebc0b015aa2fe7f0a0ad9472415ad4d441e8ce5e1599446664d727.jpg)

![7c086757f0e970fc0653c0e3160cdb9a7c7afd693adf8eafc877ca738c16107a.jpg](../iclr_results/925_KGARevion_ An AI Agent for Knowledge-Intensive Biomedical QA/tables/7c086757f0e970fc0653c0e3160cdb9a7c7afd693adf8eafc877ca738c16107a.jpg)

![9ccf3a03ece79725bb56794af31223fe0fd70b21e12921b63cd33e46c4409d5d.jpg](../iclr_results/925_KGARevion_ An AI Agent for Knowledge-Intensive Biomedical QA/tables/9ccf3a03ece79725bb56794af31223fe0fd70b21e12921b63cd33e46c4409d5d.jpg)

![9df6bdf5559d4fbff3d7bce484b37d3d8994a8c5d332d1c15ffc1642eeda29e3.jpg](../iclr_results/925_KGARevion_ An AI Agent for Knowledge-Intensive Biomedical QA/tables/9df6bdf5559d4fbff3d7bce484b37d3d8994a8c5d332d1c15ffc1642eeda29e3.jpg)

![aa688b19585fbf3ae19adcb8c8a8e03ff4eadd42dab994a864578297dc99edb2.jpg](../iclr_results/925_KGARevion_ An AI Agent for Knowledge-Intensive Biomedical QA/tables/aa688b19585fbf3ae19adcb8c8a8e03ff4eadd42dab994a864578297dc99edb2.jpg)

![ac38f2f5ddecc19a39381810374896b05e88539698eb38cce65b3871c04964c2.jpg](../iclr_results/925_KGARevion_ An AI Agent for Knowledge-Intensive Biomedical QA/tables/ac38f2f5ddecc19a39381810374896b05e88539698eb38cce65b3871c04964c2.jpg)

![c7d3a5db86c95ba12751017c0f6db2472611c6c68dc6cecaa24d39ada883e067.jpg](../iclr_results/925_KGARevion_ An AI Agent for Knowledge-Intensive Biomedical QA/tables/c7d3a5db86c95ba12751017c0f6db2472611c6c68dc6cecaa24d39ada883e067.jpg)

## DOTS: Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search


### Images

![26a5d444e4f73174ae22e4c8540d6b0f1059f22023029e0d2d36054b8f50cd25.jpg](../iclr_results/926_DOTS_ Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search/images/26a5d444e4f73174ae22e4c8540d6b0f1059f22023029e0d2d36054b8f50cd25.jpg)

![5b25c6db3ad70f5c71beb1218f2c1cdaf85a51debb7047be66e8f0ceea12d205.jpg](../iclr_results/926_DOTS_ Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search/images/5b25c6db3ad70f5c71beb1218f2c1cdaf85a51debb7047be66e8f0ceea12d205.jpg)

![a71ec03109fa09dd6785f07043cc135fd6e48d947157673fe20f4419cef55c70.jpg](../iclr_results/926_DOTS_ Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search/images/a71ec03109fa09dd6785f07043cc135fd6e48d947157673fe20f4419cef55c70.jpg)

### Tables

![03162a9269e24c4ee039c92f76d5758c33e0985b4a30ea447c5e17e536f947d9.jpg](../iclr_results/926_DOTS_ Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search/tables/03162a9269e24c4ee039c92f76d5758c33e0985b4a30ea447c5e17e536f947d9.jpg)

![272736eb9acbd561b272285582522f344e8402cd8424676476082c5eda5dd55b.jpg](../iclr_results/926_DOTS_ Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search/tables/272736eb9acbd561b272285582522f344e8402cd8424676476082c5eda5dd55b.jpg)

![3421a9599cb2e2024fd28cb5a4c224c0c0ca0495799571aa0d2d55ccb7efd402.jpg](../iclr_results/926_DOTS_ Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search/tables/3421a9599cb2e2024fd28cb5a4c224c0c0ca0495799571aa0d2d55ccb7efd402.jpg)

![5c03c8e6fc6a078947ef3ba1a026dc676e1094d9d8ca258bdfd769a2373cf22f.jpg](../iclr_results/926_DOTS_ Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search/tables/5c03c8e6fc6a078947ef3ba1a026dc676e1094d9d8ca258bdfd769a2373cf22f.jpg)

![5fdfab1d813a23e9c6824cdcb74b1d32b3a07ac19eb9ff1df8f3ef201d357d04.jpg](../iclr_results/926_DOTS_ Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search/tables/5fdfab1d813a23e9c6824cdcb74b1d32b3a07ac19eb9ff1df8f3ef201d357d04.jpg)

![73321771dcafd1df73198460e094cfd6e95d9dec110dbc3cb4b9e994a9561c7d.jpg](../iclr_results/926_DOTS_ Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search/tables/73321771dcafd1df73198460e094cfd6e95d9dec110dbc3cb4b9e994a9561c7d.jpg)

![b2d71166b0194f2d53ca30c1cd8c09ce7907c44987a49f87d3f7d80e6cfc045b.jpg](../iclr_results/926_DOTS_ Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search/tables/b2d71166b0194f2d53ca30c1cd8c09ce7907c44987a49f87d3f7d80e6cfc045b.jpg)

![b3dd350debd5d2144accf9032ced969ff63f5ff02f92dfa98c32e27f72865679.jpg](../iclr_results/926_DOTS_ Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search/tables/b3dd350debd5d2144accf9032ced969ff63f5ff02f92dfa98c32e27f72865679.jpg)

![bee30f3eea080672e7e0c4ac01ba8a1677ff922e98be615f330a86fc937c61c1.jpg](../iclr_results/926_DOTS_ Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search/tables/bee30f3eea080672e7e0c4ac01ba8a1677ff922e98be615f330a86fc937c61c1.jpg)

![ca04326d16a115ef2e04b780dbe1bd24fc975b5f9e3b2dffe26118c46625aba9.jpg](../iclr_results/926_DOTS_ Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search/tables/ca04326d16a115ef2e04b780dbe1bd24fc975b5f9e3b2dffe26118c46625aba9.jpg)

![df02a0a48c9bab50027bed52ea4b28cc425a1cbb4c6439f821eef37787a43623.jpg](../iclr_results/926_DOTS_ Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search/tables/df02a0a48c9bab50027bed52ea4b28cc425a1cbb4c6439f821eef37787a43623.jpg)

![e59991414be41e0ee3904e5d0dc12418c4cb537c766c5bc336161eb3a480e6a6.jpg](../iclr_results/926_DOTS_ Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search/tables/e59991414be41e0ee3904e5d0dc12418c4cb537c766c5bc336161eb3a480e6a6.jpg)

![e5ca47cc2e69f5f26ee7dfd606eaafa455fde004359ff308ef1771ddd5527184.jpg](../iclr_results/926_DOTS_ Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search/tables/e5ca47cc2e69f5f26ee7dfd606eaafa455fde004359ff308ef1771ddd5527184.jpg)

![f5dfc8d6672e132e6b6b4cd7d92d27e35fc90538d7d29a4c84cb3ee8e428f193.jpg](../iclr_results/926_DOTS_ Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search/tables/f5dfc8d6672e132e6b6b4cd7d92d27e35fc90538d7d29a4c84cb3ee8e428f193.jpg)

![f877588c51ff2b6c621fe258b716da606d107093d38783e5055fa4c9396c4b08.jpg](../iclr_results/926_DOTS_ Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search/tables/f877588c51ff2b6c621fe258b716da606d107093d38783e5055fa4c9396c4b08.jpg)

## Learning the Complexity of Weakly Noisy Quantum States


### Images

![793c45282f33260c7b75564fd5be56d259c34c27f3824cada55fa61fce0229a3.jpg](../iclr_results/927_Learning the Complexity of Weakly Noisy Quantum States/images/793c45282f33260c7b75564fd5be56d259c34c27f3824cada55fa61fce0229a3.jpg)

![a1eb4ab3b584d10be1922a8c8fedba85246556ab187581e59ab14724516d175a.jpg](../iclr_results/927_Learning the Complexity of Weakly Noisy Quantum States/images/a1eb4ab3b584d10be1922a8c8fedba85246556ab187581e59ab14724516d175a.jpg)

![ab25409b20ae77529ddc75cc24482af10296755eb7a09e3f19c0df9232a2c8f6.jpg](../iclr_results/927_Learning the Complexity of Weakly Noisy Quantum States/images/ab25409b20ae77529ddc75cc24482af10296755eb7a09e3f19c0df9232a2c8f6.jpg)

## RazorAttention: Efficient KV Cache Compression Through Retrieval Heads


### Images

![173c3797b0ae9754c340404125a6a846d41b07443cb533234bfb34b87c111115.jpg](../iclr_results/928_RazorAttention_ Efficient KV Cache Compression Through Retrieval Heads/images/173c3797b0ae9754c340404125a6a846d41b07443cb533234bfb34b87c111115.jpg)

![439877b7f3b210a11c716c8dd308a32e3601686f12658390f88062bfccfd52be.jpg](../iclr_results/928_RazorAttention_ Efficient KV Cache Compression Through Retrieval Heads/images/439877b7f3b210a11c716c8dd308a32e3601686f12658390f88062bfccfd52be.jpg)

![8ea09ba0bb78c37d573442513ff2f68145bb2f54cf9821310d88432687164690.jpg](../iclr_results/928_RazorAttention_ Efficient KV Cache Compression Through Retrieval Heads/images/8ea09ba0bb78c37d573442513ff2f68145bb2f54cf9821310d88432687164690.jpg)

![ac800c6f9a518645d2b7efc72df70208c63cdbf901171e038ea7de759c9c1ff5.jpg](../iclr_results/928_RazorAttention_ Efficient KV Cache Compression Through Retrieval Heads/images/ac800c6f9a518645d2b7efc72df70208c63cdbf901171e038ea7de759c9c1ff5.jpg)

![b5a157388f3e4f7828fddb4e363830ea9d391593a9bcbc23d621599aebf770e1.jpg](../iclr_results/928_RazorAttention_ Efficient KV Cache Compression Through Retrieval Heads/images/b5a157388f3e4f7828fddb4e363830ea9d391593a9bcbc23d621599aebf770e1.jpg)

![ba031545613a7a795698898b8f064ee09253e8ce1a270a32fb0c2a8d83edf9fd.jpg](../iclr_results/928_RazorAttention_ Efficient KV Cache Compression Through Retrieval Heads/images/ba031545613a7a795698898b8f064ee09253e8ce1a270a32fb0c2a8d83edf9fd.jpg)

![c096157ff4863d3dff73954457db9f5290e92d5a449606fc2b466894a6e420c6.jpg](../iclr_results/928_RazorAttention_ Efficient KV Cache Compression Through Retrieval Heads/images/c096157ff4863d3dff73954457db9f5290e92d5a449606fc2b466894a6e420c6.jpg)

![c40ae3e9b79736e3d3db0204b6fe7e8662e61eabc5495073f0ed0a70d381e596.jpg](../iclr_results/928_RazorAttention_ Efficient KV Cache Compression Through Retrieval Heads/images/c40ae3e9b79736e3d3db0204b6fe7e8662e61eabc5495073f0ed0a70d381e596.jpg)

### Tables

![13bfbc4079286df0f8a4336c9e9c8583cd419c33f2a90857ac53815b2eda353f.jpg](../iclr_results/928_RazorAttention_ Efficient KV Cache Compression Through Retrieval Heads/tables/13bfbc4079286df0f8a4336c9e9c8583cd419c33f2a90857ac53815b2eda353f.jpg)

![1df9084745183c79f5900b994ed763a6254ed8a642963ec767d0f9d8ba95df9c.jpg](../iclr_results/928_RazorAttention_ Efficient KV Cache Compression Through Retrieval Heads/tables/1df9084745183c79f5900b994ed763a6254ed8a642963ec767d0f9d8ba95df9c.jpg)

![213f341771fa82007695c4b9117c63b730ecfc7eb6a66c66c4755ff7876cae6d.jpg](../iclr_results/928_RazorAttention_ Efficient KV Cache Compression Through Retrieval Heads/tables/213f341771fa82007695c4b9117c63b730ecfc7eb6a66c66c4755ff7876cae6d.jpg)

![33ad1855d40d69dcfd5f6087b5cb6384f11d8399931d0e6512f8daea79c34087.jpg](../iclr_results/928_RazorAttention_ Efficient KV Cache Compression Through Retrieval Heads/tables/33ad1855d40d69dcfd5f6087b5cb6384f11d8399931d0e6512f8daea79c34087.jpg)

![859115abbf4e2898d415c058920ef0a4b6a814e748f918d04ba28ea397d05768.jpg](../iclr_results/928_RazorAttention_ Efficient KV Cache Compression Through Retrieval Heads/tables/859115abbf4e2898d415c058920ef0a4b6a814e748f918d04ba28ea397d05768.jpg)

![c59006dba85cc2ee41dbf2ad83b55469691716a7b0e0a1707f49fd480a0e343b.jpg](../iclr_results/928_RazorAttention_ Efficient KV Cache Compression Through Retrieval Heads/tables/c59006dba85cc2ee41dbf2ad83b55469691716a7b0e0a1707f49fd480a0e343b.jpg)

![de9a11a3895197556647cd063552c1222ad3b5fecdf36a4ed8292597dfcab42e.jpg](../iclr_results/928_RazorAttention_ Efficient KV Cache Compression Through Retrieval Heads/tables/de9a11a3895197556647cd063552c1222ad3b5fecdf36a4ed8292597dfcab42e.jpg)

## An Image is Worth More Than 16x16 Patches: Exploring Transformers on Individual Pixels


### Images

![1d4708cf6f1f8d7ba5a41cbc43129660acffa50cd0f8265e77a55710b0d8fd68.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/images/1d4708cf6f1f8d7ba5a41cbc43129660acffa50cd0f8265e77a55710b0d8fd68.jpg)

![26e63f1eeb60c8fdcd599be8103c9e283676625f5d1643855c8b9bb2881d52f7.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/images/26e63f1eeb60c8fdcd599be8103c9e283676625f5d1643855c8b9bb2881d52f7.jpg)

![42d5d4eca191bd20640eccc6bf4f4a3b121b2e617558c70ea6abd4ce48b30d09.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/images/42d5d4eca191bd20640eccc6bf4f4a3b121b2e617558c70ea6abd4ce48b30d09.jpg)

![60df538a0c50094aea53c0c8a9d727fe8f9f1c1f33febd7b6765793b914fd589.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/images/60df538a0c50094aea53c0c8a9d727fe8f9f1c1f33febd7b6765793b914fd589.jpg)

![8390903c093dbaf90a2dff500cba7498241926d1698654f97f438cb2e876ac34.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/images/8390903c093dbaf90a2dff500cba7498241926d1698654f97f438cb2e876ac34.jpg)

![a209cf6697557980b220a5c4710dd1d8be23c4ddb448e3c83afc22128bb988bc.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/images/a209cf6697557980b220a5c4710dd1d8be23c4ddb448e3c83afc22128bb988bc.jpg)

![c7ddc6056dbc100c7d89a3583e611d8a9a9bf2a19a864394699fb8e7103199d4.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/images/c7ddc6056dbc100c7d89a3583e611d8a9a9bf2a19a864394699fb8e7103199d4.jpg)

![d855836bf402da0385e3495e4b28be36ab4e343f0be0cfcb6b070f607ececa94.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/images/d855836bf402da0385e3495e4b28be36ab4e343f0be0cfcb6b070f607ececa94.jpg)

![dcc6be28e691f718d4794d7f75dc3ce2bf365d9d31abca7564c9c17c182079f8.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/images/dcc6be28e691f718d4794d7f75dc3ce2bf365d9d31abca7564c9c17c182079f8.jpg)

![eca7c4ec6b34e6487b8d562c5ee3323a20601dbd297c4219a3f18e7a480d697d.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/images/eca7c4ec6b34e6487b8d562c5ee3323a20601dbd297c4219a3f18e7a480d697d.jpg)

### Tables

![1afb9c7a1e31650a4619dfed5dc601c12b2d9af19e26294cc0de7f42272dd31b.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/tables/1afb9c7a1e31650a4619dfed5dc601c12b2d9af19e26294cc0de7f42272dd31b.jpg)

![31d0728c3b1f80cfd5a47815bb948cbffd479fee3cabf552be4d02f7ae0c22ab.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/tables/31d0728c3b1f80cfd5a47815bb948cbffd479fee3cabf552be4d02f7ae0c22ab.jpg)

![38f1ab787c0b8d527ee9c22f84bc2362795df2b827a41c7aae8ea94caa577ef1.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/tables/38f1ab787c0b8d527ee9c22f84bc2362795df2b827a41c7aae8ea94caa577ef1.jpg)

![4cb112efa06c9dc5803bef6d02bcfd9e02a175b9403a474e51269b3e4b155003.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/tables/4cb112efa06c9dc5803bef6d02bcfd9e02a175b9403a474e51269b3e4b155003.jpg)

![650b3adc50c18daff372144ba4d56cf7b8af4bdf76735d15a6920e929255816a.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/tables/650b3adc50c18daff372144ba4d56cf7b8af4bdf76735d15a6920e929255816a.jpg)

![9a1ad78bddf27e65b29e2918b57a72e40168d461a393d4ad6b545447dada7589.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/tables/9a1ad78bddf27e65b29e2918b57a72e40168d461a393d4ad6b545447dada7589.jpg)

![a17315d29b39e1936c5bc42dc74b06dea6dc17408f20951bc8383e96c48c0512.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/tables/a17315d29b39e1936c5bc42dc74b06dea6dc17408f20951bc8383e96c48c0512.jpg)

![bcb3449e4d526af4f857dd6c331236e398cc8e2129d19ec2f7190da2549deb8d.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/tables/bcb3449e4d526af4f857dd6c331236e398cc8e2129d19ec2f7190da2549deb8d.jpg)

![be6f94527b93e74c9547df5531f9ddf572b2dd086e1bad254296a5618448f43d.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/tables/be6f94527b93e74c9547df5531f9ddf572b2dd086e1bad254296a5618448f43d.jpg)

![c9bfea06f3bcc88edce887f52b205fb63690bff738c8026ff0f63be56169705c.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/tables/c9bfea06f3bcc88edce887f52b205fb63690bff738c8026ff0f63be56169705c.jpg)

![cba5338feb8aff6600010de87a757dcd28c679a400d884e587704b62f354a71c.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/tables/cba5338feb8aff6600010de87a757dcd28c679a400d884e587704b62f354a71c.jpg)

![da9ac1ccd936d75918c79b1046584606a8ca501d235285ecbecca2e2e61dfe5e.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/tables/da9ac1ccd936d75918c79b1046584606a8ca501d235285ecbecca2e2e61dfe5e.jpg)

![dec2cbd0d78d820ac307e2cf471e0ad4904e6e90c11d54c85cd1e6d7ef422f89.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/tables/dec2cbd0d78d820ac307e2cf471e0ad4904e6e90c11d54c85cd1e6d7ef422f89.jpg)

![e247a7b43625bd383df7aa3838dd283f556946a50e92d94f69cd750f946e9438.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/tables/e247a7b43625bd383df7aa3838dd283f556946a50e92d94f69cd750f946e9438.jpg)

![f9888baea43ff8e663a8b8dd6d62f3b2dc38461e23b1a787a1f5a031931e4dee.jpg](../iclr_results/929_An Image is Worth More Than 16x16 Patches_ Exploring Transformers on Individual Pixels/tables/f9888baea43ff8e663a8b8dd6d62f3b2dc38461e23b1a787a1f5a031931e4dee.jpg)

## BAMDP Shaping: a Unified Framework for Intrinsic Motivation and Reward Shaping


### Images

![173dc37feffc0f69da9b277017f2712c158915d63d553deeecd3c318a2826d7f.jpg](../iclr_results/930_BAMDP Shaping_ a Unified Framework for Intrinsic Motivation and Reward Shaping/images/173dc37feffc0f69da9b277017f2712c158915d63d553deeecd3c318a2826d7f.jpg)

![217f643bce9bb6b052a0424dd5227fe1368cbbae65f1fa6836a11b0926ed0474.jpg](../iclr_results/930_BAMDP Shaping_ a Unified Framework for Intrinsic Motivation and Reward Shaping/images/217f643bce9bb6b052a0424dd5227fe1368cbbae65f1fa6836a11b0926ed0474.jpg)

![2aa0656999f65aaf9940b10bbc75aeb59515fb7fe3acc5b8cecadc0adcc63fc1.jpg](../iclr_results/930_BAMDP Shaping_ a Unified Framework for Intrinsic Motivation and Reward Shaping/images/2aa0656999f65aaf9940b10bbc75aeb59515fb7fe3acc5b8cecadc0adcc63fc1.jpg)

![37b40fd61f17afc10161542463414658b4cdf661bfc8b589d0fa18f483bc02a9.jpg](../iclr_results/930_BAMDP Shaping_ a Unified Framework for Intrinsic Motivation and Reward Shaping/images/37b40fd61f17afc10161542463414658b4cdf661bfc8b589d0fa18f483bc02a9.jpg)

![3c1c5477bd324f43ff6cdfefeeb0bd66401b60402c295ca63668e44ecad110ca.jpg](../iclr_results/930_BAMDP Shaping_ a Unified Framework for Intrinsic Motivation and Reward Shaping/images/3c1c5477bd324f43ff6cdfefeeb0bd66401b60402c295ca63668e44ecad110ca.jpg)

![4978d42edfd0bd40be5c50a398f4d470b399914a872cc46e6e436224d0f93553.jpg](../iclr_results/930_BAMDP Shaping_ a Unified Framework for Intrinsic Motivation and Reward Shaping/images/4978d42edfd0bd40be5c50a398f4d470b399914a872cc46e6e436224d0f93553.jpg)

![5b15062386ffa2e06aeb004febb90ba087932cef9948c846ec95917040877407.jpg](../iclr_results/930_BAMDP Shaping_ a Unified Framework for Intrinsic Motivation and Reward Shaping/images/5b15062386ffa2e06aeb004febb90ba087932cef9948c846ec95917040877407.jpg)

![619acf2e1d759508d8d2d17726aafa73e1cbb7503da805a76779f1acd09e99cb.jpg](../iclr_results/930_BAMDP Shaping_ a Unified Framework for Intrinsic Motivation and Reward Shaping/images/619acf2e1d759508d8d2d17726aafa73e1cbb7503da805a76779f1acd09e99cb.jpg)

![9ca98d594d2c9c7e3e170b59780f99efe395cfadce56acf2b07559002bf60c8a.jpg](../iclr_results/930_BAMDP Shaping_ a Unified Framework for Intrinsic Motivation and Reward Shaping/images/9ca98d594d2c9c7e3e170b59780f99efe395cfadce56acf2b07559002bf60c8a.jpg)

![f7449140eeacb936f24ce766ac136ee0a12055f34fedcf4eaa8c11dfc1ee1143.jpg](../iclr_results/930_BAMDP Shaping_ a Unified Framework for Intrinsic Motivation and Reward Shaping/images/f7449140eeacb936f24ce766ac136ee0a12055f34fedcf4eaa8c11dfc1ee1143.jpg)

### Tables

![c4b73bc4452daaa83d12cc49647161f9e5865ce13947e1bc148be74f865029b6.jpg](../iclr_results/930_BAMDP Shaping_ a Unified Framework for Intrinsic Motivation and Reward Shaping/tables/c4b73bc4452daaa83d12cc49647161f9e5865ce13947e1bc148be74f865029b6.jpg)

## Rationalizing and Augmenting Dynamic Graph Neural Networks


### Images

![02905c7c9a5d648a2b07b3b8e4bd1c6c5b173b11aa8863241ce375881f55b35c.jpg](../iclr_results/931_Rationalizing and Augmenting Dynamic Graph Neural Networks/images/02905c7c9a5d648a2b07b3b8e4bd1c6c5b173b11aa8863241ce375881f55b35c.jpg)

![2a796e1577c9aa7767b1ece8661d2cedb30753968fd5d1575ca3cd19a9f63622.jpg](../iclr_results/931_Rationalizing and Augmenting Dynamic Graph Neural Networks/images/2a796e1577c9aa7767b1ece8661d2cedb30753968fd5d1575ca3cd19a9f63622.jpg)

![2b6e72aaced46070850c38639c26fd702bb7f47e9edbd4bf4bc2333780a948c9.jpg](../iclr_results/931_Rationalizing and Augmenting Dynamic Graph Neural Networks/images/2b6e72aaced46070850c38639c26fd702bb7f47e9edbd4bf4bc2333780a948c9.jpg)

![5d2360b26348d471b142a3bde7dfa52888e08c346b2b566cc3552a4dfbc36125.jpg](../iclr_results/931_Rationalizing and Augmenting Dynamic Graph Neural Networks/images/5d2360b26348d471b142a3bde7dfa52888e08c346b2b566cc3552a4dfbc36125.jpg)

![abd0be2e5680956279174ffcd11b298ca8bbfefa751884a3aa56eb6fcb6c504d.jpg](../iclr_results/931_Rationalizing and Augmenting Dynamic Graph Neural Networks/images/abd0be2e5680956279174ffcd11b298ca8bbfefa751884a3aa56eb6fcb6c504d.jpg)

![b416a79e24ce4f9f17a5d7781fc9206d516abb2b149497a8b3dbdabf9837c99e.jpg](../iclr_results/931_Rationalizing and Augmenting Dynamic Graph Neural Networks/images/b416a79e24ce4f9f17a5d7781fc9206d516abb2b149497a8b3dbdabf9837c99e.jpg)

![d1faa0fb0daf2c7084e2f5f895f4ddb95cea62dfcea7f3ea5b5ea57619327b24.jpg](../iclr_results/931_Rationalizing and Augmenting Dynamic Graph Neural Networks/images/d1faa0fb0daf2c7084e2f5f895f4ddb95cea62dfcea7f3ea5b5ea57619327b24.jpg)

![d9c11304d3359d0ba51c0ada983606d242e763011324212c8bd3e21602a2fae2.jpg](../iclr_results/931_Rationalizing and Augmenting Dynamic Graph Neural Networks/images/d9c11304d3359d0ba51c0ada983606d242e763011324212c8bd3e21602a2fae2.jpg)

### Tables

![4ec5cbb8f59dc74eb569747968e50b11b625ebc776b0e69c821d708ee3690aa6.jpg](../iclr_results/931_Rationalizing and Augmenting Dynamic Graph Neural Networks/tables/4ec5cbb8f59dc74eb569747968e50b11b625ebc776b0e69c821d708ee3690aa6.jpg)

![66f1062e3de8685467f819a3a00caca167daf9cb36e6a463c6183762e26561b5.jpg](../iclr_results/931_Rationalizing and Augmenting Dynamic Graph Neural Networks/tables/66f1062e3de8685467f819a3a00caca167daf9cb36e6a463c6183762e26561b5.jpg)

![7dba9f5ee8ddb0e23f4042010a964c5c98bd18bc6637088771af5a6caed3033c.jpg](../iclr_results/931_Rationalizing and Augmenting Dynamic Graph Neural Networks/tables/7dba9f5ee8ddb0e23f4042010a964c5c98bd18bc6637088771af5a6caed3033c.jpg)

![9d1aaaef624b92605af7393143d5b2605395b788f61785c496016f3c12449b16.jpg](../iclr_results/931_Rationalizing and Augmenting Dynamic Graph Neural Networks/tables/9d1aaaef624b92605af7393143d5b2605395b788f61785c496016f3c12449b16.jpg)

![a908e2c547ed96c036adcda7897768d71cefabdd947fc43f985feb412b2e0333.jpg](../iclr_results/931_Rationalizing and Augmenting Dynamic Graph Neural Networks/tables/a908e2c547ed96c036adcda7897768d71cefabdd947fc43f985feb412b2e0333.jpg)

![ba480b6d6470070339a77222958f9d1890d015b82eaeabeacdbf2a820b128223.jpg](../iclr_results/931_Rationalizing and Augmenting Dynamic Graph Neural Networks/tables/ba480b6d6470070339a77222958f9d1890d015b82eaeabeacdbf2a820b128223.jpg)

![c4d8ae21a9cde58a646bad9434549cd117b5e1a44ddfbd3308ea6db969b878ac.jpg](../iclr_results/931_Rationalizing and Augmenting Dynamic Graph Neural Networks/tables/c4d8ae21a9cde58a646bad9434549cd117b5e1a44ddfbd3308ea6db969b878ac.jpg)

![c8adc4f9198500f38a28086ee673f1241b1b7694b62c2957dcd234f18466f10a.jpg](../iclr_results/931_Rationalizing and Augmenting Dynamic Graph Neural Networks/tables/c8adc4f9198500f38a28086ee673f1241b1b7694b62c2957dcd234f18466f10a.jpg)

![ca5cff6b4d7e5ad2b32bf19133539258563e7a00fd8192f4feb17bd33188d351.jpg](../iclr_results/931_Rationalizing and Augmenting Dynamic Graph Neural Networks/tables/ca5cff6b4d7e5ad2b32bf19133539258563e7a00fd8192f4feb17bd33188d351.jpg)

![cd4f98f13c3d4c3139d7cc2c4d220da42dfc6d800b29e184feb8f361db1c0a26.jpg](../iclr_results/931_Rationalizing and Augmenting Dynamic Graph Neural Networks/tables/cd4f98f13c3d4c3139d7cc2c4d220da42dfc6d800b29e184feb8f361db1c0a26.jpg)

![f66d92f57605b21b51d9defbd162956f7fa402a1c3cd16815ee0785bba3f8d9b.jpg](../iclr_results/931_Rationalizing and Augmenting Dynamic Graph Neural Networks/tables/f66d92f57605b21b51d9defbd162956f7fa402a1c3cd16815ee0785bba3f8d9b.jpg)

## Training-Free Diffusion Model Alignment with Sampling Demons


### Images

![00c217ec445fd48612b828707b2ea7a1d0c69bfb1414ad1362000d7296795ee6.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/images/00c217ec445fd48612b828707b2ea7a1d0c69bfb1414ad1362000d7296795ee6.jpg)

![0fa9a12b2ed304b5a38e665afbc1e7bcec31eda2877aeef7ae22db4016e137a4.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/images/0fa9a12b2ed304b5a38e665afbc1e7bcec31eda2877aeef7ae22db4016e137a4.jpg)

![33bed49ab252f7cec4d9d96594fc6e183893775af18ac0c87180fa242db4d65b.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/images/33bed49ab252f7cec4d9d96594fc6e183893775af18ac0c87180fa242db4d65b.jpg)

![438fa7f8ace64b84f2679a1eb4c42947358313ae8dcd56fa0ccd2225dec06dde.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/images/438fa7f8ace64b84f2679a1eb4c42947358313ae8dcd56fa0ccd2225dec06dde.jpg)

![47ad9e1068268e367fe039cfa30db27ce68928df2ce18169bd71390ca809f69e.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/images/47ad9e1068268e367fe039cfa30db27ce68928df2ce18169bd71390ca809f69e.jpg)

![4a3130fe213765fef83165a605292bf906aa4c283434ac81402d0f41d39ea374.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/images/4a3130fe213765fef83165a605292bf906aa4c283434ac81402d0f41d39ea374.jpg)

![619bafa2bc12938519e8ebae2c06979bae81a14c0f3399304e34b3b656087afe.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/images/619bafa2bc12938519e8ebae2c06979bae81a14c0f3399304e34b3b656087afe.jpg)

![6b2dc634917ead77225d455acf1ce59cf27845f4a9d5ff953a81ed8a51da08e3.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/images/6b2dc634917ead77225d455acf1ce59cf27845f4a9d5ff953a81ed8a51da08e3.jpg)

![6d3dc8c719e96a185ebf8c82217573f915569911a6cf4e73e2e7d5b64ed95f10.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/images/6d3dc8c719e96a185ebf8c82217573f915569911a6cf4e73e2e7d5b64ed95f10.jpg)

![8b64d9ac068c833735536ac739633cc8125f349371a1b72ed368710f6c8755a5.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/images/8b64d9ac068c833735536ac739633cc8125f349371a1b72ed368710f6c8755a5.jpg)

![95550e53f7bac005f75f9c445e839001ea69388e0261a8cc5612d3063b23928f.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/images/95550e53f7bac005f75f9c445e839001ea69388e0261a8cc5612d3063b23928f.jpg)

![9de40c7ff3066d061f6554001feb2f3dba59c4c12379e5fea85369235b66ba0f.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/images/9de40c7ff3066d061f6554001feb2f3dba59c4c12379e5fea85369235b66ba0f.jpg)

![a885fef641d9ff2ccc272b33d7c337421a4426f9df05cdf1e5dfd9d2361b43ce.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/images/a885fef641d9ff2ccc272b33d7c337421a4426f9df05cdf1e5dfd9d2361b43ce.jpg)

![b451cb4847ec914fcd3d378b720819912ff4eb0452038fcf018d9a4e0570fdd6.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/images/b451cb4847ec914fcd3d378b720819912ff4eb0452038fcf018d9a4e0570fdd6.jpg)

![bc6f66a2e766f3ef7b661c9ba84a4c0e7e4f48f63be173c48c4a3a50e74186c7.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/images/bc6f66a2e766f3ef7b661c9ba84a4c0e7e4f48f63be173c48c4a3a50e74186c7.jpg)

![cb05c4c627b33bc26d51beadeecf61fea4c6fa4eac033de0619f21267c70cf57.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/images/cb05c4c627b33bc26d51beadeecf61fea4c6fa4eac033de0619f21267c70cf57.jpg)

![dedc11f11e6a309b712e260addd6464ee009d75d74aa9e78bf4b652493df290f.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/images/dedc11f11e6a309b712e260addd6464ee009d75d74aa9e78bf4b652493df290f.jpg)

### Tables

![04632c8d68a49add6085c7d0e4b295f64c26a66266183b8daf5f22dc73c3c6f1.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/tables/04632c8d68a49add6085c7d0e4b295f64c26a66266183b8daf5f22dc73c3c6f1.jpg)

![0b099b9b4a9340b95552617efda33b10139dd41551e4d2694ff1ca62273a2fad.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/tables/0b099b9b4a9340b95552617efda33b10139dd41551e4d2694ff1ca62273a2fad.jpg)

![0ce523a94602f33e6548f7e11a846396b7735ee20e4692d3bb84925d45fa0808.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/tables/0ce523a94602f33e6548f7e11a846396b7735ee20e4692d3bb84925d45fa0808.jpg)

![16b042f9b8f7d5da402b8cd56644f71d476eed1f915797c43d16abb85a54692d.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/tables/16b042f9b8f7d5da402b8cd56644f71d476eed1f915797c43d16abb85a54692d.jpg)

![2430202084dc16c7462757f65c588461835b87f3a15f7a1a6c937b51677dff05.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/tables/2430202084dc16c7462757f65c588461835b87f3a15f7a1a6c937b51677dff05.jpg)

![2e8cce8009de9604b6524cb76446fd54750618393525bd29b80ec685f1f3a9cd.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/tables/2e8cce8009de9604b6524cb76446fd54750618393525bd29b80ec685f1f3a9cd.jpg)

![2f0703931c8b5e9132c18c6b591a55296412e1b92157ffe2f0ec4ad370d52b8f.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/tables/2f0703931c8b5e9132c18c6b591a55296412e1b92157ffe2f0ec4ad370d52b8f.jpg)

![43faaa5ea7ecc6e017ba01f82ca2dbf74414079603dda67d53054198d6b5b174.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/tables/43faaa5ea7ecc6e017ba01f82ca2dbf74414079603dda67d53054198d6b5b174.jpg)

![5d4c7be1c40a3bb2a44cf7f1ddc189a6a9720f09cb638d099aa3cd661373b765.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/tables/5d4c7be1c40a3bb2a44cf7f1ddc189a6a9720f09cb638d099aa3cd661373b765.jpg)

![70dfbe2238ddf53317c15254b3d035b9a33f362e0442f34e21fe58027d21c2c4.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/tables/70dfbe2238ddf53317c15254b3d035b9a33f362e0442f34e21fe58027d21c2c4.jpg)

![828606709de5981bb517e41d5ee57341c30e102fed7bc0f60ac90ae2fae90824.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/tables/828606709de5981bb517e41d5ee57341c30e102fed7bc0f60ac90ae2fae90824.jpg)

![a5f3b97cfe7a0cfe0d0a2ea724cb8788fd554ac4dbb9e2ac5677827f4ce10dd0.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/tables/a5f3b97cfe7a0cfe0d0a2ea724cb8788fd554ac4dbb9e2ac5677827f4ce10dd0.jpg)

![bb4e043ce8ff2f3c44533fe15798f08d23fb3537b87c1aa5ebe4ed4b55f4ae62.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/tables/bb4e043ce8ff2f3c44533fe15798f08d23fb3537b87c1aa5ebe4ed4b55f4ae62.jpg)

![cc4e4d8ebdd3020e026cdcdc6db9a876c7a9efccc9f84fae5df9eeaadf10c7ab.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/tables/cc4e4d8ebdd3020e026cdcdc6db9a876c7a9efccc9f84fae5df9eeaadf10c7ab.jpg)

![f12a6150369dc5a3dab555f7d6a341445752342fa7b381f2cd04c7a3b9b9bbfb.jpg](../iclr_results/932_Training-Free Diffusion Model Alignment with Sampling Demons/tables/f12a6150369dc5a3dab555f7d6a341445752342fa7b381f2cd04c7a3b9b9bbfb.jpg)

## Stochastic Semi-Gradient Descent for Learning Mean Field Games with Population-Aware Function Approximation

### Images

![06d7c058c0ad18756efc1564cea1ec68f3bbc5849898b35071a9fde2213f5435.jpg](../iclr_results/933_Stochastic Semi-Gradient Descent for Learning Mean Field Games with Population-Aware Function Approx/images/06d7c058c0ad18756efc1564cea1ec68f3bbc5849898b35071a9fde2213f5435.jpg)

![17e8f78add6555c71183611f24d400f52bc106d240655562f155dbc05514a17a.jpg](../iclr_results/933_Stochastic Semi-Gradient Descent for Learning Mean Field Games with Population-Aware Function Approx/images/17e8f78add6555c71183611f24d400f52bc106d240655562f155dbc05514a17a.jpg)

![1a4b450825b4cb82170de32657a8379720b977fe5c2854a9d3727b18162edcf4.jpg](../iclr_results/933_Stochastic Semi-Gradient Descent for Learning Mean Field Games with Population-Aware Function Approx/images/1a4b450825b4cb82170de32657a8379720b977fe5c2854a9d3727b18162edcf4.jpg)

![1edf0ad744bb382cbd3228bd889dd1e067247331b602c599dac9bfc646542760.jpg](../iclr_results/933_Stochastic Semi-Gradient Descent for Learning Mean Field Games with Population-Aware Function Approx/images/1edf0ad744bb382cbd3228bd889dd1e067247331b602c599dac9bfc646542760.jpg)

![5a734741058675960bd664ddae4e9fdcaf2cbf9319efc2076cf384cfbcb04858.jpg](../iclr_results/933_Stochastic Semi-Gradient Descent for Learning Mean Field Games with Population-Aware Function Approx/images/5a734741058675960bd664ddae4e9fdcaf2cbf9319efc2076cf384cfbcb04858.jpg)

![68486ef6d74573c93950563d8c89891ac7c84e5f9a1323958f13e4930a91f6e0.jpg](../iclr_results/933_Stochastic Semi-Gradient Descent for Learning Mean Field Games with Population-Aware Function Approx/images/68486ef6d74573c93950563d8c89891ac7c84e5f9a1323958f13e4930a91f6e0.jpg)

![734b21d96b6cf928ab3af2f3885df8ff6c88f3e733d6b27fab34a96c5f9d0dca.jpg](../iclr_results/933_Stochastic Semi-Gradient Descent for Learning Mean Field Games with Population-Aware Function Approx/images/734b21d96b6cf928ab3af2f3885df8ff6c88f3e733d6b27fab34a96c5f9d0dca.jpg)

![8b81f6ff9a1a4ede904ffa8f84f69a2a01f279aea7650f237989c10e1be60284.jpg](../iclr_results/933_Stochastic Semi-Gradient Descent for Learning Mean Field Games with Population-Aware Function Approx/images/8b81f6ff9a1a4ede904ffa8f84f69a2a01f279aea7650f237989c10e1be60284.jpg)

![9d797b6b144e66979c6c69cb4a62f4e2f7ec0da0de813dc5077dbaeef9bb1517.jpg](../iclr_results/933_Stochastic Semi-Gradient Descent for Learning Mean Field Games with Population-Aware Function Approx/images/9d797b6b144e66979c6c69cb4a62f4e2f7ec0da0de813dc5077dbaeef9bb1517.jpg)

![b1bdbfa6f931f9542cc3bf4888b1d8b801293003dc6b756151cf049f3c1467ac.jpg](../iclr_results/933_Stochastic Semi-Gradient Descent for Learning Mean Field Games with Population-Aware Function Approx/images/b1bdbfa6f931f9542cc3bf4888b1d8b801293003dc6b756151cf049f3c1467ac.jpg)

![cbd5da1b195ef17781f6385e4f6b47250f810dd4cab728f7ba10b317900333f5.jpg](../iclr_results/933_Stochastic Semi-Gradient Descent for Learning Mean Field Games with Population-Aware Function Approx/images/cbd5da1b195ef17781f6385e4f6b47250f810dd4cab728f7ba10b317900333f5.jpg)

![e4130683a2deabce9b87aa06f158add9d5cb179153d3321a9e1bdee7f653378a.jpg](../iclr_results/933_Stochastic Semi-Gradient Descent for Learning Mean Field Games with Population-Aware Function Approx/images/e4130683a2deabce9b87aa06f158add9d5cb179153d3321a9e1bdee7f653378a.jpg)

![f2e6c34912c0e86bc63d9ec8384cf54febf8d0cddd07ad4f59c46ac665c97641.jpg](../iclr_results/933_Stochastic Semi-Gradient Descent for Learning Mean Field Games with Population-Aware Function Approx/images/f2e6c34912c0e86bc63d9ec8384cf54febf8d0cddd07ad4f59c46ac665c97641.jpg)

### Tables

![0f9ff5665fc151196cddbb17939919e2f10964f4458b54b896d57cc1f76016a8.jpg](../iclr_results/933_Stochastic Semi-Gradient Descent for Learning Mean Field Games with Population-Aware Function Approx/tables/0f9ff5665fc151196cddbb17939919e2f10964f4458b54b896d57cc1f76016a8.jpg)

![51aacd62e88504a8e74ddcd9e58a4076f0f993ceefec36963d3bf69d9473e033.jpg](../iclr_results/933_Stochastic Semi-Gradient Descent for Learning Mean Field Games with Population-Aware Function Approx/tables/51aacd62e88504a8e74ddcd9e58a4076f0f993ceefec36963d3bf69d9473e033.jpg)

![74ce1404d576a8f1c0166fcd7d107157a3184859f13acd501a91304f056104d4.jpg](../iclr_results/933_Stochastic Semi-Gradient Descent for Learning Mean Field Games with Population-Aware Function Approx/tables/74ce1404d576a8f1c0166fcd7d107157a3184859f13acd501a91304f056104d4.jpg)

![b64ad90d45e1d36b338828bb02730c3e91f64feb1aea63fffb8a47703cf69bb9.jpg](../iclr_results/933_Stochastic Semi-Gradient Descent for Learning Mean Field Games with Population-Aware Function Approx/tables/b64ad90d45e1d36b338828bb02730c3e91f64feb1aea63fffb8a47703cf69bb9.jpg)
