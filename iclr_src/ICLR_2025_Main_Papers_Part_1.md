# ICLR 2025 Main Conference Papers

**Summary:** 36 papers with extracted content:
- 📊 Total images: 44031
- 📋 Total tables: 33468
- 📄 Total files: 77499

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 1 of 100

## 目录 (Table of Contents)

1. [Probabilistic Learning to Defer: Handling Missing Expert Annotations and Controlling Workload Distribution](#Probabilistic-Learning-to-Defer-Handling-Missing-Expert-Annotations-and-Controlling-Workload-Distribution)
2. [Kinetix: Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks](#Kinetix-Investigating-the-Training-of-General-Agents-through-Open-Ended-Physics-Based-Control-Tasks)
3. [Joint Graph Rewiring and Feature Denoising via Spectral Resonance](#Joint-Graph-Rewiring-and-Feature-Denoising-via-Spectral-Resonance)
4. [AFlow: Automating Agentic Workflow Generation](#AFlow-Automating-Agentic-Workflow-Generation)
5. [Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo](#Syntactic-and-Semantic-Control-of-Large-Language-Models-via-Sequential-Monte-Carlo)
6. [Can a MISL Fly? Analysis and Ingredients for Mutual Information Skill Learning](#Can-a-MISL-Fly-Analysis-and-Ingredients-for-Mutual-Information-Skill-Learning)
7. [OLMoE: Open Mixture-of-Experts Language Models](#OLMoE-Open-Mixture-of-Experts-Language-Models)
8. [Learning to Discretize Denoising Diffusion ODEs](#Learning-to-Discretize-Denoising-Diffusion-ODEs)
9. [When Selection Meets Intervention: Additional Complexities in Causal Discovery](#When-Selection-Meets-Intervention-Additional-Complexities-in-Causal-Discovery)
10. [Scaling Laws for Precision](#Scaling-Laws-for-Precision)
11. [Loopy: Taming Audio-Driven Portrait Avatar with Long-Term Motion Dependency](#Loopy-Taming-Audio-Driven-Portrait-Avatar-with-Long-Term-Motion-Dependency)
12. [Progressive distillation induces an implicit curriculum](#Progressive-distillation-induces-an-implicit-curriculum)
13. [Diffusion-Based Planning for Autonomous Driving with Flexible Guidance](#Diffusion-Based-Planning-for-Autonomous-Driving-with-Flexible-Guidance)
14. [Open-World Reinforcement Learning over Long Short-Term Imagination](#Open-World-Reinforcement-Learning-over-Long-Short-Term-Imagination)
15. [Faster Cascades via Speculative Decoding](#Faster-Cascades-via-Speculative-Decoding)
16. [DEPT: Decoupled Embeddings for Pre-training Language Models](#DEPT-Decoupled-Embeddings-for-Pre-training-Language-Models)
17. [When is Task Vector Provably Effective for Model Editing? A Generalization Analysis of Nonlinear Transformers](#When-is-Task-Vector-Provably-Effective-for-Model-Editing-A-Generalization-Analysis-of-Nonlinear-Transformers)
18. [Learning to Search from Demonstration Sequences](#Learning-to-Search-from-Demonstration-Sequences)
19. [Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks](#Learning-Distributions-of-Complex-Fluid-Simulations-with-Diffusion-Graph-Networks)
20. [Capturing the Temporal Dependence of Training Data Influence](#Capturing-the-Temporal-Dependence-of-Training-Data-Influence)
21. [Two Effects, One Trigger: On the Modality Gap, Object Bias, and Information Imbalance in Contrastive Vision-Language Models](#Two-Effects-One-Trigger-On-the-Modality-Gap-Object-Bias-and-Information-Imbalance-in-Contrastive-Vision-Language-Models)
22. [Scaling In-the-Wild Training for Diffusion-based Illumination Harmonization and Editing by Imposing Consistent Light Transport](#Scaling-In-the-Wild-Training-for-Diffusion-based-Illumination-Harmonization-and-Editing-by-Imposing-Consistent-Light-Transport)
23. [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](#Block-Diffusion-Interpolating-Between-Autoregressive-and-Diffusion-Language-Models)
24. [Robustness Inspired Graph Backdoor Defense](#Robustness-Inspired-Graph-Backdoor-Defense)
25. [Flow Matching with General Discrete Paths: A Kinetic-Optimal Perspective](#Flow-Matching-with-General-Discrete-Paths-A-Kinetic-Optimal-Perspective)
26. [Scaling and evaluating sparse autoencoders](#Scaling-and-evaluating-sparse-autoencoders)
27. [Cybench: A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models](#Cybench-A-Framework-for-Evaluating-Cybersecurity-Capabilities-and-Risks-of-Language-Models)
28. [Booster: Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation](#Booster-Tackling-Harmful-Fine-tuning-for-Large-Language-Models-via-Attenuating-Harmful-Perturbation)
29. [Learning Dynamics of LLM Finetuning](#Learning-Dynamics-of-LLM-Finetuning)
30. [Classic but Everlasting: Traditional Gradient-Based Algorithms Converge Fast Even in Time-Varying Multi-Player Games](#Classic-but-Everlasting-Traditional-Gradient-Based-Algorithms-Converge-Fast-Even-in-Time-Varying-Multi-Player-Games)
31. [MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation Experts](#MoE-Accelerating-Mixture-of-Experts-Methods-with-Zero-Computation-Experts)
32. [Cheating Automatic LLM Benchmarks: Null Models Achieve High Win Rates](#Cheating-Automatic-LLM-Benchmarks-Null-Models-Achieve-High-Win-Rates)
33. [Tractable Multi-Agent Reinforcement Learning through Behavioral Economics](#Tractable-Multi-Agent-Reinforcement-Learning-through-Behavioral-Economics)
34. [Do as We Do, Not as You Think: the Conformity of Large Language Models](#Do-as-We-Do-Not-as-You-Think-the-Conformity-of-Large-Language-Models)
35. [Linear Representations of Political Perspective Emerge in Large Language Models](#Linear-Representations-of-Political-Perspective-Emerge-in-Large-Language-Models)
36. [Rethinking Reward Modeling in Preference-based Large Language Model Alignment](#Rethinking-Reward-Modeling-in-Preference-based-Large-Language-Model-Alignment)

---


## Probabilistic Learning to Defer: Handling Missing Expert Annotations and Controlling Workload Distribution


### Images

![34e066269504068cd751a46dee1e113f330ad73ae9711582c8f6aa3264d2464f.jpg](../iclr_results/1_Probabilistic Learning to Defer_ Handling Missing Expert Annotations and Controlling Workload Distri/images/34e066269504068cd751a46dee1e113f330ad73ae9711582c8f6aa3264d2464f.jpg)

![7061cc11768acd21ca7fbe7e02ff1e0e3e3c1351ba9963c35a7e66801988ba74.jpg](../iclr_results/1_Probabilistic Learning to Defer_ Handling Missing Expert Annotations and Controlling Workload Distri/images/7061cc11768acd21ca7fbe7e02ff1e0e3e3c1351ba9963c35a7e66801988ba74.jpg)

![7810ca1d38b0ec247a69e394b2780bf6ca48886b792e3f5e32f083482c199723.jpg](../iclr_results/1_Probabilistic Learning to Defer_ Handling Missing Expert Annotations and Controlling Workload Distri/images/7810ca1d38b0ec247a69e394b2780bf6ca48886b792e3f5e32f083482c199723.jpg)

![898720be7b557e4d2d322eeadca1785f47414a59dc009494121d577ab338fdfc.jpg](../iclr_results/1_Probabilistic Learning to Defer_ Handling Missing Expert Annotations and Controlling Workload Distri/images/898720be7b557e4d2d322eeadca1785f47414a59dc009494121d577ab338fdfc.jpg)

![933858ad4b3814382db3521899659e4f80ef3325a75fce63071c04147ce3f50f.jpg](../iclr_results/1_Probabilistic Learning to Defer_ Handling Missing Expert Annotations and Controlling Workload Distri/images/933858ad4b3814382db3521899659e4f80ef3325a75fce63071c04147ce3f50f.jpg)

![ac1221d7f5fa93ea1d6485f87db7f6f70c760e3ee5d8126f4354e06be88dde20.jpg](../iclr_results/1_Probabilistic Learning to Defer_ Handling Missing Expert Annotations and Controlling Workload Distri/images/ac1221d7f5fa93ea1d6485f87db7f6f70c760e3ee5d8126f4354e06be88dde20.jpg)

![b2cc305293001e653bced0be021013b337527e250e7ebee5de63f82177feeb9e.jpg](../iclr_results/1_Probabilistic Learning to Defer_ Handling Missing Expert Annotations and Controlling Workload Distri/images/b2cc305293001e653bced0be021013b337527e250e7ebee5de63f82177feeb9e.jpg)

### Tables

![13c735a0947e362195bdba315487ffeea37e7a69a2be23807f05c0e52b8eebee.jpg](../iclr_results/1_Probabilistic Learning to Defer_ Handling Missing Expert Annotations and Controlling Workload Distri/tables/13c735a0947e362195bdba315487ffeea37e7a69a2be23807f05c0e52b8eebee.jpg)

![3d2e7ff20574a09a63a7091569194c53d8ef0977c6dbea031283bfe37580a5d3.jpg](../iclr_results/1_Probabilistic Learning to Defer_ Handling Missing Expert Annotations and Controlling Workload Distri/tables/3d2e7ff20574a09a63a7091569194c53d8ef0977c6dbea031283bfe37580a5d3.jpg)

![91ddd608bb0478a001f9aa95805ac844e5568b4d790c90d02b2208367ee81d70.jpg](../iclr_results/1_Probabilistic Learning to Defer_ Handling Missing Expert Annotations and Controlling Workload Distri/tables/91ddd608bb0478a001f9aa95805ac844e5568b4d790c90d02b2208367ee81d70.jpg)

![c40d4ac81b9790546f04b743abb1220e5f623bb24a018340020a528886e7900e.jpg](../iclr_results/1_Probabilistic Learning to Defer_ Handling Missing Expert Annotations and Controlling Workload Distri/tables/c40d4ac81b9790546f04b743abb1220e5f623bb24a018340020a528886e7900e.jpg)

## Kinetix: Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks


### Images

![094ddd610ce733bad11be2313cce45f689b3378091ea92310ab0e3a9fe52b996.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/094ddd610ce733bad11be2313cce45f689b3378091ea92310ab0e3a9fe52b996.jpg)

![266ae513dfe5eb7543e62add024123b19d4c31b67c630a29cf5d6dc34427aff2.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/266ae513dfe5eb7543e62add024123b19d4c31b67c630a29cf5d6dc34427aff2.jpg)

![29f41fb85716c8326ecc8cbddc6f116cc4b31720ffc9cf9519b682295d44c35f.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/29f41fb85716c8326ecc8cbddc6f116cc4b31720ffc9cf9519b682295d44c35f.jpg)

![2b01a2d107d545bb4a02d34bd66edb38d425e41a1eb006c9619a841843c94b3a.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/2b01a2d107d545bb4a02d34bd66edb38d425e41a1eb006c9619a841843c94b3a.jpg)

![38dfaddbe778fd38eb8a79fa5f323fd49b69ce6c2f44589f7528221baee53dd2.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/38dfaddbe778fd38eb8a79fa5f323fd49b69ce6c2f44589f7528221baee53dd2.jpg)

![3f07714d5abf73510aac648e979bdc918930b652a55b44b8581cbdca458d4500.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/3f07714d5abf73510aac648e979bdc918930b652a55b44b8581cbdca458d4500.jpg)

![5234bf09a4ac5b912fad2b02b4c48745de9c2ac53d011dbdd025cc91f46f9b50.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/5234bf09a4ac5b912fad2b02b4c48745de9c2ac53d011dbdd025cc91f46f9b50.jpg)

![57713e9448f30be159e70fa62ed91227492a4a7f95be4e3af8964a94ca251d9f.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/57713e9448f30be159e70fa62ed91227492a4a7f95be4e3af8964a94ca251d9f.jpg)

![588a634db72a992efa6cb4a63e1ff0550ae78b69048741a964d039c2f903baab.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/588a634db72a992efa6cb4a63e1ff0550ae78b69048741a964d039c2f903baab.jpg)

![5f06da93749d88da02d69e7d4a2479a6635141a5b669ffb94f61eaa23a0a8ea4.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/5f06da93749d88da02d69e7d4a2479a6635141a5b669ffb94f61eaa23a0a8ea4.jpg)

![6487a719defb717da360cb8c1dba936cd6413624053bcbe6617c88812665d14a.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/6487a719defb717da360cb8c1dba936cd6413624053bcbe6617c88812665d14a.jpg)

![6e6bc1a981856180909707a95f8661999c2684beb8a6316f58224beaaafaf57a.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/6e6bc1a981856180909707a95f8661999c2684beb8a6316f58224beaaafaf57a.jpg)

![6e7d9613f9e9a0e3b80701664f8cb55c96210c4bb289962f6f66ae2e0f241f32.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/6e7d9613f9e9a0e3b80701664f8cb55c96210c4bb289962f6f66ae2e0f241f32.jpg)

![6e8d7cecb555ec4a6b58b1ea9c4a6cbd362d3c2d401aaff9f7d27b5bc05ba802.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/6e8d7cecb555ec4a6b58b1ea9c4a6cbd362d3c2d401aaff9f7d27b5bc05ba802.jpg)

![87e02b17386078ee57ac35f61b991380820153a294366d1bfeeb541ad230f966.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/87e02b17386078ee57ac35f61b991380820153a294366d1bfeeb541ad230f966.jpg)

![8fec4f4d59ff70e564ed5c53744c9f2b2074616d744748c0dc7dba16ac94cd96.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/8fec4f4d59ff70e564ed5c53744c9f2b2074616d744748c0dc7dba16ac94cd96.jpg)

![94529d142de8475c97f6e58f6f7d99d4c26acc9bc1c38ed5f52084365251e8e3.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/94529d142de8475c97f6e58f6f7d99d4c26acc9bc1c38ed5f52084365251e8e3.jpg)

![aaf4db9e1034f4bed89460a46cfeb51dfe6e311dfb1af21debaca340455b3cdc.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/aaf4db9e1034f4bed89460a46cfeb51dfe6e311dfb1af21debaca340455b3cdc.jpg)

![ab84129ab174bad23850384f5cca15016d7bce8464f1feaf9795f7fd7f20edaf.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/ab84129ab174bad23850384f5cca15016d7bce8464f1feaf9795f7fd7f20edaf.jpg)

![b48b06ec8491f8534a9997ccfcc0a7aaad0274c01e3e2d67edf3e87ff637e0f5.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/b48b06ec8491f8534a9997ccfcc0a7aaad0274c01e3e2d67edf3e87ff637e0f5.jpg)

![bf00dd77ac070e7c23f3a42958bdc7a49b53feb1ace98db95d1dd9f3411d53b7.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/bf00dd77ac070e7c23f3a42958bdc7a49b53feb1ace98db95d1dd9f3411d53b7.jpg)

![c812b221816a813ae34d95eddc22e650c6df805cd82c1e9af413e3d5317b0774.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/c812b221816a813ae34d95eddc22e650c6df805cd82c1e9af413e3d5317b0774.jpg)

![cc0e1bf12d0c02c66dcb4654b73abe972e338d75a6810a3571abe9b02a85fc33.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/cc0e1bf12d0c02c66dcb4654b73abe972e338d75a6810a3571abe9b02a85fc33.jpg)

![cec6e034cd98c80071a5792dda77f62cedfe509607218caa19bc30619d1296ad.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/cec6e034cd98c80071a5792dda77f62cedfe509607218caa19bc30619d1296ad.jpg)

![d2b62e7fc1c434d85f18bb58b92e5543904921ad077618c29fcb6f64c0693347.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/d2b62e7fc1c434d85f18bb58b92e5543904921ad077618c29fcb6f64c0693347.jpg)

![d379b0801e61b6643434f7d902d91cf6774c5e49db65b3822f814e553441b016.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/d379b0801e61b6643434f7d902d91cf6774c5e49db65b3822f814e553441b016.jpg)

![d45213f339e492a9f790f51c9685f71cc3753b50591562ff2bea36d90bcc9119.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/d45213f339e492a9f790f51c9685f71cc3753b50591562ff2bea36d90bcc9119.jpg)

![de0d37cc1a0b4c52fbd34e6c3102029fd6b88e628f1088b93613a7f1786f600f.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/de0d37cc1a0b4c52fbd34e6c3102029fd6b88e628f1088b93613a7f1786f600f.jpg)

![ea144c51a10a02973dd3a3674695e16168b9071bf6e95bd592a8969d1abc12ce.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/ea144c51a10a02973dd3a3674695e16168b9071bf6e95bd592a8969d1abc12ce.jpg)

![f7a89199ecdb41e67a92274c57f51746d58524bdf9b3e8f9955d25e10da49710.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/f7a89199ecdb41e67a92274c57f51746d58524bdf9b3e8f9955d25e10da49710.jpg)

![f8d13778cf6eee13b9a2af923c50e484ce6fb5158cc755b2a7320c8c1400f05c.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/f8d13778cf6eee13b9a2af923c50e484ce6fb5158cc755b2a7320c8c1400f05c.jpg)

![fd6f7f1b47ded26d57154758c86488d4d3c78a4ff95fd40b950da2e23261aee8.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/fd6f7f1b47ded26d57154758c86488d4d3c78a4ff95fd40b950da2e23261aee8.jpg)

![ffa48192dc2aecf1455a62cd37b7d84f2818891b43897499c70696c0b0b26009.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/images/ffa48192dc2aecf1455a62cd37b7d84f2818891b43897499c70696c0b0b26009.jpg)

### Tables

![12c8ced62a530c77e1358395b0527673c596eb79a1de78aa3f0dcdd6a9e16439.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/tables/12c8ced62a530c77e1358395b0527673c596eb79a1de78aa3f0dcdd6a9e16439.jpg)

![22365a68084ce24e2963de4d4ab307e099da90b1b9e6f0648d361b2a348f8bb0.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/tables/22365a68084ce24e2963de4d4ab307e099da90b1b9e6f0648d361b2a348f8bb0.jpg)

![7634fc2496ee7d1033d008f8e4c4efe46da8b88305ee36837e6a1921ee50727d.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/tables/7634fc2496ee7d1033d008f8e4c4efe46da8b88305ee36837e6a1921ee50727d.jpg)

![913b42c554dae48b468203390ab230f7fb4418e7e5692d214fd3d7c183ccc354.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/tables/913b42c554dae48b468203390ab230f7fb4418e7e5692d214fd3d7c183ccc354.jpg)

![9f13f84cd100707815e64e87264b9d5bbfcb23d8f23e17cb089bb3793dd57a76.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/tables/9f13f84cd100707815e64e87264b9d5bbfcb23d8f23e17cb089bb3793dd57a76.jpg)

![bb8d1cc5333217306d537725d5fc00acde5b59b14530aebb3042bfc2de51d77a.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/tables/bb8d1cc5333217306d537725d5fc00acde5b59b14530aebb3042bfc2de51d77a.jpg)

![ec31895dcf23aa5c2fa3b90b453b2bf0d49b8da9561e6e247c956e1d7bd9ef60.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/tables/ec31895dcf23aa5c2fa3b90b453b2bf0d49b8da9561e6e247c956e1d7bd9ef60.jpg)

![fbc96c4765ae43697206c401d87618246b4a955825cc41fcfded792429bf3aac.jpg](../iclr_results/2_Kinetix_ Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks/tables/fbc96c4765ae43697206c401d87618246b4a955825cc41fcfded792429bf3aac.jpg)

## Joint Graph Rewiring and Feature Denoising via Spectral Resonance


### Images

![133f40ab9933d55b4e1c3b9486cf5325a7260a5e0142710e4a417fbe9fa042f5.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/images/133f40ab9933d55b4e1c3b9486cf5325a7260a5e0142710e4a417fbe9fa042f5.jpg)

![1550f4e4b22cba51576d24c064c7b1226824cfbbbeb261704d65eddca2704eac.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/images/1550f4e4b22cba51576d24c064c7b1226824cfbbbeb261704d65eddca2704eac.jpg)

![1e695aef04bf8c22f05c21c5b51a49307fa0172e7eddc7ea993344c215b21fef.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/images/1e695aef04bf8c22f05c21c5b51a49307fa0172e7eddc7ea993344c215b21fef.jpg)

![2e7c3ad84c79b6cfb4675a4296cdb8b5a485ac7625aa50c4fb26252216a1d3a5.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/images/2e7c3ad84c79b6cfb4675a4296cdb8b5a485ac7625aa50c4fb26252216a1d3a5.jpg)

![3a90e36ca7d83bba294caee333c60d9889cf74ded99f43e5ea774153b08e2776.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/images/3a90e36ca7d83bba294caee333c60d9889cf74ded99f43e5ea774153b08e2776.jpg)

![3cf630f88eb0afddfe5a8715b04cce3cec7dc1640666d0f7e7d20c89f164dee9.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/images/3cf630f88eb0afddfe5a8715b04cce3cec7dc1640666d0f7e7d20c89f164dee9.jpg)

![5c2b6a0c64dc1f7840e336ffda6650cb7e658ffd59ee96a39191d0248450c8d3.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/images/5c2b6a0c64dc1f7840e336ffda6650cb7e658ffd59ee96a39191d0248450c8d3.jpg)

![67edec4d9e2bbc66e6432ff86c3067ae741231855d986cff73b671ec6669e571.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/images/67edec4d9e2bbc66e6432ff86c3067ae741231855d986cff73b671ec6669e571.jpg)

![6c38f6a9c72e4068e055c41c714150eb9b849e089dd407c1b9b8a483b21a6860.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/images/6c38f6a9c72e4068e055c41c714150eb9b849e089dd407c1b9b8a483b21a6860.jpg)

![7a90197b6c2ed829a883f966c3dabc763b821e22e0e8de26f4c10ad66595277c.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/images/7a90197b6c2ed829a883f966c3dabc763b821e22e0e8de26f4c10ad66595277c.jpg)

![96123d982d551647b91e302c27faed4368e707a9d21293d1bd2d7bb2bc4b88e2.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/images/96123d982d551647b91e302c27faed4368e707a9d21293d1bd2d7bb2bc4b88e2.jpg)

![aa367796d59a4a2f320b9a8de3a208aeff1771bb99c7f35fcae31d87849aef27.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/images/aa367796d59a4a2f320b9a8de3a208aeff1771bb99c7f35fcae31d87849aef27.jpg)

![b06292892c59411d1903dbf90fadb83478b341b1c84fb650730dc8a77a0b1545.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/images/b06292892c59411d1903dbf90fadb83478b341b1c84fb650730dc8a77a0b1545.jpg)

![d024f685f026b0f751e894fbb884e0b8a7cff64a11bac40f5e9b44ffd4d6c9f9.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/images/d024f685f026b0f751e894fbb884e0b8a7cff64a11bac40f5e9b44ffd4d6c9f9.jpg)

![d49e09bbecbfc8bee290908cbb4306f4a2cc994f670a7680e37395ff4b2b3f0f.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/images/d49e09bbecbfc8bee290908cbb4306f4a2cc994f670a7680e37395ff4b2b3f0f.jpg)

![d821c933bceb0ba0a722d0d57f4d52f34e671fc26736ca1dc176890ef3bfa6b3.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/images/d821c933bceb0ba0a722d0d57f4d52f34e671fc26736ca1dc176890ef3bfa6b3.jpg)

![dac5da92e57633d62089dfdd2c1626f84ab4530baf8523eb2f219b50321f210d.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/images/dac5da92e57633d62089dfdd2c1626f84ab4530baf8523eb2f219b50321f210d.jpg)

![df3dcc7eccab9f7d06debe036639136aa7afd0141d454acda14aecdc64e47c0d.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/images/df3dcc7eccab9f7d06debe036639136aa7afd0141d454acda14aecdc64e47c0d.jpg)

![fcada120904f40274488845b4589ff23f96b7e2f523a0473890b8d764900eab7.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/images/fcada120904f40274488845b4589ff23f96b7e2f523a0473890b8d764900eab7.jpg)

### Tables

![0570aca42c4a04ad322c60f40e866842b832560dfaf3e47d7bbe4d7e894362f3.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/0570aca42c4a04ad322c60f40e866842b832560dfaf3e47d7bbe4d7e894362f3.jpg)

![13d5dd23a12d4c9f12949db22d7b4246e8098c0e1ff56ae6caba044bd07cfc08.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/13d5dd23a12d4c9f12949db22d7b4246e8098c0e1ff56ae6caba044bd07cfc08.jpg)

![2ac0dadcc9f589a4cfbbdbbbe8b637ce58d51cf3b8cf1da9d597870163e1f270.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/2ac0dadcc9f589a4cfbbdbbbe8b637ce58d51cf3b8cf1da9d597870163e1f270.jpg)

![2dc9df02255b05796cd128fe8e3b898a425a4aa9a31c64855985d6bb8116c358.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/2dc9df02255b05796cd128fe8e3b898a425a4aa9a31c64855985d6bb8116c358.jpg)

![3cfd8bed39376be0bdff4f8f89e00c01a82d25d830df87d456070615b40a7b2a.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/3cfd8bed39376be0bdff4f8f89e00c01a82d25d830df87d456070615b40a7b2a.jpg)

![4ba857390f964b19400a60baa7f162bc08f98e001aa146716d8dd4ece82d84d0.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/4ba857390f964b19400a60baa7f162bc08f98e001aa146716d8dd4ece82d84d0.jpg)

![61f15e326b58f0b66037191002a19265cfca4ab7cbb3c1afa794652befd71432.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/61f15e326b58f0b66037191002a19265cfca4ab7cbb3c1afa794652befd71432.jpg)

![679a5115f1f18f25c9622ba3173d5fb08b11a1710deb71f56cd2543af442f532.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/679a5115f1f18f25c9622ba3173d5fb08b11a1710deb71f56cd2543af442f532.jpg)

![6a16dd15a609610e7c95e5c2dfb0ec83896ccee18d76fa9b5d2c1c0662590920.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/6a16dd15a609610e7c95e5c2dfb0ec83896ccee18d76fa9b5d2c1c0662590920.jpg)

![6c30b6e2621ceac3cd12922a54af00cf3967170265feb434faa61344da784b46.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/6c30b6e2621ceac3cd12922a54af00cf3967170265feb434faa61344da784b46.jpg)

![72c8fb55107acfa19bce33b325b91de84d304596043a0aeaaa469dce5b8b4973.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/72c8fb55107acfa19bce33b325b91de84d304596043a0aeaaa469dce5b8b4973.jpg)

![807298719aa3f8f9003b008b2a492c8658f78d0f4844093c44da72e20d715d16.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/807298719aa3f8f9003b008b2a492c8658f78d0f4844093c44da72e20d715d16.jpg)

![8681ab696b3be276fab4b2323f854b4631b4d57d6e9fddf8dd3ebe4a523e6c81.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/8681ab696b3be276fab4b2323f854b4631b4d57d6e9fddf8dd3ebe4a523e6c81.jpg)

![92a6a7996a37355532927e5091d86926bf11388c9ecd707f675f64398fc3d107.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/92a6a7996a37355532927e5091d86926bf11388c9ecd707f675f64398fc3d107.jpg)

![9a35ae8532db6e856379b1172aa0c9963288f764e5fe301013718402fa9af9b5.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/9a35ae8532db6e856379b1172aa0c9963288f764e5fe301013718402fa9af9b5.jpg)

![9f7a807909477ca562b51941743efc8f34b24dabaace713bdca218c4f4bb14de.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/9f7a807909477ca562b51941743efc8f34b24dabaace713bdca218c4f4bb14de.jpg)

![a0cefe55b3b97884cc2df254328ab88d59004ea79d5592304e9421409950a17a.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/a0cefe55b3b97884cc2df254328ab88d59004ea79d5592304e9421409950a17a.jpg)

![bda9fa0f53cd9011ce2465d7deca5b6eeb5c2fcf36d6a5275f46d6b3afab5555.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/bda9fa0f53cd9011ce2465d7deca5b6eeb5c2fcf36d6a5275f46d6b3afab5555.jpg)

![c23f487feabdf3b9ffad22fc9f2b4b404b264eb5d41bf17195c172c062d7ac4e.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/c23f487feabdf3b9ffad22fc9f2b4b404b264eb5d41bf17195c172c062d7ac4e.jpg)

![c6ab9a161eddd16455e2ae04aefddc9dd24e1642b87644e5282f12c743f07ddb.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/c6ab9a161eddd16455e2ae04aefddc9dd24e1642b87644e5282f12c743f07ddb.jpg)

![c7d7fc0a47539284774766d8238923920989c997e926bf3bb96b72d7e5bd6e55.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/c7d7fc0a47539284774766d8238923920989c997e926bf3bb96b72d7e5bd6e55.jpg)

![c810dae5044d39e7508793ff58b1a34d26516f5e5cda8152ed02a9fe480055e8.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/c810dae5044d39e7508793ff58b1a34d26516f5e5cda8152ed02a9fe480055e8.jpg)

![d545c61e9f1a73d68700d20f675637a2d06afae17444e8632b986b8ae508dec6.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/d545c61e9f1a73d68700d20f675637a2d06afae17444e8632b986b8ae508dec6.jpg)

![eeed753b036b324c7f2fe1a1ead302a248038129d6bba6e2dfc15b91893f233c.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/eeed753b036b324c7f2fe1a1ead302a248038129d6bba6e2dfc15b91893f233c.jpg)

![ef4d17afbba360b5e6b3e99aa6901b8d57580901ac10bd0356d5d0d947bb36bb.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/ef4d17afbba360b5e6b3e99aa6901b8d57580901ac10bd0356d5d0d947bb36bb.jpg)

![ef6d208ce332127ac5147392174814efc70dc6dc8837f40a20e3d0ca51024afb.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/ef6d208ce332127ac5147392174814efc70dc6dc8837f40a20e3d0ca51024afb.jpg)

![f0bc5d5b838ab471362f15b037b81602915e8c6eb009572651d00a493d621514.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/f0bc5d5b838ab471362f15b037b81602915e8c6eb009572651d00a493d621514.jpg)

![fb79cb24fc78ba11b20fb29fb56dad80c62b4b7d1522ac39ea6460800dab2002.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/fb79cb24fc78ba11b20fb29fb56dad80c62b4b7d1522ac39ea6460800dab2002.jpg)

![ff9fd551cb9f1dd4f91c2f08e55910e12de5384907de98d2f4f25fa6df46e730.jpg](../iclr_results/3_Joint Graph Rewiring and Feature Denoising via Spectral Resonance/tables/ff9fd551cb9f1dd4f91c2f08e55910e12de5384907de98d2f4f25fa6df46e730.jpg)

## AFlow: Automating Agentic Workflow Generation


### Images

![07c2dfd93be6cbc2494df51f6d6f9ee013844c15a8825cede1044a8c6a0b0bcf.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/07c2dfd93be6cbc2494df51f6d6f9ee013844c15a8825cede1044a8c6a0b0bcf.jpg)

![2009d4191da21f7a62f21e0b6225eca00dca015c533c7667e15138f9c2341b7e.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/2009d4191da21f7a62f21e0b6225eca00dca015c533c7667e15138f9c2341b7e.jpg)

![21ed70afc16de17756e527dfebb86efbf94d667788d532286508a455cc0f08fe.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/21ed70afc16de17756e527dfebb86efbf94d667788d532286508a455cc0f08fe.jpg)

![28fd3032112430abffbf9437303c79f31be9b8474008f6530a23823966bc5d0d.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/28fd3032112430abffbf9437303c79f31be9b8474008f6530a23823966bc5d0d.jpg)

![3088730efa3be2c78130ec9e1bf82ee8bc13682ac80161fe0ad6154e28910da4.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/3088730efa3be2c78130ec9e1bf82ee8bc13682ac80161fe0ad6154e28910da4.jpg)

![3920d1ed1cabb74779825f139056ac8a132813c85be9c01a632dcb765151913e.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/3920d1ed1cabb74779825f139056ac8a132813c85be9c01a632dcb765151913e.jpg)

![3a8ce04f1ff58c32239aaebab8d09faa7a1fd287d8d037b0b2de21c65eeb4bbf.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/3a8ce04f1ff58c32239aaebab8d09faa7a1fd287d8d037b0b2de21c65eeb4bbf.jpg)

![57d1ee7045b373143f3db4a659e5b693816175cde4dddcf03e5624a6a05b10bc.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/57d1ee7045b373143f3db4a659e5b693816175cde4dddcf03e5624a6a05b10bc.jpg)

![58184cff0c3cbeb516c358ef0fe1b9e3305719c0a1ed3ec8cf49dc0052799869.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/58184cff0c3cbeb516c358ef0fe1b9e3305719c0a1ed3ec8cf49dc0052799869.jpg)

![6b92323f75d9dae9d7c327132f83802aea050e78510c14408aa3889e0ce253af.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/6b92323f75d9dae9d7c327132f83802aea050e78510c14408aa3889e0ce253af.jpg)

![8033e42da0bed836269095a9b384a656ff01c15e2a5da12740f95ef4a90db40c.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/8033e42da0bed836269095a9b384a656ff01c15e2a5da12740f95ef4a90db40c.jpg)

![84521e2ea5118e9e137ade386942b17aef494238cacfbb75c34544db0d437cfd.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/84521e2ea5118e9e137ade386942b17aef494238cacfbb75c34544db0d437cfd.jpg)

![937b88ba16f000ee1ed7a6756bde04a1886a46ac53238b20f8d4318e8afb6efa.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/937b88ba16f000ee1ed7a6756bde04a1886a46ac53238b20f8d4318e8afb6efa.jpg)

![96d32219d44624909f3d59a4ce980c624bd82d1c90e382dacaf79d7f61f7ff20.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/96d32219d44624909f3d59a4ce980c624bd82d1c90e382dacaf79d7f61f7ff20.jpg)

![98d62ae0593dce8496746f4bd86b54ee86f6439b844c3c63db660d55b3818d28.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/98d62ae0593dce8496746f4bd86b54ee86f6439b844c3c63db660d55b3818d28.jpg)

![9c01394df6a783d5a2f1f2da7b3c811b343b79f98ebfccaa4209799806ac1177.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/9c01394df6a783d5a2f1f2da7b3c811b343b79f98ebfccaa4209799806ac1177.jpg)

![9e43cf24079ed1bcb4061e438dced759be49e04ea15f0cbd7c8a52f5891725cb.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/9e43cf24079ed1bcb4061e438dced759be49e04ea15f0cbd7c8a52f5891725cb.jpg)

![ab79a4269693c3f80ccd7fc71d6a9aa84671973c604d613fd6c9ce9dd8ebc76e.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/ab79a4269693c3f80ccd7fc71d6a9aa84671973c604d613fd6c9ce9dd8ebc76e.jpg)

![be924d8775c5c70e3327137f38b864cb727259523a0c97d3f71b346fea9493fb.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/be924d8775c5c70e3327137f38b864cb727259523a0c97d3f71b346fea9493fb.jpg)

![c0ee93d49b5230076a32a54b0e938b6023cdb49e0613066012d76dd5441adb83.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/c0ee93d49b5230076a32a54b0e938b6023cdb49e0613066012d76dd5441adb83.jpg)

![c1d0ac0ac5e8dfeb015be045a6c570f1ed33d98a4abc3fc8308602fa3e980e04.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/c1d0ac0ac5e8dfeb015be045a6c570f1ed33d98a4abc3fc8308602fa3e980e04.jpg)

![c36d91b0b437171cdc15c2062af0ca43083389d17763542af57302b3f794cba9.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/c36d91b0b437171cdc15c2062af0ca43083389d17763542af57302b3f794cba9.jpg)

![deb5e8b2be802f68c46f97970213760b2f35822bcc99115847a8cb9f39ba288f.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/deb5e8b2be802f68c46f97970213760b2f35822bcc99115847a8cb9f39ba288f.jpg)

![e69faeded44a5ad1e2af19f41f3a36cedace3321331f00c092f753dd0a0f3d4e.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/e69faeded44a5ad1e2af19f41f3a36cedace3321331f00c092f753dd0a0f3d4e.jpg)

![f7fce03987dbaf62d6f2e9a2ae6db1b98e76f5c5616c9f9bb4b0ca13b0cfd9b2.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/f7fce03987dbaf62d6f2e9a2ae6db1b98e76f5c5616c9f9bb4b0ca13b0cfd9b2.jpg)

![fa5627a03f3995c86f6de9f7080b768c14b164a84b1c9058f19396d916dc1337.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/images/fa5627a03f3995c86f6de9f7080b768c14b164a84b1c9058f19396d916dc1337.jpg)

### Tables

![056363aee4b3bd0adc73bc6f56911a457efc3104c0228c29f4e9af17252ff322.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/tables/056363aee4b3bd0adc73bc6f56911a457efc3104c0228c29f4e9af17252ff322.jpg)

![3d8a9a2bdd49c2404a2dcf8be2439daccc341530f5ed830923ed6beba2249e74.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/tables/3d8a9a2bdd49c2404a2dcf8be2439daccc341530f5ed830923ed6beba2249e74.jpg)

![7307595e24bb1917ce6cf9f4d6e24398a52363ad5961ddd79bc1476f04d838a5.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/tables/7307595e24bb1917ce6cf9f4d6e24398a52363ad5961ddd79bc1476f04d838a5.jpg)

![85e7c5373a1644e2be03a04aab42a468c4ebb9cc34d5e131b87850a617dc051b.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/tables/85e7c5373a1644e2be03a04aab42a468c4ebb9cc34d5e131b87850a617dc051b.jpg)

![93daaddf74fa388c3e27e858da48b22e8547276e57f8fb589fe76048a9a07f79.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/tables/93daaddf74fa388c3e27e858da48b22e8547276e57f8fb589fe76048a9a07f79.jpg)

![942076585790946550bb740e9a673172996b974d0df0df37239865a9a4e03907.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/tables/942076585790946550bb740e9a673172996b974d0df0df37239865a9a4e03907.jpg)

![b80541550696137e96a47dbcc6cbfdaaaa190f9dcb744aea107eb9d21af306e3.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/tables/b80541550696137e96a47dbcc6cbfdaaaa190f9dcb744aea107eb9d21af306e3.jpg)

![d64a6cc1d87f3fe33f0eeaedb2f02b7f13363fe4088860d4fc832934dd3d2218.jpg](../iclr_results/4_AFlow_ Automating Agentic Workflow Generation/tables/d64a6cc1d87f3fe33f0eeaedb2f02b7f13363fe4088860d4fc832934dd3d2218.jpg)

## Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo


### Images

![0ea4d0aa4d1eadd06526e17a38430eca4a29a340ea1c595aca590aed47a6fb3b.jpg](../iclr_results/6_Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo/images/0ea4d0aa4d1eadd06526e17a38430eca4a29a340ea1c595aca590aed47a6fb3b.jpg)

![4cd58800c6c53024f928894e02821b29a1b033d440b1f33783a07a815c5b8a63.jpg](../iclr_results/6_Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo/images/4cd58800c6c53024f928894e02821b29a1b033d440b1f33783a07a815c5b8a63.jpg)

![62544f911cff1064da220422e510dce0c655357f82c6e931cd6cee9bc2bae5df.jpg](../iclr_results/6_Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo/images/62544f911cff1064da220422e510dce0c655357f82c6e931cd6cee9bc2bae5df.jpg)

![64fa87cccd36cb0f7f23fab0b3367fc53559ee7f56bde1f4d9308193e83c495d.jpg](../iclr_results/6_Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo/images/64fa87cccd36cb0f7f23fab0b3367fc53559ee7f56bde1f4d9308193e83c495d.jpg)

### Tables

![0e9cda54460b60a7bb9faaab786bf341af3c20139fdf1f12fc54960d2d489797.jpg](../iclr_results/6_Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo/tables/0e9cda54460b60a7bb9faaab786bf341af3c20139fdf1f12fc54960d2d489797.jpg)

![35e77e50cf7ae138ab7bf1c40810cae07f34ca5deec0cf3706bfea07e5add866.jpg](../iclr_results/6_Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo/tables/35e77e50cf7ae138ab7bf1c40810cae07f34ca5deec0cf3706bfea07e5add866.jpg)

![574fc7c671127253ecfe8df8aa518595a52281d8a977eb5fe578fd3c74569c70.jpg](../iclr_results/6_Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo/tables/574fc7c671127253ecfe8df8aa518595a52281d8a977eb5fe578fd3c74569c70.jpg)

![8319c5ef5d9d020ab1038bf4f636a936440f0936a9bc260e9858c8cc240c0d95.jpg](../iclr_results/6_Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo/tables/8319c5ef5d9d020ab1038bf4f636a936440f0936a9bc260e9858c8cc240c0d95.jpg)

![9c8389def9a1c231131e3efb8d67c3a9f87602732042109bae74c29867637664.jpg](../iclr_results/6_Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo/tables/9c8389def9a1c231131e3efb8d67c3a9f87602732042109bae74c29867637664.jpg)

![9e97fcfb1cb39130ab11ca8d4d3898b65f5573d9b22ea732548abc2a9d9220d0.jpg](../iclr_results/6_Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo/tables/9e97fcfb1cb39130ab11ca8d4d3898b65f5573d9b22ea732548abc2a9d9220d0.jpg)

![b1876cdbb7059a96ecca7e14535717bae83a76c37687e2c7c83c93f787c7d224.jpg](../iclr_results/6_Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo/tables/b1876cdbb7059a96ecca7e14535717bae83a76c37687e2c7c83c93f787c7d224.jpg)

![ba9ac89d7161aae8a5cbaebdcd2ed7d3598c4a31ce12bcaa0ddaaec7a2b53d4a.jpg](../iclr_results/6_Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo/tables/ba9ac89d7161aae8a5cbaebdcd2ed7d3598c4a31ce12bcaa0ddaaec7a2b53d4a.jpg)

![c3bcaf6fa5393b110eca573ebebbb5b1090b6603cf9f0edf3e6079d9bc6da7c5.jpg](../iclr_results/6_Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo/tables/c3bcaf6fa5393b110eca573ebebbb5b1090b6603cf9f0edf3e6079d9bc6da7c5.jpg)

![e0cca2c212a1e64dade4de34ff0c21d319dc00dc90bea35c3e655a9dace11be9.jpg](../iclr_results/6_Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo/tables/e0cca2c212a1e64dade4de34ff0c21d319dc00dc90bea35c3e655a9dace11be9.jpg)

## Can a MISL Fly? Analysis and Ingredients for Mutual Information Skill Learning


### Images

![2af0d8e49119b51797ec8be82f9a3e8bfc702d6dd252aa7d5b51c5fd652679d4.jpg](../iclr_results/7_Can a MISL Fly_ Analysis and Ingredients for Mutual Information Skill Learning/images/2af0d8e49119b51797ec8be82f9a3e8bfc702d6dd252aa7d5b51c5fd652679d4.jpg)

![3f9c2e4e91060b46f0a751f0fe47928076e4fa61e43d30bec3797271f56466cd.jpg](../iclr_results/7_Can a MISL Fly_ Analysis and Ingredients for Mutual Information Skill Learning/images/3f9c2e4e91060b46f0a751f0fe47928076e4fa61e43d30bec3797271f56466cd.jpg)

![40529486a6561e5c398b04cfed6bf791da5978631d6ff35cace4d0b1f6f93cdf.jpg](../iclr_results/7_Can a MISL Fly_ Analysis and Ingredients for Mutual Information Skill Learning/images/40529486a6561e5c398b04cfed6bf791da5978631d6ff35cace4d0b1f6f93cdf.jpg)

![44c253247ffff46a7a97fc92a7a4b16ca3f3a74a3287b43cff4d5e787f4a4dcc.jpg](../iclr_results/7_Can a MISL Fly_ Analysis and Ingredients for Mutual Information Skill Learning/images/44c253247ffff46a7a97fc92a7a4b16ca3f3a74a3287b43cff4d5e787f4a4dcc.jpg)

![8c25e5f2f161d7f29621226007aed9273efd861ec08fbaafa32beff9500ee4d9.jpg](../iclr_results/7_Can a MISL Fly_ Analysis and Ingredients for Mutual Information Skill Learning/images/8c25e5f2f161d7f29621226007aed9273efd861ec08fbaafa32beff9500ee4d9.jpg)

![9268e1227112102fef77c08fcef8452087a1881df1293093ce52e136294250f1.jpg](../iclr_results/7_Can a MISL Fly_ Analysis and Ingredients for Mutual Information Skill Learning/images/9268e1227112102fef77c08fcef8452087a1881df1293093ce52e136294250f1.jpg)

![9d87ea04cf009e2155bd77c9444903f66e9659637513c7c38c07d17cafa3290f.jpg](../iclr_results/7_Can a MISL Fly_ Analysis and Ingredients for Mutual Information Skill Learning/images/9d87ea04cf009e2155bd77c9444903f66e9659637513c7c38c07d17cafa3290f.jpg)

![a5cea90d707771e8841f765462034e71bba97e4311fd28076fcd2484f509b671.jpg](../iclr_results/7_Can a MISL Fly_ Analysis and Ingredients for Mutual Information Skill Learning/images/a5cea90d707771e8841f765462034e71bba97e4311fd28076fcd2484f509b671.jpg)

![c1fb10279c13e79d272da0380c3319ff557c0fe770cacdfc733dfe6dcf516411.jpg](../iclr_results/7_Can a MISL Fly_ Analysis and Ingredients for Mutual Information Skill Learning/images/c1fb10279c13e79d272da0380c3319ff557c0fe770cacdfc733dfe6dcf516411.jpg)

![e722e5cc852aee38f13a2fef3fdc001e973d75f0c198b5b9792c77965a4fcea7.jpg](../iclr_results/7_Can a MISL Fly_ Analysis and Ingredients for Mutual Information Skill Learning/images/e722e5cc852aee38f13a2fef3fdc001e973d75f0c198b5b9792c77965a4fcea7.jpg)

![f69f3dd4ac8be1e1459c68c07ad041d2b8db3ff390b341d359e25d39eea4e28e.jpg](../iclr_results/7_Can a MISL Fly_ Analysis and Ingredients for Mutual Information Skill Learning/images/f69f3dd4ac8be1e1459c68c07ad041d2b8db3ff390b341d359e25d39eea4e28e.jpg)

![ff3508fae1868215ba07b8c750c50a5e9ec2497dd5a86b2a60b3719c540b1e0a.jpg](../iclr_results/7_Can a MISL Fly_ Analysis and Ingredients for Mutual Information Skill Learning/images/ff3508fae1868215ba07b8c750c50a5e9ec2497dd5a86b2a60b3719c540b1e0a.jpg)

### Tables

![374d1aa79a668648c1956346205e3c13cd6ac76ebb12412d3d06974bbe2ceac5.jpg](../iclr_results/7_Can a MISL Fly_ Analysis and Ingredients for Mutual Information Skill Learning/tables/374d1aa79a668648c1956346205e3c13cd6ac76ebb12412d3d06974bbe2ceac5.jpg)

![57d8fc9301c12b66faefa186e76466fe92663bb78f5af6a24e5e9c777ce67c40.jpg](../iclr_results/7_Can a MISL Fly_ Analysis and Ingredients for Mutual Information Skill Learning/tables/57d8fc9301c12b66faefa186e76466fe92663bb78f5af6a24e5e9c777ce67c40.jpg)

![77b31aec535ebf7ec035d5293bdba907edaca86b8af8e5a1953fcf963c7a7e10.jpg](../iclr_results/7_Can a MISL Fly_ Analysis and Ingredients for Mutual Information Skill Learning/tables/77b31aec535ebf7ec035d5293bdba907edaca86b8af8e5a1953fcf963c7a7e10.jpg)

## OLMoE: Open Mixture-of-Experts Language Models


### Images

![07827a871d3530b1ddad1e728f20699f8262885704a9c0d60d389ff98ad65fde.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/07827a871d3530b1ddad1e728f20699f8262885704a9c0d60d389ff98ad65fde.jpg)

![08060cc028509c0dc81a8d40c294ca381d40e6eac81ff4c125abfc3eb2ea1863.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/08060cc028509c0dc81a8d40c294ca381d40e6eac81ff4c125abfc3eb2ea1863.jpg)

![082467fb427f59ed8a6cf97c5a368a592ee7ac08554ea2e44cac8322299e89d7.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/082467fb427f59ed8a6cf97c5a368a592ee7ac08554ea2e44cac8322299e89d7.jpg)

![08bd21230bd7238cc211052abb54a2099f19e610f10007ab2cdcad1bc93486fc.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/08bd21230bd7238cc211052abb54a2099f19e610f10007ab2cdcad1bc93486fc.jpg)

![093255a1888b8de660edb3ecbd6f3867dd6bd0b7eac6a1e71652d144982b7289.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/093255a1888b8de660edb3ecbd6f3867dd6bd0b7eac6a1e71652d144982b7289.jpg)

![0b08fda22857388af9348a4ecf7fdb7bd297d703d0ad76e7a4aa3afb20d9a7c7.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/0b08fda22857388af9348a4ecf7fdb7bd297d703d0ad76e7a4aa3afb20d9a7c7.jpg)

![1e78565d08087c5ee6248805d091c32803c09b050ff2311f76028340e36fb3b2.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/1e78565d08087c5ee6248805d091c32803c09b050ff2311f76028340e36fb3b2.jpg)

![1fc8a31d8ee498d7ce45b15daf66d3a05aaa7010d9860c14a2f99c2fc57507cc.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/1fc8a31d8ee498d7ce45b15daf66d3a05aaa7010d9860c14a2f99c2fc57507cc.jpg)

![23e685567a36c57cadcfda571722c8454709ef9f9bb4edc3d88d2536da6c6344.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/23e685567a36c57cadcfda571722c8454709ef9f9bb4edc3d88d2536da6c6344.jpg)

![240ab24fe7e1f81e25c9680b01b83f8bb96028b34ee56b88025b5d42a202759a.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/240ab24fe7e1f81e25c9680b01b83f8bb96028b34ee56b88025b5d42a202759a.jpg)

![3eae68f032daad961702969c00ca765fb55d1dd33446c718b2ce3349f573895f.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/3eae68f032daad961702969c00ca765fb55d1dd33446c718b2ce3349f573895f.jpg)

![3ef72bfb66bc8f32b0abea3563be3628caca8957c5f0f6d3ca05915845ef1d28.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/3ef72bfb66bc8f32b0abea3563be3628caca8957c5f0f6d3ca05915845ef1d28.jpg)

![4e9a5862a72f7bd4eada5c04002abc39c49a2a3a5ec38f4668af295b7d6bb18c.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/4e9a5862a72f7bd4eada5c04002abc39c49a2a3a5ec38f4668af295b7d6bb18c.jpg)

![50ef55528459a66af295a7d08d496b28eed89b738a106f86c1ace98c172a4d33.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/50ef55528459a66af295a7d08d496b28eed89b738a106f86c1ace98c172a4d33.jpg)

![50f5bc38e548021d9adf559fd9b7ec474b6f4e545cf05e011d326e61b3790c8d.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/50f5bc38e548021d9adf559fd9b7ec474b6f4e545cf05e011d326e61b3790c8d.jpg)

![645736fbf47ef247bb2172eae5b6f419fba0d8f062fdecb0baaa619986b5ef3b.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/645736fbf47ef247bb2172eae5b6f419fba0d8f062fdecb0baaa619986b5ef3b.jpg)

![70b69710941ffd9fb00a562d7cf719259b7ef9fe3bb6f8ec3f4e523329a8de68.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/70b69710941ffd9fb00a562d7cf719259b7ef9fe3bb6f8ec3f4e523329a8de68.jpg)

![74bd9b6ca483598aeecaa5e99880ced122ce97c6d625e3f52af8013a614e89af.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/74bd9b6ca483598aeecaa5e99880ced122ce97c6d625e3f52af8013a614e89af.jpg)

![74c881c0bb669ff3234d47c790428916d309a09ee272fe45d79cf7f83cad058a.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/74c881c0bb669ff3234d47c790428916d309a09ee272fe45d79cf7f83cad058a.jpg)

![9ec31d3cbbb6a792844e0a07f0b12b3f4efa9048509b8dfa7da0fd7ed4793b28.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/9ec31d3cbbb6a792844e0a07f0b12b3f4efa9048509b8dfa7da0fd7ed4793b28.jpg)

![a5f7e32f6d940318b8f72466ddf91dc681c93834e0b01eeccdf3cd99a0112194.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/a5f7e32f6d940318b8f72466ddf91dc681c93834e0b01eeccdf3cd99a0112194.jpg)

![a862b3da8a85e3922bc22005f71210ab80d99cafd6a6d107245eae5931109039.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/a862b3da8a85e3922bc22005f71210ab80d99cafd6a6d107245eae5931109039.jpg)

![ad06c66ab79c7a45d7b4b34fba0ecee4a5a3cef2a39fa37dfa11055d02171b51.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/ad06c66ab79c7a45d7b4b34fba0ecee4a5a3cef2a39fa37dfa11055d02171b51.jpg)

![ad2e4f2898d4f8f0fba813b1349f7cd01052f19cb46730d72a3ea8647f0905f5.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/ad2e4f2898d4f8f0fba813b1349f7cd01052f19cb46730d72a3ea8647f0905f5.jpg)

![b38b6cfca10ca3c6f300a2ba1510d1fe2b2afa456009b9adbf89068491525099.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/b38b6cfca10ca3c6f300a2ba1510d1fe2b2afa456009b9adbf89068491525099.jpg)

![c1a86665d213aa9b071e654155c65bbf099af7275e94380d8c45ad59687e8710.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/c1a86665d213aa9b071e654155c65bbf099af7275e94380d8c45ad59687e8710.jpg)

![c2406d9ec5e933793a34d5a9c055fd46990d6842dd4f56b7a04e222725a18ba7.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/c2406d9ec5e933793a34d5a9c055fd46990d6842dd4f56b7a04e222725a18ba7.jpg)

![c699f0e926227c9c03cca6a81c129226778513f6f1d1729940b9f27c5c499a82.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/c699f0e926227c9c03cca6a81c129226778513f6f1d1729940b9f27c5c499a82.jpg)

![c8563519c54d5a6248687c9d29f1092bf57a1a7ca782ee8e0cf6d6b83988ba0a.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/c8563519c54d5a6248687c9d29f1092bf57a1a7ca782ee8e0cf6d6b83988ba0a.jpg)

![c953084eff73461fcd169ff02154a3dc8d60e77d2c605c3aa5f4bb555c018500.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/c953084eff73461fcd169ff02154a3dc8d60e77d2c605c3aa5f4bb555c018500.jpg)

![d4271a436e1b240a9890cda547c8c21a8c4c2663680df2644a7fc600fc4a7f03.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/d4271a436e1b240a9890cda547c8c21a8c4c2663680df2644a7fc600fc4a7f03.jpg)

![e1293042832e51b0dbf8324a2fac5a0c09b57300539b3918599250d5265bf060.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/e1293042832e51b0dbf8324a2fac5a0c09b57300539b3918599250d5265bf060.jpg)

![ee1613f2ff0d107e52dcdbfb07ada43a5eba27facddf745b6d2ae757e56cf006.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/ee1613f2ff0d107e52dcdbfb07ada43a5eba27facddf745b6d2ae757e56cf006.jpg)

![f0fcbf976ad1b8a678f5e60948b0b376061b2f8a92660ffa30f555508cc3a240.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/f0fcbf976ad1b8a678f5e60948b0b376061b2f8a92660ffa30f555508cc3a240.jpg)

![f50bef42fb70ff9c51c526f838b996dc1edf0a131e669c7b99566ae877d2e963.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/f50bef42fb70ff9c51c526f838b996dc1edf0a131e669c7b99566ae877d2e963.jpg)

![f61d4ef32296b617742b818892524281991805ebd0d752aaf9a49c8449fcfc65.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/f61d4ef32296b617742b818892524281991805ebd0d752aaf9a49c8449fcfc65.jpg)

![f69d237bc94977dbf88045cd613860946c42495d3aea92a64f2fcc09d003987d.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/f69d237bc94977dbf88045cd613860946c42495d3aea92a64f2fcc09d003987d.jpg)

![f99eda56cdf95d6b68dea4d2b26d0650ad74739fcef399199ee9276de337551d.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/images/f99eda56cdf95d6b68dea4d2b26d0650ad74739fcef399199ee9276de337551d.jpg)

### Tables

![007fdc402ac0fb4b90c4572cb7d8639c8d461e180dc27fcf38c92f01dd284c62.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/tables/007fdc402ac0fb4b90c4572cb7d8639c8d461e180dc27fcf38c92f01dd284c62.jpg)

![0a0b510efb3fa212333ee8062b7bffc4c647d14d29f2380272d97d68a510cc3f.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/tables/0a0b510efb3fa212333ee8062b7bffc4c647d14d29f2380272d97d68a510cc3f.jpg)

![12d9e147395dcc9b9ce2b6da6521d38414631b6c0fcdb6d0b1a042402a05629a.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/tables/12d9e147395dcc9b9ce2b6da6521d38414631b6c0fcdb6d0b1a042402a05629a.jpg)

![190ee4ca0d467781b84bd83e68cc4058266afb9ce94ed91b82de6c3fe9ff8622.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/tables/190ee4ca0d467781b84bd83e68cc4058266afb9ce94ed91b82de6c3fe9ff8622.jpg)

![1b12a5266ea881b18127c805e6b0a5ecc2095a31a116fac7d63874a30398a129.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/tables/1b12a5266ea881b18127c805e6b0a5ecc2095a31a116fac7d63874a30398a129.jpg)

![22549ffabfc427e674105ecdd59627fae6ae0c926503949599d87a73acabc241.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/tables/22549ffabfc427e674105ecdd59627fae6ae0c926503949599d87a73acabc241.jpg)

![322f9bcf04e8868b50adb54271af6f4b8a2d36838be802a5e6b128dcebebe249.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/tables/322f9bcf04e8868b50adb54271af6f4b8a2d36838be802a5e6b128dcebebe249.jpg)

![339887db198fa2622f244a725fb5532df2191364a9e72596c91ab53c2e22f73b.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/tables/339887db198fa2622f244a725fb5532df2191364a9e72596c91ab53c2e22f73b.jpg)

![343b1bcbfe468db10eeedb2ea39497f3c0d950e8f871a7c8737e6d284880b164.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/tables/343b1bcbfe468db10eeedb2ea39497f3c0d950e8f871a7c8737e6d284880b164.jpg)

![690b0a636996a89839632b0f182f3a7efa6e8351e415f6a1ccf937e0caf007f3.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/tables/690b0a636996a89839632b0f182f3a7efa6e8351e415f6a1ccf937e0caf007f3.jpg)

![76274182c032a86a39c9f819e10946487c4e078f0152b27f55837b66c0250391.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/tables/76274182c032a86a39c9f819e10946487c4e078f0152b27f55837b66c0250391.jpg)

![7dfd0838ef6294709916de28309081bbdc34e1391bc9752172738a0fc46c0c26.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/tables/7dfd0838ef6294709916de28309081bbdc34e1391bc9752172738a0fc46c0c26.jpg)

![8923cc4a48270fd8bfef88a1142863dbc9372223eab2b27655ea8529ea9aebe4.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/tables/8923cc4a48270fd8bfef88a1142863dbc9372223eab2b27655ea8529ea9aebe4.jpg)

![a26305706243d7868cad7b1fa838153b792e10c1ce2b027480e790475eef7012.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/tables/a26305706243d7868cad7b1fa838153b792e10c1ce2b027480e790475eef7012.jpg)

![c2f3cc660b4c2b4311c29b6ff8696851cf5b4327e3dc32c79cb4d793f47a4c81.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/tables/c2f3cc660b4c2b4311c29b6ff8696851cf5b4327e3dc32c79cb4d793f47a4c81.jpg)

![ddb2c0b362c1aa67982045648250d3a02333fdaf4e49e4e2603329f9cbd9c954.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/tables/ddb2c0b362c1aa67982045648250d3a02333fdaf4e49e4e2603329f9cbd9c954.jpg)

![eff02e4daf5c898958569c8c40ce68636e42e1e995e25811c059047cb3ff750b.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/tables/eff02e4daf5c898958569c8c40ce68636e42e1e995e25811c059047cb3ff750b.jpg)

![ffd1fa1c4635b8f91d64cfac01bbfa40a47804ef8eb46cd120de1bdd489f3f98.jpg](../iclr_results/8_OLMoE_ Open Mixture-of-Experts Language Models/tables/ffd1fa1c4635b8f91d64cfac01bbfa40a47804ef8eb46cd120de1bdd489f3f98.jpg)

## Learning to Discretize Denoising Diffusion ODEs


### Images

![0382d881ff5e8efa2e2941063f25afbe9a294b89587fe4c042dd9f890c2b1a27.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/images/0382d881ff5e8efa2e2941063f25afbe9a294b89587fe4c042dd9f890c2b1a27.jpg)

![0efd1e9bc092fcb83508609d1091bec9dac4bc7d64d9b977eaad77a2729b2b1e.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/images/0efd1e9bc092fcb83508609d1091bec9dac4bc7d64d9b977eaad77a2729b2b1e.jpg)

![0f57dd24f45764d76e4c4540f423f8f59943dba77f6d3eee147d0ade9f645ae5.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/images/0f57dd24f45764d76e4c4540f423f8f59943dba77f6d3eee147d0ade9f645ae5.jpg)

![1e004c1e6cfaf5b2a7b87f3a09aacd9ec73cf63c1638b109aa1bea470ab6df45.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/images/1e004c1e6cfaf5b2a7b87f3a09aacd9ec73cf63c1638b109aa1bea470ab6df45.jpg)

![20cf36eee15fceaf43cc4467b98b3c7c421a1a4d109c6b30722875edb45a4e17.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/images/20cf36eee15fceaf43cc4467b98b3c7c421a1a4d109c6b30722875edb45a4e17.jpg)

![23445deb5d70fd14373c21eaee84b00bace3a52051bae7e647dd871b5bf7701a.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/images/23445deb5d70fd14373c21eaee84b00bace3a52051bae7e647dd871b5bf7701a.jpg)

![3b042ffa0bbce156aa185b9ad00721701870211d082512d13b1324ace08973c8.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/images/3b042ffa0bbce156aa185b9ad00721701870211d082512d13b1324ace08973c8.jpg)

![47a4f7325e4afaaaa6cfc6971401bf371e4310140437ae5a85590da348d4f11d.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/images/47a4f7325e4afaaaa6cfc6971401bf371e4310140437ae5a85590da348d4f11d.jpg)

![59b3815279dad0ab93cda11635afbef8144707eef5474c689b816f02ebd4e868.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/images/59b3815279dad0ab93cda11635afbef8144707eef5474c689b816f02ebd4e868.jpg)

![5ae959599802b6247b9a862a427be56cc61fdfad65c5e3ec06baa3e864a70de6.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/images/5ae959599802b6247b9a862a427be56cc61fdfad65c5e3ec06baa3e864a70de6.jpg)

![88ae77327a5d3bd87f13bcc0ad1d7cab79e88f0cb7f09a2dca0abfff02c4e7fd.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/images/88ae77327a5d3bd87f13bcc0ad1d7cab79e88f0cb7f09a2dca0abfff02c4e7fd.jpg)

![8ca38983ccf93b6ed105ced722b03cbe840cb017102582664433bcada5b9d9e9.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/images/8ca38983ccf93b6ed105ced722b03cbe840cb017102582664433bcada5b9d9e9.jpg)

![8e48746ffb42ea11cea964926f00a76a95bb51b4a230c910fa4cadad912522ee.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/images/8e48746ffb42ea11cea964926f00a76a95bb51b4a230c910fa4cadad912522ee.jpg)

![96db10fcee3811c9876484e4c9f0a792f66d1c47490f1e590a544e706a0ff0fb.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/images/96db10fcee3811c9876484e4c9f0a792f66d1c47490f1e590a544e706a0ff0fb.jpg)

![9a6263c765e550fea761a16da6354f0fc16470e5a22bec6aeedf45b52a9e165a.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/images/9a6263c765e550fea761a16da6354f0fc16470e5a22bec6aeedf45b52a9e165a.jpg)

![9ac68ba458685f98149f3ac80d51c76392c0c78f59b5a06244387c89ca70493b.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/images/9ac68ba458685f98149f3ac80d51c76392c0c78f59b5a06244387c89ca70493b.jpg)

![a5ec69b08582c423fa4ef4b9151df382186b541192a9211c44d1060437f8c1ba.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/images/a5ec69b08582c423fa4ef4b9151df382186b541192a9211c44d1060437f8c1ba.jpg)

![b572cb3cd158315d355ffc714187e2a172bee587ef7b83b7d28f413713bee773.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/images/b572cb3cd158315d355ffc714187e2a172bee587ef7b83b7d28f413713bee773.jpg)

![ba01096c48c9e162d58f5863393f13e7fac484e460fa92f5bff23a8eb5b7d486.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/images/ba01096c48c9e162d58f5863393f13e7fac484e460fa92f5bff23a8eb5b7d486.jpg)

![c94e6a1d25fd75c63e9711736e9841cae795dd769a9bc59afca7ef80d9877e34.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/images/c94e6a1d25fd75c63e9711736e9841cae795dd769a9bc59afca7ef80d9877e34.jpg)

![cc95f64cfdf3ae326d354be17891e240b9d4238da497a65bdf27037e2ebea072.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/images/cc95f64cfdf3ae326d354be17891e240b9d4238da497a65bdf27037e2ebea072.jpg)

![d02abbcf2f4606d0342d310d3b9feb48f663950b6f8b2e54d46cb98afbe4b517.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/images/d02abbcf2f4606d0342d310d3b9feb48f663950b6f8b2e54d46cb98afbe4b517.jpg)

![e946e83186ef1e41e6702d72d450013c0abb19ea5f03b7c4b8b096c60fff7b0e.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/images/e946e83186ef1e41e6702d72d450013c0abb19ea5f03b7c4b8b096c60fff7b0e.jpg)

### Tables

![1515c3307af1866712b50f48af2d910a9018b124d30480b9691933f577bc6d63.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/tables/1515c3307af1866712b50f48af2d910a9018b124d30480b9691933f577bc6d63.jpg)

![16f885dc6a2980765abe7aa0b2c5a368baa669e075d0d17babb38678c16791bd.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/tables/16f885dc6a2980765abe7aa0b2c5a368baa669e075d0d17babb38678c16791bd.jpg)

![30130acd4b73f67e944afc5ae6cca5466de6cfdf7068a2c9fcc1584f7d163b1a.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/tables/30130acd4b73f67e944afc5ae6cca5466de6cfdf7068a2c9fcc1584f7d163b1a.jpg)

![314c9410fa7c248fa86cef3ea0dc53a1b6e79c6c62c778157b370470bfa1d121.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/tables/314c9410fa7c248fa86cef3ea0dc53a1b6e79c6c62c778157b370470bfa1d121.jpg)

![3d896b391139f42eda5003ab4acf21df9d2a7287557a6af836b02bee29ea0123.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/tables/3d896b391139f42eda5003ab4acf21df9d2a7287557a6af836b02bee29ea0123.jpg)

![3e61664fe21e0d04173677665a06f388463cfdace98b0a75067b166668d11dfc.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/tables/3e61664fe21e0d04173677665a06f388463cfdace98b0a75067b166668d11dfc.jpg)

![458b4c9ada4b93b9eef4895aae81e7570f25fdc5cbe7621f50ccac86c6ee17c0.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/tables/458b4c9ada4b93b9eef4895aae81e7570f25fdc5cbe7621f50ccac86c6ee17c0.jpg)

![585380f82f5d55b838411b273522cd373a68a42008f5a05548ff99f460c81ae0.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/tables/585380f82f5d55b838411b273522cd373a68a42008f5a05548ff99f460c81ae0.jpg)

![5f161d7ec993bde0e89c5545e0c665b50749fa4dbd610b81bb73f4afe1c83e78.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/tables/5f161d7ec993bde0e89c5545e0c665b50749fa4dbd610b81bb73f4afe1c83e78.jpg)

![605223e532adb9a2a705a40cada14c5199812425d9a2d6d3b381a27a5482bcb8.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/tables/605223e532adb9a2a705a40cada14c5199812425d9a2d6d3b381a27a5482bcb8.jpg)

![6243a110747b7c2f7487299060c3e7ff784bcf25e433d8800eaac23b4da2c0f4.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/tables/6243a110747b7c2f7487299060c3e7ff784bcf25e433d8800eaac23b4da2c0f4.jpg)

![69867e059a0673d674bdc9fd7f2ca300a92a4fa97fcd0322b596513cc95657c4.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/tables/69867e059a0673d674bdc9fd7f2ca300a92a4fa97fcd0322b596513cc95657c4.jpg)

![6d0ee0a7acdb4e06837b8a29bccf8f18280fc6eecd7e53ac93b9b70ea010b82d.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/tables/6d0ee0a7acdb4e06837b8a29bccf8f18280fc6eecd7e53ac93b9b70ea010b82d.jpg)

![811d6e191529c72ef6a63fe1c6252144389b2b2305aebee7311eb678201a4ad9.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/tables/811d6e191529c72ef6a63fe1c6252144389b2b2305aebee7311eb678201a4ad9.jpg)

![b01bfc69feef92151112443bea24b52dcfb0e7ed9641339ab95b58c7571a2277.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/tables/b01bfc69feef92151112443bea24b52dcfb0e7ed9641339ab95b58c7571a2277.jpg)

![b4a0a97578b6192b50af731e4825bee794ad1ff9b9de6c88e0d42c5b62d37a71.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/tables/b4a0a97578b6192b50af731e4825bee794ad1ff9b9de6c88e0d42c5b62d37a71.jpg)

![b52b7e4da5311318da9809ffb3c50030d340aa4eb80c34119963d7ded0b9baa1.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/tables/b52b7e4da5311318da9809ffb3c50030d340aa4eb80c34119963d7ded0b9baa1.jpg)

![c5ce78bf3208554ade2c219bdfe15ca2f65adeea71b1835dc3d9556edb92e182.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/tables/c5ce78bf3208554ade2c219bdfe15ca2f65adeea71b1835dc3d9556edb92e182.jpg)

![e3662fdf5af0910044372105322113f10877ca216b8b64d80c73098cf6fc66a2.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/tables/e3662fdf5af0910044372105322113f10877ca216b8b64d80c73098cf6fc66a2.jpg)

![ebe15354dfe669462e2dcdc616a6b516926ebe3f47d728549d53e7060b2b6a2a.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/tables/ebe15354dfe669462e2dcdc616a6b516926ebe3f47d728549d53e7060b2b6a2a.jpg)

![f1f1ba06a85a9ea13f79a0c1a5bbc7a91a282f7669f558e7d2d76f5777de9a95.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/tables/f1f1ba06a85a9ea13f79a0c1a5bbc7a91a282f7669f558e7d2d76f5777de9a95.jpg)

![f3db6e000f63614d63f90006882da90efb86f72b5da4c649d3d499f8f1fdcc1f.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/tables/f3db6e000f63614d63f90006882da90efb86f72b5da4c649d3d499f8f1fdcc1f.jpg)

![fced1190063d179900b4496e12ad4bffea736fdf7b5f5874302c66daefd17af0.jpg](../iclr_results/9_Learning to Discretize Denoising Diffusion ODEs/tables/fced1190063d179900b4496e12ad4bffea736fdf7b5f5874302c66daefd17af0.jpg)

## When Selection Meets Intervention: Additional Complexities in Causal Discovery


### Images

![376b1937a781d8865b9d948934306fcbe6a2aada662a0997a8307f2cabd46f65.jpg](../iclr_results/10_When Selection Meets Intervention_ Additional Complexities in Causal Discovery/images/376b1937a781d8865b9d948934306fcbe6a2aada662a0997a8307f2cabd46f65.jpg)

![404b19b004ea818ddc4e5564aeb4df32946f4cf24ff2ab6273bf3c2a9c802c9c.jpg](../iclr_results/10_When Selection Meets Intervention_ Additional Complexities in Causal Discovery/images/404b19b004ea818ddc4e5564aeb4df32946f4cf24ff2ab6273bf3c2a9c802c9c.jpg)

![4a0322e6e058f0d28445a99be6a9142e2a317263eb5fbfe9ff1ec34ca36f8568.jpg](../iclr_results/10_When Selection Meets Intervention_ Additional Complexities in Causal Discovery/images/4a0322e6e058f0d28445a99be6a9142e2a317263eb5fbfe9ff1ec34ca36f8568.jpg)

![5aa477f4030862a61b71be9f25a9f91628b733f687bd94ed0d26823dc7e59319.jpg](../iclr_results/10_When Selection Meets Intervention_ Additional Complexities in Causal Discovery/images/5aa477f4030862a61b71be9f25a9f91628b733f687bd94ed0d26823dc7e59319.jpg)

![6fc01ff5431432d305f6ae9e18935b8df84b70e98625de4d0bf990cb84dcf68d.jpg](../iclr_results/10_When Selection Meets Intervention_ Additional Complexities in Causal Discovery/images/6fc01ff5431432d305f6ae9e18935b8df84b70e98625de4d0bf990cb84dcf68d.jpg)

![71c0bfd8186ccd2d522bf5d1a795fb091f239d3ae3788b3c7377d3cfa375711f.jpg](../iclr_results/10_When Selection Meets Intervention_ Additional Complexities in Causal Discovery/images/71c0bfd8186ccd2d522bf5d1a795fb091f239d3ae3788b3c7377d3cfa375711f.jpg)

![7cd5364f8f90031bfd19f9349ba6f8272099585301e345cc548a7e29d1f2bb0c.jpg](../iclr_results/10_When Selection Meets Intervention_ Additional Complexities in Causal Discovery/images/7cd5364f8f90031bfd19f9349ba6f8272099585301e345cc548a7e29d1f2bb0c.jpg)

![8d7ed4751f38b6d7e383844d8bd090721517329d6fa7f07e24b2269114c40d26.jpg](../iclr_results/10_When Selection Meets Intervention_ Additional Complexities in Causal Discovery/images/8d7ed4751f38b6d7e383844d8bd090721517329d6fa7f07e24b2269114c40d26.jpg)

![9f023e59890591408d9808d4c19eeff2233cc7ba5904cb36e11a697145e5bfe5.jpg](../iclr_results/10_When Selection Meets Intervention_ Additional Complexities in Causal Discovery/images/9f023e59890591408d9808d4c19eeff2233cc7ba5904cb36e11a697145e5bfe5.jpg)

![c88310bf4a177fd6d4a5d8feaa96b7fdc4e4dc21946eabce275fdf800b9d766d.jpg](../iclr_results/10_When Selection Meets Intervention_ Additional Complexities in Causal Discovery/images/c88310bf4a177fd6d4a5d8feaa96b7fdc4e4dc21946eabce275fdf800b9d766d.jpg)

![ce0aca314ee848c78e3a0df5b6621f4c56605b1bf601540ebee9dc1e2328ba2c.jpg](../iclr_results/10_When Selection Meets Intervention_ Additional Complexities in Causal Discovery/images/ce0aca314ee848c78e3a0df5b6621f4c56605b1bf601540ebee9dc1e2328ba2c.jpg)

![db7817bf8dd6c0336f7baa3dd98c0eddd29ede9bd0e04acdb40e552ec3aad368.jpg](../iclr_results/10_When Selection Meets Intervention_ Additional Complexities in Causal Discovery/images/db7817bf8dd6c0336f7baa3dd98c0eddd29ede9bd0e04acdb40e552ec3aad368.jpg)

### Tables

![921f4de13aeb28365f79b82c59191bebf057b019f9950932fdf29486749fb5dd.jpg](../iclr_results/10_When Selection Meets Intervention_ Additional Complexities in Causal Discovery/tables/921f4de13aeb28365f79b82c59191bebf057b019f9950932fdf29486749fb5dd.jpg)

## Scaling Laws for Precision


### Images

![2893b341e8fc191eab765b3c8326418a33a2853f75f582c3bd2273412b82f349.jpg](../iclr_results/11_Scaling Laws for Precision/images/2893b341e8fc191eab765b3c8326418a33a2853f75f582c3bd2273412b82f349.jpg)

![34b351d321ea4502e41a6f5087881490f647885546b43b114a7cf4be845a6b49.jpg](../iclr_results/11_Scaling Laws for Precision/images/34b351d321ea4502e41a6f5087881490f647885546b43b114a7cf4be845a6b49.jpg)

![42266830767b874fe702ff7fed408d8309319ca81f6f0e15a5ac08ccaf6db234.jpg](../iclr_results/11_Scaling Laws for Precision/images/42266830767b874fe702ff7fed408d8309319ca81f6f0e15a5ac08ccaf6db234.jpg)

![4413d87da0b253966ae2c90f00f662c9cb24463930497d786eaad868393a4fa6.jpg](../iclr_results/11_Scaling Laws for Precision/images/4413d87da0b253966ae2c90f00f662c9cb24463930497d786eaad868393a4fa6.jpg)

![49bdd66cbdbd33496d033c0352ab1fe5af06ce1de1b7641290dbc19cd39adb82.jpg](../iclr_results/11_Scaling Laws for Precision/images/49bdd66cbdbd33496d033c0352ab1fe5af06ce1de1b7641290dbc19cd39adb82.jpg)

![4d48080fcf19d447ccc11511121b5cca22504d8304d84636b3d2dce2f6c29d1b.jpg](../iclr_results/11_Scaling Laws for Precision/images/4d48080fcf19d447ccc11511121b5cca22504d8304d84636b3d2dce2f6c29d1b.jpg)

![4f9ec3406dc19220d3fda5a95e178d12efb1028567f26ef937cb5fd1e5a67a72.jpg](../iclr_results/11_Scaling Laws for Precision/images/4f9ec3406dc19220d3fda5a95e178d12efb1028567f26ef937cb5fd1e5a67a72.jpg)

![5853f345df0e5ea6c6194a33487b0fb1cb758d9f95e27b9c34e6bbe2bc26b225.jpg](../iclr_results/11_Scaling Laws for Precision/images/5853f345df0e5ea6c6194a33487b0fb1cb758d9f95e27b9c34e6bbe2bc26b225.jpg)

![6f35041a949654b2dc7b592ccf8f61fcf8adc02bfeeae969542647e920242def.jpg](../iclr_results/11_Scaling Laws for Precision/images/6f35041a949654b2dc7b592ccf8f61fcf8adc02bfeeae969542647e920242def.jpg)

![86183cfbcf257e0319dac512a7c0ff2385a669ae1308748d8028e5063dc330e2.jpg](../iclr_results/11_Scaling Laws for Precision/images/86183cfbcf257e0319dac512a7c0ff2385a669ae1308748d8028e5063dc330e2.jpg)

![90ad032804870b5201468e062b06a77ec45d325ec69b2fefde0259e9e80e4dbc.jpg](../iclr_results/11_Scaling Laws for Precision/images/90ad032804870b5201468e062b06a77ec45d325ec69b2fefde0259e9e80e4dbc.jpg)

![9b6d4fc05c8dcd3748c28a41ba6f1a5c123df875879285bd064b646726371fb6.jpg](../iclr_results/11_Scaling Laws for Precision/images/9b6d4fc05c8dcd3748c28a41ba6f1a5c123df875879285bd064b646726371fb6.jpg)

![a8aea2bb9dd8e3d5e96c2f9014318a8174a70a535fffa9844e3940a0ae88b0b0.jpg](../iclr_results/11_Scaling Laws for Precision/images/a8aea2bb9dd8e3d5e96c2f9014318a8174a70a535fffa9844e3940a0ae88b0b0.jpg)

![ad7b98d35bec0329885fae229cb7a8c90484353a93a910a179634f25ce70829c.jpg](../iclr_results/11_Scaling Laws for Precision/images/ad7b98d35bec0329885fae229cb7a8c90484353a93a910a179634f25ce70829c.jpg)

![af613b7d9964623598672e713687db4bdb8c113c9dbe29ad566708081fef555a.jpg](../iclr_results/11_Scaling Laws for Precision/images/af613b7d9964623598672e713687db4bdb8c113c9dbe29ad566708081fef555a.jpg)

![b687d73fd6cb58105567d5b988e8f43fb094e4ac7ced1f1e5431b0be28d3c4c1.jpg](../iclr_results/11_Scaling Laws for Precision/images/b687d73fd6cb58105567d5b988e8f43fb094e4ac7ced1f1e5431b0be28d3c4c1.jpg)

![d7bad5771f4970068aa92de7402ed99d94cbb4386a6f2dee5149bf9905f30312.jpg](../iclr_results/11_Scaling Laws for Precision/images/d7bad5771f4970068aa92de7402ed99d94cbb4386a6f2dee5149bf9905f30312.jpg)

![ea55dccb3e33d6cf41ef69e12400dabd6e4cc94317298c42188097cb2de35568.jpg](../iclr_results/11_Scaling Laws for Precision/images/ea55dccb3e33d6cf41ef69e12400dabd6e4cc94317298c42188097cb2de35568.jpg)

![edda7d39f649faf36ea3bdf779a70d627db8134070bca711a6bc2c1af418dfa2.jpg](../iclr_results/11_Scaling Laws for Precision/images/edda7d39f649faf36ea3bdf779a70d627db8134070bca711a6bc2c1af418dfa2.jpg)

![f61b65c94ad4d7d8acba0e92827bb128a2d00479320070314ab2e39e1a3cf983.jpg](../iclr_results/11_Scaling Laws for Precision/images/f61b65c94ad4d7d8acba0e92827bb128a2d00479320070314ab2e39e1a3cf983.jpg)

![fc247893c29e985a71ffa195b5cf410698780151075a172a395922bfe687cf4a.jpg](../iclr_results/11_Scaling Laws for Precision/images/fc247893c29e985a71ffa195b5cf410698780151075a172a395922bfe687cf4a.jpg)

### Tables

![03be0d9e9647fa97e641cbdd09cf15cdff4c0a613ba98de1b2f36db14cbeeac0.jpg](../iclr_results/11_Scaling Laws for Precision/tables/03be0d9e9647fa97e641cbdd09cf15cdff4c0a613ba98de1b2f36db14cbeeac0.jpg)

![6e5ed6f96d2dba28ce2792cdbf8d681e80fee1076871b58e4ad48460b61a675d.jpg](../iclr_results/11_Scaling Laws for Precision/tables/6e5ed6f96d2dba28ce2792cdbf8d681e80fee1076871b58e4ad48460b61a675d.jpg)

## Loopy: Taming Audio-Driven Portrait Avatar with Long-Term Motion Dependency


### Images

![3e9ff1befffffbf5c56cc17f01b3e55ce5a5e66f44c606f1ee7e6175808e83a5.jpg](../iclr_results/12_Loopy_ Taming Audio-Driven Portrait Avatar with Long-Term Motion Dependency/images/3e9ff1befffffbf5c56cc17f01b3e55ce5a5e66f44c606f1ee7e6175808e83a5.jpg)

![6168ebb78728502b430c3d6823e759f4c0f806c2b86bddef8988827357481985.jpg](../iclr_results/12_Loopy_ Taming Audio-Driven Portrait Avatar with Long-Term Motion Dependency/images/6168ebb78728502b430c3d6823e759f4c0f806c2b86bddef8988827357481985.jpg)

![6345d99d31daf4da98cefafd1b6d3e03c6716544d85f9212a5b498140c8ac348.jpg](../iclr_results/12_Loopy_ Taming Audio-Driven Portrait Avatar with Long-Term Motion Dependency/images/6345d99d31daf4da98cefafd1b6d3e03c6716544d85f9212a5b498140c8ac348.jpg)

![750efe4b455aa9b635ff5721077b40dd27917f8c5f2398316bd47a737c303293.jpg](../iclr_results/12_Loopy_ Taming Audio-Driven Portrait Avatar with Long-Term Motion Dependency/images/750efe4b455aa9b635ff5721077b40dd27917f8c5f2398316bd47a737c303293.jpg)

![75776a162e5949c6d2b5ec13ea4bdeeda6061332737bf490483da69aa627006a.jpg](../iclr_results/12_Loopy_ Taming Audio-Driven Portrait Avatar with Long-Term Motion Dependency/images/75776a162e5949c6d2b5ec13ea4bdeeda6061332737bf490483da69aa627006a.jpg)

![91a2470c76f2abf3d6f29a36327e0dc4374f781f14aa49481a02fc38b4632f34.jpg](../iclr_results/12_Loopy_ Taming Audio-Driven Portrait Avatar with Long-Term Motion Dependency/images/91a2470c76f2abf3d6f29a36327e0dc4374f781f14aa49481a02fc38b4632f34.jpg)

![c63554cf412582dcb92a3808868672191a9362f48b896a716b7b769406264213.jpg](../iclr_results/12_Loopy_ Taming Audio-Driven Portrait Avatar with Long-Term Motion Dependency/images/c63554cf412582dcb92a3808868672191a9362f48b896a716b7b769406264213.jpg)

### Tables

![08447c7b0ee693854de78e09ac7836658c634d8041cacdf05c0aab548273930e.jpg](../iclr_results/12_Loopy_ Taming Audio-Driven Portrait Avatar with Long-Term Motion Dependency/tables/08447c7b0ee693854de78e09ac7836658c634d8041cacdf05c0aab548273930e.jpg)

![13f2c9d697376c4888fc20ad8c2b1b1f4470cbfb13b2cb7742e0e7a579414ec5.jpg](../iclr_results/12_Loopy_ Taming Audio-Driven Portrait Avatar with Long-Term Motion Dependency/tables/13f2c9d697376c4888fc20ad8c2b1b1f4470cbfb13b2cb7742e0e7a579414ec5.jpg)

![383a933df8d13834e26facec60373affe04c8368e97893f0c6d5e9c11b0138b1.jpg](../iclr_results/12_Loopy_ Taming Audio-Driven Portrait Avatar with Long-Term Motion Dependency/tables/383a933df8d13834e26facec60373affe04c8368e97893f0c6d5e9c11b0138b1.jpg)

![895e14665440b8b0b7d76713d2a03c18752c7ba6e44c29f8581f4db94bc89e91.jpg](../iclr_results/12_Loopy_ Taming Audio-Driven Portrait Avatar with Long-Term Motion Dependency/tables/895e14665440b8b0b7d76713d2a03c18752c7ba6e44c29f8581f4db94bc89e91.jpg)

![9068d6864aa19bc2c4a39681723498af36267d15b04204495b60a24520e8d4c9.jpg](../iclr_results/12_Loopy_ Taming Audio-Driven Portrait Avatar with Long-Term Motion Dependency/tables/9068d6864aa19bc2c4a39681723498af36267d15b04204495b60a24520e8d4c9.jpg)

![c516a036dfc3ea3b05d4008483e33997dd2b5e1b802ffa07854e6bc73d48037d.jpg](../iclr_results/12_Loopy_ Taming Audio-Driven Portrait Avatar with Long-Term Motion Dependency/tables/c516a036dfc3ea3b05d4008483e33997dd2b5e1b802ffa07854e6bc73d48037d.jpg)

![e9d3dcc2eaea159e9772ff1e2917fc4aa491bee6a4b51b05ef4f96cc9db2b4eb.jpg](../iclr_results/12_Loopy_ Taming Audio-Driven Portrait Avatar with Long-Term Motion Dependency/tables/e9d3dcc2eaea159e9772ff1e2917fc4aa491bee6a4b51b05ef4f96cc9db2b4eb.jpg)

![febfa161e34c038154259bc4b9626760ad02a5fab8c856d661c501b5e7dd609e.jpg](../iclr_results/12_Loopy_ Taming Audio-Driven Portrait Avatar with Long-Term Motion Dependency/tables/febfa161e34c038154259bc4b9626760ad02a5fab8c856d661c501b5e7dd609e.jpg)

## Progressive distillation induces an implicit curriculum


### Images

![003cf0dd85d0849d3dae66eb8e8e7d9434a910005077cc7c3c99d2d003da117e.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/003cf0dd85d0849d3dae66eb8e8e7d9434a910005077cc7c3c99d2d003da117e.jpg)

![066127f8c0d33083c699a57fd989b1da1395a73e2fcad0a5b61a82e5f92a3f9a.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/066127f8c0d33083c699a57fd989b1da1395a73e2fcad0a5b61a82e5f92a3f9a.jpg)

![078b8768f2f37925e1dd627fbd8372f24c6ba2575fb5511166febcf88864e4e5.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/078b8768f2f37925e1dd627fbd8372f24c6ba2575fb5511166febcf88864e4e5.jpg)

![13d085d8268ce618ef642cec5e43d7292eabf0d8ee37f49e5ce8544e07d57b22.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/13d085d8268ce618ef642cec5e43d7292eabf0d8ee37f49e5ce8544e07d57b22.jpg)

![2e1015239d3ca38c859b6b94bdb30d35065286e37ef3b0707f3289c7a4eab08f.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/2e1015239d3ca38c859b6b94bdb30d35065286e37ef3b0707f3289c7a4eab08f.jpg)

![313c0dac9eaa6726650251db7a0176d06ad398e39eeed2ac157624e3207e7f8f.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/313c0dac9eaa6726650251db7a0176d06ad398e39eeed2ac157624e3207e7f8f.jpg)

![387bd66e1aec5546b04fa7d1f16c89eb380670c6dc21f5577aa10df0f433ba30.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/387bd66e1aec5546b04fa7d1f16c89eb380670c6dc21f5577aa10df0f433ba30.jpg)

![38fcbea8252bec1589f4d454661c371ad8da045ce74a1771ab7a4084a11fec1c.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/38fcbea8252bec1589f4d454661c371ad8da045ce74a1771ab7a4084a11fec1c.jpg)

![3963a51eec4c8ee468c49965f1258723eeb8b8246c6f6b6cd14139a19d21245a.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/3963a51eec4c8ee468c49965f1258723eeb8b8246c6f6b6cd14139a19d21245a.jpg)

![3f4717511cd13267feff19b28ccd73de7bc8a58d522e2dc9ba455c6890100e67.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/3f4717511cd13267feff19b28ccd73de7bc8a58d522e2dc9ba455c6890100e67.jpg)

![401d669677cb8b6d74af4a1945d5f74d3d2f47848bd19f88cb816b21eb2d8e0b.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/401d669677cb8b6d74af4a1945d5f74d3d2f47848bd19f88cb816b21eb2d8e0b.jpg)

![42c3afc86acab2443629f802df55b0391a2530405ec23af6264532f63a62dc63.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/42c3afc86acab2443629f802df55b0391a2530405ec23af6264532f63a62dc63.jpg)

![4cc2f740f14506788714fdf6d258f7b062c94baa517d1519726d1620bdb638c7.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/4cc2f740f14506788714fdf6d258f7b062c94baa517d1519726d1620bdb638c7.jpg)

![50f355c58cec412aed73f6ec5052445abe850955eab689c53a5385520c00f5e1.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/50f355c58cec412aed73f6ec5052445abe850955eab689c53a5385520c00f5e1.jpg)

![58139ad61bc7f0c6639f50625a4fe8870b8593697a1b388bec3ef0a52bbed1fa.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/58139ad61bc7f0c6639f50625a4fe8870b8593697a1b388bec3ef0a52bbed1fa.jpg)

![5fba86beb01886c3c59b04bd3754ab5c6a437485676ca038235a926726506f8a.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/5fba86beb01886c3c59b04bd3754ab5c6a437485676ca038235a926726506f8a.jpg)

![67ccb9b6725e972f8c6262f18a4a7a6afb6678c9a22c17585dfaa1091f150746.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/67ccb9b6725e972f8c6262f18a4a7a6afb6678c9a22c17585dfaa1091f150746.jpg)

![6ce56a91cf31a3ce8c12ba4ffcfb44c864a646a4ea1d1d64a1a996c9c2b89cae.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/6ce56a91cf31a3ce8c12ba4ffcfb44c864a646a4ea1d1d64a1a996c9c2b89cae.jpg)

![75a545e038b33a139306f8893652b71c14b2711bbcb86811f5d7c7fa134297af.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/75a545e038b33a139306f8893652b71c14b2711bbcb86811f5d7c7fa134297af.jpg)

![78c963e1e69caac4f7255f0431119cfa2cc884cbb59165936c52b9e3781574db.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/78c963e1e69caac4f7255f0431119cfa2cc884cbb59165936c52b9e3781574db.jpg)

![7b0bdb1cd1eeb1601ba81d51ee91e13bf79c2990fd0bb99b80d6228d9d8860c6.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/7b0bdb1cd1eeb1601ba81d51ee91e13bf79c2990fd0bb99b80d6228d9d8860c6.jpg)

![7f38780f822896f304044dc546e2802e8d8d6250f67f2511a63199df620257fe.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/7f38780f822896f304044dc546e2802e8d8d6250f67f2511a63199df620257fe.jpg)

![80956fdd9b65f2bd0c979aae3db5effdfdeb1213dc32bc5a936562bdb4a6aecc.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/80956fdd9b65f2bd0c979aae3db5effdfdeb1213dc32bc5a936562bdb4a6aecc.jpg)

![8dad9c7127c0cd153d47cc89139100ef011c8a87fddfb1dd5e5db5f2efbe647c.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/8dad9c7127c0cd153d47cc89139100ef011c8a87fddfb1dd5e5db5f2efbe647c.jpg)

![95e4213c11d90b99f63e16c620a1c3c84c6aaa168d6b3beffe967896d3d2d84b.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/95e4213c11d90b99f63e16c620a1c3c84c6aaa168d6b3beffe967896d3d2d84b.jpg)

![99e5646aece6277a284023fb4e6a09aa72b65f9ad5b4709955695eeed61ac363.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/99e5646aece6277a284023fb4e6a09aa72b65f9ad5b4709955695eeed61ac363.jpg)

![9bfe534025c5d908e5fe0c109acd67996ff10aa1855d0915bf583a016dc56b21.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/9bfe534025c5d908e5fe0c109acd67996ff10aa1855d0915bf583a016dc56b21.jpg)

![be5796a69bb772d13df1a70c0788031e64182af1566d9343bef22f89d578796c.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/be5796a69bb772d13df1a70c0788031e64182af1566d9343bef22f89d578796c.jpg)

![c29ae518e32b2e434b6125297cd27eeaf387f24461a920d68319ed1613bd6998.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/c29ae518e32b2e434b6125297cd27eeaf387f24461a920d68319ed1613bd6998.jpg)

![cb64be8d3b271610fca30d641585c23daf785526a34acc1f901326ac82e66a10.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/cb64be8d3b271610fca30d641585c23daf785526a34acc1f901326ac82e66a10.jpg)

![cea16d618d0c8ae2794c790487f0e9f6a2f627a83835b8578a0deeacd11d7518.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/cea16d618d0c8ae2794c790487f0e9f6a2f627a83835b8578a0deeacd11d7518.jpg)

![d77a8edd9cac4d91cf751166486b5824b9fa5c5ba0fb2f9d765354f66183b691.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/d77a8edd9cac4d91cf751166486b5824b9fa5c5ba0fb2f9d765354f66183b691.jpg)

![db4aaffef9f6abfb2373dd47ec07ff5bc97b4c336e7653560a4363a3c7d7234f.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/db4aaffef9f6abfb2373dd47ec07ff5bc97b4c336e7653560a4363a3c7d7234f.jpg)

![dc0f5407095b972be52a77eb1d675e03193b5d18eb34ad911e63a20b75f404f0.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/images/dc0f5407095b972be52a77eb1d675e03193b5d18eb34ad911e63a20b75f404f0.jpg)

### Tables

![12c4fbe4b5ccf8282ecbf0eb9e286cfef57524d94f53e79962dd224ce97f25ca.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/tables/12c4fbe4b5ccf8282ecbf0eb9e286cfef57524d94f53e79962dd224ce97f25ca.jpg)

![8479b058c9448539059e3af91694c41177d839d43633269e1a1c9fcceb67967c.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/tables/8479b058c9448539059e3af91694c41177d839d43633269e1a1c9fcceb67967c.jpg)

![ffbc21fb424effdb9bb3dad1d5cf6dc70d5394edd8e8761b04a8cabdef776eae.jpg](../iclr_results/13_Progressive distillation induces an implicit curriculum/tables/ffbc21fb424effdb9bb3dad1d5cf6dc70d5394edd8e8761b04a8cabdef776eae.jpg)

## Diffusion-Based Planning for Autonomous Driving with Flexible Guidance


### Images

![0814c619e6380e38f8df6f08b9eaee88b7971d3c64896d51223525eebcaa71b2.jpg](../iclr_results/14_Diffusion-Based Planning for Autonomous Driving with Flexible Guidance/images/0814c619e6380e38f8df6f08b9eaee88b7971d3c64896d51223525eebcaa71b2.jpg)

![0e6ce788458a4180819bd24275d26d80c56b873e637106c39be1a1ea62b4500f.jpg](../iclr_results/14_Diffusion-Based Planning for Autonomous Driving with Flexible Guidance/images/0e6ce788458a4180819bd24275d26d80c56b873e637106c39be1a1ea62b4500f.jpg)

![19ff31fe225e7e5c7fa3677ea605f5ea93f5250164b2609b100b4ac210253ea1.jpg](../iclr_results/14_Diffusion-Based Planning for Autonomous Driving with Flexible Guidance/images/19ff31fe225e7e5c7fa3677ea605f5ea93f5250164b2609b100b4ac210253ea1.jpg)

![5bfc04c50f504ba7971bdcb6e6f5d3b8779dcc44b343e718bdb6d4258338ecb9.jpg](../iclr_results/14_Diffusion-Based Planning for Autonomous Driving with Flexible Guidance/images/5bfc04c50f504ba7971bdcb6e6f5d3b8779dcc44b343e718bdb6d4258338ecb9.jpg)

![5c3276c48e299327279d01f245f4cf2c367b60e0d8ea307cc689962b2e7d0eee.jpg](../iclr_results/14_Diffusion-Based Planning for Autonomous Driving with Flexible Guidance/images/5c3276c48e299327279d01f245f4cf2c367b60e0d8ea307cc689962b2e7d0eee.jpg)

![5f89a177f38413c20bba392c43e2951664118962b71c592718e464ced001d41d.jpg](../iclr_results/14_Diffusion-Based Planning for Autonomous Driving with Flexible Guidance/images/5f89a177f38413c20bba392c43e2951664118962b71c592718e464ced001d41d.jpg)

![84a51733d43d6e30875cda431deb198827c1f5463cdf09b3a8069304fe775984.jpg](../iclr_results/14_Diffusion-Based Planning for Autonomous Driving with Flexible Guidance/images/84a51733d43d6e30875cda431deb198827c1f5463cdf09b3a8069304fe775984.jpg)

![b71542cf26e711b7e382023d8da4213442228e93cda5c6f5dc84f1d5282972f1.jpg](../iclr_results/14_Diffusion-Based Planning for Autonomous Driving with Flexible Guidance/images/b71542cf26e711b7e382023d8da4213442228e93cda5c6f5dc84f1d5282972f1.jpg)

![b96f97efefdf8008c4106f87d32e8c5df51ae1b3c40c5aa999097cc4d0216f89.jpg](../iclr_results/14_Diffusion-Based Planning for Autonomous Driving with Flexible Guidance/images/b96f97efefdf8008c4106f87d32e8c5df51ae1b3c40c5aa999097cc4d0216f89.jpg)

![c72dd96ed885472e10bac721ca70a2bcf79fc94d1ab37fd168c77ca0e85e67df.jpg](../iclr_results/14_Diffusion-Based Planning for Autonomous Driving with Flexible Guidance/images/c72dd96ed885472e10bac721ca70a2bcf79fc94d1ab37fd168c77ca0e85e67df.jpg)

![c8069119654c7e2d3635e06e32e22eb51818e6c1dce49dc06104bd3db35fc796.jpg](../iclr_results/14_Diffusion-Based Planning for Autonomous Driving with Flexible Guidance/images/c8069119654c7e2d3635e06e32e22eb51818e6c1dce49dc06104bd3db35fc796.jpg)

### Tables

![227f7209f79561a638baee54d2fbc4538abef2c539ae77afa0c4ed477ed045cc.jpg](../iclr_results/14_Diffusion-Based Planning for Autonomous Driving with Flexible Guidance/tables/227f7209f79561a638baee54d2fbc4538abef2c539ae77afa0c4ed477ed045cc.jpg)

![33b121017b7317375f2fe4608efd37736f3c7100aa765f84fcc901d43c5f6d3e.jpg](../iclr_results/14_Diffusion-Based Planning for Autonomous Driving with Flexible Guidance/tables/33b121017b7317375f2fe4608efd37736f3c7100aa765f84fcc901d43c5f6d3e.jpg)

![9de52f0125849043188cd0fb7c1a143262841ac5b41f71fb88fe88cd5f12d9a1.jpg](../iclr_results/14_Diffusion-Based Planning for Autonomous Driving with Flexible Guidance/tables/9de52f0125849043188cd0fb7c1a143262841ac5b41f71fb88fe88cd5f12d9a1.jpg)

![d8b4819197a6177348312300b5fc673780d99979907c30a3fb682b73852d81c2.jpg](../iclr_results/14_Diffusion-Based Planning for Autonomous Driving with Flexible Guidance/tables/d8b4819197a6177348312300b5fc673780d99979907c30a3fb682b73852d81c2.jpg)

![f62e58463dc479ec54f4d0e5afea39a26df073864f7b8918de1266bb087bbcdc.jpg](../iclr_results/14_Diffusion-Based Planning for Autonomous Driving with Flexible Guidance/tables/f62e58463dc479ec54f4d0e5afea39a26df073864f7b8918de1266bb087bbcdc.jpg)

## Open-World Reinforcement Learning over Long Short-Term Imagination


### Images

![07a475fec8f2e2dc3c8d69c15e023a6c6d3573228d4412ae39bb4350940e87c6.jpg](../iclr_results/15_Open-World Reinforcement Learning over Long Short-Term Imagination/images/07a475fec8f2e2dc3c8d69c15e023a6c6d3573228d4412ae39bb4350940e87c6.jpg)

![3639af139b42f0bee2e6cf59e85419b9b63f64e7d9fa1d7991876465968fb4de.jpg](../iclr_results/15_Open-World Reinforcement Learning over Long Short-Term Imagination/images/3639af139b42f0bee2e6cf59e85419b9b63f64e7d9fa1d7991876465968fb4de.jpg)

![54d01962bb729680c24a5d2e1060ac8bda1ca194a06f6b1ec8e8e1b66bcbbb05.jpg](../iclr_results/15_Open-World Reinforcement Learning over Long Short-Term Imagination/images/54d01962bb729680c24a5d2e1060ac8bda1ca194a06f6b1ec8e8e1b66bcbbb05.jpg)

![57a49d1685ff3e99ffd35518500241afb187223688c39084350d7e7cda05facd.jpg](../iclr_results/15_Open-World Reinforcement Learning over Long Short-Term Imagination/images/57a49d1685ff3e99ffd35518500241afb187223688c39084350d7e7cda05facd.jpg)

![7ed40c8c0182d258273fea5440b21ed5ba25f9172cb88a5040c13c8eef5abedf.jpg](../iclr_results/15_Open-World Reinforcement Learning over Long Short-Term Imagination/images/7ed40c8c0182d258273fea5440b21ed5ba25f9172cb88a5040c13c8eef5abedf.jpg)

![884ec1c2adbdc3fb9ee0f16feccf8aebc2ef272a5405f0f6c4f7265876d5e1f1.jpg](../iclr_results/15_Open-World Reinforcement Learning over Long Short-Term Imagination/images/884ec1c2adbdc3fb9ee0f16feccf8aebc2ef272a5405f0f6c4f7265876d5e1f1.jpg)

![889fa125778d7fab0f0741ca3880e9b0e670431a83d8ade5802c2bc26de36eb6.jpg](../iclr_results/15_Open-World Reinforcement Learning over Long Short-Term Imagination/images/889fa125778d7fab0f0741ca3880e9b0e670431a83d8ade5802c2bc26de36eb6.jpg)

![a5c35d766dd5e55c85eb95ed4bc5b6084f6eb7427c656d6ca61c7772be73e4c6.jpg](../iclr_results/15_Open-World Reinforcement Learning over Long Short-Term Imagination/images/a5c35d766dd5e55c85eb95ed4bc5b6084f6eb7427c656d6ca61c7772be73e4c6.jpg)

![c214ed7408d6207abf77ea0a1c3a82725e0c42cc5e39537d6ed19667cb58149a.jpg](../iclr_results/15_Open-World Reinforcement Learning over Long Short-Term Imagination/images/c214ed7408d6207abf77ea0a1c3a82725e0c42cc5e39537d6ed19667cb58149a.jpg)

![c5fbdf3e195a65ed46cf03fdc4a5e22bb82174f70645e16ab46f1a7b7c11b141.jpg](../iclr_results/15_Open-World Reinforcement Learning over Long Short-Term Imagination/images/c5fbdf3e195a65ed46cf03fdc4a5e22bb82174f70645e16ab46f1a7b7c11b141.jpg)

![d2107e9ece73b02bf2a6523d95cc3fb926a94d8f2b86906f3503ad3c6add547f.jpg](../iclr_results/15_Open-World Reinforcement Learning over Long Short-Term Imagination/images/d2107e9ece73b02bf2a6523d95cc3fb926a94d8f2b86906f3503ad3c6add547f.jpg)

![d911512cac9147f4fcc20a4383902b3df37f7ee18f0a165c0f1b587ada9777b5.jpg](../iclr_results/15_Open-World Reinforcement Learning over Long Short-Term Imagination/images/d911512cac9147f4fcc20a4383902b3df37f7ee18f0a165c0f1b587ada9777b5.jpg)

![df9178dc2ebcbf2cc73203e952fabcbdb0b09fbe2c9bc55fbea15c5d72e81600.jpg](../iclr_results/15_Open-World Reinforcement Learning over Long Short-Term Imagination/images/df9178dc2ebcbf2cc73203e952fabcbdb0b09fbe2c9bc55fbea15c5d72e81600.jpg)

![ef95e4b3583cb836a5aa157f48e37de1fabfc17942a983643de605ee967e1207.jpg](../iclr_results/15_Open-World Reinforcement Learning over Long Short-Term Imagination/images/ef95e4b3583cb836a5aa157f48e37de1fabfc17942a983643de605ee967e1207.jpg)

![f5518dc5ab019a799160139125f928c3779c4f947aaf308b9ac43be4d73bba5b.jpg](../iclr_results/15_Open-World Reinforcement Learning over Long Short-Term Imagination/images/f5518dc5ab019a799160139125f928c3779c4f947aaf308b9ac43be4d73bba5b.jpg)

![f855f7552e8c8d02eb2bbb2cb680820432a47c965376e6233e12aa76f78921d9.jpg](../iclr_results/15_Open-World Reinforcement Learning over Long Short-Term Imagination/images/f855f7552e8c8d02eb2bbb2cb680820432a47c965376e6233e12aa76f78921d9.jpg)

### Tables

![24d424c6b9bc6ab5ebdcc731e383a95281af2b5e8f11fdf48afc33d8fb2ceab7.jpg](../iclr_results/15_Open-World Reinforcement Learning over Long Short-Term Imagination/tables/24d424c6b9bc6ab5ebdcc731e383a95281af2b5e8f11fdf48afc33d8fb2ceab7.jpg)

![47e50216d9196bd8a9ef29015871e41938f3def2a7e5acaf50aeffa25c790bae.jpg](../iclr_results/15_Open-World Reinforcement Learning over Long Short-Term Imagination/tables/47e50216d9196bd8a9ef29015871e41938f3def2a7e5acaf50aeffa25c790bae.jpg)

![4f4f0d2bd0c849ed2a2ce19723808a5fdb49d635df0b0e05c9915245d3d9b35f.jpg](../iclr_results/15_Open-World Reinforcement Learning over Long Short-Term Imagination/tables/4f4f0d2bd0c849ed2a2ce19723808a5fdb49d635df0b0e05c9915245d3d9b35f.jpg)

![e440acbcef8db2137d73433d596068acd445506fb87d774a39cc5cea6269ddf2.jpg](../iclr_results/15_Open-World Reinforcement Learning over Long Short-Term Imagination/tables/e440acbcef8db2137d73433d596068acd445506fb87d774a39cc5cea6269ddf2.jpg)

## Faster Cascades via Speculative Decoding


### Images

![1c4abb8e4045af7cc6f411d7af10da4a16badb8f2ac844c290614f80369018b2.jpg](../iclr_results/16_Faster Cascades via Speculative Decoding/images/1c4abb8e4045af7cc6f411d7af10da4a16badb8f2ac844c290614f80369018b2.jpg)

![1ecd5c0021785330aeb587ad5f4f2e30ff92cb79ce42c21c7e4e30450c9bdcee.jpg](../iclr_results/16_Faster Cascades via Speculative Decoding/images/1ecd5c0021785330aeb587ad5f4f2e30ff92cb79ce42c21c7e4e30450c9bdcee.jpg)

![2282dc6fcc61e4f9278311ad8d80f92a70de7dafe9b995a4618d51b8eeea813e.jpg](../iclr_results/16_Faster Cascades via Speculative Decoding/images/2282dc6fcc61e4f9278311ad8d80f92a70de7dafe9b995a4618d51b8eeea813e.jpg)

![3933910a68b5a77ef2ad35c19aa87e7a5a7b8a02e879e6a3a73b3c5f8e8d9aaa.jpg](../iclr_results/16_Faster Cascades via Speculative Decoding/images/3933910a68b5a77ef2ad35c19aa87e7a5a7b8a02e879e6a3a73b3c5f8e8d9aaa.jpg)

![4d2c2acd4158732ef70b37b532fd97690dbdefd28718924548067599f1584f67.jpg](../iclr_results/16_Faster Cascades via Speculative Decoding/images/4d2c2acd4158732ef70b37b532fd97690dbdefd28718924548067599f1584f67.jpg)

![5ad1082e5387588315cef93dd12e8b507916c658d2a43e7343f85fac6a09ce50.jpg](../iclr_results/16_Faster Cascades via Speculative Decoding/images/5ad1082e5387588315cef93dd12e8b507916c658d2a43e7343f85fac6a09ce50.jpg)

![778c9047f2895beb825b14733f92fa5ee8cff2cc22a9ffad30af949711f7b23f.jpg](../iclr_results/16_Faster Cascades via Speculative Decoding/images/778c9047f2895beb825b14733f92fa5ee8cff2cc22a9ffad30af949711f7b23f.jpg)

![8b47388d78ca78704ceab9b5b71ab5708ebf04fef4a450f928b7f3f6cdc71a67.jpg](../iclr_results/16_Faster Cascades via Speculative Decoding/images/8b47388d78ca78704ceab9b5b71ab5708ebf04fef4a450f928b7f3f6cdc71a67.jpg)

![968e0e6f6674668cd4934f365119c44c744184051e342db21c8caab8c19fb300.jpg](../iclr_results/16_Faster Cascades via Speculative Decoding/images/968e0e6f6674668cd4934f365119c44c744184051e342db21c8caab8c19fb300.jpg)

![baf3ea822c82a0e68a95ab429126fc1bbce1793ac0e5f6bfa5bd332dc88fa790.jpg](../iclr_results/16_Faster Cascades via Speculative Decoding/images/baf3ea822c82a0e68a95ab429126fc1bbce1793ac0e5f6bfa5bd332dc88fa790.jpg)

![d8eefdc9ad7749f1eec7a71701237d7738736c98d64dc6df7626c95c89eb38b1.jpg](../iclr_results/16_Faster Cascades via Speculative Decoding/images/d8eefdc9ad7749f1eec7a71701237d7738736c98d64dc6df7626c95c89eb38b1.jpg)

![dbf6450454734235869effc4d02d9a912d090a5e0dc09b0bc05b1b15e6af3dd6.jpg](../iclr_results/16_Faster Cascades via Speculative Decoding/images/dbf6450454734235869effc4d02d9a912d090a5e0dc09b0bc05b1b15e6af3dd6.jpg)

### Tables

![158cd45750dfa5c2e09becc15b5add04e9893d7a395cdb3943ade48e9d00fb20.jpg](../iclr_results/16_Faster Cascades via Speculative Decoding/tables/158cd45750dfa5c2e09becc15b5add04e9893d7a395cdb3943ade48e9d00fb20.jpg)

![608da4c0200ac0b98658f28a0f816e70abe863d7fd0aaa8f4a545ab659df0c47.jpg](../iclr_results/16_Faster Cascades via Speculative Decoding/tables/608da4c0200ac0b98658f28a0f816e70abe863d7fd0aaa8f4a545ab659df0c47.jpg)

![8cdf395c59a9a029a4ad30421edc1d31f9a47c4d9992c97e53d8e692abb75943.jpg](../iclr_results/16_Faster Cascades via Speculative Decoding/tables/8cdf395c59a9a029a4ad30421edc1d31f9a47c4d9992c97e53d8e692abb75943.jpg)

![9c9b0048fdf3bd9061c67c0814d9715ec9759bec058740e9ea24830a72c9f0e3.jpg](../iclr_results/16_Faster Cascades via Speculative Decoding/tables/9c9b0048fdf3bd9061c67c0814d9715ec9759bec058740e9ea24830a72c9f0e3.jpg)

![f7a8b40635e7a22cdac637aa6bf13b33ac99fbed8cb2768f543502c7512faf6e.jpg](../iclr_results/16_Faster Cascades via Speculative Decoding/tables/f7a8b40635e7a22cdac637aa6bf13b33ac99fbed8cb2768f543502c7512faf6e.jpg)

## DEPT: Decoupled Embeddings for Pre-training Language Models


### Images

![160b139924f6b537857c90520e3e0475b2acd041b29e24e47e8a165f18fdb099.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/images/160b139924f6b537857c90520e3e0475b2acd041b29e24e47e8a165f18fdb099.jpg)

![3eeeb6eee9df26259423bcb52beef4325da8d699ea218faf95e0fb949962b88f.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/images/3eeeb6eee9df26259423bcb52beef4325da8d699ea218faf95e0fb949962b88f.jpg)

![45d4c1ab5811a2d12a25e5acf69018645f77d1ecd664562eea0f17af271672a9.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/images/45d4c1ab5811a2d12a25e5acf69018645f77d1ecd664562eea0f17af271672a9.jpg)

![7b9ddbb60c6278d5bbf00fff41c1ba1f2e664b27e7b390a5d39b7fe9245bb414.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/images/7b9ddbb60c6278d5bbf00fff41c1ba1f2e664b27e7b390a5d39b7fe9245bb414.jpg)

![99ef73a4454a9ca70bff4c5e2c0cc9d3934decca4f86235fd6b95effd731a5b6.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/images/99ef73a4454a9ca70bff4c5e2c0cc9d3934decca4f86235fd6b95effd731a5b6.jpg)

![a4938c57e388f8c93aae09ec5a3f3ff2b483b6ce1f437c7e8dd53a66d5230bc5.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/images/a4938c57e388f8c93aae09ec5a3f3ff2b483b6ce1f437c7e8dd53a66d5230bc5.jpg)

![f41bd33b68a40bc28eaa304a4966d87ae52397bef3a71dc5f31213f65f7e6d57.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/images/f41bd33b68a40bc28eaa304a4966d87ae52397bef3a71dc5f31213f65f7e6d57.jpg)

### Tables

![062c2daabd403658f4d53bd796d4501ff5da75c24b339f383be154e71d532768.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/tables/062c2daabd403658f4d53bd796d4501ff5da75c24b339f383be154e71d532768.jpg)

![0dc9ae372148c4cb72e644b3cd831a597a82940c457c05e4d26d08d56e557948.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/tables/0dc9ae372148c4cb72e644b3cd831a597a82940c457c05e4d26d08d56e557948.jpg)

![0e73b0da111f0a6961f666509a7db63688ea155ac3e383aa131c62484f76c86f.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/tables/0e73b0da111f0a6961f666509a7db63688ea155ac3e383aa131c62484f76c86f.jpg)

![22c74d2a8aeed5580bb5d65be3c30cbc456262b50eb147dd0aa3b71e73abcb2a.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/tables/22c74d2a8aeed5580bb5d65be3c30cbc456262b50eb147dd0aa3b71e73abcb2a.jpg)

![3097ca452a85366f790206fa5c47f239cb7cb2f46d5eefb2f9efdfd7649fc6d2.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/tables/3097ca452a85366f790206fa5c47f239cb7cb2f46d5eefb2f9efdfd7649fc6d2.jpg)

![3f808ad792e8b39138c37c1ac5a48857bbdc286020d106989be3c361075b6bf9.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/tables/3f808ad792e8b39138c37c1ac5a48857bbdc286020d106989be3c361075b6bf9.jpg)

![40470e65177b77bcb465927d452eda0df52898695f28f8c3e301487fb4e7de2a.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/tables/40470e65177b77bcb465927d452eda0df52898695f28f8c3e301487fb4e7de2a.jpg)

![73243b2356c347ce342e6a91e008cbaf429ad899c945035413dee01cbb4f4221.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/tables/73243b2356c347ce342e6a91e008cbaf429ad899c945035413dee01cbb4f4221.jpg)

![7d73a38c1727c8fc0ba605db2e44df37eb947d2335bdf2eb1e1f36b234187ede.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/tables/7d73a38c1727c8fc0ba605db2e44df37eb947d2335bdf2eb1e1f36b234187ede.jpg)

![8dd12d6a8f1b04834ea6ed3062d21fe7e05b30722b9862cf9a045f8d4e1c8046.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/tables/8dd12d6a8f1b04834ea6ed3062d21fe7e05b30722b9862cf9a045f8d4e1c8046.jpg)

![91fb15c28ffab30f78d06148889d158fe83b72a9f2a94f38734cc120bfde0fc1.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/tables/91fb15c28ffab30f78d06148889d158fe83b72a9f2a94f38734cc120bfde0fc1.jpg)

![9f087f07fa384c38bb4d55d75a28912d8989dd048580d2daf79ed140c159fa0c.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/tables/9f087f07fa384c38bb4d55d75a28912d8989dd048580d2daf79ed140c159fa0c.jpg)

![a8b030705fe92d37ca5759b29e973e312acc01998f31e2fe00cdeea3dbed4722.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/tables/a8b030705fe92d37ca5759b29e973e312acc01998f31e2fe00cdeea3dbed4722.jpg)

![b8ff86b572e547f68873de0f6376dcae344a6add4ef639435c7abbc0a4ef89fd.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/tables/b8ff86b572e547f68873de0f6376dcae344a6add4ef639435c7abbc0a4ef89fd.jpg)

![bb6007f2a9bd17cb511315d55e1787a638502a8182ef1ab144549c393de999dd.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/tables/bb6007f2a9bd17cb511315d55e1787a638502a8182ef1ab144549c393de999dd.jpg)

![c6eddaa5bcd665c3defb22c8d3a496db4161a5b0d52dfe26edf6364d8ab5b26c.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/tables/c6eddaa5bcd665c3defb22c8d3a496db4161a5b0d52dfe26edf6364d8ab5b26c.jpg)

![d257d5bec34e9f56a3d1c007019d99305661e6d894efe379a3ef99f56186d236.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/tables/d257d5bec34e9f56a3d1c007019d99305661e6d894efe379a3ef99f56186d236.jpg)

![e8c84f96781cd45d0adadcbd1a4e070dc2949df81bd41c4952551b95a1fe0ed5.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/tables/e8c84f96781cd45d0adadcbd1a4e070dc2949df81bd41c4952551b95a1fe0ed5.jpg)

![ec930c1f6bea4e2ca529cc1e2ccf9756ece27558d8e01f88403294117b117193.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/tables/ec930c1f6bea4e2ca529cc1e2ccf9756ece27558d8e01f88403294117b117193.jpg)

![f62fcb79e11641d344d44566d9d3a1f16bfca6b6ae3e147d1460e5c1e4311dcb.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/tables/f62fcb79e11641d344d44566d9d3a1f16bfca6b6ae3e147d1460e5c1e4311dcb.jpg)

![fa83fd8557d60723f09b7e8d64fd026048168010b7d761308f9dd871fef92363.jpg](../iclr_results/17_DEPT_ Decoupled Embeddings for Pre-training Language Models/tables/fa83fd8557d60723f09b7e8d64fd026048168010b7d761308f9dd871fef92363.jpg)

## When is Task Vector Provably Effective for Model Editing? A Generalization Analysis of Nonlinear Transformers


### Images

![71b64128ad2f43bfb71a591fad5343b8fb2885ff43a2a75c77eec0335d4f7247.jpg](../iclr_results/19_When is Task Vector Provably Effective for Model Editing_ A Generalization Analysis of Nonlinear Tra/images/71b64128ad2f43bfb71a591fad5343b8fb2885ff43a2a75c77eec0335d4f7247.jpg)

![a597e8e3b802abe3bf38f6e21172797e227dd264c631ce115b676492f9db3336.jpg](../iclr_results/19_When is Task Vector Provably Effective for Model Editing_ A Generalization Analysis of Nonlinear Tra/images/a597e8e3b802abe3bf38f6e21172797e227dd264c631ce115b676492f9db3336.jpg)

### Tables

![41a79f725c21e6e20b71983f1ea7fb2b78099e3d86a84abda0a31a2f7dca23a2.jpg](../iclr_results/19_When is Task Vector Provably Effective for Model Editing_ A Generalization Analysis of Nonlinear Tra/tables/41a79f725c21e6e20b71983f1ea7fb2b78099e3d86a84abda0a31a2f7dca23a2.jpg)

![73348f40ab8b08c8fa067dc8fe8d896969d6bfac8d9d6bfb13b48d46f703c27b.jpg](../iclr_results/19_When is Task Vector Provably Effective for Model Editing_ A Generalization Analysis of Nonlinear Tra/tables/73348f40ab8b08c8fa067dc8fe8d896969d6bfac8d9d6bfb13b48d46f703c27b.jpg)

![7fc298096c36cc98c6d5362de7d927b0802713fe4728fb6d20573f3de6f3e8d8.jpg](../iclr_results/19_When is Task Vector Provably Effective for Model Editing_ A Generalization Analysis of Nonlinear Tra/tables/7fc298096c36cc98c6d5362de7d927b0802713fe4728fb6d20573f3de6f3e8d8.jpg)

![82bfba064de265adbe8f61c680227dc4964f2ed25577c3db5b97e11ac56d5483.jpg](../iclr_results/19_When is Task Vector Provably Effective for Model Editing_ A Generalization Analysis of Nonlinear Tra/tables/82bfba064de265adbe8f61c680227dc4964f2ed25577c3db5b97e11ac56d5483.jpg)

![b6d1424ea2d4c21a26e4a1208052690f3b2b7620184e74312070b599913d9cff.jpg](../iclr_results/19_When is Task Vector Provably Effective for Model Editing_ A Generalization Analysis of Nonlinear Tra/tables/b6d1424ea2d4c21a26e4a1208052690f3b2b7620184e74312070b599913d9cff.jpg)

![f80c4d8ace73dd38f165d3d0c7515aa932e05840726bf157438d34f4d4fb95f0.jpg](../iclr_results/19_When is Task Vector Provably Effective for Model Editing_ A Generalization Analysis of Nonlinear Tra/tables/f80c4d8ace73dd38f165d3d0c7515aa932e05840726bf157438d34f4d4fb95f0.jpg)

## Learning to Search from Demonstration Sequences


### Images

![08a978a818677efbc274f128d5c84a6f1626a7ac0c51326f5f080170046e5a73.jpg](../iclr_results/20_Learning to Search from Demonstration Sequences/images/08a978a818677efbc274f128d5c84a6f1626a7ac0c51326f5f080170046e5a73.jpg)

![0db08df04b58ee69c90f69d1be5c86bc73fe5b18f5f0fad4122287628905303f.jpg](../iclr_results/20_Learning to Search from Demonstration Sequences/images/0db08df04b58ee69c90f69d1be5c86bc73fe5b18f5f0fad4122287628905303f.jpg)

![192117eb6bbd0ca8dfd1a97bc9b916066ae14c3df7370477214c6dc14e98955c.jpg](../iclr_results/20_Learning to Search from Demonstration Sequences/images/192117eb6bbd0ca8dfd1a97bc9b916066ae14c3df7370477214c6dc14e98955c.jpg)

![5a2639947848304782a60d630e472f272ed0033fd063480c778eeede150d0829.jpg](../iclr_results/20_Learning to Search from Demonstration Sequences/images/5a2639947848304782a60d630e472f272ed0033fd063480c778eeede150d0829.jpg)

![7dfefccadd2c78043c8e2c5c1fb8fa4f36d10d4b416f5781c17d69c9f2a52694.jpg](../iclr_results/20_Learning to Search from Demonstration Sequences/images/7dfefccadd2c78043c8e2c5c1fb8fa4f36d10d4b416f5781c17d69c9f2a52694.jpg)

![938088fc96daa027a39441b919468d75136175f1a04ee65e5b20f0875821014b.jpg](../iclr_results/20_Learning to Search from Demonstration Sequences/images/938088fc96daa027a39441b919468d75136175f1a04ee65e5b20f0875821014b.jpg)

![dba27ae36d321714268ed5fcbaaa44cfa5c950c9ae66a9274b80b79887eec141.jpg](../iclr_results/20_Learning to Search from Demonstration Sequences/images/dba27ae36d321714268ed5fcbaaa44cfa5c950c9ae66a9274b80b79887eec141.jpg)

### Tables

![2f07056fcf346727e2bd7e92ca535a488bb4b41dc196cc0c4416e5fdc373eb05.jpg](../iclr_results/20_Learning to Search from Demonstration Sequences/tables/2f07056fcf346727e2bd7e92ca535a488bb4b41dc196cc0c4416e5fdc373eb05.jpg)

![5ed4d5454732cb62cb2adf5093bd28c267de4508e3a5d6b0d7e34afb13ddd7d9.jpg](../iclr_results/20_Learning to Search from Demonstration Sequences/tables/5ed4d5454732cb62cb2adf5093bd28c267de4508e3a5d6b0d7e34afb13ddd7d9.jpg)

![79a8aaa36a3b2188a617540d36cd9a9dc3d20fa13f1c0d0ac82e056cb4428b6a.jpg](../iclr_results/20_Learning to Search from Demonstration Sequences/tables/79a8aaa36a3b2188a617540d36cd9a9dc3d20fa13f1c0d0ac82e056cb4428b6a.jpg)

![a081a7f0709bc517c684e0f9274aa207d0c7d2ddd6d57508468b1e8a3fd0be97.jpg](../iclr_results/20_Learning to Search from Demonstration Sequences/tables/a081a7f0709bc517c684e0f9274aa207d0c7d2ddd6d57508468b1e8a3fd0be97.jpg)

![a60d91ec288773e535c7f88bc1868b78b3b99917ebc61161fe1f864b9decefeb.jpg](../iclr_results/20_Learning to Search from Demonstration Sequences/tables/a60d91ec288773e535c7f88bc1868b78b3b99917ebc61161fe1f864b9decefeb.jpg)

![b236a33c220f3144d401fb345375a3e822a8c825dc31fe88f4ddb8cc521f8c23.jpg](../iclr_results/20_Learning to Search from Demonstration Sequences/tables/b236a33c220f3144d401fb345375a3e822a8c825dc31fe88f4ddb8cc521f8c23.jpg)

![ba018ada83d8243eb59925d79f2ecc5581d27b424b7f01eb9a9dd091a3ebdbfa.jpg](../iclr_results/20_Learning to Search from Demonstration Sequences/tables/ba018ada83d8243eb59925d79f2ecc5581d27b424b7f01eb9a9dd091a3ebdbfa.jpg)

![f4d92a32a297e58d84761db5bea8671a65887064463084f6cf937eea92f12f79.jpg](../iclr_results/20_Learning to Search from Demonstration Sequences/tables/f4d92a32a297e58d84761db5bea8671a65887064463084f6cf937eea92f12f79.jpg)

## Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks


### Images

![014232370602c4adf76ccad16c0294c84a31326f8f1203009a72593addc52c00.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/images/014232370602c4adf76ccad16c0294c84a31326f8f1203009a72593addc52c00.jpg)

![0eb529ae871a24a3b40c6411a5db9e24aca24bb04da1d90eda947b62f405715b.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/images/0eb529ae871a24a3b40c6411a5db9e24aca24bb04da1d90eda947b62f405715b.jpg)

![194c784fab0f83ba153bb5f40fb47c33c17d0356802fe96148214c766a04cc58.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/images/194c784fab0f83ba153bb5f40fb47c33c17d0356802fe96148214c766a04cc58.jpg)

![2a052cdad1a07521c1930caa1c2b7c5a06c11c759d53292a9437a1db4480cfa7.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/images/2a052cdad1a07521c1930caa1c2b7c5a06c11c759d53292a9437a1db4480cfa7.jpg)

![2b78146bf4dabba60f6efb28dd130a540cd0a956fd1764452df25c5774620297.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/images/2b78146bf4dabba60f6efb28dd130a540cd0a956fd1764452df25c5774620297.jpg)

![32fcf4e0b1072495177f469a4feebef7df0a93c60c0c57a792d98d374adf8c11.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/images/32fcf4e0b1072495177f469a4feebef7df0a93c60c0c57a792d98d374adf8c11.jpg)

![3af7a069222b209b7c025af093c9e16da06d4594ae7b2bb3ab59e1cde94fcf41.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/images/3af7a069222b209b7c025af093c9e16da06d4594ae7b2bb3ab59e1cde94fcf41.jpg)

![49d5c03d5c50e954bc242bd9af1df079bcbcbfef97155c3df349f88131f9a8fc.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/images/49d5c03d5c50e954bc242bd9af1df079bcbcbfef97155c3df349f88131f9a8fc.jpg)

![56f5d8ae451d4d0e92556789a5b20df8a78f19b3b4bfd2db648ac7ee8006c7ee.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/images/56f5d8ae451d4d0e92556789a5b20df8a78f19b3b4bfd2db648ac7ee8006c7ee.jpg)

![57a29579589aaa4431bccdf4b691ed62fd6414a0dad46a392fbd06d50b589b24.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/images/57a29579589aaa4431bccdf4b691ed62fd6414a0dad46a392fbd06d50b589b24.jpg)

![5fb7acf088bac08f48779b75dc26a372a3801ea4c08eedee32e2279188364a1b.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/images/5fb7acf088bac08f48779b75dc26a372a3801ea4c08eedee32e2279188364a1b.jpg)

![62f07c2b56e1551d0811ac2237e6ed68bae523a0c3070f698b3b313f78f8afc6.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/images/62f07c2b56e1551d0811ac2237e6ed68bae523a0c3070f698b3b313f78f8afc6.jpg)

![962e8508e1ef26596ca5f79fd919e6e48ec5c523fe00da56abf56951d44a5c4d.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/images/962e8508e1ef26596ca5f79fd919e6e48ec5c523fe00da56abf56951d44a5c4d.jpg)

![990c8866c7c8ba7dc11249d5f845e8f03c4e8a5f2af7821f49352aa8c05a03ef.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/images/990c8866c7c8ba7dc11249d5f845e8f03c4e8a5f2af7821f49352aa8c05a03ef.jpg)

![ae2c64b57d56c96137ac4c7dbf945f6fc4c1959f70398a07fa0621bb501eec6b.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/images/ae2c64b57d56c96137ac4c7dbf945f6fc4c1959f70398a07fa0621bb501eec6b.jpg)

![d0ef76ffe6c48881adb89f6facf270ba6d345f1361b316590550e4900b29101d.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/images/d0ef76ffe6c48881adb89f6facf270ba6d345f1361b316590550e4900b29101d.jpg)

![d3d889d835367f884512e5ba501ec810fc5e8f985b5b8fc3ad636b4e399e211f.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/images/d3d889d835367f884512e5ba501ec810fc5e8f985b5b8fc3ad636b4e399e211f.jpg)

![d43eb96d50e90531f631ff882a1ae6652f7f5c8fb45a67001da0c85c64d601d8.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/images/d43eb96d50e90531f631ff882a1ae6652f7f5c8fb45a67001da0c85c64d601d8.jpg)

![f8ae6cd9bcf38a79f10c5ed8adfe0e962727631dfd1d88b1b8940aaeb021cd5b.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/images/f8ae6cd9bcf38a79f10c5ed8adfe0e962727631dfd1d88b1b8940aaeb021cd5b.jpg)

![fdc81e56f269922738d8723b2424cf2c22bd846113cdeb6f554202c012cce280.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/images/fdc81e56f269922738d8723b2424cf2c22bd846113cdeb6f554202c012cce280.jpg)

### Tables

![10abe0a46218b6a344526219c5abd3207fd5fb7d0c21f7248261a8762df91bd5.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/tables/10abe0a46218b6a344526219c5abd3207fd5fb7d0c21f7248261a8762df91bd5.jpg)

![203830f4365ee16d91ebb63a68ac840a56a0259013ed3a3e6d9cda09bcb25fde.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/tables/203830f4365ee16d91ebb63a68ac840a56a0259013ed3a3e6d9cda09bcb25fde.jpg)

![33db288bb23a9461602a78f3a858e1ba51b076dfb9c7cfaad97ee4e77d460c66.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/tables/33db288bb23a9461602a78f3a858e1ba51b076dfb9c7cfaad97ee4e77d460c66.jpg)

![3698e44c9e47d5cf9a484e40f1d28fdce7dea00ec926e36c5d8e413e850214c0.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/tables/3698e44c9e47d5cf9a484e40f1d28fdce7dea00ec926e36c5d8e413e850214c0.jpg)

![5b9c70e2fc0f8cfc7d2bbf64986163daaaa5cab2135c50b880802ce8d146d41f.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/tables/5b9c70e2fc0f8cfc7d2bbf64986163daaaa5cab2135c50b880802ce8d146d41f.jpg)

![6ed2e912729303e9b7891919d234b7f1b174b7a5d463c71f568aab5e5903e2c8.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/tables/6ed2e912729303e9b7891919d234b7f1b174b7a5d463c71f568aab5e5903e2c8.jpg)

![73c604e85a4884bef792948cfd71a47e57a02615da7a1f719a221d92164c6549.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/tables/73c604e85a4884bef792948cfd71a47e57a02615da7a1f719a221d92164c6549.jpg)

![892069197f0dfc029bfaa3e12b9dfce74d503d738d91e73ae3b38ba96b9387a1.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/tables/892069197f0dfc029bfaa3e12b9dfce74d503d738d91e73ae3b38ba96b9387a1.jpg)

![9e004154b9c813eee8c20d6a04e2a8898b8a84aec3c2cc9531c4eabd9aec2f99.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/tables/9e004154b9c813eee8c20d6a04e2a8898b8a84aec3c2cc9531c4eabd9aec2f99.jpg)

![a8fe8c3dfc971a6c1d3bd1c642addd1dbb8d4dc2a38931f03b29da06a65f4f3a.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/tables/a8fe8c3dfc971a6c1d3bd1c642addd1dbb8d4dc2a38931f03b29da06a65f4f3a.jpg)

![ac316fb5dd5f8b9629a50f71bc2ebebb82443541b53c99146c036111db71ae84.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/tables/ac316fb5dd5f8b9629a50f71bc2ebebb82443541b53c99146c036111db71ae84.jpg)

![bd7aa3a7ad005f49baf42cc0e31fe732b7e39c58504663c8887a084fa5c2df2c.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/tables/bd7aa3a7ad005f49baf42cc0e31fe732b7e39c58504663c8887a084fa5c2df2c.jpg)

![c521ec1f78f5060be02ba31789e5d6397c4f10d97202a7b8cf7ca9f9d3dc556b.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/tables/c521ec1f78f5060be02ba31789e5d6397c4f10d97202a7b8cf7ca9f9d3dc556b.jpg)

![cb353fd17a324a7d2edc5faa3d62af36b2d6731b81ff57c46a7888f2225e8e80.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/tables/cb353fd17a324a7d2edc5faa3d62af36b2d6731b81ff57c46a7888f2225e8e80.jpg)

![cfd5489d72db14fa9af665fa531680da4deeddd292b6538a4ef2bbc75fe6b248.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/tables/cfd5489d72db14fa9af665fa531680da4deeddd292b6538a4ef2bbc75fe6b248.jpg)

![ddc62c40cfc058e6a1b4fbaf67f711fc709adb67d22f1a576f29032c14bed63c.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/tables/ddc62c40cfc058e6a1b4fbaf67f711fc709adb67d22f1a576f29032c14bed63c.jpg)

![f1dac7f02acf4150282d1734a37ac5dc57beb64e397f57bd0cd6a37043092165.jpg](../iclr_results/21_Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks/tables/f1dac7f02acf4150282d1734a37ac5dc57beb64e397f57bd0cd6a37043092165.jpg)

## Capturing the Temporal Dependence of Training Data Influence


### Images

![2828380b4d6b5783f9d2373c7434c380418538b7d2a1382448b246eabc7ae455.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/images/2828380b4d6b5783f9d2373c7434c380418538b7d2a1382448b246eabc7ae455.jpg)

![2ebf1897b466507fe86d6b78b5e57b611f6aa61bf434901b1a0c4f48bd6895be.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/images/2ebf1897b466507fe86d6b78b5e57b611f6aa61bf434901b1a0c4f48bd6895be.jpg)

![4e7f75810ae2dc62af29b1df164653014e377c3e05c8586ad41045cfa094de82.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/images/4e7f75810ae2dc62af29b1df164653014e377c3e05c8586ad41045cfa094de82.jpg)

![539fbfa782a5354b4d1bd925ef48c3107d3b27c874c0bfcb0f22176a5ff6d404.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/images/539fbfa782a5354b4d1bd925ef48c3107d3b27c874c0bfcb0f22176a5ff6d404.jpg)

![5caf64160d6e40e6b7a8fa958f5df8fbe45b7c33ce86fe23df069b46d8a39cb9.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/images/5caf64160d6e40e6b7a8fa958f5df8fbe45b7c33ce86fe23df069b46d8a39cb9.jpg)

![6d7f12e82e17bf44257f38c22928b089071adb82dd987a9d81e5313f3ec92250.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/images/6d7f12e82e17bf44257f38c22928b089071adb82dd987a9d81e5313f3ec92250.jpg)

![77d3a1aa2c218bead00c96e28b38862c487b0c31331e52555fe78b804ac973bb.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/images/77d3a1aa2c218bead00c96e28b38862c487b0c31331e52555fe78b804ac973bb.jpg)

![8105f37ff27000b3f59f08af267b28cfe450e3a5c510e60c13eaeaa14c09ac75.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/images/8105f37ff27000b3f59f08af267b28cfe450e3a5c510e60c13eaeaa14c09ac75.jpg)

![8399fb377df228056b8c1adea9f0f840a17a58542b80ecb5f059e14141217b2e.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/images/8399fb377df228056b8c1adea9f0f840a17a58542b80ecb5f059e14141217b2e.jpg)

![854c3b4b6bb13df09e75473383577db74aa671fefa0da884106714e86b3a5500.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/images/854c3b4b6bb13df09e75473383577db74aa671fefa0da884106714e86b3a5500.jpg)

![8af667bac81bebff56ac529062926c206908bbabe1b3439aa783e7f049603709.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/images/8af667bac81bebff56ac529062926c206908bbabe1b3439aa783e7f049603709.jpg)

![94b630fca10d8095f1e2c728fa66e2dafdd33165721736df26e1fe5262f07a8d.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/images/94b630fca10d8095f1e2c728fa66e2dafdd33165721736df26e1fe5262f07a8d.jpg)

![a75414881e16315a3f07727291d7dea031f94b1530f868bd35667bde23045dd0.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/images/a75414881e16315a3f07727291d7dea031f94b1530f868bd35667bde23045dd0.jpg)

![a8e5ec66a058e65bff8095e4f2e01eeb98b410e26e38959a5d3432e403230ec5.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/images/a8e5ec66a058e65bff8095e4f2e01eeb98b410e26e38959a5d3432e403230ec5.jpg)

![adda194c172f21758085342d02ed59c9e5312c47032a6ec24a40abe70ff972bd.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/images/adda194c172f21758085342d02ed59c9e5312c47032a6ec24a40abe70ff972bd.jpg)

![c3f68b6a13d4c66670b079b7ecf1a160d5c8b7ff1e592a730b005714344b7b20.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/images/c3f68b6a13d4c66670b079b7ecf1a160d5c8b7ff1e592a730b005714344b7b20.jpg)

![eeb376640e63928a3de6a44e68e4e7a88f1bc6132d160e81a1c43c848febad01.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/images/eeb376640e63928a3de6a44e68e4e7a88f1bc6132d160e81a1c43c848febad01.jpg)

![f086b17a16e8235a8722daafa2d8d9bf3c0f35db8c936c38de75769d4543b091.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/images/f086b17a16e8235a8722daafa2d8d9bf3c0f35db8c936c38de75769d4543b091.jpg)

![f4b0177aeccbb9def861abaa70162a31381b57b22a070bab6509d7da1f6050c2.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/images/f4b0177aeccbb9def861abaa70162a31381b57b22a070bab6509d7da1f6050c2.jpg)

![f5e1134307ae381eb8754cac9446f3f97db656916c14903194cc42ca3c8566a8.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/images/f5e1134307ae381eb8754cac9446f3f97db656916c14903194cc42ca3c8566a8.jpg)

### Tables

![55da94ab03892cdce61106ec6e0c98dd64e1a685de158859c4b0a6aac8fc64c0.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/tables/55da94ab03892cdce61106ec6e0c98dd64e1a685de158859c4b0a6aac8fc64c0.jpg)

![60e20f4126f320d04cd1a7a8f5606808a80deb563b94d35c026e4e60d81005e4.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/tables/60e20f4126f320d04cd1a7a8f5606808a80deb563b94d35c026e4e60d81005e4.jpg)

![75ef4f183f37b9916987fddd9d5f798e938c0bad248ee163d47f752d1d77e5d2.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/tables/75ef4f183f37b9916987fddd9d5f798e938c0bad248ee163d47f752d1d77e5d2.jpg)

![cd70ed158b3140a6174a8d4c5823c7bda84b4dee7a468876d09828e2d1197f70.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/tables/cd70ed158b3140a6174a8d4c5823c7bda84b4dee7a468876d09828e2d1197f70.jpg)

![ed76f14bad8e87579e427a5a36700f27e28c38bd6b797c0dff44657935682bc6.jpg](../iclr_results/22_Capturing the Temporal Dependence of Training Data Influence/tables/ed76f14bad8e87579e427a5a36700f27e28c38bd6b797c0dff44657935682bc6.jpg)

## Two Effects, One Trigger: On the Modality Gap, Object Bias, and Information Imbalance in Contrastive Vision-Language Models


### Images

![115f0eff842ef182c31044af7e74d398285aff122373c2e1893de45eff1ca5a4.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/images/115f0eff842ef182c31044af7e74d398285aff122373c2e1893de45eff1ca5a4.jpg)

![138cf1bdadbf38da1f249b727be5d193354356a3bbef57a06a74ac644aaaa28f.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/images/138cf1bdadbf38da1f249b727be5d193354356a3bbef57a06a74ac644aaaa28f.jpg)

![3240c7eeafed3b3c3412ffb92b9685a037b29b4c68775b6ddce158b3cd9f355c.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/images/3240c7eeafed3b3c3412ffb92b9685a037b29b4c68775b6ddce158b3cd9f355c.jpg)

![351a4aff0c9cd0410f4199c4f8cc6de332a2a3ffed14731995b5bf217a80ebd9.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/images/351a4aff0c9cd0410f4199c4f8cc6de332a2a3ffed14731995b5bf217a80ebd9.jpg)

![483100c8ed2da8681b2bcc74772a78f3e11ca0f81955ecabd71ff143386f1e1b.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/images/483100c8ed2da8681b2bcc74772a78f3e11ca0f81955ecabd71ff143386f1e1b.jpg)

![8b00ac1f6276366c0a9a6225aaf6676ed597f66835af0d200159bc7b96d5dc57.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/images/8b00ac1f6276366c0a9a6225aaf6676ed597f66835af0d200159bc7b96d5dc57.jpg)

![8e8badb6ac54b2d1e41195c9a2b05f73b6d24ac034d2cdd074d7a1e972219202.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/images/8e8badb6ac54b2d1e41195c9a2b05f73b6d24ac034d2cdd074d7a1e972219202.jpg)

![9607859c38b9337cb3431562389de22f26b9fe14f0683fb1d6ab6ca92d15681d.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/images/9607859c38b9337cb3431562389de22f26b9fe14f0683fb1d6ab6ca92d15681d.jpg)

![b01ece53ca8c2a41f8ad0adc71d28f72aa2fbb26a56c2dd79565a8a828e76192.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/images/b01ece53ca8c2a41f8ad0adc71d28f72aa2fbb26a56c2dd79565a8a828e76192.jpg)

![b0ca1f0cf56a55b8b241774169844fa13d773c1f0d13ef30d36997aa32c97188.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/images/b0ca1f0cf56a55b8b241774169844fa13d773c1f0d13ef30d36997aa32c97188.jpg)

![b1c96846ab07780cb766d8e615ba46c9f8cf91889ee483ad160c0610155fa05e.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/images/b1c96846ab07780cb766d8e615ba46c9f8cf91889ee483ad160c0610155fa05e.jpg)

![c00e58edad59eba5674c442d29a55f933736a900ab1e9c0278fdea733c61263a.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/images/c00e58edad59eba5674c442d29a55f933736a900ab1e9c0278fdea733c61263a.jpg)

![cae6924dcb297d12382f23c2fcb26cfc991f436e113b94f0f5602e85b3a430d5.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/images/cae6924dcb297d12382f23c2fcb26cfc991f436e113b94f0f5602e85b3a430d5.jpg)

![cc4cac67326e04839d99913d0cd5b75e923db61730462fee3fd9523a79025928.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/images/cc4cac67326e04839d99913d0cd5b75e923db61730462fee3fd9523a79025928.jpg)

![d3d5562b765794115386794dd3b56e4de9473bdb5a012fac0a1379ac140a8d13.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/images/d3d5562b765794115386794dd3b56e4de9473bdb5a012fac0a1379ac140a8d13.jpg)

![ee2be685621cda948cbabf95879813868a6d71d2bf788710542df2276c12f69d.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/images/ee2be685621cda948cbabf95879813868a6d71d2bf788710542df2276c12f69d.jpg)

![f8f8bfa70335b1ecf236b1935109b551096068d678bd37467c68ec9fe355f11f.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/images/f8f8bfa70335b1ecf236b1935109b551096068d678bd37467c68ec9fe355f11f.jpg)

![f921486d81c0864640ca5d71c64750adfafa83131423790524389cc99975b970.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/images/f921486d81c0864640ca5d71c64750adfafa83131423790524389cc99975b970.jpg)

![f97d88610b71bd856997bdd8fbb91bcb64a401c61df45b3207d3df88f3719e05.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/images/f97d88610b71bd856997bdd8fbb91bcb64a401c61df45b3207d3df88f3719e05.jpg)

### Tables

![057a45c42cc8fdb88db8ed969af102de535e332bef24299468c05c1b12403cfe.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/tables/057a45c42cc8fdb88db8ed969af102de535e332bef24299468c05c1b12403cfe.jpg)

![09f467e81a74db9ce0d298c15c7fcf1aba5405744cb37a6a656874c8311d6d94.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/tables/09f467e81a74db9ce0d298c15c7fcf1aba5405744cb37a6a656874c8311d6d94.jpg)

![43fdc82049e6e10290749cdf2a7f9b073c380f83ac34a8a1cb0a81ae7e68e8a2.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/tables/43fdc82049e6e10290749cdf2a7f9b073c380f83ac34a8a1cb0a81ae7e68e8a2.jpg)

![9b039e44628624a5aba32cdf7bfc4ba0f77d15ebb14c0f3a2b8a50a199f2e6ff.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/tables/9b039e44628624a5aba32cdf7bfc4ba0f77d15ebb14c0f3a2b8a50a199f2e6ff.jpg)

![a20b49f176f7254485c6cb49aa76ec3bfe9ae21e45f94d7e980743baa846bb06.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/tables/a20b49f176f7254485c6cb49aa76ec3bfe9ae21e45f94d7e980743baa846bb06.jpg)

![b905030d7783b348d2a249f6376556e7c06deb26a33141d5a824df687d0ca890.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/tables/b905030d7783b348d2a249f6376556e7c06deb26a33141d5a824df687d0ca890.jpg)

![d7b8b395569d3f66f5524a6f5426ccda8041591cc240e337a60555f9f404bf92.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/tables/d7b8b395569d3f66f5524a6f5426ccda8041591cc240e337a60555f9f404bf92.jpg)

![d8c425a5fb0b71fd73e7eb4a3bd99e16001e92457ba6ea43c600f438dc4c760f.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/tables/d8c425a5fb0b71fd73e7eb4a3bd99e16001e92457ba6ea43c600f438dc4c760f.jpg)

![e3f348abbe7e7dfea36a86bd066ec62b7d7cbd1e499222cf09eb36977513ba78.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/tables/e3f348abbe7e7dfea36a86bd066ec62b7d7cbd1e499222cf09eb36977513ba78.jpg)

![eb7540cac79a69ba679ef987c1eb5ec632eff4028a086487af97170820b24042.jpg](../iclr_results/23_Two Effects, One Trigger_ On the Modality Gap, Object Bias, and Information Imbalance in Contrastive/tables/eb7540cac79a69ba679ef987c1eb5ec632eff4028a086487af97170820b24042.jpg)

## Scaling In-the-Wild Training for Diffusion-based Illumination Harmonization and Editing by Imposing Consistent Light Transport


### Images

![2dcfed8b95d152c6696f5b0da05e98bd22d6dd6d6fddc1442aeadbde0452f114.jpg](../iclr_results/24_Scaling In-the-Wild Training for Diffusion-based Illumination Harmonization and Editing by Imposing /images/2dcfed8b95d152c6696f5b0da05e98bd22d6dd6d6fddc1442aeadbde0452f114.jpg)

![33e1527938c773d553a7f0492af16b9206bc13c62ec5a0353c39b5fab8f8fe66.jpg](../iclr_results/24_Scaling In-the-Wild Training for Diffusion-based Illumination Harmonization and Editing by Imposing /images/33e1527938c773d553a7f0492af16b9206bc13c62ec5a0353c39b5fab8f8fe66.jpg)

![4685e7a7ca706ca458096ee1e1c77c7bc1dd9dd1ccc1309d35452f6d4f6ad922.jpg](../iclr_results/24_Scaling In-the-Wild Training for Diffusion-based Illumination Harmonization and Editing by Imposing /images/4685e7a7ca706ca458096ee1e1c77c7bc1dd9dd1ccc1309d35452f6d4f6ad922.jpg)

![6333fdcfa36209194b02835120c84b90b9619c7ad7d4f53dbf5f9d69218c83ba.jpg](../iclr_results/24_Scaling In-the-Wild Training for Diffusion-based Illumination Harmonization and Editing by Imposing /images/6333fdcfa36209194b02835120c84b90b9619c7ad7d4f53dbf5f9d69218c83ba.jpg)

![70707e15ca70e7c24b2878f8f8ad33c1a034799aa6078cfbcb33cac9288a01de.jpg](../iclr_results/24_Scaling In-the-Wild Training for Diffusion-based Illumination Harmonization and Editing by Imposing /images/70707e15ca70e7c24b2878f8f8ad33c1a034799aa6078cfbcb33cac9288a01de.jpg)

![b1f741e327e6a7bc435ca0adebca045f4fb98e6ad8596092835fe80c50d04cbb.jpg](../iclr_results/24_Scaling In-the-Wild Training for Diffusion-based Illumination Harmonization and Editing by Imposing /images/b1f741e327e6a7bc435ca0adebca045f4fb98e6ad8596092835fe80c50d04cbb.jpg)

### Tables

![2936e82d01da775278628fd463b61d3a231664cdf38891e2d4c85bace67c07e1.jpg](../iclr_results/24_Scaling In-the-Wild Training for Diffusion-based Illumination Harmonization and Editing by Imposing /tables/2936e82d01da775278628fd463b61d3a231664cdf38891e2d4c85bace67c07e1.jpg)

## Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models


### Images

![063ec1786c77885b539daf898d951e1f9716c2d118e64fa74b4e864ab2509b90.jpg](../iclr_results/25_Block Diffusion_ Interpolating Between Autoregressive and Diffusion Language Models/images/063ec1786c77885b539daf898d951e1f9716c2d118e64fa74b4e864ab2509b90.jpg)

![2cb24d8f2a1113b5cc4d9ff1d01178846ffb7cd1e85a89496b516907352c0c8b.jpg](../iclr_results/25_Block Diffusion_ Interpolating Between Autoregressive and Diffusion Language Models/images/2cb24d8f2a1113b5cc4d9ff1d01178846ffb7cd1e85a89496b516907352c0c8b.jpg)

![53af9f3b7f6cd53a24ce1d80054f46827a0fe33601b95720ec82c34c7dbdc385.jpg](../iclr_results/25_Block Diffusion_ Interpolating Between Autoregressive and Diffusion Language Models/images/53af9f3b7f6cd53a24ce1d80054f46827a0fe33601b95720ec82c34c7dbdc385.jpg)

![7bd8577bc3348e4b7be26e0e0491d55afd9fd7b0dd777b7068929ab896f2e5e4.jpg](../iclr_results/25_Block Diffusion_ Interpolating Between Autoregressive and Diffusion Language Models/images/7bd8577bc3348e4b7be26e0e0491d55afd9fd7b0dd777b7068929ab896f2e5e4.jpg)

![ba83464018713d708507d3d20413c6a92c49faeaf2ef1babe19095743555b46b.jpg](../iclr_results/25_Block Diffusion_ Interpolating Between Autoregressive and Diffusion Language Models/images/ba83464018713d708507d3d20413c6a92c49faeaf2ef1babe19095743555b46b.jpg)

![d49cb09aadaf56de9624acc973ba5e18668e607a3b00341a35ff4db562413941.jpg](../iclr_results/25_Block Diffusion_ Interpolating Between Autoregressive and Diffusion Language Models/images/d49cb09aadaf56de9624acc973ba5e18668e607a3b00341a35ff4db562413941.jpg)

### Tables

![3be44892c5a5fb94554e74e29ac9330cdfb854e8623aaa0163619f1a30fc68a6.jpg](../iclr_results/25_Block Diffusion_ Interpolating Between Autoregressive and Diffusion Language Models/tables/3be44892c5a5fb94554e74e29ac9330cdfb854e8623aaa0163619f1a30fc68a6.jpg)

![87063d7a2efec78c2908af6ae411ebe1973476f65db2cc59b271199e09ddb4fa.jpg](../iclr_results/25_Block Diffusion_ Interpolating Between Autoregressive and Diffusion Language Models/tables/87063d7a2efec78c2908af6ae411ebe1973476f65db2cc59b271199e09ddb4fa.jpg)

![8db5b593d8364f503eff6d17affc2f4bbc0b95031f3a5e8856315293f5dfa208.jpg](../iclr_results/25_Block Diffusion_ Interpolating Between Autoregressive and Diffusion Language Models/tables/8db5b593d8364f503eff6d17affc2f4bbc0b95031f3a5e8856315293f5dfa208.jpg)

![a354eb730e6cd9050e990324e9d0b509cdbae086daffee61144e6dc79cae7845.jpg](../iclr_results/25_Block Diffusion_ Interpolating Between Autoregressive and Diffusion Language Models/tables/a354eb730e6cd9050e990324e9d0b509cdbae086daffee61144e6dc79cae7845.jpg)

![de4e4380f871ebeda2febc11a2b096b90c7778b7e6b4450147878eefd2221f64.jpg](../iclr_results/25_Block Diffusion_ Interpolating Between Autoregressive and Diffusion Language Models/tables/de4e4380f871ebeda2febc11a2b096b90c7778b7e6b4450147878eefd2221f64.jpg)

![e605071b9943d4275dd2f55d918b017e468ad1a9b9857a6f70d950b5929fff82.jpg](../iclr_results/25_Block Diffusion_ Interpolating Between Autoregressive and Diffusion Language Models/tables/e605071b9943d4275dd2f55d918b017e468ad1a9b9857a6f70d950b5929fff82.jpg)

![e93d4ca66ef390e325f04ab200c2efcca196c339b46aacd35b26635b9704e488.jpg](../iclr_results/25_Block Diffusion_ Interpolating Between Autoregressive and Diffusion Language Models/tables/e93d4ca66ef390e325f04ab200c2efcca196c339b46aacd35b26635b9704e488.jpg)

![ecc441be1acf46a3efa523e4da0c70c5678459f683071df9409bc305a267d23c.jpg](../iclr_results/25_Block Diffusion_ Interpolating Between Autoregressive and Diffusion Language Models/tables/ecc441be1acf46a3efa523e4da0c70c5678459f683071df9409bc305a267d23c.jpg)

## Robustness Inspired Graph Backdoor Defense


### Images

![0fd72023f9258fd7d37a5d0010cd300094bb05a6ba95e01cd9ce832d5ee288b1.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/images/0fd72023f9258fd7d37a5d0010cd300094bb05a6ba95e01cd9ce832d5ee288b1.jpg)

![22dd1610e7784e6eb221b2a0069df9d16f342a26eac8e32e2ff7b73664963517.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/images/22dd1610e7784e6eb221b2a0069df9d16f342a26eac8e32e2ff7b73664963517.jpg)

![6a589d4c7745051d4ef5e8c467526bd29247005573233a266868149cb09a7720.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/images/6a589d4c7745051d4ef5e8c467526bd29247005573233a266868149cb09a7720.jpg)

![77177645049d01fe08e04d8428dac06460ea99b205c0085d73f733742f65d177.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/images/77177645049d01fe08e04d8428dac06460ea99b205c0085d73f733742f65d177.jpg)

![7a6c48415d517f5e106d7a46485538ffa09e2de5afbdefe75951d5c8f9793341.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/images/7a6c48415d517f5e106d7a46485538ffa09e2de5afbdefe75951d5c8f9793341.jpg)

![7b13b4c69f132e0263891671152bcfd2934ce8b9fa3c35d5812049ff914a372a.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/images/7b13b4c69f132e0263891671152bcfd2934ce8b9fa3c35d5812049ff914a372a.jpg)

![81a55a9a7338f9962e3d17cb1ab3101480b037e7dcaa95b8fa1adde81560da62.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/images/81a55a9a7338f9962e3d17cb1ab3101480b037e7dcaa95b8fa1adde81560da62.jpg)

![9f7b6fd29b1409443b07fcd866576cc2efc4e55050125f6e75e153109a6561fe.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/images/9f7b6fd29b1409443b07fcd866576cc2efc4e55050125f6e75e153109a6561fe.jpg)

![bd1b68ba6bbd6194e8d16a00acfbd04e6b2085501a32a5d3f1eabed2e404150f.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/images/bd1b68ba6bbd6194e8d16a00acfbd04e6b2085501a32a5d3f1eabed2e404150f.jpg)

![c53b129a116634dd59571cbed962c57a302bbb0c9da7f4ac7273c5371f347b51.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/images/c53b129a116634dd59571cbed962c57a302bbb0c9da7f4ac7273c5371f347b51.jpg)

![db95f3840cc3e396a579fd2a7d8d175efcd2114a952785971393a2e2bb19cfcc.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/images/db95f3840cc3e396a579fd2a7d8d175efcd2114a952785971393a2e2bb19cfcc.jpg)

![f87cee4f792ab6926943f7843fd66da9c563889753cfabd81e323d7aacf04d8b.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/images/f87cee4f792ab6926943f7843fd66da9c563889753cfabd81e323d7aacf04d8b.jpg)

### Tables

![49cbae6d56b5d394db0684e40413eb09d585688039b4efd677ea36386a4712b0.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/tables/49cbae6d56b5d394db0684e40413eb09d585688039b4efd677ea36386a4712b0.jpg)

![695f74f0d95d482fbb615a6e80d41e9dd1b5a4d2b314d939f322a16329ec5d56.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/tables/695f74f0d95d482fbb615a6e80d41e9dd1b5a4d2b314d939f322a16329ec5d56.jpg)

![72bc3e18e9872d9f15e6102255d32258bd680866f93777632c3efe071b0f7e31.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/tables/72bc3e18e9872d9f15e6102255d32258bd680866f93777632c3efe071b0f7e31.jpg)

![73550a4040ecc0612c88daf8bc9fe4dd5c2cf7b9a0507382df048d9f2edf4038.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/tables/73550a4040ecc0612c88daf8bc9fe4dd5c2cf7b9a0507382df048d9f2edf4038.jpg)

![735cfcb0c3ccea7177e9140b2b2a259d1b51440227c80e0666182591f96b8c0b.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/tables/735cfcb0c3ccea7177e9140b2b2a259d1b51440227c80e0666182591f96b8c0b.jpg)

![7b14e6e3383c465abd58a677e15d23582dd58b061401336c9f5f4869ae607384.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/tables/7b14e6e3383c465abd58a677e15d23582dd58b061401336c9f5f4869ae607384.jpg)

![863a78f9aa2719123b662afbd9a68657c936ece9d2909c4765078b1fefbfaa61.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/tables/863a78f9aa2719123b662afbd9a68657c936ece9d2909c4765078b1fefbfaa61.jpg)

![895c058cf68a8afee63ae5f24973ef8d36616e10cb9df33f724033a41f354062.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/tables/895c058cf68a8afee63ae5f24973ef8d36616e10cb9df33f724033a41f354062.jpg)

![c6809b4ee7f96459f69987d565e54112fe283317a5caaeaf7c9e1cae3bbd27fa.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/tables/c6809b4ee7f96459f69987d565e54112fe283317a5caaeaf7c9e1cae3bbd27fa.jpg)

![db0c209d47fdbff651630a723ab7e907b90c0069637ec7bc74ccbf5ebf85e949.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/tables/db0c209d47fdbff651630a723ab7e907b90c0069637ec7bc74ccbf5ebf85e949.jpg)

![e62498f51ec51151b2e4edf32189552e02d658df5c142970727fdedd2c4e6be6.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/tables/e62498f51ec51151b2e4edf32189552e02d658df5c142970727fdedd2c4e6be6.jpg)

![e6341bdf7b5481bbd9aa1a825d8dcd3ce37faad06eae346730b42eb5f707d967.jpg](../iclr_results/26_Robustness Inspired Graph Backdoor Defense/tables/e6341bdf7b5481bbd9aa1a825d8dcd3ce37faad06eae346730b42eb5f707d967.jpg)

## Flow Matching with General Discrete Paths: A Kinetic-Optimal Perspective


### Images

![37cd1165a00bb755cf4def65665d6e4a7be7e3298264ae4b4cf479ec5106020e.jpg](../iclr_results/27_Flow Matching with General Discrete Paths_ A Kinetic-Optimal Perspective/images/37cd1165a00bb755cf4def65665d6e4a7be7e3298264ae4b4cf479ec5106020e.jpg)

![6f92f184f8b39b2041adc1549943af362d037e69fa119565d32885b200ebda17.jpg](../iclr_results/27_Flow Matching with General Discrete Paths_ A Kinetic-Optimal Perspective/images/6f92f184f8b39b2041adc1549943af362d037e69fa119565d32885b200ebda17.jpg)

![d0080f01dfef48a0cebf0a77e39f76384ff29957a1378bd4d6704d11be2e608e.jpg](../iclr_results/27_Flow Matching with General Discrete Paths_ A Kinetic-Optimal Perspective/images/d0080f01dfef48a0cebf0a77e39f76384ff29957a1378bd4d6704d11be2e608e.jpg)

### Tables

![1712680f2551937f6ad9880ad2262bb8afd3a8af473e380f6d74c6e74884c451.jpg](../iclr_results/27_Flow Matching with General Discrete Paths_ A Kinetic-Optimal Perspective/tables/1712680f2551937f6ad9880ad2262bb8afd3a8af473e380f6d74c6e74884c451.jpg)

![410f75e996c2b58881ae2630a88b76527c9ad18039ade4f6225a242b265cb795.jpg](../iclr_results/27_Flow Matching with General Discrete Paths_ A Kinetic-Optimal Perspective/tables/410f75e996c2b58881ae2630a88b76527c9ad18039ade4f6225a242b265cb795.jpg)

![6172a7287aeea85d6c312da6b1c9e8be3e48fb8ca9110724fa848ed382db82f2.jpg](../iclr_results/27_Flow Matching with General Discrete Paths_ A Kinetic-Optimal Perspective/tables/6172a7287aeea85d6c312da6b1c9e8be3e48fb8ca9110724fa848ed382db82f2.jpg)

![b25d32ae4e84ed6457270fe659594f148c5f03a755ef63cbcf8dc4c9b8084856.jpg](../iclr_results/27_Flow Matching with General Discrete Paths_ A Kinetic-Optimal Perspective/tables/b25d32ae4e84ed6457270fe659594f148c5f03a755ef63cbcf8dc4c9b8084856.jpg)

![ba862a8e5cee3fbdfa228b86440094fef40b9e5141a83f64fd9e25233ef2dc37.jpg](../iclr_results/27_Flow Matching with General Discrete Paths_ A Kinetic-Optimal Perspective/tables/ba862a8e5cee3fbdfa228b86440094fef40b9e5141a83f64fd9e25233ef2dc37.jpg)

![d077eab4a103d7d785682019814caba547ce28dec0a7c89a44ebbef9b80b22ca.jpg](../iclr_results/27_Flow Matching with General Discrete Paths_ A Kinetic-Optimal Perspective/tables/d077eab4a103d7d785682019814caba547ce28dec0a7c89a44ebbef9b80b22ca.jpg)

## Scaling and evaluating sparse autoencoders


### Images

![11911b7321579182a0fc4d84b68299a52c5aab97eec01b5fa7f57b15fd70327d.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/11911b7321579182a0fc4d84b68299a52c5aab97eec01b5fa7f57b15fd70327d.jpg)

![21a301e9266962160cc93e877e52855d65d577ccf69d5305a9610088a3e26f9a.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/21a301e9266962160cc93e877e52855d65d577ccf69d5305a9610088a3e26f9a.jpg)

![241e93267e55a8523086a5f3f6b83b3b7e7031d68e3703399f7c681ee80d47c1.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/241e93267e55a8523086a5f3f6b83b3b7e7031d68e3703399f7c681ee80d47c1.jpg)

![2d13c293c28c02ebaf42e61fe34d16b4d0a10da1d7131045531228a1a1fa20f4.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/2d13c293c28c02ebaf42e61fe34d16b4d0a10da1d7131045531228a1a1fa20f4.jpg)

![2f68ba26321d27e1aa91be48f70c8d1babcf37393acfb3bf60b3b294a0b2cdc4.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/2f68ba26321d27e1aa91be48f70c8d1babcf37393acfb3bf60b3b294a0b2cdc4.jpg)

![30242fbd62f7984cdaf5df8c3f327fae9a1f086b9bc98d7fafbb8a4e7d0bcc01.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/30242fbd62f7984cdaf5df8c3f327fae9a1f086b9bc98d7fafbb8a4e7d0bcc01.jpg)

![3c5947f4cc920b3123e65406a3fbfd5cc051f3e39c57b075a4e93f6c23201eb8.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/3c5947f4cc920b3123e65406a3fbfd5cc051f3e39c57b075a4e93f6c23201eb8.jpg)

![3d85d5bf1794c43f53cb12da3f84fc023842af913de5b688748cf06c447ba489.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/3d85d5bf1794c43f53cb12da3f84fc023842af913de5b688748cf06c447ba489.jpg)

![49be385065ba7c8ee9c7a4768cbff99e225293945e4cb9ac07160286a3e1a2c7.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/49be385065ba7c8ee9c7a4768cbff99e225293945e4cb9ac07160286a3e1a2c7.jpg)

![49f7e07686c4eddffe64aa7245d059f03f29c1b7e18a73aec866137f643a9f85.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/49f7e07686c4eddffe64aa7245d059f03f29c1b7e18a73aec866137f643a9f85.jpg)

![6753111645025e02e408d4a9787c5f979dcb5509a91ee1eb0d0d743f026e66c5.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/6753111645025e02e408d4a9787c5f979dcb5509a91ee1eb0d0d743f026e66c5.jpg)

![68dd667b87206796153833d779d566a8f63a6fbb2040672bdc7f83b1504a7354.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/68dd667b87206796153833d779d566a8f63a6fbb2040672bdc7f83b1504a7354.jpg)

![73ae1ab793081664b245d9669a57c444c5421e40b8e2332caf813f5cf1e19140.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/73ae1ab793081664b245d9669a57c444c5421e40b8e2332caf813f5cf1e19140.jpg)

![7709c4b8ab89d68983f6cf35da7ca28f3a8deb6efb1e3c98d1e4794dbd92c8e2.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/7709c4b8ab89d68983f6cf35da7ca28f3a8deb6efb1e3c98d1e4794dbd92c8e2.jpg)

![7ee9aa18abe00907128c2e98dba00b9ac6bd18720aebdebface5e51f2f78fe32.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/7ee9aa18abe00907128c2e98dba00b9ac6bd18720aebdebface5e51f2f78fe32.jpg)

![8d42c512eed42fa986c7cd4b90592f044cb12f178f3e0465da47d3585dee5f70.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/8d42c512eed42fa986c7cd4b90592f044cb12f178f3e0465da47d3585dee5f70.jpg)

![964abf779b754f7877a258ec96b44f68dbb7c724be16478746c52e71daa18c04.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/964abf779b754f7877a258ec96b44f68dbb7c724be16478746c52e71daa18c04.jpg)

![9f3d878f9b3a83b3cbc7c41277c9fc65e7d35e3d90a55d8aecaa6718944ef362.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/9f3d878f9b3a83b3cbc7c41277c9fc65e7d35e3d90a55d8aecaa6718944ef362.jpg)

![abd9f28c4ecc363723e59ebc752c36a148843055ef589e34961eaf6615ef9347.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/abd9f28c4ecc363723e59ebc752c36a148843055ef589e34961eaf6615ef9347.jpg)

![abe7485e89d10cca100285ed5c87bb859b286673bb28c1648b643d840f50df1f.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/abe7485e89d10cca100285ed5c87bb859b286673bb28c1648b643d840f50df1f.jpg)

![ade03c65dc203413ae58141f5455608c3788bebef3ceaaed725fe9704dbca05a.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/ade03c65dc203413ae58141f5455608c3788bebef3ceaaed725fe9704dbca05a.jpg)

![af0c3b7c8c8c2af4a36c297d5b537fd3153d0d831a1c4e8bbe5d0f0c791c59de.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/af0c3b7c8c8c2af4a36c297d5b537fd3153d0d831a1c4e8bbe5d0f0c791c59de.jpg)

![b1dce83aad79e0c2ac46eb51a3229464e37a204953b2e8dbf3d091aceb153a5e.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/b1dce83aad79e0c2ac46eb51a3229464e37a204953b2e8dbf3d091aceb153a5e.jpg)

![c1e90b66fbb37a03fe2b05ec9451e95f12983b1714d5911e6a5e99147409b1d8.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/c1e90b66fbb37a03fe2b05ec9451e95f12983b1714d5911e6a5e99147409b1d8.jpg)

![c534d127582965e9d089c0f5095414fb0d68d2cc942d5d0a112da9ed07a45908.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/c534d127582965e9d089c0f5095414fb0d68d2cc942d5d0a112da9ed07a45908.jpg)

![c5566ff3c5c509284257502aff34d054f662237d1a6368d726a9c508f35d4c42.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/c5566ff3c5c509284257502aff34d054f662237d1a6368d726a9c508f35d4c42.jpg)

![d3b294a3d73fff60178ee291439d0dac111d5c26509958c3fd36d94f3be0da11.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/d3b294a3d73fff60178ee291439d0dac111d5c26509958c3fd36d94f3be0da11.jpg)

![d3d0a4b4b24c273f92679f720bd8b6b063ad156431ba56161d56fd55637f7e25.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/d3d0a4b4b24c273f92679f720bd8b6b063ad156431ba56161d56fd55637f7e25.jpg)

![d4e28f5502eabd507646afa705348dfd453df361754797c1af13c6e733906917.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/d4e28f5502eabd507646afa705348dfd453df361754797c1af13c6e733906917.jpg)

![ddc28b8f469c96eae275b9a1eb22be4c93df0bfe8d402675d77ea37f730e3dc3.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/ddc28b8f469c96eae275b9a1eb22be4c93df0bfe8d402675d77ea37f730e3dc3.jpg)

![e19f5cb630ddfb822d3646e2c696b2e940969bc8385408dc6eda13d33f0e1f74.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/e19f5cb630ddfb822d3646e2c696b2e940969bc8385408dc6eda13d33f0e1f74.jpg)

![ea94aee10a0e59f1146759b69b1d794dc3dc8cb724de1e0300d2264d0979b472.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/ea94aee10a0e59f1146759b69b1d794dc3dc8cb724de1e0300d2264d0979b472.jpg)

![eecff7fb9366ee9ced1c8f8fe75f9c00aba19e99841593dfdfb46ff501721bd2.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/eecff7fb9366ee9ced1c8f8fe75f9c00aba19e99841593dfdfb46ff501721bd2.jpg)

![f11e8eca5d16728479f16b598d71dc59348445753528f42d11f92a06d051b930.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/f11e8eca5d16728479f16b598d71dc59348445753528f42d11f92a06d051b930.jpg)

![ffd63414778decbcc269cf0be52a6ab5afba4b2638f622853c7a4f7229a8055a.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/images/ffd63414778decbcc269cf0be52a6ab5afba4b2638f622853c7a4f7229a8055a.jpg)

### Tables

![ba2a9dcf20ca7b78331f5d581dacb52a5ce9be737af71ab5ec9de87189beda9e.jpg](../iclr_results/28_Scaling and evaluating sparse autoencoders/tables/ba2a9dcf20ca7b78331f5d581dacb52a5ce9be737af71ab5ec9de87189beda9e.jpg)

## Cybench: A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models


### Images

![2b23c81088ae392d7c44e2f76730548ae57368ff364f0de54515ee2587f93269.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/images/2b23c81088ae392d7c44e2f76730548ae57368ff364f0de54515ee2587f93269.jpg)

![2e1b28805e327e6d68b6865558f9ae44b391f6a2c8446af108fa01eef3beb614.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/images/2e1b28805e327e6d68b6865558f9ae44b391f6a2c8446af108fa01eef3beb614.jpg)

![4d460f1b76b3767fadc2234495c79522f927dab8e48d18baeee61247eab1c70d.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/images/4d460f1b76b3767fadc2234495c79522f927dab8e48d18baeee61247eab1c70d.jpg)

![650a1d336c671c9c5b05b05cd197c06917447f4b894d4cea0f6559c63f1ec569.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/images/650a1d336c671c9c5b05b05cd197c06917447f4b894d4cea0f6559c63f1ec569.jpg)

![9aeb7a7877fe0c2956e7a744c4fe029950638423cec73c5a3859a84dc2dd0482.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/images/9aeb7a7877fe0c2956e7a744c4fe029950638423cec73c5a3859a84dc2dd0482.jpg)

![e35dfc1fec65a0cf9e3114697fb6eb55e78f8f185fc32f54682dcecbd00bbcf5.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/images/e35dfc1fec65a0cf9e3114697fb6eb55e78f8f185fc32f54682dcecbd00bbcf5.jpg)

### Tables

![00ba4ead77c9323bdf85ea1f8544b790bfd20d4d96e9dedcc0815816fa7d164c.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/00ba4ead77c9323bdf85ea1f8544b790bfd20d4d96e9dedcc0815816fa7d164c.jpg)

![073ea731d3a1b0d3362b41cb8c1a0b3e632920c91d96136726f9bb7fce85aadf.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/073ea731d3a1b0d3362b41cb8c1a0b3e632920c91d96136726f9bb7fce85aadf.jpg)

![079c4fbaa85d7643abbc558c7f9d96bdadb8f70de1af4e1e50d59346d629ba78.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/079c4fbaa85d7643abbc558c7f9d96bdadb8f70de1af4e1e50d59346d629ba78.jpg)

![0aac2c088f30294688af822b2c26002a515e6b6cad6c7982ca2bf5e8a1c5f117.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/0aac2c088f30294688af822b2c26002a515e6b6cad6c7982ca2bf5e8a1c5f117.jpg)

![0cf3305d1e0ffd02f7fafe3ecde45fd091bf68f6147f863bb648886149ea298c.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/0cf3305d1e0ffd02f7fafe3ecde45fd091bf68f6147f863bb648886149ea298c.jpg)

![0d249e729ef8b6fe9ce3580a5ffbee1a6d0baf0731bf31a19969b2ebbe5ba260.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/0d249e729ef8b6fe9ce3580a5ffbee1a6d0baf0731bf31a19969b2ebbe5ba260.jpg)

![0e7641d90f9c443f2ebeb375e2c82796a08e71607f614e87e18385ffb4e68233.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/0e7641d90f9c443f2ebeb375e2c82796a08e71607f614e87e18385ffb4e68233.jpg)

![133e2aff92c8f86451e024c1aa1096fc2ae6c0c2d1ea0081d55507de40295fb3.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/133e2aff92c8f86451e024c1aa1096fc2ae6c0c2d1ea0081d55507de40295fb3.jpg)

![138e7f371530269ed63db0477cc44b43f3e2fab409ef9aacb9b8091ac895c6b3.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/138e7f371530269ed63db0477cc44b43f3e2fab409ef9aacb9b8091ac895c6b3.jpg)

![1715c4c272ae09e7d0d5f1aac25a11b854a61cc4cd60c82498a4aa35b6f357bf.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/1715c4c272ae09e7d0d5f1aac25a11b854a61cc4cd60c82498a4aa35b6f357bf.jpg)

![17f62e7158ce858b475e8fa9c922fd6e14eb25825eb3d996373fc8a5a2bf64f6.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/17f62e7158ce858b475e8fa9c922fd6e14eb25825eb3d996373fc8a5a2bf64f6.jpg)

![18b6fae8ded71dfb76a7998f1a574d65e2d95348887d4b3cab1302f06d8f3180.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/18b6fae8ded71dfb76a7998f1a574d65e2d95348887d4b3cab1302f06d8f3180.jpg)

![24d5fe97515f872f612d689651daeebf0aba3df63e3917130b5444ebf4d3005b.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/24d5fe97515f872f612d689651daeebf0aba3df63e3917130b5444ebf4d3005b.jpg)

![280ebfd2fac84771417056bf55890e214a833196cdf62d8745b9255d570705b6.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/280ebfd2fac84771417056bf55890e214a833196cdf62d8745b9255d570705b6.jpg)

![2e4f260d5eb2cd041d9796603ab1501105fc43a96896195239cb597b92dcfd1f.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/2e4f260d5eb2cd041d9796603ab1501105fc43a96896195239cb597b92dcfd1f.jpg)

![35bec55c7c8ace203d7e19da56eaada659d9b71b06b528b080d16ed4ed2629e3.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/35bec55c7c8ace203d7e19da56eaada659d9b71b06b528b080d16ed4ed2629e3.jpg)

![3a23fe65a3a84b1661898d3576efd147aae0a2768b39f0e8bc880ff1c9ba77f2.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/3a23fe65a3a84b1661898d3576efd147aae0a2768b39f0e8bc880ff1c9ba77f2.jpg)

![42ed0b96ff6022d82e38ece0ef3e381ba6edf4db80426ffc36d15f56ce2ac6b1.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/42ed0b96ff6022d82e38ece0ef3e381ba6edf4db80426ffc36d15f56ce2ac6b1.jpg)

![44caa599df968eb082174676be452961bc39f2356cc5ade7f169e57ecf9da3ea.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/44caa599df968eb082174676be452961bc39f2356cc5ade7f169e57ecf9da3ea.jpg)

![533a6695c0c59026080521ee3bb92da5e29c508301c96de72b0dc33a598fb6cc.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/533a6695c0c59026080521ee3bb92da5e29c508301c96de72b0dc33a598fb6cc.jpg)

![534462e375f1f882bc7838c3cae148b49a04eec9030bad3b9a0c933fc86e5d5a.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/534462e375f1f882bc7838c3cae148b49a04eec9030bad3b9a0c933fc86e5d5a.jpg)

![538000ae08135b01d5910f9a748945238ce39c73e4482a7f037d1d39a62ba82c.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/538000ae08135b01d5910f9a748945238ce39c73e4482a7f037d1d39a62ba82c.jpg)

![5475c8280ef862a680a46599a1d403eedc91f864464b9d87d464d39b2c8c60ab.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/5475c8280ef862a680a46599a1d403eedc91f864464b9d87d464d39b2c8c60ab.jpg)

![5541db845eb5654c50bae6c6250e53f24c52b495cef912f8baf0de353df89d1e.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/5541db845eb5654c50bae6c6250e53f24c52b495cef912f8baf0de353df89d1e.jpg)

![5655425c2a3be09a71b3f34b8a1a271a65a92a440aa819020991fe2544827c7f.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/5655425c2a3be09a71b3f34b8a1a271a65a92a440aa819020991fe2544827c7f.jpg)

![57dc42f3578965581a0deb4388aa93c457c0ec6b05d38a1aaf00b4f744807e31.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/57dc42f3578965581a0deb4388aa93c457c0ec6b05d38a1aaf00b4f744807e31.jpg)

![59722cadaca0097a96982c19c87462b7bf22c8ca2686e63d781db2f576cf4ac5.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/59722cadaca0097a96982c19c87462b7bf22c8ca2686e63d781db2f576cf4ac5.jpg)

![5b72f4655628f5c8148b189736ee3811ede7b781bdc5837d126465f668d860b3.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/5b72f4655628f5c8148b189736ee3811ede7b781bdc5837d126465f668d860b3.jpg)

![5ba2b185d577cd7a925946734c7523b1b8208bd6b427c19722abe9066d2e52af.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/5ba2b185d577cd7a925946734c7523b1b8208bd6b427c19722abe9066d2e52af.jpg)

![5edbfb817f5d210dc97ca6e8f675b01b2ace70a4b262b9b1fc92d03db38339f7.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/5edbfb817f5d210dc97ca6e8f675b01b2ace70a4b262b9b1fc92d03db38339f7.jpg)

![609dcc71849c088b0450e40aa8546724ceafe732db2b566b293a3def9f77f38c.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/609dcc71849c088b0450e40aa8546724ceafe732db2b566b293a3def9f77f38c.jpg)

![68b5b5a24f49cb5f1c4169bb1e18618186a27dec5a026faf062ba59c5f0f6a67.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/68b5b5a24f49cb5f1c4169bb1e18618186a27dec5a026faf062ba59c5f0f6a67.jpg)

![69407e4bc59ac4df378a50269a671346b1d91d0b46e03ffc55184413012ff62b.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/69407e4bc59ac4df378a50269a671346b1d91d0b46e03ffc55184413012ff62b.jpg)

![6a85b263f764fccb2c23444a5264efcd91b59081eac744f6c5a1bca3f2cfd913.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/6a85b263f764fccb2c23444a5264efcd91b59081eac744f6c5a1bca3f2cfd913.jpg)

![6c5e0312374d1894fe2eeeef9a069527ad30a51dead36584803cd824d33653e4.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/6c5e0312374d1894fe2eeeef9a069527ad30a51dead36584803cd824d33653e4.jpg)

![6fa9302c7dba4eabe9aed7be759156ea4f9125ff0b6019f91a98a9158eca326a.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/6fa9302c7dba4eabe9aed7be759156ea4f9125ff0b6019f91a98a9158eca326a.jpg)

![71ec0e753c7c2f653da78c6bd2e158a3e85db2f7ec0923f71debb45a3515d73e.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/71ec0e753c7c2f653da78c6bd2e158a3e85db2f7ec0923f71debb45a3515d73e.jpg)

![7559af3d80d9482bcfee24cbf891e53e3008b6cb84352442e08f9d991abf28c4.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/7559af3d80d9482bcfee24cbf891e53e3008b6cb84352442e08f9d991abf28c4.jpg)

![7731c54bf2c4e0e29c01455be9950239aadaff4c05d17d6c01af90c287d1c7e3.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/7731c54bf2c4e0e29c01455be9950239aadaff4c05d17d6c01af90c287d1c7e3.jpg)

![801537e99bf2b63b24a3000ab1554987a31c2d099900ce2baf13dcf4eea4892a.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/801537e99bf2b63b24a3000ab1554987a31c2d099900ce2baf13dcf4eea4892a.jpg)

![859feae9548179f6bb8dea78222b5e892dd0e7740af325efa26856e488380bb2.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/859feae9548179f6bb8dea78222b5e892dd0e7740af325efa26856e488380bb2.jpg)

![8825e19de3eefac1160b1f931b57d5e24dbc91b2ae1f76060c8430d144431b18.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/8825e19de3eefac1160b1f931b57d5e24dbc91b2ae1f76060c8430d144431b18.jpg)

![88c576b1260bf2c8e9f61709db8c902db4d4a0408f917c5530e888b3de26777c.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/88c576b1260bf2c8e9f61709db8c902db4d4a0408f917c5530e888b3de26777c.jpg)

![8a650449a6b6acf7b6fcc1824d445ba9958755c12e3f4a84a51d5545895d4ee1.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/8a650449a6b6acf7b6fcc1824d445ba9958755c12e3f4a84a51d5545895d4ee1.jpg)

![8c3438af9b119580f52753cba595ba4d35657c87e5de437302f1a0b2539239db.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/8c3438af9b119580f52753cba595ba4d35657c87e5de437302f1a0b2539239db.jpg)

![8cb4b84549a73842aab728e82500896cd8eddab394ea696f7b2eb672e4d4f829.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/8cb4b84549a73842aab728e82500896cd8eddab394ea696f7b2eb672e4d4f829.jpg)

![8d8779551268d2e856007b8a153881466ec35ab9a065c5de733849b0a1347312.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/8d8779551268d2e856007b8a153881466ec35ab9a065c5de733849b0a1347312.jpg)

![8e60e11bda4693255f9277921b5fc3e24e0e728b5cf52139ac0d90f9a906da36.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/8e60e11bda4693255f9277921b5fc3e24e0e728b5cf52139ac0d90f9a906da36.jpg)

![8f90a24cd2f2e53bc7c44ce463a44159edf9f795699d3714f5a2935bcd1da98d.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/8f90a24cd2f2e53bc7c44ce463a44159edf9f795699d3714f5a2935bcd1da98d.jpg)

![924a7339e8078e811ffbc81339e80760b331bbc907e9736b955df22a4da093da.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/924a7339e8078e811ffbc81339e80760b331bbc907e9736b955df22a4da093da.jpg)

![987f202844b5d187fb767b9f7c502528f9aba8e682a6fb9985a83cddff4b879a.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/987f202844b5d187fb767b9f7c502528f9aba8e682a6fb9985a83cddff4b879a.jpg)

![a0966794d5525d9111b8f5b1a66b280595769aabfa60187500e9706c04c697e0.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/a0966794d5525d9111b8f5b1a66b280595769aabfa60187500e9706c04c697e0.jpg)

![a4c089ecf8b6717093df5d60aacb7a5af336b9417f018c7ff1afb10bb1e76bed.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/a4c089ecf8b6717093df5d60aacb7a5af336b9417f018c7ff1afb10bb1e76bed.jpg)

![a59ee28bf492f43f8e183169361f33c2c515d551be612b516f3d40a3a9aa590d.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/a59ee28bf492f43f8e183169361f33c2c515d551be612b516f3d40a3a9aa590d.jpg)

![aa8970c40c76d878ba3435771c9eb4091cbaae5afc615b252d6aa6ffa63c30c1.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/aa8970c40c76d878ba3435771c9eb4091cbaae5afc615b252d6aa6ffa63c30c1.jpg)

![afd62d89e99c42907b8b3dc2a2fb79276526319235fb84f07c80820993d91df5.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/afd62d89e99c42907b8b3dc2a2fb79276526319235fb84f07c80820993d91df5.jpg)

![baea590fb176d9307fee3778960df1f406cf46f6672fea5e05d4b27c7eb33b24.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/baea590fb176d9307fee3778960df1f406cf46f6672fea5e05d4b27c7eb33b24.jpg)

![bd1a6304444e4c945fdf8973403f45d6a60cb19c0194b742c76c938ddf424d01.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/bd1a6304444e4c945fdf8973403f45d6a60cb19c0194b742c76c938ddf424d01.jpg)

![c15e81ba7e2f0f0833e1ca5a03f57d35bed394ca90f0417f85a93686a1144ee0.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/c15e81ba7e2f0f0833e1ca5a03f57d35bed394ca90f0417f85a93686a1144ee0.jpg)

![c1853d90d9dbc1bad22ffe31ff50bbb6e4092887be745e77b80615eb2e047a1b.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/c1853d90d9dbc1bad22ffe31ff50bbb6e4092887be745e77b80615eb2e047a1b.jpg)

![c214fba53d35ddd4540f6d004983bcb3b18942cb9e36494466cb4a78f4928917.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/c214fba53d35ddd4540f6d004983bcb3b18942cb9e36494466cb4a78f4928917.jpg)

![ca08ecd395562156397a92dba481e6e434b3924d72c7cf9a13c167002ec9d600.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/ca08ecd395562156397a92dba481e6e434b3924d72c7cf9a13c167002ec9d600.jpg)

![cb0d6c52703e6789d69e7946e7d47bdc83c41133badc6579df48b7e1fe994343.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/cb0d6c52703e6789d69e7946e7d47bdc83c41133badc6579df48b7e1fe994343.jpg)

![cb543453d2764d0b5b53802c1ffe9f6d5b5ce2cca1f88aae475136a2d3698849.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/cb543453d2764d0b5b53802c1ffe9f6d5b5ce2cca1f88aae475136a2d3698849.jpg)

![cdc6331fd68dec58c574c834412214b7e85b8937370877f266a7471b1938ad26.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/cdc6331fd68dec58c574c834412214b7e85b8937370877f266a7471b1938ad26.jpg)

![d097815b90db8ab0ad2a22eea1f283bdaea941c1716c4d0613abbfc7f7127328.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/d097815b90db8ab0ad2a22eea1f283bdaea941c1716c4d0613abbfc7f7127328.jpg)

![d207c30466a880b94c337188c6679db7208e542041bfd42078a95df18992b5fc.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/d207c30466a880b94c337188c6679db7208e542041bfd42078a95df18992b5fc.jpg)

![d2914ea50c398b7ad151038b02ba7bb581bfaeed1819939fae8850f407e57225.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/d2914ea50c398b7ad151038b02ba7bb581bfaeed1819939fae8850f407e57225.jpg)

![e4ffd223b745bd6794305062f4bfedf5fbcf241ceaca03d2a3fdf3d5f2541512.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/e4ffd223b745bd6794305062f4bfedf5fbcf241ceaca03d2a3fdf3d5f2541512.jpg)

![e6d8a5aa97091c4c2cd4a61a42d2166d2dad631ca2735640f9b7c17599c90cd6.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/e6d8a5aa97091c4c2cd4a61a42d2166d2dad631ca2735640f9b7c17599c90cd6.jpg)

![efee7d5b258842ef7703e0ebdbbf3b0db6a801153da80da24a63481b771982cb.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/efee7d5b258842ef7703e0ebdbbf3b0db6a801153da80da24a63481b771982cb.jpg)

![f160ec749a17215b4532769b026679e0cb12248cbaf6b9c3d13d8bd423d23c3c.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/f160ec749a17215b4532769b026679e0cb12248cbaf6b9c3d13d8bd423d23c3c.jpg)

![f42620694230de6223488eec9e47c38a78e127e7083416b40c9baa7b4909890f.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/f42620694230de6223488eec9e47c38a78e127e7083416b40c9baa7b4909890f.jpg)

![f6c95f9df2fcfe926f5c39a8523d02f367c579c9e6639b89a2239f5d487a6b2d.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/f6c95f9df2fcfe926f5c39a8523d02f367c579c9e6639b89a2239f5d487a6b2d.jpg)

![fa1ddc1614f0621ff322a0c6073d130b0050feefa89e3496787deac099e2b298.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/fa1ddc1614f0621ff322a0c6073d130b0050feefa89e3496787deac099e2b298.jpg)

![ffbc07f827821376bb387dc0cc70160693ba33f1fd3b730bf46c33b6b1d4ddec.jpg](../iclr_results/29_Cybench_ A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models/tables/ffbc07f827821376bb387dc0cc70160693ba33f1fd3b730bf46c33b6b1d4ddec.jpg)

## Booster: Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation


### Images

![5ca0a930e82926bbf72c6ecd0651c8bfaa0935e313654faa5d1dd34859230e16.jpg](../iclr_results/30_Booster_ Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation/images/5ca0a930e82926bbf72c6ecd0651c8bfaa0935e313654faa5d1dd34859230e16.jpg)

![7b4f7fb6556bc234c68b0cf9fd1e4ad0d9ccd14809435e22a119278a7a1aacdf.jpg](../iclr_results/30_Booster_ Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation/images/7b4f7fb6556bc234c68b0cf9fd1e4ad0d9ccd14809435e22a119278a7a1aacdf.jpg)

![82fff3b4451d97bd9448078625f2e35f75bb8243857fd14066a1795ce253f251.jpg](../iclr_results/30_Booster_ Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation/images/82fff3b4451d97bd9448078625f2e35f75bb8243857fd14066a1795ce253f251.jpg)

![eb6bfc7fa1103bf6b316da861c40455bd07abe6aee80d0fbd7426ae80488c045.jpg](../iclr_results/30_Booster_ Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation/images/eb6bfc7fa1103bf6b316da861c40455bd07abe6aee80d0fbd7426ae80488c045.jpg)

![f9a1fa76c0d87e67c2faf9f7ca9a9fd30df78325eff95ebcef1af774e6b8e885.jpg](../iclr_results/30_Booster_ Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation/images/f9a1fa76c0d87e67c2faf9f7ca9a9fd30df78325eff95ebcef1af774e6b8e885.jpg)

### Tables

![00e22c9298100fd77f403aadde6eff94e59b840bad50cd3d3bef2383d2663424.jpg](../iclr_results/30_Booster_ Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation/tables/00e22c9298100fd77f403aadde6eff94e59b840bad50cd3d3bef2383d2663424.jpg)

![02f18c6b365c2fdc41bde8464dd87d2535aecd5cdea1003254491feab3b0f64d.jpg](../iclr_results/30_Booster_ Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation/tables/02f18c6b365c2fdc41bde8464dd87d2535aecd5cdea1003254491feab3b0f64d.jpg)

![350166a8acc3479c4d018248e20e0a67495848134640fc9db54652a1a3e5e471.jpg](../iclr_results/30_Booster_ Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation/tables/350166a8acc3479c4d018248e20e0a67495848134640fc9db54652a1a3e5e471.jpg)

![37eda020680ed0a9a3bf5335266f6702880888ed386f127e5b857e1de6ac01d7.jpg](../iclr_results/30_Booster_ Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation/tables/37eda020680ed0a9a3bf5335266f6702880888ed386f127e5b857e1de6ac01d7.jpg)

![3a33ea2bf70a410787516cfb69f66aa83813da17c8f204e2d7f5249b71a12480.jpg](../iclr_results/30_Booster_ Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation/tables/3a33ea2bf70a410787516cfb69f66aa83813da17c8f204e2d7f5249b71a12480.jpg)

![5f49b4d4b3ed630ecaa20cd89aab5c0258eec299754188737fbd3be4c645dfc2.jpg](../iclr_results/30_Booster_ Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation/tables/5f49b4d4b3ed630ecaa20cd89aab5c0258eec299754188737fbd3be4c645dfc2.jpg)

![647c3e9ae827b119c66c838153ca759df69577c2206d2fed0a5b4da54f179748.jpg](../iclr_results/30_Booster_ Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation/tables/647c3e9ae827b119c66c838153ca759df69577c2206d2fed0a5b4da54f179748.jpg)

![64a94e6717acad541e193a8f62320d77d909cc02567f37322d4ab2640eaa3fb4.jpg](../iclr_results/30_Booster_ Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation/tables/64a94e6717acad541e193a8f62320d77d909cc02567f37322d4ab2640eaa3fb4.jpg)

![848ae76fa698ba0b8e7bafac763bc1bf2b00ad98f6fa88f307f2951dbf2db81f.jpg](../iclr_results/30_Booster_ Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation/tables/848ae76fa698ba0b8e7bafac763bc1bf2b00ad98f6fa88f307f2951dbf2db81f.jpg)

![9011896c1704881d8f70aa7380c852a4edec28b53a5934bfdd0d37e97045867e.jpg](../iclr_results/30_Booster_ Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation/tables/9011896c1704881d8f70aa7380c852a4edec28b53a5934bfdd0d37e97045867e.jpg)

![b6895ba8f3f9e2bcc2202b0d3fb029d7856a9181c2df2bed931c0a3ecfb7a088.jpg](../iclr_results/30_Booster_ Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation/tables/b6895ba8f3f9e2bcc2202b0d3fb029d7856a9181c2df2bed931c0a3ecfb7a088.jpg)

![b89296b965e18883b79b05863bf44d4716d7cb1541b8a81711bda401a716f4de.jpg](../iclr_results/30_Booster_ Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation/tables/b89296b965e18883b79b05863bf44d4716d7cb1541b8a81711bda401a716f4de.jpg)

![bb1695fbdb9f17c977db84ccb25ef6c501f60d9e9291db570f974f242a611341.jpg](../iclr_results/30_Booster_ Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation/tables/bb1695fbdb9f17c977db84ccb25ef6c501f60d9e9291db570f974f242a611341.jpg)

![bcdd32463ac8f159e004a5491134ca8d332ba1272be654e1178136a2d79f9de2.jpg](../iclr_results/30_Booster_ Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation/tables/bcdd32463ac8f159e004a5491134ca8d332ba1272be654e1178136a2d79f9de2.jpg)

![c373cd52907150f015e0c83cf0063f5ef3f860663703ec4d93fdf34624160f66.jpg](../iclr_results/30_Booster_ Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation/tables/c373cd52907150f015e0c83cf0063f5ef3f860663703ec4d93fdf34624160f66.jpg)

![d1041e5258116d110580d0c1e4d066917570a8f531569ed14dfe791b26cab38f.jpg](../iclr_results/30_Booster_ Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation/tables/d1041e5258116d110580d0c1e4d066917570a8f531569ed14dfe791b26cab38f.jpg)

![f8220c44d5af36cbce0718fb22409e4dd1b41fccf640c5971488ee0156f5e367.jpg](../iclr_results/30_Booster_ Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation/tables/f8220c44d5af36cbce0718fb22409e4dd1b41fccf640c5971488ee0156f5e367.jpg)

## Learning Dynamics of LLM Finetuning


### Images

![10c519a4c1dcc9e4022808ba1004eb9b2b01a59117620408d25f5f56641ee421.jpg](../iclr_results/31_Learning Dynamics of LLM Finetuning/images/10c519a4c1dcc9e4022808ba1004eb9b2b01a59117620408d25f5f56641ee421.jpg)

![24b895740b9b49a6bbd511865e0762d28a8f70a046f2a88fe9372f9d87f1912f.jpg](../iclr_results/31_Learning Dynamics of LLM Finetuning/images/24b895740b9b49a6bbd511865e0762d28a8f70a046f2a88fe9372f9d87f1912f.jpg)

![2a61bb1f213be092bfb8c3fe14740efca31a72c435d717833b4ae7a67aeb3a07.jpg](../iclr_results/31_Learning Dynamics of LLM Finetuning/images/2a61bb1f213be092bfb8c3fe14740efca31a72c435d717833b4ae7a67aeb3a07.jpg)

![2f45401658b10e6bfb3821981705a833ac8b34a16c038a4dc27d92ee0441525f.jpg](../iclr_results/31_Learning Dynamics of LLM Finetuning/images/2f45401658b10e6bfb3821981705a833ac8b34a16c038a4dc27d92ee0441525f.jpg)

![307c3f5379d6c771e003799f56d594455688b3a89582c2a264dc1e0bea155003.jpg](../iclr_results/31_Learning Dynamics of LLM Finetuning/images/307c3f5379d6c771e003799f56d594455688b3a89582c2a264dc1e0bea155003.jpg)

![3994201f9ffb4a3849d12acad2aa08bbcc28da8b04871688c8412d9b300996d2.jpg](../iclr_results/31_Learning Dynamics of LLM Finetuning/images/3994201f9ffb4a3849d12acad2aa08bbcc28da8b04871688c8412d9b300996d2.jpg)

![65f41e8a50c0e2cc84121c45e8ae8bfbb19d52c4f3a05f2379cb178782bac04f.jpg](../iclr_results/31_Learning Dynamics of LLM Finetuning/images/65f41e8a50c0e2cc84121c45e8ae8bfbb19d52c4f3a05f2379cb178782bac04f.jpg)

![73c8b3028145790d24ac3d7dd8a26759885e4e4eb03b44b4d7ff000c6563490e.jpg](../iclr_results/31_Learning Dynamics of LLM Finetuning/images/73c8b3028145790d24ac3d7dd8a26759885e4e4eb03b44b4d7ff000c6563490e.jpg)

![817a50532e9a07d3f084ea9c97252c28f38d7d0eafb549a8b73f10955ec02002.jpg](../iclr_results/31_Learning Dynamics of LLM Finetuning/images/817a50532e9a07d3f084ea9c97252c28f38d7d0eafb549a8b73f10955ec02002.jpg)

![85caeadcef9429700430af80dbe44f8ec79d750b2c08dbdb3af56bd5c358cb6e.jpg](../iclr_results/31_Learning Dynamics of LLM Finetuning/images/85caeadcef9429700430af80dbe44f8ec79d750b2c08dbdb3af56bd5c358cb6e.jpg)

![8fe03878b288594b9196d9a2ee70af927dac52cd130c91c4a98f83cbb995d077.jpg](../iclr_results/31_Learning Dynamics of LLM Finetuning/images/8fe03878b288594b9196d9a2ee70af927dac52cd130c91c4a98f83cbb995d077.jpg)

![967bb88451bd2ea2be078ad61a836f02fadccd43a1f319b2de71d17d109d2daf.jpg](../iclr_results/31_Learning Dynamics of LLM Finetuning/images/967bb88451bd2ea2be078ad61a836f02fadccd43a1f319b2de71d17d109d2daf.jpg)

![994ee09b40be9dbc3b78df62970b6226de82db1ce59303d961f3d3dec5c2a24b.jpg](../iclr_results/31_Learning Dynamics of LLM Finetuning/images/994ee09b40be9dbc3b78df62970b6226de82db1ce59303d961f3d3dec5c2a24b.jpg)

![a647086cab7ff2e00adb2c6ca7f8eed9fec3dbe376c9303d9e1a7ff17db6410e.jpg](../iclr_results/31_Learning Dynamics of LLM Finetuning/images/a647086cab7ff2e00adb2c6ca7f8eed9fec3dbe376c9303d9e1a7ff17db6410e.jpg)

![a9d592a9ea42ff2472ba15f5c5319b093e930b5fbf88ecd4c15bc26f9e94a2f0.jpg](../iclr_results/31_Learning Dynamics of LLM Finetuning/images/a9d592a9ea42ff2472ba15f5c5319b093e930b5fbf88ecd4c15bc26f9e94a2f0.jpg)

![ab63c8fb38fb5da0a4f6483424c83dc4850b34fedc25bd34ef4af1c3bf87e712.jpg](../iclr_results/31_Learning Dynamics of LLM Finetuning/images/ab63c8fb38fb5da0a4f6483424c83dc4850b34fedc25bd34ef4af1c3bf87e712.jpg)

![adbd48b0a2aad48fdddec47479e10be6ab8da0c52f57bce52481a45a25fe789b.jpg](../iclr_results/31_Learning Dynamics of LLM Finetuning/images/adbd48b0a2aad48fdddec47479e10be6ab8da0c52f57bce52481a45a25fe789b.jpg)

![addf6d42240fcbfba89af9d2fa77314f196ffd92bee33f3c99a77aa95db2332d.jpg](../iclr_results/31_Learning Dynamics of LLM Finetuning/images/addf6d42240fcbfba89af9d2fa77314f196ffd92bee33f3c99a77aa95db2332d.jpg)

![b223d891599ad26ea1cf216043adde76ab980ef3fc316aee3650ad20d15d7281.jpg](../iclr_results/31_Learning Dynamics of LLM Finetuning/images/b223d891599ad26ea1cf216043adde76ab980ef3fc316aee3650ad20d15d7281.jpg)

![d05d5631e35cf4da942430886e48291222c1dd89fcb26437a434520ba81a5c5c.jpg](../iclr_results/31_Learning Dynamics of LLM Finetuning/images/d05d5631e35cf4da942430886e48291222c1dd89fcb26437a434520ba81a5c5c.jpg)

![f6dd3a1d8ceb86cb06f93dc0c1dc63677d72e99d322da776285cfc4502d1c030.jpg](../iclr_results/31_Learning Dynamics of LLM Finetuning/images/f6dd3a1d8ceb86cb06f93dc0c1dc63677d72e99d322da776285cfc4502d1c030.jpg)

### Tables

![8e83ea74ce5fa8c8a6d956c4e7a8e63c2178a6ebf202e0f507c67f7386c3adc1.jpg](../iclr_results/31_Learning Dynamics of LLM Finetuning/tables/8e83ea74ce5fa8c8a6d956c4e7a8e63c2178a6ebf202e0f507c67f7386c3adc1.jpg)

## Classic but Everlasting: Traditional Gradient-Based Algorithms Converge Fast Even in Time-Varying Multi-Player Games


### Images

![76607fd5e0bb63f8d2f536f998abe9fef7ddeaaf557611d3f45c8ede62bebb84.jpg](../iclr_results/32_Classic but Everlasting_ Traditional Gradient-Based Algorithms Converge Fast Even in Time-Varying Mu/images/76607fd5e0bb63f8d2f536f998abe9fef7ddeaaf557611d3f45c8ede62bebb84.jpg)

![a4365d4697e2b59a64dc949ebc416081f28fe6b1a4948a5398ec7346e680aea7.jpg](../iclr_results/32_Classic but Everlasting_ Traditional Gradient-Based Algorithms Converge Fast Even in Time-Varying Mu/images/a4365d4697e2b59a64dc949ebc416081f28fe6b1a4948a5398ec7346e680aea7.jpg)

![ccb96072f9b5e642e1080b0cf28079391ea628991d7df1f26c87ba48e865f6ad.jpg](../iclr_results/32_Classic but Everlasting_ Traditional Gradient-Based Algorithms Converge Fast Even in Time-Varying Mu/images/ccb96072f9b5e642e1080b0cf28079391ea628991d7df1f26c87ba48e865f6ad.jpg)

## MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation Experts


### Images

![0ee070ef324b4f73fb4e855be037a73f8aed4295ef6d7f61c896fdcc509d592e.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/0ee070ef324b4f73fb4e855be037a73f8aed4295ef6d7f61c896fdcc509d592e.jpg)

![0f4ca131e0d01d6b695c60618a2f8a8e812fd535aebf6ed3cfa7e92cca4969e6.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/0f4ca131e0d01d6b695c60618a2f8a8e812fd535aebf6ed3cfa7e92cca4969e6.jpg)

![15e21561486cc80c5acfcd4c47b134f080b84e810a1b0b5150d0dfe250f34301.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/15e21561486cc80c5acfcd4c47b134f080b84e810a1b0b5150d0dfe250f34301.jpg)

![1ba8022b0bdacd1bffac420dde3a888b61eae8cf8741420701c696941324bdca.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/1ba8022b0bdacd1bffac420dde3a888b61eae8cf8741420701c696941324bdca.jpg)

![1c5ed35905e17ac6488132b5aaee681e37fb0f1f36a308cf54b10a866f0b0667.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/1c5ed35905e17ac6488132b5aaee681e37fb0f1f36a308cf54b10a866f0b0667.jpg)

![1dbd4879a9a1cf46c31d3ed694af2ec4c1c826ac7a3faf7e6b114cf49968be50.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/1dbd4879a9a1cf46c31d3ed694af2ec4c1c826ac7a3faf7e6b114cf49968be50.jpg)

![206e6701d11ec0bfd440a3722734deb4c0c03ade5b04fea302dadc8d79e37c08.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/206e6701d11ec0bfd440a3722734deb4c0c03ade5b04fea302dadc8d79e37c08.jpg)

![2ae50a4ccbbec3334ac1eb9af57f22aaac0965e0051731e8e5714261d6892a1d.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/2ae50a4ccbbec3334ac1eb9af57f22aaac0965e0051731e8e5714261d6892a1d.jpg)

![2bbcc41673fa443d1e51408cde5e8e53fad567312c41a04aa82dcf57c28673b4.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/2bbcc41673fa443d1e51408cde5e8e53fad567312c41a04aa82dcf57c28673b4.jpg)

![2f5b640ae7448c94a376c07ee8c7ab49a3da496a6e79cd2f5c8c3c3df2b3c9aa.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/2f5b640ae7448c94a376c07ee8c7ab49a3da496a6e79cd2f5c8c3c3df2b3c9aa.jpg)

![4140033cc373610963d33b1c93c192ee2a565b88f613f291dd0321dd85b4b8e9.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/4140033cc373610963d33b1c93c192ee2a565b88f613f291dd0321dd85b4b8e9.jpg)

![5527a8d9be1ea4f4cbad82a007ae341712cadf78e3932ff89562b63f744a0347.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/5527a8d9be1ea4f4cbad82a007ae341712cadf78e3932ff89562b63f744a0347.jpg)

![5873ec6468a07ba3a6cc2dfb29fcc37cce1b28528cbbaf7ba6f2465215c51c19.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/5873ec6468a07ba3a6cc2dfb29fcc37cce1b28528cbbaf7ba6f2465215c51c19.jpg)

![6699c2b1da78f57b3363648d0c0a5367687fb6ba25c564f86ae3540f967fa727.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/6699c2b1da78f57b3363648d0c0a5367687fb6ba25c564f86ae3540f967fa727.jpg)

![81929d969bfce4dcf67a7f983d7556a0be0ebc82b6257d63d9363417a9d65332.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/81929d969bfce4dcf67a7f983d7556a0be0ebc82b6257d63d9363417a9d65332.jpg)

![8b4219907832c59e91a1ceb80ac99411f7aee8220b9bb7b64ad2ba816e536fc3.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/8b4219907832c59e91a1ceb80ac99411f7aee8220b9bb7b64ad2ba816e536fc3.jpg)

![8f5b74cce1727e6ae7d7d6c9f5ff5f2939f114e5fc90f3f56be06d558744c463.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/8f5b74cce1727e6ae7d7d6c9f5ff5f2939f114e5fc90f3f56be06d558744c463.jpg)

![92d74f1acfc09a7aac618818847d854f6263bf39f3866a90c860c4fc2cac9ee3.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/92d74f1acfc09a7aac618818847d854f6263bf39f3866a90c860c4fc2cac9ee3.jpg)

![a8c3a4cf6f195022ba0613acaff98fc9be9b76893866887ea92d5270904c2a78.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/a8c3a4cf6f195022ba0613acaff98fc9be9b76893866887ea92d5270904c2a78.jpg)

![a9a29cef4f63e36f855bf13061f8cb0afcd64777850cda7a541b158d4ce7389f.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/a9a29cef4f63e36f855bf13061f8cb0afcd64777850cda7a541b158d4ce7389f.jpg)

![acc70db66b85bc1de987a1dc9d8dffe66e0a0bf17eb135c2d8447563100924ce.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/acc70db66b85bc1de987a1dc9d8dffe66e0a0bf17eb135c2d8447563100924ce.jpg)

![d671a4be0fcebb342c61ab17b3bccf5ab4bcb2a9f34dc57499c3893ec5f27892.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/d671a4be0fcebb342c61ab17b3bccf5ab4bcb2a9f34dc57499c3893ec5f27892.jpg)

![e6b87f551f42678dc6d8357a6cadba509d0121395faa3627e95ea88aa1708fe0.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/e6b87f551f42678dc6d8357a6cadba509d0121395faa3627e95ea88aa1708fe0.jpg)

![ed8652276167ef9819ba6243df5a3237fa87270d875cb0e50af7715a92eb2ad7.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/ed8652276167ef9819ba6243df5a3237fa87270d875cb0e50af7715a92eb2ad7.jpg)

![eda09c69161c634a51edb5bfea3951f4c01396a842c91fb16f3012b3a689d60e.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/eda09c69161c634a51edb5bfea3951f4c01396a842c91fb16f3012b3a689d60e.jpg)

![ee184e0979c71139663b7e3734b461f0aec4420802eb29e49e4e9fd7676ef9a1.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/ee184e0979c71139663b7e3734b461f0aec4420802eb29e49e4e9fd7676ef9a1.jpg)

![f71eeff4701f3b562621ccdc1ae1daf91ba187ede502d4f7f0b104e9f2812d6f.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/f71eeff4701f3b562621ccdc1ae1daf91ba187ede502d4f7f0b104e9f2812d6f.jpg)

![f84dfa70b3445f382f25e2049b1772c504289188ca8b2645e55a2f97dc7e8794.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/images/f84dfa70b3445f382f25e2049b1772c504289188ca8b2645e55a2f97dc7e8794.jpg)

### Tables

![05e8b34820996c0acb77fdbb9a7eff465212731940d9c5ca86feb66aa3735c9e.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/tables/05e8b34820996c0acb77fdbb9a7eff465212731940d9c5ca86feb66aa3735c9e.jpg)

![34c643c5534cd0af89820ca4d22d17ae143d20a30664599c869ad1d695079a33.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/tables/34c643c5534cd0af89820ca4d22d17ae143d20a30664599c869ad1d695079a33.jpg)

![4b63dba49b00a9d80dc78842df77788c21ef76fc80429a96a2205b71a9c03d3d.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/tables/4b63dba49b00a9d80dc78842df77788c21ef76fc80429a96a2205b71a9c03d3d.jpg)

![4c67820071575c122175ea32e7657db2596b6561b3c0f921c0c1f85accec584e.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/tables/4c67820071575c122175ea32e7657db2596b6561b3c0f921c0c1f85accec584e.jpg)

![59efab3f95ff504e196c20b172856c7006ca4551cc3a747b484a6e5527569616.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/tables/59efab3f95ff504e196c20b172856c7006ca4551cc3a747b484a6e5527569616.jpg)

![65e32d653e985096c3021e6b06e3c2b4c63ee0e19b416537db3f0f27873c507c.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/tables/65e32d653e985096c3021e6b06e3c2b4c63ee0e19b416537db3f0f27873c507c.jpg)

![676b7d17e97f543d71eae5e96993ffb5ad5b7a90b34b66beba13f8a126360def.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/tables/676b7d17e97f543d71eae5e96993ffb5ad5b7a90b34b66beba13f8a126360def.jpg)

![7711875ddc70bd7d95bdea0f6d96aab88b1b308e5d56c100d781125378bf9696.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/tables/7711875ddc70bd7d95bdea0f6d96aab88b1b308e5d56c100d781125378bf9696.jpg)

![88c06bfc2d84541493c0935a651bd938483e4ebca1f60ac9370890069043dcce.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/tables/88c06bfc2d84541493c0935a651bd938483e4ebca1f60ac9370890069043dcce.jpg)

![fc20db97be31ab14a7729e1e9c908571d641da8a077b96898eeea7f766c6d029.jpg](../iclr_results/33_MoE++_ Accelerating Mixture-of-Experts Methods with Zero-Computation Experts/tables/fc20db97be31ab14a7729e1e9c908571d641da8a077b96898eeea7f766c6d029.jpg)

## Cheating Automatic LLM Benchmarks: Null Models Achieve High Win Rates


### Images

![15b795098868ad5c50026b4e9d834f8ea0ad21614ab0cf05d5150007ba43a8aa.jpg](../iclr_results/34_Cheating Automatic LLM Benchmarks_ Null Models Achieve High Win Rates/images/15b795098868ad5c50026b4e9d834f8ea0ad21614ab0cf05d5150007ba43a8aa.jpg)

![20b521b5938e0bc2f207ad82efb1e6c1af42a3a835e24dc0f91716dd45cb805c.jpg](../iclr_results/34_Cheating Automatic LLM Benchmarks_ Null Models Achieve High Win Rates/images/20b521b5938e0bc2f207ad82efb1e6c1af42a3a835e24dc0f91716dd45cb805c.jpg)

![3340f81a478845632fd8b761c62cc7bb438ad9462ef0432c9b71f29211f2f9e1.jpg](../iclr_results/34_Cheating Automatic LLM Benchmarks_ Null Models Achieve High Win Rates/images/3340f81a478845632fd8b761c62cc7bb438ad9462ef0432c9b71f29211f2f9e1.jpg)

![3b42133024488ffac99654d51b10ba866915b4fe3dd311c59901c0b60f037b6a.jpg](../iclr_results/34_Cheating Automatic LLM Benchmarks_ Null Models Achieve High Win Rates/images/3b42133024488ffac99654d51b10ba866915b4fe3dd311c59901c0b60f037b6a.jpg)

![5eafb7d5685e268f1b032820569084b1a4795d3a642c72a49d25f241a9b8922c.jpg](../iclr_results/34_Cheating Automatic LLM Benchmarks_ Null Models Achieve High Win Rates/images/5eafb7d5685e268f1b032820569084b1a4795d3a642c72a49d25f241a9b8922c.jpg)

![6dd2a25d902cf52a7f6e049485c829b4c29d3ee6653fab50640bf2071f1d2309.jpg](../iclr_results/34_Cheating Automatic LLM Benchmarks_ Null Models Achieve High Win Rates/images/6dd2a25d902cf52a7f6e049485c829b4c29d3ee6653fab50640bf2071f1d2309.jpg)

![74bad2482fa7343239dbc8a9ff49e0e07129d29dc001a6490bcfa61f2f9b5b9a.jpg](../iclr_results/34_Cheating Automatic LLM Benchmarks_ Null Models Achieve High Win Rates/images/74bad2482fa7343239dbc8a9ff49e0e07129d29dc001a6490bcfa61f2f9b5b9a.jpg)

![abc67e7579f1ba35b0af619a532e1b75dbc799c78ccadc7673ffd6a394e27ac0.jpg](../iclr_results/34_Cheating Automatic LLM Benchmarks_ Null Models Achieve High Win Rates/images/abc67e7579f1ba35b0af619a532e1b75dbc799c78ccadc7673ffd6a394e27ac0.jpg)

![cc28c45597f955ea372fa9e4798accfd2f7c63074c5e4d8ae25f49a928c5b870.jpg](../iclr_results/34_Cheating Automatic LLM Benchmarks_ Null Models Achieve High Win Rates/images/cc28c45597f955ea372fa9e4798accfd2f7c63074c5e4d8ae25f49a928c5b870.jpg)

### Tables

![0909093deb0618d684fb071e74e04a5d60f57186920362828750cdb5e8560006.jpg](../iclr_results/34_Cheating Automatic LLM Benchmarks_ Null Models Achieve High Win Rates/tables/0909093deb0618d684fb071e74e04a5d60f57186920362828750cdb5e8560006.jpg)

![198c697237aaacdb8fba5f1969386677b23a5604587e9626e7f9112d8bdad1cc.jpg](../iclr_results/34_Cheating Automatic LLM Benchmarks_ Null Models Achieve High Win Rates/tables/198c697237aaacdb8fba5f1969386677b23a5604587e9626e7f9112d8bdad1cc.jpg)

![3240e3c422148c11031872c848bf61ea35104d31faf9add1063c57331a4abee2.jpg](../iclr_results/34_Cheating Automatic LLM Benchmarks_ Null Models Achieve High Win Rates/tables/3240e3c422148c11031872c848bf61ea35104d31faf9add1063c57331a4abee2.jpg)

![452cd474417c515455544b738f803ff8d2144dd150c410b4a6b630c3d37e11ab.jpg](../iclr_results/34_Cheating Automatic LLM Benchmarks_ Null Models Achieve High Win Rates/tables/452cd474417c515455544b738f803ff8d2144dd150c410b4a6b630c3d37e11ab.jpg)

![5515eacc5823980230bd33b87999bb8802dddf15cffd69d1bdb4c725cbac7d76.jpg](../iclr_results/34_Cheating Automatic LLM Benchmarks_ Null Models Achieve High Win Rates/tables/5515eacc5823980230bd33b87999bb8802dddf15cffd69d1bdb4c725cbac7d76.jpg)

![598d69b3879743b32dffb4476fb4eff3da33df0a1a177f05906085fddb9b5dc1.jpg](../iclr_results/34_Cheating Automatic LLM Benchmarks_ Null Models Achieve High Win Rates/tables/598d69b3879743b32dffb4476fb4eff3da33df0a1a177f05906085fddb9b5dc1.jpg)

![8676847a0ba3d27fb5075e8983575f9d6dd1c946051cf598a6ac18bd7033c7c0.jpg](../iclr_results/34_Cheating Automatic LLM Benchmarks_ Null Models Achieve High Win Rates/tables/8676847a0ba3d27fb5075e8983575f9d6dd1c946051cf598a6ac18bd7033c7c0.jpg)

![c6828c4003272976b702ee748c95b58187d76a2f6489c2d12811eb3c6154fe93.jpg](../iclr_results/34_Cheating Automatic LLM Benchmarks_ Null Models Achieve High Win Rates/tables/c6828c4003272976b702ee748c95b58187d76a2f6489c2d12811eb3c6154fe93.jpg)

![efaeac453fa7a7380564322faf9ba7f01f50685d8634102c017e82239027f471.jpg](../iclr_results/34_Cheating Automatic LLM Benchmarks_ Null Models Achieve High Win Rates/tables/efaeac453fa7a7380564322faf9ba7f01f50685d8634102c017e82239027f471.jpg)

## Tractable Multi-Agent Reinforcement Learning through Behavioral Economics


### Images

![0a978bba8b11c704f73e96553a4674aa98464bc17fbd2eb54a9b3e70d5569234.jpg](../iclr_results/35_Tractable Multi-Agent Reinforcement Learning through Behavioral Economics/images/0a978bba8b11c704f73e96553a4674aa98464bc17fbd2eb54a9b3e70d5569234.jpg)

![fc7b9f0a045d80a0b7916f758d462b5f22187902873c169695bfc64c23d88da8.jpg](../iclr_results/35_Tractable Multi-Agent Reinforcement Learning through Behavioral Economics/images/fc7b9f0a045d80a0b7916f758d462b5f22187902873c169695bfc64c23d88da8.jpg)

## Do as We Do, Not as You Think: the Conformity of Large Language Models


### Images

![0506d0df3918499ca8e2697253bd9cedc6bad9eabd1e253a650fde208e81b4ef.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/images/0506d0df3918499ca8e2697253bd9cedc6bad9eabd1e253a650fde208e81b4ef.jpg)

![0ac6af92f37384217c29bf1b7698dc153fea8752231658900d55c437d22a4e73.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/images/0ac6af92f37384217c29bf1b7698dc153fea8752231658900d55c437d22a4e73.jpg)

![19412129f2a6fa6a388339551c70540e4fc4c23d42ec8a80dda074fd1cd90beb.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/images/19412129f2a6fa6a388339551c70540e4fc4c23d42ec8a80dda074fd1cd90beb.jpg)

![2955ba86c2d441299fa4db732a56f1931381529d46a890857207fe981895a265.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/images/2955ba86c2d441299fa4db732a56f1931381529d46a890857207fe981895a265.jpg)

![29d96c313792fd02fe04f4566d5410d4ce634c053cea8aef87f776e7a33cfc96.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/images/29d96c313792fd02fe04f4566d5410d4ce634c053cea8aef87f776e7a33cfc96.jpg)

![48fa3f52f59532e625d5eca6220ff0617bf0764f7f382fcc05189f05fc669aa6.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/images/48fa3f52f59532e625d5eca6220ff0617bf0764f7f382fcc05189f05fc669aa6.jpg)

![59f8d86c401b29ccf72a12fa95403d7c3425bed319598f85601b7dec9c29a4dd.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/images/59f8d86c401b29ccf72a12fa95403d7c3425bed319598f85601b7dec9c29a4dd.jpg)

![68f689af785d894ad1b2f4bb891a94b2258f617c5df47e37def9a23413991286.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/images/68f689af785d894ad1b2f4bb891a94b2258f617c5df47e37def9a23413991286.jpg)

![698e0623565b01415de0f39ed93177f65828cbafb90237bb8424591577c59b89.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/images/698e0623565b01415de0f39ed93177f65828cbafb90237bb8424591577c59b89.jpg)

![6ce0d7b44acd9c2665ddf8822e5bcfd11bb7914edda0c676b82c77dab7c30401.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/images/6ce0d7b44acd9c2665ddf8822e5bcfd11bb7914edda0c676b82c77dab7c30401.jpg)

![70871701f1d01ec56562e7491addf96375ca8e95c9bafcbbb709ba6f1d678925.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/images/70871701f1d01ec56562e7491addf96375ca8e95c9bafcbbb709ba6f1d678925.jpg)

![7e230abd0a810177e8a9e480e5a1c2374073866364fd9b5dc4bb819def615e4a.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/images/7e230abd0a810177e8a9e480e5a1c2374073866364fd9b5dc4bb819def615e4a.jpg)

![88a905e55b9d782e231bdeb57535394a4fe499916bb28fce381e4e937f61e715.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/images/88a905e55b9d782e231bdeb57535394a4fe499916bb28fce381e4e937f61e715.jpg)

![eadaffc51315dd52edefb723cb5524261137da6431d7adac0e43c653fb2ef221.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/images/eadaffc51315dd52edefb723cb5524261137da6431d7adac0e43c653fb2ef221.jpg)

![ed09e164bc60ffd92ed642bc55fb6a3ddba657cc89546ca747ca85aebf85661f.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/images/ed09e164bc60ffd92ed642bc55fb6a3ddba657cc89546ca747ca85aebf85661f.jpg)

![eee5df04d2481df9f41bd7113cf00c71a3834b250d2012ba0ef607a9ea1164f4.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/images/eee5df04d2481df9f41bd7113cf00c71a3834b250d2012ba0ef607a9ea1164f4.jpg)

![fd0fbba6e7489acc444ea311f3ef421cc331005959ac75f7f73e8b3d3532e403.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/images/fd0fbba6e7489acc444ea311f3ef421cc331005959ac75f7f73e8b3d3532e403.jpg)

![fe4a10dcb67d14dbabca2d804c26a2aed966809be49f3864e7dca9d93ea63562.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/images/fe4a10dcb67d14dbabca2d804c26a2aed966809be49f3864e7dca9d93ea63562.jpg)

![fecdef9dc7fe807fd1030312fc794b6f22469c7eb73042b244296a61151a1c25.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/images/fecdef9dc7fe807fd1030312fc794b6f22469c7eb73042b244296a61151a1c25.jpg)

### Tables

![071f0aed3ab56aa86cbec094bf73e97e98cccca214c8119a60f4f9fb2d9bedf5.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/tables/071f0aed3ab56aa86cbec094bf73e97e98cccca214c8119a60f4f9fb2d9bedf5.jpg)

![0a583ac8f7924ffafd97456502c9deec67d6fdda5ab31a1513e59b79e23cd852.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/tables/0a583ac8f7924ffafd97456502c9deec67d6fdda5ab31a1513e59b79e23cd852.jpg)

![0bbee2986b426998b937326e671fe14a62be0d5f16dc9dc59498f32d12eb7a1d.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/tables/0bbee2986b426998b937326e671fe14a62be0d5f16dc9dc59498f32d12eb7a1d.jpg)

![114a77880767f2fdfaaf671d2e7bae4b255d159e55367992d379b2981763d1ef.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/tables/114a77880767f2fdfaaf671d2e7bae4b255d159e55367992d379b2981763d1ef.jpg)

![180b11e0c16d54cce3063807ba98acbe69295eeba64149fb6d25a3f5b509188c.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/tables/180b11e0c16d54cce3063807ba98acbe69295eeba64149fb6d25a3f5b509188c.jpg)

![1e3eb93aa7755a0a069a35e6c37aec8b13ec1fa612a7127734b3179d85b5e330.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/tables/1e3eb93aa7755a0a069a35e6c37aec8b13ec1fa612a7127734b3179d85b5e330.jpg)

![2f5c79daaf3257527dafe5d77103a4845a1da94567a1e8deab9e143446082c71.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/tables/2f5c79daaf3257527dafe5d77103a4845a1da94567a1e8deab9e143446082c71.jpg)

![2f5d2d3fd9b805843f891e4b4437559ae7f46cefd0d26f9ba67a1dc3944557f6.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/tables/2f5d2d3fd9b805843f891e4b4437559ae7f46cefd0d26f9ba67a1dc3944557f6.jpg)

![3e86805c7f12b153b3af25c2cc4030008f5d587d98a72e89975ac32b2723fbf6.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/tables/3e86805c7f12b153b3af25c2cc4030008f5d587d98a72e89975ac32b2723fbf6.jpg)

![3f587fed2e8c89ea75d8c9a6d52ba39e0d46bc0f33b2c99cb6c0ecfdbd0d8a88.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/tables/3f587fed2e8c89ea75d8c9a6d52ba39e0d46bc0f33b2c99cb6c0ecfdbd0d8a88.jpg)

![4319abcb3a2fb42f1f2da838e7501ea378ff9a4a8f2c34d03f09c686dce0a600.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/tables/4319abcb3a2fb42f1f2da838e7501ea378ff9a4a8f2c34d03f09c686dce0a600.jpg)

![5a7ce16af8a64db69f1a8d3f3849831c02b23b1af702741406de04f5907efb96.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/tables/5a7ce16af8a64db69f1a8d3f3849831c02b23b1af702741406de04f5907efb96.jpg)

![772fc4ae355d43d105843c77988b5fd4bc64b4743a432a55b6638def394846e1.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/tables/772fc4ae355d43d105843c77988b5fd4bc64b4743a432a55b6638def394846e1.jpg)

![7c35966821f3ab163c6df2b0c82ae0aa369accae14a1653ff0f758b298956c92.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/tables/7c35966821f3ab163c6df2b0c82ae0aa369accae14a1653ff0f758b298956c92.jpg)

![8100ad78af5f67f1c46e5abebafba4299d5b9c6090ea166775c402bb62a07cdf.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/tables/8100ad78af5f67f1c46e5abebafba4299d5b9c6090ea166775c402bb62a07cdf.jpg)

![886262a2d34fdcc0c8cc70e7d8ef80783feb6826b570aee013ee4e98585fc2b9.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/tables/886262a2d34fdcc0c8cc70e7d8ef80783feb6826b570aee013ee4e98585fc2b9.jpg)

![917f70e379d43138617c75e0f995c5529d4c9cc3d224ddd755fed08ce70fd540.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/tables/917f70e379d43138617c75e0f995c5529d4c9cc3d224ddd755fed08ce70fd540.jpg)

![b92648cf7a1e557bb5324cc233a1806ba38843d99e1ab591376b57be859346b8.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/tables/b92648cf7a1e557bb5324cc233a1806ba38843d99e1ab591376b57be859346b8.jpg)

![fd3a7ab1418577ba5a6e853dd79dcd1daa18bf786d017445dbb73a0c2686d862.jpg](../iclr_results/36_Do as We Do, Not as You Think_ the Conformity of Large Language Models/tables/fd3a7ab1418577ba5a6e853dd79dcd1daa18bf786d017445dbb73a0c2686d862.jpg)

## Linear Representations of Political Perspective Emerge in Large Language Models


### Images

![20940972f9344a8257f1070c547b499988915e6f345c16ff226ffe5d6ba42cd3.jpg](../iclr_results/38_Linear Representations of Political Perspective Emerge in Large Language Models/images/20940972f9344a8257f1070c547b499988915e6f345c16ff226ffe5d6ba42cd3.jpg)

![3a8d682e4af4bed3c8e92579d6b54f32b21cee1a48995285e35b50970ad1b97b.jpg](../iclr_results/38_Linear Representations of Political Perspective Emerge in Large Language Models/images/3a8d682e4af4bed3c8e92579d6b54f32b21cee1a48995285e35b50970ad1b97b.jpg)

![3b81b19fd613aa89ad9f1a0cdda831f01833a13dd402ed31684404001eeceee9.jpg](../iclr_results/38_Linear Representations of Political Perspective Emerge in Large Language Models/images/3b81b19fd613aa89ad9f1a0cdda831f01833a13dd402ed31684404001eeceee9.jpg)

![4077f133318606eb85521e2a003bfe097769b784b1d5d287e64dbb222eb5b662.jpg](../iclr_results/38_Linear Representations of Political Perspective Emerge in Large Language Models/images/4077f133318606eb85521e2a003bfe097769b784b1d5d287e64dbb222eb5b662.jpg)

![5211ab438ba852237138af7a527d781f1e2b59a273d59602d5683da1881cd9c7.jpg](../iclr_results/38_Linear Representations of Political Perspective Emerge in Large Language Models/images/5211ab438ba852237138af7a527d781f1e2b59a273d59602d5683da1881cd9c7.jpg)

![58729b106d33eed2f8ca70d1f765633f26ce91daf598f5d94658c0e2f60db613.jpg](../iclr_results/38_Linear Representations of Political Perspective Emerge in Large Language Models/images/58729b106d33eed2f8ca70d1f765633f26ce91daf598f5d94658c0e2f60db613.jpg)

![6e6abf0c87e8fffc5509b838bc460f279df95cff7c506a2743a1b89d5906211e.jpg](../iclr_results/38_Linear Representations of Political Perspective Emerge in Large Language Models/images/6e6abf0c87e8fffc5509b838bc460f279df95cff7c506a2743a1b89d5906211e.jpg)

![7891b5bf0e04a77db9cb4d4f487838b8b7ae880b7dfe2ded37312c21f3904e01.jpg](../iclr_results/38_Linear Representations of Political Perspective Emerge in Large Language Models/images/7891b5bf0e04a77db9cb4d4f487838b8b7ae880b7dfe2ded37312c21f3904e01.jpg)

![872c3dd90dfeaa9b0f97348d73041ca4b8544a5606ba49415ea72acfd34c4b58.jpg](../iclr_results/38_Linear Representations of Political Perspective Emerge in Large Language Models/images/872c3dd90dfeaa9b0f97348d73041ca4b8544a5606ba49415ea72acfd34c4b58.jpg)

![a09019d04d31ad990908ffdcd6aeddf7fc092f16828eba660744450496195a9c.jpg](../iclr_results/38_Linear Representations of Political Perspective Emerge in Large Language Models/images/a09019d04d31ad990908ffdcd6aeddf7fc092f16828eba660744450496195a9c.jpg)

![a9beb52c98a87fc37eb73f7573fb57449b2730cefbcd3c00429bde1cd9188b5e.jpg](../iclr_results/38_Linear Representations of Political Perspective Emerge in Large Language Models/images/a9beb52c98a87fc37eb73f7573fb57449b2730cefbcd3c00429bde1cd9188b5e.jpg)

![c1ca0bd113c088b8f69600470eb97c1ea698fbacae48c6bc275edca85f2f4fd9.jpg](../iclr_results/38_Linear Representations of Political Perspective Emerge in Large Language Models/images/c1ca0bd113c088b8f69600470eb97c1ea698fbacae48c6bc275edca85f2f4fd9.jpg)

![e3b4a84c86287695690810f60ce8d36ec36075b889ccf0babb095602378cd100.jpg](../iclr_results/38_Linear Representations of Political Perspective Emerge in Large Language Models/images/e3b4a84c86287695690810f60ce8d36ec36075b889ccf0babb095602378cd100.jpg)

![e95d4495fbbbcf46f5526ec8f2fd5cc56656b926e9ad671084b61544381c26b2.jpg](../iclr_results/38_Linear Representations of Political Perspective Emerge in Large Language Models/images/e95d4495fbbbcf46f5526ec8f2fd5cc56656b926e9ad671084b61544381c26b2.jpg)

![efe4f2e62e02ea2c528853026e4de5fb1a935caf8c337bcd968828778d5adae5.jpg](../iclr_results/38_Linear Representations of Political Perspective Emerge in Large Language Models/images/efe4f2e62e02ea2c528853026e4de5fb1a935caf8c337bcd968828778d5adae5.jpg)

![f180f77ac0d5733d85631a6ea9e4aafaacdae72b51f5d6938c27e314a9f558cb.jpg](../iclr_results/38_Linear Representations of Political Perspective Emerge in Large Language Models/images/f180f77ac0d5733d85631a6ea9e4aafaacdae72b51f5d6938c27e314a9f558cb.jpg)

![f95b5e08eaa3ecd3aa6517516e9fc4c1839dd80ebbed092d9f2c922f84bf08e2.jpg](../iclr_results/38_Linear Representations of Political Perspective Emerge in Large Language Models/images/f95b5e08eaa3ecd3aa6517516e9fc4c1839dd80ebbed092d9f2c922f84bf08e2.jpg)

### Tables

![2f9d0eff32d0d00f75c4638d2b13b67c2b9f41c0061cfea025202ab738c57d8e.jpg](../iclr_results/38_Linear Representations of Political Perspective Emerge in Large Language Models/tables/2f9d0eff32d0d00f75c4638d2b13b67c2b9f41c0061cfea025202ab738c57d8e.jpg)

![37b2eba4307905ac34a5023ab119157f1b92e60303f60aab1d2e29743a0450ed.jpg](../iclr_results/38_Linear Representations of Political Perspective Emerge in Large Language Models/tables/37b2eba4307905ac34a5023ab119157f1b92e60303f60aab1d2e29743a0450ed.jpg)

![5594b41f4d8f46021eaa2e2518bd8198afa03bbfa5db67a464eb7f654578dad8.jpg](../iclr_results/38_Linear Representations of Political Perspective Emerge in Large Language Models/tables/5594b41f4d8f46021eaa2e2518bd8198afa03bbfa5db67a464eb7f654578dad8.jpg)

![a3f3a0671f9132d6de636d635173b89191fc14d0f7f93f1445ba79d5151ebb41.jpg](../iclr_results/38_Linear Representations of Political Perspective Emerge in Large Language Models/tables/a3f3a0671f9132d6de636d635173b89191fc14d0f7f93f1445ba79d5151ebb41.jpg)

![a84021d2409fbb1e050055edceef51d8ae30e814dfade152fe40ceb92687dabe.jpg](../iclr_results/38_Linear Representations of Political Perspective Emerge in Large Language Models/tables/a84021d2409fbb1e050055edceef51d8ae30e814dfade152fe40ceb92687dabe.jpg)

## Rethinking Reward Modeling in Preference-based Large Language Model Alignment

### Images

![17d926f785d4626bdf36393d1e0ce92f0926f3786acbbd0ecdaaaa0ed44e6adb.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/17d926f785d4626bdf36393d1e0ce92f0926f3786acbbd0ecdaaaa0ed44e6adb.jpg)

![3162b28d13d3e703738b56acf8c96057bbd12fafcb283db8f95876982c3853f2.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/3162b28d13d3e703738b56acf8c96057bbd12fafcb283db8f95876982c3853f2.jpg)

![4bc76dbcb53d7cd551c3a2365a432ef9a135db2582f7862d8538aef8bc18166c.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/4bc76dbcb53d7cd551c3a2365a432ef9a135db2582f7862d8538aef8bc18166c.jpg)

![51e5ee4b6730ff35fa3ff29080a82a536d3021c91cf5f51795c6a2c580edcd47.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/51e5ee4b6730ff35fa3ff29080a82a536d3021c91cf5f51795c6a2c580edcd47.jpg)

![51f6e5b985e79fcd0e3f5b1f84bc4f6dea1c9b4f4a0c508a4911f6334f5d5a74.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/51f6e5b985e79fcd0e3f5b1f84bc4f6dea1c9b4f4a0c508a4911f6334f5d5a74.jpg)

![53cb5870ee184a2c7b9c19701e69773ce94eb01aa9fb6dff2ede791bae29af7c.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/53cb5870ee184a2c7b9c19701e69773ce94eb01aa9fb6dff2ede791bae29af7c.jpg)

![55c3af1622f5c2660ce9473f8c544d342c72364826c2e02ff39f92d42bdebbdd.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/55c3af1622f5c2660ce9473f8c544d342c72364826c2e02ff39f92d42bdebbdd.jpg)

![5f0c3eb731b7581f26155325ac35d9dd0d8e8e12a0a772e2f64464b891bd9d11.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/5f0c3eb731b7581f26155325ac35d9dd0d8e8e12a0a772e2f64464b891bd9d11.jpg)

![72be06683921d3387355ee50820dd6d81c5e9cdd2a4dd1e888d1396e759e9c9a.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/72be06683921d3387355ee50820dd6d81c5e9cdd2a4dd1e888d1396e759e9c9a.jpg)

![7971c6fa1baf8d5fb8fd2d9cbddcc6c9fe97bdc084f9d8c01483aa24e2a4818e.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/7971c6fa1baf8d5fb8fd2d9cbddcc6c9fe97bdc084f9d8c01483aa24e2a4818e.jpg)

![7ad0b647ebad47018aceec52cc49fd726319fa3a5a46e7e668edcca7365d82f8.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/7ad0b647ebad47018aceec52cc49fd726319fa3a5a46e7e668edcca7365d82f8.jpg)

![7c01f9129a0500320e094ab028d8d0b4ee3b62424d6cabd1e7697b2d5842457f.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/7c01f9129a0500320e094ab028d8d0b4ee3b62424d6cabd1e7697b2d5842457f.jpg)

![7d9b8a7903dc080f534bd20776d8c4a87f84ce04538e4ffc4ad9dec415de1002.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/7d9b8a7903dc080f534bd20776d8c4a87f84ce04538e4ffc4ad9dec415de1002.jpg)

![924f1552e14e5eaadb22b06239ae8ab0073be4945d33c84057f06cd49f83b58d.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/924f1552e14e5eaadb22b06239ae8ab0073be4945d33c84057f06cd49f83b58d.jpg)

![a9e383e251d662b01955ba52c0307ca7a1d06b14c9bbb7f124244ba7cb4fe96a.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/a9e383e251d662b01955ba52c0307ca7a1d06b14c9bbb7f124244ba7cb4fe96a.jpg)

![b78dd5f70fe8def14336b62ab054d94d291f6651a8fba0d57b686ad07d256be3.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/b78dd5f70fe8def14336b62ab054d94d291f6651a8fba0d57b686ad07d256be3.jpg)

![b87061257e4b7dd6410225188ac967f8bf5c5b31129293740bb27a05649e3a8d.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/b87061257e4b7dd6410225188ac967f8bf5c5b31129293740bb27a05649e3a8d.jpg)

![be96504e8ef5308d457ea73f8d8a31405a8a7505b021a6966fee79104833cdd2.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/be96504e8ef5308d457ea73f8d8a31405a8a7505b021a6966fee79104833cdd2.jpg)

![c4fbe49db277c72a49bc80f5bb02163cdd48fc8c9681494cd48aad5ad09ab380.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/c4fbe49db277c72a49bc80f5bb02163cdd48fc8c9681494cd48aad5ad09ab380.jpg)

![d1cd5e343fc6f2ae5196ac22cd5e9c95033e61c27209b228f16292cabfa8077f.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/d1cd5e343fc6f2ae5196ac22cd5e9c95033e61c27209b228f16292cabfa8077f.jpg)

![da626c790add449959d9dc094046f6ffd05facb2c411cd0ac69c59f4992c63a9.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/da626c790add449959d9dc094046f6ffd05facb2c411cd0ac69c59f4992c63a9.jpg)

![df57b1ba1adc697576b53e7ef52cab0d51050ce5b34d3db1042d6be14345e0b6.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/df57b1ba1adc697576b53e7ef52cab0d51050ce5b34d3db1042d6be14345e0b6.jpg)

![e09378e41ac0b06fe05a8baeae2e50103f3e72c2268524fc415f3a829b90b22f.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/e09378e41ac0b06fe05a8baeae2e50103f3e72c2268524fc415f3a829b90b22f.jpg)

![e7dd9aa2fbf79e2d0ad90e51c8d6cb63525ca5de10e1852f0b94676e69778b0f.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/e7dd9aa2fbf79e2d0ad90e51c8d6cb63525ca5de10e1852f0b94676e69778b0f.jpg)

![ee75985c80b509cac4f07cebf5d913eb548196acada1012243b880a34b9e7787.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/ee75985c80b509cac4f07cebf5d913eb548196acada1012243b880a34b9e7787.jpg)

![f6d07d60ff02c570d201f0a14a9f79a4fe75aeb06530f7d6282b54eb402e749f.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/f6d07d60ff02c570d201f0a14a9f79a4fe75aeb06530f7d6282b54eb402e749f.jpg)

![fedfbdc603ccff7d3f85bce875f465fa711e1f7563dc896e603c5050fa1eb713.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/fedfbdc603ccff7d3f85bce875f465fa711e1f7563dc896e603c5050fa1eb713.jpg)

![fefb35af84cf0f0e64d7e8e15556616be53f77ec31ea93703098e19de930e240.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/images/fefb35af84cf0f0e64d7e8e15556616be53f77ec31ea93703098e19de930e240.jpg)

### Tables

![14b8e54569ae3fe0132fd43bef4a9b13e82df2562d0e63704affdb1a8b794b8a.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/tables/14b8e54569ae3fe0132fd43bef4a9b13e82df2562d0e63704affdb1a8b794b8a.jpg)

![982975ad0ad24df165e5ba9fd20edc0a23bbecfd3a8bdc4ba375139848300345.jpg](../iclr_results/39_Rethinking Reward Modeling in Preference-based Large Language Model Alignment/tables/982975ad0ad24df165e5ba9fd20edc0a23bbecfd3a8bdc4ba375139848300345.jpg)
