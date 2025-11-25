# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 95 of 100

## 目录 (Table of Contents)

1. [kNN Attention Demystified: A Theoretical Exploration for Scalable Transformers](#kNN-Attention-Demystified-A-Theoretical-Exploration-for-Scalable-Transformers)
2. [Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Designing Models](#Structuring-Benchmark-into-Knowledge-Graphs-to-Assist-Large-Language-Models-in-Retrieving-and-Designing-Models)
3. [Single-agent Poisoning Attacks Suffice to Ruin Multi-Agent Learning](#Single-agent-Poisoning-Attacks-Suffice-to-Ruin-Multi-Agent-Learning)
4. [Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models](#Cross-Modal-Safety-Mechanism-Transfer-in-Large-Vision-Language-Models)
5. [Instruct-SkillMix: A Powerful Pipeline for LLM Instruction Tuning](#Instruct-SkillMix-A-Powerful-Pipeline-for-LLM-Instruction-Tuning)
6. [FedTMOS: Efficient One-Shot Federated Learning with Tsetlin Machine](#FedTMOS-Efficient-One-Shot-Federated-Learning-with-Tsetlin-Machine)
7. [Test of Time: A Benchmark for Evaluating LLMs on Temporal Reasoning](#Test-of-Time-A-Benchmark-for-Evaluating-LLMs-on-Temporal-Reasoning)
8. [Demystifying Online Clustering of Bandits: Enhanced Exploration Under Stochastic and Smoothed Adversarial Contexts](#Demystifying-Online-Clustering-of-Bandits-Enhanced-Exploration-Under-Stochastic-and-Smoothed-Adversarial-Contexts)
9. [Durable Quantization Conditioned Misalignment Attack on Large Language Models](#Durable-Quantization-Conditioned-Misalignment-Attack-on-Large-Language-Models)
10. [Learning a Fast Mixing Exogenous Block MDP using a Single Trajectory](#Learning-a-Fast-Mixing-Exogenous-Block-MDP-using-a-Single-Trajectory)
11. [Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition](#Efficient-Automated-Circuit-Discovery-in-Transformers-using-Contextual-Decomposition)
12. [RankSHAP: Shapley Value Based Feature Attributions for Learning to Rank](#RankSHAP-Shapley-Value-Based-Feature-Attributions-for-Learning-to-Rank)
13. [Tailoring Mixup to Data for Calibration](#Tailoring-Mixup-to-Data-for-Calibration)
14. [BrainOOD: Out-of-distribution Generalizable Brain Network Analysis](#BrainOOD-Out-of-distribution-Generalizable-Brain-Network-Analysis)
15. [AI2TALE: An Innovative Information Theory-based Approach for Learning to Localize Phishing Attacks](#AI2TALE-An-Innovative-Information-Theory-based-Approach-for-Learning-to-Localize-Phishing-Attacks)
16. [Enhancing Federated Domain Adaptation with Multi-Domain Prototype-Based Federated Fine-Tuning](#Enhancing-Federated-Domain-Adaptation-with-Multi-Domain-Prototype-Based-Federated-Fine-Tuning)
17. [Machine Unlearning via Simulated Oracle Matching](#Machine-Unlearning-via-Simulated-Oracle-Matching)
18. [BRAID: Input-driven Nonlinear Dynamical Modeling of Neural-Behavioral Data](#BRAID-Input-driven-Nonlinear-Dynamical-Modeling-of-Neural-Behavioral-Data)
19. [Scaling Speech-Text Pre-training with Synthetic Interleaved Data](#Scaling-Speech-Text-Pre-training-with-Synthetic-Interleaved-Data)
20. [Input Space Mode Connectivity in Deep Neural Networks](#Input-Space-Mode-Connectivity-in-Deep-Neural-Networks)
21. [Automatic Curriculum Expert Iteration for Reliable LLM Reasoning](#Automatic-Curriculum-Expert-Iteration-for-Reliable-LLM-Reasoning)
22. [Extendable and Iterative Structure Learning Strategy for Bayesian Networks](#Extendable-and-Iterative-Structure-Learning-Strategy-for-Bayesian-Networks)
23. [An Information Criterion for Controlled Disentanglement of Multimodal Data](#An-Information-Criterion-for-Controlled-Disentanglement-of-Multimodal-Data)
24. [Simulating Human-like Daily Activities with Desire-driven Autonomy](#Simulating-Human-like-Daily-Activities-with-Desire-driven-Autonomy)
25. [ConMix: Contrastive Mixup at Representation Level for Long-tailed Deep Clustering](#ConMix-Contrastive-Mixup-at-Representation-Level-for-Long-tailed-Deep-Clustering)
26. [CR2PQ: Continuous Relative Rotary Positional Query for Dense Visual Representation Learning](#CR2PQ-Continuous-Relative-Rotary-Positional-Query-for-Dense-Visual-Representation-Learning)
27. [Iterative Substructure Extraction for Molecular Relational Learning with Interactive Graph Information Bottleneck](#Iterative-Substructure-Extraction-for-Molecular-Relational-Learning-with-Interactive-Graph-Information-Bottleneck)
28. [Fast training and sampling of Restricted Boltzmann Machines](#Fast-training-and-sampling-of-Restricted-Boltzmann-Machines)
29. [Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout](#Model-Agnostic-Knowledge-Guided-Correction-for-Improved-Neural-Surrogate-Rollout)
30. [On Speeding Up Language Model Evaluation](#On-Speeding-Up-Language-Model-Evaluation)
31. [Video Action Differencing](#Video-Action-Differencing)
32. [UniCoTT: A Unified Framework for Structural Chain-of-Thought Distillation](#UniCoTT-A-Unified-Framework-for-Structural-Chain-of-Thought-Distillation)
33. [Selective Label Enhancement Learning for Test-Time Adaptation](#Selective-Label-Enhancement-Learning-for-Test-Time-Adaptation)
34. [Multi-Label Node Classification with Label Influence Propagation](#Multi-Label-Node-Classification-with-Label-Influence-Propagation)
35. [CREMA: Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion](#CREMA-Generalizable-and-Efficient-Video-Language-Reasoning-via-Multimodal-Modular-Fusion)
36. [Learn-by-interact: A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments](#Learn-by-interact-A-Data-Centric-Framework-For-Self-Adaptive-Agents-in-Realistic-Environments)
37. [From Pixels to Tokens: Byte-Pair Encoding on Quantized Visual Modalities](#From-Pixels-to-Tokens-Byte-Pair-Encoding-on-Quantized-Visual-Modalities)

---


## kNN Attention Demystified: A Theoretical Exploration for Scalable Transformers

### Images

![0104b97f5e91ff86d2da7a3142cf7cbfe6527599f4790b918e8d1890daffe522.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/images/0104b97f5e91ff86d2da7a3142cf7cbfe6527599f4790b918e8d1890daffe522.jpg)

![063f8d4936d4580400e46f20eb7063942a5984c436d9466e460e87c13054cd0c.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/images/063f8d4936d4580400e46f20eb7063942a5984c436d9466e460e87c13054cd0c.jpg)

![15531490c4348d3680cf5a7f778e4a77d23d9013432cc8ac4cc5c8396bf86ad1.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/images/15531490c4348d3680cf5a7f778e4a77d23d9013432cc8ac4cc5c8396bf86ad1.jpg)

![2dc03bb2ba2a07b57a97f6593bcb3a091e8c4e8688685fb889895e9b44036edc.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/images/2dc03bb2ba2a07b57a97f6593bcb3a091e8c4e8688685fb889895e9b44036edc.jpg)

![472656becdfb5e73f0dd85fcd65ac59b486fd8afc81f1fd83501faa12eaba739.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/images/472656becdfb5e73f0dd85fcd65ac59b486fd8afc81f1fd83501faa12eaba739.jpg)

![678e9b21bf1a500db970f7fe05e1cb3a1c5b47619ac474760ec2969bbdd62a1b.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/images/678e9b21bf1a500db970f7fe05e1cb3a1c5b47619ac474760ec2969bbdd62a1b.jpg)

![6b63caf3d99807bd5c866e4e0a06cc5854dceca12703fd00aaca43ecc59c807b.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/images/6b63caf3d99807bd5c866e4e0a06cc5854dceca12703fd00aaca43ecc59c807b.jpg)

![6b7585babdb57e97c19ab4d6dae950e2fecdb590567318f60f650e6045f8f11d.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/images/6b7585babdb57e97c19ab4d6dae950e2fecdb590567318f60f650e6045f8f11d.jpg)

![8517517ca36808dc9f4ea880579bdd154c3ef02746ca45063e2cc832b9ba559e.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/images/8517517ca36808dc9f4ea880579bdd154c3ef02746ca45063e2cc832b9ba559e.jpg)

![87990154d1c6ba692b3957932b2d22a79ee10111c194cc988f5b22ab896a9069.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/images/87990154d1c6ba692b3957932b2d22a79ee10111c194cc988f5b22ab896a9069.jpg)

![89f57d7e4ba9be91788105050e525aefe26974213338ec836c0ad66290f231c5.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/images/89f57d7e4ba9be91788105050e525aefe26974213338ec836c0ad66290f231c5.jpg)

![9c384eef414061f681566f2623429f89d182b74f0fe7724cba4ee57fc2292240.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/images/9c384eef414061f681566f2623429f89d182b74f0fe7724cba4ee57fc2292240.jpg)

![a11fb8ea04729dc91e12d67c693d031608e1b47a333b46f8f81d490bfb82be49.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/images/a11fb8ea04729dc91e12d67c693d031608e1b47a333b46f8f81d490bfb82be49.jpg)

![af69dbd79f7ddb7d2840c709934e96aa208a39d564a4b606f3ac586f439408c8.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/images/af69dbd79f7ddb7d2840c709934e96aa208a39d564a4b606f3ac586f439408c8.jpg)

![b456bd06c2c4633dfab62dcfa7a74a216bdd757ca4a9580f54556dce4a8fc7bf.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/images/b456bd06c2c4633dfab62dcfa7a74a216bdd757ca4a9580f54556dce4a8fc7bf.jpg)

![b5da2d636b0e6657ec9a7fde03678d33583ea216a304c38aa0da6cbfc84a3cc5.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/images/b5da2d636b0e6657ec9a7fde03678d33583ea216a304c38aa0da6cbfc84a3cc5.jpg)

![eeb7efbc4fae3ae498b1e9a8d0d50330ade4497f2c925769734947c2530f678a.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/images/eeb7efbc4fae3ae498b1e9a8d0d50330ade4497f2c925769734947c2530f678a.jpg)

![f27778130dbfd909cce10e624d88b30a167783f6f428cdd06f7a500653eca08f.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/images/f27778130dbfd909cce10e624d88b30a167783f6f428cdd06f7a500653eca08f.jpg)

![f92e164d52dc8e3b1fc41c9990f28b86e0de6cb79128f0de74926e301536b87e.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/images/f92e164d52dc8e3b1fc41c9990f28b86e0de6cb79128f0de74926e301536b87e.jpg)

### Tables

![0a0846ba105878a0871e0f8c0d07122547324abcf09e8c984b84cc16d9bd70b0.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/tables/0a0846ba105878a0871e0f8c0d07122547324abcf09e8c984b84cc16d9bd70b0.jpg)

![2f3249196cc2dc748ae88652651d36318fb9753e57c6862729dab1e369abfd41.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/tables/2f3249196cc2dc748ae88652651d36318fb9753e57c6862729dab1e369abfd41.jpg)

![378d66a2d9f2c5acd8f8ff575581bbaa1b8344931c99087e49a31a6bb5faa676.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/tables/378d66a2d9f2c5acd8f8ff575581bbaa1b8344931c99087e49a31a6bb5faa676.jpg)

![49a7d0416d94657b707890199d169a3b05a79ced4e65ba1cd2a4cc1dec4532d4.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/tables/49a7d0416d94657b707890199d169a3b05a79ced4e65ba1cd2a4cc1dec4532d4.jpg)

![5a08978fbd90799cad9cecfabb951915f120bd581e559304bae88a3dd764690d.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/tables/5a08978fbd90799cad9cecfabb951915f120bd581e559304bae88a3dd764690d.jpg)

![8b7ba3a71fc2b526eb320ec3b103450a0632ca2217781fdee480a1de2ec26492.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/tables/8b7ba3a71fc2b526eb320ec3b103450a0632ca2217781fdee480a1de2ec26492.jpg)

![c3caf12dbb59e54f063d0d5ea6ddec584be055ba9a2163f912649bcd931b59a1.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/tables/c3caf12dbb59e54f063d0d5ea6ddec584be055ba9a2163f912649bcd931b59a1.jpg)

![f3ecdff8d7c119815eb287ea595a316af94b1fc9144aefb9dea6a125dcc521f8.jpg](../iclr_results/3495_FreDF_ Learning to Forecast in the Frequency Domain/tables/f3ecdff8d7c119815eb287ea595a316af94b1fc9144aefb9dea6a125dcc521f8.jpg)

## kNN Attention Demystified: A Theoretical Exploration for Scalable Transformers


### Images

![42dd67310cdf7c67c1a7e55c5723211b37245712aee81b30af76091bdf1c4eb2.jpg](../iclr_results/3496_kNN Attention Demystified_ A Theoretical Exploration for Scalable Transformers/images/42dd67310cdf7c67c1a7e55c5723211b37245712aee81b30af76091bdf1c4eb2.jpg)

![5c7420e3714e5280ca1709f2b33a6040997d7885eeae1a11ee8f76a3c4e230fa.jpg](../iclr_results/3496_kNN Attention Demystified_ A Theoretical Exploration for Scalable Transformers/images/5c7420e3714e5280ca1709f2b33a6040997d7885eeae1a11ee8f76a3c4e230fa.jpg)

![67de0d934f9b17039ee97840a3194a5a2c0e9b575583115e828e8eef34be5076.jpg](../iclr_results/3496_kNN Attention Demystified_ A Theoretical Exploration for Scalable Transformers/images/67de0d934f9b17039ee97840a3194a5a2c0e9b575583115e828e8eef34be5076.jpg)

![7da4b781dffa443370323c55eec5a5e65cd80c4e2616c05d2509fbcb49d4d02a.jpg](../iclr_results/3496_kNN Attention Demystified_ A Theoretical Exploration for Scalable Transformers/images/7da4b781dffa443370323c55eec5a5e65cd80c4e2616c05d2509fbcb49d4d02a.jpg)

![9935e45c1ad1251e181b158deeb9954d63a6b944fd2fa942fc13ea5bdc969e3a.jpg](../iclr_results/3496_kNN Attention Demystified_ A Theoretical Exploration for Scalable Transformers/images/9935e45c1ad1251e181b158deeb9954d63a6b944fd2fa942fc13ea5bdc969e3a.jpg)

![b9ee6bc516006710534cb7aa3087196e029f6b58806a90190fdda8349ea14a85.jpg](../iclr_results/3496_kNN Attention Demystified_ A Theoretical Exploration for Scalable Transformers/images/b9ee6bc516006710534cb7aa3087196e029f6b58806a90190fdda8349ea14a85.jpg)

![f6efe4afe1ffd95da98acc3a31e652786f3b29858e4b501492849f0c690df3a2.jpg](../iclr_results/3496_kNN Attention Demystified_ A Theoretical Exploration for Scalable Transformers/images/f6efe4afe1ffd95da98acc3a31e652786f3b29858e4b501492849f0c690df3a2.jpg)

### Tables

![04383b19115c9d9f8dcab31f3d8f3c25c3ae970c2b8673abfe407e79245180a0.jpg](../iclr_results/3496_kNN Attention Demystified_ A Theoretical Exploration for Scalable Transformers/tables/04383b19115c9d9f8dcab31f3d8f3c25c3ae970c2b8673abfe407e79245180a0.jpg)

![338ca3d2510189583f0ff3e3d00a8fe60d492c2626ece1988e7c18c8cb6b44ae.jpg](../iclr_results/3496_kNN Attention Demystified_ A Theoretical Exploration for Scalable Transformers/tables/338ca3d2510189583f0ff3e3d00a8fe60d492c2626ece1988e7c18c8cb6b44ae.jpg)

![67cde74499eb822658a08428b9d03f61d2703be19543a3707b78e90e61d9703a.jpg](../iclr_results/3496_kNN Attention Demystified_ A Theoretical Exploration for Scalable Transformers/tables/67cde74499eb822658a08428b9d03f61d2703be19543a3707b78e90e61d9703a.jpg)

![73b459af9fcacb4a744a44acd27322e55f5d7663a95733f56546b63f2dba3be6.jpg](../iclr_results/3496_kNN Attention Demystified_ A Theoretical Exploration for Scalable Transformers/tables/73b459af9fcacb4a744a44acd27322e55f5d7663a95733f56546b63f2dba3be6.jpg)

## Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Designing Models


### Images

![10167906f38950b48c158c679c27645ab306a3ca72813aa8f26b99265a156613.jpg](../iclr_results/3497_Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Design/images/10167906f38950b48c158c679c27645ab306a3ca72813aa8f26b99265a156613.jpg)

![2655e47142fbf842d60439cc92fc7acfbabb44ac46d3f042678c74e2df839b5b.jpg](../iclr_results/3497_Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Design/images/2655e47142fbf842d60439cc92fc7acfbabb44ac46d3f042678c74e2df839b5b.jpg)

![354bc097834b30f17506f251b749f114de91cbbeee57223f41a8438edacc16ab.jpg](../iclr_results/3497_Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Design/images/354bc097834b30f17506f251b749f114de91cbbeee57223f41a8438edacc16ab.jpg)

![4ce970d4c1b2d51ca49c603632ed7b321dfb1a50dc244f6f62362f77f691d26c.jpg](../iclr_results/3497_Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Design/images/4ce970d4c1b2d51ca49c603632ed7b321dfb1a50dc244f6f62362f77f691d26c.jpg)

![8098dc4e108838c3d334ad3f6ec588a379ae47b71e28a966e02e01d54350e2fa.jpg](../iclr_results/3497_Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Design/images/8098dc4e108838c3d334ad3f6ec588a379ae47b71e28a966e02e01d54350e2fa.jpg)

![85a8c33485a666577f468eae483b48f122c454ca1bd7d5f03adbb733576bdc5b.jpg](../iclr_results/3497_Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Design/images/85a8c33485a666577f468eae483b48f122c454ca1bd7d5f03adbb733576bdc5b.jpg)

![89a8cda8dfff51d344a76723f7eb45207885a3557d38d1a0a14a7f4cdc7ef2da.jpg](../iclr_results/3497_Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Design/images/89a8cda8dfff51d344a76723f7eb45207885a3557d38d1a0a14a7f4cdc7ef2da.jpg)

![c78067ce62893e90d1fe9471f5254aecddb9808c6d3d012827cb8b6ed59462f9.jpg](../iclr_results/3497_Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Design/images/c78067ce62893e90d1fe9471f5254aecddb9808c6d3d012827cb8b6ed59462f9.jpg)

![e0dd1694b0f9535396c692b5dd86350c007792d8f740ca66b3ae6e91d4c37cb4.jpg](../iclr_results/3497_Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Design/images/e0dd1694b0f9535396c692b5dd86350c007792d8f740ca66b3ae6e91d4c37cb4.jpg)

### Tables

![06818f882e45649d51a39a63f42a666b461cdd12c1ed67514569789a4c9aa560.jpg](../iclr_results/3497_Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Design/tables/06818f882e45649d51a39a63f42a666b461cdd12c1ed67514569789a4c9aa560.jpg)

![0f47f57f546722460671559e1bb4d6bca228c08357dd78cc2c8ba7fcfc283111.jpg](../iclr_results/3497_Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Design/tables/0f47f57f546722460671559e1bb4d6bca228c08357dd78cc2c8ba7fcfc283111.jpg)

![46a5f753dc38bfbe4ff37bb1a745407428d3769fe89779bd9326432141d1f1b7.jpg](../iclr_results/3497_Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Design/tables/46a5f753dc38bfbe4ff37bb1a745407428d3769fe89779bd9326432141d1f1b7.jpg)

![51add1cd3ee0367f16d95e698df7f8efbdcb178ab2fa51ff5367639e54f34915.jpg](../iclr_results/3497_Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Design/tables/51add1cd3ee0367f16d95e698df7f8efbdcb178ab2fa51ff5367639e54f34915.jpg)

![627856503c96b9f96ae48f68f25f95fc449e02145ca9da2c1e3c5dc7c1049e4e.jpg](../iclr_results/3497_Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Design/tables/627856503c96b9f96ae48f68f25f95fc449e02145ca9da2c1e3c5dc7c1049e4e.jpg)

![639d37045b15f79ba2167bdea21f0c2f102c3a3b61a077caa72cb5308251a302.jpg](../iclr_results/3497_Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Design/tables/639d37045b15f79ba2167bdea21f0c2f102c3a3b61a077caa72cb5308251a302.jpg)

![6611d36643f354b01078c31114608ef486e79f4966bf12e090ab99b2c2c0ad67.jpg](../iclr_results/3497_Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Design/tables/6611d36643f354b01078c31114608ef486e79f4966bf12e090ab99b2c2c0ad67.jpg)

![75bb5c055a5860d7fca7b6f6535a16fcb8001fd0c226b0f99c1471bde8b26afc.jpg](../iclr_results/3497_Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Design/tables/75bb5c055a5860d7fca7b6f6535a16fcb8001fd0c226b0f99c1471bde8b26afc.jpg)

![952efdc7a3268c8ab54cf542d8f040e86c83513d472f32d894291043d710f2ab.jpg](../iclr_results/3497_Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Design/tables/952efdc7a3268c8ab54cf542d8f040e86c83513d472f32d894291043d710f2ab.jpg)

![acadd6f0c2397af413e20ac95b6c0723238b0e931de2ce487980e6b246e1dd05.jpg](../iclr_results/3497_Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Design/tables/acadd6f0c2397af413e20ac95b6c0723238b0e931de2ce487980e6b246e1dd05.jpg)

![b4c74bb0ded07d8e560b9ba23124ddca23cf1948578d0bb93623f5f3abf02239.jpg](../iclr_results/3497_Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Design/tables/b4c74bb0ded07d8e560b9ba23124ddca23cf1948578d0bb93623f5f3abf02239.jpg)

![f55d3df01f1ee928c61df26c1f9d0976a7ad5d8dbc518957f489773b2dcf0800.jpg](../iclr_results/3497_Structuring Benchmark into Knowledge Graphs to Assist Large Language Models in Retrieving and Design/tables/f55d3df01f1ee928c61df26c1f9d0976a7ad5d8dbc518957f489773b2dcf0800.jpg)

## Single-agent Poisoning Attacks Suffice to Ruin Multi-Agent Learning


### Images

![9986e2c139922c7492c5712c3f361aaa61ea20d2be302b695916465921826776.jpg](../iclr_results/3498_Single-agent Poisoning Attacks Suffice to Ruin Multi-Agent Learning/images/9986e2c139922c7492c5712c3f361aaa61ea20d2be302b695916465921826776.jpg)

![9fd777a24e2d7515267b10279b5d1cf08c67f309548e1e606a67277036a61b8c.jpg](../iclr_results/3498_Single-agent Poisoning Attacks Suffice to Ruin Multi-Agent Learning/images/9fd777a24e2d7515267b10279b5d1cf08c67f309548e1e606a67277036a61b8c.jpg)

![a9a9fa7317a0229a69652c5c59786921ff8c0ddc8efcd6f84130f3d12184ea3f.jpg](../iclr_results/3498_Single-agent Poisoning Attacks Suffice to Ruin Multi-Agent Learning/images/a9a9fa7317a0229a69652c5c59786921ff8c0ddc8efcd6f84130f3d12184ea3f.jpg)

![b05e42edd32811ede1cd243a2c093fe34348b4a3c87f3dbb4e32ed1678f2c226.jpg](../iclr_results/3498_Single-agent Poisoning Attacks Suffice to Ruin Multi-Agent Learning/images/b05e42edd32811ede1cd243a2c093fe34348b4a3c87f3dbb4e32ed1678f2c226.jpg)

![bfd42be960f4f95f61e57ba8fb0cd06c41bd7da8da3d257810210179696b3b09.jpg](../iclr_results/3498_Single-agent Poisoning Attacks Suffice to Ruin Multi-Agent Learning/images/bfd42be960f4f95f61e57ba8fb0cd06c41bd7da8da3d257810210179696b3b09.jpg)

![e31d0073b05a035be896d9d03d37140b96fe872ddb29858f48c6467d135749b4.jpg](../iclr_results/3498_Single-agent Poisoning Attacks Suffice to Ruin Multi-Agent Learning/images/e31d0073b05a035be896d9d03d37140b96fe872ddb29858f48c6467d135749b4.jpg)

![ea258e73fb0ca408374a0241f962809df5de7249bf1c7b30515c90cfea4fd856.jpg](../iclr_results/3498_Single-agent Poisoning Attacks Suffice to Ruin Multi-Agent Learning/images/ea258e73fb0ca408374a0241f962809df5de7249bf1c7b30515c90cfea4fd856.jpg)

### Tables

![5f3be6a0ed2698f8e03826ecbd9493f27e14caf23280f350de3f234e5bdea792.jpg](../iclr_results/3498_Single-agent Poisoning Attacks Suffice to Ruin Multi-Agent Learning/tables/5f3be6a0ed2698f8e03826ecbd9493f27e14caf23280f350de3f234e5bdea792.jpg)

## Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models


### Images

![20230bbafc4b533741fce814b7999a39cf357e8dca1e952876f2d31f47d0a092.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/images/20230bbafc4b533741fce814b7999a39cf357e8dca1e952876f2d31f47d0a092.jpg)

![3d902d3341ec29b4ad9cdcaf2553b0f14072547a66a27e67217cb6746c8f9837.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/images/3d902d3341ec29b4ad9cdcaf2553b0f14072547a66a27e67217cb6746c8f9837.jpg)

![4780e619d1255585c9d380eb25507e455f3cccd3f75c7f39392ae1447f83f8db.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/images/4780e619d1255585c9d380eb25507e455f3cccd3f75c7f39392ae1447f83f8db.jpg)

![538a1c21e2670beefc759cd5fd297c6c11a1b587f10d188456b4a723a86e36f2.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/images/538a1c21e2670beefc759cd5fd297c6c11a1b587f10d188456b4a723a86e36f2.jpg)

![5f17673e7b81d485daff0964a7af116b5383b4418948735940fe50599a4f4afe.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/images/5f17673e7b81d485daff0964a7af116b5383b4418948735940fe50599a4f4afe.jpg)

![8ebf9203dd11b98dd8900e37741caf6e760c486400794b09c8d786951301ddb4.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/images/8ebf9203dd11b98dd8900e37741caf6e760c486400794b09c8d786951301ddb4.jpg)

![8f6220f31c0100a9e74bf4237e3a5e56cb1db8dbc8cfc2de0debe69c77c3de17.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/images/8f6220f31c0100a9e74bf4237e3a5e56cb1db8dbc8cfc2de0debe69c77c3de17.jpg)

![9289c489901f31b6be8c16be0e1bb04a3a9437cf8d678a0736b1b4f6427935b9.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/images/9289c489901f31b6be8c16be0e1bb04a3a9437cf8d678a0736b1b4f6427935b9.jpg)

![b466fa14e9f39d11eaa31936affbf5e286ec952f189f720dc6c056dd9eb4924b.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/images/b466fa14e9f39d11eaa31936affbf5e286ec952f189f720dc6c056dd9eb4924b.jpg)

![d7f35a10e3a422c8121e689fe3920bcb589918c365b07d3bfc42d27c971b8eec.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/images/d7f35a10e3a422c8121e689fe3920bcb589918c365b07d3bfc42d27c971b8eec.jpg)

![f7e638c585c83f6e5270ef325a2e0be98b767dd521122f9ba225554113d8e897.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/images/f7e638c585c83f6e5270ef325a2e0be98b767dd521122f9ba225554113d8e897.jpg)

![f811f1171d345286a4ea0d0b6aa9340f929a028453dfe8ae02501a438ee5e7c4.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/images/f811f1171d345286a4ea0d0b6aa9340f929a028453dfe8ae02501a438ee5e7c4.jpg)

![fb3a3907411cb692325ad4f1a74668169499e2a8d29ed094989babdcab0c5b19.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/images/fb3a3907411cb692325ad4f1a74668169499e2a8d29ed094989babdcab0c5b19.jpg)

![fcbe82fb169e967472e51cb9e7cb91ca44e6d74035543e2e638139ac90bab162.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/images/fcbe82fb169e967472e51cb9e7cb91ca44e6d74035543e2e638139ac90bab162.jpg)

### Tables

![08c66e6f5d742dd028b731ecfc6308c7fb9e7baa0d4f1061f06d2f29156f9711.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/tables/08c66e6f5d742dd028b731ecfc6308c7fb9e7baa0d4f1061f06d2f29156f9711.jpg)

![0be331a10a049118fed09edbb17d0cfbfd8fb1f29e81114ff66cf88af07dae62.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/tables/0be331a10a049118fed09edbb17d0cfbfd8fb1f29e81114ff66cf88af07dae62.jpg)

![0dc792a366de790d3b13eeeb58939bd118d0283036c2ab88945d045ab32ac8ec.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/tables/0dc792a366de790d3b13eeeb58939bd118d0283036c2ab88945d045ab32ac8ec.jpg)

![16765fcf7d7a84ff2347436dae95aa6237a6aff77ea6aae88f2f98e6d7effeac.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/tables/16765fcf7d7a84ff2347436dae95aa6237a6aff77ea6aae88f2f98e6d7effeac.jpg)

![4416c93fcbbf5618789534a5c5f990cf24e582fe16890ba2570d1922075b5d0c.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/tables/4416c93fcbbf5618789534a5c5f990cf24e582fe16890ba2570d1922075b5d0c.jpg)

![46bf6d782b8c3978b5f265d9f89621dff8ad2024c9e88288421b0b6affe962cc.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/tables/46bf6d782b8c3978b5f265d9f89621dff8ad2024c9e88288421b0b6affe962cc.jpg)

![8aac6f30dd1ff2053d16cf67e7b527fe36813cdcaab79f65192e7737a3e9b928.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/tables/8aac6f30dd1ff2053d16cf67e7b527fe36813cdcaab79f65192e7737a3e9b928.jpg)

![a18c3bf7fecf20786f10d7a7212bf6550675d6d55f868ccab06325b256aa787b.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/tables/a18c3bf7fecf20786f10d7a7212bf6550675d6d55f868ccab06325b256aa787b.jpg)

![b7383629e25ccd9c1360dea4e7859b5b5fe21d78451a1f43c709e9b387050168.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/tables/b7383629e25ccd9c1360dea4e7859b5b5fe21d78451a1f43c709e9b387050168.jpg)

![d046051e0c25fd7b840dfe834e6bf38c8d46c5306add913db7a1fe6508abceac.jpg](../iclr_results/3499_Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models/tables/d046051e0c25fd7b840dfe834e6bf38c8d46c5306add913db7a1fe6508abceac.jpg)

## Instruct-SkillMix: A Powerful Pipeline for LLM Instruction Tuning


### Images

![0d4e0b6ecf94e37e119452d29bf5210abcd65f686175370037e1d2d32e0c13c8.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/images/0d4e0b6ecf94e37e119452d29bf5210abcd65f686175370037e1d2d32e0c13c8.jpg)

![6a3830f42fad0bb9bf0b5b261eb9d142f486c4d30b3deb6c77a72ac2dba1ab0b.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/images/6a3830f42fad0bb9bf0b5b261eb9d142f486c4d30b3deb6c77a72ac2dba1ab0b.jpg)

![a4ae45d6a11c7432e5f55d2ffe7329a3eed7a166b16cfafa6b6bc8219e6d9f28.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/images/a4ae45d6a11c7432e5f55d2ffe7329a3eed7a166b16cfafa6b6bc8219e6d9f28.jpg)

![c75efe74739b26079493689398c194bea7f69ae301ce97c1916615931962bda5.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/images/c75efe74739b26079493689398c194bea7f69ae301ce97c1916615931962bda5.jpg)

![cb4165190f7485414fbccb8355eccf3b1f1594fdfbfff799e611d3619977f6a4.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/images/cb4165190f7485414fbccb8355eccf3b1f1594fdfbfff799e611d3619977f6a4.jpg)

### Tables

![00804fc09ccac30271c3dcb2a81a83827f9604a762f8d74e0b055e3daedeaa11.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/00804fc09ccac30271c3dcb2a81a83827f9604a762f8d74e0b055e3daedeaa11.jpg)

![0257eb83ba4356a3e9f788eff2ec344f0f646ba24288ba88d219fd11972d5c83.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/0257eb83ba4356a3e9f788eff2ec344f0f646ba24288ba88d219fd11972d5c83.jpg)

![032e5c9ebb7743840865f9cb78b00e20d554a743697839dd948bfabf0341e18e.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/032e5c9ebb7743840865f9cb78b00e20d554a743697839dd948bfabf0341e18e.jpg)

![094049eebc369e9c140de6ae0e9dd6969abdb599451449d08a7629692a880d07.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/094049eebc369e9c140de6ae0e9dd6969abdb599451449d08a7629692a880d07.jpg)

![096ae237e70cb1105a32fdf854cc37e0a934ba93e044b89d7ed449cc119f788a.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/096ae237e70cb1105a32fdf854cc37e0a934ba93e044b89d7ed449cc119f788a.jpg)

![0bece443573edbefec0cef6d49a0b76c0ff903aabbab2694c571ca461ccb1bc4.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/0bece443573edbefec0cef6d49a0b76c0ff903aabbab2694c571ca461ccb1bc4.jpg)

![0c0e8eadc8027bbf5b25dc5f066423dc5d24000ab2a8aa700ac30022a9379e18.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/0c0e8eadc8027bbf5b25dc5f066423dc5d24000ab2a8aa700ac30022a9379e18.jpg)

![0f6298821a2347ca50d0cdff04d5417aec4813ef95366059e8a2128091f1994c.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/0f6298821a2347ca50d0cdff04d5417aec4813ef95366059e8a2128091f1994c.jpg)

![123e12749f7eb7632ba4feedef717f309c9ae98dc041645daa8c963abdadcf15.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/123e12749f7eb7632ba4feedef717f309c9ae98dc041645daa8c963abdadcf15.jpg)

![1aa6c4043794d263cf5e479db35aab215d35752adf1ccbac366ef07341657ef3.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/1aa6c4043794d263cf5e479db35aab215d35752adf1ccbac366ef07341657ef3.jpg)

![1ab194668fdceb0b577978a6e1e8ce320819a2e9511b2cd1d53d75317e7148d9.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/1ab194668fdceb0b577978a6e1e8ce320819a2e9511b2cd1d53d75317e7148d9.jpg)

![1bdbb59d763342f67a6179f5ccb90f57bf1c11c7fa00419e39c44151044cdfed.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/1bdbb59d763342f67a6179f5ccb90f57bf1c11c7fa00419e39c44151044cdfed.jpg)

![23d4c85cf8a7296b164be44303c2741d4af36c9ca7c0471c0596264076d28ce5.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/23d4c85cf8a7296b164be44303c2741d4af36c9ca7c0471c0596264076d28ce5.jpg)

![252945e162fd154b13094988db8f53a1e8d0f82e7982e4902a9e4d0e3b7c059b.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/252945e162fd154b13094988db8f53a1e8d0f82e7982e4902a9e4d0e3b7c059b.jpg)

![26653a6ac4908b9f0c3720703d5f9e1ece63d9e6d5c58ae89a3fd2ee06fdac0f.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/26653a6ac4908b9f0c3720703d5f9e1ece63d9e6d5c58ae89a3fd2ee06fdac0f.jpg)

![295fd3917c63e746bf642af69aa3904ee01076a7ddbcc76af4fb6a91b4e90cc9.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/295fd3917c63e746bf642af69aa3904ee01076a7ddbcc76af4fb6a91b4e90cc9.jpg)

![31b3a1084fd4a7228a081e23c418e8c39c83dc15ee0e883bc40f11656e3487b6.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/31b3a1084fd4a7228a081e23c418e8c39c83dc15ee0e883bc40f11656e3487b6.jpg)

![32a6fedccfe0500746cb64f36ded17aeb3feca39ab52d91b5cc7f4ac0776a261.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/32a6fedccfe0500746cb64f36ded17aeb3feca39ab52d91b5cc7f4ac0776a261.jpg)

![332874cefe83fbc763c5b4f4d785c8926b890ff27efbd49501c37962292a2eef.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/332874cefe83fbc763c5b4f4d785c8926b890ff27efbd49501c37962292a2eef.jpg)

![33d60707f40a6c27dc8fe3fa16b673b46c6ef9965ffa53e0bb72336fe1de8f1b.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/33d60707f40a6c27dc8fe3fa16b673b46c6ef9965ffa53e0bb72336fe1de8f1b.jpg)

![366c312e5e076a8ff971089e9061586bfefc794b38f218cf7bd1e8b089f3f899.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/366c312e5e076a8ff971089e9061586bfefc794b38f218cf7bd1e8b089f3f899.jpg)

![39e66f2a98f22cc64e7105c60871d96ebc578621d92d8eab1c5a524af4d84db3.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/39e66f2a98f22cc64e7105c60871d96ebc578621d92d8eab1c5a524af4d84db3.jpg)

![3bd24f48178647d495442fa27ac37453c924aa296b2bc39d5922894525ac7435.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/3bd24f48178647d495442fa27ac37453c924aa296b2bc39d5922894525ac7435.jpg)

![3c0d6c5a34ba6f5a8aa8ea7836ca8557cfdf6649e3d83dc704fb69c863c70319.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/3c0d6c5a34ba6f5a8aa8ea7836ca8557cfdf6649e3d83dc704fb69c863c70319.jpg)

![3d8201dd7e104538cd71690b173ba9fea329239dd22d33bf8fb6415e939a2d8a.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/3d8201dd7e104538cd71690b173ba9fea329239dd22d33bf8fb6415e939a2d8a.jpg)

![3edfa2c8e8e5df012aba8843f8f814e89993ba215b8ea8b048f5ec942a1001e5.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/3edfa2c8e8e5df012aba8843f8f814e89993ba215b8ea8b048f5ec942a1001e5.jpg)

![3fda21f9e464f49e776957db52b12cf22c83dbaf80a5efa5825c6f744481a55f.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/3fda21f9e464f49e776957db52b12cf22c83dbaf80a5efa5825c6f744481a55f.jpg)

![44ae21eca71df7c5c2a565599cdcb63f21a2a7205890f6cea068c7b6e9191af8.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/44ae21eca71df7c5c2a565599cdcb63f21a2a7205890f6cea068c7b6e9191af8.jpg)

![47e33a28f7fcc8241aa98d7eef5f525db2c476f8a25b3ceab0fb20775b303d78.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/47e33a28f7fcc8241aa98d7eef5f525db2c476f8a25b3ceab0fb20775b303d78.jpg)

![49b739a47eaee89eb68a3be2da4883d89075325838436eefdc9840740c7d94a7.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/49b739a47eaee89eb68a3be2da4883d89075325838436eefdc9840740c7d94a7.jpg)

![4bb323ca4f8e48008b55231cdaa796e62c3b28d4b6ec0d83f2b6ba93c25add27.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/4bb323ca4f8e48008b55231cdaa796e62c3b28d4b6ec0d83f2b6ba93c25add27.jpg)

![4be43535ad1d47fb639cce2faa908e051d3af23cf1ee4a9aa6a08d9f403fb63b.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/4be43535ad1d47fb639cce2faa908e051d3af23cf1ee4a9aa6a08d9f403fb63b.jpg)

![4bf0e5b67f968cb48da43799b55f129432d4e7587168134f692eb046b8d6fe08.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/4bf0e5b67f968cb48da43799b55f129432d4e7587168134f692eb046b8d6fe08.jpg)

![4ddd8dc1ce4e24cdd8a74eb02c1911d70eca613bf74314355febd19a6d98a1d5.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/4ddd8dc1ce4e24cdd8a74eb02c1911d70eca613bf74314355febd19a6d98a1d5.jpg)

![4e553efd49ecdacbf93e05f14a9c2f6677f4c17fd383f59d0dda33738a18fbaa.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/4e553efd49ecdacbf93e05f14a9c2f6677f4c17fd383f59d0dda33738a18fbaa.jpg)

![4fa8d32721f3e1a1eb3899a6e297a068f5af095a7b306f1560eea649ade9f259.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/4fa8d32721f3e1a1eb3899a6e297a068f5af095a7b306f1560eea649ade9f259.jpg)

![50c20e52290792a8f2c8bc44ca0eec77ddfa6493fc0cec88de958704f8111ca1.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/50c20e52290792a8f2c8bc44ca0eec77ddfa6493fc0cec88de958704f8111ca1.jpg)

![56800f1842ffd26d063ec3a856a5eb4cfa939c205df4fabe6bb1920fc3a10495.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/56800f1842ffd26d063ec3a856a5eb4cfa939c205df4fabe6bb1920fc3a10495.jpg)

![5877169bff10bbcd52b0be28d2f42b7a9cb9b34de9b906636b6f9bcd833fa4b6.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/5877169bff10bbcd52b0be28d2f42b7a9cb9b34de9b906636b6f9bcd833fa4b6.jpg)

![5a5288e71526c1e32c549c6f565d37e703b453574cc947dec069a8f5645de838.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/5a5288e71526c1e32c549c6f565d37e703b453574cc947dec069a8f5645de838.jpg)

![5baeca956686742d6b1ce93128246b11e0ee21057bdc3e75f52cc928f4d82da8.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/5baeca956686742d6b1ce93128246b11e0ee21057bdc3e75f52cc928f4d82da8.jpg)

![5bf099b0d34bc2af364b52727ab9f0a0d0720cab3afdf90f8c2e25f9f24f2393.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/5bf099b0d34bc2af364b52727ab9f0a0d0720cab3afdf90f8c2e25f9f24f2393.jpg)

![618741a83b03384f8f3f88db6899efb4e6ec1f59310a9c839b8649dc744f96e5.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/618741a83b03384f8f3f88db6899efb4e6ec1f59310a9c839b8649dc744f96e5.jpg)

![67d82e4d47b75b1221fe5dc8c44577aa70857d852263d876adbbb29da98f8282.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/67d82e4d47b75b1221fe5dc8c44577aa70857d852263d876adbbb29da98f8282.jpg)

![67f4ba528751f4d30ce2083b0a2e0f4cf8581f733ff63517c6a72cd1ec09c7c6.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/67f4ba528751f4d30ce2083b0a2e0f4cf8581f733ff63517c6a72cd1ec09c7c6.jpg)

![6c9f39f43c62a0261e20eb26c9025dc285e82baed01c01cadb6a61acc9af3322.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/6c9f39f43c62a0261e20eb26c9025dc285e82baed01c01cadb6a61acc9af3322.jpg)

![6cbec9de125791933e17d008019d2facda28ddfca72536ee956769641b19a46f.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/6cbec9de125791933e17d008019d2facda28ddfca72536ee956769641b19a46f.jpg)

![6d324d021b30af3c72d7705bcc4364edad5ce1e0f61a3bad9453071909cf46f6.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/6d324d021b30af3c72d7705bcc4364edad5ce1e0f61a3bad9453071909cf46f6.jpg)

![6feaa6ee7c3d475562e947589b56ae3a11d1f950f6cb63d2ff95bd24ca49805e.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/6feaa6ee7c3d475562e947589b56ae3a11d1f950f6cb63d2ff95bd24ca49805e.jpg)

![70a781ba9c8e687e787a03b0e1e4da74e3b05995d9aead27e0802f6a89da4969.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/70a781ba9c8e687e787a03b0e1e4da74e3b05995d9aead27e0802f6a89da4969.jpg)

![70eeee5eed4ed629f1200351cb6b026ba61f79e8d66280db72ee091177cd8079.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/70eeee5eed4ed629f1200351cb6b026ba61f79e8d66280db72ee091177cd8079.jpg)

![74d49e32796088d1cf157e4ba859acab09f988966e009537b0c8adc854203e9c.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/74d49e32796088d1cf157e4ba859acab09f988966e009537b0c8adc854203e9c.jpg)

![79e6d10068182d19e497cd0ed7e7d0173fc7da4cffd3dcea76562e4ac54f8c6d.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/79e6d10068182d19e497cd0ed7e7d0173fc7da4cffd3dcea76562e4ac54f8c6d.jpg)

![7aca06143d88d392185cad434a20660d85ac415caf932c442153d5ee16620577.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/7aca06143d88d392185cad434a20660d85ac415caf932c442153d5ee16620577.jpg)

![7acb0c9e232cbfca90c90d8c11c83a09a3d845065fc6e13eff0ffc13a93ba592.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/7acb0c9e232cbfca90c90d8c11c83a09a3d845065fc6e13eff0ffc13a93ba592.jpg)

![82ec90df0e849e822c4165e173d33c266af3efb7a8d549ea2ffc282fc0c3163c.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/82ec90df0e849e822c4165e173d33c266af3efb7a8d549ea2ffc282fc0c3163c.jpg)

![89a3ca055a52a68e7fee99eec39956ac156ce415f0d803d255267c6fb7939e06.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/89a3ca055a52a68e7fee99eec39956ac156ce415f0d803d255267c6fb7939e06.jpg)

![8a5f9bfd55493518027769986b868a6af5de75062cf1f95eef2a86c4563a6e78.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/8a5f9bfd55493518027769986b868a6af5de75062cf1f95eef2a86c4563a6e78.jpg)

![8ac8d3ece95ed9a530ce6bd8ab322eb55bb29d3474e15e5ce845d52a9b761afa.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/8ac8d3ece95ed9a530ce6bd8ab322eb55bb29d3474e15e5ce845d52a9b761afa.jpg)

![8cd6f8f67dfee885ffbce71636cb4b5941b045f48451e45f0410d80658061a81.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/8cd6f8f67dfee885ffbce71636cb4b5941b045f48451e45f0410d80658061a81.jpg)

![8fb512e59b0afdf5dab2e9a0c39d1ae5cc68a810b49fc6946a803db1010d0f55.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/8fb512e59b0afdf5dab2e9a0c39d1ae5cc68a810b49fc6946a803db1010d0f55.jpg)

![9032be33ede27c4ec60e15c667d73e073b115e9ba139992c89a7c044caa1b861.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/9032be33ede27c4ec60e15c667d73e073b115e9ba139992c89a7c044caa1b861.jpg)

![95509234d2dcd67eaa3f8a5221ca57af2ee53f98e444a6dc1418976dfef1a0e2.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/95509234d2dcd67eaa3f8a5221ca57af2ee53f98e444a6dc1418976dfef1a0e2.jpg)

![9589d658eb8b6e81c91a1acb4844ac81874cfa366825bbfae47ef33bceea3ad3.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/9589d658eb8b6e81c91a1acb4844ac81874cfa366825bbfae47ef33bceea3ad3.jpg)

![9604f683e5d435f41804ae82e49a9e83ee7fbc18063835e0219b2f3476bfedb1.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/9604f683e5d435f41804ae82e49a9e83ee7fbc18063835e0219b2f3476bfedb1.jpg)

![966255311033c11c623ce18ca6206121da445b68ac31e95bff93c5d28c33bff0.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/966255311033c11c623ce18ca6206121da445b68ac31e95bff93c5d28c33bff0.jpg)

![9cbae69361ae4ce65bae231bc56e692ed32c256082fa56374fd7da01645f1c51.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/9cbae69361ae4ce65bae231bc56e692ed32c256082fa56374fd7da01645f1c51.jpg)

![9cc64b0546d525233cac20d49c801d5ccf33f98c2ba748209fe67749192ae895.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/9cc64b0546d525233cac20d49c801d5ccf33f98c2ba748209fe67749192ae895.jpg)

![9eb5dea2e191c45f435dd9f1e514a5ff08cd95e8cda492fc8b864c549251906f.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/9eb5dea2e191c45f435dd9f1e514a5ff08cd95e8cda492fc8b864c549251906f.jpg)

![a0b479155a26687d9a1ad99ac102c47f42f86ea1c34487251303bd5f11f4432a.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/a0b479155a26687d9a1ad99ac102c47f42f86ea1c34487251303bd5f11f4432a.jpg)

![a147c38ab8bd925cba3ae37b09606c76e1eda1f3ad52df1b88231bce46809273.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/a147c38ab8bd925cba3ae37b09606c76e1eda1f3ad52df1b88231bce46809273.jpg)

![ac2de2cdb8d9853662256ae9bb33f7e56d5e94ab210f4b384f588e5fb187e0fb.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/ac2de2cdb8d9853662256ae9bb33f7e56d5e94ab210f4b384f588e5fb187e0fb.jpg)

![aec9d21f93d2f340c71f26b0f507572e7eea50f8570496bf7555821c78287d71.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/aec9d21f93d2f340c71f26b0f507572e7eea50f8570496bf7555821c78287d71.jpg)

![af19af886654698b0894b3e541c3802603184f4fa1808be99938ea24c788ed6d.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/af19af886654698b0894b3e541c3802603184f4fa1808be99938ea24c788ed6d.jpg)

![b32f05bad63b8957244fc463936b23c93864c7ae2bb5029d137cf8b9e981c76b.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/b32f05bad63b8957244fc463936b23c93864c7ae2bb5029d137cf8b9e981c76b.jpg)

![b3a568787ee413cbf35aed0dd9257d6d57fac1861e8c5282310700094401debb.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/b3a568787ee413cbf35aed0dd9257d6d57fac1861e8c5282310700094401debb.jpg)

![c03f89408770e6d8000f8a715382a343c192a17360bf59d0a1ec1cec622ca2e9.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/c03f89408770e6d8000f8a715382a343c192a17360bf59d0a1ec1cec622ca2e9.jpg)

![c12e72f9dc84ad6023e24bfb578982e39ea45b8d1b12531b48721f821f935791.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/c12e72f9dc84ad6023e24bfb578982e39ea45b8d1b12531b48721f821f935791.jpg)

![c3ccc75a74ace740b226e5204624f14e45665c1c7c0e2e869ca6d8fb9d8de441.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/c3ccc75a74ace740b226e5204624f14e45665c1c7c0e2e869ca6d8fb9d8de441.jpg)

![c4b8a5a63fbe9409010de94210777b6e100f7b4c21cea6e05d12686deda086fb.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/c4b8a5a63fbe9409010de94210777b6e100f7b4c21cea6e05d12686deda086fb.jpg)

![c4c2cbe34603ebaa211fd6a4ee04ba45207536f12c076dfc512848bb8ab15629.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/c4c2cbe34603ebaa211fd6a4ee04ba45207536f12c076dfc512848bb8ab15629.jpg)

![c5865820b3c5c10269cd8ffff02a19cf56613d6b629d0e463f75acb81c6c0ed8.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/c5865820b3c5c10269cd8ffff02a19cf56613d6b629d0e463f75acb81c6c0ed8.jpg)

![c73dfbc26bee62bd26e643740852b8725968d39f224e55695f8b19d16d8e1cd8.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/c73dfbc26bee62bd26e643740852b8725968d39f224e55695f8b19d16d8e1cd8.jpg)

![c803da4980721e475f6afd6f72ab794e5e3a677bcb79e4094785e6f61a32ccc2.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/c803da4980721e475f6afd6f72ab794e5e3a677bcb79e4094785e6f61a32ccc2.jpg)

![cb1f00e26288c38caea7a9fbf6a761baadec3af8a0cb5248bed34acf8cf5f5c8.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/cb1f00e26288c38caea7a9fbf6a761baadec3af8a0cb5248bed34acf8cf5f5c8.jpg)

![cb7cddef78f2b1074a7a31b6a1d12af0073feb9f7bbb89b43066bbb2014710b9.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/cb7cddef78f2b1074a7a31b6a1d12af0073feb9f7bbb89b43066bbb2014710b9.jpg)

![cb88d082d266de56056e47b4ea74e8260a2230de054b18cbffc67a34b684c107.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/cb88d082d266de56056e47b4ea74e8260a2230de054b18cbffc67a34b684c107.jpg)

![cc47744f6f5991a5d61ca5807630c6c746664a70c9c3ec8872680e9f6b182d4e.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/cc47744f6f5991a5d61ca5807630c6c746664a70c9c3ec8872680e9f6b182d4e.jpg)

![cc6c2c7db809a790da1efa9cb8af7e75cee2a3859fed41c299044ad6b452bd08.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/cc6c2c7db809a790da1efa9cb8af7e75cee2a3859fed41c299044ad6b452bd08.jpg)

![cee2b72d6129769156514ec2cd2358f28e0e67bd0e05c52b86b6a1bcfa3b710a.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/cee2b72d6129769156514ec2cd2358f28e0e67bd0e05c52b86b6a1bcfa3b710a.jpg)

![cff1bc1bfc64c506f9f17a6d5140035ebf6666edc6c7d3219e84648afab7645b.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/cff1bc1bfc64c506f9f17a6d5140035ebf6666edc6c7d3219e84648afab7645b.jpg)

![d29921952b31ab24822a0733425c8f34d81a33a1a9096f66c5387050471d9f11.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/d29921952b31ab24822a0733425c8f34d81a33a1a9096f66c5387050471d9f11.jpg)

![d37b847c237336a0b6301dd3e769246930285a8a3043b9ac6c165e467c11dc33.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/d37b847c237336a0b6301dd3e769246930285a8a3043b9ac6c165e467c11dc33.jpg)

![d3b2c7cf1391120b1049e8ad8399ffa85c461c6ae7974eb77fe5ed33cac80c7c.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/d3b2c7cf1391120b1049e8ad8399ffa85c461c6ae7974eb77fe5ed33cac80c7c.jpg)

![d538524660b84abd9ebe622446d390a9ba93ee6d3adfb3a12d4253082a1e5c7a.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/d538524660b84abd9ebe622446d390a9ba93ee6d3adfb3a12d4253082a1e5c7a.jpg)

![da74d3f2e0783f5f3623f5533222c41f1465f3232fca012422d356257c5ca658.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/da74d3f2e0783f5f3623f5533222c41f1465f3232fca012422d356257c5ca658.jpg)

![daf9963edbf6201ddda8ca023dde65e8fe8ddde6d836613681d08fbf309bedf3.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/daf9963edbf6201ddda8ca023dde65e8fe8ddde6d836613681d08fbf309bedf3.jpg)

![e137977dd34ab7b4ee4b0252efe9d11cc96135b05559b835200d8c8ac31901ef.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/e137977dd34ab7b4ee4b0252efe9d11cc96135b05559b835200d8c8ac31901ef.jpg)

![e3379085d7bbab16a21e6538a7441432bd15171416013969eedcc51ecf027be3.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/e3379085d7bbab16a21e6538a7441432bd15171416013969eedcc51ecf027be3.jpg)

![e589300472e9a69ed708f696dba658efbc1463935db40bd9d6727d2892d8b3be.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/e589300472e9a69ed708f696dba658efbc1463935db40bd9d6727d2892d8b3be.jpg)

![ea312f10c2bfa2c5dd63eb045dc0d98cbf611315001b75c74ffc1caff7a05914.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/ea312f10c2bfa2c5dd63eb045dc0d98cbf611315001b75c74ffc1caff7a05914.jpg)

![eca73e7d3b63b644148551749895e0b2927ae5b39b848e234129955ca07cd074.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/eca73e7d3b63b644148551749895e0b2927ae5b39b848e234129955ca07cd074.jpg)

![eda9748b53c16bfd40dc80e5d397d1638c4ec7e206adc0f7666a7d8bb6b3f147.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/eda9748b53c16bfd40dc80e5d397d1638c4ec7e206adc0f7666a7d8bb6b3f147.jpg)

![f1ff2988e449666bfb0899a2e443121a4c7a0beeef647343a31e1717bbc99cb7.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/f1ff2988e449666bfb0899a2e443121a4c7a0beeef647343a31e1717bbc99cb7.jpg)

![f568bb802bd33bc8cbd5505dc6268b48b3c46b99e13c3b4f2f49c8dc1b6edf20.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/f568bb802bd33bc8cbd5505dc6268b48b3c46b99e13c3b4f2f49c8dc1b6edf20.jpg)

![f632f2fb8d9337160e0f8e30399bd8aaff527da78ad5df070b02bdc2e286dd09.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/f632f2fb8d9337160e0f8e30399bd8aaff527da78ad5df070b02bdc2e286dd09.jpg)

![facaea0052ade6f49d2b850bb759c6f7af4e75027b0109942d34a04caeadfca7.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/facaea0052ade6f49d2b850bb759c6f7af4e75027b0109942d34a04caeadfca7.jpg)

![fb19ba1dd939b927547cd5ccb4b8057a476d2ff6d853963bb4819179295317c7.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/fb19ba1dd939b927547cd5ccb4b8057a476d2ff6d853963bb4819179295317c7.jpg)

![fe836fad87585b77b59c53171b1395c8cbd0cfbb1c1e53475948399fcfb27c57.jpg](../iclr_results/3500_Instruct-SkillMix_ A Powerful Pipeline for LLM Instruction Tuning/tables/fe836fad87585b77b59c53171b1395c8cbd0cfbb1c1e53475948399fcfb27c57.jpg)

## FedTMOS: Efficient One-Shot Federated Learning with Tsetlin Machine


### Images

![17f2bc31c9f465a99c8fe548a7c0f0039e5e095aa149d9a4dc97e103482a820e.jpg](../iclr_results/3501_FedTMOS_ Efficient One-Shot Federated Learning with Tsetlin Machine/images/17f2bc31c9f465a99c8fe548a7c0f0039e5e095aa149d9a4dc97e103482a820e.jpg)

![74628fa6a1b335a0548c5aced394ecb0596523bd886907eefea6f91d0d23cfb3.jpg](../iclr_results/3501_FedTMOS_ Efficient One-Shot Federated Learning with Tsetlin Machine/images/74628fa6a1b335a0548c5aced394ecb0596523bd886907eefea6f91d0d23cfb3.jpg)

![ae7086b16d2b07e161b2a9930d8086c112762e83b6a8c0fda2e210eb17d3427e.jpg](../iclr_results/3501_FedTMOS_ Efficient One-Shot Federated Learning with Tsetlin Machine/images/ae7086b16d2b07e161b2a9930d8086c112762e83b6a8c0fda2e210eb17d3427e.jpg)

![d03447c4b7e6b983353d41419e7d93bda89d5f5eb671abbb4f92766d82ad2a17.jpg](../iclr_results/3501_FedTMOS_ Efficient One-Shot Federated Learning with Tsetlin Machine/images/d03447c4b7e6b983353d41419e7d93bda89d5f5eb671abbb4f92766d82ad2a17.jpg)

### Tables

![025cbf963cf1d7ca9167d46780ca71a7e87681d60abfbcc8f745ade115c0f32e.jpg](../iclr_results/3501_FedTMOS_ Efficient One-Shot Federated Learning with Tsetlin Machine/tables/025cbf963cf1d7ca9167d46780ca71a7e87681d60abfbcc8f745ade115c0f32e.jpg)

![250374c224a4f051f863a397c0dbb68527bf54fe8f650c32330541cedf080253.jpg](../iclr_results/3501_FedTMOS_ Efficient One-Shot Federated Learning with Tsetlin Machine/tables/250374c224a4f051f863a397c0dbb68527bf54fe8f650c32330541cedf080253.jpg)

![2cadc7f3569a88a1b0556318ba2fe0db36984ff1d58579f13b6a0bb8eff295c3.jpg](../iclr_results/3501_FedTMOS_ Efficient One-Shot Federated Learning with Tsetlin Machine/tables/2cadc7f3569a88a1b0556318ba2fe0db36984ff1d58579f13b6a0bb8eff295c3.jpg)

![5e4600e96494fccc7c64532141cad394fde9c488f1f05fd29f0021d482902e55.jpg](../iclr_results/3501_FedTMOS_ Efficient One-Shot Federated Learning with Tsetlin Machine/tables/5e4600e96494fccc7c64532141cad394fde9c488f1f05fd29f0021d482902e55.jpg)

![860408f80f6dd18c50f04c6cc7d6ddb90bd0c53084a9ae32dafa348ae34d39e0.jpg](../iclr_results/3501_FedTMOS_ Efficient One-Shot Federated Learning with Tsetlin Machine/tables/860408f80f6dd18c50f04c6cc7d6ddb90bd0c53084a9ae32dafa348ae34d39e0.jpg)

![8759410731f6c35647c50acfb6fcd6100546f021a26cd75498ec7b4d9dc4a36a.jpg](../iclr_results/3501_FedTMOS_ Efficient One-Shot Federated Learning with Tsetlin Machine/tables/8759410731f6c35647c50acfb6fcd6100546f021a26cd75498ec7b4d9dc4a36a.jpg)

![8b6a54814331f8cf3b2637b7d5e7e9e7499b9fc07513b448b12676996cc9e635.jpg](../iclr_results/3501_FedTMOS_ Efficient One-Shot Federated Learning with Tsetlin Machine/tables/8b6a54814331f8cf3b2637b7d5e7e9e7499b9fc07513b448b12676996cc9e635.jpg)

![a64c3bf35f63cc4901be569e6a6efe58a1f9026e677e5ef7db8a3324b348f4a3.jpg](../iclr_results/3501_FedTMOS_ Efficient One-Shot Federated Learning with Tsetlin Machine/tables/a64c3bf35f63cc4901be569e6a6efe58a1f9026e677e5ef7db8a3324b348f4a3.jpg)

![ab771da3f891580adf96d2cbd027beea9907f123c71428ddad1b0382cf9f1c96.jpg](../iclr_results/3501_FedTMOS_ Efficient One-Shot Federated Learning with Tsetlin Machine/tables/ab771da3f891580adf96d2cbd027beea9907f123c71428ddad1b0382cf9f1c96.jpg)

![b9c2289a66bb65301d3cc69dafbcf500b2b040f5a72719d4ae95e089eee9de0e.jpg](../iclr_results/3501_FedTMOS_ Efficient One-Shot Federated Learning with Tsetlin Machine/tables/b9c2289a66bb65301d3cc69dafbcf500b2b040f5a72719d4ae95e089eee9de0e.jpg)

![c4b8abd3a3b8af33a2786f4c982ebce0d772555e00332b26514c74ef722c6f04.jpg](../iclr_results/3501_FedTMOS_ Efficient One-Shot Federated Learning with Tsetlin Machine/tables/c4b8abd3a3b8af33a2786f4c982ebce0d772555e00332b26514c74ef722c6f04.jpg)

![dd27786d1ee9989bfad35082f4181c375d93f6456030df45b79c0a041fb5d523.jpg](../iclr_results/3501_FedTMOS_ Efficient One-Shot Federated Learning with Tsetlin Machine/tables/dd27786d1ee9989bfad35082f4181c375d93f6456030df45b79c0a041fb5d523.jpg)

## Test of Time: A Benchmark for Evaluating LLMs on Temporal Reasoning


### Images

![3bd9e4529a4d6b94ee06b7c42679526ba94598df09b9a3e05ba8e11ad8e0b3f0.jpg](../iclr_results/3502_Test of Time_ A Benchmark for Evaluating LLMs on Temporal Reasoning/images/3bd9e4529a4d6b94ee06b7c42679526ba94598df09b9a3e05ba8e11ad8e0b3f0.jpg)

![6623fdc6b51f3f3b2feda1e647b4eb24972941d3aa0540a36a1885f807c1fcf8.jpg](../iclr_results/3502_Test of Time_ A Benchmark for Evaluating LLMs on Temporal Reasoning/images/6623fdc6b51f3f3b2feda1e647b4eb24972941d3aa0540a36a1885f807c1fcf8.jpg)

![a7f9bd5132384bac954a9e98647f04675e1632b269804d2ccfa05dcb1deaac68.jpg](../iclr_results/3502_Test of Time_ A Benchmark for Evaluating LLMs on Temporal Reasoning/images/a7f9bd5132384bac954a9e98647f04675e1632b269804d2ccfa05dcb1deaac68.jpg)

![b531de384d7f5ae164a4a06f7a59fd24f6546956899aa46cb933aa8e4852e7cf.jpg](../iclr_results/3502_Test of Time_ A Benchmark for Evaluating LLMs on Temporal Reasoning/images/b531de384d7f5ae164a4a06f7a59fd24f6546956899aa46cb933aa8e4852e7cf.jpg)

![e0cc751b3f4bb863c0969305fb3b2d0c3bc569744321c74c5e33045cfd387842.jpg](../iclr_results/3502_Test of Time_ A Benchmark for Evaluating LLMs on Temporal Reasoning/images/e0cc751b3f4bb863c0969305fb3b2d0c3bc569744321c74c5e33045cfd387842.jpg)

### Tables

![00485f877f0ea19ea45016a75d3b5f588f7765dde1a2dc01155785df125d4784.jpg](../iclr_results/3502_Test of Time_ A Benchmark for Evaluating LLMs on Temporal Reasoning/tables/00485f877f0ea19ea45016a75d3b5f588f7765dde1a2dc01155785df125d4784.jpg)

![2cacc35404bb08a275215de0ee62d8def0fc6872a61276e3238ead76e485660b.jpg](../iclr_results/3502_Test of Time_ A Benchmark for Evaluating LLMs on Temporal Reasoning/tables/2cacc35404bb08a275215de0ee62d8def0fc6872a61276e3238ead76e485660b.jpg)

![59c29f63cca1a9fc043124e8e6c7b83ede00a0e3b46d27454e51353bf7906e97.jpg](../iclr_results/3502_Test of Time_ A Benchmark for Evaluating LLMs on Temporal Reasoning/tables/59c29f63cca1a9fc043124e8e6c7b83ede00a0e3b46d27454e51353bf7906e97.jpg)

![6bd9d622b723584fe9b94e9abe222ccb519dfe772384aa46fa18fabe1b14b046.jpg](../iclr_results/3502_Test of Time_ A Benchmark for Evaluating LLMs on Temporal Reasoning/tables/6bd9d622b723584fe9b94e9abe222ccb519dfe772384aa46fa18fabe1b14b046.jpg)

![73e577999d2cb92e96c5c5530ff5fc606c0adb7528242ec5d3cb15ecb298ef4f.jpg](../iclr_results/3502_Test of Time_ A Benchmark for Evaluating LLMs on Temporal Reasoning/tables/73e577999d2cb92e96c5c5530ff5fc606c0adb7528242ec5d3cb15ecb298ef4f.jpg)

![9365fe9f32c1bc357ec89e957326299f6b1a0c230e67d01929d4192e8580790b.jpg](../iclr_results/3502_Test of Time_ A Benchmark for Evaluating LLMs on Temporal Reasoning/tables/9365fe9f32c1bc357ec89e957326299f6b1a0c230e67d01929d4192e8580790b.jpg)

![98f169dbdacdf9dcca1c86a9d7da4fea943eb74f6a777b3499f3faa75071509b.jpg](../iclr_results/3502_Test of Time_ A Benchmark for Evaluating LLMs on Temporal Reasoning/tables/98f169dbdacdf9dcca1c86a9d7da4fea943eb74f6a777b3499f3faa75071509b.jpg)

![b30ef84aaf187c52cfa1c9b5c1513382e872a96aeea1b300ece2cedd1224a81c.jpg](../iclr_results/3502_Test of Time_ A Benchmark for Evaluating LLMs on Temporal Reasoning/tables/b30ef84aaf187c52cfa1c9b5c1513382e872a96aeea1b300ece2cedd1224a81c.jpg)

![cd5bcef96fe70ef5c59aba45d5fb6f658470d90a233ce3d11cb536617ec7a288.jpg](../iclr_results/3502_Test of Time_ A Benchmark for Evaluating LLMs on Temporal Reasoning/tables/cd5bcef96fe70ef5c59aba45d5fb6f658470d90a233ce3d11cb536617ec7a288.jpg)

![da71396d19e651e9ee767a2b2047e9a2dd7fa47416b22ca9c01fcf1920bf6554.jpg](../iclr_results/3502_Test of Time_ A Benchmark for Evaluating LLMs on Temporal Reasoning/tables/da71396d19e651e9ee767a2b2047e9a2dd7fa47416b22ca9c01fcf1920bf6554.jpg)

![e88e38152f22c9cab1993405dc899c576088ba918b38b2ab5d28139c00b7d5dc.jpg](../iclr_results/3502_Test of Time_ A Benchmark for Evaluating LLMs on Temporal Reasoning/tables/e88e38152f22c9cab1993405dc899c576088ba918b38b2ab5d28139c00b7d5dc.jpg)

![f570d5a95b5800b2c094aaeb064675fce50638fcb4720936004dda9bec7a47a5.jpg](../iclr_results/3502_Test of Time_ A Benchmark for Evaluating LLMs on Temporal Reasoning/tables/f570d5a95b5800b2c094aaeb064675fce50638fcb4720936004dda9bec7a47a5.jpg)

## Demystifying Online Clustering of Bandits: Enhanced Exploration Under Stochastic and Smoothed Adversarial Contexts


### Images

![0649ff61af1cf48fe4691f8261e0e0249b73e92725a8d74b6a58a2c9c8e99a24.jpg](../iclr_results/3503_Demystifying Online Clustering of Bandits_ Enhanced Exploration Under Stochastic and Smoothed Advers/images/0649ff61af1cf48fe4691f8261e0e0249b73e92725a8d74b6a58a2c9c8e99a24.jpg)

![1f73ea5aedfa5356222fb68f1a788a87bdd0d0ae50a977561eee1df5f07f63a9.jpg](../iclr_results/3503_Demystifying Online Clustering of Bandits_ Enhanced Exploration Under Stochastic and Smoothed Advers/images/1f73ea5aedfa5356222fb68f1a788a87bdd0d0ae50a977561eee1df5f07f63a9.jpg)

![263ae5f6319e9d117397ea81111e934cbd7736a7906d69bed20cb9ae21611abc.jpg](../iclr_results/3503_Demystifying Online Clustering of Bandits_ Enhanced Exploration Under Stochastic and Smoothed Advers/images/263ae5f6319e9d117397ea81111e934cbd7736a7906d69bed20cb9ae21611abc.jpg)

![a54d60c0c5ac5b12d27c78fcf004b84c85cc3bed2daf4da0836d86915f2ec768.jpg](../iclr_results/3503_Demystifying Online Clustering of Bandits_ Enhanced Exploration Under Stochastic and Smoothed Advers/images/a54d60c0c5ac5b12d27c78fcf004b84c85cc3bed2daf4da0836d86915f2ec768.jpg)

![f32513861311641cd0fa343a94ae2947698b72eeb5f50a0dcedee9010554821c.jpg](../iclr_results/3503_Demystifying Online Clustering of Bandits_ Enhanced Exploration Under Stochastic and Smoothed Advers/images/f32513861311641cd0fa343a94ae2947698b72eeb5f50a0dcedee9010554821c.jpg)

### Tables

![1ddc3a9861b40ab0cb65fc626450a3763541c6b855df1d4c89155a46af9c7972.jpg](../iclr_results/3503_Demystifying Online Clustering of Bandits_ Enhanced Exploration Under Stochastic and Smoothed Advers/tables/1ddc3a9861b40ab0cb65fc626450a3763541c6b855df1d4c89155a46af9c7972.jpg)

![970ed4e1a318fad017bdcff772bb5de7b719130d8ebcedef13cd0c59ed55b5e2.jpg](../iclr_results/3503_Demystifying Online Clustering of Bandits_ Enhanced Exploration Under Stochastic and Smoothed Advers/tables/970ed4e1a318fad017bdcff772bb5de7b719130d8ebcedef13cd0c59ed55b5e2.jpg)

![ed406b49fed932ea4ac68ad04efba048de7e4d96689bf98db2cf20c94488cf66.jpg](../iclr_results/3503_Demystifying Online Clustering of Bandits_ Enhanced Exploration Under Stochastic and Smoothed Advers/tables/ed406b49fed932ea4ac68ad04efba048de7e4d96689bf98db2cf20c94488cf66.jpg)

## Durable Quantization Conditioned Misalignment Attack on Large Language Models


### Images

![575f6c6efc7cfd48704b1b5d89d12e7b1a92c0eef93f62058957c846fca2f5f0.jpg](../iclr_results/3504_Durable Quantization Conditioned Misalignment Attack on Large Language Models/images/575f6c6efc7cfd48704b1b5d89d12e7b1a92c0eef93f62058957c846fca2f5f0.jpg)

![7aaf52af7fff12d4f63c3d3cfc13ab5af7b580f0a4f8f9ec141c7cd91807c66a.jpg](../iclr_results/3504_Durable Quantization Conditioned Misalignment Attack on Large Language Models/images/7aaf52af7fff12d4f63c3d3cfc13ab5af7b580f0a4f8f9ec141c7cd91807c66a.jpg)

### Tables

![76d03109d3d076f111844f2afd52f27d17f2fb2a77284c57bc8546f771ae71cf.jpg](../iclr_results/3504_Durable Quantization Conditioned Misalignment Attack on Large Language Models/tables/76d03109d3d076f111844f2afd52f27d17f2fb2a77284c57bc8546f771ae71cf.jpg)

![99dd242debeaf395d850503dc8076e5078a5e14ac2e56bdb9a1ef29e12076546.jpg](../iclr_results/3504_Durable Quantization Conditioned Misalignment Attack on Large Language Models/tables/99dd242debeaf395d850503dc8076e5078a5e14ac2e56bdb9a1ef29e12076546.jpg)

![a44fbc47b213e14652452cbc42c8f766c8e7290a8dc438513edd8a48df27d94b.jpg](../iclr_results/3504_Durable Quantization Conditioned Misalignment Attack on Large Language Models/tables/a44fbc47b213e14652452cbc42c8f766c8e7290a8dc438513edd8a48df27d94b.jpg)

![b033f754f4b1f8bd73b3c1c811fe0edf5339325391687f6e6bb53e40d89172f3.jpg](../iclr_results/3504_Durable Quantization Conditioned Misalignment Attack on Large Language Models/tables/b033f754f4b1f8bd73b3c1c811fe0edf5339325391687f6e6bb53e40d89172f3.jpg)

![e4c58746ea0aa99dc4e0c134609a3cff582451f42abf7291416ed144b26a8ab4.jpg](../iclr_results/3504_Durable Quantization Conditioned Misalignment Attack on Large Language Models/tables/e4c58746ea0aa99dc4e0c134609a3cff582451f42abf7291416ed144b26a8ab4.jpg)

## Learning a Fast Mixing Exogenous Block MDP using a Single Trajectory


### Images

![079b05fff1b6c4d196f2fff1c3d21a916ad7fdc0f750d012e70b4da099edca36.jpg](../iclr_results/3505_Learning a Fast Mixing Exogenous Block MDP using a Single Trajectory/images/079b05fff1b6c4d196f2fff1c3d21a916ad7fdc0f750d012e70b4da099edca36.jpg)

![423a5d01dfa63b96a54d56a8d98753c8763c5ab68d97ba9a4c756bf3f7aab8bc.jpg](../iclr_results/3505_Learning a Fast Mixing Exogenous Block MDP using a Single Trajectory/images/423a5d01dfa63b96a54d56a8d98753c8763c5ab68d97ba9a4c756bf3f7aab8bc.jpg)

![4dd3962fdfec141c93adc69dc14dd6c39ac9af9a2bb1fb14e7189980cacefd5c.jpg](../iclr_results/3505_Learning a Fast Mixing Exogenous Block MDP using a Single Trajectory/images/4dd3962fdfec141c93adc69dc14dd6c39ac9af9a2bb1fb14e7189980cacefd5c.jpg)

![6e330ede35ef53918418ff165ac37cd3c2562a61e4ae17e379b517f649938fb4.jpg](../iclr_results/3505_Learning a Fast Mixing Exogenous Block MDP using a Single Trajectory/images/6e330ede35ef53918418ff165ac37cd3c2562a61e4ae17e379b517f649938fb4.jpg)

![82c5b094679b90c52ff7c394d575768022fa7d338ee887bf0165bf7f0d4dea23.jpg](../iclr_results/3505_Learning a Fast Mixing Exogenous Block MDP using a Single Trajectory/images/82c5b094679b90c52ff7c394d575768022fa7d338ee887bf0165bf7f0d4dea23.jpg)

![9a7097c644624dab5925ea75ea660ca1a0c7d32b1e1eb670dd1e36a6dfba941b.jpg](../iclr_results/3505_Learning a Fast Mixing Exogenous Block MDP using a Single Trajectory/images/9a7097c644624dab5925ea75ea660ca1a0c7d32b1e1eb670dd1e36a6dfba941b.jpg)

![9b8c6cc674ae383ad54a403a9d1dace1529357004cb011c144152d79653d650c.jpg](../iclr_results/3505_Learning a Fast Mixing Exogenous Block MDP using a Single Trajectory/images/9b8c6cc674ae383ad54a403a9d1dace1529357004cb011c144152d79653d650c.jpg)

![a6ba52fd1bd08d2dd354d6bbf9ea92e9493f19ac1187b7ea073fdb425917b4cf.jpg](../iclr_results/3505_Learning a Fast Mixing Exogenous Block MDP using a Single Trajectory/images/a6ba52fd1bd08d2dd354d6bbf9ea92e9493f19ac1187b7ea073fdb425917b4cf.jpg)

![ba38de2c045572a25a2f36b1e56425c565e2113589b5f2cd21e6c155387b46b8.jpg](../iclr_results/3505_Learning a Fast Mixing Exogenous Block MDP using a Single Trajectory/images/ba38de2c045572a25a2f36b1e56425c565e2113589b5f2cd21e6c155387b46b8.jpg)

![c1000ae24c94af77e78906657e6d862b08d0a3bfa27c2cdb216d3d4f2295a33e.jpg](../iclr_results/3505_Learning a Fast Mixing Exogenous Block MDP using a Single Trajectory/images/c1000ae24c94af77e78906657e6d862b08d0a3bfa27c2cdb216d3d4f2295a33e.jpg)

![e46f10326a9529c21dfc2734c23b763377232020234e5362f201d6cd327afd4f.jpg](../iclr_results/3505_Learning a Fast Mixing Exogenous Block MDP using a Single Trajectory/images/e46f10326a9529c21dfc2734c23b763377232020234e5362f201d6cd327afd4f.jpg)

![f689bdcb7a4177fc580705a6e09005445d4fcc0538a85f2edf2dc691ca8bece1.jpg](../iclr_results/3505_Learning a Fast Mixing Exogenous Block MDP using a Single Trajectory/images/f689bdcb7a4177fc580705a6e09005445d4fcc0538a85f2edf2dc691ca8bece1.jpg)

### Tables

![0890fd1ed57977a4b06dffa947ca2b809542dda8a53923538a15b302e3a66686.jpg](../iclr_results/3505_Learning a Fast Mixing Exogenous Block MDP using a Single Trajectory/tables/0890fd1ed57977a4b06dffa947ca2b809542dda8a53923538a15b302e3a66686.jpg)

![22f8385d742000c8af374d9d9a19a7ecd0e3e6f5e0f046e03d72a7a57176b242.jpg](../iclr_results/3505_Learning a Fast Mixing Exogenous Block MDP using a Single Trajectory/tables/22f8385d742000c8af374d9d9a19a7ecd0e3e6f5e0f046e03d72a7a57176b242.jpg)

![57ea72df5f5dfe7ae67713f98fd43f765632a49aba8fdacf8306501a624547b9.jpg](../iclr_results/3505_Learning a Fast Mixing Exogenous Block MDP using a Single Trajectory/tables/57ea72df5f5dfe7ae67713f98fd43f765632a49aba8fdacf8306501a624547b9.jpg)

![acab0a28f9caf7c449a0c71f425e36dd7a64b8df010c38581320ef34a8713a0b.jpg](../iclr_results/3505_Learning a Fast Mixing Exogenous Block MDP using a Single Trajectory/tables/acab0a28f9caf7c449a0c71f425e36dd7a64b8df010c38581320ef34a8713a0b.jpg)

## Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition


### Images

![030f26dc1f3c1b8fae2387abaac200547ef2ee2d3029252a47d67005806f6fb5.jpg](../iclr_results/3506_Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition/images/030f26dc1f3c1b8fae2387abaac200547ef2ee2d3029252a47d67005806f6fb5.jpg)

![18392d580c8e91c0e72fde892916063b738e2f9319df06dd29285b8e82a2a484.jpg](../iclr_results/3506_Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition/images/18392d580c8e91c0e72fde892916063b738e2f9319df06dd29285b8e82a2a484.jpg)

![30582906b8fbb2be36090a7c197d0d0b6ba0d0f6107647be58b23a8dbb06c7a9.jpg](../iclr_results/3506_Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition/images/30582906b8fbb2be36090a7c197d0d0b6ba0d0f6107647be58b23a8dbb06c7a9.jpg)

![50c1e077e3b2cac2e623eebe0ed6a1cec181408071fbd7a58154e56576494cc9.jpg](../iclr_results/3506_Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition/images/50c1e077e3b2cac2e623eebe0ed6a1cec181408071fbd7a58154e56576494cc9.jpg)

![61acd82ded25fbb4926e600ea60de1a2414ed126188dd2f60280983dcc72d83a.jpg](../iclr_results/3506_Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition/images/61acd82ded25fbb4926e600ea60de1a2414ed126188dd2f60280983dcc72d83a.jpg)

![9f8093b915a789e8e7ffa3292363bff70f1cb66d4c13b9a7ad34b761d9b3fe14.jpg](../iclr_results/3506_Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition/images/9f8093b915a789e8e7ffa3292363bff70f1cb66d4c13b9a7ad34b761d9b3fe14.jpg)

![abeb9fd3db11720136ea2a8fddefb6a0a4ea68b2f522ac9e913f202c63a1e693.jpg](../iclr_results/3506_Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition/images/abeb9fd3db11720136ea2a8fddefb6a0a4ea68b2f522ac9e913f202c63a1e693.jpg)

### Tables

![5a635119213488efc297f6013195728da00f1962774513a41ae00fa1db2ed067.jpg](../iclr_results/3506_Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition/tables/5a635119213488efc297f6013195728da00f1962774513a41ae00fa1db2ed067.jpg)

![7b45d01b59aa228357baa1d816d20bbc512f67519253c4d3ba165c5961130ea0.jpg](../iclr_results/3506_Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition/tables/7b45d01b59aa228357baa1d816d20bbc512f67519253c4d3ba165c5961130ea0.jpg)

![a8e55ae4fbfdf0ae2b616ef9babe7c89e0e6fc4d4ba07f18095758e59732ed7e.jpg](../iclr_results/3506_Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition/tables/a8e55ae4fbfdf0ae2b616ef9babe7c89e0e6fc4d4ba07f18095758e59732ed7e.jpg)

## RankSHAP: Shapley Value Based Feature Attributions for Learning to Rank


### Images

![29d12ddeb8341ad3c1213b4c81b6639f8f66d3130fe5dd722a1f3b5680debbc2.jpg](../iclr_results/3507_RankSHAP_ Shapley Value Based Feature Attributions for Learning to Rank/images/29d12ddeb8341ad3c1213b4c81b6639f8f66d3130fe5dd722a1f3b5680debbc2.jpg)

![53e19563523718e84429559da35bd6a268e2ed91211101374d91542b91e3e344.jpg](../iclr_results/3507_RankSHAP_ Shapley Value Based Feature Attributions for Learning to Rank/images/53e19563523718e84429559da35bd6a268e2ed91211101374d91542b91e3e344.jpg)

![d793edd6ca66464743002c7cab838086a8e17b43a0d78da440a9b01f64cdeb81.jpg](../iclr_results/3507_RankSHAP_ Shapley Value Based Feature Attributions for Learning to Rank/images/d793edd6ca66464743002c7cab838086a8e17b43a0d78da440a9b01f64cdeb81.jpg)

### Tables

![3bbaf3aac36b69842ef4dcc78b6869e4abc0aed7265d317df5663b26bed9fc60.jpg](../iclr_results/3507_RankSHAP_ Shapley Value Based Feature Attributions for Learning to Rank/tables/3bbaf3aac36b69842ef4dcc78b6869e4abc0aed7265d317df5663b26bed9fc60.jpg)

![4eb2fbd99320bdef384e17ae3a7846e24e83e26b0f5566f5d8beaf39bf64e182.jpg](../iclr_results/3507_RankSHAP_ Shapley Value Based Feature Attributions for Learning to Rank/tables/4eb2fbd99320bdef384e17ae3a7846e24e83e26b0f5566f5d8beaf39bf64e182.jpg)

![75b64523f2e7586d0646558192f90cf8bedf544dba0316d304a273bd2263102b.jpg](../iclr_results/3507_RankSHAP_ Shapley Value Based Feature Attributions for Learning to Rank/tables/75b64523f2e7586d0646558192f90cf8bedf544dba0316d304a273bd2263102b.jpg)

![7f3f98d9f2f6540ebf20167937a62ece99aaf13faef99cb955444fcc9aee8785.jpg](../iclr_results/3507_RankSHAP_ Shapley Value Based Feature Attributions for Learning to Rank/tables/7f3f98d9f2f6540ebf20167937a62ece99aaf13faef99cb955444fcc9aee8785.jpg)

![ac476b5228f4280295221b7a5be5c57df4fe1ee2b007a6a694ab339302756fe3.jpg](../iclr_results/3507_RankSHAP_ Shapley Value Based Feature Attributions for Learning to Rank/tables/ac476b5228f4280295221b7a5be5c57df4fe1ee2b007a6a694ab339302756fe3.jpg)

![c3b1861988d9818a9088bde40cd2101f17f1c1f6ca5af84a5e32fd892064e4f7.jpg](../iclr_results/3507_RankSHAP_ Shapley Value Based Feature Attributions for Learning to Rank/tables/c3b1861988d9818a9088bde40cd2101f17f1c1f6ca5af84a5e32fd892064e4f7.jpg)

## Tailoring Mixup to Data for Calibration


### Images

![43787f6413dcbc96a90501ced3125ade1bc6478b363e6ac1796763c92129791b.jpg](../iclr_results/3508_Tailoring Mixup to Data for Calibration/images/43787f6413dcbc96a90501ced3125ade1bc6478b363e6ac1796763c92129791b.jpg)

![71ea3a0086d5f0ccada53bb38764f17cd87c4d26a73549661b272aa3d5e58c82.jpg](../iclr_results/3508_Tailoring Mixup to Data for Calibration/images/71ea3a0086d5f0ccada53bb38764f17cd87c4d26a73549661b272aa3d5e58c82.jpg)

![b813b2e1f8d098d92d169a7a21400951cf02202fee06d14859ca117b10b6b0d1.jpg](../iclr_results/3508_Tailoring Mixup to Data for Calibration/images/b813b2e1f8d098d92d169a7a21400951cf02202fee06d14859ca117b10b6b0d1.jpg)

![c48e527c6bf9666677a26b1128f356ea3c4b52c178f392b7e352d40f3cd6a97a.jpg](../iclr_results/3508_Tailoring Mixup to Data for Calibration/images/c48e527c6bf9666677a26b1128f356ea3c4b52c178f392b7e352d40f3cd6a97a.jpg)

### Tables

![18cc96fb99047d92034a553c027cd107167579b0070e1afc51d9db76aa5fb5b6.jpg](../iclr_results/3508_Tailoring Mixup to Data for Calibration/tables/18cc96fb99047d92034a553c027cd107167579b0070e1afc51d9db76aa5fb5b6.jpg)

![2d958217de58d1fc5522e7719ea3cbc3b6698ae692a837bedd90c509cd074182.jpg](../iclr_results/3508_Tailoring Mixup to Data for Calibration/tables/2d958217de58d1fc5522e7719ea3cbc3b6698ae692a837bedd90c509cd074182.jpg)

![358a2416bdd83e23f11cf9e1e135262bcbc950a00c0664ebb645aa5029ca4d9d.jpg](../iclr_results/3508_Tailoring Mixup to Data for Calibration/tables/358a2416bdd83e23f11cf9e1e135262bcbc950a00c0664ebb645aa5029ca4d9d.jpg)

![4bbd84099118a107784a78077bc431419eeef9c215095e9007c729a544d3f960.jpg](../iclr_results/3508_Tailoring Mixup to Data for Calibration/tables/4bbd84099118a107784a78077bc431419eeef9c215095e9007c729a544d3f960.jpg)

![4d0029d73e5a6cd45e9ef6a86da6106d9960ec51a4dbbeb9e81a4d786d583ce9.jpg](../iclr_results/3508_Tailoring Mixup to Data for Calibration/tables/4d0029d73e5a6cd45e9ef6a86da6106d9960ec51a4dbbeb9e81a4d786d583ce9.jpg)

![64c98060a2fdae3963b00bd80b033a65248c357c6a1858e3b8bf958987955240.jpg](../iclr_results/3508_Tailoring Mixup to Data for Calibration/tables/64c98060a2fdae3963b00bd80b033a65248c357c6a1858e3b8bf958987955240.jpg)

![6525ad75741de3712764c3fac5839b4d724a000ac07012d872315e6b583caf0f.jpg](../iclr_results/3508_Tailoring Mixup to Data for Calibration/tables/6525ad75741de3712764c3fac5839b4d724a000ac07012d872315e6b583caf0f.jpg)

![709679508a02b2720d37f24bf7b7a964ce735fb993edddbccc729cc314a30fa5.jpg](../iclr_results/3508_Tailoring Mixup to Data for Calibration/tables/709679508a02b2720d37f24bf7b7a964ce735fb993edddbccc729cc314a30fa5.jpg)

![74da8bf62666a1ffb6b878fd33e99303a167fefadbc241468832d1be9b141b8e.jpg](../iclr_results/3508_Tailoring Mixup to Data for Calibration/tables/74da8bf62666a1ffb6b878fd33e99303a167fefadbc241468832d1be9b141b8e.jpg)

![7b200dbafcfc364d52a49496cd5e5303eb51273d16db519ca9a5cf8085fdcc70.jpg](../iclr_results/3508_Tailoring Mixup to Data for Calibration/tables/7b200dbafcfc364d52a49496cd5e5303eb51273d16db519ca9a5cf8085fdcc70.jpg)

![99d6f6024f73f5d8ad9f85609484e60460f7a01f861443dce250ae5e8465ca0a.jpg](../iclr_results/3508_Tailoring Mixup to Data for Calibration/tables/99d6f6024f73f5d8ad9f85609484e60460f7a01f861443dce250ae5e8465ca0a.jpg)

![a837cd6457b05d33d7bd3d4d9fbf9162b69c11fc0f9d3f44ec8f6a9d533ddc4f.jpg](../iclr_results/3508_Tailoring Mixup to Data for Calibration/tables/a837cd6457b05d33d7bd3d4d9fbf9162b69c11fc0f9d3f44ec8f6a9d533ddc4f.jpg)

![b237014cc7bc475cd47f6aaa16a402142469df2dc6b37d25981c4f6109ae6e0b.jpg](../iclr_results/3508_Tailoring Mixup to Data for Calibration/tables/b237014cc7bc475cd47f6aaa16a402142469df2dc6b37d25981c4f6109ae6e0b.jpg)

![ba68399b2069c8b3de2cbabfb9fdb1a1e0c0551782933828e27274db52518c44.jpg](../iclr_results/3508_Tailoring Mixup to Data for Calibration/tables/ba68399b2069c8b3de2cbabfb9fdb1a1e0c0551782933828e27274db52518c44.jpg)

![c6f8ddd365ac9832c76189d172bb2a821ec65e1f065118a6cb8eed13a82256c4.jpg](../iclr_results/3508_Tailoring Mixup to Data for Calibration/tables/c6f8ddd365ac9832c76189d172bb2a821ec65e1f065118a6cb8eed13a82256c4.jpg)

![ed0884c2e1e7d93065338df2feadb40dbeb9154d01fae0a3916ee4b5fb8c065b.jpg](../iclr_results/3508_Tailoring Mixup to Data for Calibration/tables/ed0884c2e1e7d93065338df2feadb40dbeb9154d01fae0a3916ee4b5fb8c065b.jpg)

![f74deba9a25fd18621dda605f9147d014400efa59dd67db66f8fa8f4917a187a.jpg](../iclr_results/3508_Tailoring Mixup to Data for Calibration/tables/f74deba9a25fd18621dda605f9147d014400efa59dd67db66f8fa8f4917a187a.jpg)

![fe694386ac0ab04ad910b5fa0434b8218ed21c0126706b4ba16d9d12f0d8c229.jpg](../iclr_results/3508_Tailoring Mixup to Data for Calibration/tables/fe694386ac0ab04ad910b5fa0434b8218ed21c0126706b4ba16d9d12f0d8c229.jpg)

## BrainOOD: Out-of-distribution Generalizable Brain Network Analysis


### Images

![40e0b86a098ffd2f861e9f6daf193d8c404db615c41cb1b0338bee21ab623863.jpg](../iclr_results/3509_BrainOOD_ Out-of-distribution Generalizable Brain Network Analysis/images/40e0b86a098ffd2f861e9f6daf193d8c404db615c41cb1b0338bee21ab623863.jpg)

![902df3e37993d6701374759cccde1ca6f03bc320092fd1a93479fdff861668b4.jpg](../iclr_results/3509_BrainOOD_ Out-of-distribution Generalizable Brain Network Analysis/images/902df3e37993d6701374759cccde1ca6f03bc320092fd1a93479fdff861668b4.jpg)

![c5b916779961ef194696115898d55110ff5754b136e10d2d89f9204a25efa8b6.jpg](../iclr_results/3509_BrainOOD_ Out-of-distribution Generalizable Brain Network Analysis/images/c5b916779961ef194696115898d55110ff5754b136e10d2d89f9204a25efa8b6.jpg)

![e6ef30493386d4436c4a28db6ca243af6cd26f8e2aa49b5c49f26777af1a7f98.jpg](../iclr_results/3509_BrainOOD_ Out-of-distribution Generalizable Brain Network Analysis/images/e6ef30493386d4436c4a28db6ca243af6cd26f8e2aa49b5c49f26777af1a7f98.jpg)

### Tables

![4e047941954c2e6707b60d776698d0c249f2b1fddc9ba3a02b21d467d0cc1833.jpg](../iclr_results/3509_BrainOOD_ Out-of-distribution Generalizable Brain Network Analysis/tables/4e047941954c2e6707b60d776698d0c249f2b1fddc9ba3a02b21d467d0cc1833.jpg)

![c8445a171a78f1e31a84bfb4e6f58cd31d38b85c5a6b253efdcf3a31d490e32e.jpg](../iclr_results/3509_BrainOOD_ Out-of-distribution Generalizable Brain Network Analysis/tables/c8445a171a78f1e31a84bfb4e6f58cd31d38b85c5a6b253efdcf3a31d490e32e.jpg)

![e35fbc2482ed29cb6a1a20a3e87203f6da67f141e8fc62e1d9a9d1327ae20042.jpg](../iclr_results/3509_BrainOOD_ Out-of-distribution Generalizable Brain Network Analysis/tables/e35fbc2482ed29cb6a1a20a3e87203f6da67f141e8fc62e1d9a9d1327ae20042.jpg)

![fe636fe72d2b785a6f7e4ab2e2ffb696c0a5b8e2e2e2ec15ffc701b7d7835afb.jpg](../iclr_results/3509_BrainOOD_ Out-of-distribution Generalizable Brain Network Analysis/tables/fe636fe72d2b785a6f7e4ab2e2ffb696c0a5b8e2e2e2ec15ffc701b7d7835afb.jpg)

## AI2TALE: An Innovative Information Theory-based Approach for Learning to Localize Phishing Attacks


### Images

![7eb6238bd85d7d62c2b7ee32f128ce19963d86a5e0241cbfbbf76f1340b57fdc.jpg](../iclr_results/3510_AI2TALE_ An Innovative Information Theory-based Approach for Learning to Localize Phishing Attacks/images/7eb6238bd85d7d62c2b7ee32f128ce19963d86a5e0241cbfbbf76f1340b57fdc.jpg)

![9c1419f4f928cef6f62b869faac4cfb53c42eb3603d9eb02d0e231f7a2352f9e.jpg](../iclr_results/3510_AI2TALE_ An Innovative Information Theory-based Approach for Learning to Localize Phishing Attacks/images/9c1419f4f928cef6f62b869faac4cfb53c42eb3603d9eb02d0e231f7a2352f9e.jpg)

![f43f318b5cff9c47b0995a18d9deb28a5baadc685b82cf7e4e6dad3c42b4f78d.jpg](../iclr_results/3510_AI2TALE_ An Innovative Information Theory-based Approach for Learning to Localize Phishing Attacks/images/f43f318b5cff9c47b0995a18d9deb28a5baadc685b82cf7e4e6dad3c42b4f78d.jpg)

### Tables

![2b6975c1a99dfe7282780e67e7961302a95e870b91d39079780fa89034a4b1be.jpg](../iclr_results/3510_AI2TALE_ An Innovative Information Theory-based Approach for Learning to Localize Phishing Attacks/tables/2b6975c1a99dfe7282780e67e7961302a95e870b91d39079780fa89034a4b1be.jpg)

![2bbedef4f877bd4da21b7ca493ebdeab873b424a57b99efc36b474162fb2f2c5.jpg](../iclr_results/3510_AI2TALE_ An Innovative Information Theory-based Approach for Learning to Localize Phishing Attacks/tables/2bbedef4f877bd4da21b7ca493ebdeab873b424a57b99efc36b474162fb2f2c5.jpg)

![469a27a13d83d57c361a1f2bfd8e1f5e14f9fdab79d8c26d4123b15514bd44f5.jpg](../iclr_results/3510_AI2TALE_ An Innovative Information Theory-based Approach for Learning to Localize Phishing Attacks/tables/469a27a13d83d57c361a1f2bfd8e1f5e14f9fdab79d8c26d4123b15514bd44f5.jpg)

![da6f73747478761e93ce32f036f0c3eba23cd2f08dfd3481bb5d3fd6e8b11668.jpg](../iclr_results/3510_AI2TALE_ An Innovative Information Theory-based Approach for Learning to Localize Phishing Attacks/tables/da6f73747478761e93ce32f036f0c3eba23cd2f08dfd3481bb5d3fd6e8b11668.jpg)

![e6694fb7d61187b14a318f440cafde35bc70ac0f98da2cd9615e7838ec7dfbff.jpg](../iclr_results/3510_AI2TALE_ An Innovative Information Theory-based Approach for Learning to Localize Phishing Attacks/tables/e6694fb7d61187b14a318f440cafde35bc70ac0f98da2cd9615e7838ec7dfbff.jpg)

## Enhancing Federated Domain Adaptation with Multi-Domain Prototype-Based Federated Fine-Tuning


### Images

![0433cff12562b20abb608f1106c4b58757d3de3267d7bfa6677ee0ae3277420c.jpg](../iclr_results/3511_Enhancing Federated Domain Adaptation with Multi-Domain Prototype-Based Federated Fine-Tuning/images/0433cff12562b20abb608f1106c4b58757d3de3267d7bfa6677ee0ae3277420c.jpg)

![0bc048086009db3587026419b78737e3b7ad8b588b34e9c196b600296706246b.jpg](../iclr_results/3511_Enhancing Federated Domain Adaptation with Multi-Domain Prototype-Based Federated Fine-Tuning/images/0bc048086009db3587026419b78737e3b7ad8b588b34e9c196b600296706246b.jpg)

![1d858ed882ee0cd7ba6e65210b8d752d667dbe96737377619f9f9058b93d4eb0.jpg](../iclr_results/3511_Enhancing Federated Domain Adaptation with Multi-Domain Prototype-Based Federated Fine-Tuning/images/1d858ed882ee0cd7ba6e65210b8d752d667dbe96737377619f9f9058b93d4eb0.jpg)

![238e0d0d2488d69de97c43dc8b7055c81ca908e6b97468cdae3dd4938dd7336b.jpg](../iclr_results/3511_Enhancing Federated Domain Adaptation with Multi-Domain Prototype-Based Federated Fine-Tuning/images/238e0d0d2488d69de97c43dc8b7055c81ca908e6b97468cdae3dd4938dd7336b.jpg)

![4d1b6ab025aa45f86dcb081d7018af3c4e6e45c79d5339a2ee0f7363c20eae02.jpg](../iclr_results/3511_Enhancing Federated Domain Adaptation with Multi-Domain Prototype-Based Federated Fine-Tuning/images/4d1b6ab025aa45f86dcb081d7018af3c4e6e45c79d5339a2ee0f7363c20eae02.jpg)

![530a003a68962320970476f99f54cbc30d3cf77995f272fbf20f19ac491bd832.jpg](../iclr_results/3511_Enhancing Federated Domain Adaptation with Multi-Domain Prototype-Based Federated Fine-Tuning/images/530a003a68962320970476f99f54cbc30d3cf77995f272fbf20f19ac491bd832.jpg)

![b20ae35071435c39bd07892fb5053e9cb5f41b12ee39b3d8e1f87c89553d51cc.jpg](../iclr_results/3511_Enhancing Federated Domain Adaptation with Multi-Domain Prototype-Based Federated Fine-Tuning/images/b20ae35071435c39bd07892fb5053e9cb5f41b12ee39b3d8e1f87c89553d51cc.jpg)

![c1a27158aa664aad9bfcc14cf1eac3ca7abf0d08b11d6b9d5ef9a82995c65be7.jpg](../iclr_results/3511_Enhancing Federated Domain Adaptation with Multi-Domain Prototype-Based Federated Fine-Tuning/images/c1a27158aa664aad9bfcc14cf1eac3ca7abf0d08b11d6b9d5ef9a82995c65be7.jpg)

![dbdfc4799fde6fa74b44394132cf1bece079404986dbf3e6b40ef71659c52ec5.jpg](../iclr_results/3511_Enhancing Federated Domain Adaptation with Multi-Domain Prototype-Based Federated Fine-Tuning/images/dbdfc4799fde6fa74b44394132cf1bece079404986dbf3e6b40ef71659c52ec5.jpg)

### Tables

![0bd9023654634516e852843b7da9a90fc13c7ead7eebc1f2c128d0e4b52cf36a.jpg](../iclr_results/3511_Enhancing Federated Domain Adaptation with Multi-Domain Prototype-Based Federated Fine-Tuning/tables/0bd9023654634516e852843b7da9a90fc13c7ead7eebc1f2c128d0e4b52cf36a.jpg)

![13d1fe172d3ae60300896622491f458bb55c77a4bbdbc6488bf90a2c8efac2d2.jpg](../iclr_results/3511_Enhancing Federated Domain Adaptation with Multi-Domain Prototype-Based Federated Fine-Tuning/tables/13d1fe172d3ae60300896622491f458bb55c77a4bbdbc6488bf90a2c8efac2d2.jpg)

![16bda6ba59ad4eff3875972f1e1298150703a7d1912f9616f1f3dc20de48d762.jpg](../iclr_results/3511_Enhancing Federated Domain Adaptation with Multi-Domain Prototype-Based Federated Fine-Tuning/tables/16bda6ba59ad4eff3875972f1e1298150703a7d1912f9616f1f3dc20de48d762.jpg)

![a39db7911231c7a9902b3b3993d9e9c928d025a890f2caac53d87fb4a088091e.jpg](../iclr_results/3511_Enhancing Federated Domain Adaptation with Multi-Domain Prototype-Based Federated Fine-Tuning/tables/a39db7911231c7a9902b3b3993d9e9c928d025a890f2caac53d87fb4a088091e.jpg)

![aaea8c6744b494a80ceebc0d81eaa73e9a1907b66aa973a8b9bc44460d61ee80.jpg](../iclr_results/3511_Enhancing Federated Domain Adaptation with Multi-Domain Prototype-Based Federated Fine-Tuning/tables/aaea8c6744b494a80ceebc0d81eaa73e9a1907b66aa973a8b9bc44460d61ee80.jpg)

![e40153338635cfea5387a6239d16acb1c25679e438f502620a33ca62238ad385.jpg](../iclr_results/3511_Enhancing Federated Domain Adaptation with Multi-Domain Prototype-Based Federated Fine-Tuning/tables/e40153338635cfea5387a6239d16acb1c25679e438f502620a33ca62238ad385.jpg)

![e910fc33363a06228b9e1c725825237772632cbc5e5926a75de642ce8d91f305.jpg](../iclr_results/3511_Enhancing Federated Domain Adaptation with Multi-Domain Prototype-Based Federated Fine-Tuning/tables/e910fc33363a06228b9e1c725825237772632cbc5e5926a75de642ce8d91f305.jpg)

## Machine Unlearning via Simulated Oracle Matching


### Images

![16da81fa1215f16965984433a04fbfec44a01e3efcffc07af00e493844eb266e.jpg](../iclr_results/3512_Machine Unlearning via Simulated Oracle Matching/images/16da81fa1215f16965984433a04fbfec44a01e3efcffc07af00e493844eb266e.jpg)

![1da3a08afc4a7f0953095cc6b97dc66f499a65590fd1d75540e3d3723fa3b628.jpg](../iclr_results/3512_Machine Unlearning via Simulated Oracle Matching/images/1da3a08afc4a7f0953095cc6b97dc66f499a65590fd1d75540e3d3723fa3b628.jpg)

![2fb110dd73012e5828deb7b4f4c317477ad211b203689d64f57664c837da863d.jpg](../iclr_results/3512_Machine Unlearning via Simulated Oracle Matching/images/2fb110dd73012e5828deb7b4f4c317477ad211b203689d64f57664c837da863d.jpg)

![3a6113f237f4348448c81d14f5bccbb6d3ea454d77f927d838c5b95f90c778f1.jpg](../iclr_results/3512_Machine Unlearning via Simulated Oracle Matching/images/3a6113f237f4348448c81d14f5bccbb6d3ea454d77f927d838c5b95f90c778f1.jpg)

![3a7e6b1fdbbe7ba00ad71267fff5a101170217966b52d22fb65e9da29f48b89f.jpg](../iclr_results/3512_Machine Unlearning via Simulated Oracle Matching/images/3a7e6b1fdbbe7ba00ad71267fff5a101170217966b52d22fb65e9da29f48b89f.jpg)

![3cd9331fa027f6b456af1b72cc5b88c4ece33d80886bd026a1b35258bc56d6bf.jpg](../iclr_results/3512_Machine Unlearning via Simulated Oracle Matching/images/3cd9331fa027f6b456af1b72cc5b88c4ece33d80886bd026a1b35258bc56d6bf.jpg)

![5e7923e2bda6882951659df2265d004a72661049603f4edc0b391ea4213a7705.jpg](../iclr_results/3512_Machine Unlearning via Simulated Oracle Matching/images/5e7923e2bda6882951659df2265d004a72661049603f4edc0b391ea4213a7705.jpg)

![7f6d451cc875e24e461e558be858d845c897f426299631028d57976b1623ded3.jpg](../iclr_results/3512_Machine Unlearning via Simulated Oracle Matching/images/7f6d451cc875e24e461e558be858d845c897f426299631028d57976b1623ded3.jpg)

![933f2cf472ba6ac19bf2b260d26985e64a22571dc963008acc72c3aa4490ef16.jpg](../iclr_results/3512_Machine Unlearning via Simulated Oracle Matching/images/933f2cf472ba6ac19bf2b260d26985e64a22571dc963008acc72c3aa4490ef16.jpg)

![941e67ce76e848fed1b15faed64d550f654cb2162db3954bcddb732d51bbb428.jpg](../iclr_results/3512_Machine Unlearning via Simulated Oracle Matching/images/941e67ce76e848fed1b15faed64d550f654cb2162db3954bcddb732d51bbb428.jpg)

![a777f758faef27a16b7b98677a3f9fcbe5299291c6bf016b7c6ad3d2c7257e1f.jpg](../iclr_results/3512_Machine Unlearning via Simulated Oracle Matching/images/a777f758faef27a16b7b98677a3f9fcbe5299291c6bf016b7c6ad3d2c7257e1f.jpg)

![ac03ca903621bd1fa2f4ac0947089a74947ec968132cacd1fc138213a275806f.jpg](../iclr_results/3512_Machine Unlearning via Simulated Oracle Matching/images/ac03ca903621bd1fa2f4ac0947089a74947ec968132cacd1fc138213a275806f.jpg)

![b2219751cfab7cb2e4a6d42866a24a49f91b19b6ad81a5686c4c810e89cc85fa.jpg](../iclr_results/3512_Machine Unlearning via Simulated Oracle Matching/images/b2219751cfab7cb2e4a6d42866a24a49f91b19b6ad81a5686c4c810e89cc85fa.jpg)

![b82c7ca0e794e846cf6c1e5222c9419d01532c67d78a72890a0540d915c9f519.jpg](../iclr_results/3512_Machine Unlearning via Simulated Oracle Matching/images/b82c7ca0e794e846cf6c1e5222c9419d01532c67d78a72890a0540d915c9f519.jpg)

![ca4ccd27704400a83b914b1a72adb14a6ebd19b579f48576a820c7669cbb4d8a.jpg](../iclr_results/3512_Machine Unlearning via Simulated Oracle Matching/images/ca4ccd27704400a83b914b1a72adb14a6ebd19b579f48576a820c7669cbb4d8a.jpg)

![d139858c2b1c7daeab08dd50e4537e119331ce946aef0f3e4e5f7f80d55493fd.jpg](../iclr_results/3512_Machine Unlearning via Simulated Oracle Matching/images/d139858c2b1c7daeab08dd50e4537e119331ce946aef0f3e4e5f7f80d55493fd.jpg)

![d79a5d9e51c0cd0bcae78607d39630d2a478d8eacb1d7088374bb70a94448b45.jpg](../iclr_results/3512_Machine Unlearning via Simulated Oracle Matching/images/d79a5d9e51c0cd0bcae78607d39630d2a478d8eacb1d7088374bb70a94448b45.jpg)

![dc6b51d4c8d8ad2102be24ba94dac957244c34bd480632849a6e89dee077b91e.jpg](../iclr_results/3512_Machine Unlearning via Simulated Oracle Matching/images/dc6b51d4c8d8ad2102be24ba94dac957244c34bd480632849a6e89dee077b91e.jpg)

![dc8d139dcb607b9bc37b814c3593681d64ac97b2b08423760829b17d55e959e2.jpg](../iclr_results/3512_Machine Unlearning via Simulated Oracle Matching/images/dc8d139dcb607b9bc37b814c3593681d64ac97b2b08423760829b17d55e959e2.jpg)

![e3bf6bc87cdee955db9e1b541aebb8722bd6f06c92bd9e2ce3d8ac995a1852b3.jpg](../iclr_results/3512_Machine Unlearning via Simulated Oracle Matching/images/e3bf6bc87cdee955db9e1b541aebb8722bd6f06c92bd9e2ce3d8ac995a1852b3.jpg)

![e9fcd3cc111300cf446387bb2ba3a981cbaf76e3fb866eacb797d3b80a3ff7d8.jpg](../iclr_results/3512_Machine Unlearning via Simulated Oracle Matching/images/e9fcd3cc111300cf446387bb2ba3a981cbaf76e3fb866eacb797d3b80a3ff7d8.jpg)

![eb10fadbfa29e7abcc28bcaf3f24be46d5cbe8a9fc73d45386060fd9e096063e.jpg](../iclr_results/3512_Machine Unlearning via Simulated Oracle Matching/images/eb10fadbfa29e7abcc28bcaf3f24be46d5cbe8a9fc73d45386060fd9e096063e.jpg)

### Tables

![418a87a99331a063b1d4a92656e501116f3383c7a8d3fd49df6ff6125f30d3c0.jpg](../iclr_results/3512_Machine Unlearning via Simulated Oracle Matching/tables/418a87a99331a063b1d4a92656e501116f3383c7a8d3fd49df6ff6125f30d3c0.jpg)

## BRAID: Input-driven Nonlinear Dynamical Modeling of Neural-Behavioral Data


### Images

![1db4e1c5a69d3f0673abc0516a97d235651702cfa07f08e4dd43ed225b90f1c5.jpg](../iclr_results/3513_BRAID_ Input-driven Nonlinear Dynamical Modeling of Neural-Behavioral Data/images/1db4e1c5a69d3f0673abc0516a97d235651702cfa07f08e4dd43ed225b90f1c5.jpg)

![3bb87256f557fbb7a863b9cbc3ebf8d7329ef6776b69251f1a38a30333308742.jpg](../iclr_results/3513_BRAID_ Input-driven Nonlinear Dynamical Modeling of Neural-Behavioral Data/images/3bb87256f557fbb7a863b9cbc3ebf8d7329ef6776b69251f1a38a30333308742.jpg)

![68ee9da810b8fc7878facca7f8d11cd71cd7597183f5308a06b48b17751608e3.jpg](../iclr_results/3513_BRAID_ Input-driven Nonlinear Dynamical Modeling of Neural-Behavioral Data/images/68ee9da810b8fc7878facca7f8d11cd71cd7597183f5308a06b48b17751608e3.jpg)

![747d60ec04d86858282b56d32600defad22966563c949b07bc2866765416f6cd.jpg](../iclr_results/3513_BRAID_ Input-driven Nonlinear Dynamical Modeling of Neural-Behavioral Data/images/747d60ec04d86858282b56d32600defad22966563c949b07bc2866765416f6cd.jpg)

![7c16cbb937a7e872c4693597f89b9ecae68af082d0d1509dd159043df176eab8.jpg](../iclr_results/3513_BRAID_ Input-driven Nonlinear Dynamical Modeling of Neural-Behavioral Data/images/7c16cbb937a7e872c4693597f89b9ecae68af082d0d1509dd159043df176eab8.jpg)

![8400aaa7c4b1c702bb53330537209d2275150c57bec48421eb0604d9d7194d83.jpg](../iclr_results/3513_BRAID_ Input-driven Nonlinear Dynamical Modeling of Neural-Behavioral Data/images/8400aaa7c4b1c702bb53330537209d2275150c57bec48421eb0604d9d7194d83.jpg)

![96f50b98a2a947b3250bed1df5af9dc16deb7871a411d9d18e74c22eb0322fc1.jpg](../iclr_results/3513_BRAID_ Input-driven Nonlinear Dynamical Modeling of Neural-Behavioral Data/images/96f50b98a2a947b3250bed1df5af9dc16deb7871a411d9d18e74c22eb0322fc1.jpg)

![c0494c52b5d68f7bfd21d326dbdd12ef2ba75e9466b1dc0d85e264276ca10c50.jpg](../iclr_results/3513_BRAID_ Input-driven Nonlinear Dynamical Modeling of Neural-Behavioral Data/images/c0494c52b5d68f7bfd21d326dbdd12ef2ba75e9466b1dc0d85e264276ca10c50.jpg)

![c476acd83d895e9356a1fc32c5be74c0e12ec8fe990dbdfa95602b40f3ee3383.jpg](../iclr_results/3513_BRAID_ Input-driven Nonlinear Dynamical Modeling of Neural-Behavioral Data/images/c476acd83d895e9356a1fc32c5be74c0e12ec8fe990dbdfa95602b40f3ee3383.jpg)

![d831dd7548380083084ee0ecd893ee383e4e8dd61a9d4d40ec6d7aa803f957a7.jpg](../iclr_results/3513_BRAID_ Input-driven Nonlinear Dynamical Modeling of Neural-Behavioral Data/images/d831dd7548380083084ee0ecd893ee383e4e8dd61a9d4d40ec6d7aa803f957a7.jpg)

### Tables

![160528c187e8189563b39fc8380611d8626824c9f80bc1b035123b6e0d6a263a.jpg](../iclr_results/3513_BRAID_ Input-driven Nonlinear Dynamical Modeling of Neural-Behavioral Data/tables/160528c187e8189563b39fc8380611d8626824c9f80bc1b035123b6e0d6a263a.jpg)

![1f4815f13020c0bef15d57fb97f68bc3af13d32bcfe79b2c7a0a5cc7a5a69818.jpg](../iclr_results/3513_BRAID_ Input-driven Nonlinear Dynamical Modeling of Neural-Behavioral Data/tables/1f4815f13020c0bef15d57fb97f68bc3af13d32bcfe79b2c7a0a5cc7a5a69818.jpg)

![5982237d1104d55a6fe8f9269956a9a19a4bf302a4ee123c8eb6d3e5894ce69b.jpg](../iclr_results/3513_BRAID_ Input-driven Nonlinear Dynamical Modeling of Neural-Behavioral Data/tables/5982237d1104d55a6fe8f9269956a9a19a4bf302a4ee123c8eb6d3e5894ce69b.jpg)

![70091474b9abbc4e51bd761517c703f45e9534600e27fa93fe47ca5d286687df.jpg](../iclr_results/3513_BRAID_ Input-driven Nonlinear Dynamical Modeling of Neural-Behavioral Data/tables/70091474b9abbc4e51bd761517c703f45e9534600e27fa93fe47ca5d286687df.jpg)

![8cc8ec0ea9af4c3de44f1365023fae55882fe4fb0af13904bb0a5aacb10ef65f.jpg](../iclr_results/3513_BRAID_ Input-driven Nonlinear Dynamical Modeling of Neural-Behavioral Data/tables/8cc8ec0ea9af4c3de44f1365023fae55882fe4fb0af13904bb0a5aacb10ef65f.jpg)

![94be9dedd5269d8f5e0aec8f4c85c0364247571447c53e023892704c156afdac.jpg](../iclr_results/3513_BRAID_ Input-driven Nonlinear Dynamical Modeling of Neural-Behavioral Data/tables/94be9dedd5269d8f5e0aec8f4c85c0364247571447c53e023892704c156afdac.jpg)

![97e2fb610a7076a18190ca8edad0df0ba1dbf178d54ab432d910529cebb25036.jpg](../iclr_results/3513_BRAID_ Input-driven Nonlinear Dynamical Modeling of Neural-Behavioral Data/tables/97e2fb610a7076a18190ca8edad0df0ba1dbf178d54ab432d910529cebb25036.jpg)

![99bfd088236b26d8e1da0c682d873be20cf2852d4b8124165f2e90149d1de0ef.jpg](../iclr_results/3513_BRAID_ Input-driven Nonlinear Dynamical Modeling of Neural-Behavioral Data/tables/99bfd088236b26d8e1da0c682d873be20cf2852d4b8124165f2e90149d1de0ef.jpg)

![d97d9a212f700369de2f7b4d49b58dee78cfd92da2441c70552db576027a9e03.jpg](../iclr_results/3513_BRAID_ Input-driven Nonlinear Dynamical Modeling of Neural-Behavioral Data/tables/d97d9a212f700369de2f7b4d49b58dee78cfd92da2441c70552db576027a9e03.jpg)

![e4ee718015993e71ad3c9ec4eed7f44969084e676f637ba345441aa1810eaa8b.jpg](../iclr_results/3513_BRAID_ Input-driven Nonlinear Dynamical Modeling of Neural-Behavioral Data/tables/e4ee718015993e71ad3c9ec4eed7f44969084e676f637ba345441aa1810eaa8b.jpg)

![ed295931ab2e852c63cdd97e02c8cc3c792ead88c4992139ef977e161db8e07d.jpg](../iclr_results/3513_BRAID_ Input-driven Nonlinear Dynamical Modeling of Neural-Behavioral Data/tables/ed295931ab2e852c63cdd97e02c8cc3c792ead88c4992139ef977e161db8e07d.jpg)

## Scaling Speech-Text Pre-training with Synthetic Interleaved Data


### Images

![68ab1fa66d374c7f8a657d67f74c001f71df97b2133cbeb013dfbd4452c85630.jpg](../iclr_results/3514_Scaling Speech-Text Pre-training with Synthetic Interleaved Data/images/68ab1fa66d374c7f8a657d67f74c001f71df97b2133cbeb013dfbd4452c85630.jpg)

![ad2da3044f5beee573b35731f1e55acf2d368b5b1c4c4b201e97c8a1ebf360a3.jpg](../iclr_results/3514_Scaling Speech-Text Pre-training with Synthetic Interleaved Data/images/ad2da3044f5beee573b35731f1e55acf2d368b5b1c4c4b201e97c8a1ebf360a3.jpg)

### Tables

![17fa99a8dced5d0c234e124036d05f70815b68381b35aa947b70fff24525a05f.jpg](../iclr_results/3514_Scaling Speech-Text Pre-training with Synthetic Interleaved Data/tables/17fa99a8dced5d0c234e124036d05f70815b68381b35aa947b70fff24525a05f.jpg)

![2243f8f9951068d47e87088419a99f8f5b34ef6e7b23921eff5d11e1e67df51a.jpg](../iclr_results/3514_Scaling Speech-Text Pre-training with Synthetic Interleaved Data/tables/2243f8f9951068d47e87088419a99f8f5b34ef6e7b23921eff5d11e1e67df51a.jpg)

![2634e78c5c56074aac84018474a148372921e7e1b1408f7d09ddbdf74684b6ec.jpg](../iclr_results/3514_Scaling Speech-Text Pre-training with Synthetic Interleaved Data/tables/2634e78c5c56074aac84018474a148372921e7e1b1408f7d09ddbdf74684b6ec.jpg)

![5a06bac4ceb448e45d44835a5cefb6eb540b5dff57adc2e16cfeceb046445638.jpg](../iclr_results/3514_Scaling Speech-Text Pre-training with Synthetic Interleaved Data/tables/5a06bac4ceb448e45d44835a5cefb6eb540b5dff57adc2e16cfeceb046445638.jpg)

![813da4763fa6c1ac7099e3343ee42e3c97e5c66aaf2dd5eb30e12aff5f0636a8.jpg](../iclr_results/3514_Scaling Speech-Text Pre-training with Synthetic Interleaved Data/tables/813da4763fa6c1ac7099e3343ee42e3c97e5c66aaf2dd5eb30e12aff5f0636a8.jpg)

![872d25a8085f82b29a3d88897fec60f177e147109d5f3e8b1e1785cad3bae707.jpg](../iclr_results/3514_Scaling Speech-Text Pre-training with Synthetic Interleaved Data/tables/872d25a8085f82b29a3d88897fec60f177e147109d5f3e8b1e1785cad3bae707.jpg)

![b6dc12f94c62ac3b49b3674b526382b10d2629af27bbb041be0b2c2152c53917.jpg](../iclr_results/3514_Scaling Speech-Text Pre-training with Synthetic Interleaved Data/tables/b6dc12f94c62ac3b49b3674b526382b10d2629af27bbb041be0b2c2152c53917.jpg)

![c1c7881f63bd816b1314b67ef0066d9f8efaf58f58b1b82a3ebd4cd500685570.jpg](../iclr_results/3514_Scaling Speech-Text Pre-training with Synthetic Interleaved Data/tables/c1c7881f63bd816b1314b67ef0066d9f8efaf58f58b1b82a3ebd4cd500685570.jpg)

![c6ffc6075d929d0c87dbcae766c42f112cdc857d65cdc3b3a20673944c1ec173.jpg](../iclr_results/3514_Scaling Speech-Text Pre-training with Synthetic Interleaved Data/tables/c6ffc6075d929d0c87dbcae766c42f112cdc857d65cdc3b3a20673944c1ec173.jpg)

![ce2d9359d0d30c5e3470b29084cf3246e2e0a6977b0c4ac4faaf0fe4fb333433.jpg](../iclr_results/3514_Scaling Speech-Text Pre-training with Synthetic Interleaved Data/tables/ce2d9359d0d30c5e3470b29084cf3246e2e0a6977b0c4ac4faaf0fe4fb333433.jpg)

![e0dcbe03fa2406e23bf2acbcf4f0aeb56949bc080f44b2eca3b9ed2531577cf4.jpg](../iclr_results/3514_Scaling Speech-Text Pre-training with Synthetic Interleaved Data/tables/e0dcbe03fa2406e23bf2acbcf4f0aeb56949bc080f44b2eca3b9ed2531577cf4.jpg)

![f471d495c14c3368a5f4e8ac2a685b96ec56983a0917ecf9db03ad9dfbd21a5f.jpg](../iclr_results/3514_Scaling Speech-Text Pre-training with Synthetic Interleaved Data/tables/f471d495c14c3368a5f4e8ac2a685b96ec56983a0917ecf9db03ad9dfbd21a5f.jpg)

![fabfe7c6909951edfb74b8e98b41ac40290f8288baab80a72fd0f84449e4b8a3.jpg](../iclr_results/3514_Scaling Speech-Text Pre-training with Synthetic Interleaved Data/tables/fabfe7c6909951edfb74b8e98b41ac40290f8288baab80a72fd0f84449e4b8a3.jpg)

## Input Space Mode Connectivity in Deep Neural Networks


### Images

![0ddb878d362455a98fb7901731a2b496e3efdf9ee1d3542253ea9b0d14795f7d.jpg](../iclr_results/3515_Input Space Mode Connectivity in Deep Neural Networks/images/0ddb878d362455a98fb7901731a2b496e3efdf9ee1d3542253ea9b0d14795f7d.jpg)

![1e8dd4d0f039087ea0ccea01e39e23b12af3566261f5b62629172586d694885e.jpg](../iclr_results/3515_Input Space Mode Connectivity in Deep Neural Networks/images/1e8dd4d0f039087ea0ccea01e39e23b12af3566261f5b62629172586d694885e.jpg)

![2e398dcc9f09472a99b98fd1466b758c920c724ad0823712aa71193f5878d2f0.jpg](../iclr_results/3515_Input Space Mode Connectivity in Deep Neural Networks/images/2e398dcc9f09472a99b98fd1466b758c920c724ad0823712aa71193f5878d2f0.jpg)

![4b421baded0de34097bc5ecdd1cba01a55f211485c6ccd9fd5327535eb543d2b.jpg](../iclr_results/3515_Input Space Mode Connectivity in Deep Neural Networks/images/4b421baded0de34097bc5ecdd1cba01a55f211485c6ccd9fd5327535eb543d2b.jpg)

![54c69233507ee9120f02278732ce1e401ee0e2cdfc6f3a4f7debaa5b176f8eac.jpg](../iclr_results/3515_Input Space Mode Connectivity in Deep Neural Networks/images/54c69233507ee9120f02278732ce1e401ee0e2cdfc6f3a4f7debaa5b176f8eac.jpg)

![5f1fc3946df5c8f99a770b97d61a3561084b1b5bb456c3a5dc1dd65db8ed8f58.jpg](../iclr_results/3515_Input Space Mode Connectivity in Deep Neural Networks/images/5f1fc3946df5c8f99a770b97d61a3561084b1b5bb456c3a5dc1dd65db8ed8f58.jpg)

![5f5d6fba3ce909ebe029a3b66a8370345c29ab85fa1af93f5d3c37f50057948f.jpg](../iclr_results/3515_Input Space Mode Connectivity in Deep Neural Networks/images/5f5d6fba3ce909ebe029a3b66a8370345c29ab85fa1af93f5d3c37f50057948f.jpg)

![63429c0f0e46fa961263fcfd1c7e51312c186f5f087c7efc0f8a3e5b74af2404.jpg](../iclr_results/3515_Input Space Mode Connectivity in Deep Neural Networks/images/63429c0f0e46fa961263fcfd1c7e51312c186f5f087c7efc0f8a3e5b74af2404.jpg)

![739a3bf3f5d15ff428c2153dc21fcdd8ec209dcdd74e5b5cb1f1d6b57c7ffae0.jpg](../iclr_results/3515_Input Space Mode Connectivity in Deep Neural Networks/images/739a3bf3f5d15ff428c2153dc21fcdd8ec209dcdd74e5b5cb1f1d6b57c7ffae0.jpg)

![8dc52593f61a80daa6dc89c03020466e7e2f2103b425958a2f9c0e46ae2e6922.jpg](../iclr_results/3515_Input Space Mode Connectivity in Deep Neural Networks/images/8dc52593f61a80daa6dc89c03020466e7e2f2103b425958a2f9c0e46ae2e6922.jpg)

![94f7415d49a7e73232b48a14da0812ef111a861f50048776a1dde5d44a17bd87.jpg](../iclr_results/3515_Input Space Mode Connectivity in Deep Neural Networks/images/94f7415d49a7e73232b48a14da0812ef111a861f50048776a1dde5d44a17bd87.jpg)

![9f068d1008f32fb595e06e3090d9ecab709171ec06a8bf901b2af4b947ea9f00.jpg](../iclr_results/3515_Input Space Mode Connectivity in Deep Neural Networks/images/9f068d1008f32fb595e06e3090d9ecab709171ec06a8bf901b2af4b947ea9f00.jpg)

![c7fc1f35b22726d63b90e6dcc0a957dd330f3a38308e91e77c22a78896ee3258.jpg](../iclr_results/3515_Input Space Mode Connectivity in Deep Neural Networks/images/c7fc1f35b22726d63b90e6dcc0a957dd330f3a38308e91e77c22a78896ee3258.jpg)

![cb0b372ad1308132e4b432aa327c27507dc1b5580bbf829b48963341dea60e41.jpg](../iclr_results/3515_Input Space Mode Connectivity in Deep Neural Networks/images/cb0b372ad1308132e4b432aa327c27507dc1b5580bbf829b48963341dea60e41.jpg)

![deea3924a40519c1f459ed22140013c8e9581914cf15ea90bae910bb8b392099.jpg](../iclr_results/3515_Input Space Mode Connectivity in Deep Neural Networks/images/deea3924a40519c1f459ed22140013c8e9581914cf15ea90bae910bb8b392099.jpg)

![e5977f53bcf96100dddd0eb6ce6695e745cb34ac1417765be4c9bdf3e186fff9.jpg](../iclr_results/3515_Input Space Mode Connectivity in Deep Neural Networks/images/e5977f53bcf96100dddd0eb6ce6695e745cb34ac1417765be4c9bdf3e186fff9.jpg)

![e85796d2cc20e3b6e71d44cb20e3f38f6a72b13db572f3d81ae8805a18e171e2.jpg](../iclr_results/3515_Input Space Mode Connectivity in Deep Neural Networks/images/e85796d2cc20e3b6e71d44cb20e3f38f6a72b13db572f3d81ae8805a18e171e2.jpg)

![e9d96f35ee0a471136fcbfb124693b973d5367b456ff8aa30f2ea23f885f7e9b.jpg](../iclr_results/3515_Input Space Mode Connectivity in Deep Neural Networks/images/e9d96f35ee0a471136fcbfb124693b973d5367b456ff8aa30f2ea23f885f7e9b.jpg)

### Tables

![d5fdc086692ce7cbaefec1ae121529b8fe20fba160b9980821c59ab5d66928c9.jpg](../iclr_results/3515_Input Space Mode Connectivity in Deep Neural Networks/tables/d5fdc086692ce7cbaefec1ae121529b8fe20fba160b9980821c59ab5d66928c9.jpg)

![dfae61761bf0143906a910c95a638ae65502a957e7e7419e556d2d3d5350e91a.jpg](../iclr_results/3515_Input Space Mode Connectivity in Deep Neural Networks/tables/dfae61761bf0143906a910c95a638ae65502a957e7e7419e556d2d3d5350e91a.jpg)

## Automatic Curriculum Expert Iteration for Reliable LLM Reasoning


### Images

![2b19b4e0c8882affc9a4eb1411a1ed5bfc0a98a4f3c15fb846f485879c700ad9.jpg](../iclr_results/3516_Automatic Curriculum Expert Iteration for Reliable LLM Reasoning/images/2b19b4e0c8882affc9a4eb1411a1ed5bfc0a98a4f3c15fb846f485879c700ad9.jpg)

![64ba75899a9d5c9b5cadab04106cdd13609ba21146a34adca07d811075956bcb.jpg](../iclr_results/3516_Automatic Curriculum Expert Iteration for Reliable LLM Reasoning/images/64ba75899a9d5c9b5cadab04106cdd13609ba21146a34adca07d811075956bcb.jpg)

![6907426d047e30e36e6e99e6e24fd430fc573c6a75ffefa4aa45def905cb41c6.jpg](../iclr_results/3516_Automatic Curriculum Expert Iteration for Reliable LLM Reasoning/images/6907426d047e30e36e6e99e6e24fd430fc573c6a75ffefa4aa45def905cb41c6.jpg)

![73b5cce964463d0278275154b5cd28c40edb7223abbf85eca9a677e667236f53.jpg](../iclr_results/3516_Automatic Curriculum Expert Iteration for Reliable LLM Reasoning/images/73b5cce964463d0278275154b5cd28c40edb7223abbf85eca9a677e667236f53.jpg)

![83cda20abc921f8650a130d2838500e612b4c4d2bea193ba40d200104961ed47.jpg](../iclr_results/3516_Automatic Curriculum Expert Iteration for Reliable LLM Reasoning/images/83cda20abc921f8650a130d2838500e612b4c4d2bea193ba40d200104961ed47.jpg)

![9697c4720cb912fbc8ec7078dc084296e918fa4f41c28f4719c24ccddc0bcceb.jpg](../iclr_results/3516_Automatic Curriculum Expert Iteration for Reliable LLM Reasoning/images/9697c4720cb912fbc8ec7078dc084296e918fa4f41c28f4719c24ccddc0bcceb.jpg)

![9888903bf921599dfac1d9042b689e2cf0d65d052a602e5ca75f672586253a02.jpg](../iclr_results/3516_Automatic Curriculum Expert Iteration for Reliable LLM Reasoning/images/9888903bf921599dfac1d9042b689e2cf0d65d052a602e5ca75f672586253a02.jpg)

### Tables

![045ae49ecdfee52276b6291a842314e23e74c63e744f203c72fd8fe399d4f3ee.jpg](../iclr_results/3516_Automatic Curriculum Expert Iteration for Reliable LLM Reasoning/tables/045ae49ecdfee52276b6291a842314e23e74c63e744f203c72fd8fe399d4f3ee.jpg)

![459622e368f2f32354b78de2d9d35458f2aa7cb0f7d4584883cfe071bd0991b8.jpg](../iclr_results/3516_Automatic Curriculum Expert Iteration for Reliable LLM Reasoning/tables/459622e368f2f32354b78de2d9d35458f2aa7cb0f7d4584883cfe071bd0991b8.jpg)

![524e0207c430637e8ef1508e5d693750226b743b6e0834fed8c131c8d316a805.jpg](../iclr_results/3516_Automatic Curriculum Expert Iteration for Reliable LLM Reasoning/tables/524e0207c430637e8ef1508e5d693750226b743b6e0834fed8c131c8d316a805.jpg)

![58d8a41a3f3d18887d7c15c50827f875dd173b16695a0f0e55f25cf31a885c0c.jpg](../iclr_results/3516_Automatic Curriculum Expert Iteration for Reliable LLM Reasoning/tables/58d8a41a3f3d18887d7c15c50827f875dd173b16695a0f0e55f25cf31a885c0c.jpg)

![9e072561e67585a9df75c5d7165bcde28ddbc4d6ab2ea7c63136ea6ac8c695c3.jpg](../iclr_results/3516_Automatic Curriculum Expert Iteration for Reliable LLM Reasoning/tables/9e072561e67585a9df75c5d7165bcde28ddbc4d6ab2ea7c63136ea6ac8c695c3.jpg)

## Extendable and Iterative Structure Learning Strategy for Bayesian Networks


### Images

![0b30fd5a93466f790f0183229d8a446b0ee565bc8b06846415a70cc8bd06d885.jpg](../iclr_results/3517_Extendable and Iterative Structure Learning Strategy for Bayesian Networks/images/0b30fd5a93466f790f0183229d8a446b0ee565bc8b06846415a70cc8bd06d885.jpg)

![0e1709f20ba37c444c8ab43a2602c391c2014a8df585bef8ff1210236b5faf10.jpg](../iclr_results/3517_Extendable and Iterative Structure Learning Strategy for Bayesian Networks/images/0e1709f20ba37c444c8ab43a2602c391c2014a8df585bef8ff1210236b5faf10.jpg)

### Tables

![02a849a780bde75a254e04be885fa1bbb153f4cc8f401e91ce9db876ee119e0e.jpg](../iclr_results/3517_Extendable and Iterative Structure Learning Strategy for Bayesian Networks/tables/02a849a780bde75a254e04be885fa1bbb153f4cc8f401e91ce9db876ee119e0e.jpg)

![168f84271c694e2d28daacc8fc376ca5020432ca4ad0c3e9287a3ab82efb2dee.jpg](../iclr_results/3517_Extendable and Iterative Structure Learning Strategy for Bayesian Networks/tables/168f84271c694e2d28daacc8fc376ca5020432ca4ad0c3e9287a3ab82efb2dee.jpg)

![711acf3b49aaa9d54ba0f3935e604ccc8dfbf96092cf25701941780c7f8888c0.jpg](../iclr_results/3517_Extendable and Iterative Structure Learning Strategy for Bayesian Networks/tables/711acf3b49aaa9d54ba0f3935e604ccc8dfbf96092cf25701941780c7f8888c0.jpg)

![7193708d32a263cc8881f3cf9d07f3443b942ddeea60a45065190398aa34f251.jpg](../iclr_results/3517_Extendable and Iterative Structure Learning Strategy for Bayesian Networks/tables/7193708d32a263cc8881f3cf9d07f3443b942ddeea60a45065190398aa34f251.jpg)

![d0ac18447b8ba827adb2bb40b207159c11a9a0cef7120e78dd1e799f973e5187.jpg](../iclr_results/3517_Extendable and Iterative Structure Learning Strategy for Bayesian Networks/tables/d0ac18447b8ba827adb2bb40b207159c11a9a0cef7120e78dd1e799f973e5187.jpg)

![dc88343e10817da270c4033eca2b4f84b3dd8c7df0d17bdf04ad0e5066e105e4.jpg](../iclr_results/3517_Extendable and Iterative Structure Learning Strategy for Bayesian Networks/tables/dc88343e10817da270c4033eca2b4f84b3dd8c7df0d17bdf04ad0e5066e105e4.jpg)

![e4de957ef1b9583edca7d4560679b734816e4aa4709200d746ff15cdded0ec14.jpg](../iclr_results/3517_Extendable and Iterative Structure Learning Strategy for Bayesian Networks/tables/e4de957ef1b9583edca7d4560679b734816e4aa4709200d746ff15cdded0ec14.jpg)

![ff8d428be00467ed93e09ea4a89872c753667d9482162ee6c3bb7aa702b37946.jpg](../iclr_results/3517_Extendable and Iterative Structure Learning Strategy for Bayesian Networks/tables/ff8d428be00467ed93e09ea4a89872c753667d9482162ee6c3bb7aa702b37946.jpg)

## An Information Criterion for Controlled Disentanglement of Multimodal Data


### Images

![1c7c89eadfc850ec500b7ac565cdb06a8c8c1801ef9c9f62e4d4cca646d99692.jpg](../iclr_results/3518_An Information Criterion for Controlled Disentanglement of Multimodal Data/images/1c7c89eadfc850ec500b7ac565cdb06a8c8c1801ef9c9f62e4d4cca646d99692.jpg)

![23894581356fee78f332c8edbeed1738d7f658108e74bd9fc1c01eef32f9dd64.jpg](../iclr_results/3518_An Information Criterion for Controlled Disentanglement of Multimodal Data/images/23894581356fee78f332c8edbeed1738d7f658108e74bd9fc1c01eef32f9dd64.jpg)

![3e54372a767a93233eaa4916c4f52aaaa4d053524c59acc8073f44ff4380beaf.jpg](../iclr_results/3518_An Information Criterion for Controlled Disentanglement of Multimodal Data/images/3e54372a767a93233eaa4916c4f52aaaa4d053524c59acc8073f44ff4380beaf.jpg)

![7058606a87390ad348feb969e0be5bca57b767cf789ae618647bd69e33f20ea9.jpg](../iclr_results/3518_An Information Criterion for Controlled Disentanglement of Multimodal Data/images/7058606a87390ad348feb969e0be5bca57b767cf789ae618647bd69e33f20ea9.jpg)

![72a38a2093675432f5511b387da4025c5587b47e5e37360cb2a5a7461458acec.jpg](../iclr_results/3518_An Information Criterion for Controlled Disentanglement of Multimodal Data/images/72a38a2093675432f5511b387da4025c5587b47e5e37360cb2a5a7461458acec.jpg)

![8693f57212b4f71138091b3e63c9e1ffa53bd0d5fb6a685fff2f6af5fecad434.jpg](../iclr_results/3518_An Information Criterion for Controlled Disentanglement of Multimodal Data/images/8693f57212b4f71138091b3e63c9e1ffa53bd0d5fb6a685fff2f6af5fecad434.jpg)

![8a03d67407f1bddb78c474c073e32a4b97535def71dd4703c407463953eeab0c.jpg](../iclr_results/3518_An Information Criterion for Controlled Disentanglement of Multimodal Data/images/8a03d67407f1bddb78c474c073e32a4b97535def71dd4703c407463953eeab0c.jpg)

![c44d704fb86eb7c4714c0c204966645f48403e030d230499391f00ee8a6e16d7.jpg](../iclr_results/3518_An Information Criterion for Controlled Disentanglement of Multimodal Data/images/c44d704fb86eb7c4714c0c204966645f48403e030d230499391f00ee8a6e16d7.jpg)

![e66a95ba51002d507c26445828b4df1815e27e4b44f73a3b9bbe42519da472d2.jpg](../iclr_results/3518_An Information Criterion for Controlled Disentanglement of Multimodal Data/images/e66a95ba51002d507c26445828b4df1815e27e4b44f73a3b9bbe42519da472d2.jpg)

![e7c017bca95979b35326d49d43fcf86cc147aa8611bae263a2e8cbbd0a8fd5e6.jpg](../iclr_results/3518_An Information Criterion for Controlled Disentanglement of Multimodal Data/images/e7c017bca95979b35326d49d43fcf86cc147aa8611bae263a2e8cbbd0a8fd5e6.jpg)

![fb3306969842f580a5ff0698d54bd93305420711d97608d10649c4598bed07b6.jpg](../iclr_results/3518_An Information Criterion for Controlled Disentanglement of Multimodal Data/images/fb3306969842f580a5ff0698d54bd93305420711d97608d10649c4598bed07b6.jpg)

### Tables

![0121e08e0b110b20640ccc868bd0bf5191bfe3a320a5a06bb994f1d881723d7e.jpg](../iclr_results/3518_An Information Criterion for Controlled Disentanglement of Multimodal Data/tables/0121e08e0b110b20640ccc868bd0bf5191bfe3a320a5a06bb994f1d881723d7e.jpg)

![3481520857bd4b04defb977ca3362808c7e95a3c336882bf9602d3fcb0b540d7.jpg](../iclr_results/3518_An Information Criterion for Controlled Disentanglement of Multimodal Data/tables/3481520857bd4b04defb977ca3362808c7e95a3c336882bf9602d3fcb0b540d7.jpg)

![6564e56bceeba0072bb3ae0b9bf99b63bd72530b8a1b61b3e287b47bb1fc7c75.jpg](../iclr_results/3518_An Information Criterion for Controlled Disentanglement of Multimodal Data/tables/6564e56bceeba0072bb3ae0b9bf99b63bd72530b8a1b61b3e287b47bb1fc7c75.jpg)

![aaa647168304fabffb8d312b9ea486c63c56d10513ef73c5bdcf4cc7f2b34f4c.jpg](../iclr_results/3518_An Information Criterion for Controlled Disentanglement of Multimodal Data/tables/aaa647168304fabffb8d312b9ea486c63c56d10513ef73c5bdcf4cc7f2b34f4c.jpg)

![b1d7c0bda8173bf5caf53f38381ced6c815cb4f484f831a4fcd0a5a3c6f51946.jpg](../iclr_results/3518_An Information Criterion for Controlled Disentanglement of Multimodal Data/tables/b1d7c0bda8173bf5caf53f38381ced6c815cb4f484f831a4fcd0a5a3c6f51946.jpg)

![d78022d59a60944b5634317c6fd2c5000eccb4451adc3391938d69315ebeb894.jpg](../iclr_results/3518_An Information Criterion for Controlled Disentanglement of Multimodal Data/tables/d78022d59a60944b5634317c6fd2c5000eccb4451adc3391938d69315ebeb894.jpg)

![ea362385ea83001cbdabcc1eafe4f31d9fcd6958d19fd2bb428ea2020b86ac93.jpg](../iclr_results/3518_An Information Criterion for Controlled Disentanglement of Multimodal Data/tables/ea362385ea83001cbdabcc1eafe4f31d9fcd6958d19fd2bb428ea2020b86ac93.jpg)

## Simulating Human-like Daily Activities with Desire-driven Autonomy


### Images

![05129bff5d950c7e2f05a7d00dc980746cd652f1e266323f590a61b9a7e0e118.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/images/05129bff5d950c7e2f05a7d00dc980746cd652f1e266323f590a61b9a7e0e118.jpg)

![090175a7ee36e4f3fffd368486c9aa82a17864090e29e2de818dd013182b463f.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/images/090175a7ee36e4f3fffd368486c9aa82a17864090e29e2de818dd013182b463f.jpg)

![1546a10303124782014201a87be55ebff4ab00c603eb63fd9388ea413f558202.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/images/1546a10303124782014201a87be55ebff4ab00c603eb63fd9388ea413f558202.jpg)

![353d7bab7a51dbe45217c4827dc99cf248c6fea86c4e53e011c4b187b179e33c.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/images/353d7bab7a51dbe45217c4827dc99cf248c6fea86c4e53e011c4b187b179e33c.jpg)

![59a81d27eb11dbc9ba36046742222c3a65c4f277a64b02b8ad9be7d618951bac.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/images/59a81d27eb11dbc9ba36046742222c3a65c4f277a64b02b8ad9be7d618951bac.jpg)

![606d4c50844fa61381bb05c428e3a2c241c7be16948dc8704f400e74fb84632a.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/images/606d4c50844fa61381bb05c428e3a2c241c7be16948dc8704f400e74fb84632a.jpg)

![65b967439982e7467fcf134c4afdd97264ae012507ed73b0f0b006e5e81581d5.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/images/65b967439982e7467fcf134c4afdd97264ae012507ed73b0f0b006e5e81581d5.jpg)

![6d7afe08d1a06f870d0253f60bbc7ce69be6a41db8fb5f12ef38cf8206f2fc96.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/images/6d7afe08d1a06f870d0253f60bbc7ce69be6a41db8fb5f12ef38cf8206f2fc96.jpg)

![764f96b2df485af0af78a9da4a42cdf101068931062501c199c0fc9517df21d9.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/images/764f96b2df485af0af78a9da4a42cdf101068931062501c199c0fc9517df21d9.jpg)

![89de3032c37be0b645e6d26a170394203e60f0ef02c767ffdf03a90852e97955.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/images/89de3032c37be0b645e6d26a170394203e60f0ef02c767ffdf03a90852e97955.jpg)

![ade99a96c96c82f3ebd3b1810ac2db2e55197d8ad37032228d3906e23b206844.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/images/ade99a96c96c82f3ebd3b1810ac2db2e55197d8ad37032228d3906e23b206844.jpg)

![af7a98ce6b1f30f803daf8cbab29aa23c68953b70590541d82f6834f0a504bc8.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/images/af7a98ce6b1f30f803daf8cbab29aa23c68953b70590541d82f6834f0a504bc8.jpg)

![bd17eee26a794439943b8a8d84e2dc9028da74fe8744abd9565d9887ab65ab40.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/images/bd17eee26a794439943b8a8d84e2dc9028da74fe8744abd9565d9887ab65ab40.jpg)

![d04f24b438f9b3d334ad2a910089507c6d5f21ffc75ae123b092e1a260d02b26.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/images/d04f24b438f9b3d334ad2a910089507c6d5f21ffc75ae123b092e1a260d02b26.jpg)

![da0c2771c968715611e086463157dff7ce165aa60756760cae01a85e220bb7e4.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/images/da0c2771c968715611e086463157dff7ce165aa60756760cae01a85e220bb7e4.jpg)

![e36564b8482004260553bb982711fb44d0845d74dfdf7e8d034c37fd5c39dfe8.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/images/e36564b8482004260553bb982711fb44d0845d74dfdf7e8d034c37fd5c39dfe8.jpg)

![f1e32184cbda2fde24ad48d4560d7d9e0c8c245edf1c4e580940a4092086676d.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/images/f1e32184cbda2fde24ad48d4560d7d9e0c8c245edf1c4e580940a4092086676d.jpg)

![fd6f185fd800d626b8a6663c3808feee84522325455371b645798fabbf49a04e.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/images/fd6f185fd800d626b8a6663c3808feee84522325455371b645798fabbf49a04e.jpg)

### Tables

![04d09a7336cebe30e60743582a22e69f8bbd51ba4f7d3052e76db1aa5291a281.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/tables/04d09a7336cebe30e60743582a22e69f8bbd51ba4f7d3052e76db1aa5291a281.jpg)

![0c8bd753e336f93cb70c532cb89f68bbcee8a136380c55d768174a691d92701e.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/tables/0c8bd753e336f93cb70c532cb89f68bbcee8a136380c55d768174a691d92701e.jpg)

![163a597a14e94b2304e711b6476cbd1b953b59e4f860ebef0f56c6aa1426f5c6.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/tables/163a597a14e94b2304e711b6476cbd1b953b59e4f860ebef0f56c6aa1426f5c6.jpg)

![423df40bcbb3078640efae8077ef3d78c811dfa9eec82dff382b81243ec1cf95.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/tables/423df40bcbb3078640efae8077ef3d78c811dfa9eec82dff382b81243ec1cf95.jpg)

![770c226ccfc392be3877d764769df4f071f960fa5ef60555721714478084a4e2.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/tables/770c226ccfc392be3877d764769df4f071f960fa5ef60555721714478084a4e2.jpg)

![c327903d55151d2a08b796d4c204919733f4264bd0cd2e79be19ad8c37749d61.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/tables/c327903d55151d2a08b796d4c204919733f4264bd0cd2e79be19ad8c37749d61.jpg)

![de8241e116b2fe745306adfb34dd5da70c24f7596be3c7396edb7fab0a01f152.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/tables/de8241e116b2fe745306adfb34dd5da70c24f7596be3c7396edb7fab0a01f152.jpg)

![e5e8240c8e53c9282dab797f77757a1139a2929c44eb2d2cfbe57f5cad6f6b69.jpg](../iclr_results/3519_Simulating Human-like Daily Activities with Desire-driven Autonomy/tables/e5e8240c8e53c9282dab797f77757a1139a2929c44eb2d2cfbe57f5cad6f6b69.jpg)

## ConMix: Contrastive Mixup at Representation Level for Long-tailed Deep Clustering


### Images

![6111b872dcaf5160954794bc4e025fa96a9026e8e5f593cf46eb8c0e01d35626.jpg](../iclr_results/3520_ConMix_ Contrastive Mixup at Representation Level for Long-tailed Deep Clustering/images/6111b872dcaf5160954794bc4e025fa96a9026e8e5f593cf46eb8c0e01d35626.jpg)

![fc0ef2ffcb1d7327b49d74da22cce8422c4f892af42ea088960bec431a3741c0.jpg](../iclr_results/3520_ConMix_ Contrastive Mixup at Representation Level for Long-tailed Deep Clustering/images/fc0ef2ffcb1d7327b49d74da22cce8422c4f892af42ea088960bec431a3741c0.jpg)

### Tables

![2c6acf233c3652759288de396efc7c510d51fcdc600cfe9630d9ffc84a3a1054.jpg](../iclr_results/3520_ConMix_ Contrastive Mixup at Representation Level for Long-tailed Deep Clustering/tables/2c6acf233c3652759288de396efc7c510d51fcdc600cfe9630d9ffc84a3a1054.jpg)

![6289d8f5d7a61b9a56951833e448d90b312b82bcaff86bb2572be128a0da1e6d.jpg](../iclr_results/3520_ConMix_ Contrastive Mixup at Representation Level for Long-tailed Deep Clustering/tables/6289d8f5d7a61b9a56951833e448d90b312b82bcaff86bb2572be128a0da1e6d.jpg)

![77988a1a7f94a04f2fe76c3c6e57f2dfdeb48bfc60b4fb957c586d457be08772.jpg](../iclr_results/3520_ConMix_ Contrastive Mixup at Representation Level for Long-tailed Deep Clustering/tables/77988a1a7f94a04f2fe76c3c6e57f2dfdeb48bfc60b4fb957c586d457be08772.jpg)

![844c9fc29977b66612f1f7c4e2104d3c42856cb1c981babb294ecb7bbef97011.jpg](../iclr_results/3520_ConMix_ Contrastive Mixup at Representation Level for Long-tailed Deep Clustering/tables/844c9fc29977b66612f1f7c4e2104d3c42856cb1c981babb294ecb7bbef97011.jpg)

![888b13971745294347cce344574d74f77ce98345f142a00a6b71b20be075bf98.jpg](../iclr_results/3520_ConMix_ Contrastive Mixup at Representation Level for Long-tailed Deep Clustering/tables/888b13971745294347cce344574d74f77ce98345f142a00a6b71b20be075bf98.jpg)

![983a9abc491f4091be4d1c93d5ddecd127ed39514d6b6af706a023f134c436d8.jpg](../iclr_results/3520_ConMix_ Contrastive Mixup at Representation Level for Long-tailed Deep Clustering/tables/983a9abc491f4091be4d1c93d5ddecd127ed39514d6b6af706a023f134c436d8.jpg)

![a023086231176e3079d3ddd5a4a2bc5a705e1608d1ab7e58302aa85c77dc4747.jpg](../iclr_results/3520_ConMix_ Contrastive Mixup at Representation Level for Long-tailed Deep Clustering/tables/a023086231176e3079d3ddd5a4a2bc5a705e1608d1ab7e58302aa85c77dc4747.jpg)

![a6bec5ac296931adbe30f3bf63175c5c09b5ab783e23d7004d595d484227ba31.jpg](../iclr_results/3520_ConMix_ Contrastive Mixup at Representation Level for Long-tailed Deep Clustering/tables/a6bec5ac296931adbe30f3bf63175c5c09b5ab783e23d7004d595d484227ba31.jpg)

![aa123218bb92ce9610f1472816177209246028ad9fd636ecd137db212629fe6f.jpg](../iclr_results/3520_ConMix_ Contrastive Mixup at Representation Level for Long-tailed Deep Clustering/tables/aa123218bb92ce9610f1472816177209246028ad9fd636ecd137db212629fe6f.jpg)

![ee67af2a4e250e2e21e7a8886ff6cd44cbdec31c4c4f0c1f714869552916bb53.jpg](../iclr_results/3520_ConMix_ Contrastive Mixup at Representation Level for Long-tailed Deep Clustering/tables/ee67af2a4e250e2e21e7a8886ff6cd44cbdec31c4c4f0c1f714869552916bb53.jpg)

## CR2PQ: Continuous Relative Rotary Positional Query for Dense Visual Representation Learning


### Images

![5a9ebbde7d1a74d4b05f1b1bb8f87904a7c0c6306337469677edf420188ae1b4.jpg](../iclr_results/3521_CR2PQ_ Continuous Relative Rotary Positional Query for Dense Visual Representation Learning/images/5a9ebbde7d1a74d4b05f1b1bb8f87904a7c0c6306337469677edf420188ae1b4.jpg)

![676e588c80da0ee3f2f900e004d6e8d64d24d68af777b7e18819d2364c415dbb.jpg](../iclr_results/3521_CR2PQ_ Continuous Relative Rotary Positional Query for Dense Visual Representation Learning/images/676e588c80da0ee3f2f900e004d6e8d64d24d68af777b7e18819d2364c415dbb.jpg)

![c2eec536c52c21e53b06dbe500ee5d9ca207ad4831767db66e46c35300036a91.jpg](../iclr_results/3521_CR2PQ_ Continuous Relative Rotary Positional Query for Dense Visual Representation Learning/images/c2eec536c52c21e53b06dbe500ee5d9ca207ad4831767db66e46c35300036a91.jpg)

![e60e7ac239be1974b6eea1e2e9532007a2fd0ee13dcf35390df8ec350414f032.jpg](../iclr_results/3521_CR2PQ_ Continuous Relative Rotary Positional Query for Dense Visual Representation Learning/images/e60e7ac239be1974b6eea1e2e9532007a2fd0ee13dcf35390df8ec350414f032.jpg)

### Tables

![1e5cdc06ef2fde9d1d3e9311411b2ee6c5115cd3a9211e31544375bf90bf0bf3.jpg](../iclr_results/3521_CR2PQ_ Continuous Relative Rotary Positional Query for Dense Visual Representation Learning/tables/1e5cdc06ef2fde9d1d3e9311411b2ee6c5115cd3a9211e31544375bf90bf0bf3.jpg)

![27f2f76be1575148b3c22935aa55c0a51a6fa98f703342042a0e4b9375c01940.jpg](../iclr_results/3521_CR2PQ_ Continuous Relative Rotary Positional Query for Dense Visual Representation Learning/tables/27f2f76be1575148b3c22935aa55c0a51a6fa98f703342042a0e4b9375c01940.jpg)

![2dc1bb652be633bda94aa962c59a4f6e6ff1bb46ea425d72d1b76cc4d28dbdeb.jpg](../iclr_results/3521_CR2PQ_ Continuous Relative Rotary Positional Query for Dense Visual Representation Learning/tables/2dc1bb652be633bda94aa962c59a4f6e6ff1bb46ea425d72d1b76cc4d28dbdeb.jpg)

![4799c2e5555fccc864ef04383786e5bf353609dbe89a5532c7fc1c2aea820e12.jpg](../iclr_results/3521_CR2PQ_ Continuous Relative Rotary Positional Query for Dense Visual Representation Learning/tables/4799c2e5555fccc864ef04383786e5bf353609dbe89a5532c7fc1c2aea820e12.jpg)

![74273c6afe8b6dfe91ca77217d3c2571d46f075bc2cef19512649b5aea3363ec.jpg](../iclr_results/3521_CR2PQ_ Continuous Relative Rotary Positional Query for Dense Visual Representation Learning/tables/74273c6afe8b6dfe91ca77217d3c2571d46f075bc2cef19512649b5aea3363ec.jpg)

![80b47bd43640d51adf990a36f0951c763d20200d9dd3d34745a0c2574fcd12d6.jpg](../iclr_results/3521_CR2PQ_ Continuous Relative Rotary Positional Query for Dense Visual Representation Learning/tables/80b47bd43640d51adf990a36f0951c763d20200d9dd3d34745a0c2574fcd12d6.jpg)

![a3c2c2cd08b426f4f052c64e7898276c450988470aa0e936b0401095cc1fee7a.jpg](../iclr_results/3521_CR2PQ_ Continuous Relative Rotary Positional Query for Dense Visual Representation Learning/tables/a3c2c2cd08b426f4f052c64e7898276c450988470aa0e936b0401095cc1fee7a.jpg)

![b42d8c80f81ba2d70dc6ad6d4536a96301e86e41058a963a002516932a87afc8.jpg](../iclr_results/3521_CR2PQ_ Continuous Relative Rotary Positional Query for Dense Visual Representation Learning/tables/b42d8c80f81ba2d70dc6ad6d4536a96301e86e41058a963a002516932a87afc8.jpg)

## Iterative Substructure Extraction for Molecular Relational Learning with Interactive Graph Information Bottleneck


### Images

![5bfa7bce349d6fbe26d7e84ae628c2f4bddb1170caf39fec415a68c3caed608f.jpg](../iclr_results/3522_Iterative Substructure Extraction for Molecular Relational Learning with Interactive Graph Informati/images/5bfa7bce349d6fbe26d7e84ae628c2f4bddb1170caf39fec415a68c3caed608f.jpg)

![cd41a8460ab8045d65a7606b05468c8456032ad854be4bd9144140aea68c320f.jpg](../iclr_results/3522_Iterative Substructure Extraction for Molecular Relational Learning with Interactive Graph Informati/images/cd41a8460ab8045d65a7606b05468c8456032ad854be4bd9144140aea68c320f.jpg)

![dbca4620840c7d0492df45e3e1a8482bda0b309abf34dab657358a3f80745f1e.jpg](../iclr_results/3522_Iterative Substructure Extraction for Molecular Relational Learning with Interactive Graph Informati/images/dbca4620840c7d0492df45e3e1a8482bda0b309abf34dab657358a3f80745f1e.jpg)

![fc44c0696f0ef4884fa2282319aaa301059e1530d91a487ce97b9c28fb94af42.jpg](../iclr_results/3522_Iterative Substructure Extraction for Molecular Relational Learning with Interactive Graph Informati/images/fc44c0696f0ef4884fa2282319aaa301059e1530d91a487ce97b9c28fb94af42.jpg)

![feb41ae96afe360d56fd951564d2152b773d40ad2d50dd2785bae46f0fe99331.jpg](../iclr_results/3522_Iterative Substructure Extraction for Molecular Relational Learning with Interactive Graph Informati/images/feb41ae96afe360d56fd951564d2152b773d40ad2d50dd2785bae46f0fe99331.jpg)

### Tables

![021dcdac0a126abbf00cf52b449d1cfabee021ac8f11c976d7744772a49f7a0c.jpg](../iclr_results/3522_Iterative Substructure Extraction for Molecular Relational Learning with Interactive Graph Informati/tables/021dcdac0a126abbf00cf52b449d1cfabee021ac8f11c976d7744772a49f7a0c.jpg)

![2bce0812f60f60c48bdd79aadbe39d39fb7b42fc2fa929c9a40ea6a0e5cb96be.jpg](../iclr_results/3522_Iterative Substructure Extraction for Molecular Relational Learning with Interactive Graph Informati/tables/2bce0812f60f60c48bdd79aadbe39d39fb7b42fc2fa929c9a40ea6a0e5cb96be.jpg)

![334fef2f65377920226d04fad4c6495aeb2d90dacbbcfa324f10e3bd85a2b337.jpg](../iclr_results/3522_Iterative Substructure Extraction for Molecular Relational Learning with Interactive Graph Informati/tables/334fef2f65377920226d04fad4c6495aeb2d90dacbbcfa324f10e3bd85a2b337.jpg)

![44fb1e79f00aa35635b9c4ee42700daa7cf5559422c8346a0fb910a3570aa2db.jpg](../iclr_results/3522_Iterative Substructure Extraction for Molecular Relational Learning with Interactive Graph Informati/tables/44fb1e79f00aa35635b9c4ee42700daa7cf5559422c8346a0fb910a3570aa2db.jpg)

![468acea90fa83e69d8ea8a49f5247746c2ea95c8d49bbb6177eadec3bffb9b3a.jpg](../iclr_results/3522_Iterative Substructure Extraction for Molecular Relational Learning with Interactive Graph Informati/tables/468acea90fa83e69d8ea8a49f5247746c2ea95c8d49bbb6177eadec3bffb9b3a.jpg)

![54d0ed53853c695a3a0c9ef36bc0428d28e8d82d816ab7e5949c95353fe90141.jpg](../iclr_results/3522_Iterative Substructure Extraction for Molecular Relational Learning with Interactive Graph Informati/tables/54d0ed53853c695a3a0c9ef36bc0428d28e8d82d816ab7e5949c95353fe90141.jpg)

![8f526dbeeb5fb5c33e3d7a5b2cb821c6a3a8aaa15b0a55f33e164a9bdd4d1ef4.jpg](../iclr_results/3522_Iterative Substructure Extraction for Molecular Relational Learning with Interactive Graph Informati/tables/8f526dbeeb5fb5c33e3d7a5b2cb821c6a3a8aaa15b0a55f33e164a9bdd4d1ef4.jpg)

![931d981a6a433dd5fae266771765d4b8e25c6f31bd7d2828c90f13775b159eec.jpg](../iclr_results/3522_Iterative Substructure Extraction for Molecular Relational Learning with Interactive Graph Informati/tables/931d981a6a433dd5fae266771765d4b8e25c6f31bd7d2828c90f13775b159eec.jpg)

![992fdf2f0705e494eacd9904c24a7b2648d63c4a6dab42e6b46e105ff7ac77bc.jpg](../iclr_results/3522_Iterative Substructure Extraction for Molecular Relational Learning with Interactive Graph Informati/tables/992fdf2f0705e494eacd9904c24a7b2648d63c4a6dab42e6b46e105ff7ac77bc.jpg)

![9ae26f0bd76469aaf5acdb26e6b145a17071d059205f2615007957ef18162dc7.jpg](../iclr_results/3522_Iterative Substructure Extraction for Molecular Relational Learning with Interactive Graph Informati/tables/9ae26f0bd76469aaf5acdb26e6b145a17071d059205f2615007957ef18162dc7.jpg)

![d5c71026dc34eeef516b2eb8134cf2e0c91e11dc37bc39c18c726ff030f89d6b.jpg](../iclr_results/3522_Iterative Substructure Extraction for Molecular Relational Learning with Interactive Graph Informati/tables/d5c71026dc34eeef516b2eb8134cf2e0c91e11dc37bc39c18c726ff030f89d6b.jpg)

![e0457e9b2e44a1b7c93463c2d1d1255968b71b515c16b93ae6be53fd44f8b360.jpg](../iclr_results/3522_Iterative Substructure Extraction for Molecular Relational Learning with Interactive Graph Informati/tables/e0457e9b2e44a1b7c93463c2d1d1255968b71b515c16b93ae6be53fd44f8b360.jpg)

![eb401d60c38dccdc642f45df590179b00c043982c3d86955bbc614bc438da62c.jpg](../iclr_results/3522_Iterative Substructure Extraction for Molecular Relational Learning with Interactive Graph Informati/tables/eb401d60c38dccdc642f45df590179b00c043982c3d86955bbc614bc438da62c.jpg)

![fcf3fb1a40b75fc3d6fc98af28a55c979e3e9d8dc6bed162ed726bd8d9d97b1c.jpg](../iclr_results/3522_Iterative Substructure Extraction for Molecular Relational Learning with Interactive Graph Informati/tables/fcf3fb1a40b75fc3d6fc98af28a55c979e3e9d8dc6bed162ed726bd8d9d97b1c.jpg)

## Fast training and sampling of Restricted Boltzmann Machines


### Images

![02251c66d02606891a79b45c5a0387c03c33273422b995f9bc0d48b1332c392e.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/images/02251c66d02606891a79b45c5a0387c03c33273422b995f9bc0d48b1332c392e.jpg)

![05546877f46a7336a6a3e3ee382df60f9e52f236ff9da2e305761ee2ce8d9886.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/images/05546877f46a7336a6a3e3ee382df60f9e52f236ff9da2e305761ee2ce8d9886.jpg)

![24a90914ac505c4e948affa5fecb7ba77c3dfb00f50f7d4bb16301f497fa5380.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/images/24a90914ac505c4e948affa5fecb7ba77c3dfb00f50f7d4bb16301f497fa5380.jpg)

![293d0927bee6ff3c163c13f3368c4b1ed84ac74173238c4dbdabbaa52346d289.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/images/293d0927bee6ff3c163c13f3368c4b1ed84ac74173238c4dbdabbaa52346d289.jpg)

![2dec7bd8fa7805918614c02a1e5081ae3bb748ed33988e4ce6c1819e7021104d.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/images/2dec7bd8fa7805918614c02a1e5081ae3bb748ed33988e4ce6c1819e7021104d.jpg)

![2fd442a44a1d1aa06377073d9db1322e95fd0be08c2d9a4f761d2fb0eb9217f5.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/images/2fd442a44a1d1aa06377073d9db1322e95fd0be08c2d9a4f761d2fb0eb9217f5.jpg)

![4147d8687af09702b598fca08f67966152d72c69a9d09cfbc7f72f0b37c8fada.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/images/4147d8687af09702b598fca08f67966152d72c69a9d09cfbc7f72f0b37c8fada.jpg)

![4ed5415c95f3f86993a606d1bc8a3fa3c408abebe10265602c3acaa3e3c1163b.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/images/4ed5415c95f3f86993a606d1bc8a3fa3c408abebe10265602c3acaa3e3c1163b.jpg)

![50a17b3ef2c21415df7d193b43b38cdad0eb3ad7e08f3a3afcc23b5c3f2d2c27.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/images/50a17b3ef2c21415df7d193b43b38cdad0eb3ad7e08f3a3afcc23b5c3f2d2c27.jpg)

![55ae04ed103af120b3f4df5ea39f973397d20418b5d8ddfe69f8d78c744ad4f7.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/images/55ae04ed103af120b3f4df5ea39f973397d20418b5d8ddfe69f8d78c744ad4f7.jpg)

![56e2d0f88366f52ded3a9c6d949f443b25c4a5d4762f4f2a601eb616f432647e.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/images/56e2d0f88366f52ded3a9c6d949f443b25c4a5d4762f4f2a601eb616f432647e.jpg)

![58b3382534afd7854265165b12419fae3b0a76423bf125213e06e94bbc4b58a0.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/images/58b3382534afd7854265165b12419fae3b0a76423bf125213e06e94bbc4b58a0.jpg)

![6572e49c039418ae25dd9a070b2a0835cdd06dee280b4003003d9d86af1e650c.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/images/6572e49c039418ae25dd9a070b2a0835cdd06dee280b4003003d9d86af1e650c.jpg)

![7a49ee1596d8c5e7bac1b55103075018b9a7012bf0574d9a08c821cb0bc690a4.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/images/7a49ee1596d8c5e7bac1b55103075018b9a7012bf0574d9a08c821cb0bc690a4.jpg)

![7b96834e3738d142231a4d67f7dbda56aa8f7bb63700d9e5418e804689617348.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/images/7b96834e3738d142231a4d67f7dbda56aa8f7bb63700d9e5418e804689617348.jpg)

![8fddd41175681a0b352ebcc0e8422815f5de06096a3c69b0ccd3bec8fc13f7a7.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/images/8fddd41175681a0b352ebcc0e8422815f5de06096a3c69b0ccd3bec8fc13f7a7.jpg)

![9994b89cf0e874f49358e253276ed116026f55a35ed4fac2cfcda22f395f67ed.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/images/9994b89cf0e874f49358e253276ed116026f55a35ed4fac2cfcda22f395f67ed.jpg)

![a6d16baba2fbd4dd887d0f7973d9f5e8142fd1d4487c19c1066f59db9452a4f9.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/images/a6d16baba2fbd4dd887d0f7973d9f5e8142fd1d4487c19c1066f59db9452a4f9.jpg)

![b3744451c5f51befc9ef14444a0c6b620fd38c6c44b358c8f79010672a21ca23.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/images/b3744451c5f51befc9ef14444a0c6b620fd38c6c44b358c8f79010672a21ca23.jpg)

![bb2ff1d313bde158aece879f3b5a8689ebb71c046b3c57380940fd7d5f990b6d.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/images/bb2ff1d313bde158aece879f3b5a8689ebb71c046b3c57380940fd7d5f990b6d.jpg)

### Tables

![36936d75fb55af59bff47e1173388afc16f0362e161be1e73f6b16ddc0a5845c.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/tables/36936d75fb55af59bff47e1173388afc16f0362e161be1e73f6b16ddc0a5845c.jpg)

![3fd8b47a4c9e328092fe4d0b95457549cd9e50e2623b0a73ce65ce331b4deca3.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/tables/3fd8b47a4c9e328092fe4d0b95457549cd9e50e2623b0a73ce65ce331b4deca3.jpg)

![531da8d22a11953548d7007860ba9a0aac0d922ca2f1ea2017dcc05f62daf266.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/tables/531da8d22a11953548d7007860ba9a0aac0d922ca2f1ea2017dcc05f62daf266.jpg)

![6a10994c8ee34a94397ba08b3fa1c5e902f17e3f8347952d9aa511605d452ea4.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/tables/6a10994c8ee34a94397ba08b3fa1c5e902f17e3f8347952d9aa511605d452ea4.jpg)

![bbe78be4d0414a1e9e770517cbab613f80f7555beb2c8e8c94699db9ca2eab85.jpg](../iclr_results/3523_Fast training and sampling of Restricted Boltzmann Machines/tables/bbe78be4d0414a1e9e770517cbab613f80f7555beb2c8e8c94699db9ca2eab85.jpg)

## Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout


### Images

![3c2d37775d193830691bbc742b3219d47446febb207bf104740c2ab5ae81ef26.jpg](../iclr_results/3524_Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout/images/3c2d37775d193830691bbc742b3219d47446febb207bf104740c2ab5ae81ef26.jpg)

![3d086792943c35da748bcf30c9fd1752b76c86be66e4f593dc336c24831067fc.jpg](../iclr_results/3524_Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout/images/3d086792943c35da748bcf30c9fd1752b76c86be66e4f593dc336c24831067fc.jpg)

![4db4ccd9e819ac11b9644789cbb76468431af0398fda26d0d912035cdb4f9bd3.jpg](../iclr_results/3524_Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout/images/4db4ccd9e819ac11b9644789cbb76468431af0398fda26d0d912035cdb4f9bd3.jpg)

![7c143775a6a3340d73fdf2a2530a6cd51ffd2ccbe1831be84c3bb2272d41db27.jpg](../iclr_results/3524_Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout/images/7c143775a6a3340d73fdf2a2530a6cd51ffd2ccbe1831be84c3bb2272d41db27.jpg)

![8eb998c16918b0b1ef8b572af5440d92f89e534d4c2f6ba28d60fc8d7f7c6ea6.jpg](../iclr_results/3524_Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout/images/8eb998c16918b0b1ef8b572af5440d92f89e534d4c2f6ba28d60fc8d7f7c6ea6.jpg)

![d568a7d4ab2a15b7594f25c670de3c782cbd56f8695adc82accfc993623bffde.jpg](../iclr_results/3524_Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout/images/d568a7d4ab2a15b7594f25c670de3c782cbd56f8695adc82accfc993623bffde.jpg)

![ea4dcf995c42d02c6371a1b2da99672941986c39d1696da87a00fbade55cdc30.jpg](../iclr_results/3524_Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout/images/ea4dcf995c42d02c6371a1b2da99672941986c39d1696da87a00fbade55cdc30.jpg)

![fd3da00a6e30b240796958d74cfce1b16abe000b6a34b842110a4f50ada871bd.jpg](../iclr_results/3524_Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout/images/fd3da00a6e30b240796958d74cfce1b16abe000b6a34b842110a4f50ada871bd.jpg)

### Tables

![123ab9fe19ee5f8ab2093cc8e0e197e132759bba7dfabce2a091b8122f97acdb.jpg](../iclr_results/3524_Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout/tables/123ab9fe19ee5f8ab2093cc8e0e197e132759bba7dfabce2a091b8122f97acdb.jpg)

![2ee0c0da334b0619cf44ef08586f427f620f8e94bd4326577c1814d41bad2139.jpg](../iclr_results/3524_Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout/tables/2ee0c0da334b0619cf44ef08586f427f620f8e94bd4326577c1814d41bad2139.jpg)

![59d329b1a53b4fdb06b3c6b03d1feb4e8f7c6870c41c56a0abb65f68dec2774a.jpg](../iclr_results/3524_Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout/tables/59d329b1a53b4fdb06b3c6b03d1feb4e8f7c6870c41c56a0abb65f68dec2774a.jpg)

![5fd6dc88ea1f7438a9c8f934d0f6a66c8ef8ebc4811264376e87dec89cfcb820.jpg](../iclr_results/3524_Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout/tables/5fd6dc88ea1f7438a9c8f934d0f6a66c8ef8ebc4811264376e87dec89cfcb820.jpg)

![7312c292da55290c36de6433da6952ac65a087d1dfb1753e4874d0887c8c4d66.jpg](../iclr_results/3524_Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout/tables/7312c292da55290c36de6433da6952ac65a087d1dfb1753e4874d0887c8c4d66.jpg)

![96a02730dac367bfe8117947256360feb92971e596d011ca5d1cf51b2d6a74d0.jpg](../iclr_results/3524_Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout/tables/96a02730dac367bfe8117947256360feb92971e596d011ca5d1cf51b2d6a74d0.jpg)

![a4be304684bc81a5c2fcb6338d0b90780b69533126736b075ce75b5affd5e241.jpg](../iclr_results/3524_Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout/tables/a4be304684bc81a5c2fcb6338d0b90780b69533126736b075ce75b5affd5e241.jpg)

![ab675b99093f4d8298457406d374eb618d019467259f9b938e5df805996057b2.jpg](../iclr_results/3524_Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout/tables/ab675b99093f4d8298457406d374eb618d019467259f9b938e5df805996057b2.jpg)

![d2ceb92c33b5b786492458b23224f9a5af2ef069901c9e024ff428e36cb8e9a2.jpg](../iclr_results/3524_Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout/tables/d2ceb92c33b5b786492458b23224f9a5af2ef069901c9e024ff428e36cb8e9a2.jpg)

![d623650511a31707e31910aea22203e7eb0e9f0684e14da5336c8f8d4f576f44.jpg](../iclr_results/3524_Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout/tables/d623650511a31707e31910aea22203e7eb0e9f0684e14da5336c8f8d4f576f44.jpg)

![e6a3c8be974742a383a14d8adf8989209073fad2810d8da1237f0668469da171.jpg](../iclr_results/3524_Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout/tables/e6a3c8be974742a383a14d8adf8989209073fad2810d8da1237f0668469da171.jpg)

![ede8a4264047e093ffdab60b167a392ae837a6e84a1cfcae6fc646444aeb2d87.jpg](../iclr_results/3524_Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout/tables/ede8a4264047e093ffdab60b167a392ae837a6e84a1cfcae6fc646444aeb2d87.jpg)

![f31c236c6b387f1ae9d7efbc57cbff4c1d8aea08e9336b46e02255deaf52d570.jpg](../iclr_results/3524_Model-Agnostic Knowledge Guided Correction for Improved Neural Surrogate Rollout/tables/f31c236c6b387f1ae9d7efbc57cbff4c1d8aea08e9336b46e02255deaf52d570.jpg)

## On Speeding Up Language Model Evaluation


### Images

![24e4b981cc05e7b87065a40e6a9f74dd19d1306755629314ee78eea0099de1fd.jpg](../iclr_results/3525_On Speeding Up Language Model Evaluation/images/24e4b981cc05e7b87065a40e6a9f74dd19d1306755629314ee78eea0099de1fd.jpg)

![8e6e0cb13c92ed74eeb3da3bd7a1e31f5a981f5dc3087d373feccfc4481b37a5.jpg](../iclr_results/3525_On Speeding Up Language Model Evaluation/images/8e6e0cb13c92ed74eeb3da3bd7a1e31f5a981f5dc3087d373feccfc4481b37a5.jpg)

![aca2155d4e6f339c4b7de8c861d970fbc2342bcb4a447672a658ec314eee2490.jpg](../iclr_results/3525_On Speeding Up Language Model Evaluation/images/aca2155d4e6f339c4b7de8c861d970fbc2342bcb4a447672a658ec314eee2490.jpg)

![d12aeffd1b873b80ed3e8323b97beeb5eba9d95c24d39356b2ccbf5f92db80a5.jpg](../iclr_results/3525_On Speeding Up Language Model Evaluation/images/d12aeffd1b873b80ed3e8323b97beeb5eba9d95c24d39356b2ccbf5f92db80a5.jpg)

![de2c789e04872908f6b5609778d4fa7bb24bad3f581ef33a482ae3ae7ae863ab.jpg](../iclr_results/3525_On Speeding Up Language Model Evaluation/images/de2c789e04872908f6b5609778d4fa7bb24bad3f581ef33a482ae3ae7ae863ab.jpg)

![ec331932405fbb948e87f19a6b762bfa2d813661e9ffa0f7e175fb4421b138b9.jpg](../iclr_results/3525_On Speeding Up Language Model Evaluation/images/ec331932405fbb948e87f19a6b762bfa2d813661e9ffa0f7e175fb4421b138b9.jpg)

![edd8f1fee638ae9b679493a9ed4a06ed93d560e1585aa572dbdb4f8b086f2c01.jpg](../iclr_results/3525_On Speeding Up Language Model Evaluation/images/edd8f1fee638ae9b679493a9ed4a06ed93d560e1585aa572dbdb4f8b086f2c01.jpg)

### Tables

![716fd012a59bdf616cc18a9c90354b6b8b885af16abc3a538d26e54200815ea7.jpg](../iclr_results/3525_On Speeding Up Language Model Evaluation/tables/716fd012a59bdf616cc18a9c90354b6b8b885af16abc3a538d26e54200815ea7.jpg)

![7da81e7228b35f522f9a76f2b02879b0674dbbcc8c14145873fd98a8089ed592.jpg](../iclr_results/3525_On Speeding Up Language Model Evaluation/tables/7da81e7228b35f522f9a76f2b02879b0674dbbcc8c14145873fd98a8089ed592.jpg)

![c584d9fa6de0a51438360199a50746fba9b7a232007dc6dda724342a225d50eb.jpg](../iclr_results/3525_On Speeding Up Language Model Evaluation/tables/c584d9fa6de0a51438360199a50746fba9b7a232007dc6dda724342a225d50eb.jpg)

![f4ef98305d32e1bd90cd8e844cc6094909a146d24c02d363955653d565cb1434.jpg](../iclr_results/3525_On Speeding Up Language Model Evaluation/tables/f4ef98305d32e1bd90cd8e844cc6094909a146d24c02d363955653d565cb1434.jpg)

## Video Action Differencing


### Images

![382b8cf9971a54a0da940a27e9935530290a5deab7afcf79cb3f40661ac12bf1.jpg](../iclr_results/3526_Video Action Differencing/images/382b8cf9971a54a0da940a27e9935530290a5deab7afcf79cb3f40661ac12bf1.jpg)

![67ddec3ec59fe6a3bce49ada3c078d303d1b5ce7e027ec41b121f003986947e3.jpg](../iclr_results/3526_Video Action Differencing/images/67ddec3ec59fe6a3bce49ada3c078d303d1b5ce7e027ec41b121f003986947e3.jpg)

![c60887a35c73b161c2937db58ec8640fabbfbc970fac4cbabd79b9923c460482.jpg](../iclr_results/3526_Video Action Differencing/images/c60887a35c73b161c2937db58ec8640fabbfbc970fac4cbabd79b9923c460482.jpg)

![dcdfb1d0a24275ffa694ef96471df82567cf4df80be8f63d8735af47507df086.jpg](../iclr_results/3526_Video Action Differencing/images/dcdfb1d0a24275ffa694ef96471df82567cf4df80be8f63d8735af47507df086.jpg)

### Tables

![0d37b31acf8ab61951797cb00a7013a1f8596918c9cf9e9742610243cc637463.jpg](../iclr_results/3526_Video Action Differencing/tables/0d37b31acf8ab61951797cb00a7013a1f8596918c9cf9e9742610243cc637463.jpg)

![1056c199fc107c61a7b81069a3ecaad04096f972d94fe8b86ccf0f4e10705731.jpg](../iclr_results/3526_Video Action Differencing/tables/1056c199fc107c61a7b81069a3ecaad04096f972d94fe8b86ccf0f4e10705731.jpg)

![1a48e47841b18a005325b9713565f5eb784a89a2fa91f10fe9e4bfbec1b48a4a.jpg](../iclr_results/3526_Video Action Differencing/tables/1a48e47841b18a005325b9713565f5eb784a89a2fa91f10fe9e4bfbec1b48a4a.jpg)

![1b7d080e3b778b471ef262d4fd2df96c1c884fe3449ddae8c25b45330538878d.jpg](../iclr_results/3526_Video Action Differencing/tables/1b7d080e3b778b471ef262d4fd2df96c1c884fe3449ddae8c25b45330538878d.jpg)

![4d8d6f4af266484848288658caaecde27c246d35f6df6e5f856fa70350194fa2.jpg](../iclr_results/3526_Video Action Differencing/tables/4d8d6f4af266484848288658caaecde27c246d35f6df6e5f856fa70350194fa2.jpg)

![505f60c0ee79c67f56991f629312adbbc03eee13966a5add41b3acafbb0f1044.jpg](../iclr_results/3526_Video Action Differencing/tables/505f60c0ee79c67f56991f629312adbbc03eee13966a5add41b3acafbb0f1044.jpg)

![8133bdaf6d6a314b31d4d3497628e1ba1c7eadbb93a47434a480d01efdd99642.jpg](../iclr_results/3526_Video Action Differencing/tables/8133bdaf6d6a314b31d4d3497628e1ba1c7eadbb93a47434a480d01efdd99642.jpg)

![82eda77c5ac2bd6b06c705303bb201c8ec876c92eb99f396ab502862700146c8.jpg](../iclr_results/3526_Video Action Differencing/tables/82eda77c5ac2bd6b06c705303bb201c8ec876c92eb99f396ab502862700146c8.jpg)

![91d8f3f1e382f49ac9d9de9633712e331abb4fd4b7a188ac302095b302312fa4.jpg](../iclr_results/3526_Video Action Differencing/tables/91d8f3f1e382f49ac9d9de9633712e331abb4fd4b7a188ac302095b302312fa4.jpg)

![b8d8b7fce82f63d42f4f98c7d1e1683bb302b81f56a42507cf92254a761b059d.jpg](../iclr_results/3526_Video Action Differencing/tables/b8d8b7fce82f63d42f4f98c7d1e1683bb302b81f56a42507cf92254a761b059d.jpg)

![cb7f63e13d23a0b03b640852d3a2a49c626d56f3cc469916f61ea03dcf86fba2.jpg](../iclr_results/3526_Video Action Differencing/tables/cb7f63e13d23a0b03b640852d3a2a49c626d56f3cc469916f61ea03dcf86fba2.jpg)

![cc1aa9f745455e2674ef21a6333cf122cbd139a1243c74b0e71c8b7b246af462.jpg](../iclr_results/3526_Video Action Differencing/tables/cc1aa9f745455e2674ef21a6333cf122cbd139a1243c74b0e71c8b7b246af462.jpg)

![ce5ea4cf9daefc128afe6891ba63ac3fda1d6d7d8d36e053172784acb8a24d56.jpg](../iclr_results/3526_Video Action Differencing/tables/ce5ea4cf9daefc128afe6891ba63ac3fda1d6d7d8d36e053172784acb8a24d56.jpg)

![e088d50365eb68c73aac46c6c7512f67f2c503bd9d53229f17e25fb56dcbe005.jpg](../iclr_results/3526_Video Action Differencing/tables/e088d50365eb68c73aac46c6c7512f67f2c503bd9d53229f17e25fb56dcbe005.jpg)

![e63dadbccf66175f9b3bb8c42473132746fa74017b2f923a971e02dc0203b32b.jpg](../iclr_results/3526_Video Action Differencing/tables/e63dadbccf66175f9b3bb8c42473132746fa74017b2f923a971e02dc0203b32b.jpg)

![e9c498b15dde8547e37c2fdc46898d6dceb1993eaad69d232630aff4e4ebc9c9.jpg](../iclr_results/3526_Video Action Differencing/tables/e9c498b15dde8547e37c2fdc46898d6dceb1993eaad69d232630aff4e4ebc9c9.jpg)

![fd1c7468f0627290c75e63911aeb5e4daa2c3f25dd17bbc2a6d76a9c4acd059e.jpg](../iclr_results/3526_Video Action Differencing/tables/fd1c7468f0627290c75e63911aeb5e4daa2c3f25dd17bbc2a6d76a9c4acd059e.jpg)

## UniCoTT: A Unified Framework for Structural Chain-of-Thought Distillation


### Images

![67ba5bb9784cc1b11655d1161774475e7d015eeb92cf47872c231526f91c133a.jpg](../iclr_results/3527_UniCoTT_ A Unified Framework for Structural Chain-of-Thought Distillation/images/67ba5bb9784cc1b11655d1161774475e7d015eeb92cf47872c231526f91c133a.jpg)

![9db73114bfd5d282fc6f4455acd01a452108d1af9e81c13dbed876ac745f9b44.jpg](../iclr_results/3527_UniCoTT_ A Unified Framework for Structural Chain-of-Thought Distillation/images/9db73114bfd5d282fc6f4455acd01a452108d1af9e81c13dbed876ac745f9b44.jpg)

![a4baf32e578ff6293d8deb816a973e59b345ab90a3cf652132eb3f4840594a54.jpg](../iclr_results/3527_UniCoTT_ A Unified Framework for Structural Chain-of-Thought Distillation/images/a4baf32e578ff6293d8deb816a973e59b345ab90a3cf652132eb3f4840594a54.jpg)

![a6a5562726e91e41036fe845c93a1f77c8bea13ae07a740b86aa5e45e69c306a.jpg](../iclr_results/3527_UniCoTT_ A Unified Framework for Structural Chain-of-Thought Distillation/images/a6a5562726e91e41036fe845c93a1f77c8bea13ae07a740b86aa5e45e69c306a.jpg)

![ef21c22ca73bc8936ce43ab423dc726f6218f2313faa8ff063ffadbe80d98e68.jpg](../iclr_results/3527_UniCoTT_ A Unified Framework for Structural Chain-of-Thought Distillation/images/ef21c22ca73bc8936ce43ab423dc726f6218f2313faa8ff063ffadbe80d98e68.jpg)

### Tables

![01ac26fb8d3ad7b94b07171cc41d220c1a7eb6573819d27434d388064edd17f8.jpg](../iclr_results/3527_UniCoTT_ A Unified Framework for Structural Chain-of-Thought Distillation/tables/01ac26fb8d3ad7b94b07171cc41d220c1a7eb6573819d27434d388064edd17f8.jpg)

![129a906bdcf9b70f1affd261f5a25f9d5049a34852ebcf507d1dcdc279dffafb.jpg](../iclr_results/3527_UniCoTT_ A Unified Framework for Structural Chain-of-Thought Distillation/tables/129a906bdcf9b70f1affd261f5a25f9d5049a34852ebcf507d1dcdc279dffafb.jpg)

![1fa712235e3daba7e8ac9927e748a82d9a2bd51c7d93acef4b205c37ed8f484c.jpg](../iclr_results/3527_UniCoTT_ A Unified Framework for Structural Chain-of-Thought Distillation/tables/1fa712235e3daba7e8ac9927e748a82d9a2bd51c7d93acef4b205c37ed8f484c.jpg)

![22363ab33258fca2af84353af1a67d948cec49bcaed83b870c7973d8a112ba1f.jpg](../iclr_results/3527_UniCoTT_ A Unified Framework for Structural Chain-of-Thought Distillation/tables/22363ab33258fca2af84353af1a67d948cec49bcaed83b870c7973d8a112ba1f.jpg)

![26228f736e37426a1e1f721f382d94c2acffd82d7b7620db1de3dfdc90b61225.jpg](../iclr_results/3527_UniCoTT_ A Unified Framework for Structural Chain-of-Thought Distillation/tables/26228f736e37426a1e1f721f382d94c2acffd82d7b7620db1de3dfdc90b61225.jpg)

![2abc7c0288188dffa8d1a5b9c617b68577ce4fc2169442f9d74a4b51fcef6c60.jpg](../iclr_results/3527_UniCoTT_ A Unified Framework for Structural Chain-of-Thought Distillation/tables/2abc7c0288188dffa8d1a5b9c617b68577ce4fc2169442f9d74a4b51fcef6c60.jpg)

![3362ea7d56a710891fec0b99ab9cce67c93bbadde523ffe4c2c4f8bd549ac1aa.jpg](../iclr_results/3527_UniCoTT_ A Unified Framework for Structural Chain-of-Thought Distillation/tables/3362ea7d56a710891fec0b99ab9cce67c93bbadde523ffe4c2c4f8bd549ac1aa.jpg)

![47ed19b7487f8c60166e4e0d85107191a83a5883aa13e91c221a2ff4fa92a39f.jpg](../iclr_results/3527_UniCoTT_ A Unified Framework for Structural Chain-of-Thought Distillation/tables/47ed19b7487f8c60166e4e0d85107191a83a5883aa13e91c221a2ff4fa92a39f.jpg)

![69e5b583382e6fc6779737839136cb41e793e5ddc8d2882651a3a11a44b74584.jpg](../iclr_results/3527_UniCoTT_ A Unified Framework for Structural Chain-of-Thought Distillation/tables/69e5b583382e6fc6779737839136cb41e793e5ddc8d2882651a3a11a44b74584.jpg)

![7a443b354e0e9babf7825b81506077733341cc6f40022abeac0d67118572fbfa.jpg](../iclr_results/3527_UniCoTT_ A Unified Framework for Structural Chain-of-Thought Distillation/tables/7a443b354e0e9babf7825b81506077733341cc6f40022abeac0d67118572fbfa.jpg)

![8f6f3242e50c2909f96baaa7fe07281cad78707933cb5b84406e9aa143bc9f58.jpg](../iclr_results/3527_UniCoTT_ A Unified Framework for Structural Chain-of-Thought Distillation/tables/8f6f3242e50c2909f96baaa7fe07281cad78707933cb5b84406e9aa143bc9f58.jpg)

![afa062432969e7dbd37b75a62d6b5c0c0ec3ba66f0548b53cb5b28eaa397cd48.jpg](../iclr_results/3527_UniCoTT_ A Unified Framework for Structural Chain-of-Thought Distillation/tables/afa062432969e7dbd37b75a62d6b5c0c0ec3ba66f0548b53cb5b28eaa397cd48.jpg)

![b2d9e51bb893eba70e2354b4e5162166c93765ac0c57407f9bb149c6ee23277f.jpg](../iclr_results/3527_UniCoTT_ A Unified Framework for Structural Chain-of-Thought Distillation/tables/b2d9e51bb893eba70e2354b4e5162166c93765ac0c57407f9bb149c6ee23277f.jpg)

![c28b4191952fa81b79cefa4e2b0802cc56e9a8730452009a721fb40790a293e9.jpg](../iclr_results/3527_UniCoTT_ A Unified Framework for Structural Chain-of-Thought Distillation/tables/c28b4191952fa81b79cefa4e2b0802cc56e9a8730452009a721fb40790a293e9.jpg)

![ce6488daa5962fd3f6702e2b974ad5aadc257ccdd6d703b0813dccd8f43c975a.jpg](../iclr_results/3527_UniCoTT_ A Unified Framework for Structural Chain-of-Thought Distillation/tables/ce6488daa5962fd3f6702e2b974ad5aadc257ccdd6d703b0813dccd8f43c975a.jpg)

![cef40d365a4634eca6d585d39d18d8e667039adb2f88a3a08c7f06ad48f416da.jpg](../iclr_results/3527_UniCoTT_ A Unified Framework for Structural Chain-of-Thought Distillation/tables/cef40d365a4634eca6d585d39d18d8e667039adb2f88a3a08c7f06ad48f416da.jpg)

![d8c938fe0f99c668405e41b029a6a13450f2c76bb1964b1f5435b95fc25969b0.jpg](../iclr_results/3527_UniCoTT_ A Unified Framework for Structural Chain-of-Thought Distillation/tables/d8c938fe0f99c668405e41b029a6a13450f2c76bb1964b1f5435b95fc25969b0.jpg)

![e9b47ef7c32f922a582114341e64e5588e920213dd8b9fa12576edf8746b6db1.jpg](../iclr_results/3527_UniCoTT_ A Unified Framework for Structural Chain-of-Thought Distillation/tables/e9b47ef7c32f922a582114341e64e5588e920213dd8b9fa12576edf8746b6db1.jpg)

## Selective Label Enhancement Learning for Test-Time Adaptation


### Images

![1496f850ed3b9d013e7a2fef9f48c01a45bda8077c09c487580c99ce84281f25.jpg](../iclr_results/3528_Selective Label Enhancement Learning for Test-Time Adaptation/images/1496f850ed3b9d013e7a2fef9f48c01a45bda8077c09c487580c99ce84281f25.jpg)

![263a0bf4d863349f4a2d5023ddc2f9ad8cc8adbd6e2975aa1f2e58c7ebf694f7.jpg](../iclr_results/3528_Selective Label Enhancement Learning for Test-Time Adaptation/images/263a0bf4d863349f4a2d5023ddc2f9ad8cc8adbd6e2975aa1f2e58c7ebf694f7.jpg)

![9b8053931eedd09af0ad3df14f0013645e5315e0f25bc8d532ba4060b1f3f7ec.jpg](../iclr_results/3528_Selective Label Enhancement Learning for Test-Time Adaptation/images/9b8053931eedd09af0ad3df14f0013645e5315e0f25bc8d532ba4060b1f3f7ec.jpg)

![afb20fbc55837b80f1ebe47fcce79172982eba0cec08fc893a9a46d132e16d58.jpg](../iclr_results/3528_Selective Label Enhancement Learning for Test-Time Adaptation/images/afb20fbc55837b80f1ebe47fcce79172982eba0cec08fc893a9a46d132e16d58.jpg)

### Tables

![22abf94ac10f1a0bb24f031b2167bbe0a23a8b709704e07f82017fd1fca25378.jpg](../iclr_results/3528_Selective Label Enhancement Learning for Test-Time Adaptation/tables/22abf94ac10f1a0bb24f031b2167bbe0a23a8b709704e07f82017fd1fca25378.jpg)

![2b384dadc24f4cf5d11c0a81f33d64d6d0d0f8e95c2eec23d95101f244844e35.jpg](../iclr_results/3528_Selective Label Enhancement Learning for Test-Time Adaptation/tables/2b384dadc24f4cf5d11c0a81f33d64d6d0d0f8e95c2eec23d95101f244844e35.jpg)

![37a76e7119defff21beeb8fcbef0900179031eef94539a2bbbacda5fadaa6449.jpg](../iclr_results/3528_Selective Label Enhancement Learning for Test-Time Adaptation/tables/37a76e7119defff21beeb8fcbef0900179031eef94539a2bbbacda5fadaa6449.jpg)

![39396b05674ebf00cc6739e6dce0d0514e934271eadb175ea5689360d83b95bf.jpg](../iclr_results/3528_Selective Label Enhancement Learning for Test-Time Adaptation/tables/39396b05674ebf00cc6739e6dce0d0514e934271eadb175ea5689360d83b95bf.jpg)

![45906b2c15df55e4847cc1ad4da8bd650329a7dd0f541f9e8ea1b3ee8e13d227.jpg](../iclr_results/3528_Selective Label Enhancement Learning for Test-Time Adaptation/tables/45906b2c15df55e4847cc1ad4da8bd650329a7dd0f541f9e8ea1b3ee8e13d227.jpg)

![48f1b2bb1575ec72962f8873fcca2fedc5a88ee78790d3bedb5fd90b6c3c0021.jpg](../iclr_results/3528_Selective Label Enhancement Learning for Test-Time Adaptation/tables/48f1b2bb1575ec72962f8873fcca2fedc5a88ee78790d3bedb5fd90b6c3c0021.jpg)

![6771c0d6e1098daf2ef406fe414aa3741c2b66b78c65cf257075605f2486ec61.jpg](../iclr_results/3528_Selective Label Enhancement Learning for Test-Time Adaptation/tables/6771c0d6e1098daf2ef406fe414aa3741c2b66b78c65cf257075605f2486ec61.jpg)

![76df9754a61178c90fee129378c024cee3c57f684bc977aede0e0f6aa4ae0eda.jpg](../iclr_results/3528_Selective Label Enhancement Learning for Test-Time Adaptation/tables/76df9754a61178c90fee129378c024cee3c57f684bc977aede0e0f6aa4ae0eda.jpg)

![7fe98d99569e1659934472b18797d243906d60c0ecab150a6f5b3439245ac203.jpg](../iclr_results/3528_Selective Label Enhancement Learning for Test-Time Adaptation/tables/7fe98d99569e1659934472b18797d243906d60c0ecab150a6f5b3439245ac203.jpg)

![9e68f7a476ad38bbd801897d6fb23eb292fa49556ee4032b040473014b501db7.jpg](../iclr_results/3528_Selective Label Enhancement Learning for Test-Time Adaptation/tables/9e68f7a476ad38bbd801897d6fb23eb292fa49556ee4032b040473014b501db7.jpg)

![bfe96af0886ba8f946ae244490af0c789aa50497459ab824554a4c0f7b5a6674.jpg](../iclr_results/3528_Selective Label Enhancement Learning for Test-Time Adaptation/tables/bfe96af0886ba8f946ae244490af0c789aa50497459ab824554a4c0f7b5a6674.jpg)

![e3ef212e2396af9bcb08b8e2658fe1fc7f919373b1ab214b930994d916ceae88.jpg](../iclr_results/3528_Selective Label Enhancement Learning for Test-Time Adaptation/tables/e3ef212e2396af9bcb08b8e2658fe1fc7f919373b1ab214b930994d916ceae88.jpg)

![e568eda0201f89c055b7a83c57797eb59fd84c189e522df8c31a6606192f3b61.jpg](../iclr_results/3528_Selective Label Enhancement Learning for Test-Time Adaptation/tables/e568eda0201f89c055b7a83c57797eb59fd84c189e522df8c31a6606192f3b61.jpg)

## Multi-Label Node Classification with Label Influence Propagation


### Images

![088c0514b1aaf94e981c8c2bf518b0fa9fc0f288852f04f7dc6d9a9bd59e655b.jpg](../iclr_results/3529_Multi-Label Node Classification with Label Influence Propagation/images/088c0514b1aaf94e981c8c2bf518b0fa9fc0f288852f04f7dc6d9a9bd59e655b.jpg)

![372ae97ae6e8503a6e924a1f565f2233cddf8ffd9fed27974f237c7f2f8d2440.jpg](../iclr_results/3529_Multi-Label Node Classification with Label Influence Propagation/images/372ae97ae6e8503a6e924a1f565f2233cddf8ffd9fed27974f237c7f2f8d2440.jpg)

![39adb3237199a4d1a6f6d5622320f88fe0c27e6d558c36c47c1813b835841b07.jpg](../iclr_results/3529_Multi-Label Node Classification with Label Influence Propagation/images/39adb3237199a4d1a6f6d5622320f88fe0c27e6d558c36c47c1813b835841b07.jpg)

![93d33abc5408926bd9ec14738aae5d03328e0ab5f8f6e8bd699b90ad278896e7.jpg](../iclr_results/3529_Multi-Label Node Classification with Label Influence Propagation/images/93d33abc5408926bd9ec14738aae5d03328e0ab5f8f6e8bd699b90ad278896e7.jpg)

![a45f50d1b1a2179a6d1b9ea914135d4c372841fae38f5ebddb4702de6f9fa4cb.jpg](../iclr_results/3529_Multi-Label Node Classification with Label Influence Propagation/images/a45f50d1b1a2179a6d1b9ea914135d4c372841fae38f5ebddb4702de6f9fa4cb.jpg)

![b475eda8a2407c86f17d4da585b3a214344cf7e6977498d6641c21c1567a39b4.jpg](../iclr_results/3529_Multi-Label Node Classification with Label Influence Propagation/images/b475eda8a2407c86f17d4da585b3a214344cf7e6977498d6641c21c1567a39b4.jpg)

![f539c143bebc3904c82cf256231b776b6deff50ccc5406be0564a02f49003e67.jpg](../iclr_results/3529_Multi-Label Node Classification with Label Influence Propagation/images/f539c143bebc3904c82cf256231b776b6deff50ccc5406be0564a02f49003e67.jpg)

### Tables

![2019858b547db74667a5528eee951e13f8a3369d4848926f7776c4ef7918cae7.jpg](../iclr_results/3529_Multi-Label Node Classification with Label Influence Propagation/tables/2019858b547db74667a5528eee951e13f8a3369d4848926f7776c4ef7918cae7.jpg)

![2d757797bef996e7ff65d4fa2a63edce594182a22efd548f5d6833fe33dbbd10.jpg](../iclr_results/3529_Multi-Label Node Classification with Label Influence Propagation/tables/2d757797bef996e7ff65d4fa2a63edce594182a22efd548f5d6833fe33dbbd10.jpg)

![556c0ef1a43aef8cff393c60e094157c15eaae5bc1d2bcc2f0ac04e6928d18aa.jpg](../iclr_results/3529_Multi-Label Node Classification with Label Influence Propagation/tables/556c0ef1a43aef8cff393c60e094157c15eaae5bc1d2bcc2f0ac04e6928d18aa.jpg)

![5acdc8e4901ba76d57d79ded08e743f005438f36a7d75305d46eb026a43263eb.jpg](../iclr_results/3529_Multi-Label Node Classification with Label Influence Propagation/tables/5acdc8e4901ba76d57d79ded08e743f005438f36a7d75305d46eb026a43263eb.jpg)

![6d09d0e9f9b9414cbd27d9c03a4eadc4fd0655922c1c9116ce8493744c08cdd4.jpg](../iclr_results/3529_Multi-Label Node Classification with Label Influence Propagation/tables/6d09d0e9f9b9414cbd27d9c03a4eadc4fd0655922c1c9116ce8493744c08cdd4.jpg)

![7e6c78e7767ed1e10db87dd7e5a866eed64c5b0632a1b9f5568ba8dfb28809b5.jpg](../iclr_results/3529_Multi-Label Node Classification with Label Influence Propagation/tables/7e6c78e7767ed1e10db87dd7e5a866eed64c5b0632a1b9f5568ba8dfb28809b5.jpg)

![a331e1111292cff122d1e7a51747e88774ebf7eec410acf8a0a073b75e65efbd.jpg](../iclr_results/3529_Multi-Label Node Classification with Label Influence Propagation/tables/a331e1111292cff122d1e7a51747e88774ebf7eec410acf8a0a073b75e65efbd.jpg)

![b22eba2a5382d8bf73501e2d0d54d5fddf13fd9d9936dfd1f1d16da22a7eb2fc.jpg](../iclr_results/3529_Multi-Label Node Classification with Label Influence Propagation/tables/b22eba2a5382d8bf73501e2d0d54d5fddf13fd9d9936dfd1f1d16da22a7eb2fc.jpg)

![c9883328f0ac67c45c65c6cbc6cb225c488a1ef948565ebd8e2f5bb4ecb4b295.jpg](../iclr_results/3529_Multi-Label Node Classification with Label Influence Propagation/tables/c9883328f0ac67c45c65c6cbc6cb225c488a1ef948565ebd8e2f5bb4ecb4b295.jpg)

![db5cce0ca3ee54f9adceddc6f84214fc7d7d2e1a1d802bd81a45c5e95a05be8a.jpg](../iclr_results/3529_Multi-Label Node Classification with Label Influence Propagation/tables/db5cce0ca3ee54f9adceddc6f84214fc7d7d2e1a1d802bd81a45c5e95a05be8a.jpg)

![f1b7754e06ab4f9f5005be39689e4778a1bc5a4639846a1822c94cec3debdbf4.jpg](../iclr_results/3529_Multi-Label Node Classification with Label Influence Propagation/tables/f1b7754e06ab4f9f5005be39689e4778a1bc5a4639846a1822c94cec3debdbf4.jpg)

## CREMA: Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion


### Images

![066505ad8bbe106b7a95e90903dbcec088aa20905666056722907da46e6aa997.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/images/066505ad8bbe106b7a95e90903dbcec088aa20905666056722907da46e6aa997.jpg)

![962338ff56cf7b7f93f039710ce053d49cfab36a2f75121f78d9bd44b8d7c9cd.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/images/962338ff56cf7b7f93f039710ce053d49cfab36a2f75121f78d9bd44b8d7c9cd.jpg)

![af08e8509445256bc5de7dc25318a0e28cbe5f0e36e0bd86a10fe752178922ef.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/images/af08e8509445256bc5de7dc25318a0e28cbe5f0e36e0bd86a10fe752178922ef.jpg)

![b15efe7e9ca312ca58f0d3e0b14b36944232192bbd812c15eee3d7726f5183d4.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/images/b15efe7e9ca312ca58f0d3e0b14b36944232192bbd812c15eee3d7726f5183d4.jpg)

![dd3be80532e3daa1a29bbd40a0ea19ea90df821eefe8cf8ceb944bfb0df8d3f7.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/images/dd3be80532e3daa1a29bbd40a0ea19ea90df821eefe8cf8ceb944bfb0df8d3f7.jpg)

### Tables

![1eb52e511431b24277029541f710df93b4780b24cafa548c0c7de0f9bd149eb9.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/tables/1eb52e511431b24277029541f710df93b4780b24cafa548c0c7de0f9bd149eb9.jpg)

![2a588018a34e8ff9600de1977720b7dc94d5f6504fdfdff1bb58c34544218d23.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/tables/2a588018a34e8ff9600de1977720b7dc94d5f6504fdfdff1bb58c34544218d23.jpg)

![3b009099b4a3a41ed2d5edd0612ca3f57665b37d2985044f0b74ec7e8349a6f4.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/tables/3b009099b4a3a41ed2d5edd0612ca3f57665b37d2985044f0b74ec7e8349a6f4.jpg)

![4319524ab65a9ccc8e55ec17d0d69db6e063d80a786a2c45d76d9365411a7296.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/tables/4319524ab65a9ccc8e55ec17d0d69db6e063d80a786a2c45d76d9365411a7296.jpg)

![5fc99418d0fb33f91596968d9e73f13901e6734dbaa844fc899e660fe543d84b.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/tables/5fc99418d0fb33f91596968d9e73f13901e6734dbaa844fc899e660fe543d84b.jpg)

![61f9c88faff192af12ad9820f363efe730b2dfb1b235d0181cbc630a052f1caa.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/tables/61f9c88faff192af12ad9820f363efe730b2dfb1b235d0181cbc630a052f1caa.jpg)

![77749882146dabce00979188fb7a4bd5ca954078b75f312d3fb538c78e8b64f6.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/tables/77749882146dabce00979188fb7a4bd5ca954078b75f312d3fb538c78e8b64f6.jpg)

![8313148cc0f36ef2d8ac8c468c50d9bd382b54796fd02785c3067a530664d483.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/tables/8313148cc0f36ef2d8ac8c468c50d9bd382b54796fd02785c3067a530664d483.jpg)

![83da78b92719575ac20a5d6130dc1af0654ca20d3b5af2c6667223d7cd5944ae.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/tables/83da78b92719575ac20a5d6130dc1af0654ca20d3b5af2c6667223d7cd5944ae.jpg)

![93f0528d0bd5df28f9e41ab7f50a22dceef2a94bcb81e6b77b853e0b9d7c0b9f.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/tables/93f0528d0bd5df28f9e41ab7f50a22dceef2a94bcb81e6b77b853e0b9d7c0b9f.jpg)

![9809b7a777a00a1da7fbc63aa5545b9b545bff493db57b79d5b38de5a3f21af5.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/tables/9809b7a777a00a1da7fbc63aa5545b9b545bff493db57b79d5b38de5a3f21af5.jpg)

![af8889fd28b2402b2c4f97f361e47e488353b1f7ccabea6b2252be19f2e442c9.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/tables/af8889fd28b2402b2c4f97f361e47e488353b1f7ccabea6b2252be19f2e442c9.jpg)

![c05aa220b7d0b62ad4e0f9892d517ee534410e79f8ffcb14338ffb956321091a.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/tables/c05aa220b7d0b62ad4e0f9892d517ee534410e79f8ffcb14338ffb956321091a.jpg)

![d29bed9b5a8bf5bff760f096da31e9556cae6b4b905c3d69ef860c0c2dcba7e2.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/tables/d29bed9b5a8bf5bff760f096da31e9556cae6b4b905c3d69ef860c0c2dcba7e2.jpg)

![d8069ebecffcc2e0f399156ace203de4d74edc362e569802f1b794d8c0e9598b.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/tables/d8069ebecffcc2e0f399156ace203de4d74edc362e569802f1b794d8c0e9598b.jpg)

![dfd29b99189d602f00809ea20017f4b1d459e2fa6a6f75db32378bff25bd919e.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/tables/dfd29b99189d602f00809ea20017f4b1d459e2fa6a6f75db32378bff25bd919e.jpg)

![e234dafd46b007e5b895df6be64fa89e029983a318399c62082920b54cfce003.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/tables/e234dafd46b007e5b895df6be64fa89e029983a318399c62082920b54cfce003.jpg)

![e4e762a6ad5f74eb862cb98ce6559ee30c2c63f4fd17eda17d78e34634dfdf69.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/tables/e4e762a6ad5f74eb862cb98ce6559ee30c2c63f4fd17eda17d78e34634dfdf69.jpg)

![e7b03ff5e2cf47f4c645355e133a89ddb747fe17cec7571e6447608b740e7ee4.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/tables/e7b03ff5e2cf47f4c645355e133a89ddb747fe17cec7571e6447608b740e7ee4.jpg)

![e9aedcf271207b10eb74105b2a55aa94b6bf51b49516127db12721f5c050f5fc.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/tables/e9aedcf271207b10eb74105b2a55aa94b6bf51b49516127db12721f5c050f5fc.jpg)

![fb87cbe5433e4d8792717714103468706c6c84904ac9bf20c936132eb0375eb8.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/tables/fb87cbe5433e4d8792717714103468706c6c84904ac9bf20c936132eb0375eb8.jpg)

![feb583d1cdc4d717dbaf32c38581b05ea82c19e86de18894847b8d7d95e6236e.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/tables/feb583d1cdc4d717dbaf32c38581b05ea82c19e86de18894847b8d7d95e6236e.jpg)

![ff977b7099713e5f5296f1fe8e48a0b8298e95620135ba37aabc2b838549bf44.jpg](../iclr_results/3530_CREMA_ Generalizable and Efficient Video-Language Reasoning via Multimodal Modular Fusion/tables/ff977b7099713e5f5296f1fe8e48a0b8298e95620135ba37aabc2b838549bf44.jpg)

## Learn-by-interact: A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments


### Images

![0515a4c9ee8db58fc57acacfa76c7fde9fcb8f2f862fafefe25b707bba7f300a.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/0515a4c9ee8db58fc57acacfa76c7fde9fcb8f2f862fafefe25b707bba7f300a.jpg)

![10f5c4e45c52b15b9f1efa4362431c0d05f13c02d9bdba6a31290a1f5c6c001a.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/10f5c4e45c52b15b9f1efa4362431c0d05f13c02d9bdba6a31290a1f5c6c001a.jpg)

![179ffe3b3df46ee9b4bfe3ed15bf9ec9f5d23ebd1c5a913088efc71d4da1eac7.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/179ffe3b3df46ee9b4bfe3ed15bf9ec9f5d23ebd1c5a913088efc71d4da1eac7.jpg)

![17cd70801807cb1a5e2527edfccd3f635b13e14354cd7b6cfd8e8d2a93cadd31.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/17cd70801807cb1a5e2527edfccd3f635b13e14354cd7b6cfd8e8d2a93cadd31.jpg)

![1818e352bfb380b10f8b480a382af0f3e487b753dbc3c9b9a04511b1ad58d7fd.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/1818e352bfb380b10f8b480a382af0f3e487b753dbc3c9b9a04511b1ad58d7fd.jpg)

![1c424381cb0a9de931437b0e0d74e064cdf906f9240b3662e033291107745424.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/1c424381cb0a9de931437b0e0d74e064cdf906f9240b3662e033291107745424.jpg)

![2345bbdc8c8d3b310e98fdfa5f48220fd54ab45402d37e00da9097a04a5d8444.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/2345bbdc8c8d3b310e98fdfa5f48220fd54ab45402d37e00da9097a04a5d8444.jpg)

![36058103e7504c664f27ecd005ac872a8b793673c4075515f7f623de1069632a.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/36058103e7504c664f27ecd005ac872a8b793673c4075515f7f623de1069632a.jpg)

![3f27199be6cd2c6105c668c9c1dd4d52c7c753cf89aa54714b32e7a03bf7c77c.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/3f27199be6cd2c6105c668c9c1dd4d52c7c753cf89aa54714b32e7a03bf7c77c.jpg)

![3fd0ec7a722d904f36fd0d5c0b0bf5c526db6bf6bb6662a06e7402f7d089301d.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/3fd0ec7a722d904f36fd0d5c0b0bf5c526db6bf6bb6662a06e7402f7d089301d.jpg)

![412fae2f076fc36fa9c132eb26d0a569bc34a78de68c2965d494810dcf7634bc.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/412fae2f076fc36fa9c132eb26d0a569bc34a78de68c2965d494810dcf7634bc.jpg)

![4196dec83c8243c4fdf928fc60562d643c8ad6a4d67430b0203e36eec3f9b912.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/4196dec83c8243c4fdf928fc60562d643c8ad6a4d67430b0203e36eec3f9b912.jpg)

![5214567066d35ecd12091e9388147c58a935885b8561532df0ec0291fd996832.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/5214567066d35ecd12091e9388147c58a935885b8561532df0ec0291fd996832.jpg)

![55b90f9c35802a83ed9de718fb6fbd15bb8fe1a0909ed089db8058eb85bd5d8e.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/55b90f9c35802a83ed9de718fb6fbd15bb8fe1a0909ed089db8058eb85bd5d8e.jpg)

![599c1b76e7de366251e7742db682087862e57eb4f84d093a894cd1084d2e0ed8.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/599c1b76e7de366251e7742db682087862e57eb4f84d093a894cd1084d2e0ed8.jpg)

![5b11a3604162933bb29416086faaedb3fcad2526f837e4b0d5bb4e49ab21c406.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/5b11a3604162933bb29416086faaedb3fcad2526f837e4b0d5bb4e49ab21c406.jpg)

![5cda2f8c7f513a3cb091561044be6a9406501ed0bfe2aec37b419c014d52b34b.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/5cda2f8c7f513a3cb091561044be6a9406501ed0bfe2aec37b419c014d52b34b.jpg)

![5d20f97a2cd474daea7d615058be4daddb0ac16bad91a02e618b6fbb132ec7b4.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/5d20f97a2cd474daea7d615058be4daddb0ac16bad91a02e618b6fbb132ec7b4.jpg)

![67acedce0dcf175d2b5a1727b7824a16ceb97c0f854e03b2c694ad64b539c39c.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/67acedce0dcf175d2b5a1727b7824a16ceb97c0f854e03b2c694ad64b539c39c.jpg)

![6bbdcbc5c3972eaf34f0c114ff80c0a55ac94a682ba5ac9720fa7da236f33e29.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/6bbdcbc5c3972eaf34f0c114ff80c0a55ac94a682ba5ac9720fa7da236f33e29.jpg)

![6c3a807f4b1833a089c6407f8d71a2089b1f5cb985c4dabb4fd1e790e2af9ded.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/6c3a807f4b1833a089c6407f8d71a2089b1f5cb985c4dabb4fd1e790e2af9ded.jpg)

![6d738d54d0d2751a7b8494762520757bdb7ebce77644d07e1995b0a95583082e.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/6d738d54d0d2751a7b8494762520757bdb7ebce77644d07e1995b0a95583082e.jpg)

![7e3766f0b6057428f444eaef26714bb66c5b01e9e4db8f136a0054da501ea8bd.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/7e3766f0b6057428f444eaef26714bb66c5b01e9e4db8f136a0054da501ea8bd.jpg)

![81e6309e81108066f1dbe84c677ae043b0e5784c51b68df2ea148185ce479d1c.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/81e6309e81108066f1dbe84c677ae043b0e5784c51b68df2ea148185ce479d1c.jpg)

![8bacb0a8567d7a41182d4c4a81ac540654b1e34a47eed7d26d43dbdaeafef220.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/8bacb0a8567d7a41182d4c4a81ac540654b1e34a47eed7d26d43dbdaeafef220.jpg)

![8bf3f2ccd9e7e5e25e8293b3579a3fbb1a4cb860d9ccf260a9de453bfe24f851.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/8bf3f2ccd9e7e5e25e8293b3579a3fbb1a4cb860d9ccf260a9de453bfe24f851.jpg)

![8e92d1a7befe3cd88689eb486617bd2e8696ccfdd55d7f29c9eafc50588e1e67.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/8e92d1a7befe3cd88689eb486617bd2e8696ccfdd55d7f29c9eafc50588e1e67.jpg)

![9379e71a268059aa7cc04e0bf6b64ba09dc37faf895006c25b23d64704b2d145.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/9379e71a268059aa7cc04e0bf6b64ba09dc37faf895006c25b23d64704b2d145.jpg)

![9dd8dd2e4b1e8c15d5b59ea5daba9db8dcfe36ac645ac56202cd1645e29babb8.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/9dd8dd2e4b1e8c15d5b59ea5daba9db8dcfe36ac645ac56202cd1645e29babb8.jpg)

![9df9e41362818a2bb9bcae0178e940dfe0fb484fbff6bb9daeb59ec67b107600.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/9df9e41362818a2bb9bcae0178e940dfe0fb484fbff6bb9daeb59ec67b107600.jpg)

![a173327982b40c402e9cd4b3b9248ddb0022bfbd293e2c97e40073f17db27ce8.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/a173327982b40c402e9cd4b3b9248ddb0022bfbd293e2c97e40073f17db27ce8.jpg)

![a558a5765093dcdcaf3584d8f8f13aa7593743ca100a57c29e8279f39b2eb974.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/a558a5765093dcdcaf3584d8f8f13aa7593743ca100a57c29e8279f39b2eb974.jpg)

![a73ffe175bd391a707624d0ef6b3adcde0676dec32d4eb6d3a94338b65058950.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/a73ffe175bd391a707624d0ef6b3adcde0676dec32d4eb6d3a94338b65058950.jpg)

![a77e131dcb45c7841ecfc6d4aff6ae7c0b85c22e89779735fd8bcf18c9ba2f18.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/a77e131dcb45c7841ecfc6d4aff6ae7c0b85c22e89779735fd8bcf18c9ba2f18.jpg)

![aa65ff3264d946e55c6d29a8457cea03499851ac0ebdf6d6979e59390c3ade6c.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/aa65ff3264d946e55c6d29a8457cea03499851ac0ebdf6d6979e59390c3ade6c.jpg)

![b48b2de5898606f239dca2c2319574fd53aa0f8828b573468edf44869909c6e2.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/b48b2de5898606f239dca2c2319574fd53aa0f8828b573468edf44869909c6e2.jpg)

![b767c60c741ae82f530d8abbf69961abcea21b588d707aaace230b4a3385cb06.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/b767c60c741ae82f530d8abbf69961abcea21b588d707aaace230b4a3385cb06.jpg)

![ba042a44e23e1ac78653872a57077b47e5eb843b6046fa8a2df0635ec308286f.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/ba042a44e23e1ac78653872a57077b47e5eb843b6046fa8a2df0635ec308286f.jpg)

![c1b8aa12fdee4845d0de6b5c41fb075784545bbced0a300af6115c3eb2d6a875.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/c1b8aa12fdee4845d0de6b5c41fb075784545bbced0a300af6115c3eb2d6a875.jpg)

![c53bbb622f5c2c03e7e028860087c739f45fb6226a8ecd9fa725f3ccd7ae193b.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/c53bbb622f5c2c03e7e028860087c739f45fb6226a8ecd9fa725f3ccd7ae193b.jpg)

![c982d0b9ee1f27bf9cf84f1838c9bedcffaaaaa4a54033e8eebb0561b15aa450.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/c982d0b9ee1f27bf9cf84f1838c9bedcffaaaaa4a54033e8eebb0561b15aa450.jpg)

![c9d73189a85434562e02c4dd6fc970b5a5c6172df71d700bee54b1b9d43ed315.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/c9d73189a85434562e02c4dd6fc970b5a5c6172df71d700bee54b1b9d43ed315.jpg)

![cbf2eb9dae513c90d3da3297a84d65ff34bed60b394b64607aff05694a8b53a3.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/cbf2eb9dae513c90d3da3297a84d65ff34bed60b394b64607aff05694a8b53a3.jpg)

![d4ef69aeaa80401a672559919ca68495f7ba88ceb64c1f36b38dd52d2881bb2d.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/d4ef69aeaa80401a672559919ca68495f7ba88ceb64c1f36b38dd52d2881bb2d.jpg)

![dc3dc3e760ba3f542cc41f4e6bd7de79e886b2e8e20039394f0921bff77d12de.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/dc3dc3e760ba3f542cc41f4e6bd7de79e886b2e8e20039394f0921bff77d12de.jpg)

![dd09ed5d1ae41cefe5fe59b85e3c77df6995cdcf58c1d4755e6ec711a7c35b3d.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/dd09ed5d1ae41cefe5fe59b85e3c77df6995cdcf58c1d4755e6ec711a7c35b3d.jpg)

![dd11b60217e9ab76ced4165fcce7561e364d0ecdefbb3c8ac435425b339ee5aa.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/dd11b60217e9ab76ced4165fcce7561e364d0ecdefbb3c8ac435425b339ee5aa.jpg)

![dd2de321bcd9c7b508f951357d797329c910648be166ca325f612540a0316be1.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/dd2de321bcd9c7b508f951357d797329c910648be166ca325f612540a0316be1.jpg)

![e4f861733ae0f5628ede6b5189efa34559556bcf9a5e1fb7ab93a6a2b64b5922.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/e4f861733ae0f5628ede6b5189efa34559556bcf9a5e1fb7ab93a6a2b64b5922.jpg)

![e7cc79459478c18efa84e9fcd5405a57044c0face9ebb1df3168237b3ade468e.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/e7cc79459478c18efa84e9fcd5405a57044c0face9ebb1df3168237b3ade468e.jpg)

![eac740404d34bec77f7a2a06d1283e362149968c25508e4a9fc2462c19a6026e.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/eac740404d34bec77f7a2a06d1283e362149968c25508e4a9fc2462c19a6026e.jpg)

![ebdef28014b2113555805cb362d2667a3cd23652c92324edad0fc153987b2542.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/ebdef28014b2113555805cb362d2667a3cd23652c92324edad0fc153987b2542.jpg)

![f35eb1cff9d86add85fd6c46fbbb144d53a18581a347ae00991532dfa1363c19.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/f35eb1cff9d86add85fd6c46fbbb144d53a18581a347ae00991532dfa1363c19.jpg)

![f63cb33a0c1a20a8d9fe40230b617fa32ba2c0d244d4bc0ebd0058a72a3d0c68.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/f63cb33a0c1a20a8d9fe40230b617fa32ba2c0d244d4bc0ebd0058a72a3d0c68.jpg)

![f6b3590539c977a0501bd8ce9bcdd8c6da91df0c50b4afa8d3cbd2376b4713c6.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/images/f6b3590539c977a0501bd8ce9bcdd8c6da91df0c50b4afa8d3cbd2376b4713c6.jpg)

### Tables

![011a34759d618d0c5dc8bf584596d3877bd72709aac2415f25cfa929a1934806.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/tables/011a34759d618d0c5dc8bf584596d3877bd72709aac2415f25cfa929a1934806.jpg)

![0f8502b5afc99a282e408222c7cda52f7b29729fc0a0e9f27720983812047594.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/tables/0f8502b5afc99a282e408222c7cda52f7b29729fc0a0e9f27720983812047594.jpg)

![113f6264eeb9febe250520486f6607edb906f8c0ba7a7c4056928a0694bcb03d.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/tables/113f6264eeb9febe250520486f6607edb906f8c0ba7a7c4056928a0694bcb03d.jpg)

![1987e8abcd29c0fc469b0b89a60f33bb0d1d36e65f2d567e09191acd2ee8c80f.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/tables/1987e8abcd29c0fc469b0b89a60f33bb0d1d36e65f2d567e09191acd2ee8c80f.jpg)

![223e3185aabae4a7ae4c89cb5a814418a327addd4397e7166aa93016d2cdf9e9.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/tables/223e3185aabae4a7ae4c89cb5a814418a327addd4397e7166aa93016d2cdf9e9.jpg)

![5b692a4a0637efc0e4a8e704c37e94bb5ac7c5983b6df7bed2a2ef62ccbc5283.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/tables/5b692a4a0637efc0e4a8e704c37e94bb5ac7c5983b6df7bed2a2ef62ccbc5283.jpg)

![5c3d974ac9e01f8468545a074424d42b0d6b1314044a3d5330ac974e213bc10e.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/tables/5c3d974ac9e01f8468545a074424d42b0d6b1314044a3d5330ac974e213bc10e.jpg)

![64f389938c2eb5f936062e7221126c8de2d7b3d46d34589e27c36da444e8057a.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/tables/64f389938c2eb5f936062e7221126c8de2d7b3d46d34589e27c36da444e8057a.jpg)

![8497dedf64659210c43373cea5a6fe516eedc202bafe0538f95d1d5925a9dfbe.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/tables/8497dedf64659210c43373cea5a6fe516eedc202bafe0538f95d1d5925a9dfbe.jpg)

![853d1dd480bd9304dec76b6b84ef5c3df4a9008cdfb1c960d7f92bbecdcd792a.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/tables/853d1dd480bd9304dec76b6b84ef5c3df4a9008cdfb1c960d7f92bbecdcd792a.jpg)

![8995778f39ef9b034de6ff8286451817ba7a8a290926ced60933cb49f002dcd0.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/tables/8995778f39ef9b034de6ff8286451817ba7a8a290926ced60933cb49f002dcd0.jpg)

![9b1c1ed4c736da3a8aaeb8b2993f2fdb01bd4bc3c3e90c076c40da6edae7285c.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/tables/9b1c1ed4c736da3a8aaeb8b2993f2fdb01bd4bc3c3e90c076c40da6edae7285c.jpg)

![a9fdda4c44466b08eb7477b54bcfc3e8941ba8f4a4305e2fd4df9d2abbab4cd5.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/tables/a9fdda4c44466b08eb7477b54bcfc3e8941ba8f4a4305e2fd4df9d2abbab4cd5.jpg)

![b58214401b407203d82d71f606b54a92fb3fd13a2be1bfdeacdd0b211e80e861.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/tables/b58214401b407203d82d71f606b54a92fb3fd13a2be1bfdeacdd0b211e80e861.jpg)

![bb65cb897d783b40169cffe449dd387eb7ab10ccf2a695da5f334dbb260e1325.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/tables/bb65cb897d783b40169cffe449dd387eb7ab10ccf2a695da5f334dbb260e1325.jpg)

![c02d50831ffef12521b9eacaa540d891a6135521eacb19b113dd5dde19d3fbc4.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/tables/c02d50831ffef12521b9eacaa540d891a6135521eacb19b113dd5dde19d3fbc4.jpg)

![c30df3aebffe8d8826f2d064860e9b0cac0a1a5102f0c9682de096b79e7fa34f.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/tables/c30df3aebffe8d8826f2d064860e9b0cac0a1a5102f0c9682de096b79e7fa34f.jpg)

![c8417d988a8e779697785f3155756e5721a2062c9af6b5ef1d358fa888f7bc25.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/tables/c8417d988a8e779697785f3155756e5721a2062c9af6b5ef1d358fa888f7bc25.jpg)

![da0ef68d885becf6f4ce2cf1eafbc8ac521ad7ebdf79c45963b22ea305b4cb83.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/tables/da0ef68d885becf6f4ce2cf1eafbc8ac521ad7ebdf79c45963b22ea305b4cb83.jpg)

![e607af936ed172be2ee3df8815a24d2f1a2f8307f43ca6ca41e1e467881d93c8.jpg](../iclr_results/3531_Learn-by-interact_ A Data-Centric Framework For Self-Adaptive Agents in Realistic Environments/tables/e607af936ed172be2ee3df8815a24d2f1a2f8307f43ca6ca41e1e467881d93c8.jpg)

## From Pixels to Tokens: Byte-Pair Encoding on Quantized Visual Modalities

### Images

![2228a29e95b5e3c82c20dbe0272393aad730a38c9dcc72502728def10cdf7c19.jpg](../iclr_results/3532_From Pixels to Tokens_ Byte-Pair Encoding on Quantized Visual Modalities/images/2228a29e95b5e3c82c20dbe0272393aad730a38c9dcc72502728def10cdf7c19.jpg)

![66167eccd4b951efa888491c85d4cf5005bc96a28aa709045ce36eea27ec2091.jpg](../iclr_results/3532_From Pixels to Tokens_ Byte-Pair Encoding on Quantized Visual Modalities/images/66167eccd4b951efa888491c85d4cf5005bc96a28aa709045ce36eea27ec2091.jpg)

![7bddd6ea6c852c109b2e2b493e802e168d34e37edba4d7ba38c38a838ea71677.jpg](../iclr_results/3532_From Pixels to Tokens_ Byte-Pair Encoding on Quantized Visual Modalities/images/7bddd6ea6c852c109b2e2b493e802e168d34e37edba4d7ba38c38a838ea71677.jpg)

![803a743060ed182df2247ca3095b8cb5180db14dfdb5479682b1b19068aff51a.jpg](../iclr_results/3532_From Pixels to Tokens_ Byte-Pair Encoding on Quantized Visual Modalities/images/803a743060ed182df2247ca3095b8cb5180db14dfdb5479682b1b19068aff51a.jpg)

### Tables

![13f9559ca36c1e865a90e385ba35e767c2373ab42a651b0ce4951050c418d250.jpg](../iclr_results/3532_From Pixels to Tokens_ Byte-Pair Encoding on Quantized Visual Modalities/tables/13f9559ca36c1e865a90e385ba35e767c2373ab42a651b0ce4951050c418d250.jpg)

![1abd1d882d56989dc2d7d43682e83ba453213b5d6860dbe7bc92d0c334f13e78.jpg](../iclr_results/3532_From Pixels to Tokens_ Byte-Pair Encoding on Quantized Visual Modalities/tables/1abd1d882d56989dc2d7d43682e83ba453213b5d6860dbe7bc92d0c334f13e78.jpg)

![6cf04f2bb4644ac3870da4c3cd8a432353800b059d837f6d3c450b0f252e0730.jpg](../iclr_results/3532_From Pixels to Tokens_ Byte-Pair Encoding on Quantized Visual Modalities/tables/6cf04f2bb4644ac3870da4c3cd8a432353800b059d837f6d3c450b0f252e0730.jpg)

![775edf9751f8e1dd8d8be640d651fff99fb9310e340f3f7d1b02761a6a372f86.jpg](../iclr_results/3532_From Pixels to Tokens_ Byte-Pair Encoding on Quantized Visual Modalities/tables/775edf9751f8e1dd8d8be640d651fff99fb9310e340f3f7d1b02761a6a372f86.jpg)

![7a26f2dcb852d568fc4ee8734cfe29620ccceb20312b59148ed678aa098f0b80.jpg](../iclr_results/3532_From Pixels to Tokens_ Byte-Pair Encoding on Quantized Visual Modalities/tables/7a26f2dcb852d568fc4ee8734cfe29620ccceb20312b59148ed678aa098f0b80.jpg)

![7a522ab0b1e5bb9452f33a2e16258fd10b17067fda7eda7c28b3ab896e1a6763.jpg](../iclr_results/3532_From Pixels to Tokens_ Byte-Pair Encoding on Quantized Visual Modalities/tables/7a522ab0b1e5bb9452f33a2e16258fd10b17067fda7eda7c28b3ab896e1a6763.jpg)

![7d27675d76b8f0c51dda3ad8500d7fa5c64ac5811b4a6aac6bfe8d9149ae6881.jpg](../iclr_results/3532_From Pixels to Tokens_ Byte-Pair Encoding on Quantized Visual Modalities/tables/7d27675d76b8f0c51dda3ad8500d7fa5c64ac5811b4a6aac6bfe8d9149ae6881.jpg)

![cd29fcb3a0a5a7fede4a8770f223b49fa69f752349bc78ddfe2031bceadbf670.jpg](../iclr_results/3532_From Pixels to Tokens_ Byte-Pair Encoding on Quantized Visual Modalities/tables/cd29fcb3a0a5a7fede4a8770f223b49fa69f752349bc78ddfe2031bceadbf670.jpg)

![d09db4df88717587489e210e5491892ea8299875e7b261e41db3b4bce77152b2.jpg](../iclr_results/3532_From Pixels to Tokens_ Byte-Pair Encoding on Quantized Visual Modalities/tables/d09db4df88717587489e210e5491892ea8299875e7b261e41db3b4bce77152b2.jpg)

![d2a356897cb3f7b1b37cb4e919d8cb657aae5fdd52ad6cb4f551d735771f5bc0.jpg](../iclr_results/3532_From Pixels to Tokens_ Byte-Pair Encoding on Quantized Visual Modalities/tables/d2a356897cb3f7b1b37cb4e919d8cb657aae5fdd52ad6cb4f551d735771f5bc0.jpg)

![ed253674557a445b70be92bd736eaf1e9b2f781c15b7096653b32f259915c381.jpg](../iclr_results/3532_From Pixels to Tokens_ Byte-Pair Encoding on Quantized Visual Modalities/tables/ed253674557a445b70be92bd736eaf1e9b2f781c15b7096653b32f259915c381.jpg)

![f8d468a298295cfcd3027d9399796a22d397c076575bcb5b79a603eb33363a12.jpg](../iclr_results/3532_From Pixels to Tokens_ Byte-Pair Encoding on Quantized Visual Modalities/tables/f8d468a298295cfcd3027d9399796a22d397c076575bcb5b79a603eb33363a12.jpg)
