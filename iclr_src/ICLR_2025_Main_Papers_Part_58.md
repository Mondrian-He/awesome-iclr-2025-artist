# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 58 of 100

## 目录 (Table of Contents)

1. [TypedThinker: Diversify Large Language Model Reasoning with Typed Thinking](#TypedThinker-Diversify-Large-Language-Model-Reasoning-with-Typed-Thinking)
2. [SEAL: Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection](#SEAL-Safety-enhanced-Aligned-LLM-Fine-tuning-via-Bilevel-Data-Selection)
3. [Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport](#Accelerating-3D-Molecule-Generation-via-Jointly-Geometric-Optimal-Transport)
4. [Neural Dueling Bandits: Preference-Based Optimization with Human Feedback](#Neural-Dueling-Bandits-Preference-Based-Optimization-with-Human-Feedback)
5. [Divergence of Neural Tangent Kernel in Classification Problems](#Divergence-of-Neural-Tangent-Kernel-in-Classification-Problems)
6. [Regret Bounds for Episodic Risk-Sensitive Linear Quadratic Regulator](#Regret-Bounds-for-Episodic-Risk-Sensitive-Linear-Quadratic-Regulator)
7. [How Low Can You Go? Searching for the Intrinsic Dimensionality of Complex Networks using Metric Node Embeddings](#How-Low-Can-You-Go-Searching-for-the-Intrinsic-Dimensionality-of-Complex-Networks-using-Metric-Node-Embeddings)
8. [Advancing Prompt-Based Methods for Replay-Independent General Continual Learning](#Advancing-Prompt-Based-Methods-for-Replay-Independent-General-Continual-Learning)
9. [Robustness Auditing for Linear Regression: To Singularity and Beyond](#Robustness-Auditing-for-Linear-Regression-To-Singularity-and-Beyond)
10. [Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents](#Agent-Security-Bench-ASB-Formalizing-and-Benchmarking-Attacks-and-Defenses-in-LLM-based-Agents)
11. [RA-TTA: Retrieval-Augmented Test-Time Adaptation for Vision-Language Models](#RA-TTA-Retrieval-Augmented-Test-Time-Adaptation-for-Vision-Language-Models)
12. [Decision Tree Induction Through LLMs via Semantically-Aware Evolution](#Decision-Tree-Induction-Through-LLMs-via-Semantically-Aware-Evolution)
13. [Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning](#Fictitious-Synthetic-Data-Can-Improve-LLM-Factuality-via-Prerequisite-Learning)
14. [Think Then React: Towards Unconstrained Action-to-Reaction Motion Generation](#Think-Then-React-Towards-Unconstrained-Action-to-Reaction-Motion-Generation)
15. [Towards a Theoretical Understanding of Synthetic Data in LLM Post-Training: A Reverse-Bottleneck Perspective](#Towards-a-Theoretical-Understanding-of-Synthetic-Data-in-LLM-Post-Training-A-Reverse-Bottleneck-Perspective)
16. [Triples as the Key: Structuring Makes Decomposition and Verification Easier in LLM-based TableQA](#Triples-as-the-Key-Structuring-Makes-Decomposition-and-Verification-Easier-in-LLM-based-TableQA)
17. [How DNNs break the Curse of Dimensionality: Compositionality and Symmetry Learning](#How-DNNs-break-the-Curse-of-Dimensionality-Compositionality-and-Symmetry-Learning)
18. [Protein Language Model Fitness is a Matter of Preference](#Protein-Language-Model-Fitness-is-a-Matter-of-Preference)
19. [Gnothi Seauton: Empowering Faithful Self-Interpretability in Black-Box Transformers](#Gnothi-Seauton-Empowering-Faithful-Self-Interpretability-in-Black-Box-Transformers)
20. [McEval: Massively Multilingual Code Evaluation](#McEval-Massively-Multilingual-Code-Evaluation)
21. [Privacy-Aware Lifelong Learning](#Privacy-Aware-Lifelong-Learning)
22. [MRAG-Bench: Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models](#MRAG-Bench-Vision-Centric-Evaluation-for-Retrieval-Augmented-Multimodal-Models)
23. [Learning Graph Invariance by Harnessing Spuriosity](#Learning-Graph-Invariance-by-Harnessing-Spuriosity)
24. [Mitigating Spurious Correlations in Zero-Shot Multimodal Models](#Mitigating-Spurious-Correlations-in-Zero-Shot-Multimodal-Models)
25. [The Breakdown of Gaussian Universality in Classification of High-dimensional Linear Factor Mixtures](#The-Breakdown-of-Gaussian-Universality-in-Classification-of-High-dimensional-Linear-Factor-Mixtures)
26. [Accurate and Scalable Graph Neural Networks via Message Invariance](#Accurate-and-Scalable-Graph-Neural-Networks-via-Message-Invariance)
27. [Strong Preferences Affect the Robustness of Preference Models and Value Alignment](#Strong-Preferences-Affect-the-Robustness-of-Preference-Models-and-Value-Alignment)
28. [VICtoR: Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation](#VICtoR-Learning-Hierarchical-Vision-Instruction-Correlation-Rewards-for-Long-horizon-Manipulation)
29. [LeanAgent: Lifelong Learning for Formal Theorem Proving](#LeanAgent-Lifelong-Learning-for-Formal-Theorem-Proving)
30. [BOFormer: Learning to Solve Multi-Objective Bayesian Optimization via Non-Markovian RL](#BOFormer-Learning-to-Solve-Multi-Objective-Bayesian-Optimization-via-Non-Markovian-RL)
31. [Combining Induction and Transduction for Abstract Reasoning](#Combining-Induction-and-Transduction-for-Abstract-Reasoning)
32. [Adversarial Training for Defense Against Label Poisoning Attacks](#Adversarial-Training-for-Defense-Against-Label-Poisoning-Attacks)
33. [SAM-CP: Marrying SAM with Composable Prompts for Versatile Segmentation](#SAM-CP-Marrying-SAM-with-Composable-Prompts-for-Versatile-Segmentation)
34. [Matryoshka Multimodal Models](#Matryoshka-Multimodal-Models)
35. [On Rollouts in Model-Based Reinforcement Learning](#On-Rollouts-in-Model-Based-Reinforcement-Learning)
36. [Information Theoretic Text-to-Image Alignment](#Information-Theoretic-Text-to-Image-Alignment)
37. [Uncertainty Herding: One Active Learning Method for All Label Budgets](#Uncertainty-Herding-One-Active-Learning-Method-for-All-Label-Budgets)

---


## TypedThinker: Diversify Large Language Model Reasoning with Typed Thinking

### Images

![93afb58e70e19df645ce2bd34b1837375245f309f35d78507e18d9d141cdb5ed.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/images/93afb58e70e19df645ce2bd34b1837375245f309f35d78507e18d9d141cdb5ed.jpg)

![a4a94441be833f0980976f960fefa61d3aefdeebc5d25ba7f55cae085342579d.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/images/a4a94441be833f0980976f960fefa61d3aefdeebc5d25ba7f55cae085342579d.jpg)

### Tables

![11ba5b9705dce1cd7084c66e4cdee58980113c5afb664af0fdd4bc72417594cc.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/11ba5b9705dce1cd7084c66e4cdee58980113c5afb664af0fdd4bc72417594cc.jpg)

![29cb8bd14bf97c21db3b242e99d08e3912939c3cb174eb2bb6ec78d274aae079.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/29cb8bd14bf97c21db3b242e99d08e3912939c3cb174eb2bb6ec78d274aae079.jpg)

![2d46363897404ebee86c1f7fecbc6602f30e3156005d6ef5bf434d45f20fc7f7.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/2d46363897404ebee86c1f7fecbc6602f30e3156005d6ef5bf434d45f20fc7f7.jpg)

![2d4669fbe7e43a401a8806dd1b45171195950c56efe1edac58a7b197c03fe0ff.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/2d4669fbe7e43a401a8806dd1b45171195950c56efe1edac58a7b197c03fe0ff.jpg)

![3ee3b071c48591cb401abbdc54e5bdb9ed045c3fe5dc3354027e5484d0ff7651.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/3ee3b071c48591cb401abbdc54e5bdb9ed045c3fe5dc3354027e5484d0ff7651.jpg)

![6409ac9ebf7d373c8159918e0995f02c594ce81019c7bbbfa3034fa636b7757f.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/6409ac9ebf7d373c8159918e0995f02c594ce81019c7bbbfa3034fa636b7757f.jpg)

![6c1fd9abe0da280adecd9ec85c931ef1ccb9b4ea631aef782b25371e41c365cb.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/6c1fd9abe0da280adecd9ec85c931ef1ccb9b4ea631aef782b25371e41c365cb.jpg)

![79e936261e8e2e144421c1b7a077ca2b6bf9f02907fb08722f36826ed3a17c4d.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/79e936261e8e2e144421c1b7a077ca2b6bf9f02907fb08722f36826ed3a17c4d.jpg)

![7da0337723860eff1b03f0309d72f6515a157d0703cfce180e18c17ddcfc255a.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/7da0337723860eff1b03f0309d72f6515a157d0703cfce180e18c17ddcfc255a.jpg)

![7e588f5d39e6a9f374da2e7e39f0f7ca4d92dd9aaf0a7fc5e6ccd9a0b6f72186.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/7e588f5d39e6a9f374da2e7e39f0f7ca4d92dd9aaf0a7fc5e6ccd9a0b6f72186.jpg)

![878191f45108b29d41e5cb88d8254d7f42ec19f0be4239a3e90b7829196e4fc5.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/878191f45108b29d41e5cb88d8254d7f42ec19f0be4239a3e90b7829196e4fc5.jpg)

![8c04dd367951baf37f2ca7e0380899589f0ff2f3f132dc8e2c5445d0d7fded4b.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/8c04dd367951baf37f2ca7e0380899589f0ff2f3f132dc8e2c5445d0d7fded4b.jpg)

![8cbd8ca365465943e7e1b35577546a7231b58e8c2dc88065ca3ccd2a7d629c2e.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/8cbd8ca365465943e7e1b35577546a7231b58e8c2dc88065ca3ccd2a7d629c2e.jpg)

![9102ca7566a850f6b7e8700082d4290091e2c6e083afee1752ed45ac7d6b40eb.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/9102ca7566a850f6b7e8700082d4290091e2c6e083afee1752ed45ac7d6b40eb.jpg)

![97e0ccd9354a3e27ed7bd24874a1f668faaaea20a4ce98179bbdd69c5af3ff9f.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/97e0ccd9354a3e27ed7bd24874a1f668faaaea20a4ce98179bbdd69c5af3ff9f.jpg)

![d09ab8664fec8deb8b706138c6190bb41c39823f3144fe0862a752447f17e453.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/d09ab8664fec8deb8b706138c6190bb41c39823f3144fe0862a752447f17e453.jpg)

![d3aeab703cfa444622f0d693ce17aebea0f27fb9f59fffb7a0335a505de34647.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/d3aeab703cfa444622f0d693ce17aebea0f27fb9f59fffb7a0335a505de34647.jpg)

![ed429b394fba2dff18c2b6ac00f7e3e49d4cd0002dff02c904f5ceb6118f0044.jpg](../iclr_results/2120_MCNC_ Manifold-Constrained Reparameterization for Neural Compression/tables/ed429b394fba2dff18c2b6ac00f7e3e49d4cd0002dff02c904f5ceb6118f0044.jpg)

## TypedThinker: Diversify Large Language Model Reasoning with Typed Thinking


### Images

![1f190da7250a45f860a672a4500ddf1ba2286058e6d7c7fc4cd3a7382720c15f.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/images/1f190da7250a45f860a672a4500ddf1ba2286058e6d7c7fc4cd3a7382720c15f.jpg)

![5439223283ff60790ec4c0a6825b5ad99b045f122e749d001da37691819439bb.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/images/5439223283ff60790ec4c0a6825b5ad99b045f122e749d001da37691819439bb.jpg)

![6eb9bb1910d17cc3cc831a0064d08a721468f3df543be8cc8965e5b208042978.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/images/6eb9bb1910d17cc3cc831a0064d08a721468f3df543be8cc8965e5b208042978.jpg)

![7f02543f27b5558097956bb35f231c3fce5629a0fc27867106634b04edc00640.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/images/7f02543f27b5558097956bb35f231c3fce5629a0fc27867106634b04edc00640.jpg)

![b78a7a3a5fe9d1dba9777327a1a0961a6b7e855a27b2aefa93f92e677b6fa92d.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/images/b78a7a3a5fe9d1dba9777327a1a0961a6b7e855a27b2aefa93f92e677b6fa92d.jpg)

![d8c6434a9006d83cff7f896672ae18f38f02dc94736c812a82479b8067c12654.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/images/d8c6434a9006d83cff7f896672ae18f38f02dc94736c812a82479b8067c12654.jpg)

### Tables

![176e27c01fb4c16f76584f86f1cde7a74d2b40798b654bf49fa9ea3b1d92bf56.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/176e27c01fb4c16f76584f86f1cde7a74d2b40798b654bf49fa9ea3b1d92bf56.jpg)

![1f2cd394df85463853bdf08d60d9f3fe9bbab572cb62f696109ca149da0d497a.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/1f2cd394df85463853bdf08d60d9f3fe9bbab572cb62f696109ca149da0d497a.jpg)

![20027d48fc465ffd831e22e11932110222acb4caeb501487a6c02c0f3626c9bb.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/20027d48fc465ffd831e22e11932110222acb4caeb501487a6c02c0f3626c9bb.jpg)

![25cf1aaa2dfe588e540db6268dd018173a724815904fecca195677b41a3ab616.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/25cf1aaa2dfe588e540db6268dd018173a724815904fecca195677b41a3ab616.jpg)

![27f15e3a05cdb1ef89a606f9baa0fc5e40a1579bafb0571d7f49772cca5b6e33.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/27f15e3a05cdb1ef89a606f9baa0fc5e40a1579bafb0571d7f49772cca5b6e33.jpg)

![3285f67317f2012481a3a4c05522e7184fd18f96532312eb7e23555914abc1ee.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/3285f67317f2012481a3a4c05522e7184fd18f96532312eb7e23555914abc1ee.jpg)

![41faaa5a96b9a2130bfc6ce5781e93f36388934948e72a520f9672bb12f28b1f.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/41faaa5a96b9a2130bfc6ce5781e93f36388934948e72a520f9672bb12f28b1f.jpg)

![5ac2a6be0387286b2209c50b7f2200dc19bfd40c29413fc7c5ba398073059125.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/5ac2a6be0387286b2209c50b7f2200dc19bfd40c29413fc7c5ba398073059125.jpg)

![694e7fd591d64ed27c7944ebbd3e8e3a7e3581f0aadc4faecd95a4bcffb6bc58.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/694e7fd591d64ed27c7944ebbd3e8e3a7e3581f0aadc4faecd95a4bcffb6bc58.jpg)

![7efbf059505b898fc24986637a997b2acddb6924169b074cc011fff11725540b.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/7efbf059505b898fc24986637a997b2acddb6924169b074cc011fff11725540b.jpg)

![a7a807d0eeba60f5d7244edd804486d679863887e3472700c47639ba25e6a1fb.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/a7a807d0eeba60f5d7244edd804486d679863887e3472700c47639ba25e6a1fb.jpg)

![aa5850810bca84eb4ac511a72a657592bfc343a14a60a8451960f6a4998a3ca9.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/aa5850810bca84eb4ac511a72a657592bfc343a14a60a8451960f6a4998a3ca9.jpg)

![b1e155cc8bef5796ac14709b203dfb306a4b08b644810dd3985daeea81d4f4c6.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/b1e155cc8bef5796ac14709b203dfb306a4b08b644810dd3985daeea81d4f4c6.jpg)

![cd49e0252223a577b3974b3dc57ef5f839fbd1159d9c7611a6e94880a4b5951b.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/cd49e0252223a577b3974b3dc57ef5f839fbd1159d9c7611a6e94880a4b5951b.jpg)

![cfbb80c9bee1c44643f83628da2affa56f37b44f41560e98a8060f8e03a821ad.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/cfbb80c9bee1c44643f83628da2affa56f37b44f41560e98a8060f8e03a821ad.jpg)

![dd5ea29150bddc3583d3f835222da5280fba71e5083bc6b3bfae13b1e568b8a3.jpg](../iclr_results/2121_TypedThinker_ Diversify Large Language Model Reasoning with Typed Thinking/tables/dd5ea29150bddc3583d3f835222da5280fba71e5083bc6b3bfae13b1e568b8a3.jpg)

## SEAL: Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection


### Images

![078e37192de3a13e8a4940142ba4ec162efce4f930f8c80cae1770d8b9d5329c.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/images/078e37192de3a13e8a4940142ba4ec162efce4f930f8c80cae1770d8b9d5329c.jpg)

![184fa5435de66d22882430742ed74750061aea64f596cd8eb05826488cf29c39.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/images/184fa5435de66d22882430742ed74750061aea64f596cd8eb05826488cf29c39.jpg)

![243ea7bcc044ce03e2982df0757184ef702974aa1f8627c43294799a2b0b5a57.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/images/243ea7bcc044ce03e2982df0757184ef702974aa1f8627c43294799a2b0b5a57.jpg)

![3baaaff459daea8556ad4d53df4cf3bdde51215fa3b13f88893c52167d12ca78.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/images/3baaaff459daea8556ad4d53df4cf3bdde51215fa3b13f88893c52167d12ca78.jpg)

![5f7baf735a4c2326fd4a362a90ff5ed15351cfbc381de89ee4bcf582a1a79938.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/images/5f7baf735a4c2326fd4a362a90ff5ed15351cfbc381de89ee4bcf582a1a79938.jpg)

![6bb2b97932b5d21e1d19fd91ac364b719afdf9946a48e4cf4b1fc7b36d5b994a.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/images/6bb2b97932b5d21e1d19fd91ac364b719afdf9946a48e4cf4b1fc7b36d5b994a.jpg)

![6cf968073d75ae86febbb5af542eeac40ebe7efa7957e0945ca6c80944a21c79.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/images/6cf968073d75ae86febbb5af542eeac40ebe7efa7957e0945ca6c80944a21c79.jpg)

![80e66421eb1424f8622d29a5521c40eedc4339d271c33be902fb003fbeaa6290.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/images/80e66421eb1424f8622d29a5521c40eedc4339d271c33be902fb003fbeaa6290.jpg)

![ed00fb0e915b1102914b3b79d08f964dfb7e9e3cbd0d804781b3110fe8a003cd.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/images/ed00fb0e915b1102914b3b79d08f964dfb7e9e3cbd0d804781b3110fe8a003cd.jpg)

### Tables

![1a9b3b528215cdd20fdc602190faa78a19f476e192a33ac55e2450e70efc3818.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/tables/1a9b3b528215cdd20fdc602190faa78a19f476e192a33ac55e2450e70efc3818.jpg)

![2e7c4c809544ec80e10482ef70fc881680875d1cdc3dac1544fe762b54cde3e5.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/tables/2e7c4c809544ec80e10482ef70fc881680875d1cdc3dac1544fe762b54cde3e5.jpg)

![406a16b00e68732d88a3f01942cb4d76e474dce5ebaba99de858f87943132e0d.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/tables/406a16b00e68732d88a3f01942cb4d76e474dce5ebaba99de858f87943132e0d.jpg)

![55f8b03c3151e28b517093fae14654df19a64adc364df8bc0fd4b052f5ec24cd.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/tables/55f8b03c3151e28b517093fae14654df19a64adc364df8bc0fd4b052f5ec24cd.jpg)

![5a6708878d7008648598d17cf13b79c723bcbedb7dee0ba95bd7932388d08e0c.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/tables/5a6708878d7008648598d17cf13b79c723bcbedb7dee0ba95bd7932388d08e0c.jpg)

![5aa2e163619754f5c0e7bf444b9970eca88eeec6ee2b9fa7c0d8f39e2ec93448.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/tables/5aa2e163619754f5c0e7bf444b9970eca88eeec6ee2b9fa7c0d8f39e2ec93448.jpg)

![71e36ca59f1bfe1aa91a8342c7be9dd2ddf9311352119121fe3cfc79c430269c.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/tables/71e36ca59f1bfe1aa91a8342c7be9dd2ddf9311352119121fe3cfc79c430269c.jpg)

![9cb98e9627f87d6f76cf03739279e0d2364c36dbc17b314ab7c9b53ac33c2802.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/tables/9cb98e9627f87d6f76cf03739279e0d2364c36dbc17b314ab7c9b53ac33c2802.jpg)

![a698fa12977bea1d8f3a91750d58baba865b21e6916bf0f732a1b07ff17b1385.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/tables/a698fa12977bea1d8f3a91750d58baba865b21e6916bf0f732a1b07ff17b1385.jpg)

![ab078c94188ca9dce14a66f292bbabf6ab348e1ba94b4d3a637897c9fd5ebbd6.jpg](../iclr_results/2122_SEAL_ Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection/tables/ab078c94188ca9dce14a66f292bbabf6ab348e1ba94b4d3a637897c9fd5ebbd6.jpg)

## Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport


### Images

![01ed98587b943a272919f5a638fba66379c7d10fb53b423f4d9e767ce6a196d0.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/images/01ed98587b943a272919f5a638fba66379c7d10fb53b423f4d9e767ce6a196d0.jpg)

![66a543967df0d2013c4a24182b17e398d6e0459a34f472b7df240089038d9519.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/images/66a543967df0d2013c4a24182b17e398d6e0459a34f472b7df240089038d9519.jpg)

![b9918abc3704483363dbcea56d64c6ba2e532bfabbbe29a6e67ec1bc72577e59.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/images/b9918abc3704483363dbcea56d64c6ba2e532bfabbbe29a6e67ec1bc72577e59.jpg)

![db890d970e89c5feecd0e9297ff1d8963d4d5ab2aedff8bb0e026bd034c61f28.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/images/db890d970e89c5feecd0e9297ff1d8963d4d5ab2aedff8bb0e026bd034c61f28.jpg)

![e3c495af33a3a200dd3715a15d99181678e360a3991b7f55e4f08df430671224.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/images/e3c495af33a3a200dd3715a15d99181678e360a3991b7f55e4f08df430671224.jpg)

![fdf57da2750d81445c3100e621e7fbcdc0762018433f57e4d1167875c734b2cf.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/images/fdf57da2750d81445c3100e621e7fbcdc0762018433f57e4d1167875c734b2cf.jpg)

### Tables

![1951265d88e2a96995cff428314b202550cc2b8bc98a7e359b35bcb5987affff.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/tables/1951265d88e2a96995cff428314b202550cc2b8bc98a7e359b35bcb5987affff.jpg)

![291c818f439ab9450d9c642594512d010c5c789b8d6e1cc3429503cb549331e5.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/tables/291c818f439ab9450d9c642594512d010c5c789b8d6e1cc3429503cb549331e5.jpg)

![4abd0298d34fbedb6c8b4769b72e77a320f58ec9a0c7c6f176f61ea01ed3da05.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/tables/4abd0298d34fbedb6c8b4769b72e77a320f58ec9a0c7c6f176f61ea01ed3da05.jpg)

![677eba64b0b85f1e0df50e761a73bfacac72a673bf2af4f185bbccf0ab7c10e7.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/tables/677eba64b0b85f1e0df50e761a73bfacac72a673bf2af4f185bbccf0ab7c10e7.jpg)

![850b1b976479d8f39d989361e24b9ba40b7504b1b7f1e32fbac5be7267d320d7.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/tables/850b1b976479d8f39d989361e24b9ba40b7504b1b7f1e32fbac5be7267d320d7.jpg)

![d9bc8641d1717dbd29d4732c3630ff9b9bea21075292f254ff8c420b57a336ef.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/tables/d9bc8641d1717dbd29d4732c3630ff9b9bea21075292f254ff8c420b57a336ef.jpg)

![f590e847c9f080b4e323a297f26333a0a8dfb69bd0bc821e0cb7d802bf273197.jpg](../iclr_results/2123_Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport/tables/f590e847c9f080b4e323a297f26333a0a8dfb69bd0bc821e0cb7d802bf273197.jpg)

## Neural Dueling Bandits: Preference-Based Optimization with Human Feedback


### Images

![0c99e0fbbb70e1d04ec1d2c02d0c3d3f986222cf16fb195fa12290da455e40f3.jpg](../iclr_results/2124_Neural Dueling Bandits_ Preference-Based Optimization with Human Feedback/images/0c99e0fbbb70e1d04ec1d2c02d0c3d3f986222cf16fb195fa12290da455e40f3.jpg)

![15654db7f0ef8cc5c4470a74ea601c9adb9172c397472eb400b0dec8c1473c93.jpg](../iclr_results/2124_Neural Dueling Bandits_ Preference-Based Optimization with Human Feedback/images/15654db7f0ef8cc5c4470a74ea601c9adb9172c397472eb400b0dec8c1473c93.jpg)

![27444cf1199e3baa648aa66215022309de92f2a8f0ff01e949a4a7d98cc939c0.jpg](../iclr_results/2124_Neural Dueling Bandits_ Preference-Based Optimization with Human Feedback/images/27444cf1199e3baa648aa66215022309de92f2a8f0ff01e949a4a7d98cc939c0.jpg)

![52d909aab8c08e099660c9f61510b0b6703c5a6b8c2df57153764e605bf0af41.jpg](../iclr_results/2124_Neural Dueling Bandits_ Preference-Based Optimization with Human Feedback/images/52d909aab8c08e099660c9f61510b0b6703c5a6b8c2df57153764e605bf0af41.jpg)

![6343216ed37a86f5065375e819436d920b7c39431249ce79ac18fcb21848039c.jpg](../iclr_results/2124_Neural Dueling Bandits_ Preference-Based Optimization with Human Feedback/images/6343216ed37a86f5065375e819436d920b7c39431249ce79ac18fcb21848039c.jpg)

![6ff5d815be635b45431bb912836344b99d75f0f1e2d507533e7ed2b5f16eb567.jpg](../iclr_results/2124_Neural Dueling Bandits_ Preference-Based Optimization with Human Feedback/images/6ff5d815be635b45431bb912836344b99d75f0f1e2d507533e7ed2b5f16eb567.jpg)

![7ebbcfa9f0cbd15cd9fa1134a835779a74c1e329fe060968a8988e3ba7f8a108.jpg](../iclr_results/2124_Neural Dueling Bandits_ Preference-Based Optimization with Human Feedback/images/7ebbcfa9f0cbd15cd9fa1134a835779a74c1e329fe060968a8988e3ba7f8a108.jpg)

![88833834c53a1af3be38d35385897861864b878bf33cdfd769c3ad84d8f9f43f.jpg](../iclr_results/2124_Neural Dueling Bandits_ Preference-Based Optimization with Human Feedback/images/88833834c53a1af3be38d35385897861864b878bf33cdfd769c3ad84d8f9f43f.jpg)

![de4479d9682f3b06ff68ae40e42f42d2c311545d316c02094a44a44e770cb05d.jpg](../iclr_results/2124_Neural Dueling Bandits_ Preference-Based Optimization with Human Feedback/images/de4479d9682f3b06ff68ae40e42f42d2c311545d316c02094a44a44e770cb05d.jpg)

## Divergence of Neural Tangent Kernel in Classification Problems


### Images

![347f66e26e76b04d7f9e120b9ce64c3b577f399db36c586eea3741f0aacf9680.jpg](../iclr_results/2125_Divergence of Neural Tangent Kernel in Classification Problems/images/347f66e26e76b04d7f9e120b9ce64c3b577f399db36c586eea3741f0aacf9680.jpg)

![70b7c5a09fe4185e1703713e5a338aab38ad301ddbef248129263d3043556e7d.jpg](../iclr_results/2125_Divergence of Neural Tangent Kernel in Classification Problems/images/70b7c5a09fe4185e1703713e5a338aab38ad301ddbef248129263d3043556e7d.jpg)

![8a2af81ab9c8ab3ccd6e5d2d11982ae077f9e576e0adeee00bd01a44045afe89.jpg](../iclr_results/2125_Divergence of Neural Tangent Kernel in Classification Problems/images/8a2af81ab9c8ab3ccd6e5d2d11982ae077f9e576e0adeee00bd01a44045afe89.jpg)

![d876d7d2ecad6ef8211f858d370234a4072e779917eda2e4a465beb1a19bf9b3.jpg](../iclr_results/2125_Divergence of Neural Tangent Kernel in Classification Problems/images/d876d7d2ecad6ef8211f858d370234a4072e779917eda2e4a465beb1a19bf9b3.jpg)

## Regret Bounds for Episodic Risk-Sensitive Linear Quadratic Regulator


### Images

![006632cb6fcd5a4dbfca67c3ad765d2d7563ee0a2b8542a3dbbc051e12a4d19c.jpg](../iclr_results/2126_Regret Bounds for Episodic Risk-Sensitive Linear Quadratic Regulator/images/006632cb6fcd5a4dbfca67c3ad765d2d7563ee0a2b8542a3dbbc051e12a4d19c.jpg)

![049b0f9c7abe37d212d9478934d0cbf179cc5efdc0fad73599481ff8f4f9405a.jpg](../iclr_results/2126_Regret Bounds for Episodic Risk-Sensitive Linear Quadratic Regulator/images/049b0f9c7abe37d212d9478934d0cbf179cc5efdc0fad73599481ff8f4f9405a.jpg)

![42c2833ee8527d5e50fed26dc00af4c0382b77262717aef492bb057bdb29e0d7.jpg](../iclr_results/2126_Regret Bounds for Episodic Risk-Sensitive Linear Quadratic Regulator/images/42c2833ee8527d5e50fed26dc00af4c0382b77262717aef492bb057bdb29e0d7.jpg)

## How Low Can You Go? Searching for the Intrinsic Dimensionality of Complex Networks using Metric Node Embeddings


### Images

![0fd288e2d62baf145648832e503897a4bdd47f1820ee736dcf1ce191ce33a80d.jpg](../iclr_results/2127_How Low Can You Go_ Searching for the Intrinsic Dimensionality of Complex Networks using Metric Node/images/0fd288e2d62baf145648832e503897a4bdd47f1820ee736dcf1ce191ce33a80d.jpg)

![1bc8bf179e7adad2719a319180a63b2f31463bfcf9a45614c78197d96360a6b6.jpg](../iclr_results/2127_How Low Can You Go_ Searching for the Intrinsic Dimensionality of Complex Networks using Metric Node/images/1bc8bf179e7adad2719a319180a63b2f31463bfcf9a45614c78197d96360a6b6.jpg)

![2b98686147e1c6bd6ac55d327fbf35f508a19ffa690b9c3a2cf04ac501a14965.jpg](../iclr_results/2127_How Low Can You Go_ Searching for the Intrinsic Dimensionality of Complex Networks using Metric Node/images/2b98686147e1c6bd6ac55d327fbf35f508a19ffa690b9c3a2cf04ac501a14965.jpg)

![37fc49660424c1ad15a029af87dc5a7ca5d20ae2deabe6fae126b9fbe3d09e84.jpg](../iclr_results/2127_How Low Can You Go_ Searching for the Intrinsic Dimensionality of Complex Networks using Metric Node/images/37fc49660424c1ad15a029af87dc5a7ca5d20ae2deabe6fae126b9fbe3d09e84.jpg)

![409a0bbf9109420b51d6a6d5f670735f58b067172eccb491db56e46b8adca984.jpg](../iclr_results/2127_How Low Can You Go_ Searching for the Intrinsic Dimensionality of Complex Networks using Metric Node/images/409a0bbf9109420b51d6a6d5f670735f58b067172eccb491db56e46b8adca984.jpg)

![7a47cbde6ca6230db0e601e1137718ee85282aba6e3d90e4ca9fe15a31366e76.jpg](../iclr_results/2127_How Low Can You Go_ Searching for the Intrinsic Dimensionality of Complex Networks using Metric Node/images/7a47cbde6ca6230db0e601e1137718ee85282aba6e3d90e4ca9fe15a31366e76.jpg)

![7eb2e14de932dc2d7893575353a71ff48b024195d0ca4f86b125f05c9902f0a0.jpg](../iclr_results/2127_How Low Can You Go_ Searching for the Intrinsic Dimensionality of Complex Networks using Metric Node/images/7eb2e14de932dc2d7893575353a71ff48b024195d0ca4f86b125f05c9902f0a0.jpg)

![829c162ac3cb0fc988d85cab8dd5209d0816547c87ca2f733e64af4a3609cf21.jpg](../iclr_results/2127_How Low Can You Go_ Searching for the Intrinsic Dimensionality of Complex Networks using Metric Node/images/829c162ac3cb0fc988d85cab8dd5209d0816547c87ca2f733e64af4a3609cf21.jpg)

![af0fd96a3b0f4e4827d71ed5ce11783ee66cee0c3f95b7f13fc8540c61c1462e.jpg](../iclr_results/2127_How Low Can You Go_ Searching for the Intrinsic Dimensionality of Complex Networks using Metric Node/images/af0fd96a3b0f4e4827d71ed5ce11783ee66cee0c3f95b7f13fc8540c61c1462e.jpg)

![b35eb8fee390cd02b92273304e2e207017186ccdd766485a44cc6fc4d5cb1e8f.jpg](../iclr_results/2127_How Low Can You Go_ Searching for the Intrinsic Dimensionality of Complex Networks using Metric Node/images/b35eb8fee390cd02b92273304e2e207017186ccdd766485a44cc6fc4d5cb1e8f.jpg)

![ffdcc1909549fc6fbb826f92c9677c871982a24cc92334b7edfd75515e41724b.jpg](../iclr_results/2127_How Low Can You Go_ Searching for the Intrinsic Dimensionality of Complex Networks using Metric Node/images/ffdcc1909549fc6fbb826f92c9677c871982a24cc92334b7edfd75515e41724b.jpg)

### Tables

![13b9f7972aff31b5c9e4157a90f3b0d8e2769f2d07ae41c667931b2561d09c19.jpg](../iclr_results/2127_How Low Can You Go_ Searching for the Intrinsic Dimensionality of Complex Networks using Metric Node/tables/13b9f7972aff31b5c9e4157a90f3b0d8e2769f2d07ae41c667931b2561d09c19.jpg)

![24e7cd3af8c9303466f5f1cfb178601400d6e909fb91115d2ee00f4cff2494af.jpg](../iclr_results/2127_How Low Can You Go_ Searching for the Intrinsic Dimensionality of Complex Networks using Metric Node/tables/24e7cd3af8c9303466f5f1cfb178601400d6e909fb91115d2ee00f4cff2494af.jpg)

![50fba1e695ffffcf0d3a2b335276264f6836b3581e0898c7bac02562f4113ce9.jpg](../iclr_results/2127_How Low Can You Go_ Searching for the Intrinsic Dimensionality of Complex Networks using Metric Node/tables/50fba1e695ffffcf0d3a2b335276264f6836b3581e0898c7bac02562f4113ce9.jpg)

![7b43b1f77aceff3a6bfff94247663fa5514bd1c146e5326e1b277a6c45b63ca7.jpg](../iclr_results/2127_How Low Can You Go_ Searching for the Intrinsic Dimensionality of Complex Networks using Metric Node/tables/7b43b1f77aceff3a6bfff94247663fa5514bd1c146e5326e1b277a6c45b63ca7.jpg)

![a92fd7f0be233d5578a0fe5c1606ef0bb3300f142e86525acf99e20c079d3417.jpg](../iclr_results/2127_How Low Can You Go_ Searching for the Intrinsic Dimensionality of Complex Networks using Metric Node/tables/a92fd7f0be233d5578a0fe5c1606ef0bb3300f142e86525acf99e20c079d3417.jpg)

![b58a314af1e4dff291f941473fc2d87b6f291ce0ffaa124f82f0a244177fc051.jpg](../iclr_results/2127_How Low Can You Go_ Searching for the Intrinsic Dimensionality of Complex Networks using Metric Node/tables/b58a314af1e4dff291f941473fc2d87b6f291ce0ffaa124f82f0a244177fc051.jpg)

![bfc7ca4dd7d153c2402a58034fbe08275542a65d898845a5cf24df673a21ea0a.jpg](../iclr_results/2127_How Low Can You Go_ Searching for the Intrinsic Dimensionality of Complex Networks using Metric Node/tables/bfc7ca4dd7d153c2402a58034fbe08275542a65d898845a5cf24df673a21ea0a.jpg)

![c125ee3a2ff144aff18c14af8101d56e1f3fc6e7dc2527d64ade07ed1edc58e7.jpg](../iclr_results/2127_How Low Can You Go_ Searching for the Intrinsic Dimensionality of Complex Networks using Metric Node/tables/c125ee3a2ff144aff18c14af8101d56e1f3fc6e7dc2527d64ade07ed1edc58e7.jpg)

## Advancing Prompt-Based Methods for Replay-Independent General Continual Learning


### Images

![189bd895bcdd3f00f7a8cc206acdb2c652ef6975a5fd08a2ac250b7eb67b2fc9.jpg](../iclr_results/2128_Advancing Prompt-Based Methods for Replay-Independent General Continual Learning/images/189bd895bcdd3f00f7a8cc206acdb2c652ef6975a5fd08a2ac250b7eb67b2fc9.jpg)

![c74bef5cde6106badc5807c3e63f41c28cb09c62b80f6f1856771e35c3cc14a0.jpg](../iclr_results/2128_Advancing Prompt-Based Methods for Replay-Independent General Continual Learning/images/c74bef5cde6106badc5807c3e63f41c28cb09c62b80f6f1856771e35c3cc14a0.jpg)

![e5e89e4a369dbd1dfa7ad4f125a8386ca31680ed3783c21078f27d841e7a5547.jpg](../iclr_results/2128_Advancing Prompt-Based Methods for Replay-Independent General Continual Learning/images/e5e89e4a369dbd1dfa7ad4f125a8386ca31680ed3783c21078f27d841e7a5547.jpg)

### Tables

![0601810acb1e9a2d72a52d861a6b436845794334a4506629bc50ee3ac31cd7e1.jpg](../iclr_results/2128_Advancing Prompt-Based Methods for Replay-Independent General Continual Learning/tables/0601810acb1e9a2d72a52d861a6b436845794334a4506629bc50ee3ac31cd7e1.jpg)

![0cf7f970373ee2c6418e9f5149dc2fcbb4d767f7b622dc621e57071fdfcea2a2.jpg](../iclr_results/2128_Advancing Prompt-Based Methods for Replay-Independent General Continual Learning/tables/0cf7f970373ee2c6418e9f5149dc2fcbb4d767f7b622dc621e57071fdfcea2a2.jpg)

![270a2928c56a0c455be05b5ee893d976eac44baa0491a421284368d00050e178.jpg](../iclr_results/2128_Advancing Prompt-Based Methods for Replay-Independent General Continual Learning/tables/270a2928c56a0c455be05b5ee893d976eac44baa0491a421284368d00050e178.jpg)

![3c995c4d7833d1e052d35ab11df78e919a6c956c00bd29ca87594db35136fb70.jpg](../iclr_results/2128_Advancing Prompt-Based Methods for Replay-Independent General Continual Learning/tables/3c995c4d7833d1e052d35ab11df78e919a6c956c00bd29ca87594db35136fb70.jpg)

![4582c725a588759241e800db7f0fc118ba228ad682a1e84e675af3f6ac1b906b.jpg](../iclr_results/2128_Advancing Prompt-Based Methods for Replay-Independent General Continual Learning/tables/4582c725a588759241e800db7f0fc118ba228ad682a1e84e675af3f6ac1b906b.jpg)

![5fcbe4496ec6a913cd0443a584f55087c9b92ff178534f216aa3b2381d50688e.jpg](../iclr_results/2128_Advancing Prompt-Based Methods for Replay-Independent General Continual Learning/tables/5fcbe4496ec6a913cd0443a584f55087c9b92ff178534f216aa3b2381d50688e.jpg)

![7f547acfbcca4f951700fdcc7625cafba6e3092936591061945a94f0157b5f6f.jpg](../iclr_results/2128_Advancing Prompt-Based Methods for Replay-Independent General Continual Learning/tables/7f547acfbcca4f951700fdcc7625cafba6e3092936591061945a94f0157b5f6f.jpg)

![84330667008c84f8d237547697a30fe86943629b52f269522ea46b7eb4a744db.jpg](../iclr_results/2128_Advancing Prompt-Based Methods for Replay-Independent General Continual Learning/tables/84330667008c84f8d237547697a30fe86943629b52f269522ea46b7eb4a744db.jpg)

![8b6c2dd2a76459b81df5ad800963c620b22b83c840e003b879ebdd97350c4666.jpg](../iclr_results/2128_Advancing Prompt-Based Methods for Replay-Independent General Continual Learning/tables/8b6c2dd2a76459b81df5ad800963c620b22b83c840e003b879ebdd97350c4666.jpg)

![c8bd74d5b9c5204130f0d05004c7d545570c825729d47baa493559351a8eb88a.jpg](../iclr_results/2128_Advancing Prompt-Based Methods for Replay-Independent General Continual Learning/tables/c8bd74d5b9c5204130f0d05004c7d545570c825729d47baa493559351a8eb88a.jpg)

![d0740b4d5eb6197d5b28e11e3b5343c7c4ca2efb2e3f01ab9c68139a8b9403e1.jpg](../iclr_results/2128_Advancing Prompt-Based Methods for Replay-Independent General Continual Learning/tables/d0740b4d5eb6197d5b28e11e3b5343c7c4ca2efb2e3f01ab9c68139a8b9403e1.jpg)

![e5af1a10ee99ddec082bcbafc2552f968071ac304e6b7bd7aa0eb3dbc67c765a.jpg](../iclr_results/2128_Advancing Prompt-Based Methods for Replay-Independent General Continual Learning/tables/e5af1a10ee99ddec082bcbafc2552f968071ac304e6b7bd7aa0eb3dbc67c765a.jpg)

![ebd571c51993ba0300807f3950ff1394e982fcdb94218418c75445d5400572c9.jpg](../iclr_results/2128_Advancing Prompt-Based Methods for Replay-Independent General Continual Learning/tables/ebd571c51993ba0300807f3950ff1394e982fcdb94218418c75445d5400572c9.jpg)

![f098c0ea8552eb309670bd7a01c312db5db129a3be9b2658b80431437bbfe5f7.jpg](../iclr_results/2128_Advancing Prompt-Based Methods for Replay-Independent General Continual Learning/tables/f098c0ea8552eb309670bd7a01c312db5db129a3be9b2658b80431437bbfe5f7.jpg)

## Robustness Auditing for Linear Regression: To Singularity and Beyond


### Images

![47a1fb43f03dccfbf85ed05a8bb7a8df5d0fb1f1af6baa0ecf13a0cd55ee2530.jpg](../iclr_results/2129_Robustness Auditing for Linear Regression_ To Singularity and Beyond/images/47a1fb43f03dccfbf85ed05a8bb7a8df5d0fb1f1af6baa0ecf13a0cd55ee2530.jpg)

![59cc2f845ae7f7b0d892407a52f82fa4520c1726362e5c010d313559de7fc5fc.jpg](../iclr_results/2129_Robustness Auditing for Linear Regression_ To Singularity and Beyond/images/59cc2f845ae7f7b0d892407a52f82fa4520c1726362e5c010d313559de7fc5fc.jpg)

![815bb879ac4fca62967926ddd4b9a2e1c1c74fef9adc5e02ab5d3ae63593c114.jpg](../iclr_results/2129_Robustness Auditing for Linear Regression_ To Singularity and Beyond/images/815bb879ac4fca62967926ddd4b9a2e1c1c74fef9adc5e02ab5d3ae63593c114.jpg)

![9078298ce28e3834e6946fddb3eff8613e6103eb03f27d60dd0ea469565f1741.jpg](../iclr_results/2129_Robustness Auditing for Linear Regression_ To Singularity and Beyond/images/9078298ce28e3834e6946fddb3eff8613e6103eb03f27d60dd0ea469565f1741.jpg)

![9744e3a005d76ef441f8d3165f756a1f4fdef2502a49a5f6af414f534ee83f98.jpg](../iclr_results/2129_Robustness Auditing for Linear Regression_ To Singularity and Beyond/images/9744e3a005d76ef441f8d3165f756a1f4fdef2502a49a5f6af414f534ee83f98.jpg)

![a108ee3239d2e9bd6f8deb8bd2fcb388f79c11be9788b1bdbeb9e0b663fdb781.jpg](../iclr_results/2129_Robustness Auditing for Linear Regression_ To Singularity and Beyond/images/a108ee3239d2e9bd6f8deb8bd2fcb388f79c11be9788b1bdbeb9e0b663fdb781.jpg)

![ab36c66332d78ce2bd260bedde3747a56e81e06dd68382a768457f9c7200968c.jpg](../iclr_results/2129_Robustness Auditing for Linear Regression_ To Singularity and Beyond/images/ab36c66332d78ce2bd260bedde3747a56e81e06dd68382a768457f9c7200968c.jpg)

![b45a2d8a0de25a122b827280bcf47932f5e53538600da65b34b76c38b64bf9b6.jpg](../iclr_results/2129_Robustness Auditing for Linear Regression_ To Singularity and Beyond/images/b45a2d8a0de25a122b827280bcf47932f5e53538600da65b34b76c38b64bf9b6.jpg)

![f1312e7a8e8e638b274774761bfdb83c24e26945acd6cb3cdb0dcc2fddd7b5af.jpg](../iclr_results/2129_Robustness Auditing for Linear Regression_ To Singularity and Beyond/images/f1312e7a8e8e638b274774761bfdb83c24e26945acd6cb3cdb0dcc2fddd7b5af.jpg)

![f17299093efd10f9b75a301af9ca29baef13e08010adbfd1e36c58a8ebda5638.jpg](../iclr_results/2129_Robustness Auditing for Linear Regression_ To Singularity and Beyond/images/f17299093efd10f9b75a301af9ca29baef13e08010adbfd1e36c58a8ebda5638.jpg)

### Tables

![22f264341d6c92f96c417afe2a4304cdfdf1d5bb8f7d096a7945750f3d7a5adf.jpg](../iclr_results/2129_Robustness Auditing for Linear Regression_ To Singularity and Beyond/tables/22f264341d6c92f96c417afe2a4304cdfdf1d5bb8f7d096a7945750f3d7a5adf.jpg)

![3fca38d646df1712ca519bb58fc13624230d83d671b3a3026c411d7d7f72c7ef.jpg](../iclr_results/2129_Robustness Auditing for Linear Regression_ To Singularity and Beyond/tables/3fca38d646df1712ca519bb58fc13624230d83d671b3a3026c411d7d7f72c7ef.jpg)

![57335dd356f355e4bde71eb9fa432074bd7dd37a48ccf6750582c5319b2092ec.jpg](../iclr_results/2129_Robustness Auditing for Linear Regression_ To Singularity and Beyond/tables/57335dd356f355e4bde71eb9fa432074bd7dd37a48ccf6750582c5319b2092ec.jpg)

![d63fcc72b5ddd935ce236184b2c285f4642a59713d66bbb3f9b4d545614b0112.jpg](../iclr_results/2129_Robustness Auditing for Linear Regression_ To Singularity and Beyond/tables/d63fcc72b5ddd935ce236184b2c285f4642a59713d66bbb3f9b4d545614b0112.jpg)

## Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents


### Images

![533b560853816127b9b889720f3c88cce04fedb03700160b3c97f45be4f49cca.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/images/533b560853816127b9b889720f3c88cce04fedb03700160b3c97f45be4f49cca.jpg)

![77ac8ca9d088dc5f4f8c54c43508ce9df978a035c40793259e6073de96ade89f.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/images/77ac8ca9d088dc5f4f8c54c43508ce9df978a035c40793259e6073de96ade89f.jpg)

![9756ebd5581267173fe3a76dbc69ddf32042623102adb5c51811f194cb12c438.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/images/9756ebd5581267173fe3a76dbc69ddf32042623102adb5c51811f194cb12c438.jpg)

![bef6ea944d6b6f80bfbc9b20e582f7ae2ed8f8add20fcfce3451c2d169c833c5.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/images/bef6ea944d6b6f80bfbc9b20e582f7ae2ed8f8add20fcfce3451c2d169c833c5.jpg)

### Tables

![0f0683bd104db7eaca1f020c947b657107cfd92e29c21b8971cf01b2a84ef721.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/tables/0f0683bd104db7eaca1f020c947b657107cfd92e29c21b8971cf01b2a84ef721.jpg)

![16ec2a52731e469368b33d368e96766423b3344096a0df3bb6e7f1d6b85c733c.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/tables/16ec2a52731e469368b33d368e96766423b3344096a0df3bb6e7f1d6b85c733c.jpg)

![2437bf3b8b34ae97582d63044e0f21653a01e1cb6a85cb46360de153064439f6.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/tables/2437bf3b8b34ae97582d63044e0f21653a01e1cb6a85cb46360de153064439f6.jpg)

![3535bfb75ae2202c1282f77c9f852d23bad4369e45f5fdf0d03e913652034ac1.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/tables/3535bfb75ae2202c1282f77c9f852d23bad4369e45f5fdf0d03e913652034ac1.jpg)

![4e9a33beb9068a9a5c9684729d849af4f1e6bcadd6f6c46d8d97587d02f4c07b.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/tables/4e9a33beb9068a9a5c9684729d849af4f1e6bcadd6f6c46d8d97587d02f4c07b.jpg)

![549aa2f9eefd278d524d5a4867f02b9732513bc580bd1fb252cc938b5f9de21a.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/tables/549aa2f9eefd278d524d5a4867f02b9732513bc580bd1fb252cc938b5f9de21a.jpg)

![6742f8f3f1d29c2dbe022d0376656c61eb1c5a1abe34fda5d1613b61d7508fc6.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/tables/6742f8f3f1d29c2dbe022d0376656c61eb1c5a1abe34fda5d1613b61d7508fc6.jpg)

![68449bba148593334f5798cbdd3862a121653e66810c820bd7004ca6459126cc.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/tables/68449bba148593334f5798cbdd3862a121653e66810c820bd7004ca6459126cc.jpg)

![77805fd1034a1af0157d8a8323900b691f3b7fb18154d9e1c59dfe5162d28b6a.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/tables/77805fd1034a1af0157d8a8323900b691f3b7fb18154d9e1c59dfe5162d28b6a.jpg)

![8de1dc875a637a2368db6007d94ec07a3cae3a4269dd8bc4a98b5c03e12fec25.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/tables/8de1dc875a637a2368db6007d94ec07a3cae3a4269dd8bc4a98b5c03e12fec25.jpg)

![8e832173bb05874e35e99b95515b531333e670dec92a9ed518dfc6bf88983eb2.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/tables/8e832173bb05874e35e99b95515b531333e670dec92a9ed518dfc6bf88983eb2.jpg)

![9d52b70f4616400586d2dc3eb1eb145afd5def207a80d4365d4970a3f6d221b0.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/tables/9d52b70f4616400586d2dc3eb1eb145afd5def207a80d4365d4970a3f6d221b0.jpg)

![aa92f6ff38eddbd10625d2867e51b4ee13a6b1d1af397b652abe117f7925c7a4.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/tables/aa92f6ff38eddbd10625d2867e51b4ee13a6b1d1af397b652abe117f7925c7a4.jpg)

![b653f39e6ecdd4d34aa2b3eb3e71711098a8b78d8aaa963ca36076938c4a0e69.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/tables/b653f39e6ecdd4d34aa2b3eb3e71711098a8b78d8aaa963ca36076938c4a0e69.jpg)

![c32a7d9bbe7b341cfd3dd7f0d589a648b36b760b477921cdaf16c6929e1a4ecf.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/tables/c32a7d9bbe7b341cfd3dd7f0d589a648b36b760b477921cdaf16c6929e1a4ecf.jpg)

![c581abb3bcf40cf06d7a75e63a38ecf2fc17f4a62adbecc4e23d5c74cb9616da.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/tables/c581abb3bcf40cf06d7a75e63a38ecf2fc17f4a62adbecc4e23d5c74cb9616da.jpg)

![c676ee4820a7cf3ce12710cf819e96d451c568a786c2add7d54759cc5fb95c4f.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/tables/c676ee4820a7cf3ce12710cf819e96d451c568a786c2add7d54759cc5fb95c4f.jpg)

![cd64a48e0541a05beb52309f711b610c4a9fe1752087463bca73d7a4dbbf2df2.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/tables/cd64a48e0541a05beb52309f711b610c4a9fe1752087463bca73d7a4dbbf2df2.jpg)

![cf9d4bf6432f48d39b9194947d87d384e5c794be0a0ed954deaab0f59666cd7c.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/tables/cf9d4bf6432f48d39b9194947d87d384e5c794be0a0ed954deaab0f59666cd7c.jpg)

![ed13ec47b453522944ccadf253083aec2e23fab0af950f12a8298d48f629a66a.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/tables/ed13ec47b453522944ccadf253083aec2e23fab0af950f12a8298d48f629a66a.jpg)

![eec9c33a400496c180d9fde3340351b14402315b108bdcbcf357b549a974fa77.jpg](../iclr_results/2130_Agent Security Bench (ASB)_ Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents/tables/eec9c33a400496c180d9fde3340351b14402315b108bdcbcf357b549a974fa77.jpg)

## RA-TTA: Retrieval-Augmented Test-Time Adaptation for Vision-Language Models


### Images

![049ac3df6d32e9ee20ed9d6e5e041287452d0832464ee12f3caf659d9089b0c6.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/images/049ac3df6d32e9ee20ed9d6e5e041287452d0832464ee12f3caf659d9089b0c6.jpg)

![1dfa8e02b71e4a8edfbebce31d925347f9cb4e34bf03b41f3610d7a8cac74a69.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/images/1dfa8e02b71e4a8edfbebce31d925347f9cb4e34bf03b41f3610d7a8cac74a69.jpg)

![2230794c7ebae43b85135406c6f908fbcfe3c7b2c13c11b33b819bb2c8a0b2b6.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/images/2230794c7ebae43b85135406c6f908fbcfe3c7b2c13c11b33b819bb2c8a0b2b6.jpg)

![291ae45ea644eb63512169e282433d13b3f43d0875b636f777c7a83ed05a9bd0.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/images/291ae45ea644eb63512169e282433d13b3f43d0875b636f777c7a83ed05a9bd0.jpg)

![48f2b2294e994ab270d65d133d8f760cf3f0530c476e5d0909c50abf8944873b.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/images/48f2b2294e994ab270d65d133d8f760cf3f0530c476e5d0909c50abf8944873b.jpg)

![4bd2c80861886109e622c844d4f7f0c6f3bdeb5d7fce062dbc0cf99712796acc.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/images/4bd2c80861886109e622c844d4f7f0c6f3bdeb5d7fce062dbc0cf99712796acc.jpg)

![4f695203dee73173467b97dab712f955a2aa1b0abe97da600013476bda6e8b23.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/images/4f695203dee73173467b97dab712f955a2aa1b0abe97da600013476bda6e8b23.jpg)

![5bc807c2adc15531b3325413ac98e3eb9df9d77260496273484c9535857e0e26.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/images/5bc807c2adc15531b3325413ac98e3eb9df9d77260496273484c9535857e0e26.jpg)

![6469f755f3021ff38f54bcb745c8bae78996204372f890355259c09e6341b49b.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/images/6469f755f3021ff38f54bcb745c8bae78996204372f890355259c09e6341b49b.jpg)

![6bd2966d35959611b8a07860cb1865000412c0b037b43ee5173aae8fa320b8ab.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/images/6bd2966d35959611b8a07860cb1865000412c0b037b43ee5173aae8fa320b8ab.jpg)

![6ecc74284b4d2fda078985aa43643c2d19f167dd3721af9691e36a584fcaec15.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/images/6ecc74284b4d2fda078985aa43643c2d19f167dd3721af9691e36a584fcaec15.jpg)

![75b0287b477f942787f4662cb4678ec776c37d1a6646c27f1ce91d62cbb2ef1e.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/images/75b0287b477f942787f4662cb4678ec776c37d1a6646c27f1ce91d62cbb2ef1e.jpg)

![77023788483825b8f71834e1ab4a8c09edff0f9906f3bb21b45c5f06f5142cd1.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/images/77023788483825b8f71834e1ab4a8c09edff0f9906f3bb21b45c5f06f5142cd1.jpg)

![7c119bceef05641a4337d260aeeea61fac6265542350b0c4d640f53af8d37d54.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/images/7c119bceef05641a4337d260aeeea61fac6265542350b0c4d640f53af8d37d54.jpg)

![7cbd9151dd485055f0956d542737ba556867dd067038cced565b43bada96a69c.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/images/7cbd9151dd485055f0956d542737ba556867dd067038cced565b43bada96a69c.jpg)

![85dd562f09fbbf5de163598d8569ae22e5c9786601b03a72a62800b217570cff.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/images/85dd562f09fbbf5de163598d8569ae22e5c9786601b03a72a62800b217570cff.jpg)

![9d58707726e608d51f7d934ee3179a914c975e53338812b78c2c520d8da8d589.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/images/9d58707726e608d51f7d934ee3179a914c975e53338812b78c2c520d8da8d589.jpg)

![c08453bc43e65b350d8e3f1756b5fc464503168e02e9236a53f9c5bdd2fa3175.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/images/c08453bc43e65b350d8e3f1756b5fc464503168e02e9236a53f9c5bdd2fa3175.jpg)

![d3272dafef922694cfcd0bbe5889f41ba83e3f57eaf61111308365f409292ffe.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/images/d3272dafef922694cfcd0bbe5889f41ba83e3f57eaf61111308365f409292ffe.jpg)

![e3ef3710f4ab97fcb3d3c6f2679a8abf6f647f5680aaddf41446396623ba0d38.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/images/e3ef3710f4ab97fcb3d3c6f2679a8abf6f647f5680aaddf41446396623ba0d38.jpg)

![e49f8a5af800544168dc64f209861fd0c399dbf50485b7bac465254054a793c5.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/images/e49f8a5af800544168dc64f209861fd0c399dbf50485b7bac465254054a793c5.jpg)

![fc94a155013821fca3d3d2ee0d2fac004c54787e3f9e1961688a49933af8dae5.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/images/fc94a155013821fca3d3d2ee0d2fac004c54787e3f9e1961688a49933af8dae5.jpg)

### Tables

![015e88b320a17d2567026c2449f060dd39c9158bc2da3853c5a53c106f8cd4c8.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/tables/015e88b320a17d2567026c2449f060dd39c9158bc2da3853c5a53c106f8cd4c8.jpg)

![10d0cec2a884c87a8c83328864be0213e90f23d3208ae7570641f55f25ee80cd.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/tables/10d0cec2a884c87a8c83328864be0213e90f23d3208ae7570641f55f25ee80cd.jpg)

![3788aa857c66f084cc7ca680327edc209deeaa795885e76c1537c975b1f07433.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/tables/3788aa857c66f084cc7ca680327edc209deeaa795885e76c1537c975b1f07433.jpg)

![4a49651256f3695c3a5db3e529c755c70af7f2478af8f169e884407e61249148.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/tables/4a49651256f3695c3a5db3e529c755c70af7f2478af8f169e884407e61249148.jpg)

![71d8c0e6c2ce806d142f26e4c92aacd1cf2d30723c685b200661c71b5d4c2ce0.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/tables/71d8c0e6c2ce806d142f26e4c92aacd1cf2d30723c685b200661c71b5d4c2ce0.jpg)

![9ed610cdf24a050ffe60f5cc9f33c4b167180c85a4bbaff5ac258a2f635aaafe.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/tables/9ed610cdf24a050ffe60f5cc9f33c4b167180c85a4bbaff5ac258a2f635aaafe.jpg)

![a43a757cab0e228e6f95bf5c69dce963973ac2a473bf7ccf478129307fe81583.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/tables/a43a757cab0e228e6f95bf5c69dce963973ac2a473bf7ccf478129307fe81583.jpg)

![a8e7fd3dcfedaa2487ff93f0c332d7329b83570a1da3f05248b6a76c6a81f13e.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/tables/a8e7fd3dcfedaa2487ff93f0c332d7329b83570a1da3f05248b6a76c6a81f13e.jpg)

![bd7d28fa4f5c8f4971f6255785e369b41fe261ecd7fb82d53c18f3fd3c3be63e.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/tables/bd7d28fa4f5c8f4971f6255785e369b41fe261ecd7fb82d53c18f3fd3c3be63e.jpg)

![d0df3bf5380bce15ddfd83c91e18c5bbb616a1549111ace9b5b14cd4f5cb19b1.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/tables/d0df3bf5380bce15ddfd83c91e18c5bbb616a1549111ace9b5b14cd4f5cb19b1.jpg)

![ea3d0dca1a217fdeb76ce2a0d661117f2c39b998922b1c7dfb01a81e3900d9d4.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/tables/ea3d0dca1a217fdeb76ce2a0d661117f2c39b998922b1c7dfb01a81e3900d9d4.jpg)

![ee68c216ce40279af2921965b965611cb40211d4e49a457fa3148c7673e5d180.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/tables/ee68c216ce40279af2921965b965611cb40211d4e49a457fa3148c7673e5d180.jpg)

![f5fe8d1ca0da8884c56dca2c10c8f0be644343eeeb43775c1e5269c08fb1fd7d.jpg](../iclr_results/2131_RA-TTA_ Retrieval-Augmented Test-Time Adaptation for Vision-Language Models/tables/f5fe8d1ca0da8884c56dca2c10c8f0be644343eeeb43775c1e5269c08fb1fd7d.jpg)

## Decision Tree Induction Through LLMs via Semantically-Aware Evolution


### Images

![0034a52fc69507ad7e0038a999f65cf7e782c51cb3d0bd26e323af9edad72ad2.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/images/0034a52fc69507ad7e0038a999f65cf7e782c51cb3d0bd26e323af9edad72ad2.jpg)

![2057d6b8349ecfc7f0584e5fdd0d93f0a5ff3886137b608e87fb6f15fd39fbef.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/images/2057d6b8349ecfc7f0584e5fdd0d93f0a5ff3886137b608e87fb6f15fd39fbef.jpg)

![2e71c728449ca425072bbab1a3653893c031ed9fd1ab44c73f5d681dfaf79164.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/images/2e71c728449ca425072bbab1a3653893c031ed9fd1ab44c73f5d681dfaf79164.jpg)

![313e2c822f1e1fb9234692eb2aee5d101c2e1da6b2c962db1341f4a936f8266e.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/images/313e2c822f1e1fb9234692eb2aee5d101c2e1da6b2c962db1341f4a936f8266e.jpg)

![3bba499d6f91c583b09ff6bd228d3d5ad0050d400c570bc5e2188be5d9bd0d5c.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/images/3bba499d6f91c583b09ff6bd228d3d5ad0050d400c570bc5e2188be5d9bd0d5c.jpg)

![42d01d47f1c0fe9a0178f193f2223a9778b911653ffc3c365733dd1449d1cfe1.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/images/42d01d47f1c0fe9a0178f193f2223a9778b911653ffc3c365733dd1449d1cfe1.jpg)

![655bc7ae38b254d6bf1b3db99444058400cc2811c95be8d30727bd58a027625f.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/images/655bc7ae38b254d6bf1b3db99444058400cc2811c95be8d30727bd58a027625f.jpg)

![7c754efce7eb8c9f36329885d1be7efe00afc88c4859ab88d1ed8c137e55b189.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/images/7c754efce7eb8c9f36329885d1be7efe00afc88c4859ab88d1ed8c137e55b189.jpg)

![7e7b79c5f528a8c03e4c54a515303975ce4c9fc82e134f6196c545f0c76941cf.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/images/7e7b79c5f528a8c03e4c54a515303975ce4c9fc82e134f6196c545f0c76941cf.jpg)

![854eed95749bbb0e66e2bfcb09a218bc9d8de177a4436b758b5669f91d7a8302.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/images/854eed95749bbb0e66e2bfcb09a218bc9d8de177a4436b758b5669f91d7a8302.jpg)

![93813f5cb63d2dbdd0971e8497be456e5c5bbe501eda2291261acab3145f0631.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/images/93813f5cb63d2dbdd0971e8497be456e5c5bbe501eda2291261acab3145f0631.jpg)

![aa469bb9c922f33a7f69468d170e1e657f1bf796c613819247e843b792442ee3.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/images/aa469bb9c922f33a7f69468d170e1e657f1bf796c613819247e843b792442ee3.jpg)

![af5c7271df306cd5ddb845623a3c731cd970567831704ee90c2d7dd2ecec1db0.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/images/af5c7271df306cd5ddb845623a3c731cd970567831704ee90c2d7dd2ecec1db0.jpg)

![bf0e3c3a012dc39b380b39d6c01efddf6bbc584464bb7b3fe5ae65a43d867db2.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/images/bf0e3c3a012dc39b380b39d6c01efddf6bbc584464bb7b3fe5ae65a43d867db2.jpg)

![c1789bc12c8737743332e11f5357c307b1932ac04daef92f13bc0f253807af44.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/images/c1789bc12c8737743332e11f5357c307b1932ac04daef92f13bc0f253807af44.jpg)

![c6f3da4aa4f559d922b4bac7db901b0347866af87cc11a29856be76e3e0f7ff8.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/images/c6f3da4aa4f559d922b4bac7db901b0347866af87cc11a29856be76e3e0f7ff8.jpg)

![d1a42ebec71a517f6684501d7e6d2754b47cc599824fcebf06ada9fbd436bbff.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/images/d1a42ebec71a517f6684501d7e6d2754b47cc599824fcebf06ada9fbd436bbff.jpg)

![e2fd5e543ef500329c6ca5b9448a8d80c7569c3b5dba57178a05616171a9d3ae.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/images/e2fd5e543ef500329c6ca5b9448a8d80c7569c3b5dba57178a05616171a9d3ae.jpg)

![e5c811acc1c3aa1906bad08b69a65bfde132eb1f2dc2cc84c23f38a44fef50f9.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/images/e5c811acc1c3aa1906bad08b69a65bfde132eb1f2dc2cc84c23f38a44fef50f9.jpg)

![e7536e0da0c1efa16ff7d4507e4089c489d5f3ff88dfb3c21b2f9ae9001c38e5.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/images/e7536e0da0c1efa16ff7d4507e4089c489d5f3ff88dfb3c21b2f9ae9001c38e5.jpg)

### Tables

![03ec469901df54e4e96fe88032b1d848bcca222971e414163ea3421d6608abd5.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/tables/03ec469901df54e4e96fe88032b1d848bcca222971e414163ea3421d6608abd5.jpg)

![048aa4fcec35ef4a1ed6f07f48d61eb5d92cffb38fe3b06605372729502bd983.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/tables/048aa4fcec35ef4a1ed6f07f48d61eb5d92cffb38fe3b06605372729502bd983.jpg)

![1aa37528fd484803d2cd3a80a9548e6b49f443beb1bf871a0683012807eea8ce.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/tables/1aa37528fd484803d2cd3a80a9548e6b49f443beb1bf871a0683012807eea8ce.jpg)

![1efebcabba6110ccd106495c16964d9bfd933c4d5ca9387da2f432b25416e3b3.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/tables/1efebcabba6110ccd106495c16964d9bfd933c4d5ca9387da2f432b25416e3b3.jpg)

![29fdc900439467d5777abd24eeea682f8f54ee56c2518a3518f03dba10fc5fd2.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/tables/29fdc900439467d5777abd24eeea682f8f54ee56c2518a3518f03dba10fc5fd2.jpg)

![4538c7307dd0fd9a6a1eb395640e6e4cdd2d1d969d5add1364cac594c97fda53.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/tables/4538c7307dd0fd9a6a1eb395640e6e4cdd2d1d969d5add1364cac594c97fda53.jpg)

![4ba62f41860ef576db1187cbf2f4c56741a23566bac3c0d0a17f6800e34a1344.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/tables/4ba62f41860ef576db1187cbf2f4c56741a23566bac3c0d0a17f6800e34a1344.jpg)

![637a3de02899bf2db69a545a64c2c3703228deacb841fa93ce318ab9ec3424c1.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/tables/637a3de02899bf2db69a545a64c2c3703228deacb841fa93ce318ab9ec3424c1.jpg)

![83c2c4c856a840a6fb89fedb6440bac77c9c5e82cab6799f3161e9f49828d8d8.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/tables/83c2c4c856a840a6fb89fedb6440bac77c9c5e82cab6799f3161e9f49828d8d8.jpg)

![9def93280d04b7539bece30f8a3963a7cd8673d988d00c67af333ec085a0a27f.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/tables/9def93280d04b7539bece30f8a3963a7cd8673d988d00c67af333ec085a0a27f.jpg)

![aeb0befc656e1dd89c9613e6b133b5c4d14c6b8acbab75c2f86b5317ef89014b.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/tables/aeb0befc656e1dd89c9613e6b133b5c4d14c6b8acbab75c2f86b5317ef89014b.jpg)

![c59bff4f298dfeeb23cedf36cd2fdd995b9803f537822e3c95b951ac702ab7e3.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/tables/c59bff4f298dfeeb23cedf36cd2fdd995b9803f537822e3c95b951ac702ab7e3.jpg)

![c7708a374aac33be2fd91b20c04beebbb6ab3755b5e3698618500f1f1859e8a6.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/tables/c7708a374aac33be2fd91b20c04beebbb6ab3755b5e3698618500f1f1859e8a6.jpg)

![ddb0564969c79b2d95ca7b1a06092bfd665235346e6e388c40044baba0080bbf.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/tables/ddb0564969c79b2d95ca7b1a06092bfd665235346e6e388c40044baba0080bbf.jpg)

![eef19423cf228fb13147aeb1069e187fd500b797b4b5044b03418ba85eb45d15.jpg](../iclr_results/2132_Decision Tree Induction Through LLMs via Semantically-Aware Evolution/tables/eef19423cf228fb13147aeb1069e187fd500b797b4b5044b03418ba85eb45d15.jpg)

## Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning


### Images

![1cf87c1daa5ebba5855fb94310337fd9e72f3214b84f312635b4f406ef4ca92c.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/images/1cf87c1daa5ebba5855fb94310337fd9e72f3214b84f312635b4f406ef4ca92c.jpg)

![40f9702fb30ae88bee7aee83fac3bb72b1368bb6f5785260f5add9f2e475a06e.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/images/40f9702fb30ae88bee7aee83fac3bb72b1368bb6f5785260f5add9f2e475a06e.jpg)

![4f6cd46bc33e251f582ff350513add1a44f2d0cd87cdbabb079242fc232eb954.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/images/4f6cd46bc33e251f582ff350513add1a44f2d0cd87cdbabb079242fc232eb954.jpg)

![681d5cf9d10d2e252270c29952ca37e8ffe36ac9f9bfa489da8f59fea7f7cddc.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/images/681d5cf9d10d2e252270c29952ca37e8ffe36ac9f9bfa489da8f59fea7f7cddc.jpg)

![77a58dffb595694cb4966b9fef41a0819e2182f87178b4e55073816be08a4006.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/images/77a58dffb595694cb4966b9fef41a0819e2182f87178b4e55073816be08a4006.jpg)

![8277eb4655127e73f8a13a7935b28ba9b3b859c709336d13436570c2e599c040.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/images/8277eb4655127e73f8a13a7935b28ba9b3b859c709336d13436570c2e599c040.jpg)

![8de9e9501653595f982532cd271c0fddaa94576fc3fbe0b71366b23e15af1de5.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/images/8de9e9501653595f982532cd271c0fddaa94576fc3fbe0b71366b23e15af1de5.jpg)

![9872501634275a1fbac8dfa0ee2126c860e68b467af1ddd8d1305098aa0fbc3a.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/images/9872501634275a1fbac8dfa0ee2126c860e68b467af1ddd8d1305098aa0fbc3a.jpg)

![9fc53e50ca57b760799709529ee4a91f4a2ddce1378a9661376661d6572c3532.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/images/9fc53e50ca57b760799709529ee4a91f4a2ddce1378a9661376661d6572c3532.jpg)

![e7933175e6ec637e3462a591b6181fc77ed10908e342dd844ad2f1a6caedd265.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/images/e7933175e6ec637e3462a591b6181fc77ed10908e342dd844ad2f1a6caedd265.jpg)

![f447fd7dacf1716ed304f57471e6a1694332ab13ec832a4a93dfc93f86d3ead0.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/images/f447fd7dacf1716ed304f57471e6a1694332ab13ec832a4a93dfc93f86d3ead0.jpg)

![f6fc94e42b36951e19fce0783902ac6dbc6d3e79ec7f9ac9038e1414cc6d3253.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/images/f6fc94e42b36951e19fce0783902ac6dbc6d3e79ec7f9ac9038e1414cc6d3253.jpg)

### Tables

![089c8ddb8c78c5e90a8fb6394dcbb47ac39175446edf1299cdaa05c21665b816.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/tables/089c8ddb8c78c5e90a8fb6394dcbb47ac39175446edf1299cdaa05c21665b816.jpg)

![1b829ab3d03841e49f041599365a0a1716da493bd041cc8c1529bbf049058b11.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/tables/1b829ab3d03841e49f041599365a0a1716da493bd041cc8c1529bbf049058b11.jpg)

![405e1688f068799c5adcbaf6b1d0d15b375f636ce9e520989dfcba685697bbda.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/tables/405e1688f068799c5adcbaf6b1d0d15b375f636ce9e520989dfcba685697bbda.jpg)

![5cb37755ffd1fa4d37faaa8b200fec8089c8492755cd7b91735006b40f03e5ef.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/tables/5cb37755ffd1fa4d37faaa8b200fec8089c8492755cd7b91735006b40f03e5ef.jpg)

![5db9ca48b67338193d9a55a2abd4c910bbb098fde60d3667d13009cfb87f353e.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/tables/5db9ca48b67338193d9a55a2abd4c910bbb098fde60d3667d13009cfb87f353e.jpg)

![66f8bd5fab29011e46aaa8ac0a9e7dfb0f71b636afb8f7e92834a420f85b139a.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/tables/66f8bd5fab29011e46aaa8ac0a9e7dfb0f71b636afb8f7e92834a420f85b139a.jpg)

![6b056fc66c050ed16d676e885cad0a27a98e009f560178637d7a1771a1a6d37b.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/tables/6b056fc66c050ed16d676e885cad0a27a98e009f560178637d7a1771a1a6d37b.jpg)

![7c18a53caa6e57048bdd49366a087564e6a439cce4fba37605f58ab72fc1e725.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/tables/7c18a53caa6e57048bdd49366a087564e6a439cce4fba37605f58ab72fc1e725.jpg)

![a2035f4356a62ca4892d05bfc6e876c11f861d879e002001b78349f7132cfb33.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/tables/a2035f4356a62ca4892d05bfc6e876c11f861d879e002001b78349f7132cfb33.jpg)

![aa2028c516a97a2476159ef669fee838b23a1cea2175d4c56e22bedd03a734f6.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/tables/aa2028c516a97a2476159ef669fee838b23a1cea2175d4c56e22bedd03a734f6.jpg)

![ad9bf20a432bd4119e3b16c0b7c8697ac46d26e0cbac1150d7653a7c2fb868b9.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/tables/ad9bf20a432bd4119e3b16c0b7c8697ac46d26e0cbac1150d7653a7c2fb868b9.jpg)

![c25a54b69738ef4eb3b5e9f70985fa2061534293de463b138d997aa06f933186.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/tables/c25a54b69738ef4eb3b5e9f70985fa2061534293de463b138d997aa06f933186.jpg)

![dbdde17e09fa286cf2c1a632c6f2b6fee6067b9007873f97969c348815351ca1.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/tables/dbdde17e09fa286cf2c1a632c6f2b6fee6067b9007873f97969c348815351ca1.jpg)

![e8390f4782e9703398944a48f50d1a41dddca82c77fd5011a9f93ec153fb9404.jpg](../iclr_results/2133_Fictitious Synthetic Data Can Improve LLM Factuality via Prerequisite Learning/tables/e8390f4782e9703398944a48f50d1a41dddca82c77fd5011a9f93ec153fb9404.jpg)

## Think Then React: Towards Unconstrained Action-to-Reaction Motion Generation


### Images

![09a958dd9505d71d8f7008201787736cece965d57ce86f3fdfed368f1be529df.jpg](../iclr_results/2134_Think Then React_ Towards Unconstrained Action-to-Reaction Motion Generation/images/09a958dd9505d71d8f7008201787736cece965d57ce86f3fdfed368f1be529df.jpg)

![164a6f0861830810ccb1414b39275f1c460122d559861cef515227c656e0a5ad.jpg](../iclr_results/2134_Think Then React_ Towards Unconstrained Action-to-Reaction Motion Generation/images/164a6f0861830810ccb1414b39275f1c460122d559861cef515227c656e0a5ad.jpg)

![1798d11ea154004a386627367933f36aa5e39f3d22409e992568c2a6e70783e9.jpg](../iclr_results/2134_Think Then React_ Towards Unconstrained Action-to-Reaction Motion Generation/images/1798d11ea154004a386627367933f36aa5e39f3d22409e992568c2a6e70783e9.jpg)

![284268c4b64d95dbaa88c8b63e00dfd33b6d6cf423451bd8b3a4b5025d8d8cbf.jpg](../iclr_results/2134_Think Then React_ Towards Unconstrained Action-to-Reaction Motion Generation/images/284268c4b64d95dbaa88c8b63e00dfd33b6d6cf423451bd8b3a4b5025d8d8cbf.jpg)

![35bc1692e1fca4680086e700c7acb289c3efdc7e0e591c6081a4a9435dd092be.jpg](../iclr_results/2134_Think Then React_ Towards Unconstrained Action-to-Reaction Motion Generation/images/35bc1692e1fca4680086e700c7acb289c3efdc7e0e591c6081a4a9435dd092be.jpg)

![3991931b74b8b6af2475527238f8bd39d84c3b0813f8570e207d3d533dfe3489.jpg](../iclr_results/2134_Think Then React_ Towards Unconstrained Action-to-Reaction Motion Generation/images/3991931b74b8b6af2475527238f8bd39d84c3b0813f8570e207d3d533dfe3489.jpg)

![73a0c58dbb370a682dcb9aa062ea2893d745fb024ef7ae2c3100088522fe340f.jpg](../iclr_results/2134_Think Then React_ Towards Unconstrained Action-to-Reaction Motion Generation/images/73a0c58dbb370a682dcb9aa062ea2893d745fb024ef7ae2c3100088522fe340f.jpg)

![7b54fbe0ef4963f470ff64887c292260d5694de9b437b1eaa1a3a843ec1b64fb.jpg](../iclr_results/2134_Think Then React_ Towards Unconstrained Action-to-Reaction Motion Generation/images/7b54fbe0ef4963f470ff64887c292260d5694de9b437b1eaa1a3a843ec1b64fb.jpg)

![9c309f814c31eb9872c3cc7a9f94cd96a07481e14c15ac5dfe65e3a5566e721a.jpg](../iclr_results/2134_Think Then React_ Towards Unconstrained Action-to-Reaction Motion Generation/images/9c309f814c31eb9872c3cc7a9f94cd96a07481e14c15ac5dfe65e3a5566e721a.jpg)

![a8b249ac24a728ff105b940911fc0e3cb044ca622ea5a1f65e6ae77c6b7c471e.jpg](../iclr_results/2134_Think Then React_ Towards Unconstrained Action-to-Reaction Motion Generation/images/a8b249ac24a728ff105b940911fc0e3cb044ca622ea5a1f65e6ae77c6b7c471e.jpg)

### Tables

![053112ed654615d2b4a6329a2f701032929a6b3f2ec111a5a8ca33957d579062.jpg](../iclr_results/2134_Think Then React_ Towards Unconstrained Action-to-Reaction Motion Generation/tables/053112ed654615d2b4a6329a2f701032929a6b3f2ec111a5a8ca33957d579062.jpg)

![60094f242020ff817df3ac70f5224c445448d940a416917cb4d06673a177ef6d.jpg](../iclr_results/2134_Think Then React_ Towards Unconstrained Action-to-Reaction Motion Generation/tables/60094f242020ff817df3ac70f5224c445448d940a416917cb4d06673a177ef6d.jpg)

![7e3cd68fb00ea3fe6fb5172232e28e10039b531917099ecdfcc3149ba77f6a46.jpg](../iclr_results/2134_Think Then React_ Towards Unconstrained Action-to-Reaction Motion Generation/tables/7e3cd68fb00ea3fe6fb5172232e28e10039b531917099ecdfcc3149ba77f6a46.jpg)

## Towards a Theoretical Understanding of Synthetic Data in LLM Post-Training: A Reverse-Bottleneck Perspective


### Images

![02c01050df5b1d4d9b78b81ba708e0768d569beeece2b7740073a555e4f426ec.jpg](../iclr_results/2135_Towards a Theoretical Understanding of Synthetic Data in LLM Post-Training_ A Reverse-Bottleneck Per/images/02c01050df5b1d4d9b78b81ba708e0768d569beeece2b7740073a555e4f426ec.jpg)

![0f77d0516082df90edd2568e2e468c55eb0983edc9dc82e4cd615bdb5f87777b.jpg](../iclr_results/2135_Towards a Theoretical Understanding of Synthetic Data in LLM Post-Training_ A Reverse-Bottleneck Per/images/0f77d0516082df90edd2568e2e468c55eb0983edc9dc82e4cd615bdb5f87777b.jpg)

![594e7e796519b6e1c8c376568c6d8a4c8b2346732c1fe2a87a10ca6e19c60673.jpg](../iclr_results/2135_Towards a Theoretical Understanding of Synthetic Data in LLM Post-Training_ A Reverse-Bottleneck Per/images/594e7e796519b6e1c8c376568c6d8a4c8b2346732c1fe2a87a10ca6e19c60673.jpg)

![c6ec0d41c21f2786abc3df4418643695945df997b45c7ce112efda36c5b85643.jpg](../iclr_results/2135_Towards a Theoretical Understanding of Synthetic Data in LLM Post-Training_ A Reverse-Bottleneck Per/images/c6ec0d41c21f2786abc3df4418643695945df997b45c7ce112efda36c5b85643.jpg)

![fece4b361f88ee5de307fa5814877a00655d47ac3bae0866547449002a14435b.jpg](../iclr_results/2135_Towards a Theoretical Understanding of Synthetic Data in LLM Post-Training_ A Reverse-Bottleneck Per/images/fece4b361f88ee5de307fa5814877a00655d47ac3bae0866547449002a14435b.jpg)

### Tables

![29bf7668f6cfae0df70d76afcb00928e0a9eb1225c3d5e5f6c6fd728a7433e5a.jpg](../iclr_results/2135_Towards a Theoretical Understanding of Synthetic Data in LLM Post-Training_ A Reverse-Bottleneck Per/tables/29bf7668f6cfae0df70d76afcb00928e0a9eb1225c3d5e5f6c6fd728a7433e5a.jpg)

![53a10450b934036fdda11640ff13ca490eb1dde16360b8672f2096c245240523.jpg](../iclr_results/2135_Towards a Theoretical Understanding of Synthetic Data in LLM Post-Training_ A Reverse-Bottleneck Per/tables/53a10450b934036fdda11640ff13ca490eb1dde16360b8672f2096c245240523.jpg)

![78bef1937c3d2d4d9a2ba642acb447929323bee359e5efdba7aa686e0effd046.jpg](../iclr_results/2135_Towards a Theoretical Understanding of Synthetic Data in LLM Post-Training_ A Reverse-Bottleneck Per/tables/78bef1937c3d2d4d9a2ba642acb447929323bee359e5efdba7aa686e0effd046.jpg)

![fe42639cbe87e46c27776b682abf41f8c5ea9201939975869fb7b181b9455391.jpg](../iclr_results/2135_Towards a Theoretical Understanding of Synthetic Data in LLM Post-Training_ A Reverse-Bottleneck Per/tables/fe42639cbe87e46c27776b682abf41f8c5ea9201939975869fb7b181b9455391.jpg)

## Triples as the Key: Structuring Makes Decomposition and Verification Easier in LLM-based TableQA


### Images

![157cb1f0f83129ca1888b6baa9f59929a720329481b745bb561cce60dbff8f3b.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/images/157cb1f0f83129ca1888b6baa9f59929a720329481b745bb561cce60dbff8f3b.jpg)

![5c313ca211dbabfe79447615e7e88256023af6947604e3a2c06844e06053d666.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/images/5c313ca211dbabfe79447615e7e88256023af6947604e3a2c06844e06053d666.jpg)

![768b679dfd89014f2954c78a2e0b32697b20e651d9e2593cc474477f9bf63d89.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/images/768b679dfd89014f2954c78a2e0b32697b20e651d9e2593cc474477f9bf63d89.jpg)

![d4ec13997fe3c3617cca4825e5ade9961f67557ecdfcb8f5009e318c2ba973d3.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/images/d4ec13997fe3c3617cca4825e5ade9961f67557ecdfcb8f5009e318c2ba973d3.jpg)

![e6784f2f6ed46087ec0522001c0b9db39b5706806ac401aeed6b63116dc7d0db.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/images/e6784f2f6ed46087ec0522001c0b9db39b5706806ac401aeed6b63116dc7d0db.jpg)

![f4e1b7472b984bd2f1caa97d2e9aee6131f9aef97c301a069aa6a7fd642f9f00.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/images/f4e1b7472b984bd2f1caa97d2e9aee6131f9aef97c301a069aa6a7fd642f9f00.jpg)

### Tables

![0edbdef2a9c1a6892b1db33468dafe9c150e76b7665654eb124e0c8d17d828f6.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/tables/0edbdef2a9c1a6892b1db33468dafe9c150e76b7665654eb124e0c8d17d828f6.jpg)

![36166fb9e84aa8cc6e214adf88fe74e7b7010573e7653b9de2c08f66483120f4.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/tables/36166fb9e84aa8cc6e214adf88fe74e7b7010573e7653b9de2c08f66483120f4.jpg)

![41b3228cb6246a15027e5526b40a040b1ef7d980e682e638f849d4b12a7d0a33.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/tables/41b3228cb6246a15027e5526b40a040b1ef7d980e682e638f849d4b12a7d0a33.jpg)

![46461ee5e8f035fcfc70987ba37901be86ec0d7bfd20c91eb9dc4e5c1be35aa5.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/tables/46461ee5e8f035fcfc70987ba37901be86ec0d7bfd20c91eb9dc4e5c1be35aa5.jpg)

![4c143e8e75bd21a0e37ccada46424d6e7e9ceeca463608a9e90e3e6281e8da7d.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/tables/4c143e8e75bd21a0e37ccada46424d6e7e9ceeca463608a9e90e3e6281e8da7d.jpg)

![4cb31ac0a2d286d3da79143e6537404ecd7dabf0395564fb173654cf6b719190.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/tables/4cb31ac0a2d286d3da79143e6537404ecd7dabf0395564fb173654cf6b719190.jpg)

![4d565c2cf7cd1ae2f86758a7fd9f9e9d229ceba3fb6abd3ae2d52e2d1569d672.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/tables/4d565c2cf7cd1ae2f86758a7fd9f9e9d229ceba3fb6abd3ae2d52e2d1569d672.jpg)

![78d2484be36f24a949b86555c0f9dc105128cbc3cb3ef9d3638c5b57655eeba9.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/tables/78d2484be36f24a949b86555c0f9dc105128cbc3cb3ef9d3638c5b57655eeba9.jpg)

![8c010903138da246271462bc4244d797d5859e6fb91cabc95872bc50b2614821.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/tables/8c010903138da246271462bc4244d797d5859e6fb91cabc95872bc50b2614821.jpg)

![931cce66136ccf6c7a737b4ce78dad3dbf7f0187fa4f70aec057951d74f52290.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/tables/931cce66136ccf6c7a737b4ce78dad3dbf7f0187fa4f70aec057951d74f52290.jpg)

![995c4cb5a132abadbb32c9e5d7736b33ba541cff6431f7f284e9684a988d2531.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/tables/995c4cb5a132abadbb32c9e5d7736b33ba541cff6431f7f284e9684a988d2531.jpg)

![9fcadf5aa1d9091595699f620293cd1b2b0f87893ffafbf16b8015c0f454595b.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/tables/9fcadf5aa1d9091595699f620293cd1b2b0f87893ffafbf16b8015c0f454595b.jpg)

![ae9f87dca08a39cd5a675f4ec3810d7cd1d629ee6a21cce16c182289f460ed2b.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/tables/ae9f87dca08a39cd5a675f4ec3810d7cd1d629ee6a21cce16c182289f460ed2b.jpg)

![b823b54f977ebce96be27d946147a384c75cb3f7e84499d4b293490d54ebc8ca.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/tables/b823b54f977ebce96be27d946147a384c75cb3f7e84499d4b293490d54ebc8ca.jpg)

![c2990212efddda1d24eedf5d97fa36bf0d41270273700a9e6f286c038e07377b.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/tables/c2990212efddda1d24eedf5d97fa36bf0d41270273700a9e6f286c038e07377b.jpg)

![cbc5337f86ec0b335cb69f767aec5584d89e831e451e3eaad37751f42ccd5752.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/tables/cbc5337f86ec0b335cb69f767aec5584d89e831e451e3eaad37751f42ccd5752.jpg)

![cd5dc70b0e6c45e4477f9bf9b562e0e85e68ec9513ba81bf49dbf3f394c94cc2.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/tables/cd5dc70b0e6c45e4477f9bf9b562e0e85e68ec9513ba81bf49dbf3f394c94cc2.jpg)

![d6e17c82a7d706b8af3f970a25f25cc73ec0a2a1ff70e89864a529983c5a2573.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/tables/d6e17c82a7d706b8af3f970a25f25cc73ec0a2a1ff70e89864a529983c5a2573.jpg)

![d9168f708c5f7f0f12c4f28e0dab5ca5d575cb4a86680d825995f715e463b5d4.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/tables/d9168f708c5f7f0f12c4f28e0dab5ca5d575cb4a86680d825995f715e463b5d4.jpg)

![dbda4b325e852521e9ed917eebed18c2edaa172f94c834a1043478039dcd885e.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/tables/dbda4b325e852521e9ed917eebed18c2edaa172f94c834a1043478039dcd885e.jpg)

![eed4e543dbaf388c64ee9c8d3aca92800f4ff637d8bae0d53ae545e01d34b386.jpg](../iclr_results/2136_Triples as the Key_ Structuring Makes Decomposition and Verification Easier in LLM-based TableQA/tables/eed4e543dbaf388c64ee9c8d3aca92800f4ff637d8bae0d53ae545e01d34b386.jpg)

## How DNNs break the Curse of Dimensionality: Compositionality and Symmetry Learning


### Images

![165cbe386108a68ff21df9cde6b95fbb959c1b0f7f2e0696b6f2beec46b93a46.jpg](../iclr_results/2137_How DNNs break the Curse of Dimensionality_ Compositionality and Symmetry Learning/images/165cbe386108a68ff21df9cde6b95fbb959c1b0f7f2e0696b6f2beec46b93a46.jpg)

![646fee2550122c6479551c47ba433ed901c1a4af12744d1d5886bea4a4b46e50.jpg](../iclr_results/2137_How DNNs break the Curse of Dimensionality_ Compositionality and Symmetry Learning/images/646fee2550122c6479551c47ba433ed901c1a4af12744d1d5886bea4a4b46e50.jpg)

![65fb4d408fd7f467c0fba7837ef79d49613ddb2b396be6fc27232a31bc8a8c76.jpg](../iclr_results/2137_How DNNs break the Curse of Dimensionality_ Compositionality and Symmetry Learning/images/65fb4d408fd7f467c0fba7837ef79d49613ddb2b396be6fc27232a31bc8a8c76.jpg)

![a2c4ae40a6a70a1fae339e81a0d0e05c861a7ad49210e0dab70fbb5a81601baa.jpg](../iclr_results/2137_How DNNs break the Curse of Dimensionality_ Compositionality and Symmetry Learning/images/a2c4ae40a6a70a1fae339e81a0d0e05c861a7ad49210e0dab70fbb5a81601baa.jpg)

## Protein Language Model Fitness is a Matter of Preference


### Images

![0143be1c06c1f945bce838dd67829121aad8fddf6fad6f65496c080f0a6b90a5.jpg](../iclr_results/2138_Protein Language Model Fitness is a Matter of Preference/images/0143be1c06c1f945bce838dd67829121aad8fddf6fad6f65496c080f0a6b90a5.jpg)

![06002948dd9877b012ec7b19a5f83eb8401f4c3bf3586adcdfe0ff1ccc66981f.jpg](../iclr_results/2138_Protein Language Model Fitness is a Matter of Preference/images/06002948dd9877b012ec7b19a5f83eb8401f4c3bf3586adcdfe0ff1ccc66981f.jpg)

![39e59dfc516f41a14b5b992428b1b9adc838b84560e74d4e55df1c64fe6d4107.jpg](../iclr_results/2138_Protein Language Model Fitness is a Matter of Preference/images/39e59dfc516f41a14b5b992428b1b9adc838b84560e74d4e55df1c64fe6d4107.jpg)

![486ef59322ec606e83e2c70d5851523f8d4297efddaee0b7880cbc3d6f7f098a.jpg](../iclr_results/2138_Protein Language Model Fitness is a Matter of Preference/images/486ef59322ec606e83e2c70d5851523f8d4297efddaee0b7880cbc3d6f7f098a.jpg)

![6450d36f1adee0473a5c526d106a6b8a714e121e6f70f792a49633c4068308a8.jpg](../iclr_results/2138_Protein Language Model Fitness is a Matter of Preference/images/6450d36f1adee0473a5c526d106a6b8a714e121e6f70f792a49633c4068308a8.jpg)

![b07f0f9eb2732aaff3b1514e38a90e3f4329c14a11c9d1aeae59e9d6ab339bdb.jpg](../iclr_results/2138_Protein Language Model Fitness is a Matter of Preference/images/b07f0f9eb2732aaff3b1514e38a90e3f4329c14a11c9d1aeae59e9d6ab339bdb.jpg)

![c139eb179337f91e43880e06b0b969b7e34dd86e71fc414cafcbc2e21b27cf08.jpg](../iclr_results/2138_Protein Language Model Fitness is a Matter of Preference/images/c139eb179337f91e43880e06b0b969b7e34dd86e71fc414cafcbc2e21b27cf08.jpg)

![e96b6856e0c1194cea941e8cf2d6614cef7635568f8badec816ab48793c346ad.jpg](../iclr_results/2138_Protein Language Model Fitness is a Matter of Preference/images/e96b6856e0c1194cea941e8cf2d6614cef7635568f8badec816ab48793c346ad.jpg)

![fde98cdad10c6dea78846e00c1dc3bdb2bf81cc653232a9d0fa03d583be464a9.jpg](../iclr_results/2138_Protein Language Model Fitness is a Matter of Preference/images/fde98cdad10c6dea78846e00c1dc3bdb2bf81cc653232a9d0fa03d583be464a9.jpg)

### Tables

![3062783a3c5605a98be889b9f564071d0529e313f3c7b1915086caaeccaba133.jpg](../iclr_results/2138_Protein Language Model Fitness is a Matter of Preference/tables/3062783a3c5605a98be889b9f564071d0529e313f3c7b1915086caaeccaba133.jpg)

![54b45f7819ed037b191bd5be07420751bc548adc507f72b78dece73a38a8bf7d.jpg](../iclr_results/2138_Protein Language Model Fitness is a Matter of Preference/tables/54b45f7819ed037b191bd5be07420751bc548adc507f72b78dece73a38a8bf7d.jpg)

## Gnothi Seauton: Empowering Faithful Self-Interpretability in Black-Box Transformers


### Images

![00e506fb9a3cbaec73eb70988c645e833c4658e27afd5117f7057a86ece8dd7e.jpg](../iclr_results/2139_Gnothi Seauton_ Empowering Faithful Self-Interpretability in Black-Box Transformers/images/00e506fb9a3cbaec73eb70988c645e833c4658e27afd5117f7057a86ece8dd7e.jpg)

![07747a589811d6bf062e1e7081ce75e7bf2718328df2b58b2ccb69f4286f3536.jpg](../iclr_results/2139_Gnothi Seauton_ Empowering Faithful Self-Interpretability in Black-Box Transformers/images/07747a589811d6bf062e1e7081ce75e7bf2718328df2b58b2ccb69f4286f3536.jpg)

![0839e95be030e22a9f53f86874756a76b967877e980dea581d995a221f11c42f.jpg](../iclr_results/2139_Gnothi Seauton_ Empowering Faithful Self-Interpretability in Black-Box Transformers/images/0839e95be030e22a9f53f86874756a76b967877e980dea581d995a221f11c42f.jpg)

![1b95492fe3b86b4d598fd381bbac4bb26ad8e4f1e0a7f5ba1c22f5aa9bb9491f.jpg](../iclr_results/2139_Gnothi Seauton_ Empowering Faithful Self-Interpretability in Black-Box Transformers/images/1b95492fe3b86b4d598fd381bbac4bb26ad8e4f1e0a7f5ba1c22f5aa9bb9491f.jpg)

![3030d0656c7da1b0101e9ff94063c408aa0112a2c38128632180aa9a13335ae6.jpg](../iclr_results/2139_Gnothi Seauton_ Empowering Faithful Self-Interpretability in Black-Box Transformers/images/3030d0656c7da1b0101e9ff94063c408aa0112a2c38128632180aa9a13335ae6.jpg)

![60e3451ccbe839d10a6e2cf074aa78ea7e49b14cc4cdccb376587ed914537ce9.jpg](../iclr_results/2139_Gnothi Seauton_ Empowering Faithful Self-Interpretability in Black-Box Transformers/images/60e3451ccbe839d10a6e2cf074aa78ea7e49b14cc4cdccb376587ed914537ce9.jpg)

![636c7927a4ff2fe8bcc5bec173241891695be4a28106031711a69bc49d0a5630.jpg](../iclr_results/2139_Gnothi Seauton_ Empowering Faithful Self-Interpretability in Black-Box Transformers/images/636c7927a4ff2fe8bcc5bec173241891695be4a28106031711a69bc49d0a5630.jpg)

![784e915e766a7dcfe20d4df6857bc120f1ec6b8479575729ebc6753c27984b75.jpg](../iclr_results/2139_Gnothi Seauton_ Empowering Faithful Self-Interpretability in Black-Box Transformers/images/784e915e766a7dcfe20d4df6857bc120f1ec6b8479575729ebc6753c27984b75.jpg)

![7d39c6eefe6b4822bba426be35938cc33005f92511c10cd306ef6319bbb32526.jpg](../iclr_results/2139_Gnothi Seauton_ Empowering Faithful Self-Interpretability in Black-Box Transformers/images/7d39c6eefe6b4822bba426be35938cc33005f92511c10cd306ef6319bbb32526.jpg)

![901ef20ff4a1773718e932ff259b319720d6c58c0a0da9fc9d550020b8cf4594.jpg](../iclr_results/2139_Gnothi Seauton_ Empowering Faithful Self-Interpretability in Black-Box Transformers/images/901ef20ff4a1773718e932ff259b319720d6c58c0a0da9fc9d550020b8cf4594.jpg)

![a09c1a775890f6f61b0ed987f8a81a3906947013544280e51a09d49ebad56e9e.jpg](../iclr_results/2139_Gnothi Seauton_ Empowering Faithful Self-Interpretability in Black-Box Transformers/images/a09c1a775890f6f61b0ed987f8a81a3906947013544280e51a09d49ebad56e9e.jpg)

![e21bce8c731c8d8027864279c0cd58f43935ec2799dd29471e2d0e4c52e61cbd.jpg](../iclr_results/2139_Gnothi Seauton_ Empowering Faithful Self-Interpretability in Black-Box Transformers/images/e21bce8c731c8d8027864279c0cd58f43935ec2799dd29471e2d0e4c52e61cbd.jpg)

![f86a48f1c38fdcc0c7265b0e4d3dfe32a8464ef17d3051486d5201c7950845ae.jpg](../iclr_results/2139_Gnothi Seauton_ Empowering Faithful Self-Interpretability in Black-Box Transformers/images/f86a48f1c38fdcc0c7265b0e4d3dfe32a8464ef17d3051486d5201c7950845ae.jpg)

### Tables

![091ff843b7e45ea6d0d8512bcd8604f8e0e0b560b2d347ed7ecebfb3db749465.jpg](../iclr_results/2139_Gnothi Seauton_ Empowering Faithful Self-Interpretability in Black-Box Transformers/tables/091ff843b7e45ea6d0d8512bcd8604f8e0e0b560b2d347ed7ecebfb3db749465.jpg)

![30ae981e7bfa330a2e3d49e1c1f372624d8383f6b301e83ad9eb52e68beb6c80.jpg](../iclr_results/2139_Gnothi Seauton_ Empowering Faithful Self-Interpretability in Black-Box Transformers/tables/30ae981e7bfa330a2e3d49e1c1f372624d8383f6b301e83ad9eb52e68beb6c80.jpg)

![3e06f523e7ef26829fb024117a54304ca12262298f583992f3b96c28ad295639.jpg](../iclr_results/2139_Gnothi Seauton_ Empowering Faithful Self-Interpretability in Black-Box Transformers/tables/3e06f523e7ef26829fb024117a54304ca12262298f583992f3b96c28ad295639.jpg)

![50de6083cf13cff0c1446dfa094d44d0ddcdd09cd5fc1f1554fbf3c924cead03.jpg](../iclr_results/2139_Gnothi Seauton_ Empowering Faithful Self-Interpretability in Black-Box Transformers/tables/50de6083cf13cff0c1446dfa094d44d0ddcdd09cd5fc1f1554fbf3c924cead03.jpg)

![9a083f2fe4050299d57a64e8bd39e8afe4b545dd502edad47d8343c202d64565.jpg](../iclr_results/2139_Gnothi Seauton_ Empowering Faithful Self-Interpretability in Black-Box Transformers/tables/9a083f2fe4050299d57a64e8bd39e8afe4b545dd502edad47d8343c202d64565.jpg)

![a577ec9c69cf3bd67b921becc9b3fb6c7352535f40c53e4e9bf0d4f6579f0ee2.jpg](../iclr_results/2139_Gnothi Seauton_ Empowering Faithful Self-Interpretability in Black-Box Transformers/tables/a577ec9c69cf3bd67b921becc9b3fb6c7352535f40c53e4e9bf0d4f6579f0ee2.jpg)

![a7c303f4c59c6a1630e7b6836bf7bc451ea2e0009d0f558cb6ba00007a8f8d82.jpg](../iclr_results/2139_Gnothi Seauton_ Empowering Faithful Self-Interpretability in Black-Box Transformers/tables/a7c303f4c59c6a1630e7b6836bf7bc451ea2e0009d0f558cb6ba00007a8f8d82.jpg)

![bee18c7bfd608cd0ff7384b30bfc325dcfb6ca7f30f3da8152c82a5a227392f9.jpg](../iclr_results/2139_Gnothi Seauton_ Empowering Faithful Self-Interpretability in Black-Box Transformers/tables/bee18c7bfd608cd0ff7384b30bfc325dcfb6ca7f30f3da8152c82a5a227392f9.jpg)

![c521c056bd527a315b6c3cc0d6169478dd3b29df432859fe9715e18c50345104.jpg](../iclr_results/2139_Gnothi Seauton_ Empowering Faithful Self-Interpretability in Black-Box Transformers/tables/c521c056bd527a315b6c3cc0d6169478dd3b29df432859fe9715e18c50345104.jpg)

![fa7d199840c97d4154edc4de89cee30d2b70389e22409282a3408087e3169e24.jpg](../iclr_results/2139_Gnothi Seauton_ Empowering Faithful Self-Interpretability in Black-Box Transformers/tables/fa7d199840c97d4154edc4de89cee30d2b70389e22409282a3408087e3169e24.jpg)

## McEval: Massively Multilingual Code Evaluation


### Images

![06b5e2860f7d980a3ed4fd2ecb5be565cc2b9ad6da88d193c414123a20431fe0.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/images/06b5e2860f7d980a3ed4fd2ecb5be565cc2b9ad6da88d193c414123a20431fe0.jpg)

![1d77511080542752583dd99e75ec8a6b30db7cbc5ad4c4912312ce8829eaefe7.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/images/1d77511080542752583dd99e75ec8a6b30db7cbc5ad4c4912312ce8829eaefe7.jpg)

![25059ae43c70589544b8c9cd1f011fb208c68755b67fc74563bd5b93434780bb.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/images/25059ae43c70589544b8c9cd1f011fb208c68755b67fc74563bd5b93434780bb.jpg)

![29c09573c838746b58d5f2a026e1a40e081bec8840d50c7b02ffbf1e34a1ee9d.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/images/29c09573c838746b58d5f2a026e1a40e081bec8840d50c7b02ffbf1e34a1ee9d.jpg)

![2da720fb7385b1bb30ad0c1dfec05a9222c11e2fe5456855137853c6029f0a0f.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/images/2da720fb7385b1bb30ad0c1dfec05a9222c11e2fe5456855137853c6029f0a0f.jpg)

![416f513ae2c427977ae18bf68be7190c58b1db792266a8dc18148a2e191240e9.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/images/416f513ae2c427977ae18bf68be7190c58b1db792266a8dc18148a2e191240e9.jpg)

![4564b166d7608be216fb8cad7a7f3f87b48b27f35b183b8b1c39912c46efd716.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/images/4564b166d7608be216fb8cad7a7f3f87b48b27f35b183b8b1c39912c46efd716.jpg)

![46e7bdb747ebf9830cb7dc8dc0fa6f0f8fbd4acf610fdd1d126ff6c48ddee34f.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/images/46e7bdb747ebf9830cb7dc8dc0fa6f0f8fbd4acf610fdd1d126ff6c48ddee34f.jpg)

![4e5ea00627626091b8139b0090d068a36203924f6dc92a20adaa0132ba99e52c.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/images/4e5ea00627626091b8139b0090d068a36203924f6dc92a20adaa0132ba99e52c.jpg)

![86cc72aa638640a2494013a38af92ea80309427b0994485d9b8fcea1ae984d3d.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/images/86cc72aa638640a2494013a38af92ea80309427b0994485d9b8fcea1ae984d3d.jpg)

![98d9d56c763c5866ef5a6806941b4903de96e7b9ae7add982c5c29b2a071a4fd.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/images/98d9d56c763c5866ef5a6806941b4903de96e7b9ae7add982c5c29b2a071a4fd.jpg)

![a007e7e884c88bcb3dc32b094e4aa7c6c1bdc141ae1e03172022f0f8aef05fb9.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/images/a007e7e884c88bcb3dc32b094e4aa7c6c1bdc141ae1e03172022f0f8aef05fb9.jpg)

![a7a79fa5b3f66b08e104d3f399086b60966149723e6324f661b2aec9da0df51a.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/images/a7a79fa5b3f66b08e104d3f399086b60966149723e6324f661b2aec9da0df51a.jpg)

![c34c086cd2873e90ca7ffcce72b90fa2742d706ad4b79516b6b2d665ff691382.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/images/c34c086cd2873e90ca7ffcce72b90fa2742d706ad4b79516b6b2d665ff691382.jpg)

![cc2b5e1659f10262ed29e4a5dbf971570fa868ebbeceb625391718ee9c6a9de2.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/images/cc2b5e1659f10262ed29e4a5dbf971570fa868ebbeceb625391718ee9c6a9de2.jpg)

![e62e1e52aac3f8c88dcc5fb5bf138bc23e17abfb156a97103b7747c0f7b0ec48.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/images/e62e1e52aac3f8c88dcc5fb5bf138bc23e17abfb156a97103b7747c0f7b0ec48.jpg)

![f16aaa3f643899f210670fedb7250d14ba283f8d63fe9d3d27c8b6e058d8bb33.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/images/f16aaa3f643899f210670fedb7250d14ba283f8d63fe9d3d27c8b6e058d8bb33.jpg)

![f52e1791958941cc56fd7486796d2d52fd033b33c745008a35c4fd6d226b68f2.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/images/f52e1791958941cc56fd7486796d2d52fd033b33c745008a35c4fd6d226b68f2.jpg)

![f73b203d81680f4facdcf88a29f8626dd198368d499af7bb2b656765ea3fda0f.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/images/f73b203d81680f4facdcf88a29f8626dd198368d499af7bb2b656765ea3fda0f.jpg)

### Tables

![0b485d23eb380f52070017a9328e2a5018ec1c53a910f9d96fc73be665fa1722.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/tables/0b485d23eb380f52070017a9328e2a5018ec1c53a910f9d96fc73be665fa1722.jpg)

![14b4a0d55419fe63718df140f45f4fbdfd948cb11861b4dab0e9977c0abd7a49.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/tables/14b4a0d55419fe63718df140f45f4fbdfd948cb11861b4dab0e9977c0abd7a49.jpg)

![17fc8888ba68bd902645af30094e9faa6611b3936a9faf97be1e714629e79b72.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/tables/17fc8888ba68bd902645af30094e9faa6611b3936a9faf97be1e714629e79b72.jpg)

![3c34e3561b2f6cc97ffcaf526f2f5ffcb327c266354e52d563de849a9743e52a.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/tables/3c34e3561b2f6cc97ffcaf526f2f5ffcb327c266354e52d563de849a9743e52a.jpg)

![5c961ea05c3c4dedd0686c7faf52959799874c239ce7cf3e7824a36391a150a8.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/tables/5c961ea05c3c4dedd0686c7faf52959799874c239ce7cf3e7824a36391a150a8.jpg)

![7f7c7cedfe88665c74600a725bd2b05ead63833491b9f4df44f705e42e475693.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/tables/7f7c7cedfe88665c74600a725bd2b05ead63833491b9f4df44f705e42e475693.jpg)

![893bc31ef73993ca57de5fc312ef0c27d9b1a732d57bd6c410eb0a537009d2cf.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/tables/893bc31ef73993ca57de5fc312ef0c27d9b1a732d57bd6c410eb0a537009d2cf.jpg)

![b86eaa03566280843bbc784497d1373b08d131f1435172422b9b99c21b5f45c9.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/tables/b86eaa03566280843bbc784497d1373b08d131f1435172422b9b99c21b5f45c9.jpg)

![d5f4478f1a3c45fc5cf8613936fa3be3f2179612d8b5fed529c2d64114d59960.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/tables/d5f4478f1a3c45fc5cf8613936fa3be3f2179612d8b5fed529c2d64114d59960.jpg)

![da641c9581d1202ef8f03e680008cd94e5360ee7a498146b804132709c37400f.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/tables/da641c9581d1202ef8f03e680008cd94e5360ee7a498146b804132709c37400f.jpg)

![f299e596832f42c7b97b220b4348154dc7779be8d2a66a8a0d20cee97641320e.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/tables/f299e596832f42c7b97b220b4348154dc7779be8d2a66a8a0d20cee97641320e.jpg)

![f3dc74f8078afebdca57320126f58deb3aa1f28b1608abbd946e0b3db79b065c.jpg](../iclr_results/2140_McEval_ Massively Multilingual Code Evaluation/tables/f3dc74f8078afebdca57320126f58deb3aa1f28b1608abbd946e0b3db79b065c.jpg)

## Privacy-Aware Lifelong Learning


### Images

![2f450e2c20a541d91548b4d2eb32b3f231e387c0bffe049f992bdffbffc04c92.jpg](../iclr_results/2141_Privacy-Aware Lifelong Learning/images/2f450e2c20a541d91548b4d2eb32b3f231e387c0bffe049f992bdffbffc04c92.jpg)

![772dc89a41edd06eed87a98b2445c779b2e0a71afeab574866a0db7e00031381.jpg](../iclr_results/2141_Privacy-Aware Lifelong Learning/images/772dc89a41edd06eed87a98b2445c779b2e0a71afeab574866a0db7e00031381.jpg)

![f6fc5bb5b82995b34d1f64e78b62a21bdb60d33179d61d7273945912afa3db69.jpg](../iclr_results/2141_Privacy-Aware Lifelong Learning/images/f6fc5bb5b82995b34d1f64e78b62a21bdb60d33179d61d7273945912afa3db69.jpg)

### Tables

![1e8a3f5e84119119df10d3a1e49b4142cf204d82eb34d7c7230eba156a6a72aa.jpg](../iclr_results/2141_Privacy-Aware Lifelong Learning/tables/1e8a3f5e84119119df10d3a1e49b4142cf204d82eb34d7c7230eba156a6a72aa.jpg)

![3cedcae0d8b52422cdf8d49ca00753ed92f3a7191a224e77a3538015d6d3d847.jpg](../iclr_results/2141_Privacy-Aware Lifelong Learning/tables/3cedcae0d8b52422cdf8d49ca00753ed92f3a7191a224e77a3538015d6d3d847.jpg)

![4057b2bfbb032cb2c27d3ba752ac34c62e8542b1f31d81f77a7cc9703a2df79d.jpg](../iclr_results/2141_Privacy-Aware Lifelong Learning/tables/4057b2bfbb032cb2c27d3ba752ac34c62e8542b1f31d81f77a7cc9703a2df79d.jpg)

![4a660821a061622ea05f3f9f436ccdf939b9bb5473b33ffc8b35d31f84e84af5.jpg](../iclr_results/2141_Privacy-Aware Lifelong Learning/tables/4a660821a061622ea05f3f9f436ccdf939b9bb5473b33ffc8b35d31f84e84af5.jpg)

![5c9e5802f9770603d3b1907e29d2657e045fe32f60aaf6f37e0e8b3c8aa7a066.jpg](../iclr_results/2141_Privacy-Aware Lifelong Learning/tables/5c9e5802f9770603d3b1907e29d2657e045fe32f60aaf6f37e0e8b3c8aa7a066.jpg)

![66f3d7a8b892d637493893d18e0c4070a22f949ea919e8bcd1b7c6bb6fe9e77c.jpg](../iclr_results/2141_Privacy-Aware Lifelong Learning/tables/66f3d7a8b892d637493893d18e0c4070a22f949ea919e8bcd1b7c6bb6fe9e77c.jpg)

![8fa06a967966a0c8cdd040576bf1c220f080f4179f06f09deca2edd364e6643b.jpg](../iclr_results/2141_Privacy-Aware Lifelong Learning/tables/8fa06a967966a0c8cdd040576bf1c220f080f4179f06f09deca2edd364e6643b.jpg)

![c4c2ee22e25480bc0b5dba8f0d04948e6766c2117ddf74b13c56df165722b99c.jpg](../iclr_results/2141_Privacy-Aware Lifelong Learning/tables/c4c2ee22e25480bc0b5dba8f0d04948e6766c2117ddf74b13c56df165722b99c.jpg)

![c74f91c653b20c03ecd3f295b5c582051d6d29413f55b2ec0ce1a42327937653.jpg](../iclr_results/2141_Privacy-Aware Lifelong Learning/tables/c74f91c653b20c03ecd3f295b5c582051d6d29413f55b2ec0ce1a42327937653.jpg)

![f8b6f3b00eda7cdd13e9cf5b66b28a7e028ff5ee2d8d6f9e231426ed89009328.jpg](../iclr_results/2141_Privacy-Aware Lifelong Learning/tables/f8b6f3b00eda7cdd13e9cf5b66b28a7e028ff5ee2d8d6f9e231426ed89009328.jpg)

## MRAG-Bench: Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models


### Images

![4e278d0378f6206dc58a694b33f42df71b7ffa094e95ceaaa49475779c2ceb7f.jpg](../iclr_results/2142_MRAG-Bench_ Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models/images/4e278d0378f6206dc58a694b33f42df71b7ffa094e95ceaaa49475779c2ceb7f.jpg)

![528dd66175d2fd54126be14c35cbe630c2b5995d6d44d193042395702f7ab978.jpg](../iclr_results/2142_MRAG-Bench_ Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models/images/528dd66175d2fd54126be14c35cbe630c2b5995d6d44d193042395702f7ab978.jpg)

![6a71cd0f5f867180d74f36c927a987f265c7c285ef99c0bf893ac65496c02851.jpg](../iclr_results/2142_MRAG-Bench_ Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models/images/6a71cd0f5f867180d74f36c927a987f265c7c285ef99c0bf893ac65496c02851.jpg)

![8377041ba1e6434d26c58caaf11f1d3613e8abd2654b5e3a65642c318d621ca0.jpg](../iclr_results/2142_MRAG-Bench_ Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models/images/8377041ba1e6434d26c58caaf11f1d3613e8abd2654b5e3a65642c318d621ca0.jpg)

![b0eab733273cc5621a0b688c91505a07495b9e44919a6d307f5be230c2d92eb5.jpg](../iclr_results/2142_MRAG-Bench_ Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models/images/b0eab733273cc5621a0b688c91505a07495b9e44919a6d307f5be230c2d92eb5.jpg)

![f58ba0a3f8dda33b383917f5d7f1faf5d9bbfc17ec8efa88f4427f3ce57f8dca.jpg](../iclr_results/2142_MRAG-Bench_ Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models/images/f58ba0a3f8dda33b383917f5d7f1faf5d9bbfc17ec8efa88f4427f3ce57f8dca.jpg)

![f7c7aaf9cd097220a39c25e7b17b9815c16af8f9017a3635f1fdfca3e3048cc4.jpg](../iclr_results/2142_MRAG-Bench_ Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models/images/f7c7aaf9cd097220a39c25e7b17b9815c16af8f9017a3635f1fdfca3e3048cc4.jpg)

### Tables

![67acee8becda44395fb0c88ff0f090205d7500a0f35760ccc6f2e018618b8371.jpg](../iclr_results/2142_MRAG-Bench_ Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models/tables/67acee8becda44395fb0c88ff0f090205d7500a0f35760ccc6f2e018618b8371.jpg)

![ab73abd5e564b9a1113bab7c76e03b3b562a785852993fd0c3ab93b30a02cea1.jpg](../iclr_results/2142_MRAG-Bench_ Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models/tables/ab73abd5e564b9a1113bab7c76e03b3b562a785852993fd0c3ab93b30a02cea1.jpg)

![ab9acd4ccc458903132c45eed87c2133d6d75f8180211f5938784d6036fbf796.jpg](../iclr_results/2142_MRAG-Bench_ Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models/tables/ab9acd4ccc458903132c45eed87c2133d6d75f8180211f5938784d6036fbf796.jpg)

![acb46adf78931c43cd098efb66fde4efa961d8c9268b891b5834a91ff488d3c2.jpg](../iclr_results/2142_MRAG-Bench_ Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models/tables/acb46adf78931c43cd098efb66fde4efa961d8c9268b891b5834a91ff488d3c2.jpg)

![bac7bffbcbbc30df0e008a0728d2c119b6c9bd40f37461d93cf05f5e8bf8cfbb.jpg](../iclr_results/2142_MRAG-Bench_ Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models/tables/bac7bffbcbbc30df0e008a0728d2c119b6c9bd40f37461d93cf05f5e8bf8cfbb.jpg)

![fcca83d314e63b35ee0f78c430b4d4c8e1ccd53a2d9e80384b9c488c4f77d405.jpg](../iclr_results/2142_MRAG-Bench_ Vision-Centric Evaluation for Retrieval-Augmented Multimodal Models/tables/fcca83d314e63b35ee0f78c430b4d4c8e1ccd53a2d9e80384b9c488c4f77d405.jpg)

## Learning Graph Invariance by Harnessing Spuriosity


### Images

![0cfcca7f84378450aac8c1b7486c4d47c09bc5277a637ded526d1decdc3b12e2.jpg](../iclr_results/2143_Learning Graph Invariance by Harnessing Spuriosity/images/0cfcca7f84378450aac8c1b7486c4d47c09bc5277a637ded526d1decdc3b12e2.jpg)

![5571facfbfd08e8efcaf8b485176f42a1e8cc45339ac1ccfe8297a5b7ac3a94a.jpg](../iclr_results/2143_Learning Graph Invariance by Harnessing Spuriosity/images/5571facfbfd08e8efcaf8b485176f42a1e8cc45339ac1ccfe8297a5b7ac3a94a.jpg)

![7cb3b95889395bc5e406e3f29fc2b19c8910e3be8071ea3bae7518f277090131.jpg](../iclr_results/2143_Learning Graph Invariance by Harnessing Spuriosity/images/7cb3b95889395bc5e406e3f29fc2b19c8910e3be8071ea3bae7518f277090131.jpg)

![7f2c844310573a6d9699b0b49e99580352b0284d037cfee70dc71db3eb9fb1a3.jpg](../iclr_results/2143_Learning Graph Invariance by Harnessing Spuriosity/images/7f2c844310573a6d9699b0b49e99580352b0284d037cfee70dc71db3eb9fb1a3.jpg)

![9434dab53afc417c8ef2a1b30d4da30f7366ce5bbbc42cb0535414b0ba402b59.jpg](../iclr_results/2143_Learning Graph Invariance by Harnessing Spuriosity/images/9434dab53afc417c8ef2a1b30d4da30f7366ce5bbbc42cb0535414b0ba402b59.jpg)

![cddd468b977faa99b41bb739251cd6076469d080cf3b9f10851d97160c879c98.jpg](../iclr_results/2143_Learning Graph Invariance by Harnessing Spuriosity/images/cddd468b977faa99b41bb739251cd6076469d080cf3b9f10851d97160c879c98.jpg)

![e3b9d00d892e853034f3eef76cc340ca699fa10cbee433022e2d1ac29aa13da5.jpg](../iclr_results/2143_Learning Graph Invariance by Harnessing Spuriosity/images/e3b9d00d892e853034f3eef76cc340ca699fa10cbee433022e2d1ac29aa13da5.jpg)

### Tables

![0ffd91dedb68293262f15a892fef79a5f23ccd763a6d86b7ca490f053b6638f3.jpg](../iclr_results/2143_Learning Graph Invariance by Harnessing Spuriosity/tables/0ffd91dedb68293262f15a892fef79a5f23ccd763a6d86b7ca490f053b6638f3.jpg)

![328823ba058fa4f58c42344ca5e0c15a31c2c202393e73f601faa39d4d57cf1c.jpg](../iclr_results/2143_Learning Graph Invariance by Harnessing Spuriosity/tables/328823ba058fa4f58c42344ca5e0c15a31c2c202393e73f601faa39d4d57cf1c.jpg)

![5672864342e26d53c91d3350aa4f4937ff7831b1b97c1c4dab54cc0fedfa0c8d.jpg](../iclr_results/2143_Learning Graph Invariance by Harnessing Spuriosity/tables/5672864342e26d53c91d3350aa4f4937ff7831b1b97c1c4dab54cc0fedfa0c8d.jpg)

![831704a5ca91aa3f35f1c72b97d07903e44ead2709a382bf19174315ba941edc.jpg](../iclr_results/2143_Learning Graph Invariance by Harnessing Spuriosity/tables/831704a5ca91aa3f35f1c72b97d07903e44ead2709a382bf19174315ba941edc.jpg)

![86cead2864619dacf85db0f5e1700d51864bf18e99f98c03ebe8148ebfad33cb.jpg](../iclr_results/2143_Learning Graph Invariance by Harnessing Spuriosity/tables/86cead2864619dacf85db0f5e1700d51864bf18e99f98c03ebe8148ebfad33cb.jpg)

![a5b209356fe344a375f951ac0599efde8b4c931b2fc24262b1db35cff63528fc.jpg](../iclr_results/2143_Learning Graph Invariance by Harnessing Spuriosity/tables/a5b209356fe344a375f951ac0599efde8b4c931b2fc24262b1db35cff63528fc.jpg)

![b439690e6723ea044742dfc47390fa5554e93aaf03d4e95b4e8c237b1a09db19.jpg](../iclr_results/2143_Learning Graph Invariance by Harnessing Spuriosity/tables/b439690e6723ea044742dfc47390fa5554e93aaf03d4e95b4e8c237b1a09db19.jpg)

![e3309dd62c8176bfbd2fc778df2d8181928291077b414e1b5d5c506cc0ded993.jpg](../iclr_results/2143_Learning Graph Invariance by Harnessing Spuriosity/tables/e3309dd62c8176bfbd2fc778df2d8181928291077b414e1b5d5c506cc0ded993.jpg)

## Mitigating Spurious Correlations in Zero-Shot Multimodal Models


### Images

![0dd3c15bc8b0ec04eb6b0751539aac0ed7fdfddacdc745858a9dbd2395c6fcc4.jpg](../iclr_results/2144_Mitigating Spurious Correlations in Zero-Shot Multimodal Models/images/0dd3c15bc8b0ec04eb6b0751539aac0ed7fdfddacdc745858a9dbd2395c6fcc4.jpg)

![1f2dbf846000e26c3bdde937b74c3b46a84000b5ec98098ef5ce91a5229725e9.jpg](../iclr_results/2144_Mitigating Spurious Correlations in Zero-Shot Multimodal Models/images/1f2dbf846000e26c3bdde937b74c3b46a84000b5ec98098ef5ce91a5229725e9.jpg)

![25aecefc62626225e16b59f6bfefb441fa8767b637c7a6f74c7ac15458ffc440.jpg](../iclr_results/2144_Mitigating Spurious Correlations in Zero-Shot Multimodal Models/images/25aecefc62626225e16b59f6bfefb441fa8767b637c7a6f74c7ac15458ffc440.jpg)

![45607287725feb1ecaf4ad8201b02ff32f040fc8f9b90a0b555aa483e50bc048.jpg](../iclr_results/2144_Mitigating Spurious Correlations in Zero-Shot Multimodal Models/images/45607287725feb1ecaf4ad8201b02ff32f040fc8f9b90a0b555aa483e50bc048.jpg)

![71fb368f01229775a459c0a59096b1af02a498f61c6cf49314b153f60e07857c.jpg](../iclr_results/2144_Mitigating Spurious Correlations in Zero-Shot Multimodal Models/images/71fb368f01229775a459c0a59096b1af02a498f61c6cf49314b153f60e07857c.jpg)

![729a346f1b55d2b61eb6432cfffdf84ec9ad2a49140cd3f6eb46ac881c62e899.jpg](../iclr_results/2144_Mitigating Spurious Correlations in Zero-Shot Multimodal Models/images/729a346f1b55d2b61eb6432cfffdf84ec9ad2a49140cd3f6eb46ac881c62e899.jpg)

![7b62cc9e9d20d0eb58b38f376815d7556f1db405524b5a23d79d915cac060a7b.jpg](../iclr_results/2144_Mitigating Spurious Correlations in Zero-Shot Multimodal Models/images/7b62cc9e9d20d0eb58b38f376815d7556f1db405524b5a23d79d915cac060a7b.jpg)

![912323b992610fbc1cabbf17a2a426fa67656c18a8e810def8f9f1b0dfbe5db0.jpg](../iclr_results/2144_Mitigating Spurious Correlations in Zero-Shot Multimodal Models/images/912323b992610fbc1cabbf17a2a426fa67656c18a8e810def8f9f1b0dfbe5db0.jpg)

![d476ea5be5d90820aad9729a65a21bdb71dbfb85aa779fcc0b3f6616b4418c99.jpg](../iclr_results/2144_Mitigating Spurious Correlations in Zero-Shot Multimodal Models/images/d476ea5be5d90820aad9729a65a21bdb71dbfb85aa779fcc0b3f6616b4418c99.jpg)

![e5f840de51ed182eed00521b63a03c43af38907ea31e316a1f67823a25ce9c4b.jpg](../iclr_results/2144_Mitigating Spurious Correlations in Zero-Shot Multimodal Models/images/e5f840de51ed182eed00521b63a03c43af38907ea31e316a1f67823a25ce9c4b.jpg)

### Tables

![22d91e6f01df81e20747cf1a26a89e1a0098b59c94b3c7bc1748fafdf2103054.jpg](../iclr_results/2144_Mitigating Spurious Correlations in Zero-Shot Multimodal Models/tables/22d91e6f01df81e20747cf1a26a89e1a0098b59c94b3c7bc1748fafdf2103054.jpg)

![2e4475f15c0f14141085229d07bd79bc4cc012302ade75f1cf39876d903afdc4.jpg](../iclr_results/2144_Mitigating Spurious Correlations in Zero-Shot Multimodal Models/tables/2e4475f15c0f14141085229d07bd79bc4cc012302ade75f1cf39876d903afdc4.jpg)

![2f662e6f22c18faa9a511ea15a54da9ef48322fb5aac4d0b62b8d812134eaff7.jpg](../iclr_results/2144_Mitigating Spurious Correlations in Zero-Shot Multimodal Models/tables/2f662e6f22c18faa9a511ea15a54da9ef48322fb5aac4d0b62b8d812134eaff7.jpg)

![34c194eb7612c1a3ca2b8f27aa02a9e02eab490082927b01a9237ccd819c9cce.jpg](../iclr_results/2144_Mitigating Spurious Correlations in Zero-Shot Multimodal Models/tables/34c194eb7612c1a3ca2b8f27aa02a9e02eab490082927b01a9237ccd819c9cce.jpg)

![39fc654b654963bd452f778e6e4754087df66b6fa683b0f54a901ad8f30a17d0.jpg](../iclr_results/2144_Mitigating Spurious Correlations in Zero-Shot Multimodal Models/tables/39fc654b654963bd452f778e6e4754087df66b6fa683b0f54a901ad8f30a17d0.jpg)

![5041e13e918bd1ea1bee3cc8d7d73c8eda75714ce4666d5b49a99531d22142e6.jpg](../iclr_results/2144_Mitigating Spurious Correlations in Zero-Shot Multimodal Models/tables/5041e13e918bd1ea1bee3cc8d7d73c8eda75714ce4666d5b49a99531d22142e6.jpg)

![640d3f84ef5c9399d363c52d9e0c40be0b1aa2e83d57116c49eecb39b8f9d662.jpg](../iclr_results/2144_Mitigating Spurious Correlations in Zero-Shot Multimodal Models/tables/640d3f84ef5c9399d363c52d9e0c40be0b1aa2e83d57116c49eecb39b8f9d662.jpg)

![6a38e91d206867321b8dfdd1fe59b016c2567ad02464300d0ff3b3e730d57cd3.jpg](../iclr_results/2144_Mitigating Spurious Correlations in Zero-Shot Multimodal Models/tables/6a38e91d206867321b8dfdd1fe59b016c2567ad02464300d0ff3b3e730d57cd3.jpg)

![a083a732eb858e6f8f5582b16ba0c60bca20e2cb7998f7198910269369917c20.jpg](../iclr_results/2144_Mitigating Spurious Correlations in Zero-Shot Multimodal Models/tables/a083a732eb858e6f8f5582b16ba0c60bca20e2cb7998f7198910269369917c20.jpg)

![c04c5fdbca7c6b568aadebabaf6b6fe0a68b278d2a1399c61f21945ad581c2b7.jpg](../iclr_results/2144_Mitigating Spurious Correlations in Zero-Shot Multimodal Models/tables/c04c5fdbca7c6b568aadebabaf6b6fe0a68b278d2a1399c61f21945ad581c2b7.jpg)

![e27ebd4a607ba8a9f3e938b496e890c368321d93fd2558f153190592f10c41b4.jpg](../iclr_results/2144_Mitigating Spurious Correlations in Zero-Shot Multimodal Models/tables/e27ebd4a607ba8a9f3e938b496e890c368321d93fd2558f153190592f10c41b4.jpg)

![e7ba6ef36a930bd2cd665e98abecec49a8fed7811b6be4e5fe89c1ba034fd88a.jpg](../iclr_results/2144_Mitigating Spurious Correlations in Zero-Shot Multimodal Models/tables/e7ba6ef36a930bd2cd665e98abecec49a8fed7811b6be4e5fe89c1ba034fd88a.jpg)

## The Breakdown of Gaussian Universality in Classification of High-dimensional Linear Factor Mixtures


### Images

![124a5ed14705dbb4e6699e84f41dca748ce00e201eef772f409c2232b595d9d5.jpg](../iclr_results/2145_The Breakdown of Gaussian Universality in Classification of High-dimensional Linear Factor Mixtures/images/124a5ed14705dbb4e6699e84f41dca748ce00e201eef772f409c2232b595d9d5.jpg)

![133bdacd4facecf0ead7bff0e5cecf80240e03268887cdd6e3031ada1d4434d9.jpg](../iclr_results/2145_The Breakdown of Gaussian Universality in Classification of High-dimensional Linear Factor Mixtures/images/133bdacd4facecf0ead7bff0e5cecf80240e03268887cdd6e3031ada1d4434d9.jpg)

![1e44bef567e90968e896a21dbdebf29310707fb52038bd041288094b33eb5e6e.jpg](../iclr_results/2145_The Breakdown of Gaussian Universality in Classification of High-dimensional Linear Factor Mixtures/images/1e44bef567e90968e896a21dbdebf29310707fb52038bd041288094b33eb5e6e.jpg)

![243f6dc9192c6f61610b7dbcbbe0175694bd0fb3ff3a6975f9dfa149e1fe8e61.jpg](../iclr_results/2145_The Breakdown of Gaussian Universality in Classification of High-dimensional Linear Factor Mixtures/images/243f6dc9192c6f61610b7dbcbbe0175694bd0fb3ff3a6975f9dfa149e1fe8e61.jpg)

![424cbfb637a613e11b5b616b59361e111ed2848863fc25e6116836ce457994ab.jpg](../iclr_results/2145_The Breakdown of Gaussian Universality in Classification of High-dimensional Linear Factor Mixtures/images/424cbfb637a613e11b5b616b59361e111ed2848863fc25e6116836ce457994ab.jpg)

![4d6528261e3bb50f86fbfeb5b7be24d2b8b5c792de9cb8f51c156a4cb54736ac.jpg](../iclr_results/2145_The Breakdown of Gaussian Universality in Classification of High-dimensional Linear Factor Mixtures/images/4d6528261e3bb50f86fbfeb5b7be24d2b8b5c792de9cb8f51c156a4cb54736ac.jpg)

![4dba7814dd153f9e5ee0c35f3cfc8108fe249e982e83b0aa5e39567b0a07865d.jpg](../iclr_results/2145_The Breakdown of Gaussian Universality in Classification of High-dimensional Linear Factor Mixtures/images/4dba7814dd153f9e5ee0c35f3cfc8108fe249e982e83b0aa5e39567b0a07865d.jpg)

![8ec61f5817e05a6ceaaa1d9e86524f9f65c9954c392907d137c41f61f58f0510.jpg](../iclr_results/2145_The Breakdown of Gaussian Universality in Classification of High-dimensional Linear Factor Mixtures/images/8ec61f5817e05a6ceaaa1d9e86524f9f65c9954c392907d137c41f61f58f0510.jpg)

![9a412df3b9231a54994b3415a258b37903718979618f95a1f4b453ff8ef9e77f.jpg](../iclr_results/2145_The Breakdown of Gaussian Universality in Classification of High-dimensional Linear Factor Mixtures/images/9a412df3b9231a54994b3415a258b37903718979618f95a1f4b453ff8ef9e77f.jpg)

![eaddcbc2b812b100272279fcf33c89da1bb5729e22c9bcf10c7d44a8445271f0.jpg](../iclr_results/2145_The Breakdown of Gaussian Universality in Classification of High-dimensional Linear Factor Mixtures/images/eaddcbc2b812b100272279fcf33c89da1bb5729e22c9bcf10c7d44a8445271f0.jpg)

## Accurate and Scalable Graph Neural Networks via Message Invariance


### Images

![0e5fe6f826863da03af3c46077b7517403990262b4d8aef47ca7d27f87bc3782.jpg](../iclr_results/2146_Accurate and Scalable Graph Neural Networks via Message Invariance/images/0e5fe6f826863da03af3c46077b7517403990262b4d8aef47ca7d27f87bc3782.jpg)

![30bc0a570965333beed72df038c9f246b2283bd17c3ce5ca74ee572205249727.jpg](../iclr_results/2146_Accurate and Scalable Graph Neural Networks via Message Invariance/images/30bc0a570965333beed72df038c9f246b2283bd17c3ce5ca74ee572205249727.jpg)

![7eff6433d7fc129fafc454db132e81f20371b83f95b06f48a0433fd16f8b70cf.jpg](../iclr_results/2146_Accurate and Scalable Graph Neural Networks via Message Invariance/images/7eff6433d7fc129fafc454db132e81f20371b83f95b06f48a0433fd16f8b70cf.jpg)

![85473920a0a7b66b9e03793f74b4fbfc5b4d7e2b25d9a13188a379b2cf96a83d.jpg](../iclr_results/2146_Accurate and Scalable Graph Neural Networks via Message Invariance/images/85473920a0a7b66b9e03793f74b4fbfc5b4d7e2b25d9a13188a379b2cf96a83d.jpg)

![8f786852b55d117704d459c1f8f89119dcf0857ed9dd72ff62d18928974cfe06.jpg](../iclr_results/2146_Accurate and Scalable Graph Neural Networks via Message Invariance/images/8f786852b55d117704d459c1f8f89119dcf0857ed9dd72ff62d18928974cfe06.jpg)

![9c94b1b0aa6f20a8d61977956a2406057eb48d751dad3546dca4d008fb25cf3d.jpg](../iclr_results/2146_Accurate and Scalable Graph Neural Networks via Message Invariance/images/9c94b1b0aa6f20a8d61977956a2406057eb48d751dad3546dca4d008fb25cf3d.jpg)

![b0ee31acfdd2c98b07a721dfcf94b01fd1a6898c852f8abfbbb6ea567b97c5fe.jpg](../iclr_results/2146_Accurate and Scalable Graph Neural Networks via Message Invariance/images/b0ee31acfdd2c98b07a721dfcf94b01fd1a6898c852f8abfbbb6ea567b97c5fe.jpg)

![c17330744da7c372f6912fb253156e0b0f7bd6da6d0b6c02482920af8c01293f.jpg](../iclr_results/2146_Accurate and Scalable Graph Neural Networks via Message Invariance/images/c17330744da7c372f6912fb253156e0b0f7bd6da6d0b6c02482920af8c01293f.jpg)

![fbcc35bb4db80897eaaef2d36f5eaa67565c4464fc24f228f4e071078989df68.jpg](../iclr_results/2146_Accurate and Scalable Graph Neural Networks via Message Invariance/images/fbcc35bb4db80897eaaef2d36f5eaa67565c4464fc24f228f4e071078989df68.jpg)

### Tables

![29c421c4a7028d52acf49aa8febc35bfffa01b083255454b179c053da1e13d4c.jpg](../iclr_results/2146_Accurate and Scalable Graph Neural Networks via Message Invariance/tables/29c421c4a7028d52acf49aa8febc35bfffa01b083255454b179c053da1e13d4c.jpg)

![611f4ec349aa26831c8824e0b0943fc432caaa080f8cd0aa83741b7b1d562b46.jpg](../iclr_results/2146_Accurate and Scalable Graph Neural Networks via Message Invariance/tables/611f4ec349aa26831c8824e0b0943fc432caaa080f8cd0aa83741b7b1d562b46.jpg)

![67a0ab98ba1ddacbf847644588b285b6bdffdd7d489ad6423616e3e4053bb686.jpg](../iclr_results/2146_Accurate and Scalable Graph Neural Networks via Message Invariance/tables/67a0ab98ba1ddacbf847644588b285b6bdffdd7d489ad6423616e3e4053bb686.jpg)

![90a765f9107c1ea948e82d4350761a469cb4eab88e567c537a5a56f9a2c782eb.jpg](../iclr_results/2146_Accurate and Scalable Graph Neural Networks via Message Invariance/tables/90a765f9107c1ea948e82d4350761a469cb4eab88e567c537a5a56f9a2c782eb.jpg)

![a38f4e0e06bbe614baf0617815473a0181333eda157aeb09d6a1a87ec0d700d7.jpg](../iclr_results/2146_Accurate and Scalable Graph Neural Networks via Message Invariance/tables/a38f4e0e06bbe614baf0617815473a0181333eda157aeb09d6a1a87ec0d700d7.jpg)

![aa1d6c3a7335c992ab1def16742d336ab05b39a62e11d65143b1f19051bb5e76.jpg](../iclr_results/2146_Accurate and Scalable Graph Neural Networks via Message Invariance/tables/aa1d6c3a7335c992ab1def16742d336ab05b39a62e11d65143b1f19051bb5e76.jpg)

![b44bd9450bfb86225d3550e752bea74be14864fb3445a640b7120a59d522b7e5.jpg](../iclr_results/2146_Accurate and Scalable Graph Neural Networks via Message Invariance/tables/b44bd9450bfb86225d3550e752bea74be14864fb3445a640b7120a59d522b7e5.jpg)

![ec368822eac0bdc8e4a29ee5a1867b36c297f2f44b900d4fd030b00abc544d9b.jpg](../iclr_results/2146_Accurate and Scalable Graph Neural Networks via Message Invariance/tables/ec368822eac0bdc8e4a29ee5a1867b36c297f2f44b900d4fd030b00abc544d9b.jpg)

![f255bd0507dc711f0acb30e467710b3a04abfcd3490cb4e18c72301e364edc69.jpg](../iclr_results/2146_Accurate and Scalable Graph Neural Networks via Message Invariance/tables/f255bd0507dc711f0acb30e467710b3a04abfcd3490cb4e18c72301e364edc69.jpg)

## Strong Preferences Affect the Robustness of Preference Models and Value Alignment


### Images

![3512ed4a0035713801c956ced4663fa48f91a685d519b722f32d7dcd16746511.jpg](../iclr_results/2147_Strong Preferences Affect the Robustness of Preference Models and Value Alignment/images/3512ed4a0035713801c956ced4663fa48f91a685d519b722f32d7dcd16746511.jpg)

![432a655fa3b207cba52c3d3e50ec054066971590e1c9507d7d27d5c2b659f94b.jpg](../iclr_results/2147_Strong Preferences Affect the Robustness of Preference Models and Value Alignment/images/432a655fa3b207cba52c3d3e50ec054066971590e1c9507d7d27d5c2b659f94b.jpg)

![6b19d4375a08ffd48adaa9d2cc02e1bf150a4866c953fa3732fe21713638e6e1.jpg](../iclr_results/2147_Strong Preferences Affect the Robustness of Preference Models and Value Alignment/images/6b19d4375a08ffd48adaa9d2cc02e1bf150a4866c953fa3732fe21713638e6e1.jpg)

![95084610957879534298948a87c511a95286b3ed3dde555240754aa13c79c629.jpg](../iclr_results/2147_Strong Preferences Affect the Robustness of Preference Models and Value Alignment/images/95084610957879534298948a87c511a95286b3ed3dde555240754aa13c79c629.jpg)

![a0cf1a96db5e8fc455f94d972fc58a3f86cc598ff73c31168b43d70a9773daf5.jpg](../iclr_results/2147_Strong Preferences Affect the Robustness of Preference Models and Value Alignment/images/a0cf1a96db5e8fc455f94d972fc58a3f86cc598ff73c31168b43d70a9773daf5.jpg)

### Tables

![45b7fb518ce1357cf3c706d16b73183218e4fed0743b8e575dbd22bcafcd239d.jpg](../iclr_results/2147_Strong Preferences Affect the Robustness of Preference Models and Value Alignment/tables/45b7fb518ce1357cf3c706d16b73183218e4fed0743b8e575dbd22bcafcd239d.jpg)

![c3d17d8000a05b76bfa1d20fc77c7f88dc4b0ba824b4cc7621eed4bf115c4014.jpg](../iclr_results/2147_Strong Preferences Affect the Robustness of Preference Models and Value Alignment/tables/c3d17d8000a05b76bfa1d20fc77c7f88dc4b0ba824b4cc7621eed4bf115c4014.jpg)

![e5dda6b1efde0e29a356df79d0db84156996a7fe2ad07d1d70088ee892374629.jpg](../iclr_results/2147_Strong Preferences Affect the Robustness of Preference Models and Value Alignment/tables/e5dda6b1efde0e29a356df79d0db84156996a7fe2ad07d1d70088ee892374629.jpg)

## VICtoR: Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation


### Images

![0931ec3ad5bad5ec0081776666e0a7965abbfb56511963604aa366af5397c34e.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/images/0931ec3ad5bad5ec0081776666e0a7965abbfb56511963604aa366af5397c34e.jpg)

![1f0aa318efd9719f509c705fd06784bdaa8d12e8aa169ad1b08617c4c3ef0ac7.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/images/1f0aa318efd9719f509c705fd06784bdaa8d12e8aa169ad1b08617c4c3ef0ac7.jpg)

![47d9e8c926916edc3efa210b9d88044b95ea4ab3d0e06a77dd1cb8779bd10e67.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/images/47d9e8c926916edc3efa210b9d88044b95ea4ab3d0e06a77dd1cb8779bd10e67.jpg)

![47ea2e948cea2cccefe0efb639914da445d69ed1282f87c3150fff0c0ba0506b.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/images/47ea2e948cea2cccefe0efb639914da445d69ed1282f87c3150fff0c0ba0506b.jpg)

![49724c4cd2d679e2a0ccdb686b87a033546bbcb4438be454bebf80c906ff7bd1.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/images/49724c4cd2d679e2a0ccdb686b87a033546bbcb4438be454bebf80c906ff7bd1.jpg)

![5bd308608c8d30337c79872c6f66cbac39ef5a5bc894adfeb63358173830d231.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/images/5bd308608c8d30337c79872c6f66cbac39ef5a5bc894adfeb63358173830d231.jpg)

![5d690e15b1259b01bd51f5e3f3cecb540a4f64f6c7394bbebcc08abcd3349612.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/images/5d690e15b1259b01bd51f5e3f3cecb540a4f64f6c7394bbebcc08abcd3349612.jpg)

![6b87478dc816448714be1f01bd04731176c9fdacd4a3c4614ea452c0c92edeef.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/images/6b87478dc816448714be1f01bd04731176c9fdacd4a3c4614ea452c0c92edeef.jpg)

![6ec89d3614ef0e887926770490824d055e4ecb26f5ee2dde07edba0d4f001881.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/images/6ec89d3614ef0e887926770490824d055e4ecb26f5ee2dde07edba0d4f001881.jpg)

![8a6dfc2059089fc7773beae3244ebeb3367f90547ba1435d459e60dda5646045.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/images/8a6dfc2059089fc7773beae3244ebeb3367f90547ba1435d459e60dda5646045.jpg)

![a47eb524b98d01da3833ead2fe5c7fc5a6ffb760a5ab47b83fb41687bcb6379a.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/images/a47eb524b98d01da3833ead2fe5c7fc5a6ffb760a5ab47b83fb41687bcb6379a.jpg)

![ad6ba28d1212cda86d0d3ed2987d93c92de1322dd4a012e6cd177e0442fd020b.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/images/ad6ba28d1212cda86d0d3ed2987d93c92de1322dd4a012e6cd177e0442fd020b.jpg)

![c95637b6a63929d573ff65614be8366e3a5abe2b3df9c37cc45ea74f3ca69682.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/images/c95637b6a63929d573ff65614be8366e3a5abe2b3df9c37cc45ea74f3ca69682.jpg)

![cac6aeaa2a48436943a5708ff272f608d15ec24cac54fda918e4a78ba41b9bc4.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/images/cac6aeaa2a48436943a5708ff272f608d15ec24cac54fda918e4a78ba41b9bc4.jpg)

![e6d924cfc60e386d3ac8bb986bf8f68ae8613ec7442972909b6eef77ea9c6ea8.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/images/e6d924cfc60e386d3ac8bb986bf8f68ae8613ec7442972909b6eef77ea9c6ea8.jpg)

![ecab2d2e14d1858d75ce956fc86266e7312a6c356f6560456c2f06a18e836b2e.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/images/ecab2d2e14d1858d75ce956fc86266e7312a6c356f6560456c2f06a18e836b2e.jpg)

### Tables

![0fcfff7c43b8b45d0fc4ad3b5e18d0f5b542d38a2fe5c20d7c1b52c3f0ca77f0.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/tables/0fcfff7c43b8b45d0fc4ad3b5e18d0f5b542d38a2fe5c20d7c1b52c3f0ca77f0.jpg)

![12bab61609939ab4b72a4b1b18733bab7f745cbd1e4527b2c7546263134f83cb.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/tables/12bab61609939ab4b72a4b1b18733bab7f745cbd1e4527b2c7546263134f83cb.jpg)

![19584c31afc609da72416be8d4bc628b53b6627b001d41e2947de5046a73deda.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/tables/19584c31afc609da72416be8d4bc628b53b6627b001d41e2947de5046a73deda.jpg)

![2315773501d25a5d0f9fafcb6aa6611a5e1903a9e797d8242e5adb435e52a054.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/tables/2315773501d25a5d0f9fafcb6aa6611a5e1903a9e797d8242e5adb435e52a054.jpg)

![30f6c262b3528f681e628264a37218b712d266fc1dd300340b5b8e598f11a0a0.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/tables/30f6c262b3528f681e628264a37218b712d266fc1dd300340b5b8e598f11a0a0.jpg)

![3d457496357af17d8aa48f2c858be5fe776e2f8ce5389ec48d4452806266b739.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/tables/3d457496357af17d8aa48f2c858be5fe776e2f8ce5389ec48d4452806266b739.jpg)

![65ed8ac5c6796cab2f443b567f302c2db0a3bd52e6a50028f8f4026177016e09.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/tables/65ed8ac5c6796cab2f443b567f302c2db0a3bd52e6a50028f8f4026177016e09.jpg)

![6fea41f0c240525106309e6b8859131e15c247f5c32d76a5df88cc79e9979d9d.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/tables/6fea41f0c240525106309e6b8859131e15c247f5c32d76a5df88cc79e9979d9d.jpg)

![9444c0328cc030a23b51c3bf38bd2e4c62e70a7550d4e174c2988201a9834b95.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/tables/9444c0328cc030a23b51c3bf38bd2e4c62e70a7550d4e174c2988201a9834b95.jpg)

![9a70a4292f369f6be9c0aabee7a96303ed18ce7426e29f19e1a2aacfc19cfa57.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/tables/9a70a4292f369f6be9c0aabee7a96303ed18ce7426e29f19e1a2aacfc19cfa57.jpg)

![ae8e00c4ca4e0514a934170d43597b46fc6f5385659e9d39021f7bcf2d255395.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/tables/ae8e00c4ca4e0514a934170d43597b46fc6f5385659e9d39021f7bcf2d255395.jpg)

![d5c85f30aaeee69a374be03bfa8da322ae48420cb1ae2c9a9b4816a1d4fb865f.jpg](../iclr_results/2148_VICtoR_ Learning Hierarchical Vision-Instruction Correlation Rewards for Long-horizon Manipulation/tables/d5c85f30aaeee69a374be03bfa8da322ae48420cb1ae2c9a9b4816a1d4fb865f.jpg)

## LeanAgent: Lifelong Learning for Formal Theorem Proving


### Images

![009bf155eaf20889ce5dde1ab310a3cc0f85c0b7d853e2af8a6edff74f6a2223.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/009bf155eaf20889ce5dde1ab310a3cc0f85c0b7d853e2af8a6edff74f6a2223.jpg)

![073b36ba0f20a6618e12ed3a051ecb05b34a85051a2f6c0eae9f14629cb65715.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/073b36ba0f20a6618e12ed3a051ecb05b34a85051a2f6c0eae9f14629cb65715.jpg)

![2e0dfe2db3ec6b302bd752321fe6786ac9311a02674de364debc488d778c1b51.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/2e0dfe2db3ec6b302bd752321fe6786ac9311a02674de364debc488d778c1b51.jpg)

![349fc77b723ddea3275790bfe9448471cd3fb9b3f8156b6d86ac052b372e3dc5.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/349fc77b723ddea3275790bfe9448471cd3fb9b3f8156b6d86ac052b372e3dc5.jpg)

![34a58a7a6dfb28228cae37816bfeb3126dea88b135fb715a0e63ff6697a0857e.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/34a58a7a6dfb28228cae37816bfeb3126dea88b135fb715a0e63ff6697a0857e.jpg)

![386027e73b26361306257b122f782e6431d605f46b7566b85dbc710bb6278273.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/386027e73b26361306257b122f782e6431d605f46b7566b85dbc710bb6278273.jpg)

![395c4454e1c79ccdecfefb2e8577cf5da073767d83febe41b71f23fc5f3aa9fc.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/395c4454e1c79ccdecfefb2e8577cf5da073767d83febe41b71f23fc5f3aa9fc.jpg)

![4058b6131563a3d057c4838c42cf75735d67d8ca1ec750d8f09d2cf889184432.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/4058b6131563a3d057c4838c42cf75735d67d8ca1ec750d8f09d2cf889184432.jpg)

![426189f31f725d977896785762c8d4adb98c956ea218a90c43e4e911e4661669.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/426189f31f725d977896785762c8d4adb98c956ea218a90c43e4e911e4661669.jpg)

![86608db785c31ded26bb8151da088d08343ec94fe335044c79c71d4a5b7a202a.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/86608db785c31ded26bb8151da088d08343ec94fe335044c79c71d4a5b7a202a.jpg)

![8dfd648b76e566edfbb30e2e8008206f1b1d6367cf3e8b1cb5072b075024e255.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/8dfd648b76e566edfbb30e2e8008206f1b1d6367cf3e8b1cb5072b075024e255.jpg)

![90980fa90ce156ba64143ec98f937f2c9041b66535c39013781ce1fe448bd21f.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/90980fa90ce156ba64143ec98f937f2c9041b66535c39013781ce1fe448bd21f.jpg)

![9f4841c7a549163eb851e33baa7ba5e735efdab1c7756a43374beed1ca2d15e9.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/9f4841c7a549163eb851e33baa7ba5e735efdab1c7756a43374beed1ca2d15e9.jpg)

![a4aa5c43b05547ac5664afbf593742bd754c9a3344fd425e06724d2a3a9f427c.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/a4aa5c43b05547ac5664afbf593742bd754c9a3344fd425e06724d2a3a9f427c.jpg)

![ab38a89f9441c5b12ba20220671b0fd0dcf1312da635f90e5e8235ac2506be3d.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/ab38a89f9441c5b12ba20220671b0fd0dcf1312da635f90e5e8235ac2506be3d.jpg)

![b0e76688dd51e5401c6350123f742b530cb6001c2f85ca74757f670546aa3950.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/b0e76688dd51e5401c6350123f742b530cb6001c2f85ca74757f670546aa3950.jpg)

![b642a90224f590a00e768b0a663dd83ba1ddde4a68c1895aa2eee9b080934e91.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/b642a90224f590a00e768b0a663dd83ba1ddde4a68c1895aa2eee9b080934e91.jpg)

![b6d5dd1e49d3c7d4bdfc3d221995c8953fc8205ab66ffeb0bd3724ce21f4ec52.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/b6d5dd1e49d3c7d4bdfc3d221995c8953fc8205ab66ffeb0bd3724ce21f4ec52.jpg)

![b8155a17a28708aba5a70b48d1c82009c98eb53dad1a66028fb5889362c69917.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/b8155a17a28708aba5a70b48d1c82009c98eb53dad1a66028fb5889362c69917.jpg)

![bd44666f84d7b06822f6c19f1f68b8fa9939c7b35fd3839fe6fe9823df64fe62.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/bd44666f84d7b06822f6c19f1f68b8fa9939c7b35fd3839fe6fe9823df64fe62.jpg)

![cb0a6df18a40f33d4b612213a58c04fe49f611965fa24b1e67cf7c6de16594cf.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/cb0a6df18a40f33d4b612213a58c04fe49f611965fa24b1e67cf7c6de16594cf.jpg)

![ccb86d57d335f3d25ebd4b728f200182631993702357176023141486d41ab0e1.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/ccb86d57d335f3d25ebd4b728f200182631993702357176023141486d41ab0e1.jpg)

![cefb7d773bbafee4218721c1b9f797bfe06a6306b1dfcad4ccdf615be542a0e0.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/cefb7d773bbafee4218721c1b9f797bfe06a6306b1dfcad4ccdf615be542a0e0.jpg)

![d4861dbab5ceb1503b3124a210b645921d956ac6bb01fda263830f9762844b32.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/d4861dbab5ceb1503b3124a210b645921d956ac6bb01fda263830f9762844b32.jpg)

![d76008d6e92e20b5b064b24e0301e5b92c7479a0ad22132670c015eb36e98380.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/d76008d6e92e20b5b064b24e0301e5b92c7479a0ad22132670c015eb36e98380.jpg)

![d7c325ebda47b7b65dba4d2aa540781cfcbe778006dc30b3016d0fa32e46de10.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/d7c325ebda47b7b65dba4d2aa540781cfcbe778006dc30b3016d0fa32e46de10.jpg)

![e481b21569ad0591da71aba75081742897fe9b594fa5f6e672c88b959c04e2a5.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/e481b21569ad0591da71aba75081742897fe9b594fa5f6e672c88b959c04e2a5.jpg)

![e6c89c16ffb0be733fddd392433808c7aaf1fa806babfb4a034e2768bf7507b5.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/e6c89c16ffb0be733fddd392433808c7aaf1fa806babfb4a034e2768bf7507b5.jpg)

![f0d40a49987a48cf464f41d3cd163688bed48cbc73fcf869cc030eeb0f6e8512.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/f0d40a49987a48cf464f41d3cd163688bed48cbc73fcf869cc030eeb0f6e8512.jpg)

![f2e0a4c8299e1efd86c20c0595de4237756306a7a6796dcda8f5921bc694fbfa.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/f2e0a4c8299e1efd86c20c0595de4237756306a7a6796dcda8f5921bc694fbfa.jpg)

![f66e4971a4bee00a81d32efad12c938c29795d8af77771445aa09eca721c574a.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/f66e4971a4bee00a81d32efad12c938c29795d8af77771445aa09eca721c574a.jpg)

![fe2184d88aeb647beb1cd8767ecde6efe175404c647d12bed77c1e54c64d1c31.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/images/fe2184d88aeb647beb1cd8767ecde6efe175404c647d12bed77c1e54c64d1c31.jpg)

### Tables

![15b02e25083f649f05016c00d0b05bffbc016420441ff360b765d74ada6b4975.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/tables/15b02e25083f649f05016c00d0b05bffbc016420441ff360b765d74ada6b4975.jpg)

![5fabf150376bba64210fdec5bb1fe3037764209f47519e3bf8d74b5a99dff67b.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/tables/5fabf150376bba64210fdec5bb1fe3037764209f47519e3bf8d74b5a99dff67b.jpg)

![619753c36c5fa2663cf6bf9c4227bbf7eb360827f4f2933156ab0d10a8f04a65.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/tables/619753c36c5fa2663cf6bf9c4227bbf7eb360827f4f2933156ab0d10a8f04a65.jpg)

![79ee41c98d6c94a1b5ef8ee50ed0b2c162bb682636ab339e94f8a8fe1db87cec.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/tables/79ee41c98d6c94a1b5ef8ee50ed0b2c162bb682636ab339e94f8a8fe1db87cec.jpg)

![7f9d4d6f1dcc8f9568b5259d2948e50ce7a019eb858c95c84a20071ccff32788.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/tables/7f9d4d6f1dcc8f9568b5259d2948e50ce7a019eb858c95c84a20071ccff32788.jpg)

![899ac779f8d185f7fdd34bf9210cad6567da078d479b11c95668645cf69de375.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/tables/899ac779f8d185f7fdd34bf9210cad6567da078d479b11c95668645cf69de375.jpg)

![8abef60bfb0a772d7b781c1156d46ba33b3de052c9441b795589f2feabf753d4.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/tables/8abef60bfb0a772d7b781c1156d46ba33b3de052c9441b795589f2feabf753d4.jpg)

![933bd52c665456536095c5cf1cfd015fba5bf23e3cb80d0c6b69e8e862b44c4e.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/tables/933bd52c665456536095c5cf1cfd015fba5bf23e3cb80d0c6b69e8e862b44c4e.jpg)

![bfa85928997a246b3fee639d92e8b2dd320bcc51e6019a72e2ad60eb3465090e.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/tables/bfa85928997a246b3fee639d92e8b2dd320bcc51e6019a72e2ad60eb3465090e.jpg)

![eac9a290a39d81fd166ae0ec2d8e939d78ad88255b48c62843d483ff7f43e025.jpg](../iclr_results/2149_LeanAgent_ Lifelong Learning for Formal Theorem Proving/tables/eac9a290a39d81fd166ae0ec2d8e939d78ad88255b48c62843d483ff7f43e025.jpg)

## BOFormer: Learning to Solve Multi-Objective Bayesian Optimization via Non-Markovian RL


### Images

![23a2312c35fa7e71818c5835a1fdcda4e23ef8defe29f8bfffb0340aabc8c23f.jpg](../iclr_results/2150_BOFormer_ Learning to Solve Multi-Objective Bayesian Optimization via Non-Markovian RL/images/23a2312c35fa7e71818c5835a1fdcda4e23ef8defe29f8bfffb0340aabc8c23f.jpg)

![4683a7e6323446189dab19ff82a0f18e37f05189d41e8f0150f6aef762aae870.jpg](../iclr_results/2150_BOFormer_ Learning to Solve Multi-Objective Bayesian Optimization via Non-Markovian RL/images/4683a7e6323446189dab19ff82a0f18e37f05189d41e8f0150f6aef762aae870.jpg)

![696f85fd9778ea6d8af765c239c8894706e6866c8cc1d2ae4823937e73d3d57e.jpg](../iclr_results/2150_BOFormer_ Learning to Solve Multi-Objective Bayesian Optimization via Non-Markovian RL/images/696f85fd9778ea6d8af765c239c8894706e6866c8cc1d2ae4823937e73d3d57e.jpg)

![6b1d98fc681486a8c5fa5a8cc00be9d602a07a8631d022a7a5d0dc277d419953.jpg](../iclr_results/2150_BOFormer_ Learning to Solve Multi-Objective Bayesian Optimization via Non-Markovian RL/images/6b1d98fc681486a8c5fa5a8cc00be9d602a07a8631d022a7a5d0dc277d419953.jpg)

![6d2813a3f7f3f121b8bbb6b0d945c32795eaf784a39752b61d46840807404df9.jpg](../iclr_results/2150_BOFormer_ Learning to Solve Multi-Objective Bayesian Optimization via Non-Markovian RL/images/6d2813a3f7f3f121b8bbb6b0d945c32795eaf784a39752b61d46840807404df9.jpg)

![6fed688dcca4b9369e75e4b0c5e844f89f3d80b254ca8c1057d3ef1a15be5d96.jpg](../iclr_results/2150_BOFormer_ Learning to Solve Multi-Objective Bayesian Optimization via Non-Markovian RL/images/6fed688dcca4b9369e75e4b0c5e844f89f3d80b254ca8c1057d3ef1a15be5d96.jpg)

![8d1ae05a607ad5bd7be91de32e625fc2994855dbff2749a8d3a70b9f4a2974fb.jpg](../iclr_results/2150_BOFormer_ Learning to Solve Multi-Objective Bayesian Optimization via Non-Markovian RL/images/8d1ae05a607ad5bd7be91de32e625fc2994855dbff2749a8d3a70b9f4a2974fb.jpg)

![9ebc86e3ad2de519c2c1c1dcaef3e954db88b51788be42df4565b1a3c329771c.jpg](../iclr_results/2150_BOFormer_ Learning to Solve Multi-Objective Bayesian Optimization via Non-Markovian RL/images/9ebc86e3ad2de519c2c1c1dcaef3e954db88b51788be42df4565b1a3c329771c.jpg)

![bed9125504448c7591d7b3e597fb2cf8d47db7e27417bcfd64ff0baebdd1649c.jpg](../iclr_results/2150_BOFormer_ Learning to Solve Multi-Objective Bayesian Optimization via Non-Markovian RL/images/bed9125504448c7591d7b3e597fb2cf8d47db7e27417bcfd64ff0baebdd1649c.jpg)

![c592ba080bf704a6bc89880ff63de4e141b9617507481d4a49dc9d24791e31c5.jpg](../iclr_results/2150_BOFormer_ Learning to Solve Multi-Objective Bayesian Optimization via Non-Markovian RL/images/c592ba080bf704a6bc89880ff63de4e141b9617507481d4a49dc9d24791e31c5.jpg)

![c6a15ac746bcdbe6794abd3985824c167ee355ad642a43d5fe01165a0bc69a46.jpg](../iclr_results/2150_BOFormer_ Learning to Solve Multi-Objective Bayesian Optimization via Non-Markovian RL/images/c6a15ac746bcdbe6794abd3985824c167ee355ad642a43d5fe01165a0bc69a46.jpg)

![cbb54dc4d6800645184c2e210d3d92cecc552ad5f9f5279588081d551b662bb3.jpg](../iclr_results/2150_BOFormer_ Learning to Solve Multi-Objective Bayesian Optimization via Non-Markovian RL/images/cbb54dc4d6800645184c2e210d3d92cecc552ad5f9f5279588081d551b662bb3.jpg)

![e3da3ad3a2eb2c5e526ad345fe7b7eeb4d85667995b936442fcdf379517a4e23.jpg](../iclr_results/2150_BOFormer_ Learning to Solve Multi-Objective Bayesian Optimization via Non-Markovian RL/images/e3da3ad3a2eb2c5e526ad345fe7b7eeb4d85667995b936442fcdf379517a4e23.jpg)

### Tables

![16f2e2d193f3b14b49ef0d8fc9a15b1ef1d5faf4285c1d3b899ca778184b5d57.jpg](../iclr_results/2150_BOFormer_ Learning to Solve Multi-Objective Bayesian Optimization via Non-Markovian RL/tables/16f2e2d193f3b14b49ef0d8fc9a15b1ef1d5faf4285c1d3b899ca778184b5d57.jpg)

![e07e842a1726359bc9cc6590c8052d2742eeaab15de05c2ad4ff275b26824c0b.jpg](../iclr_results/2150_BOFormer_ Learning to Solve Multi-Objective Bayesian Optimization via Non-Markovian RL/tables/e07e842a1726359bc9cc6590c8052d2742eeaab15de05c2ad4ff275b26824c0b.jpg)

![fc7ff6214aed5ce77f51766c45150edfb7f78f75679e19a4bc2f7bedafcf516a.jpg](../iclr_results/2150_BOFormer_ Learning to Solve Multi-Objective Bayesian Optimization via Non-Markovian RL/tables/fc7ff6214aed5ce77f51766c45150edfb7f78f75679e19a4bc2f7bedafcf516a.jpg)

## Combining Induction and Transduction for Abstract Reasoning


### Images

![08f705272057448f29b8897f90f3a18312be50daf8984decb5d2394c9659b6ed.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/08f705272057448f29b8897f90f3a18312be50daf8984decb5d2394c9659b6ed.jpg)

![157d290cfde90b57354ac3d91b9d367986521fa55cc9c11521c43ceb1b6ab94e.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/157d290cfde90b57354ac3d91b9d367986521fa55cc9c11521c43ceb1b6ab94e.jpg)

![22479ad9edcd09c5281b4af3024cf1fd45409114d0ac8360b62a4fd974d081eb.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/22479ad9edcd09c5281b4af3024cf1fd45409114d0ac8360b62a4fd974d081eb.jpg)

![288fafeca5a7a90f08876eefacf193d0ec1636c64c6dbce7d1eb342f56cc4161.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/288fafeca5a7a90f08876eefacf193d0ec1636c64c6dbce7d1eb342f56cc4161.jpg)

![28b52e93449dd9cb78f141b56f5a2086fc0ab5d8e7738e120e611dbd2b68824a.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/28b52e93449dd9cb78f141b56f5a2086fc0ab5d8e7738e120e611dbd2b68824a.jpg)

![2d06fb0133832cf54e4873c9ada99021b472c9f6fc4d5345f7d79f008df2b3cb.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/2d06fb0133832cf54e4873c9ada99021b472c9f6fc4d5345f7d79f008df2b3cb.jpg)

![3921329b2f35afe6bd856b4f268b4d8f3a6f32b8d302e8bd83d19a98fb08506d.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/3921329b2f35afe6bd856b4f268b4d8f3a6f32b8d302e8bd83d19a98fb08506d.jpg)

![3959ebe71620d41abcf3f01f92cc6d6ec653fe316f516b418602276f62ddfa23.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/3959ebe71620d41abcf3f01f92cc6d6ec653fe316f516b418602276f62ddfa23.jpg)

![430fe903cc27a7494298d2c513a35ae7427922bc980320e6330247425b4f3f8e.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/430fe903cc27a7494298d2c513a35ae7427922bc980320e6330247425b4f3f8e.jpg)

![4c7bf563f5d4da5d31743848d49dc032ca31d0abc0072089638944dfbafdae6a.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/4c7bf563f5d4da5d31743848d49dc032ca31d0abc0072089638944dfbafdae6a.jpg)

![4d12bf915e1b4a5c710e7a7dcb324a6efa3475c82ba7e0d981efee0f9c7db90a.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/4d12bf915e1b4a5c710e7a7dcb324a6efa3475c82ba7e0d981efee0f9c7db90a.jpg)

![6b675275ef907026dcbca9fe4159a8c9e94db07c4b237f8e214995fec3042be4.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/6b675275ef907026dcbca9fe4159a8c9e94db07c4b237f8e214995fec3042be4.jpg)

![74629cd7ce5b73781fcd2d479e6f53ba7d1ce6d6dc3babecd39f5a1e8539afc7.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/74629cd7ce5b73781fcd2d479e6f53ba7d1ce6d6dc3babecd39f5a1e8539afc7.jpg)

![7c5a43409cc230af014bece8a84729ac092e64d4da538932da231087c4330903.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/7c5a43409cc230af014bece8a84729ac092e64d4da538932da231087c4330903.jpg)

![8d1a50873a81e2a2d979c942bf11070ba27d12c654c0905bfb17c343532e93e6.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/8d1a50873a81e2a2d979c942bf11070ba27d12c654c0905bfb17c343532e93e6.jpg)

![9460db26ef0460d6f8e6a635a2977c37815eec74a748249ff81687c70249f1c4.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/9460db26ef0460d6f8e6a635a2977c37815eec74a748249ff81687c70249f1c4.jpg)

![9621c031ab4b1a80a7d34b361b8b799d5c7a699314b1a88e143e72efdc4aad6c.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/9621c031ab4b1a80a7d34b361b8b799d5c7a699314b1a88e143e72efdc4aad6c.jpg)

![a4585ec98da8b0bed21687fc6311911adb72b5c716afb98187f665b611b25849.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/a4585ec98da8b0bed21687fc6311911adb72b5c716afb98187f665b611b25849.jpg)

![abd489b513863b8d520dfaad57d9c1a446fe662760afabaf1cadeb2cd7b2f796.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/abd489b513863b8d520dfaad57d9c1a446fe662760afabaf1cadeb2cd7b2f796.jpg)

![b8e16e34db3c7d6d65b91247cbfe8a9a6a5d0066382f1e9e5acf05541de38f7f.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/b8e16e34db3c7d6d65b91247cbfe8a9a6a5d0066382f1e9e5acf05541de38f7f.jpg)

![be102c5a58fa4053f1656645e53579ec0889fa90acad4648f0077efae3f02842.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/be102c5a58fa4053f1656645e53579ec0889fa90acad4648f0077efae3f02842.jpg)

![c01619725a435d03d7f7aee1494472aef623117401a09d7bc431bce57f590a5d.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/c01619725a435d03d7f7aee1494472aef623117401a09d7bc431bce57f590a5d.jpg)

![c11de086034f48b5cd44bcdba4b69a7b8e31a4c98cafe9a28ac8d0284b212697.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/c11de086034f48b5cd44bcdba4b69a7b8e31a4c98cafe9a28ac8d0284b212697.jpg)

![cd234575185ae104efb991776cd713671b5a9a4a831f7dab93972de0394bcfce.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/cd234575185ae104efb991776cd713671b5a9a4a831f7dab93972de0394bcfce.jpg)

![ce580be32027abb9462f6cb072700edd165b422eea27d259e9ce7f780d45c4ed.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/ce580be32027abb9462f6cb072700edd165b422eea27d259e9ce7f780d45c4ed.jpg)

![d710130c0ab71e20ee7c0ac0c61470f8a381792b724993fccc082c4c61cceff0.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/d710130c0ab71e20ee7c0ac0c61470f8a381792b724993fccc082c4c61cceff0.jpg)

![ec7d12c2bdf1c8df1b57fbb52bcbb030a4a252b082cfcb55d8a0e1cc470f8afc.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/ec7d12c2bdf1c8df1b57fbb52bcbb030a4a252b082cfcb55d8a0e1cc470f8afc.jpg)

![ee759b8613eceb8a17c8075554439b44fbfe84104b9d6098f2cc037bcea0f294.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/ee759b8613eceb8a17c8075554439b44fbfe84104b9d6098f2cc037bcea0f294.jpg)

![ff4c0a6508203e02194ba34ff0b75c4b8e954a08f66d87d747199a1be58d731d.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/images/ff4c0a6508203e02194ba34ff0b75c4b8e954a08f66d87d747199a1be58d731d.jpg)

### Tables

![3d3a6c0ed80b06f207c55d6c60ed015406feeadc94bdcb05a750bf0279c0cce6.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/tables/3d3a6c0ed80b06f207c55d6c60ed015406feeadc94bdcb05a750bf0279c0cce6.jpg)

![5dab07a5f04a64eddca3fce70d64d3a6083adaec3b6126be20a3d31981196b00.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/tables/5dab07a5f04a64eddca3fce70d64d3a6083adaec3b6126be20a3d31981196b00.jpg)

![6b807f5887b454fd4eae73ac535116a9faaf0c108985b62ac8f564061bf24113.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/tables/6b807f5887b454fd4eae73ac535116a9faaf0c108985b62ac8f564061bf24113.jpg)

![746f3f856dd9881229b5ce1a62ea0d05fc7942ead28547708ff92420880ccbe2.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/tables/746f3f856dd9881229b5ce1a62ea0d05fc7942ead28547708ff92420880ccbe2.jpg)

![b1808561c3c08167fff2cc7997aea26db842fb0d213d41d910b0997c42469fd6.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/tables/b1808561c3c08167fff2cc7997aea26db842fb0d213d41d910b0997c42469fd6.jpg)

![c20856a0be7024d28d728d5e63fcb564e52fa73a026c849ead05f14f001bc577.jpg](../iclr_results/2151_Combining Induction and Transduction for Abstract Reasoning/tables/c20856a0be7024d28d728d5e63fcb564e52fa73a026c849ead05f14f001bc577.jpg)

## Adversarial Training for Defense Against Label Poisoning Attacks


### Images

![0f7e6b0983c77161ea39efa9f428edf194a22475dce2ce28f3a7df7d7bdec5d0.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/images/0f7e6b0983c77161ea39efa9f428edf194a22475dce2ce28f3a7df7d7bdec5d0.jpg)

![2ebb594a918f6aa86f77abbcf84f27d42154163d5828d266db206df3a9afcd6b.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/images/2ebb594a918f6aa86f77abbcf84f27d42154163d5828d266db206df3a9afcd6b.jpg)

![4f6ef3be0812d0a0a292dda47cc9ba6a469766dccd226487d5f4f9ecb31de1c7.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/images/4f6ef3be0812d0a0a292dda47cc9ba6a469766dccd226487d5f4f9ecb31de1c7.jpg)

![65e4c40457b30bf77aec6574bb33ee329fc2da870b479fcee3980b1614f5af08.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/images/65e4c40457b30bf77aec6574bb33ee329fc2da870b479fcee3980b1614f5af08.jpg)

![6c3b2276c6ce1e1f5a3749f27edcbf047018556844f4b67622d0bf6e081b0b9d.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/images/6c3b2276c6ce1e1f5a3749f27edcbf047018556844f4b67622d0bf6e081b0b9d.jpg)

![7aeff9e84726c10e02e7acc980b80c0733c5e35ea29c18c87a0ee030c04eceb1.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/images/7aeff9e84726c10e02e7acc980b80c0733c5e35ea29c18c87a0ee030c04eceb1.jpg)

![7e7ed1cc7f8b2cc1fa2f7ca2e6247d865649a0e0e351a2961516286b89861208.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/images/7e7ed1cc7f8b2cc1fa2f7ca2e6247d865649a0e0e351a2961516286b89861208.jpg)

![93595ad7ebbfb72cb6af6c27fd8a0da9841979469596b9a8a2d912f06c89bbdd.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/images/93595ad7ebbfb72cb6af6c27fd8a0da9841979469596b9a8a2d912f06c89bbdd.jpg)

![96c9083aad3b5cb99d092f7b9b0b4cc4109f76bb6fcbac6f7eed98305150abc3.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/images/96c9083aad3b5cb99d092f7b9b0b4cc4109f76bb6fcbac6f7eed98305150abc3.jpg)

![a2ccb4de16a72f708fcb1fc3a7c9a0ebc44f28617cafd50fa92982b3eb8d5e19.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/images/a2ccb4de16a72f708fcb1fc3a7c9a0ebc44f28617cafd50fa92982b3eb8d5e19.jpg)

![cf3917822828f66c9e9c3c05c460eb6b90cfb5498f2c876b52ff4067596041d5.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/images/cf3917822828f66c9e9c3c05c460eb6b90cfb5498f2c876b52ff4067596041d5.jpg)

![d4e6ed5e20a16e85c92ed17626922313d7559094e7caa2cc82df1753e1fb190f.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/images/d4e6ed5e20a16e85c92ed17626922313d7559094e7caa2cc82df1753e1fb190f.jpg)

![d7d166bed6845bc59e4bce0e4c03e93c3c8f52ba4d28e7a2d34b652cd05f381c.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/images/d7d166bed6845bc59e4bce0e4c03e93c3c8f52ba4d28e7a2d34b652cd05f381c.jpg)

![d8568576f196a71dbb235a6c6203f347ae0ae2a8cb50b7addf3667b19ca2a552.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/images/d8568576f196a71dbb235a6c6203f347ae0ae2a8cb50b7addf3667b19ca2a552.jpg)

![ea2c7f93df294516fd671c4ecf792f532521864c1a3d967bd04ca309a187e8a6.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/images/ea2c7f93df294516fd671c4ecf792f532521864c1a3d967bd04ca309a187e8a6.jpg)

![f18a344a8e09f5b425161c7142bc79265dba5090690fc9a769e69a87731c9b1e.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/images/f18a344a8e09f5b425161c7142bc79265dba5090690fc9a769e69a87731c9b1e.jpg)

![fba7033ed0bf931dbe6bdbec9386530042d322340be6efa444eb655751dca04b.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/images/fba7033ed0bf931dbe6bdbec9386530042d322340be6efa444eb655751dca04b.jpg)

### Tables

![1038d64fd34f9eb60108f294eeef5e105a9217671c1855f368ca7aad9e570d90.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/tables/1038d64fd34f9eb60108f294eeef5e105a9217671c1855f368ca7aad9e570d90.jpg)

![50377c34b6626f046a990e82fc266e3bb656c3ac9f0a3b360dfce83ac8cf42a7.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/tables/50377c34b6626f046a990e82fc266e3bb656c3ac9f0a3b360dfce83ac8cf42a7.jpg)

![7f1010b714371d48e00a582f7f49aa6deef0a12719121ddc9d64773ec67ad005.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/tables/7f1010b714371d48e00a582f7f49aa6deef0a12719121ddc9d64773ec67ad005.jpg)

![80c4fb3e7c9e69dbf338231e08fa6efa6e2b302e53c961c9f2b7747a79efcdd2.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/tables/80c4fb3e7c9e69dbf338231e08fa6efa6e2b302e53c961c9f2b7747a79efcdd2.jpg)

![892822d3b1de9f703ed7db6e85bdd67f33ecee9f11bbaabc3ad1d1ecb7f8eb3f.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/tables/892822d3b1de9f703ed7db6e85bdd67f33ecee9f11bbaabc3ad1d1ecb7f8eb3f.jpg)

![89b1300bdf32fa2b51c06e2e23f4dc333a53938577ca33de453ba979370ab78d.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/tables/89b1300bdf32fa2b51c06e2e23f4dc333a53938577ca33de453ba979370ab78d.jpg)

![8d4bc8851178d26fcd06e411e10a450060c2169999bf061ca04525f3da2f180a.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/tables/8d4bc8851178d26fcd06e411e10a450060c2169999bf061ca04525f3da2f180a.jpg)

![bb44d5e022e06ceb8c2b8b6795be63b4388dda2ff47686bd6238f267c52c076e.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/tables/bb44d5e022e06ceb8c2b8b6795be63b4388dda2ff47686bd6238f267c52c076e.jpg)

![bf796f80f89e6296d247f309dfe7393cef6d29083d2cd84eecd8991d0a804fc4.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/tables/bf796f80f89e6296d247f309dfe7393cef6d29083d2cd84eecd8991d0a804fc4.jpg)

![c021ef27f09b85c525a6585e5044823d17c299433a1f7f29330d34c5e6219988.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/tables/c021ef27f09b85c525a6585e5044823d17c299433a1f7f29330d34c5e6219988.jpg)

![d7a3db20d578ee415c1a29ff6fba764beb473359738a81edcd1961db51fb7524.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/tables/d7a3db20d578ee415c1a29ff6fba764beb473359738a81edcd1961db51fb7524.jpg)

![e16e7056e97883433c0a84ab33864086a55428178063665b5befa54b2e3d224d.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/tables/e16e7056e97883433c0a84ab33864086a55428178063665b5befa54b2e3d224d.jpg)

![ed4a6013af5fbe2477450fe56fea0efe14910c816657c273cb1091a9fdf58f89.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/tables/ed4a6013af5fbe2477450fe56fea0efe14910c816657c273cb1091a9fdf58f89.jpg)

![efcf8516fe721d84a4f72ad09186dd1fa4363864550988c0a7c07de35c24a37a.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/tables/efcf8516fe721d84a4f72ad09186dd1fa4363864550988c0a7c07de35c24a37a.jpg)

![f3ca5901a651994ec75c7e6cb04a7812ec2e1c01e573a02fc6600ef829043ecd.jpg](../iclr_results/2152_Adversarial Training for Defense Against Label Poisoning Attacks/tables/f3ca5901a651994ec75c7e6cb04a7812ec2e1c01e573a02fc6600ef829043ecd.jpg)

## SAM-CP: Marrying SAM with Composable Prompts for Versatile Segmentation


### Images

![141ad6af820e6aefa020c507860dc48f175761467cb5ea74bcd0f836875e9acf.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/images/141ad6af820e6aefa020c507860dc48f175761467cb5ea74bcd0f836875e9acf.jpg)

![1929f8f5adc85f824a1c07cfa678d6b19d0c82f6aeb9730f705789240bc42c09.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/images/1929f8f5adc85f824a1c07cfa678d6b19d0c82f6aeb9730f705789240bc42c09.jpg)

![297f18ef2b8a55a690d32492ffb33580e7956c3f6fc3198a7729f5fc9774de1f.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/images/297f18ef2b8a55a690d32492ffb33580e7956c3f6fc3198a7729f5fc9774de1f.jpg)

![2f3817f9ea767608ad6638bdf07b6eb0ea27be84e973c7ca178143d3934ca827.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/images/2f3817f9ea767608ad6638bdf07b6eb0ea27be84e973c7ca178143d3934ca827.jpg)

![2f95c9f0df82605f8934a751f4cf941e2e756e04cb0acf358e9a79fa2dda9b5e.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/images/2f95c9f0df82605f8934a751f4cf941e2e756e04cb0acf358e9a79fa2dda9b5e.jpg)

![4c4d3c2a018a56cbaa40e7eb8b846fafc40a1be4e9c1f85269e8c542f98bc605.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/images/4c4d3c2a018a56cbaa40e7eb8b846fafc40a1be4e9c1f85269e8c542f98bc605.jpg)

![aaefc2c3e1e460eb6ca4858a8d8dc73627c496c8011809a945189c822500ed4f.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/images/aaefc2c3e1e460eb6ca4858a8d8dc73627c496c8011809a945189c822500ed4f.jpg)

![deac4170b02b23d031800223cb59537272f5c9b380409f1cc89156bf6b00348b.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/images/deac4170b02b23d031800223cb59537272f5c9b380409f1cc89156bf6b00348b.jpg)

![df80ff68f1660df8a7e66c948e186f5b5481fad969b42875c4a97b9d62410122.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/images/df80ff68f1660df8a7e66c948e186f5b5481fad969b42875c4a97b9d62410122.jpg)

![e0e85439fb9b5ffabc41cd9f50b681346d6c169f3cb7b11b6a7459461c77a67c.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/images/e0e85439fb9b5ffabc41cd9f50b681346d6c169f3cb7b11b6a7459461c77a67c.jpg)

![f12471fcd49604b60db5ca50303b74b1bcf2f40f4d0e5c7006e7902a8276c004.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/images/f12471fcd49604b60db5ca50303b74b1bcf2f40f4d0e5c7006e7902a8276c004.jpg)

![f925f50584fd4bf353ce7b1ad528a9f5769578f9418d0e687e996811179505e1.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/images/f925f50584fd4bf353ce7b1ad528a9f5769578f9418d0e687e996811179505e1.jpg)

### Tables

![2d8470d4a642f013f93fb6b4eff9a20b98e59720e74d96dc49d80b91fb0baed3.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/tables/2d8470d4a642f013f93fb6b4eff9a20b98e59720e74d96dc49d80b91fb0baed3.jpg)

![2e26b46dbf2452aac185d7d2ff367242ca9890cf9ed5a5611b4581e63cbb83cb.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/tables/2e26b46dbf2452aac185d7d2ff367242ca9890cf9ed5a5611b4581e63cbb83cb.jpg)

![4339102be1cf3a0b8661824d3ab817b2d9753d61a9978696035884d6ee3290a6.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/tables/4339102be1cf3a0b8661824d3ab817b2d9753d61a9978696035884d6ee3290a6.jpg)

![45774729dce083b8d2ce3fc710dfd68185f731da8639cc3c51f90f3e6e701684.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/tables/45774729dce083b8d2ce3fc710dfd68185f731da8639cc3c51f90f3e6e701684.jpg)

![782d3b5deffd877c29d1219ac671639f9cc66491966e05692867c1c8e4be525c.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/tables/782d3b5deffd877c29d1219ac671639f9cc66491966e05692867c1c8e4be525c.jpg)

![8663976eeeae675865670dbe53513a3a6a8c3843f5db435eadd3497745d60d16.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/tables/8663976eeeae675865670dbe53513a3a6a8c3843f5db435eadd3497745d60d16.jpg)

![9b63ed602458c8462b304e8b3f3199f056f11fadf5af997c1ba4f6e7f340cf55.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/tables/9b63ed602458c8462b304e8b3f3199f056f11fadf5af997c1ba4f6e7f340cf55.jpg)

![c0feb2afc1a38e5c94846c5cef2c052f3ea37688dc352f0913d5c4e4980b53e8.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/tables/c0feb2afc1a38e5c94846c5cef2c052f3ea37688dc352f0913d5c4e4980b53e8.jpg)

![c32d83e03cd01d909fb40db80ee447788fe038795338ffd17a5435bba959cfe6.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/tables/c32d83e03cd01d909fb40db80ee447788fe038795338ffd17a5435bba959cfe6.jpg)

![c9e21070e06175eaf34f458c3a0261737779750bdbd7fc4712e00c08bda10943.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/tables/c9e21070e06175eaf34f458c3a0261737779750bdbd7fc4712e00c08bda10943.jpg)

![d582c9dc2a973217a8f00aedf846f9a5df5bfd10e54db1b662886407547d5f49.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/tables/d582c9dc2a973217a8f00aedf846f9a5df5bfd10e54db1b662886407547d5f49.jpg)

![de3c55469efa4cfa0a4326939ba7619b22070aec2a8b4163d5d63af183109e00.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/tables/de3c55469efa4cfa0a4326939ba7619b22070aec2a8b4163d5d63af183109e00.jpg)

![ef247f7bc8f967f377340bd1d5d0fdcee6e041c58495d0805d80516382093544.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/tables/ef247f7bc8f967f377340bd1d5d0fdcee6e041c58495d0805d80516382093544.jpg)

![f6e11520bfbff7c2f7b1d029a13d3b7c8fb6774805e69571b5dd9180bb17454d.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/tables/f6e11520bfbff7c2f7b1d029a13d3b7c8fb6774805e69571b5dd9180bb17454d.jpg)

![fd233f4c00dd15de428be37f05911ac0108dc5ddcbf67215769cd77b90b6e204.jpg](../iclr_results/2153_SAM-CP_ Marrying SAM with Composable Prompts for Versatile Segmentation/tables/fd233f4c00dd15de428be37f05911ac0108dc5ddcbf67215769cd77b90b6e204.jpg)

## Matryoshka Multimodal Models


### Images

![04a2bc482615922992567207fd2ac195869264cff586b04d33f152c9df829deb.jpg](../iclr_results/2154_Matryoshka Multimodal Models/images/04a2bc482615922992567207fd2ac195869264cff586b04d33f152c9df829deb.jpg)

![178280f51f4b6ca0890d4fdae18124321fa363478ffe351a6bb89a54c4621be4.jpg](../iclr_results/2154_Matryoshka Multimodal Models/images/178280f51f4b6ca0890d4fdae18124321fa363478ffe351a6bb89a54c4621be4.jpg)

![650b4b8d2dc9729e8215b0fe0b1a3fd229732ac018eec68a00a58729ed145763.jpg](../iclr_results/2154_Matryoshka Multimodal Models/images/650b4b8d2dc9729e8215b0fe0b1a3fd229732ac018eec68a00a58729ed145763.jpg)

![802e6fc738c92ce50e00c22bdb1554916552c685ffa9f79b89d8cf9873645114.jpg](../iclr_results/2154_Matryoshka Multimodal Models/images/802e6fc738c92ce50e00c22bdb1554916552c685ffa9f79b89d8cf9873645114.jpg)

![a9ab27c112d1e3ef581dfe2bf19891caafc42cad6a09db2637aacabeca113de4.jpg](../iclr_results/2154_Matryoshka Multimodal Models/images/a9ab27c112d1e3ef581dfe2bf19891caafc42cad6a09db2637aacabeca113de4.jpg)

![b4124709f539b55755aa17033f096a9e5b673723d103055734cbce588a6bfa42.jpg](../iclr_results/2154_Matryoshka Multimodal Models/images/b4124709f539b55755aa17033f096a9e5b673723d103055734cbce588a6bfa42.jpg)

### Tables

![1eeedeaa660aed468ac24d2f73332d4cc55641f6250d4e226367cf304be04be2.jpg](../iclr_results/2154_Matryoshka Multimodal Models/tables/1eeedeaa660aed468ac24d2f73332d4cc55641f6250d4e226367cf304be04be2.jpg)

![1f20f535a5e524dc1fa4d8bd0aa144d65ee855c3a398128b014ed9facf3ceed7.jpg](../iclr_results/2154_Matryoshka Multimodal Models/tables/1f20f535a5e524dc1fa4d8bd0aa144d65ee855c3a398128b014ed9facf3ceed7.jpg)

![2242e5a2169a1746772991fb3e4d81c960597f53fca58feeecfc47ac9ff7c3b2.jpg](../iclr_results/2154_Matryoshka Multimodal Models/tables/2242e5a2169a1746772991fb3e4d81c960597f53fca58feeecfc47ac9ff7c3b2.jpg)

![34011ede6362c5c49ad42b39e71c6683d4f19fa498bcd38a98da9ea8c1b61c0c.jpg](../iclr_results/2154_Matryoshka Multimodal Models/tables/34011ede6362c5c49ad42b39e71c6683d4f19fa498bcd38a98da9ea8c1b61c0c.jpg)

![3b07b786ec87daa5d51a955652473023b34a161d843ae5edd240a8a64a387986.jpg](../iclr_results/2154_Matryoshka Multimodal Models/tables/3b07b786ec87daa5d51a955652473023b34a161d843ae5edd240a8a64a387986.jpg)

![3fdbdf32f9965ffcbde2b0c9dfebd037b0eee263198783479dbcd8c4be7bdbea.jpg](../iclr_results/2154_Matryoshka Multimodal Models/tables/3fdbdf32f9965ffcbde2b0c9dfebd037b0eee263198783479dbcd8c4be7bdbea.jpg)

![77eb35cf97b944000074dfe0cec2b08f2a58c407ecf30dd9176e7d9ee7cb72ca.jpg](../iclr_results/2154_Matryoshka Multimodal Models/tables/77eb35cf97b944000074dfe0cec2b08f2a58c407ecf30dd9176e7d9ee7cb72ca.jpg)

![8d331e21960ea6603c78832b13016dbdc3309e9ad857952620d094bffe2b168c.jpg](../iclr_results/2154_Matryoshka Multimodal Models/tables/8d331e21960ea6603c78832b13016dbdc3309e9ad857952620d094bffe2b168c.jpg)

![8e60381310581cdfaff11232b7d38ba8450f549cb3604f9dec0096e583a6ed0a.jpg](../iclr_results/2154_Matryoshka Multimodal Models/tables/8e60381310581cdfaff11232b7d38ba8450f549cb3604f9dec0096e583a6ed0a.jpg)

![912509c88b5a83b67cdcc253a863e640ab32ded26a1ef5f4f729239abd3463d1.jpg](../iclr_results/2154_Matryoshka Multimodal Models/tables/912509c88b5a83b67cdcc253a863e640ab32ded26a1ef5f4f729239abd3463d1.jpg)

![96b2feebbe467de58a25465e3bc695a8786d8831b31200bd5127ae2a79b5ef73.jpg](../iclr_results/2154_Matryoshka Multimodal Models/tables/96b2feebbe467de58a25465e3bc695a8786d8831b31200bd5127ae2a79b5ef73.jpg)

![a9d1b0f38525536be87aa66ad19a1b5ab86cee1a896b4f87ea4ee7b22e79526a.jpg](../iclr_results/2154_Matryoshka Multimodal Models/tables/a9d1b0f38525536be87aa66ad19a1b5ab86cee1a896b4f87ea4ee7b22e79526a.jpg)

![aa1f63b100640da7db2165d99a4c69ea605955186d699d555a6f4332e377827a.jpg](../iclr_results/2154_Matryoshka Multimodal Models/tables/aa1f63b100640da7db2165d99a4c69ea605955186d699d555a6f4332e377827a.jpg)

![c39790263f2f2e8368203b56249715f423c430b6674ee49297802648a7a90397.jpg](../iclr_results/2154_Matryoshka Multimodal Models/tables/c39790263f2f2e8368203b56249715f423c430b6674ee49297802648a7a90397.jpg)

![d145d6b9abf849c71634b3421b3df275c416af368700e5ab5dcae8e5dbb65c56.jpg](../iclr_results/2154_Matryoshka Multimodal Models/tables/d145d6b9abf849c71634b3421b3df275c416af368700e5ab5dcae8e5dbb65c56.jpg)

![db9d0d2c96153da05a29df808a59ba9fb073c5782742c8996ca074e19206152f.jpg](../iclr_results/2154_Matryoshka Multimodal Models/tables/db9d0d2c96153da05a29df808a59ba9fb073c5782742c8996ca074e19206152f.jpg)

![e8476ae7c352b03db6c742dca7b3b2fc8ef20292be8a2a53eab658e0154a19cf.jpg](../iclr_results/2154_Matryoshka Multimodal Models/tables/e8476ae7c352b03db6c742dca7b3b2fc8ef20292be8a2a53eab658e0154a19cf.jpg)

![ff5c5f553743ba004c1ff0b64cd0056a80172eaa20e7e9bf7d53e1ea97796f3a.jpg](../iclr_results/2154_Matryoshka Multimodal Models/tables/ff5c5f553743ba004c1ff0b64cd0056a80172eaa20e7e9bf7d53e1ea97796f3a.jpg)

## On Rollouts in Model-Based Reinforcement Learning


### Images

![0a3ad16369f7785bde374a95787d0149636eee2425b39ef67130581f03a06959.jpg](../iclr_results/2155_On Rollouts in Model-Based Reinforcement Learning/images/0a3ad16369f7785bde374a95787d0149636eee2425b39ef67130581f03a06959.jpg)

![13ca504042b561f998d8809b77516766163e267b84c67002e3b018949717e01b.jpg](../iclr_results/2155_On Rollouts in Model-Based Reinforcement Learning/images/13ca504042b561f998d8809b77516766163e267b84c67002e3b018949717e01b.jpg)

![3ab264f42571de57ed6d236b36c5a10f63410a7df2251ad309ce6e8e9cf0962e.jpg](../iclr_results/2155_On Rollouts in Model-Based Reinforcement Learning/images/3ab264f42571de57ed6d236b36c5a10f63410a7df2251ad309ce6e8e9cf0962e.jpg)

![4de730b7ad04cf871c1a7cff56419db5989ccae3a57234f9bda7c0c8d223f4ec.jpg](../iclr_results/2155_On Rollouts in Model-Based Reinforcement Learning/images/4de730b7ad04cf871c1a7cff56419db5989ccae3a57234f9bda7c0c8d223f4ec.jpg)

![4e8abe0eb531a80ddc9e4712ef6bed2dc1461f75e154404560ef2a1d4ef9dd9f.jpg](../iclr_results/2155_On Rollouts in Model-Based Reinforcement Learning/images/4e8abe0eb531a80ddc9e4712ef6bed2dc1461f75e154404560ef2a1d4ef9dd9f.jpg)

![6482da9d1e2204848cb87753bb9b447dc9bce652de4e728a063479a003166907.jpg](../iclr_results/2155_On Rollouts in Model-Based Reinforcement Learning/images/6482da9d1e2204848cb87753bb9b447dc9bce652de4e728a063479a003166907.jpg)

![6cd05cfb174438a80417cb020beb0b5ea9edfc207a1697538c686767056efbe6.jpg](../iclr_results/2155_On Rollouts in Model-Based Reinforcement Learning/images/6cd05cfb174438a80417cb020beb0b5ea9edfc207a1697538c686767056efbe6.jpg)

![6f9975b26a96a2f1058420a49a8e4fca288e2f91b4a08b919be1796c741a2d85.jpg](../iclr_results/2155_On Rollouts in Model-Based Reinforcement Learning/images/6f9975b26a96a2f1058420a49a8e4fca288e2f91b4a08b919be1796c741a2d85.jpg)

![7a6b7dbb41be9c013ebb29d4169dea7128bd8f37c6f2f7b927797f167061c593.jpg](../iclr_results/2155_On Rollouts in Model-Based Reinforcement Learning/images/7a6b7dbb41be9c013ebb29d4169dea7128bd8f37c6f2f7b927797f167061c593.jpg)

![9a74cd220be1a2a61d49e72e51683bf6b098e9fddb22766c064c69bf0effd8bd.jpg](../iclr_results/2155_On Rollouts in Model-Based Reinforcement Learning/images/9a74cd220be1a2a61d49e72e51683bf6b098e9fddb22766c064c69bf0effd8bd.jpg)

![e719f7ba9b3d7bafe34a77325f83c99bb4c3eedc410e13f47a8a9861ab136478.jpg](../iclr_results/2155_On Rollouts in Model-Based Reinforcement Learning/images/e719f7ba9b3d7bafe34a77325f83c99bb4c3eedc410e13f47a8a9861ab136478.jpg)

![efe26fe47f74c5f736276abc7b73d3ed8a5e3e9fe65d7b6e2c791db0d58eadc4.jpg](../iclr_results/2155_On Rollouts in Model-Based Reinforcement Learning/images/efe26fe47f74c5f736276abc7b73d3ed8a5e3e9fe65d7b6e2c791db0d58eadc4.jpg)

### Tables

![35d7ed2d6e6ab43569234716a179e22a994a2411436145f6515666bccf4c8431.jpg](../iclr_results/2155_On Rollouts in Model-Based Reinforcement Learning/tables/35d7ed2d6e6ab43569234716a179e22a994a2411436145f6515666bccf4c8431.jpg)

![6a5639dfad586f10942485160c5a1553e2052ca94f1642539c560fc309ba98bf.jpg](../iclr_results/2155_On Rollouts in Model-Based Reinforcement Learning/tables/6a5639dfad586f10942485160c5a1553e2052ca94f1642539c560fc309ba98bf.jpg)

![7024c0e9dec9aa46af2eeb104a45d4b1a1d9c2f1b7a177e82c854b3975ffa32d.jpg](../iclr_results/2155_On Rollouts in Model-Based Reinforcement Learning/tables/7024c0e9dec9aa46af2eeb104a45d4b1a1d9c2f1b7a177e82c854b3975ffa32d.jpg)

![a5619591d21026db8c175c6bf2b25b566d7c69dda9cdfbde639507845070d186.jpg](../iclr_results/2155_On Rollouts in Model-Based Reinforcement Learning/tables/a5619591d21026db8c175c6bf2b25b566d7c69dda9cdfbde639507845070d186.jpg)

![b0c6eec939c20bf5cda73a49e476c053b80a252257a617f768148c1bccfcddc3.jpg](../iclr_results/2155_On Rollouts in Model-Based Reinforcement Learning/tables/b0c6eec939c20bf5cda73a49e476c053b80a252257a617f768148c1bccfcddc3.jpg)

## Information Theoretic Text-to-Image Alignment


### Images

![0810ca22f343df835f39aff4d69b269101356e91f6d370d0af5da9284d147996.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/0810ca22f343df835f39aff4d69b269101356e91f6d370d0af5da9284d147996.jpg)

![124293125f49241c085ad446224e0b6ddb8f45b8bd0c927894cb4e45eaacf4b1.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/124293125f49241c085ad446224e0b6ddb8f45b8bd0c927894cb4e45eaacf4b1.jpg)

![170138b3afe339ba8e9254f2b3d069fcd3e0feae6cd7ec891b42be47770aa1da.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/170138b3afe339ba8e9254f2b3d069fcd3e0feae6cd7ec891b42be47770aa1da.jpg)

![1ac8d416f1ef130fafb6dcb06ae53949ba5b7a1c43a0f6cf0f4538ba56ea5021.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/1ac8d416f1ef130fafb6dcb06ae53949ba5b7a1c43a0f6cf0f4538ba56ea5021.jpg)

![1ba278de89f91bbf7eac34a02f46c77202d28736453486e19b101f5d51d2caf5.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/1ba278de89f91bbf7eac34a02f46c77202d28736453486e19b101f5d51d2caf5.jpg)

![28cf98d768d2273d7932a356348b8b3ad6647cf43df67a345b4f3edbe9a068a1.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/28cf98d768d2273d7932a356348b8b3ad6647cf43df67a345b4f3edbe9a068a1.jpg)

![2ac4909b9f7f702a7a8a9a8bacf8da7120c8f835d2c3d3c831ba0073dc138acd.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/2ac4909b9f7f702a7a8a9a8bacf8da7120c8f835d2c3d3c831ba0073dc138acd.jpg)

![327fe0119ea5c6a7ff406dd0e24a66780683fd2f6da9e0cc89dba717ec69f1a5.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/327fe0119ea5c6a7ff406dd0e24a66780683fd2f6da9e0cc89dba717ec69f1a5.jpg)

![46aff564f3429f7531f6e278e82ed1418a1914aa19f2970703a3b932affbf519.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/46aff564f3429f7531f6e278e82ed1418a1914aa19f2970703a3b932affbf519.jpg)

![4e036441d329d22807782dc6785e7eadbac204838faca480c432e14faeca09b8.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/4e036441d329d22807782dc6785e7eadbac204838faca480c432e14faeca09b8.jpg)

![690893518da977a9721fe124f67ab1301e4a71ceef53a356eb39ab9039849365.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/690893518da977a9721fe124f67ab1301e4a71ceef53a356eb39ab9039849365.jpg)

![6fccc4a7e0f707a3f7baec7c71dd315ab0cf5043ddc8460a85587b0dc0e8cbdf.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/6fccc4a7e0f707a3f7baec7c71dd315ab0cf5043ddc8460a85587b0dc0e8cbdf.jpg)

![731adfb743312c2e3698df65f4725e2c3a89bed5cdc5abe5ebd8707cf705b29f.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/731adfb743312c2e3698df65f4725e2c3a89bed5cdc5abe5ebd8707cf705b29f.jpg)

![85468d67ba5d88996d3d9b9af8cfcfadbdf3cd18cfa4fe367a3ea4caed03bf2c.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/85468d67ba5d88996d3d9b9af8cfcfadbdf3cd18cfa4fe367a3ea4caed03bf2c.jpg)

![912ca28e12742cf3ef14f5c9785c70b0a851258377928fd367bde4b1196b20c1.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/912ca28e12742cf3ef14f5c9785c70b0a851258377928fd367bde4b1196b20c1.jpg)

![987349840db340fb5b00114e844c3c2549af3aabede1b5ebbc7fa3207af00ecc.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/987349840db340fb5b00114e844c3c2549af3aabede1b5ebbc7fa3207af00ecc.jpg)

![aca781f11f42257c08c34c3cb04e7573fbab4c1aea6f4d4956ea46a65596a58f.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/aca781f11f42257c08c34c3cb04e7573fbab4c1aea6f4d4956ea46a65596a58f.jpg)

![b284b8439aa26200a3c1c2b3c842adb74e404e7956fd5ddccc9a89be86cb6d8f.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/b284b8439aa26200a3c1c2b3c842adb74e404e7956fd5ddccc9a89be86cb6d8f.jpg)

![b74c1e88b92fb04d63b2a2195c7307114223a41b4ad44a15ffba7d385320e273.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/b74c1e88b92fb04d63b2a2195c7307114223a41b4ad44a15ffba7d385320e273.jpg)

![bb6768f32e8af4078ecf9942ad4212f64dedc8ded621d13feb90e07bdab56401.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/bb6768f32e8af4078ecf9942ad4212f64dedc8ded621d13feb90e07bdab56401.jpg)

![caa4dafffe023ea491b1e8e01d694f7b6962a636c862948311be8bc75160dc04.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/caa4dafffe023ea491b1e8e01d694f7b6962a636c862948311be8bc75160dc04.jpg)

![ce73a5583244c03065e46525dc554adfbe75e5430dec3a319dd632f6dcdc11f0.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/ce73a5583244c03065e46525dc554adfbe75e5430dec3a319dd632f6dcdc11f0.jpg)

![d32438daff66c32813caaad38397ffa365107ad1eb225b7954ea00179b5c40b5.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/d32438daff66c32813caaad38397ffa365107ad1eb225b7954ea00179b5c40b5.jpg)

![efb0dfa43557bd35b9b9ef3c30fc4e4096f79d42c83a4f36b8e77c2dee1391ea.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/efb0dfa43557bd35b9b9ef3c30fc4e4096f79d42c83a4f36b8e77c2dee1391ea.jpg)

![f07add10800398918e73080bf51cf54bf721011636a3b0523c869eea650ad642.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/f07add10800398918e73080bf51cf54bf721011636a3b0523c869eea650ad642.jpg)

![ff30694527a30c2bd4046ca9ae982ee7bb9a5e3a4d8fad376aa9913833ec48bc.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/images/ff30694527a30c2bd4046ca9ae982ee7bb9a5e3a4d8fad376aa9913833ec48bc.jpg)

### Tables

![1e695a87779b9091e146f49a59e67d9e4aada6ca801bc89a4c18cf46bbf4dcde.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/tables/1e695a87779b9091e146f49a59e67d9e4aada6ca801bc89a4c18cf46bbf4dcde.jpg)

![227fb16680cea9442a4c65899b5d2f1df6f2070cc45ac48d12022796075fc5f8.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/tables/227fb16680cea9442a4c65899b5d2f1df6f2070cc45ac48d12022796075fc5f8.jpg)

![2b87dd191b44187ee335f9dc071cc793303ac1a99ab6b5d79be385b81bc3272e.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/tables/2b87dd191b44187ee335f9dc071cc793303ac1a99ab6b5d79be385b81bc3272e.jpg)

![383d5f0fb8cd1a8b201632abec7e2a327a0c969eec00b58c49fff451060a3627.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/tables/383d5f0fb8cd1a8b201632abec7e2a327a0c969eec00b58c49fff451060a3627.jpg)

![38aad08bbfc350b49b97fc7481be23cc226099d01c324c02a666ce7fe22f5ccd.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/tables/38aad08bbfc350b49b97fc7481be23cc226099d01c324c02a666ce7fe22f5ccd.jpg)

![3e201d4dd39c169db9b9202279819f14e1fab061dcc8f8f1a233bcb775f8dba3.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/tables/3e201d4dd39c169db9b9202279819f14e1fab061dcc8f8f1a233bcb775f8dba3.jpg)

![6edad437aef7fc0345881095afa466a5213a5a8487a1f44fc99e26ac13924244.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/tables/6edad437aef7fc0345881095afa466a5213a5a8487a1f44fc99e26ac13924244.jpg)

![92b5e9e3b739452d5ea63bc203cb3fbbcda18d2633e2c999d450d50b4e4c2465.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/tables/92b5e9e3b739452d5ea63bc203cb3fbbcda18d2633e2c999d450d50b4e4c2465.jpg)

![960972ba1014fe619f36c44201e289d65bb66e5dbaeae48832572c3f6f15f310.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/tables/960972ba1014fe619f36c44201e289d65bb66e5dbaeae48832572c3f6f15f310.jpg)

![9f8df3c20454457bfb01af770b8e9011d08f2ee0c6220121222fe7ac98dc90a8.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/tables/9f8df3c20454457bfb01af770b8e9011d08f2ee0c6220121222fe7ac98dc90a8.jpg)

![a3fe27089375371ee89cdd6d8721705af7d6ee1d54450f2564db277d162b41ad.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/tables/a3fe27089375371ee89cdd6d8721705af7d6ee1d54450f2564db277d162b41ad.jpg)

![bb4c267122d93932c90a8a6bd3b5b26a81390f18f287e741673a4a1c68b2e0d2.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/tables/bb4c267122d93932c90a8a6bd3b5b26a81390f18f287e741673a4a1c68b2e0d2.jpg)

![bdac2cf13083bc98655c70957e0c9c569b512dd01705ac004a3af01e2583902d.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/tables/bdac2cf13083bc98655c70957e0c9c569b512dd01705ac004a3af01e2583902d.jpg)

![ce0439104cf60621fc58ab8a867bfb627857cbf4ba719b4516ae69a2828d54bb.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/tables/ce0439104cf60621fc58ab8a867bfb627857cbf4ba719b4516ae69a2828d54bb.jpg)

![d008bc89031fcb06f1b853a1c3948516884870ccbe9d2b6f1c3a51d305be0ff6.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/tables/d008bc89031fcb06f1b853a1c3948516884870ccbe9d2b6f1c3a51d305be0ff6.jpg)

![fceda13da2de7c6c4b50fdfac19242abbc772a0726c7c7d03a5d4ec05b4e592b.jpg](../iclr_results/2156_Information Theoretic Text-to-Image Alignment/tables/fceda13da2de7c6c4b50fdfac19242abbc772a0726c7c7d03a5d4ec05b4e592b.jpg)

## Uncertainty Herding: One Active Learning Method for All Label Budgets

### Images

![0312f1faef49e72582e5033e38ac15c993a764195488615cfae0f38e0dc86760.jpg](../iclr_results/2157_Uncertainty Herding_ One Active Learning Method for All Label Budgets/images/0312f1faef49e72582e5033e38ac15c993a764195488615cfae0f38e0dc86760.jpg)

![07ea16d35f4a0e9d0b3497fa3e10b960698476287f91ed0809289611f681cf16.jpg](../iclr_results/2157_Uncertainty Herding_ One Active Learning Method for All Label Budgets/images/07ea16d35f4a0e9d0b3497fa3e10b960698476287f91ed0809289611f681cf16.jpg)

![19b9eb018f8e6f52e61ef1d0b8adca1ee700bbebf8c31199a32024ce825fc4b3.jpg](../iclr_results/2157_Uncertainty Herding_ One Active Learning Method for All Label Budgets/images/19b9eb018f8e6f52e61ef1d0b8adca1ee700bbebf8c31199a32024ce825fc4b3.jpg)

![3683975bcdbcb98e8830b8c619235059e22144e1d91bba3080267a1d38bbcc95.jpg](../iclr_results/2157_Uncertainty Herding_ One Active Learning Method for All Label Budgets/images/3683975bcdbcb98e8830b8c619235059e22144e1d91bba3080267a1d38bbcc95.jpg)

![538aa58f57368096c3e6d50d4f579d33d71ad78b0ff17efbeceb8333a44afb24.jpg](../iclr_results/2157_Uncertainty Herding_ One Active Learning Method for All Label Budgets/images/538aa58f57368096c3e6d50d4f579d33d71ad78b0ff17efbeceb8333a44afb24.jpg)

![b19c1521bdd6670f0c75ce65336d78ad46e423771ef4448e4bc09c702391c98a.jpg](../iclr_results/2157_Uncertainty Herding_ One Active Learning Method for All Label Budgets/images/b19c1521bdd6670f0c75ce65336d78ad46e423771ef4448e4bc09c702391c98a.jpg)

![ed6fc70602d4220fa6c84b35482cefea6b48fdee9695ff8bb29e1eba1dd08136.jpg](../iclr_results/2157_Uncertainty Herding_ One Active Learning Method for All Label Budgets/images/ed6fc70602d4220fa6c84b35482cefea6b48fdee9695ff8bb29e1eba1dd08136.jpg)

![f293a0b5cb9962f050f8863b0a4c009356653ea06ebd78a844f176361ea044df.jpg](../iclr_results/2157_Uncertainty Herding_ One Active Learning Method for All Label Budgets/images/f293a0b5cb9962f050f8863b0a4c009356653ea06ebd78a844f176361ea044df.jpg)

### Tables

![cf843aea74e9928822b577b32fdeffe452f43d31d8b901b756cd3d7a3f602e19.jpg](../iclr_results/2157_Uncertainty Herding_ One Active Learning Method for All Label Budgets/tables/cf843aea74e9928822b577b32fdeffe452f43d31d8b901b756cd3d7a3f602e19.jpg)
