# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 6 of 100

## 目录 (Table of Contents)

1. [Amortized Control of Continuous State Space Feynman-Kac Model for Irregular Time Series](#Amortized-Control-of-Continuous-State-Space-Feynman-Kac-Model-for-Irregular-Time-Series)
2. [TetSphere Splatting: Representing High-Quality Geometry with Lagrangian Volumetric Meshes](#TetSphere-Splatting-Representing-High-Quality-Geometry-with-Lagrangian-Volumetric-Meshes)
3. [MoDeGPT: Modular Decomposition for Large Language Model Compression](#MoDeGPT-Modular-Decomposition-for-Large-Language-Model-Compression)
4. [Do Vision-Language Models Represent Space and How? Evaluating Spatial Frame of Reference under Ambiguities](#Do-Vision-Language-Models-Represent-Space-and-How-Evaluating-Spatial-Frame-of-Reference-under-Ambiguities)
5. [Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects](#Geometry-aware-RL-for-Manipulation-of-Varying-Shapes-and-Deformable-Objects)
6. [MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering](#MLE-bench-Evaluating-Machine-Learning-Agents-on-Machine-Learning-Engineering)
7. [Safety Alignment Should be Made More Than Just a Few Tokens Deep](#Safety-Alignment-Should-be-Made-More-Than-Just-a-Few-Tokens-Deep)
8. [Variational Diffusion Posterior Sampling with Midpoint Guidance](#Variational-Diffusion-Posterior-Sampling-with-Midpoint-Guidance)
9. [NeuralPlane: Structured 3D Reconstruction in Planar Primitives with Neural Fields](#NeuralPlane-Structured-3D-Reconstruction-in-Planar-Primitives-with-Neural-Fields)
10. [SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning](#SD-LoRA-Scalable-Decoupled-Low-Rank-Adaptation-for-Class-Incremental-Learning)
11. [Energy-based Backdoor Defense Against Federated Graph Learning](#Energy-based-Backdoor-Defense-Against-Federated-Graph-Learning)
12. [Prioritized Generative Replay](#Prioritized-Generative-Replay)
13. [A Probabilistic Perspective on Unlearning and Alignment for Large Language Models](#A-Probabilistic-Perspective-on-Unlearning-and-Alignment-for-Large-Language-Models)
14. [Exploring The Loss Landscape Of Regularized Neural Networks Via Convex Duality](#Exploring-The-Loss-Landscape-Of-Regularized-Neural-Networks-Via-Convex-Duality)
15. [Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning](#Flat-Reward-in-Policy-Parameter-Space-Implies-Robust-Reinforcement-Learning)
16. [Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning](#Scaling-LLM-Test-Time-Compute-Optimally-Can-be-More-Effective-than-Scaling-Parameters-for-Reasoning)
17. [Compositional Entailment Learning for Hyperbolic Vision-Language Models](#Compositional-Entailment-Learning-for-Hyperbolic-Vision-Language-Models)
18. [OptionZero: Planning with Learned Options](#OptionZero-Planning-with-Learned-Options)
19. [On the Identification of Temporal Causal Representation with Instantaneous Dependence](#On-the-Identification-of-Temporal-Causal-Representation-with-Instantaneous-Dependence)
20. [Towards Understanding Why FixMatch Generalizes Better Than Supervised Learning](#Towards-Understanding-Why-FixMatch-Generalizes-Better-Than-Supervised-Learning)
21. [RMP-SAM: Towards Real-Time Multi-Purpose Segment Anything](#RMP-SAM-Towards-Real-Time-Multi-Purpose-Segment-Anything)
22. [Open-Vocabulary Customization from CLIP via Data-Free Knowledge Distillation](#Open-Vocabulary-Customization-from-CLIP-via-Data-Free-Knowledge-Distillation)
23. [Wide Neural Networks Trained with Weight Decay Provably Exhibit Neural Collapse](#Wide-Neural-Networks-Trained-with-Weight-Decay-Provably-Exhibit-Neural-Collapse)
24. [TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis](#TimeMixer-A-General-Time-Series-Pattern-Machine-for-Universal-Predictive-Analysis)
25. [ProtComposer: Compositional Protein Structure Generation with 3D Ellipsoids](#ProtComposer-Compositional-Protein-Structure-Generation-with-3D-Ellipsoids)
26. [Synthetic continued pretraining](#Synthetic-continued-pretraining)
27. [ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability](#ReDeEP-Detecting-Hallucination-in-Retrieval-Augmented-Generation-via-Mechanistic-Interpretability)
28. [LOKI: A Comprehensive Synthetic Data Detection Benchmark using Large Multimodal Models](#LOKI-A-Comprehensive-Synthetic-Data-Detection-Benchmark-using-Large-Multimodal-Models)
29. [Revisiting Zeroth-Order Optimization:  Minimum-Variance Two-Point Estimators and  Directionally Aligned Perturbations](#Revisiting-Zeroth-Order-Optimization-Minimum-Variance-Two-Point-Estimators-and-Directionally-Aligned-Perturbations)
30. [BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval](#BRIGHT-A-Realistic-and-Challenging-Benchmark-for-Reasoning-Intensive-Retrieval)
31. [Adversarial Perturbations Cannot Reliably Protect Artists From Generative AI](#Adversarial-Perturbations-Cannot-Reliably-Protect-Artists-From-Generative-AI)
32. [MaRS: A Fast Sampler for Mean Reverting Diffusion based on ODE and SDE Solvers](#MaRS-A-Fast-Sampler-for-Mean-Reverting-Diffusion-based-on-ODE-and-SDE-Solvers)
33. [Robust Function-Calling for On-Device Language Model via Function Masking](#Robust-Function-Calling-for-On-Device-Language-Model-via-Function-Masking)
34. [GOLD: Graph Out-of-Distribution Detection via Implicit Adversarial Latent Generation](#GOLD-Graph-Out-of-Distribution-Detection-via-Implicit-Adversarial-Latent-Generation)
35. [Advantage-Guided Distillation for Preference Alignment in Small Language Models](#Advantage-Guided-Distillation-for-Preference-Alignment-in-Small-Language-Models)
36. [JudgeLM: Fine-tuned Large Language Models are Scalable Judges](#JudgeLM-Fine-tuned-Large-Language-Models-are-Scalable-Judges)
37. [Stem-OB: Generalizable Visual Imitation Learning with Stem-Like Convergent Observation through Diffusion Inversion](#Stem-OB-Generalizable-Visual-Imitation-Learning-with-Stem-Like-Convergent-Observation-through-Diffusion-Inversion)

---


## Amortized Control of Continuous State Space Feynman-Kac Model for Irregular Time Series

### Images

![4e5a38f27eb21622427a720a76e7a0d41c33931fd136356276a8dcc1980b07ef.jpg](../iclr_results/187_The%20Complexity%20of%20Two-Team%20Polymatrix%20Games%20with%20Independent%20Adversaries/images/4e5a38f27eb21622427a720a76e7a0d41c33931fd136356276a8dcc1980b07ef.jpg)

![824a8e74da5858b8c0843917e6637a658af0fefe5ed636c799042c4ada79232f.jpg](../iclr_results/187_The%20Complexity%20of%20Two-Team%20Polymatrix%20Games%20with%20Independent%20Adversaries/images/824a8e74da5858b8c0843917e6637a658af0fefe5ed636c799042c4ada79232f.jpg)

## Amortized Control of Continuous State Space Feynman-Kac Model for Irregular Time Series


### Images

![305499386356c09e7aac1d8590d7794c5ca17a2fdd60576bc2c9f4c623068460.jpg](../iclr_results/188_Amortized%20Control%20of%20Continuous%20State%20Space%20Feynman-Kac%20Model%20for%20Irregular%20Time%20Series/images/305499386356c09e7aac1d8590d7794c5ca17a2fdd60576bc2c9f4c623068460.jpg)

![3efaf89ae28ae52154c1e5148ab1a0809e39b28218da0005035caaec6de0f626.jpg](../iclr_results/188_Amortized%20Control%20of%20Continuous%20State%20Space%20Feynman-Kac%20Model%20for%20Irregular%20Time%20Series/images/3efaf89ae28ae52154c1e5148ab1a0809e39b28218da0005035caaec6de0f626.jpg)

![a1b7ed94d61d8252cc21d05b51ff453c59ced5a23dae71e061d56c7ac0cd3bfb.jpg](../iclr_results/188_Amortized%20Control%20of%20Continuous%20State%20Space%20Feynman-Kac%20Model%20for%20Irregular%20Time%20Series/images/a1b7ed94d61d8252cc21d05b51ff453c59ced5a23dae71e061d56c7ac0cd3bfb.jpg)

![ad67aa47fdd49e2e5b5e305aaa4f0e562f94c134465d24ffaad15f3a15bd4337.jpg](../iclr_results/188_Amortized%20Control%20of%20Continuous%20State%20Space%20Feynman-Kac%20Model%20for%20Irregular%20Time%20Series/images/ad67aa47fdd49e2e5b5e305aaa4f0e562f94c134465d24ffaad15f3a15bd4337.jpg)

### Tables

![585c038be1c00983ef43d661013c5073be9ec02e5372c6027ad7fdb7acd01a57.jpg](../iclr_results/188_Amortized%20Control%20of%20Continuous%20State%20Space%20Feynman-Kac%20Model%20for%20Irregular%20Time%20Series/tables/585c038be1c00983ef43d661013c5073be9ec02e5372c6027ad7fdb7acd01a57.jpg)

![d7223782aaeee4d4f6c8c5d1845d31eda1f31c0d85deb83faa063ae61e9f1acb.jpg](../iclr_results/188_Amortized%20Control%20of%20Continuous%20State%20Space%20Feynman-Kac%20Model%20for%20Irregular%20Time%20Series/tables/d7223782aaeee4d4f6c8c5d1845d31eda1f31c0d85deb83faa063ae61e9f1acb.jpg)

![db5715ce633fb82e65a748cc39b8c721ded05ee26bc031b35415813d81d4fb6f.jpg](../iclr_results/188_Amortized%20Control%20of%20Continuous%20State%20Space%20Feynman-Kac%20Model%20for%20Irregular%20Time%20Series/tables/db5715ce633fb82e65a748cc39b8c721ded05ee26bc031b35415813d81d4fb6f.jpg)

![dfdc92bdfd3799e2317ca6428f42324cd7584acba0cc8bc344e405d60f3eb4e4.jpg](../iclr_results/188_Amortized%20Control%20of%20Continuous%20State%20Space%20Feynman-Kac%20Model%20for%20Irregular%20Time%20Series/tables/dfdc92bdfd3799e2317ca6428f42324cd7584acba0cc8bc344e405d60f3eb4e4.jpg)

## TetSphere Splatting: Representing High-Quality Geometry with Lagrangian Volumetric Meshes


### Images

![0c46bd8bb551bdf6e7aec7ac1eaebe63ef67b96faec42670fab6797fa192a2fb.jpg](../iclr_results/189_TetSphere%20Splatting_%20Representing%20High-Quality%20Geometry%20with%20Lagrangian%20Volumetric%20Meshes/images/0c46bd8bb551bdf6e7aec7ac1eaebe63ef67b96faec42670fab6797fa192a2fb.jpg)

![361f0e55ca44bc12d1b558d986f511d28e9d36cf5a8d680f0ee401efe3841f93.jpg](../iclr_results/189_TetSphere%20Splatting_%20Representing%20High-Quality%20Geometry%20with%20Lagrangian%20Volumetric%20Meshes/images/361f0e55ca44bc12d1b558d986f511d28e9d36cf5a8d680f0ee401efe3841f93.jpg)

![3b0d227b44c71730f5baa173ce80b6c5097845945cd8b3635a3d47ba27a7845c.jpg](../iclr_results/189_TetSphere%20Splatting_%20Representing%20High-Quality%20Geometry%20with%20Lagrangian%20Volumetric%20Meshes/images/3b0d227b44c71730f5baa173ce80b6c5097845945cd8b3635a3d47ba27a7845c.jpg)

![3b389e13787fa59e76cad6f882d7b1253fff2f909fea41ef2815913a9325bc2f.jpg](../iclr_results/189_TetSphere%20Splatting_%20Representing%20High-Quality%20Geometry%20with%20Lagrangian%20Volumetric%20Meshes/images/3b389e13787fa59e76cad6f882d7b1253fff2f909fea41ef2815913a9325bc2f.jpg)

![5337914644000c17cd8fbec6523d8fa31439c416a199d0b1448095911ff63795.jpg](../iclr_results/189_TetSphere%20Splatting_%20Representing%20High-Quality%20Geometry%20with%20Lagrangian%20Volumetric%20Meshes/images/5337914644000c17cd8fbec6523d8fa31439c416a199d0b1448095911ff63795.jpg)

![6b32206deaacfe3d59d262d13f9fc88c3748487eb431a521d2399726295a8097.jpg](../iclr_results/189_TetSphere%20Splatting_%20Representing%20High-Quality%20Geometry%20with%20Lagrangian%20Volumetric%20Meshes/images/6b32206deaacfe3d59d262d13f9fc88c3748487eb431a521d2399726295a8097.jpg)

![8c8405ea858121eac407f21d84b7d574fc378944b330d053671b4490d787c586.jpg](../iclr_results/189_TetSphere%20Splatting_%20Representing%20High-Quality%20Geometry%20with%20Lagrangian%20Volumetric%20Meshes/images/8c8405ea858121eac407f21d84b7d574fc378944b330d053671b4490d787c586.jpg)

![94bb575b0e4d5520ff5bf2eb4174b7e8157573f720c9281a7a8312f99e9b9b6a.jpg](../iclr_results/189_TetSphere%20Splatting_%20Representing%20High-Quality%20Geometry%20with%20Lagrangian%20Volumetric%20Meshes/images/94bb575b0e4d5520ff5bf2eb4174b7e8157573f720c9281a7a8312f99e9b9b6a.jpg)

![99bb3192a2db7ddfad6cd611f20f4be30a8e810f74ed72660c4bd06d5e8256e7.jpg](../iclr_results/189_TetSphere%20Splatting_%20Representing%20High-Quality%20Geometry%20with%20Lagrangian%20Volumetric%20Meshes/images/99bb3192a2db7ddfad6cd611f20f4be30a8e810f74ed72660c4bd06d5e8256e7.jpg)

![a31c6595a1aee4a8b7f5880c866e75e3f34c60e18612411ea09de44c26105793.jpg](../iclr_results/189_TetSphere%20Splatting_%20Representing%20High-Quality%20Geometry%20with%20Lagrangian%20Volumetric%20Meshes/images/a31c6595a1aee4a8b7f5880c866e75e3f34c60e18612411ea09de44c26105793.jpg)

![b15d1f6f7cbe842740ffe14bbaab483969c47862f2e94e6d1ecc7c658451b5ef.jpg](../iclr_results/189_TetSphere%20Splatting_%20Representing%20High-Quality%20Geometry%20with%20Lagrangian%20Volumetric%20Meshes/images/b15d1f6f7cbe842740ffe14bbaab483969c47862f2e94e6d1ecc7c658451b5ef.jpg)

![b8b27393769e1e8093a79987a9e5bdafc6979f9b628542e601cca3aef4a218a5.jpg](../iclr_results/189_TetSphere%20Splatting_%20Representing%20High-Quality%20Geometry%20with%20Lagrangian%20Volumetric%20Meshes/images/b8b27393769e1e8093a79987a9e5bdafc6979f9b628542e601cca3aef4a218a5.jpg)

![d3bb3268f60d99c3a78140e3baa3a086530fc06625067c8246492f81d2282f15.jpg](../iclr_results/189_TetSphere%20Splatting_%20Representing%20High-Quality%20Geometry%20with%20Lagrangian%20Volumetric%20Meshes/images/d3bb3268f60d99c3a78140e3baa3a086530fc06625067c8246492f81d2282f15.jpg)

![fb80dfcf6cca1cffde089fe1a74f61cf49a73d85b227dc3cfdeb21ffa25027ec.jpg](../iclr_results/189_TetSphere%20Splatting_%20Representing%20High-Quality%20Geometry%20with%20Lagrangian%20Volumetric%20Meshes/images/fb80dfcf6cca1cffde089fe1a74f61cf49a73d85b227dc3cfdeb21ffa25027ec.jpg)

### Tables

![057db8024251cfeb446c4a74a70fe3f4f7d014bd1b838ab8cef003a970933a86.jpg](../iclr_results/189_TetSphere%20Splatting_%20Representing%20High-Quality%20Geometry%20with%20Lagrangian%20Volumetric%20Meshes/tables/057db8024251cfeb446c4a74a70fe3f4f7d014bd1b838ab8cef003a970933a86.jpg)

![3abd11104c33928ed0c782e76232905d183c40e46cf0696ce72afb13c2711008.jpg](../iclr_results/189_TetSphere%20Splatting_%20Representing%20High-Quality%20Geometry%20with%20Lagrangian%20Volumetric%20Meshes/tables/3abd11104c33928ed0c782e76232905d183c40e46cf0696ce72afb13c2711008.jpg)

![d0568ca564ad55a87d99e597253a68d8a0470d5e0d5d8864a73b53c813cd8b8d.jpg](../iclr_results/189_TetSphere%20Splatting_%20Representing%20High-Quality%20Geometry%20with%20Lagrangian%20Volumetric%20Meshes/tables/d0568ca564ad55a87d99e597253a68d8a0470d5e0d5d8864a73b53c813cd8b8d.jpg)

![d3fab46e8ddeb2d28a1299b090c6a77dea9ee9c41853f07a0af8062cc20c922c.jpg](../iclr_results/189_TetSphere%20Splatting_%20Representing%20High-Quality%20Geometry%20with%20Lagrangian%20Volumetric%20Meshes/tables/d3fab46e8ddeb2d28a1299b090c6a77dea9ee9c41853f07a0af8062cc20c922c.jpg)

![f161eaedb86a73195147f8dbd85635f46016619f45b78d45f3caee2dcdb8c1eb.jpg](../iclr_results/189_TetSphere%20Splatting_%20Representing%20High-Quality%20Geometry%20with%20Lagrangian%20Volumetric%20Meshes/tables/f161eaedb86a73195147f8dbd85635f46016619f45b78d45f3caee2dcdb8c1eb.jpg)

## MoDeGPT: Modular Decomposition for Large Language Model Compression


### Images

![076f948d38db0d957215f779b8b1de65709eee1daf2e7363bf16c8ae31c4fcdd.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/images/076f948d38db0d957215f779b8b1de65709eee1daf2e7363bf16c8ae31c4fcdd.jpg)

![339640512176ba5f7f3ad84c0ae5a27300224c99b385eb08e054918db7ed11eb.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/images/339640512176ba5f7f3ad84c0ae5a27300224c99b385eb08e054918db7ed11eb.jpg)

![3af5a02c7330240f92000e5f8e98e6355daca7ab7a0332ecef83b316ef4a79d6.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/images/3af5a02c7330240f92000e5f8e98e6355daca7ab7a0332ecef83b316ef4a79d6.jpg)

![480e205eede646d1adfe3a93b5ead231fb3904362e2cc9587e76793b839dfa93.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/images/480e205eede646d1adfe3a93b5ead231fb3904362e2cc9587e76793b839dfa93.jpg)

![7f1089bc60c5ef6eaa93c8cec4b0252403584c2e0254dd2300048178ad6d5680.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/images/7f1089bc60c5ef6eaa93c8cec4b0252403584c2e0254dd2300048178ad6d5680.jpg)

![8b8237af3e44ba4d2293afacccee7922c91d9b5c64d23f0c8b4663f0183bb850.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/images/8b8237af3e44ba4d2293afacccee7922c91d9b5c64d23f0c8b4663f0183bb850.jpg)

![973a3187a4c0360f87e8ca4241de0be67fb99412cab5949385d91d6abc6b1a8c.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/images/973a3187a4c0360f87e8ca4241de0be67fb99412cab5949385d91d6abc6b1a8c.jpg)

![9c8e93bbb2eed96ab1d79c34829e84b34cb3ba9257aee15c92a93cc1733b102f.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/images/9c8e93bbb2eed96ab1d79c34829e84b34cb3ba9257aee15c92a93cc1733b102f.jpg)

![ac1e95f6dc6c7bce353b0a1aa2d067be7a146a593c54f680b50eed2e32fa5761.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/images/ac1e95f6dc6c7bce353b0a1aa2d067be7a146a593c54f680b50eed2e32fa5761.jpg)

![b26c6c609decc2547cd15dafdbf52d4dc3cb6b04c3522bb5c4cf0566feca505e.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/images/b26c6c609decc2547cd15dafdbf52d4dc3cb6b04c3522bb5c4cf0566feca505e.jpg)

![b872fba21c591bbf16d46524372e9f80434366a03c7e387f1813200c68250e92.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/images/b872fba21c591bbf16d46524372e9f80434366a03c7e387f1813200c68250e92.jpg)

![f745e626eaa254163be4788a6b4a30f20d75f2ae7e9065d81121658d5cd279ad.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/images/f745e626eaa254163be4788a6b4a30f20d75f2ae7e9065d81121658d5cd279ad.jpg)

### Tables

![0365d9d5a553040d6fce98c7dc033d731cffb79e2d549102b7df91b1a0769228.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/0365d9d5a553040d6fce98c7dc033d731cffb79e2d549102b7df91b1a0769228.jpg)

![0453356d899e1f828273719b08f71a01bf25d76957add3912c3bae1cf0d74679.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/0453356d899e1f828273719b08f71a01bf25d76957add3912c3bae1cf0d74679.jpg)

![1233870dbd21499f38aee54c686d05ff8a576d9500bd724b19f0a97bff0d7d84.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/1233870dbd21499f38aee54c686d05ff8a576d9500bd724b19f0a97bff0d7d84.jpg)

![15ba1b05461f7738a1973cfd2005e643eff040e9bf860e8c1d4b1752c4017036.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/15ba1b05461f7738a1973cfd2005e643eff040e9bf860e8c1d4b1752c4017036.jpg)

![241d2efb2fc361b470f67da6e94eabd5fe10bc7b40910671f99901a6ad81d9fc.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/241d2efb2fc361b470f67da6e94eabd5fe10bc7b40910671f99901a6ad81d9fc.jpg)

![24f459a20ae98329a4786cfcfbbde6715bac7ff405c25a692ea15c7a7799af45.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/24f459a20ae98329a4786cfcfbbde6715bac7ff405c25a692ea15c7a7799af45.jpg)

![25f41f2558ed9909ec8348891c8fb0de461d4c6da384f07022c1ae47f80d6089.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/25f41f2558ed9909ec8348891c8fb0de461d4c6da384f07022c1ae47f80d6089.jpg)

![26ddbb3ac252b4793200e9e73241960d4501646b72e60ec34d2ad769cb5d41be.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/26ddbb3ac252b4793200e9e73241960d4501646b72e60ec34d2ad769cb5d41be.jpg)

![28323f1316002c51906ef593066fc6df7847665d51478979f1b065dfdf621882.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/28323f1316002c51906ef593066fc6df7847665d51478979f1b065dfdf621882.jpg)

![2873ccf6bbebee5a3605a23c38dc64d2808e5d34f8f3041874c321536edce71c.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/2873ccf6bbebee5a3605a23c38dc64d2808e5d34f8f3041874c321536edce71c.jpg)

![341b4f0860b8fd8ae2c5ae1e08d82f0731edeb452d7775e4b0a7e21aed30641b.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/341b4f0860b8fd8ae2c5ae1e08d82f0731edeb452d7775e4b0a7e21aed30641b.jpg)

![3a1d5276936524e1ab85411dba18061286261104e8be5b70fa124079f13985f3.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/3a1d5276936524e1ab85411dba18061286261104e8be5b70fa124079f13985f3.jpg)

![447a0ba02a178384d3c5524b908969f52e0e75eb3b51d8b14c8d7fdd54e668b1.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/447a0ba02a178384d3c5524b908969f52e0e75eb3b51d8b14c8d7fdd54e668b1.jpg)

![44aadb6556563e6cd50fec24d00aa7c1e00f3b571b197c5b052eb58d4e385b5f.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/44aadb6556563e6cd50fec24d00aa7c1e00f3b571b197c5b052eb58d4e385b5f.jpg)

![49c8a2194e5c87c901c385e0b20e1407755e3ce98e278777f22c6e4aa57a1dcf.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/49c8a2194e5c87c901c385e0b20e1407755e3ce98e278777f22c6e4aa57a1dcf.jpg)

![4c3f202a53cbc9f1f2eb5c738f6527a877085115f5accf2144b0ce263ae680ef.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/4c3f202a53cbc9f1f2eb5c738f6527a877085115f5accf2144b0ce263ae680ef.jpg)

![5bfc83b7197e17378ab8e8b5a5a974d1f0a4244a2c11bef4d65bed2dde95393b.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/5bfc83b7197e17378ab8e8b5a5a974d1f0a4244a2c11bef4d65bed2dde95393b.jpg)

![60362d3ce5b7f0a1f7f533dd0fb19e49809970e29f1ca148a029d8a1c4023f22.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/60362d3ce5b7f0a1f7f533dd0fb19e49809970e29f1ca148a029d8a1c4023f22.jpg)

![68998b02dfa4f72ccdf73643e76944b13803e72043b306055b1958bb730a2c8e.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/68998b02dfa4f72ccdf73643e76944b13803e72043b306055b1958bb730a2c8e.jpg)

![72814d44c6de56ea3ec69d0492efba04fda90bc9ac57ba4c1fb85172064104fb.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/72814d44c6de56ea3ec69d0492efba04fda90bc9ac57ba4c1fb85172064104fb.jpg)

![72a41da4fb6125db14cf0048461fb5f2020ba5d213d66c2c61dcf08414fbfc7f.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/72a41da4fb6125db14cf0048461fb5f2020ba5d213d66c2c61dcf08414fbfc7f.jpg)

![87734e495d8f2229707c9de56a67fb1b4daceaf703975daf5731e3b7735d1822.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/87734e495d8f2229707c9de56a67fb1b4daceaf703975daf5731e3b7735d1822.jpg)

![87efe5778f0f27e21f7ce6c5b9f293275d7cd4ab4e8f5c176f8094d5b1cf23b6.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/87efe5778f0f27e21f7ce6c5b9f293275d7cd4ab4e8f5c176f8094d5b1cf23b6.jpg)

![88d17a871cb46a27374fd9b54346372abe5b3a0c6a575588d36612f9ba3210ef.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/88d17a871cb46a27374fd9b54346372abe5b3a0c6a575588d36612f9ba3210ef.jpg)

![8f0e2326dc25d7a52eef29a30c7ef1b53c2b98711bfd8d8edec5407307fe7026.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/8f0e2326dc25d7a52eef29a30c7ef1b53c2b98711bfd8d8edec5407307fe7026.jpg)

![932d92a6725f5f5ef1f41fb28c2edfa67a0baf6a6a1df27c5156c578b0137f40.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/932d92a6725f5f5ef1f41fb28c2edfa67a0baf6a6a1df27c5156c578b0137f40.jpg)

![9c1d96a1dfde19ac0ac9ef1bed5d2060f4b2d48ddc42ac970843738ce95f3f8b.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/9c1d96a1dfde19ac0ac9ef1bed5d2060f4b2d48ddc42ac970843738ce95f3f8b.jpg)

![a87ad72f29bb372d22d749893f260ef301e34857d5ca4635f0e992bb779e6758.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/a87ad72f29bb372d22d749893f260ef301e34857d5ca4635f0e992bb779e6758.jpg)

![a95f302608e61a1df50e74a525645fcf9c8471c460f1a8a4eaad009bd9d707ca.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/a95f302608e61a1df50e74a525645fcf9c8471c460f1a8a4eaad009bd9d707ca.jpg)

![aa7a650afd55095c857eedecf9e2469ef276ac4bbca6b4159a7b51405666febe.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/aa7a650afd55095c857eedecf9e2469ef276ac4bbca6b4159a7b51405666febe.jpg)

![c23f4c135510b79b8fd82e0ae2cafc03fc8896b1c1d2dc723be2c25605d89fdb.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/c23f4c135510b79b8fd82e0ae2cafc03fc8896b1c1d2dc723be2c25605d89fdb.jpg)

![c3c0eec523921a6d639fd961a26b35b4f0219a5dceefc03321566c15a410818f.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/c3c0eec523921a6d639fd961a26b35b4f0219a5dceefc03321566c15a410818f.jpg)

![c7cb359d84a307cdfdcde27f0c1c72d82c8826743beb8c1b4d0fe0f072e9a8ea.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/c7cb359d84a307cdfdcde27f0c1c72d82c8826743beb8c1b4d0fe0f072e9a8ea.jpg)

![e0047ec6c29802bfaac86b0cf62089441f34cb1aebbadf0da27bccb80520cfae.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/e0047ec6c29802bfaac86b0cf62089441f34cb1aebbadf0da27bccb80520cfae.jpg)

![e24bd8985ba0741aea47320111a3f4c521d1cb6006abfb49ac051f4071540d86.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/e24bd8985ba0741aea47320111a3f4c521d1cb6006abfb49ac051f4071540d86.jpg)

![ee51b181db60dfb32c942966239c882cf1631c4d2e8d0f2503dd9d2778b4d043.jpg](../iclr_results/190_MoDeGPT_%20Modular%20Decomposition%20for%20Large%20Language%20Model%20Compression/tables/ee51b181db60dfb32c942966239c882cf1631c4d2e8d0f2503dd9d2778b4d043.jpg)

## Do Vision-Language Models Represent Space and How? Evaluating Spatial Frame of Reference under Ambiguities


### Images

![035d8e3d9c8056541f874bd7716bf716ca7c9822a2ac44116a37c02c012a8f99.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/images/035d8e3d9c8056541f874bd7716bf716ca7c9822a2ac44116a37c02c012a8f99.jpg)

![21d59e1e9f5dc9ef06ecd1833fa98de06cae2281ac03a73941e4360bae0e9049.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/images/21d59e1e9f5dc9ef06ecd1833fa98de06cae2281ac03a73941e4360bae0e9049.jpg)

![39ecf7b5dae1b1a958a62452c1e7dac24893cefc11090396abf65fbeff3e9698.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/images/39ecf7b5dae1b1a958a62452c1e7dac24893cefc11090396abf65fbeff3e9698.jpg)

![3be9036e196433d9c41c19e34078c90472d97ecff38fefad7337d8fdc5ec7c4a.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/images/3be9036e196433d9c41c19e34078c90472d97ecff38fefad7337d8fdc5ec7c4a.jpg)

![3ca769c85c466ff436a65f642e5460fd7d02fb612df8917a74e4404efac894f7.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/images/3ca769c85c466ff436a65f642e5460fd7d02fb612df8917a74e4404efac894f7.jpg)

![503f5bba141b6823b6f93691d7c3e7004f106280f3bdbf117246ea48a00a9df8.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/images/503f5bba141b6823b6f93691d7c3e7004f106280f3bdbf117246ea48a00a9df8.jpg)

![52be6e36e47705bf7213658c2013afe3e62551b4f3e72ff725b4b0dcc2d63621.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/images/52be6e36e47705bf7213658c2013afe3e62551b4f3e72ff725b4b0dcc2d63621.jpg)

![53d2ba6fb6e9e294f4f0440a090c0c047e5bea16134918f0bb79f3d6042d0d8b.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/images/53d2ba6fb6e9e294f4f0440a090c0c047e5bea16134918f0bb79f3d6042d0d8b.jpg)

![8aaa0b1342a4950e9f037511e3a11a12648dd6c31ecb16caec01c0b9dea1e6fc.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/images/8aaa0b1342a4950e9f037511e3a11a12648dd6c31ecb16caec01c0b9dea1e6fc.jpg)

![8acd50b066818472d665307fc684e5183b8459355d482f89918105dae9454586.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/images/8acd50b066818472d665307fc684e5183b8459355d482f89918105dae9454586.jpg)

![8cfbaf9564ce629fde77ec5a6aed66a72ecb793e6ba45ab2cf966ca043282528.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/images/8cfbaf9564ce629fde77ec5a6aed66a72ecb793e6ba45ab2cf966ca043282528.jpg)

![9b26422d4a0a5152993c69522f87c66704b4d64d1810d920b67b865ac5aee1c5.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/images/9b26422d4a0a5152993c69522f87c66704b4d64d1810d920b67b865ac5aee1c5.jpg)

![a1678b006976efb03d802e5c97adadfb2f0ccc4570fee2d8d41f976c5287372b.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/images/a1678b006976efb03d802e5c97adadfb2f0ccc4570fee2d8d41f976c5287372b.jpg)

![aa5e3c05ca0a633ad6d8ca618e5cccff63ef7b3526307f6e2ae1c8e5bd2ed436.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/images/aa5e3c05ca0a633ad6d8ca618e5cccff63ef7b3526307f6e2ae1c8e5bd2ed436.jpg)

![c204de5f9b248be24695146691d98bf5097e5710a65a4e0dad3ca4c4621d9ef1.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/images/c204de5f9b248be24695146691d98bf5097e5710a65a4e0dad3ca4c4621d9ef1.jpg)

![db5e33abdf19f5597cc1d10d1fa0a4e7fd016f77fcd49c04d5e8c3e140d53988.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/images/db5e33abdf19f5597cc1d10d1fa0a4e7fd016f77fcd49c04d5e8c3e140d53988.jpg)

![e2ab5ff2bed46bc50c6b447c1a75c1fa5bd373370deec9125e319bf36bbdb319.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/images/e2ab5ff2bed46bc50c6b447c1a75c1fa5bd373370deec9125e319bf36bbdb319.jpg)

![ed28a4b117a70e7a010794396e81006360ff71795fdd7660d1a42eb0ab5ecde9.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/images/ed28a4b117a70e7a010794396e81006360ff71795fdd7660d1a42eb0ab5ecde9.jpg)

![ef0a8133acbb0b8eec48fd83f582a3033a683b9fd2ca97d2515d4b16a048b9a4.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/images/ef0a8133acbb0b8eec48fd83f582a3033a683b9fd2ca97d2515d4b16a048b9a4.jpg)

### Tables

![1eb82caf7fccce1c1c63a9928fa772c6350c2f674655b03f030b71c2348104ad.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/tables/1eb82caf7fccce1c1c63a9928fa772c6350c2f674655b03f030b71c2348104ad.jpg)

![3be1d739bfe440c2ed43e9ff998dfd04bcf06a8d96e447446ab5bb0f6f713e41.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/tables/3be1d739bfe440c2ed43e9ff998dfd04bcf06a8d96e447446ab5bb0f6f713e41.jpg)

![5411f0f5c3c92c2f5efcce32f1e670df1dede3c9b6fca2244a57422dd0193746.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/tables/5411f0f5c3c92c2f5efcce32f1e670df1dede3c9b6fca2244a57422dd0193746.jpg)

![627e89f77362fbfec1cdc25d972fff0ab3122df300ef6fe962c246ea33718591.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/tables/627e89f77362fbfec1cdc25d972fff0ab3122df300ef6fe962c246ea33718591.jpg)

![7637b30af9273c79fa6e5f8338e4a9b18489fd59e5490636f729db7abdf82566.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/tables/7637b30af9273c79fa6e5f8338e4a9b18489fd59e5490636f729db7abdf82566.jpg)

![9aa27c27863440656c376daad74bb98016c8b771ce951dbbbd4eaa95cc778195.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/tables/9aa27c27863440656c376daad74bb98016c8b771ce951dbbbd4eaa95cc778195.jpg)

![a32c5d159a33280b7d3d49fea1b02d1fb5893406f929a235cddda2f5140f9e6d.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/tables/a32c5d159a33280b7d3d49fea1b02d1fb5893406f929a235cddda2f5140f9e6d.jpg)

![a5ee1cb9b64fca95f7441c9fa636aeeea1f7c676c6d603433c8714d30b02e1c4.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/tables/a5ee1cb9b64fca95f7441c9fa636aeeea1f7c676c6d603433c8714d30b02e1c4.jpg)

![b275e4d36f0350b40a2bd5bebf1d75449e45af6970a54a7c3d4cf53302d3e9f6.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/tables/b275e4d36f0350b40a2bd5bebf1d75449e45af6970a54a7c3d4cf53302d3e9f6.jpg)

![bc87a0c5ec4e19b95c157e1f63587fe32bf3908b69fc743a9c4fdadadadb9ed5.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/tables/bc87a0c5ec4e19b95c157e1f63587fe32bf3908b69fc743a9c4fdadadadb9ed5.jpg)

![e4038bddf9dd055adbf8eb5b9c18c06abfed203d70f7bc9500eda6ea570c56bd.jpg](../iclr_results/191_Do%20Vision-Language%20Models%20Represent%20Space%20and%20How_%20Evaluating%20Spatial%20Frame%20of%20Reference%20under%20Ambig/tables/e4038bddf9dd055adbf8eb5b9c18c06abfed203d70f7bc9500eda6ea570c56bd.jpg)

## Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects


### Images

![075fcb69778ab25b73fbb29cba6be4384d5240cd09d40302459d3ebf716263d8.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/075fcb69778ab25b73fbb29cba6be4384d5240cd09d40302459d3ebf716263d8.jpg)

![08f176bdb46350435613e7bad5361eb1b636dd5e6097b7c8435be20ba1bdde62.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/08f176bdb46350435613e7bad5361eb1b636dd5e6097b7c8435be20ba1bdde62.jpg)

![2adbccacecad16adc0264e024399150e2d692764207c5f3f52c1fb176650b0f1.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/2adbccacecad16adc0264e024399150e2d692764207c5f3f52c1fb176650b0f1.jpg)

![3e5bd0f51bf7a42517944510cc94bb84f65564df19530d8b11a8135fa1d92c1b.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/3e5bd0f51bf7a42517944510cc94bb84f65564df19530d8b11a8135fa1d92c1b.jpg)

![3f0b02ef457b00af7e46d2dc8b11802ecf43915ab145f6b22156a1e3ab849bbe.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/3f0b02ef457b00af7e46d2dc8b11802ecf43915ab145f6b22156a1e3ab849bbe.jpg)

![458014564a0f8b332c8fd8b319fe720536143ec9e2160ee33ab9d748e54e7613.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/458014564a0f8b332c8fd8b319fe720536143ec9e2160ee33ab9d748e54e7613.jpg)

![593803617fdb6c6040c3d58e4da3f2c93205eab229ebc09152af1b3573ba5789.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/593803617fdb6c6040c3d58e4da3f2c93205eab229ebc09152af1b3573ba5789.jpg)

![5a1ffc99d0eb7543b1d377e547bb0e32c55c3709cb752801e0489ec8c4d9fd96.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/5a1ffc99d0eb7543b1d377e547bb0e32c55c3709cb752801e0489ec8c4d9fd96.jpg)

![6a0e2ad5da0afbc3a8d822d2c4e531093e283d417e9dff771c36fbb0a34544f4.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/6a0e2ad5da0afbc3a8d822d2c4e531093e283d417e9dff771c36fbb0a34544f4.jpg)

![6b4c31157aa3a2c0fbab051a98d5e97723792d7bedd7723ce295199c6bbcbd5d.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/6b4c31157aa3a2c0fbab051a98d5e97723792d7bedd7723ce295199c6bbcbd5d.jpg)

![7520ae58d88f9a1c0f555f24c480d159fac2e867b7dc07641fcf32ec8f531136.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/7520ae58d88f9a1c0f555f24c480d159fac2e867b7dc07641fcf32ec8f531136.jpg)

![763126e34f6f6d40288f64b244e7850b8a00f946c5d6cebd5ed7351e4e90de67.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/763126e34f6f6d40288f64b244e7850b8a00f946c5d6cebd5ed7351e4e90de67.jpg)

![8aa960c87b3018257a7aa55fe33d19ff44746599ca7e8e2f695e94b4e81f280d.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/8aa960c87b3018257a7aa55fe33d19ff44746599ca7e8e2f695e94b4e81f280d.jpg)

![8e72c78424c5f03e5d4cba845a354315984b5dc54e5f34d24a48ec46ea0ed56a.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/8e72c78424c5f03e5d4cba845a354315984b5dc54e5f34d24a48ec46ea0ed56a.jpg)

![96431b795d59548f5b9fcbc907fbd7c85c9966989f0de97df4b800d01d04aa20.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/96431b795d59548f5b9fcbc907fbd7c85c9966989f0de97df4b800d01d04aa20.jpg)

![9c2142b1d6d4d56884ed76eac31470d0c7992a339a2d0f4f38e596556e20b4ff.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/9c2142b1d6d4d56884ed76eac31470d0c7992a339a2d0f4f38e596556e20b4ff.jpg)

![b83a12b25f0e8b7f9ab680d43cd886dfe3ba3751b418f791655c387f82197757.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/b83a12b25f0e8b7f9ab680d43cd886dfe3ba3751b418f791655c387f82197757.jpg)

![bb2d9508874fe79a3f27a49f232b2e251bce9616104296e0226de82c0d8e48f1.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/bb2d9508874fe79a3f27a49f232b2e251bce9616104296e0226de82c0d8e48f1.jpg)

![c7b12b0a57295f38a0ee95ffec5291c026c64c5e7ee3e00b8f2c7a4149d4beda.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/c7b12b0a57295f38a0ee95ffec5291c026c64c5e7ee3e00b8f2c7a4149d4beda.jpg)

![cee83605304313ef9e011c1d35e382239e4e42914e017153ad99d6bae014b70c.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/cee83605304313ef9e011c1d35e382239e4e42914e017153ad99d6bae014b70c.jpg)

![ceef9faa47cfe82993cb19a8abc5c198aba62af47d5b0df2f0b072cd6bb90370.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/ceef9faa47cfe82993cb19a8abc5c198aba62af47d5b0df2f0b072cd6bb90370.jpg)

![d017c400e396df43531ab7b9acf5e263a76c0d42d3a852564c65d533bb639940.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/d017c400e396df43531ab7b9acf5e263a76c0d42d3a852564c65d533bb639940.jpg)

![f0fc9e70f7282f1c911a8901ac7ac7f2a50d4480635c1686de36f96143b2a32a.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/f0fc9e70f7282f1c911a8901ac7ac7f2a50d4480635c1686de36f96143b2a32a.jpg)

![f8de026f7269a8a7f262ee3f19842adc844a8081e1481feb3341ecda4397047e.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/images/f8de026f7269a8a7f262ee3f19842adc844a8081e1481feb3341ecda4397047e.jpg)

### Tables

![205d21985dfea2059ed11aa98e1150eabb8ada34b73dee322216286a24c8b488.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/tables/205d21985dfea2059ed11aa98e1150eabb8ada34b73dee322216286a24c8b488.jpg)

![3f1f77a69793cff229ceb3017fcfde8fcc4803f79cd1ad86a2f1412ae061406a.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/tables/3f1f77a69793cff229ceb3017fcfde8fcc4803f79cd1ad86a2f1412ae061406a.jpg)

![58f6141f7652e1ebeb1fc17b67d31a84972fba94dac46d56a94d17c4af9a22f9.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/tables/58f6141f7652e1ebeb1fc17b67d31a84972fba94dac46d56a94d17c4af9a22f9.jpg)

![6b67157d6642a88a92164c4086c17483590135224c5b191f8812244a2cf25870.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/tables/6b67157d6642a88a92164c4086c17483590135224c5b191f8812244a2cf25870.jpg)

![f43e7b42bfee0a842ba46867b2592c1723dcee569b648772257e672b28bf68e6.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/tables/f43e7b42bfee0a842ba46867b2592c1723dcee569b648772257e672b28bf68e6.jpg)

![fe7a7bf18d475b9f845c84404999a769d9f8bf6f20096df4f6a0a69498fbc4fd.jpg](../iclr_results/192_Geometry-aware%20RL%20for%20Manipulation%20of%20Varying%20Shapes%20and%20Deformable%20Objects/tables/fe7a7bf18d475b9f845c84404999a769d9f8bf6f20096df4f6a0a69498fbc4fd.jpg)

## MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering


### Images

![113de8f7172c0798479571c5b8ff11f55f8a6cdbb8412c3de3b8fab8c7b61750.jpg](../iclr_results/193_MLE-bench_%20Evaluating%20Machine%20Learning%20Agents%20on%20Machine%20Learning%20Engineering/images/113de8f7172c0798479571c5b8ff11f55f8a6cdbb8412c3de3b8fab8c7b61750.jpg)

![2a596f4365c5b973641b57831fac35fa050edc8ed609a8c205c159f12bd534ef.jpg](../iclr_results/193_MLE-bench_%20Evaluating%20Machine%20Learning%20Agents%20on%20Machine%20Learning%20Engineering/images/2a596f4365c5b973641b57831fac35fa050edc8ed609a8c205c159f12bd534ef.jpg)

![35ea3926d7db527dfcebca563541f2cf9a98495d1dafa304e1523c30d127628f.jpg](../iclr_results/193_MLE-bench_%20Evaluating%20Machine%20Learning%20Agents%20on%20Machine%20Learning%20Engineering/images/35ea3926d7db527dfcebca563541f2cf9a98495d1dafa304e1523c30d127628f.jpg)

![74bbf95dd86b178b92320aacd8510d71cef589aa366742d31206ca2caa09e858.jpg](../iclr_results/193_MLE-bench_%20Evaluating%20Machine%20Learning%20Agents%20on%20Machine%20Learning%20Engineering/images/74bbf95dd86b178b92320aacd8510d71cef589aa366742d31206ca2caa09e858.jpg)

![a36f2d640a30621247bf69a85b47dfb80731a70536ec9baa2b13d4fd2728816c.jpg](../iclr_results/193_MLE-bench_%20Evaluating%20Machine%20Learning%20Agents%20on%20Machine%20Learning%20Engineering/images/a36f2d640a30621247bf69a85b47dfb80731a70536ec9baa2b13d4fd2728816c.jpg)

![cb64380fcd4cfce35d22852573195945b9790bd3d4c9b26fd30f69f1576bbc03.jpg](../iclr_results/193_MLE-bench_%20Evaluating%20Machine%20Learning%20Agents%20on%20Machine%20Learning%20Engineering/images/cb64380fcd4cfce35d22852573195945b9790bd3d4c9b26fd30f69f1576bbc03.jpg)

![cd23bd5753d620c2fe6c14742fec1b31da1931dc9336a89527ea5c2d702eeb0f.jpg](../iclr_results/193_MLE-bench_%20Evaluating%20Machine%20Learning%20Agents%20on%20Machine%20Learning%20Engineering/images/cd23bd5753d620c2fe6c14742fec1b31da1931dc9336a89527ea5c2d702eeb0f.jpg)

![e71f43e6bb95e75d232fbeff7e4bc006b8abdaadd848957ed4ef706271ce6c23.jpg](../iclr_results/193_MLE-bench_%20Evaluating%20Machine%20Learning%20Agents%20on%20Machine%20Learning%20Engineering/images/e71f43e6bb95e75d232fbeff7e4bc006b8abdaadd848957ed4ef706271ce6c23.jpg)

### Tables

![4234dd653565ccc15638699f95e882b71d96676dd136a69dc92a1cbfea583928.jpg](../iclr_results/193_MLE-bench_%20Evaluating%20Machine%20Learning%20Agents%20on%20Machine%20Learning%20Engineering/tables/4234dd653565ccc15638699f95e882b71d96676dd136a69dc92a1cbfea583928.jpg)

![42fae37f26dddbdd33628e5609705a9fac29c9d3d012c129987e2cc21f4c2ad4.jpg](../iclr_results/193_MLE-bench_%20Evaluating%20Machine%20Learning%20Agents%20on%20Machine%20Learning%20Engineering/tables/42fae37f26dddbdd33628e5609705a9fac29c9d3d012c129987e2cc21f4c2ad4.jpg)

![5326c2d86de9e612caddb783226a502e5b8b746ee377adfd6aebe8e0cf1b2861.jpg](../iclr_results/193_MLE-bench_%20Evaluating%20Machine%20Learning%20Agents%20on%20Machine%20Learning%20Engineering/tables/5326c2d86de9e612caddb783226a502e5b8b746ee377adfd6aebe8e0cf1b2861.jpg)

![6aa6ed9819a07a96b05f4f33016865e788548cb1d99154a8d83285d1968946f0.jpg](../iclr_results/193_MLE-bench_%20Evaluating%20Machine%20Learning%20Agents%20on%20Machine%20Learning%20Engineering/tables/6aa6ed9819a07a96b05f4f33016865e788548cb1d99154a8d83285d1968946f0.jpg)

![8f9e9638dd0bd1468da51332d86fa0b8b2121a3a819b542b150b13e01024a1f4.jpg](../iclr_results/193_MLE-bench_%20Evaluating%20Machine%20Learning%20Agents%20on%20Machine%20Learning%20Engineering/tables/8f9e9638dd0bd1468da51332d86fa0b8b2121a3a819b542b150b13e01024a1f4.jpg)

![a2a3ac6e5bdc8ebcd791f790d5a0fbfab7c67e8dd6dcded8d8da34e13f76006b.jpg](../iclr_results/193_MLE-bench_%20Evaluating%20Machine%20Learning%20Agents%20on%20Machine%20Learning%20Engineering/tables/a2a3ac6e5bdc8ebcd791f790d5a0fbfab7c67e8dd6dcded8d8da34e13f76006b.jpg)

![a2b042947cfaec87cf1d5195c3f17d51edceb23e7f5098141694da80d8e444ae.jpg](../iclr_results/193_MLE-bench_%20Evaluating%20Machine%20Learning%20Agents%20on%20Machine%20Learning%20Engineering/tables/a2b042947cfaec87cf1d5195c3f17d51edceb23e7f5098141694da80d8e444ae.jpg)

![a955eb4a4ffc139cd8a794695cf81a56816937423750039f2b6bad08528686e0.jpg](../iclr_results/193_MLE-bench_%20Evaluating%20Machine%20Learning%20Agents%20on%20Machine%20Learning%20Engineering/tables/a955eb4a4ffc139cd8a794695cf81a56816937423750039f2b6bad08528686e0.jpg)

![b4a159474db7db6a25f234bd3adb7cb7b4a7484ce71119ad3e796d4d0bb4fd52.jpg](../iclr_results/193_MLE-bench_%20Evaluating%20Machine%20Learning%20Agents%20on%20Machine%20Learning%20Engineering/tables/b4a159474db7db6a25f234bd3adb7cb7b4a7484ce71119ad3e796d4d0bb4fd52.jpg)

![b4e81fff91ad8a8b5c868ee181f11025a106efb454706f3251ecb11b6082c851.jpg](../iclr_results/193_MLE-bench_%20Evaluating%20Machine%20Learning%20Agents%20on%20Machine%20Learning%20Engineering/tables/b4e81fff91ad8a8b5c868ee181f11025a106efb454706f3251ecb11b6082c851.jpg)

![b58e59c70ee4db1bc80a0df1be5f93989a02a40a903a3ae47d5040aaa997549e.jpg](../iclr_results/193_MLE-bench_%20Evaluating%20Machine%20Learning%20Agents%20on%20Machine%20Learning%20Engineering/tables/b58e59c70ee4db1bc80a0df1be5f93989a02a40a903a3ae47d5040aaa997549e.jpg)

![c4f310303903e89b650dd516d27366ebea61e9fdc9d0837a313f9ae109210c2f.jpg](../iclr_results/193_MLE-bench_%20Evaluating%20Machine%20Learning%20Agents%20on%20Machine%20Learning%20Engineering/tables/c4f310303903e89b650dd516d27366ebea61e9fdc9d0837a313f9ae109210c2f.jpg)

![e4d9125a4c159a01e27c65af0c409efdab1a1484b3e74424182cd1bef79c456e.jpg](../iclr_results/193_MLE-bench_%20Evaluating%20Machine%20Learning%20Agents%20on%20Machine%20Learning%20Engineering/tables/e4d9125a4c159a01e27c65af0c409efdab1a1484b3e74424182cd1bef79c456e.jpg)

![f636819df7570467fb31f1a35a3144b5ae1a3eda01931d04bf4dbef02b88b809.jpg](../iclr_results/193_MLE-bench_%20Evaluating%20Machine%20Learning%20Agents%20on%20Machine%20Learning%20Engineering/tables/f636819df7570467fb31f1a35a3144b5ae1a3eda01931d04bf4dbef02b88b809.jpg)

## Safety Alignment Should be Made More Than Just a Few Tokens Deep


### Images

![4b800220cba658daf974a22cfe61008d50b895c3da4ef7308bb28d72ef6f0d68.jpg](../iclr_results/194_Safety%20Alignment%20Should%20be%20Made%20More%20Than%20Just%20a%20Few%20Tokens%20Deep/images/4b800220cba658daf974a22cfe61008d50b895c3da4ef7308bb28d72ef6f0d68.jpg)

![76da006efc42a6b1dae5551122abd80c97b6b4667e97ca1bc5e5c9ec6fe7e9b7.jpg](../iclr_results/194_Safety%20Alignment%20Should%20be%20Made%20More%20Than%20Just%20a%20Few%20Tokens%20Deep/images/76da006efc42a6b1dae5551122abd80c97b6b4667e97ca1bc5e5c9ec6fe7e9b7.jpg)

![a752cf94b9a071b6b3579c8c5f52f08ff3c1ffc075fc12767552c929e51fbd64.jpg](../iclr_results/194_Safety%20Alignment%20Should%20be%20Made%20More%20Than%20Just%20a%20Few%20Tokens%20Deep/images/a752cf94b9a071b6b3579c8c5f52f08ff3c1ffc075fc12767552c929e51fbd64.jpg)

![c8d375bbbd04c393eac2856f828c23018f177f1732c9937d10d1ca7bc94624f1.jpg](../iclr_results/194_Safety%20Alignment%20Should%20be%20Made%20More%20Than%20Just%20a%20Few%20Tokens%20Deep/images/c8d375bbbd04c393eac2856f828c23018f177f1732c9937d10d1ca7bc94624f1.jpg)

![fc5fe47dae4fdc141e6a62f745e5eb229ed2006fc1aa21b11e101de4207e08cf.jpg](../iclr_results/194_Safety%20Alignment%20Should%20be%20Made%20More%20Than%20Just%20a%20Few%20Tokens%20Deep/images/fc5fe47dae4fdc141e6a62f745e5eb229ed2006fc1aa21b11e101de4207e08cf.jpg)

### Tables

![008ae47c21da533bb3c4da819adb5769e2b64943865bbf4686194b364c4a8d0f.jpg](../iclr_results/194_Safety%20Alignment%20Should%20be%20Made%20More%20Than%20Just%20a%20Few%20Tokens%20Deep/tables/008ae47c21da533bb3c4da819adb5769e2b64943865bbf4686194b364c4a8d0f.jpg)

![021e4d231bf996345f84bccb102ca6bea654b195adeb64ae1269c0fc1de30964.jpg](../iclr_results/194_Safety%20Alignment%20Should%20be%20Made%20More%20Than%20Just%20a%20Few%20Tokens%20Deep/tables/021e4d231bf996345f84bccb102ca6bea654b195adeb64ae1269c0fc1de30964.jpg)

![075b928526b8c7bcd8d6c6335348b53837c04281f6c1d5899e725f08735d76e2.jpg](../iclr_results/194_Safety%20Alignment%20Should%20be%20Made%20More%20Than%20Just%20a%20Few%20Tokens%20Deep/tables/075b928526b8c7bcd8d6c6335348b53837c04281f6c1d5899e725f08735d76e2.jpg)

![11242c98edd254c173173a8fcf34cdb4f8fa7bef28ce98f08da0a1f5815256d6.jpg](../iclr_results/194_Safety%20Alignment%20Should%20be%20Made%20More%20Than%20Just%20a%20Few%20Tokens%20Deep/tables/11242c98edd254c173173a8fcf34cdb4f8fa7bef28ce98f08da0a1f5815256d6.jpg)

![1777ae38040217bb772e4daeead2389ef748ff80be3a710af6a399289ea167ea.jpg](../iclr_results/194_Safety%20Alignment%20Should%20be%20Made%20More%20Than%20Just%20a%20Few%20Tokens%20Deep/tables/1777ae38040217bb772e4daeead2389ef748ff80be3a710af6a399289ea167ea.jpg)

![2677d0d1cbbd5d6cb4eed185073c4adfa40f1fbd5fc3b24a4d1b89e80a7ef36c.jpg](../iclr_results/194_Safety%20Alignment%20Should%20be%20Made%20More%20Than%20Just%20a%20Few%20Tokens%20Deep/tables/2677d0d1cbbd5d6cb4eed185073c4adfa40f1fbd5fc3b24a4d1b89e80a7ef36c.jpg)

![27ca532c216bc9a04a18810e42d082ccebae9b154dc9ba802f1b993d68c65769.jpg](../iclr_results/194_Safety%20Alignment%20Should%20be%20Made%20More%20Than%20Just%20a%20Few%20Tokens%20Deep/tables/27ca532c216bc9a04a18810e42d082ccebae9b154dc9ba802f1b993d68c65769.jpg)

![5672d4b273b83c2db2c6e908e839f40d626b01b1c813ec2d794376dd049a28f4.jpg](../iclr_results/194_Safety%20Alignment%20Should%20be%20Made%20More%20Than%20Just%20a%20Few%20Tokens%20Deep/tables/5672d4b273b83c2db2c6e908e839f40d626b01b1c813ec2d794376dd049a28f4.jpg)

![62945a7f55096ef0118ebdc4aeb3cdcbfb830f50b25660484ad21c0a49eb75a6.jpg](../iclr_results/194_Safety%20Alignment%20Should%20be%20Made%20More%20Than%20Just%20a%20Few%20Tokens%20Deep/tables/62945a7f55096ef0118ebdc4aeb3cdcbfb830f50b25660484ad21c0a49eb75a6.jpg)

![a0f446657228b702b2b0ddf0a9075e9c7c9f2b52a65247bcab3fc8f8cf4447ac.jpg](../iclr_results/194_Safety%20Alignment%20Should%20be%20Made%20More%20Than%20Just%20a%20Few%20Tokens%20Deep/tables/a0f446657228b702b2b0ddf0a9075e9c7c9f2b52a65247bcab3fc8f8cf4447ac.jpg)

![afb2f38a816e4b2baa1ee19505f5e5d8deb3ae850c30e3ac47437caf6b1b168a.jpg](../iclr_results/194_Safety%20Alignment%20Should%20be%20Made%20More%20Than%20Just%20a%20Few%20Tokens%20Deep/tables/afb2f38a816e4b2baa1ee19505f5e5d8deb3ae850c30e3ac47437caf6b1b168a.jpg)

![e9a6b5af4bcb99640eea55da3c9f8daeed12f6babd8c1d1987f3e3093573bd42.jpg](../iclr_results/194_Safety%20Alignment%20Should%20be%20Made%20More%20Than%20Just%20a%20Few%20Tokens%20Deep/tables/e9a6b5af4bcb99640eea55da3c9f8daeed12f6babd8c1d1987f3e3093573bd42.jpg)

![fc5245a4525737203787da49392de6f0211c41744dd65e7b6f54fffafef19e27.jpg](../iclr_results/194_Safety%20Alignment%20Should%20be%20Made%20More%20Than%20Just%20a%20Few%20Tokens%20Deep/tables/fc5245a4525737203787da49392de6f0211c41744dd65e7b6f54fffafef19e27.jpg)

## Variational Diffusion Posterior Sampling with Midpoint Guidance


### Images

![0f0b6d0a932771a47aeed22d6dce3917136395e0a2ccc44e78eb1f027365b58e.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/0f0b6d0a932771a47aeed22d6dce3917136395e0a2ccc44e78eb1f027365b58e.jpg)

![1b6f92201b8e27343e20ff5459171295e347772e1bbb6e4f39460fe0a56ee771.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/1b6f92201b8e27343e20ff5459171295e347772e1bbb6e4f39460fe0a56ee771.jpg)

![1c597c82a884027482596b78cbe5088d7c18b7823e92217ff51eb5bbadcbbea2.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/1c597c82a884027482596b78cbe5088d7c18b7823e92217ff51eb5bbadcbbea2.jpg)

![2544df905dc9d9fcd7e43795822ba77f0248debc45ceb44e4ff85bd2b688f781.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/2544df905dc9d9fcd7e43795822ba77f0248debc45ceb44e4ff85bd2b688f781.jpg)

![37dd514426969a7f9b50b7cde810ffb0d621a3932ae2ad2fadf9d6ee336c9468.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/37dd514426969a7f9b50b7cde810ffb0d621a3932ae2ad2fadf9d6ee336c9468.jpg)

![48b412de9426a4e9464e04108bf6fafd250f8eb4bbf1406088f0a739a1c9b842.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/48b412de9426a4e9464e04108bf6fafd250f8eb4bbf1406088f0a739a1c9b842.jpg)

![4dce02578e419ea7d9175f31602da06e50be31793624be5f8605286149cd6dce.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/4dce02578e419ea7d9175f31602da06e50be31793624be5f8605286149cd6dce.jpg)

![4f8ac03db7a5c8d552322f2b5416b32721fc6ac3571d8142896ac3a3b25bc7a2.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/4f8ac03db7a5c8d552322f2b5416b32721fc6ac3571d8142896ac3a3b25bc7a2.jpg)

![51f4c0634a1696e14dd4b06a54fdb82c68eb7dec2ac661faa9d602e52c7709c7.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/51f4c0634a1696e14dd4b06a54fdb82c68eb7dec2ac661faa9d602e52c7709c7.jpg)

![7911c4922bb93c6247f9e544c0eea4913082ae1ad51a78916112fd20f8493424.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/7911c4922bb93c6247f9e544c0eea4913082ae1ad51a78916112fd20f8493424.jpg)

![893f173a31f5cdaa145f3ed566dd2556027a1ed8460164b96c1fd06a6d5cbec0.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/893f173a31f5cdaa145f3ed566dd2556027a1ed8460164b96c1fd06a6d5cbec0.jpg)

![8b73f3e2006cfc625b8797a74317e74b3d6481c295321d4cfafefa84307640f1.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/8b73f3e2006cfc625b8797a74317e74b3d6481c295321d4cfafefa84307640f1.jpg)

![9500fd563d28bd30fd22426363072e76d43cb55c8bc8f68ef975594e7b9b76ca.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/9500fd563d28bd30fd22426363072e76d43cb55c8bc8f68ef975594e7b9b76ca.jpg)

![9b64aba5f30230d9475b387e1e3fd9cf99d3602af3f15432a21ee1fbb32582e3.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/9b64aba5f30230d9475b387e1e3fd9cf99d3602af3f15432a21ee1fbb32582e3.jpg)

![9df036e96141168b3c567701506f8d3b4f0a23171eee0bee72e93f7246bc6101.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/9df036e96141168b3c567701506f8d3b4f0a23171eee0bee72e93f7246bc6101.jpg)

![a3d543ed4c25f437605f9ad6f09e13bf7c23812667f585ecc07c7e7b7d8502d5.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/a3d543ed4c25f437605f9ad6f09e13bf7c23812667f585ecc07c7e7b7d8502d5.jpg)

![a8adc80ea8a57f92970d696298739ee883ec9d97a22db9a380ffb9f4dee9171c.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/a8adc80ea8a57f92970d696298739ee883ec9d97a22db9a380ffb9f4dee9171c.jpg)

![aab01302eb508c352443d4bb628d27179e9d9478e0839ff3aa170e2648fecde6.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/aab01302eb508c352443d4bb628d27179e9d9478e0839ff3aa170e2648fecde6.jpg)

![acf576186edff724a74a1a09038bda60795f05932f8825ae768411cd6f0acf1e.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/acf576186edff724a74a1a09038bda60795f05932f8825ae768411cd6f0acf1e.jpg)

![bab5d3901df9298ffe8416d239f7c4246deb76b16468e836b96acad7cf77c881.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/bab5d3901df9298ffe8416d239f7c4246deb76b16468e836b96acad7cf77c881.jpg)

![c4a7dfd3f40b3473e38770bc1a07c6c985e4f2db43523625681507deba45b44e.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/c4a7dfd3f40b3473e38770bc1a07c6c985e4f2db43523625681507deba45b44e.jpg)

![c4d15422a8582e8ad7e5a6f2f3f75038d4c0253143614176c9e56f0f65660ee9.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/c4d15422a8582e8ad7e5a6f2f3f75038d4c0253143614176c9e56f0f65660ee9.jpg)

![d8deabae054a8e469f1c0297c651f27c6ffdd701717fe95ff62eff4908ff2003.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/d8deabae054a8e469f1c0297c651f27c6ffdd701717fe95ff62eff4908ff2003.jpg)

![e02e97175df6803e42ad52f27a82221d320cf2f6d506209b63ebf79d2b300c32.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/e02e97175df6803e42ad52f27a82221d320cf2f6d506209b63ebf79d2b300c32.jpg)

![e6e8a5610acd8c00d9d7c64e1fa0f80b2d871c931c5fc63271e97d59ca3fdd3e.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/e6e8a5610acd8c00d9d7c64e1fa0f80b2d871c931c5fc63271e97d59ca3fdd3e.jpg)

![f19405218db50c35e16f04b5ad2d7e4dba447e69100dc391ce0e864314202e50.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/f19405218db50c35e16f04b5ad2d7e4dba447e69100dc391ce0e864314202e50.jpg)

![fe754699e54bbb4098a14f44e2c71023b0086b285f70931809b74eb1ba733685.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/images/fe754699e54bbb4098a14f44e2c71023b0086b285f70931809b74eb1ba733685.jpg)

### Tables

![1e1b897c902f33612e83e252445304a3cc3ba92b803b84a7218b01317a738b52.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/tables/1e1b897c902f33612e83e252445304a3cc3ba92b803b84a7218b01317a738b52.jpg)

![2cc73b86768101342a2f4650ae0d80fc1945d552178630d70a4988dda08e4880.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/tables/2cc73b86768101342a2f4650ae0d80fc1945d552178630d70a4988dda08e4880.jpg)

![3a924825aafa484813ca9de9ff9d532c618df445063340d23d6e3f700612a83a.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/tables/3a924825aafa484813ca9de9ff9d532c618df445063340d23d6e3f700612a83a.jpg)

![3dd1032cac2799d71ba2f3ef57148d27525895ee076f5c3d305bc24c10d87542.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/tables/3dd1032cac2799d71ba2f3ef57148d27525895ee076f5c3d305bc24c10d87542.jpg)

![4339f84fd33f56c56ee41f6688c777eb847f3a96e106c77456453a64bce40304.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/tables/4339f84fd33f56c56ee41f6688c777eb847f3a96e106c77456453a64bce40304.jpg)

![53ea1d79b93bdaddd7fbd7c0ee482d98be2867673a376df86ec829da1c3b70ee.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/tables/53ea1d79b93bdaddd7fbd7c0ee482d98be2867673a376df86ec829da1c3b70ee.jpg)

![57d1aed8de1991c4eff5bd73a74313574b50337806815ef7ecea4afb7ab75bf1.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/tables/57d1aed8de1991c4eff5bd73a74313574b50337806815ef7ecea4afb7ab75bf1.jpg)

![8c19964bc695c6950733ca5bfef21ad9f35ec0041e441cfddcbf12d57e05d6b2.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/tables/8c19964bc695c6950733ca5bfef21ad9f35ec0041e441cfddcbf12d57e05d6b2.jpg)

![8de3572676173c45c113a06045a869b6056f26f91726b157dc1abc4e17f37238.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/tables/8de3572676173c45c113a06045a869b6056f26f91726b157dc1abc4e17f37238.jpg)

![ab8dead2e905b78828be2c04947e93e77264199b87242b381516d01798348a1d.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/tables/ab8dead2e905b78828be2c04947e93e77264199b87242b381516d01798348a1d.jpg)

![b03fd9db2b8d253f0968c73807cad9895ff2539629cf0ca951084b824c67ae1a.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/tables/b03fd9db2b8d253f0968c73807cad9895ff2539629cf0ca951084b824c67ae1a.jpg)

![c2d4206af2a7326c293b2c4a8fa1536f9e52b684670383bf42a2d086a6a2e971.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/tables/c2d4206af2a7326c293b2c4a8fa1536f9e52b684670383bf42a2d086a6a2e971.jpg)

![d1ab97bdf655db73ea86658a973434beba3dc2740cbad11b7f04941713c33f0b.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/tables/d1ab97bdf655db73ea86658a973434beba3dc2740cbad11b7f04941713c33f0b.jpg)

![d746e21124df404ed6f6ed7c25269bf1b4a969722672d3e8abdb152b254d690e.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/tables/d746e21124df404ed6f6ed7c25269bf1b4a969722672d3e8abdb152b254d690e.jpg)

![de5039ec5d2081d6a35e2b883564387b32ab4502ed0b4e6ff96148704d9393f7.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/tables/de5039ec5d2081d6a35e2b883564387b32ab4502ed0b4e6ff96148704d9393f7.jpg)

![ec29b99cafdba7d76715c8bed8c2bc9eba5b803a6185cfd843afba9adcea1935.jpg](../iclr_results/195_Variational%20Diffusion%20Posterior%20Sampling%20with%20Midpoint%20Guidance/tables/ec29b99cafdba7d76715c8bed8c2bc9eba5b803a6185cfd843afba9adcea1935.jpg)

## NeuralPlane: Structured 3D Reconstruction in Planar Primitives with Neural Fields


### Images

![09e31952bb976bec8f2639b2584ce55af4590b1193551153c23a9982768f98d7.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/images/09e31952bb976bec8f2639b2584ce55af4590b1193551153c23a9982768f98d7.jpg)

![0b5bcf209c64d6769356aa6178f748a145f96ff5e71189c0f495ad74ca11b4e1.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/images/0b5bcf209c64d6769356aa6178f748a145f96ff5e71189c0f495ad74ca11b4e1.jpg)

![3dcb521e2b26a0589369aa1e4d1a1ca7ea3b28ad0471d7e2ec6000373293b563.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/images/3dcb521e2b26a0589369aa1e4d1a1ca7ea3b28ad0471d7e2ec6000373293b563.jpg)

![4485dcd63c0c6bf170f5c4721ddf122b6e0017129e5eed7aafd82d5cbe2685cd.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/images/4485dcd63c0c6bf170f5c4721ddf122b6e0017129e5eed7aafd82d5cbe2685cd.jpg)

![5d675b0f7adc1fa8074dc31c501e0663d89721de799b65ad0960af74ec329f55.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/images/5d675b0f7adc1fa8074dc31c501e0663d89721de799b65ad0960af74ec329f55.jpg)

![657f5782a396918835d5635c2ecee22df2536572da6c62d264db16aa5917d3ea.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/images/657f5782a396918835d5635c2ecee22df2536572da6c62d264db16aa5917d3ea.jpg)

![6aaf4eafe95d0f3283782167ce7c9bf1ff9c8646a59cedb9d45fb97dc448c9a7.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/images/6aaf4eafe95d0f3283782167ce7c9bf1ff9c8646a59cedb9d45fb97dc448c9a7.jpg)

![935fde968ada1e492512886a163b7f454dfeef4cefeb351a3b7fdb3fb9d89df2.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/images/935fde968ada1e492512886a163b7f454dfeef4cefeb351a3b7fdb3fb9d89df2.jpg)

![9a74b6b504362bf005c502c9c29996c579c4aab1b07ac558605d6ad6bc96b5cf.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/images/9a74b6b504362bf005c502c9c29996c579c4aab1b07ac558605d6ad6bc96b5cf.jpg)

![9adbf459f8f6d631cc6d5c4218f5ed1637067fd3b7f61674e4e7eb64be13f4b3.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/images/9adbf459f8f6d631cc6d5c4218f5ed1637067fd3b7f61674e4e7eb64be13f4b3.jpg)

![a08a1803ea9b6c8817da5d2029851e467af62acceeada861b04fe8a6ac1cda8d.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/images/a08a1803ea9b6c8817da5d2029851e467af62acceeada861b04fe8a6ac1cda8d.jpg)

![a1421dcd08d7e7d2e4f1bb242b9f739ce0aa85403b5f4e4b7f6c3be7ab10bda3.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/images/a1421dcd08d7e7d2e4f1bb242b9f739ce0aa85403b5f4e4b7f6c3be7ab10bda3.jpg)

![bcde04b87dc4245bf6f58ac8a5c9891d0d5719e153c29a76c8a963c977bc6002.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/images/bcde04b87dc4245bf6f58ac8a5c9891d0d5719e153c29a76c8a963c977bc6002.jpg)

![ca59f7e3ba5acd5286693df9e46eec57c75755b627bceb1d35e7ef41c332ad04.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/images/ca59f7e3ba5acd5286693df9e46eec57c75755b627bceb1d35e7ef41c332ad04.jpg)

![d249b3cd06ad7774cccf1e9757c357acd34d0df93d0810e4bf9645c470553954.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/images/d249b3cd06ad7774cccf1e9757c357acd34d0df93d0810e4bf9645c470553954.jpg)

![d6672230bdf2a2c3b8667d9cd63aea9613f1f819bd2f4eba11315e4210f64ade.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/images/d6672230bdf2a2c3b8667d9cd63aea9613f1f819bd2f4eba11315e4210f64ade.jpg)

### Tables

![2a4381d6e6a4d21aff5d6b06dab3bb3cf32ca2a049c60fe7e71dde76f6704d81.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/tables/2a4381d6e6a4d21aff5d6b06dab3bb3cf32ca2a049c60fe7e71dde76f6704d81.jpg)

![3d3cc57231a5c0f82d3200ac4034514bc31d6da662ae4920f0f5f245d0951fd7.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/tables/3d3cc57231a5c0f82d3200ac4034514bc31d6da662ae4920f0f5f245d0951fd7.jpg)

![683620c0004a89d5c9d75223030b14ab5f12107cd275067d57b32b3ce2a051fe.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/tables/683620c0004a89d5c9d75223030b14ab5f12107cd275067d57b32b3ce2a051fe.jpg)

![704c1429281ec803bf8299c9935b44c2742c5ba0a4a3a7c2396d925fad7bfc1e.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/tables/704c1429281ec803bf8299c9935b44c2742c5ba0a4a3a7c2396d925fad7bfc1e.jpg)

![8024dec0b22aebb749cbc70cf078ddfd1818fa952243e825baed5d6830ebb16d.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/tables/8024dec0b22aebb749cbc70cf078ddfd1818fa952243e825baed5d6830ebb16d.jpg)

![a0f10efcc352f13f9a1473e38922050336cf006fa61b7b6ca296cdcb515d2fa0.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/tables/a0f10efcc352f13f9a1473e38922050336cf006fa61b7b6ca296cdcb515d2fa0.jpg)

![cd6a503d75d8d31afa6b58fc20c2ea4eb5af59a0f9d85a03ed6b8eb69735578a.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/tables/cd6a503d75d8d31afa6b58fc20c2ea4eb5af59a0f9d85a03ed6b8eb69735578a.jpg)

![e5f203e44bb9362aec009f4ed610e13c7cc93701745055ae3f1b5f3259009881.jpg](../iclr_results/196_NeuralPlane_%20Structured%203D%20Reconstruction%20in%20Planar%20Primitives%20with%20Neural%20Fields/tables/e5f203e44bb9362aec009f4ed610e13c7cc93701745055ae3f1b5f3259009881.jpg)

## SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning


### Images

![24ea61501cba9b76bc1b8bb963ab7b0bc466f7a3db140509d005cecfc8476994.jpg](../iclr_results/197_SD-LoRA_%20Scalable%20Decoupled%20Low-Rank%20Adaptation%20for%20Class%20Incremental%20Learning/images/24ea61501cba9b76bc1b8bb963ab7b0bc466f7a3db140509d005cecfc8476994.jpg)

![3eefbd216c50d9f7392704dab2e0d71248639c3f50360140cfba7ff56f8b9fd6.jpg](../iclr_results/197_SD-LoRA_%20Scalable%20Decoupled%20Low-Rank%20Adaptation%20for%20Class%20Incremental%20Learning/images/3eefbd216c50d9f7392704dab2e0d71248639c3f50360140cfba7ff56f8b9fd6.jpg)

![4ac8662c11be378837cd6730c2f964dfaddc81db1af125e9c0e7d64ba92f3002.jpg](../iclr_results/197_SD-LoRA_%20Scalable%20Decoupled%20Low-Rank%20Adaptation%20for%20Class%20Incremental%20Learning/images/4ac8662c11be378837cd6730c2f964dfaddc81db1af125e9c0e7d64ba92f3002.jpg)

![54c7383ff887852315c06459abcdee358b68680598ec6a62db9caeb327d0a60f.jpg](../iclr_results/197_SD-LoRA_%20Scalable%20Decoupled%20Low-Rank%20Adaptation%20for%20Class%20Incremental%20Learning/images/54c7383ff887852315c06459abcdee358b68680598ec6a62db9caeb327d0a60f.jpg)

![715a437f872443d8345ea1977d6711d4e82ad056340bf114540e271255e51a00.jpg](../iclr_results/197_SD-LoRA_%20Scalable%20Decoupled%20Low-Rank%20Adaptation%20for%20Class%20Incremental%20Learning/images/715a437f872443d8345ea1977d6711d4e82ad056340bf114540e271255e51a00.jpg)

![75b5ef43c1fc7b2df41777eac4256226f9225b68af4900ddab0f094e4e9218cb.jpg](../iclr_results/197_SD-LoRA_%20Scalable%20Decoupled%20Low-Rank%20Adaptation%20for%20Class%20Incremental%20Learning/images/75b5ef43c1fc7b2df41777eac4256226f9225b68af4900ddab0f094e4e9218cb.jpg)

![8a5e1a2bd17c827016ee07aab28070afa18c80d8c42f97f51929453eb2507590.jpg](../iclr_results/197_SD-LoRA_%20Scalable%20Decoupled%20Low-Rank%20Adaptation%20for%20Class%20Incremental%20Learning/images/8a5e1a2bd17c827016ee07aab28070afa18c80d8c42f97f51929453eb2507590.jpg)

### Tables

![14ac2cf9c3e0a3c52221dab9c184711b75ba7913f77a01f9f9018af64a8694de.jpg](../iclr_results/197_SD-LoRA_%20Scalable%20Decoupled%20Low-Rank%20Adaptation%20for%20Class%20Incremental%20Learning/tables/14ac2cf9c3e0a3c52221dab9c184711b75ba7913f77a01f9f9018af64a8694de.jpg)

![2d21edeb9fab4100dc8c9d9a8d30d4cab21061dde839e2c4b458e22dac09382b.jpg](../iclr_results/197_SD-LoRA_%20Scalable%20Decoupled%20Low-Rank%20Adaptation%20for%20Class%20Incremental%20Learning/tables/2d21edeb9fab4100dc8c9d9a8d30d4cab21061dde839e2c4b458e22dac09382b.jpg)

![3e8aaaeac3f44ce15a1c57779305918e617f13490f2e47f55ae1302ac5b1df3e.jpg](../iclr_results/197_SD-LoRA_%20Scalable%20Decoupled%20Low-Rank%20Adaptation%20for%20Class%20Incremental%20Learning/tables/3e8aaaeac3f44ce15a1c57779305918e617f13490f2e47f55ae1302ac5b1df3e.jpg)

![44712f9dd949b444f6ca331ee24cafb9fbe269133f17d71ab0de809c999571ca.jpg](../iclr_results/197_SD-LoRA_%20Scalable%20Decoupled%20Low-Rank%20Adaptation%20for%20Class%20Incremental%20Learning/tables/44712f9dd949b444f6ca331ee24cafb9fbe269133f17d71ab0de809c999571ca.jpg)

![8ce5877883d23cc74272d8b3aedc75678fa94adde2169771591a67ff678ded88.jpg](../iclr_results/197_SD-LoRA_%20Scalable%20Decoupled%20Low-Rank%20Adaptation%20for%20Class%20Incremental%20Learning/tables/8ce5877883d23cc74272d8b3aedc75678fa94adde2169771591a67ff678ded88.jpg)

![da97092c8af025d91b948cebd9e9fdfa8e8af723ea5b06d7534ca02e69e4f876.jpg](../iclr_results/197_SD-LoRA_%20Scalable%20Decoupled%20Low-Rank%20Adaptation%20for%20Class%20Incremental%20Learning/tables/da97092c8af025d91b948cebd9e9fdfa8e8af723ea5b06d7534ca02e69e4f876.jpg)

## Energy-based Backdoor Defense Against Federated Graph Learning


### Images

![0939934d893d41c7c0652e7d70c59efded0e937f644c0887f06c45393524cca3.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/images/0939934d893d41c7c0652e7d70c59efded0e937f644c0887f06c45393524cca3.jpg)

![0cf0ef53cad69c50663ca8d8caf3187c2ef7f84a2d06a749cabb907a9113bfc4.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/images/0cf0ef53cad69c50663ca8d8caf3187c2ef7f84a2d06a749cabb907a9113bfc4.jpg)

![2948f9e393bb5a3d3381b0771d64ec7f08ab6b837da348be16503c9a6e8b8fa3.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/images/2948f9e393bb5a3d3381b0771d64ec7f08ab6b837da348be16503c9a6e8b8fa3.jpg)

![2cb242b924d463ef8f98b403e65aa95e4cb740495d889b000f7d526acb5a41d2.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/images/2cb242b924d463ef8f98b403e65aa95e4cb740495d889b000f7d526acb5a41d2.jpg)

![2e64dc908d4baa0f7fd9a09c88abb3a8205d3a1fd78fbc24a690a3a579a80ef8.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/images/2e64dc908d4baa0f7fd9a09c88abb3a8205d3a1fd78fbc24a690a3a579a80ef8.jpg)

![2f566db7f21bc03b8370ec7ee5d6eebde2f5e0bd5d8690df864f7bbb50dc7a8e.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/images/2f566db7f21bc03b8370ec7ee5d6eebde2f5e0bd5d8690df864f7bbb50dc7a8e.jpg)

![40b6b137c82c12665cd77389f8978bc7c131071dc7fcb2b3852e2f83930f7878.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/images/40b6b137c82c12665cd77389f8978bc7c131071dc7fcb2b3852e2f83930f7878.jpg)

![43c54334ccc59acd053e9a23a650b3bdbfcc36ae269b178279b35a9d03217268.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/images/43c54334ccc59acd053e9a23a650b3bdbfcc36ae269b178279b35a9d03217268.jpg)

![68ec2b21c2ec9799be574727c9e13c4a5a9634ffdab4771b77f90b4b0dd071f6.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/images/68ec2b21c2ec9799be574727c9e13c4a5a9634ffdab4771b77f90b4b0dd071f6.jpg)

![6f9875abc60af549891aa22992e0dd78229c3228eafc5ff69ba5ea542c2246b2.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/images/6f9875abc60af549891aa22992e0dd78229c3228eafc5ff69ba5ea542c2246b2.jpg)

![80652a2e908be63d70a05d3f41ac234033959dce052fd1c83d4c4a51117722b5.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/images/80652a2e908be63d70a05d3f41ac234033959dce052fd1c83d4c4a51117722b5.jpg)

![87f84241adda9b6226cc117eb3158124d4d8d96922a01610a0dd229051e58c6f.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/images/87f84241adda9b6226cc117eb3158124d4d8d96922a01610a0dd229051e58c6f.jpg)

![9198dac3039dcd00dcec2680f3dd02039602aeecb568c29d3b2fbb2bc6622a7b.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/images/9198dac3039dcd00dcec2680f3dd02039602aeecb568c29d3b2fbb2bc6622a7b.jpg)

![9c27b1daea98a6e4c8eaa3e0308d1ae8407999fb2148ca6d1a4bd51d32d4dfb5.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/images/9c27b1daea98a6e4c8eaa3e0308d1ae8407999fb2148ca6d1a4bd51d32d4dfb5.jpg)

![a43868583a6a7cbf9fce7c03788cacf33b3d85ce18120ea7b1d6fd1fed8f6798.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/images/a43868583a6a7cbf9fce7c03788cacf33b3d85ce18120ea7b1d6fd1fed8f6798.jpg)

![c2774b2a13502fc8cf6fec329c6064a3cebf3bd2ae77ca38296c1f10d79279c6.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/images/c2774b2a13502fc8cf6fec329c6064a3cebf3bd2ae77ca38296c1f10d79279c6.jpg)

![ddad47a33af4d034ca15e6708816f73a4fd597c436bbdac4a93fb3622299a606.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/images/ddad47a33af4d034ca15e6708816f73a4fd597c436bbdac4a93fb3622299a606.jpg)

![e56cd990f069b1dfed873df5ede4fabc985e5c6f213bec1dacc2de75f1a9f2e6.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/images/e56cd990f069b1dfed873df5ede4fabc985e5c6f213bec1dacc2de75f1a9f2e6.jpg)

![f3ebba93f93b640613d13c5177f81288414c563c0309580dc81733853b5373d1.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/images/f3ebba93f93b640613d13c5177f81288414c563c0309580dc81733853b5373d1.jpg)

![f5bbf0b3f6356abca27c38336e007a7947ffba49c10254161500f09c9a5f948b.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/images/f5bbf0b3f6356abca27c38336e007a7947ffba49c10254161500f09c9a5f948b.jpg)

### Tables

![169e6568525112ffd90b96e0d60ed4f699afecce936c6cadff5b8a77a64ae837.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/tables/169e6568525112ffd90b96e0d60ed4f699afecce936c6cadff5b8a77a64ae837.jpg)

![1fd0c1d26ff617f93fc765fcad01cf8e8c91fa3e298950c729ca6d07913f880d.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/tables/1fd0c1d26ff617f93fc765fcad01cf8e8c91fa3e298950c729ca6d07913f880d.jpg)

![2e289404f399871b8394f3998b432701ec05e4f86a85e1a75438b7322e18f927.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/tables/2e289404f399871b8394f3998b432701ec05e4f86a85e1a75438b7322e18f927.jpg)

![32bcb15dc78be4f2a6c2e500441c557c201d40f8eda441d284b9aeaf6bf5828d.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/tables/32bcb15dc78be4f2a6c2e500441c557c201d40f8eda441d284b9aeaf6bf5828d.jpg)

![5016cdc0661148c0c64052ce8bc80a2aabc4a6a7bb89de4bd265c96a7ebb1ed5.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/tables/5016cdc0661148c0c64052ce8bc80a2aabc4a6a7bb89de4bd265c96a7ebb1ed5.jpg)

![54fb54b2145483e6c1794640e2b4704623077c531a796485e85a52611eb37bb8.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/tables/54fb54b2145483e6c1794640e2b4704623077c531a796485e85a52611eb37bb8.jpg)

![84d5dd2128f16b3054f96bd9d5e75304b774cd6e1bc084d0875f8caa91eff001.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/tables/84d5dd2128f16b3054f96bd9d5e75304b774cd6e1bc084d0875f8caa91eff001.jpg)

![8f12eb81b5ced914f361cd69d8c50ea8f38c4d33a2c25765eda6d376d6f7e1f9.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/tables/8f12eb81b5ced914f361cd69d8c50ea8f38c4d33a2c25765eda6d376d6f7e1f9.jpg)

![dab6c3ce4774e9c229f523c37c4392a8495097def646c56a3f205618d5488c36.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/tables/dab6c3ce4774e9c229f523c37c4392a8495097def646c56a3f205618d5488c36.jpg)

![de4701a59968757e0599e37c4a238c1b0ec77e879ee754c481d4db825ed64381.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/tables/de4701a59968757e0599e37c4a238c1b0ec77e879ee754c481d4db825ed64381.jpg)

![ffd67928387a3ead5e1506ab7d1e3f008b8da45c1d4ff356c376c52e8fe525c6.jpg](../iclr_results/198_Energy-based%20Backdoor%20Defense%20Against%20Federated%20Graph%20Learning/tables/ffd67928387a3ead5e1506ab7d1e3f008b8da45c1d4ff356c376c52e8fe525c6.jpg)

## Prioritized Generative Replay


### Images

![1b340e6294f509ca3626e2a70ab3c780f052d499e347dbd978a5d3bec9f59d86.jpg](../iclr_results/199_Prioritized%20Generative%20Replay/images/1b340e6294f509ca3626e2a70ab3c780f052d499e347dbd978a5d3bec9f59d86.jpg)

![2615c6f9e73f7cf9938644bda62e4c45013f93fa970c9ff27f3bb46050e55a49.jpg](../iclr_results/199_Prioritized%20Generative%20Replay/images/2615c6f9e73f7cf9938644bda62e4c45013f93fa970c9ff27f3bb46050e55a49.jpg)

![5a7327157d3a62d15b92419f71157a50b2ca85b1adbcad280cb1e21351f9df6a.jpg](../iclr_results/199_Prioritized%20Generative%20Replay/images/5a7327157d3a62d15b92419f71157a50b2ca85b1adbcad280cb1e21351f9df6a.jpg)

![641f0794cc7fcfe3fd4a560ce108b47159ff7892947d6e66316ac1847de395e5.jpg](../iclr_results/199_Prioritized%20Generative%20Replay/images/641f0794cc7fcfe3fd4a560ce108b47159ff7892947d6e66316ac1847de395e5.jpg)

![64373d73783c60faa2bf25d8fd34ff4cc446f5802e6c7cff35d77c4e199e0e3e.jpg](../iclr_results/199_Prioritized%20Generative%20Replay/images/64373d73783c60faa2bf25d8fd34ff4cc446f5802e6c7cff35d77c4e199e0e3e.jpg)

![69574fa98cbf49ef8d547a7c328f55d50f5b44afb2ba00e39aa4aa6298c32916.jpg](../iclr_results/199_Prioritized%20Generative%20Replay/images/69574fa98cbf49ef8d547a7c328f55d50f5b44afb2ba00e39aa4aa6298c32916.jpg)

![aaff34f0c8e3fb17133b75ae45af7356837799cf1b6701e8efa79b05b7110f2c.jpg](../iclr_results/199_Prioritized%20Generative%20Replay/images/aaff34f0c8e3fb17133b75ae45af7356837799cf1b6701e8efa79b05b7110f2c.jpg)

![bac43a651208125ae07e36d9cdbc3e5ca9b261af40fea2793117c0d8318a0b1f.jpg](../iclr_results/199_Prioritized%20Generative%20Replay/images/bac43a651208125ae07e36d9cdbc3e5ca9b261af40fea2793117c0d8318a0b1f.jpg)

![c4956f5267aa3136573166c0be49dd0f702f94b58a2199bc12c889810843c013.jpg](../iclr_results/199_Prioritized%20Generative%20Replay/images/c4956f5267aa3136573166c0be49dd0f702f94b58a2199bc12c889810843c013.jpg)

### Tables

![1523091303972a4d2b79ebe9a7a0e332f5ea701e68ceee14906924ce154b341a.jpg](../iclr_results/199_Prioritized%20Generative%20Replay/tables/1523091303972a4d2b79ebe9a7a0e332f5ea701e68ceee14906924ce154b341a.jpg)

![4a80acbd5f05dfc40932425e3063038cb2b1992eafff9cdf54b53cdd8528274c.jpg](../iclr_results/199_Prioritized%20Generative%20Replay/tables/4a80acbd5f05dfc40932425e3063038cb2b1992eafff9cdf54b53cdd8528274c.jpg)

![62d330787b9d1e38c003f002fe9fb4b5c73a5d33cca56e50de99c82e4c759995.jpg](../iclr_results/199_Prioritized%20Generative%20Replay/tables/62d330787b9d1e38c003f002fe9fb4b5c73a5d33cca56e50de99c82e4c759995.jpg)

![67d717fb50c2e3766b2a8d91b9422373bec2fde60fe0f1cd30f32decadf31bb6.jpg](../iclr_results/199_Prioritized%20Generative%20Replay/tables/67d717fb50c2e3766b2a8d91b9422373bec2fde60fe0f1cd30f32decadf31bb6.jpg)

![68947766a1c519a0993d03b5cc79d65f3524d015478ac64430c01662af7dade1.jpg](../iclr_results/199_Prioritized%20Generative%20Replay/tables/68947766a1c519a0993d03b5cc79d65f3524d015478ac64430c01662af7dade1.jpg)

![7be92c4b01c8d543f7baa3107c8e62551fe40418e6043ab90dbcb1c661dabce1.jpg](../iclr_results/199_Prioritized%20Generative%20Replay/tables/7be92c4b01c8d543f7baa3107c8e62551fe40418e6043ab90dbcb1c661dabce1.jpg)

![957ba3239cd88292e6c451472eb773ef40f375bfc943471fdaa7f28263f2f297.jpg](../iclr_results/199_Prioritized%20Generative%20Replay/tables/957ba3239cd88292e6c451472eb773ef40f375bfc943471fdaa7f28263f2f297.jpg)

![b757e12e23b3e07600df835793277f12e9dce3222233a939dcbdcca46c34ad31.jpg](../iclr_results/199_Prioritized%20Generative%20Replay/tables/b757e12e23b3e07600df835793277f12e9dce3222233a939dcbdcca46c34ad31.jpg)

![efe6923a1db9a86a564f722a967f4210fe50d519f58a5b59eed0a60be7b69d46.jpg](../iclr_results/199_Prioritized%20Generative%20Replay/tables/efe6923a1db9a86a564f722a967f4210fe50d519f58a5b59eed0a60be7b69d46.jpg)

![fbd30bbe52b4970d0a0d46a9e9f80c0fa8da81cf5ecee31b3b47d5ca94d556d4.jpg](../iclr_results/199_Prioritized%20Generative%20Replay/tables/fbd30bbe52b4970d0a0d46a9e9f80c0fa8da81cf5ecee31b3b47d5ca94d556d4.jpg)

## A Probabilistic Perspective on Unlearning and Alignment for Large Language Models


### Images

![192430986084ea7db3552d7d41567eddba61458390f60824ccc05d55c4ff768d.jpg](../iclr_results/200_A%20Probabilistic%20Perspective%20on%20Unlearning%20and%20Alignment%20for%20Large%20Language%20Models/images/192430986084ea7db3552d7d41567eddba61458390f60824ccc05d55c4ff768d.jpg)

![1b8ac8955049aed636ec299e368d2eb18a86bfeecf0b9a79c1af784968dea012.jpg](../iclr_results/200_A%20Probabilistic%20Perspective%20on%20Unlearning%20and%20Alignment%20for%20Large%20Language%20Models/images/1b8ac8955049aed636ec299e368d2eb18a86bfeecf0b9a79c1af784968dea012.jpg)

![4c255b78e1bfecc651003bde44261a8be7f2f2f59081acd63bc4212ab627ab02.jpg](../iclr_results/200_A%20Probabilistic%20Perspective%20on%20Unlearning%20and%20Alignment%20for%20Large%20Language%20Models/images/4c255b78e1bfecc651003bde44261a8be7f2f2f59081acd63bc4212ab627ab02.jpg)

![828c6ea84ff0cd46ec783575ace930fb303906d0d093ce547d5eb09b0ec5c543.jpg](../iclr_results/200_A%20Probabilistic%20Perspective%20on%20Unlearning%20and%20Alignment%20for%20Large%20Language%20Models/images/828c6ea84ff0cd46ec783575ace930fb303906d0d093ce547d5eb09b0ec5c543.jpg)

![908851f01b16affc8175e5fb150f350789c93778ce9726c87d2f6aa9271c22c8.jpg](../iclr_results/200_A%20Probabilistic%20Perspective%20on%20Unlearning%20and%20Alignment%20for%20Large%20Language%20Models/images/908851f01b16affc8175e5fb150f350789c93778ce9726c87d2f6aa9271c22c8.jpg)

![cfb485484c30ea9a5d04fbbd85549ba6dd0460e613e123c3a451f604d02e1e58.jpg](../iclr_results/200_A%20Probabilistic%20Perspective%20on%20Unlearning%20and%20Alignment%20for%20Large%20Language%20Models/images/cfb485484c30ea9a5d04fbbd85549ba6dd0460e613e123c3a451f604d02e1e58.jpg)

### Tables

![1596a9283e4968da82316b9fb9bd50094111cb305de768b639d6a88ce026c01d.jpg](../iclr_results/200_A%20Probabilistic%20Perspective%20on%20Unlearning%20and%20Alignment%20for%20Large%20Language%20Models/tables/1596a9283e4968da82316b9fb9bd50094111cb305de768b639d6a88ce026c01d.jpg)

![2afc992f41ea8371815bbde42ffefeef01043f1608d8315253c8b95d9f4b2e2a.jpg](../iclr_results/200_A%20Probabilistic%20Perspective%20on%20Unlearning%20and%20Alignment%20for%20Large%20Language%20Models/tables/2afc992f41ea8371815bbde42ffefeef01043f1608d8315253c8b95d9f4b2e2a.jpg)

![96d0d0bc668ec631f967d8d7476099f8210185ba1e7d1643b91f8c0cdfd0c8a7.jpg](../iclr_results/200_A%20Probabilistic%20Perspective%20on%20Unlearning%20and%20Alignment%20for%20Large%20Language%20Models/tables/96d0d0bc668ec631f967d8d7476099f8210185ba1e7d1643b91f8c0cdfd0c8a7.jpg)

![9eee52d4d62bc12e387d4ce1ed59388186cca978cd22e8e5df4eac50f139d876.jpg](../iclr_results/200_A%20Probabilistic%20Perspective%20on%20Unlearning%20and%20Alignment%20for%20Large%20Language%20Models/tables/9eee52d4d62bc12e387d4ce1ed59388186cca978cd22e8e5df4eac50f139d876.jpg)

## Exploring The Loss Landscape Of Regularized Neural Networks Via Convex Duality


### Images

![072f2627c9e050d0856be89659c5a41bdbdbc4519e73eed3fbcfa2af72c039ab.jpg](../iclr_results/201_Exploring%20The%20Loss%20Landscape%20Of%20Regularized%20Neural%20Networks%20Via%20Convex%20Duality/images/072f2627c9e050d0856be89659c5a41bdbdbc4519e73eed3fbcfa2af72c039ab.jpg)

![796ff7554b2dbe1d04490cb229410eccbd3a3cf52270f1fab01d68e0d1b1426d.jpg](../iclr_results/201_Exploring%20The%20Loss%20Landscape%20Of%20Regularized%20Neural%20Networks%20Via%20Convex%20Duality/images/796ff7554b2dbe1d04490cb229410eccbd3a3cf52270f1fab01d68e0d1b1426d.jpg)

![8bce9dbd2f5739ebc5c8ac4a7a02e12302aa8ffaae90026c17170d0e4bb31a3e.jpg](../iclr_results/201_Exploring%20The%20Loss%20Landscape%20Of%20Regularized%20Neural%20Networks%20Via%20Convex%20Duality/images/8bce9dbd2f5739ebc5c8ac4a7a02e12302aa8ffaae90026c17170d0e4bb31a3e.jpg)

![ac91a791e6c8540e42f76fd80dde02a41489a7b15c98f28667ec7dc0334275fb.jpg](../iclr_results/201_Exploring%20The%20Loss%20Landscape%20Of%20Regularized%20Neural%20Networks%20Via%20Convex%20Duality/images/ac91a791e6c8540e42f76fd80dde02a41489a7b15c98f28667ec7dc0334275fb.jpg)

![cf10e4c4d0011a00cadae8482d317f0797486c939a726ad201ddedef4b58cd23.jpg](../iclr_results/201_Exploring%20The%20Loss%20Landscape%20Of%20Regularized%20Neural%20Networks%20Via%20Convex%20Duality/images/cf10e4c4d0011a00cadae8482d317f0797486c939a726ad201ddedef4b58cd23.jpg)

![cfafd84994a3e9f3714c11ab8b5834e53dc5be7857adc6c2b889d026cdc5fe38.jpg](../iclr_results/201_Exploring%20The%20Loss%20Landscape%20Of%20Regularized%20Neural%20Networks%20Via%20Convex%20Duality/images/cfafd84994a3e9f3714c11ab8b5834e53dc5be7857adc6c2b889d026cdc5fe38.jpg)

![daf28e3a02ade72dc8e4aeedabea593d989aaf4e3351ee3f109dd075651c4151.jpg](../iclr_results/201_Exploring%20The%20Loss%20Landscape%20Of%20Regularized%20Neural%20Networks%20Via%20Convex%20Duality/images/daf28e3a02ade72dc8e4aeedabea593d989aaf4e3351ee3f109dd075651c4151.jpg)

![eadb26714b72f3ad86a3b8ebf3ab9b018c589423bac5083034b6f9fe4729bc02.jpg](../iclr_results/201_Exploring%20The%20Loss%20Landscape%20Of%20Regularized%20Neural%20Networks%20Via%20Convex%20Duality/images/eadb26714b72f3ad86a3b8ebf3ab9b018c589423bac5083034b6f9fe4729bc02.jpg)

![f682755451cc3005df5626310f19a2db5190e6e9e81565057f2dc688dc926bb5.jpg](../iclr_results/201_Exploring%20The%20Loss%20Landscape%20Of%20Regularized%20Neural%20Networks%20Via%20Convex%20Duality/images/f682755451cc3005df5626310f19a2db5190e6e9e81565057f2dc688dc926bb5.jpg)

## Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning


### Images

![055e8562312449c395de7995d925f3b3d8f45d739788d2aba83fb463ac5c54f1.jpg](../iclr_results/202_Flat%20Reward%20in%20Policy%20Parameter%20Space%20Implies%20Robust%20Reinforcement%20Learning/images/055e8562312449c395de7995d925f3b3d8f45d739788d2aba83fb463ac5c54f1.jpg)

![0ff3c3d220a7d383c64adcb7a7b8f863cb2a0f662a1dbb84e25e12ed87c35a04.jpg](../iclr_results/202_Flat%20Reward%20in%20Policy%20Parameter%20Space%20Implies%20Robust%20Reinforcement%20Learning/images/0ff3c3d220a7d383c64adcb7a7b8f863cb2a0f662a1dbb84e25e12ed87c35a04.jpg)

![2936218e52672cb65e208316c733f5a85ddfb226ce71ec96b7200db19a0001c7.jpg](../iclr_results/202_Flat%20Reward%20in%20Policy%20Parameter%20Space%20Implies%20Robust%20Reinforcement%20Learning/images/2936218e52672cb65e208316c733f5a85ddfb226ce71ec96b7200db19a0001c7.jpg)

![88a6353d883eee296f58d73b5809f16f28485c47a02e20de0631121c34eec760.jpg](../iclr_results/202_Flat%20Reward%20in%20Policy%20Parameter%20Space%20Implies%20Robust%20Reinforcement%20Learning/images/88a6353d883eee296f58d73b5809f16f28485c47a02e20de0631121c34eec760.jpg)

![a5f62906ac0896b07980f4445b682b8de29a4ab2e88670edaaf48d196560fbd6.jpg](../iclr_results/202_Flat%20Reward%20in%20Policy%20Parameter%20Space%20Implies%20Robust%20Reinforcement%20Learning/images/a5f62906ac0896b07980f4445b682b8de29a4ab2e88670edaaf48d196560fbd6.jpg)

![c5c5b5aa91507553e2ddab585226e53573f0ce920afdc86ee089f10ccf94361b.jpg](../iclr_results/202_Flat%20Reward%20in%20Policy%20Parameter%20Space%20Implies%20Robust%20Reinforcement%20Learning/images/c5c5b5aa91507553e2ddab585226e53573f0ce920afdc86ee089f10ccf94361b.jpg)

![d73f40371da2a31c603011f0601d439226758f9225b2ccad2787813caa1e50b5.jpg](../iclr_results/202_Flat%20Reward%20in%20Policy%20Parameter%20Space%20Implies%20Robust%20Reinforcement%20Learning/images/d73f40371da2a31c603011f0601d439226758f9225b2ccad2787813caa1e50b5.jpg)

![ebfdae388f87bd9519fe22b656d4e7e38bd8114a0859243f79a0442ea2d0dced.jpg](../iclr_results/202_Flat%20Reward%20in%20Policy%20Parameter%20Space%20Implies%20Robust%20Reinforcement%20Learning/images/ebfdae388f87bd9519fe22b656d4e7e38bd8114a0859243f79a0442ea2d0dced.jpg)

### Tables

![2ed0b435b705f9d51927c7e75e403b7cb9494c7ac807bc3ba381a81389fd2b8b.jpg](../iclr_results/202_Flat%20Reward%20in%20Policy%20Parameter%20Space%20Implies%20Robust%20Reinforcement%20Learning/tables/2ed0b435b705f9d51927c7e75e403b7cb9494c7ac807bc3ba381a81389fd2b8b.jpg)

![327b72c76e01304968f716e403565b8b1284121e79bcc9d7e0e9157570b324c7.jpg](../iclr_results/202_Flat%20Reward%20in%20Policy%20Parameter%20Space%20Implies%20Robust%20Reinforcement%20Learning/tables/327b72c76e01304968f716e403565b8b1284121e79bcc9d7e0e9157570b324c7.jpg)

![400c4858144248356920939f807f4976f4f38a3599551435335077fe00233f99.jpg](../iclr_results/202_Flat%20Reward%20in%20Policy%20Parameter%20Space%20Implies%20Robust%20Reinforcement%20Learning/tables/400c4858144248356920939f807f4976f4f38a3599551435335077fe00233f99.jpg)

![424e7e39315aabc72009c5d6834875cfcf21e0389abc3f6117ed5e7f4e8a03c5.jpg](../iclr_results/202_Flat%20Reward%20in%20Policy%20Parameter%20Space%20Implies%20Robust%20Reinforcement%20Learning/tables/424e7e39315aabc72009c5d6834875cfcf21e0389abc3f6117ed5e7f4e8a03c5.jpg)

![4da9c100c106db23c9855d9ee71707379b6b4cd2741f3bf3b24eed38159a9298.jpg](../iclr_results/202_Flat%20Reward%20in%20Policy%20Parameter%20Space%20Implies%20Robust%20Reinforcement%20Learning/tables/4da9c100c106db23c9855d9ee71707379b6b4cd2741f3bf3b24eed38159a9298.jpg)

![651f23622ff00a06c95702a81db43af0eda77fc27cf4975c45cc942ad7334fa0.jpg](../iclr_results/202_Flat%20Reward%20in%20Policy%20Parameter%20Space%20Implies%20Robust%20Reinforcement%20Learning/tables/651f23622ff00a06c95702a81db43af0eda77fc27cf4975c45cc942ad7334fa0.jpg)

![c063759aa091cbe180fcebbb03b648a06055b2dc00a506e6955e832c86477ba6.jpg](../iclr_results/202_Flat%20Reward%20in%20Policy%20Parameter%20Space%20Implies%20Robust%20Reinforcement%20Learning/tables/c063759aa091cbe180fcebbb03b648a06055b2dc00a506e6955e832c86477ba6.jpg)

![f1edceb239c0e6d4ab1986330b4c19af912138ac945de8cd581ecd43a5d66c8a.jpg](../iclr_results/202_Flat%20Reward%20in%20Policy%20Parameter%20Space%20Implies%20Robust%20Reinforcement%20Learning/tables/f1edceb239c0e6d4ab1986330b4c19af912138ac945de8cd581ecd43a5d66c8a.jpg)

![fa8c53b41386ceab35ca7c24493c23e4781284ee3ad9acc20374f9ca3f56d5f9.jpg](../iclr_results/202_Flat%20Reward%20in%20Policy%20Parameter%20Space%20Implies%20Robust%20Reinforcement%20Learning/tables/fa8c53b41386ceab35ca7c24493c23e4781284ee3ad9acc20374f9ca3f56d5f9.jpg)

## Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning


### Images

![0a768b5d305402cd78b4100512dc921ee148c3f1cc28245262b961d87f11bb1f.jpg](../iclr_results/203_Scaling%20LLM%20Test-Time%20Compute%20Optimally%20Can%20be%20More%20Effective%20than%20Scaling%20Parameters%20for%20Reasoning/images/0a768b5d305402cd78b4100512dc921ee148c3f1cc28245262b961d87f11bb1f.jpg)

![2c8a1bbdd3e0d41b9c7d680360cbf52f84ae07777a407a3e8937ce4edbaaaa57.jpg](../iclr_results/203_Scaling%20LLM%20Test-Time%20Compute%20Optimally%20Can%20be%20More%20Effective%20than%20Scaling%20Parameters%20for%20Reasoning/images/2c8a1bbdd3e0d41b9c7d680360cbf52f84ae07777a407a3e8937ce4edbaaaa57.jpg)

![2d493cac03b7cfd4d279d19c0e63f74d1355e5fcd0cd322ec8040c5bdcbdeb45.jpg](../iclr_results/203_Scaling%20LLM%20Test-Time%20Compute%20Optimally%20Can%20be%20More%20Effective%20than%20Scaling%20Parameters%20for%20Reasoning/images/2d493cac03b7cfd4d279d19c0e63f74d1355e5fcd0cd322ec8040c5bdcbdeb45.jpg)

![2f478b201380511406db1e290f9daa834910e7c0db36e820251ab17889672eda.jpg](../iclr_results/203_Scaling%20LLM%20Test-Time%20Compute%20Optimally%20Can%20be%20More%20Effective%20than%20Scaling%20Parameters%20for%20Reasoning/images/2f478b201380511406db1e290f9daa834910e7c0db36e820251ab17889672eda.jpg)

![3fd789dfcc16abb5c802f64be9672a3eb6883063d56a924a30a1c17bcc095c13.jpg](../iclr_results/203_Scaling%20LLM%20Test-Time%20Compute%20Optimally%20Can%20be%20More%20Effective%20than%20Scaling%20Parameters%20for%20Reasoning/images/3fd789dfcc16abb5c802f64be9672a3eb6883063d56a924a30a1c17bcc095c13.jpg)

![6bdbc9bd59a76f341ef578ec9893693d8ef987cfc9b29900811155a10d0ad7a0.jpg](../iclr_results/203_Scaling%20LLM%20Test-Time%20Compute%20Optimally%20Can%20be%20More%20Effective%20than%20Scaling%20Parameters%20for%20Reasoning/images/6bdbc9bd59a76f341ef578ec9893693d8ef987cfc9b29900811155a10d0ad7a0.jpg)

![704077b59a86e6d54a694be35229102ccd59c12ec03ffdba6e00cc5d6c4b365d.jpg](../iclr_results/203_Scaling%20LLM%20Test-Time%20Compute%20Optimally%20Can%20be%20More%20Effective%20than%20Scaling%20Parameters%20for%20Reasoning/images/704077b59a86e6d54a694be35229102ccd59c12ec03ffdba6e00cc5d6c4b365d.jpg)

![70da7fb580312a07e751a7b3d1bc68685deb1a79229578920f54a5a258c6d2e4.jpg](../iclr_results/203_Scaling%20LLM%20Test-Time%20Compute%20Optimally%20Can%20be%20More%20Effective%20than%20Scaling%20Parameters%20for%20Reasoning/images/70da7fb580312a07e751a7b3d1bc68685deb1a79229578920f54a5a258c6d2e4.jpg)

![78bf3dc339f916e3adea7e29ec0601e33c832646dfe20067c455ba8110c844de.jpg](../iclr_results/203_Scaling%20LLM%20Test-Time%20Compute%20Optimally%20Can%20be%20More%20Effective%20than%20Scaling%20Parameters%20for%20Reasoning/images/78bf3dc339f916e3adea7e29ec0601e33c832646dfe20067c455ba8110c844de.jpg)

![8bd82e388d1095913e8a82c4e2c58213a2567c70d042e20ca2ebc20a5da83a9a.jpg](../iclr_results/203_Scaling%20LLM%20Test-Time%20Compute%20Optimally%20Can%20be%20More%20Effective%20than%20Scaling%20Parameters%20for%20Reasoning/images/8bd82e388d1095913e8a82c4e2c58213a2567c70d042e20ca2ebc20a5da83a9a.jpg)

![978430283ac2e71a27421726954b1ad34c220479111865d91dabbd2e197eebe2.jpg](../iclr_results/203_Scaling%20LLM%20Test-Time%20Compute%20Optimally%20Can%20be%20More%20Effective%20than%20Scaling%20Parameters%20for%20Reasoning/images/978430283ac2e71a27421726954b1ad34c220479111865d91dabbd2e197eebe2.jpg)

![aa77a167543f5647cd291b9e571ef32dd254b199b31387509ef7681b56617fce.jpg](../iclr_results/203_Scaling%20LLM%20Test-Time%20Compute%20Optimally%20Can%20be%20More%20Effective%20than%20Scaling%20Parameters%20for%20Reasoning/images/aa77a167543f5647cd291b9e571ef32dd254b199b31387509ef7681b56617fce.jpg)

![ab4e947e3bdacaf40d432a4268ce4c79f53ffbd8cdc9a8e879a9e5b8349082d4.jpg](../iclr_results/203_Scaling%20LLM%20Test-Time%20Compute%20Optimally%20Can%20be%20More%20Effective%20than%20Scaling%20Parameters%20for%20Reasoning/images/ab4e947e3bdacaf40d432a4268ce4c79f53ffbd8cdc9a8e879a9e5b8349082d4.jpg)

![b30ed5ad7334baa295ee526d76405ddd07273d33eab9fe382ea540957dc31f63.jpg](../iclr_results/203_Scaling%20LLM%20Test-Time%20Compute%20Optimally%20Can%20be%20More%20Effective%20than%20Scaling%20Parameters%20for%20Reasoning/images/b30ed5ad7334baa295ee526d76405ddd07273d33eab9fe382ea540957dc31f63.jpg)

![b529f303b7e1b060a27d68d8740e690cec27cc4869979d65dc684b9250abe75a.jpg](../iclr_results/203_Scaling%20LLM%20Test-Time%20Compute%20Optimally%20Can%20be%20More%20Effective%20than%20Scaling%20Parameters%20for%20Reasoning/images/b529f303b7e1b060a27d68d8740e690cec27cc4869979d65dc684b9250abe75a.jpg)

![bfeb6dce70eb51c2d7d4de1f1502beac3ddbfd699e3464f02af66b516ec7d5d5.jpg](../iclr_results/203_Scaling%20LLM%20Test-Time%20Compute%20Optimally%20Can%20be%20More%20Effective%20than%20Scaling%20Parameters%20for%20Reasoning/images/bfeb6dce70eb51c2d7d4de1f1502beac3ddbfd699e3464f02af66b516ec7d5d5.jpg)

![d0fa1d08145e596f9e71f63ebb39959b3d4d35bd140fabc75832c923d84b1f90.jpg](../iclr_results/203_Scaling%20LLM%20Test-Time%20Compute%20Optimally%20Can%20be%20More%20Effective%20than%20Scaling%20Parameters%20for%20Reasoning/images/d0fa1d08145e596f9e71f63ebb39959b3d4d35bd140fabc75832c923d84b1f90.jpg)

![df6c2584719204fb865211911db752dde8e36b5958b0a486cee9cb26563e7c0b.jpg](../iclr_results/203_Scaling%20LLM%20Test-Time%20Compute%20Optimally%20Can%20be%20More%20Effective%20than%20Scaling%20Parameters%20for%20Reasoning/images/df6c2584719204fb865211911db752dde8e36b5958b0a486cee9cb26563e7c0b.jpg)

![f11847882f3ec4ca1e1a65ee88b626b98870c3ef064778dd4fc1c5d68ac619c7.jpg](../iclr_results/203_Scaling%20LLM%20Test-Time%20Compute%20Optimally%20Can%20be%20More%20Effective%20than%20Scaling%20Parameters%20for%20Reasoning/images/f11847882f3ec4ca1e1a65ee88b626b98870c3ef064778dd4fc1c5d68ac619c7.jpg)

### Tables

![2220792cac31664e55a8f3a5f25f1aef33c0afa6f6659422f3aefd7e3fe7fc2a.jpg](../iclr_results/203_Scaling%20LLM%20Test-Time%20Compute%20Optimally%20Can%20be%20More%20Effective%20than%20Scaling%20Parameters%20for%20Reasoning/tables/2220792cac31664e55a8f3a5f25f1aef33c0afa6f6659422f3aefd7e3fe7fc2a.jpg)

## Compositional Entailment Learning for Hyperbolic Vision-Language Models


### Images

![12e8c8f43424f5827ef5a216a6417b6e01d0a234350bc7dd1a29532ed38d45f8.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/images/12e8c8f43424f5827ef5a216a6417b6e01d0a234350bc7dd1a29532ed38d45f8.jpg)

![2439f8023db7b99b91e11d3115e185e3bd284e17849b3f2bb16a0f49cf5f2b7f.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/images/2439f8023db7b99b91e11d3115e185e3bd284e17849b3f2bb16a0f49cf5f2b7f.jpg)

![44c41f760aac421c8c87aa9474e3bf5c9229e3a7e1f95cde1c507040e5e09958.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/images/44c41f760aac421c8c87aa9474e3bf5c9229e3a7e1f95cde1c507040e5e09958.jpg)

![4a26ff83fa5446ee08e82184b25d4e95bff692633bfdf4901b69b3892e5ff8fa.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/images/4a26ff83fa5446ee08e82184b25d4e95bff692633bfdf4901b69b3892e5ff8fa.jpg)

![7673681916a1226005df5e305c2cb18a98f8fff422199cea687c2b1b8111da73.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/images/7673681916a1226005df5e305c2cb18a98f8fff422199cea687c2b1b8111da73.jpg)

![860b55476b7769702719ddff30ce2bbac977efa0af4869cf12996a7d37dd765e.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/images/860b55476b7769702719ddff30ce2bbac977efa0af4869cf12996a7d37dd765e.jpg)

![863b3eb59a065a7dd2a90d6ca5ae2df733d264b7775c4864b903bc49d2955c31.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/images/863b3eb59a065a7dd2a90d6ca5ae2df733d264b7775c4864b903bc49d2955c31.jpg)

![8887fe74afe111c18e6ad0393c5ffe8eba48cdc2cd34f17e0676b212479f5950.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/images/8887fe74afe111c18e6ad0393c5ffe8eba48cdc2cd34f17e0676b212479f5950.jpg)

![9c0230e0d242b4e2588c4d95ae3e8eb619826d361b85dacb8e3e25f499ff89a5.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/images/9c0230e0d242b4e2588c4d95ae3e8eb619826d361b85dacb8e3e25f499ff89a5.jpg)

![ab2e803ecc849bff8811b7ca468cf56a1d1edfb54d11681500c04c472fe171f8.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/images/ab2e803ecc849bff8811b7ca468cf56a1d1edfb54d11681500c04c472fe171f8.jpg)

![b02d231207f047836451772184d83d643f54faa02886d8fc078706b83462af79.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/images/b02d231207f047836451772184d83d643f54faa02886d8fc078706b83462af79.jpg)

![b1a811acd3cae9c2d306896e63dada506d755203eb9dd0d1393ece186eb32301.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/images/b1a811acd3cae9c2d306896e63dada506d755203eb9dd0d1393ece186eb32301.jpg)

![c46a94ea23ff443cd042a7f82be31019d6f5875a48a213119f9002bca1f35208.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/images/c46a94ea23ff443cd042a7f82be31019d6f5875a48a213119f9002bca1f35208.jpg)

![f5008280b67dd3d440d4b6ced60b0f87f96a9817cb91d98c9042fbeea968d966.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/images/f5008280b67dd3d440d4b6ced60b0f87f96a9817cb91d98c9042fbeea968d966.jpg)

![f7d3bf160e0e2e82bb5fb3de7f7bbda7fd663880654126320c3410ee6b2ec435.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/images/f7d3bf160e0e2e82bb5fb3de7f7bbda7fd663880654126320c3410ee6b2ec435.jpg)

![feb75e79fc1134e3e9649f614ccc30546afcc93250d2c3081559f9e172a936ae.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/images/feb75e79fc1134e3e9649f614ccc30546afcc93250d2c3081559f9e172a936ae.jpg)

### Tables

![12bdc1e5606c49b402701916ac28965bb54951790c3dab099e4252dd90c3765c.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/tables/12bdc1e5606c49b402701916ac28965bb54951790c3dab099e4252dd90c3765c.jpg)

![2542eb1220ae6b58e154bcb4ab358b16e1f033c0068d30613ce47902e711fdad.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/tables/2542eb1220ae6b58e154bcb4ab358b16e1f033c0068d30613ce47902e711fdad.jpg)

![39c20595817f382ce4cd65cbcffb7cf99d524387e7514dc1a2328bc249af4007.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/tables/39c20595817f382ce4cd65cbcffb7cf99d524387e7514dc1a2328bc249af4007.jpg)

![4cd365e133f290bf979a92441dbceca91b8bf328396d861b654ce32e3fe7ad83.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/tables/4cd365e133f290bf979a92441dbceca91b8bf328396d861b654ce32e3fe7ad83.jpg)

![7f3abd855646d2535ca6b818676d9fecdb24c5ff52405adabd88f8b26f3cb358.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/tables/7f3abd855646d2535ca6b818676d9fecdb24c5ff52405adabd88f8b26f3cb358.jpg)

![890dc03b00a6290703acc3763eda3645f74758d2fdb67942ebfefcdebe95bb01.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/tables/890dc03b00a6290703acc3763eda3645f74758d2fdb67942ebfefcdebe95bb01.jpg)

![8e6cf3653ca0feacb155a5fa059366ed1cbbe8769f799b9d7a3094d87698c13f.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/tables/8e6cf3653ca0feacb155a5fa059366ed1cbbe8769f799b9d7a3094d87698c13f.jpg)

![bbf08523582b6a0d0199a086823287c73b2a34aa6b413d4a94653fad3826d309.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/tables/bbf08523582b6a0d0199a086823287c73b2a34aa6b413d4a94653fad3826d309.jpg)

![c2c8c03a92290bcd72beb78b5880dcce93160f2fe2515bfd2bee8c0b738f937f.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/tables/c2c8c03a92290bcd72beb78b5880dcce93160f2fe2515bfd2bee8c0b738f937f.jpg)

![f0e09152c141ad191d6152ef98fb30482f5e4934818918ae5869121e3c3aca2f.jpg](../iclr_results/204_Compositional%20Entailment%20Learning%20for%20Hyperbolic%20Vision-Language%20Models/tables/f0e09152c141ad191d6152ef98fb30482f5e4934818918ae5869121e3c3aca2f.jpg)

## OptionZero: Planning with Learned Options


### Images

![0168c87ac4bc5256d609f1b5d50316b52dc73147fb3c9ec7e3d30e4853bfb49c.jpg](../iclr_results/205_OptionZero_%20Planning%20with%20Learned%20Options/images/0168c87ac4bc5256d609f1b5d50316b52dc73147fb3c9ec7e3d30e4853bfb49c.jpg)

![07a786816c93945a8bd6a4d00ac123c4a72031f28aa1f03f1293076d92616ac7.jpg](../iclr_results/205_OptionZero_%20Planning%20with%20Learned%20Options/images/07a786816c93945a8bd6a4d00ac123c4a72031f28aa1f03f1293076d92616ac7.jpg)

![30ee9877722e9a98af1884bd36115a357deadf176fa5772eead76e530c5b97a8.jpg](../iclr_results/205_OptionZero_%20Planning%20with%20Learned%20Options/images/30ee9877722e9a98af1884bd36115a357deadf176fa5772eead76e530c5b97a8.jpg)

![7e9f0434c4eb12724e7245f4f3fcd4d5b19b8d7789e3adb29549c2d782b407c7.jpg](../iclr_results/205_OptionZero_%20Planning%20with%20Learned%20Options/images/7e9f0434c4eb12724e7245f4f3fcd4d5b19b8d7789e3adb29549c2d782b407c7.jpg)

![944cea104a5ebe4f8c78a5a84ca65a37ce4c9be6d26ab61fba5051ac9c0a759e.jpg](../iclr_results/205_OptionZero_%20Planning%20with%20Learned%20Options/images/944cea104a5ebe4f8c78a5a84ca65a37ce4c9be6d26ab61fba5051ac9c0a759e.jpg)

![ae1a8486e05d53c75e78dd43fbd295a62c17577e223eb8be460d4c61f0af7fb8.jpg](../iclr_results/205_OptionZero_%20Planning%20with%20Learned%20Options/images/ae1a8486e05d53c75e78dd43fbd295a62c17577e223eb8be460d4c61f0af7fb8.jpg)

![c56dc929f22945de402365386c527fb17a16a716463c9e4fda58c7c80a484986.jpg](../iclr_results/205_OptionZero_%20Planning%20with%20Learned%20Options/images/c56dc929f22945de402365386c527fb17a16a716463c9e4fda58c7c80a484986.jpg)

### Tables

![0e525e14c2fc7785e5ae039bc7293078ce5fafff03abe191ddac1142f69aa708.jpg](../iclr_results/205_OptionZero_%20Planning%20with%20Learned%20Options/tables/0e525e14c2fc7785e5ae039bc7293078ce5fafff03abe191ddac1142f69aa708.jpg)

![350aeecfcdb30cc2692d01c814c2fc8d8bc7da1c1e973786472743696857054e.jpg](../iclr_results/205_OptionZero_%20Planning%20with%20Learned%20Options/tables/350aeecfcdb30cc2692d01c814c2fc8d8bc7da1c1e973786472743696857054e.jpg)

![37a32cbdeb6c65ad79698ffd0c2f726c6a70ae3fabd4bb1012f05539e1c093b3.jpg](../iclr_results/205_OptionZero_%20Planning%20with%20Learned%20Options/tables/37a32cbdeb6c65ad79698ffd0c2f726c6a70ae3fabd4bb1012f05539e1c093b3.jpg)

![4217c7883415e4dce7114bd76bf9d5795b9edc840bbf087dbe2a924391268856.jpg](../iclr_results/205_OptionZero_%20Planning%20with%20Learned%20Options/tables/4217c7883415e4dce7114bd76bf9d5795b9edc840bbf087dbe2a924391268856.jpg)

![653650cad186f95b783217dd5841aa5207a1d8f828a2ef042d8ee2228cf39d7f.jpg](../iclr_results/205_OptionZero_%20Planning%20with%20Learned%20Options/tables/653650cad186f95b783217dd5841aa5207a1d8f828a2ef042d8ee2228cf39d7f.jpg)

![6add674fd6e67976904134cf05d97fc601433d8bcb0f5f871a94b0865e33d017.jpg](../iclr_results/205_OptionZero_%20Planning%20with%20Learned%20Options/tables/6add674fd6e67976904134cf05d97fc601433d8bcb0f5f871a94b0865e33d017.jpg)

![77bcdbbee3a82fea0c124ee836873c5d32178661fe30609bc9f3ee45557a3078.jpg](../iclr_results/205_OptionZero_%20Planning%20with%20Learned%20Options/tables/77bcdbbee3a82fea0c124ee836873c5d32178661fe30609bc9f3ee45557a3078.jpg)

![77d7622a94cbc9b22302c575d65807b9522c603b822bdc4e77b772468a160dd2.jpg](../iclr_results/205_OptionZero_%20Planning%20with%20Learned%20Options/tables/77d7622a94cbc9b22302c575d65807b9522c603b822bdc4e77b772468a160dd2.jpg)

![77de0bd8fe27bf332ab2ab35c0df03d935ae135691ac6ac47a0d1799dd8697e7.jpg](../iclr_results/205_OptionZero_%20Planning%20with%20Learned%20Options/tables/77de0bd8fe27bf332ab2ab35c0df03d935ae135691ac6ac47a0d1799dd8697e7.jpg)

![8561357c213dbd2ad7ac744bb87a65ec4ee23f8232b0017fe1af5089a480b23f.jpg](../iclr_results/205_OptionZero_%20Planning%20with%20Learned%20Options/tables/8561357c213dbd2ad7ac744bb87a65ec4ee23f8232b0017fe1af5089a480b23f.jpg)

![ccc42e7eb141ecaa502de7c5d055bf8e904dacd028c8fc7fadd58e005fa6dff9.jpg](../iclr_results/205_OptionZero_%20Planning%20with%20Learned%20Options/tables/ccc42e7eb141ecaa502de7c5d055bf8e904dacd028c8fc7fadd58e005fa6dff9.jpg)

![d3a81d427ff1e8360dfee68ef1269a2121cca190973d45e2945718a12f5b8c06.jpg](../iclr_results/205_OptionZero_%20Planning%20with%20Learned%20Options/tables/d3a81d427ff1e8360dfee68ef1269a2121cca190973d45e2945718a12f5b8c06.jpg)

![ddbd1c7a6882089f25bce5e7c7aadd23095b8f7c9fb4d3f059357d3d3a1374e5.jpg](../iclr_results/205_OptionZero_%20Planning%20with%20Learned%20Options/tables/ddbd1c7a6882089f25bce5e7c7aadd23095b8f7c9fb4d3f059357d3d3a1374e5.jpg)

![f0fad7a4d70c36e31a26a39b867b89453c8bfab8249d7521f8e0d1b7854ef006.jpg](../iclr_results/205_OptionZero_%20Planning%20with%20Learned%20Options/tables/f0fad7a4d70c36e31a26a39b867b89453c8bfab8249d7521f8e0d1b7854ef006.jpg)

## On the Identification of Temporal Causal Representation with Instantaneous Dependence


### Images

![0b5295ab544b5255a2d8a862d0760cbce9c716193a9796a9756750680d325233.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/images/0b5295ab544b5255a2d8a862d0760cbce9c716193a9796a9756750680d325233.jpg)

![26ebd7cce794e30ea498cbc0161f80cc9ece82ce1bd796b700afad64c5e38a5d.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/images/26ebd7cce794e30ea498cbc0161f80cc9ece82ce1bd796b700afad64c5e38a5d.jpg)

![4f2d375cceb9f1685d311c0209b45c13e869eeac5841fc48deb44b0565e16245.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/images/4f2d375cceb9f1685d311c0209b45c13e869eeac5841fc48deb44b0565e16245.jpg)

![56f1b04deffd609f808fe18112bb22575fbce9c4611ad6bd6ecd76a6edb97b1c.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/images/56f1b04deffd609f808fe18112bb22575fbce9c4611ad6bd6ecd76a6edb97b1c.jpg)

![a494fdc1b2674257d8eb42fb509a4df2c6a9d816477f4f99bdaf01c0cc6906c7.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/images/a494fdc1b2674257d8eb42fb509a4df2c6a9d816477f4f99bdaf01c0cc6906c7.jpg)

![b84550ef9488faea6130e552e335949d6e78a22cd2fa09de6449139ca7d56e29.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/images/b84550ef9488faea6130e552e335949d6e78a22cd2fa09de6449139ca7d56e29.jpg)

![d770901fdc43936b24898a540adee027c16b7bc922097dcd86a74fdd730c118a.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/images/d770901fdc43936b24898a540adee027c16b7bc922097dcd86a74fdd730c118a.jpg)

![e0dbb2da51b9f411c2d04285ef27793dbd718bc088e4c996300542f91117348d.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/images/e0dbb2da51b9f411c2d04285ef27793dbd718bc088e4c996300542f91117348d.jpg)

![ea761bcfe92f4051866bf539f621dfa0898126831f20d83cf37d4fff7b5728b9.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/images/ea761bcfe92f4051866bf539f621dfa0898126831f20d83cf37d4fff7b5728b9.jpg)

![f944f72105bfc8a1cbad6a87bb56a50801dc95fb0b6a6b7a0507984787e28350.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/images/f944f72105bfc8a1cbad6a87bb56a50801dc95fb0b6a6b7a0507984787e28350.jpg)

### Tables

![314638863a1e9058e63d50499936f40f1943081e6fb8571f3866402ba75e33d2.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/tables/314638863a1e9058e63d50499936f40f1943081e6fb8571f3866402ba75e33d2.jpg)

![335af84607983d1fa0f164aca637194a2994f2990e36205007e48d6690f8f0ad.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/tables/335af84607983d1fa0f164aca637194a2994f2990e36205007e48d6690f8f0ad.jpg)

![76363df5f39247f18c261d97d5ad85609d2e995e49d14618b3efc26e6024d2f7.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/tables/76363df5f39247f18c261d97d5ad85609d2e995e49d14618b3efc26e6024d2f7.jpg)

![78ec892f53bc51eaed5b803c25d30eadd97fddb738dbedbc4509ea203d1e2490.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/tables/78ec892f53bc51eaed5b803c25d30eadd97fddb738dbedbc4509ea203d1e2490.jpg)

![7b6b0991c698f0e33ad757fa66d68b0cb4f73bbe23ccc4d6a4665879f00de953.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/tables/7b6b0991c698f0e33ad757fa66d68b0cb4f73bbe23ccc4d6a4665879f00de953.jpg)

![7ce2903aa9d8d310d7be9d40eb1548f9313e40608740434f0efe819444ddda89.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/tables/7ce2903aa9d8d310d7be9d40eb1548f9313e40608740434f0efe819444ddda89.jpg)

![7d9cd11332c5482898dcf8355cad6e9ff764f3539fc126fd0b9f1583d7cb39db.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/tables/7d9cd11332c5482898dcf8355cad6e9ff764f3539fc126fd0b9f1583d7cb39db.jpg)

![7e1222198c57ce1fedc0739df28a584b4d6cc94e504d5a4cef30e209c9642de9.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/tables/7e1222198c57ce1fedc0739df28a584b4d6cc94e504d5a4cef30e209c9642de9.jpg)

![8ac85f42fc66e172eda6177ce0c7f057421c85ce25c9621f1fa790215e0b9873.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/tables/8ac85f42fc66e172eda6177ce0c7f057421c85ce25c9621f1fa790215e0b9873.jpg)

![8af3a8ce185a0113cf1ae6946f01365a8b3b5b3bebe4964d3598dd6acd109750.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/tables/8af3a8ce185a0113cf1ae6946f01365a8b3b5b3bebe4964d3598dd6acd109750.jpg)

![8f7d5ec1cd131ceb32a3f4f068b7dd202f5cf13175d2ece5db45cedd26effb72.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/tables/8f7d5ec1cd131ceb32a3f4f068b7dd202f5cf13175d2ece5db45cedd26effb72.jpg)

![b037a0ad4145a087b10ca77554609e96289fde9382079712756ac51abe8724f1.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/tables/b037a0ad4145a087b10ca77554609e96289fde9382079712756ac51abe8724f1.jpg)

![b54725625a99a32e213c504f2fdbce1041aaa9f1d67485c3cb136926acf33314.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/tables/b54725625a99a32e213c504f2fdbce1041aaa9f1d67485c3cb136926acf33314.jpg)

![bed9f09363fd2d6ecc73ee8516aec9a82bd3118ffaed2b3d497ddbd895193f7e.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/tables/bed9f09363fd2d6ecc73ee8516aec9a82bd3118ffaed2b3d497ddbd895193f7e.jpg)

![ca41da9661ef13809a5072c6dc4618c38c39feed2a908c69a61f042ecbb46e80.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/tables/ca41da9661ef13809a5072c6dc4618c38c39feed2a908c69a61f042ecbb46e80.jpg)

![d54fa1f15d9fccc39a05a6e07b6b1c1fbd634f48771f9d6d2ca40bf506c9a73f.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/tables/d54fa1f15d9fccc39a05a6e07b6b1c1fbd634f48771f9d6d2ca40bf506c9a73f.jpg)

![d64838c7d54f503ce8a2b3cf0b5f71813230bab3ea5dd7e0791294f8930cb103.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/tables/d64838c7d54f503ce8a2b3cf0b5f71813230bab3ea5dd7e0791294f8930cb103.jpg)

![e189dde8d14f0cdbc7a080cae35f45eb19957a0211913b9a34c4db8b1c44e815.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/tables/e189dde8d14f0cdbc7a080cae35f45eb19957a0211913b9a34c4db8b1c44e815.jpg)

![e65f925c44436299a722a47bd62ccbf8287095c75db820da42c943d5777a16a4.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/tables/e65f925c44436299a722a47bd62ccbf8287095c75db820da42c943d5777a16a4.jpg)

![ed06a590084735484bc95ed7d4d7f4773f452dbd852ca1d8b0de14e30fc30748.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/tables/ed06a590084735484bc95ed7d4d7f4773f452dbd852ca1d8b0de14e30fc30748.jpg)

![f4267f42ff37bf04d1249e7a5a3a378ff35181603db955a62b1b8e6aaf33f6f1.jpg](../iclr_results/206_On%20the%20Identification%20of%20Temporal%20Causal%20Representation%20with%20Instantaneous%20Dependence/tables/f4267f42ff37bf04d1249e7a5a3a378ff35181603db955a62b1b8e6aaf33f6f1.jpg)

## Towards Understanding Why FixMatch Generalizes Better Than Supervised Learning


### Images

![28c02271d92ff32af721b18a6822959ccc42206a18e7de87a1f41fcd6e98d554.jpg](../iclr_results/207_Towards%20Understanding%20Why%20FixMatch%20Generalizes%20Better%20Than%20Supervised%20Learning/images/28c02271d92ff32af721b18a6822959ccc42206a18e7de87a1f41fcd6e98d554.jpg)

![2b49a70309c0eb53c62f3e81db570002e7535fbcde8733d5ac2dfaf319c4608a.jpg](../iclr_results/207_Towards%20Understanding%20Why%20FixMatch%20Generalizes%20Better%20Than%20Supervised%20Learning/images/2b49a70309c0eb53c62f3e81db570002e7535fbcde8733d5ac2dfaf319c4608a.jpg)

![a3976201730e17bf951cd90b615f2258232a7291815d4b018e15091ddc1c18be.jpg](../iclr_results/207_Towards%20Understanding%20Why%20FixMatch%20Generalizes%20Better%20Than%20Supervised%20Learning/images/a3976201730e17bf951cd90b615f2258232a7291815d4b018e15091ddc1c18be.jpg)

![a5337576c3e5bc4a2b34d069606f5610128c75ced67f2598611a0e8ac5e3f4a0.jpg](../iclr_results/207_Towards%20Understanding%20Why%20FixMatch%20Generalizes%20Better%20Than%20Supervised%20Learning/images/a5337576c3e5bc4a2b34d069606f5610128c75ced67f2598611a0e8ac5e3f4a0.jpg)

![ebe6f4d9a5790ef7301699587cabb879058c5aeb196a36ac77260006e827c9b8.jpg](../iclr_results/207_Towards%20Understanding%20Why%20FixMatch%20Generalizes%20Better%20Than%20Supervised%20Learning/images/ebe6f4d9a5790ef7301699587cabb879058c5aeb196a36ac77260006e827c9b8.jpg)

### Tables

![029f8bb28283fb078419eed0f867f51817e5b18db5ebb46339b4ae237bcc66e4.jpg](../iclr_results/207_Towards%20Understanding%20Why%20FixMatch%20Generalizes%20Better%20Than%20Supervised%20Learning/tables/029f8bb28283fb078419eed0f867f51817e5b18db5ebb46339b4ae237bcc66e4.jpg)

![06c1c7bcbb6c64b02235ad426ab292a2ea5359be5f558a00592cbd76e6767a84.jpg](../iclr_results/207_Towards%20Understanding%20Why%20FixMatch%20Generalizes%20Better%20Than%20Supervised%20Learning/tables/06c1c7bcbb6c64b02235ad426ab292a2ea5359be5f558a00592cbd76e6767a84.jpg)

![0e0f9370233a97aaba8af960bedec3754f9b3afeb1fd4e56aca4725f0c9d86b0.jpg](../iclr_results/207_Towards%20Understanding%20Why%20FixMatch%20Generalizes%20Better%20Than%20Supervised%20Learning/tables/0e0f9370233a97aaba8af960bedec3754f9b3afeb1fd4e56aca4725f0c9d86b0.jpg)

![1fd40f3c5ebaf336c58e446774720928bfc8172f611288a3febbac67d0d40e91.jpg](../iclr_results/207_Towards%20Understanding%20Why%20FixMatch%20Generalizes%20Better%20Than%20Supervised%20Learning/tables/1fd40f3c5ebaf336c58e446774720928bfc8172f611288a3febbac67d0d40e91.jpg)

![2f9d346d3bb064cbd34d152b697e3e24c81df0f6869a178afed5fc754c76c606.jpg](../iclr_results/207_Towards%20Understanding%20Why%20FixMatch%20Generalizes%20Better%20Than%20Supervised%20Learning/tables/2f9d346d3bb064cbd34d152b697e3e24c81df0f6869a178afed5fc754c76c606.jpg)

![4b186ff0969f381d83f52cd348bc4902e022c1d72784641be47008faf2907522.jpg](../iclr_results/207_Towards%20Understanding%20Why%20FixMatch%20Generalizes%20Better%20Than%20Supervised%20Learning/tables/4b186ff0969f381d83f52cd348bc4902e022c1d72784641be47008faf2907522.jpg)

![4d56fb2a19b9888a161afa675c3c5a434c5807b831ef3503af6af373166f428d.jpg](../iclr_results/207_Towards%20Understanding%20Why%20FixMatch%20Generalizes%20Better%20Than%20Supervised%20Learning/tables/4d56fb2a19b9888a161afa675c3c5a434c5807b831ef3503af6af373166f428d.jpg)

![c1ea9c57ba5c3de61e3f2765fb3fa8f474446744a1868cee92f05e723fbba21b.jpg](../iclr_results/207_Towards%20Understanding%20Why%20FixMatch%20Generalizes%20Better%20Than%20Supervised%20Learning/tables/c1ea9c57ba5c3de61e3f2765fb3fa8f474446744a1868cee92f05e723fbba21b.jpg)

![eb55391a432cd0bf08b6ecc2d78f6704db7fb3cf32172ca3af135d7b4290f6dc.jpg](../iclr_results/207_Towards%20Understanding%20Why%20FixMatch%20Generalizes%20Better%20Than%20Supervised%20Learning/tables/eb55391a432cd0bf08b6ecc2d78f6704db7fb3cf32172ca3af135d7b4290f6dc.jpg)

![f709d29751fa644d04d44795b4ca2248c7eb4e10d9635fd74eea23eebf9c6a12.jpg](../iclr_results/207_Towards%20Understanding%20Why%20FixMatch%20Generalizes%20Better%20Than%20Supervised%20Learning/tables/f709d29751fa644d04d44795b4ca2248c7eb4e10d9635fd74eea23eebf9c6a12.jpg)

## RMP-SAM: Towards Real-Time Multi-Purpose Segment Anything


### Images

![10384e7db1efaa52f5853a6b33556173868b645e02731867f90e4ab05798f400.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/images/10384e7db1efaa52f5853a6b33556173868b645e02731867f90e4ab05798f400.jpg)

![17cb1e0a77ccf7e81397985888b692b2f73efea4ad190c439bce4e93252f6b0d.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/images/17cb1e0a77ccf7e81397985888b692b2f73efea4ad190c439bce4e93252f6b0d.jpg)

![2802e34c786dcbb018f2128b113c356d83c90a83d5c9a15b57d4de539a5c855b.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/images/2802e34c786dcbb018f2128b113c356d83c90a83d5c9a15b57d4de539a5c855b.jpg)

![4da628700a32fd78234a31af64753938a973ad780381a972463267060a063c67.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/images/4da628700a32fd78234a31af64753938a973ad780381a972463267060a063c67.jpg)

![53c3854be31d7cc4d799484d1cdf7dfb7d0598e6144026eca12c32cd8df3c7f8.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/images/53c3854be31d7cc4d799484d1cdf7dfb7d0598e6144026eca12c32cd8df3c7f8.jpg)

![54e7db47c8e5ef6ade0b8c56e6304ca749a77df144491620149d462b2aef78e6.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/images/54e7db47c8e5ef6ade0b8c56e6304ca749a77df144491620149d462b2aef78e6.jpg)

![5a967ee6990fdfe13f4fe9f4034c529a168b9385c851e5358ff5ccdc3f26ac00.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/images/5a967ee6990fdfe13f4fe9f4034c529a168b9385c851e5358ff5ccdc3f26ac00.jpg)

![b18785c88c18a819f660bd2a356f231dc7971f1d5bce6431cc02914d6fd56103.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/images/b18785c88c18a819f660bd2a356f231dc7971f1d5bce6431cc02914d6fd56103.jpg)

![bb58587f937d89911924d3f8e5e3cb9bd182e6dcdd65afc644c39d530e6e9cf6.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/images/bb58587f937d89911924d3f8e5e3cb9bd182e6dcdd65afc644c39d530e6e9cf6.jpg)

![fdc6275627332c51af64aa80ce76b08ac52cba95dbc4c2626bb8d33fefa32460.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/images/fdc6275627332c51af64aa80ce76b08ac52cba95dbc4c2626bb8d33fefa32460.jpg)

### Tables

![0439ee44e8e104066d4038cd709ce29e117a8d407b543980172d398f82caae08.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/tables/0439ee44e8e104066d4038cd709ce29e117a8d407b543980172d398f82caae08.jpg)

![06bea453b587b8db6a33705fad16ad21990e0543c3c4e9db44ee27bf932e5d9a.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/tables/06bea453b587b8db6a33705fad16ad21990e0543c3c4e9db44ee27bf932e5d9a.jpg)

![1edba8026bc30a2df5ceb812c5d8226d5cc2d4d8174d7cab7da529c337adeaa5.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/tables/1edba8026bc30a2df5ceb812c5d8226d5cc2d4d8174d7cab7da529c337adeaa5.jpg)

![2fde0e9a2d7ff7399b8029c9f5e85b7219c57bc90470ca4c004c5de33c2b4882.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/tables/2fde0e9a2d7ff7399b8029c9f5e85b7219c57bc90470ca4c004c5de33c2b4882.jpg)

![37edcb9855556b384d7295288f4b80e9157f9e04dac5076378f83bc782eb2157.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/tables/37edcb9855556b384d7295288f4b80e9157f9e04dac5076378f83bc782eb2157.jpg)

![3b15693c29a977448116599f1b12596eba0c0ff334475578a91eaecd3dde42e7.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/tables/3b15693c29a977448116599f1b12596eba0c0ff334475578a91eaecd3dde42e7.jpg)

![3ef4badb15731d1140e80b1e1891b9a798bdadacba20b8a50d9c7be54b0822b8.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/tables/3ef4badb15731d1140e80b1e1891b9a798bdadacba20b8a50d9c7be54b0822b8.jpg)

![41dd90f7aa00313fdc69c4a6d42bcf5a673dc8ac2d622ebcf315d2cd648ba4a3.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/tables/41dd90f7aa00313fdc69c4a6d42bcf5a673dc8ac2d622ebcf315d2cd648ba4a3.jpg)

![5527fa1891eeb94eb39baa422438f9f10c0828310f8e477912ccba1243fc6c1d.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/tables/5527fa1891eeb94eb39baa422438f9f10c0828310f8e477912ccba1243fc6c1d.jpg)

![67404999c5b96a388c298b19e499a8ec52bb1824e890017c0e2b15a580806d7a.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/tables/67404999c5b96a388c298b19e499a8ec52bb1824e890017c0e2b15a580806d7a.jpg)

![82fe6a08b706c894c8ccfd5b3df3a765936d561fe141f1035837873875e3855b.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/tables/82fe6a08b706c894c8ccfd5b3df3a765936d561fe141f1035837873875e3855b.jpg)

![86881a4f62b802aa5842c181fbd199f29df091cb006673397e43ccde29a35dde.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/tables/86881a4f62b802aa5842c181fbd199f29df091cb006673397e43ccde29a35dde.jpg)

![886aaca2f7b568303520b970a7a95fb75f6093530d8170afb4d6e52c7ddd9f9f.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/tables/886aaca2f7b568303520b970a7a95fb75f6093530d8170afb4d6e52c7ddd9f9f.jpg)

![8c06e366d0698886c39dae1fc1c90d3e003bbeb9c3416962fb4ab047ef979b26.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/tables/8c06e366d0698886c39dae1fc1c90d3e003bbeb9c3416962fb4ab047ef979b26.jpg)

![957313d529048b973a4a925673c0599a820cd9f771c85a0ebeb15b3df558933e.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/tables/957313d529048b973a4a925673c0599a820cd9f771c85a0ebeb15b3df558933e.jpg)

![ad98e5ea2971717a652c88ace33f9ffc04e10ff5420ce4813518ccc943947466.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/tables/ad98e5ea2971717a652c88ace33f9ffc04e10ff5420ce4813518ccc943947466.jpg)

![ba8c70ec31c14dbd18dc04e52db1eb31c4e5533d0f710b3b48b50d9682c53f6c.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/tables/ba8c70ec31c14dbd18dc04e52db1eb31c4e5533d0f710b3b48b50d9682c53f6c.jpg)

![ffacdb0d62f17231244623a26168d0a52fce6ae48eeb45c24bc470e5be30507e.jpg](../iclr_results/208_RMP-SAM_%20Towards%20Real-Time%20Multi-Purpose%20Segment%20Anything/tables/ffacdb0d62f17231244623a26168d0a52fce6ae48eeb45c24bc470e5be30507e.jpg)

## Open-Vocabulary Customization from CLIP via Data-Free Knowledge Distillation


### Images

![19a0817a5853ac90ee644d6d6e6f6ddce315ed586718dad7896bd9cf292588ba.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/images/19a0817a5853ac90ee644d6d6e6f6ddce315ed586718dad7896bd9cf292588ba.jpg)

![4aaf27e10a40a32f965907cc6db34b5f84b0e58947fb135c0c10bbeda0e4e483.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/images/4aaf27e10a40a32f965907cc6db34b5f84b0e58947fb135c0c10bbeda0e4e483.jpg)

![553066ef340fb2bdd382271d68eae29ed68fbf39f6d6f6417da6be35e8fde53e.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/images/553066ef340fb2bdd382271d68eae29ed68fbf39f6d6f6417da6be35e8fde53e.jpg)

![718b6d8f7b211cfdce9663aa534a0c2f7c9c342860c108050b26f168234a132e.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/images/718b6d8f7b211cfdce9663aa534a0c2f7c9c342860c108050b26f168234a132e.jpg)

![7dd5bd310d7e653c80485b71a66995c493ef39729cee04e100fa5caccbf59d0a.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/images/7dd5bd310d7e653c80485b71a66995c493ef39729cee04e100fa5caccbf59d0a.jpg)

![913c861389978a292cfb80686783b546013729d15be99608713fc39013d6888f.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/images/913c861389978a292cfb80686783b546013729d15be99608713fc39013d6888f.jpg)

![9548ced5068971671e8cb45a0f5ac5c73454b6742098da529c6356dc3513d47b.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/images/9548ced5068971671e8cb45a0f5ac5c73454b6742098da529c6356dc3513d47b.jpg)

![97c4f771ee85c929ace57231f64f9db98fd916a299475348d8711565ac30333d.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/images/97c4f771ee85c929ace57231f64f9db98fd916a299475348d8711565ac30333d.jpg)

![e3c2cfb73ca3fe7e9c6276fd01e9cf6f9ffb0c9dadf343f86bdaf35923918da0.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/images/e3c2cfb73ca3fe7e9c6276fd01e9cf6f9ffb0c9dadf343f86bdaf35923918da0.jpg)

![ee121b717a397fda9b6f92e5015ed2fff8388d2e4424db50cd14f27294058832.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/images/ee121b717a397fda9b6f92e5015ed2fff8388d2e4424db50cd14f27294058832.jpg)

### Tables

![17f26f8f535eaba9cc99ff68e5d3a0022651647974011ad11871fa15065af86a.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/tables/17f26f8f535eaba9cc99ff68e5d3a0022651647974011ad11871fa15065af86a.jpg)

![35b75c8d92a8dcf99bcf16ac148646f31125374ca2b02b831dc4de58b28559ec.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/tables/35b75c8d92a8dcf99bcf16ac148646f31125374ca2b02b831dc4de58b28559ec.jpg)

![58c2dbfce8a5a83dfbfcb8c6cd54709db0f0e22de27fc0a48f7ce243a025db09.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/tables/58c2dbfce8a5a83dfbfcb8c6cd54709db0f0e22de27fc0a48f7ce243a025db09.jpg)

![6cca969cdbf9ce718dbc5b07a68dd2450591618f74e3798f76449b26d5d40290.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/tables/6cca969cdbf9ce718dbc5b07a68dd2450591618f74e3798f76449b26d5d40290.jpg)

![7c4d636509f6b88dddf705112edeb2c3cd3453189e3bb146da9fac012d8fa1bc.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/tables/7c4d636509f6b88dddf705112edeb2c3cd3453189e3bb146da9fac012d8fa1bc.jpg)

![8a3e7df9a80b095e65cb1a6db51a59cb0d1ff00c01f1d317fbf3c9cf06383a4e.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/tables/8a3e7df9a80b095e65cb1a6db51a59cb0d1ff00c01f1d317fbf3c9cf06383a4e.jpg)

![98f10c8450aff037a0cedc7776180911791284bfe592ce9c344ecb314b332eae.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/tables/98f10c8450aff037a0cedc7776180911791284bfe592ce9c344ecb314b332eae.jpg)

![9f5bc118172c2e04bd2f5eafcf473e376e4713ab968ce0294d8241a9a0e27365.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/tables/9f5bc118172c2e04bd2f5eafcf473e376e4713ab968ce0294d8241a9a0e27365.jpg)

![a57d4ebb1bb791e9ecd1befc2c02148bb572fe5e537aac626a809c945c32cddc.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/tables/a57d4ebb1bb791e9ecd1befc2c02148bb572fe5e537aac626a809c945c32cddc.jpg)

![a65b7d1e8dc5c861501e6cc42400fd454efcb0cb504d6eee371b833ea9b5446b.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/tables/a65b7d1e8dc5c861501e6cc42400fd454efcb0cb504d6eee371b833ea9b5446b.jpg)

![b746341c9003722cc4c436bca3ed674e95ea9b23b48ecb3e85748fb1cd87f299.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/tables/b746341c9003722cc4c436bca3ed674e95ea9b23b48ecb3e85748fb1cd87f299.jpg)

![c1847963acdf13c6e3c7c406b0afbb58b11094b4a6c7773b743ffe2c29979f70.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/tables/c1847963acdf13c6e3c7c406b0afbb58b11094b4a6c7773b743ffe2c29979f70.jpg)

![c4b69096916b1143d3ad5e27b11fd05e2550c6922ff0468f0523ab826e5be1df.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/tables/c4b69096916b1143d3ad5e27b11fd05e2550c6922ff0468f0523ab826e5be1df.jpg)

![c8275505df6d03f75d2160865f5662557cd780493007d6d6e25b1026234422b9.jpg](../iclr_results/209_Open-Vocabulary%20Customization%20from%20CLIP%20via%20Data-Free%20Knowledge%20Distillation/tables/c8275505df6d03f75d2160865f5662557cd780493007d6d6e25b1026234422b9.jpg)

## Wide Neural Networks Trained with Weight Decay Provably Exhibit Neural Collapse


### Images

![524a69755df58b2c941bd4c21c6d49084ce062ea03e79dab8e5ed763d9f6fe5d.jpg](../iclr_results/210_Wide%20Neural%20Networks%20Trained%20with%20Weight%20Decay%20Provably%20Exhibit%20Neural%20Collapse/images/524a69755df58b2c941bd4c21c6d49084ce062ea03e79dab8e5ed763d9f6fe5d.jpg)

![907b8674d90d515850634304393f91a1b0ed6881170db20bde5a3cbb75ae759b.jpg](../iclr_results/210_Wide%20Neural%20Networks%20Trained%20with%20Weight%20Decay%20Provably%20Exhibit%20Neural%20Collapse/images/907b8674d90d515850634304393f91a1b0ed6881170db20bde5a3cbb75ae759b.jpg)

![ce1555f78ad92090adaa62765f8d2cef3be83803d2c1a3ebd2ebbd182712772c.jpg](../iclr_results/210_Wide%20Neural%20Networks%20Trained%20with%20Weight%20Decay%20Provably%20Exhibit%20Neural%20Collapse/images/ce1555f78ad92090adaa62765f8d2cef3be83803d2c1a3ebd2ebbd182712772c.jpg)

![d0eba4d79663bc8ca27c8b9958a5fdde5edd0e32e58a61cdd39f1e5b51d939ef.jpg](../iclr_results/210_Wide%20Neural%20Networks%20Trained%20with%20Weight%20Decay%20Provably%20Exhibit%20Neural%20Collapse/images/d0eba4d79663bc8ca27c8b9958a5fdde5edd0e32e58a61cdd39f1e5b51d939ef.jpg)

![dc60122936b4c999dcfec4b309e9e7b4762a6c8fce7ce7cb290b21fc476c70d7.jpg](../iclr_results/210_Wide%20Neural%20Networks%20Trained%20with%20Weight%20Decay%20Provably%20Exhibit%20Neural%20Collapse/images/dc60122936b4c999dcfec4b309e9e7b4762a6c8fce7ce7cb290b21fc476c70d7.jpg)

## TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis


### Images

![02a96c36d0ff07f6e5202e032b34ce2c901cc158a2739973615cbb5abaec98a9.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/02a96c36d0ff07f6e5202e032b34ce2c901cc158a2739973615cbb5abaec98a9.jpg)

![0316a1109f646b363a6bfb25b0cf14ada56ab04211d8c70d344d41b2fb319572.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/0316a1109f646b363a6bfb25b0cf14ada56ab04211d8c70d344d41b2fb319572.jpg)

![118485de6fbde78c6a0b1d8be799f00929ddeea6713c6cc2015d78ba84374953.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/118485de6fbde78c6a0b1d8be799f00929ddeea6713c6cc2015d78ba84374953.jpg)

![194f832c8021289aa93bbf47efdb280627bc8b0955f42782b53c68b5f71f0af1.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/194f832c8021289aa93bbf47efdb280627bc8b0955f42782b53c68b5f71f0af1.jpg)

![1ffa8285857874a468e60036f4e469770e5dc2a7074ef39142bf4a3326b05315.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/1ffa8285857874a468e60036f4e469770e5dc2a7074ef39142bf4a3326b05315.jpg)

![2068d52c21ce4ff0a5acac430370e56623ef7730b9a5ea1cdeba0d1e4c8e9fc2.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/2068d52c21ce4ff0a5acac430370e56623ef7730b9a5ea1cdeba0d1e4c8e9fc2.jpg)

![4058c2e2761016cbd96ac7709cce1c37e85c78882603931ea05eb4f3001f3fea.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/4058c2e2761016cbd96ac7709cce1c37e85c78882603931ea05eb4f3001f3fea.jpg)

![42f19110bb328286b0ce8f2122c1f5c36e8d8f43cd0a7d910746c8a608619906.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/42f19110bb328286b0ce8f2122c1f5c36e8d8f43cd0a7d910746c8a608619906.jpg)

![4de47685065efe5f70a1523037c0059535f6bd86f776dffc3bcb58b707cc22ec.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/4de47685065efe5f70a1523037c0059535f6bd86f776dffc3bcb58b707cc22ec.jpg)

![51fa31ffc7abadce7a67a06817aa96ff43642f389f44498f2aca21675b3edc08.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/51fa31ffc7abadce7a67a06817aa96ff43642f389f44498f2aca21675b3edc08.jpg)

![6a31575e0b51c73092e098297c05a0a6aef46f89dce39ca37f96ea26e0af18e7.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/6a31575e0b51c73092e098297c05a0a6aef46f89dce39ca37f96ea26e0af18e7.jpg)

![73bab5ab56be6be6e8d23d6e461e232a1861950c30cdcd72ab2c351b19645783.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/73bab5ab56be6be6e8d23d6e461e232a1861950c30cdcd72ab2c351b19645783.jpg)

![74b7ba9876897ce56986054d4095382b79034ad75c828bb55fec84c0cda55d26.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/74b7ba9876897ce56986054d4095382b79034ad75c828bb55fec84c0cda55d26.jpg)

![7d6b678470c4194c7efb8c68fffc6873af8000469268e7efe6ae96cdeb0247fc.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/7d6b678470c4194c7efb8c68fffc6873af8000469268e7efe6ae96cdeb0247fc.jpg)

![8702663d6e53a01b6f2456a060f036af52f5d639e12bbb3b09abe925e806ee26.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/8702663d6e53a01b6f2456a060f036af52f5d639e12bbb3b09abe925e806ee26.jpg)

![8d298b1e70de4e11c68b7480f05f7d4e82fe5353dbe280d9d8253f7c7841a06e.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/8d298b1e70de4e11c68b7480f05f7d4e82fe5353dbe280d9d8253f7c7841a06e.jpg)

![90d819388c5ba1119276bb4bd598a4171604a2a51f37c9ddcab4124cbb838c3a.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/90d819388c5ba1119276bb4bd598a4171604a2a51f37c9ddcab4124cbb838c3a.jpg)

![a2c8d6468c842ee355ebb9b604f5429cb299ce025095baacd7e8e335b92c6fdf.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/a2c8d6468c842ee355ebb9b604f5429cb299ce025095baacd7e8e335b92c6fdf.jpg)

![ac0433a06ec81dfa4d272077ce9d5dec149a173fd91d89fa4231e2f12492e3b0.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/ac0433a06ec81dfa4d272077ce9d5dec149a173fd91d89fa4231e2f12492e3b0.jpg)

![be4c5bbed2af0e8677367296266b369e70e48ce2f6c4ba3b25d4891f1d7a1804.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/be4c5bbed2af0e8677367296266b369e70e48ce2f6c4ba3b25d4891f1d7a1804.jpg)

![bfe5402d7cfebb6e17e967a174cfa4eb829c28bf2a2243553676a7e972779e29.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/bfe5402d7cfebb6e17e967a174cfa4eb829c28bf2a2243553676a7e972779e29.jpg)

![c58cb27fb3fe41d6051eaf415a6808dc483a4a820e8460acffde63f92a01a5a9.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/c58cb27fb3fe41d6051eaf415a6808dc483a4a820e8460acffde63f92a01a5a9.jpg)

![cc6c7315b16c1c9246300d59c78bf7b8fb9c11626d7a5237ae9ece4207c4f4b1.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/cc6c7315b16c1c9246300d59c78bf7b8fb9c11626d7a5237ae9ece4207c4f4b1.jpg)

![f69bde1ecd084edf0908eac7caf237b90b5316ac5f8310910875d28bff06a5e6.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/f69bde1ecd084edf0908eac7caf237b90b5316ac5f8310910875d28bff06a5e6.jpg)

![fab1e368f549984f478b676b0eb1f765660bdd7d6aa9ede4821e1a9b1cfb8b69.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/fab1e368f549984f478b676b0eb1f765660bdd7d6aa9ede4821e1a9b1cfb8b69.jpg)

![ffe2f8b0247550cc90472407bd7e9b4671d0616b67efc20b7a1999357d4bd940.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/images/ffe2f8b0247550cc90472407bd7e9b4671d0616b67efc20b7a1999357d4bd940.jpg)

### Tables

![18cb73dce3b514683e07cac43317de7a14fee074a5cd043a74e950b39d4a95a2.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/tables/18cb73dce3b514683e07cac43317de7a14fee074a5cd043a74e950b39d4a95a2.jpg)

![198617a83930fe0b36860e96880466ad4dee39ad54a5d17a1ceb4b2b1e226a04.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/tables/198617a83930fe0b36860e96880466ad4dee39ad54a5d17a1ceb4b2b1e226a04.jpg)

![1bb4af1fa03cde25288b6b99f6de056f8595f2e6fa7048734034336a6791f837.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/tables/1bb4af1fa03cde25288b6b99f6de056f8595f2e6fa7048734034336a6791f837.jpg)

![1bf5a13d293c8f2d7962f5a632b69df972040a943218a8cc418f6baf4b5832dc.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/tables/1bf5a13d293c8f2d7962f5a632b69df972040a943218a8cc418f6baf4b5832dc.jpg)

![2c2c24c890f553c535e81c83eaa3fca5b9a4e9533d91dec97c7240b00e88c05e.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/tables/2c2c24c890f553c535e81c83eaa3fca5b9a4e9533d91dec97c7240b00e88c05e.jpg)

![3098c0d802000eabf26fdbaa19812b29f0b72afbc8583105d8c3d7356698f2db.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/tables/3098c0d802000eabf26fdbaa19812b29f0b72afbc8583105d8c3d7356698f2db.jpg)

![410ff8ab579bce7791bfc7bc54e55e18fbc01cf322f64da0909657f2ed6085eb.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/tables/410ff8ab579bce7791bfc7bc54e55e18fbc01cf322f64da0909657f2ed6085eb.jpg)

![530f9cdd22304f28920d7db56d966346cd8567f61980edabd099a12536311265.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/tables/530f9cdd22304f28920d7db56d966346cd8567f61980edabd099a12536311265.jpg)

![5aef3a85f7eda28abf57db446b2e0d10d07ff17b2c11144722f2b30a6ddc1153.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/tables/5aef3a85f7eda28abf57db446b2e0d10d07ff17b2c11144722f2b30a6ddc1153.jpg)

![600ff4b61bed70c31e1dbd134c9dd4d0f92727271a489bc6d5fbeb58f48fb146.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/tables/600ff4b61bed70c31e1dbd134c9dd4d0f92727271a489bc6d5fbeb58f48fb146.jpg)

![8e01977bd6ed40839e1a33d9b97150ea8f4a243afefa441536552a7f31236b7b.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/tables/8e01977bd6ed40839e1a33d9b97150ea8f4a243afefa441536552a7f31236b7b.jpg)

![923525edbc44b8fb5d8c18399abebe67ff3225123bc4e76a315e362348063f7a.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/tables/923525edbc44b8fb5d8c18399abebe67ff3225123bc4e76a315e362348063f7a.jpg)

![9ca593cc007a47ca4c8cd1c45059dc8d39098f80f6b2fdec90c3102e525910f5.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/tables/9ca593cc007a47ca4c8cd1c45059dc8d39098f80f6b2fdec90c3102e525910f5.jpg)

![a312d1ca95da87f50b99e517b739db20746f92978ecdad8c9f1849cefa31a28e.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/tables/a312d1ca95da87f50b99e517b739db20746f92978ecdad8c9f1849cefa31a28e.jpg)

![ab23a74753519949bc231d7eb00c38c08845b3bc94ff93ee513cafc29ca9269f.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/tables/ab23a74753519949bc231d7eb00c38c08845b3bc94ff93ee513cafc29ca9269f.jpg)

![b6f66f8a6e6327fe33c41786f434f59164e177f0517af90e47258a56f6a7d094.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/tables/b6f66f8a6e6327fe33c41786f434f59164e177f0517af90e47258a56f6a7d094.jpg)

![c0b5a59c10d778422fce1ffded7681487f7254b21a45bbd8664efad345cdc133.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/tables/c0b5a59c10d778422fce1ffded7681487f7254b21a45bbd8664efad345cdc133.jpg)

![d594166701b37a2c6c5cd2f2b087079d7b7a58937192242ef366cf5ae534aafb.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/tables/d594166701b37a2c6c5cd2f2b087079d7b7a58937192242ef366cf5ae534aafb.jpg)

![dea9b4f322a41a66572f612d303d676f02f7860d834d4c137c336dcd3194722a.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/tables/dea9b4f322a41a66572f612d303d676f02f7860d834d4c137c336dcd3194722a.jpg)

![e952ecae808675835b5bfaabb88765d680734c534da3c20f0ea4e282a5cd7718.jpg](../iclr_results/211_TimeMixer%2B%2B_%20A%20General%20Time%20Series%20Pattern%20Machine%20for%20Universal%20Predictive%20Analysis/tables/e952ecae808675835b5bfaabb88765d680734c534da3c20f0ea4e282a5cd7718.jpg)

## ProtComposer: Compositional Protein Structure Generation with 3D Ellipsoids


### Images

![0aea503d990e295edab29b6abc842f9f2f660d38fd697a9e3cd751fca21e0246.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/0aea503d990e295edab29b6abc842f9f2f660d38fd697a9e3cd751fca21e0246.jpg)

![0e50b73f2fae2fefe294ddbb85dd7d33fe470ae3adf4643ce1e33bc98d3af1b4.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/0e50b73f2fae2fefe294ddbb85dd7d33fe470ae3adf4643ce1e33bc98d3af1b4.jpg)

![12a2ebe113f8111944d209491aecb563059ae832f1fb7609b753d50dce16cfdd.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/12a2ebe113f8111944d209491aecb563059ae832f1fb7609b753d50dce16cfdd.jpg)

![1c0eb4ec3c8918fa58d38f4c5befc18895b5b998cc89689a4fd700dc0fac10af.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/1c0eb4ec3c8918fa58d38f4c5befc18895b5b998cc89689a4fd700dc0fac10af.jpg)

![216587e1fc59afe0bfe97c436f0c6a3a5145aa3442f32ac9a748cede5bcbbb7a.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/216587e1fc59afe0bfe97c436f0c6a3a5145aa3442f32ac9a748cede5bcbbb7a.jpg)

![2823fbfd175af86b9713a05c925a348e5c1fbb8b521966ceee6b2bfa3fd046b6.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/2823fbfd175af86b9713a05c925a348e5c1fbb8b521966ceee6b2bfa3fd046b6.jpg)

![2e3a0df600da217141e93a1a7258d63c42a5d3b302e7e9599039d00239646d73.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/2e3a0df600da217141e93a1a7258d63c42a5d3b302e7e9599039d00239646d73.jpg)

![2efe1093b13af39951a85899e6e0a9b3b204f62f1b2388aea27ce76cc91e109b.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/2efe1093b13af39951a85899e6e0a9b3b204f62f1b2388aea27ce76cc91e109b.jpg)

![3a21d49548548c12f60681758b84eb2868693c9cca34c13cc1527c70d7883b9f.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/3a21d49548548c12f60681758b84eb2868693c9cca34c13cc1527c70d7883b9f.jpg)

![3ce5b201691ae1f03dafe72f4bf53263956c63f3f7b45991dbd9dc3a0a91e713.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/3ce5b201691ae1f03dafe72f4bf53263956c63f3f7b45991dbd9dc3a0a91e713.jpg)

![3ff926828957715e44e7056a612d7c09cbad234cc9f6c66933b217bf92ec47de.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/3ff926828957715e44e7056a612d7c09cbad234cc9f6c66933b217bf92ec47de.jpg)

![459ae46724c6d3f15e469137ef2065856c4f4e6128166b1ef8496a63095fd2a6.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/459ae46724c6d3f15e469137ef2065856c4f4e6128166b1ef8496a63095fd2a6.jpg)

![4c85fad6bd90e94154edc871518bf3bcc1bfde9ab7f317850e65467e36ed5a72.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/4c85fad6bd90e94154edc871518bf3bcc1bfde9ab7f317850e65467e36ed5a72.jpg)

![500ecb1ef5939ef0adde00e57b0e8fb331225c0655c03dab3ec29b098acf3c2d.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/500ecb1ef5939ef0adde00e57b0e8fb331225c0655c03dab3ec29b098acf3c2d.jpg)

![57bec0f6eeb31184ec20c85ad41e7921aac83d5635c2d152c07e3f59145e2b80.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/57bec0f6eeb31184ec20c85ad41e7921aac83d5635c2d152c07e3f59145e2b80.jpg)

![651319f0826f1230d67bd3f001de537f2407d071e818095dc5d9a3e72b21e103.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/651319f0826f1230d67bd3f001de537f2407d071e818095dc5d9a3e72b21e103.jpg)

![74470eade1f7b267c7c29771bc2a513797382e36d4d9de04ff59c56bb9fb555a.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/74470eade1f7b267c7c29771bc2a513797382e36d4d9de04ff59c56bb9fb555a.jpg)

![856c7062d49e823235efe2566c5964b87c94ce6851d90474ef276757b6524638.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/856c7062d49e823235efe2566c5964b87c94ce6851d90474ef276757b6524638.jpg)

![8801433094cd0bf71090766c2357937ad640bf85de65a8d543930c90a4e5356a.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/8801433094cd0bf71090766c2357937ad640bf85de65a8d543930c90a4e5356a.jpg)

![963a96b2e08eddcaed6bee9a598ea32be6e6da3e623d44d7b9b6418987a37f3b.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/963a96b2e08eddcaed6bee9a598ea32be6e6da3e623d44d7b9b6418987a37f3b.jpg)

![c6958fa6c651cff1a6b992c032d9ea2df1399c22e5ae0102d7b317ae0035876b.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/c6958fa6c651cff1a6b992c032d9ea2df1399c22e5ae0102d7b317ae0035876b.jpg)

![cb62b817bdb6af7cb51b6e1646f1cb2f9eabffb2dc8af0730eaffdc442534bb2.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/cb62b817bdb6af7cb51b6e1646f1cb2f9eabffb2dc8af0730eaffdc442534bb2.jpg)

![cc7ceaf655e54f787263bcad9e36fcdccc4fd5e43e87f9a2100816c13dc88df9.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/cc7ceaf655e54f787263bcad9e36fcdccc4fd5e43e87f9a2100816c13dc88df9.jpg)

![dd6c1d1f84fff0774d4b6ac07be4e9895703522a020bffddca61174618951daf.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/images/dd6c1d1f84fff0774d4b6ac07be4e9895703522a020bffddca61174618951daf.jpg)

### Tables

![3115d2e4532b278471a2b2e62c0ec07e6fcec82e5f0fec973c2063aab9a0271c.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/tables/3115d2e4532b278471a2b2e62c0ec07e6fcec82e5f0fec973c2063aab9a0271c.jpg)

![3343bdb119118651cb9983965f60594d2bc6f3222186746e1c59253adbaab217.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/tables/3343bdb119118651cb9983965f60594d2bc6f3222186746e1c59253adbaab217.jpg)

![824795d0ec5a94f7657975c96dfc03b74b26a401d8c6b3493224a98fadf935f8.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/tables/824795d0ec5a94f7657975c96dfc03b74b26a401d8c6b3493224a98fadf935f8.jpg)

![cd2abc1f8e669cd3bdab1a5e98dd1610376285e22825faf59a74810b9fa47a33.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/tables/cd2abc1f8e669cd3bdab1a5e98dd1610376285e22825faf59a74810b9fa47a33.jpg)

![d58d04c80dabc73d0a08d0921aa38ae39dd137459b01238108ffccecc50a599c.jpg](../iclr_results/212_ProtComposer_%20Compositional%20Protein%20Structure%20Generation%20with%203D%20Ellipsoids/tables/d58d04c80dabc73d0a08d0921aa38ae39dd137459b01238108ffccecc50a599c.jpg)

## Synthetic continued pretraining


### Images

![18d443fb640c14cd12b30faa4d092ba999a1d53cc85455384f4cfdf476fecd30.jpg](../iclr_results/213_Synthetic%20continued%20pretraining/images/18d443fb640c14cd12b30faa4d092ba999a1d53cc85455384f4cfdf476fecd30.jpg)

![1f04fa725465d7d9d55db3c581ba04dbf26451c879ed5b92e4f33230eeaf928d.jpg](../iclr_results/213_Synthetic%20continued%20pretraining/images/1f04fa725465d7d9d55db3c581ba04dbf26451c879ed5b92e4f33230eeaf928d.jpg)

![53d892fc1aa58f7551e4ebb635f66a868e33af91545608affd8744e6ecaab51a.jpg](../iclr_results/213_Synthetic%20continued%20pretraining/images/53d892fc1aa58f7551e4ebb635f66a868e33af91545608affd8744e6ecaab51a.jpg)

![851a9c8ff1477623c529e5441bf1db73840e32bff540b7691a9bcd4fe93ddad5.jpg](../iclr_results/213_Synthetic%20continued%20pretraining/images/851a9c8ff1477623c529e5441bf1db73840e32bff540b7691a9bcd4fe93ddad5.jpg)

![a1ada425a58c9deb25351119a2951ff8029406f960c1d951da8fd5ff37131a42.jpg](../iclr_results/213_Synthetic%20continued%20pretraining/images/a1ada425a58c9deb25351119a2951ff8029406f960c1d951da8fd5ff37131a42.jpg)

![ac27eb7eda7798a470d52b61962fdd38ff2ba88a42e546afd3008db85b86b202.jpg](../iclr_results/213_Synthetic%20continued%20pretraining/images/ac27eb7eda7798a470d52b61962fdd38ff2ba88a42e546afd3008db85b86b202.jpg)

![b8ca698f9bf2b76466635c092467795745d51c599d2cd87e3358e9560e106d7d.jpg](../iclr_results/213_Synthetic%20continued%20pretraining/images/b8ca698f9bf2b76466635c092467795745d51c599d2cd87e3358e9560e106d7d.jpg)

![c6ebf3e3004e7516f8308e4bc6d38a243f008b6626cfe3107a1aaaa6efd3da31.jpg](../iclr_results/213_Synthetic%20continued%20pretraining/images/c6ebf3e3004e7516f8308e4bc6d38a243f008b6626cfe3107a1aaaa6efd3da31.jpg)

![d3d047b59cc646384f30e68091f1804fcfb5db60ba4916ed34a12554adcc0190.jpg](../iclr_results/213_Synthetic%20continued%20pretraining/images/d3d047b59cc646384f30e68091f1804fcfb5db60ba4916ed34a12554adcc0190.jpg)

### Tables

![005bc1dee9725507b76552602f96898287a3234260be5c1fb8ca7f29c47bde39.jpg](../iclr_results/213_Synthetic%20continued%20pretraining/tables/005bc1dee9725507b76552602f96898287a3234260be5c1fb8ca7f29c47bde39.jpg)

![1478bf2ddd9f78e1457610b7906aa521f9d4faaf5fb9324826c1dc280b2f2f01.jpg](../iclr_results/213_Synthetic%20continued%20pretraining/tables/1478bf2ddd9f78e1457610b7906aa521f9d4faaf5fb9324826c1dc280b2f2f01.jpg)

![1b03ddb7629b6d99a8bd15a38bafdb74ec0ea290b04ccc3b08dc055dcf4bfaa3.jpg](../iclr_results/213_Synthetic%20continued%20pretraining/tables/1b03ddb7629b6d99a8bd15a38bafdb74ec0ea290b04ccc3b08dc055dcf4bfaa3.jpg)

![3e162ecf2a0e4913a48d898fc387fa121bf637e1fffca685759c02dca5a72ada.jpg](../iclr_results/213_Synthetic%20continued%20pretraining/tables/3e162ecf2a0e4913a48d898fc387fa121bf637e1fffca685759c02dca5a72ada.jpg)

![9d6651cfd4d7510dad058df5147e701c1e04a6f01abeddd71ba7004a78bb8734.jpg](../iclr_results/213_Synthetic%20continued%20pretraining/tables/9d6651cfd4d7510dad058df5147e701c1e04a6f01abeddd71ba7004a78bb8734.jpg)

![f3dfbfe011ebd6290378b48180b436198b6f8a1dcf51d22f9c345e5334b791e7.jpg](../iclr_results/213_Synthetic%20continued%20pretraining/tables/f3dfbfe011ebd6290378b48180b436198b6f8a1dcf51d22f9c345e5334b791e7.jpg)

## ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability


### Images

![26a1467f3bfc0b5642f31db89d03e804e936581a6dc4337485358c280d85b4c9.jpg](../iclr_results/214_ReDeEP_%20Detecting%20Hallucination%20in%20Retrieval-Augmented%20Generation%20via%20Mechanistic%20Interpretability/images/26a1467f3bfc0b5642f31db89d03e804e936581a6dc4337485358c280d85b4c9.jpg)

![54c5ac58ff8b48ca1f4b9bc8e1449227ecd790ac12e2d5c5f7762fe13c084fdd.jpg](../iclr_results/214_ReDeEP_%20Detecting%20Hallucination%20in%20Retrieval-Augmented%20Generation%20via%20Mechanistic%20Interpretability/images/54c5ac58ff8b48ca1f4b9bc8e1449227ecd790ac12e2d5c5f7762fe13c084fdd.jpg)

![624523c284370871ad998562b0e43c6fe4245e8d2eda29108a87802e6146bcf7.jpg](../iclr_results/214_ReDeEP_%20Detecting%20Hallucination%20in%20Retrieval-Augmented%20Generation%20via%20Mechanistic%20Interpretability/images/624523c284370871ad998562b0e43c6fe4245e8d2eda29108a87802e6146bcf7.jpg)

![69247100825a10bcdf044142150a72e5d5d27b8d38e13934a86627ddcf295591.jpg](../iclr_results/214_ReDeEP_%20Detecting%20Hallucination%20in%20Retrieval-Augmented%20Generation%20via%20Mechanistic%20Interpretability/images/69247100825a10bcdf044142150a72e5d5d27b8d38e13934a86627ddcf295591.jpg)

![d37a1495d171a6552105185d188fc704ac550ca10bb9d970982253ff8fdd221d.jpg](../iclr_results/214_ReDeEP_%20Detecting%20Hallucination%20in%20Retrieval-Augmented%20Generation%20via%20Mechanistic%20Interpretability/images/d37a1495d171a6552105185d188fc704ac550ca10bb9d970982253ff8fdd221d.jpg)

![dba8fd2635bdd87e73972c4132b2887d57cc372a300d1db0c058ae7d013b40d7.jpg](../iclr_results/214_ReDeEP_%20Detecting%20Hallucination%20in%20Retrieval-Augmented%20Generation%20via%20Mechanistic%20Interpretability/images/dba8fd2635bdd87e73972c4132b2887d57cc372a300d1db0c058ae7d013b40d7.jpg)

![e0e0da6c2685d1dad711e6c558823812499bb27615a0b4900f265533b7e79d2c.jpg](../iclr_results/214_ReDeEP_%20Detecting%20Hallucination%20in%20Retrieval-Augmented%20Generation%20via%20Mechanistic%20Interpretability/images/e0e0da6c2685d1dad711e6c558823812499bb27615a0b4900f265533b7e79d2c.jpg)

![e4720467f11e053b2dd0ec5d1b3547b4747ad46432fe35b1fdea18704e1a9865.jpg](../iclr_results/214_ReDeEP_%20Detecting%20Hallucination%20in%20Retrieval-Augmented%20Generation%20via%20Mechanistic%20Interpretability/images/e4720467f11e053b2dd0ec5d1b3547b4747ad46432fe35b1fdea18704e1a9865.jpg)

![f46b12c60d9b34a87ac7d905eb220f2d37230d6434eaf82def67d57bf56274ac.jpg](../iclr_results/214_ReDeEP_%20Detecting%20Hallucination%20in%20Retrieval-Augmented%20Generation%20via%20Mechanistic%20Interpretability/images/f46b12c60d9b34a87ac7d905eb220f2d37230d6434eaf82def67d57bf56274ac.jpg)

### Tables

![622c6f77f42e8ebdbd039ffeb796fb0abbad2a44ab7cdcd0ec9ad8273e4f147d.jpg](../iclr_results/214_ReDeEP_%20Detecting%20Hallucination%20in%20Retrieval-Augmented%20Generation%20via%20Mechanistic%20Interpretability/tables/622c6f77f42e8ebdbd039ffeb796fb0abbad2a44ab7cdcd0ec9ad8273e4f147d.jpg)

![86253e3f3d676ea0f2074f05d7acb852a574458006e5241d2823a81d108d6123.jpg](../iclr_results/214_ReDeEP_%20Detecting%20Hallucination%20in%20Retrieval-Augmented%20Generation%20via%20Mechanistic%20Interpretability/tables/86253e3f3d676ea0f2074f05d7acb852a574458006e5241d2823a81d108d6123.jpg)

![9458c4937a3628831d2b33479a6ca62e3da4fa71ae72504174575fee5217c340.jpg](../iclr_results/214_ReDeEP_%20Detecting%20Hallucination%20in%20Retrieval-Augmented%20Generation%20via%20Mechanistic%20Interpretability/tables/9458c4937a3628831d2b33479a6ca62e3da4fa71ae72504174575fee5217c340.jpg)

![aca8b1bacb60556b6fc672e54dc54bac056fc757be1ca842f3746986c19e1caa.jpg](../iclr_results/214_ReDeEP_%20Detecting%20Hallucination%20in%20Retrieval-Augmented%20Generation%20via%20Mechanistic%20Interpretability/tables/aca8b1bacb60556b6fc672e54dc54bac056fc757be1ca842f3746986c19e1caa.jpg)

![c5ae78fddba7cba3b701ff82461e74fe43727717dce8af59d07b2bb81a424f77.jpg](../iclr_results/214_ReDeEP_%20Detecting%20Hallucination%20in%20Retrieval-Augmented%20Generation%20via%20Mechanistic%20Interpretability/tables/c5ae78fddba7cba3b701ff82461e74fe43727717dce8af59d07b2bb81a424f77.jpg)

![dbb1fb7f73f7fc77d79dac3033035bdedf3c49724f4414887fcbbfaef7855197.jpg](../iclr_results/214_ReDeEP_%20Detecting%20Hallucination%20in%20Retrieval-Augmented%20Generation%20via%20Mechanistic%20Interpretability/tables/dbb1fb7f73f7fc77d79dac3033035bdedf3c49724f4414887fcbbfaef7855197.jpg)

## LOKI: A Comprehensive Synthetic Data Detection Benchmark using Large Multimodal Models


### Images

![074b03c5d83b0b5575c7376567f44b5eb6799c7804ee67a13ea97e42f41ee599.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/074b03c5d83b0b5575c7376567f44b5eb6799c7804ee67a13ea97e42f41ee599.jpg)

![0802da89803c2a539ee7bfa8d61fe4a58259d88eb03b0c9dedd283784fe4c058.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/0802da89803c2a539ee7bfa8d61fe4a58259d88eb03b0c9dedd283784fe4c058.jpg)

![0c2132bb2f4b167b73c3cabbdf7237df202846d079d76ffe5f758211f7d7a47f.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/0c2132bb2f4b167b73c3cabbdf7237df202846d079d76ffe5f758211f7d7a47f.jpg)

![0d8080bc77023d7ae78865f6aa7f4d767ab18eec438962ee5d4ba27bd6b6f43a.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/0d8080bc77023d7ae78865f6aa7f4d767ab18eec438962ee5d4ba27bd6b6f43a.jpg)

![0e51ea75760fb2665f7eb9f7ee2bf572139343cc238b9fbbd949b21d3e16cf5e.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/0e51ea75760fb2665f7eb9f7ee2bf572139343cc238b9fbbd949b21d3e16cf5e.jpg)

![141385540e4af5c0990984efae0b46cc16e04227afe75962c23b91d3234e1a24.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/141385540e4af5c0990984efae0b46cc16e04227afe75962c23b91d3234e1a24.jpg)

![158f018f288a5f8c38d67bd517059f4cdf81a61d12ab3ae38b028c7981e49a5f.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/158f018f288a5f8c38d67bd517059f4cdf81a61d12ab3ae38b028c7981e49a5f.jpg)

![18414a31914aafbcfdb415383523f1490456d283712d64b068c94e3c3af7c060.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/18414a31914aafbcfdb415383523f1490456d283712d64b068c94e3c3af7c060.jpg)

![1c719da8eeaf902aa848937a144026b181be309dd2cf7cbede8c8d4503addd02.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/1c719da8eeaf902aa848937a144026b181be309dd2cf7cbede8c8d4503addd02.jpg)

![24b9df1363ccf20f56e2e3f20d1ae5e9f29c87aa3ef119db936661731709c4e6.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/24b9df1363ccf20f56e2e3f20d1ae5e9f29c87aa3ef119db936661731709c4e6.jpg)

![2a612b729a32f7ec70f046a2df7e1936159b1bc604f551ce63e93f7c7d4c23a2.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/2a612b729a32f7ec70f046a2df7e1936159b1bc604f551ce63e93f7c7d4c23a2.jpg)

![2c7b66ae1144e5edcad9a43bce35c0f6114e66e3b0f693e0547fd32650d4f2ce.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/2c7b66ae1144e5edcad9a43bce35c0f6114e66e3b0f693e0547fd32650d4f2ce.jpg)

![2ccb85aeaf38a76d773f9bb4b21078d8615863c5edccf9e67200c742a4e61159.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/2ccb85aeaf38a76d773f9bb4b21078d8615863c5edccf9e67200c742a4e61159.jpg)

![3c2999a1bc0e6b6bb11f3d51a033631eaea29e85450dee6dc0f0f192c4d49a6c.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/3c2999a1bc0e6b6bb11f3d51a033631eaea29e85450dee6dc0f0f192c4d49a6c.jpg)

![3cb7eb86297de49f6015d8df7a2125d1879b4b5dcf268d3e252b84f8f4f78386.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/3cb7eb86297de49f6015d8df7a2125d1879b4b5dcf268d3e252b84f8f4f78386.jpg)

![4098636c4b58654b5c2f9033c9bcb4e4dfe9281c82c2aff9ebfe787d0f2d6a1a.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/4098636c4b58654b5c2f9033c9bcb4e4dfe9281c82c2aff9ebfe787d0f2d6a1a.jpg)

![478afe7312e3521fcbe1e0ed5fad63ad65b4ae4390820ce66e21cc02c2164dff.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/478afe7312e3521fcbe1e0ed5fad63ad65b4ae4390820ce66e21cc02c2164dff.jpg)

![47dcfb65ad7996d5b0b6da83d3be078a8393fbaa86a2a1b77902fbb8a317a97f.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/47dcfb65ad7996d5b0b6da83d3be078a8393fbaa86a2a1b77902fbb8a317a97f.jpg)

![482984387467e52f87a5858f46ce2a6a374b2039ed47be956a59e9f8bc35911c.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/482984387467e52f87a5858f46ce2a6a374b2039ed47be956a59e9f8bc35911c.jpg)

![4be347e666963d3bdf73f990abf88eef0472f1ac076e2b45c209bd678176f3e3.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/4be347e666963d3bdf73f990abf88eef0472f1ac076e2b45c209bd678176f3e3.jpg)

![4c54d0605feb9ba8e42ee208fbb3ceae6061f08b29c8b134be81a47cbf98246e.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/4c54d0605feb9ba8e42ee208fbb3ceae6061f08b29c8b134be81a47cbf98246e.jpg)

![4ceb4639b52aeae9ded1fead11510d5795bf292e3b10396a165c620e67284337.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/4ceb4639b52aeae9ded1fead11510d5795bf292e3b10396a165c620e67284337.jpg)

![501b20ebe707b8628dd804726f2ad9cc15d77bd03aed7277f41324c0eba84699.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/501b20ebe707b8628dd804726f2ad9cc15d77bd03aed7277f41324c0eba84699.jpg)

![58fb8a45f9a35e8db0b4a66e1845fddb027726e3497c290eeb8debe0680372f1.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/58fb8a45f9a35e8db0b4a66e1845fddb027726e3497c290eeb8debe0680372f1.jpg)

![5b4d71b56aac5b4f79ce2951a0eff673df06074fef84c73c5808d7037d3f235d.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/5b4d71b56aac5b4f79ce2951a0eff673df06074fef84c73c5808d7037d3f235d.jpg)

![5f3be6675bff49204aa1345096d61d3264cf61bfdd0d475d3eb0b329c6135f94.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/5f3be6675bff49204aa1345096d61d3264cf61bfdd0d475d3eb0b329c6135f94.jpg)

![62e68aabe934a7758511123b90436d520fae639304d47dc40340f0be8e49f518.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/62e68aabe934a7758511123b90436d520fae639304d47dc40340f0be8e49f518.jpg)

![63743ef7758d6737ab585122389d788c3538554f337d8be561044262cc0dfced.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/63743ef7758d6737ab585122389d788c3538554f337d8be561044262cc0dfced.jpg)

![6eaa36ba79fbaaa88cf5edca5b72d82bc27c7791c1fff3a2a99579683cf54621.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/6eaa36ba79fbaaa88cf5edca5b72d82bc27c7791c1fff3a2a99579683cf54621.jpg)

![7739a9603283e60a4a76acb866cc9d8e4ad055e1d359f5e9545bb1faf8dcb066.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/7739a9603283e60a4a76acb866cc9d8e4ad055e1d359f5e9545bb1faf8dcb066.jpg)

![7bac412d74c2ab675200bda7b94d89a7f29ef2180f98b1e4d419a57a89903f0f.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/7bac412d74c2ab675200bda7b94d89a7f29ef2180f98b1e4d419a57a89903f0f.jpg)

![84489ea416ecebb61063f810e193da4450e0fa0dd098ed1041fe66dc99ffd313.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/84489ea416ecebb61063f810e193da4450e0fa0dd098ed1041fe66dc99ffd313.jpg)

![8db7023d4f2916c8ad6e456de0df6e621cd0dc96c47b8620f2bc8e9b6fe867f2.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/8db7023d4f2916c8ad6e456de0df6e621cd0dc96c47b8620f2bc8e9b6fe867f2.jpg)

![96ae7c52d8c7072c1b82a15a703f9e0f032c3d291f4da569b803e0c264fb92be.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/96ae7c52d8c7072c1b82a15a703f9e0f032c3d291f4da569b803e0c264fb92be.jpg)

![98137302631e75c46b5dce6f9f575d5a82b36c9593bd534b0c626a141ec6c736.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/98137302631e75c46b5dce6f9f575d5a82b36c9593bd534b0c626a141ec6c736.jpg)

![9ef5964dcd7483293476a7f2f520b6ef34926edf6261e639a9e75c32e6ee35fc.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/9ef5964dcd7483293476a7f2f520b6ef34926edf6261e639a9e75c32e6ee35fc.jpg)

![9efa181d300647c68bd151dc47fef5686301a48a844b6b8dca97ebfa270f1c5d.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/9efa181d300647c68bd151dc47fef5686301a48a844b6b8dca97ebfa270f1c5d.jpg)

![a37e8b0782a3b9f020207010764d962773b950c6dd83e41f7f99458ccc4c1b92.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/a37e8b0782a3b9f020207010764d962773b950c6dd83e41f7f99458ccc4c1b92.jpg)

![a835be5cbfdfe7b8d3b9b45ac1a3a4126a0802dac3d2deec72a873dd36dfa931.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/a835be5cbfdfe7b8d3b9b45ac1a3a4126a0802dac3d2deec72a873dd36dfa931.jpg)

![aef9ddb5fef502aa93d45f2864f2c2d66737ef0de644df406a8138e174e26d11.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/aef9ddb5fef502aa93d45f2864f2c2d66737ef0de644df406a8138e174e26d11.jpg)

![c37160c2445ef76ff8abf7706aa0a745de74649eea3e948f090894f4d174ff8f.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/c37160c2445ef76ff8abf7706aa0a745de74649eea3e948f090894f4d174ff8f.jpg)

![c76ae84d21558390d938e5075023efca0bc3700d56c7cfab099a0acf4449b701.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/c76ae84d21558390d938e5075023efca0bc3700d56c7cfab099a0acf4449b701.jpg)

![c8e658567c03a87277ca0d7cccbb3f08849fbe6608d9c436d2dca74bba4a15aa.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/c8e658567c03a87277ca0d7cccbb3f08849fbe6608d9c436d2dca74bba4a15aa.jpg)

![cbd991c9a650cd91219a247331e109850a957e655d7b2b2b95084b3f7a0c356a.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/cbd991c9a650cd91219a247331e109850a957e655d7b2b2b95084b3f7a0c356a.jpg)

![cebd653b0c56d49d7de9e268eeef72d32872e0a65e3e2bfbdbbc448124ae902a.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/cebd653b0c56d49d7de9e268eeef72d32872e0a65e3e2bfbdbbc448124ae902a.jpg)

![cf970b0a2cc9ed22298d392c241e364687b41a5cf56cf3b2bf7986eff261a01e.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/cf970b0a2cc9ed22298d392c241e364687b41a5cf56cf3b2bf7986eff261a01e.jpg)

![d3bcb6c79e2b9e53d9c022b2784b2918ad45852e2ab48455ad9ffecc3322020b.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/d3bcb6c79e2b9e53d9c022b2784b2918ad45852e2ab48455ad9ffecc3322020b.jpg)

![d432ab166eb2e65955a67ce2964b1b511cbc22016fc2a85e1b61d584a26f4e4b.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/d432ab166eb2e65955a67ce2964b1b511cbc22016fc2a85e1b61d584a26f4e4b.jpg)

![d44b3d305c2fbc13a6ab1958b1c15829d0987e410456a3bccebb17dcecccb67c.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/d44b3d305c2fbc13a6ab1958b1c15829d0987e410456a3bccebb17dcecccb67c.jpg)

![e5c2ccafb52123c9a574dfcc8281761c7d3591dc39a8ca1fc45fddd196c35334.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/e5c2ccafb52123c9a574dfcc8281761c7d3591dc39a8ca1fc45fddd196c35334.jpg)

![e7825ae966dd7496c279e5377b854e8a4cada076437765d50005af3ac2ea00d7.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/e7825ae966dd7496c279e5377b854e8a4cada076437765d50005af3ac2ea00d7.jpg)

![f046bd0e6b60f79838d97388169b7d602b4e77edfc8c38daf6315728fa3cb89e.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/f046bd0e6b60f79838d97388169b7d602b4e77edfc8c38daf6315728fa3cb89e.jpg)

![f4ad058dd9b705f034f2da4d38d417f8b802110303ae9b1ecea916a9528b445b.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/f4ad058dd9b705f034f2da4d38d417f8b802110303ae9b1ecea916a9528b445b.jpg)

![f59a75a4bc695febbb6f8f172b8aa9349c9f50e20413ab475ea6f125e3f14734.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/f59a75a4bc695febbb6f8f172b8aa9349c9f50e20413ab475ea6f125e3f14734.jpg)

![f9cacbf4b7660979330492e4143307dc653bb0e0891014e014ae46eecf50f0d9.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/images/f9cacbf4b7660979330492e4143307dc653bb0e0891014e014ae46eecf50f0d9.jpg)

### Tables

![0d47c24beba21aa39201eab1ee7f6ca1d238aa8b2d2077b6097be1f1dd74497a.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/0d47c24beba21aa39201eab1ee7f6ca1d238aa8b2d2077b6097be1f1dd74497a.jpg)

![0f071873aa6d8f49184cda59e8a6107c541ac955ad0b229f556656d57ef0f157.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/0f071873aa6d8f49184cda59e8a6107c541ac955ad0b229f556656d57ef0f157.jpg)

![3798855884ffdbb333eac822523beabd8c82637e74652d2da4a3f9451119dfba.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/3798855884ffdbb333eac822523beabd8c82637e74652d2da4a3f9451119dfba.jpg)

![3af61ff68c4a7b27b4f2ab43a23297c6b82c0fad4744ed02438a7efaf512ea62.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/3af61ff68c4a7b27b4f2ab43a23297c6b82c0fad4744ed02438a7efaf512ea62.jpg)

![4b4da119484e3e00c7739f24fe0854d78fd6a9f79dc731e5b1c6f5072e6ca307.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/4b4da119484e3e00c7739f24fe0854d78fd6a9f79dc731e5b1c6f5072e6ca307.jpg)

![5614eb28c5c8e6d571a63f070a207371a6bf9d5b34f2f3cd6a73514842174a58.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/5614eb28c5c8e6d571a63f070a207371a6bf9d5b34f2f3cd6a73514842174a58.jpg)

![657ff7217572adb2fb8de659398274003cbf684dbf9148116c33df3eeb4b9d47.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/657ff7217572adb2fb8de659398274003cbf684dbf9148116c33df3eeb4b9d47.jpg)

![6c173ab05293a52ec2cb7e7a66cc240a232e606654fd709f19f783c122271f25.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/6c173ab05293a52ec2cb7e7a66cc240a232e606654fd709f19f783c122271f25.jpg)

![6e9971a8d21b5344a46051344a87d4d3624cf74272eeabe292327df5328acf59.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/6e9971a8d21b5344a46051344a87d4d3624cf74272eeabe292327df5328acf59.jpg)

![709817e80b446697ce9226f41750dabfa94df9148baa0d1134759538b93b9036.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/709817e80b446697ce9226f41750dabfa94df9148baa0d1134759538b93b9036.jpg)

![71f56dc9a3cff74304462ca4043262534898fd875843936344727fedb4288602.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/71f56dc9a3cff74304462ca4043262534898fd875843936344727fedb4288602.jpg)

![7491bd383014ab726754c56ffe2b230e5542ad5a4c817efe0ed4ced11e4ac4cf.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/7491bd383014ab726754c56ffe2b230e5542ad5a4c817efe0ed4ced11e4ac4cf.jpg)

![7749931f3cba41e706ea151eacef53d7a8e96a7ac739aa43475dab32ef47d465.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/7749931f3cba41e706ea151eacef53d7a8e96a7ac739aa43475dab32ef47d465.jpg)

![77dee98e2e6a9502010379e205f9467e09b0931ca6f88e0575aad96a315d6448.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/77dee98e2e6a9502010379e205f9467e09b0931ca6f88e0575aad96a315d6448.jpg)

![8289fd5cb9d15bce48f1b7cb9064f189ab65c9cde940547ca8843352aa65aff9.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/8289fd5cb9d15bce48f1b7cb9064f189ab65c9cde940547ca8843352aa65aff9.jpg)

![843cbd09db8208e1eeddb3c844fa566b66dc92f7f36ddbef9fcf941696e01fda.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/843cbd09db8208e1eeddb3c844fa566b66dc92f7f36ddbef9fcf941696e01fda.jpg)

![aa919f47a682291ca93fc0e4015f7b957d5f53ea94010bf7f8db569b0d249ba5.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/aa919f47a682291ca93fc0e4015f7b957d5f53ea94010bf7f8db569b0d249ba5.jpg)

![acc05ab3a2ddd599768f5359e234a004cc9146d54a01491809aa982c48ad02b0.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/acc05ab3a2ddd599768f5359e234a004cc9146d54a01491809aa982c48ad02b0.jpg)

![af9230c9cdb714f3c5520a3c6261a9511db4d947658b6aec011f0562e8aef9ce.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/af9230c9cdb714f3c5520a3c6261a9511db4d947658b6aec011f0562e8aef9ce.jpg)

![b333a0030ac1aa6ea5aab340ea00c5c6c7378c8386c3701c48b442351e4b8adb.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/b333a0030ac1aa6ea5aab340ea00c5c6c7378c8386c3701c48b442351e4b8adb.jpg)

![b4bd50f69d1c4e60c1bba29a695514988326aaeff00a0356038bb5310eb322e1.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/b4bd50f69d1c4e60c1bba29a695514988326aaeff00a0356038bb5310eb322e1.jpg)

![bfd93da3bf50ecb5d257b45fa3637913e9d94d094062856a253dd0723da7acb5.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/bfd93da3bf50ecb5d257b45fa3637913e9d94d094062856a253dd0723da7acb5.jpg)

![cdac4f40cea257be9254b38cb4e24074d660f05e34a36f2ca7a63d1ea3b4f063.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/cdac4f40cea257be9254b38cb4e24074d660f05e34a36f2ca7a63d1ea3b4f063.jpg)

![e2d96a881fc8e3b91232533bd0c4ff1d1de584f02855d2f4333e62440848568a.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/e2d96a881fc8e3b91232533bd0c4ff1d1de584f02855d2f4333e62440848568a.jpg)

![e912926b6a4f2e3dc1270484e2ad6646ad4108bcacd0ef99036c3ca445f95646.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/e912926b6a4f2e3dc1270484e2ad6646ad4108bcacd0ef99036c3ca445f95646.jpg)

![f48cfc2473b76bfcb25355ff4c764d3a02f350f7e1f816604eee7dc8955adc98.jpg](../iclr_results/215_LOKI_%20A%20Comprehensive%20Synthetic%20Data%20Detection%20Benchmark%20using%20Large%20Multimodal%20Models/tables/f48cfc2473b76bfcb25355ff4c764d3a02f350f7e1f816604eee7dc8955adc98.jpg)

## Revisiting Zeroth-Order Optimization:  Minimum-Variance Two-Point Estimators and  Directionally Aligned Perturbations


### Images

![00b87c39c10cd48814c2b410cfbb3ea8f402f0b3c25d839c73ac79a095f12820.jpg](../iclr_results/216_Revisiting%20Zeroth-Order%20Optimization_%20%20Minimum-Variance%20Two-Point%20Estimators%20and%20%20Directionally%20Alig/images/00b87c39c10cd48814c2b410cfbb3ea8f402f0b3c25d839c73ac79a095f12820.jpg)

![18184f0202ec17c61c1ba0964e6dca038be28cfc09c2930d7771b2a1e04125f4.jpg](../iclr_results/216_Revisiting%20Zeroth-Order%20Optimization_%20%20Minimum-Variance%20Two-Point%20Estimators%20and%20%20Directionally%20Alig/images/18184f0202ec17c61c1ba0964e6dca038be28cfc09c2930d7771b2a1e04125f4.jpg)

![2f76831e6eaf521b56640102300d563fc7ad55151d8e0ffa0256d67b56570cee.jpg](../iclr_results/216_Revisiting%20Zeroth-Order%20Optimization_%20%20Minimum-Variance%20Two-Point%20Estimators%20and%20%20Directionally%20Alig/images/2f76831e6eaf521b56640102300d563fc7ad55151d8e0ffa0256d67b56570cee.jpg)

![31887fb25cb0162bd3bd930120f196122e230222a3d7ef387e7fc19f1c1c797a.jpg](../iclr_results/216_Revisiting%20Zeroth-Order%20Optimization_%20%20Minimum-Variance%20Two-Point%20Estimators%20and%20%20Directionally%20Alig/images/31887fb25cb0162bd3bd930120f196122e230222a3d7ef387e7fc19f1c1c797a.jpg)

![afbb669a61b8407e777e28bf307e67435731d4102b766826ca09351b0a258629.jpg](../iclr_results/216_Revisiting%20Zeroth-Order%20Optimization_%20%20Minimum-Variance%20Two-Point%20Estimators%20and%20%20Directionally%20Alig/images/afbb669a61b8407e777e28bf307e67435731d4102b766826ca09351b0a258629.jpg)

![ba9d23a8c32d79b5c64b03a01ae824a4fb8210cfa035059cb5677b888120df78.jpg](../iclr_results/216_Revisiting%20Zeroth-Order%20Optimization_%20%20Minimum-Variance%20Two-Point%20Estimators%20and%20%20Directionally%20Alig/images/ba9d23a8c32d79b5c64b03a01ae824a4fb8210cfa035059cb5677b888120df78.jpg)

![cf5fabd006489c12c3480886fb9284b6cc3d0544e17404466c5ae565ccf901b7.jpg](../iclr_results/216_Revisiting%20Zeroth-Order%20Optimization_%20%20Minimum-Variance%20Two-Point%20Estimators%20and%20%20Directionally%20Alig/images/cf5fabd006489c12c3480886fb9284b6cc3d0544e17404466c5ae565ccf901b7.jpg)

![f8c28768acde293445133150c8fd26416c9f2ad12939b82a73c5f3215060dfb6.jpg](../iclr_results/216_Revisiting%20Zeroth-Order%20Optimization_%20%20Minimum-Variance%20Two-Point%20Estimators%20and%20%20Directionally%20Alig/images/f8c28768acde293445133150c8fd26416c9f2ad12939b82a73c5f3215060dfb6.jpg)

![ff6795717f70d1acce42b63474b12954bc24eed0f1e07dfd7798563c40d80305.jpg](../iclr_results/216_Revisiting%20Zeroth-Order%20Optimization_%20%20Minimum-Variance%20Two-Point%20Estimators%20and%20%20Directionally%20Alig/images/ff6795717f70d1acce42b63474b12954bc24eed0f1e07dfd7798563c40d80305.jpg)

### Tables

![5b4b61182194bc400b88204d2bd9b77eeb14545a998c0ead732beed3fd5eca28.jpg](../iclr_results/216_Revisiting%20Zeroth-Order%20Optimization_%20%20Minimum-Variance%20Two-Point%20Estimators%20and%20%20Directionally%20Alig/tables/5b4b61182194bc400b88204d2bd9b77eeb14545a998c0ead732beed3fd5eca28.jpg)

![d03a80640a9c84a4acfa27e6f2202ddd674ecfd7cfcd3fb96b767cb6f3abc6fe.jpg](../iclr_results/216_Revisiting%20Zeroth-Order%20Optimization_%20%20Minimum-Variance%20Two-Point%20Estimators%20and%20%20Directionally%20Alig/tables/d03a80640a9c84a4acfa27e6f2202ddd674ecfd7cfcd3fb96b767cb6f3abc6fe.jpg)

## BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval


### Images

![b48537f24a5949cd0f3bff2557e02af5b5c836750a0d3e10b7eb753715e51319.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/images/b48537f24a5949cd0f3bff2557e02af5b5c836750a0d3e10b7eb753715e51319.jpg)

![d0ce6709c79e77f69340cec277e649c0058b9851f741adb89f336038e0722d9c.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/images/d0ce6709c79e77f69340cec277e649c0058b9851f741adb89f336038e0722d9c.jpg)

![fc7e52c57877e5d61a1f44c0c38094ea676c4c8facd0360243db29c43f8bc0f9.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/images/fc7e52c57877e5d61a1f44c0c38094ea676c4c8facd0360243db29c43f8bc0f9.jpg)

### Tables

![0008f447afb52a3aaabc787dad01a2c7996aaa2fc58da84fb8c7998edf94686a.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/0008f447afb52a3aaabc787dad01a2c7996aaa2fc58da84fb8c7998edf94686a.jpg)

![0522d09fe08e0722e1c9316f5a17aae8d8faac04c863a69f120ef1c929f4aa53.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/0522d09fe08e0722e1c9316f5a17aae8d8faac04c863a69f120ef1c929f4aa53.jpg)

![1a978867ef9672255e807364a28326dc80003ff640121b6902017f4b7dd0998a.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/1a978867ef9672255e807364a28326dc80003ff640121b6902017f4b7dd0998a.jpg)

![1b7be5d8fbb3eabea53fbcd137fc38f18d56e4bfc544337ea9d75f2e402ca767.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/1b7be5d8fbb3eabea53fbcd137fc38f18d56e4bfc544337ea9d75f2e402ca767.jpg)

![1c1e62c2e3a81e8a4b301d8f81bff60fc51755657ae10e0855cbfc164a17c3d1.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/1c1e62c2e3a81e8a4b301d8f81bff60fc51755657ae10e0855cbfc164a17c3d1.jpg)

![2290589088b47146b0a55095fef04a948efbb5932d3e9503067bae089a8098b6.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/2290589088b47146b0a55095fef04a948efbb5932d3e9503067bae089a8098b6.jpg)

![22c8db7e79c4e32fb60c60459abb5db047ee197e0fab4ce140f2b852779d031e.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/22c8db7e79c4e32fb60c60459abb5db047ee197e0fab4ce140f2b852779d031e.jpg)

![240d1a038cbae346cf74be04d4fa75932209e52345a6c396f7d3d8995f14a81f.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/240d1a038cbae346cf74be04d4fa75932209e52345a6c396f7d3d8995f14a81f.jpg)

![2505cfb89730b8768b3ee37df89f34253c42d0392ea0b1bf46a61f10a97604f6.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/2505cfb89730b8768b3ee37df89f34253c42d0392ea0b1bf46a61f10a97604f6.jpg)

![260f85e51b5865e44a5c7a4ddb64ff167964796d22e3b3eedd1391afffac5427.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/260f85e51b5865e44a5c7a4ddb64ff167964796d22e3b3eedd1391afffac5427.jpg)

![3a6a0ac63349e9a2c5a6f893fa299dca7dd29442317c580a54ad457a33fc2dfa.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/3a6a0ac63349e9a2c5a6f893fa299dca7dd29442317c580a54ad457a33fc2dfa.jpg)

![402a061ca1c5d52c033c0abbbde98c3d456b0bd9406f89e809befce0a8ca2b0b.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/402a061ca1c5d52c033c0abbbde98c3d456b0bd9406f89e809befce0a8ca2b0b.jpg)

![45a5e3f18b762e6ed7d7f3f76830d6d4f3054657728837afe364018fb930639e.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/45a5e3f18b762e6ed7d7f3f76830d6d4f3054657728837afe364018fb930639e.jpg)

![45edb0ef610f1a0fb7e24d7f3f4b343f8202dd011593c0a2afa84f036c93e6cb.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/45edb0ef610f1a0fb7e24d7f3f4b343f8202dd011593c0a2afa84f036c93e6cb.jpg)

![4d67ccaeda745de5db03aba16149e6fa3856d5a41f804baf6e56a7b3c7f93958.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/4d67ccaeda745de5db03aba16149e6fa3856d5a41f804baf6e56a7b3c7f93958.jpg)

![4f4842f3ce3e7fd156815fa6d676d15eb1c2112343af06dbefe97e8d60428f9f.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/4f4842f3ce3e7fd156815fa6d676d15eb1c2112343af06dbefe97e8d60428f9f.jpg)

![545184b64feacd83750e35a4977cd271423df296ee75342cb0f40d647855deb9.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/545184b64feacd83750e35a4977cd271423df296ee75342cb0f40d647855deb9.jpg)

![54c55b776badd04814f2604a2550e04e36ea54ab693363e417026698c4b89d69.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/54c55b776badd04814f2604a2550e04e36ea54ab693363e417026698c4b89d69.jpg)

![560342077f93579c0d2fef345ce9b296a63eea6732a5dc09cf33fcf84152c878.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/560342077f93579c0d2fef345ce9b296a63eea6732a5dc09cf33fcf84152c878.jpg)

![669fcad125928eadc9ad8df1214bf5bc9db493d2d56f4cb4436c955da3047ef1.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/669fcad125928eadc9ad8df1214bf5bc9db493d2d56f4cb4436c955da3047ef1.jpg)

![6c726cb649fe9f021704d5f7e017c383ac7a02bff81bcb4693c94b23b41e7694.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/6c726cb649fe9f021704d5f7e017c383ac7a02bff81bcb4693c94b23b41e7694.jpg)

![6f26af633c95b472c96bd2e5ed39fdb1f741ae5f608dc2f5a8c64e9191d28c6c.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/6f26af633c95b472c96bd2e5ed39fdb1f741ae5f608dc2f5a8c64e9191d28c6c.jpg)

![72e77e9bb4f37231dc2aca71c5304dee6092780e4c6caa4f23360de676acf337.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/72e77e9bb4f37231dc2aca71c5304dee6092780e4c6caa4f23360de676acf337.jpg)

![73ba6b7b809cfd53de5cbe99ad1dd084b574ec69e9fc83b25aaa23ec45a1ec35.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/73ba6b7b809cfd53de5cbe99ad1dd084b574ec69e9fc83b25aaa23ec45a1ec35.jpg)

![774fbd9fce94ce6fc5c50f689081325c02da1dc5f2da66db9bb73e3ac297b0cb.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/774fbd9fce94ce6fc5c50f689081325c02da1dc5f2da66db9bb73e3ac297b0cb.jpg)

![7f52dda35e67b56c1204d58f74a5ed0230163defea8e2d914043e99f3353c65a.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/7f52dda35e67b56c1204d58f74a5ed0230163defea8e2d914043e99f3353c65a.jpg)

![7fca2e8a309abbfd43ac778987cdb019c14d1695661a806266f8b3ecbc36b8f4.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/7fca2e8a309abbfd43ac778987cdb019c14d1695661a806266f8b3ecbc36b8f4.jpg)

![81a11d2a725ff2f349edcfc277436b2e550e5ddb8940d8561d7839949f4c083f.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/81a11d2a725ff2f349edcfc277436b2e550e5ddb8940d8561d7839949f4c083f.jpg)

![887229cb4777808db49ef1bdfbd67980dd15661f6e54a8dc2451e81df4cf0a29.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/887229cb4777808db49ef1bdfbd67980dd15661f6e54a8dc2451e81df4cf0a29.jpg)

![8ace49b288b7c711d498d331225ff8e9e74c9e26684d3f81106cd488ac71b4cb.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/8ace49b288b7c711d498d331225ff8e9e74c9e26684d3f81106cd488ac71b4cb.jpg)

![a042d0881e4f5c102fa73aa5a2875421c33643b71a544b387aad999aa44ca206.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/a042d0881e4f5c102fa73aa5a2875421c33643b71a544b387aad999aa44ca206.jpg)

![a1ba68ff29333239ac9dbf0019c4550ce8face7b7f0f3efdfafd69e60861ad63.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/a1ba68ff29333239ac9dbf0019c4550ce8face7b7f0f3efdfafd69e60861ad63.jpg)

![ac68b8f2ae6040a7002c8936bd33cb43c90d52f7d86829d55e6a742426900ea7.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/ac68b8f2ae6040a7002c8936bd33cb43c90d52f7d86829d55e6a742426900ea7.jpg)

![acb13766b76e219b686e819c81dc936e750ef86add6d50d319d8e9ff2b21d6a0.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/acb13766b76e219b686e819c81dc936e750ef86add6d50d319d8e9ff2b21d6a0.jpg)

![aee05aa05f65e3fae56d18f5c09a016f21c54673d62e401708d3a648547d36fd.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/aee05aa05f65e3fae56d18f5c09a016f21c54673d62e401708d3a648547d36fd.jpg)

![b3a02f6710273f5adbc6b288a95c27ce94904d1a4f2d5d1a3c1ce04c65aa17c0.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/b3a02f6710273f5adbc6b288a95c27ce94904d1a4f2d5d1a3c1ce04c65aa17c0.jpg)

![b43803365e02ae91119653ce16c17b4a03ed39b0dcb46a0cbc0b1848d0a25f80.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/b43803365e02ae91119653ce16c17b4a03ed39b0dcb46a0cbc0b1848d0a25f80.jpg)

![b9611af4fb9c1df48f80830e63171fa520c090f1040e0c4556c95aca80495615.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/b9611af4fb9c1df48f80830e63171fa520c090f1040e0c4556c95aca80495615.jpg)

![bc988ae37b7939e2a3b3376e9bdbd192b508a6c33d4c192438013637413e87d9.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/bc988ae37b7939e2a3b3376e9bdbd192b508a6c33d4c192438013637413e87d9.jpg)

![c0ed5e6cfe1d762a19ca511ef0803ea734969a955addecd0ee017cd65d72b12c.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/c0ed5e6cfe1d762a19ca511ef0803ea734969a955addecd0ee017cd65d72b12c.jpg)

![cceb9738fb2077faa557fa99bb5cb2fb7b1672ed982328b4484ec0e17664968e.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/cceb9738fb2077faa557fa99bb5cb2fb7b1672ed982328b4484ec0e17664968e.jpg)

![d3bf64213d51359e7eb2a11f41d72a8d489f83eac1fdbdd9a3f1c00a756fdcb5.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/d3bf64213d51359e7eb2a11f41d72a8d489f83eac1fdbdd9a3f1c00a756fdcb5.jpg)

![e836b0e0e4df97f8710b50bb2fc8f68ecf8dabc08d0fbbf52837c70d48d4cc8c.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/e836b0e0e4df97f8710b50bb2fc8f68ecf8dabc08d0fbbf52837c70d48d4cc8c.jpg)

![f1c53f972975ca0842c41de6a062c2276543e487ff643e4ffb9ea80bab310d77.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/f1c53f972975ca0842c41de6a062c2276543e487ff643e4ffb9ea80bab310d77.jpg)

![fa870a3c19da3b1c5d805ed2ce294aa2ec525d92378dce15dd1fdcafc7aa62f9.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/fa870a3c19da3b1c5d805ed2ce294aa2ec525d92378dce15dd1fdcafc7aa62f9.jpg)

![fd12158d792863956732995f46ca118e1981a7e1ef1d84f5396b042e77663f3c.jpg](../iclr_results/217_BRIGHT_%20A%20Realistic%20and%20Challenging%20Benchmark%20for%20Reasoning-Intensive%20Retrieval/tables/fd12158d792863956732995f46ca118e1981a7e1ef1d84f5396b042e77663f3c.jpg)

## Adversarial Perturbations Cannot Reliably Protect Artists From Generative AI


### Images

![022f58400659807adb4f5a473c54cfab94579a7d752459f4e93a05961ef241af.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/022f58400659807adb4f5a473c54cfab94579a7d752459f4e93a05961ef241af.jpg)

![030b2de881f8513b08c711a819133a5edd5979f938220ce2a8491f6767e4f496.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/030b2de881f8513b08c711a819133a5edd5979f938220ce2a8491f6767e4f496.jpg)

![036874ae136fdba51805028a7f3b0baba1cfdc632b4b45320a8b1a2f40b3209e.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/036874ae136fdba51805028a7f3b0baba1cfdc632b4b45320a8b1a2f40b3209e.jpg)

![08366a8801d918994470d33b251ea4e8554886b50544af59a7774f61dfb17e38.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/08366a8801d918994470d33b251ea4e8554886b50544af59a7774f61dfb17e38.jpg)

![1815d585fe960d33c0561924cf9013869c486088203a48f952f387b1e8374780.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/1815d585fe960d33c0561924cf9013869c486088203a48f952f387b1e8374780.jpg)

![1c1bfe2ff5f17d88fa85a6deae22431ddb0bee6f9dae4a7a43c9e57c58c71beb.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/1c1bfe2ff5f17d88fa85a6deae22431ddb0bee6f9dae4a7a43c9e57c58c71beb.jpg)

![271e3e810e6519eff8108fd53ba37a4c059d8fbdb0fa79ac87cdc77f5095fe5f.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/271e3e810e6519eff8108fd53ba37a4c059d8fbdb0fa79ac87cdc77f5095fe5f.jpg)

![29ad1da671ba403aea93073247ccd856a631ad1b340e9bf9b03a553aa003cfc1.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/29ad1da671ba403aea93073247ccd856a631ad1b340e9bf9b03a553aa003cfc1.jpg)

![374819d9fb969108eca07df1905c5e56725834d942fa21f9a8fca845b398b053.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/374819d9fb969108eca07df1905c5e56725834d942fa21f9a8fca845b398b053.jpg)

![3daacee3213f187d47ecb082f6636038721b3edcd9c60560bb64286d95c498da.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/3daacee3213f187d47ecb082f6636038721b3edcd9c60560bb64286d95c498da.jpg)

![4728573e1e2c2e3e3b077c522808e6ef420864857f736c04b9fc78742ad4a8f3.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/4728573e1e2c2e3e3b077c522808e6ef420864857f736c04b9fc78742ad4a8f3.jpg)

![486651921d9714aa0277d2ba2f4b379df92f29c61223002a0576fd865b61e7e9.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/486651921d9714aa0277d2ba2f4b379df92f29c61223002a0576fd865b61e7e9.jpg)

![5848b2730a175faec331b40dd646801f42acecdf1b98b95b4d6d38cda22f1b62.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/5848b2730a175faec331b40dd646801f42acecdf1b98b95b4d6d38cda22f1b62.jpg)

![58f1b5ea8ac2c6b8f8e3914404ffe8117f38cc6df7fb2800b36f08335deead38.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/58f1b5ea8ac2c6b8f8e3914404ffe8117f38cc6df7fb2800b36f08335deead38.jpg)

![5f4ba6c0f574787be0c93f648cc24edb1f1a4c902bcdac7cd9d8dcca2cc0e1d3.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/5f4ba6c0f574787be0c93f648cc24edb1f1a4c902bcdac7cd9d8dcca2cc0e1d3.jpg)

![66a436decd69b0c5ba5680ad0ce3e2d269307a8a7de8d3809e36c9ad0c0ce548.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/66a436decd69b0c5ba5680ad0ce3e2d269307a8a7de8d3809e36c9ad0c0ce548.jpg)

![718eea66e27741790a8bac6b0cd676f137c7b2e60a4798c6691624d3c9055c67.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/718eea66e27741790a8bac6b0cd676f137c7b2e60a4798c6691624d3c9055c67.jpg)

![72a37eb62c33a316cd197107536f99c34dd2d0f3487696967fec1ed01e9998fb.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/72a37eb62c33a316cd197107536f99c34dd2d0f3487696967fec1ed01e9998fb.jpg)

![7f3f2f4ee9e73fa72e24982184718bbc3199231a6fc92f01ccb3c939184394fa.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/7f3f2f4ee9e73fa72e24982184718bbc3199231a6fc92f01ccb3c939184394fa.jpg)

![90ceb18f32701506a13340553ae77b91fe28de759c4c28dd5a9f402bfca64c8a.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/90ceb18f32701506a13340553ae77b91fe28de759c4c28dd5a9f402bfca64c8a.jpg)

![9c0a2fe832e5fb1d1ac4493590c1d5249816e2e9788c094fc254d2eae2838b24.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/9c0a2fe832e5fb1d1ac4493590c1d5249816e2e9788c094fc254d2eae2838b24.jpg)

![9e3210092787089823d6e0d6565f78923d0292223f59995e6afa409ef4bf0d00.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/9e3210092787089823d6e0d6565f78923d0292223f59995e6afa409ef4bf0d00.jpg)

![a529832bf5be1891f368cc3005cff7c893bddab4aeb7294b53af8ca65f443d50.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/a529832bf5be1891f368cc3005cff7c893bddab4aeb7294b53af8ca65f443d50.jpg)

![af1c698b7914847258389a78c501862854428dc59f7e83b37276a128787d354a.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/af1c698b7914847258389a78c501862854428dc59f7e83b37276a128787d354a.jpg)

![b769abf8e2cd7a867aecc16e73d8b56d3c1e76d842ac8897a138cbd779f52493.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/b769abf8e2cd7a867aecc16e73d8b56d3c1e76d842ac8897a138cbd779f52493.jpg)

![b83707958cb22ed3b5269047e97ad507ce52581997e1e36368ea32eed7b509f7.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/b83707958cb22ed3b5269047e97ad507ce52581997e1e36368ea32eed7b509f7.jpg)

![c139a1c0ae599f18b64a839819852ee40455df9b0b4ad7f6e935ec74ca4c4739.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/c139a1c0ae599f18b64a839819852ee40455df9b0b4ad7f6e935ec74ca4c4739.jpg)

![d9b1f213c8b10efc93eeab2734559b7308092e50885a38526aa6b956ea91d1fa.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/d9b1f213c8b10efc93eeab2734559b7308092e50885a38526aa6b956ea91d1fa.jpg)

![e4a6e7737e13667132e94a4ab05e68722459fb0e088d07b3db6db40925facf51.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/e4a6e7737e13667132e94a4ab05e68722459fb0e088d07b3db6db40925facf51.jpg)

![ea7216d64c93c2bc9f124702e50258db636fb69dde74033d8a778a47bf774cfc.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/ea7216d64c93c2bc9f124702e50258db636fb69dde74033d8a778a47bf774cfc.jpg)

![efeae461f0b05e31f78f182de868293933f5f08c6b18e18214ddfdb08c523537.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/efeae461f0b05e31f78f182de868293933f5f08c6b18e18214ddfdb08c523537.jpg)

![f17da874410e98f130aa08d4aa8dabd9b27733f4134f4a1f2cc0bea73d954d83.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/f17da874410e98f130aa08d4aa8dabd9b27733f4134f4a1f2cc0bea73d954d83.jpg)

![f9ae2c49dc5d9891c7c5f262492ee9f833d8efcabfe72ebc6906571cecffb3a0.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/images/f9ae2c49dc5d9891c7c5f262492ee9f833d8efcabfe72ebc6906571cecffb3a0.jpg)

### Tables

![47a579e60d244a5767850cb10020e4b17ea9b0088737a9eab339990a675c5ffb.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/tables/47a579e60d244a5767850cb10020e4b17ea9b0088737a9eab339990a675c5ffb.jpg)

![8b2d5529bd705472f44dca8212c0b2b2b18208de87edf3d0166c2d2a09312d86.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/tables/8b2d5529bd705472f44dca8212c0b2b2b18208de87edf3d0166c2d2a09312d86.jpg)

![c316130b9cd79e5450b08ef3ae74278211f33ceb31e100b30b312c05e6276b02.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/tables/c316130b9cd79e5450b08ef3ae74278211f33ceb31e100b30b312c05e6276b02.jpg)

![eb7f4dc9e98ae71c267f991cd97a6c2beecf2a7d48dbb9e5eb5fdd97a27d8e42.jpg](../iclr_results/218_Adversarial%20Perturbations%20Cannot%20Reliably%20Protect%20Artists%20From%20Generative%20AI/tables/eb7f4dc9e98ae71c267f991cd97a6c2beecf2a7d48dbb9e5eb5fdd97a27d8e42.jpg)

## MaRS: A Fast Sampler for Mean Reverting Diffusion based on ODE and SDE Solvers


### Images

![20b9a53f2ed5e846d344a03e57b1ecb7e25cfea9bba365b7b791fc6f41ad3fb0.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/images/20b9a53f2ed5e846d344a03e57b1ecb7e25cfea9bba365b7b791fc6f41ad3fb0.jpg)

![4000966cb295a9830e97aa35b7157ab6ff9aa23eb6acc506bf6ef7ead5b51060.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/images/4000966cb295a9830e97aa35b7157ab6ff9aa23eb6acc506bf6ef7ead5b51060.jpg)

![6209c14a2e850e4b63382688ba7e73327df5e5b7fcaaa302ca846b832b151edb.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/images/6209c14a2e850e4b63382688ba7e73327df5e5b7fcaaa302ca846b832b151edb.jpg)

![7093999d021f9400dced49cd3f65a0f226e9dc52e062bcc59f5b77cba9e284c0.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/images/7093999d021f9400dced49cd3f65a0f226e9dc52e062bcc59f5b77cba9e284c0.jpg)

![a86624822a22dd341d73759a08dc92d7f303b69a8e6dd45de346c59a86fb93e6.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/images/a86624822a22dd341d73759a08dc92d7f303b69a8e6dd45de346c59a86fb93e6.jpg)

![c2e7f2189db57138a2659733ff010ba2486f1faec5f4fd92e384aac5956fddd3.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/images/c2e7f2189db57138a2659733ff010ba2486f1faec5f4fd92e384aac5956fddd3.jpg)

![de366208ab4c8494dd888c51f7da573e18c7a502babada5c9e18265b78f2beed.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/images/de366208ab4c8494dd888c51f7da573e18c7a502babada5c9e18265b78f2beed.jpg)

### Tables

![0114070584da4a5982b740a84685ec1ab929d785b2337caf8eeb5313a89b3581.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/tables/0114070584da4a5982b740a84685ec1ab929d785b2337caf8eeb5313a89b3581.jpg)

![1344987130516d08f70deafa491bef0b3aea061548e0b81e9bac44ed3881dbb5.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/tables/1344987130516d08f70deafa491bef0b3aea061548e0b81e9bac44ed3881dbb5.jpg)

![14b5b039603ed81b2bedd1c644b1341ebc4679cfaeda835b610ce0c27777aae1.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/tables/14b5b039603ed81b2bedd1c644b1341ebc4679cfaeda835b610ce0c27777aae1.jpg)

![1d7ad6dca196a00bb0179f9929ab3265a5bbad99ffcdc7d6c38d83908dbaa0c1.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/tables/1d7ad6dca196a00bb0179f9929ab3265a5bbad99ffcdc7d6c38d83908dbaa0c1.jpg)

![2968d528a9e51e2c8f10dcec544780bb11793d9768eb27264884f9e14357606a.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/tables/2968d528a9e51e2c8f10dcec544780bb11793d9768eb27264884f9e14357606a.jpg)

![34a0fe7d4afb6c3b0643deeec3d121b96d491533aec7de1c24662a8874781a72.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/tables/34a0fe7d4afb6c3b0643deeec3d121b96d491533aec7de1c24662a8874781a72.jpg)

![3aa237f4be7447c02e4650bc6c6f56954d61c197444be39e626b54665cb07bee.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/tables/3aa237f4be7447c02e4650bc6c6f56954d61c197444be39e626b54665cb07bee.jpg)

![4b630a1f082df4903ead9b8186d2f540e56e786513c30b70a33edc5e06959eeb.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/tables/4b630a1f082df4903ead9b8186d2f540e56e786513c30b70a33edc5e06959eeb.jpg)

![5a8c7ad1f7d31da12f248bee77460eb30d4f008bf0462e50a7afd01c60ed9e45.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/tables/5a8c7ad1f7d31da12f248bee77460eb30d4f008bf0462e50a7afd01c60ed9e45.jpg)

![5ef574c5991ede47d352c5de90c54f92e4eaadc49dcfcf99edc775e32c470463.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/tables/5ef574c5991ede47d352c5de90c54f92e4eaadc49dcfcf99edc775e32c470463.jpg)

![61f6b2d561d2e8e2e207a7814e21a67db4ec1697b4a22056db419cf8e497a9fa.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/tables/61f6b2d561d2e8e2e207a7814e21a67db4ec1697b4a22056db419cf8e497a9fa.jpg)

![654274a383f7a42ebfc7e8e9fdc3ed09135a67f6b9ab2f132d12371ff87379cc.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/tables/654274a383f7a42ebfc7e8e9fdc3ed09135a67f6b9ab2f132d12371ff87379cc.jpg)

![8b267b69aed98570ae7c1b29af723022c2d3b00d061b1bf683cbf4d8b1f735ae.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/tables/8b267b69aed98570ae7c1b29af723022c2d3b00d061b1bf683cbf4d8b1f735ae.jpg)

![95dc8c38e9f639b6d6992f7cf8e0e37088afafcb92639a3262c751a2ccc45b99.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/tables/95dc8c38e9f639b6d6992f7cf8e0e37088afafcb92639a3262c751a2ccc45b99.jpg)

![aa7fe8827b53d115230f2d9c87fe20c034c70f16967d2874b6d547546440ac29.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/tables/aa7fe8827b53d115230f2d9c87fe20c034c70f16967d2874b6d547546440ac29.jpg)

![e451669a1db5d024dacad7232b7dddeca7317b3dfe7e79a0b28fc76d20e476f4.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/tables/e451669a1db5d024dacad7232b7dddeca7317b3dfe7e79a0b28fc76d20e476f4.jpg)

![e5e8d687fa3179172ebfcbc4e1eba93bd7e76ac9e54ecb17b654915ed935c22b.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/tables/e5e8d687fa3179172ebfcbc4e1eba93bd7e76ac9e54ecb17b654915ed935c22b.jpg)

![fd226392839a09901b86732309a643c06a657e670a98505fb1d8686a1a4e8894.jpg](../iclr_results/219_MaRS_%20A%20Fast%20Sampler%20for%20Mean%20Reverting%20Diffusion%20based%20on%20ODE%20and%20SDE%20Solvers/tables/fd226392839a09901b86732309a643c06a657e670a98505fb1d8686a1a4e8894.jpg)

## Robust Function-Calling for On-Device Language Model via Function Masking


### Images

![3b7f0cdd1dbe96f35caad97749162f39adbf253c2c35da8fa0b83ac4a0dd48de.jpg](../iclr_results/220_Robust%20Function-Calling%20for%20On-Device%20Language%20Model%20via%20Function%20Masking/images/3b7f0cdd1dbe96f35caad97749162f39adbf253c2c35da8fa0b83ac4a0dd48de.jpg)

![524ba80586713ddf5eeca083b5b69b96981fc8bbd0a72ac63ec8c5cfd1fdd218.jpg](../iclr_results/220_Robust%20Function-Calling%20for%20On-Device%20Language%20Model%20via%20Function%20Masking/images/524ba80586713ddf5eeca083b5b69b96981fc8bbd0a72ac63ec8c5cfd1fdd218.jpg)

![8fda173c886444f8c488125644e68418a0318cdd97357694dec857fb251e7d59.jpg](../iclr_results/220_Robust%20Function-Calling%20for%20On-Device%20Language%20Model%20via%20Function%20Masking/images/8fda173c886444f8c488125644e68418a0318cdd97357694dec857fb251e7d59.jpg)

![a1ff33de8100c0099513b6ef40a18da91ac8e1bfa1fff67fb3f939ffbce9b124.jpg](../iclr_results/220_Robust%20Function-Calling%20for%20On-Device%20Language%20Model%20via%20Function%20Masking/images/a1ff33de8100c0099513b6ef40a18da91ac8e1bfa1fff67fb3f939ffbce9b124.jpg)

![cd5f33fb5914ba640c0e5ea5ad9c6685a26bbfbeff313fc393108f6eedf6075e.jpg](../iclr_results/220_Robust%20Function-Calling%20for%20On-Device%20Language%20Model%20via%20Function%20Masking/images/cd5f33fb5914ba640c0e5ea5ad9c6685a26bbfbeff313fc393108f6eedf6075e.jpg)

![f40c9e8055a5a4b02f43a07b68808b3a16274c83904ee4f79d2f6f3ae70460f6.jpg](../iclr_results/220_Robust%20Function-Calling%20for%20On-Device%20Language%20Model%20via%20Function%20Masking/images/f40c9e8055a5a4b02f43a07b68808b3a16274c83904ee4f79d2f6f3ae70460f6.jpg)

### Tables

![0dcac688bed0c6ac6f46ae0a43cea5c80753e19e9447978d339dacd03e1e4820.jpg](../iclr_results/220_Robust%20Function-Calling%20for%20On-Device%20Language%20Model%20via%20Function%20Masking/tables/0dcac688bed0c6ac6f46ae0a43cea5c80753e19e9447978d339dacd03e1e4820.jpg)

![0e2a423ca8b64ec220e4463e286d9fe56ca1f0745404c14e244a9d97365a545f.jpg](../iclr_results/220_Robust%20Function-Calling%20for%20On-Device%20Language%20Model%20via%20Function%20Masking/tables/0e2a423ca8b64ec220e4463e286d9fe56ca1f0745404c14e244a9d97365a545f.jpg)

![1c0d0877a13964b686bf3702bb35d910af3b68c8191fdb3de2708b18b01b28cd.jpg](../iclr_results/220_Robust%20Function-Calling%20for%20On-Device%20Language%20Model%20via%20Function%20Masking/tables/1c0d0877a13964b686bf3702bb35d910af3b68c8191fdb3de2708b18b01b28cd.jpg)

![38adcfd7bc52ad8b20d9719ab6657010d0d8cf667652cfbb45b174446f579f57.jpg](../iclr_results/220_Robust%20Function-Calling%20for%20On-Device%20Language%20Model%20via%20Function%20Masking/tables/38adcfd7bc52ad8b20d9719ab6657010d0d8cf667652cfbb45b174446f579f57.jpg)

![64387c51d64d3d75ce15a5a059493c0fc1ad36f237878e79478480a2c523514c.jpg](../iclr_results/220_Robust%20Function-Calling%20for%20On-Device%20Language%20Model%20via%20Function%20Masking/tables/64387c51d64d3d75ce15a5a059493c0fc1ad36f237878e79478480a2c523514c.jpg)

![69d4f165a51d42fbd48519b8d3bce797d1d6990a198f24f50641a0b2fd08d32b.jpg](../iclr_results/220_Robust%20Function-Calling%20for%20On-Device%20Language%20Model%20via%20Function%20Masking/tables/69d4f165a51d42fbd48519b8d3bce797d1d6990a198f24f50641a0b2fd08d32b.jpg)

![7377b546c4acb568837b1575ecc22f1bbfeddc2165a6cecfbe4035e38f4b4e1f.jpg](../iclr_results/220_Robust%20Function-Calling%20for%20On-Device%20Language%20Model%20via%20Function%20Masking/tables/7377b546c4acb568837b1575ecc22f1bbfeddc2165a6cecfbe4035e38f4b4e1f.jpg)

![86b5c3c9f5c004108bef05b36dc3afaf886a3aa0218fc21235f3a5cf83225fc4.jpg](../iclr_results/220_Robust%20Function-Calling%20for%20On-Device%20Language%20Model%20via%20Function%20Masking/tables/86b5c3c9f5c004108bef05b36dc3afaf886a3aa0218fc21235f3a5cf83225fc4.jpg)

![8f6ce191b72f0a3bda28b2d7946f9fca1b5abaadbbbf6cee3b20f8fe8084d49a.jpg](../iclr_results/220_Robust%20Function-Calling%20for%20On-Device%20Language%20Model%20via%20Function%20Masking/tables/8f6ce191b72f0a3bda28b2d7946f9fca1b5abaadbbbf6cee3b20f8fe8084d49a.jpg)

![a6ab5d8a57bfd835655d5ea95367547c084ebf4fb86b086fc2756d277c92ab3d.jpg](../iclr_results/220_Robust%20Function-Calling%20for%20On-Device%20Language%20Model%20via%20Function%20Masking/tables/a6ab5d8a57bfd835655d5ea95367547c084ebf4fb86b086fc2756d277c92ab3d.jpg)

![b55ae717f74a000a97a9e47986f71bbb70e246e3cc841f5ae645d62065ae204d.jpg](../iclr_results/220_Robust%20Function-Calling%20for%20On-Device%20Language%20Model%20via%20Function%20Masking/tables/b55ae717f74a000a97a9e47986f71bbb70e246e3cc841f5ae645d62065ae204d.jpg)

![c441cd01e43e3dc7de9d93e117d76bc6bd10e9169ad3a8cdeeadb7d69171bb8e.jpg](../iclr_results/220_Robust%20Function-Calling%20for%20On-Device%20Language%20Model%20via%20Function%20Masking/tables/c441cd01e43e3dc7de9d93e117d76bc6bd10e9169ad3a8cdeeadb7d69171bb8e.jpg)

![cd577fdd329a8d75580b537c3c87e532ec2c4abeb6adf9efabdc5d516f7a9363.jpg](../iclr_results/220_Robust%20Function-Calling%20for%20On-Device%20Language%20Model%20via%20Function%20Masking/tables/cd577fdd329a8d75580b537c3c87e532ec2c4abeb6adf9efabdc5d516f7a9363.jpg)

![d0dffbbd5e07f153b612cf9a4bd96bf60df1106090bfaa6af0cd02f5529689ce.jpg](../iclr_results/220_Robust%20Function-Calling%20for%20On-Device%20Language%20Model%20via%20Function%20Masking/tables/d0dffbbd5e07f153b612cf9a4bd96bf60df1106090bfaa6af0cd02f5529689ce.jpg)

![e73cf1001c47415834bef28e186325265f05b901efa0f084d05ff421bc410e88.jpg](../iclr_results/220_Robust%20Function-Calling%20for%20On-Device%20Language%20Model%20via%20Function%20Masking/tables/e73cf1001c47415834bef28e186325265f05b901efa0f084d05ff421bc410e88.jpg)

![eb9a88ff1a01fbdae1679addd0ed65c54faa57e92c0cafbf81211e2d4b4f31df.jpg](../iclr_results/220_Robust%20Function-Calling%20for%20On-Device%20Language%20Model%20via%20Function%20Masking/tables/eb9a88ff1a01fbdae1679addd0ed65c54faa57e92c0cafbf81211e2d4b4f31df.jpg)

## GOLD: Graph Out-of-Distribution Detection via Implicit Adversarial Latent Generation


### Images

![2c3507bd56982079a8a2a837f19bf5045eac4fcb98e7263e91390aed3383c5a8.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/images/2c3507bd56982079a8a2a837f19bf5045eac4fcb98e7263e91390aed3383c5a8.jpg)

![4206edf9b5c4ef8e03a946fd5188322e932c7d0b713f14849ff677f921aa3e6b.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/images/4206edf9b5c4ef8e03a946fd5188322e932c7d0b713f14849ff677f921aa3e6b.jpg)

![7a09b5a0797258a33f05a84956b3c9f953215d07d830dfa684086b562fdf1e84.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/images/7a09b5a0797258a33f05a84956b3c9f953215d07d830dfa684086b562fdf1e84.jpg)

![8c39bf840c72616c68fc68792a5a4e67274eaf2dce934c3c8d6766de8ddfb174.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/images/8c39bf840c72616c68fc68792a5a4e67274eaf2dce934c3c8d6766de8ddfb174.jpg)

![ae45a91daa0bae21ccac083f0225fffef827aeea328a162fe7bef15689cfd607.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/images/ae45a91daa0bae21ccac083f0225fffef827aeea328a162fe7bef15689cfd607.jpg)

![c26691043d58f26649f908d7f65d33b6590c5f0a335b3e58fd3409b184667f2e.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/images/c26691043d58f26649f908d7f65d33b6590c5f0a335b3e58fd3409b184667f2e.jpg)

![d72ae7f532ea0555170e1dd533248498b3f907603b1453fa798dda5c85ff03cd.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/images/d72ae7f532ea0555170e1dd533248498b3f907603b1453fa798dda5c85ff03cd.jpg)

### Tables

![0b545e2f2206f5b3c86ec50877f584a0da285fce61a5ba74c37d2f662df2b57b.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/tables/0b545e2f2206f5b3c86ec50877f584a0da285fce61a5ba74c37d2f662df2b57b.jpg)

![0e99793d40cd5b52e04b99430e7a01c4f4bdd78b016616f41206c8353a185cad.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/tables/0e99793d40cd5b52e04b99430e7a01c4f4bdd78b016616f41206c8353a185cad.jpg)

![0f3c85e22e081b23f1dce00f50541c4a96956e7e1523a68ebc81ef42b42086d9.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/tables/0f3c85e22e081b23f1dce00f50541c4a96956e7e1523a68ebc81ef42b42086d9.jpg)

![17f201adc9b046c2a4bfd3bdb5b1620a32d5f49ba60ba49c9016bdb41d5c6cd6.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/tables/17f201adc9b046c2a4bfd3bdb5b1620a32d5f49ba60ba49c9016bdb41d5c6cd6.jpg)

![2564c077dd4c279f40b7b2b1fcf4f093a8401daab2e1ad6c566c935d24c41ecd.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/tables/2564c077dd4c279f40b7b2b1fcf4f093a8401daab2e1ad6c566c935d24c41ecd.jpg)

![350d44396eb520e63936ed2d05beb790c382f46e7302be2ae18bb367491a3c35.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/tables/350d44396eb520e63936ed2d05beb790c382f46e7302be2ae18bb367491a3c35.jpg)

![3e7281f3066aa67adf82b75250a931277ba14b949ac706b5faca34dbfa6405fe.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/tables/3e7281f3066aa67adf82b75250a931277ba14b949ac706b5faca34dbfa6405fe.jpg)

![3eb83a80bbe12fddb88f2b3f70d65982be69aac8c7daa9299c5d50208428983d.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/tables/3eb83a80bbe12fddb88f2b3f70d65982be69aac8c7daa9299c5d50208428983d.jpg)

![40b5bdccd7a406aec79668e23d634366eb6d6fecbf163d2c3d56066a0db60fd7.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/tables/40b5bdccd7a406aec79668e23d634366eb6d6fecbf163d2c3d56066a0db60fd7.jpg)

![4f423565533de9c2350869ddacf4559f220fa0c7c36465823e56e031155540d3.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/tables/4f423565533de9c2350869ddacf4559f220fa0c7c36465823e56e031155540d3.jpg)

![52ffd39f873b11029e28fc9a391b021bb64d3826acbf8cc4e1877fcdcb27760e.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/tables/52ffd39f873b11029e28fc9a391b021bb64d3826acbf8cc4e1877fcdcb27760e.jpg)

![69e3e84b200e8d75d1f7666a5fd16b70808c9d5cc8e15d3b93d8e836f56cc64b.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/tables/69e3e84b200e8d75d1f7666a5fd16b70808c9d5cc8e15d3b93d8e836f56cc64b.jpg)

![99a5f692b4971df17e76d84b3429416a964b8cd9a7331253385996c7c47aae23.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/tables/99a5f692b4971df17e76d84b3429416a964b8cd9a7331253385996c7c47aae23.jpg)

![99d4e4f935038ecb7582ac307455ca8615e65bcc24167e736e8112890dbbc296.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/tables/99d4e4f935038ecb7582ac307455ca8615e65bcc24167e736e8112890dbbc296.jpg)

![a2f42664c43c0ff54e1f3c8251c63d45e9a6b31b8c23ac831629f0837abd774e.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/tables/a2f42664c43c0ff54e1f3c8251c63d45e9a6b31b8c23ac831629f0837abd774e.jpg)

![ba11e37a57ff85ab9a6655721e0d613b516b92495332c5dfa1fbabf60c9dd445.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/tables/ba11e37a57ff85ab9a6655721e0d613b516b92495332c5dfa1fbabf60c9dd445.jpg)

![bfdfe039b6a7c4ab0d9a1c175186c15da46ab79019556732281ef3994097fa76.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/tables/bfdfe039b6a7c4ab0d9a1c175186c15da46ab79019556732281ef3994097fa76.jpg)

![c28b55fb68b7f77dfb8c6018c4f448830525357cf0afd777e9ab0a290c387c63.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/tables/c28b55fb68b7f77dfb8c6018c4f448830525357cf0afd777e9ab0a290c387c63.jpg)

![c2a2ff5251e1b40216baedc3a2ada09dad11f2a0d00cbeb119e36203ab8d0baf.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/tables/c2a2ff5251e1b40216baedc3a2ada09dad11f2a0d00cbeb119e36203ab8d0baf.jpg)

![cee7d11f82eac52648b227a4630050c4df73952a53961540ee4d65c958486ff6.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/tables/cee7d11f82eac52648b227a4630050c4df73952a53961540ee4d65c958486ff6.jpg)

![e0dc34d64e82611e848babe9f251a477c9a8f4c25bf7974214638855b94ac4d0.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/tables/e0dc34d64e82611e848babe9f251a477c9a8f4c25bf7974214638855b94ac4d0.jpg)

![ee1db9a71ea19fc53d37e2bf3ebb962bd19dfdb6b6208353ab73f2a363cc249d.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/tables/ee1db9a71ea19fc53d37e2bf3ebb962bd19dfdb6b6208353ab73f2a363cc249d.jpg)

![f58a70565bd682ff287e5ed2c0c2cb23495a1142ea5b94f9d5fadace35f71f4d.jpg](../iclr_results/221_GOLD_%20Graph%20Out-of-Distribution%20Detection%20via%20Implicit%20Adversarial%20Latent%20Generation/tables/f58a70565bd682ff287e5ed2c0c2cb23495a1142ea5b94f9d5fadace35f71f4d.jpg)

## Advantage-Guided Distillation for Preference Alignment in Small Language Models


### Images

![4c821f14f3c2326216fbcfff8dad39382e72560cf764379be9a0a7b341420476.jpg](../iclr_results/222_Advantage-Guided%20Distillation%20for%20Preference%20Alignment%20in%20Small%20Language%20Models/images/4c821f14f3c2326216fbcfff8dad39382e72560cf764379be9a0a7b341420476.jpg)

![5faa13b1c098fcb3bd7a2e8756c4a54944da0eec766544c6fe9b0e4e1bcdfc24.jpg](../iclr_results/222_Advantage-Guided%20Distillation%20for%20Preference%20Alignment%20in%20Small%20Language%20Models/images/5faa13b1c098fcb3bd7a2e8756c4a54944da0eec766544c6fe9b0e4e1bcdfc24.jpg)

![6f70fe407636a9ede3667700262cfda4ba2a28c515c8e94357f18750b2e59e1e.jpg](../iclr_results/222_Advantage-Guided%20Distillation%20for%20Preference%20Alignment%20in%20Small%20Language%20Models/images/6f70fe407636a9ede3667700262cfda4ba2a28c515c8e94357f18750b2e59e1e.jpg)

![8705cb94c701e5564ac629a4691c5c4f530cd334442c90a10f52c7ee5c53468a.jpg](../iclr_results/222_Advantage-Guided%20Distillation%20for%20Preference%20Alignment%20in%20Small%20Language%20Models/images/8705cb94c701e5564ac629a4691c5c4f530cd334442c90a10f52c7ee5c53468a.jpg)

![e9d640cc9dedca5aa9ef1169fb34f216206101ac92d766139c48636928d25c6e.jpg](../iclr_results/222_Advantage-Guided%20Distillation%20for%20Preference%20Alignment%20in%20Small%20Language%20Models/images/e9d640cc9dedca5aa9ef1169fb34f216206101ac92d766139c48636928d25c6e.jpg)

### Tables

![0f53ae0c36992f5b53e6ed6862052c1186becb073d5743a2152598043209b8dc.jpg](../iclr_results/222_Advantage-Guided%20Distillation%20for%20Preference%20Alignment%20in%20Small%20Language%20Models/tables/0f53ae0c36992f5b53e6ed6862052c1186becb073d5743a2152598043209b8dc.jpg)

![178f4391d203d1108a131c170ff2b53db9fec2ccb8f206ca5b3ceb218e89ac1c.jpg](../iclr_results/222_Advantage-Guided%20Distillation%20for%20Preference%20Alignment%20in%20Small%20Language%20Models/tables/178f4391d203d1108a131c170ff2b53db9fec2ccb8f206ca5b3ceb218e89ac1c.jpg)

![76f007c7bc73d30b56e16d42a96662c4648b89dba490d1f09ca260a7732aa570.jpg](../iclr_results/222_Advantage-Guided%20Distillation%20for%20Preference%20Alignment%20in%20Small%20Language%20Models/tables/76f007c7bc73d30b56e16d42a96662c4648b89dba490d1f09ca260a7732aa570.jpg)

![942d3001d13b7095b552fe1c9559d8e4c56d70a230e3d84b60d42c6c226e140c.jpg](../iclr_results/222_Advantage-Guided%20Distillation%20for%20Preference%20Alignment%20in%20Small%20Language%20Models/tables/942d3001d13b7095b552fe1c9559d8e4c56d70a230e3d84b60d42c6c226e140c.jpg)

![cdef0256499e771703080f26567b78b6eb6190983e2443c1d14bda9734fc8adf.jpg](../iclr_results/222_Advantage-Guided%20Distillation%20for%20Preference%20Alignment%20in%20Small%20Language%20Models/tables/cdef0256499e771703080f26567b78b6eb6190983e2443c1d14bda9734fc8adf.jpg)

![eafafd5ad29dceccfb9cacda651c84a0f1592a97541a0e0922cdecfdda6e7e47.jpg](../iclr_results/222_Advantage-Guided%20Distillation%20for%20Preference%20Alignment%20in%20Small%20Language%20Models/tables/eafafd5ad29dceccfb9cacda651c84a0f1592a97541a0e0922cdecfdda6e7e47.jpg)

![ff67627f7827a858d643382625675c50ccfbb29d6e504bde588f5348c9ba87b9.jpg](../iclr_results/222_Advantage-Guided%20Distillation%20for%20Preference%20Alignment%20in%20Small%20Language%20Models/tables/ff67627f7827a858d643382625675c50ccfbb29d6e504bde588f5348c9ba87b9.jpg)

## JudgeLM: Fine-tuned Large Language Models are Scalable Judges


### Images

![05069d0893d8c012dab2b8777f99425645f2f88c41f601330415805b73386526.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/images/05069d0893d8c012dab2b8777f99425645f2f88c41f601330415805b73386526.jpg)

![352380abdd043c166ebf7f33e54871712873840eb0a28b4997169c94cff8af84.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/images/352380abdd043c166ebf7f33e54871712873840eb0a28b4997169c94cff8af84.jpg)

![40c169660f735cecb20d8dabd5c09e5f40c22116006500d8a095ee7450043752.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/images/40c169660f735cecb20d8dabd5c09e5f40c22116006500d8a095ee7450043752.jpg)

![71c65dd9011f0df05d18d3e6b4d90807c38a7519606a3fd57e14a5623914f444.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/images/71c65dd9011f0df05d18d3e6b4d90807c38a7519606a3fd57e14a5623914f444.jpg)

![72c7759da6378535284d17e56cce8de033e50a4f1ab7fa197adefdb2b2a1ae3e.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/images/72c7759da6378535284d17e56cce8de033e50a4f1ab7fa197adefdb2b2a1ae3e.jpg)

![73f64764a62befe05b08e2e3089dbdfe7863cffe39df81f3b9fb32998ff6e2f4.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/images/73f64764a62befe05b08e2e3089dbdfe7863cffe39df81f3b9fb32998ff6e2f4.jpg)

![744e65edf126e196af7d3dd16668e1cd0f4f5b24859f736984a9a701d7bb0f9e.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/images/744e65edf126e196af7d3dd16668e1cd0f4f5b24859f736984a9a701d7bb0f9e.jpg)

![aaa3c67ccaecf5ac78d77c3e92cd19cff3d448ea76317e593df8ed4e7994b218.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/images/aaa3c67ccaecf5ac78d77c3e92cd19cff3d448ea76317e593df8ed4e7994b218.jpg)

![adeb3f70ec95b7bea49133f7d7afe8cbab4bba71438e9996ee8a6c1173e17aea.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/images/adeb3f70ec95b7bea49133f7d7afe8cbab4bba71438e9996ee8a6c1173e17aea.jpg)

![bef707e0d7f76b1a73ca7c653ee5640039a3d526cd6fbf1db62593e913a57adf.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/images/bef707e0d7f76b1a73ca7c653ee5640039a3d526cd6fbf1db62593e913a57adf.jpg)

![ebefda17b01f51d31e460892da9f10ef8b1f56d48d1f3a37b46bfc328fd5bfee.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/images/ebefda17b01f51d31e460892da9f10ef8b1f56d48d1f3a37b46bfc328fd5bfee.jpg)

### Tables

![12ad4ad1c5d9b21b0c3f4a7fb36f2e03cf368b48738a15b02c012b7ecf89e979.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/tables/12ad4ad1c5d9b21b0c3f4a7fb36f2e03cf368b48738a15b02c012b7ecf89e979.jpg)

![137e2a41b2cbf8bdd9e9c1c47efb764d8b766970825efb358f05dd7b87507f3f.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/tables/137e2a41b2cbf8bdd9e9c1c47efb764d8b766970825efb358f05dd7b87507f3f.jpg)

![15f332bd4275e46903137b7b05dea65bf24d7379efdf0bc1c6bf70e577db12ad.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/tables/15f332bd4275e46903137b7b05dea65bf24d7379efdf0bc1c6bf70e577db12ad.jpg)

![178430aa890998c94baf084a9c305742c7d6688bfc0020fd43c9bf7714737b21.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/tables/178430aa890998c94baf084a9c305742c7d6688bfc0020fd43c9bf7714737b21.jpg)

![25ae2cebdaf169d3296602dd7642fb73a0b13c24ee34742ff946138bbef930c0.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/tables/25ae2cebdaf169d3296602dd7642fb73a0b13c24ee34742ff946138bbef930c0.jpg)

![4af3036e682a9a1616413f1d36a90a30d0462273fdca227294db816398a1e107.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/tables/4af3036e682a9a1616413f1d36a90a30d0462273fdca227294db816398a1e107.jpg)

![4c76f9225c836ce2db7ca2f7380ee4bef024e14b71f7b88e345ffa14d258e734.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/tables/4c76f9225c836ce2db7ca2f7380ee4bef024e14b71f7b88e345ffa14d258e734.jpg)

![5beb9e73c86b69dab1cf9a52eb1e5fb8df2b6376af7fa7b44b253209f0f020bb.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/tables/5beb9e73c86b69dab1cf9a52eb1e5fb8df2b6376af7fa7b44b253209f0f020bb.jpg)

![630a19bd84f9e3cbd2a726e6b709f2798772429d078a5bff024e54c867d6db76.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/tables/630a19bd84f9e3cbd2a726e6b709f2798772429d078a5bff024e54c867d6db76.jpg)

![6899230b7a8da637656290618ba8d8dc47b493c864719cee450cf5866aecebe7.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/tables/6899230b7a8da637656290618ba8d8dc47b493c864719cee450cf5866aecebe7.jpg)

![6d5554c6a81802e80fd9a8c4650789c5afbb48fb17e7b18f1f03436886c515db.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/tables/6d5554c6a81802e80fd9a8c4650789c5afbb48fb17e7b18f1f03436886c515db.jpg)

![73a031a517283cde1d34a83dabc8cfada5821441c361b8c62fd77a22346bdba1.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/tables/73a031a517283cde1d34a83dabc8cfada5821441c361b8c62fd77a22346bdba1.jpg)

![744ee10f7e648ada24091a5d1e0c784192694b64982402adc29e26fab95f2d1f.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/tables/744ee10f7e648ada24091a5d1e0c784192694b64982402adc29e26fab95f2d1f.jpg)

![a33e7978d6caaebe9bbe07240e0a4af4b363d71eaeb7329cad6cb274fd9e74f0.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/tables/a33e7978d6caaebe9bbe07240e0a4af4b363d71eaeb7329cad6cb274fd9e74f0.jpg)

![b1f240a73090a531b49d60cbffdb29264bb72338775f2bcbc34ad528c3505e4f.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/tables/b1f240a73090a531b49d60cbffdb29264bb72338775f2bcbc34ad528c3505e4f.jpg)

![b2172487c2015c494d4475d7a1e2d4a4ee0bd34b0baf6c2ab333625c4938975e.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/tables/b2172487c2015c494d4475d7a1e2d4a4ee0bd34b0baf6c2ab333625c4938975e.jpg)

![bb32216f5b5322920c6298786c6eb81416314a216f2b97a2a9b6b20695d92430.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/tables/bb32216f5b5322920c6298786c6eb81416314a216f2b97a2a9b6b20695d92430.jpg)

![c8ebe51b68eda35d387b79190cc6bf35ca5c639ef91770429d8ea60ad7180685.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/tables/c8ebe51b68eda35d387b79190cc6bf35ca5c639ef91770429d8ea60ad7180685.jpg)

![de8020961f4de32a0a4874ce0c6496f5faff65b8513c312c139e3989704448fb.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/tables/de8020961f4de32a0a4874ce0c6496f5faff65b8513c312c139e3989704448fb.jpg)

![e5a90cf3aeb6647f6352101ae6921e0fe315604c425eb1f0db6a9f796574aa39.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/tables/e5a90cf3aeb6647f6352101ae6921e0fe315604c425eb1f0db6a9f796574aa39.jpg)

![eb7bf662f4fc77ec9a7ca8002239e875e7c3d80871244a3d6e03344aadc4f656.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/tables/eb7bf662f4fc77ec9a7ca8002239e875e7c3d80871244a3d6e03344aadc4f656.jpg)

![f5f151c14f781beea0298254f116c80feccf394160a5a5f673fc7f433b692cae.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/tables/f5f151c14f781beea0298254f116c80feccf394160a5a5f673fc7f433b692cae.jpg)

![f9fe2a9284028c8033d20a195886fa83c6610cb13d836d1e50d8bc28cf156028.jpg](../iclr_results/223_JudgeLM_%20Fine-tuned%20Large%20Language%20Models%20are%20Scalable%20Judges/tables/f9fe2a9284028c8033d20a195886fa83c6610cb13d836d1e50d8bc28cf156028.jpg)

## Stem-OB: Generalizable Visual Imitation Learning with Stem-Like Convergent Observation through Diffusion Inversion

### Images

![0995b8137d0a52f791063a9efc5b05cfc0e81a05bf680cb240bed33c6c76f60f.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/images/0995b8137d0a52f791063a9efc5b05cfc0e81a05bf680cb240bed33c6c76f60f.jpg)

![1510d391db538d0c3d148f5ae77445629433d2cddff0ccd822bcefb11bf2d206.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/images/1510d391db538d0c3d148f5ae77445629433d2cddff0ccd822bcefb11bf2d206.jpg)

![278776e83a4d2543dfe1612c8f2857bc2d3ffcd9712743e6d2a9e60bffdf4bf7.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/images/278776e83a4d2543dfe1612c8f2857bc2d3ffcd9712743e6d2a9e60bffdf4bf7.jpg)

![35ecbcd1c3ed7362f30b4d0b3fbde880cdb8f7c1383d123103ab7e45ba8c2c37.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/images/35ecbcd1c3ed7362f30b4d0b3fbde880cdb8f7c1383d123103ab7e45ba8c2c37.jpg)

![36a25a556af53f03a4551fd1a02833b8cb8cf9684b2166988426fbaf490a66bf.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/images/36a25a556af53f03a4551fd1a02833b8cb8cf9684b2166988426fbaf490a66bf.jpg)

![5f3f15a7e426fc2d83ed86b5e8eb4992d370ded2207bb6f327d1c0a53eab9cb5.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/images/5f3f15a7e426fc2d83ed86b5e8eb4992d370ded2207bb6f327d1c0a53eab9cb5.jpg)

![6239c48d0241d00d4510f340f3dc5b31a9614719bb496113f7a745d3326c0716.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/images/6239c48d0241d00d4510f340f3dc5b31a9614719bb496113f7a745d3326c0716.jpg)

![633f0def6abca7d7ee3c53d042944993e270c37eecaabfb7c513f84e89818696.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/images/633f0def6abca7d7ee3c53d042944993e270c37eecaabfb7c513f84e89818696.jpg)

![67da74f24969a4ca1d8d7980ecb87c4e0094ec48b2c8f813ae2718cf49166988.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/images/67da74f24969a4ca1d8d7980ecb87c4e0094ec48b2c8f813ae2718cf49166988.jpg)

![75a1f8e09bb8f94b9661f83135c07dbca60cfbab851bd904e1b6f539cb223a43.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/images/75a1f8e09bb8f94b9661f83135c07dbca60cfbab851bd904e1b6f539cb223a43.jpg)

![7762cc1518843a739bd2c0a9cc021c66e1d1c9373a3bde6c219066c61cc44e38.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/images/7762cc1518843a739bd2c0a9cc021c66e1d1c9373a3bde6c219066c61cc44e38.jpg)

![790d30e93718cf2b5469df688fd38dbd4e3ba12305626b509fd5828197a8a74c.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/images/790d30e93718cf2b5469df688fd38dbd4e3ba12305626b509fd5828197a8a74c.jpg)

![7a05ef58d452c379008876642c5954fa81731bf1ed2c7f131eb8730d3c6a109f.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/images/7a05ef58d452c379008876642c5954fa81731bf1ed2c7f131eb8730d3c6a109f.jpg)

![7c99c07668dde60f7bb518c9fb3ed7fe9ba7493342e7284023ee8b321e089676.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/images/7c99c07668dde60f7bb518c9fb3ed7fe9ba7493342e7284023ee8b321e089676.jpg)

![87ad513d50d08dde620f3d88990b0b6a85921cf33c5013f9714ae26fd9169cb7.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/images/87ad513d50d08dde620f3d88990b0b6a85921cf33c5013f9714ae26fd9169cb7.jpg)

![8b133b55eda552cefcfc7960968d7fb1364960b6d3c3c1945d77916fff1fca38.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/images/8b133b55eda552cefcfc7960968d7fb1364960b6d3c3c1945d77916fff1fca38.jpg)

![944c3d23b1f3032b79776b362c3fd5b5189d845f3d4a24d9e655fc623e37c87a.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/images/944c3d23b1f3032b79776b362c3fd5b5189d845f3d4a24d9e655fc623e37c87a.jpg)

![9aa5b92eb5a35535c0fb5e9aee82edf77324210b9cb96168c05731d45e6d91ae.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/images/9aa5b92eb5a35535c0fb5e9aee82edf77324210b9cb96168c05731d45e6d91ae.jpg)

![a86e5b04ececa54d6738eb3295bfb2a493f37baba846fd1255cdf2779c0714e1.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/images/a86e5b04ececa54d6738eb3295bfb2a493f37baba846fd1255cdf2779c0714e1.jpg)

![b422ae496ea0d700f080ff1a6b98b82e7cab067919959d9c4a7bfa7664620da9.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/images/b422ae496ea0d700f080ff1a6b98b82e7cab067919959d9c4a7bfa7664620da9.jpg)

### Tables

![1a51c58dba146d6f1cf9a7f906b96bd64d93318ef5b175f99da4ed715034e9cb.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/tables/1a51c58dba146d6f1cf9a7f906b96bd64d93318ef5b175f99da4ed715034e9cb.jpg)

![28c8dc8bad23885d016a8c2e996bfd810a9da8fc4cdc0c643f3510e69fec8cd0.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/tables/28c8dc8bad23885d016a8c2e996bfd810a9da8fc4cdc0c643f3510e69fec8cd0.jpg)

![2f3f96d528fe93c33b76534e87721bfa0f06b33b39c2aa3ff4af9f39d01e21cb.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/tables/2f3f96d528fe93c33b76534e87721bfa0f06b33b39c2aa3ff4af9f39d01e21cb.jpg)

![32b3fb37f9d08502d5af3ca2f09172cd5d2faf22088d803468e779ec767020cf.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/tables/32b3fb37f9d08502d5af3ca2f09172cd5d2faf22088d803468e779ec767020cf.jpg)

![43a71c6e2b99bc118560785a0486928426fdfaff80d335222e6bdf5add97b181.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/tables/43a71c6e2b99bc118560785a0486928426fdfaff80d335222e6bdf5add97b181.jpg)

![4baded3a29762aa99a3d70bd1ac6045f2c45c6a82484d1d6a9744bff2ec64259.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/tables/4baded3a29762aa99a3d70bd1ac6045f2c45c6a82484d1d6a9744bff2ec64259.jpg)

![7fcf6094d63c03855123355267034f26207f9d9e907689d867ac37c4aa7c5de5.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/tables/7fcf6094d63c03855123355267034f26207f9d9e907689d867ac37c4aa7c5de5.jpg)

![81a3726dd96d937ffd31ff0c285c62cdb6b03c8fa4bb107bf59d18c2c9479548.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/tables/81a3726dd96d937ffd31ff0c285c62cdb6b03c8fa4bb107bf59d18c2c9479548.jpg)

![8596b0feff663348f311d79ed2d2dbbeb0bdb348de6b8058c796ce8c85eeccd1.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/tables/8596b0feff663348f311d79ed2d2dbbeb0bdb348de6b8058c796ce8c85eeccd1.jpg)

![8eec005eec1f0dca2d59446df0d9223ace38f9072298d178ffcc2402c335bdaa.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/tables/8eec005eec1f0dca2d59446df0d9223ace38f9072298d178ffcc2402c335bdaa.jpg)

![a25174a034f4ae7b973096a4b8609fcd2486ab441f4a53170a73924f311acad4.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/tables/a25174a034f4ae7b973096a4b8609fcd2486ab441f4a53170a73924f311acad4.jpg)

![b64c5fc398ef24eb167059e4f5a1d35abdd91e0cacb8aff52e408d48671070be.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/tables/b64c5fc398ef24eb167059e4f5a1d35abdd91e0cacb8aff52e408d48671070be.jpg)

![ca2cb2768f1c6f8e8bad210c409d51f9dfc5c32c88d2c97611c280c99ece32e8.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/tables/ca2cb2768f1c6f8e8bad210c409d51f9dfc5c32c88d2c97611c280c99ece32e8.jpg)

![cc31dfc07f275872a0f8bc454e214fd3ab9dc29042b79b673cb820c4a40ae1e5.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/tables/cc31dfc07f275872a0f8bc454e214fd3ab9dc29042b79b673cb820c4a40ae1e5.jpg)

![cea97580d8b0776bac9b00c426bedfe4c54e6152b99988fc814851d30a486d7a.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/tables/cea97580d8b0776bac9b00c426bedfe4c54e6152b99988fc814851d30a486d7a.jpg)

![e23a6ff120d23a720f56e9e8eb45f6b6d8d3f6c2eadd01a2dc707ce56280c50c.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/tables/e23a6ff120d23a720f56e9e8eb45f6b6d8d3f6c2eadd01a2dc707ce56280c50c.jpg)

![ff1e516cc7899184b30e0065f7b247a5912cd806c565d37822268641d9e4d6e9.jpg](../iclr_results/225_Stem-OB_%20Generalizable%20Visual%20Imitation%20Learning%20with%20Stem-Like%20Convergent%20Observation%20through%20Diffu/tables/ff1e516cc7899184b30e0065f7b247a5912cd806c565d37822268641d9e4d6e9.jpg)
