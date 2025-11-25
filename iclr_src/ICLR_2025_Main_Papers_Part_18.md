# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 18 of 100

## 目录 (Table of Contents)

1. [SSOLE: Rethinking Orthogonal Low-rank Embedding for Self-Supervised Learning](#SSOLE-Rethinking-Orthogonal-Low-rank-Embedding-for-Self-Supervised-Learning)
2. [EgoSim: Egocentric Exploration in Virtual Worlds with Multi-modal Conditioning](#EgoSim-Egocentric-Exploration-in-Virtual-Worlds-with-Multi-modal-Conditioning)
3. [Deep Kernel Relative Test for Machine-generated Text Detection](#Deep-Kernel-Relative-Test-for-Machine-generated-Text-Detection)
4. [Random Is All You Need: Random Noise Injection on Feature Statistics for Generalizable Deep Image Denoising](#Random-Is-All-You-Need-Random-Noise-Injection-on-Feature-Statistics-for-Generalizable-Deep-Image-Denoising)
5. [GOAL: A Generalist Combinatorial Optimization Agent Learner](#GOAL-A-Generalist-Combinatorial-Optimization-Agent-Learner)
6. [Integral Performance Approximation for Continuous-Time Reinforcement Learning Control](#Integral-Performance-Approximation-for-Continuous-Time-Reinforcement-Learning-Control)
7. [FLOPS: Forward Learning with OPtimal Sampling](#FLOPS-Forward-Learning-with-OPtimal-Sampling)
8. [Once-for-All: Controllable Generative Image Compression with Dynamic Granularity Adaptation](#Once-for-All-Controllable-Generative-Image-Compression-with-Dynamic-Granularity-Adaptation)
9. [Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers](#Attention-in-Large-Language-Models-Yields-Efficient-Zero-Shot-Re-Rankers)
10. [LICO: Large Language Models for In-Context Molecular Optimization](#LICO-Large-Language-Models-for-In-Context-Molecular-Optimization)
11. [SiReRAG: Indexing Similar and Related Information for Multihop Reasoning](#SiReRAG-Indexing-Similar-and-Related-Information-for-Multihop-Reasoning)
12. [AutoBencher: Towards Declarative Benchmark Construction](#AutoBencher-Towards-Declarative-Benchmark-Construction)
13. [DenoiseVAE: Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-training](#DenoiseVAE-Learning-Molecule-Adaptive-Noise-Distributions-for-Denoising-based-3D-Molecular-Pre-training)
14. [ELFS: Label-Free Coreset Selection with Proxy Training Dynamics](#ELFS-Label-Free-Coreset-Selection-with-Proxy-Training-Dynamics)
15. [Generative Inbetweening: Adapting Image-to-Video Models for Keyframe Interpolation](#Generative-Inbetweening-Adapting-Image-to-Video-Models-for-Keyframe-Interpolation)
16. [UniWav: Towards Unified Pre-training for Speech Representation Learning and Generation](#UniWav-Towards-Unified-Pre-training-for-Speech-Representation-Learning-and-Generation)
17. [Forewarned is Forearmed:  Harnessing LLMs for Data Synthesis via Failure-induced Exploration](#Forewarned-is-Forearmed-Harnessing-LLMs-for-Data-Synthesis-via-Failure-induced-Exploration)
18. [Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models](#Effective-and-Efficient-Time-Varying-Counterfactual-Prediction-with-State-Space-Models)
19. [On the expressiveness and spectral bias of KANs](#On-the-expressiveness-and-spectral-bias-of-KANs)
20. [Federated Class-Incremental Learning: A Hybrid Approach Using Latent Exemplars and Data-Free Techniques to Address Local and Global Forgetting](#Federated-Class-Incremental-Learning-A-Hybrid-Approach-Using-Latent-Exemplars-and-Data-Free-Techniques-to-Address-Local-and-Global-Forgetting)
21. [AIMS.au: A Dataset for the Analysis of Modern Slavery Countermeasures in Corporate Statements](#AIMSau-A-Dataset-for-the-Analysis-of-Modern-Slavery-Countermeasures-in-Corporate-Statements)
22. [ThermalGaussian: Thermal 3D Gaussian Splatting](#ThermalGaussian-Thermal-3D-Gaussian-Splatting)
23. [Scaling Wearable Foundation Models](#Scaling-Wearable-Foundation-Models)
24. [Omni-MATH: A Universal Olympiad Level Mathematic Benchmark for Large Language Models](#Omni-MATH-A-Universal-Olympiad-Level-Mathematic-Benchmark-for-Large-Language-Models)
25. [Language-Image Models with 3D Understanding](#Language-Image-Models-with-3D-Understanding)
26. [NoVo: Norm Voting off Hallucinations with Attention Heads in Large Language Models](#NoVo-Norm-Voting-off-Hallucinations-with-Attention-Heads-in-Large-Language-Models)
27. [Generative Monoculture in Large Language Models](#Generative-Monoculture-in-Large-Language-Models)
28. [Accelerating Diffusion Transformers with Token-wise Feature Caching](#Accelerating-Diffusion-Transformers-with-Token-wise-Feature-Caching)
29. [Point-SAM: Promptable 3D Segmentation Model for Point Clouds](#Point-SAM-Promptable-3D-Segmentation-Model-for-Point-Clouds)
30. [Towards Understanding the Universality of Transformers for Next-Token Prediction](#Towards-Understanding-the-Universality-of-Transformers-for-Next-Token-Prediction)
31. [Disentangling Representations through Multi-task Learning](#Disentangling-Representations-through-Multi-task-Learning)
32. [APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding](#APE-Faster-and-Longer-Context-Augmented-Generation-via-Adaptive-Parallel-Encoding)
33. [Robots Pre-train Robots: Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets](#Robots-Pre-train-Robots-Manipulation-Centric-Robotic-Representation-from-Large-Scale-Robot-Datasets)
34. [Causally Motivated Sycophancy Mitigation for Large Language Models](#Causally-Motivated-Sycophancy-Mitigation-for-Large-Language-Models)
35. [Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron](#Understanding-and-Enhancing-Safety-Mechanisms-of-LLMs-via-Safety-Specific-Neuron)
36. [Pacmann: Efficient Private Approximate Nearest Neighbor Search](#Pacmann-Efficient-Private-Approximate-Nearest-Neighbor-Search)
37. [Understanding the Generalization of In-Context Learning in Transformers: An Empirical Study](#Understanding-the-Generalization-of-In-Context-Learning-in-Transformers-An-Empirical-Study)

---


## SSOLE: Rethinking Orthogonal Low-rank Embedding for Self-Supervised Learning

### Images

![0de5620d79a6de6884d5ce1bcdc974b05717eb4d07ce35d0d261ac349a8663a8.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/0de5620d79a6de6884d5ce1bcdc974b05717eb4d07ce35d0d261ac349a8663a8.jpg)

![0f4b6a7e6496c1b8a93fd74b3485a7f1be11c3e5d1ed1bf05f13c7d37f6a55dd.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/0f4b6a7e6496c1b8a93fd74b3485a7f1be11c3e5d1ed1bf05f13c7d37f6a55dd.jpg)

![14f2e27ee23d475c353c406ca93325679994839d5b96963a9039ef7d2ec2b63d.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/14f2e27ee23d475c353c406ca93325679994839d5b96963a9039ef7d2ec2b63d.jpg)

![1df3a32c4d9226c3e0052a483122e12338d290456053cdaaef456623d6c923b7.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/1df3a32c4d9226c3e0052a483122e12338d290456053cdaaef456623d6c923b7.jpg)

![1e35357b874206fdd8ea596117e403c3d231ba780254f19304d7f4c8b0d5eb4e.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/1e35357b874206fdd8ea596117e403c3d231ba780254f19304d7f4c8b0d5eb4e.jpg)

![3329f575273862ac85ad724ce60be77d7c412b2dc53459bdf765bdbcd0a8e718.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/3329f575273862ac85ad724ce60be77d7c412b2dc53459bdf765bdbcd0a8e718.jpg)

![37ddb17023c5b3380b8c57ae6b35b975a5aaad1e39e3ee0f97c259269993c239.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/37ddb17023c5b3380b8c57ae6b35b975a5aaad1e39e3ee0f97c259269993c239.jpg)

![42fc8c0ecf03207ec35f1e09652d447297c01f1fa7f395f2123eaa0ee84eec47.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/42fc8c0ecf03207ec35f1e09652d447297c01f1fa7f395f2123eaa0ee84eec47.jpg)

![44ae9eed691494f2003ebd43cdb3bc5b9dde9a9d2c6858ab0c3ee3461d407978.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/44ae9eed691494f2003ebd43cdb3bc5b9dde9a9d2c6858ab0c3ee3461d407978.jpg)

![46409271b9681db9bb992b5fad650ba5aad7fc2208dab676e42921d1a4a8e7ac.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/46409271b9681db9bb992b5fad650ba5aad7fc2208dab676e42921d1a4a8e7ac.jpg)

![4d26fc338fda4ad044c4b6f1cb04fbe1d16b7f762e75cd9a8348eb539ad85e62.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/4d26fc338fda4ad044c4b6f1cb04fbe1d16b7f762e75cd9a8348eb539ad85e62.jpg)

![4df34740786a6e4a975590ea2c404f434a6d5065a8de2ed41cd455043dddb4a0.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/4df34740786a6e4a975590ea2c404f434a6d5065a8de2ed41cd455043dddb4a0.jpg)

![5c9c143f3e9739208126d792aeadc67e25778aedcc3e57e7a1d9fb7179380c23.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/5c9c143f3e9739208126d792aeadc67e25778aedcc3e57e7a1d9fb7179380c23.jpg)

![619fa0e267859b3198c3883f3823418638852d7cb353ab6e6dcd3bdba34afc26.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/619fa0e267859b3198c3883f3823418638852d7cb353ab6e6dcd3bdba34afc26.jpg)

![6bef268675f36c9177ff20b25ffdf2d8653bd017129b08797f429d64d078d8e2.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/6bef268675f36c9177ff20b25ffdf2d8653bd017129b08797f429d64d078d8e2.jpg)

![6bfdd9bec16cf5e0911540cb8545bf757fbc495295cb1f4b680792a101810044.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/6bfdd9bec16cf5e0911540cb8545bf757fbc495295cb1f4b680792a101810044.jpg)

![744a4e6a8e794829c12581955c829fe53b384f28c11497bee129f77dfd68d50c.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/744a4e6a8e794829c12581955c829fe53b384f28c11497bee129f77dfd68d50c.jpg)

![747ca38d0a5f379603ddd174b84ad500dc280fded43d84f541ea41d0047a9e2e.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/747ca38d0a5f379603ddd174b84ad500dc280fded43d84f541ea41d0047a9e2e.jpg)

![76f9fe52e873b3d67ef3daaa5dee582befe2d2dd3b7ec1416582cdea34f9c22a.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/76f9fe52e873b3d67ef3daaa5dee582befe2d2dd3b7ec1416582cdea34f9c22a.jpg)

![79f6498d93f926c165bc238cca95524d9af4dbc03378521fee06324b90b3b874.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/79f6498d93f926c165bc238cca95524d9af4dbc03378521fee06324b90b3b874.jpg)

![82bb45a686673a37d1166b73002b4ed8f8c3cd7276ece13dbd709dfe50297780.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/82bb45a686673a37d1166b73002b4ed8f8c3cd7276ece13dbd709dfe50297780.jpg)

![83da410b19eb9eb3b02cefa6dbb5c578f9db82a5bc38cefc00c9c4001905277f.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/83da410b19eb9eb3b02cefa6dbb5c578f9db82a5bc38cefc00c9c4001905277f.jpg)

![8ab1689c211c08aac7dce1aed3a352a402afc198a5d8e8d942d912f88b3dbb81.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/8ab1689c211c08aac7dce1aed3a352a402afc198a5d8e8d942d912f88b3dbb81.jpg)

![93017b979b56e2dc3393c54f4758d82b2e9d38128e834a5ee40f37d0f239141f.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/93017b979b56e2dc3393c54f4758d82b2e9d38128e834a5ee40f37d0f239141f.jpg)

![93b62c8303ae822684829a87ed1f8ca2ecf37ce4f628540235ef8b5688514d27.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/93b62c8303ae822684829a87ed1f8ca2ecf37ce4f628540235ef8b5688514d27.jpg)

![99b1cad0891cd0d9a2fff17fc82c8e132652008ae4ea8fd9153ae1cdcb79c1bf.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/99b1cad0891cd0d9a2fff17fc82c8e132652008ae4ea8fd9153ae1cdcb79c1bf.jpg)

![9b719b9f1d6fcda08150b5f04ca1d8cb81b983a1916b9c91960f2017ec309b2a.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/9b719b9f1d6fcda08150b5f04ca1d8cb81b983a1916b9c91960f2017ec309b2a.jpg)

![a093bf1ea492f4a1e332273e739fd6d2832ac74dd3700d64b15ef35f4c3caa31.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/a093bf1ea492f4a1e332273e739fd6d2832ac74dd3700d64b15ef35f4c3caa31.jpg)

![a6c3666c4ca8dd9aef02d405c80f1af118026a56cb0c500ae63641b9c7864887.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/a6c3666c4ca8dd9aef02d405c80f1af118026a56cb0c500ae63641b9c7864887.jpg)

![ab0f7c79600fe5b392f986698effc56c6804f769b2dc9130e545df202cda6f51.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/ab0f7c79600fe5b392f986698effc56c6804f769b2dc9130e545df202cda6f51.jpg)

![ac57df4be58cf2a2af85fdf0f10e14fd0074603be57154fc38c6d2b28abfc13b.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/ac57df4be58cf2a2af85fdf0f10e14fd0074603be57154fc38c6d2b28abfc13b.jpg)

![ba9f4c5993f91c31a8172f4bfb7c44fc8d6ab15fda53ec06109b0c33dd14fda9.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/ba9f4c5993f91c31a8172f4bfb7c44fc8d6ab15fda53ec06109b0c33dd14fda9.jpg)

![bda6d7ce903c3c9542f9daecf3482876c161e3651b5aa405ceb6221422c40d13.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/bda6d7ce903c3c9542f9daecf3482876c161e3651b5aa405ceb6221422c40d13.jpg)

![c5d1985fa53e02af6014699b659bf3e964431e52b53dbfcd7c60cec6591d606d.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/c5d1985fa53e02af6014699b659bf3e964431e52b53dbfcd7c60cec6591d606d.jpg)

![cbd524fe186daec226d7d0370c30d2deb7ecfa521a3835adf36d8a363a9d3e98.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/cbd524fe186daec226d7d0370c30d2deb7ecfa521a3835adf36d8a363a9d3e98.jpg)

![d34e810677af912bf16aaa0117c197da78513f98254b71b7579dc4f186b2c545.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/d34e810677af912bf16aaa0117c197da78513f98254b71b7579dc4f186b2c545.jpg)

![d740c07ed90bf9b222bf1a76e6fe44be52ce79a5cde47764c76af934a49a2c0f.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/d740c07ed90bf9b222bf1a76e6fe44be52ce79a5cde47764c76af934a49a2c0f.jpg)

![d81071dbdc64879b12c8aae8b04f55aa7c3d3fbd5f643b80b61388286dcfe74a.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/d81071dbdc64879b12c8aae8b04f55aa7c3d3fbd5f643b80b61388286dcfe74a.jpg)

![d91b4570f496f4adb2eb7a25c19f40491055f34c0233bed565e07634c04c1e25.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/d91b4570f496f4adb2eb7a25c19f40491055f34c0233bed565e07634c04c1e25.jpg)

![df97465672432c0024ae5a04b2cee66296960e1711a7b80f4c0df22f483cb13b.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/df97465672432c0024ae5a04b2cee66296960e1711a7b80f4c0df22f483cb13b.jpg)

![e56345c677f838cce1987ed0139aab3ab898419243ed14056978dd953f0539d2.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/e56345c677f838cce1987ed0139aab3ab898419243ed14056978dd953f0539d2.jpg)

![ef2eb5c605af6500ce8ae7f2ab236f0583617a985e7d1ab516b9ec092de51461.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/ef2eb5c605af6500ce8ae7f2ab236f0583617a985e7d1ab516b9ec092de51461.jpg)

![f21bf6a3e2f1879aa924b69c4378070a31f645d23934aabfa497b449ffb2108f.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/f21bf6a3e2f1879aa924b69c4378070a31f645d23934aabfa497b449ffb2108f.jpg)

![f5edd732237b55761fb8c911c6ab7f7b41f8cc0cc78f66391f1a7b99f03df1b3.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/images/f5edd732237b55761fb8c911c6ab7f7b41f8cc0cc78f66391f1a7b99f03df1b3.jpg)

### Tables

![856e8d7a271a5cf3c0e5699ef023be9d4abdb80a60474ad2a88fb2dbbad705af.jpg](../iclr_results/635_Do Stochastic, Feel Noiseless_ Stable Stochastic Optimization via a Double Momentum Mechanism/tables/856e8d7a271a5cf3c0e5699ef023be9d4abdb80a60474ad2a88fb2dbbad705af.jpg)

## SSOLE: Rethinking Orthogonal Low-rank Embedding for Self-Supervised Learning


### Images

![40582dc8f9a39d21c4b1f5d9d4d2272c9719aeb573d0a876068f7cad1848378c.jpg](../iclr_results/636_SSOLE_ Rethinking Orthogonal Low-rank Embedding for Self-Supervised Learning/images/40582dc8f9a39d21c4b1f5d9d4d2272c9719aeb573d0a876068f7cad1848378c.jpg)

### Tables

![3ec52ec694acef74e11af05a15342262bcd213252282f915bc623adfdf929146.jpg](../iclr_results/636_SSOLE_ Rethinking Orthogonal Low-rank Embedding for Self-Supervised Learning/tables/3ec52ec694acef74e11af05a15342262bcd213252282f915bc623adfdf929146.jpg)

![75ae18a92d27a2a8f6741a5a377931ac012f0c45774934b9029104b411447bbe.jpg](../iclr_results/636_SSOLE_ Rethinking Orthogonal Low-rank Embedding for Self-Supervised Learning/tables/75ae18a92d27a2a8f6741a5a377931ac012f0c45774934b9029104b411447bbe.jpg)

![94ac5a9c782919ac7674a1ac6839a69076ce0f21b53246f6694965ad96fcf611.jpg](../iclr_results/636_SSOLE_ Rethinking Orthogonal Low-rank Embedding for Self-Supervised Learning/tables/94ac5a9c782919ac7674a1ac6839a69076ce0f21b53246f6694965ad96fcf611.jpg)

![c5f4630b079d7ccb9a8d95b99f5a25a6c96086ed94abd2a3297c66da2f57a984.jpg](../iclr_results/636_SSOLE_ Rethinking Orthogonal Low-rank Embedding for Self-Supervised Learning/tables/c5f4630b079d7ccb9a8d95b99f5a25a6c96086ed94abd2a3297c66da2f57a984.jpg)

![dd9a83b703aa80591fcc07dd940aa94e73ab48deb37da462ef4c4b854133a2e3.jpg](../iclr_results/636_SSOLE_ Rethinking Orthogonal Low-rank Embedding for Self-Supervised Learning/tables/dd9a83b703aa80591fcc07dd940aa94e73ab48deb37da462ef4c4b854133a2e3.jpg)

![e57ad3354e983722b17f3d28d7aa258d674b35584bd1b81aacf086da1aed20ef.jpg](../iclr_results/636_SSOLE_ Rethinking Orthogonal Low-rank Embedding for Self-Supervised Learning/tables/e57ad3354e983722b17f3d28d7aa258d674b35584bd1b81aacf086da1aed20ef.jpg)

## EgoSim: Egocentric Exploration in Virtual Worlds with Multi-modal Conditioning


### Images

![50e64230b55f3aa28ade0511f2782f1f5d9da86383087db79fdabb0573d8a6a7.jpg](../iclr_results/637_EgoSim_ Egocentric Exploration in Virtual Worlds with Multi-modal Conditioning/images/50e64230b55f3aa28ade0511f2782f1f5d9da86383087db79fdabb0573d8a6a7.jpg)

![5e631c16b0be1a18051a3e06573b2784a80422e703d060a1c14fd84aa78feb9f.jpg](../iclr_results/637_EgoSim_ Egocentric Exploration in Virtual Worlds with Multi-modal Conditioning/images/5e631c16b0be1a18051a3e06573b2784a80422e703d060a1c14fd84aa78feb9f.jpg)

![6cddc446888ab2e24c6e329a51d2d059fcbcb353cabf07d2fb2e7bfbf0764251.jpg](../iclr_results/637_EgoSim_ Egocentric Exploration in Virtual Worlds with Multi-modal Conditioning/images/6cddc446888ab2e24c6e329a51d2d059fcbcb353cabf07d2fb2e7bfbf0764251.jpg)

![6e0ebd52bb5b67994188b1b5a384f6e8ea06c346a12f4771eca66927ccd51bcc.jpg](../iclr_results/637_EgoSim_ Egocentric Exploration in Virtual Worlds with Multi-modal Conditioning/images/6e0ebd52bb5b67994188b1b5a384f6e8ea06c346a12f4771eca66927ccd51bcc.jpg)

![75548418f8d1e7906a4dca196801bb08c444077a6547ac5dcb79f50a3188afbd.jpg](../iclr_results/637_EgoSim_ Egocentric Exploration in Virtual Worlds with Multi-modal Conditioning/images/75548418f8d1e7906a4dca196801bb08c444077a6547ac5dcb79f50a3188afbd.jpg)

![88098b7d4606b00a4a96bf8a4bffb2731b73fb7691f00f7ce8068c6aae77de34.jpg](../iclr_results/637_EgoSim_ Egocentric Exploration in Virtual Worlds with Multi-modal Conditioning/images/88098b7d4606b00a4a96bf8a4bffb2731b73fb7691f00f7ce8068c6aae77de34.jpg)

![ace0d1c71cc60368104458f43af79453fcc3a8ca32cd56cd7c58aa6a4e1da89a.jpg](../iclr_results/637_EgoSim_ Egocentric Exploration in Virtual Worlds with Multi-modal Conditioning/images/ace0d1c71cc60368104458f43af79453fcc3a8ca32cd56cd7c58aa6a4e1da89a.jpg)

![adaac28e48848cef97a6608a8f81efefb840d65a63a38770caacfbc408f1ccc2.jpg](../iclr_results/637_EgoSim_ Egocentric Exploration in Virtual Worlds with Multi-modal Conditioning/images/adaac28e48848cef97a6608a8f81efefb840d65a63a38770caacfbc408f1ccc2.jpg)

![cb0b51ab3433831bd6a7f7c893f655e61bbeb93f7c396283aeece178dfc1d566.jpg](../iclr_results/637_EgoSim_ Egocentric Exploration in Virtual Worlds with Multi-modal Conditioning/images/cb0b51ab3433831bd6a7f7c893f655e61bbeb93f7c396283aeece178dfc1d566.jpg)

![e3d1e759f97af2aa9d6b0993ec5286704166747693218fe3a455383076f7acc4.jpg](../iclr_results/637_EgoSim_ Egocentric Exploration in Virtual Worlds with Multi-modal Conditioning/images/e3d1e759f97af2aa9d6b0993ec5286704166747693218fe3a455383076f7acc4.jpg)

![eccc9ce241b7486d5c23a8094695a13babc676d94ef4594332d9f77ea31c3ce0.jpg](../iclr_results/637_EgoSim_ Egocentric Exploration in Virtual Worlds with Multi-modal Conditioning/images/eccc9ce241b7486d5c23a8094695a13babc676d94ef4594332d9f77ea31c3ce0.jpg)

![ef0107fc05e7b16b95f2af802f39fd3f5af58e9f0e115188f21cf2a6a49447c4.jpg](../iclr_results/637_EgoSim_ Egocentric Exploration in Virtual Worlds with Multi-modal Conditioning/images/ef0107fc05e7b16b95f2af802f39fd3f5af58e9f0e115188f21cf2a6a49447c4.jpg)

![f164c6fd6dcfe7181d6b451fa87921c2a0d42493c44302561a04bc225629461b.jpg](../iclr_results/637_EgoSim_ Egocentric Exploration in Virtual Worlds with Multi-modal Conditioning/images/f164c6fd6dcfe7181d6b451fa87921c2a0d42493c44302561a04bc225629461b.jpg)

### Tables

![01ba0a27b036d5dc4275489b8ea091e5f2d4c30afd9358702b5eaa7d913d1959.jpg](../iclr_results/637_EgoSim_ Egocentric Exploration in Virtual Worlds with Multi-modal Conditioning/tables/01ba0a27b036d5dc4275489b8ea091e5f2d4c30afd9358702b5eaa7d913d1959.jpg)

![6a7faa0145772b52b1b53ccc2d8441db01c44fc54ec593a2da8e84e26b7bcf70.jpg](../iclr_results/637_EgoSim_ Egocentric Exploration in Virtual Worlds with Multi-modal Conditioning/tables/6a7faa0145772b52b1b53ccc2d8441db01c44fc54ec593a2da8e84e26b7bcf70.jpg)

![7665450b85500bae47a50f70b0608b554db5ca555a68138a366f017c18469953.jpg](../iclr_results/637_EgoSim_ Egocentric Exploration in Virtual Worlds with Multi-modal Conditioning/tables/7665450b85500bae47a50f70b0608b554db5ca555a68138a366f017c18469953.jpg)

![e984361b1006aa0d8758de8c6737b00d9da3f8030d7df4a2d430ea7afca4d04b.jpg](../iclr_results/637_EgoSim_ Egocentric Exploration in Virtual Worlds with Multi-modal Conditioning/tables/e984361b1006aa0d8758de8c6737b00d9da3f8030d7df4a2d430ea7afca4d04b.jpg)

![f37c8dc1a285bc0596db1973fee4c04c020578b54a58ccee8459fff841a2f357.jpg](../iclr_results/637_EgoSim_ Egocentric Exploration in Virtual Worlds with Multi-modal Conditioning/tables/f37c8dc1a285bc0596db1973fee4c04c020578b54a58ccee8459fff841a2f357.jpg)

## Deep Kernel Relative Test for Machine-generated Text Detection


### Images

![15b670f379b7ca535b05d17e1688d2c2da4f43a734be9a730dc55ab3237288c9.jpg](../iclr_results/638_Deep Kernel Relative Test for Machine-generated Text Detection/images/15b670f379b7ca535b05d17e1688d2c2da4f43a734be9a730dc55ab3237288c9.jpg)

![d99e1aca593213bbd90fc98d9c403ef31e47a94091acc498c1de4935a4f1280a.jpg](../iclr_results/638_Deep Kernel Relative Test for Machine-generated Text Detection/images/d99e1aca593213bbd90fc98d9c403ef31e47a94091acc498c1de4935a4f1280a.jpg)

![dacf0b1952bb6ac6a48d12448ec111dd3f2d6ed46aa9c25d59b6d5bd799b3d9c.jpg](../iclr_results/638_Deep Kernel Relative Test for Machine-generated Text Detection/images/dacf0b1952bb6ac6a48d12448ec111dd3f2d6ed46aa9c25d59b6d5bd799b3d9c.jpg)

### Tables

![039755f95ea70764ba8f63825ebde764a5e6c1204f6d0da53c580bc2d6d6120e.jpg](../iclr_results/638_Deep Kernel Relative Test for Machine-generated Text Detection/tables/039755f95ea70764ba8f63825ebde764a5e6c1204f6d0da53c580bc2d6d6120e.jpg)

![05373338c944bc78207c4ccc5d4b7295e042b9d8582e0f9a087d66067506ca43.jpg](../iclr_results/638_Deep Kernel Relative Test for Machine-generated Text Detection/tables/05373338c944bc78207c4ccc5d4b7295e042b9d8582e0f9a087d66067506ca43.jpg)

![16cf57a86cfab194b6e03e32193200fd74adb1d91ce5e8d327613088aa5d8901.jpg](../iclr_results/638_Deep Kernel Relative Test for Machine-generated Text Detection/tables/16cf57a86cfab194b6e03e32193200fd74adb1d91ce5e8d327613088aa5d8901.jpg)

![265232298a02990a9161045af9395cb4d2ca47711ab06a7d5fdd061593468809.jpg](../iclr_results/638_Deep Kernel Relative Test for Machine-generated Text Detection/tables/265232298a02990a9161045af9395cb4d2ca47711ab06a7d5fdd061593468809.jpg)

![2d2aa310517a7cfe7e5801abb04edffabdebea2f556ced28cfd79ce2b4ac89bb.jpg](../iclr_results/638_Deep Kernel Relative Test for Machine-generated Text Detection/tables/2d2aa310517a7cfe7e5801abb04edffabdebea2f556ced28cfd79ce2b4ac89bb.jpg)

![6d998b40896ae7083e310fe251a661cd1b4ebe6af02bfd4f014d184c2457d9ed.jpg](../iclr_results/638_Deep Kernel Relative Test for Machine-generated Text Detection/tables/6d998b40896ae7083e310fe251a661cd1b4ebe6af02bfd4f014d184c2457d9ed.jpg)

![84721ce5ad9bb61f2e05f53096de66ae17264fd5c42737f676314b23a5787d15.jpg](../iclr_results/638_Deep Kernel Relative Test for Machine-generated Text Detection/tables/84721ce5ad9bb61f2e05f53096de66ae17264fd5c42737f676314b23a5787d15.jpg)

![85eb7d44c4430abb927ef88e16e3279fa86f005c843da2153e2efd5420a3fa75.jpg](../iclr_results/638_Deep Kernel Relative Test for Machine-generated Text Detection/tables/85eb7d44c4430abb927ef88e16e3279fa86f005c843da2153e2efd5420a3fa75.jpg)

![8a2830c4614cf4609ce271b6201cb3202d20b00a61704030d340785c691323f6.jpg](../iclr_results/638_Deep Kernel Relative Test for Machine-generated Text Detection/tables/8a2830c4614cf4609ce271b6201cb3202d20b00a61704030d340785c691323f6.jpg)

![9fa14f149056f1ed8953f9435c43307249dd5b5fcd4591b47788fe7150e5f462.jpg](../iclr_results/638_Deep Kernel Relative Test for Machine-generated Text Detection/tables/9fa14f149056f1ed8953f9435c43307249dd5b5fcd4591b47788fe7150e5f462.jpg)

![d7e36923d74bd6a2248d2431a744b00586025e0be8cdfe5887a5aef17b7f0980.jpg](../iclr_results/638_Deep Kernel Relative Test for Machine-generated Text Detection/tables/d7e36923d74bd6a2248d2431a744b00586025e0be8cdfe5887a5aef17b7f0980.jpg)

![da33435d1342d1379eaf740f4b24f98fd0b07301eaf9b5569f420deb850a66d6.jpg](../iclr_results/638_Deep Kernel Relative Test for Machine-generated Text Detection/tables/da33435d1342d1379eaf740f4b24f98fd0b07301eaf9b5569f420deb850a66d6.jpg)

![dce979cf1f9be24a973334c0c8906ff9288132e7678ddba4b90f64bedb04db13.jpg](../iclr_results/638_Deep Kernel Relative Test for Machine-generated Text Detection/tables/dce979cf1f9be24a973334c0c8906ff9288132e7678ddba4b90f64bedb04db13.jpg)

![f4cb52ad495e8c3afec3a865d55ea94ead80991185b1ecd5bb3bc44bb47a5797.jpg](../iclr_results/638_Deep Kernel Relative Test for Machine-generated Text Detection/tables/f4cb52ad495e8c3afec3a865d55ea94ead80991185b1ecd5bb3bc44bb47a5797.jpg)

![f744f58307466e2bae666667c8281652b9b4d6a50f1c5c7dc00589f5e8816eca.jpg](../iclr_results/638_Deep Kernel Relative Test for Machine-generated Text Detection/tables/f744f58307466e2bae666667c8281652b9b4d6a50f1c5c7dc00589f5e8816eca.jpg)

## Random Is All You Need: Random Noise Injection on Feature Statistics for Generalizable Deep Image Denoising


### Images

![03c71f5296a493c3fef7a96f6f0e26b72ede310fcdd2fab3f0c1561a91a83280.jpg](../iclr_results/639_Random Is All You Need_ Random Noise Injection on Feature Statistics for Generalizable Deep Image De/images/03c71f5296a493c3fef7a96f6f0e26b72ede310fcdd2fab3f0c1561a91a83280.jpg)

![059494583ab97562f5563756e3c1359028c82efb894f662ffe34a07094497808.jpg](../iclr_results/639_Random Is All You Need_ Random Noise Injection on Feature Statistics for Generalizable Deep Image De/images/059494583ab97562f5563756e3c1359028c82efb894f662ffe34a07094497808.jpg)

![214998bceffb37f10d5ec289f21bde3ecba0dc9216ea82c69555879c6c1c1db4.jpg](../iclr_results/639_Random Is All You Need_ Random Noise Injection on Feature Statistics for Generalizable Deep Image De/images/214998bceffb37f10d5ec289f21bde3ecba0dc9216ea82c69555879c6c1c1db4.jpg)

![2ec683e67ab1a4e15f7b332133a13cdadf9b3a41a37554b10c281dbea3f34bbb.jpg](../iclr_results/639_Random Is All You Need_ Random Noise Injection on Feature Statistics for Generalizable Deep Image De/images/2ec683e67ab1a4e15f7b332133a13cdadf9b3a41a37554b10c281dbea3f34bbb.jpg)

![4e7d778bcb74b3feed3a94b8a60316f6c0a68675bac4743ac78e470b9d3ccb5b.jpg](../iclr_results/639_Random Is All You Need_ Random Noise Injection on Feature Statistics for Generalizable Deep Image De/images/4e7d778bcb74b3feed3a94b8a60316f6c0a68675bac4743ac78e470b9d3ccb5b.jpg)

![685cfe8eecb51fa4c806416b860ad94331be8734fc45a9ec316e3df798a6b7d5.jpg](../iclr_results/639_Random Is All You Need_ Random Noise Injection on Feature Statistics for Generalizable Deep Image De/images/685cfe8eecb51fa4c806416b860ad94331be8734fc45a9ec316e3df798a6b7d5.jpg)

![a7d80c8127d9c392f45ad403813f637690d4f97ec74f6e6f67960c6418e5349c.jpg](../iclr_results/639_Random Is All You Need_ Random Noise Injection on Feature Statistics for Generalizable Deep Image De/images/a7d80c8127d9c392f45ad403813f637690d4f97ec74f6e6f67960c6418e5349c.jpg)

![cc8622a7888c9084650895f83d9cb443aeac3cefb7e4b7a6ac8e1ec5058540c7.jpg](../iclr_results/639_Random Is All You Need_ Random Noise Injection on Feature Statistics for Generalizable Deep Image De/images/cc8622a7888c9084650895f83d9cb443aeac3cefb7e4b7a6ac8e1ec5058540c7.jpg)

![dbca333045d3a3b8285e42633467c4d1310e70f65995441532e147fc881bcaa4.jpg](../iclr_results/639_Random Is All You Need_ Random Noise Injection on Feature Statistics for Generalizable Deep Image De/images/dbca333045d3a3b8285e42633467c4d1310e70f65995441532e147fc881bcaa4.jpg)

### Tables

![048c81d7bed36f839a6e3c20fa7a801c0f04120c24cfcb189fd30533785bc015.jpg](../iclr_results/639_Random Is All You Need_ Random Noise Injection on Feature Statistics for Generalizable Deep Image De/tables/048c81d7bed36f839a6e3c20fa7a801c0f04120c24cfcb189fd30533785bc015.jpg)

![1e3484695ffb7a761c4e6b549d050b64a49d9fc7bf9d22689ac53630817bc7fd.jpg](../iclr_results/639_Random Is All You Need_ Random Noise Injection on Feature Statistics for Generalizable Deep Image De/tables/1e3484695ffb7a761c4e6b549d050b64a49d9fc7bf9d22689ac53630817bc7fd.jpg)

![78d827a5ab274a78581d46e445f748be9fdca0c3d02ecfefba65afa29b41bec3.jpg](../iclr_results/639_Random Is All You Need_ Random Noise Injection on Feature Statistics for Generalizable Deep Image De/tables/78d827a5ab274a78581d46e445f748be9fdca0c3d02ecfefba65afa29b41bec3.jpg)

![a7147d8457a7f71334f151d48c0b8ab2e0d2538298c1e61f050dc509249e6599.jpg](../iclr_results/639_Random Is All You Need_ Random Noise Injection on Feature Statistics for Generalizable Deep Image De/tables/a7147d8457a7f71334f151d48c0b8ab2e0d2538298c1e61f050dc509249e6599.jpg)

![ad71b4aa655ce36745da2a85570ed7ddced0a88a1159740248982c45ca1207c0.jpg](../iclr_results/639_Random Is All You Need_ Random Noise Injection on Feature Statistics for Generalizable Deep Image De/tables/ad71b4aa655ce36745da2a85570ed7ddced0a88a1159740248982c45ca1207c0.jpg)

![be513efe1c75646bf157a9bbbfa955e680b22f1f7e81ce8a21fffc23025f5c65.jpg](../iclr_results/639_Random Is All You Need_ Random Noise Injection on Feature Statistics for Generalizable Deep Image De/tables/be513efe1c75646bf157a9bbbfa955e680b22f1f7e81ce8a21fffc23025f5c65.jpg)

![cb4783c581e127ba27af2f230358f702173869df06c0f9bfa33d74ea677744ee.jpg](../iclr_results/639_Random Is All You Need_ Random Noise Injection on Feature Statistics for Generalizable Deep Image De/tables/cb4783c581e127ba27af2f230358f702173869df06c0f9bfa33d74ea677744ee.jpg)

![e103b8818f2ec5b543459ad93bd431fffb49c5f4a30b72e1065f710be35c5515.jpg](../iclr_results/639_Random Is All You Need_ Random Noise Injection on Feature Statistics for Generalizable Deep Image De/tables/e103b8818f2ec5b543459ad93bd431fffb49c5f4a30b72e1065f710be35c5515.jpg)

![ee1a2464a0d0b63ab809bc194c2911fd83c7704bc897015e288d0575a4247f57.jpg](../iclr_results/639_Random Is All You Need_ Random Noise Injection on Feature Statistics for Generalizable Deep Image De/tables/ee1a2464a0d0b63ab809bc194c2911fd83c7704bc897015e288d0575a4247f57.jpg)

![f7a13b4d67b87ba7c57d6ed222f6521d76506eb7993ac69a9bae401a46a34a1f.jpg](../iclr_results/639_Random Is All You Need_ Random Noise Injection on Feature Statistics for Generalizable Deep Image De/tables/f7a13b4d67b87ba7c57d6ed222f6521d76506eb7993ac69a9bae401a46a34a1f.jpg)

![ff47c332a262c86fb20e44dfb5b3c4bb37ab1a776f43cff2607750ffddffcad3.jpg](../iclr_results/639_Random Is All You Need_ Random Noise Injection on Feature Statistics for Generalizable Deep Image De/tables/ff47c332a262c86fb20e44dfb5b3c4bb37ab1a776f43cff2607750ffddffcad3.jpg)

## GOAL: A Generalist Combinatorial Optimization Agent Learner


### Images

![012f6dc6b7254eb3ba28da4d4282e71e009c6af613cbb6fc8e2c6a91ba28840c.jpg](../iclr_results/640_GOAL_ A Generalist Combinatorial Optimization Agent Learner/images/012f6dc6b7254eb3ba28da4d4282e71e009c6af613cbb6fc8e2c6a91ba28840c.jpg)

![0f33db3125e36371976784578bc32e340dbf74fe46338696560df6a47fe4dc9e.jpg](../iclr_results/640_GOAL_ A Generalist Combinatorial Optimization Agent Learner/images/0f33db3125e36371976784578bc32e340dbf74fe46338696560df6a47fe4dc9e.jpg)

![130a51573dee7f7a18101726a4faac2abbfadb1075ff065917e2e2e4f9555022.jpg](../iclr_results/640_GOAL_ A Generalist Combinatorial Optimization Agent Learner/images/130a51573dee7f7a18101726a4faac2abbfadb1075ff065917e2e2e4f9555022.jpg)

![19b78199dff30aa6adb37523d3e416f6d7738799c744538948cb7c3172ec5360.jpg](../iclr_results/640_GOAL_ A Generalist Combinatorial Optimization Agent Learner/images/19b78199dff30aa6adb37523d3e416f6d7738799c744538948cb7c3172ec5360.jpg)

![29f1701c5c9c81947accc976ded94d43007ce05bd8ffcf340bb01b34229660aa.jpg](../iclr_results/640_GOAL_ A Generalist Combinatorial Optimization Agent Learner/images/29f1701c5c9c81947accc976ded94d43007ce05bd8ffcf340bb01b34229660aa.jpg)

![3b619ce6dcab1fc52afc5f3a05f7ed7db9e73cf64a6af882c98f3719998c7ed6.jpg](../iclr_results/640_GOAL_ A Generalist Combinatorial Optimization Agent Learner/images/3b619ce6dcab1fc52afc5f3a05f7ed7db9e73cf64a6af882c98f3719998c7ed6.jpg)

![534b9066740e1b9c71e9f0a91b73204bc635f1963c62cadf803fd751cbae3a41.jpg](../iclr_results/640_GOAL_ A Generalist Combinatorial Optimization Agent Learner/images/534b9066740e1b9c71e9f0a91b73204bc635f1963c62cadf803fd751cbae3a41.jpg)

![7b44a2a1643e4a9767cd080584d105c360ada3ee62f35d683bd5b8d59c3d9197.jpg](../iclr_results/640_GOAL_ A Generalist Combinatorial Optimization Agent Learner/images/7b44a2a1643e4a9767cd080584d105c360ada3ee62f35d683bd5b8d59c3d9197.jpg)

![7e5c8df0c30e5b272a0a218db81018eb8928502912ed773f11c3cbb9de7a3ee5.jpg](../iclr_results/640_GOAL_ A Generalist Combinatorial Optimization Agent Learner/images/7e5c8df0c30e5b272a0a218db81018eb8928502912ed773f11c3cbb9de7a3ee5.jpg)

![82cb06fbbe2a6d489f6d485491068fc291dc72d6e3ffe0508ae0bfc50ac1a373.jpg](../iclr_results/640_GOAL_ A Generalist Combinatorial Optimization Agent Learner/images/82cb06fbbe2a6d489f6d485491068fc291dc72d6e3ffe0508ae0bfc50ac1a373.jpg)

![9fc1a9ec3690ab9cd6a702ee83062db48e9bbcea89d0a02108f48e28e6de927d.jpg](../iclr_results/640_GOAL_ A Generalist Combinatorial Optimization Agent Learner/images/9fc1a9ec3690ab9cd6a702ee83062db48e9bbcea89d0a02108f48e28e6de927d.jpg)

![ba5ac1e5fe5b23742223a84735f3cdeeb040b23c3cdcc6e3df50e38c6afc74a9.jpg](../iclr_results/640_GOAL_ A Generalist Combinatorial Optimization Agent Learner/images/ba5ac1e5fe5b23742223a84735f3cdeeb040b23c3cdcc6e3df50e38c6afc74a9.jpg)

![c935a7727daab91f6f257b78f6797ef0c2eb6fc5eb22c4db9e6bc14975e9d6d9.jpg](../iclr_results/640_GOAL_ A Generalist Combinatorial Optimization Agent Learner/images/c935a7727daab91f6f257b78f6797ef0c2eb6fc5eb22c4db9e6bc14975e9d6d9.jpg)

![d0e4ccf21f77d25b4c925bd2e2c08505f397db25110946ffc0376bfc9404ab61.jpg](../iclr_results/640_GOAL_ A Generalist Combinatorial Optimization Agent Learner/images/d0e4ccf21f77d25b4c925bd2e2c08505f397db25110946ffc0376bfc9404ab61.jpg)

![d9b099ba590b9e56d3a83677692788640878f5c43d470afe1905fff0f4656e10.jpg](../iclr_results/640_GOAL_ A Generalist Combinatorial Optimization Agent Learner/images/d9b099ba590b9e56d3a83677692788640878f5c43d470afe1905fff0f4656e10.jpg)

![f49ab30c53719f3d344059d41321db46b3bf29b237d19dcc070eba26bf875936.jpg](../iclr_results/640_GOAL_ A Generalist Combinatorial Optimization Agent Learner/images/f49ab30c53719f3d344059d41321db46b3bf29b237d19dcc070eba26bf875936.jpg)

### Tables

![3846540ba30762d2d39acac1562cac4880e65defea5021151e02e882a971fc48.jpg](../iclr_results/640_GOAL_ A Generalist Combinatorial Optimization Agent Learner/tables/3846540ba30762d2d39acac1562cac4880e65defea5021151e02e882a971fc48.jpg)

![3d9d38de80eb5114534d0909d1c96124b51f1f636ae12ba2da7dc66c1a9d247f.jpg](../iclr_results/640_GOAL_ A Generalist Combinatorial Optimization Agent Learner/tables/3d9d38de80eb5114534d0909d1c96124b51f1f636ae12ba2da7dc66c1a9d247f.jpg)

## Integral Performance Approximation for Continuous-Time Reinforcement Learning Control


### Images

![11c298efee741ae2e58b0c444696b3f5bf9f829f618fc49846f2426e161398ff.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/images/11c298efee741ae2e58b0c444696b3f5bf9f829f618fc49846f2426e161398ff.jpg)

![1d52534a29ee64fa42aa13bc10c33c830cdf52ad4144f33a099c33160ee879e0.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/images/1d52534a29ee64fa42aa13bc10c33c830cdf52ad4144f33a099c33160ee879e0.jpg)

![2ae45aa9c517fb544277ca8c097a9ad1a66f760139f57e0a7adcf3b96937dcd7.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/images/2ae45aa9c517fb544277ca8c097a9ad1a66f760139f57e0a7adcf3b96937dcd7.jpg)

![32a6a81f4053f26de3548c83ca0a9333e523426706e84e8d05e57ff11fb4104f.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/images/32a6a81f4053f26de3548c83ca0a9333e523426706e84e8d05e57ff11fb4104f.jpg)

![6082cf5a3d2b85de9bdbe82fdbd336d71ac96d245bf52167c0dd3deb97e5f1aa.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/images/6082cf5a3d2b85de9bdbe82fdbd336d71ac96d245bf52167c0dd3deb97e5f1aa.jpg)

![619e00e9f5dc3241e4957cf4d59a83c84f60aaa9f9ced2e0c8ce7a38d42c1ed5.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/images/619e00e9f5dc3241e4957cf4d59a83c84f60aaa9f9ced2e0c8ce7a38d42c1ed5.jpg)

![8dfdb0aa17137cb3a08f3751cbeb2534f485eaeb0f3de84c5b1b3d3d4c7c16ed.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/images/8dfdb0aa17137cb3a08f3751cbeb2534f485eaeb0f3de84c5b1b3d3d4c7c16ed.jpg)

![9178f964743417dd03d08cbb4b29f5189cc8c152cb41d2d055be73f8b53a2921.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/images/9178f964743417dd03d08cbb4b29f5189cc8c152cb41d2d055be73f8b53a2921.jpg)

![9da5fa848fc71f7d724a6280935e2e960813538258916cc52c22512a80ebf126.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/images/9da5fa848fc71f7d724a6280935e2e960813538258916cc52c22512a80ebf126.jpg)

![a8dc54c0b567a6ce5aa8d526b0ea4e8b6440bf098ece144c4d20e72267559c7c.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/images/a8dc54c0b567a6ce5aa8d526b0ea4e8b6440bf098ece144c4d20e72267559c7c.jpg)

![bb8c7e631646b874c48119e9496e8d299ad0ea35d3248d39c3ea883c23a376ce.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/images/bb8c7e631646b874c48119e9496e8d299ad0ea35d3248d39c3ea883c23a376ce.jpg)

![c07c2ca54e4d1a2ecc77f5e43ee3f2dd9c5884d94396a3bc65d69a5249afcb84.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/images/c07c2ca54e4d1a2ecc77f5e43ee3f2dd9c5884d94396a3bc65d69a5249afcb84.jpg)

![cb0a68a70f9662146c3eebbc9c534b67e79e38e247b1304fc18ef43454d633d6.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/images/cb0a68a70f9662146c3eebbc9c534b67e79e38e247b1304fc18ef43454d633d6.jpg)

![d4e5f7c6c0f5de003071eb86cff68a27c55640c76cece38a524cfdcc8a2e7fef.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/images/d4e5f7c6c0f5de003071eb86cff68a27c55640c76cece38a524cfdcc8a2e7fef.jpg)

![e12ac4b9a69a1318418f999288e57becdd2961d0743cc7717b64d691a81a8e41.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/images/e12ac4b9a69a1318418f999288e57becdd2961d0743cc7717b64d691a81a8e41.jpg)

![e5cc5242641da8922f3f63da5fbc4582958b779022fb8dca5e936aee012746bb.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/images/e5cc5242641da8922f3f63da5fbc4582958b779022fb8dca5e936aee012746bb.jpg)

![f2b63d03024c59fc532951c72ed4941ef73a6d925ae6fe26eddf7573b64ebb77.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/images/f2b63d03024c59fc532951c72ed4941ef73a6d925ae6fe26eddf7573b64ebb77.jpg)

![f454018c5c50af5547a610e6573ec80d7694afa49594898b45a72b4b5c322376.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/images/f454018c5c50af5547a610e6573ec80d7694afa49594898b45a72b4b5c322376.jpg)

![f937f9dc8911c76f4a105d9ca45dc1a2fb1575093ac433ab1b9d8873c1b28e68.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/images/f937f9dc8911c76f4a105d9ca45dc1a2fb1575093ac433ab1b9d8873c1b28e68.jpg)

### Tables

![0158d843f76fd5663e6b6b4da595280f28a0591f63df9c181295b09d23d405a3.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/tables/0158d843f76fd5663e6b6b4da595280f28a0591f63df9c181295b09d23d405a3.jpg)

![3641b030800ba5682ad9896b73303ed31d8f173fbcbfae085109d1f497ea416e.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/tables/3641b030800ba5682ad9896b73303ed31d8f173fbcbfae085109d1f497ea416e.jpg)

![385046aeaf879557760a48fb1889897850561210a53fd61725b6e7e700cc169b.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/tables/385046aeaf879557760a48fb1889897850561210a53fd61725b6e7e700cc169b.jpg)

![3a58ca3d69185716c07d795414756c5b50e7750f9d95d51a3980ac8d1cbaaf8e.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/tables/3a58ca3d69185716c07d795414756c5b50e7750f9d95d51a3980ac8d1cbaaf8e.jpg)

![3c3ab9bad65f7f0062262de358b4fb20aec6bf8ff7b710262f30edf719428ff6.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/tables/3c3ab9bad65f7f0062262de358b4fb20aec6bf8ff7b710262f30edf719428ff6.jpg)

![43bdb03474d08dbee9737b84ab4c367846bdf3db07a3fba1696e2c6feb250a2b.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/tables/43bdb03474d08dbee9737b84ab4c367846bdf3db07a3fba1696e2c6feb250a2b.jpg)

![47373e6af1d11f1e085b81324f48668d737d4d3fd26447fb712e467e14065b6e.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/tables/47373e6af1d11f1e085b81324f48668d737d4d3fd26447fb712e467e14065b6e.jpg)

![480489814043516fa2cfa3124bab1aa4ffa418f5904d5053b0dbaa1da4914767.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/tables/480489814043516fa2cfa3124bab1aa4ffa418f5904d5053b0dbaa1da4914767.jpg)

![601d005852b392209bfd1a3ea0130eb01228ad5d40d25c605195ef7413829122.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/tables/601d005852b392209bfd1a3ea0130eb01228ad5d40d25c605195ef7413829122.jpg)

![854b90667a9107a1fb5baabe10b7b94dc326548b63f8de2dd84949d9da27ec10.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/tables/854b90667a9107a1fb5baabe10b7b94dc326548b63f8de2dd84949d9da27ec10.jpg)

![8b3eb6a70d30ccfec40fe850268652fd924a5ec8fe5006489ac393e3a7fb14c0.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/tables/8b3eb6a70d30ccfec40fe850268652fd924a5ec8fe5006489ac393e3a7fb14c0.jpg)

![8bacc71ff41096844faf18f77b6b88fca83a4e419b5c1c5b551632893cda6e77.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/tables/8bacc71ff41096844faf18f77b6b88fca83a4e419b5c1c5b551632893cda6e77.jpg)

![a9b7ef5c446d59b02b26165d966b6fb9ec6927fad99367ebad3dc7f7b5883c65.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/tables/a9b7ef5c446d59b02b26165d966b6fb9ec6927fad99367ebad3dc7f7b5883c65.jpg)

![db3fed1bb407cbc4fd27776f331e99116cc2649e209890f41f48d0f3a2e7ec4c.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/tables/db3fed1bb407cbc4fd27776f331e99116cc2649e209890f41f48d0f3a2e7ec4c.jpg)

![ec9254bf529d48d97faf02b92db61ca424e8b38814193bf1381f2584bff2c184.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/tables/ec9254bf529d48d97faf02b92db61ca424e8b38814193bf1381f2584bff2c184.jpg)

![edebf3fe9e925c744c6ef533ee433d94e42ca9a1cbbdc92e4264878f57b3fc42.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/tables/edebf3fe9e925c744c6ef533ee433d94e42ca9a1cbbdc92e4264878f57b3fc42.jpg)

![ee569cad26fd63c5c1f88e58532ed217352b1b82604bc55f39770891cc30d0cd.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/tables/ee569cad26fd63c5c1f88e58532ed217352b1b82604bc55f39770891cc30d0cd.jpg)

![f3a55772c0e25c37ec646eae16cf0581a13afee48184c08e2562e905523e9a45.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/tables/f3a55772c0e25c37ec646eae16cf0581a13afee48184c08e2562e905523e9a45.jpg)

![fa2133df8e0776f5d632f32aacc4b75a9397be3b699fc5db7b0ec10b245d34e2.jpg](../iclr_results/641_Integral Performance Approximation for Continuous-Time Reinforcement Learning Control/tables/fa2133df8e0776f5d632f32aacc4b75a9397be3b699fc5db7b0ec10b245d34e2.jpg)

## FLOPS: Forward Learning with OPtimal Sampling


### Images

![5df835b7f153f0f2720321e19d4358176cdd95bbc74a867f30d063c470ab6790.jpg](../iclr_results/642_FLOPS_ Forward Learning with OPtimal Sampling/images/5df835b7f153f0f2720321e19d4358176cdd95bbc74a867f30d063c470ab6790.jpg)

![7b386ae3a6022d53ea5dfd485fbd88f71aec10c8fb96e46aaba21906c08df810.jpg](../iclr_results/642_FLOPS_ Forward Learning with OPtimal Sampling/images/7b386ae3a6022d53ea5dfd485fbd88f71aec10c8fb96e46aaba21906c08df810.jpg)

![7e559b403402b65f5685f46363191bc4ff7f90ad01c1352026c3b98e46a867bb.jpg](../iclr_results/642_FLOPS_ Forward Learning with OPtimal Sampling/images/7e559b403402b65f5685f46363191bc4ff7f90ad01c1352026c3b98e46a867bb.jpg)

![c894baaa9bb3a8538315fa5f533f43d7beb056719462f4927910e9055ebf32d1.jpg](../iclr_results/642_FLOPS_ Forward Learning with OPtimal Sampling/images/c894baaa9bb3a8538315fa5f533f43d7beb056719462f4927910e9055ebf32d1.jpg)

![e1e71b3faf0ade3ad5099f6584397593ce37186c0efd7eb590ecd0a14edd8b7c.jpg](../iclr_results/642_FLOPS_ Forward Learning with OPtimal Sampling/images/e1e71b3faf0ade3ad5099f6584397593ce37186c0efd7eb590ecd0a14edd8b7c.jpg)

![e8deb14d445833d924d43161f8742c3b7216c593be235581ffb8169acc12e798.jpg](../iclr_results/642_FLOPS_ Forward Learning with OPtimal Sampling/images/e8deb14d445833d924d43161f8742c3b7216c593be235581ffb8169acc12e798.jpg)

### Tables

![7120060d2cd9be723fc75199e65cd1736b86c8533cd3570894ddaeb8b320eded.jpg](../iclr_results/642_FLOPS_ Forward Learning with OPtimal Sampling/tables/7120060d2cd9be723fc75199e65cd1736b86c8533cd3570894ddaeb8b320eded.jpg)

![d1abb8190e10cf0648525bd4b2e00d6bdae0aa8cc11a09637b7104da29376ede.jpg](../iclr_results/642_FLOPS_ Forward Learning with OPtimal Sampling/tables/d1abb8190e10cf0648525bd4b2e00d6bdae0aa8cc11a09637b7104da29376ede.jpg)

![ff0f5185b023d6739665a20112952aaef3e3117464689a4a0aa3b1d516ab2a69.jpg](../iclr_results/642_FLOPS_ Forward Learning with OPtimal Sampling/tables/ff0f5185b023d6739665a20112952aaef3e3117464689a4a0aa3b1d516ab2a69.jpg)

## Once-for-All: Controllable Generative Image Compression with Dynamic Granularity Adaptation


### Images

![02cf5382806ce99adccf914ea1a1d509d6e4106503dafa8c4f4f4d58c35ade68.jpg](../iclr_results/643_Once-for-All_ Controllable Generative Image Compression with Dynamic Granularity Adaptation/images/02cf5382806ce99adccf914ea1a1d509d6e4106503dafa8c4f4f4d58c35ade68.jpg)

![0b8d4299238000119cf01baf30281542e480241f31b7b68631776b5a78e08852.jpg](../iclr_results/643_Once-for-All_ Controllable Generative Image Compression with Dynamic Granularity Adaptation/images/0b8d4299238000119cf01baf30281542e480241f31b7b68631776b5a78e08852.jpg)

![335fd567108849dd821355db61ec43a461ba3e3b0efd8bf7a593071f5fcdc099.jpg](../iclr_results/643_Once-for-All_ Controllable Generative Image Compression with Dynamic Granularity Adaptation/images/335fd567108849dd821355db61ec43a461ba3e3b0efd8bf7a593071f5fcdc099.jpg)

![3769e0ea96d108fa8db1a63c046453844549ae97082153da808adc01ced04a8b.jpg](../iclr_results/643_Once-for-All_ Controllable Generative Image Compression with Dynamic Granularity Adaptation/images/3769e0ea96d108fa8db1a63c046453844549ae97082153da808adc01ced04a8b.jpg)

![392d02cfba55c8c980d1c1248c960fb6ee655ba9e4a3d5512675f4af4a2b55c5.jpg](../iclr_results/643_Once-for-All_ Controllable Generative Image Compression with Dynamic Granularity Adaptation/images/392d02cfba55c8c980d1c1248c960fb6ee655ba9e4a3d5512675f4af4a2b55c5.jpg)

![5651cefae3af7498a0524eb024b6fd3f57f5306f76924e65b60e3935994fec1b.jpg](../iclr_results/643_Once-for-All_ Controllable Generative Image Compression with Dynamic Granularity Adaptation/images/5651cefae3af7498a0524eb024b6fd3f57f5306f76924e65b60e3935994fec1b.jpg)

![6eb56f775f02a39d5e8c5dc6b48de4a6a75e17231c533c56c60fcefab517ebc2.jpg](../iclr_results/643_Once-for-All_ Controllable Generative Image Compression with Dynamic Granularity Adaptation/images/6eb56f775f02a39d5e8c5dc6b48de4a6a75e17231c533c56c60fcefab517ebc2.jpg)

![79445829f223e0af56424d5f6df61826284177ea5275641bc8a4a5b02ec8d38d.jpg](../iclr_results/643_Once-for-All_ Controllable Generative Image Compression with Dynamic Granularity Adaptation/images/79445829f223e0af56424d5f6df61826284177ea5275641bc8a4a5b02ec8d38d.jpg)

![7e0728d1b2d95bf2a6b8ae075196de3eaa6141811c47aa1b7cdce99e7e0719fe.jpg](../iclr_results/643_Once-for-All_ Controllable Generative Image Compression with Dynamic Granularity Adaptation/images/7e0728d1b2d95bf2a6b8ae075196de3eaa6141811c47aa1b7cdce99e7e0719fe.jpg)

![8e393faa334344060c8c545ae415d341358272fead4a16ca2138daf0af565e90.jpg](../iclr_results/643_Once-for-All_ Controllable Generative Image Compression with Dynamic Granularity Adaptation/images/8e393faa334344060c8c545ae415d341358272fead4a16ca2138daf0af565e90.jpg)

![a0d4e6168e2676079e0f0f4c302f522bed40904778e8a256ed37eb6dc7e70c53.jpg](../iclr_results/643_Once-for-All_ Controllable Generative Image Compression with Dynamic Granularity Adaptation/images/a0d4e6168e2676079e0f0f4c302f522bed40904778e8a256ed37eb6dc7e70c53.jpg)

![cadc289e10013fbf948be415064f1645d871f1cde5fd0f71ff0c2ae876c69a93.jpg](../iclr_results/643_Once-for-All_ Controllable Generative Image Compression with Dynamic Granularity Adaptation/images/cadc289e10013fbf948be415064f1645d871f1cde5fd0f71ff0c2ae876c69a93.jpg)

![e3d14b936a927e820264d92fba5963a23a4e6f56f603e6e7e470a8c4a9bd4ed1.jpg](../iclr_results/643_Once-for-All_ Controllable Generative Image Compression with Dynamic Granularity Adaptation/images/e3d14b936a927e820264d92fba5963a23a4e6f56f603e6e7e470a8c4a9bd4ed1.jpg)

![e750926db1ae2ab08163b586a19a739d7861363d29facae3accaefe98aebc69b.jpg](../iclr_results/643_Once-for-All_ Controllable Generative Image Compression with Dynamic Granularity Adaptation/images/e750926db1ae2ab08163b586a19a739d7861363d29facae3accaefe98aebc69b.jpg)

![fb9c1332488b4d5f594cee371a33c5945274bbab259dc2fde4120cd5887da5b5.jpg](../iclr_results/643_Once-for-All_ Controllable Generative Image Compression with Dynamic Granularity Adaptation/images/fb9c1332488b4d5f594cee371a33c5945274bbab259dc2fde4120cd5887da5b5.jpg)

### Tables

![0835070dba2b6b3fcfe4e4c097a807738044dc643c6eb2c2047aa5b190365042.jpg](../iclr_results/643_Once-for-All_ Controllable Generative Image Compression with Dynamic Granularity Adaptation/tables/0835070dba2b6b3fcfe4e4c097a807738044dc643c6eb2c2047aa5b190365042.jpg)

![13fbb46f5151fc690ba3c0f5a0bef5033e21e921e944b4035302c3604cee88c2.jpg](../iclr_results/643_Once-for-All_ Controllable Generative Image Compression with Dynamic Granularity Adaptation/tables/13fbb46f5151fc690ba3c0f5a0bef5033e21e921e944b4035302c3604cee88c2.jpg)

![b9f689972ac78572ec69194696afda70437220fecfa0f9b99e5a3b66dbcf45a1.jpg](../iclr_results/643_Once-for-All_ Controllable Generative Image Compression with Dynamic Granularity Adaptation/tables/b9f689972ac78572ec69194696afda70437220fecfa0f9b99e5a3b66dbcf45a1.jpg)

## Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers


### Images

![20674551190b3ae4fe16848f9f47d72b4b5619dbf944070a5827a05e83605b6c.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/images/20674551190b3ae4fe16848f9f47d72b4b5619dbf944070a5827a05e83605b6c.jpg)

![27f932b8ccf32e236d4f9e2bc5807cb74a7256667a03bef1fa22392575fdc446.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/images/27f932b8ccf32e236d4f9e2bc5807cb74a7256667a03bef1fa22392575fdc446.jpg)

![4c61d1174a02f819eabbc21794e6311b762ecc094927aa59bbf2854a57d01702.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/images/4c61d1174a02f819eabbc21794e6311b762ecc094927aa59bbf2854a57d01702.jpg)

![5e42404743d969510a95457113a032e34d5c682610b9d4af64c6d7bfe406c797.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/images/5e42404743d969510a95457113a032e34d5c682610b9d4af64c6d7bfe406c797.jpg)

![70e0c3aeda6d2fcf3fb81f652433dccef515208f3fb0daae5fd098d753f0cb0b.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/images/70e0c3aeda6d2fcf3fb81f652433dccef515208f3fb0daae5fd098d753f0cb0b.jpg)

### Tables

![01ab869aa9797b75fcde2ef7288f87cbda58f6ebd6fa2d6e93778e7813f96d19.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/tables/01ab869aa9797b75fcde2ef7288f87cbda58f6ebd6fa2d6e93778e7813f96d19.jpg)

![01bf7f352e6db9234a0417d05304b18b04890da1d05fec39906389f92a340bf6.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/tables/01bf7f352e6db9234a0417d05304b18b04890da1d05fec39906389f92a340bf6.jpg)

![0e3791ecfcb6cfcc2d116484112f240f2527d3764fe8eadeb9aee2bd848b7fc3.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/tables/0e3791ecfcb6cfcc2d116484112f240f2527d3764fe8eadeb9aee2bd848b7fc3.jpg)

![10caea334296e17bc31116bac163393316e7cfc6f6c542d054f0cf1f47fba1a3.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/tables/10caea334296e17bc31116bac163393316e7cfc6f6c542d054f0cf1f47fba1a3.jpg)

![320ac700661e642306fc62072a0c98f7b9b99c64ec7778bbc3a1abb159b7a803.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/tables/320ac700661e642306fc62072a0c98f7b9b99c64ec7778bbc3a1abb159b7a803.jpg)

![3aa60baecfb988801445044c96e92de44853940ce13b9ca3a51f6458def44d6a.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/tables/3aa60baecfb988801445044c96e92de44853940ce13b9ca3a51f6458def44d6a.jpg)

![4bd73c13a6cdf4a048776e40acc119e7db07314c0b2fe2122626460ced80372a.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/tables/4bd73c13a6cdf4a048776e40acc119e7db07314c0b2fe2122626460ced80372a.jpg)

![53aad4faf6fcbfe5499511a445c90c09ece91b8ffdc59abd9aca59b5c7a3f0fc.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/tables/53aad4faf6fcbfe5499511a445c90c09ece91b8ffdc59abd9aca59b5c7a3f0fc.jpg)

![70dc419c759bf15188c0094fcdb168a864da6572841773843827051edb57ef29.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/tables/70dc419c759bf15188c0094fcdb168a864da6572841773843827051edb57ef29.jpg)

![8e0f879c68fd648d10875334d76236e75100680e34b5ca39094b1e7e48a408fa.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/tables/8e0f879c68fd648d10875334d76236e75100680e34b5ca39094b1e7e48a408fa.jpg)

![90e30e5f1b9ea732f6a2392b5bce6a9b36a7a75db44a7c7ddd5cd9bf0cb19cfd.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/tables/90e30e5f1b9ea732f6a2392b5bce6a9b36a7a75db44a7c7ddd5cd9bf0cb19cfd.jpg)

![a97e4023ee0b00bc279a8661b5312142298147e6e1979176310849b732facc8b.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/tables/a97e4023ee0b00bc279a8661b5312142298147e6e1979176310849b732facc8b.jpg)

![beaa241ea438d60ae69a487ebeddacefbe7c1a1965b33dbf44d2d2d854ac2af6.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/tables/beaa241ea438d60ae69a487ebeddacefbe7c1a1965b33dbf44d2d2d854ac2af6.jpg)

![c8c6e6460b89300d439bb6a3a526328961fa2c13987df2461422ca9eec752b43.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/tables/c8c6e6460b89300d439bb6a3a526328961fa2c13987df2461422ca9eec752b43.jpg)

![d002536243aac1cbcc9a2f78e98db7a08649ad5d0371386f6575c555c28f6a67.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/tables/d002536243aac1cbcc9a2f78e98db7a08649ad5d0371386f6575c555c28f6a67.jpg)

![d0b256d2b1f3e7782f3b5e30292d6cafacc21b67b89954d0697e55f05ee94098.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/tables/d0b256d2b1f3e7782f3b5e30292d6cafacc21b67b89954d0697e55f05ee94098.jpg)

![db0c0b74dc80170b76aa1e5b2ce40cd6084ae9302cb61e0fb32cddd923efca4c.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/tables/db0c0b74dc80170b76aa1e5b2ce40cd6084ae9302cb61e0fb32cddd923efca4c.jpg)

![fcddffe31716f8f86dfd99cc11ece8c8e236a01784aa27a6b2976514e28bdfc6.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/tables/fcddffe31716f8f86dfd99cc11ece8c8e236a01784aa27a6b2976514e28bdfc6.jpg)

![fe0ca20ae2501929bd833b5cc7579b884ea4d0c14ba464bea49516f1e06ebdc5.jpg](../iclr_results/644_Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers/tables/fe0ca20ae2501929bd833b5cc7579b884ea4d0c14ba464bea49516f1e06ebdc5.jpg)

## LICO: Large Language Models for In-Context Molecular Optimization


### Images

![49e040261d94beae3dcff44470d9b47f07b9613132bf814b3134f3a63cbbc0e5.jpg](../iclr_results/645_LICO_ Large Language Models for In-Context Molecular Optimization/images/49e040261d94beae3dcff44470d9b47f07b9613132bf814b3134f3a63cbbc0e5.jpg)

![802437fddef9cb78d13b6f7b9671102d3326290e0c97f43bd33587439315b0b0.jpg](../iclr_results/645_LICO_ Large Language Models for In-Context Molecular Optimization/images/802437fddef9cb78d13b6f7b9671102d3326290e0c97f43bd33587439315b0b0.jpg)

![9c09f9639e82cf41a728df339aa9e65bbc3f1d85ab6873523a36b71229b6cfbb.jpg](../iclr_results/645_LICO_ Large Language Models for In-Context Molecular Optimization/images/9c09f9639e82cf41a728df339aa9e65bbc3f1d85ab6873523a36b71229b6cfbb.jpg)

![abd1f13aaa5d69f64e401ba18e637c196460d404d8930b352a0f876878e1a2a5.jpg](../iclr_results/645_LICO_ Large Language Models for In-Context Molecular Optimization/images/abd1f13aaa5d69f64e401ba18e637c196460d404d8930b352a0f876878e1a2a5.jpg)

![e09d03157b071f675ffb3e4af599004124adf0c796b6f5a698cf3b97276f8826.jpg](../iclr_results/645_LICO_ Large Language Models for In-Context Molecular Optimization/images/e09d03157b071f675ffb3e4af599004124adf0c796b6f5a698cf3b97276f8826.jpg)

### Tables

![0245b9421d6b5de50ad37200a6481b682d4c8a56a04532bef97a29938e16fd93.jpg](../iclr_results/645_LICO_ Large Language Models for In-Context Molecular Optimization/tables/0245b9421d6b5de50ad37200a6481b682d4c8a56a04532bef97a29938e16fd93.jpg)

![06692ff7218af0f77e2bc6145e2e0760c4857f895ba39d5276feb645e86389b4.jpg](../iclr_results/645_LICO_ Large Language Models for In-Context Molecular Optimization/tables/06692ff7218af0f77e2bc6145e2e0760c4857f895ba39d5276feb645e86389b4.jpg)

![0a85d30b81dea4cd4cbb924d80cbe0f4286b7233f6dd42371a5e988957ef02a3.jpg](../iclr_results/645_LICO_ Large Language Models for In-Context Molecular Optimization/tables/0a85d30b81dea4cd4cbb924d80cbe0f4286b7233f6dd42371a5e988957ef02a3.jpg)

![1c8e586688aef01d809cf15241aa062021627802cd18545e20b653c46a9c1d89.jpg](../iclr_results/645_LICO_ Large Language Models for In-Context Molecular Optimization/tables/1c8e586688aef01d809cf15241aa062021627802cd18545e20b653c46a9c1d89.jpg)

![28b873b0fc83c28ea42dd8dcb59a7bca8cfdb477f5102b21150d227d7b56f753.jpg](../iclr_results/645_LICO_ Large Language Models for In-Context Molecular Optimization/tables/28b873b0fc83c28ea42dd8dcb59a7bca8cfdb477f5102b21150d227d7b56f753.jpg)

![2ce0fac36ef1930735c926dce5039dc2999ab1cfec598922356486a9973263f2.jpg](../iclr_results/645_LICO_ Large Language Models for In-Context Molecular Optimization/tables/2ce0fac36ef1930735c926dce5039dc2999ab1cfec598922356486a9973263f2.jpg)

![716db424fce116b80de3af253be7f80680f54c8a548a06ba2fe3764a66127e13.jpg](../iclr_results/645_LICO_ Large Language Models for In-Context Molecular Optimization/tables/716db424fce116b80de3af253be7f80680f54c8a548a06ba2fe3764a66127e13.jpg)

![b4b3e063e2bbf0b5535272bd7317204af02c44c3c599bc2cb48ef4fca83405d3.jpg](../iclr_results/645_LICO_ Large Language Models for In-Context Molecular Optimization/tables/b4b3e063e2bbf0b5535272bd7317204af02c44c3c599bc2cb48ef4fca83405d3.jpg)

![b50cdd465d6b81d617d1cee8a2e1d8f954e215809712cf6234c0b1d0f91b8155.jpg](../iclr_results/645_LICO_ Large Language Models for In-Context Molecular Optimization/tables/b50cdd465d6b81d617d1cee8a2e1d8f954e215809712cf6234c0b1d0f91b8155.jpg)

![c8cd42ab22f6f57307bf086d2e7b299c24a4f2160f93c941bab1e9bb5f5ed061.jpg](../iclr_results/645_LICO_ Large Language Models for In-Context Molecular Optimization/tables/c8cd42ab22f6f57307bf086d2e7b299c24a4f2160f93c941bab1e9bb5f5ed061.jpg)

![f1bd9e8129a8da281d7a740c91d8b27f53aec943ff85a1ed4d38edb8726c14d1.jpg](../iclr_results/645_LICO_ Large Language Models for In-Context Molecular Optimization/tables/f1bd9e8129a8da281d7a740c91d8b27f53aec943ff85a1ed4d38edb8726c14d1.jpg)

## SiReRAG: Indexing Similar and Related Information for Multihop Reasoning


### Images

![1c692576b3fe17d6970debb0b80ec0892a662ceb80d3a04c0787fb3c9faab512.jpg](../iclr_results/646_SiReRAG_ Indexing Similar and Related Information for Multihop Reasoning/images/1c692576b3fe17d6970debb0b80ec0892a662ceb80d3a04c0787fb3c9faab512.jpg)

![2294395e50a023fe7f8b2218e6cdab46b89cf80329e8126971338b378327b0f5.jpg](../iclr_results/646_SiReRAG_ Indexing Similar and Related Information for Multihop Reasoning/images/2294395e50a023fe7f8b2218e6cdab46b89cf80329e8126971338b378327b0f5.jpg)

![3309346c9d96f655b2d289307a949acbb58b712878d10c823d3d70972dc020ba.jpg](../iclr_results/646_SiReRAG_ Indexing Similar and Related Information for Multihop Reasoning/images/3309346c9d96f655b2d289307a949acbb58b712878d10c823d3d70972dc020ba.jpg)

![a29586ca5866c53605623598a44ee44dc1f200407548e10afb88e7666f74ed63.jpg](../iclr_results/646_SiReRAG_ Indexing Similar and Related Information for Multihop Reasoning/images/a29586ca5866c53605623598a44ee44dc1f200407548e10afb88e7666f74ed63.jpg)

![db30f44b02c56945ce621917302fe2a826340dc9202da86a2e7d93c383a70ca5.jpg](../iclr_results/646_SiReRAG_ Indexing Similar and Related Information for Multihop Reasoning/images/db30f44b02c56945ce621917302fe2a826340dc9202da86a2e7d93c383a70ca5.jpg)

### Tables

![0a6d4ce1cf415d61df53a2f6f2ba9f2c5d337d7c1e5b9232bce5f33d8ca4f026.jpg](../iclr_results/646_SiReRAG_ Indexing Similar and Related Information for Multihop Reasoning/tables/0a6d4ce1cf415d61df53a2f6f2ba9f2c5d337d7c1e5b9232bce5f33d8ca4f026.jpg)

![0d5ef781ed6db7f8d2a4e479528e4ff0645a48a18d622520b3bba0074d6f0975.jpg](../iclr_results/646_SiReRAG_ Indexing Similar and Related Information for Multihop Reasoning/tables/0d5ef781ed6db7f8d2a4e479528e4ff0645a48a18d622520b3bba0074d6f0975.jpg)

![29ce6fc52416b7aeadc2ab67aebfd8ca6fc68436346ad72c40d0add02acf9bd6.jpg](../iclr_results/646_SiReRAG_ Indexing Similar and Related Information for Multihop Reasoning/tables/29ce6fc52416b7aeadc2ab67aebfd8ca6fc68436346ad72c40d0add02acf9bd6.jpg)

![3695621cce7f445af234c031f73b0efcdcd5203c87b5dec3a6a95cf5c536943a.jpg](../iclr_results/646_SiReRAG_ Indexing Similar and Related Information for Multihop Reasoning/tables/3695621cce7f445af234c031f73b0efcdcd5203c87b5dec3a6a95cf5c536943a.jpg)

![3cf4b886f5cdef2a90e088226d97dc735cc57c4b952abb0566ff9cf0eb8b6b0e.jpg](../iclr_results/646_SiReRAG_ Indexing Similar and Related Information for Multihop Reasoning/tables/3cf4b886f5cdef2a90e088226d97dc735cc57c4b952abb0566ff9cf0eb8b6b0e.jpg)

![524f5668117eb5ed64a0a58105c331207ebcdfd0b331b9396c9220bcffd2c080.jpg](../iclr_results/646_SiReRAG_ Indexing Similar and Related Information for Multihop Reasoning/tables/524f5668117eb5ed64a0a58105c331207ebcdfd0b331b9396c9220bcffd2c080.jpg)

![54d15bae396450c372c0f877a8eaed69614da4e529c2f4b93c63656119426c29.jpg](../iclr_results/646_SiReRAG_ Indexing Similar and Related Information for Multihop Reasoning/tables/54d15bae396450c372c0f877a8eaed69614da4e529c2f4b93c63656119426c29.jpg)

![5690f456f0223869b7affd066cd49133135448ed30ac24f035c6cdebaf331722.jpg](../iclr_results/646_SiReRAG_ Indexing Similar and Related Information for Multihop Reasoning/tables/5690f456f0223869b7affd066cd49133135448ed30ac24f035c6cdebaf331722.jpg)

![84b5fe9202ff61c9eba3c560fdc2d68731e747c33bd995d0cf625c850d86692b.jpg](../iclr_results/646_SiReRAG_ Indexing Similar and Related Information for Multihop Reasoning/tables/84b5fe9202ff61c9eba3c560fdc2d68731e747c33bd995d0cf625c850d86692b.jpg)

![8a90669cb00f0bc81099b41ff33dd1253ff622e4549546f4fe1065d811f5fecd.jpg](../iclr_results/646_SiReRAG_ Indexing Similar and Related Information for Multihop Reasoning/tables/8a90669cb00f0bc81099b41ff33dd1253ff622e4549546f4fe1065d811f5fecd.jpg)

![da502573789e53bc4484871377145f17e9d7bdeb64a3fd73b4e701188f140d2a.jpg](../iclr_results/646_SiReRAG_ Indexing Similar and Related Information for Multihop Reasoning/tables/da502573789e53bc4484871377145f17e9d7bdeb64a3fd73b4e701188f140d2a.jpg)

## AutoBencher: Towards Declarative Benchmark Construction


### Images

![33ec41d8ce896123081bec246df9eb90df02e634b4d4b356f032426ab36b4a1e.jpg](../iclr_results/647_AutoBencher_ Towards Declarative Benchmark Construction/images/33ec41d8ce896123081bec246df9eb90df02e634b4d4b356f032426ab36b4a1e.jpg)

![62f200da43ce25cd928daf78dccb3908c9a40605097f2e8fafbfc14e624cff35.jpg](../iclr_results/647_AutoBencher_ Towards Declarative Benchmark Construction/images/62f200da43ce25cd928daf78dccb3908c9a40605097f2e8fafbfc14e624cff35.jpg)

![6ff7bbca12fd751be68c308950fede9fbe10f2272c13aa77dd8092173c79c344.jpg](../iclr_results/647_AutoBencher_ Towards Declarative Benchmark Construction/images/6ff7bbca12fd751be68c308950fede9fbe10f2272c13aa77dd8092173c79c344.jpg)

![c2bf7ff8b60913bc9357aed9b684e35caa0506477a084703e575321757d4f499.jpg](../iclr_results/647_AutoBencher_ Towards Declarative Benchmark Construction/images/c2bf7ff8b60913bc9357aed9b684e35caa0506477a084703e575321757d4f499.jpg)

![d3de1b59f27d04f88eec5c1d67f2d509076498955ad4c9fa2ecb03b2a9b4c7c2.jpg](../iclr_results/647_AutoBencher_ Towards Declarative Benchmark Construction/images/d3de1b59f27d04f88eec5c1d67f2d509076498955ad4c9fa2ecb03b2a9b4c7c2.jpg)

![d537735614a8c30546f51c6119aedc09354943945a6b9d3179b623e5b05ace62.jpg](../iclr_results/647_AutoBencher_ Towards Declarative Benchmark Construction/images/d537735614a8c30546f51c6119aedc09354943945a6b9d3179b623e5b05ace62.jpg)

![e0545778bede8c08eeada10ee0604650088206e894042a090994a5f73dbff4c3.jpg](../iclr_results/647_AutoBencher_ Towards Declarative Benchmark Construction/images/e0545778bede8c08eeada10ee0604650088206e894042a090994a5f73dbff4c3.jpg)

![e5191c14466ae60df81c64d97989fa15efe507ec3f2783bf33e1ab0c06595183.jpg](../iclr_results/647_AutoBencher_ Towards Declarative Benchmark Construction/images/e5191c14466ae60df81c64d97989fa15efe507ec3f2783bf33e1ab0c06595183.jpg)

### Tables

![12424d3e79c98b3a8664e812b557edd511a46e294d9e29e57e7c6d7b522effbb.jpg](../iclr_results/647_AutoBencher_ Towards Declarative Benchmark Construction/tables/12424d3e79c98b3a8664e812b557edd511a46e294d9e29e57e7c6d7b522effbb.jpg)

![16c7d5ecdcc425ce5f0133705eebf97c67802884a320516a47ec352fd51216ad.jpg](../iclr_results/647_AutoBencher_ Towards Declarative Benchmark Construction/tables/16c7d5ecdcc425ce5f0133705eebf97c67802884a320516a47ec352fd51216ad.jpg)

![16f199d7d995c948b2a7f902fb56ebf36a8f7cb9b7b06c11a1af1020d955240c.jpg](../iclr_results/647_AutoBencher_ Towards Declarative Benchmark Construction/tables/16f199d7d995c948b2a7f902fb56ebf36a8f7cb9b7b06c11a1af1020d955240c.jpg)

![1c0511cecc63a7281076ae4ac52d3ef06a00eeff0fc4401cfb9c55b902fd20f4.jpg](../iclr_results/647_AutoBencher_ Towards Declarative Benchmark Construction/tables/1c0511cecc63a7281076ae4ac52d3ef06a00eeff0fc4401cfb9c55b902fd20f4.jpg)

![240600e2dfa7f403a283b914398fa6c24a9a50904591cee8c922e5f95fab8d36.jpg](../iclr_results/647_AutoBencher_ Towards Declarative Benchmark Construction/tables/240600e2dfa7f403a283b914398fa6c24a9a50904591cee8c922e5f95fab8d36.jpg)

![247e1080a5750dad4208d8e8d9b7c11588f14c5e599bfd5a0cc7a547cbabd253.jpg](../iclr_results/647_AutoBencher_ Towards Declarative Benchmark Construction/tables/247e1080a5750dad4208d8e8d9b7c11588f14c5e599bfd5a0cc7a547cbabd253.jpg)

![34775e66f5064c8c030599730f68186408425bd1af15553180088691c9d1fb7c.jpg](../iclr_results/647_AutoBencher_ Towards Declarative Benchmark Construction/tables/34775e66f5064c8c030599730f68186408425bd1af15553180088691c9d1fb7c.jpg)

![43a2961c922fbef808ac45e6a259cb3b40499a07cc77f6279c5cf88201ec1fe7.jpg](../iclr_results/647_AutoBencher_ Towards Declarative Benchmark Construction/tables/43a2961c922fbef808ac45e6a259cb3b40499a07cc77f6279c5cf88201ec1fe7.jpg)

![53223fe147221dadea25a75a45029e992f0e0c8ef780223a8dbb966e95c18cd7.jpg](../iclr_results/647_AutoBencher_ Towards Declarative Benchmark Construction/tables/53223fe147221dadea25a75a45029e992f0e0c8ef780223a8dbb966e95c18cd7.jpg)

![92868848a7d3ce38e41ae4fb88c581b53565afa0ec2da7fdf8ec515542ab30bf.jpg](../iclr_results/647_AutoBencher_ Towards Declarative Benchmark Construction/tables/92868848a7d3ce38e41ae4fb88c581b53565afa0ec2da7fdf8ec515542ab30bf.jpg)

![c079b695b19523fc399bf2341f4f1e8ba30eb84688c98b0949d2cae877479116.jpg](../iclr_results/647_AutoBencher_ Towards Declarative Benchmark Construction/tables/c079b695b19523fc399bf2341f4f1e8ba30eb84688c98b0949d2cae877479116.jpg)

![c7514d727fd55620cfb3a99b5e825798d2f5f3bfacec8239ca853d31f1940a26.jpg](../iclr_results/647_AutoBencher_ Towards Declarative Benchmark Construction/tables/c7514d727fd55620cfb3a99b5e825798d2f5f3bfacec8239ca853d31f1940a26.jpg)

![ca8b7effbcdd28999de81c876f42023f347e59570ef3590c4ed4defe9d1da2e8.jpg](../iclr_results/647_AutoBencher_ Towards Declarative Benchmark Construction/tables/ca8b7effbcdd28999de81c876f42023f347e59570ef3590c4ed4defe9d1da2e8.jpg)

![fff3ebb9d7ac5fc57c84ac5266f99292856ce0455c60f59de1e389799ddff56d.jpg](../iclr_results/647_AutoBencher_ Towards Declarative Benchmark Construction/tables/fff3ebb9d7ac5fc57c84ac5266f99292856ce0455c60f59de1e389799ddff56d.jpg)

## DenoiseVAE: Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-training


### Images

![3b397dda8d81062936245a4fdf08eb5026dbb6745489b5f09f30228b27ea0205.jpg](../iclr_results/648_DenoiseVAE_ Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-trai/images/3b397dda8d81062936245a4fdf08eb5026dbb6745489b5f09f30228b27ea0205.jpg)

![6e45ebeca50417ca79b058db878811561b2d43dbae6084521384c6226f8fed9a.jpg](../iclr_results/648_DenoiseVAE_ Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-trai/images/6e45ebeca50417ca79b058db878811561b2d43dbae6084521384c6226f8fed9a.jpg)

![744e3902423cb2ac77a3a13c8ff73b8c4c7a0ab17dcdebe1ddd2afc95226ac86.jpg](../iclr_results/648_DenoiseVAE_ Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-trai/images/744e3902423cb2ac77a3a13c8ff73b8c4c7a0ab17dcdebe1ddd2afc95226ac86.jpg)

![8fd4b10bac87c7cf53ba8e1abfdb00c97931d6b61d75c682c0156123942c4b97.jpg](../iclr_results/648_DenoiseVAE_ Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-trai/images/8fd4b10bac87c7cf53ba8e1abfdb00c97931d6b61d75c682c0156123942c4b97.jpg)

![9cad2234cdc5f722468132d919825b40dc9ae0549680672d503f51b637d9a54f.jpg](../iclr_results/648_DenoiseVAE_ Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-trai/images/9cad2234cdc5f722468132d919825b40dc9ae0549680672d503f51b637d9a54f.jpg)

![f44ca3cb9e4715dc9423434856efbbfc38f7e61e978c17e028f4f9ee8d9e2a42.jpg](../iclr_results/648_DenoiseVAE_ Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-trai/images/f44ca3cb9e4715dc9423434856efbbfc38f7e61e978c17e028f4f9ee8d9e2a42.jpg)

![f916e008ba76416e028ffabcca7286b603b184af2fc12295068e643177d5a4d3.jpg](../iclr_results/648_DenoiseVAE_ Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-trai/images/f916e008ba76416e028ffabcca7286b603b184af2fc12295068e643177d5a4d3.jpg)

### Tables

![10c61b18a68b941766c63aab7930c966acc5c255d1c93352255b82b1db8e90d6.jpg](../iclr_results/648_DenoiseVAE_ Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-trai/tables/10c61b18a68b941766c63aab7930c966acc5c255d1c93352255b82b1db8e90d6.jpg)

![20fdb5c366d7d159ccc63ac61b70cec158c59904259646df3804d88c7864b09d.jpg](../iclr_results/648_DenoiseVAE_ Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-trai/tables/20fdb5c366d7d159ccc63ac61b70cec158c59904259646df3804d88c7864b09d.jpg)

![3cae45a5cff21690525384b8ae632a96d626a40d80d91023a0cd0045230c25c2.jpg](../iclr_results/648_DenoiseVAE_ Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-trai/tables/3cae45a5cff21690525384b8ae632a96d626a40d80d91023a0cd0045230c25c2.jpg)

![4453c9e42f88b833919bfe646848ff69f0d31bfb3e7c2c490956e19158ab0485.jpg](../iclr_results/648_DenoiseVAE_ Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-trai/tables/4453c9e42f88b833919bfe646848ff69f0d31bfb3e7c2c490956e19158ab0485.jpg)

![78c53c564e9a64fd5cc1f402be834735a2e4a7b643bf7b3ea6fe92b69fae003d.jpg](../iclr_results/648_DenoiseVAE_ Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-trai/tables/78c53c564e9a64fd5cc1f402be834735a2e4a7b643bf7b3ea6fe92b69fae003d.jpg)

![7b74ae88d21192e0409dc3536c141f3aeb9bd90a9f024029636649ac7a018554.jpg](../iclr_results/648_DenoiseVAE_ Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-trai/tables/7b74ae88d21192e0409dc3536c141f3aeb9bd90a9f024029636649ac7a018554.jpg)

![b1de9d78b587e5e459329dee78c90cbc12599271253a31942ea7bab8b7c5dba4.jpg](../iclr_results/648_DenoiseVAE_ Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-trai/tables/b1de9d78b587e5e459329dee78c90cbc12599271253a31942ea7bab8b7c5dba4.jpg)

![be4287423fea5417517bb7fe83ab9646a763c295802292b69e53509a1dc0e974.jpg](../iclr_results/648_DenoiseVAE_ Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-trai/tables/be4287423fea5417517bb7fe83ab9646a763c295802292b69e53509a1dc0e974.jpg)

![cd74974c1f1a75bfb2cf25d8803b9c06b04ff97790f2531a0a2cf9f106794a39.jpg](../iclr_results/648_DenoiseVAE_ Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-trai/tables/cd74974c1f1a75bfb2cf25d8803b9c06b04ff97790f2531a0a2cf9f106794a39.jpg)

![cf46baadb1f4b765d69f3d55d6819fb99e8635d271ac356c90f298ada7d641f2.jpg](../iclr_results/648_DenoiseVAE_ Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-trai/tables/cf46baadb1f4b765d69f3d55d6819fb99e8635d271ac356c90f298ada7d641f2.jpg)

![d9d6e9d54fce0a6422b50a0a47ee643cf499e752c6840d71115510da9397f432.jpg](../iclr_results/648_DenoiseVAE_ Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-trai/tables/d9d6e9d54fce0a6422b50a0a47ee643cf499e752c6840d71115510da9397f432.jpg)

![e012e8e3161f43bd1675e8256f36a3fb1841910382a8580474fe30c6bfbe4a27.jpg](../iclr_results/648_DenoiseVAE_ Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-trai/tables/e012e8e3161f43bd1675e8256f36a3fb1841910382a8580474fe30c6bfbe4a27.jpg)

![fa2700f0038551119467a528221794ddbc749c99065197d869ff9cea3175a315.jpg](../iclr_results/648_DenoiseVAE_ Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-trai/tables/fa2700f0038551119467a528221794ddbc749c99065197d869ff9cea3175a315.jpg)

## ELFS: Label-Free Coreset Selection with Proxy Training Dynamics


### Images

![179020e9efe7cbbf44421697d73144e42d3a0d882526a7c7e82aa8ca6ecfc246.jpg](../iclr_results/649_ELFS_ Label-Free Coreset Selection with Proxy Training Dynamics/images/179020e9efe7cbbf44421697d73144e42d3a0d882526a7c7e82aa8ca6ecfc246.jpg)

![32abc5248f36d1f686cb062171581d03da4b667695ee22ab159d4f9195add75b.jpg](../iclr_results/649_ELFS_ Label-Free Coreset Selection with Proxy Training Dynamics/images/32abc5248f36d1f686cb062171581d03da4b667695ee22ab159d4f9195add75b.jpg)

![3d788327ecee4633122ce0a055e7cf56659752553acf06cd21d06439d1871740.jpg](../iclr_results/649_ELFS_ Label-Free Coreset Selection with Proxy Training Dynamics/images/3d788327ecee4633122ce0a055e7cf56659752553acf06cd21d06439d1871740.jpg)

![45ee283911a96d1d102bc4e31ec600c8628e446ab152561152c7c194bf85d690.jpg](../iclr_results/649_ELFS_ Label-Free Coreset Selection with Proxy Training Dynamics/images/45ee283911a96d1d102bc4e31ec600c8628e446ab152561152c7c194bf85d690.jpg)

![705e19471073d58c722eed8afcb835beb4009a08080f39151c148444c1509df2.jpg](../iclr_results/649_ELFS_ Label-Free Coreset Selection with Proxy Training Dynamics/images/705e19471073d58c722eed8afcb835beb4009a08080f39151c148444c1509df2.jpg)

![89b80a9bfd3b616ae07d0f10399c250c7f75392d148717ca9952a013e2b3882d.jpg](../iclr_results/649_ELFS_ Label-Free Coreset Selection with Proxy Training Dynamics/images/89b80a9bfd3b616ae07d0f10399c250c7f75392d148717ca9952a013e2b3882d.jpg)

![996d2273d721db191e166e912381f5329358b95d12bc8be621738f8f0b2997ca.jpg](../iclr_results/649_ELFS_ Label-Free Coreset Selection with Proxy Training Dynamics/images/996d2273d721db191e166e912381f5329358b95d12bc8be621738f8f0b2997ca.jpg)

![c5d36b76e7dab41e5db65b77599500cf171b24684ce329ed7664c9b2a81ea22f.jpg](../iclr_results/649_ELFS_ Label-Free Coreset Selection with Proxy Training Dynamics/images/c5d36b76e7dab41e5db65b77599500cf171b24684ce329ed7664c9b2a81ea22f.jpg)

![c61b9813f946cf033e01f8e63e9176a4095353edd79b7bb2ef5dbbb2234d88f7.jpg](../iclr_results/649_ELFS_ Label-Free Coreset Selection with Proxy Training Dynamics/images/c61b9813f946cf033e01f8e63e9176a4095353edd79b7bb2ef5dbbb2234d88f7.jpg)

![e31afa2080ebfd74cb4f7284036540df46cc692c22157e124d22192608a3333c.jpg](../iclr_results/649_ELFS_ Label-Free Coreset Selection with Proxy Training Dynamics/images/e31afa2080ebfd74cb4f7284036540df46cc692c22157e124d22192608a3333c.jpg)

### Tables

![204a5d76bd4536cba41eaf12b429ed20584f774af8c4f17489ad2473e3944829.jpg](../iclr_results/649_ELFS_ Label-Free Coreset Selection with Proxy Training Dynamics/tables/204a5d76bd4536cba41eaf12b429ed20584f774af8c4f17489ad2473e3944829.jpg)

![26fe6908c28e35535e6bb723af0e8f2336d4e2f76b301288f087c2fe551d301c.jpg](../iclr_results/649_ELFS_ Label-Free Coreset Selection with Proxy Training Dynamics/tables/26fe6908c28e35535e6bb723af0e8f2336d4e2f76b301288f087c2fe551d301c.jpg)

![7e87badd81b1335b71690319f490613defeca7132879c915c585bdd3911f7810.jpg](../iclr_results/649_ELFS_ Label-Free Coreset Selection with Proxy Training Dynamics/tables/7e87badd81b1335b71690319f490613defeca7132879c915c585bdd3911f7810.jpg)

![89c9ceda2a30db52d89691cbdd62e5e677faa04427eedd31db524a9d902681ac.jpg](../iclr_results/649_ELFS_ Label-Free Coreset Selection with Proxy Training Dynamics/tables/89c9ceda2a30db52d89691cbdd62e5e677faa04427eedd31db524a9d902681ac.jpg)

![8a0b50d7878aafb64385d87021be2c19e19ed412e490c172ad4d173f39ac464c.jpg](../iclr_results/649_ELFS_ Label-Free Coreset Selection with Proxy Training Dynamics/tables/8a0b50d7878aafb64385d87021be2c19e19ed412e490c172ad4d173f39ac464c.jpg)

![9baea112c112bb99f8af0f9155362d5e2f824d2389280115a242221df0ad9b3d.jpg](../iclr_results/649_ELFS_ Label-Free Coreset Selection with Proxy Training Dynamics/tables/9baea112c112bb99f8af0f9155362d5e2f824d2389280115a242221df0ad9b3d.jpg)

![a1067b7497db82c0a5982be3b6d2fc11a40882048dfc741dac386ee02ab9a6c3.jpg](../iclr_results/649_ELFS_ Label-Free Coreset Selection with Proxy Training Dynamics/tables/a1067b7497db82c0a5982be3b6d2fc11a40882048dfc741dac386ee02ab9a6c3.jpg)

![afd68bd28bef52b84a8438bf70c2412d8a52a99d838223defd00777444f87bb4.jpg](../iclr_results/649_ELFS_ Label-Free Coreset Selection with Proxy Training Dynamics/tables/afd68bd28bef52b84a8438bf70c2412d8a52a99d838223defd00777444f87bb4.jpg)

![d1e444b8b1269d43d99781944a70e36e1558ced7d1a1b28999e1d578e869a547.jpg](../iclr_results/649_ELFS_ Label-Free Coreset Selection with Proxy Training Dynamics/tables/d1e444b8b1269d43d99781944a70e36e1558ced7d1a1b28999e1d578e869a547.jpg)

![da232c5261ad3cdef0c0559f7c72412bc16be660ceaf9414227ff52d8b430dd4.jpg](../iclr_results/649_ELFS_ Label-Free Coreset Selection with Proxy Training Dynamics/tables/da232c5261ad3cdef0c0559f7c72412bc16be660ceaf9414227ff52d8b430dd4.jpg)

![da9789176ac81fee9aaca5dde004369a43225cf18fa8901125a42b161eccd78d.jpg](../iclr_results/649_ELFS_ Label-Free Coreset Selection with Proxy Training Dynamics/tables/da9789176ac81fee9aaca5dde004369a43225cf18fa8901125a42b161eccd78d.jpg)

![e5d5e40107bdd911d924a98aab9a50513e81a016421bafc6e4887267213bb453.jpg](../iclr_results/649_ELFS_ Label-Free Coreset Selection with Proxy Training Dynamics/tables/e5d5e40107bdd911d924a98aab9a50513e81a016421bafc6e4887267213bb453.jpg)

## Generative Inbetweening: Adapting Image-to-Video Models for Keyframe Interpolation


### Images

![0f25ed334079647599634a7cf7dcd8c0f807bda471f3e0afa73271c21bc790e1.jpg](../iclr_results/650_Generative Inbetweening_ Adapting Image-to-Video Models for Keyframe Interpolation/images/0f25ed334079647599634a7cf7dcd8c0f807bda471f3e0afa73271c21bc790e1.jpg)

![13dae18bb9bc535361984b96ed62cbd53a9d3e5bd60cb678a5df8b242c0a99e1.jpg](../iclr_results/650_Generative Inbetweening_ Adapting Image-to-Video Models for Keyframe Interpolation/images/13dae18bb9bc535361984b96ed62cbd53a9d3e5bd60cb678a5df8b242c0a99e1.jpg)

![13e0e98f171389357871fe5a419201f0dbbf82bd0abe122719067ce72eb5072b.jpg](../iclr_results/650_Generative Inbetweening_ Adapting Image-to-Video Models for Keyframe Interpolation/images/13e0e98f171389357871fe5a419201f0dbbf82bd0abe122719067ce72eb5072b.jpg)

![24e3b52f624c0b955a1b349f335dbd74ebe20584775d314d67110213c07ee39c.jpg](../iclr_results/650_Generative Inbetweening_ Adapting Image-to-Video Models for Keyframe Interpolation/images/24e3b52f624c0b955a1b349f335dbd74ebe20584775d314d67110213c07ee39c.jpg)

![2caaae65e909971cf6392197d9f79d9dbe74795d533f846c496f5090ce1be822.jpg](../iclr_results/650_Generative Inbetweening_ Adapting Image-to-Video Models for Keyframe Interpolation/images/2caaae65e909971cf6392197d9f79d9dbe74795d533f846c496f5090ce1be822.jpg)

![58a419c580011232fca32815d14837a13f802ace8b5d1f753ae0c1d4953c7200.jpg](../iclr_results/650_Generative Inbetweening_ Adapting Image-to-Video Models for Keyframe Interpolation/images/58a419c580011232fca32815d14837a13f802ace8b5d1f753ae0c1d4953c7200.jpg)

![66048bc8b3a4f7791111badf669aa5843e71a35425b4dfca3e4e611e723ca73e.jpg](../iclr_results/650_Generative Inbetweening_ Adapting Image-to-Video Models for Keyframe Interpolation/images/66048bc8b3a4f7791111badf669aa5843e71a35425b4dfca3e4e611e723ca73e.jpg)

![b44de75b5c68a28e05bf2f3fe6e5c1449819e4f30f64049a4c18101f1bc28c7e.jpg](../iclr_results/650_Generative Inbetweening_ Adapting Image-to-Video Models for Keyframe Interpolation/images/b44de75b5c68a28e05bf2f3fe6e5c1449819e4f30f64049a4c18101f1bc28c7e.jpg)

![e9b50e95aa3baf011359025d26ce218db8bf3004c8f6eaaeae09e525745440b5.jpg](../iclr_results/650_Generative Inbetweening_ Adapting Image-to-Video Models for Keyframe Interpolation/images/e9b50e95aa3baf011359025d26ce218db8bf3004c8f6eaaeae09e525745440b5.jpg)

### Tables

![0b63fc4c57b8e641b8583f556c86af0a3190698e6d872430d6c91f66b64dc94a.jpg](../iclr_results/650_Generative Inbetweening_ Adapting Image-to-Video Models for Keyframe Interpolation/tables/0b63fc4c57b8e641b8583f556c86af0a3190698e6d872430d6c91f66b64dc94a.jpg)

![fab68c78f2428e8029911837a6c293fefe4291ce4a3aaa35e4828956d324bd29.jpg](../iclr_results/650_Generative Inbetweening_ Adapting Image-to-Video Models for Keyframe Interpolation/tables/fab68c78f2428e8029911837a6c293fefe4291ce4a3aaa35e4828956d324bd29.jpg)

![ffa81832ea4e11f153437ced43b3ab0c90e42204e62203b91cae3ec643af2775.jpg](../iclr_results/650_Generative Inbetweening_ Adapting Image-to-Video Models for Keyframe Interpolation/tables/ffa81832ea4e11f153437ced43b3ab0c90e42204e62203b91cae3ec643af2775.jpg)

## UniWav: Towards Unified Pre-training for Speech Representation Learning and Generation


### Images

![3d681a4627a1d6745c721ebb3d8bd7865d92198ff8a6add60da44123264d2363.jpg](../iclr_results/651_UniWav_ Towards Unified Pre-training for Speech Representation Learning and Generation/images/3d681a4627a1d6745c721ebb3d8bd7865d92198ff8a6add60da44123264d2363.jpg)

![4c85fa51013883c573b347ae56411fa6a42da88ab596e480f638bebc9f8288c5.jpg](../iclr_results/651_UniWav_ Towards Unified Pre-training for Speech Representation Learning and Generation/images/4c85fa51013883c573b347ae56411fa6a42da88ab596e480f638bebc9f8288c5.jpg)

![ac8edab62e6f7f81e8b40e1b32032ffa42f46d957a6da12dc458e8a31e944903.jpg](../iclr_results/651_UniWav_ Towards Unified Pre-training for Speech Representation Learning and Generation/images/ac8edab62e6f7f81e8b40e1b32032ffa42f46d957a6da12dc458e8a31e944903.jpg)

### Tables

![1507d0aa5abea1a6ff1762adb7621e4c588eba52519b2173a7acbad7902a7549.jpg](../iclr_results/651_UniWav_ Towards Unified Pre-training for Speech Representation Learning and Generation/tables/1507d0aa5abea1a6ff1762adb7621e4c588eba52519b2173a7acbad7902a7549.jpg)

![15eac2edd1c854fb56f9f197a1d4aca2be0bead99c118320733e33190de7acf8.jpg](../iclr_results/651_UniWav_ Towards Unified Pre-training for Speech Representation Learning and Generation/tables/15eac2edd1c854fb56f9f197a1d4aca2be0bead99c118320733e33190de7acf8.jpg)

![24eb6f4ac62cbccc965b8d86013aeba0e0b7efc5da3abdb38ec93d60adeecfe2.jpg](../iclr_results/651_UniWav_ Towards Unified Pre-training for Speech Representation Learning and Generation/tables/24eb6f4ac62cbccc965b8d86013aeba0e0b7efc5da3abdb38ec93d60adeecfe2.jpg)

![6a37b7ba24b8ffa047ffcccd563a24538930284c85869fa33548159d1640ff3a.jpg](../iclr_results/651_UniWav_ Towards Unified Pre-training for Speech Representation Learning and Generation/tables/6a37b7ba24b8ffa047ffcccd563a24538930284c85869fa33548159d1640ff3a.jpg)

![9e815760898016cd91046aaa1d19659c60535bb0e22e6c583fe2db9cf8af46cd.jpg](../iclr_results/651_UniWav_ Towards Unified Pre-training for Speech Representation Learning and Generation/tables/9e815760898016cd91046aaa1d19659c60535bb0e22e6c583fe2db9cf8af46cd.jpg)

![c21374cabce908019f3f598078e61ead7e32130736d9c5e82c973ceec163f823.jpg](../iclr_results/651_UniWav_ Towards Unified Pre-training for Speech Representation Learning and Generation/tables/c21374cabce908019f3f598078e61ead7e32130736d9c5e82c973ceec163f823.jpg)

![d72c300a08b81aa2ad7c0ab1c1afb9e4b4f1018c58c19116eccf3350839e2559.jpg](../iclr_results/651_UniWav_ Towards Unified Pre-training for Speech Representation Learning and Generation/tables/d72c300a08b81aa2ad7c0ab1c1afb9e4b4f1018c58c19116eccf3350839e2559.jpg)

## Forewarned is Forearmed:  Harnessing LLMs for Data Synthesis via Failure-induced Exploration


### Images

![bd154ee7d02edb8f9e4d7f4836f0bbf18d1bdd204e70291d3a959572edd41601.jpg](../iclr_results/652_Forewarned is Forearmed_  Harnessing LLMs for Data Synthesis via Failure-induced Exploration/images/bd154ee7d02edb8f9e4d7f4836f0bbf18d1bdd204e70291d3a959572edd41601.jpg)

![d46a79b95b6f0520e1f0625c0e103fbc657bf18a35c71ddc25b3a7e3ec7fd284.jpg](../iclr_results/652_Forewarned is Forearmed_  Harnessing LLMs for Data Synthesis via Failure-induced Exploration/images/d46a79b95b6f0520e1f0625c0e103fbc657bf18a35c71ddc25b3a7e3ec7fd284.jpg)

### Tables

![2a98ed8c2b81c1191e61e2708ae6b6096275128d89028d637d32d92facb312bd.jpg](../iclr_results/652_Forewarned is Forearmed_  Harnessing LLMs for Data Synthesis via Failure-induced Exploration/tables/2a98ed8c2b81c1191e61e2708ae6b6096275128d89028d637d32d92facb312bd.jpg)

![3ad1b0bf1c16ce60c5391738ffaef3da629f89591cda7736f433dd9589cb48ef.jpg](../iclr_results/652_Forewarned is Forearmed_  Harnessing LLMs for Data Synthesis via Failure-induced Exploration/tables/3ad1b0bf1c16ce60c5391738ffaef3da629f89591cda7736f433dd9589cb48ef.jpg)

![7eb43fa2bedfc05ac1de9bc4d579c267e79445304403cba7866c3bbe4a536daa.jpg](../iclr_results/652_Forewarned is Forearmed_  Harnessing LLMs for Data Synthesis via Failure-induced Exploration/tables/7eb43fa2bedfc05ac1de9bc4d579c267e79445304403cba7866c3bbe4a536daa.jpg)

![89f35a3a0f729a39d27a8efccae4bc65230f2b02717e7631b0e5a9acd76aa24d.jpg](../iclr_results/652_Forewarned is Forearmed_  Harnessing LLMs for Data Synthesis via Failure-induced Exploration/tables/89f35a3a0f729a39d27a8efccae4bc65230f2b02717e7631b0e5a9acd76aa24d.jpg)

![abc6ff73425effb0d60f77bfc8f14ee6afa62ffaa064d19d0d3a7f71d53f70f8.jpg](../iclr_results/652_Forewarned is Forearmed_  Harnessing LLMs for Data Synthesis via Failure-induced Exploration/tables/abc6ff73425effb0d60f77bfc8f14ee6afa62ffaa064d19d0d3a7f71d53f70f8.jpg)

![d67c80fa5d973137f771c512847ae3db40bbf6235b725353908b58405a46c564.jpg](../iclr_results/652_Forewarned is Forearmed_  Harnessing LLMs for Data Synthesis via Failure-induced Exploration/tables/d67c80fa5d973137f771c512847ae3db40bbf6235b725353908b58405a46c564.jpg)

![d8abdec9114e4270e4a0f8271d6c0751697f6b430b0651bbc570dbf8778681ca.jpg](../iclr_results/652_Forewarned is Forearmed_  Harnessing LLMs for Data Synthesis via Failure-induced Exploration/tables/d8abdec9114e4270e4a0f8271d6c0751697f6b430b0651bbc570dbf8778681ca.jpg)

![df9dab3e173e439723fc1a3e6704d791faf35dba6e858f69c4e7f29df4c542e1.jpg](../iclr_results/652_Forewarned is Forearmed_  Harnessing LLMs for Data Synthesis via Failure-induced Exploration/tables/df9dab3e173e439723fc1a3e6704d791faf35dba6e858f69c4e7f29df4c542e1.jpg)

![e1600e03de2e97606ed1fb96018b0caea12dfa80b22d9e31a2742e5a2ad2072b.jpg](../iclr_results/652_Forewarned is Forearmed_  Harnessing LLMs for Data Synthesis via Failure-induced Exploration/tables/e1600e03de2e97606ed1fb96018b0caea12dfa80b22d9e31a2742e5a2ad2072b.jpg)

![e1a40ee4066e0476a76fe8bada1e68e918a949b115b8a48b90f9cdde166928de.jpg](../iclr_results/652_Forewarned is Forearmed_  Harnessing LLMs for Data Synthesis via Failure-induced Exploration/tables/e1a40ee4066e0476a76fe8bada1e68e918a949b115b8a48b90f9cdde166928de.jpg)

## Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models


### Images

![59f3964169d50502292df685ae8f54fcf48808dadcaa0ab5642eacbc0945586a.jpg](../iclr_results/653_Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models/images/59f3964169d50502292df685ae8f54fcf48808dadcaa0ab5642eacbc0945586a.jpg)

![92d7f51c7d31bbabb38e2ca7500668bb2bc13f18d138acf814227fdc10a52956.jpg](../iclr_results/653_Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models/images/92d7f51c7d31bbabb38e2ca7500668bb2bc13f18d138acf814227fdc10a52956.jpg)

![9598829ef464125808bdf3ea4ab1a0d8547fa69fcdc33ca62b54280e590661f0.jpg](../iclr_results/653_Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models/images/9598829ef464125808bdf3ea4ab1a0d8547fa69fcdc33ca62b54280e590661f0.jpg)

![b481427385ba8e6eca7e079b93f1efb60fc822355f01f6a0e5f89deffc868ad1.jpg](../iclr_results/653_Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models/images/b481427385ba8e6eca7e079b93f1efb60fc822355f01f6a0e5f89deffc868ad1.jpg)

![bc058ed5c86a2b2f3e401619696b530852f30a8cbcae47f0e7e49d43e1b8969b.jpg](../iclr_results/653_Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models/images/bc058ed5c86a2b2f3e401619696b530852f30a8cbcae47f0e7e49d43e1b8969b.jpg)

![d5b0c84ac1ecd540d52aee0b234e6b98f338b3e25f81c8dc070f6dd533a4e709.jpg](../iclr_results/653_Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models/images/d5b0c84ac1ecd540d52aee0b234e6b98f338b3e25f81c8dc070f6dd533a4e709.jpg)

![e4efb5b5d2a77d3a5d096a8cf44b7f4e862d3d7e7991a04c1fe9f3af59aeb561.jpg](../iclr_results/653_Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models/images/e4efb5b5d2a77d3a5d096a8cf44b7f4e862d3d7e7991a04c1fe9f3af59aeb561.jpg)

![ff1d8ce53de0dd7cf2bec8377e2f10b43c3940c32e3bc635502b179e5afbe863.jpg](../iclr_results/653_Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models/images/ff1d8ce53de0dd7cf2bec8377e2f10b43c3940c32e3bc635502b179e5afbe863.jpg)

### Tables

![1f387fa752ef4cd3560ae15a727d795465981bb56752bc8ce53b18d257e7f67f.jpg](../iclr_results/653_Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models/tables/1f387fa752ef4cd3560ae15a727d795465981bb56752bc8ce53b18d257e7f67f.jpg)

![269855bce2fe18afdac6497ae9faceb2e452cf29616054fbeb51da9325b2a369.jpg](../iclr_results/653_Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models/tables/269855bce2fe18afdac6497ae9faceb2e452cf29616054fbeb51da9325b2a369.jpg)

![4f81e60c1d9e065c4f7dafecfa21fb721edddc9799a2fed942e8feaebd96e441.jpg](../iclr_results/653_Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models/tables/4f81e60c1d9e065c4f7dafecfa21fb721edddc9799a2fed942e8feaebd96e441.jpg)

![52e845e3916f87d26fab35a4a0e12a43f1ebf9b03c031ac0d0d37baf6fd8bb5b.jpg](../iclr_results/653_Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models/tables/52e845e3916f87d26fab35a4a0e12a43f1ebf9b03c031ac0d0d37baf6fd8bb5b.jpg)

![57bbcf2f8495f002bdf924b11d092876f4d6a0e200a2e57d78c07f7506d6d83c.jpg](../iclr_results/653_Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models/tables/57bbcf2f8495f002bdf924b11d092876f4d6a0e200a2e57d78c07f7506d6d83c.jpg)

![8ae3246c9462d2f3a68af0a4b117bc6fca8408a769b3b493e840283d97e2311e.jpg](../iclr_results/653_Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models/tables/8ae3246c9462d2f3a68af0a4b117bc6fca8408a769b3b493e840283d97e2311e.jpg)

![910ca9b30a9ecd120fcb1f85c24e0317029a4259d1c41cd9b0848dbbd9880357.jpg](../iclr_results/653_Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models/tables/910ca9b30a9ecd120fcb1f85c24e0317029a4259d1c41cd9b0848dbbd9880357.jpg)

![a6c15784220ac24b411095933ff1fbed9658a243c666989847be9f2a8dce028c.jpg](../iclr_results/653_Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models/tables/a6c15784220ac24b411095933ff1fbed9658a243c666989847be9f2a8dce028c.jpg)

![afd407047027b20c18d42f0e8ae6a0f9621714d520e01d44a3967604529ae932.jpg](../iclr_results/653_Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models/tables/afd407047027b20c18d42f0e8ae6a0f9621714d520e01d44a3967604529ae932.jpg)

![c84db93afca905f52154e995e31cbfb71f8668eb445b5ee9ecd1671fbb22e293.jpg](../iclr_results/653_Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models/tables/c84db93afca905f52154e995e31cbfb71f8668eb445b5ee9ecd1671fbb22e293.jpg)

![d10524472ff2c7cc9e8aa2df03b064d09ae8de971def6297c1183b8b4e194de3.jpg](../iclr_results/653_Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models/tables/d10524472ff2c7cc9e8aa2df03b064d09ae8de971def6297c1183b8b4e194de3.jpg)

![d9b54c4df65713e5270f35f3380a9e21b6f32dee8cc128f396cba4898edf1563.jpg](../iclr_results/653_Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models/tables/d9b54c4df65713e5270f35f3380a9e21b6f32dee8cc128f396cba4898edf1563.jpg)

![e12de156bdde6008ba70378e0c4074e56fe23710b28860eb66ef3104c55373a2.jpg](../iclr_results/653_Effective and Efficient Time-Varying Counterfactual Prediction with State-Space Models/tables/e12de156bdde6008ba70378e0c4074e56fe23710b28860eb66ef3104c55373a2.jpg)

## On the expressiveness and spectral bias of KANs


### Images

![2326ed0252402d49ebf94acb0084ee4bbfafeb5c5c138ae34835e0e006178291.jpg](../iclr_results/654_On the expressiveness and spectral bias of KANs/images/2326ed0252402d49ebf94acb0084ee4bbfafeb5c5c138ae34835e0e006178291.jpg)

![3224c1e30d5603064ac8459ede99c4474d7bc70c0ded9781ff18439fb3a584c7.jpg](../iclr_results/654_On the expressiveness and spectral bias of KANs/images/3224c1e30d5603064ac8459ede99c4474d7bc70c0ded9781ff18439fb3a584c7.jpg)

![3a4c67e298070a8f7696b71a1be584ac19c3746671e8173d285e093a471ba3ec.jpg](../iclr_results/654_On the expressiveness and spectral bias of KANs/images/3a4c67e298070a8f7696b71a1be584ac19c3746671e8173d285e093a471ba3ec.jpg)

![7456d920883f84c41f10cb5720921f5d6ffef80a113d476790356b7a195e13a6.jpg](../iclr_results/654_On the expressiveness and spectral bias of KANs/images/7456d920883f84c41f10cb5720921f5d6ffef80a113d476790356b7a195e13a6.jpg)

![766cab2a9455b07d313402156feea16d2d66b6a670e835bb9ad81b0fa9927835.jpg](../iclr_results/654_On the expressiveness and spectral bias of KANs/images/766cab2a9455b07d313402156feea16d2d66b6a670e835bb9ad81b0fa9927835.jpg)

![967d94ea527b1b5d729ce4f7c4d5c3d818bbab66c233cf714780c8259922b804.jpg](../iclr_results/654_On the expressiveness and spectral bias of KANs/images/967d94ea527b1b5d729ce4f7c4d5c3d818bbab66c233cf714780c8259922b804.jpg)

![c91fdd40df39587727def68faa25d1a091552211453565631754a8c198463b61.jpg](../iclr_results/654_On the expressiveness and spectral bias of KANs/images/c91fdd40df39587727def68faa25d1a091552211453565631754a8c198463b61.jpg)

## Federated Class-Incremental Learning: A Hybrid Approach Using Latent Exemplars and Data-Free Techniques to Address Local and Global Forgetting


### Images

![17d159b296559ee1340733d9c3051202cf7bcf1a958dca16bc9ecc9c0b0518b0.jpg](../iclr_results/655_Federated Class-Incremental Learning_ A Hybrid Approach Using Latent Exemplars and Data-Free Techniq/images/17d159b296559ee1340733d9c3051202cf7bcf1a958dca16bc9ecc9c0b0518b0.jpg)

![202e586f4c5cf89016a7a82fc95ab5ee1d65d8fc1f1a70d3b16dd5142d58fb82.jpg](../iclr_results/655_Federated Class-Incremental Learning_ A Hybrid Approach Using Latent Exemplars and Data-Free Techniq/images/202e586f4c5cf89016a7a82fc95ab5ee1d65d8fc1f1a70d3b16dd5142d58fb82.jpg)

![b6847caa1efaaf27d27287b284050340d1a038ad8d57dd2b5d181d03bf1d7681.jpg](../iclr_results/655_Federated Class-Incremental Learning_ A Hybrid Approach Using Latent Exemplars and Data-Free Techniq/images/b6847caa1efaaf27d27287b284050340d1a038ad8d57dd2b5d181d03bf1d7681.jpg)

### Tables

![b6cf65c60210d1f91b9a63d1fe9d84dff2246d9bb7a61adadbd38ddd260f3ad2.jpg](../iclr_results/655_Federated Class-Incremental Learning_ A Hybrid Approach Using Latent Exemplars and Data-Free Techniq/tables/b6cf65c60210d1f91b9a63d1fe9d84dff2246d9bb7a61adadbd38ddd260f3ad2.jpg)

![d36205089e693ac31998440b873415d967e4781b9490fca1b1bb35b1dde3b486.jpg](../iclr_results/655_Federated Class-Incremental Learning_ A Hybrid Approach Using Latent Exemplars and Data-Free Techniq/tables/d36205089e693ac31998440b873415d967e4781b9490fca1b1bb35b1dde3b486.jpg)

![df50f4b5ca8adb06664bc800293b2880e0680d1ce870f5da3ee9384ea70b7164.jpg](../iclr_results/655_Federated Class-Incremental Learning_ A Hybrid Approach Using Latent Exemplars and Data-Free Techniq/tables/df50f4b5ca8adb06664bc800293b2880e0680d1ce870f5da3ee9384ea70b7164.jpg)

## AIMS.au: A Dataset for the Analysis of Modern Slavery Countermeasures in Corporate Statements


### Images

![2b24c7775d54b08d47a95f1e94c20fc28e487213a73849f6b406fd68e2283dbf.jpg](../iclr_results/656_AIMS.au_ A Dataset for the Analysis of Modern Slavery Countermeasures in Corporate Statements/images/2b24c7775d54b08d47a95f1e94c20fc28e487213a73849f6b406fd68e2283dbf.jpg)

![3c9956c7dea045505d2c17c12fced982d3f89a67a0477da75d459842b7f6d1bb.jpg](../iclr_results/656_AIMS.au_ A Dataset for the Analysis of Modern Slavery Countermeasures in Corporate Statements/images/3c9956c7dea045505d2c17c12fced982d3f89a67a0477da75d459842b7f6d1bb.jpg)

![3ee518be817d8b1998f846250058ebe4ea945e8ea40366a5e3c94fb1288ba87a.jpg](../iclr_results/656_AIMS.au_ A Dataset for the Analysis of Modern Slavery Countermeasures in Corporate Statements/images/3ee518be817d8b1998f846250058ebe4ea945e8ea40366a5e3c94fb1288ba87a.jpg)

![9442cdf9e0dff5def10f3c513c4d0b41454b55927785728d59495452dbbf1c03.jpg](../iclr_results/656_AIMS.au_ A Dataset for the Analysis of Modern Slavery Countermeasures in Corporate Statements/images/9442cdf9e0dff5def10f3c513c4d0b41454b55927785728d59495452dbbf1c03.jpg)

![f31c18b3145fe96b268236b6e891e5503d7fec60fe99c5c23cb6e33084262407.jpg](../iclr_results/656_AIMS.au_ A Dataset for the Analysis of Modern Slavery Countermeasures in Corporate Statements/images/f31c18b3145fe96b268236b6e891e5503d7fec60fe99c5c23cb6e33084262407.jpg)

### Tables

![1f528420db15cfd5596138ab2a56941a5bd386ddb3a0ce024dd8e90e131e994b.jpg](../iclr_results/656_AIMS.au_ A Dataset for the Analysis of Modern Slavery Countermeasures in Corporate Statements/tables/1f528420db15cfd5596138ab2a56941a5bd386ddb3a0ce024dd8e90e131e994b.jpg)

![484f8a383dbff1680d554653e36ce2f0235286079bc1b444288200cd211c2b78.jpg](../iclr_results/656_AIMS.au_ A Dataset for the Analysis of Modern Slavery Countermeasures in Corporate Statements/tables/484f8a383dbff1680d554653e36ce2f0235286079bc1b444288200cd211c2b78.jpg)

![4c530f9c11ced9603fffb1e5e433999a0c74cdd900f764dc5c2169e9cf1990e1.jpg](../iclr_results/656_AIMS.au_ A Dataset for the Analysis of Modern Slavery Countermeasures in Corporate Statements/tables/4c530f9c11ced9603fffb1e5e433999a0c74cdd900f764dc5c2169e9cf1990e1.jpg)

![625a9b4b83469a76190c261d4627111dae531d0d70ba6c0b4acff8d7acffda86.jpg](../iclr_results/656_AIMS.au_ A Dataset for the Analysis of Modern Slavery Countermeasures in Corporate Statements/tables/625a9b4b83469a76190c261d4627111dae531d0d70ba6c0b4acff8d7acffda86.jpg)

![940e0a72940a612467c5c3e4b02609ba260a6cff1bf75f67db9971fc965c6496.jpg](../iclr_results/656_AIMS.au_ A Dataset for the Analysis of Modern Slavery Countermeasures in Corporate Statements/tables/940e0a72940a612467c5c3e4b02609ba260a6cff1bf75f67db9971fc965c6496.jpg)

![e41aaca176b96465baee9a1350240a115ab4b0df86cb9fb3d31b73d70584df50.jpg](../iclr_results/656_AIMS.au_ A Dataset for the Analysis of Modern Slavery Countermeasures in Corporate Statements/tables/e41aaca176b96465baee9a1350240a115ab4b0df86cb9fb3d31b73d70584df50.jpg)

![e766df7c6619fdf783a87d747a39393e9193fa832b1ab9bcd9b2c728ae8b37bf.jpg](../iclr_results/656_AIMS.au_ A Dataset for the Analysis of Modern Slavery Countermeasures in Corporate Statements/tables/e766df7c6619fdf783a87d747a39393e9193fa832b1ab9bcd9b2c728ae8b37bf.jpg)

![f844a2d3db5fc2dce19514c4b1449a442ffc5f6d1bdd9abc228141887c6c10c7.jpg](../iclr_results/656_AIMS.au_ A Dataset for the Analysis of Modern Slavery Countermeasures in Corporate Statements/tables/f844a2d3db5fc2dce19514c4b1449a442ffc5f6d1bdd9abc228141887c6c10c7.jpg)

## ThermalGaussian: Thermal 3D Gaussian Splatting


### Images

![04ab64f52d61fb149724666f97091106256771e97d2587fa27c5d669f90a16e6.jpg](../iclr_results/657_ThermalGaussian_ Thermal 3D Gaussian Splatting/images/04ab64f52d61fb149724666f97091106256771e97d2587fa27c5d669f90a16e6.jpg)

![29fcc0f2679b62865d6731c792a95c16615125df53d383d472250bba256a6afb.jpg](../iclr_results/657_ThermalGaussian_ Thermal 3D Gaussian Splatting/images/29fcc0f2679b62865d6731c792a95c16615125df53d383d472250bba256a6afb.jpg)

![84aabed71e2f2b0017463fc9a5ae3a2bc4fa108b795bf96635a99bf3c0d9f7da.jpg](../iclr_results/657_ThermalGaussian_ Thermal 3D Gaussian Splatting/images/84aabed71e2f2b0017463fc9a5ae3a2bc4fa108b795bf96635a99bf3c0d9f7da.jpg)

![b8333485a3c26dc5a1cc179b25169b933e6a07aeab82b60d2b8d63fd595aaabd.jpg](../iclr_results/657_ThermalGaussian_ Thermal 3D Gaussian Splatting/images/b8333485a3c26dc5a1cc179b25169b933e6a07aeab82b60d2b8d63fd595aaabd.jpg)

![e2535681526e141cfdad0c290a73a9e344f888ee012c341faf54bc674bba74cb.jpg](../iclr_results/657_ThermalGaussian_ Thermal 3D Gaussian Splatting/images/e2535681526e141cfdad0c290a73a9e344f888ee012c341faf54bc674bba74cb.jpg)

![eb5e94853b727ecc4f30a0097bc79f7a75f78f0710b94f1a08bd6f14febec6a4.jpg](../iclr_results/657_ThermalGaussian_ Thermal 3D Gaussian Splatting/images/eb5e94853b727ecc4f30a0097bc79f7a75f78f0710b94f1a08bd6f14febec6a4.jpg)

![fdb3b60a0e1eea4559c5f12ab45b1ed49e024a0591b5f274ed93dd7373a294a9.jpg](../iclr_results/657_ThermalGaussian_ Thermal 3D Gaussian Splatting/images/fdb3b60a0e1eea4559c5f12ab45b1ed49e024a0591b5f274ed93dd7373a294a9.jpg)

### Tables

![024ef4b3f39a2159ab919c206c4a84595ef7d12eeeb34afe89d5e8d771bfe2fa.jpg](../iclr_results/657_ThermalGaussian_ Thermal 3D Gaussian Splatting/tables/024ef4b3f39a2159ab919c206c4a84595ef7d12eeeb34afe89d5e8d771bfe2fa.jpg)

![9954debea389e6034710932076af5441a1bf66c53f13b18f3ac1ecef71156462.jpg](../iclr_results/657_ThermalGaussian_ Thermal 3D Gaussian Splatting/tables/9954debea389e6034710932076af5441a1bf66c53f13b18f3ac1ecef71156462.jpg)

![e78aeba2e0896cb35fcdce232de6fc35a81f5f3c1141fc82b5acf8e93239f057.jpg](../iclr_results/657_ThermalGaussian_ Thermal 3D Gaussian Splatting/tables/e78aeba2e0896cb35fcdce232de6fc35a81f5f3c1141fc82b5acf8e93239f057.jpg)

![f2578f70eb0cf0ed6317e6fdc31b7229697a21ad5ed5eeee73bc525b39d24730.jpg](../iclr_results/657_ThermalGaussian_ Thermal 3D Gaussian Splatting/tables/f2578f70eb0cf0ed6317e6fdc31b7229697a21ad5ed5eeee73bc525b39d24730.jpg)

## Scaling Wearable Foundation Models


### Images

![1b8a67e51bbad60d877388fdfafa218ae2e038e6ee613a93f7d352792c09a07f.jpg](../iclr_results/658_Scaling Wearable Foundation Models/images/1b8a67e51bbad60d877388fdfafa218ae2e038e6ee613a93f7d352792c09a07f.jpg)

![2adbc5776c2ac7c2f87730a0dc6b1d4cd6db3275811b630d9d0257768742b316.jpg](../iclr_results/658_Scaling Wearable Foundation Models/images/2adbc5776c2ac7c2f87730a0dc6b1d4cd6db3275811b630d9d0257768742b316.jpg)

![2c9b36ff375f409e1836a0c240e89b6058857ea2a06f9879b629afa2119f7e34.jpg](../iclr_results/658_Scaling Wearable Foundation Models/images/2c9b36ff375f409e1836a0c240e89b6058857ea2a06f9879b629afa2119f7e34.jpg)

![2f941671d330ea0196e014d75c50436244fbce8758cddacdfc6bed8336fa005b.jpg](../iclr_results/658_Scaling Wearable Foundation Models/images/2f941671d330ea0196e014d75c50436244fbce8758cddacdfc6bed8336fa005b.jpg)

![304321a8d21bc70db38c81ac22300b83a33614b3cb81f1efd0f93c4e4b5cfe7a.jpg](../iclr_results/658_Scaling Wearable Foundation Models/images/304321a8d21bc70db38c81ac22300b83a33614b3cb81f1efd0f93c4e4b5cfe7a.jpg)

![32312565684c69e529033511081aa732466b4fb72cb766a2f96ce1d4fc1fd4a2.jpg](../iclr_results/658_Scaling Wearable Foundation Models/images/32312565684c69e529033511081aa732466b4fb72cb766a2f96ce1d4fc1fd4a2.jpg)

![854e291449b8a3ec50e68e68207720d69a2ca8418f0637a40d034d8472f9f21e.jpg](../iclr_results/658_Scaling Wearable Foundation Models/images/854e291449b8a3ec50e68e68207720d69a2ca8418f0637a40d034d8472f9f21e.jpg)

![8f4e3114df5166570d1ab608e123d73030fe63860510bcdfff8001d0800e62b5.jpg](../iclr_results/658_Scaling Wearable Foundation Models/images/8f4e3114df5166570d1ab608e123d73030fe63860510bcdfff8001d0800e62b5.jpg)

![94b9b5982d2a0382a6e828cea1af7e44403b2c7c82341c6f5cdb182f3454b85c.jpg](../iclr_results/658_Scaling Wearable Foundation Models/images/94b9b5982d2a0382a6e828cea1af7e44403b2c7c82341c6f5cdb182f3454b85c.jpg)

![a51fce8d7a39a11123f767e2cf8f466865861bd9e27cb4c5b0d468b905993dc5.jpg](../iclr_results/658_Scaling Wearable Foundation Models/images/a51fce8d7a39a11123f767e2cf8f466865861bd9e27cb4c5b0d468b905993dc5.jpg)

![dcb2186401f33647a922be41818fe0f7bc49ec4b3be60b893c0b5a493fb07b93.jpg](../iclr_results/658_Scaling Wearable Foundation Models/images/dcb2186401f33647a922be41818fe0f7bc49ec4b3be60b893c0b5a493fb07b93.jpg)

![e7548449ad0dcf6f034489882e3b32b4ea8a415067037f58bf9df7d2443ff945.jpg](../iclr_results/658_Scaling Wearable Foundation Models/images/e7548449ad0dcf6f034489882e3b32b4ea8a415067037f58bf9df7d2443ff945.jpg)

![e84921f4e9841bcbdba9863193991d00a3bf250a3dcb580844d6cc11b16dc6f0.jpg](../iclr_results/658_Scaling Wearable Foundation Models/images/e84921f4e9841bcbdba9863193991d00a3bf250a3dcb580844d6cc11b16dc6f0.jpg)

![fb7f556cd1b1f4bb434651c6d10f67dbc9f8fdfd0b18e9643b764dba8a22c82c.jpg](../iclr_results/658_Scaling Wearable Foundation Models/images/fb7f556cd1b1f4bb434651c6d10f67dbc9f8fdfd0b18e9643b764dba8a22c82c.jpg)

### Tables

![09ace18e881de136a1a63288dbbc86135fe4b39ae332832b713335493ba8f553.jpg](../iclr_results/658_Scaling Wearable Foundation Models/tables/09ace18e881de136a1a63288dbbc86135fe4b39ae332832b713335493ba8f553.jpg)

![0c99700122094f3d9e077414fbb9b0a358798065e16dd6eb253e86b73ee057ab.jpg](../iclr_results/658_Scaling Wearable Foundation Models/tables/0c99700122094f3d9e077414fbb9b0a358798065e16dd6eb253e86b73ee057ab.jpg)

![0df6df4c3bebb8d476e898f58022036c9cacbc3d3ae1122c7bf3a8b90424e26b.jpg](../iclr_results/658_Scaling Wearable Foundation Models/tables/0df6df4c3bebb8d476e898f58022036c9cacbc3d3ae1122c7bf3a8b90424e26b.jpg)

![1a7e13acd5c9ce2cd78a20227aa954674a1317822e1af7b09f420bddbc787dfc.jpg](../iclr_results/658_Scaling Wearable Foundation Models/tables/1a7e13acd5c9ce2cd78a20227aa954674a1317822e1af7b09f420bddbc787dfc.jpg)

![3261d45c22dc90f5333438910a032ff8246f70cd63561453290fceaffd35badf.jpg](../iclr_results/658_Scaling Wearable Foundation Models/tables/3261d45c22dc90f5333438910a032ff8246f70cd63561453290fceaffd35badf.jpg)

![32d867bff67b595612092392b7b46688ae8018271509e5480dd1fccad4b0c2bf.jpg](../iclr_results/658_Scaling Wearable Foundation Models/tables/32d867bff67b595612092392b7b46688ae8018271509e5480dd1fccad4b0c2bf.jpg)

![3596e79fdf63419fcb90d01a8346ba79f69b07a486b055a3a5bb504f5e7bff52.jpg](../iclr_results/658_Scaling Wearable Foundation Models/tables/3596e79fdf63419fcb90d01a8346ba79f69b07a486b055a3a5bb504f5e7bff52.jpg)

![385d7ce135cca197113512bc35baa5564c15b258c3aad07bee53511060c4dc5b.jpg](../iclr_results/658_Scaling Wearable Foundation Models/tables/385d7ce135cca197113512bc35baa5564c15b258c3aad07bee53511060c4dc5b.jpg)

![494717c36162a3ef17ef893ba355ea6f784a3b08e7e3d8b1f92d91b08a94aab3.jpg](../iclr_results/658_Scaling Wearable Foundation Models/tables/494717c36162a3ef17ef893ba355ea6f784a3b08e7e3d8b1f92d91b08a94aab3.jpg)

![4cdc9c1dcdec5db67be359f0cae10c123371260205f413f2e30a2c3a8ac6a822.jpg](../iclr_results/658_Scaling Wearable Foundation Models/tables/4cdc9c1dcdec5db67be359f0cae10c123371260205f413f2e30a2c3a8ac6a822.jpg)

![6dfd1279e6e2ef7ba0fa4022f8274ccb17238d815f2056f1a7a7eecafe203592.jpg](../iclr_results/658_Scaling Wearable Foundation Models/tables/6dfd1279e6e2ef7ba0fa4022f8274ccb17238d815f2056f1a7a7eecafe203592.jpg)

![6ff9f4302a9e71ea2a2ae926faae5c6658036427eace010cab471368442ef7e4.jpg](../iclr_results/658_Scaling Wearable Foundation Models/tables/6ff9f4302a9e71ea2a2ae926faae5c6658036427eace010cab471368442ef7e4.jpg)

![75e17bf521f20b1ef6b2f78eb396de1c873d9f7342a6d2b61564fa8df1e40423.jpg](../iclr_results/658_Scaling Wearable Foundation Models/tables/75e17bf521f20b1ef6b2f78eb396de1c873d9f7342a6d2b61564fa8df1e40423.jpg)

![7ad960658b2714bc85244e072f12c4d0cba188bf348b1a882f2d838bbb80d9ef.jpg](../iclr_results/658_Scaling Wearable Foundation Models/tables/7ad960658b2714bc85244e072f12c4d0cba188bf348b1a882f2d838bbb80d9ef.jpg)

![865d4a92f7246bb2603796cf53c0f18b3fdfb0cd667106aa38022aa4fa670d97.jpg](../iclr_results/658_Scaling Wearable Foundation Models/tables/865d4a92f7246bb2603796cf53c0f18b3fdfb0cd667106aa38022aa4fa670d97.jpg)

![928727606aac21298576a75b4c64ba24a12e8b18c4553c731639c0827707d30b.jpg](../iclr_results/658_Scaling Wearable Foundation Models/tables/928727606aac21298576a75b4c64ba24a12e8b18c4553c731639c0827707d30b.jpg)

![93d3beebf3e750e79a1214d5070480de472251227c6e27c3ce7a27e6da0fc866.jpg](../iclr_results/658_Scaling Wearable Foundation Models/tables/93d3beebf3e750e79a1214d5070480de472251227c6e27c3ce7a27e6da0fc866.jpg)

![9c1f3ba176d240d84cb3d1894d713e79788544c1d54ea236ce8b8c34ac5dc7d8.jpg](../iclr_results/658_Scaling Wearable Foundation Models/tables/9c1f3ba176d240d84cb3d1894d713e79788544c1d54ea236ce8b8c34ac5dc7d8.jpg)

![c098b452a584cdd1ed90bd42c1f0c8cbd13151181579f8d13039290603c8935c.jpg](../iclr_results/658_Scaling Wearable Foundation Models/tables/c098b452a584cdd1ed90bd42c1f0c8cbd13151181579f8d13039290603c8935c.jpg)

![c7d30f0bf1e1bc2ed9d2a59565f9fcb2894fc8bff0962f584f5928ba348176d1.jpg](../iclr_results/658_Scaling Wearable Foundation Models/tables/c7d30f0bf1e1bc2ed9d2a59565f9fcb2894fc8bff0962f584f5928ba348176d1.jpg)

![dd28a1db06e6d5c6e48852ebe763c0cb3ad11c90a62735c825da332efeece781.jpg](../iclr_results/658_Scaling Wearable Foundation Models/tables/dd28a1db06e6d5c6e48852ebe763c0cb3ad11c90a62735c825da332efeece781.jpg)

![e42728931969779b1a02d56147aeaa4921d382d37db9617d009e9e9cdfb61930.jpg](../iclr_results/658_Scaling Wearable Foundation Models/tables/e42728931969779b1a02d56147aeaa4921d382d37db9617d009e9e9cdfb61930.jpg)

![eca66e880fb89b4e815c0551ef93677cf1cc3687b5b273e9b794ef9e22fcf215.jpg](../iclr_results/658_Scaling Wearable Foundation Models/tables/eca66e880fb89b4e815c0551ef93677cf1cc3687b5b273e9b794ef9e22fcf215.jpg)

## Omni-MATH: A Universal Olympiad Level Mathematic Benchmark for Large Language Models


### Images

![067b2d05baad0e909d5b4fc119ca8f0ceaa41360dba17d710d4d8148d581a017.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/images/067b2d05baad0e909d5b4fc119ca8f0ceaa41360dba17d710d4d8148d581a017.jpg)

![10dd49e3b15c0c039877ee8911a5a499c08c9172fb3445c342a100502d784270.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/images/10dd49e3b15c0c039877ee8911a5a499c08c9172fb3445c342a100502d784270.jpg)

![2a564c1fd7c6f763eba74819fe81fea8b897037e903b0ffa5d51a303c211c630.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/images/2a564c1fd7c6f763eba74819fe81fea8b897037e903b0ffa5d51a303c211c630.jpg)

![2f6ff0f9303b9ac1a0eb7e1fd83267ce71df5f2af36a0be387d174e4a38e066d.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/images/2f6ff0f9303b9ac1a0eb7e1fd83267ce71df5f2af36a0be387d174e4a38e066d.jpg)

![48cf895fc69b3ce696fc816bf0576a1b657fe07a9b7a8d312e30ebe15b81a16b.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/images/48cf895fc69b3ce696fc816bf0576a1b657fe07a9b7a8d312e30ebe15b81a16b.jpg)

![4b2e7ad3601ce68ad3eec29db451546c9f3091e5b2b06343c301958d2fdebbc1.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/images/4b2e7ad3601ce68ad3eec29db451546c9f3091e5b2b06343c301958d2fdebbc1.jpg)

![93942d589f2a2635c681ec7bafaa23a7800b95db8f17c663e30eb97d872bb853.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/images/93942d589f2a2635c681ec7bafaa23a7800b95db8f17c663e30eb97d872bb853.jpg)

![a05ee6332076549f26f1232cf151cc858409c864de15c844a4a1849cbb50bc0d.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/images/a05ee6332076549f26f1232cf151cc858409c864de15c844a4a1849cbb50bc0d.jpg)

![ab1859c1fac8d4e845aa8d0ea5fe711153c4ec0ec3ef5a6d20f4f568c5cecfd5.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/images/ab1859c1fac8d4e845aa8d0ea5fe711153c4ec0ec3ef5a6d20f4f568c5cecfd5.jpg)

![b47c554bea967dd6e17f05848120603344481c8e8b7971724b7847ceb61dc103.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/images/b47c554bea967dd6e17f05848120603344481c8e8b7971724b7847ceb61dc103.jpg)

![ca48ab705e4a37b3a81807dce57a8d5835ef9fb3730481ab6b63b67ab6180b9c.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/images/ca48ab705e4a37b3a81807dce57a8d5835ef9fb3730481ab6b63b67ab6180b9c.jpg)

![cad09aca4bac4821fcd36468fca2146c2e55a9b88be42c73e03fc8fd8b659715.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/images/cad09aca4bac4821fcd36468fca2146c2e55a9b88be42c73e03fc8fd8b659715.jpg)

![da316d3cef6d77c244a6f7c93d5d23e7a581fb6bc64a89904dbfc1d54bd683ff.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/images/da316d3cef6d77c244a6f7c93d5d23e7a581fb6bc64a89904dbfc1d54bd683ff.jpg)

![e24a65d3fe7656c151735068776cdc135b581863c9e7fbf5ea9f7f947f2c5e49.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/images/e24a65d3fe7656c151735068776cdc135b581863c9e7fbf5ea9f7f947f2c5e49.jpg)

![e4967d700b7d3693a3d164a7ad084c657bff22ca42512dfcc7bd4ec057ae0b1b.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/images/e4967d700b7d3693a3d164a7ad084c657bff22ca42512dfcc7bd4ec057ae0b1b.jpg)

### Tables

![30fd9d6fa2679c26c5d911cdb50647b6d7fce2fd2ff53602b85110bd3cf3106b.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/tables/30fd9d6fa2679c26c5d911cdb50647b6d7fce2fd2ff53602b85110bd3cf3106b.jpg)

![5907342a14e467afdde02e1e36ea1412e40c64ed04e2cf66a6a1561ee0e6b056.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/tables/5907342a14e467afdde02e1e36ea1412e40c64ed04e2cf66a6a1561ee0e6b056.jpg)

![5ef9bc8b85a5064fbac1c07da6319b75076c716f06d5c94fb2f9ac06bd4838d9.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/tables/5ef9bc8b85a5064fbac1c07da6319b75076c716f06d5c94fb2f9ac06bd4838d9.jpg)

![7d0bb0fb9e11c4f2fa852a0c9a03d3cfef04ef9cf353bb5215c995124465f966.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/tables/7d0bb0fb9e11c4f2fa852a0c9a03d3cfef04ef9cf353bb5215c995124465f966.jpg)

![9dd447400d8fc5978b8b69c5ac34d7cb6b789f1200db79cb6f838c25cfefe19b.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/tables/9dd447400d8fc5978b8b69c5ac34d7cb6b789f1200db79cb6f838c25cfefe19b.jpg)

![a3a5247f43b62812ec0d3912f054829a2a0a79a0959e72d5c57b20ef6e0a0c2a.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/tables/a3a5247f43b62812ec0d3912f054829a2a0a79a0959e72d5c57b20ef6e0a0c2a.jpg)

![b3e42b379915082ed3f94e3bd28f8d2b4bdf810b8c0154335b1f80750b43fd4b.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/tables/b3e42b379915082ed3f94e3bd28f8d2b4bdf810b8c0154335b1f80750b43fd4b.jpg)

![cd835310fe9066828aa9f761e9706827cacede2b1ee8a8d6ca91ef327f80d959.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/tables/cd835310fe9066828aa9f761e9706827cacede2b1ee8a8d6ca91ef327f80d959.jpg)

![ee5e5e7953f4df3b0a6794e9554c277341a11767d64a19e74a2545edde3f7165.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/tables/ee5e5e7953f4df3b0a6794e9554c277341a11767d64a19e74a2545edde3f7165.jpg)

![ee7cfd4409757800fd64520fe24d6d2abfa231230e69055bdf7b07ffd97bbaa1.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/tables/ee7cfd4409757800fd64520fe24d6d2abfa231230e69055bdf7b07ffd97bbaa1.jpg)

![f3bef0c0b1d15661a7985653777489b848a50ffe0686a3ce1321d396cacab16f.jpg](../iclr_results/659_Omni-MATH_ A Universal Olympiad Level Mathematic Benchmark for Large Language Models/tables/f3bef0c0b1d15661a7985653777489b848a50ffe0686a3ce1321d396cacab16f.jpg)

## Language-Image Models with 3D Understanding


### Images

![14ffa7438b2202ed5adda4400b8aca1b31617e170a410d7705f0d208ecf3ff8d.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/14ffa7438b2202ed5adda4400b8aca1b31617e170a410d7705f0d208ecf3ff8d.jpg)

![190f2b198d2f0b72131590936628df5eaedf3023346e2575fc8fc59a53e3b759.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/190f2b198d2f0b72131590936628df5eaedf3023346e2575fc8fc59a53e3b759.jpg)

![1d395448e0b9cec46e69ac0bb73079dbcd51b2fb1d87722aef2d98ceb5e3b9ff.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/1d395448e0b9cec46e69ac0bb73079dbcd51b2fb1d87722aef2d98ceb5e3b9ff.jpg)

![28e118c4c11892b9c9b48806f6aa1e12f2f40ca58bf54d8101ab95a7c9919f80.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/28e118c4c11892b9c9b48806f6aa1e12f2f40ca58bf54d8101ab95a7c9919f80.jpg)

![3c1dd108854719e1b731ea0554c2261c4d3a6c1d4d89c70edd39835a498dd877.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/3c1dd108854719e1b731ea0554c2261c4d3a6c1d4d89c70edd39835a498dd877.jpg)

![4a31b2245811bb997fc4a1f01bb8f44889476ae9408a2089c5a91cdbe7098e89.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/4a31b2245811bb997fc4a1f01bb8f44889476ae9408a2089c5a91cdbe7098e89.jpg)

![4e8b2292f38979c77d1f3c2da8a28972efbbe3e9bebf39a99423a47580924ca8.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/4e8b2292f38979c77d1f3c2da8a28972efbbe3e9bebf39a99423a47580924ca8.jpg)

![4eec61b361f7755cea283bc6ec748033cd63f9a29498fe2253d75db64c6be5e4.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/4eec61b361f7755cea283bc6ec748033cd63f9a29498fe2253d75db64c6be5e4.jpg)

![5b82db7f80cc067d2bcddb073a8b18de89fd30abdf530e8cec9073546073125a.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/5b82db7f80cc067d2bcddb073a8b18de89fd30abdf530e8cec9073546073125a.jpg)

![667223574dd93cbc971c7bbf03fcfeb574c1a3f57040dff2242c26594cef3a11.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/667223574dd93cbc971c7bbf03fcfeb574c1a3f57040dff2242c26594cef3a11.jpg)

![6af8f8205a45eb71fb1dbb949113dc29dca9413bba6b4ac21faa7e0c1f3690a9.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/6af8f8205a45eb71fb1dbb949113dc29dca9413bba6b4ac21faa7e0c1f3690a9.jpg)

![7b7f6a3eb8d97b9f2493d759b6f7acc20bf816ead92ef56b16bebaa10b142cc4.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/7b7f6a3eb8d97b9f2493d759b6f7acc20bf816ead92ef56b16bebaa10b142cc4.jpg)

![819d90165117897b78a9b9f76ef3b8791cceafc88eee5e0bdcf9827cdd5d5449.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/819d90165117897b78a9b9f76ef3b8791cceafc88eee5e0bdcf9827cdd5d5449.jpg)

![911e90b3651928a89bb87d8777d457c1a4aea4c84e44da954858f2a4c100e2d9.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/911e90b3651928a89bb87d8777d457c1a4aea4c84e44da954858f2a4c100e2d9.jpg)

![95bd18b97ff9dce00f5b9e099acd220ba987072400187f4b5d3289d185e79077.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/95bd18b97ff9dce00f5b9e099acd220ba987072400187f4b5d3289d185e79077.jpg)

![97ddaca7fcaf5e78ca9d47ec02f6980b1757a8b887c92577f021209271787cfb.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/97ddaca7fcaf5e78ca9d47ec02f6980b1757a8b887c92577f021209271787cfb.jpg)

![9bd93323739f029f18a1a127933c4281eaf6f1638db8811960baf1e760ff0a79.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/9bd93323739f029f18a1a127933c4281eaf6f1638db8811960baf1e760ff0a79.jpg)

![a7c90ae237edfc8bfc58b752e871ebac18a3a568857cf4ef9496b9c4f87960b8.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/a7c90ae237edfc8bfc58b752e871ebac18a3a568857cf4ef9496b9c4f87960b8.jpg)

![a9a2355dc46f0528df79998be42501f595d49546189f45da3be6f44613ad7a4c.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/a9a2355dc46f0528df79998be42501f595d49546189f45da3be6f44613ad7a4c.jpg)

![a9a4cd92d3fbfe211afb14e6200b0c1e2f569dada896a480b4d0800deb7cdedd.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/a9a4cd92d3fbfe211afb14e6200b0c1e2f569dada896a480b4d0800deb7cdedd.jpg)

![ab491b0205ea8336d2717d6aba6406eec1374940631ed038096e272410c4e7dc.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/ab491b0205ea8336d2717d6aba6406eec1374940631ed038096e272410c4e7dc.jpg)

![b835b28e88eaa71cbab65ada37347a4059b1e4cfddd6aeddfa6352c7dae53e15.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/b835b28e88eaa71cbab65ada37347a4059b1e4cfddd6aeddfa6352c7dae53e15.jpg)

![b90a52cc0292c4e8b3e7c03ebdd53f095b060c7d127f7ea40260fa3fbc7cdc7a.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/b90a52cc0292c4e8b3e7c03ebdd53f095b060c7d127f7ea40260fa3fbc7cdc7a.jpg)

![b92f36500cb76f60c9a39aa10c55f23d898bc7bd85183ae35eef64b578369ff2.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/b92f36500cb76f60c9a39aa10c55f23d898bc7bd85183ae35eef64b578369ff2.jpg)

![bc6c07bd5e9512b03d3fc9800f4b8c8ef4b47fd135ffc2f66efebffb7870fc1d.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/bc6c07bd5e9512b03d3fc9800f4b8c8ef4b47fd135ffc2f66efebffb7870fc1d.jpg)

![c15214b2fb2e0724f2d3150822b4f80952dc4cf00aeab840dd0b050ad0192ce5.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/c15214b2fb2e0724f2d3150822b4f80952dc4cf00aeab840dd0b050ad0192ce5.jpg)

![c9c0e1562aa80ecfb7b38f5de533afa9005f4b52c9227d62f78cced7d0bc5241.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/c9c0e1562aa80ecfb7b38f5de533afa9005f4b52c9227d62f78cced7d0bc5241.jpg)

![d0df56b6279e14b066a7909752dfd5486edcc298197ceb492ee50b1140db26b2.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/d0df56b6279e14b066a7909752dfd5486edcc298197ceb492ee50b1140db26b2.jpg)

![d5f2c46f80ed4643342c99b0c093ba08469a171bba5e186a929481eeafcdb045.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/d5f2c46f80ed4643342c99b0c093ba08469a171bba5e186a929481eeafcdb045.jpg)

![d70e4ff8c14033274e3fa888a50903c4b2483f227a4ee11f1a6f9fab453c2d0f.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/d70e4ff8c14033274e3fa888a50903c4b2483f227a4ee11f1a6f9fab453c2d0f.jpg)

![e54ebea9f3ce71fc0108e9891c573e40a5a2e2c480423490ddcfb3b4638665ce.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/e54ebea9f3ce71fc0108e9891c573e40a5a2e2c480423490ddcfb3b4638665ce.jpg)

![e9af47e4d9a086cf1265aa37b9d1d203c728a0c1dd49466245ee5a85d5760f77.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/e9af47e4d9a086cf1265aa37b9d1d203c728a0c1dd49466245ee5a85d5760f77.jpg)

![ee620f13a0d9848b7de573374d539f8cd16ed4a68cac6cdcdebfca358bcdbd8f.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/ee620f13a0d9848b7de573374d539f8cd16ed4a68cac6cdcdebfca358bcdbd8f.jpg)

![f430849bf6b286b8b318c56e95088c223aa250b68d0a1dda038d910147b775a9.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/f430849bf6b286b8b318c56e95088c223aa250b68d0a1dda038d910147b775a9.jpg)

![f83f6602eb3d13f865c530c79b46b156530985fa043188d4e17c01cf75c47062.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/f83f6602eb3d13f865c530c79b46b156530985fa043188d4e17c01cf75c47062.jpg)

![fc9706a7570aad4cb59a24810e337f6980ddfc6a9cb116a80d761197456dd3b1.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/fc9706a7570aad4cb59a24810e337f6980ddfc6a9cb116a80d761197456dd3b1.jpg)

![ff74052230482ec151c103b3538e433a29845b3cdfb04b32b843f0f7bfeea080.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/images/ff74052230482ec151c103b3538e433a29845b3cdfb04b32b843f0f7bfeea080.jpg)

### Tables

![1cda7833b257fb784f1b79e3daf25a13127aec94b9a88e79d7ca6f597bb55edd.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/tables/1cda7833b257fb784f1b79e3daf25a13127aec94b9a88e79d7ca6f597bb55edd.jpg)

![1ece83af78276cc02edb6365c454676169fbf5bebf1f6ef7dfc1fed50949ebe5.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/tables/1ece83af78276cc02edb6365c454676169fbf5bebf1f6ef7dfc1fed50949ebe5.jpg)

![37dd4eedf76478e053e478feeb0de99acc4036629bd2878dd6879938a5279631.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/tables/37dd4eedf76478e053e478feeb0de99acc4036629bd2878dd6879938a5279631.jpg)

![4b9b6bb5110ed76a4b5e64684928c85cb16069affe2641d85cd059fe861a7970.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/tables/4b9b6bb5110ed76a4b5e64684928c85cb16069affe2641d85cd059fe861a7970.jpg)

![4dd143c94574500566a9318c80737369be0d64d603f874c91aabb13c309affcf.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/tables/4dd143c94574500566a9318c80737369be0d64d603f874c91aabb13c309affcf.jpg)

![c07d63e474df66eff8f6e6743cc0779aa0d760ebdc2042bd84e3d04d96286921.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/tables/c07d63e474df66eff8f6e6743cc0779aa0d760ebdc2042bd84e3d04d96286921.jpg)

![da1a67258ddc55e92a76bf8a029e146387a02d08e8826c9eca7ed595a52e0a64.jpg](../iclr_results/660_Language-Image Models with 3D Understanding/tables/da1a67258ddc55e92a76bf8a029e146387a02d08e8826c9eca7ed595a52e0a64.jpg)

## NoVo: Norm Voting off Hallucinations with Attention Heads in Large Language Models


### Images

![0ab4977e3af0fdaa7096efd006030e4f1db05b8b1453a776810104cf9160f1d0.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/images/0ab4977e3af0fdaa7096efd006030e4f1db05b8b1453a776810104cf9160f1d0.jpg)

![314ca566e68b0da993fd3cecbd2cc8e33f8008d1e49cee6e403caaba93fb4b81.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/images/314ca566e68b0da993fd3cecbd2cc8e33f8008d1e49cee6e403caaba93fb4b81.jpg)

![42af015ca7c1aefa36cf1b23f380e3b005cd4bc6aaa701e61822b1b577beee32.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/images/42af015ca7c1aefa36cf1b23f380e3b005cd4bc6aaa701e61822b1b577beee32.jpg)

![4a7c167d6b403507fa1f4703ab48a2711a827d01c989922d1efb44fd02bba57b.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/images/4a7c167d6b403507fa1f4703ab48a2711a827d01c989922d1efb44fd02bba57b.jpg)

![5a0c085525f6f401ef22332adc10f284bebbe00c235fcc81cfb25322ae7a3c7f.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/images/5a0c085525f6f401ef22332adc10f284bebbe00c235fcc81cfb25322ae7a3c7f.jpg)

![714166a38ff90e1d9d86e25de86603d87bada2e81cea5a4a320d7189d0f3c02a.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/images/714166a38ff90e1d9d86e25de86603d87bada2e81cea5a4a320d7189d0f3c02a.jpg)

![b453d085e477395acb23bcb833872fa7b1ec7e25563ab55fc1a3640824a9c75e.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/images/b453d085e477395acb23bcb833872fa7b1ec7e25563ab55fc1a3640824a9c75e.jpg)

![bcc3dcfc21bc0e2b7762736a83a333deac8ab84fb06c22b3c3495f0d32980330.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/images/bcc3dcfc21bc0e2b7762736a83a333deac8ab84fb06c22b3c3495f0d32980330.jpg)

![c0bd86281afcd3de22bf554d01410a559b41416d992de33a2d3a8013d924f90e.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/images/c0bd86281afcd3de22bf554d01410a559b41416d992de33a2d3a8013d924f90e.jpg)

![c59752cd39c03c4a6c0badfa1e17785dc025aef98fe55bd88c80baf031dce466.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/images/c59752cd39c03c4a6c0badfa1e17785dc025aef98fe55bd88c80baf031dce466.jpg)

![d801d338d55a0aa87b68545c605e81ca7889a66820ff7fdafdcd27e86ec6b394.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/images/d801d338d55a0aa87b68545c605e81ca7889a66820ff7fdafdcd27e86ec6b394.jpg)

![f95587f77daa921549531fd43b644bc78cc9c046200564208f503d2f00633321.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/images/f95587f77daa921549531fd43b644bc78cc9c046200564208f503d2f00633321.jpg)

![fa1be0e34616ca2c29a87db643d53909238c47259a14baf9d3c5afa4e6d1e1c8.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/images/fa1be0e34616ca2c29a87db643d53909238c47259a14baf9d3c5afa4e6d1e1c8.jpg)

### Tables

![028f4e5e00912335d6fa79c3c3979fdf117e5885c9fa319e4a6bfec156fc7960.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/tables/028f4e5e00912335d6fa79c3c3979fdf117e5885c9fa319e4a6bfec156fc7960.jpg)

![069cc960acc1626d0b95c5172dbc148e77f3d805ebe0921b2f944bc517d0bde0.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/tables/069cc960acc1626d0b95c5172dbc148e77f3d805ebe0921b2f944bc517d0bde0.jpg)

![10a57de642b948811bc9c4cbc412175addd111413a27fee93cfd72bb66de2b33.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/tables/10a57de642b948811bc9c4cbc412175addd111413a27fee93cfd72bb66de2b33.jpg)

![248c94f807e4aef8e3c91091b91382e57309853b5d347389822522ff55bf790d.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/tables/248c94f807e4aef8e3c91091b91382e57309853b5d347389822522ff55bf790d.jpg)

![2a6f359f164285d6b08e997494341b529ada22f1503c6cabfc93aeebfddcca87.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/tables/2a6f359f164285d6b08e997494341b529ada22f1503c6cabfc93aeebfddcca87.jpg)

![538af8bd46304d76b9ae0218739868584d7378dcaf67eb7f33493b8a4cb8c584.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/tables/538af8bd46304d76b9ae0218739868584d7378dcaf67eb7f33493b8a4cb8c584.jpg)

![93f03f2dbc4df27f86701351e0a02a17dd93644b8c7124bf77b562ecd1e4ae44.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/tables/93f03f2dbc4df27f86701351e0a02a17dd93644b8c7124bf77b562ecd1e4ae44.jpg)

![9470886602e0a7a688af1ed111657138516d154aff6f28a393930a34570daaba.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/tables/9470886602e0a7a688af1ed111657138516d154aff6f28a393930a34570daaba.jpg)

![aa9731329fea9acc6a04c9fe219300a85a1aab1c7f69e2b65a6e7bf254989658.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/tables/aa9731329fea9acc6a04c9fe219300a85a1aab1c7f69e2b65a6e7bf254989658.jpg)

![b9c5b5d785e35458eeb20999e74fde3732fd0438a66fb324922d2f047bfc7664.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/tables/b9c5b5d785e35458eeb20999e74fde3732fd0438a66fb324922d2f047bfc7664.jpg)

![df823a720b5f2f594527384e69bdcfb06b753704fbe64ab19c5a1cbad47010c0.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/tables/df823a720b5f2f594527384e69bdcfb06b753704fbe64ab19c5a1cbad47010c0.jpg)

![e4d7081f36b8673fb2088991e9d68457d56ab5efaf103fe568c226192ce05842.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/tables/e4d7081f36b8673fb2088991e9d68457d56ab5efaf103fe568c226192ce05842.jpg)

![ec8dc000615b743b339e2203d724850ffa6cd732b69130ac00488a4912c85404.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/tables/ec8dc000615b743b339e2203d724850ffa6cd732b69130ac00488a4912c85404.jpg)

![f3e7941a88a967634188119c651c621d37a1bb722220470913330943d7981b0b.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/tables/f3e7941a88a967634188119c651c621d37a1bb722220470913330943d7981b0b.jpg)

![fd429625a3fabb5668c7a49bbd0b4d4861722f5fdd2e707db167ff47f47e616f.jpg](../iclr_results/661_NoVo_ Norm Voting off Hallucinations with Attention Heads in Large Language Models/tables/fd429625a3fabb5668c7a49bbd0b4d4861722f5fdd2e707db167ff47f47e616f.jpg)

## Generative Monoculture in Large Language Models


### Images

![0a50e28f2ce53e6ad39ae90b8dd7ed2f03bd69b7b22bd03403ae96b15fb2426d.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/0a50e28f2ce53e6ad39ae90b8dd7ed2f03bd69b7b22bd03403ae96b15fb2426d.jpg)

![0aeeca8442fa2a392507da8ea1fc1fabd13b2a2ac37069ab7f7cd92c9688700d.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/0aeeca8442fa2a392507da8ea1fc1fabd13b2a2ac37069ab7f7cd92c9688700d.jpg)

![12968ddbc5aec3f5d8063759c5b768ffb061a091e07faa18eeb899915d2ae2e5.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/12968ddbc5aec3f5d8063759c5b768ffb061a091e07faa18eeb899915d2ae2e5.jpg)

![15216f93bdbfcd74b65ad1bd959af92046c02121af12d346ac94b76d18b476f9.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/15216f93bdbfcd74b65ad1bd959af92046c02121af12d346ac94b76d18b476f9.jpg)

![15d72a4f270ea55c533d3cb8aa0cb3fc1a6a468238170d4395c9dad98b770d49.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/15d72a4f270ea55c533d3cb8aa0cb3fc1a6a468238170d4395c9dad98b770d49.jpg)

![1f219a4b3326c440367f4c69b0391c87fe3ab2af33ce9fa175e1c41bdc8241c2.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/1f219a4b3326c440367f4c69b0391c87fe3ab2af33ce9fa175e1c41bdc8241c2.jpg)

![2387a6e312bcf89cf7d15770fdb8a5f034cbad18e2fc34284443de5d80ca530e.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/2387a6e312bcf89cf7d15770fdb8a5f034cbad18e2fc34284443de5d80ca530e.jpg)

![34d497d1290eb2110c0a6e56ec11d106310d9d1ff0de04899857c6798462aa0e.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/34d497d1290eb2110c0a6e56ec11d106310d9d1ff0de04899857c6798462aa0e.jpg)

![35f4fb604904489a0a99250531bc1d3b06834cac689af0cb4a08c395f070cab4.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/35f4fb604904489a0a99250531bc1d3b06834cac689af0cb4a08c395f070cab4.jpg)

![3cc5e314bac970134362d7add50d9423405922621a1d48bb5c5b2b6b90ffb31b.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/3cc5e314bac970134362d7add50d9423405922621a1d48bb5c5b2b6b90ffb31b.jpg)

![5ef16c3ffa7a9d3ecb791cc19451efa05f3afb782f779f6e8e5f87722c820746.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/5ef16c3ffa7a9d3ecb791cc19451efa05f3afb782f779f6e8e5f87722c820746.jpg)

![5ef839c3252fc82957a4cdcbadba5c2e52ba48f786c3d6df677744790a9a6c0b.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/5ef839c3252fc82957a4cdcbadba5c2e52ba48f786c3d6df677744790a9a6c0b.jpg)

![7bcafa50d2cbcb1498a70f7c824a462d8b554b1f77a3c0713ed871e8f5577030.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/7bcafa50d2cbcb1498a70f7c824a462d8b554b1f77a3c0713ed871e8f5577030.jpg)

![7f411767b993a570c599a6a10571ba8be064291c9fe023f6dd61b882d6d2ff46.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/7f411767b993a570c599a6a10571ba8be064291c9fe023f6dd61b882d6d2ff46.jpg)

![82299e7d364c1b688681eefc36b4d77af6da7873b8e9b7a3f57c1287c60dfb56.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/82299e7d364c1b688681eefc36b4d77af6da7873b8e9b7a3f57c1287c60dfb56.jpg)

![8ef87bf63fe6574adeaf57cb3062f1d561efa6b7465be6ee51dbcf0824294cab.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/8ef87bf63fe6574adeaf57cb3062f1d561efa6b7465be6ee51dbcf0824294cab.jpg)

![92439f0694a3c07a1c39f6e84bba498f5fb3106b55abdf9180d6f4c362325476.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/92439f0694a3c07a1c39f6e84bba498f5fb3106b55abdf9180d6f4c362325476.jpg)

![93197f89fb2888fddbaf0340a0d8564843273dbc23a120ff9d7ab1ff002b26c1.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/93197f89fb2888fddbaf0340a0d8564843273dbc23a120ff9d7ab1ff002b26c1.jpg)

![959f1a2334dd35b56fdd5d61b10be06c17f10585db3232fd88e7f9f4cd655790.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/959f1a2334dd35b56fdd5d61b10be06c17f10585db3232fd88e7f9f4cd655790.jpg)

![96193c61c67a90748cb3d40eb673a7a99e3ecadc2a81e3ff9678b93a8ee1ce13.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/96193c61c67a90748cb3d40eb673a7a99e3ecadc2a81e3ff9678b93a8ee1ce13.jpg)

![a0d026afdadcbc15bd36aae2afef1236d17551c32b677c1032bdb79f69790d36.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/a0d026afdadcbc15bd36aae2afef1236d17551c32b677c1032bdb79f69790d36.jpg)

![a3018db3698e41ca1e90df692ed326e0adcf6209cc97ff305c23b10a49fece02.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/a3018db3698e41ca1e90df692ed326e0adcf6209cc97ff305c23b10a49fece02.jpg)

![a9c399993148c4730f4243f0a6147484db927f5430b9f585b05d21037e2d80c2.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/a9c399993148c4730f4243f0a6147484db927f5430b9f585b05d21037e2d80c2.jpg)

![c9bdfc92f615f0195d1c30cee12da5315e849d1d7d03746b78268dc15aca36f3.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/c9bdfc92f615f0195d1c30cee12da5315e849d1d7d03746b78268dc15aca36f3.jpg)

![d1ca4c354f0ba3b7786f08f6b76b2aa6c8cec93f4177e9c236b9283870e9611f.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/d1ca4c354f0ba3b7786f08f6b76b2aa6c8cec93f4177e9c236b9283870e9611f.jpg)

![dc07b0560086bf570d720755dc392ebbae9339ae758607916796c0a40925ccef.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/dc07b0560086bf570d720755dc392ebbae9339ae758607916796c0a40925ccef.jpg)

![dc59d7f363a6853cafa485890708a98a92c70d28042a821f1f8a684c851cf287.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/dc59d7f363a6853cafa485890708a98a92c70d28042a821f1f8a684c851cf287.jpg)

![e16615331e95f6b068f2e26446f8bc5132e8e6556fa03ef223f973d734590087.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/e16615331e95f6b068f2e26446f8bc5132e8e6556fa03ef223f973d734590087.jpg)

![eb1b176a735bdbab400c799d19f87db4de5a4900ac870db2fdf4d30a735c0fb4.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/eb1b176a735bdbab400c799d19f87db4de5a4900ac870db2fdf4d30a735c0fb4.jpg)

![eed2031b8c18f2af8f8463c3534be568de326f95d06a14b4d7b4af25e0fc7b2d.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/eed2031b8c18f2af8f8463c3534be568de326f95d06a14b4d7b4af25e0fc7b2d.jpg)

![f84dfed786e34ec344b058a4ccb921f41094be34a8b33e590ee3dc12c879eecd.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/f84dfed786e34ec344b058a4ccb921f41094be34a8b33e590ee3dc12c879eecd.jpg)

![ff84dadeb60d278bdfcbab7a18689835acebbaf9d13af4765930d429490ff6b8.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/images/ff84dadeb60d278bdfcbab7a18689835acebbaf9d13af4765930d429490ff6b8.jpg)

### Tables

![122775110148cb377d319162f3c1703b5883b19db7ac998ccf596af5a7be1909.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/tables/122775110148cb377d319162f3c1703b5883b19db7ac998ccf596af5a7be1909.jpg)

![1fc3f8a72dc851a13aa5d359381906810895b22c70d2426e018475df8143b5e5.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/tables/1fc3f8a72dc851a13aa5d359381906810895b22c70d2426e018475df8143b5e5.jpg)

![4aacfe0e5a0abd70c0881f7e61b30152911867c040b9ab63beef6971b0d048ce.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/tables/4aacfe0e5a0abd70c0881f7e61b30152911867c040b9ab63beef6971b0d048ce.jpg)

![4bf24117004066420a104f3a92afff65a9d68288ee3945e59491eabd36ab271c.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/tables/4bf24117004066420a104f3a92afff65a9d68288ee3945e59491eabd36ab271c.jpg)

![ccfa93d9a4bb8b7dbe36b1388208fa4cc60aed28ba0777f59620bf462f2c696d.jpg](../iclr_results/662_Generative Monoculture in Large Language Models/tables/ccfa93d9a4bb8b7dbe36b1388208fa4cc60aed28ba0777f59620bf462f2c696d.jpg)

## Accelerating Diffusion Transformers with Token-wise Feature Caching


### Images

![012229f4637a083563fc5ebc862ba95dfdcca1d6e5bceb3ef172c2df10744807.jpg](../iclr_results/663_Accelerating Diffusion Transformers with Token-wise Feature Caching/images/012229f4637a083563fc5ebc862ba95dfdcca1d6e5bceb3ef172c2df10744807.jpg)

![0c20ce9defa7b15ca76dc696dad5e2efc61040eb64a13e7074233f7f08e9ea8c.jpg](../iclr_results/663_Accelerating Diffusion Transformers with Token-wise Feature Caching/images/0c20ce9defa7b15ca76dc696dad5e2efc61040eb64a13e7074233f7f08e9ea8c.jpg)

![1ad81969201d6f2e8a7b1a6f24f01527abd2e4f1f0844de108f1bcd9600b7842.jpg](../iclr_results/663_Accelerating Diffusion Transformers with Token-wise Feature Caching/images/1ad81969201d6f2e8a7b1a6f24f01527abd2e4f1f0844de108f1bcd9600b7842.jpg)

![3abf61305e41fc8ded91e9fc90c27773bfae885ab0975d3338511b1556eaa40e.jpg](../iclr_results/663_Accelerating Diffusion Transformers with Token-wise Feature Caching/images/3abf61305e41fc8ded91e9fc90c27773bfae885ab0975d3338511b1556eaa40e.jpg)

![4dcf4aeb7f7a25f87ff70a06b22f5e889bd9edecd9c0fb403acccdeca659ce46.jpg](../iclr_results/663_Accelerating Diffusion Transformers with Token-wise Feature Caching/images/4dcf4aeb7f7a25f87ff70a06b22f5e889bd9edecd9c0fb403acccdeca659ce46.jpg)

![52dda155404467d4e7b7cf2fbf6d0425afe5c9870b7d67c4783c7123b1cb6661.jpg](../iclr_results/663_Accelerating Diffusion Transformers with Token-wise Feature Caching/images/52dda155404467d4e7b7cf2fbf6d0425afe5c9870b7d67c4783c7123b1cb6661.jpg)

![5befa84ac40fd4d1e390f497d3e470ce719c23f4f55327dc8e7ae61bc8fc9da6.jpg](../iclr_results/663_Accelerating Diffusion Transformers with Token-wise Feature Caching/images/5befa84ac40fd4d1e390f497d3e470ce719c23f4f55327dc8e7ae61bc8fc9da6.jpg)

![6f6fecdc84ff58e1b52503c4ef152962510a7536dce814272f29c3ae7b44418b.jpg](../iclr_results/663_Accelerating Diffusion Transformers with Token-wise Feature Caching/images/6f6fecdc84ff58e1b52503c4ef152962510a7536dce814272f29c3ae7b44418b.jpg)

![819fa27ae47ae8279e346ffec88fbfedf2216b4ee3cf179e2a5a5f6c351ead70.jpg](../iclr_results/663_Accelerating Diffusion Transformers with Token-wise Feature Caching/images/819fa27ae47ae8279e346ffec88fbfedf2216b4ee3cf179e2a5a5f6c351ead70.jpg)

![a33e247c3b7a5f2b3706a37d37a4587268faae2e6e4eefdc948205b694588af3.jpg](../iclr_results/663_Accelerating Diffusion Transformers with Token-wise Feature Caching/images/a33e247c3b7a5f2b3706a37d37a4587268faae2e6e4eefdc948205b694588af3.jpg)

![a6c9cd8395e3981483cb88474c9503e20256b3fcc95210d4a2030ecdb6682bc7.jpg](../iclr_results/663_Accelerating Diffusion Transformers with Token-wise Feature Caching/images/a6c9cd8395e3981483cb88474c9503e20256b3fcc95210d4a2030ecdb6682bc7.jpg)

![b14873950a2cce6859d52ac8906e1ff0ecafb9ca289fb98036bac8800768ad04.jpg](../iclr_results/663_Accelerating Diffusion Transformers with Token-wise Feature Caching/images/b14873950a2cce6859d52ac8906e1ff0ecafb9ca289fb98036bac8800768ad04.jpg)

![c2809344568714ec0d060155050f170477b4179acdbfedfc0e4c62aaf57becf2.jpg](../iclr_results/663_Accelerating Diffusion Transformers with Token-wise Feature Caching/images/c2809344568714ec0d060155050f170477b4179acdbfedfc0e4c62aaf57becf2.jpg)

![e4ffd7cdf13301c57022296206584655d6689ab8d83fa4b694ba57c1e50f1432.jpg](../iclr_results/663_Accelerating Diffusion Transformers with Token-wise Feature Caching/images/e4ffd7cdf13301c57022296206584655d6689ab8d83fa4b694ba57c1e50f1432.jpg)

![f73df189c10faecccb79a2b8671fb890ce1263df773d3ef7b02f1b8b39e76cf8.jpg](../iclr_results/663_Accelerating Diffusion Transformers with Token-wise Feature Caching/images/f73df189c10faecccb79a2b8671fb890ce1263df773d3ef7b02f1b8b39e76cf8.jpg)

### Tables

![42e793bbe4bccfe083a6c543212d15e02362afb22681ca53f896ff7ca83022ed.jpg](../iclr_results/663_Accelerating Diffusion Transformers with Token-wise Feature Caching/tables/42e793bbe4bccfe083a6c543212d15e02362afb22681ca53f896ff7ca83022ed.jpg)

![4a7d4ba273bcf6aa67535b5a5eac586a5ab6657d805f257b84473935550eecfc.jpg](../iclr_results/663_Accelerating Diffusion Transformers with Token-wise Feature Caching/tables/4a7d4ba273bcf6aa67535b5a5eac586a5ab6657d805f257b84473935550eecfc.jpg)

![774bcea7a224d33af2a7ef1b75567891017bd52228b74104854d872efe9a4d0f.jpg](../iclr_results/663_Accelerating Diffusion Transformers with Token-wise Feature Caching/tables/774bcea7a224d33af2a7ef1b75567891017bd52228b74104854d872efe9a4d0f.jpg)

![9d7e0ff73cd2a5c59fa2d8365238bca518443eb1744054073cfa477373030047.jpg](../iclr_results/663_Accelerating Diffusion Transformers with Token-wise Feature Caching/tables/9d7e0ff73cd2a5c59fa2d8365238bca518443eb1744054073cfa477373030047.jpg)

![a1ebb6da1e9079a2b0d9834a9a1591083ae8f89690678195b1d11c085c13be31.jpg](../iclr_results/663_Accelerating Diffusion Transformers with Token-wise Feature Caching/tables/a1ebb6da1e9079a2b0d9834a9a1591083ae8f89690678195b1d11c085c13be31.jpg)

![b21966926160472fded8434981b8e38960c8cdac202d8cb8fa1a5c9630306429.jpg](../iclr_results/663_Accelerating Diffusion Transformers with Token-wise Feature Caching/tables/b21966926160472fded8434981b8e38960c8cdac202d8cb8fa1a5c9630306429.jpg)

![dfdce7a9ea968d69b278bc71defb99df1573bc17d74f6ed6e179e68381e4bea6.jpg](../iclr_results/663_Accelerating Diffusion Transformers with Token-wise Feature Caching/tables/dfdce7a9ea968d69b278bc71defb99df1573bc17d74f6ed6e179e68381e4bea6.jpg)

## Point-SAM: Promptable 3D Segmentation Model for Point Clouds


### Images

![381f83745b5f91cee9f8518ae333bd52e0436e38a9ce00796c2166a8636d8ee1.jpg](../iclr_results/664_Point-SAM_ Promptable 3D Segmentation Model for Point Clouds/images/381f83745b5f91cee9f8518ae333bd52e0436e38a9ce00796c2166a8636d8ee1.jpg)

![4429cc1134054db70420d6c034cf7b4411b18495ff4598ceb35572c4a30156d7.jpg](../iclr_results/664_Point-SAM_ Promptable 3D Segmentation Model for Point Clouds/images/4429cc1134054db70420d6c034cf7b4411b18495ff4598ceb35572c4a30156d7.jpg)

![5baef2c861f51d8fed6f941c7a01a649f91b6d45d40f4caf9da79dd86d295644.jpg](../iclr_results/664_Point-SAM_ Promptable 3D Segmentation Model for Point Clouds/images/5baef2c861f51d8fed6f941c7a01a649f91b6d45d40f4caf9da79dd86d295644.jpg)

![70c1f95a20c6be92f122029aa4906d722c01f4122d9726920ba041aa3ba0aa75.jpg](../iclr_results/664_Point-SAM_ Promptable 3D Segmentation Model for Point Clouds/images/70c1f95a20c6be92f122029aa4906d722c01f4122d9726920ba041aa3ba0aa75.jpg)

![86cef6cb16a5b3fd4e83e9479abd5d6f8425ce3bbc11c2eed2e3a607371f5341.jpg](../iclr_results/664_Point-SAM_ Promptable 3D Segmentation Model for Point Clouds/images/86cef6cb16a5b3fd4e83e9479abd5d6f8425ce3bbc11c2eed2e3a607371f5341.jpg)

![b5e41e0ccf13b77bdabe7acebbf17ac927d85ce2c7387b20d5a0e5815608f315.jpg](../iclr_results/664_Point-SAM_ Promptable 3D Segmentation Model for Point Clouds/images/b5e41e0ccf13b77bdabe7acebbf17ac927d85ce2c7387b20d5a0e5815608f315.jpg)

![e8601efb247fe9c75b0802637e50f912f25d7224e7b0d51897d50193db5f0e13.jpg](../iclr_results/664_Point-SAM_ Promptable 3D Segmentation Model for Point Clouds/images/e8601efb247fe9c75b0802637e50f912f25d7224e7b0d51897d50193db5f0e13.jpg)

### Tables

![05466a4bf142a66601f94143776416aec0254e275e9ec305d27e51064eb6c194.jpg](../iclr_results/664_Point-SAM_ Promptable 3D Segmentation Model for Point Clouds/tables/05466a4bf142a66601f94143776416aec0254e275e9ec305d27e51064eb6c194.jpg)

![2bd884d1171fb572c124e973976449274bb947b6ed79d0f3145f62bd58eb792d.jpg](../iclr_results/664_Point-SAM_ Promptable 3D Segmentation Model for Point Clouds/tables/2bd884d1171fb572c124e973976449274bb947b6ed79d0f3145f62bd58eb792d.jpg)

![5400ceef5e0566e0994599fbd1a947088b0db01617ac06b2c498e9ff09a232ad.jpg](../iclr_results/664_Point-SAM_ Promptable 3D Segmentation Model for Point Clouds/tables/5400ceef5e0566e0994599fbd1a947088b0db01617ac06b2c498e9ff09a232ad.jpg)

![5bdb366ccaa892f2287645a24bef6465b97193cbe51330bbfb382d7599a482cb.jpg](../iclr_results/664_Point-SAM_ Promptable 3D Segmentation Model for Point Clouds/tables/5bdb366ccaa892f2287645a24bef6465b97193cbe51330bbfb382d7599a482cb.jpg)

![7d16254ab44189174ce6f21cd3aea12f8cf3f161960d50c88900594fb1b53986.jpg](../iclr_results/664_Point-SAM_ Promptable 3D Segmentation Model for Point Clouds/tables/7d16254ab44189174ce6f21cd3aea12f8cf3f161960d50c88900594fb1b53986.jpg)

![8bfcf83fd6449de97e95df08ce6846b23bb4fdabc6e1bf3b9680b866a8b35dc6.jpg](../iclr_results/664_Point-SAM_ Promptable 3D Segmentation Model for Point Clouds/tables/8bfcf83fd6449de97e95df08ce6846b23bb4fdabc6e1bf3b9680b866a8b35dc6.jpg)

![8c5a486c9f1348605983de07ed806180de677dfbdbdac8b49b54f39ee4bdd6b2.jpg](../iclr_results/664_Point-SAM_ Promptable 3D Segmentation Model for Point Clouds/tables/8c5a486c9f1348605983de07ed806180de677dfbdbdac8b49b54f39ee4bdd6b2.jpg)

![943d19da0e481d2adf9077372d6eb7416b6b144182bb286aa88c308984d54a26.jpg](../iclr_results/664_Point-SAM_ Promptable 3D Segmentation Model for Point Clouds/tables/943d19da0e481d2adf9077372d6eb7416b6b144182bb286aa88c308984d54a26.jpg)

![a303ef1ef84416e9bdb7d1a616675ca8d32d55a5bbdf48c961416f80dfe5b968.jpg](../iclr_results/664_Point-SAM_ Promptable 3D Segmentation Model for Point Clouds/tables/a303ef1ef84416e9bdb7d1a616675ca8d32d55a5bbdf48c961416f80dfe5b968.jpg)

![befde18af4a5255e0310b2be9a7148997d84df0c13e1dc34a521709d90e317a0.jpg](../iclr_results/664_Point-SAM_ Promptable 3D Segmentation Model for Point Clouds/tables/befde18af4a5255e0310b2be9a7148997d84df0c13e1dc34a521709d90e317a0.jpg)

![ccc1f0f43cc0f0cb651b66c60215913d68d074a1f3e8e8444339420e00c360c1.jpg](../iclr_results/664_Point-SAM_ Promptable 3D Segmentation Model for Point Clouds/tables/ccc1f0f43cc0f0cb651b66c60215913d68d074a1f3e8e8444339420e00c360c1.jpg)

![d761ba793da5837ecb00b1c98c8baa89aee34bc0ac43f166f9f138b6d662fb0c.jpg](../iclr_results/664_Point-SAM_ Promptable 3D Segmentation Model for Point Clouds/tables/d761ba793da5837ecb00b1c98c8baa89aee34bc0ac43f166f9f138b6d662fb0c.jpg)

![d7976b61235461d9efa8416e41194ce02a0ffe23e55852e6336c931bbc6ce591.jpg](../iclr_results/664_Point-SAM_ Promptable 3D Segmentation Model for Point Clouds/tables/d7976b61235461d9efa8416e41194ce02a0ffe23e55852e6336c931bbc6ce591.jpg)

![ec58ce379bbb1c538a5abccc19b117b4adc7244bb2b1fa1332946265ab5b1c09.jpg](../iclr_results/664_Point-SAM_ Promptable 3D Segmentation Model for Point Clouds/tables/ec58ce379bbb1c538a5abccc19b117b4adc7244bb2b1fa1332946265ab5b1c09.jpg)

## Towards Understanding the Universality of Transformers for Next-Token Prediction


### Images

![072ce60a856763edd77bdbfd0b2d00eef4ef1d229bb77e98fd961864a9f085d2.jpg](../iclr_results/665_Towards Understanding the Universality of Transformers for Next-Token Prediction/images/072ce60a856763edd77bdbfd0b2d00eef4ef1d229bb77e98fd961864a9f085d2.jpg)

![0f49440f62f7fa7b04fcd4698a3afbcf1f4d70de06e6bc34c34e035c7c036419.jpg](../iclr_results/665_Towards Understanding the Universality of Transformers for Next-Token Prediction/images/0f49440f62f7fa7b04fcd4698a3afbcf1f4d70de06e6bc34c34e035c7c036419.jpg)

![98c6900bacbd2d3fe720e18cbe65f34a4fd24fbe9b394475cd1413dabd884ab1.jpg](../iclr_results/665_Towards Understanding the Universality of Transformers for Next-Token Prediction/images/98c6900bacbd2d3fe720e18cbe65f34a4fd24fbe9b394475cd1413dabd884ab1.jpg)

![d0595bcd999ad25a01eb788b57b02466764cb0dd0d5aa272780972edb7c8bd2d.jpg](../iclr_results/665_Towards Understanding the Universality of Transformers for Next-Token Prediction/images/d0595bcd999ad25a01eb788b57b02466764cb0dd0d5aa272780972edb7c8bd2d.jpg)

## Disentangling Representations through Multi-task Learning


### Images

![0a664e02c93b09a266bbca803d037d9f7509af5f7dddc23ad89108240bfb2cff.jpg](../iclr_results/666_Disentangling Representations through Multi-task Learning/images/0a664e02c93b09a266bbca803d037d9f7509af5f7dddc23ad89108240bfb2cff.jpg)

![0d6daf66b0b651635043eadd1d1b517371cff9a5b4504f2358c9c31b91f00bf0.jpg](../iclr_results/666_Disentangling Representations through Multi-task Learning/images/0d6daf66b0b651635043eadd1d1b517371cff9a5b4504f2358c9c31b91f00bf0.jpg)

![0efe99a417da362cf277bba58f3d1c2a7ac368033c8ab62354fd1ddabdebb7a8.jpg](../iclr_results/666_Disentangling Representations through Multi-task Learning/images/0efe99a417da362cf277bba58f3d1c2a7ac368033c8ab62354fd1ddabdebb7a8.jpg)

![10dd5aef390a3b6dfe22f1653326e10a454dfa668b249a574d71a77200e1c3ac.jpg](../iclr_results/666_Disentangling Representations through Multi-task Learning/images/10dd5aef390a3b6dfe22f1653326e10a454dfa668b249a574d71a77200e1c3ac.jpg)

![1776cc168529e0da86be531b4d072486b9c9082179516d13e64e189b8decd2fe.jpg](../iclr_results/666_Disentangling Representations through Multi-task Learning/images/1776cc168529e0da86be531b4d072486b9c9082179516d13e64e189b8decd2fe.jpg)

![1838bce14ca89fa18e3f517de0c363d880f50302237952af5008f44eb4783937.jpg](../iclr_results/666_Disentangling Representations through Multi-task Learning/images/1838bce14ca89fa18e3f517de0c363d880f50302237952af5008f44eb4783937.jpg)

![2de401068df5fc19548dec55ecb0c169e364a1eec5e7b9a738f865b4c918d26c.jpg](../iclr_results/666_Disentangling Representations through Multi-task Learning/images/2de401068df5fc19548dec55ecb0c169e364a1eec5e7b9a738f865b4c918d26c.jpg)

![4330bf91e1b56c72d45507a8ff241ae00a02e392e8c573e592ed801c127353ea.jpg](../iclr_results/666_Disentangling Representations through Multi-task Learning/images/4330bf91e1b56c72d45507a8ff241ae00a02e392e8c573e592ed801c127353ea.jpg)

![643b3778f0c96f0802f0127877ebed849d23471e14e005aac19f5c24b7cc43ce.jpg](../iclr_results/666_Disentangling Representations through Multi-task Learning/images/643b3778f0c96f0802f0127877ebed849d23471e14e005aac19f5c24b7cc43ce.jpg)

![73328176842d9e1d705e80cf1b0a29c1dc7e8caf4e9d327d9138cf338d3547f7.jpg](../iclr_results/666_Disentangling Representations through Multi-task Learning/images/73328176842d9e1d705e80cf1b0a29c1dc7e8caf4e9d327d9138cf338d3547f7.jpg)

![94ecb6396b63dbf99f7e3e742b7f000700dd2111a9cbfc45513527f8fc5489a1.jpg](../iclr_results/666_Disentangling Representations through Multi-task Learning/images/94ecb6396b63dbf99f7e3e742b7f000700dd2111a9cbfc45513527f8fc5489a1.jpg)

![a2f70e103da2eda41e58fd6e009d4666ec0b21dfbed232af906332a89cef78de.jpg](../iclr_results/666_Disentangling Representations through Multi-task Learning/images/a2f70e103da2eda41e58fd6e009d4666ec0b21dfbed232af906332a89cef78de.jpg)

![b9ce729767c06713f52309a5e6b692ad0c878af19cd9eafafc8480f19d6afbf4.jpg](../iclr_results/666_Disentangling Representations through Multi-task Learning/images/b9ce729767c06713f52309a5e6b692ad0c878af19cd9eafafc8480f19d6afbf4.jpg)

![c668ab07b365afc4921815b2cd24bd7fac110ef54afbc250a26eee37172504d9.jpg](../iclr_results/666_Disentangling Representations through Multi-task Learning/images/c668ab07b365afc4921815b2cd24bd7fac110ef54afbc250a26eee37172504d9.jpg)

![c717ea87c9942a7e9ecbbf476ded61f0689511f089d241d47056343050f3cb66.jpg](../iclr_results/666_Disentangling Representations through Multi-task Learning/images/c717ea87c9942a7e9ecbbf476ded61f0689511f089d241d47056343050f3cb66.jpg)

![f324f9539e5e481e3ab0e3d1fee2b28659a273526b16a265ad56beb1bcf72ab5.jpg](../iclr_results/666_Disentangling Representations through Multi-task Learning/images/f324f9539e5e481e3ab0e3d1fee2b28659a273526b16a265ad56beb1bcf72ab5.jpg)

![f907a3d5c7e4a187b44a62c63c20534f46f39af66de7bd614301f0f433a1f95a.jpg](../iclr_results/666_Disentangling Representations through Multi-task Learning/images/f907a3d5c7e4a187b44a62c63c20534f46f39af66de7bd614301f0f433a1f95a.jpg)

### Tables

![1f4c734f033d5b25259c04bb2e7e685eb4bd1bb325790fb8a234c30f42563613.jpg](../iclr_results/666_Disentangling Representations through Multi-task Learning/tables/1f4c734f033d5b25259c04bb2e7e685eb4bd1bb325790fb8a234c30f42563613.jpg)

## APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding


### Images

![051da219575cdd9731e5684eb8722bfadf013c24aabc0f35de0febcc216b1007.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/images/051da219575cdd9731e5684eb8722bfadf013c24aabc0f35de0febcc216b1007.jpg)

![37b4b046850746d3556368dfd88f77b52e425178c71e58820ec9ad8adf78c4f4.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/images/37b4b046850746d3556368dfd88f77b52e425178c71e58820ec9ad8adf78c4f4.jpg)

![3ad85e864a743fa1b373680b245872e9a30d111993020a9f4cc3dedde97f4b78.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/images/3ad85e864a743fa1b373680b245872e9a30d111993020a9f4cc3dedde97f4b78.jpg)

![52ec0022462b7bae144d395b51330957ce5f8f6509e248bc6ea57cc60b3f6782.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/images/52ec0022462b7bae144d395b51330957ce5f8f6509e248bc6ea57cc60b3f6782.jpg)

![8bcecb695b7e1b54434e001b5c2c7ef8719ca3eb3d5a52ee7a882ddab08cf4a4.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/images/8bcecb695b7e1b54434e001b5c2c7ef8719ca3eb3d5a52ee7a882ddab08cf4a4.jpg)

![90b427400fda60b672fde18253b5192b4c5dbe8062e85aa32829cb1f265accb0.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/images/90b427400fda60b672fde18253b5192b4c5dbe8062e85aa32829cb1f265accb0.jpg)

![90ea307f2fde83ae5175c2b1221a3836ffe0dfd9ba0347c9516d85689504e318.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/images/90ea307f2fde83ae5175c2b1221a3836ffe0dfd9ba0347c9516d85689504e318.jpg)

![92f210bd5113c9656be9fbe49fa3905f126f27e12f68f8667a1e5b152b29a232.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/images/92f210bd5113c9656be9fbe49fa3905f126f27e12f68f8667a1e5b152b29a232.jpg)

![9f9057037c320c71239deb39bd25651a8bda5bc973eee8f526283550c459c7c7.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/images/9f9057037c320c71239deb39bd25651a8bda5bc973eee8f526283550c459c7c7.jpg)

![a52a88bd7240f76416aff58fff2221594a67c4bc7cad92cfa005dfb9cfb41d9d.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/images/a52a88bd7240f76416aff58fff2221594a67c4bc7cad92cfa005dfb9cfb41d9d.jpg)

![b77a160a4176409c93dee5753d80ac9e0807d7054f6b26c4fbb944a8ee4a55b9.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/images/b77a160a4176409c93dee5753d80ac9e0807d7054f6b26c4fbb944a8ee4a55b9.jpg)

![cdb99d2539d8846823852c25f9911c37fba17f59ed8c25e9203368af2b5257dd.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/images/cdb99d2539d8846823852c25f9911c37fba17f59ed8c25e9203368af2b5257dd.jpg)

![d710b30fc38af96ec70588c083b8d16f0eff9118ce91659fb3a28cfc357e6062.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/images/d710b30fc38af96ec70588c083b8d16f0eff9118ce91659fb3a28cfc357e6062.jpg)

![d8e9da3a89ef9734c739517a5c29b993613a08b68bddcd36b51b53bdb8cd46e3.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/images/d8e9da3a89ef9734c739517a5c29b993613a08b68bddcd36b51b53bdb8cd46e3.jpg)

![e277c6266a60bdca23b4270350eb37df3fed4a120d180762e0dacbf3bc2b5863.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/images/e277c6266a60bdca23b4270350eb37df3fed4a120d180762e0dacbf3bc2b5863.jpg)

![e3f60d71ae500b627c5ae485f3296808bb49c04b973862075fcbc2c682cba54f.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/images/e3f60d71ae500b627c5ae485f3296808bb49c04b973862075fcbc2c682cba54f.jpg)

![e45120254c464183710c893982f95467991f07d1b0684c06ac5763a533e6458c.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/images/e45120254c464183710c893982f95467991f07d1b0684c06ac5763a533e6458c.jpg)

![e924f7047caa763c4e0468dd7342553bc391a5c12bffe721f085f4faea71d883.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/images/e924f7047caa763c4e0468dd7342553bc391a5c12bffe721f085f4faea71d883.jpg)

![f27a4d744b6a23e8e8b8dc9457a5b4ac46e2a1182660f0e14f616bba0f70a2e9.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/images/f27a4d744b6a23e8e8b8dc9457a5b4ac46e2a1182660f0e14f616bba0f70a2e9.jpg)

![fa0e7a873f0504fa8e8169c38301d653ba3b3cde51bd493f584febc6568bdf84.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/images/fa0e7a873f0504fa8e8169c38301d653ba3b3cde51bd493f584febc6568bdf84.jpg)

### Tables

![1d894412aa50f88ca6f9de2edfcdd9a6fbd75f33865bcb26efa1199ab52cb58a.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/tables/1d894412aa50f88ca6f9de2edfcdd9a6fbd75f33865bcb26efa1199ab52cb58a.jpg)

![278fa0fcfdd5e55438ab33db455b364b8bacc59f1ab2ddfa8469046e855f994e.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/tables/278fa0fcfdd5e55438ab33db455b364b8bacc59f1ab2ddfa8469046e855f994e.jpg)

![2b395d29bfac4f8c66386a4737b2a4a4c28482bfbe94be4c912892ff310e4e47.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/tables/2b395d29bfac4f8c66386a4737b2a4a4c28482bfbe94be4c912892ff310e4e47.jpg)

![7f6c5f0ad13d632177f6964f5cb82f9b20127b87154bd0c2e6b18c5872079c23.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/tables/7f6c5f0ad13d632177f6964f5cb82f9b20127b87154bd0c2e6b18c5872079c23.jpg)

![9ada77f9a1768e5e49c14f07c207c58ecdf4cc84c73bcf42984a328057e1789f.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/tables/9ada77f9a1768e5e49c14f07c207c58ecdf4cc84c73bcf42984a328057e1789f.jpg)

![aa7de70721bc07f30b2ba308c37db45aa86aac52553da2cde43db2ef92322b91.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/tables/aa7de70721bc07f30b2ba308c37db45aa86aac52553da2cde43db2ef92322b91.jpg)

![d24e79ee42f7b7a9ce39e2c039b997480e9b52ed0789051b7e7f81129a734181.jpg](../iclr_results/668_APE_ Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding/tables/d24e79ee42f7b7a9ce39e2c039b997480e9b52ed0789051b7e7f81129a734181.jpg)

## Robots Pre-train Robots: Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets


### Images

![36ed230378353bfa963ee4a81c42508f5eb51667e6dce8526ae7b5cf8c1fedb9.jpg](../iclr_results/669_Robots Pre-train Robots_ Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets/images/36ed230378353bfa963ee4a81c42508f5eb51667e6dce8526ae7b5cf8c1fedb9.jpg)

![4343e35ec5b9fd3e4bb65c6606f6509c7782b066d96fc821f5938dbff50fecb5.jpg](../iclr_results/669_Robots Pre-train Robots_ Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets/images/4343e35ec5b9fd3e4bb65c6606f6509c7782b066d96fc821f5938dbff50fecb5.jpg)

![4b05b9c08daefc53ada98ccb2667540fcc372900814e01e499d66903ddc4a5e3.jpg](../iclr_results/669_Robots Pre-train Robots_ Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets/images/4b05b9c08daefc53ada98ccb2667540fcc372900814e01e499d66903ddc4a5e3.jpg)

![55ade71ac0d53291dd0c79e28ed9324ea621d25aba6370cce769e6d7356cd791.jpg](../iclr_results/669_Robots Pre-train Robots_ Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets/images/55ade71ac0d53291dd0c79e28ed9324ea621d25aba6370cce769e6d7356cd791.jpg)

![66b1402896dd2a40205694560b78379b4ef9d18760f5db89f51900c2b71c9b0c.jpg](../iclr_results/669_Robots Pre-train Robots_ Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets/images/66b1402896dd2a40205694560b78379b4ef9d18760f5db89f51900c2b71c9b0c.jpg)

![797b2d4fcb2c9c99fccb9ec112fda6daa0df4ce96b05518c83319c11568fd4b5.jpg](../iclr_results/669_Robots Pre-train Robots_ Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets/images/797b2d4fcb2c9c99fccb9ec112fda6daa0df4ce96b05518c83319c11568fd4b5.jpg)

![87da03d1776282c608dc30e49ab88900be877b42dcc1036966b1584261550413.jpg](../iclr_results/669_Robots Pre-train Robots_ Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets/images/87da03d1776282c608dc30e49ab88900be877b42dcc1036966b1584261550413.jpg)

![8f1e082935862baa975ac6516567713bf872c780755fb3f4375558155a3d8fac.jpg](../iclr_results/669_Robots Pre-train Robots_ Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets/images/8f1e082935862baa975ac6516567713bf872c780755fb3f4375558155a3d8fac.jpg)

![b013f5f4c5dccee8835913ce725ef2557b0480f9af61541bc2e75a5e68a4af5a.jpg](../iclr_results/669_Robots Pre-train Robots_ Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets/images/b013f5f4c5dccee8835913ce725ef2557b0480f9af61541bc2e75a5e68a4af5a.jpg)

![bf538bb5c1657da2b0688e070190970f1db7e3aa4be36c6f8fa6f7f4e91880b3.jpg](../iclr_results/669_Robots Pre-train Robots_ Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets/images/bf538bb5c1657da2b0688e070190970f1db7e3aa4be36c6f8fa6f7f4e91880b3.jpg)

![c6caa2a5d69b02fa47a55c4ed6b3eca6a0a2ccb703942342c723d913cad3a062.jpg](../iclr_results/669_Robots Pre-train Robots_ Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets/images/c6caa2a5d69b02fa47a55c4ed6b3eca6a0a2ccb703942342c723d913cad3a062.jpg)

![cab99ae76343f7cfe1c6b194c87f0efe55126207ae09e279729ca5f581cc5ed5.jpg](../iclr_results/669_Robots Pre-train Robots_ Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets/images/cab99ae76343f7cfe1c6b194c87f0efe55126207ae09e279729ca5f581cc5ed5.jpg)

![cc0059e201b7c8ee85a31183e7a6e830e561666708ccbbb6f14380c3eef1729d.jpg](../iclr_results/669_Robots Pre-train Robots_ Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets/images/cc0059e201b7c8ee85a31183e7a6e830e561666708ccbbb6f14380c3eef1729d.jpg)

![cce314eb51850530e05cf0358c6ae068d2fd4787da2b784842bbebbe43407a49.jpg](../iclr_results/669_Robots Pre-train Robots_ Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets/images/cce314eb51850530e05cf0358c6ae068d2fd4787da2b784842bbebbe43407a49.jpg)

![e4c3be95acde5fa7a40b7a8ce9e7bb901b7dcc7d3371b56c9da2d3dc1f8c7505.jpg](../iclr_results/669_Robots Pre-train Robots_ Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets/images/e4c3be95acde5fa7a40b7a8ce9e7bb901b7dcc7d3371b56c9da2d3dc1f8c7505.jpg)

### Tables

![94474c39ee04194f39ed325b4347048a2acbd8031d6ac2901d3dc7663dbea220.jpg](../iclr_results/669_Robots Pre-train Robots_ Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets/tables/94474c39ee04194f39ed325b4347048a2acbd8031d6ac2901d3dc7663dbea220.jpg)

![98415029126c2de11f6ffdbc9b79770b6f69fa0c5e573699d0de6e61ba65dd6d.jpg](../iclr_results/669_Robots Pre-train Robots_ Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets/tables/98415029126c2de11f6ffdbc9b79770b6f69fa0c5e573699d0de6e61ba65dd6d.jpg)

![a3611bf549f48c77b8ac775ab8f82ca4f97eea9a5f492001da14d43c3c20c188.jpg](../iclr_results/669_Robots Pre-train Robots_ Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets/tables/a3611bf549f48c77b8ac775ab8f82ca4f97eea9a5f492001da14d43c3c20c188.jpg)

![d2e599e4cd1fb2a8c7cd12104e465b5ec134ed7c2af0d4c8e10c36415ec2c6fd.jpg](../iclr_results/669_Robots Pre-train Robots_ Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets/tables/d2e599e4cd1fb2a8c7cd12104e465b5ec134ed7c2af0d4c8e10c36415ec2c6fd.jpg)

![e683f36d574ac10b6e7bc60e2b78b3ebb532fafce7ccf1baf71414ffc5fd81dd.jpg](../iclr_results/669_Robots Pre-train Robots_ Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets/tables/e683f36d574ac10b6e7bc60e2b78b3ebb532fafce7ccf1baf71414ffc5fd81dd.jpg)

## Causally Motivated Sycophancy Mitigation for Large Language Models


### Images

![0a2e0c620757dc97f64c7a25a3caba802218d2adb73a6c6718ee98d39f430bfe.jpg](../iclr_results/670_Causally Motivated Sycophancy Mitigation for Large Language Models/images/0a2e0c620757dc97f64c7a25a3caba802218d2adb73a6c6718ee98d39f430bfe.jpg)

![1b1ea81406b74dad98bf346c12ce36de3d4fc350d8ddfc94f083d14cd7a9b3a3.jpg](../iclr_results/670_Causally Motivated Sycophancy Mitigation for Large Language Models/images/1b1ea81406b74dad98bf346c12ce36de3d4fc350d8ddfc94f083d14cd7a9b3a3.jpg)

![1f0113ca9776ff5311c5a4a0a51d93c66db45836b580291f30808da6407d0d76.jpg](../iclr_results/670_Causally Motivated Sycophancy Mitigation for Large Language Models/images/1f0113ca9776ff5311c5a4a0a51d93c66db45836b580291f30808da6407d0d76.jpg)

![2686228758b3e9cfed037f845fbfd554f07a4bf7873cafd93e510a57f2bc7a25.jpg](../iclr_results/670_Causally Motivated Sycophancy Mitigation for Large Language Models/images/2686228758b3e9cfed037f845fbfd554f07a4bf7873cafd93e510a57f2bc7a25.jpg)

![2b436477f9c36bc013734eb3d86a10450b00cdebffeab2b348adf48a92d76417.jpg](../iclr_results/670_Causally Motivated Sycophancy Mitigation for Large Language Models/images/2b436477f9c36bc013734eb3d86a10450b00cdebffeab2b348adf48a92d76417.jpg)

![4c05b63097964f747a8b0cc7ea46182db37f8e94a872aca21115d3249882bc53.jpg](../iclr_results/670_Causally Motivated Sycophancy Mitigation for Large Language Models/images/4c05b63097964f747a8b0cc7ea46182db37f8e94a872aca21115d3249882bc53.jpg)

![52928add7669bbafc93fea8082e11e14a335fbff8a8c2e66d7a8fb20e680a289.jpg](../iclr_results/670_Causally Motivated Sycophancy Mitigation for Large Language Models/images/52928add7669bbafc93fea8082e11e14a335fbff8a8c2e66d7a8fb20e680a289.jpg)

![6724966bd0c9ab41b724485e9a151bc3306d7a440e4a0ccb0998dbae22dda301.jpg](../iclr_results/670_Causally Motivated Sycophancy Mitigation for Large Language Models/images/6724966bd0c9ab41b724485e9a151bc3306d7a440e4a0ccb0998dbae22dda301.jpg)

![6c20970489d350672a21a5212db5a73219d59daa1eb34be3722a6087ba58899f.jpg](../iclr_results/670_Causally Motivated Sycophancy Mitigation for Large Language Models/images/6c20970489d350672a21a5212db5a73219d59daa1eb34be3722a6087ba58899f.jpg)

![ce66c96d065b8524ca75c5a71c55b190a971adfa1b174422d85767f1dede425a.jpg](../iclr_results/670_Causally Motivated Sycophancy Mitigation for Large Language Models/images/ce66c96d065b8524ca75c5a71c55b190a971adfa1b174422d85767f1dede425a.jpg)

![d2f5cd93b33d1537511003f54555df87c77f2705d83dd50b5fe78bd559a6c7c2.jpg](../iclr_results/670_Causally Motivated Sycophancy Mitigation for Large Language Models/images/d2f5cd93b33d1537511003f54555df87c77f2705d83dd50b5fe78bd559a6c7c2.jpg)

### Tables

![1a269e227f3bb11ce25fd2b4038112db7e2067302e57c7188cc598d7cd09146b.jpg](../iclr_results/670_Causally Motivated Sycophancy Mitigation for Large Language Models/tables/1a269e227f3bb11ce25fd2b4038112db7e2067302e57c7188cc598d7cd09146b.jpg)

![1a28a0891dffd344351c1272647cc3b5c69a413b4e8d6aa05f8fb2afde441860.jpg](../iclr_results/670_Causally Motivated Sycophancy Mitigation for Large Language Models/tables/1a28a0891dffd344351c1272647cc3b5c69a413b4e8d6aa05f8fb2afde441860.jpg)

![1f9c604d1c1078985693208bab0bd200e7b66cbdfe729fc1f0e0bb0260f55bc4.jpg](../iclr_results/670_Causally Motivated Sycophancy Mitigation for Large Language Models/tables/1f9c604d1c1078985693208bab0bd200e7b66cbdfe729fc1f0e0bb0260f55bc4.jpg)

![46759dfadd2d9e092541eeb320231b746ad735e99d14d6b5c8eb1eaace9f52fb.jpg](../iclr_results/670_Causally Motivated Sycophancy Mitigation for Large Language Models/tables/46759dfadd2d9e092541eeb320231b746ad735e99d14d6b5c8eb1eaace9f52fb.jpg)

![489a918ee1dd41c7f99ec41ebbd7197604bb528fdb463af9afb6739b51846b2d.jpg](../iclr_results/670_Causally Motivated Sycophancy Mitigation for Large Language Models/tables/489a918ee1dd41c7f99ec41ebbd7197604bb528fdb463af9afb6739b51846b2d.jpg)

![ad522a364b6798a12b893fd059e392038c22913cef74317021c4fba28d474745.jpg](../iclr_results/670_Causally Motivated Sycophancy Mitigation for Large Language Models/tables/ad522a364b6798a12b893fd059e392038c22913cef74317021c4fba28d474745.jpg)

![c6ac8afbf155b2a67f88161dbc896928b1fa53d28951083f1335ae43c25c51a6.jpg](../iclr_results/670_Causally Motivated Sycophancy Mitigation for Large Language Models/tables/c6ac8afbf155b2a67f88161dbc896928b1fa53d28951083f1335ae43c25c51a6.jpg)

## Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron


### Images

![31e2f3e0ad07c1dbbcf329c73bf5fff004e0bee3c7ee9e63d15769c46f5dacb4.jpg](../iclr_results/671_Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron/images/31e2f3e0ad07c1dbbcf329c73bf5fff004e0bee3c7ee9e63d15769c46f5dacb4.jpg)

![42c1f80907de8f03b36205fc7d5897c8ac3e8aae086133ec83b05692c59fd17e.jpg](../iclr_results/671_Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron/images/42c1f80907de8f03b36205fc7d5897c8ac3e8aae086133ec83b05692c59fd17e.jpg)

![5bb7f5974f39307632a2dc1911109d3d2fcc4b9c232c095e330877980044691a.jpg](../iclr_results/671_Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron/images/5bb7f5974f39307632a2dc1911109d3d2fcc4b9c232c095e330877980044691a.jpg)

![6958453cdc4760f502f12dd7beeb0eebc1d2f7515efb8d8707ba6b6065e3b4ac.jpg](../iclr_results/671_Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron/images/6958453cdc4760f502f12dd7beeb0eebc1d2f7515efb8d8707ba6b6065e3b4ac.jpg)

![6ad07fb2d3c2e37bde9e5db1bb1672e0fc507789887c707fc4ac20b9ef4ebe49.jpg](../iclr_results/671_Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron/images/6ad07fb2d3c2e37bde9e5db1bb1672e0fc507789887c707fc4ac20b9ef4ebe49.jpg)

![cebe116b1e78fe5fff3de2f072ca12a57501e02750f0005cb4d2a7a235ae3d83.jpg](../iclr_results/671_Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron/images/cebe116b1e78fe5fff3de2f072ca12a57501e02750f0005cb4d2a7a235ae3d83.jpg)

![dbc34ee67b6340d908e0ad59fbad8621c8974d64988f99859b73f23069f406f0.jpg](../iclr_results/671_Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron/images/dbc34ee67b6340d908e0ad59fbad8621c8974d64988f99859b73f23069f406f0.jpg)

### Tables

![1ffa422e7ec22553064825da3d57af8e0e23c15b5381747411d81857662d89a6.jpg](../iclr_results/671_Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron/tables/1ffa422e7ec22553064825da3d57af8e0e23c15b5381747411d81857662d89a6.jpg)

![3024c0062039c866c11edd104e2c85e6c76d05c0b7b794eb83b0c50cf03437ad.jpg](../iclr_results/671_Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron/tables/3024c0062039c866c11edd104e2c85e6c76d05c0b7b794eb83b0c50cf03437ad.jpg)

![3307fe0771066a202d4cbcc4d01521f0620fc6e927f84372c46b2276bc600e6e.jpg](../iclr_results/671_Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron/tables/3307fe0771066a202d4cbcc4d01521f0620fc6e927f84372c46b2276bc600e6e.jpg)

![3800735b178f3174c4e8713b008ddae1c1d739de8afa0ffb5dffdcc0d075d767.jpg](../iclr_results/671_Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron/tables/3800735b178f3174c4e8713b008ddae1c1d739de8afa0ffb5dffdcc0d075d767.jpg)

![85833bcb0ca99ec4175b4af6c8d1650a2be1914412b9fc7b2c061e793b54f4c9.jpg](../iclr_results/671_Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron/tables/85833bcb0ca99ec4175b4af6c8d1650a2be1914412b9fc7b2c061e793b54f4c9.jpg)

## Pacmann: Efficient Private Approximate Nearest Neighbor Search


### Images

![1678b027388b46c0b9a3b1a368e2676d17ad621b53f38eda07e30342c33cc6ab.jpg](../iclr_results/672_Pacmann_ Efficient Private Approximate Nearest Neighbor Search/images/1678b027388b46c0b9a3b1a368e2676d17ad621b53f38eda07e30342c33cc6ab.jpg)

![1b27493ddaffe0d27347aea3226c60e4a48d22eb6686ecff0bc7feb54fc366ae.jpg](../iclr_results/672_Pacmann_ Efficient Private Approximate Nearest Neighbor Search/images/1b27493ddaffe0d27347aea3226c60e4a48d22eb6686ecff0bc7feb54fc366ae.jpg)

![6ab00fc247384c0828774b1981eedb86e522c4fdd4defcd21b88d0b65c3463b7.jpg](../iclr_results/672_Pacmann_ Efficient Private Approximate Nearest Neighbor Search/images/6ab00fc247384c0828774b1981eedb86e522c4fdd4defcd21b88d0b65c3463b7.jpg)

![9e405a90f917597b164c2de6e09d5cf3446c5f197f67d526f6e1b248a845cf17.jpg](../iclr_results/672_Pacmann_ Efficient Private Approximate Nearest Neighbor Search/images/9e405a90f917597b164c2de6e09d5cf3446c5f197f67d526f6e1b248a845cf17.jpg)

![bfd2e53eb6aac03271c86961a81622ab19a8ef2238083fcac71f8ce8663feb2a.jpg](../iclr_results/672_Pacmann_ Efficient Private Approximate Nearest Neighbor Search/images/bfd2e53eb6aac03271c86961a81622ab19a8ef2238083fcac71f8ce8663feb2a.jpg)

### Tables

![23cf4cc18aa176a13e358df304c00353ce92646a333dd0759a8bf529137abe9b.jpg](../iclr_results/672_Pacmann_ Efficient Private Approximate Nearest Neighbor Search/tables/23cf4cc18aa176a13e358df304c00353ce92646a333dd0759a8bf529137abe9b.jpg)

## Understanding the Generalization of In-Context Learning in Transformers: An Empirical Study

### Images

![0782dcd7050b991a0ccc823b6688fe6f8dd785df6dac9cca3e77392d15c2a75a.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/images/0782dcd7050b991a0ccc823b6688fe6f8dd785df6dac9cca3e77392d15c2a75a.jpg)

![1285b06d09800ab842f786ff4e6e4b17f917bd8b644da30cd28b914e02463819.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/images/1285b06d09800ab842f786ff4e6e4b17f917bd8b644da30cd28b914e02463819.jpg)

![191f5b6d21fb12aa20cd7e9ace042f7496b28255a021b409c966e0054acf8702.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/images/191f5b6d21fb12aa20cd7e9ace042f7496b28255a021b409c966e0054acf8702.jpg)

![1b2df203d45f59a21d9916e05145130355103e1293eb30bdf43c0de3033d1a73.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/images/1b2df203d45f59a21d9916e05145130355103e1293eb30bdf43c0de3033d1a73.jpg)

![1d3e0f5e5f66a674fc4744ec2be4fce45b8533fd003d24804e709c256496a30f.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/images/1d3e0f5e5f66a674fc4744ec2be4fce45b8533fd003d24804e709c256496a30f.jpg)

![2d6ea6a2a0a5db305ddd058d19e959cb0c2a56de52a8733e7e0eee03f9b415af.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/images/2d6ea6a2a0a5db305ddd058d19e959cb0c2a56de52a8733e7e0eee03f9b415af.jpg)

![4599b60bc448b996f977fa0d29be38540fbb9590e6def3e9aa41f653ba502196.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/images/4599b60bc448b996f977fa0d29be38540fbb9590e6def3e9aa41f653ba502196.jpg)

![493d4a6429e50da2be10bfa7ea4de7622e5eae026633c8f5c4759dca669d6aa7.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/images/493d4a6429e50da2be10bfa7ea4de7622e5eae026633c8f5c4759dca669d6aa7.jpg)

![60783376c9d9c0f9d9dacc1340d348a0d6ff2c25ae68e6e751dcf73ddd53a45b.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/images/60783376c9d9c0f9d9dacc1340d348a0d6ff2c25ae68e6e751dcf73ddd53a45b.jpg)

![8f011929791764b1d4b1019add293d1313078ed29b6dd5295b63e8e7d21c2e6e.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/images/8f011929791764b1d4b1019add293d1313078ed29b6dd5295b63e8e7d21c2e6e.jpg)

![8f4a468c1fb5eed2130932bea6090bf8704e53babbf681a6e6cdcbe49bfb7817.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/images/8f4a468c1fb5eed2130932bea6090bf8704e53babbf681a6e6cdcbe49bfb7817.jpg)

![8f6b06f8a851110db379ca11b0d751b8e2fc78186bbd5c904a312177ddd21c92.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/images/8f6b06f8a851110db379ca11b0d751b8e2fc78186bbd5c904a312177ddd21c92.jpg)

![935490e6a13840c0fd19daa02b1ace8c5122948b1b31a9bb66e6c557adf5ed30.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/images/935490e6a13840c0fd19daa02b1ace8c5122948b1b31a9bb66e6c557adf5ed30.jpg)

![a889ab9c2cbb29d2645c0ababf698fc353b51a6732fe54ef9c9285741a0837a9.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/images/a889ab9c2cbb29d2645c0ababf698fc353b51a6732fe54ef9c9285741a0837a9.jpg)

![ad093d219bc61c81ae6df65bd3295c88fd8a6e25e360c953dccef90731fbc556.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/images/ad093d219bc61c81ae6df65bd3295c88fd8a6e25e360c953dccef90731fbc556.jpg)

![b8e1ec352d9a47954d6e8882d8b3affa92d50dd4ee65ad50bd35df9dc9f53459.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/images/b8e1ec352d9a47954d6e8882d8b3affa92d50dd4ee65ad50bd35df9dc9f53459.jpg)

![ca72751cce09ca866655645cfcc87db5cc12c87f40ca3908cb7b5dd6828aa379.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/images/ca72751cce09ca866655645cfcc87db5cc12c87f40ca3908cb7b5dd6828aa379.jpg)

![cb27945114339ead3c5ef2a53c63199174746354fc431dc486e7e75aac90f353.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/images/cb27945114339ead3c5ef2a53c63199174746354fc431dc486e7e75aac90f353.jpg)

![ef50e047904cbd036a20595cd81de4221065d6e167b64f791b3993bf32c1455c.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/images/ef50e047904cbd036a20595cd81de4221065d6e167b64f791b3993bf32c1455c.jpg)

### Tables

![06add3651bfe32f09f2e337f5afa8b4dd3c9753d03352d4ac7ca672d6f7bd9c9.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/tables/06add3651bfe32f09f2e337f5afa8b4dd3c9753d03352d4ac7ca672d6f7bd9c9.jpg)

![0c00265895d5d32d3e9b8959ae7bb5d54fde220a16d5c09c52fef6a9abb9b1dc.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/tables/0c00265895d5d32d3e9b8959ae7bb5d54fde220a16d5c09c52fef6a9abb9b1dc.jpg)

![1b99c5148e8ce742ff940e28071f6ea67a056aadb955238c4acbe37692b0dd61.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/tables/1b99c5148e8ce742ff940e28071f6ea67a056aadb955238c4acbe37692b0dd61.jpg)

![3485ae86b2c9f55fc9e4dfa0032eebb86ea974a44b737913311eb9e7c67ad9bf.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/tables/3485ae86b2c9f55fc9e4dfa0032eebb86ea974a44b737913311eb9e7c67ad9bf.jpg)

![40c4cef7f8c0e6e7f2858ef0a39b4347453dd3192e12062e0c02ff17afc1bafb.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/tables/40c4cef7f8c0e6e7f2858ef0a39b4347453dd3192e12062e0c02ff17afc1bafb.jpg)

![42e8bab17fac361166effd7449bdebb2017761c12659e7f263a4cfc68cfb84e5.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/tables/42e8bab17fac361166effd7449bdebb2017761c12659e7f263a4cfc68cfb84e5.jpg)

![5321ab768180704e8eb686725d417af70fa9875c3b525d4e129e4a60bdab543d.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/tables/5321ab768180704e8eb686725d417af70fa9875c3b525d4e129e4a60bdab543d.jpg)

![659dc6c73789bae7964872fb5613b48b37e4592c29225e1e8cbf1ca90fdc70a8.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/tables/659dc6c73789bae7964872fb5613b48b37e4592c29225e1e8cbf1ca90fdc70a8.jpg)

![661ded41545d9dfcbae140c9d15ecbac09f35b3532d437e15ad82134e1003c71.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/tables/661ded41545d9dfcbae140c9d15ecbac09f35b3532d437e15ad82134e1003c71.jpg)

![694e428a89b6108aa7298df8f7ef20441c3a99d28d0f4cc0cd40b63f2843cbbe.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/tables/694e428a89b6108aa7298df8f7ef20441c3a99d28d0f4cc0cd40b63f2843cbbe.jpg)

![7fb3faa2184253d06643730da5a9de7efddeda9c0e66e631f42d9d3d74a4f858.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/tables/7fb3faa2184253d06643730da5a9de7efddeda9c0e66e631f42d9d3d74a4f858.jpg)

![8c5228dd5716c40515d89ebdbf27a2b98f604974fe59a6e104d6a88884017be7.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/tables/8c5228dd5716c40515d89ebdbf27a2b98f604974fe59a6e104d6a88884017be7.jpg)

![9c2811f5341b046c9af850254540412c3ed156d05ae6bc052cbc62f1285d4f86.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/tables/9c2811f5341b046c9af850254540412c3ed156d05ae6bc052cbc62f1285d4f86.jpg)

![9c68f8b8239f815dfcb5fa89f0629cb62d4a80af55dd12cd0b8f56394e455c3f.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/tables/9c68f8b8239f815dfcb5fa89f0629cb62d4a80af55dd12cd0b8f56394e455c3f.jpg)

![a2a1430869e8f5594ea1ba150ec8ef4b7d6953016f2b3dd924384f74efb6959e.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/tables/a2a1430869e8f5594ea1ba150ec8ef4b7d6953016f2b3dd924384f74efb6959e.jpg)

![aab1737d7d52f89a6ebcec259140bb6849a166c89493a59a081936c92aae57b8.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/tables/aab1737d7d52f89a6ebcec259140bb6849a166c89493a59a081936c92aae57b8.jpg)

![b28ce265608168e55f2704fedf3d87624e6955059d409e760be2da61d3cbf2bf.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/tables/b28ce265608168e55f2704fedf3d87624e6955059d409e760be2da61d3cbf2bf.jpg)

![ba66637d06a530c0b1e1e143fcc1b91fe45a903ea246a386d3aab8967ff483e3.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/tables/ba66637d06a530c0b1e1e143fcc1b91fe45a903ea246a386d3aab8967ff483e3.jpg)

![e2be61c4fc96e47c0dea8e386dfdd367d70de0935cc92a6e6ae61ebd51c92747.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/tables/e2be61c4fc96e47c0dea8e386dfdd367d70de0935cc92a6e6ae61ebd51c92747.jpg)

![fb25d7600eef53ff415a82e9844bb0c7ca4cd047e91893ce46763a5a051d71a4.jpg](../iclr_results/673_Understanding the Generalization of In-Context Learning in Transformers_ An Empirical Study/tables/fb25d7600eef53ff415a82e9844bb0c7ca4cd047e91893ce46763a5a051d71a4.jpg)
