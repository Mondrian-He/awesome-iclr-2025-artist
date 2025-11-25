# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 41 of 100

## 目录 (Table of Contents)

1. [LLaRA: Supercharging Robot Learning Data for Vision-Language Policy](#LLaRA-Supercharging-Robot-Learning-Data-for-Vision-Language-Policy)
2. [Can We Talk Models Into Seeing the World Differently?](#Can-We-Talk-Models-Into-Seeing-the-World-Differently)
3. [ClassDiffusion: More Aligned Personalization Tuning with Explicit Class Guidance](#ClassDiffusion-More-Aligned-Personalization-Tuning-with-Explicit-Class-Guidance)
4. [Utility-Directed Conformal Prediction: A Decision-Aware Framework for Actionable Uncertainty Quantification](#Utility-Directed-Conformal-Prediction-A-Decision-Aware-Framework-for-Actionable-Uncertainty-Quantification)
5. [Fast Summation of Radial Kernels via QMC Slicing](#Fast-Summation-of-Radial-Kernels-via-QMC-Slicing)
6. [Stochastic variance-reduced Gaussian variational inference on the Bures-Wasserstein manifold](#Stochastic-variance-reduced-Gaussian-variational-inference-on-the-Bures-Wasserstein-manifold)
7. [Why In-Context Learning Models are Good Few-Shot Learners?](#Why-In-Context-Learning-Models-are-Good-Few-Shot-Learners)
8. [3DitScene: Editing Any Scene via Language-guided Disentangled Gaussian Splatting](#3DitScene-Editing-Any-Scene-via-Language-guided-Disentangled-Gaussian-Splatting)
9. [Diffusion-NPO: Negative Preference Optimization for Better Preference Aligned Generation of Diffusion Models](#Diffusion-NPO-Negative-Preference-Optimization-for-Better-Preference-Aligned-Generation-of-Diffusion-Models)
10. [Duoduo CLIP: Efficient 3D Understanding with Multi-View Images](#Duoduo-CLIP-Efficient-3D-Understanding-with-Multi-View-Images)
11. [DenseGrounding: Improving Dense Language-Vision Semantics for Ego-centric 3D Visual Grounding](#DenseGrounding-Improving-Dense-Language-Vision-Semantics-for-Ego-centric-3D-Visual-Grounding)
12. [Provably Robust Explainable Graph Neural Networks against Graph Perturbation Attacks](#Provably-Robust-Explainable-Graph-Neural-Networks-against-Graph-Perturbation-Attacks)
13. [Near-optimal Active Regression of Single-Index Models](#Near-optimal-Active-Regression-of-Single-Index-Models)
14. [The Optimization Landscape of SGD Across the Feature Learning Strength](#The-Optimization-Landscape-of-SGD-Across-the-Feature-Learning-Strength)
15. [Spatial-Mamba: Effective Visual State Space Models via Structure-Aware State Fusion](#Spatial-Mamba-Effective-Visual-State-Space-Models-via-Structure-Aware-State-Fusion)
16. [EcoFace: Audio-Visual Emotional Co-Disentanglement Speech-Driven 3D Talking Face Generation](#EcoFace-Audio-Visual-Emotional-Co-Disentanglement-Speech-Driven-3D-Talking-Face-Generation)
17. [Adam-mini: Use Fewer Learning Rates To Gain More](#Adam-mini-Use-Fewer-Learning-Rates-To-Gain-More)
18. [Improving Graph Neural Networks by Learning Continuous Edge Directions](#Improving-Graph-Neural-Networks-by-Learning-Continuous-Edge-Directions)
19. [MuPT: A Generative Symbolic Music Pretrained Transformer](#MuPT-A-Generative-Symbolic-Music-Pretrained-Transformer)
20. [Residual Connections and Normalization Can Provably Prevent Oversmoothing in GNNs](#Residual-Connections-and-Normalization-Can-Provably-Prevent-Oversmoothing-in-GNNs)
21. [Minimax Optimal Reinforcement Learning with Quasi-Optimism](#Minimax-Optimal-Reinforcement-Learning-with-Quasi-Optimism)
22. [Interpreting Language Reward Models via Contrastive Explanations](#Interpreting-Language-Reward-Models-via-Contrastive-Explanations)
23. [Lipschitz Bandits in Optimal Space](#Lipschitz-Bandits-in-Optimal-Space)
24. [Bootstrapped Model Predictive Control](#Bootstrapped-Model-Predictive-Control)
25. [Disentangling 3D Animal Pose Dynamics with Scrubbed Conditional Latent Variables](#Disentangling-3D-Animal-Pose-Dynamics-with-Scrubbed-Conditional-Latent-Variables)
26. [Simple Guidance Mechanisms for Discrete Diffusion Models](#Simple-Guidance-Mechanisms-for-Discrete-Diffusion-Models)
27. [MM-EMBED: UNIVERSAL MULTIMODAL RETRIEVAL WITH MULTIMODAL LLMS](#MM-EMBED-UNIVERSAL-MULTIMODAL-RETRIEVAL-WITH-MULTIMODAL-LLMS)
28. [Language Guided Skill Discovery](#Language-Guided-Skill-Discovery)
29. [Valid Conformal Prediction for Dynamic GNNs](#Valid-Conformal-Prediction-for-Dynamic-GNNs)
30. [ECHOPulse: ECG Controlled Echocardio-gram Video Generation](#ECHOPulse-ECG-Controlled-Echocardio-gram-Video-Generation)
31. [SymDiff: Equivariant Diffusion via Stochastic Symmetrisation](#SymDiff-Equivariant-Diffusion-via-Stochastic-Symmetrisation)
32. [Parameter and Memory Efficient Pretraining via Low-rank Riemannian Optimization](#Parameter-and-Memory-Efficient-Pretraining-via-Low-rank-Riemannian-Optimization)
33. [Deep Distributed Optimization for Large-Scale Quadratic Programming](#Deep-Distributed-Optimization-for-Large-Scale-Quadratic-Programming)
34. [Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification](#Dynamic-LLaVA-Efficient-Multimodal-Large-Language-Models-via-Dynamic-Vision-language-Context-Sparsification)
35. [Neural Stochastic Differential Equations for Uncertainty-Aware Offline RL](#Neural-Stochastic-Differential-Equations-for-Uncertainty-Aware-Offline-RL)
36. [DPaI: Differentiable Pruning at Initialization with Node-Path Balance Principle](#DPaI-Differentiable-Pruning-at-Initialization-with-Node-Path-Balance-Principle)
37. [Improving Pretraining Data Using Perplexity Correlations](#Improving-Pretraining-Data-Using-Perplexity-Correlations)

---


## LLaRA: Supercharging Robot Learning Data for Vision-Language Policy

### Images

![03a76a9463259e7da53011791d7d3f0f339cbda79c24fba4db853574f24ac20a.jpg](../iclr_results/1490_DeciMamba_%20Exploring%20the%20Length%20Extrapolation%20Potential%20of%20Mamba/images/03a76a9463259e7da53011791d7d3f0f339cbda79c24fba4db853574f24ac20a.jpg)

![11c85f8aa21576104d999def780c3db58fa042dac8c01acc5def6c45172a7fcc.jpg](../iclr_results/1490_DeciMamba_%20Exploring%20the%20Length%20Extrapolation%20Potential%20of%20Mamba/images/11c85f8aa21576104d999def780c3db58fa042dac8c01acc5def6c45172a7fcc.jpg)

![2037153a8e798d0b498018dc300092d762a9ddd2be690ff5771836b2d480b5d0.jpg](../iclr_results/1490_DeciMamba_%20Exploring%20the%20Length%20Extrapolation%20Potential%20of%20Mamba/images/2037153a8e798d0b498018dc300092d762a9ddd2be690ff5771836b2d480b5d0.jpg)

![239c6c57c1dc70c8a8115f4ff06d887c0f43d3a2ade7ea236975fb031355187e.jpg](../iclr_results/1490_DeciMamba_%20Exploring%20the%20Length%20Extrapolation%20Potential%20of%20Mamba/images/239c6c57c1dc70c8a8115f4ff06d887c0f43d3a2ade7ea236975fb031355187e.jpg)

![445353766a8d4c5a6db2a6d086ffdc8388f8fa9e346e43ebab6d9e5133f7c275.jpg](../iclr_results/1490_DeciMamba_%20Exploring%20the%20Length%20Extrapolation%20Potential%20of%20Mamba/images/445353766a8d4c5a6db2a6d086ffdc8388f8fa9e346e43ebab6d9e5133f7c275.jpg)

![7b2a6cf0078a0eec458d9419691bc25fb87b7f6b0d0b031f3d787b7367b66c79.jpg](../iclr_results/1490_DeciMamba_%20Exploring%20the%20Length%20Extrapolation%20Potential%20of%20Mamba/images/7b2a6cf0078a0eec458d9419691bc25fb87b7f6b0d0b031f3d787b7367b66c79.jpg)

![7fd6af15d10acee9b6e56d965fcaf2a1d43320bf5b2fd0e7c157df95471334d5.jpg](../iclr_results/1490_DeciMamba_%20Exploring%20the%20Length%20Extrapolation%20Potential%20of%20Mamba/images/7fd6af15d10acee9b6e56d965fcaf2a1d43320bf5b2fd0e7c157df95471334d5.jpg)

![bfa76d906aa363ae8058c4dfec66b59701b21d24aaf567a95d7dd3ce4310c6b2.jpg](../iclr_results/1490_DeciMamba_%20Exploring%20the%20Length%20Extrapolation%20Potential%20of%20Mamba/images/bfa76d906aa363ae8058c4dfec66b59701b21d24aaf567a95d7dd3ce4310c6b2.jpg)

![c05b6c98ada91dc1964fcc0bc892f7082c5d6b1e409ec654a22b7cccefb9b8d1.jpg](../iclr_results/1490_DeciMamba_%20Exploring%20the%20Length%20Extrapolation%20Potential%20of%20Mamba/images/c05b6c98ada91dc1964fcc0bc892f7082c5d6b1e409ec654a22b7cccefb9b8d1.jpg)

![ebed44fed332f96eda6917c99409b02048cbd58e5e4d86a23436ec1841078c59.jpg](../iclr_results/1490_DeciMamba_%20Exploring%20the%20Length%20Extrapolation%20Potential%20of%20Mamba/images/ebed44fed332f96eda6917c99409b02048cbd58e5e4d86a23436ec1841078c59.jpg)

![f0ed8db48c44549ad3bf0a7ae17ad3b7948249da437cf904ae5e853fc2aedcfe.jpg](../iclr_results/1490_DeciMamba_%20Exploring%20the%20Length%20Extrapolation%20Potential%20of%20Mamba/images/f0ed8db48c44549ad3bf0a7ae17ad3b7948249da437cf904ae5e853fc2aedcfe.jpg)

![fdec2662de45a67d91bf6560c856ff0ca9829de4a8c418e5ba9f63cb877e183a.jpg](../iclr_results/1490_DeciMamba_%20Exploring%20the%20Length%20Extrapolation%20Potential%20of%20Mamba/images/fdec2662de45a67d91bf6560c856ff0ca9829de4a8c418e5ba9f63cb877e183a.jpg)

### Tables

![0f52001ba376829f511de6b876a25c0269644eacd9d4daae4e32b2bc27b1c26e.jpg](../iclr_results/1490_DeciMamba_%20Exploring%20the%20Length%20Extrapolation%20Potential%20of%20Mamba/tables/0f52001ba376829f511de6b876a25c0269644eacd9d4daae4e32b2bc27b1c26e.jpg)

![26f775449f150cf1217d6860dabf823cad9e3f15b00b0b5d07735c9272fa97a7.jpg](../iclr_results/1490_DeciMamba_%20Exploring%20the%20Length%20Extrapolation%20Potential%20of%20Mamba/tables/26f775449f150cf1217d6860dabf823cad9e3f15b00b0b5d07735c9272fa97a7.jpg)

![297ccdb0158938bc85113a977491fc6289fbdf498ff7534b25ee7e3e37bf119d.jpg](../iclr_results/1490_DeciMamba_%20Exploring%20the%20Length%20Extrapolation%20Potential%20of%20Mamba/tables/297ccdb0158938bc85113a977491fc6289fbdf498ff7534b25ee7e3e37bf119d.jpg)

![3b5106d204e7e243af23452ea5badfe43a137eb7334f5ff85b9284946621c351.jpg](../iclr_results/1490_DeciMamba_%20Exploring%20the%20Length%20Extrapolation%20Potential%20of%20Mamba/tables/3b5106d204e7e243af23452ea5badfe43a137eb7334f5ff85b9284946621c351.jpg)

![5661920dd8a00db1841d5d8612ec0050411eddfe0c171cbb7dccb361230ec2c1.jpg](../iclr_results/1490_DeciMamba_%20Exploring%20the%20Length%20Extrapolation%20Potential%20of%20Mamba/tables/5661920dd8a00db1841d5d8612ec0050411eddfe0c171cbb7dccb361230ec2c1.jpg)

![a0fd490b3822437280d4cc8d9edd3dbec8c0b8f41e908c79ffdd99f6c6fadb9c.jpg](../iclr_results/1490_DeciMamba_%20Exploring%20the%20Length%20Extrapolation%20Potential%20of%20Mamba/tables/a0fd490b3822437280d4cc8d9edd3dbec8c0b8f41e908c79ffdd99f6c6fadb9c.jpg)

![caf208e16a55b16c9bde044aea526e83ac2d850babc5a6b4461313845c47821f.jpg](../iclr_results/1490_DeciMamba_%20Exploring%20the%20Length%20Extrapolation%20Potential%20of%20Mamba/tables/caf208e16a55b16c9bde044aea526e83ac2d850babc5a6b4461313845c47821f.jpg)

![e1de42d12b02521a8ea6fa886d306906d449ed4218826b725ab40a46f98eb88a.jpg](../iclr_results/1490_DeciMamba_%20Exploring%20the%20Length%20Extrapolation%20Potential%20of%20Mamba/tables/e1de42d12b02521a8ea6fa886d306906d449ed4218826b725ab40a46f98eb88a.jpg)

![ff4a88778b32843cb618ee5124cfa163e798081eb4afb16b381da1b4f74507c5.jpg](../iclr_results/1490_DeciMamba_%20Exploring%20the%20Length%20Extrapolation%20Potential%20of%20Mamba/tables/ff4a88778b32843cb618ee5124cfa163e798081eb4afb16b381da1b4f74507c5.jpg)

## LLaRA: Supercharging Robot Learning Data for Vision-Language Policy


### Images

![02637d634fb842a53334187e97f7334ab68b485806e91e295b52e606f7551e07.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/images/02637d634fb842a53334187e97f7334ab68b485806e91e295b52e606f7551e07.jpg)

![0b1786056e6b7870e3867bb672e4e30d0b9df96f4c06ad46a0a4b9b4b9a272c9.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/images/0b1786056e6b7870e3867bb672e4e30d0b9df96f4c06ad46a0a4b9b4b9a272c9.jpg)

![236c97fa4cdf96c4a4cb24cefa879de8cf54fefa5abd236e54f091086993cafa.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/images/236c97fa4cdf96c4a4cb24cefa879de8cf54fefa5abd236e54f091086993cafa.jpg)

![31f6041dbb53fd033fa3ecc73e01e67a8113b52cf25df9f4f0d863eaf7d5538f.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/images/31f6041dbb53fd033fa3ecc73e01e67a8113b52cf25df9f4f0d863eaf7d5538f.jpg)

![3378a60eee458452a2e9707dec889dea846ee55005ed14799617bd769f41e134.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/images/3378a60eee458452a2e9707dec889dea846ee55005ed14799617bd769f41e134.jpg)

![45301348e019d72542343d83190b1c30bac6dd30dbda7e008cb52d8c1480f57b.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/images/45301348e019d72542343d83190b1c30bac6dd30dbda7e008cb52d8c1480f57b.jpg)

![5209e4349dae9c8cc84f075b8707e38b2a791efe88c66eb3fc052e6d65caa7c6.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/images/5209e4349dae9c8cc84f075b8707e38b2a791efe88c66eb3fc052e6d65caa7c6.jpg)

![5a86200d6e0396e1ac88586097afe77023724b7466e84c4de63cc7c7c77948d6.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/images/5a86200d6e0396e1ac88586097afe77023724b7466e84c4de63cc7c7c77948d6.jpg)

![6072714b4f7dbe518dfb8e8758004baadcdd6d2635937c1568a29754758e6d92.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/images/6072714b4f7dbe518dfb8e8758004baadcdd6d2635937c1568a29754758e6d92.jpg)

![72e9de2018904894c5e9e970d7cc1896e1f8e4bbb3f1dc0c9a4c46dfe0325c1b.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/images/72e9de2018904894c5e9e970d7cc1896e1f8e4bbb3f1dc0c9a4c46dfe0325c1b.jpg)

![7ac3490feac56d81cd80ccc312545e84acc78357739f0c92232593e81b6aa577.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/images/7ac3490feac56d81cd80ccc312545e84acc78357739f0c92232593e81b6aa577.jpg)

![89f5e569bafd34b26890a24fb1ec724a6084c646acf597a95fbcd4f5a9bc244a.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/images/89f5e569bafd34b26890a24fb1ec724a6084c646acf597a95fbcd4f5a9bc244a.jpg)

![9cd0a010d713d9c310bc7e92f50b201df762a5e80c3b56c280b8212b3c46e207.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/images/9cd0a010d713d9c310bc7e92f50b201df762a5e80c3b56c280b8212b3c46e207.jpg)

![9f89a2d41bf53af78ef2a44ca32886e29dc290d0f11e4f81db5338025bb2548e.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/images/9f89a2d41bf53af78ef2a44ca32886e29dc290d0f11e4f81db5338025bb2548e.jpg)

![a0e0ab65ae47aed541392ae698268795aae858d419421a642c622766194f3391.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/images/a0e0ab65ae47aed541392ae698268795aae858d419421a642c622766194f3391.jpg)

![a859f9143513ff333f57fdb70c67402872a1ab71f579db7c43d7c37530995971.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/images/a859f9143513ff333f57fdb70c67402872a1ab71f579db7c43d7c37530995971.jpg)

![c2940229c1186c0692e52957e6f83eb36863f638da6063c0318943bb103fe855.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/images/c2940229c1186c0692e52957e6f83eb36863f638da6063c0318943bb103fe855.jpg)

![d74389b9d5ab929600ebc373452c8123f7c73e364d546917c5d1d984391d0459.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/images/d74389b9d5ab929600ebc373452c8123f7c73e364d546917c5d1d984391d0459.jpg)

### Tables

![00009ee42da05110b6c8a91f5738272428b32c76aa947d733ff03594873df7fb.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/00009ee42da05110b6c8a91f5738272428b32c76aa947d733ff03594873df7fb.jpg)

![0b93e52acda466eee6a036e53e80340db21e0e7f140025e96f9559b605b3386b.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/0b93e52acda466eee6a036e53e80340db21e0e7f140025e96f9559b605b3386b.jpg)

![1a7e0ba6982ca681f52631c850fb0220e3ceddc2d76bba996e864877b9808b4d.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/1a7e0ba6982ca681f52631c850fb0220e3ceddc2d76bba996e864877b9808b4d.jpg)

![2a575e8d2cef1410bf246827b89baa66f93f09adb0571334004dc57673befd00.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/2a575e8d2cef1410bf246827b89baa66f93f09adb0571334004dc57673befd00.jpg)

![2f304c6a162847751e5a1056121d9fd4981e23df9d1d4f3abc6379443cb0b07a.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/2f304c6a162847751e5a1056121d9fd4981e23df9d1d4f3abc6379443cb0b07a.jpg)

![329bf12ac4ec6c5c4962de450834098ae64e64df26f950e21a1b56269fac28ec.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/329bf12ac4ec6c5c4962de450834098ae64e64df26f950e21a1b56269fac28ec.jpg)

![50e9000e850af5c29c10e1a043198a8093425c9914326ceab0c62fad0b816121.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/50e9000e850af5c29c10e1a043198a8093425c9914326ceab0c62fad0b816121.jpg)

![5a3d1acd180a5e1c77e638bab2238d6be054800c98ed372cbc637aa26636291c.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/5a3d1acd180a5e1c77e638bab2238d6be054800c98ed372cbc637aa26636291c.jpg)

![61a245d65353b2304fdd3b87b25c958a74ccb957a70dcb5a61a352e1dd055f5a.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/61a245d65353b2304fdd3b87b25c958a74ccb957a70dcb5a61a352e1dd055f5a.jpg)

![6b27cbb65abbb6b4a934c031a68124b06e51422137674e44d12e17a6f4cbc5c6.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/6b27cbb65abbb6b4a934c031a68124b06e51422137674e44d12e17a6f4cbc5c6.jpg)

![6b3e5d536c3ca26ffde9b15eb36681c1edd598422ce7c731d3eb3feb4d282dd8.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/6b3e5d536c3ca26ffde9b15eb36681c1edd598422ce7c731d3eb3feb4d282dd8.jpg)

![6e502611fa0c5b45c48369bb4a4289f767fa1a58a84c10d9006262a64c8376c0.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/6e502611fa0c5b45c48369bb4a4289f767fa1a58a84c10d9006262a64c8376c0.jpg)

![6fd73e137399af627f5129d81592cef5012faa2eed53bb39e0421caecb281f98.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/6fd73e137399af627f5129d81592cef5012faa2eed53bb39e0421caecb281f98.jpg)

![70e251b76678292276ed30ebd8ea0393e826d107131a2766312a8d321b45620b.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/70e251b76678292276ed30ebd8ea0393e826d107131a2766312a8d321b45620b.jpg)

![9bfed2fa11082fac2f4ac2ecb75b82525cdc5767572c49f2ba1602126228ea5f.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/9bfed2fa11082fac2f4ac2ecb75b82525cdc5767572c49f2ba1602126228ea5f.jpg)

![a390ee3551bc5d3a07db9843cff8ccd4111ceb3a563360dac9c930036f090909.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/a390ee3551bc5d3a07db9843cff8ccd4111ceb3a563360dac9c930036f090909.jpg)

![a7b96f3763237cff88fe40e8f0143cbfaec88af88b2e2caa0437b55a242149df.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/a7b96f3763237cff88fe40e8f0143cbfaec88af88b2e2caa0437b55a242149df.jpg)

![ae88c54f2b06bedadfdaeeb78c318304e6e9f11226191a74e3fa23b356063d6a.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/ae88c54f2b06bedadfdaeeb78c318304e6e9f11226191a74e3fa23b356063d6a.jpg)

![aea4984ea36c3f88e40dfe1997bd8e953b8621f55d17b44707fe463c5a03d0d0.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/aea4984ea36c3f88e40dfe1997bd8e953b8621f55d17b44707fe463c5a03d0d0.jpg)

![b355a5ce3334e3392663b0f6cc4b79ecb128b79448a2926638070e9f417c23d0.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/b355a5ce3334e3392663b0f6cc4b79ecb128b79448a2926638070e9f417c23d0.jpg)

![b9bc6df5f2c0cd5539633a791b9b9097d3fd636fc8efe2c7b2aa42254883bc46.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/b9bc6df5f2c0cd5539633a791b9b9097d3fd636fc8efe2c7b2aa42254883bc46.jpg)

![c54b4b38ca284600e943f8810ea83e9fa72f400716571bfcfbcd465076051237.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/c54b4b38ca284600e943f8810ea83e9fa72f400716571bfcfbcd465076051237.jpg)

![cc8425a9b653f9684a3ee5eff00c33225c191fe320e6c1cc6eb763748a4aa7cf.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/cc8425a9b653f9684a3ee5eff00c33225c191fe320e6c1cc6eb763748a4aa7cf.jpg)

![d1b55088708c15275d22a752609c7b5b538b39066b0a42bc3e65d0ad34f67610.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/d1b55088708c15275d22a752609c7b5b538b39066b0a42bc3e65d0ad34f67610.jpg)

![d55f3661d83ed590160b143d2b4ecbba9bb82510d97dead122901a1478e6f29b.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/d55f3661d83ed590160b143d2b4ecbba9bb82510d97dead122901a1478e6f29b.jpg)

![e15ec850ab3345961ca8186e5b2aeb62fa883566a0cf36c4e9921f19efb0250e.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/e15ec850ab3345961ca8186e5b2aeb62fa883566a0cf36c4e9921f19efb0250e.jpg)

![e42c8c0ab30d5d0b99b4fb804c63a5502e5907d728fbe5a67a31435f6fa29fdf.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/e42c8c0ab30d5d0b99b4fb804c63a5502e5907d728fbe5a67a31435f6fa29fdf.jpg)

![fb7da552293fc5a0493aeda288a17773fad211f34422968c428606f6d65d1114.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/fb7da552293fc5a0493aeda288a17773fad211f34422968c428606f6d65d1114.jpg)

![fdcbcd3460eeda3c98b0a7bfa7fb94dc76df34500044c8c624afde64bc7758fc.jpg](../iclr_results/1491_LLaRA_%20Supercharging%20Robot%20Learning%20Data%20for%20Vision-Language%20Policy/tables/fdcbcd3460eeda3c98b0a7bfa7fb94dc76df34500044c8c624afde64bc7758fc.jpg)

## Can We Talk Models Into Seeing the World Differently?


### Images

![25df196a58956bee2b0b05c7f55491d19a47fb0ee3622ae4e00b0bdfd5bfce9d.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/images/25df196a58956bee2b0b05c7f55491d19a47fb0ee3622ae4e00b0bdfd5bfce9d.jpg)

![3f235b7efb4cb2eb4d4bc2288ad956169b0d688710f506b541c27eeb1071e38b.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/images/3f235b7efb4cb2eb4d4bc2288ad956169b0d688710f506b541c27eeb1071e38b.jpg)

![537f323a210a4d7b7fcaf3f37297e3b3e356dfc4deea4512ab6b648f973994b0.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/images/537f323a210a4d7b7fcaf3f37297e3b3e356dfc4deea4512ab6b648f973994b0.jpg)

![5dd9c74e648ad03af004425f0282790a8df64048e483e299ce7e60dfc3a04c53.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/images/5dd9c74e648ad03af004425f0282790a8df64048e483e299ce7e60dfc3a04c53.jpg)

![6deb7c08a90e66d85a2c7e2fc853e4ec3c5160a45116604ad92df84003236829.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/images/6deb7c08a90e66d85a2c7e2fc853e4ec3c5160a45116604ad92df84003236829.jpg)

![75a820b613cb109ea37eabcf0efe46469d8db04ae7e73f17596c0a4a1e1bfa64.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/images/75a820b613cb109ea37eabcf0efe46469d8db04ae7e73f17596c0a4a1e1bfa64.jpg)

![80aa5796d21296681d9ac75b4f36741548abf3b1611a9e542c11ce33c989216b.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/images/80aa5796d21296681d9ac75b4f36741548abf3b1611a9e542c11ce33c989216b.jpg)

![82e9ee4e358d915a765554f56e959f76c6642ae4c2cf434ae2dc5c37ad4c76de.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/images/82e9ee4e358d915a765554f56e959f76c6642ae4c2cf434ae2dc5c37ad4c76de.jpg)

![959382a383166bb0dd6922838657fcd490178943d53592d6c8a8d9c77ee15373.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/images/959382a383166bb0dd6922838657fcd490178943d53592d6c8a8d9c77ee15373.jpg)

![cf33f4a011d1ef9e03a94922013b6f7f27f80492aa3e08a6676964ca28fc86e9.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/images/cf33f4a011d1ef9e03a94922013b6f7f27f80492aa3e08a6676964ca28fc86e9.jpg)

![dc1ac84b60f108280f8e7ef4382806b266ec08ee2c27a5b8f598b2ced3e01767.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/images/dc1ac84b60f108280f8e7ef4382806b266ec08ee2c27a5b8f598b2ced3e01767.jpg)

![f304ed3eddb0600d4654da1e10c5208f44f35dc71e33a08714397e9c10c4cfa5.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/images/f304ed3eddb0600d4654da1e10c5208f44f35dc71e33a08714397e9c10c4cfa5.jpg)

![fbb0a53cfdd5ecbff1bcbea03e8338158e4f62f2b71b36e4a04d02e9b0819dc8.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/images/fbb0a53cfdd5ecbff1bcbea03e8338158e4f62f2b71b36e4a04d02e9b0819dc8.jpg)

### Tables

![1704fbb0af04b2c82abd9192cb80bf15b90ceee2d3600a16168361d00d70490b.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/tables/1704fbb0af04b2c82abd9192cb80bf15b90ceee2d3600a16168361d00d70490b.jpg)

![1c60b87e0257ea795d172e0483d37b5fd68fb7039c585c704f0357a90b963bc0.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/tables/1c60b87e0257ea795d172e0483d37b5fd68fb7039c585c704f0357a90b963bc0.jpg)

![1f87be2e6a46aa51628c4409bdf49960ef3ce28a025cd72b3f485c09ac44e411.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/tables/1f87be2e6a46aa51628c4409bdf49960ef3ce28a025cd72b3f485c09ac44e411.jpg)

![294e2a7f6a9fb773993f67250c0b8dff735414bfb02fb8876d73dff286c4aca0.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/tables/294e2a7f6a9fb773993f67250c0b8dff735414bfb02fb8876d73dff286c4aca0.jpg)

![2a6c425a0e1bc7bc3e3255931bd093af0d1588d78fa5b29740e81812579bc51c.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/tables/2a6c425a0e1bc7bc3e3255931bd093af0d1588d78fa5b29740e81812579bc51c.jpg)

![5ab58923991c398852c556182670ba6a49aec5fe8f759eeef57bf74bf378a0e0.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/tables/5ab58923991c398852c556182670ba6a49aec5fe8f759eeef57bf74bf378a0e0.jpg)

![67a21daef7aaada761585fc84abbac0a1b673c744e2606c9fe11f4b0bc93bd6e.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/tables/67a21daef7aaada761585fc84abbac0a1b673c744e2606c9fe11f4b0bc93bd6e.jpg)

![ba7656f63a7cdab37ea275e206d376d1d0a1bafa88d568efafe08f34434ae8ca.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/tables/ba7656f63a7cdab37ea275e206d376d1d0a1bafa88d568efafe08f34434ae8ca.jpg)

![bb585f6fa69ae25e68f20bba1aa805a91d8c081499180d50b2d8d6c9527509d8.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/tables/bb585f6fa69ae25e68f20bba1aa805a91d8c081499180d50b2d8d6c9527509d8.jpg)

![f1dbd50da2c0d3c1badad9d1787b21bb9159babcfbb2b52e75041437e2c765b3.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/tables/f1dbd50da2c0d3c1badad9d1787b21bb9159babcfbb2b52e75041437e2c765b3.jpg)

![ff437f62dac4efe625e51995aa962d4f840d5e09c4f606b7a75fb982b6eb8dd8.jpg](../iclr_results/1492_Can%20We%20Talk%20Models%20Into%20Seeing%20the%20World%20Differently_/tables/ff437f62dac4efe625e51995aa962d4f840d5e09c4f606b7a75fb982b6eb8dd8.jpg)

## ClassDiffusion: More Aligned Personalization Tuning with Explicit Class Guidance


### Images

![06cf70e4839a4a6989c2ffc42180b6451200d9564e3fb211e2a9b0a5f344b297.jpg](../iclr_results/1493_ClassDiffusion_%20More%20Aligned%20Personalization%20Tuning%20with%20Explicit%20Class%20Guidance/images/06cf70e4839a4a6989c2ffc42180b6451200d9564e3fb211e2a9b0a5f344b297.jpg)

![0b52f74b4fcd13c6b2b146931e29c272e14f6ba4c96b1186de81b3c865c11d01.jpg](../iclr_results/1493_ClassDiffusion_%20More%20Aligned%20Personalization%20Tuning%20with%20Explicit%20Class%20Guidance/images/0b52f74b4fcd13c6b2b146931e29c272e14f6ba4c96b1186de81b3c865c11d01.jpg)

![0e776400ac8817b3761fc2fbc772c15caaa36af0b3e7ea5e9ff3f97e6915d3d5.jpg](../iclr_results/1493_ClassDiffusion_%20More%20Aligned%20Personalization%20Tuning%20with%20Explicit%20Class%20Guidance/images/0e776400ac8817b3761fc2fbc772c15caaa36af0b3e7ea5e9ff3f97e6915d3d5.jpg)

![51b1b2c07210322f7da80bd1d5240b7c6ae2e1068598573dce0f3ffbf3a2b7de.jpg](../iclr_results/1493_ClassDiffusion_%20More%20Aligned%20Personalization%20Tuning%20with%20Explicit%20Class%20Guidance/images/51b1b2c07210322f7da80bd1d5240b7c6ae2e1068598573dce0f3ffbf3a2b7de.jpg)

![54943739288dc312e32d09dcec395a087fdac6e7ac6bd6dfab86cfdb4d5c2eb7.jpg](../iclr_results/1493_ClassDiffusion_%20More%20Aligned%20Personalization%20Tuning%20with%20Explicit%20Class%20Guidance/images/54943739288dc312e32d09dcec395a087fdac6e7ac6bd6dfab86cfdb4d5c2eb7.jpg)

![5ec74038b026a5550cef1410930f38fcfa1647105df2bbae82fd73b3c339ea6c.jpg](../iclr_results/1493_ClassDiffusion_%20More%20Aligned%20Personalization%20Tuning%20with%20Explicit%20Class%20Guidance/images/5ec74038b026a5550cef1410930f38fcfa1647105df2bbae82fd73b3c339ea6c.jpg)

![5f6fbccb352fc6e821063f218d21e6f8d4279f3249e81759c6d52b381b1bb8e1.jpg](../iclr_results/1493_ClassDiffusion_%20More%20Aligned%20Personalization%20Tuning%20with%20Explicit%20Class%20Guidance/images/5f6fbccb352fc6e821063f218d21e6f8d4279f3249e81759c6d52b381b1bb8e1.jpg)

![743ebb67f76cd9499350e2b09e5400e85f29bc16409d4638bb06578803f578c4.jpg](../iclr_results/1493_ClassDiffusion_%20More%20Aligned%20Personalization%20Tuning%20with%20Explicit%20Class%20Guidance/images/743ebb67f76cd9499350e2b09e5400e85f29bc16409d4638bb06578803f578c4.jpg)

![785b673ae8233712a26dbbf15a5b2653e602153d8d569480221a7895518a11d7.jpg](../iclr_results/1493_ClassDiffusion_%20More%20Aligned%20Personalization%20Tuning%20with%20Explicit%20Class%20Guidance/images/785b673ae8233712a26dbbf15a5b2653e602153d8d569480221a7895518a11d7.jpg)

![81f11152990f771755b2abdb70888ec4ab2acb33c58978ae2699c30fbb7b6c7b.jpg](../iclr_results/1493_ClassDiffusion_%20More%20Aligned%20Personalization%20Tuning%20with%20Explicit%20Class%20Guidance/images/81f11152990f771755b2abdb70888ec4ab2acb33c58978ae2699c30fbb7b6c7b.jpg)

![92bcc29dee6b0604362da47a8f1e2cbbf26f43d052f03aa6ac8494ca68cf2bec.jpg](../iclr_results/1493_ClassDiffusion_%20More%20Aligned%20Personalization%20Tuning%20with%20Explicit%20Class%20Guidance/images/92bcc29dee6b0604362da47a8f1e2cbbf26f43d052f03aa6ac8494ca68cf2bec.jpg)

![a27d0d410176554d653b75eefa7dd97f410b066416c7caa04fa7f3e539f7e540.jpg](../iclr_results/1493_ClassDiffusion_%20More%20Aligned%20Personalization%20Tuning%20with%20Explicit%20Class%20Guidance/images/a27d0d410176554d653b75eefa7dd97f410b066416c7caa04fa7f3e539f7e540.jpg)

![a3e4289bd58abd97ee2dcdc0af6f5828946f51b088872fc7503dfefc50af4e5c.jpg](../iclr_results/1493_ClassDiffusion_%20More%20Aligned%20Personalization%20Tuning%20with%20Explicit%20Class%20Guidance/images/a3e4289bd58abd97ee2dcdc0af6f5828946f51b088872fc7503dfefc50af4e5c.jpg)

![bc79c9970ec4e8c0d188153d969ac8c48f80e8d85272106d86d4aaf015eef906.jpg](../iclr_results/1493_ClassDiffusion_%20More%20Aligned%20Personalization%20Tuning%20with%20Explicit%20Class%20Guidance/images/bc79c9970ec4e8c0d188153d969ac8c48f80e8d85272106d86d4aaf015eef906.jpg)

![cace14eef8a87e824fffdee7ccbc24cc9eb92a5db3e68921b551fb3c23a35a2e.jpg](../iclr_results/1493_ClassDiffusion_%20More%20Aligned%20Personalization%20Tuning%20with%20Explicit%20Class%20Guidance/images/cace14eef8a87e824fffdee7ccbc24cc9eb92a5db3e68921b551fb3c23a35a2e.jpg)

![eadbd05a9c8929ac2eab9481dba7130d58b5cd4e4dff617d6e52f4d102dd4e86.jpg](../iclr_results/1493_ClassDiffusion_%20More%20Aligned%20Personalization%20Tuning%20with%20Explicit%20Class%20Guidance/images/eadbd05a9c8929ac2eab9481dba7130d58b5cd4e4dff617d6e52f4d102dd4e86.jpg)

### Tables

![3df835458ae7f1c41cbcfc072d9ffbce74ec3c622f7912bfd176503651a4425f.jpg](../iclr_results/1493_ClassDiffusion_%20More%20Aligned%20Personalization%20Tuning%20with%20Explicit%20Class%20Guidance/tables/3df835458ae7f1c41cbcfc072d9ffbce74ec3c622f7912bfd176503651a4425f.jpg)

![4a5f54d1d00c9a6fb3578833f23abe3a00bd88caf68535c9a2bd316319aa183d.jpg](../iclr_results/1493_ClassDiffusion_%20More%20Aligned%20Personalization%20Tuning%20with%20Explicit%20Class%20Guidance/tables/4a5f54d1d00c9a6fb3578833f23abe3a00bd88caf68535c9a2bd316319aa183d.jpg)

![4f5a6d182ba60c09b945e14fa0e6a9c9ffbf2179e3ccf39c13c5e0a77748118b.jpg](../iclr_results/1493_ClassDiffusion_%20More%20Aligned%20Personalization%20Tuning%20with%20Explicit%20Class%20Guidance/tables/4f5a6d182ba60c09b945e14fa0e6a9c9ffbf2179e3ccf39c13c5e0a77748118b.jpg)

![c4581094cc03a0e062dad1504328d689d2e7700c4ff3b0ddb84871a1d1a79295.jpg](../iclr_results/1493_ClassDiffusion_%20More%20Aligned%20Personalization%20Tuning%20with%20Explicit%20Class%20Guidance/tables/c4581094cc03a0e062dad1504328d689d2e7700c4ff3b0ddb84871a1d1a79295.jpg)

## Utility-Directed Conformal Prediction: A Decision-Aware Framework for Actionable Uncertainty Quantification


### Images

![349b1be05edfca1b382ffc7506e319ce1e137ec28efe0c0c6ef1aa040c6f8057.jpg](../iclr_results/1494_Utility-Directed%20Conformal%20Prediction_%20A%20Decision-Aware%20Framework%20for%20Actionable%20Uncertainty%20Quantif/images/349b1be05edfca1b382ffc7506e319ce1e137ec28efe0c0c6ef1aa040c6f8057.jpg)

![5753cf33defddc5b08277cc0e12ca60a9e2d3f995a7c316dcb1861980fafc00a.jpg](../iclr_results/1494_Utility-Directed%20Conformal%20Prediction_%20A%20Decision-Aware%20Framework%20for%20Actionable%20Uncertainty%20Quantif/images/5753cf33defddc5b08277cc0e12ca60a9e2d3f995a7c316dcb1861980fafc00a.jpg)

![82b91849cbe525f956026b524e6febad907362d8d25a1ec18a27328fca621f2a.jpg](../iclr_results/1494_Utility-Directed%20Conformal%20Prediction_%20A%20Decision-Aware%20Framework%20for%20Actionable%20Uncertainty%20Quantif/images/82b91849cbe525f956026b524e6febad907362d8d25a1ec18a27328fca621f2a.jpg)

![83f1d37cb4ce4eedbee86a83629f6411634cdcc245f0676b6583cd1efc9e6a1c.jpg](../iclr_results/1494_Utility-Directed%20Conformal%20Prediction_%20A%20Decision-Aware%20Framework%20for%20Actionable%20Uncertainty%20Quantif/images/83f1d37cb4ce4eedbee86a83629f6411634cdcc245f0676b6583cd1efc9e6a1c.jpg)

![8d37b2a1c99b969abf5999ac138f93b10ec30879c3ef1af751e13cdd339051cf.jpg](../iclr_results/1494_Utility-Directed%20Conformal%20Prediction_%20A%20Decision-Aware%20Framework%20for%20Actionable%20Uncertainty%20Quantif/images/8d37b2a1c99b969abf5999ac138f93b10ec30879c3ef1af751e13cdd339051cf.jpg)

![acd9a87622758c1326c25890c288efbe75b49d1a5d00df6c76c9dd67d8400b31.jpg](../iclr_results/1494_Utility-Directed%20Conformal%20Prediction_%20A%20Decision-Aware%20Framework%20for%20Actionable%20Uncertainty%20Quantif/images/acd9a87622758c1326c25890c288efbe75b49d1a5d00df6c76c9dd67d8400b31.jpg)

![c80eafd447e45d6e7c7e8a453ba40cdf6569ccf4a21964d90cf8f451b2a4c53f.jpg](../iclr_results/1494_Utility-Directed%20Conformal%20Prediction_%20A%20Decision-Aware%20Framework%20for%20Actionable%20Uncertainty%20Quantif/images/c80eafd447e45d6e7c7e8a453ba40cdf6569ccf4a21964d90cf8f451b2a4c53f.jpg)

![e98ec75d64f0d1e6d28ed2f294a28f29f6d1d9a82a2d2e87a29132e17cd3bb69.jpg](../iclr_results/1494_Utility-Directed%20Conformal%20Prediction_%20A%20Decision-Aware%20Framework%20for%20Actionable%20Uncertainty%20Quantif/images/e98ec75d64f0d1e6d28ed2f294a28f29f6d1d9a82a2d2e87a29132e17cd3bb69.jpg)

### Tables

![303cf991eb67f07acac24d1f40874aaca1638410fc769587463a66dfe17db4da.jpg](../iclr_results/1494_Utility-Directed%20Conformal%20Prediction_%20A%20Decision-Aware%20Framework%20for%20Actionable%20Uncertainty%20Quantif/tables/303cf991eb67f07acac24d1f40874aaca1638410fc769587463a66dfe17db4da.jpg)

![4c7510ba0a135be9fd2601d809ac5fd7b9ddf4ebcee5d71147c6782ef2da2560.jpg](../iclr_results/1494_Utility-Directed%20Conformal%20Prediction_%20A%20Decision-Aware%20Framework%20for%20Actionable%20Uncertainty%20Quantif/tables/4c7510ba0a135be9fd2601d809ac5fd7b9ddf4ebcee5d71147c6782ef2da2560.jpg)

![57b705f91924b6e22fb2dfb227805904a1375beb313e1f7022d1ce5ade6b3147.jpg](../iclr_results/1494_Utility-Directed%20Conformal%20Prediction_%20A%20Decision-Aware%20Framework%20for%20Actionable%20Uncertainty%20Quantif/tables/57b705f91924b6e22fb2dfb227805904a1375beb313e1f7022d1ce5ade6b3147.jpg)

![7b8e71deb04b06551fa86684ec9d802912dbea341f0c04d7aeb3ad60ae3b59b8.jpg](../iclr_results/1494_Utility-Directed%20Conformal%20Prediction_%20A%20Decision-Aware%20Framework%20for%20Actionable%20Uncertainty%20Quantif/tables/7b8e71deb04b06551fa86684ec9d802912dbea341f0c04d7aeb3ad60ae3b59b8.jpg)

![813a20fdeb2d13ed8636ba2e18d741e1a69d04179c139635be1e8ce24475b630.jpg](../iclr_results/1494_Utility-Directed%20Conformal%20Prediction_%20A%20Decision-Aware%20Framework%20for%20Actionable%20Uncertainty%20Quantif/tables/813a20fdeb2d13ed8636ba2e18d741e1a69d04179c139635be1e8ce24475b630.jpg)

![813da6f324d4658d48ea78379f999ecde0431e53b35bf853353f29f113a94662.jpg](../iclr_results/1494_Utility-Directed%20Conformal%20Prediction_%20A%20Decision-Aware%20Framework%20for%20Actionable%20Uncertainty%20Quantif/tables/813da6f324d4658d48ea78379f999ecde0431e53b35bf853353f29f113a94662.jpg)

![88c361ec73c3e81e1713dfce67e8c653a558c80c2ce1b97a235cc158c070d3b8.jpg](../iclr_results/1494_Utility-Directed%20Conformal%20Prediction_%20A%20Decision-Aware%20Framework%20for%20Actionable%20Uncertainty%20Quantif/tables/88c361ec73c3e81e1713dfce67e8c653a558c80c2ce1b97a235cc158c070d3b8.jpg)

![9e8d6719ddd5e1b9436ebb87e5382d106547f1aae9a5975678632370e284b64a.jpg](../iclr_results/1494_Utility-Directed%20Conformal%20Prediction_%20A%20Decision-Aware%20Framework%20for%20Actionable%20Uncertainty%20Quantif/tables/9e8d6719ddd5e1b9436ebb87e5382d106547f1aae9a5975678632370e284b64a.jpg)

![a6569ae4722cdb294fc23ad8dcd7e3386c735d47b3b8e3eab030ceea3e5b8684.jpg](../iclr_results/1494_Utility-Directed%20Conformal%20Prediction_%20A%20Decision-Aware%20Framework%20for%20Actionable%20Uncertainty%20Quantif/tables/a6569ae4722cdb294fc23ad8dcd7e3386c735d47b3b8e3eab030ceea3e5b8684.jpg)

![c4c4224eb61baf57a055cc0a3754fdc4c615beaa4bd0a735e7ca566a1ce6f0f2.jpg](../iclr_results/1494_Utility-Directed%20Conformal%20Prediction_%20A%20Decision-Aware%20Framework%20for%20Actionable%20Uncertainty%20Quantif/tables/c4c4224eb61baf57a055cc0a3754fdc4c615beaa4bd0a735e7ca566a1ce6f0f2.jpg)

![c8d35bab8eec3b4dd243de7fdb2010d8468ffcfab02f8057dfa5afbddf739168.jpg](../iclr_results/1494_Utility-Directed%20Conformal%20Prediction_%20A%20Decision-Aware%20Framework%20for%20Actionable%20Uncertainty%20Quantif/tables/c8d35bab8eec3b4dd243de7fdb2010d8468ffcfab02f8057dfa5afbddf739168.jpg)

![f23291869dc5e954565892a794b0660a62de119663b31d81dfb3986657aa5a61.jpg](../iclr_results/1494_Utility-Directed%20Conformal%20Prediction_%20A%20Decision-Aware%20Framework%20for%20Actionable%20Uncertainty%20Quantif/tables/f23291869dc5e954565892a794b0660a62de119663b31d81dfb3986657aa5a61.jpg)

![f3d5d7601d28ce7c3f408705c56af5465d353dc9f2e44a36c60d74c62cda3289.jpg](../iclr_results/1494_Utility-Directed%20Conformal%20Prediction_%20A%20Decision-Aware%20Framework%20for%20Actionable%20Uncertainty%20Quantif/tables/f3d5d7601d28ce7c3f408705c56af5465d353dc9f2e44a36c60d74c62cda3289.jpg)

![f77e1b0c6a8138c28e33847f949827a216db00d79faeb7463fbfdcadbf5a07d5.jpg](../iclr_results/1494_Utility-Directed%20Conformal%20Prediction_%20A%20Decision-Aware%20Framework%20for%20Actionable%20Uncertainty%20Quantif/tables/f77e1b0c6a8138c28e33847f949827a216db00d79faeb7463fbfdcadbf5a07d5.jpg)

## Fast Summation of Radial Kernels via QMC Slicing


### Images

![4922f4c072b27d7527d1eb69acd4bf2c4e90a46668348fdcd99219131da5c725.jpg](../iclr_results/1495_Fast%20Summation%20of%20Radial%20Kernels%20via%20QMC%20Slicing/images/4922f4c072b27d7527d1eb69acd4bf2c4e90a46668348fdcd99219131da5c725.jpg)

![4a96b5b989fa0ce506da36105530ba8a0aac31f2caff69ba1e8f0335ebfec388.jpg](../iclr_results/1495_Fast%20Summation%20of%20Radial%20Kernels%20via%20QMC%20Slicing/images/4a96b5b989fa0ce506da36105530ba8a0aac31f2caff69ba1e8f0335ebfec388.jpg)

![54a8c69704d54c13dd10e933a06f6b068a29d47f005d23e99b41be5e6a40535b.jpg](../iclr_results/1495_Fast%20Summation%20of%20Radial%20Kernels%20via%20QMC%20Slicing/images/54a8c69704d54c13dd10e933a06f6b068a29d47f005d23e99b41be5e6a40535b.jpg)

![833591e5e9b55db358c775420211e2ee1eb5d6d13cf4980c103e1305f491cb96.jpg](../iclr_results/1495_Fast%20Summation%20of%20Radial%20Kernels%20via%20QMC%20Slicing/images/833591e5e9b55db358c775420211e2ee1eb5d6d13cf4980c103e1305f491cb96.jpg)

![91ccb9e4f59ed4008aa540d56f4ae8aa2b0ed734b4c4798b491e0a3e1b94d44f.jpg](../iclr_results/1495_Fast%20Summation%20of%20Radial%20Kernels%20via%20QMC%20Slicing/images/91ccb9e4f59ed4008aa540d56f4ae8aa2b0ed734b4c4798b491e0a3e1b94d44f.jpg)

![9d39a5f1743988f900e8356294395ad58ce1032ff89e59a633ec5883ab546e2b.jpg](../iclr_results/1495_Fast%20Summation%20of%20Radial%20Kernels%20via%20QMC%20Slicing/images/9d39a5f1743988f900e8356294395ad58ce1032ff89e59a633ec5883ab546e2b.jpg)

![a623d72d87e3f1cf1a9f94ef7e6f325f14ee95a0b89ecb61e27722251f583e69.jpg](../iclr_results/1495_Fast%20Summation%20of%20Radial%20Kernels%20via%20QMC%20Slicing/images/a623d72d87e3f1cf1a9f94ef7e6f325f14ee95a0b89ecb61e27722251f583e69.jpg)

![d1667a8223353227346ce1ed915bad28a976cd1ab7c8e210aae92be3df117e8c.jpg](../iclr_results/1495_Fast%20Summation%20of%20Radial%20Kernels%20via%20QMC%20Slicing/images/d1667a8223353227346ce1ed915bad28a976cd1ab7c8e210aae92be3df117e8c.jpg)

![dad51cf75a5ca6bad8f2077fa0106db185cd26693f5283631ab4405a460691b2.jpg](../iclr_results/1495_Fast%20Summation%20of%20Radial%20Kernels%20via%20QMC%20Slicing/images/dad51cf75a5ca6bad8f2077fa0106db185cd26693f5283631ab4405a460691b2.jpg)

![e419b8eb8d7929d37060d22c377bb4d84d9628e1804481bdf9575305485498b8.jpg](../iclr_results/1495_Fast%20Summation%20of%20Radial%20Kernels%20via%20QMC%20Slicing/images/e419b8eb8d7929d37060d22c377bb4d84d9628e1804481bdf9575305485498b8.jpg)

![f422f532840a500f704423d1cac4263ad1c3a60fa0b9e6401c8ceec4b404b049.jpg](../iclr_results/1495_Fast%20Summation%20of%20Radial%20Kernels%20via%20QMC%20Slicing/images/f422f532840a500f704423d1cac4263ad1c3a60fa0b9e6401c8ceec4b404b049.jpg)

### Tables

![1a5a98c0e89df0c000db096ea288180077071ea2726312e54ca7b549c8a9c587.jpg](../iclr_results/1495_Fast%20Summation%20of%20Radial%20Kernels%20via%20QMC%20Slicing/tables/1a5a98c0e89df0c000db096ea288180077071ea2726312e54ca7b549c8a9c587.jpg)

![4700c3f09bc5af8965c0c35d6f9f8dc172a2fffc00b7672802df9b8837aebdc9.jpg](../iclr_results/1495_Fast%20Summation%20of%20Radial%20Kernels%20via%20QMC%20Slicing/tables/4700c3f09bc5af8965c0c35d6f9f8dc172a2fffc00b7672802df9b8837aebdc9.jpg)

![4cf9e353ead1e384cb71df7597ba58541d71a1750761a4437bfdbde9740a1f7a.jpg](../iclr_results/1495_Fast%20Summation%20of%20Radial%20Kernels%20via%20QMC%20Slicing/tables/4cf9e353ead1e384cb71df7597ba58541d71a1750761a4437bfdbde9740a1f7a.jpg)

![83511d2a60541430a40a764bb0b7dafbdbba87588f20f16859b7cf47d46a888c.jpg](../iclr_results/1495_Fast%20Summation%20of%20Radial%20Kernels%20via%20QMC%20Slicing/tables/83511d2a60541430a40a764bb0b7dafbdbba87588f20f16859b7cf47d46a888c.jpg)

![a034d2f705e92f9531057a92992306509932b61400d20b7c8e7816f5e1993258.jpg](../iclr_results/1495_Fast%20Summation%20of%20Radial%20Kernels%20via%20QMC%20Slicing/tables/a034d2f705e92f9531057a92992306509932b61400d20b7c8e7816f5e1993258.jpg)

![bd9a97f0dfbc61d586571b037dfdd49295fa90ef9be5c0a7dc68b09db9f9e086.jpg](../iclr_results/1495_Fast%20Summation%20of%20Radial%20Kernels%20via%20QMC%20Slicing/tables/bd9a97f0dfbc61d586571b037dfdd49295fa90ef9be5c0a7dc68b09db9f9e086.jpg)

![dc9e9cadc30581dd49a634a05bfc12902711d035e0412bff4bf94cd7c3d9c3a8.jpg](../iclr_results/1495_Fast%20Summation%20of%20Radial%20Kernels%20via%20QMC%20Slicing/tables/dc9e9cadc30581dd49a634a05bfc12902711d035e0412bff4bf94cd7c3d9c3a8.jpg)

![e86f073057e7671809a8c0c3b4901bed9175d703b752bedf0a872437e26a2295.jpg](../iclr_results/1495_Fast%20Summation%20of%20Radial%20Kernels%20via%20QMC%20Slicing/tables/e86f073057e7671809a8c0c3b4901bed9175d703b752bedf0a872437e26a2295.jpg)

![f3aa121fc7d863caf2ec2d6615c5e6ce013b89c390b3a72a31906941d37353bf.jpg](../iclr_results/1495_Fast%20Summation%20of%20Radial%20Kernels%20via%20QMC%20Slicing/tables/f3aa121fc7d863caf2ec2d6615c5e6ce013b89c390b3a72a31906941d37353bf.jpg)

![f699ca107cda194db03950210e4216f480760e441a5ba7743ddca23bfa467d4b.jpg](../iclr_results/1495_Fast%20Summation%20of%20Radial%20Kernels%20via%20QMC%20Slicing/tables/f699ca107cda194db03950210e4216f480760e441a5ba7743ddca23bfa467d4b.jpg)

## Stochastic variance-reduced Gaussian variational inference on the Bures-Wasserstein manifold


### Images

![03201a97dcebfa5d8958bdae2a3076df087fc88d01b193f865af983e8725a3e0.jpg](../iclr_results/1496_Stochastic%20variance-reduced%20Gaussian%20variational%20inference%20on%20the%20Bures-Wasserstein%20manifold/images/03201a97dcebfa5d8958bdae2a3076df087fc88d01b193f865af983e8725a3e0.jpg)

![12a99c9ea9e85c19e6c5b46cf514146aa3836a6fbfca395ed222d025b540cef3.jpg](../iclr_results/1496_Stochastic%20variance-reduced%20Gaussian%20variational%20inference%20on%20the%20Bures-Wasserstein%20manifold/images/12a99c9ea9e85c19e6c5b46cf514146aa3836a6fbfca395ed222d025b540cef3.jpg)

![16f7400437b49f0c1e6e0474cf9dd268a4e3b0f27ed0fecaf74108bbab4b064d.jpg](../iclr_results/1496_Stochastic%20variance-reduced%20Gaussian%20variational%20inference%20on%20the%20Bures-Wasserstein%20manifold/images/16f7400437b49f0c1e6e0474cf9dd268a4e3b0f27ed0fecaf74108bbab4b064d.jpg)

![2e9f86dbbe8ddfed3245efb70eaa161d60dd19844884f5fe2253e3c48d3b7f31.jpg](../iclr_results/1496_Stochastic%20variance-reduced%20Gaussian%20variational%20inference%20on%20the%20Bures-Wasserstein%20manifold/images/2e9f86dbbe8ddfed3245efb70eaa161d60dd19844884f5fe2253e3c48d3b7f31.jpg)

![5a1497ce5a088d5c8ac0668c4c5f3cebf2ea110931dcac74d903af5509eba1c2.jpg](../iclr_results/1496_Stochastic%20variance-reduced%20Gaussian%20variational%20inference%20on%20the%20Bures-Wasserstein%20manifold/images/5a1497ce5a088d5c8ac0668c4c5f3cebf2ea110931dcac74d903af5509eba1c2.jpg)

![67e10f0e4ba5c6257ef4abc113c469715ef70edf30ecef54fc0e20889935f1f6.jpg](../iclr_results/1496_Stochastic%20variance-reduced%20Gaussian%20variational%20inference%20on%20the%20Bures-Wasserstein%20manifold/images/67e10f0e4ba5c6257ef4abc113c469715ef70edf30ecef54fc0e20889935f1f6.jpg)

![9db597ebedee643d0ce04f33bd5c12531e4de116c8601794f8fc2f7f42f36370.jpg](../iclr_results/1496_Stochastic%20variance-reduced%20Gaussian%20variational%20inference%20on%20the%20Bures-Wasserstein%20manifold/images/9db597ebedee643d0ce04f33bd5c12531e4de116c8601794f8fc2f7f42f36370.jpg)

![aaff1937c537af312a1e838b20f2882691646ec5236f033fc7188cc3e92f9fb7.jpg](../iclr_results/1496_Stochastic%20variance-reduced%20Gaussian%20variational%20inference%20on%20the%20Bures-Wasserstein%20manifold/images/aaff1937c537af312a1e838b20f2882691646ec5236f033fc7188cc3e92f9fb7.jpg)

![d99793748c50dde284d8ccf66ecb08ba760158353b49a51e53fdcd24e1b9d3c3.jpg](../iclr_results/1496_Stochastic%20variance-reduced%20Gaussian%20variational%20inference%20on%20the%20Bures-Wasserstein%20manifold/images/d99793748c50dde284d8ccf66ecb08ba760158353b49a51e53fdcd24e1b9d3c3.jpg)

![db5dc5213744281f0186ad253fa1f267bcb150c070aec137a6faf7dc454e77e1.jpg](../iclr_results/1496_Stochastic%20variance-reduced%20Gaussian%20variational%20inference%20on%20the%20Bures-Wasserstein%20manifold/images/db5dc5213744281f0186ad253fa1f267bcb150c070aec137a6faf7dc454e77e1.jpg)

![e36564f24ff65fe1c6da9e910d0dc2345182ab737bfc6139ab40ba171bc77ab6.jpg](../iclr_results/1496_Stochastic%20variance-reduced%20Gaussian%20variational%20inference%20on%20the%20Bures-Wasserstein%20manifold/images/e36564f24ff65fe1c6da9e910d0dc2345182ab737bfc6139ab40ba171bc77ab6.jpg)

![e72492fdd6897badbd019b5ea327feb3786fb65275ee6cd311b19b2794bd6421.jpg](../iclr_results/1496_Stochastic%20variance-reduced%20Gaussian%20variational%20inference%20on%20the%20Bures-Wasserstein%20manifold/images/e72492fdd6897badbd019b5ea327feb3786fb65275ee6cd311b19b2794bd6421.jpg)

### Tables

![259ed559a411f2ae7a1962671fc1eb163123d7f41326b7b65074b51619ab5a18.jpg](../iclr_results/1496_Stochastic%20variance-reduced%20Gaussian%20variational%20inference%20on%20the%20Bures-Wasserstein%20manifold/tables/259ed559a411f2ae7a1962671fc1eb163123d7f41326b7b65074b51619ab5a18.jpg)

![a4f5e0a47e0f89021183ff28499e3393bfbaf1615e77f1990ae6bb051da94615.jpg](../iclr_results/1496_Stochastic%20variance-reduced%20Gaussian%20variational%20inference%20on%20the%20Bures-Wasserstein%20manifold/tables/a4f5e0a47e0f89021183ff28499e3393bfbaf1615e77f1990ae6bb051da94615.jpg)

## Why In-Context Learning Models are Good Few-Shot Learners?


### Images

![0b331a8c1f44235ccca9889eb596f8e685ad2cb2e6671f75790d156b849fa8dd.jpg](../iclr_results/1497_Why%20In-Context%20Learning%20Models%20are%20Good%20Few-Shot%20Learners_/images/0b331a8c1f44235ccca9889eb596f8e685ad2cb2e6671f75790d156b849fa8dd.jpg)

![0f74b425c70da04e6d96fe498eafaa377dd386d2b4a04e1566577e17bc2e5a2e.jpg](../iclr_results/1497_Why%20In-Context%20Learning%20Models%20are%20Good%20Few-Shot%20Learners_/images/0f74b425c70da04e6d96fe498eafaa377dd386d2b4a04e1566577e17bc2e5a2e.jpg)

![38811b427013739cbe396586cb492839f283b8336fa38bc8b06aa86e086e119c.jpg](../iclr_results/1497_Why%20In-Context%20Learning%20Models%20are%20Good%20Few-Shot%20Learners_/images/38811b427013739cbe396586cb492839f283b8336fa38bc8b06aa86e086e119c.jpg)

![4b473d6c929549f155d029cf55607f6d7e8e5717291e99741e16c90f1559d965.jpg](../iclr_results/1497_Why%20In-Context%20Learning%20Models%20are%20Good%20Few-Shot%20Learners_/images/4b473d6c929549f155d029cf55607f6d7e8e5717291e99741e16c90f1559d965.jpg)

![6444d2ec3cee33e7056791167a3cbf4ebd3755e87138f58a34c091e453584565.jpg](../iclr_results/1497_Why%20In-Context%20Learning%20Models%20are%20Good%20Few-Shot%20Learners_/images/6444d2ec3cee33e7056791167a3cbf4ebd3755e87138f58a34c091e453584565.jpg)

![6bef3bc66969d928c7cb1556e18e98c3a5011ef58fb32b0f7f4165932b99ce51.jpg](../iclr_results/1497_Why%20In-Context%20Learning%20Models%20are%20Good%20Few-Shot%20Learners_/images/6bef3bc66969d928c7cb1556e18e98c3a5011ef58fb32b0f7f4165932b99ce51.jpg)

![7d584269636b1fe7b18ca4b35596e637194f0fcf5962bb023795c2695fd95c98.jpg](../iclr_results/1497_Why%20In-Context%20Learning%20Models%20are%20Good%20Few-Shot%20Learners_/images/7d584269636b1fe7b18ca4b35596e637194f0fcf5962bb023795c2695fd95c98.jpg)

![883058ea59d3d6c9cb7a22a2f059594f76bfa57420c1a944c8403d9861bcba5f.jpg](../iclr_results/1497_Why%20In-Context%20Learning%20Models%20are%20Good%20Few-Shot%20Learners_/images/883058ea59d3d6c9cb7a22a2f059594f76bfa57420c1a944c8403d9861bcba5f.jpg)

![a7f59dbff771c63eb25f030146e3f1ba27c24dcdaa9fd7435fd9e52ba5acdba0.jpg](../iclr_results/1497_Why%20In-Context%20Learning%20Models%20are%20Good%20Few-Shot%20Learners_/images/a7f59dbff771c63eb25f030146e3f1ba27c24dcdaa9fd7435fd9e52ba5acdba0.jpg)

![b55eee46da12366bb1e0dffdfc708270efd9247f326aa8e9bae3051d648824fd.jpg](../iclr_results/1497_Why%20In-Context%20Learning%20Models%20are%20Good%20Few-Shot%20Learners_/images/b55eee46da12366bb1e0dffdfc708270efd9247f326aa8e9bae3051d648824fd.jpg)

![bae903c830f46b4bc4cb4566fb4b9a8f59ffd5b4114027f25157494776831814.jpg](../iclr_results/1497_Why%20In-Context%20Learning%20Models%20are%20Good%20Few-Shot%20Learners_/images/bae903c830f46b4bc4cb4566fb4b9a8f59ffd5b4114027f25157494776831814.jpg)

![bd6272dba3679a932b45a194105681a10b46441caf53c0184f5ce05101670f18.jpg](../iclr_results/1497_Why%20In-Context%20Learning%20Models%20are%20Good%20Few-Shot%20Learners_/images/bd6272dba3679a932b45a194105681a10b46441caf53c0184f5ce05101670f18.jpg)

![c0b30170a16c19892a2f276dcecc0fb3df5b58eab467d43da56504d7aa57c959.jpg](../iclr_results/1497_Why%20In-Context%20Learning%20Models%20are%20Good%20Few-Shot%20Learners_/images/c0b30170a16c19892a2f276dcecc0fb3df5b58eab467d43da56504d7aa57c959.jpg)

![e5f512a7b4ca5b9ecd12bce0c305a07186ff25f8800b53e4921643d7d915d627.jpg](../iclr_results/1497_Why%20In-Context%20Learning%20Models%20are%20Good%20Few-Shot%20Learners_/images/e5f512a7b4ca5b9ecd12bce0c305a07186ff25f8800b53e4921643d7d915d627.jpg)

![f0ba5f3c7ba62fa5fc18f760f05ca3ffcfcd38e4577e1ed097c2a0289f2091a0.jpg](../iclr_results/1497_Why%20In-Context%20Learning%20Models%20are%20Good%20Few-Shot%20Learners_/images/f0ba5f3c7ba62fa5fc18f760f05ca3ffcfcd38e4577e1ed097c2a0289f2091a0.jpg)

![fac7307caf39f4a3b860cb0ef39c98f13b5d52a4ad8273467a99054ad40c9142.jpg](../iclr_results/1497_Why%20In-Context%20Learning%20Models%20are%20Good%20Few-Shot%20Learners_/images/fac7307caf39f4a3b860cb0ef39c98f13b5d52a4ad8273467a99054ad40c9142.jpg)

### Tables

![4c516961834151f127965462ee8d983d114c58b7dfe8e42ba42c8f3b6a3c6284.jpg](../iclr_results/1497_Why%20In-Context%20Learning%20Models%20are%20Good%20Few-Shot%20Learners_/tables/4c516961834151f127965462ee8d983d114c58b7dfe8e42ba42c8f3b6a3c6284.jpg)

## 3DitScene: Editing Any Scene via Language-guided Disentangled Gaussian Splatting


### Images

![079a29c90c7a8d522b3afa4561cac5e042c1d14bce09ad36de20ac37d51e989a.jpg](../iclr_results/1498_3DitScene_%20Editing%20Any%20Scene%20via%20Language-guided%20Disentangled%20Gaussian%20Splatting/images/079a29c90c7a8d522b3afa4561cac5e042c1d14bce09ad36de20ac37d51e989a.jpg)

![1c243852496321a91ec6d0d496a6313dc221ba00814d3cd6f43a5d029a8ac597.jpg](../iclr_results/1498_3DitScene_%20Editing%20Any%20Scene%20via%20Language-guided%20Disentangled%20Gaussian%20Splatting/images/1c243852496321a91ec6d0d496a6313dc221ba00814d3cd6f43a5d029a8ac597.jpg)

![1f156e501341f8683c16612bdefa4932c15603c3b2b8f8cc1d55479fe889110e.jpg](../iclr_results/1498_3DitScene_%20Editing%20Any%20Scene%20via%20Language-guided%20Disentangled%20Gaussian%20Splatting/images/1f156e501341f8683c16612bdefa4932c15603c3b2b8f8cc1d55479fe889110e.jpg)

![322adc0031868dba942e05a76d3541c620676b644f11fbc17022faad3036e8f5.jpg](../iclr_results/1498_3DitScene_%20Editing%20Any%20Scene%20via%20Language-guided%20Disentangled%20Gaussian%20Splatting/images/322adc0031868dba942e05a76d3541c620676b644f11fbc17022faad3036e8f5.jpg)

![5721115d2e258977a631da8c53182adc7b024bdd5b6e4b6349e7525695b15a67.jpg](../iclr_results/1498_3DitScene_%20Editing%20Any%20Scene%20via%20Language-guided%20Disentangled%20Gaussian%20Splatting/images/5721115d2e258977a631da8c53182adc7b024bdd5b6e4b6349e7525695b15a67.jpg)

![584e97bb84bdd3d56b1e54bd126f7625f9b943e9fad8f38f2d020821c2c3a51f.jpg](../iclr_results/1498_3DitScene_%20Editing%20Any%20Scene%20via%20Language-guided%20Disentangled%20Gaussian%20Splatting/images/584e97bb84bdd3d56b1e54bd126f7625f9b943e9fad8f38f2d020821c2c3a51f.jpg)

![77b54e9a2ec9aeee11b222827a77badada9125810a1920461f0ef6dfeadc6bb7.jpg](../iclr_results/1498_3DitScene_%20Editing%20Any%20Scene%20via%20Language-guided%20Disentangled%20Gaussian%20Splatting/images/77b54e9a2ec9aeee11b222827a77badada9125810a1920461f0ef6dfeadc6bb7.jpg)

![80e08c049202b9730507dfc07dea0e0c9606dba2440d7634882ce7ece917ea23.jpg](../iclr_results/1498_3DitScene_%20Editing%20Any%20Scene%20via%20Language-guided%20Disentangled%20Gaussian%20Splatting/images/80e08c049202b9730507dfc07dea0e0c9606dba2440d7634882ce7ece917ea23.jpg)

![b7647e404f00cd28734c76a0daaf25eb4afb9a4d93cf8a37f563781c24886729.jpg](../iclr_results/1498_3DitScene_%20Editing%20Any%20Scene%20via%20Language-guided%20Disentangled%20Gaussian%20Splatting/images/b7647e404f00cd28734c76a0daaf25eb4afb9a4d93cf8a37f563781c24886729.jpg)

![c61332141ebb1c9e94e778866c31305990a03930647a5f227274a86bb9294555.jpg](../iclr_results/1498_3DitScene_%20Editing%20Any%20Scene%20via%20Language-guided%20Disentangled%20Gaussian%20Splatting/images/c61332141ebb1c9e94e778866c31305990a03930647a5f227274a86bb9294555.jpg)

![d7231a2fd2dd4837e59888a2ddb4bc80a657417b438c7e978023bddeb7d1ef1f.jpg](../iclr_results/1498_3DitScene_%20Editing%20Any%20Scene%20via%20Language-guided%20Disentangled%20Gaussian%20Splatting/images/d7231a2fd2dd4837e59888a2ddb4bc80a657417b438c7e978023bddeb7d1ef1f.jpg)

![df08e100f42834f8afa54c4fc0a1e8259be448b076c31de7665a043e96242eeb.jpg](../iclr_results/1498_3DitScene_%20Editing%20Any%20Scene%20via%20Language-guided%20Disentangled%20Gaussian%20Splatting/images/df08e100f42834f8afa54c4fc0a1e8259be448b076c31de7665a043e96242eeb.jpg)

![e630e203d9dd26eb9a4e9ce718460c5b1542e44354d2f9d4da6a4e1d6a80bae3.jpg](../iclr_results/1498_3DitScene_%20Editing%20Any%20Scene%20via%20Language-guided%20Disentangled%20Gaussian%20Splatting/images/e630e203d9dd26eb9a4e9ce718460c5b1542e44354d2f9d4da6a4e1d6a80bae3.jpg)

![f3e7524b07a192c13ff021577093770ac097177f720241616e841aad7f7b53b6.jpg](../iclr_results/1498_3DitScene_%20Editing%20Any%20Scene%20via%20Language-guided%20Disentangled%20Gaussian%20Splatting/images/f3e7524b07a192c13ff021577093770ac097177f720241616e841aad7f7b53b6.jpg)

### Tables

![9d323ba497955b97b1ae16cd959caa7f9944ebfc60af775de979c4699d315bb1.jpg](../iclr_results/1498_3DitScene_%20Editing%20Any%20Scene%20via%20Language-guided%20Disentangled%20Gaussian%20Splatting/tables/9d323ba497955b97b1ae16cd959caa7f9944ebfc60af775de979c4699d315bb1.jpg)

## Diffusion-NPO: Negative Preference Optimization for Better Preference Aligned Generation of Diffusion Models


### Images

![0af3f56c29a4ed5ca3d878c234a5473c41367e9bc87e10e53f8c3306e3d94bd0.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/0af3f56c29a4ed5ca3d878c234a5473c41367e9bc87e10e53f8c3306e3d94bd0.jpg)

![0efc550d64e0739c83416049681f3c73f14d48e2438264c7b92d7e98aa306d50.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/0efc550d64e0739c83416049681f3c73f14d48e2438264c7b92d7e98aa306d50.jpg)

![11b9519e345cb951da6dfa2b74339038f907f881e7c1b70f8625d57e0d613abf.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/11b9519e345cb951da6dfa2b74339038f907f881e7c1b70f8625d57e0d613abf.jpg)

![1cf33c4ce4b0f7cfe0fc2ca757318f37b2f79c9358acea5e331544dbf658c62a.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/1cf33c4ce4b0f7cfe0fc2ca757318f37b2f79c9358acea5e331544dbf658c62a.jpg)

![1e2c9530616718c8596b6d8d7962a9f0892522e9a4ea8bece05b3ae42a6f38f3.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/1e2c9530616718c8596b6d8d7962a9f0892522e9a4ea8bece05b3ae42a6f38f3.jpg)

![275ed2dcada55b2f4ebd3177a6ed4367ae9b4cd55beee2b19fc8cf2fdc56100b.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/275ed2dcada55b2f4ebd3177a6ed4367ae9b4cd55beee2b19fc8cf2fdc56100b.jpg)

![2a81a9a947471c93fce7a9d90b2e2e1412085aa8b99061d1e90d069bcf830c68.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/2a81a9a947471c93fce7a9d90b2e2e1412085aa8b99061d1e90d069bcf830c68.jpg)

![38f8476f5020e66cc985bf8e4c04840e8ec94dacabd9cfc18c97b454196e91ba.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/38f8476f5020e66cc985bf8e4c04840e8ec94dacabd9cfc18c97b454196e91ba.jpg)

![3a1bdf7190af2503c7b1da51ef402617893e58e58bff0aef6b394d4b3ced3744.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/3a1bdf7190af2503c7b1da51ef402617893e58e58bff0aef6b394d4b3ced3744.jpg)

![3b60e55ddcce64242185f5f8ec356966073f5b27b55361b3e38672e586d32429.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/3b60e55ddcce64242185f5f8ec356966073f5b27b55361b3e38672e586d32429.jpg)

![449608b18adf3bcb2c5238eea009b811bceabc7edb62b48934805bba9364c029.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/449608b18adf3bcb2c5238eea009b811bceabc7edb62b48934805bba9364c029.jpg)

![4891c02879a012dae3179eb4b421aa0394e17fce98e5d3ef857122cdf1828863.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/4891c02879a012dae3179eb4b421aa0394e17fce98e5d3ef857122cdf1828863.jpg)

![514f8f375196cb5240eced80afffa4756a20b8cdf5493162fdbc2fe670ee2dcf.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/514f8f375196cb5240eced80afffa4756a20b8cdf5493162fdbc2fe670ee2dcf.jpg)

![5385880a261af7f7a03d3f7ddb550c4d838f29069dfa71731f2bc578f0626907.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/5385880a261af7f7a03d3f7ddb550c4d838f29069dfa71731f2bc578f0626907.jpg)

![5f19c29ab2692067d358a21c04f67c96141c734eb9736043bb954002c3e4450e.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/5f19c29ab2692067d358a21c04f67c96141c734eb9736043bb954002c3e4450e.jpg)

![66bf7745877810005aeeab59f9bdd1e9aabe83771f4d6a2e40ee5b4d4713ca87.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/66bf7745877810005aeeab59f9bdd1e9aabe83771f4d6a2e40ee5b4d4713ca87.jpg)

![6bd5d8ac87b3bea4d014068fa6057a524df2b8266d7d3cc4a1c75d9301a86d9d.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/6bd5d8ac87b3bea4d014068fa6057a524df2b8266d7d3cc4a1c75d9301a86d9d.jpg)

![7a97b0c0c193b1e544f6d35f7e53f37226af02e61ae3ab21971c3391351a23ef.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/7a97b0c0c193b1e544f6d35f7e53f37226af02e61ae3ab21971c3391351a23ef.jpg)

![7e07b236f3bb3475a2fb95f9ba4941592829dc6e1c6c7c7a5c42bd5e91cb510f.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/7e07b236f3bb3475a2fb95f9ba4941592829dc6e1c6c7c7a5c42bd5e91cb510f.jpg)

![85668301288ca4e8989bbc2b9bc94d6143d1defa2b78c9178aec0c3122799a2a.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/85668301288ca4e8989bbc2b9bc94d6143d1defa2b78c9178aec0c3122799a2a.jpg)

![8ca471dc8f32cc6ad435ff693bc4f2a06e5d0a58257906500f8796a48bef5582.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/8ca471dc8f32cc6ad435ff693bc4f2a06e5d0a58257906500f8796a48bef5582.jpg)

![8e8fdca5030936e5c1ce54a706d916d5d1d8691bbe87399829e1d0d13881aed0.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/8e8fdca5030936e5c1ce54a706d916d5d1d8691bbe87399829e1d0d13881aed0.jpg)

![94acdd46a84c988762b69faab0ba16a36d3596dbbdaef0fe9f5a5c0c59e8b7f0.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/94acdd46a84c988762b69faab0ba16a36d3596dbbdaef0fe9f5a5c0c59e8b7f0.jpg)

![973e381dbf05a33ac187431f17143149de3521ee8b2f8e55d8350dd5fd012908.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/973e381dbf05a33ac187431f17143149de3521ee8b2f8e55d8350dd5fd012908.jpg)

![9b92dd8c6655fb061bdacf3cb146480e2c0f8ad57a56e62c0ea1c25b8d79b122.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/9b92dd8c6655fb061bdacf3cb146480e2c0f8ad57a56e62c0ea1c25b8d79b122.jpg)

![a29873d01e0d5977e3c0a0fa638b7dc3e5d03c6976f02e47bb889e39dfb56efd.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/a29873d01e0d5977e3c0a0fa638b7dc3e5d03c6976f02e47bb889e39dfb56efd.jpg)

![aaf60f324d71424cf50d9385c77340b9a25596c575f17dcb99421a0f649e128b.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/aaf60f324d71424cf50d9385c77340b9a25596c575f17dcb99421a0f649e128b.jpg)

![ac0bd830d5660fab6fd80cc878f09e6d942bbd145587aa1c4885fcb703a2b678.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/ac0bd830d5660fab6fd80cc878f09e6d942bbd145587aa1c4885fcb703a2b678.jpg)

![bcd16106857c8691a1942289131f5e326ed92112ff43cbda6b650fe6142ad8f0.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/bcd16106857c8691a1942289131f5e326ed92112ff43cbda6b650fe6142ad8f0.jpg)

![bda8c5d8da5fafe84bbbad47b8e1033ada9855c09232add43787bafcaf9c50a1.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/bda8c5d8da5fafe84bbbad47b8e1033ada9855c09232add43787bafcaf9c50a1.jpg)

![c3248e0b7ee8ac567f1b94cb066f8d65a51f2fc8ab3d21421f46c51604e063be.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/c3248e0b7ee8ac567f1b94cb066f8d65a51f2fc8ab3d21421f46c51604e063be.jpg)

![cdfae1202396a57c8e4fe36e45805f5b60668584080d48716d97a38941ef74a4.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/cdfae1202396a57c8e4fe36e45805f5b60668584080d48716d97a38941ef74a4.jpg)

![dd66adbcda9763867c29411ab1a0fe92eb5baa789d7c95fe4b2e3e7c0b84804a.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/dd66adbcda9763867c29411ab1a0fe92eb5baa789d7c95fe4b2e3e7c0b84804a.jpg)

![dea0fbe1fd5eef28bf0700ee562b034d85e999d6e8fb2fc399f63e88e9141579.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/dea0fbe1fd5eef28bf0700ee562b034d85e999d6e8fb2fc399f63e88e9141579.jpg)

![e31c737f56bea430c563711010dcd18200f19615d269c027278c9d213ca60955.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/e31c737f56bea430c563711010dcd18200f19615d269c027278c9d213ca60955.jpg)

![e4278fee2a7508c3fd49e8d03d9d8e01d45bc97489dd66bf9e24f9a5bed9b3d3.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/e4278fee2a7508c3fd49e8d03d9d8e01d45bc97489dd66bf9e24f9a5bed9b3d3.jpg)

![e8cf56f3c25847643cb3793fff66a753771d936f1d5bd4a8347b6cd0996e4952.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/e8cf56f3c25847643cb3793fff66a753771d936f1d5bd4a8347b6cd0996e4952.jpg)

![ea82bd691584e10567a508c32094865f54ba2685b1636eb1b4e157abc7c4b645.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/ea82bd691584e10567a508c32094865f54ba2685b1636eb1b4e157abc7c4b645.jpg)

![ec6bb09711dbc157cdae0e7406df8dbeaaa6611a92bf7b940986f9102346bc89.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/ec6bb09711dbc157cdae0e7406df8dbeaaa6611a92bf7b940986f9102346bc89.jpg)

![fa374ae657e355ee1eee44c2feafd96af69e78268d10e9e55615828331a96b53.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/fa374ae657e355ee1eee44c2feafd96af69e78268d10e9e55615828331a96b53.jpg)

![fd33c26fe4f85cf10e62e94a9e22957ca6c6ace138d757594f73d7cf34bf2c78.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/fd33c26fe4f85cf10e62e94a9e22957ca6c6ace138d757594f73d7cf34bf2c78.jpg)

![ff847e6ad43fe478f12b6c2bb012e9b68e58946ee0765d77f4657aaf894530db.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/images/ff847e6ad43fe478f12b6c2bb012e9b68e58946ee0765d77f4657aaf894530db.jpg)

### Tables

![15692befefdc9456f089427f7be551987b3d58f5f67e21a17f07f840a9700fb6.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/tables/15692befefdc9456f089427f7be551987b3d58f5f67e21a17f07f840a9700fb6.jpg)

![2e002bec49e4bfcb9f00525eb2334b3b6ee14252183793e4bf501c9f03bfd35e.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/tables/2e002bec49e4bfcb9f00525eb2334b3b6ee14252183793e4bf501c9f03bfd35e.jpg)

![3edb7c3c44c86a27cca3150260174ac7d9c8e1f3dda724e17e15755bf1729b53.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/tables/3edb7c3c44c86a27cca3150260174ac7d9c8e1f3dda724e17e15755bf1729b53.jpg)

![3f49bb08cfa43ea3d33ec68b5527067b6b7e4828673790d0a309708b9f65c350.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/tables/3f49bb08cfa43ea3d33ec68b5527067b6b7e4828673790d0a309708b9f65c350.jpg)

![69f0a5b07b67548a6eb66b995fd2c628ac22ddbefc6816232a2c57c4f2ea4e32.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/tables/69f0a5b07b67548a6eb66b995fd2c628ac22ddbefc6816232a2c57c4f2ea4e32.jpg)

![7ea9e8c12f74ee5bdf2520991ea6fff36d85c725d3c0f4e858b8ee14c1fa01b9.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/tables/7ea9e8c12f74ee5bdf2520991ea6fff36d85c725d3c0f4e858b8ee14c1fa01b9.jpg)

![95925e7f5b92a92f645e781812cfa0229985e61a9760bf3da745b1fdb3f2b6e7.jpg](../iclr_results/1499_Diffusion-NPO_%20Negative%20Preference%20Optimization%20for%20Better%20Preference%20Aligned%20Generation%20of%20Diffusio/tables/95925e7f5b92a92f645e781812cfa0229985e61a9760bf3da745b1fdb3f2b6e7.jpg)

## Duoduo CLIP: Efficient 3D Understanding with Multi-View Images


### Images

![399de39ac08143ceab50e473931d8a6eb12abe9c9d79ad4fad81eed3e963bb71.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/images/399de39ac08143ceab50e473931d8a6eb12abe9c9d79ad4fad81eed3e963bb71.jpg)

![444fa37166a7dbd415d6fec21295347e3d5330db7b40eb12ed834c447f5d1b41.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/images/444fa37166a7dbd415d6fec21295347e3d5330db7b40eb12ed834c447f5d1b41.jpg)

![4a88089bd9615ff152cba8f1a84508be355a726e338dd49e814c45e6a1543f0d.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/images/4a88089bd9615ff152cba8f1a84508be355a726e338dd49e814c45e6a1543f0d.jpg)

![583f74207ab785cc26a4cf1e198c19df180efea1709af29501cc09c245ee5108.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/images/583f74207ab785cc26a4cf1e198c19df180efea1709af29501cc09c245ee5108.jpg)

![5913438c53d19cab891f98693b1f8b87b2e851b741864a9949fe02df01228434.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/images/5913438c53d19cab891f98693b1f8b87b2e851b741864a9949fe02df01228434.jpg)

![8084fa1d3abc11bffe2cc5f2892ecedcbd5e7f1cb70661a510f058106d08a0ae.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/images/8084fa1d3abc11bffe2cc5f2892ecedcbd5e7f1cb70661a510f058106d08a0ae.jpg)

![8b51d2a621df2ece51f54b764aa285f20292de17af188dd7b8bc1a06e1c5f00a.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/images/8b51d2a621df2ece51f54b764aa285f20292de17af188dd7b8bc1a06e1c5f00a.jpg)

![aed33c7342f6fb531fa811ec26d65dafbb7a4e5c17fe374744e1dfde8d2e9835.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/images/aed33c7342f6fb531fa811ec26d65dafbb7a4e5c17fe374744e1dfde8d2e9835.jpg)

![aff23a2325fff69a5d32b3433b5561ed3e67f3cfee7f44d8d9805d4f583dbf98.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/images/aff23a2325fff69a5d32b3433b5561ed3e67f3cfee7f44d8d9805d4f583dbf98.jpg)

![b1db00f370beaf7aba13c6bbe9b95e59008e5d956a38907c201a9a0638eda827.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/images/b1db00f370beaf7aba13c6bbe9b95e59008e5d956a38907c201a9a0638eda827.jpg)

![b62dfcce5edc2e4b50e381587671aa465599fb753233848c45ba775a66902498.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/images/b62dfcce5edc2e4b50e381587671aa465599fb753233848c45ba775a66902498.jpg)

![b81c86cfda0c344c7f34b26c26f047fad2260caab901e68b1f2e42efefe0cb88.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/images/b81c86cfda0c344c7f34b26c26f047fad2260caab901e68b1f2e42efefe0cb88.jpg)

![c37d9dcd393fefff01bbed975a2c6ef3b71f2665b8003b4ef0ae9fb8f26b90b8.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/images/c37d9dcd393fefff01bbed975a2c6ef3b71f2665b8003b4ef0ae9fb8f26b90b8.jpg)

### Tables

![06dba9ca8454dc54cb247264360961014e2d8ce2af8b48dbc273d43d7fd0893a.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/tables/06dba9ca8454dc54cb247264360961014e2d8ce2af8b48dbc273d43d7fd0893a.jpg)

![14a70e6c40f6f744079114b3de9121b93d7b2b5d676d3c057099a7c63740dc13.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/tables/14a70e6c40f6f744079114b3de9121b93d7b2b5d676d3c057099a7c63740dc13.jpg)

![2ddcebd8fd1f9985752c0bb7a20908ff2078f8f3ff8b25c023fd01b24e11517b.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/tables/2ddcebd8fd1f9985752c0bb7a20908ff2078f8f3ff8b25c023fd01b24e11517b.jpg)

![3c2da2aaa07fae6326d41e212ef9d29c4985ff82b41d8f90416a52b632df3d19.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/tables/3c2da2aaa07fae6326d41e212ef9d29c4985ff82b41d8f90416a52b632df3d19.jpg)

![3cd5850a1a52096c740f9eeb1d37699d758561cdc6e7022d5c9ce125bf989e89.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/tables/3cd5850a1a52096c740f9eeb1d37699d758561cdc6e7022d5c9ce125bf989e89.jpg)

![5cbbcb14ee8a4cb4ed113a8256dc80cbb0af47f085706af2dafea24d97703811.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/tables/5cbbcb14ee8a4cb4ed113a8256dc80cbb0af47f085706af2dafea24d97703811.jpg)

![70cab01c65bdd2e8938dbc44422f920d9d37cd1084d1ff6c55b33931cc05d7db.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/tables/70cab01c65bdd2e8938dbc44422f920d9d37cd1084d1ff6c55b33931cc05d7db.jpg)

![7d336fb23d8aa03e003612d5cb86502f7c7d80bbf6bb768448e15ca2d861a0bc.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/tables/7d336fb23d8aa03e003612d5cb86502f7c7d80bbf6bb768448e15ca2d861a0bc.jpg)

![83924d11ab30a430da5da4265daad999b59ca73615bb907a3017d1999f6ef176.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/tables/83924d11ab30a430da5da4265daad999b59ca73615bb907a3017d1999f6ef176.jpg)

![a46a4af218245e0e4a74a85acd6c6c0e27687703069a17e6addccef72ef4771d.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/tables/a46a4af218245e0e4a74a85acd6c6c0e27687703069a17e6addccef72ef4771d.jpg)

![bfac5e33d479806255cc33cfef21d5b9aa982041a52014e424f2eceebfedb54d.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/tables/bfac5e33d479806255cc33cfef21d5b9aa982041a52014e424f2eceebfedb54d.jpg)

![cba38a91b7bb9032090e241c8aa72dd36999e36ff5f988b875b9c0f97f04eb64.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/tables/cba38a91b7bb9032090e241c8aa72dd36999e36ff5f988b875b9c0f97f04eb64.jpg)

![f009796203ec44b3aa047be7b50a1a068794ca3a5e0181ca1f9db2d22d5166ae.jpg](../iclr_results/1500_Duoduo%20CLIP_%20Efficient%203D%20Understanding%20with%20Multi-View%20Images/tables/f009796203ec44b3aa047be7b50a1a068794ca3a5e0181ca1f9db2d22d5166ae.jpg)

## DenseGrounding: Improving Dense Language-Vision Semantics for Ego-centric 3D Visual Grounding


### Images

![1c3d23edf7b563cca23bcd39d7d1c78e36d95347f1a3be72e2af053c616b3396.jpg](../iclr_results/1501_DenseGrounding_%20Improving%20Dense%20Language-Vision%20Semantics%20for%20Ego-centric%203D%20Visual%20Grounding/images/1c3d23edf7b563cca23bcd39d7d1c78e36d95347f1a3be72e2af053c616b3396.jpg)

![3e5b433c879c1d6d70876e5d3314e4c51513c249b062d2a1f02a549ea8165056.jpg](../iclr_results/1501_DenseGrounding_%20Improving%20Dense%20Language-Vision%20Semantics%20for%20Ego-centric%203D%20Visual%20Grounding/images/3e5b433c879c1d6d70876e5d3314e4c51513c249b062d2a1f02a549ea8165056.jpg)

![4b3e80caeea99ce9f4a349fdc88298294ccca5eda5fa85137e0d01d7924e7984.jpg](../iclr_results/1501_DenseGrounding_%20Improving%20Dense%20Language-Vision%20Semantics%20for%20Ego-centric%203D%20Visual%20Grounding/images/4b3e80caeea99ce9f4a349fdc88298294ccca5eda5fa85137e0d01d7924e7984.jpg)

![9373df0fe770d9e25bc5da69ecbc5e70cbf6fe63aa882fdf39fb159705922680.jpg](../iclr_results/1501_DenseGrounding_%20Improving%20Dense%20Language-Vision%20Semantics%20for%20Ego-centric%203D%20Visual%20Grounding/images/9373df0fe770d9e25bc5da69ecbc5e70cbf6fe63aa882fdf39fb159705922680.jpg)

![9fbfd06d6153836fb91ed23f380955893b681f6d9754c9d3f8ca925721f1db4d.jpg](../iclr_results/1501_DenseGrounding_%20Improving%20Dense%20Language-Vision%20Semantics%20for%20Ego-centric%203D%20Visual%20Grounding/images/9fbfd06d6153836fb91ed23f380955893b681f6d9754c9d3f8ca925721f1db4d.jpg)

### Tables

![02d2fa74238a7b0f6cdb988e0f8937ad7a7df39bf4ae1c15d34c5cbe2ee4c5fe.jpg](../iclr_results/1501_DenseGrounding_%20Improving%20Dense%20Language-Vision%20Semantics%20for%20Ego-centric%203D%20Visual%20Grounding/tables/02d2fa74238a7b0f6cdb988e0f8937ad7a7df39bf4ae1c15d34c5cbe2ee4c5fe.jpg)

![2b5f1b1e2532ad080cedafe3d2b7d2bc6a1f75a1864e35823b0e953768de5033.jpg](../iclr_results/1501_DenseGrounding_%20Improving%20Dense%20Language-Vision%20Semantics%20for%20Ego-centric%203D%20Visual%20Grounding/tables/2b5f1b1e2532ad080cedafe3d2b7d2bc6a1f75a1864e35823b0e953768de5033.jpg)

![66028f5264f0b72c9f8734725eb85a4a3a28eae5a7e55fb5190745edc0d3b021.jpg](../iclr_results/1501_DenseGrounding_%20Improving%20Dense%20Language-Vision%20Semantics%20for%20Ego-centric%203D%20Visual%20Grounding/tables/66028f5264f0b72c9f8734725eb85a4a3a28eae5a7e55fb5190745edc0d3b021.jpg)

![670efc6b1d582a5c9fe4e75e214c7b03b299810dbccc4f71bb21c6cd4cc839e8.jpg](../iclr_results/1501_DenseGrounding_%20Improving%20Dense%20Language-Vision%20Semantics%20for%20Ego-centric%203D%20Visual%20Grounding/tables/670efc6b1d582a5c9fe4e75e214c7b03b299810dbccc4f71bb21c6cd4cc839e8.jpg)

![7146f4049e44751e7ced388afa4ba71fbba8001b6942c5ec4ca2f6868473353e.jpg](../iclr_results/1501_DenseGrounding_%20Improving%20Dense%20Language-Vision%20Semantics%20for%20Ego-centric%203D%20Visual%20Grounding/tables/7146f4049e44751e7ced388afa4ba71fbba8001b6942c5ec4ca2f6868473353e.jpg)

![a730da590791e032119e352f039849415b20bb81d79afb047d2aba45fba117a9.jpg](../iclr_results/1501_DenseGrounding_%20Improving%20Dense%20Language-Vision%20Semantics%20for%20Ego-centric%203D%20Visual%20Grounding/tables/a730da590791e032119e352f039849415b20bb81d79afb047d2aba45fba117a9.jpg)

## Provably Robust Explainable Graph Neural Networks against Graph Perturbation Attacks


### Images

![222a39163c9f314322e46092c7020319bf21bb3b6f1cab738446398a9371684b.jpg](../iclr_results/1502_Provably%20Robust%20Explainable%20Graph%20Neural%20Networks%20against%20Graph%20Perturbation%20Attacks/images/222a39163c9f314322e46092c7020319bf21bb3b6f1cab738446398a9371684b.jpg)

![346f9d9f58774743280b78f9f4abb418c0eceaba8bd9e3e45c44827f5a6def0c.jpg](../iclr_results/1502_Provably%20Robust%20Explainable%20Graph%20Neural%20Networks%20against%20Graph%20Perturbation%20Attacks/images/346f9d9f58774743280b78f9f4abb418c0eceaba8bd9e3e45c44827f5a6def0c.jpg)

![3da817974d8b176716344f68219122309a42f1173fae00e8c86fe8ce217382fb.jpg](../iclr_results/1502_Provably%20Robust%20Explainable%20Graph%20Neural%20Networks%20against%20Graph%20Perturbation%20Attacks/images/3da817974d8b176716344f68219122309a42f1173fae00e8c86fe8ce217382fb.jpg)

![3e54d5a7760ef7724df2da9a438e18cefdbfd3a7a22a9335534215868e5046db.jpg](../iclr_results/1502_Provably%20Robust%20Explainable%20Graph%20Neural%20Networks%20against%20Graph%20Perturbation%20Attacks/images/3e54d5a7760ef7724df2da9a438e18cefdbfd3a7a22a9335534215868e5046db.jpg)

![5a538ae8ffcb6b0fe90dd50d5e9e3f8ed0061302d87afd1f90a7dc631189c505.jpg](../iclr_results/1502_Provably%20Robust%20Explainable%20Graph%20Neural%20Networks%20against%20Graph%20Perturbation%20Attacks/images/5a538ae8ffcb6b0fe90dd50d5e9e3f8ed0061302d87afd1f90a7dc631189c505.jpg)

![ab975dcdc1d36308a9ec6e2ec2bb066d49129812efff0d96c4743fd76d25daa9.jpg](../iclr_results/1502_Provably%20Robust%20Explainable%20Graph%20Neural%20Networks%20against%20Graph%20Perturbation%20Attacks/images/ab975dcdc1d36308a9ec6e2ec2bb066d49129812efff0d96c4743fd76d25daa9.jpg)

![ace5f9a0d6ed2f8835ebf5c5f04e1269d17a0e6f0f546a4e3fc559a4418f886a.jpg](../iclr_results/1502_Provably%20Robust%20Explainable%20Graph%20Neural%20Networks%20against%20Graph%20Perturbation%20Attacks/images/ace5f9a0d6ed2f8835ebf5c5f04e1269d17a0e6f0f546a4e3fc559a4418f886a.jpg)

![b19934e607721c5ead85fa23e55ccce5613129b2102e530963b406b5bf15eca1.jpg](../iclr_results/1502_Provably%20Robust%20Explainable%20Graph%20Neural%20Networks%20against%20Graph%20Perturbation%20Attacks/images/b19934e607721c5ead85fa23e55ccce5613129b2102e530963b406b5bf15eca1.jpg)

![c36abb9d3c7a7778aa8644575149bb62c2aacb2a8660c49476de906dc05c68ec.jpg](../iclr_results/1502_Provably%20Robust%20Explainable%20Graph%20Neural%20Networks%20against%20Graph%20Perturbation%20Attacks/images/c36abb9d3c7a7778aa8644575149bb62c2aacb2a8660c49476de906dc05c68ec.jpg)

![c65c7edb5de0695ea754fa4876b991dae5cfaeb07f7f99127257de7d27f7394c.jpg](../iclr_results/1502_Provably%20Robust%20Explainable%20Graph%20Neural%20Networks%20against%20Graph%20Perturbation%20Attacks/images/c65c7edb5de0695ea754fa4876b991dae5cfaeb07f7f99127257de7d27f7394c.jpg)

![d626631e03839bd645431ea1e5376ad40b2724141d14a2312c395a1666877b55.jpg](../iclr_results/1502_Provably%20Robust%20Explainable%20Graph%20Neural%20Networks%20against%20Graph%20Perturbation%20Attacks/images/d626631e03839bd645431ea1e5376ad40b2724141d14a2312c395a1666877b55.jpg)

![fd5ec23fd47decf06f9e1c3df9c8cc62b8dc43e8c8b862252bbebcd4b93b850a.jpg](../iclr_results/1502_Provably%20Robust%20Explainable%20Graph%20Neural%20Networks%20against%20Graph%20Perturbation%20Attacks/images/fd5ec23fd47decf06f9e1c3df9c8cc62b8dc43e8c8b862252bbebcd4b93b850a.jpg)

### Tables

![3b10c32f37e085d4c6cafab427015faf483cd4504e774418f2526029afdac380.jpg](../iclr_results/1502_Provably%20Robust%20Explainable%20Graph%20Neural%20Networks%20against%20Graph%20Perturbation%20Attacks/tables/3b10c32f37e085d4c6cafab427015faf483cd4504e774418f2526029afdac380.jpg)

![3ff25fd7d4eada45aa7afc948cd4db721c0caa0b9ec1c5dbfccff912e2d96f65.jpg](../iclr_results/1502_Provably%20Robust%20Explainable%20Graph%20Neural%20Networks%20against%20Graph%20Perturbation%20Attacks/tables/3ff25fd7d4eada45aa7afc948cd4db721c0caa0b9ec1c5dbfccff912e2d96f65.jpg)

![66f9b10220cc17c68cb8bd629d855a256715bfd803c484596492d9e89883085f.jpg](../iclr_results/1502_Provably%20Robust%20Explainable%20Graph%20Neural%20Networks%20against%20Graph%20Perturbation%20Attacks/tables/66f9b10220cc17c68cb8bd629d855a256715bfd803c484596492d9e89883085f.jpg)

![6746b37c466c6a1bfa11a13d7b68947ea43e5854d0b642c0ba53be90c0d16e16.jpg](../iclr_results/1502_Provably%20Robust%20Explainable%20Graph%20Neural%20Networks%20against%20Graph%20Perturbation%20Attacks/tables/6746b37c466c6a1bfa11a13d7b68947ea43e5854d0b642c0ba53be90c0d16e16.jpg)

![844ab514bf2378a7d5da99de0afa98d5d9dcb85215202e1bf24f38c591dc3529.jpg](../iclr_results/1502_Provably%20Robust%20Explainable%20Graph%20Neural%20Networks%20against%20Graph%20Perturbation%20Attacks/tables/844ab514bf2378a7d5da99de0afa98d5d9dcb85215202e1bf24f38c591dc3529.jpg)

![90fb0dd2017896997e24454910c6915392139315bac8c89f5e0ccf92b94e7b1b.jpg](../iclr_results/1502_Provably%20Robust%20Explainable%20Graph%20Neural%20Networks%20against%20Graph%20Perturbation%20Attacks/tables/90fb0dd2017896997e24454910c6915392139315bac8c89f5e0ccf92b94e7b1b.jpg)

![a273411254d31cd222a626b4d24641aa92ed7ea5a3e1c610c31c5ac27a0dbbe3.jpg](../iclr_results/1502_Provably%20Robust%20Explainable%20Graph%20Neural%20Networks%20against%20Graph%20Perturbation%20Attacks/tables/a273411254d31cd222a626b4d24641aa92ed7ea5a3e1c610c31c5ac27a0dbbe3.jpg)

![c0ddbf3f531e686043bbbcd65e2cff5f71521a7006b9e604beab17a11c21294b.jpg](../iclr_results/1502_Provably%20Robust%20Explainable%20Graph%20Neural%20Networks%20against%20Graph%20Perturbation%20Attacks/tables/c0ddbf3f531e686043bbbcd65e2cff5f71521a7006b9e604beab17a11c21294b.jpg)

![d5dd528dbd682c0ea0b1cc727d914c5f2945b0ecc732859c7103ad8cc6c9de47.jpg](../iclr_results/1502_Provably%20Robust%20Explainable%20Graph%20Neural%20Networks%20against%20Graph%20Perturbation%20Attacks/tables/d5dd528dbd682c0ea0b1cc727d914c5f2945b0ecc732859c7103ad8cc6c9de47.jpg)

## Near-optimal Active Regression of Single-Index Models


### Images

![18b1480c3e060a82f2797f68853e49758fd820c141a10b8ff55a355f4b0bbdc4.jpg](../iclr_results/1503_Near-optimal%20Active%20Regression%20of%20Single-Index%20Models/images/18b1480c3e060a82f2797f68853e49758fd820c141a10b8ff55a355f4b0bbdc4.jpg)

![e4eea3cd3d68dbc5e8e45bc8f6cffe9df36d59b97991f52a4fbbc94236b4c1b1.jpg](../iclr_results/1503_Near-optimal%20Active%20Regression%20of%20Single-Index%20Models/images/e4eea3cd3d68dbc5e8e45bc8f6cffe9df36d59b97991f52a4fbbc94236b4c1b1.jpg)

## The Optimization Landscape of SGD Across the Feature Learning Strength


### Images

![0ba7fcb0b0c1813f8f69717a1e3a0db700e12bf5d35992d9634ca2004cae5821.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/0ba7fcb0b0c1813f8f69717a1e3a0db700e12bf5d35992d9634ca2004cae5821.jpg)

![139a77111c206220ce79d0531349b12b5f1405ad2b590e2c8940f2f83b53d60b.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/139a77111c206220ce79d0531349b12b5f1405ad2b590e2c8940f2f83b53d60b.jpg)

![151d304e3c34f7dc24f91ce35260745a4802a3434adf999bce44bf461c3758b4.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/151d304e3c34f7dc24f91ce35260745a4802a3434adf999bce44bf461c3758b4.jpg)

![1c07ab309572db440b566ff3cffadbd26e21f2c4d8316ca65e06b73e8a55580d.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/1c07ab309572db440b566ff3cffadbd26e21f2c4d8316ca65e06b73e8a55580d.jpg)

![1dce60660e8815611eb6d1a3ca7fa071c1bdd9f595ce8d4a2dd268d6fde33a4d.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/1dce60660e8815611eb6d1a3ca7fa071c1bdd9f595ce8d4a2dd268d6fde33a4d.jpg)

![2da92d5b23f6efe0d122b4f8eef8aa4d953d686d0cee79d41eeed2c3d18130cd.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/2da92d5b23f6efe0d122b4f8eef8aa4d953d686d0cee79d41eeed2c3d18130cd.jpg)

![300ce84affa9962372f7e0d9ac6a030f2028e695af8bbf1726c33acc1c365d29.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/300ce84affa9962372f7e0d9ac6a030f2028e695af8bbf1726c33acc1c365d29.jpg)

![300d0bb9e0bbde1a9643dab959266e9defdde381432a9542d7e098280d5fe6af.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/300d0bb9e0bbde1a9643dab959266e9defdde381432a9542d7e098280d5fe6af.jpg)

![31b203d30990180c1e387f24cfc3556444e5c763a6fcfa04a6b2d9a8727d84d7.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/31b203d30990180c1e387f24cfc3556444e5c763a6fcfa04a6b2d9a8727d84d7.jpg)

![3638a9c742bd6cab2f78c5afec2644a45140ca7ce879b12ad36e9f58d419cd52.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/3638a9c742bd6cab2f78c5afec2644a45140ca7ce879b12ad36e9f58d419cd52.jpg)

![36497d05288e05e48a018ca64e7e1043abe4acba217275cfd50fdea4249bc898.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/36497d05288e05e48a018ca64e7e1043abe4acba217275cfd50fdea4249bc898.jpg)

![38f7150f5b6513b15d7d7a8ab9159e079c5ff5256585d50bd64c09290c6fe712.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/38f7150f5b6513b15d7d7a8ab9159e079c5ff5256585d50bd64c09290c6fe712.jpg)

![3f5c6c5b476b6c0ca2d02480ee50697baf032148a4dc891b29843f07bbac46c3.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/3f5c6c5b476b6c0ca2d02480ee50697baf032148a4dc891b29843f07bbac46c3.jpg)

![4069ea6b0b85d6c0b30cc1b67cced95cc74836c68e1b47e1a1e04b18a64a3965.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/4069ea6b0b85d6c0b30cc1b67cced95cc74836c68e1b47e1a1e04b18a64a3965.jpg)

![40fea378d46eb514ed75d1f7fe23a9eb28ebd770d677d7fdbf8c08b502131fbf.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/40fea378d46eb514ed75d1f7fe23a9eb28ebd770d677d7fdbf8c08b502131fbf.jpg)

![45725208c7f3c10718c60dab02f11b9100903dc174a284d577db514c5114d6f6.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/45725208c7f3c10718c60dab02f11b9100903dc174a284d577db514c5114d6f6.jpg)

![47091581f9683166c4441a4bacc8bcab33c42eaa4f706a366c27abbeed96f967.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/47091581f9683166c4441a4bacc8bcab33c42eaa4f706a366c27abbeed96f967.jpg)

![4ae7efb880dc05f1babe403b2953aa9fe1d79b9452bf687260d67a24ffc7fa66.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/4ae7efb880dc05f1babe403b2953aa9fe1d79b9452bf687260d67a24ffc7fa66.jpg)

![4b99f30947e14556fc9bab27b9b983e87ac495d73ab0217cd4bd4300fa90120b.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/4b99f30947e14556fc9bab27b9b983e87ac495d73ab0217cd4bd4300fa90120b.jpg)

![4bc29354e5a7675fa21c7b0b6bc4b580f47fd9c8d7a8ca8265d776e795532d5c.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/4bc29354e5a7675fa21c7b0b6bc4b580f47fd9c8d7a8ca8265d776e795532d5c.jpg)

![4cb9ab938f0a9323d3b6a86c6175c6955a48c0c7986f07c8dfae12c2e2d5d169.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/4cb9ab938f0a9323d3b6a86c6175c6955a48c0c7986f07c8dfae12c2e2d5d169.jpg)

![574889b970fe59c795e167c3708b311c4fc44b44aa7acfe681c73e353f340441.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/574889b970fe59c795e167c3708b311c4fc44b44aa7acfe681c73e353f340441.jpg)

![6270536c85532d195c928500e56f4ebf2e8ad5985ee574006912fa3ab6f040e9.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/6270536c85532d195c928500e56f4ebf2e8ad5985ee574006912fa3ab6f040e9.jpg)

![6c96ea92ddbac9225907d21bd8afc25e1c8115ac150806509db93fecedfe7b25.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/6c96ea92ddbac9225907d21bd8afc25e1c8115ac150806509db93fecedfe7b25.jpg)

![7175cce2c8b5c6cde2f3022a2dbaa2f8f2300aacbf5000ead7293c9e1c5b6798.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/7175cce2c8b5c6cde2f3022a2dbaa2f8f2300aacbf5000ead7293c9e1c5b6798.jpg)

![7dcd8772b9cec93e14b8cd98dd8f267d92b7f859587e80a67e55a0d956ab82cb.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/7dcd8772b9cec93e14b8cd98dd8f267d92b7f859587e80a67e55a0d956ab82cb.jpg)

![7dfcca3c3b0cbe398aebc169b5511159e5507e9fe2750c0b331ecb49fa687598.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/7dfcca3c3b0cbe398aebc169b5511159e5507e9fe2750c0b331ecb49fa687598.jpg)

![80c0959dc2c0de28d11e1d98fe4ad835ac425cac5302514465a813b6a122df7e.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/80c0959dc2c0de28d11e1d98fe4ad835ac425cac5302514465a813b6a122df7e.jpg)

![90dcd4e65b5e7de675ba3fd48fb7069ef2e018cdfcb15b1e1457469b23bd01e7.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/90dcd4e65b5e7de675ba3fd48fb7069ef2e018cdfcb15b1e1457469b23bd01e7.jpg)

![93849d10e734494d706bf8583388d1b30034450c9de6999ee2f855c6cf0ad11b.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/93849d10e734494d706bf8583388d1b30034450c9de6999ee2f855c6cf0ad11b.jpg)

![9d76406e8f64559592fb0f15476041595f1d708349d96dc84a61392d055a667a.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/9d76406e8f64559592fb0f15476041595f1d708349d96dc84a61392d055a667a.jpg)

![9e3ba7a7dda2b4605080329196401d65f6f6c05f522867243a51b301146830b9.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/9e3ba7a7dda2b4605080329196401d65f6f6c05f522867243a51b301146830b9.jpg)

![a27486c80e959f48676bc4cc6c97936d1f7f501edbff9189647c3e6c7afebc13.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/a27486c80e959f48676bc4cc6c97936d1f7f501edbff9189647c3e6c7afebc13.jpg)

![a84212ace39bce5dca82f6b62e912112c7b9b76cf4a00d9f7fbd9f1b1c8e790c.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/a84212ace39bce5dca82f6b62e912112c7b9b76cf4a00d9f7fbd9f1b1c8e790c.jpg)

![aa0a22c54238396097a2b47f854b4d954ade71c4e259dff382488a13cc9f7ea8.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/aa0a22c54238396097a2b47f854b4d954ade71c4e259dff382488a13cc9f7ea8.jpg)

![ab18913abbf7cbd251fdef14f63f39afdc21c4978d9e3fd83e1bcb5efd70f002.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/ab18913abbf7cbd251fdef14f63f39afdc21c4978d9e3fd83e1bcb5efd70f002.jpg)

![b76352dd3174f2ecb16367f01db335568cc57204a5346acd156f5d2944b9fe0e.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/b76352dd3174f2ecb16367f01db335568cc57204a5346acd156f5d2944b9fe0e.jpg)

![c7c80deeff4680817bcf8f86bf5d2c06155b4ba824e5115029834663c1074333.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/c7c80deeff4680817bcf8f86bf5d2c06155b4ba824e5115029834663c1074333.jpg)

![c9101270b674b1b8219f5d816b6863cd6d745ed4ff3fdf70841a9e863e446838.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/c9101270b674b1b8219f5d816b6863cd6d745ed4ff3fdf70841a9e863e446838.jpg)

![df1a9db9d9b32d913e7d8d30f1eaed7b077ee520c4f7c710d4b3a6f92dd06b2a.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/df1a9db9d9b32d913e7d8d30f1eaed7b077ee520c4f7c710d4b3a6f92dd06b2a.jpg)

![e6bdce20b41653bdd9396770dcf8a90d92c6b08e3723f6148cec31f2e66f0c3a.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/e6bdce20b41653bdd9396770dcf8a90d92c6b08e3723f6148cec31f2e66f0c3a.jpg)

![e8716b1a4083a3f491bcfdff40d96c5b7e51a5927c2ede820242b18c717a26dc.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/e8716b1a4083a3f491bcfdff40d96c5b7e51a5927c2ede820242b18c717a26dc.jpg)

![eb4ad9b94bb3d670ea27155a1bc1122ea2c236e9e48ad5d6346300d753b2c4cd.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/eb4ad9b94bb3d670ea27155a1bc1122ea2c236e9e48ad5d6346300d753b2c4cd.jpg)

![f81fcc30cc45fd8b809da3cff1f2a2f5e082e3e6d2eb85c554d767c484381e4d.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/f81fcc30cc45fd8b809da3cff1f2a2f5e082e3e6d2eb85c554d767c484381e4d.jpg)

![f958852097f2922f06869ad98c9d8df209bfbc1d4d5ad1a4f3f0f913205c7c09.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/images/f958852097f2922f06869ad98c9d8df209bfbc1d4d5ad1a4f3f0f913205c7c09.jpg)

### Tables

![81f32c9558da61c54d57ff74ef7988bb0a579c3d1b3a3a4baee326cf5d60f688.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/tables/81f32c9558da61c54d57ff74ef7988bb0a579c3d1b3a3a4baee326cf5d60f688.jpg)

![b961531eeed6ab52b2575f3308d8e339e49fb78dc1aa9b7339d80ddefa0e0acf.jpg](../iclr_results/1504_The%20Optimization%20Landscape%20of%20SGD%20Across%20the%20Feature%20Learning%20Strength/tables/b961531eeed6ab52b2575f3308d8e339e49fb78dc1aa9b7339d80ddefa0e0acf.jpg)

## Spatial-Mamba: Effective Visual State Space Models via Structure-Aware State Fusion


### Images

![0846de80523e1e81f0034055d98467054fa0b0e263891cc8b2f0e71ee04ec546.jpg](../iclr_results/1505_Spatial-Mamba_%20Effective%20Visual%20State%20Space%20Models%20via%20Structure-Aware%20State%20Fusion/images/0846de80523e1e81f0034055d98467054fa0b0e263891cc8b2f0e71ee04ec546.jpg)

![2c15cc231d95f92267732e8ad641704ca2aa39cfa0ff06f42d04b9c700f2b86c.jpg](../iclr_results/1505_Spatial-Mamba_%20Effective%20Visual%20State%20Space%20Models%20via%20Structure-Aware%20State%20Fusion/images/2c15cc231d95f92267732e8ad641704ca2aa39cfa0ff06f42d04b9c700f2b86c.jpg)

![3f88411afdfa9fac6a0c804472162209eee3e30443ffabda9501befa28f2d809.jpg](../iclr_results/1505_Spatial-Mamba_%20Effective%20Visual%20State%20Space%20Models%20via%20Structure-Aware%20State%20Fusion/images/3f88411afdfa9fac6a0c804472162209eee3e30443ffabda9501befa28f2d809.jpg)

![69ba71b988bfdd45fcef724f982758e6964ee7fa9fe8d4783b689eb074d1b561.jpg](../iclr_results/1505_Spatial-Mamba_%20Effective%20Visual%20State%20Space%20Models%20via%20Structure-Aware%20State%20Fusion/images/69ba71b988bfdd45fcef724f982758e6964ee7fa9fe8d4783b689eb074d1b561.jpg)

![9160760588b15058663860efd4fe97731118a30af89053dcf618bb3bfbe1967a.jpg](../iclr_results/1505_Spatial-Mamba_%20Effective%20Visual%20State%20Space%20Models%20via%20Structure-Aware%20State%20Fusion/images/9160760588b15058663860efd4fe97731118a30af89053dcf618bb3bfbe1967a.jpg)

![b8c157c66ce52b138ea490ce58a9f720404c3f0dd9d8f14ee69eba25a00488b0.jpg](../iclr_results/1505_Spatial-Mamba_%20Effective%20Visual%20State%20Space%20Models%20via%20Structure-Aware%20State%20Fusion/images/b8c157c66ce52b138ea490ce58a9f720404c3f0dd9d8f14ee69eba25a00488b0.jpg)

![cf491b2767d84357966386de18261b693127380d2546c96263b3f40ac511d3b0.jpg](../iclr_results/1505_Spatial-Mamba_%20Effective%20Visual%20State%20Space%20Models%20via%20Structure-Aware%20State%20Fusion/images/cf491b2767d84357966386de18261b693127380d2546c96263b3f40ac511d3b0.jpg)

![f9876fecb2b5955cf5a45aacaf9dacaab13ecb07a76dc0b1405c67caa5e2e7a4.jpg](../iclr_results/1505_Spatial-Mamba_%20Effective%20Visual%20State%20Space%20Models%20via%20Structure-Aware%20State%20Fusion/images/f9876fecb2b5955cf5a45aacaf9dacaab13ecb07a76dc0b1405c67caa5e2e7a4.jpg)

![ffad88f93bd7bf99b32b2c6ff3b35a50e9dc61c033cd1546f295e4df10fe3bc9.jpg](../iclr_results/1505_Spatial-Mamba_%20Effective%20Visual%20State%20Space%20Models%20via%20Structure-Aware%20State%20Fusion/images/ffad88f93bd7bf99b32b2c6ff3b35a50e9dc61c033cd1546f295e4df10fe3bc9.jpg)

### Tables

![27e0b891a498e3d76926e45f74070f44f4b992a6ea80c6687734baa00f5d1b2e.jpg](../iclr_results/1505_Spatial-Mamba_%20Effective%20Visual%20State%20Space%20Models%20via%20Structure-Aware%20State%20Fusion/tables/27e0b891a498e3d76926e45f74070f44f4b992a6ea80c6687734baa00f5d1b2e.jpg)

![3a93edb8960d0c9f5d5cebe936559630f34250782c725961b7f314a5bf16e49e.jpg](../iclr_results/1505_Spatial-Mamba_%20Effective%20Visual%20State%20Space%20Models%20via%20Structure-Aware%20State%20Fusion/tables/3a93edb8960d0c9f5d5cebe936559630f34250782c725961b7f314a5bf16e49e.jpg)

![520cdca51822ed33dcdd2fdb082cf8f569b24b9e0e4614daa13e796225d8fd9e.jpg](../iclr_results/1505_Spatial-Mamba_%20Effective%20Visual%20State%20Space%20Models%20via%20Structure-Aware%20State%20Fusion/tables/520cdca51822ed33dcdd2fdb082cf8f569b24b9e0e4614daa13e796225d8fd9e.jpg)

![67df5570b2b39d3df94734319ca379e652842c9e8c48fa1409131084fde32b61.jpg](../iclr_results/1505_Spatial-Mamba_%20Effective%20Visual%20State%20Space%20Models%20via%20Structure-Aware%20State%20Fusion/tables/67df5570b2b39d3df94734319ca379e652842c9e8c48fa1409131084fde32b61.jpg)

![7b46c354b37906bed6a27214d05278ca54945b8f7344f4cada68f182cfda7ec1.jpg](../iclr_results/1505_Spatial-Mamba_%20Effective%20Visual%20State%20Space%20Models%20via%20Structure-Aware%20State%20Fusion/tables/7b46c354b37906bed6a27214d05278ca54945b8f7344f4cada68f182cfda7ec1.jpg)

![97cb6c07432044ab312cc2a5f37706bff0562bc6db41d1bfeb9642fb4849ed72.jpg](../iclr_results/1505_Spatial-Mamba_%20Effective%20Visual%20State%20Space%20Models%20via%20Structure-Aware%20State%20Fusion/tables/97cb6c07432044ab312cc2a5f37706bff0562bc6db41d1bfeb9642fb4849ed72.jpg)

![ee5d1943c6bf3c1f275abf8ed46c8075953c33d3b1f1cd20297a4bd772c4014d.jpg](../iclr_results/1505_Spatial-Mamba_%20Effective%20Visual%20State%20Space%20Models%20via%20Structure-Aware%20State%20Fusion/tables/ee5d1943c6bf3c1f275abf8ed46c8075953c33d3b1f1cd20297a4bd772c4014d.jpg)

![f3eb640d8be971f0d2127e6eb546150be18b022259e22530be0201508e7f67cb.jpg](../iclr_results/1505_Spatial-Mamba_%20Effective%20Visual%20State%20Space%20Models%20via%20Structure-Aware%20State%20Fusion/tables/f3eb640d8be971f0d2127e6eb546150be18b022259e22530be0201508e7f67cb.jpg)

## EcoFace: Audio-Visual Emotional Co-Disentanglement Speech-Driven 3D Talking Face Generation


### Images

![1dde3a5b3f80f10d9b7496146a6a68fbd45c0fa68bf90919ed610f9f8ff280a0.jpg](../iclr_results/1506_EcoFace_%20Audio-Visual%20Emotional%20Co-Disentanglement%20Speech-Driven%203D%20Talking%20Face%20Generation/images/1dde3a5b3f80f10d9b7496146a6a68fbd45c0fa68bf90919ed610f9f8ff280a0.jpg)

![5d51ce989ee9688a3e50a69654ed3823762a44fae4158824087d06091f264d80.jpg](../iclr_results/1506_EcoFace_%20Audio-Visual%20Emotional%20Co-Disentanglement%20Speech-Driven%203D%20Talking%20Face%20Generation/images/5d51ce989ee9688a3e50a69654ed3823762a44fae4158824087d06091f264d80.jpg)

![6a1d76514840d97183d9e2ffb20789a882b8dd32140ff082d6c566802785e661.jpg](../iclr_results/1506_EcoFace_%20Audio-Visual%20Emotional%20Co-Disentanglement%20Speech-Driven%203D%20Talking%20Face%20Generation/images/6a1d76514840d97183d9e2ffb20789a882b8dd32140ff082d6c566802785e661.jpg)

![6e987993bb30afcd64339e47d1e3014d404a978b6d5afbaf29cabbb710f44591.jpg](../iclr_results/1506_EcoFace_%20Audio-Visual%20Emotional%20Co-Disentanglement%20Speech-Driven%203D%20Talking%20Face%20Generation/images/6e987993bb30afcd64339e47d1e3014d404a978b6d5afbaf29cabbb710f44591.jpg)

![832c0a3c02e6c3764554337f434f79eb4b0d9c1ef9cd20d3c5bf1cd3be8070c5.jpg](../iclr_results/1506_EcoFace_%20Audio-Visual%20Emotional%20Co-Disentanglement%20Speech-Driven%203D%20Talking%20Face%20Generation/images/832c0a3c02e6c3764554337f434f79eb4b0d9c1ef9cd20d3c5bf1cd3be8070c5.jpg)

![89e4c81c7425aaf91399e506a24b137af96284fa61f25ab744d1d5fb9ad75f42.jpg](../iclr_results/1506_EcoFace_%20Audio-Visual%20Emotional%20Co-Disentanglement%20Speech-Driven%203D%20Talking%20Face%20Generation/images/89e4c81c7425aaf91399e506a24b137af96284fa61f25ab744d1d5fb9ad75f42.jpg)

![952f50e6ab7ee99aea68ab18e30c4f456af08c0a5193d4df93c29281115e7ace.jpg](../iclr_results/1506_EcoFace_%20Audio-Visual%20Emotional%20Co-Disentanglement%20Speech-Driven%203D%20Talking%20Face%20Generation/images/952f50e6ab7ee99aea68ab18e30c4f456af08c0a5193d4df93c29281115e7ace.jpg)

![afd24fd83c0559c741858208ca2d4a8a3601cb55214dc625be2feb80af3aa6f9.jpg](../iclr_results/1506_EcoFace_%20Audio-Visual%20Emotional%20Co-Disentanglement%20Speech-Driven%203D%20Talking%20Face%20Generation/images/afd24fd83c0559c741858208ca2d4a8a3601cb55214dc625be2feb80af3aa6f9.jpg)

![c0c5511a5ee3f18792dc31ffd42aef6592b9fb4e86d73898f58daadb377d186c.jpg](../iclr_results/1506_EcoFace_%20Audio-Visual%20Emotional%20Co-Disentanglement%20Speech-Driven%203D%20Talking%20Face%20Generation/images/c0c5511a5ee3f18792dc31ffd42aef6592b9fb4e86d73898f58daadb377d186c.jpg)

![cc419671136ab6c02b9f5b3bcc31bc2c2319510989871318ec027969c9e1b65a.jpg](../iclr_results/1506_EcoFace_%20Audio-Visual%20Emotional%20Co-Disentanglement%20Speech-Driven%203D%20Talking%20Face%20Generation/images/cc419671136ab6c02b9f5b3bcc31bc2c2319510989871318ec027969c9e1b65a.jpg)

![de0715155c85063c28431af7f4754e2d243fe6a25c2b4bb3240b3c671674cb00.jpg](../iclr_results/1506_EcoFace_%20Audio-Visual%20Emotional%20Co-Disentanglement%20Speech-Driven%203D%20Talking%20Face%20Generation/images/de0715155c85063c28431af7f4754e2d243fe6a25c2b4bb3240b3c671674cb00.jpg)

![e276ed7bc5dc57a69359199381baa8e7f1ccca04e3d5f8824f3e07c61444dacc.jpg](../iclr_results/1506_EcoFace_%20Audio-Visual%20Emotional%20Co-Disentanglement%20Speech-Driven%203D%20Talking%20Face%20Generation/images/e276ed7bc5dc57a69359199381baa8e7f1ccca04e3d5f8824f3e07c61444dacc.jpg)

![e80baaef833c931c9d23096c04c594b46dea4666eb60a9a3559546325bc731e1.jpg](../iclr_results/1506_EcoFace_%20Audio-Visual%20Emotional%20Co-Disentanglement%20Speech-Driven%203D%20Talking%20Face%20Generation/images/e80baaef833c931c9d23096c04c594b46dea4666eb60a9a3559546325bc731e1.jpg)

![f724b0e8bf4b57f3820c7c668b05bf8e3f883e91842b29b7267e107e47cbbc30.jpg](../iclr_results/1506_EcoFace_%20Audio-Visual%20Emotional%20Co-Disentanglement%20Speech-Driven%203D%20Talking%20Face%20Generation/images/f724b0e8bf4b57f3820c7c668b05bf8e3f883e91842b29b7267e107e47cbbc30.jpg)

![f906488e244c28813110de4945c4b4a0aecc8c05f6b28029a86ceee9cf7bd636.jpg](../iclr_results/1506_EcoFace_%20Audio-Visual%20Emotional%20Co-Disentanglement%20Speech-Driven%203D%20Talking%20Face%20Generation/images/f906488e244c28813110de4945c4b4a0aecc8c05f6b28029a86ceee9cf7bd636.jpg)

### Tables

![021cac7360c958c26a40c97df5953f38fdfdb97b5010988a593faf56991d8fd6.jpg](../iclr_results/1506_EcoFace_%20Audio-Visual%20Emotional%20Co-Disentanglement%20Speech-Driven%203D%20Talking%20Face%20Generation/tables/021cac7360c958c26a40c97df5953f38fdfdb97b5010988a593faf56991d8fd6.jpg)

![2de6c03d5155e6f6e22c3a980d95698a046b4b72a7c1359ed64d76e593e1dd11.jpg](../iclr_results/1506_EcoFace_%20Audio-Visual%20Emotional%20Co-Disentanglement%20Speech-Driven%203D%20Talking%20Face%20Generation/tables/2de6c03d5155e6f6e22c3a980d95698a046b4b72a7c1359ed64d76e593e1dd11.jpg)

![3d044fefdf9c8bc0269dde0983b9a5c3ea3f8edc86eacf2aeec61fcdaf16b534.jpg](../iclr_results/1506_EcoFace_%20Audio-Visual%20Emotional%20Co-Disentanglement%20Speech-Driven%203D%20Talking%20Face%20Generation/tables/3d044fefdf9c8bc0269dde0983b9a5c3ea3f8edc86eacf2aeec61fcdaf16b534.jpg)

![af141db6ec00a8704d6be64df477b1cb04cec06e95bd6b27831b7053bd4c6467.jpg](../iclr_results/1506_EcoFace_%20Audio-Visual%20Emotional%20Co-Disentanglement%20Speech-Driven%203D%20Talking%20Face%20Generation/tables/af141db6ec00a8704d6be64df477b1cb04cec06e95bd6b27831b7053bd4c6467.jpg)

![afc58b7bd6022e5f95e912da95e203d9a3374e3794644f13c63ce50fdcd79399.jpg](../iclr_results/1506_EcoFace_%20Audio-Visual%20Emotional%20Co-Disentanglement%20Speech-Driven%203D%20Talking%20Face%20Generation/tables/afc58b7bd6022e5f95e912da95e203d9a3374e3794644f13c63ce50fdcd79399.jpg)

## Adam-mini: Use Fewer Learning Rates To Gain More


### Images

![035a0d257b04fa3b96c40a099bcad6c3da8eb14dd9682f3e25961932513a823c.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/images/035a0d257b04fa3b96c40a099bcad6c3da8eb14dd9682f3e25961932513a823c.jpg)

![09215fc57e719c5fffd4c6a1610e0d6b8ab1c4d52dd8339c9cc6394f6e06eb8a.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/images/09215fc57e719c5fffd4c6a1610e0d6b8ab1c4d52dd8339c9cc6394f6e06eb8a.jpg)

![09b31f85c84d4088d0b6e6c65c0f78dd3cf70865e70440b271fd1f38abdf9066.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/images/09b31f85c84d4088d0b6e6c65c0f78dd3cf70865e70440b271fd1f38abdf9066.jpg)

![2bcfd0aaa993ed54ff7a7cca1938b75a067e99b83b343162ec3b694bd22428e8.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/images/2bcfd0aaa993ed54ff7a7cca1938b75a067e99b83b343162ec3b694bd22428e8.jpg)

![311372bcbedaad82dea94fc1521fc3d4512028ff2687844935bb17d901f6e5e7.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/images/311372bcbedaad82dea94fc1521fc3d4512028ff2687844935bb17d901f6e5e7.jpg)

![3488acc48f0eb71634d4cff25215ea1f6fa35bdd26913afb3c834f80b84d9367.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/images/3488acc48f0eb71634d4cff25215ea1f6fa35bdd26913afb3c834f80b84d9367.jpg)

![36080294567029c5d96e3d1f8033b2a5294f5c0fdcd21aee3791ac7a02463354.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/images/36080294567029c5d96e3d1f8033b2a5294f5c0fdcd21aee3791ac7a02463354.jpg)

![3ecce1eeb23fe8a9c0b01c6ab69e9c149c638f615ab9a404d0468329ace77ba7.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/images/3ecce1eeb23fe8a9c0b01c6ab69e9c149c638f615ab9a404d0468329ace77ba7.jpg)

![43abea57cd94d559f72975be4658a3b2f01f85bcdf8f38a7b78c54e9f8e730c8.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/images/43abea57cd94d559f72975be4658a3b2f01f85bcdf8f38a7b78c54e9f8e730c8.jpg)

![4e24d96b4dca39e76add4feff8f101aa09f95d9a77ef953ab0f55bc0abd18a90.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/images/4e24d96b4dca39e76add4feff8f101aa09f95d9a77ef953ab0f55bc0abd18a90.jpg)

![7f5f4bab89555dec568c65f71b5da055cacfec24ba512348655d781a0825924b.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/images/7f5f4bab89555dec568c65f71b5da055cacfec24ba512348655d781a0825924b.jpg)

![9dbbbf620702f1693972c6ad38051a70f16351b01ae043cfc4fcc7733725ad70.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/images/9dbbbf620702f1693972c6ad38051a70f16351b01ae043cfc4fcc7733725ad70.jpg)

![a1ec2a215953c0b713a482dbf464fe80506aa7f76e6820259fe2e9e82a8fdc64.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/images/a1ec2a215953c0b713a482dbf464fe80506aa7f76e6820259fe2e9e82a8fdc64.jpg)

![a39055b43a2b3cf8dfbfeeda94b7a7d4dc79dd5216de46db01ad67923bf55a4b.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/images/a39055b43a2b3cf8dfbfeeda94b7a7d4dc79dd5216de46db01ad67923bf55a4b.jpg)

![ae79980a7320e2f04d00a65f6f6a754822895e067ae0581c38489bd301f2f22a.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/images/ae79980a7320e2f04d00a65f6f6a754822895e067ae0581c38489bd301f2f22a.jpg)

![cf61f9d30856e9a7dec12e25c282afabbc9454ca21e2c68088d9428c82fdf0b8.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/images/cf61f9d30856e9a7dec12e25c282afabbc9454ca21e2c68088d9428c82fdf0b8.jpg)

![dc1f265f10146f0f21daddb67c1198855d691a48aa95ec69eafb95f2ed51fa54.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/images/dc1f265f10146f0f21daddb67c1198855d691a48aa95ec69eafb95f2ed51fa54.jpg)

![de4ffb3f81d9d974e8e89a98f9f13527bab93d0fac409093c5731e808b8ac000.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/images/de4ffb3f81d9d974e8e89a98f9f13527bab93d0fac409093c5731e808b8ac000.jpg)

![e25382eebb4630e850bab53fe1abc288bd0e543c86cc6ef2e4221363c621be2a.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/images/e25382eebb4630e850bab53fe1abc288bd0e543c86cc6ef2e4221363c621be2a.jpg)

![e7729fd6423d1ccebc6803439a6c28eca4a377d730075f0c2ee055a218be7a6e.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/images/e7729fd6423d1ccebc6803439a6c28eca4a377d730075f0c2ee055a218be7a6e.jpg)

![ee071da4b5c8af4fd69fe60bd98692c4351811f4bdbed06f7286c20116c6498e.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/images/ee071da4b5c8af4fd69fe60bd98692c4351811f4bdbed06f7286c20116c6498e.jpg)

![f35b2f227d1821bf5a79245b17a9b7e2e4429bd0748c4df879865b3a9ad39ac8.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/images/f35b2f227d1821bf5a79245b17a9b7e2e4429bd0748c4df879865b3a9ad39ac8.jpg)

![f733bea0fda8587a79aa6ca582cb6ff364fec5e7f3340b6ca73c43ed38326f41.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/images/f733bea0fda8587a79aa6ca582cb6ff364fec5e7f3340b6ca73c43ed38326f41.jpg)

### Tables

![0070e805b38edd2fa5c813c1c1b3d3ac55ed8b371866b4ddf7d3b7ee5452152b.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/tables/0070e805b38edd2fa5c813c1c1b3d3ac55ed8b371866b4ddf7d3b7ee5452152b.jpg)

![183c9993c7d26870870b9a08fd150a218dd48d0c8a2deaa97f427a97caeb9518.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/tables/183c9993c7d26870870b9a08fd150a218dd48d0c8a2deaa97f427a97caeb9518.jpg)

![3076aedd9eb530d6b3ae57ae5447cfd0be53ec67fb3af3cd2fa6841988f688c0.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/tables/3076aedd9eb530d6b3ae57ae5447cfd0be53ec67fb3af3cd2fa6841988f688c0.jpg)

![3d0c6c2aaf2afbadcae92594872267dc6b3a5610a892c1fb30248ad339ec32cc.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/tables/3d0c6c2aaf2afbadcae92594872267dc6b3a5610a892c1fb30248ad339ec32cc.jpg)

![4171247c1953f61bfbfd5c863dcfbdea77b84385e745ad28f0a40c041d8d21eb.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/tables/4171247c1953f61bfbfd5c863dcfbdea77b84385e745ad28f0a40c041d8d21eb.jpg)

![6bf49d2bf4d6546ea5954949a5b4c0b596de820e7d171abfb76c75dbc6f6fe20.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/tables/6bf49d2bf4d6546ea5954949a5b4c0b596de820e7d171abfb76c75dbc6f6fe20.jpg)

![6e6794d1b3d06567faeb80b798bebade8a56dd9cf0bc612e82e5601cea73f3d7.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/tables/6e6794d1b3d06567faeb80b798bebade8a56dd9cf0bc612e82e5601cea73f3d7.jpg)

![778f67a7f221c41c348d05b26ea175bdb249d923dd1802bf4d705800a1315149.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/tables/778f67a7f221c41c348d05b26ea175bdb249d923dd1802bf4d705800a1315149.jpg)

![7ca1541681d0c2ecf5a9912d095199e6ad4a805ebe53cdd294521e454ca4c588.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/tables/7ca1541681d0c2ecf5a9912d095199e6ad4a805ebe53cdd294521e454ca4c588.jpg)

![9500b15dea4b7ca4601971be9856ade5a6a2c11ca09fe9f4ec6d600d7ad888a9.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/tables/9500b15dea4b7ca4601971be9856ade5a6a2c11ca09fe9f4ec6d600d7ad888a9.jpg)

![c6293782b1d2d90cbdfee1b78499c250af0933406bc44e1efa3cb1be9c496455.jpg](../iclr_results/1507_Adam-mini_%20Use%20Fewer%20Learning%20Rates%20To%20Gain%20More/tables/c6293782b1d2d90cbdfee1b78499c250af0933406bc44e1efa3cb1be9c496455.jpg)

## Improving Graph Neural Networks by Learning Continuous Edge Directions


### Images

![18730782d3a65a1bb4746d1abf4387beed3b74f4814d6fae6053de40e4df3485.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/images/18730782d3a65a1bb4746d1abf4387beed3b74f4814d6fae6053de40e4df3485.jpg)

![1e0dfd6a1d021ab9049e754daeaa93325bdb30c00914038d57047e072e9ae9fa.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/images/1e0dfd6a1d021ab9049e754daeaa93325bdb30c00914038d57047e072e9ae9fa.jpg)

![2b74f1849518a88117c6358282325ce0ddfc314474790d2385052d258c064132.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/images/2b74f1849518a88117c6358282325ce0ddfc314474790d2385052d258c064132.jpg)

![2b9e793fd902b06a5b663a7ed79e3b26361b7b9942270378aa149966bdb70642.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/images/2b9e793fd902b06a5b663a7ed79e3b26361b7b9942270378aa149966bdb70642.jpg)

![7c7dca3c69aa5eda919bc2fc723d28d0a1b1ffe91eac7ed9ee41d50d27e2dab8.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/images/7c7dca3c69aa5eda919bc2fc723d28d0a1b1ffe91eac7ed9ee41d50d27e2dab8.jpg)

![7f56282cc1cb533582860780f7bc545c37d7961aa177c4b4e7fa42fd6aa05884.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/images/7f56282cc1cb533582860780f7bc545c37d7961aa177c4b4e7fa42fd6aa05884.jpg)

![91108532298c59e8359c6ab5b020ed4fd523abf81f08e49b0bf2f2a4dc8b74d2.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/images/91108532298c59e8359c6ab5b020ed4fd523abf81f08e49b0bf2f2a4dc8b74d2.jpg)

![a8adda6e1f425b3ac630276ddf4b1f0a9d9766b799e9c6deba72539f45ac37dc.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/images/a8adda6e1f425b3ac630276ddf4b1f0a9d9766b799e9c6deba72539f45ac37dc.jpg)

![aa9871acba0b458d40d639762285891cecec15d6bb46956de92d217e30269990.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/images/aa9871acba0b458d40d639762285891cecec15d6bb46956de92d217e30269990.jpg)

![aefb4e7127ca8120c9fd0933c083e0e8096c7a0f9a329e30b81c1326776f4a6c.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/images/aefb4e7127ca8120c9fd0933c083e0e8096c7a0f9a329e30b81c1326776f4a6c.jpg)

![afb0ac5a1c07505fec35c809c592614e1d2e46761fc44126d95291cc72b8d538.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/images/afb0ac5a1c07505fec35c809c592614e1d2e46761fc44126d95291cc72b8d538.jpg)

![b63edf8efff1b28e4fa7d6ee626e24f7e2210afb940c9158ef0349d615659b72.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/images/b63edf8efff1b28e4fa7d6ee626e24f7e2210afb940c9158ef0349d615659b72.jpg)

![ec0a08cbdf9e06037ef158482146e2610ce70594a7fefbb9d7957c173cfd0a86.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/images/ec0a08cbdf9e06037ef158482146e2610ce70594a7fefbb9d7957c173cfd0a86.jpg)

### Tables

![0793f7b56b90c196f1f7999b93ceb101a94bdd24ce5a36bd4aa4495605a6149b.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/tables/0793f7b56b90c196f1f7999b93ceb101a94bdd24ce5a36bd4aa4495605a6149b.jpg)

![19bb27e4d1476f8f8718f49b8a372873153de193f4124ed0390b06228d4fefe0.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/tables/19bb27e4d1476f8f8718f49b8a372873153de193f4124ed0390b06228d4fefe0.jpg)

![237cf9815ed3113677688b3b18960c2c1cec88dea46413502206fca02a93b3cb.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/tables/237cf9815ed3113677688b3b18960c2c1cec88dea46413502206fca02a93b3cb.jpg)

![28047182ed50e7c2d1d69f77106b2df8bbb2055130b7699d90ec36141b6c3d55.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/tables/28047182ed50e7c2d1d69f77106b2df8bbb2055130b7699d90ec36141b6c3d55.jpg)

![2f5a7a25c8636c91c34a82bbd2d3c2881869178a0f36edee2424ee7cf97809b3.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/tables/2f5a7a25c8636c91c34a82bbd2d3c2881869178a0f36edee2424ee7cf97809b3.jpg)

![398545c7355ef3b9a6b3b94996b692d635db12666e86dcf086b612c234d9426b.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/tables/398545c7355ef3b9a6b3b94996b692d635db12666e86dcf086b612c234d9426b.jpg)

![4c059bf7494730c49a6f6f54bb1beb6e7635af2f6ed268433268fba250faf428.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/tables/4c059bf7494730c49a6f6f54bb1beb6e7635af2f6ed268433268fba250faf428.jpg)

![5d2308be50f3823c578dd90ec43b53271ec4d62a5e7c9de1506c3c13ed4ea56d.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/tables/5d2308be50f3823c578dd90ec43b53271ec4d62a5e7c9de1506c3c13ed4ea56d.jpg)

![612fb8bfa0c3a2f6e7a0b6bb43ae9d81beeadee83d3128c69c133d83b0466088.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/tables/612fb8bfa0c3a2f6e7a0b6bb43ae9d81beeadee83d3128c69c133d83b0466088.jpg)

![616210be30d75631bde9735a645d09d58d507f96576d07a4a76cec896b54e8ef.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/tables/616210be30d75631bde9735a645d09d58d507f96576d07a4a76cec896b54e8ef.jpg)

![8da524855084ab7b8a8b03d3d8c23a3e1ce0e985d4c3af4ef4cc54d9a5fafac5.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/tables/8da524855084ab7b8a8b03d3d8c23a3e1ce0e985d4c3af4ef4cc54d9a5fafac5.jpg)

![904416126077da831c5f6f41f624b33c90be799dd6e8698eab0911441e9c0af3.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/tables/904416126077da831c5f6f41f624b33c90be799dd6e8698eab0911441e9c0af3.jpg)

![9185a8aaf777030dd4f014b0a8e01f60c69da872307debf43df43b02969fd50b.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/tables/9185a8aaf777030dd4f014b0a8e01f60c69da872307debf43df43b02969fd50b.jpg)

![a733ec9bd567e1fe90052da1d3627e428622aec4eaa8084028ed1a69d1118a62.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/tables/a733ec9bd567e1fe90052da1d3627e428622aec4eaa8084028ed1a69d1118a62.jpg)

![df426ae57b39a2ebb0b3abbcc115397cb1a3efc49b4cdb6a56902844d7329de1.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/tables/df426ae57b39a2ebb0b3abbcc115397cb1a3efc49b4cdb6a56902844d7329de1.jpg)

![ebd49b2c16c2e25499d855368979e681a02cb6c30e8c9b6452ef6555f502d6ae.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/tables/ebd49b2c16c2e25499d855368979e681a02cb6c30e8c9b6452ef6555f502d6ae.jpg)

![f8127ef12aa69b585145536d34bd50807f94592a2abfb533b706a0054ee3679a.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/tables/f8127ef12aa69b585145536d34bd50807f94592a2abfb533b706a0054ee3679a.jpg)

![fe81719660d63248ade4c5e83aadef592c9e94a0563e6a5be65b0ea21007a342.jpg](../iclr_results/1508_Improving%20Graph%20Neural%20Networks%20by%20Learning%20Continuous%20Edge%20Directions/tables/fe81719660d63248ade4c5e83aadef592c9e94a0563e6a5be65b0ea21007a342.jpg)

## MuPT: A Generative Symbolic Music Pretrained Transformer


### Images

![03e5e05af504bbe7c62128e0711358c900f1c1ce4e94431aeda2f47d7d1fefd9.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/images/03e5e05af504bbe7c62128e0711358c900f1c1ce4e94431aeda2f47d7d1fefd9.jpg)

![068e476b254606b93a7814e6640823fbd61665846deb0791940d6a5ab33449d9.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/images/068e476b254606b93a7814e6640823fbd61665846deb0791940d6a5ab33449d9.jpg)

![1003fdc83d58931daa2d76a583d8dce0d600f654dedbd2b69edd1496761cb4b7.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/images/1003fdc83d58931daa2d76a583d8dce0d600f654dedbd2b69edd1496761cb4b7.jpg)

![1bfbb90c02f9468fbba799347f597019ecf326aca552d6376ed095048acaca9f.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/images/1bfbb90c02f9468fbba799347f597019ecf326aca552d6376ed095048acaca9f.jpg)

![221840c9592e37cddd08244a30b998621b40df909ee02fcbd743e52473069e88.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/images/221840c9592e37cddd08244a30b998621b40df909ee02fcbd743e52473069e88.jpg)

![2ccb7f7ddb810360634406cd3460879eff083796989f26bd615f9f8a26d33c79.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/images/2ccb7f7ddb810360634406cd3460879eff083796989f26bd615f9f8a26d33c79.jpg)

![2e40c7831280a4cee44277f4c754d50e5b6a2e7d342cdec070f4894f219b5704.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/images/2e40c7831280a4cee44277f4c754d50e5b6a2e7d342cdec070f4894f219b5704.jpg)

![3651123d6f51b46501bed31467f9b0f93cfa48260dbbe662ca6404ceed0ed2db.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/images/3651123d6f51b46501bed31467f9b0f93cfa48260dbbe662ca6404ceed0ed2db.jpg)

![36a60735ef7fce4f9dccf34ca727730c279f1001dcb6236535a976cda1481da6.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/images/36a60735ef7fce4f9dccf34ca727730c279f1001dcb6236535a976cda1481da6.jpg)

![524a063d99fc4d4f5fd1a88ed1218b6835d03dbe0665a246673d384738e63505.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/images/524a063d99fc4d4f5fd1a88ed1218b6835d03dbe0665a246673d384738e63505.jpg)

![666127e219c617f5ee689d7932c262fc3369b9bceb23f1f879839001fac17de7.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/images/666127e219c617f5ee689d7932c262fc3369b9bceb23f1f879839001fac17de7.jpg)

![89817bac4be1a0152d0e946c50d463563670ce94aef1a9c286fb358f28150b9a.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/images/89817bac4be1a0152d0e946c50d463563670ce94aef1a9c286fb358f28150b9a.jpg)

![b7461be39307b3ac3b3dee85542609b2421bceba6ca3b1d05511a9e62f8aa85a.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/images/b7461be39307b3ac3b3dee85542609b2421bceba6ca3b1d05511a9e62f8aa85a.jpg)

![e80ab2230bb58ad892ac8357b9f0374825e1718f70388acaad59f41663d5b10a.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/images/e80ab2230bb58ad892ac8357b9f0374825e1718f70388acaad59f41663d5b10a.jpg)

### Tables

![0314c9ba05584d6f23e4e4fd353254fca7b794caf4dbbea7f357042cf3745642.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/tables/0314c9ba05584d6f23e4e4fd353254fca7b794caf4dbbea7f357042cf3745642.jpg)

![102df89a6b44122590ede315f77f9a40a630b1dcfa403a791e7064dfdc2c94c8.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/tables/102df89a6b44122590ede315f77f9a40a630b1dcfa403a791e7064dfdc2c94c8.jpg)

![16d7cf4f0e8d79e999bf27ab9da3eacd115ea723bdc3bb709bdf6c00d2d7fad1.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/tables/16d7cf4f0e8d79e999bf27ab9da3eacd115ea723bdc3bb709bdf6c00d2d7fad1.jpg)

![2058e34b4ab01de0a827fbb65b64ea28de917274fc2c355da2ae7cc753f115d2.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/tables/2058e34b4ab01de0a827fbb65b64ea28de917274fc2c355da2ae7cc753f115d2.jpg)

![409f2f08e814fdd080830e47073a1bc7b195afe57cee32e59887ebc5e455055d.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/tables/409f2f08e814fdd080830e47073a1bc7b195afe57cee32e59887ebc5e455055d.jpg)

![4a3b607450d7fc9e6eb87b7e2d99c67b99c6945b2e8d968e6a57b65bd1a57308.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/tables/4a3b607450d7fc9e6eb87b7e2d99c67b99c6945b2e8d968e6a57b65bd1a57308.jpg)

![4bec4e2f1f1ec11ab3ecd57c5f1b1b300d5f3529d7118286ca0ee9c591fcd3bb.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/tables/4bec4e2f1f1ec11ab3ecd57c5f1b1b300d5f3529d7118286ca0ee9c591fcd3bb.jpg)

![520d081ed58e6f48798aae1cd1a2d063e813294b6cbc1d83b42d37df19f180af.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/tables/520d081ed58e6f48798aae1cd1a2d063e813294b6cbc1d83b42d37df19f180af.jpg)

![80d424e37a38007ad8c11ac5514c20e544c017cda564817e298760b7797e890a.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/tables/80d424e37a38007ad8c11ac5514c20e544c017cda564817e298760b7797e890a.jpg)

![82f6bfdfd0a160fe4daecc04be61ba3af72f13a1ebd636a11ef0da5656a685a1.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/tables/82f6bfdfd0a160fe4daecc04be61ba3af72f13a1ebd636a11ef0da5656a685a1.jpg)

![a166585ec4dd4611826bddc9f466456e81483391f573404a87d6335732f4958a.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/tables/a166585ec4dd4611826bddc9f466456e81483391f573404a87d6335732f4958a.jpg)

![ecc15b783711df05c7dae550daf1dfc76b069276615743bf8d204e006d007c6c.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/tables/ecc15b783711df05c7dae550daf1dfc76b069276615743bf8d204e006d007c6c.jpg)

![f5de6507e1eb6094344d41b1b531858ed24cfc10c894352fbde3064e34a5ebbb.jpg](../iclr_results/1509_MuPT_%20A%20Generative%20Symbolic%20Music%20Pretrained%20Transformer/tables/f5de6507e1eb6094344d41b1b531858ed24cfc10c894352fbde3064e34a5ebbb.jpg)

## Residual Connections and Normalization Can Provably Prevent Oversmoothing in GNNs


### Images

![a5d0997c7e1d0d018842231be9595e4b68666590ab3c77a561c6f5e68f87c5bc.jpg](../iclr_results/1510_Residual%20Connections%20and%20Normalization%20Can%20Provably%20Prevent%20Oversmoothing%20in%20GNNs/images/a5d0997c7e1d0d018842231be9595e4b68666590ab3c77a561c6f5e68f87c5bc.jpg)

![c9ea3a8ce555dd1f3aad4239acb98a67cfaedd601064efcc4d07a3c229688213.jpg](../iclr_results/1510_Residual%20Connections%20and%20Normalization%20Can%20Provably%20Prevent%20Oversmoothing%20in%20GNNs/images/c9ea3a8ce555dd1f3aad4239acb98a67cfaedd601064efcc4d07a3c229688213.jpg)

### Tables

![0e8c0c63d6d57279e9c00642061c55022e304fb9e0e7dff9234467cc014a8a37.jpg](../iclr_results/1510_Residual%20Connections%20and%20Normalization%20Can%20Provably%20Prevent%20Oversmoothing%20in%20GNNs/tables/0e8c0c63d6d57279e9c00642061c55022e304fb9e0e7dff9234467cc014a8a37.jpg)

![57f7c8b54f1b797ce949fd70c7afc6398ed8ac711daca98bf62002d1f6583e9d.jpg](../iclr_results/1510_Residual%20Connections%20and%20Normalization%20Can%20Provably%20Prevent%20Oversmoothing%20in%20GNNs/tables/57f7c8b54f1b797ce949fd70c7afc6398ed8ac711daca98bf62002d1f6583e9d.jpg)

![d8ce54242984b8be3455bbaa043faf734607f51622c986378fb4b02cbcbd44c7.jpg](../iclr_results/1510_Residual%20Connections%20and%20Normalization%20Can%20Provably%20Prevent%20Oversmoothing%20in%20GNNs/tables/d8ce54242984b8be3455bbaa043faf734607f51622c986378fb4b02cbcbd44c7.jpg)

![e32a5e0efdc80a8838d1091f5fe76e8c6ed62df66f86652bd16d7b18d398f005.jpg](../iclr_results/1510_Residual%20Connections%20and%20Normalization%20Can%20Provably%20Prevent%20Oversmoothing%20in%20GNNs/tables/e32a5e0efdc80a8838d1091f5fe76e8c6ed62df66f86652bd16d7b18d398f005.jpg)

## Minimax Optimal Reinforcement Learning with Quasi-Optimism


### Images

![1dd2a00d75e4387885cc755c492bbf8630fd959bf4704359fd4acc57fe2ddc15.jpg](../iclr_results/1511_Minimax%20Optimal%20Reinforcement%20Learning%20with%20Quasi-Optimism/images/1dd2a00d75e4387885cc755c492bbf8630fd959bf4704359fd4acc57fe2ddc15.jpg)

![426e61db37ca88b1aa6ee3b4306d5022bef7d00e1706cfdafd5807f657ee6481.jpg](../iclr_results/1511_Minimax%20Optimal%20Reinforcement%20Learning%20with%20Quasi-Optimism/images/426e61db37ca88b1aa6ee3b4306d5022bef7d00e1706cfdafd5807f657ee6481.jpg)

![4cef2a87ea479aedd3307479673430061ed0befd044118f056857f84faa863b9.jpg](../iclr_results/1511_Minimax%20Optimal%20Reinforcement%20Learning%20with%20Quasi-Optimism/images/4cef2a87ea479aedd3307479673430061ed0befd044118f056857f84faa863b9.jpg)

![614fb71ffef66c146b56c25c8cfb0b0501bce95a2ab77ab85d4f29a8509e9172.jpg](../iclr_results/1511_Minimax%20Optimal%20Reinforcement%20Learning%20with%20Quasi-Optimism/images/614fb71ffef66c146b56c25c8cfb0b0501bce95a2ab77ab85d4f29a8509e9172.jpg)

![b5b41760e945db5a73b1941e7ef872f7ead7607ce6ce5d3338d6e925a993b723.jpg](../iclr_results/1511_Minimax%20Optimal%20Reinforcement%20Learning%20with%20Quasi-Optimism/images/b5b41760e945db5a73b1941e7ef872f7ead7607ce6ce5d3338d6e925a993b723.jpg)

![d176d8db6af9e63c10c0c37dc606d67159dd4041895342d36cc15741b35f0b70.jpg](../iclr_results/1511_Minimax%20Optimal%20Reinforcement%20Learning%20with%20Quasi-Optimism/images/d176d8db6af9e63c10c0c37dc606d67159dd4041895342d36cc15741b35f0b70.jpg)

![f65d3342387513c597a5569aca39dff19c22ae232a2f119535facbbad49e2852.jpg](../iclr_results/1511_Minimax%20Optimal%20Reinforcement%20Learning%20with%20Quasi-Optimism/images/f65d3342387513c597a5569aca39dff19c22ae232a2f119535facbbad49e2852.jpg)

### Tables

![4a7c24c0ab5d1459c2f446663388deaaa0cc0a915807fc4f417682ed57f7949f.jpg](../iclr_results/1511_Minimax%20Optimal%20Reinforcement%20Learning%20with%20Quasi-Optimism/tables/4a7c24c0ab5d1459c2f446663388deaaa0cc0a915807fc4f417682ed57f7949f.jpg)

![4dfb904a621f8c9e1000e62944359332f1b5012e266698d1f3f83a8443716b7d.jpg](../iclr_results/1511_Minimax%20Optimal%20Reinforcement%20Learning%20with%20Quasi-Optimism/tables/4dfb904a621f8c9e1000e62944359332f1b5012e266698d1f3f83a8443716b7d.jpg)

![649cb5c1bfd20229ac39aa8b5df745eeddd3b3f48e4d775b4de673e7358d2c5d.jpg](../iclr_results/1511_Minimax%20Optimal%20Reinforcement%20Learning%20with%20Quasi-Optimism/tables/649cb5c1bfd20229ac39aa8b5df745eeddd3b3f48e4d775b4de673e7358d2c5d.jpg)

![cc4f29143aae4bed3da12f41587f6f4db5cad450348318992f1c099d4375ad2a.jpg](../iclr_results/1511_Minimax%20Optimal%20Reinforcement%20Learning%20with%20Quasi-Optimism/tables/cc4f29143aae4bed3da12f41587f6f4db5cad450348318992f1c099d4375ad2a.jpg)

## Interpreting Language Reward Models via Contrastive Explanations


### Images

![2580f22b2e855d5bb04b30c5d6926bc29d778c4425349b3451a2591380e45fcc.jpg](../iclr_results/1512_Interpreting%20Language%20Reward%20Models%20via%20Contrastive%20Explanations/images/2580f22b2e855d5bb04b30c5d6926bc29d778c4425349b3451a2591380e45fcc.jpg)

![4536cd33165851e0aa7166779f06c038c9b250fe6b751113e5938b64939a6b0c.jpg](../iclr_results/1512_Interpreting%20Language%20Reward%20Models%20via%20Contrastive%20Explanations/images/4536cd33165851e0aa7166779f06c038c9b250fe6b751113e5938b64939a6b0c.jpg)

![8caca9e31e4ad0c043a786c6790239d417d3a90e5a7dd4571d2514605ded46ff.jpg](../iclr_results/1512_Interpreting%20Language%20Reward%20Models%20via%20Contrastive%20Explanations/images/8caca9e31e4ad0c043a786c6790239d417d3a90e5a7dd4571d2514605ded46ff.jpg)

![b13f046070ed999abd15aba6ff7d5270fbbf521adb5de3138892a6ca3a7a0c1e.jpg](../iclr_results/1512_Interpreting%20Language%20Reward%20Models%20via%20Contrastive%20Explanations/images/b13f046070ed999abd15aba6ff7d5270fbbf521adb5de3138892a6ca3a7a0c1e.jpg)

![b8ee0a8384ab696b2edb694a78cccacc6bbe1805ca725fcc084d81c7780d32c4.jpg](../iclr_results/1512_Interpreting%20Language%20Reward%20Models%20via%20Contrastive%20Explanations/images/b8ee0a8384ab696b2edb694a78cccacc6bbe1805ca725fcc084d81c7780d32c4.jpg)

![bda0679dc5bcd76ee28366dbe11d1b8a2353531b2bccd6bfb258a01a2f1fceb6.jpg](../iclr_results/1512_Interpreting%20Language%20Reward%20Models%20via%20Contrastive%20Explanations/images/bda0679dc5bcd76ee28366dbe11d1b8a2353531b2bccd6bfb258a01a2f1fceb6.jpg)

![e885f08b4d31a274c84ef2efbd5d998e2bb2385720f1ac18a5654182832c4c1a.jpg](../iclr_results/1512_Interpreting%20Language%20Reward%20Models%20via%20Contrastive%20Explanations/images/e885f08b4d31a274c84ef2efbd5d998e2bb2385720f1ac18a5654182832c4c1a.jpg)

### Tables

![0c6cddc8dbc7fcb140d4718d2d33b9f732718731981890f65d8c91e952d95814.jpg](../iclr_results/1512_Interpreting%20Language%20Reward%20Models%20via%20Contrastive%20Explanations/tables/0c6cddc8dbc7fcb140d4718d2d33b9f732718731981890f65d8c91e952d95814.jpg)

![5b435d86ad5f0eaafb4057622450e54be265ffc79a3570145a02823f32cb7780.jpg](../iclr_results/1512_Interpreting%20Language%20Reward%20Models%20via%20Contrastive%20Explanations/tables/5b435d86ad5f0eaafb4057622450e54be265ffc79a3570145a02823f32cb7780.jpg)

![79a57ca4a714f7f6898427f4e18a33f6d91cc16f9f986b0e1b5188ba4e7d419f.jpg](../iclr_results/1512_Interpreting%20Language%20Reward%20Models%20via%20Contrastive%20Explanations/tables/79a57ca4a714f7f6898427f4e18a33f6d91cc16f9f986b0e1b5188ba4e7d419f.jpg)

![cf3b54826ed1a6a65439ecd28fa60e6ae979c63858da06374eef36a4f3d5ef21.jpg](../iclr_results/1512_Interpreting%20Language%20Reward%20Models%20via%20Contrastive%20Explanations/tables/cf3b54826ed1a6a65439ecd28fa60e6ae979c63858da06374eef36a4f3d5ef21.jpg)

![d47b530a6cf74360f49efdb4ee83fc70b5e8c845de06135d6afc9f3630616194.jpg](../iclr_results/1512_Interpreting%20Language%20Reward%20Models%20via%20Contrastive%20Explanations/tables/d47b530a6cf74360f49efdb4ee83fc70b5e8c845de06135d6afc9f3630616194.jpg)

![e4433da9c34360cb0221f63684e461b3891e47572db97b97d3fe434a3effd087.jpg](../iclr_results/1512_Interpreting%20Language%20Reward%20Models%20via%20Contrastive%20Explanations/tables/e4433da9c34360cb0221f63684e461b3891e47572db97b97d3fe434a3effd087.jpg)

![f983ae7a1b2e6b50313a1340efa0aa467236872088480163957929ff17c0a849.jpg](../iclr_results/1512_Interpreting%20Language%20Reward%20Models%20via%20Contrastive%20Explanations/tables/f983ae7a1b2e6b50313a1340efa0aa467236872088480163957929ff17c0a849.jpg)

## Lipschitz Bandits in Optimal Space


### Images

![1b4115dab469a3bb59e197fe56c40ea5ae27c59d5bab3b5743abd61f8c51f8c7.jpg](../iclr_results/1513_Lipschitz%20Bandits%20in%20Optimal%20Space/images/1b4115dab469a3bb59e197fe56c40ea5ae27c59d5bab3b5743abd61f8c51f8c7.jpg)

![9a86b771e8c9d92acbf5eb3f0c5bdd0b0c4884e145174e96454a8037555ded2e.jpg](../iclr_results/1513_Lipschitz%20Bandits%20in%20Optimal%20Space/images/9a86b771e8c9d92acbf5eb3f0c5bdd0b0c4884e145174e96454a8037555ded2e.jpg)

![a5084cad36a7d9dee12354bd7ee9a62e08b3ba1e799c8fe3ea547c4c056bbfd5.jpg](../iclr_results/1513_Lipschitz%20Bandits%20in%20Optimal%20Space/images/a5084cad36a7d9dee12354bd7ee9a62e08b3ba1e799c8fe3ea547c4c056bbfd5.jpg)

![f8e66b0d6857eb6c7af930017e94e1f89e3db9b1217011616cb17aa37b6ce24e.jpg](../iclr_results/1513_Lipschitz%20Bandits%20in%20Optimal%20Space/images/f8e66b0d6857eb6c7af930017e94e1f89e3db9b1217011616cb17aa37b6ce24e.jpg)

## Bootstrapped Model Predictive Control


### Images

![0bb0c215cbfa409d009ca8316ceb6b8cd8fd400678d1339d87e80dc6b4710cee.jpg](../iclr_results/1514_Bootstrapped%20Model%20Predictive%20Control/images/0bb0c215cbfa409d009ca8316ceb6b8cd8fd400678d1339d87e80dc6b4710cee.jpg)

![0d1bfc626275ced8240192f5217c35f693caf42a4270e69ab18f5e82fbf19d9b.jpg](../iclr_results/1514_Bootstrapped%20Model%20Predictive%20Control/images/0d1bfc626275ced8240192f5217c35f693caf42a4270e69ab18f5e82fbf19d9b.jpg)

![0ff665156bb4e52a215086d831c174fe41469f913bb2a3e3c9dde7f807930eaa.jpg](../iclr_results/1514_Bootstrapped%20Model%20Predictive%20Control/images/0ff665156bb4e52a215086d831c174fe41469f913bb2a3e3c9dde7f807930eaa.jpg)

![1aafea2e6a6e3c24769e06b03c6d813258daa5212ce179e7c410c90ee06b0285.jpg](../iclr_results/1514_Bootstrapped%20Model%20Predictive%20Control/images/1aafea2e6a6e3c24769e06b03c6d813258daa5212ce179e7c410c90ee06b0285.jpg)

![29327d3a08a1e0efdc110bee93a9a3396700bbecafd1f4b5d48e04975a23dd2f.jpg](../iclr_results/1514_Bootstrapped%20Model%20Predictive%20Control/images/29327d3a08a1e0efdc110bee93a9a3396700bbecafd1f4b5d48e04975a23dd2f.jpg)

![2b60a0d2653c7ca06221b70701ce40014ad86dc6d4c72c4346f31215a4327a19.jpg](../iclr_results/1514_Bootstrapped%20Model%20Predictive%20Control/images/2b60a0d2653c7ca06221b70701ce40014ad86dc6d4c72c4346f31215a4327a19.jpg)

![83f1a431fc1ba9b091457e4698d4a35c7a0e20c82317254d1e183d0cd8d6d0f1.jpg](../iclr_results/1514_Bootstrapped%20Model%20Predictive%20Control/images/83f1a431fc1ba9b091457e4698d4a35c7a0e20c82317254d1e183d0cd8d6d0f1.jpg)

![8edf36e9f92f366fd36e212f6155a5e49085e005461d60862a66818e4410186b.jpg](../iclr_results/1514_Bootstrapped%20Model%20Predictive%20Control/images/8edf36e9f92f366fd36e212f6155a5e49085e005461d60862a66818e4410186b.jpg)

![9ad138705395dc21b10dccddf5d8a23084d5096944119fd5ced318e85df6cb12.jpg](../iclr_results/1514_Bootstrapped%20Model%20Predictive%20Control/images/9ad138705395dc21b10dccddf5d8a23084d5096944119fd5ced318e85df6cb12.jpg)

![9e5105f6ab1bbbb5fe256dcb2107bc5c3bed9a60d39098f5951ebf3bef8b9863.jpg](../iclr_results/1514_Bootstrapped%20Model%20Predictive%20Control/images/9e5105f6ab1bbbb5fe256dcb2107bc5c3bed9a60d39098f5951ebf3bef8b9863.jpg)

![db9eef85ce22f74d652533706c7342b3d5ae429bf225d199f45f9e5bb9675c1f.jpg](../iclr_results/1514_Bootstrapped%20Model%20Predictive%20Control/images/db9eef85ce22f74d652533706c7342b3d5ae429bf225d199f45f9e5bb9675c1f.jpg)

![fa4e2810dcb514db35b90aa1f7effd57755b05edc22701b25c4dca8a7247365c.jpg](../iclr_results/1514_Bootstrapped%20Model%20Predictive%20Control/images/fa4e2810dcb514db35b90aa1f7effd57755b05edc22701b25c4dca8a7247365c.jpg)

### Tables

![68ae3f06fc5ec18deedf3d2fe2d879bb9083832b477e8ead33ce66e3357e7ddb.jpg](../iclr_results/1514_Bootstrapped%20Model%20Predictive%20Control/tables/68ae3f06fc5ec18deedf3d2fe2d879bb9083832b477e8ead33ce66e3357e7ddb.jpg)

![c82605035105466b1b8827ea35050d951568c10373cbc4535f57fe93d0ece881.jpg](../iclr_results/1514_Bootstrapped%20Model%20Predictive%20Control/tables/c82605035105466b1b8827ea35050d951568c10373cbc4535f57fe93d0ece881.jpg)

## Disentangling 3D Animal Pose Dynamics with Scrubbed Conditional Latent Variables


### Images

![10d50262baf72a72ddacdc71b1b49b2682b5d213b7e25eb364e534ab42890341.jpg](../iclr_results/1515_Disentangling%203D%20Animal%20Pose%20Dynamics%20with%20Scrubbed%20Conditional%20Latent%20Variables/images/10d50262baf72a72ddacdc71b1b49b2682b5d213b7e25eb364e534ab42890341.jpg)

![38c86d10f485843f78cd554f225d0fbcdf335ffa11643ea28baa77d4aba1cce4.jpg](../iclr_results/1515_Disentangling%203D%20Animal%20Pose%20Dynamics%20with%20Scrubbed%20Conditional%20Latent%20Variables/images/38c86d10f485843f78cd554f225d0fbcdf335ffa11643ea28baa77d4aba1cce4.jpg)

![61c7b4d1d6299f06f60ca58d549e4036a7e6b03d321e9427d2c79732f1c43cc1.jpg](../iclr_results/1515_Disentangling%203D%20Animal%20Pose%20Dynamics%20with%20Scrubbed%20Conditional%20Latent%20Variables/images/61c7b4d1d6299f06f60ca58d549e4036a7e6b03d321e9427d2c79732f1c43cc1.jpg)

![682cbd3f0bb94109f95bb808b15502089a87d9aab124e8003d0ff285b5b289ff.jpg](../iclr_results/1515_Disentangling%203D%20Animal%20Pose%20Dynamics%20with%20Scrubbed%20Conditional%20Latent%20Variables/images/682cbd3f0bb94109f95bb808b15502089a87d9aab124e8003d0ff285b5b289ff.jpg)

![7a38e921df39317ac535ce5eb1ae0d1eda5c93a97474c5dc536a8f22ee89695d.jpg](../iclr_results/1515_Disentangling%203D%20Animal%20Pose%20Dynamics%20with%20Scrubbed%20Conditional%20Latent%20Variables/images/7a38e921df39317ac535ce5eb1ae0d1eda5c93a97474c5dc536a8f22ee89695d.jpg)

![8e87e83ac5e1deccc50e0b1bdd2bfdbdf39510875861ffa5d51b79c9b72786bf.jpg](../iclr_results/1515_Disentangling%203D%20Animal%20Pose%20Dynamics%20with%20Scrubbed%20Conditional%20Latent%20Variables/images/8e87e83ac5e1deccc50e0b1bdd2bfdbdf39510875861ffa5d51b79c9b72786bf.jpg)

![a82ced66f99838d2393254973b2bb6235bf54bd2c9959af8584c3718921b272c.jpg](../iclr_results/1515_Disentangling%203D%20Animal%20Pose%20Dynamics%20with%20Scrubbed%20Conditional%20Latent%20Variables/images/a82ced66f99838d2393254973b2bb6235bf54bd2c9959af8584c3718921b272c.jpg)

![b37cb0555c28d806c8c99e845a3904e25b5e87af7b26b90960026b6c3df9c1ff.jpg](../iclr_results/1515_Disentangling%203D%20Animal%20Pose%20Dynamics%20with%20Scrubbed%20Conditional%20Latent%20Variables/images/b37cb0555c28d806c8c99e845a3904e25b5e87af7b26b90960026b6c3df9c1ff.jpg)

![b38131a45bfc36a1cc659baff844d67fe522034ee25e228b85312206e20b5b9f.jpg](../iclr_results/1515_Disentangling%203D%20Animal%20Pose%20Dynamics%20with%20Scrubbed%20Conditional%20Latent%20Variables/images/b38131a45bfc36a1cc659baff844d67fe522034ee25e228b85312206e20b5b9f.jpg)

![fca337792366d6c471dcf5d4e5582e8c2a6a95bb26d47d5d13e95ae25cc9b443.jpg](../iclr_results/1515_Disentangling%203D%20Animal%20Pose%20Dynamics%20with%20Scrubbed%20Conditional%20Latent%20Variables/images/fca337792366d6c471dcf5d4e5582e8c2a6a95bb26d47d5d13e95ae25cc9b443.jpg)

### Tables

![1e1c1120f566f738936a7e452aaae51fdfd04a965b6bdf08e13b6bcc79328df8.jpg](../iclr_results/1515_Disentangling%203D%20Animal%20Pose%20Dynamics%20with%20Scrubbed%20Conditional%20Latent%20Variables/tables/1e1c1120f566f738936a7e452aaae51fdfd04a965b6bdf08e13b6bcc79328df8.jpg)

![57f7568405fbf69c4d4618d9d6adecf667c079c33414fae24455059df85379a6.jpg](../iclr_results/1515_Disentangling%203D%20Animal%20Pose%20Dynamics%20with%20Scrubbed%20Conditional%20Latent%20Variables/tables/57f7568405fbf69c4d4618d9d6adecf667c079c33414fae24455059df85379a6.jpg)

![6dc20a6d42ec71770286087f5629b859c72fb1f1a3b53e59807dbf69d8ad05a2.jpg](../iclr_results/1515_Disentangling%203D%20Animal%20Pose%20Dynamics%20with%20Scrubbed%20Conditional%20Latent%20Variables/tables/6dc20a6d42ec71770286087f5629b859c72fb1f1a3b53e59807dbf69d8ad05a2.jpg)

![7d063f4dbc6afd8330c3397b673902da00ff485a357827f1ddda588aa7a88173.jpg](../iclr_results/1515_Disentangling%203D%20Animal%20Pose%20Dynamics%20with%20Scrubbed%20Conditional%20Latent%20Variables/tables/7d063f4dbc6afd8330c3397b673902da00ff485a357827f1ddda588aa7a88173.jpg)

![8ad130d52ed783e3925e4f94ac485d8d36c5f336c72e25c7dfc3d849c4b0a1ba.jpg](../iclr_results/1515_Disentangling%203D%20Animal%20Pose%20Dynamics%20with%20Scrubbed%20Conditional%20Latent%20Variables/tables/8ad130d52ed783e3925e4f94ac485d8d36c5f336c72e25c7dfc3d849c4b0a1ba.jpg)

![bbcb6ce850590d9e5aa596294d6abe27a843a2082febfa8a956b1bf560ba5d05.jpg](../iclr_results/1515_Disentangling%203D%20Animal%20Pose%20Dynamics%20with%20Scrubbed%20Conditional%20Latent%20Variables/tables/bbcb6ce850590d9e5aa596294d6abe27a843a2082febfa8a956b1bf560ba5d05.jpg)

![beb4f05e901b680bde6f9bf2d2d90f10009190a95e4d881ce8fedd6cee69169f.jpg](../iclr_results/1515_Disentangling%203D%20Animal%20Pose%20Dynamics%20with%20Scrubbed%20Conditional%20Latent%20Variables/tables/beb4f05e901b680bde6f9bf2d2d90f10009190a95e4d881ce8fedd6cee69169f.jpg)

![e5c19f2ea84d1101d6fc3d06b34e854a6c4c18663bbc9521a8325d21fa5c2255.jpg](../iclr_results/1515_Disentangling%203D%20Animal%20Pose%20Dynamics%20with%20Scrubbed%20Conditional%20Latent%20Variables/tables/e5c19f2ea84d1101d6fc3d06b34e854a6c4c18663bbc9521a8325d21fa5c2255.jpg)

![f2b0ce88a7cf392dfb96d80a47826348764f2d2bc5ab3f35941066df706039b6.jpg](../iclr_results/1515_Disentangling%203D%20Animal%20Pose%20Dynamics%20with%20Scrubbed%20Conditional%20Latent%20Variables/tables/f2b0ce88a7cf392dfb96d80a47826348764f2d2bc5ab3f35941066df706039b6.jpg)

![fdd6713da483fbfb52ed87ba54119a7b729275ca009358349534097e4ba09c0e.jpg](../iclr_results/1515_Disentangling%203D%20Animal%20Pose%20Dynamics%20with%20Scrubbed%20Conditional%20Latent%20Variables/tables/fdd6713da483fbfb52ed87ba54119a7b729275ca009358349534097e4ba09c0e.jpg)

## Simple Guidance Mechanisms for Discrete Diffusion Models


### Images

![0a209c70892f06759696d628825cf7be3aab152ef89517460da69af32233c488.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/images/0a209c70892f06759696d628825cf7be3aab152ef89517460da69af32233c488.jpg)

![86fbf7bbb12526e365cb364cedf97a3a101dd0d794e4028528231675d877785d.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/images/86fbf7bbb12526e365cb364cedf97a3a101dd0d794e4028528231675d877785d.jpg)

![a8f1b7da99ee84cb29d7a1675180c1dcc745d80a6bbf4dae0d2f8348c9ffd1f2.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/images/a8f1b7da99ee84cb29d7a1675180c1dcc745d80a6bbf4dae0d2f8348c9ffd1f2.jpg)

![a9922c03abcbed6d249aa77d74f52d27f1584afc7b1a0603fdd78a5cf676162e.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/images/a9922c03abcbed6d249aa77d74f52d27f1584afc7b1a0603fdd78a5cf676162e.jpg)

![ae5ec0604d934da4ec217f75330e1db16300ae4dcd9db1ca1de2aaa8c36787d4.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/images/ae5ec0604d934da4ec217f75330e1db16300ae4dcd9db1ca1de2aaa8c36787d4.jpg)

![b9a1848e9329d6f06cfbbf62009f53b6eb8263e85d2c21bf01a194efd75de218.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/images/b9a1848e9329d6f06cfbbf62009f53b6eb8263e85d2c21bf01a194efd75de218.jpg)

### Tables

![12a660d84a3048f15deb0f4acbbc1de4d838b207fb864e714444abc5ddba9c95.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/12a660d84a3048f15deb0f4acbbc1de4d838b207fb864e714444abc5ddba9c95.jpg)

![1d11a6be8c5ba4997072dd172569208a3d8da9bb6a10268975a2151a182ff84f.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/1d11a6be8c5ba4997072dd172569208a3d8da9bb6a10268975a2151a182ff84f.jpg)

![1fdb987c74b1c7f4d513d81232c462dac83c1e10500119edb515da7b905414ba.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/1fdb987c74b1c7f4d513d81232c462dac83c1e10500119edb515da7b905414ba.jpg)

![2bfd03dff6180c0b6fc70f9b5f7cbe48d28ed3e34c4605d9a74cad34f34db6e1.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/2bfd03dff6180c0b6fc70f9b5f7cbe48d28ed3e34c4605d9a74cad34f34db6e1.jpg)

![2d2d5491902842ed8ea5e739cefaec73b92e330a63679f13d9cfb6adc470f56c.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/2d2d5491902842ed8ea5e739cefaec73b92e330a63679f13d9cfb6adc470f56c.jpg)

![2d485c53c93e0e1de5f41bc44d2c9aa5fa836f3d0ceb5e784c9ba52242576127.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/2d485c53c93e0e1de5f41bc44d2c9aa5fa836f3d0ceb5e784c9ba52242576127.jpg)

![2e35c7167fdb8d5462d3ecfdd7d85147da58f94069292b5a46c1b191a78c3420.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/2e35c7167fdb8d5462d3ecfdd7d85147da58f94069292b5a46c1b191a78c3420.jpg)

![3ba8395abc0d10da7557d90c04de50b415ba3c1642eaedfa4dc7c447abd43ba2.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/3ba8395abc0d10da7557d90c04de50b415ba3c1642eaedfa4dc7c447abd43ba2.jpg)

![41a1f2665546cf37cc34c142455fd87e67397b09e782af5427e4f82e900b2b69.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/41a1f2665546cf37cc34c142455fd87e67397b09e782af5427e4f82e900b2b69.jpg)

![4a4151a01584a519dff8471811328edfc433ff0bb2a0f5ee7a989f09884cb327.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/4a4151a01584a519dff8471811328edfc433ff0bb2a0f5ee7a989f09884cb327.jpg)

![4a6b414163a1b54926503457b4b03a6611436ac864cc739952bbabcc6f8fd933.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/4a6b414163a1b54926503457b4b03a6611436ac864cc739952bbabcc6f8fd933.jpg)

![4fc99acffbb1934d631b3ce5495d07d0100d6beeb2b150a9c7aa1040467c89cc.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/4fc99acffbb1934d631b3ce5495d07d0100d6beeb2b150a9c7aa1040467c89cc.jpg)

![5511cdad75b366d584c44410f3704937076776c1f732497c581a26467e617621.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/5511cdad75b366d584c44410f3704937076776c1f732497c581a26467e617621.jpg)

![5562adb3c1e96d07a497ed4e00c428b810bf311f7972066da5a2a0e6a5fa15ac.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/5562adb3c1e96d07a497ed4e00c428b810bf311f7972066da5a2a0e6a5fa15ac.jpg)

![566e3ae2c45d394e68bb49579536cce096e5c6c4c147cd7a76ee5a37a59ad85a.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/566e3ae2c45d394e68bb49579536cce096e5c6c4c147cd7a76ee5a37a59ad85a.jpg)

![5aa189bdf48844e5ca680e97b09f8e1ca7005ea9ea4cfc44eaefc5e9ef4031ce.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/5aa189bdf48844e5ca680e97b09f8e1ca7005ea9ea4cfc44eaefc5e9ef4031ce.jpg)

![60a99d7502c980863db688608b952c9743614a479bee40f9961d9740124b5376.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/60a99d7502c980863db688608b952c9743614a479bee40f9961d9740124b5376.jpg)

![641f6b9adbc4dc3ecf9b4e2d214ea2649421e64ab2071ae666dc2210099acac2.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/641f6b9adbc4dc3ecf9b4e2d214ea2649421e64ab2071ae666dc2210099acac2.jpg)

![674502809505af792f08e398478b8d8a558f2acd28b02a978c38fa9017e2a64d.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/674502809505af792f08e398478b8d8a558f2acd28b02a978c38fa9017e2a64d.jpg)

![6e07a956b063292b5af73f2ecd9015e7b1502b773b0a9eb24a996eb758e87047.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/6e07a956b063292b5af73f2ecd9015e7b1502b773b0a9eb24a996eb758e87047.jpg)

![6e43404337e0247dbae3e510725e81caafdd16958265d4e5fc790e825ec5470d.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/6e43404337e0247dbae3e510725e81caafdd16958265d4e5fc790e825ec5470d.jpg)

![6ee5135c37ba3a0241e0ea1a887b9fcb6c0d17caee1fb693d0804d52adb670a4.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/6ee5135c37ba3a0241e0ea1a887b9fcb6c0d17caee1fb693d0804d52adb670a4.jpg)

![72924aee6fd95f5729744eceec834dbebc41d199fb383f85719ff360a29fb646.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/72924aee6fd95f5729744eceec834dbebc41d199fb383f85719ff360a29fb646.jpg)

![79d9ecf826f7e1881bac77dc53c6673f4bbf029ee67aaa1b4b3cd7bd114421be.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/79d9ecf826f7e1881bac77dc53c6673f4bbf029ee67aaa1b4b3cd7bd114421be.jpg)

![7cee6df98fc14dd0d2c096785ef7ec19a536f33aeb3c682f099cdd2e7316e194.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/7cee6df98fc14dd0d2c096785ef7ec19a536f33aeb3c682f099cdd2e7316e194.jpg)

![7ede2675aaa0f85281dfb8c4613eafc73212db1700f88ba149f0344c279a5ce4.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/7ede2675aaa0f85281dfb8c4613eafc73212db1700f88ba149f0344c279a5ce4.jpg)

![85d253992ac2711116a7551674e31e555f8fb134803b20cae7e532ace4e08295.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/85d253992ac2711116a7551674e31e555f8fb134803b20cae7e532ace4e08295.jpg)

![886f4ca16bced5b8cd1230052c66b1e0a233889db1cb63e3a2f72257f5781ca9.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/886f4ca16bced5b8cd1230052c66b1e0a233889db1cb63e3a2f72257f5781ca9.jpg)

![a645fee0e7e2af8a4505cc31b8f46d73272750814d191020e5e2aa71bb31b9ed.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/a645fee0e7e2af8a4505cc31b8f46d73272750814d191020e5e2aa71bb31b9ed.jpg)

![b486b62f2b277d2bd703b481df17ede0e4b7825f974752aaa5471c8dcd20f661.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/b486b62f2b277d2bd703b481df17ede0e4b7825f974752aaa5471c8dcd20f661.jpg)

![bfe97d2a6f51f2875b6be56df9e7151474a1d4eb3f8f6c4d8ee2ce510c25810a.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/bfe97d2a6f51f2875b6be56df9e7151474a1d4eb3f8f6c4d8ee2ce510c25810a.jpg)

![ccd8e00eca5c38103a69775ddf5f982ea9ce2537ae89e73ebe8535be21f402af.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/ccd8e00eca5c38103a69775ddf5f982ea9ce2537ae89e73ebe8535be21f402af.jpg)

![d4e7c9e8d5d52d1e2bd7ee960b9de040955561c29f603affa6a31ec784596ad6.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/d4e7c9e8d5d52d1e2bd7ee960b9de040955561c29f603affa6a31ec784596ad6.jpg)

![dab268515c8fd2417445c63a9142ca7380d10b9b319e39b5da30d8aabdd8e68c.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/dab268515c8fd2417445c63a9142ca7380d10b9b319e39b5da30d8aabdd8e68c.jpg)

![dd3bdbe044f90339cbf9c46bbadc9dd93ba3bbf0243d22f523bfcaf22bb824be.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/dd3bdbe044f90339cbf9c46bbadc9dd93ba3bbf0243d22f523bfcaf22bb824be.jpg)

![f381462587037485aa892cd8f82699850e7ec71500abea1d8a41684d215b4efd.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/f381462587037485aa892cd8f82699850e7ec71500abea1d8a41684d215b4efd.jpg)

![f764bbd1bebb8cf893236ed2eab90741ec1b523d27b89e637d3f7d7914fb3a1e.jpg](../iclr_results/1516_Simple%20Guidance%20Mechanisms%20for%20Discrete%20Diffusion%20Models/tables/f764bbd1bebb8cf893236ed2eab90741ec1b523d27b89e637d3f7d7914fb3a1e.jpg)

## MM-EMBED: UNIVERSAL MULTIMODAL RETRIEVAL WITH MULTIMODAL LLMS


### Images

![771939372d927225de1847a4f0c25a1327ddfbfafeb6795bcdabad5ab2f2ed44.jpg](../iclr_results/1517_MM-EMBED_%20UNIVERSAL%20MULTIMODAL%20RETRIEVAL%20WITH%20MULTIMODAL%20LLMS/images/771939372d927225de1847a4f0c25a1327ddfbfafeb6795bcdabad5ab2f2ed44.jpg)

![aea3c425f50a2ab3190c5f085e9b2e4cf6c74d819627ca00f50059dbd3ab18cd.jpg](../iclr_results/1517_MM-EMBED_%20UNIVERSAL%20MULTIMODAL%20RETRIEVAL%20WITH%20MULTIMODAL%20LLMS/images/aea3c425f50a2ab3190c5f085e9b2e4cf6c74d819627ca00f50059dbd3ab18cd.jpg)

### Tables

![02aeec0d7680c807917a14f76b4c8862b7a94ba373dbed266a1528f8f79fd67b.jpg](../iclr_results/1517_MM-EMBED_%20UNIVERSAL%20MULTIMODAL%20RETRIEVAL%20WITH%20MULTIMODAL%20LLMS/tables/02aeec0d7680c807917a14f76b4c8862b7a94ba373dbed266a1528f8f79fd67b.jpg)

![0abb0592d19193e786d2582b2252295a4955f27b44d1ec129d891c50bdd392c1.jpg](../iclr_results/1517_MM-EMBED_%20UNIVERSAL%20MULTIMODAL%20RETRIEVAL%20WITH%20MULTIMODAL%20LLMS/tables/0abb0592d19193e786d2582b2252295a4955f27b44d1ec129d891c50bdd392c1.jpg)

![213d228e2bae1232155aab7b7678b6821cf4c379c5ebe799ca394dfcd5ee85e1.jpg](../iclr_results/1517_MM-EMBED_%20UNIVERSAL%20MULTIMODAL%20RETRIEVAL%20WITH%20MULTIMODAL%20LLMS/tables/213d228e2bae1232155aab7b7678b6821cf4c379c5ebe799ca394dfcd5ee85e1.jpg)

![3909059f58365d13e4563da4a321076e03fc22f1085aafa436a29461557bc92c.jpg](../iclr_results/1517_MM-EMBED_%20UNIVERSAL%20MULTIMODAL%20RETRIEVAL%20WITH%20MULTIMODAL%20LLMS/tables/3909059f58365d13e4563da4a321076e03fc22f1085aafa436a29461557bc92c.jpg)

![6229f4c5c3d956720a20749a6dcd2bf0cdc8266550c9cf380ee4a6093e3dc6e9.jpg](../iclr_results/1517_MM-EMBED_%20UNIVERSAL%20MULTIMODAL%20RETRIEVAL%20WITH%20MULTIMODAL%20LLMS/tables/6229f4c5c3d956720a20749a6dcd2bf0cdc8266550c9cf380ee4a6093e3dc6e9.jpg)

![6d02f24a2485ba1ad9138f720fdc812ed1da32cad56608e642e7520bd7afbbd4.jpg](../iclr_results/1517_MM-EMBED_%20UNIVERSAL%20MULTIMODAL%20RETRIEVAL%20WITH%20MULTIMODAL%20LLMS/tables/6d02f24a2485ba1ad9138f720fdc812ed1da32cad56608e642e7520bd7afbbd4.jpg)

![6e6b4bf0d4a6194a1fe6dbd53ea59da9086942ca1272bb39dd4d2dbcd3067627.jpg](../iclr_results/1517_MM-EMBED_%20UNIVERSAL%20MULTIMODAL%20RETRIEVAL%20WITH%20MULTIMODAL%20LLMS/tables/6e6b4bf0d4a6194a1fe6dbd53ea59da9086942ca1272bb39dd4d2dbcd3067627.jpg)

![8cf34e5a13bc26ab41cb58309ed2d6c8c955be20a2e722e3e3dcd187d9770a00.jpg](../iclr_results/1517_MM-EMBED_%20UNIVERSAL%20MULTIMODAL%20RETRIEVAL%20WITH%20MULTIMODAL%20LLMS/tables/8cf34e5a13bc26ab41cb58309ed2d6c8c955be20a2e722e3e3dcd187d9770a00.jpg)

![8e797caf8a0d13b32a94638b386c9f7a4b3cec040ec3c83d8226803755facb37.jpg](../iclr_results/1517_MM-EMBED_%20UNIVERSAL%20MULTIMODAL%20RETRIEVAL%20WITH%20MULTIMODAL%20LLMS/tables/8e797caf8a0d13b32a94638b386c9f7a4b3cec040ec3c83d8226803755facb37.jpg)

![a41d91585a7531ade65c6f7767868ab0b701756ecb9873ba38f6df8eaf17c287.jpg](../iclr_results/1517_MM-EMBED_%20UNIVERSAL%20MULTIMODAL%20RETRIEVAL%20WITH%20MULTIMODAL%20LLMS/tables/a41d91585a7531ade65c6f7767868ab0b701756ecb9873ba38f6df8eaf17c287.jpg)

![aa1b4367e5a6386a11e73e3db2016bd4af0e5af7285b776ca68b5171e89f36f1.jpg](../iclr_results/1517_MM-EMBED_%20UNIVERSAL%20MULTIMODAL%20RETRIEVAL%20WITH%20MULTIMODAL%20LLMS/tables/aa1b4367e5a6386a11e73e3db2016bd4af0e5af7285b776ca68b5171e89f36f1.jpg)

![af7f9530e1c48858f2d8830e4000737398ef90fb6385f56f9cb520dfdf7be858.jpg](../iclr_results/1517_MM-EMBED_%20UNIVERSAL%20MULTIMODAL%20RETRIEVAL%20WITH%20MULTIMODAL%20LLMS/tables/af7f9530e1c48858f2d8830e4000737398ef90fb6385f56f9cb520dfdf7be858.jpg)

![cedf2832df4528738b5b9c726396c26b945b1ad0922e731fd15f30a0eda55cd8.jpg](../iclr_results/1517_MM-EMBED_%20UNIVERSAL%20MULTIMODAL%20RETRIEVAL%20WITH%20MULTIMODAL%20LLMS/tables/cedf2832df4528738b5b9c726396c26b945b1ad0922e731fd15f30a0eda55cd8.jpg)

![df79b3b43694eb29cfd8813fc0852d97d3059ef67bf1a004700c5441b306792a.jpg](../iclr_results/1517_MM-EMBED_%20UNIVERSAL%20MULTIMODAL%20RETRIEVAL%20WITH%20MULTIMODAL%20LLMS/tables/df79b3b43694eb29cfd8813fc0852d97d3059ef67bf1a004700c5441b306792a.jpg)

![e3ef35c50a3b0d770ca4e55ad2210d2a0b2d63e147927d8ca4acfb47f3cc3662.jpg](../iclr_results/1517_MM-EMBED_%20UNIVERSAL%20MULTIMODAL%20RETRIEVAL%20WITH%20MULTIMODAL%20LLMS/tables/e3ef35c50a3b0d770ca4e55ad2210d2a0b2d63e147927d8ca4acfb47f3cc3662.jpg)

![ee9461617b7be28ab377829195d57d47920aa8417c022f40ab32de0029ebbfee.jpg](../iclr_results/1517_MM-EMBED_%20UNIVERSAL%20MULTIMODAL%20RETRIEVAL%20WITH%20MULTIMODAL%20LLMS/tables/ee9461617b7be28ab377829195d57d47920aa8417c022f40ab32de0029ebbfee.jpg)

![f179c1e10e2b890f05370cba235930956013f4299dd9b31da06834b42857f177.jpg](../iclr_results/1517_MM-EMBED_%20UNIVERSAL%20MULTIMODAL%20RETRIEVAL%20WITH%20MULTIMODAL%20LLMS/tables/f179c1e10e2b890f05370cba235930956013f4299dd9b31da06834b42857f177.jpg)

![f8e8203e83363dff75d3b568e97b481fb93884afa7168b53310daa56bb295b24.jpg](../iclr_results/1517_MM-EMBED_%20UNIVERSAL%20MULTIMODAL%20RETRIEVAL%20WITH%20MULTIMODAL%20LLMS/tables/f8e8203e83363dff75d3b568e97b481fb93884afa7168b53310daa56bb295b24.jpg)

## Language Guided Skill Discovery


### Images

![11e51344b94779a8f55a59c6cae47bcc557e74f6f3044588f5f0c8ab0207f29c.jpg](../iclr_results/1518_Language%20Guided%20Skill%20Discovery/images/11e51344b94779a8f55a59c6cae47bcc557e74f6f3044588f5f0c8ab0207f29c.jpg)

![1c74235d37ccaeea3ff987506b4e503d115067b1698c5ed9fba8e367c7913d78.jpg](../iclr_results/1518_Language%20Guided%20Skill%20Discovery/images/1c74235d37ccaeea3ff987506b4e503d115067b1698c5ed9fba8e367c7913d78.jpg)

![4115f77f5fe0ccc0db199497f6cb496e3e9d7c92ccd980e92a779991b5b0b8d4.jpg](../iclr_results/1518_Language%20Guided%20Skill%20Discovery/images/4115f77f5fe0ccc0db199497f6cb496e3e9d7c92ccd980e92a779991b5b0b8d4.jpg)

![44f5dc29fba490b7b0c59974103e8ebec6f7fe4f69eb66f1e1bfe231c581dd1c.jpg](../iclr_results/1518_Language%20Guided%20Skill%20Discovery/images/44f5dc29fba490b7b0c59974103e8ebec6f7fe4f69eb66f1e1bfe231c581dd1c.jpg)

![58645d4c5a5de0bf197c58314594b30d342903154e5155311cb16e9eb5d84d44.jpg](../iclr_results/1518_Language%20Guided%20Skill%20Discovery/images/58645d4c5a5de0bf197c58314594b30d342903154e5155311cb16e9eb5d84d44.jpg)

![6cba806d0f24c20108e82aee78c007384da55d3501beee145c7ec38e05469162.jpg](../iclr_results/1518_Language%20Guided%20Skill%20Discovery/images/6cba806d0f24c20108e82aee78c007384da55d3501beee145c7ec38e05469162.jpg)

![7050517a9b24fac67150bac91552972686f75ae3cd352229600b4448036e44ea.jpg](../iclr_results/1518_Language%20Guided%20Skill%20Discovery/images/7050517a9b24fac67150bac91552972686f75ae3cd352229600b4448036e44ea.jpg)

![8540e81e6ed5ef7267d8641c5a4e616bba2e51e19518e6b2f25afe8b9d540ab1.jpg](../iclr_results/1518_Language%20Guided%20Skill%20Discovery/images/8540e81e6ed5ef7267d8641c5a4e616bba2e51e19518e6b2f25afe8b9d540ab1.jpg)

![981b8f8c3e84299a3888357da1f02e03ae1b2224e9db0d3fcc92c6910ca3c8d9.jpg](../iclr_results/1518_Language%20Guided%20Skill%20Discovery/images/981b8f8c3e84299a3888357da1f02e03ae1b2224e9db0d3fcc92c6910ca3c8d9.jpg)

![b55c9026f0f284ac121ab8929f9ce6b5f680e1235c7383005e0f96ec62d382bc.jpg](../iclr_results/1518_Language%20Guided%20Skill%20Discovery/images/b55c9026f0f284ac121ab8929f9ce6b5f680e1235c7383005e0f96ec62d382bc.jpg)

![f4bca87fa5c274540a71e6b88f41b0390aad20634e0b9b553f7ee71d8a200408.jpg](../iclr_results/1518_Language%20Guided%20Skill%20Discovery/images/f4bca87fa5c274540a71e6b88f41b0390aad20634e0b9b553f7ee71d8a200408.jpg)

![fbc9188b40ad047d2086e66fdff03a58ce265819bc869fc870b879e51808127c.jpg](../iclr_results/1518_Language%20Guided%20Skill%20Discovery/images/fbc9188b40ad047d2086e66fdff03a58ce265819bc869fc870b879e51808127c.jpg)

### Tables

![371d52c8e894b92113d254ccf76e1e156d672ed22088830f6fb670cca92e184e.jpg](../iclr_results/1518_Language%20Guided%20Skill%20Discovery/tables/371d52c8e894b92113d254ccf76e1e156d672ed22088830f6fb670cca92e184e.jpg)

![42283864c0b29ee5b31fb6bd77c5f1d1dfd4b96bf5ac5f8f8571cc9e64685db5.jpg](../iclr_results/1518_Language%20Guided%20Skill%20Discovery/tables/42283864c0b29ee5b31fb6bd77c5f1d1dfd4b96bf5ac5f8f8571cc9e64685db5.jpg)

![84a59344f8b6a8efe4312e270a29eb4b92f42fa78df15445babadd29bcfdcbda.jpg](../iclr_results/1518_Language%20Guided%20Skill%20Discovery/tables/84a59344f8b6a8efe4312e270a29eb4b92f42fa78df15445babadd29bcfdcbda.jpg)

## Valid Conformal Prediction for Dynamic GNNs


### Images

![13d7518e460b40bc928ba997605e09b6fc85385733436e55cd2ff658f36d6572.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/images/13d7518e460b40bc928ba997605e09b6fc85385733436e55cd2ff658f36d6572.jpg)

![1916972b5f59dc7b61664d5fa2a39a667502e5f6241f83146c27f22c99851f4d.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/images/1916972b5f59dc7b61664d5fa2a39a667502e5f6241f83146c27f22c99851f4d.jpg)

![54b5277a19f3770f51dba88c4fc02a84a7ab1140099dff60305b02c429af2b8c.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/images/54b5277a19f3770f51dba88c4fc02a84a7ab1140099dff60305b02c429af2b8c.jpg)

![7c0016dcb6cb3037a61a14a96d953c54df0cb5334dcd6f748233ab28cd3eb086.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/images/7c0016dcb6cb3037a61a14a96d953c54df0cb5334dcd6f748233ab28cd3eb086.jpg)

![b4d89d00995a88e4dbccb64465bcbe353862d0c667024dd73dde2a29923734d2.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/images/b4d89d00995a88e4dbccb64465bcbe353862d0c667024dd73dde2a29923734d2.jpg)

![c09e958d1db6eff2fb20c77d18ca829dd99f93fc99b331fd17b0347ae9234f6a.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/images/c09e958d1db6eff2fb20c77d18ca829dd99f93fc99b331fd17b0347ae9234f6a.jpg)

### Tables

![02331e5b101d1f0d5fae3c049203655877f587dee57bfa240b9a2bf23c2cca83.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/02331e5b101d1f0d5fae3c049203655877f587dee57bfa240b9a2bf23c2cca83.jpg)

![03c90599c464c154355ad2cb88c637fdfec3e2e41277d55715598dc0692c96cd.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/03c90599c464c154355ad2cb88c637fdfec3e2e41277d55715598dc0692c96cd.jpg)

![167303d7f10a5cd584b9d23617e8f3b7e921ef076e904f3c680793a72c49a870.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/167303d7f10a5cd584b9d23617e8f3b7e921ef076e904f3c680793a72c49a870.jpg)

![1bbc028dde09beceff06d5334dbef6f29241dd230f2c6d87fc8d7306fbdc4dd0.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/1bbc028dde09beceff06d5334dbef6f29241dd230f2c6d87fc8d7306fbdc4dd0.jpg)

![27382812586ff1f1e9f447f7cc5cda9e11c075d52bc8f3afff63b54e903959b0.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/27382812586ff1f1e9f447f7cc5cda9e11c075d52bc8f3afff63b54e903959b0.jpg)

![2b4db79eee558a8bb6a92f846c8f003d82c2bfc25f4ba227239f107c1e125f04.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/2b4db79eee558a8bb6a92f846c8f003d82c2bfc25f4ba227239f107c1e125f04.jpg)

![2d4bab475a793cc9708fe982de42520d28abc42279b89c4c71ada05c913b8622.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/2d4bab475a793cc9708fe982de42520d28abc42279b89c4c71ada05c913b8622.jpg)

![2dca98698b722a4fc607657a4a78354e1b7d165c566a55b6f7df23671131c5fa.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/2dca98698b722a4fc607657a4a78354e1b7d165c566a55b6f7df23671131c5fa.jpg)

![424be1ed6053623be976078698ef956a88d666ee389ac3674a91cd57c0d1d0ed.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/424be1ed6053623be976078698ef956a88d666ee389ac3674a91cd57c0d1d0ed.jpg)

![44352ab95ce1196603290af4d55b02be2c731a45a5c5dd8952dd741098c84dcb.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/44352ab95ce1196603290af4d55b02be2c731a45a5c5dd8952dd741098c84dcb.jpg)

![496864b315df1c89cc14ac365e7ed8cc10d59570d2c3d1aef631cebe4e087dba.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/496864b315df1c89cc14ac365e7ed8cc10d59570d2c3d1aef631cebe4e087dba.jpg)

![5df89879a06a5074426e7894e678f5194ede2a482c4f9c14982fe3dcce1ca464.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/5df89879a06a5074426e7894e678f5194ede2a482c4f9c14982fe3dcce1ca464.jpg)

![6bb3f9ce38f15e2ae3bf2e490c170abf558db8180ce51a92febd838825f7b32e.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/6bb3f9ce38f15e2ae3bf2e490c170abf558db8180ce51a92febd838825f7b32e.jpg)

![6c7c8b77abf3c54f6fa358569ecbbf072e0e20ebd180adba1d653c6654aed5a9.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/6c7c8b77abf3c54f6fa358569ecbbf072e0e20ebd180adba1d653c6654aed5a9.jpg)

![7170a6e70d6ae8159b959053203c3934c709ac10dafbc52ebaed5af301d6bca9.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/7170a6e70d6ae8159b959053203c3934c709ac10dafbc52ebaed5af301d6bca9.jpg)

![78bad3cfe521fe60e7dcfc3ece997dd0bbdc29dc0105c5e13e5244c9cdc085e4.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/78bad3cfe521fe60e7dcfc3ece997dd0bbdc29dc0105c5e13e5244c9cdc085e4.jpg)

![7d59671c10ca0bee905782d561792a0c15fa665dc1e29e2bd7cc966cea544b09.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/7d59671c10ca0bee905782d561792a0c15fa665dc1e29e2bd7cc966cea544b09.jpg)

![866c25940d2c2d5866bcbac604a386ef92d492a10db234d1ac480d3b3e31801e.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/866c25940d2c2d5866bcbac604a386ef92d492a10db234d1ac480d3b3e31801e.jpg)

![a31a4f5ad91d418c47f06f58da1a75f279aa3ff5445aa25dcca49d99ae0f4877.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/a31a4f5ad91d418c47f06f58da1a75f279aa3ff5445aa25dcca49d99ae0f4877.jpg)

![acd91e8144430b70b0c513fc6502c79187830337c6925a8f23831d02a23892de.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/acd91e8144430b70b0c513fc6502c79187830337c6925a8f23831d02a23892de.jpg)

![bfa05a641f1f8fe29ca6dbd3a967fb672b71ccc06fa56d9ea6523c832f7650fc.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/bfa05a641f1f8fe29ca6dbd3a967fb672b71ccc06fa56d9ea6523c832f7650fc.jpg)

![c28bf5494c190b3b50d8524e0f583fcfcf85e1af8baa208708d4f1adb58709f2.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/c28bf5494c190b3b50d8524e0f583fcfcf85e1af8baa208708d4f1adb58709f2.jpg)

![c565eb1616eefb3dbd13601163e0dfc9fcb518eb51a3cf7e198a4d936732bf56.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/c565eb1616eefb3dbd13601163e0dfc9fcb518eb51a3cf7e198a4d936732bf56.jpg)

![d394df3a6e5529feac269ded9f1ffa0ba61241a632d25fdfdd21159553c06c16.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/d394df3a6e5529feac269ded9f1ffa0ba61241a632d25fdfdd21159553c06c16.jpg)

![dcdd68bdc5fdc58938a67d55edbed73dc36fea858e496ea1d5849d8f5f6bbf75.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/dcdd68bdc5fdc58938a67d55edbed73dc36fea858e496ea1d5849d8f5f6bbf75.jpg)

![e1b2c5953b418bf8a42bee0eb68022eec73397c32be4728d81dc3d5ec9637a60.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/e1b2c5953b418bf8a42bee0eb68022eec73397c32be4728d81dc3d5ec9637a60.jpg)

![f2f84353c38e3a9444b12341b41c0bd2a1b357a90c9375870d4cc8bea6fcf27f.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/f2f84353c38e3a9444b12341b41c0bd2a1b357a90c9375870d4cc8bea6fcf27f.jpg)

![f9912ef6621be7ae5212cfc5e4fa124734d9d7a26eeafdab63f89e8e124477f0.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/f9912ef6621be7ae5212cfc5e4fa124734d9d7a26eeafdab63f89e8e124477f0.jpg)

![fd57dd9ac7a711111d70d95bc093eeec23be9d4c3c0d84d0049f1115e14815b6.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/fd57dd9ac7a711111d70d95bc093eeec23be9d4c3c0d84d0049f1115e14815b6.jpg)

![fdccb8e7c2b28e72f12831bb5718500ea51058d4f6514dbdd2ab3ef8da001f6d.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/fdccb8e7c2b28e72f12831bb5718500ea51058d4f6514dbdd2ab3ef8da001f6d.jpg)

![fef105b1cdd407e8f4867c14616cab10477bf7295052f648b4e1dd6cc5a5ef9f.jpg](../iclr_results/1519_Valid%20Conformal%20Prediction%20for%20Dynamic%20GNNs/tables/fef105b1cdd407e8f4867c14616cab10477bf7295052f648b4e1dd6cc5a5ef9f.jpg)

## ECHOPulse: ECG Controlled Echocardio-gram Video Generation


### Images

![06b66f92ffcbd4bd6d1dd884422a40e2f66688cae42c3f826cff3ad568755997.jpg](../iclr_results/1520_ECHOPulse_%20ECG%20Controlled%20Echocardio-gram%20Video%20Generation/images/06b66f92ffcbd4bd6d1dd884422a40e2f66688cae42c3f826cff3ad568755997.jpg)

![2868598b064bba1762e3ef265eb0a80a389730745d2011dd47f7ea5da62ab96c.jpg](../iclr_results/1520_ECHOPulse_%20ECG%20Controlled%20Echocardio-gram%20Video%20Generation/images/2868598b064bba1762e3ef265eb0a80a389730745d2011dd47f7ea5da62ab96c.jpg)

![4d3afed353151fe583c911660d10e41bf7121a7a228dc670a6b2dfdea02d09e2.jpg](../iclr_results/1520_ECHOPulse_%20ECG%20Controlled%20Echocardio-gram%20Video%20Generation/images/4d3afed353151fe583c911660d10e41bf7121a7a228dc670a6b2dfdea02d09e2.jpg)

![573a932d68c6119cbb7c7a43481d4f6af61c9bfc6afa5ea6d83db084858e0975.jpg](../iclr_results/1520_ECHOPulse_%20ECG%20Controlled%20Echocardio-gram%20Video%20Generation/images/573a932d68c6119cbb7c7a43481d4f6af61c9bfc6afa5ea6d83db084858e0975.jpg)

![60b546d3343f0c4b2e671db133a776c1ce391f74fc8e9b1f34b88b58ea862491.jpg](../iclr_results/1520_ECHOPulse_%20ECG%20Controlled%20Echocardio-gram%20Video%20Generation/images/60b546d3343f0c4b2e671db133a776c1ce391f74fc8e9b1f34b88b58ea862491.jpg)

![6c2c4055fbf6a1e79cba36f417ca773eebad8b85c734a5117174e468c823beea.jpg](../iclr_results/1520_ECHOPulse_%20ECG%20Controlled%20Echocardio-gram%20Video%20Generation/images/6c2c4055fbf6a1e79cba36f417ca773eebad8b85c734a5117174e468c823beea.jpg)

![70eb657438bb8c21a0e2e1e1aad55b2779572844d615f8b02db9a206c1ed5699.jpg](../iclr_results/1520_ECHOPulse_%20ECG%20Controlled%20Echocardio-gram%20Video%20Generation/images/70eb657438bb8c21a0e2e1e1aad55b2779572844d615f8b02db9a206c1ed5699.jpg)

![ae92fa62f98161c2150632f55c0f374e04d919348865fb6942438f53caeaccfd.jpg](../iclr_results/1520_ECHOPulse_%20ECG%20Controlled%20Echocardio-gram%20Video%20Generation/images/ae92fa62f98161c2150632f55c0f374e04d919348865fb6942438f53caeaccfd.jpg)

![bb89d246e709e844d6f6d5dbd70393b247f787ab3c42861cc6d685eddb1e913e.jpg](../iclr_results/1520_ECHOPulse_%20ECG%20Controlled%20Echocardio-gram%20Video%20Generation/images/bb89d246e709e844d6f6d5dbd70393b247f787ab3c42861cc6d685eddb1e913e.jpg)

![e6258fc03aee0477b7d2c3b60fb9c92bdb6e92d34ef177f2ac9a10c579ed2c0c.jpg](../iclr_results/1520_ECHOPulse_%20ECG%20Controlled%20Echocardio-gram%20Video%20Generation/images/e6258fc03aee0477b7d2c3b60fb9c92bdb6e92d34ef177f2ac9a10c579ed2c0c.jpg)

![f5b5939b8510fb7390eeff9390587c9603a6bc955a82589253f8d5f039135eb8.jpg](../iclr_results/1520_ECHOPulse_%20ECG%20Controlled%20Echocardio-gram%20Video%20Generation/images/f5b5939b8510fb7390eeff9390587c9603a6bc955a82589253f8d5f039135eb8.jpg)

### Tables

![0114b1f4da5e3e2f4e53d9707811cf66f80473413d4a80c573c3f2b2095c5ca4.jpg](../iclr_results/1520_ECHOPulse_%20ECG%20Controlled%20Echocardio-gram%20Video%20Generation/tables/0114b1f4da5e3e2f4e53d9707811cf66f80473413d4a80c573c3f2b2095c5ca4.jpg)

![0186a03a1e780ee32c40bb7cca78a99766ba21f20e81221ed8221d6089b95200.jpg](../iclr_results/1520_ECHOPulse_%20ECG%20Controlled%20Echocardio-gram%20Video%20Generation/tables/0186a03a1e780ee32c40bb7cca78a99766ba21f20e81221ed8221d6089b95200.jpg)

![5a792caa77cf3209503e87a0a3fab998a3fb93c62a9a6f2fb46dc807baccd9f0.jpg](../iclr_results/1520_ECHOPulse_%20ECG%20Controlled%20Echocardio-gram%20Video%20Generation/tables/5a792caa77cf3209503e87a0a3fab998a3fb93c62a9a6f2fb46dc807baccd9f0.jpg)

![6cf0198b0b0897519083af0b7195755722ed2ff9ef50a8b97a3f215853a1bf70.jpg](../iclr_results/1520_ECHOPulse_%20ECG%20Controlled%20Echocardio-gram%20Video%20Generation/tables/6cf0198b0b0897519083af0b7195755722ed2ff9ef50a8b97a3f215853a1bf70.jpg)

![716075b581368db0c9042118a64006f267cac55a768470d0197192557dd28a09.jpg](../iclr_results/1520_ECHOPulse_%20ECG%20Controlled%20Echocardio-gram%20Video%20Generation/tables/716075b581368db0c9042118a64006f267cac55a768470d0197192557dd28a09.jpg)

![78acde5e36b4de1f7cbb72fd228317fc325abb3924fb53ccaa46b40cb01d3c1d.jpg](../iclr_results/1520_ECHOPulse_%20ECG%20Controlled%20Echocardio-gram%20Video%20Generation/tables/78acde5e36b4de1f7cbb72fd228317fc325abb3924fb53ccaa46b40cb01d3c1d.jpg)

![bfd07c7a5c0637556b810b47fdc547c8d0a175ba2194926f4f15cdaac8958701.jpg](../iclr_results/1520_ECHOPulse_%20ECG%20Controlled%20Echocardio-gram%20Video%20Generation/tables/bfd07c7a5c0637556b810b47fdc547c8d0a175ba2194926f4f15cdaac8958701.jpg)

## SymDiff: Equivariant Diffusion via Stochastic Symmetrisation


### Images

![4d3b0e1762ca672b9012705b5f6291154e83676576d4d079d7dabec585aa782c.jpg](../iclr_results/1521_SymDiff_%20Equivariant%20Diffusion%20via%20Stochastic%20Symmetrisation/images/4d3b0e1762ca672b9012705b5f6291154e83676576d4d079d7dabec585aa782c.jpg)

![b2b364542a1c47165796886a89280b159c4b47d82d2020f081f1cc7c08018408.jpg](../iclr_results/1521_SymDiff_%20Equivariant%20Diffusion%20via%20Stochastic%20Symmetrisation/images/b2b364542a1c47165796886a89280b159c4b47d82d2020f081f1cc7c08018408.jpg)

### Tables

![4ac5cb552c79880bbb601452799046e21f3e6c71fcd80eb1e0629a8ba5c68d09.jpg](../iclr_results/1521_SymDiff_%20Equivariant%20Diffusion%20via%20Stochastic%20Symmetrisation/tables/4ac5cb552c79880bbb601452799046e21f3e6c71fcd80eb1e0629a8ba5c68d09.jpg)

![8b6fabc406945c68e71f156852064aafb73dd88186a383dd620eb9f0ad85dbe4.jpg](../iclr_results/1521_SymDiff_%20Equivariant%20Diffusion%20via%20Stochastic%20Symmetrisation/tables/8b6fabc406945c68e71f156852064aafb73dd88186a383dd620eb9f0ad85dbe4.jpg)

![99d1253720593881666a0d73b4e9bd05bad5ae1c2f498a1960eeba886598a52f.jpg](../iclr_results/1521_SymDiff_%20Equivariant%20Diffusion%20via%20Stochastic%20Symmetrisation/tables/99d1253720593881666a0d73b4e9bd05bad5ae1c2f498a1960eeba886598a52f.jpg)

![a03af8601579405c84d3ce4a9a0ae0c0e1a04068a1ec4bc0960d67a10e3673ad.jpg](../iclr_results/1521_SymDiff_%20Equivariant%20Diffusion%20via%20Stochastic%20Symmetrisation/tables/a03af8601579405c84d3ce4a9a0ae0c0e1a04068a1ec4bc0960d67a10e3673ad.jpg)

![a1d45b5c408d5a545f1ce2a439bdea6923956140ffec58dcf11fa57824acaa69.jpg](../iclr_results/1521_SymDiff_%20Equivariant%20Diffusion%20via%20Stochastic%20Symmetrisation/tables/a1d45b5c408d5a545f1ce2a439bdea6923956140ffec58dcf11fa57824acaa69.jpg)

![adddec6f7ed96b9f1ca936a1eb65989d12f4b560bb4e43101c9571017dacbe09.jpg](../iclr_results/1521_SymDiff_%20Equivariant%20Diffusion%20via%20Stochastic%20Symmetrisation/tables/adddec6f7ed96b9f1ca936a1eb65989d12f4b560bb4e43101c9571017dacbe09.jpg)

![bfed08eff842c58f07c202fde379c504d53aeb55d6f20f3cb430109ba4907fc4.jpg](../iclr_results/1521_SymDiff_%20Equivariant%20Diffusion%20via%20Stochastic%20Symmetrisation/tables/bfed08eff842c58f07c202fde379c504d53aeb55d6f20f3cb430109ba4907fc4.jpg)

![e7b0f08caa3cbf2b0e01752161724faef27cf801354a5690fce8cd1377661ec8.jpg](../iclr_results/1521_SymDiff_%20Equivariant%20Diffusion%20via%20Stochastic%20Symmetrisation/tables/e7b0f08caa3cbf2b0e01752161724faef27cf801354a5690fce8cd1377661ec8.jpg)

## Parameter and Memory Efficient Pretraining via Low-rank Riemannian Optimization


### Images

![01247261861aeeab6975c796a441c6a80272ed4989142a3380391093aeb2aff4.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/images/01247261861aeeab6975c796a441c6a80272ed4989142a3380391093aeb2aff4.jpg)

![02573f8507de67543000e41b6cb27502a3f0b34dfc265f353dee48ca157ff7cf.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/images/02573f8507de67543000e41b6cb27502a3f0b34dfc265f353dee48ca157ff7cf.jpg)

![14274fa7ac927e7b47eb1b4bb84b13deba75015ef400e5b413778d2505f2fbfa.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/images/14274fa7ac927e7b47eb1b4bb84b13deba75015ef400e5b413778d2505f2fbfa.jpg)

![1a20d99ea60d024aeeb00b078c4bb1b9471e14224957711d244f1aed3929f7bf.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/images/1a20d99ea60d024aeeb00b078c4bb1b9471e14224957711d244f1aed3929f7bf.jpg)

![1f5c94e62714d614b336b9dd7af561351b456a942977647f24fea606af136a02.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/images/1f5c94e62714d614b336b9dd7af561351b456a942977647f24fea606af136a02.jpg)

![2f4eb5b96ddfb54b63e4f7f47990c39b763e41c42ae1d567bb54716d247eccd1.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/images/2f4eb5b96ddfb54b63e4f7f47990c39b763e41c42ae1d567bb54716d247eccd1.jpg)

![3bbbf4c4b496fa83792d52d37730008120814889a805f839c0054c4f54467aaf.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/images/3bbbf4c4b496fa83792d52d37730008120814889a805f839c0054c4f54467aaf.jpg)

![5aa9234f060950621ee6f351746a4cc470e9f38c16260ea5c86b0c5d5c0e375f.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/images/5aa9234f060950621ee6f351746a4cc470e9f38c16260ea5c86b0c5d5c0e375f.jpg)

![724f8f5e3d325b0c515d5921e9bc6639ec4cad163c08c6249854094cf1cd34c8.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/images/724f8f5e3d325b0c515d5921e9bc6639ec4cad163c08c6249854094cf1cd34c8.jpg)

![7692945cde64593ffe80831f06965d6658cca0e9f39a2d8155b939f86e833b1c.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/images/7692945cde64593ffe80831f06965d6658cca0e9f39a2d8155b939f86e833b1c.jpg)

![7780ee230c5b55cb1dfa79bd8e1e165fb655b04f416d6827366f3a93e0c303f7.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/images/7780ee230c5b55cb1dfa79bd8e1e165fb655b04f416d6827366f3a93e0c303f7.jpg)

![955d322d53794f9a18cdb1f9379efe8f74987ff073dd9bf03011683d580f60f4.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/images/955d322d53794f9a18cdb1f9379efe8f74987ff073dd9bf03011683d580f60f4.jpg)

![98d38fe61ab7690f980200132895e45e324c21a372db8907e2be65d00e9acda1.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/images/98d38fe61ab7690f980200132895e45e324c21a372db8907e2be65d00e9acda1.jpg)

![9c7830f13ad5e808915ac1d3e4617c70cc5d952cd7ac95355bc2c8eb012cf73a.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/images/9c7830f13ad5e808915ac1d3e4617c70cc5d952cd7ac95355bc2c8eb012cf73a.jpg)

![9c99e8d52d697d3ae7f6e4b5eaca6e6549e95ceea743477fa69803c762a18059.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/images/9c99e8d52d697d3ae7f6e4b5eaca6e6549e95ceea743477fa69803c762a18059.jpg)

![bdb09d54fa3e14b0e80acb865c0f4ee510cf76ac273674ecf7260cb6123ec65e.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/images/bdb09d54fa3e14b0e80acb865c0f4ee510cf76ac273674ecf7260cb6123ec65e.jpg)

![dbbf5287831411abd804d64f2f4ba424fa7790b73be0fbd527a958b36304f4fe.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/images/dbbf5287831411abd804d64f2f4ba424fa7790b73be0fbd527a958b36304f4fe.jpg)

![f9605e6fc391f3546f7d09a29b379699ef62b3083d9de7324a4645cdbec32ece.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/images/f9605e6fc391f3546f7d09a29b379699ef62b3083d9de7324a4645cdbec32ece.jpg)

### Tables

![0e5e71bab71e5fa051d5805bd9e30be8b9bedaf708b6309d56e06d55ce6067ab.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/tables/0e5e71bab71e5fa051d5805bd9e30be8b9bedaf708b6309d56e06d55ce6067ab.jpg)

![1985960f82ac415f8ada3d99aedd8c72f00f70389b1737043cd92a37e89b9e53.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/tables/1985960f82ac415f8ada3d99aedd8c72f00f70389b1737043cd92a37e89b9e53.jpg)

![268f5b92df48271ed244e1f8c78291a1c28033901673cc7465fb1ab1d76e8845.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/tables/268f5b92df48271ed244e1f8c78291a1c28033901673cc7465fb1ab1d76e8845.jpg)

![63c2153e65b570ec242ea8471ad2e5bbcaa9f1bdaa19ed5f5b235d2862e50dc0.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/tables/63c2153e65b570ec242ea8471ad2e5bbcaa9f1bdaa19ed5f5b235d2862e50dc0.jpg)

![867df9b31bb341fa39093dde74df521e4589944ffb28bd6b3ef2c9bf5a530e67.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/tables/867df9b31bb341fa39093dde74df521e4589944ffb28bd6b3ef2c9bf5a530e67.jpg)

![b9a27e0ba26cda7db8f217a4f14a8114257c5e261170074007a79ba3ef717427.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/tables/b9a27e0ba26cda7db8f217a4f14a8114257c5e261170074007a79ba3ef717427.jpg)

![c7f7010ecd09b4ac4363d867521e73c4c12a8b799e89b42324b99ff51f63a860.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/tables/c7f7010ecd09b4ac4363d867521e73c4c12a8b799e89b42324b99ff51f63a860.jpg)

![dd0578f283dc418756ff9f412479e28194129ff31353d3893d17c941de3f4daa.jpg](../iclr_results/1522_Parameter%20and%20Memory%20Efficient%20Pretraining%20via%20Low-rank%20Riemannian%20Optimization/tables/dd0578f283dc418756ff9f412479e28194129ff31353d3893d17c941de3f4daa.jpg)

## Deep Distributed Optimization for Large-Scale Quadratic Programming


### Images

![11482249414a11e86719fe79d04366390bd2f1b6e8e76433c06d3ea5b9cbe7e8.jpg](../iclr_results/1523_Deep%20Distributed%20Optimization%20for%20Large-Scale%20Quadratic%20Programming/images/11482249414a11e86719fe79d04366390bd2f1b6e8e76433c06d3ea5b9cbe7e8.jpg)

![178fae91deb4fdcbc5d4b3b988bf2c5e252f74ec68aa7eef0120cf4e49eb8ffc.jpg](../iclr_results/1523_Deep%20Distributed%20Optimization%20for%20Large-Scale%20Quadratic%20Programming/images/178fae91deb4fdcbc5d4b3b988bf2c5e252f74ec68aa7eef0120cf4e49eb8ffc.jpg)

![305e04beb21aa80f5294ba8c7283386af809f6f70beb4359e47200517915f1f5.jpg](../iclr_results/1523_Deep%20Distributed%20Optimization%20for%20Large-Scale%20Quadratic%20Programming/images/305e04beb21aa80f5294ba8c7283386af809f6f70beb4359e47200517915f1f5.jpg)

![396f7c0117ec173a89b41738d44cc7c442803a0809c9896161e9fbe5a0865b75.jpg](../iclr_results/1523_Deep%20Distributed%20Optimization%20for%20Large-Scale%20Quadratic%20Programming/images/396f7c0117ec173a89b41738d44cc7c442803a0809c9896161e9fbe5a0865b75.jpg)

![4c1b29ba482ab16bda7f0d7e763e3129be3b5dc6eab6515e12bca6a89098a665.jpg](../iclr_results/1523_Deep%20Distributed%20Optimization%20for%20Large-Scale%20Quadratic%20Programming/images/4c1b29ba482ab16bda7f0d7e763e3129be3b5dc6eab6515e12bca6a89098a665.jpg)

![78289bb4e77b00bfc4f4368a8d8512d5830f9dffdc7aa8a7b4646b534e682c77.jpg](../iclr_results/1523_Deep%20Distributed%20Optimization%20for%20Large-Scale%20Quadratic%20Programming/images/78289bb4e77b00bfc4f4368a8d8512d5830f9dffdc7aa8a7b4646b534e682c77.jpg)

![9ffe530cc95008b2d6205020695b97bfae6e731808a2b2e6daa690d1d961c2cf.jpg](../iclr_results/1523_Deep%20Distributed%20Optimization%20for%20Large-Scale%20Quadratic%20Programming/images/9ffe530cc95008b2d6205020695b97bfae6e731808a2b2e6daa690d1d961c2cf.jpg)

![c374df95d1a8892566cbe8681a486161b40b755e62f49c8faf4a2da68a29aa91.jpg](../iclr_results/1523_Deep%20Distributed%20Optimization%20for%20Large-Scale%20Quadratic%20Programming/images/c374df95d1a8892566cbe8681a486161b40b755e62f49c8faf4a2da68a29aa91.jpg)

![d36acd17bb18e6a26ae35404cc159576e9ae0127129ace266af8cc001049d341.jpg](../iclr_results/1523_Deep%20Distributed%20Optimization%20for%20Large-Scale%20Quadratic%20Programming/images/d36acd17bb18e6a26ae35404cc159576e9ae0127129ace266af8cc001049d341.jpg)

![e964d88510522c168c1fd3842a96eaab20d9d8b523e661ec1942692c7741be04.jpg](../iclr_results/1523_Deep%20Distributed%20Optimization%20for%20Large-Scale%20Quadratic%20Programming/images/e964d88510522c168c1fd3842a96eaab20d9d8b523e661ec1942692c7741be04.jpg)

![eaa4689677c73bbc7cbd0f2b6729c362aa4eb36a3ddc81e81a35b164a2bc55a9.jpg](../iclr_results/1523_Deep%20Distributed%20Optimization%20for%20Large-Scale%20Quadratic%20Programming/images/eaa4689677c73bbc7cbd0f2b6729c362aa4eb36a3ddc81e81a35b164a2bc55a9.jpg)

### Tables

![6665b2a45a81dd230e7cef4595ca75f03b08bffb32914df05d81db31abfb9f6b.jpg](../iclr_results/1523_Deep%20Distributed%20Optimization%20for%20Large-Scale%20Quadratic%20Programming/tables/6665b2a45a81dd230e7cef4595ca75f03b08bffb32914df05d81db31abfb9f6b.jpg)

![81c94a1cb6003d6e3780afb4c712b643ca17fb9f80687706d6f3cd4040b50d40.jpg](../iclr_results/1523_Deep%20Distributed%20Optimization%20for%20Large-Scale%20Quadratic%20Programming/tables/81c94a1cb6003d6e3780afb4c712b643ca17fb9f80687706d6f3cd4040b50d40.jpg)

![8d58a066312fde140831d2c0bf10dde592dd83633563a7a15755271c7be3b098.jpg](../iclr_results/1523_Deep%20Distributed%20Optimization%20for%20Large-Scale%20Quadratic%20Programming/tables/8d58a066312fde140831d2c0bf10dde592dd83633563a7a15755271c7be3b098.jpg)

![9a10c05a2c7587c0ac5520a7bb9c53e93372ae900c4e9563a9f6345a6028c386.jpg](../iclr_results/1523_Deep%20Distributed%20Optimization%20for%20Large-Scale%20Quadratic%20Programming/tables/9a10c05a2c7587c0ac5520a7bb9c53e93372ae900c4e9563a9f6345a6028c386.jpg)

![b45a98325c24bbacb78be65a15855b061d98d00afad0a058cb0f108976d625b5.jpg](../iclr_results/1523_Deep%20Distributed%20Optimization%20for%20Large-Scale%20Quadratic%20Programming/tables/b45a98325c24bbacb78be65a15855b061d98d00afad0a058cb0f108976d625b5.jpg)

## Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification


### Images

![0ed019f513389040424626951e4ee7fa5d3653758e2ac353c1cdf84c5e42e172.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/images/0ed019f513389040424626951e4ee7fa5d3653758e2ac353c1cdf84c5e42e172.jpg)

![1f6e63c2de4682a5fa8be12209b5a9614652def333ccd51528682e84bf432531.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/images/1f6e63c2de4682a5fa8be12209b5a9614652def333ccd51528682e84bf432531.jpg)

![23645d440a7b19ef4b94243562a80665a6c090f76c5bb13494c9724de2f9c725.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/images/23645d440a7b19ef4b94243562a80665a6c090f76c5bb13494c9724de2f9c725.jpg)

![478b330aaff99a85fe0c796d1f9c75e6872c4b77424949310bd341d126340f54.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/images/478b330aaff99a85fe0c796d1f9c75e6872c4b77424949310bd341d126340f54.jpg)

![6bdc7980bb1a81b64cbe8b35773c06e7b5b5d770e667da82c39e3d78623fbde4.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/images/6bdc7980bb1a81b64cbe8b35773c06e7b5b5d770e667da82c39e3d78623fbde4.jpg)

![825521337a633201c89f276cefd77f820eca0450c2d9109baa5222fdf3e8e611.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/images/825521337a633201c89f276cefd77f820eca0450c2d9109baa5222fdf3e8e611.jpg)

![ab5ad11d8cb5fb63f22e6b942849a227585b7126c1c0a618ebd1db5c48fa055b.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/images/ab5ad11d8cb5fb63f22e6b942849a227585b7126c1c0a618ebd1db5c48fa055b.jpg)

![bde52c54f40746adc00b9a61ce927aa748c5280266fed9d96a72fd24f625ccae.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/images/bde52c54f40746adc00b9a61ce927aa748c5280266fed9d96a72fd24f625ccae.jpg)

![d03e52111449b2e7f444617c3d28eaff71f40137470d20be820447ca2c8ed550.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/images/d03e52111449b2e7f444617c3d28eaff71f40137470d20be820447ca2c8ed550.jpg)

![e6550fb432b296b0ac0df2122383c2c7306fde34dca9680687aee48c56929152.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/images/e6550fb432b296b0ac0df2122383c2c7306fde34dca9680687aee48c56929152.jpg)

![eb8bc126b41f9b62f08e455bce6a98a22d32da37b5b9422952fea5d2468fc369.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/images/eb8bc126b41f9b62f08e455bce6a98a22d32da37b5b9422952fea5d2468fc369.jpg)

### Tables

![054980625e23ccadb7eca9cb3212f4ec293d7a9f12df83018658a94f78167bd6.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/tables/054980625e23ccadb7eca9cb3212f4ec293d7a9f12df83018658a94f78167bd6.jpg)

![094f876d54d152a0c15667b921b0c2622637355709d33126e89041df9f391d99.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/tables/094f876d54d152a0c15667b921b0c2622637355709d33126e89041df9f391d99.jpg)

![0f7e165fca75406f0bc42102a9174c0731bbd7213ca755db20dee12257ee6997.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/tables/0f7e165fca75406f0bc42102a9174c0731bbd7213ca755db20dee12257ee6997.jpg)

![13c37fa6b9b45ffcd3545628d828e38f1ebc78172987e4fb28d31bfcba708573.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/tables/13c37fa6b9b45ffcd3545628d828e38f1ebc78172987e4fb28d31bfcba708573.jpg)

![3d129fa0e4e9254d70a8c1e4329a35f9c0b586fb79b1822c1ae39b439db32f38.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/tables/3d129fa0e4e9254d70a8c1e4329a35f9c0b586fb79b1822c1ae39b439db32f38.jpg)

![4dc07f5dca9856825a80f2da26febe6d74c64a1aace1eccf513a9df24feceb3d.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/tables/4dc07f5dca9856825a80f2da26febe6d74c64a1aace1eccf513a9df24feceb3d.jpg)

![52b756c935a4f32aed201f3a871fdaa8bef86d7e67a005aff583f0ce8ab2683b.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/tables/52b756c935a4f32aed201f3a871fdaa8bef86d7e67a005aff583f0ce8ab2683b.jpg)

![5b0faa26e7bfa718d9708fd9d58081b10105eef38b53a2c86297b78d5936dc42.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/tables/5b0faa26e7bfa718d9708fd9d58081b10105eef38b53a2c86297b78d5936dc42.jpg)

![5e00420eb2771e4adb23465922218cfa8f2678cbb193d4f3426f2fd4f2f7879e.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/tables/5e00420eb2771e4adb23465922218cfa8f2678cbb193d4f3426f2fd4f2f7879e.jpg)

![9297d01b2b1c1ec86514bbf6fa52eead4f259b6bbaa759d1acde70c13106b086.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/tables/9297d01b2b1c1ec86514bbf6fa52eead4f259b6bbaa759d1acde70c13106b086.jpg)

![a8375f47d48ba2c04133e810c54aa0e654699e1e7e0a7ece4ab4514944f73053.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/tables/a8375f47d48ba2c04133e810c54aa0e654699e1e7e0a7ece4ab4514944f73053.jpg)

![a9bd71f4db35a801156793307a875c0913df4b050d43d0f84682a82b378391f7.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/tables/a9bd71f4db35a801156793307a875c0913df4b050d43d0f84682a82b378391f7.jpg)

![b583137329748b994e0e45ff4c16ccf74aa8c75de15ad03ae0fe4cf82da20712.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/tables/b583137329748b994e0e45ff4c16ccf74aa8c75de15ad03ae0fe4cf82da20712.jpg)

![bc0854249fa245d80c1a646f8f93a3b34d4fbfe3e580f813156bc54e615c719b.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/tables/bc0854249fa245d80c1a646f8f93a3b34d4fbfe3e580f813156bc54e615c719b.jpg)

![c0483f9cca38f0b700d0586c0d18e774ce4a42962f33680e91779a2f83554dee.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/tables/c0483f9cca38f0b700d0586c0d18e774ce4a42962f33680e91779a2f83554dee.jpg)

![c5079fd63a148b37d257536244c3a22468549de1e7c17cd5c40155087b70c66c.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/tables/c5079fd63a148b37d257536244c3a22468549de1e7c17cd5c40155087b70c66c.jpg)

![f0a10dfa266a67a9749b6b32f0397e211d26f2ef92bcc6bb4ced4157f9d0a41a.jpg](../iclr_results/1524_Dynamic-LLaVA_%20Efficient%20Multimodal%20Large%20Language%20Models%20via%20Dynamic%20Vision-language%20Context%20Sparsi/tables/f0a10dfa266a67a9749b6b32f0397e211d26f2ef92bcc6bb4ced4157f9d0a41a.jpg)

## Neural Stochastic Differential Equations for Uncertainty-Aware Offline RL


### Images

![77bdefc52dc90001c27075ead98f3f551caae56ee9dcef7b0ebc37712a59ac59.jpg](../iclr_results/1525_Neural%20Stochastic%20Differential%20Equations%20for%20Uncertainty-Aware%20Offline%20RL/images/77bdefc52dc90001c27075ead98f3f551caae56ee9dcef7b0ebc37712a59ac59.jpg)

![8847639be5317a95910ef7d58ae50f4cb1d668a65c4d4fd0542ff30683f9c20e.jpg](../iclr_results/1525_Neural%20Stochastic%20Differential%20Equations%20for%20Uncertainty-Aware%20Offline%20RL/images/8847639be5317a95910ef7d58ae50f4cb1d668a65c4d4fd0542ff30683f9c20e.jpg)

![8994e7365e0b0cca907ab917a4f07e477e13d58d1925e8b3006ca3dfc109bcc8.jpg](../iclr_results/1525_Neural%20Stochastic%20Differential%20Equations%20for%20Uncertainty-Aware%20Offline%20RL/images/8994e7365e0b0cca907ab917a4f07e477e13d58d1925e8b3006ca3dfc109bcc8.jpg)

![8daf07f3fd2dd1aaad06a6e6fba91b3a2f160f125d15d3e033f155994f772db3.jpg](../iclr_results/1525_Neural%20Stochastic%20Differential%20Equations%20for%20Uncertainty-Aware%20Offline%20RL/images/8daf07f3fd2dd1aaad06a6e6fba91b3a2f160f125d15d3e033f155994f772db3.jpg)

![907983c27eed0b43b7bfd86feb72e5f8b50061dd961319d78809c5bac4dd078a.jpg](../iclr_results/1525_Neural%20Stochastic%20Differential%20Equations%20for%20Uncertainty-Aware%20Offline%20RL/images/907983c27eed0b43b7bfd86feb72e5f8b50061dd961319d78809c5bac4dd078a.jpg)

![9864f4a270643687093a5377411c4038dddb2409e29f2e9b342007bec4ece994.jpg](../iclr_results/1525_Neural%20Stochastic%20Differential%20Equations%20for%20Uncertainty-Aware%20Offline%20RL/images/9864f4a270643687093a5377411c4038dddb2409e29f2e9b342007bec4ece994.jpg)

![cc4269195c2607fdf169a8fc3397dcc765516806431c63ad430d6d0d45ae8771.jpg](../iclr_results/1525_Neural%20Stochastic%20Differential%20Equations%20for%20Uncertainty-Aware%20Offline%20RL/images/cc4269195c2607fdf169a8fc3397dcc765516806431c63ad430d6d0d45ae8771.jpg)

![d1f88725b3ac2f21d8a75016e6b2b9f28ed0184889a7e9a7b23955cb60549082.jpg](../iclr_results/1525_Neural%20Stochastic%20Differential%20Equations%20for%20Uncertainty-Aware%20Offline%20RL/images/d1f88725b3ac2f21d8a75016e6b2b9f28ed0184889a7e9a7b23955cb60549082.jpg)

![e9612816dae0402f4fed56ddfe3ff231f15517cabd733594f4c829fd1124b2c6.jpg](../iclr_results/1525_Neural%20Stochastic%20Differential%20Equations%20for%20Uncertainty-Aware%20Offline%20RL/images/e9612816dae0402f4fed56ddfe3ff231f15517cabd733594f4c829fd1124b2c6.jpg)

![f2e512718c549a162ba35a3d06de659a75fa2600efa3052d124481ef92c591a0.jpg](../iclr_results/1525_Neural%20Stochastic%20Differential%20Equations%20for%20Uncertainty-Aware%20Offline%20RL/images/f2e512718c549a162ba35a3d06de659a75fa2600efa3052d124481ef92c591a0.jpg)

![f7649520d5a16fd7b87c1305de79cc9ec9b7dd03a8bc9bb51d9bb05924c5915a.jpg](../iclr_results/1525_Neural%20Stochastic%20Differential%20Equations%20for%20Uncertainty-Aware%20Offline%20RL/images/f7649520d5a16fd7b87c1305de79cc9ec9b7dd03a8bc9bb51d9bb05924c5915a.jpg)

![ffca3904585feb23a051952e89517b3dcae7ceb437f544d362e5aa1854beaea7.jpg](../iclr_results/1525_Neural%20Stochastic%20Differential%20Equations%20for%20Uncertainty-Aware%20Offline%20RL/images/ffca3904585feb23a051952e89517b3dcae7ceb437f544d362e5aa1854beaea7.jpg)

### Tables

![32145e682a3f4ebb559a8bbd1d3e6a8b8634e1b51ef72875bb434ce9684508b5.jpg](../iclr_results/1525_Neural%20Stochastic%20Differential%20Equations%20for%20Uncertainty-Aware%20Offline%20RL/tables/32145e682a3f4ebb559a8bbd1d3e6a8b8634e1b51ef72875bb434ce9684508b5.jpg)

![965f4fd109778ee1d1ad8cf51eec1a35a1f16c35d59283df09221fee7138fb1d.jpg](../iclr_results/1525_Neural%20Stochastic%20Differential%20Equations%20for%20Uncertainty-Aware%20Offline%20RL/tables/965f4fd109778ee1d1ad8cf51eec1a35a1f16c35d59283df09221fee7138fb1d.jpg)

![b31853854068049bd29e7522833e9be39adc7386c77a4fb71b3c667e482b1f53.jpg](../iclr_results/1525_Neural%20Stochastic%20Differential%20Equations%20for%20Uncertainty-Aware%20Offline%20RL/tables/b31853854068049bd29e7522833e9be39adc7386c77a4fb71b3c667e482b1f53.jpg)

![e237b3bdb747db011721d6881c1846e6cf8c213a9ac2631ba187f7baf8c4dcd2.jpg](../iclr_results/1525_Neural%20Stochastic%20Differential%20Equations%20for%20Uncertainty-Aware%20Offline%20RL/tables/e237b3bdb747db011721d6881c1846e6cf8c213a9ac2631ba187f7baf8c4dcd2.jpg)

## DPaI: Differentiable Pruning at Initialization with Node-Path Balance Principle


### Images

![1ea1e01f7b493531818097c472323b141bf588db7cdfd6ed46ba14e8bf0406d0.jpg](../iclr_results/1526_DPaI_%20Differentiable%20Pruning%20at%20Initialization%20with%20Node-Path%20Balance%20Principle/images/1ea1e01f7b493531818097c472323b141bf588db7cdfd6ed46ba14e8bf0406d0.jpg)

![30f0d8b20980a79888405e4c63396bfad592002f36cd58657f52d6add951a3e5.jpg](../iclr_results/1526_DPaI_%20Differentiable%20Pruning%20at%20Initialization%20with%20Node-Path%20Balance%20Principle/images/30f0d8b20980a79888405e4c63396bfad592002f36cd58657f52d6add951a3e5.jpg)

![647af08c1b0a414cbe67c8591a1c3557ee50e65babbc714dd7b0f08686c0dd46.jpg](../iclr_results/1526_DPaI_%20Differentiable%20Pruning%20at%20Initialization%20with%20Node-Path%20Balance%20Principle/images/647af08c1b0a414cbe67c8591a1c3557ee50e65babbc714dd7b0f08686c0dd46.jpg)

![97cecf3c453d4748f0bb2ca4ecc90bfc09c059989b13d768563eb37e6365e162.jpg](../iclr_results/1526_DPaI_%20Differentiable%20Pruning%20at%20Initialization%20with%20Node-Path%20Balance%20Principle/images/97cecf3c453d4748f0bb2ca4ecc90bfc09c059989b13d768563eb37e6365e162.jpg)

![b7a225c5bef46286009f6bf57ed3b828159d54c9b7fd09e61947fa1c414f58ad.jpg](../iclr_results/1526_DPaI_%20Differentiable%20Pruning%20at%20Initialization%20with%20Node-Path%20Balance%20Principle/images/b7a225c5bef46286009f6bf57ed3b828159d54c9b7fd09e61947fa1c414f58ad.jpg)

### Tables

![121becda5088a84b5fbc9c234bf62779f35d5321f90d6845ca9532169c559785.jpg](../iclr_results/1526_DPaI_%20Differentiable%20Pruning%20at%20Initialization%20with%20Node-Path%20Balance%20Principle/tables/121becda5088a84b5fbc9c234bf62779f35d5321f90d6845ca9532169c559785.jpg)

![3a97c0b10022231ad38e71567f0306ef7dc5a67c236e3dbd0022ff4c89ebc73f.jpg](../iclr_results/1526_DPaI_%20Differentiable%20Pruning%20at%20Initialization%20with%20Node-Path%20Balance%20Principle/tables/3a97c0b10022231ad38e71567f0306ef7dc5a67c236e3dbd0022ff4c89ebc73f.jpg)

![462473888ac8d51361a2ae9ca39fe9820489806892160c29ba194b0cf7a60239.jpg](../iclr_results/1526_DPaI_%20Differentiable%20Pruning%20at%20Initialization%20with%20Node-Path%20Balance%20Principle/tables/462473888ac8d51361a2ae9ca39fe9820489806892160c29ba194b0cf7a60239.jpg)

![4fc501b006875a436d0866559a415afdcd82fb5db7f0b543d44d509c4a4db512.jpg](../iclr_results/1526_DPaI_%20Differentiable%20Pruning%20at%20Initialization%20with%20Node-Path%20Balance%20Principle/tables/4fc501b006875a436d0866559a415afdcd82fb5db7f0b543d44d509c4a4db512.jpg)

![8165cc0e0b29dfd562be6793af2dfb012c782c6a1d52a480bf83b8249a8c478c.jpg](../iclr_results/1526_DPaI_%20Differentiable%20Pruning%20at%20Initialization%20with%20Node-Path%20Balance%20Principle/tables/8165cc0e0b29dfd562be6793af2dfb012c782c6a1d52a480bf83b8249a8c478c.jpg)

![825bc734c0d0e581d0fe8694c08eecb4cd6a77f6c8ab7f9cc874b7768497a951.jpg](../iclr_results/1526_DPaI_%20Differentiable%20Pruning%20at%20Initialization%20with%20Node-Path%20Balance%20Principle/tables/825bc734c0d0e581d0fe8694c08eecb4cd6a77f6c8ab7f9cc874b7768497a951.jpg)

![c5ff62cdc54144e5b2c1d06161ae0fa6b8fdb27ec1823543c82301a3115489b6.jpg](../iclr_results/1526_DPaI_%20Differentiable%20Pruning%20at%20Initialization%20with%20Node-Path%20Balance%20Principle/tables/c5ff62cdc54144e5b2c1d06161ae0fa6b8fdb27ec1823543c82301a3115489b6.jpg)

![d73cb80925fdba25dfc64a523ebf306deaab9bcfadce2f640353bd6b883b8788.jpg](../iclr_results/1526_DPaI_%20Differentiable%20Pruning%20at%20Initialization%20with%20Node-Path%20Balance%20Principle/tables/d73cb80925fdba25dfc64a523ebf306deaab9bcfadce2f640353bd6b883b8788.jpg)

![e36ae66f0622034836749968b0a7758f29ade3a9963d6a1ab8caee568c8e8a28.jpg](../iclr_results/1526_DPaI_%20Differentiable%20Pruning%20at%20Initialization%20with%20Node-Path%20Balance%20Principle/tables/e36ae66f0622034836749968b0a7758f29ade3a9963d6a1ab8caee568c8e8a28.jpg)

## Improving Pretraining Data Using Perplexity Correlations

### Images

![1d268afa9b35c30ce7ebcd083a5326a5c5d60d4303bd32dcea789a40e2bf7d27.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/1d268afa9b35c30ce7ebcd083a5326a5c5d60d4303bd32dcea789a40e2bf7d27.jpg)

![2a158ceb7714ae25e442a0db96197b82407ea0f8ffcf010d4621004ba5efd7a4.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/2a158ceb7714ae25e442a0db96197b82407ea0f8ffcf010d4621004ba5efd7a4.jpg)

![4e79035c0cd65438617b144f524fedeaa5af2ca4aaa18f3760672bf33390af8e.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/4e79035c0cd65438617b144f524fedeaa5af2ca4aaa18f3760672bf33390af8e.jpg)

![5e19ec07c33e124fa46f3e1d5582b91548419fab8faa039a0daf7ec4610766d8.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/5e19ec07c33e124fa46f3e1d5582b91548419fab8faa039a0daf7ec4610766d8.jpg)

![81343e7c1a8c299b12eb50a4ee9fdf166599781ba4a97e4e6655bdafb51b8c6a.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/81343e7c1a8c299b12eb50a4ee9fdf166599781ba4a97e4e6655bdafb51b8c6a.jpg)

![9202e45f55a071b23ec3e5312769386743f66bd92629ab2828d26babba4a0976.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/9202e45f55a071b23ec3e5312769386743f66bd92629ab2828d26babba4a0976.jpg)

![99975b5616b3153762909fb3f2fb406afbde6f8c48056e9ce9fff1b1fb943b60.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/99975b5616b3153762909fb3f2fb406afbde6f8c48056e9ce9fff1b1fb943b60.jpg)

![dbe9ec000edd171ed5c2b56103bf54437c9bfae3a793646ec576c75467232c58.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/dbe9ec000edd171ed5c2b56103bf54437c9bfae3a793646ec576c75467232c58.jpg)

![e6676764b287514c89d0cbb7c980a05b7ef9819b6d41bfa356f15b485a44fe59.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/e6676764b287514c89d0cbb7c980a05b7ef9819b6d41bfa356f15b485a44fe59.jpg)

![f0185f277ddc004ba7037aebbd825154051d449c623fd98a4d6ec9a603d756b7.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/f0185f277ddc004ba7037aebbd825154051d449c623fd98a4d6ec9a603d756b7.jpg)

![f0fbebbeab339406e87c08671f60db9303b61eee9d1df92c9cf7ffe0e9d6a9b8.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/f0fbebbeab339406e87c08671f60db9303b61eee9d1df92c9cf7ffe0e9d6a9b8.jpg)

![f25e06f0cd4feda481b87bb8fa59c22568def92710a66ccc32abedca84d323ea.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/f25e06f0cd4feda481b87bb8fa59c22568def92710a66ccc32abedca84d323ea.jpg)

![f31f95cf85c3e523a2c58bb0a0e57cb76044583fd7e61c6f404fc9a5f1789cb7.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/f31f95cf85c3e523a2c58bb0a0e57cb76044583fd7e61c6f404fc9a5f1789cb7.jpg)

![f6dc741281fc84694da8bec513561a6182e994536ee4d01ddea4517e91d71243.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/images/f6dc741281fc84694da8bec513561a6182e994536ee4d01ddea4517e91d71243.jpg)

### Tables

![18ebef067a02af2971b3e20f3a22c5b46cf5911cf2ad5c1994c36bb523e4776b.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/tables/18ebef067a02af2971b3e20f3a22c5b46cf5911cf2ad5c1994c36bb523e4776b.jpg)

![2a7448554e37e096b1de4bbe4f70ccf9f5335f5c7226885211d5edc64d5a0a9b.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/tables/2a7448554e37e096b1de4bbe4f70ccf9f5335f5c7226885211d5edc64d5a0a9b.jpg)

![6d8e6d3badc33041768386d644672ebd05786731b25c569cea601e853be95143.jpg](../iclr_results/1527_Improving%20Pretraining%20Data%20Using%20Perplexity%20Correlations/tables/6d8e6d3badc33041768386d644672ebd05786731b25c569cea601e853be95143.jpg)
