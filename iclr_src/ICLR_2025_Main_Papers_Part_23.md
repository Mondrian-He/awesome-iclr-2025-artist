# ICLR 2025 Main Conference Papers

**Summary:** 36 papers with extracted content:
- 📊 Total images: 44031
- 📋 Total tables: 33468
- 📄 Total files: 77499

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 23 of 100

## 目录 (Table of Contents)

1. [Logic-Logit: A Logic-Based Approach to Choice Modeling](#Logic-Logit-A-Logic-Based-Approach-to-Choice-Modeling)
2. [The Labyrinth of Links: Navigating the Associative Maze of Multi-modal LLMs](#The-Labyrinth-of-Links-Navigating-the-Associative-Maze-of-Multi-modal-LLMs)
3. [Neuron Platonic Intrinsic Representation From Dynamics Using Contrastive Learning](#Neuron-Platonic-Intrinsic-Representation-From-Dynamics-Using-Contrastive-Learning)
4. [Beyond Canonicalization: How Tensorial Messages Improve Equivariant Message Passing](#Beyond-Canonicalization-How-Tensorial-Messages-Improve-Equivariant-Message-Passing)
5. [OccProphet: Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with an Observer-Forecaster-Refiner Framework](#OccProphet-Pushing-the-Efficiency-Frontier-of-Camera-Only-4D-Occupancy-Forecasting-with-an-Observer-Forecaster-Refiner-Framework)
6. [PINP: Physics-Informed Neural Predictor with latent estimation of fluid flows](#PINP-Physics-Informed-Neural-Predictor-with-latent-estimation-of-fluid-flows)
7. [Visual-O1: Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoning](#Visual-O1-Understanding-Ambiguous-Instructions-via-Multi-modal-Multi-turn-Chain-of-thoughts-Reasoning)
8. [MMKE-Bench: A Multimodal Editing Benchmark for Diverse Visual Knowledge](#MMKE-Bench-A-Multimodal-Editing-Benchmark-for-Diverse-Visual-Knowledge)
9. [Multimodal Quantitative Language for Generative Recommendation](#Multimodal-Quantitative-Language-for-Generative-Recommendation)
10. [Does SGD really happen in tiny subspaces?](#Does-SGD-really-happen-in-tiny-subspaces)
11. [Q-SFT: Q-Learning for Language Models via Supervised Fine-Tuning](#Q-SFT-Q-Learning-for-Language-Models-via-Supervised-Fine-Tuning)
12. [PostCast: Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Modeling](#PostCast-Generalizable-Postprocessing-for-Precipitation-Nowcasting-via-Unsupervised-Blurriness-Modeling)
13. [Decoupled Subgraph Federated Learning](#Decoupled-Subgraph-Federated-Learning)
14. [AniSDF: Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruction](#AniSDF-Fused-Granularity-Neural-Surfaces-with-Anisotropic-Encoding-for-High-Fidelity-3D-Reconstruction)
15. [Schur's Positive-Definite Network: Deep Learning in the SPD cone with structure](#Schurs-Positive-Definite-Network-Deep-Learning-in-the-SPD-cone-with-structure)
16. [Selective Attention Improves Transformer](#Selective-Attention-Improves-Transformer)
17. [VideoShield: Regulating Diffusion-based Video Generation Models via Watermarking](#VideoShield-Regulating-Diffusion-based-Video-Generation-Models-via-Watermarking)
18. [Reconciling Model Multiplicity for Downstream Decision Making](#Reconciling-Model-Multiplicity-for-Downstream-Decision-Making)
19. [Unbounded: A Generative Infinite Game of Character Life Simulation](#Unbounded-A-Generative-Infinite-Game-of-Character-Life-Simulation)
20. [Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting](#Flow-Matching-with-Gaussian-Process-Priors-for-Probabilistic-Time-Series-Forecasting)
21. [FlowDec: A flow-based full-band general audio codec with high perceptual quality](#FlowDec-A-flow-based-full-band-general-audio-codec-with-high-perceptual-quality)
22. [IDArb: Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations](#IDArb-Intrinsic-Decomposition-for-Arbitrary-Number-of-Input-Views-and-Illuminations)
23. [Towards Certification of Uncertainty Calibration under Adversarial Attacks](#Towards-Certification-of-Uncertainty-Calibration-under-Adversarial-Attacks)
24. [VAE-Var: Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology](#VAE-Var-Variational-Autoencoder-Enhanced-Variational-Methods-for-Data-Assimilation-in-Meteorology)
25. [TODO: Enhancing LLM Alignment with Ternary Preferences](#TODO-Enhancing-LLM-Alignment-with-Ternary-Preferences)
26. [Deep Kernel Posterior Learning under Infinite Variance Prior Weights](#Deep-Kernel-Posterior-Learning-under-Infinite-Variance-Prior-Weights)
27. [Latent-EnSF: A Latent Ensemble Score Filter for High-Dimensional Data Assimilation with Sparse Observation Data](#Latent-EnSF-A-Latent-Ensemble-Score-Filter-for-High-Dimensional-Data-Assimilation-with-Sparse-Observation-Data)
28. [Trajectory-Class-Aware Multi-Agent Reinforcement Learning](#Trajectory-Class-Aware-Multi-Agent-Reinforcement-Learning)
29. [EG4D: Explicit Generation of 4D Object without Score Distillation](#EG4D-Explicit-Generation-of-4D-Object-without-Score-Distillation)
30. [The impact of allocation strategies in subset learning on the expressive power of neural networks](#The-impact-of-allocation-strategies-in-subset-learning-on-the-expressive-power-of-neural-networks)
31. [Learning on One Mode: Addressing Multi-modality in Offline Reinforcement Learning](#Learning-on-One-Mode-Addressing-Multi-modality-in-Offline-Reinforcement-Learning)
32. [Weak-to-Strong Generalization Through the Data-Centric Lens](#Weak-to-Strong-Generalization-Through-the-Data-Centric-Lens)
33. [VideoWebArena:  Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks](#VideoWebArena-Evaluating-Long-Context-Multimodal-Agents-with-Video-Understanding-Web-Tasks)
34. [OmniKV: Dynamic Context Selection for Efficient Long-Context LLMs](#OmniKV-Dynamic-Context-Selection-for-Efficient-Long-Context-LLMs)
35. [Functional Homotopy: Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Attacks](#Functional-Homotopy-Smoothing-Discrete-Optimization-via-Continuous-Parameters-for-LLM-Jailbreak-Attacks)
36. [ScImage: How good are multimodal large language models at scientific text-to-image generation?](#ScImage-How-good-are-multimodal-large-language-models-at-scientific-text-to-image-generation)

---


## Logic-Logit: A Logic-Based Approach to Choice Modeling

### Images

![ffe6aaeb07ead305eafa42bb085cd452bdbffc9b5ee6f83d57dcb950c2176b7f.jpg](../iclr_results/838_Memory Efficient Transformer Adapter for Dense Predictions/images/ffe6aaeb07ead305eafa42bb085cd452bdbffc9b5ee6f83d57dcb950c2176b7f.jpg)

### Tables

![3c95ba8d62e4a3ceaf61efb536922b7f5d7da1acaae2411a7992f182dc61261d.jpg](../iclr_results/838_Memory Efficient Transformer Adapter for Dense Predictions/tables/3c95ba8d62e4a3ceaf61efb536922b7f5d7da1acaae2411a7992f182dc61261d.jpg)

![54036c51943443ac02d79dda6f740a694418a6dfb4d0d46a86a48a468eca9013.jpg](../iclr_results/838_Memory Efficient Transformer Adapter for Dense Predictions/tables/54036c51943443ac02d79dda6f740a694418a6dfb4d0d46a86a48a468eca9013.jpg)

![98b9eda3513aaddb0d3d266fafd16daa4b83794d874911af689b5a7db4c62c30.jpg](../iclr_results/838_Memory Efficient Transformer Adapter for Dense Predictions/tables/98b9eda3513aaddb0d3d266fafd16daa4b83794d874911af689b5a7db4c62c30.jpg)

![9f2ce6c1b1c965f28c81aa7cca433189df845f6c6a881b9ba551b7a6a66972e3.jpg](../iclr_results/838_Memory Efficient Transformer Adapter for Dense Predictions/tables/9f2ce6c1b1c965f28c81aa7cca433189df845f6c6a881b9ba551b7a6a66972e3.jpg)

![c9c5c0ad512757000bcc29721f990c640b31a2b6a5020d1e92678cd323ce4ed2.jpg](../iclr_results/838_Memory Efficient Transformer Adapter for Dense Predictions/tables/c9c5c0ad512757000bcc29721f990c640b31a2b6a5020d1e92678cd323ce4ed2.jpg)

## Logic-Logit: A Logic-Based Approach to Choice Modeling


### Images

![2458dcb53b3321ff1528f8c02b276ca67141009b28298d777e3fd376792dbf82.jpg](../iclr_results/839_Logic-Logit_ A Logic-Based Approach to Choice Modeling/images/2458dcb53b3321ff1528f8c02b276ca67141009b28298d777e3fd376792dbf82.jpg)

![2ace2638a1d7cfd405a5a712793e87ce5a359862fb0ceafa0c23785c8b46aea8.jpg](../iclr_results/839_Logic-Logit_ A Logic-Based Approach to Choice Modeling/images/2ace2638a1d7cfd405a5a712793e87ce5a359862fb0ceafa0c23785c8b46aea8.jpg)

![2bd87c453db0a9718344c573e380a24ba90aac981f91c69777dee95596f34377.jpg](../iclr_results/839_Logic-Logit_ A Logic-Based Approach to Choice Modeling/images/2bd87c453db0a9718344c573e380a24ba90aac981f91c69777dee95596f34377.jpg)

![45feb431fadb7ab437604e1b9968c57fbc3ed75bdf08f9550f0fcf299a6acf64.jpg](../iclr_results/839_Logic-Logit_ A Logic-Based Approach to Choice Modeling/images/45feb431fadb7ab437604e1b9968c57fbc3ed75bdf08f9550f0fcf299a6acf64.jpg)

![74dd4269ed51ae060d202432753b4f51a85cb53ad0629ab88d0455e8147cfb14.jpg](../iclr_results/839_Logic-Logit_ A Logic-Based Approach to Choice Modeling/images/74dd4269ed51ae060d202432753b4f51a85cb53ad0629ab88d0455e8147cfb14.jpg)

![78e04bbb98cdd8ef23fa429ee84a03682240ea9d3d3f563518c8f3e6c4c0c735.jpg](../iclr_results/839_Logic-Logit_ A Logic-Based Approach to Choice Modeling/images/78e04bbb98cdd8ef23fa429ee84a03682240ea9d3d3f563518c8f3e6c4c0c735.jpg)

![d5e9b7520f9d2ade3448cacbe5f7c3c234ec6c8b03b9d15a8f263c6d6272cf86.jpg](../iclr_results/839_Logic-Logit_ A Logic-Based Approach to Choice Modeling/images/d5e9b7520f9d2ade3448cacbe5f7c3c234ec6c8b03b9d15a8f263c6d6272cf86.jpg)

![ded3aad843a5dbe516d4d4d2721674861ef62aadb0eba5104a3130676639ffbf.jpg](../iclr_results/839_Logic-Logit_ A Logic-Based Approach to Choice Modeling/images/ded3aad843a5dbe516d4d4d2721674861ef62aadb0eba5104a3130676639ffbf.jpg)

### Tables

![087a425c315608141ec36d6fb22ded1e26fdc1f0d28be538561af37a4306ecfc.jpg](../iclr_results/839_Logic-Logit_ A Logic-Based Approach to Choice Modeling/tables/087a425c315608141ec36d6fb22ded1e26fdc1f0d28be538561af37a4306ecfc.jpg)

![15a98ff6591b096c8d42c8d2afde7f4fe9394eb81adc7329a49af7cd0d35cfbc.jpg](../iclr_results/839_Logic-Logit_ A Logic-Based Approach to Choice Modeling/tables/15a98ff6591b096c8d42c8d2afde7f4fe9394eb81adc7329a49af7cd0d35cfbc.jpg)

![479ef3062ca1b7840dd792c07eb456fd0330ed823e00227f887da5d4e67fb063.jpg](../iclr_results/839_Logic-Logit_ A Logic-Based Approach to Choice Modeling/tables/479ef3062ca1b7840dd792c07eb456fd0330ed823e00227f887da5d4e67fb063.jpg)

![4edd3b070283b7825c01af118bfdbb812705794036efad4e3fb793bfc932d9a5.jpg](../iclr_results/839_Logic-Logit_ A Logic-Based Approach to Choice Modeling/tables/4edd3b070283b7825c01af118bfdbb812705794036efad4e3fb793bfc932d9a5.jpg)

![557b79426041c96a269e75d526d06aedcaea818769cb08e996d26cfb05cd5598.jpg](../iclr_results/839_Logic-Logit_ A Logic-Based Approach to Choice Modeling/tables/557b79426041c96a269e75d526d06aedcaea818769cb08e996d26cfb05cd5598.jpg)

![5d092bf9ae139b3ff249cd76a06d654d8a86ea875177f4eb283ec5b1eb442d99.jpg](../iclr_results/839_Logic-Logit_ A Logic-Based Approach to Choice Modeling/tables/5d092bf9ae139b3ff249cd76a06d654d8a86ea875177f4eb283ec5b1eb442d99.jpg)

![72affc83a200d1b30ce97974f75823611008cd6d1754dc42f51866d53764c904.jpg](../iclr_results/839_Logic-Logit_ A Logic-Based Approach to Choice Modeling/tables/72affc83a200d1b30ce97974f75823611008cd6d1754dc42f51866d53764c904.jpg)

![86af4457b663e70ebf64595e0a205f4f029c4ab6b214a357232037ecc7a258fb.jpg](../iclr_results/839_Logic-Logit_ A Logic-Based Approach to Choice Modeling/tables/86af4457b663e70ebf64595e0a205f4f029c4ab6b214a357232037ecc7a258fb.jpg)

![b78c1a3ec0d7e991067be81a09d6e758140ced32c12df9103f6b86cc4700e4aa.jpg](../iclr_results/839_Logic-Logit_ A Logic-Based Approach to Choice Modeling/tables/b78c1a3ec0d7e991067be81a09d6e758140ced32c12df9103f6b86cc4700e4aa.jpg)

![cf26a3989664aab780352668076b090bb6e8456148c00d0297dd2ba9dc2d799d.jpg](../iclr_results/839_Logic-Logit_ A Logic-Based Approach to Choice Modeling/tables/cf26a3989664aab780352668076b090bb6e8456148c00d0297dd2ba9dc2d799d.jpg)

![e33d60bc1ef394f2888ee51352eab1ba161cdf075b42748d8a930a54cb8058e8.jpg](../iclr_results/839_Logic-Logit_ A Logic-Based Approach to Choice Modeling/tables/e33d60bc1ef394f2888ee51352eab1ba161cdf075b42748d8a930a54cb8058e8.jpg)

## The Labyrinth of Links: Navigating the Associative Maze of Multi-modal LLMs


### Images

![28249d17c424e20dcc3d3008901a549e2be080f01e466311cc19702d1fbcf09b.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/images/28249d17c424e20dcc3d3008901a549e2be080f01e466311cc19702d1fbcf09b.jpg)

![4758405132c2112e9d8480bca84dcf7cfda22a05a57d855b8bcd8c19eb559694.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/images/4758405132c2112e9d8480bca84dcf7cfda22a05a57d855b8bcd8c19eb559694.jpg)

![5673621b8aa94bf200a405e1f7fe6317c888ec0f5d14b6b30c49e74287029df0.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/images/5673621b8aa94bf200a405e1f7fe6317c888ec0f5d14b6b30c49e74287029df0.jpg)

![5e22ba7f712aa69a4a67cdc9ceff21c5c40c221aa17a70a9717a9eb3bfec30ed.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/images/5e22ba7f712aa69a4a67cdc9ceff21c5c40c221aa17a70a9717a9eb3bfec30ed.jpg)

![76675446ca985b3b4ccbaf55ad855137a909e5e54aaa9e5435cfb805c8375d86.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/images/76675446ca985b3b4ccbaf55ad855137a909e5e54aaa9e5435cfb805c8375d86.jpg)

![7c3df337352f409b36bad037f119610e5ee8dc8bb5c311ab2eb3d1e4169537d8.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/images/7c3df337352f409b36bad037f119610e5ee8dc8bb5c311ab2eb3d1e4169537d8.jpg)

![a6a929f2ad93271564c95218827cc479aa67e5c3ae32876ec926356c3ba9bbce.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/images/a6a929f2ad93271564c95218827cc479aa67e5c3ae32876ec926356c3ba9bbce.jpg)

![ad7eaf211f1b173c3705ed2f0ff5a4c94d572a33f0db9fdab2b632751ae9829a.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/images/ad7eaf211f1b173c3705ed2f0ff5a4c94d572a33f0db9fdab2b632751ae9829a.jpg)

![ae1c9e8382818198c980c73d6b79468a62e5a614100db94406a904c6a797737c.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/images/ae1c9e8382818198c980c73d6b79468a62e5a614100db94406a904c6a797737c.jpg)

![cf4b4014f79153bec79e10da1c0de8c53c243d83e763d0a1e8f8b495904941a0.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/images/cf4b4014f79153bec79e10da1c0de8c53c243d83e763d0a1e8f8b495904941a0.jpg)

### Tables

![0c1fadba52fa667f49f1865d2cec92570803aed80c333b56c4b3bff395ea89f7.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/tables/0c1fadba52fa667f49f1865d2cec92570803aed80c333b56c4b3bff395ea89f7.jpg)

![0ccf676216196941b80f75c15db9ff796d82c9f133dbe4cbcdcc3326626df6fe.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/tables/0ccf676216196941b80f75c15db9ff796d82c9f133dbe4cbcdcc3326626df6fe.jpg)

![10b8dd9a5d70cbd6edd00905c270462f03d22a7c6dcfaa047d1bc5cba671715f.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/tables/10b8dd9a5d70cbd6edd00905c270462f03d22a7c6dcfaa047d1bc5cba671715f.jpg)

![1212f523caad3cce9b85788ca6ad7161a83d8c2c4a9fcb25d8d047c82fd08a37.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/tables/1212f523caad3cce9b85788ca6ad7161a83d8c2c4a9fcb25d8d047c82fd08a37.jpg)

![2ac9c32f1df096306398ad6cf3efa5d8670eb83801e260b056e8156673ca2291.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/tables/2ac9c32f1df096306398ad6cf3efa5d8670eb83801e260b056e8156673ca2291.jpg)

![2ebd30812e25eabaf75d330ada30408f31e0e62dee2d1385b9809111b70edfec.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/tables/2ebd30812e25eabaf75d330ada30408f31e0e62dee2d1385b9809111b70edfec.jpg)

![42be18ef9cfa75ca20b54f0b9fc14455124824239a04d18378a6d6d14c020a41.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/tables/42be18ef9cfa75ca20b54f0b9fc14455124824239a04d18378a6d6d14c020a41.jpg)

![468d11e622402e150392d730a00b041afed798d64d6af09b8f241f2e935f4512.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/tables/468d11e622402e150392d730a00b041afed798d64d6af09b8f241f2e935f4512.jpg)

![697ba3e0a826748715f03af585295db06d061c932a60cbab7f0a45856f935302.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/tables/697ba3e0a826748715f03af585295db06d061c932a60cbab7f0a45856f935302.jpg)

![72d8ce12b31261d829dda6916df08152134ce40c173c8b7d8738284739b78b74.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/tables/72d8ce12b31261d829dda6916df08152134ce40c173c8b7d8738284739b78b74.jpg)

![75d092c12f368b309700be0f52f01f51d26f3545ca51d90179e3456f50a83ef1.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/tables/75d092c12f368b309700be0f52f01f51d26f3545ca51d90179e3456f50a83ef1.jpg)

![8184a52f1c3eb60790c02bd5cab510a9782f8891711132ef6fcd86df65b19896.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/tables/8184a52f1c3eb60790c02bd5cab510a9782f8891711132ef6fcd86df65b19896.jpg)

![915004a0dc3aa7eec3b8493634d24f26036eecfe5afdc2ecb227d2b367ca1275.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/tables/915004a0dc3aa7eec3b8493634d24f26036eecfe5afdc2ecb227d2b367ca1275.jpg)

![a38a278517bb23de2164553334261718c53426afbfcf9e55b96141868f8a0191.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/tables/a38a278517bb23de2164553334261718c53426afbfcf9e55b96141868f8a0191.jpg)

![b1fbc4665987e324d7f2b26006a6f664d21f2a69095f3b55036eb951c56322aa.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/tables/b1fbc4665987e324d7f2b26006a6f664d21f2a69095f3b55036eb951c56322aa.jpg)

![b6cb961bac1b963539421bf98b94170190190fe861f902f345f7369cdc761f88.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/tables/b6cb961bac1b963539421bf98b94170190190fe861f902f345f7369cdc761f88.jpg)

![e27b2ba81d0abb65f200ddb296bc8e3b5bbd0727fcacd82b4f1c19c5f88e3e13.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/tables/e27b2ba81d0abb65f200ddb296bc8e3b5bbd0727fcacd82b4f1c19c5f88e3e13.jpg)

![eb7b3fb962397eef612ad102aa4803283a43e4f31e56fc3aa656535ef695a151.jpg](../iclr_results/840_The Labyrinth of Links_ Navigating the Associative Maze of Multi-modal LLMs/tables/eb7b3fb962397eef612ad102aa4803283a43e4f31e56fc3aa656535ef695a151.jpg)

## Neuron Platonic Intrinsic Representation From Dynamics Using Contrastive Learning


### Images

![081d3b71694cceec18c8dda049dde2b3d16f90cf5d9edff821a85cc23c3088ec.jpg](../iclr_results/841_Neuron Platonic Intrinsic Representation From Dynamics Using Contrastive Learning/images/081d3b71694cceec18c8dda049dde2b3d16f90cf5d9edff821a85cc23c3088ec.jpg)

![19cd6e9ac1d8044300c8e5a758f96c86fa0c4df3419fd6c97e05b67388143332.jpg](../iclr_results/841_Neuron Platonic Intrinsic Representation From Dynamics Using Contrastive Learning/images/19cd6e9ac1d8044300c8e5a758f96c86fa0c4df3419fd6c97e05b67388143332.jpg)

![ee355e9d51e07dcaacb70c35cee3eb85dc5f74f77126b8d78c9f80cd485240d9.jpg](../iclr_results/841_Neuron Platonic Intrinsic Representation From Dynamics Using Contrastive Learning/images/ee355e9d51e07dcaacb70c35cee3eb85dc5f74f77126b8d78c9f80cd485240d9.jpg)

### Tables

![20b7f6fd62553eea9704bac68fa5ffcaf5d0d01123192f3d2322372cf3c3b7e6.jpg](../iclr_results/841_Neuron Platonic Intrinsic Representation From Dynamics Using Contrastive Learning/tables/20b7f6fd62553eea9704bac68fa5ffcaf5d0d01123192f3d2322372cf3c3b7e6.jpg)

![72cc1c82e7deef375e5fe27df073a735dc13aeac6c661f0a54fa16ec72b675c2.jpg](../iclr_results/841_Neuron Platonic Intrinsic Representation From Dynamics Using Contrastive Learning/tables/72cc1c82e7deef375e5fe27df073a735dc13aeac6c661f0a54fa16ec72b675c2.jpg)

![a8bca42876c890cde544f64afc977a1332ad235436f9ebaae286e24ef582e8a4.jpg](../iclr_results/841_Neuron Platonic Intrinsic Representation From Dynamics Using Contrastive Learning/tables/a8bca42876c890cde544f64afc977a1332ad235436f9ebaae286e24ef582e8a4.jpg)

![ae37f3ec89af25680a98320ca7ce3dada72cf33922ff4887a493fb48d0c81066.jpg](../iclr_results/841_Neuron Platonic Intrinsic Representation From Dynamics Using Contrastive Learning/tables/ae37f3ec89af25680a98320ca7ce3dada72cf33922ff4887a493fb48d0c81066.jpg)

![e099bb4666229839ebd8059278d8730659e0b6595ab0a9a8da6bc30550df88dc.jpg](../iclr_results/841_Neuron Platonic Intrinsic Representation From Dynamics Using Contrastive Learning/tables/e099bb4666229839ebd8059278d8730659e0b6595ab0a9a8da6bc30550df88dc.jpg)

![f5ab8886be15d30e90e4ab613976260b457f91918bdb434088aa4e152ee209ce.jpg](../iclr_results/841_Neuron Platonic Intrinsic Representation From Dynamics Using Contrastive Learning/tables/f5ab8886be15d30e90e4ab613976260b457f91918bdb434088aa4e152ee209ce.jpg)

## Beyond Canonicalization: How Tensorial Messages Improve Equivariant Message Passing


### Images

![8a5f08c46b19a87208f923eece43e0e07ed6194dbe8aaf11dd733fd68ea0e166.jpg](../iclr_results/842_Beyond Canonicalization_ How Tensorial Messages Improve Equivariant Message Passing/images/8a5f08c46b19a87208f923eece43e0e07ed6194dbe8aaf11dd733fd68ea0e166.jpg)

![9098a14c3645c5dc56bd1d243d3b86b61c481ccc262edc9d18c8062945836759.jpg](../iclr_results/842_Beyond Canonicalization_ How Tensorial Messages Improve Equivariant Message Passing/images/9098a14c3645c5dc56bd1d243d3b86b61c481ccc262edc9d18c8062945836759.jpg)

![9607f73af152061f77b2291705c9e0826dd54ba3ccd20db86667661fbfcf9999.jpg](../iclr_results/842_Beyond Canonicalization_ How Tensorial Messages Improve Equivariant Message Passing/images/9607f73af152061f77b2291705c9e0826dd54ba3ccd20db86667661fbfcf9999.jpg)

![a5167f60ff78ad2ec0f56760ec230099417ce43c104a3ca61315edfd291ea3c9.jpg](../iclr_results/842_Beyond Canonicalization_ How Tensorial Messages Improve Equivariant Message Passing/images/a5167f60ff78ad2ec0f56760ec230099417ce43c104a3ca61315edfd291ea3c9.jpg)

![cd0bedccf1211bcebaf8d9e7a4bde9d07ff822482082100a6e44502fc1d34fa2.jpg](../iclr_results/842_Beyond Canonicalization_ How Tensorial Messages Improve Equivariant Message Passing/images/cd0bedccf1211bcebaf8d9e7a4bde9d07ff822482082100a6e44502fc1d34fa2.jpg)

### Tables

![035b75b4e692a0d13edbd022521ee4652744c4901ad1c4dd13ececbe8ee47b5f.jpg](../iclr_results/842_Beyond Canonicalization_ How Tensorial Messages Improve Equivariant Message Passing/tables/035b75b4e692a0d13edbd022521ee4652744c4901ad1c4dd13ececbe8ee47b5f.jpg)

![2fc0ddb6fc066aebc9aa29f46abd4e6ea1144c23f3e980db2d218330276eb79f.jpg](../iclr_results/842_Beyond Canonicalization_ How Tensorial Messages Improve Equivariant Message Passing/tables/2fc0ddb6fc066aebc9aa29f46abd4e6ea1144c23f3e980db2d218330276eb79f.jpg)

![3903d50a1e27287a3c5842b57f90f8d6602389cf23981bd787b73a715360bcf8.jpg](../iclr_results/842_Beyond Canonicalization_ How Tensorial Messages Improve Equivariant Message Passing/tables/3903d50a1e27287a3c5842b57f90f8d6602389cf23981bd787b73a715360bcf8.jpg)

![40289f52a3b53de9589e629765a0fa4fc20937e1e723a1cc56e300ffe6a430c6.jpg](../iclr_results/842_Beyond Canonicalization_ How Tensorial Messages Improve Equivariant Message Passing/tables/40289f52a3b53de9589e629765a0fa4fc20937e1e723a1cc56e300ffe6a430c6.jpg)

![567c8a2625c2352597f8d325fbaf0510ecfa5f1051551e54a818c9e1f97489d0.jpg](../iclr_results/842_Beyond Canonicalization_ How Tensorial Messages Improve Equivariant Message Passing/tables/567c8a2625c2352597f8d325fbaf0510ecfa5f1051551e54a818c9e1f97489d0.jpg)

![5dfc267e809e54d7464a1be0817c2023f45fa448a13ef9f9d0e06e54ca7e9507.jpg](../iclr_results/842_Beyond Canonicalization_ How Tensorial Messages Improve Equivariant Message Passing/tables/5dfc267e809e54d7464a1be0817c2023f45fa448a13ef9f9d0e06e54ca7e9507.jpg)

![72cb0b8e6b614ad480d5ac4f0816f7ac120a778a93ad0ffec5f951827535b3c6.jpg](../iclr_results/842_Beyond Canonicalization_ How Tensorial Messages Improve Equivariant Message Passing/tables/72cb0b8e6b614ad480d5ac4f0816f7ac120a778a93ad0ffec5f951827535b3c6.jpg)

![8a49dbbccf3066d48d11184b6ed8db6e4174e247e2403283b8b29aca8f139c41.jpg](../iclr_results/842_Beyond Canonicalization_ How Tensorial Messages Improve Equivariant Message Passing/tables/8a49dbbccf3066d48d11184b6ed8db6e4174e247e2403283b8b29aca8f139c41.jpg)

![d51241456a7be3b96189235a5d91a9cb24ce7c87e1222aebb47b6354dc1fb274.jpg](../iclr_results/842_Beyond Canonicalization_ How Tensorial Messages Improve Equivariant Message Passing/tables/d51241456a7be3b96189235a5d91a9cb24ce7c87e1222aebb47b6354dc1fb274.jpg)

![fa4e3aac9787f2e80f7b110a34214b6c099e90588e9d49a42a82b85943483b77.jpg](../iclr_results/842_Beyond Canonicalization_ How Tensorial Messages Improve Equivariant Message Passing/tables/fa4e3aac9787f2e80f7b110a34214b6c099e90588e9d49a42a82b85943483b77.jpg)

## OccProphet: Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with an Observer-Forecaster-Refiner Framework


### Images

![10e80c83028b6d81cff3639033cd21e688f1725b3c2842685c217d538a814df3.jpg](../iclr_results/843_OccProphet_ Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with an Observer/images/10e80c83028b6d81cff3639033cd21e688f1725b3c2842685c217d538a814df3.jpg)

![291ac98c325664d989fed3a49936963dcbf875fb520861c5c07f3f6a4b5a6df5.jpg](../iclr_results/843_OccProphet_ Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with an Observer/images/291ac98c325664d989fed3a49936963dcbf875fb520861c5c07f3f6a4b5a6df5.jpg)

![4749742c44f720afc015e98636bdb18a2213ea50cb82276d285e854036621c41.jpg](../iclr_results/843_OccProphet_ Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with an Observer/images/4749742c44f720afc015e98636bdb18a2213ea50cb82276d285e854036621c41.jpg)

![5ceece3e375ae9b33aac96937aafd8cc84f60a6865511fc8ecbd7399a3dc855a.jpg](../iclr_results/843_OccProphet_ Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with an Observer/images/5ceece3e375ae9b33aac96937aafd8cc84f60a6865511fc8ecbd7399a3dc855a.jpg)

![673916d226d9ad352c4b03e8994d15fae29b05ba4fbf0a97169c0d1014833475.jpg](../iclr_results/843_OccProphet_ Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with an Observer/images/673916d226d9ad352c4b03e8994d15fae29b05ba4fbf0a97169c0d1014833475.jpg)

![6d488739332c5c1cac8d83ffae8128bcab7acda9abc2a084525d2bb7e8926ed3.jpg](../iclr_results/843_OccProphet_ Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with an Observer/images/6d488739332c5c1cac8d83ffae8128bcab7acda9abc2a084525d2bb7e8926ed3.jpg)

![964b3453dddf6f6524154c815754392f8940c97475d05d5644363bbb94eb6d9e.jpg](../iclr_results/843_OccProphet_ Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with an Observer/images/964b3453dddf6f6524154c815754392f8940c97475d05d5644363bbb94eb6d9e.jpg)

![9d9d1f4b025e7c0071f20b9af38f02fbd813a1f54909ab08350ed05576d763b4.jpg](../iclr_results/843_OccProphet_ Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with an Observer/images/9d9d1f4b025e7c0071f20b9af38f02fbd813a1f54909ab08350ed05576d763b4.jpg)

![b7b2c23150fdf27a3f5f53f20257c220725c988697531324d6d30609f9ab36b4.jpg](../iclr_results/843_OccProphet_ Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with an Observer/images/b7b2c23150fdf27a3f5f53f20257c220725c988697531324d6d30609f9ab36b4.jpg)

![bd24e3e6ef54d2e9eedfa434684cadd7edc09349137cace9aa1139f83be8bed1.jpg](../iclr_results/843_OccProphet_ Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with an Observer/images/bd24e3e6ef54d2e9eedfa434684cadd7edc09349137cace9aa1139f83be8bed1.jpg)

![be68d0ec8fe217e6ab9fa66850687d6545f617033899df4ba040d638e290611c.jpg](../iclr_results/843_OccProphet_ Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with an Observer/images/be68d0ec8fe217e6ab9fa66850687d6545f617033899df4ba040d638e290611c.jpg)

![e88624068f0653ef7b29f2fef86c89962f1b0844fa2b4647cf6414a9d7c7d650.jpg](../iclr_results/843_OccProphet_ Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with an Observer/images/e88624068f0653ef7b29f2fef86c89962f1b0844fa2b4647cf6414a9d7c7d650.jpg)

![eafe1bf42f419ecae62efa1fc5db726ec0d443e1c977fba5d9b042d3884cbcf9.jpg](../iclr_results/843_OccProphet_ Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with an Observer/images/eafe1bf42f419ecae62efa1fc5db726ec0d443e1c977fba5d9b042d3884cbcf9.jpg)

### Tables

![26e3110fab9766761059dead43e1dff563f31ac08d55e7bdbb084bd17f4f1464.jpg](../iclr_results/843_OccProphet_ Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with an Observer/tables/26e3110fab9766761059dead43e1dff563f31ac08d55e7bdbb084bd17f4f1464.jpg)

![612d9189be9e8cece7b91e1292b31ca9400b4eb7eb57475969e981beb6356633.jpg](../iclr_results/843_OccProphet_ Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with an Observer/tables/612d9189be9e8cece7b91e1292b31ca9400b4eb7eb57475969e981beb6356633.jpg)

![93c20b2a0b52e5bb76e30456d0e079f9fd9fdd901a012773b9ca4ad1e9c097f6.jpg](../iclr_results/843_OccProphet_ Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with an Observer/tables/93c20b2a0b52e5bb76e30456d0e079f9fd9fdd901a012773b9ca4ad1e9c097f6.jpg)

![a6e5883a76fc46083a5a0f8b3cef804dbcd4ed1a795919b9a074989da2b51ed3.jpg](../iclr_results/843_OccProphet_ Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with an Observer/tables/a6e5883a76fc46083a5a0f8b3cef804dbcd4ed1a795919b9a074989da2b51ed3.jpg)

![ae1e72cbabb0841083c810aff03bc2dab94a9aad3c9766ed028a5c64e1b52025.jpg](../iclr_results/843_OccProphet_ Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with an Observer/tables/ae1e72cbabb0841083c810aff03bc2dab94a9aad3c9766ed028a5c64e1b52025.jpg)

![edad216bad3eb55c5d96a3f1b3720acd3ca4b62fbd7f7c427db3c2e2230eab7d.jpg](../iclr_results/843_OccProphet_ Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with an Observer/tables/edad216bad3eb55c5d96a3f1b3720acd3ca4b62fbd7f7c427db3c2e2230eab7d.jpg)

![f3ecd05ac4f079446644150e67548e3b9dd3a123175334611b742e7330dc8c5c.jpg](../iclr_results/843_OccProphet_ Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with an Observer/tables/f3ecd05ac4f079446644150e67548e3b9dd3a123175334611b742e7330dc8c5c.jpg)

![f74583a69c9c3f0f6f92fe8de4fba68081316de536866bee138b0e1dd2baaac2.jpg](../iclr_results/843_OccProphet_ Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting with an Observer/tables/f74583a69c9c3f0f6f92fe8de4fba68081316de536866bee138b0e1dd2baaac2.jpg)

## PINP: Physics-Informed Neural Predictor with latent estimation of fluid flows


### Images

![24997c71eb4d5e3e7871408fcca6decf528dad4898279b3d96002e959770c85e.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/24997c71eb4d5e3e7871408fcca6decf528dad4898279b3d96002e959770c85e.jpg)

![27eb9336ae80c4aa3c74cd771619816bc467a660f461177c1e1629b605349cc7.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/27eb9336ae80c4aa3c74cd771619816bc467a660f461177c1e1629b605349cc7.jpg)

![2d2b3263624e93a3532abd387e61c803ade54c3f9ae849058efa32347c29071c.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/2d2b3263624e93a3532abd387e61c803ade54c3f9ae849058efa32347c29071c.jpg)

![357b478cb422fa54e54b7974e2fda2ec7c3788f808ea4bbb3bb2d789f7edac19.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/357b478cb422fa54e54b7974e2fda2ec7c3788f808ea4bbb3bb2d789f7edac19.jpg)

![3b0f2a9df6e5b6670a35d316ab1748a9f60fdcebf42815c57e1c7fcd97ae144b.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/3b0f2a9df6e5b6670a35d316ab1748a9f60fdcebf42815c57e1c7fcd97ae144b.jpg)

![429d84c6508fbb043c194935c87686aa7e7331dcc719098d270ad7e9b13f4db3.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/429d84c6508fbb043c194935c87686aa7e7331dcc719098d270ad7e9b13f4db3.jpg)

![48ac537b80f61e080109e8dd33f01c2d768cc384634eff57c836312b1c0d7c6c.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/48ac537b80f61e080109e8dd33f01c2d768cc384634eff57c836312b1c0d7c6c.jpg)

![528f5189e1930e5fcb12de903575bb05e9362520f3cfd9aca175285dadfbf0c7.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/528f5189e1930e5fcb12de903575bb05e9362520f3cfd9aca175285dadfbf0c7.jpg)

![6ba175c8b3768211276ea92b910c3bd756991e3c86f7c19709489f2b6f4c320a.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/6ba175c8b3768211276ea92b910c3bd756991e3c86f7c19709489f2b6f4c320a.jpg)

![7adaa5b8240b0d691aee3c2ebc32a9d45f90fd69c0a1bb0a814aecd5ef8370d4.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/7adaa5b8240b0d691aee3c2ebc32a9d45f90fd69c0a1bb0a814aecd5ef8370d4.jpg)

![80c4fd260444f441f1c45a2236c130f9617f240228aac22d4ea5720330c50331.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/80c4fd260444f441f1c45a2236c130f9617f240228aac22d4ea5720330c50331.jpg)

![837b918adce25bf0d99f70f9a659b6653dab0f9862cca81ea566d4f17825db78.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/837b918adce25bf0d99f70f9a659b6653dab0f9862cca81ea566d4f17825db78.jpg)

![8ae6173b181db30876aeee45f235d0e59f3981ed9ce1c524135841a95f3776ce.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/8ae6173b181db30876aeee45f235d0e59f3981ed9ce1c524135841a95f3776ce.jpg)

![b23b4b4c6d506ec232870b06e118eaa247ccd617cbf13fd5369cabe7bdf5609a.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/b23b4b4c6d506ec232870b06e118eaa247ccd617cbf13fd5369cabe7bdf5609a.jpg)

![b7846360e2cc6e1e8ca4cbed8afe887ac4b0158873c32ee46ffa213fde6ff1f8.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/b7846360e2cc6e1e8ca4cbed8afe887ac4b0158873c32ee46ffa213fde6ff1f8.jpg)

![bb996f0b8310d96529c65afd8aeaefa75582e88bd059d5fb57067486cfba66c6.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/bb996f0b8310d96529c65afd8aeaefa75582e88bd059d5fb57067486cfba66c6.jpg)

![c3977d869f8ca91ba459e38a48a886993beb43aaed4e53fab936cea8048e22b5.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/c3977d869f8ca91ba459e38a48a886993beb43aaed4e53fab936cea8048e22b5.jpg)

![ca3191b44c2b71551949c74e66d252e17c24d75d0598452a41e7bb75b84a75af.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/ca3191b44c2b71551949c74e66d252e17c24d75d0598452a41e7bb75b84a75af.jpg)

![cbf5d47ad5371ee5146fd27afd97dba9f6365beba8aa1c852fa2002d29904f73.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/cbf5d47ad5371ee5146fd27afd97dba9f6365beba8aa1c852fa2002d29904f73.jpg)

![ccf02a0f1c8dc5c1f90b2dfb2aa8caa1cc8850a85b852b11919540fc1c7cd9e0.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/ccf02a0f1c8dc5c1f90b2dfb2aa8caa1cc8850a85b852b11919540fc1c7cd9e0.jpg)

![ce72d695821f10a070362a3b635a6f373b91cf56876ecd1903214bd8b03d6c9c.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/ce72d695821f10a070362a3b635a6f373b91cf56876ecd1903214bd8b03d6c9c.jpg)

![d6a00779859b6b7494d740c8821ae101ed12c530c6d9ad67ff3cbd9927912cac.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/d6a00779859b6b7494d740c8821ae101ed12c530c6d9ad67ff3cbd9927912cac.jpg)

![da69998a82fd6c04c7f3e517409bf9b4c4c77e8903f6734260ab5d9c509be07f.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/da69998a82fd6c04c7f3e517409bf9b4c4c77e8903f6734260ab5d9c509be07f.jpg)

![de4f904f13dd739b758fe668754732e1d36f1255a888a5d433eba3efa3f80289.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/de4f904f13dd739b758fe668754732e1d36f1255a888a5d433eba3efa3f80289.jpg)

![e312f635f7736c119a8375b2a75de168f5195d92da0fbb13f752ab2d320a44af.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/e312f635f7736c119a8375b2a75de168f5195d92da0fbb13f752ab2d320a44af.jpg)

![e481f98f2a5201216f9138c3adeb12421e05bbd88565105a4ee8c5d5b882a432.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/e481f98f2a5201216f9138c3adeb12421e05bbd88565105a4ee8c5d5b882a432.jpg)

![ed3578ca3472c0a114b0b0e6b53aa202c5db22602950d1ca3b3f437ae7d70d94.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/ed3578ca3472c0a114b0b0e6b53aa202c5db22602950d1ca3b3f437ae7d70d94.jpg)

![fad974c9d3a5b73f69816d06f2535918d03a080407895bc7fda7b57708f68317.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/images/fad974c9d3a5b73f69816d06f2535918d03a080407895bc7fda7b57708f68317.jpg)

### Tables

![127ed06565f9817ad61a2ca61d042ca3751365ad3c16026ef2ead12cf09c7eb5.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/tables/127ed06565f9817ad61a2ca61d042ca3751365ad3c16026ef2ead12cf09c7eb5.jpg)

![26504303737c22f87c1eda25c178a422a5e676f0ec217660a2b5e9e66306dd4b.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/tables/26504303737c22f87c1eda25c178a422a5e676f0ec217660a2b5e9e66306dd4b.jpg)

![2784fff68dda572a2a6dcfc3df65bd1fe0e39a09ea9be04f68ce5561526b7fa7.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/tables/2784fff68dda572a2a6dcfc3df65bd1fe0e39a09ea9be04f68ce5561526b7fa7.jpg)

![33c111fa4f906d5d0e0d3d425e6eda9ec47eae06f83157d38ab429df96e56ef6.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/tables/33c111fa4f906d5d0e0d3d425e6eda9ec47eae06f83157d38ab429df96e56ef6.jpg)

![45b80d6f8103d980afc78768ee72e05a30a288f8de11c2b2ad51fcd4cf531da3.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/tables/45b80d6f8103d980afc78768ee72e05a30a288f8de11c2b2ad51fcd4cf531da3.jpg)

![504396813355fd44703411a5721863acd53f7c387de61af35073c5e34b9e10c5.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/tables/504396813355fd44703411a5721863acd53f7c387de61af35073c5e34b9e10c5.jpg)

![5522b451e388e93dd100a9be46b2e64762d00d3c4e42e0b184e2b45d4f420255.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/tables/5522b451e388e93dd100a9be46b2e64762d00d3c4e42e0b184e2b45d4f420255.jpg)

![62936dd92260563bfe02ca3a7a24ab33683e23f9216f8158db398edd01f032ca.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/tables/62936dd92260563bfe02ca3a7a24ab33683e23f9216f8158db398edd01f032ca.jpg)

![8623035289c29ce5f64c4fd4debe2285d0490fa37c62cec585af03ef1be21852.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/tables/8623035289c29ce5f64c4fd4debe2285d0490fa37c62cec585af03ef1be21852.jpg)

![d959a685b9971859652162ef7a5350f4ec0258e11c8feee4b263a1d48ed9c732.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/tables/d959a685b9971859652162ef7a5350f4ec0258e11c8feee4b263a1d48ed9c732.jpg)

![e792c6f5d2be95060e34cdaa4094ae5d82ef9d7b34655146d8f3086814453dbc.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/tables/e792c6f5d2be95060e34cdaa4094ae5d82ef9d7b34655146d8f3086814453dbc.jpg)

![f04e49442a96cf9e217403098de192f2c603608b3c8a4698a5aa20d51d9b46c8.jpg](../iclr_results/844_PINP_ Physics-Informed Neural Predictor with latent estimation of fluid flows/tables/f04e49442a96cf9e217403098de192f2c603608b3c8a4698a5aa20d51d9b46c8.jpg)

## Visual-O1: Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoning


### Images

![498415ed925838fbc7e56152173ae9ef1c4f99fb973d842fd15a68d7d3984037.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/images/498415ed925838fbc7e56152173ae9ef1c4f99fb973d842fd15a68d7d3984037.jpg)

![51d5936ce6d2bb03b64c493fd3825ade1e5d1b981c91f4c8ccc2dea986623110.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/images/51d5936ce6d2bb03b64c493fd3825ade1e5d1b981c91f4c8ccc2dea986623110.jpg)

![78d70b6d216e16fe3623d7d657558693067d0cf055b36db42fa5f398288068f5.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/images/78d70b6d216e16fe3623d7d657558693067d0cf055b36db42fa5f398288068f5.jpg)

![7bf5950641382166002876af0456e6398230d69fe1856b10f68affbb93891a2b.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/images/7bf5950641382166002876af0456e6398230d69fe1856b10f68affbb93891a2b.jpg)

![86230588dc48b9188a129abce69469b57b5dd864386da6c46ea3e48e61093d42.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/images/86230588dc48b9188a129abce69469b57b5dd864386da6c46ea3e48e61093d42.jpg)

![867848adc5138c9c1dc8fbc30b72ee67ef5e6d483dbf13e22a4b1e5a102bc950.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/images/867848adc5138c9c1dc8fbc30b72ee67ef5e6d483dbf13e22a4b1e5a102bc950.jpg)

![8f9c10dda9469f6aaa2d8889daee691fc2a17055b46f82bfd0c2950aacb58017.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/images/8f9c10dda9469f6aaa2d8889daee691fc2a17055b46f82bfd0c2950aacb58017.jpg)

![a5c39d0a43cce9c8215ee766922fe509405f81866dc84e4cb68105bcef3ef1b5.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/images/a5c39d0a43cce9c8215ee766922fe509405f81866dc84e4cb68105bcef3ef1b5.jpg)

![d491f273ca281c9b0eafc40497542f86e71b7e8aab56c0d181e88efe59b2851a.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/images/d491f273ca281c9b0eafc40497542f86e71b7e8aab56c0d181e88efe59b2851a.jpg)

![f04ee7f8dd02591555b8cc48af9a2264777b01edec1f01f4365f91be30db8674.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/images/f04ee7f8dd02591555b8cc48af9a2264777b01edec1f01f4365f91be30db8674.jpg)

![f5f94f74c7cde6559db00aa5c8a6a1b8a4951ba25bf93f365a59140207091512.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/images/f5f94f74c7cde6559db00aa5c8a6a1b8a4951ba25bf93f365a59140207091512.jpg)

### Tables

![00b4efdecd6741bae7afb96a39ec145e94291d055f6a749f01a3e8b8d7cbec59.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/tables/00b4efdecd6741bae7afb96a39ec145e94291d055f6a749f01a3e8b8d7cbec59.jpg)

![01b1aa212ddefdf805b41a112d9fbed59800c534dcd4a51ea6df7583b1968dd4.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/tables/01b1aa212ddefdf805b41a112d9fbed59800c534dcd4a51ea6df7583b1968dd4.jpg)

![0eb33d6dd31178067d2ca50b25d66a097557d338d7cab3f939786524df0f8d78.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/tables/0eb33d6dd31178067d2ca50b25d66a097557d338d7cab3f939786524df0f8d78.jpg)

![272fd7a881614ebd7303931b30610e4105728054710893abe8e795b8fee973e9.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/tables/272fd7a881614ebd7303931b30610e4105728054710893abe8e795b8fee973e9.jpg)

![4102d6b66940884576fb14913aabc961a9969a4f3d3aaf5292520f03fcd2540a.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/tables/4102d6b66940884576fb14913aabc961a9969a4f3d3aaf5292520f03fcd2540a.jpg)

![446017e2a2822808c1fe512ce9b2084f2898d98b230ee6ebadd4f509bb282aae.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/tables/446017e2a2822808c1fe512ce9b2084f2898d98b230ee6ebadd4f509bb282aae.jpg)

![4dcc52a704840abe591716d60de1920af94f8c031cdef44698c07ce784251413.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/tables/4dcc52a704840abe591716d60de1920af94f8c031cdef44698c07ce784251413.jpg)

![64eb64d04016301395631dd5eed60b625d56ef809f59d95bd19dd251bdd8f3c4.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/tables/64eb64d04016301395631dd5eed60b625d56ef809f59d95bd19dd251bdd8f3c4.jpg)

![69998bd2ecf68a5185fac042109836a3eb5435073da5bb356eae211585f1452f.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/tables/69998bd2ecf68a5185fac042109836a3eb5435073da5bb356eae211585f1452f.jpg)

![6f88f0c7a6a07ea154dea1ec5c36e4432cbabab524973544e46e7f35501728b0.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/tables/6f88f0c7a6a07ea154dea1ec5c36e4432cbabab524973544e46e7f35501728b0.jpg)

![8fc382dcf31e28e62cd46abba9e57f93142bfe324a8b546ad5746ed9bb12da5d.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/tables/8fc382dcf31e28e62cd46abba9e57f93142bfe324a8b546ad5746ed9bb12da5d.jpg)

![95d7ed02f58756529f9e1d18be16fa454552f3d00ff9df92677f737cc308813d.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/tables/95d7ed02f58756529f9e1d18be16fa454552f3d00ff9df92677f737cc308813d.jpg)

![a2fcc8a0ecf9211eae269eaed19157964a4c0ac3fcb42efb425d44dfa33a0b46.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/tables/a2fcc8a0ecf9211eae269eaed19157964a4c0ac3fcb42efb425d44dfa33a0b46.jpg)

![a95f72710d9b10282e1eda733951ffbf9e20ab9a2c4a9e3fd7b1717ff4f35927.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/tables/a95f72710d9b10282e1eda733951ffbf9e20ab9a2c4a9e3fd7b1717ff4f35927.jpg)

![b2038cb960ef2134930ef704d9e7aaf80ad4603cd03db4b5608ba9420eb2d7e1.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/tables/b2038cb960ef2134930ef704d9e7aaf80ad4603cd03db4b5608ba9420eb2d7e1.jpg)

![bbefa61fd7925212f3c2643ef7a70ee76fca4bc440534df9bd9a8888ee860e29.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/tables/bbefa61fd7925212f3c2643ef7a70ee76fca4bc440534df9bd9a8888ee860e29.jpg)

![c301b7270bafd60b00bc7e1e2e0c62abdcde03de698d2df1ca4ac09c2d084800.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/tables/c301b7270bafd60b00bc7e1e2e0c62abdcde03de698d2df1ca4ac09c2d084800.jpg)

![dfec378d1d76bfa1ede4fd0eb18ce9b5e71b49a3fb53c477f6f8246aa7d0031c.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/tables/dfec378d1d76bfa1ede4fd0eb18ce9b5e71b49a3fb53c477f6f8246aa7d0031c.jpg)

![f6671cffe350e07a9b2998c2acdf6504702ebf652cce945c69496d65c181611d.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/tables/f6671cffe350e07a9b2998c2acdf6504702ebf652cce945c69496d65c181611d.jpg)

![f8a594419fd01e82a27e5c790d2d8f51cb9d4e13571f810ea5bd649de2ea7648.jpg](../iclr_results/845_Visual-O1_ Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoni/tables/f8a594419fd01e82a27e5c790d2d8f51cb9d4e13571f810ea5bd649de2ea7648.jpg)

## MMKE-Bench: A Multimodal Editing Benchmark for Diverse Visual Knowledge


### Images

![00ec9465942b27a7d511d2d4ea60616b955018331843ca391b3e6b5f9c240425.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/00ec9465942b27a7d511d2d4ea60616b955018331843ca391b3e6b5f9c240425.jpg)

![0bb6da7a3c9d34ba8a5055859ad59310fab784d627b6f11fef39c7272fde8357.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/0bb6da7a3c9d34ba8a5055859ad59310fab784d627b6f11fef39c7272fde8357.jpg)

![12c7e6b9347792969221707eb7aad12fbc0e09cd317da0329d12268391f1aec2.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/12c7e6b9347792969221707eb7aad12fbc0e09cd317da0329d12268391f1aec2.jpg)

![1d1292d81d6e8e4635613e0cb965a6cf8934da1eb1b2cf3d7085ef4c8d3e1a05.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/1d1292d81d6e8e4635613e0cb965a6cf8934da1eb1b2cf3d7085ef4c8d3e1a05.jpg)

![27d737191cf5a8e3de1a259d429c19117e37d6a1739aa4c306aeef9964a26973.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/27d737191cf5a8e3de1a259d429c19117e37d6a1739aa4c306aeef9964a26973.jpg)

![2e29f2875783b05860977f0befb2922c6650f1f552bb22b88fc0383435ab67ef.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/2e29f2875783b05860977f0befb2922c6650f1f552bb22b88fc0383435ab67ef.jpg)

![43071b9c105f7092bdea01124c92c194daa8493d16cb09b4b84dc707aecb1b9c.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/43071b9c105f7092bdea01124c92c194daa8493d16cb09b4b84dc707aecb1b9c.jpg)

![4c78855c0cecc4e3e57e0b6a57813e4b69be4516ecf5f2ec895e593e7522a493.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/4c78855c0cecc4e3e57e0b6a57813e4b69be4516ecf5f2ec895e593e7522a493.jpg)

![6197a76d69bb80a0080f66745164ccf9c368f796215d416a02cfd6cc4c1d6a84.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/6197a76d69bb80a0080f66745164ccf9c368f796215d416a02cfd6cc4c1d6a84.jpg)

![670184e9887d53d32650a621a300eb57c0f00156057cb9190f7fa83f5744ad64.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/670184e9887d53d32650a621a300eb57c0f00156057cb9190f7fa83f5744ad64.jpg)

![7ba4a96b3ff55d26df33de4181f3eae65e575a3b6b058879e2d5333acb6c5442.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/7ba4a96b3ff55d26df33de4181f3eae65e575a3b6b058879e2d5333acb6c5442.jpg)

![8fc32cf364fc5365d57c2b1b2584f97ea7ab50a056ba3a49144cbf7c874f64c0.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/8fc32cf364fc5365d57c2b1b2584f97ea7ab50a056ba3a49144cbf7c874f64c0.jpg)

![9759e554aa94bb648073bd56d0c7ed21105118f45a2ea8471b71aec940f25fd9.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/9759e554aa94bb648073bd56d0c7ed21105118f45a2ea8471b71aec940f25fd9.jpg)

![9cedbb53f1c9d7535e3dff583ae6e877097a6b0544209b81b3a725732c6f7360.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/9cedbb53f1c9d7535e3dff583ae6e877097a6b0544209b81b3a725732c6f7360.jpg)

![9d33c22ef5e558c70b8f9c5a0b15b9a1c73cb7645885eaf455c2826a5b13a9ef.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/9d33c22ef5e558c70b8f9c5a0b15b9a1c73cb7645885eaf455c2826a5b13a9ef.jpg)

![a2e418e0d1362a1249bb62d1ddc58b747351aa1319926cddf7f5119eac919664.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/a2e418e0d1362a1249bb62d1ddc58b747351aa1319926cddf7f5119eac919664.jpg)

![a66de59379920b748532c9e5859ab32a428e85889135a97ee0b6c0c3a2445a79.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/a66de59379920b748532c9e5859ab32a428e85889135a97ee0b6c0c3a2445a79.jpg)

![c6d231fcf6457faf6f99e445be5684a249415abd094a4ecafb5a6c43b308358d.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/c6d231fcf6457faf6f99e445be5684a249415abd094a4ecafb5a6c43b308358d.jpg)

![cb3cde95f9c4dc218efeda06eaeaadb4de6e6dcfb0e2568a053c898c51f56002.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/cb3cde95f9c4dc218efeda06eaeaadb4de6e6dcfb0e2568a053c898c51f56002.jpg)

![da16c462867d727c281be0adb0e3f3ff4c461e9cf3e48dd2b0ce5a6539215b4f.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/da16c462867d727c281be0adb0e3f3ff4c461e9cf3e48dd2b0ce5a6539215b4f.jpg)

![e155a960e8dcd792180ac35887796a113c5efb96aaf1a403cba606626efbe207.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/e155a960e8dcd792180ac35887796a113c5efb96aaf1a403cba606626efbe207.jpg)

![e3e392d7154d6f4632d7bd21ad849aebc8f85f6404be7edb6bc682b2993f874f.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/e3e392d7154d6f4632d7bd21ad849aebc8f85f6404be7edb6bc682b2993f874f.jpg)

![e9071ccb67f00ccf46e5512ea77ea6ea4a417386c61cebda6369923184c46ed4.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/e9071ccb67f00ccf46e5512ea77ea6ea4a417386c61cebda6369923184c46ed4.jpg)

![f1b38a308784bd0d319341955ce74c85b2da9e339068970df2999bc803ec2e6d.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/f1b38a308784bd0d319341955ce74c85b2da9e339068970df2999bc803ec2e6d.jpg)

![f2f4e9a5285be37e9a9aaa0b20ec7b2948edeb234051203f55f7de38c489c5ee.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/f2f4e9a5285be37e9a9aaa0b20ec7b2948edeb234051203f55f7de38c489c5ee.jpg)

![f63b80f7e8bb0d55c67c876cbb3a87bf3ea3f7646a2c55636be40a3d99cb7d8b.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/images/f63b80f7e8bb0d55c67c876cbb3a87bf3ea3f7646a2c55636be40a3d99cb7d8b.jpg)

### Tables

![051305678778c75271706afc1a2b81a529f4184d1e54ffc1559675fc8caecc3d.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/tables/051305678778c75271706afc1a2b81a529f4184d1e54ffc1559675fc8caecc3d.jpg)

![1871e9b24ab43cda5005bf4890d05f6f80a906e63ead6cb3fd2df1925f6eb384.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/tables/1871e9b24ab43cda5005bf4890d05f6f80a906e63ead6cb3fd2df1925f6eb384.jpg)

![20168ca3a248eb66a29ea7bc43c0c8bf03e8dd60e66632c81ded480c91fd14dc.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/tables/20168ca3a248eb66a29ea7bc43c0c8bf03e8dd60e66632c81ded480c91fd14dc.jpg)

![23dbc672d8afda0485dfbf21479ceee18bee781c2c4c27b24f6badf9edc2ba6d.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/tables/23dbc672d8afda0485dfbf21479ceee18bee781c2c4c27b24f6badf9edc2ba6d.jpg)

![466c4c4f87d329d634806431174270a46f2aa3228df163466ac5d455dcaa9af9.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/tables/466c4c4f87d329d634806431174270a46f2aa3228df163466ac5d455dcaa9af9.jpg)

![522525b6e7e3c12e8854e53df325f0fceb1391358430340cee2a07142e1bf9ac.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/tables/522525b6e7e3c12e8854e53df325f0fceb1391358430340cee2a07142e1bf9ac.jpg)

![57d2bf7e6a1b9b6a80316d9e4ef9d8f0926d5112b3d8518b4b26cc40b6cd904a.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/tables/57d2bf7e6a1b9b6a80316d9e4ef9d8f0926d5112b3d8518b4b26cc40b6cd904a.jpg)

![6727d0f7528702de015f53f4548e3cf8f31c15fa83193c95a941e29289f91cb2.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/tables/6727d0f7528702de015f53f4548e3cf8f31c15fa83193c95a941e29289f91cb2.jpg)

![71cc35ab6df44ebf9beb69d08cac58868fd3d064125965cdb99f329e277225a5.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/tables/71cc35ab6df44ebf9beb69d08cac58868fd3d064125965cdb99f329e277225a5.jpg)

![71eed0f28fd4380f73608325eafec61a8790286a3dc6750d43dd8178aef19b36.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/tables/71eed0f28fd4380f73608325eafec61a8790286a3dc6750d43dd8178aef19b36.jpg)

![80720673940e10185b0c14e82db4d896ee3376a945e99d479b58f762543f5fe0.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/tables/80720673940e10185b0c14e82db4d896ee3376a945e99d479b58f762543f5fe0.jpg)

![a1687dc9256be3f9b4ee98b3209184af53667bd049aac16389207716ebadaca9.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/tables/a1687dc9256be3f9b4ee98b3209184af53667bd049aac16389207716ebadaca9.jpg)

![a26c7ed012951a963dbf056a65185901d45810811d3fa0ec13474245fe55887f.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/tables/a26c7ed012951a963dbf056a65185901d45810811d3fa0ec13474245fe55887f.jpg)

![a5787f8aee614dae232612427f4b89cecc13f1ee7ad25a48ad6f1f8719f3eb64.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/tables/a5787f8aee614dae232612427f4b89cecc13f1ee7ad25a48ad6f1f8719f3eb64.jpg)

![c502e616c69765c7edcbbce67a44575307351fb9cb449cc89c2f8b04fdb64597.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/tables/c502e616c69765c7edcbbce67a44575307351fb9cb449cc89c2f8b04fdb64597.jpg)

![fd6d41add9f4ac07086dfe094af9931172da7ae89f336fa41168b812a2d223ea.jpg](../iclr_results/846_MMKE-Bench_ A Multimodal Editing Benchmark for Diverse Visual Knowledge/tables/fd6d41add9f4ac07086dfe094af9931172da7ae89f336fa41168b812a2d223ea.jpg)

## Multimodal Quantitative Language for Generative Recommendation


### Images

![2212a444c7518e3e6e34e5e330df710810c88a6fe8f5ddb2d8c73394fa4d730a.jpg](../iclr_results/847_Multimodal Quantitative Language for Generative Recommendation/images/2212a444c7518e3e6e34e5e330df710810c88a6fe8f5ddb2d8c73394fa4d730a.jpg)

![2698a1996d8435fa86645dbb8d9fe0e53bda42e2df4a5cb0a16fb90f5cf5bc6e.jpg](../iclr_results/847_Multimodal Quantitative Language for Generative Recommendation/images/2698a1996d8435fa86645dbb8d9fe0e53bda42e2df4a5cb0a16fb90f5cf5bc6e.jpg)

![a032430f7362edf10dcaea3191138b5d6ebfd8c4a28dfc01ce435c77e64e5f3f.jpg](../iclr_results/847_Multimodal Quantitative Language for Generative Recommendation/images/a032430f7362edf10dcaea3191138b5d6ebfd8c4a28dfc01ce435c77e64e5f3f.jpg)

![c2b2a7a92c78b586fe8c655c63d0552b026477d5a81f111becd824cd2d7e0042.jpg](../iclr_results/847_Multimodal Quantitative Language for Generative Recommendation/images/c2b2a7a92c78b586fe8c655c63d0552b026477d5a81f111becd824cd2d7e0042.jpg)

### Tables

![20d998ca234c8edaae74e2af0e64cf4d51fa13d25beb904570100fb29886e137.jpg](../iclr_results/847_Multimodal Quantitative Language for Generative Recommendation/tables/20d998ca234c8edaae74e2af0e64cf4d51fa13d25beb904570100fb29886e137.jpg)

![27091d4aa4279c244224e3727b5fff70ad598098a5a255b64e8968ee3709ca24.jpg](../iclr_results/847_Multimodal Quantitative Language for Generative Recommendation/tables/27091d4aa4279c244224e3727b5fff70ad598098a5a255b64e8968ee3709ca24.jpg)

![28ec38f24b4abab3ac23ac83a57225a6d6cb03496a4e4fad1e36f7303c2c6ca6.jpg](../iclr_results/847_Multimodal Quantitative Language for Generative Recommendation/tables/28ec38f24b4abab3ac23ac83a57225a6d6cb03496a4e4fad1e36f7303c2c6ca6.jpg)

![2be96b659662203853df78fea4be5163fd4e995e34643609c31b68e2ad6c9817.jpg](../iclr_results/847_Multimodal Quantitative Language for Generative Recommendation/tables/2be96b659662203853df78fea4be5163fd4e995e34643609c31b68e2ad6c9817.jpg)

![35b219636b66e98ba3566e3b414ab719450eca32e4d16131e66f70dbc0005ecb.jpg](../iclr_results/847_Multimodal Quantitative Language for Generative Recommendation/tables/35b219636b66e98ba3566e3b414ab719450eca32e4d16131e66f70dbc0005ecb.jpg)

![85f2edea0d3328fd14295ffb5b2e888dafe0e5886c0c80c2255c92d0162023f8.jpg](../iclr_results/847_Multimodal Quantitative Language for Generative Recommendation/tables/85f2edea0d3328fd14295ffb5b2e888dafe0e5886c0c80c2255c92d0162023f8.jpg)

![abe648209d9efd3448d1cd9f0d315668655486d5c8355bc13089dce08fca297d.jpg](../iclr_results/847_Multimodal Quantitative Language for Generative Recommendation/tables/abe648209d9efd3448d1cd9f0d315668655486d5c8355bc13089dce08fca297d.jpg)

![df9b4055f1fcac392948b2d44fc68d51542b166f01f6417fb3d1ca568e8d6ac8.jpg](../iclr_results/847_Multimodal Quantitative Language for Generative Recommendation/tables/df9b4055f1fcac392948b2d44fc68d51542b166f01f6417fb3d1ca568e8d6ac8.jpg)

## Does SGD really happen in tiny subspaces?


### Images

![06b09f749cc1ae3523d68cee74506cbf2d05a825b23bc408d554030c9c37a5d4.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/06b09f749cc1ae3523d68cee74506cbf2d05a825b23bc408d554030c9c37a5d4.jpg)

![0bf5e7d4dacbc39ca1d0bfbdd13a73a207d24c9ff0b01f19b3d80edbb9eb6a7d.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/0bf5e7d4dacbc39ca1d0bfbdd13a73a207d24c9ff0b01f19b3d80edbb9eb6a7d.jpg)

![102c7349d878f7d35afd1a2f9349d872918dadaa34a742be968d97781b06fd1c.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/102c7349d878f7d35afd1a2f9349d872918dadaa34a742be968d97781b06fd1c.jpg)

![10f98352847900a9d32a13d750989dcebaf9c38c442252243d283bfdbf8852b8.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/10f98352847900a9d32a13d750989dcebaf9c38c442252243d283bfdbf8852b8.jpg)

![253cc9d37c558e7acb114cd95c7ba18d4e817baccb72432e0b7a8e18855f86a6.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/253cc9d37c558e7acb114cd95c7ba18d4e817baccb72432e0b7a8e18855f86a6.jpg)

![26e07f0ec14f255b139c5e0e9ca66de44e6073a0379c2db02c426c0ace10c70f.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/26e07f0ec14f255b139c5e0e9ca66de44e6073a0379c2db02c426c0ace10c70f.jpg)

![2c0ba3a2ac5bd4c1b467a549692eb55045d42699bdc5c64eea32d8ae87074224.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/2c0ba3a2ac5bd4c1b467a549692eb55045d42699bdc5c64eea32d8ae87074224.jpg)

![2ce758e00acfcf6ab9c0331f1bb68391ddf8a1464f109639ab93782595b46974.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/2ce758e00acfcf6ab9c0331f1bb68391ddf8a1464f109639ab93782595b46974.jpg)

![327c578e9a66ae3ad87807a94dae8a243cc1a99db9d9e9b236e4bafc57ddade5.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/327c578e9a66ae3ad87807a94dae8a243cc1a99db9d9e9b236e4bafc57ddade5.jpg)

![3ed664355e1aa5103a764468da51522c332f4942e7f499a334d0b4f3939d434f.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/3ed664355e1aa5103a764468da51522c332f4942e7f499a334d0b4f3939d434f.jpg)

![42799b0641d0c96857ebd72e52924611f5ca386c342fc15e4d337de05b5c4f07.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/42799b0641d0c96857ebd72e52924611f5ca386c342fc15e4d337de05b5c4f07.jpg)

![4db356cc0c657ed03fe7454445c55e4a4bb361361f03a8c62e230b6acdb5dbf7.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/4db356cc0c657ed03fe7454445c55e4a4bb361361f03a8c62e230b6acdb5dbf7.jpg)

![4f3ca3791d0ee1f5062631aa1421a1a11daa142fdf9a2be1cb0241bb605eabda.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/4f3ca3791d0ee1f5062631aa1421a1a11daa142fdf9a2be1cb0241bb605eabda.jpg)

![4fac92aecc2121d7918cfc93dc6d9912da06f005710829201bc1d70be7faf386.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/4fac92aecc2121d7918cfc93dc6d9912da06f005710829201bc1d70be7faf386.jpg)

![53546b94ab51e22520e6ef9abe5fb26223e15adb31e516b2f3eb405ef092cf4c.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/53546b94ab51e22520e6ef9abe5fb26223e15adb31e516b2f3eb405ef092cf4c.jpg)

![5542f15c02a61493250e7d4f031b0f173cb270735ea69b610ddd3fe5aab7701e.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/5542f15c02a61493250e7d4f031b0f173cb270735ea69b610ddd3fe5aab7701e.jpg)

![59ca6673f01737f5a87a7485bdef14594657716ba57362ede020d276b5ec269e.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/59ca6673f01737f5a87a7485bdef14594657716ba57362ede020d276b5ec269e.jpg)

![631dd99b29bda16919e5b91980ee74a2a0dc10ac8977ba15dbb18338d83287d5.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/631dd99b29bda16919e5b91980ee74a2a0dc10ac8977ba15dbb18338d83287d5.jpg)

![64646c99040a63ef069fa7a007ded80bf7514b92d1cac6896eb7598cbbff0cf3.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/64646c99040a63ef069fa7a007ded80bf7514b92d1cac6896eb7598cbbff0cf3.jpg)

![655ceea1f5482822f52da69c29ca2188e50e8371306f5458d216a3aa78a176db.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/655ceea1f5482822f52da69c29ca2188e50e8371306f5458d216a3aa78a176db.jpg)

![66f693cb481400926830040daaea98bd3841b2f1fd4ea9151fbea61bd2f7b442.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/66f693cb481400926830040daaea98bd3841b2f1fd4ea9151fbea61bd2f7b442.jpg)

![6889b4c57f0b361e0640ef5d86231ec2c8c8fdfcbf7f6eb8c0840ffacb471163.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/6889b4c57f0b361e0640ef5d86231ec2c8c8fdfcbf7f6eb8c0840ffacb471163.jpg)

![69820a768d2aa6f56ce4df39b5dda375034e1ff11827b24829bedc5c8ff191f9.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/69820a768d2aa6f56ce4df39b5dda375034e1ff11827b24829bedc5c8ff191f9.jpg)

![85ccbc65ab2026cb5c73f84fc43f3c723ce104497354ff88cb0f6b16c06eddea.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/85ccbc65ab2026cb5c73f84fc43f3c723ce104497354ff88cb0f6b16c06eddea.jpg)

![8ba8d39321adc9fa191f0d2f84c014172f9bc1b310d0068be0106be2d16d52a6.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/8ba8d39321adc9fa191f0d2f84c014172f9bc1b310d0068be0106be2d16d52a6.jpg)

![98fed387be79901f9a0953dcfb60303325426a9a4cf5c3025ef8c4024810767b.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/98fed387be79901f9a0953dcfb60303325426a9a4cf5c3025ef8c4024810767b.jpg)

![994d72d8557d96033bc13cb106a126dbef21b3bd9a2e77050293041418ecb298.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/994d72d8557d96033bc13cb106a126dbef21b3bd9a2e77050293041418ecb298.jpg)

![a15f77d8bf002ef0725e2103d35cc3aaacc2260ce855eed4579d59b961c2d92b.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/a15f77d8bf002ef0725e2103d35cc3aaacc2260ce855eed4579d59b961c2d92b.jpg)

![a2667dc5926b165c294eacaf06ba0db0089cecc9b0e5494d44fecd132cd69260.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/a2667dc5926b165c294eacaf06ba0db0089cecc9b0e5494d44fecd132cd69260.jpg)

![a9d799a7e8e2800b67c5bcb3d655161bf922c5137b567bfef8b1bb8f8707056d.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/a9d799a7e8e2800b67c5bcb3d655161bf922c5137b567bfef8b1bb8f8707056d.jpg)

![b0d91b59d504fce45ffd33dba837bef6b0069570b29269464ec116c505d6eb73.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/b0d91b59d504fce45ffd33dba837bef6b0069570b29269464ec116c505d6eb73.jpg)

![bdd1d3608a2695d4b4b2d9a748028f7106a5cfe9aa474d8a6e828a198c65c496.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/bdd1d3608a2695d4b4b2d9a748028f7106a5cfe9aa474d8a6e828a198c65c496.jpg)

![c09be1e4d662324ed407b1bdab91dd54dba9d838b07bce37eed164a8917d2f0e.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/c09be1e4d662324ed407b1bdab91dd54dba9d838b07bce37eed164a8917d2f0e.jpg)

![c21fab2bfb0f87133fb30dfa15052b1f6816cd53d321f821b8430ba4dcbb2c84.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/c21fab2bfb0f87133fb30dfa15052b1f6816cd53d321f821b8430ba4dcbb2c84.jpg)

![c51940f7923c356172b40a14d498d2fd405fbac744ca05bbc52e98cc7d464004.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/c51940f7923c356172b40a14d498d2fd405fbac744ca05bbc52e98cc7d464004.jpg)

![c80d3b555984392482d2112e47912926bb8499a057e7e0bc97271625afad3560.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/c80d3b555984392482d2112e47912926bb8499a057e7e0bc97271625afad3560.jpg)

![c921c3415f103a2acfdb04386625574dcd11b6732fc7ece2c4027ee5cc3b15b8.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/c921c3415f103a2acfdb04386625574dcd11b6732fc7ece2c4027ee5cc3b15b8.jpg)

![cadec9ae5ff3bd563f31829209c7cc7f192503dfa2083ada818ff90cd6bc5c05.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/cadec9ae5ff3bd563f31829209c7cc7f192503dfa2083ada818ff90cd6bc5c05.jpg)

![cf020dbdcbb6b997a4bec4d30788ed47f39fe259d1b586cd26be4de5ae432474.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/cf020dbdcbb6b997a4bec4d30788ed47f39fe259d1b586cd26be4de5ae432474.jpg)

![d374752c1d567345e90fdd5c9aa698acaf1203b2f5b1b395a4eb579d5b9f4894.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/d374752c1d567345e90fdd5c9aa698acaf1203b2f5b1b395a4eb579d5b9f4894.jpg)

![d9cc90a41e30452a1028296a079328cc6a55f6ce6ba357c0b284c4aa24e2b75d.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/d9cc90a41e30452a1028296a079328cc6a55f6ce6ba357c0b284c4aa24e2b75d.jpg)

![ddc5b2e3c8e80f0cce53bd6efd732270ac7ae7eae8e6307799733e1776cfc684.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/ddc5b2e3c8e80f0cce53bd6efd732270ac7ae7eae8e6307799733e1776cfc684.jpg)

![e17b869fa217e69a1aa429f4c3daa32030812601f122d3e83e2e1e00474145e3.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/e17b869fa217e69a1aa429f4c3daa32030812601f122d3e83e2e1e00474145e3.jpg)

![e752353cc5e74fee15c3bb6914e5420348204c0b4c3ad6dc5256dd784e591bda.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/e752353cc5e74fee15c3bb6914e5420348204c0b4c3ad6dc5256dd784e591bda.jpg)

![f1cebfb8a891e6bd517a76090e46bf9e893b5154f5bfc76121e55c2713f84dc8.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/f1cebfb8a891e6bd517a76090e46bf9e893b5154f5bfc76121e55c2713f84dc8.jpg)

![ff4c64f5bba27bc54f455d7ecdb79691c86a41f68852473189bc187bd76d2716.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/images/ff4c64f5bba27bc54f455d7ecdb79691c86a41f68852473189bc187bd76d2716.jpg)

### Tables

![046ab72f1490e3ad859213e2b53dd844a15744c699a42434a02ba8c32e9de5cb.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/tables/046ab72f1490e3ad859213e2b53dd844a15744c699a42434a02ba8c32e9de5cb.jpg)

![c877000edb88f08d1df8ccd9eeffef81e8792c9706354558b6a1b64150b7fc11.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/tables/c877000edb88f08d1df8ccd9eeffef81e8792c9706354558b6a1b64150b7fc11.jpg)

![fa22da4618634df39b4bf8667cc6daab267c2e6048da6feb7ca92eb939f9cf32.jpg](../iclr_results/848_Does SGD really happen in tiny subspaces_/tables/fa22da4618634df39b4bf8667cc6daab267c2e6048da6feb7ca92eb939f9cf32.jpg)

## Q-SFT: Q-Learning for Language Models via Supervised Fine-Tuning


### Images

![37609f894ab9417309e8af58b09703f9456936f4c63583d1a40ba994759600d3.jpg](../iclr_results/849_Q-SFT_ Q-Learning for Language Models via Supervised Fine-Tuning/images/37609f894ab9417309e8af58b09703f9456936f4c63583d1a40ba994759600d3.jpg)

![4c4d06d8e840a50f1e004cc8b52342e821a72f80f28f60230c7f47e7b0c9c17a.jpg](../iclr_results/849_Q-SFT_ Q-Learning for Language Models via Supervised Fine-Tuning/images/4c4d06d8e840a50f1e004cc8b52342e821a72f80f28f60230c7f47e7b0c9c17a.jpg)

![4da7d8c7a14e020759789f907fb8c76dbe4a252e3b14ee213be277b2dd12837a.jpg](../iclr_results/849_Q-SFT_ Q-Learning for Language Models via Supervised Fine-Tuning/images/4da7d8c7a14e020759789f907fb8c76dbe4a252e3b14ee213be277b2dd12837a.jpg)

![bece6f1936f52652f016609b6753e71c57e2dae1f069bb2dc09a1bb987c7d9b6.jpg](../iclr_results/849_Q-SFT_ Q-Learning for Language Models via Supervised Fine-Tuning/images/bece6f1936f52652f016609b6753e71c57e2dae1f069bb2dc09a1bb987c7d9b6.jpg)

![d46579fe7289e6ba6af5ae20402061b9ab3b3e578a9c5c927b6a74a9ba631a99.jpg](../iclr_results/849_Q-SFT_ Q-Learning for Language Models via Supervised Fine-Tuning/images/d46579fe7289e6ba6af5ae20402061b9ab3b3e578a9c5c927b6a74a9ba631a99.jpg)

### Tables

![2545f763168f68146283ddaed180eaf3bb57370323b6a4e8f8a05fe4a85fdcaf.jpg](../iclr_results/849_Q-SFT_ Q-Learning for Language Models via Supervised Fine-Tuning/tables/2545f763168f68146283ddaed180eaf3bb57370323b6a4e8f8a05fe4a85fdcaf.jpg)

![509ea55b0d896f7d632064fae61c0c9719ed26c2a7984c3659d4ad5e82851d52.jpg](../iclr_results/849_Q-SFT_ Q-Learning for Language Models via Supervised Fine-Tuning/tables/509ea55b0d896f7d632064fae61c0c9719ed26c2a7984c3659d4ad5e82851d52.jpg)

![7fbc0d4f94126374dea514df7fc9c435041b226f96b8da2f42eced4bb32852f3.jpg](../iclr_results/849_Q-SFT_ Q-Learning for Language Models via Supervised Fine-Tuning/tables/7fbc0d4f94126374dea514df7fc9c435041b226f96b8da2f42eced4bb32852f3.jpg)

![cd40a5ff6b086050a09c01d7dc966cb7cf783837295d2a7c7210d3a03eb9a440.jpg](../iclr_results/849_Q-SFT_ Q-Learning for Language Models via Supervised Fine-Tuning/tables/cd40a5ff6b086050a09c01d7dc966cb7cf783837295d2a7c7210d3a03eb9a440.jpg)

## PostCast: Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Modeling


### Images

![0b1ccf932f52e97697ad053578eb5507fccdbe5973733a983d5455cf52bb445c.jpg](../iclr_results/850_PostCast_ Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Mode/images/0b1ccf932f52e97697ad053578eb5507fccdbe5973733a983d5455cf52bb445c.jpg)

![13f82dce44dae7bb392548ca00377b5236f6685664a23218c8bd36d0d73f81df.jpg](../iclr_results/850_PostCast_ Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Mode/images/13f82dce44dae7bb392548ca00377b5236f6685664a23218c8bd36d0d73f81df.jpg)

![24b65159882f8fe86ae4c9a0abc63137c9df841abdd5cf6d74b11d686ae06d10.jpg](../iclr_results/850_PostCast_ Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Mode/images/24b65159882f8fe86ae4c9a0abc63137c9df841abdd5cf6d74b11d686ae06d10.jpg)

![419abac79195d251ac72bb6c4a448e09614afcfceef76a9b8bb3959c82e3b808.jpg](../iclr_results/850_PostCast_ Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Mode/images/419abac79195d251ac72bb6c4a448e09614afcfceef76a9b8bb3959c82e3b808.jpg)

![47d03f6c9da1d2bb3692189a0a74ec9c97961dda799a48321a01d9dd50cc03ea.jpg](../iclr_results/850_PostCast_ Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Mode/images/47d03f6c9da1d2bb3692189a0a74ec9c97961dda799a48321a01d9dd50cc03ea.jpg)

![57d18df34a10e007396b7785b1a6bd0194085fbe139b0f366d7071df4b6de1a1.jpg](../iclr_results/850_PostCast_ Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Mode/images/57d18df34a10e007396b7785b1a6bd0194085fbe139b0f366d7071df4b6de1a1.jpg)

![60c66e2a36aa26f1e2d6bd423e17887024a9f360ff4e77033257e2da819a3297.jpg](../iclr_results/850_PostCast_ Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Mode/images/60c66e2a36aa26f1e2d6bd423e17887024a9f360ff4e77033257e2da819a3297.jpg)

![6a833bb8f73582cacc2010f7466df7f487d1b8217dd3ebdc54e8d8dec5e30883.jpg](../iclr_results/850_PostCast_ Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Mode/images/6a833bb8f73582cacc2010f7466df7f487d1b8217dd3ebdc54e8d8dec5e30883.jpg)

![79ea413743b498c36d1124e9e3b3f9b2899e4de5a6e169f3dd4e6f425c2b389f.jpg](../iclr_results/850_PostCast_ Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Mode/images/79ea413743b498c36d1124e9e3b3f9b2899e4de5a6e169f3dd4e6f425c2b389f.jpg)

![82f61cd1ba8265e201f9a7f23b68e26fc87803f834d262d629246f8c833316f9.jpg](../iclr_results/850_PostCast_ Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Mode/images/82f61cd1ba8265e201f9a7f23b68e26fc87803f834d262d629246f8c833316f9.jpg)

![8755ba91ca925c153e9bd32cec32dba33718de6287eb28d553666f2e7279cbd4.jpg](../iclr_results/850_PostCast_ Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Mode/images/8755ba91ca925c153e9bd32cec32dba33718de6287eb28d553666f2e7279cbd4.jpg)

![96fd63cf6a4ac32a98adf35aa26310835d6836886d6499f988f52502de3f5ac1.jpg](../iclr_results/850_PostCast_ Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Mode/images/96fd63cf6a4ac32a98adf35aa26310835d6836886d6499f988f52502de3f5ac1.jpg)

![c7f254d6b9021252fa8fd550847fc6446302b0674439a1514dc440112c904d9c.jpg](../iclr_results/850_PostCast_ Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Mode/images/c7f254d6b9021252fa8fd550847fc6446302b0674439a1514dc440112c904d9c.jpg)

![cfdc5bd4059c48db95aedc3dee2a6ce66b3d954432c717cfc408ecd00133c95c.jpg](../iclr_results/850_PostCast_ Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Mode/images/cfdc5bd4059c48db95aedc3dee2a6ce66b3d954432c717cfc408ecd00133c95c.jpg)

![d35966f37fc364d2ff0ee280d9db0247687c963738a63ce5e150ab0f4a974d05.jpg](../iclr_results/850_PostCast_ Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Mode/images/d35966f37fc364d2ff0ee280d9db0247687c963738a63ce5e150ab0f4a974d05.jpg)

![e7b57dad43aa47d8a5904be8366448453a5d2d345138d15e6910d6b69814dcba.jpg](../iclr_results/850_PostCast_ Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Mode/images/e7b57dad43aa47d8a5904be8366448453a5d2d345138d15e6910d6b69814dcba.jpg)

![fb7e619cef4c652ce36d2b3c7731c913af0e5a9f766545ea9a7259bb7b3cebc2.jpg](../iclr_results/850_PostCast_ Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Mode/images/fb7e619cef4c652ce36d2b3c7731c913af0e5a9f766545ea9a7259bb7b3cebc2.jpg)

### Tables

![31a8f055f69d7a0110eb963318868e59f0f35c7bcfa7e48b9644f7d5796bac91.jpg](../iclr_results/850_PostCast_ Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Mode/tables/31a8f055f69d7a0110eb963318868e59f0f35c7bcfa7e48b9644f7d5796bac91.jpg)

![69e839d389ef6447a6198fd25e8e9488c1ef084fe7a3d305ac2148c79947d8e5.jpg](../iclr_results/850_PostCast_ Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Mode/tables/69e839d389ef6447a6198fd25e8e9488c1ef084fe7a3d305ac2148c79947d8e5.jpg)

![703d2fb5ecda5ff41f134ab41b39ec392ba31e392ff501c77683d0077ba875ab.jpg](../iclr_results/850_PostCast_ Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Mode/tables/703d2fb5ecda5ff41f134ab41b39ec392ba31e392ff501c77683d0077ba875ab.jpg)

![a6353ded914923111b290a8aacaec4f0c2220b8d3652c2412a0e41423dde117e.jpg](../iclr_results/850_PostCast_ Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Mode/tables/a6353ded914923111b290a8aacaec4f0c2220b8d3652c2412a0e41423dde117e.jpg)

![f9e5bab4f4e4548500fcdd0d27cc985e28a98be73762c4778fbd92b004b1ed75.jpg](../iclr_results/850_PostCast_ Generalizable Postprocessing for Precipitation Nowcasting via Unsupervised Blurriness Mode/tables/f9e5bab4f4e4548500fcdd0d27cc985e28a98be73762c4778fbd92b004b1ed75.jpg)

## Decoupled Subgraph Federated Learning


### Images

![0c0894ecbe59d25a0f2e4fba6cac00acbe3d12c7747559b5f322ec52fec90bf4.jpg](../iclr_results/851_Decoupled Subgraph Federated Learning/images/0c0894ecbe59d25a0f2e4fba6cac00acbe3d12c7747559b5f322ec52fec90bf4.jpg)

![24c2ac9ab7bf5943a1fc1c278d1aba79697c9b0c89cd90106c9da6471afcc8b6.jpg](../iclr_results/851_Decoupled Subgraph Federated Learning/images/24c2ac9ab7bf5943a1fc1c278d1aba79697c9b0c89cd90106c9da6471afcc8b6.jpg)

![258a73eaa6db0fc06605dec3261b68aaaadf48f21e9426048b9f2103b00f5f9b.jpg](../iclr_results/851_Decoupled Subgraph Federated Learning/images/258a73eaa6db0fc06605dec3261b68aaaadf48f21e9426048b9f2103b00f5f9b.jpg)

![d81f17283ad6685e1358ad08353c766fa967d05fb60f49167bb55446783780e4.jpg](../iclr_results/851_Decoupled Subgraph Federated Learning/images/d81f17283ad6685e1358ad08353c766fa967d05fb60f49167bb55446783780e4.jpg)

![e4dfd5da0a156925191e990db5437fbe2e54f1f54edc0145cc5f7ad3629e3b88.jpg](../iclr_results/851_Decoupled Subgraph Federated Learning/images/e4dfd5da0a156925191e990db5437fbe2e54f1f54edc0145cc5f7ad3629e3b88.jpg)

![e968554dcba8b560f10eae1625d817c2938de6d3f0b5113efd4d750e5ac5d7dc.jpg](../iclr_results/851_Decoupled Subgraph Federated Learning/images/e968554dcba8b560f10eae1625d817c2938de6d3f0b5113efd4d750e5ac5d7dc.jpg)

### Tables

![00a74c76400a862c94ea5c9d58c68c18b84d92297777eb54165379ad4d530839.jpg](../iclr_results/851_Decoupled Subgraph Federated Learning/tables/00a74c76400a862c94ea5c9d58c68c18b84d92297777eb54165379ad4d530839.jpg)

![0709dce3775cc9fceeba175fcf0f2b5fb60f6bba17dd5c1b6ec25fbeb346b453.jpg](../iclr_results/851_Decoupled Subgraph Federated Learning/tables/0709dce3775cc9fceeba175fcf0f2b5fb60f6bba17dd5c1b6ec25fbeb346b453.jpg)

![08010542565afc5874a9b1ac7fdb41c59bbc2bb361d40481b4341f28748e6c35.jpg](../iclr_results/851_Decoupled Subgraph Federated Learning/tables/08010542565afc5874a9b1ac7fdb41c59bbc2bb361d40481b4341f28748e6c35.jpg)

![208160b142171c6d598631971fd493b8e865943830ffe2b92f4db01f263af385.jpg](../iclr_results/851_Decoupled Subgraph Federated Learning/tables/208160b142171c6d598631971fd493b8e865943830ffe2b92f4db01f263af385.jpg)

![64c700ad4de83ae1457f9c0f235e5d052405b05e61d32960ce16c63a6c58cee8.jpg](../iclr_results/851_Decoupled Subgraph Federated Learning/tables/64c700ad4de83ae1457f9c0f235e5d052405b05e61d32960ce16c63a6c58cee8.jpg)

![7686aa747f8d0d4610857f0582a519be33de1a96a6bf5f44d5bb579fca7a47a7.jpg](../iclr_results/851_Decoupled Subgraph Federated Learning/tables/7686aa747f8d0d4610857f0582a519be33de1a96a6bf5f44d5bb579fca7a47a7.jpg)

![81bc750f7dac8d4d9403f71e25a0ca2986ca1f3b2fabb07f01fd2fc33e51e2ad.jpg](../iclr_results/851_Decoupled Subgraph Federated Learning/tables/81bc750f7dac8d4d9403f71e25a0ca2986ca1f3b2fabb07f01fd2fc33e51e2ad.jpg)

![b7ce67e881670ac84f13f29f4a0463966a860f8b65924043d3a70231861a6c4b.jpg](../iclr_results/851_Decoupled Subgraph Federated Learning/tables/b7ce67e881670ac84f13f29f4a0463966a860f8b65924043d3a70231861a6c4b.jpg)

![be5dafb7cbdb7428a075c8125ea0c15c2c8c7fe8019d89fd52f2993c98b745f8.jpg](../iclr_results/851_Decoupled Subgraph Federated Learning/tables/be5dafb7cbdb7428a075c8125ea0c15c2c8c7fe8019d89fd52f2993c98b745f8.jpg)

![c811cda072d328ea6c77e3983e0c245217bc4bea1e3c9de29b383ca0153f6af5.jpg](../iclr_results/851_Decoupled Subgraph Federated Learning/tables/c811cda072d328ea6c77e3983e0c245217bc4bea1e3c9de29b383ca0153f6af5.jpg)

![dbc28dfc4ef0f4519c808c731bccb02836ae3059fcb89927b3b180e9543e576f.jpg](../iclr_results/851_Decoupled Subgraph Federated Learning/tables/dbc28dfc4ef0f4519c808c731bccb02836ae3059fcb89927b3b180e9543e576f.jpg)

![e2d983cf3c8ad6721a27a86d21cb08660d331fc248dfd504ec4687a93057e2a0.jpg](../iclr_results/851_Decoupled Subgraph Federated Learning/tables/e2d983cf3c8ad6721a27a86d21cb08660d331fc248dfd504ec4687a93057e2a0.jpg)

## AniSDF: Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruction


### Images

![084d67d8af97baf5c8096afa80521af0392a43b2e8dbf124e542e02ccb664baa.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/images/084d67d8af97baf5c8096afa80521af0392a43b2e8dbf124e542e02ccb664baa.jpg)

![1b5d33fead53215314c1f521801cf61cfa359a772446d426fa3dc5841fd0bdf5.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/images/1b5d33fead53215314c1f521801cf61cfa359a772446d426fa3dc5841fd0bdf5.jpg)

![1c95b80b679b0e1b20d6a976902ebdfb08fe2288c86a633ad0bbf121a8f05526.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/images/1c95b80b679b0e1b20d6a976902ebdfb08fe2288c86a633ad0bbf121a8f05526.jpg)

![22d338e538827ca6b36da91923898a71bdd185459b25d732fe30ed0928c00b3e.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/images/22d338e538827ca6b36da91923898a71bdd185459b25d732fe30ed0928c00b3e.jpg)

![2d6355dbf7e20659237c137027475ae78fb77df0c9d1b60c7d5e23983bb469e7.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/images/2d6355dbf7e20659237c137027475ae78fb77df0c9d1b60c7d5e23983bb469e7.jpg)

![321a470d551722fa9f6cc81c5cd6de8ebb4f09382694edfcde827df5667d7d49.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/images/321a470d551722fa9f6cc81c5cd6de8ebb4f09382694edfcde827df5667d7d49.jpg)

![37a3c606dabe3dadf1a9df5b65b4204934cea18707e906902e8b723ed7e06887.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/images/37a3c606dabe3dadf1a9df5b65b4204934cea18707e906902e8b723ed7e06887.jpg)

![40bf23187633c27d48cdc57b842fc15f489b1e30216acf732755d37d5ad79ccb.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/images/40bf23187633c27d48cdc57b842fc15f489b1e30216acf732755d37d5ad79ccb.jpg)

![43548a425cc29aaa37e079ceb5692a421ae1043002c6c820080e6b0d0f3bbd13.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/images/43548a425cc29aaa37e079ceb5692a421ae1043002c6c820080e6b0d0f3bbd13.jpg)

![5b975795358708c078031fc15ee7cb1d5c9c5ae9a853255daf166d5184d4e1b4.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/images/5b975795358708c078031fc15ee7cb1d5c9c5ae9a853255daf166d5184d4e1b4.jpg)

![6b5620b930b6ba15b82ad20fbb2b5e8d07eded0071f5f71fb6eb6d69c16cf9c7.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/images/6b5620b930b6ba15b82ad20fbb2b5e8d07eded0071f5f71fb6eb6d69c16cf9c7.jpg)

![7c4b4f7534d2d173063c5688d17caa951cc984d6a63ac17aab5b3481741be8cb.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/images/7c4b4f7534d2d173063c5688d17caa951cc984d6a63ac17aab5b3481741be8cb.jpg)

![816a495723756653a7cebd2c18b680933474b41700e6b8f815a143a8629a34f9.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/images/816a495723756653a7cebd2c18b680933474b41700e6b8f815a143a8629a34f9.jpg)

![84a110e2bb6953752bd34f0f7a0aae67e8f6416de4862675b1b3305bb73dfcaa.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/images/84a110e2bb6953752bd34f0f7a0aae67e8f6416de4862675b1b3305bb73dfcaa.jpg)

![88838f18c09256da55db6fb6af1143c6b7b51f551621715936e85117fa5b5cbb.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/images/88838f18c09256da55db6fb6af1143c6b7b51f551621715936e85117fa5b5cbb.jpg)

![cc8fa95c5940ee8927b5e29b4db5a6912c009b5873fba1e8cd113fb8ace3e48a.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/images/cc8fa95c5940ee8927b5e29b4db5a6912c009b5873fba1e8cd113fb8ace3e48a.jpg)

![ccb95bd5860b8ae21787feb17a8471da437dd562f7086b0da48e808e9257f9cd.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/images/ccb95bd5860b8ae21787feb17a8471da437dd562f7086b0da48e808e9257f9cd.jpg)

![d817bc81143654abf5a06c2f1c60e4a30a560f52fe8b76cff2f6e4463382835c.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/images/d817bc81143654abf5a06c2f1c60e4a30a560f52fe8b76cff2f6e4463382835c.jpg)

![ed13545c5245d0bb140ff844ca8355adc7e1ade17e9fdd3ed8a35108e3e10a39.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/images/ed13545c5245d0bb140ff844ca8355adc7e1ade17e9fdd3ed8a35108e3e10a39.jpg)

![eef52a90e5327eb35544f66b9ffcd9004b7369e1879a62b2baf16a276039af2f.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/images/eef52a90e5327eb35544f66b9ffcd9004b7369e1879a62b2baf16a276039af2f.jpg)

![fe8bc561a13bf848fbebf488a0a59bacedf28ef37036c1afde760ea8dd45ffdc.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/images/fe8bc561a13bf848fbebf488a0a59bacedf28ef37036c1afde760ea8dd45ffdc.jpg)

![fef270099bb180768d390a074fb34519d47bfff7e73725e910189491d104cdda.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/images/fef270099bb180768d390a074fb34519d47bfff7e73725e910189491d104cdda.jpg)

![fef55fc14fdd33dc4d405200dd101594ce8da04a8cac62a46aa9f090248728c6.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/images/fef55fc14fdd33dc4d405200dd101594ce8da04a8cac62a46aa9f090248728c6.jpg)

### Tables

![25654bf383d9963f4ecfe4be94b3ac6568a3e6ec0bacfedbf3ce4babca9a722e.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/tables/25654bf383d9963f4ecfe4be94b3ac6568a3e6ec0bacfedbf3ce4babca9a722e.jpg)

![2e1b8a4a1ba3783d7262bec67660f207f27e39abd62cfa00e51ec5ed3cc736b5.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/tables/2e1b8a4a1ba3783d7262bec67660f207f27e39abd62cfa00e51ec5ed3cc736b5.jpg)

![59f5cea774765c3ff4c874226ec0ced46ae7194a6b4c6caf1d40c2f3cb0a8cc0.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/tables/59f5cea774765c3ff4c874226ec0ced46ae7194a6b4c6caf1d40c2f3cb0a8cc0.jpg)

![de60120bf5497fcb15cbe5857ad625d5bac071796b71e802b4f3f69017542ce4.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/tables/de60120bf5497fcb15cbe5857ad625d5bac071796b71e802b4f3f69017542ce4.jpg)

![fd1a86385eeb449ea6e6c470aaf06dbd404907d446e86d7326a12b36678edfce.jpg](../iclr_results/852_AniSDF_ Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruct/tables/fd1a86385eeb449ea6e6c470aaf06dbd404907d446e86d7326a12b36678edfce.jpg)

## Schur's Positive-Definite Network: Deep Learning in the SPD cone with structure


### Images

![0cdef0311685e3bf4021e5cbedc623423a913d1137cd186492b6c0a166c752a6.jpg](../iclr_results/853_Schur's Positive-Definite Network_ Deep Learning in the SPD cone with structure/images/0cdef0311685e3bf4021e5cbedc623423a913d1137cd186492b6c0a166c752a6.jpg)

![0d52505db148ba207099dd33e29876fc68cfafd71c827c39e1cf6fdf7ae31ba7.jpg](../iclr_results/853_Schur's Positive-Definite Network_ Deep Learning in the SPD cone with structure/images/0d52505db148ba207099dd33e29876fc68cfafd71c827c39e1cf6fdf7ae31ba7.jpg)

![2109d2d48f149443167248dc894236dc8e70fad098d5d0d9ab224ab8250f86ba.jpg](../iclr_results/853_Schur's Positive-Definite Network_ Deep Learning in the SPD cone with structure/images/2109d2d48f149443167248dc894236dc8e70fad098d5d0d9ab224ab8250f86ba.jpg)

![215ff0dfe95ab8d5f0f56b74a0fefa4b50575278d93b71a435cddcae09169e84.jpg](../iclr_results/853_Schur's Positive-Definite Network_ Deep Learning in the SPD cone with structure/images/215ff0dfe95ab8d5f0f56b74a0fefa4b50575278d93b71a435cddcae09169e84.jpg)

![31bd7c583130287bbec1e5f5390cf6cf953b6d27107ed9bd87ecfe5a2f78ce6c.jpg](../iclr_results/853_Schur's Positive-Definite Network_ Deep Learning in the SPD cone with structure/images/31bd7c583130287bbec1e5f5390cf6cf953b6d27107ed9bd87ecfe5a2f78ce6c.jpg)

![478649c356628961dc455f5c7c8887962a7ed3db5ecf9a8721b150ee8cf4e400.jpg](../iclr_results/853_Schur's Positive-Definite Network_ Deep Learning in the SPD cone with structure/images/478649c356628961dc455f5c7c8887962a7ed3db5ecf9a8721b150ee8cf4e400.jpg)

![6e9bc12493e8a07d48e7faf3e8cfdd54d6828b090ece89ed2a20011f724ad821.jpg](../iclr_results/853_Schur's Positive-Definite Network_ Deep Learning in the SPD cone with structure/images/6e9bc12493e8a07d48e7faf3e8cfdd54d6828b090ece89ed2a20011f724ad821.jpg)

![8710fd3ab3c0523901529387b8629e8665e09e0faa5f12a9f94a55b752582e65.jpg](../iclr_results/853_Schur's Positive-Definite Network_ Deep Learning in the SPD cone with structure/images/8710fd3ab3c0523901529387b8629e8665e09e0faa5f12a9f94a55b752582e65.jpg)

![b647b781cb125f5f298ed3233e9b1dd9ff57bda40d691adb0debcbe929f10dc1.jpg](../iclr_results/853_Schur's Positive-Definite Network_ Deep Learning in the SPD cone with structure/images/b647b781cb125f5f298ed3233e9b1dd9ff57bda40d691adb0debcbe929f10dc1.jpg)

![f5b4ea938ad23cb834e4428a9bb10a6f21cc1b8e2279323c2bc10c196ad5de53.jpg](../iclr_results/853_Schur's Positive-Definite Network_ Deep Learning in the SPD cone with structure/images/f5b4ea938ad23cb834e4428a9bb10a6f21cc1b8e2279323c2bc10c196ad5de53.jpg)

## Selective Attention Improves Transformer


### Images

![19d6601c2b50f4f907f6a1767768f4086962ae88bb06f344a2e9dd4b2c09747d.jpg](../iclr_results/854_Selective Attention Improves Transformer/images/19d6601c2b50f4f907f6a1767768f4086962ae88bb06f344a2e9dd4b2c09747d.jpg)

![2b38332506d4ac467dba2f8cf1f00c6e65dafe476ec91ba6199746d11bf6be64.jpg](../iclr_results/854_Selective Attention Improves Transformer/images/2b38332506d4ac467dba2f8cf1f00c6e65dafe476ec91ba6199746d11bf6be64.jpg)

![377173579834f076375b136605fe121a0f370b07ac74f69f4f409fe70f606486.jpg](../iclr_results/854_Selective Attention Improves Transformer/images/377173579834f076375b136605fe121a0f370b07ac74f69f4f409fe70f606486.jpg)

![4e45817a812cc6be79eb6e300438e5217ffa8c33aaf4736284ef3269d65798e3.jpg](../iclr_results/854_Selective Attention Improves Transformer/images/4e45817a812cc6be79eb6e300438e5217ffa8c33aaf4736284ef3269d65798e3.jpg)

![58537cafdde591eb8e3dd47251b9d7a810a8a40288ddb1c128120acea46c143d.jpg](../iclr_results/854_Selective Attention Improves Transformer/images/58537cafdde591eb8e3dd47251b9d7a810a8a40288ddb1c128120acea46c143d.jpg)

![c50d54c0c6d7b2edbebeff94c51a71d14c751cd591df5dcdc5a97a96a8f52da4.jpg](../iclr_results/854_Selective Attention Improves Transformer/images/c50d54c0c6d7b2edbebeff94c51a71d14c751cd591df5dcdc5a97a96a8f52da4.jpg)

![ceef72a9ac83ccfd2c86c6321075e5181fa780a9f5dc48c159b40a2ba90c4282.jpg](../iclr_results/854_Selective Attention Improves Transformer/images/ceef72a9ac83ccfd2c86c6321075e5181fa780a9f5dc48c159b40a2ba90c4282.jpg)

![d77b46e2bcb41f809237c5b163c495c07dfc7ad9338289a7b1c0094f3d5f827f.jpg](../iclr_results/854_Selective Attention Improves Transformer/images/d77b46e2bcb41f809237c5b163c495c07dfc7ad9338289a7b1c0094f3d5f827f.jpg)

![dde50b9996f462ca469e69e5f15d2a230e9d4a349a6d4363be54e47912a7e446.jpg](../iclr_results/854_Selective Attention Improves Transformer/images/dde50b9996f462ca469e69e5f15d2a230e9d4a349a6d4363be54e47912a7e446.jpg)

![fe7f2e182775beff3e063ca8cae11daf660f40fd4313064538f944b99569d15c.jpg](../iclr_results/854_Selective Attention Improves Transformer/images/fe7f2e182775beff3e063ca8cae11daf660f40fd4313064538f944b99569d15c.jpg)

![ff54a823f1e9837bc4e0da11001390b266495478f7dacc83b0e929124fea2832.jpg](../iclr_results/854_Selective Attention Improves Transformer/images/ff54a823f1e9837bc4e0da11001390b266495478f7dacc83b0e929124fea2832.jpg)

### Tables

![058b45550ac44dea9b2520731bf415ff9173cbe85b0483c4415cc3ee6aad5e3a.jpg](../iclr_results/854_Selective Attention Improves Transformer/tables/058b45550ac44dea9b2520731bf415ff9173cbe85b0483c4415cc3ee6aad5e3a.jpg)

![15160d7e6c7aede86ed3414b68c56c4b83dfda76adf0b49e4b1664295ccd791c.jpg](../iclr_results/854_Selective Attention Improves Transformer/tables/15160d7e6c7aede86ed3414b68c56c4b83dfda76adf0b49e4b1664295ccd791c.jpg)

![3566c7838f18ae9ddb550a633fcad7518aaf0f05bbaf58dc682afa5d58538341.jpg](../iclr_results/854_Selective Attention Improves Transformer/tables/3566c7838f18ae9ddb550a633fcad7518aaf0f05bbaf58dc682afa5d58538341.jpg)

![389e39efaa06c4930c0b028ca15883ef3048484aa75b6e677f6fe904a1267d90.jpg](../iclr_results/854_Selective Attention Improves Transformer/tables/389e39efaa06c4930c0b028ca15883ef3048484aa75b6e677f6fe904a1267d90.jpg)

![57782582c0d1ed29b2be342a1482c2f2d656279a6c2ec1c66982ee7f56b3add3.jpg](../iclr_results/854_Selective Attention Improves Transformer/tables/57782582c0d1ed29b2be342a1482c2f2d656279a6c2ec1c66982ee7f56b3add3.jpg)

![73541c4da0a8db3352ef4c3c7a988afcaa6afe8d2bc6d23a55187f1376814260.jpg](../iclr_results/854_Selective Attention Improves Transformer/tables/73541c4da0a8db3352ef4c3c7a988afcaa6afe8d2bc6d23a55187f1376814260.jpg)

![7723a84d48b1e4b34bcc70d8a7eb91025d03d6d0ad8011c5da9d7c0d8d26f034.jpg](../iclr_results/854_Selective Attention Improves Transformer/tables/7723a84d48b1e4b34bcc70d8a7eb91025d03d6d0ad8011c5da9d7c0d8d26f034.jpg)

![cd2517ef640c4aff9370a5272b022512978124dd30fd5c38b8200226537b8dfa.jpg](../iclr_results/854_Selective Attention Improves Transformer/tables/cd2517ef640c4aff9370a5272b022512978124dd30fd5c38b8200226537b8dfa.jpg)

## VideoShield: Regulating Diffusion-based Video Generation Models via Watermarking


### Images

![0476062a0d72fe595d2bc7ad0e2433e6d6361fc5b0d48c0b14b2d886b3dc3dae.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/0476062a0d72fe595d2bc7ad0e2433e6d6361fc5b0d48c0b14b2d886b3dc3dae.jpg)

![065ec4ac4b8b22d33f6c67dc7c289bd2baca431a07a7ac0e141b0e2300fe8aea.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/065ec4ac4b8b22d33f6c67dc7c289bd2baca431a07a7ac0e141b0e2300fe8aea.jpg)

![06760256ebee76d372c54115dfd4ed1d06c1fc386162a989313f7b27b616c192.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/06760256ebee76d372c54115dfd4ed1d06c1fc386162a989313f7b27b616c192.jpg)

![0a6557bafc540d714a6a0bab32a154f392e894f4f87b225adaed08aa40d4b56d.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/0a6557bafc540d714a6a0bab32a154f392e894f4f87b225adaed08aa40d4b56d.jpg)

![1d1aafdac29a5a1a8b27d1117a6c0d39003e992d36f48cc458299de4e76169a8.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/1d1aafdac29a5a1a8b27d1117a6c0d39003e992d36f48cc458299de4e76169a8.jpg)

![280113738a5055ba3a65796a0ccebc0e4b926269a052363c0b3e9245f758e260.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/280113738a5055ba3a65796a0ccebc0e4b926269a052363c0b3e9245f758e260.jpg)

![4872e96bc8c1e261cb41ddcbd455c16502f58f1d495da3daf0ecf48b891b9931.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/4872e96bc8c1e261cb41ddcbd455c16502f58f1d495da3daf0ecf48b891b9931.jpg)

![4ba7d330433a0c8c098fbaad566af92465296ee083ea14ae435b995fd107f1a7.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/4ba7d330433a0c8c098fbaad566af92465296ee083ea14ae435b995fd107f1a7.jpg)

![4dfc473777fa02ffdf4fbdd737379c9ff1251cf358803f7952c3b2cd64636a80.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/4dfc473777fa02ffdf4fbdd737379c9ff1251cf358803f7952c3b2cd64636a80.jpg)

![4e2dfac2e3b419208669cace6f939ad42f9dbf80aab382c717c63efc24d90e39.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/4e2dfac2e3b419208669cace6f939ad42f9dbf80aab382c717c63efc24d90e39.jpg)

![516353c0c6035bd851b041bc3fe3aa99726e49a881364a38c8816bbe64cb47e2.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/516353c0c6035bd851b041bc3fe3aa99726e49a881364a38c8816bbe64cb47e2.jpg)

![7f52ae8ada4e8459c8740747a075be07a00cc8df39a214e39a7a47baa0bb6387.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/7f52ae8ada4e8459c8740747a075be07a00cc8df39a214e39a7a47baa0bb6387.jpg)

![7fdb428ee64f560838f3773c65b76c40f1a8ceeca449926fe4ed25a241b3f777.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/7fdb428ee64f560838f3773c65b76c40f1a8ceeca449926fe4ed25a241b3f777.jpg)

![845e090abe130c3c50271a5fd154b9a9cd0c660534123329d75621a7a738b311.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/845e090abe130c3c50271a5fd154b9a9cd0c660534123329d75621a7a738b311.jpg)

![b900009b8e9f4c9ab7cde6c7ba5649dd90e3583ed2c4953a10e771f46bad9a6e.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/b900009b8e9f4c9ab7cde6c7ba5649dd90e3583ed2c4953a10e771f46bad9a6e.jpg)

![c127dcd45c9e22f015c63a9b5d4c16ec4fc1a71d537eb579922b76e04cf37b1c.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/c127dcd45c9e22f015c63a9b5d4c16ec4fc1a71d537eb579922b76e04cf37b1c.jpg)

![c18afddc80de02e92b64b47b9ed04b5086756b98788d5f8ded9aea3d7c7fde73.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/c18afddc80de02e92b64b47b9ed04b5086756b98788d5f8ded9aea3d7c7fde73.jpg)

![c2075524a42b8c52b5f53a0ba148b14976bc511205263986d7c99a85cdd3602f.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/c2075524a42b8c52b5f53a0ba148b14976bc511205263986d7c99a85cdd3602f.jpg)

![cb89c97f64c1dde6259002a89ab0c76725897044227562106f6ca18fb6e965b8.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/cb89c97f64c1dde6259002a89ab0c76725897044227562106f6ca18fb6e965b8.jpg)

![e53d91391570703178f733531973576f4b359b404d60e8fac390cb09127a2516.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/e53d91391570703178f733531973576f4b359b404d60e8fac390cb09127a2516.jpg)

![ec124e488624c0ee01e62283a3d9b73c7fe903e4e8313ec5b6cae7afd97d25ec.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/ec124e488624c0ee01e62283a3d9b73c7fe903e4e8313ec5b6cae7afd97d25ec.jpg)

![f350042a6ab04323a83e5c5273b7f667842dd1f8d43534aff3644a7d16c4d064.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/f350042a6ab04323a83e5c5273b7f667842dd1f8d43534aff3644a7d16c4d064.jpg)

![f970416ae9a894a7ed7bc8aa5f55e5f4564595a2c042bfea0ffa3dd062014de9.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/f970416ae9a894a7ed7bc8aa5f55e5f4564595a2c042bfea0ffa3dd062014de9.jpg)

![feb8103a9d4eac12a07ed7e9dee1b88817af62e090cdc8316ee96aa4a1b24305.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/images/feb8103a9d4eac12a07ed7e9dee1b88817af62e090cdc8316ee96aa4a1b24305.jpg)

### Tables

![055e3564c876b51c4523adc4c0ded9d9bdb1e60e8bee1180ea3b19222f648dee.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/tables/055e3564c876b51c4523adc4c0ded9d9bdb1e60e8bee1180ea3b19222f648dee.jpg)

![0ee962e5cc6541f099938ca88a3f91f85d1ed1b7f4e24add050a7f0397eedbe4.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/tables/0ee962e5cc6541f099938ca88a3f91f85d1ed1b7f4e24add050a7f0397eedbe4.jpg)

![28cab4086912d5d9076215dfe8edb882e8657b950891653b59223beae6ce399b.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/tables/28cab4086912d5d9076215dfe8edb882e8657b950891653b59223beae6ce399b.jpg)

![4078c887d1e7ce97660145b4e17cd8549cce4c50517e7ef42dd1bfa50eb3df24.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/tables/4078c887d1e7ce97660145b4e17cd8549cce4c50517e7ef42dd1bfa50eb3df24.jpg)

![491699767103b7c65c2a060fc7c5d9abf9b4560483de089e0be1c36b85f65b15.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/tables/491699767103b7c65c2a060fc7c5d9abf9b4560483de089e0be1c36b85f65b15.jpg)

![501fc091533afb5f7f1ebc94fc47b565e2b919c27f2464949705b09b522c7010.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/tables/501fc091533afb5f7f1ebc94fc47b565e2b919c27f2464949705b09b522c7010.jpg)

![6319cde31fa279d9d5f9d30affc868823c74b90784a48825f4461955c5c3cb4a.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/tables/6319cde31fa279d9d5f9d30affc868823c74b90784a48825f4461955c5c3cb4a.jpg)

![63802f85168f9995d5372b181d74ab28bda4d203efacbf4fe58e7d0ac670bbc6.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/tables/63802f85168f9995d5372b181d74ab28bda4d203efacbf4fe58e7d0ac670bbc6.jpg)

![6c41c22c50a4b2baf87d8643b9a3145dcaebc39105522cdbeef2ea3559c40d51.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/tables/6c41c22c50a4b2baf87d8643b9a3145dcaebc39105522cdbeef2ea3559c40d51.jpg)

![72e932007b16ea2fc0498124edd300779a109c283b6bb4c6946ccf19f546e665.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/tables/72e932007b16ea2fc0498124edd300779a109c283b6bb4c6946ccf19f546e665.jpg)

![80444f6b3722f6178f94afa451d091a5c2108edd729d7bc21fd3f2da6eac3dfa.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/tables/80444f6b3722f6178f94afa451d091a5c2108edd729d7bc21fd3f2da6eac3dfa.jpg)

![84ee628b80093a984a1ef2868b419163e359960d08af8bb8f9309999c56d4970.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/tables/84ee628b80093a984a1ef2868b419163e359960d08af8bb8f9309999c56d4970.jpg)

![87d2c7ce00023ecbcfda9d56d990cccfced7bc58379c02b137ed435fec67b1f7.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/tables/87d2c7ce00023ecbcfda9d56d990cccfced7bc58379c02b137ed435fec67b1f7.jpg)

![8eba5313f32368d0fb1640231950a9992c984dbe1fa8179b6bb3fb16447acac3.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/tables/8eba5313f32368d0fb1640231950a9992c984dbe1fa8179b6bb3fb16447acac3.jpg)

![ac9ed2d2c33fdbc649ea980078e592894dd4fd47978c91665bf4781b21e65dc0.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/tables/ac9ed2d2c33fdbc649ea980078e592894dd4fd47978c91665bf4781b21e65dc0.jpg)

![c12631bc969f42ac488129c2cc0b13848e748617dbf937d79522ed9ad02147e7.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/tables/c12631bc969f42ac488129c2cc0b13848e748617dbf937d79522ed9ad02147e7.jpg)

![cf1157199d19e27f7511263e79e95d5cf9f385cf1b43788b1c95cf5300291ff0.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/tables/cf1157199d19e27f7511263e79e95d5cf9f385cf1b43788b1c95cf5300291ff0.jpg)

![d819f2bb109032ed4c1d57943a3500e732070350aad98ac6f3e31c21dc13deb4.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/tables/d819f2bb109032ed4c1d57943a3500e732070350aad98ac6f3e31c21dc13deb4.jpg)

![ff1116e33909da1745b006c36961d9a9fb1e2f56332b59869ade1c6fd1033263.jpg](../iclr_results/855_VideoShield_ Regulating Diffusion-based Video Generation Models via Watermarking/tables/ff1116e33909da1745b006c36961d9a9fb1e2f56332b59869ade1c6fd1033263.jpg)

## Reconciling Model Multiplicity for Downstream Decision Making


### Images

![1ab7a381ea6ca268aff25939b339933526f36b825a49836d6cddbc71cf5070a8.jpg](../iclr_results/856_Reconciling Model Multiplicity for Downstream Decision Making/images/1ab7a381ea6ca268aff25939b339933526f36b825a49836d6cddbc71cf5070a8.jpg)

![24aab181969b714ad66168f6e094d4be6271fa7977c46d3f2562e032b9b33b66.jpg](../iclr_results/856_Reconciling Model Multiplicity for Downstream Decision Making/images/24aab181969b714ad66168f6e094d4be6271fa7977c46d3f2562e032b9b33b66.jpg)

![26299615a39adc647e0080ef40279c016933f48877964c32f1245b622779fd15.jpg](../iclr_results/856_Reconciling Model Multiplicity for Downstream Decision Making/images/26299615a39adc647e0080ef40279c016933f48877964c32f1245b622779fd15.jpg)

![32419678a6d148c979e1c1aebd99b14c13077edee4a5978c337c912076bb33a9.jpg](../iclr_results/856_Reconciling Model Multiplicity for Downstream Decision Making/images/32419678a6d148c979e1c1aebd99b14c13077edee4a5978c337c912076bb33a9.jpg)

![4ba9dcf9d4a254294735b5ca3bc60643b3cd5467264cf7ce14491607cfa59448.jpg](../iclr_results/856_Reconciling Model Multiplicity for Downstream Decision Making/images/4ba9dcf9d4a254294735b5ca3bc60643b3cd5467264cf7ce14491607cfa59448.jpg)

![a2d98b2b19bd2fdba62ce24f8f1dbb54cd72ee014ae51373c566a777bce93600.jpg](../iclr_results/856_Reconciling Model Multiplicity for Downstream Decision Making/images/a2d98b2b19bd2fdba62ce24f8f1dbb54cd72ee014ae51373c566a777bce93600.jpg)

![ae107abaa6ad00f5cd259bb1bd6cef87d37f0f10231286811776d156bd51aa62.jpg](../iclr_results/856_Reconciling Model Multiplicity for Downstream Decision Making/images/ae107abaa6ad00f5cd259bb1bd6cef87d37f0f10231286811776d156bd51aa62.jpg)

![b887bba6e53748abeeaa3fc9a6c45da49bf52f8cfe48dc11d3f214de03c846e6.jpg](../iclr_results/856_Reconciling Model Multiplicity for Downstream Decision Making/images/b887bba6e53748abeeaa3fc9a6c45da49bf52f8cfe48dc11d3f214de03c846e6.jpg)

## Unbounded: A Generative Infinite Game of Character Life Simulation


### Images

![0548341f628d17f1b13e78add31d6d83257275b440449f461514acb0d62c4218.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/0548341f628d17f1b13e78add31d6d83257275b440449f461514acb0d62c4218.jpg)

![056ad9d0aa22a2a00f15c9e6bfb3ae9f196308a17afe8aa2d573c454f5a494ab.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/056ad9d0aa22a2a00f15c9e6bfb3ae9f196308a17afe8aa2d573c454f5a494ab.jpg)

![060139e8d099d969ec4eaf2cd89dd32cf3bfb8bb52380d2692d98a6834a163a4.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/060139e8d099d969ec4eaf2cd89dd32cf3bfb8bb52380d2692d98a6834a163a4.jpg)

![0acb1e0c7eb0b50681b53c47e40dc1b01a4fbb83d88e0401f19f7d37c5a6af51.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/0acb1e0c7eb0b50681b53c47e40dc1b01a4fbb83d88e0401f19f7d37c5a6af51.jpg)

![0ff984e00809e8120f82ebd8f3253e0859c66cf04ce204ff6d25a8080b56900e.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/0ff984e00809e8120f82ebd8f3253e0859c66cf04ce204ff6d25a8080b56900e.jpg)

![16a43c9e1c66dfe4a27c6d36867522d9d605e5f91a5acf1f3dc6cf94919d1313.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/16a43c9e1c66dfe4a27c6d36867522d9d605e5f91a5acf1f3dc6cf94919d1313.jpg)

![1b4703ec4e3e92e6d1624a725fdfb18acb63bc435b692340cacc728081768ab5.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/1b4703ec4e3e92e6d1624a725fdfb18acb63bc435b692340cacc728081768ab5.jpg)

![1fc0c95e481f65453663baf2c1b14e8369745d6188bb112372ac7326091eda31.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/1fc0c95e481f65453663baf2c1b14e8369745d6188bb112372ac7326091eda31.jpg)

![23713e8517093a7966f8e50c147082e056b007191b017fa25d913ab083466814.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/23713e8517093a7966f8e50c147082e056b007191b017fa25d913ab083466814.jpg)

![2ccec938000e15da7ea3b4d1c244d83253199322186a6b840b0820cdcd934e1d.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/2ccec938000e15da7ea3b4d1c244d83253199322186a6b840b0820cdcd934e1d.jpg)

![2e5fa941c8cc113d5d1cccdc89c4541680900473237249474a8d74801cecaa3c.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/2e5fa941c8cc113d5d1cccdc89c4541680900473237249474a8d74801cecaa3c.jpg)

![300b86983dc4c4b2725de3350f1702f24d4b7ef179d6e20137e616726a9ba77b.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/300b86983dc4c4b2725de3350f1702f24d4b7ef179d6e20137e616726a9ba77b.jpg)

![42b09daa293339012c3333d52aa47c4a0c26a36e0b6a309c9f1a39ac331aac87.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/42b09daa293339012c3333d52aa47c4a0c26a36e0b6a309c9f1a39ac331aac87.jpg)

![49349f9136436419a9039003423e6ac1456688c0a5bcbcf5814f8d4df1d1de34.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/49349f9136436419a9039003423e6ac1456688c0a5bcbcf5814f8d4df1d1de34.jpg)

![57ac10ac0f914d0aeb4fce2203b7af8e46fcdc0df70d41c09b33cb960ee027c6.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/57ac10ac0f914d0aeb4fce2203b7af8e46fcdc0df70d41c09b33cb960ee027c6.jpg)

![636ac2ace214c3e59d8935f57de0ab36682b24e06d3cf086c26fbd1308679972.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/636ac2ace214c3e59d8935f57de0ab36682b24e06d3cf086c26fbd1308679972.jpg)

![6380f551b89dd12057c965022e079802f6811ed228c8708f9d414ed32b8d08a1.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/6380f551b89dd12057c965022e079802f6811ed228c8708f9d414ed32b8d08a1.jpg)

![6b7a8af8fb3e973df8748135aa0741e111bcd614a6ce65ac48cdac93f2fc35f2.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/6b7a8af8fb3e973df8748135aa0741e111bcd614a6ce65ac48cdac93f2fc35f2.jpg)

![703c4ba5d8ddd2891d161a357051a0c1c43c5f4edd1da34ba49d215e878443cc.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/703c4ba5d8ddd2891d161a357051a0c1c43c5f4edd1da34ba49d215e878443cc.jpg)

![71edd0e7a7fdb0f753955b6f7c80b2036c12904c9967fcfd911897d1a82c2792.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/71edd0e7a7fdb0f753955b6f7c80b2036c12904c9967fcfd911897d1a82c2792.jpg)

![75ff323500b6451fb57748b693c9216b8d26bfee525aeb016a2efa7f84bd1f03.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/75ff323500b6451fb57748b693c9216b8d26bfee525aeb016a2efa7f84bd1f03.jpg)

![76a825cee331f38f9ac4f384b605406ef8f4916792f5865e6a99287a54380c41.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/76a825cee331f38f9ac4f384b605406ef8f4916792f5865e6a99287a54380c41.jpg)

![78c7d8c79223765cc2d0d7c9af92011b8cff6979a3ad1f0a138c91d384848685.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/78c7d8c79223765cc2d0d7c9af92011b8cff6979a3ad1f0a138c91d384848685.jpg)

![8353630282fe90139cf95c7a2666e3f7211141bb3cfe4b4796dc62526479e860.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/8353630282fe90139cf95c7a2666e3f7211141bb3cfe4b4796dc62526479e860.jpg)

![889c0f1d5d997a8d968c6185348472efc11803164db4d04b9626c7acba6a22b8.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/889c0f1d5d997a8d968c6185348472efc11803164db4d04b9626c7acba6a22b8.jpg)

![8a829dbd68fac25b0600e5e2f0e4bbc461931ef3e9ca4c7a3a616217588345dc.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/8a829dbd68fac25b0600e5e2f0e4bbc461931ef3e9ca4c7a3a616217588345dc.jpg)

![95aea15ce572169ee3084470a3d06dc21d18deb11440022d5828cc663fa8da8d.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/95aea15ce572169ee3084470a3d06dc21d18deb11440022d5828cc663fa8da8d.jpg)

![95eca4eeac6d9e24cf15c61e0527bbb76056306bff73fdc7af2b6b04515ea018.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/95eca4eeac6d9e24cf15c61e0527bbb76056306bff73fdc7af2b6b04515ea018.jpg)

![ab0cb13ded4f87ecf8b48593d55d75f7243a9b05e0b420f1d4c1d6a0ff2d1f31.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/ab0cb13ded4f87ecf8b48593d55d75f7243a9b05e0b420f1d4c1d6a0ff2d1f31.jpg)

![ab1d2436cfeb3d316cd4d754865fd2343d8ca8502b2a6ed25cda88bd4aef69df.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/ab1d2436cfeb3d316cd4d754865fd2343d8ca8502b2a6ed25cda88bd4aef69df.jpg)

![c10c75aab38e8e744f656b8bd7fb8e7f3c39b141f4b02c736eb8dbc3959bffca.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/c10c75aab38e8e744f656b8bd7fb8e7f3c39b141f4b02c736eb8dbc3959bffca.jpg)

![c1aeeb2882c0f2fa1fb434397313f0c64b771e9cac3e4e4933a72c79d443a83a.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/c1aeeb2882c0f2fa1fb434397313f0c64b771e9cac3e4e4933a72c79d443a83a.jpg)

![c25e429c6b4bb05d072d5a04cfa046837d755df982ee3fe3912d5e8a5d46fc75.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/c25e429c6b4bb05d072d5a04cfa046837d755df982ee3fe3912d5e8a5d46fc75.jpg)

![c8aea74b27d527a683dcf08df5ca4e0c074fb34dc6b941de9baff5a5ff15321c.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/c8aea74b27d527a683dcf08df5ca4e0c074fb34dc6b941de9baff5a5ff15321c.jpg)

![ccf31f253aaf7fe1c0a86cb6360d9186f92082d6c4ea2b20e9d6305ccc9938be.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/ccf31f253aaf7fe1c0a86cb6360d9186f92082d6c4ea2b20e9d6305ccc9938be.jpg)

![cfd0f5b5a9bf837f9ff872d8f6e5c7370c40e78ee75605dd021201690da6d563.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/cfd0f5b5a9bf837f9ff872d8f6e5c7370c40e78ee75605dd021201690da6d563.jpg)

![d49a6ff6e1850acefcc00055b53ea9ba34de46932fb55d6ce45ee8c4c97763f4.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/d49a6ff6e1850acefcc00055b53ea9ba34de46932fb55d6ce45ee8c4c97763f4.jpg)

![e7f6e136aff8c8fa55db907330cbc71c36d3bd7e62c6b491abc6013db6fbe037.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/e7f6e136aff8c8fa55db907330cbc71c36d3bd7e62c6b491abc6013db6fbe037.jpg)

![ec6ba7627f4f316a123ed2f0099a9590fa543d74cae9800964457e69f5eed7ea.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/ec6ba7627f4f316a123ed2f0099a9590fa543d74cae9800964457e69f5eed7ea.jpg)

![ed958372cc608ed44c0fa551695a20895e9fa5bcf0f27a7bc0f1624fc68b3c8b.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/ed958372cc608ed44c0fa551695a20895e9fa5bcf0f27a7bc0f1624fc68b3c8b.jpg)

![f3972d5d1375452b4de2e997f0a7d4cd8510ec1bead9c464be2b783906d7d42c.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/images/f3972d5d1375452b4de2e997f0a7d4cd8510ec1bead9c464be2b783906d7d42c.jpg)

### Tables

![60922ee2dcaea53f54b16b73ab795ba4724ceb41fb5e83e097516a9e15b3b87c.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/tables/60922ee2dcaea53f54b16b73ab795ba4724ceb41fb5e83e097516a9e15b3b87c.jpg)

![67a9d460af9604e1a5de6cdf3a11a99ddaf47c82c14a923535d6aba9758bbbcd.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/tables/67a9d460af9604e1a5de6cdf3a11a99ddaf47c82c14a923535d6aba9758bbbcd.jpg)

![6bf13220850526034dff75ae3374efc5cac78921fbf6c305e09e87e8b6696229.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/tables/6bf13220850526034dff75ae3374efc5cac78921fbf6c305e09e87e8b6696229.jpg)

![6dace415fbc31483174cdd6f6ebb807a5fa21179f896e1671995ea758fe0b3ec.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/tables/6dace415fbc31483174cdd6f6ebb807a5fa21179f896e1671995ea758fe0b3ec.jpg)

![ba1b1b179632ff3e41808a33c56162a176be0c181414b9a3b452a12c8bc6d966.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/tables/ba1b1b179632ff3e41808a33c56162a176be0c181414b9a3b452a12c8bc6d966.jpg)

![f944ab7809095e13c1c2a8b64731d3cdc8a5f6c976848b667a5a324ae703d483.jpg](../iclr_results/857_Unbounded_ A Generative Infinite Game of Character Life Simulation/tables/f944ab7809095e13c1c2a8b64731d3cdc8a5f6c976848b667a5a324ae703d483.jpg)

## Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting


### Images

![497516efdaeda8370c48e5223f35479f395dfdeb134fc30f6a80fd6da1d5e47e.jpg](../iclr_results/858_Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting/images/497516efdaeda8370c48e5223f35479f395dfdeb134fc30f6a80fd6da1d5e47e.jpg)

![5fe7bce6486d3646d2ba869f528a684a9d9e8e374855a0ea794d2d4634c1f837.jpg](../iclr_results/858_Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting/images/5fe7bce6486d3646d2ba869f528a684a9d9e8e374855a0ea794d2d4634c1f837.jpg)

![6d2966c93549fe618b43592c8eb5a92ee8e160413fca4ffaab331012cc0bbc92.jpg](../iclr_results/858_Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting/images/6d2966c93549fe618b43592c8eb5a92ee8e160413fca4ffaab331012cc0bbc92.jpg)

![74d2a3c65274423c6920805c6a91beca89fdfae8bbf563c4781bcbd5a0800460.jpg](../iclr_results/858_Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting/images/74d2a3c65274423c6920805c6a91beca89fdfae8bbf563c4781bcbd5a0800460.jpg)

![a50727c1c432d7fdea06cda8417f44db584bc849a37a639775e146563d0bb16d.jpg](../iclr_results/858_Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting/images/a50727c1c432d7fdea06cda8417f44db584bc849a37a639775e146563d0bb16d.jpg)

![c5f90970d7b97d6205c91661d0c982d54f6ad133ee6ba43c5596daee015a17bd.jpg](../iclr_results/858_Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting/images/c5f90970d7b97d6205c91661d0c982d54f6ad133ee6ba43c5596daee015a17bd.jpg)

![cc930e562bf61126376a0fb00c22d06499fa2bdc6287034573ad7b6b0527eb1a.jpg](../iclr_results/858_Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting/images/cc930e562bf61126376a0fb00c22d06499fa2bdc6287034573ad7b6b0527eb1a.jpg)

### Tables

![04373c61dc6dabfd9b48f8922ea5d77c9eaf2d5a7ef8cf7e1e10f3f5cd1983f1.jpg](../iclr_results/858_Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting/tables/04373c61dc6dabfd9b48f8922ea5d77c9eaf2d5a7ef8cf7e1e10f3f5cd1983f1.jpg)

![047d2de831dc04f5f0e0ea8a6b291584fdb3ec7a6adbc675ca052cd0f2eaf338.jpg](../iclr_results/858_Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting/tables/047d2de831dc04f5f0e0ea8a6b291584fdb3ec7a6adbc675ca052cd0f2eaf338.jpg)

![1778f945e2d7ec7a3a0c34aeae200a215fd1fa7dad7f649ed8d0632e2829f7e5.jpg](../iclr_results/858_Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting/tables/1778f945e2d7ec7a3a0c34aeae200a215fd1fa7dad7f649ed8d0632e2829f7e5.jpg)

![1874f56c3cb1cd7656e6e400acf16dd7b9e6b52e3588324f90ad5d5691e5df5f.jpg](../iclr_results/858_Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting/tables/1874f56c3cb1cd7656e6e400acf16dd7b9e6b52e3588324f90ad5d5691e5df5f.jpg)

![20350f92966fb86ff70d062c0a1028cf879af1aab0464be0fd278e8b6de3a17e.jpg](../iclr_results/858_Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting/tables/20350f92966fb86ff70d062c0a1028cf879af1aab0464be0fd278e8b6de3a17e.jpg)

![435ea5a8838f107373bf8f83e10cb5c12e978dbe18ac38d5f1c72843d2842d47.jpg](../iclr_results/858_Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting/tables/435ea5a8838f107373bf8f83e10cb5c12e978dbe18ac38d5f1c72843d2842d47.jpg)

![4f48a8bef6cbb1c12d5bcc9d1fd4e33c33dbd436044a0214df1f410958048de4.jpg](../iclr_results/858_Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting/tables/4f48a8bef6cbb1c12d5bcc9d1fd4e33c33dbd436044a0214df1f410958048de4.jpg)

![539a6cebb2c4fec6ab8c394e9c0ea5230278920cd1e0932944b0f243bb432b53.jpg](../iclr_results/858_Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting/tables/539a6cebb2c4fec6ab8c394e9c0ea5230278920cd1e0932944b0f243bb432b53.jpg)

![725fbcc41a7e59de411130bfbacb57d65bf38b960f2b012a375fa6b9378bab56.jpg](../iclr_results/858_Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting/tables/725fbcc41a7e59de411130bfbacb57d65bf38b960f2b012a375fa6b9378bab56.jpg)

![7b8ff70873b4ec2ab5bae4b41bb02c1365e6ab009375473ea9b0a0a5568a04a7.jpg](../iclr_results/858_Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting/tables/7b8ff70873b4ec2ab5bae4b41bb02c1365e6ab009375473ea9b0a0a5568a04a7.jpg)

![82bcc249166d7e0116279f08738a8c24f71c89b21cd29ffa50789da4818a4287.jpg](../iclr_results/858_Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting/tables/82bcc249166d7e0116279f08738a8c24f71c89b21cd29ffa50789da4818a4287.jpg)

![85af122cacff8b4bb927e20edd8bffe84a6283d4727c8cd58674024736d529cb.jpg](../iclr_results/858_Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting/tables/85af122cacff8b4bb927e20edd8bffe84a6283d4727c8cd58674024736d529cb.jpg)

![94eae8686fcb6f659da0bf138c7922a19ed5a095f091a3e15a87be447fedbd27.jpg](../iclr_results/858_Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting/tables/94eae8686fcb6f659da0bf138c7922a19ed5a095f091a3e15a87be447fedbd27.jpg)

![ab950da0c44ed54013a36c1b47e3093ce32c9d02ff6104b5b0bb41adc247be03.jpg](../iclr_results/858_Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting/tables/ab950da0c44ed54013a36c1b47e3093ce32c9d02ff6104b5b0bb41adc247be03.jpg)

![c3b630632cd4ff8462e31395d038aa24025e80c3b35b87817cf19a4d4231fe06.jpg](../iclr_results/858_Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting/tables/c3b630632cd4ff8462e31395d038aa24025e80c3b35b87817cf19a4d4231fe06.jpg)

![f4c48f22006a7eceb34ab679c9e63cb2e670dd9ba4d5d18c34274b5cad4d6038.jpg](../iclr_results/858_Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting/tables/f4c48f22006a7eceb34ab679c9e63cb2e670dd9ba4d5d18c34274b5cad4d6038.jpg)

## FlowDec: A flow-based full-band general audio codec with high perceptual quality


### Images

![0393c8b8faa9e9487b3df4ba57e5012b0aba3a18a7486414b93e658e0d37ba34.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/images/0393c8b8faa9e9487b3df4ba57e5012b0aba3a18a7486414b93e658e0d37ba34.jpg)

![0768aa0e8b5c259ee73b3e18c32bab03008b10167e00b865746e137b272a9200.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/images/0768aa0e8b5c259ee73b3e18c32bab03008b10167e00b865746e137b272a9200.jpg)

![1121223a034cfee613b34b3cf1de85fb72e7ffbc6de0a755d330db6566f9a597.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/images/1121223a034cfee613b34b3cf1de85fb72e7ffbc6de0a755d330db6566f9a597.jpg)

![2101c11490e81d88faa9318fdf897aeaf49053072308803b57a90263908e3038.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/images/2101c11490e81d88faa9318fdf897aeaf49053072308803b57a90263908e3038.jpg)

![2271cd8ddd006cdfb298f6632b10a5d7bc651e6e55145171a5dfd5ece80566d5.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/images/2271cd8ddd006cdfb298f6632b10a5d7bc651e6e55145171a5dfd5ece80566d5.jpg)

![316289f446d7b9da6c60ef9f56769e04eb7166337e2b92c3c59d7d31f05fa4ae.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/images/316289f446d7b9da6c60ef9f56769e04eb7166337e2b92c3c59d7d31f05fa4ae.jpg)

![5caccd7c1f92154905ba902f19cc4c561053dba6007c742a624ded34c668c9c1.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/images/5caccd7c1f92154905ba902f19cc4c561053dba6007c742a624ded34c668c9c1.jpg)

![60785d5cc21c615a1f5376bee0abe4b57046a77c35bf6e337ebcdd3c044f886c.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/images/60785d5cc21c615a1f5376bee0abe4b57046a77c35bf6e337ebcdd3c044f886c.jpg)

![802d8fd43a23f5dfca63649c86404420c8fd6c834a8a68668bf2918019a5c906.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/images/802d8fd43a23f5dfca63649c86404420c8fd6c834a8a68668bf2918019a5c906.jpg)

![9b5dcd201e265bd862681c5136c586fa4ee244bf352fe534f96e729e3aaa1a2e.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/images/9b5dcd201e265bd862681c5136c586fa4ee244bf352fe534f96e729e3aaa1a2e.jpg)

![aa1388e6a1be90e5280e767898ef611946cb245c8e20857c5be12da233b89185.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/images/aa1388e6a1be90e5280e767898ef611946cb245c8e20857c5be12da233b89185.jpg)

![b71a57e5f8171d83ebcd2a903fc7d70c803a7b632fe3ed20b8ac1087ea4d4882.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/images/b71a57e5f8171d83ebcd2a903fc7d70c803a7b632fe3ed20b8ac1087ea4d4882.jpg)

![be76836da3da721dbc63c6a02b197dcdef9ce2df66452985b4abe288ed964543.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/images/be76836da3da721dbc63c6a02b197dcdef9ce2df66452985b4abe288ed964543.jpg)

![cddb601bfebafc39b6cb36e22a086ca9e45984315b8f000e23755bdb1ff29fa2.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/images/cddb601bfebafc39b6cb36e22a086ca9e45984315b8f000e23755bdb1ff29fa2.jpg)

![cfcdc4b4766ee224528c9cf3e60bd44e2499895a922410193d9b96c6d51a58b5.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/images/cfcdc4b4766ee224528c9cf3e60bd44e2499895a922410193d9b96c6d51a58b5.jpg)

![e084c6c1e58d89f2c03c7b608020b0a2a3a9d104d18f45e6b866d0ed26fecc41.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/images/e084c6c1e58d89f2c03c7b608020b0a2a3a9d104d18f45e6b866d0ed26fecc41.jpg)

![e93994cd2f23def4833023c9228dc79828da115b9dbcf8a0f1e4418d2df7339f.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/images/e93994cd2f23def4833023c9228dc79828da115b9dbcf8a0f1e4418d2df7339f.jpg)

![ea50e0aa7fe7370bb999d9a094489194ddcbce89b33425d6c15e8445bba61acb.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/images/ea50e0aa7fe7370bb999d9a094489194ddcbce89b33425d6c15e8445bba61acb.jpg)

### Tables

![001e203e6af614f318fdbac949fc2d0e108586761f41702282b97d3063d8b6ed.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/tables/001e203e6af614f318fdbac949fc2d0e108586761f41702282b97d3063d8b6ed.jpg)

![0284341201be05d199ab69c700b12a5244c7e947a62c0c2807276f9c53d2b8e1.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/tables/0284341201be05d199ab69c700b12a5244c7e947a62c0c2807276f9c53d2b8e1.jpg)

![07d54df6630c5d2483af3175a8c65d54fb3053d98c54335146207a8bbf4e4d89.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/tables/07d54df6630c5d2483af3175a8c65d54fb3053d98c54335146207a8bbf4e4d89.jpg)

![0e4a9a5d782b6ef236c3c29a3b9cd25b3a15d7d37094bddf6b1823df34be6261.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/tables/0e4a9a5d782b6ef236c3c29a3b9cd25b3a15d7d37094bddf6b1823df34be6261.jpg)

![962078f608e08addb76f5667858124b7658ff9039b518d42a3c926a1506d662b.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/tables/962078f608e08addb76f5667858124b7658ff9039b518d42a3c926a1506d662b.jpg)

![a6e0e177cde205d8f710d8aa892ae7ed6f272746ecbc68aa28adbdf57a6671a7.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/tables/a6e0e177cde205d8f710d8aa892ae7ed6f272746ecbc68aa28adbdf57a6671a7.jpg)

![c4f3b0110ad4007100cf1a2fa74b0fb3b08b18ba2a1ebbf52e8aa84fe50fc5a5.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/tables/c4f3b0110ad4007100cf1a2fa74b0fb3b08b18ba2a1ebbf52e8aa84fe50fc5a5.jpg)

![de2a948e291b1e540a6dbcea0fe91972c00afc2c3a1186120537117ac8935deb.jpg](../iclr_results/859_FlowDec_ A flow-based full-band general audio codec with high perceptual quality/tables/de2a948e291b1e540a6dbcea0fe91972c00afc2c3a1186120537117ac8935deb.jpg)

## IDArb: Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations


### Images

![15f877415cc9aa4b3120ebf42af78cb3f59324be9869c5823b393b4174381ec0.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/images/15f877415cc9aa4b3120ebf42af78cb3f59324be9869c5823b393b4174381ec0.jpg)

![1d628e3ab8f7be84a56113fa45b6dc4a902c466149658e7f9c1a2796c5238aef.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/images/1d628e3ab8f7be84a56113fa45b6dc4a902c466149658e7f9c1a2796c5238aef.jpg)

![244dfd0681a6a0151a6e9b5f20381f28db47682ee1de8c68d81736bc21d688da.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/images/244dfd0681a6a0151a6e9b5f20381f28db47682ee1de8c68d81736bc21d688da.jpg)

![29306637e9b6850f79fbe4ec90180232c07d02a34cac181095025171e0534fc4.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/images/29306637e9b6850f79fbe4ec90180232c07d02a34cac181095025171e0534fc4.jpg)

![2b7fd6cea01af51c0f18b90edcc01df7a40459e20a0967e514886a9d57b4bebc.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/images/2b7fd6cea01af51c0f18b90edcc01df7a40459e20a0967e514886a9d57b4bebc.jpg)

![2bac0a1a8b6cb40fb9fa2af998b340155067889e98cb2f7e9a583128cdd4e395.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/images/2bac0a1a8b6cb40fb9fa2af998b340155067889e98cb2f7e9a583128cdd4e395.jpg)

![2d989b8467d2c6b3d2b5731a12d5abdecc6d0f68ea69dcde9a2fc0fd3854456b.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/images/2d989b8467d2c6b3d2b5731a12d5abdecc6d0f68ea69dcde9a2fc0fd3854456b.jpg)

![5314dff527cb572db517d7d3f6d3f00c3253257bcb2b17182012aa703512dbd2.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/images/5314dff527cb572db517d7d3f6d3f00c3253257bcb2b17182012aa703512dbd2.jpg)

![5e324f45f81d5f62bcc7cb28c3d4c95cea4ba15705595d5f6d2c3ab1fd0a6807.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/images/5e324f45f81d5f62bcc7cb28c3d4c95cea4ba15705595d5f6d2c3ab1fd0a6807.jpg)

![660dc942b024068f345d86662f0d6ab4bc0b383fb56c5722b4850a48239824cf.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/images/660dc942b024068f345d86662f0d6ab4bc0b383fb56c5722b4850a48239824cf.jpg)

![6bc26bb97ac974a7814e167eb6368ec0299341b10e6aea2428dede7ecf114430.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/images/6bc26bb97ac974a7814e167eb6368ec0299341b10e6aea2428dede7ecf114430.jpg)

![878406c6510190e1911d33ee7b8f4b2011efffea6abc661642fb9c60da1c284a.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/images/878406c6510190e1911d33ee7b8f4b2011efffea6abc661642fb9c60da1c284a.jpg)

![9f5a983d49d56353f6409b91b583ad078ff1a4b5a5bf9285e541a36f884b86a7.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/images/9f5a983d49d56353f6409b91b583ad078ff1a4b5a5bf9285e541a36f884b86a7.jpg)

![a2b07e9945c7958e3f94f8e15f8b35d38e07864219c335a0061e01cd293316d1.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/images/a2b07e9945c7958e3f94f8e15f8b35d38e07864219c335a0061e01cd293316d1.jpg)

![a59083d05c1ce8667ef194e991f0f10b4e0becb0ae469f5baf6c11cbebecf11e.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/images/a59083d05c1ce8667ef194e991f0f10b4e0becb0ae469f5baf6c11cbebecf11e.jpg)

![a9817cb7a9347607198543a4efc4338673c4c0f1e0f20e15f06487ea981ff2bc.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/images/a9817cb7a9347607198543a4efc4338673c4c0f1e0f20e15f06487ea981ff2bc.jpg)

![d22d32f68763c1f6d36b4d32b0d27c46a3f6bc3e53c67c08344845c7da48777c.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/images/d22d32f68763c1f6d36b4d32b0d27c46a3f6bc3e53c67c08344845c7da48777c.jpg)

![d7f3de90f7803bc40374ef20126fb98f2887b84e6306182a8d702fb3a880925a.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/images/d7f3de90f7803bc40374ef20126fb98f2887b84e6306182a8d702fb3a880925a.jpg)

![d8a34d7cb1faab8ca78f1c5b3e0bd421c718fd1c0515276146d0ca38c9747696.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/images/d8a34d7cb1faab8ca78f1c5b3e0bd421c718fd1c0515276146d0ca38c9747696.jpg)

![e373bd48615cb496275f94d7c76dc376e8ec2b704e51f39ee3323e1cfc5f104a.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/images/e373bd48615cb496275f94d7c76dc376e8ec2b704e51f39ee3323e1cfc5f104a.jpg)

![e3c17206cb43f437e25606554d232273b074d70fe5bb9999ab91d671a8bd2a60.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/images/e3c17206cb43f437e25606554d232273b074d70fe5bb9999ab91d671a8bd2a60.jpg)

![e9bb115ebaddc639adc75ff80c9480bedf4dd001c65127a40badc6b8ca69dafe.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/images/e9bb115ebaddc639adc75ff80c9480bedf4dd001c65127a40badc6b8ca69dafe.jpg)

### Tables

![19b4ffe7a790d50f2696c22e915bc9a45d86e6c037b6b3312c21a9373e26b2fc.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/tables/19b4ffe7a790d50f2696c22e915bc9a45d86e6c037b6b3312c21a9373e26b2fc.jpg)

![1cac8bebf1f2ea9587d78209a348a7c571b9b194b44e9247c0d318b5d8e5034e.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/tables/1cac8bebf1f2ea9587d78209a348a7c571b9b194b44e9247c0d318b5d8e5034e.jpg)

![226557432352f5cfb12603a6618530db73ab8607d7fb7f80bb1316e90ce89d26.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/tables/226557432352f5cfb12603a6618530db73ab8607d7fb7f80bb1316e90ce89d26.jpg)

![2cbfd0a0d24efc0c3b6ff93c59ed474a6d54d4f79b0d2d53a0093862f5d54704.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/tables/2cbfd0a0d24efc0c3b6ff93c59ed474a6d54d4f79b0d2d53a0093862f5d54704.jpg)

![8a15cd6c0ac5e363d6f03347ce8e0486c4dd7f733e21a9e0a80104b43fc185f8.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/tables/8a15cd6c0ac5e363d6f03347ce8e0486c4dd7f733e21a9e0a80104b43fc185f8.jpg)

![a7cc37ae5d2b355a212f0a1704549d33fea15d0aea8e7f783fac253788b12fee.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/tables/a7cc37ae5d2b355a212f0a1704549d33fea15d0aea8e7f783fac253788b12fee.jpg)

![a84d7d5570051ee89650211029c3d9aa6ac22ef66b89dabe943dd61bfd1010c1.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/tables/a84d7d5570051ee89650211029c3d9aa6ac22ef66b89dabe943dd61bfd1010c1.jpg)

![f85a8475f8d36994b51529728502240af95aaf95d07380139901facbe609a0da.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/tables/f85a8475f8d36994b51529728502240af95aaf95d07380139901facbe609a0da.jpg)

![fb70c863f31715bdbb7e38444055b227543d8bf66525d1d945fccf6dd2811d80.jpg](../iclr_results/860_IDArb_ Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations/tables/fb70c863f31715bdbb7e38444055b227543d8bf66525d1d945fccf6dd2811d80.jpg)

## Towards Certification of Uncertainty Calibration under Adversarial Attacks


### Images

![00469da822437795f37e4157697bc593427f8a5ef61f35dc8f2a10f1b7cd83d3.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/images/00469da822437795f37e4157697bc593427f8a5ef61f35dc8f2a10f1b7cd83d3.jpg)

![0ebaec38ed2c9fd978e57ef92b90d83e4cde77bd861be40c32743fa2699ec5f1.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/images/0ebaec38ed2c9fd978e57ef92b90d83e4cde77bd861be40c32743fa2699ec5f1.jpg)

![1de941d6856b6d5518e74fd4c86a4847d3c894c92f1cb3665c386a83c2130ab0.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/images/1de941d6856b6d5518e74fd4c86a4847d3c894c92f1cb3665c386a83c2130ab0.jpg)

![2fafca57bfddb2c8c35d5c87a812f5416a950db9da6fdc4fa604eb860eb1dcd7.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/images/2fafca57bfddb2c8c35d5c87a812f5416a950db9da6fdc4fa604eb860eb1dcd7.jpg)

![36f422f55443a4202e380508b8f6d0ac4762eba9be82608a2a88c52fb335df27.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/images/36f422f55443a4202e380508b8f6d0ac4762eba9be82608a2a88c52fb335df27.jpg)

![59f1e0f262ea821d5c73e30df5cb4e8f17b17ecc18543c036615df2ef942056c.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/images/59f1e0f262ea821d5c73e30df5cb4e8f17b17ecc18543c036615df2ef942056c.jpg)

![603412fd59c1a329bef2073f3ff550c05f23f642f8c82fb76b267c6f8b99a7a9.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/images/603412fd59c1a329bef2073f3ff550c05f23f642f8c82fb76b267c6f8b99a7a9.jpg)

![7bbd0e931f5ed6e64e2b492ae958a35caf370396fb656dc005495af3d76db03e.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/images/7bbd0e931f5ed6e64e2b492ae958a35caf370396fb656dc005495af3d76db03e.jpg)

![7fc700c1eb1b66a1c5c493d2f1db008d7251382e8267f534cb1d8dc27309e471.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/images/7fc700c1eb1b66a1c5c493d2f1db008d7251382e8267f534cb1d8dc27309e471.jpg)

![8d7d17ca1b60219f564ec7ecf30d45011820e18a6239d6b1f34533cae9698063.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/images/8d7d17ca1b60219f564ec7ecf30d45011820e18a6239d6b1f34533cae9698063.jpg)

![93995dae95e890da67b29069ff7f61df004180207f2360b354d293824645af1b.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/images/93995dae95e890da67b29069ff7f61df004180207f2360b354d293824645af1b.jpg)

![99b6a84b0e5279c10951231d55b1ebcc282fa63a94fbf6a03717b63c3191e73e.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/images/99b6a84b0e5279c10951231d55b1ebcc282fa63a94fbf6a03717b63c3191e73e.jpg)

![aa0b324d6c5a198091455378fea89a7331d9bb7c776b4ab27edeb52fd08724c2.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/images/aa0b324d6c5a198091455378fea89a7331d9bb7c776b4ab27edeb52fd08724c2.jpg)

![ad60731e5962892e57d96cd14969b6abeebcbf420633841cd35c4020a807efee.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/images/ad60731e5962892e57d96cd14969b6abeebcbf420633841cd35c4020a807efee.jpg)

![c829776c6bae06f37ee0aa7965a295660071d394e643b12d30b357e484e860d8.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/images/c829776c6bae06f37ee0aa7965a295660071d394e643b12d30b357e484e860d8.jpg)

![cf5e43d7700d29e20ae47d498ea3d0eed71eef37f2036ce28357ae31c50890bc.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/images/cf5e43d7700d29e20ae47d498ea3d0eed71eef37f2036ce28357ae31c50890bc.jpg)

![e607358ca152e5f0b6b33de4f2bf5ada9b85409e5650b0b2241360356b64a816.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/images/e607358ca152e5f0b6b33de4f2bf5ada9b85409e5650b0b2241360356b64a816.jpg)

![f5066a6a142b7a45f63aec445cd69db4605903b2e458a1af724ee3dc45626add.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/images/f5066a6a142b7a45f63aec445cd69db4605903b2e458a1af724ee3dc45626add.jpg)

![fae23ae1ca3f207dfdf213f5eaa85a8f064d747f58a3d8aa2341a323da4e7081.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/images/fae23ae1ca3f207dfdf213f5eaa85a8f064d747f58a3d8aa2341a323da4e7081.jpg)

### Tables

![042e9ea61d48fa47f89840502b182c4ff33259534bcf41b47fe2340e1c0cbddb.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/tables/042e9ea61d48fa47f89840502b182c4ff33259534bcf41b47fe2340e1c0cbddb.jpg)

![268d307ac145fff6a92e08070ebf6bb72bfe169de23591691ad16d289636535c.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/tables/268d307ac145fff6a92e08070ebf6bb72bfe169de23591691ad16d289636535c.jpg)

![44eb0de54141f1d4a18c6bd39b0f9cf5df53cf8c937bb2a7d4a6e5a03b6516de.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/tables/44eb0de54141f1d4a18c6bd39b0f9cf5df53cf8c937bb2a7d4a6e5a03b6516de.jpg)

![5eec43e8c10517a7ce056ef89af8b1f1b245279da13d8b1eac883205982577a0.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/tables/5eec43e8c10517a7ce056ef89af8b1f1b245279da13d8b1eac883205982577a0.jpg)

![7afba8a9a9b6a4be7ac7eb8b69eb9d1e1f28618d2e4bdcb6b3e3eb487f7fb20a.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/tables/7afba8a9a9b6a4be7ac7eb8b69eb9d1e1f28618d2e4bdcb6b3e3eb487f7fb20a.jpg)

![866135a21e54b6d1b1ad258b98d63e420650b0557315276fad618edcd9209abd.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/tables/866135a21e54b6d1b1ad258b98d63e420650b0557315276fad618edcd9209abd.jpg)

![872b660b62ef84772023a35a918aadb0704e15c456c5bfb5288cf40fa34ffa2d.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/tables/872b660b62ef84772023a35a918aadb0704e15c456c5bfb5288cf40fa34ffa2d.jpg)

![92ec6ed27054dbac431d2c9a80383d05e9eebe6e7d8a6e7c4b168a21516b866f.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/tables/92ec6ed27054dbac431d2c9a80383d05e9eebe6e7d8a6e7c4b168a21516b866f.jpg)

![eec0fc0095823159c2ff1a3e491f9d89c2376254d0785d07dc2af90196c81325.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/tables/eec0fc0095823159c2ff1a3e491f9d89c2376254d0785d07dc2af90196c81325.jpg)

![f4c78f94a36b01e7b12f7c38ee305960c4c93c736ac1a818d4acc1184e64a8d7.jpg](../iclr_results/861_Towards Certification of Uncertainty Calibration under Adversarial Attacks/tables/f4c78f94a36b01e7b12f7c38ee305960c4c93c736ac1a818d4acc1184e64a8d7.jpg)

## VAE-Var: Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology


### Images

![0043f46563eef0149fd74dc83568a12f950fdfaffb0a117fd6bb35ec74f48843.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/0043f46563eef0149fd74dc83568a12f950fdfaffb0a117fd6bb35ec74f48843.jpg)

![040f9ec3de55620fb36b5c9c5a7e1513f0fe53ec5afa86972867c7b66293bfd3.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/040f9ec3de55620fb36b5c9c5a7e1513f0fe53ec5afa86972867c7b66293bfd3.jpg)

![072b53a60ca137d88796b30da98fd0464643cbebe86c2f8cff19f002db3404a5.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/072b53a60ca137d88796b30da98fd0464643cbebe86c2f8cff19f002db3404a5.jpg)

![0b9e04e1233a0715f77337b4b34c533cb2b835685b7a3c429d19e2d3ddf2662a.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/0b9e04e1233a0715f77337b4b34c533cb2b835685b7a3c429d19e2d3ddf2662a.jpg)

![1d7340795d56630882109048eef6f0fcde4977a62ff19f3202ca02fb8bad959e.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/1d7340795d56630882109048eef6f0fcde4977a62ff19f3202ca02fb8bad959e.jpg)

![24b3441879c3894bfa9aded1bfd0badd1d67e543fa669ab1747057c680924d32.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/24b3441879c3894bfa9aded1bfd0badd1d67e543fa669ab1747057c680924d32.jpg)

![29d93340b7aa964cf6ecfa71d5571cefa808b9e5dce7060803ad723c8afc5939.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/29d93340b7aa964cf6ecfa71d5571cefa808b9e5dce7060803ad723c8afc5939.jpg)

![29fed94e98756e7c0fe717dca145ae8a65b50babb53b19bc74803603a75858aa.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/29fed94e98756e7c0fe717dca145ae8a65b50babb53b19bc74803603a75858aa.jpg)

![379de1b2f8590dc071a4decffe5674fe5c1375f1fd585ba805c75e96672c89dd.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/379de1b2f8590dc071a4decffe5674fe5c1375f1fd585ba805c75e96672c89dd.jpg)

![50013d5dc39c6e79c57e6e6d23be2517e7128be1781841ddb0f01d8b4f4a4220.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/50013d5dc39c6e79c57e6e6d23be2517e7128be1781841ddb0f01d8b4f4a4220.jpg)

![52392b6315493db2c07bad134fc85e33be3123a8da371ff68a01edce05d705d1.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/52392b6315493db2c07bad134fc85e33be3123a8da371ff68a01edce05d705d1.jpg)

![66c5043a7d34bc51c5b0245083d14a69589bb5605739339a79a92f9d153a3ad5.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/66c5043a7d34bc51c5b0245083d14a69589bb5605739339a79a92f9d153a3ad5.jpg)

![75a3e92ad53910747e63bd6428165ec75b7ec120864e746db31888ebf8d7e8ce.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/75a3e92ad53910747e63bd6428165ec75b7ec120864e746db31888ebf8d7e8ce.jpg)

![797fbeaa7189d2c331cd0e361665639ca835723b526d828830ba8be76b2279e8.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/797fbeaa7189d2c331cd0e361665639ca835723b526d828830ba8be76b2279e8.jpg)

![79bee8ba080e7ea05810e0f1424c9782997644ce89212865af78ebadc992b54e.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/79bee8ba080e7ea05810e0f1424c9782997644ce89212865af78ebadc992b54e.jpg)

![7a6f71f02962b29808eca6456ae3162cdbfaf14f39ae06ece9ec28646f5427e3.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/7a6f71f02962b29808eca6456ae3162cdbfaf14f39ae06ece9ec28646f5427e3.jpg)

![7ea4b841a55c066043903f86f5df379b8f5388592ceacacfaef734f443521c06.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/7ea4b841a55c066043903f86f5df379b8f5388592ceacacfaef734f443521c06.jpg)

![891b1056487bc3cfc9609137745231e93d6c7d068207caef2a1c1e1b862de2e8.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/891b1056487bc3cfc9609137745231e93d6c7d068207caef2a1c1e1b862de2e8.jpg)

![8fed539236268bf97efff3d1f9a9f6720270dc0a76c5dff92f1cdd241a557c0e.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/8fed539236268bf97efff3d1f9a9f6720270dc0a76c5dff92f1cdd241a557c0e.jpg)

![91df630d1032905ee529bcd25584420569fd99da637ce6b164755f12334175a4.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/91df630d1032905ee529bcd25584420569fd99da637ce6b164755f12334175a4.jpg)

![986df934eb276975f6aec79bf3af7b34a9af8a78371e96f96a79ca3d3427777e.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/986df934eb276975f6aec79bf3af7b34a9af8a78371e96f96a79ca3d3427777e.jpg)

![a7c230b101f1dede1d5f3e7cf69c2a99983643d58370ba49d3576f1c9e2d2987.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/a7c230b101f1dede1d5f3e7cf69c2a99983643d58370ba49d3576f1c9e2d2987.jpg)

![ce297e5109a1c670c723fff9a9a0ae629c946cb454c93e31a113edc723010f3a.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/ce297e5109a1c670c723fff9a9a0ae629c946cb454c93e31a113edc723010f3a.jpg)

![da0093e048f8c98732d6dbdf50e12bc7fdf38b7ea4b779f46b5f2837370e5b9b.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/da0093e048f8c98732d6dbdf50e12bc7fdf38b7ea4b779f46b5f2837370e5b9b.jpg)

![e375e00865c0768db858e276b8a172082884237e689820f0c0d318ebc0a7c7b0.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/e375e00865c0768db858e276b8a172082884237e689820f0c0d318ebc0a7c7b0.jpg)

![f6110346212d566f372684be42c6395790983be3a6ff8b84e03405734d808f80.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/f6110346212d566f372684be42c6395790983be3a6ff8b84e03405734d808f80.jpg)

![f79f5f1af55f8ea9549ff5dceaa00d88cb8947ee5ce5a0f23b7a06e579feb28c.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/images/f79f5f1af55f8ea9549ff5dceaa00d88cb8947ee5ce5a0f23b7a06e579feb28c.jpg)

### Tables

![66e2eab95266bccdbc5efad7a40f9a9d27cb0ba1b74fd8a7150d7c96a92ff109.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/tables/66e2eab95266bccdbc5efad7a40f9a9d27cb0ba1b74fd8a7150d7c96a92ff109.jpg)

![a5e9a7cb1d3d767f8bd0229a6c928277b6bd90d6d5e3a7cb4c36dfb0a34bc41b.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/tables/a5e9a7cb1d3d767f8bd0229a6c928277b6bd90d6d5e3a7cb4c36dfb0a34bc41b.jpg)

![c5bd50d18a681df1b95cee28dc48ef4f4588f73aaeb573073dd572004a72a52c.jpg](../iclr_results/862_VAE-Var_ Variational Autoencoder-Enhanced Variational Methods for Data Assimilation in Meteorology/tables/c5bd50d18a681df1b95cee28dc48ef4f4588f73aaeb573073dd572004a72a52c.jpg)

## TODO: Enhancing LLM Alignment with Ternary Preferences


### Images

![0694c8f7ed14b56c1f0c04114c869ed144f256fc55146ab1feda8b7608d6e2e0.jpg](../iclr_results/863_TODO_ Enhancing LLM Alignment with Ternary Preferences/images/0694c8f7ed14b56c1f0c04114c869ed144f256fc55146ab1feda8b7608d6e2e0.jpg)

![2733a96d504f270292ad2ad7bfcb0b5e67fac2bf6de144da451b235edf1b959e.jpg](../iclr_results/863_TODO_ Enhancing LLM Alignment with Ternary Preferences/images/2733a96d504f270292ad2ad7bfcb0b5e67fac2bf6de144da451b235edf1b959e.jpg)

![2a1b3b562857f6ffd2e6b0ae72384f738c141ca7ef77326ee8c1ed98917d21af.jpg](../iclr_results/863_TODO_ Enhancing LLM Alignment with Ternary Preferences/images/2a1b3b562857f6ffd2e6b0ae72384f738c141ca7ef77326ee8c1ed98917d21af.jpg)

![556c88a0ed44ff9c6695988625b5f77ea2dfc64ebb0205c83e6e10379f0d0727.jpg](../iclr_results/863_TODO_ Enhancing LLM Alignment with Ternary Preferences/images/556c88a0ed44ff9c6695988625b5f77ea2dfc64ebb0205c83e6e10379f0d0727.jpg)

![a41cfa59e14637b940320e43fd977384b2370f2c35b8cc8d3d0eab7b80e6f9df.jpg](../iclr_results/863_TODO_ Enhancing LLM Alignment with Ternary Preferences/images/a41cfa59e14637b940320e43fd977384b2370f2c35b8cc8d3d0eab7b80e6f9df.jpg)

![b0dc40dddf91dcc0acb66df8174357dcbdea3eedecdc93e60bae7f33e8222881.jpg](../iclr_results/863_TODO_ Enhancing LLM Alignment with Ternary Preferences/images/b0dc40dddf91dcc0acb66df8174357dcbdea3eedecdc93e60bae7f33e8222881.jpg)

### Tables

![2e40f4c22e45943f9272a65500ade9c8ac96aaddd030b5c93658783f5afe59cd.jpg](../iclr_results/863_TODO_ Enhancing LLM Alignment with Ternary Preferences/tables/2e40f4c22e45943f9272a65500ade9c8ac96aaddd030b5c93658783f5afe59cd.jpg)

![2f6179e7a687aa66589820782a0282b6e51f9cc8506a768067ddafeebf7aba1c.jpg](../iclr_results/863_TODO_ Enhancing LLM Alignment with Ternary Preferences/tables/2f6179e7a687aa66589820782a0282b6e51f9cc8506a768067ddafeebf7aba1c.jpg)

![38883a36ab4b24fda00c2314e181a982cca07af6f7ed59bd987918118ba16250.jpg](../iclr_results/863_TODO_ Enhancing LLM Alignment with Ternary Preferences/tables/38883a36ab4b24fda00c2314e181a982cca07af6f7ed59bd987918118ba16250.jpg)

![3a08d945e728cf78fc380819878322b4090fb423d3d54c25623efe2753b861e9.jpg](../iclr_results/863_TODO_ Enhancing LLM Alignment with Ternary Preferences/tables/3a08d945e728cf78fc380819878322b4090fb423d3d54c25623efe2753b861e9.jpg)

![57727bdb6277e904f4350a1d66be39f7cd52708b6984cdbf596e5d9385e4b3f4.jpg](../iclr_results/863_TODO_ Enhancing LLM Alignment with Ternary Preferences/tables/57727bdb6277e904f4350a1d66be39f7cd52708b6984cdbf596e5d9385e4b3f4.jpg)

![6ce746208c7756218dc248e4240bc3f54a03cd9de6d75e071fe4aa3a43d264bb.jpg](../iclr_results/863_TODO_ Enhancing LLM Alignment with Ternary Preferences/tables/6ce746208c7756218dc248e4240bc3f54a03cd9de6d75e071fe4aa3a43d264bb.jpg)

![718b748261c74b1810c35bbd27dc3eeb485eea184ffcbaf8b1c259dd5e13848c.jpg](../iclr_results/863_TODO_ Enhancing LLM Alignment with Ternary Preferences/tables/718b748261c74b1810c35bbd27dc3eeb485eea184ffcbaf8b1c259dd5e13848c.jpg)

![a773977d7923d258c7f4e9f94ee24cbc4cf6979c51b799ce6305ae1e980abc53.jpg](../iclr_results/863_TODO_ Enhancing LLM Alignment with Ternary Preferences/tables/a773977d7923d258c7f4e9f94ee24cbc4cf6979c51b799ce6305ae1e980abc53.jpg)

![a8e1869f648c358fbb43480acf4410bd6c4c29ca8dd7137bfacd1908b62070eb.jpg](../iclr_results/863_TODO_ Enhancing LLM Alignment with Ternary Preferences/tables/a8e1869f648c358fbb43480acf4410bd6c4c29ca8dd7137bfacd1908b62070eb.jpg)

![b231dbab93148add5acc049013a2293e36600c0402f5030e8005c527b02455e9.jpg](../iclr_results/863_TODO_ Enhancing LLM Alignment with Ternary Preferences/tables/b231dbab93148add5acc049013a2293e36600c0402f5030e8005c527b02455e9.jpg)

![bae68b06a483e5c671a53afbc88727a3b37a3695d1a520b24baeeaa30d4f47da.jpg](../iclr_results/863_TODO_ Enhancing LLM Alignment with Ternary Preferences/tables/bae68b06a483e5c671a53afbc88727a3b37a3695d1a520b24baeeaa30d4f47da.jpg)

![cb44a7d2c35b5fa0d7d555500e0e0d6303590968f907437bd7134d7c19608f22.jpg](../iclr_results/863_TODO_ Enhancing LLM Alignment with Ternary Preferences/tables/cb44a7d2c35b5fa0d7d555500e0e0d6303590968f907437bd7134d7c19608f22.jpg)

![ee0266702270db2be776ac014e43e21aab2aafe871840c65be9bbea7912e205a.jpg](../iclr_results/863_TODO_ Enhancing LLM Alignment with Ternary Preferences/tables/ee0266702270db2be776ac014e43e21aab2aafe871840c65be9bbea7912e205a.jpg)

## Deep Kernel Posterior Learning under Infinite Variance Prior Weights


### Images

![29e2229f11a06b83a580fcfa4731690ebfd10f08b6102612d42ae57a1a5f4957.jpg](../iclr_results/864_Deep Kernel Posterior Learning under Infinite Variance Prior Weights/images/29e2229f11a06b83a580fcfa4731690ebfd10f08b6102612d42ae57a1a5f4957.jpg)

![64c6a2287ffc3736b363484b42ef60f087099b916fa8b13c1a077128438dda7e.jpg](../iclr_results/864_Deep Kernel Posterior Learning under Infinite Variance Prior Weights/images/64c6a2287ffc3736b363484b42ef60f087099b916fa8b13c1a077128438dda7e.jpg)

![7d0004f0112bd026e1e7d2e36abb1192d44bbc66693893c8efc5b72f8004f97c.jpg](../iclr_results/864_Deep Kernel Posterior Learning under Infinite Variance Prior Weights/images/7d0004f0112bd026e1e7d2e36abb1192d44bbc66693893c8efc5b72f8004f97c.jpg)

![82b125bad6c34fc042c11dd49b25d3ae8ac63cf49c7f93ec5e17e31e1e20bfd6.jpg](../iclr_results/864_Deep Kernel Posterior Learning under Infinite Variance Prior Weights/images/82b125bad6c34fc042c11dd49b25d3ae8ac63cf49c7f93ec5e17e31e1e20bfd6.jpg)

![874c0e22228c062c8a2a0410de38f4f4e58d146bd903adfcf39b4a6e6f4ddaa8.jpg](../iclr_results/864_Deep Kernel Posterior Learning under Infinite Variance Prior Weights/images/874c0e22228c062c8a2a0410de38f4f4e58d146bd903adfcf39b4a6e6f4ddaa8.jpg)

![89f86027a6c9ebede81ef8f469c826b28b3bd6850fc477822763ba2cc3a5dab4.jpg](../iclr_results/864_Deep Kernel Posterior Learning under Infinite Variance Prior Weights/images/89f86027a6c9ebede81ef8f469c826b28b3bd6850fc477822763ba2cc3a5dab4.jpg)

![8e53e154ae65d18262b9c6b7a7e0cb8d13ac32fb9f27e7a49a26b782636c3466.jpg](../iclr_results/864_Deep Kernel Posterior Learning under Infinite Variance Prior Weights/images/8e53e154ae65d18262b9c6b7a7e0cb8d13ac32fb9f27e7a49a26b782636c3466.jpg)

![a52a2936070aa4343c651e4412cefea47f8644387f282c683349405441d8171f.jpg](../iclr_results/864_Deep Kernel Posterior Learning under Infinite Variance Prior Weights/images/a52a2936070aa4343c651e4412cefea47f8644387f282c683349405441d8171f.jpg)

![a622a468ecc6dd1ee60c3564fff40260d7563fe3eac01012b13773cdb9574b3a.jpg](../iclr_results/864_Deep Kernel Posterior Learning under Infinite Variance Prior Weights/images/a622a468ecc6dd1ee60c3564fff40260d7563fe3eac01012b13773cdb9574b3a.jpg)

![a70766d392c5d638edb752b62dd388d76aa5ac74034fa86ceaa565d3a6f4a5ec.jpg](../iclr_results/864_Deep Kernel Posterior Learning under Infinite Variance Prior Weights/images/a70766d392c5d638edb752b62dd388d76aa5ac74034fa86ceaa565d3a6f4a5ec.jpg)

![db809cf10c1360ea7b6cfe59749503307603cfd7f269d09b81c62194a9f5b75d.jpg](../iclr_results/864_Deep Kernel Posterior Learning under Infinite Variance Prior Weights/images/db809cf10c1360ea7b6cfe59749503307603cfd7f269d09b81c62194a9f5b75d.jpg)

![de8edd6246fb5395799df708f489fc7f6c9bb7197d5a386b2266915a02cd16c2.jpg](../iclr_results/864_Deep Kernel Posterior Learning under Infinite Variance Prior Weights/images/de8edd6246fb5395799df708f489fc7f6c9bb7197d5a386b2266915a02cd16c2.jpg)

![f2e34ac90a54e4476065f3b0ab93c4bf1cec4a82b5f22b59d4b1c38b087e90e6.jpg](../iclr_results/864_Deep Kernel Posterior Learning under Infinite Variance Prior Weights/images/f2e34ac90a54e4476065f3b0ab93c4bf1cec4a82b5f22b59d4b1c38b087e90e6.jpg)

![fc9e6ade0641dec7a3e2ecd88cc9524981d33176f68de2fd4135fd430caafa7f.jpg](../iclr_results/864_Deep Kernel Posterior Learning under Infinite Variance Prior Weights/images/fc9e6ade0641dec7a3e2ecd88cc9524981d33176f68de2fd4135fd430caafa7f.jpg)

### Tables

![2632c4ff085bdee8f77567c98243bced279a7bc445511f819183506590ec25bc.jpg](../iclr_results/864_Deep Kernel Posterior Learning under Infinite Variance Prior Weights/tables/2632c4ff085bdee8f77567c98243bced279a7bc445511f819183506590ec25bc.jpg)

![2b19ee0afea412bcfecd83f84020102a37771cc3f9680eebbfdf317bc12ab180.jpg](../iclr_results/864_Deep Kernel Posterior Learning under Infinite Variance Prior Weights/tables/2b19ee0afea412bcfecd83f84020102a37771cc3f9680eebbfdf317bc12ab180.jpg)

![38cb5c6b824e96bde829d9e8bdacb2b95a01c3fa805ad9c97ed3b48ab11c373c.jpg](../iclr_results/864_Deep Kernel Posterior Learning under Infinite Variance Prior Weights/tables/38cb5c6b824e96bde829d9e8bdacb2b95a01c3fa805ad9c97ed3b48ab11c373c.jpg)

![4cad0a649c38c53fe3914fa90a914c939fc940fe773139685223592e2408ee33.jpg](../iclr_results/864_Deep Kernel Posterior Learning under Infinite Variance Prior Weights/tables/4cad0a649c38c53fe3914fa90a914c939fc940fe773139685223592e2408ee33.jpg)

![7ec7d88b30de08c1f8fa46994d2c57c821956c09ba7c0f7b6fc8acd536051cd2.jpg](../iclr_results/864_Deep Kernel Posterior Learning under Infinite Variance Prior Weights/tables/7ec7d88b30de08c1f8fa46994d2c57c821956c09ba7c0f7b6fc8acd536051cd2.jpg)

![82794998e8bde40df5f7b170c15201bd95638cf11975b025998e4836944992ee.jpg](../iclr_results/864_Deep Kernel Posterior Learning under Infinite Variance Prior Weights/tables/82794998e8bde40df5f7b170c15201bd95638cf11975b025998e4836944992ee.jpg)

![a520789cfad911fe63053b2244c8c4ac83d4b3fa92e9a757284440a6208350f6.jpg](../iclr_results/864_Deep Kernel Posterior Learning under Infinite Variance Prior Weights/tables/a520789cfad911fe63053b2244c8c4ac83d4b3fa92e9a757284440a6208350f6.jpg)

![c3ce6bf3a55deff6ca03fecae86e18194c9dc81c139f131dec46537c044552c3.jpg](../iclr_results/864_Deep Kernel Posterior Learning under Infinite Variance Prior Weights/tables/c3ce6bf3a55deff6ca03fecae86e18194c9dc81c139f131dec46537c044552c3.jpg)

## Latent-EnSF: A Latent Ensemble Score Filter for High-Dimensional Data Assimilation with Sparse Observation Data


### Images

![0e836b9e0d1a25c4138cba6ca9c0cd329383a077fb38c26476b4219f6cacf6f0.jpg](../iclr_results/865_Latent-EnSF_ A Latent Ensemble Score Filter for High-Dimensional Data Assimilation with Sparse Obser/images/0e836b9e0d1a25c4138cba6ca9c0cd329383a077fb38c26476b4219f6cacf6f0.jpg)

![113b3d4b66b983f0ee5b3d64f62c9ac5798c532f7d0653d2998c034491d2cde0.jpg](../iclr_results/865_Latent-EnSF_ A Latent Ensemble Score Filter for High-Dimensional Data Assimilation with Sparse Obser/images/113b3d4b66b983f0ee5b3d64f62c9ac5798c532f7d0653d2998c034491d2cde0.jpg)

![29193d15231c800f6fe83e7c433e05af50418024cc77378af5baa326e2731b89.jpg](../iclr_results/865_Latent-EnSF_ A Latent Ensemble Score Filter for High-Dimensional Data Assimilation with Sparse Obser/images/29193d15231c800f6fe83e7c433e05af50418024cc77378af5baa326e2731b89.jpg)

![3e134133edacf6587e9a59470bd1e5cd01ab2c1d2ddb4c132078c059bd5b85a2.jpg](../iclr_results/865_Latent-EnSF_ A Latent Ensemble Score Filter for High-Dimensional Data Assimilation with Sparse Obser/images/3e134133edacf6587e9a59470bd1e5cd01ab2c1d2ddb4c132078c059bd5b85a2.jpg)

![49d1924acd84b7550ed4bb630621e9618da6343d4a47a62837935c508e67e8dc.jpg](../iclr_results/865_Latent-EnSF_ A Latent Ensemble Score Filter for High-Dimensional Data Assimilation with Sparse Obser/images/49d1924acd84b7550ed4bb630621e9618da6343d4a47a62837935c508e67e8dc.jpg)

![54bad2870631ad739c493a9a43e15fdf2fffb432bbe9888b61fcc69fd1c53648.jpg](../iclr_results/865_Latent-EnSF_ A Latent Ensemble Score Filter for High-Dimensional Data Assimilation with Sparse Obser/images/54bad2870631ad739c493a9a43e15fdf2fffb432bbe9888b61fcc69fd1c53648.jpg)

![59dd0ada4f9bc9ff539506a89d34bbd2d80d83c79eb6ea0846233d52611bb8c1.jpg](../iclr_results/865_Latent-EnSF_ A Latent Ensemble Score Filter for High-Dimensional Data Assimilation with Sparse Obser/images/59dd0ada4f9bc9ff539506a89d34bbd2d80d83c79eb6ea0846233d52611bb8c1.jpg)

![79ac42e39544d8d3980714b2526ae6b92b059e7f86ebbb81f658a8cf2c491159.jpg](../iclr_results/865_Latent-EnSF_ A Latent Ensemble Score Filter for High-Dimensional Data Assimilation with Sparse Obser/images/79ac42e39544d8d3980714b2526ae6b92b059e7f86ebbb81f658a8cf2c491159.jpg)

![add6511313978f00e13d8b408669ee34f4067588245170a7244d7c541ca88e6b.jpg](../iclr_results/865_Latent-EnSF_ A Latent Ensemble Score Filter for High-Dimensional Data Assimilation with Sparse Obser/images/add6511313978f00e13d8b408669ee34f4067588245170a7244d7c541ca88e6b.jpg)

![af5ba5eea60c73231126bd5fd0088e03385333dd157f7439727a6930070b948f.jpg](../iclr_results/865_Latent-EnSF_ A Latent Ensemble Score Filter for High-Dimensional Data Assimilation with Sparse Obser/images/af5ba5eea60c73231126bd5fd0088e03385333dd157f7439727a6930070b948f.jpg)

![b9b82bcfec0131ecace40f2e5e4fce638435229df18332cc4fc69b822b7477c4.jpg](../iclr_results/865_Latent-EnSF_ A Latent Ensemble Score Filter for High-Dimensional Data Assimilation with Sparse Obser/images/b9b82bcfec0131ecace40f2e5e4fce638435229df18332cc4fc69b822b7477c4.jpg)

![bdf702a8f87d6e481769c0279096237af8545cb295f03f0fd7af3b5d18e8c107.jpg](../iclr_results/865_Latent-EnSF_ A Latent Ensemble Score Filter for High-Dimensional Data Assimilation with Sparse Obser/images/bdf702a8f87d6e481769c0279096237af8545cb295f03f0fd7af3b5d18e8c107.jpg)

![be85c03d879e5aaf038cd4aca7e13453e29b9bc36a16a09abfb3956453b56377.jpg](../iclr_results/865_Latent-EnSF_ A Latent Ensemble Score Filter for High-Dimensional Data Assimilation with Sparse Obser/images/be85c03d879e5aaf038cd4aca7e13453e29b9bc36a16a09abfb3956453b56377.jpg)

![db22e64fa9a6702ae3521774c22451154e44d4241d361b1baa5c407589f21f77.jpg](../iclr_results/865_Latent-EnSF_ A Latent Ensemble Score Filter for High-Dimensional Data Assimilation with Sparse Obser/images/db22e64fa9a6702ae3521774c22451154e44d4241d361b1baa5c407589f21f77.jpg)

![e704c4712ce3a1e5d24bde756fa2d2649c122aa91f236ae7d84d6261b1932cff.jpg](../iclr_results/865_Latent-EnSF_ A Latent Ensemble Score Filter for High-Dimensional Data Assimilation with Sparse Obser/images/e704c4712ce3a1e5d24bde756fa2d2649c122aa91f236ae7d84d6261b1932cff.jpg)

![ef3db9cfa03eec6540cd6eae67df31145510bc5b0b8c9c45f635f8ad96b1326b.jpg](../iclr_results/865_Latent-EnSF_ A Latent Ensemble Score Filter for High-Dimensional Data Assimilation with Sparse Obser/images/ef3db9cfa03eec6540cd6eae67df31145510bc5b0b8c9c45f635f8ad96b1326b.jpg)

### Tables

![002acf0746f299b574fa03e538a1e537effea3f1cd32db4b321ccd136494e4c6.jpg](../iclr_results/865_Latent-EnSF_ A Latent Ensemble Score Filter for High-Dimensional Data Assimilation with Sparse Obser/tables/002acf0746f299b574fa03e538a1e537effea3f1cd32db4b321ccd136494e4c6.jpg)

![5adef20c4674ac777f3b5eddf768101c7a15878d5a8b459b140dfaa280abad40.jpg](../iclr_results/865_Latent-EnSF_ A Latent Ensemble Score Filter for High-Dimensional Data Assimilation with Sparse Obser/tables/5adef20c4674ac777f3b5eddf768101c7a15878d5a8b459b140dfaa280abad40.jpg)

## Trajectory-Class-Aware Multi-Agent Reinforcement Learning


### Images

![0382e84ad148e9bbafab913e977508e7afba254badab7893fa06670d6cec9c4a.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/0382e84ad148e9bbafab913e977508e7afba254badab7893fa06670d6cec9c4a.jpg)

![1305932944d1f23e06a55d5c7d3a3f58f8574a8793081460306cd563dda82db0.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/1305932944d1f23e06a55d5c7d3a3f58f8574a8793081460306cd563dda82db0.jpg)

![13262a085506529a01490849da985f09bf2a280477087475c09eec31519b31dc.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/13262a085506529a01490849da985f09bf2a280477087475c09eec31519b31dc.jpg)

![19346766295934f6d2d53419a1e9273429209cad082c6fedfc918ad8f0e03ce2.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/19346766295934f6d2d53419a1e9273429209cad082c6fedfc918ad8f0e03ce2.jpg)

![1a4b3eb964072b42bdeda8c6e97a8679f8ab3e1a9f542c9659fb38ba2eadd715.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/1a4b3eb964072b42bdeda8c6e97a8679f8ab3e1a9f542c9659fb38ba2eadd715.jpg)

![2577960af2f0bb33d41c07af8543ac5ab4962ecf14a8b2ce55debd58e804fe07.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/2577960af2f0bb33d41c07af8543ac5ab4962ecf14a8b2ce55debd58e804fe07.jpg)

![2639cba27c7734ed2acfa10aa312f380e29a4f5c76b2c9611d20afc0051a0545.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/2639cba27c7734ed2acfa10aa312f380e29a4f5c76b2c9611d20afc0051a0545.jpg)

![2e7a74590e3e0632938c3e4d4c0096d5bf692ab28f349c5481f3720830e9b51f.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/2e7a74590e3e0632938c3e4d4c0096d5bf692ab28f349c5481f3720830e9b51f.jpg)

![39fd55dfd444a440c6593cc4e5d76ba3b1d266df298ff14c8d18163955a43166.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/39fd55dfd444a440c6593cc4e5d76ba3b1d266df298ff14c8d18163955a43166.jpg)

![3e19de98ef7023a9dce47017955896ed524877462f0541d0eed5b251c6b6e1c3.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/3e19de98ef7023a9dce47017955896ed524877462f0541d0eed5b251c6b6e1c3.jpg)

![426acd7c74d44fbda66254df7cb252b4aeedc62a3fbba799e1a52b32ef0a16c0.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/426acd7c74d44fbda66254df7cb252b4aeedc62a3fbba799e1a52b32ef0a16c0.jpg)

![485190cea9e323648f9cc6d7d1e025e226e058fd959f4aedcb6b5ebbd7455a61.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/485190cea9e323648f9cc6d7d1e025e226e058fd959f4aedcb6b5ebbd7455a61.jpg)

![4aa01102e75508a748bcf00fa80cb7b1046bfec33ee04d2efc5a66bd7e1304e1.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/4aa01102e75508a748bcf00fa80cb7b1046bfec33ee04d2efc5a66bd7e1304e1.jpg)

![53f84acddb5f9ccccaddaaa54e2a20b261b133f25af7a6330bda549e2bbef92c.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/53f84acddb5f9ccccaddaaa54e2a20b261b133f25af7a6330bda549e2bbef92c.jpg)

![675f8f7414625569231103b58ffa8d70541bc3abac70e30fd569f740ad9acdaf.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/675f8f7414625569231103b58ffa8d70541bc3abac70e30fd569f740ad9acdaf.jpg)

![6850fa03056deea2358d73cc5f2e79c94722c25e8f2537de68fda979d504023e.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/6850fa03056deea2358d73cc5f2e79c94722c25e8f2537de68fda979d504023e.jpg)

![7c4626d903427e1d9d79a46a84bc27f1948ace351d99512631a6bf4c91444650.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/7c4626d903427e1d9d79a46a84bc27f1948ace351d99512631a6bf4c91444650.jpg)

![7ce09a5bdaf238ba62f96a00c3efd1f43090ac7b2f1f030df53ada897b85d35e.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/7ce09a5bdaf238ba62f96a00c3efd1f43090ac7b2f1f030df53ada897b85d35e.jpg)

![7fdea44a34d171cb8d88b0af7805332e506fca035b2239fd00ebaf155f1cf0ea.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/7fdea44a34d171cb8d88b0af7805332e506fca035b2239fd00ebaf155f1cf0ea.jpg)

![8a21605d25d63484e35c4056db46a12204b6681d5e67debd71895a04ad705073.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/8a21605d25d63484e35c4056db46a12204b6681d5e67debd71895a04ad705073.jpg)

![9772bc3c19560812d6d7de3789a4aa24aaeb540a641c830a4112cd8e85331c79.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/9772bc3c19560812d6d7de3789a4aa24aaeb540a641c830a4112cd8e85331c79.jpg)

![aefe09baaf2d6cd4010bbdc7de88f77ac330de3c8e2e6fcc2ad0261e04b3d8aa.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/aefe09baaf2d6cd4010bbdc7de88f77ac330de3c8e2e6fcc2ad0261e04b3d8aa.jpg)

![b96fc4ff2b5ca865294fea57c9808676ac79dd3141135fc0aacfc2b5f733db59.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/b96fc4ff2b5ca865294fea57c9808676ac79dd3141135fc0aacfc2b5f733db59.jpg)

![bcf671c08a8ba93cca6e3d1dbdceb911cc2f3e922d49a5fbe60fc5c5dd1dd777.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/bcf671c08a8ba93cca6e3d1dbdceb911cc2f3e922d49a5fbe60fc5c5dd1dd777.jpg)

![c14404b0c1896e036ecfa20c3be29b45ad16d309ef7ebca495c47883191a8567.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/c14404b0c1896e036ecfa20c3be29b45ad16d309ef7ebca495c47883191a8567.jpg)

![c4bca9ed7e610eb0f1b023b24e092147f49e4b0336d84ec87eda49d7ced90728.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/c4bca9ed7e610eb0f1b023b24e092147f49e4b0336d84ec87eda49d7ced90728.jpg)

![c6e84581feaeab45c17ea9f7369064e2cd8341fd9c7c0146efca10322cb7c05f.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/c6e84581feaeab45c17ea9f7369064e2cd8341fd9c7c0146efca10322cb7c05f.jpg)

![d1402f64825ed3b263fbacc85aa63a3bc466e8ed4e0a63d5fdb9569f4b234a1d.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/d1402f64825ed3b263fbacc85aa63a3bc466e8ed4e0a63d5fdb9569f4b234a1d.jpg)

![d3e3d3d1f58257cb4b9a216600617cb6c52859177f42f80cf592dd0a03e6bbef.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/d3e3d3d1f58257cb4b9a216600617cb6c52859177f42f80cf592dd0a03e6bbef.jpg)

![d43b9271dbff9a519c5babfad0775ebc64da3351c176037c59ea37a4fe0cbef7.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/d43b9271dbff9a519c5babfad0775ebc64da3351c176037c59ea37a4fe0cbef7.jpg)

![e06f249b50474bf79dd9cce3f1b4d28d89b18fa68d88b3458af3f5e175618f2e.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/images/e06f249b50474bf79dd9cce3f1b4d28d89b18fa68d88b3458af3f5e175618f2e.jpg)

### Tables

![0b8ef119b9d9fca86428967c0a94533daeb7f31eb2f36f90c8cdf1a566599608.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/tables/0b8ef119b9d9fca86428967c0a94533daeb7f31eb2f36f90c8cdf1a566599608.jpg)

![0fdabe3512ab6018978579ebb6a9d495753a0e6b50f746cb47e7a85de9a41d05.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/tables/0fdabe3512ab6018978579ebb6a9d495753a0e6b50f746cb47e7a85de9a41d05.jpg)

![3fae9c0edb849f85fdc30fdc99a030a764fdd6c0d76abf348000b25cebe2dec4.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/tables/3fae9c0edb849f85fdc30fdc99a030a764fdd6c0d76abf348000b25cebe2dec4.jpg)

![789144a213f242fd5d8a180a10c146716076092db5ab7c2fb1c990675a79ada8.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/tables/789144a213f242fd5d8a180a10c146716076092db5ab7c2fb1c990675a79ada8.jpg)

![86b0c84ad90efc7393102e75a14b69c357ddc4348cda782a168190ef75e776cd.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/tables/86b0c84ad90efc7393102e75a14b69c357ddc4348cda782a168190ef75e776cd.jpg)

![a3dd18aee029a0b2a21ec1f9a4db6b065616721de6afe7384fa735b93a9a58f6.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/tables/a3dd18aee029a0b2a21ec1f9a4db6b065616721de6afe7384fa735b93a9a58f6.jpg)

![c57f707b99f08f0537a0c8d13e09dbd3943f8c292d8eb3767bf2d0ef9bfde377.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/tables/c57f707b99f08f0537a0c8d13e09dbd3943f8c292d8eb3767bf2d0ef9bfde377.jpg)

![db5b8f76f8a9c845b4767393403534581cc43b5acc3737f4ad4bc5eed6757a74.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/tables/db5b8f76f8a9c845b4767393403534581cc43b5acc3737f4ad4bc5eed6757a74.jpg)

![dfe10be0e80570b6dacec1d959ddf18ab954280810d0426317a9c13e83e4f5e9.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/tables/dfe10be0e80570b6dacec1d959ddf18ab954280810d0426317a9c13e83e4f5e9.jpg)

![e4641a7b12d0244d75b4bb05ee24dd508551359ab0922aa6d62c2e484c5db9e5.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/tables/e4641a7b12d0244d75b4bb05ee24dd508551359ab0922aa6d62c2e484c5db9e5.jpg)

![f014935fb68d820983a260083cfad8721410fe39e4574981ce61bd7341df432a.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/tables/f014935fb68d820983a260083cfad8721410fe39e4574981ce61bd7341df432a.jpg)

![f49d8d8d3a591d13e46a8adb845af5635ee533f1fbbd2ce51cd5d5a1205ec7f9.jpg](../iclr_results/866_Trajectory-Class-Aware Multi-Agent Reinforcement Learning/tables/f49d8d8d3a591d13e46a8adb845af5635ee533f1fbbd2ce51cd5d5a1205ec7f9.jpg)

## EG4D: Explicit Generation of 4D Object without Score Distillation


### Images

![077f9ef0bd332539b26f94bfa9a7a709a70c3cdc125b347305e62daa318c191b.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/077f9ef0bd332539b26f94bfa9a7a709a70c3cdc125b347305e62daa318c191b.jpg)

![0e8128214c91a39dca769fe5fab8ff2e940639f2436e4b1767137eb62a3a435c.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/0e8128214c91a39dca769fe5fab8ff2e940639f2436e4b1767137eb62a3a435c.jpg)

![18c05e634b36176104ba15ada16a8e754ea6fe85305b94bf597942e43aa1d37f.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/18c05e634b36176104ba15ada16a8e754ea6fe85305b94bf597942e43aa1d37f.jpg)

![19b6206a8da1a8faa152f37ebaf826e00915f344aaf3817906be6166782247aa.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/19b6206a8da1a8faa152f37ebaf826e00915f344aaf3817906be6166782247aa.jpg)

![1ace455bf8532853c7343dd3dcd25f80cec09acfc7dd352ffd69da0d955fbdb5.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/1ace455bf8532853c7343dd3dcd25f80cec09acfc7dd352ffd69da0d955fbdb5.jpg)

![1c39e5d9ff2cd510d2cb203567ec4ff504e51021ada0bcc1edd0350cb2fb238a.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/1c39e5d9ff2cd510d2cb203567ec4ff504e51021ada0bcc1edd0350cb2fb238a.jpg)

![26a176f20f1e2ec1f285e45f48b05c5d926b10c656686d4e931085bc75a98b76.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/26a176f20f1e2ec1f285e45f48b05c5d926b10c656686d4e931085bc75a98b76.jpg)

![4c4863404bd5fe64e69f6efe63d9ceb310d0f8ccc5182398e17bba768e81222f.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/4c4863404bd5fe64e69f6efe63d9ceb310d0f8ccc5182398e17bba768e81222f.jpg)

![4e9196b8f3e3a7821c3d9ecd8b54dc07a6e8408427b7ad3605ba9be04b433e4f.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/4e9196b8f3e3a7821c3d9ecd8b54dc07a6e8408427b7ad3605ba9be04b433e4f.jpg)

![4f02c35c0dd5673f3d3e8869f27dc4e42b62071bef9fdc327b41eec3d70f616b.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/4f02c35c0dd5673f3d3e8869f27dc4e42b62071bef9fdc327b41eec3d70f616b.jpg)

![4f2625940443d13d42f9e4804454843305fb6609a0a3e817fff883e6c306cf73.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/4f2625940443d13d42f9e4804454843305fb6609a0a3e817fff883e6c306cf73.jpg)

![51445aada776432959358f42975afaf8e8b2377b2145ee20dd17e36d76c48726.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/51445aada776432959358f42975afaf8e8b2377b2145ee20dd17e36d76c48726.jpg)

![56fce13c99dbaef708e2a5fb9547a92dfa8c7e0f9c3805fcfa978dfabe47fc58.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/56fce13c99dbaef708e2a5fb9547a92dfa8c7e0f9c3805fcfa978dfabe47fc58.jpg)

![646a7c552e1e30dbae40d1f62d96e1b0f3784e620c6b5411655d519525742629.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/646a7c552e1e30dbae40d1f62d96e1b0f3784e620c6b5411655d519525742629.jpg)

![6853e548d5d5d33085109c85c9c263ef8d1e0666a04126dc59b77f949c94a8de.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/6853e548d5d5d33085109c85c9c263ef8d1e0666a04126dc59b77f949c94a8de.jpg)

![7dcddb732cb7dabeaf3b439618678f8d7b75c04771a480afcae038937ceffd87.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/7dcddb732cb7dabeaf3b439618678f8d7b75c04771a480afcae038937ceffd87.jpg)

![93cecbd346d77bac8aab13daf72d8c47491deac4853a55f3df4d859e5ec69ebf.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/93cecbd346d77bac8aab13daf72d8c47491deac4853a55f3df4d859e5ec69ebf.jpg)

![b28dbb028425dbfbc74e2cf7661560aa2d9ea3d3b3bf16a39dd30a198aee9f73.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/b28dbb028425dbfbc74e2cf7661560aa2d9ea3d3b3bf16a39dd30a198aee9f73.jpg)

![c1b49d548c775b04971067521de47b4f7fc05b1185905eeb5de1a1e2567cc409.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/c1b49d548c775b04971067521de47b4f7fc05b1185905eeb5de1a1e2567cc409.jpg)

![cf5b2e91bd7570bd182e040ddcb65d571ab2670465f5d380048bbe21b2d761bd.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/cf5b2e91bd7570bd182e040ddcb65d571ab2670465f5d380048bbe21b2d761bd.jpg)

![e0b0762182c9f44f022bf341782f23422f7efff6d0856e4bfbace6c3af0e1660.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/e0b0762182c9f44f022bf341782f23422f7efff6d0856e4bfbace6c3af0e1660.jpg)

![ea7d3f9b6ddc6d82cb943b0c5ee6804cd9b2d457315512e6c2a09088d7a13fee.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/ea7d3f9b6ddc6d82cb943b0c5ee6804cd9b2d457315512e6c2a09088d7a13fee.jpg)

![ec4ce901150d85ec17e6fbf2a75cd44cbe7b386874699da15456bacabda0d42b.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/ec4ce901150d85ec17e6fbf2a75cd44cbe7b386874699da15456bacabda0d42b.jpg)

![fca9482de545f6d7dd9b967c5d7bd82d6303a27cc9dc1e6424a270ce6a42a9fc.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/images/fca9482de545f6d7dd9b967c5d7bd82d6303a27cc9dc1e6424a270ce6a42a9fc.jpg)

### Tables

![351e1de09761e1f430aeaf60fe95dac4eaf6655493a916440f3d010d6c0392e0.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/tables/351e1de09761e1f430aeaf60fe95dac4eaf6655493a916440f3d010d6c0392e0.jpg)

![4774cb7e4692ed2657094ba33d95690731224aa85213efa3ae4a15e3297ede05.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/tables/4774cb7e4692ed2657094ba33d95690731224aa85213efa3ae4a15e3297ede05.jpg)

![5e123938ab622a1dcdf1dd22f244987955287343e818210a48e4218a741f69b2.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/tables/5e123938ab622a1dcdf1dd22f244987955287343e818210a48e4218a741f69b2.jpg)

![caebc73a2247c444939ed510824410dd076364f9b1388922e552aa7ec7a3abc3.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/tables/caebc73a2247c444939ed510824410dd076364f9b1388922e552aa7ec7a3abc3.jpg)

![e2e191d01249caf1a3b05a741b384b6a261ca454e00016a29a2ebcbb4c2e7f4d.jpg](../iclr_results/867_EG4D_ Explicit Generation of 4D Object without Score Distillation/tables/e2e191d01249caf1a3b05a741b384b6a261ca454e00016a29a2ebcbb4c2e7f4d.jpg)

## The impact of allocation strategies in subset learning on the expressive power of neural networks


### Images

![14725ff9f0985f113e4ab21da439fe8ed7d3553dc0587cb7059fcdfddbae8eaa.jpg](../iclr_results/868_The impact of allocation strategies in subset learning on the expressive power of neural networks/images/14725ff9f0985f113e4ab21da439fe8ed7d3553dc0587cb7059fcdfddbae8eaa.jpg)

![b028c523f85c81de1ef4fd9bc7f8f373acd23e7590a31feaab77091dc532918e.jpg](../iclr_results/868_The impact of allocation strategies in subset learning on the expressive power of neural networks/images/b028c523f85c81de1ef4fd9bc7f8f373acd23e7590a31feaab77091dc532918e.jpg)

![d9d2148b94a84e07f13aafb3d9e3b07c5c1236762c267259f5c25990d105eb70.jpg](../iclr_results/868_The impact of allocation strategies in subset learning on the expressive power of neural networks/images/d9d2148b94a84e07f13aafb3d9e3b07c5c1236762c267259f5c25990d105eb70.jpg)

## Learning on One Mode: Addressing Multi-modality in Offline Reinforcement Learning


### Images

![1b1c594c1c84e18c4b329a2204cc79fd11afcded4feac8ff6d5db434a44e0b26.jpg](../iclr_results/869_Learning on One Mode_ Addressing Multi-modality in Offline Reinforcement Learning/images/1b1c594c1c84e18c4b329a2204cc79fd11afcded4feac8ff6d5db434a44e0b26.jpg)

![27455d11b4ae052d358bef84a473d105e47eedfc5a6e92fd186efa1b4549fb4b.jpg](../iclr_results/869_Learning on One Mode_ Addressing Multi-modality in Offline Reinforcement Learning/images/27455d11b4ae052d358bef84a473d105e47eedfc5a6e92fd186efa1b4549fb4b.jpg)

![2dbd9166668626faead17378bb938e69c9a70ab408cb8514b8de28cfb2d30c91.jpg](../iclr_results/869_Learning on One Mode_ Addressing Multi-modality in Offline Reinforcement Learning/images/2dbd9166668626faead17378bb938e69c9a70ab408cb8514b8de28cfb2d30c91.jpg)

![6a87f5b39229df549b838c3082b4673dccaa6658ed57456debd464a38e14bbac.jpg](../iclr_results/869_Learning on One Mode_ Addressing Multi-modality in Offline Reinforcement Learning/images/6a87f5b39229df549b838c3082b4673dccaa6658ed57456debd464a38e14bbac.jpg)

![8d60ae8cb7b7318535d7c833cc0419582b93e4cd624c5095d2c6c9c5e1610607.jpg](../iclr_results/869_Learning on One Mode_ Addressing Multi-modality in Offline Reinforcement Learning/images/8d60ae8cb7b7318535d7c833cc0419582b93e4cd624c5095d2c6c9c5e1610607.jpg)

### Tables

![0ce8a3cdbdff227a64b625a4f2d5d167d7df166ef132c6dce24e08a14d5be468.jpg](../iclr_results/869_Learning on One Mode_ Addressing Multi-modality in Offline Reinforcement Learning/tables/0ce8a3cdbdff227a64b625a4f2d5d167d7df166ef132c6dce24e08a14d5be468.jpg)

![afe2941b6fd34f23756fc6c81a4289a7002eb743b11da0814982dd93f786681c.jpg](../iclr_results/869_Learning on One Mode_ Addressing Multi-modality in Offline Reinforcement Learning/tables/afe2941b6fd34f23756fc6c81a4289a7002eb743b11da0814982dd93f786681c.jpg)

![b4a73f7f9c0be2d1b33756f8ec8536eedef406ebbd1a287fa099ea76e5bf59bc.jpg](../iclr_results/869_Learning on One Mode_ Addressing Multi-modality in Offline Reinforcement Learning/tables/b4a73f7f9c0be2d1b33756f8ec8536eedef406ebbd1a287fa099ea76e5bf59bc.jpg)

## Weak-to-Strong Generalization Through the Data-Centric Lens


### Images

![031bb235940145517352cce7681ed0d05b3ff7ce65d25e1a7102d15aa56ff61b.jpg](../iclr_results/870_Weak-to-Strong Generalization Through the Data-Centric Lens/images/031bb235940145517352cce7681ed0d05b3ff7ce65d25e1a7102d15aa56ff61b.jpg)

![4899515273f410ef3e03fced5b0df649eee82d8e92d6b5c3d3a5d53755959089.jpg](../iclr_results/870_Weak-to-Strong Generalization Through the Data-Centric Lens/images/4899515273f410ef3e03fced5b0df649eee82d8e92d6b5c3d3a5d53755959089.jpg)

![58cc7811bcb7e111f67eabd455a302b11e88fbaa28b2397f2e08812d5cb08c31.jpg](../iclr_results/870_Weak-to-Strong Generalization Through the Data-Centric Lens/images/58cc7811bcb7e111f67eabd455a302b11e88fbaa28b2397f2e08812d5cb08c31.jpg)

![71eb4ad5f55cd168728b22ddaaf540b179decaa59f652c36825328e9bf7a9190.jpg](../iclr_results/870_Weak-to-Strong Generalization Through the Data-Centric Lens/images/71eb4ad5f55cd168728b22ddaaf540b179decaa59f652c36825328e9bf7a9190.jpg)

![790bbebfb8b131b1667a5b71c344ae9f2401f97c842a11e8ae7acf55669a0d30.jpg](../iclr_results/870_Weak-to-Strong Generalization Through the Data-Centric Lens/images/790bbebfb8b131b1667a5b71c344ae9f2401f97c842a11e8ae7acf55669a0d30.jpg)

![7b3a5533a201c7b5ba280141da41fb365cbc44a406a7e917ea27ca45b34ad01e.jpg](../iclr_results/870_Weak-to-Strong Generalization Through the Data-Centric Lens/images/7b3a5533a201c7b5ba280141da41fb365cbc44a406a7e917ea27ca45b34ad01e.jpg)

![9a8d1e018dc65dd0d320fc640074c8bae35abd61085215205c913b6b36b76286.jpg](../iclr_results/870_Weak-to-Strong Generalization Through the Data-Centric Lens/images/9a8d1e018dc65dd0d320fc640074c8bae35abd61085215205c913b6b36b76286.jpg)

![a323aaf38f6b823b534bda98c9e60bb7683ea5117f16339b4fe8a2f61a34b112.jpg](../iclr_results/870_Weak-to-Strong Generalization Through the Data-Centric Lens/images/a323aaf38f6b823b534bda98c9e60bb7683ea5117f16339b4fe8a2f61a34b112.jpg)

![ae0328bfa16dbb9bfe72d780406bc80da0d1509032ec1c9f496385a9c90042e0.jpg](../iclr_results/870_Weak-to-Strong Generalization Through the Data-Centric Lens/images/ae0328bfa16dbb9bfe72d780406bc80da0d1509032ec1c9f496385a9c90042e0.jpg)

![b056bde0b548cceaacbb2859865d1e1d7a2ee32fe03554fdefb82239c439fce2.jpg](../iclr_results/870_Weak-to-Strong Generalization Through the Data-Centric Lens/images/b056bde0b548cceaacbb2859865d1e1d7a2ee32fe03554fdefb82239c439fce2.jpg)

![c23b47383c505b26bc7ea6accd1c8a47d86ce88f9d4b5d66f7249c81fd095e6b.jpg](../iclr_results/870_Weak-to-Strong Generalization Through the Data-Centric Lens/images/c23b47383c505b26bc7ea6accd1c8a47d86ce88f9d4b5d66f7249c81fd095e6b.jpg)

![c305c5a9d429cb2c3e872a71cd7232e1ee1acf028d1713e84e877706cdb2e5b2.jpg](../iclr_results/870_Weak-to-Strong Generalization Through the Data-Centric Lens/images/c305c5a9d429cb2c3e872a71cd7232e1ee1acf028d1713e84e877706cdb2e5b2.jpg)

![c38f0e04b3e69364876817f81d86592c8d92bb7965ace524ac12b6cf1c9f2f1a.jpg](../iclr_results/870_Weak-to-Strong Generalization Through the Data-Centric Lens/images/c38f0e04b3e69364876817f81d86592c8d92bb7965ace524ac12b6cf1c9f2f1a.jpg)

![d336b308af3efdcba5ea074a2aaf8aa7a5b6255526cec2745edbebd420206a3b.jpg](../iclr_results/870_Weak-to-Strong Generalization Through the Data-Centric Lens/images/d336b308af3efdcba5ea074a2aaf8aa7a5b6255526cec2745edbebd420206a3b.jpg)

![d7263fa0e8004778600d97cfb994b68b802f8703118f26d8484f72dfb25af092.jpg](../iclr_results/870_Weak-to-Strong Generalization Through the Data-Centric Lens/images/d7263fa0e8004778600d97cfb994b68b802f8703118f26d8484f72dfb25af092.jpg)

![e0087aad612ec945f70f92ce38436c2e3b900605e638368c5e40b306ef0b2e5d.jpg](../iclr_results/870_Weak-to-Strong Generalization Through the Data-Centric Lens/images/e0087aad612ec945f70f92ce38436c2e3b900605e638368c5e40b306ef0b2e5d.jpg)

![e4b7f8809346d09900966f3cae52e75d599268b912e505ae9d82dbbee0d5d8ea.jpg](../iclr_results/870_Weak-to-Strong Generalization Through the Data-Centric Lens/images/e4b7f8809346d09900966f3cae52e75d599268b912e505ae9d82dbbee0d5d8ea.jpg)

![e5ba71cc93485a3f048685e7fdc2f683ed433af3bfe6ff114938a9dfb589088c.jpg](../iclr_results/870_Weak-to-Strong Generalization Through the Data-Centric Lens/images/e5ba71cc93485a3f048685e7fdc2f683ed433af3bfe6ff114938a9dfb589088c.jpg)

![f8a8fc1272e8e5b59b66a5bf900c200e74dfea2ef25b9f7cd0b330022b4f6d61.jpg](../iclr_results/870_Weak-to-Strong Generalization Through the Data-Centric Lens/images/f8a8fc1272e8e5b59b66a5bf900c200e74dfea2ef25b9f7cd0b330022b4f6d61.jpg)

![fc319295cd80cdd0e4dd05311b96ed94ee686216382cabdc0ee96fdda081b939.jpg](../iclr_results/870_Weak-to-Strong Generalization Through the Data-Centric Lens/images/fc319295cd80cdd0e4dd05311b96ed94ee686216382cabdc0ee96fdda081b939.jpg)

### Tables

![34935038f4e202d07b4c42b26a58b8349a1b69f259d2c60794f92006081b68a8.jpg](../iclr_results/870_Weak-to-Strong Generalization Through the Data-Centric Lens/tables/34935038f4e202d07b4c42b26a58b8349a1b69f259d2c60794f92006081b68a8.jpg)

![8364319aaa325c60f842902ce3d827406321358dd1803ffc25992a3e3e1c9612.jpg](../iclr_results/870_Weak-to-Strong Generalization Through the Data-Centric Lens/tables/8364319aaa325c60f842902ce3d827406321358dd1803ffc25992a3e3e1c9612.jpg)

![b10cfbe623729a0c96fdcada485285e0833b44a77d429577a84d0499e83f22ab.jpg](../iclr_results/870_Weak-to-Strong Generalization Through the Data-Centric Lens/tables/b10cfbe623729a0c96fdcada485285e0833b44a77d429577a84d0499e83f22ab.jpg)

## VideoWebArena:  Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks


### Images

![37691f4e06f91c0ca8363dd87bdfe1239dfa67d9b8f74c02965da3da355116ba.jpg](../iclr_results/871_VideoWebArena_  Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks/images/37691f4e06f91c0ca8363dd87bdfe1239dfa67d9b8f74c02965da3da355116ba.jpg)

![4f924f855b8ea006052a8ff41d18e4c44f2b0276d5e880563af4ad9cc72e9cd8.jpg](../iclr_results/871_VideoWebArena_  Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks/images/4f924f855b8ea006052a8ff41d18e4c44f2b0276d5e880563af4ad9cc72e9cd8.jpg)

![584a41694cce0c14e553e70d837bd37dfa7c80b2873ae4fccde055119396003c.jpg](../iclr_results/871_VideoWebArena_  Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks/images/584a41694cce0c14e553e70d837bd37dfa7c80b2873ae4fccde055119396003c.jpg)

![be7338ae409aa005ebe68496ea36656de4ea73dddb08dc27e152ad566e3ae12e.jpg](../iclr_results/871_VideoWebArena_  Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks/images/be7338ae409aa005ebe68496ea36656de4ea73dddb08dc27e152ad566e3ae12e.jpg)

![c7b897723d9a41d91d755f3ab5bab94e8cb173e1adfb23d24d4618bc8e96c228.jpg](../iclr_results/871_VideoWebArena_  Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks/images/c7b897723d9a41d91d755f3ab5bab94e8cb173e1adfb23d24d4618bc8e96c228.jpg)

![d20468bbf653079afecdc042c50785739cc7ce7ae04117216787595ea4453979.jpg](../iclr_results/871_VideoWebArena_  Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks/images/d20468bbf653079afecdc042c50785739cc7ce7ae04117216787595ea4453979.jpg)

![fb9583cbebce66e211bc7d4f0033ff6c5dab64bb87510d35245e54f605f270d9.jpg](../iclr_results/871_VideoWebArena_  Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks/images/fb9583cbebce66e211bc7d4f0033ff6c5dab64bb87510d35245e54f605f270d9.jpg)

![fc8d0930e9c1e422f1e13492b5427856af9bc0fe1045c4d58d0b25c978d50435.jpg](../iclr_results/871_VideoWebArena_  Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks/images/fc8d0930e9c1e422f1e13492b5427856af9bc0fe1045c4d58d0b25c978d50435.jpg)

### Tables

![3c6319ff42a65f6663bc302804c72cc771ce95970d4e44b0c456a3d8efcdb06e.jpg](../iclr_results/871_VideoWebArena_  Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks/tables/3c6319ff42a65f6663bc302804c72cc771ce95970d4e44b0c456a3d8efcdb06e.jpg)

![4c87f027c400d2f0533e31766ed6cfe48f5d0402d55d881905485be6fab05d25.jpg](../iclr_results/871_VideoWebArena_  Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks/tables/4c87f027c400d2f0533e31766ed6cfe48f5d0402d55d881905485be6fab05d25.jpg)

![71be7e28821cab803aa9a84e225d7aac2cae333b1c0d450a003581a6fae3fc13.jpg](../iclr_results/871_VideoWebArena_  Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks/tables/71be7e28821cab803aa9a84e225d7aac2cae333b1c0d450a003581a6fae3fc13.jpg)

![7a47f2d3536f9664418d144ba440849e9f73b9c84db51a98fd633d0d03126b91.jpg](../iclr_results/871_VideoWebArena_  Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks/tables/7a47f2d3536f9664418d144ba440849e9f73b9c84db51a98fd633d0d03126b91.jpg)

![7d9b44b9e2a44fa25f91042e66c19543e9fab7e2f87864a7aca67ddf3e11c3aa.jpg](../iclr_results/871_VideoWebArena_  Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks/tables/7d9b44b9e2a44fa25f91042e66c19543e9fab7e2f87864a7aca67ddf3e11c3aa.jpg)

![d8fd3465871d832e0758dfa5f4af916a1c328ce05fc91dad79199c1388f12d7e.jpg](../iclr_results/871_VideoWebArena_  Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks/tables/d8fd3465871d832e0758dfa5f4af916a1c328ce05fc91dad79199c1388f12d7e.jpg)

![db9c54295ce7784ec9e9885529f95542a3f47d1177810e2abe94c5c05bf47c70.jpg](../iclr_results/871_VideoWebArena_  Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks/tables/db9c54295ce7784ec9e9885529f95542a3f47d1177810e2abe94c5c05bf47c70.jpg)

![e5773aa7b459a2b5ea88c69e7ad3310138195d35be998208843999a34aa7fa4c.jpg](../iclr_results/871_VideoWebArena_  Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks/tables/e5773aa7b459a2b5ea88c69e7ad3310138195d35be998208843999a34aa7fa4c.jpg)

![e65bab37e26bf154ebac236138f60b21d83fc24e7ef202a34d019bfa66fda581.jpg](../iclr_results/871_VideoWebArena_  Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks/tables/e65bab37e26bf154ebac236138f60b21d83fc24e7ef202a34d019bfa66fda581.jpg)

## OmniKV: Dynamic Context Selection for Efficient Long-Context LLMs


### Images

![3c3bdd5464835cef9fa3a1eefe53ef3bd1fe1f4874fb981baf00d151da4276a5.jpg](../iclr_results/872_OmniKV_ Dynamic Context Selection for Efficient Long-Context LLMs/images/3c3bdd5464835cef9fa3a1eefe53ef3bd1fe1f4874fb981baf00d151da4276a5.jpg)

![4d2d313ca79714d28f44db0294332ade4bc40398ff61d52bbe76733df590d5da.jpg](../iclr_results/872_OmniKV_ Dynamic Context Selection for Efficient Long-Context LLMs/images/4d2d313ca79714d28f44db0294332ade4bc40398ff61d52bbe76733df590d5da.jpg)

![580a6502c35e961fe5e64bd594e6ed6aaa55257b05837e9b4c071fae40e99044.jpg](../iclr_results/872_OmniKV_ Dynamic Context Selection for Efficient Long-Context LLMs/images/580a6502c35e961fe5e64bd594e6ed6aaa55257b05837e9b4c071fae40e99044.jpg)

![6bf676cac1d7454544b1a2b565660e6108f1c718502aad543dc17aba84741983.jpg](../iclr_results/872_OmniKV_ Dynamic Context Selection for Efficient Long-Context LLMs/images/6bf676cac1d7454544b1a2b565660e6108f1c718502aad543dc17aba84741983.jpg)

![7a0be8fc847cdffe09e460459ab69449112f353564cbba427c7afc4b48f1124e.jpg](../iclr_results/872_OmniKV_ Dynamic Context Selection for Efficient Long-Context LLMs/images/7a0be8fc847cdffe09e460459ab69449112f353564cbba427c7afc4b48f1124e.jpg)

![c0b443c37cdc9f5ea989d59874eb928fea380e90e032c822db36d83bac5da05e.jpg](../iclr_results/872_OmniKV_ Dynamic Context Selection for Efficient Long-Context LLMs/images/c0b443c37cdc9f5ea989d59874eb928fea380e90e032c822db36d83bac5da05e.jpg)

![c635f504b6e699aae57a04ea61e1a0095b69b99a0ee1e6ec607dbeb61018d9cc.jpg](../iclr_results/872_OmniKV_ Dynamic Context Selection for Efficient Long-Context LLMs/images/c635f504b6e699aae57a04ea61e1a0095b69b99a0ee1e6ec607dbeb61018d9cc.jpg)

![cf0033e4b0e9a229b1c1774e406067b57c0e0df278191532e5ce1b07c3e9d0d2.jpg](../iclr_results/872_OmniKV_ Dynamic Context Selection for Efficient Long-Context LLMs/images/cf0033e4b0e9a229b1c1774e406067b57c0e0df278191532e5ce1b07c3e9d0d2.jpg)

![db3fbd7d19985c81a5e45ff398fd17a11d88d72faf70f6b23c6dba6d0162b904.jpg](../iclr_results/872_OmniKV_ Dynamic Context Selection for Efficient Long-Context LLMs/images/db3fbd7d19985c81a5e45ff398fd17a11d88d72faf70f6b23c6dba6d0162b904.jpg)

### Tables

![13c5fdc632e5acaa9d122f1de36b3196241ae181fbc7ef426fdf64f51d2a4304.jpg](../iclr_results/872_OmniKV_ Dynamic Context Selection for Efficient Long-Context LLMs/tables/13c5fdc632e5acaa9d122f1de36b3196241ae181fbc7ef426fdf64f51d2a4304.jpg)

![1b353bcdfa7a7b9b0005665e4834965f94536c084f094ea75c7a8e6daa53dc28.jpg](../iclr_results/872_OmniKV_ Dynamic Context Selection for Efficient Long-Context LLMs/tables/1b353bcdfa7a7b9b0005665e4834965f94536c084f094ea75c7a8e6daa53dc28.jpg)

![344c2a2fc61ab6eb469e184637db0e4be1df257cadd3b55bed901829ddd87870.jpg](../iclr_results/872_OmniKV_ Dynamic Context Selection for Efficient Long-Context LLMs/tables/344c2a2fc61ab6eb469e184637db0e4be1df257cadd3b55bed901829ddd87870.jpg)

![4386965d598522569ac4d9b087de076a0f89795ef017d9f1771975199ff81041.jpg](../iclr_results/872_OmniKV_ Dynamic Context Selection for Efficient Long-Context LLMs/tables/4386965d598522569ac4d9b087de076a0f89795ef017d9f1771975199ff81041.jpg)

![5eed0a56aed5aca4c7d201e4b8df730d778c9bed5e30eac05f8cfc44288ed98e.jpg](../iclr_results/872_OmniKV_ Dynamic Context Selection for Efficient Long-Context LLMs/tables/5eed0a56aed5aca4c7d201e4b8df730d778c9bed5e30eac05f8cfc44288ed98e.jpg)

![6a0bed584b61d03417395af6a8fe36efff359f87bebee207d7ef26cc2411a1d9.jpg](../iclr_results/872_OmniKV_ Dynamic Context Selection for Efficient Long-Context LLMs/tables/6a0bed584b61d03417395af6a8fe36efff359f87bebee207d7ef26cc2411a1d9.jpg)

![6f3e9b90ab8b864640670455b7f161753837bef60a8057114839fb77f82b27f9.jpg](../iclr_results/872_OmniKV_ Dynamic Context Selection for Efficient Long-Context LLMs/tables/6f3e9b90ab8b864640670455b7f161753837bef60a8057114839fb77f82b27f9.jpg)

![92cdede11cf539281cf7b30cfe11c400116ae3e5bda9997dceaaf67a65adb30d.jpg](../iclr_results/872_OmniKV_ Dynamic Context Selection for Efficient Long-Context LLMs/tables/92cdede11cf539281cf7b30cfe11c400116ae3e5bda9997dceaaf67a65adb30d.jpg)

![9d23e0438864a48e7ce0f384b73f90b6c08da6d915943b75b0ba11a3dc174d46.jpg](../iclr_results/872_OmniKV_ Dynamic Context Selection for Efficient Long-Context LLMs/tables/9d23e0438864a48e7ce0f384b73f90b6c08da6d915943b75b0ba11a3dc174d46.jpg)

![a011adfb9eb01335b2eb3b72413357edda578c34f7123c0bf4586a354c3a6077.jpg](../iclr_results/872_OmniKV_ Dynamic Context Selection for Efficient Long-Context LLMs/tables/a011adfb9eb01335b2eb3b72413357edda578c34f7123c0bf4586a354c3a6077.jpg)

![c4ada7c07d76b9d6d5dbdcdb7f1b41a284b90788799af5ca6cd422d6a616e9df.jpg](../iclr_results/872_OmniKV_ Dynamic Context Selection for Efficient Long-Context LLMs/tables/c4ada7c07d76b9d6d5dbdcdb7f1b41a284b90788799af5ca6cd422d6a616e9df.jpg)

![c8fa5d4aab000ae054e5bcdcc68c0a22d4c954bf327b4bad2cd3bbd9da02bea2.jpg](../iclr_results/872_OmniKV_ Dynamic Context Selection for Efficient Long-Context LLMs/tables/c8fa5d4aab000ae054e5bcdcc68c0a22d4c954bf327b4bad2cd3bbd9da02bea2.jpg)

![e9c0cf84762ea04096a3b7a70b6d892a1bb6af66b1942e6db7953c650c5375e0.jpg](../iclr_results/872_OmniKV_ Dynamic Context Selection for Efficient Long-Context LLMs/tables/e9c0cf84762ea04096a3b7a70b6d892a1bb6af66b1942e6db7953c650c5375e0.jpg)

![feee2e8bf289fdd6193e931203a79bccb6179e8037046747bfc96ebc93820672.jpg](../iclr_results/872_OmniKV_ Dynamic Context Selection for Efficient Long-Context LLMs/tables/feee2e8bf289fdd6193e931203a79bccb6179e8037046747bfc96ebc93820672.jpg)

## Functional Homotopy: Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Attacks


### Images

![2d66c25a38415824e0e2e9b4041b046649b83b1d25ff0b5e19a0b614ea6042ac.jpg](../iclr_results/873_Functional Homotopy_ Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Att/images/2d66c25a38415824e0e2e9b4041b046649b83b1d25ff0b5e19a0b614ea6042ac.jpg)

![2eb9a2f75b48a688519462cc8b6a557641080c433d5323826cd5d64c4565c459.jpg](../iclr_results/873_Functional Homotopy_ Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Att/images/2eb9a2f75b48a688519462cc8b6a557641080c433d5323826cd5d64c4565c459.jpg)

![38f564d2b7197213fcba13ca9a363b7a0b6257fa9b54492ab1a399d2ca7eca50.jpg](../iclr_results/873_Functional Homotopy_ Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Att/images/38f564d2b7197213fcba13ca9a363b7a0b6257fa9b54492ab1a399d2ca7eca50.jpg)

![3a9c43f2cf4eb10f64b90d4e45c71ccc56422b2b0db0be9f6b57e22bf1533b2a.jpg](../iclr_results/873_Functional Homotopy_ Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Att/images/3a9c43f2cf4eb10f64b90d4e45c71ccc56422b2b0db0be9f6b57e22bf1533b2a.jpg)

![4b3a4ddf9ac6009557a3ea222872438266b554dc8897bcfb03802f5ed55e5eb3.jpg](../iclr_results/873_Functional Homotopy_ Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Att/images/4b3a4ddf9ac6009557a3ea222872438266b554dc8897bcfb03802f5ed55e5eb3.jpg)

![7eb1004a7a8316f906465d6a5687a6de54bec9d6c8d4d7aa06f62768e2804161.jpg](../iclr_results/873_Functional Homotopy_ Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Att/images/7eb1004a7a8316f906465d6a5687a6de54bec9d6c8d4d7aa06f62768e2804161.jpg)

![8858c2940a01047205b5bc1ac18ebcdad4c2f6537637b4d4ce1ccd1bd534f95c.jpg](../iclr_results/873_Functional Homotopy_ Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Att/images/8858c2940a01047205b5bc1ac18ebcdad4c2f6537637b4d4ce1ccd1bd534f95c.jpg)

![8864cf7f90b76905c929ea4104d10b04b9c1a6fc1b3fd5d6b97263d3167cede1.jpg](../iclr_results/873_Functional Homotopy_ Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Att/images/8864cf7f90b76905c929ea4104d10b04b9c1a6fc1b3fd5d6b97263d3167cede1.jpg)

![ad56dbc4772d9a7f4b8adfbd2cbe7340734b4190668614522703b6504176a4ce.jpg](../iclr_results/873_Functional Homotopy_ Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Att/images/ad56dbc4772d9a7f4b8adfbd2cbe7340734b4190668614522703b6504176a4ce.jpg)

![e05a3318f866b4d14eda7b8269e89b100262c5cee7a4cbc3862a859bcb692dc1.jpg](../iclr_results/873_Functional Homotopy_ Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Att/images/e05a3318f866b4d14eda7b8269e89b100262c5cee7a4cbc3862a859bcb692dc1.jpg)

![ec992ea2733a5017ced4bb0377a4beedfe556a6c0a2b518d9b7edba8a6041e63.jpg](../iclr_results/873_Functional Homotopy_ Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Att/images/ec992ea2733a5017ced4bb0377a4beedfe556a6c0a2b518d9b7edba8a6041e63.jpg)

### Tables

![bd1dcc22f7e2b00d488af13076cf5c05bfadd043c4497a1b02c7993bfddaaba1.jpg](../iclr_results/873_Functional Homotopy_ Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Att/tables/bd1dcc22f7e2b00d488af13076cf5c05bfadd043c4497a1b02c7993bfddaaba1.jpg)

![cbc86b42944fc561fb2378a74e23296bfdaf48a1ad521ff69bec7307b393ca93.jpg](../iclr_results/873_Functional Homotopy_ Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Att/tables/cbc86b42944fc561fb2378a74e23296bfdaf48a1ad521ff69bec7307b393ca93.jpg)

![d645640494dfa1b181287df28a998c8372f57f2b9d60c3b3179c23f8ba431339.jpg](../iclr_results/873_Functional Homotopy_ Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Att/tables/d645640494dfa1b181287df28a998c8372f57f2b9d60c3b3179c23f8ba431339.jpg)

## ScImage: How good are multimodal large language models at scientific text-to-image generation?

### Images

![05f6e88ac0472a2770b5275bd8855fe2aed757507a079ecbcc9f4aa59a9ff553.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/images/05f6e88ac0472a2770b5275bd8855fe2aed757507a079ecbcc9f4aa59a9ff553.jpg)

![067c4ead62717e78f68bddcb20845ca3929cf0b0a81c3c6e23577d890e6dd334.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/images/067c4ead62717e78f68bddcb20845ca3929cf0b0a81c3c6e23577d890e6dd334.jpg)

![0fa878ae8f582d40744bc6e9f4dd5778665cbe09de13a375a9ecb98017ac9c64.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/images/0fa878ae8f582d40744bc6e9f4dd5778665cbe09de13a375a9ecb98017ac9c64.jpg)

![13f5536da8fa81ec60bd2dbc7be559c01c0477fe1e6684ccc5c88c6386271f95.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/images/13f5536da8fa81ec60bd2dbc7be559c01c0477fe1e6684ccc5c88c6386271f95.jpg)

![15359e6bafa57456b0a773850d31daa2bdb855c382af448004b0ed455a5c71dd.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/images/15359e6bafa57456b0a773850d31daa2bdb855c382af448004b0ed455a5c71dd.jpg)

![227f4932736e8014388b7505179106d09a67ab2744feb09e2aa45e7e300a53b3.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/images/227f4932736e8014388b7505179106d09a67ab2744feb09e2aa45e7e300a53b3.jpg)

![373ec0ed132a00f3f4c07c997b82eae92e94dc4af6bc733691546b0b3f61af2f.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/images/373ec0ed132a00f3f4c07c997b82eae92e94dc4af6bc733691546b0b3f61af2f.jpg)

![64f4195186b199cf8cad9ac3dc11da9a93a710f35d8820a9cdc1e1586e18b5a9.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/images/64f4195186b199cf8cad9ac3dc11da9a93a710f35d8820a9cdc1e1586e18b5a9.jpg)

![66862f912b20674972d3785a07d54fa15b194d32aa750530c066c9ef926acad0.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/images/66862f912b20674972d3785a07d54fa15b194d32aa750530c066c9ef926acad0.jpg)

![7be6c7be712b02bafa245f38da4789f5d5ff015b2385b11f6b544a6bb1da45fd.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/images/7be6c7be712b02bafa245f38da4789f5d5ff015b2385b11f6b544a6bb1da45fd.jpg)

![86b529bb1d0bd42ee12b4249eb3d7a1a96513ee637ba2f484a4f535abb662596.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/images/86b529bb1d0bd42ee12b4249eb3d7a1a96513ee637ba2f484a4f535abb662596.jpg)

![8826231076824f6f90bf127de7c975920c80560d9e060faa6a018f2f17b76ca7.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/images/8826231076824f6f90bf127de7c975920c80560d9e060faa6a018f2f17b76ca7.jpg)

![927a9dcb422c7766e3d93bab10dceb912a09a98b7e75192445483c16493111dc.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/images/927a9dcb422c7766e3d93bab10dceb912a09a98b7e75192445483c16493111dc.jpg)

![9d01a90e53db04954ee971fc73bd59ac46969c523b493c16241fdbfaf2d54c61.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/images/9d01a90e53db04954ee971fc73bd59ac46969c523b493c16241fdbfaf2d54c61.jpg)

![9dd56d3103196cd3a4c038dd543a1298186afec4df7d9b14c986e1b77ca67fd1.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/images/9dd56d3103196cd3a4c038dd543a1298186afec4df7d9b14c986e1b77ca67fd1.jpg)

![a264d4d3663313e9fc038c55223379c2e3384c9e8623dc36f045ad78a93b35bb.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/images/a264d4d3663313e9fc038c55223379c2e3384c9e8623dc36f045ad78a93b35bb.jpg)

![b277a8a485659eb0fc38c253e626ed75aa6b33eb9b11a8a736f23067f7ba70d3.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/images/b277a8a485659eb0fc38c253e626ed75aa6b33eb9b11a8a736f23067f7ba70d3.jpg)

![b8905eedaf455cb3a9dca5438e1b7ec533df1ee272d9ec90aeffd06adc82ea05.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/images/b8905eedaf455cb3a9dca5438e1b7ec533df1ee272d9ec90aeffd06adc82ea05.jpg)

![da8f809d0402150bde9214ac8d718b17ab79a8397f41107c572f267eb5585f35.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/images/da8f809d0402150bde9214ac8d718b17ab79a8397f41107c572f267eb5585f35.jpg)

![e2f387df132c84a4061db0e2439ce2dfd9d753d07da0eee19dabeef6a647001c.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/images/e2f387df132c84a4061db0e2439ce2dfd9d753d07da0eee19dabeef6a647001c.jpg)

![e9b6d97a6a8e690c5db02ac3768eef253282a4cff3c9e8cf8937f0418763d0b9.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/images/e9b6d97a6a8e690c5db02ac3768eef253282a4cff3c9e8cf8937f0418763d0b9.jpg)

![f63c5f128476022674c4f9cedd65fd211b0861397ff14e9e645a72eb227ea63d.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/images/f63c5f128476022674c4f9cedd65fd211b0861397ff14e9e645a72eb227ea63d.jpg)

![fd82066fb008ae9fbf02fdad78251b3c027b638352619335f34842abee3563e4.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/images/fd82066fb008ae9fbf02fdad78251b3c027b638352619335f34842abee3563e4.jpg)

### Tables

![0d95436ee1c078d4e57e13d760bbb4cb3d76568e52e237878a18d6b7d63fc87b.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/tables/0d95436ee1c078d4e57e13d760bbb4cb3d76568e52e237878a18d6b7d63fc87b.jpg)

![10c2393ae67392eb575a4382d78ca756966f170fc6dca0052f9a516f8e7f056b.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/tables/10c2393ae67392eb575a4382d78ca756966f170fc6dca0052f9a516f8e7f056b.jpg)

![1bdd8129e22dab9d701f89179bfc21599c5ef5b52778e6a3a9e3eaabea9c36d4.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/tables/1bdd8129e22dab9d701f89179bfc21599c5ef5b52778e6a3a9e3eaabea9c36d4.jpg)

![2169554155ed35334abebead210e22db2ca22902207fd2ab1d7cdfbbd82bcf2b.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/tables/2169554155ed35334abebead210e22db2ca22902207fd2ab1d7cdfbbd82bcf2b.jpg)

![30eebbb0a0d131967a5e54aafe34e3ea8ed0d8753710c4c0f330a932fabe2126.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/tables/30eebbb0a0d131967a5e54aafe34e3ea8ed0d8753710c4c0f330a932fabe2126.jpg)

![3273e278fc6cf47e5940de4faad087df4d64f434f05aa9a2e82041f2ffdfbc2c.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/tables/3273e278fc6cf47e5940de4faad087df4d64f434f05aa9a2e82041f2ffdfbc2c.jpg)

![374a5ac34bcafc9b73245a0180d3f2c8ec89fe1ea9436ab92c5c5156fc1a9b55.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/tables/374a5ac34bcafc9b73245a0180d3f2c8ec89fe1ea9436ab92c5c5156fc1a9b55.jpg)

![3fe8002bbfcb198e3fdf1345cfc8df4fba8bf3f371f29a0b9ed04e0f93b2f841.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/tables/3fe8002bbfcb198e3fdf1345cfc8df4fba8bf3f371f29a0b9ed04e0f93b2f841.jpg)

![4826589cd7b6796218088d59af745fb62ed7a802a24e8a95b722e98c5a44c561.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/tables/4826589cd7b6796218088d59af745fb62ed7a802a24e8a95b722e98c5a44c561.jpg)

![6e2e950a7d22be8101084180422edaf8f04cb472e76f25d63a47b73a5d844184.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/tables/6e2e950a7d22be8101084180422edaf8f04cb472e76f25d63a47b73a5d844184.jpg)

![6e7f721ff826d555357db6835fc35a4e8dbff9f4d34ad2b3e78a3795730f0ccf.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/tables/6e7f721ff826d555357db6835fc35a4e8dbff9f4d34ad2b3e78a3795730f0ccf.jpg)

![9d64f7176634ce7941c4373c2e64577c9c90538d16fdf13e54487ee69a73e329.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/tables/9d64f7176634ce7941c4373c2e64577c9c90538d16fdf13e54487ee69a73e329.jpg)

![a7abd5000fee60bf0b4c10cfd474f8a391c31d3c4fc38b3ed4fd2e64b3e22868.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/tables/a7abd5000fee60bf0b4c10cfd474f8a391c31d3c4fc38b3ed4fd2e64b3e22868.jpg)

![b317556e3041de27ae231b804a6fd0092c49955b24c5dd4769a900e4a711d026.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/tables/b317556e3041de27ae231b804a6fd0092c49955b24c5dd4769a900e4a711d026.jpg)

![b344041dc2af7a24e2646a1969ac299250d65a808c194789c4749c3280ce0dcd.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/tables/b344041dc2af7a24e2646a1969ac299250d65a808c194789c4749c3280ce0dcd.jpg)

![c8830c70ce3b03ce497f7ee007722d5cf5ea7e2f26fe93b48acb2b6eded6b7da.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/tables/c8830c70ce3b03ce497f7ee007722d5cf5ea7e2f26fe93b48acb2b6eded6b7da.jpg)

![dad9291ca9e7f1d5df0bf4c6402574af29616186ed147f87803feea3dd1131dc.jpg](../iclr_results/874_ScImage_ How good are multimodal large language models at scientific text-to-image generation_/tables/dad9291ca9e7f1d5df0bf4c6402574af29616186ed147f87803feea3dd1131dc.jpg)
