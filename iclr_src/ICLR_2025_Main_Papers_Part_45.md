# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 45 of 100

## 目录 (Table of Contents)

1. [Diffusion$^2$: Dynamic 3D Content Generation via Score Composition of Video and Multi-view Diffusion Models](#Diffusion2-Dynamic-3D-Content-Generation-via-Score-Composition-of-Video-and-Multi-view-Diffusion-Models)
2. [Enhancing End-to-End Autonomous Driving with Latent World Model](#Enhancing-End-to-End-Autonomous-Driving-with-Latent-World-Model)
3. [Optimality of Matrix Mechanism on $\ell_p^p$-metric](#Optimality-of-Matrix-Mechanism-on-ell_pp-metric)
4. [Can We Ignore Labels in Out of Distribution Detection?](#Can-We-Ignore-Labels-in-Out-of-Distribution-Detection)
5. [Statistical Advantages of Perturbing Cosine Router in Mixture of Experts](#Statistical-Advantages-of-Perturbing-Cosine-Router-in-Mixture-of-Experts)
6. [Tuning Timestep-Distilled Diffusion Model Using Pairwise Sample Optimization](#Tuning-Timestep-Distilled-Diffusion-Model-Using-Pairwise-Sample-Optimization)
7. [Enabling Realtime Reinforcement Learning at Scale with Staggered Asynchronous Inference](#Enabling-Realtime-Reinforcement-Learning-at-Scale-with-Staggered-Asynchronous-Inference)
8. [On Evaluating the Durability of Safeguards for Open-Weight LLMs](#On-Evaluating-the-Durability-of-Safeguards-for-Open-Weight-LLMs)
9. [GReaTer: Gradients Over Reasoning Makes Smaller Language Models Strong Prompt Optimizers](#GReaTer-Gradients-Over-Reasoning-Makes-Smaller-Language-Models-Strong-Prompt-Optimizers)
10. [Looking Backward: Retrospective Backward Synthesis for Goal-Conditioned GFlowNets](#Looking-Backward-Retrospective-Backward-Synthesis-for-Goal-Conditioned-GFlowNets)
11. [When LLMs Play the Telephone Game: Cultural Attractors as Conceptual Tools to Evaluate LLMs in Multi-turn Settings](#When-LLMs-Play-the-Telephone-Game-Cultural-Attractors-as-Conceptual-Tools-to-Evaluate-LLMs-in-Multi-turn-Settings)
12. [LongMamba: Enhancing Mamba's Long-Context Capabilities via Training-Free Receptive Field Enlargement](#LongMamba-Enhancing-Mambas-Long-Context-Capabilities-via-Training-Free-Receptive-Field-Enlargement)
13. [Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning](#Unlearning-or-Obfuscating-Jogging-the-Memory-of-Unlearned-LLMs-via-Benign-Relearning)
14. [What is Wrong with Perplexity for Long-context Language Modeling?](#What-is-Wrong-with-Perplexity-for-Long-context-Language-Modeling)
15. [Recovery of Causal Graph Involving Latent Variables via Homologous Surrogates](#Recovery-of-Causal-Graph-Involving-Latent-Variables-via-Homologous-Surrogates)
16. [OpenPRM: Building Open-domain Process-based Reward Models with Preference Trees](#OpenPRM-Building-Open-domain-Process-based-Reward-Models-with-Preference-Trees)
17. [Efficient Discovery of Pareto Front for Multi-Objective Reinforcement Learning](#Efficient-Discovery-of-Pareto-Front-for-Multi-Objective-Reinforcement-Learning)
18. [TOMATO: Assessing Visual Temporal Reasoning Capabilities in Multimodal Foundation Models](#TOMATO-Assessing-Visual-Temporal-Reasoning-Capabilities-in-Multimodal-Foundation-Models)
19. [Amulet: ReAlignment During Test Time for Personalized Preference Adaptation of LLMs](#Amulet-ReAlignment-During-Test-Time-for-Personalized-Preference-Adaptation-of-LLMs)
20. [Controllable Satellite-to-Street-View Synthesis with Precise Pose Alignment and Zero-Shot Environmental Control](#Controllable-Satellite-to-Street-View-Synthesis-with-Precise-Pose-Alignment-and-Zero-Shot-Environmental-Control)
21. [MIA-DPO: Multi-Image Augmented Direct Preference Optimization For Large Vision-Language Models](#MIA-DPO-Multi-Image-Augmented-Direct-Preference-Optimization-For-Large-Vision-Language-Models)
22. [API Pack: A Massive Multi-Programming Language Dataset for API Call Generation](#API-Pack-A-Massive-Multi-Programming-Language-Dataset-for-API-Call-Generation)
23. [Capability Localization: Capabilities Can be Localized rather than Individual Knowledge](#Capability-Localization-Capabilities-Can-be-Localized-rather-than-Individual-Knowledge)
24. [Federated Continual Learning Goes Online: Uncertainty-Aware Memory Management for Vision Tasks and Beyond](#Federated-Continual-Learning-Goes-Online-Uncertainty-Aware-Memory-Management-for-Vision-Tasks-and-Beyond)
25. [Lasso Bandit with Compatibility Condition on Optimal Arm](#Lasso-Bandit-with-Compatibility-Condition-on-Optimal-Arm)
26. [Safety-Prioritizing Curricula for Constrained Reinforcement Learning](#Safety-Prioritizing-Curricula-for-Constrained-Reinforcement-Learning)
27. [Action Sequence Augmentation for Action Anticipation](#Action-Sequence-Augmentation-for-Action-Anticipation)
28. [The Rise and Down of Babel Tower: Investigating the Evolution Process of Multilingual Code Large Language Model](#The-Rise-and-Down-of-Babel-Tower-Investigating-the-Evolution-Process-of-Multilingual-Code-Large-Language-Model)
29. [Parameter Expanded Stochastic Gradient Markov Chain Monte Carlo](#Parameter-Expanded-Stochastic-Gradient-Markov-Chain-Monte-Carlo)
30. [What Secrets Do Your Manifolds Hold? Understanding the Local Geometry of Generative Models](#What-Secrets-Do-Your-Manifolds-Hold-Understanding-the-Local-Geometry-of-Generative-Models)
31. [TD-Paint: Faster Diffusion Inpainting Through Time-Aware Pixel Conditioning](#TD-Paint-Faster-Diffusion-Inpainting-Through-Time-Aware-Pixel-Conditioning)
32. [Why Does the Effective Context Length of LLMs Fall Short?](#Why-Does-the-Effective-Context-Length-of-LLMs-Fall-Short)
33. [Vision CNNs trained to estimate spatial latents learned similar ventral-stream-aligned representations](#Vision-CNNs-trained-to-estimate-spatial-latents-learned-similar-ventral-stream-aligned-representations)
34. [Value-aligned Behavior Cloning for Offline Reinforcement Learning via Bi-level Optimization](#Value-aligned-Behavior-Cloning-for-Offline-Reinforcement-Learning-via-Bi-level-Optimization)
35. [Persistent Pre-training Poisoning of LLMs](#Persistent-Pre-training-Poisoning-of-LLMs)
36. [BadRobot: Jailbreaking Embodied LLM Agents in the Physical World](#BadRobot-Jailbreaking-Embodied-LLM-Agents-in-the-Physical-World)
37. [Disentangled Representation Learning with the Gromov-Monge Gap](#Disentangled-Representation-Learning-with-the-Gromov-Monge-Gap)

---


## Diffusion$^2$: Dynamic 3D Content Generation via Score Composition of Video and Multi-view Diffusion Models

### Images

![06760f862216e98a23bc77d53dd706d546c9489170cd22b91416cba16b68a3b8.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/images/06760f862216e98a23bc77d53dd706d546c9489170cd22b91416cba16b68a3b8.jpg)

![10fe5cb5f736e82638dfd2f07acabc0326d75769a618c02c366082c18b7c44ff.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/images/10fe5cb5f736e82638dfd2f07acabc0326d75769a618c02c366082c18b7c44ff.jpg)

![227ac23f22fa5aa76b1babc9c197929c23783e01f0a069a4cbf69b234774521b.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/images/227ac23f22fa5aa76b1babc9c197929c23783e01f0a069a4cbf69b234774521b.jpg)

![23f654c82018b2439b040e67c3d219d19b27ca97c99594a8a607057301d54465.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/images/23f654c82018b2439b040e67c3d219d19b27ca97c99594a8a607057301d54465.jpg)

![39dc9ba35d293aa5178159e1aa198b45b52befd8f8bd1275eed9653bf78d1510.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/images/39dc9ba35d293aa5178159e1aa198b45b52befd8f8bd1275eed9653bf78d1510.jpg)

![42003ff826251649f3d98c6d2d2dca37028e1165ef1a959c4c96a82d029d6b9b.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/images/42003ff826251649f3d98c6d2d2dca37028e1165ef1a959c4c96a82d029d6b9b.jpg)

![54d235510816e7fd5a8cf58d0d8b5667a43b67326f95792fb55b37af4c961c5e.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/images/54d235510816e7fd5a8cf58d0d8b5667a43b67326f95792fb55b37af4c961c5e.jpg)

![56a0f7244a1df26083dd76b9cff8e3fa189768fc3d2d3e3b479441eca674bd89.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/images/56a0f7244a1df26083dd76b9cff8e3fa189768fc3d2d3e3b479441eca674bd89.jpg)

![5beba549a07ad9297bb4606625a38233180e66b94fd58d9d1da241aacc553e63.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/images/5beba549a07ad9297bb4606625a38233180e66b94fd58d9d1da241aacc553e63.jpg)

![5db36c25ad07b8f7150dde6314433517fe2c24de03724c625f80f8ff5c5cfca3.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/images/5db36c25ad07b8f7150dde6314433517fe2c24de03724c625f80f8ff5c5cfca3.jpg)

![7a9255228a21dca20c5ffac1bd769d5d21be9a5c8dd74b5180d787d13f60f61f.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/images/7a9255228a21dca20c5ffac1bd769d5d21be9a5c8dd74b5180d787d13f60f61f.jpg)

![81ca4367b9ddb2b1321ee08b68236a3972a6b9814f9b3985b0e4484381443a0c.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/images/81ca4367b9ddb2b1321ee08b68236a3972a6b9814f9b3985b0e4484381443a0c.jpg)

![987eecd6bef704ef10dba5652dca3b058bfc292ed868340a80eae01fa0eaac35.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/images/987eecd6bef704ef10dba5652dca3b058bfc292ed868340a80eae01fa0eaac35.jpg)

![9d11f37c3b4d37a2ff162d8207dd2a64eecc7fe290c449ddae3503d8dc8d0ec7.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/images/9d11f37c3b4d37a2ff162d8207dd2a64eecc7fe290c449ddae3503d8dc8d0ec7.jpg)

![b1c315fd0071444e34f1367f910d35eb4352a8d1878646d60c610c6a6a02333e.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/images/b1c315fd0071444e34f1367f910d35eb4352a8d1878646d60c610c6a6a02333e.jpg)

![c4092ea9c33bde74e0c7d3707ac1211a96f50b2090fb531382f58a4e57957855.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/images/c4092ea9c33bde74e0c7d3707ac1211a96f50b2090fb531382f58a4e57957855.jpg)

![cd5477595bb7bc2befbdf54b82b0ad99add8293d9a19d64e41678d2f25fc81c5.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/images/cd5477595bb7bc2befbdf54b82b0ad99add8293d9a19d64e41678d2f25fc81c5.jpg)

![ceecc4f22205c8cd3238ab446c5623a94e6f6fff820454de703a588e68907f7f.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/images/ceecc4f22205c8cd3238ab446c5623a94e6f6fff820454de703a588e68907f7f.jpg)

![e95957f6d477b842364d7a3142e5393dffd29dd30f2e7bd291508dc95bf5fa07.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/images/e95957f6d477b842364d7a3142e5393dffd29dd30f2e7bd291508dc95bf5fa07.jpg)

![fd7623e90275b74e4c190cb9f3989ee00668a6b4b3b2c433d1063db599d621a6.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/images/fd7623e90275b74e4c190cb9f3989ee00668a6b4b3b2c433d1063db599d621a6.jpg)

### Tables

![08a8fe991e2c0730ff265f8de5a1acaf758592ae3c69e50e55ca2a2b7815fd64.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/tables/08a8fe991e2c0730ff265f8de5a1acaf758592ae3c69e50e55ca2a2b7815fd64.jpg)

![0a9021c85bfb51b28bb0a83f3d1b88e6665823f449b06dc929c92bb1857b4539.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/tables/0a9021c85bfb51b28bb0a83f3d1b88e6665823f449b06dc929c92bb1857b4539.jpg)

![0e42812f1fe0f53c07d381b4df3223cd168dbdcac26e6f6e098bac56ee1deed8.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/tables/0e42812f1fe0f53c07d381b4df3223cd168dbdcac26e6f6e098bac56ee1deed8.jpg)

![184c6b4809acc3735104eba405c318a280cb7968eedd89da7200b28ca67ed527.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/tables/184c6b4809acc3735104eba405c318a280cb7968eedd89da7200b28ca67ed527.jpg)

![1ad1a10deb00afeef2277b514b6f8401cc41dd122962e5cd84288154e37ae789.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/tables/1ad1a10deb00afeef2277b514b6f8401cc41dd122962e5cd84288154e37ae789.jpg)

![204c737dce3f3def9fb61235610bbec1c8b6f131b0fb22162fadc0a91f438859.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/tables/204c737dce3f3def9fb61235610bbec1c8b6f131b0fb22162fadc0a91f438859.jpg)

![344a9ee12eb26ecdcd9d69308fcdf17a8557e9650cab88d76262eed8b3cad719.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/tables/344a9ee12eb26ecdcd9d69308fcdf17a8557e9650cab88d76262eed8b3cad719.jpg)

![56b047738ac2280db5ae0173fd6731b9e1d213d07f698fc084efa5929e9f6883.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/tables/56b047738ac2280db5ae0173fd6731b9e1d213d07f698fc084efa5929e9f6883.jpg)

![5e98535678acb6aa0aa112009d31916e3e64ecae2e61c50cd68fd4d098e8590f.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/tables/5e98535678acb6aa0aa112009d31916e3e64ecae2e61c50cd68fd4d098e8590f.jpg)

![7f3144613e95abdc22cc4176e4d23a8359c9f21d08da34f16ca978953b5c322d.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/tables/7f3144613e95abdc22cc4176e4d23a8359c9f21d08da34f16ca978953b5c322d.jpg)

![835987a24ca8fe86a1f432a7f5f0f266fbfc1720ae725a6d6460a01fd9bd76c8.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/tables/835987a24ca8fe86a1f432a7f5f0f266fbfc1720ae725a6d6460a01fd9bd76c8.jpg)

![a36f36ff38c37d2a8b7ec7b2c43f0a3d3cfbe6f20533526451959788b4d02c17.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/tables/a36f36ff38c37d2a8b7ec7b2c43f0a3d3cfbe6f20533526451959788b4d02c17.jpg)

![cf3f942396f146b16d6b928dd5e6237bf22d7588ac32a94c6cd9d57977b1b2be.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/tables/cf3f942396f146b16d6b928dd5e6237bf22d7588ac32a94c6cd9d57977b1b2be.jpg)

![de02876a657cb6c3ad975a1afe8ecb0282ee1a2ab814e30056430587cfd9827b.jpg](../iclr_results/1639_Teaching%20Human%20Behavior%20Improves%20Content%20Understanding%20Abilities%20Of%20VLMs/tables/de02876a657cb6c3ad975a1afe8ecb0282ee1a2ab814e30056430587cfd9827b.jpg)

## Diffusion$^2$: Dynamic 3D Content Generation via Score Composition of Video and Multi-view Diffusion Models


### Images

![0a5ee29168dc8fa9b9ae613265eb5675feb6c5e82ce3cc962f9d213d53d51b6f.jpg](../iclr_results/1640_Diffusion%24%5E2%24_%20Dynamic%203D%20Content%20Generation%20via%20Score%20Composition%20of%20Video%20and%20Multi-view%20Diffusion/images/0a5ee29168dc8fa9b9ae613265eb5675feb6c5e82ce3cc962f9d213d53d51b6f.jpg)

![1971d93e97e122c96f57d5ce511666823fb32b3ab85992abf3e516b8a77a2e83.jpg](../iclr_results/1640_Diffusion%24%5E2%24_%20Dynamic%203D%20Content%20Generation%20via%20Score%20Composition%20of%20Video%20and%20Multi-view%20Diffusion/images/1971d93e97e122c96f57d5ce511666823fb32b3ab85992abf3e516b8a77a2e83.jpg)

![3b0260d75a74fd002027249130a1f5d4727567c177b5c6caf0c8fec8acf41af3.jpg](../iclr_results/1640_Diffusion%24%5E2%24_%20Dynamic%203D%20Content%20Generation%20via%20Score%20Composition%20of%20Video%20and%20Multi-view%20Diffusion/images/3b0260d75a74fd002027249130a1f5d4727567c177b5c6caf0c8fec8acf41af3.jpg)

![4d02c1c30108eb60e3bd72345035fe1a70ae586f37b737d56b7c96f0f9ceaf89.jpg](../iclr_results/1640_Diffusion%24%5E2%24_%20Dynamic%203D%20Content%20Generation%20via%20Score%20Composition%20of%20Video%20and%20Multi-view%20Diffusion/images/4d02c1c30108eb60e3bd72345035fe1a70ae586f37b737d56b7c96f0f9ceaf89.jpg)

![74f46884015b70653cbc8dc4a935a1285db2ffc57d6c86789168085860ae3702.jpg](../iclr_results/1640_Diffusion%24%5E2%24_%20Dynamic%203D%20Content%20Generation%20via%20Score%20Composition%20of%20Video%20and%20Multi-view%20Diffusion/images/74f46884015b70653cbc8dc4a935a1285db2ffc57d6c86789168085860ae3702.jpg)

![7a0fd85d60ceaf53de9c38c48e7f0967257c41b9be525d6ed306607b04ff2508.jpg](../iclr_results/1640_Diffusion%24%5E2%24_%20Dynamic%203D%20Content%20Generation%20via%20Score%20Composition%20of%20Video%20and%20Multi-view%20Diffusion/images/7a0fd85d60ceaf53de9c38c48e7f0967257c41b9be525d6ed306607b04ff2508.jpg)

![999829195c3886788759643f82c68f0065330c63c0f545ad4ed0f9e2f9291b2b.jpg](../iclr_results/1640_Diffusion%24%5E2%24_%20Dynamic%203D%20Content%20Generation%20via%20Score%20Composition%20of%20Video%20and%20Multi-view%20Diffusion/images/999829195c3886788759643f82c68f0065330c63c0f545ad4ed0f9e2f9291b2b.jpg)

![a68ac7b084f6b74473fcddc04ae5fae8fe83bd760f271e770274b5da7201b72c.jpg](../iclr_results/1640_Diffusion%24%5E2%24_%20Dynamic%203D%20Content%20Generation%20via%20Score%20Composition%20of%20Video%20and%20Multi-view%20Diffusion/images/a68ac7b084f6b74473fcddc04ae5fae8fe83bd760f271e770274b5da7201b72c.jpg)

![b95366b372e25f3d5ed51b239f5923164175386558c04aefb33034fa2dff5851.jpg](../iclr_results/1640_Diffusion%24%5E2%24_%20Dynamic%203D%20Content%20Generation%20via%20Score%20Composition%20of%20Video%20and%20Multi-view%20Diffusion/images/b95366b372e25f3d5ed51b239f5923164175386558c04aefb33034fa2dff5851.jpg)

![d558bac27c0b94d90d71866a356f82f563dc086f1e0b0db7f4458fba36a7843c.jpg](../iclr_results/1640_Diffusion%24%5E2%24_%20Dynamic%203D%20Content%20Generation%20via%20Score%20Composition%20of%20Video%20and%20Multi-view%20Diffusion/images/d558bac27c0b94d90d71866a356f82f563dc086f1e0b0db7f4458fba36a7843c.jpg)

![df817535dcd0d2e129c990556956ab74dd2cae3607da8be0b8b380dc0207053f.jpg](../iclr_results/1640_Diffusion%24%5E2%24_%20Dynamic%203D%20Content%20Generation%20via%20Score%20Composition%20of%20Video%20and%20Multi-view%20Diffusion/images/df817535dcd0d2e129c990556956ab74dd2cae3607da8be0b8b380dc0207053f.jpg)

![e018127925b967d457011dcf6ea7fd20dcbf1d35899a4e90b2bb258cde7abd9e.jpg](../iclr_results/1640_Diffusion%24%5E2%24_%20Dynamic%203D%20Content%20Generation%20via%20Score%20Composition%20of%20Video%20and%20Multi-view%20Diffusion/images/e018127925b967d457011dcf6ea7fd20dcbf1d35899a4e90b2bb258cde7abd9e.jpg)

### Tables

![2c062872b1849688c65ef10ee569dec593f80c24835f088e9475754e9b0380ff.jpg](../iclr_results/1640_Diffusion%24%5E2%24_%20Dynamic%203D%20Content%20Generation%20via%20Score%20Composition%20of%20Video%20and%20Multi-view%20Diffusion/tables/2c062872b1849688c65ef10ee569dec593f80c24835f088e9475754e9b0380ff.jpg)

![3b16bc5032637394db9d5ebbb44efb2d039c9c6add7d3d849d4eca7e50f56dfa.jpg](../iclr_results/1640_Diffusion%24%5E2%24_%20Dynamic%203D%20Content%20Generation%20via%20Score%20Composition%20of%20Video%20and%20Multi-view%20Diffusion/tables/3b16bc5032637394db9d5ebbb44efb2d039c9c6add7d3d849d4eca7e50f56dfa.jpg)

![5cde7612c4c1430f2be3449043b8c6c3ec581bad5a3406d1ba0b44eb2e4cffc7.jpg](../iclr_results/1640_Diffusion%24%5E2%24_%20Dynamic%203D%20Content%20Generation%20via%20Score%20Composition%20of%20Video%20and%20Multi-view%20Diffusion/tables/5cde7612c4c1430f2be3449043b8c6c3ec581bad5a3406d1ba0b44eb2e4cffc7.jpg)

## Enhancing End-to-End Autonomous Driving with Latent World Model


### Images

![26c3b8aba1bbc10b0af8939c0dc17169500e609077fed1f569c779ebf6168003.jpg](../iclr_results/1641_Enhancing%20End-to-End%20Autonomous%20Driving%20with%20Latent%20World%20Model/images/26c3b8aba1bbc10b0af8939c0dc17169500e609077fed1f569c779ebf6168003.jpg)

![5c75f3159c397d7827e4f6867821e022377b5cbce84e482079cec3f8335d24c8.jpg](../iclr_results/1641_Enhancing%20End-to-End%20Autonomous%20Driving%20with%20Latent%20World%20Model/images/5c75f3159c397d7827e4f6867821e022377b5cbce84e482079cec3f8335d24c8.jpg)

![704af81ac98b1d97d8f4a137c52dbb3fab69b82f8a4e1941456d577d2953d12e.jpg](../iclr_results/1641_Enhancing%20End-to-End%20Autonomous%20Driving%20with%20Latent%20World%20Model/images/704af81ac98b1d97d8f4a137c52dbb3fab69b82f8a4e1941456d577d2953d12e.jpg)

![b1d0d246fb36126b73ce365bcb8018602ca76fb1409ec187f27ea51a68b96c7c.jpg](../iclr_results/1641_Enhancing%20End-to-End%20Autonomous%20Driving%20with%20Latent%20World%20Model/images/b1d0d246fb36126b73ce365bcb8018602ca76fb1409ec187f27ea51a68b96c7c.jpg)

![d47a79ac8e56aea3260b800b09ad29d247ffe0886fbb7a78e7b9213fada3086f.jpg](../iclr_results/1641_Enhancing%20End-to-End%20Autonomous%20Driving%20with%20Latent%20World%20Model/images/d47a79ac8e56aea3260b800b09ad29d247ffe0886fbb7a78e7b9213fada3086f.jpg)

![ec72135823267aae0b6ec544c0a611acaca2317b98c13240d31769b727493b45.jpg](../iclr_results/1641_Enhancing%20End-to-End%20Autonomous%20Driving%20with%20Latent%20World%20Model/images/ec72135823267aae0b6ec544c0a611acaca2317b98c13240d31769b727493b45.jpg)

### Tables

![124326e75f2cf93fadf55b7ede6f019255b9731d4510af3065663e58001588fa.jpg](../iclr_results/1641_Enhancing%20End-to-End%20Autonomous%20Driving%20with%20Latent%20World%20Model/tables/124326e75f2cf93fadf55b7ede6f019255b9731d4510af3065663e58001588fa.jpg)

![14021290a72d9fd1b0d425db5196f3cdf63589fe934f3c5a1b4cea78792f16ce.jpg](../iclr_results/1641_Enhancing%20End-to-End%20Autonomous%20Driving%20with%20Latent%20World%20Model/tables/14021290a72d9fd1b0d425db5196f3cdf63589fe934f3c5a1b4cea78792f16ce.jpg)

![3c0090ba0d0e8b46835abac76126e7fa11dbc37bf43de4c25b7ab71ac9c715a2.jpg](../iclr_results/1641_Enhancing%20End-to-End%20Autonomous%20Driving%20with%20Latent%20World%20Model/tables/3c0090ba0d0e8b46835abac76126e7fa11dbc37bf43de4c25b7ab71ac9c715a2.jpg)

![770ffa79b6b22e4a14c2a30c93dcd7dff62fdba2a7e6e12fe7a9b22d7549cd0e.jpg](../iclr_results/1641_Enhancing%20End-to-End%20Autonomous%20Driving%20with%20Latent%20World%20Model/tables/770ffa79b6b22e4a14c2a30c93dcd7dff62fdba2a7e6e12fe7a9b22d7549cd0e.jpg)

![aca80a3d05528517fe65d7fdc024a72bc21c385af1f4aab0693b2885cdb8729c.jpg](../iclr_results/1641_Enhancing%20End-to-End%20Autonomous%20Driving%20with%20Latent%20World%20Model/tables/aca80a3d05528517fe65d7fdc024a72bc21c385af1f4aab0693b2885cdb8729c.jpg)

![c92c7a7a1377436e8bde6e0e523969ddd4b913e2ba63b7613d0b8649378814bd.jpg](../iclr_results/1641_Enhancing%20End-to-End%20Autonomous%20Driving%20with%20Latent%20World%20Model/tables/c92c7a7a1377436e8bde6e0e523969ddd4b913e2ba63b7613d0b8649378814bd.jpg)

![d62c577d14677b557af113736f4b671b18308ab86d825e6e5ea333be3551926e.jpg](../iclr_results/1641_Enhancing%20End-to-End%20Autonomous%20Driving%20with%20Latent%20World%20Model/tables/d62c577d14677b557af113736f4b671b18308ab86d825e6e5ea333be3551926e.jpg)

![f3dc3a8139efc0edd0058f50ee766decf298d993270b1efbcc572d966f8b5c2c.jpg](../iclr_results/1641_Enhancing%20End-to-End%20Autonomous%20Driving%20with%20Latent%20World%20Model/tables/f3dc3a8139efc0edd0058f50ee766decf298d993270b1efbcc572d966f8b5c2c.jpg)

![fba2490317cd9f6a940479d09cf744d4885ba60845e849a96efac15d61ec580f.jpg](../iclr_results/1641_Enhancing%20End-to-End%20Autonomous%20Driving%20with%20Latent%20World%20Model/tables/fba2490317cd9f6a940479d09cf744d4885ba60845e849a96efac15d61ec580f.jpg)

## Optimality of Matrix Mechanism on $\ell_p^p$-metric


### Images

![c7f768ebd25f13c8d4b36df7560129d16cb3f5548c8c6151445402815e2f5488.jpg](../iclr_results/1642_Optimality%20of%20Matrix%20Mechanism%20on%20%24_ell_p%5Ep%24-metric/images/c7f768ebd25f13c8d4b36df7560129d16cb3f5548c8c6151445402815e2f5488.jpg)

## Can We Ignore Labels in Out of Distribution Detection?


### Images

![41097bc372fead8ea87437e0956b98e696130f9bec473c5313b4b9983f16a225.jpg](../iclr_results/1643_Can%20We%20Ignore%20Labels%20in%20Out%20of%20Distribution%20Detection_/images/41097bc372fead8ea87437e0956b98e696130f9bec473c5313b4b9983f16a225.jpg)

![503bbe2c872ef001ed8b39e0f80cea06a0bf36d0c3cf81b113d781299f17200c.jpg](../iclr_results/1643_Can%20We%20Ignore%20Labels%20in%20Out%20of%20Distribution%20Detection_/images/503bbe2c872ef001ed8b39e0f80cea06a0bf36d0c3cf81b113d781299f17200c.jpg)

![573adba28e6128d383705f2073a6fb91c356e0f542e0c585984dab7a703001ea.jpg](../iclr_results/1643_Can%20We%20Ignore%20Labels%20in%20Out%20of%20Distribution%20Detection_/images/573adba28e6128d383705f2073a6fb91c356e0f542e0c585984dab7a703001ea.jpg)

![797530a3cf5f402aaf281b7b20dbc60e5c20fb172c80bbdf47b0df88001ead07.jpg](../iclr_results/1643_Can%20We%20Ignore%20Labels%20in%20Out%20of%20Distribution%20Detection_/images/797530a3cf5f402aaf281b7b20dbc60e5c20fb172c80bbdf47b0df88001ead07.jpg)

![8086ae3f4ad7b63ae4857171ed92fe75227675e95c2c949c893471ee1a413228.jpg](../iclr_results/1643_Can%20We%20Ignore%20Labels%20in%20Out%20of%20Distribution%20Detection_/images/8086ae3f4ad7b63ae4857171ed92fe75227675e95c2c949c893471ee1a413228.jpg)

### Tables

![8dd21ff550208844f3a3c47f2913534fc87cc114d144398ac8a9fbda54444762.jpg](../iclr_results/1643_Can%20We%20Ignore%20Labels%20in%20Out%20of%20Distribution%20Detection_/tables/8dd21ff550208844f3a3c47f2913534fc87cc114d144398ac8a9fbda54444762.jpg)

![bb11613cd166e3325bc878ba2ee7ad86550ac9f6412564fdf9dec8f3ba458fb1.jpg](../iclr_results/1643_Can%20We%20Ignore%20Labels%20in%20Out%20of%20Distribution%20Detection_/tables/bb11613cd166e3325bc878ba2ee7ad86550ac9f6412564fdf9dec8f3ba458fb1.jpg)

![fac1c6aac6841f35632d791fac8eb66498ba004fd9287adbd3617535fd8b8312.jpg](../iclr_results/1643_Can%20We%20Ignore%20Labels%20in%20Out%20of%20Distribution%20Detection_/tables/fac1c6aac6841f35632d791fac8eb66498ba004fd9287adbd3617535fd8b8312.jpg)

## Statistical Advantages of Perturbing Cosine Router in Mixture of Experts


### Images

![8a8c73fb76fbff0012e141033ba03b75f3f67eeff7e3012e193fc9b6652c1796.jpg](../iclr_results/1644_Statistical%20Advantages%20of%20Perturbing%20Cosine%20Router%20in%20Mixture%20of%20Experts/images/8a8c73fb76fbff0012e141033ba03b75f3f67eeff7e3012e193fc9b6652c1796.jpg)

![ae7ccaddf3d43dcd06a62364395ecc4495ca4708a999806016decc5e0702e18d.jpg](../iclr_results/1644_Statistical%20Advantages%20of%20Perturbing%20Cosine%20Router%20in%20Mixture%20of%20Experts/images/ae7ccaddf3d43dcd06a62364395ecc4495ca4708a999806016decc5e0702e18d.jpg)

![d858c85d89b809af5a7c38f681db8b309f1229bdf861f08e89c977a344559cde.jpg](../iclr_results/1644_Statistical%20Advantages%20of%20Perturbing%20Cosine%20Router%20in%20Mixture%20of%20Experts/images/d858c85d89b809af5a7c38f681db8b309f1229bdf861f08e89c977a344559cde.jpg)

### Tables

![0caa80ae2ace4ef8c186c22629a6e014402faa7933d94dada8f0dc5945a3c4b3.jpg](../iclr_results/1644_Statistical%20Advantages%20of%20Perturbing%20Cosine%20Router%20in%20Mixture%20of%20Experts/tables/0caa80ae2ace4ef8c186c22629a6e014402faa7933d94dada8f0dc5945a3c4b3.jpg)

![5f1dd856b9d9bf89083da8b25be4b3b17b58314390d28845223cd0acddf5ea30.jpg](../iclr_results/1644_Statistical%20Advantages%20of%20Perturbing%20Cosine%20Router%20in%20Mixture%20of%20Experts/tables/5f1dd856b9d9bf89083da8b25be4b3b17b58314390d28845223cd0acddf5ea30.jpg)

![8abc66ef6127fe80771a7a27b5ca82385218a22fabe5f87e97e096dda5b6d6e0.jpg](../iclr_results/1644_Statistical%20Advantages%20of%20Perturbing%20Cosine%20Router%20in%20Mixture%20of%20Experts/tables/8abc66ef6127fe80771a7a27b5ca82385218a22fabe5f87e97e096dda5b6d6e0.jpg)

![a1e4590eb5135ef2666479a9d66c0a7d0f8fb41111bae473c9b88a43340787e9.jpg](../iclr_results/1644_Statistical%20Advantages%20of%20Perturbing%20Cosine%20Router%20in%20Mixture%20of%20Experts/tables/a1e4590eb5135ef2666479a9d66c0a7d0f8fb41111bae473c9b88a43340787e9.jpg)

![aacbcd6dad626936f98b55b1105f9a3a535fc33613adea47366e937c0d125be0.jpg](../iclr_results/1644_Statistical%20Advantages%20of%20Perturbing%20Cosine%20Router%20in%20Mixture%20of%20Experts/tables/aacbcd6dad626936f98b55b1105f9a3a535fc33613adea47366e937c0d125be0.jpg)

## Tuning Timestep-Distilled Diffusion Model Using Pairwise Sample Optimization


### Images

![2fca6a5a2980f07923ac29d40f379ceb7206e470acd88c4be312a506230820c4.jpg](../iclr_results/1645_Tuning%20Timestep-Distilled%20Diffusion%20Model%20Using%20Pairwise%20Sample%20Optimization/images/2fca6a5a2980f07923ac29d40f379ceb7206e470acd88c4be312a506230820c4.jpg)

![4466ad68d98265f83464b597eb46a5ed517669727be37371e1b20ecfbb85bc84.jpg](../iclr_results/1645_Tuning%20Timestep-Distilled%20Diffusion%20Model%20Using%20Pairwise%20Sample%20Optimization/images/4466ad68d98265f83464b597eb46a5ed517669727be37371e1b20ecfbb85bc84.jpg)

![8e9453d7e2ba76a6b3c7c2fecf3b6a8bb783df7bcc9ba219fa3832b95908978e.jpg](../iclr_results/1645_Tuning%20Timestep-Distilled%20Diffusion%20Model%20Using%20Pairwise%20Sample%20Optimization/images/8e9453d7e2ba76a6b3c7c2fecf3b6a8bb783df7bcc9ba219fa3832b95908978e.jpg)

![a0b95bd2e2b4c1fc01d5c5fc113c67cf0d1a754a7e75132623aca425d4e8beaa.jpg](../iclr_results/1645_Tuning%20Timestep-Distilled%20Diffusion%20Model%20Using%20Pairwise%20Sample%20Optimization/images/a0b95bd2e2b4c1fc01d5c5fc113c67cf0d1a754a7e75132623aca425d4e8beaa.jpg)

![a57b2a96bdf9df5942895dea14c29fd48a5757e304fa71dad721ed74d06789af.jpg](../iclr_results/1645_Tuning%20Timestep-Distilled%20Diffusion%20Model%20Using%20Pairwise%20Sample%20Optimization/images/a57b2a96bdf9df5942895dea14c29fd48a5757e304fa71dad721ed74d06789af.jpg)

![aa16e3283fa56e9147800e2762b715a13d75426a724d289fba2f24b34957d937.jpg](../iclr_results/1645_Tuning%20Timestep-Distilled%20Diffusion%20Model%20Using%20Pairwise%20Sample%20Optimization/images/aa16e3283fa56e9147800e2762b715a13d75426a724d289fba2f24b34957d937.jpg)

![b01b2025d38f6921accbcfd30ad367e2ff620344893de6ba9cf624c76048574b.jpg](../iclr_results/1645_Tuning%20Timestep-Distilled%20Diffusion%20Model%20Using%20Pairwise%20Sample%20Optimization/images/b01b2025d38f6921accbcfd30ad367e2ff620344893de6ba9cf624c76048574b.jpg)

![cc24cb034071a95d0f1d5336894295d4c5b9aa51d34da3c0ffd516b1b5271bc1.jpg](../iclr_results/1645_Tuning%20Timestep-Distilled%20Diffusion%20Model%20Using%20Pairwise%20Sample%20Optimization/images/cc24cb034071a95d0f1d5336894295d4c5b9aa51d34da3c0ffd516b1b5271bc1.jpg)

![fadf7b3339d947e4805d025cbb32ec476e7ba401ddd20496f8865a3dd30ea0fc.jpg](../iclr_results/1645_Tuning%20Timestep-Distilled%20Diffusion%20Model%20Using%20Pairwise%20Sample%20Optimization/images/fadf7b3339d947e4805d025cbb32ec476e7ba401ddd20496f8865a3dd30ea0fc.jpg)

### Tables

![0a68eec1d01729fb57201126c9d45759fd4cd2346b3c079696676e381eef113d.jpg](../iclr_results/1645_Tuning%20Timestep-Distilled%20Diffusion%20Model%20Using%20Pairwise%20Sample%20Optimization/tables/0a68eec1d01729fb57201126c9d45759fd4cd2346b3c079696676e381eef113d.jpg)

![224b13f75b438b63fef6e1b61a874911c2149488c8ae96f652ea2d9b89adb08e.jpg](../iclr_results/1645_Tuning%20Timestep-Distilled%20Diffusion%20Model%20Using%20Pairwise%20Sample%20Optimization/tables/224b13f75b438b63fef6e1b61a874911c2149488c8ae96f652ea2d9b89adb08e.jpg)

![8038a440bd53f17362c7aadcdc2e7a377bf2c227d0b09e9486bee557b09385d6.jpg](../iclr_results/1645_Tuning%20Timestep-Distilled%20Diffusion%20Model%20Using%20Pairwise%20Sample%20Optimization/tables/8038a440bd53f17362c7aadcdc2e7a377bf2c227d0b09e9486bee557b09385d6.jpg)

![b94026ab1a1bb5c3777f54d8acaffebe0153f788e789affb5ba13a9493c64ec9.jpg](../iclr_results/1645_Tuning%20Timestep-Distilled%20Diffusion%20Model%20Using%20Pairwise%20Sample%20Optimization/tables/b94026ab1a1bb5c3777f54d8acaffebe0153f788e789affb5ba13a9493c64ec9.jpg)

## Enabling Realtime Reinforcement Learning at Scale with Staggered Asynchronous Inference


### Images

![0b3ddc22de2320db97c563c2823f28a618427036ad7c0addeac33f285ad5d76d.jpg](../iclr_results/1646_Enabling%20Realtime%20Reinforcement%20Learning%20at%20Scale%20with%20Staggered%20Asynchronous%20Inference/images/0b3ddc22de2320db97c563c2823f28a618427036ad7c0addeac33f285ad5d76d.jpg)

![1be48de1d4305b5f875ea9bebc1667dadc8460396dcdd1b5ba2e868a004f66fd.jpg](../iclr_results/1646_Enabling%20Realtime%20Reinforcement%20Learning%20at%20Scale%20with%20Staggered%20Asynchronous%20Inference/images/1be48de1d4305b5f875ea9bebc1667dadc8460396dcdd1b5ba2e868a004f66fd.jpg)

![27fc503994126ccedb7edc476bbd033e3af43338afa33361c3a77639354cff4e.jpg](../iclr_results/1646_Enabling%20Realtime%20Reinforcement%20Learning%20at%20Scale%20with%20Staggered%20Asynchronous%20Inference/images/27fc503994126ccedb7edc476bbd033e3af43338afa33361c3a77639354cff4e.jpg)

![3421022bcdea61b4e93e8ca413f09f2188b38b8895a1f07c13f12bfe027b8013.jpg](../iclr_results/1646_Enabling%20Realtime%20Reinforcement%20Learning%20at%20Scale%20with%20Staggered%20Asynchronous%20Inference/images/3421022bcdea61b4e93e8ca413f09f2188b38b8895a1f07c13f12bfe027b8013.jpg)

![553f9a860386ad5d1e7214ef408a77cbe0c02ef57252113f61d3bdbee369a51a.jpg](../iclr_results/1646_Enabling%20Realtime%20Reinforcement%20Learning%20at%20Scale%20with%20Staggered%20Asynchronous%20Inference/images/553f9a860386ad5d1e7214ef408a77cbe0c02ef57252113f61d3bdbee369a51a.jpg)

![5afd5f270fedac5a275b9ebed33a1998e1c6849f29322abfadab2192e93a36d0.jpg](../iclr_results/1646_Enabling%20Realtime%20Reinforcement%20Learning%20at%20Scale%20with%20Staggered%20Asynchronous%20Inference/images/5afd5f270fedac5a275b9ebed33a1998e1c6849f29322abfadab2192e93a36d0.jpg)

![69c4fe0f94c898e3f6d6260bf889f2713287e36048ae743f62eaddb6bd67fbb5.jpg](../iclr_results/1646_Enabling%20Realtime%20Reinforcement%20Learning%20at%20Scale%20with%20Staggered%20Asynchronous%20Inference/images/69c4fe0f94c898e3f6d6260bf889f2713287e36048ae743f62eaddb6bd67fbb5.jpg)

![898e721c378449f935760ad1608adadf7072419dbb092a08a26e4961163e5044.jpg](../iclr_results/1646_Enabling%20Realtime%20Reinforcement%20Learning%20at%20Scale%20with%20Staggered%20Asynchronous%20Inference/images/898e721c378449f935760ad1608adadf7072419dbb092a08a26e4961163e5044.jpg)

![8d56ce6e444f3e81408f05296f1da93c847a41ffd89248da5f4be349a5a26146.jpg](../iclr_results/1646_Enabling%20Realtime%20Reinforcement%20Learning%20at%20Scale%20with%20Staggered%20Asynchronous%20Inference/images/8d56ce6e444f3e81408f05296f1da93c847a41ffd89248da5f4be349a5a26146.jpg)

![8e2e27a7827ec8f086e01b23c49d36b28a25b4e85fc7f11806ff26627fafb5d7.jpg](../iclr_results/1646_Enabling%20Realtime%20Reinforcement%20Learning%20at%20Scale%20with%20Staggered%20Asynchronous%20Inference/images/8e2e27a7827ec8f086e01b23c49d36b28a25b4e85fc7f11806ff26627fafb5d7.jpg)

![9c498b2bb52620593f24cea94c8e5eb3b29f7e2cf48a65d053931251c4d97d67.jpg](../iclr_results/1646_Enabling%20Realtime%20Reinforcement%20Learning%20at%20Scale%20with%20Staggered%20Asynchronous%20Inference/images/9c498b2bb52620593f24cea94c8e5eb3b29f7e2cf48a65d053931251c4d97d67.jpg)

![a289e5706e23368ec410afde2770e4beadd871ed8a44c87608e7a202d3f63b81.jpg](../iclr_results/1646_Enabling%20Realtime%20Reinforcement%20Learning%20at%20Scale%20with%20Staggered%20Asynchronous%20Inference/images/a289e5706e23368ec410afde2770e4beadd871ed8a44c87608e7a202d3f63b81.jpg)

![a51cd28907f19145a9495faede74afd1a380991fbfc45619740691f417c4ad47.jpg](../iclr_results/1646_Enabling%20Realtime%20Reinforcement%20Learning%20at%20Scale%20with%20Staggered%20Asynchronous%20Inference/images/a51cd28907f19145a9495faede74afd1a380991fbfc45619740691f417c4ad47.jpg)

![b61579b864ff859da6816de56f06f2ba1af4639d1966722fade36c40299eafc5.jpg](../iclr_results/1646_Enabling%20Realtime%20Reinforcement%20Learning%20at%20Scale%20with%20Staggered%20Asynchronous%20Inference/images/b61579b864ff859da6816de56f06f2ba1af4639d1966722fade36c40299eafc5.jpg)

![bd05afa63b7ef1d619b3e04999b90cbd79faa36eb522112c333ca794761763db.jpg](../iclr_results/1646_Enabling%20Realtime%20Reinforcement%20Learning%20at%20Scale%20with%20Staggered%20Asynchronous%20Inference/images/bd05afa63b7ef1d619b3e04999b90cbd79faa36eb522112c333ca794761763db.jpg)

![c212dcc1440a5c4c0835ee3813e4d28ed6d6088f2dd1908410631f7a2e758f57.jpg](../iclr_results/1646_Enabling%20Realtime%20Reinforcement%20Learning%20at%20Scale%20with%20Staggered%20Asynchronous%20Inference/images/c212dcc1440a5c4c0835ee3813e4d28ed6d6088f2dd1908410631f7a2e758f57.jpg)

![ebea2f2dbd07a98d3ff3c74e27d0c16f71b541c6aedab15bfcdb20c41b8038a6.jpg](../iclr_results/1646_Enabling%20Realtime%20Reinforcement%20Learning%20at%20Scale%20with%20Staggered%20Asynchronous%20Inference/images/ebea2f2dbd07a98d3ff3c74e27d0c16f71b541c6aedab15bfcdb20c41b8038a6.jpg)

![f46b2b56a202e171b716d04aa35d891c13932b7fd6d2f7a085c7ed71eeea64f8.jpg](../iclr_results/1646_Enabling%20Realtime%20Reinforcement%20Learning%20at%20Scale%20with%20Staggered%20Asynchronous%20Inference/images/f46b2b56a202e171b716d04aa35d891c13932b7fd6d2f7a085c7ed71eeea64f8.jpg)

### Tables

![244b97ee993f40b761f14294a0465697d6cd2ca9e8d4a45dd1b63d7dad910a8b.jpg](../iclr_results/1646_Enabling%20Realtime%20Reinforcement%20Learning%20at%20Scale%20with%20Staggered%20Asynchronous%20Inference/tables/244b97ee993f40b761f14294a0465697d6cd2ca9e8d4a45dd1b63d7dad910a8b.jpg)

## On Evaluating the Durability of Safeguards for Open-Weight LLMs


### Images

![0b3680ae98199183e294f6316753cd193af4b165a96cf2790c99aa0dbb5d8f06.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/images/0b3680ae98199183e294f6316753cd193af4b165a96cf2790c99aa0dbb5d8f06.jpg)

![26a70485a47b943fd520c334095928730bb44623728dac9e6734ee55c3e6476c.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/images/26a70485a47b943fd520c334095928730bb44623728dac9e6734ee55c3e6476c.jpg)

![3f19e3e196ea3e0f7e2a50adc5308b4082d9f11da7ba3cb9b77f24a22a0acd02.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/images/3f19e3e196ea3e0f7e2a50adc5308b4082d9f11da7ba3cb9b77f24a22a0acd02.jpg)

![778dc30182d4cfe4bfb021b8331db0bae555da1ee68dc2d0dc7ff5b8100745c5.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/images/778dc30182d4cfe4bfb021b8331db0bae555da1ee68dc2d0dc7ff5b8100745c5.jpg)

![87170f223644484b846364bcdaea17a110c9a689ac05947b61cc0825e6326aca.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/images/87170f223644484b846364bcdaea17a110c9a689ac05947b61cc0825e6326aca.jpg)

![8efd6140aea8463a6cc4fc2f8b2a9bdd15407737ccd62fa333f838009ee1f1e6.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/images/8efd6140aea8463a6cc4fc2f8b2a9bdd15407737ccd62fa333f838009ee1f1e6.jpg)

![917e385d9820534b4ee1b3f1076b494ca1c00e86445d7a362f9e65b7187486a5.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/images/917e385d9820534b4ee1b3f1076b494ca1c00e86445d7a362f9e65b7187486a5.jpg)

![b50b9358d46aefabf5cf817482eb7287fb70075c306b0bbfcfb8ee1c5f5776b5.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/images/b50b9358d46aefabf5cf817482eb7287fb70075c306b0bbfcfb8ee1c5f5776b5.jpg)

![d242cd058b26ffe03ed565f825624a15e03b97a0ca4dd904939bad8436413dcd.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/images/d242cd058b26ffe03ed565f825624a15e03b97a0ca4dd904939bad8436413dcd.jpg)

![d83b7f4e16ae713bbcf36469788b789fb7c2cc230bad437e109028d67229fc62.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/images/d83b7f4e16ae713bbcf36469788b789fb7c2cc230bad437e109028d67229fc62.jpg)

![e1ff7eb2f8eeecebb46d05c39c046f860af5eac166571a09c7dfa7abc0472458.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/images/e1ff7eb2f8eeecebb46d05c39c046f860af5eac166571a09c7dfa7abc0472458.jpg)

![ed2e600746cf5832bdb8fb555bbb1f80b6e3cee3667a34e7378f1f68104a0334.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/images/ed2e600746cf5832bdb8fb555bbb1f80b6e3cee3667a34e7378f1f68104a0334.jpg)

![fa7158c7f18eb281ff7c2fdfe1c8132b1021e557bf91a676e5523b98282cd8dc.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/images/fa7158c7f18eb281ff7c2fdfe1c8132b1021e557bf91a676e5523b98282cd8dc.jpg)

### Tables

![13eaffeb3500a74ff09b456ced5f5d6b78e7aea428405f463ec9bb75d952505f.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/tables/13eaffeb3500a74ff09b456ced5f5d6b78e7aea428405f463ec9bb75d952505f.jpg)

![181ba3f48894aca103f9a7f0171001fdc19aa29b73af844e76f9aa0022086a72.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/tables/181ba3f48894aca103f9a7f0171001fdc19aa29b73af844e76f9aa0022086a72.jpg)

![199cf69b4bc4e6fa8f61be90f3b281088a169086af1e9afbcb24277ec9bd038c.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/tables/199cf69b4bc4e6fa8f61be90f3b281088a169086af1e9afbcb24277ec9bd038c.jpg)

![1dff55b96463ebf0192dc417c09c413cba6032fd580274a88eefb9edf6af2af3.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/tables/1dff55b96463ebf0192dc417c09c413cba6032fd580274a88eefb9edf6af2af3.jpg)

![2576e743044984f14b0b5c5c3ae4467396f7453997de321d683bce1836e35d35.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/tables/2576e743044984f14b0b5c5c3ae4467396f7453997de321d683bce1836e35d35.jpg)

![4d0a8ae4e8d9f98872cb8e545ff217a890dad5569e934c14dd59c04644412481.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/tables/4d0a8ae4e8d9f98872cb8e545ff217a890dad5569e934c14dd59c04644412481.jpg)

![52241acec882b2dcb301da25f6ad0dc0cf07856396e8ccd3e690a3aa1bd8fe17.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/tables/52241acec882b2dcb301da25f6ad0dc0cf07856396e8ccd3e690a3aa1bd8fe17.jpg)

![80e867616a552d201397894aa47fa5f47055861782d1accb85c1f5881c71644c.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/tables/80e867616a552d201397894aa47fa5f47055861782d1accb85c1f5881c71644c.jpg)

![8394cf6487bf23fe47696298bded464576361019adbcd3128d3cc4b1dab0c73b.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/tables/8394cf6487bf23fe47696298bded464576361019adbcd3128d3cc4b1dab0c73b.jpg)

![acd3838b49241e8673ab937ef58eadf1d541590d0d66c049350a05c646a6745c.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/tables/acd3838b49241e8673ab937ef58eadf1d541590d0d66c049350a05c646a6745c.jpg)

![b71af713c2bb91274c0a74c327e104aeacb51a44a30c137e1a524b4515d0cfa1.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/tables/b71af713c2bb91274c0a74c327e104aeacb51a44a30c137e1a524b4515d0cfa1.jpg)

![bf2276a79c6abee3a563a60cda7ea3a75685dfa15ab53b08a4afc1075c4cee94.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/tables/bf2276a79c6abee3a563a60cda7ea3a75685dfa15ab53b08a4afc1075c4cee94.jpg)

![ca61af4efdcd705fb88c521247667e5985137bab04c59b06d631234409ae84da.jpg](../iclr_results/1647_On%20Evaluating%20the%20Durability%20of%20Safeguards%20for%20Open-Weight%20LLMs/tables/ca61af4efdcd705fb88c521247667e5985137bab04c59b06d631234409ae84da.jpg)

## GReaTer: Gradients Over Reasoning Makes Smaller Language Models Strong Prompt Optimizers


### Images

![37e312b51b44e452f3d3e5f56d1c2b316c7848d28cbc642ea8eecb295fdd6912.jpg](../iclr_results/1648_GReaTer_%20Gradients%20Over%20Reasoning%20Makes%20Smaller%20Language%20Models%20Strong%20Prompt%20Optimizers/images/37e312b51b44e452f3d3e5f56d1c2b316c7848d28cbc642ea8eecb295fdd6912.jpg)

![3ca5dfbe5de82ef94322aa012e34e4442c2b9e68c590038a519d5efd511f77d2.jpg](../iclr_results/1648_GReaTer_%20Gradients%20Over%20Reasoning%20Makes%20Smaller%20Language%20Models%20Strong%20Prompt%20Optimizers/images/3ca5dfbe5de82ef94322aa012e34e4442c2b9e68c590038a519d5efd511f77d2.jpg)

![5d9a89355c0ad1ed9d1fbbbdf8cac36bb9e0d98212fb946b5d282a42da5f2639.jpg](../iclr_results/1648_GReaTer_%20Gradients%20Over%20Reasoning%20Makes%20Smaller%20Language%20Models%20Strong%20Prompt%20Optimizers/images/5d9a89355c0ad1ed9d1fbbbdf8cac36bb9e0d98212fb946b5d282a42da5f2639.jpg)

![63cf917f46a44379d1f35f5bada666b301dbf7b27d3d148abd624784f33cb3bc.jpg](../iclr_results/1648_GReaTer_%20Gradients%20Over%20Reasoning%20Makes%20Smaller%20Language%20Models%20Strong%20Prompt%20Optimizers/images/63cf917f46a44379d1f35f5bada666b301dbf7b27d3d148abd624784f33cb3bc.jpg)

![79728be71bbc226d2d617e910c074715a800663812a81819ee76d6ed670becb6.jpg](../iclr_results/1648_GReaTer_%20Gradients%20Over%20Reasoning%20Makes%20Smaller%20Language%20Models%20Strong%20Prompt%20Optimizers/images/79728be71bbc226d2d617e910c074715a800663812a81819ee76d6ed670becb6.jpg)

![9471b23db36010c4b4e3522486b8eeb5b3c37fb133c27ca467d4b0eea30bd163.jpg](../iclr_results/1648_GReaTer_%20Gradients%20Over%20Reasoning%20Makes%20Smaller%20Language%20Models%20Strong%20Prompt%20Optimizers/images/9471b23db36010c4b4e3522486b8eeb5b3c37fb133c27ca467d4b0eea30bd163.jpg)

![a6e76875e4f4571a8417904871c15df91dd84a23d315e5349450745becd4efff.jpg](../iclr_results/1648_GReaTer_%20Gradients%20Over%20Reasoning%20Makes%20Smaller%20Language%20Models%20Strong%20Prompt%20Optimizers/images/a6e76875e4f4571a8417904871c15df91dd84a23d315e5349450745becd4efff.jpg)

![c993c2ec912a384ef37a0945bae0d2058f050b107a54518252534755745fbf8c.jpg](../iclr_results/1648_GReaTer_%20Gradients%20Over%20Reasoning%20Makes%20Smaller%20Language%20Models%20Strong%20Prompt%20Optimizers/images/c993c2ec912a384ef37a0945bae0d2058f050b107a54518252534755745fbf8c.jpg)

### Tables

![05b421feae384412bee52dc19a2bcf7570a51afcaad8607533963ef809a78309.jpg](../iclr_results/1648_GReaTer_%20Gradients%20Over%20Reasoning%20Makes%20Smaller%20Language%20Models%20Strong%20Prompt%20Optimizers/tables/05b421feae384412bee52dc19a2bcf7570a51afcaad8607533963ef809a78309.jpg)

![07364988a36c635848fbd8e4a4b837e1daa8eaa60faa3293ad04c9172c74a84d.jpg](../iclr_results/1648_GReaTer_%20Gradients%20Over%20Reasoning%20Makes%20Smaller%20Language%20Models%20Strong%20Prompt%20Optimizers/tables/07364988a36c635848fbd8e4a4b837e1daa8eaa60faa3293ad04c9172c74a84d.jpg)

![0ae74b35fa68d7c62e9b90a71c95d081abdd8b632ba16a42bc5db742ea30df65.jpg](../iclr_results/1648_GReaTer_%20Gradients%20Over%20Reasoning%20Makes%20Smaller%20Language%20Models%20Strong%20Prompt%20Optimizers/tables/0ae74b35fa68d7c62e9b90a71c95d081abdd8b632ba16a42bc5db742ea30df65.jpg)

![166da18dd986414f81ce2271a18a210d4635bcc830be0b25cef8c9d316780a4d.jpg](../iclr_results/1648_GReaTer_%20Gradients%20Over%20Reasoning%20Makes%20Smaller%20Language%20Models%20Strong%20Prompt%20Optimizers/tables/166da18dd986414f81ce2271a18a210d4635bcc830be0b25cef8c9d316780a4d.jpg)

![200b1daa29ac31449fdf2730ce462886de3b039df6d5dced41be1c605c670589.jpg](../iclr_results/1648_GReaTer_%20Gradients%20Over%20Reasoning%20Makes%20Smaller%20Language%20Models%20Strong%20Prompt%20Optimizers/tables/200b1daa29ac31449fdf2730ce462886de3b039df6d5dced41be1c605c670589.jpg)

![29161618a6146ed5e00e26f577a0424569afa5eec232a20abf81cf4f24a2408d.jpg](../iclr_results/1648_GReaTer_%20Gradients%20Over%20Reasoning%20Makes%20Smaller%20Language%20Models%20Strong%20Prompt%20Optimizers/tables/29161618a6146ed5e00e26f577a0424569afa5eec232a20abf81cf4f24a2408d.jpg)

![3140fca1653df00563006fe5d9df90f34dca7af9427926f46235fd695eb7c02a.jpg](../iclr_results/1648_GReaTer_%20Gradients%20Over%20Reasoning%20Makes%20Smaller%20Language%20Models%20Strong%20Prompt%20Optimizers/tables/3140fca1653df00563006fe5d9df90f34dca7af9427926f46235fd695eb7c02a.jpg)

![3fa170716c161819425188c45740beeb46444d6e37bfe5066d526d28888bd16e.jpg](../iclr_results/1648_GReaTer_%20Gradients%20Over%20Reasoning%20Makes%20Smaller%20Language%20Models%20Strong%20Prompt%20Optimizers/tables/3fa170716c161819425188c45740beeb46444d6e37bfe5066d526d28888bd16e.jpg)

![42ca547f2cd9f23ac1f71364d7ab11730f38129a42fe996cad015fa8ae73c167.jpg](../iclr_results/1648_GReaTer_%20Gradients%20Over%20Reasoning%20Makes%20Smaller%20Language%20Models%20Strong%20Prompt%20Optimizers/tables/42ca547f2cd9f23ac1f71364d7ab11730f38129a42fe996cad015fa8ae73c167.jpg)

![4f798a77dc4b8f944abdc4256eafdf4a634fa44245875fad56fd4b23dd90880a.jpg](../iclr_results/1648_GReaTer_%20Gradients%20Over%20Reasoning%20Makes%20Smaller%20Language%20Models%20Strong%20Prompt%20Optimizers/tables/4f798a77dc4b8f944abdc4256eafdf4a634fa44245875fad56fd4b23dd90880a.jpg)

![8a4961315dfab1ae33aa846dd1c967fed4eff176d23906df8b025f37b32b7dc0.jpg](../iclr_results/1648_GReaTer_%20Gradients%20Over%20Reasoning%20Makes%20Smaller%20Language%20Models%20Strong%20Prompt%20Optimizers/tables/8a4961315dfab1ae33aa846dd1c967fed4eff176d23906df8b025f37b32b7dc0.jpg)

![95a6aabac7b3ca1efe64ff6ffce549f6211f817c26d12d1eeb559011a8ff7cd3.jpg](../iclr_results/1648_GReaTer_%20Gradients%20Over%20Reasoning%20Makes%20Smaller%20Language%20Models%20Strong%20Prompt%20Optimizers/tables/95a6aabac7b3ca1efe64ff6ffce549f6211f817c26d12d1eeb559011a8ff7cd3.jpg)

![a794db01deba96c3f0744a2bc31489b65b0cdf37f8a6cd5bb22a33938c10d098.jpg](../iclr_results/1648_GReaTer_%20Gradients%20Over%20Reasoning%20Makes%20Smaller%20Language%20Models%20Strong%20Prompt%20Optimizers/tables/a794db01deba96c3f0744a2bc31489b65b0cdf37f8a6cd5bb22a33938c10d098.jpg)

![beccb680b6d8b04749dd7cd39eff71de9819276c715ddcd53975234a1a33befd.jpg](../iclr_results/1648_GReaTer_%20Gradients%20Over%20Reasoning%20Makes%20Smaller%20Language%20Models%20Strong%20Prompt%20Optimizers/tables/beccb680b6d8b04749dd7cd39eff71de9819276c715ddcd53975234a1a33befd.jpg)

![f7a318cbd30069cf4524078b5db81d5890a0990193b595c5db2063357ccbdd5d.jpg](../iclr_results/1648_GReaTer_%20Gradients%20Over%20Reasoning%20Makes%20Smaller%20Language%20Models%20Strong%20Prompt%20Optimizers/tables/f7a318cbd30069cf4524078b5db81d5890a0990193b595c5db2063357ccbdd5d.jpg)

## Looking Backward: Retrospective Backward Synthesis for Goal-Conditioned GFlowNets


### Images

![1f68d4a9a6f336392ad19004d44421e90654554dcf11c23716bde4d3fbbd2611.jpg](../iclr_results/1649_Looking%20Backward_%20Retrospective%20Backward%20Synthesis%20for%20Goal-Conditioned%20GFlowNets/images/1f68d4a9a6f336392ad19004d44421e90654554dcf11c23716bde4d3fbbd2611.jpg)

![20b4ba07ff538462e2c7ca4a3d131a080089418b855abd30dff09594439ab6a8.jpg](../iclr_results/1649_Looking%20Backward_%20Retrospective%20Backward%20Synthesis%20for%20Goal-Conditioned%20GFlowNets/images/20b4ba07ff538462e2c7ca4a3d131a080089418b855abd30dff09594439ab6a8.jpg)

![265e092f9b8fb4b35338fd743c32f155edfe9dfe34f9ced4f5a254425304afd9.jpg](../iclr_results/1649_Looking%20Backward_%20Retrospective%20Backward%20Synthesis%20for%20Goal-Conditioned%20GFlowNets/images/265e092f9b8fb4b35338fd743c32f155edfe9dfe34f9ced4f5a254425304afd9.jpg)

![27780a2b204fab8ef8f20cdbca89d8723877c7d0670eb0d4ce220f29370ea4cc.jpg](../iclr_results/1649_Looking%20Backward_%20Retrospective%20Backward%20Synthesis%20for%20Goal-Conditioned%20GFlowNets/images/27780a2b204fab8ef8f20cdbca89d8723877c7d0670eb0d4ce220f29370ea4cc.jpg)

![48eb9ddce9cd206f4a8952962f44b25951d2b638f17984426a5eb9fc884603de.jpg](../iclr_results/1649_Looking%20Backward_%20Retrospective%20Backward%20Synthesis%20for%20Goal-Conditioned%20GFlowNets/images/48eb9ddce9cd206f4a8952962f44b25951d2b638f17984426a5eb9fc884603de.jpg)

![5f12a48adb66ecf2ed5e1e8b568fc499fe394fbd38bb2bc6aa1d8b95ae7575db.jpg](../iclr_results/1649_Looking%20Backward_%20Retrospective%20Backward%20Synthesis%20for%20Goal-Conditioned%20GFlowNets/images/5f12a48adb66ecf2ed5e1e8b568fc499fe394fbd38bb2bc6aa1d8b95ae7575db.jpg)

![615100987080807610e857dfb53372b62fa9a1394f1f5692a89bea97f2ebab42.jpg](../iclr_results/1649_Looking%20Backward_%20Retrospective%20Backward%20Synthesis%20for%20Goal-Conditioned%20GFlowNets/images/615100987080807610e857dfb53372b62fa9a1394f1f5692a89bea97f2ebab42.jpg)

![6c42e5810308da19b90690eec5b89137b1704cd3240110c24d5285dd1997a444.jpg](../iclr_results/1649_Looking%20Backward_%20Retrospective%20Backward%20Synthesis%20for%20Goal-Conditioned%20GFlowNets/images/6c42e5810308da19b90690eec5b89137b1704cd3240110c24d5285dd1997a444.jpg)

![6e6b9e8445478be6ac4ffa29fa5ba7743eb329e3f03b6ddb0e4930a38ffc2093.jpg](../iclr_results/1649_Looking%20Backward_%20Retrospective%20Backward%20Synthesis%20for%20Goal-Conditioned%20GFlowNets/images/6e6b9e8445478be6ac4ffa29fa5ba7743eb329e3f03b6ddb0e4930a38ffc2093.jpg)

![70cf05806a2ac0b9ce73bfae554958d6305b4ce106dc43505fe0b9f452f42459.jpg](../iclr_results/1649_Looking%20Backward_%20Retrospective%20Backward%20Synthesis%20for%20Goal-Conditioned%20GFlowNets/images/70cf05806a2ac0b9ce73bfae554958d6305b4ce106dc43505fe0b9f452f42459.jpg)

![998fb18a231837d863e7f87c5a5fe8301d5900d23e41be010a2e22d0c2fbcdf6.jpg](../iclr_results/1649_Looking%20Backward_%20Retrospective%20Backward%20Synthesis%20for%20Goal-Conditioned%20GFlowNets/images/998fb18a231837d863e7f87c5a5fe8301d5900d23e41be010a2e22d0c2fbcdf6.jpg)

![9c7169b2e1ae37e2cdd3ba6999eafd424593a7fd87f660baaf14ed984eca966b.jpg](../iclr_results/1649_Looking%20Backward_%20Retrospective%20Backward%20Synthesis%20for%20Goal-Conditioned%20GFlowNets/images/9c7169b2e1ae37e2cdd3ba6999eafd424593a7fd87f660baaf14ed984eca966b.jpg)

![b53d3bd68d9701eb755872637c9fe061c0cb248df181a83650005eafa1d439f0.jpg](../iclr_results/1649_Looking%20Backward_%20Retrospective%20Backward%20Synthesis%20for%20Goal-Conditioned%20GFlowNets/images/b53d3bd68d9701eb755872637c9fe061c0cb248df181a83650005eafa1d439f0.jpg)

![b5fe44b0c766a28f1d341be1647293b618934dd45a1e7c14fb0b67d311385665.jpg](../iclr_results/1649_Looking%20Backward_%20Retrospective%20Backward%20Synthesis%20for%20Goal-Conditioned%20GFlowNets/images/b5fe44b0c766a28f1d341be1647293b618934dd45a1e7c14fb0b67d311385665.jpg)

![c717635263221d4038d43182fad15bc070abde87e8bc727ebb80e81ec7d0d4f7.jpg](../iclr_results/1649_Looking%20Backward_%20Retrospective%20Backward%20Synthesis%20for%20Goal-Conditioned%20GFlowNets/images/c717635263221d4038d43182fad15bc070abde87e8bc727ebb80e81ec7d0d4f7.jpg)

![cd24f31d673cfbc4823faa9e03d4d5e25fb8341c57e2303fc382769076f421c5.jpg](../iclr_results/1649_Looking%20Backward_%20Retrospective%20Backward%20Synthesis%20for%20Goal-Conditioned%20GFlowNets/images/cd24f31d673cfbc4823faa9e03d4d5e25fb8341c57e2303fc382769076f421c5.jpg)

![f7d7081ae95b4ae7c866cd5a2e03dd8cb8e7a15fa6e659e061e1c5a8da542b8b.jpg](../iclr_results/1649_Looking%20Backward_%20Retrospective%20Backward%20Synthesis%20for%20Goal-Conditioned%20GFlowNets/images/f7d7081ae95b4ae7c866cd5a2e03dd8cb8e7a15fa6e659e061e1c5a8da542b8b.jpg)

![fa3c91290965e8f791b7a165479a437eb5489bbc42d0ba4cfedb2b1400a8aca0.jpg](../iclr_results/1649_Looking%20Backward_%20Retrospective%20Backward%20Synthesis%20for%20Goal-Conditioned%20GFlowNets/images/fa3c91290965e8f791b7a165479a437eb5489bbc42d0ba4cfedb2b1400a8aca0.jpg)

## When LLMs Play the Telephone Game: Cultural Attractors as Conceptual Tools to Evaluate LLMs in Multi-turn Settings


### Images

![00dcbd0272a184bde663482d813fe2b9c563a43c48f025a06a4293b97a4c53c7.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/00dcbd0272a184bde663482d813fe2b9c563a43c48f025a06a4293b97a4c53c7.jpg)

![055ac050a9827c24526178e62aee21f716a06234b3a55334b9d337764a0a7313.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/055ac050a9827c24526178e62aee21f716a06234b3a55334b9d337764a0a7313.jpg)

![0703935236c5361e148ba513a2c084d7ce0469f35ce3a9e22432977c1ba9ec9d.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/0703935236c5361e148ba513a2c084d7ce0469f35ce3a9e22432977c1ba9ec9d.jpg)

![1a699af8c358e6ee80d5dfd80af6179d8381b9bf02c52ce0505958189742b06e.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/1a699af8c358e6ee80d5dfd80af6179d8381b9bf02c52ce0505958189742b06e.jpg)

![220580e726de59bbb7a8dc95e6bbdae3380999adbfacd5732b44f2c9aa317395.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/220580e726de59bbb7a8dc95e6bbdae3380999adbfacd5732b44f2c9aa317395.jpg)

![2376b7f973effd4fbc76a221b9206e0cd08ea0066c1d78a75f81513a18efb888.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/2376b7f973effd4fbc76a221b9206e0cd08ea0066c1d78a75f81513a18efb888.jpg)

![2fb1bd1d1b6cc22c772dc4c8131c5c49c8376054669be89c4eba95e04928a613.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/2fb1bd1d1b6cc22c772dc4c8131c5c49c8376054669be89c4eba95e04928a613.jpg)

![3f117424cf070f17a222ebd1e281aefef810255480037d9562dfb41895bad8a6.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/3f117424cf070f17a222ebd1e281aefef810255480037d9562dfb41895bad8a6.jpg)

![467a69ce3f70718fb1a4bd94523c6ea7ac274409e46b1a86879c5ce2512ca149.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/467a69ce3f70718fb1a4bd94523c6ea7ac274409e46b1a86879c5ce2512ca149.jpg)

![46fedf15ecdb9f2bda0ac3ed2df7ba8d82e07b6857a0ef8b6c5bd1410cfd32f7.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/46fedf15ecdb9f2bda0ac3ed2df7ba8d82e07b6857a0ef8b6c5bd1410cfd32f7.jpg)

![5e85762eb6a3d672fec2fa818d07cb099be23df114f7d5de685271922f21607c.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/5e85762eb6a3d672fec2fa818d07cb099be23df114f7d5de685271922f21607c.jpg)

![607eeb7daddf9b2ea0cfdd19843fdb6759b395d12865508fd6902a954db74de3.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/607eeb7daddf9b2ea0cfdd19843fdb6759b395d12865508fd6902a954db74de3.jpg)

![615e8d36ad5e2c3d10ead5c618b31b868d8878ba59ba19cd5e9c05e80085d34c.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/615e8d36ad5e2c3d10ead5c618b31b868d8878ba59ba19cd5e9c05e80085d34c.jpg)

![66bf4bee8a92566c8b5eb81629e7f4987afb4d8045b4c614ee938c3a20a1533d.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/66bf4bee8a92566c8b5eb81629e7f4987afb4d8045b4c614ee938c3a20a1533d.jpg)

![693f3cf2e7856f6f3bdca52d63728299692d2956b0cb5b097923d65ae2c2e299.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/693f3cf2e7856f6f3bdca52d63728299692d2956b0cb5b097923d65ae2c2e299.jpg)

![6f47d38b0545495f279cff784117f1d15cdf68ae72aa7e5e4a698e6f38289169.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/6f47d38b0545495f279cff784117f1d15cdf68ae72aa7e5e4a698e6f38289169.jpg)

![80a3037dde16c1c4af5654a6200f656146e2b34ad4d100f55ce818adcd05eb8e.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/80a3037dde16c1c4af5654a6200f656146e2b34ad4d100f55ce818adcd05eb8e.jpg)

![84a286d7383bd2af6079c6585c9a3b2214a1e5f4d3e56e4d7c532e8801d8ba8d.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/84a286d7383bd2af6079c6585c9a3b2214a1e5f4d3e56e4d7c532e8801d8ba8d.jpg)

![850a95af29182703129cf97a7d0575f215e98b6979866084e1289a1931a265ed.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/850a95af29182703129cf97a7d0575f215e98b6979866084e1289a1931a265ed.jpg)

![864a882eab2755343e92fc0b22fb6ee3eadf657c272fbc002489a7c06638b48e.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/864a882eab2755343e92fc0b22fb6ee3eadf657c272fbc002489a7c06638b48e.jpg)

![918ab44a2ce7b7fa1d0b564cd1f252a108d12d6e354a1b4f0d166f4adb8683d0.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/918ab44a2ce7b7fa1d0b564cd1f252a108d12d6e354a1b4f0d166f4adb8683d0.jpg)

![9255d2cb189d4bdaebdde36b456fa85973e623868e3d761ac3136792297d7b31.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/9255d2cb189d4bdaebdde36b456fa85973e623868e3d761ac3136792297d7b31.jpg)

![92d72bce3ae94e133c509cd71590a2bfe4593ac853fd030204ae2244ccdf4f82.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/92d72bce3ae94e133c509cd71590a2bfe4593ac853fd030204ae2244ccdf4f82.jpg)

![97b52f599705c313a4f6803c618e9ec27e5f5558cfe7316fa42aa6bf8c92f1a0.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/97b52f599705c313a4f6803c618e9ec27e5f5558cfe7316fa42aa6bf8c92f1a0.jpg)

![aa35165d6c8f6f7c5a392ac6f994a209e51bf67ba6dc9ac119248124f8149459.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/aa35165d6c8f6f7c5a392ac6f994a209e51bf67ba6dc9ac119248124f8149459.jpg)

![aaa3aa107e026825c7c117e3dd17230c697cab26dad150b06a12b962bc20018e.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/aaa3aa107e026825c7c117e3dd17230c697cab26dad150b06a12b962bc20018e.jpg)

![afdc7ffbb01a369d795e3c7a243600865a6447f9a30bd63d5e42dec8c3ac2cf1.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/afdc7ffbb01a369d795e3c7a243600865a6447f9a30bd63d5e42dec8c3ac2cf1.jpg)

![b2d67ca5349dab924b6632c998d47b312358dd7a679e70d00ea304d9951170c3.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/b2d67ca5349dab924b6632c998d47b312358dd7a679e70d00ea304d9951170c3.jpg)

![b530b2a53a3e3c3675461a659dd44c5a9b300250f768a8348135c751e4c9ac1b.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/b530b2a53a3e3c3675461a659dd44c5a9b300250f768a8348135c751e4c9ac1b.jpg)

![b91b900b53e5a1ba7162816cf685cc5a0cd52a97915c94f6b2aed3a80e06cd4d.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/b91b900b53e5a1ba7162816cf685cc5a0cd52a97915c94f6b2aed3a80e06cd4d.jpg)

![c7efe549543fa35903d55b9111094644c79e5f115808c1844a418c3c48a1cbdb.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/c7efe549543fa35903d55b9111094644c79e5f115808c1844a418c3c48a1cbdb.jpg)

![df312cb9ee021adc815550310f68409676e38e3a0b084fc2777e4a18bc89a189.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/df312cb9ee021adc815550310f68409676e38e3a0b084fc2777e4a18bc89a189.jpg)

![f5ddd211317c6c6041dc306d606f12d14e3e9299be4edc31bd9e2bdf04ddbf2f.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/images/f5ddd211317c6c6041dc306d606f12d14e3e9299be4edc31bd9e2bdf04ddbf2f.jpg)

### Tables

![2f18f5865bb2a25d9b9e107413bfb1dc82c34428160d397f70879d9ab0d6e4f0.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/tables/2f18f5865bb2a25d9b9e107413bfb1dc82c34428160d397f70879d9ab0d6e4f0.jpg)

![38a3456044c80bf97bfb54e31e11391d4319f1f24d99094d70fbabeb66c27079.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/tables/38a3456044c80bf97bfb54e31e11391d4319f1f24d99094d70fbabeb66c27079.jpg)

![4b8925f1cdc4e336e830bbf6c509c140a5d7d6200cf66d1e68bf6b88b4d922c4.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/tables/4b8925f1cdc4e336e830bbf6c509c140a5d7d6200cf66d1e68bf6b88b4d922c4.jpg)

![57b53cc1e0f024ad0fc4927ee5b13140587a9169e9f549add0cb8975d4728abd.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/tables/57b53cc1e0f024ad0fc4927ee5b13140587a9169e9f549add0cb8975d4728abd.jpg)

![612c6640021ab6cf3185f7aedd2268c67003565f6ee9ec5dc3855bbdd805cd5b.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/tables/612c6640021ab6cf3185f7aedd2268c67003565f6ee9ec5dc3855bbdd805cd5b.jpg)

![61a9864f3b3cb01798b64945aa6632d782e7bd6e8cc3f9d3d522970e7c6d4f9f.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/tables/61a9864f3b3cb01798b64945aa6632d782e7bd6e8cc3f9d3d522970e7c6d4f9f.jpg)

![803a94c4198e7021b437dea5b41e4fb484cba0a898505aa42ae949402f10a6db.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/tables/803a94c4198e7021b437dea5b41e4fb484cba0a898505aa42ae949402f10a6db.jpg)

![8527dec6cf021715a5a9ceede239ac11f0366cad31dc40f6f57226a0973aae68.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/tables/8527dec6cf021715a5a9ceede239ac11f0366cad31dc40f6f57226a0973aae68.jpg)

![897c344532298b4452df871c56429b0cc6bc12f2da47ac224241382ec8952832.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/tables/897c344532298b4452df871c56429b0cc6bc12f2da47ac224241382ec8952832.jpg)

![8983d53d2a7f46aabd5ea2cf168f718c396643df162b3588ea56b379351553be.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/tables/8983d53d2a7f46aabd5ea2cf168f718c396643df162b3588ea56b379351553be.jpg)

![a47e6ede83d996481b202aabe4542b017ba63a4dc2d78475762ff045341414a7.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/tables/a47e6ede83d996481b202aabe4542b017ba63a4dc2d78475762ff045341414a7.jpg)

![e11a7da75c40a534616256bffdeaab48d3be8bac2fe6f2e6a11484546918fa60.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/tables/e11a7da75c40a534616256bffdeaab48d3be8bac2fe6f2e6a11484546918fa60.jpg)

![f3dcd45adbaba2e39902ee2abf48dad2fc7d14deca4a165e55fca8d1e9ee9c92.jpg](../iclr_results/1650_When%20LLMs%20Play%20the%20Telephone%20Game_%20Cultural%20Attractors%20as%20Conceptual%20Tools%20to%20Evaluate%20LLMs%20in%20Multi/tables/f3dcd45adbaba2e39902ee2abf48dad2fc7d14deca4a165e55fca8d1e9ee9c92.jpg)

## LongMamba: Enhancing Mamba's Long-Context Capabilities via Training-Free Receptive Field Enlargement


### Images

![41d60e73b3c6a3c772947aff0046744b4c96a0e61e598397d610988c4bbf85f5.jpg](../iclr_results/1651_LongMamba_%20Enhancing%20Mamba%27s%20Long-Context%20Capabilities%20via%20Training-Free%20Receptive%20Field%20Enlargement/images/41d60e73b3c6a3c772947aff0046744b4c96a0e61e598397d610988c4bbf85f5.jpg)

![6950d419298668a84dbb61d2141a62c8a26c1ed1bdad17f7128cf8447978e0ac.jpg](../iclr_results/1651_LongMamba_%20Enhancing%20Mamba%27s%20Long-Context%20Capabilities%20via%20Training-Free%20Receptive%20Field%20Enlargement/images/6950d419298668a84dbb61d2141a62c8a26c1ed1bdad17f7128cf8447978e0ac.jpg)

![91bef7ad11a42a508b0c4b9b7db3806db9d01e1b7d4c5e3659911b4560c6b7e9.jpg](../iclr_results/1651_LongMamba_%20Enhancing%20Mamba%27s%20Long-Context%20Capabilities%20via%20Training-Free%20Receptive%20Field%20Enlargement/images/91bef7ad11a42a508b0c4b9b7db3806db9d01e1b7d4c5e3659911b4560c6b7e9.jpg)

![99b33cd22a6486d436eb287e88b8d484fa325c9824bdf8745ea377fad2fd8dc4.jpg](../iclr_results/1651_LongMamba_%20Enhancing%20Mamba%27s%20Long-Context%20Capabilities%20via%20Training-Free%20Receptive%20Field%20Enlargement/images/99b33cd22a6486d436eb287e88b8d484fa325c9824bdf8745ea377fad2fd8dc4.jpg)

![9e648f0002cc1bc2a7f40af6c1206e5896483a119eaf816162bdc3ebeda24241.jpg](../iclr_results/1651_LongMamba_%20Enhancing%20Mamba%27s%20Long-Context%20Capabilities%20via%20Training-Free%20Receptive%20Field%20Enlargement/images/9e648f0002cc1bc2a7f40af6c1206e5896483a119eaf816162bdc3ebeda24241.jpg)

![f80d229da5686810f15a1078831a307f108caf6bd99366e50acb9f3399b88955.jpg](../iclr_results/1651_LongMamba_%20Enhancing%20Mamba%27s%20Long-Context%20Capabilities%20via%20Training-Free%20Receptive%20Field%20Enlargement/images/f80d229da5686810f15a1078831a307f108caf6bd99366e50acb9f3399b88955.jpg)

### Tables

![22dcf22362314882050623efa9b9861c063a8c576a3bad6b260f54d20601f418.jpg](../iclr_results/1651_LongMamba_%20Enhancing%20Mamba%27s%20Long-Context%20Capabilities%20via%20Training-Free%20Receptive%20Field%20Enlargement/tables/22dcf22362314882050623efa9b9861c063a8c576a3bad6b260f54d20601f418.jpg)

![2d5e4220c4273142d36663543aff89ebc5f115153b19ae43d7fa940bed964727.jpg](../iclr_results/1651_LongMamba_%20Enhancing%20Mamba%27s%20Long-Context%20Capabilities%20via%20Training-Free%20Receptive%20Field%20Enlargement/tables/2d5e4220c4273142d36663543aff89ebc5f115153b19ae43d7fa940bed964727.jpg)

![466d1dc6d3df4c63c1909a25ee7b40c892453133c6ef4f144771ba360d2ad4b3.jpg](../iclr_results/1651_LongMamba_%20Enhancing%20Mamba%27s%20Long-Context%20Capabilities%20via%20Training-Free%20Receptive%20Field%20Enlargement/tables/466d1dc6d3df4c63c1909a25ee7b40c892453133c6ef4f144771ba360d2ad4b3.jpg)

![510f171894383bd156d79ca9aa94969cdd6b4d44f8ced163b3258699e30950da.jpg](../iclr_results/1651_LongMamba_%20Enhancing%20Mamba%27s%20Long-Context%20Capabilities%20via%20Training-Free%20Receptive%20Field%20Enlargement/tables/510f171894383bd156d79ca9aa94969cdd6b4d44f8ced163b3258699e30950da.jpg)

![c0874340082547f88e90d41c76797db8ca35a015a07e5db76c0c3fa310bfd18b.jpg](../iclr_results/1651_LongMamba_%20Enhancing%20Mamba%27s%20Long-Context%20Capabilities%20via%20Training-Free%20Receptive%20Field%20Enlargement/tables/c0874340082547f88e90d41c76797db8ca35a015a07e5db76c0c3fa310bfd18b.jpg)

![f9e1e1d2b804bd4458462d254ba456e3f1d06a2492986ea3a920e10d22bd5437.jpg](../iclr_results/1651_LongMamba_%20Enhancing%20Mamba%27s%20Long-Context%20Capabilities%20via%20Training-Free%20Receptive%20Field%20Enlargement/tables/f9e1e1d2b804bd4458462d254ba456e3f1d06a2492986ea3a920e10d22bd5437.jpg)

## Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning


### Images

![25bac0e36acf1b6043d7f7f7e151cf868e780f930bcaa4ae62cc8290d5f934be.jpg](../iclr_results/1652_Unlearning%20or%20Obfuscating_%20Jogging%20the%20Memory%20of%20Unlearned%20LLMs%20via%20Benign%20Relearning/images/25bac0e36acf1b6043d7f7f7e151cf868e780f930bcaa4ae62cc8290d5f934be.jpg)

![7c54a2a348ac832e04618778ab98613c0a43c68cb0a49ca9552256eacae40ea8.jpg](../iclr_results/1652_Unlearning%20or%20Obfuscating_%20Jogging%20the%20Memory%20of%20Unlearned%20LLMs%20via%20Benign%20Relearning/images/7c54a2a348ac832e04618778ab98613c0a43c68cb0a49ca9552256eacae40ea8.jpg)

![938b736d235da10f6176f49137717973cacf09c34b5cd935a175e6bd17ab2461.jpg](../iclr_results/1652_Unlearning%20or%20Obfuscating_%20Jogging%20the%20Memory%20of%20Unlearned%20LLMs%20via%20Benign%20Relearning/images/938b736d235da10f6176f49137717973cacf09c34b5cd935a175e6bd17ab2461.jpg)

![a600e5246cc600d1908a8e0b721de8519c56580e84fa958c46a7a78d10c9ada3.jpg](../iclr_results/1652_Unlearning%20or%20Obfuscating_%20Jogging%20the%20Memory%20of%20Unlearned%20LLMs%20via%20Benign%20Relearning/images/a600e5246cc600d1908a8e0b721de8519c56580e84fa958c46a7a78d10c9ada3.jpg)

![c1f6511783b3fa057adb50e8c31b62040c1203ed6e2c9759bffb9276cff51db6.jpg](../iclr_results/1652_Unlearning%20or%20Obfuscating_%20Jogging%20the%20Memory%20of%20Unlearned%20LLMs%20via%20Benign%20Relearning/images/c1f6511783b3fa057adb50e8c31b62040c1203ed6e2c9759bffb9276cff51db6.jpg)

![c4b02bca8af8b37215688cc7946d165a93b3c34c82d3de0467944630e4256b3d.jpg](../iclr_results/1652_Unlearning%20or%20Obfuscating_%20Jogging%20the%20Memory%20of%20Unlearned%20LLMs%20via%20Benign%20Relearning/images/c4b02bca8af8b37215688cc7946d165a93b3c34c82d3de0467944630e4256b3d.jpg)

![cabf8d4796c18c252fa9746ca296650512a0a1200157f0c6b9f7e310783c00c7.jpg](../iclr_results/1652_Unlearning%20or%20Obfuscating_%20Jogging%20the%20Memory%20of%20Unlearned%20LLMs%20via%20Benign%20Relearning/images/cabf8d4796c18c252fa9746ca296650512a0a1200157f0c6b9f7e310783c00c7.jpg)

![eb3dbeae7de14242e033e77dae4a4486d4530d86603c16eb4295061ed047d4e3.jpg](../iclr_results/1652_Unlearning%20or%20Obfuscating_%20Jogging%20the%20Memory%20of%20Unlearned%20LLMs%20via%20Benign%20Relearning/images/eb3dbeae7de14242e033e77dae4a4486d4530d86603c16eb4295061ed047d4e3.jpg)

### Tables

![05a1d5ae3c8f3c536dc5381c71a12f14af7d67209dda7dc170c65cb627c936e6.jpg](../iclr_results/1652_Unlearning%20or%20Obfuscating_%20Jogging%20the%20Memory%20of%20Unlearned%20LLMs%20via%20Benign%20Relearning/tables/05a1d5ae3c8f3c536dc5381c71a12f14af7d67209dda7dc170c65cb627c936e6.jpg)

![068108896219d3caf096f8c86822525bed2f98a24daf1fd9ed56756bd65ff13a.jpg](../iclr_results/1652_Unlearning%20or%20Obfuscating_%20Jogging%20the%20Memory%20of%20Unlearned%20LLMs%20via%20Benign%20Relearning/tables/068108896219d3caf096f8c86822525bed2f98a24daf1fd9ed56756bd65ff13a.jpg)

![09cda8a8a3ab4e55f25ab5e7771635d74393c364d0b11c8adfa9b603b28557d2.jpg](../iclr_results/1652_Unlearning%20or%20Obfuscating_%20Jogging%20the%20Memory%20of%20Unlearned%20LLMs%20via%20Benign%20Relearning/tables/09cda8a8a3ab4e55f25ab5e7771635d74393c364d0b11c8adfa9b603b28557d2.jpg)

![11b4706f8a699f11046fbe48bd91e77acbc85716f727db08e0041276455b338b.jpg](../iclr_results/1652_Unlearning%20or%20Obfuscating_%20Jogging%20the%20Memory%20of%20Unlearned%20LLMs%20via%20Benign%20Relearning/tables/11b4706f8a699f11046fbe48bd91e77acbc85716f727db08e0041276455b338b.jpg)

![19d57c52353b2b7219c07cc92ac6b5bc0ae5a1473d6fd015c58ed36ca7759f81.jpg](../iclr_results/1652_Unlearning%20or%20Obfuscating_%20Jogging%20the%20Memory%20of%20Unlearned%20LLMs%20via%20Benign%20Relearning/tables/19d57c52353b2b7219c07cc92ac6b5bc0ae5a1473d6fd015c58ed36ca7759f81.jpg)

![3f31b484bb400d27a9a285d08524687fddd2dee426497c8ae83f224c25bb86d7.jpg](../iclr_results/1652_Unlearning%20or%20Obfuscating_%20Jogging%20the%20Memory%20of%20Unlearned%20LLMs%20via%20Benign%20Relearning/tables/3f31b484bb400d27a9a285d08524687fddd2dee426497c8ae83f224c25bb86d7.jpg)

![95ff844b7d47407d02608374b5c7238f86216b121ab04ddc56447b8e1d1d2e34.jpg](../iclr_results/1652_Unlearning%20or%20Obfuscating_%20Jogging%20the%20Memory%20of%20Unlearned%20LLMs%20via%20Benign%20Relearning/tables/95ff844b7d47407d02608374b5c7238f86216b121ab04ddc56447b8e1d1d2e34.jpg)

![9afe4583cf81989d5c04ed90608b7c21984dea17bbd4fcc40af9e4599fea827d.jpg](../iclr_results/1652_Unlearning%20or%20Obfuscating_%20Jogging%20the%20Memory%20of%20Unlearned%20LLMs%20via%20Benign%20Relearning/tables/9afe4583cf81989d5c04ed90608b7c21984dea17bbd4fcc40af9e4599fea827d.jpg)

![bbd9c6506e940b787487d71962728bb1f674502fd2293117be11b6f7d5f959f7.jpg](../iclr_results/1652_Unlearning%20or%20Obfuscating_%20Jogging%20the%20Memory%20of%20Unlearned%20LLMs%20via%20Benign%20Relearning/tables/bbd9c6506e940b787487d71962728bb1f674502fd2293117be11b6f7d5f959f7.jpg)

![d80aedd95621ae12662d1257206abfd41a2d4959c6f59cc5c45387939ac8d7cd.jpg](../iclr_results/1652_Unlearning%20or%20Obfuscating_%20Jogging%20the%20Memory%20of%20Unlearned%20LLMs%20via%20Benign%20Relearning/tables/d80aedd95621ae12662d1257206abfd41a2d4959c6f59cc5c45387939ac8d7cd.jpg)

## What is Wrong with Perplexity for Long-context Language Modeling?


### Images

![034c0927ed0c2ee1dd88e9a095c820f29f037909363e0b18a6cdfb7b6c204945.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/images/034c0927ed0c2ee1dd88e9a095c820f29f037909363e0b18a6cdfb7b6c204945.jpg)

![141f1821848aa156b67c520b04836aeb39bdbfc0074a8b1ee20360112827f74d.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/images/141f1821848aa156b67c520b04836aeb39bdbfc0074a8b1ee20360112827f74d.jpg)

![1ea9759a772c48ba58182dea03d5b8ec781582c7491d5fbb86994894700aa3a3.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/images/1ea9759a772c48ba58182dea03d5b8ec781582c7491d5fbb86994894700aa3a3.jpg)

![20bbffe06539ff0aa29b1cfad905f866f79644766a9f9287c014f4ad1dd9d835.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/images/20bbffe06539ff0aa29b1cfad905f866f79644766a9f9287c014f4ad1dd9d835.jpg)

![321e32cb4966eadff51572bc22128bae39a43b3b593f27d7f5082e59c265e378.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/images/321e32cb4966eadff51572bc22128bae39a43b3b593f27d7f5082e59c265e378.jpg)

![3d02ce14ea077b9c81d16dae3c5d551455a70d13f8ac5c014b0e930c19e1bafa.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/images/3d02ce14ea077b9c81d16dae3c5d551455a70d13f8ac5c014b0e930c19e1bafa.jpg)

![4930aefc1105e2e0b44ae53efd2afc84afde962153c1d19c8e9abc2a6ada79ea.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/images/4930aefc1105e2e0b44ae53efd2afc84afde962153c1d19c8e9abc2a6ada79ea.jpg)

![4da1d587c6ffa33d01a4826bd5150bc8e13cbd72fb5b00f4e589b3edd5dc7e98.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/images/4da1d587c6ffa33d01a4826bd5150bc8e13cbd72fb5b00f4e589b3edd5dc7e98.jpg)

![54fc9b76af681c772ac06f7c8203aff21a4705874fd0af5af3c08c8fafdde69f.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/images/54fc9b76af681c772ac06f7c8203aff21a4705874fd0af5af3c08c8fafdde69f.jpg)

![715de39b651a7fb2aecb8171a54b61875d5b384ed821269555d1407eef57ef2f.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/images/715de39b651a7fb2aecb8171a54b61875d5b384ed821269555d1407eef57ef2f.jpg)

![7fc7f1ecbdf8bfdf46ba5dfa0a6049caac1cb9eaa98a97aee15c0b786ef03382.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/images/7fc7f1ecbdf8bfdf46ba5dfa0a6049caac1cb9eaa98a97aee15c0b786ef03382.jpg)

![c1abb4e54fd21061917995a72ebe85036099080b4206fecf276205bb65dc42c8.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/images/c1abb4e54fd21061917995a72ebe85036099080b4206fecf276205bb65dc42c8.jpg)

![ef2dd5dc0f804c4e453da8d7084950e6008d2745e23f4f43cd8504ab94ed8300.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/images/ef2dd5dc0f804c4e453da8d7084950e6008d2745e23f4f43cd8504ab94ed8300.jpg)

![f3900e3b3eb9237a02ed119a4522d01ee1757761fa083bd6fe353256f33450da.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/images/f3900e3b3eb9237a02ed119a4522d01ee1757761fa083bd6fe353256f33450da.jpg)

### Tables

![008046d6f46dbd615b33ae079765953ae05033dbddab86905a33ccbfac6d8365.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/tables/008046d6f46dbd615b33ae079765953ae05033dbddab86905a33ccbfac6d8365.jpg)

![1fd42ecdb9ac5e6c92516375d73baf0f3534508eb1d50868fac8b7de00d8ab93.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/tables/1fd42ecdb9ac5e6c92516375d73baf0f3534508eb1d50868fac8b7de00d8ab93.jpg)

![5ce42b17c0b4269bdeacea3dc73049b34768cb7e1b75044db7f8b35dcf67d147.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/tables/5ce42b17c0b4269bdeacea3dc73049b34768cb7e1b75044db7f8b35dcf67d147.jpg)

![7c815bd16fe5df6d4b56585ee20f021fca302a0ca9f2bb18756b36287ef40d8d.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/tables/7c815bd16fe5df6d4b56585ee20f021fca302a0ca9f2bb18756b36287ef40d8d.jpg)

![a67160b2291f379805e651f110c54e6b242f22f9f05250fa1e6babee33169b20.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/tables/a67160b2291f379805e651f110c54e6b242f22f9f05250fa1e6babee33169b20.jpg)

![bb4808a591b03c544ed87f07592b36af7ede2b75ce48992a6d4f471e42a7198a.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/tables/bb4808a591b03c544ed87f07592b36af7ede2b75ce48992a6d4f471e42a7198a.jpg)

![caf4d136898a75786c8b57f8d5290982ff93ae294b0a1937090366613286f716.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/tables/caf4d136898a75786c8b57f8d5290982ff93ae294b0a1937090366613286f716.jpg)

![cb048196893a412c8e5ca7a348f816978d1281f3b6f52923a9d531e093566d27.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/tables/cb048196893a412c8e5ca7a348f816978d1281f3b6f52923a9d531e093566d27.jpg)

![d1ff459f063b890a88380bcfae74c0e5ec5fac3e1eb05101ce3c274501020780.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/tables/d1ff459f063b890a88380bcfae74c0e5ec5fac3e1eb05101ce3c274501020780.jpg)

![dec4fa1832fb956624d8012c4a9926ddb7b749a8a4960a24b86d97d86b5e0ca0.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/tables/dec4fa1832fb956624d8012c4a9926ddb7b749a8a4960a24b86d97d86b5e0ca0.jpg)

![e9f00d4de4ba11716a7d61d38b1c912d19fffd7192ea906feaf36441441c401d.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/tables/e9f00d4de4ba11716a7d61d38b1c912d19fffd7192ea906feaf36441441c401d.jpg)

![eab61128d212b18976ec0b9025afdf4c2c9eb1ecc28a3cea8c0edba3e25b0f2e.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/tables/eab61128d212b18976ec0b9025afdf4c2c9eb1ecc28a3cea8c0edba3e25b0f2e.jpg)

![f3e4c0142e46640717ba101f69b29e4aa29fa29ccbdc66e0b72d3e4ae67a99a2.jpg](../iclr_results/1653_What%20is%20Wrong%20with%20Perplexity%20for%20Long-context%20Language%20Modeling_/tables/f3e4c0142e46640717ba101f69b29e4aa29fa29ccbdc66e0b72d3e4ae67a99a2.jpg)

## Recovery of Causal Graph Involving Latent Variables via Homologous Surrogates


### Images

![0a08239042d1bdab7d0b592bb742a3d18a7738cacbc8249501bf08808216e53f.jpg](../iclr_results/1654_Recovery%20of%20Causal%20Graph%20Involving%20Latent%20Variables%20via%20Homologous%20Surrogates/images/0a08239042d1bdab7d0b592bb742a3d18a7738cacbc8249501bf08808216e53f.jpg)

![0de5d0d1b77ef2d660d0ab3ee33d451e4c123a8d6266714f5c19cd967ac39fdc.jpg](../iclr_results/1654_Recovery%20of%20Causal%20Graph%20Involving%20Latent%20Variables%20via%20Homologous%20Surrogates/images/0de5d0d1b77ef2d660d0ab3ee33d451e4c123a8d6266714f5c19cd967ac39fdc.jpg)

![36583ab7e54b64c20452f448144805de268936a878d9c9f34c88a81f45443ff9.jpg](../iclr_results/1654_Recovery%20of%20Causal%20Graph%20Involving%20Latent%20Variables%20via%20Homologous%20Surrogates/images/36583ab7e54b64c20452f448144805de268936a878d9c9f34c88a81f45443ff9.jpg)

![44111c3a7f3dd13b7f29bc9eaeb0a08ef042431732314ce8be1076a61949523c.jpg](../iclr_results/1654_Recovery%20of%20Causal%20Graph%20Involving%20Latent%20Variables%20via%20Homologous%20Surrogates/images/44111c3a7f3dd13b7f29bc9eaeb0a08ef042431732314ce8be1076a61949523c.jpg)

![525a8c1e3589619cc6a0af7ca208fa68a610399db3f0ce0f78b6735417da8a82.jpg](../iclr_results/1654_Recovery%20of%20Causal%20Graph%20Involving%20Latent%20Variables%20via%20Homologous%20Surrogates/images/525a8c1e3589619cc6a0af7ca208fa68a610399db3f0ce0f78b6735417da8a82.jpg)

![587941646dba6c1cacddbe862e5f1fc8d2d7146dea9cafd6cde5d6149ad43490.jpg](../iclr_results/1654_Recovery%20of%20Causal%20Graph%20Involving%20Latent%20Variables%20via%20Homologous%20Surrogates/images/587941646dba6c1cacddbe862e5f1fc8d2d7146dea9cafd6cde5d6149ad43490.jpg)

![94ce0f939f116e75c77b5cdbd1cfc6508eef493aea44ec234207a59010c4fb36.jpg](../iclr_results/1654_Recovery%20of%20Causal%20Graph%20Involving%20Latent%20Variables%20via%20Homologous%20Surrogates/images/94ce0f939f116e75c77b5cdbd1cfc6508eef493aea44ec234207a59010c4fb36.jpg)

![d5d64f37596f1a736ea3b2a8183604219429aae33cd125a895984ae2dbae6a93.jpg](../iclr_results/1654_Recovery%20of%20Causal%20Graph%20Involving%20Latent%20Variables%20via%20Homologous%20Surrogates/images/d5d64f37596f1a736ea3b2a8183604219429aae33cd125a895984ae2dbae6a93.jpg)

### Tables

![02d902f823e7c3e09dbfdcadd8ece8111bfb49c6b72d1578ba5e961318f0ae36.jpg](../iclr_results/1654_Recovery%20of%20Causal%20Graph%20Involving%20Latent%20Variables%20via%20Homologous%20Surrogates/tables/02d902f823e7c3e09dbfdcadd8ece8111bfb49c6b72d1578ba5e961318f0ae36.jpg)

![ac7c7e103e3dbd11ffda0dc47066c2496a37372ef98ae29b3496827f0e59fdbb.jpg](../iclr_results/1654_Recovery%20of%20Causal%20Graph%20Involving%20Latent%20Variables%20via%20Homologous%20Surrogates/tables/ac7c7e103e3dbd11ffda0dc47066c2496a37372ef98ae29b3496827f0e59fdbb.jpg)

## OpenPRM: Building Open-domain Process-based Reward Models with Preference Trees


### Images

![3b41e83f9f1c531e4b8edc259057353b375717ec080de63e7828cd59470e16c5.jpg](../iclr_results/1655_OpenPRM_%20Building%20Open-domain%20Process-based%20Reward%20Models%20with%20Preference%20Trees/images/3b41e83f9f1c531e4b8edc259057353b375717ec080de63e7828cd59470e16c5.jpg)

![5ab642eeebae1d09492d4a7a936899a9164c8470df614cd6cd0e67cb67b70053.jpg](../iclr_results/1655_OpenPRM_%20Building%20Open-domain%20Process-based%20Reward%20Models%20with%20Preference%20Trees/images/5ab642eeebae1d09492d4a7a936899a9164c8470df614cd6cd0e67cb67b70053.jpg)

![60a33f8d6234e32d442aeb978fcf46c4d535c015e6bb375abe0fb19d706df546.jpg](../iclr_results/1655_OpenPRM_%20Building%20Open-domain%20Process-based%20Reward%20Models%20with%20Preference%20Trees/images/60a33f8d6234e32d442aeb978fcf46c4d535c015e6bb375abe0fb19d706df546.jpg)

![6bce25a3cf372c8cccc02338d6d9a38110f23d8d6734a30c54669267b989ddf6.jpg](../iclr_results/1655_OpenPRM_%20Building%20Open-domain%20Process-based%20Reward%20Models%20with%20Preference%20Trees/images/6bce25a3cf372c8cccc02338d6d9a38110f23d8d6734a30c54669267b989ddf6.jpg)

![9544fd72fe916294616a878c5309c2e3c4d6d6c1a57e983ba40b2b9991c05c80.jpg](../iclr_results/1655_OpenPRM_%20Building%20Open-domain%20Process-based%20Reward%20Models%20with%20Preference%20Trees/images/9544fd72fe916294616a878c5309c2e3c4d6d6c1a57e983ba40b2b9991c05c80.jpg)

![97577f630f7834c61f8ac1468d887aa4f12b9c641bd019b807f44bc2184bc1b3.jpg](../iclr_results/1655_OpenPRM_%20Building%20Open-domain%20Process-based%20Reward%20Models%20with%20Preference%20Trees/images/97577f630f7834c61f8ac1468d887aa4f12b9c641bd019b807f44bc2184bc1b3.jpg)

![b5852db1e4ed9fd7eb885696c521c4f0adddff33069e45a3e52233a9fdce96da.jpg](../iclr_results/1655_OpenPRM_%20Building%20Open-domain%20Process-based%20Reward%20Models%20with%20Preference%20Trees/images/b5852db1e4ed9fd7eb885696c521c4f0adddff33069e45a3e52233a9fdce96da.jpg)

![b965c2f37375b956f3cdfba0803c2ca596ca176d87193c151474db559cacb70d.jpg](../iclr_results/1655_OpenPRM_%20Building%20Open-domain%20Process-based%20Reward%20Models%20with%20Preference%20Trees/images/b965c2f37375b956f3cdfba0803c2ca596ca176d87193c151474db559cacb70d.jpg)

![c562fba4ecda22444fd71f86032100221b977acd9c2d4ccca1a3be076f13fdd2.jpg](../iclr_results/1655_OpenPRM_%20Building%20Open-domain%20Process-based%20Reward%20Models%20with%20Preference%20Trees/images/c562fba4ecda22444fd71f86032100221b977acd9c2d4ccca1a3be076f13fdd2.jpg)

![d2751b614a42d693a2d6e0a68a0fcf3c70b5de555e3d71ce8e51f4de9ff0bb6a.jpg](../iclr_results/1655_OpenPRM_%20Building%20Open-domain%20Process-based%20Reward%20Models%20with%20Preference%20Trees/images/d2751b614a42d693a2d6e0a68a0fcf3c70b5de555e3d71ce8e51f4de9ff0bb6a.jpg)

![e1e1eb2eca0eb97759dc8a762b7b963e7de32199c675642bd34d8c3620b6dd17.jpg](../iclr_results/1655_OpenPRM_%20Building%20Open-domain%20Process-based%20Reward%20Models%20with%20Preference%20Trees/images/e1e1eb2eca0eb97759dc8a762b7b963e7de32199c675642bd34d8c3620b6dd17.jpg)

![ec75a4913aec3fbd0a84b87ab5a14ebcd3ec24a7de5f6cc6e73b4bfb7e1c07bb.jpg](../iclr_results/1655_OpenPRM_%20Building%20Open-domain%20Process-based%20Reward%20Models%20with%20Preference%20Trees/images/ec75a4913aec3fbd0a84b87ab5a14ebcd3ec24a7de5f6cc6e73b4bfb7e1c07bb.jpg)

### Tables

![1451f3bac0e79ecd8c96b057f297e989eba5e339f06f023257ed37d3f5fe7140.jpg](../iclr_results/1655_OpenPRM_%20Building%20Open-domain%20Process-based%20Reward%20Models%20with%20Preference%20Trees/tables/1451f3bac0e79ecd8c96b057f297e989eba5e339f06f023257ed37d3f5fe7140.jpg)

![25a611d9ab1af5c241a246f5924786a4afb6ed9a8c03535b870db7f273ecccf9.jpg](../iclr_results/1655_OpenPRM_%20Building%20Open-domain%20Process-based%20Reward%20Models%20with%20Preference%20Trees/tables/25a611d9ab1af5c241a246f5924786a4afb6ed9a8c03535b870db7f273ecccf9.jpg)

![69b938c7a4d60770a4d374893517b664cb75e114e7b17395380e182a72b28a77.jpg](../iclr_results/1655_OpenPRM_%20Building%20Open-domain%20Process-based%20Reward%20Models%20with%20Preference%20Trees/tables/69b938c7a4d60770a4d374893517b664cb75e114e7b17395380e182a72b28a77.jpg)

![83041f5bb70d37e0dc17a900586a6f4f59d3a61077e1557cc05c524a188621cb.jpg](../iclr_results/1655_OpenPRM_%20Building%20Open-domain%20Process-based%20Reward%20Models%20with%20Preference%20Trees/tables/83041f5bb70d37e0dc17a900586a6f4f59d3a61077e1557cc05c524a188621cb.jpg)

![def9d41e473b9d776dbcd132761d4e6dc0641c2134a0a6f531e04c98c233fd2a.jpg](../iclr_results/1655_OpenPRM_%20Building%20Open-domain%20Process-based%20Reward%20Models%20with%20Preference%20Trees/tables/def9d41e473b9d776dbcd132761d4e6dc0641c2134a0a6f531e04c98c233fd2a.jpg)

![e412176cca3b5e16b248e9b2486e7a67b3c9477271682e263b665d1d51c4e5f4.jpg](../iclr_results/1655_OpenPRM_%20Building%20Open-domain%20Process-based%20Reward%20Models%20with%20Preference%20Trees/tables/e412176cca3b5e16b248e9b2486e7a67b3c9477271682e263b665d1d51c4e5f4.jpg)

![fedc783817179d02a59138750e2d468cdffdc3f6a61df218c871490748d0693a.jpg](../iclr_results/1655_OpenPRM_%20Building%20Open-domain%20Process-based%20Reward%20Models%20with%20Preference%20Trees/tables/fedc783817179d02a59138750e2d468cdffdc3f6a61df218c871490748d0693a.jpg)

## Efficient Discovery of Pareto Front for Multi-Objective Reinforcement Learning


### Images

![053802f6870c5b884150a08187eb842d41dd8645f23a396acef6482f1d61eeb3.jpg](../iclr_results/1656_Efficient%20Discovery%20of%20Pareto%20Front%20for%20Multi-Objective%20Reinforcement%20Learning/images/053802f6870c5b884150a08187eb842d41dd8645f23a396acef6482f1d61eeb3.jpg)

![08241c694eeab0fa14466f89c842c3b809263e59899d501455b3397bf5f3a80e.jpg](../iclr_results/1656_Efficient%20Discovery%20of%20Pareto%20Front%20for%20Multi-Objective%20Reinforcement%20Learning/images/08241c694eeab0fa14466f89c842c3b809263e59899d501455b3397bf5f3a80e.jpg)

![3249df7ce492e8ade2eab44066207271963c1f7966d23f3e820307dfae134ac3.jpg](../iclr_results/1656_Efficient%20Discovery%20of%20Pareto%20Front%20for%20Multi-Objective%20Reinforcement%20Learning/images/3249df7ce492e8ade2eab44066207271963c1f7966d23f3e820307dfae134ac3.jpg)

![343879ed5f8f9bbcbc1d125d2678edd27d67944e610e9dd7fc33decb33ac7031.jpg](../iclr_results/1656_Efficient%20Discovery%20of%20Pareto%20Front%20for%20Multi-Objective%20Reinforcement%20Learning/images/343879ed5f8f9bbcbc1d125d2678edd27d67944e610e9dd7fc33decb33ac7031.jpg)

![3a526bb65835aacbde6f99a74c908ab06f185eb566922d5f9df627f8298a0cd0.jpg](../iclr_results/1656_Efficient%20Discovery%20of%20Pareto%20Front%20for%20Multi-Objective%20Reinforcement%20Learning/images/3a526bb65835aacbde6f99a74c908ab06f185eb566922d5f9df627f8298a0cd0.jpg)

![44d6d9a4a6199d87a7ab96eb166e6630131dbcb896f3ba6b88e4a22945b0dee3.jpg](../iclr_results/1656_Efficient%20Discovery%20of%20Pareto%20Front%20for%20Multi-Objective%20Reinforcement%20Learning/images/44d6d9a4a6199d87a7ab96eb166e6630131dbcb896f3ba6b88e4a22945b0dee3.jpg)

![ad3abaeebb87f29b7b974e29ac061511612de38b5e14697ede8cf0511eab8ac9.jpg](../iclr_results/1656_Efficient%20Discovery%20of%20Pareto%20Front%20for%20Multi-Objective%20Reinforcement%20Learning/images/ad3abaeebb87f29b7b974e29ac061511612de38b5e14697ede8cf0511eab8ac9.jpg)

![f0181286a8b60e80dca251a0a79007275fbf199e37e7e4218ab0de3ebf5d3ade.jpg](../iclr_results/1656_Efficient%20Discovery%20of%20Pareto%20Front%20for%20Multi-Objective%20Reinforcement%20Learning/images/f0181286a8b60e80dca251a0a79007275fbf199e37e7e4218ab0de3ebf5d3ade.jpg)

### Tables

![03b75d89a9cd4b5784782c0259d5bc07af342f2601eeb749952a624327aaaebc.jpg](../iclr_results/1656_Efficient%20Discovery%20of%20Pareto%20Front%20for%20Multi-Objective%20Reinforcement%20Learning/tables/03b75d89a9cd4b5784782c0259d5bc07af342f2601eeb749952a624327aaaebc.jpg)

![04def2a8369660994163ac0772d8e1d9f772ca66fc42a222477e4ff7ebec91d5.jpg](../iclr_results/1656_Efficient%20Discovery%20of%20Pareto%20Front%20for%20Multi-Objective%20Reinforcement%20Learning/tables/04def2a8369660994163ac0772d8e1d9f772ca66fc42a222477e4ff7ebec91d5.jpg)

![249498323eed1340dbe729035f3a72ed47e7d571452f137fb8cfb58a96adc79f.jpg](../iclr_results/1656_Efficient%20Discovery%20of%20Pareto%20Front%20for%20Multi-Objective%20Reinforcement%20Learning/tables/249498323eed1340dbe729035f3a72ed47e7d571452f137fb8cfb58a96adc79f.jpg)

![26d93e07ac667b52349f88964782816a2228bb66ce315207b9d04e893180b9e9.jpg](../iclr_results/1656_Efficient%20Discovery%20of%20Pareto%20Front%20for%20Multi-Objective%20Reinforcement%20Learning/tables/26d93e07ac667b52349f88964782816a2228bb66ce315207b9d04e893180b9e9.jpg)

![37b5ef0f844aae73ae24a7e727b76ebbc2b182fd489362beb4d47f15b708f57e.jpg](../iclr_results/1656_Efficient%20Discovery%20of%20Pareto%20Front%20for%20Multi-Objective%20Reinforcement%20Learning/tables/37b5ef0f844aae73ae24a7e727b76ebbc2b182fd489362beb4d47f15b708f57e.jpg)

![6fbbc10eec25880bb9082146403dfeed5b3c69cef9afb87cfdddf6b452688fb1.jpg](../iclr_results/1656_Efficient%20Discovery%20of%20Pareto%20Front%20for%20Multi-Objective%20Reinforcement%20Learning/tables/6fbbc10eec25880bb9082146403dfeed5b3c69cef9afb87cfdddf6b452688fb1.jpg)

![7629bc0a1224bf34594050c009147244644b81b590185b1ebe7e7d9b3830f24e.jpg](../iclr_results/1656_Efficient%20Discovery%20of%20Pareto%20Front%20for%20Multi-Objective%20Reinforcement%20Learning/tables/7629bc0a1224bf34594050c009147244644b81b590185b1ebe7e7d9b3830f24e.jpg)

![97d4ce0c4b0b5c6c13eadbf6c894f477990fe66f1d2244a622f1d90c9f9185e7.jpg](../iclr_results/1656_Efficient%20Discovery%20of%20Pareto%20Front%20for%20Multi-Objective%20Reinforcement%20Learning/tables/97d4ce0c4b0b5c6c13eadbf6c894f477990fe66f1d2244a622f1d90c9f9185e7.jpg)

![b30cc545a0b09cc89b429247e3bf01eae52f3f103278e444d538b4ab8d2fe94c.jpg](../iclr_results/1656_Efficient%20Discovery%20of%20Pareto%20Front%20for%20Multi-Objective%20Reinforcement%20Learning/tables/b30cc545a0b09cc89b429247e3bf01eae52f3f103278e444d538b4ab8d2fe94c.jpg)

![c4fbf208c28c7f7ad11e64647938b6f46a79ef19f6a872114f813ca1ad0a39c6.jpg](../iclr_results/1656_Efficient%20Discovery%20of%20Pareto%20Front%20for%20Multi-Objective%20Reinforcement%20Learning/tables/c4fbf208c28c7f7ad11e64647938b6f46a79ef19f6a872114f813ca1ad0a39c6.jpg)

![cc8304682a2b7bd5752873ff34c516533cc4e7ea1cc56516ffdc6e4711c99e69.jpg](../iclr_results/1656_Efficient%20Discovery%20of%20Pareto%20Front%20for%20Multi-Objective%20Reinforcement%20Learning/tables/cc8304682a2b7bd5752873ff34c516533cc4e7ea1cc56516ffdc6e4711c99e69.jpg)

![f0e88facc8bbeaa8fdd20e7d89f649fc80df9334ce75acf39f41729102377f81.jpg](../iclr_results/1656_Efficient%20Discovery%20of%20Pareto%20Front%20for%20Multi-Objective%20Reinforcement%20Learning/tables/f0e88facc8bbeaa8fdd20e7d89f649fc80df9334ce75acf39f41729102377f81.jpg)

![ffc7309374ede2080175f1df7ad51c6aa4c2f4a07536ebe3843f5c9413930d7c.jpg](../iclr_results/1656_Efficient%20Discovery%20of%20Pareto%20Front%20for%20Multi-Objective%20Reinforcement%20Learning/tables/ffc7309374ede2080175f1df7ad51c6aa4c2f4a07536ebe3843f5c9413930d7c.jpg)

## TOMATO: Assessing Visual Temporal Reasoning Capabilities in Multimodal Foundation Models


### Images

![00723a11e09660c2f79482b9e51234e7111f5c45562625b0766dbea966c2ec8e.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/00723a11e09660c2f79482b9e51234e7111f5c45562625b0766dbea966c2ec8e.jpg)

![04c6e3640f5926fef6d054e97a8941a4de43a7884a4e037f6d622a1ec072b25e.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/04c6e3640f5926fef6d054e97a8941a4de43a7884a4e037f6d622a1ec072b25e.jpg)

![058b3b1eebcf1603bd25e40377f490c1dd539e9d1e78cc4f774aad231bc1b04d.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/058b3b1eebcf1603bd25e40377f490c1dd539e9d1e78cc4f774aad231bc1b04d.jpg)

![111d39a1070552265e505ddefc905f759c02b2f3bd52c5f02e4e5fafc4d89f2d.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/111d39a1070552265e505ddefc905f759c02b2f3bd52c5f02e4e5fafc4d89f2d.jpg)

![1367aef6f7b225e73942a53bf1d9109e481a96f45a4d74035463a5342c1655a4.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/1367aef6f7b225e73942a53bf1d9109e481a96f45a4d74035463a5342c1655a4.jpg)

![1482da84ea18892d1838db7491ff5109c738be1749d873cbe3ee212b3b9aa359.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/1482da84ea18892d1838db7491ff5109c738be1749d873cbe3ee212b3b9aa359.jpg)

![15283d3d7c1630b3df80bd97724e864681dc72f75478ecb25631dba9951eeaab.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/15283d3d7c1630b3df80bd97724e864681dc72f75478ecb25631dba9951eeaab.jpg)

![15896261f544b8be92289e842bf94c5ca857b205872c6b639e7192d3aa2d480c.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/15896261f544b8be92289e842bf94c5ca857b205872c6b639e7192d3aa2d480c.jpg)

![1c5b405dff77efc12244aeaf7535494208137344aba2c4f84ed16ca1a308b56a.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/1c5b405dff77efc12244aeaf7535494208137344aba2c4f84ed16ca1a308b56a.jpg)

![1fff61db63c551d59191ba916a1989e5761c3f2717c0bee6fb7e1bfb3862b72e.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/1fff61db63c551d59191ba916a1989e5761c3f2717c0bee6fb7e1bfb3862b72e.jpg)

![2206754fe1f187f58cab457df6693a31ee6f6beeec928b8552667b134cd7f4b7.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/2206754fe1f187f58cab457df6693a31ee6f6beeec928b8552667b134cd7f4b7.jpg)

![2395b2286e8a65f50aacd9692a52230809022c52902c4a70296a80c1315fd67c.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/2395b2286e8a65f50aacd9692a52230809022c52902c4a70296a80c1315fd67c.jpg)

![2444d018f546bbe8d2831b6733f4ba0de062c51dafe8ff0844ce63e04b83cc38.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/2444d018f546bbe8d2831b6733f4ba0de062c51dafe8ff0844ce63e04b83cc38.jpg)

![26646fb0eb32cd6c561e414a14674e68956c5932a7c74bab2ed775b7e28ecd03.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/26646fb0eb32cd6c561e414a14674e68956c5932a7c74bab2ed775b7e28ecd03.jpg)

![2a9c8a13e9b75fa3675f43452443ce41873916688adc9cda40552237d3b47c5b.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/2a9c8a13e9b75fa3675f43452443ce41873916688adc9cda40552237d3b47c5b.jpg)

![2c45b5716b97a48f4dbd6c07356fc2bcbd16d1309abb2b3c9805469fdb81069f.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/2c45b5716b97a48f4dbd6c07356fc2bcbd16d1309abb2b3c9805469fdb81069f.jpg)

![2f8cba4adab13d98bf520bc4bfe7351f3a847bb366c5be83d874643a2f4dcd94.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/2f8cba4adab13d98bf520bc4bfe7351f3a847bb366c5be83d874643a2f4dcd94.jpg)

![308450d902a65bdc15d208e7aded7c18111b55c4b477fcb387cbd05d691e83c0.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/308450d902a65bdc15d208e7aded7c18111b55c4b477fcb387cbd05d691e83c0.jpg)

![317407fec925628b954859705b3a218bcff16ef5aafbef656d09637120d663fb.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/317407fec925628b954859705b3a218bcff16ef5aafbef656d09637120d663fb.jpg)

![33324a6bd57ba0b4846769921f7fe257c60bb2cd5d5290e40e7100b942e43343.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/33324a6bd57ba0b4846769921f7fe257c60bb2cd5d5290e40e7100b942e43343.jpg)

![37621cc074876d089bd41a7dbe50f987cf4a33e2b277f7911c50251dee9a891a.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/37621cc074876d089bd41a7dbe50f987cf4a33e2b277f7911c50251dee9a891a.jpg)

![3bd72ec656e012d0a1137313899bc917f8ee45c8204dfe17e762d168fccd6fc3.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/3bd72ec656e012d0a1137313899bc917f8ee45c8204dfe17e762d168fccd6fc3.jpg)

![3e1101f80fac5f924c347a65e230972194906872b942060fd684523f3af83e3f.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/3e1101f80fac5f924c347a65e230972194906872b942060fd684523f3af83e3f.jpg)

![443c6e876290361631151c7c81e90ea2ac20bfe382137ba74aacca0bce2b0ff5.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/443c6e876290361631151c7c81e90ea2ac20bfe382137ba74aacca0bce2b0ff5.jpg)

![44b5f2903baa2137b395aec51fe95862fa79feae842dd785c08876e3db4fecea.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/44b5f2903baa2137b395aec51fe95862fa79feae842dd785c08876e3db4fecea.jpg)

![4d8a0f627b8ae29d6302fb4560c1add3dd4812fb7107138dde5d0378e98b5fa8.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/4d8a0f627b8ae29d6302fb4560c1add3dd4812fb7107138dde5d0378e98b5fa8.jpg)

![4f17431fc22bf188d962275d1ed73f81e20337ec64bc31a6ea7a249749bff98f.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/4f17431fc22bf188d962275d1ed73f81e20337ec64bc31a6ea7a249749bff98f.jpg)

![521ad4c52aff67c06ace428db632050dd4cbb06df0699c804da4d40b18e06314.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/521ad4c52aff67c06ace428db632050dd4cbb06df0699c804da4d40b18e06314.jpg)

![568acd89e7ca5d0f149c3bb1abb99b93423ddbd995c942f1e1f2c755fd14c566.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/568acd89e7ca5d0f149c3bb1abb99b93423ddbd995c942f1e1f2c755fd14c566.jpg)

![5bba5b031a9a590dfca51106e8dd1a9d50a065677019f51b774fc09903bd6c20.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/5bba5b031a9a590dfca51106e8dd1a9d50a065677019f51b774fc09903bd6c20.jpg)

![5d9dc404ea840dcf0783da8105b9412196e024524ad6889d976b85cd0460b007.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/5d9dc404ea840dcf0783da8105b9412196e024524ad6889d976b85cd0460b007.jpg)

![5e02f189dd755e40d3a9e5abf584c462cc4f0ea5869ad55d1667218fad16a31e.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/5e02f189dd755e40d3a9e5abf584c462cc4f0ea5869ad55d1667218fad16a31e.jpg)

![61e69c5dad4bcb4df9594bc0ed3867e5bc47146421ea240f119cd12092db328d.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/61e69c5dad4bcb4df9594bc0ed3867e5bc47146421ea240f119cd12092db328d.jpg)

![67a75a9810577114bbab1293dfcdca8856a40ec034235f2ff6213ac0e8358a49.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/67a75a9810577114bbab1293dfcdca8856a40ec034235f2ff6213ac0e8358a49.jpg)

![689a7c4f314be6a43272fb1e0a1f4c9b7949742894c725d297cd1830459a4e6b.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/689a7c4f314be6a43272fb1e0a1f4c9b7949742894c725d297cd1830459a4e6b.jpg)

![6988a4ae03708d1bf423b13a28dbacec1fb53caa3ac00df980fd549b6708d4fc.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/6988a4ae03708d1bf423b13a28dbacec1fb53caa3ac00df980fd549b6708d4fc.jpg)

![7868b0239adde0208a6e30b1674b8da290ccf443df36fe6d60b6a09d46df43eb.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/7868b0239adde0208a6e30b1674b8da290ccf443df36fe6d60b6a09d46df43eb.jpg)

![796d0f8693fcf1e136f07f543d8d91a1725d3554d1647e9bb82bb27b9d8294cf.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/796d0f8693fcf1e136f07f543d8d91a1725d3554d1647e9bb82bb27b9d8294cf.jpg)

![82d3f547b2aaf2794d18e71519968d8361c2cc6820de2b02eccc357878e8d1cd.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/82d3f547b2aaf2794d18e71519968d8361c2cc6820de2b02eccc357878e8d1cd.jpg)

![82f4f1904f326450a5e263adadcf5a188c8e45f4b05a25b25c3129dbfc22981b.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/82f4f1904f326450a5e263adadcf5a188c8e45f4b05a25b25c3129dbfc22981b.jpg)

![83cb6713c8c3560fa83a20c7c92a4663e25126c0320296addb5e6ac6e252164c.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/83cb6713c8c3560fa83a20c7c92a4663e25126c0320296addb5e6ac6e252164c.jpg)

![88de1092e271064cdaea87e219106b820aaca7639f9d35230f2d4b7197385e1f.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/88de1092e271064cdaea87e219106b820aaca7639f9d35230f2d4b7197385e1f.jpg)

![8a8a74365d65294e207e1aac8ea0d895447179ca19e4a540b76f1ba290895777.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/8a8a74365d65294e207e1aac8ea0d895447179ca19e4a540b76f1ba290895777.jpg)

![8b96de426f832cd581c777b594a15a52fe1b0270a9f5c623c177fab5c3590060.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/8b96de426f832cd581c777b594a15a52fe1b0270a9f5c623c177fab5c3590060.jpg)

![8f65f7c9aa058bf4e80fe950fd087d65cfc0b3a591266b7d2a8a6154458854a2.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/8f65f7c9aa058bf4e80fe950fd087d65cfc0b3a591266b7d2a8a6154458854a2.jpg)

![911ad8dcf6dafbbd906ebff6619a5839a13a415fd74c876504cfc97278ea509d.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/911ad8dcf6dafbbd906ebff6619a5839a13a415fd74c876504cfc97278ea509d.jpg)

![916fb395d2324fbf652d08925f2faf46589670b3c494644f04f9f7f122e5509c.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/916fb395d2324fbf652d08925f2faf46589670b3c494644f04f9f7f122e5509c.jpg)

![91852dd54d4aa1ee84fa6b44c96d2bcad2078f16d73ed731afc80628e496a9ca.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/91852dd54d4aa1ee84fa6b44c96d2bcad2078f16d73ed731afc80628e496a9ca.jpg)

![91fa9471af2c5c138b6f794194afbe87c6cdb10e6b8801cf95e026f7ecef9005.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/91fa9471af2c5c138b6f794194afbe87c6cdb10e6b8801cf95e026f7ecef9005.jpg)

![96508abdff8022f38121e83eb9e341dfcc9806b1f6057326e2cee4a4e5333d93.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/96508abdff8022f38121e83eb9e341dfcc9806b1f6057326e2cee4a4e5333d93.jpg)

![981302d7f824155e8661148abb6ba4f5f2c0e357c2937e07a900592d187552a2.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/981302d7f824155e8661148abb6ba4f5f2c0e357c2937e07a900592d187552a2.jpg)

![9b9c6bea03edc9100c555494e6cffb40cfe26e321e8dd5919a9efd03178bbb80.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/9b9c6bea03edc9100c555494e6cffb40cfe26e321e8dd5919a9efd03178bbb80.jpg)

![9febe0e63af790402df3f1fe716b5a832301ceaccf2dd1c6837ba10426624242.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/9febe0e63af790402df3f1fe716b5a832301ceaccf2dd1c6837ba10426624242.jpg)

![a086c3ef572f89653948bd9bb5d751186c2cc7994674128d9f348f9f1d618b28.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/a086c3ef572f89653948bd9bb5d751186c2cc7994674128d9f348f9f1d618b28.jpg)

![aa6b55ce268e107e5fed5144dc47545a073d86afaa33003b614b87bcfc14c533.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/aa6b55ce268e107e5fed5144dc47545a073d86afaa33003b614b87bcfc14c533.jpg)

![ab4f0b28b247fdedab1e09455db6267119e47b46652348b1458c77a2b49cf97f.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/ab4f0b28b247fdedab1e09455db6267119e47b46652348b1458c77a2b49cf97f.jpg)

![aeec32ca263ebb51eda7d7867675e698cfb534b673e3ffcbd300d1d8d00297b4.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/aeec32ca263ebb51eda7d7867675e698cfb534b673e3ffcbd300d1d8d00297b4.jpg)

![b1f67a452db57b0ae4d6e2743e17ad1df3d95f9dbf6a963d761ac3e26d70b639.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/b1f67a452db57b0ae4d6e2743e17ad1df3d95f9dbf6a963d761ac3e26d70b639.jpg)

![b37f12a20b93d1a96cbbe604398b717f84e85973785539c39195ef0bad322271.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/b37f12a20b93d1a96cbbe604398b717f84e85973785539c39195ef0bad322271.jpg)

![b7c4b01863409d85bd5c2155153934b7e21cc5dbcf3ce71528d7b8f222271284.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/b7c4b01863409d85bd5c2155153934b7e21cc5dbcf3ce71528d7b8f222271284.jpg)

![bc99cb58b72c9234d15af9af719c99b0b7e1bbf3be90a94fc0fa3aa312554a58.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/bc99cb58b72c9234d15af9af719c99b0b7e1bbf3be90a94fc0fa3aa312554a58.jpg)

![bd5dab1333dacbb611a5cc83d64b22ff09a4c885dff3df70f985567626c579da.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/bd5dab1333dacbb611a5cc83d64b22ff09a4c885dff3df70f985567626c579da.jpg)

![c19df231b922a26cafa3464f3a2a90da60c12c9e5ad81964495eac53cdcbaa2e.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/c19df231b922a26cafa3464f3a2a90da60c12c9e5ad81964495eac53cdcbaa2e.jpg)

![c38e16a03a61e751836622611f343093d55661eea56ae9fcc287f25c8db25ff5.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/c38e16a03a61e751836622611f343093d55661eea56ae9fcc287f25c8db25ff5.jpg)

![c4f8e0dda685ec12dcde9091c01fcac0c3ea05180ac563ff8df8393d339aab94.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/c4f8e0dda685ec12dcde9091c01fcac0c3ea05180ac563ff8df8393d339aab94.jpg)

![c6981a4821b5ce960cb8216f951022a367dedf05f92645c3315b1eec58aeef4d.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/c6981a4821b5ce960cb8216f951022a367dedf05f92645c3315b1eec58aeef4d.jpg)

![c94085517bc39f4abe3e9c558b8d24ec643dcf6d7b40517e4c1fcbe28b291348.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/c94085517bc39f4abe3e9c558b8d24ec643dcf6d7b40517e4c1fcbe28b291348.jpg)

![cab896b15f1e7d54b465afe6ee081b7a365fdd09d60bb1f997d74522738fa411.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/cab896b15f1e7d54b465afe6ee081b7a365fdd09d60bb1f997d74522738fa411.jpg)

![cb9fa6819056f097c541102af9fba1dbb4702bf941bbea3fe9d1deb0f67f84d6.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/cb9fa6819056f097c541102af9fba1dbb4702bf941bbea3fe9d1deb0f67f84d6.jpg)

![cd506eabd327c43befcc3af204a1ed82a133709fec9ca3a2e9a94b3f0c3a97d1.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/cd506eabd327c43befcc3af204a1ed82a133709fec9ca3a2e9a94b3f0c3a97d1.jpg)

![d4e33bc298ceaafded2b84355532dd65aa29afbbe96e28ada008b4e04dd537c2.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/d4e33bc298ceaafded2b84355532dd65aa29afbbe96e28ada008b4e04dd537c2.jpg)

![d89063bfe760ecb02987b2502f7f6cd017efa152bd44814788c7091f21c65639.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/d89063bfe760ecb02987b2502f7f6cd017efa152bd44814788c7091f21c65639.jpg)

![da6da10bcd00361d649449e9a856185720cd50b8e4820299e4abff8d0f6f0185.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/da6da10bcd00361d649449e9a856185720cd50b8e4820299e4abff8d0f6f0185.jpg)

![e8e2c34a2eed6327e2248d3e7f3ab293f94ecd3ce86547a5318ec237c59524bf.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/e8e2c34a2eed6327e2248d3e7f3ab293f94ecd3ce86547a5318ec237c59524bf.jpg)

![eb6e41612ddcb7650e87fe013b97992ee6e8e4cbd70335e27453f0884aed684e.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/eb6e41612ddcb7650e87fe013b97992ee6e8e4cbd70335e27453f0884aed684e.jpg)

![f4f0af7b0d67fcdfc0d8f86dfe9b8529f59a70694af14b534840b1ccd56bda0b.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/f4f0af7b0d67fcdfc0d8f86dfe9b8529f59a70694af14b534840b1ccd56bda0b.jpg)

![fab190b57df442fa34e9cdd407332b05890a2b1d4770b2cc9e2e1d90cc36cc11.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/fab190b57df442fa34e9cdd407332b05890a2b1d4770b2cc9e2e1d90cc36cc11.jpg)

![fd1f4beebbc126739a9850eecc8cdac1f5959a816641ce9b089204a4027682dd.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/fd1f4beebbc126739a9850eecc8cdac1f5959a816641ce9b089204a4027682dd.jpg)

![fd38224d5eeca54210d6e8d1d0ba18b27ebabd88ac8c421e5ec02974595f8172.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/fd38224d5eeca54210d6e8d1d0ba18b27ebabd88ac8c421e5ec02974595f8172.jpg)

![fdfde66363c66cd4846abdd214799bb9014bbc62789fba71b29f3ddfe3b0ca1e.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/images/fdfde66363c66cd4846abdd214799bb9014bbc62789fba71b29f3ddfe3b0ca1e.jpg)

### Tables

![1185a3ae2ba06fb4234996f2d5d4b2a8e3e661b832c23dd292fefe83c9a3b0e6.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/tables/1185a3ae2ba06fb4234996f2d5d4b2a8e3e661b832c23dd292fefe83c9a3b0e6.jpg)

![13256551aa070548b37d91cf28ddc57da485c425ee23b22c9843a87ddca11368.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/tables/13256551aa070548b37d91cf28ddc57da485c425ee23b22c9843a87ddca11368.jpg)

![16f40b1980f44d90c527b4e92c755db0042a771675e89b6a711b33e1c69a07aa.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/tables/16f40b1980f44d90c527b4e92c755db0042a771675e89b6a711b33e1c69a07aa.jpg)

![20a8c545d1a2dbe45844e171ef1f0b69bfe36d7264c23af7d3f58e31dbc0e20e.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/tables/20a8c545d1a2dbe45844e171ef1f0b69bfe36d7264c23af7d3f58e31dbc0e20e.jpg)

![3162bebcd43991a8761d41e48905fc5f03ec60980f95fb6c0fa149d1125f5e72.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/tables/3162bebcd43991a8761d41e48905fc5f03ec60980f95fb6c0fa149d1125f5e72.jpg)

![36dbbb4f1501637505c1451a1871913165522d476553a89ea06dbc1275ae9f2f.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/tables/36dbbb4f1501637505c1451a1871913165522d476553a89ea06dbc1275ae9f2f.jpg)

![37a625d3afc0878037c66f68f070eaa458d6cc919b5c79846dd619ed3863a5e4.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/tables/37a625d3afc0878037c66f68f070eaa458d6cc919b5c79846dd619ed3863a5e4.jpg)

![4a67247ef01db02303a054bb7ba31a2272cee3407526f47c46beaf219c0790bd.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/tables/4a67247ef01db02303a054bb7ba31a2272cee3407526f47c46beaf219c0790bd.jpg)

![583131eed27f071473c90e09e9800c43e19cd7aa04227c9000476a1e2aa32e57.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/tables/583131eed27f071473c90e09e9800c43e19cd7aa04227c9000476a1e2aa32e57.jpg)

![66a5f4f0eee4f72930f3db159bde057d5a83c4f408d0854a3ed04ad3d0a018d8.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/tables/66a5f4f0eee4f72930f3db159bde057d5a83c4f408d0854a3ed04ad3d0a018d8.jpg)

![8cc7e75ec9021d7306c40de37887d742f65b29386fa24ddd2552ccc44b8a624f.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/tables/8cc7e75ec9021d7306c40de37887d742f65b29386fa24ddd2552ccc44b8a624f.jpg)

![8d35384e190e413314b6c17b1e9d90bc4e84950f41ac5d14c0256b7d532aea7c.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/tables/8d35384e190e413314b6c17b1e9d90bc4e84950f41ac5d14c0256b7d532aea7c.jpg)

![95ae5429100df1415cc278e1cb74c8902a99a4f342e39ac2e142549f63feda56.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/tables/95ae5429100df1415cc278e1cb74c8902a99a4f342e39ac2e142549f63feda56.jpg)

![b3653ba621415c2cc61ee67c612994ab99a1d095b2d0fca70ad0e23413693155.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/tables/b3653ba621415c2cc61ee67c612994ab99a1d095b2d0fca70ad0e23413693155.jpg)

![bc6197d67874b7778f4fa31e008e65c2dedfae9ba0c8093836f514bb200fd7e7.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/tables/bc6197d67874b7778f4fa31e008e65c2dedfae9ba0c8093836f514bb200fd7e7.jpg)

![c8bf19198cb88f10eaa3f46e04142dc1d8b282d2b75ab352172496e8448b499f.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/tables/c8bf19198cb88f10eaa3f46e04142dc1d8b282d2b75ab352172496e8448b499f.jpg)

![dc7aedf6cd1bd1568200e37bd3c13a6d787df85ebdf2991593175b1a813efe15.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/tables/dc7aedf6cd1bd1568200e37bd3c13a6d787df85ebdf2991593175b1a813efe15.jpg)

![eae3a0e939a5f942f6a5056f3d8aef5b17d56aff987e75be54e27c300a87500a.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/tables/eae3a0e939a5f942f6a5056f3d8aef5b17d56aff987e75be54e27c300a87500a.jpg)

![ef75b12422ca72cdf1a94920a62b1032be0dab3d7246febf4e92a3b322f7dd84.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/tables/ef75b12422ca72cdf1a94920a62b1032be0dab3d7246febf4e92a3b322f7dd84.jpg)

![f2ac7b0322c385b53729b8adf82eb9b34e8772e36cd3b7f9076f59e7207f7184.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/tables/f2ac7b0322c385b53729b8adf82eb9b34e8772e36cd3b7f9076f59e7207f7184.jpg)

![f2b673433dbf87bd2223267f5ad66d17f899e2442f3cad2738f5880ef521d117.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/tables/f2b673433dbf87bd2223267f5ad66d17f899e2442f3cad2738f5880ef521d117.jpg)

![f9ca95ba60d193d3032f51e7f774c18bb423adc0ec0b4c1ba0f4227d37fa3206.jpg](../iclr_results/1657_TOMATO_%20Assessing%20Visual%20Temporal%20Reasoning%20Capabilities%20in%20Multimodal%20Foundation%20Models/tables/f9ca95ba60d193d3032f51e7f774c18bb423adc0ec0b4c1ba0f4227d37fa3206.jpg)

## Amulet: ReAlignment During Test Time for Personalized Preference Adaptation of LLMs


### Images

![0d44fe37e23dadb4c6e7c08c90593fb4883bc328c86748fecc45080f6a2bf30f.jpg](../iclr_results/1658_Amulet_%20ReAlignment%20During%20Test%20Time%20for%20Personalized%20Preference%20Adaptation%20of%20LLMs/images/0d44fe37e23dadb4c6e7c08c90593fb4883bc328c86748fecc45080f6a2bf30f.jpg)

![103a67fed8875474bb20d3b15012dc3a11c38600be461a1d654c16f3c6cdcc41.jpg](../iclr_results/1658_Amulet_%20ReAlignment%20During%20Test%20Time%20for%20Personalized%20Preference%20Adaptation%20of%20LLMs/images/103a67fed8875474bb20d3b15012dc3a11c38600be461a1d654c16f3c6cdcc41.jpg)

![2c242e75ed5eb71c268737e3bef3117514f9d4c0ea22c8d2dc0e837786e797f1.jpg](../iclr_results/1658_Amulet_%20ReAlignment%20During%20Test%20Time%20for%20Personalized%20Preference%20Adaptation%20of%20LLMs/images/2c242e75ed5eb71c268737e3bef3117514f9d4c0ea22c8d2dc0e837786e797f1.jpg)

![33efe83b9295e4f575f376c1a9904a8f2f34d5c0cb33791eee05dfeeec99615a.jpg](../iclr_results/1658_Amulet_%20ReAlignment%20During%20Test%20Time%20for%20Personalized%20Preference%20Adaptation%20of%20LLMs/images/33efe83b9295e4f575f376c1a9904a8f2f34d5c0cb33791eee05dfeeec99615a.jpg)

![61d93ad418f3733b826341d3080d2fd537743c799d69abc93a578a16b1857a72.jpg](../iclr_results/1658_Amulet_%20ReAlignment%20During%20Test%20Time%20for%20Personalized%20Preference%20Adaptation%20of%20LLMs/images/61d93ad418f3733b826341d3080d2fd537743c799d69abc93a578a16b1857a72.jpg)

![681bd97992ac878cf7d80dab6506c2abeb0a3bf43f168e8695ef08bb4a29f710.jpg](../iclr_results/1658_Amulet_%20ReAlignment%20During%20Test%20Time%20for%20Personalized%20Preference%20Adaptation%20of%20LLMs/images/681bd97992ac878cf7d80dab6506c2abeb0a3bf43f168e8695ef08bb4a29f710.jpg)

![8642cddbe546cd1e2b27a22957e8bde9a78b6827b19146f0aa3b67ed9310836b.jpg](../iclr_results/1658_Amulet_%20ReAlignment%20During%20Test%20Time%20for%20Personalized%20Preference%20Adaptation%20of%20LLMs/images/8642cddbe546cd1e2b27a22957e8bde9a78b6827b19146f0aa3b67ed9310836b.jpg)

![8c027ba28924fb3fe8903018a2fd101dd688f79b5daf8fb18e7c4e878649023e.jpg](../iclr_results/1658_Amulet_%20ReAlignment%20During%20Test%20Time%20for%20Personalized%20Preference%20Adaptation%20of%20LLMs/images/8c027ba28924fb3fe8903018a2fd101dd688f79b5daf8fb18e7c4e878649023e.jpg)

![94fc6267e0c0e089948cc2df2e7e8f3efa608945063c755191fff84d2e76477a.jpg](../iclr_results/1658_Amulet_%20ReAlignment%20During%20Test%20Time%20for%20Personalized%20Preference%20Adaptation%20of%20LLMs/images/94fc6267e0c0e089948cc2df2e7e8f3efa608945063c755191fff84d2e76477a.jpg)

![9767e360523fd5424132e5a9777db0a69fa58b8ac9fc6d271d826fe60fb78a36.jpg](../iclr_results/1658_Amulet_%20ReAlignment%20During%20Test%20Time%20for%20Personalized%20Preference%20Adaptation%20of%20LLMs/images/9767e360523fd5424132e5a9777db0a69fa58b8ac9fc6d271d826fe60fb78a36.jpg)

![df1cefbb93094c101f8fe816ba11e03f1b86639de351704eecb79f66471b3362.jpg](../iclr_results/1658_Amulet_%20ReAlignment%20During%20Test%20Time%20for%20Personalized%20Preference%20Adaptation%20of%20LLMs/images/df1cefbb93094c101f8fe816ba11e03f1b86639de351704eecb79f66471b3362.jpg)

![f7b2493a6e63919ba80e5233ab29167451d327a4821515b345604cf3a09cee12.jpg](../iclr_results/1658_Amulet_%20ReAlignment%20During%20Test%20Time%20for%20Personalized%20Preference%20Adaptation%20of%20LLMs/images/f7b2493a6e63919ba80e5233ab29167451d327a4821515b345604cf3a09cee12.jpg)

### Tables

![01280b9dcac843e6b6b96860301c49811284113e095db9b6cee1b0e95911092b.jpg](../iclr_results/1658_Amulet_%20ReAlignment%20During%20Test%20Time%20for%20Personalized%20Preference%20Adaptation%20of%20LLMs/tables/01280b9dcac843e6b6b96860301c49811284113e095db9b6cee1b0e95911092b.jpg)

![0d6e6f2a1db95574bf26fda324f797983b8b03301a231c3c5f507e8421390636.jpg](../iclr_results/1658_Amulet_%20ReAlignment%20During%20Test%20Time%20for%20Personalized%20Preference%20Adaptation%20of%20LLMs/tables/0d6e6f2a1db95574bf26fda324f797983b8b03301a231c3c5f507e8421390636.jpg)

![2974a977dbddc4113a1b453dff5be12aa062be3c8b46879f63f5347c65840554.jpg](../iclr_results/1658_Amulet_%20ReAlignment%20During%20Test%20Time%20for%20Personalized%20Preference%20Adaptation%20of%20LLMs/tables/2974a977dbddc4113a1b453dff5be12aa062be3c8b46879f63f5347c65840554.jpg)

![743a4ae696477bfdeb3b6da32e426e3ccfa75bcd5fdf7f9c42c1f062ed21bb75.jpg](../iclr_results/1658_Amulet_%20ReAlignment%20During%20Test%20Time%20for%20Personalized%20Preference%20Adaptation%20of%20LLMs/tables/743a4ae696477bfdeb3b6da32e426e3ccfa75bcd5fdf7f9c42c1f062ed21bb75.jpg)

![cff3b590108eb6cd3446baa26b1773591e771bc31bf764e517d3c2f578ebb0c4.jpg](../iclr_results/1658_Amulet_%20ReAlignment%20During%20Test%20Time%20for%20Personalized%20Preference%20Adaptation%20of%20LLMs/tables/cff3b590108eb6cd3446baa26b1773591e771bc31bf764e517d3c2f578ebb0c4.jpg)

![dd10402285cc5c3f32fbec8799249f2d4edda7334a0ea85a2203773f76d3d809.jpg](../iclr_results/1658_Amulet_%20ReAlignment%20During%20Test%20Time%20for%20Personalized%20Preference%20Adaptation%20of%20LLMs/tables/dd10402285cc5c3f32fbec8799249f2d4edda7334a0ea85a2203773f76d3d809.jpg)

![f42001fb05c81585a1f4f9c3af84384f34c611bbe9e6e43be7bc3c46f2877e3d.jpg](../iclr_results/1658_Amulet_%20ReAlignment%20During%20Test%20Time%20for%20Personalized%20Preference%20Adaptation%20of%20LLMs/tables/f42001fb05c81585a1f4f9c3af84384f34c611bbe9e6e43be7bc3c46f2877e3d.jpg)

## Controllable Satellite-to-Street-View Synthesis with Precise Pose Alignment and Zero-Shot Environmental Control


### Images

![05ee18f7b3842a89ec4c21ad973294914f1858477fdec9fe11d709aa59310291.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/images/05ee18f7b3842a89ec4c21ad973294914f1858477fdec9fe11d709aa59310291.jpg)

![2720b34c32b7854c13c1b7277b24be91a4ffcd2fd67f8d4fef4822334ae668b5.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/images/2720b34c32b7854c13c1b7277b24be91a4ffcd2fd67f8d4fef4822334ae668b5.jpg)

![43a574b28b338729a12ec53408d03c3480b14e22688cd2dcd40d71383b516af8.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/images/43a574b28b338729a12ec53408d03c3480b14e22688cd2dcd40d71383b516af8.jpg)

![51de9bcd2abe8c83dda0b20d7e03469e21966b2efe0445a1ae9aa1274cd21ef5.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/images/51de9bcd2abe8c83dda0b20d7e03469e21966b2efe0445a1ae9aa1274cd21ef5.jpg)

![5d4f09f47214da5305323243296ba81bc1c6a7acfb2fce05c0f33d958ee51863.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/images/5d4f09f47214da5305323243296ba81bc1c6a7acfb2fce05c0f33d958ee51863.jpg)

![61fa799611684e6e2e44e1cb41d5e44790c0d70c2ca566926586c5d16708ff74.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/images/61fa799611684e6e2e44e1cb41d5e44790c0d70c2ca566926586c5d16708ff74.jpg)

![6a81458640a80b47fc2d210497afcfd6a3b1bca46eab63206f3153bdd914c782.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/images/6a81458640a80b47fc2d210497afcfd6a3b1bca46eab63206f3153bdd914c782.jpg)

![78e3e1995f57a31ba4e1944ae3068243f2644bdcc484e9d99123d884f6048811.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/images/78e3e1995f57a31ba4e1944ae3068243f2644bdcc484e9d99123d884f6048811.jpg)

![982fc167671b0bb19b02c5eddb9c74d19b00cd63abe3a734e6723854d5c01e1e.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/images/982fc167671b0bb19b02c5eddb9c74d19b00cd63abe3a734e6723854d5c01e1e.jpg)

![a0daa6eb8706d15b0949e07cf65378ed862fa059ff215296b343be6adcdd3c2b.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/images/a0daa6eb8706d15b0949e07cf65378ed862fa059ff215296b343be6adcdd3c2b.jpg)

![ab250ad1d2888d6ca892c1b7d60956fcbfdb1973200c687e09e9fbad41c1c82b.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/images/ab250ad1d2888d6ca892c1b7d60956fcbfdb1973200c687e09e9fbad41c1c82b.jpg)

![b6d11557bd06d54b8fd7c299f2dd30e83be623f04b498b904847a2a6db9a7abc.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/images/b6d11557bd06d54b8fd7c299f2dd30e83be623f04b498b904847a2a6db9a7abc.jpg)

![bb0715ee4719c1da6833b201c0e185a5a2ad998c406c50f17c82ecc0001c6311.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/images/bb0715ee4719c1da6833b201c0e185a5a2ad998c406c50f17c82ecc0001c6311.jpg)

![c2e46e979172dce101005a2565f560f1d48ff660763266ec345a45a5311296b8.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/images/c2e46e979172dce101005a2565f560f1d48ff660763266ec345a45a5311296b8.jpg)

![ed1a16e0efac8db636f1ca25e6e74c2601eb732965a122796b32794ab87b2514.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/images/ed1a16e0efac8db636f1ca25e6e74c2601eb732965a122796b32794ab87b2514.jpg)

![ed1cb129760ce45e914af450eaec763a31f4c6e009e5b5b4e88e0344cbbef7dd.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/images/ed1cb129760ce45e914af450eaec763a31f4c6e009e5b5b4e88e0344cbbef7dd.jpg)

![f00625ccbf6f2c0f42b08a46079ec9c98423d22e73eb5deef36404b370ef39fc.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/images/f00625ccbf6f2c0f42b08a46079ec9c98423d22e73eb5deef36404b370ef39fc.jpg)

### Tables

![201de1ae9f867b6b9d9f46919fe99c31241795b1bab97f2b57a1c294aef418fd.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/tables/201de1ae9f867b6b9d9f46919fe99c31241795b1bab97f2b57a1c294aef418fd.jpg)

![26e46983fb47b26d8b5dd4b030e337de4e435e68e561d036f37d3235c4ce9b39.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/tables/26e46983fb47b26d8b5dd4b030e337de4e435e68e561d036f37d3235c4ce9b39.jpg)

![46df5b53b6c00f23e8dae5dd675703c3427c9beaca32fcedc6dd6aa3e3da6995.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/tables/46df5b53b6c00f23e8dae5dd675703c3427c9beaca32fcedc6dd6aa3e3da6995.jpg)

![57e5812f532563c16a8456bdf7e8d31e43c2c214532d2e43a8b79e3811b3ee27.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/tables/57e5812f532563c16a8456bdf7e8d31e43c2c214532d2e43a8b79e3811b3ee27.jpg)

![848288a8a60b6794edba7fa49bcb459b3341cb8bb6097e72f05ee8b4a4ec271d.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/tables/848288a8a60b6794edba7fa49bcb459b3341cb8bb6097e72f05ee8b4a4ec271d.jpg)

![a3e210346dc8103c5a1cb4ac58eb00830729b620c89f9c7ff5e62494a7e13a5d.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/tables/a3e210346dc8103c5a1cb4ac58eb00830729b620c89f9c7ff5e62494a7e13a5d.jpg)

![d7e876fa46ab1fe9648aab79325c09f8ec56855d1362bcc7298f035304a86564.jpg](../iclr_results/1659_Controllable%20Satellite-to-Street-View%20Synthesis%20with%20Precise%20Pose%20Alignment%20and%20Zero-Shot%20Environmen/tables/d7e876fa46ab1fe9648aab79325c09f8ec56855d1362bcc7298f035304a86564.jpg)

## MIA-DPO: Multi-Image Augmented Direct Preference Optimization For Large Vision-Language Models


### Images

![03850b3b1b8fe6e5cad6c3c6fa57aa1d2638fafb91f897ddb9363138035e302b.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/03850b3b1b8fe6e5cad6c3c6fa57aa1d2638fafb91f897ddb9363138035e302b.jpg)

![058868f6ba310b7a6d51338ec7adb71d3dbff2e5fadecca2bf571c00b67c3d32.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/058868f6ba310b7a6d51338ec7adb71d3dbff2e5fadecca2bf571c00b67c3d32.jpg)

![083fa084d24f344b5a637c02157eff90c72dec97123f18a64e3902b6034f747e.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/083fa084d24f344b5a637c02157eff90c72dec97123f18a64e3902b6034f747e.jpg)

![0b44d47eefccdc72410521f4d25ae10cdc1aff996881daae52c421bbeff2f99b.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/0b44d47eefccdc72410521f4d25ae10cdc1aff996881daae52c421bbeff2f99b.jpg)

![109ca82afa14cebe079e9017379c5c6320dab8670f9b4d265eba1ecd6acc4751.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/109ca82afa14cebe079e9017379c5c6320dab8670f9b4d265eba1ecd6acc4751.jpg)

![12769c98225a2ada23df7e6ffe3e7c6a95b27c2e1208d2eba50cb2a231b3c5de.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/12769c98225a2ada23df7e6ffe3e7c6a95b27c2e1208d2eba50cb2a231b3c5de.jpg)

![171b5641a5ee8e92a3ffe60dee580340d06f4da7f6d431ef5728f14010f10d09.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/171b5641a5ee8e92a3ffe60dee580340d06f4da7f6d431ef5728f14010f10d09.jpg)

![1ab112983e1c0160dd646550c14326a65e7bae06df22d084001e9df82f1e342b.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/1ab112983e1c0160dd646550c14326a65e7bae06df22d084001e9df82f1e342b.jpg)

![29a23cd9b53f819a7b33ae139cd6cf9faff5fcf8551c3516ac2f8edf65f03e5c.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/29a23cd9b53f819a7b33ae139cd6cf9faff5fcf8551c3516ac2f8edf65f03e5c.jpg)

![2f746178cf8beab793e5fc4c618877223312aded26110bef1c5740fdb6f5386c.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/2f746178cf8beab793e5fc4c618877223312aded26110bef1c5740fdb6f5386c.jpg)

![343cfada4ebc1cdc3a8ec02bf0ea2f2b8356e406f7bb68c6166bbe4b9cd39d10.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/343cfada4ebc1cdc3a8ec02bf0ea2f2b8356e406f7bb68c6166bbe4b9cd39d10.jpg)

![37d0a367008ea29f8157e33b591cc836e473878e8990380a95ae6d404edffa00.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/37d0a367008ea29f8157e33b591cc836e473878e8990380a95ae6d404edffa00.jpg)

![3d8489fa0cfbd0c5ac9465d3d5453587f5267e6f0109ff4c4cbe8856236ec777.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/3d8489fa0cfbd0c5ac9465d3d5453587f5267e6f0109ff4c4cbe8856236ec777.jpg)

![43b57a7ec754d4d447b128a6abd7fd5726385d577aa6c59ed4c0bdcabf4cf16c.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/43b57a7ec754d4d447b128a6abd7fd5726385d577aa6c59ed4c0bdcabf4cf16c.jpg)

![53c75ce03633a8776e523b434eaa409fe1c3f2e520b0860beaa150d9f567882d.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/53c75ce03633a8776e523b434eaa409fe1c3f2e520b0860beaa150d9f567882d.jpg)

![5c128c7bd2b27ad09a88e6434700bd1283047fa089c62a4e5fbba020dfaa0fb8.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/5c128c7bd2b27ad09a88e6434700bd1283047fa089c62a4e5fbba020dfaa0fb8.jpg)

![5f48076cbbbf5a7e23997cd80e2173fcb3721718c1a7f57af4a7c2958fc8cec1.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/5f48076cbbbf5a7e23997cd80e2173fcb3721718c1a7f57af4a7c2958fc8cec1.jpg)

![61f189e107dd49664c118d7ad22730a437c6369cac705e369de25235bf598e11.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/61f189e107dd49664c118d7ad22730a437c6369cac705e369de25235bf598e11.jpg)

![65432d3fd31814f7291350773fca59886e348541bee25df7ef1bd9917720bb7f.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/65432d3fd31814f7291350773fca59886e348541bee25df7ef1bd9917720bb7f.jpg)

![68e19406c9a8470daf14a144429a628ef1bcb3b12cb1f24d2671756d2833d238.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/68e19406c9a8470daf14a144429a628ef1bcb3b12cb1f24d2671756d2833d238.jpg)

![692833ef049c6390e22b229099b4e3468c1fa7e0de11a3ab587bbe0e4538b829.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/692833ef049c6390e22b229099b4e3468c1fa7e0de11a3ab587bbe0e4538b829.jpg)

![6a7f51fe00c5b38c2222012a2f3dbde6c697da08a5b8b4a528d9599ebbb16902.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/6a7f51fe00c5b38c2222012a2f3dbde6c697da08a5b8b4a528d9599ebbb16902.jpg)

![6b570e3665ae30970d0b12687dd905d2ae4b3ec5231cc654b2ff271791c3edf9.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/6b570e3665ae30970d0b12687dd905d2ae4b3ec5231cc654b2ff271791c3edf9.jpg)

![7389b772dfaa455ce5daecedf079a271907d7537b885bbac5c72d831b2e3fb96.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/7389b772dfaa455ce5daecedf079a271907d7537b885bbac5c72d831b2e3fb96.jpg)

![744fd50ed876aeaad5e19aaf0b7ccb025ee4fd6bcd424ff9260d5f855385d0f1.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/744fd50ed876aeaad5e19aaf0b7ccb025ee4fd6bcd424ff9260d5f855385d0f1.jpg)

![7a38fe34202b09b17b3af79bacf6c29cb683a11dea5c4174b237e3cddd0359cf.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/7a38fe34202b09b17b3af79bacf6c29cb683a11dea5c4174b237e3cddd0359cf.jpg)

![7b066c01d325e8873f4ceea79830a46f61d1938e946a8c4dcd05ee057a21066c.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/7b066c01d325e8873f4ceea79830a46f61d1938e946a8c4dcd05ee057a21066c.jpg)

![805a97f0ad4f87b9592ba414598d9e9f155b61eb98e3b567953cc02b93fc9c89.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/805a97f0ad4f87b9592ba414598d9e9f155b61eb98e3b567953cc02b93fc9c89.jpg)

![8061912cde2a36771f2e067cb49dc52734dbf24b5d3d74f662352035844b62b2.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/8061912cde2a36771f2e067cb49dc52734dbf24b5d3d74f662352035844b62b2.jpg)

![822d7c824ea56b004a8b3ef1d81fd1e1a29ed3eef956bcab8234e4b631b7615f.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/822d7c824ea56b004a8b3ef1d81fd1e1a29ed3eef956bcab8234e4b631b7615f.jpg)

![9061b7a86bd8910ddc30a40f120c290d8c10b4764e44c3651dea53b566fe3510.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/9061b7a86bd8910ddc30a40f120c290d8c10b4764e44c3651dea53b566fe3510.jpg)

![9225dc47fba383e765410a4a03b78cef45a316667b24d77a050436c9429d6281.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/9225dc47fba383e765410a4a03b78cef45a316667b24d77a050436c9429d6281.jpg)

![9438880521f753659e6ea1bd36ba09c12951fb24f9b2cae28db5e0970ecf11ac.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/9438880521f753659e6ea1bd36ba09c12951fb24f9b2cae28db5e0970ecf11ac.jpg)

![98eed7c618b9ba8f2d1e25c7b7be2a9c5578bcb9ac5762480d24cb39bf7abdc8.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/98eed7c618b9ba8f2d1e25c7b7be2a9c5578bcb9ac5762480d24cb39bf7abdc8.jpg)

![9c4596fd9cbf7a8026e27e56ab44d1d7c0254e943ecee141294a19a1c707d468.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/9c4596fd9cbf7a8026e27e56ab44d1d7c0254e943ecee141294a19a1c707d468.jpg)

![9e53eee9cfe3fc6d3fb70d9a9bfe1d8089639da870d5293d8ccf30e98a188d7b.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/9e53eee9cfe3fc6d3fb70d9a9bfe1d8089639da870d5293d8ccf30e98a188d7b.jpg)

![a20d0b47e0ddbd9ef65d0468918ed6944072887aae9f85ed558fb6c25fe61304.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/a20d0b47e0ddbd9ef65d0468918ed6944072887aae9f85ed558fb6c25fe61304.jpg)

![a867a947c43fba279140aae18939541cf9c8909ae428321a34d57a987caa9f3c.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/a867a947c43fba279140aae18939541cf9c8909ae428321a34d57a987caa9f3c.jpg)

![ad76d63bc17a5510451320ced0fa31745477ce068b786d77145b2681665f5d02.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/ad76d63bc17a5510451320ced0fa31745477ce068b786d77145b2681665f5d02.jpg)

![aec3cb17d6ce56ffd1f4b09e853c16c4a81cdb8fa47405ea19ee3ac05d1bff54.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/aec3cb17d6ce56ffd1f4b09e853c16c4a81cdb8fa47405ea19ee3ac05d1bff54.jpg)

![b1e4f677a42e4ec4c073bd4f19c28ea6ddaf30b31fa3ef8543b730627c821630.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/b1e4f677a42e4ec4c073bd4f19c28ea6ddaf30b31fa3ef8543b730627c821630.jpg)

![ba098cd9989a6de41aa8162dba448b630d7aab0ed640c74052bb1b6f7ec6838d.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/ba098cd9989a6de41aa8162dba448b630d7aab0ed640c74052bb1b6f7ec6838d.jpg)

![bb9bdeaf17d82170b2a7cd146fb843a51dce606f3031c7811e8cfc4a2aeef2b4.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/bb9bdeaf17d82170b2a7cd146fb843a51dce606f3031c7811e8cfc4a2aeef2b4.jpg)

![bd4019c51b1426404df225e58157691f3fecf66404100dd736e177c02008e0c0.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/bd4019c51b1426404df225e58157691f3fecf66404100dd736e177c02008e0c0.jpg)

![d1147a4a240b53625da8b1d13c331e2b2587bfe76c77137ef0b0bfd0390735a4.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/d1147a4a240b53625da8b1d13c331e2b2587bfe76c77137ef0b0bfd0390735a4.jpg)

![d6bef4e53e7ab306ddc127df3e99e80f61bfce91fc6a563d00cc5bbba5cc4575.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/d6bef4e53e7ab306ddc127df3e99e80f61bfce91fc6a563d00cc5bbba5cc4575.jpg)

![e0225a3d634992d66d1b7b16e8bb2c487ebde98d9a80d393e991fce728fb55df.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/e0225a3d634992d66d1b7b16e8bb2c487ebde98d9a80d393e991fce728fb55df.jpg)

![e4fd62cc9985590cc1f1ca2915625d6a221d86a275fc9726b85011216f31952c.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/e4fd62cc9985590cc1f1ca2915625d6a221d86a275fc9726b85011216f31952c.jpg)

![e53f601f8dc2cd3512f3792e8c011dfccc3255e35e9d833018d88339a3156c3b.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/e53f601f8dc2cd3512f3792e8c011dfccc3255e35e9d833018d88339a3156c3b.jpg)

![e776f56ba1ce1f22f8e28fb1ce20221dff08ea23b6ddbc4e66abe2eaf7280fc1.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/e776f56ba1ce1f22f8e28fb1ce20221dff08ea23b6ddbc4e66abe2eaf7280fc1.jpg)

![e8eb39d58120605dc1c386ddb3f9ba49e18b6ea4a1047e6c6a6e4743eb069133.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/e8eb39d58120605dc1c386ddb3f9ba49e18b6ea4a1047e6c6a6e4743eb069133.jpg)

![f5d574dc0f4b3ee84b658e1f56fae9c2b9175b7e39a96377329fe2f2e8994e9b.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/images/f5d574dc0f4b3ee84b658e1f56fae9c2b9175b7e39a96377329fe2f2e8994e9b.jpg)

### Tables

![1304c62ebb3a2cc70cab06e174dd406c3c15bb4f70ce2d2a99df4649d17e52a9.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/tables/1304c62ebb3a2cc70cab06e174dd406c3c15bb4f70ce2d2a99df4649d17e52a9.jpg)

![1b2143403a6eba655fa792158d528c1312c171bd90e6a6582aa1a9222904d976.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/tables/1b2143403a6eba655fa792158d528c1312c171bd90e6a6582aa1a9222904d976.jpg)

![1c378d26eda0349241b56b7378b85423e9bec08935af1d94efa16ae2b05a814c.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/tables/1c378d26eda0349241b56b7378b85423e9bec08935af1d94efa16ae2b05a814c.jpg)

![5bf2434773c94c850d4c75ef07f8c87504d7ec282e7ca6d883103c3256b7d24f.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/tables/5bf2434773c94c850d4c75ef07f8c87504d7ec282e7ca6d883103c3256b7d24f.jpg)

![62e191ee10c1faffbbeb3603f06563a4c54a8aca8a1cde882436c4bf12550347.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/tables/62e191ee10c1faffbbeb3603f06563a4c54a8aca8a1cde882436c4bf12550347.jpg)

![6e68a76f591f4ba054e8877fb71072437e1ce47c88d5ba67ed857617dc294de6.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/tables/6e68a76f591f4ba054e8877fb71072437e1ce47c88d5ba67ed857617dc294de6.jpg)

![74ad4f18afee06ae15fe1fb6892af6654eaab623a20c21c3d027703ad8908008.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/tables/74ad4f18afee06ae15fe1fb6892af6654eaab623a20c21c3d027703ad8908008.jpg)

![830fbb452c1b0c46605a4d55cc1ff39be6000e2f70deff64ede8de45576cbfb8.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/tables/830fbb452c1b0c46605a4d55cc1ff39be6000e2f70deff64ede8de45576cbfb8.jpg)

![8da1e3a1c42cbe699b721ebafa96428ec9615e3b951c8645e42a41e8f2b59615.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/tables/8da1e3a1c42cbe699b721ebafa96428ec9615e3b951c8645e42a41e8f2b59615.jpg)

![8f6ecde7249208e420c2f529099d280c09e75ff26723d2eab9d0861aa93e41d7.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/tables/8f6ecde7249208e420c2f529099d280c09e75ff26723d2eab9d0861aa93e41d7.jpg)

![a0630e10de8cf99f0c3db7191185cc18509c33da0be01f65f5d6346677cd9062.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/tables/a0630e10de8cf99f0c3db7191185cc18509c33da0be01f65f5d6346677cd9062.jpg)

![ab051ab4d636dea4e73fbf95fbed06aa8d3ea06915964502eb061179edbd5f53.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/tables/ab051ab4d636dea4e73fbf95fbed06aa8d3ea06915964502eb061179edbd5f53.jpg)

![b5cdc61aa71d2ab92d565bd21ceaf7da18f80429721b335c52934b9ee3369c34.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/tables/b5cdc61aa71d2ab92d565bd21ceaf7da18f80429721b335c52934b9ee3369c34.jpg)

![b9d40b654e00057b239c5456179438339ca4598feed57b65b80ea6ecea20afe5.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/tables/b9d40b654e00057b239c5456179438339ca4598feed57b65b80ea6ecea20afe5.jpg)

![df3f556f3d4527bd112464f75bf9772f15dcf92cd45fdd9d06f1a4cde04f81ec.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/tables/df3f556f3d4527bd112464f75bf9772f15dcf92cd45fdd9d06f1a4cde04f81ec.jpg)

![eb68d4ce6bc90b606d151acae23b4e2959855c91223b09bda9c63e1be2efdca0.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/tables/eb68d4ce6bc90b606d151acae23b4e2959855c91223b09bda9c63e1be2efdca0.jpg)

![ed1e73a42e24794da5fd215379ff0496b25912e2a467f42da9df0e64ff77ca73.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/tables/ed1e73a42e24794da5fd215379ff0496b25912e2a467f42da9df0e64ff77ca73.jpg)

![f52b0c7834055b08d12569dbf9d1396e24734b3cd8ecd9aed6b1a779c5453aa6.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/tables/f52b0c7834055b08d12569dbf9d1396e24734b3cd8ecd9aed6b1a779c5453aa6.jpg)

![fad326d1fbbcf7fd22c7ce519db900fdd292ea61fe09c447566ed9cb60b0d813.jpg](../iclr_results/1660_MIA-DPO_%20Multi-Image%20Augmented%20Direct%20Preference%20Optimization%20For%20Large%20Vision-Language%20Models/tables/fad326d1fbbcf7fd22c7ce519db900fdd292ea61fe09c447566ed9cb60b0d813.jpg)

## API Pack: A Massive Multi-Programming Language Dataset for API Call Generation


### Images

![075bdebb82d3ec874cbf9874af3b743052d60ea3f7c2f20e54b790b067c24870.jpg](../iclr_results/1661_API%20Pack_%20A%20Massive%20Multi-Programming%20Language%20Dataset%20for%20API%20Call%20Generation/images/075bdebb82d3ec874cbf9874af3b743052d60ea3f7c2f20e54b790b067c24870.jpg)

![0c11b87642941e111402ea94e78d288849adf5f594ae28e8c7119f392e9ad74a.jpg](../iclr_results/1661_API%20Pack_%20A%20Massive%20Multi-Programming%20Language%20Dataset%20for%20API%20Call%20Generation/images/0c11b87642941e111402ea94e78d288849adf5f594ae28e8c7119f392e9ad74a.jpg)

![149958a63af85aef40a6ceb11a06ba0ff5f98c3f2530036458cbc8976b0a41ac.jpg](../iclr_results/1661_API%20Pack_%20A%20Massive%20Multi-Programming%20Language%20Dataset%20for%20API%20Call%20Generation/images/149958a63af85aef40a6ceb11a06ba0ff5f98c3f2530036458cbc8976b0a41ac.jpg)

![49813003479398e3cf361658a2c10a5b4d62c0bde438c9a222788027553309de.jpg](../iclr_results/1661_API%20Pack_%20A%20Massive%20Multi-Programming%20Language%20Dataset%20for%20API%20Call%20Generation/images/49813003479398e3cf361658a2c10a5b4d62c0bde438c9a222788027553309de.jpg)

![61672b2c9bced469e4acc6a21c6566ac0a56f8e8963fbe219f1345612696c008.jpg](../iclr_results/1661_API%20Pack_%20A%20Massive%20Multi-Programming%20Language%20Dataset%20for%20API%20Call%20Generation/images/61672b2c9bced469e4acc6a21c6566ac0a56f8e8963fbe219f1345612696c008.jpg)

![9b30a384130e2d38d1c5d565775cd47ee5f8eff732620247b144e22e40afdbdc.jpg](../iclr_results/1661_API%20Pack_%20A%20Massive%20Multi-Programming%20Language%20Dataset%20for%20API%20Call%20Generation/images/9b30a384130e2d38d1c5d565775cd47ee5f8eff732620247b144e22e40afdbdc.jpg)

![c87bb53d8f6dd6dd1f586f0fccbe439bd9095f6e2ac427f9b71b4016f7250722.jpg](../iclr_results/1661_API%20Pack_%20A%20Massive%20Multi-Programming%20Language%20Dataset%20for%20API%20Call%20Generation/images/c87bb53d8f6dd6dd1f586f0fccbe439bd9095f6e2ac427f9b71b4016f7250722.jpg)

![ce0cf57ec470517cf8c55cc04cfea320925c1d6577545a6fbdc8c87c8a80bcb7.jpg](../iclr_results/1661_API%20Pack_%20A%20Massive%20Multi-Programming%20Language%20Dataset%20for%20API%20Call%20Generation/images/ce0cf57ec470517cf8c55cc04cfea320925c1d6577545a6fbdc8c87c8a80bcb7.jpg)

### Tables

![0917dcf681c1d3e96d5299b3d3a6723027e1de1aa40f457f3b296ecf843c425d.jpg](../iclr_results/1661_API%20Pack_%20A%20Massive%20Multi-Programming%20Language%20Dataset%20for%20API%20Call%20Generation/tables/0917dcf681c1d3e96d5299b3d3a6723027e1de1aa40f457f3b296ecf843c425d.jpg)

![5d024931dd065e8186f24afdf09c6430c8d227cb14bf5a2df777cfbf25fb9846.jpg](../iclr_results/1661_API%20Pack_%20A%20Massive%20Multi-Programming%20Language%20Dataset%20for%20API%20Call%20Generation/tables/5d024931dd065e8186f24afdf09c6430c8d227cb14bf5a2df777cfbf25fb9846.jpg)

![5deafea7d9c70e21edf2b022f3559e08f1a9a70d8d8d39c5f148516cc7217af5.jpg](../iclr_results/1661_API%20Pack_%20A%20Massive%20Multi-Programming%20Language%20Dataset%20for%20API%20Call%20Generation/tables/5deafea7d9c70e21edf2b022f3559e08f1a9a70d8d8d39c5f148516cc7217af5.jpg)

![5f3c0ca2a9d5183d31e91cac58bfbe7f54f86a03af77995e1f49fb490fee9dcc.jpg](../iclr_results/1661_API%20Pack_%20A%20Massive%20Multi-Programming%20Language%20Dataset%20for%20API%20Call%20Generation/tables/5f3c0ca2a9d5183d31e91cac58bfbe7f54f86a03af77995e1f49fb490fee9dcc.jpg)

![74276e86775ba3031eabd7d194e596d8ff2a9b4a1c74b54539f3d5ad2aedff32.jpg](../iclr_results/1661_API%20Pack_%20A%20Massive%20Multi-Programming%20Language%20Dataset%20for%20API%20Call%20Generation/tables/74276e86775ba3031eabd7d194e596d8ff2a9b4a1c74b54539f3d5ad2aedff32.jpg)

![7c6022c936a9f01aa5d648d8cc2b268d1b9077445858badc71186639962cce94.jpg](../iclr_results/1661_API%20Pack_%20A%20Massive%20Multi-Programming%20Language%20Dataset%20for%20API%20Call%20Generation/tables/7c6022c936a9f01aa5d648d8cc2b268d1b9077445858badc71186639962cce94.jpg)

![a8fc5fb3d26ea12e8d752fc8838f7f7c538d2cf0c9c7e5ad92a9574907c5a6a9.jpg](../iclr_results/1661_API%20Pack_%20A%20Massive%20Multi-Programming%20Language%20Dataset%20for%20API%20Call%20Generation/tables/a8fc5fb3d26ea12e8d752fc8838f7f7c538d2cf0c9c7e5ad92a9574907c5a6a9.jpg)

![e9af02e7570544be73c2574b5bf4368c1d4a0cf2128cdf1bc2305fd0a1811101.jpg](../iclr_results/1661_API%20Pack_%20A%20Massive%20Multi-Programming%20Language%20Dataset%20for%20API%20Call%20Generation/tables/e9af02e7570544be73c2574b5bf4368c1d4a0cf2128cdf1bc2305fd0a1811101.jpg)

## Capability Localization: Capabilities Can be Localized rather than Individual Knowledge


### Images

![24c54dee1a76dc6d7c9c5fa3c041357a49aa85cb3fa49806f9e4ca8c80b60012.jpg](../iclr_results/1662_Capability%20Localization_%20Capabilities%20Can%20be%20Localized%20rather%20than%20Individual%20Knowledge/images/24c54dee1a76dc6d7c9c5fa3c041357a49aa85cb3fa49806f9e4ca8c80b60012.jpg)

![2db426979a7bdafcf9c765e6b65377d8bd986a31048fcf4c684ddf6ea2e9c3dc.jpg](../iclr_results/1662_Capability%20Localization_%20Capabilities%20Can%20be%20Localized%20rather%20than%20Individual%20Knowledge/images/2db426979a7bdafcf9c765e6b65377d8bd986a31048fcf4c684ddf6ea2e9c3dc.jpg)

![2e685cdfd28b7b6dd7f6a11f50e489e5b817d695c245f11f0ae1e30754042d75.jpg](../iclr_results/1662_Capability%20Localization_%20Capabilities%20Can%20be%20Localized%20rather%20than%20Individual%20Knowledge/images/2e685cdfd28b7b6dd7f6a11f50e489e5b817d695c245f11f0ae1e30754042d75.jpg)

![35c92eed39ad63ace79b163e2d2dfe396907ec9d80c7713d6590a7136263f4f5.jpg](../iclr_results/1662_Capability%20Localization_%20Capabilities%20Can%20be%20Localized%20rather%20than%20Individual%20Knowledge/images/35c92eed39ad63ace79b163e2d2dfe396907ec9d80c7713d6590a7136263f4f5.jpg)

![4640fc50819b1e702e4ae227ccb605a2eeb36d97173ab10d649be00f6cc036e2.jpg](../iclr_results/1662_Capability%20Localization_%20Capabilities%20Can%20be%20Localized%20rather%20than%20Individual%20Knowledge/images/4640fc50819b1e702e4ae227ccb605a2eeb36d97173ab10d649be00f6cc036e2.jpg)

![5d2e471fd3a53d033ca76962c72bf6dfc6305792acbd3b7618dfc7ff55b41b95.jpg](../iclr_results/1662_Capability%20Localization_%20Capabilities%20Can%20be%20Localized%20rather%20than%20Individual%20Knowledge/images/5d2e471fd3a53d033ca76962c72bf6dfc6305792acbd3b7618dfc7ff55b41b95.jpg)

![a87096a9a2a2a5d7ff3eaf5b21db9dd7b11656bb28256b67b26251c2a3e7eaef.jpg](../iclr_results/1662_Capability%20Localization_%20Capabilities%20Can%20be%20Localized%20rather%20than%20Individual%20Knowledge/images/a87096a9a2a2a5d7ff3eaf5b21db9dd7b11656bb28256b67b26251c2a3e7eaef.jpg)

![be093743f266a770040a220f1c9063a914d216500ed2ca5323d66d67e5d8752b.jpg](../iclr_results/1662_Capability%20Localization_%20Capabilities%20Can%20be%20Localized%20rather%20than%20Individual%20Knowledge/images/be093743f266a770040a220f1c9063a914d216500ed2ca5323d66d67e5d8752b.jpg)

![e056599925a900d7284058e6727d4628769b3c1f2b60e564aa6e9d299a0d3d02.jpg](../iclr_results/1662_Capability%20Localization_%20Capabilities%20Can%20be%20Localized%20rather%20than%20Individual%20Knowledge/images/e056599925a900d7284058e6727d4628769b3c1f2b60e564aa6e9d299a0d3d02.jpg)

### Tables

![24f816e0d6b214767720c82797dd7f62846fe1012cc9b119238ac8f9a2d1b04b.jpg](../iclr_results/1662_Capability%20Localization_%20Capabilities%20Can%20be%20Localized%20rather%20than%20Individual%20Knowledge/tables/24f816e0d6b214767720c82797dd7f62846fe1012cc9b119238ac8f9a2d1b04b.jpg)

![3ed2085a694cf5f6c66cf681848ee1705db1a7fbae0a949415ff06519d239fe5.jpg](../iclr_results/1662_Capability%20Localization_%20Capabilities%20Can%20be%20Localized%20rather%20than%20Individual%20Knowledge/tables/3ed2085a694cf5f6c66cf681848ee1705db1a7fbae0a949415ff06519d239fe5.jpg)

![7fb51cfbde04e83bac89da7774a7d2e97b61f0166cb8427b4c5fa071b294c7e9.jpg](../iclr_results/1662_Capability%20Localization_%20Capabilities%20Can%20be%20Localized%20rather%20than%20Individual%20Knowledge/tables/7fb51cfbde04e83bac89da7774a7d2e97b61f0166cb8427b4c5fa071b294c7e9.jpg)

![b897f3609544a2b7e4e9e6dce75dfb754908c40b42998de3c75e450feda2a638.jpg](../iclr_results/1662_Capability%20Localization_%20Capabilities%20Can%20be%20Localized%20rather%20than%20Individual%20Knowledge/tables/b897f3609544a2b7e4e9e6dce75dfb754908c40b42998de3c75e450feda2a638.jpg)

![c8f7a0d949aa5f3857d2058230bb5a6fedc5ee3b6a7a903a890f999f1a0fdb56.jpg](../iclr_results/1662_Capability%20Localization_%20Capabilities%20Can%20be%20Localized%20rather%20than%20Individual%20Knowledge/tables/c8f7a0d949aa5f3857d2058230bb5a6fedc5ee3b6a7a903a890f999f1a0fdb56.jpg)

## Federated Continual Learning Goes Online: Uncertainty-Aware Memory Management for Vision Tasks and Beyond


### Images

![07fc78c2b59b52d6119569a0ba1886b4e2abf4bb41ea6de76cd8fb75f535378f.jpg](../iclr_results/1663_Federated%20Continual%20Learning%20Goes%20Online_%20Uncertainty-Aware%20Memory%20Management%20for%20Vision%20Tasks%20and%20B/images/07fc78c2b59b52d6119569a0ba1886b4e2abf4bb41ea6de76cd8fb75f535378f.jpg)

![6dfdeaf29668ebf7691ab516cbe3892852378a27e8ad6698eb92b655e0ce463f.jpg](../iclr_results/1663_Federated%20Continual%20Learning%20Goes%20Online_%20Uncertainty-Aware%20Memory%20Management%20for%20Vision%20Tasks%20and%20B/images/6dfdeaf29668ebf7691ab516cbe3892852378a27e8ad6698eb92b655e0ce463f.jpg)

![9ddec543fbaeb3dba357a56e68d206eb5c41058fbeedf108136c2ba7a9ed1e75.jpg](../iclr_results/1663_Federated%20Continual%20Learning%20Goes%20Online_%20Uncertainty-Aware%20Memory%20Management%20for%20Vision%20Tasks%20and%20B/images/9ddec543fbaeb3dba357a56e68d206eb5c41058fbeedf108136c2ba7a9ed1e75.jpg)

![bcb5dbfaf7b0b80487df57639eddc43b3bb748048b37445f0eb4750ab217e159.jpg](../iclr_results/1663_Federated%20Continual%20Learning%20Goes%20Online_%20Uncertainty-Aware%20Memory%20Management%20for%20Vision%20Tasks%20and%20B/images/bcb5dbfaf7b0b80487df57639eddc43b3bb748048b37445f0eb4750ab217e159.jpg)

![bd1d58578bae091ba003f7aa744831d1e21fe5cd70b01e2d59042e31ad8d2755.jpg](../iclr_results/1663_Federated%20Continual%20Learning%20Goes%20Online_%20Uncertainty-Aware%20Memory%20Management%20for%20Vision%20Tasks%20and%20B/images/bd1d58578bae091ba003f7aa744831d1e21fe5cd70b01e2d59042e31ad8d2755.jpg)

### Tables

![0eeb89791e80b102e9d19a7aeb74fe8c3390c0f4250e06fe58e31fb951bb557f.jpg](../iclr_results/1663_Federated%20Continual%20Learning%20Goes%20Online_%20Uncertainty-Aware%20Memory%20Management%20for%20Vision%20Tasks%20and%20B/tables/0eeb89791e80b102e9d19a7aeb74fe8c3390c0f4250e06fe58e31fb951bb557f.jpg)

![26bb029716abf097d844918bbf7112b674ba313e55ba1675bf01de379bdfb4cf.jpg](../iclr_results/1663_Federated%20Continual%20Learning%20Goes%20Online_%20Uncertainty-Aware%20Memory%20Management%20for%20Vision%20Tasks%20and%20B/tables/26bb029716abf097d844918bbf7112b674ba313e55ba1675bf01de379bdfb4cf.jpg)

![3342950a781a26cd2014737dda3b12ce5ee35f1f9eadfe0efe4f8654392c0035.jpg](../iclr_results/1663_Federated%20Continual%20Learning%20Goes%20Online_%20Uncertainty-Aware%20Memory%20Management%20for%20Vision%20Tasks%20and%20B/tables/3342950a781a26cd2014737dda3b12ce5ee35f1f9eadfe0efe4f8654392c0035.jpg)

![48af28987eafd3c6ef740c47cd3335401b678c2cbbc7f052d114e8b14c231ff5.jpg](../iclr_results/1663_Federated%20Continual%20Learning%20Goes%20Online_%20Uncertainty-Aware%20Memory%20Management%20for%20Vision%20Tasks%20and%20B/tables/48af28987eafd3c6ef740c47cd3335401b678c2cbbc7f052d114e8b14c231ff5.jpg)

![48d68317fc06ca6aae325e608d715f0dd6c9e5550daf548d0f5972b2cf1da588.jpg](../iclr_results/1663_Federated%20Continual%20Learning%20Goes%20Online_%20Uncertainty-Aware%20Memory%20Management%20for%20Vision%20Tasks%20and%20B/tables/48d68317fc06ca6aae325e608d715f0dd6c9e5550daf548d0f5972b2cf1da588.jpg)

![4bdcc0895b103ba013d645db2687b129a570f012d01add81c129ba4481447c35.jpg](../iclr_results/1663_Federated%20Continual%20Learning%20Goes%20Online_%20Uncertainty-Aware%20Memory%20Management%20for%20Vision%20Tasks%20and%20B/tables/4bdcc0895b103ba013d645db2687b129a570f012d01add81c129ba4481447c35.jpg)

![57c44145f63769dacdb7007be6e6e12a364f5721a265c3a4a3b8295597c52e02.jpg](../iclr_results/1663_Federated%20Continual%20Learning%20Goes%20Online_%20Uncertainty-Aware%20Memory%20Management%20for%20Vision%20Tasks%20and%20B/tables/57c44145f63769dacdb7007be6e6e12a364f5721a265c3a4a3b8295597c52e02.jpg)

![7630ce439799a47e81944ec558cb65e0b3676e7aa13f62e5287bca2eb1e6eb85.jpg](../iclr_results/1663_Federated%20Continual%20Learning%20Goes%20Online_%20Uncertainty-Aware%20Memory%20Management%20for%20Vision%20Tasks%20and%20B/tables/7630ce439799a47e81944ec558cb65e0b3676e7aa13f62e5287bca2eb1e6eb85.jpg)

![94a17f0408dd73ced5fbc4331cce3e1bd79d4c31aa14df312b88c218a299eefa.jpg](../iclr_results/1663_Federated%20Continual%20Learning%20Goes%20Online_%20Uncertainty-Aware%20Memory%20Management%20for%20Vision%20Tasks%20and%20B/tables/94a17f0408dd73ced5fbc4331cce3e1bd79d4c31aa14df312b88c218a299eefa.jpg)

![9a1784d893d25ddeafd461cda84e9b19c6fd9d899409640907aeaf9b5f32e7f5.jpg](../iclr_results/1663_Federated%20Continual%20Learning%20Goes%20Online_%20Uncertainty-Aware%20Memory%20Management%20for%20Vision%20Tasks%20and%20B/tables/9a1784d893d25ddeafd461cda84e9b19c6fd9d899409640907aeaf9b5f32e7f5.jpg)

![d52428b0965ef13889ee93b55cbd86ce80af992a565bb7042e067bbd80d41a1e.jpg](../iclr_results/1663_Federated%20Continual%20Learning%20Goes%20Online_%20Uncertainty-Aware%20Memory%20Management%20for%20Vision%20Tasks%20and%20B/tables/d52428b0965ef13889ee93b55cbd86ce80af992a565bb7042e067bbd80d41a1e.jpg)

![d9e623ac2e4eccafe21140d8d0e2d407c74c532d666c4aed530b54f2ed1eafad.jpg](../iclr_results/1663_Federated%20Continual%20Learning%20Goes%20Online_%20Uncertainty-Aware%20Memory%20Management%20for%20Vision%20Tasks%20and%20B/tables/d9e623ac2e4eccafe21140d8d0e2d407c74c532d666c4aed530b54f2ed1eafad.jpg)

![dcd431db1eb6bb00a721e3af7fbe6ec50cfb707500879f27b36c1cfc62c743ca.jpg](../iclr_results/1663_Federated%20Continual%20Learning%20Goes%20Online_%20Uncertainty-Aware%20Memory%20Management%20for%20Vision%20Tasks%20and%20B/tables/dcd431db1eb6bb00a721e3af7fbe6ec50cfb707500879f27b36c1cfc62c743ca.jpg)

![e90fa2f4036fe542d32e284d71235b4bd69e3b353fead6c55b2b48db7cfec5ab.jpg](../iclr_results/1663_Federated%20Continual%20Learning%20Goes%20Online_%20Uncertainty-Aware%20Memory%20Management%20for%20Vision%20Tasks%20and%20B/tables/e90fa2f4036fe542d32e284d71235b4bd69e3b353fead6c55b2b48db7cfec5ab.jpg)

![e960c93503a0cdb04a3cf75115f89bf9070496b222695280286e998a9086c26a.jpg](../iclr_results/1663_Federated%20Continual%20Learning%20Goes%20Online_%20Uncertainty-Aware%20Memory%20Management%20for%20Vision%20Tasks%20and%20B/tables/e960c93503a0cdb04a3cf75115f89bf9070496b222695280286e998a9086c26a.jpg)

![eec98f2e5fb81ad288a21e5d0930b7f2ad98d219b354c4ca53077b94fbb7715e.jpg](../iclr_results/1663_Federated%20Continual%20Learning%20Goes%20Online_%20Uncertainty-Aware%20Memory%20Management%20for%20Vision%20Tasks%20and%20B/tables/eec98f2e5fb81ad288a21e5d0930b7f2ad98d219b354c4ca53077b94fbb7715e.jpg)

## Lasso Bandit with Compatibility Condition on Optimal Arm


### Images

![115633b98c5d720df560c104c1a67f1d1fc97b82cfeec85279ec3d9fdb077e1e.jpg](../iclr_results/1664_Lasso%20Bandit%20with%20Compatibility%20Condition%20on%20Optimal%20Arm/images/115633b98c5d720df560c104c1a67f1d1fc97b82cfeec85279ec3d9fdb077e1e.jpg)

![a638fe1fa07972de86ea1611c3cee31014868cdc7f461c5105c650130e3fa485.jpg](../iclr_results/1664_Lasso%20Bandit%20with%20Compatibility%20Condition%20on%20Optimal%20Arm/images/a638fe1fa07972de86ea1611c3cee31014868cdc7f461c5105c650130e3fa485.jpg)

![d82a8a0ef38c46019d36875f9b40b6036eb73a5182c45c5ba5f61c0cab253cec.jpg](../iclr_results/1664_Lasso%20Bandit%20with%20Compatibility%20Condition%20on%20Optimal%20Arm/images/d82a8a0ef38c46019d36875f9b40b6036eb73a5182c45c5ba5f61c0cab253cec.jpg)

![e21608b37a95b109e9b531399e3021bf54a8b5c633b5cfa1d7cbbc5c7ee3a665.jpg](../iclr_results/1664_Lasso%20Bandit%20with%20Compatibility%20Condition%20on%20Optimal%20Arm/images/e21608b37a95b109e9b531399e3021bf54a8b5c633b5cfa1d7cbbc5c7ee3a665.jpg)

### Tables

![515e1d3fdd9d5b2b71ebdb74b1b7981c7b684dd773e144e219f29d4a167098ae.jpg](../iclr_results/1664_Lasso%20Bandit%20with%20Compatibility%20Condition%20on%20Optimal%20Arm/tables/515e1d3fdd9d5b2b71ebdb74b1b7981c7b684dd773e144e219f29d4a167098ae.jpg)

![8e2e6e596ef4858876edf711debd631fd12c5b156caf41bb0b118c68558b22b8.jpg](../iclr_results/1664_Lasso%20Bandit%20with%20Compatibility%20Condition%20on%20Optimal%20Arm/tables/8e2e6e596ef4858876edf711debd631fd12c5b156caf41bb0b118c68558b22b8.jpg)

## Safety-Prioritizing Curricula for Constrained Reinforcement Learning


### Images

![1b318e05b4b041199372df0da787d0362f1a427dafd618baf571f290ec4e5ac8.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/1b318e05b4b041199372df0da787d0362f1a427dafd618baf571f290ec4e5ac8.jpg)

![202211d3dcb56c0d16ffcaa10bc739842063e62faeefc83ae1fe8a06d7ee4c50.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/202211d3dcb56c0d16ffcaa10bc739842063e62faeefc83ae1fe8a06d7ee4c50.jpg)

![339b3179a561b5ead30e759de36ae5ffa2ecd28331cb47e5ade27591650e1164.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/339b3179a561b5ead30e759de36ae5ffa2ecd28331cb47e5ade27591650e1164.jpg)

![3a21fb24cba40fae4df787f7e22d75cbad98cdfa05fcfa2634c237666c893c27.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/3a21fb24cba40fae4df787f7e22d75cbad98cdfa05fcfa2634c237666c893c27.jpg)

![532788c7bc162a4c806a7da2124ecc20e4ed150ef380cacc7a024accf3067d1a.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/532788c7bc162a4c806a7da2124ecc20e4ed150ef380cacc7a024accf3067d1a.jpg)

![757ea72548fa578b770861e694d2eb92ac5d10369b78fabf71d8693b2a864eb1.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/757ea72548fa578b770861e694d2eb92ac5d10369b78fabf71d8693b2a864eb1.jpg)

![795f75e954539b62ffa2c5c0a957e0dbc49b6ffc957a23ba0328343272b7a18f.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/795f75e954539b62ffa2c5c0a957e0dbc49b6ffc957a23ba0328343272b7a18f.jpg)

![857355ed89e06101a110ff5e15bfbccd81b72679cb935bba939e703a8c3ba946.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/857355ed89e06101a110ff5e15bfbccd81b72679cb935bba939e703a8c3ba946.jpg)

![8aaa1cabb8e85fd9027b000af764fdce93dfc4c3e13289e1d03f1165c337d914.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/8aaa1cabb8e85fd9027b000af764fdce93dfc4c3e13289e1d03f1165c337d914.jpg)

![915aba40b08f7688205b4c7d1a4631b5e37b5ad086409451ef806c3c36885281.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/915aba40b08f7688205b4c7d1a4631b5e37b5ad086409451ef806c3c36885281.jpg)

![9494f4b5df8d817c1ec2122ad8ef8cc4c444c145b2adf06f95e14b16239331e3.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/9494f4b5df8d817c1ec2122ad8ef8cc4c444c145b2adf06f95e14b16239331e3.jpg)

![94e2be672e065694e08bf47dc9517674269f3340f51a80d052c3c1809e1434c1.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/94e2be672e065694e08bf47dc9517674269f3340f51a80d052c3c1809e1434c1.jpg)

![988f98d83cc9542675e3eeb815170dc4ca4f5a7612ea68b047c1faf58d97d04f.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/988f98d83cc9542675e3eeb815170dc4ca4f5a7612ea68b047c1faf58d97d04f.jpg)

![9b109636fb6d222d98544b9ce18add9e3892f0097cffd1cf71aec6da6b10c312.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/9b109636fb6d222d98544b9ce18add9e3892f0097cffd1cf71aec6da6b10c312.jpg)

![a8b60bfdd33a8ec4afdc95514eb7ce1bc7f3bebee67446f71916ce29037e4546.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/a8b60bfdd33a8ec4afdc95514eb7ce1bc7f3bebee67446f71916ce29037e4546.jpg)

![ac0bb9f2716c42ab02f77c6dcd767eba1c8252e197b2eab0f6f54d7f56d96ae8.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/ac0bb9f2716c42ab02f77c6dcd767eba1c8252e197b2eab0f6f54d7f56d96ae8.jpg)

![af74152a25a623cf1e7663dba39347b2ea8760f1d0bf619d78c07fceeff4ee8b.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/af74152a25a623cf1e7663dba39347b2ea8760f1d0bf619d78c07fceeff4ee8b.jpg)

![b82849cf3f10815f169eb30d105bd58b9aaa10f2309c011080be525bbe5d900c.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/b82849cf3f10815f169eb30d105bd58b9aaa10f2309c011080be525bbe5d900c.jpg)

![ba76ad820279290788218094e945435a0fa75c33bc97bdf64762445dff4ebb40.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/ba76ad820279290788218094e945435a0fa75c33bc97bdf64762445dff4ebb40.jpg)

![c8db4e67f72aced98a365f1ffe09050fcacc69a7f1f0f04e30e356f0fb28a5fd.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/c8db4e67f72aced98a365f1ffe09050fcacc69a7f1f0f04e30e356f0fb28a5fd.jpg)

![edf6a7736681f28ed439b73fbb62676f3adc92acd2c71c79652d710caed97600.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/edf6a7736681f28ed439b73fbb62676f3adc92acd2c71c79652d710caed97600.jpg)

![f3213af37b234252ee73df040c84745a7e6fcaea1e6409eae9912f1ea35820bd.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/f3213af37b234252ee73df040c84745a7e6fcaea1e6409eae9912f1ea35820bd.jpg)

![f8527d82bf3ab753f58877258a19fe3dcea3ca74d03355905510035ac0f6278d.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/f8527d82bf3ab753f58877258a19fe3dcea3ca74d03355905510035ac0f6278d.jpg)

![fbaf65eb9e28108d86a124731c9895c0cd35707beeffcb22be6de266d864424b.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/fbaf65eb9e28108d86a124731c9895c0cd35707beeffcb22be6de266d864424b.jpg)

![fcdbae444f2bfc6ed50e0bd7ffe32356c34f0b6c043debb1f1931ac83e6b8ab1.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/images/fcdbae444f2bfc6ed50e0bd7ffe32356c34f0b6c043debb1f1931ac83e6b8ab1.jpg)

### Tables

![1ae3413e585c9332deb6673908aa979d663f627245fce2b7046eeda51b357cb5.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/tables/1ae3413e585c9332deb6673908aa979d663f627245fce2b7046eeda51b357cb5.jpg)

![88b941d48ed8f131d4ea6936ee806d7b17325ec0d746d1e7cadb0265927f7d59.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/tables/88b941d48ed8f131d4ea6936ee806d7b17325ec0d746d1e7cadb0265927f7d59.jpg)

![b005524ef34d6ff38d46806b4706f1105684b1266fbd34ee9ae840df8452bde1.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/tables/b005524ef34d6ff38d46806b4706f1105684b1266fbd34ee9ae840df8452bde1.jpg)

![b072278fcdb6495197d89441aa49d184ebf02f0c2225a434133f71bc5b3af62a.jpg](../iclr_results/1665_Safety-Prioritizing%20Curricula%20for%20Constrained%20Reinforcement%20Learning/tables/b072278fcdb6495197d89441aa49d184ebf02f0c2225a434133f71bc5b3af62a.jpg)

## Action Sequence Augmentation for Action Anticipation


### Images

![2bc797213da0a7f5b0f611005c162fa06a8c1a4d1545dee06aebdf16cde2f380.jpg](../iclr_results/1666_Action%20Sequence%20Augmentation%20for%20Action%20Anticipation/images/2bc797213da0a7f5b0f611005c162fa06a8c1a4d1545dee06aebdf16cde2f380.jpg)

![52e35cb1433c5adaf975a7e06c27fbc12337022d10cf5713ab035610090cea54.jpg](../iclr_results/1666_Action%20Sequence%20Augmentation%20for%20Action%20Anticipation/images/52e35cb1433c5adaf975a7e06c27fbc12337022d10cf5713ab035610090cea54.jpg)

![910160262bc6d900c32f1ac6c3bdb7ce950052c4209d65035b5fc45e6822ae3d.jpg](../iclr_results/1666_Action%20Sequence%20Augmentation%20for%20Action%20Anticipation/images/910160262bc6d900c32f1ac6c3bdb7ce950052c4209d65035b5fc45e6822ae3d.jpg)

![cd242d9c55e9e612802e74ee92f0b789ce0a24dff9a279a26cc957477f69172f.jpg](../iclr_results/1666_Action%20Sequence%20Augmentation%20for%20Action%20Anticipation/images/cd242d9c55e9e612802e74ee92f0b789ce0a24dff9a279a26cc957477f69172f.jpg)

![e64628b7f6439d56685489e825ea2c41a308b4cf4a422ee31c59425ee299a0bd.jpg](../iclr_results/1666_Action%20Sequence%20Augmentation%20for%20Action%20Anticipation/images/e64628b7f6439d56685489e825ea2c41a308b4cf4a422ee31c59425ee299a0bd.jpg)

### Tables

![1063fde13679d5680f95cee8a7e4a9b68c1b305150855c7b3515f5d5bbd80f2c.jpg](../iclr_results/1666_Action%20Sequence%20Augmentation%20for%20Action%20Anticipation/tables/1063fde13679d5680f95cee8a7e4a9b68c1b305150855c7b3515f5d5bbd80f2c.jpg)

![2c7b7752d4c72e55d0dbc4b40c7df22f69b8e37bb763c265de4867464037a95d.jpg](../iclr_results/1666_Action%20Sequence%20Augmentation%20for%20Action%20Anticipation/tables/2c7b7752d4c72e55d0dbc4b40c7df22f69b8e37bb763c265de4867464037a95d.jpg)

![52ebdd748f064dadddebe71bb599d8cdf860fab16960c480f9a2150bc907bf06.jpg](../iclr_results/1666_Action%20Sequence%20Augmentation%20for%20Action%20Anticipation/tables/52ebdd748f064dadddebe71bb599d8cdf860fab16960c480f9a2150bc907bf06.jpg)

![58c4bc3594beee8ba5f9f52a3d0f0ff6f7feb11417cc3db57694fa3111f85073.jpg](../iclr_results/1666_Action%20Sequence%20Augmentation%20for%20Action%20Anticipation/tables/58c4bc3594beee8ba5f9f52a3d0f0ff6f7feb11417cc3db57694fa3111f85073.jpg)

![7bd1eed9756877e67769901e8730a7703ac2970fa758791677a5e7ffeee1ec78.jpg](../iclr_results/1666_Action%20Sequence%20Augmentation%20for%20Action%20Anticipation/tables/7bd1eed9756877e67769901e8730a7703ac2970fa758791677a5e7ffeee1ec78.jpg)

![7db6b6cc15d8432deab3b7c30af0c8971f56b5ee550d84fa17915703bdcd425b.jpg](../iclr_results/1666_Action%20Sequence%20Augmentation%20for%20Action%20Anticipation/tables/7db6b6cc15d8432deab3b7c30af0c8971f56b5ee550d84fa17915703bdcd425b.jpg)

![99363e02f56f04bdf33be79eac86f0a723486be5db3c5bb4209cdb69d85fcf46.jpg](../iclr_results/1666_Action%20Sequence%20Augmentation%20for%20Action%20Anticipation/tables/99363e02f56f04bdf33be79eac86f0a723486be5db3c5bb4209cdb69d85fcf46.jpg)

![b172cbc7bfdb5b3b0fca53b7ace46aea4b9286ecbc6d83f2bc9677e800d87a55.jpg](../iclr_results/1666_Action%20Sequence%20Augmentation%20for%20Action%20Anticipation/tables/b172cbc7bfdb5b3b0fca53b7ace46aea4b9286ecbc6d83f2bc9677e800d87a55.jpg)

![c2c2d1844771d39088386ed32b2871939409f7009d49ae8037a8e303fd5dcb96.jpg](../iclr_results/1666_Action%20Sequence%20Augmentation%20for%20Action%20Anticipation/tables/c2c2d1844771d39088386ed32b2871939409f7009d49ae8037a8e303fd5dcb96.jpg)

![cddc5a79a0519566f9067f4ff148138dfc27feff72c4003fd125b8ade5cb386c.jpg](../iclr_results/1666_Action%20Sequence%20Augmentation%20for%20Action%20Anticipation/tables/cddc5a79a0519566f9067f4ff148138dfc27feff72c4003fd125b8ade5cb386c.jpg)

![dd2acf8b898e26d41e3d4e033666a6222e3e1bb82a19e4495c89daa02519606e.jpg](../iclr_results/1666_Action%20Sequence%20Augmentation%20for%20Action%20Anticipation/tables/dd2acf8b898e26d41e3d4e033666a6222e3e1bb82a19e4495c89daa02519606e.jpg)

![e3e1509cdb3a83520a39c54b2479e524340d829e85872a6c5a5a353cd680d5e6.jpg](../iclr_results/1666_Action%20Sequence%20Augmentation%20for%20Action%20Anticipation/tables/e3e1509cdb3a83520a39c54b2479e524340d829e85872a6c5a5a353cd680d5e6.jpg)

![fb27275e4e8615f7165841e4088e5e548253e620be7fe013a0252e19dd77243e.jpg](../iclr_results/1666_Action%20Sequence%20Augmentation%20for%20Action%20Anticipation/tables/fb27275e4e8615f7165841e4088e5e548253e620be7fe013a0252e19dd77243e.jpg)

## The Rise and Down of Babel Tower: Investigating the Evolution Process of Multilingual Code Large Language Model


### Images

![18d1d6a37e54fed99de9763b5ac043d193a5bc64e1be3aecdb2d759f50f0d0f6.jpg](../iclr_results/1667_The%20Rise%20and%20Down%20of%20Babel%20Tower_%20Investigating%20the%20Evolution%20Process%20of%20Multilingual%20Code%20Large%20Lan/images/18d1d6a37e54fed99de9763b5ac043d193a5bc64e1be3aecdb2d759f50f0d0f6.jpg)

![6c28886998ce9d9f1d836f31dee578ef41895ced2fbe4ecc2cfc07789767f82b.jpg](../iclr_results/1667_The%20Rise%20and%20Down%20of%20Babel%20Tower_%20Investigating%20the%20Evolution%20Process%20of%20Multilingual%20Code%20Large%20Lan/images/6c28886998ce9d9f1d836f31dee578ef41895ced2fbe4ecc2cfc07789767f82b.jpg)

![7ab41ae4ddfb3696e35a047a51d9a15e1be5200786c0f1a354dd2770617170dc.jpg](../iclr_results/1667_The%20Rise%20and%20Down%20of%20Babel%20Tower_%20Investigating%20the%20Evolution%20Process%20of%20Multilingual%20Code%20Large%20Lan/images/7ab41ae4ddfb3696e35a047a51d9a15e1be5200786c0f1a354dd2770617170dc.jpg)

![7f46ea2c16f80e3cee215ac7093bd313f1af21efd993c1f83fe5b727d326f647.jpg](../iclr_results/1667_The%20Rise%20and%20Down%20of%20Babel%20Tower_%20Investigating%20the%20Evolution%20Process%20of%20Multilingual%20Code%20Large%20Lan/images/7f46ea2c16f80e3cee215ac7093bd313f1af21efd993c1f83fe5b727d326f647.jpg)

![9eac18f182eed8bd3000412ad1a7e4183f7e26b72ddcd6e37ecccba3d1e32143.jpg](../iclr_results/1667_The%20Rise%20and%20Down%20of%20Babel%20Tower_%20Investigating%20the%20Evolution%20Process%20of%20Multilingual%20Code%20Large%20Lan/images/9eac18f182eed8bd3000412ad1a7e4183f7e26b72ddcd6e37ecccba3d1e32143.jpg)

![aad125ff5a37720afac199f27b9dfd6dde3f1608c74695c3ac8cdd04654dc2df.jpg](../iclr_results/1667_The%20Rise%20and%20Down%20of%20Babel%20Tower_%20Investigating%20the%20Evolution%20Process%20of%20Multilingual%20Code%20Large%20Lan/images/aad125ff5a37720afac199f27b9dfd6dde3f1608c74695c3ac8cdd04654dc2df.jpg)

![b7ece5472df65f9c627a8b7c1b5b469dfcdbb109c36489ff89d2bf2af80134c7.jpg](../iclr_results/1667_The%20Rise%20and%20Down%20of%20Babel%20Tower_%20Investigating%20the%20Evolution%20Process%20of%20Multilingual%20Code%20Large%20Lan/images/b7ece5472df65f9c627a8b7c1b5b469dfcdbb109c36489ff89d2bf2af80134c7.jpg)

![e94c0403a1d8f65a1d993bc2ea9d2407c4c5e12547f1218d14e8840f78ce50b6.jpg](../iclr_results/1667_The%20Rise%20and%20Down%20of%20Babel%20Tower_%20Investigating%20the%20Evolution%20Process%20of%20Multilingual%20Code%20Large%20Lan/images/e94c0403a1d8f65a1d993bc2ea9d2407c4c5e12547f1218d14e8840f78ce50b6.jpg)

![ee700303cf329b665f6e3c21616704493168af697dacf13f4175a1ead32ca91c.jpg](../iclr_results/1667_The%20Rise%20and%20Down%20of%20Babel%20Tower_%20Investigating%20the%20Evolution%20Process%20of%20Multilingual%20Code%20Large%20Lan/images/ee700303cf329b665f6e3c21616704493168af697dacf13f4175a1ead32ca91c.jpg)

### Tables

![2deb89b2292debaa5985643fc8ad2a5040fa1205f34b623ce8abe6485fcefa6d.jpg](../iclr_results/1667_The%20Rise%20and%20Down%20of%20Babel%20Tower_%20Investigating%20the%20Evolution%20Process%20of%20Multilingual%20Code%20Large%20Lan/tables/2deb89b2292debaa5985643fc8ad2a5040fa1205f34b623ce8abe6485fcefa6d.jpg)

![2e6ff6442182f19f0b4cb04b04231de65ec80bcac7c881def65f246280fc8097.jpg](../iclr_results/1667_The%20Rise%20and%20Down%20of%20Babel%20Tower_%20Investigating%20the%20Evolution%20Process%20of%20Multilingual%20Code%20Large%20Lan/tables/2e6ff6442182f19f0b4cb04b04231de65ec80bcac7c881def65f246280fc8097.jpg)

![b0b9f57e758841197cdfff47b73893f8e8c0445480bc6975ef5f456c06749bbf.jpg](../iclr_results/1667_The%20Rise%20and%20Down%20of%20Babel%20Tower_%20Investigating%20the%20Evolution%20Process%20of%20Multilingual%20Code%20Large%20Lan/tables/b0b9f57e758841197cdfff47b73893f8e8c0445480bc6975ef5f456c06749bbf.jpg)

![fc1fbd88315df3e889b7e8841f606ef5cf52d076c92f00ac3200c6a469749d66.jpg](../iclr_results/1667_The%20Rise%20and%20Down%20of%20Babel%20Tower_%20Investigating%20the%20Evolution%20Process%20of%20Multilingual%20Code%20Large%20Lan/tables/fc1fbd88315df3e889b7e8841f606ef5cf52d076c92f00ac3200c6a469749d66.jpg)

## Parameter Expanded Stochastic Gradient Markov Chain Monte Carlo


### Images

![18fb4fd728498608c17d814f5b931cb68bcae05866b07201e1f50730f3a8699a.jpg](../iclr_results/1668_Parameter%20Expanded%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo/images/18fb4fd728498608c17d814f5b931cb68bcae05866b07201e1f50730f3a8699a.jpg)

![22d170b09c398ba913f84bb1b9087220b2dcb1248508d18dc81b0ab306f8951c.jpg](../iclr_results/1668_Parameter%20Expanded%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo/images/22d170b09c398ba913f84bb1b9087220b2dcb1248508d18dc81b0ab306f8951c.jpg)

![31c1fc2eee6ce49f069e5effd664c1cdee77e7bf89dd2e7abaf5fc931326af97.jpg](../iclr_results/1668_Parameter%20Expanded%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo/images/31c1fc2eee6ce49f069e5effd664c1cdee77e7bf89dd2e7abaf5fc931326af97.jpg)

![5af6b5504e35a49317eed86fc7e6e7d9f1e04de550abb665ecfcaf7063e5133b.jpg](../iclr_results/1668_Parameter%20Expanded%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo/images/5af6b5504e35a49317eed86fc7e6e7d9f1e04de550abb665ecfcaf7063e5133b.jpg)

![81ab85187495fc11e3c846ebba4f707f488d16002f8cd59f313616c93c48926e.jpg](../iclr_results/1668_Parameter%20Expanded%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo/images/81ab85187495fc11e3c846ebba4f707f488d16002f8cd59f313616c93c48926e.jpg)

![834eef597884473b67a060991b9c66a08cf123d407dfa05a40b9a0625f2f767c.jpg](../iclr_results/1668_Parameter%20Expanded%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo/images/834eef597884473b67a060991b9c66a08cf123d407dfa05a40b9a0625f2f767c.jpg)

![9452e701b7a545553c5a5cf21c2adb08887cabff4c44906200d3a054c91b3c80.jpg](../iclr_results/1668_Parameter%20Expanded%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo/images/9452e701b7a545553c5a5cf21c2adb08887cabff4c44906200d3a054c91b3c80.jpg)

![951ef3ca1f667f056f8620c093427bca44395c1bcc071bb9c9a0b42ad687a819.jpg](../iclr_results/1668_Parameter%20Expanded%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo/images/951ef3ca1f667f056f8620c093427bca44395c1bcc071bb9c9a0b42ad687a819.jpg)

![a2c60143948015950db90ed6a950c697d4a3159bf70e70b5833d4f422499db5a.jpg](../iclr_results/1668_Parameter%20Expanded%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo/images/a2c60143948015950db90ed6a950c697d4a3159bf70e70b5833d4f422499db5a.jpg)

![d62f92f5f55652477ace0b13fba44d7e19f9d9dea11aa10b78f34ed3d899f1e1.jpg](../iclr_results/1668_Parameter%20Expanded%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo/images/d62f92f5f55652477ace0b13fba44d7e19f9d9dea11aa10b78f34ed3d899f1e1.jpg)

![e2d1d51e3eae49063d97ce356b09c9a629ba54731d6c9768f401e2ae7d9b50ef.jpg](../iclr_results/1668_Parameter%20Expanded%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo/images/e2d1d51e3eae49063d97ce356b09c9a629ba54731d6c9768f401e2ae7d9b50ef.jpg)

### Tables

![05026daaa8c5c74c74d48aea984b67941009d4b6d697884148b31473d6b39488.jpg](../iclr_results/1668_Parameter%20Expanded%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo/tables/05026daaa8c5c74c74d48aea984b67941009d4b6d697884148b31473d6b39488.jpg)

![1944dafa73dbeda005ea2b34b3af2e38430168f7a9a552eea944c83c7d73f245.jpg](../iclr_results/1668_Parameter%20Expanded%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo/tables/1944dafa73dbeda005ea2b34b3af2e38430168f7a9a552eea944c83c7d73f245.jpg)

![8afcae915a0fb7ea2abe64da679fc7456293061c77c03e7bec971bed55e02a4d.jpg](../iclr_results/1668_Parameter%20Expanded%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo/tables/8afcae915a0fb7ea2abe64da679fc7456293061c77c03e7bec971bed55e02a4d.jpg)

![8fcc813acaae5c512bcf848b34e4c789534301ae2ca47d9a74ada868faf14cd8.jpg](../iclr_results/1668_Parameter%20Expanded%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo/tables/8fcc813acaae5c512bcf848b34e4c789534301ae2ca47d9a74ada868faf14cd8.jpg)

![a0cfa05269c50f4e6ecaff119dc8ac3449a43c42d51a468661fcba9b6e03e0a2.jpg](../iclr_results/1668_Parameter%20Expanded%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo/tables/a0cfa05269c50f4e6ecaff119dc8ac3449a43c42d51a468661fcba9b6e03e0a2.jpg)

![a624d7700046092aa8a98e4b93527f93bd9403f9493cc35f0c6f93c448fc20b9.jpg](../iclr_results/1668_Parameter%20Expanded%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo/tables/a624d7700046092aa8a98e4b93527f93bd9403f9493cc35f0c6f93c448fc20b9.jpg)

![a9982a5c483d3975f47424550923213ed39631f94900feaaf37f7d3c43d52d91.jpg](../iclr_results/1668_Parameter%20Expanded%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo/tables/a9982a5c483d3975f47424550923213ed39631f94900feaaf37f7d3c43d52d91.jpg)

![ba5cdb6861d61195b46d61909b6bbeb8ddaa7c16a08d10887b0813a566889250.jpg](../iclr_results/1668_Parameter%20Expanded%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo/tables/ba5cdb6861d61195b46d61909b6bbeb8ddaa7c16a08d10887b0813a566889250.jpg)

![c15ac4a3731563d1d2f4d540fd06d807342519b2d3ff4b9a02f23112d9906e21.jpg](../iclr_results/1668_Parameter%20Expanded%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo/tables/c15ac4a3731563d1d2f4d540fd06d807342519b2d3ff4b9a02f23112d9906e21.jpg)

![c33d93bf6a076cd3b6afd1a02a537a081cbd6936ff1ab3aafcf18b7682605317.jpg](../iclr_results/1668_Parameter%20Expanded%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo/tables/c33d93bf6a076cd3b6afd1a02a537a081cbd6936ff1ab3aafcf18b7682605317.jpg)

![cda22a36bf7d53486e65742308a7abe01c2338fbf9b4c6fb7201a0d529195882.jpg](../iclr_results/1668_Parameter%20Expanded%20Stochastic%20Gradient%20Markov%20Chain%20Monte%20Carlo/tables/cda22a36bf7d53486e65742308a7abe01c2338fbf9b4c6fb7201a0d529195882.jpg)

## What Secrets Do Your Manifolds Hold? Understanding the Local Geometry of Generative Models


### Images

![096b7f4813096a827ade9bcc14b4f08287b0840abd2dd5302220cb462049de2c.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/096b7f4813096a827ade9bcc14b4f08287b0840abd2dd5302220cb462049de2c.jpg)

![0ba6d98662a08fe6f1df3c3450d73aa4c0408281166fe2bc42039c3040abc9fb.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/0ba6d98662a08fe6f1df3c3450d73aa4c0408281166fe2bc42039c3040abc9fb.jpg)

![0e8834339f7a657b222509f924a59e1d7d05257676fae8c65765242a6a606b71.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/0e8834339f7a657b222509f924a59e1d7d05257676fae8c65765242a6a606b71.jpg)

![2031ebbe9ca61a51ace797bab3305f4e45db92f07500ab4d84c70b2f139293b9.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/2031ebbe9ca61a51ace797bab3305f4e45db92f07500ab4d84c70b2f139293b9.jpg)

![2111a6d1639b91761ea1edf05b75210156ddc61c1ae8fba62be0dbb71e847416.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/2111a6d1639b91761ea1edf05b75210156ddc61c1ae8fba62be0dbb71e847416.jpg)

![38cff946a1752c337b155c8b61f5c765df9b43674cf5c835ee000afd5313133b.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/38cff946a1752c337b155c8b61f5c765df9b43674cf5c835ee000afd5313133b.jpg)

![3f8c912a6960fc5bede8dc6f0f7d5df03d5aa7b3f6226e856d6c8da55fadba92.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/3f8c912a6960fc5bede8dc6f0f7d5df03d5aa7b3f6226e856d6c8da55fadba92.jpg)

![47cf955f4d11a99a0a29d6eb4f1c143da43c7b496bac1144c42a2540c263b67f.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/47cf955f4d11a99a0a29d6eb4f1c143da43c7b496bac1144c42a2540c263b67f.jpg)

![4c4523875f652bb54324c325562aba5dd489db4dbb16f10712a3348afda7f989.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/4c4523875f652bb54324c325562aba5dd489db4dbb16f10712a3348afda7f989.jpg)

![54694a28aad7c9a8fd1d8405c46577f3ba82fe48599cabdbb4dbb5be289543ec.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/54694a28aad7c9a8fd1d8405c46577f3ba82fe48599cabdbb4dbb5be289543ec.jpg)

![5bd5047a15114aa4c1d303853c04222676d3970a2850cba777d4a58d36416ed8.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/5bd5047a15114aa4c1d303853c04222676d3970a2850cba777d4a58d36416ed8.jpg)

![72116026492226290dfb8fbafbb7a4ed2ba76d77014159c0141612b13ccf9d9b.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/72116026492226290dfb8fbafbb7a4ed2ba76d77014159c0141612b13ccf9d9b.jpg)

![7a55b957e4e357891f470642c95e1ecfdd94d6eecf61f7d69bc22d03c754d774.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/7a55b957e4e357891f470642c95e1ecfdd94d6eecf61f7d69bc22d03c754d774.jpg)

![7df7971205f20f9309eab378d3f8167cd94c0af4358ad45ab215c9023fb451b6.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/7df7971205f20f9309eab378d3f8167cd94c0af4358ad45ab215c9023fb451b6.jpg)

![7e9408d5e5e0f870ad0e8f8eace25ee7e6692ed99253c9e5d1ba5b64f0fb23f3.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/7e9408d5e5e0f870ad0e8f8eace25ee7e6692ed99253c9e5d1ba5b64f0fb23f3.jpg)

![89afa826c34e7c451b6dd9167c1f1b48016c4b83e11085b8be07ae2f3a4cca7e.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/89afa826c34e7c451b6dd9167c1f1b48016c4b83e11085b8be07ae2f3a4cca7e.jpg)

![997a04dca5944cb136dd3f5e75a3e694c53c3f0b41a03e0c0a9bf6813b349767.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/997a04dca5944cb136dd3f5e75a3e694c53c3f0b41a03e0c0a9bf6813b349767.jpg)

![a292c3d80d286a5a3f10d17cacb843fc4aca2945b39528a2037cec7d5e16e744.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/a292c3d80d286a5a3f10d17cacb843fc4aca2945b39528a2037cec7d5e16e744.jpg)

![a83a28bc4efde0af1e0c51c3b0b42bfe5385ef058a9741da7be6f4910897a0a4.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/a83a28bc4efde0af1e0c51c3b0b42bfe5385ef058a9741da7be6f4910897a0a4.jpg)

![aaaa2ff0cffb4800ef64cbb3be4df3f8b9ba86c0dbf742e06264c42025545aa4.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/aaaa2ff0cffb4800ef64cbb3be4df3f8b9ba86c0dbf742e06264c42025545aa4.jpg)

![ac879302a665ae90e77bf45fca464fd3c865c22674cdd6aefea4e3a71f6f4ca1.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/ac879302a665ae90e77bf45fca464fd3c865c22674cdd6aefea4e3a71f6f4ca1.jpg)

![ae0294f1d70fab668f36c3cd27a987cc899097fcb6bee6e1b7af07820af832b1.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/ae0294f1d70fab668f36c3cd27a987cc899097fcb6bee6e1b7af07820af832b1.jpg)

![b63a56a98d69b8f470a24f0d5dc5cb62e66ad084852a08bb85232066f9ec8926.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/b63a56a98d69b8f470a24f0d5dc5cb62e66ad084852a08bb85232066f9ec8926.jpg)

![b8c54b05036b3f270e109022ba272b5ca751dbf135b7051031af626daa122112.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/b8c54b05036b3f270e109022ba272b5ca751dbf135b7051031af626daa122112.jpg)

![b953344da65723d606b5132aac94776689eaff57d7927dbd74f89c9a630185c4.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/b953344da65723d606b5132aac94776689eaff57d7927dbd74f89c9a630185c4.jpg)

![c7938b90dea2b3885d31c07a7d42206065d8a973b8c90172b8d047ef7b3b5bbe.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/c7938b90dea2b3885d31c07a7d42206065d8a973b8c90172b8d047ef7b3b5bbe.jpg)

![ca7bd6b432911bf3d2724bbc00b3bd4c43690d930bcb9d81bcf99a5c6e8e6f58.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/ca7bd6b432911bf3d2724bbc00b3bd4c43690d930bcb9d81bcf99a5c6e8e6f58.jpg)

![d0f02cacd2f6814c2611affa613e77d4ed5f9e5c775ffb4bf6febeff2ce63c51.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/d0f02cacd2f6814c2611affa613e77d4ed5f9e5c775ffb4bf6febeff2ce63c51.jpg)

![e23d9a2a5f49593ed6312e7fa31c58361aadf72f8c57f7ec31ab9073115b4c50.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/e23d9a2a5f49593ed6312e7fa31c58361aadf72f8c57f7ec31ab9073115b4c50.jpg)

![e39038f7979eb09fc887e4508bd84d6b8ec7e248ef4aafcb568fd88bb3bd5f97.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/e39038f7979eb09fc887e4508bd84d6b8ec7e248ef4aafcb568fd88bb3bd5f97.jpg)

![f397ed5fb54aaec8419d7946838c66f61d256eea935d5ea3a83bd317cc650e9b.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/f397ed5fb54aaec8419d7946838c66f61d256eea935d5ea3a83bd317cc650e9b.jpg)

![f74b0dadbd8bbe752610ad3db5e58c7330005ec16034f45c0444bd4f2db64db0.jpg](../iclr_results/1669_What%20Secrets%20Do%20Your%20Manifolds%20Hold_%20Understanding%20the%20Local%20Geometry%20of%20Generative%20Models/images/f74b0dadbd8bbe752610ad3db5e58c7330005ec16034f45c0444bd4f2db64db0.jpg)

## TD-Paint: Faster Diffusion Inpainting Through Time-Aware Pixel Conditioning


### Images

![00b1be7526a3cf9e9898a36e67218982c80299d0429132565184491f28f20821.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/00b1be7526a3cf9e9898a36e67218982c80299d0429132565184491f28f20821.jpg)

![013adfa12b8294e869dae426847345a42b7ae020267ac986209b03f95461a46b.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/013adfa12b8294e869dae426847345a42b7ae020267ac986209b03f95461a46b.jpg)

![0e6ecdb4370e672ddec93559105e1478e1d5b4b1016f64dd6d2d75a47e6c7c83.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/0e6ecdb4370e672ddec93559105e1478e1d5b4b1016f64dd6d2d75a47e6c7c83.jpg)

![145bae91ea6a88a2e1434868a3e2a6035658c49f33b8a271bcd6751b9117b2e3.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/145bae91ea6a88a2e1434868a3e2a6035658c49f33b8a271bcd6751b9117b2e3.jpg)

![14fd3676fc33dbf655edd7b234f3908d0b652779444bf8db28b38757d85b9745.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/14fd3676fc33dbf655edd7b234f3908d0b652779444bf8db28b38757d85b9745.jpg)

![31d190eaecd21720e3902a940b48b69f4ccb32f907960bc7d3fbe8b413b8ae45.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/31d190eaecd21720e3902a940b48b69f4ccb32f907960bc7d3fbe8b413b8ae45.jpg)

![36a3153af612fcaec6913b7784b259d6909a8b4ca99fe27c4610ef9048f2d230.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/36a3153af612fcaec6913b7784b259d6909a8b4ca99fe27c4610ef9048f2d230.jpg)

![3d711b779ed721e3fc89d2972f40cab445876084c936541cfa937f03ed2210ee.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/3d711b779ed721e3fc89d2972f40cab445876084c936541cfa937f03ed2210ee.jpg)

![46754ff85296d0293e7f90fc2b6e1fb601e647a4d8202c5f6391720004a36ae8.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/46754ff85296d0293e7f90fc2b6e1fb601e647a4d8202c5f6391720004a36ae8.jpg)

![498fcfb89a2f455015f3b5f67d9dddec086cbf614868177e0a154ae3f507d3b6.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/498fcfb89a2f455015f3b5f67d9dddec086cbf614868177e0a154ae3f507d3b6.jpg)

![603f9b11c1d2b309ed84c21cdb4700f17f95b7547deed87bad1dbc852273bd5b.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/603f9b11c1d2b309ed84c21cdb4700f17f95b7547deed87bad1dbc852273bd5b.jpg)

![6152a03909b1020dfe5a070931a5aa3a1fbfe07aa9ce5371919236b3c85808f3.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/6152a03909b1020dfe5a070931a5aa3a1fbfe07aa9ce5371919236b3c85808f3.jpg)

![74de030de0a91240143c9e6284167ecc5e4d45de90c8c7ea6e5bc63fb1dd9adb.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/74de030de0a91240143c9e6284167ecc5e4d45de90c8c7ea6e5bc63fb1dd9adb.jpg)

![7ce7b32c5362f96dda135e258ce7f8a49519de3273dd832a99808580e8599fa9.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/7ce7b32c5362f96dda135e258ce7f8a49519de3273dd832a99808580e8599fa9.jpg)

![7f618e129449efb9c04475f10c8c7e19d337faae27c1dc2402f3d8122a2ecb89.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/7f618e129449efb9c04475f10c8c7e19d337faae27c1dc2402f3d8122a2ecb89.jpg)

![941c94555095478e1026a7407e64954bf3c89b0c7894a3781d2156223d19ee1e.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/941c94555095478e1026a7407e64954bf3c89b0c7894a3781d2156223d19ee1e.jpg)

![a0aca647abe6caee70c5d9288a3cbc4dd0bcff906b3295fea407cf971bf9834b.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/a0aca647abe6caee70c5d9288a3cbc4dd0bcff906b3295fea407cf971bf9834b.jpg)

![ac5db672497f43f9a5c08e20a08efe23d4e83097719a85e684f56c09b638df09.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/ac5db672497f43f9a5c08e20a08efe23d4e83097719a85e684f56c09b638df09.jpg)

![b059b2ea83eb8b8070d3f130a85573c46fd9ab74b71466d631d91e3701ab66cc.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/b059b2ea83eb8b8070d3f130a85573c46fd9ab74b71466d631d91e3701ab66cc.jpg)

![b2e5c7e46a58efc2f397c17aee3fb0832812c5f528c6f91595c9b6f0e4c9ea36.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/b2e5c7e46a58efc2f397c17aee3fb0832812c5f528c6f91595c9b6f0e4c9ea36.jpg)

![be9ce6dbe9ee907838d129173aa7f2dbb228528787befc7a65c39af615230d23.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/be9ce6dbe9ee907838d129173aa7f2dbb228528787befc7a65c39af615230d23.jpg)

![cfc817165434b707ce60c3d7d11a809f4ee75cdc82d7b395a2f7b99af72a3db5.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/cfc817165434b707ce60c3d7d11a809f4ee75cdc82d7b395a2f7b99af72a3db5.jpg)

![ddba1df10274ab6de7f8a0b55693012b054ee2b1775b8fb2a55e4561d1326420.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/ddba1df10274ab6de7f8a0b55693012b054ee2b1775b8fb2a55e4561d1326420.jpg)

![e192915158f2bef89bbc7905d1091b03209e282ece2c0236086e5afd255e9439.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/e192915158f2bef89bbc7905d1091b03209e282ece2c0236086e5afd255e9439.jpg)

![ed67bd856c6b99d9e3ecb279bb23f58007250484d72f1a993482669e591d8680.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/images/ed67bd856c6b99d9e3ecb279bb23f58007250484d72f1a993482669e591d8680.jpg)

### Tables

![485e3ad945c363e017b128d145b153f29d581b77f647ebaa9985cf2053495136.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/tables/485e3ad945c363e017b128d145b153f29d581b77f647ebaa9985cf2053495136.jpg)

![5f99a1b5fde85b8169257b18bff5b4950925412a8b5c178dfb569449cdf6f802.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/tables/5f99a1b5fde85b8169257b18bff5b4950925412a8b5c178dfb569449cdf6f802.jpg)

![6530acca5c6b8646c99ac8e373f77d7d4946564905891211291bf9780543c4b6.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/tables/6530acca5c6b8646c99ac8e373f77d7d4946564905891211291bf9780543c4b6.jpg)

![854d924c9a20a97118769242a592156fcac1b79ce64731a4d6dcd34a5c08ce98.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/tables/854d924c9a20a97118769242a592156fcac1b79ce64731a4d6dcd34a5c08ce98.jpg)

![9e37fabcfd720e24140220340a2d07f3563cf771d236ead7d0499855619a09f2.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/tables/9e37fabcfd720e24140220340a2d07f3563cf771d236ead7d0499855619a09f2.jpg)

![cd69d28b6bdc17490396d4a9735d967673dfea9e36877f311482a7424bdd3516.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/tables/cd69d28b6bdc17490396d4a9735d967673dfea9e36877f311482a7424bdd3516.jpg)

![e8f8e99fe1277ddd21d46a2715ece2059ba1b3bdc5b6ab4cb876aa8271fe430a.jpg](../iclr_results/1670_TD-Paint_%20Faster%20Diffusion%20Inpainting%20Through%20Time-Aware%20Pixel%20Conditioning/tables/e8f8e99fe1277ddd21d46a2715ece2059ba1b3bdc5b6ab4cb876aa8271fe430a.jpg)

## Why Does the Effective Context Length of LLMs Fall Short?


### Images

![13b0ad7b615d150394a03c5a397a2d69290afe4b2733c4632d56ff689230db86.jpg](../iclr_results/1671_Why%20Does%20the%20Effective%20Context%20Length%20of%20LLMs%20Fall%20Short_/images/13b0ad7b615d150394a03c5a397a2d69290afe4b2733c4632d56ff689230db86.jpg)

![16e827488c79b5afcf1a12f63e63e37dfa00fffde6d4814b59811cdee1f1bd4c.jpg](../iclr_results/1671_Why%20Does%20the%20Effective%20Context%20Length%20of%20LLMs%20Fall%20Short_/images/16e827488c79b5afcf1a12f63e63e37dfa00fffde6d4814b59811cdee1f1bd4c.jpg)

![56cdda4e179cd45656bef04a151ff4339bb7513ed387d8d8649b964473a63504.jpg](../iclr_results/1671_Why%20Does%20the%20Effective%20Context%20Length%20of%20LLMs%20Fall%20Short_/images/56cdda4e179cd45656bef04a151ff4339bb7513ed387d8d8649b964473a63504.jpg)

![6934d537d10b2e662571d365346e376a648fd936a80aa33b9fca63f64df5934e.jpg](../iclr_results/1671_Why%20Does%20the%20Effective%20Context%20Length%20of%20LLMs%20Fall%20Short_/images/6934d537d10b2e662571d365346e376a648fd936a80aa33b9fca63f64df5934e.jpg)

![876c2bd0c9dbe386e9732ece8e9012fdf8b26a7c05dd5f6c611a76ebf5fa1008.jpg](../iclr_results/1671_Why%20Does%20the%20Effective%20Context%20Length%20of%20LLMs%20Fall%20Short_/images/876c2bd0c9dbe386e9732ece8e9012fdf8b26a7c05dd5f6c611a76ebf5fa1008.jpg)

![8b75f20f10844ca5669a8a5ec414f8bae38a64c498a94ecaa9d74f6b961ac3c1.jpg](../iclr_results/1671_Why%20Does%20the%20Effective%20Context%20Length%20of%20LLMs%20Fall%20Short_/images/8b75f20f10844ca5669a8a5ec414f8bae38a64c498a94ecaa9d74f6b961ac3c1.jpg)

![a5b3c0cf685d18fdeaf9454c98fb323bb3177f2da6c16c3270fc4f7a1e43c7ac.jpg](../iclr_results/1671_Why%20Does%20the%20Effective%20Context%20Length%20of%20LLMs%20Fall%20Short_/images/a5b3c0cf685d18fdeaf9454c98fb323bb3177f2da6c16c3270fc4f7a1e43c7ac.jpg)

![a69ee8b8cd81960bbff3a144b7861ca95a7ec454103c6bda4dad68c187bed8c4.jpg](../iclr_results/1671_Why%20Does%20the%20Effective%20Context%20Length%20of%20LLMs%20Fall%20Short_/images/a69ee8b8cd81960bbff3a144b7861ca95a7ec454103c6bda4dad68c187bed8c4.jpg)

![bf5233b4222f45df6c0a61af6133ba9ecd774f37284af3174b7072681faf0724.jpg](../iclr_results/1671_Why%20Does%20the%20Effective%20Context%20Length%20of%20LLMs%20Fall%20Short_/images/bf5233b4222f45df6c0a61af6133ba9ecd774f37284af3174b7072681faf0724.jpg)

### Tables

![3e1958f8998f8ddff33c498e21ee315f22761fe72a18a4ca48629f8d86ef68ac.jpg](../iclr_results/1671_Why%20Does%20the%20Effective%20Context%20Length%20of%20LLMs%20Fall%20Short_/tables/3e1958f8998f8ddff33c498e21ee315f22761fe72a18a4ca48629f8d86ef68ac.jpg)

![80753decb13d07b3cae0802eb3f4f506dd736ede4cd228eb7f53bc0d685571bd.jpg](../iclr_results/1671_Why%20Does%20the%20Effective%20Context%20Length%20of%20LLMs%20Fall%20Short_/tables/80753decb13d07b3cae0802eb3f4f506dd736ede4cd228eb7f53bc0d685571bd.jpg)

![874ccc7c0f0a24fbd22287d0194ac6e718057e530448cff7ca02f3ed8de8f292.jpg](../iclr_results/1671_Why%20Does%20the%20Effective%20Context%20Length%20of%20LLMs%20Fall%20Short_/tables/874ccc7c0f0a24fbd22287d0194ac6e718057e530448cff7ca02f3ed8de8f292.jpg)

![964f12d5eb968263cb757544f4e5e13d7df45210681abcbfc453530a025f0043.jpg](../iclr_results/1671_Why%20Does%20the%20Effective%20Context%20Length%20of%20LLMs%20Fall%20Short_/tables/964f12d5eb968263cb757544f4e5e13d7df45210681abcbfc453530a025f0043.jpg)

![d4df8f2af12786eecdbbb07325adb2a2d8d4b3cdcd78c909f57649e201a9fc19.jpg](../iclr_results/1671_Why%20Does%20the%20Effective%20Context%20Length%20of%20LLMs%20Fall%20Short_/tables/d4df8f2af12786eecdbbb07325adb2a2d8d4b3cdcd78c909f57649e201a9fc19.jpg)

## Vision CNNs trained to estimate spatial latents learned similar ventral-stream-aligned representations


### Images

![177bae2f7f36d2531fcea32a617b6c2a87ff4370e677663ac3b2d15ecfd756e0.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/images/177bae2f7f36d2531fcea32a617b6c2a87ff4370e677663ac3b2d15ecfd756e0.jpg)

![1c7498a74ebc5e39da0f025d6af7bb2a1e28ef14f491dde4344f2d63048f9729.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/images/1c7498a74ebc5e39da0f025d6af7bb2a1e28ef14f491dde4344f2d63048f9729.jpg)

![2290c3e897255ff1a82a4ff4813867c624ff8bac2755f3a289b44be31fca03b8.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/images/2290c3e897255ff1a82a4ff4813867c624ff8bac2755f3a289b44be31fca03b8.jpg)

![40e5316c7d4c33f5707def719443492334756735e5c3282dc43dd050ff9f2161.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/images/40e5316c7d4c33f5707def719443492334756735e5c3282dc43dd050ff9f2161.jpg)

![4b0188f8feac025e00b621072c4284953add2eb31f76056135485ce17b3f4dc6.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/images/4b0188f8feac025e00b621072c4284953add2eb31f76056135485ce17b3f4dc6.jpg)

![4d33c70648273f2c9abde3717accdfc5f466132650af33916e022b28999d05fb.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/images/4d33c70648273f2c9abde3717accdfc5f466132650af33916e022b28999d05fb.jpg)

![58f75dd5dc514eed028525af712d3c0ccee7dceeea75c3e7603cf5549ac8b06a.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/images/58f75dd5dc514eed028525af712d3c0ccee7dceeea75c3e7603cf5549ac8b06a.jpg)

![5a2c37ac472b75f583b93c077f2f991af690fc2e6794d3e8cf907e6190f4193a.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/images/5a2c37ac472b75f583b93c077f2f991af690fc2e6794d3e8cf907e6190f4193a.jpg)

![5ccc17c93a4b38668898c3c87d4461a697dc32c5132250acede903e0cc87b9d3.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/images/5ccc17c93a4b38668898c3c87d4461a697dc32c5132250acede903e0cc87b9d3.jpg)

![67f219779e558f4d22edcd29e8ceabedccc2871ef9b7953aba3417bde822a3ec.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/images/67f219779e558f4d22edcd29e8ceabedccc2871ef9b7953aba3417bde822a3ec.jpg)

![684143d6b3d1cbca4518746fb00a04767c6f440c7d7c7de12e275315af4d1130.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/images/684143d6b3d1cbca4518746fb00a04767c6f440c7d7c7de12e275315af4d1130.jpg)

![838adba6e490405992b156f960f50fb5f08838c86c31a570e78b358658f20392.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/images/838adba6e490405992b156f960f50fb5f08838c86c31a570e78b358658f20392.jpg)

![91d713a78303ef02f81b1b3fd1d467bdfc8ae62aad29f0b349d85eded98e1e3e.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/images/91d713a78303ef02f81b1b3fd1d467bdfc8ae62aad29f0b349d85eded98e1e3e.jpg)

![a5ec21e7072d55f2d15e8d7505252e5544753aae2791400a37c55dc592ed8bbf.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/images/a5ec21e7072d55f2d15e8d7505252e5544753aae2791400a37c55dc592ed8bbf.jpg)

![bb6f4ca39f95f6a8a3b09d38c6d213d4dbc4b167e7b9f144b566a530897443f2.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/images/bb6f4ca39f95f6a8a3b09d38c6d213d4dbc4b167e7b9f144b566a530897443f2.jpg)

![c33b1c432d5b48f3ea3ad630978512c01a41195f1bde61a4fb5e007750c01784.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/images/c33b1c432d5b48f3ea3ad630978512c01a41195f1bde61a4fb5e007750c01784.jpg)

![d8c267949a1e3c252ca05e310b9b7527c86c34b38404506c563430f80f8131e2.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/images/d8c267949a1e3c252ca05e310b9b7527c86c34b38404506c563430f80f8131e2.jpg)

![e1f2b4a30fb5b35fd5435d936c50080ef3842185a93e390576698f6b34fa6a19.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/images/e1f2b4a30fb5b35fd5435d936c50080ef3842185a93e390576698f6b34fa6a19.jpg)

![e48f2bf4902193f6c8de24c916d1b96870039e16688e9a3afd8389f8c6a955d3.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/images/e48f2bf4902193f6c8de24c916d1b96870039e16688e9a3afd8389f8c6a955d3.jpg)

![e522a9d113fa3e49bb8b735f0f00181aa7f02fa225dc0fde9ef831b2d3acf5d7.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/images/e522a9d113fa3e49bb8b735f0f00181aa7f02fa225dc0fde9ef831b2d3acf5d7.jpg)

![e745707c5f36adf4efc1dd77fe7a8cc590a8f253cc9ef49a6bc5acf498aac30b.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/images/e745707c5f36adf4efc1dd77fe7a8cc590a8f253cc9ef49a6bc5acf498aac30b.jpg)

### Tables

![0385c0d29410b38d48cc6b71e34810f3d7253bbfad66aa2f1601fdcc8813ccdf.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/tables/0385c0d29410b38d48cc6b71e34810f3d7253bbfad66aa2f1601fdcc8813ccdf.jpg)

![823c137a9c61d1443aff7bc5365d465719939cbd55fcd51eec6742461d52f97b.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/tables/823c137a9c61d1443aff7bc5365d465719939cbd55fcd51eec6742461d52f97b.jpg)

![d87329e050f549e8d137f6a0d6c1c012b628d54b13e54b42d14fa52dbe4afaaf.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/tables/d87329e050f549e8d137f6a0d6c1c012b628d54b13e54b42d14fa52dbe4afaaf.jpg)

![da6a737fbca7aac645c300f7db7784f2ad9f656e4bcbb3dce94f3b87fce0c97e.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/tables/da6a737fbca7aac645c300f7db7784f2ad9f656e4bcbb3dce94f3b87fce0c97e.jpg)

![ef7986f377599ac4333677804f99aa651a9c326e5414abb6e573c167a120f40a.jpg](../iclr_results/1672_Vision%20CNNs%20trained%20to%20estimate%20spatial%20latents%20learned%20similar%20ventral-stream-aligned%20representatio/tables/ef7986f377599ac4333677804f99aa651a9c326e5414abb6e573c167a120f40a.jpg)

## Value-aligned Behavior Cloning for Offline Reinforcement Learning via Bi-level Optimization


### Images

![0c3c78a20fe06ccc233ef2c29aaee85a2674e6764d6ba44be14dde0de0783879.jpg](../iclr_results/1673_Value-aligned%20Behavior%20Cloning%20for%20Offline%20Reinforcement%20Learning%20via%20Bi-level%20Optimization/images/0c3c78a20fe06ccc233ef2c29aaee85a2674e6764d6ba44be14dde0de0783879.jpg)

![0dfd9a073d23ce09d11d34cce968e2478d92e8bbc5dbbd781f9d03532f5fc1e2.jpg](../iclr_results/1673_Value-aligned%20Behavior%20Cloning%20for%20Offline%20Reinforcement%20Learning%20via%20Bi-level%20Optimization/images/0dfd9a073d23ce09d11d34cce968e2478d92e8bbc5dbbd781f9d03532f5fc1e2.jpg)

![127e7a07e5deea31d58ed3fbdc4538869027092fa6d8980c3fafc7ddfe4d0e15.jpg](../iclr_results/1673_Value-aligned%20Behavior%20Cloning%20for%20Offline%20Reinforcement%20Learning%20via%20Bi-level%20Optimization/images/127e7a07e5deea31d58ed3fbdc4538869027092fa6d8980c3fafc7ddfe4d0e15.jpg)

![3e5c2614aff353e61a947760f24a3894b4b2b0adf07db8f13b82e8af9a5a5176.jpg](../iclr_results/1673_Value-aligned%20Behavior%20Cloning%20for%20Offline%20Reinforcement%20Learning%20via%20Bi-level%20Optimization/images/3e5c2614aff353e61a947760f24a3894b4b2b0adf07db8f13b82e8af9a5a5176.jpg)

![4cf42b4eed0579b8d5694dcfa35eba0bb331c18330a05ec9663884f082680129.jpg](../iclr_results/1673_Value-aligned%20Behavior%20Cloning%20for%20Offline%20Reinforcement%20Learning%20via%20Bi-level%20Optimization/images/4cf42b4eed0579b8d5694dcfa35eba0bb331c18330a05ec9663884f082680129.jpg)

![70cd1a0b1bd3e974b2ef209c07f43f8b87f6df6b0021c184144315cec666028d.jpg](../iclr_results/1673_Value-aligned%20Behavior%20Cloning%20for%20Offline%20Reinforcement%20Learning%20via%20Bi-level%20Optimization/images/70cd1a0b1bd3e974b2ef209c07f43f8b87f6df6b0021c184144315cec666028d.jpg)

![7f87a7ee94d5bfa08716349034501cab9c527c6bcb4d5f5baa2a1b4e8c87e09c.jpg](../iclr_results/1673_Value-aligned%20Behavior%20Cloning%20for%20Offline%20Reinforcement%20Learning%20via%20Bi-level%20Optimization/images/7f87a7ee94d5bfa08716349034501cab9c527c6bcb4d5f5baa2a1b4e8c87e09c.jpg)

![84b7e8d1a68b6c8be5253509e1df47d6384846fc76e0105538b4835286f5413d.jpg](../iclr_results/1673_Value-aligned%20Behavior%20Cloning%20for%20Offline%20Reinforcement%20Learning%20via%20Bi-level%20Optimization/images/84b7e8d1a68b6c8be5253509e1df47d6384846fc76e0105538b4835286f5413d.jpg)

![85a252e570164b26be76a16a665ae8d90fd13ac290857c0a54381cb5d85c0889.jpg](../iclr_results/1673_Value-aligned%20Behavior%20Cloning%20for%20Offline%20Reinforcement%20Learning%20via%20Bi-level%20Optimization/images/85a252e570164b26be76a16a665ae8d90fd13ac290857c0a54381cb5d85c0889.jpg)

![8e88c9337734e95a82a66f6ce3bb0fd7abd66566c5291bd6757a6747f1ec4d08.jpg](../iclr_results/1673_Value-aligned%20Behavior%20Cloning%20for%20Offline%20Reinforcement%20Learning%20via%20Bi-level%20Optimization/images/8e88c9337734e95a82a66f6ce3bb0fd7abd66566c5291bd6757a6747f1ec4d08.jpg)

![957ee9542aec7022a0159b31de39f735b05bc970853bd1da9d59604e561c6780.jpg](../iclr_results/1673_Value-aligned%20Behavior%20Cloning%20for%20Offline%20Reinforcement%20Learning%20via%20Bi-level%20Optimization/images/957ee9542aec7022a0159b31de39f735b05bc970853bd1da9d59604e561c6780.jpg)

![c32bc2755cc5fe8c1177ca8f3d3857ca147533707359d180c2d05d56f9d47599.jpg](../iclr_results/1673_Value-aligned%20Behavior%20Cloning%20for%20Offline%20Reinforcement%20Learning%20via%20Bi-level%20Optimization/images/c32bc2755cc5fe8c1177ca8f3d3857ca147533707359d180c2d05d56f9d47599.jpg)

![ce866575038548674d074244d9ee045c3ddb22e141f7b97e8bf77000c52af059.jpg](../iclr_results/1673_Value-aligned%20Behavior%20Cloning%20for%20Offline%20Reinforcement%20Learning%20via%20Bi-level%20Optimization/images/ce866575038548674d074244d9ee045c3ddb22e141f7b97e8bf77000c52af059.jpg)

![defec53cdf776302974dae59bd1bb9f273736a61e4f983f67b9d1ed06512f88b.jpg](../iclr_results/1673_Value-aligned%20Behavior%20Cloning%20for%20Offline%20Reinforcement%20Learning%20via%20Bi-level%20Optimization/images/defec53cdf776302974dae59bd1bb9f273736a61e4f983f67b9d1ed06512f88b.jpg)

![e4c5063e56028b66608006ba4228519bf9330fe1a7f258fe3efe9f007541fc24.jpg](../iclr_results/1673_Value-aligned%20Behavior%20Cloning%20for%20Offline%20Reinforcement%20Learning%20via%20Bi-level%20Optimization/images/e4c5063e56028b66608006ba4228519bf9330fe1a7f258fe3efe9f007541fc24.jpg)

### Tables

![1a154e0e81e7bb6764c78d71a4100437d2031f476691a7ff620a320a08a7ef10.jpg](../iclr_results/1673_Value-aligned%20Behavior%20Cloning%20for%20Offline%20Reinforcement%20Learning%20via%20Bi-level%20Optimization/tables/1a154e0e81e7bb6764c78d71a4100437d2031f476691a7ff620a320a08a7ef10.jpg)

![32dbacc09742bcde22fa02178059511ff1f4864cbd7b2e92bbd9e3dd5e0395f3.jpg](../iclr_results/1673_Value-aligned%20Behavior%20Cloning%20for%20Offline%20Reinforcement%20Learning%20via%20Bi-level%20Optimization/tables/32dbacc09742bcde22fa02178059511ff1f4864cbd7b2e92bbd9e3dd5e0395f3.jpg)

![560b6fc09a374e09aa06b91cc75b8f0ac7e8f75eb1aab8c4505db09fc31cb895.jpg](../iclr_results/1673_Value-aligned%20Behavior%20Cloning%20for%20Offline%20Reinforcement%20Learning%20via%20Bi-level%20Optimization/tables/560b6fc09a374e09aa06b91cc75b8f0ac7e8f75eb1aab8c4505db09fc31cb895.jpg)

![88d0574b35cff98bd35cc17cac08deb9549a1d262c5f60dec4a89e376ca0cfc6.jpg](../iclr_results/1673_Value-aligned%20Behavior%20Cloning%20for%20Offline%20Reinforcement%20Learning%20via%20Bi-level%20Optimization/tables/88d0574b35cff98bd35cc17cac08deb9549a1d262c5f60dec4a89e376ca0cfc6.jpg)

![914d945ccf0bd6b94f8079703e72f38fbf28041b247fca190171d3c1efa80d24.jpg](../iclr_results/1673_Value-aligned%20Behavior%20Cloning%20for%20Offline%20Reinforcement%20Learning%20via%20Bi-level%20Optimization/tables/914d945ccf0bd6b94f8079703e72f38fbf28041b247fca190171d3c1efa80d24.jpg)

![ce5a91bf1c7c94207be5048310ecbfe7ed1fb881f59933eb385abd8be0cef8bd.jpg](../iclr_results/1673_Value-aligned%20Behavior%20Cloning%20for%20Offline%20Reinforcement%20Learning%20via%20Bi-level%20Optimization/tables/ce5a91bf1c7c94207be5048310ecbfe7ed1fb881f59933eb385abd8be0cef8bd.jpg)

![e47d36e5f5aa7d3b76ef08f99abb39cd77a27a1f7dd434dcaa175d56112dd1d9.jpg](../iclr_results/1673_Value-aligned%20Behavior%20Cloning%20for%20Offline%20Reinforcement%20Learning%20via%20Bi-level%20Optimization/tables/e47d36e5f5aa7d3b76ef08f99abb39cd77a27a1f7dd434dcaa175d56112dd1d9.jpg)

![e9b9535d07682505fbbd6b122e32b9389d4c39d7bc8b22103c7600c06640d7fa.jpg](../iclr_results/1673_Value-aligned%20Behavior%20Cloning%20for%20Offline%20Reinforcement%20Learning%20via%20Bi-level%20Optimization/tables/e9b9535d07682505fbbd6b122e32b9389d4c39d7bc8b22103c7600c06640d7fa.jpg)

## Persistent Pre-training Poisoning of LLMs


### Images

![13962eeda6fe3e58a4d4f00e31793df1e1a8143f7c1f1263630cc0297b977c41.jpg](../iclr_results/1674_Persistent%20Pre-training%20Poisoning%20of%20LLMs/images/13962eeda6fe3e58a4d4f00e31793df1e1a8143f7c1f1263630cc0297b977c41.jpg)

![14093e158b0f14b57c87d2fd645eb37a7538df0852d632ecd3db4095fc4dae33.jpg](../iclr_results/1674_Persistent%20Pre-training%20Poisoning%20of%20LLMs/images/14093e158b0f14b57c87d2fd645eb37a7538df0852d632ecd3db4095fc4dae33.jpg)

![175226abff5610f0f8e719aca47e27ad14dbb9280644e8044bfdce11fcbd1a36.jpg](../iclr_results/1674_Persistent%20Pre-training%20Poisoning%20of%20LLMs/images/175226abff5610f0f8e719aca47e27ad14dbb9280644e8044bfdce11fcbd1a36.jpg)

![2c215cafb59e498240734a6fd32788d221093dd86eacd3a1f1701383a4424573.jpg](../iclr_results/1674_Persistent%20Pre-training%20Poisoning%20of%20LLMs/images/2c215cafb59e498240734a6fd32788d221093dd86eacd3a1f1701383a4424573.jpg)

![2fc777d1bfc76879d32f93811db95437b64febac1eb99348064722508fb33b1d.jpg](../iclr_results/1674_Persistent%20Pre-training%20Poisoning%20of%20LLMs/images/2fc777d1bfc76879d32f93811db95437b64febac1eb99348064722508fb33b1d.jpg)

![54ccf0fbce3a2d1bba605fb6e800ce8a2e9a32cf1ecbdbac593353779d9d07df.jpg](../iclr_results/1674_Persistent%20Pre-training%20Poisoning%20of%20LLMs/images/54ccf0fbce3a2d1bba605fb6e800ce8a2e9a32cf1ecbdbac593353779d9d07df.jpg)

![5c45256f9366bcbdc4c0cc03378a7937303ba0408b1e6addf51a61d9eca9f9f4.jpg](../iclr_results/1674_Persistent%20Pre-training%20Poisoning%20of%20LLMs/images/5c45256f9366bcbdc4c0cc03378a7937303ba0408b1e6addf51a61d9eca9f9f4.jpg)

![65c99282d7fff106c4ce1d15c0a0cdeb03448cf1ff5fc9883f0be54a049d2084.jpg](../iclr_results/1674_Persistent%20Pre-training%20Poisoning%20of%20LLMs/images/65c99282d7fff106c4ce1d15c0a0cdeb03448cf1ff5fc9883f0be54a049d2084.jpg)

![6b7b30ca73a1dc2b8dc0fa3db7e25dea6c79a1b15fe9c68bf7ce9bb6106fc457.jpg](../iclr_results/1674_Persistent%20Pre-training%20Poisoning%20of%20LLMs/images/6b7b30ca73a1dc2b8dc0fa3db7e25dea6c79a1b15fe9c68bf7ce9bb6106fc457.jpg)

![7ed2992ca03e829fd2b38a0b9e9844ba96a90f6ee1fc956459c2b372ec84a17b.jpg](../iclr_results/1674_Persistent%20Pre-training%20Poisoning%20of%20LLMs/images/7ed2992ca03e829fd2b38a0b9e9844ba96a90f6ee1fc956459c2b372ec84a17b.jpg)

![8d4d0804d000817560684dd52df78fa92a5d935c60170f6b90ebe5fd2131a2e7.jpg](../iclr_results/1674_Persistent%20Pre-training%20Poisoning%20of%20LLMs/images/8d4d0804d000817560684dd52df78fa92a5d935c60170f6b90ebe5fd2131a2e7.jpg)

![a366d771c7c968375b35d85466111ab59d28f04a3308b9a1925c04231dc0d86e.jpg](../iclr_results/1674_Persistent%20Pre-training%20Poisoning%20of%20LLMs/images/a366d771c7c968375b35d85466111ab59d28f04a3308b9a1925c04231dc0d86e.jpg)

### Tables

![1186dd6ecb02ae5cf754f70e9a8f0ed55fb15058573882324828075d4edf6ade.jpg](../iclr_results/1674_Persistent%20Pre-training%20Poisoning%20of%20LLMs/tables/1186dd6ecb02ae5cf754f70e9a8f0ed55fb15058573882324828075d4edf6ade.jpg)

![726dbb9286786e63f8dad52fe68b614f34b4e55bf607ee4d3abfa4e641fa3033.jpg](../iclr_results/1674_Persistent%20Pre-training%20Poisoning%20of%20LLMs/tables/726dbb9286786e63f8dad52fe68b614f34b4e55bf607ee4d3abfa4e641fa3033.jpg)

![ed3e9a005e158449dd37c7649301b2e5c115df0e97883f62ce7f4ac3ad5dc932.jpg](../iclr_results/1674_Persistent%20Pre-training%20Poisoning%20of%20LLMs/tables/ed3e9a005e158449dd37c7649301b2e5c115df0e97883f62ce7f4ac3ad5dc932.jpg)

![fce45ed8566a31155d14793c4390adc7897894dcebc52270b8a65da48e60ec07.jpg](../iclr_results/1674_Persistent%20Pre-training%20Poisoning%20of%20LLMs/tables/fce45ed8566a31155d14793c4390adc7897894dcebc52270b8a65da48e60ec07.jpg)

## BadRobot: Jailbreaking Embodied LLM Agents in the Physical World


### Images

![0bd5f30fe468d62f8c4d7a630b9931dacafd07457cf391f0a0bd24f93b629988.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/images/0bd5f30fe468d62f8c4d7a630b9931dacafd07457cf391f0a0bd24f93b629988.jpg)

![20b8ac582eeae51f583dd78e6812870209f1c11a5a1ad5a7a6f016b0b81545a1.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/images/20b8ac582eeae51f583dd78e6812870209f1c11a5a1ad5a7a6f016b0b81545a1.jpg)

![2e49480e2344d29fe94d76d3afe20152dcb26ff0e3dcaf0c4df64a46062d5bd4.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/images/2e49480e2344d29fe94d76d3afe20152dcb26ff0e3dcaf0c4df64a46062d5bd4.jpg)

![35c41ec51997b59f6c15b360748fcc792fbc92397b633bc4f8180478a2eb3f51.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/images/35c41ec51997b59f6c15b360748fcc792fbc92397b633bc4f8180478a2eb3f51.jpg)

![4d5417a4f5b046516dcce05bfc9d7ed61a843447e6f828470f80ad6a3a8456a1.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/images/4d5417a4f5b046516dcce05bfc9d7ed61a843447e6f828470f80ad6a3a8456a1.jpg)

![501da0610d7c96322c9d75311a78893f7bdcbbd77979c55a3bb1e4d6bfa21af7.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/images/501da0610d7c96322c9d75311a78893f7bdcbbd77979c55a3bb1e4d6bfa21af7.jpg)

![50423daf143fc3e1e3b044b1299342a648b6ffefa7fccd6439b63d8fd8fd0c14.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/images/50423daf143fc3e1e3b044b1299342a648b6ffefa7fccd6439b63d8fd8fd0c14.jpg)

![58de316f267dcb784b4b402d7353ba402389d170140b110ca575e9d47a6a1033.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/images/58de316f267dcb784b4b402d7353ba402389d170140b110ca575e9d47a6a1033.jpg)

![729340cf6ab5964e81f6ff5ead3c9cb6b542a01c38f62f7571df244f3c25fefe.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/images/729340cf6ab5964e81f6ff5ead3c9cb6b542a01c38f62f7571df244f3c25fefe.jpg)

![7580fecabe5eb553a062106d6f4226317b86e54d8599f4983f1ec87069f0f129.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/images/7580fecabe5eb553a062106d6f4226317b86e54d8599f4983f1ec87069f0f129.jpg)

![76c37569068e53c33c0f6ea877dbbe17aee71e41c05b6c5b0272a11f1d97a70e.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/images/76c37569068e53c33c0f6ea877dbbe17aee71e41c05b6c5b0272a11f1d97a70e.jpg)

![7a376b02d29ed51e61202f0383d7c60692147b5ae3565a481269ab2103a526d2.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/images/7a376b02d29ed51e61202f0383d7c60692147b5ae3565a481269ab2103a526d2.jpg)

![82c397c48df947179389d929da77a3812332d727a87046ca7216bc94bcfdcb63.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/images/82c397c48df947179389d929da77a3812332d727a87046ca7216bc94bcfdcb63.jpg)

![892c15cd56ede3d4701a354f9d556c0e8aec6cbba5e29d7d16779f7f80d90f75.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/images/892c15cd56ede3d4701a354f9d556c0e8aec6cbba5e29d7d16779f7f80d90f75.jpg)

![a44f6406c273377403916afa9c2fefba38186bdfc1b25a0b2b88917f5920ad00.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/images/a44f6406c273377403916afa9c2fefba38186bdfc1b25a0b2b88917f5920ad00.jpg)

![a6ad07c510b15e657b3d8305ecde6159ec80e520f85f7c3dadcabbdf0882e3ff.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/images/a6ad07c510b15e657b3d8305ecde6159ec80e520f85f7c3dadcabbdf0882e3ff.jpg)

![bbdc90b848e0ac2055a95ebbc45c70998c20cff297cc79f272e3e92ae2334810.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/images/bbdc90b848e0ac2055a95ebbc45c70998c20cff297cc79f272e3e92ae2334810.jpg)

![de55ed72c7e68a68d7867fd9323ac86d5da7496fa997cd5cfc15b70e7f3b4329.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/images/de55ed72c7e68a68d7867fd9323ac86d5da7496fa997cd5cfc15b70e7f3b4329.jpg)

![e0b860c06ea42ea70bbf9ea96392f1048ef847ffd7912028dd1603596b4222d4.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/images/e0b860c06ea42ea70bbf9ea96392f1048ef847ffd7912028dd1603596b4222d4.jpg)

![f25a82dc6bbda5e6326d72882aab2485420ba2f14b380cf08e936632f1d5b262.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/images/f25a82dc6bbda5e6326d72882aab2485420ba2f14b380cf08e936632f1d5b262.jpg)

![f58b8450f2e8bf3bde1b58e8a0411a6f258ae326a25ec5daa23f32de14d3f239.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/images/f58b8450f2e8bf3bde1b58e8a0411a6f258ae326a25ec5daa23f32de14d3f239.jpg)

![fcf61ffc0d2b1804546e262875ec58812d32535a7c938c6caf8a3e7dada3f384.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/images/fcf61ffc0d2b1804546e262875ec58812d32535a7c938c6caf8a3e7dada3f384.jpg)

### Tables

![1395f048bd07a33cdb4f4c83496a674139bfaa777052d6d634d9fa8df6ab7702.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/tables/1395f048bd07a33cdb4f4c83496a674139bfaa777052d6d634d9fa8df6ab7702.jpg)

![2245199d417af0d728ef23c5b575eb6eb46dc777c8e29f1bac51e25f580e31dc.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/tables/2245199d417af0d728ef23c5b575eb6eb46dc777c8e29f1bac51e25f580e31dc.jpg)

![6f405d764900442d0a4d765a701712c79877f68be908d2629c98a7944543359c.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/tables/6f405d764900442d0a4d765a701712c79877f68be908d2629c98a7944543359c.jpg)

![77910564febf7def4bc7ca03c4ed4ee8105b547cf95761b04819211c9b172fc5.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/tables/77910564febf7def4bc7ca03c4ed4ee8105b547cf95761b04819211c9b172fc5.jpg)

![c34520f16db6218c48320ee7305662953c3ded334a8f63722c33edfd85d2d7e2.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/tables/c34520f16db6218c48320ee7305662953c3ded334a8f63722c33edfd85d2d7e2.jpg)

![fb68e7cffa1cedd4fbed3e36ca3cf74c9aebe209577ca540decbc5e8463b5f82.jpg](../iclr_results/1675_BadRobot_%20Jailbreaking%20Embodied%20LLM%20Agents%20in%20the%20Physical%20World/tables/fb68e7cffa1cedd4fbed3e36ca3cf74c9aebe209577ca540decbc5e8463b5f82.jpg)

## Disentangled Representation Learning with the Gromov-Monge Gap

### Images

![2304f0192e0b9d46b0a16bf4d2a7ba537b639db1d6bf5b3e3f32663f06647a5b.jpg](../iclr_results/1676_Disentangled%20Representation%20Learning%20with%20the%20Gromov-Monge%20Gap/images/2304f0192e0b9d46b0a16bf4d2a7ba537b639db1d6bf5b3e3f32663f06647a5b.jpg)

![3d3879aa3abe7400e4f95dcf61e9695539ba30e3cc8a278505b61008656abfca.jpg](../iclr_results/1676_Disentangled%20Representation%20Learning%20with%20the%20Gromov-Monge%20Gap/images/3d3879aa3abe7400e4f95dcf61e9695539ba30e3cc8a278505b61008656abfca.jpg)

![5ba02de3b9c33d2339ca9eac4738001d8e02724eff082248a047f7e54d597cc3.jpg](../iclr_results/1676_Disentangled%20Representation%20Learning%20with%20the%20Gromov-Monge%20Gap/images/5ba02de3b9c33d2339ca9eac4738001d8e02724eff082248a047f7e54d597cc3.jpg)

![9f0e9f747acd15bd465b9dabc24a42775541aabd092940a8041f35b4547128db.jpg](../iclr_results/1676_Disentangled%20Representation%20Learning%20with%20the%20Gromov-Monge%20Gap/images/9f0e9f747acd15bd465b9dabc24a42775541aabd092940a8041f35b4547128db.jpg)

![b99069c0a6f35e3071f7500ad5a2ed2e7905ac84acb94cc711044250689cd53b.jpg](../iclr_results/1676_Disentangled%20Representation%20Learning%20with%20the%20Gromov-Monge%20Gap/images/b99069c0a6f35e3071f7500ad5a2ed2e7905ac84acb94cc711044250689cd53b.jpg)

![ba830eaed06e181a292ce54ccf3fcc114d03995b1b18d09fd80360a018d59ab5.jpg](../iclr_results/1676_Disentangled%20Representation%20Learning%20with%20the%20Gromov-Monge%20Gap/images/ba830eaed06e181a292ce54ccf3fcc114d03995b1b18d09fd80360a018d59ab5.jpg)

![d82aa432f075b75a2bd31e50612b54c6e7be808d66857fe3260608c1d325d167.jpg](../iclr_results/1676_Disentangled%20Representation%20Learning%20with%20the%20Gromov-Monge%20Gap/images/d82aa432f075b75a2bd31e50612b54c6e7be808d66857fe3260608c1d325d167.jpg)

![e3e7b8702853a91aabd93d37d42bc10e50f1f75018b6baf516dca29eb85776a4.jpg](../iclr_results/1676_Disentangled%20Representation%20Learning%20with%20the%20Gromov-Monge%20Gap/images/e3e7b8702853a91aabd93d37d42bc10e50f1f75018b6baf516dca29eb85776a4.jpg)

### Tables

![0aa95c3badac7a106b3e1aa62571caf1131cc5f5307914f0e7ce3449d3d5b126.jpg](../iclr_results/1676_Disentangled%20Representation%20Learning%20with%20the%20Gromov-Monge%20Gap/tables/0aa95c3badac7a106b3e1aa62571caf1131cc5f5307914f0e7ce3449d3d5b126.jpg)

![2f6e27226ac6a66736bcaa17ecb4253f4830f98582668fc67b918652cac0ea9f.jpg](../iclr_results/1676_Disentangled%20Representation%20Learning%20with%20the%20Gromov-Monge%20Gap/tables/2f6e27226ac6a66736bcaa17ecb4253f4830f98582668fc67b918652cac0ea9f.jpg)

![32b380d8eaf70ef9b227a6dd9840acb085410e6ebb10793453da675ba1ca7664.jpg](../iclr_results/1676_Disentangled%20Representation%20Learning%20with%20the%20Gromov-Monge%20Gap/tables/32b380d8eaf70ef9b227a6dd9840acb085410e6ebb10793453da675ba1ca7664.jpg)

![79204ff7af81a25e0593071fc6e1dfdbfc5aa26453f272bb7399e9324533becd.jpg](../iclr_results/1676_Disentangled%20Representation%20Learning%20with%20the%20Gromov-Monge%20Gap/tables/79204ff7af81a25e0593071fc6e1dfdbfc5aa26453f272bb7399e9324533becd.jpg)

![d1b322fe03dfc9c571ba5d394faab6ec6fc5bbe9d23d10fc92579ca21ca45581.jpg](../iclr_results/1676_Disentangled%20Representation%20Learning%20with%20the%20Gromov-Monge%20Gap/tables/d1b322fe03dfc9c571ba5d394faab6ec6fc5bbe9d23d10fc92579ca21ca45581.jpg)

![d274f2182f97d87dda4eaabbd7014681683dd48023f77bb071596f6981681d41.jpg](../iclr_results/1676_Disentangled%20Representation%20Learning%20with%20the%20Gromov-Monge%20Gap/tables/d274f2182f97d87dda4eaabbd7014681683dd48023f77bb071596f6981681d41.jpg)

![e5e6cce94f1c696dac1a97b5382ae5d055ad4182dd931dcde5c646e701f7e2ab.jpg](../iclr_results/1676_Disentangled%20Representation%20Learning%20with%20the%20Gromov-Monge%20Gap/tables/e5e6cce94f1c696dac1a97b5382ae5d055ad4182dd931dcde5c646e701f7e2ab.jpg)
