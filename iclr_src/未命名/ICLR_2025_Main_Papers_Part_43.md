# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 43 of 100

## 目录 (Table of Contents)

1. [Sharper Guarantees for Learning Neural Network Classifiers with Gradient Methods](#Sharper-Guarantees-for-Learning-Neural-Network-Classifiers-with-Gradient-Methods)
2. [Sharpness-Aware Black-Box Optimization](#Sharpness-Aware-Black-Box-Optimization)
3. [Model Risk-sensitive Offline Reinforcement Learning](#Model-Risk-sensitive-Offline-Reinforcement-Learning)
4. [BANGS: Game-theoretic Node Selection for Graph Self-Training](#BANGS-Game-theoretic-Node-Selection-for-Graph-Self-Training)
5. [RNNs are not Transformers (Yet):  The Key Bottleneck on In-Context Retrieval](#RNNs-are-not-Transformers-Yet-The-Key-Bottleneck-on-In-Context-Retrieval)
6. [Mix-CPT: A Domain Adaptation Framework via Decoupling Knowledge Learning and Format Alignment](#Mix-CPT-A-Domain-Adaptation-Framework-via-Decoupling-Knowledge-Learning-and-Format-Alignment)
7. [Sensitivity Verification for Additive Decision Tree Ensembles](#Sensitivity-Verification-for-Additive-Decision-Tree-Ensembles)
8. [An Auditing Test to Detect Behavioral Shift in Language Models](#An-Auditing-Test-to-Detect-Behavioral-Shift-in-Language-Models)
9. [Rethinking the role of frames for SE(3)-invariant crystal structure modeling](#Rethinking-the-role-of-frames-for-SE3-invariant-crystal-structure-modeling)
10. [Learning to Select Nodes in Branch and Bound with Sufficient Tree Representation](#Learning-to-Select-Nodes-in-Branch-and-Bound-with-Sufficient-Tree-Representation)
11. [PortLLM: Personalizing Evolving Large Language Models with Training-Free and Portable Model Patches](#PortLLM-Personalizing-Evolving-Large-Language-Models-with-Training-Free-and-Portable-Model-Patches)
12. [T-JEPA: Augmentation-Free Self-Supervised Learning for Tabular Data](#T-JEPA-Augmentation-Free-Self-Supervised-Learning-for-Tabular-Data)
13. [Drop-Upcycling: Training Sparse Mixture of Experts with Partial Re-initialization](#Drop-Upcycling-Training-Sparse-Mixture-of-Experts-with-Partial-Re-initialization)
14. [Beyond Surface Structure: A Causal Assessment of LLMs' Comprehension ability](#Beyond-Surface-Structure-A-Causal-Assessment-of-LLMs-Comprehension-ability)
15. [Unify ML4TSP: Drawing Methodological Principles for TSP and Beyond from Streamlined Design Space of Learning and Search](#Unify-ML4TSP-Drawing-Methodological-Principles-for-TSP-and-Beyond-from-Streamlined-Design-Space-of-Learning-and-Search)
16. [Minimal Variance Model Aggregation: A principled, non-intrusive, and versatile integration of black box models](#Minimal-Variance-Model-Aggregation-A-principled-non-intrusive-and-versatile-integration-of-black-box-models)
17. [Rethinking Visual Counterfactual Explanations Through Region Constraint](#Rethinking-Visual-Counterfactual-Explanations-Through-Region-Constraint)
18. [A Conditional Independence Test in the Presence of Discretization](#A-Conditional-Independence-Test-in-the-Presence-of-Discretization)
19. [Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression](#Basis-Sharing-Cross-Layer-Parameter-Sharing-for-Large-Language-Model-Compression)
20. [ClimaQA: An Automated Evaluation Framework for Climate Question Answering Models](#ClimaQA-An-Automated-Evaluation-Framework-for-Climate-Question-Answering-Models)
21. [Directional Gradient Projection for Robust Fine-Tuning of Foundation Models](#Directional-Gradient-Projection-for-Robust-Fine-Tuning-of-Foundation-Models)
22. [SCBench: A KV Cache-Centric Analysis of Long-Context Methods](#SCBench-A-KV-Cache-Centric-Analysis-of-Long-Context-Methods)
23. [Score Forgetting Distillation: A Swift, Data-Free Method for Machine Unlearning in Diffusion Models](#Score-Forgetting-Distillation-A-Swift-Data-Free-Method-for-Machine-Unlearning-in-Diffusion-Models)
24. [GraphBridge: Towards Arbitrary Transfer Learning in GNNs](#GraphBridge-Towards-Arbitrary-Transfer-Learning-in-GNNs)
25. [Build-A-Scene: Interactive 3D Layout Control for Diffusion-Based Image Generation](#Build-A-Scene-Interactive-3D-Layout-Control-for-Diffusion-Based-Image-Generation)
26. [Strategist: Self-improvement of LLM Decision Making via Bi-Level Tree Search](#Strategist-Self-improvement-of-LLM-Decision-Making-via-Bi-Level-Tree-Search)
27. [Precise Parameter Localization for Textual Generation in Diffusion Models](#Precise-Parameter-Localization-for-Textual-Generation-in-Diffusion-Models)
28. [LIFe-GoM: Generalizable Human Rendering with Learned Iterative Feedback Over Multi-Resolution Gaussians-on-Mesh](#LIFe-GoM-Generalizable-Human-Rendering-with-Learned-Iterative-Feedback-Over-Multi-Resolution-Gaussians-on-Mesh)
29. [MAI: A Multi-turn Aggregation-Iteration Model for Composed Image Retrieval](#MAI-A-Multi-turn-Aggregation-Iteration-Model-for-Composed-Image-Retrieval)
30. [A Truncated Newton Method for Optimal Transport](#A-Truncated-Newton-Method-for-Optimal-Transport)
31. [Depth Any Video with Scalable Synthetic Data](#Depth-Any-Video-with-Scalable-Synthetic-Data)
32. [Residual-MPPI: Online Policy Customization for Continuous Control](#Residual-MPPI-Online-Policy-Customization-for-Continuous-Control)
33. [Efficient Biological Data Acquisition through Inference Set Design](#Efficient-Biological-Data-Acquisition-through-Inference-Set-Design)
34. [Dynamic Loss-Based Sample Reweighting for Improved Large Language Model Pretraining](#Dynamic-Loss-Based-Sample-Reweighting-for-Improved-Large-Language-Model-Pretraining)
35. [RFWave: Multi-band Rectified Flow for Audio Waveform Reconstruction](#RFWave-Multi-band-Rectified-Flow-for-Audio-Waveform-Reconstruction)
36. [Rare event modeling with self-regularized normalizing flows: what can we learn from a single failure?](#Rare-event-modeling-with-self-regularized-normalizing-flows-what-can-we-learn-from-a-single-failure)
37. [DocMIA: Document-Level Membership Inference Attacks against DocVQA Models](#DocMIA-Document-Level-Membership-Inference-Attacks-against-DocVQA-Models)

---


## Sharper Guarantees for Learning Neural Network Classifiers with Gradient Methods

### Images

![0ad64d650149733cf1adbd476ecc629c3f3c1b00bfcf77d99cdd77d77bfe9031.jpg](../iclr_results/1564_Improving%20Complex%20Reasoning%20with%20Dynamic%20Prompt%20Corruption_%20A%20Soft%20Prompt%20Optimization%20Approach/images/0ad64d650149733cf1adbd476ecc629c3f3c1b00bfcf77d99cdd77d77bfe9031.jpg)

![70a1d10726b37ed13ee5d01534f0eedb970a047cb3919646b0e62d54af44c8c3.jpg](../iclr_results/1564_Improving%20Complex%20Reasoning%20with%20Dynamic%20Prompt%20Corruption_%20A%20Soft%20Prompt%20Optimization%20Approach/images/70a1d10726b37ed13ee5d01534f0eedb970a047cb3919646b0e62d54af44c8c3.jpg)

![75bb62d9316560b01e10b523c1e8f9afcf5a09459e7be1bba09ae1b08f2280ea.jpg](../iclr_results/1564_Improving%20Complex%20Reasoning%20with%20Dynamic%20Prompt%20Corruption_%20A%20Soft%20Prompt%20Optimization%20Approach/images/75bb62d9316560b01e10b523c1e8f9afcf5a09459e7be1bba09ae1b08f2280ea.jpg)

![85d5f76336a7bf07946ff555b6258655d97d01f58dd0ce716344eb661fa70864.jpg](../iclr_results/1564_Improving%20Complex%20Reasoning%20with%20Dynamic%20Prompt%20Corruption_%20A%20Soft%20Prompt%20Optimization%20Approach/images/85d5f76336a7bf07946ff555b6258655d97d01f58dd0ce716344eb661fa70864.jpg)

![b10ff10fa07f7cbc0a60f637728cca1dee923c9eaeed8284bc4ca9e72c0b00f8.jpg](../iclr_results/1564_Improving%20Complex%20Reasoning%20with%20Dynamic%20Prompt%20Corruption_%20A%20Soft%20Prompt%20Optimization%20Approach/images/b10ff10fa07f7cbc0a60f637728cca1dee923c9eaeed8284bc4ca9e72c0b00f8.jpg)

![b9611f4e3ea830b4675cbf5cd0aa998649bd54b88b9c898f0e3e1bdec907d866.jpg](../iclr_results/1564_Improving%20Complex%20Reasoning%20with%20Dynamic%20Prompt%20Corruption_%20A%20Soft%20Prompt%20Optimization%20Approach/images/b9611f4e3ea830b4675cbf5cd0aa998649bd54b88b9c898f0e3e1bdec907d866.jpg)

![f787e3b797b2dc424d8aad582168fb200b8b7c629a1234fb1449d3e101474bbb.jpg](../iclr_results/1564_Improving%20Complex%20Reasoning%20with%20Dynamic%20Prompt%20Corruption_%20A%20Soft%20Prompt%20Optimization%20Approach/images/f787e3b797b2dc424d8aad582168fb200b8b7c629a1234fb1449d3e101474bbb.jpg)

### Tables

![bcf3c0b7f1d360e9b6b507a92662ef1d41bc30b451dfdfaed3d3de8480494335.jpg](../iclr_results/1564_Improving%20Complex%20Reasoning%20with%20Dynamic%20Prompt%20Corruption_%20A%20Soft%20Prompt%20Optimization%20Approach/tables/bcf3c0b7f1d360e9b6b507a92662ef1d41bc30b451dfdfaed3d3de8480494335.jpg)

![c8af5dbe82fca565deda08ebcc76c60bd92242b7fdab74e08af950858eda890b.jpg](../iclr_results/1564_Improving%20Complex%20Reasoning%20with%20Dynamic%20Prompt%20Corruption_%20A%20Soft%20Prompt%20Optimization%20Approach/tables/c8af5dbe82fca565deda08ebcc76c60bd92242b7fdab74e08af950858eda890b.jpg)

![d5967ebb8a2a4ef24da25dcfa38cb0cb75b6381a1296a2e485de98104698ac03.jpg](../iclr_results/1564_Improving%20Complex%20Reasoning%20with%20Dynamic%20Prompt%20Corruption_%20A%20Soft%20Prompt%20Optimization%20Approach/tables/d5967ebb8a2a4ef24da25dcfa38cb0cb75b6381a1296a2e485de98104698ac03.jpg)

![dd26ad2cf6d6faf7a13d86a6ae0f02014cc9e6aea17dcd0969068dad7ef6e3a0.jpg](../iclr_results/1564_Improving%20Complex%20Reasoning%20with%20Dynamic%20Prompt%20Corruption_%20A%20Soft%20Prompt%20Optimization%20Approach/tables/dd26ad2cf6d6faf7a13d86a6ae0f02014cc9e6aea17dcd0969068dad7ef6e3a0.jpg)

## Sharper Guarantees for Learning Neural Network Classifiers with Gradient Methods


### Images

![53f54675123252ef694fc0e5e693a1cbad82cb1b6fb6377e33b9c811a5b7bba5.jpg](../iclr_results/1565_Sharper%20Guarantees%20for%20Learning%20Neural%20Network%20Classifiers%20with%20Gradient%20Methods/images/53f54675123252ef694fc0e5e693a1cbad82cb1b6fb6377e33b9c811a5b7bba5.jpg)

![63d91540444807019dc5af354a7ec08ab0b8b352f760632fd28b43d9ed6f9f82.jpg](../iclr_results/1565_Sharper%20Guarantees%20for%20Learning%20Neural%20Network%20Classifiers%20with%20Gradient%20Methods/images/63d91540444807019dc5af354a7ec08ab0b8b352f760632fd28b43d9ed6f9f82.jpg)

![c4595e537628479e18192944801b2db67b533fa76d7e9d78ab719704f61afdd4.jpg](../iclr_results/1565_Sharper%20Guarantees%20for%20Learning%20Neural%20Network%20Classifiers%20with%20Gradient%20Methods/images/c4595e537628479e18192944801b2db67b533fa76d7e9d78ab719704f61afdd4.jpg)

![d3080ed2d910e54448e93531295364f67513dea4569aad0baa29fce3b735712e.jpg](../iclr_results/1565_Sharper%20Guarantees%20for%20Learning%20Neural%20Network%20Classifiers%20with%20Gradient%20Methods/images/d3080ed2d910e54448e93531295364f67513dea4569aad0baa29fce3b735712e.jpg)

### Tables

![41af06af175dca11b25d99dd06854d8697f9c864b64b0ba99a03253d826fe42d.jpg](../iclr_results/1565_Sharper%20Guarantees%20for%20Learning%20Neural%20Network%20Classifiers%20with%20Gradient%20Methods/tables/41af06af175dca11b25d99dd06854d8697f9c864b64b0ba99a03253d826fe42d.jpg)

![b605597a1417d3d547953af74184a62b0e0fcb9603d674efbd69b529e9b1d1a1.jpg](../iclr_results/1565_Sharper%20Guarantees%20for%20Learning%20Neural%20Network%20Classifiers%20with%20Gradient%20Methods/tables/b605597a1417d3d547953af74184a62b0e0fcb9603d674efbd69b529e9b1d1a1.jpg)

## Sharpness-Aware Black-Box Optimization


### Images

![3404f2e43763315013f53609958737956aac8c8203017f3d687435398e637ec4.jpg](../iclr_results/1566_Sharpness-Aware%20Black-Box%20Optimization/images/3404f2e43763315013f53609958737956aac8c8203017f3d687435398e637ec4.jpg)

![422b50be6fbec596745f34eeeeecbec6af3543c77a73fbaf551ebcdd2978debd.jpg](../iclr_results/1566_Sharpness-Aware%20Black-Box%20Optimization/images/422b50be6fbec596745f34eeeeecbec6af3543c77a73fbaf551ebcdd2978debd.jpg)

![74594e7a4d55820a14f4dc146ba587083fd60916921dd3ec8c79ee09b90b8ee2.jpg](../iclr_results/1566_Sharpness-Aware%20Black-Box%20Optimization/images/74594e7a4d55820a14f4dc146ba587083fd60916921dd3ec8c79ee09b90b8ee2.jpg)

![75a7d564c80f680fb890c8f5ae1ffed1b5b20c25078140c908e33bc568738ddf.jpg](../iclr_results/1566_Sharpness-Aware%20Black-Box%20Optimization/images/75a7d564c80f680fb890c8f5ae1ffed1b5b20c25078140c908e33bc568738ddf.jpg)

![9cf61b65fb6cb37745b789c98823f573253deaa17eaa12961fc6526517a54234.jpg](../iclr_results/1566_Sharpness-Aware%20Black-Box%20Optimization/images/9cf61b65fb6cb37745b789c98823f573253deaa17eaa12961fc6526517a54234.jpg)

![a11462c30a2efebce72ad0b05bc1bf715f944e4583c6c665f30c4d9a91ec9fcd.jpg](../iclr_results/1566_Sharpness-Aware%20Black-Box%20Optimization/images/a11462c30a2efebce72ad0b05bc1bf715f944e4583c6c665f30c4d9a91ec9fcd.jpg)

### Tables

![567d725e9c92164bcb187fff0c67a2aa1f02b3d5dbc47d47528b9e9f16ecefa5.jpg](../iclr_results/1566_Sharpness-Aware%20Black-Box%20Optimization/tables/567d725e9c92164bcb187fff0c67a2aa1f02b3d5dbc47d47528b9e9f16ecefa5.jpg)

![854e70acfc48252d595f8925ca154fb98c1bf3d363c7dcbe489a6b59984d3a14.jpg](../iclr_results/1566_Sharpness-Aware%20Black-Box%20Optimization/tables/854e70acfc48252d595f8925ca154fb98c1bf3d363c7dcbe489a6b59984d3a14.jpg)

![87868fa2fd51e395e93c1388cda50f2a84a72755ff40619c2f8ea901308fc454.jpg](../iclr_results/1566_Sharpness-Aware%20Black-Box%20Optimization/tables/87868fa2fd51e395e93c1388cda50f2a84a72755ff40619c2f8ea901308fc454.jpg)

## Model Risk-sensitive Offline Reinforcement Learning


### Images

![21ec35266a68f41e4eb1aa78c8d5980118f32f6d91d4861459574b051da65120.jpg](../iclr_results/1568_Model%20Risk-sensitive%20Offline%20Reinforcement%20Learning/images/21ec35266a68f41e4eb1aa78c8d5980118f32f6d91d4861459574b051da65120.jpg)

![3b2495bbd609954c0903f25f42647c915d09fc81fc748e172ace3a942e991315.jpg](../iclr_results/1568_Model%20Risk-sensitive%20Offline%20Reinforcement%20Learning/images/3b2495bbd609954c0903f25f42647c915d09fc81fc748e172ace3a942e991315.jpg)

![3f6113024fce575deb055bbea0dfcfbf5651df83d7dde961a94ec46def5c2614.jpg](../iclr_results/1568_Model%20Risk-sensitive%20Offline%20Reinforcement%20Learning/images/3f6113024fce575deb055bbea0dfcfbf5651df83d7dde961a94ec46def5c2614.jpg)

![40eb86e33fb0a3ffce871baa7cbe2d8bb3e664008e14cf2761ad7b0f2a0a785b.jpg](../iclr_results/1568_Model%20Risk-sensitive%20Offline%20Reinforcement%20Learning/images/40eb86e33fb0a3ffce871baa7cbe2d8bb3e664008e14cf2761ad7b0f2a0a785b.jpg)

![7928566872b7f245790f0595cdacd8de0cdcd8cacd5e0b2021a180f35a7f6a89.jpg](../iclr_results/1568_Model%20Risk-sensitive%20Offline%20Reinforcement%20Learning/images/7928566872b7f245790f0595cdacd8de0cdcd8cacd5e0b2021a180f35a7f6a89.jpg)

![7ace73445d1de063b9967d8ed48e2a1b702d8f1252ca2a53dbc552bf6a54f1cc.jpg](../iclr_results/1568_Model%20Risk-sensitive%20Offline%20Reinforcement%20Learning/images/7ace73445d1de063b9967d8ed48e2a1b702d8f1252ca2a53dbc552bf6a54f1cc.jpg)

![7c15a72d3d9268c7886fe847ae46fb0bd642450ab969697e39eabc5eb4f02938.jpg](../iclr_results/1568_Model%20Risk-sensitive%20Offline%20Reinforcement%20Learning/images/7c15a72d3d9268c7886fe847ae46fb0bd642450ab969697e39eabc5eb4f02938.jpg)

![8fc40ff4064f2ea0ab2451579d7633501789b63c9c06f31948afa51f00b8793a.jpg](../iclr_results/1568_Model%20Risk-sensitive%20Offline%20Reinforcement%20Learning/images/8fc40ff4064f2ea0ab2451579d7633501789b63c9c06f31948afa51f00b8793a.jpg)

![93db8bbfaa76433c2cbbb6b9a35468e484dcbd44a5ba53e8110f12f17918dda6.jpg](../iclr_results/1568_Model%20Risk-sensitive%20Offline%20Reinforcement%20Learning/images/93db8bbfaa76433c2cbbb6b9a35468e484dcbd44a5ba53e8110f12f17918dda6.jpg)

### Tables

![1210029d8d6b807d5a0554729c5e08702b234c109443467f5cbacdb8fc0162c0.jpg](../iclr_results/1568_Model%20Risk-sensitive%20Offline%20Reinforcement%20Learning/tables/1210029d8d6b807d5a0554729c5e08702b234c109443467f5cbacdb8fc0162c0.jpg)

![3c04eb111e95b2f1ade1c62aca827c105256e65d65df1107072cc1d469ba09bf.jpg](../iclr_results/1568_Model%20Risk-sensitive%20Offline%20Reinforcement%20Learning/tables/3c04eb111e95b2f1ade1c62aca827c105256e65d65df1107072cc1d469ba09bf.jpg)

![69e6136d3482d609dcd029558a5fbcb2fb2e2c7d951e7d1361ba189c27f88706.jpg](../iclr_results/1568_Model%20Risk-sensitive%20Offline%20Reinforcement%20Learning/tables/69e6136d3482d609dcd029558a5fbcb2fb2e2c7d951e7d1361ba189c27f88706.jpg)

![7f6f11af1ad2c24a608397b1713d8dbb17390c595a9ed976a98f353d9b614f26.jpg](../iclr_results/1568_Model%20Risk-sensitive%20Offline%20Reinforcement%20Learning/tables/7f6f11af1ad2c24a608397b1713d8dbb17390c595a9ed976a98f353d9b614f26.jpg)

![8932606c2e7234767fe47dcd1a5a99788aa9abe3425a6445654d0a9563f0b828.jpg](../iclr_results/1568_Model%20Risk-sensitive%20Offline%20Reinforcement%20Learning/tables/8932606c2e7234767fe47dcd1a5a99788aa9abe3425a6445654d0a9563f0b828.jpg)

![961c3c1226d9f3c90cd14b73d50a4f0d8edfc8711b4dd97a83a778698a3aa180.jpg](../iclr_results/1568_Model%20Risk-sensitive%20Offline%20Reinforcement%20Learning/tables/961c3c1226d9f3c90cd14b73d50a4f0d8edfc8711b4dd97a83a778698a3aa180.jpg)

![fbdb26c5b9fa6f354ac6c00ae26a9e992a1ea49814d467386e3e6c3366764e0c.jpg](../iclr_results/1568_Model%20Risk-sensitive%20Offline%20Reinforcement%20Learning/tables/fbdb26c5b9fa6f354ac6c00ae26a9e992a1ea49814d467386e3e6c3366764e0c.jpg)

## BANGS: Game-theoretic Node Selection for Graph Self-Training


### Images

![24f0c8b1d67881369293e1856a6c1b223e78327e86ad829d693880c5eaebd757.jpg](../iclr_results/1569_BANGS_%20Game-theoretic%20Node%20Selection%20for%20Graph%20Self-Training/images/24f0c8b1d67881369293e1856a6c1b223e78327e86ad829d693880c5eaebd757.jpg)

![3b575f1d216c7a9615b1befc920ff8a58a9937bb86cf32fac17a9ed2b8202124.jpg](../iclr_results/1569_BANGS_%20Game-theoretic%20Node%20Selection%20for%20Graph%20Self-Training/images/3b575f1d216c7a9615b1befc920ff8a58a9937bb86cf32fac17a9ed2b8202124.jpg)

![66eed1f4609f16c4fe043bb122aa240a2e194fa8bd87a6716397b1b1db5856c7.jpg](../iclr_results/1569_BANGS_%20Game-theoretic%20Node%20Selection%20for%20Graph%20Self-Training/images/66eed1f4609f16c4fe043bb122aa240a2e194fa8bd87a6716397b1b1db5856c7.jpg)

![d1c0e590efd608e9f07735787a0221b0dc011501ed37daa6ff42e32d11e84975.jpg](../iclr_results/1569_BANGS_%20Game-theoretic%20Node%20Selection%20for%20Graph%20Self-Training/images/d1c0e590efd608e9f07735787a0221b0dc011501ed37daa6ff42e32d11e84975.jpg)

### Tables

![0c3f6864d6f4276fbcd7b729bcc3cce8bf726d5c58a446f95070b916fe7624db.jpg](../iclr_results/1569_BANGS_%20Game-theoretic%20Node%20Selection%20for%20Graph%20Self-Training/tables/0c3f6864d6f4276fbcd7b729bcc3cce8bf726d5c58a446f95070b916fe7624db.jpg)

![4359b19fc6c01df3d414e2657b77918544b7b11a5fd17a7a4fb8b6e9a9237891.jpg](../iclr_results/1569_BANGS_%20Game-theoretic%20Node%20Selection%20for%20Graph%20Self-Training/tables/4359b19fc6c01df3d414e2657b77918544b7b11a5fd17a7a4fb8b6e9a9237891.jpg)

![45adcf8c791429c1d92c068aefbeaf0a10778a9989ad5b34bf2115c22cae6878.jpg](../iclr_results/1569_BANGS_%20Game-theoretic%20Node%20Selection%20for%20Graph%20Self-Training/tables/45adcf8c791429c1d92c068aefbeaf0a10778a9989ad5b34bf2115c22cae6878.jpg)

![5b5e84eecebfc7c8d444429c1005a24457db9438e227ff4ea80a0ec233f388aa.jpg](../iclr_results/1569_BANGS_%20Game-theoretic%20Node%20Selection%20for%20Graph%20Self-Training/tables/5b5e84eecebfc7c8d444429c1005a24457db9438e227ff4ea80a0ec233f388aa.jpg)

![61c46f192b8ce1d9fa38842ff390873160895987b33c73d1ca7039b289d02b85.jpg](../iclr_results/1569_BANGS_%20Game-theoretic%20Node%20Selection%20for%20Graph%20Self-Training/tables/61c46f192b8ce1d9fa38842ff390873160895987b33c73d1ca7039b289d02b85.jpg)

![62af174ce2472e7aa6ef218d6f690a4f2bb339cbcc5f2f72e4bcac5a8e5a7911.jpg](../iclr_results/1569_BANGS_%20Game-theoretic%20Node%20Selection%20for%20Graph%20Self-Training/tables/62af174ce2472e7aa6ef218d6f690a4f2bb339cbcc5f2f72e4bcac5a8e5a7911.jpg)

![6bd29c7b7e0f59a5d8041220bc6d0bfa55afef26a4ad483ab1cf6263196d25c6.jpg](../iclr_results/1569_BANGS_%20Game-theoretic%20Node%20Selection%20for%20Graph%20Self-Training/tables/6bd29c7b7e0f59a5d8041220bc6d0bfa55afef26a4ad483ab1cf6263196d25c6.jpg)

![85cd39ca26cbc45108a98618536577540f56b0d4c8587e4c41b619bf480a1d2d.jpg](../iclr_results/1569_BANGS_%20Game-theoretic%20Node%20Selection%20for%20Graph%20Self-Training/tables/85cd39ca26cbc45108a98618536577540f56b0d4c8587e4c41b619bf480a1d2d.jpg)

![85dcfb5e5dfac0f045c599011f3eb6a9dc5c8577db1466c398c7aee87a5af558.jpg](../iclr_results/1569_BANGS_%20Game-theoretic%20Node%20Selection%20for%20Graph%20Self-Training/tables/85dcfb5e5dfac0f045c599011f3eb6a9dc5c8577db1466c398c7aee87a5af558.jpg)

## RNNs are not Transformers (Yet):  The Key Bottleneck on In-Context Retrieval


### Images

![0343f065c250f68c3648582233cb54ba9424e83f68420b5ba75698fcc62daef5.jpg](../iclr_results/1570_RNNs%20are%20not%20Transformers%20%28Yet)_  The Key Bottleneck on In-Context Retrieval/images/0343f065c250f68c3648582233cb54ba9424e83f68420b5ba75698fcc62daef5.jpg)

![4771c5ba7d3ba138af77d8fc1642664a42a6dd4798fd720771d9c170306f3d20.jpg](../iclr_results/1570_RNNs%20are%20not%20Transformers%20%28Yet)_  The Key Bottleneck on In-Context Retrieval/images/4771c5ba7d3ba138af77d8fc1642664a42a6dd4798fd720771d9c170306f3d20.jpg)

![b9fdc2c748b8a8fd3fbea1f60eb5236dd760bf8d9b6a00812f2ec80d829fd777.jpg](../iclr_results/1570_RNNs%20are%20not%20Transformers%20%28Yet)_  The Key Bottleneck on In-Context Retrieval/images/b9fdc2c748b8a8fd3fbea1f60eb5236dd760bf8d9b6a00812f2ec80d829fd777.jpg)

![c635a70329f612d6f3a2685fbf7ce839b12e8246db6f4ccac0a6ae702a6b17e0.jpg](../iclr_results/1570_RNNs%20are%20not%20Transformers%20%28Yet)_  The Key Bottleneck on In-Context Retrieval/images/c635a70329f612d6f3a2685fbf7ce839b12e8246db6f4ccac0a6ae702a6b17e0.jpg)

![d6047b0e484efff2e1bded42948dc452e385bd937dfc140b1cfa58a972515afe.jpg](../iclr_results/1570_RNNs%20are%20not%20Transformers%20%28Yet)_  The Key Bottleneck on In-Context Retrieval/images/d6047b0e484efff2e1bded42948dc452e385bd937dfc140b1cfa58a972515afe.jpg)

![ed7b37d5dbcca044a4bfa0c40b646378f4a75a453b5d0facab423cfdcd20c5f3.jpg](../iclr_results/1570_RNNs%20are%20not%20Transformers%20%28Yet)_  The Key Bottleneck on In-Context Retrieval/images/ed7b37d5dbcca044a4bfa0c40b646378f4a75a453b5d0facab423cfdcd20c5f3.jpg)

### Tables

![bd6955c540e9bb532273aec8617821b255e44b382a885558735e4493f00f9748.jpg](../iclr_results/1570_RNNs%20are%20not%20Transformers%20%28Yet)_  The Key Bottleneck on In-Context Retrieval/tables/bd6955c540e9bb532273aec8617821b255e44b382a885558735e4493f00f9748.jpg)

## Mix-CPT: A Domain Adaptation Framework via Decoupling Knowledge Learning and Format Alignment


### Images

![46f69d37d523e5d7d53a6703a4cd433ed9b21b2855a49d1ee5af9303c8337079.jpg](../iclr_results/1571_Mix-CPT_%20A%20Domain%20Adaptation%20Framework%20via%20Decoupling%20Knowledge%20Learning%20and%20Format%20Alignment/images/46f69d37d523e5d7d53a6703a4cd433ed9b21b2855a49d1ee5af9303c8337079.jpg)

![58199b5c9cad2b8f493f6048f3eb20367fb93b4424ab32f24f164124b0861c6e.jpg](../iclr_results/1571_Mix-CPT_%20A%20Domain%20Adaptation%20Framework%20via%20Decoupling%20Knowledge%20Learning%20and%20Format%20Alignment/images/58199b5c9cad2b8f493f6048f3eb20367fb93b4424ab32f24f164124b0861c6e.jpg)

![61367f5bfbcfef02b0388c73d45c3b5678b65bfd2bb4a009013b45ab1f66f072.jpg](../iclr_results/1571_Mix-CPT_%20A%20Domain%20Adaptation%20Framework%20via%20Decoupling%20Knowledge%20Learning%20and%20Format%20Alignment/images/61367f5bfbcfef02b0388c73d45c3b5678b65bfd2bb4a009013b45ab1f66f072.jpg)

![6d818a4c67ff51192cabdfc3ed510512491d765f34c38da1bb3d4874d55a25a0.jpg](../iclr_results/1571_Mix-CPT_%20A%20Domain%20Adaptation%20Framework%20via%20Decoupling%20Knowledge%20Learning%20and%20Format%20Alignment/images/6d818a4c67ff51192cabdfc3ed510512491d765f34c38da1bb3d4874d55a25a0.jpg)

![7dd2d8eb4ebd28df21dadcbd371a8318638a51908fdc41d61c5e1dbabf1a7667.jpg](../iclr_results/1571_Mix-CPT_%20A%20Domain%20Adaptation%20Framework%20via%20Decoupling%20Knowledge%20Learning%20and%20Format%20Alignment/images/7dd2d8eb4ebd28df21dadcbd371a8318638a51908fdc41d61c5e1dbabf1a7667.jpg)

![829d3f30b12adee0b29d6855fe06ac70e895e3125c36fabd76af493c77c0842f.jpg](../iclr_results/1571_Mix-CPT_%20A%20Domain%20Adaptation%20Framework%20via%20Decoupling%20Knowledge%20Learning%20and%20Format%20Alignment/images/829d3f30b12adee0b29d6855fe06ac70e895e3125c36fabd76af493c77c0842f.jpg)

![d5c802e52020069d8d9ac555e144dca5f740a953a95247f701a51826d6b2f375.jpg](../iclr_results/1571_Mix-CPT_%20A%20Domain%20Adaptation%20Framework%20via%20Decoupling%20Knowledge%20Learning%20and%20Format%20Alignment/images/d5c802e52020069d8d9ac555e144dca5f740a953a95247f701a51826d6b2f375.jpg)

### Tables

![5532131404d01f859c2a0fbf520b9c17ec8e1a542e1e67ad8e98bcefbc4c48b2.jpg](../iclr_results/1571_Mix-CPT_%20A%20Domain%20Adaptation%20Framework%20via%20Decoupling%20Knowledge%20Learning%20and%20Format%20Alignment/tables/5532131404d01f859c2a0fbf520b9c17ec8e1a542e1e67ad8e98bcefbc4c48b2.jpg)

![6f2c4e45cd7875ed9a4486416148479c1f36eb499f4980dc593c59aa0a9f8860.jpg](../iclr_results/1571_Mix-CPT_%20A%20Domain%20Adaptation%20Framework%20via%20Decoupling%20Knowledge%20Learning%20and%20Format%20Alignment/tables/6f2c4e45cd7875ed9a4486416148479c1f36eb499f4980dc593c59aa0a9f8860.jpg)

![95cfd2a363874f1edf5d4d251af3c52ad14c20faaf135d7866b3367193b81057.jpg](../iclr_results/1571_Mix-CPT_%20A%20Domain%20Adaptation%20Framework%20via%20Decoupling%20Knowledge%20Learning%20and%20Format%20Alignment/tables/95cfd2a363874f1edf5d4d251af3c52ad14c20faaf135d7866b3367193b81057.jpg)

![ea72dbeeec1d7fca6d0fbdb3bc1a7e702af2fa2d3eaecabfffe9fb8bc76dca35.jpg](../iclr_results/1571_Mix-CPT_%20A%20Domain%20Adaptation%20Framework%20via%20Decoupling%20Knowledge%20Learning%20and%20Format%20Alignment/tables/ea72dbeeec1d7fca6d0fbdb3bc1a7e702af2fa2d3eaecabfffe9fb8bc76dca35.jpg)

## Sensitivity Verification for Additive Decision Tree Ensembles


### Images

![4127291a9b4e3699d20a909d2ddd4ba51c65f60119e912512b0c623c57ba6814.jpg](../iclr_results/1572_Sensitivity%20Verification%20for%20Additive%20Decision%20Tree%20Ensembles/images/4127291a9b4e3699d20a909d2ddd4ba51c65f60119e912512b0c623c57ba6814.jpg)

![9b144afae934fdb2e0446db4aea267caf56052b5a06e7dbe1c1edb65d5eb7fbf.jpg](../iclr_results/1572_Sensitivity%20Verification%20for%20Additive%20Decision%20Tree%20Ensembles/images/9b144afae934fdb2e0446db4aea267caf56052b5a06e7dbe1c1edb65d5eb7fbf.jpg)

### Tables

![1b336dbc5313ff6862cd15ba1138bf6a694979196ab60feefe45162cbc29e633.jpg](../iclr_results/1572_Sensitivity%20Verification%20for%20Additive%20Decision%20Tree%20Ensembles/tables/1b336dbc5313ff6862cd15ba1138bf6a694979196ab60feefe45162cbc29e633.jpg)

![7f04295b528df8f09ac77b04bb2caecdddaa6e8ef7d5772bdb8c1f902629f53a.jpg](../iclr_results/1572_Sensitivity%20Verification%20for%20Additive%20Decision%20Tree%20Ensembles/tables/7f04295b528df8f09ac77b04bb2caecdddaa6e8ef7d5772bdb8c1f902629f53a.jpg)

![8f50fb48bfa4e88561ab4860feb511270d4130746ee5e2ea3c1cd7fa68cecaa2.jpg](../iclr_results/1572_Sensitivity%20Verification%20for%20Additive%20Decision%20Tree%20Ensembles/tables/8f50fb48bfa4e88561ab4860feb511270d4130746ee5e2ea3c1cd7fa68cecaa2.jpg)

![af5cb9df014ab383f30068f888f0378fbb8a1163566fea731a428f9a31124db2.jpg](../iclr_results/1572_Sensitivity%20Verification%20for%20Additive%20Decision%20Tree%20Ensembles/tables/af5cb9df014ab383f30068f888f0378fbb8a1163566fea731a428f9a31124db2.jpg)

![cc3294cbbbeb8d9c74281866f766292bcce1e4ae2769d4aae5c5ae3943588f9a.jpg](../iclr_results/1572_Sensitivity%20Verification%20for%20Additive%20Decision%20Tree%20Ensembles/tables/cc3294cbbbeb8d9c74281866f766292bcce1e4ae2769d4aae5c5ae3943588f9a.jpg)

![f6cdf9287bb552898596333e3cba22028b3db29c72ec6158a8268ff43514d5fa.jpg](../iclr_results/1572_Sensitivity%20Verification%20for%20Additive%20Decision%20Tree%20Ensembles/tables/f6cdf9287bb552898596333e3cba22028b3db29c72ec6158a8268ff43514d5fa.jpg)

## An Auditing Test to Detect Behavioral Shift in Language Models


### Images

![232a072aefc373ec24ab10a63a935255541baf8583a2be556357147bde93fd00.jpg](../iclr_results/1573_An%20Auditing%20Test%20to%20Detect%20Behavioral%20Shift%20in%20Language%20Models/images/232a072aefc373ec24ab10a63a935255541baf8583a2be556357147bde93fd00.jpg)

![322d026d2f8a2eb16cf026398cf57867136cc9fbd4085f98c3883a43efb131a7.jpg](../iclr_results/1573_An%20Auditing%20Test%20to%20Detect%20Behavioral%20Shift%20in%20Language%20Models/images/322d026d2f8a2eb16cf026398cf57867136cc9fbd4085f98c3883a43efb131a7.jpg)

![3788f3b9a5ee3f9bac369eff97b83dc0c38744403eaf8fdc4ef2bff8be38b867.jpg](../iclr_results/1573_An%20Auditing%20Test%20to%20Detect%20Behavioral%20Shift%20in%20Language%20Models/images/3788f3b9a5ee3f9bac369eff97b83dc0c38744403eaf8fdc4ef2bff8be38b867.jpg)

![43517563ebf0f967ad791e43fb7343652b812bdfaf68055efbac814257a408ec.jpg](../iclr_results/1573_An%20Auditing%20Test%20to%20Detect%20Behavioral%20Shift%20in%20Language%20Models/images/43517563ebf0f967ad791e43fb7343652b812bdfaf68055efbac814257a408ec.jpg)

![4c83fc68f29b25dcd353504611ab61b307fb38c4ff150970a25093ba51cc4296.jpg](../iclr_results/1573_An%20Auditing%20Test%20to%20Detect%20Behavioral%20Shift%20in%20Language%20Models/images/4c83fc68f29b25dcd353504611ab61b307fb38c4ff150970a25093ba51cc4296.jpg)

![50fad75166590caf8b193c7a9e7a90f6410ae28c4be780c314808a6ce9f074bb.jpg](../iclr_results/1573_An%20Auditing%20Test%20to%20Detect%20Behavioral%20Shift%20in%20Language%20Models/images/50fad75166590caf8b193c7a9e7a90f6410ae28c4be780c314808a6ce9f074bb.jpg)

![5a3574a0312fa0c080ae605a124c35b93606c75707f93c1c4bc4f167f3d66091.jpg](../iclr_results/1573_An%20Auditing%20Test%20to%20Detect%20Behavioral%20Shift%20in%20Language%20Models/images/5a3574a0312fa0c080ae605a124c35b93606c75707f93c1c4bc4f167f3d66091.jpg)

![60ff8e1b335d2c6f8be7dfaf9587e646ca36b4e4517d25dc11081f8d00bad224.jpg](../iclr_results/1573_An%20Auditing%20Test%20to%20Detect%20Behavioral%20Shift%20in%20Language%20Models/images/60ff8e1b335d2c6f8be7dfaf9587e646ca36b4e4517d25dc11081f8d00bad224.jpg)

![7642661d1827d2319f29aba7c8f147171a8c1959e77e517164910e78f5e7e724.jpg](../iclr_results/1573_An%20Auditing%20Test%20to%20Detect%20Behavioral%20Shift%20in%20Language%20Models/images/7642661d1827d2319f29aba7c8f147171a8c1959e77e517164910e78f5e7e724.jpg)

![b4dfe3cfdd44dd8a9bb399fed9891efa53b07387ef19e6dcab72f438ac42ea8d.jpg](../iclr_results/1573_An%20Auditing%20Test%20to%20Detect%20Behavioral%20Shift%20in%20Language%20Models/images/b4dfe3cfdd44dd8a9bb399fed9891efa53b07387ef19e6dcab72f438ac42ea8d.jpg)

![cfb21a1bf2a62702ba7fc7b2533e9558abd10f45e41f9f02a00d984df75649d0.jpg](../iclr_results/1573_An%20Auditing%20Test%20to%20Detect%20Behavioral%20Shift%20in%20Language%20Models/images/cfb21a1bf2a62702ba7fc7b2533e9558abd10f45e41f9f02a00d984df75649d0.jpg)

![f7168ca0a405acad61d062d78ef38ed38c4a6066d6930bfda14e43acc7d12265.jpg](../iclr_results/1573_An%20Auditing%20Test%20to%20Detect%20Behavioral%20Shift%20in%20Language%20Models/images/f7168ca0a405acad61d062d78ef38ed38c4a6066d6930bfda14e43acc7d12265.jpg)

![ff099a90336d2a4aa74e254494deadae1e94278cab1c6fe6eb7dc673edeaa0df.jpg](../iclr_results/1573_An%20Auditing%20Test%20to%20Detect%20Behavioral%20Shift%20in%20Language%20Models/images/ff099a90336d2a4aa74e254494deadae1e94278cab1c6fe6eb7dc673edeaa0df.jpg)

### Tables

![11ab52d40989ad5c00311df0f5ec019de091feba4377d7e35c96dab7082445cc.jpg](../iclr_results/1573_An%20Auditing%20Test%20to%20Detect%20Behavioral%20Shift%20in%20Language%20Models/tables/11ab52d40989ad5c00311df0f5ec019de091feba4377d7e35c96dab7082445cc.jpg)

![8472efc600cf73ab6d8601782f8f87205cdca067b63d15664243b6b0223f0ffb.jpg](../iclr_results/1573_An%20Auditing%20Test%20to%20Detect%20Behavioral%20Shift%20in%20Language%20Models/tables/8472efc600cf73ab6d8601782f8f87205cdca067b63d15664243b6b0223f0ffb.jpg)

![cdf1be946e27ed9d02f76c5c36e2264aa342a5e774e3e867d470429b3ee90d09.jpg](../iclr_results/1573_An%20Auditing%20Test%20to%20Detect%20Behavioral%20Shift%20in%20Language%20Models/tables/cdf1be946e27ed9d02f76c5c36e2264aa342a5e774e3e867d470429b3ee90d09.jpg)

![e4e7f5edb41a40c75a560c01fcee78855a25f07295e60005f5b3227d7f602394.jpg](../iclr_results/1573_An%20Auditing%20Test%20to%20Detect%20Behavioral%20Shift%20in%20Language%20Models/tables/e4e7f5edb41a40c75a560c01fcee78855a25f07295e60005f5b3227d7f602394.jpg)

## Rethinking the role of frames for SE(3)-invariant crystal structure modeling


### Images

![0e9754c8804f76b61aeca048c2b4696e3158769af83c59d487d6e97782251cc2.jpg](../iclr_results/1574_Rethinking%20the%20role%20of%20frames%20for%20SE%283)-invariant crystal structure modeling/images/0e9754c8804f76b61aeca048c2b4696e3158769af83c59d487d6e97782251cc2.jpg)

![158336d1ecf37f836b2a7e7f70d5243eb7510e016e874f26738b0bb9026ce6a5.jpg](../iclr_results/1574_Rethinking%20the%20role%20of%20frames%20for%20SE%283)-invariant crystal structure modeling/images/158336d1ecf37f836b2a7e7f70d5243eb7510e016e874f26738b0bb9026ce6a5.jpg)

![29df8a389caecc901f8c144a8f9f62fb8ab46c3b5c676e01e69a80b2fe9915da.jpg](../iclr_results/1574_Rethinking%20the%20role%20of%20frames%20for%20SE%283)-invariant crystal structure modeling/images/29df8a389caecc901f8c144a8f9f62fb8ab46c3b5c676e01e69a80b2fe9915da.jpg)

![8d334df77a463d5d06665d5bf8829ef61fa7753e08a3193ae32a44af54f831c5.jpg](../iclr_results/1574_Rethinking%20the%20role%20of%20frames%20for%20SE%283)-invariant crystal structure modeling/images/8d334df77a463d5d06665d5bf8829ef61fa7753e08a3193ae32a44af54f831c5.jpg)

![a02a2004e01d345e0f643953baec3b65e6845355fa1da2e5d311c1cde378dbbc.jpg](../iclr_results/1574_Rethinking%20the%20role%20of%20frames%20for%20SE%283)-invariant crystal structure modeling/images/a02a2004e01d345e0f643953baec3b65e6845355fa1da2e5d311c1cde378dbbc.jpg)

![e4cba15c7097eecd56dbfb04d81d67db25ed1f8589f81d6304d82489d65a929c.jpg](../iclr_results/1574_Rethinking%20the%20role%20of%20frames%20for%20SE%283)-invariant crystal structure modeling/images/e4cba15c7097eecd56dbfb04d81d67db25ed1f8589f81d6304d82489d65a929c.jpg)

![f489f5dc046757dc7c43a9ccffaa94052db59f0a5100ea7a5a119549d2afbd71.jpg](../iclr_results/1574_Rethinking%20the%20role%20of%20frames%20for%20SE%283)-invariant crystal structure modeling/images/f489f5dc046757dc7c43a9ccffaa94052db59f0a5100ea7a5a119549d2afbd71.jpg)

### Tables

![550a8b416a23efdcbdeed9315b2137820a51485df73bfc22026f76f0e26f66e8.jpg](../iclr_results/1574_Rethinking%20the%20role%20of%20frames%20for%20SE%283)-invariant crystal structure modeling/tables/550a8b416a23efdcbdeed9315b2137820a51485df73bfc22026f76f0e26f66e8.jpg)

![58e11c2ce3bd91749e8369d907fb0084c968495568f363ac0718544596a9ead2.jpg](../iclr_results/1574_Rethinking%20the%20role%20of%20frames%20for%20SE%283)-invariant crystal structure modeling/tables/58e11c2ce3bd91749e8369d907fb0084c968495568f363ac0718544596a9ead2.jpg)

![5ae1f725eae5e4aae4a459d507c605afb4ac1372d798c77b437b87d732a644c5.jpg](../iclr_results/1574_Rethinking%20the%20role%20of%20frames%20for%20SE%283)-invariant crystal structure modeling/tables/5ae1f725eae5e4aae4a459d507c605afb4ac1372d798c77b437b87d732a644c5.jpg)

![960643b317d6287531f594df69ac51668edd341f2b6887183beddd8843d9d179.jpg](../iclr_results/1574_Rethinking%20the%20role%20of%20frames%20for%20SE%283)-invariant crystal structure modeling/tables/960643b317d6287531f594df69ac51668edd341f2b6887183beddd8843d9d179.jpg)

![cd1021798432faf0b2eb0b383aaa67a3f55e28810c63f1c65da13bb4ad2c32ed.jpg](../iclr_results/1574_Rethinking%20the%20role%20of%20frames%20for%20SE%283)-invariant crystal structure modeling/tables/cd1021798432faf0b2eb0b383aaa67a3f55e28810c63f1c65da13bb4ad2c32ed.jpg)

![ceda51a29c9219031d6800248774ddc70a794665be82bb3db94d070d9ea20416.jpg](../iclr_results/1574_Rethinking%20the%20role%20of%20frames%20for%20SE%283)-invariant crystal structure modeling/tables/ceda51a29c9219031d6800248774ddc70a794665be82bb3db94d070d9ea20416.jpg)

![d32f80924a4f62acc27be3e00e672b11c957f1ca8d9fde3dff24fb8277f80bd1.jpg](../iclr_results/1574_Rethinking%20the%20role%20of%20frames%20for%20SE%283)-invariant crystal structure modeling/tables/d32f80924a4f62acc27be3e00e672b11c957f1ca8d9fde3dff24fb8277f80bd1.jpg)

![ea3a59068124c9c4d5950550b7e8a06ef3d58cac7cae2875addb2d16a9c2b494.jpg](../iclr_results/1574_Rethinking%20the%20role%20of%20frames%20for%20SE%283)-invariant crystal structure modeling/tables/ea3a59068124c9c4d5950550b7e8a06ef3d58cac7cae2875addb2d16a9c2b494.jpg)

![fe75eed4d62a658bdcf4083435082ad56cfe56ed9f41985718374f741d81bd62.jpg](../iclr_results/1574_Rethinking%20the%20role%20of%20frames%20for%20SE%283)-invariant crystal structure modeling/tables/fe75eed4d62a658bdcf4083435082ad56cfe56ed9f41985718374f741d81bd62.jpg)

## Learning to Select Nodes in Branch and Bound with Sufficient Tree Representation


### Images

![0927c0c94fad7197fd1371010cde0c1eab5b24ad5ccf698e090f3e51c5ec186e.jpg](../iclr_results/1575_Learning%20to%20Select%20Nodes%20in%20Branch%20and%20Bound%20with%20Sufficient%20Tree%20Representation/images/0927c0c94fad7197fd1371010cde0c1eab5b24ad5ccf698e090f3e51c5ec186e.jpg)

![3cee97860a6896f4f198e2d1d96533b1c5665acf2da19214cfb85e16e450a868.jpg](../iclr_results/1575_Learning%20to%20Select%20Nodes%20in%20Branch%20and%20Bound%20with%20Sufficient%20Tree%20Representation/images/3cee97860a6896f4f198e2d1d96533b1c5665acf2da19214cfb85e16e450a868.jpg)

![9c0e3f41e071246601d38a03147c74e60769feea2e2909f42043d7694eb9f4f2.jpg](../iclr_results/1575_Learning%20to%20Select%20Nodes%20in%20Branch%20and%20Bound%20with%20Sufficient%20Tree%20Representation/images/9c0e3f41e071246601d38a03147c74e60769feea2e2909f42043d7694eb9f4f2.jpg)

![d425b71b720744e49fafeb34e03709f65498a677b28e91ba51ab9039ec0811d8.jpg](../iclr_results/1575_Learning%20to%20Select%20Nodes%20in%20Branch%20and%20Bound%20with%20Sufficient%20Tree%20Representation/images/d425b71b720744e49fafeb34e03709f65498a677b28e91ba51ab9039ec0811d8.jpg)

### Tables

![03a63bde21d397e1b6478a0a8dc2a19361939c384dcf5e5b4445fd94f528635e.jpg](../iclr_results/1575_Learning%20to%20Select%20Nodes%20in%20Branch%20and%20Bound%20with%20Sufficient%20Tree%20Representation/tables/03a63bde21d397e1b6478a0a8dc2a19361939c384dcf5e5b4445fd94f528635e.jpg)

![0518d71dddfa6e4fc507303446cab52f5bc5e798ca2f0572c45b90ea1e723534.jpg](../iclr_results/1575_Learning%20to%20Select%20Nodes%20in%20Branch%20and%20Bound%20with%20Sufficient%20Tree%20Representation/tables/0518d71dddfa6e4fc507303446cab52f5bc5e798ca2f0572c45b90ea1e723534.jpg)

![0b76f1b86b68a4352d647c29e0bf71f60b84c62e54f2f89179d38d493b724201.jpg](../iclr_results/1575_Learning%20to%20Select%20Nodes%20in%20Branch%20and%20Bound%20with%20Sufficient%20Tree%20Representation/tables/0b76f1b86b68a4352d647c29e0bf71f60b84c62e54f2f89179d38d493b724201.jpg)

![1a5be96e18b1b1a80eb9326a04e259c29e76b3c9474a0f0dd94dc078065652e3.jpg](../iclr_results/1575_Learning%20to%20Select%20Nodes%20in%20Branch%20and%20Bound%20with%20Sufficient%20Tree%20Representation/tables/1a5be96e18b1b1a80eb9326a04e259c29e76b3c9474a0f0dd94dc078065652e3.jpg)

![1ec21395a8c27a5b61114230eafc3dee997f52ffeb0ebc100397e46054eada95.jpg](../iclr_results/1575_Learning%20to%20Select%20Nodes%20in%20Branch%20and%20Bound%20with%20Sufficient%20Tree%20Representation/tables/1ec21395a8c27a5b61114230eafc3dee997f52ffeb0ebc100397e46054eada95.jpg)

![3868221f7e771d0b56cf897f410eb61030e96a51ed3c5b21eaf031a6fd1def65.jpg](../iclr_results/1575_Learning%20to%20Select%20Nodes%20in%20Branch%20and%20Bound%20with%20Sufficient%20Tree%20Representation/tables/3868221f7e771d0b56cf897f410eb61030e96a51ed3c5b21eaf031a6fd1def65.jpg)

![3a5249c7748810ec902512bcb7d98db8c7271652a6d87ef0207abe97f0d372e3.jpg](../iclr_results/1575_Learning%20to%20Select%20Nodes%20in%20Branch%20and%20Bound%20with%20Sufficient%20Tree%20Representation/tables/3a5249c7748810ec902512bcb7d98db8c7271652a6d87ef0207abe97f0d372e3.jpg)

![497391fda548977bf4a1cdcc529309c79c8e399db4564e7d1aec93ae81e144ef.jpg](../iclr_results/1575_Learning%20to%20Select%20Nodes%20in%20Branch%20and%20Bound%20with%20Sufficient%20Tree%20Representation/tables/497391fda548977bf4a1cdcc529309c79c8e399db4564e7d1aec93ae81e144ef.jpg)

![7ecace8cade1e4dd0493f9146a1118da49474bc69c248be4e2eac27b59cfbaf7.jpg](../iclr_results/1575_Learning%20to%20Select%20Nodes%20in%20Branch%20and%20Bound%20with%20Sufficient%20Tree%20Representation/tables/7ecace8cade1e4dd0493f9146a1118da49474bc69c248be4e2eac27b59cfbaf7.jpg)

![a3508c0ecb19f7477c2f0f54319d6f4f4b219ccf81ced2c46eea53a17f551a11.jpg](../iclr_results/1575_Learning%20to%20Select%20Nodes%20in%20Branch%20and%20Bound%20with%20Sufficient%20Tree%20Representation/tables/a3508c0ecb19f7477c2f0f54319d6f4f4b219ccf81ced2c46eea53a17f551a11.jpg)

![ba7531d793fe16286755912964cef18b92ad495c80b19126d6dcab49e20d0fa7.jpg](../iclr_results/1575_Learning%20to%20Select%20Nodes%20in%20Branch%20and%20Bound%20with%20Sufficient%20Tree%20Representation/tables/ba7531d793fe16286755912964cef18b92ad495c80b19126d6dcab49e20d0fa7.jpg)

![cd3856a78326fac4b5c06156b2f7b6d8ea01d187427c773ff9dca95ca2f04b03.jpg](../iclr_results/1575_Learning%20to%20Select%20Nodes%20in%20Branch%20and%20Bound%20with%20Sufficient%20Tree%20Representation/tables/cd3856a78326fac4b5c06156b2f7b6d8ea01d187427c773ff9dca95ca2f04b03.jpg)

![d0507fbf91451c50a4dd93862150808450c5c9167c9ae3059a7cd87a6aa2d843.jpg](../iclr_results/1575_Learning%20to%20Select%20Nodes%20in%20Branch%20and%20Bound%20with%20Sufficient%20Tree%20Representation/tables/d0507fbf91451c50a4dd93862150808450c5c9167c9ae3059a7cd87a6aa2d843.jpg)

![d5023c0c4086451f67e7c54459a1df9dae5859565c7385dff14623b922b8ddc5.jpg](../iclr_results/1575_Learning%20to%20Select%20Nodes%20in%20Branch%20and%20Bound%20with%20Sufficient%20Tree%20Representation/tables/d5023c0c4086451f67e7c54459a1df9dae5859565c7385dff14623b922b8ddc5.jpg)

![d6047f4199bb054dd6ba6e42aac67a6e91eeb0dbfdd7e508585ec3fa9e7a6ca1.jpg](../iclr_results/1575_Learning%20to%20Select%20Nodes%20in%20Branch%20and%20Bound%20with%20Sufficient%20Tree%20Representation/tables/d6047f4199bb054dd6ba6e42aac67a6e91eeb0dbfdd7e508585ec3fa9e7a6ca1.jpg)

![e4a56134c6ca0e9abaea7743dc42ca52daf291c7c23d7b50ed825c5ec1be9755.jpg](../iclr_results/1575_Learning%20to%20Select%20Nodes%20in%20Branch%20and%20Bound%20with%20Sufficient%20Tree%20Representation/tables/e4a56134c6ca0e9abaea7743dc42ca52daf291c7c23d7b50ed825c5ec1be9755.jpg)

## PortLLM: Personalizing Evolving Large Language Models with Training-Free and Portable Model Patches


### Images

![b6c98b92971b5e1320fe8947bfb0759314bd93874a1f6d0e91126829866cd868.jpg](../iclr_results/1576_PortLLM_%20Personalizing%20Evolving%20Large%20Language%20Models%20with%20Training-Free%20and%20Portable%20Model%20Patches/images/b6c98b92971b5e1320fe8947bfb0759314bd93874a1f6d0e91126829866cd868.jpg)

![fd8d300e431c679067ac5365b849afb1a6eaebcdcbbccc8ab7f291198c185b7f.jpg](../iclr_results/1576_PortLLM_%20Personalizing%20Evolving%20Large%20Language%20Models%20with%20Training-Free%20and%20Portable%20Model%20Patches/images/fd8d300e431c679067ac5365b849afb1a6eaebcdcbbccc8ab7f291198c185b7f.jpg)

### Tables

![081106f7cbbb6dfb3bd616f3c1196986a00279ef1ab6a99df612d335c4cf6bba.jpg](../iclr_results/1576_PortLLM_%20Personalizing%20Evolving%20Large%20Language%20Models%20with%20Training-Free%20and%20Portable%20Model%20Patches/tables/081106f7cbbb6dfb3bd616f3c1196986a00279ef1ab6a99df612d335c4cf6bba.jpg)

![1d24ce84b614235668cb2af5e655a1af251665612fe78253a6751e81e7822098.jpg](../iclr_results/1576_PortLLM_%20Personalizing%20Evolving%20Large%20Language%20Models%20with%20Training-Free%20and%20Portable%20Model%20Patches/tables/1d24ce84b614235668cb2af5e655a1af251665612fe78253a6751e81e7822098.jpg)

![22e48de088cfbc5e40851a9a933ea72301235289a8a15063ac07e3337d27fc4f.jpg](../iclr_results/1576_PortLLM_%20Personalizing%20Evolving%20Large%20Language%20Models%20with%20Training-Free%20and%20Portable%20Model%20Patches/tables/22e48de088cfbc5e40851a9a933ea72301235289a8a15063ac07e3337d27fc4f.jpg)

![4b8b79e2a4c26916fafe96f1ab5991be9f6c76c4c10996de199a42bdd29b9c9c.jpg](../iclr_results/1576_PortLLM_%20Personalizing%20Evolving%20Large%20Language%20Models%20with%20Training-Free%20and%20Portable%20Model%20Patches/tables/4b8b79e2a4c26916fafe96f1ab5991be9f6c76c4c10996de199a42bdd29b9c9c.jpg)

![a87ec57b90f8652809ee793bbe63c07c59aa26732ec46d27a0e17ddf0b3d9192.jpg](../iclr_results/1576_PortLLM_%20Personalizing%20Evolving%20Large%20Language%20Models%20with%20Training-Free%20and%20Portable%20Model%20Patches/tables/a87ec57b90f8652809ee793bbe63c07c59aa26732ec46d27a0e17ddf0b3d9192.jpg)

![baa89c2f39a6af7386c3189550ed4376e3958b87ad52893a58224ad7ab4f5152.jpg](../iclr_results/1576_PortLLM_%20Personalizing%20Evolving%20Large%20Language%20Models%20with%20Training-Free%20and%20Portable%20Model%20Patches/tables/baa89c2f39a6af7386c3189550ed4376e3958b87ad52893a58224ad7ab4f5152.jpg)

![e1ca517ee248bf79f88793d0358ca28b7efb1989ba8eb87885965cc2a612d6ef.jpg](../iclr_results/1576_PortLLM_%20Personalizing%20Evolving%20Large%20Language%20Models%20with%20Training-Free%20and%20Portable%20Model%20Patches/tables/e1ca517ee248bf79f88793d0358ca28b7efb1989ba8eb87885965cc2a612d6ef.jpg)

![f83594e9437848e90b04623893b3cd856e764cb9851e222f847cf9ab3174de86.jpg](../iclr_results/1576_PortLLM_%20Personalizing%20Evolving%20Large%20Language%20Models%20with%20Training-Free%20and%20Portable%20Model%20Patches/tables/f83594e9437848e90b04623893b3cd856e764cb9851e222f847cf9ab3174de86.jpg)

## T-JEPA: Augmentation-Free Self-Supervised Learning for Tabular Data


### Images

![19a3bddfd38e52bf9e04b5a13252d0b419ce8b5aa142ac183a03416d3967a5b3.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/images/19a3bddfd38e52bf9e04b5a13252d0b419ce8b5aa142ac183a03416d3967a5b3.jpg)

![22c921b2125d56c093f04fc3d9cc3b5097f4ca04a713cd998018f8f005378116.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/images/22c921b2125d56c093f04fc3d9cc3b5097f4ca04a713cd998018f8f005378116.jpg)

![45a64223c9292b8ac231719ead3f05873928051443eba91defc03a2f35fc1289.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/images/45a64223c9292b8ac231719ead3f05873928051443eba91defc03a2f35fc1289.jpg)

![5311427da097543ee7057434a61ef62f8ea1d932387a3139ee95116477574b2e.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/images/5311427da097543ee7057434a61ef62f8ea1d932387a3139ee95116477574b2e.jpg)

![7f848b1f660e83f2c3f4e0ddfbeb88c7fd7fdf444a013d17851a7ef1c7c2263f.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/images/7f848b1f660e83f2c3f4e0ddfbeb88c7fd7fdf444a013d17851a7ef1c7c2263f.jpg)

![85d620b197c34509277334ad2c2acc4d770cc401a5444358545840ee96f94a27.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/images/85d620b197c34509277334ad2c2acc4d770cc401a5444358545840ee96f94a27.jpg)

![983a7927fe7806a37ade704004a71b6922aca6666398cb4fae8a55762155b753.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/images/983a7927fe7806a37ade704004a71b6922aca6666398cb4fae8a55762155b753.jpg)

### Tables

![0f6f1b4d93305e8eeb3f8704316fcfebc4fd68ced29c220aecaff55b61a3388f.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/tables/0f6f1b4d93305e8eeb3f8704316fcfebc4fd68ced29c220aecaff55b61a3388f.jpg)

![136411912912e094c38fc64c8fef39494ac6bbbd588b9b3bdade95bc95b7a0f9.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/tables/136411912912e094c38fc64c8fef39494ac6bbbd588b9b3bdade95bc95b7a0f9.jpg)

![196fc769ef7262031fff85adf85e678c5be7d6eb80d00d5a2cec629bee5e89ee.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/tables/196fc769ef7262031fff85adf85e678c5be7d6eb80d00d5a2cec629bee5e89ee.jpg)

![224b2e2972abf871f39f45e2386bb47c55776fef5246aeb10de79b805ad5771e.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/tables/224b2e2972abf871f39f45e2386bb47c55776fef5246aeb10de79b805ad5771e.jpg)

![348503466d29579844c3d80c0626e8f865beb161f2c01322ba763f9a58a5e4ca.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/tables/348503466d29579844c3d80c0626e8f865beb161f2c01322ba763f9a58a5e4ca.jpg)

![593137c7001652779de571df1ad4c255c38dc7e79ee9708fb7f1a8b5bfea58e4.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/tables/593137c7001652779de571df1ad4c255c38dc7e79ee9708fb7f1a8b5bfea58e4.jpg)

![70e963a8e706778cf7f30ab519e04f645bda89ec7a3ccfd974d03250a484eb65.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/tables/70e963a8e706778cf7f30ab519e04f645bda89ec7a3ccfd974d03250a484eb65.jpg)

![754b1a4cb0f108ec7ee5737b41594a89b4b812dab0548e7df16d3c6f07018e56.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/tables/754b1a4cb0f108ec7ee5737b41594a89b4b812dab0548e7df16d3c6f07018e56.jpg)

![79d621b5489a96e33ce8798a9172cd58937bdbd9d9586db11a41c8594115b792.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/tables/79d621b5489a96e33ce8798a9172cd58937bdbd9d9586db11a41c8594115b792.jpg)

![7f0c4c12b75d65e7c4d66875a2f961d1b080397fc0ae601842a4bc763efcf7e8.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/tables/7f0c4c12b75d65e7c4d66875a2f961d1b080397fc0ae601842a4bc763efcf7e8.jpg)

![8de3fc0070975aebff9b2c54121d60f332f6f49c5d2b333b3af059c2f00a1c94.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/tables/8de3fc0070975aebff9b2c54121d60f332f6f49c5d2b333b3af059c2f00a1c94.jpg)

![9dcea5dc340586102597e87c7f60d2c5f231a6303d4a0cce5bf60211eb6dcc0b.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/tables/9dcea5dc340586102597e87c7f60d2c5f231a6303d4a0cce5bf60211eb6dcc0b.jpg)

![b094b1f56c8d541f9c769b88b985e35929c0286d6ce94cfaf7200cbad727611a.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/tables/b094b1f56c8d541f9c769b88b985e35929c0286d6ce94cfaf7200cbad727611a.jpg)

![cb1c6ed26f8ac83b4d2aa6c924a759795083bf264ce252cbd5a621ed7eb3f60f.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/tables/cb1c6ed26f8ac83b4d2aa6c924a759795083bf264ce252cbd5a621ed7eb3f60f.jpg)

![d96c13d65905ca96d8928cc1ebfa25a9b764254886b9dedc416c2f01ce9ae9d0.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/tables/d96c13d65905ca96d8928cc1ebfa25a9b764254886b9dedc416c2f01ce9ae9d0.jpg)

![db0edb854354cf188df9a079beb603587782a69d7072f67b54b81a17722acb06.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/tables/db0edb854354cf188df9a079beb603587782a69d7072f67b54b81a17722acb06.jpg)

![e6aeecc945e33c9de90bd407d4f343094c83e1233b444906bb3ed95fbbabbab7.jpg](../iclr_results/1577_T-JEPA_%20Augmentation-Free%20Self-Supervised%20Learning%20for%20Tabular%20Data/tables/e6aeecc945e33c9de90bd407d4f343094c83e1233b444906bb3ed95fbbabbab7.jpg)

## Drop-Upcycling: Training Sparse Mixture of Experts with Partial Re-initialization


### Images

![04fe9706920358b4a317e09a4507e93da9d6a5493a22a86ab789ebe35de276db.jpg](../iclr_results/1578_Drop-Upcycling_%20Training%20Sparse%20Mixture%20of%20Experts%20with%20Partial%20Re-initialization/images/04fe9706920358b4a317e09a4507e93da9d6a5493a22a86ab789ebe35de276db.jpg)

![3903419f80441f5211f07a854a3c4244e2435d7ef2929551563ed33ee53fc879.jpg](../iclr_results/1578_Drop-Upcycling_%20Training%20Sparse%20Mixture%20of%20Experts%20with%20Partial%20Re-initialization/images/3903419f80441f5211f07a854a3c4244e2435d7ef2929551563ed33ee53fc879.jpg)

![45329ff0eda86f0937a0fc6ec758138fb1ad0d1e8e35d85983ea2a0993010bfa.jpg](../iclr_results/1578_Drop-Upcycling_%20Training%20Sparse%20Mixture%20of%20Experts%20with%20Partial%20Re-initialization/images/45329ff0eda86f0937a0fc6ec758138fb1ad0d1e8e35d85983ea2a0993010bfa.jpg)

![571fff5d6b99f17478f5d38926b23e4d6dc7cf690b59f45f9e0cec8bc4468d05.jpg](../iclr_results/1578_Drop-Upcycling_%20Training%20Sparse%20Mixture%20of%20Experts%20with%20Partial%20Re-initialization/images/571fff5d6b99f17478f5d38926b23e4d6dc7cf690b59f45f9e0cec8bc4468d05.jpg)

![5c8442d442ffadb2d2d3d6bc5fe149d5ed43b137ca567ba5e4ca92f97b014b89.jpg](../iclr_results/1578_Drop-Upcycling_%20Training%20Sparse%20Mixture%20of%20Experts%20with%20Partial%20Re-initialization/images/5c8442d442ffadb2d2d3d6bc5fe149d5ed43b137ca567ba5e4ca92f97b014b89.jpg)

![7329479f384469495e92e714d371109fbd1ec569ae3adb2d0b4c7453b1855870.jpg](../iclr_results/1578_Drop-Upcycling_%20Training%20Sparse%20Mixture%20of%20Experts%20with%20Partial%20Re-initialization/images/7329479f384469495e92e714d371109fbd1ec569ae3adb2d0b4c7453b1855870.jpg)

![a6614931fd8b9bc1b708146d4752e81aed7d938911a89a0d6015c30447ce26c3.jpg](../iclr_results/1578_Drop-Upcycling_%20Training%20Sparse%20Mixture%20of%20Experts%20with%20Partial%20Re-initialization/images/a6614931fd8b9bc1b708146d4752e81aed7d938911a89a0d6015c30447ce26c3.jpg)

![b9f02f5b04b23fba05965a0764d0aaf4dea8a76ee7ad54c80fbbc9729ea431ed.jpg](../iclr_results/1578_Drop-Upcycling_%20Training%20Sparse%20Mixture%20of%20Experts%20with%20Partial%20Re-initialization/images/b9f02f5b04b23fba05965a0764d0aaf4dea8a76ee7ad54c80fbbc9729ea431ed.jpg)

![be93652763dd9e386ad96b6b2a34f204bdc0320111581182df01f4dbec302751.jpg](../iclr_results/1578_Drop-Upcycling_%20Training%20Sparse%20Mixture%20of%20Experts%20with%20Partial%20Re-initialization/images/be93652763dd9e386ad96b6b2a34f204bdc0320111581182df01f4dbec302751.jpg)

![bfedc4c99050c25ed8828d46aaee3837352a4fda766b028585b0a401d1b56be4.jpg](../iclr_results/1578_Drop-Upcycling_%20Training%20Sparse%20Mixture%20of%20Experts%20with%20Partial%20Re-initialization/images/bfedc4c99050c25ed8828d46aaee3837352a4fda766b028585b0a401d1b56be4.jpg)

![eb14599c939aeb0d5571a0fda1c0a4a5222e12033b3dc21eae73b756ec3a58fc.jpg](../iclr_results/1578_Drop-Upcycling_%20Training%20Sparse%20Mixture%20of%20Experts%20with%20Partial%20Re-initialization/images/eb14599c939aeb0d5571a0fda1c0a4a5222e12033b3dc21eae73b756ec3a58fc.jpg)

![ec9a4036afe2191c4d26e72b5a40fb71bbe6941ee21a36baf799c044e11fc03f.jpg](../iclr_results/1578_Drop-Upcycling_%20Training%20Sparse%20Mixture%20of%20Experts%20with%20Partial%20Re-initialization/images/ec9a4036afe2191c4d26e72b5a40fb71bbe6941ee21a36baf799c044e11fc03f.jpg)

![f516f43dda1308a00a03bc2b4cfad38b34097247ad8dde78ea835ea7e708dbc5.jpg](../iclr_results/1578_Drop-Upcycling_%20Training%20Sparse%20Mixture%20of%20Experts%20with%20Partial%20Re-initialization/images/f516f43dda1308a00a03bc2b4cfad38b34097247ad8dde78ea835ea7e708dbc5.jpg)

![fc0f46fbc6d1a6017b71ac11b09677bcd6cdc1015bbaeb02e2c899543c2d4a74.jpg](../iclr_results/1578_Drop-Upcycling_%20Training%20Sparse%20Mixture%20of%20Experts%20with%20Partial%20Re-initialization/images/fc0f46fbc6d1a6017b71ac11b09677bcd6cdc1015bbaeb02e2c899543c2d4a74.jpg)

### Tables

![3930795958272bd81b7f244a25c74c4ce4042103f625d0cae657e95db75c5d0d.jpg](../iclr_results/1578_Drop-Upcycling_%20Training%20Sparse%20Mixture%20of%20Experts%20with%20Partial%20Re-initialization/tables/3930795958272bd81b7f244a25c74c4ce4042103f625d0cae657e95db75c5d0d.jpg)

![3f31fbb529aa3a18ca8bf5688833e9bce142fc88293e724b22d81a940b7b669f.jpg](../iclr_results/1578_Drop-Upcycling_%20Training%20Sparse%20Mixture%20of%20Experts%20with%20Partial%20Re-initialization/tables/3f31fbb529aa3a18ca8bf5688833e9bce142fc88293e724b22d81a940b7b669f.jpg)

![45b3d244a74853c630018bea62bdb7cfa7b7d87b60e747be488b171ecf83e6a2.jpg](../iclr_results/1578_Drop-Upcycling_%20Training%20Sparse%20Mixture%20of%20Experts%20with%20Partial%20Re-initialization/tables/45b3d244a74853c630018bea62bdb7cfa7b7d87b60e747be488b171ecf83e6a2.jpg)

![468991f3020cf2c6894b92c75306d1d4f5db337d2fe51cb9500a911e5c347ea6.jpg](../iclr_results/1578_Drop-Upcycling_%20Training%20Sparse%20Mixture%20of%20Experts%20with%20Partial%20Re-initialization/tables/468991f3020cf2c6894b92c75306d1d4f5db337d2fe51cb9500a911e5c347ea6.jpg)

![78d9dea279642fb30e5877f53cd9c8a95a1ef87afad5cf2b176cb68e60f1c000.jpg](../iclr_results/1578_Drop-Upcycling_%20Training%20Sparse%20Mixture%20of%20Experts%20with%20Partial%20Re-initialization/tables/78d9dea279642fb30e5877f53cd9c8a95a1ef87afad5cf2b176cb68e60f1c000.jpg)

![81c1645b59e56cfe6b105f087f8e932a622b12c35118c2b0883bd7c08a1fd171.jpg](../iclr_results/1578_Drop-Upcycling_%20Training%20Sparse%20Mixture%20of%20Experts%20with%20Partial%20Re-initialization/tables/81c1645b59e56cfe6b105f087f8e932a622b12c35118c2b0883bd7c08a1fd171.jpg)

![b901d27e27dddea6af3eed395b0b7c4482947a8f1cda7559664046ce6f50d31b.jpg](../iclr_results/1578_Drop-Upcycling_%20Training%20Sparse%20Mixture%20of%20Experts%20with%20Partial%20Re-initialization/tables/b901d27e27dddea6af3eed395b0b7c4482947a8f1cda7559664046ce6f50d31b.jpg)

![d60b319133850fc2a3c5a054480d4636b90111e3ef28330268b2ad45f86251d0.jpg](../iclr_results/1578_Drop-Upcycling_%20Training%20Sparse%20Mixture%20of%20Experts%20with%20Partial%20Re-initialization/tables/d60b319133850fc2a3c5a054480d4636b90111e3ef28330268b2ad45f86251d0.jpg)

## Beyond Surface Structure: A Causal Assessment of LLMs' Comprehension ability


### Images

![15fc28ed29d9a6da973ea3d9c954db7b59920e4346fd5636b6c5d354373fd3a9.jpg](../iclr_results/1579_Beyond%20Surface%20Structure_%20A%20Causal%20Assessment%20of%20LLMs%27%20Comprehension%20ability/images/15fc28ed29d9a6da973ea3d9c954db7b59920e4346fd5636b6c5d354373fd3a9.jpg)

![19f1f8c61cf7712f39145b188204b75c4fbb2c2ee3df76f9e3bce08e7b952235.jpg](../iclr_results/1579_Beyond%20Surface%20Structure_%20A%20Causal%20Assessment%20of%20LLMs%27%20Comprehension%20ability/images/19f1f8c61cf7712f39145b188204b75c4fbb2c2ee3df76f9e3bce08e7b952235.jpg)

![259d24117503de298febf71850be660eff2844b6979af9d78e36e5fdb5888862.jpg](../iclr_results/1579_Beyond%20Surface%20Structure_%20A%20Causal%20Assessment%20of%20LLMs%27%20Comprehension%20ability/images/259d24117503de298febf71850be660eff2844b6979af9d78e36e5fdb5888862.jpg)

![25f7a2321eae50639c94f186f227f9c609b88b3bd84929d265bf4d98f0b5ceb2.jpg](../iclr_results/1579_Beyond%20Surface%20Structure_%20A%20Causal%20Assessment%20of%20LLMs%27%20Comprehension%20ability/images/25f7a2321eae50639c94f186f227f9c609b88b3bd84929d265bf4d98f0b5ceb2.jpg)

![2fc10fd396a1a5612ffaee3ab8021f690809e400bf3ff8b6f1c5bfa9104450cd.jpg](../iclr_results/1579_Beyond%20Surface%20Structure_%20A%20Causal%20Assessment%20of%20LLMs%27%20Comprehension%20ability/images/2fc10fd396a1a5612ffaee3ab8021f690809e400bf3ff8b6f1c5bfa9104450cd.jpg)

![3b444766cffeb5229fd4503a8b418fc5448ae6220149634b0d7c93d81005509e.jpg](../iclr_results/1579_Beyond%20Surface%20Structure_%20A%20Causal%20Assessment%20of%20LLMs%27%20Comprehension%20ability/images/3b444766cffeb5229fd4503a8b418fc5448ae6220149634b0d7c93d81005509e.jpg)

![530d2153a52291fbda5489739eab1265347824816dc2415fb07f81ecef9a24bb.jpg](../iclr_results/1579_Beyond%20Surface%20Structure_%20A%20Causal%20Assessment%20of%20LLMs%27%20Comprehension%20ability/images/530d2153a52291fbda5489739eab1265347824816dc2415fb07f81ecef9a24bb.jpg)

![6eb0881f3088068d6c777a536e1cf0794fa9611eba5c5c06965a2cfd3c32afab.jpg](../iclr_results/1579_Beyond%20Surface%20Structure_%20A%20Causal%20Assessment%20of%20LLMs%27%20Comprehension%20ability/images/6eb0881f3088068d6c777a536e1cf0794fa9611eba5c5c06965a2cfd3c32afab.jpg)

![82f262ddc515aeae5bd3d0470c7bfe88efa19c712fd2a3ddaffbf96c42a2d14c.jpg](../iclr_results/1579_Beyond%20Surface%20Structure_%20A%20Causal%20Assessment%20of%20LLMs%27%20Comprehension%20ability/images/82f262ddc515aeae5bd3d0470c7bfe88efa19c712fd2a3ddaffbf96c42a2d14c.jpg)

![9e0df4d42d909c77b572663394642da10e1bac37e0b0e7faf6f836a5d847cabe.jpg](../iclr_results/1579_Beyond%20Surface%20Structure_%20A%20Causal%20Assessment%20of%20LLMs%27%20Comprehension%20ability/images/9e0df4d42d909c77b572663394642da10e1bac37e0b0e7faf6f836a5d847cabe.jpg)

![b3aeb115a18bb7e9ce8a661a4bd6d4aafb5c755a98fdc1a088fb893108258bf8.jpg](../iclr_results/1579_Beyond%20Surface%20Structure_%20A%20Causal%20Assessment%20of%20LLMs%27%20Comprehension%20ability/images/b3aeb115a18bb7e9ce8a661a4bd6d4aafb5c755a98fdc1a088fb893108258bf8.jpg)

![d58c1c4f9f1334f0bc8f5b9f7bbbcbf111f6906c38ea625ed5a534fbf88a8003.jpg](../iclr_results/1579_Beyond%20Surface%20Structure_%20A%20Causal%20Assessment%20of%20LLMs%27%20Comprehension%20ability/images/d58c1c4f9f1334f0bc8f5b9f7bbbcbf111f6906c38ea625ed5a534fbf88a8003.jpg)

![d92e58fc56001a08aa49d6bde8636857504a838ae512d7db408f060b5a39f7b7.jpg](../iclr_results/1579_Beyond%20Surface%20Structure_%20A%20Causal%20Assessment%20of%20LLMs%27%20Comprehension%20ability/images/d92e58fc56001a08aa49d6bde8636857504a838ae512d7db408f060b5a39f7b7.jpg)

### Tables

![00c742ad6d9daa6f6612d97976ff20a6fe0f7f79ac4a4d3283b9ff0fc898edf0.jpg](../iclr_results/1579_Beyond%20Surface%20Structure_%20A%20Causal%20Assessment%20of%20LLMs%27%20Comprehension%20ability/tables/00c742ad6d9daa6f6612d97976ff20a6fe0f7f79ac4a4d3283b9ff0fc898edf0.jpg)

![1592c9f0f44f4128367ff14e7175ac574bc4a0feb33ee7b1d85e4ce8a5697e72.jpg](../iclr_results/1579_Beyond%20Surface%20Structure_%20A%20Causal%20Assessment%20of%20LLMs%27%20Comprehension%20ability/tables/1592c9f0f44f4128367ff14e7175ac574bc4a0feb33ee7b1d85e4ce8a5697e72.jpg)

![3f151cb3ba8b42295dd616020c54e0b2296b271b63ea9b8be1b562d6aa367f0a.jpg](../iclr_results/1579_Beyond%20Surface%20Structure_%20A%20Causal%20Assessment%20of%20LLMs%27%20Comprehension%20ability/tables/3f151cb3ba8b42295dd616020c54e0b2296b271b63ea9b8be1b562d6aa367f0a.jpg)

![5c41db1ff8ef2d7a58e983a76b0f1742fe6eb091ce5846273f61fc4ce8f9a83f.jpg](../iclr_results/1579_Beyond%20Surface%20Structure_%20A%20Causal%20Assessment%20of%20LLMs%27%20Comprehension%20ability/tables/5c41db1ff8ef2d7a58e983a76b0f1742fe6eb091ce5846273f61fc4ce8f9a83f.jpg)

![903e55350de1db0c548487a7a9d62f11b4abfd4842a0f181c6371653a66aa8c6.jpg](../iclr_results/1579_Beyond%20Surface%20Structure_%20A%20Causal%20Assessment%20of%20LLMs%27%20Comprehension%20ability/tables/903e55350de1db0c548487a7a9d62f11b4abfd4842a0f181c6371653a66aa8c6.jpg)

![add1024d0da8b09ab43f5d200de9b6a6312fed28a75d31f366b0ac92d6527994.jpg](../iclr_results/1579_Beyond%20Surface%20Structure_%20A%20Causal%20Assessment%20of%20LLMs%27%20Comprehension%20ability/tables/add1024d0da8b09ab43f5d200de9b6a6312fed28a75d31f366b0ac92d6527994.jpg)

![d148d57666f4484d552e1e9b8c20408b971b15d551d694beadebc48fc9704e09.jpg](../iclr_results/1579_Beyond%20Surface%20Structure_%20A%20Causal%20Assessment%20of%20LLMs%27%20Comprehension%20ability/tables/d148d57666f4484d552e1e9b8c20408b971b15d551d694beadebc48fc9704e09.jpg)

![d9022cbba80a46c942dc63cb80038a9142cabd78eac3f2314c96a72f5fc2fd68.jpg](../iclr_results/1579_Beyond%20Surface%20Structure_%20A%20Causal%20Assessment%20of%20LLMs%27%20Comprehension%20ability/tables/d9022cbba80a46c942dc63cb80038a9142cabd78eac3f2314c96a72f5fc2fd68.jpg)

![e200dd5af916e18fcc772de001870fb0d21acb8583c8e585b5d30c0da5c9e3b7.jpg](../iclr_results/1579_Beyond%20Surface%20Structure_%20A%20Causal%20Assessment%20of%20LLMs%27%20Comprehension%20ability/tables/e200dd5af916e18fcc772de001870fb0d21acb8583c8e585b5d30c0da5c9e3b7.jpg)

## Unify ML4TSP: Drawing Methodological Principles for TSP and Beyond from Streamlined Design Space of Learning and Search


### Images

![0b5deea059d07d5361e08caf6203348946243ce2739c633c73fa4e757774c972.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/images/0b5deea059d07d5361e08caf6203348946243ce2739c633c73fa4e757774c972.jpg)

![17381cc778347072a48933103df2d9d39756f816bb1eb5e324c71fed70efd556.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/images/17381cc778347072a48933103df2d9d39756f816bb1eb5e324c71fed70efd556.jpg)

![1738ab37ce1a2165ad0c58074ea7ce65263a3ee666d4e140ce2539e5f0d77dd0.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/images/1738ab37ce1a2165ad0c58074ea7ce65263a3ee666d4e140ce2539e5f0d77dd0.jpg)

![18385b0128d975d7fb0956b150e7e42174412094dc7d3b4e66e40b0628f607e4.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/images/18385b0128d975d7fb0956b150e7e42174412094dc7d3b4e66e40b0628f607e4.jpg)

![3f761e6dcc39380ecab949a234306dc69c8197a2607af9c1518a206dc0c93f0c.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/images/3f761e6dcc39380ecab949a234306dc69c8197a2607af9c1518a206dc0c93f0c.jpg)

![70dce3b8c9ff1a994033f5466db0613d99769aa6fe6046172390849009f0c450.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/images/70dce3b8c9ff1a994033f5466db0613d99769aa6fe6046172390849009f0c450.jpg)

![88ea284347e39f0a9f50c6626fc865622adc3108eeacc0ec409142e2113a6f80.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/images/88ea284347e39f0a9f50c6626fc865622adc3108eeacc0ec409142e2113a6f80.jpg)

![88f6d45d3d14ce3d102cfcc3147870f75605db9b3880e947bafabfc56ba4c69b.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/images/88f6d45d3d14ce3d102cfcc3147870f75605db9b3880e947bafabfc56ba4c69b.jpg)

![c99a73e1906193a453b3e5a171f6ac717a1f6165a75230c2b39351638017579a.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/images/c99a73e1906193a453b3e5a171f6ac717a1f6165a75230c2b39351638017579a.jpg)

![e6cce4f385ae2b3f8ab0ed493739bd66f13e0cb7414f03079852d3767a8ba3e9.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/images/e6cce4f385ae2b3f8ab0ed493739bd66f13e0cb7414f03079852d3767a8ba3e9.jpg)

### Tables

![029a188a3ae300635f2c2721cd5dc51db7d2ec3f33936bd6748eb5adf6830685.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/tables/029a188a3ae300635f2c2721cd5dc51db7d2ec3f33936bd6748eb5adf6830685.jpg)

![0e4c701460eb9eadfc82031412f39e7bde9b9688ee563f520d2e2a41eb0e029d.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/tables/0e4c701460eb9eadfc82031412f39e7bde9b9688ee563f520d2e2a41eb0e029d.jpg)

![1805a3f6b6495b701503228c56b8ec2d18eff25dd6d616fd9c94c12af6c8648b.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/tables/1805a3f6b6495b701503228c56b8ec2d18eff25dd6d616fd9c94c12af6c8648b.jpg)

![1b52e0e84e75e04ebcac6d14d1043d9b337f645dc5c992831b48431a90956b3c.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/tables/1b52e0e84e75e04ebcac6d14d1043d9b337f645dc5c992831b48431a90956b3c.jpg)

![2d2312943a428e8a80d31a4c73c573bdec3ed189a773f02fc824ce6b6b1231a6.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/tables/2d2312943a428e8a80d31a4c73c573bdec3ed189a773f02fc824ce6b6b1231a6.jpg)

![3314221fe6223803979a18020027c2c39ee0cd658d1ff2484cb1a2799bbe9bc4.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/tables/3314221fe6223803979a18020027c2c39ee0cd658d1ff2484cb1a2799bbe9bc4.jpg)

![4fca2a27779290500f804ea34f33a8a7a44193b5ade69ad66f2bcae79d80d214.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/tables/4fca2a27779290500f804ea34f33a8a7a44193b5ade69ad66f2bcae79d80d214.jpg)

![50e9b423678897ddd9647ebd4bdfb9407ec97e4f9c83551d01b8ad0580643742.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/tables/50e9b423678897ddd9647ebd4bdfb9407ec97e4f9c83551d01b8ad0580643742.jpg)

![582e4d9b2c8747e904d6b1cb25a844e1d8c600869f7cf3e19b6ed2b2097b719a.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/tables/582e4d9b2c8747e904d6b1cb25a844e1d8c600869f7cf3e19b6ed2b2097b719a.jpg)

![589d5ecebc3fe7f27613b4912e9d30a8f595b4e9338ab73054f8e76964212a37.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/tables/589d5ecebc3fe7f27613b4912e9d30a8f595b4e9338ab73054f8e76964212a37.jpg)

![5d742f24d09359bb2b14ab0b897c664d107da533caba33093e9a28627673ac50.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/tables/5d742f24d09359bb2b14ab0b897c664d107da533caba33093e9a28627673ac50.jpg)

![79390ace24588cf9c84cca4fa35f75110e9652cf766c3153d8c712cbc95a0c83.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/tables/79390ace24588cf9c84cca4fa35f75110e9652cf766c3153d8c712cbc95a0c83.jpg)

![9debfa91dfa97fe7f3bb569549d6edfd4d11a3a29331ed3c2d074a936b0e6a04.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/tables/9debfa91dfa97fe7f3bb569549d6edfd4d11a3a29331ed3c2d074a936b0e6a04.jpg)

![e00e4b4dc439ce0a49de279fa2d300803a3da9f27af59b719c064a20d7c944f8.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/tables/e00e4b4dc439ce0a49de279fa2d300803a3da9f27af59b719c064a20d7c944f8.jpg)

![f011474f3e0b6ffb3c1f4dee059a60840b3a90834c03a317d87cab4ec34b617b.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/tables/f011474f3e0b6ffb3c1f4dee059a60840b3a90834c03a317d87cab4ec34b617b.jpg)

![f86e73369c39f96aaa1ba3995d48c34c75c52a11561c2bdb58160d9c4b8afd38.jpg](../iclr_results/1580_Unify%20ML4TSP_%20Drawing%20Methodological%20Principles%20for%20TSP%20and%20Beyond%20from%20Streamlined%20Design%20Space%20of%20/tables/f86e73369c39f96aaa1ba3995d48c34c75c52a11561c2bdb58160d9c4b8afd38.jpg)

## Minimal Variance Model Aggregation: A principled, non-intrusive, and versatile integration of black box models


### Images

![009254483c9d35dfe4bc4de4f360526ae9f0603aaebfe87dd90a648b91f86642.jpg](../iclr_results/1581_Minimal%20Variance%20Model%20Aggregation_%20A%20principled%2C%20non-intrusive%2C%20and%20versatile%20integration%20of%20black%20/images/009254483c9d35dfe4bc4de4f360526ae9f0603aaebfe87dd90a648b91f86642.jpg)

![034aed3c5674285e838e0f35f9fe0dc919df25397d32baf721fde2f45913bbe2.jpg](../iclr_results/1581_Minimal%20Variance%20Model%20Aggregation_%20A%20principled%2C%20non-intrusive%2C%20and%20versatile%20integration%20of%20black%20/images/034aed3c5674285e838e0f35f9fe0dc919df25397d32baf721fde2f45913bbe2.jpg)

![1eedf840bb16492ce3c62e7b34269022535072bec23a877e38729d45a0365a7f.jpg](../iclr_results/1581_Minimal%20Variance%20Model%20Aggregation_%20A%20principled%2C%20non-intrusive%2C%20and%20versatile%20integration%20of%20black%20/images/1eedf840bb16492ce3c62e7b34269022535072bec23a877e38729d45a0365a7f.jpg)

![203a6bc174f76e091899caea8ffcd673a284f7e9f36994e650ebdb5cd5813e96.jpg](../iclr_results/1581_Minimal%20Variance%20Model%20Aggregation_%20A%20principled%2C%20non-intrusive%2C%20and%20versatile%20integration%20of%20black%20/images/203a6bc174f76e091899caea8ffcd673a284f7e9f36994e650ebdb5cd5813e96.jpg)

![43799ffc174cd9ae7f950dc7b1a69595597fd5bd6a746cad9e0cd01c6e2532d4.jpg](../iclr_results/1581_Minimal%20Variance%20Model%20Aggregation_%20A%20principled%2C%20non-intrusive%2C%20and%20versatile%20integration%20of%20black%20/images/43799ffc174cd9ae7f950dc7b1a69595597fd5bd6a746cad9e0cd01c6e2532d4.jpg)

![64239679d4ebb21cf5d0f1966781c9fd63046b10c58c5359e2f66bae1360fe37.jpg](../iclr_results/1581_Minimal%20Variance%20Model%20Aggregation_%20A%20principled%2C%20non-intrusive%2C%20and%20versatile%20integration%20of%20black%20/images/64239679d4ebb21cf5d0f1966781c9fd63046b10c58c5359e2f66bae1360fe37.jpg)

![ae4004b5f10fc68c03528921c7ef4fac1b20a7928154f76d660b7167ad379e6b.jpg](../iclr_results/1581_Minimal%20Variance%20Model%20Aggregation_%20A%20principled%2C%20non-intrusive%2C%20and%20versatile%20integration%20of%20black%20/images/ae4004b5f10fc68c03528921c7ef4fac1b20a7928154f76d660b7167ad379e6b.jpg)

![c26e52aa7b484a3a969db0314182f7f339b1b195af9fab82bbee08f3971074b9.jpg](../iclr_results/1581_Minimal%20Variance%20Model%20Aggregation_%20A%20principled%2C%20non-intrusive%2C%20and%20versatile%20integration%20of%20black%20/images/c26e52aa7b484a3a969db0314182f7f339b1b195af9fab82bbee08f3971074b9.jpg)

### Tables

![9c792af492214a09276b977663fd2d0fd32a555f5376842d73ab66fa9ab5fbd3.jpg](../iclr_results/1581_Minimal%20Variance%20Model%20Aggregation_%20A%20principled%2C%20non-intrusive%2C%20and%20versatile%20integration%20of%20black%20/tables/9c792af492214a09276b977663fd2d0fd32a555f5376842d73ab66fa9ab5fbd3.jpg)

![af13a0f6beb4996159840f4c892d7c888267c0ee102d3e2a463ee0eda057d7f2.jpg](../iclr_results/1581_Minimal%20Variance%20Model%20Aggregation_%20A%20principled%2C%20non-intrusive%2C%20and%20versatile%20integration%20of%20black%20/tables/af13a0f6beb4996159840f4c892d7c888267c0ee102d3e2a463ee0eda057d7f2.jpg)

## Rethinking Visual Counterfactual Explanations Through Region Constraint


### Images

![37d32816e1a7440c43f24de2c07b872f4a47690d88ce11022cc4a11a193a516a.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/37d32816e1a7440c43f24de2c07b872f4a47690d88ce11022cc4a11a193a516a.jpg)

![3ac665088d6cdb058fdd0b620b19b089c37ef7ecffe7667a05c3af5472c9d7e4.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/3ac665088d6cdb058fdd0b620b19b089c37ef7ecffe7667a05c3af5472c9d7e4.jpg)

![44def85c95572e6b4ac043e1f04d66a961847b998f5b48d544e51a7566a46760.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/44def85c95572e6b4ac043e1f04d66a961847b998f5b48d544e51a7566a46760.jpg)

![547ae3b39304780266413dd13bd7030fa59cd901c53f87e69f1f4236d0f668f8.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/547ae3b39304780266413dd13bd7030fa59cd901c53f87e69f1f4236d0f668f8.jpg)

![5af5b532d01390d3def8f2db4969bea2f571fea4dd25f6cfb908950c0df363b1.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/5af5b532d01390d3def8f2db4969bea2f571fea4dd25f6cfb908950c0df363b1.jpg)

![68760491539b884ca42c61face69b5945c00e23b062231532e490df56dae5b57.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/68760491539b884ca42c61face69b5945c00e23b062231532e490df56dae5b57.jpg)

![6bbd0e37ec12611ebf90add32249ce0da896d990ded7f8e0b092319689e17de4.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/6bbd0e37ec12611ebf90add32249ce0da896d990ded7f8e0b092319689e17de4.jpg)

![6fa0b1006b6195736da14ae61ac4d1d9916f86bfc73e98d469ad6336a4901d38.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/6fa0b1006b6195736da14ae61ac4d1d9916f86bfc73e98d469ad6336a4901d38.jpg)

![78a96b242abb297b28b088903c9a9243c113cbc94d061b0920a533de45e251b3.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/78a96b242abb297b28b088903c9a9243c113cbc94d061b0920a533de45e251b3.jpg)

![79065622721d5ad41d5067f326a8c57606437cbdd97b2e7c607390f627b4e745.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/79065622721d5ad41d5067f326a8c57606437cbdd97b2e7c607390f627b4e745.jpg)

![7c6b2358d34a2b35ec8638acc3cfb28d8dab3baad6ce6c2d856c33ff8a0cb901.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/7c6b2358d34a2b35ec8638acc3cfb28d8dab3baad6ce6c2d856c33ff8a0cb901.jpg)

![8253b8f1e452b2f8705d30f378fd3e8dba9f649efabd23bb2c4428230aea0d7c.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/8253b8f1e452b2f8705d30f378fd3e8dba9f649efabd23bb2c4428230aea0d7c.jpg)

![88d25885357204cc9aa089f5b14d20cbead27106f1a1926f753c8718dc5bf97d.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/88d25885357204cc9aa089f5b14d20cbead27106f1a1926f753c8718dc5bf97d.jpg)

![91f70169051b678fb2a2b2a390999fab1dd9d2ae8c7b60b9f40d91fd35e53ab8.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/91f70169051b678fb2a2b2a390999fab1dd9d2ae8c7b60b9f40d91fd35e53ab8.jpg)

![b51eea492d9e4be42b368b6deb6c0c51592a93cf36bf8714d19885d9f7cd4240.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/b51eea492d9e4be42b368b6deb6c0c51592a93cf36bf8714d19885d9f7cd4240.jpg)

![bbc934340f27d3e4bea6c227abbd1266f3d32a1fd3f587540db2e4ce76060610.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/bbc934340f27d3e4bea6c227abbd1266f3d32a1fd3f587540db2e4ce76060610.jpg)

![be18998d610f9e695383c161b5d1bdbe4282a8c5da78e9879a64cc040e8edb4f.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/be18998d610f9e695383c161b5d1bdbe4282a8c5da78e9879a64cc040e8edb4f.jpg)

![bf2306da1f83556a2e4ba64c3d032ddb13c0182f30171e8d8dc358d6ce352a8e.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/bf2306da1f83556a2e4ba64c3d032ddb13c0182f30171e8d8dc358d6ce352a8e.jpg)

![c58a70b140f5a75192c99bdf4f0d12290b09e6d374a6ec5a3896fb09420f2cd0.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/c58a70b140f5a75192c99bdf4f0d12290b09e6d374a6ec5a3896fb09420f2cd0.jpg)

![c62e0b56f1237d9adc8bcfdf4b60ddcf460f2da7dfba26485e39141a4b9adab3.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/c62e0b56f1237d9adc8bcfdf4b60ddcf460f2da7dfba26485e39141a4b9adab3.jpg)

![c768b826ff93cd7f5df82ba694c0460f931adda7f5834cc65ede1d8450b48b07.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/c768b826ff93cd7f5df82ba694c0460f931adda7f5834cc65ede1d8450b48b07.jpg)

![c7f96d7340da19643f245fb447261f99bc7c96e328dd7349f8efeecfb15704c5.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/c7f96d7340da19643f245fb447261f99bc7c96e328dd7349f8efeecfb15704c5.jpg)

![c9f2b7a84f7a75a25c6197e5e56f40e65ee51908ce47de667fc492e5b4699da2.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/c9f2b7a84f7a75a25c6197e5e56f40e65ee51908ce47de667fc492e5b4699da2.jpg)

![cfa5c65c00f52813b784a7bca4cda8cfaec5d644548171d2b6b769671e37e3b8.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/cfa5c65c00f52813b784a7bca4cda8cfaec5d644548171d2b6b769671e37e3b8.jpg)

![d9a3d93f1598b6d7ef1bd68454e8508c54fafcb05420c0deea44fb12bfe03c32.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/d9a3d93f1598b6d7ef1bd68454e8508c54fafcb05420c0deea44fb12bfe03c32.jpg)

![e0ccac5c3e5989efe6601cd69bd6184e1e51cd6bfdb8e13864461feaacbcdea5.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/e0ccac5c3e5989efe6601cd69bd6184e1e51cd6bfdb8e13864461feaacbcdea5.jpg)

![e96d35c465dc65a70a104e7824a9a3463f6936b7b927cc54de0320fa4e45e6d3.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/e96d35c465dc65a70a104e7824a9a3463f6936b7b927cc54de0320fa4e45e6d3.jpg)

![f05524596f3b18fd398571219c33e398b24735a0fce99369b38114f4c14b5eb9.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/f05524596f3b18fd398571219c33e398b24735a0fce99369b38114f4c14b5eb9.jpg)

![f354baced89dcbb23f454303687342b6b36b6b962ebc94b711732f09e3f78971.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/f354baced89dcbb23f454303687342b6b36b6b962ebc94b711732f09e3f78971.jpg)

![f740e40af1e57e81242a78fd3930e5957b440cddbeef9cceb2796d7b3a3673be.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/f740e40af1e57e81242a78fd3930e5957b440cddbeef9cceb2796d7b3a3673be.jpg)

![fa20f008b35ec81cb6a01bdf932b210aa5e5d6439f82021e9e2277577e318078.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/fa20f008b35ec81cb6a01bdf932b210aa5e5d6439f82021e9e2277577e318078.jpg)

![fbcf865609ec553b650d82c3ff25a952247b950a1d34d98dcee026d29bc937c0.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/fbcf865609ec553b650d82c3ff25a952247b950a1d34d98dcee026d29bc937c0.jpg)

![fd496e25baf5066727eec7c2cadcae4125810d4aefe4a11e0069b3b02947aac0.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/fd496e25baf5066727eec7c2cadcae4125810d4aefe4a11e0069b3b02947aac0.jpg)

![ffe1e7bd973f132cdec6c08217ca99f99b680155b4e2c9c9904298e7d40b4207.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/images/ffe1e7bd973f132cdec6c08217ca99f99b680155b4e2c9c9904298e7d40b4207.jpg)

### Tables

![05116f311ba0059a6afa2ff03c7c7be3ab906602da85188bdeec5f1423fa63b8.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/tables/05116f311ba0059a6afa2ff03c7c7be3ab906602da85188bdeec5f1423fa63b8.jpg)

![134547b4ef8c223656b32507e0b6cdc117ccb99bd61795076c8c589340b9ebd8.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/tables/134547b4ef8c223656b32507e0b6cdc117ccb99bd61795076c8c589340b9ebd8.jpg)

![1e4617c252cfc6d4eabb6341ad1df2ed74d1be033e531000cd0e4d86c89d0f6d.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/tables/1e4617c252cfc6d4eabb6341ad1df2ed74d1be033e531000cd0e4d86c89d0f6d.jpg)

![217fb0ede7a6dbbd4f7beeb0570472040084474b5c41368e077612672f8f6bd6.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/tables/217fb0ede7a6dbbd4f7beeb0570472040084474b5c41368e077612672f8f6bd6.jpg)

![23cdcc7c1f7426fb945f8b928ac193392aa64d929fb566ae4e42a2a7ee44b202.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/tables/23cdcc7c1f7426fb945f8b928ac193392aa64d929fb566ae4e42a2a7ee44b202.jpg)

![2dc632d05b5e92c6750e82833dd58ff419552506df0b7ff0461b216a6d514796.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/tables/2dc632d05b5e92c6750e82833dd58ff419552506df0b7ff0461b216a6d514796.jpg)

![30ebeb13ec375d3116c4244eef9aaf9e468e0415c5d1995e9d2294474750bbd4.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/tables/30ebeb13ec375d3116c4244eef9aaf9e468e0415c5d1995e9d2294474750bbd4.jpg)

![3f35a565da8305502d872c87cef7bb00cfa5031782cc1c442e156756a9731cfa.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/tables/3f35a565da8305502d872c87cef7bb00cfa5031782cc1c442e156756a9731cfa.jpg)

![586b32d416fc852e29e553399b8b91c3afa5ace659f3293f64dd24b725060de4.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/tables/586b32d416fc852e29e553399b8b91c3afa5ace659f3293f64dd24b725060de4.jpg)

![b48654ea3e9af8894e264c03940e88ac6df8bc3a5c40d41ee3224fcc50e90821.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/tables/b48654ea3e9af8894e264c03940e88ac6df8bc3a5c40d41ee3224fcc50e90821.jpg)

![ca39d6df1c35798525d162a616dea73d8ede9218596612ade28d9f335cac36f0.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/tables/ca39d6df1c35798525d162a616dea73d8ede9218596612ade28d9f335cac36f0.jpg)

![d4bcf28bd7040b9314b7dcc8f478121165f03df8e6850bea3fdad5e42d89c48a.jpg](../iclr_results/1582_Rethinking%20Visual%20Counterfactual%20Explanations%20Through%20Region%20Constraint/tables/d4bcf28bd7040b9314b7dcc8f478121165f03df8e6850bea3fdad5e42d89c48a.jpg)

## A Conditional Independence Test in the Presence of Discretization


### Images

![212da1831e1431635e8da690fcdc7b37d4f5c3c118aabd36f127806fb84a9432.jpg](../iclr_results/1583_A%20Conditional%20Independence%20Test%20in%20the%20Presence%20of%20Discretization/images/212da1831e1431635e8da690fcdc7b37d4f5c3c118aabd36f127806fb84a9432.jpg)

![2d70f060b8403f40761766bd12f71313308f214c2730558fac1a740f0ecdc4df.jpg](../iclr_results/1583_A%20Conditional%20Independence%20Test%20in%20the%20Presence%20of%20Discretization/images/2d70f060b8403f40761766bd12f71313308f214c2730558fac1a740f0ecdc4df.jpg)

![363c6c30bc1d4edf423364032fdd5629346a349cf0ae9aa7a9e18cc26b8a3566.jpg](../iclr_results/1583_A%20Conditional%20Independence%20Test%20in%20the%20Presence%20of%20Discretization/images/363c6c30bc1d4edf423364032fdd5629346a349cf0ae9aa7a9e18cc26b8a3566.jpg)

![6f2edbf0045d0f692b803517b72538abaf07dc3f2f600e4f7b93e7c811a64d9b.jpg](../iclr_results/1583_A%20Conditional%20Independence%20Test%20in%20the%20Presence%20of%20Discretization/images/6f2edbf0045d0f692b803517b72538abaf07dc3f2f600e4f7b93e7c811a64d9b.jpg)

![78880a16e49b1c8575de233b0518b9d51c0a8e048f62e3fc1708628fd99a3b71.jpg](../iclr_results/1583_A%20Conditional%20Independence%20Test%20in%20the%20Presence%20of%20Discretization/images/78880a16e49b1c8575de233b0518b9d51c0a8e048f62e3fc1708628fd99a3b71.jpg)

![b773f44b4c1ab3eaad71f77ea40738079f04d968e3fd9dffca5647369c67b8d5.jpg](../iclr_results/1583_A%20Conditional%20Independence%20Test%20in%20the%20Presence%20of%20Discretization/images/b773f44b4c1ab3eaad71f77ea40738079f04d968e3fd9dffca5647369c67b8d5.jpg)

![bce1379aec3b6ce75825f72da4ffa58d664d06c5e0ec07b4f93d050f85796edf.jpg](../iclr_results/1583_A%20Conditional%20Independence%20Test%20in%20the%20Presence%20of%20Discretization/images/bce1379aec3b6ce75825f72da4ffa58d664d06c5e0ec07b4f93d050f85796edf.jpg)

![cdc2b1e9b954553127f452ff6e16e10a8d8af13364ee7ce96a56336813e93834.jpg](../iclr_results/1583_A%20Conditional%20Independence%20Test%20in%20the%20Presence%20of%20Discretization/images/cdc2b1e9b954553127f452ff6e16e10a8d8af13364ee7ce96a56336813e93834.jpg)

![d1e826540d255ee173d48bc7080ee442820e089b599c04ffd13871a76c6c4dc3.jpg](../iclr_results/1583_A%20Conditional%20Independence%20Test%20in%20the%20Presence%20of%20Discretization/images/d1e826540d255ee173d48bc7080ee442820e089b599c04ffd13871a76c6c4dc3.jpg)

### Tables

![11ca2355d5fde93ef2e2c6e603f24e17adf38ae0135337bd773d79ca5a8435dc.jpg](../iclr_results/1583_A%20Conditional%20Independence%20Test%20in%20the%20Presence%20of%20Discretization/tables/11ca2355d5fde93ef2e2c6e603f24e17adf38ae0135337bd773d79ca5a8435dc.jpg)

## Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression


### Images

![158624f0a920ef308e4bb6da8d0a60308175fb8ecbab6e2e2f0191e5f5f3f2b6.jpg](../iclr_results/1584_Basis%20Sharing_%20Cross-Layer%20Parameter%20Sharing%20for%20Large%20Language%20Model%20Compression/images/158624f0a920ef308e4bb6da8d0a60308175fb8ecbab6e2e2f0191e5f5f3f2b6.jpg)

![3f9094af3402dd4cf8ae22eea47468066a0ef2ef6e97fb988d46895ce599014b.jpg](../iclr_results/1584_Basis%20Sharing_%20Cross-Layer%20Parameter%20Sharing%20for%20Large%20Language%20Model%20Compression/images/3f9094af3402dd4cf8ae22eea47468066a0ef2ef6e97fb988d46895ce599014b.jpg)

![4c7c42b03d3f8e786ed357d77f55e9ec092c62a37433a23be2826486a35ce07e.jpg](../iclr_results/1584_Basis%20Sharing_%20Cross-Layer%20Parameter%20Sharing%20for%20Large%20Language%20Model%20Compression/images/4c7c42b03d3f8e786ed357d77f55e9ec092c62a37433a23be2826486a35ce07e.jpg)

![62cfaa02de551c9990c916ca947b1a9bf7591a78769e2d68fd4e89e025867df0.jpg](../iclr_results/1584_Basis%20Sharing_%20Cross-Layer%20Parameter%20Sharing%20for%20Large%20Language%20Model%20Compression/images/62cfaa02de551c9990c916ca947b1a9bf7591a78769e2d68fd4e89e025867df0.jpg)

![8376372b43e3fcd5c5ab9b03a73109c791f22a732156365f6c51dc05b83c0849.jpg](../iclr_results/1584_Basis%20Sharing_%20Cross-Layer%20Parameter%20Sharing%20for%20Large%20Language%20Model%20Compression/images/8376372b43e3fcd5c5ab9b03a73109c791f22a732156365f6c51dc05b83c0849.jpg)

![aa65f5d689f569a29493c6bba96db9e24f4d34192399e441bb608749eff604db.jpg](../iclr_results/1584_Basis%20Sharing_%20Cross-Layer%20Parameter%20Sharing%20for%20Large%20Language%20Model%20Compression/images/aa65f5d689f569a29493c6bba96db9e24f4d34192399e441bb608749eff604db.jpg)

![ac4ab00d9e9b53487a442e61a5e7a658caed89b0bcd27c8155a5bc4fd33abab6.jpg](../iclr_results/1584_Basis%20Sharing_%20Cross-Layer%20Parameter%20Sharing%20for%20Large%20Language%20Model%20Compression/images/ac4ab00d9e9b53487a442e61a5e7a658caed89b0bcd27c8155a5bc4fd33abab6.jpg)

![d0a8d671030bd0a35b679ef0e4541dda678529b342c0ff34ea13c74cf45866a2.jpg](../iclr_results/1584_Basis%20Sharing_%20Cross-Layer%20Parameter%20Sharing%20for%20Large%20Language%20Model%20Compression/images/d0a8d671030bd0a35b679ef0e4541dda678529b342c0ff34ea13c74cf45866a2.jpg)

### Tables

![1e928882699e988e5eb4d7d7f95e07b87f34099e438884caff32da4ecda602c3.jpg](../iclr_results/1584_Basis%20Sharing_%20Cross-Layer%20Parameter%20Sharing%20for%20Large%20Language%20Model%20Compression/tables/1e928882699e988e5eb4d7d7f95e07b87f34099e438884caff32da4ecda602c3.jpg)

![21009dea407cf2212262a61bf02adec923a39182091a126b206a5c8f70017d1c.jpg](../iclr_results/1584_Basis%20Sharing_%20Cross-Layer%20Parameter%20Sharing%20for%20Large%20Language%20Model%20Compression/tables/21009dea407cf2212262a61bf02adec923a39182091a126b206a5c8f70017d1c.jpg)

![38f627d87c5582fcd3ed19bb9729fc48faea7377880e53a5cbbc9907db48ba87.jpg](../iclr_results/1584_Basis%20Sharing_%20Cross-Layer%20Parameter%20Sharing%20for%20Large%20Language%20Model%20Compression/tables/38f627d87c5582fcd3ed19bb9729fc48faea7377880e53a5cbbc9907db48ba87.jpg)

![56322a93eeb00f99994709760fb63b55f743b2df1a797838754a7b8852ab4233.jpg](../iclr_results/1584_Basis%20Sharing_%20Cross-Layer%20Parameter%20Sharing%20for%20Large%20Language%20Model%20Compression/tables/56322a93eeb00f99994709760fb63b55f743b2df1a797838754a7b8852ab4233.jpg)

![5e3f11c38751b81d7d42cbe299956a6f0da476d6170e8b1d5df6e7e68263c070.jpg](../iclr_results/1584_Basis%20Sharing_%20Cross-Layer%20Parameter%20Sharing%20for%20Large%20Language%20Model%20Compression/tables/5e3f11c38751b81d7d42cbe299956a6f0da476d6170e8b1d5df6e7e68263c070.jpg)

![6c535d0528014167764697963eaf467353a1faa81a98aaeaffc33f3116ee3050.jpg](../iclr_results/1584_Basis%20Sharing_%20Cross-Layer%20Parameter%20Sharing%20for%20Large%20Language%20Model%20Compression/tables/6c535d0528014167764697963eaf467353a1faa81a98aaeaffc33f3116ee3050.jpg)

![79efba02d122625e6d3dffb9b295a8117c94193a34b9262ec727bbe35a6afab5.jpg](../iclr_results/1584_Basis%20Sharing_%20Cross-Layer%20Parameter%20Sharing%20for%20Large%20Language%20Model%20Compression/tables/79efba02d122625e6d3dffb9b295a8117c94193a34b9262ec727bbe35a6afab5.jpg)

![8987f4791372c85d6a62a087f135decc5e103a3860ba1652bb181ad4b8385b8b.jpg](../iclr_results/1584_Basis%20Sharing_%20Cross-Layer%20Parameter%20Sharing%20for%20Large%20Language%20Model%20Compression/tables/8987f4791372c85d6a62a087f135decc5e103a3860ba1652bb181ad4b8385b8b.jpg)

![b4d9b4057afba7ddc4f8cefca2c730056419392c518b539c4356ef4c486b1ee8.jpg](../iclr_results/1584_Basis%20Sharing_%20Cross-Layer%20Parameter%20Sharing%20for%20Large%20Language%20Model%20Compression/tables/b4d9b4057afba7ddc4f8cefca2c730056419392c518b539c4356ef4c486b1ee8.jpg)

![c46487bbba609589f86c6c64f080e6dc564d16ce6567d7ca1cee8cab85c7f803.jpg](../iclr_results/1584_Basis%20Sharing_%20Cross-Layer%20Parameter%20Sharing%20for%20Large%20Language%20Model%20Compression/tables/c46487bbba609589f86c6c64f080e6dc564d16ce6567d7ca1cee8cab85c7f803.jpg)

![d8b45adf87a1632395ecc6c30b0a0f63e0f7153ebc6c623e76b2e5fa918f4e3d.jpg](../iclr_results/1584_Basis%20Sharing_%20Cross-Layer%20Parameter%20Sharing%20for%20Large%20Language%20Model%20Compression/tables/d8b45adf87a1632395ecc6c30b0a0f63e0f7153ebc6c623e76b2e5fa918f4e3d.jpg)

![e81a590f9d7e153afd3b65a231db16858b37b2e6bc5390bb90f53f8ac53b78e1.jpg](../iclr_results/1584_Basis%20Sharing_%20Cross-Layer%20Parameter%20Sharing%20for%20Large%20Language%20Model%20Compression/tables/e81a590f9d7e153afd3b65a231db16858b37b2e6bc5390bb90f53f8ac53b78e1.jpg)

![f1bf0013f18289c47972dfb4ba1213e90e525c77909326d070b8bec1d22ea422.jpg](../iclr_results/1584_Basis%20Sharing_%20Cross-Layer%20Parameter%20Sharing%20for%20Large%20Language%20Model%20Compression/tables/f1bf0013f18289c47972dfb4ba1213e90e525c77909326d070b8bec1d22ea422.jpg)

## ClimaQA: An Automated Evaluation Framework for Climate Question Answering Models


### Images

![017063dcaa84695b2260363fa75138124b4fe06d28f5862098d7f15fd6dc8380.jpg](../iclr_results/1585_ClimaQA_%20An%20Automated%20Evaluation%20Framework%20for%20Climate%20Question%20Answering%20Models/images/017063dcaa84695b2260363fa75138124b4fe06d28f5862098d7f15fd6dc8380.jpg)

![64676b9b9c9a14c343cac652905529e56e7eeb0031d8d1eb36996bc145484c06.jpg](../iclr_results/1585_ClimaQA_%20An%20Automated%20Evaluation%20Framework%20for%20Climate%20Question%20Answering%20Models/images/64676b9b9c9a14c343cac652905529e56e7eeb0031d8d1eb36996bc145484c06.jpg)

![756f21c1a6be5f81d0d1690cfd68dbbd500aa82dbfa856de0e83fb54a3ac3d91.jpg](../iclr_results/1585_ClimaQA_%20An%20Automated%20Evaluation%20Framework%20for%20Climate%20Question%20Answering%20Models/images/756f21c1a6be5f81d0d1690cfd68dbbd500aa82dbfa856de0e83fb54a3ac3d91.jpg)

![a7552ee5448701d046ca34dedafd56ef362ff2b801829713d3cd1ca22235fa6f.jpg](../iclr_results/1585_ClimaQA_%20An%20Automated%20Evaluation%20Framework%20for%20Climate%20Question%20Answering%20Models/images/a7552ee5448701d046ca34dedafd56ef362ff2b801829713d3cd1ca22235fa6f.jpg)

![dca1f1c5025f4957fbf4239032b5d314e986cf69006db4afd086fffa0de67d2b.jpg](../iclr_results/1585_ClimaQA_%20An%20Automated%20Evaluation%20Framework%20for%20Climate%20Question%20Answering%20Models/images/dca1f1c5025f4957fbf4239032b5d314e986cf69006db4afd086fffa0de67d2b.jpg)

![df0274bc180d3d5ed19ac1455e5da61c5d59894e88b05149ffe5a1a0b96ca190.jpg](../iclr_results/1585_ClimaQA_%20An%20Automated%20Evaluation%20Framework%20for%20Climate%20Question%20Answering%20Models/images/df0274bc180d3d5ed19ac1455e5da61c5d59894e88b05149ffe5a1a0b96ca190.jpg)

### Tables

![07caf788b52fdd91b022dc987e7d6b7d9f871fbfb03e971124ee7e6c4aac56a8.jpg](../iclr_results/1585_ClimaQA_%20An%20Automated%20Evaluation%20Framework%20for%20Climate%20Question%20Answering%20Models/tables/07caf788b52fdd91b022dc987e7d6b7d9f871fbfb03e971124ee7e6c4aac56a8.jpg)

![0967e97a8d452318b73824c82197651fbed934579cbf87749afcc729e493aa72.jpg](../iclr_results/1585_ClimaQA_%20An%20Automated%20Evaluation%20Framework%20for%20Climate%20Question%20Answering%20Models/tables/0967e97a8d452318b73824c82197651fbed934579cbf87749afcc729e493aa72.jpg)

![1d3b52ea850f5c6adcb29d536cf64ddbc2b05047b3234c748806ec037c71ba72.jpg](../iclr_results/1585_ClimaQA_%20An%20Automated%20Evaluation%20Framework%20for%20Climate%20Question%20Answering%20Models/tables/1d3b52ea850f5c6adcb29d536cf64ddbc2b05047b3234c748806ec037c71ba72.jpg)

![293525f90053d35496d335c5a80fc90bd984885c2b3b2ba1f3ca7e38aa8cda93.jpg](../iclr_results/1585_ClimaQA_%20An%20Automated%20Evaluation%20Framework%20for%20Climate%20Question%20Answering%20Models/tables/293525f90053d35496d335c5a80fc90bd984885c2b3b2ba1f3ca7e38aa8cda93.jpg)

![370c505996850c48114de098db66cfdd67df11c444fc95bdf2c3aea525a87896.jpg](../iclr_results/1585_ClimaQA_%20An%20Automated%20Evaluation%20Framework%20for%20Climate%20Question%20Answering%20Models/tables/370c505996850c48114de098db66cfdd67df11c444fc95bdf2c3aea525a87896.jpg)

![375fbc2526108b1c9ff4c94d5c842b464884186edaebae9902442fb91f4dad64.jpg](../iclr_results/1585_ClimaQA_%20An%20Automated%20Evaluation%20Framework%20for%20Climate%20Question%20Answering%20Models/tables/375fbc2526108b1c9ff4c94d5c842b464884186edaebae9902442fb91f4dad64.jpg)

![41e1d633cfbf1afc21943290c9637115100e2b7cbf1967c34a04cf6fe49dbfbe.jpg](../iclr_results/1585_ClimaQA_%20An%20Automated%20Evaluation%20Framework%20for%20Climate%20Question%20Answering%20Models/tables/41e1d633cfbf1afc21943290c9637115100e2b7cbf1967c34a04cf6fe49dbfbe.jpg)

![90ab0b8fc8f79caaffe898aff62fda30d50c8b78b618d89a710d8d38a08a8cbd.jpg](../iclr_results/1585_ClimaQA_%20An%20Automated%20Evaluation%20Framework%20for%20Climate%20Question%20Answering%20Models/tables/90ab0b8fc8f79caaffe898aff62fda30d50c8b78b618d89a710d8d38a08a8cbd.jpg)

![d9f8f93af94879e8d9fc206ea6010717794a099fd51e9399788c6bb493560be3.jpg](../iclr_results/1585_ClimaQA_%20An%20Automated%20Evaluation%20Framework%20for%20Climate%20Question%20Answering%20Models/tables/d9f8f93af94879e8d9fc206ea6010717794a099fd51e9399788c6bb493560be3.jpg)

![e8cfc83bd464e41ecf623bb14dcf9964d57c92e6240db3f30b475f7f733bb195.jpg](../iclr_results/1585_ClimaQA_%20An%20Automated%20Evaluation%20Framework%20for%20Climate%20Question%20Answering%20Models/tables/e8cfc83bd464e41ecf623bb14dcf9964d57c92e6240db3f30b475f7f733bb195.jpg)

![fe84a0b0c9e905bb5605afc7e0ee57778765c679c0ab94ad92f9852d72733ac2.jpg](../iclr_results/1585_ClimaQA_%20An%20Automated%20Evaluation%20Framework%20for%20Climate%20Question%20Answering%20Models/tables/fe84a0b0c9e905bb5605afc7e0ee57778765c679c0ab94ad92f9852d72733ac2.jpg)

## Directional Gradient Projection for Robust Fine-Tuning of Foundation Models


### Images

![2c261ff3cc7afaf5d8f8254831d2926dbd53fc1e2344ed2e2e880777c6bae171.jpg](../iclr_results/1586_Directional%20Gradient%20Projection%20for%20Robust%20Fine-Tuning%20of%20Foundation%20Models/images/2c261ff3cc7afaf5d8f8254831d2926dbd53fc1e2344ed2e2e880777c6bae171.jpg)

![2ec9bb38abadd36a06d375de71a25ce61f31e5d91ff34a0ddc580efd52700059.jpg](../iclr_results/1586_Directional%20Gradient%20Projection%20for%20Robust%20Fine-Tuning%20of%20Foundation%20Models/images/2ec9bb38abadd36a06d375de71a25ce61f31e5d91ff34a0ddc580efd52700059.jpg)

![3c3a34e70f6f20eab9849e328b3d59c438014d7b6b30aec74f793c83f346cf30.jpg](../iclr_results/1586_Directional%20Gradient%20Projection%20for%20Robust%20Fine-Tuning%20of%20Foundation%20Models/images/3c3a34e70f6f20eab9849e328b3d59c438014d7b6b30aec74f793c83f346cf30.jpg)

![5c56770c5046d7d4213991c27d0a07937cf644ed9c3846feab91b241f844d4d2.jpg](../iclr_results/1586_Directional%20Gradient%20Projection%20for%20Robust%20Fine-Tuning%20of%20Foundation%20Models/images/5c56770c5046d7d4213991c27d0a07937cf644ed9c3846feab91b241f844d4d2.jpg)

![64d4c454cddba20e6bbdf27b137562f354796fc655ac241b197eeb1e7d281303.jpg](../iclr_results/1586_Directional%20Gradient%20Projection%20for%20Robust%20Fine-Tuning%20of%20Foundation%20Models/images/64d4c454cddba20e6bbdf27b137562f354796fc655ac241b197eeb1e7d281303.jpg)

![84290a1591adefb84b0c136c1fae484a5436107de0a9c2797cf3440d2111df28.jpg](../iclr_results/1586_Directional%20Gradient%20Projection%20for%20Robust%20Fine-Tuning%20of%20Foundation%20Models/images/84290a1591adefb84b0c136c1fae484a5436107de0a9c2797cf3440d2111df28.jpg)

![8dc9785afcd92c956b30c4c1be119d08b8d6f14a6f5ddcbd1cd58d6170aca1b0.jpg](../iclr_results/1586_Directional%20Gradient%20Projection%20for%20Robust%20Fine-Tuning%20of%20Foundation%20Models/images/8dc9785afcd92c956b30c4c1be119d08b8d6f14a6f5ddcbd1cd58d6170aca1b0.jpg)

![933bc70d5229e76f33bae5c7015b3ca9d9d43a728031f3a9a760c1eb306a24cc.jpg](../iclr_results/1586_Directional%20Gradient%20Projection%20for%20Robust%20Fine-Tuning%20of%20Foundation%20Models/images/933bc70d5229e76f33bae5c7015b3ca9d9d43a728031f3a9a760c1eb306a24cc.jpg)

![dda0918b26a4e1ec151d7d6a87ac018d5c6ef184bc89260fe3b0bbf59c780f79.jpg](../iclr_results/1586_Directional%20Gradient%20Projection%20for%20Robust%20Fine-Tuning%20of%20Foundation%20Models/images/dda0918b26a4e1ec151d7d6a87ac018d5c6ef184bc89260fe3b0bbf59c780f79.jpg)

### Tables

![14622f646ed124d9e520ac315de772f43029fc1a56fb188541ecc5e5fdb72a99.jpg](../iclr_results/1586_Directional%20Gradient%20Projection%20for%20Robust%20Fine-Tuning%20of%20Foundation%20Models/tables/14622f646ed124d9e520ac315de772f43029fc1a56fb188541ecc5e5fdb72a99.jpg)

![1a7f66f9e1bea416980d3113ccec1963a457fb1a9aa3b069f2cb6b298a92575c.jpg](../iclr_results/1586_Directional%20Gradient%20Projection%20for%20Robust%20Fine-Tuning%20of%20Foundation%20Models/tables/1a7f66f9e1bea416980d3113ccec1963a457fb1a9aa3b069f2cb6b298a92575c.jpg)

![31d71b822916db8215c495a8f792562181fd13ddb26e2353f04a4de25502954c.jpg](../iclr_results/1586_Directional%20Gradient%20Projection%20for%20Robust%20Fine-Tuning%20of%20Foundation%20Models/tables/31d71b822916db8215c495a8f792562181fd13ddb26e2353f04a4de25502954c.jpg)

![324176ee6aaf4100c2b6dd2185eead54f6a4b394c8f4ca170c3ca4e6ae70a5ff.jpg](../iclr_results/1586_Directional%20Gradient%20Projection%20for%20Robust%20Fine-Tuning%20of%20Foundation%20Models/tables/324176ee6aaf4100c2b6dd2185eead54f6a4b394c8f4ca170c3ca4e6ae70a5ff.jpg)

![6374914860313b1dff8186bc3ccd0f48b0eb324aae2f936d97db37772e8c6279.jpg](../iclr_results/1586_Directional%20Gradient%20Projection%20for%20Robust%20Fine-Tuning%20of%20Foundation%20Models/tables/6374914860313b1dff8186bc3ccd0f48b0eb324aae2f936d97db37772e8c6279.jpg)

![901ea8850a402a9951a69bba7173c928fdef486dfc6e040974bd12d44254b49c.jpg](../iclr_results/1586_Directional%20Gradient%20Projection%20for%20Robust%20Fine-Tuning%20of%20Foundation%20Models/tables/901ea8850a402a9951a69bba7173c928fdef486dfc6e040974bd12d44254b49c.jpg)

![97aba02cff7ae4c59892c2d071dccca1d4cd2b1de48f8224f1a6e4c95264a3a5.jpg](../iclr_results/1586_Directional%20Gradient%20Projection%20for%20Robust%20Fine-Tuning%20of%20Foundation%20Models/tables/97aba02cff7ae4c59892c2d071dccca1d4cd2b1de48f8224f1a6e4c95264a3a5.jpg)

![97d68525973be3362d895bcf4714eba253eb7b9a940ffb9304c62bf9b1bcffb4.jpg](../iclr_results/1586_Directional%20Gradient%20Projection%20for%20Robust%20Fine-Tuning%20of%20Foundation%20Models/tables/97d68525973be3362d895bcf4714eba253eb7b9a940ffb9304c62bf9b1bcffb4.jpg)

![9e5ae7b8ae760a27d3d49a7cee7a7111fb1b0669ef6287015a14936de45b951c.jpg](../iclr_results/1586_Directional%20Gradient%20Projection%20for%20Robust%20Fine-Tuning%20of%20Foundation%20Models/tables/9e5ae7b8ae760a27d3d49a7cee7a7111fb1b0669ef6287015a14936de45b951c.jpg)

![d51e344178a1a3e5a5ecc886064e66b537f499719d0c0a12623fa1c693b47d3a.jpg](../iclr_results/1586_Directional%20Gradient%20Projection%20for%20Robust%20Fine-Tuning%20of%20Foundation%20Models/tables/d51e344178a1a3e5a5ecc886064e66b537f499719d0c0a12623fa1c693b47d3a.jpg)

![e4f5094f214b915c5d0a3206256b5fe206d689fdbd7dd072ee5d4414819c20b4.jpg](../iclr_results/1586_Directional%20Gradient%20Projection%20for%20Robust%20Fine-Tuning%20of%20Foundation%20Models/tables/e4f5094f214b915c5d0a3206256b5fe206d689fdbd7dd072ee5d4414819c20b4.jpg)

![e52fec1b075a7f352108af6133f74ffc6c525b9e5499e3a72e8851c66ace64a3.jpg](../iclr_results/1586_Directional%20Gradient%20Projection%20for%20Robust%20Fine-Tuning%20of%20Foundation%20Models/tables/e52fec1b075a7f352108af6133f74ffc6c525b9e5499e3a72e8851c66ace64a3.jpg)

![ffef89459f363791b3278cc63a9541bb37bf51dab21d0ee288cce685c596cd33.jpg](../iclr_results/1586_Directional%20Gradient%20Projection%20for%20Robust%20Fine-Tuning%20of%20Foundation%20Models/tables/ffef89459f363791b3278cc63a9541bb37bf51dab21d0ee288cce685c596cd33.jpg)

## SCBench: A KV Cache-Centric Analysis of Long-Context Methods


### Images

![05788b1ae540fddead38d158100c078860b357e7d9e183e41ce0710ab1b46589.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/images/05788b1ae540fddead38d158100c078860b357e7d9e183e41ce0710ab1b46589.jpg)

![19219c753247721a651caae8f620bdfa4f82cf8d9fca1e48846d8dc84de0d630.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/images/19219c753247721a651caae8f620bdfa4f82cf8d9fca1e48846d8dc84de0d630.jpg)

![30f86e67aaff39ad36dfc264e677db9f45550e531fe12bad90f9b058caee3b1a.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/images/30f86e67aaff39ad36dfc264e677db9f45550e531fe12bad90f9b058caee3b1a.jpg)

![6822646abf7a0bf8835cfe51877d12e7402880a5d81ede6b64dff866d4b79d1f.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/images/6822646abf7a0bf8835cfe51877d12e7402880a5d81ede6b64dff866d4b79d1f.jpg)

![6bf0e2528c10182b7f7c69acab95641fa656ea9d81c3c782364c8f528b240b9c.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/images/6bf0e2528c10182b7f7c69acab95641fa656ea9d81c3c782364c8f528b240b9c.jpg)

![918d8f8841a0eb909395802636d87d5dff74530ee57c1b44eb753c96e4ecd56c.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/images/918d8f8841a0eb909395802636d87d5dff74530ee57c1b44eb753c96e4ecd56c.jpg)

![926eea3a1775218d05901726789b770cedc112edbd5b2e25394f738453a0f833.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/images/926eea3a1775218d05901726789b770cedc112edbd5b2e25394f738453a0f833.jpg)

![ab70d0776b15b114c6a9713f1c0748077062de10f6009c9133a1fd4b62a10824.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/images/ab70d0776b15b114c6a9713f1c0748077062de10f6009c9133a1fd4b62a10824.jpg)

![ace288cb7acbe63c90d3e3a4c912d9898004cc157ab888163da3412754f21d2d.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/images/ace288cb7acbe63c90d3e3a4c912d9898004cc157ab888163da3412754f21d2d.jpg)

![e98397e37534ce23eb111d0d7f4cb1576e1ab03bff1a2ef4ae33c97b517de160.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/images/e98397e37534ce23eb111d0d7f4cb1576e1ab03bff1a2ef4ae33c97b517de160.jpg)

### Tables

![0c020c62c748845089e2666fa94fb8c9299f2ea741a322733c0f6e07d7138a6b.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/tables/0c020c62c748845089e2666fa94fb8c9299f2ea741a322733c0f6e07d7138a6b.jpg)

![135d99d21efdb9a4ac1f43068f028ca94cfcbd1c360215eb64a09f7630ddccfe.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/tables/135d99d21efdb9a4ac1f43068f028ca94cfcbd1c360215eb64a09f7630ddccfe.jpg)

![2f2f7977bd711b12006aabef40f8f0d3f3c1bc88c8af5d6f3914b1725bbc046e.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/tables/2f2f7977bd711b12006aabef40f8f0d3f3c1bc88c8af5d6f3914b1725bbc046e.jpg)

![355acdcbc4e53899795657de8d4c5c101c80dbd2121c1d2d79678587899e1e3d.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/tables/355acdcbc4e53899795657de8d4c5c101c80dbd2121c1d2d79678587899e1e3d.jpg)

![37c861e2556d73f32ae92e7a97ebb48df6156add009def5fe53a0ed1a000ca03.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/tables/37c861e2556d73f32ae92e7a97ebb48df6156add009def5fe53a0ed1a000ca03.jpg)

![3f4e1cabed1249ed5ab1af396be0dc92cc936dcc6c02aa84d35280f346a1c4ee.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/tables/3f4e1cabed1249ed5ab1af396be0dc92cc936dcc6c02aa84d35280f346a1c4ee.jpg)

![4369e4351e539db8aed8c628bf814dbc9b60fce9e94289c16f95a115c2040184.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/tables/4369e4351e539db8aed8c628bf814dbc9b60fce9e94289c16f95a115c2040184.jpg)

![59b4761d045379fa37fc8f8158915796f31e7819328d7cbf534b693450287de1.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/tables/59b4761d045379fa37fc8f8158915796f31e7819328d7cbf534b693450287de1.jpg)

![6f54a8590ad2e6f82d886011a09b21148cfa74a91bafb68643ecfc14c6267c04.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/tables/6f54a8590ad2e6f82d886011a09b21148cfa74a91bafb68643ecfc14c6267c04.jpg)

![7d459a3b8e892dd272914d40ae9bfac9629d6913ae70c9478c05915c981f3edb.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/tables/7d459a3b8e892dd272914d40ae9bfac9629d6913ae70c9478c05915c981f3edb.jpg)

![848963fc90f834339e16ed890ac6820d19fa39fc49ed62790aa1cee317dcadb6.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/tables/848963fc90f834339e16ed890ac6820d19fa39fc49ed62790aa1cee317dcadb6.jpg)

![8498fe495b1d89d8ab5e65973c55b4a80bc66fa36eec406320720264600e15ab.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/tables/8498fe495b1d89d8ab5e65973c55b4a80bc66fa36eec406320720264600e15ab.jpg)

![9a6811e5fb42ba1cc340c3f5e5107eb952940712b8d8bf18941beb04fe4dcd3d.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/tables/9a6811e5fb42ba1cc340c3f5e5107eb952940712b8d8bf18941beb04fe4dcd3d.jpg)

![af5af91b0cdf275eece16a87a4d2bfaa863aa9d99cf506c0f6ed98519b1d49ec.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/tables/af5af91b0cdf275eece16a87a4d2bfaa863aa9d99cf506c0f6ed98519b1d49ec.jpg)

![b7696e913753a021a2bd0fa3a6fd0859a992b1a06dac3f42212d337fa08322a7.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/tables/b7696e913753a021a2bd0fa3a6fd0859a992b1a06dac3f42212d337fa08322a7.jpg)

![eea41940032ed92e45e7f28aa9d77ed2e1190ffb12bcb8cdce634c3297269888.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/tables/eea41940032ed92e45e7f28aa9d77ed2e1190ffb12bcb8cdce634c3297269888.jpg)

![fa0e69d559387e26385a9bc057583173da59c72233634b38774bb6d8305723a3.jpg](../iclr_results/1587_SCBench_%20A%20KV%20Cache-Centric%20Analysis%20of%20Long-Context%20Methods/tables/fa0e69d559387e26385a9bc057583173da59c72233634b38774bb6d8305723a3.jpg)

## Score Forgetting Distillation: A Swift, Data-Free Method for Machine Unlearning in Diffusion Models


### Images

![05b668097f98b58d14c0494028c2044fb03e906cd31fb7ca83bfee275d174be0.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/images/05b668097f98b58d14c0494028c2044fb03e906cd31fb7ca83bfee275d174be0.jpg)

![18ae2c8d380af4a33434030206dd214d92705f8459f92129941c6d786074346e.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/images/18ae2c8d380af4a33434030206dd214d92705f8459f92129941c6d786074346e.jpg)

![215a81508761a732b332065b96d465649e66f7a4ff7227c5b6ae81c5e08d4459.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/images/215a81508761a732b332065b96d465649e66f7a4ff7227c5b6ae81c5e08d4459.jpg)

![3e43035d6ba554a5c385489b1b98217b24624cecbf1684b792bd5586aadded88.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/images/3e43035d6ba554a5c385489b1b98217b24624cecbf1684b792bd5586aadded88.jpg)

![410d7716c40b8751884fb601c95bd461c682c8d43cea1b89c79119c2d08f120b.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/images/410d7716c40b8751884fb601c95bd461c682c8d43cea1b89c79119c2d08f120b.jpg)

![4fcdd03515738ccbc22eb3fe31035c58c8b184e771468124edf7cb3a1b4b8559.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/images/4fcdd03515738ccbc22eb3fe31035c58c8b184e771468124edf7cb3a1b4b8559.jpg)

![6709898f8ccd71a474cd4f37d12438083be6822b55e0da7054e280becbecaa96.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/images/6709898f8ccd71a474cd4f37d12438083be6822b55e0da7054e280becbecaa96.jpg)

![6b2acd82084c82a1af4a376f2dc1173301da309d40e4f4e0115eea99e00e27af.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/images/6b2acd82084c82a1af4a376f2dc1173301da309d40e4f4e0115eea99e00e27af.jpg)

![7f1be16a63531d12af782edf8bb2e3413d89ceacb7623990229f5ab9a7667fdb.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/images/7f1be16a63531d12af782edf8bb2e3413d89ceacb7623990229f5ab9a7667fdb.jpg)

![828cb678bc920bc31231f8cc9aa49a444f3afa2027e5693bda15e3a5ecc797f1.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/images/828cb678bc920bc31231f8cc9aa49a444f3afa2027e5693bda15e3a5ecc797f1.jpg)

![900da8ca84645d7b069ffee9722c86d5cb8c88465c1f1024fbcfca5926c7e2fd.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/images/900da8ca84645d7b069ffee9722c86d5cb8c88465c1f1024fbcfca5926c7e2fd.jpg)

![912cdf940e0ddae0a23b72cf03c2c46e29de34e9cb658894239b0af35b2b3b3f.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/images/912cdf940e0ddae0a23b72cf03c2c46e29de34e9cb658894239b0af35b2b3b3f.jpg)

![a8d122ec8217779a028bdf77ce2c28e5b213000e95dff02cd49eda4042c18f80.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/images/a8d122ec8217779a028bdf77ce2c28e5b213000e95dff02cd49eda4042c18f80.jpg)

### Tables

![023fbc6fa568e78d412abf722f2bda165d81465d8809bb76678c476a45905d9e.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/tables/023fbc6fa568e78d412abf722f2bda165d81465d8809bb76678c476a45905d9e.jpg)

![0d057bd6ac85c36ad98644f2ea2fb071e3133d32b52dfdc518e22c7c4c3f1fa4.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/tables/0d057bd6ac85c36ad98644f2ea2fb071e3133d32b52dfdc518e22c7c4c3f1fa4.jpg)

![1e51f91441e19dd797827590810f440d5cbaec5439435a17493e5634af58674f.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/tables/1e51f91441e19dd797827590810f440d5cbaec5439435a17493e5634af58674f.jpg)

![36e8e6ee147ddea6bb32d00c738b96527dc1ab4bd6be13008ac55a0b5f015ee4.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/tables/36e8e6ee147ddea6bb32d00c738b96527dc1ab4bd6be13008ac55a0b5f015ee4.jpg)

![3e6286acdbd9437ddd5de0ab5cbb0e94ec9b30c5f2d460f60d8fc3757ced1e78.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/tables/3e6286acdbd9437ddd5de0ab5cbb0e94ec9b30c5f2d460f60d8fc3757ced1e78.jpg)

![442a956e7a55f68fbf8fedb2a5c841d0a7dcb675ca5c69c39ec75b1b616ae242.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/tables/442a956e7a55f68fbf8fedb2a5c841d0a7dcb675ca5c69c39ec75b1b616ae242.jpg)

![64950663b89e4e08baf8bdf03e93479ecc6d4c5208d5645968eb8dcbcb9da063.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/tables/64950663b89e4e08baf8bdf03e93479ecc6d4c5208d5645968eb8dcbcb9da063.jpg)

![75c95b84600d1697421922c5b1981fae7856083fe8ac686a0a882de3092ae5a5.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/tables/75c95b84600d1697421922c5b1981fae7856083fe8ac686a0a882de3092ae5a5.jpg)

![7877beefba92c653b682f643f27a76fc364ffae4bd00d93ec4011b911d75cc75.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/tables/7877beefba92c653b682f643f27a76fc364ffae4bd00d93ec4011b911d75cc75.jpg)

![a5f912ce1b8d690c1f43d0edcfe86f15852a0d0f6fb69223bbe9c09037428798.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/tables/a5f912ce1b8d690c1f43d0edcfe86f15852a0d0f6fb69223bbe9c09037428798.jpg)

![cdbd5b5961cd732adf9fbe4f65302d6f6baf5e0539fb786545b4071443becc81.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/tables/cdbd5b5961cd732adf9fbe4f65302d6f6baf5e0539fb786545b4071443becc81.jpg)

![d5308147cf7a77dccf91c8eb161f874a0664c53e5def78b144d3b8ca0fdb3374.jpg](../iclr_results/1588_Score%20Forgetting%20Distillation_%20A%20Swift%2C%20Data-Free%20Method%20for%20Machine%20Unlearning%20in%20Diffusion%20Models/tables/d5308147cf7a77dccf91c8eb161f874a0664c53e5def78b144d3b8ca0fdb3374.jpg)

## GraphBridge: Towards Arbitrary Transfer Learning in GNNs


### Images

![516c6c940bff6a91a6d5586958d3455dba772519d2eb5c78c2c27b21f83f420b.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/images/516c6c940bff6a91a6d5586958d3455dba772519d2eb5c78c2c27b21f83f420b.jpg)

![544b55c4a6732f70646efb70d2af2895fd1ad5d594ea1d865696730e87e691d6.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/images/544b55c4a6732f70646efb70d2af2895fd1ad5d594ea1d865696730e87e691d6.jpg)

![650d2f0be25dc553a4584417156ff2beecd3bd681a63ae349fcfb4a24b0361b6.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/images/650d2f0be25dc553a4584417156ff2beecd3bd681a63ae349fcfb4a24b0361b6.jpg)

![6c2a4f6fd5a56cbf49bac282e79456925d5b8e315154c1981bc680786eaf9e80.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/images/6c2a4f6fd5a56cbf49bac282e79456925d5b8e315154c1981bc680786eaf9e80.jpg)

![7092fa5059e2fb95f0adcde349f4c1e7eb5357edc71dbf9ee31674319d8b5abb.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/images/7092fa5059e2fb95f0adcde349f4c1e7eb5357edc71dbf9ee31674319d8b5abb.jpg)

![76630af37369e0c67c014a02cf08d9d1554fe0a3780e072569e1f933f8f61a36.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/images/76630af37369e0c67c014a02cf08d9d1554fe0a3780e072569e1f933f8f61a36.jpg)

### Tables

![02aae052a30f3d54f17b13247f62fd4611d12dc96b28100fcb97c0aa94672502.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/tables/02aae052a30f3d54f17b13247f62fd4611d12dc96b28100fcb97c0aa94672502.jpg)

![0cc913d9253226dbc9d842fbaefb3417dfb5eafbb961109eca49badb048b5874.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/tables/0cc913d9253226dbc9d842fbaefb3417dfb5eafbb961109eca49badb048b5874.jpg)

![2579fa1e178bb77ecb18af9144fffdb06748e9b87276d0fdeabd973e14d59d9e.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/tables/2579fa1e178bb77ecb18af9144fffdb06748e9b87276d0fdeabd973e14d59d9e.jpg)

![3c5a4f6ff0b0d3fbb939f3e0e22e9d7652b99720043b6d64f109d53d6d72ba08.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/tables/3c5a4f6ff0b0d3fbb939f3e0e22e9d7652b99720043b6d64f109d53d6d72ba08.jpg)

![41413c6184b93621d06b2385e329eb5c6c43f7d53f458e061bc4644ec9b48ef3.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/tables/41413c6184b93621d06b2385e329eb5c6c43f7d53f458e061bc4644ec9b48ef3.jpg)

![598d48bb10fe19b6ce673526d60eaf0e4e45e05b788bfdc7916d9f0888f4d759.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/tables/598d48bb10fe19b6ce673526d60eaf0e4e45e05b788bfdc7916d9f0888f4d759.jpg)

![806f98d4537e84f9b1c441833b829ca32210c7f0ee74ddac51d00f35fd6f6809.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/tables/806f98d4537e84f9b1c441833b829ca32210c7f0ee74ddac51d00f35fd6f6809.jpg)

![85d8fb435147274d7e4f2b5ff3ca4e0682c6e210f0e1476968be74fd03dc4459.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/tables/85d8fb435147274d7e4f2b5ff3ca4e0682c6e210f0e1476968be74fd03dc4459.jpg)

![8b93bd5a70f5ee508cda9ba2122686ac5fef93e4414b0840c94574658ba536a9.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/tables/8b93bd5a70f5ee508cda9ba2122686ac5fef93e4414b0840c94574658ba536a9.jpg)

![8da1f3156f6b90f6c49970efccf090514954046cd1b26c248efd47818840f17d.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/tables/8da1f3156f6b90f6c49970efccf090514954046cd1b26c248efd47818840f17d.jpg)

![8f630f5a491348138e13e34cbbb2c18664bb4edab5a62f59f33e4bf7ada735e7.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/tables/8f630f5a491348138e13e34cbbb2c18664bb4edab5a62f59f33e4bf7ada735e7.jpg)

![924e370f01a7fe9abc24894d33ae052c40fefba88a0b565244f17e8e04dc40cb.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/tables/924e370f01a7fe9abc24894d33ae052c40fefba88a0b565244f17e8e04dc40cb.jpg)

![9f1b186323c0020f141f4ce299979548cfac5e238503c9acd4b2a6165c75637e.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/tables/9f1b186323c0020f141f4ce299979548cfac5e238503c9acd4b2a6165c75637e.jpg)

![a2c8a56f504803b6371fd25683ee54202792d799d3947228a325ff30110df711.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/tables/a2c8a56f504803b6371fd25683ee54202792d799d3947228a325ff30110df711.jpg)

![a896d17ad3b74b5fc7285b4a31f2ebe599df26169f388dd87673987391107a37.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/tables/a896d17ad3b74b5fc7285b4a31f2ebe599df26169f388dd87673987391107a37.jpg)

![e2d0f68ded1180d7a7f6f47cbac2eec1392b9de55408b193b7615ea6b16a3ea1.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/tables/e2d0f68ded1180d7a7f6f47cbac2eec1392b9de55408b193b7615ea6b16a3ea1.jpg)

![e878c5636d6d11ec51aede8a84104a2e05bff8d3a5f9dbe1473515e18eb02de5.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/tables/e878c5636d6d11ec51aede8a84104a2e05bff8d3a5f9dbe1473515e18eb02de5.jpg)

![eb26a15f4edcb3f58f04ff836afff52f5810b141fd39cee300716c247b0eb917.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/tables/eb26a15f4edcb3f58f04ff836afff52f5810b141fd39cee300716c247b0eb917.jpg)

![ee53ff6f3ada7db7e14d8e8d51bf60acc4ee2946fc084aea014a583b478eb4ee.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/tables/ee53ff6f3ada7db7e14d8e8d51bf60acc4ee2946fc084aea014a583b478eb4ee.jpg)

![f63d41d95998898b517cd2959ba21fa8c66777174aaf235cc0c3a00e777920a6.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/tables/f63d41d95998898b517cd2959ba21fa8c66777174aaf235cc0c3a00e777920a6.jpg)

![fcd741de2f6d3c4903fc897ffc88f505df1f6a6866ac948ae713d96df9e5c52d.jpg](../iclr_results/1589_GraphBridge_%20Towards%20Arbitrary%20Transfer%20Learning%20in%20GNNs/tables/fcd741de2f6d3c4903fc897ffc88f505df1f6a6866ac948ae713d96df9e5c52d.jpg)

## Build-A-Scene: Interactive 3D Layout Control for Diffusion-Based Image Generation


### Images

![5ad04ee10d8de6f5e24fa593993f349774a577253a8b004e896b020e6448f715.jpg](../iclr_results/1590_Build-A-Scene_%20Interactive%203D%20Layout%20Control%20for%20Diffusion-Based%20Image%20Generation/images/5ad04ee10d8de6f5e24fa593993f349774a577253a8b004e896b020e6448f715.jpg)

![67b85e39f196e32a88e4c8ee7e57afc40aa47b8a09dad9eb35151dd4498d03d2.jpg](../iclr_results/1590_Build-A-Scene_%20Interactive%203D%20Layout%20Control%20for%20Diffusion-Based%20Image%20Generation/images/67b85e39f196e32a88e4c8ee7e57afc40aa47b8a09dad9eb35151dd4498d03d2.jpg)

![69b9c4886ad08b119e495d36b028f43bcb0c50895dd2aa6ce28a9c422fa73217.jpg](../iclr_results/1590_Build-A-Scene_%20Interactive%203D%20Layout%20Control%20for%20Diffusion-Based%20Image%20Generation/images/69b9c4886ad08b119e495d36b028f43bcb0c50895dd2aa6ce28a9c422fa73217.jpg)

![abc0e7990c28087c92ef465ad2485a6bd0aa310b1f3ba3d1e65aa6118577dcb0.jpg](../iclr_results/1590_Build-A-Scene_%20Interactive%203D%20Layout%20Control%20for%20Diffusion-Based%20Image%20Generation/images/abc0e7990c28087c92ef465ad2485a6bd0aa310b1f3ba3d1e65aa6118577dcb0.jpg)

![b6ab8f80ab4116bda897584f3cfacf3b4deac42ae2496caccde1f98c108e654d.jpg](../iclr_results/1590_Build-A-Scene_%20Interactive%203D%20Layout%20Control%20for%20Diffusion-Based%20Image%20Generation/images/b6ab8f80ab4116bda897584f3cfacf3b4deac42ae2496caccde1f98c108e654d.jpg)

![b718a0ad793098888f0725abecc68343141d823524d55cdd673f692d6b1126a6.jpg](../iclr_results/1590_Build-A-Scene_%20Interactive%203D%20Layout%20Control%20for%20Diffusion-Based%20Image%20Generation/images/b718a0ad793098888f0725abecc68343141d823524d55cdd673f692d6b1126a6.jpg)

![cfaa1248c75fbfb61d9dc77ff024cbf41e1604af05095d90368954cf37ebb2ad.jpg](../iclr_results/1590_Build-A-Scene_%20Interactive%203D%20Layout%20Control%20for%20Diffusion-Based%20Image%20Generation/images/cfaa1248c75fbfb61d9dc77ff024cbf41e1604af05095d90368954cf37ebb2ad.jpg)

![d8fbebfbb87dff3cb96e27f558dc9abdaf4f109c0ffa84f5a8be73a4c2d39547.jpg](../iclr_results/1590_Build-A-Scene_%20Interactive%203D%20Layout%20Control%20for%20Diffusion-Based%20Image%20Generation/images/d8fbebfbb87dff3cb96e27f558dc9abdaf4f109c0ffa84f5a8be73a4c2d39547.jpg)

![e877d1bee881d9a5e195fea1a8c6f5079d72926b81170e88aa2b5ab93797d5b5.jpg](../iclr_results/1590_Build-A-Scene_%20Interactive%203D%20Layout%20Control%20for%20Diffusion-Based%20Image%20Generation/images/e877d1bee881d9a5e195fea1a8c6f5079d72926b81170e88aa2b5ab93797d5b5.jpg)

![ef176efa723a2c7f2ff67dfa0365e703aa3db6097ff4befea3539d8b47e0ebff.jpg](../iclr_results/1590_Build-A-Scene_%20Interactive%203D%20Layout%20Control%20for%20Diffusion-Based%20Image%20Generation/images/ef176efa723a2c7f2ff67dfa0365e703aa3db6097ff4befea3539d8b47e0ebff.jpg)

![f632a6ae611b09db9020458572c0b07d881e5e3ec16e785f5208b4a92b70ffe8.jpg](../iclr_results/1590_Build-A-Scene_%20Interactive%203D%20Layout%20Control%20for%20Diffusion-Based%20Image%20Generation/images/f632a6ae611b09db9020458572c0b07d881e5e3ec16e785f5208b4a92b70ffe8.jpg)

![f744b7bcb54d3de14dfc10c96b782b2ab618817243cbb70df05899e3cdd17c59.jpg](../iclr_results/1590_Build-A-Scene_%20Interactive%203D%20Layout%20Control%20for%20Diffusion-Based%20Image%20Generation/images/f744b7bcb54d3de14dfc10c96b782b2ab618817243cbb70df05899e3cdd17c59.jpg)

![fe8054ca59475075293cc6e7d553028d5523097b931c2afa157abae224f0a08c.jpg](../iclr_results/1590_Build-A-Scene_%20Interactive%203D%20Layout%20Control%20for%20Diffusion-Based%20Image%20Generation/images/fe8054ca59475075293cc6e7d553028d5523097b931c2afa157abae224f0a08c.jpg)

### Tables

![77aadd6632da0d37d7d0527cfd71f305eda7123267a0c36d03384f5d34089b3a.jpg](../iclr_results/1590_Build-A-Scene_%20Interactive%203D%20Layout%20Control%20for%20Diffusion-Based%20Image%20Generation/tables/77aadd6632da0d37d7d0527cfd71f305eda7123267a0c36d03384f5d34089b3a.jpg)

## Strategist: Self-improvement of LLM Decision Making via Bi-Level Tree Search


### Images

![082e5fe5515cb81645f3243154ff9f8822c2436cc0cb2e353b99400fdca2bc88.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/082e5fe5515cb81645f3243154ff9f8822c2436cc0cb2e353b99400fdca2bc88.jpg)

![0cb08b8efe146b247a90999bc7caf149f7abc1288784c02c6ca2099b1b4ca4b9.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/0cb08b8efe146b247a90999bc7caf149f7abc1288784c02c6ca2099b1b4ca4b9.jpg)

![2e195766251193b39230e50848f65de5548dd01a5f07d23ff909d7f80fd1155f.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/2e195766251193b39230e50848f65de5548dd01a5f07d23ff909d7f80fd1155f.jpg)

![2f98dec966d22f46439847f144d087f3123bd3a5a539d4598b612a652eeaeb98.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/2f98dec966d22f46439847f144d087f3123bd3a5a539d4598b612a652eeaeb98.jpg)

![32426c2436beb1829208da77306fb35114649a79153d76e7eba59c3bc3180c71.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/32426c2436beb1829208da77306fb35114649a79153d76e7eba59c3bc3180c71.jpg)

![47ba5239c50ccd54f291d78880176a0bd25a762ed07d34344366d57ac16ef485.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/47ba5239c50ccd54f291d78880176a0bd25a762ed07d34344366d57ac16ef485.jpg)

![486260e68e20c20580a00679dc690082040bd290d49739f40623d87d0fcbeebb.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/486260e68e20c20580a00679dc690082040bd290d49739f40623d87d0fcbeebb.jpg)

![4e4ee6e666842b5454ed6f99547d1b29fa1c04d8bf46a61c2811bf65b9b90b23.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/4e4ee6e666842b5454ed6f99547d1b29fa1c04d8bf46a61c2811bf65b9b90b23.jpg)

![573ae2fc535675acd9bcb6d726972f5c2ab4d45d30b474d0d3b3f65ba4b60244.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/573ae2fc535675acd9bcb6d726972f5c2ab4d45d30b474d0d3b3f65ba4b60244.jpg)

![597c09bd44d3f98cb3538079f44ebf12daaa5b48802aaed63e4dc3221a8f0379.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/597c09bd44d3f98cb3538079f44ebf12daaa5b48802aaed63e4dc3221a8f0379.jpg)

![7eb6fa39a9772932f71a02e77e5f3a5e85247c80d66fdc930870135957f45882.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/7eb6fa39a9772932f71a02e77e5f3a5e85247c80d66fdc930870135957f45882.jpg)

![83a77a38281e604cb68293b1c3e439cd0069f4f0abd718aeae0807cdc76dd864.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/83a77a38281e604cb68293b1c3e439cd0069f4f0abd718aeae0807cdc76dd864.jpg)

![884c50272bf1994de6d8ed1c7c47e6d1711a1ebd4746cd9bdd4e721633b92ed0.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/884c50272bf1994de6d8ed1c7c47e6d1711a1ebd4746cd9bdd4e721633b92ed0.jpg)

![88cadb2349b2988da617fe0ab3916f52194c066d6fbf5ac3f74915d63cf21b3f.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/88cadb2349b2988da617fe0ab3916f52194c066d6fbf5ac3f74915d63cf21b3f.jpg)

![8c9b18aacd40d654d01fe864a84e84b34919c58764de3b9bd94b08739a0cd568.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/8c9b18aacd40d654d01fe864a84e84b34919c58764de3b9bd94b08739a0cd568.jpg)

![9941ee32e4bd8282cb0b39e860b1cf676c616785cd9ae6922710d9975b23b266.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/9941ee32e4bd8282cb0b39e860b1cf676c616785cd9ae6922710d9975b23b266.jpg)

![9d1d2e6706185685241b4875d5f89edaf81175dd5b1395fa67c5ce759aa62a8e.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/9d1d2e6706185685241b4875d5f89edaf81175dd5b1395fa67c5ce759aa62a8e.jpg)

![a109c8a5a4f1c9fa533e672fed84ef4d51a92d12287fabd4929f049f3e04f5df.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/a109c8a5a4f1c9fa533e672fed84ef4d51a92d12287fabd4929f049f3e04f5df.jpg)

![c42c66e32ae5958103077f6118ff90b8788e718582e62c93e3e1cd3f82f41bf0.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/c42c66e32ae5958103077f6118ff90b8788e718582e62c93e3e1cd3f82f41bf0.jpg)

![c87ce276c617613ff017d2f810cb0495998c2c0ddfc79c217af54aec2c25816b.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/c87ce276c617613ff017d2f810cb0495998c2c0ddfc79c217af54aec2c25816b.jpg)

![dad9686fdcb781d8e0a97ad3093f57e05ebde608914b68bc5389c1c866b72df6.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/dad9686fdcb781d8e0a97ad3093f57e05ebde608914b68bc5389c1c866b72df6.jpg)

![e219fe9f2dbadb6d0d768ad327d5499bb653a59894652a803f49150460a59eee.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/e219fe9f2dbadb6d0d768ad327d5499bb653a59894652a803f49150460a59eee.jpg)

![ea4c901dce032309c236f030be03b9fbd8d6e8d11609636401c713d28d0eb2ac.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/ea4c901dce032309c236f030be03b9fbd8d6e8d11609636401c713d28d0eb2ac.jpg)

![f5fad789aff6c66a44e1c7ba8ce7207fbb503e53cc84eca8856c75b87d6155af.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/f5fad789aff6c66a44e1c7ba8ce7207fbb503e53cc84eca8856c75b87d6155af.jpg)

![ffc0ac1ac5cd54db524f5a5a343a91c95c8cd7643cb8f1efe43b865a53f8e38b.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/images/ffc0ac1ac5cd54db524f5a5a343a91c95c8cd7643cb8f1efe43b865a53f8e38b.jpg)

### Tables

![102a31894bf497db63043a118aced1e425d786ecdf9f2aeb94622697dbf06461.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/tables/102a31894bf497db63043a118aced1e425d786ecdf9f2aeb94622697dbf06461.jpg)

![275234c8d66f61dd322e16c99d7af5cc555a4c7afb4a323d70697c5f7ff159f7.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/tables/275234c8d66f61dd322e16c99d7af5cc555a4c7afb4a323d70697c5f7ff159f7.jpg)

![3b727fe7987f89a38a907978a7cc977f313c8b7f4bdff87210cfcde7025b02df.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/tables/3b727fe7987f89a38a907978a7cc977f313c8b7f4bdff87210cfcde7025b02df.jpg)

![4b57ebfa348ed32c3d78675f293f5cebee7412d2ddd9425ac7899ee2298ebc15.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/tables/4b57ebfa348ed32c3d78675f293f5cebee7412d2ddd9425ac7899ee2298ebc15.jpg)

![596b0221a3472b5ddea0353c2f3f3f8f8118f931b0f46020e25aa670ee4b23ad.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/tables/596b0221a3472b5ddea0353c2f3f3f8f8118f931b0f46020e25aa670ee4b23ad.jpg)

![8aceb1df07bb787b422d895dfab639501a4902e21bcbaa0585b6cad9541e0b8c.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/tables/8aceb1df07bb787b422d895dfab639501a4902e21bcbaa0585b6cad9541e0b8c.jpg)

![9516125c61b1eee63740466bc96d86d113fe5d4e880e036d65998dd1ae56670f.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/tables/9516125c61b1eee63740466bc96d86d113fe5d4e880e036d65998dd1ae56670f.jpg)

![9bcab66bb0f65167700e9aa00ea8483d9779ffaf29eb9cca34ec680fb81d05c9.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/tables/9bcab66bb0f65167700e9aa00ea8483d9779ffaf29eb9cca34ec680fb81d05c9.jpg)

![aa157bbf69fea439a4917f4983cda5e2641cd2471a04ec53f6c4cb75f53bb233.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/tables/aa157bbf69fea439a4917f4983cda5e2641cd2471a04ec53f6c4cb75f53bb233.jpg)

![b9492123c20c8ed3fb87695bae686c787ea5e9534ba26e03bd1a728366b8d7ec.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/tables/b9492123c20c8ed3fb87695bae686c787ea5e9534ba26e03bd1a728366b8d7ec.jpg)

![bae6f2d1495eee9b78bfa2ef0360aba3d22ebed6b2c0498241e3c1e6ed8f5a84.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/tables/bae6f2d1495eee9b78bfa2ef0360aba3d22ebed6b2c0498241e3c1e6ed8f5a84.jpg)

![c07bdea893536ee7caa4a0b0df9066ddbe59e1cb5f45af4943b2be73e92d7c05.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/tables/c07bdea893536ee7caa4a0b0df9066ddbe59e1cb5f45af4943b2be73e92d7c05.jpg)

![c405cec21b6f03cdd7ac686e0bb323f50ff0d23de7e05eb7778327aee7e992e5.jpg](../iclr_results/1591_Strategist_%20Self-improvement%20of%20LLM%20Decision%20Making%20via%20Bi-Level%20Tree%20Search/tables/c405cec21b6f03cdd7ac686e0bb323f50ff0d23de7e05eb7778327aee7e992e5.jpg)

## Precise Parameter Localization for Textual Generation in Diffusion Models


### Images

![09287552f884e46fb0d36cd4af19bde11d289f3c08ac880b40db7bda62c43b5e.jpg](../iclr_results/1592_Precise%20Parameter%20Localization%20for%20Textual%20Generation%20in%20Diffusion%20Models/images/09287552f884e46fb0d36cd4af19bde11d289f3c08ac880b40db7bda62c43b5e.jpg)

![13f44eeb0abb7ce6628701690348a6a6186300e4518b8cec9b8494ae85fe106a.jpg](../iclr_results/1592_Precise%20Parameter%20Localization%20for%20Textual%20Generation%20in%20Diffusion%20Models/images/13f44eeb0abb7ce6628701690348a6a6186300e4518b8cec9b8494ae85fe106a.jpg)

![27dd19c89c033a6bdbc144eaa6a1e48a7242015eca3dc2df178132529820d529.jpg](../iclr_results/1592_Precise%20Parameter%20Localization%20for%20Textual%20Generation%20in%20Diffusion%20Models/images/27dd19c89c033a6bdbc144eaa6a1e48a7242015eca3dc2df178132529820d529.jpg)

![39151376242008b34e4bf12f2f35cd2527e1746bb80a1efafc6328a32e98ddcb.jpg](../iclr_results/1592_Precise%20Parameter%20Localization%20for%20Textual%20Generation%20in%20Diffusion%20Models/images/39151376242008b34e4bf12f2f35cd2527e1746bb80a1efafc6328a32e98ddcb.jpg)

![4005d3c86e31982d0b3e49437831ac3678708be6fb8f7254dd753a2dee94fcab.jpg](../iclr_results/1592_Precise%20Parameter%20Localization%20for%20Textual%20Generation%20in%20Diffusion%20Models/images/4005d3c86e31982d0b3e49437831ac3678708be6fb8f7254dd753a2dee94fcab.jpg)

![585334d337fcabbdf89547902369e94f2ee8b855c702e26e4289f62748e5799a.jpg](../iclr_results/1592_Precise%20Parameter%20Localization%20for%20Textual%20Generation%20in%20Diffusion%20Models/images/585334d337fcabbdf89547902369e94f2ee8b855c702e26e4289f62748e5799a.jpg)

![788b8987f75116d050d2aa69c0cda806cd5f27b4b518e4e6249d14194723df15.jpg](../iclr_results/1592_Precise%20Parameter%20Localization%20for%20Textual%20Generation%20in%20Diffusion%20Models/images/788b8987f75116d050d2aa69c0cda806cd5f27b4b518e4e6249d14194723df15.jpg)

![8c9ab27380899e68d9367af464a0605869b6951a9fb671f8a5db8d7b3d608476.jpg](../iclr_results/1592_Precise%20Parameter%20Localization%20for%20Textual%20Generation%20in%20Diffusion%20Models/images/8c9ab27380899e68d9367af464a0605869b6951a9fb671f8a5db8d7b3d608476.jpg)

![9531ad6c5eecd7fd5c2b6a78ff2930a2d3e6be5514eb3d483ed08afd3897672f.jpg](../iclr_results/1592_Precise%20Parameter%20Localization%20for%20Textual%20Generation%20in%20Diffusion%20Models/images/9531ad6c5eecd7fd5c2b6a78ff2930a2d3e6be5514eb3d483ed08afd3897672f.jpg)

![a076595975333cecd854b9e96371e118788d850ffbe87e66a9ddef27a1a58a2f.jpg](../iclr_results/1592_Precise%20Parameter%20Localization%20for%20Textual%20Generation%20in%20Diffusion%20Models/images/a076595975333cecd854b9e96371e118788d850ffbe87e66a9ddef27a1a58a2f.jpg)

![a58c0ff4e6e80b1f62e91c047301ec14be1246b30b974ae9592c9a81ecc710a2.jpg](../iclr_results/1592_Precise%20Parameter%20Localization%20for%20Textual%20Generation%20in%20Diffusion%20Models/images/a58c0ff4e6e80b1f62e91c047301ec14be1246b30b974ae9592c9a81ecc710a2.jpg)

![b6dd86b461f5efbf535317ca42a7ab283931b72ce3198ac5a7c7d150269f3b12.jpg](../iclr_results/1592_Precise%20Parameter%20Localization%20for%20Textual%20Generation%20in%20Diffusion%20Models/images/b6dd86b461f5efbf535317ca42a7ab283931b72ce3198ac5a7c7d150269f3b12.jpg)

![de39d28266ecb4554d24bdd06a9e0d94c9add2a89e350adf1a6ab34960a34bfa.jpg](../iclr_results/1592_Precise%20Parameter%20Localization%20for%20Textual%20Generation%20in%20Diffusion%20Models/images/de39d28266ecb4554d24bdd06a9e0d94c9add2a89e350adf1a6ab34960a34bfa.jpg)

![eb6e2f882d8f1b0134f87ed059ccd82fb92316cf3e1e073c77a7db3a63494647.jpg](../iclr_results/1592_Precise%20Parameter%20Localization%20for%20Textual%20Generation%20in%20Diffusion%20Models/images/eb6e2f882d8f1b0134f87ed059ccd82fb92316cf3e1e073c77a7db3a63494647.jpg)

![f039dc6db3397cb25dc9d286665294b07643811f0df1ae36250508dbbe384f90.jpg](../iclr_results/1592_Precise%20Parameter%20Localization%20for%20Textual%20Generation%20in%20Diffusion%20Models/images/f039dc6db3397cb25dc9d286665294b07643811f0df1ae36250508dbbe384f90.jpg)

### Tables

![077496e8ac7c56877a3b9fd2cb65fdbe2a50bf66ec0d288f4363a316a2fceb43.jpg](../iclr_results/1592_Precise%20Parameter%20Localization%20for%20Textual%20Generation%20in%20Diffusion%20Models/tables/077496e8ac7c56877a3b9fd2cb65fdbe2a50bf66ec0d288f4363a316a2fceb43.jpg)

![08e55413a1f95054139ffa883b81e3713162e48315f70dd52a656364f612e02d.jpg](../iclr_results/1592_Precise%20Parameter%20Localization%20for%20Textual%20Generation%20in%20Diffusion%20Models/tables/08e55413a1f95054139ffa883b81e3713162e48315f70dd52a656364f612e02d.jpg)

![0934fa197050e7e4e086330b6513bab5bceae92b8375949397497df210dd7899.jpg](../iclr_results/1592_Precise%20Parameter%20Localization%20for%20Textual%20Generation%20in%20Diffusion%20Models/tables/0934fa197050e7e4e086330b6513bab5bceae92b8375949397497df210dd7899.jpg)

![5700d5b5ebc548f407439b1d268799c9fb08fbb4bcd242dd6e022aa113edd224.jpg](../iclr_results/1592_Precise%20Parameter%20Localization%20for%20Textual%20Generation%20in%20Diffusion%20Models/tables/5700d5b5ebc548f407439b1d268799c9fb08fbb4bcd242dd6e022aa113edd224.jpg)

![628294b01c17bab045d30010e2da37b39ef51197a4fa75e6e91dcd2d8e2f2037.jpg](../iclr_results/1592_Precise%20Parameter%20Localization%20for%20Textual%20Generation%20in%20Diffusion%20Models/tables/628294b01c17bab045d30010e2da37b39ef51197a4fa75e6e91dcd2d8e2f2037.jpg)

![ca23f78302db2c80cd6330fd5475f983f40d63f9fb75aa35e7cee0bb1b1266ac.jpg](../iclr_results/1592_Precise%20Parameter%20Localization%20for%20Textual%20Generation%20in%20Diffusion%20Models/tables/ca23f78302db2c80cd6330fd5475f983f40d63f9fb75aa35e7cee0bb1b1266ac.jpg)

![d8d50cf13cd168bb9be6ad21085dce1dfc5f031be5b31fd07daae9b2db8a9910.jpg](../iclr_results/1592_Precise%20Parameter%20Localization%20for%20Textual%20Generation%20in%20Diffusion%20Models/tables/d8d50cf13cd168bb9be6ad21085dce1dfc5f031be5b31fd07daae9b2db8a9910.jpg)

![e0dd1dfa741211c77b75f3e25b520f49b0e111136d510985a565ff003982a463.jpg](../iclr_results/1592_Precise%20Parameter%20Localization%20for%20Textual%20Generation%20in%20Diffusion%20Models/tables/e0dd1dfa741211c77b75f3e25b520f49b0e111136d510985a565ff003982a463.jpg)

## LIFe-GoM: Generalizable Human Rendering with Learned Iterative Feedback Over Multi-Resolution Gaussians-on-Mesh


### Images

![0b299d6eb87aaeea8a1eb6406bd7336d7bfa29c39cef5175fd0b9cd041febf6b.jpg](../iclr_results/1593_LIFe-GoM_%20Generalizable%20Human%20Rendering%20with%20Learned%20Iterative%20Feedback%20Over%20Multi-Resolution%20Gaussi/images/0b299d6eb87aaeea8a1eb6406bd7336d7bfa29c39cef5175fd0b9cd041febf6b.jpg)

![109480c35775f0432e64c490a4f5c087c859f3ce234bc69518ede77688ea3b53.jpg](../iclr_results/1593_LIFe-GoM_%20Generalizable%20Human%20Rendering%20with%20Learned%20Iterative%20Feedback%20Over%20Multi-Resolution%20Gaussi/images/109480c35775f0432e64c490a4f5c087c859f3ce234bc69518ede77688ea3b53.jpg)

![2aefdc2db09bd4db15c8c96a35ebb1b9c8ddcd30d75547b783dc2981567feef0.jpg](../iclr_results/1593_LIFe-GoM_%20Generalizable%20Human%20Rendering%20with%20Learned%20Iterative%20Feedback%20Over%20Multi-Resolution%20Gaussi/images/2aefdc2db09bd4db15c8c96a35ebb1b9c8ddcd30d75547b783dc2981567feef0.jpg)

![326720898639654a914076a2d22e89870ee46c03ac8e91a2084b368306864bd1.jpg](../iclr_results/1593_LIFe-GoM_%20Generalizable%20Human%20Rendering%20with%20Learned%20Iterative%20Feedback%20Over%20Multi-Resolution%20Gaussi/images/326720898639654a914076a2d22e89870ee46c03ac8e91a2084b368306864bd1.jpg)

![39d6b0247a310966c6f506d386f51cc7db23046870f95bde4a3657fc531ca078.jpg](../iclr_results/1593_LIFe-GoM_%20Generalizable%20Human%20Rendering%20with%20Learned%20Iterative%20Feedback%20Over%20Multi-Resolution%20Gaussi/images/39d6b0247a310966c6f506d386f51cc7db23046870f95bde4a3657fc531ca078.jpg)

![818bf805e0c891c23859e7f7fdfc3dd1736771d3d3b1f58fe0065f799318774e.jpg](../iclr_results/1593_LIFe-GoM_%20Generalizable%20Human%20Rendering%20with%20Learned%20Iterative%20Feedback%20Over%20Multi-Resolution%20Gaussi/images/818bf805e0c891c23859e7f7fdfc3dd1736771d3d3b1f58fe0065f799318774e.jpg)

![93d93f0e966fa7207dd20ed0061fc1295bff48cfc3ce936b7e930a4a90760819.jpg](../iclr_results/1593_LIFe-GoM_%20Generalizable%20Human%20Rendering%20with%20Learned%20Iterative%20Feedback%20Over%20Multi-Resolution%20Gaussi/images/93d93f0e966fa7207dd20ed0061fc1295bff48cfc3ce936b7e930a4a90760819.jpg)

![96ea1bc3b25ef7dd234516f597d3af91862459b59bd34d4e4e4b26d07b7bc050.jpg](../iclr_results/1593_LIFe-GoM_%20Generalizable%20Human%20Rendering%20with%20Learned%20Iterative%20Feedback%20Over%20Multi-Resolution%20Gaussi/images/96ea1bc3b25ef7dd234516f597d3af91862459b59bd34d4e4e4b26d07b7bc050.jpg)

![97ab2abfd3a842632fad3d0bae5d78a323d29cfa8f4744b3d001aa8e58419ac5.jpg](../iclr_results/1593_LIFe-GoM_%20Generalizable%20Human%20Rendering%20with%20Learned%20Iterative%20Feedback%20Over%20Multi-Resolution%20Gaussi/images/97ab2abfd3a842632fad3d0bae5d78a323d29cfa8f4744b3d001aa8e58419ac5.jpg)

![994bddddb30d7dcb712b51b445a0ad9fb1e353d70e240ffcd4855caca868c567.jpg](../iclr_results/1593_LIFe-GoM_%20Generalizable%20Human%20Rendering%20with%20Learned%20Iterative%20Feedback%20Over%20Multi-Resolution%20Gaussi/images/994bddddb30d7dcb712b51b445a0ad9fb1e353d70e240ffcd4855caca868c567.jpg)

![a92d41a6503a03fc8ede9ca42ee471aec98a9cc31de4063f9871282459021ac5.jpg](../iclr_results/1593_LIFe-GoM_%20Generalizable%20Human%20Rendering%20with%20Learned%20Iterative%20Feedback%20Over%20Multi-Resolution%20Gaussi/images/a92d41a6503a03fc8ede9ca42ee471aec98a9cc31de4063f9871282459021ac5.jpg)

![be46b365391c6dc45613f232e3c4b460d66013d7e77ab1735e11a672f9b554c5.jpg](../iclr_results/1593_LIFe-GoM_%20Generalizable%20Human%20Rendering%20with%20Learned%20Iterative%20Feedback%20Over%20Multi-Resolution%20Gaussi/images/be46b365391c6dc45613f232e3c4b460d66013d7e77ab1735e11a672f9b554c5.jpg)

![fca5f8dd11fb954cf987906fd3bd4a3cc811733d1e15e1e775456ee09d2a3880.jpg](../iclr_results/1593_LIFe-GoM_%20Generalizable%20Human%20Rendering%20with%20Learned%20Iterative%20Feedback%20Over%20Multi-Resolution%20Gaussi/images/fca5f8dd11fb954cf987906fd3bd4a3cc811733d1e15e1e775456ee09d2a3880.jpg)

### Tables

![0d0253ec1fc2ed190e89782b9946526f2bcda204e09ff99f875f98f1ae0110e0.jpg](../iclr_results/1593_LIFe-GoM_%20Generalizable%20Human%20Rendering%20with%20Learned%20Iterative%20Feedback%20Over%20Multi-Resolution%20Gaussi/tables/0d0253ec1fc2ed190e89782b9946526f2bcda204e09ff99f875f98f1ae0110e0.jpg)

![13f9ed9478cca98f07921b2d2f3635194ba55d54079acef5cb774304ee771b4d.jpg](../iclr_results/1593_LIFe-GoM_%20Generalizable%20Human%20Rendering%20with%20Learned%20Iterative%20Feedback%20Over%20Multi-Resolution%20Gaussi/tables/13f9ed9478cca98f07921b2d2f3635194ba55d54079acef5cb774304ee771b4d.jpg)

![20343547cb7eb72405cb7a7552841bdc5c90253156a727e4bc2b7078854f01f2.jpg](../iclr_results/1593_LIFe-GoM_%20Generalizable%20Human%20Rendering%20with%20Learned%20Iterative%20Feedback%20Over%20Multi-Resolution%20Gaussi/tables/20343547cb7eb72405cb7a7552841bdc5c90253156a727e4bc2b7078854f01f2.jpg)

![5a5097b0fefaa447527941e54b09aba3c9613e7931c035dd1936da9602208fee.jpg](../iclr_results/1593_LIFe-GoM_%20Generalizable%20Human%20Rendering%20with%20Learned%20Iterative%20Feedback%20Over%20Multi-Resolution%20Gaussi/tables/5a5097b0fefaa447527941e54b09aba3c9613e7931c035dd1936da9602208fee.jpg)

![80ed7327c2bac8c6e0df7423fafb578408f19a916166144f03db7120098bff9d.jpg](../iclr_results/1593_LIFe-GoM_%20Generalizable%20Human%20Rendering%20with%20Learned%20Iterative%20Feedback%20Over%20Multi-Resolution%20Gaussi/tables/80ed7327c2bac8c6e0df7423fafb578408f19a916166144f03db7120098bff9d.jpg)

![900eef712704fabdab37be6b119feac82ca7e36a775a8fb2b9f216ed609a7295.jpg](../iclr_results/1593_LIFe-GoM_%20Generalizable%20Human%20Rendering%20with%20Learned%20Iterative%20Feedback%20Over%20Multi-Resolution%20Gaussi/tables/900eef712704fabdab37be6b119feac82ca7e36a775a8fb2b9f216ed609a7295.jpg)

![b0992a3867a49c531b4da6878a01d3d23d95ba9bec1f85cd208226162e8d3722.jpg](../iclr_results/1593_LIFe-GoM_%20Generalizable%20Human%20Rendering%20with%20Learned%20Iterative%20Feedback%20Over%20Multi-Resolution%20Gaussi/tables/b0992a3867a49c531b4da6878a01d3d23d95ba9bec1f85cd208226162e8d3722.jpg)

![f9d77423781a1a3f7b390a485ff0ba884467cb3e22851b2de7ba4476d0f9105f.jpg](../iclr_results/1593_LIFe-GoM_%20Generalizable%20Human%20Rendering%20with%20Learned%20Iterative%20Feedback%20Over%20Multi-Resolution%20Gaussi/tables/f9d77423781a1a3f7b390a485ff0ba884467cb3e22851b2de7ba4476d0f9105f.jpg)

## MAI: A Multi-turn Aggregation-Iteration Model for Composed Image Retrieval


### Images

![0570c4ea695d17e28f53842f97d1dafae1f56df1f451e57a418d7a372953dadf.jpg](../iclr_results/1594_MAI_%20A%20Multi-turn%20Aggregation-Iteration%20Model%20for%20Composed%20Image%20Retrieval/images/0570c4ea695d17e28f53842f97d1dafae1f56df1f451e57a418d7a372953dadf.jpg)

![14a8703e6dd07584db9b3770bc22f5ea155b1e7265ce0d5a9567f52ea22789dc.jpg](../iclr_results/1594_MAI_%20A%20Multi-turn%20Aggregation-Iteration%20Model%20for%20Composed%20Image%20Retrieval/images/14a8703e6dd07584db9b3770bc22f5ea155b1e7265ce0d5a9567f52ea22789dc.jpg)

![2fb521f9efa2e7a21376476df5ab0d1a4bbe76361f01e92a82f82b258b9e8729.jpg](../iclr_results/1594_MAI_%20A%20Multi-turn%20Aggregation-Iteration%20Model%20for%20Composed%20Image%20Retrieval/images/2fb521f9efa2e7a21376476df5ab0d1a4bbe76361f01e92a82f82b258b9e8729.jpg)

![44f7a56aaec79cd3c88660671a0819a53f8a37ed414ef89c1a10e8ee890b3773.jpg](../iclr_results/1594_MAI_%20A%20Multi-turn%20Aggregation-Iteration%20Model%20for%20Composed%20Image%20Retrieval/images/44f7a56aaec79cd3c88660671a0819a53f8a37ed414ef89c1a10e8ee890b3773.jpg)

![46f8177d9852e1aae9f72a362a171d2c0e59f915da7cf9ba31fb8ff5ecd806f7.jpg](../iclr_results/1594_MAI_%20A%20Multi-turn%20Aggregation-Iteration%20Model%20for%20Composed%20Image%20Retrieval/images/46f8177d9852e1aae9f72a362a171d2c0e59f915da7cf9ba31fb8ff5ecd806f7.jpg)

![79c6534833f8bcda249e3b88cac9c7dc060fcb311585bd4b4cb508cce026c2cd.jpg](../iclr_results/1594_MAI_%20A%20Multi-turn%20Aggregation-Iteration%20Model%20for%20Composed%20Image%20Retrieval/images/79c6534833f8bcda249e3b88cac9c7dc060fcb311585bd4b4cb508cce026c2cd.jpg)

![e9726352b9095e9faaba241a18ed4830b61f950b00668266101eb71dcd1c686f.jpg](../iclr_results/1594_MAI_%20A%20Multi-turn%20Aggregation-Iteration%20Model%20for%20Composed%20Image%20Retrieval/images/e9726352b9095e9faaba241a18ed4830b61f950b00668266101eb71dcd1c686f.jpg)

### Tables

![2030ddec9876b49fec303d9e49e3c1a2525e3dbd9b18a41be30a7fbdbe1a82b3.jpg](../iclr_results/1594_MAI_%20A%20Multi-turn%20Aggregation-Iteration%20Model%20for%20Composed%20Image%20Retrieval/tables/2030ddec9876b49fec303d9e49e3c1a2525e3dbd9b18a41be30a7fbdbe1a82b3.jpg)

![49965dd77de2c7b8d3b54b4b5a6041141df369abbabd743eb77a0d717fb8125f.jpg](../iclr_results/1594_MAI_%20A%20Multi-turn%20Aggregation-Iteration%20Model%20for%20Composed%20Image%20Retrieval/tables/49965dd77de2c7b8d3b54b4b5a6041141df369abbabd743eb77a0d717fb8125f.jpg)

![59f4bdaf39ffa5754d36e2c6b92d4a26227ac91cd7285a14ac19219d93dc9fba.jpg](../iclr_results/1594_MAI_%20A%20Multi-turn%20Aggregation-Iteration%20Model%20for%20Composed%20Image%20Retrieval/tables/59f4bdaf39ffa5754d36e2c6b92d4a26227ac91cd7285a14ac19219d93dc9fba.jpg)

![6eb5441b738254c2c7dff5154b28b23639d8bd8a41d4baa3934202d826b8b70d.jpg](../iclr_results/1594_MAI_%20A%20Multi-turn%20Aggregation-Iteration%20Model%20for%20Composed%20Image%20Retrieval/tables/6eb5441b738254c2c7dff5154b28b23639d8bd8a41d4baa3934202d826b8b70d.jpg)

![6ef1ffa4d4e4e81136a821fa191956c9b986acb49ad8027496257452a03ae4ae.jpg](../iclr_results/1594_MAI_%20A%20Multi-turn%20Aggregation-Iteration%20Model%20for%20Composed%20Image%20Retrieval/tables/6ef1ffa4d4e4e81136a821fa191956c9b986acb49ad8027496257452a03ae4ae.jpg)

![81c0c5aac226011d21297bffa6652c5db85f8d7c7ed4b98dcc0adbaa8af50a93.jpg](../iclr_results/1594_MAI_%20A%20Multi-turn%20Aggregation-Iteration%20Model%20for%20Composed%20Image%20Retrieval/tables/81c0c5aac226011d21297bffa6652c5db85f8d7c7ed4b98dcc0adbaa8af50a93.jpg)

![9fd91fa4f8d1acd5f0f4bb0c022e3963ba3e837aaed94190cd67f4c6e313ba3b.jpg](../iclr_results/1594_MAI_%20A%20Multi-turn%20Aggregation-Iteration%20Model%20for%20Composed%20Image%20Retrieval/tables/9fd91fa4f8d1acd5f0f4bb0c022e3963ba3e837aaed94190cd67f4c6e313ba3b.jpg)

![b79178459218b2e12ac5b1743b714947da5a3332e2245728d0c5bd56174951bb.jpg](../iclr_results/1594_MAI_%20A%20Multi-turn%20Aggregation-Iteration%20Model%20for%20Composed%20Image%20Retrieval/tables/b79178459218b2e12ac5b1743b714947da5a3332e2245728d0c5bd56174951bb.jpg)

![e02679499af2ed8a0d64ca85ee208bc330f8f09feffd183aa903ac18714c7900.jpg](../iclr_results/1594_MAI_%20A%20Multi-turn%20Aggregation-Iteration%20Model%20for%20Composed%20Image%20Retrieval/tables/e02679499af2ed8a0d64ca85ee208bc330f8f09feffd183aa903ac18714c7900.jpg)

![fef2c41e3f85353f803e4d03392f18241ad207abfda96943e63702af8feafd93.jpg](../iclr_results/1594_MAI_%20A%20Multi-turn%20Aggregation-Iteration%20Model%20for%20Composed%20Image%20Retrieval/tables/fef2c41e3f85353f803e4d03392f18241ad207abfda96943e63702af8feafd93.jpg)

## A Truncated Newton Method for Optimal Transport


### Images

![02a7955cd4796351a1a07e0fb6272b9e72d1784ba43315e13b3aa05ad69e36c2.jpg](../iclr_results/1595_A%20Truncated%20Newton%20Method%20for%20Optimal%20Transport/images/02a7955cd4796351a1a07e0fb6272b9e72d1784ba43315e13b3aa05ad69e36c2.jpg)

![1adb0df5d62f195bd88431a647f8883cb84a247ff7288886f15ea2ef5fe5ad9f.jpg](../iclr_results/1595_A%20Truncated%20Newton%20Method%20for%20Optimal%20Transport/images/1adb0df5d62f195bd88431a647f8883cb84a247ff7288886f15ea2ef5fe5ad9f.jpg)

![363d357bb6a2002c976cbbb476184e9a27f7a47f52adc0f54c9cf8d4629f3325.jpg](../iclr_results/1595_A%20Truncated%20Newton%20Method%20for%20Optimal%20Transport/images/363d357bb6a2002c976cbbb476184e9a27f7a47f52adc0f54c9cf8d4629f3325.jpg)

![3ae0410eec7e39c929c07efa80829253b25435da093563a426d27977febea90e.jpg](../iclr_results/1595_A%20Truncated%20Newton%20Method%20for%20Optimal%20Transport/images/3ae0410eec7e39c929c07efa80829253b25435da093563a426d27977febea90e.jpg)

![3fc4cd2197fd6466cafd534fc6625ad2875cce9eb64855a7998b2384b5008060.jpg](../iclr_results/1595_A%20Truncated%20Newton%20Method%20for%20Optimal%20Transport/images/3fc4cd2197fd6466cafd534fc6625ad2875cce9eb64855a7998b2384b5008060.jpg)

![52f14ae6326a9af56dd5eb82c6f3612aeae0611a1f1b3858cf5d1d64b4d59e5c.jpg](../iclr_results/1595_A%20Truncated%20Newton%20Method%20for%20Optimal%20Transport/images/52f14ae6326a9af56dd5eb82c6f3612aeae0611a1f1b3858cf5d1d64b4d59e5c.jpg)

![58a803d90ce2d2952f8e4b1ee208fd0eb42a88c86b099d00e3d51e73d4c5db0f.jpg](../iclr_results/1595_A%20Truncated%20Newton%20Method%20for%20Optimal%20Transport/images/58a803d90ce2d2952f8e4b1ee208fd0eb42a88c86b099d00e3d51e73d4c5db0f.jpg)

![62a66e051b5b8dee99a027d6674682729c23227f91830a29d1572d3d59869ac5.jpg](../iclr_results/1595_A%20Truncated%20Newton%20Method%20for%20Optimal%20Transport/images/62a66e051b5b8dee99a027d6674682729c23227f91830a29d1572d3d59869ac5.jpg)

![82eec6c72b208b292c444a09f2849224119fff239946ffb5c10f44f124d6ac58.jpg](../iclr_results/1595_A%20Truncated%20Newton%20Method%20for%20Optimal%20Transport/images/82eec6c72b208b292c444a09f2849224119fff239946ffb5c10f44f124d6ac58.jpg)

![846f46c9fe3ab9f0b2174b5317ef65385c0c86591c8a9bbbb8766178b6890189.jpg](../iclr_results/1595_A%20Truncated%20Newton%20Method%20for%20Optimal%20Transport/images/846f46c9fe3ab9f0b2174b5317ef65385c0c86591c8a9bbbb8766178b6890189.jpg)

![9c8045d9ba7fe6351a896fcf2088e8ed077334f1869a59fb71e6bd154b18b8a1.jpg](../iclr_results/1595_A%20Truncated%20Newton%20Method%20for%20Optimal%20Transport/images/9c8045d9ba7fe6351a896fcf2088e8ed077334f1869a59fb71e6bd154b18b8a1.jpg)

![a2393495631eeacbb7133dc93c1be3adbcf93c4f09043fcb38ae8828bc83e330.jpg](../iclr_results/1595_A%20Truncated%20Newton%20Method%20for%20Optimal%20Transport/images/a2393495631eeacbb7133dc93c1be3adbcf93c4f09043fcb38ae8828bc83e330.jpg)

![a96fabb1a264337480014b7d6a9466e775057c54e286fe65c2de1b7530591582.jpg](../iclr_results/1595_A%20Truncated%20Newton%20Method%20for%20Optimal%20Transport/images/a96fabb1a264337480014b7d6a9466e775057c54e286fe65c2de1b7530591582.jpg)

![b3ae435deea9a29558b9e0ffc08e5a98add865bd0fb2a255a8d04fd1e41043a9.jpg](../iclr_results/1595_A%20Truncated%20Newton%20Method%20for%20Optimal%20Transport/images/b3ae435deea9a29558b9e0ffc08e5a98add865bd0fb2a255a8d04fd1e41043a9.jpg)

![cf0b67509c33e3c268131dc80dbe2c0f408f239cec794defe2c4650c7d119940.jpg](../iclr_results/1595_A%20Truncated%20Newton%20Method%20for%20Optimal%20Transport/images/cf0b67509c33e3c268131dc80dbe2c0f408f239cec794defe2c4650c7d119940.jpg)

![dcc050461da816753022b0a515df376645449544e07bc0889954da650299951e.jpg](../iclr_results/1595_A%20Truncated%20Newton%20Method%20for%20Optimal%20Transport/images/dcc050461da816753022b0a515df376645449544e07bc0889954da650299951e.jpg)

![e376f3e6e005e827d855fb363eb7fbaa1c19cae93c567c2b4ec22693eb012380.jpg](../iclr_results/1595_A%20Truncated%20Newton%20Method%20for%20Optimal%20Transport/images/e376f3e6e005e827d855fb363eb7fbaa1c19cae93c567c2b4ec22693eb012380.jpg)

![f2ea33c4fc12daef1107039971f527eceef155155c24004a9d938cb95de72b2e.jpg](../iclr_results/1595_A%20Truncated%20Newton%20Method%20for%20Optimal%20Transport/images/f2ea33c4fc12daef1107039971f527eceef155155c24004a9d938cb95de72b2e.jpg)

![fd23e8153344faabf45394fd6ce3f27f3a900caf92e7b3f37f9431c58a758721.jpg](../iclr_results/1595_A%20Truncated%20Newton%20Method%20for%20Optimal%20Transport/images/fd23e8153344faabf45394fd6ce3f27f3a900caf92e7b3f37f9431c58a758721.jpg)

### Tables

![07716d1750e8d22266e9d22ae4da8433b7d9a6a346ada63dc716dcfa994e8e95.jpg](../iclr_results/1595_A%20Truncated%20Newton%20Method%20for%20Optimal%20Transport/tables/07716d1750e8d22266e9d22ae4da8433b7d9a6a346ada63dc716dcfa994e8e95.jpg)

![a87d7129614e2bf37d94289652dd7f735bc6a41f2409b8d99e2182019d4193db.jpg](../iclr_results/1595_A%20Truncated%20Newton%20Method%20for%20Optimal%20Transport/tables/a87d7129614e2bf37d94289652dd7f735bc6a41f2409b8d99e2182019d4193db.jpg)

![c2f8ee5c825d7668446ce9ffef3763bf15e7efe35f548dcfc55a806bad9c8f7a.jpg](../iclr_results/1595_A%20Truncated%20Newton%20Method%20for%20Optimal%20Transport/tables/c2f8ee5c825d7668446ce9ffef3763bf15e7efe35f548dcfc55a806bad9c8f7a.jpg)

![d008dd718a028838af8d8ff26806eeb4691117a63ab74fde13045210d8309e2e.jpg](../iclr_results/1595_A%20Truncated%20Newton%20Method%20for%20Optimal%20Transport/tables/d008dd718a028838af8d8ff26806eeb4691117a63ab74fde13045210d8309e2e.jpg)

## Depth Any Video with Scalable Synthetic Data


### Images

![3b850d10b72e42850afe44f981bdce553b55707a762d4d625da7d54d8282e44e.jpg](../iclr_results/1596_Depth%20Any%20Video%20with%20Scalable%20Synthetic%20Data/images/3b850d10b72e42850afe44f981bdce553b55707a762d4d625da7d54d8282e44e.jpg)

![404726edb31dc83dc3e6df5cd8ff754b60bfc7bbe99a39f8fe47830ef47cb5fd.jpg](../iclr_results/1596_Depth%20Any%20Video%20with%20Scalable%20Synthetic%20Data/images/404726edb31dc83dc3e6df5cd8ff754b60bfc7bbe99a39f8fe47830ef47cb5fd.jpg)

![9da0f77e9a98ef5eb00df70bc6085e5d63bea8668edbef8e069b09627ef24e45.jpg](../iclr_results/1596_Depth%20Any%20Video%20with%20Scalable%20Synthetic%20Data/images/9da0f77e9a98ef5eb00df70bc6085e5d63bea8668edbef8e069b09627ef24e45.jpg)

![c8f8a5db0489969ea08d352d991c43beb2172c87372aadc3322b06270d4f1991.jpg](../iclr_results/1596_Depth%20Any%20Video%20with%20Scalable%20Synthetic%20Data/images/c8f8a5db0489969ea08d352d991c43beb2172c87372aadc3322b06270d4f1991.jpg)

![e009e6fd3032d5e6d8afbf4bbda848cd27fd48d5a23dabd8c8044370ac54e29c.jpg](../iclr_results/1596_Depth%20Any%20Video%20with%20Scalable%20Synthetic%20Data/images/e009e6fd3032d5e6d8afbf4bbda848cd27fd48d5a23dabd8c8044370ac54e29c.jpg)

![e8f3276384f5e00cc5790905baa15c83c29ce94e3fdc0de7d327a5a863c30082.jpg](../iclr_results/1596_Depth%20Any%20Video%20with%20Scalable%20Synthetic%20Data/images/e8f3276384f5e00cc5790905baa15c83c29ce94e3fdc0de7d327a5a863c30082.jpg)

![fa25717de37efb533915862dac99cb4c9d3610f4080497a8558399a16a4f2d64.jpg](../iclr_results/1596_Depth%20Any%20Video%20with%20Scalable%20Synthetic%20Data/images/fa25717de37efb533915862dac99cb4c9d3610f4080497a8558399a16a4f2d64.jpg)

### Tables

![05895f3b2f0814ac92a49173f049193e69903643d8c475bb9e25c79f72e48327.jpg](../iclr_results/1596_Depth%20Any%20Video%20with%20Scalable%20Synthetic%20Data/tables/05895f3b2f0814ac92a49173f049193e69903643d8c475bb9e25c79f72e48327.jpg)

![4ef949e0e2a09768c2dd83df95f6bd0a82f74d92d0d2e1ef3689dc6e6652b4f8.jpg](../iclr_results/1596_Depth%20Any%20Video%20with%20Scalable%20Synthetic%20Data/tables/4ef949e0e2a09768c2dd83df95f6bd0a82f74d92d0d2e1ef3689dc6e6652b4f8.jpg)

![756afe4f86ad58b05c2871fa91dd4b60fc8091bd38ac0c52e8433cc32f01104b.jpg](../iclr_results/1596_Depth%20Any%20Video%20with%20Scalable%20Synthetic%20Data/tables/756afe4f86ad58b05c2871fa91dd4b60fc8091bd38ac0c52e8433cc32f01104b.jpg)

![db52e22da3841eb9e07bd582b32be4c8332683f7f18ae06cc2aae4f1debcf9b6.jpg](../iclr_results/1596_Depth%20Any%20Video%20with%20Scalable%20Synthetic%20Data/tables/db52e22da3841eb9e07bd582b32be4c8332683f7f18ae06cc2aae4f1debcf9b6.jpg)

![f05763cde76c4e9517a98d0fb78686a03c9a2396319e43786cdd1516c52be806.jpg](../iclr_results/1596_Depth%20Any%20Video%20with%20Scalable%20Synthetic%20Data/tables/f05763cde76c4e9517a98d0fb78686a03c9a2396319e43786cdd1516c52be806.jpg)

![f6adebe5d6033d645c69f5427eb58fb6c5418c2deb1c45d5ba104a1351473d1e.jpg](../iclr_results/1596_Depth%20Any%20Video%20with%20Scalable%20Synthetic%20Data/tables/f6adebe5d6033d645c69f5427eb58fb6c5418c2deb1c45d5ba104a1351473d1e.jpg)

## Residual-MPPI: Online Policy Customization for Continuous Control


### Images

![158d6dc291d4c8fc45a4634d45ba0b556fe24190703d0a45587b23e5986eb094.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/images/158d6dc291d4c8fc45a4634d45ba0b556fe24190703d0a45587b23e5986eb094.jpg)

![384058b4c35bfc97d6741fa34ec2e1983bfbab5c0f41517b63414a85048310ab.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/images/384058b4c35bfc97d6741fa34ec2e1983bfbab5c0f41517b63414a85048310ab.jpg)

![3e2e8dc376912e5c90b4d4560b25b34ce023b9e7e0fc77eb5d08c8dfc751ffff.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/images/3e2e8dc376912e5c90b4d4560b25b34ce023b9e7e0fc77eb5d08c8dfc751ffff.jpg)

![442ac951dad226a3e661bb5d9554b7742ea3a3711f17117d2db0306343173a0c.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/images/442ac951dad226a3e661bb5d9554b7742ea3a3711f17117d2db0306343173a0c.jpg)

![623efd6b1726f994b939c57944f59586ac88643a1acf5a3e6286cd278fd4348f.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/images/623efd6b1726f994b939c57944f59586ac88643a1acf5a3e6286cd278fd4348f.jpg)

![64ff2bff0d342d7173ae4f65db20c2f217a50d5b9ecc608e9faa6a6717a26108.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/images/64ff2bff0d342d7173ae4f65db20c2f217a50d5b9ecc608e9faa6a6717a26108.jpg)

![6c83856cb7dc7c40077e3ef9810a95767ad0cd54167ae5135e2d441556dccf05.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/images/6c83856cb7dc7c40077e3ef9810a95767ad0cd54167ae5135e2d441556dccf05.jpg)

![85392a23969cda101ac7cb602d55699ad85210435823f728d19a2568be0f8158.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/images/85392a23969cda101ac7cb602d55699ad85210435823f728d19a2568be0f8158.jpg)

![98c207a915fc48ef68e610cc94b26cea740147096edd5c45179064c2edd9ecd9.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/images/98c207a915fc48ef68e610cc94b26cea740147096edd5c45179064c2edd9ecd9.jpg)

![a44213d55ed6fccbb311efeb57f9d50e9bb0bf68a71ddaed21ca0fc0014a1420.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/images/a44213d55ed6fccbb311efeb57f9d50e9bb0bf68a71ddaed21ca0fc0014a1420.jpg)

![a5a0384ea49c66209b10678fa24b9c9bc36bc46528c50cf0cc3d7994ed02bb33.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/images/a5a0384ea49c66209b10678fa24b9c9bc36bc46528c50cf0cc3d7994ed02bb33.jpg)

![a795755b5c46938b0d269af34d2be323bdd6adbde03518fffe9cd2ecfb35f3bb.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/images/a795755b5c46938b0d269af34d2be323bdd6adbde03518fffe9cd2ecfb35f3bb.jpg)

![cab86477be4823d40f1acb6ab0808fe1480a3de0869e821f1317e63ab5b2fdab.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/images/cab86477be4823d40f1acb6ab0808fe1480a3de0869e821f1317e63ab5b2fdab.jpg)

![cb2e82e5936f177f61288849a4f35248d254509278800c8569caab6f222fe64d.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/images/cb2e82e5936f177f61288849a4f35248d254509278800c8569caab6f222fe64d.jpg)

![d5baead9b2454a606ff3b54ae176982fdfabc53c83e3f1bc10f9cb8e8aee85d8.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/images/d5baead9b2454a606ff3b54ae176982fdfabc53c83e3f1bc10f9cb8e8aee85d8.jpg)

![d66e1b17272ac2211bff1ea19793bf9c8b99046bdd129e46107443da1452453c.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/images/d66e1b17272ac2211bff1ea19793bf9c8b99046bdd129e46107443da1452453c.jpg)

![fd08b4a56c9169c2c9018a4f5d7afda2d7b1339f53dabfd4b1a2f698d3e7c3b8.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/images/fd08b4a56c9169c2c9018a4f5d7afda2d7b1339f53dabfd4b1a2f698d3e7c3b8.jpg)

### Tables

![07feec3b6f9978426ae4e6d13fe27bf53dd6e7b163db125208d8f0a1bc5ccf19.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/tables/07feec3b6f9978426ae4e6d13fe27bf53dd6e7b163db125208d8f0a1bc5ccf19.jpg)

![0876af44e12e5b09ab4cbb66d9ea5fec018d45209071770aa859e5a94b14b0f9.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/tables/0876af44e12e5b09ab4cbb66d9ea5fec018d45209071770aa859e5a94b14b0f9.jpg)

![10692772d18bce94886d08c8734afa0c699e1a6cb2e3d43e3c0772df2fd6488b.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/tables/10692772d18bce94886d08c8734afa0c699e1a6cb2e3d43e3c0772df2fd6488b.jpg)

![25e779fa1ee93c74f06f636359f1eef8998a877335279fd7b9b05460ac6d25e6.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/tables/25e779fa1ee93c74f06f636359f1eef8998a877335279fd7b9b05460ac6d25e6.jpg)

![2acfae9c6a17d239b2f4534ff07fa4a04fec4b3021862dc4290da891d814e290.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/tables/2acfae9c6a17d239b2f4534ff07fa4a04fec4b3021862dc4290da891d814e290.jpg)

![582b596e504f85b79146b59c023e758dc00577301a2df99b60bce0ce949d6ea2.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/tables/582b596e504f85b79146b59c023e758dc00577301a2df99b60bce0ce949d6ea2.jpg)

![5baab6c1a89efc2f4b23a0686670fed3c15aab04e0f95b753c43a2eee06cbc23.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/tables/5baab6c1a89efc2f4b23a0686670fed3c15aab04e0f95b753c43a2eee06cbc23.jpg)

![5cac8ad310a7e533aa13cba4d7e42741a79333152073976e60d6c553ffe6187a.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/tables/5cac8ad310a7e533aa13cba4d7e42741a79333152073976e60d6c553ffe6187a.jpg)

![6e1a4706c0ce1ea652fbb14c1ca909f194062551c52b5ca4d34bab3295a977f5.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/tables/6e1a4706c0ce1ea652fbb14c1ca909f194062551c52b5ca4d34bab3295a977f5.jpg)

![a5567849008c81cf937b5c13ae8e7b051a2ca0d58f6c9cb7c7a0bc134a438ff2.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/tables/a5567849008c81cf937b5c13ae8e7b051a2ca0d58f6c9cb7c7a0bc134a438ff2.jpg)

![b192edfc4a98a5f2d2890dcf9524703aaa0438400bbe9bc35b62c3450525e64e.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/tables/b192edfc4a98a5f2d2890dcf9524703aaa0438400bbe9bc35b62c3450525e64e.jpg)

![bce913cad0552294c0df3081883bc5552553cb03dab85d453b9152307262acf7.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/tables/bce913cad0552294c0df3081883bc5552553cb03dab85d453b9152307262acf7.jpg)

![c0cb8a3e6c9f7902be380e61d21bc8887d2d658a5e751494548391dbc03e8398.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/tables/c0cb8a3e6c9f7902be380e61d21bc8887d2d658a5e751494548391dbc03e8398.jpg)

![d37c8cfe7b0a254f4839a55148828042e1449a8b6a1ffb7dbf11293c7b0db58f.jpg](../iclr_results/1597_Residual-MPPI_%20Online%20Policy%20Customization%20for%20Continuous%20Control/tables/d37c8cfe7b0a254f4839a55148828042e1449a8b6a1ffb7dbf11293c7b0db58f.jpg)

## Efficient Biological Data Acquisition through Inference Set Design


### Images

![11865bac7e93ce76066334b8de60a46e746d89b28b48fc260c959dde1cb5efc5.jpg](../iclr_results/1598_Efficient%20Biological%20Data%20Acquisition%20through%20Inference%20Set%20Design/images/11865bac7e93ce76066334b8de60a46e746d89b28b48fc260c959dde1cb5efc5.jpg)

![7af74f7f73766bd5e8e7b8d105f88e0014a16d904ae2ac1c21ede9b5d141f24e.jpg](../iclr_results/1598_Efficient%20Biological%20Data%20Acquisition%20through%20Inference%20Set%20Design/images/7af74f7f73766bd5e8e7b8d105f88e0014a16d904ae2ac1c21ede9b5d141f24e.jpg)

![7e1b891f6662dc4eb807483bd1aaf5cded2be4066703f297b42d04f5cc671b2d.jpg](../iclr_results/1598_Efficient%20Biological%20Data%20Acquisition%20through%20Inference%20Set%20Design/images/7e1b891f6662dc4eb807483bd1aaf5cded2be4066703f297b42d04f5cc671b2d.jpg)

![89fb5a69f5038867706e108263f30709464ef2e0d69b1ebcbfb43cb0e8550241.jpg](../iclr_results/1598_Efficient%20Biological%20Data%20Acquisition%20through%20Inference%20Set%20Design/images/89fb5a69f5038867706e108263f30709464ef2e0d69b1ebcbfb43cb0e8550241.jpg)

![8c9f9bb2e5fe6c059a3099b351bf8553bef4b8d98524b6892c225d68609ec3b5.jpg](../iclr_results/1598_Efficient%20Biological%20Data%20Acquisition%20through%20Inference%20Set%20Design/images/8c9f9bb2e5fe6c059a3099b351bf8553bef4b8d98524b6892c225d68609ec3b5.jpg)

![a76af1873934bff2769e5c428a1ac9a57cf072c6b7e030cc902830e456a1a816.jpg](../iclr_results/1598_Efficient%20Biological%20Data%20Acquisition%20through%20Inference%20Set%20Design/images/a76af1873934bff2769e5c428a1ac9a57cf072c6b7e030cc902830e456a1a816.jpg)

![b5dcb1c749c58dd8d477e73a0b14117f1b9e040d147142619520bc7055d11557.jpg](../iclr_results/1598_Efficient%20Biological%20Data%20Acquisition%20through%20Inference%20Set%20Design/images/b5dcb1c749c58dd8d477e73a0b14117f1b9e040d147142619520bc7055d11557.jpg)

![b962426ace7ea6b4ccff44adb3b9fa380bc4547fb1236f7db0cd364eb4aa9842.jpg](../iclr_results/1598_Efficient%20Biological%20Data%20Acquisition%20through%20Inference%20Set%20Design/images/b962426ace7ea6b4ccff44adb3b9fa380bc4547fb1236f7db0cd364eb4aa9842.jpg)

![bf3a51c32d2c273536c33b7aa306aff960fd772c4f11e76d1c4c6ff07353c408.jpg](../iclr_results/1598_Efficient%20Biological%20Data%20Acquisition%20through%20Inference%20Set%20Design/images/bf3a51c32d2c273536c33b7aa306aff960fd772c4f11e76d1c4c6ff07353c408.jpg)

![cd83867482b5f0889e2d8d3377f446311833c58e52a058864f2148e93c4061e1.jpg](../iclr_results/1598_Efficient%20Biological%20Data%20Acquisition%20through%20Inference%20Set%20Design/images/cd83867482b5f0889e2d8d3377f446311833c58e52a058864f2148e93c4061e1.jpg)

![e607d11bfc7155514a12a14388bd0bb808fa817468989f40756df13d70b7d3dd.jpg](../iclr_results/1598_Efficient%20Biological%20Data%20Acquisition%20through%20Inference%20Set%20Design/images/e607d11bfc7155514a12a14388bd0bb808fa817468989f40756df13d70b7d3dd.jpg)

![ef0eeceae1aadea002336949651b510ede8223659e557f7d67e1e89ac18ddecb.jpg](../iclr_results/1598_Efficient%20Biological%20Data%20Acquisition%20through%20Inference%20Set%20Design/images/ef0eeceae1aadea002336949651b510ede8223659e557f7d67e1e89ac18ddecb.jpg)

![ef146886eebe789a06cd3be2b5ca42e2b3feefc63bbeba755f567d74af69921e.jpg](../iclr_results/1598_Efficient%20Biological%20Data%20Acquisition%20through%20Inference%20Set%20Design/images/ef146886eebe789a06cd3be2b5ca42e2b3feefc63bbeba755f567d74af69921e.jpg)

![f820d13ae19811b7b404067c492a0a90f2db5cb11d8a1c1f6e9a771fad13b036.jpg](../iclr_results/1598_Efficient%20Biological%20Data%20Acquisition%20through%20Inference%20Set%20Design/images/f820d13ae19811b7b404067c492a0a90f2db5cb11d8a1c1f6e9a771fad13b036.jpg)

### Tables

![9c8bd84506205c38d78f8ad6035517c7ecf37a73c340547666162945319d28ff.jpg](../iclr_results/1598_Efficient%20Biological%20Data%20Acquisition%20through%20Inference%20Set%20Design/tables/9c8bd84506205c38d78f8ad6035517c7ecf37a73c340547666162945319d28ff.jpg)

![a4366b4c1d876d92cfd95fb20ed45ddf10c6e4f1d6f0ee42527b25a0cd18b85b.jpg](../iclr_results/1598_Efficient%20Biological%20Data%20Acquisition%20through%20Inference%20Set%20Design/tables/a4366b4c1d876d92cfd95fb20ed45ddf10c6e4f1d6f0ee42527b25a0cd18b85b.jpg)

## Dynamic Loss-Based Sample Reweighting for Improved Large Language Model Pretraining


### Images

![49959135649d393fee1766992fed4331bf0535239abdc650cc6e4db48b424384.jpg](../iclr_results/1599_Dynamic%20Loss-Based%20Sample%20Reweighting%20for%20Improved%20Large%20Language%20Model%20Pretraining/images/49959135649d393fee1766992fed4331bf0535239abdc650cc6e4db48b424384.jpg)

![6da1d1da531c8cd53259f400a0e187580862bf8892e05bca39351f992b7e459e.jpg](../iclr_results/1599_Dynamic%20Loss-Based%20Sample%20Reweighting%20for%20Improved%20Large%20Language%20Model%20Pretraining/images/6da1d1da531c8cd53259f400a0e187580862bf8892e05bca39351f992b7e459e.jpg)

![7be73a4dc0392d13ec71b6099e4dcaf7abbf2c4e5c4cf685d100465102370340.jpg](../iclr_results/1599_Dynamic%20Loss-Based%20Sample%20Reweighting%20for%20Improved%20Large%20Language%20Model%20Pretraining/images/7be73a4dc0392d13ec71b6099e4dcaf7abbf2c4e5c4cf685d100465102370340.jpg)

![7bff958b2572be222ea9ae9461894058c6d5c9b31a5cccbbc98c29d61609baba.jpg](../iclr_results/1599_Dynamic%20Loss-Based%20Sample%20Reweighting%20for%20Improved%20Large%20Language%20Model%20Pretraining/images/7bff958b2572be222ea9ae9461894058c6d5c9b31a5cccbbc98c29d61609baba.jpg)

![8d81044fdaede0de0cf0f4443fe1a5d9ffe7d4cd0eebeea23f51a9b40d13c59e.jpg](../iclr_results/1599_Dynamic%20Loss-Based%20Sample%20Reweighting%20for%20Improved%20Large%20Language%20Model%20Pretraining/images/8d81044fdaede0de0cf0f4443fe1a5d9ffe7d4cd0eebeea23f51a9b40d13c59e.jpg)

![9e32e8da3d323cf2026024b6daa077f40daaa0563e4691e363ebf053aa169d9c.jpg](../iclr_results/1599_Dynamic%20Loss-Based%20Sample%20Reweighting%20for%20Improved%20Large%20Language%20Model%20Pretraining/images/9e32e8da3d323cf2026024b6daa077f40daaa0563e4691e363ebf053aa169d9c.jpg)

![9e834a6bac0622315dd2efdcab0721ecfc259f8560033961284c210ed347ef58.jpg](../iclr_results/1599_Dynamic%20Loss-Based%20Sample%20Reweighting%20for%20Improved%20Large%20Language%20Model%20Pretraining/images/9e834a6bac0622315dd2efdcab0721ecfc259f8560033961284c210ed347ef58.jpg)

![d3849fab75f8a6dcfefed73bf7b0497ee85e53b3f506e2a708f15ba6909dc9fc.jpg](../iclr_results/1599_Dynamic%20Loss-Based%20Sample%20Reweighting%20for%20Improved%20Large%20Language%20Model%20Pretraining/images/d3849fab75f8a6dcfefed73bf7b0497ee85e53b3f506e2a708f15ba6909dc9fc.jpg)

### Tables

![053f43a3aa4671994fd43e22b741a2e52bd068c1a648e9278e9650640f1fa45f.jpg](../iclr_results/1599_Dynamic%20Loss-Based%20Sample%20Reweighting%20for%20Improved%20Large%20Language%20Model%20Pretraining/tables/053f43a3aa4671994fd43e22b741a2e52bd068c1a648e9278e9650640f1fa45f.jpg)

![05c4d919dcae462e53fb6e106019fdd6b22783f57c53f86d3efe5ace742a3633.jpg](../iclr_results/1599_Dynamic%20Loss-Based%20Sample%20Reweighting%20for%20Improved%20Large%20Language%20Model%20Pretraining/tables/05c4d919dcae462e53fb6e106019fdd6b22783f57c53f86d3efe5ace742a3633.jpg)

![06c3cbaac6635e4e7acfe641eea7ee7de263cff6d25ca856657512a63aadf132.jpg](../iclr_results/1599_Dynamic%20Loss-Based%20Sample%20Reweighting%20for%20Improved%20Large%20Language%20Model%20Pretraining/tables/06c3cbaac6635e4e7acfe641eea7ee7de263cff6d25ca856657512a63aadf132.jpg)

![072c7edbb47b9eb2b6f1cd47656ac4fcc39817201b71bbbbeae248524261b8b8.jpg](../iclr_results/1599_Dynamic%20Loss-Based%20Sample%20Reweighting%20for%20Improved%20Large%20Language%20Model%20Pretraining/tables/072c7edbb47b9eb2b6f1cd47656ac4fcc39817201b71bbbbeae248524261b8b8.jpg)

![11548b9dc9aaeaaa3ae54551eedfeb580a9b47ed01c73f838b4353979175d28d.jpg](../iclr_results/1599_Dynamic%20Loss-Based%20Sample%20Reweighting%20for%20Improved%20Large%20Language%20Model%20Pretraining/tables/11548b9dc9aaeaaa3ae54551eedfeb580a9b47ed01c73f838b4353979175d28d.jpg)

![2aa06bd24152016b80fda85b5d3c59a0a8e9d89281961d5956774425ea17da2c.jpg](../iclr_results/1599_Dynamic%20Loss-Based%20Sample%20Reweighting%20for%20Improved%20Large%20Language%20Model%20Pretraining/tables/2aa06bd24152016b80fda85b5d3c59a0a8e9d89281961d5956774425ea17da2c.jpg)

![65443b071a185024f60a43d9c4dee467656d0e2d8e7957e3b09d3b06817a92ab.jpg](../iclr_results/1599_Dynamic%20Loss-Based%20Sample%20Reweighting%20for%20Improved%20Large%20Language%20Model%20Pretraining/tables/65443b071a185024f60a43d9c4dee467656d0e2d8e7957e3b09d3b06817a92ab.jpg)

![7a1aeff2a1af2276a670364b0ae4a41e9a2ae390da5b8e8ec062c581f866633c.jpg](../iclr_results/1599_Dynamic%20Loss-Based%20Sample%20Reweighting%20for%20Improved%20Large%20Language%20Model%20Pretraining/tables/7a1aeff2a1af2276a670364b0ae4a41e9a2ae390da5b8e8ec062c581f866633c.jpg)

![a081fca5393af761b017ad78f7fabf14757d911b915f8276c2f4af351b11dcda.jpg](../iclr_results/1599_Dynamic%20Loss-Based%20Sample%20Reweighting%20for%20Improved%20Large%20Language%20Model%20Pretraining/tables/a081fca5393af761b017ad78f7fabf14757d911b915f8276c2f4af351b11dcda.jpg)

![b49e42c74a1dc7f02a2dd90e50221c62aaefb2d4797a11b03714a990945c90a0.jpg](../iclr_results/1599_Dynamic%20Loss-Based%20Sample%20Reweighting%20for%20Improved%20Large%20Language%20Model%20Pretraining/tables/b49e42c74a1dc7f02a2dd90e50221c62aaefb2d4797a11b03714a990945c90a0.jpg)

![d69f99753b01e06f5a404827c1becb956a4a190498d216d381d178bb877448e9.jpg](../iclr_results/1599_Dynamic%20Loss-Based%20Sample%20Reweighting%20for%20Improved%20Large%20Language%20Model%20Pretraining/tables/d69f99753b01e06f5a404827c1becb956a4a190498d216d381d178bb877448e9.jpg)

![d8579663ac3eb6251164db167977d8464a7cfe4a953090167456b9921946df6f.jpg](../iclr_results/1599_Dynamic%20Loss-Based%20Sample%20Reweighting%20for%20Improved%20Large%20Language%20Model%20Pretraining/tables/d8579663ac3eb6251164db167977d8464a7cfe4a953090167456b9921946df6f.jpg)

## RFWave: Multi-band Rectified Flow for Audio Waveform Reconstruction


### Images

![362357b23568037aca97c03f46348db9617495c892aabac61a37bde0b80bf97e.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/images/362357b23568037aca97c03f46348db9617495c892aabac61a37bde0b80bf97e.jpg)

![40979fc02f8d9e788aeaf2bb0021f35d892cfd7973f0f9747d3e7a02bb2cc7fd.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/images/40979fc02f8d9e788aeaf2bb0021f35d892cfd7973f0f9747d3e7a02bb2cc7fd.jpg)

![497ee34e16c5e4a4c9d7dfb6a88690dfe12e1196314247c2ed508d03ee492841.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/images/497ee34e16c5e4a4c9d7dfb6a88690dfe12e1196314247c2ed508d03ee492841.jpg)

![6531910dfa36077e51857fab6cac7bb97e322f5e03c0ebec3ae4551e50de1816.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/images/6531910dfa36077e51857fab6cac7bb97e322f5e03c0ebec3ae4551e50de1816.jpg)

![6e0e2b898ae9a5d131dd2ad7d82fb1088fa2f686c43a3fa50e1966c8350d1733.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/images/6e0e2b898ae9a5d131dd2ad7d82fb1088fa2f686c43a3fa50e1966c8350d1733.jpg)

![7f5d866ad6d1101e4d01f2f1039b323fe67741dfdf9cd9b10868db961c21cb91.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/images/7f5d866ad6d1101e4d01f2f1039b323fe67741dfdf9cd9b10868db961c21cb91.jpg)

![85810528e7c981bcd90d1d886e1176c92eed2f8ce0dffe31f8ebe75804e16661.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/images/85810528e7c981bcd90d1d886e1176c92eed2f8ce0dffe31f8ebe75804e16661.jpg)

![95018c351f06e1e26a1af63613ebcd51ec51f16074323039f52b85fec7e133a6.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/images/95018c351f06e1e26a1af63613ebcd51ec51f16074323039f52b85fec7e133a6.jpg)

![a8e2ed1d4049bf12af86bc7e91ebdc825dc34ade67714f685fd80a42711c4d15.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/images/a8e2ed1d4049bf12af86bc7e91ebdc825dc34ade67714f685fd80a42711c4d15.jpg)

![b6bc2d28af55e2972510b455fbe1223b365d96a9e4e3fba093ed40abb070381e.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/images/b6bc2d28af55e2972510b455fbe1223b365d96a9e4e3fba093ed40abb070381e.jpg)

![c332fc612466fc4ad1a0067b734f9db73e99a48d886207ed15257530169132ee.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/images/c332fc612466fc4ad1a0067b734f9db73e99a48d886207ed15257530169132ee.jpg)

![dd2aeb9f061c6dc43592a55d0b0f9000f09fb870e78ecce12607186393e8c0d3.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/images/dd2aeb9f061c6dc43592a55d0b0f9000f09fb870e78ecce12607186393e8c0d3.jpg)

![df84601ea3b9f969efca2e057f8a47600441ebb8646dcc16096a67e270ccd6c6.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/images/df84601ea3b9f969efca2e057f8a47600441ebb8646dcc16096a67e270ccd6c6.jpg)

![f7c3bb7497bdf8bbe9f54a8467e64ac73fe6d7cb0866f2b6550b060a1ff3cf9b.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/images/f7c3bb7497bdf8bbe9f54a8467e64ac73fe6d7cb0866f2b6550b060a1ff3cf9b.jpg)

### Tables

![3b027fe2aae29175fc38e1d59599b8ac4d7b7616ec235265e7ba25497112a2b2.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/tables/3b027fe2aae29175fc38e1d59599b8ac4d7b7616ec235265e7ba25497112a2b2.jpg)

![567b20329893f3a53c3ff9b3837d9c3977398c5c013cfc24605f2d7b4c85743c.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/tables/567b20329893f3a53c3ff9b3837d9c3977398c5c013cfc24605f2d7b4c85743c.jpg)

![6200f8c2a6bffe4a15298e1ece6c6fce93690241d984081ba16a825c72d68b8d.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/tables/6200f8c2a6bffe4a15298e1ece6c6fce93690241d984081ba16a825c72d68b8d.jpg)

![8465af5a98bade0ce7ba2fb64d22bfd2bd0e37407429efa4b191e6986fd8f361.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/tables/8465af5a98bade0ce7ba2fb64d22bfd2bd0e37407429efa4b191e6986fd8f361.jpg)

![88e2af9751f0f1eac3ca3ad307af8c68f91b91c4913745c2e8c32ebf166417d0.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/tables/88e2af9751f0f1eac3ca3ad307af8c68f91b91c4913745c2e8c32ebf166417d0.jpg)

![a7bff57d1ce3f4b6dae77925341c8b4c688720244b63f8b7d1612c2a91b52206.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/tables/a7bff57d1ce3f4b6dae77925341c8b4c688720244b63f8b7d1612c2a91b52206.jpg)

![b18fd3e83d2d530af7beff1ac69553821f2e8628abae6734973e285427859363.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/tables/b18fd3e83d2d530af7beff1ac69553821f2e8628abae6734973e285427859363.jpg)

![b79a5dd5e1d45e99eef086fe603e45a5f31537c914440bb0038eb7817a991043.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/tables/b79a5dd5e1d45e99eef086fe603e45a5f31537c914440bb0038eb7817a991043.jpg)

![bc29cc05b2b0b42558bd2229ff0a19a077d13c6b4fa066ad8cc3107eef9db7da.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/tables/bc29cc05b2b0b42558bd2229ff0a19a077d13c6b4fa066ad8cc3107eef9db7da.jpg)

![c2c9494e986f4758ab0fc793032c2a43350221f0fedc0e7ef0328f8d6f41dfd9.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/tables/c2c9494e986f4758ab0fc793032c2a43350221f0fedc0e7ef0328f8d6f41dfd9.jpg)

![c3b455b11568e1b7d1110ec7dd4b3ccdfe9e2e124760cc8af2f7e76177ec59d2.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/tables/c3b455b11568e1b7d1110ec7dd4b3ccdfe9e2e124760cc8af2f7e76177ec59d2.jpg)

![dc41258a310ee8de00c52e8897eb4dd4ed476cef394980065a715f2889c98528.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/tables/dc41258a310ee8de00c52e8897eb4dd4ed476cef394980065a715f2889c98528.jpg)

![eca50464b2a2cff2d3d5d09d91df392a7c2e20ded2c8dfcccbc518719b8b52e7.jpg](../iclr_results/1600_RFWave_%20Multi-band%20Rectified%20Flow%20for%20Audio%20Waveform%20Reconstruction/tables/eca50464b2a2cff2d3d5d09d91df392a7c2e20ded2c8dfcccbc518719b8b52e7.jpg)

## Rare event modeling with self-regularized normalizing flows: what can we learn from a single failure?


### Images

![0e9c1f050517e7f984c99fb3cb67205b6d3ce2ae690afc61bc4abe7ad9c70e0a.jpg](../iclr_results/1601_Rare%20event%20modeling%20with%20self-regularized%20normalizing%20flows_%20what%20can%20we%20learn%20from%20a%20single%20failure/images/0e9c1f050517e7f984c99fb3cb67205b6d3ce2ae690afc61bc4abe7ad9c70e0a.jpg)

![1c81c1d99a2b06460a4cab922d7a9a7bf3ec7e6e12590f5ebd6e7121de9eb3a5.jpg](../iclr_results/1601_Rare%20event%20modeling%20with%20self-regularized%20normalizing%20flows_%20what%20can%20we%20learn%20from%20a%20single%20failure/images/1c81c1d99a2b06460a4cab922d7a9a7bf3ec7e6e12590f5ebd6e7121de9eb3a5.jpg)

![3aac79d5d98a33e63179ec62dd30afba71818c884b4bb6eb2e33c1b1e3e98799.jpg](../iclr_results/1601_Rare%20event%20modeling%20with%20self-regularized%20normalizing%20flows_%20what%20can%20we%20learn%20from%20a%20single%20failure/images/3aac79d5d98a33e63179ec62dd30afba71818c884b4bb6eb2e33c1b1e3e98799.jpg)

![4156473770f993b49327e0859da0c25d2e6ff34cd4722bff28f0b8390a11f883.jpg](../iclr_results/1601_Rare%20event%20modeling%20with%20self-regularized%20normalizing%20flows_%20what%20can%20we%20learn%20from%20a%20single%20failure/images/4156473770f993b49327e0859da0c25d2e6ff34cd4722bff28f0b8390a11f883.jpg)

![47fdf82e802ac0221fc6079288eb7f48982737b1fb99299c741c16b6f4485257.jpg](../iclr_results/1601_Rare%20event%20modeling%20with%20self-regularized%20normalizing%20flows_%20what%20can%20we%20learn%20from%20a%20single%20failure/images/47fdf82e802ac0221fc6079288eb7f48982737b1fb99299c741c16b6f4485257.jpg)

![5c81b46dd58fa4da78fda4a7b02fb019d4dcbbddb7d65e359784ba21c652af66.jpg](../iclr_results/1601_Rare%20event%20modeling%20with%20self-regularized%20normalizing%20flows_%20what%20can%20we%20learn%20from%20a%20single%20failure/images/5c81b46dd58fa4da78fda4a7b02fb019d4dcbbddb7d65e359784ba21c652af66.jpg)

![a17dbb3cd93099a6d3382a2bdf41608cec4ad437d56f8fc81b6d5e8a392d2dfe.jpg](../iclr_results/1601_Rare%20event%20modeling%20with%20self-regularized%20normalizing%20flows_%20what%20can%20we%20learn%20from%20a%20single%20failure/images/a17dbb3cd93099a6d3382a2bdf41608cec4ad437d56f8fc81b6d5e8a392d2dfe.jpg)

![cf48189b7b350d2259acb90aadc6868e66da184b76c38296d548da6ed39a22a9.jpg](../iclr_results/1601_Rare%20event%20modeling%20with%20self-regularized%20normalizing%20flows_%20what%20can%20we%20learn%20from%20a%20single%20failure/images/cf48189b7b350d2259acb90aadc6868e66da184b76c38296d548da6ed39a22a9.jpg)

![d6540b5ec0115c8b02ca62b4a57cdacccf4d103dc06d4662c1708c77a31ede74.jpg](../iclr_results/1601_Rare%20event%20modeling%20with%20self-regularized%20normalizing%20flows_%20what%20can%20we%20learn%20from%20a%20single%20failure/images/d6540b5ec0115c8b02ca62b4a57cdacccf4d103dc06d4662c1708c77a31ede74.jpg)

![e02978d1d83fd9e0c6f2df61ab07b54f090f024235765889fee3e39dd57bcbb4.jpg](../iclr_results/1601_Rare%20event%20modeling%20with%20self-regularized%20normalizing%20flows_%20what%20can%20we%20learn%20from%20a%20single%20failure/images/e02978d1d83fd9e0c6f2df61ab07b54f090f024235765889fee3e39dd57bcbb4.jpg)

![eafd175bfe7efe96d350440b7c3261b4dadac49df12726b9d4059cf89db71d1d.jpg](../iclr_results/1601_Rare%20event%20modeling%20with%20self-regularized%20normalizing%20flows_%20what%20can%20we%20learn%20from%20a%20single%20failure/images/eafd175bfe7efe96d350440b7c3261b4dadac49df12726b9d4059cf89db71d1d.jpg)

![f6d6689ade23e1ac8ed9c0868a4ab818f599d5053819e26f4c6a38cadd284558.jpg](../iclr_results/1601_Rare%20event%20modeling%20with%20self-regularized%20normalizing%20flows_%20what%20can%20we%20learn%20from%20a%20single%20failure/images/f6d6689ade23e1ac8ed9c0868a4ab818f599d5053819e26f4c6a38cadd284558.jpg)

### Tables

![126b216cb84749b47f6d11e58f29934b8706f0627533ff9f7df94374e3fb74e5.jpg](../iclr_results/1601_Rare%20event%20modeling%20with%20self-regularized%20normalizing%20flows_%20what%20can%20we%20learn%20from%20a%20single%20failure/tables/126b216cb84749b47f6d11e58f29934b8706f0627533ff9f7df94374e3fb74e5.jpg)

![2aff9c64eb3017c935e8e2468f076a6a0aa23f7618b4d41f2b799bc0cb096b8a.jpg](../iclr_results/1601_Rare%20event%20modeling%20with%20self-regularized%20normalizing%20flows_%20what%20can%20we%20learn%20from%20a%20single%20failure/tables/2aff9c64eb3017c935e8e2468f076a6a0aa23f7618b4d41f2b799bc0cb096b8a.jpg)

![640bf9b702ecf1bbcbf7a2057055b6ba0e35ac58feb8d6d8df3742aff4165fc6.jpg](../iclr_results/1601_Rare%20event%20modeling%20with%20self-regularized%20normalizing%20flows_%20what%20can%20we%20learn%20from%20a%20single%20failure/tables/640bf9b702ecf1bbcbf7a2057055b6ba0e35ac58feb8d6d8df3742aff4165fc6.jpg)

![7d2dcc29c473eaac645e4785d94194693286e40726413cd566f7eed8a7b49593.jpg](../iclr_results/1601_Rare%20event%20modeling%20with%20self-regularized%20normalizing%20flows_%20what%20can%20we%20learn%20from%20a%20single%20failure/tables/7d2dcc29c473eaac645e4785d94194693286e40726413cd566f7eed8a7b49593.jpg)

![9d00a9c3914d98a4021b910ce62683b29548d7fd1df831ef9472dadcbac0b53d.jpg](../iclr_results/1601_Rare%20event%20modeling%20with%20self-regularized%20normalizing%20flows_%20what%20can%20we%20learn%20from%20a%20single%20failure/tables/9d00a9c3914d98a4021b910ce62683b29548d7fd1df831ef9472dadcbac0b53d.jpg)

![e3f600dbee01a55222a2514f527be8d47cc998408b1122a550546fc97b5c7a9a.jpg](../iclr_results/1601_Rare%20event%20modeling%20with%20self-regularized%20normalizing%20flows_%20what%20can%20we%20learn%20from%20a%20single%20failure/tables/e3f600dbee01a55222a2514f527be8d47cc998408b1122a550546fc97b5c7a9a.jpg)

![eb8348af2297f25dcc65d662995a9d655e92983ea1975f8170a84beada955367.jpg](../iclr_results/1601_Rare%20event%20modeling%20with%20self-regularized%20normalizing%20flows_%20what%20can%20we%20learn%20from%20a%20single%20failure/tables/eb8348af2297f25dcc65d662995a9d655e92983ea1975f8170a84beada955367.jpg)

![ef6c71269e2b6aa1cc54aea8ba65e82dd344b57d6b3c1bb91b0a9a4044aa36b1.jpg](../iclr_results/1601_Rare%20event%20modeling%20with%20self-regularized%20normalizing%20flows_%20what%20can%20we%20learn%20from%20a%20single%20failure/tables/ef6c71269e2b6aa1cc54aea8ba65e82dd344b57d6b3c1bb91b0a9a4044aa36b1.jpg)

![fbff7777c13a2fd4988e25a4b0451f6667865ae1cc4acdd052df8cb8031e16dc.jpg](../iclr_results/1601_Rare%20event%20modeling%20with%20self-regularized%20normalizing%20flows_%20what%20can%20we%20learn%20from%20a%20single%20failure/tables/fbff7777c13a2fd4988e25a4b0451f6667865ae1cc4acdd052df8cb8031e16dc.jpg)

![fe7cdef1557371a3a0cd5aa8190dee485105876ee73f7b6d8c4c03da65c3e120.jpg](../iclr_results/1601_Rare%20event%20modeling%20with%20self-regularized%20normalizing%20flows_%20what%20can%20we%20learn%20from%20a%20single%20failure/tables/fe7cdef1557371a3a0cd5aa8190dee485105876ee73f7b6d8c4c03da65c3e120.jpg)

## DocMIA: Document-Level Membership Inference Attacks against DocVQA Models

### Images

![0a2d61ca3b6916c9845f273d1a66d7c6f2f7bc13afbc71a06c854ab676756170.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/images/0a2d61ca3b6916c9845f273d1a66d7c6f2f7bc13afbc71a06c854ab676756170.jpg)

![27447f142d16f38a7e3d8738183807ed30fc6b78a8e5e5650aadd6143fb4b508.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/images/27447f142d16f38a7e3d8738183807ed30fc6b78a8e5e5650aadd6143fb4b508.jpg)

![2bd64ee37789e937221d291a49891882f316f4fd0c875a4bf6f6f4ac2cddf0f9.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/images/2bd64ee37789e937221d291a49891882f316f4fd0c875a4bf6f6f4ac2cddf0f9.jpg)

![34e25222d5a2b3684be48f313469e3c71a80bf37b22f76c224bf998795c959e3.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/images/34e25222d5a2b3684be48f313469e3c71a80bf37b22f76c224bf998795c959e3.jpg)

![478a4986478acd9d2b78e9cdddeacf2bc944a2c427494aee8a6474210122c4a8.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/images/478a4986478acd9d2b78e9cdddeacf2bc944a2c427494aee8a6474210122c4a8.jpg)

![49f9e0a1f061dd81d8d581aca88d8124625c408595fbb8822244f2a9d33bf4f6.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/images/49f9e0a1f061dd81d8d581aca88d8124625c408595fbb8822244f2a9d33bf4f6.jpg)

![4bc634b3c6285bf5a3244cece59ad1acc9f3c4160ff7d47c6ea9e277cc388820.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/images/4bc634b3c6285bf5a3244cece59ad1acc9f3c4160ff7d47c6ea9e277cc388820.jpg)

![57e0e3f00388c0a4e02463bc381f5575e3126487c7d0c983a1f10e14907f4b42.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/images/57e0e3f00388c0a4e02463bc381f5575e3126487c7d0c983a1f10e14907f4b42.jpg)

![b5468d7296686e6350aa68152ed7fb5d8011e6068faefb2f0f4e1a83f9591cf0.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/images/b5468d7296686e6350aa68152ed7fb5d8011e6068faefb2f0f4e1a83f9591cf0.jpg)

![c4e4e54a5116cad043e6d10e038256c78dd85894b878f1edb6bc91f9ef6c7f84.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/images/c4e4e54a5116cad043e6d10e038256c78dd85894b878f1edb6bc91f9ef6c7f84.jpg)

### Tables

![155f408dcfb5799726dbaa0c3faf9671e61826110f2a1f1bb81a673271b79a5b.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/tables/155f408dcfb5799726dbaa0c3faf9671e61826110f2a1f1bb81a673271b79a5b.jpg)

![16c11c4534b70d3c1fca2d5a5fe9b38f59482ffa092c629c239ab88061876733.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/tables/16c11c4534b70d3c1fca2d5a5fe9b38f59482ffa092c629c239ab88061876733.jpg)

![1f82b48afb9af049608ebaed1c09c8cf76ea07eaf0a50b735652e335797ed60b.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/tables/1f82b48afb9af049608ebaed1c09c8cf76ea07eaf0a50b735652e335797ed60b.jpg)

![23973b195b8bc874ee7d5005e33a03357d3c8c6c65c5be71fb2a9e3599086b86.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/tables/23973b195b8bc874ee7d5005e33a03357d3c8c6c65c5be71fb2a9e3599086b86.jpg)

![23ab8bcff8588a553da5bfdb399bc75a7e0a6e4b650ffb37a83dae33693d2288.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/tables/23ab8bcff8588a553da5bfdb399bc75a7e0a6e4b650ffb37a83dae33693d2288.jpg)

![4f7e560b190676f9791ce3f77901530110267fa0051f9624187c517a36c0e691.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/tables/4f7e560b190676f9791ce3f77901530110267fa0051f9624187c517a36c0e691.jpg)

![55eb7e6178373b5bd8c387c0432d15aa6409e6bbe1b99088d147f95ab744739f.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/tables/55eb7e6178373b5bd8c387c0432d15aa6409e6bbe1b99088d147f95ab744739f.jpg)

![603f158be146bfb708f97b42ce8d4867800e02ec3cb7247aeb8b5b12577c8a96.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/tables/603f158be146bfb708f97b42ce8d4867800e02ec3cb7247aeb8b5b12577c8a96.jpg)

![6437be1be6e917e2d1b8c0d55f6c044a28c34669caa8f738f9abfe78fcdd482e.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/tables/6437be1be6e917e2d1b8c0d55f6c044a28c34669caa8f738f9abfe78fcdd482e.jpg)

![6c5832a3805fbdb9a6bc17391ada280b854637bbdfdbf24276627f030fd83da2.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/tables/6c5832a3805fbdb9a6bc17391ada280b854637bbdfdbf24276627f030fd83da2.jpg)

![77cb56f29722a7d58abd581e3b892fcb5e4a50eef0a3962b83b787f823f3a015.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/tables/77cb56f29722a7d58abd581e3b892fcb5e4a50eef0a3962b83b787f823f3a015.jpg)

![8a9148bbea8062f8f9c003485e5a4a50fc7b6461b5f48e17be2aa89717ddc606.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/tables/8a9148bbea8062f8f9c003485e5a4a50fc7b6461b5f48e17be2aa89717ddc606.jpg)

![90b7df8ed6966595b08ecc08443d76fd4fbadbde4d43a2a7fa0f38cda12d3275.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/tables/90b7df8ed6966595b08ecc08443d76fd4fbadbde4d43a2a7fa0f38cda12d3275.jpg)

![953acbd6e3b0d7be5bbf0e7b12739cb04b212636d17dc67d69de37ad58f7e4b3.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/tables/953acbd6e3b0d7be5bbf0e7b12739cb04b212636d17dc67d69de37ad58f7e4b3.jpg)

![9765ba971047cbce85c5d7a03976d3cb2f95d205a2e8fed9dd45f70618aaa32c.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/tables/9765ba971047cbce85c5d7a03976d3cb2f95d205a2e8fed9dd45f70618aaa32c.jpg)

![9cfc9d33be6db087e5ecaf3dfe13706098cd645433b5905b6b6992d2f099465f.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/tables/9cfc9d33be6db087e5ecaf3dfe13706098cd645433b5905b6b6992d2f099465f.jpg)

![b600f991ff42105c3667bcc03defba518b226bd6c92ddf9035c90354cdbac25e.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/tables/b600f991ff42105c3667bcc03defba518b226bd6c92ddf9035c90354cdbac25e.jpg)

![c9c18768d81a0e8a74205387e5285422da258a2ee69712bf1281157d4cec0fd2.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/tables/c9c18768d81a0e8a74205387e5285422da258a2ee69712bf1281157d4cec0fd2.jpg)

![ccb64f969ef031fa77b718c06e7b3a4d4084abbc55780e2809ffc1f9dc88708c.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/tables/ccb64f969ef031fa77b718c06e7b3a4d4084abbc55780e2809ffc1f9dc88708c.jpg)

![d0b819630ab1900a5117ecccf2d3aa9ec08390ef93b0b07e2cd15c8511bbb9f9.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/tables/d0b819630ab1900a5117ecccf2d3aa9ec08390ef93b0b07e2cd15c8511bbb9f9.jpg)

![d0fd028a1c66b4d4de37451011dce86ee385b8a02ca019ddc8bcb25b29221922.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/tables/d0fd028a1c66b4d4de37451011dce86ee385b8a02ca019ddc8bcb25b29221922.jpg)

![e27d50d6839d29a7c5be3709928363e0e7a54cacba152e1d769ba82261c70f49.jpg](../iclr_results/1602_DocMIA_%20Document-Level%20Membership%20Inference%20Attacks%20against%20DocVQA%20Models/tables/e27d50d6839d29a7c5be3709928363e0e7a54cacba152e1d769ba82261c70f49.jpg)
