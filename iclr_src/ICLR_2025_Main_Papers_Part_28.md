# ICLR 2025 Main Conference Papers

**Summary:** 36 papers with extracted content:
- 📊 Total images: 44031
- 📋 Total tables: 33468
- 📄 Total files: 77499

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 28 of 100

## 目录 (Table of Contents)

1. [Personalized Visual Instruction Tuning](#Personalized-Visual-Instruction-Tuning)
2. [Quest: Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model](#Quest-Query-centric-Data-Synthesis-Approach-for-Long-context-Scaling-of-Large-Language-Model)
3. [DRoC: Elevating Large Language Models for Complex Vehicle Routing via Decomposed Retrieval of Constraints](#DRoC-Elevating-Large-Language-Models-for-Complex-Vehicle-Routing-via-Decomposed-Retrieval-of-Constraints)
4. [Dynamic-SUPERB Phase-2: A Collaboratively Expanding Benchmark for Measuring the Capabilities of Spoken Language Models with 180 Tasks](#Dynamic-SUPERB-Phase-2-A-Collaboratively-Expanding-Benchmark-for-Measuring-the-Capabilities-of-Spoken-Language-Models-with-180-Tasks)
5. [Train Small, Infer Large: Memory-Efficient LoRA Training for Large Language Models](#Train-Small-Infer-Large-Memory-Efficient-LoRA-Training-for-Large-Language-Models)
6. [Robust LLM safeguarding via refusal feature adversarial training](#Robust-LLM-safeguarding-via-refusal-feature-adversarial-training)
7. [MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models](#MMed-RAG-Versatile-Multimodal-RAG-System-for-Medical-Vision-Language-Models)
8. [What Are Good Positional Encodings for Directed Graphs?](#What-Are-Good-Positional-Encodings-for-Directed-Graphs)
9. [Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs](#Deep-Incomplete-Multi-view-Learning-via-Cyclic-Permutation-of-VAEs)
10. [Language Models Need Inductive Biases to Count Inductively](#Language-Models-Need-Inductive-Biases-to-Count-Inductively)
11. [Articulate-Anything:  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Model](#Articulate-Anything-Automatic-Modeling-of-Articulated-Objects-via-a-Vision-Language-Foundation-Model)
12. [Jailbreak Antidote: Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Language Models](#Jailbreak-Antidote-Runtime-Safety-Utility-Balance-via-Sparse-Representation-Adjustment-in-Large-Language-Models)
13. [Swing-by Dynamics in Concept Learning and Compositional Generalization](#Swing-by-Dynamics-in-Concept-Learning-and-Compositional-Generalization)
14. [An Evolved Universal Transformer Memory](#An-Evolved-Universal-Transformer-Memory)
15. [On Discriminative Probabilistic Modeling for Self-Supervised Representation Learning](#On-Discriminative-Probabilistic-Modeling-for-Self-Supervised-Representation-Learning)
16. [VCR: A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text](#VCR-A-Task-for-Pixel-Level-Complex-Reasoning-in-Vision-Language-Models-via-Restoring-Occluded-Text)
17. [TIGER: Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation](#TIGER-Time-frequency-Interleaved-Gain-Extraction-and-Reconstruction-for-Efficient-Speech-Separation)
18. [Minimal Impact ControlNet: Advancing Multi-ControlNet Integration](#Minimal-Impact-ControlNet-Advancing-Multi-ControlNet-Integration)
19. [Beyond the convexity assumption: Realistic tabular data generation under quantifier-free real linear constraints](#Beyond-the-convexity-assumption-Realistic-tabular-data-generation-under-quantifier-free-real-linear-constraints)
20. [Training-Free Dataset Pruning for Instance Segmentation](#Training-Free-Dataset-Pruning-for-Instance-Segmentation)
21. [From Probability to Counterfactuals: the Increasing Complexity of Satisfiability in Pearl's Causal Hierarchy](#From-Probability-to-Counterfactuals-the-Increasing-Complexity-of-Satisfiability-in-Pearls-Causal-Hierarchy)
22. [Efficient Alternating Minimization with Applications to Weighted Low Rank Approximation](#Efficient-Alternating-Minimization-with-Applications-to-Weighted-Low-Rank-Approximation)
23. [Robust Transfer of Safety-Constrained Reinforcement Learning Agents](#Robust-Transfer-of-Safety-Constrained-Reinforcement-Learning-Agents)
24. [Online-to-Offline RL for Agent Alignment](#Online-to-Offline-RL-for-Agent-Alignment)
25. [Self-Introspective Decoding: Alleviating Hallucinations for Large Vision-Language Models](#Self-Introspective-Decoding-Alleviating-Hallucinations-for-Large-Vision-Language-Models)
26. [Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling](#Faster-Inference-of-Flow-Based-Generative-Models-via-Improved-Data-Noise-Coupling)
27. [ConvCodeWorld: Benchmarking Conversational Code Generation in Reproducible Feedback Environments](#ConvCodeWorld-Benchmarking-Conversational-Code-Generation-in-Reproducible-Feedback-Environments)
28. [{$\tau$}-bench: A Benchmark for \underline{T}ool-\underline{A}gent-\underline{U}ser Interaction in Real-World Domains](#tau-bench-A-Benchmark-for-underlineTool-underlineAgent-underlineUser-Interaction-in-Real-World-Domains)
29. [Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior](#Bayesian-Image-Regression-with-Soft-thresholded-Conditional-Autoregressive-Prior)
30. [Beyond Mere Token Analysis: A Hypergraph Metric Space Framework for Defending Against Socially Engineered LLM Attacks](#Beyond-Mere-Token-Analysis-A-Hypergraph-Metric-Space-Framework-for-Defending-Against-Socially-Engineered-LLM-Attacks)
31. [OCEAN: Offline Chain-of-thought Evaluation and Alignment in Large Language Models](#OCEAN-Offline-Chain-of-thought-Evaluation-and-Alignment-in-Large-Language-Models)
32. [Hallo2: Long-Duration and High-Resolution Audio-Driven Portrait Image Animation](#Hallo2-Long-Duration-and-High-Resolution-Audio-Driven-Portrait-Image-Animation)
33. [Learning system dynamics without forgetting](#Learning-system-dynamics-without-forgetting)
34. [UIFace: Unleashing Inherent Model Capabilities to Enhance Intra-Class Diversity in Synthetic Face Recognition](#UIFace-Unleashing-Inherent-Model-Capabilities-to-Enhance-Intra-Class-Diversity-in-Synthetic-Face-Recognition)
35. [SWE-bench Multimodal: Do AI Systems Generalize to Visual Software Domains?](#SWE-bench-Multimodal-Do-AI-Systems-Generalize-to-Visual-Software-Domains)
36. [Optimal Protocols for Continual Learning via Statistical Physics and Control Theory](#Optimal-Protocols-for-Continual-Learning-via-Statistical-Physics-and-Control-Theory)

---


## Personalized Visual Instruction Tuning

### Images

![165b428ee9407b75aac3ca237aedfdbc73d1e3d5a3baf6fef027c7f0ee52fc0d.jpg](../iclr_results/1020_Partial Gromov-Wasserstein Metric/images/165b428ee9407b75aac3ca237aedfdbc73d1e3d5a3baf6fef027c7f0ee52fc0d.jpg)

![1e4e16e5542de430b6d4a26d3b3b2f32b4d483200c6f427adaa663a8d5ab63fa.jpg](../iclr_results/1020_Partial Gromov-Wasserstein Metric/images/1e4e16e5542de430b6d4a26d3b3b2f32b4d483200c6f427adaa663a8d5ab63fa.jpg)

![68a0eaa256f9d09c76fc2fdc6759efaf4b4f1ab6818a0175704872c21a14e59b.jpg](../iclr_results/1020_Partial Gromov-Wasserstein Metric/images/68a0eaa256f9d09c76fc2fdc6759efaf4b4f1ab6818a0175704872c21a14e59b.jpg)

![6ba5b8d75284abf2f4ee452552f99c8d9ce4db0f7a4e596d564ec54980e98e07.jpg](../iclr_results/1020_Partial Gromov-Wasserstein Metric/images/6ba5b8d75284abf2f4ee452552f99c8d9ce4db0f7a4e596d564ec54980e98e07.jpg)

![75d402eaab329e4d39bfe47a562a7d308549ad719f0ad32358d7d765253b9d10.jpg](../iclr_results/1020_Partial Gromov-Wasserstein Metric/images/75d402eaab329e4d39bfe47a562a7d308549ad719f0ad32358d7d765253b9d10.jpg)

![7936e160ae8c25669127e5cf38a521019d9d59d990c79000ceef328af17b9d8b.jpg](../iclr_results/1020_Partial Gromov-Wasserstein Metric/images/7936e160ae8c25669127e5cf38a521019d9d59d990c79000ceef328af17b9d8b.jpg)

![896d2fb703c78a0de6095b38c10b271685f041bf051117759b1eb24a92872c56.jpg](../iclr_results/1020_Partial Gromov-Wasserstein Metric/images/896d2fb703c78a0de6095b38c10b271685f041bf051117759b1eb24a92872c56.jpg)

![aad1b39b91facef2560e3908507ed5f27409e7e7b16f899fc4214ab2bbdf4161.jpg](../iclr_results/1020_Partial Gromov-Wasserstein Metric/images/aad1b39b91facef2560e3908507ed5f27409e7e7b16f899fc4214ab2bbdf4161.jpg)

![b470eaa441cf471e4fbc112e6ce5ed9834a5e9e14ab72809973022a2fdcdfd3a.jpg](../iclr_results/1020_Partial Gromov-Wasserstein Metric/images/b470eaa441cf471e4fbc112e6ce5ed9834a5e9e14ab72809973022a2fdcdfd3a.jpg)

![d1568821a96c479b6cbb8f3d9a92e2f3ba915cf8d45a723bee5625f557b1ba23.jpg](../iclr_results/1020_Partial Gromov-Wasserstein Metric/images/d1568821a96c479b6cbb8f3d9a92e2f3ba915cf8d45a723bee5625f557b1ba23.jpg)

![ebee040cbca382d1331bb0f9092a9dc5851919802ba35ad5db512e6eac1b79b2.jpg](../iclr_results/1020_Partial Gromov-Wasserstein Metric/images/ebee040cbca382d1331bb0f9092a9dc5851919802ba35ad5db512e6eac1b79b2.jpg)

![f117cfb3c382d8316e8e94924d9d6b6a4436c9c9e655e14446e6dc8d1e82edef.jpg](../iclr_results/1020_Partial Gromov-Wasserstein Metric/images/f117cfb3c382d8316e8e94924d9d6b6a4436c9c9e655e14446e6dc8d1e82edef.jpg)

![f1eeb0701aef630cfab859ac26afc0f44197bebe2035b9eee6adc7bebb898576.jpg](../iclr_results/1020_Partial Gromov-Wasserstein Metric/images/f1eeb0701aef630cfab859ac26afc0f44197bebe2035b9eee6adc7bebb898576.jpg)

### Tables

![379ac63c7ce65e8dbd340ab3fbc29330dc3f497cf6a7d8578081dc290cd0f3fa.jpg](../iclr_results/1020_Partial Gromov-Wasserstein Metric/tables/379ac63c7ce65e8dbd340ab3fbc29330dc3f497cf6a7d8578081dc290cd0f3fa.jpg)

![3b42948d4dde8c5ef5002981eca3e9c45d1ff01dc50db4f59ff33ccd7e30a94b.jpg](../iclr_results/1020_Partial Gromov-Wasserstein Metric/tables/3b42948d4dde8c5ef5002981eca3e9c45d1ff01dc50db4f59ff33ccd7e30a94b.jpg)

![4ec1a681c130a9b0e1779208a0f7beffefd926df35ee0282920371804b3a5fd0.jpg](../iclr_results/1020_Partial Gromov-Wasserstein Metric/tables/4ec1a681c130a9b0e1779208a0f7beffefd926df35ee0282920371804b3a5fd0.jpg)

![79bc4a054caf02b4f0593ce315e3589604f312beeb02a0000d492256c4beca49.jpg](../iclr_results/1020_Partial Gromov-Wasserstein Metric/tables/79bc4a054caf02b4f0593ce315e3589604f312beeb02a0000d492256c4beca49.jpg)

![7eda4e4bc422e2c4be0ad4c12a6e4ace0c9f3657a26ed12bd29a5bac138b12d5.jpg](../iclr_results/1020_Partial Gromov-Wasserstein Metric/tables/7eda4e4bc422e2c4be0ad4c12a6e4ace0c9f3657a26ed12bd29a5bac138b12d5.jpg)

![87c522e23066fff047fdf405fde08e3447b01a23992cdd1909c1ca07526fedf3.jpg](../iclr_results/1020_Partial Gromov-Wasserstein Metric/tables/87c522e23066fff047fdf405fde08e3447b01a23992cdd1909c1ca07526fedf3.jpg)

![a9560e57a638ea32c289113552c659d36b7fc50d4e651b8d4acbf530df2629f3.jpg](../iclr_results/1020_Partial Gromov-Wasserstein Metric/tables/a9560e57a638ea32c289113552c659d36b7fc50d4e651b8d4acbf530df2629f3.jpg)

![ad5127a0a0282750172342457af5557df7aee9defaf2dd2dd39cf8b275df1ac1.jpg](../iclr_results/1020_Partial Gromov-Wasserstein Metric/tables/ad5127a0a0282750172342457af5557df7aee9defaf2dd2dd39cf8b275df1ac1.jpg)

![fa4c3a4f4ac317ef7b50198a3a481afc4701cfb8d1a843394cd3089f6486a40e.jpg](../iclr_results/1020_Partial Gromov-Wasserstein Metric/tables/fa4c3a4f4ac317ef7b50198a3a481afc4701cfb8d1a843394cd3089f6486a40e.jpg)

![fea6dc1dd9ba225c16f2574902907720779af3ff0d4fa8be526d55632d447166.jpg](../iclr_results/1020_Partial Gromov-Wasserstein Metric/tables/fea6dc1dd9ba225c16f2574902907720779af3ff0d4fa8be526d55632d447166.jpg)

## Personalized Visual Instruction Tuning


### Images

![00009d373d2aaf7247f30f2bec72a04dd17196bdbc72407abb2b7b277a89ec76.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/images/00009d373d2aaf7247f30f2bec72a04dd17196bdbc72407abb2b7b277a89ec76.jpg)

![0559dc2b23d68e08908af3b8f03476acd8a66b16c276c0ba1275b7ef2c406db2.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/images/0559dc2b23d68e08908af3b8f03476acd8a66b16c276c0ba1275b7ef2c406db2.jpg)

![1766ce07e9b64fce61de2b41c72ce79f852ad4d4609352a95bf81bbbe2185efb.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/images/1766ce07e9b64fce61de2b41c72ce79f852ad4d4609352a95bf81bbbe2185efb.jpg)

![19deeac43a5d3c308563a8a69dba3cdf9093ff81b0a68cebf1635f12a1d38fae.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/images/19deeac43a5d3c308563a8a69dba3cdf9093ff81b0a68cebf1635f12a1d38fae.jpg)

![1d4c4d0dcb1e8191b522f629dca069093640de4937c4bb79721d972d3c6178e4.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/images/1d4c4d0dcb1e8191b522f629dca069093640de4937c4bb79721d972d3c6178e4.jpg)

![21c34088c493d967e111a8a3777b56b3764fcdc6bb2d1771777e4f70f0af1f96.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/images/21c34088c493d967e111a8a3777b56b3764fcdc6bb2d1771777e4f70f0af1f96.jpg)

![3abf4e329c22b471c8d9f577d0fec8c0f72ce6ec111cf673d9197649ee078a6e.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/images/3abf4e329c22b471c8d9f577d0fec8c0f72ce6ec111cf673d9197649ee078a6e.jpg)

![4e54cb2d21aa8d4585c40036589eb2df7af7dd87ea0c2ee64a2b355a5637f3d0.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/images/4e54cb2d21aa8d4585c40036589eb2df7af7dd87ea0c2ee64a2b355a5637f3d0.jpg)

![5508460d3c2c9eeac62d66d77f3f39ca5cd286bd5cb9e39a7c5abbd92cae1a1e.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/images/5508460d3c2c9eeac62d66d77f3f39ca5cd286bd5cb9e39a7c5abbd92cae1a1e.jpg)

![5551b229a51b777195d3798656b0614f05309efd65967e79c75b3501384fb12a.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/images/5551b229a51b777195d3798656b0614f05309efd65967e79c75b3501384fb12a.jpg)

![5e264798ef222c9254e188fe66b7809b2db7969937b4fd89c1f0af5f52e1be43.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/images/5e264798ef222c9254e188fe66b7809b2db7969937b4fd89c1f0af5f52e1be43.jpg)

![7bbb9749ad05d5b92c0b7773709aeee7af708e8d1b3218fbaa5e4e536019d98c.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/images/7bbb9749ad05d5b92c0b7773709aeee7af708e8d1b3218fbaa5e4e536019d98c.jpg)

![8acad305f27fad92bca79d1237be340b8a96a341904376b03af5e5c1ecdc44af.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/images/8acad305f27fad92bca79d1237be340b8a96a341904376b03af5e5c1ecdc44af.jpg)

![8e8647ed86d54df46607a9fbf8a7a276be13cac48dbfd512aacb25645d65e6c7.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/images/8e8647ed86d54df46607a9fbf8a7a276be13cac48dbfd512aacb25645d65e6c7.jpg)

![8ff3d6c68335b17d9b75785fd59a63a7843dba42a3a4fff75d70c358ecfc1322.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/images/8ff3d6c68335b17d9b75785fd59a63a7843dba42a3a4fff75d70c358ecfc1322.jpg)

![9a40805b82792f6f56f14f0d10e34e56f5dba83063d0358853afde2bb4015d48.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/images/9a40805b82792f6f56f14f0d10e34e56f5dba83063d0358853afde2bb4015d48.jpg)

![a479d1b32e2a19865ca62df7088ab78fbea5bf772d4ba2bea219dc5829b0c33a.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/images/a479d1b32e2a19865ca62df7088ab78fbea5bf772d4ba2bea219dc5829b0c33a.jpg)

![bdbf960dd026384bdc1ad5ee73679e662e36edb0501de7422c314052f760d070.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/images/bdbf960dd026384bdc1ad5ee73679e662e36edb0501de7422c314052f760d070.jpg)

![c033e4b7369e19027b33b2a5aa22370dff700d0b9507401384463a14b92e0dc9.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/images/c033e4b7369e19027b33b2a5aa22370dff700d0b9507401384463a14b92e0dc9.jpg)

![ca69090b0d3b1df61716faece1917dd2a8f110a4153b4f46539a0f1e35eebf5d.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/images/ca69090b0d3b1df61716faece1917dd2a8f110a4153b4f46539a0f1e35eebf5d.jpg)

![e84b00c2a4a85cae70a4b7bc5eaf89e18e38d9410a63247b176b5f97b76e76c4.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/images/e84b00c2a4a85cae70a4b7bc5eaf89e18e38d9410a63247b176b5f97b76e76c4.jpg)

### Tables

![252cd169b0ee23051451c08fb162eb51e50203b831e293aabf9ea1476645c479.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/tables/252cd169b0ee23051451c08fb162eb51e50203b831e293aabf9ea1476645c479.jpg)

![af023f1ae0f7b6c196837f6d9804218ef8a585e892025bbf1c0c26c7713e454b.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/tables/af023f1ae0f7b6c196837f6d9804218ef8a585e892025bbf1c0c26c7713e454b.jpg)

![b7053fd191715363b8b4cd0e419348d16ed2e9781f8a1fcced3ca8f7c0a249a1.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/tables/b7053fd191715363b8b4cd0e419348d16ed2e9781f8a1fcced3ca8f7c0a249a1.jpg)

![cb390401382e93bad395a940e56ddcf9f55d71201ad02c170dd6f913feb497e6.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/tables/cb390401382e93bad395a940e56ddcf9f55d71201ad02c170dd6f913feb497e6.jpg)

![ea5a128ab00121151bb57383c05dce3816a947eb9291b49b4a8ede7802ebe066.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/tables/ea5a128ab00121151bb57383c05dce3816a947eb9291b49b4a8ede7802ebe066.jpg)

![f86220e441e1eb95bf6e7a9718ba8cca17fc9f163fc04bee233e10877c2d62b1.jpg](../iclr_results/1021_Personalized Visual Instruction Tuning/tables/f86220e441e1eb95bf6e7a9718ba8cca17fc9f163fc04bee233e10877c2d62b1.jpg)

## Quest: Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model


### Images

![0db5438fe400a9fc40568af18e332cb0eb77b30b94991645cfe9c346f95a0f74.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/images/0db5438fe400a9fc40568af18e332cb0eb77b30b94991645cfe9c346f95a0f74.jpg)

![5b673d915fce854a7726a705c7d6fd2d670fdcfb55ff32ebfa95641ca2a71cb3.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/images/5b673d915fce854a7726a705c7d6fd2d670fdcfb55ff32ebfa95641ca2a71cb3.jpg)

![62722b7d9cda8e1b6e4906d8a5c0e965d54c4d75b428cb240190518cc75dcc75.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/images/62722b7d9cda8e1b6e4906d8a5c0e965d54c4d75b428cb240190518cc75dcc75.jpg)

![6c1839870b2acdb5470bd635a76746460f1a1e49daf03886da0d5696c1573e34.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/images/6c1839870b2acdb5470bd635a76746460f1a1e49daf03886da0d5696c1573e34.jpg)

![7d0247b7b0ee04b8ac8343c2975f28bd9e616704976cc0b49f2f60950785a04c.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/images/7d0247b7b0ee04b8ac8343c2975f28bd9e616704976cc0b49f2f60950785a04c.jpg)

![9b01d70fc63831aa04e3511708873f48291f0a43df05f530264864baf11b8b59.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/images/9b01d70fc63831aa04e3511708873f48291f0a43df05f530264864baf11b8b59.jpg)

![9f536e47776efd155e55f577fcd32a296c633cff4077ef051ddf95158d8daca7.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/images/9f536e47776efd155e55f577fcd32a296c633cff4077ef051ddf95158d8daca7.jpg)

![c846f24f5c879ddf012c3a306586b77acec0383597c1a7a7e7b675e0fdfff055.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/images/c846f24f5c879ddf012c3a306586b77acec0383597c1a7a7e7b675e0fdfff055.jpg)

![c8859cd6676c2e2f1905ad4ef8292ec90f27b067431c7525bf8edcc157ee82f2.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/images/c8859cd6676c2e2f1905ad4ef8292ec90f27b067431c7525bf8edcc157ee82f2.jpg)

![e0c80c3e5354bc7b5e65c309a2d0e95cd6936f364511c064fbe23a32cfcf35d1.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/images/e0c80c3e5354bc7b5e65c309a2d0e95cd6936f364511c064fbe23a32cfcf35d1.jpg)

![fbcc891c64f28500a4fd0db501cfd9a47ff92c437cf2d075045e3d7e3825c61f.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/images/fbcc891c64f28500a4fd0db501cfd9a47ff92c437cf2d075045e3d7e3825c61f.jpg)

### Tables

![0cf06428bbc38005d36eb223a7bcb5489e90726abcbdb6b26e85711472f178a4.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/tables/0cf06428bbc38005d36eb223a7bcb5489e90726abcbdb6b26e85711472f178a4.jpg)

![1748ed8ec159699e2d144a5e1b61d8701aadaa04e385aff115c834e5a8d7eee2.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/tables/1748ed8ec159699e2d144a5e1b61d8701aadaa04e385aff115c834e5a8d7eee2.jpg)

![1b00bf2a51c91082356908bd31df818b7f21e47e1d7fdd756f11404dd669eb39.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/tables/1b00bf2a51c91082356908bd31df818b7f21e47e1d7fdd756f11404dd669eb39.jpg)

![33ff68f88330dae2fd46ce517db5d7683b607604f3af0abf8b12d570e3e48cc0.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/tables/33ff68f88330dae2fd46ce517db5d7683b607604f3af0abf8b12d570e3e48cc0.jpg)

![3a128ab35a838cbf5d31738e2f5840912de441e24aeb71f7611eff58ca9a76ab.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/tables/3a128ab35a838cbf5d31738e2f5840912de441e24aeb71f7611eff58ca9a76ab.jpg)

![510b4cc3a23a869883e209c6389dfc87ed9461a1a32cbdd745691d982035f3fc.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/tables/510b4cc3a23a869883e209c6389dfc87ed9461a1a32cbdd745691d982035f3fc.jpg)

![7046def1c42e173560b2f0d96e878de2b04001ff1a51e05bc3dffd1132e77ae1.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/tables/7046def1c42e173560b2f0d96e878de2b04001ff1a51e05bc3dffd1132e77ae1.jpg)

![713714ca53159ef688314648b545f34ef5e5350713e139af0704b7954eb93ddf.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/tables/713714ca53159ef688314648b545f34ef5e5350713e139af0704b7954eb93ddf.jpg)

![73a0c004c9df426640ced12953bcbacdac961a7d31af21a44ab1ab7def378bd0.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/tables/73a0c004c9df426640ced12953bcbacdac961a7d31af21a44ab1ab7def378bd0.jpg)

![76911093227ba272d164f966153ad581332fafc4ddf8b1fa30a826554fd77518.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/tables/76911093227ba272d164f966153ad581332fafc4ddf8b1fa30a826554fd77518.jpg)

![8e157888372bf0b578c6ef49e5b39515969334de6cb9c2d5d874c18959a15752.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/tables/8e157888372bf0b578c6ef49e5b39515969334de6cb9c2d5d874c18959a15752.jpg)

![b2a315d3c4b6a17d91a889a83397e165fdfd9c27c2da382cc4bd5199d6c84dbf.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/tables/b2a315d3c4b6a17d91a889a83397e165fdfd9c27c2da382cc4bd5199d6c84dbf.jpg)

![b3b214c080160cab19fe01702565c5a601fc0506247c43568ea984b6b464887f.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/tables/b3b214c080160cab19fe01702565c5a601fc0506247c43568ea984b6b464887f.jpg)

![b74cd857647569f96269a2e8200ef6baad76520b04e380316fa445d8f4cd7e9c.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/tables/b74cd857647569f96269a2e8200ef6baad76520b04e380316fa445d8f4cd7e9c.jpg)

![e0676f0791dd727218a9a8d2029b6c6459b00a9fc26f22c2e912444e2297ec6d.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/tables/e0676f0791dd727218a9a8d2029b6c6459b00a9fc26f22c2e912444e2297ec6d.jpg)

![e15e97329bd961ae200cf9298b2c540082ac96f6865e1b169bbd37ae85356738.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/tables/e15e97329bd961ae200cf9298b2c540082ac96f6865e1b169bbd37ae85356738.jpg)

![f80d94e9b883eabe94f2734191c12e12f6c9d69a6e24b87d3674cb17465c25e3.jpg](../iclr_results/1022_Quest_ Query-centric Data Synthesis Approach for Long-context Scaling of Large Language Model/tables/f80d94e9b883eabe94f2734191c12e12f6c9d69a6e24b87d3674cb17465c25e3.jpg)

## DRoC: Elevating Large Language Models for Complex Vehicle Routing via Decomposed Retrieval of Constraints


### Images

![0796d2fcc67b94d88381c5c5f9b5caab33827d214cef70bcca49d56e8c2ed6e6.jpg](../iclr_results/1023_DRoC_ Elevating Large Language Models for Complex Vehicle Routing via Decomposed Retrieval of Constr/images/0796d2fcc67b94d88381c5c5f9b5caab33827d214cef70bcca49d56e8c2ed6e6.jpg)

![72d0f9f41dcc8332f51f4d00d16e9fdbe79a27f4979b5ea6511beeeb5b4bb9e4.jpg](../iclr_results/1023_DRoC_ Elevating Large Language Models for Complex Vehicle Routing via Decomposed Retrieval of Constr/images/72d0f9f41dcc8332f51f4d00d16e9fdbe79a27f4979b5ea6511beeeb5b4bb9e4.jpg)

![98784de296b09ff5f7820dd376c335201d9f72619da7b37dc887c4be84538be8.jpg](../iclr_results/1023_DRoC_ Elevating Large Language Models for Complex Vehicle Routing via Decomposed Retrieval of Constr/images/98784de296b09ff5f7820dd376c335201d9f72619da7b37dc887c4be84538be8.jpg)

![a3f1e898958d29bf3429261aa65bf3f5de50b2362cb6a5c28da844453e0fc7a8.jpg](../iclr_results/1023_DRoC_ Elevating Large Language Models for Complex Vehicle Routing via Decomposed Retrieval of Constr/images/a3f1e898958d29bf3429261aa65bf3f5de50b2362cb6a5c28da844453e0fc7a8.jpg)

![b1ce2b5b0c1d33e85fcbc92cb284b2e4b476e0bd562aa0661e3eea239eb1360d.jpg](../iclr_results/1023_DRoC_ Elevating Large Language Models for Complex Vehicle Routing via Decomposed Retrieval of Constr/images/b1ce2b5b0c1d33e85fcbc92cb284b2e4b476e0bd562aa0661e3eea239eb1360d.jpg)

![bbed7c565fec7428379920ec423039cf90df192d6e5c0f0e98260cd9d469e85b.jpg](../iclr_results/1023_DRoC_ Elevating Large Language Models for Complex Vehicle Routing via Decomposed Retrieval of Constr/images/bbed7c565fec7428379920ec423039cf90df192d6e5c0f0e98260cd9d469e85b.jpg)

![c3c1e289bd47fb597ef0f89935761c722d1db7b3c2f72235e3cb9898f7ce44b5.jpg](../iclr_results/1023_DRoC_ Elevating Large Language Models for Complex Vehicle Routing via Decomposed Retrieval of Constr/images/c3c1e289bd47fb597ef0f89935761c722d1db7b3c2f72235e3cb9898f7ce44b5.jpg)

![d6cae9510127a1e514619f235bb38e31bfbcba02315773a8fb9ff4459d3ae208.jpg](../iclr_results/1023_DRoC_ Elevating Large Language Models for Complex Vehicle Routing via Decomposed Retrieval of Constr/images/d6cae9510127a1e514619f235bb38e31bfbcba02315773a8fb9ff4459d3ae208.jpg)

![fcb03b1344c5d73bd2ba877cff66949f204415ac1e6af408d4203e0538c3affd.jpg](../iclr_results/1023_DRoC_ Elevating Large Language Models for Complex Vehicle Routing via Decomposed Retrieval of Constr/images/fcb03b1344c5d73bd2ba877cff66949f204415ac1e6af408d4203e0538c3affd.jpg)

![ffd8bf4d24ef9bf54c0076d62d44bd117431cc175b33a8806f26cecb3b945cb8.jpg](../iclr_results/1023_DRoC_ Elevating Large Language Models for Complex Vehicle Routing via Decomposed Retrieval of Constr/images/ffd8bf4d24ef9bf54c0076d62d44bd117431cc175b33a8806f26cecb3b945cb8.jpg)

### Tables

![07196642e749f7408c5b703903dbd4dd3036cd6df9147f06958ffd32b36d917a.jpg](../iclr_results/1023_DRoC_ Elevating Large Language Models for Complex Vehicle Routing via Decomposed Retrieval of Constr/tables/07196642e749f7408c5b703903dbd4dd3036cd6df9147f06958ffd32b36d917a.jpg)

![1adfa155cf26c40dac4b69fa31f3bef7b448011232d7b78557b49c48954deed2.jpg](../iclr_results/1023_DRoC_ Elevating Large Language Models for Complex Vehicle Routing via Decomposed Retrieval of Constr/tables/1adfa155cf26c40dac4b69fa31f3bef7b448011232d7b78557b49c48954deed2.jpg)

![1cc4af782067c93c88e2946a994ea4389bf0167bcdb8fbc0b74de97e37db4436.jpg](../iclr_results/1023_DRoC_ Elevating Large Language Models for Complex Vehicle Routing via Decomposed Retrieval of Constr/tables/1cc4af782067c93c88e2946a994ea4389bf0167bcdb8fbc0b74de97e37db4436.jpg)

![340d213bc9d2cf686e3cedba41bf32974c850528c05746f78b998f05a76e8bdc.jpg](../iclr_results/1023_DRoC_ Elevating Large Language Models for Complex Vehicle Routing via Decomposed Retrieval of Constr/tables/340d213bc9d2cf686e3cedba41bf32974c850528c05746f78b998f05a76e8bdc.jpg)

![3d444537f65c14944f83d855f87feca4a384cff52b68a1622965b584d22f8d6e.jpg](../iclr_results/1023_DRoC_ Elevating Large Language Models for Complex Vehicle Routing via Decomposed Retrieval of Constr/tables/3d444537f65c14944f83d855f87feca4a384cff52b68a1622965b584d22f8d6e.jpg)

![b0612d4f8eeece248fef16b3b1994483d79b77700ec5776903f90a28c2d4bcf8.jpg](../iclr_results/1023_DRoC_ Elevating Large Language Models for Complex Vehicle Routing via Decomposed Retrieval of Constr/tables/b0612d4f8eeece248fef16b3b1994483d79b77700ec5776903f90a28c2d4bcf8.jpg)

![b76bc8c18178c50d53339c294066b50a7a5d5cecaf83821740a5db80d95665e7.jpg](../iclr_results/1023_DRoC_ Elevating Large Language Models for Complex Vehicle Routing via Decomposed Retrieval of Constr/tables/b76bc8c18178c50d53339c294066b50a7a5d5cecaf83821740a5db80d95665e7.jpg)

## Dynamic-SUPERB Phase-2: A Collaboratively Expanding Benchmark for Measuring the Capabilities of Spoken Language Models with 180 Tasks


### Images

![0e6a5bb09866db73d83f7d8474fe26ca401d6fe2d10f95fdb33634ce19d2b010.jpg](../iclr_results/1024_Dynamic-SUPERB Phase-2_ A Collaboratively Expanding Benchmark for Measuring the Capabilities of Spok/images/0e6a5bb09866db73d83f7d8474fe26ca401d6fe2d10f95fdb33634ce19d2b010.jpg)

![301d508d49981ac2ab541521d6fb79d1fe6f38d8ba649f1650293fbe5ceece00.jpg](../iclr_results/1024_Dynamic-SUPERB Phase-2_ A Collaboratively Expanding Benchmark for Measuring the Capabilities of Spok/images/301d508d49981ac2ab541521d6fb79d1fe6f38d8ba649f1650293fbe5ceece00.jpg)

![7f7c9d14b283e94bd71ec628f9c7ad6c38c772db48c5cda16fb6a0268454910d.jpg](../iclr_results/1024_Dynamic-SUPERB Phase-2_ A Collaboratively Expanding Benchmark for Measuring the Capabilities of Spok/images/7f7c9d14b283e94bd71ec628f9c7ad6c38c772db48c5cda16fb6a0268454910d.jpg)

![88ab9471556144b9edc12e8751b3d0bda78fb2c17a0b7d633dba202e7b57a60a.jpg](../iclr_results/1024_Dynamic-SUPERB Phase-2_ A Collaboratively Expanding Benchmark for Measuring the Capabilities of Spok/images/88ab9471556144b9edc12e8751b3d0bda78fb2c17a0b7d633dba202e7b57a60a.jpg)

![980649d485b38bd52fdfcf217bde83c170f0d945ba67f78e7bbcc66da4d96ab0.jpg](../iclr_results/1024_Dynamic-SUPERB Phase-2_ A Collaboratively Expanding Benchmark for Measuring the Capabilities of Spok/images/980649d485b38bd52fdfcf217bde83c170f0d945ba67f78e7bbcc66da4d96ab0.jpg)

![aff4b4c3a281b7e3335fccc14bc9329bea38591187d91af6ff3ad93ab1734777.jpg](../iclr_results/1024_Dynamic-SUPERB Phase-2_ A Collaboratively Expanding Benchmark for Measuring the Capabilities of Spok/images/aff4b4c3a281b7e3335fccc14bc9329bea38591187d91af6ff3ad93ab1734777.jpg)

![d6b8f353bcd6abb9e30d65d05642e43caddcfdcb78e91feb3ce3c387f4bae6ba.jpg](../iclr_results/1024_Dynamic-SUPERB Phase-2_ A Collaboratively Expanding Benchmark for Measuring the Capabilities of Spok/images/d6b8f353bcd6abb9e30d65d05642e43caddcfdcb78e91feb3ce3c387f4bae6ba.jpg)

### Tables

![116318342a283ff64c3afc14491b2fe2870be82b5250751b73d79ea0c6152969.jpg](../iclr_results/1024_Dynamic-SUPERB Phase-2_ A Collaboratively Expanding Benchmark for Measuring the Capabilities of Spok/tables/116318342a283ff64c3afc14491b2fe2870be82b5250751b73d79ea0c6152969.jpg)

![37d09e8d18372f0418b815752b28e1f29202ad7222053acfb8a3074fff495b3a.jpg](../iclr_results/1024_Dynamic-SUPERB Phase-2_ A Collaboratively Expanding Benchmark for Measuring the Capabilities of Spok/tables/37d09e8d18372f0418b815752b28e1f29202ad7222053acfb8a3074fff495b3a.jpg)

![6ee0bdd7115c7896f0b3973e8fa042eb1c09609f1172a1122c5c62bd71a26890.jpg](../iclr_results/1024_Dynamic-SUPERB Phase-2_ A Collaboratively Expanding Benchmark for Measuring the Capabilities of Spok/tables/6ee0bdd7115c7896f0b3973e8fa042eb1c09609f1172a1122c5c62bd71a26890.jpg)

![865a8c384eb10dfcb29f341b56429c6b86d5684b55c62cc4a3ce370881e8320d.jpg](../iclr_results/1024_Dynamic-SUPERB Phase-2_ A Collaboratively Expanding Benchmark for Measuring the Capabilities of Spok/tables/865a8c384eb10dfcb29f341b56429c6b86d5684b55c62cc4a3ce370881e8320d.jpg)

![8a9f1e91456f62880584de49468fdd0fa2a434a0cd9b2c3554a636cd16138c8d.jpg](../iclr_results/1024_Dynamic-SUPERB Phase-2_ A Collaboratively Expanding Benchmark for Measuring the Capabilities of Spok/tables/8a9f1e91456f62880584de49468fdd0fa2a434a0cd9b2c3554a636cd16138c8d.jpg)

![91e9b46ebd2ac445c28fde0b13fb15b2a8606b9c4abde7a11780a65a9898254b.jpg](../iclr_results/1024_Dynamic-SUPERB Phase-2_ A Collaboratively Expanding Benchmark for Measuring the Capabilities of Spok/tables/91e9b46ebd2ac445c28fde0b13fb15b2a8606b9c4abde7a11780a65a9898254b.jpg)

![ae8a04edebf6149d5980983687c75e30d1d499b8bca2682ccd7421561dab7fbe.jpg](../iclr_results/1024_Dynamic-SUPERB Phase-2_ A Collaboratively Expanding Benchmark for Measuring the Capabilities of Spok/tables/ae8a04edebf6149d5980983687c75e30d1d499b8bca2682ccd7421561dab7fbe.jpg)

![c6a5ab5768bf1302b60d50dfdf396ed5aebe2092f2671b1865f3b102ccb0ac2e.jpg](../iclr_results/1024_Dynamic-SUPERB Phase-2_ A Collaboratively Expanding Benchmark for Measuring the Capabilities of Spok/tables/c6a5ab5768bf1302b60d50dfdf396ed5aebe2092f2671b1865f3b102ccb0ac2e.jpg)

![f670d19d52092f0bf425c590979d2bc2dd12a95dd56d737a128f4395c431b8fc.jpg](../iclr_results/1024_Dynamic-SUPERB Phase-2_ A Collaboratively Expanding Benchmark for Measuring the Capabilities of Spok/tables/f670d19d52092f0bf425c590979d2bc2dd12a95dd56d737a128f4395c431b8fc.jpg)

![faa74e20cf35a986a73fb172d0255c0f945d53801e8de1567d30c07200564f27.jpg](../iclr_results/1024_Dynamic-SUPERB Phase-2_ A Collaboratively Expanding Benchmark for Measuring the Capabilities of Spok/tables/faa74e20cf35a986a73fb172d0255c0f945d53801e8de1567d30c07200564f27.jpg)

## Train Small, Infer Large: Memory-Efficient LoRA Training for Large Language Models


### Images

![092a752b9bdd2c148dea69f0a3e1d79372d03f2e72a58a1aa7386e59c6620d2b.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/images/092a752b9bdd2c148dea69f0a3e1d79372d03f2e72a58a1aa7386e59c6620d2b.jpg)

![18298362d5573d18199dfd5ae6924661a1bcb23e554856aa0bba5a95dc432887.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/images/18298362d5573d18199dfd5ae6924661a1bcb23e554856aa0bba5a95dc432887.jpg)

![1ce701bc2990e0ca624f2e2dcecf039d8f69bc7c480804499d3d2a089606c840.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/images/1ce701bc2990e0ca624f2e2dcecf039d8f69bc7c480804499d3d2a089606c840.jpg)

![2bdd0676d43892d2980ef55f0b27d1033653151b2fd6f6fad0f04e05e2e686c1.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/images/2bdd0676d43892d2980ef55f0b27d1033653151b2fd6f6fad0f04e05e2e686c1.jpg)

![3c3aff4f388760a484e370030ec89a542429231f3128ceaf7cb70bb90c1751c4.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/images/3c3aff4f388760a484e370030ec89a542429231f3128ceaf7cb70bb90c1751c4.jpg)

![4f4c6a057bd5183e7f584a80eedd1fc9e9663e84ebd7eec6aca0c1c3f4b56fd6.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/images/4f4c6a057bd5183e7f584a80eedd1fc9e9663e84ebd7eec6aca0c1c3f4b56fd6.jpg)

![695dc4dedcf1020c61c11977a05aefadad76a05911e80bf294322e0bfc1d2193.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/images/695dc4dedcf1020c61c11977a05aefadad76a05911e80bf294322e0bfc1d2193.jpg)

![8274fb43ad5e7c61983f11c97a7a18042550624a351fd611ba1aa148d88d427a.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/images/8274fb43ad5e7c61983f11c97a7a18042550624a351fd611ba1aa148d88d427a.jpg)

![880dc1996981775313c6d10d1fba586102be107bacc8b2b399fe589be986cdc9.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/images/880dc1996981775313c6d10d1fba586102be107bacc8b2b399fe589be986cdc9.jpg)

![9735ae4323c6d3d70dae79623683e5b31e131c2a26eef291a0c6fe18eb693729.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/images/9735ae4323c6d3d70dae79623683e5b31e131c2a26eef291a0c6fe18eb693729.jpg)

![b797fc8e2031870a2b7ec9b8f76145ce682851b130e580db34d4a0416bb082cd.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/images/b797fc8e2031870a2b7ec9b8f76145ce682851b130e580db34d4a0416bb082cd.jpg)

![b864a6bc430af1d0f7ecc325f85d4129fe0dc855b389f400b6d948c5411a034d.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/images/b864a6bc430af1d0f7ecc325f85d4129fe0dc855b389f400b6d948c5411a034d.jpg)

![bfc7d13286083208ea1001eb82104755ac558783f4e618bc89560e1ab72b81c1.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/images/bfc7d13286083208ea1001eb82104755ac558783f4e618bc89560e1ab72b81c1.jpg)

![d55a7a1ea860b7f41cb4c311bb9692fed376c7b27701638754644906ce14215b.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/images/d55a7a1ea860b7f41cb4c311bb9692fed376c7b27701638754644906ce14215b.jpg)

![d651f370f2ed5e581e1681f5de0c1fe6b5eeed8b059db060bfed0488eba0f183.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/images/d651f370f2ed5e581e1681f5de0c1fe6b5eeed8b059db060bfed0488eba0f183.jpg)

![e5ae7e0018eff3ecee9e8030acc9f7024d384a3ca2ff48b511b8fae625c3c555.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/images/e5ae7e0018eff3ecee9e8030acc9f7024d384a3ca2ff48b511b8fae625c3c555.jpg)

### Tables

![402432f5b47d8654a6f5654356a347b844e2a117ad230346f810038cc20dccdd.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/tables/402432f5b47d8654a6f5654356a347b844e2a117ad230346f810038cc20dccdd.jpg)

![4633c1b14f8b7d84056fd616e7b138309d0da1c54a8ec58426e123b5e0839dd1.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/tables/4633c1b14f8b7d84056fd616e7b138309d0da1c54a8ec58426e123b5e0839dd1.jpg)

![4dae7de5785f7510c33ffe2faf2be4f0197bd4b638e2976422dc481c76cacc44.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/tables/4dae7de5785f7510c33ffe2faf2be4f0197bd4b638e2976422dc481c76cacc44.jpg)

![806a008fc9e8c4d1d66cec0f167ecc736fb52ce407ec44972a4a52b7908f38f8.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/tables/806a008fc9e8c4d1d66cec0f167ecc736fb52ce407ec44972a4a52b7908f38f8.jpg)

![8b2ba7ac13b6cbca029093c147424d1cc67bcdfba525485ab16e3d8eec32cf8b.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/tables/8b2ba7ac13b6cbca029093c147424d1cc67bcdfba525485ab16e3d8eec32cf8b.jpg)

![9205471278a8fc1c85c72dc9a8dad0fc32850f14509f8957a7137c590ab3ccc7.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/tables/9205471278a8fc1c85c72dc9a8dad0fc32850f14509f8957a7137c590ab3ccc7.jpg)

![a34b2f875c85b5c4fb9ff590cdd68ba01fa08d102ca6f8d881f62b995cf505eb.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/tables/a34b2f875c85b5c4fb9ff590cdd68ba01fa08d102ca6f8d881f62b995cf505eb.jpg)

![f45e1a430de5a4bd2e3e2acd259295d2010ff0941f2f8fa4a896ab5ff69307d7.jpg](../iclr_results/1025_Train Small, Infer Large_ Memory-Efficient LoRA Training for Large Language Models/tables/f45e1a430de5a4bd2e3e2acd259295d2010ff0941f2f8fa4a896ab5ff69307d7.jpg)

## Robust LLM safeguarding via refusal feature adversarial training


### Images

![28c3a2d83ac5df757a60d432f561e8a77d520335d3cc6af88d6f5d44e3f80665.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/images/28c3a2d83ac5df757a60d432f561e8a77d520335d3cc6af88d6f5d44e3f80665.jpg)

![39bcd63c4277d057c78917139b2f3b0ba9272937b6db57146498a7521f36d68b.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/images/39bcd63c4277d057c78917139b2f3b0ba9272937b6db57146498a7521f36d68b.jpg)

![488eda7f0ea51ba515171cdc916fc90dda830c4ed655398d144bd87f4929df80.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/images/488eda7f0ea51ba515171cdc916fc90dda830c4ed655398d144bd87f4929df80.jpg)

![527fefd4cfef7d991980d9594785cadb227794ec8279ef03350d97e633a7ee71.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/images/527fefd4cfef7d991980d9594785cadb227794ec8279ef03350d97e633a7ee71.jpg)

![6e4279a042e3a1c54cc7dee98f335a3ad67c390b2f02a9b2724c15bd4a6ab71c.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/images/6e4279a042e3a1c54cc7dee98f335a3ad67c390b2f02a9b2724c15bd4a6ab71c.jpg)

![6f3c64fa81d3e4f10e87804d5e26f8d939264ad3669a0a2ef61dd38c1e49b269.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/images/6f3c64fa81d3e4f10e87804d5e26f8d939264ad3669a0a2ef61dd38c1e49b269.jpg)

![78f7d4df9746213e2299d8ac1fb84c1f3103ba3585a818b9449be41b2b8a15ff.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/images/78f7d4df9746213e2299d8ac1fb84c1f3103ba3585a818b9449be41b2b8a15ff.jpg)

![7922f2f6c82b0400079c9684e3b710d1b8e2a8e624a0a87d013f5475993493fc.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/images/7922f2f6c82b0400079c9684e3b710d1b8e2a8e624a0a87d013f5475993493fc.jpg)

![894d1b02155ad6e8f2530564653bad9f185b1557828e7770973405578e35c53b.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/images/894d1b02155ad6e8f2530564653bad9f185b1557828e7770973405578e35c53b.jpg)

![a6de88827b8f737005159c1b19dc0b266c7d0adefdbd1c681dadf63e8f35267d.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/images/a6de88827b8f737005159c1b19dc0b266c7d0adefdbd1c681dadf63e8f35267d.jpg)

![a7435aa874e09720b4df704cc5f6d066eb877ff237adaaed4e2653dbe45b69c7.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/images/a7435aa874e09720b4df704cc5f6d066eb877ff237adaaed4e2653dbe45b69c7.jpg)

![b919544e0c879edb59b2103924936dc565e3b4020d603d295892bdce1c9298ff.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/images/b919544e0c879edb59b2103924936dc565e3b4020d603d295892bdce1c9298ff.jpg)

![ba12429a1a4c48274ff85f4ba36d23076269dfcf35312988e51f1acedb9527ea.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/images/ba12429a1a4c48274ff85f4ba36d23076269dfcf35312988e51f1acedb9527ea.jpg)

![c63f8b86ae64e1b0b88c70f4da64ecfd7ff85ca74b9f7d6c919fa6833bcc745f.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/images/c63f8b86ae64e1b0b88c70f4da64ecfd7ff85ca74b9f7d6c919fa6833bcc745f.jpg)

![c852bdeceb3d6758d3dac946a4bceb32d652b2da9af95b08b06da2ec57fc1dd0.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/images/c852bdeceb3d6758d3dac946a4bceb32d652b2da9af95b08b06da2ec57fc1dd0.jpg)

![d3101870538949c0c651922ef6c518825f282640ba91c9153b3eab7e57f58e88.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/images/d3101870538949c0c651922ef6c518825f282640ba91c9153b3eab7e57f58e88.jpg)

![d88aae413bd7c1955b4cb70863c84106d11d382d58bcfd1b60461d51bdf84f61.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/images/d88aae413bd7c1955b4cb70863c84106d11d382d58bcfd1b60461d51bdf84f61.jpg)

![f75b413b4c0fd48b625846790a6d6ef80bc49ff878268934d33eb63ccbc1509b.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/images/f75b413b4c0fd48b625846790a6d6ef80bc49ff878268934d33eb63ccbc1509b.jpg)

![fd775ddbd021d5b07705490b5d7441b33bf4ccdc1d76b0a402ebc72ab697b313.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/images/fd775ddbd021d5b07705490b5d7441b33bf4ccdc1d76b0a402ebc72ab697b313.jpg)

### Tables

![0065d5024c461a5cfd09c11a4db38317fa74cf6a3155c6cc5e5bb715b92de937.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/tables/0065d5024c461a5cfd09c11a4db38317fa74cf6a3155c6cc5e5bb715b92de937.jpg)

![1420277c97182ec69cf996a2ed3ea9ed3940448fdac72c2ddb48c73413824e9a.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/tables/1420277c97182ec69cf996a2ed3ea9ed3940448fdac72c2ddb48c73413824e9a.jpg)

![19ebd38e2761dc047f2e9f85de32ce73b431ae886878cf32d33cb3959d453b4e.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/tables/19ebd38e2761dc047f2e9f85de32ce73b431ae886878cf32d33cb3959d453b4e.jpg)

![1c038f796c04596e5ea3b5915b4970dec4f276f84c6a19c0fa4163d660b6b603.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/tables/1c038f796c04596e5ea3b5915b4970dec4f276f84c6a19c0fa4163d660b6b603.jpg)

![414afa9a24ee1b7ed97f26012daf721d0d9d587fd1eeb0217cbbe53c44d7f1ce.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/tables/414afa9a24ee1b7ed97f26012daf721d0d9d587fd1eeb0217cbbe53c44d7f1ce.jpg)

![5444f72329c8040efd482668958f180b39c4a6fadcc337e814ba919f8becb985.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/tables/5444f72329c8040efd482668958f180b39c4a6fadcc337e814ba919f8becb985.jpg)

![8db28a916d57a69159f99a001f4afc44c5ca52fe745d8ddfbcab7c369dd11c83.jpg](../iclr_results/1026_Robust LLM safeguarding via refusal feature adversarial training/tables/8db28a916d57a69159f99a001f4afc44c5ca52fe745d8ddfbcab7c369dd11c83.jpg)

## MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models


### Images

![0543176ba74572c14a9877f9cf4fcd6abc751b63379c31b8b699e306ce135b2e.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/images/0543176ba74572c14a9877f9cf4fcd6abc751b63379c31b8b699e306ce135b2e.jpg)

![1eb37de89665bbb9b698c38f0ea17c3668cbe14bc27f210a5dda53fbda77a3b1.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/images/1eb37de89665bbb9b698c38f0ea17c3668cbe14bc27f210a5dda53fbda77a3b1.jpg)

![65f5b57799b09c3aefcd6b014029f289252060fbe2e0cd4ad028d6408c702e0c.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/images/65f5b57799b09c3aefcd6b014029f289252060fbe2e0cd4ad028d6408c702e0c.jpg)

![9c22cd9a9938b90245a8730b8140b1b1e6a66f3a28f8d89ee03fc2e4cf93f885.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/images/9c22cd9a9938b90245a8730b8140b1b1e6a66f3a28f8d89ee03fc2e4cf93f885.jpg)

![a109e21cc1c62a43e719ccc13b33258710f5c1f45be93eb164cff68297eb692f.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/images/a109e21cc1c62a43e719ccc13b33258710f5c1f45be93eb164cff68297eb692f.jpg)

### Tables

![04bec53b9c96f313cd754c319aec12aa17ee02ca4f19b197e00d8d4efbcde591.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/tables/04bec53b9c96f313cd754c319aec12aa17ee02ca4f19b197e00d8d4efbcde591.jpg)

![0925317fc04e0e75a1ac9bc853b5c08b6f0d7fc2377742737bd284ca08b84d36.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/tables/0925317fc04e0e75a1ac9bc853b5c08b6f0d7fc2377742737bd284ca08b84d36.jpg)

![1224f12cff25bf07fdd216314582ef8feb63a2db55a9dfb8121b273d0fdcdbcd.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/tables/1224f12cff25bf07fdd216314582ef8feb63a2db55a9dfb8121b273d0fdcdbcd.jpg)

![2f4c9c43c3edfa620424c18fbcda94b76cf46aec0922493a826b2b133f6e7cc1.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/tables/2f4c9c43c3edfa620424c18fbcda94b76cf46aec0922493a826b2b133f6e7cc1.jpg)

![30b1e3561ba1f81bd3ffc12dc46c9846fc90e448d08804d0be94e518370768a9.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/tables/30b1e3561ba1f81bd3ffc12dc46c9846fc90e448d08804d0be94e518370768a9.jpg)

![3f6c3a57f02dc8fa8d845c1af7ccb943c333f91c3ea7e21ecaddfa53bab47658.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/tables/3f6c3a57f02dc8fa8d845c1af7ccb943c333f91c3ea7e21ecaddfa53bab47658.jpg)

![47341dc81c485b8fae467a41c674a9d15b39991961446ba0425dcc7dcb67c4ce.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/tables/47341dc81c485b8fae467a41c674a9d15b39991961446ba0425dcc7dcb67c4ce.jpg)

![55608529f879d3377df09657e2a71a3b15538d2379da5a1a88ca091be4c9d031.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/tables/55608529f879d3377df09657e2a71a3b15538d2379da5a1a88ca091be4c9d031.jpg)

![5aa070507883a151a23e507420d01e2b71251ea97fc19847bfd9de4b5f83da03.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/tables/5aa070507883a151a23e507420d01e2b71251ea97fc19847bfd9de4b5f83da03.jpg)

![6f26381659ae0f13b227d8c638db7b14cdc58b32433631fa76da0c91ee5f66ef.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/tables/6f26381659ae0f13b227d8c638db7b14cdc58b32433631fa76da0c91ee5f66ef.jpg)

![6f5ae68ea333071ed7e01155918ea751e71278e155b9ca16f264288b5d2fc093.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/tables/6f5ae68ea333071ed7e01155918ea751e71278e155b9ca16f264288b5d2fc093.jpg)

![a127c461707d72bab09423513480fde2f038804820e42c09582ff4b848e6c663.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/tables/a127c461707d72bab09423513480fde2f038804820e42c09582ff4b848e6c663.jpg)

![a33a8d1b3428ac6534394ed548f017bbe79a2a5b40b055b5b6a2183ab18585d6.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/tables/a33a8d1b3428ac6534394ed548f017bbe79a2a5b40b055b5b6a2183ab18585d6.jpg)

![b5a324f6c0aff076bffec1834e99f0858d573786ad9ebb63d8e03d3d676897fb.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/tables/b5a324f6c0aff076bffec1834e99f0858d573786ad9ebb63d8e03d3d676897fb.jpg)

![b8e7eef2b61acd0ba81f09983a110bedccdb14b5623e1c9ede19e2e1db2d309a.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/tables/b8e7eef2b61acd0ba81f09983a110bedccdb14b5623e1c9ede19e2e1db2d309a.jpg)

![bb102238bbf3700228c3b87f630ec75a661fa9626739b4abf59449d1f7a2a1ed.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/tables/bb102238bbf3700228c3b87f630ec75a661fa9626739b4abf59449d1f7a2a1ed.jpg)

![bb2c5d7dbcb070e35bb23ad9b6bb3e62ac82e220cebe04006eb98398650cb518.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/tables/bb2c5d7dbcb070e35bb23ad9b6bb3e62ac82e220cebe04006eb98398650cb518.jpg)

![ddb1575a04a561e80a4db734a040ccba14607fbbecaabe09d970fc0ca5be83d6.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/tables/ddb1575a04a561e80a4db734a040ccba14607fbbecaabe09d970fc0ca5be83d6.jpg)

![e4581970ddd7134c85bbde8c009a211a7b7d76890be03d6edb7e875f8d4ce2e1.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/tables/e4581970ddd7134c85bbde8c009a211a7b7d76890be03d6edb7e875f8d4ce2e1.jpg)

![eba43fc2a4321e3f15fdebdd1e5d6366f61090e6a3984b36a72eec30ab8afdee.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/tables/eba43fc2a4321e3f15fdebdd1e5d6366f61090e6a3984b36a72eec30ab8afdee.jpg)

![ebee390efed0bb4f93578eaad7f3e63c654fb06a9660e3db0dce763d28378c58.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/tables/ebee390efed0bb4f93578eaad7f3e63c654fb06a9660e3db0dce763d28378c58.jpg)

![ee7901c55050f02d6dfaca00bddafb6f6543346b681c8eb578d05ed124febfb1.jpg](../iclr_results/1027_MMed-RAG_ Versatile Multimodal RAG System for Medical Vision Language Models/tables/ee7901c55050f02d6dfaca00bddafb6f6543346b681c8eb578d05ed124febfb1.jpg)

## What Are Good Positional Encodings for Directed Graphs?


### Images

![30f17b1e761d6a72c072e02b0dcd0231a0316aa15aab1d94f662ed49abf31da1.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/images/30f17b1e761d6a72c072e02b0dcd0231a0316aa15aab1d94f662ed49abf31da1.jpg)

![407406a2f651c48cd2db17cdce2fb3f58a6b1e36744994f8d518ee6c79b8dc87.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/images/407406a2f651c48cd2db17cdce2fb3f58a6b1e36744994f8d518ee6c79b8dc87.jpg)

![443766c76a79b3474abcfa8c7de6501ca7d6bb8507271d24a3bd4fa86f77bf0c.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/images/443766c76a79b3474abcfa8c7de6501ca7d6bb8507271d24a3bd4fa86f77bf0c.jpg)

![827a4539bc070a8776a7e706583686e3493192e14048057d8756bfb2b16d8c80.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/images/827a4539bc070a8776a7e706583686e3493192e14048057d8756bfb2b16d8c80.jpg)

![854d85dee3895c8ddeb7cecfa0f41e19b7859436b5a659f7f5f56813ec6b351e.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/images/854d85dee3895c8ddeb7cecfa0f41e19b7859436b5a659f7f5f56813ec6b351e.jpg)

![b30bb1c3f549774e1c6b11c05ca06cebec5eaa3c90846074f2900209304b1035.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/images/b30bb1c3f549774e1c6b11c05ca06cebec5eaa3c90846074f2900209304b1035.jpg)

![d1dfb4f7f73e7df7b0991098f8d7280d6038d8915364aef0a459d422afed80fd.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/images/d1dfb4f7f73e7df7b0991098f8d7280d6038d8915364aef0a459d422afed80fd.jpg)

![d67b0c7b3434f3588b8bce12b09655385baafc4133f7298f06087b740fe725e3.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/images/d67b0c7b3434f3588b8bce12b09655385baafc4133f7298f06087b740fe725e3.jpg)

![f71d027a8f81cba211f4deb88c67691ced9790d3dca5e59a6d2709e9dc77d21b.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/images/f71d027a8f81cba211f4deb88c67691ced9790d3dca5e59a6d2709e9dc77d21b.jpg)

### Tables

![20d77dd4c39bb9e6117833cce359dd179eb41a194f69f934fa5c848a4f5821f5.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/tables/20d77dd4c39bb9e6117833cce359dd179eb41a194f69f934fa5c848a4f5821f5.jpg)

![2fc130423fffbc16ef9785178b55dd0859427e426fcf0fcb4a7f51cfe20cea4b.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/tables/2fc130423fffbc16ef9785178b55dd0859427e426fcf0fcb4a7f51cfe20cea4b.jpg)

![30c6235c1365017e3b82923063de98b362759515c4a6ae46de567b6261d65d2d.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/tables/30c6235c1365017e3b82923063de98b362759515c4a6ae46de567b6261d65d2d.jpg)

![3f8f64be6a848b0337baa558edb66248adaca5d98646a651353ebd8d884899b3.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/tables/3f8f64be6a848b0337baa558edb66248adaca5d98646a651353ebd8d884899b3.jpg)

![50a31d6a79839c8121f4dcf15477848c3f6e0410cc7aed08a1544197e1e61421.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/tables/50a31d6a79839c8121f4dcf15477848c3f6e0410cc7aed08a1544197e1e61421.jpg)

![559ea3d18b26859f90a3a9f9a7ce1e08699cf2043eba82b4ad754f8c9a8b99c7.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/tables/559ea3d18b26859f90a3a9f9a7ce1e08699cf2043eba82b4ad754f8c9a8b99c7.jpg)

![6e074e6136a4511cea2d115da92600309962294516eb60d7f2348d3493b2b078.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/tables/6e074e6136a4511cea2d115da92600309962294516eb60d7f2348d3493b2b078.jpg)

![80a2fbe31a51336f80010cb673f9c16e18589aa7518b7a3e39f9f057c3d98fbf.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/tables/80a2fbe31a51336f80010cb673f9c16e18589aa7518b7a3e39f9f057c3d98fbf.jpg)

![86f23c4dd68f7bbb0699d8b5801c4ee7efa9b094f434d4576f94916f734e1d91.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/tables/86f23c4dd68f7bbb0699d8b5801c4ee7efa9b094f434d4576f94916f734e1d91.jpg)

![8a0d18115e400ade2e8284840e1848d071b13cdcd75ea83e5c9554b44297da6b.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/tables/8a0d18115e400ade2e8284840e1848d071b13cdcd75ea83e5c9554b44297da6b.jpg)

![8f501e9a506d3779253a15edfb62d28bb49129af17e7dcc85339b583bad7bcb8.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/tables/8f501e9a506d3779253a15edfb62d28bb49129af17e7dcc85339b583bad7bcb8.jpg)

![9e43bce392d0064ef0be2758f3d35c58998c317da71f5083d2a41646e0c7e49f.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/tables/9e43bce392d0064ef0be2758f3d35c58998c317da71f5083d2a41646e0c7e49f.jpg)

![a07ccb3fae92efe1e395c8495b1f704a74a358cc6e14838a1e719e836e78154e.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/tables/a07ccb3fae92efe1e395c8495b1f704a74a358cc6e14838a1e719e836e78154e.jpg)

![a51770000af653d2c3604ff2712bf509be859eaf3363e869ca40a55514d83b73.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/tables/a51770000af653d2c3604ff2712bf509be859eaf3363e869ca40a55514d83b73.jpg)

![aa33f1d65b5769c741467075e4f4ccedb1579e48bab2b547dc0409ebab83ec68.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/tables/aa33f1d65b5769c741467075e4f4ccedb1579e48bab2b547dc0409ebab83ec68.jpg)

![c05757637094cb832b6fa8844a736fc4cd5532dae51afaccfaf19e1ca1eda2eb.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/tables/c05757637094cb832b6fa8844a736fc4cd5532dae51afaccfaf19e1ca1eda2eb.jpg)

![c44cfd2901e128c258c88fc211237adf1d4acc708b966bee1c7d7d3ef6b49a27.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/tables/c44cfd2901e128c258c88fc211237adf1d4acc708b966bee1c7d7d3ef6b49a27.jpg)

![d3d4f40a1feaf519f997ce8c24d8ea073628a57757554ad2becaa8f994713699.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/tables/d3d4f40a1feaf519f997ce8c24d8ea073628a57757554ad2becaa8f994713699.jpg)

![f1b8d4abe36724eddb3b05ee9061a61c537a6364e4410d13127bd36a3f3a2750.jpg](../iclr_results/1028_What Are Good Positional Encodings for Directed Graphs_/tables/f1b8d4abe36724eddb3b05ee9061a61c537a6364e4410d13127bd36a3f3a2750.jpg)

## Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs


### Images

![10e17e9e02f0dd5cde9c5acbfe8c7e1d3f53d0195679c4b87f0cf181484d7780.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/images/10e17e9e02f0dd5cde9c5acbfe8c7e1d3f53d0195679c4b87f0cf181484d7780.jpg)

![302d9fdb1928d013a379ccd8d2cd272b88e8583867dd001f670d60d281f35d92.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/images/302d9fdb1928d013a379ccd8d2cd272b88e8583867dd001f670d60d281f35d92.jpg)

![3661149036a00743182320ff9cb85d690aeca92569e30781e4bb2a45b1e18bab.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/images/3661149036a00743182320ff9cb85d690aeca92569e30781e4bb2a45b1e18bab.jpg)

![6096896f382fa5bed12e6a0b0a1f95ab1f352e85353eeab0ef7f46942ed4fa38.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/images/6096896f382fa5bed12e6a0b0a1f95ab1f352e85353eeab0ef7f46942ed4fa38.jpg)

![6bbe8ed216bbe388cdba647fc78c567e0c19b20b3383b0ea4e5c4d1600330882.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/images/6bbe8ed216bbe388cdba647fc78c567e0c19b20b3383b0ea4e5c4d1600330882.jpg)

![6bf61bf0b98fc43b791a2d2573dc39647d55ab8b9c5fd5f0a236becb226b7673.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/images/6bf61bf0b98fc43b791a2d2573dc39647d55ab8b9c5fd5f0a236becb226b7673.jpg)

![7b1f51a3c224885e0c198c7142759bc9b268aa01dd2d1f95b161191ab7cd37bf.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/images/7b1f51a3c224885e0c198c7142759bc9b268aa01dd2d1f95b161191ab7cd37bf.jpg)

![81b8ace151f0fc2124b509875c7255bc1cccbfe2f8424f101974943527d8d80d.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/images/81b8ace151f0fc2124b509875c7255bc1cccbfe2f8424f101974943527d8d80d.jpg)

![abd8e7a9dbc984dc4a522d59e354c749e6a778fec2edbbeb8219bb1e2ef5a19e.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/images/abd8e7a9dbc984dc4a522d59e354c749e6a778fec2edbbeb8219bb1e2ef5a19e.jpg)

![badd3359101b1ba2d5447515b02b1f0b939d2d55ee51aca2e328e6caa527c657.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/images/badd3359101b1ba2d5447515b02b1f0b939d2d55ee51aca2e328e6caa527c657.jpg)

![cc0dda4d84819cc7e03a580213ad7a9eb78f254ac607b104e56764b85f0eb35f.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/images/cc0dda4d84819cc7e03a580213ad7a9eb78f254ac607b104e56764b85f0eb35f.jpg)

![cfc3ec38396f67359407b02ce5ea5a796239f647b073580fb13e7446a22c23be.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/images/cfc3ec38396f67359407b02ce5ea5a796239f647b073580fb13e7446a22c23be.jpg)

![fca21a94d210ce27ee37702e4d7f75e29d8e5173baf4f2c995aad1704d66bfa2.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/images/fca21a94d210ce27ee37702e4d7f75e29d8e5173baf4f2c995aad1704d66bfa2.jpg)

![fe03104f460daa03673cf726b93d3d8b69095a08dafd275aeb2b7ac98a0e016c.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/images/fe03104f460daa03673cf726b93d3d8b69095a08dafd275aeb2b7ac98a0e016c.jpg)

### Tables

![0c4ebc0512501b5f7cc30dbd75efc0e6120c87572ba6ece79f00253eea8d61c0.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/tables/0c4ebc0512501b5f7cc30dbd75efc0e6120c87572ba6ece79f00253eea8d61c0.jpg)

![48b124f83136a747cb26b5c1822000db097cc6be75e03f05663fd5f9255ac1f3.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/tables/48b124f83136a747cb26b5c1822000db097cc6be75e03f05663fd5f9255ac1f3.jpg)

![6608cf6c6507b268d16fbe9ef0022e4db82685115ada35e087a65ba509b622ed.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/tables/6608cf6c6507b268d16fbe9ef0022e4db82685115ada35e087a65ba509b622ed.jpg)

![6b15d94525688c867d6c5389586491add9196d5f2a66bd9517dc3b3b65a3bdaf.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/tables/6b15d94525688c867d6c5389586491add9196d5f2a66bd9517dc3b3b65a3bdaf.jpg)

![6e60ef485f62ed8f87ebb72633d19f981b7f2159ec4c3cf75a5809163e800131.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/tables/6e60ef485f62ed8f87ebb72633d19f981b7f2159ec4c3cf75a5809163e800131.jpg)

![7eeac8edbbadc24c853c2819bfdeae5bfffdab46089ffa1a978535314f56a5d4.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/tables/7eeac8edbbadc24c853c2819bfdeae5bfffdab46089ffa1a978535314f56a5d4.jpg)

![846a0bf0ad1a5eef00ee6cdf68cc3f248dcf9fe4a9c800626f3176fa8ad27d51.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/tables/846a0bf0ad1a5eef00ee6cdf68cc3f248dcf9fe4a9c800626f3176fa8ad27d51.jpg)

![af09056d000293112ac4e871a292406baa39f891043c94e1cafa831914924e88.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/tables/af09056d000293112ac4e871a292406baa39f891043c94e1cafa831914924e88.jpg)

![e51488c3445fe1e47f950f76e98fedb71fff6a5011c61f2126d4a0a4f399a463.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/tables/e51488c3445fe1e47f950f76e98fedb71fff6a5011c61f2126d4a0a4f399a463.jpg)

![f85c41c84cd0ca1e15b61f8ee44b23e199c5dbf58702b283fb4c29ef99f540c7.jpg](../iclr_results/1029_Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs/tables/f85c41c84cd0ca1e15b61f8ee44b23e199c5dbf58702b283fb4c29ef99f540c7.jpg)

## Language Models Need Inductive Biases to Count Inductively


### Images

![051f234ce6228067f4b0752ee1eb40dc1a5eb7acdc376810b527a89f7dad2f10.jpg](../iclr_results/1030_Language Models Need Inductive Biases to Count Inductively/images/051f234ce6228067f4b0752ee1eb40dc1a5eb7acdc376810b527a89f7dad2f10.jpg)

![606c19d6bc20a49d06dc050af96c3c796b64791343fa3e422ce229e843c826b6.jpg](../iclr_results/1030_Language Models Need Inductive Biases to Count Inductively/images/606c19d6bc20a49d06dc050af96c3c796b64791343fa3e422ce229e843c826b6.jpg)

![7f687291cb3f29aeef079fb271580d58b8e438c7149650ae923bb71c173db12c.jpg](../iclr_results/1030_Language Models Need Inductive Biases to Count Inductively/images/7f687291cb3f29aeef079fb271580d58b8e438c7149650ae923bb71c173db12c.jpg)

![ad0ebdc486091f09e31308133c9404df651cb49a3b3d4583accdb9c64f1c0a00.jpg](../iclr_results/1030_Language Models Need Inductive Biases to Count Inductively/images/ad0ebdc486091f09e31308133c9404df651cb49a3b3d4583accdb9c64f1c0a00.jpg)

![af42da0f23c6baa38d2c4c569c156338df8ee909124e4d2dfb9768f4564922ab.jpg](../iclr_results/1030_Language Models Need Inductive Biases to Count Inductively/images/af42da0f23c6baa38d2c4c569c156338df8ee909124e4d2dfb9768f4564922ab.jpg)

![f72c3f984e839fcdcd83aa5a7686a58b9a1786c9618f7cf77caa9b46338039d3.jpg](../iclr_results/1030_Language Models Need Inductive Biases to Count Inductively/images/f72c3f984e839fcdcd83aa5a7686a58b9a1786c9618f7cf77caa9b46338039d3.jpg)

![f753649998efbb8c39599a4694070b0466585276d161d517e06ff53c2a5c315e.jpg](../iclr_results/1030_Language Models Need Inductive Biases to Count Inductively/images/f753649998efbb8c39599a4694070b0466585276d161d517e06ff53c2a5c315e.jpg)

### Tables

![0a7005e231c2bf9c91448a0c9406be6abe7471a02d333ad4417dc982bee391c0.jpg](../iclr_results/1030_Language Models Need Inductive Biases to Count Inductively/tables/0a7005e231c2bf9c91448a0c9406be6abe7471a02d333ad4417dc982bee391c0.jpg)

![23142a8e288b00d4d660fdeb1bd0ec5fb040db0b4462cd0e166a7b7f191b7f9b.jpg](../iclr_results/1030_Language Models Need Inductive Biases to Count Inductively/tables/23142a8e288b00d4d660fdeb1bd0ec5fb040db0b4462cd0e166a7b7f191b7f9b.jpg)

![524243578454abdd5c2cfd778c10654207dc22b996f3d3da3b3fe964426c9bfe.jpg](../iclr_results/1030_Language Models Need Inductive Biases to Count Inductively/tables/524243578454abdd5c2cfd778c10654207dc22b996f3d3da3b3fe964426c9bfe.jpg)

![5611cde04e5259d3085f8e7a7fb6e92a7596eccf083dc9807c19298d3e9127d4.jpg](../iclr_results/1030_Language Models Need Inductive Biases to Count Inductively/tables/5611cde04e5259d3085f8e7a7fb6e92a7596eccf083dc9807c19298d3e9127d4.jpg)

![772c406a8b16949c7c2c7c858a05b1e66050396364bdd412542f0488fb2d487d.jpg](../iclr_results/1030_Language Models Need Inductive Biases to Count Inductively/tables/772c406a8b16949c7c2c7c858a05b1e66050396364bdd412542f0488fb2d487d.jpg)

![89dab74cd73306a0b43a0c9b2081fed5947a60e64fb81817212dba950ebac53b.jpg](../iclr_results/1030_Language Models Need Inductive Biases to Count Inductively/tables/89dab74cd73306a0b43a0c9b2081fed5947a60e64fb81817212dba950ebac53b.jpg)

![9f515365c3b61977534871cf49b83b58507e2fe9e9dabe2c6e92b7f89b4d7108.jpg](../iclr_results/1030_Language Models Need Inductive Biases to Count Inductively/tables/9f515365c3b61977534871cf49b83b58507e2fe9e9dabe2c6e92b7f89b4d7108.jpg)

![b1f4d882d3a8add2cb22ce5eabaeb335cd53da7dc8bb2b93cd17fae1452aff9f.jpg](../iclr_results/1030_Language Models Need Inductive Biases to Count Inductively/tables/b1f4d882d3a8add2cb22ce5eabaeb335cd53da7dc8bb2b93cd17fae1452aff9f.jpg)

![b61ef59798f7fa144aa4f7adb2d62e5738b69a867b6ec2a38646b65113aad254.jpg](../iclr_results/1030_Language Models Need Inductive Biases to Count Inductively/tables/b61ef59798f7fa144aa4f7adb2d62e5738b69a867b6ec2a38646b65113aad254.jpg)

![bd8e68da1fad04f5f902b3eef6a84f77491e315a343ccdaabb877f0e808a4bf6.jpg](../iclr_results/1030_Language Models Need Inductive Biases to Count Inductively/tables/bd8e68da1fad04f5f902b3eef6a84f77491e315a343ccdaabb877f0e808a4bf6.jpg)

![ef6b59977e34454f99c4a495ee13a7544a21de52704c93470b0bcd91b644dac5.jpg](../iclr_results/1030_Language Models Need Inductive Biases to Count Inductively/tables/ef6b59977e34454f99c4a495ee13a7544a21de52704c93470b0bcd91b644dac5.jpg)

## Articulate-Anything:  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Model


### Images

![055cc93b25466a6aa3fbe08bd4ab947b96d976e0fc1b9d5b55b071c5c7c9a022.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/055cc93b25466a6aa3fbe08bd4ab947b96d976e0fc1b9d5b55b071c5c7c9a022.jpg)

![1343776fc1d65f99196284e64568913fa1857dbe6791c340419f816c7ca937ef.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/1343776fc1d65f99196284e64568913fa1857dbe6791c340419f816c7ca937ef.jpg)

![19d118815246766e9d083187490cc35651d23d51e2f560aeea72f33bf7a20151.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/19d118815246766e9d083187490cc35651d23d51e2f560aeea72f33bf7a20151.jpg)

![1da2c2805776479bb144a407f4d44ce8fa38b5684db0254a00057489907ba0c1.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/1da2c2805776479bb144a407f4d44ce8fa38b5684db0254a00057489907ba0c1.jpg)

![2b065881dd4a0c442ff8b62dad39ca9a8822ac2fe43dbc21282b3f6f78d3e548.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/2b065881dd4a0c442ff8b62dad39ca9a8822ac2fe43dbc21282b3f6f78d3e548.jpg)

![2bb477decfb90a2f4963a8c73708633a939dfd304711e9c12b80042c266ff026.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/2bb477decfb90a2f4963a8c73708633a939dfd304711e9c12b80042c266ff026.jpg)

![388ec59062c5b89c334d7b0fc1a7a1c6f0b57b201dfe9c45d43e47fba7cd1266.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/388ec59062c5b89c334d7b0fc1a7a1c6f0b57b201dfe9c45d43e47fba7cd1266.jpg)

![3fb359814600b09dcb36a0927e7fa70ff8a01a494767223b9b84c8bced1a1f75.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/3fb359814600b09dcb36a0927e7fa70ff8a01a494767223b9b84c8bced1a1f75.jpg)

![408daa220d82309d1e050f2147692d39b558c5d941d748a52501535692c49e26.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/408daa220d82309d1e050f2147692d39b558c5d941d748a52501535692c49e26.jpg)

![4bb7c53bbe36521e626554980f461aad4a8b164983194bc2f351b17f7ac47d9e.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/4bb7c53bbe36521e626554980f461aad4a8b164983194bc2f351b17f7ac47d9e.jpg)

![68fcaf61955edd6cbe1edb9516a20cc457096b4fdc156983d24329a6c4fdeb09.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/68fcaf61955edd6cbe1edb9516a20cc457096b4fdc156983d24329a6c4fdeb09.jpg)

![6f02091449b7925509791d49f467b6d66a3911b459a6911ea6c520b99a53a095.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/6f02091449b7925509791d49f467b6d66a3911b459a6911ea6c520b99a53a095.jpg)

![7425894abc92837035d7103dba1457924ae1a349543ab5a2a6a73c7ee6ba564a.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/7425894abc92837035d7103dba1457924ae1a349543ab5a2a6a73c7ee6ba564a.jpg)

![96adb971162caf83b9bd0329bfe0a71a2c00a149fa16aa020f34a484b3f60e39.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/96adb971162caf83b9bd0329bfe0a71a2c00a149fa16aa020f34a484b3f60e39.jpg)

![a7cffd32734441db54c6576c34ce963b434dd93b3fba7fa25b74a497e77f1cf7.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/a7cffd32734441db54c6576c34ce963b434dd93b3fba7fa25b74a497e77f1cf7.jpg)

![b61e96ab0ca448de1027ee38c1560561d42c6731df12dfcbdb39dcdfc8bbed1a.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/b61e96ab0ca448de1027ee38c1560561d42c6731df12dfcbdb39dcdfc8bbed1a.jpg)

![c1b4263b8826c272cee133d204a550f681eeb09710b1c963194010b211396dad.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/c1b4263b8826c272cee133d204a550f681eeb09710b1c963194010b211396dad.jpg)

![cef2620efca59b1e0b83336e625f1fcefc79a20fd0fecdc10a30746764b17644.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/cef2620efca59b1e0b83336e625f1fcefc79a20fd0fecdc10a30746764b17644.jpg)

![dafdbe4ebda34fdcbfd56025a00dce38692e4586b8699da06d66ea372ffba128.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/dafdbe4ebda34fdcbfd56025a00dce38692e4586b8699da06d66ea372ffba128.jpg)

![df1207b11277dbdb43bf7896aaac9c56d9a87a94313b9dfeb2b604b7479e934c.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/df1207b11277dbdb43bf7896aaac9c56d9a87a94313b9dfeb2b604b7479e934c.jpg)

![e0acd8c9095084e06755f638e9ced5add48d5397786a156fcfe45a6c48dabdd9.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/e0acd8c9095084e06755f638e9ced5add48d5397786a156fcfe45a6c48dabdd9.jpg)

![e3744f6c972dc2a36f6daad4e2b6735f1c4932bd565603b9be51249fe421f6f0.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/e3744f6c972dc2a36f6daad4e2b6735f1c4932bd565603b9be51249fe421f6f0.jpg)

![ed8915f6694701a36821fdcb0aadafa460c79cdac77c7182f9e56179503e167e.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/ed8915f6694701a36821fdcb0aadafa460c79cdac77c7182f9e56179503e167e.jpg)

![edf5c99e7c640f48024ae579131289477b90f87fa08b747cf410af0a750272d0.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/edf5c99e7c640f48024ae579131289477b90f87fa08b747cf410af0a750272d0.jpg)

![ee561674ed0c0b8417feefce01c0f4b343fedaa534e2ee9d30a61a27e4d1b52f.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/ee561674ed0c0b8417feefce01c0f4b343fedaa534e2ee9d30a61a27e4d1b52f.jpg)

![ef8bafbc4078dca50d143cfe6393130adfff9e470a0e5789dc4d646978707cfc.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/images/ef8bafbc4078dca50d143cfe6393130adfff9e470a0e5789dc4d646978707cfc.jpg)

### Tables

![2cac75011cac8fe4534eed2a3216598ad652060f3505135936950fa2c3d3de04.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/tables/2cac75011cac8fe4534eed2a3216598ad652060f3505135936950fa2c3d3de04.jpg)

![3234886ee8f8cb873ed770ee0f2a2f381d9b4887019577120e0764acd98ce294.jpg](../iclr_results/1031_Articulate-Anything_  Automatic Modeling of Articulated Objects via a Vision-Language Foundation Mod/tables/3234886ee8f8cb873ed770ee0f2a2f381d9b4887019577120e0764acd98ce294.jpg)

## Jailbreak Antidote: Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Language Models


### Images

![04e18878ada848692d9d92a37e4cbae90fed22e320c0dd92e0a0a959d37e41ca.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/images/04e18878ada848692d9d92a37e4cbae90fed22e320c0dd92e0a0a959d37e41ca.jpg)

![52e2c86d0a49b8fec1b712d91afae0994f6eda1871cdd5bb86d804fa588b5825.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/images/52e2c86d0a49b8fec1b712d91afae0994f6eda1871cdd5bb86d804fa588b5825.jpg)

![6ff609c869a2b843a7ccc28e5485907380e5f40f8b121cd1209e7df06749a763.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/images/6ff609c869a2b843a7ccc28e5485907380e5f40f8b121cd1209e7df06749a763.jpg)

![76b4d55a6e3f0be24ea6f09ff5e7c34edd9ba077184f8545c717eafe93821640.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/images/76b4d55a6e3f0be24ea6f09ff5e7c34edd9ba077184f8545c717eafe93821640.jpg)

![7afd0246b91d020d77dae82f59d5397661928892d3aa8b5ea17610b438153592.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/images/7afd0246b91d020d77dae82f59d5397661928892d3aa8b5ea17610b438153592.jpg)

![7e4a4917d4bf7b53a2943fc6761b796becf99ab0d234b6d49bd0fa51e8c9168f.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/images/7e4a4917d4bf7b53a2943fc6761b796becf99ab0d234b6d49bd0fa51e8c9168f.jpg)

![879b9eddc803f1ee512183418071a46491aac42919974ecf2e38dbbabdd77d07.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/images/879b9eddc803f1ee512183418071a46491aac42919974ecf2e38dbbabdd77d07.jpg)

![a9d07df02069d3227a3d58b40748e5d902c4b1cfa7c50edc1745dbe8143736b3.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/images/a9d07df02069d3227a3d58b40748e5d902c4b1cfa7c50edc1745dbe8143736b3.jpg)

![aab9e585b8fb748e71b4100aff646a954bb7a4d2b4c76015686cb9cba85162ee.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/images/aab9e585b8fb748e71b4100aff646a954bb7a4d2b4c76015686cb9cba85162ee.jpg)

![c6f80273f44e3a307ac833b7be5e34be473aacc95854b2ff52b270f15883784b.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/images/c6f80273f44e3a307ac833b7be5e34be473aacc95854b2ff52b270f15883784b.jpg)

![cb545b676679b0404acfa03cccbba068d76235ed9cd5bdd0ff680bf4caf7d4e1.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/images/cb545b676679b0404acfa03cccbba068d76235ed9cd5bdd0ff680bf4caf7d4e1.jpg)

![d27db15d1dedc903e270958b8e9d69db1964992d556f165d9b48b5d6832d6b01.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/images/d27db15d1dedc903e270958b8e9d69db1964992d556f165d9b48b5d6832d6b01.jpg)

![d443b0596ca94da607cf24bdb7c5041faf4a63a3bb7107dd2a43365f37db54c9.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/images/d443b0596ca94da607cf24bdb7c5041faf4a63a3bb7107dd2a43365f37db54c9.jpg)

![d82f564bac5c42cdc13bc1fe36eac0a6b9c8127204177c82400c34621337a1bf.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/images/d82f564bac5c42cdc13bc1fe36eac0a6b9c8127204177c82400c34621337a1bf.jpg)

![da43ac429c8ec2abef1339d6dbc1fcae4826070a9d471611f04d8000c18bf2bf.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/images/da43ac429c8ec2abef1339d6dbc1fcae4826070a9d471611f04d8000c18bf2bf.jpg)

![e944d25ca2f9a339251f4b2f14b2db01f03310ed3a747f908789e198d94f07b7.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/images/e944d25ca2f9a339251f4b2f14b2db01f03310ed3a747f908789e198d94f07b7.jpg)

![ffefbce06fe782e821a18d3d500cfc81643474f6163b933d497edc33b9ac8b12.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/images/ffefbce06fe782e821a18d3d500cfc81643474f6163b933d497edc33b9ac8b12.jpg)

### Tables

![2d6547e65ecedc9769f52cd9e10dc756774442260bcbe1872b760e4ed2058823.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/tables/2d6547e65ecedc9769f52cd9e10dc756774442260bcbe1872b760e4ed2058823.jpg)

![328edf26712aebb026744cccb6adc72fcd6b3c2e3ebe6efc76819df048155602.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/tables/328edf26712aebb026744cccb6adc72fcd6b3c2e3ebe6efc76819df048155602.jpg)

![447e5a33ee215746405d533dd6d6f13fc236913ae185c2139e566164e2778c22.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/tables/447e5a33ee215746405d533dd6d6f13fc236913ae185c2139e566164e2778c22.jpg)

![6f24a63a6530596acb9cc2b9f896b3a0fdc515e4f69ca416a8c0bf5cebd57560.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/tables/6f24a63a6530596acb9cc2b9f896b3a0fdc515e4f69ca416a8c0bf5cebd57560.jpg)

![78a13f4bb329f995fcda1a44658c7b368af750a012d9cbc5b86bcfca1aeb23f7.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/tables/78a13f4bb329f995fcda1a44658c7b368af750a012d9cbc5b86bcfca1aeb23f7.jpg)

![7f9916c432e8adfa40e1ed699ee3cfbda3e75fbba877b5e8f8878e3d50393f02.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/tables/7f9916c432e8adfa40e1ed699ee3cfbda3e75fbba877b5e8f8878e3d50393f02.jpg)

![8530371530ef14ec6062134b8ca302d9e905e6a4bb163aeefd0c8cb028333977.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/tables/8530371530ef14ec6062134b8ca302d9e905e6a4bb163aeefd0c8cb028333977.jpg)

![9f5f1345bf4c60e5f4a6922100385e252a515e675195c86a585b11f9722e9338.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/tables/9f5f1345bf4c60e5f4a6922100385e252a515e675195c86a585b11f9722e9338.jpg)

![bbcaddcbd8c2636a6add571d8013504196866d162cf21060ae1c9bbf5d9d3049.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/tables/bbcaddcbd8c2636a6add571d8013504196866d162cf21060ae1c9bbf5d9d3049.jpg)

![c8f263d7a57a2ff7e2d1385f58d51af2178bbbc3af087db104b3f4dd1d818921.jpg](../iclr_results/1032_Jailbreak Antidote_ Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Lan/tables/c8f263d7a57a2ff7e2d1385f58d51af2178bbbc3af087db104b3f4dd1d818921.jpg)

## Swing-by Dynamics in Concept Learning and Compositional Generalization


### Images

![031340361e635326bbe8898b2a4d4546da498856e8079d3108fb12287a4b9004.jpg](../iclr_results/1033_Swing-by Dynamics in Concept Learning and Compositional Generalization/images/031340361e635326bbe8898b2a4d4546da498856e8079d3108fb12287a4b9004.jpg)

![0558fcc4e706b7eeae21b92ae8701373bbdd9e3935ef7a5facbdf2bda2c2398a.jpg](../iclr_results/1033_Swing-by Dynamics in Concept Learning and Compositional Generalization/images/0558fcc4e706b7eeae21b92ae8701373bbdd9e3935ef7a5facbdf2bda2c2398a.jpg)

![0f4e76f7fe4f695516d6dc57e118673e90ece64f937ea56fbacb744d0a198834.jpg](../iclr_results/1033_Swing-by Dynamics in Concept Learning and Compositional Generalization/images/0f4e76f7fe4f695516d6dc57e118673e90ece64f937ea56fbacb744d0a198834.jpg)

![154cef51c8d23b1f2f5bacbdb0fbfb94b207d77eaf2058fceb50a4f702285cd3.jpg](../iclr_results/1033_Swing-by Dynamics in Concept Learning and Compositional Generalization/images/154cef51c8d23b1f2f5bacbdb0fbfb94b207d77eaf2058fceb50a4f702285cd3.jpg)

![2dcfb253319ed55f92ad4875ecd73e0bfe32cc6ac1fef000330407d7a774da46.jpg](../iclr_results/1033_Swing-by Dynamics in Concept Learning and Compositional Generalization/images/2dcfb253319ed55f92ad4875ecd73e0bfe32cc6ac1fef000330407d7a774da46.jpg)

![42694d4d4ad76cb6ec56a4354f30c0748177f2a7f0e8c7a025b1231e85a92f69.jpg](../iclr_results/1033_Swing-by Dynamics in Concept Learning and Compositional Generalization/images/42694d4d4ad76cb6ec56a4354f30c0748177f2a7f0e8c7a025b1231e85a92f69.jpg)

![49d9b73c0f31ab6a43582188f8986d00258a44a5dae74cc2cc0e40b033510827.jpg](../iclr_results/1033_Swing-by Dynamics in Concept Learning and Compositional Generalization/images/49d9b73c0f31ab6a43582188f8986d00258a44a5dae74cc2cc0e40b033510827.jpg)

![65ec5e078cd737d02c2845b662a9ac6be58a6afd2d7e7c0db2949bc1a8ac72a7.jpg](../iclr_results/1033_Swing-by Dynamics in Concept Learning and Compositional Generalization/images/65ec5e078cd737d02c2845b662a9ac6be58a6afd2d7e7c0db2949bc1a8ac72a7.jpg)

![6fb1ec4027c99e3a62e4af01e1225e25ed822fec81b93f66a88246fa8f4717f6.jpg](../iclr_results/1033_Swing-by Dynamics in Concept Learning and Compositional Generalization/images/6fb1ec4027c99e3a62e4af01e1225e25ed822fec81b93f66a88246fa8f4717f6.jpg)

![7d7bd2ad5c19b5742ef340b5f7da93688271e590b966c1252ecf9a000a2020b2.jpg](../iclr_results/1033_Swing-by Dynamics in Concept Learning and Compositional Generalization/images/7d7bd2ad5c19b5742ef340b5f7da93688271e590b966c1252ecf9a000a2020b2.jpg)

![82f8563424a3ada0bbffceee953b1c5b9a653a1e7b48c15cae17bbc9b6cc8da2.jpg](../iclr_results/1033_Swing-by Dynamics in Concept Learning and Compositional Generalization/images/82f8563424a3ada0bbffceee953b1c5b9a653a1e7b48c15cae17bbc9b6cc8da2.jpg)

![84b5e3e59d8af676996177c22202be9638e148dbccae37812ca0b6efb6a85dc3.jpg](../iclr_results/1033_Swing-by Dynamics in Concept Learning and Compositional Generalization/images/84b5e3e59d8af676996177c22202be9638e148dbccae37812ca0b6efb6a85dc3.jpg)

![8c24324c9e62020d00ceca6a36e69258b943ac869a44bbb202d7a8d5b0d812f5.jpg](../iclr_results/1033_Swing-by Dynamics in Concept Learning and Compositional Generalization/images/8c24324c9e62020d00ceca6a36e69258b943ac869a44bbb202d7a8d5b0d812f5.jpg)

![9e7b822847fd5c4ec059e238e137d1d601393b2a633b7227ae980159f594365b.jpg](../iclr_results/1033_Swing-by Dynamics in Concept Learning and Compositional Generalization/images/9e7b822847fd5c4ec059e238e137d1d601393b2a633b7227ae980159f594365b.jpg)

![b83b158fa847ecc947d74e458bee3bd67ad6487e4a80e8d590d55d31a247c752.jpg](../iclr_results/1033_Swing-by Dynamics in Concept Learning and Compositional Generalization/images/b83b158fa847ecc947d74e458bee3bd67ad6487e4a80e8d590d55d31a247c752.jpg)

![baaed56db690765accd4350cbec503339c8666b6ee18509809d9a3d1132fd6ba.jpg](../iclr_results/1033_Swing-by Dynamics in Concept Learning and Compositional Generalization/images/baaed56db690765accd4350cbec503339c8666b6ee18509809d9a3d1132fd6ba.jpg)

![e094b5502bc3a0e295faac2665b440afa8bab33450e0815ba9e2b2948543f4c7.jpg](../iclr_results/1033_Swing-by Dynamics in Concept Learning and Compositional Generalization/images/e094b5502bc3a0e295faac2665b440afa8bab33450e0815ba9e2b2948543f4c7.jpg)

![e3a4b89fcafe51e83a8da63a56a42c5301e2b59b55f5cf63ea420c7f226ec0fa.jpg](../iclr_results/1033_Swing-by Dynamics in Concept Learning and Compositional Generalization/images/e3a4b89fcafe51e83a8da63a56a42c5301e2b59b55f5cf63ea420c7f226ec0fa.jpg)

![eb42e6d6ac636dac803c6c464f89bd04f70364b68022f1a07e21bde5e45f9b70.jpg](../iclr_results/1033_Swing-by Dynamics in Concept Learning and Compositional Generalization/images/eb42e6d6ac636dac803c6c464f89bd04f70364b68022f1a07e21bde5e45f9b70.jpg)

![ef82aaadf4b257f617dea316f83b519b85d4e709f544cef8c1e8d93b2d5172b8.jpg](../iclr_results/1033_Swing-by Dynamics in Concept Learning and Compositional Generalization/images/ef82aaadf4b257f617dea316f83b519b85d4e709f544cef8c1e8d93b2d5172b8.jpg)

![f3cbb7462b03004ff4efedd624a2ca00715c348dfcfc86ade146e955ac1c1cf0.jpg](../iclr_results/1033_Swing-by Dynamics in Concept Learning and Compositional Generalization/images/f3cbb7462b03004ff4efedd624a2ca00715c348dfcfc86ade146e955ac1c1cf0.jpg)

## An Evolved Universal Transformer Memory


### Images

![00d6a91d87716859dfca03bb4189e4451f2f5822ce7b5f6847291cf3bbb7e588.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/images/00d6a91d87716859dfca03bb4189e4451f2f5822ce7b5f6847291cf3bbb7e588.jpg)

![090fbde9a0095af86b9d66bb67ac9c7786c8a20ba01d6d56eb6f21f98be0cf59.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/images/090fbde9a0095af86b9d66bb67ac9c7786c8a20ba01d6d56eb6f21f98be0cf59.jpg)

![10ae0b25700e1a03def6eaa154cde669eceb46d65371f26c033643ea0184fcbe.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/images/10ae0b25700e1a03def6eaa154cde669eceb46d65371f26c033643ea0184fcbe.jpg)

![50bb8ae0309240d013109f18bbc9060afa781a9873405fa03da182ba16b86964.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/images/50bb8ae0309240d013109f18bbc9060afa781a9873405fa03da182ba16b86964.jpg)

![58ae4c4c97118da84030827f1ef47e16f2a0475792d61f61b9ae6263b1a0a177.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/images/58ae4c4c97118da84030827f1ef47e16f2a0475792d61f61b9ae6263b1a0a177.jpg)

![632f53502e1607d5d914ae343ba25fabeb1fb917bd6875a279eb214c1e4c9b1b.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/images/632f53502e1607d5d914ae343ba25fabeb1fb917bd6875a279eb214c1e4c9b1b.jpg)

![7000a20d3c2c077af57f681b4d42d6b83b7c420b7d78f2a2977e6134f8024470.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/images/7000a20d3c2c077af57f681b4d42d6b83b7c420b7d78f2a2977e6134f8024470.jpg)

![7a3601bd7453bfbdb06a3b6e34aa2c0ee0bcee08f3953784429893d3a6398491.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/images/7a3601bd7453bfbdb06a3b6e34aa2c0ee0bcee08f3953784429893d3a6398491.jpg)

![973ae46ad810ea335ce6e316826637a183570e8da13283214e28d2daeb866157.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/images/973ae46ad810ea335ce6e316826637a183570e8da13283214e28d2daeb866157.jpg)

![a088f66925c7281683eeb861cb3b4986a7a1b01c6bf0166f9f46b534b497b8de.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/images/a088f66925c7281683eeb861cb3b4986a7a1b01c6bf0166f9f46b534b497b8de.jpg)

![b01b348c81b9f87ec92b83f62deef6e34f29430b5725481ca07d18ce1e67fa07.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/images/b01b348c81b9f87ec92b83f62deef6e34f29430b5725481ca07d18ce1e67fa07.jpg)

![b14b0eedd75bf5e64c68ef3eaf6839ce8a877e826c8671bcfe547e3b34770aa1.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/images/b14b0eedd75bf5e64c68ef3eaf6839ce8a877e826c8671bcfe547e3b34770aa1.jpg)

![b856d7e66749b62c7ad46ff5e081bf096d8c14e886f5a2d9c38e2ebc5e9db651.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/images/b856d7e66749b62c7ad46ff5e081bf096d8c14e886f5a2d9c38e2ebc5e9db651.jpg)

![b9c2fd917059c4ba9dff8233719862624757f67af9066893e1e9025d527448ca.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/images/b9c2fd917059c4ba9dff8233719862624757f67af9066893e1e9025d527448ca.jpg)

![e4dc4555784aa258d2f105dcba0584fb93198d0f8fcff9386e4d2ecda53f9655.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/images/e4dc4555784aa258d2f105dcba0584fb93198d0f8fcff9386e4d2ecda53f9655.jpg)

### Tables

![190787a3ed7d7dfedaff526ea43d0a6a18557127ee2d0755ac741e9932c4e853.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/190787a3ed7d7dfedaff526ea43d0a6a18557127ee2d0755ac741e9932c4e853.jpg)

![1ea1d1df2c46506334efe7f64fee4a408efda392e1cbd5bd5dd07f759d1e0b8d.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/1ea1d1df2c46506334efe7f64fee4a408efda392e1cbd5bd5dd07f759d1e0b8d.jpg)

![1f40ada6cef87eb849b35bbfd210054b671aebd0b16c78d59f3376a272262e7a.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/1f40ada6cef87eb849b35bbfd210054b671aebd0b16c78d59f3376a272262e7a.jpg)

![2b0fce88ecf5f22887d548367edc7dfe47946c84e681f33bda7d120b98aa6b01.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/2b0fce88ecf5f22887d548367edc7dfe47946c84e681f33bda7d120b98aa6b01.jpg)

![2bd118a79441b7f6453568cb589035b1a2c2c1c1d57a13c46b31f3bacfa20797.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/2bd118a79441b7f6453568cb589035b1a2c2c1c1d57a13c46b31f3bacfa20797.jpg)

![432aa28849c5c3f8dc176b9e476d76fdb292da83973911e7e944a1df915ca8c7.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/432aa28849c5c3f8dc176b9e476d76fdb292da83973911e7e944a1df915ca8c7.jpg)

![486114ee9f29c9ab6c78f66ad0079b38fa69610d45c915a716521cddfd1108ce.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/486114ee9f29c9ab6c78f66ad0079b38fa69610d45c915a716521cddfd1108ce.jpg)

![56538da030679f067dd95d2e80d5600631686d1f0e82991eac689456545e52e6.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/56538da030679f067dd95d2e80d5600631686d1f0e82991eac689456545e52e6.jpg)

![5f82a7da07179291648923985849cad837a41a9739fb942169bda9fd550d37d9.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/5f82a7da07179291648923985849cad837a41a9739fb942169bda9fd550d37d9.jpg)

![69cf31094d12d09335313d233753cea3b8fc1307c467d36478b7471d2e0ca57f.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/69cf31094d12d09335313d233753cea3b8fc1307c467d36478b7471d2e0ca57f.jpg)

![6f82e1484af498c0f12d192ec4b7b356c2303847799bdb48f3b15946a74ea8bc.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/6f82e1484af498c0f12d192ec4b7b356c2303847799bdb48f3b15946a74ea8bc.jpg)

![76a4c1f7037945b7d767b8d152adb104125fd5bfdb514486f9202da893813afe.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/76a4c1f7037945b7d767b8d152adb104125fd5bfdb514486f9202da893813afe.jpg)

![7b48fe9e7515a52ce97cc5fb086a6ca80d030766924b31238323b7e5e270782a.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/7b48fe9e7515a52ce97cc5fb086a6ca80d030766924b31238323b7e5e270782a.jpg)

![84480f4102e64675c38eb2493c016e1c467dc93f9a5776b770537735f78fe1c8.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/84480f4102e64675c38eb2493c016e1c467dc93f9a5776b770537735f78fe1c8.jpg)

![85db5aed4e22b8d21917ab8cb51be41e74886c5408f5443c1c4aa2dbdfefc186.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/85db5aed4e22b8d21917ab8cb51be41e74886c5408f5443c1c4aa2dbdfefc186.jpg)

![89965459872d1ae6097e7f77a73c37b7bc1c67d469f43dca7dff2e8dc113dc1b.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/89965459872d1ae6097e7f77a73c37b7bc1c67d469f43dca7dff2e8dc113dc1b.jpg)

![983c67611f500bcb5e7dfdbce9c3e158f2d19150f1da43ee3b70fd8be8dc6e39.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/983c67611f500bcb5e7dfdbce9c3e158f2d19150f1da43ee3b70fd8be8dc6e39.jpg)

![9ad556b5fb7b52237d05447929f822ccc33c0aa35beb48b5b648cf33dfda1318.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/9ad556b5fb7b52237d05447929f822ccc33c0aa35beb48b5b648cf33dfda1318.jpg)

![a6a6fb7a07cbe3246ef133c15ba66eaecb65d78f7a2169f2ad25cb680a3d2b08.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/a6a6fb7a07cbe3246ef133c15ba66eaecb65d78f7a2169f2ad25cb680a3d2b08.jpg)

![c1d261fecef10f9b62226f20ced7d0310dc46952ac46232f52069463c1fa6a17.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/c1d261fecef10f9b62226f20ced7d0310dc46952ac46232f52069463c1fa6a17.jpg)

![c8bd591675bddacce9fc7606c740f72e6e32186e6f67e42e07fa032e9d3eb041.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/c8bd591675bddacce9fc7606c740f72e6e32186e6f67e42e07fa032e9d3eb041.jpg)

![c92b6221d4fe114305b34c96d7817c31a266c32a0010142000ca873f26360549.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/c92b6221d4fe114305b34c96d7817c31a266c32a0010142000ca873f26360549.jpg)

![d2361cf0d3c51db3d2c9483b78a3621b6979b126da85f656d70a622276db2935.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/d2361cf0d3c51db3d2c9483b78a3621b6979b126da85f656d70a622276db2935.jpg)

![e95c4115694fb629f37765689c2e38a6a1a1b3ba8db7b7ffa54b2789d868478d.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/e95c4115694fb629f37765689c2e38a6a1a1b3ba8db7b7ffa54b2789d868478d.jpg)

![efb98bef7051cf6469b6598ce829e5d6b231f1f4df2b8868e3311ba6f32a07a3.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/efb98bef7051cf6469b6598ce829e5d6b231f1f4df2b8868e3311ba6f32a07a3.jpg)

![fb38b11ebbfa9316482159c0dc2125e22d9034208b24a02efa7cdb86a95414a4.jpg](../iclr_results/1034_An Evolved Universal Transformer Memory/tables/fb38b11ebbfa9316482159c0dc2125e22d9034208b24a02efa7cdb86a95414a4.jpg)

## On Discriminative Probabilistic Modeling for Self-Supervised Representation Learning


### Images

![0fda3586b592e5256108664afc5911c9eee177057252a0d8a0b056890586b9ea.jpg](../iclr_results/1035_On Discriminative Probabilistic Modeling for Self-Supervised Representation Learning/images/0fda3586b592e5256108664afc5911c9eee177057252a0d8a0b056890586b9ea.jpg)

![323c750be4cab54d5ee5425dc5e12e8441578d587a64b2468e232fa071b5bfea.jpg](../iclr_results/1035_On Discriminative Probabilistic Modeling for Self-Supervised Representation Learning/images/323c750be4cab54d5ee5425dc5e12e8441578d587a64b2468e232fa071b5bfea.jpg)

![721a19294e17de6853ea475b5c1e2ea0dd746b67127f88d8da2fda360c937815.jpg](../iclr_results/1035_On Discriminative Probabilistic Modeling for Self-Supervised Representation Learning/images/721a19294e17de6853ea475b5c1e2ea0dd746b67127f88d8da2fda360c937815.jpg)

![ea6e0104c0f15f64da5ad54780968c0677dbd35f9a011896f0c0f4ec8d84b8c4.jpg](../iclr_results/1035_On Discriminative Probabilistic Modeling for Self-Supervised Representation Learning/images/ea6e0104c0f15f64da5ad54780968c0677dbd35f9a011896f0c0f4ec8d84b8c4.jpg)

![f77268af0bed0c990f8ec2362e3c7b5bb2a494767a852c9d8c0242f378fceb5b.jpg](../iclr_results/1035_On Discriminative Probabilistic Modeling for Self-Supervised Representation Learning/images/f77268af0bed0c990f8ec2362e3c7b5bb2a494767a852c9d8c0242f378fceb5b.jpg)

![f8732bc032c9b10288b0a4a4694ac09c714f33fd92f47ec6af3ff7e1e1359f13.jpg](../iclr_results/1035_On Discriminative Probabilistic Modeling for Self-Supervised Representation Learning/images/f8732bc032c9b10288b0a4a4694ac09c714f33fd92f47ec6af3ff7e1e1359f13.jpg)

![f92a7bf883dbfc3c32543e6ae52a1fbcec8a6787cfcc3c1bf533d8a377f85616.jpg](../iclr_results/1035_On Discriminative Probabilistic Modeling for Self-Supervised Representation Learning/images/f92a7bf883dbfc3c32543e6ae52a1fbcec8a6787cfcc3c1bf533d8a377f85616.jpg)

![fad1cac90f480d9aeb5bb04fbe5036ddee5517c2db3bfd46f3d979b20f6ad0e8.jpg](../iclr_results/1035_On Discriminative Probabilistic Modeling for Self-Supervised Representation Learning/images/fad1cac90f480d9aeb5bb04fbe5036ddee5517c2db3bfd46f3d979b20f6ad0e8.jpg)

### Tables

![7519a8b0f816c70e745328b5aaff0206c2f7b95ca0538d0335f8ada352cb6887.jpg](../iclr_results/1035_On Discriminative Probabilistic Modeling for Self-Supervised Representation Learning/tables/7519a8b0f816c70e745328b5aaff0206c2f7b95ca0538d0335f8ada352cb6887.jpg)

![77c44264d1f2e3eae8233398b0cd223131a346199d38300471eac3728f63158e.jpg](../iclr_results/1035_On Discriminative Probabilistic Modeling for Self-Supervised Representation Learning/tables/77c44264d1f2e3eae8233398b0cd223131a346199d38300471eac3728f63158e.jpg)

![a6ff5eb5dcc5c063d2eab8f60b3725a249b0b34728270e53b7e7deb57b6cf5a9.jpg](../iclr_results/1035_On Discriminative Probabilistic Modeling for Self-Supervised Representation Learning/tables/a6ff5eb5dcc5c063d2eab8f60b3725a249b0b34728270e53b7e7deb57b6cf5a9.jpg)

![bcec24c42cb4258aed02b38bcdd01897deb1a71d418c44cb530e1522ee83618f.jpg](../iclr_results/1035_On Discriminative Probabilistic Modeling for Self-Supervised Representation Learning/tables/bcec24c42cb4258aed02b38bcdd01897deb1a71d418c44cb530e1522ee83618f.jpg)

## VCR: A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text


### Images

![1bf897743a6bdfe1f5e866212151d4d961adc876b3448a315dc2f69cd4a8615a.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/images/1bf897743a6bdfe1f5e866212151d4d961adc876b3448a315dc2f69cd4a8615a.jpg)

![2d8c775b5b1f934910584f8359d0a9fa9152b1619dc66d5303adaedeecfd1f92.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/images/2d8c775b5b1f934910584f8359d0a9fa9152b1619dc66d5303adaedeecfd1f92.jpg)

![4472b56808ac7413b06e5ceae45fd419db97c144389b1b3cfe783a5cad4c6f7c.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/images/4472b56808ac7413b06e5ceae45fd419db97c144389b1b3cfe783a5cad4c6f7c.jpg)

![522d9cfb5d766e9f481e59f5f4637a3867b7272d0e4290aa80ec1fd745ef9d85.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/images/522d9cfb5d766e9f481e59f5f4637a3867b7272d0e4290aa80ec1fd745ef9d85.jpg)

![5275089a5d8c4db475f316965ba9f52458b90c111c5261c3d4e200d87408e518.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/images/5275089a5d8c4db475f316965ba9f52458b90c111c5261c3d4e200d87408e518.jpg)

![53b61a29301f83f63f44d45cb010290e59e63ac0300df78e2f6d666335a2ce36.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/images/53b61a29301f83f63f44d45cb010290e59e63ac0300df78e2f6d666335a2ce36.jpg)

![543a5c3b115414901c35be5f70aef28a6c175cfd23ce15e2569d9d9e574e48f4.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/images/543a5c3b115414901c35be5f70aef28a6c175cfd23ce15e2569d9d9e574e48f4.jpg)

![572f20fd25d5ed2e183de5fe26a7a13a41122d06eeab1ebfc61c8830d6c06fdc.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/images/572f20fd25d5ed2e183de5fe26a7a13a41122d06eeab1ebfc61c8830d6c06fdc.jpg)

![64173481fe541f105775b171f21da6e82cd8042d5fead486aefccdc315b49b9f.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/images/64173481fe541f105775b171f21da6e82cd8042d5fead486aefccdc315b49b9f.jpg)

![9475c2ccd47843901f73a113188870833b7f2beb4735054a1344e27504bee3b3.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/images/9475c2ccd47843901f73a113188870833b7f2beb4735054a1344e27504bee3b3.jpg)

![b544e4dd1d54dd27a39c74b8efd5ce9f3b79be8e1988c93386c9fcf978ed02dd.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/images/b544e4dd1d54dd27a39c74b8efd5ce9f3b79be8e1988c93386c9fcf978ed02dd.jpg)

![bd38a9185ca9e4a0d903dda558e875bd32912f630603e4abf7ba14c29a98d264.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/images/bd38a9185ca9e4a0d903dda558e875bd32912f630603e4abf7ba14c29a98d264.jpg)

![da6d51fa1a41a857acdf1fd8fe107edd998fc9ddc2bbe297b669fbda9c916565.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/images/da6d51fa1a41a857acdf1fd8fe107edd998fc9ddc2bbe297b669fbda9c916565.jpg)

![e4f6ee9c2dc27d89b38ebc128943681e107e1d037c1e76c0c494baa912e4867a.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/images/e4f6ee9c2dc27d89b38ebc128943681e107e1d037c1e76c0c494baa912e4867a.jpg)

![e6dce2b5c6629a3ddd998eb8b44f99f8b3207337701865c84f4d6b4a524a061f.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/images/e6dce2b5c6629a3ddd998eb8b44f99f8b3207337701865c84f4d6b4a524a061f.jpg)

![f41131c602651b28472550c6c3befe17d37ab7ea7fe3e2a608ea78412d72f311.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/images/f41131c602651b28472550c6c3befe17d37ab7ea7fe3e2a608ea78412d72f311.jpg)

![fcd122adb0bb8ed6375ee9e1df00c51950c5b4ee45f782c77396b8441bc347c0.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/images/fcd122adb0bb8ed6375ee9e1df00c51950c5b4ee45f782c77396b8441bc347c0.jpg)

### Tables

![6f69cf0dd20022507c4e439f46561de6270b4f7dd313522888ba80c2620183a3.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/tables/6f69cf0dd20022507c4e439f46561de6270b4f7dd313522888ba80c2620183a3.jpg)

![7111ba7f6045b6197bbc35727c216539920dd9c632ed05e49bf26bdeeeb0043c.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/tables/7111ba7f6045b6197bbc35727c216539920dd9c632ed05e49bf26bdeeeb0043c.jpg)

![915cd2ef3e0309ac9ce73fa6803ca80717de7d5ba11a464bf95268deabb45d23.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/tables/915cd2ef3e0309ac9ce73fa6803ca80717de7d5ba11a464bf95268deabb45d23.jpg)

![9aab7b30f90d5d9e0520a3920df2987ee7fe2bf82334170a85b3b9deb5346d88.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/tables/9aab7b30f90d5d9e0520a3920df2987ee7fe2bf82334170a85b3b9deb5346d88.jpg)

![a072fcec8b557ef4b58b11a05a857dfd54b896500bb8a3f6f14149a9f550f566.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/tables/a072fcec8b557ef4b58b11a05a857dfd54b896500bb8a3f6f14149a9f550f566.jpg)

![b3515b0fc2178849aa19b681b9a99834f821e4ae3697d2b493842b6c6237f2d3.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/tables/b3515b0fc2178849aa19b681b9a99834f821e4ae3697d2b493842b6c6237f2d3.jpg)

![cb3f4c0bd789dff383194e6fe4beb357219f34aafe91516b1da3d567c7197350.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/tables/cb3f4c0bd789dff383194e6fe4beb357219f34aafe91516b1da3d567c7197350.jpg)

![ccf87b1179427697d658a8b5c76e5056757370600795f744aec6010b0aa27818.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/tables/ccf87b1179427697d658a8b5c76e5056757370600795f744aec6010b0aa27818.jpg)

![cd34597ab7c836c2129ba943bacf517e2cecc059c496746a6a0b3bccf06f88ff.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/tables/cd34597ab7c836c2129ba943bacf517e2cecc059c496746a6a0b3bccf06f88ff.jpg)

![e8e18f4f3d4c3d294b697918c7ae2c3b9f8f0669e38eaf3b7c17a01a28eb507c.jpg](../iclr_results/1036_VCR_ A Task for Pixel-Level Complex Reasoning in Vision Language Models via Restoring Occluded Text/tables/e8e18f4f3d4c3d294b697918c7ae2c3b9f8f0669e38eaf3b7c17a01a28eb507c.jpg)

## TIGER: Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation


### Images

![454abc857ae9b5d21b97d719a64853704f1ed8e1eee45e6ecc737f1449bd274b.jpg](../iclr_results/1037_TIGER_ Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation/images/454abc857ae9b5d21b97d719a64853704f1ed8e1eee45e6ecc737f1449bd274b.jpg)

![4bd5c2c128ab4c9eeba45f219d22b89cfbb656eb5dec4ef305fed8455e56ba49.jpg](../iclr_results/1037_TIGER_ Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation/images/4bd5c2c128ab4c9eeba45f219d22b89cfbb656eb5dec4ef305fed8455e56ba49.jpg)

![5b3666e9dda9887b66e5bb2a6bad12d0a3826d5be6033b6217397015710fe554.jpg](../iclr_results/1037_TIGER_ Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation/images/5b3666e9dda9887b66e5bb2a6bad12d0a3826d5be6033b6217397015710fe554.jpg)

![9f8b3b804f1df71576ac9b744ba5000f2ef2fbf6691b38794985af7f19c27a2d.jpg](../iclr_results/1037_TIGER_ Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation/images/9f8b3b804f1df71576ac9b744ba5000f2ef2fbf6691b38794985af7f19c27a2d.jpg)

![e45ca16dea94eeaa92ecfc86e948ef136480a6b58d44e10130d79ef2f5d0ace4.jpg](../iclr_results/1037_TIGER_ Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation/images/e45ca16dea94eeaa92ecfc86e948ef136480a6b58d44e10130d79ef2f5d0ace4.jpg)

### Tables

![014f1757e4ff6cf903ff5e1a070f1287633e458b08e5a3e003eb259876752891.jpg](../iclr_results/1037_TIGER_ Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation/tables/014f1757e4ff6cf903ff5e1a070f1287633e458b08e5a3e003eb259876752891.jpg)

![0fe0bbd6a1bfc4eac212aa02802ca0eee955264d7f0763da94cae020535c96b5.jpg](../iclr_results/1037_TIGER_ Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation/tables/0fe0bbd6a1bfc4eac212aa02802ca0eee955264d7f0763da94cae020535c96b5.jpg)

![15ff302ca98d72fef832207c6f554540e3d1a46e7d02c5d11b778dfe6222c1f0.jpg](../iclr_results/1037_TIGER_ Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation/tables/15ff302ca98d72fef832207c6f554540e3d1a46e7d02c5d11b778dfe6222c1f0.jpg)

![22b295c57cf02600699f072d7605eb30a256f2fb287872e09c8f918f9b6a70cc.jpg](../iclr_results/1037_TIGER_ Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation/tables/22b295c57cf02600699f072d7605eb30a256f2fb287872e09c8f918f9b6a70cc.jpg)

![5032f8b758df37bfee28171c5a9f9621a0ac44b09968f3796cbe53db579f4516.jpg](../iclr_results/1037_TIGER_ Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation/tables/5032f8b758df37bfee28171c5a9f9621a0ac44b09968f3796cbe53db579f4516.jpg)

![7854611e0ed648b4b6711bdebc22236a0350d0c9221454544f9a24558b4b7386.jpg](../iclr_results/1037_TIGER_ Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation/tables/7854611e0ed648b4b6711bdebc22236a0350d0c9221454544f9a24558b4b7386.jpg)

![85b58eec66ec4edcb3c1953d313c945fb976e44454b8170293fe68c1c0ee2c97.jpg](../iclr_results/1037_TIGER_ Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation/tables/85b58eec66ec4edcb3c1953d313c945fb976e44454b8170293fe68c1c0ee2c97.jpg)

![94a8bc5d87728a99d02db4af00bdae4469c20779c522e8a3a7c864597ef7ba42.jpg](../iclr_results/1037_TIGER_ Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation/tables/94a8bc5d87728a99d02db4af00bdae4469c20779c522e8a3a7c864597ef7ba42.jpg)

![b1cf224e04d5259d4721877636b47cc3c1dccb4e9c78168408c30dc1b2fb0d1f.jpg](../iclr_results/1037_TIGER_ Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation/tables/b1cf224e04d5259d4721877636b47cc3c1dccb4e9c78168408c30dc1b2fb0d1f.jpg)

![b422e5f99f3eda6a70f054302f7ff08ddded6b340dedf44cac5bc875d38fe2d8.jpg](../iclr_results/1037_TIGER_ Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation/tables/b422e5f99f3eda6a70f054302f7ff08ddded6b340dedf44cac5bc875d38fe2d8.jpg)

![c2c104813e70d41392487046f5a0469482284b1fa12adf9f1a0c661127993ae5.jpg](../iclr_results/1037_TIGER_ Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation/tables/c2c104813e70d41392487046f5a0469482284b1fa12adf9f1a0c661127993ae5.jpg)

![fde0423b4415438fdee0149d5c897ac598e945aeea627c7924b777ab4979ed67.jpg](../iclr_results/1037_TIGER_ Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation/tables/fde0423b4415438fdee0149d5c897ac598e945aeea627c7924b777ab4979ed67.jpg)

## Minimal Impact ControlNet: Advancing Multi-ControlNet Integration


### Images

![10479156976fdd76a109e143bd473cc2955a09c61140641d82447e1212a3816e.jpg](../iclr_results/1038_Minimal Impact ControlNet_ Advancing Multi-ControlNet Integration/images/10479156976fdd76a109e143bd473cc2955a09c61140641d82447e1212a3816e.jpg)

![3c9e229c431791ce672f146c4bb55eaca4cf10b27f8de161fa6e8ed68c84d9c4.jpg](../iclr_results/1038_Minimal Impact ControlNet_ Advancing Multi-ControlNet Integration/images/3c9e229c431791ce672f146c4bb55eaca4cf10b27f8de161fa6e8ed68c84d9c4.jpg)

![7d13ec745a9a9ff11b4ea2cc1123851be9295a60052a6bad7e814f2215fec4d8.jpg](../iclr_results/1038_Minimal Impact ControlNet_ Advancing Multi-ControlNet Integration/images/7d13ec745a9a9ff11b4ea2cc1123851be9295a60052a6bad7e814f2215fec4d8.jpg)

![8aff713041dae08cd21ba65e64c4d525ea4d6f564140454512805d7d5cbdf23d.jpg](../iclr_results/1038_Minimal Impact ControlNet_ Advancing Multi-ControlNet Integration/images/8aff713041dae08cd21ba65e64c4d525ea4d6f564140454512805d7d5cbdf23d.jpg)

![9075bff2143beefd13f9c6334f01a7c4d9185a701bbc31a912de9ea551dddb94.jpg](../iclr_results/1038_Minimal Impact ControlNet_ Advancing Multi-ControlNet Integration/images/9075bff2143beefd13f9c6334f01a7c4d9185a701bbc31a912de9ea551dddb94.jpg)

![9529ab37805073d3c96e2da601fb814aa7f8193775373c355ee887dfba28e14c.jpg](../iclr_results/1038_Minimal Impact ControlNet_ Advancing Multi-ControlNet Integration/images/9529ab37805073d3c96e2da601fb814aa7f8193775373c355ee887dfba28e14c.jpg)

![b9df10c69dfd6daf8f8b540404608ed3000a1716f29cf53a318229c7bf160fc6.jpg](../iclr_results/1038_Minimal Impact ControlNet_ Advancing Multi-ControlNet Integration/images/b9df10c69dfd6daf8f8b540404608ed3000a1716f29cf53a318229c7bf160fc6.jpg)

![bb626a732a1abf91a7f1cf7f9cc4d8d8a1b407bad40d4748aeb8957703068bbe.jpg](../iclr_results/1038_Minimal Impact ControlNet_ Advancing Multi-ControlNet Integration/images/bb626a732a1abf91a7f1cf7f9cc4d8d8a1b407bad40d4748aeb8957703068bbe.jpg)

![bca452dc56bff5a6efbc5bf44be426123bb1510487edefeb1d1482f6ed997b4f.jpg](../iclr_results/1038_Minimal Impact ControlNet_ Advancing Multi-ControlNet Integration/images/bca452dc56bff5a6efbc5bf44be426123bb1510487edefeb1d1482f6ed997b4f.jpg)

![cfb3a7c156a47c40b6a06cf3a75e392622f9807f1c1a6e34ad5350d9de6e91f6.jpg](../iclr_results/1038_Minimal Impact ControlNet_ Advancing Multi-ControlNet Integration/images/cfb3a7c156a47c40b6a06cf3a75e392622f9807f1c1a6e34ad5350d9de6e91f6.jpg)

![e9dd37b73128ef29e665e9a1add4e13000a9ad98f14d5f15809ad3dbb16e324c.jpg](../iclr_results/1038_Minimal Impact ControlNet_ Advancing Multi-ControlNet Integration/images/e9dd37b73128ef29e665e9a1add4e13000a9ad98f14d5f15809ad3dbb16e324c.jpg)

![f0a6296275534af590b154280f727a156a56e904df73cb8b998f295ad7c023af.jpg](../iclr_results/1038_Minimal Impact ControlNet_ Advancing Multi-ControlNet Integration/images/f0a6296275534af590b154280f727a156a56e904df73cb8b998f295ad7c023af.jpg)

### Tables

![21cf32eb954eae1f9c7574a340d2ce38a3efc5bf95e8463f8193ee86ad4f8cb6.jpg](../iclr_results/1038_Minimal Impact ControlNet_ Advancing Multi-ControlNet Integration/tables/21cf32eb954eae1f9c7574a340d2ce38a3efc5bf95e8463f8193ee86ad4f8cb6.jpg)

![449d2eec0f7a08ac691ea8910745f2e16d3a1d0d3240141fc2650c08416b1bb5.jpg](../iclr_results/1038_Minimal Impact ControlNet_ Advancing Multi-ControlNet Integration/tables/449d2eec0f7a08ac691ea8910745f2e16d3a1d0d3240141fc2650c08416b1bb5.jpg)

![4e9f0b25afcf25d3840db14b5e10cbc8b9f09ce037a17b84aec2618118071dcb.jpg](../iclr_results/1038_Minimal Impact ControlNet_ Advancing Multi-ControlNet Integration/tables/4e9f0b25afcf25d3840db14b5e10cbc8b9f09ce037a17b84aec2618118071dcb.jpg)

![5ba12ffabb0f3811eeb07d42bfa4a65e870b93b8f288a1c88d6ceb78d738d420.jpg](../iclr_results/1038_Minimal Impact ControlNet_ Advancing Multi-ControlNet Integration/tables/5ba12ffabb0f3811eeb07d42bfa4a65e870b93b8f288a1c88d6ceb78d738d420.jpg)

![95cf6d8c3b4c0c91f70a184daa2d4aaf8b03de884e90985efbf402db439de0ea.jpg](../iclr_results/1038_Minimal Impact ControlNet_ Advancing Multi-ControlNet Integration/tables/95cf6d8c3b4c0c91f70a184daa2d4aaf8b03de884e90985efbf402db439de0ea.jpg)

![a863407825474e64545ebf8eab264648dcc430cd9ec3620a6e9e89faf1057bfe.jpg](../iclr_results/1038_Minimal Impact ControlNet_ Advancing Multi-ControlNet Integration/tables/a863407825474e64545ebf8eab264648dcc430cd9ec3620a6e9e89faf1057bfe.jpg)

![e384cf06c7689980b9f2fb27b5334d722504610bbac8bfa52fa6bae299e2d8dd.jpg](../iclr_results/1038_Minimal Impact ControlNet_ Advancing Multi-ControlNet Integration/tables/e384cf06c7689980b9f2fb27b5334d722504610bbac8bfa52fa6bae299e2d8dd.jpg)

## Beyond the convexity assumption: Realistic tabular data generation under quantifier-free real linear constraints


### Images

![16aecefb93d94bf6d97b6df6063c25bc7ac6274f16d1e8e5544fe2738cdafc84.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/images/16aecefb93d94bf6d97b6df6063c25bc7ac6274f16d1e8e5544fe2738cdafc84.jpg)

![34ab498efef9072fa82a590d64f748edd6025570e1643b969faf007d73f2fbed.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/images/34ab498efef9072fa82a590d64f748edd6025570e1643b969faf007d73f2fbed.jpg)

![6755ae408f8cf3e3059b47b3148161e81e599535d1a089ec890c95ecef84ceee.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/images/6755ae408f8cf3e3059b47b3148161e81e599535d1a089ec890c95ecef84ceee.jpg)

![6834ee7958e5a72cb17fad01ca03c3e46cc742f0b52c3a0a4cc02b4de9aafe4f.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/images/6834ee7958e5a72cb17fad01ca03c3e46cc742f0b52c3a0a4cc02b4de9aafe4f.jpg)

![6af35fa0b2b3de6cd9b2626b4900ed89937fc3da965e160a068978c3480d5711.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/images/6af35fa0b2b3de6cd9b2626b4900ed89937fc3da965e160a068978c3480d5711.jpg)

![81ea3cb6dc6f6a74ca34cf1bc48bdffd8b19001362b4cb7096c7842018ef750b.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/images/81ea3cb6dc6f6a74ca34cf1bc48bdffd8b19001362b4cb7096c7842018ef750b.jpg)

![b8454f6d294cc2f0d82dc79f4db2fd766673f8ebb07ed5a567a1c9de7e1b2abc.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/images/b8454f6d294cc2f0d82dc79f4db2fd766673f8ebb07ed5a567a1c9de7e1b2abc.jpg)

![e84d2d76c811113f3fef9d04aead46301a1d321decf08028ff49c14c7fd52e3a.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/images/e84d2d76c811113f3fef9d04aead46301a1d321decf08028ff49c14c7fd52e3a.jpg)

### Tables

![01a8bfaf1706750ade217b53f243fb0573de36b7aac0932a94b931ad0557b200.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/01a8bfaf1706750ade217b53f243fb0573de36b7aac0932a94b931ad0557b200.jpg)

![1614d529cd92a853ee7517e24a1b3ff4986eb0e9d7a5aab3e0fa40c1671c3591.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/1614d529cd92a853ee7517e24a1b3ff4986eb0e9d7a5aab3e0fa40c1671c3591.jpg)

![222de0013ce65a3128e4fe8bb1591c8b6e576abd88d805558b36a441c73b5db4.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/222de0013ce65a3128e4fe8bb1591c8b6e576abd88d805558b36a441c73b5db4.jpg)

![2516d2b0482475f71f785e09124afedbdaea780e659a56150ad1f01b5425894c.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/2516d2b0482475f71f785e09124afedbdaea780e659a56150ad1f01b5425894c.jpg)

![2f931a34498c69150af2b701b260117e17feb7bc1fc43fffbc87d76057a73d01.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/2f931a34498c69150af2b701b260117e17feb7bc1fc43fffbc87d76057a73d01.jpg)

![3a716b4d731f5aae4ac2cec441a3b1070789d04e20f9f1799f0b3373d5240af3.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/3a716b4d731f5aae4ac2cec441a3b1070789d04e20f9f1799f0b3373d5240af3.jpg)

![458fbffe1d4f7cbc36e611ee8faa6ed1ad529d8d276ef23370c0b835aa8556ba.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/458fbffe1d4f7cbc36e611ee8faa6ed1ad529d8d276ef23370c0b835aa8556ba.jpg)

![46a43bb84fe343a0aef26c9e5b348558b25af188708f3d363070d24ef1027a36.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/46a43bb84fe343a0aef26c9e5b348558b25af188708f3d363070d24ef1027a36.jpg)

![4fa7c4ba5920c0b358def51afa9b2c340a7e5b1b3c6eaac087df03ca4b6792bd.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/4fa7c4ba5920c0b358def51afa9b2c340a7e5b1b3c6eaac087df03ca4b6792bd.jpg)

![517f87fe2291a7559c6751b9bad0550886fe5efc942535312e6455f8d6e3318c.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/517f87fe2291a7559c6751b9bad0550886fe5efc942535312e6455f8d6e3318c.jpg)

![596f5562af5b5ae397bf8e59cb33b1c95d536ac16553a0bf8e8cc9fc92395373.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/596f5562af5b5ae397bf8e59cb33b1c95d536ac16553a0bf8e8cc9fc92395373.jpg)

![5fcde58b35c23f8cfec97fe9cdcf4a4da2ceef07bd9aa407e976dd3ec8ffac58.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/5fcde58b35c23f8cfec97fe9cdcf4a4da2ceef07bd9aa407e976dd3ec8ffac58.jpg)

![6a878259ffa9360e84082d9b3f0a4d2474ea0010e05e4ea3d3f1cb754c7a2d44.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/6a878259ffa9360e84082d9b3f0a4d2474ea0010e05e4ea3d3f1cb754c7a2d44.jpg)

![80a0229858f4129c87844f84e2fade15f62fc08ee130b91ad529078101c4dbee.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/80a0229858f4129c87844f84e2fade15f62fc08ee130b91ad529078101c4dbee.jpg)

![88b9346f49d501484a4ab1dea6c6167429a4249e38de6b241b1463ba7b0ccec5.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/88b9346f49d501484a4ab1dea6c6167429a4249e38de6b241b1463ba7b0ccec5.jpg)

![8d3e5d075e59c0c35e2b9fd690f81fbdc72e1a951ac5a50c73ec0190e531b5d5.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/8d3e5d075e59c0c35e2b9fd690f81fbdc72e1a951ac5a50c73ec0190e531b5d5.jpg)

![a4eed38609c9c098166e48df6f602eed7854899045b7355ad0fb374f9f01a0af.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/a4eed38609c9c098166e48df6f602eed7854899045b7355ad0fb374f9f01a0af.jpg)

![af5d1f673049462c9d9c16525dfca80bd7f11314d5de75609d40eba0cece96b8.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/af5d1f673049462c9d9c16525dfca80bd7f11314d5de75609d40eba0cece96b8.jpg)

![bcbd110d955c74a39ce79934691e617f3dd317eaa796e2b1abd2047b079145c3.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/bcbd110d955c74a39ce79934691e617f3dd317eaa796e2b1abd2047b079145c3.jpg)

![d0cc1af92154575b19d763cfc20ff0d4ce728697c88f1444c363e0690d338fc9.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/d0cc1af92154575b19d763cfc20ff0d4ce728697c88f1444c363e0690d338fc9.jpg)

![d3e627592d1463402b61e785dc64911077eae38ecfbc927990ba68ea9bd2c966.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/d3e627592d1463402b61e785dc64911077eae38ecfbc927990ba68ea9bd2c966.jpg)

![e33d9d3c807fc90c9a9fd911b05f246875a7645e0c7c00fd1f316909c68ce89a.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/e33d9d3c807fc90c9a9fd911b05f246875a7645e0c7c00fd1f316909c68ce89a.jpg)

![e6eba553b964085f56ecd8d44ae07761e836e82749e6ce7000ffacfa003fa3c1.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/e6eba553b964085f56ecd8d44ae07761e836e82749e6ce7000ffacfa003fa3c1.jpg)

![ee9dcf693c0808635be262b181034e87b4fd8f71772d4a0e1b89e57b15a3039a.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/ee9dcf693c0808635be262b181034e87b4fd8f71772d4a0e1b89e57b15a3039a.jpg)

![f13bce1740c90c13d3e97b319e794a7d5205c0545e869ab6680eff3be4b8fe56.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/f13bce1740c90c13d3e97b319e794a7d5205c0545e869ab6680eff3be4b8fe56.jpg)

![f73aa1b9f51d7ffed40c14074032b0e2ca76586fa97dae2d3d14d19191180935.jpg](../iclr_results/1039_Beyond the convexity assumption_ Realistic tabular data generation under quantifier-free real linear/tables/f73aa1b9f51d7ffed40c14074032b0e2ca76586fa97dae2d3d14d19191180935.jpg)

## Training-Free Dataset Pruning for Instance Segmentation


### Images

![1530e9c50149b9810b3c5158d3da2ab87bd3c83296ea5c8b98d4f82bc89b6f3b.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/images/1530e9c50149b9810b3c5158d3da2ab87bd3c83296ea5c8b98d4f82bc89b6f3b.jpg)

![33f7e91d915251cb9b1e31043a767223c839c250188d53030363e302d10eeeb7.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/images/33f7e91d915251cb9b1e31043a767223c839c250188d53030363e302d10eeeb7.jpg)

![38bdfa05cee4941430357b31e835f0703b5f2dcb9dcec35720da0953bc2adfc5.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/images/38bdfa05cee4941430357b31e835f0703b5f2dcb9dcec35720da0953bc2adfc5.jpg)

![61cd1040df5aeac46ef67b65f1ecba86bb526df8ddb8fb6ebafc6c003e927480.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/images/61cd1040df5aeac46ef67b65f1ecba86bb526df8ddb8fb6ebafc6c003e927480.jpg)

![8da7eb9821478f05316d199b9f483596109bf0792807c49cbd7f04b1e74a7440.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/images/8da7eb9821478f05316d199b9f483596109bf0792807c49cbd7f04b1e74a7440.jpg)

![9652a8f1aade29a3cb4d2b3b42f897811103b12337db6f39dfb60ba697f9fd60.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/images/9652a8f1aade29a3cb4d2b3b42f897811103b12337db6f39dfb60ba697f9fd60.jpg)

![968351c8e93da2eb65b63c47890a16f28731b980ca643de0053474663f6382c8.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/images/968351c8e93da2eb65b63c47890a16f28731b980ca643de0053474663f6382c8.jpg)

![9834055c832cc6d85b2e01903350a344e8f7b6cf389122204e086c875d86ed0b.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/images/9834055c832cc6d85b2e01903350a344e8f7b6cf389122204e086c875d86ed0b.jpg)

![b3c2566e39a1b9ea3edc353e1e23cd7b53312c7ff02595d2c616e8e559031c02.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/images/b3c2566e39a1b9ea3edc353e1e23cd7b53312c7ff02595d2c616e8e559031c02.jpg)

![db64a8a8f150066e2630be9970f4422c4c6fbf2f3e270885767b643a005d0043.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/images/db64a8a8f150066e2630be9970f4422c4c6fbf2f3e270885767b643a005d0043.jpg)

![e7f611ab2905a67261582514eb52f6bc41caf5a443fdbc1f7075ec81f8c2c62f.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/images/e7f611ab2905a67261582514eb52f6bc41caf5a443fdbc1f7075ec81f8c2c62f.jpg)

### Tables

![04446dc4c4e983aca096bb7fe3cf61d684815391347f3a1f841790e345bd0e8e.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/tables/04446dc4c4e983aca096bb7fe3cf61d684815391347f3a1f841790e345bd0e8e.jpg)

![0949642c2557ae0b6bf3886f432ba8b57e573f92de4b2826bdfa70f584deb118.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/tables/0949642c2557ae0b6bf3886f432ba8b57e573f92de4b2826bdfa70f584deb118.jpg)

![108388106ff585321b66ff293eb5244b2dfdcf701aa107b67dce5826a3be1792.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/tables/108388106ff585321b66ff293eb5244b2dfdcf701aa107b67dce5826a3be1792.jpg)

![1b281f4db22ee02821a397d9cb0625faeb9fb04720078acc6c71207254c518a5.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/tables/1b281f4db22ee02821a397d9cb0625faeb9fb04720078acc6c71207254c518a5.jpg)

![2d4d1399e06ec5c45b6a18dccc83cf51131eff74f9ce8a274e074d608b3692e9.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/tables/2d4d1399e06ec5c45b6a18dccc83cf51131eff74f9ce8a274e074d608b3692e9.jpg)

![50226e646f88491357b3ca4fca0c5c22e37f782119de628a2310e37ac5c8a029.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/tables/50226e646f88491357b3ca4fca0c5c22e37f782119de628a2310e37ac5c8a029.jpg)

![62bee56c0846cb64bc5fbd044504512edf2cf979fb511bf21024c696f6190757.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/tables/62bee56c0846cb64bc5fbd044504512edf2cf979fb511bf21024c696f6190757.jpg)

![81a93aeea4c10455648e71684c488fc28d2058cc18b53b7ae2e8074ed88897f4.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/tables/81a93aeea4c10455648e71684c488fc28d2058cc18b53b7ae2e8074ed88897f4.jpg)

![8e4f8733a750461e42db6f8ca1e5df5a285caf8cea44f686c5df1c2b42dd43f8.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/tables/8e4f8733a750461e42db6f8ca1e5df5a285caf8cea44f686c5df1c2b42dd43f8.jpg)

![93e0f09c52158fb68fba88f7c026b64e96338179241d6801de86b88c10af7a68.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/tables/93e0f09c52158fb68fba88f7c026b64e96338179241d6801de86b88c10af7a68.jpg)

![963407f9b09201bcb8a0a3310b35439858e99160281104dace69830a4356f900.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/tables/963407f9b09201bcb8a0a3310b35439858e99160281104dace69830a4356f900.jpg)

![96f129c1d6cc15a0af19fe4c210f082c2a9225309ddfa9d20147db472f3c0b33.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/tables/96f129c1d6cc15a0af19fe4c210f082c2a9225309ddfa9d20147db472f3c0b33.jpg)

![9cbb1ca4bb5c500323e87770ca9c7fec79edd259a0bf6214e2b958dac425f94c.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/tables/9cbb1ca4bb5c500323e87770ca9c7fec79edd259a0bf6214e2b958dac425f94c.jpg)

![ebb3a6131c677d5a098ba11d6846aec7a6e5b9b1f09caf4ecb433dfdf272096d.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/tables/ebb3a6131c677d5a098ba11d6846aec7a6e5b9b1f09caf4ecb433dfdf272096d.jpg)

![ec55e754ecce9ddb206e8e4429b96aa8c96fa8533c8bbedec624bdd0e9c3c2a3.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/tables/ec55e754ecce9ddb206e8e4429b96aa8c96fa8533c8bbedec624bdd0e9c3c2a3.jpg)

![ed518ad8fef7df21e02dc30b81d3a1c4c4f19c4b91f259aed844d54d5d777869.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/tables/ed518ad8fef7df21e02dc30b81d3a1c4c4f19c4b91f259aed844d54d5d777869.jpg)

![ed91a6a9cc847af8676d084dcf6435d92d344880ece68a7e2ffc1b77f90a0554.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/tables/ed91a6a9cc847af8676d084dcf6435d92d344880ece68a7e2ffc1b77f90a0554.jpg)

![f15f12e4d527e5d6b0ebd312629e1f1978f08f9d13bd89d2405d355aa0f53842.jpg](../iclr_results/1040_Training-Free Dataset Pruning for Instance Segmentation/tables/f15f12e4d527e5d6b0ebd312629e1f1978f08f9d13bd89d2405d355aa0f53842.jpg)

## From Probability to Counterfactuals: the Increasing Complexity of Satisfiability in Pearl's Causal Hierarchy


### Images

![3efd279add3766f7a8e84ae22ec3f9733bacec8b42c2858c9b959f51673b2c13.jpg](../iclr_results/1041_From Probability to Counterfactuals_ the Increasing Complexity of Satisfiability in Pearl's Causal H/images/3efd279add3766f7a8e84ae22ec3f9733bacec8b42c2858c9b959f51673b2c13.jpg)

![89bd439c59f0419e5d7d4ff65d9f9501c639c0f0287395a9c13a1596816caf1f.jpg](../iclr_results/1041_From Probability to Counterfactuals_ the Increasing Complexity of Satisfiability in Pearl's Causal H/images/89bd439c59f0419e5d7d4ff65d9f9501c639c0f0287395a9c13a1596816caf1f.jpg)

### Tables

![0392e9e1befb6fb4d5d6c6fe84b3a12b4f4fea59015ed7ed03d4e2a5181d7872.jpg](../iclr_results/1041_From Probability to Counterfactuals_ the Increasing Complexity of Satisfiability in Pearl's Causal H/tables/0392e9e1befb6fb4d5d6c6fe84b3a12b4f4fea59015ed7ed03d4e2a5181d7872.jpg)

![09a3895d54e3ea35f46bbf04ed65bd7bf7a99f0e87c7f7684e3f9841d2db4e42.jpg](../iclr_results/1041_From Probability to Counterfactuals_ the Increasing Complexity of Satisfiability in Pearl's Causal H/tables/09a3895d54e3ea35f46bbf04ed65bd7bf7a99f0e87c7f7684e3f9841d2db4e42.jpg)

![583761e2776e47856c87b0c97df2a7a5652888bade51fb51a94cda59564fdd7b.jpg](../iclr_results/1041_From Probability to Counterfactuals_ the Increasing Complexity of Satisfiability in Pearl's Causal H/tables/583761e2776e47856c87b0c97df2a7a5652888bade51fb51a94cda59564fdd7b.jpg)

![e8a1a8bf51840b43bd20333547da68951ba8a84d83e3e4a39a73f905bc66c419.jpg](../iclr_results/1041_From Probability to Counterfactuals_ the Increasing Complexity of Satisfiability in Pearl's Causal H/tables/e8a1a8bf51840b43bd20333547da68951ba8a84d83e3e4a39a73f905bc66c419.jpg)

## Efficient Alternating Minimization with Applications to Weighted Low Rank Approximation


### Tables

![2f57754222a026e350fb3cd0b522069824408921ea2f8dfc9d57c21a78e54e37.jpg](../iclr_results/1042_Efficient Alternating Minimization with Applications to Weighted Low Rank Approximation/tables/2f57754222a026e350fb3cd0b522069824408921ea2f8dfc9d57c21a78e54e37.jpg)

![4ace0c39452b968a6df9576a01a5237a7f9df6a36a0b85bc2ad51ff62451072f.jpg](../iclr_results/1042_Efficient Alternating Minimization with Applications to Weighted Low Rank Approximation/tables/4ace0c39452b968a6df9576a01a5237a7f9df6a36a0b85bc2ad51ff62451072f.jpg)

![538a673adecfe035f84f66a0dace80cb39a13cc4ab1e68a3a02771eb0f9e56d4.jpg](../iclr_results/1042_Efficient Alternating Minimization with Applications to Weighted Low Rank Approximation/tables/538a673adecfe035f84f66a0dace80cb39a13cc4ab1e68a3a02771eb0f9e56d4.jpg)

![5e1c0b9ed4eb23f0ccbf3e81ae67c8343dae220a2e384cc91822f865aa38d200.jpg](../iclr_results/1042_Efficient Alternating Minimization with Applications to Weighted Low Rank Approximation/tables/5e1c0b9ed4eb23f0ccbf3e81ae67c8343dae220a2e384cc91822f865aa38d200.jpg)

![66356b3f10f4b5f7c3bc6dcda60d997394f2ce0c357464a7c74960319d25a432.jpg](../iclr_results/1042_Efficient Alternating Minimization with Applications to Weighted Low Rank Approximation/tables/66356b3f10f4b5f7c3bc6dcda60d997394f2ce0c357464a7c74960319d25a432.jpg)

![6d2f8a88c39d8f922a57ae3326a883ae773de3735d078953ad966b00fcd5f4d3.jpg](../iclr_results/1042_Efficient Alternating Minimization with Applications to Weighted Low Rank Approximation/tables/6d2f8a88c39d8f922a57ae3326a883ae773de3735d078953ad966b00fcd5f4d3.jpg)

![7493f38e9637106b3b5e05756378149cbc7b4906ed5c7d14fa1ded9523275490.jpg](../iclr_results/1042_Efficient Alternating Minimization with Applications to Weighted Low Rank Approximation/tables/7493f38e9637106b3b5e05756378149cbc7b4906ed5c7d14fa1ded9523275490.jpg)

![84b97c2e09e1f8f8bfb4459df5914f628d31df022663ea6e89c910fdb63eae38.jpg](../iclr_results/1042_Efficient Alternating Minimization with Applications to Weighted Low Rank Approximation/tables/84b97c2e09e1f8f8bfb4459df5914f628d31df022663ea6e89c910fdb63eae38.jpg)

![f9c5854175a1e331a4c2160a7b4692d71435ee5a78aa62409231b14159693244.jpg](../iclr_results/1042_Efficient Alternating Minimization with Applications to Weighted Low Rank Approximation/tables/f9c5854175a1e331a4c2160a7b4692d71435ee5a78aa62409231b14159693244.jpg)

![ffeacc901652948833835473ce62894377de6017d1b9318d13f92f991d3c3607.jpg](../iclr_results/1042_Efficient Alternating Minimization with Applications to Weighted Low Rank Approximation/tables/ffeacc901652948833835473ce62894377de6017d1b9318d13f92f991d3c3607.jpg)

## Robust Transfer of Safety-Constrained Reinforcement Learning Agents


### Images

![01d6ec9b2f4eb46db9c0947ba4f23ef09e48ee2c4e82d39efb5b1d0dbf1bcf8e.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/01d6ec9b2f4eb46db9c0947ba4f23ef09e48ee2c4e82d39efb5b1d0dbf1bcf8e.jpg)

![029f60398849c445257349121c84caccc69f95ecd598fb593bf5f03474f87bdd.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/029f60398849c445257349121c84caccc69f95ecd598fb593bf5f03474f87bdd.jpg)

![02ec3676d4b4122a5eb05ad5f5ac2d046ff26588dc33de0efe3504dbc125f6ca.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/02ec3676d4b4122a5eb05ad5f5ac2d046ff26588dc33de0efe3504dbc125f6ca.jpg)

![040d5e0c27314633e7b273c174dc5e0edf493de52452c583c05181b810925ab4.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/040d5e0c27314633e7b273c174dc5e0edf493de52452c583c05181b810925ab4.jpg)

![05384b78ace13f8aeac85939736289f9425115120334bda058ff627bf1b4b400.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/05384b78ace13f8aeac85939736289f9425115120334bda058ff627bf1b4b400.jpg)

![0723f580f2ec8fd6bcaf6d09a00eec3ad2d353a7a1503dddc3156b65d6067b4e.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/0723f580f2ec8fd6bcaf6d09a00eec3ad2d353a7a1503dddc3156b65d6067b4e.jpg)

![0835814c8033f510a1263264e7a381ed7af9d82d0c223140a0758374196407c4.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/0835814c8033f510a1263264e7a381ed7af9d82d0c223140a0758374196407c4.jpg)

![0b732df5700ad760cd370ac6965d08eeddb2483eca5860613e78a2b30dac8539.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/0b732df5700ad760cd370ac6965d08eeddb2483eca5860613e78a2b30dac8539.jpg)

![0cc62dc4dd70bb389b2d9fac3f1b917c52a78f2daa5623e40406e08b6e82596a.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/0cc62dc4dd70bb389b2d9fac3f1b917c52a78f2daa5623e40406e08b6e82596a.jpg)

![0d5e0b66f5e45a828f6bd0915c011d31317175da541bed6d79857dc9ef7d33b6.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/0d5e0b66f5e45a828f6bd0915c011d31317175da541bed6d79857dc9ef7d33b6.jpg)

![0e4e965911a0b99288b28566e2bf2466cda64f2d859229cb5ada3ee9e9ebfe83.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/0e4e965911a0b99288b28566e2bf2466cda64f2d859229cb5ada3ee9e9ebfe83.jpg)

![0ee21bd848f94ddfe20e6543749d7565d59729bb8f8a0b5998de548040835b8e.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/0ee21bd848f94ddfe20e6543749d7565d59729bb8f8a0b5998de548040835b8e.jpg)

![13bb785bd0938e6f97aa681b0409a3f3be43254a3105119fd350764c39755e1a.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/13bb785bd0938e6f97aa681b0409a3f3be43254a3105119fd350764c39755e1a.jpg)

![17fd994fbb65e682fc7d8ee581a11ac6d59649325dbd4c82eb294a757e418dfd.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/17fd994fbb65e682fc7d8ee581a11ac6d59649325dbd4c82eb294a757e418dfd.jpg)

![1b59742f49d9bc171f69e5f57d63bc85875487c7685427fcddff86443fddebc4.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/1b59742f49d9bc171f69e5f57d63bc85875487c7685427fcddff86443fddebc4.jpg)

![22d86b1c1a16662b349c132854610dc00517a1614039a41d600f86637ebdb42d.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/22d86b1c1a16662b349c132854610dc00517a1614039a41d600f86637ebdb42d.jpg)

![22e5fd655d992017ea56a9434486a898ed7a4aa850715d94d01fdaa436a61e63.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/22e5fd655d992017ea56a9434486a898ed7a4aa850715d94d01fdaa436a61e63.jpg)

![230fbcbfc8eb44fa98b960276d4c394003337b186ae81aeb3577fc88aa0be16b.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/230fbcbfc8eb44fa98b960276d4c394003337b186ae81aeb3577fc88aa0be16b.jpg)

![230ffc7e62d4715cebb6ca017cd8a5540d515a8a877b06b539df42121b57075a.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/230ffc7e62d4715cebb6ca017cd8a5540d515a8a877b06b539df42121b57075a.jpg)

![23bc068e5b005ff7b37c08506665808fcc897aef8e2c7829bf513ad61f45f946.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/23bc068e5b005ff7b37c08506665808fcc897aef8e2c7829bf513ad61f45f946.jpg)

![257d957e24caf4f6cd5f053e280603204432037f553aa4a88fe2aa88951cc9d6.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/257d957e24caf4f6cd5f053e280603204432037f553aa4a88fe2aa88951cc9d6.jpg)

![270ce64455d4cfac0fe08f24575c03adf0136fd381239f56ad292a5184dae906.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/270ce64455d4cfac0fe08f24575c03adf0136fd381239f56ad292a5184dae906.jpg)

![288a8f884464be110da4d3bfdb07e709c86de63578a458b75bc4fdeda0f88092.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/288a8f884464be110da4d3bfdb07e709c86de63578a458b75bc4fdeda0f88092.jpg)

![2ca776dccc6dfdfd39f29f3c5bf881745a0903a48bc795b48a809ecd41e67d2a.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/2ca776dccc6dfdfd39f29f3c5bf881745a0903a48bc795b48a809ecd41e67d2a.jpg)

![312b744a752ce8e49ad449826feda491b625573e5df84017c92dacd43b434860.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/312b744a752ce8e49ad449826feda491b625573e5df84017c92dacd43b434860.jpg)

![353042a18417b9c51915a0c4107d29165c89dd1d5cecc3fb20dd67ebf333aa67.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/353042a18417b9c51915a0c4107d29165c89dd1d5cecc3fb20dd67ebf333aa67.jpg)

![37478e7180845c298425648fc0673719f5c448533a42986a1763fa9888586f16.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/37478e7180845c298425648fc0673719f5c448533a42986a1763fa9888586f16.jpg)

![38260a049c19260eb825af87157224abaf0acfdd9aa3019c5993399f86f39807.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/38260a049c19260eb825af87157224abaf0acfdd9aa3019c5993399f86f39807.jpg)

![3ab3e50d22679eda7dcceb5d05a9499b7c16811bc01a81bc250780014fed025d.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/3ab3e50d22679eda7dcceb5d05a9499b7c16811bc01a81bc250780014fed025d.jpg)

![3c56875136d76dfde2f1ac1ad10426175cce7f4eb38e7dac961fa8e25a8b5305.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/3c56875136d76dfde2f1ac1ad10426175cce7f4eb38e7dac961fa8e25a8b5305.jpg)

![3e411f127257e6d2527905b70a3a26314cea734bb038e486d49010bcd1311487.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/3e411f127257e6d2527905b70a3a26314cea734bb038e486d49010bcd1311487.jpg)

![473a67a4460fd51fef30a1a7db79204974209628061487609d422ad7a65fca82.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/473a67a4460fd51fef30a1a7db79204974209628061487609d422ad7a65fca82.jpg)

![49377e5482bb65f9a072ea84852e614c4ff1b2c9fc6fc8252fa0742defcee6dc.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/49377e5482bb65f9a072ea84852e614c4ff1b2c9fc6fc8252fa0742defcee6dc.jpg)

![4d8608b8636930ef6be35f478e62d0de5df024b2374c74e99f0c51646492df69.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/4d8608b8636930ef6be35f478e62d0de5df024b2374c74e99f0c51646492df69.jpg)

![54fec51ceba07c3e67117dd6a7d3e713ad65b53b61eb45b22b7e65edab0b3397.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/54fec51ceba07c3e67117dd6a7d3e713ad65b53b61eb45b22b7e65edab0b3397.jpg)

![56b5c547c9e25849582d3c340ea3c60ddbf94a6e27559841a337385fa0f846b7.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/56b5c547c9e25849582d3c340ea3c60ddbf94a6e27559841a337385fa0f846b7.jpg)

![576b7e70daf21ee9f07a3b120457ca6834fced228f214457a0a31bf00c5a49ce.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/576b7e70daf21ee9f07a3b120457ca6834fced228f214457a0a31bf00c5a49ce.jpg)

![5d4eee425af9749e2e4dad39575fa597004b75147cacada095a32bdc13b7fe20.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/5d4eee425af9749e2e4dad39575fa597004b75147cacada095a32bdc13b7fe20.jpg)

![5d8c004dd6c9fbfa3df8c6b5d29bde4ff59b635dff286ee99db2076fe2cb4e80.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/5d8c004dd6c9fbfa3df8c6b5d29bde4ff59b635dff286ee99db2076fe2cb4e80.jpg)

![5fed7f2ceda5d5fc26ec9c8475c5b5f0d73cc0ab3b47410ec9b9e1b9dc23e519.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/5fed7f2ceda5d5fc26ec9c8475c5b5f0d73cc0ab3b47410ec9b9e1b9dc23e519.jpg)

![6035983510bf7f8a83750533b03c7d8e6ea322bd8d91d18d3833ef2cd2654903.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/6035983510bf7f8a83750533b03c7d8e6ea322bd8d91d18d3833ef2cd2654903.jpg)

![618edde0177beb294d9ab84337f6772615d76321c8796ec6b2f7baf71b95d59b.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/618edde0177beb294d9ab84337f6772615d76321c8796ec6b2f7baf71b95d59b.jpg)

![649267deed1e970e8444c26859f4bafa8d270d0b204cbe9a89dd861f6976f48e.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/649267deed1e970e8444c26859f4bafa8d270d0b204cbe9a89dd861f6976f48e.jpg)

![64fe0ca993ef861d208c6592952d76d766da172e10219a244f8a7aae07e74fc2.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/64fe0ca993ef861d208c6592952d76d766da172e10219a244f8a7aae07e74fc2.jpg)

![65b5383496a90d1eae2bd17e37660b75aa9a3314e03e1de74a9e8833d93ceb65.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/65b5383496a90d1eae2bd17e37660b75aa9a3314e03e1de74a9e8833d93ceb65.jpg)

![6bf8abd29b0e83c2c94c320be75b3bd82c83af73e76187510ee8fd12d87a7ab8.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/6bf8abd29b0e83c2c94c320be75b3bd82c83af73e76187510ee8fd12d87a7ab8.jpg)

![6d7144f7fc5c87b9e8cbe28f63a0dda66d648345860ec7038ff2aa15624b8eba.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/6d7144f7fc5c87b9e8cbe28f63a0dda66d648345860ec7038ff2aa15624b8eba.jpg)

![75dd6a88738ed175a3364b2f7b0e5d5f8d749e075af2b350092fe3665e7eeda0.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/75dd6a88738ed175a3364b2f7b0e5d5f8d749e075af2b350092fe3665e7eeda0.jpg)

![77cc5fe0c2d487dff6feb14a6fabeff91ffdf98f64a1d1206a8c3a5b8ff634f2.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/77cc5fe0c2d487dff6feb14a6fabeff91ffdf98f64a1d1206a8c3a5b8ff634f2.jpg)

![7a77f5f53ce0009f4f3ee79052ebbb42572754e1dd10e3b596d43dad6f35406b.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/7a77f5f53ce0009f4f3ee79052ebbb42572754e1dd10e3b596d43dad6f35406b.jpg)

![7a7e8bd002c8599adf98f6f36c4b04674f792fa47587a43a422f4888bb3d9c95.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/7a7e8bd002c8599adf98f6f36c4b04674f792fa47587a43a422f4888bb3d9c95.jpg)

![7c06fc4fc4dd88b9c86b85d299bc3b0c33f4d783ed5c6732fed7a3226c4ce706.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/7c06fc4fc4dd88b9c86b85d299bc3b0c33f4d783ed5c6732fed7a3226c4ce706.jpg)

![7d72eda5076d89079acf89dbc38c6791c8b20693ae379b05ed293da197d646aa.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/7d72eda5076d89079acf89dbc38c6791c8b20693ae379b05ed293da197d646aa.jpg)

![840b3dcdca1bb18c81e0b998f8c777a0f5019cc307fcd04020e25ad49a3c7b39.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/840b3dcdca1bb18c81e0b998f8c777a0f5019cc307fcd04020e25ad49a3c7b39.jpg)

![861017d41de6433ace014e23c5c9dabdde2d2195c0d99c396977cba404d0fddf.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/861017d41de6433ace014e23c5c9dabdde2d2195c0d99c396977cba404d0fddf.jpg)

![8725a6bcd75d4b7d2ec761edc714730e8f4bb056417ca6ce3afc73b68210868c.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/8725a6bcd75d4b7d2ec761edc714730e8f4bb056417ca6ce3afc73b68210868c.jpg)

![8ddc1e0c38099fe955107236bcb450a21fc069455a37ee3ce2cdf036e9349752.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/8ddc1e0c38099fe955107236bcb450a21fc069455a37ee3ce2cdf036e9349752.jpg)

![90942a33bf1e929d2e7c231eb38dec2f36529504187238ceeabf030d95180f67.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/90942a33bf1e929d2e7c231eb38dec2f36529504187238ceeabf030d95180f67.jpg)

![91fccd9beed5fc9a660f1e684fd981c45674c954322da886f41585e4bf688d73.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/91fccd9beed5fc9a660f1e684fd981c45674c954322da886f41585e4bf688d73.jpg)

![95c85206404262e7fff070e92314d670d21034ae2a390813e9eb0824b540e46a.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/95c85206404262e7fff070e92314d670d21034ae2a390813e9eb0824b540e46a.jpg)

![967a60f24386186bade4de4dd75dd9c6b06eebb66cb571951dd0b01e2dfd516e.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/967a60f24386186bade4de4dd75dd9c6b06eebb66cb571951dd0b01e2dfd516e.jpg)

![982ecb65b90649347da7972068f617d96240b562060d6720d260303f266a4e53.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/982ecb65b90649347da7972068f617d96240b562060d6720d260303f266a4e53.jpg)

![9c17e75c98c4558614f79aa2a74e02e4d74a75465cdaee8c42d4c97ace494f7f.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/9c17e75c98c4558614f79aa2a74e02e4d74a75465cdaee8c42d4c97ace494f7f.jpg)

![9e6ac5ec91f1a90ac6fbb74437ba91e38cbc100aa60f9b5dbc607e6c3a61ec57.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/9e6ac5ec91f1a90ac6fbb74437ba91e38cbc100aa60f9b5dbc607e6c3a61ec57.jpg)

![a11d7d558c9d0ea9b98f994dd80de12b46b320267a224a4b752d714afb0ca327.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/a11d7d558c9d0ea9b98f994dd80de12b46b320267a224a4b752d714afb0ca327.jpg)

![a1e1afe371715f7b4be5ecc3955c6e2fc2ce91182a2ab5042e71436411101529.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/a1e1afe371715f7b4be5ecc3955c6e2fc2ce91182a2ab5042e71436411101529.jpg)

![a48933105b5d90b5276f29784dc05b283b0870826d19a7bfe16152cfc062130e.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/a48933105b5d90b5276f29784dc05b283b0870826d19a7bfe16152cfc062130e.jpg)

![a93e8e7bc4e3430292c97e6c3691e91ba3d2e8df56db9b555e5b8511daeebbdc.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/a93e8e7bc4e3430292c97e6c3691e91ba3d2e8df56db9b555e5b8511daeebbdc.jpg)

![ab39360fa2a973e27ade3461e3fa986dca5f2ddd1776aaf524d06784d9d85ecd.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/ab39360fa2a973e27ade3461e3fa986dca5f2ddd1776aaf524d06784d9d85ecd.jpg)

![abaec68abcbf7c840f1bd2d4a93af0e6c52cad633a016a94500d673622ce7b31.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/abaec68abcbf7c840f1bd2d4a93af0e6c52cad633a016a94500d673622ce7b31.jpg)

![abfce6efe653415e06e740d5002e5deded4f90b90fdd1906dc628908a099a8ef.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/abfce6efe653415e06e740d5002e5deded4f90b90fdd1906dc628908a099a8ef.jpg)

![ac3dbccf722c0df7212549ce77ba57aefa43b9eb9961b5163ad6bb605ebf0a25.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/ac3dbccf722c0df7212549ce77ba57aefa43b9eb9961b5163ad6bb605ebf0a25.jpg)

![b160d1722aa5ac19252ad363122bbebc57dec16796a5275f996819df5ffdc019.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/b160d1722aa5ac19252ad363122bbebc57dec16796a5275f996819df5ffdc019.jpg)

![b3de49ab725309eee2c7c833baa5b50925d233d0d3df6864fb2218150d2a9b7e.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/b3de49ab725309eee2c7c833baa5b50925d233d0d3df6864fb2218150d2a9b7e.jpg)

![b8a5478924e07881327721a3b84517d64f0945edc36cd2d224719d8fadb81ccc.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/b8a5478924e07881327721a3b84517d64f0945edc36cd2d224719d8fadb81ccc.jpg)

![bb7737b767ef11fd9f3d74dbd608b6c90d9be8468c63fac826d417ff4b1b3ef8.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/bb7737b767ef11fd9f3d74dbd608b6c90d9be8468c63fac826d417ff4b1b3ef8.jpg)

![bc1dd9af54696df2278f4b6b500fa492b3e094967e9126f00778e72dd7789a5f.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/bc1dd9af54696df2278f4b6b500fa492b3e094967e9126f00778e72dd7789a5f.jpg)

![be39e85eb21274de2df0de14b69404b8447a9009637ee890a8c7e3a427941605.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/be39e85eb21274de2df0de14b69404b8447a9009637ee890a8c7e3a427941605.jpg)

![c00d476b9661e5952dc9fe801849bb9eca7f9be43124a5b8303911f8aff28283.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/c00d476b9661e5952dc9fe801849bb9eca7f9be43124a5b8303911f8aff28283.jpg)

![c0d768a7fc6354a03d1c153d703e16c6f8a3e46e6e41c9f5eb790910559457bf.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/c0d768a7fc6354a03d1c153d703e16c6f8a3e46e6e41c9f5eb790910559457bf.jpg)

![c0f3eeae1733a6a50a64ef3d506ef65f1daf4d8587911f6c1881e838e20ef6ed.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/c0f3eeae1733a6a50a64ef3d506ef65f1daf4d8587911f6c1881e838e20ef6ed.jpg)

![c3dbe68925d49309c1acd8b8c931e0884d1a4e97d573a266ad6d09b3fa9338bc.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/c3dbe68925d49309c1acd8b8c931e0884d1a4e97d573a266ad6d09b3fa9338bc.jpg)

![c50f34d951007dcee1f19902580b082574fdac12c874bab2a02f0bd4f81c44a1.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/c50f34d951007dcee1f19902580b082574fdac12c874bab2a02f0bd4f81c44a1.jpg)

![c828f1197eed2e04cfb0f237b193490daccfaf87f6d8966f3e01c9ac667272e6.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/c828f1197eed2e04cfb0f237b193490daccfaf87f6d8966f3e01c9ac667272e6.jpg)

![c931a49db8bf76e552b02ffde574cf4ec78a04d10494a75975abaeafb16dd84e.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/c931a49db8bf76e552b02ffde574cf4ec78a04d10494a75975abaeafb16dd84e.jpg)

![c9d5f0b7ec07d6f5463cf81fedecb2500dfb88e19d40ed3a39e4023414a3c5ae.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/c9d5f0b7ec07d6f5463cf81fedecb2500dfb88e19d40ed3a39e4023414a3c5ae.jpg)

![cdac896204b5fa43d75739efe6c4763ea17b6469c642fb0c5abddb81ddbd7f47.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/cdac896204b5fa43d75739efe6c4763ea17b6469c642fb0c5abddb81ddbd7f47.jpg)

![daa4e6094fefaedf6f3f936e86010d5ee3b2b210a3d757c389dccc4e27da3244.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/daa4e6094fefaedf6f3f936e86010d5ee3b2b210a3d757c389dccc4e27da3244.jpg)

![dab465fd74445055b01363c780bf61bb396fcae68375ac70f9ce224263d89221.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/dab465fd74445055b01363c780bf61bb396fcae68375ac70f9ce224263d89221.jpg)

![dc83949c8b757a4148a0de83a58ca86affd6a77dcdc83e3b2da2145184eb3904.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/dc83949c8b757a4148a0de83a58ca86affd6a77dcdc83e3b2da2145184eb3904.jpg)

![dc8f8ad592e57f52b85e2aa44ba801d6b75c4fa84328886a1451caedef94c0b8.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/dc8f8ad592e57f52b85e2aa44ba801d6b75c4fa84328886a1451caedef94c0b8.jpg)

![ddee7aa8a0c91f5af316c7bf7ee0689b81bacd5f747a48c6e6345f4c59d7307b.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/ddee7aa8a0c91f5af316c7bf7ee0689b81bacd5f747a48c6e6345f4c59d7307b.jpg)

![df7e576a2fd2fe63f03da6c0bcf99a0bb0e140f47fc85fa59274869cbb3b3944.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/df7e576a2fd2fe63f03da6c0bcf99a0bb0e140f47fc85fa59274869cbb3b3944.jpg)

![e3f5b6b32a6287ddc247f1b71215e525aa3b88965421835f02f6ec243579acfd.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/e3f5b6b32a6287ddc247f1b71215e525aa3b88965421835f02f6ec243579acfd.jpg)

![e401b4cb5207d78a062409c2d39b1c1d7b46dd2c910083b0573c8dc148b290ef.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/e401b4cb5207d78a062409c2d39b1c1d7b46dd2c910083b0573c8dc148b290ef.jpg)

![e5985c7439d77018f85b6ca972cc1c4ece0c683c53a5f767f1dc51c3df4212f8.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/e5985c7439d77018f85b6ca972cc1c4ece0c683c53a5f767f1dc51c3df4212f8.jpg)

![e6f8bd55a8ba4a28df05544d01f50d8cb6eb1bc0a066926d31e0ddb45492fb85.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/e6f8bd55a8ba4a28df05544d01f50d8cb6eb1bc0a066926d31e0ddb45492fb85.jpg)

![e8365224ec9035abe5961e351806e8d4e736bd8a81d23832a0add56a6a27ab48.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/e8365224ec9035abe5961e351806e8d4e736bd8a81d23832a0add56a6a27ab48.jpg)

![e98fb1ebd1ed7457e509369bbfc5bd1751ba54cb087faca61e9e841cb2af33ea.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/e98fb1ebd1ed7457e509369bbfc5bd1751ba54cb087faca61e9e841cb2af33ea.jpg)

![ec6ba22f252cbfbebf731ca12e9c0a1e9b3cb9568b07a42e1f60497ad81f35bf.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/ec6ba22f252cbfbebf731ca12e9c0a1e9b3cb9568b07a42e1f60497ad81f35bf.jpg)

![ee06df03e3d9410c10a2c20310d5e30a6d566ca0a72e81802eb74f110e6c4eb3.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/ee06df03e3d9410c10a2c20310d5e30a6d566ca0a72e81802eb74f110e6c4eb3.jpg)

![f653c6e67842e0de4c94a72ca9b533ec6c1f777901df065fab956ae4ff0289fe.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/f653c6e67842e0de4c94a72ca9b533ec6c1f777901df065fab956ae4ff0289fe.jpg)

![fa9f6118c924b432a8099bb17bd3d3e71357c03da9051db2749605ab6a571d57.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/fa9f6118c924b432a8099bb17bd3d3e71357c03da9051db2749605ab6a571d57.jpg)

![fb4b38b34a8e7068b769dc14f5c2f80d64ab6900ab51fefcb8a70ec60cb7ad35.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/fb4b38b34a8e7068b769dc14f5c2f80d64ab6900ab51fefcb8a70ec60cb7ad35.jpg)

![fd68e26161699ece88aa009e002c67813fc487fbd416db95d632895ac2c8f79a.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/fd68e26161699ece88aa009e002c67813fc487fbd416db95d632895ac2c8f79a.jpg)

![ff4ad8ba8744d3824cb72f365f9977baacf06094588f80f3d3aefe9401ba7e08.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/images/ff4ad8ba8744d3824cb72f365f9977baacf06094588f80f3d3aefe9401ba7e08.jpg)

### Tables

![1d6a9b63dfea7cb6fbb98481582d5d6054574905d8019dc33beb337686bcecc5.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/tables/1d6a9b63dfea7cb6fbb98481582d5d6054574905d8019dc33beb337686bcecc5.jpg)

![32d1ea927e7bb161024d1d2f26ae3e43f4530281aa707f25dd1f347e00bf4256.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/tables/32d1ea927e7bb161024d1d2f26ae3e43f4530281aa707f25dd1f347e00bf4256.jpg)

![43b2abce5fc52d5bc62f9b4514a7608bc0de723e26d86a6240c7061b3f7d761d.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/tables/43b2abce5fc52d5bc62f9b4514a7608bc0de723e26d86a6240c7061b3f7d761d.jpg)

![6727fc2335aef43e4f4c81986d3d5dd26c67cb7ab141c7ac1db4488503a9d7c4.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/tables/6727fc2335aef43e4f4c81986d3d5dd26c67cb7ab141c7ac1db4488503a9d7c4.jpg)

![86caf6dbfb59cd488833bcadf740f36108cf646aaca6868825f3eb0442a79ff8.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/tables/86caf6dbfb59cd488833bcadf740f36108cf646aaca6868825f3eb0442a79ff8.jpg)

![9c3d33291fb6ba3039dda9ae3c6f41fd7c17eeeb511a518ab4d9fe69a5e01058.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/tables/9c3d33291fb6ba3039dda9ae3c6f41fd7c17eeeb511a518ab4d9fe69a5e01058.jpg)

![d43086f2051ffc363738cdd0cf0f6087f5cb11b2b4aa1a2e97879a4ec55e8f8e.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/tables/d43086f2051ffc363738cdd0cf0f6087f5cb11b2b4aa1a2e97879a4ec55e8f8e.jpg)

![d9143b9c8c0cb037d69371cfa0562492c3bb3e9cd32ae4fc2c5736ec79310891.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/tables/d9143b9c8c0cb037d69371cfa0562492c3bb3e9cd32ae4fc2c5736ec79310891.jpg)

![eb79d48ead7013391c4b74801c827357f149bc80d312af2d9f2d2a9d06db1f78.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/tables/eb79d48ead7013391c4b74801c827357f149bc80d312af2d9f2d2a9d06db1f78.jpg)

![f454f5ffd4f0253f63b56823784685e3fb174907e2eaf3d9edddecc5e71d1277.jpg](../iclr_results/1043_Robust Transfer of Safety-Constrained Reinforcement Learning Agents/tables/f454f5ffd4f0253f63b56823784685e3fb174907e2eaf3d9edddecc5e71d1277.jpg)

## Online-to-Offline RL for Agent Alignment


### Images

![2314322947641d65243c4fd5088e96cb095107c269394c6c4655563b2878a113.jpg](../iclr_results/1044_Online-to-Offline RL for Agent Alignment/images/2314322947641d65243c4fd5088e96cb095107c269394c6c4655563b2878a113.jpg)

![29e0dd6c787a8c4fd0a9b92c76eab501d74620928103ee2bed383c044816f9c7.jpg](../iclr_results/1044_Online-to-Offline RL for Agent Alignment/images/29e0dd6c787a8c4fd0a9b92c76eab501d74620928103ee2bed383c044816f9c7.jpg)

![31c99d5d2a09dbbb5b461d8581d68c4f71f2409c549ba39af9fbf6d5ca9107db.jpg](../iclr_results/1044_Online-to-Offline RL for Agent Alignment/images/31c99d5d2a09dbbb5b461d8581d68c4f71f2409c549ba39af9fbf6d5ca9107db.jpg)

![581f5350b3efeb6414fdb6e775d5fcfa7a7b029cdd49baf5f7f9646aadbf4e01.jpg](../iclr_results/1044_Online-to-Offline RL for Agent Alignment/images/581f5350b3efeb6414fdb6e775d5fcfa7a7b029cdd49baf5f7f9646aadbf4e01.jpg)

![66d2f0855d83db5264488566b16a9952fdb15819f6ba750f242068b1f316209e.jpg](../iclr_results/1044_Online-to-Offline RL for Agent Alignment/images/66d2f0855d83db5264488566b16a9952fdb15819f6ba750f242068b1f316209e.jpg)

![804737c4d7826e92a052aff7763fb0d798e3e49cdd7fdf8b3d68339931817627.jpg](../iclr_results/1044_Online-to-Offline RL for Agent Alignment/images/804737c4d7826e92a052aff7763fb0d798e3e49cdd7fdf8b3d68339931817627.jpg)

![8219e3c72905d7c9a8b55afb2f7ba030a0f31390fb4638ab14bb3fc15c0166ec.jpg](../iclr_results/1044_Online-to-Offline RL for Agent Alignment/images/8219e3c72905d7c9a8b55afb2f7ba030a0f31390fb4638ab14bb3fc15c0166ec.jpg)

![8aec52f9a9a96e36bcbb587629942973ae172311a2d55dadb827a7e17dd51703.jpg](../iclr_results/1044_Online-to-Offline RL for Agent Alignment/images/8aec52f9a9a96e36bcbb587629942973ae172311a2d55dadb827a7e17dd51703.jpg)

![907e17bb75988ed28e46da563201eb10b7db59896b3f80037243c3623095a916.jpg](../iclr_results/1044_Online-to-Offline RL for Agent Alignment/images/907e17bb75988ed28e46da563201eb10b7db59896b3f80037243c3623095a916.jpg)

![c8eea40fc9154b0a3505d0cc288b583b86aa9ec642cd038dde1a28d6832fb2f3.jpg](../iclr_results/1044_Online-to-Offline RL for Agent Alignment/images/c8eea40fc9154b0a3505d0cc288b583b86aa9ec642cd038dde1a28d6832fb2f3.jpg)

![dd6b626ce18fddd6a7882b015f707421c8d979158bd4588e4da9ff481b13e359.jpg](../iclr_results/1044_Online-to-Offline RL for Agent Alignment/images/dd6b626ce18fddd6a7882b015f707421c8d979158bd4588e4da9ff481b13e359.jpg)

### Tables

![2063bc96e9c25f227e8f00feda5dc05efdcb8e20e1e7511963a9bc30971431a6.jpg](../iclr_results/1044_Online-to-Offline RL for Agent Alignment/tables/2063bc96e9c25f227e8f00feda5dc05efdcb8e20e1e7511963a9bc30971431a6.jpg)

![5c0527ff7fe7a77cc0939f5a59d583ed1358102390b4f133846698bebc940b9c.jpg](../iclr_results/1044_Online-to-Offline RL for Agent Alignment/tables/5c0527ff7fe7a77cc0939f5a59d583ed1358102390b4f133846698bebc940b9c.jpg)

![62a516d4607285469571b4bb511922892a6a331c26a3fa3226d0f5d97b4979ae.jpg](../iclr_results/1044_Online-to-Offline RL for Agent Alignment/tables/62a516d4607285469571b4bb511922892a6a331c26a3fa3226d0f5d97b4979ae.jpg)

![8b3bd56aa11533dafeca8744b6212bb40891516f062dc17b7a611bcb882b3d3c.jpg](../iclr_results/1044_Online-to-Offline RL for Agent Alignment/tables/8b3bd56aa11533dafeca8744b6212bb40891516f062dc17b7a611bcb882b3d3c.jpg)

![c89d18a32cb6ffd85a282f5ebcc24ff61a3847080875abbbcb7cd63db926a6cf.jpg](../iclr_results/1044_Online-to-Offline RL for Agent Alignment/tables/c89d18a32cb6ffd85a282f5ebcc24ff61a3847080875abbbcb7cd63db926a6cf.jpg)

![ce2ff196af3547f1be14fd71a977cf3879125c8847e9c8b4923c2af309a8024b.jpg](../iclr_results/1044_Online-to-Offline RL for Agent Alignment/tables/ce2ff196af3547f1be14fd71a977cf3879125c8847e9c8b4923c2af309a8024b.jpg)

![df319287bb6d85d6a6d28c262fac0426735124fe5f4412b21f8199178025d2a2.jpg](../iclr_results/1044_Online-to-Offline RL for Agent Alignment/tables/df319287bb6d85d6a6d28c262fac0426735124fe5f4412b21f8199178025d2a2.jpg)

## Self-Introspective Decoding: Alleviating Hallucinations for Large Vision-Language Models


### Images

![39107e84cc1673522bf7d2dc7c5809744469f04cd64bd762ca1991a72ccdbfb3.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/images/39107e84cc1673522bf7d2dc7c5809744469f04cd64bd762ca1991a72ccdbfb3.jpg)

![3d8447632193a355bf3c453e6c87bc6ceec5bf62c51c481edd0758bf81c34c92.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/images/3d8447632193a355bf3c453e6c87bc6ceec5bf62c51c481edd0758bf81c34c92.jpg)

![585b5dcb6c3011eef0f2b3ee48b07b28122497c80aa58b5d5d0d0bc12ecdbded.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/images/585b5dcb6c3011eef0f2b3ee48b07b28122497c80aa58b5d5d0d0bc12ecdbded.jpg)

![6030b550372c92ebcbdf03c12623957d86f5d61b0e23d1a0266ee7a2efe10638.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/images/6030b550372c92ebcbdf03c12623957d86f5d61b0e23d1a0266ee7a2efe10638.jpg)

![84a9c6ee36adbb63e7bf735895834327d0a2127e993bf932a0a9ecde3e3e66d6.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/images/84a9c6ee36adbb63e7bf735895834327d0a2127e993bf932a0a9ecde3e3e66d6.jpg)

![8e076c98d5e1b8df8801da49632b20c92e896dec7a63c86a5d0a7d5e7ad1ea68.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/images/8e076c98d5e1b8df8801da49632b20c92e896dec7a63c86a5d0a7d5e7ad1ea68.jpg)

![939a1cf7cd27b430de68bd51ca317f2aaec80ac1301b2328f2b7f971404a85ab.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/images/939a1cf7cd27b430de68bd51ca317f2aaec80ac1301b2328f2b7f971404a85ab.jpg)

![a01cffce3ea6ef2409bdc32ed7dd473708a5a4c335f60cac13b239a0d2b9684c.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/images/a01cffce3ea6ef2409bdc32ed7dd473708a5a4c335f60cac13b239a0d2b9684c.jpg)

![c386298a60129dff5e65ee933f320c441ef87be205ccea1b8a52849a317f888a.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/images/c386298a60129dff5e65ee933f320c441ef87be205ccea1b8a52849a317f888a.jpg)

![d08514475510bd8101687d00cce93f8138cee016649171f6ae30eb593b3b86cb.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/images/d08514475510bd8101687d00cce93f8138cee016649171f6ae30eb593b3b86cb.jpg)

![e076fab8eac068bf08d9a803698add80fde7cc0b44e9f26c90bf5f0146cb72c0.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/images/e076fab8eac068bf08d9a803698add80fde7cc0b44e9f26c90bf5f0146cb72c0.jpg)

![ea0fe3f2dc4deeccf55c031669e02559eaa945cd31ee9cc7257e0168fa6f9bfc.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/images/ea0fe3f2dc4deeccf55c031669e02559eaa945cd31ee9cc7257e0168fa6f9bfc.jpg)

![fb1844a9a31412c646432ccbb2f72d6b2ee1f057f7fc713e6405159fd91abc77.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/images/fb1844a9a31412c646432ccbb2f72d6b2ee1f057f7fc713e6405159fd91abc77.jpg)

### Tables

![11db56f4a06603302ad1d223da314f512c322f22c5ff773e91a5e999e4854c80.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/tables/11db56f4a06603302ad1d223da314f512c322f22c5ff773e91a5e999e4854c80.jpg)

![1ac99829d2806ac0ac5fa84a28c511667398757ade9be88467f65101ca63e1a5.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/tables/1ac99829d2806ac0ac5fa84a28c511667398757ade9be88467f65101ca63e1a5.jpg)

![3ff6db295ae2f5e9a4fefa879114cd4a668d2eb4d46ca8f6491832ca698f03e4.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/tables/3ff6db295ae2f5e9a4fefa879114cd4a668d2eb4d46ca8f6491832ca698f03e4.jpg)

![47e5417062061cbd9470ef71c86b2e2bb5c93c21436d1bab4db1f478a7610ebc.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/tables/47e5417062061cbd9470ef71c86b2e2bb5c93c21436d1bab4db1f478a7610ebc.jpg)

![6ae791e803ffd37fab9fc552747c83fdc67aff34fcba2c141fb2ffab2b168fc5.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/tables/6ae791e803ffd37fab9fc552747c83fdc67aff34fcba2c141fb2ffab2b168fc5.jpg)

![7056944547c4f100a95bb4227d04f6cd720286869c7784c47fa3f2002681611a.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/tables/7056944547c4f100a95bb4227d04f6cd720286869c7784c47fa3f2002681611a.jpg)

![785408e531ca364100de7aec4737dbd31d918a6b96629bc6d7c766c92eff93f8.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/tables/785408e531ca364100de7aec4737dbd31d918a6b96629bc6d7c766c92eff93f8.jpg)

![8df114c0b00ed6c59bc0023a93984fe550bb5ca9691d871ee91aeef95da30328.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/tables/8df114c0b00ed6c59bc0023a93984fe550bb5ca9691d871ee91aeef95da30328.jpg)

![b33b8d16f8974660f71dab7a003ec71bac01f75a596af00728532f854ba16d8b.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/tables/b33b8d16f8974660f71dab7a003ec71bac01f75a596af00728532f854ba16d8b.jpg)

![b596c4fe0c4e2d74a4fadc13f3d35073c2620c77074d7a4849c4cda438a4bb02.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/tables/b596c4fe0c4e2d74a4fadc13f3d35073c2620c77074d7a4849c4cda438a4bb02.jpg)

![c3b8af90b66bb8808e9c241db505228ee6cad35387afe805760a4759ad5ddd33.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/tables/c3b8af90b66bb8808e9c241db505228ee6cad35387afe805760a4759ad5ddd33.jpg)

![cd8e56aa5da5b3b598f77494cc49239e1e2b5dd5d4e5e5f3e13722b89ab72787.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/tables/cd8e56aa5da5b3b598f77494cc49239e1e2b5dd5d4e5e5f3e13722b89ab72787.jpg)

![f2dc5717b91c51606a8b28e5458c578e10c15283e73708d028ed020ec5a0bd2c.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/tables/f2dc5717b91c51606a8b28e5458c578e10c15283e73708d028ed020ec5a0bd2c.jpg)

![f9fc348b9eb3e20665bd83627fc90ab3459eff624efc267f73a18dc596c1bed5.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/tables/f9fc348b9eb3e20665bd83627fc90ab3459eff624efc267f73a18dc596c1bed5.jpg)

![fa56c76b9931a0dba47ef52a8cde53afc887c1763abad1711e66f5edfa0bfe2e.jpg](../iclr_results/1045_Self-Introspective Decoding_ Alleviating Hallucinations for Large Vision-Language Models/tables/fa56c76b9931a0dba47ef52a8cde53afc887c1763abad1711e66f5edfa0bfe2e.jpg)

## Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling


### Images

![12b94f685ab7c0227f3750ea28a19a39efb297415edb1332420283c4ff3ffad4.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/images/12b94f685ab7c0227f3750ea28a19a39efb297415edb1332420283c4ff3ffad4.jpg)

![12c0044f8d282aea335d90a1ac18e4650f5eb90ee63255d7911e2e2672992640.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/images/12c0044f8d282aea335d90a1ac18e4650f5eb90ee63255d7911e2e2672992640.jpg)

![16bda5bbdec47cba3365c1c56a3025f6f77d9cf62c7b8c0929d38de0ded9dd02.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/images/16bda5bbdec47cba3365c1c56a3025f6f77d9cf62c7b8c0929d38de0ded9dd02.jpg)

![25b9e846c3a693a23cd15f7fc17c86965fa56f95b1510409298cd3dc3e83283a.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/images/25b9e846c3a693a23cd15f7fc17c86965fa56f95b1510409298cd3dc3e83283a.jpg)

![362f222cdaee6175b6c49c6a50df74b6f6376ba6334d3be469ce9cb2dbe72540.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/images/362f222cdaee6175b6c49c6a50df74b6f6376ba6334d3be469ce9cb2dbe72540.jpg)

![50d0358072b173881e21ca484b2480abaa7057347044ed7d83457e35428c2c4b.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/images/50d0358072b173881e21ca484b2480abaa7057347044ed7d83457e35428c2c4b.jpg)

![53462127cb743f09a29c572185bf9f3f5f3303d15160c1739b6025f915f52906.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/images/53462127cb743f09a29c572185bf9f3f5f3303d15160c1739b6025f915f52906.jpg)

![8c83cb7ddce7720075152f76965a5466c226d5f0cfdbcc99f9862d8f1c556b10.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/images/8c83cb7ddce7720075152f76965a5466c226d5f0cfdbcc99f9862d8f1c556b10.jpg)

![8e4a6e1d28d8df91317ef6e4b9ae954a85b4149995a78b6bed6bfa0630f0cc7f.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/images/8e4a6e1d28d8df91317ef6e4b9ae954a85b4149995a78b6bed6bfa0630f0cc7f.jpg)

![922fc55d2c24c6f4a088d00202b8210be524849f9487ac20a0275154638d9304.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/images/922fc55d2c24c6f4a088d00202b8210be524849f9487ac20a0275154638d9304.jpg)

![a30f37acf358f02f690b83d1214ac93df0334ac0366c6a5644735481bf4f19dc.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/images/a30f37acf358f02f690b83d1214ac93df0334ac0366c6a5644735481bf4f19dc.jpg)

![a6f3e7df20145399a311c303179f47808f7cc9ed925d5d4da3c6ebb80bbd10bc.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/images/a6f3e7df20145399a311c303179f47808f7cc9ed925d5d4da3c6ebb80bbd10bc.jpg)

![a960add23c4c638c41f93a05b17731d6c45233c15c4db350d01513a7313c2ff6.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/images/a960add23c4c638c41f93a05b17731d6c45233c15c4db350d01513a7313c2ff6.jpg)

![aac7ce2815a221f1522d3e1895228b42a699a3e4902905df27dd33046abefa54.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/images/aac7ce2815a221f1522d3e1895228b42a699a3e4902905df27dd33046abefa54.jpg)

![bcabd7848fdfe62e58b13b09998a278c81a9d02c3ac0e1283c7b000218ae9cd0.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/images/bcabd7848fdfe62e58b13b09998a278c81a9d02c3ac0e1283c7b000218ae9cd0.jpg)

![be9a6e3bf0e6c7e1d7667c13e034aef1b28885858dca7f8b07bb48c3509d7299.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/images/be9a6e3bf0e6c7e1d7667c13e034aef1b28885858dca7f8b07bb48c3509d7299.jpg)

![cca2a1e834f57a03c419f7005bdf7b4b618950e5d532156433adbe512108bfd7.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/images/cca2a1e834f57a03c419f7005bdf7b4b618950e5d532156433adbe512108bfd7.jpg)

![d3e51fbb3d4609d91da52102ad4e8766b75ef7c0e7ff936837bbdc45447c88be.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/images/d3e51fbb3d4609d91da52102ad4e8766b75ef7c0e7ff936837bbdc45447c88be.jpg)

![da7b5027a39ca45404dc9a2bc87996f0fef9c3eb35da7931b8bac83c3d81b8d9.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/images/da7b5027a39ca45404dc9a2bc87996f0fef9c3eb35da7931b8bac83c3d81b8d9.jpg)

![ee9f8b105272f654706383b77b7cb7a6405ac3f132a75a66af80e4bb43ce863d.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/images/ee9f8b105272f654706383b77b7cb7a6405ac3f132a75a66af80e4bb43ce863d.jpg)

![f03c59636bdf9be3b31005f96f0e27e1c7ac3d943d23c3d2103a858710e156d7.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/images/f03c59636bdf9be3b31005f96f0e27e1c7ac3d943d23c3d2103a858710e156d7.jpg)

### Tables

![2bbdcbdfedc8eac56182e1080867c3a53dde3770e53b645879df2a328af21c08.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/tables/2bbdcbdfedc8eac56182e1080867c3a53dde3770e53b645879df2a328af21c08.jpg)

![561465628d6b82f34a5ec39bca1de4643e81ba4691a7415df0ee321f92193577.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/tables/561465628d6b82f34a5ec39bca1de4643e81ba4691a7415df0ee321f92193577.jpg)

![5e1bcfbab6b59485d0d57d9001e31d9dd4eda8a833139696cb6c61f921be3f91.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/tables/5e1bcfbab6b59485d0d57d9001e31d9dd4eda8a833139696cb6c61f921be3f91.jpg)

![b8082b4cd208c31aec22f35ceb5cb25879b8721d272992cbd7b23f60e6f19181.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/tables/b8082b4cd208c31aec22f35ceb5cb25879b8721d272992cbd7b23f60e6f19181.jpg)

![e2ce1f9bc0b03f4c6e3d9b907417dee7580975b3baa2952dbad2ede7d75adc44.jpg](../iclr_results/1046_Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling/tables/e2ce1f9bc0b03f4c6e3d9b907417dee7580975b3baa2952dbad2ede7d75adc44.jpg)

## ConvCodeWorld: Benchmarking Conversational Code Generation in Reproducible Feedback Environments


### Images

![208141cf0bf9131d617efb63652f04b3bcf24049ab1469e92b1fae09f27e35ad.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/images/208141cf0bf9131d617efb63652f04b3bcf24049ab1469e92b1fae09f27e35ad.jpg)

![294cf5c0869815cabbdccd97c184b08407d8de9bf5b586ebc0be4c2088983004.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/images/294cf5c0869815cabbdccd97c184b08407d8de9bf5b586ebc0be4c2088983004.jpg)

![4e87c4f83322dd7546f62c20d721e41762514e3b6a1afa3f87a926598d109970.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/images/4e87c4f83322dd7546f62c20d721e41762514e3b6a1afa3f87a926598d109970.jpg)

![56065ac56c63ea20779fc1776fd12741003096ac33a67a77a51fdaf9474e1892.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/images/56065ac56c63ea20779fc1776fd12741003096ac33a67a77a51fdaf9474e1892.jpg)

![56f2803c7748c8d758bc1f06ad6c3e18d0f1ddd855d2a99314c46ae6225974a8.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/images/56f2803c7748c8d758bc1f06ad6c3e18d0f1ddd855d2a99314c46ae6225974a8.jpg)

![6084e0ea4d8efbf62d6d386599861386765d07cc5ee3b2526b3828b0691d75d1.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/images/6084e0ea4d8efbf62d6d386599861386765d07cc5ee3b2526b3828b0691d75d1.jpg)

![69be06d10b21fd11c8a4bac66a14b10641ae63ecad0e28237197d99f5fdc7e31.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/images/69be06d10b21fd11c8a4bac66a14b10641ae63ecad0e28237197d99f5fdc7e31.jpg)

![6c709383dd1e7da3ab2f2ba1f17e47de4c2c4733354a663ca0d96077c53590a9.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/images/6c709383dd1e7da3ab2f2ba1f17e47de4c2c4733354a663ca0d96077c53590a9.jpg)

![70827260b7615135e41344cea6da46937946b50dda9a6bf9734050c3daad9655.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/images/70827260b7615135e41344cea6da46937946b50dda9a6bf9734050c3daad9655.jpg)

![74c5affc7048aac9e893a5a111495d149fb6e79a0a8e0c2c28a7b68642ce67d4.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/images/74c5affc7048aac9e893a5a111495d149fb6e79a0a8e0c2c28a7b68642ce67d4.jpg)

![7b19b41dd56fa14f86835edf3cd5ad8bf62ac66bbcbcba770dbd032654e65872.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/images/7b19b41dd56fa14f86835edf3cd5ad8bf62ac66bbcbcba770dbd032654e65872.jpg)

![81cddd86809a2f69804d90f9c11a3abdaba8b7131d7dcee2db6493799eac7f5e.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/images/81cddd86809a2f69804d90f9c11a3abdaba8b7131d7dcee2db6493799eac7f5e.jpg)

![83d6d171dbca7d6b30bde819fb98f34ae9a2b0fea1832d2c67e6302c09d3e7b0.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/images/83d6d171dbca7d6b30bde819fb98f34ae9a2b0fea1832d2c67e6302c09d3e7b0.jpg)

![89c34c2e9938beb1bc9dc522a02429a69e8cebf73d87edf499fb9c2e2547f809.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/images/89c34c2e9938beb1bc9dc522a02429a69e8cebf73d87edf499fb9c2e2547f809.jpg)

![8a0ee2d8b8933a71b15a3bd4c0a0640985e784d62ecc1c736e917883f1e63e4d.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/images/8a0ee2d8b8933a71b15a3bd4c0a0640985e784d62ecc1c736e917883f1e63e4d.jpg)

![9c643eeb708a1971c98d0434b2e53a200255b4d18c49bc209992f0e8f16e683b.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/images/9c643eeb708a1971c98d0434b2e53a200255b4d18c49bc209992f0e8f16e683b.jpg)

![a5e0354d4a8820c79eccc41e9537fd1059f94d4ed0f0a317952b5ab9665991b5.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/images/a5e0354d4a8820c79eccc41e9537fd1059f94d4ed0f0a317952b5ab9665991b5.jpg)

![adc9664bd7463844c45bb87c75cb075cc240bd93db7307451e7c35896e6ea800.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/images/adc9664bd7463844c45bb87c75cb075cc240bd93db7307451e7c35896e6ea800.jpg)

![d2d00a971434a7a3c1e2b2d2e320d686d379d3b8621874dbdd6b2ed3cd52bdd4.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/images/d2d00a971434a7a3c1e2b2d2e320d686d379d3b8621874dbdd6b2ed3cd52bdd4.jpg)

![d7bcb7677e6c2f6da3e73d9a9eb574c5b34c117b37c6230224caca5af00dac8d.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/images/d7bcb7677e6c2f6da3e73d9a9eb574c5b34c117b37c6230224caca5af00dac8d.jpg)

![e5b9710e7e116c6eb468ce01127641175bcc3aa85a21be91c1f5220a434ca285.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/images/e5b9710e7e116c6eb468ce01127641175bcc3aa85a21be91c1f5220a434ca285.jpg)

### Tables

![0fa7c0fc0500158f14e7cce08bc4bdd95eb3a85dfc9fd0127df26e716dea6fbb.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/tables/0fa7c0fc0500158f14e7cce08bc4bdd95eb3a85dfc9fd0127df26e716dea6fbb.jpg)

![0fb950dc45fc44237b9ad226e71a206cabd923b9db40e45d40e37aedf60201b5.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/tables/0fb950dc45fc44237b9ad226e71a206cabd923b9db40e45d40e37aedf60201b5.jpg)

![130d1c4426d90ffc028f53516d9aefa0aa19253fd1c88fdd17527b118fe5325e.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/tables/130d1c4426d90ffc028f53516d9aefa0aa19253fd1c88fdd17527b118fe5325e.jpg)

![139a2224b39015f7954c5165b75acd5175e2551d5aa48a7f73211919b4898d0d.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/tables/139a2224b39015f7954c5165b75acd5175e2551d5aa48a7f73211919b4898d0d.jpg)

![1a5002d83d2ed4f2eb002bfb9a42314383c6a94d76a8f9ded7374cd4af3e1ea7.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/tables/1a5002d83d2ed4f2eb002bfb9a42314383c6a94d76a8f9ded7374cd4af3e1ea7.jpg)

![2df7b901021eeabc7093691b27d4302b9f2dd49f16f1050b72beffb0d3e8186f.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/tables/2df7b901021eeabc7093691b27d4302b9f2dd49f16f1050b72beffb0d3e8186f.jpg)

![3a35e2a62092b2d4dd8fa18aee5acf0c2fcc39760a9ef8599a55322d50b91f27.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/tables/3a35e2a62092b2d4dd8fa18aee5acf0c2fcc39760a9ef8599a55322d50b91f27.jpg)

![4ae9ab85dca14f3ccd33fa8c42482ece6a9da6eff622d6bdae2326ac4362353c.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/tables/4ae9ab85dca14f3ccd33fa8c42482ece6a9da6eff622d6bdae2326ac4362353c.jpg)

![528c6ef3ed2cd2820d11294fc4c4badda3c0a28ff67ca726d9dd715251a0e256.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/tables/528c6ef3ed2cd2820d11294fc4c4badda3c0a28ff67ca726d9dd715251a0e256.jpg)

![53c638c4bacee3ca5a44a35ab89508d5a956be607a725bf2fed82074e665ee13.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/tables/53c638c4bacee3ca5a44a35ab89508d5a956be607a725bf2fed82074e665ee13.jpg)

![6fd6ddc963a740efbdc4193875f5a2ba5ffab9efc9b86e77630f339110ffacd9.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/tables/6fd6ddc963a740efbdc4193875f5a2ba5ffab9efc9b86e77630f339110ffacd9.jpg)

![a32f4d98ea33937913a219d1de3a6332102004a559fed5f0f0082ba196c6312c.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/tables/a32f4d98ea33937913a219d1de3a6332102004a559fed5f0f0082ba196c6312c.jpg)

![cd5ddbff5d05e5f0406f9e2a33234d0def374fc5fe298b46d427fb9ad4dac4cd.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/tables/cd5ddbff5d05e5f0406f9e2a33234d0def374fc5fe298b46d427fb9ad4dac4cd.jpg)

![ce983ef11917c0936e1dc07fd672f59c0fc2339018bdee992da65dfe045ce9f8.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/tables/ce983ef11917c0936e1dc07fd672f59c0fc2339018bdee992da65dfe045ce9f8.jpg)

![d96175a246f4caa8103435ec653aca4cb970ce1bef68ce67a834e1a7ab35b2ad.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/tables/d96175a246f4caa8103435ec653aca4cb970ce1bef68ce67a834e1a7ab35b2ad.jpg)

![e4ffc4fbe3eb7308ee82aaccde2d2f7ebd3cc3cdb6894fa4e6150debf0370ed8.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/tables/e4ffc4fbe3eb7308ee82aaccde2d2f7ebd3cc3cdb6894fa4e6150debf0370ed8.jpg)

![e5e85d64622b057a4d6e90df896222bed9eedbf4c645a62af3bfb6c2f20c582c.jpg](../iclr_results/1047_ConvCodeWorld_ Benchmarking Conversational Code Generation in Reproducible Feedback Environments/tables/e5e85d64622b057a4d6e90df896222bed9eedbf4c645a62af3bfb6c2f20c582c.jpg)

## {$\tau$}-bench: A Benchmark for \underline{T}ool-\underline{A}gent-\underline{U}ser Interaction in Real-World Domains


### Images

![1cc5257170258e073da2d257f8443ecce7091495b15cdf8e38a7216a4e3026c2.jpg](../iclr_results/1048_{$_tau$}-bench_ A Benchmark for _underline{T}ool-_underline{A}gent-_underline{U}ser Interaction in R/images/1cc5257170258e073da2d257f8443ecce7091495b15cdf8e38a7216a4e3026c2.jpg)

![45545178d144061ee66f1ed0e5db2e126d5dd92f337d2ec363282598b42ead67.jpg](../iclr_results/1048_{$_tau$}-bench_ A Benchmark for _underline{T}ool-_underline{A}gent-_underline{U}ser Interaction in R/images/45545178d144061ee66f1ed0e5db2e126d5dd92f337d2ec363282598b42ead67.jpg)

![8c06eb56001e34697e7314d82ce0dd2e890a4af7784de3ee43fe66a48a1c2bef.jpg](../iclr_results/1048_{$_tau$}-bench_ A Benchmark for _underline{T}ool-_underline{A}gent-_underline{U}ser Interaction in R/images/8c06eb56001e34697e7314d82ce0dd2e890a4af7784de3ee43fe66a48a1c2bef.jpg)

![a2efdf1d6e243e167b34697d572d8cde682de1886f705c5e01eb6fc45d02e3c6.jpg](../iclr_results/1048_{$_tau$}-bench_ A Benchmark for _underline{T}ool-_underline{A}gent-_underline{U}ser Interaction in R/images/a2efdf1d6e243e167b34697d572d8cde682de1886f705c5e01eb6fc45d02e3c6.jpg)

![c539caaa63c701fc4dee5e9e549bd534f4979424bf648bc5c673d071ffc1061c.jpg](../iclr_results/1048_{$_tau$}-bench_ A Benchmark for _underline{T}ool-_underline{A}gent-_underline{U}ser Interaction in R/images/c539caaa63c701fc4dee5e9e549bd534f4979424bf648bc5c673d071ffc1061c.jpg)

![dab85214bee6db4614bbd44c58f0415f3f09e2e284b4258f5214d3a5fb41bd30.jpg](../iclr_results/1048_{$_tau$}-bench_ A Benchmark for _underline{T}ool-_underline{A}gent-_underline{U}ser Interaction in R/images/dab85214bee6db4614bbd44c58f0415f3f09e2e284b4258f5214d3a5fb41bd30.jpg)

![ec319dbb3f3f0abdf5666e6386ce80eff6d84c8440b129b952625a814769ed43.jpg](../iclr_results/1048_{$_tau$}-bench_ A Benchmark for _underline{T}ool-_underline{A}gent-_underline{U}ser Interaction in R/images/ec319dbb3f3f0abdf5666e6386ce80eff6d84c8440b129b952625a814769ed43.jpg)

### Tables

![0a57a60de52831530a274663d9daafd044479aafc44d1082f564b84ba37348c9.jpg](../iclr_results/1048_{$_tau$}-bench_ A Benchmark for _underline{T}ool-_underline{A}gent-_underline{U}ser Interaction in R/tables/0a57a60de52831530a274663d9daafd044479aafc44d1082f564b84ba37348c9.jpg)

![7e986985095c39b9e16cc62e86602f62f1ef7b68fdd06e2d93aac6a681efe0cd.jpg](../iclr_results/1048_{$_tau$}-bench_ A Benchmark for _underline{T}ool-_underline{A}gent-_underline{U}ser Interaction in R/tables/7e986985095c39b9e16cc62e86602f62f1ef7b68fdd06e2d93aac6a681efe0cd.jpg)

![a527fb4ac4eb8f12b082f119a32f3020e9fbac8405bff5e4cdd2483c11d02143.jpg](../iclr_results/1048_{$_tau$}-bench_ A Benchmark for _underline{T}ool-_underline{A}gent-_underline{U}ser Interaction in R/tables/a527fb4ac4eb8f12b082f119a32f3020e9fbac8405bff5e4cdd2483c11d02143.jpg)

![b2efa519ca7f114ec0e93f82911f99f1cc79414ffc7f70e6dd6714216fe39417.jpg](../iclr_results/1048_{$_tau$}-bench_ A Benchmark for _underline{T}ool-_underline{A}gent-_underline{U}ser Interaction in R/tables/b2efa519ca7f114ec0e93f82911f99f1cc79414ffc7f70e6dd6714216fe39417.jpg)

![d6afbeb800d245c01194dfd6f04c6538687ba206e477bb9f17cff4147c6ca29d.jpg](../iclr_results/1048_{$_tau$}-bench_ A Benchmark for _underline{T}ool-_underline{A}gent-_underline{U}ser Interaction in R/tables/d6afbeb800d245c01194dfd6f04c6538687ba206e477bb9f17cff4147c6ca29d.jpg)

## Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior


### Images

![007879a861b25e212e20de736cde009040bdc9b5ca75f9349f0e8bf7a7a505ca.jpg](../iclr_results/1049_Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior/images/007879a861b25e212e20de736cde009040bdc9b5ca75f9349f0e8bf7a7a505ca.jpg)

![2e49e099436f996b73ac35706d4d38a56e5cbb26de5a799deb97c43c17125e7e.jpg](../iclr_results/1049_Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior/images/2e49e099436f996b73ac35706d4d38a56e5cbb26de5a799deb97c43c17125e7e.jpg)

![39c40006a6a8f9d4c75f2c6ccbef2196e2a18a43908fa19575b6ac350a56f58e.jpg](../iclr_results/1049_Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior/images/39c40006a6a8f9d4c75f2c6ccbef2196e2a18a43908fa19575b6ac350a56f58e.jpg)

![4c9625d1612e747b96c6327202977d6cf89cd3a376e8b9d4f379462a330d3c61.jpg](../iclr_results/1049_Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior/images/4c9625d1612e747b96c6327202977d6cf89cd3a376e8b9d4f379462a330d3c61.jpg)

![d532b6fef6be0e52b56e0a2f59fb7a8b27b60bc421f2b49300e1dfc1aaf7e151.jpg](../iclr_results/1049_Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior/images/d532b6fef6be0e52b56e0a2f59fb7a8b27b60bc421f2b49300e1dfc1aaf7e151.jpg)

![d8ca1df713156a444377bf54d8603c7b2fac28b978a0c8c70650556a519656c1.jpg](../iclr_results/1049_Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior/images/d8ca1df713156a444377bf54d8603c7b2fac28b978a0c8c70650556a519656c1.jpg)

![da8f218e480de144f400933426de89834a8b35f41224c315d1c331a4814ac565.jpg](../iclr_results/1049_Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior/images/da8f218e480de144f400933426de89834a8b35f41224c315d1c331a4814ac565.jpg)

![e0fd28597149b8cd610a79ce70ec02eb7d39acddbd321d79cc9096783fa6dd13.jpg](../iclr_results/1049_Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior/images/e0fd28597149b8cd610a79ce70ec02eb7d39acddbd321d79cc9096783fa6dd13.jpg)

### Tables

![35f0270f449b2bc42b1101692cf63bada2fb732f0fee6e0ddda1cf469a985d77.jpg](../iclr_results/1049_Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior/tables/35f0270f449b2bc42b1101692cf63bada2fb732f0fee6e0ddda1cf469a985d77.jpg)

![4b317351e0177b0ea42840165256cbf26aac9e509a4d7e51d2a45c2073887ea8.jpg](../iclr_results/1049_Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior/tables/4b317351e0177b0ea42840165256cbf26aac9e509a4d7e51d2a45c2073887ea8.jpg)

![63b354a9122b8053d3e3d5becbe4f2d74882d2e63478d268dce992619e90eff6.jpg](../iclr_results/1049_Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior/tables/63b354a9122b8053d3e3d5becbe4f2d74882d2e63478d268dce992619e90eff6.jpg)

![7236126b4e0a23362a8595aba766de7feaa54604ff553547baba712772d277ed.jpg](../iclr_results/1049_Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior/tables/7236126b4e0a23362a8595aba766de7feaa54604ff553547baba712772d277ed.jpg)

![7994d547f08f4b829e6543f77055ee5f719cbfb1dbba16ffa029c532c64720d7.jpg](../iclr_results/1049_Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior/tables/7994d547f08f4b829e6543f77055ee5f719cbfb1dbba16ffa029c532c64720d7.jpg)

![878d2731b9d0200ea589fc860534e414295e4e530fe8a483af1e6310781bd848.jpg](../iclr_results/1049_Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior/tables/878d2731b9d0200ea589fc860534e414295e4e530fe8a483af1e6310781bd848.jpg)

![a610c0c2f9ecd0afa01d209ad7723ccd3b127167bc1a91f4fa91a09db9dc6b43.jpg](../iclr_results/1049_Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior/tables/a610c0c2f9ecd0afa01d209ad7723ccd3b127167bc1a91f4fa91a09db9dc6b43.jpg)

![ae39dafc52a596218cdd18670be994482be4c2439c7f0ff1285ad62769716df4.jpg](../iclr_results/1049_Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior/tables/ae39dafc52a596218cdd18670be994482be4c2439c7f0ff1285ad62769716df4.jpg)

![b42f8155937dd78caee7017d87a8f0a259bb232ddb39aeb1954f655df970a493.jpg](../iclr_results/1049_Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior/tables/b42f8155937dd78caee7017d87a8f0a259bb232ddb39aeb1954f655df970a493.jpg)

![bda584a04c1577d52d6092fb548bbb19cb9f84062f8cdaff5148f936f2653a0b.jpg](../iclr_results/1049_Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior/tables/bda584a04c1577d52d6092fb548bbb19cb9f84062f8cdaff5148f936f2653a0b.jpg)

![c12012def844cf323f85baa90383e66527822d1099734022db94bc89ea93689d.jpg](../iclr_results/1049_Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior/tables/c12012def844cf323f85baa90383e66527822d1099734022db94bc89ea93689d.jpg)

![cf584f42321c678b9460444c44fd8087dee5e563e77dd8fa009589321ee46890.jpg](../iclr_results/1049_Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior/tables/cf584f42321c678b9460444c44fd8087dee5e563e77dd8fa009589321ee46890.jpg)

![d85a23e920d044632beccdee9f9f3c243e010df2b2cda727bb111795e0831738.jpg](../iclr_results/1049_Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior/tables/d85a23e920d044632beccdee9f9f3c243e010df2b2cda727bb111795e0831738.jpg)

![f2debe698d255b76306a28241fbb78bac55427c84536c0bf803cf2073e098248.jpg](../iclr_results/1049_Bayesian Image Regression with Soft-thresholded Conditional Autoregressive Prior/tables/f2debe698d255b76306a28241fbb78bac55427c84536c0bf803cf2073e098248.jpg)

## Beyond Mere Token Analysis: A Hypergraph Metric Space Framework for Defending Against Socially Engineered LLM Attacks


### Images

![09485bb24a5c22ec0410b04be220967554dff33a7b3643f357c8bbf6336f6bfa.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/09485bb24a5c22ec0410b04be220967554dff33a7b3643f357c8bbf6336f6bfa.jpg)

![0fbc8cc48cf0327a7497522cdd08ae3bc25ac91e4db75366b71f8b0d0b5be2b1.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/0fbc8cc48cf0327a7497522cdd08ae3bc25ac91e4db75366b71f8b0d0b5be2b1.jpg)

![216b8914c4dd5066eeeb11233c05d6dffe6d3fa544bae0efe5b6b16a48f56027.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/216b8914c4dd5066eeeb11233c05d6dffe6d3fa544bae0efe5b6b16a48f56027.jpg)

![3abb5042a765e1f448dea9912583a0ba1906642db327061c5e785b50cc08ec14.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/3abb5042a765e1f448dea9912583a0ba1906642db327061c5e785b50cc08ec14.jpg)

![413cfb1672dc90e7311cea443fd7fcc6d1c47dcff61e3de504cdf006e7dbde1f.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/413cfb1672dc90e7311cea443fd7fcc6d1c47dcff61e3de504cdf006e7dbde1f.jpg)

![4398849e706d33e4c81f4fdc6ef8a731b75fa83803b8b05e026dbf962c11846c.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/4398849e706d33e4c81f4fdc6ef8a731b75fa83803b8b05e026dbf962c11846c.jpg)

![46524629519a4dcb8c53063f459e448b7f26e82946cf9750062b94c68a53e306.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/46524629519a4dcb8c53063f459e448b7f26e82946cf9750062b94c68a53e306.jpg)

![49b52ddffcc815a343a878211cc36b85b01402a67c63a9eac2da309b820f1b37.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/49b52ddffcc815a343a878211cc36b85b01402a67c63a9eac2da309b820f1b37.jpg)

![4b02f23690c4da97c41914f124b1f7dfc1bd17c79aac4b88ed5693504adaaf7d.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/4b02f23690c4da97c41914f124b1f7dfc1bd17c79aac4b88ed5693504adaaf7d.jpg)

![5288b3c91eaf3d7146bb0ea3f8d1f0211fe4a6a7f1955fc3783a8e159baa25c7.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/5288b3c91eaf3d7146bb0ea3f8d1f0211fe4a6a7f1955fc3783a8e159baa25c7.jpg)

![55427fad0c912c2b4a95cd764dc9d23c5a335484c3adbfae545bfc025ff514e2.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/55427fad0c912c2b4a95cd764dc9d23c5a335484c3adbfae545bfc025ff514e2.jpg)

![5e2ea85912a5a4b0bee27ba27c0e2db3a128af593641fbfc4475cda58ae0a435.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/5e2ea85912a5a4b0bee27ba27c0e2db3a128af593641fbfc4475cda58ae0a435.jpg)

![648c7e74aab80302f8f0158f45a2c325f4ead50876e8390994ff254b291dcce5.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/648c7e74aab80302f8f0158f45a2c325f4ead50876e8390994ff254b291dcce5.jpg)

![659d0b1457ad6e04445a801176ceffdbdf119e0b9b090c0e93c610212e042817.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/659d0b1457ad6e04445a801176ceffdbdf119e0b9b090c0e93c610212e042817.jpg)

![6780d763853fff13c5f55c6a2b7ff5bac21af92ad353af4ce60d009072e6fee2.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/6780d763853fff13c5f55c6a2b7ff5bac21af92ad353af4ce60d009072e6fee2.jpg)

![7c696fc0b071bd4ce080a37cb8cd3bed72cb886e771c3435104417d3aa2e16ca.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/7c696fc0b071bd4ce080a37cb8cd3bed72cb886e771c3435104417d3aa2e16ca.jpg)

![897134130bc117196c480bf6766f78024fa5e69ebe0b92052d9c911a67e524f4.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/897134130bc117196c480bf6766f78024fa5e69ebe0b92052d9c911a67e524f4.jpg)

![8cc35023b07f8843280a7d3be98c0c7f9c71c766779a827659026a033c97ade4.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/8cc35023b07f8843280a7d3be98c0c7f9c71c766779a827659026a033c97ade4.jpg)

![8d2fee2298405c0f6389b4b6f7abb5eeed73f895831e70101288e3c830cc7aab.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/8d2fee2298405c0f6389b4b6f7abb5eeed73f895831e70101288e3c830cc7aab.jpg)

![90725661a59f0ceb95233e0c2d4e42ca16801335530d61c77fa776ed5ad4af93.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/90725661a59f0ceb95233e0c2d4e42ca16801335530d61c77fa776ed5ad4af93.jpg)

![9b1988c9146af5cb0156afd4e55ee115dd0fc1fc0a391d0dc50c462a133d81f2.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/9b1988c9146af5cb0156afd4e55ee115dd0fc1fc0a391d0dc50c462a133d81f2.jpg)

![a597ffcc77886dfe6a532ba8909d95a4c09d2081b8a9b1c4de85b1db58cfd0dc.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/a597ffcc77886dfe6a532ba8909d95a4c09d2081b8a9b1c4de85b1db58cfd0dc.jpg)

![cbd435241e52ee19af694b76e835921cbad9e211315c1d38cc67001964dec14c.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/cbd435241e52ee19af694b76e835921cbad9e211315c1d38cc67001964dec14c.jpg)

![cd598d0d92536aae885f81283069d959d06c4560e899fd870ddc96af8e70906e.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/cd598d0d92536aae885f81283069d959d06c4560e899fd870ddc96af8e70906e.jpg)

![cec711c73afdec565b02041fcaf67d108c42578a5d906d67ac8fff4e01bfe386.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/cec711c73afdec565b02041fcaf67d108c42578a5d906d67ac8fff4e01bfe386.jpg)

![fd1cd6714595c544743dff98917395a0ae2a05ce02f2d078c5616cceb9be46f1.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/fd1cd6714595c544743dff98917395a0ae2a05ce02f2d078c5616cceb9be46f1.jpg)

![fe2638af03756c959fe27cf1ea6cd8f5533f8a090af078de9c9eef9becc1ce10.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/images/fe2638af03756c959fe27cf1ea6cd8f5533f8a090af078de9c9eef9becc1ce10.jpg)

### Tables

![269eb5ebf9985e015c30f136f9353765b2b7d8de992bfbb365388d36a0df3a89.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/tables/269eb5ebf9985e015c30f136f9353765b2b7d8de992bfbb365388d36a0df3a89.jpg)

![2e81fd5948d05cdfe9aa6d7a0407e91042f8d0a2290a8482eac40bd6958470d0.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/tables/2e81fd5948d05cdfe9aa6d7a0407e91042f8d0a2290a8482eac40bd6958470d0.jpg)

![483588048b159cbe323d1cc9eebf732470df47ef06ecf665bb253c6ca33b8c5a.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/tables/483588048b159cbe323d1cc9eebf732470df47ef06ecf665bb253c6ca33b8c5a.jpg)

![6d24a4933fc801b9e640bec5427fc8971b7f452849c594dfdc6153238c0d571b.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/tables/6d24a4933fc801b9e640bec5427fc8971b7f452849c594dfdc6153238c0d571b.jpg)

![887e54a812b70366d05efd9c7627e11738a2bc1d533f86f88c9832d027253083.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/tables/887e54a812b70366d05efd9c7627e11738a2bc1d533f86f88c9832d027253083.jpg)

![9050db4b13f514ef85dcbb4833ab281623434da90704634a91c502333e29ddd3.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/tables/9050db4b13f514ef85dcbb4833ab281623434da90704634a91c502333e29ddd3.jpg)

![b6666f18db881a113c991c2c1a0fe524af3a4a3513f1cb95c1db8eec08acd1ae.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/tables/b6666f18db881a113c991c2c1a0fe524af3a4a3513f1cb95c1db8eec08acd1ae.jpg)

![eb2f7158ff004a4bc871594e3ab351258d0b17bd3628bf3c0fadb9e04e45453b.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/tables/eb2f7158ff004a4bc871594e3ab351258d0b17bd3628bf3c0fadb9e04e45453b.jpg)

![ec120680e066d69d97032d67f818356570ecb0a0284a2c38b130b44bb89a4124.jpg](../iclr_results/1050_Beyond Mere Token Analysis_ A Hypergraph Metric Space Framework for Defending Against Socially Engin/tables/ec120680e066d69d97032d67f818356570ecb0a0284a2c38b130b44bb89a4124.jpg)

## OCEAN: Offline Chain-of-thought Evaluation and Alignment in Large Language Models


### Images

![0b5f823f946e883cfd340cd7a6110dc59214fa4803f04facb0d4353b502dfceb.jpg](../iclr_results/1051_OCEAN_ Offline Chain-of-thought Evaluation and Alignment in Large Language Models/images/0b5f823f946e883cfd340cd7a6110dc59214fa4803f04facb0d4353b502dfceb.jpg)

![6bc4d8166b345d7705f3cb791974472d30f79627c41374d63e70af527bd2e589.jpg](../iclr_results/1051_OCEAN_ Offline Chain-of-thought Evaluation and Alignment in Large Language Models/images/6bc4d8166b345d7705f3cb791974472d30f79627c41374d63e70af527bd2e589.jpg)

![75ede83f53d65f1697a58921f7c53956fdf79e517607a16d9bee004d16b2016a.jpg](../iclr_results/1051_OCEAN_ Offline Chain-of-thought Evaluation and Alignment in Large Language Models/images/75ede83f53d65f1697a58921f7c53956fdf79e517607a16d9bee004d16b2016a.jpg)

![9d7d5256dd24329274442ab29a1b64fefaee22d4008835219abe843435500c1b.jpg](../iclr_results/1051_OCEAN_ Offline Chain-of-thought Evaluation and Alignment in Large Language Models/images/9d7d5256dd24329274442ab29a1b64fefaee22d4008835219abe843435500c1b.jpg)

### Tables

![4053f7d09b4cdec34264cefe78e4d86cc17226957efd0b2ccc7c949cca0abb05.jpg](../iclr_results/1051_OCEAN_ Offline Chain-of-thought Evaluation and Alignment in Large Language Models/tables/4053f7d09b4cdec34264cefe78e4d86cc17226957efd0b2ccc7c949cca0abb05.jpg)

![4bba2f8b7a7357863032a5f4dd6d346ea9d9032cd484d6e6f6b7c1007c1e67e0.jpg](../iclr_results/1051_OCEAN_ Offline Chain-of-thought Evaluation and Alignment in Large Language Models/tables/4bba2f8b7a7357863032a5f4dd6d346ea9d9032cd484d6e6f6b7c1007c1e67e0.jpg)

![abc785adffa06dc7868fdd15bc4b244d1fe0434c170aadbe3d18fe211723633f.jpg](../iclr_results/1051_OCEAN_ Offline Chain-of-thought Evaluation and Alignment in Large Language Models/tables/abc785adffa06dc7868fdd15bc4b244d1fe0434c170aadbe3d18fe211723633f.jpg)

![e334548d68d38d1e49ca44f24ba0ca59956efe6c3c7023d16f7395f34562a6ce.jpg](../iclr_results/1051_OCEAN_ Offline Chain-of-thought Evaluation and Alignment in Large Language Models/tables/e334548d68d38d1e49ca44f24ba0ca59956efe6c3c7023d16f7395f34562a6ce.jpg)

## Hallo2: Long-Duration and High-Resolution Audio-Driven Portrait Image Animation


### Images

![418374e5d83a057973d8838ecf02df6c6b674c07c35221fc3cda6d22118441cd.jpg](../iclr_results/1052_Hallo2_ Long-Duration and High-Resolution Audio-Driven Portrait Image Animation/images/418374e5d83a057973d8838ecf02df6c6b674c07c35221fc3cda6d22118441cd.jpg)

![635cbf816c5ef4bceb202b09f3fe307d10340e294e258507b9adb396377f012e.jpg](../iclr_results/1052_Hallo2_ Long-Duration and High-Resolution Audio-Driven Portrait Image Animation/images/635cbf816c5ef4bceb202b09f3fe307d10340e294e258507b9adb396377f012e.jpg)

![720e2c4515ff53fe7cd3dd56a28ef1896c55210c2d5ba3f83d63e0b6a022d2ce.jpg](../iclr_results/1052_Hallo2_ Long-Duration and High-Resolution Audio-Driven Portrait Image Animation/images/720e2c4515ff53fe7cd3dd56a28ef1896c55210c2d5ba3f83d63e0b6a022d2ce.jpg)

![78233a67c292336d4ecca03984a4ac873bec07de4316c37f95ddce468627d012.jpg](../iclr_results/1052_Hallo2_ Long-Duration and High-Resolution Audio-Driven Portrait Image Animation/images/78233a67c292336d4ecca03984a4ac873bec07de4316c37f95ddce468627d012.jpg)

![a2ebf74f6a675d9399b03a8ebb078f003cb61f14b81f2fd47a06f5367747b170.jpg](../iclr_results/1052_Hallo2_ Long-Duration and High-Resolution Audio-Driven Portrait Image Animation/images/a2ebf74f6a675d9399b03a8ebb078f003cb61f14b81f2fd47a06f5367747b170.jpg)

![aaaf2ffbeb1e39b8c9679d451ff8a0ccf4e60d22570b403a83c1ee0cb7c33c9b.jpg](../iclr_results/1052_Hallo2_ Long-Duration and High-Resolution Audio-Driven Portrait Image Animation/images/aaaf2ffbeb1e39b8c9679d451ff8a0ccf4e60d22570b403a83c1ee0cb7c33c9b.jpg)

![cfd992fc37f4f8e18e0b4960d3b1f870bcdb8f613298209760f2dd7dc56a189d.jpg](../iclr_results/1052_Hallo2_ Long-Duration and High-Resolution Audio-Driven Portrait Image Animation/images/cfd992fc37f4f8e18e0b4960d3b1f870bcdb8f613298209760f2dd7dc56a189d.jpg)

![d1f63989ff16a4cc6cc3162a1e493f228f0a95a423fe461be8a8bc2f29e8547f.jpg](../iclr_results/1052_Hallo2_ Long-Duration and High-Resolution Audio-Driven Portrait Image Animation/images/d1f63989ff16a4cc6cc3162a1e493f228f0a95a423fe461be8a8bc2f29e8547f.jpg)

![d58615071835d7502d090c377bb435e269f7c0ae91bf73a9312ae65070bd9017.jpg](../iclr_results/1052_Hallo2_ Long-Duration and High-Resolution Audio-Driven Portrait Image Animation/images/d58615071835d7502d090c377bb435e269f7c0ae91bf73a9312ae65070bd9017.jpg)

![f94b1e2fdc150192593fc40eced2afc82f7e605f114fb71a2b220029023735aa.jpg](../iclr_results/1052_Hallo2_ Long-Duration and High-Resolution Audio-Driven Portrait Image Animation/images/f94b1e2fdc150192593fc40eced2afc82f7e605f114fb71a2b220029023735aa.jpg)

![fb18517647bd3c4620e3aefa8cad81e7450219f8f02f06bb78e9505b6152adb3.jpg](../iclr_results/1052_Hallo2_ Long-Duration and High-Resolution Audio-Driven Portrait Image Animation/images/fb18517647bd3c4620e3aefa8cad81e7450219f8f02f06bb78e9505b6152adb3.jpg)

### Tables

![21503826b35ae3aae81011489b6e12cd969fa172df558652e362533df3d0f00d.jpg](../iclr_results/1052_Hallo2_ Long-Duration and High-Resolution Audio-Driven Portrait Image Animation/tables/21503826b35ae3aae81011489b6e12cd969fa172df558652e362533df3d0f00d.jpg)

![3736ee3f5e2159b7cb856168fa1b1efdc24848355694e7a7f400679e72a29a3e.jpg](../iclr_results/1052_Hallo2_ Long-Duration and High-Resolution Audio-Driven Portrait Image Animation/tables/3736ee3f5e2159b7cb856168fa1b1efdc24848355694e7a7f400679e72a29a3e.jpg)

![511639955bf7da6e3b30678983523c63aa66a344f31f79e77ea67325aff37d07.jpg](../iclr_results/1052_Hallo2_ Long-Duration and High-Resolution Audio-Driven Portrait Image Animation/tables/511639955bf7da6e3b30678983523c63aa66a344f31f79e77ea67325aff37d07.jpg)

![80814dbbca317e4084edc437f382a511e04122d12c9d8254ad3f3f24bbb0e1dd.jpg](../iclr_results/1052_Hallo2_ Long-Duration and High-Resolution Audio-Driven Portrait Image Animation/tables/80814dbbca317e4084edc437f382a511e04122d12c9d8254ad3f3f24bbb0e1dd.jpg)

![846563ac57b9090ab27094481f0e3d7c680bb8f31633e6e628e9454e30c14d49.jpg](../iclr_results/1052_Hallo2_ Long-Duration and High-Resolution Audio-Driven Portrait Image Animation/tables/846563ac57b9090ab27094481f0e3d7c680bb8f31633e6e628e9454e30c14d49.jpg)

![a8e0dd2710c721a63eea3b3150eef94c880b557ba326864b9e40fee8fb4e6fc8.jpg](../iclr_results/1052_Hallo2_ Long-Duration and High-Resolution Audio-Driven Portrait Image Animation/tables/a8e0dd2710c721a63eea3b3150eef94c880b557ba326864b9e40fee8fb4e6fc8.jpg)

## Learning system dynamics without forgetting


### Images

![2c8324934ad911daaea1f815220ec220d0fdb09f4839fb4a09b8e66de94c6239.jpg](../iclr_results/1053_Learning system dynamics without forgetting/images/2c8324934ad911daaea1f815220ec220d0fdb09f4839fb4a09b8e66de94c6239.jpg)

![50354eab83a0bd049f2794556e856ae428e9abb5ef94d57cd72542f196a44f56.jpg](../iclr_results/1053_Learning system dynamics without forgetting/images/50354eab83a0bd049f2794556e856ae428e9abb5ef94d57cd72542f196a44f56.jpg)

![6628fbea2e4d9fb6a4eb9f3aa49e826b325ef8262be365e3dcf6d994b8e7b03a.jpg](../iclr_results/1053_Learning system dynamics without forgetting/images/6628fbea2e4d9fb6a4eb9f3aa49e826b325ef8262be365e3dcf6d994b8e7b03a.jpg)

![8ca174598fb5e7da9f2c797f84b29809302dcee37e2454e65d6b9ae05d0ee01d.jpg](../iclr_results/1053_Learning system dynamics without forgetting/images/8ca174598fb5e7da9f2c797f84b29809302dcee37e2454e65d6b9ae05d0ee01d.jpg)

![ad161de0914b3f46773c3c5fdf96a4f1abd9d9424b0388c6467fe86d1ef19d15.jpg](../iclr_results/1053_Learning system dynamics without forgetting/images/ad161de0914b3f46773c3c5fdf96a4f1abd9d9424b0388c6467fe86d1ef19d15.jpg)

![c26d269633d6bd929fe4d4a2ae8f8513ec8e652a15455420d3b5fc71270f3267.jpg](../iclr_results/1053_Learning system dynamics without forgetting/images/c26d269633d6bd929fe4d4a2ae8f8513ec8e652a15455420d3b5fc71270f3267.jpg)

![c696a7a2c0e3055f024500a20baf8ed3ec2d1484318c4689666a746097b3fd95.jpg](../iclr_results/1053_Learning system dynamics without forgetting/images/c696a7a2c0e3055f024500a20baf8ed3ec2d1484318c4689666a746097b3fd95.jpg)

![e70dd9d8af7bce01750460c500a816f5755de209bc5d9529216ffdb63ed7dfbb.jpg](../iclr_results/1053_Learning system dynamics without forgetting/images/e70dd9d8af7bce01750460c500a816f5755de209bc5d9529216ffdb63ed7dfbb.jpg)

![e8ae15437f11c23359b34f52a57ad958080aee85a261d7b9345fd1e99efdae7d.jpg](../iclr_results/1053_Learning system dynamics without forgetting/images/e8ae15437f11c23359b34f52a57ad958080aee85a261d7b9345fd1e99efdae7d.jpg)

![ed387c879134f40ebbd9c0753b1448f2a5cea0664238fea700b024968f7807fa.jpg](../iclr_results/1053_Learning system dynamics without forgetting/images/ed387c879134f40ebbd9c0753b1448f2a5cea0664238fea700b024968f7807fa.jpg)

### Tables

![8c932207e6bf5349fb1e35bf934e9c380328b8d424059061fa4e967b02ea709b.jpg](../iclr_results/1053_Learning system dynamics without forgetting/tables/8c932207e6bf5349fb1e35bf934e9c380328b8d424059061fa4e967b02ea709b.jpg)

![bff060a535ddc74501b414cc53ebabb98cb069804556524df09501c0fe49f5bd.jpg](../iclr_results/1053_Learning system dynamics without forgetting/tables/bff060a535ddc74501b414cc53ebabb98cb069804556524df09501c0fe49f5bd.jpg)

![d9d12c83327c4eeba65cf5fa7e13d071613c962e5b0fae6accb757cc9df65e02.jpg](../iclr_results/1053_Learning system dynamics without forgetting/tables/d9d12c83327c4eeba65cf5fa7e13d071613c962e5b0fae6accb757cc9df65e02.jpg)

![e1fa0a5ed392a962e474aba8fa2c0588f48d441ebb822b3e8b6edb87e5d93e75.jpg](../iclr_results/1053_Learning system dynamics without forgetting/tables/e1fa0a5ed392a962e474aba8fa2c0588f48d441ebb822b3e8b6edb87e5d93e75.jpg)

![f5ba19c27c6eadc6486187bdffb52ccdb02303b6bddba17e51ee781c4994b32c.jpg](../iclr_results/1053_Learning system dynamics without forgetting/tables/f5ba19c27c6eadc6486187bdffb52ccdb02303b6bddba17e51ee781c4994b32c.jpg)

## UIFace: Unleashing Inherent Model Capabilities to Enhance Intra-Class Diversity in Synthetic Face Recognition


### Images

![2edcd261d71dc17034e944d3bd85dde02e3350a150fc8630ed841ca88f587250.jpg](../iclr_results/1054_UIFace_ Unleashing Inherent Model Capabilities to Enhance Intra-Class Diversity in Synthetic Face Re/images/2edcd261d71dc17034e944d3bd85dde02e3350a150fc8630ed841ca88f587250.jpg)

![3a0b7a6d2dcddb3c36e51e366e43e66b1c0c10cfbaa086d17865375284ee9221.jpg](../iclr_results/1054_UIFace_ Unleashing Inherent Model Capabilities to Enhance Intra-Class Diversity in Synthetic Face Re/images/3a0b7a6d2dcddb3c36e51e366e43e66b1c0c10cfbaa086d17865375284ee9221.jpg)

![5d5640778586234cb4595f56b670f293e859f6af95e2100801e53b40c98dbbf0.jpg](../iclr_results/1054_UIFace_ Unleashing Inherent Model Capabilities to Enhance Intra-Class Diversity in Synthetic Face Re/images/5d5640778586234cb4595f56b670f293e859f6af95e2100801e53b40c98dbbf0.jpg)

![6bd02882cb9ac3df9ee5e78a81092fae353163991298f8d11ab29805bb63084d.jpg](../iclr_results/1054_UIFace_ Unleashing Inherent Model Capabilities to Enhance Intra-Class Diversity in Synthetic Face Re/images/6bd02882cb9ac3df9ee5e78a81092fae353163991298f8d11ab29805bb63084d.jpg)

![87021c6293b0d01f0da12456a90c3dd843ee0b1ad5e616547c11f2647580ff08.jpg](../iclr_results/1054_UIFace_ Unleashing Inherent Model Capabilities to Enhance Intra-Class Diversity in Synthetic Face Re/images/87021c6293b0d01f0da12456a90c3dd843ee0b1ad5e616547c11f2647580ff08.jpg)

![bf59bece51985da1ea6ce9b6199115c466aae54051ce00582b7e68eb1c62b7b0.jpg](../iclr_results/1054_UIFace_ Unleashing Inherent Model Capabilities to Enhance Intra-Class Diversity in Synthetic Face Re/images/bf59bece51985da1ea6ce9b6199115c466aae54051ce00582b7e68eb1c62b7b0.jpg)

### Tables

![2f86a8480b6de10c3e22d1d85fead83524cff39579e68d8ce4744512bf86548c.jpg](../iclr_results/1054_UIFace_ Unleashing Inherent Model Capabilities to Enhance Intra-Class Diversity in Synthetic Face Re/tables/2f86a8480b6de10c3e22d1d85fead83524cff39579e68d8ce4744512bf86548c.jpg)

![3ed131df05ae8bc67df918afd5c96648153e39f4d94eedad53e793dbeb4d4371.jpg](../iclr_results/1054_UIFace_ Unleashing Inherent Model Capabilities to Enhance Intra-Class Diversity in Synthetic Face Re/tables/3ed131df05ae8bc67df918afd5c96648153e39f4d94eedad53e793dbeb4d4371.jpg)

![e66617beb5f62f27881bbaabb67aba261a0516ad6c5a12f13bff64a53b4131b0.jpg](../iclr_results/1054_UIFace_ Unleashing Inherent Model Capabilities to Enhance Intra-Class Diversity in Synthetic Face Re/tables/e66617beb5f62f27881bbaabb67aba261a0516ad6c5a12f13bff64a53b4131b0.jpg)

## SWE-bench Multimodal: Do AI Systems Generalize to Visual Software Domains?


### Images

![0b9caaed554d4356778155586ed5caf469f8c78e7fa4a1e4a938ec4c69e43d97.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/images/0b9caaed554d4356778155586ed5caf469f8c78e7fa4a1e4a938ec4c69e43d97.jpg)

![0ea1ec948d578174951be0ebf75450dd9cbfd5ac52c72843a379bd2da3fe2ab9.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/images/0ea1ec948d578174951be0ebf75450dd9cbfd5ac52c72843a379bd2da3fe2ab9.jpg)

![1193972509bb7a4afad01052dcd2251372cd2dcd90772839b44f4eb8ba1305c3.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/images/1193972509bb7a4afad01052dcd2251372cd2dcd90772839b44f4eb8ba1305c3.jpg)

![16205868cac7032b8643e4b2ce476690ef13c24292adaaac802cfe152c69e2e1.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/images/16205868cac7032b8643e4b2ce476690ef13c24292adaaac802cfe152c69e2e1.jpg)

![3179f14dfb037603cc91dade63dc4d08f9d014662c02234ac3930f4164f05cc2.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/images/3179f14dfb037603cc91dade63dc4d08f9d014662c02234ac3930f4164f05cc2.jpg)

![6f1146036300baa4f36e781cb1fb1c135126f6704a71c996d44baa7222981d4f.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/images/6f1146036300baa4f36e781cb1fb1c135126f6704a71c996d44baa7222981d4f.jpg)

![793a703dc76e7d08b5ca343d0931a75eec289712b94d91eb0a2807992b9a2674.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/images/793a703dc76e7d08b5ca343d0931a75eec289712b94d91eb0a2807992b9a2674.jpg)

![a466473fba88495f84e3da3e1fb21828b68462eb52730ece2c3a7af4fd531ad6.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/images/a466473fba88495f84e3da3e1fb21828b68462eb52730ece2c3a7af4fd531ad6.jpg)

![b5e9b32e0bf1131c41d585def731f91ac2362943f35362c4ef2399533fdc7a82.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/images/b5e9b32e0bf1131c41d585def731f91ac2362943f35362c4ef2399533fdc7a82.jpg)

![c0b7c666d051281f7a1837d86c7b6ee26f072eb271e2ee3a9a87708506939fa7.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/images/c0b7c666d051281f7a1837d86c7b6ee26f072eb271e2ee3a9a87708506939fa7.jpg)

![c2efa4c8c02b76fe6d2e2cf5c7a83620c8010bb290850e0e93167413a2d1fbc2.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/images/c2efa4c8c02b76fe6d2e2cf5c7a83620c8010bb290850e0e93167413a2d1fbc2.jpg)

![cc3ad89265842675f9dc49504fe8897e6d5744a464f2152e5d0f04072e9e4349.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/images/cc3ad89265842675f9dc49504fe8897e6d5744a464f2152e5d0f04072e9e4349.jpg)

![d3d1218d3303d212cf9eb6c29e11c34ae4a3751e865ad2b46965b4b20e290fa6.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/images/d3d1218d3303d212cf9eb6c29e11c34ae4a3751e865ad2b46965b4b20e290fa6.jpg)

![def5ba213c7d690e711a51e4045aeefdd2161f13bf44d633cc67c9c9446d5561.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/images/def5ba213c7d690e711a51e4045aeefdd2161f13bf44d633cc67c9c9446d5561.jpg)

### Tables

![13994f9c55f3d0e47c78d58a689decba85f158304bcdfc73b83410a1312601de.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/tables/13994f9c55f3d0e47c78d58a689decba85f158304bcdfc73b83410a1312601de.jpg)

![25649cced1d2dfef54aa12349952be38e54b4fa14f3f019d0ee026db82c261db.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/tables/25649cced1d2dfef54aa12349952be38e54b4fa14f3f019d0ee026db82c261db.jpg)

![2c28e793436a8ae83b5f5a9581e62f4bcada464ecde0e117b9ed8b6c9e8d13ef.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/tables/2c28e793436a8ae83b5f5a9581e62f4bcada464ecde0e117b9ed8b6c9e8d13ef.jpg)

![3e9bdae2e108c87477e1dbe5f51de5d1171e97bce73d6a717596d7dc3ac500e6.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/tables/3e9bdae2e108c87477e1dbe5f51de5d1171e97bce73d6a717596d7dc3ac500e6.jpg)

![45002b99e4332c902ef804f69ef7b26009c2d6d15b0f29f28188364e1ac60ab0.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/tables/45002b99e4332c902ef804f69ef7b26009c2d6d15b0f29f28188364e1ac60ab0.jpg)

![55cf9cb767260408169892699dcf47c13d1c14e418aeff377a0071d0b63dca24.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/tables/55cf9cb767260408169892699dcf47c13d1c14e418aeff377a0071d0b63dca24.jpg)

![59382fd875428b9f89903a5d60fd7102fecf936b1d97e6fe9059c2f2de7ee42c.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/tables/59382fd875428b9f89903a5d60fd7102fecf936b1d97e6fe9059c2f2de7ee42c.jpg)

![63120bfdb3815087b3998833e77a692133b75f196e319d01f8e4b78836d2daa7.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/tables/63120bfdb3815087b3998833e77a692133b75f196e319d01f8e4b78836d2daa7.jpg)

![79d6e078ee2563a07f950ab6a398cfe5897dd1b8ef4e6df8940c7ab07212b7d7.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/tables/79d6e078ee2563a07f950ab6a398cfe5897dd1b8ef4e6df8940c7ab07212b7d7.jpg)

![8704acad74246122282a53e55d82df0bb028370b76c04380dd00ce60d2817954.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/tables/8704acad74246122282a53e55d82df0bb028370b76c04380dd00ce60d2817954.jpg)

![9287cb5dec78f95e55d672384b386ca64f57b19046ab2d61afa5d5b1b5e8abe5.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/tables/9287cb5dec78f95e55d672384b386ca64f57b19046ab2d61afa5d5b1b5e8abe5.jpg)

![a5bdfa27cf2257a19fc6abbeab7a8faf545af9351810a1f959c9d847167eb1b3.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/tables/a5bdfa27cf2257a19fc6abbeab7a8faf545af9351810a1f959c9d847167eb1b3.jpg)

![b2d9fdc27d3d42f901593d9245ed32d8761da2452d9d24a25ce7d265b1df0fcc.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/tables/b2d9fdc27d3d42f901593d9245ed32d8761da2452d9d24a25ce7d265b1df0fcc.jpg)

![b6e53487a5f51f071117b6d376838ca2ce1837b8aaf01a60888dae6b1e6e182d.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/tables/b6e53487a5f51f071117b6d376838ca2ce1837b8aaf01a60888dae6b1e6e182d.jpg)

![cb3ec52c535611817c1bebc398f0e7d90f0f05295702a3978fd15a201f7d74ac.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/tables/cb3ec52c535611817c1bebc398f0e7d90f0f05295702a3978fd15a201f7d74ac.jpg)

![d8a95afb9fbb87ac649eaabc8af608c540643db671e6148c63481dbb19149c71.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/tables/d8a95afb9fbb87ac649eaabc8af608c540643db671e6148c63481dbb19149c71.jpg)

![e44d6c283ed214803b60b4b727054565d93c825b4b71ee102bac356ffc32438a.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/tables/e44d6c283ed214803b60b4b727054565d93c825b4b71ee102bac356ffc32438a.jpg)

![e687dc59e074123b1f28b6ecf807f59fc6fb35016b6f1f7ea1720a4bd0fdc313.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/tables/e687dc59e074123b1f28b6ecf807f59fc6fb35016b6f1f7ea1720a4bd0fdc313.jpg)

![e8b5cf7a3302c6c27d72cea252701b1067e75b9488dfe806109fbeba1ea643d1.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/tables/e8b5cf7a3302c6c27d72cea252701b1067e75b9488dfe806109fbeba1ea643d1.jpg)

![e9ed724104ce5ea850d15629e619890d0b2d5f1d6819e60b7fa775f5de35b4ba.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/tables/e9ed724104ce5ea850d15629e619890d0b2d5f1d6819e60b7fa775f5de35b4ba.jpg)

![ebc8e9831d07280efbd54781421f51e4ea4aeae391f83feeacf6ed93aec24da9.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/tables/ebc8e9831d07280efbd54781421f51e4ea4aeae391f83feeacf6ed93aec24da9.jpg)

![f455f1c5393c25bce363163650c7e835619ae279fd0810f8215fd25047a8ec61.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/tables/f455f1c5393c25bce363163650c7e835619ae279fd0810f8215fd25047a8ec61.jpg)

![fc93f6cdcdd046d89bc193d7d858954c9da12d80a8a3bae8d7fa224b21273bec.jpg](../iclr_results/1055_SWE-bench Multimodal_ Do AI Systems Generalize to Visual Software Domains_/tables/fc93f6cdcdd046d89bc193d7d858954c9da12d80a8a3bae8d7fa224b21273bec.jpg)

## Optimal Protocols for Continual Learning via Statistical Physics and Control Theory

### Images

![068809097617bdf2c1435da54269e3a312f760f5b8dd3de2f0ab3060d15ec2e6.jpg](../iclr_results/1056_Optimal Protocols for Continual Learning via Statistical Physics and Control Theory/images/068809097617bdf2c1435da54269e3a312f760f5b8dd3de2f0ab3060d15ec2e6.jpg)

![1781e283c9a226997a59badf3fc22ae97c093925e8de89cf7d8654d8f38d8281.jpg](../iclr_results/1056_Optimal Protocols for Continual Learning via Statistical Physics and Control Theory/images/1781e283c9a226997a59badf3fc22ae97c093925e8de89cf7d8654d8f38d8281.jpg)

![1f66e69fdc0243aa937caf80b5382e949405da89fd40e172df17c802c3991955.jpg](../iclr_results/1056_Optimal Protocols for Continual Learning via Statistical Physics and Control Theory/images/1f66e69fdc0243aa937caf80b5382e949405da89fd40e172df17c802c3991955.jpg)

![2a3b1477c25ec713b91c70def9fff43f3755a27424582e768233ecf525d1c416.jpg](../iclr_results/1056_Optimal Protocols for Continual Learning via Statistical Physics and Control Theory/images/2a3b1477c25ec713b91c70def9fff43f3755a27424582e768233ecf525d1c416.jpg)

![5d31938f7e8f281b961a324535c11ad0b025eb6f3ee1d799010a537ed6052135.jpg](../iclr_results/1056_Optimal Protocols for Continual Learning via Statistical Physics and Control Theory/images/5d31938f7e8f281b961a324535c11ad0b025eb6f3ee1d799010a537ed6052135.jpg)

![66807e3649cf5c22e99cd9bd3617c36bf19045ed68042d5bf526c317ba11e03b.jpg](../iclr_results/1056_Optimal Protocols for Continual Learning via Statistical Physics and Control Theory/images/66807e3649cf5c22e99cd9bd3617c36bf19045ed68042d5bf526c317ba11e03b.jpg)

![75c3d2cc4531a42f0d980e7087f7e60bdb94825512312f713238ff71576d831b.jpg](../iclr_results/1056_Optimal Protocols for Continual Learning via Statistical Physics and Control Theory/images/75c3d2cc4531a42f0d980e7087f7e60bdb94825512312f713238ff71576d831b.jpg)

![7c01c37eddf04cada882d2aac734514225154fc87d10579a613973031b5c0c1f.jpg](../iclr_results/1056_Optimal Protocols for Continual Learning via Statistical Physics and Control Theory/images/7c01c37eddf04cada882d2aac734514225154fc87d10579a613973031b5c0c1f.jpg)

![840970bf0eb761a3e99870de9f7a0bc7a5297ebbeff1f752e757768bd83c6d6e.jpg](../iclr_results/1056_Optimal Protocols for Continual Learning via Statistical Physics and Control Theory/images/840970bf0eb761a3e99870de9f7a0bc7a5297ebbeff1f752e757768bd83c6d6e.jpg)

![946828deb52a397b8430b6c7283a9531ec593fcfcb9b27850cc851cc3cb99e81.jpg](../iclr_results/1056_Optimal Protocols for Continual Learning via Statistical Physics and Control Theory/images/946828deb52a397b8430b6c7283a9531ec593fcfcb9b27850cc851cc3cb99e81.jpg)

![97584b3aeb8150c5824c380d7f2fbb4ddd52876d9d3bc030224cdb48267d2465.jpg](../iclr_results/1056_Optimal Protocols for Continual Learning via Statistical Physics and Control Theory/images/97584b3aeb8150c5824c380d7f2fbb4ddd52876d9d3bc030224cdb48267d2465.jpg)

![9c39d1e9438a73d997904a82e1de00f07d546058cf8e7311751cde11b25d7044.jpg](../iclr_results/1056_Optimal Protocols for Continual Learning via Statistical Physics and Control Theory/images/9c39d1e9438a73d997904a82e1de00f07d546058cf8e7311751cde11b25d7044.jpg)
