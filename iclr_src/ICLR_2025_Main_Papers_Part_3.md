# ICLR 2025 Main Conference Papers

**Summary:** 36 papers with extracted content:
- 📊 Total images: 44031
- 📋 Total tables: 33468
- 📄 Total files: 77499

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 3 of 100

## 目录 (Table of Contents)

1. [Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition](#Knowledge-Entropy-Decay-during-Language-Model-Pretraining-Hinders-New-Knowledge-Acquisition)
2. [Emergence of meta-stable clustering in mean-field transformer models](#Emergence-of-meta-stable-clustering-in-mean-field-transformer-models)
3. [Data Selection via Optimal Control for Language Models](#Data-Selection-via-Optimal-Control-for-Language-Models)
4. [Feedback Favors the Generalization of Neural ODEs](#Feedback-Favors-the-Generalization-of-Neural-ODEs)
5. [Comparing noisy neural population dynamics using optimal transport distances](#Comparing-noisy-neural-population-dynamics-using-optimal-transport-distances)
6. [Subgraph Federated Learning for Local Generalization](#Subgraph-Federated-Learning-for-Local-Generalization)
7. [RB-Modulation: Training-Free Stylization using Reference-Based Modulation](#RB-Modulation-Training-Free-Stylization-using-Reference-Based-Modulation)
8. [The Geometry of Categorical and Hierarchical Concepts in Large Language Models](#The-Geometry-of-Categorical-and-Hierarchical-Concepts-in-Large-Language-Models)
9. [TopoLM: brain-like spatio-functional organization in a topographic language model](#TopoLM-brain-like-spatio-functional-organization-in-a-topographic-language-model)
10. [Problem-Parameter-Free Federated Learning](#Problem-Parameter-Free-Federated-Learning)
11. [Latent Bayesian Optimization via Autoregressive Normalizing Flows](#Latent-Bayesian-Optimization-via-Autoregressive-Normalizing-Flows)
12. [BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions](#BigCodeBench-Benchmarking-Code-Generation-with-Diverse-Function-Calls-and-Complex-Instructions)
13. [ReGenesis: LLMs can Grow into Reasoning Generalists via Self-Improvement](#ReGenesis-LLMs-can-Grow-into-Reasoning-Generalists-via-Self-Improvement)
14. [LaMPlace: Learning to Optimize Cross-Stage Metrics in Macro Placement](#LaMPlace-Learning-to-Optimize-Cross-Stage-Metrics-in-Macro-Placement)
15. [MOS: Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection](#MOS-Model-Synergy-for-Test-Time-Adaptation-on-LiDAR-Based-3D-Object-Detection)
16. [On Conformal Isometry of Grid Cells: Learning Distance-Preserving Position Embedding](#On-Conformal-Isometry-of-Grid-Cells-Learning-Distance-Preserving-Position-Embedding)
17. [Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows](#Spider-20-Evaluating-Language-Models-on-Real-World-Enterprise-Text-to-SQL-Workflows)
18. [EmbodiedSAM: Online Segment Any 3D Thing in Real Time](#EmbodiedSAM-Online-Segment-Any-3D-Thing-in-Real-Time)
19. [Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping](#Dynamic-Multimodal-Evaluation-with-Flexible-Complexity-by-Vision-Language-Bootstrapping)
20. [LARP: Tokenizing Videos with a Learned Autoregressive Generative Prior](#LARP-Tokenizing-Videos-with-a-Learned-Autoregressive-Generative-Prior)
21. [Knowing Your Target: Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding](#Knowing-Your-Target-Target-Aware-Transformer-Makes-Better-Spatio-Temporal-Video-Grounding)
22. [Self-Improvement in Language Models: The Sharpening Mechanism](#Self-Improvement-in-Language-Models-The-Sharpening-Mechanism)
23. [Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models](#Do-I-Know-This-Entity-Knowledge-Awareness-and-Hallucinations-in-Language-Models)
24. [LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization](#LoRA-Done-RITE-Robust-Invariant-Transformation-Equilibration-for-LoRA-Optimization)
25. [Reasoning Elicitation in Language Models via Counterfactual Feedback](#Reasoning-Elicitation-in-Language-Models-via-Counterfactual-Feedback)
26. [Attention as a Hypernetwork](#Attention-as-a-Hypernetwork)
27. [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](#Unlocking-State-Tracking-in-Linear-RNNs-Through-Negative-Eigenvalues)
28. [Learning Randomized Algorithms with Transformers](#Learning-Randomized-Algorithms-with-Transformers)
29. [Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement](#Trust-or-Escalate-LLM-Judges-with-Provable-Guarantees-for-Human-Agreement)
30. [HiRA: Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models](#HiRA-Parameter-Efficient-Hadamard-High-Rank-Adaptation-for-Large-Language-Models)
31. [Proteina: Scaling Flow-based Protein Structure Generative Models](#Proteina-Scaling-Flow-based-Protein-Structure-Generative-Models)
32. [REEF: Representation Encoding Fingerprints for Large Language Models](#REEF-Representation-Encoding-Fingerprints-for-Large-Language-Models)
33. [A Decade's Battle on Dataset Bias: Are We There Yet?](#A-Decades-Battle-on-Dataset-Bias-Are-We-There-Yet)
34. [Context-Parametric Inversion: Why Instruction Finetuning May Not Actually Improve Context Reliance](#Context-Parametric-Inversion-Why-Instruction-Finetuning-May-Not-Actually-Improve-Context-Reliance)
35. [Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](#Transfusion-Predict-the-Next-Token-and-Diffuse-Images-with-One-Multi-Modal-Model)
36. [ECD: A Machine Learning Benchmark for Predicting Enhanced-Precision Electronic Charge Density in Crystalline Inorganic Materials](#ECD-A-Machine-Learning-Benchmark-for-Predicting-Enhanced-Precision-Electronic-Charge-Density-in-Crystalline-Inorganic-Materials)

---


## Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition

### Images

![1bb7bcfa376e8fbbd2bc3f06f9449a403be2cda38d26765fe6c01291d5732bb8.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/images/1bb7bcfa376e8fbbd2bc3f06f9449a403be2cda38d26765fe6c01291d5732bb8.jpg)

![35876925e3c05388b5b5a8eaa77370f239c147a807895c5a97e71f3d0310a97f.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/images/35876925e3c05388b5b5a8eaa77370f239c147a807895c5a97e71f3d0310a97f.jpg)

![39e0b7c57b46d3f1b4d115a6276b8156939e50a312ae5d99b76974bf5128a9a3.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/images/39e0b7c57b46d3f1b4d115a6276b8156939e50a312ae5d99b76974bf5128a9a3.jpg)

![50e462733b2fb060ff6ce3ee100c5ad157ea171b2965b784ef9c9a84404c093d.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/images/50e462733b2fb060ff6ce3ee100c5ad157ea171b2965b784ef9c9a84404c093d.jpg)

![6429fea79ac176aa36f67bbcdc845d1bd8630e024638a881434ba5ad1d2f7505.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/images/6429fea79ac176aa36f67bbcdc845d1bd8630e024638a881434ba5ad1d2f7505.jpg)

![6f22861d9c1406e2ec89a312c2111318e210293bf605d69366f9a93975ac7094.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/images/6f22861d9c1406e2ec89a312c2111318e210293bf605d69366f9a93975ac7094.jpg)

![6f98b4f015e71347426f423bc3e1146677c40b35f0a7c251a0f2010da2d1f9fe.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/images/6f98b4f015e71347426f423bc3e1146677c40b35f0a7c251a0f2010da2d1f9fe.jpg)

![94d66f92f4149ba7ab6c229087c6926afff385f081f11fb41337614b5ac26575.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/images/94d66f92f4149ba7ab6c229087c6926afff385f081f11fb41337614b5ac26575.jpg)

![a4ff93172cf4c97e034923c21cfb0b23cc4c2fb9115397f5b36d058cc699dc54.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/images/a4ff93172cf4c97e034923c21cfb0b23cc4c2fb9115397f5b36d058cc699dc54.jpg)

![d316a416a69ca7e71aeb38f0ff215bbbc38138c3377f5cecf06a28a7f4813821.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/images/d316a416a69ca7e71aeb38f0ff215bbbc38138c3377f5cecf06a28a7f4813821.jpg)

### Tables

![114e36e47ed7a8a2137360413ecdc0ab3c298275adafa6fbe6abc793bee368a1.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/tables/114e36e47ed7a8a2137360413ecdc0ab3c298275adafa6fbe6abc793bee368a1.jpg)

![160c96e052b81688d26c1dd39e6d64ef7aa2fb20e871d521d68c9c1bade392f8.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/tables/160c96e052b81688d26c1dd39e6d64ef7aa2fb20e871d521d68c9c1bade392f8.jpg)

![1fb394536e6429b221c6cf1c801d2a43f46146ff74846e5167fd71bf9228aab1.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/tables/1fb394536e6429b221c6cf1c801d2a43f46146ff74846e5167fd71bf9228aab1.jpg)

![24532c787f9d2ef2bca9350114d377d4b6722da9c3c4bc237d967c43ef49f4c0.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/tables/24532c787f9d2ef2bca9350114d377d4b6722da9c3c4bc237d967c43ef49f4c0.jpg)

![2bf0a0e0e04dc73dd3f488e1f4795c732752e3b22581c3228afdb24b7e73ec17.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/tables/2bf0a0e0e04dc73dd3f488e1f4795c732752e3b22581c3228afdb24b7e73ec17.jpg)

![59682f7c2655279e85a6387d5643db0432428c6ee600f4312cb070d0fabcf9e3.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/tables/59682f7c2655279e85a6387d5643db0432428c6ee600f4312cb070d0fabcf9e3.jpg)

![5e2c6f82fdaf39b23f4ad3b8ab1db9da2af48394f9ec721c75932c73ab2b6752.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/tables/5e2c6f82fdaf39b23f4ad3b8ab1db9da2af48394f9ec721c75932c73ab2b6752.jpg)

![6875e28da69db450f5a8ab9e60743ac5d4d1d634b28be81368a32ddb175c8a32.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/tables/6875e28da69db450f5a8ab9e60743ac5d4d1d634b28be81368a32ddb175c8a32.jpg)

![6f21b3bc6ad3cc23975f947e65958763ab801d4bb3e834fa9285194d1b48d932.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/tables/6f21b3bc6ad3cc23975f947e65958763ab801d4bb3e834fa9285194d1b48d932.jpg)

![9a272f7d249f4d1da20f69e2fd97ffd0fcdbb8ee278ea6aeee8807347869b045.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/tables/9a272f7d249f4d1da20f69e2fd97ffd0fcdbb8ee278ea6aeee8807347869b045.jpg)

![aa12e600c19993d05edd4f4bfbdebdb14959fa45049d1a67b95ed2432dd15255.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/tables/aa12e600c19993d05edd4f4bfbdebdb14959fa45049d1a67b95ed2432dd15255.jpg)

![b2b6bee478b2aeb867515a7075f43138fb1ba6cc93bf435a4e6a67d788752b9f.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/tables/b2b6bee478b2aeb867515a7075f43138fb1ba6cc93bf435a4e6a67d788752b9f.jpg)

![c1a553e88289845f1b6314660dc848b1bfbdb20d74e2599672c802928edfa131.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/tables/c1a553e88289845f1b6314660dc848b1bfbdb20d74e2599672c802928edfa131.jpg)

![d4643573df6928897c925ce1ecb48e6bb203847f771367bbd2315892e4e22669.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/tables/d4643573df6928897c925ce1ecb48e6bb203847f771367bbd2315892e4e22669.jpg)

![dc1eef4ed3046999738aa787d95f7e4a29f0bfdc42edaba74dd78e45bdabb180.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/tables/dc1eef4ed3046999738aa787d95f7e4a29f0bfdc42edaba74dd78e45bdabb180.jpg)

![f76d837f2ec494f509e9feef5eba4899b658c998d5c93a4f02f86f3c634ebfc4.jpg](../iclr_results/85_Language Representations Can be What Recommenders Need_ Findings and Potentials/tables/f76d837f2ec494f509e9feef5eba4899b658c998d5c93a4f02f86f3c634ebfc4.jpg)

## Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition


### Images

![1aceeca04625b0feb7465181f7922510a7f11fc0596f98407961b63fb37c8429.jpg](../iclr_results/86_Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition/images/1aceeca04625b0feb7465181f7922510a7f11fc0596f98407961b63fb37c8429.jpg)

![26b2cc1a1e4b637bbe9d3bf361ab96f136a5f28888a36b9751fc312b6aab3c75.jpg](../iclr_results/86_Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition/images/26b2cc1a1e4b637bbe9d3bf361ab96f136a5f28888a36b9751fc312b6aab3c75.jpg)

![41962cf15f9789f204eb73ac5014ff78108ada0066309ff7f868f9758c129a22.jpg](../iclr_results/86_Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition/images/41962cf15f9789f204eb73ac5014ff78108ada0066309ff7f868f9758c129a22.jpg)

![437b430ac5b63bd6dd130505837062ecf493f76d4cab1d526b6e8a05da068a97.jpg](../iclr_results/86_Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition/images/437b430ac5b63bd6dd130505837062ecf493f76d4cab1d526b6e8a05da068a97.jpg)

![46e3c760e7561259cc57cae6a338cba25de062204fbea5c85bae25834c14ef95.jpg](../iclr_results/86_Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition/images/46e3c760e7561259cc57cae6a338cba25de062204fbea5c85bae25834c14ef95.jpg)

![4d900e80391652445d8e0c218df9f9fa2a6a54436ed16f313dfef738ceefc974.jpg](../iclr_results/86_Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition/images/4d900e80391652445d8e0c218df9f9fa2a6a54436ed16f313dfef738ceefc974.jpg)

![4dde04a93e8f7d14291240cac270777df096d07a059d14e8c745edf322f62e5b.jpg](../iclr_results/86_Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition/images/4dde04a93e8f7d14291240cac270777df096d07a059d14e8c745edf322f62e5b.jpg)

![65846977f86b5e5081577f48c1ae26a21d69f1bc54271d95ed0a411f47804643.jpg](../iclr_results/86_Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition/images/65846977f86b5e5081577f48c1ae26a21d69f1bc54271d95ed0a411f47804643.jpg)

![97596a9ba656878991c2aec085b0cc59369e0d6387edd76355408af033b6d082.jpg](../iclr_results/86_Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition/images/97596a9ba656878991c2aec085b0cc59369e0d6387edd76355408af033b6d082.jpg)

![9acd63cc498aa303830a04a7dddf653ded0511419bde5e31941dbe2041bc760e.jpg](../iclr_results/86_Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition/images/9acd63cc498aa303830a04a7dddf653ded0511419bde5e31941dbe2041bc760e.jpg)

![afd8afa6fd025e717ae6f2c1fca3e75197649f9e6f5f9cb248a53a3c1581a25a.jpg](../iclr_results/86_Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition/images/afd8afa6fd025e717ae6f2c1fca3e75197649f9e6f5f9cb248a53a3c1581a25a.jpg)

![ba7620ada78eab95cc1a6e2d74bd7f06df15e3f5a5185967ccd8cdb1ac5657a8.jpg](../iclr_results/86_Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition/images/ba7620ada78eab95cc1a6e2d74bd7f06df15e3f5a5185967ccd8cdb1ac5657a8.jpg)

![c768029cdd9673c273d0372a34dfc2e5e8ea61fb60cd004b9866cdcb3050fe4a.jpg](../iclr_results/86_Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition/images/c768029cdd9673c273d0372a34dfc2e5e8ea61fb60cd004b9866cdcb3050fe4a.jpg)

![c99c7ddb3a55d5deed28ce798806bed4cd47100c818d685f62a5c7a249eef6d0.jpg](../iclr_results/86_Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition/images/c99c7ddb3a55d5deed28ce798806bed4cd47100c818d685f62a5c7a249eef6d0.jpg)

![df9969cbe9ca5ca6761dbe0053bc830ce33bbab355e2f11893817df8ca62b481.jpg](../iclr_results/86_Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition/images/df9969cbe9ca5ca6761dbe0053bc830ce33bbab355e2f11893817df8ca62b481.jpg)

![e376e7b2e505ed8a89b048e77abd541552b0ca664888120cc9f91859c951b075.jpg](../iclr_results/86_Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition/images/e376e7b2e505ed8a89b048e77abd541552b0ca664888120cc9f91859c951b075.jpg)

![f40aef26a969c1f56b9917afa2ac9f7957f151100b91df968142c3bc22480663.jpg](../iclr_results/86_Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition/images/f40aef26a969c1f56b9917afa2ac9f7957f151100b91df968142c3bc22480663.jpg)

![f63d9cf3e20994b3159b9619d7c76f82e363fbd54326a34f85419ccbb376223f.jpg](../iclr_results/86_Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition/images/f63d9cf3e20994b3159b9619d7c76f82e363fbd54326a34f85419ccbb376223f.jpg)

### Tables

![7638b2ea5784449a12681f79df511c389904a842ca954f80898e457600bfd668.jpg](../iclr_results/86_Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition/tables/7638b2ea5784449a12681f79df511c389904a842ca954f80898e457600bfd668.jpg)

![c55028e4151c07423bdd10a1162426568932c2d9cd8cc1c115e4d27e28d8b65b.jpg](../iclr_results/86_Knowledge Entropy Decay during Language Model Pretraining Hinders New Knowledge Acquisition/tables/c55028e4151c07423bdd10a1162426568932c2d9cd8cc1c115e4d27e28d8b65b.jpg)

## Emergence of meta-stable clustering in mean-field transformer models


### Images

![021c044c8ca9ba28f448f28c3448c9f2654a75e6131acc4ff8b0e7571f5c3038.jpg](../iclr_results/88_Emergence of meta-stable clustering in mean-field transformer models/images/021c044c8ca9ba28f448f28c3448c9f2654a75e6131acc4ff8b0e7571f5c3038.jpg)

![16b831706ded63ba5041d649e67e7964a58e5d159a18f0762d6455670a068bc3.jpg](../iclr_results/88_Emergence of meta-stable clustering in mean-field transformer models/images/16b831706ded63ba5041d649e67e7964a58e5d159a18f0762d6455670a068bc3.jpg)

![1e7d263c331e501ccb96bc7ec31b40b45635b09ab698ae5ba75ba5f4b6ec8fad.jpg](../iclr_results/88_Emergence of meta-stable clustering in mean-field transformer models/images/1e7d263c331e501ccb96bc7ec31b40b45635b09ab698ae5ba75ba5f4b6ec8fad.jpg)

![4434929df2ac1081137cd991cd15fdb2309efcd8c400ac8cd83828341939bfdd.jpg](../iclr_results/88_Emergence of meta-stable clustering in mean-field transformer models/images/4434929df2ac1081137cd991cd15fdb2309efcd8c400ac8cd83828341939bfdd.jpg)

![46ccf863dd9a35a9351b512dfe04a721490f89f0fe69727a79b65eb67314d2a5.jpg](../iclr_results/88_Emergence of meta-stable clustering in mean-field transformer models/images/46ccf863dd9a35a9351b512dfe04a721490f89f0fe69727a79b65eb67314d2a5.jpg)

![f0500425f06c30dd8b125a1a89b42fca99c20cc05686b3e40499f7bee49b7714.jpg](../iclr_results/88_Emergence of meta-stable clustering in mean-field transformer models/images/f0500425f06c30dd8b125a1a89b42fca99c20cc05686b3e40499f7bee49b7714.jpg)

## Data Selection via Optimal Control for Language Models


### Images

![19e15d8163c474d086c1eddcbbee093ece5dd58eec112441c8f67e42f6b58008.jpg](../iclr_results/89_Data Selection via Optimal Control for Language Models/images/19e15d8163c474d086c1eddcbbee093ece5dd58eec112441c8f67e42f6b58008.jpg)

![52968c727c9f54c75f551f256136d9b7ccce6cde6b7aea8a9484047ca7cc0912.jpg](../iclr_results/89_Data Selection via Optimal Control for Language Models/images/52968c727c9f54c75f551f256136d9b7ccce6cde6b7aea8a9484047ca7cc0912.jpg)

![678972a2c9872acc2a00228a36603e13bea3c209fad2b96cf57feff74d2114e7.jpg](../iclr_results/89_Data Selection via Optimal Control for Language Models/images/678972a2c9872acc2a00228a36603e13bea3c209fad2b96cf57feff74d2114e7.jpg)

![68f8150387c1143919c8ccbd33206281fb8e40bc4fbdedeb30e141686d376c0b.jpg](../iclr_results/89_Data Selection via Optimal Control for Language Models/images/68f8150387c1143919c8ccbd33206281fb8e40bc4fbdedeb30e141686d376c0b.jpg)

![7fc74ae3933ce62191060414675119f29b53f634f648d9ded75135ed605c167c.jpg](../iclr_results/89_Data Selection via Optimal Control for Language Models/images/7fc74ae3933ce62191060414675119f29b53f634f648d9ded75135ed605c167c.jpg)

![915dfd0f2b496540dbe1e9cf4d1bfbb03f735444da244539f324fa003ae481b2.jpg](../iclr_results/89_Data Selection via Optimal Control for Language Models/images/915dfd0f2b496540dbe1e9cf4d1bfbb03f735444da244539f324fa003ae481b2.jpg)

![b0b5b7fd15b61f05df9dcffd389c12ed22a7ace4069fac5373157809adba043f.jpg](../iclr_results/89_Data Selection via Optimal Control for Language Models/images/b0b5b7fd15b61f05df9dcffd389c12ed22a7ace4069fac5373157809adba043f.jpg)

![ca439ef3b2addfe7a4b843285e1bc863b33e236bdae0161d9adbb3f3e2b2e962.jpg](../iclr_results/89_Data Selection via Optimal Control for Language Models/images/ca439ef3b2addfe7a4b843285e1bc863b33e236bdae0161d9adbb3f3e2b2e962.jpg)

![e365398978ab15f7f4f2e40eae6f5310741b2fd00de258d4c16d71468d54b2b9.jpg](../iclr_results/89_Data Selection via Optimal Control for Language Models/images/e365398978ab15f7f4f2e40eae6f5310741b2fd00de258d4c16d71468d54b2b9.jpg)

![f8eea761982d19f0b8c6848c7d99ea5adf0d6392170b35386e7361de7dbbc573.jpg](../iclr_results/89_Data Selection via Optimal Control for Language Models/images/f8eea761982d19f0b8c6848c7d99ea5adf0d6392170b35386e7361de7dbbc573.jpg)

### Tables

![1cb44c05bd57983bf8ccb9d5becfc34ba29d98dff0291c23422dc129548f5fd8.jpg](../iclr_results/89_Data Selection via Optimal Control for Language Models/tables/1cb44c05bd57983bf8ccb9d5becfc34ba29d98dff0291c23422dc129548f5fd8.jpg)

![1cf6bcaa9c48e3b1a50735575e3da5ed2331604f8011fd629e41cf1cc12429cb.jpg](../iclr_results/89_Data Selection via Optimal Control for Language Models/tables/1cf6bcaa9c48e3b1a50735575e3da5ed2331604f8011fd629e41cf1cc12429cb.jpg)

![1f46929a9f47a25f99bae9f2c8b9f0e7a910867d3c728cafece587de95da2d39.jpg](../iclr_results/89_Data Selection via Optimal Control for Language Models/tables/1f46929a9f47a25f99bae9f2c8b9f0e7a910867d3c728cafece587de95da2d39.jpg)

![38a9a537274a45a25feb18c697f265a4ce05d62149261fb375278ef6469bfe56.jpg](../iclr_results/89_Data Selection via Optimal Control for Language Models/tables/38a9a537274a45a25feb18c697f265a4ce05d62149261fb375278ef6469bfe56.jpg)

![47b76c90d156248e126c416e33583fccf307e618d52a05fd97c0f462ecbd6608.jpg](../iclr_results/89_Data Selection via Optimal Control for Language Models/tables/47b76c90d156248e126c416e33583fccf307e618d52a05fd97c0f462ecbd6608.jpg)

![6dc3ad1c9c9e270eba8b2fb1372023833bb418371699afd2f164d42d4d180f77.jpg](../iclr_results/89_Data Selection via Optimal Control for Language Models/tables/6dc3ad1c9c9e270eba8b2fb1372023833bb418371699afd2f164d42d4d180f77.jpg)

![8160997d352ee3b5d908892ff5105d3fbdd0d5e9045988aac4486fed8df944b0.jpg](../iclr_results/89_Data Selection via Optimal Control for Language Models/tables/8160997d352ee3b5d908892ff5105d3fbdd0d5e9045988aac4486fed8df944b0.jpg)

![952e078e51f182eef2b232a37e54c4300c06d70d77bc6e89c73eb32ba50bdcc2.jpg](../iclr_results/89_Data Selection via Optimal Control for Language Models/tables/952e078e51f182eef2b232a37e54c4300c06d70d77bc6e89c73eb32ba50bdcc2.jpg)

![bb2962d40353f213231d6f5a4a131fabbc64f1091a24906dde91b30fc6dfefcb.jpg](../iclr_results/89_Data Selection via Optimal Control for Language Models/tables/bb2962d40353f213231d6f5a4a131fabbc64f1091a24906dde91b30fc6dfefcb.jpg)

![bd2b09a460f0b619ca840d7a2004b377da3eb47cdb749a0cf727c2280bfcf1e5.jpg](../iclr_results/89_Data Selection via Optimal Control for Language Models/tables/bd2b09a460f0b619ca840d7a2004b377da3eb47cdb749a0cf727c2280bfcf1e5.jpg)

![cb1a757c04e14e758c675864570ff3bd96b1e7ec8f231cb25f835ebdc654f3f2.jpg](../iclr_results/89_Data Selection via Optimal Control for Language Models/tables/cb1a757c04e14e758c675864570ff3bd96b1e7ec8f231cb25f835ebdc654f3f2.jpg)

![f2e031b012cb66fc27579cbeaa4d26f4d7c30edc92b3a1a840b7d1e6fb9eba73.jpg](../iclr_results/89_Data Selection via Optimal Control for Language Models/tables/f2e031b012cb66fc27579cbeaa4d26f4d7c30edc92b3a1a840b7d1e6fb9eba73.jpg)

## Feedback Favors the Generalization of Neural ODEs


### Images

![141416cd7baf5b741077a2624a6abce57a8280a0f14a9adf595e560a93d753a1.jpg](../iclr_results/90_Feedback Favors the Generalization of Neural ODEs/images/141416cd7baf5b741077a2624a6abce57a8280a0f14a9adf595e560a93d753a1.jpg)

![2b1caf934132116960fec3ae14a1ad7f5b9c86bd5668b9d35b071bc2321aad0b.jpg](../iclr_results/90_Feedback Favors the Generalization of Neural ODEs/images/2b1caf934132116960fec3ae14a1ad7f5b9c86bd5668b9d35b071bc2321aad0b.jpg)

![353ac68c745e29c8df3c9ed93b1086429cedf8cbd7374ecf4b7bbe13c7f04be0.jpg](../iclr_results/90_Feedback Favors the Generalization of Neural ODEs/images/353ac68c745e29c8df3c9ed93b1086429cedf8cbd7374ecf4b7bbe13c7f04be0.jpg)

![3de59be99c06ea5d47326c07100b7261aaf83a040f9618f104978a05eb9490d6.jpg](../iclr_results/90_Feedback Favors the Generalization of Neural ODEs/images/3de59be99c06ea5d47326c07100b7261aaf83a040f9618f104978a05eb9490d6.jpg)

![4b3cafb602f9f9d16a5074d3c3bfcd31b4d599d416c3558e77788181bd4c4597.jpg](../iclr_results/90_Feedback Favors the Generalization of Neural ODEs/images/4b3cafb602f9f9d16a5074d3c3bfcd31b4d599d416c3558e77788181bd4c4597.jpg)

![563b1f9ff0f4ac903cbee97d8bb00a3a3adb0b8f42d3a9cb300fc822be4a095f.jpg](../iclr_results/90_Feedback Favors the Generalization of Neural ODEs/images/563b1f9ff0f4ac903cbee97d8bb00a3a3adb0b8f42d3a9cb300fc822be4a095f.jpg)

![5e0c6f8ff9bed926bfe815dc0eaf3a5706f2b17ff7073f8d5361a1bc34db6ee0.jpg](../iclr_results/90_Feedback Favors the Generalization of Neural ODEs/images/5e0c6f8ff9bed926bfe815dc0eaf3a5706f2b17ff7073f8d5361a1bc34db6ee0.jpg)

![66cd7448bd9a5846e010ec9741875dcfb2b3ede32a30a32742da29708fad15d0.jpg](../iclr_results/90_Feedback Favors the Generalization of Neural ODEs/images/66cd7448bd9a5846e010ec9741875dcfb2b3ede32a30a32742da29708fad15d0.jpg)

![73057e15c7d6240fb375453672f3c7a3c72d7d67666dfd18d55d8a8fad36754f.jpg](../iclr_results/90_Feedback Favors the Generalization of Neural ODEs/images/73057e15c7d6240fb375453672f3c7a3c72d7d67666dfd18d55d8a8fad36754f.jpg)

![7c6602888c5991a53c523bbdce15cad10f468eaec94170eaec7c1f29b62e2a99.jpg](../iclr_results/90_Feedback Favors the Generalization of Neural ODEs/images/7c6602888c5991a53c523bbdce15cad10f468eaec94170eaec7c1f29b62e2a99.jpg)

![873afabe083b67be9490abd50f648a1777ab33de1b00a7815342e19fdcf4d459.jpg](../iclr_results/90_Feedback Favors the Generalization of Neural ODEs/images/873afabe083b67be9490abd50f648a1777ab33de1b00a7815342e19fdcf4d459.jpg)

![94149050745ca74d514f1f211687ddfc9b7536d05426e8ede37ec91abf1bf3ce.jpg](../iclr_results/90_Feedback Favors the Generalization of Neural ODEs/images/94149050745ca74d514f1f211687ddfc9b7536d05426e8ede37ec91abf1bf3ce.jpg)

![958236bc4b648cb1daa73028d64769a08c538dd16388f6823708a791b68168e5.jpg](../iclr_results/90_Feedback Favors the Generalization of Neural ODEs/images/958236bc4b648cb1daa73028d64769a08c538dd16388f6823708a791b68168e5.jpg)

![9c6862a73f971c2843be2f301c2bda2760d1546205d99d9be040cd8b716fdaad.jpg](../iclr_results/90_Feedback Favors the Generalization of Neural ODEs/images/9c6862a73f971c2843be2f301c2bda2760d1546205d99d9be040cd8b716fdaad.jpg)

![a04da881bd128791ec4681b4dd1b65ef3c6c46a4f343183ba394a4647990d588.jpg](../iclr_results/90_Feedback Favors the Generalization of Neural ODEs/images/a04da881bd128791ec4681b4dd1b65ef3c6c46a4f343183ba394a4647990d588.jpg)

![a0f350d331eea514dd4a00d464644cf65b915eda16ce623934a6580469c53eb4.jpg](../iclr_results/90_Feedback Favors the Generalization of Neural ODEs/images/a0f350d331eea514dd4a00d464644cf65b915eda16ce623934a6580469c53eb4.jpg)

![a33d05f7f09172c3e7f5b80cecedbe228b47d0141e644169cf52e47d9f86a076.jpg](../iclr_results/90_Feedback Favors the Generalization of Neural ODEs/images/a33d05f7f09172c3e7f5b80cecedbe228b47d0141e644169cf52e47d9f86a076.jpg)

![b214ca96b4692749289830ac77299d39b65da121e7de4eadc34f8ee151430275.jpg](../iclr_results/90_Feedback Favors the Generalization of Neural ODEs/images/b214ca96b4692749289830ac77299d39b65da121e7de4eadc34f8ee151430275.jpg)

![cbeb87cdf15a1bef0809b89420e88d238775237221a37225781fc475748f2120.jpg](../iclr_results/90_Feedback Favors the Generalization of Neural ODEs/images/cbeb87cdf15a1bef0809b89420e88d238775237221a37225781fc475748f2120.jpg)

![d6c61d33f3965cd9d74ccf830cf3d43e283714db9a698640e2cbf09c37fb5b1f.jpg](../iclr_results/90_Feedback Favors the Generalization of Neural ODEs/images/d6c61d33f3965cd9d74ccf830cf3d43e283714db9a698640e2cbf09c37fb5b1f.jpg)

![e8a299d26e51b5052bc6737cc36168e0d4a4df95728fc5bf488d603842c0332f.jpg](../iclr_results/90_Feedback Favors the Generalization of Neural ODEs/images/e8a299d26e51b5052bc6737cc36168e0d4a4df95728fc5bf488d603842c0332f.jpg)

![ee529f847eb4e5da1888a3dd824cbfdcc6e35d0d74c4aa0c84907916d68ef44a.jpg](../iclr_results/90_Feedback Favors the Generalization of Neural ODEs/images/ee529f847eb4e5da1888a3dd824cbfdcc6e35d0d74c4aa0c84907916d68ef44a.jpg)

![eec732852fc43f391aabef82d9f56c6024b4b461fc87e7d90a11e6ac30458ddc.jpg](../iclr_results/90_Feedback Favors the Generalization of Neural ODEs/images/eec732852fc43f391aabef82d9f56c6024b4b461fc87e7d90a11e6ac30458ddc.jpg)

## Comparing noisy neural population dynamics using optimal transport distances


### Images

![0b0fa8675e373f8e8ecd398dca1ca484dd3d350645105661d565044835433e1e.jpg](../iclr_results/91_Comparing noisy neural population dynamics using optimal transport distances/images/0b0fa8675e373f8e8ecd398dca1ca484dd3d350645105661d565044835433e1e.jpg)

![14776444e7fe4e53c76ff6bfb38805169f0064826c824f54b90a63a1e7f4fff2.jpg](../iclr_results/91_Comparing noisy neural population dynamics using optimal transport distances/images/14776444e7fe4e53c76ff6bfb38805169f0064826c824f54b90a63a1e7f4fff2.jpg)

![24bf6bfdfcb1aebd9c4b4b3a3c9b34322a69a8b10e779d3e3ad5eff1f6980771.jpg](../iclr_results/91_Comparing noisy neural population dynamics using optimal transport distances/images/24bf6bfdfcb1aebd9c4b4b3a3c9b34322a69a8b10e779d3e3ad5eff1f6980771.jpg)

![299824d688045e03971c02c653d8d84dbb3757c006c9e57cde0dffb45f633dbd.jpg](../iclr_results/91_Comparing noisy neural population dynamics using optimal transport distances/images/299824d688045e03971c02c653d8d84dbb3757c006c9e57cde0dffb45f633dbd.jpg)

![311e8a3e8c701ef26bddfd5079f696b24240a3b574ed37dc859c6b8d0caff70e.jpg](../iclr_results/91_Comparing noisy neural population dynamics using optimal transport distances/images/311e8a3e8c701ef26bddfd5079f696b24240a3b574ed37dc859c6b8d0caff70e.jpg)

![515ec4e6bc7b879ecf901cae425a066af30bfc68290da5efc195b1012af3765d.jpg](../iclr_results/91_Comparing noisy neural population dynamics using optimal transport distances/images/515ec4e6bc7b879ecf901cae425a066af30bfc68290da5efc195b1012af3765d.jpg)

![56ad95c96336dc8895128d4e3627e0bb66031536ce7352b24238f29b75f9066f.jpg](../iclr_results/91_Comparing noisy neural population dynamics using optimal transport distances/images/56ad95c96336dc8895128d4e3627e0bb66031536ce7352b24238f29b75f9066f.jpg)

![625463b28eab6e356e8857d5b68859ac61ef5e43c60016e08d75d512eefd7a65.jpg](../iclr_results/91_Comparing noisy neural population dynamics using optimal transport distances/images/625463b28eab6e356e8857d5b68859ac61ef5e43c60016e08d75d512eefd7a65.jpg)

![694a0d995bc534741b31672985c3345e91bf482988f255e62d91efb36be1b94b.jpg](../iclr_results/91_Comparing noisy neural population dynamics using optimal transport distances/images/694a0d995bc534741b31672985c3345e91bf482988f255e62d91efb36be1b94b.jpg)

![71c496305b3b050ffe9cd5f6ee7a4db41225322ce8139a5aa0bf134c74af3f3a.jpg](../iclr_results/91_Comparing noisy neural population dynamics using optimal transport distances/images/71c496305b3b050ffe9cd5f6ee7a4db41225322ce8139a5aa0bf134c74af3f3a.jpg)

![a15e453840650ffe72b795e8c3e2146324b78f2c28fa3cdc8e66a030722d6bc2.jpg](../iclr_results/91_Comparing noisy neural population dynamics using optimal transport distances/images/a15e453840650ffe72b795e8c3e2146324b78f2c28fa3cdc8e66a030722d6bc2.jpg)

![e95c7107a99c703c091b42bfae35079d24bd0e54ddb979466476d3684101d033.jpg](../iclr_results/91_Comparing noisy neural population dynamics using optimal transport distances/images/e95c7107a99c703c091b42bfae35079d24bd0e54ddb979466476d3684101d033.jpg)

![eb6868f60da92b79b48bca213d1d0ca177438f09022270e7f123f105d0917a59.jpg](../iclr_results/91_Comparing noisy neural population dynamics using optimal transport distances/images/eb6868f60da92b79b48bca213d1d0ca177438f09022270e7f123f105d0917a59.jpg)

## Subgraph Federated Learning for Local Generalization


### Images

![01856a481d4cdb9b20be866e02a6a9313bd87a11ee1efe7f74b157c35768cf02.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/images/01856a481d4cdb9b20be866e02a6a9313bd87a11ee1efe7f74b157c35768cf02.jpg)

![0de2457d2132a54f5d04d3c6c688c1ee1019104c3852a102f8b2489b7c4e9a7e.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/images/0de2457d2132a54f5d04d3c6c688c1ee1019104c3852a102f8b2489b7c4e9a7e.jpg)

![298e1fe3c752b4e07d86b3de62316e44a58fd074c0641c8097bb416bc65183f3.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/images/298e1fe3c752b4e07d86b3de62316e44a58fd074c0641c8097bb416bc65183f3.jpg)

![485459937304e4dac12c8248e12bfbdbd7366205ea12d5fbd81b62188c0c7a53.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/images/485459937304e4dac12c8248e12bfbdbd7366205ea12d5fbd81b62188c0c7a53.jpg)

![4afb6e73f8f0f8b7ac6ff938b971325adb4adc3c10d8075c5ecfe50b24626ed7.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/images/4afb6e73f8f0f8b7ac6ff938b971325adb4adc3c10d8075c5ecfe50b24626ed7.jpg)

![5cb726291cd5ca2d31ea63556f2a951f3b4d1b8fde9045125ddcfc9f2c00527f.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/images/5cb726291cd5ca2d31ea63556f2a951f3b4d1b8fde9045125ddcfc9f2c00527f.jpg)

![7379ba11beb17fffb4d2b4a24f6ec17a30f72a99738ac0b4d60941c6e5a4e050.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/images/7379ba11beb17fffb4d2b4a24f6ec17a30f72a99738ac0b4d60941c6e5a4e050.jpg)

![96dcab18c470840ff61162023456482782b4a5ecdf1a42bd4aec549968bfe3e7.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/images/96dcab18c470840ff61162023456482782b4a5ecdf1a42bd4aec549968bfe3e7.jpg)

![9e928cc1be10ce326b779b9ee54dd988b6a0c2d60bc27b4bea83b4936e754e4f.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/images/9e928cc1be10ce326b779b9ee54dd988b6a0c2d60bc27b4bea83b4936e754e4f.jpg)

### Tables

![13daace9bcb8b4bda7c12d54f6eb9eca8693105a953a95e8457d9a330a8bc036.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/tables/13daace9bcb8b4bda7c12d54f6eb9eca8693105a953a95e8457d9a330a8bc036.jpg)

![209436e48f064ca993f8e8cf90d9b706e6cffb7964addbfb69a6326f4c9c5484.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/tables/209436e48f064ca993f8e8cf90d9b706e6cffb7964addbfb69a6326f4c9c5484.jpg)

![20d3e451021b98b5d650e93ab4a178578a6b3ad57c158c0331b6d4f4d8fd20cd.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/tables/20d3e451021b98b5d650e93ab4a178578a6b3ad57c158c0331b6d4f4d8fd20cd.jpg)

![333f2ac6faa439b5c911a5b2c965bcf358ec1e097486771b7164c44141d4fac0.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/tables/333f2ac6faa439b5c911a5b2c965bcf358ec1e097486771b7164c44141d4fac0.jpg)

![3d9c74b65bc3107cfd87c3192ee25698248d9e22a382db461cbbef4a01b702a7.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/tables/3d9c74b65bc3107cfd87c3192ee25698248d9e22a382db461cbbef4a01b702a7.jpg)

![44c8a8c726c20e217a478a9b9024142d302dd1e1b8412bc5bbed830d234732df.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/tables/44c8a8c726c20e217a478a9b9024142d302dd1e1b8412bc5bbed830d234732df.jpg)

![56ee95189d0df63f0163b8a6dfc2c7546db688d100daac2c5071641380be5b16.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/tables/56ee95189d0df63f0163b8a6dfc2c7546db688d100daac2c5071641380be5b16.jpg)

![5e007961a9aeda07522f8fd544ab70598fbcb5bc921574b0422384b65df01b7d.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/tables/5e007961a9aeda07522f8fd544ab70598fbcb5bc921574b0422384b65df01b7d.jpg)

![5e12e1781c80edf276c63f6035bb0d589d9ff5578525a4d0225fe86753af9f6b.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/tables/5e12e1781c80edf276c63f6035bb0d589d9ff5578525a4d0225fe86753af9f6b.jpg)

![64ed3a08e6727d4022b2efa2882d151f42c307bd5ff6e50681f114695d735bac.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/tables/64ed3a08e6727d4022b2efa2882d151f42c307bd5ff6e50681f114695d735bac.jpg)

![6e683de10a44e6d7749832b7960b93b44da7458c19c57418a669557722463dee.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/tables/6e683de10a44e6d7749832b7960b93b44da7458c19c57418a669557722463dee.jpg)

![7d94dd2b7ba9ee25710fa7eb0c60b0506bee51b548113768165a0b4ac9d1288b.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/tables/7d94dd2b7ba9ee25710fa7eb0c60b0506bee51b548113768165a0b4ac9d1288b.jpg)

![8fdcb9055358cd35450e0d8c35fda34e43a16269f5146bd0311d6410e52b37a3.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/tables/8fdcb9055358cd35450e0d8c35fda34e43a16269f5146bd0311d6410e52b37a3.jpg)

![99faf39e368307f4978a2641a4c9015a708145c7fbbf4e6701f67c555d0629c0.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/tables/99faf39e368307f4978a2641a4c9015a708145c7fbbf4e6701f67c555d0629c0.jpg)

![b93ebc521fceae74ad61e35dce3db469ece38bf4ecc54c3ee688c11b6b76d66a.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/tables/b93ebc521fceae74ad61e35dce3db469ece38bf4ecc54c3ee688c11b6b76d66a.jpg)

![bc4f2bb4200e8ef246c05fb04408ae88961c6ac3f06de5380aa72dcf0cf19243.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/tables/bc4f2bb4200e8ef246c05fb04408ae88961c6ac3f06de5380aa72dcf0cf19243.jpg)

![c9d6f6f2898428c759dae6ae4cdc67fa546b37815ff58dc2058efec194a30ce0.jpg](../iclr_results/92_Subgraph Federated Learning for Local Generalization/tables/c9d6f6f2898428c759dae6ae4cdc67fa546b37815ff58dc2058efec194a30ce0.jpg)

## RB-Modulation: Training-Free Stylization using Reference-Based Modulation


### Images

![09474e7897d5a07fa3c0d6eef21092247f2dfad40ad718063be1f96d9149b869.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/09474e7897d5a07fa3c0d6eef21092247f2dfad40ad718063be1f96d9149b869.jpg)

![23744c9bd52daf3b30efbbbe2b8c5443fbed1b4ca18708bc9ad371ce449aa9f5.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/23744c9bd52daf3b30efbbbe2b8c5443fbed1b4ca18708bc9ad371ce449aa9f5.jpg)

![2fe2222dd18483f97037bff17e6827853b57e597b2789960c6ce31a93816a8d6.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/2fe2222dd18483f97037bff17e6827853b57e597b2789960c6ce31a93816a8d6.jpg)

![395bbdfa8f39218d9898bb8f874101baeb433e4435f7431f18d07497e844bf12.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/395bbdfa8f39218d9898bb8f874101baeb433e4435f7431f18d07497e844bf12.jpg)

![44f46e2df5b6c45e657ea805cd307d5e44caffc14f614d00607153fa4fc1d16a.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/44f46e2df5b6c45e657ea805cd307d5e44caffc14f614d00607153fa4fc1d16a.jpg)

![4698ddee0e5906bf64261d01473c212c449cb6641479738ded3882724f42e4fb.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/4698ddee0e5906bf64261d01473c212c449cb6641479738ded3882724f42e4fb.jpg)

![499a6873c11b0357038a081706fc025c2c23a8e8a22d88010cd577e22b695522.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/499a6873c11b0357038a081706fc025c2c23a8e8a22d88010cd577e22b695522.jpg)

![4c3e58434ab53d18476463e785901cabbc93eb053214428dd144e13477eb8ecf.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/4c3e58434ab53d18476463e785901cabbc93eb053214428dd144e13477eb8ecf.jpg)

![53dc9fa1c7f528b6430c4abbf68cf907a614213dda95da5d4cddc91adad71226.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/53dc9fa1c7f528b6430c4abbf68cf907a614213dda95da5d4cddc91adad71226.jpg)

![61b1a8623d810fb8ecabd0e1c5175dfd4807a41f35083d3c46b2af765f9f6115.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/61b1a8623d810fb8ecabd0e1c5175dfd4807a41f35083d3c46b2af765f9f6115.jpg)

![871acfe88f138e0f1eb0030bf69a0a23a1b30dba8c4c6b34d3fc5baa112bcca6.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/871acfe88f138e0f1eb0030bf69a0a23a1b30dba8c4c6b34d3fc5baa112bcca6.jpg)

![96df21f5548dbbf7fa72df5efba9b8898b2791bb6bb7d49ac25211cd7e0eba2e.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/96df21f5548dbbf7fa72df5efba9b8898b2791bb6bb7d49ac25211cd7e0eba2e.jpg)

![9ed59e1573e2d63cf7a183d38b9b58f847dd18c395cb08a17b062fbf7d00425f.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/9ed59e1573e2d63cf7a183d38b9b58f847dd18c395cb08a17b062fbf7d00425f.jpg)

![a3196d0863c12f6a3709b44b3622fca01142ebc81bb20507439e507c698c2c40.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/a3196d0863c12f6a3709b44b3622fca01142ebc81bb20507439e507c698c2c40.jpg)

![b63e732dbed3683dac039b745ae3026b28f617735e80a91a1aab69d5ef4ca547.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/b63e732dbed3683dac039b745ae3026b28f617735e80a91a1aab69d5ef4ca547.jpg)

![b653f332a2c9e09a2aae25f4f5e3f1b192cbe682097bfeb8e244fafe1967e32c.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/b653f332a2c9e09a2aae25f4f5e3f1b192cbe682097bfeb8e244fafe1967e32c.jpg)

![b9880c98afc7f2e6c5470ab59251fdd95839ebb4ae4bbbfeefda523061c3dfe9.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/b9880c98afc7f2e6c5470ab59251fdd95839ebb4ae4bbbfeefda523061c3dfe9.jpg)

![bf515cbcec1755d01e546427644585b21a81039afafa02bafe1983e5a742c4fb.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/bf515cbcec1755d01e546427644585b21a81039afafa02bafe1983e5a742c4fb.jpg)

![c03743dc43b102a0bfdd2fa2da889ad85418f4a03626cd586370bb5a0c63f3ad.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/c03743dc43b102a0bfdd2fa2da889ad85418f4a03626cd586370bb5a0c63f3ad.jpg)

![c8cc2a70a8dca97862bceef85420c68af53c0e6850bd1544617ec7abc3313c1c.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/c8cc2a70a8dca97862bceef85420c68af53c0e6850bd1544617ec7abc3313c1c.jpg)

![cbff2246d511d458339daacfa8aa1ac27c36986ac74cd1d7cbea67879b36f17a.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/cbff2246d511d458339daacfa8aa1ac27c36986ac74cd1d7cbea67879b36f17a.jpg)

![cc2d47ee559b7edb6f7393c651315bd94e200ebe2063e0625f22663888fbd4bf.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/cc2d47ee559b7edb6f7393c651315bd94e200ebe2063e0625f22663888fbd4bf.jpg)

![cc7b364c41cc37ca1990e2abc979611d6f529c6c0bbf0bea58cc396725178d1d.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/cc7b364c41cc37ca1990e2abc979611d6f529c6c0bbf0bea58cc396725178d1d.jpg)

![d52b8c1d663ce63be875efaa10a4d4b579bf97ab4b7d440b3f3046788e0e804a.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/images/d52b8c1d663ce63be875efaa10a4d4b579bf97ab4b7d440b3f3046788e0e804a.jpg)

### Tables

![0d5acb6ead85191fb6783a2a6e1e70c2b3cfd3202a06cfe2e9104b8079cf6bcd.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/tables/0d5acb6ead85191fb6783a2a6e1e70c2b3cfd3202a06cfe2e9104b8079cf6bcd.jpg)

![437f2c77db17cb30adf1824a8c9afe0da60d3f831c03fd833fb52877fe8e7ccc.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/tables/437f2c77db17cb30adf1824a8c9afe0da60d3f831c03fd833fb52877fe8e7ccc.jpg)

![9e4985d2ddfa22dd098071f850822f30c42ed3218bbbf5db25f5eca962e6f136.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/tables/9e4985d2ddfa22dd098071f850822f30c42ed3218bbbf5db25f5eca962e6f136.jpg)

![e825b40759ba8f6589b93419cc17b40828f6ea2f7fbb4790735a08e3a7445d98.jpg](../iclr_results/93_RB-Modulation_ Training-Free Stylization using Reference-Based Modulation/tables/e825b40759ba8f6589b93419cc17b40828f6ea2f7fbb4790735a08e3a7445d98.jpg)

## The Geometry of Categorical and Hierarchical Concepts in Large Language Models


### Images

![08ad11ea00d315ce2e6477a9733a35c91c164a39dbbe2d221bb9801583c64059.jpg](../iclr_results/94_The Geometry of Categorical and Hierarchical Concepts in Large Language Models/images/08ad11ea00d315ce2e6477a9733a35c91c164a39dbbe2d221bb9801583c64059.jpg)

![28872fc54ecf3ca78b2c9af49fa6cb39579f56fbff2f0ca33b266c8a1ce9285e.jpg](../iclr_results/94_The Geometry of Categorical and Hierarchical Concepts in Large Language Models/images/28872fc54ecf3ca78b2c9af49fa6cb39579f56fbff2f0ca33b266c8a1ce9285e.jpg)

![33e5d43152c09e789cda6a129abeac8fb04b03ce7f9a28f8f624c8b4bae54871.jpg](../iclr_results/94_The Geometry of Categorical and Hierarchical Concepts in Large Language Models/images/33e5d43152c09e789cda6a129abeac8fb04b03ce7f9a28f8f624c8b4bae54871.jpg)

![49a529b455953520441c9c941bdb91856e45ee9c1c11f45631f20684a6156d80.jpg](../iclr_results/94_The Geometry of Categorical and Hierarchical Concepts in Large Language Models/images/49a529b455953520441c9c941bdb91856e45ee9c1c11f45631f20684a6156d80.jpg)

![4fb50750efa929714d9dbf489dbef369a85faafafa1ca1e2930876b09375a2af.jpg](../iclr_results/94_The Geometry of Categorical and Hierarchical Concepts in Large Language Models/images/4fb50750efa929714d9dbf489dbef369a85faafafa1ca1e2930876b09375a2af.jpg)

![583ebe81484fb6f390f33b70f5cc3b333256616adb1abebf168bc91d0fe597cc.jpg](../iclr_results/94_The Geometry of Categorical and Hierarchical Concepts in Large Language Models/images/583ebe81484fb6f390f33b70f5cc3b333256616adb1abebf168bc91d0fe597cc.jpg)

![8054ee7f25064dc729277ef504194954525e84353ec76c705b5de69c2b96aefc.jpg](../iclr_results/94_The Geometry of Categorical and Hierarchical Concepts in Large Language Models/images/8054ee7f25064dc729277ef504194954525e84353ec76c705b5de69c2b96aefc.jpg)

![895056dba34b43e071d064d365fc90932903a1a708482bb2babbee968dca0f1e.jpg](../iclr_results/94_The Geometry of Categorical and Hierarchical Concepts in Large Language Models/images/895056dba34b43e071d064d365fc90932903a1a708482bb2babbee968dca0f1e.jpg)

![8ffa966c33b2c55a68db5a453ad37f05dcb998b438add1519ecaf5ee9ad4d795.jpg](../iclr_results/94_The Geometry of Categorical and Hierarchical Concepts in Large Language Models/images/8ffa966c33b2c55a68db5a453ad37f05dcb998b438add1519ecaf5ee9ad4d795.jpg)

![91c0308a764fa8ac78bfa7ef964465e2c5a1de3a8735bd2947b30368f19c7a53.jpg](../iclr_results/94_The Geometry of Categorical and Hierarchical Concepts in Large Language Models/images/91c0308a764fa8ac78bfa7ef964465e2c5a1de3a8735bd2947b30368f19c7a53.jpg)

![9dffdaa642a32dfa5e8e82bbe537ca8cbe1ccdc50fb39021653849ada9ee038d.jpg](../iclr_results/94_The Geometry of Categorical and Hierarchical Concepts in Large Language Models/images/9dffdaa642a32dfa5e8e82bbe537ca8cbe1ccdc50fb39021653849ada9ee038d.jpg)

![a837829aebef6a17c412febfcf3120e2cedbfeb4bd907a1642c754a24cb17983.jpg](../iclr_results/94_The Geometry of Categorical and Hierarchical Concepts in Large Language Models/images/a837829aebef6a17c412febfcf3120e2cedbfeb4bd907a1642c754a24cb17983.jpg)

![ad318244d1baf661c1a388c29944eab48061b28c2616de34a434088a02bc8682.jpg](../iclr_results/94_The Geometry of Categorical and Hierarchical Concepts in Large Language Models/images/ad318244d1baf661c1a388c29944eab48061b28c2616de34a434088a02bc8682.jpg)

![b6867ea2e5f80a3ee70c64a06f2ba1956ae5e9f79477ae14eae903598c26da53.jpg](../iclr_results/94_The Geometry of Categorical and Hierarchical Concepts in Large Language Models/images/b6867ea2e5f80a3ee70c64a06f2ba1956ae5e9f79477ae14eae903598c26da53.jpg)

![d31215fc18b6d82d368409b2bd996fbecc82831b9e686bf99b835ad57e58a9ff.jpg](../iclr_results/94_The Geometry of Categorical and Hierarchical Concepts in Large Language Models/images/d31215fc18b6d82d368409b2bd996fbecc82831b9e686bf99b835ad57e58a9ff.jpg)

![da8e502a2f13cbb56e091a8cae6d6a2fe4463c5dac8ee36f81573a34c4464279.jpg](../iclr_results/94_The Geometry of Categorical and Hierarchical Concepts in Large Language Models/images/da8e502a2f13cbb56e091a8cae6d6a2fe4463c5dac8ee36f81573a34c4464279.jpg)

![dbf57d6c5f3026fc5ca2f24079894e22d0a3ab3f830c8cef085f21ff91a82682.jpg](../iclr_results/94_The Geometry of Categorical and Hierarchical Concepts in Large Language Models/images/dbf57d6c5f3026fc5ca2f24079894e22d0a3ab3f830c8cef085f21ff91a82682.jpg)

![e4e398876479fd4fdf0e17aee314ccfb0c6c7f1f99b8966f10fd212d9d61ff84.jpg](../iclr_results/94_The Geometry of Categorical and Hierarchical Concepts in Large Language Models/images/e4e398876479fd4fdf0e17aee314ccfb0c6c7f1f99b8966f10fd212d9d61ff84.jpg)

![eeaae67c17c274a3a257ec91c148c540db3cd285611ee076d1af0ff78d6f6d88.jpg](../iclr_results/94_The Geometry of Categorical and Hierarchical Concepts in Large Language Models/images/eeaae67c17c274a3a257ec91c148c540db3cd285611ee076d1af0ff78d6f6d88.jpg)

![f12e8d8c36f10f60b7c6efc7b40f0660c635ddbac5ff5d3141e9286b4fa467b9.jpg](../iclr_results/94_The Geometry of Categorical and Hierarchical Concepts in Large Language Models/images/f12e8d8c36f10f60b7c6efc7b40f0660c635ddbac5ff5d3141e9286b4fa467b9.jpg)

### Tables

![b23104b4708ff18f666d2ab2a4a3ff6eb5bc364d60cef26c60dc3b99dd78371d.jpg](../iclr_results/94_The Geometry of Categorical and Hierarchical Concepts in Large Language Models/tables/b23104b4708ff18f666d2ab2a4a3ff6eb5bc364d60cef26c60dc3b99dd78371d.jpg)

## TopoLM: brain-like spatio-functional organization in a topographic language model


### Images

![18a5884f8935ac84473f88267846098de373f5b60ba74853546198df2b6fbbf8.jpg](../iclr_results/95_TopoLM_ brain-like spatio-functional organization in a topographic language model/images/18a5884f8935ac84473f88267846098de373f5b60ba74853546198df2b6fbbf8.jpg)

![269e9185907a1c04a2f90bea5b6cf27cf15456f71d289086bd4648078c90f18b.jpg](../iclr_results/95_TopoLM_ brain-like spatio-functional organization in a topographic language model/images/269e9185907a1c04a2f90bea5b6cf27cf15456f71d289086bd4648078c90f18b.jpg)

![2d16fab705f65fa92d62202302fefdf718ed2d280fd387730db811e117339641.jpg](../iclr_results/95_TopoLM_ brain-like spatio-functional organization in a topographic language model/images/2d16fab705f65fa92d62202302fefdf718ed2d280fd387730db811e117339641.jpg)

![31fd565d0399a2f9f099d83b76275f1496fec78c3738989fead37fb0ee7b1861.jpg](../iclr_results/95_TopoLM_ brain-like spatio-functional organization in a topographic language model/images/31fd565d0399a2f9f099d83b76275f1496fec78c3738989fead37fb0ee7b1861.jpg)

![3583ce3f068f1b81eaed06e94fb697349290ca1599e77946b0017b546d4cd3da.jpg](../iclr_results/95_TopoLM_ brain-like spatio-functional organization in a topographic language model/images/3583ce3f068f1b81eaed06e94fb697349290ca1599e77946b0017b546d4cd3da.jpg)

![3de2a30987637b88d0086997681bfb7748542b5938e44ba45891c291700c1b74.jpg](../iclr_results/95_TopoLM_ brain-like spatio-functional organization in a topographic language model/images/3de2a30987637b88d0086997681bfb7748542b5938e44ba45891c291700c1b74.jpg)

![5234dde00aa98752ac78f93920d1da581c4f73380ac7aa69dd4db0829d1c80b3.jpg](../iclr_results/95_TopoLM_ brain-like spatio-functional organization in a topographic language model/images/5234dde00aa98752ac78f93920d1da581c4f73380ac7aa69dd4db0829d1c80b3.jpg)

![68ea9d20e7a8fad1036540eb6f5cb4e01eb0fdf40abd987ed4b1604969e9eeb3.jpg](../iclr_results/95_TopoLM_ brain-like spatio-functional organization in a topographic language model/images/68ea9d20e7a8fad1036540eb6f5cb4e01eb0fdf40abd987ed4b1604969e9eeb3.jpg)

![7d3e7ada586f0d9540ff7183a782bc1ba9de93e73b8b51575fe496f1880964a2.jpg](../iclr_results/95_TopoLM_ brain-like spatio-functional organization in a topographic language model/images/7d3e7ada586f0d9540ff7183a782bc1ba9de93e73b8b51575fe496f1880964a2.jpg)

![85522b2f16878e37ac6f261b1fae2ab82a34db70cb81336c34129fd7f3ad041c.jpg](../iclr_results/95_TopoLM_ brain-like spatio-functional organization in a topographic language model/images/85522b2f16878e37ac6f261b1fae2ab82a34db70cb81336c34129fd7f3ad041c.jpg)

![95b6acd0d22d846fab6a8e4382935932f6e079b87cebba8fff2e22a03c473b1b.jpg](../iclr_results/95_TopoLM_ brain-like spatio-functional organization in a topographic language model/images/95b6acd0d22d846fab6a8e4382935932f6e079b87cebba8fff2e22a03c473b1b.jpg)

![a5422115bd35f14046e3f1eebabbaf0bc75407fc4803b458e5077b6b23fb4663.jpg](../iclr_results/95_TopoLM_ brain-like spatio-functional organization in a topographic language model/images/a5422115bd35f14046e3f1eebabbaf0bc75407fc4803b458e5077b6b23fb4663.jpg)

![c1cf71224a963f6011d3ac564617b4732aa460cdd1f2a4130d7a0f025e69c4fa.jpg](../iclr_results/95_TopoLM_ brain-like spatio-functional organization in a topographic language model/images/c1cf71224a963f6011d3ac564617b4732aa460cdd1f2a4130d7a0f025e69c4fa.jpg)

![c72c06ab5a23cbb16fd350cd86f0540ac3d64bb8fe5ab1e590fafa02016646a4.jpg](../iclr_results/95_TopoLM_ brain-like spatio-functional organization in a topographic language model/images/c72c06ab5a23cbb16fd350cd86f0540ac3d64bb8fe5ab1e590fafa02016646a4.jpg)

![e81369dd7a1e002237130e41e255511505f1265f254c590e872a7c62663c7fd0.jpg](../iclr_results/95_TopoLM_ brain-like spatio-functional organization in a topographic language model/images/e81369dd7a1e002237130e41e255511505f1265f254c590e872a7c62663c7fd0.jpg)

![e8850584438b99f99ae421fbaa5263f7fdcd25c0cce9f4b0b5a2d7f61f7f419d.jpg](../iclr_results/95_TopoLM_ brain-like spatio-functional organization in a topographic language model/images/e8850584438b99f99ae421fbaa5263f7fdcd25c0cce9f4b0b5a2d7f61f7f419d.jpg)

![fc82cc30988541be7c82b2e8ba5305e05560cd56c447a5de1812476d2f5aa6f5.jpg](../iclr_results/95_TopoLM_ brain-like spatio-functional organization in a topographic language model/images/fc82cc30988541be7c82b2e8ba5305e05560cd56c447a5de1812476d2f5aa6f5.jpg)

### Tables

![78e8c2dba063d5676b6fcf56dde85e156373923dcef5276e68436e69b6a403fa.jpg](../iclr_results/95_TopoLM_ brain-like spatio-functional organization in a topographic language model/tables/78e8c2dba063d5676b6fcf56dde85e156373923dcef5276e68436e69b6a403fa.jpg)

## Problem-Parameter-Free Federated Learning


### Images

![53eeb9bd9592b8b3f11458fad709e65c9ebb09c4dc51916c5423d611378bcf47.jpg](../iclr_results/96_Problem-Parameter-Free Federated Learning/images/53eeb9bd9592b8b3f11458fad709e65c9ebb09c4dc51916c5423d611378bcf47.jpg)

![745482e9753d696529ef74279f261ff8fceca6cd737f299e909c404e0a6932f1.jpg](../iclr_results/96_Problem-Parameter-Free Federated Learning/images/745482e9753d696529ef74279f261ff8fceca6cd737f299e909c404e0a6932f1.jpg)

![7e5651e469502be83a2bb5d807ecfbd28cc7f8fcc041d46f3f0e94a8402a0c9e.jpg](../iclr_results/96_Problem-Parameter-Free Federated Learning/images/7e5651e469502be83a2bb5d807ecfbd28cc7f8fcc041d46f3f0e94a8402a0c9e.jpg)

![96f4577ddae4d315c7d9b715bc2e959e2e59c6a7431fc8f52ba6e3063c0a78f3.jpg](../iclr_results/96_Problem-Parameter-Free Federated Learning/images/96f4577ddae4d315c7d9b715bc2e959e2e59c6a7431fc8f52ba6e3063c0a78f3.jpg)

![eaacde52e44bc2294ad1798a1520fbf29afa20e51e6ef8ef95c86ea3124f60b2.jpg](../iclr_results/96_Problem-Parameter-Free Federated Learning/images/eaacde52e44bc2294ad1798a1520fbf29afa20e51e6ef8ef95c86ea3124f60b2.jpg)

### Tables

![775118e3f449029f8912243711fbd0ce51a28b50a890f3e2969f2084f3de8a62.jpg](../iclr_results/96_Problem-Parameter-Free Federated Learning/tables/775118e3f449029f8912243711fbd0ce51a28b50a890f3e2969f2084f3de8a62.jpg)

## Latent Bayesian Optimization via Autoregressive Normalizing Flows


### Images

![126c430d332c036193c1c651a0e0f04523c5630563427f42be87c3bae335fcec.jpg](../iclr_results/97_Latent Bayesian Optimization via Autoregressive Normalizing Flows/images/126c430d332c036193c1c651a0e0f04523c5630563427f42be87c3bae335fcec.jpg)

![359bcdc682c3e3f0ae841f0f5e26afa1c77aac11f8be4cf55aa5bd7de34b6062.jpg](../iclr_results/97_Latent Bayesian Optimization via Autoregressive Normalizing Flows/images/359bcdc682c3e3f0ae841f0f5e26afa1c77aac11f8be4cf55aa5bd7de34b6062.jpg)

![37f909f7f2cc736a9af8a7f6c4366df1d93afb7e7048d13e07b17776bdddac37.jpg](../iclr_results/97_Latent Bayesian Optimization via Autoregressive Normalizing Flows/images/37f909f7f2cc736a9af8a7f6c4366df1d93afb7e7048d13e07b17776bdddac37.jpg)

![5b4b124f0f90a52b006d60b5121cbf106e022978ad971ee5b80ac3a6c2dbebef.jpg](../iclr_results/97_Latent Bayesian Optimization via Autoregressive Normalizing Flows/images/5b4b124f0f90a52b006d60b5121cbf106e022978ad971ee5b80ac3a6c2dbebef.jpg)

![62c617f621c1f3d54c0efe0bd0d14dbcf0dea978232614418e7c355e48a473c5.jpg](../iclr_results/97_Latent Bayesian Optimization via Autoregressive Normalizing Flows/images/62c617f621c1f3d54c0efe0bd0d14dbcf0dea978232614418e7c355e48a473c5.jpg)

![6e7594f8028804a142c82016132fc951a5fb50f839ca2a3f48058998a4259597.jpg](../iclr_results/97_Latent Bayesian Optimization via Autoregressive Normalizing Flows/images/6e7594f8028804a142c82016132fc951a5fb50f839ca2a3f48058998a4259597.jpg)

![71bbddebaf48e0ff4e868a56f7825a87aaf3aa1280044ebd958299ddf48f70a7.jpg](../iclr_results/97_Latent Bayesian Optimization via Autoregressive Normalizing Flows/images/71bbddebaf48e0ff4e868a56f7825a87aaf3aa1280044ebd958299ddf48f70a7.jpg)

![a8518dbe1d44edac4d16a8e6339d73aef3b5b2c4d3d0a299653455622e1be82f.jpg](../iclr_results/97_Latent Bayesian Optimization via Autoregressive Normalizing Flows/images/a8518dbe1d44edac4d16a8e6339d73aef3b5b2c4d3d0a299653455622e1be82f.jpg)

![ba9a53123119ad014bdad749392a7ba918a8fa9710609e201fd0f4031ba36dd8.jpg](../iclr_results/97_Latent Bayesian Optimization via Autoregressive Normalizing Flows/images/ba9a53123119ad014bdad749392a7ba918a8fa9710609e201fd0f4031ba36dd8.jpg)

![c1828673545d26172795867d40d2f780b01621aa0d80358af706af7969e54715.jpg](../iclr_results/97_Latent Bayesian Optimization via Autoregressive Normalizing Flows/images/c1828673545d26172795867d40d2f780b01621aa0d80358af706af7969e54715.jpg)

![c23cb4ca6de011d0639d3324f0ca9831697b0cdde69ff43759dfac9867dff8c9.jpg](../iclr_results/97_Latent Bayesian Optimization via Autoregressive Normalizing Flows/images/c23cb4ca6de011d0639d3324f0ca9831697b0cdde69ff43759dfac9867dff8c9.jpg)

![d373e4327729cf4b47f7f0a85eaad0c12b56cdec7e5cc4e27c78fb5c7b822391.jpg](../iclr_results/97_Latent Bayesian Optimization via Autoregressive Normalizing Flows/images/d373e4327729cf4b47f7f0a85eaad0c12b56cdec7e5cc4e27c78fb5c7b822391.jpg)

![d45a014affdff3bec6867819ee267fc6ca2e5fe3ef72293964e7e0997165e740.jpg](../iclr_results/97_Latent Bayesian Optimization via Autoregressive Normalizing Flows/images/d45a014affdff3bec6867819ee267fc6ca2e5fe3ef72293964e7e0997165e740.jpg)

![fc266a74ae1db415d50585237d3a8300d45e8e7395dc245c7dbeabb8b663016d.jpg](../iclr_results/97_Latent Bayesian Optimization via Autoregressive Normalizing Flows/images/fc266a74ae1db415d50585237d3a8300d45e8e7395dc245c7dbeabb8b663016d.jpg)

### Tables

![16e64dc0c8e3119e7cd98a477b629799a13a7cd74cb866dd9f9fcd72a5b01840.jpg](../iclr_results/97_Latent Bayesian Optimization via Autoregressive Normalizing Flows/tables/16e64dc0c8e3119e7cd98a477b629799a13a7cd74cb866dd9f9fcd72a5b01840.jpg)

![28a90e12ca12a8d3f5bb7dd1b2314fddc5c19af948bfa6609515f628c08802bf.jpg](../iclr_results/97_Latent Bayesian Optimization via Autoregressive Normalizing Flows/tables/28a90e12ca12a8d3f5bb7dd1b2314fddc5c19af948bfa6609515f628c08802bf.jpg)

![2b41eb41cdd74b24a0fe9127d991b2c2172efceedec071a54386b525e15c5b3a.jpg](../iclr_results/97_Latent Bayesian Optimization via Autoregressive Normalizing Flows/tables/2b41eb41cdd74b24a0fe9127d991b2c2172efceedec071a54386b525e15c5b3a.jpg)

![32e430ce680676f4a7f462affbccd96fe2f162e059a3fac1ef0cd03cb1bc4415.jpg](../iclr_results/97_Latent Bayesian Optimization via Autoregressive Normalizing Flows/tables/32e430ce680676f4a7f462affbccd96fe2f162e059a3fac1ef0cd03cb1bc4415.jpg)

![33554761a1d034e47ce074878e0eea37d00948b0647a9702c007dd9b01abe589.jpg](../iclr_results/97_Latent Bayesian Optimization via Autoregressive Normalizing Flows/tables/33554761a1d034e47ce074878e0eea37d00948b0647a9702c007dd9b01abe589.jpg)

![5021c133730876a726f8cd587ecdbaadd6fa8e8dc080e6aef02937ae6bb910e9.jpg](../iclr_results/97_Latent Bayesian Optimization via Autoregressive Normalizing Flows/tables/5021c133730876a726f8cd587ecdbaadd6fa8e8dc080e6aef02937ae6bb910e9.jpg)

![6fae10cf80dff11a33b7249b048dc79398ea3af89481399570733fef0c4913ce.jpg](../iclr_results/97_Latent Bayesian Optimization via Autoregressive Normalizing Flows/tables/6fae10cf80dff11a33b7249b048dc79398ea3af89481399570733fef0c4913ce.jpg)

![8bcd731dfdb0969b5c709d3eb926ff83ad403040528905cdae626a66fb5e1ec7.jpg](../iclr_results/97_Latent Bayesian Optimization via Autoregressive Normalizing Flows/tables/8bcd731dfdb0969b5c709d3eb926ff83ad403040528905cdae626a66fb5e1ec7.jpg)

![aa59ba804ee0aecdb5836ba9818ce886fe611110cce4cac425a609f202c5dafe.jpg](../iclr_results/97_Latent Bayesian Optimization via Autoregressive Normalizing Flows/tables/aa59ba804ee0aecdb5836ba9818ce886fe611110cce4cac425a609f202c5dafe.jpg)

## BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions


### Images

![070d2b4fc03922c72ff3f68c9eac244e53d2a198ec5bd0a03d570105813312c9.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/070d2b4fc03922c72ff3f68c9eac244e53d2a198ec5bd0a03d570105813312c9.jpg)

![0fe30d92c15b0d42be5a965f4807c9b40be967ed212e1f5180e33fb4eb3d2179.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/0fe30d92c15b0d42be5a965f4807c9b40be967ed212e1f5180e33fb4eb3d2179.jpg)

![1e84d9fe4dfa2cf96bd5cce2c2b19136aefe12e5be23dfbdad72d12e0f4cf194.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/1e84d9fe4dfa2cf96bd5cce2c2b19136aefe12e5be23dfbdad72d12e0f4cf194.jpg)

![243625518fb67b5a049b479e5e684923d5bf5ac9bf67e34bca89a50e9f6e0eb2.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/243625518fb67b5a049b479e5e684923d5bf5ac9bf67e34bca89a50e9f6e0eb2.jpg)

![2f9b17437e0f8750faf0a88b32da337ecd9723af2c1882947e38321390d34468.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/2f9b17437e0f8750faf0a88b32da337ecd9723af2c1882947e38321390d34468.jpg)

![35a4b80fb6a2bdbace05ec97a81cfc92fa8d7e2a4b6bd907fab227389122d810.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/35a4b80fb6a2bdbace05ec97a81cfc92fa8d7e2a4b6bd907fab227389122d810.jpg)

![3ee10509f7a518f7bbcbd278ac22134dc403f30834b2f8b379e6b375aa5b202e.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/3ee10509f7a518f7bbcbd278ac22134dc403f30834b2f8b379e6b375aa5b202e.jpg)

![3f2383ebce9df5d8aefe7cc20d657bcfe922513dc2a4a5f07d277c71495008a5.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/3f2383ebce9df5d8aefe7cc20d657bcfe922513dc2a4a5f07d277c71495008a5.jpg)

![4091280c27c43fca7eb360a509524d62de82b1d0d6e3e6f37fb0bc0041491558.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/4091280c27c43fca7eb360a509524d62de82b1d0d6e3e6f37fb0bc0041491558.jpg)

![47e46e6b762114c70d87711412091fb7b74a50d5710cc521def2e2d6eea87a2e.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/47e46e6b762114c70d87711412091fb7b74a50d5710cc521def2e2d6eea87a2e.jpg)

![4d861061719f34443e14ec7eb7eeb651b98325b654706aa55da8e512ad365eb0.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/4d861061719f34443e14ec7eb7eeb651b98325b654706aa55da8e512ad365eb0.jpg)

![60d073231fa3c3dd7f72b7e5aded223cc9cff364133c774a627e8c9ac1b764ad.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/60d073231fa3c3dd7f72b7e5aded223cc9cff364133c774a627e8c9ac1b764ad.jpg)

![689b1ed2a9334e2fe38f4692af35421af17a27318b2ec0959db5a0bc46e98a7a.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/689b1ed2a9334e2fe38f4692af35421af17a27318b2ec0959db5a0bc46e98a7a.jpg)

![72cd2e23fb39e7c3317f6d61905e022f6a1c850c72527abf9b169f11a7e86252.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/72cd2e23fb39e7c3317f6d61905e022f6a1c850c72527abf9b169f11a7e86252.jpg)

![7dc29537054d1f574c4a69c65df877ac81ef94c00633afe1fff41946a44d1027.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/7dc29537054d1f574c4a69c65df877ac81ef94c00633afe1fff41946a44d1027.jpg)

![88d5a01b03ffd4b8edbc34d421475cd0ab526b80c9d68db87f03067296b209bd.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/88d5a01b03ffd4b8edbc34d421475cd0ab526b80c9d68db87f03067296b209bd.jpg)

![8a02ddd96949c670259913c4b8ab529db3c54b29de0365a7e9f2c034d42ec768.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/8a02ddd96949c670259913c4b8ab529db3c54b29de0365a7e9f2c034d42ec768.jpg)

![8ecbaafdfef69d43d4b2cf3289d2e2e2b9081441a88a5159010b1332b2b9882b.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/8ecbaafdfef69d43d4b2cf3289d2e2e2b9081441a88a5159010b1332b2b9882b.jpg)

![91fcb31124157f70b168caf4788cfa24416965d0a43ad75698ff3af42d9385f1.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/91fcb31124157f70b168caf4788cfa24416965d0a43ad75698ff3af42d9385f1.jpg)

![93e40f6e3985ea672f29b393e6ccc49042ad336af1506c2b764e0d80f324bccc.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/93e40f6e3985ea672f29b393e6ccc49042ad336af1506c2b764e0d80f324bccc.jpg)

![a34d84a0ec1a7b835447c919ba431ad8bcf2ba9ec197d6f249f63403a846d5e1.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/a34d84a0ec1a7b835447c919ba431ad8bcf2ba9ec197d6f249f63403a846d5e1.jpg)

![a38d4ea8df0fc08b80e2de4c34ca8abb62ca8c785a4411dd740011be03095fba.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/a38d4ea8df0fc08b80e2de4c34ca8abb62ca8c785a4411dd740011be03095fba.jpg)

![b22ece808264c02e7db743a7b47a4fe3f7bacf7a523ba3977f5ef87269b3d206.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/b22ece808264c02e7db743a7b47a4fe3f7bacf7a523ba3977f5ef87269b3d206.jpg)

![b400f60222a36fe3fb64f6982e225fc6fc3f430d3f15c2c515ba110c2b3ea9c2.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/b400f60222a36fe3fb64f6982e225fc6fc3f430d3f15c2c515ba110c2b3ea9c2.jpg)

![b4d156d87a17fac976991ee3959a4441da44822eba6fa56b10a7dfd18569be39.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/b4d156d87a17fac976991ee3959a4441da44822eba6fa56b10a7dfd18569be39.jpg)

![c6c6863d1b8413b58cea27928fea2b7dad13e2ab3fa58d0f1678bff496c52797.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/c6c6863d1b8413b58cea27928fea2b7dad13e2ab3fa58d0f1678bff496c52797.jpg)

![cbd837a55c90f90b00d72bc3d1317afaebfbf9fdff7b9d7d1552860c08e25a4e.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/cbd837a55c90f90b00d72bc3d1317afaebfbf9fdff7b9d7d1552860c08e25a4e.jpg)

![d4953b32e68a9491cde1f94806bb5a63d8a39dc524dfa8a3d13cace9f8b3a078.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/d4953b32e68a9491cde1f94806bb5a63d8a39dc524dfa8a3d13cace9f8b3a078.jpg)

![d7a33f22cc6b0f7f67dcec3bceacf9eebbaf1d5301eea090739d9e654498e16f.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/d7a33f22cc6b0f7f67dcec3bceacf9eebbaf1d5301eea090739d9e654498e16f.jpg)

![e21b8be3ef263d9224e646e19a458405d86439b0cfd078f22500c1376603c6fa.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/e21b8be3ef263d9224e646e19a458405d86439b0cfd078f22500c1376603c6fa.jpg)

![e3536c2fd466cd00bd910b1111f3f2ec85a090e6349d81deb53e7ffcdb6afc20.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/e3536c2fd466cd00bd910b1111f3f2ec85a090e6349d81deb53e7ffcdb6afc20.jpg)

![edc50634dbe455b9a6e4b84665fe5811d44638639c5490aaa7aba6a9df7db67b.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/edc50634dbe455b9a6e4b84665fe5811d44638639c5490aaa7aba6a9df7db67b.jpg)

![f4e6b345ea2c2f61e5044824688b3536d37f8fa6f4df460beb9ac95e538b4e84.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/f4e6b345ea2c2f61e5044824688b3536d37f8fa6f4df460beb9ac95e538b4e84.jpg)

![f6dce287c56bb177d2fa20af6def9544dcbbd38962fa116ced00e709675dc180.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/f6dce287c56bb177d2fa20af6def9544dcbbd38962fa116ced00e709675dc180.jpg)

![f745991d578b2babeccee95a10450a4e9ea0f895f052bfe15defa2530bfd5d28.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/images/f745991d578b2babeccee95a10450a4e9ea0f895f052bfe15defa2530bfd5d28.jpg)

### Tables

![0b61a9360bc716b25bba6f78fceec34bfc9fba8e3e641b05a6a4318171db96e2.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/tables/0b61a9360bc716b25bba6f78fceec34bfc9fba8e3e641b05a6a4318171db96e2.jpg)

![117f3734cf78786af99ccda4a40e4399bd80ed4308e050cf2a52ba182bfd417b.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/tables/117f3734cf78786af99ccda4a40e4399bd80ed4308e050cf2a52ba182bfd417b.jpg)

![1d74fa8e63a57f462ee9b1b516b79234b81f079c498cb3a4708bf7486c78cc72.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/tables/1d74fa8e63a57f462ee9b1b516b79234b81f079c498cb3a4708bf7486c78cc72.jpg)

![21b3bc47689a97b63a101d85f27b0bdf2860f517350b7acb7ed14d6a1b796305.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/tables/21b3bc47689a97b63a101d85f27b0bdf2860f517350b7acb7ed14d6a1b796305.jpg)

![25a92f872126ccbab75b4c592b7ddcd95b2b29ed205ce522b50d82d891b3374e.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/tables/25a92f872126ccbab75b4c592b7ddcd95b2b29ed205ce522b50d82d891b3374e.jpg)

![32240e7169bdf4a735ee74f79d89d7e0307f8f982853b972a33a6c71fac67fd3.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/tables/32240e7169bdf4a735ee74f79d89d7e0307f8f982853b972a33a6c71fac67fd3.jpg)

![48b38942e34649c8e516a25f890ce0f3d091ae5846d7e69ef443fce0c49c24ab.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/tables/48b38942e34649c8e516a25f890ce0f3d091ae5846d7e69ef443fce0c49c24ab.jpg)

![5cc676523b61ae3b0aba9c85fb9ff331d074ff91393c129348e7268d42d038f8.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/tables/5cc676523b61ae3b0aba9c85fb9ff331d074ff91393c129348e7268d42d038f8.jpg)

![5d8a53e1fa105471d32ff76ea5fb4d598b49d12284fecd89dc7b07ea41c20663.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/tables/5d8a53e1fa105471d32ff76ea5fb4d598b49d12284fecd89dc7b07ea41c20663.jpg)

![82a07741dbc712cfce7ee3d7106ae3cf2e26b346094d5ea9a784643a8f071463.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/tables/82a07741dbc712cfce7ee3d7106ae3cf2e26b346094d5ea9a784643a8f071463.jpg)

![bf0fb93748ea7f947e0e7ea159927121aa22bea73c012ba29d630d880b1d4353.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/tables/bf0fb93748ea7f947e0e7ea159927121aa22bea73c012ba29d630d880b1d4353.jpg)

![c6be5ba63eb21e3bfa908721d83de5dfb27a260d0645a47d9b02afd7966839c4.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/tables/c6be5ba63eb21e3bfa908721d83de5dfb27a260d0645a47d9b02afd7966839c4.jpg)

![f2bb31911f247891f4740c4f3aa276b51738fa91bbd64113596f2c928d84dd49.jpg](../iclr_results/98_BigCodeBench_ Benchmarking Code Generation with Diverse Function Calls and Complex Instructions/tables/f2bb31911f247891f4740c4f3aa276b51738fa91bbd64113596f2c928d84dd49.jpg)

## ReGenesis: LLMs can Grow into Reasoning Generalists via Self-Improvement


### Images

![2f01f33b7146fc324812683c8e0131c21bacb4f9c4594022e4c1fe912f2c763b.jpg](../iclr_results/99_ReGenesis_ LLMs can Grow into Reasoning Generalists via Self-Improvement/images/2f01f33b7146fc324812683c8e0131c21bacb4f9c4594022e4c1fe912f2c763b.jpg)

![48f022e33dc568c818417914d89e8bfb611e136bf5e434d9d6f0b8333cfbe37e.jpg](../iclr_results/99_ReGenesis_ LLMs can Grow into Reasoning Generalists via Self-Improvement/images/48f022e33dc568c818417914d89e8bfb611e136bf5e434d9d6f0b8333cfbe37e.jpg)

![49d7d6a372b1689c683495b66f8e4f4fd2baad6d3b6e6b207974bdb5d781bdb2.jpg](../iclr_results/99_ReGenesis_ LLMs can Grow into Reasoning Generalists via Self-Improvement/images/49d7d6a372b1689c683495b66f8e4f4fd2baad6d3b6e6b207974bdb5d781bdb2.jpg)

![8823befcc3b5a47ddb927963938a3458a3e6de0c9db00df6f3d523d6355f7b01.jpg](../iclr_results/99_ReGenesis_ LLMs can Grow into Reasoning Generalists via Self-Improvement/images/8823befcc3b5a47ddb927963938a3458a3e6de0c9db00df6f3d523d6355f7b01.jpg)

### Tables

![0e27685c865ea823009b0c9b0827c58f8af29a91d3d1f33e96864d0f7d40c1a9.jpg](../iclr_results/99_ReGenesis_ LLMs can Grow into Reasoning Generalists via Self-Improvement/tables/0e27685c865ea823009b0c9b0827c58f8af29a91d3d1f33e96864d0f7d40c1a9.jpg)

![38b9dcd76d9c2bbfcc6b2ff44fff73aa66798c7c3bcfe88a94b1e6730147e7ed.jpg](../iclr_results/99_ReGenesis_ LLMs can Grow into Reasoning Generalists via Self-Improvement/tables/38b9dcd76d9c2bbfcc6b2ff44fff73aa66798c7c3bcfe88a94b1e6730147e7ed.jpg)

![47079b7bdca932b91fe300097d1e705d926a7fff089b491a7ed03b76a4caf9f8.jpg](../iclr_results/99_ReGenesis_ LLMs can Grow into Reasoning Generalists via Self-Improvement/tables/47079b7bdca932b91fe300097d1e705d926a7fff089b491a7ed03b76a4caf9f8.jpg)

![532dd51a823c71aa22c572076da0ccc629b437b73c5b75229eb1664495b714c1.jpg](../iclr_results/99_ReGenesis_ LLMs can Grow into Reasoning Generalists via Self-Improvement/tables/532dd51a823c71aa22c572076da0ccc629b437b73c5b75229eb1664495b714c1.jpg)

![5c7e8d62eadaa74eba371f63c134ddf91181866308ae63e350c4c7849efe703b.jpg](../iclr_results/99_ReGenesis_ LLMs can Grow into Reasoning Generalists via Self-Improvement/tables/5c7e8d62eadaa74eba371f63c134ddf91181866308ae63e350c4c7849efe703b.jpg)

![6d8a40f5193ee0ffb14729bbdd34b1ce33143967a38b4447519d9ade604e7a0f.jpg](../iclr_results/99_ReGenesis_ LLMs can Grow into Reasoning Generalists via Self-Improvement/tables/6d8a40f5193ee0ffb14729bbdd34b1ce33143967a38b4447519d9ade604e7a0f.jpg)

![7d630679ef1afedd920b3e332f640ca89b958e96e2aec7c9808f26f10fa442e1.jpg](../iclr_results/99_ReGenesis_ LLMs can Grow into Reasoning Generalists via Self-Improvement/tables/7d630679ef1afedd920b3e332f640ca89b958e96e2aec7c9808f26f10fa442e1.jpg)

![9c70ad5120c490f3a112f7edc39b2a625c9ae698dfcdbbe6f4c338f8cda4edc7.jpg](../iclr_results/99_ReGenesis_ LLMs can Grow into Reasoning Generalists via Self-Improvement/tables/9c70ad5120c490f3a112f7edc39b2a625c9ae698dfcdbbe6f4c338f8cda4edc7.jpg)

![a3c7a6090d3e0deed3b3c432dd0afbb209c918f402ec4007b5a97c8f225e59f2.jpg](../iclr_results/99_ReGenesis_ LLMs can Grow into Reasoning Generalists via Self-Improvement/tables/a3c7a6090d3e0deed3b3c432dd0afbb209c918f402ec4007b5a97c8f225e59f2.jpg)

![bd60391e7856aed83eeddb5eacfb2756c4ffdcfc4f2a12d25b19dc24df4e57bc.jpg](../iclr_results/99_ReGenesis_ LLMs can Grow into Reasoning Generalists via Self-Improvement/tables/bd60391e7856aed83eeddb5eacfb2756c4ffdcfc4f2a12d25b19dc24df4e57bc.jpg)

![bf8190f2ace9c1f1e0c63b0cabedf25697df851c9401deca503d888c52e9949d.jpg](../iclr_results/99_ReGenesis_ LLMs can Grow into Reasoning Generalists via Self-Improvement/tables/bf8190f2ace9c1f1e0c63b0cabedf25697df851c9401deca503d888c52e9949d.jpg)

![eff8ebbab219404ea7acf8097311e7a6ba633c15cf5db0b6baac27a6c706b392.jpg](../iclr_results/99_ReGenesis_ LLMs can Grow into Reasoning Generalists via Self-Improvement/tables/eff8ebbab219404ea7acf8097311e7a6ba633c15cf5db0b6baac27a6c706b392.jpg)

## LaMPlace: Learning to Optimize Cross-Stage Metrics in Macro Placement


### Images

![1e4e0841d969b2f4dd091bf47289c8863e677eb49783e5f087ab964ef101f58a.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/images/1e4e0841d969b2f4dd091bf47289c8863e677eb49783e5f087ab964ef101f58a.jpg)

![335af537744701041586899b5dff93426b6ee96a3e43236c7c620301806dcee2.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/images/335af537744701041586899b5dff93426b6ee96a3e43236c7c620301806dcee2.jpg)

![4e1f36994a7eaae3dc53b77c5c3f68c5810bd159e919beeb60b56016e3e465f9.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/images/4e1f36994a7eaae3dc53b77c5c3f68c5810bd159e919beeb60b56016e3e465f9.jpg)

![5eb32780afb1eb50f513a85081fde820c76f5ec35ad5dbfbbdbfdd1c6f6fe10c.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/images/5eb32780afb1eb50f513a85081fde820c76f5ec35ad5dbfbbdbfdd1c6f6fe10c.jpg)

![6cecc2b876469d54f51efd084c9a198f4cca60513942400b99a624aea21d458c.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/images/6cecc2b876469d54f51efd084c9a198f4cca60513942400b99a624aea21d458c.jpg)

![743be71ba69dd42490b2d1ccceb85672174fc9f4f37264f92054550ba50a59c6.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/images/743be71ba69dd42490b2d1ccceb85672174fc9f4f37264f92054550ba50a59c6.jpg)

![794a4ee534f3f51300f4b5a09eec63887fe7bb5b90031d21767e9dacfec0a9bb.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/images/794a4ee534f3f51300f4b5a09eec63887fe7bb5b90031d21767e9dacfec0a9bb.jpg)

![8245d68a1ed10d97303b07c5f22eea8c5203279b5a9fa7d517c8e063652f5895.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/images/8245d68a1ed10d97303b07c5f22eea8c5203279b5a9fa7d517c8e063652f5895.jpg)

![8ba3f90dd46cf7aab48826774312c4888bd73e1303d0b45ccd0aa60140dcb07e.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/images/8ba3f90dd46cf7aab48826774312c4888bd73e1303d0b45ccd0aa60140dcb07e.jpg)

![8d81b13f07cc5938f57c0c348321bb44709de18dba3bd435b7ff1f7878f14ab2.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/images/8d81b13f07cc5938f57c0c348321bb44709de18dba3bd435b7ff1f7878f14ab2.jpg)

![905ef3f561b28c73c63f577d3038e9d548a805940e21db3b6d2956a52954e180.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/images/905ef3f561b28c73c63f577d3038e9d548a805940e21db3b6d2956a52954e180.jpg)

![9d7af818b485fc07a7511b22d99ddb96c4b18f3305973a8fe4034ffcc78ed19c.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/images/9d7af818b485fc07a7511b22d99ddb96c4b18f3305973a8fe4034ffcc78ed19c.jpg)

![9ee91fe5a4107ea337e8260fbb781be5693782a4a5e3f3ec66b6cc840f4c08fc.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/images/9ee91fe5a4107ea337e8260fbb781be5693782a4a5e3f3ec66b6cc840f4c08fc.jpg)

![a18dd5adb2d89f21e714c146835e521d1162c65a213cbcae8fc4e576bc622936.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/images/a18dd5adb2d89f21e714c146835e521d1162c65a213cbcae8fc4e576bc622936.jpg)

![a75571030bd804461fe33bf16e70dc7f71d5b35a2b3dbbe9ae062bca9896bc6f.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/images/a75571030bd804461fe33bf16e70dc7f71d5b35a2b3dbbe9ae062bca9896bc6f.jpg)

![af27aadd54701b5965665e0e2ad90d6835303adac59a2e2a9ec757c2e6724d62.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/images/af27aadd54701b5965665e0e2ad90d6835303adac59a2e2a9ec757c2e6724d62.jpg)

![c649560f8dcb248515bcf18d5a82eb824aa30c4d72b290d9ae348ca03be34f3f.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/images/c649560f8dcb248515bcf18d5a82eb824aa30c4d72b290d9ae348ca03be34f3f.jpg)

![dd47ccbcf3857fe6116ff84978bf0d286bae8259a4bd3e1795e225d982b2b007.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/images/dd47ccbcf3857fe6116ff84978bf0d286bae8259a4bd3e1795e225d982b2b007.jpg)

![f3ea0f03a9450c835af8c2cc456f9a29463ec781b66cd97358fa5df06fb3a727.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/images/f3ea0f03a9450c835af8c2cc456f9a29463ec781b66cd97358fa5df06fb3a727.jpg)

### Tables

![1c7b0c16093b8638b5069fc21f0897206ff79ba6a0f1be2f91f0654032326526.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/tables/1c7b0c16093b8638b5069fc21f0897206ff79ba6a0f1be2f91f0654032326526.jpg)

![415a4c2763879f2a7f85f097058af8a5584b3847b5245f16631c254b06b326e6.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/tables/415a4c2763879f2a7f85f097058af8a5584b3847b5245f16631c254b06b326e6.jpg)

![557b6bc6cdf455d6fb111eae6316591fbb07cb946914887da5abf51535b925b0.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/tables/557b6bc6cdf455d6fb111eae6316591fbb07cb946914887da5abf51535b925b0.jpg)

![5a22e19f6afb5315115bedfd78a2c62467fbd21e16a0d1aa8ac40f15db03552e.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/tables/5a22e19f6afb5315115bedfd78a2c62467fbd21e16a0d1aa8ac40f15db03552e.jpg)

![643750239145e0f9021a71bfe4123c2f1dc62486267c3ff708a11e8d4358a007.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/tables/643750239145e0f9021a71bfe4123c2f1dc62486267c3ff708a11e8d4358a007.jpg)

![85ebe18cd6d08ddbfd131cab9718a14843fed97836cba49672fc9486279e8ca5.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/tables/85ebe18cd6d08ddbfd131cab9718a14843fed97836cba49672fc9486279e8ca5.jpg)

![aefd66b6eb0bb68cc95031f738f8dbf08ccf2687681e57884199188be3c5e9ca.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/tables/aefd66b6eb0bb68cc95031f738f8dbf08ccf2687681e57884199188be3c5e9ca.jpg)

![b81cf60fdc4d91df3c805d2df5f6936a07f8f00aa5ff9d861a8829f721375fa4.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/tables/b81cf60fdc4d91df3c805d2df5f6936a07f8f00aa5ff9d861a8829f721375fa4.jpg)

![bd9c62a3404de0aa96168e0d4198e1a16a56eb50b40f6cbccb66da4a298fe754.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/tables/bd9c62a3404de0aa96168e0d4198e1a16a56eb50b40f6cbccb66da4a298fe754.jpg)

![ce4da8603947fc1be3ddd7d6e893b2d2bf2405f6d79a4f2009cd7dad123baf13.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/tables/ce4da8603947fc1be3ddd7d6e893b2d2bf2405f6d79a4f2009cd7dad123baf13.jpg)

![da69ea88649769b8ca845dcf21752e79781600bb1bcb6d08af5eb2f03efb64e3.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/tables/da69ea88649769b8ca845dcf21752e79781600bb1bcb6d08af5eb2f03efb64e3.jpg)

![e0746e0b0fd2b086748e77f4e16a4bfe286d0f81aa4182d853a7a6a0ac213ec1.jpg](../iclr_results/100_LaMPlace_ Learning to Optimize Cross-Stage Metrics in Macro Placement/tables/e0746e0b0fd2b086748e77f4e16a4bfe286d0f81aa4182d853a7a6a0ac213ec1.jpg)

## MOS: Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection


### Images

![285b0addba6ee32d0564594578147a9bbf7887ede77e18f6b12a139dff6b40f4.jpg](../iclr_results/101_MOS_ Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection/images/285b0addba6ee32d0564594578147a9bbf7887ede77e18f6b12a139dff6b40f4.jpg)

![402962f9df0df4751ef7c911baa6de21bbbdbe4e980dccb21b070ef5ba28642b.jpg](../iclr_results/101_MOS_ Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection/images/402962f9df0df4751ef7c911baa6de21bbbdbe4e980dccb21b070ef5ba28642b.jpg)

![512dbbaa327d8e9cd3e892c084b41aeaf2fdf475df1f470ddac01c6b2c138293.jpg](../iclr_results/101_MOS_ Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection/images/512dbbaa327d8e9cd3e892c084b41aeaf2fdf475df1f470ddac01c6b2c138293.jpg)

![67a358a643f52ffa288821eb5bf30e869509579344f5b10f8cec8949f143a97b.jpg](../iclr_results/101_MOS_ Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection/images/67a358a643f52ffa288821eb5bf30e869509579344f5b10f8cec8949f143a97b.jpg)

![7b7be5d769f5526f4189d45a6f5ad005b456a2ab30449eae1a506d4b092c2f66.jpg](../iclr_results/101_MOS_ Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection/images/7b7be5d769f5526f4189d45a6f5ad005b456a2ab30449eae1a506d4b092c2f66.jpg)

![8156a1674ebebc0fd6d2e83c78905bf9fec45296259602dc6aad393cfd0eafc0.jpg](../iclr_results/101_MOS_ Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection/images/8156a1674ebebc0fd6d2e83c78905bf9fec45296259602dc6aad393cfd0eafc0.jpg)

![cdf407f2232db8ad80f0bde6989445a5948a55f491295fb4b3b727462b79eea9.jpg](../iclr_results/101_MOS_ Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection/images/cdf407f2232db8ad80f0bde6989445a5948a55f491295fb4b3b727462b79eea9.jpg)

![d7630a527c47fc1fb3a14d475b8327cebae16d903cc5bff1f5291c261c4cd717.jpg](../iclr_results/101_MOS_ Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection/images/d7630a527c47fc1fb3a14d475b8327cebae16d903cc5bff1f5291c261c4cd717.jpg)

![eb2d2cdf18a6663bb53b27174645a3d10e4d90344c5112dc1590b903878c5b3b.jpg](../iclr_results/101_MOS_ Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection/images/eb2d2cdf18a6663bb53b27174645a3d10e4d90344c5112dc1590b903878c5b3b.jpg)

### Tables

![015e84de1d78fabb56f3c1c229dfe2b9d7eaa004c876fdd92d0468492e6f7178.jpg](../iclr_results/101_MOS_ Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection/tables/015e84de1d78fabb56f3c1c229dfe2b9d7eaa004c876fdd92d0468492e6f7178.jpg)

![17809eccd331fe73271cb9586a1db52766ffb46696f72905104b2615156f27cf.jpg](../iclr_results/101_MOS_ Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection/tables/17809eccd331fe73271cb9586a1db52766ffb46696f72905104b2615156f27cf.jpg)

![1f08690124203626f5d7704998f3c2ce8888e54b378b7ebb566e296a1a663543.jpg](../iclr_results/101_MOS_ Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection/tables/1f08690124203626f5d7704998f3c2ce8888e54b378b7ebb566e296a1a663543.jpg)

![223c5260c32b8c5a263674c94d2676e7a2100c3eeab17f7d2bf2eaa7008faa4c.jpg](../iclr_results/101_MOS_ Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection/tables/223c5260c32b8c5a263674c94d2676e7a2100c3eeab17f7d2bf2eaa7008faa4c.jpg)

![60eb49efbaf51618c76bfb55eacd95e81965d866c49adaa8262ea5fad8ede784.jpg](../iclr_results/101_MOS_ Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection/tables/60eb49efbaf51618c76bfb55eacd95e81965d866c49adaa8262ea5fad8ede784.jpg)

![8ec4ef848f99a843ec9f3ff4888a5bb99c4a94901de6ec0836038eef71e65881.jpg](../iclr_results/101_MOS_ Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection/tables/8ec4ef848f99a843ec9f3ff4888a5bb99c4a94901de6ec0836038eef71e65881.jpg)

![923fdb90139dd2ef41ddecf7fe4897fbf6d055b93735f52601acba52c1041853.jpg](../iclr_results/101_MOS_ Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection/tables/923fdb90139dd2ef41ddecf7fe4897fbf6d055b93735f52601acba52c1041853.jpg)

![964b05a2630000b97bc1e7733a2c32541cf7cf585b5ed8c648fc671197eac5b7.jpg](../iclr_results/101_MOS_ Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection/tables/964b05a2630000b97bc1e7733a2c32541cf7cf585b5ed8c648fc671197eac5b7.jpg)

![ba9ae7de130ffd636775ae8035e0daab5921ec1fb0cf3b33554be8e5a016c335.jpg](../iclr_results/101_MOS_ Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection/tables/ba9ae7de130ffd636775ae8035e0daab5921ec1fb0cf3b33554be8e5a016c335.jpg)

![bde8a7ddc1ab84d6c82350d5b8dd9616a0c1e8e74a79523d360e30dd56586d0f.jpg](../iclr_results/101_MOS_ Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection/tables/bde8a7ddc1ab84d6c82350d5b8dd9616a0c1e8e74a79523d360e30dd56586d0f.jpg)

![e0be8b1c462efdf6d622d5e077191e5ff22466218f50afc9904fcfbefa6f18d7.jpg](../iclr_results/101_MOS_ Model Synergy for Test-Time Adaptation on LiDAR-Based 3D Object Detection/tables/e0be8b1c462efdf6d622d5e077191e5ff22466218f50afc9904fcfbefa6f18d7.jpg)

## On Conformal Isometry of Grid Cells: Learning Distance-Preserving Position Embedding


### Images

![0715f89dda10db7d1fd6cecadf21e34756f05ebb304023685a02f3bad75e5f7f.jpg](../iclr_results/102_On Conformal Isometry of Grid Cells_ Learning Distance-Preserving Position Embedding/images/0715f89dda10db7d1fd6cecadf21e34756f05ebb304023685a02f3bad75e5f7f.jpg)

![19b64c8e23acdf71ccefd3436e85464fa2b68a73606a257dd4fda3243fa9399f.jpg](../iclr_results/102_On Conformal Isometry of Grid Cells_ Learning Distance-Preserving Position Embedding/images/19b64c8e23acdf71ccefd3436e85464fa2b68a73606a257dd4fda3243fa9399f.jpg)

![1afc3c49d11aa69871630e03ffbb21fd0f76a200a19608bfb86ccd10238d4d5e.jpg](../iclr_results/102_On Conformal Isometry of Grid Cells_ Learning Distance-Preserving Position Embedding/images/1afc3c49d11aa69871630e03ffbb21fd0f76a200a19608bfb86ccd10238d4d5e.jpg)

![2ebae99cc4fc713e87af609476135dac5a216fc0cd4936954aedd3ae20cee1c1.jpg](../iclr_results/102_On Conformal Isometry of Grid Cells_ Learning Distance-Preserving Position Embedding/images/2ebae99cc4fc713e87af609476135dac5a216fc0cd4936954aedd3ae20cee1c1.jpg)

![318f96bb08b45d7eb03e275303ec4427aaf0f25975944f278310f6c7c9891f0c.jpg](../iclr_results/102_On Conformal Isometry of Grid Cells_ Learning Distance-Preserving Position Embedding/images/318f96bb08b45d7eb03e275303ec4427aaf0f25975944f278310f6c7c9891f0c.jpg)

![4a52ae14349f42ae77bf998cd45deb948d0df9a9cab8b9be2b032f9aca6c607a.jpg](../iclr_results/102_On Conformal Isometry of Grid Cells_ Learning Distance-Preserving Position Embedding/images/4a52ae14349f42ae77bf998cd45deb948d0df9a9cab8b9be2b032f9aca6c607a.jpg)

![4e51153a04429ea97c94e90eaa7b9e7f8de173571ba60c93cf5af5723f133a78.jpg](../iclr_results/102_On Conformal Isometry of Grid Cells_ Learning Distance-Preserving Position Embedding/images/4e51153a04429ea97c94e90eaa7b9e7f8de173571ba60c93cf5af5723f133a78.jpg)

![82974c4203f9cfa74f27ddf5067f8becf3c4e83fb2a8dcb46257067034767f4a.jpg](../iclr_results/102_On Conformal Isometry of Grid Cells_ Learning Distance-Preserving Position Embedding/images/82974c4203f9cfa74f27ddf5067f8becf3c4e83fb2a8dcb46257067034767f4a.jpg)

![87bf7b8af37c82911593220457bb2442db1d0ed33b7537a4a300f85946aaed09.jpg](../iclr_results/102_On Conformal Isometry of Grid Cells_ Learning Distance-Preserving Position Embedding/images/87bf7b8af37c82911593220457bb2442db1d0ed33b7537a4a300f85946aaed09.jpg)

![9ccd69fb23122fb4c9ba9dd860da701dde1ea8fac6719ed0dc6696ca3ec1982e.jpg](../iclr_results/102_On Conformal Isometry of Grid Cells_ Learning Distance-Preserving Position Embedding/images/9ccd69fb23122fb4c9ba9dd860da701dde1ea8fac6719ed0dc6696ca3ec1982e.jpg)

![a658ddacdc4ddf34931121a6f385787b17a1cba2e7a066bd084ca6f461bc9e68.jpg](../iclr_results/102_On Conformal Isometry of Grid Cells_ Learning Distance-Preserving Position Embedding/images/a658ddacdc4ddf34931121a6f385787b17a1cba2e7a066bd084ca6f461bc9e68.jpg)

![bc821acbcab4679b2d3b6260d825f3cec9aa8d10ebbbf491f4a622a5596267c1.jpg](../iclr_results/102_On Conformal Isometry of Grid Cells_ Learning Distance-Preserving Position Embedding/images/bc821acbcab4679b2d3b6260d825f3cec9aa8d10ebbbf491f4a622a5596267c1.jpg)

![be6ec8daafd38c89eb5df3f969df10a4e3c9521d61f6b96b2483ad29f7b69c72.jpg](../iclr_results/102_On Conformal Isometry of Grid Cells_ Learning Distance-Preserving Position Embedding/images/be6ec8daafd38c89eb5df3f969df10a4e3c9521d61f6b96b2483ad29f7b69c72.jpg)

![d8ef0a861ed6a57f92257fef3451195fff37590bdfe2bc1eadb4485566593855.jpg](../iclr_results/102_On Conformal Isometry of Grid Cells_ Learning Distance-Preserving Position Embedding/images/d8ef0a861ed6a57f92257fef3451195fff37590bdfe2bc1eadb4485566593855.jpg)

![f1fc8a7c56de80b4da4dcb60efdfdf106084225fe623cbe0b609dc7de21b5847.jpg](../iclr_results/102_On Conformal Isometry of Grid Cells_ Learning Distance-Preserving Position Embedding/images/f1fc8a7c56de80b4da4dcb60efdfdf106084225fe623cbe0b609dc7de21b5847.jpg)

### Tables

![2e9e6b6ce0ac44dac444101273da19856f53fcb971aee1e8c60dd184f2a9c6d6.jpg](../iclr_results/102_On Conformal Isometry of Grid Cells_ Learning Distance-Preserving Position Embedding/tables/2e9e6b6ce0ac44dac444101273da19856f53fcb971aee1e8c60dd184f2a9c6d6.jpg)

![b03a5534051728ee3b8035814b344df2e5175c7a0ede0c2f4a4d33af099236ea.jpg](../iclr_results/102_On Conformal Isometry of Grid Cells_ Learning Distance-Preserving Position Embedding/tables/b03a5534051728ee3b8035814b344df2e5175c7a0ede0c2f4a4d33af099236ea.jpg)

## Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows


### Images

![0b64e93c7ff505ee39593dd90ba541dff695dd5042e948ee236ae54f45966278.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/0b64e93c7ff505ee39593dd90ba541dff695dd5042e948ee236ae54f45966278.jpg)

![0fa8bc0a15b15a91c5c418a39d77f4fc2f5d391e22211e55ec26d34c1c614d95.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/0fa8bc0a15b15a91c5c418a39d77f4fc2f5d391e22211e55ec26d34c1c614d95.jpg)

![1cc2f8b43438d312d98d4d8db9700329212ca51911f000ba5aa05306a701eff7.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/1cc2f8b43438d312d98d4d8db9700329212ca51911f000ba5aa05306a701eff7.jpg)

![27b4a5b3c5411cf961ff87342e74464ceb2ee42b8fa5f25da36cb94f4255b8f6.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/27b4a5b3c5411cf961ff87342e74464ceb2ee42b8fa5f25da36cb94f4255b8f6.jpg)

![4a67b22b87aec2a1c08873b4b7846c257e13fa9e531a25f233a0354531da9648.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/4a67b22b87aec2a1c08873b4b7846c257e13fa9e531a25f233a0354531da9648.jpg)

![59b4f20ce8b7c337b3bfc33dfddadc6aa3dc119ba2e77afea59bf8d921a6e430.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/59b4f20ce8b7c337b3bfc33dfddadc6aa3dc119ba2e77afea59bf8d921a6e430.jpg)

![63bab73432b98dfcfd4024b78ea2409e41a5d1011d262841ae7441d0f2316ee7.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/63bab73432b98dfcfd4024b78ea2409e41a5d1011d262841ae7441d0f2316ee7.jpg)

![640185c8ed3c704675b9b8c09d48dd1b804915771befa3aef0824383bb3221ed.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/640185c8ed3c704675b9b8c09d48dd1b804915771befa3aef0824383bb3221ed.jpg)

![664fbee75c615e0477ae5837654b247f9e5a5e94b3d36e3acdb15e436b93c108.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/664fbee75c615e0477ae5837654b247f9e5a5e94b3d36e3acdb15e436b93c108.jpg)

![6909c98ac3a02b56528f216ad4660f9731f66f6d8462abcde8f3dc596e6a26fe.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/6909c98ac3a02b56528f216ad4660f9731f66f6d8462abcde8f3dc596e6a26fe.jpg)

![73b0dcd9189680d6dc6eae6ad89dce18d07c266713200f426ef7e312a362a900.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/73b0dcd9189680d6dc6eae6ad89dce18d07c266713200f426ef7e312a362a900.jpg)

![74df60501f13668462eb0b020696dcb3e89296471a0e63e1a0742155ddf3cddd.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/74df60501f13668462eb0b020696dcb3e89296471a0e63e1a0742155ddf3cddd.jpg)

![8d5bcc44eeb8ab52e8ec57cb7a7a447aeb6a841a329599205b0a38684f65a7bb.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/8d5bcc44eeb8ab52e8ec57cb7a7a447aeb6a841a329599205b0a38684f65a7bb.jpg)

![9b3e8e76b0e6fad10569166a52dd49471844c2e2f4fd618c50069d202894a7c7.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/9b3e8e76b0e6fad10569166a52dd49471844c2e2f4fd618c50069d202894a7c7.jpg)

![a0557aa43057486fbfb5677f67c45fddd213d8c1dbce348d096548217c28a94c.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/a0557aa43057486fbfb5677f67c45fddd213d8c1dbce348d096548217c28a94c.jpg)

![a0886c486fd2109eb7eeb221ea2a1ef36af73520d8dea2c617e152b06384f47b.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/a0886c486fd2109eb7eeb221ea2a1ef36af73520d8dea2c617e152b06384f47b.jpg)

![a385cc2ad146b7a3bfb1edc118110c6142efb0b1c27a788f782d23dd53c540bc.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/a385cc2ad146b7a3bfb1edc118110c6142efb0b1c27a788f782d23dd53c540bc.jpg)

![a757596e4bf14bd8081b887ccf60d2116b7880201ef318ee43e98408d986b51e.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/a757596e4bf14bd8081b887ccf60d2116b7880201ef318ee43e98408d986b51e.jpg)

![ba0fed8a5f153b4ee49c6b117fc5ae92eb2222c553828d45af43026fef5b6495.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/ba0fed8a5f153b4ee49c6b117fc5ae92eb2222c553828d45af43026fef5b6495.jpg)

![cbeeabf246ad687f9670205cfefed77079805916d0d146244818f65fdf3652dd.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/cbeeabf246ad687f9670205cfefed77079805916d0d146244818f65fdf3652dd.jpg)

![e5464edc0bd301de51c4b64b608d793bd70a80046c2bcd7b220432dfffcf013f.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/e5464edc0bd301de51c4b64b608d793bd70a80046c2bcd7b220432dfffcf013f.jpg)

![eee8fd2388a3dca179018302c67be07809823e237858378207e7739daf94e9cf.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/eee8fd2388a3dca179018302c67be07809823e237858378207e7739daf94e9cf.jpg)

![f1001e30fef00476b1da40f751ba05a745a80a43fd0aa18331c85a3c9e430d33.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/f1001e30fef00476b1da40f751ba05a745a80a43fd0aa18331c85a3c9e430d33.jpg)

![fc4b3bbf7bb12686263e6470cec2b2e09aaef7fc484e49882ab0b0d10f48c6d8.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/images/fc4b3bbf7bb12686263e6470cec2b2e09aaef7fc484e49882ab0b0d10f48c6d8.jpg)

### Tables

![0dd0994684ad0168688f952512f9b59c63ec853368487f248a76f3b51a13704b.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/tables/0dd0994684ad0168688f952512f9b59c63ec853368487f248a76f3b51a13704b.jpg)

![1000a743483e3ddfad13a3c446f427445b6f341ba07864be386e3053b2314b21.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/tables/1000a743483e3ddfad13a3c446f427445b6f341ba07864be386e3053b2314b21.jpg)

![17714a906f5819cf5abe95dfdd1d7b2c2f8d90b27ccd5c77183144b70f12ec8f.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/tables/17714a906f5819cf5abe95dfdd1d7b2c2f8d90b27ccd5c77183144b70f12ec8f.jpg)

![1be87204787de7fd5905d65a09dc2020d6f6000f8bf2b1d29d6f8f8243dea1df.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/tables/1be87204787de7fd5905d65a09dc2020d6f6000f8bf2b1d29d6f8f8243dea1df.jpg)

![2ea14148197eeb6f4919aa737738af30c8aef602fac943fb941f8ba52cfce3fa.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/tables/2ea14148197eeb6f4919aa737738af30c8aef602fac943fb941f8ba52cfce3fa.jpg)

![4ac6b2763fa87fa899513f901762009cd6735e095a06a8e364666928822a8d14.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/tables/4ac6b2763fa87fa899513f901762009cd6735e095a06a8e364666928822a8d14.jpg)

![5070f6baa5f79af774f13548a32fb9e06e19e355442ca7dab3a45bae9b506ebc.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/tables/5070f6baa5f79af774f13548a32fb9e06e19e355442ca7dab3a45bae9b506ebc.jpg)

![53f633e25daea9dd787068934cbb4edf27efb5fe810b2a2dd3f0531f2eb237b8.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/tables/53f633e25daea9dd787068934cbb4edf27efb5fe810b2a2dd3f0531f2eb237b8.jpg)

![690b30bc483f72367ffae3c32bf11cb5419f4854900bdaa7a15523cc23a689b9.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/tables/690b30bc483f72367ffae3c32bf11cb5419f4854900bdaa7a15523cc23a689b9.jpg)

![6bf32b0564380e0483f0daf4bc415bb26cdbd742b502913485ad975a31e05d5e.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/tables/6bf32b0564380e0483f0daf4bc415bb26cdbd742b502913485ad975a31e05d5e.jpg)

![6e640e56df4f2b8bad96315cfb960d6f99198f0c0c67fbee4dcf6e6f31436aeb.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/tables/6e640e56df4f2b8bad96315cfb960d6f99198f0c0c67fbee4dcf6e6f31436aeb.jpg)

![7299378de5edb41095498a1803c7686bf9fa6b4818d9f4ebfaa1d521ae03d9cb.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/tables/7299378de5edb41095498a1803c7686bf9fa6b4818d9f4ebfaa1d521ae03d9cb.jpg)

![7dd5955e124992a821ec84526e3e4507f2f9603c70db0d25ae2d1ceba2276d7d.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/tables/7dd5955e124992a821ec84526e3e4507f2f9603c70db0d25ae2d1ceba2276d7d.jpg)

![a57c6ecd530cd12dca639494305f1898e5cf1451cf5b618b297849a4cff68528.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/tables/a57c6ecd530cd12dca639494305f1898e5cf1451cf5b618b297849a4cff68528.jpg)

![a8d4c3fcb82a7f2bd0ba3244056fea26d4d6589ff2cbe7c8779772c9f489a2f5.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/tables/a8d4c3fcb82a7f2bd0ba3244056fea26d4d6589ff2cbe7c8779772c9f489a2f5.jpg)

![b123c721020b71ac421be45bbc95dc3d760b5395ddab53244ac78b5a3f300680.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/tables/b123c721020b71ac421be45bbc95dc3d760b5395ddab53244ac78b5a3f300680.jpg)

![bea604da6c31fac761100b592baa281087609e077910c624e8a96791c9492800.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/tables/bea604da6c31fac761100b592baa281087609e077910c624e8a96791c9492800.jpg)

![d47ab3d1a9d6024c6e86ac9e29f1bd6aa9fef77c32c29b57639bc388f276f597.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/tables/d47ab3d1a9d6024c6e86ac9e29f1bd6aa9fef77c32c29b57639bc388f276f597.jpg)

![d9077c9dbcfc1e0fd488dce2fb2daa3c2725370d09286df82a5d0353bf05c7af.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/tables/d9077c9dbcfc1e0fd488dce2fb2daa3c2725370d09286df82a5d0353bf05c7af.jpg)

![e8dc4956a07e29816fff6ea2996cfd658a76f5344b3c09f5bf726aad84ac90ec.jpg](../iclr_results/103_Spider 2.0_ Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows/tables/e8dc4956a07e29816fff6ea2996cfd658a76f5344b3c09f5bf726aad84ac90ec.jpg)

## EmbodiedSAM: Online Segment Any 3D Thing in Real Time


### Images

![18cceb255d5fea0c5be38a8c33684864f270c06b129e4c341d5d56f07611d417.jpg](../iclr_results/104_EmbodiedSAM_ Online Segment Any 3D Thing in Real Time/images/18cceb255d5fea0c5be38a8c33684864f270c06b129e4c341d5d56f07611d417.jpg)

![25514905f4db6c72d10827103316321d0c3b488b5b28505842f9e6e595d6cd7b.jpg](../iclr_results/104_EmbodiedSAM_ Online Segment Any 3D Thing in Real Time/images/25514905f4db6c72d10827103316321d0c3b488b5b28505842f9e6e595d6cd7b.jpg)

![4ae8f3e4bed3ffad4a83467c97aa6f82e9785cada8b052a1f67f6a9c3df16082.jpg](../iclr_results/104_EmbodiedSAM_ Online Segment Any 3D Thing in Real Time/images/4ae8f3e4bed3ffad4a83467c97aa6f82e9785cada8b052a1f67f6a9c3df16082.jpg)

![5c4bce71517dc335203a56aa9624f0e0df033d8e228ea9aed541a9464b0b2e8b.jpg](../iclr_results/104_EmbodiedSAM_ Online Segment Any 3D Thing in Real Time/images/5c4bce71517dc335203a56aa9624f0e0df033d8e228ea9aed541a9464b0b2e8b.jpg)

![5e356b4768d01d984e0f90a194e3833debdd1efbbe452b05c13cb78f3b28b693.jpg](../iclr_results/104_EmbodiedSAM_ Online Segment Any 3D Thing in Real Time/images/5e356b4768d01d984e0f90a194e3833debdd1efbbe452b05c13cb78f3b28b693.jpg)

![88bad5f28a6e2fac0019564407bebf779c1e12861310bd7c70d55e9e79a36c46.jpg](../iclr_results/104_EmbodiedSAM_ Online Segment Any 3D Thing in Real Time/images/88bad5f28a6e2fac0019564407bebf779c1e12861310bd7c70d55e9e79a36c46.jpg)

![8a23f301c980b8bb152dbe28702dcb9ca4f534181836c46809f8404d915dd763.jpg](../iclr_results/104_EmbodiedSAM_ Online Segment Any 3D Thing in Real Time/images/8a23f301c980b8bb152dbe28702dcb9ca4f534181836c46809f8404d915dd763.jpg)

![fa3d3a32aec3dedf5d16880bdc85e89c2881a7b2b40b419e75a9e3972d10591e.jpg](../iclr_results/104_EmbodiedSAM_ Online Segment Any 3D Thing in Real Time/images/fa3d3a32aec3dedf5d16880bdc85e89c2881a7b2b40b419e75a9e3972d10591e.jpg)

### Tables

![20dbb9db9f1ea0d41c3a0fa6484e2dcf32a1370b48d7994d85e98e13705966c2.jpg](../iclr_results/104_EmbodiedSAM_ Online Segment Any 3D Thing in Real Time/tables/20dbb9db9f1ea0d41c3a0fa6484e2dcf32a1370b48d7994d85e98e13705966c2.jpg)

![2a0bb680e15b95d81fe5223cdf29e9170c592742d56b77901340b361ebc90a79.jpg](../iclr_results/104_EmbodiedSAM_ Online Segment Any 3D Thing in Real Time/tables/2a0bb680e15b95d81fe5223cdf29e9170c592742d56b77901340b361ebc90a79.jpg)

![4afd7a338ec4796a8be816dc0e5252150f733b360b19e3ed68424fd8a19b2091.jpg](../iclr_results/104_EmbodiedSAM_ Online Segment Any 3D Thing in Real Time/tables/4afd7a338ec4796a8be816dc0e5252150f733b360b19e3ed68424fd8a19b2091.jpg)

![65802e00e6ffce7bc9faff9879c26f13840493c168a2e178403475aa0667323e.jpg](../iclr_results/104_EmbodiedSAM_ Online Segment Any 3D Thing in Real Time/tables/65802e00e6ffce7bc9faff9879c26f13840493c168a2e178403475aa0667323e.jpg)

![716090caef0d1c1979a9466ae1e92bf266fb9eb1670c4814527caa63b32908a8.jpg](../iclr_results/104_EmbodiedSAM_ Online Segment Any 3D Thing in Real Time/tables/716090caef0d1c1979a9466ae1e92bf266fb9eb1670c4814527caa63b32908a8.jpg)

![77dcd7376696aaab8f9fab9f844d7a7e038ffe8312f5571c03209012b3ac277f.jpg](../iclr_results/104_EmbodiedSAM_ Online Segment Any 3D Thing in Real Time/tables/77dcd7376696aaab8f9fab9f844d7a7e038ffe8312f5571c03209012b3ac277f.jpg)

![cc5cb490871e583323c713218054031f7e8d1e8ebd584dbde0355990b11b0bc4.jpg](../iclr_results/104_EmbodiedSAM_ Online Segment Any 3D Thing in Real Time/tables/cc5cb490871e583323c713218054031f7e8d1e8ebd584dbde0355990b11b0bc4.jpg)

![db73aa39469eb4ce31fd8e274a615e8accb2bb22c15636fbbdef9f63fef93204.jpg](../iclr_results/104_EmbodiedSAM_ Online Segment Any 3D Thing in Real Time/tables/db73aa39469eb4ce31fd8e274a615e8accb2bb22c15636fbbdef9f63fef93204.jpg)

![e20451998ef6ed954d6be86188dc9ecaca61e70604a3a3f3b71eab74a6d3f433.jpg](../iclr_results/104_EmbodiedSAM_ Online Segment Any 3D Thing in Real Time/tables/e20451998ef6ed954d6be86188dc9ecaca61e70604a3a3f3b71eab74a6d3f433.jpg)

## Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping


### Images

![1e0cb395927d1424ed4d7acb7013ccc3148c9bbbcd8327483cf33841d9eb37e2.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/images/1e0cb395927d1424ed4d7acb7013ccc3148c9bbbcd8327483cf33841d9eb37e2.jpg)

![2c00a38434eee660a79549c9121a43dd732b563a05d30296172774367ac78c60.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/images/2c00a38434eee660a79549c9121a43dd732b563a05d30296172774367ac78c60.jpg)

![32293f002341d3ca1159ed19854e123d8ee125daa1447acf5c6fd2df19943fa2.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/images/32293f002341d3ca1159ed19854e123d8ee125daa1447acf5c6fd2df19943fa2.jpg)

![496f8dbc0ef3eaaa1b3832b0e484980eb8111a82fdc81b17f3dc65d6f39e7839.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/images/496f8dbc0ef3eaaa1b3832b0e484980eb8111a82fdc81b17f3dc65d6f39e7839.jpg)

![50d0ef4aefb212f1d13e922d29430945e2b38e77be3e67900a0ddc2cb889e7a6.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/images/50d0ef4aefb212f1d13e922d29430945e2b38e77be3e67900a0ddc2cb889e7a6.jpg)

![602add0e7be4a410052e0b2f08b3aecd10f0035163ac6ef3172fc8b3fc64bbf3.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/images/602add0e7be4a410052e0b2f08b3aecd10f0035163ac6ef3172fc8b3fc64bbf3.jpg)

![9980e826dfc6768fff2b5484868561f51d5c4ca3f62f343e507190b549fb7407.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/images/9980e826dfc6768fff2b5484868561f51d5c4ca3f62f343e507190b549fb7407.jpg)

![ab6e0d8a6e67b1abf9e88401f71f04caafb6f6660da4277502aa6b882fa336bf.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/images/ab6e0d8a6e67b1abf9e88401f71f04caafb6f6660da4277502aa6b882fa336bf.jpg)

![c8a8f93da1583f058496e8eb488b984f70142e88fb4ee07822c4715f4d7abad8.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/images/c8a8f93da1583f058496e8eb488b984f70142e88fb4ee07822c4715f4d7abad8.jpg)

![cb3f37108809f4031806a6df81424a9408eceee7d79bbd72927bb19acb2f6e3d.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/images/cb3f37108809f4031806a6df81424a9408eceee7d79bbd72927bb19acb2f6e3d.jpg)

![d4dbdbe9eb675aeafdc5ce7b1b36b300c4e3b4dcdfcf250a1a9784c69dd2fdf4.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/images/d4dbdbe9eb675aeafdc5ce7b1b36b300c4e3b4dcdfcf250a1a9784c69dd2fdf4.jpg)

![e4727778ba6e88c26a5917579f86c3592653201ed7b7f7311d827f01fc70c60a.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/images/e4727778ba6e88c26a5917579f86c3592653201ed7b7f7311d827f01fc70c60a.jpg)

![eb0b0e9dbcfa5edfb76aa202f26560f9aec1fa90cda1f23f0dae1eebf1e3e833.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/images/eb0b0e9dbcfa5edfb76aa202f26560f9aec1fa90cda1f23f0dae1eebf1e3e833.jpg)

![ef9771c9776ca985a0dded512112560d9c615bdae0e1fec4b3b3fcef4f128fa1.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/images/ef9771c9776ca985a0dded512112560d9c615bdae0e1fec4b3b3fcef4f128fa1.jpg)

![f7f42f716cf917d302b23cbf2382ee36314604397293cb450c8dc517f3b424f0.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/images/f7f42f716cf917d302b23cbf2382ee36314604397293cb450c8dc517f3b424f0.jpg)

### Tables

![19122ef7175aa7cb539a38b35d034d6f09ed67869ebc1aadd4a3e4ed0791a619.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/tables/19122ef7175aa7cb539a38b35d034d6f09ed67869ebc1aadd4a3e4ed0791a619.jpg)

![1bbace4cc14594766966febda52063606f7f5674c258fd5d402b97b7704e5060.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/tables/1bbace4cc14594766966febda52063606f7f5674c258fd5d402b97b7704e5060.jpg)

![32249ef32eabcc800902cd3620ccf7dd8c1d85386ebdad25eae0305021c860c6.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/tables/32249ef32eabcc800902cd3620ccf7dd8c1d85386ebdad25eae0305021c860c6.jpg)

![4901a4d1447fd4265426cac30b02f9603c6fdfed77584e4ecdc7d9e2555b9265.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/tables/4901a4d1447fd4265426cac30b02f9603c6fdfed77584e4ecdc7d9e2555b9265.jpg)

![59a3a8ffe0deed67f212b1ef36ed85ef4787569cff6a8ec5ca299bded3baa5fb.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/tables/59a3a8ffe0deed67f212b1ef36ed85ef4787569cff6a8ec5ca299bded3baa5fb.jpg)

![66bfdc0e96d3e44193e52540f994938adcd43e4dffdbf967a35726b37290cd6b.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/tables/66bfdc0e96d3e44193e52540f994938adcd43e4dffdbf967a35726b37290cd6b.jpg)

![93ca568701b61071f222981e114507179a75de06e4415c9ecd9037518b64aa62.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/tables/93ca568701b61071f222981e114507179a75de06e4415c9ecd9037518b64aa62.jpg)

![9e5ca0b5c038da03868c75b24834ad47400109b507b87d326a5a0f1ae427328d.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/tables/9e5ca0b5c038da03868c75b24834ad47400109b507b87d326a5a0f1ae427328d.jpg)

![b6ae2c115b4cecdd9f66dbe463b9b6649c004a59f183bbce814ca51a5876f22c.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/tables/b6ae2c115b4cecdd9f66dbe463b9b6649c004a59f183bbce814ca51a5876f22c.jpg)

![c296304c20e308bd4f748fd2666895a1c29d4249a89b30da25af27a7f703daf3.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/tables/c296304c20e308bd4f748fd2666895a1c29d4249a89b30da25af27a7f703daf3.jpg)

![ce64bd87a58da157b30a4b6adc4e521bf8ae495b6c724f370f9e4183e9a9cf51.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/tables/ce64bd87a58da157b30a4b6adc4e521bf8ae495b6c724f370f9e4183e9a9cf51.jpg)

![d368f01d43a2e89e99d62162b6f1d226fb43448b0d50f49affae149ca98dc860.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/tables/d368f01d43a2e89e99d62162b6f1d226fb43448b0d50f49affae149ca98dc860.jpg)

![dafb20b6c24150290f21422b7881f5a9bfddbb425146253379ac6760af0a26f0.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/tables/dafb20b6c24150290f21422b7881f5a9bfddbb425146253379ac6760af0a26f0.jpg)

![e224f835c2ab111a8207a9959dc400ab46d38f391b9b9cbd7167e804f0dd9a57.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/tables/e224f835c2ab111a8207a9959dc400ab46d38f391b9b9cbd7167e804f0dd9a57.jpg)

![faeda35782e99c8d13d89769a4b482087a8af286827612d13c33b34f230886f3.jpg](../iclr_results/105_Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping/tables/faeda35782e99c8d13d89769a4b482087a8af286827612d13c33b34f230886f3.jpg)

## LARP: Tokenizing Videos with a Learned Autoregressive Generative Prior


### Images

![00bf0f290b2ed9dea612da4955aae86bd9943b9a93576d34e26107935ee53d20.jpg](../iclr_results/106_LARP_ Tokenizing Videos with a Learned Autoregressive Generative Prior/images/00bf0f290b2ed9dea612da4955aae86bd9943b9a93576d34e26107935ee53d20.jpg)

![18a5fbe50a0f488ab4bd9e6b46000cc9baee027154ce7d8907e05fbf9e07eb80.jpg](../iclr_results/106_LARP_ Tokenizing Videos with a Learned Autoregressive Generative Prior/images/18a5fbe50a0f488ab4bd9e6b46000cc9baee027154ce7d8907e05fbf9e07eb80.jpg)

![285ad148803b92e5ca4a7256ff503a71f769cc86e783cab76138029c3c97b51c.jpg](../iclr_results/106_LARP_ Tokenizing Videos with a Learned Autoregressive Generative Prior/images/285ad148803b92e5ca4a7256ff503a71f769cc86e783cab76138029c3c97b51c.jpg)

![53fa1fac97c2d216bdc4006c6ebdeadae7f0a0375d4eb7eac244e450598b4f47.jpg](../iclr_results/106_LARP_ Tokenizing Videos with a Learned Autoregressive Generative Prior/images/53fa1fac97c2d216bdc4006c6ebdeadae7f0a0375d4eb7eac244e450598b4f47.jpg)

![5bc0a0f6b1ce41eb1bacbd6b19237c377c55689feae09fa73c7e56994a6a4475.jpg](../iclr_results/106_LARP_ Tokenizing Videos with a Learned Autoregressive Generative Prior/images/5bc0a0f6b1ce41eb1bacbd6b19237c377c55689feae09fa73c7e56994a6a4475.jpg)

![6a285e8aa5a3e6cc9d351dbf364b209fe6dd4004e35e7857e6376b52584540e6.jpg](../iclr_results/106_LARP_ Tokenizing Videos with a Learned Autoregressive Generative Prior/images/6a285e8aa5a3e6cc9d351dbf364b209fe6dd4004e35e7857e6376b52584540e6.jpg)

![6e079da0bf75f68939171c768c704ebafdb137a3a51dec64e44c0c9c5e6d2b01.jpg](../iclr_results/106_LARP_ Tokenizing Videos with a Learned Autoregressive Generative Prior/images/6e079da0bf75f68939171c768c704ebafdb137a3a51dec64e44c0c9c5e6d2b01.jpg)

![7643b5f29197044022a6dd51e82cdaf59a70d051a7143b8c89f809b550439c2a.jpg](../iclr_results/106_LARP_ Tokenizing Videos with a Learned Autoregressive Generative Prior/images/7643b5f29197044022a6dd51e82cdaf59a70d051a7143b8c89f809b550439c2a.jpg)

![7dc1f1cb2255be73e4af973a4d53d72940306dc6395dd28fea987284473aff42.jpg](../iclr_results/106_LARP_ Tokenizing Videos with a Learned Autoregressive Generative Prior/images/7dc1f1cb2255be73e4af973a4d53d72940306dc6395dd28fea987284473aff42.jpg)

![8afb11b47103dd957228f1178dd7896fc84d7f55cd0632dd6650d8b550120e31.jpg](../iclr_results/106_LARP_ Tokenizing Videos with a Learned Autoregressive Generative Prior/images/8afb11b47103dd957228f1178dd7896fc84d7f55cd0632dd6650d8b550120e31.jpg)

![9dded8e1b99db170ec0a955fc880bbe11f40961a44201f09e4533f66b17417e0.jpg](../iclr_results/106_LARP_ Tokenizing Videos with a Learned Autoregressive Generative Prior/images/9dded8e1b99db170ec0a955fc880bbe11f40961a44201f09e4533f66b17417e0.jpg)

![d5b8de1b8c59abf031031ea2ad59637954b0f38966efbb9683fde70f12b11887.jpg](../iclr_results/106_LARP_ Tokenizing Videos with a Learned Autoregressive Generative Prior/images/d5b8de1b8c59abf031031ea2ad59637954b0f38966efbb9683fde70f12b11887.jpg)

![e27275c678a25a70821fc8c76aea759d3cfdca02c15460f489fd5afbe01d2d6d.jpg](../iclr_results/106_LARP_ Tokenizing Videos with a Learned Autoregressive Generative Prior/images/e27275c678a25a70821fc8c76aea759d3cfdca02c15460f489fd5afbe01d2d6d.jpg)

![e5fd921149d71da94f9b40e29dd156753f301ca56245533ab33c90eb83ffb8c0.jpg](../iclr_results/106_LARP_ Tokenizing Videos with a Learned Autoregressive Generative Prior/images/e5fd921149d71da94f9b40e29dd156753f301ca56245533ab33c90eb83ffb8c0.jpg)

![efdd122612d5151cd77ba057a94e359241232c8dc073246fb4bc0edb2c78db5f.jpg](../iclr_results/106_LARP_ Tokenizing Videos with a Learned Autoregressive Generative Prior/images/efdd122612d5151cd77ba057a94e359241232c8dc073246fb4bc0edb2c78db5f.jpg)

![f61b9f3e5307b4626f253f9b43309f998aae67431bcdb08379d78a06e805d928.jpg](../iclr_results/106_LARP_ Tokenizing Videos with a Learned Autoregressive Generative Prior/images/f61b9f3e5307b4626f253f9b43309f998aae67431bcdb08379d78a06e805d928.jpg)

![f74d95c9ff56479fc36a98bde413b138aacbad31c08a2b200cdae85114005f5a.jpg](../iclr_results/106_LARP_ Tokenizing Videos with a Learned Autoregressive Generative Prior/images/f74d95c9ff56479fc36a98bde413b138aacbad31c08a2b200cdae85114005f5a.jpg)

![fe1440aa913c1f06ff85b2ec1b0c2df34f8126d31d00b4211b3d76532cdd2ae7.jpg](../iclr_results/106_LARP_ Tokenizing Videos with a Learned Autoregressive Generative Prior/images/fe1440aa913c1f06ff85b2ec1b0c2df34f8126d31d00b4211b3d76532cdd2ae7.jpg)

### Tables

![1c2a0707a81c47d9dd63d8b9d9b850ff7dea23866652aa14b0e9e844ec15b28a.jpg](../iclr_results/106_LARP_ Tokenizing Videos with a Learned Autoregressive Generative Prior/tables/1c2a0707a81c47d9dd63d8b9d9b850ff7dea23866652aa14b0e9e844ec15b28a.jpg)

![283e1641b06633c7c24b34d6b0b68456d8d41a266c70283bb375347c7e3e9f1f.jpg](../iclr_results/106_LARP_ Tokenizing Videos with a Learned Autoregressive Generative Prior/tables/283e1641b06633c7c24b34d6b0b68456d8d41a266c70283bb375347c7e3e9f1f.jpg)

## Knowing Your Target: Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding


### Images

![06bd34fd5ec722841da7a3983ec090de20bc5b35f18667b11864f074e6e04611.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/images/06bd34fd5ec722841da7a3983ec090de20bc5b35f18667b11864f074e6e04611.jpg)

![203a9b5f750ba705d714f4ac0c2a43dfab60b28e29636c0a179301ec2ffeb9ce.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/images/203a9b5f750ba705d714f4ac0c2a43dfab60b28e29636c0a179301ec2ffeb9ce.jpg)

![2a9e296fc0efc9e48dfa6d518011517080495053ff98b2d87c42b40092089e52.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/images/2a9e296fc0efc9e48dfa6d518011517080495053ff98b2d87c42b40092089e52.jpg)

![34177ff1b654484924d24c31cabc77459638e6440d88e8ca1fe36f182b001284.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/images/34177ff1b654484924d24c31cabc77459638e6440d88e8ca1fe36f182b001284.jpg)

![3630f908fd391454ec9580106256cfdf8c55f1498edeb423b18035038312ae81.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/images/3630f908fd391454ec9580106256cfdf8c55f1498edeb423b18035038312ae81.jpg)

![4e5367c43f40561931cdbe3e0b17338047b97a15e7d85eeabcc12d2ff73797ce.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/images/4e5367c43f40561931cdbe3e0b17338047b97a15e7d85eeabcc12d2ff73797ce.jpg)

![4eec1b8347715405f1f4e1a5d575babeaa4ad7c35165af1cb60569a6f0e9b3d4.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/images/4eec1b8347715405f1f4e1a5d575babeaa4ad7c35165af1cb60569a6f0e9b3d4.jpg)

![5cbbce0a178e164ad615681da3aac23c28ac478fec10159d758fae9ceeec0da6.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/images/5cbbce0a178e164ad615681da3aac23c28ac478fec10159d758fae9ceeec0da6.jpg)

![750e31541e8926e0ca5b793ee4c5e5c9f9fb8b1823b98d150d3555b92ad5925a.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/images/750e31541e8926e0ca5b793ee4c5e5c9f9fb8b1823b98d150d3555b92ad5925a.jpg)

![94f091754252a6f54f56869c0f974006ee0ecf6d30ce947fc4ecaae9808a922c.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/images/94f091754252a6f54f56869c0f974006ee0ecf6d30ce947fc4ecaae9808a922c.jpg)

![b38cccb9a7868abcff27aa3d5be584fe652d83b0a87b8f5aea94cba46090e92c.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/images/b38cccb9a7868abcff27aa3d5be584fe652d83b0a87b8f5aea94cba46090e92c.jpg)

![d69089f398699939ecca79dd72687d1e5aea177d84000e2e947a52f079d5a590.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/images/d69089f398699939ecca79dd72687d1e5aea177d84000e2e947a52f079d5a590.jpg)

![e07bd5af04d5e5397cf1625c648e15c3b776e0267428b3867770d9af4fd6b4cd.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/images/e07bd5af04d5e5397cf1625c648e15c3b776e0267428b3867770d9af4fd6b4cd.jpg)

![e3ef00ed674b0723562a19cf8e15274b312bd5341ccb9c0e9d1cddab1fd37728.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/images/e3ef00ed674b0723562a19cf8e15274b312bd5341ccb9c0e9d1cddab1fd37728.jpg)

![f7d4c514e16ecac0a3c903793ca92eaa57a47684a7f26f3043c9a4bd9a6c98e7.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/images/f7d4c514e16ecac0a3c903793ca92eaa57a47684a7f26f3043c9a4bd9a6c98e7.jpg)

![ff915b1be5a0f67d770132ec3b4c2c882c0584bb943adb4bce50ee62032a788f.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/images/ff915b1be5a0f67d770132ec3b4c2c882c0584bb943adb4bce50ee62032a788f.jpg)

### Tables

![0889e8e55e0255d37566ee75a3d41db0cce678f0cd418e42fa3e8af6f5df4b49.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/tables/0889e8e55e0255d37566ee75a3d41db0cce678f0cd418e42fa3e8af6f5df4b49.jpg)

![15fd2a92c5b4f5ac5307c43bff06a8778b4e3a5be6023db46c83244bc48ee33a.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/tables/15fd2a92c5b4f5ac5307c43bff06a8778b4e3a5be6023db46c83244bc48ee33a.jpg)

![387813a20936219993b1f8a26f279b03c622e47eba4d971ecc3948a79efbeae6.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/tables/387813a20936219993b1f8a26f279b03c622e47eba4d971ecc3948a79efbeae6.jpg)

![75a49d90017da46b980788cd2668f839dbb6fbc4b679a9ecb3b8f68f52f4409b.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/tables/75a49d90017da46b980788cd2668f839dbb6fbc4b679a9ecb3b8f68f52f4409b.jpg)

![7fddc45804a565783bfbd7915f0e5fe15960e94293e41eabdb321e92b8cd4a8f.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/tables/7fddc45804a565783bfbd7915f0e5fe15960e94293e41eabdb321e92b8cd4a8f.jpg)

![9de336e59e97e47217cec5625735ef571fdd5f7451c00f4d9d54dbe1459bceef.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/tables/9de336e59e97e47217cec5625735ef571fdd5f7451c00f4d9d54dbe1459bceef.jpg)

![a5fc8cefc907b98e6a0d93c9c2f939d9d8b92e1bd5c0b06b6d5b20b9c0a140e5.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/tables/a5fc8cefc907b98e6a0d93c9c2f939d9d8b92e1bd5c0b06b6d5b20b9c0a140e5.jpg)

![c8ef632b4c64cb6f2eecd8e55c207b6cd1ea6e99fcd0f9fdb897404d4e46e67d.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/tables/c8ef632b4c64cb6f2eecd8e55c207b6cd1ea6e99fcd0f9fdb897404d4e46e67d.jpg)

![d0334a30ff894c8444075d76e4568449c418873deb863ec9ec146967a0f8f4d3.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/tables/d0334a30ff894c8444075d76e4568449c418873deb863ec9ec146967a0f8f4d3.jpg)

![e7aa30459178dab0d33ea8eeb9c91ab44e30f9325a60fe7697496bca3e9d8634.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/tables/e7aa30459178dab0d33ea8eeb9c91ab44e30f9325a60fe7697496bca3e9d8634.jpg)

![e8b33de14866d2d83ff64d17e49d31d69133c9635632b78d2c1c3aed137efbfa.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/tables/e8b33de14866d2d83ff64d17e49d31d69133c9635632b78d2c1c3aed137efbfa.jpg)

![ef59c65ea525f6e026384de850453ae168f605ad5fcee44bf1be7bbf385d37cd.jpg](../iclr_results/107_Knowing Your Target_ Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding/tables/ef59c65ea525f6e026384de850453ae168f605ad5fcee44bf1be7bbf385d37cd.jpg)

## Self-Improvement in Language Models: The Sharpening Mechanism


### Images

![18e5275141b3a9987c588d44eec073e445462b49af8e9a9452c3235cc95a053d.jpg](../iclr_results/108_Self-Improvement in Language Models_ The Sharpening Mechanism/images/18e5275141b3a9987c588d44eec073e445462b49af8e9a9452c3235cc95a053d.jpg)

![1e782a3941613684040252887e23c2c0bd1c92907ef0b246a7381438f93162a2.jpg](../iclr_results/108_Self-Improvement in Language Models_ The Sharpening Mechanism/images/1e782a3941613684040252887e23c2c0bd1c92907ef0b246a7381438f93162a2.jpg)

![48042c408d0975fca47b6a3d4afb458472408d5009038532a947691e695e9a0d.jpg](../iclr_results/108_Self-Improvement in Language Models_ The Sharpening Mechanism/images/48042c408d0975fca47b6a3d4afb458472408d5009038532a947691e695e9a0d.jpg)

![81767971a8881301e4392036a4636b5450b9059f19db9f75b2601142d2ac5b99.jpg](../iclr_results/108_Self-Improvement in Language Models_ The Sharpening Mechanism/images/81767971a8881301e4392036a4636b5450b9059f19db9f75b2601142d2ac5b99.jpg)

![9b8c0fcf1b422a3f92f46f21d25501fd5e2150b727ccc4b001e3b0c0cd376cc0.jpg](../iclr_results/108_Self-Improvement in Language Models_ The Sharpening Mechanism/images/9b8c0fcf1b422a3f92f46f21d25501fd5e2150b727ccc4b001e3b0c0cd376cc0.jpg)

![bb43dafafd507f6d9bb9bc4ffd695584ca6304f6680d81d2b9b7d0148a9d5cb7.jpg](../iclr_results/108_Self-Improvement in Language Models_ The Sharpening Mechanism/images/bb43dafafd507f6d9bb9bc4ffd695584ca6304f6680d81d2b9b7d0148a9d5cb7.jpg)

![c68a7a098255db705aca784b3a700c887c048a9adb18ffbf1034846451258a26.jpg](../iclr_results/108_Self-Improvement in Language Models_ The Sharpening Mechanism/images/c68a7a098255db705aca784b3a700c887c048a9adb18ffbf1034846451258a26.jpg)

![cb746cceba9c3318b422e6d1da775cdf3e75c0bf96fd5944811da47180f03bc0.jpg](../iclr_results/108_Self-Improvement in Language Models_ The Sharpening Mechanism/images/cb746cceba9c3318b422e6d1da775cdf3e75c0bf96fd5944811da47180f03bc0.jpg)

![e621dbf140e5c4bc62e6b8939c6d4e53ce2db08df491f77a9f6ecfc8ce26b35b.jpg](../iclr_results/108_Self-Improvement in Language Models_ The Sharpening Mechanism/images/e621dbf140e5c4bc62e6b8939c6d4e53ce2db08df491f77a9f6ecfc8ce26b35b.jpg)

### Tables

![83bb04729799bc2bd3e4297898e8aeee92b27bf71e9a3950605460b4ebe08f6c.jpg](../iclr_results/108_Self-Improvement in Language Models_ The Sharpening Mechanism/tables/83bb04729799bc2bd3e4297898e8aeee92b27bf71e9a3950605460b4ebe08f6c.jpg)

![9aee0a5ed51645615b4a1a8c80016029292b5722c85e3858bcafaf5d99e0eb26.jpg](../iclr_results/108_Self-Improvement in Language Models_ The Sharpening Mechanism/tables/9aee0a5ed51645615b4a1a8c80016029292b5722c85e3858bcafaf5d99e0eb26.jpg)

## Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models


### Images

![08ba150e4bcf8b8d0bfab02176c80eb54fe9e731c6ec8742956cfa168efaa276.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/08ba150e4bcf8b8d0bfab02176c80eb54fe9e731c6ec8742956cfa168efaa276.jpg)

![1b3004afc89a90e23145f04b52dfb918890259e3bab76e588c8699f70fa03c27.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/1b3004afc89a90e23145f04b52dfb918890259e3bab76e588c8699f70fa03c27.jpg)

![2624c7ffd69149bec0622eb9f569861a0496520a48cdd99a0c06040f11ab4f33.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/2624c7ffd69149bec0622eb9f569861a0496520a48cdd99a0c06040f11ab4f33.jpg)

![2d9be8ca83e45fe5a861b89299f48237a3e90f18c90ff2bf0bc015e0659922fa.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/2d9be8ca83e45fe5a861b89299f48237a3e90f18c90ff2bf0bc015e0659922fa.jpg)

![2e967aed8952d0a75c830bced5f6412effba0eb7c8c3b997138726f925c5c332.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/2e967aed8952d0a75c830bced5f6412effba0eb7c8c3b997138726f925c5c332.jpg)

![323fafe1429b9674c90e919991e87103b3f49012c0382fadcc8a851d7bebcba2.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/323fafe1429b9674c90e919991e87103b3f49012c0382fadcc8a851d7bebcba2.jpg)

![3e2e0e0dfccc80429e22ad5ffc9503498fd5243af29a3222563ac28a3bffefb0.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/3e2e0e0dfccc80429e22ad5ffc9503498fd5243af29a3222563ac28a3bffefb0.jpg)

![5153109b2253c1ff891b70798ffe576b8fc40de4f57f72ec66a27aaf9c7a0796.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/5153109b2253c1ff891b70798ffe576b8fc40de4f57f72ec66a27aaf9c7a0796.jpg)

![5930d6a9895579041edf08f70d727d68f746f5d02cc216ca633697edebbf567e.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/5930d6a9895579041edf08f70d727d68f746f5d02cc216ca633697edebbf567e.jpg)

![5aa6cfe976ec0c6281dc835b159f7630c78a266c29fbeda9a9e0e5a7fcf6a5b1.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/5aa6cfe976ec0c6281dc835b159f7630c78a266c29fbeda9a9e0e5a7fcf6a5b1.jpg)

![65d25a948df3e6dda49a595e64c60b46e1473b05e2f79c9c7de80e21f8984543.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/65d25a948df3e6dda49a595e64c60b46e1473b05e2f79c9c7de80e21f8984543.jpg)

![77adf8ab90522c3b47f80936b0fe828731e6e1bcbb5e7a585090e5ee68932162.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/77adf8ab90522c3b47f80936b0fe828731e6e1bcbb5e7a585090e5ee68932162.jpg)

![7ad444bf7a0eecba9068f9b607a4fbab3d22260ef199e1e3e293a954f27fb7f7.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/7ad444bf7a0eecba9068f9b607a4fbab3d22260ef199e1e3e293a954f27fb7f7.jpg)

![8375908bdcacec5a55a074d6cd8892289e41d6eee1ddd26457016f8c7b13d0f5.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/8375908bdcacec5a55a074d6cd8892289e41d6eee1ddd26457016f8c7b13d0f5.jpg)

![851ec2972017eaa758ec49e81a92125fef3dc1a71478e96631164976d9d59731.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/851ec2972017eaa758ec49e81a92125fef3dc1a71478e96631164976d9d59731.jpg)

![88b9047ce415f0f04a22f5a4c9e0acd67536ec2a6244391a4abbdcd12b6c4dac.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/88b9047ce415f0f04a22f5a4c9e0acd67536ec2a6244391a4abbdcd12b6c4dac.jpg)

![949e310b8472f94b32478a6c4c67d7446b6d2e6a7454cc635d9e88d3fbcab01a.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/949e310b8472f94b32478a6c4c67d7446b6d2e6a7454cc635d9e88d3fbcab01a.jpg)

![955ac6536628388355c082a09f9bb50a223103736c2261d1e259e15a630c7433.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/955ac6536628388355c082a09f9bb50a223103736c2261d1e259e15a630c7433.jpg)

![95ef2fdb891789bddd64cbb04a5dca7e04eaa81949099d14f7ded1eb18d85233.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/95ef2fdb891789bddd64cbb04a5dca7e04eaa81949099d14f7ded1eb18d85233.jpg)

![97f053b172003585e85ee6359c63074219a57032ca8773173939ccd03b9ee2c8.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/97f053b172003585e85ee6359c63074219a57032ca8773173939ccd03b9ee2c8.jpg)

![a99b1d4bdbe1c1bb957915243ffd88d545d6a864a255ea38430a8823d1991f7a.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/a99b1d4bdbe1c1bb957915243ffd88d545d6a864a255ea38430a8823d1991f7a.jpg)

![ab5452b00c0d4c313ddf3b333507020a51b593f7a28182429447bf78b74473e2.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/ab5452b00c0d4c313ddf3b333507020a51b593f7a28182429447bf78b74473e2.jpg)

![accc65fb0033a610f3015fc954b89eeddfd75b4330c972f54bb00be7f1e66b65.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/accc65fb0033a610f3015fc954b89eeddfd75b4330c972f54bb00be7f1e66b65.jpg)

![b40ec9ab04931f32a0f97603e5573b926d367ccf81ebbe36af48923590715537.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/b40ec9ab04931f32a0f97603e5573b926d367ccf81ebbe36af48923590715537.jpg)

![b4310a449bede353d5f9acd9cf33d7aebcbba717e4ffbd3c80369b8e9b458379.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/b4310a449bede353d5f9acd9cf33d7aebcbba717e4ffbd3c80369b8e9b458379.jpg)

![b59761365ff0e3eb7679184c462da6178d02bcef486550e94c8ae1cfe3235ad6.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/b59761365ff0e3eb7679184c462da6178d02bcef486550e94c8ae1cfe3235ad6.jpg)

![bae7aac41d9364b21686e1d79464126b8225c248d60a53c8b3840c55eab523ff.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/bae7aac41d9364b21686e1d79464126b8225c248d60a53c8b3840c55eab523ff.jpg)

![bb3de4143ca07468a1ec2f502177d3b2d32327f94f6b927c606033feb04fd5ae.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/bb3de4143ca07468a1ec2f502177d3b2d32327f94f6b927c606033feb04fd5ae.jpg)

![c020fbd73ea3d7557d282f670b9d75201d2157f372dbaa26d3d45b0a76995c5f.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/c020fbd73ea3d7557d282f670b9d75201d2157f372dbaa26d3d45b0a76995c5f.jpg)

![c705315c1203850ada67ad30c753d34046ce956f0b15182a388fd7cc92cd0cac.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/c705315c1203850ada67ad30c753d34046ce956f0b15182a388fd7cc92cd0cac.jpg)

![cbb8b40a34db747aedafb78c00e81d58e76e5310c99962de6beb5e932fe19503.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/cbb8b40a34db747aedafb78c00e81d58e76e5310c99962de6beb5e932fe19503.jpg)

![cdc13dd23762c8d0623af5b5467a534992b1a32759a6b27cbf8addc36c00e7e5.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/cdc13dd23762c8d0623af5b5467a534992b1a32759a6b27cbf8addc36c00e7e5.jpg)

![d269a863dfb1f1c63c7bdcce2f8dd5760a3a73a7337428e9108fc0efd4048ac5.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/d269a863dfb1f1c63c7bdcce2f8dd5760a3a73a7337428e9108fc0efd4048ac5.jpg)

![f330c9f0ccb6abf449fd0b36d4edbb4953809ab3c9ae0b698b7a9d111c608c92.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/f330c9f0ccb6abf449fd0b36d4edbb4953809ab3c9ae0b698b7a9d111c608c92.jpg)

![f8d53a800806af2545ea100c5689bf7ae86b70da3e4c35296c7a6a79f8645998.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/images/f8d53a800806af2545ea100c5689bf7ae86b70da3e4c35296c7a6a79f8645998.jpg)

### Tables

![6e51d656d475d66ba88cdd83d355f1329ed9ec9e38e55de8e1f987b3820d7070.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/tables/6e51d656d475d66ba88cdd83d355f1329ed9ec9e38e55de8e1f987b3820d7070.jpg)

![780b428615a8d3ae80a4186a53961c779802464c2c50886ddd82c7042e9f62a9.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/tables/780b428615a8d3ae80a4186a53961c779802464c2c50886ddd82c7042e9f62a9.jpg)

![83e8a8d2036fa5e48914337af0e1201823696fbc5365e73de915a398f8323b01.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/tables/83e8a8d2036fa5e48914337af0e1201823696fbc5365e73de915a398f8323b01.jpg)

![9367b6ae1547fa27b2b756a420973e85706076c27a1560008e1b9ffbf47e54f5.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/tables/9367b6ae1547fa27b2b756a420973e85706076c27a1560008e1b9ffbf47e54f5.jpg)

![9e405ee814211dbd676e40297fa966c577fe95726acc59f8be4d1a49e7accdae.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/tables/9e405ee814211dbd676e40297fa966c577fe95726acc59f8be4d1a49e7accdae.jpg)

![bcc51eb9742de44fab312189286b086f86866dfd2878675f462f33636d4c96c2.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/tables/bcc51eb9742de44fab312189286b086f86866dfd2878675f462f33636d4c96c2.jpg)

![c710a5c2b0047db813ff4596867d425be89b43330a690f9be1d5185a69861156.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/tables/c710a5c2b0047db813ff4596867d425be89b43330a690f9be1d5185a69861156.jpg)

![d08422f0359445acc138c6a384c85c37bcc64d32bb443fe5c73dee366bf1398f.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/tables/d08422f0359445acc138c6a384c85c37bcc64d32bb443fe5c73dee366bf1398f.jpg)

![f1103d5ede6b038c46e34715c0013daa1935cabb444c476a8f9ca708cce17b9d.jpg](../iclr_results/109_Do I Know This Entity_ Knowledge Awareness and Hallucinations in Language Models/tables/f1103d5ede6b038c46e34715c0013daa1935cabb444c476a8f9ca708cce17b9d.jpg)

## LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization


### Images

![2541e95401589ac4a1e6bb548cd2cb5cab269c582b03fe54b8b09256472b29b8.jpg](../iclr_results/110_LoRA Done RITE_ Robust Invariant Transformation Equilibration for LoRA Optimization/images/2541e95401589ac4a1e6bb548cd2cb5cab269c582b03fe54b8b09256472b29b8.jpg)

![47ac106272a87309ab28f6e7aef8c04855991faf11df20c03611242b2146c13f.jpg](../iclr_results/110_LoRA Done RITE_ Robust Invariant Transformation Equilibration for LoRA Optimization/images/47ac106272a87309ab28f6e7aef8c04855991faf11df20c03611242b2146c13f.jpg)

![4d5e8b7d82331ad7085c0c3c61eb5078f735bf4cbc44c8b8f777de41519e8f0c.jpg](../iclr_results/110_LoRA Done RITE_ Robust Invariant Transformation Equilibration for LoRA Optimization/images/4d5e8b7d82331ad7085c0c3c61eb5078f735bf4cbc44c8b8f777de41519e8f0c.jpg)

![5cb217a9dd951e2ec31a042e81295c0f109e04f711d24e5d19ee9099d1ced297.jpg](../iclr_results/110_LoRA Done RITE_ Robust Invariant Transformation Equilibration for LoRA Optimization/images/5cb217a9dd951e2ec31a042e81295c0f109e04f711d24e5d19ee9099d1ced297.jpg)

### Tables

![0b9af55f208f1fd8f3125c3e65e178eed0ece1591252e13456c927fb831ad2f7.jpg](../iclr_results/110_LoRA Done RITE_ Robust Invariant Transformation Equilibration for LoRA Optimization/tables/0b9af55f208f1fd8f3125c3e65e178eed0ece1591252e13456c927fb831ad2f7.jpg)

![3636b60d74b3143945d635cc19b0cce334b4eada557d38b75265096f8b44ebdc.jpg](../iclr_results/110_LoRA Done RITE_ Robust Invariant Transformation Equilibration for LoRA Optimization/tables/3636b60d74b3143945d635cc19b0cce334b4eada557d38b75265096f8b44ebdc.jpg)

![455e57c6980736fd6495b98c66b0672df00dce8e38cf8280ead88ed80cd66f03.jpg](../iclr_results/110_LoRA Done RITE_ Robust Invariant Transformation Equilibration for LoRA Optimization/tables/455e57c6980736fd6495b98c66b0672df00dce8e38cf8280ead88ed80cd66f03.jpg)

![56b7832ae3520292238ac789b3aca19bd8d89476d60301c9766d3777481572d8.jpg](../iclr_results/110_LoRA Done RITE_ Robust Invariant Transformation Equilibration for LoRA Optimization/tables/56b7832ae3520292238ac789b3aca19bd8d89476d60301c9766d3777481572d8.jpg)

![5d253ad4887f5a751ccbb55dfd9d4a6ce6d95a539498423b3dc694fcbfc4db84.jpg](../iclr_results/110_LoRA Done RITE_ Robust Invariant Transformation Equilibration for LoRA Optimization/tables/5d253ad4887f5a751ccbb55dfd9d4a6ce6d95a539498423b3dc694fcbfc4db84.jpg)

![c39ea2905472b51d5428d488c1e44e5efffeaf8e5e5a28e411c640b93ac7818b.jpg](../iclr_results/110_LoRA Done RITE_ Robust Invariant Transformation Equilibration for LoRA Optimization/tables/c39ea2905472b51d5428d488c1e44e5efffeaf8e5e5a28e411c640b93ac7818b.jpg)

![c98b5e9e057ccc1e6165fb5e4af7a46ce8893006dfd1137e47586f9e93b0ec70.jpg](../iclr_results/110_LoRA Done RITE_ Robust Invariant Transformation Equilibration for LoRA Optimization/tables/c98b5e9e057ccc1e6165fb5e4af7a46ce8893006dfd1137e47586f9e93b0ec70.jpg)

![dc8908f7d7e9ab0605175053ec4eebbafb7fd1dccf6fb5c4e3b1c5b4f9755a53.jpg](../iclr_results/110_LoRA Done RITE_ Robust Invariant Transformation Equilibration for LoRA Optimization/tables/dc8908f7d7e9ab0605175053ec4eebbafb7fd1dccf6fb5c4e3b1c5b4f9755a53.jpg)

![ff45f170ba834e8dd1b92a37ec6f6284aab4f7b791211208876a12b98314fd3c.jpg](../iclr_results/110_LoRA Done RITE_ Robust Invariant Transformation Equilibration for LoRA Optimization/tables/ff45f170ba834e8dd1b92a37ec6f6284aab4f7b791211208876a12b98314fd3c.jpg)

## Reasoning Elicitation in Language Models via Counterfactual Feedback


### Images

![0df4969376316cbd68e751552166e0b72d45d318a1e2cd239f2203b358e9650e.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/images/0df4969376316cbd68e751552166e0b72d45d318a1e2cd239f2203b358e9650e.jpg)

![13bdceee6554b96b5a8a760727172915633dcb2b9e1f8c5464d039bed9dd4930.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/images/13bdceee6554b96b5a8a760727172915633dcb2b9e1f8c5464d039bed9dd4930.jpg)

![15a75e9b8f2eaab2d68bf02f00d2845e320468a4a7018b5eb3a130ab4e778a54.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/images/15a75e9b8f2eaab2d68bf02f00d2845e320468a4a7018b5eb3a130ab4e778a54.jpg)

![2db37dfab76f8872c7791f8e3a9d6cd37824293ba17cdc8292d5e4208fe6a7d8.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/images/2db37dfab76f8872c7791f8e3a9d6cd37824293ba17cdc8292d5e4208fe6a7d8.jpg)

![389fbcc18d2615550c391103e3149b97209fb41df9c801dd79b9b32d46585e03.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/images/389fbcc18d2615550c391103e3149b97209fb41df9c801dd79b9b32d46585e03.jpg)

![38dc5ccf16e96f50b4dbb43c4102c7ef28ed514347e4c1729a5fb2a9b4099e88.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/images/38dc5ccf16e96f50b4dbb43c4102c7ef28ed514347e4c1729a5fb2a9b4099e88.jpg)

![3e0fb63e02abd63d64555ecb86955451623e7ce71d43c68501ff9a5dbf8467f3.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/images/3e0fb63e02abd63d64555ecb86955451623e7ce71d43c68501ff9a5dbf8467f3.jpg)

![489fe7e5850025cc4f945aa8f89982e7925b7d0684dcbc151fe6f4de413bb3e4.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/images/489fe7e5850025cc4f945aa8f89982e7925b7d0684dcbc151fe6f4de413bb3e4.jpg)

![71f19c3983fb66e7d4bdd96c8e4c81999af82896ea15ce8aded34475f2578cbb.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/images/71f19c3983fb66e7d4bdd96c8e4c81999af82896ea15ce8aded34475f2578cbb.jpg)

![74742c43eb537c0b93846fb8be38fee4b66032dd292a0e7d0eae8dd45fca790b.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/images/74742c43eb537c0b93846fb8be38fee4b66032dd292a0e7d0eae8dd45fca790b.jpg)

![9cd322a30243f399a956da24a485d8eea3934b9da4edad437d27ddfc897b2e7e.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/images/9cd322a30243f399a956da24a485d8eea3934b9da4edad437d27ddfc897b2e7e.jpg)

![a257fe2576391ac22d3fd8ef15137efb321fc826c8b63816e965e67c1fc16590.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/images/a257fe2576391ac22d3fd8ef15137efb321fc826c8b63816e965e67c1fc16590.jpg)

![ae4f8e27a495b226d72d2ef4511feb4b9fa1da81b7523f3c914c8b624880f1f6.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/images/ae4f8e27a495b226d72d2ef4511feb4b9fa1da81b7523f3c914c8b624880f1f6.jpg)

![b814b40d7475035b21379f4aa8142dad2020c27d683971d48cba3da75a80a1d7.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/images/b814b40d7475035b21379f4aa8142dad2020c27d683971d48cba3da75a80a1d7.jpg)

![b825694cc6da5afea4db8f6c6134d4ea71c53e71ca94d29168b515149731b84c.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/images/b825694cc6da5afea4db8f6c6134d4ea71c53e71ca94d29168b515149731b84c.jpg)

![ccbaf8c12ec9f207abced9547941c95f6b09b6d93c70ec207847a5f159c38fdd.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/images/ccbaf8c12ec9f207abced9547941c95f6b09b6d93c70ec207847a5f159c38fdd.jpg)

![cd9f152da5f825fcbeaf01803eea86dab4db7c28b423c26a45b2f755367c4d92.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/images/cd9f152da5f825fcbeaf01803eea86dab4db7c28b423c26a45b2f755367c4d92.jpg)

![d172c63a22bf089b0182850dfa5aee27de6760a98af2befcf34074ea6731bc1c.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/images/d172c63a22bf089b0182850dfa5aee27de6760a98af2befcf34074ea6731bc1c.jpg)

![d45f4cb85f597a33e9547a081e9b767d2e9f67d00329f6935d1f887c18d5d9f9.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/images/d45f4cb85f597a33e9547a081e9b767d2e9f67d00329f6935d1f887c18d5d9f9.jpg)

![d6a865d5cb41dbb42aa477f00d6eb4534b12f18f3585f32c603a32a80dfbcd81.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/images/d6a865d5cb41dbb42aa477f00d6eb4534b12f18f3585f32c603a32a80dfbcd81.jpg)

![db2c694a9ac91732d5d10e80d762b3f4fa81cc0efa3824cfafd664dd0dbc674c.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/images/db2c694a9ac91732d5d10e80d762b3f4fa81cc0efa3824cfafd664dd0dbc674c.jpg)

![f3d6caa046f8bd802fab9c9b85b0c44c07a09b6abb7d8e0879e5d6b9b603bdab.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/images/f3d6caa046f8bd802fab9c9b85b0c44c07a09b6abb7d8e0879e5d6b9b603bdab.jpg)

![fe2402ce678684538bfe721120f37159ccc854f8c6f6fec179ff6d3f54ead0ac.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/images/fe2402ce678684538bfe721120f37159ccc854f8c6f6fec179ff6d3f54ead0ac.jpg)

### Tables

![022b4d8c8f0a1ea6cc92bd20e65b78c292a3a99c355ab05a2374f68ef4281064.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/tables/022b4d8c8f0a1ea6cc92bd20e65b78c292a3a99c355ab05a2374f68ef4281064.jpg)

![2aea60b009defe24ae21bc97ebbacd650c988c724bb302b6ed20c10392c92bf7.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/tables/2aea60b009defe24ae21bc97ebbacd650c988c724bb302b6ed20c10392c92bf7.jpg)

![36f43be02355d984b8dc92f23b6216b6be8a9bc8c02008da8391c8aa7ab4d1f1.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/tables/36f43be02355d984b8dc92f23b6216b6be8a9bc8c02008da8391c8aa7ab4d1f1.jpg)

![4ca6823140bf0eb0dd026653103fb2173959fb33ae13ec6e73daf19bd8c424f4.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/tables/4ca6823140bf0eb0dd026653103fb2173959fb33ae13ec6e73daf19bd8c424f4.jpg)

![56f8a1caac27b395c09b254229ae535c56e6773d0dab627234919f3bf5f5f7d5.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/tables/56f8a1caac27b395c09b254229ae535c56e6773d0dab627234919f3bf5f5f7d5.jpg)

![6d10adf6de080d38f84bf58c4304000d6b23bdf75274f699f9e663497b5e8df9.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/tables/6d10adf6de080d38f84bf58c4304000d6b23bdf75274f699f9e663497b5e8df9.jpg)

![9b39eee9256197a2995aebeba516f1fb15beacaa1690d7911edc9858e4645802.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/tables/9b39eee9256197a2995aebeba516f1fb15beacaa1690d7911edc9858e4645802.jpg)

![9ec2282e2701a390d1e1858d93b7c23df2aa5f96072627c0da922cad210fba36.jpg](../iclr_results/111_Reasoning Elicitation in Language Models via Counterfactual Feedback/tables/9ec2282e2701a390d1e1858d93b7c23df2aa5f96072627c0da922cad210fba36.jpg)

## Attention as a Hypernetwork


### Images

![2032eac2e95b556f4e629e0486ea54521e5a610dc33fcd9be5f4c7c05669efd1.jpg](../iclr_results/112_Attention as a Hypernetwork/images/2032eac2e95b556f4e629e0486ea54521e5a610dc33fcd9be5f4c7c05669efd1.jpg)

![410d4b3a4f22d8a6b1421b58138f9c9232b1edc9ac1b054490e0239dc1f79b8d.jpg](../iclr_results/112_Attention as a Hypernetwork/images/410d4b3a4f22d8a6b1421b58138f9c9232b1edc9ac1b054490e0239dc1f79b8d.jpg)

![4b938dd9a69d8391d6aa085bdb07dba7ce8c48cb15b7ca84d8ada1630f99411a.jpg](../iclr_results/112_Attention as a Hypernetwork/images/4b938dd9a69d8391d6aa085bdb07dba7ce8c48cb15b7ca84d8ada1630f99411a.jpg)

![508a6fb0709ac00fe462503ca2ce2e8749434f0c30df16cbbfea012667977836.jpg](../iclr_results/112_Attention as a Hypernetwork/images/508a6fb0709ac00fe462503ca2ce2e8749434f0c30df16cbbfea012667977836.jpg)

![54ecc08b1ce40153ac608cf6c9a2a78bd8988c767f41b67dd213696141cc8ebc.jpg](../iclr_results/112_Attention as a Hypernetwork/images/54ecc08b1ce40153ac608cf6c9a2a78bd8988c767f41b67dd213696141cc8ebc.jpg)

![5d662740c2f3e2d72549f1ec1af8019714a93aace103788e163bf5938924ea78.jpg](../iclr_results/112_Attention as a Hypernetwork/images/5d662740c2f3e2d72549f1ec1af8019714a93aace103788e163bf5938924ea78.jpg)

![601c0c8d470d30250557382d9c53738423744682c1bb893f7af6061a93dd836c.jpg](../iclr_results/112_Attention as a Hypernetwork/images/601c0c8d470d30250557382d9c53738423744682c1bb893f7af6061a93dd836c.jpg)

![8165512b29de4ffb16784b91ef454ef1c81ff22589fb1d47a2f6078dfdc883b0.jpg](../iclr_results/112_Attention as a Hypernetwork/images/8165512b29de4ffb16784b91ef454ef1c81ff22589fb1d47a2f6078dfdc883b0.jpg)

![a341987aa8653223242fd4570ca4479edca037c7629f8ce799beec085a2e0330.jpg](../iclr_results/112_Attention as a Hypernetwork/images/a341987aa8653223242fd4570ca4479edca037c7629f8ce799beec085a2e0330.jpg)

![aa5540ed80f3523cb3bd472303d1d81242eae6364cd70fd0a2014d4335e98010.jpg](../iclr_results/112_Attention as a Hypernetwork/images/aa5540ed80f3523cb3bd472303d1d81242eae6364cd70fd0a2014d4335e98010.jpg)

![ab2e82c0c335f75f65d5db390dc4c77d0ae877bf70b03abf59b634ba12c6f4ae.jpg](../iclr_results/112_Attention as a Hypernetwork/images/ab2e82c0c335f75f65d5db390dc4c77d0ae877bf70b03abf59b634ba12c6f4ae.jpg)

![b3032cb9f24f5d9026f38fe6c8208962e1c7e96ca4c68b4b99fdceb590f90168.jpg](../iclr_results/112_Attention as a Hypernetwork/images/b3032cb9f24f5d9026f38fe6c8208962e1c7e96ca4c68b4b99fdceb590f90168.jpg)

![c0803e244ac071ee0ea180e589c374e30870583c6c9e496b5786f190e54526f1.jpg](../iclr_results/112_Attention as a Hypernetwork/images/c0803e244ac071ee0ea180e589c374e30870583c6c9e496b5786f190e54526f1.jpg)

![c0bb56ef51e2ba54f5006c783b47cbc63c4f5fdd5ba935ada31b25355c970288.jpg](../iclr_results/112_Attention as a Hypernetwork/images/c0bb56ef51e2ba54f5006c783b47cbc63c4f5fdd5ba935ada31b25355c970288.jpg)

![c649ed2a426d264ce6f0da9ee1132e860364f725132f8de18cc2a6ae92ca2cd8.jpg](../iclr_results/112_Attention as a Hypernetwork/images/c649ed2a426d264ce6f0da9ee1132e860364f725132f8de18cc2a6ae92ca2cd8.jpg)

![d5dca0f60240e4ff9252adcb8e9c26da754d1d27e42508eb26a3dad53247655b.jpg](../iclr_results/112_Attention as a Hypernetwork/images/d5dca0f60240e4ff9252adcb8e9c26da754d1d27e42508eb26a3dad53247655b.jpg)

### Tables

![28ebd87711fabbf07ab541065dd88c91b4fc5d469b57ac9efd80d1bdfb4dd9f6.jpg](../iclr_results/112_Attention as a Hypernetwork/tables/28ebd87711fabbf07ab541065dd88c91b4fc5d469b57ac9efd80d1bdfb4dd9f6.jpg)

![b3f69554760dfe3a6224f8d00309c0532fb0c09253128168eeca6f70a697bc90.jpg](../iclr_results/112_Attention as a Hypernetwork/tables/b3f69554760dfe3a6224f8d00309c0532fb0c09253128168eeca6f70a697bc90.jpg)

![b6e098de157f9a8bedd8c0b97c074c802dcdb0cdc61a263f795f9a0e3b656260.jpg](../iclr_results/112_Attention as a Hypernetwork/tables/b6e098de157f9a8bedd8c0b97c074c802dcdb0cdc61a263f795f9a0e3b656260.jpg)

## Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues


### Images

![112f8bf9f0e85bce72252f5b5ff5d51eaae3ba00be95257eb1db6268e9c32768.jpg](../iclr_results/113_Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues/images/112f8bf9f0e85bce72252f5b5ff5d51eaae3ba00be95257eb1db6268e9c32768.jpg)

![2caf76a620294290c784c2ab9f3ee789d4e11f857ba770af3fd74e224dfef8f7.jpg](../iclr_results/113_Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues/images/2caf76a620294290c784c2ab9f3ee789d4e11f857ba770af3fd74e224dfef8f7.jpg)

![50eeac28266ed27c428a7856a97276267f221156cce41545c201c16227c96db8.jpg](../iclr_results/113_Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues/images/50eeac28266ed27c428a7856a97276267f221156cce41545c201c16227c96db8.jpg)

![52f5b27b0069b6dde6cfdf685a3b4e144cdd6abab5c4ddcc91015efc49d46fee.jpg](../iclr_results/113_Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues/images/52f5b27b0069b6dde6cfdf685a3b4e144cdd6abab5c4ddcc91015efc49d46fee.jpg)

![641ca3f815911ab1316f2240a3a4b67d2c9784845447d10171a604b70913165e.jpg](../iclr_results/113_Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues/images/641ca3f815911ab1316f2240a3a4b67d2c9784845447d10171a604b70913165e.jpg)

![76c5cf6b354b0f69184681c5b89453dd5e80636a8b6960a8f84ff4a4e107c2a0.jpg](../iclr_results/113_Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues/images/76c5cf6b354b0f69184681c5b89453dd5e80636a8b6960a8f84ff4a4e107c2a0.jpg)

![7e6e5c6fa5b9a6043c90c0b5af8af4d6d30a1eefbc001d7fb457e31a54117508.jpg](../iclr_results/113_Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues/images/7e6e5c6fa5b9a6043c90c0b5af8af4d6d30a1eefbc001d7fb457e31a54117508.jpg)

![8eb2ac98242f807525c15618df8745eb01c0467bf3e8e0818cfe692e57a1f0c1.jpg](../iclr_results/113_Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues/images/8eb2ac98242f807525c15618df8745eb01c0467bf3e8e0818cfe692e57a1f0c1.jpg)

![abfea7bcd78bb74ae962a2f7c70f1dd02f6f9749185bf74d6fc9089865a7bf37.jpg](../iclr_results/113_Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues/images/abfea7bcd78bb74ae962a2f7c70f1dd02f6f9749185bf74d6fc9089865a7bf37.jpg)

![ba30db46e3ee2b25dd4fe62996720ffe8cce2026d85be5bf4104362bf808da07.jpg](../iclr_results/113_Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues/images/ba30db46e3ee2b25dd4fe62996720ffe8cce2026d85be5bf4104362bf808da07.jpg)

![ce424b27539aedd5ef391f7532a1598ce1451b29b65d8ae733bc1e4b689ef923.jpg](../iclr_results/113_Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues/images/ce424b27539aedd5ef391f7532a1598ce1451b29b65d8ae733bc1e4b689ef923.jpg)

![d8e9dd7db02985a53e9ce658aabb9baa645a79e194755bd72a9ecfc2b30f7037.jpg](../iclr_results/113_Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues/images/d8e9dd7db02985a53e9ce658aabb9baa645a79e194755bd72a9ecfc2b30f7037.jpg)

![eac3fac010e652339760ecfda9a3e649dd9cf81c2b6fde9cd5d5995a2eaff7fd.jpg](../iclr_results/113_Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues/images/eac3fac010e652339760ecfda9a3e649dd9cf81c2b6fde9cd5d5995a2eaff7fd.jpg)

### Tables

![9d0194109beb170b03abca38ccc89760acfc4cb8fa49f4706ee8b5d193cac9b4.jpg](../iclr_results/113_Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues/tables/9d0194109beb170b03abca38ccc89760acfc4cb8fa49f4706ee8b5d193cac9b4.jpg)

![b0da2967ef48dfe21be89bcd91d2176c28e19eba954aa750cf73b0ef325ddfa1.jpg](../iclr_results/113_Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues/tables/b0da2967ef48dfe21be89bcd91d2176c28e19eba954aa750cf73b0ef325ddfa1.jpg)

![b7b40e14a7a50bf21be9705b8de94b51deac07036b57cca892457214902dce55.jpg](../iclr_results/113_Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues/tables/b7b40e14a7a50bf21be9705b8de94b51deac07036b57cca892457214902dce55.jpg)

![b82bc4671517620910b0f1f05ce77031a527a584a43a76b6c5953ba2a02e827e.jpg](../iclr_results/113_Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues/tables/b82bc4671517620910b0f1f05ce77031a527a584a43a76b6c5953ba2a02e827e.jpg)

![ef56a91ca21cc357cff4591b938f3388559e0594dacc7a8e7bf53dad36c1be76.jpg](../iclr_results/113_Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues/tables/ef56a91ca21cc357cff4591b938f3388559e0594dacc7a8e7bf53dad36c1be76.jpg)

## Learning Randomized Algorithms with Transformers


### Images

![1737221bdd23e456cabaf9f119b8255b53f3086a5af3a1bb22f2bdc50444d479.jpg](../iclr_results/114_Learning Randomized Algorithms with Transformers/images/1737221bdd23e456cabaf9f119b8255b53f3086a5af3a1bb22f2bdc50444d479.jpg)

![3e849264019817c0ea38271d49963f5c68343702eefcf191d19a7ab5cd3899b9.jpg](../iclr_results/114_Learning Randomized Algorithms with Transformers/images/3e849264019817c0ea38271d49963f5c68343702eefcf191d19a7ab5cd3899b9.jpg)

![537a148a3bf3781fc59a5b82b6706bb16b1ecf7e76fc53053cacdd08e16e78f4.jpg](../iclr_results/114_Learning Randomized Algorithms with Transformers/images/537a148a3bf3781fc59a5b82b6706bb16b1ecf7e76fc53053cacdd08e16e78f4.jpg)

![66fca5b9e0345db9e4f2cfa2e0002bffc5f15c4261a6c404a3d700321a833a7a.jpg](../iclr_results/114_Learning Randomized Algorithms with Transformers/images/66fca5b9e0345db9e4f2cfa2e0002bffc5f15c4261a6c404a3d700321a833a7a.jpg)

![6dee92f4a13c94a3aae9b394364572f89f58b3e6f536e59ad88bda9cf85279e4.jpg](../iclr_results/114_Learning Randomized Algorithms with Transformers/images/6dee92f4a13c94a3aae9b394364572f89f58b3e6f536e59ad88bda9cf85279e4.jpg)

![78391803567d7f8fc830aff80d2904f7ead69a44d4d49fe60ff2f5385539fdb9.jpg](../iclr_results/114_Learning Randomized Algorithms with Transformers/images/78391803567d7f8fc830aff80d2904f7ead69a44d4d49fe60ff2f5385539fdb9.jpg)

![7e59dadf267f8c0d7a2c913aeb3c47fdf1256d6b9f54c90d37c2a86653e263bc.jpg](../iclr_results/114_Learning Randomized Algorithms with Transformers/images/7e59dadf267f8c0d7a2c913aeb3c47fdf1256d6b9f54c90d37c2a86653e263bc.jpg)

![89c5029dd042d4f84ac4ea571d94674be894fe8dea34bb4844a1ae715a7c2785.jpg](../iclr_results/114_Learning Randomized Algorithms with Transformers/images/89c5029dd042d4f84ac4ea571d94674be894fe8dea34bb4844a1ae715a7c2785.jpg)

![d85f4cf113e3742b44a72f6d1969e18831e13ae5071ebeed17594a54d2c1d067.jpg](../iclr_results/114_Learning Randomized Algorithms with Transformers/images/d85f4cf113e3742b44a72f6d1969e18831e13ae5071ebeed17594a54d2c1d067.jpg)

![dea510726abc2cf7660195d6409b7f45ae36f359f9d1a9a62fba5ecf8dd2ab37.jpg](../iclr_results/114_Learning Randomized Algorithms with Transformers/images/dea510726abc2cf7660195d6409b7f45ae36f359f9d1a9a62fba5ecf8dd2ab37.jpg)

### Tables

![0cfeb991bab79f22ca4b0ef18d800dc8736414a2ef8b695b7d29cc5f5aca8c99.jpg](../iclr_results/114_Learning Randomized Algorithms with Transformers/tables/0cfeb991bab79f22ca4b0ef18d800dc8736414a2ef8b695b7d29cc5f5aca8c99.jpg)

![425608e59de352f81486c0f54b71dbcc85ad9bb9963ea2cf35402c6957e0e9d7.jpg](../iclr_results/114_Learning Randomized Algorithms with Transformers/tables/425608e59de352f81486c0f54b71dbcc85ad9bb9963ea2cf35402c6957e0e9d7.jpg)

![cbe2c291ae3ef19e5a02f951a7259e10a20a565a609d9d6e46e140fe1a38c2a4.jpg](../iclr_results/114_Learning Randomized Algorithms with Transformers/tables/cbe2c291ae3ef19e5a02f951a7259e10a20a565a609d9d6e46e140fe1a38c2a4.jpg)

## Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement


### Images

![0e1114daffdc3c87522e72697b2baf1b8ca56d9466c9504a0dfa9a0b9608783b.jpg](../iclr_results/115_Trust or Escalate_ LLM Judges with Provable Guarantees for Human Agreement/images/0e1114daffdc3c87522e72697b2baf1b8ca56d9466c9504a0dfa9a0b9608783b.jpg)

![0ef53991d4f542b214873b4b6623a4cf2d2e7d3198d9c4d4507e147830364c40.jpg](../iclr_results/115_Trust or Escalate_ LLM Judges with Provable Guarantees for Human Agreement/images/0ef53991d4f542b214873b4b6623a4cf2d2e7d3198d9c4d4507e147830364c40.jpg)

![2b83d9517a937d13325bc14b05888279cb374363167b78dea0c022d2d06127c8.jpg](../iclr_results/115_Trust or Escalate_ LLM Judges with Provable Guarantees for Human Agreement/images/2b83d9517a937d13325bc14b05888279cb374363167b78dea0c022d2d06127c8.jpg)

![bb100cc2e61fd557af5fd4786835ab4e84b13a1fbc39f0cadd34555f2aeef266.jpg](../iclr_results/115_Trust or Escalate_ LLM Judges with Provable Guarantees for Human Agreement/images/bb100cc2e61fd557af5fd4786835ab4e84b13a1fbc39f0cadd34555f2aeef266.jpg)

![c66383fbcc423e2fc250852674aed7567aaf00298e00c6e1617c768e181494cd.jpg](../iclr_results/115_Trust or Escalate_ LLM Judges with Provable Guarantees for Human Agreement/images/c66383fbcc423e2fc250852674aed7567aaf00298e00c6e1617c768e181494cd.jpg)

### Tables

![0c3ee4a892650874c004582c0f1d81980f5392604d235c1902b70ef738385328.jpg](../iclr_results/115_Trust or Escalate_ LLM Judges with Provable Guarantees for Human Agreement/tables/0c3ee4a892650874c004582c0f1d81980f5392604d235c1902b70ef738385328.jpg)

![0d10a17be6c9b9c7725bc5b6f048d7b81ad4831c5bbaa8d05f65f03208142bab.jpg](../iclr_results/115_Trust or Escalate_ LLM Judges with Provable Guarantees for Human Agreement/tables/0d10a17be6c9b9c7725bc5b6f048d7b81ad4831c5bbaa8d05f65f03208142bab.jpg)

![14341b620d6fdb18b167bdaf860637ffa805aabf7769c9266feec0d9f6e5461a.jpg](../iclr_results/115_Trust or Escalate_ LLM Judges with Provable Guarantees for Human Agreement/tables/14341b620d6fdb18b167bdaf860637ffa805aabf7769c9266feec0d9f6e5461a.jpg)

![2032f00de2595402ec3919ee3f36da227002f3ecb4bd07645fe8687349670c9b.jpg](../iclr_results/115_Trust or Escalate_ LLM Judges with Provable Guarantees for Human Agreement/tables/2032f00de2595402ec3919ee3f36da227002f3ecb4bd07645fe8687349670c9b.jpg)

![24c816f3286e2e079d52f1a8b62fc411685b67a311f6ebafe0d9d32b61acb9f9.jpg](../iclr_results/115_Trust or Escalate_ LLM Judges with Provable Guarantees for Human Agreement/tables/24c816f3286e2e079d52f1a8b62fc411685b67a311f6ebafe0d9d32b61acb9f9.jpg)

![3f49cdc6d3d903dba071f5ae4da1b21784c42163d47b313f2834ad9c8e35a0f4.jpg](../iclr_results/115_Trust or Escalate_ LLM Judges with Provable Guarantees for Human Agreement/tables/3f49cdc6d3d903dba071f5ae4da1b21784c42163d47b313f2834ad9c8e35a0f4.jpg)

![7a13c91ee3f26aa410e378f28ba06adaed3f9393fbd4f3dd6baa5aaa7dea6e85.jpg](../iclr_results/115_Trust or Escalate_ LLM Judges with Provable Guarantees for Human Agreement/tables/7a13c91ee3f26aa410e378f28ba06adaed3f9393fbd4f3dd6baa5aaa7dea6e85.jpg)

![905db7c28a4f4a61ee614b5edbf08dce497e1cef67596bdefaf68928ac969559.jpg](../iclr_results/115_Trust or Escalate_ LLM Judges with Provable Guarantees for Human Agreement/tables/905db7c28a4f4a61ee614b5edbf08dce497e1cef67596bdefaf68928ac969559.jpg)

![b620ae9d742c624adaa683e4a4a0c874dd3679ee0250c14dcaf426f3f0e5538c.jpg](../iclr_results/115_Trust or Escalate_ LLM Judges with Provable Guarantees for Human Agreement/tables/b620ae9d742c624adaa683e4a4a0c874dd3679ee0250c14dcaf426f3f0e5538c.jpg)

![ccfbab025ccf854935e0b884d663200ded015bc3698c3927e126e98d6df8a977.jpg](../iclr_results/115_Trust or Escalate_ LLM Judges with Provable Guarantees for Human Agreement/tables/ccfbab025ccf854935e0b884d663200ded015bc3698c3927e126e98d6df8a977.jpg)

![d14e6bc2d410ec6d6292ca7a2a5d7f7a9908b960855c8953441a922376b5a70a.jpg](../iclr_results/115_Trust or Escalate_ LLM Judges with Provable Guarantees for Human Agreement/tables/d14e6bc2d410ec6d6292ca7a2a5d7f7a9908b960855c8953441a922376b5a70a.jpg)

## HiRA: Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models


### Images

![112a808d1c081e6c20f67cf74d147e6ae56f31fbfeec29a8d88a9c9be9a7589b.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/images/112a808d1c081e6c20f67cf74d147e6ae56f31fbfeec29a8d88a9c9be9a7589b.jpg)

![137169e82e066989b9641ad1bbe9dea8e79e3f576a3604d491d280cdddd7ef1d.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/images/137169e82e066989b9641ad1bbe9dea8e79e3f576a3604d491d280cdddd7ef1d.jpg)

![164524c3162b0588d8658af1300b73c57510505ebcd889ba76ee13d0e7ba3f3e.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/images/164524c3162b0588d8658af1300b73c57510505ebcd889ba76ee13d0e7ba3f3e.jpg)

![2e8bc835f443eb27559af27b55b6ad1d53e5834d2145f0fa75f4120788499edc.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/images/2e8bc835f443eb27559af27b55b6ad1d53e5834d2145f0fa75f4120788499edc.jpg)

![68bc0b42a8caab61ef96a1b413670ed4c1010569ba26105622fdaba3582bac94.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/images/68bc0b42a8caab61ef96a1b413670ed4c1010569ba26105622fdaba3582bac94.jpg)

![6be0a9102ded4a8768f1ce74a7588ddc0871bc995b617153210e1e09a5912859.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/images/6be0a9102ded4a8768f1ce74a7588ddc0871bc995b617153210e1e09a5912859.jpg)

![6f01b5b2f471b06183b4edc9215bf4fffceaf9fb84382011316dcf0123918625.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/images/6f01b5b2f471b06183b4edc9215bf4fffceaf9fb84382011316dcf0123918625.jpg)

![7399c4eaac70c530232cff49386a0e1bf133f538b10009ad9b8cc2ed387b183d.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/images/7399c4eaac70c530232cff49386a0e1bf133f538b10009ad9b8cc2ed387b183d.jpg)

![93c4f009d22dc2093e0bf864690e47bd40dc83eaf05c4ca7b51ecaacae63bee7.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/images/93c4f009d22dc2093e0bf864690e47bd40dc83eaf05c4ca7b51ecaacae63bee7.jpg)

![a767b7b1fb0b2c0e5122d7dbe20b699ef3fd01e00726e8d5a85a6ccbf37abe60.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/images/a767b7b1fb0b2c0e5122d7dbe20b699ef3fd01e00726e8d5a85a6ccbf37abe60.jpg)

![fd77e42fa9595d7fda8f569c93602a9ff324cbf89d8dc34060a1822097679ef1.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/images/fd77e42fa9595d7fda8f569c93602a9ff324cbf89d8dc34060a1822097679ef1.jpg)

### Tables

![04033a70a0f56d75700ec0805952d0a2eeafc856fe5e5d0d8fe9b9acbb84c49e.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/tables/04033a70a0f56d75700ec0805952d0a2eeafc856fe5e5d0d8fe9b9acbb84c49e.jpg)

![10bd5f1cb57cc810fa22859e5b637ad5795cbb9a6909e9ca879b37bd9cb5b758.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/tables/10bd5f1cb57cc810fa22859e5b637ad5795cbb9a6909e9ca879b37bd9cb5b758.jpg)

![259b5a873d81dcf11b1d3709ab9e5131d7f86874aad4d27894213c8718dc5e46.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/tables/259b5a873d81dcf11b1d3709ab9e5131d7f86874aad4d27894213c8718dc5e46.jpg)

![285c7718d09706e805b958caffa6e1bbbf520b65c72f20143afd4ca236a13498.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/tables/285c7718d09706e805b958caffa6e1bbbf520b65c72f20143afd4ca236a13498.jpg)

![640371e9670d85dce9b01f97b1efd14aad3365311594f09db2a33daf5bdc3682.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/tables/640371e9670d85dce9b01f97b1efd14aad3365311594f09db2a33daf5bdc3682.jpg)

![641df8ccf521d0027c205d39a9d9d849303e9017329772db72f27806d03238a4.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/tables/641df8ccf521d0027c205d39a9d9d849303e9017329772db72f27806d03238a4.jpg)

![7b06c7f45b3949ed5090766538d88c1d3e512aef71ede78d45a4854f2403c2b2.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/tables/7b06c7f45b3949ed5090766538d88c1d3e512aef71ede78d45a4854f2403c2b2.jpg)

![83e0cb433f1a2fcb6632bd8ca2cd94eb084ac08f8898bfcf7ec9ccfcf74dabd4.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/tables/83e0cb433f1a2fcb6632bd8ca2cd94eb084ac08f8898bfcf7ec9ccfcf74dabd4.jpg)

![846ff2fb83760df69478ed26cd015ab0b612f4a9f8ffc13d68fbf2a1dabab53f.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/tables/846ff2fb83760df69478ed26cd015ab0b612f4a9f8ffc13d68fbf2a1dabab53f.jpg)

![bacd81633ed41b5f76755e5723469fab3cf0d3e2decca2149c1fc7854bd9bba8.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/tables/bacd81633ed41b5f76755e5723469fab3cf0d3e2decca2149c1fc7854bd9bba8.jpg)

![cae53408ddbbb7a898e4fc8dd8710fc94fc8f1095cbcfaf77764a0c97f2a3813.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/tables/cae53408ddbbb7a898e4fc8dd8710fc94fc8f1095cbcfaf77764a0c97f2a3813.jpg)

![d3ccf11b1352c00c20854506954a419daf81582368ed3a1c9398f78d851e7f7f.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/tables/d3ccf11b1352c00c20854506954a419daf81582368ed3a1c9398f78d851e7f7f.jpg)

![e7c41d79e487ddbfd7536389aaa056ce2e6ff51dc550411478785d2b906bb176.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/tables/e7c41d79e487ddbfd7536389aaa056ce2e6ff51dc550411478785d2b906bb176.jpg)

![ea63962d293f137fe8416d032fa998cd6da6dc0994b8735666ad8f5fe79a33e9.jpg](../iclr_results/116_HiRA_ Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models/tables/ea63962d293f137fe8416d032fa998cd6da6dc0994b8735666ad8f5fe79a33e9.jpg)

## Proteina: Scaling Flow-based Protein Structure Generative Models


### Images

![05338d893e63e1380b8b8c092c5b2e57cbc65ff2d52758db9d9ad40ff1b397c4.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/05338d893e63e1380b8b8c092c5b2e57cbc65ff2d52758db9d9ad40ff1b397c4.jpg)

![1b7277c16206ece76d0ef1c45972c484badb0211ffeef4917be4ccc54c16239a.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/1b7277c16206ece76d0ef1c45972c484badb0211ffeef4917be4ccc54c16239a.jpg)

![21104b66dfed6ea10e95879850786eea74ea0670c05815bcf6e7dc111e7477b1.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/21104b66dfed6ea10e95879850786eea74ea0670c05815bcf6e7dc111e7477b1.jpg)

![25ead3da68caefb0b09be25319880b18470d441f2023874bee8983db157c2488.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/25ead3da68caefb0b09be25319880b18470d441f2023874bee8983db157c2488.jpg)

![2c99fa1f511abaa2508a77c9cd5d12481808eef5c168915947956c80be870dee.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/2c99fa1f511abaa2508a77c9cd5d12481808eef5c168915947956c80be870dee.jpg)

![3d04c7975ac1d11bf81edaeb776b2123a3bca04b25569d48f52fd19bebc729d1.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/3d04c7975ac1d11bf81edaeb776b2123a3bca04b25569d48f52fd19bebc729d1.jpg)

![4118e2db35dddb50150f4351decb4107f441184cde64d2eb250cd1caf5f396c0.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/4118e2db35dddb50150f4351decb4107f441184cde64d2eb250cd1caf5f396c0.jpg)

![41205599648703870afa58c0e0b37b460070d71b7a488b6fa9964549327adaa6.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/41205599648703870afa58c0e0b37b460070d71b7a488b6fa9964549327adaa6.jpg)

![44584a4ecd474166e9e945fbc9796ddb3069b2b7e4a2cad4316feb8f81e8f69e.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/44584a4ecd474166e9e945fbc9796ddb3069b2b7e4a2cad4316feb8f81e8f69e.jpg)

![44c71c064e9ef3c67406053a4236ec7f3b27daa0ffb688e3dcff35fec603108f.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/44c71c064e9ef3c67406053a4236ec7f3b27daa0ffb688e3dcff35fec603108f.jpg)

![50efd7c7ebfaf310d13f0cdedc4d01f451b4729c1c863f6877186b1fdd833ea7.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/50efd7c7ebfaf310d13f0cdedc4d01f451b4729c1c863f6877186b1fdd833ea7.jpg)

![51c10417f74e483a75a38946857eb9fef23557bb8483b66bf851b9d9a4b47d25.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/51c10417f74e483a75a38946857eb9fef23557bb8483b66bf851b9d9a4b47d25.jpg)

![6d8296a75145efd3db72fa1dfcde667e1984cff9d3676a32648f313185335442.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/6d8296a75145efd3db72fa1dfcde667e1984cff9d3676a32648f313185335442.jpg)

![6e3e0398b82a5dd877209696b29c3414d7b83d83cb78fea578ba1c74369cd517.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/6e3e0398b82a5dd877209696b29c3414d7b83d83cb78fea578ba1c74369cd517.jpg)

![72cc7fe5eaed65368b1262a4c5a408326842da71e1bf06da4a89001a161d0002.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/72cc7fe5eaed65368b1262a4c5a408326842da71e1bf06da4a89001a161d0002.jpg)

![79df876d2079788b29fd25ced386bd75d5e5c5d364423c0beb2d36ceaa01bdc9.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/79df876d2079788b29fd25ced386bd75d5e5c5d364423c0beb2d36ceaa01bdc9.jpg)

![833ad2e9b8f9ebd0ac5e5ae980678f3bbbc35ce950939ccc6e6276573c5b8648.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/833ad2e9b8f9ebd0ac5e5ae980678f3bbbc35ce950939ccc6e6276573c5b8648.jpg)

![a19da17464276d053d541106d0bffc6bf61045b54a0341fbd4bb416321b65cfd.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/a19da17464276d053d541106d0bffc6bf61045b54a0341fbd4bb416321b65cfd.jpg)

![ad517d9a9e4dd774e7e943a09853e95dc3f92b8aa4691e56738063dee0cfb921.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/ad517d9a9e4dd774e7e943a09853e95dc3f92b8aa4691e56738063dee0cfb921.jpg)

![b22a6e52ba38f27301c319cdd9a05936931bf39a4656eba481701177b5a80d4e.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/b22a6e52ba38f27301c319cdd9a05936931bf39a4656eba481701177b5a80d4e.jpg)

![bd720245ee33267b3e504307b65b9067a7638f9ccc3eec69b406bce6139132f9.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/bd720245ee33267b3e504307b65b9067a7638f9ccc3eec69b406bce6139132f9.jpg)

![c85f96199fef8b11cd542ec73b5ab2dbf2c6a78744ac73c2ec83cb6f411b93b1.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/c85f96199fef8b11cd542ec73b5ab2dbf2c6a78744ac73c2ec83cb6f411b93b1.jpg)

![cc4b881f4d6b3115c0efa562340470dd10e84c05127afba60dff30dc71e312cc.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/cc4b881f4d6b3115c0efa562340470dd10e84c05127afba60dff30dc71e312cc.jpg)

![d7176bb148e61007646e47e7cc949e7748e7ec0f8a2538426be533fe8facb6ec.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/d7176bb148e61007646e47e7cc949e7748e7ec0f8a2538426be533fe8facb6ec.jpg)

![e907df560448dc0b6a48c7bd5cfb0f0b39566c72a9a19b70d694a8a38ce02eee.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/images/e907df560448dc0b6a48c7bd5cfb0f0b39566c72a9a19b70d694a8a38ce02eee.jpg)

### Tables

![1622e3372c245fdb6abbaf139ed2ef99b394c4b08670d6e1994f6f7371096075.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/tables/1622e3372c245fdb6abbaf139ed2ef99b394c4b08670d6e1994f6f7371096075.jpg)

![1df06ccbdb25e064d3e9a40adfff82ceb00b56b2ccfcba99334f7f6ffb4479c9.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/tables/1df06ccbdb25e064d3e9a40adfff82ceb00b56b2ccfcba99334f7f6ffb4479c9.jpg)

![34937d1d7c070769e1c6d4b99ee201e6af78fcca44706e165946894f24b78653.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/tables/34937d1d7c070769e1c6d4b99ee201e6af78fcca44706e165946894f24b78653.jpg)

![46934d390b18c48d3d6f1c77a8621179d1642cc8dfcbe9e159a79b0eef43451e.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/tables/46934d390b18c48d3d6f1c77a8621179d1642cc8dfcbe9e159a79b0eef43451e.jpg)

![7316f886d1095b8fb867a13341b40ab77ae5a1f421a665ccb9aa56dc0ef39eb1.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/tables/7316f886d1095b8fb867a13341b40ab77ae5a1f421a665ccb9aa56dc0ef39eb1.jpg)

![7df2c2190e3c56a52bbd360c341d5c0a557ad6b57d6006edf48c46e6c597edec.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/tables/7df2c2190e3c56a52bbd360c341d5c0a557ad6b57d6006edf48c46e6c597edec.jpg)

![7e2eec5c7a12ce01deb06a587bc3f277f0ef02ad8feec9238b6a13d4e3bbad1b.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/tables/7e2eec5c7a12ce01deb06a587bc3f277f0ef02ad8feec9238b6a13d4e3bbad1b.jpg)

![86f8f91d30b10f67a27f8c84502ed5f04689da856bc67210b4d946d24ffd5b7c.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/tables/86f8f91d30b10f67a27f8c84502ed5f04689da856bc67210b4d946d24ffd5b7c.jpg)

![8b599610fe0d7671de2682dbe7a5193d0942fa66632d95a7bc12919de429eefa.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/tables/8b599610fe0d7671de2682dbe7a5193d0942fa66632d95a7bc12919de429eefa.jpg)

![9b89e00bdd79de6d30b6d67f01db81b685327ffcc4bd118850c917467d3d9d36.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/tables/9b89e00bdd79de6d30b6d67f01db81b685327ffcc4bd118850c917467d3d9d36.jpg)

![9e3a0913a9c67de85b2821ee8ab80ea6f0492fddeaea1d68e56562f14128859c.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/tables/9e3a0913a9c67de85b2821ee8ab80ea6f0492fddeaea1d68e56562f14128859c.jpg)

![a89e0ac12a9998e8b5c9da2bbd46b879b2ec6cca8e59ee3e11c33c3e07c8dafb.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/tables/a89e0ac12a9998e8b5c9da2bbd46b879b2ec6cca8e59ee3e11c33c3e07c8dafb.jpg)

![a98c99164b01b0e1d0b65b50594d647a32a1126709afa6e50bbf0b1d8095c536.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/tables/a98c99164b01b0e1d0b65b50594d647a32a1126709afa6e50bbf0b1d8095c536.jpg)

![c0349b26f6815edf51cdbebaf705b68f73e272465854925c4ab3ff3e31cbf54c.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/tables/c0349b26f6815edf51cdbebaf705b68f73e272465854925c4ab3ff3e31cbf54c.jpg)

![f2bfd518746c6cb322aa77a8d41f189b2bd4b9fdb26cb43d135b490dc056b133.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/tables/f2bfd518746c6cb322aa77a8d41f189b2bd4b9fdb26cb43d135b490dc056b133.jpg)

![fd7e6f233057f49f1c19c6d9a81525df5403bbc5ebb3e2ea05131d3969c00a56.jpg](../iclr_results/117_Proteina_ Scaling Flow-based Protein Structure Generative Models/tables/fd7e6f233057f49f1c19c6d9a81525df5403bbc5ebb3e2ea05131d3969c00a56.jpg)

## REEF: Representation Encoding Fingerprints for Large Language Models


### Images

![13268f7c77dcdb82549910c02197ad5097736b533de4d022043458f75c62ba32.jpg](../iclr_results/118_REEF_ Representation Encoding Fingerprints for Large Language Models/images/13268f7c77dcdb82549910c02197ad5097736b533de4d022043458f75c62ba32.jpg)

![609692949135009fbc389ce91b1c3e5aca9d7eea8b6b6098f0fadb8deda4125c.jpg](../iclr_results/118_REEF_ Representation Encoding Fingerprints for Large Language Models/images/609692949135009fbc389ce91b1c3e5aca9d7eea8b6b6098f0fadb8deda4125c.jpg)

![7736360cb1c708ef4e8490c6086f94f7c75583d6000fdf7fc7184e4dac9e45d8.jpg](../iclr_results/118_REEF_ Representation Encoding Fingerprints for Large Language Models/images/7736360cb1c708ef4e8490c6086f94f7c75583d6000fdf7fc7184e4dac9e45d8.jpg)

![8ab7ec422b7cea167f4019645e77288192546b8187dbcf07f66526f372c57502.jpg](../iclr_results/118_REEF_ Representation Encoding Fingerprints for Large Language Models/images/8ab7ec422b7cea167f4019645e77288192546b8187dbcf07f66526f372c57502.jpg)

![a229663a328cb78e228cd9aac6d3400e393e8d272c932c83559ca7be15a73353.jpg](../iclr_results/118_REEF_ Representation Encoding Fingerprints for Large Language Models/images/a229663a328cb78e228cd9aac6d3400e393e8d272c932c83559ca7be15a73353.jpg)

![c4bafdcb0b3687826ce63a3c5356e1cfbb615b2c1bc1c4ce58b1e722cdaf89a3.jpg](../iclr_results/118_REEF_ Representation Encoding Fingerprints for Large Language Models/images/c4bafdcb0b3687826ce63a3c5356e1cfbb615b2c1bc1c4ce58b1e722cdaf89a3.jpg)

![d0d00449cdb7bb906a924a2296cd60b7ce8cf65e20ef160b9f4983d2f7c108dc.jpg](../iclr_results/118_REEF_ Representation Encoding Fingerprints for Large Language Models/images/d0d00449cdb7bb906a924a2296cd60b7ce8cf65e20ef160b9f4983d2f7c108dc.jpg)

![d5e0387431254483dc59b9825364fe6098da8a9f2181d37e2d1ba50f845c260c.jpg](../iclr_results/118_REEF_ Representation Encoding Fingerprints for Large Language Models/images/d5e0387431254483dc59b9825364fe6098da8a9f2181d37e2d1ba50f845c260c.jpg)

![f878810324df05d1970738c693fe2b091d1c53580dfd885a69639d27825aae24.jpg](../iclr_results/118_REEF_ Representation Encoding Fingerprints for Large Language Models/images/f878810324df05d1970738c693fe2b091d1c53580dfd885a69639d27825aae24.jpg)

### Tables

![3bc37f9191923bc624e0cc647eeab4a0afa16b0d3b70f8751225a09d4f0bbd9c.jpg](../iclr_results/118_REEF_ Representation Encoding Fingerprints for Large Language Models/tables/3bc37f9191923bc624e0cc647eeab4a0afa16b0d3b70f8751225a09d4f0bbd9c.jpg)

![3dc048b8c83fa3c359c11ce0afde354a7a60882bed6099005e4c040a629066ba.jpg](../iclr_results/118_REEF_ Representation Encoding Fingerprints for Large Language Models/tables/3dc048b8c83fa3c359c11ce0afde354a7a60882bed6099005e4c040a629066ba.jpg)

![458526ac3e93d72f37a3e37ec2a6f97ac68463344d351b3afba34c0bd40d6743.jpg](../iclr_results/118_REEF_ Representation Encoding Fingerprints for Large Language Models/tables/458526ac3e93d72f37a3e37ec2a6f97ac68463344d351b3afba34c0bd40d6743.jpg)

![574520ffd76d9d87cb865ad7cf5ed30f417ea361f34f7c19bcf737f23b70e195.jpg](../iclr_results/118_REEF_ Representation Encoding Fingerprints for Large Language Models/tables/574520ffd76d9d87cb865ad7cf5ed30f417ea361f34f7c19bcf737f23b70e195.jpg)

![d6097b8f1da51c83532e76965c9415f654e2a5fe92240a6c68c99f70b5659451.jpg](../iclr_results/118_REEF_ Representation Encoding Fingerprints for Large Language Models/tables/d6097b8f1da51c83532e76965c9415f654e2a5fe92240a6c68c99f70b5659451.jpg)

![ed703c8ce2b3511e9583e059f1864cb7f65783c62cb0f435261f13e784761742.jpg](../iclr_results/118_REEF_ Representation Encoding Fingerprints for Large Language Models/tables/ed703c8ce2b3511e9583e059f1864cb7f65783c62cb0f435261f13e784761742.jpg)

![ef6fdc0764782c0eeeeaf2a90dbc8f3dcb57e246009a0e171a1a1ff26683cb39.jpg](../iclr_results/118_REEF_ Representation Encoding Fingerprints for Large Language Models/tables/ef6fdc0764782c0eeeeaf2a90dbc8f3dcb57e246009a0e171a1a1ff26683cb39.jpg)

## A Decade's Battle on Dataset Bias: Are We There Yet?


### Images

![11dc30a0a82bf1d327df0c218dbeb12329575b05d55463fea936ffd52bca4bf3.jpg](../iclr_results/119_A Decade's Battle on Dataset Bias_ Are We There Yet_/images/11dc30a0a82bf1d327df0c218dbeb12329575b05d55463fea936ffd52bca4bf3.jpg)

![5d4f0ab433cc260e175a0eedb9b05353ebc5d8d38a120540db80c7841dcd044e.jpg](../iclr_results/119_A Decade's Battle on Dataset Bias_ Are We There Yet_/images/5d4f0ab433cc260e175a0eedb9b05353ebc5d8d38a120540db80c7841dcd044e.jpg)

![61c286cefb05962265973753e4f6aab47d820bb1db89cdc966ee0445f66a077d.jpg](../iclr_results/119_A Decade's Battle on Dataset Bias_ Are We There Yet_/images/61c286cefb05962265973753e4f6aab47d820bb1db89cdc966ee0445f66a077d.jpg)

![c332c6f6630a7e463c0a36eb64231d45156620253558d7694f8153ff7efeb6ce.jpg](../iclr_results/119_A Decade's Battle on Dataset Bias_ Are We There Yet_/images/c332c6f6630a7e463c0a36eb64231d45156620253558d7694f8153ff7efeb6ce.jpg)

![c8fff5177e4603c761ae2f5e3983c85f8112c57429dc06a274d74281ead6d06d.jpg](../iclr_results/119_A Decade's Battle on Dataset Bias_ Are We There Yet_/images/c8fff5177e4603c761ae2f5e3983c85f8112c57429dc06a274d74281ead6d06d.jpg)

![d54cbc3b6240278797b76955c9a44c28409b8c38ec6b6c46c5fa4bc65ea09a05.jpg](../iclr_results/119_A Decade's Battle on Dataset Bias_ Are We There Yet_/images/d54cbc3b6240278797b76955c9a44c28409b8c38ec6b6c46c5fa4bc65ea09a05.jpg)

![dcf70c24344bcd243757cc6d51f5bedc36e817fc20d0e4d149bbe5280a1e1eb9.jpg](../iclr_results/119_A Decade's Battle on Dataset Bias_ Are We There Yet_/images/dcf70c24344bcd243757cc6d51f5bedc36e817fc20d0e4d149bbe5280a1e1eb9.jpg)

![e8fe0fb762d920f76ce0626419266f3d3c28e5746953e730bf68db09f6818dcc.jpg](../iclr_results/119_A Decade's Battle on Dataset Bias_ Are We There Yet_/images/e8fe0fb762d920f76ce0626419266f3d3c28e5746953e730bf68db09f6818dcc.jpg)

### Tables

![117dde26c77e745ae7fd404c6830d88f69f62ba80257d4962201d02c05d89af1.jpg](../iclr_results/119_A Decade's Battle on Dataset Bias_ Are We There Yet_/tables/117dde26c77e745ae7fd404c6830d88f69f62ba80257d4962201d02c05d89af1.jpg)

![1493abf85f045c22b0e6d912455602588e021d609e46ebb8b3b24932be1ddc6e.jpg](../iclr_results/119_A Decade's Battle on Dataset Bias_ Are We There Yet_/tables/1493abf85f045c22b0e6d912455602588e021d609e46ebb8b3b24932be1ddc6e.jpg)

![62d7cd68f959bfa8ebc8bdba06fb2d7452b86e4e0b07bd2a544935a7f0539799.jpg](../iclr_results/119_A Decade's Battle on Dataset Bias_ Are We There Yet_/tables/62d7cd68f959bfa8ebc8bdba06fb2d7452b86e4e0b07bd2a544935a7f0539799.jpg)

![748460629a280ece2523ec6df10c93154a7bf62f444347ca743253e5c333b6e8.jpg](../iclr_results/119_A Decade's Battle on Dataset Bias_ Are We There Yet_/tables/748460629a280ece2523ec6df10c93154a7bf62f444347ca743253e5c333b6e8.jpg)

![937d5c712097a71c1f9110c9a9d89136a5ff323a14daeb7692bc914fa72981f4.jpg](../iclr_results/119_A Decade's Battle on Dataset Bias_ Are We There Yet_/tables/937d5c712097a71c1f9110c9a9d89136a5ff323a14daeb7692bc914fa72981f4.jpg)

![984deb3ce5b7fc87541294c050f73dc670b78dc2e074916f0ea80432a428702c.jpg](../iclr_results/119_A Decade's Battle on Dataset Bias_ Are We There Yet_/tables/984deb3ce5b7fc87541294c050f73dc670b78dc2e074916f0ea80432a428702c.jpg)

![bb5474239c9667caafe448849254d5aeac5adb27689021e19ee6c60ddf570607.jpg](../iclr_results/119_A Decade's Battle on Dataset Bias_ Are We There Yet_/tables/bb5474239c9667caafe448849254d5aeac5adb27689021e19ee6c60ddf570607.jpg)

![d605f0c79b1ea8e9063150785fc435768f30ab49db97a74ee33c72ffcb7b038c.jpg](../iclr_results/119_A Decade's Battle on Dataset Bias_ Are We There Yet_/tables/d605f0c79b1ea8e9063150785fc435768f30ab49db97a74ee33c72ffcb7b038c.jpg)

![f19f55ebd896a944d06a6b756229cfa62981fc2a29579deb883f9d0e68301f55.jpg](../iclr_results/119_A Decade's Battle on Dataset Bias_ Are We There Yet_/tables/f19f55ebd896a944d06a6b756229cfa62981fc2a29579deb883f9d0e68301f55.jpg)

![f2df136a221243d6be02e05a0ba2c8ae02d01e7b8c73437cf0474551167610b5.jpg](../iclr_results/119_A Decade's Battle on Dataset Bias_ Are We There Yet_/tables/f2df136a221243d6be02e05a0ba2c8ae02d01e7b8c73437cf0474551167610b5.jpg)

## Context-Parametric Inversion: Why Instruction Finetuning May Not Actually Improve Context Reliance


### Images

![138102ff1a409af893bad10b4eac5e070930e7edc91f235ecec34f375b707b54.jpg](../iclr_results/120_Context-Parametric Inversion_ Why Instruction Finetuning May Not Actually Improve Context Reliance/images/138102ff1a409af893bad10b4eac5e070930e7edc91f235ecec34f375b707b54.jpg)

![1510696bdd1065a8c24854f808451f0dbf009510ea0a911109c7d75f0d3b463c.jpg](../iclr_results/120_Context-Parametric Inversion_ Why Instruction Finetuning May Not Actually Improve Context Reliance/images/1510696bdd1065a8c24854f808451f0dbf009510ea0a911109c7d75f0d3b463c.jpg)

![1583590e02ebd7100bcf6cc09f1d2135a14ae3b96929f710eb4ca29cfc5c8a07.jpg](../iclr_results/120_Context-Parametric Inversion_ Why Instruction Finetuning May Not Actually Improve Context Reliance/images/1583590e02ebd7100bcf6cc09f1d2135a14ae3b96929f710eb4ca29cfc5c8a07.jpg)

![1ca630b3715f54b76c7fbb80f9915ff74b86cfd3d024055da9765c93222cc1e7.jpg](../iclr_results/120_Context-Parametric Inversion_ Why Instruction Finetuning May Not Actually Improve Context Reliance/images/1ca630b3715f54b76c7fbb80f9915ff74b86cfd3d024055da9765c93222cc1e7.jpg)

![39d10889d9d2e9404d1563416a441f9ebdbc3680b2522cddc588652f01ac7843.jpg](../iclr_results/120_Context-Parametric Inversion_ Why Instruction Finetuning May Not Actually Improve Context Reliance/images/39d10889d9d2e9404d1563416a441f9ebdbc3680b2522cddc588652f01ac7843.jpg)

![3ba264f29d1eeb59a60c39b50f8400bd8a8d743681dbfd6e18a55573778d85e6.jpg](../iclr_results/120_Context-Parametric Inversion_ Why Instruction Finetuning May Not Actually Improve Context Reliance/images/3ba264f29d1eeb59a60c39b50f8400bd8a8d743681dbfd6e18a55573778d85e6.jpg)

![50574966d30d2273510783c97f6e70ec3e4a4a500ac23922bf1f7f0a2ff548bf.jpg](../iclr_results/120_Context-Parametric Inversion_ Why Instruction Finetuning May Not Actually Improve Context Reliance/images/50574966d30d2273510783c97f6e70ec3e4a4a500ac23922bf1f7f0a2ff548bf.jpg)

![5486163afc9a1215487a938df5e13a9ee976eb15daae7d4ebaf8106838e45a34.jpg](../iclr_results/120_Context-Parametric Inversion_ Why Instruction Finetuning May Not Actually Improve Context Reliance/images/5486163afc9a1215487a938df5e13a9ee976eb15daae7d4ebaf8106838e45a34.jpg)

![5a6db5592a8a3bd1f8283ede5cfbf60f3d4ccf5e8ac241ec43ae9b2df765fc6c.jpg](../iclr_results/120_Context-Parametric Inversion_ Why Instruction Finetuning May Not Actually Improve Context Reliance/images/5a6db5592a8a3bd1f8283ede5cfbf60f3d4ccf5e8ac241ec43ae9b2df765fc6c.jpg)

![9253b23429b082f85536ac5bc46f3cf265a61418f3f4986b366bd1639e583999.jpg](../iclr_results/120_Context-Parametric Inversion_ Why Instruction Finetuning May Not Actually Improve Context Reliance/images/9253b23429b082f85536ac5bc46f3cf265a61418f3f4986b366bd1639e583999.jpg)

![99965af7b464423e592ca9ac31ad40bb63736aeb97db1c783cc4d24373b758e2.jpg](../iclr_results/120_Context-Parametric Inversion_ Why Instruction Finetuning May Not Actually Improve Context Reliance/images/99965af7b464423e592ca9ac31ad40bb63736aeb97db1c783cc4d24373b758e2.jpg)

![b0b96e796937a135ebf0ec4cae5bb5c00e4f765c022c0fe9cc583de1e3b1c4d3.jpg](../iclr_results/120_Context-Parametric Inversion_ Why Instruction Finetuning May Not Actually Improve Context Reliance/images/b0b96e796937a135ebf0ec4cae5bb5c00e4f765c022c0fe9cc583de1e3b1c4d3.jpg)

![c698bbad16380422c986f7b6ae640bdde4be1363d4856031b29d9f4ca9422207.jpg](../iclr_results/120_Context-Parametric Inversion_ Why Instruction Finetuning May Not Actually Improve Context Reliance/images/c698bbad16380422c986f7b6ae640bdde4be1363d4856031b29d9f4ca9422207.jpg)

![d6b259451d87d2170505cb95b408a1a270e5dae1ac0952ee479ed61c360c24a6.jpg](../iclr_results/120_Context-Parametric Inversion_ Why Instruction Finetuning May Not Actually Improve Context Reliance/images/d6b259451d87d2170505cb95b408a1a270e5dae1ac0952ee479ed61c360c24a6.jpg)

![fc33bb5d1f9f708fb0af1d1ca91959e391467abb1d4f1680b70766c42c1376c8.jpg](../iclr_results/120_Context-Parametric Inversion_ Why Instruction Finetuning May Not Actually Improve Context Reliance/images/fc33bb5d1f9f708fb0af1d1ca91959e391467abb1d4f1680b70766c42c1376c8.jpg)

## Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model


### Images

![01812ba6db964fb99326003d848609c23adaa5a57522d6ca6bb07db131cfd370.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/01812ba6db964fb99326003d848609c23adaa5a57522d6ca6bb07db131cfd370.jpg)

![02e2bbd4906b8bf25026b3e4e85aaff49452bf4faf6f2563e70fb76da77ccd7a.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/02e2bbd4906b8bf25026b3e4e85aaff49452bf4faf6f2563e70fb76da77ccd7a.jpg)

![02e50b6826cf30786b71ae92cff883bc4dbfbcc3e9fb3bbf134ce2d32efbaaa2.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/02e50b6826cf30786b71ae92cff883bc4dbfbcc3e9fb3bbf134ce2d32efbaaa2.jpg)

![0987a78b1624d6a63754a8e7be1dbe28dc8f98b2f6269eaae4cc40bfd1c1a7e4.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/0987a78b1624d6a63754a8e7be1dbe28dc8f98b2f6269eaae4cc40bfd1c1a7e4.jpg)

![09bbbbf77c76d5072281a1708e962659a24502b043621dd714b6b6cfbba5a5cd.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/09bbbbf77c76d5072281a1708e962659a24502b043621dd714b6b6cfbba5a5cd.jpg)

![0c0f1039127dd9ffbc5cd3833a7c66edae732536cce3ccdd217fa38e2554c0cb.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/0c0f1039127dd9ffbc5cd3833a7c66edae732536cce3ccdd217fa38e2554c0cb.jpg)

![0daefe6c38986c50dacb4bb9902488fd234e918ba19e602b575b9d978dad1a2a.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/0daefe6c38986c50dacb4bb9902488fd234e918ba19e602b575b9d978dad1a2a.jpg)

![16bcab982b80172eac796ece10ef72bb4cb34fa84c4b8be33bd7888fbe667a5e.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/16bcab982b80172eac796ece10ef72bb4cb34fa84c4b8be33bd7888fbe667a5e.jpg)

![1ece28cc676eab362a01f8abe9d3d129419d2ffb80b1cd77d272d65051831cba.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/1ece28cc676eab362a01f8abe9d3d129419d2ffb80b1cd77d272d65051831cba.jpg)

![2876b09a1c6b1751fb795114c64e985624e064e28753fda8435c6bf762ab87af.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/2876b09a1c6b1751fb795114c64e985624e064e28753fda8435c6bf762ab87af.jpg)

![2a21db66d19b20c0a1f4265255c7a3f22191c01f38cceb6e7da28761ca57457f.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/2a21db66d19b20c0a1f4265255c7a3f22191c01f38cceb6e7da28761ca57457f.jpg)

![369fe01b0f5755abe42450c4e1f7e97c21ee53b0dae75acd194ca0c11ab7ae85.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/369fe01b0f5755abe42450c4e1f7e97c21ee53b0dae75acd194ca0c11ab7ae85.jpg)

![36a5effa7e3563c2eaa0235480fa49c67d1919fd66bbe0794aff9a2d1f09edb6.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/36a5effa7e3563c2eaa0235480fa49c67d1919fd66bbe0794aff9a2d1f09edb6.jpg)

![37471a6932a5600cbf8f45b012c4b17fed49e5009e9e4e2dfd870eb4e6e4dae1.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/37471a6932a5600cbf8f45b012c4b17fed49e5009e9e4e2dfd870eb4e6e4dae1.jpg)

![39bf4b9b6928e7d2d3de1f8b8e65d314d8ae9353276f7b93ab78c046d0fc15ad.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/39bf4b9b6928e7d2d3de1f8b8e65d314d8ae9353276f7b93ab78c046d0fc15ad.jpg)

![39ea55fca031310441236f8e9d5c2342532564dab0613e9e5879933e7998072b.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/39ea55fca031310441236f8e9d5c2342532564dab0613e9e5879933e7998072b.jpg)

![39f3eb0a4da3c0267218456af526d35e4ea612a586ba6763aecb2d489931df3c.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/39f3eb0a4da3c0267218456af526d35e4ea612a586ba6763aecb2d489931df3c.jpg)

![3f228246cd5185b2b65d5ff2466e6fd1580bc83f62150035ea193712561459d2.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/3f228246cd5185b2b65d5ff2466e6fd1580bc83f62150035ea193712561459d2.jpg)

![41c7ad2ca4c52a06379f93dc527910a65296034cddff46ade8899fa3a4b0582d.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/41c7ad2ca4c52a06379f93dc527910a65296034cddff46ade8899fa3a4b0582d.jpg)

![42c85d8b98eee853188e846ba22f5409f39b0c7ce18cf73cb64990b4f34b43c3.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/42c85d8b98eee853188e846ba22f5409f39b0c7ce18cf73cb64990b4f34b43c3.jpg)

![4ae604ef2d689d5e6be294d582ccfea5d76bef2c207dbeb0dc638f2c57f7b6fc.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/4ae604ef2d689d5e6be294d582ccfea5d76bef2c207dbeb0dc638f2c57f7b6fc.jpg)

![50e13ef6d45617e9e9647d2bebff1b132c5ff6a5e068419efb8eb7a335742831.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/50e13ef6d45617e9e9647d2bebff1b132c5ff6a5e068419efb8eb7a335742831.jpg)

![529f00c0fdcf41a6b9e76bb9a1f3e0c7642d921801d07c4b2244d996eb335d45.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/529f00c0fdcf41a6b9e76bb9a1f3e0c7642d921801d07c4b2244d996eb335d45.jpg)

![54552dfea8f0ef159f1389309d41af5a9d822a6be5cb90da7815a9ddc5ccbeb8.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/54552dfea8f0ef159f1389309d41af5a9d822a6be5cb90da7815a9ddc5ccbeb8.jpg)

![54fd6eafa8b64b21845700e83e9f1dd90e764106eef8e05d4435e376d51142ff.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/54fd6eafa8b64b21845700e83e9f1dd90e764106eef8e05d4435e376d51142ff.jpg)

![55fbf9f651adf5a3ea57bd48877dee2af3970d7cc38b8e69ec1d082348425dd0.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/55fbf9f651adf5a3ea57bd48877dee2af3970d7cc38b8e69ec1d082348425dd0.jpg)

![5e718fa2071331a6b50b00120d7ef209cd9701cfde8901e01b305f8734b61a12.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/5e718fa2071331a6b50b00120d7ef209cd9701cfde8901e01b305f8734b61a12.jpg)

![6216d7a0f87d2a035f4e051204342b08fc6b592c0e5a10b3ef813f1cb632b095.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/6216d7a0f87d2a035f4e051204342b08fc6b592c0e5a10b3ef813f1cb632b095.jpg)

![6263b9acda93858562d5a38cae091e21612974923750da7852e928d4ce7e45e2.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/6263b9acda93858562d5a38cae091e21612974923750da7852e928d4ce7e45e2.jpg)

![6c1d0a0660766e6aa412adc8cac688cc1808c7c8aebd7546e148f461e784c4d7.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/6c1d0a0660766e6aa412adc8cac688cc1808c7c8aebd7546e148f461e784c4d7.jpg)

![6d87469b6e0b97d36a826b0dbd820528a1e18a6f26490740385a0a8b75307892.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/6d87469b6e0b97d36a826b0dbd820528a1e18a6f26490740385a0a8b75307892.jpg)

![6e9b0d05b972483b1355397a9a675c628ac1bbba0ce710312e876e5edccf7d5c.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/6e9b0d05b972483b1355397a9a675c628ac1bbba0ce710312e876e5edccf7d5c.jpg)

![7058d5908aff0eff9fd06f6562f47e12ba6da4877e39abd470987f1d6fa9d1d4.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/7058d5908aff0eff9fd06f6562f47e12ba6da4877e39abd470987f1d6fa9d1d4.jpg)

![70c00b06b7cac67b6c8b997ff57a68b69fc6298ba88620db1e39da0760f15e2b.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/70c00b06b7cac67b6c8b997ff57a68b69fc6298ba88620db1e39da0760f15e2b.jpg)

![71c41428d358231eaac44434e4dadcef170524120dd6bdb4d0ae729c4183fd2a.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/71c41428d358231eaac44434e4dadcef170524120dd6bdb4d0ae729c4183fd2a.jpg)

![74b2b962ff96cd993dfcd967e0d5637ec4a93d3dcc640d1a4277a5bf09f14405.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/74b2b962ff96cd993dfcd967e0d5637ec4a93d3dcc640d1a4277a5bf09f14405.jpg)

![7618fce750303719d7dfbb50a7b78766d6be2f3ea56147dec07dbfa24d54ca6d.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/7618fce750303719d7dfbb50a7b78766d6be2f3ea56147dec07dbfa24d54ca6d.jpg)

![7721cb907728f1720b6f4e35908cec9f140e8e10dba4b8b4fe4014d0a867befe.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/7721cb907728f1720b6f4e35908cec9f140e8e10dba4b8b4fe4014d0a867befe.jpg)

![7de0bdf07a284749db8f19d5bafb7f47edf26c78a85ecdd5dbaad9d90f315bf9.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/7de0bdf07a284749db8f19d5bafb7f47edf26c78a85ecdd5dbaad9d90f315bf9.jpg)

![7e9c8699007f640f19a501e756d75b1c295f2e103f682b708963d113162c4bd1.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/7e9c8699007f640f19a501e756d75b1c295f2e103f682b708963d113162c4bd1.jpg)

![804dbcacaa884e9524b7c33cc1d25bd2613936fbb9909a0b6a99b7cf32b0a0a2.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/804dbcacaa884e9524b7c33cc1d25bd2613936fbb9909a0b6a99b7cf32b0a0a2.jpg)

![83ba87c54378e184b6206f20496982dd27592d127359129b28b0b5517519f623.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/83ba87c54378e184b6206f20496982dd27592d127359129b28b0b5517519f623.jpg)

![8461b3252a4ded7cc7698adf88a5ad334a9d10884edb0b316fb749bcf01f58ca.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/8461b3252a4ded7cc7698adf88a5ad334a9d10884edb0b316fb749bcf01f58ca.jpg)

![8502e09800a6d272fbb6dd471c52a89f4dfdc5bb07ccc09ecef509c24374393b.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/8502e09800a6d272fbb6dd471c52a89f4dfdc5bb07ccc09ecef509c24374393b.jpg)

![8543a8a3ccabb1d4afc52c4573049ee778126b0153c711613f2d055c75a5e291.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/8543a8a3ccabb1d4afc52c4573049ee778126b0153c711613f2d055c75a5e291.jpg)

![924f4d0bb2cde66f63ed79ce33379907b736eca63717b17f6a20a8939f11f993.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/924f4d0bb2cde66f63ed79ce33379907b736eca63717b17f6a20a8939f11f993.jpg)

![a32e84876215869a79f6dda4e37d7eea0067e04ed6677a609197061cc81a99a0.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/a32e84876215869a79f6dda4e37d7eea0067e04ed6677a609197061cc81a99a0.jpg)

![a43f08c044927a70e0c33d13bf6014923fec6ba3e39f79ff56ef79f2c642b841.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/a43f08c044927a70e0c33d13bf6014923fec6ba3e39f79ff56ef79f2c642b841.jpg)

![bce26fe19283d601714b6a54951babec7eeed35cdc05ca0c7aa784561e1ee86a.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/bce26fe19283d601714b6a54951babec7eeed35cdc05ca0c7aa784561e1ee86a.jpg)

![c2967a5ea6927aa8a93aa8bdc911545784e3e0069f302456aab5f7bf2c536703.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/c2967a5ea6927aa8a93aa8bdc911545784e3e0069f302456aab5f7bf2c536703.jpg)

![c6e7a3d1d7cf47d610db13b2deb07424c770157c704f7889f678442d140d643d.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/c6e7a3d1d7cf47d610db13b2deb07424c770157c704f7889f678442d140d643d.jpg)

![c80f592981e26d73af03d832876fade9f674ff20b4e03a3d27517d33648b40d2.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/c80f592981e26d73af03d832876fade9f674ff20b4e03a3d27517d33648b40d2.jpg)

![cef8c2d36f831e07edc958fdb14187e1323f4de0380f0b3d6204aa07b978cf9c.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/cef8c2d36f831e07edc958fdb14187e1323f4de0380f0b3d6204aa07b978cf9c.jpg)

![cf9f09d38109c92d6a768d00d181532483c09265e6383169e9e92e431bf75882.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/cf9f09d38109c92d6a768d00d181532483c09265e6383169e9e92e431bf75882.jpg)

![da24a6c1484b0ab6f8b96e2de995794e797ef110ca5f59f716635d8fc192a0d5.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/da24a6c1484b0ab6f8b96e2de995794e797ef110ca5f59f716635d8fc192a0d5.jpg)

![dbd208c3ade1b2f0ea47fa4b6835ab5933c7380ee460630870aec9dfc0e5fdcd.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/dbd208c3ade1b2f0ea47fa4b6835ab5933c7380ee460630870aec9dfc0e5fdcd.jpg)

![de03067f3a9e715017ed0100b283b9d088b605bc6246ae0f5bcc869f74700e92.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/de03067f3a9e715017ed0100b283b9d088b605bc6246ae0f5bcc869f74700e92.jpg)

![de22ac4fc910fe2c6e1c2ae49ba2f05423c182ff9228632dadb0db0a1e7e8900.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/de22ac4fc910fe2c6e1c2ae49ba2f05423c182ff9228632dadb0db0a1e7e8900.jpg)

![e96d7f62f08d3d480e1ed49e6ab2ff03bbaa5bfff4db6bfd5a519d374fd39375.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/e96d7f62f08d3d480e1ed49e6ab2ff03bbaa5bfff4db6bfd5a519d374fd39375.jpg)

![f231ad00f49865b86a0d1a611e659c4e305a566f7c0db0725ffa45d531f58d56.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/f231ad00f49865b86a0d1a611e659c4e305a566f7c0db0725ffa45d531f58d56.jpg)

![fce9632faf1c87b58ffdb77052de9b212cdd708db2c5ac5d23a6e6f748d413e7.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/fce9632faf1c87b58ffdb77052de9b212cdd708db2c5ac5d23a6e6f748d413e7.jpg)

![fdb8f9bf0ecd558e89099e191b0a0709dc83bbfc6268f0a8878ed5da33aecb83.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/images/fdb8f9bf0ecd558e89099e191b0a0709dc83bbfc6268f0a8878ed5da33aecb83.jpg)

### Tables

![23a4309a3667a594808c325598a71d06d92110793bd344ad55958eaf14d0c555.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/tables/23a4309a3667a594808c325598a71d06d92110793bd344ad55958eaf14d0c555.jpg)

![3117c7f3e6848dd7a8ae9e96ea480bf9ce236c725d3d7b12d771b7e704a59d5a.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/tables/3117c7f3e6848dd7a8ae9e96ea480bf9ce236c725d3d7b12d771b7e704a59d5a.jpg)

![99687135a808d9eaa94cb9527a6d193a86749b8a626028c012d4fb334cf664c8.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/tables/99687135a808d9eaa94cb9527a6d193a86749b8a626028c012d4fb334cf664c8.jpg)

![ab9509ea0899a08a3e4caa28c7938d13b784c0d6d7e46a982800f2daa5660d49.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/tables/ab9509ea0899a08a3e4caa28c7938d13b784c0d6d7e46a982800f2daa5660d49.jpg)

![b881c3b8b41fbfb6243efd4e3f23aa08677a17990e23de5ba5cb2e6af101b901.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/tables/b881c3b8b41fbfb6243efd4e3f23aa08677a17990e23de5ba5cb2e6af101b901.jpg)

![bce550ae3cfecf7e73d00373e92449349f00a7f25ab447ad5ee8142fc39fee24.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/tables/bce550ae3cfecf7e73d00373e92449349f00a7f25ab447ad5ee8142fc39fee24.jpg)

![cbecf1040dcc6b59d6a1b8fc6df4fd48784a2e067eb3c8f9b9f80eb6aede8b56.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/tables/cbecf1040dcc6b59d6a1b8fc6df4fd48784a2e067eb3c8f9b9f80eb6aede8b56.jpg)

![db357f28a336afd1900566e35f90b8af7401f6ad30be96149119f88fcdff9c64.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/tables/db357f28a336afd1900566e35f90b8af7401f6ad30be96149119f88fcdff9c64.jpg)

![de669f8874a96a1d23b49ccbe4621e27f3e65c4c6c3bb719ec9e4963ecd17b50.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/tables/de669f8874a96a1d23b49ccbe4621e27f3e65c4c6c3bb719ec9e4963ecd17b50.jpg)

![f3c8cecdf03f256505d12007cbb06f2ae296da1364c510a5aeecc7771abf02d9.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/tables/f3c8cecdf03f256505d12007cbb06f2ae296da1364c510a5aeecc7771abf02d9.jpg)

![f562042acb531ba2d8efd123e72231ed13fef0d30dc009f13bbf704017525592.jpg](../iclr_results/121_Transfusion_ Predict the Next Token and Diffuse Images with One Multi-Modal Model/tables/f562042acb531ba2d8efd123e72231ed13fef0d30dc009f13bbf704017525592.jpg)

## ECD: A Machine Learning Benchmark for Predicting Enhanced-Precision Electronic Charge Density in Crystalline Inorganic Materials

### Images

![0fd207f467a184d078d1ab54cd35b0cddac8557583d8308f37fdc41f6e6ff4dd.jpg](../iclr_results/122_ECD_ A Machine Learning Benchmark for Predicting Enhanced-Precision Electronic Charge Density in Cry/images/0fd207f467a184d078d1ab54cd35b0cddac8557583d8308f37fdc41f6e6ff4dd.jpg)

![32421d2425fd03f19ffdb6f5a00a95eac6debd0c14081e729d5d86cd265f1e22.jpg](../iclr_results/122_ECD_ A Machine Learning Benchmark for Predicting Enhanced-Precision Electronic Charge Density in Cry/images/32421d2425fd03f19ffdb6f5a00a95eac6debd0c14081e729d5d86cd265f1e22.jpg)

![58fe1595f1e5dace035baddb5ff373c4224b0c1c646ea3f4dfa097dd9bce22db.jpg](../iclr_results/122_ECD_ A Machine Learning Benchmark for Predicting Enhanced-Precision Electronic Charge Density in Cry/images/58fe1595f1e5dace035baddb5ff373c4224b0c1c646ea3f4dfa097dd9bce22db.jpg)

![6fdd52400cd0edb4a7301bbcac8dd76aa8cfd516edf675868df084ee953e6270.jpg](../iclr_results/122_ECD_ A Machine Learning Benchmark for Predicting Enhanced-Precision Electronic Charge Density in Cry/images/6fdd52400cd0edb4a7301bbcac8dd76aa8cfd516edf675868df084ee953e6270.jpg)

![87ef6645c5c61c96f881e5c982ebf2466da84267845489aa31760dbd0b58830d.jpg](../iclr_results/122_ECD_ A Machine Learning Benchmark for Predicting Enhanced-Precision Electronic Charge Density in Cry/images/87ef6645c5c61c96f881e5c982ebf2466da84267845489aa31760dbd0b58830d.jpg)

![939c1d97f94bbf2824ecdfcb2367069542010d35f39118839bac7fa704e72418.jpg](../iclr_results/122_ECD_ A Machine Learning Benchmark for Predicting Enhanced-Precision Electronic Charge Density in Cry/images/939c1d97f94bbf2824ecdfcb2367069542010d35f39118839bac7fa704e72418.jpg)

![eb365d2fd2b16c4cfb6ccdc9a1296767443ebe0a849ff4596677a2aadc016d42.jpg](../iclr_results/122_ECD_ A Machine Learning Benchmark for Predicting Enhanced-Precision Electronic Charge Density in Cry/images/eb365d2fd2b16c4cfb6ccdc9a1296767443ebe0a849ff4596677a2aadc016d42.jpg)

![ff6bdc54684246dfdd28395ede276a76279c580f7deeb5be247fa4dbe069d9ce.jpg](../iclr_results/122_ECD_ A Machine Learning Benchmark for Predicting Enhanced-Precision Electronic Charge Density in Cry/images/ff6bdc54684246dfdd28395ede276a76279c580f7deeb5be247fa4dbe069d9ce.jpg)

### Tables

![223f8c42800e2c2051c3b1eaee6a077eb5410d73575705d2a2ed537fb33abd26.jpg](../iclr_results/122_ECD_ A Machine Learning Benchmark for Predicting Enhanced-Precision Electronic Charge Density in Cry/tables/223f8c42800e2c2051c3b1eaee6a077eb5410d73575705d2a2ed537fb33abd26.jpg)

![a9c76ba59adad1970d21daa043f8bdaaa9dcadbade010db469f9d2f3bb813cbe.jpg](../iclr_results/122_ECD_ A Machine Learning Benchmark for Predicting Enhanced-Precision Electronic Charge Density in Cry/tables/a9c76ba59adad1970d21daa043f8bdaaa9dcadbade010db469f9d2f3bb813cbe.jpg)

![b011b4447e39ae855edcf83f8fc3fab8919a50290315cc3c1eb4dd2fb1f7df51.jpg](../iclr_results/122_ECD_ A Machine Learning Benchmark for Predicting Enhanced-Precision Electronic Charge Density in Cry/tables/b011b4447e39ae855edcf83f8fc3fab8919a50290315cc3c1eb4dd2fb1f7df51.jpg)

![ba8e4ce857ee2288167014610c57af20b0f22c114703a8447a170a3013b1e41b.jpg](../iclr_results/122_ECD_ A Machine Learning Benchmark for Predicting Enhanced-Precision Electronic Charge Density in Cry/tables/ba8e4ce857ee2288167014610c57af20b0f22c114703a8447a170a3013b1e41b.jpg)

![d055ebf9b67e9a838c6a0ce2a07b4c4cf157ef0dc17d4ba4d3d8dead00ed277c.jpg](../iclr_results/122_ECD_ A Machine Learning Benchmark for Predicting Enhanced-Precision Electronic Charge Density in Cry/tables/d055ebf9b67e9a838c6a0ce2a07b4c4cf157ef0dc17d4ba4d3d8dead00ed277c.jpg)
