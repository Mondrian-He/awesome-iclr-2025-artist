# ICLR 2025 Main Conference Papers

**Summary:** 36 papers with extracted content:
- 📊 Total images: 44031
- 📋 Total tables: 33468
- 📄 Total files: 77499

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 18 of 100

## 目录 (Table of Contents)

1. [Federated Class-Incremental Learning: A Hybrid Approach Using Latent Exemplars and Data-Free Techniques to Address Local and Global Forgetting](#Federated-Class-Incremental-Learning-A-Hybrid-Approach-Using-Latent-Exemplars-and-Data-Free-Techniques-to-Address-Local-and-Global-Forgetting)
2. [AIMS.au: A Dataset for the Analysis of Modern Slavery Countermeasures in Corporate Statements](#AIMSau-A-Dataset-for-the-Analysis-of-Modern-Slavery-Countermeasures-in-Corporate-Statements)
3. [ThermalGaussian: Thermal 3D Gaussian Splatting](#ThermalGaussian-Thermal-3D-Gaussian-Splatting)
4. [Scaling Wearable Foundation Models](#Scaling-Wearable-Foundation-Models)
5. [Omni-MATH: A Universal Olympiad Level Mathematic Benchmark for Large Language Models](#Omni-MATH-A-Universal-Olympiad-Level-Mathematic-Benchmark-for-Large-Language-Models)
6. [Language-Image Models with 3D Understanding](#Language-Image-Models-with-3D-Understanding)
7. [NoVo: Norm Voting off Hallucinations with Attention Heads in Large Language Models](#NoVo-Norm-Voting-off-Hallucinations-with-Attention-Heads-in-Large-Language-Models)
8. [Generative Monoculture in Large Language Models](#Generative-Monoculture-in-Large-Language-Models)
9. [Accelerating Diffusion Transformers with Token-wise Feature Caching](#Accelerating-Diffusion-Transformers-with-Token-wise-Feature-Caching)
10. [Point-SAM: Promptable 3D Segmentation Model for Point Clouds](#Point-SAM-Promptable-3D-Segmentation-Model-for-Point-Clouds)
11. [Towards Understanding the Universality of Transformers for Next-Token Prediction](#Towards-Understanding-the-Universality-of-Transformers-for-Next-Token-Prediction)
12. [Disentangling Representations through Multi-task Learning](#Disentangling-Representations-through-Multi-task-Learning)
13. [APE: Faster and Longer Context-Augmented Generation via Adaptive Parallel Encoding](#APE-Faster-and-Longer-Context-Augmented-Generation-via-Adaptive-Parallel-Encoding)
14. [Robots Pre-train Robots: Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets](#Robots-Pre-train-Robots-Manipulation-Centric-Robotic-Representation-from-Large-Scale-Robot-Datasets)
15. [Causally Motivated Sycophancy Mitigation for Large Language Models](#Causally-Motivated-Sycophancy-Mitigation-for-Large-Language-Models)
16. [Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron](#Understanding-and-Enhancing-Safety-Mechanisms-of-LLMs-via-Safety-Specific-Neuron)
17. [Pacmann: Efficient Private Approximate Nearest Neighbor Search](#Pacmann-Efficient-Private-Approximate-Nearest-Neighbor-Search)
18. [Understanding the Generalization of In-Context Learning in Transformers: An Empirical Study](#Understanding-the-Generalization-of-In-Context-Learning-in-Transformers-An-Empirical-Study)
19. [ReCogLab: a framework testing relational reasoning & cognitive hypotheses on LLMs](#ReCogLab-a-framework-testing-relational-reasoning-cognitive-hypotheses-on-LLMs)
20. [MeteoRA: Multiple-tasks Embedded LoRA for Large Language Models](#MeteoRA-Multiple-tasks-Embedded-LoRA-for-Large-Language-Models)
21. [HERO: Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning](#HERO-Human-Feedback-Efficient-Reinforcement-Learning-for-Online-Diffusion-Model-Finetuning)
22. [On the Price of Differential Privacy for Hierarchical Clustering](#On-the-Price-of-Differential-Privacy-for-Hierarchical-Clustering)
23. [Brain Mapping with Dense Features: Grounding Cortical Semantic Selectivity in Natural Images With Vision Transformers](#Brain-Mapping-with-Dense-Features-Grounding-Cortical-Semantic-Selectivity-in-Natural-Images-With-Vision-Transformers)
24. [A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Translations](#A-Unified-Framework-for-Forward-and-Inverse-Problems-in-Subsurface-Imaging-using-Latent-Space-Translations)
25. [Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding](#Contextual-Self-paced-Learning-for-Weakly-Supervised-Spatio-Temporal-Video-Grounding)
26. [Image and Video Tokenization with Binary Spherical Quantization](#Image-and-Video-Tokenization-with-Binary-Spherical-Quantization)
27. [Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment](#Mitigating-Object-Hallucination-in-MLLMs-via-Data-augmented-Phrase-level-Alignment)
28. [Simple, Good, Fast: Self-Supervised World Models Free of Baggage](#Simple-Good-Fast-Self-Supervised-World-Models-Free-of-Baggage)
29. [MDSGen: Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Generation](#MDSGen-Fast-and-Efficient-Masked-Diffusion-Temporal-Aware-Transformers-for-Open-Domain-Sound-Generation)
30. [UniCO: On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP](#UniCO-On-Unified-Combinatorial-Optimization-via-Problem-Reduction-to-Matrix-Encoded-General-TSP)
31. [Grammar Reinforcement Learning: path and cycle counting in graphs with a Context-Free Grammar and Transformer approach](#Grammar-Reinforcement-Learning-path-and-cycle-counting-in-graphs-with-a-Context-Free-Grammar-and-Transformer-approach)
32. [GPromptShield: Elevating Resilience in Graph Prompt Tuning Against Adversarial Attacks](#GPromptShield-Elevating-Resilience-in-Graph-Prompt-Tuning-Against-Adversarial-Attacks)
33. [WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling](#WavTokenizer-an-Efficient-Acoustic-Discrete-Codec-Tokenizer-for-Audio-Language-Modeling)
34. [RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation](#RDT-1B-a-Diffusion-Foundation-Model-for-Bimanual-Manipulation)
35. [Quality over Quantity in Attention Layers: When Adding More Heads Hurts](#Quality-over-Quantity-in-Attention-Layers-When-Adding-More-Heads-Hurts)
36. [Language Agents Meet Causality -- Bridging LLMs and Causal World Models](#Language-Agents-Meet-Causality-Bridging-LLMs-and-Causal-World-Models)

---


## Federated Class-Incremental Learning: A Hybrid Approach Using Latent Exemplars and Data-Free Techniques to Address Local and Global Forgetting

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

## ReCogLab: a framework testing relational reasoning & cognitive hypotheses on LLMs


### Images

![1ec775e91d75564292d90a5c10bf712fef6db78f03e1b89f0521cc701b2dc77d.jpg](../iclr_results/674_ReCogLab_ a framework testing relational reasoning & cognitive hypotheses on LLMs/images/1ec775e91d75564292d90a5c10bf712fef6db78f03e1b89f0521cc701b2dc77d.jpg)

![2d83642009866d4c4e065e0307fafb7886f8a8a137876741f099348e04cf7670.jpg](../iclr_results/674_ReCogLab_ a framework testing relational reasoning & cognitive hypotheses on LLMs/images/2d83642009866d4c4e065e0307fafb7886f8a8a137876741f099348e04cf7670.jpg)

![326fe85ecb9aa54c559a69d14660faeebba5e19ffda001e7441be220256367c9.jpg](../iclr_results/674_ReCogLab_ a framework testing relational reasoning & cognitive hypotheses on LLMs/images/326fe85ecb9aa54c559a69d14660faeebba5e19ffda001e7441be220256367c9.jpg)

![56e1ea87d463b41b0a05a89824ebf8fd3601a1d564cbbb1d4e9b8b3c42c96332.jpg](../iclr_results/674_ReCogLab_ a framework testing relational reasoning & cognitive hypotheses on LLMs/images/56e1ea87d463b41b0a05a89824ebf8fd3601a1d564cbbb1d4e9b8b3c42c96332.jpg)

![95c21c7cc52ccfaabe9f6084cddfcc640ddc766c563b9d16302f6f3b7ee921df.jpg](../iclr_results/674_ReCogLab_ a framework testing relational reasoning & cognitive hypotheses on LLMs/images/95c21c7cc52ccfaabe9f6084cddfcc640ddc766c563b9d16302f6f3b7ee921df.jpg)

![9d8901a28bc973267236cad5e17e78fac35fd1b11b449b5f30175c7f86696c0a.jpg](../iclr_results/674_ReCogLab_ a framework testing relational reasoning & cognitive hypotheses on LLMs/images/9d8901a28bc973267236cad5e17e78fac35fd1b11b449b5f30175c7f86696c0a.jpg)

![b0a05cf13d82a1fcde9aa959152f173ae0abd1172bfe27de5c43cb5e863b6fa5.jpg](../iclr_results/674_ReCogLab_ a framework testing relational reasoning & cognitive hypotheses on LLMs/images/b0a05cf13d82a1fcde9aa959152f173ae0abd1172bfe27de5c43cb5e863b6fa5.jpg)

![d26ec0235ed7c1e50fb6a00dd98fdac9963f6ab0c73253e7d648ed98ce59f6c7.jpg](../iclr_results/674_ReCogLab_ a framework testing relational reasoning & cognitive hypotheses on LLMs/images/d26ec0235ed7c1e50fb6a00dd98fdac9963f6ab0c73253e7d648ed98ce59f6c7.jpg)

![e6c140821a656ce272e9b8d149cbe9a79648b73a8b8aa2e5df40ae351ae403c3.jpg](../iclr_results/674_ReCogLab_ a framework testing relational reasoning & cognitive hypotheses on LLMs/images/e6c140821a656ce272e9b8d149cbe9a79648b73a8b8aa2e5df40ae351ae403c3.jpg)

![e7e659d43bb153bcd35c7fae1142ce523760669cb0a7084a4f58a257f320215f.jpg](../iclr_results/674_ReCogLab_ a framework testing relational reasoning & cognitive hypotheses on LLMs/images/e7e659d43bb153bcd35c7fae1142ce523760669cb0a7084a4f58a257f320215f.jpg)

![eb47529ab7fe4471bc2caa606b5e39272f4887f70e44b56ea01f48683617c918.jpg](../iclr_results/674_ReCogLab_ a framework testing relational reasoning & cognitive hypotheses on LLMs/images/eb47529ab7fe4471bc2caa606b5e39272f4887f70e44b56ea01f48683617c918.jpg)

![f384af7efcdbe081fcc554ae9eb25f45b94061cd794b0c34e0e56d306acba32c.jpg](../iclr_results/674_ReCogLab_ a framework testing relational reasoning & cognitive hypotheses on LLMs/images/f384af7efcdbe081fcc554ae9eb25f45b94061cd794b0c34e0e56d306acba32c.jpg)

### Tables

![ebc4ff48f34ba82b6a2c6dc39e5f6ca0f7598535b7187aa58bda59103837607d.jpg](../iclr_results/674_ReCogLab_ a framework testing relational reasoning & cognitive hypotheses on LLMs/tables/ebc4ff48f34ba82b6a2c6dc39e5f6ca0f7598535b7187aa58bda59103837607d.jpg)

## MeteoRA: Multiple-tasks Embedded LoRA for Large Language Models


### Images

![106b0b3e3e7f1350e29061fe85284a8e92f35e94f0e36e1cdc7908257587232e.jpg](../iclr_results/675_MeteoRA_ Multiple-tasks Embedded LoRA for Large Language Models/images/106b0b3e3e7f1350e29061fe85284a8e92f35e94f0e36e1cdc7908257587232e.jpg)

![138a7dbac1ea2e4cca1ec7066028950a91aac4ba76d99ec0325599c368f20162.jpg](../iclr_results/675_MeteoRA_ Multiple-tasks Embedded LoRA for Large Language Models/images/138a7dbac1ea2e4cca1ec7066028950a91aac4ba76d99ec0325599c368f20162.jpg)

![1f33dc579bb525857ace545ee2d87736c8edc82041f9a1ef4190bb903ae32594.jpg](../iclr_results/675_MeteoRA_ Multiple-tasks Embedded LoRA for Large Language Models/images/1f33dc579bb525857ace545ee2d87736c8edc82041f9a1ef4190bb903ae32594.jpg)

![27a4590edbc45824032d6a064f9508fddd7b3cf6f1ac91f62bca64dc49b36841.jpg](../iclr_results/675_MeteoRA_ Multiple-tasks Embedded LoRA for Large Language Models/images/27a4590edbc45824032d6a064f9508fddd7b3cf6f1ac91f62bca64dc49b36841.jpg)

![58ed3d48d73e5ec81e5d31d523589f38b7f2b24bfaeded79c9c769b5d7273a69.jpg](../iclr_results/675_MeteoRA_ Multiple-tasks Embedded LoRA for Large Language Models/images/58ed3d48d73e5ec81e5d31d523589f38b7f2b24bfaeded79c9c769b5d7273a69.jpg)

![86993bf9db1be19cf06c2f2d1e47a46e859eb63014b771d49400149d3f35f2d6.jpg](../iclr_results/675_MeteoRA_ Multiple-tasks Embedded LoRA for Large Language Models/images/86993bf9db1be19cf06c2f2d1e47a46e859eb63014b771d49400149d3f35f2d6.jpg)

![a19af79490b5f323e264107231ae8b858d0ba4a6ac4d6bbc7c7d73d9063490af.jpg](../iclr_results/675_MeteoRA_ Multiple-tasks Embedded LoRA for Large Language Models/images/a19af79490b5f323e264107231ae8b858d0ba4a6ac4d6bbc7c7d73d9063490af.jpg)

![efbd6b8bc9ea6733471cf73fcd386b0380138a4d413c4c9a3175917837e7d7f4.jpg](../iclr_results/675_MeteoRA_ Multiple-tasks Embedded LoRA for Large Language Models/images/efbd6b8bc9ea6733471cf73fcd386b0380138a4d413c4c9a3175917837e7d7f4.jpg)

### Tables

![1f0eb90ecd88fd2e326dc0b522f678bb73b88a4517112d2f6006e22d37b3ea6a.jpg](../iclr_results/675_MeteoRA_ Multiple-tasks Embedded LoRA for Large Language Models/tables/1f0eb90ecd88fd2e326dc0b522f678bb73b88a4517112d2f6006e22d37b3ea6a.jpg)

![1fd1d29f417243d8e8723bf5ed9169e2581b909c6c487119a1a1008a4e90fa13.jpg](../iclr_results/675_MeteoRA_ Multiple-tasks Embedded LoRA for Large Language Models/tables/1fd1d29f417243d8e8723bf5ed9169e2581b909c6c487119a1a1008a4e90fa13.jpg)

![2115630c3ae2425c767e5361587c5b22e418d7eecf5d19d6a22c759843d6eef7.jpg](../iclr_results/675_MeteoRA_ Multiple-tasks Embedded LoRA for Large Language Models/tables/2115630c3ae2425c767e5361587c5b22e418d7eecf5d19d6a22c759843d6eef7.jpg)

![2d926beb9b9f825688c27805f45641eff97d2e46b995bb99e801ad9282e114f0.jpg](../iclr_results/675_MeteoRA_ Multiple-tasks Embedded LoRA for Large Language Models/tables/2d926beb9b9f825688c27805f45641eff97d2e46b995bb99e801ad9282e114f0.jpg)

![3242821728e95c73d8eecf60a6eb2f4d3ac6fec51547c2994bec28b28fc701c9.jpg](../iclr_results/675_MeteoRA_ Multiple-tasks Embedded LoRA for Large Language Models/tables/3242821728e95c73d8eecf60a6eb2f4d3ac6fec51547c2994bec28b28fc701c9.jpg)

![3a863e18d40bd3ae6a04bec88b51dd22217dd6fc8851929cda285b1bdfafd594.jpg](../iclr_results/675_MeteoRA_ Multiple-tasks Embedded LoRA for Large Language Models/tables/3a863e18d40bd3ae6a04bec88b51dd22217dd6fc8851929cda285b1bdfafd594.jpg)

![4e7df677922776bd10beef5614e6a5a4198937a551a6e4b5722ea23443ad67f6.jpg](../iclr_results/675_MeteoRA_ Multiple-tasks Embedded LoRA for Large Language Models/tables/4e7df677922776bd10beef5614e6a5a4198937a551a6e4b5722ea23443ad67f6.jpg)

![64332fbe86edb0b84f817ec6af7a171be1b13a90f4ccf8f62eac5017d90bed51.jpg](../iclr_results/675_MeteoRA_ Multiple-tasks Embedded LoRA for Large Language Models/tables/64332fbe86edb0b84f817ec6af7a171be1b13a90f4ccf8f62eac5017d90bed51.jpg)

![678caecb6e7cd23c95703cf91f43ee891ef85c5ed47a2d94430f984450aaf7d7.jpg](../iclr_results/675_MeteoRA_ Multiple-tasks Embedded LoRA for Large Language Models/tables/678caecb6e7cd23c95703cf91f43ee891ef85c5ed47a2d94430f984450aaf7d7.jpg)

![ba8686ea68a0cacdefbb34f6aca3ee12b5362d714421a082003603ba0bb164b9.jpg](../iclr_results/675_MeteoRA_ Multiple-tasks Embedded LoRA for Large Language Models/tables/ba8686ea68a0cacdefbb34f6aca3ee12b5362d714421a082003603ba0bb164b9.jpg)

![c3119729525ce5ab917f4f561b5b404cea36df2c04b3bdbf4c65270fc25aa358.jpg](../iclr_results/675_MeteoRA_ Multiple-tasks Embedded LoRA for Large Language Models/tables/c3119729525ce5ab917f4f561b5b404cea36df2c04b3bdbf4c65270fc25aa358.jpg)

## HERO: Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning


### Images

![115354e3a3f1763dc53cd82c2b71c763696cd44073b4007bc18ed7ddcea40443.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/images/115354e3a3f1763dc53cd82c2b71c763696cd44073b4007bc18ed7ddcea40443.jpg)

![28681d81e4d31f5c078fe79af8907e64238f6d80f4324aeeac682cd9429cbf10.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/images/28681d81e4d31f5c078fe79af8907e64238f6d80f4324aeeac682cd9429cbf10.jpg)

![29e767cec9d0b2af5d1a39f4fd77c4acd50fbe1875b5de1e8b496b906c9d3ca8.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/images/29e767cec9d0b2af5d1a39f4fd77c4acd50fbe1875b5de1e8b496b906c9d3ca8.jpg)

![2acd47247c84521ca6fb1193a86b58269a27411d68d780ec78f76ab391a03985.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/images/2acd47247c84521ca6fb1193a86b58269a27411d68d780ec78f76ab391a03985.jpg)

![2ef63a05d1f5dc93686e594c62bee56b346cc66a07ff9634a2860f7898fefc0a.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/images/2ef63a05d1f5dc93686e594c62bee56b346cc66a07ff9634a2860f7898fefc0a.jpg)

![3aa0716b497fb344209633d8cbbb73ff5280923cfc1d8adbdcc235188553469a.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/images/3aa0716b497fb344209633d8cbbb73ff5280923cfc1d8adbdcc235188553469a.jpg)

![4faa4148a28a83508303a9de94ab67ce3576f722af205187b8de5d4713dbeb58.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/images/4faa4148a28a83508303a9de94ab67ce3576f722af205187b8de5d4713dbeb58.jpg)

![603f675590b328807a0b9dfcefe0680536b17c1c4aacbfe755bc4162e9ccd998.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/images/603f675590b328807a0b9dfcefe0680536b17c1c4aacbfe755bc4162e9ccd998.jpg)

![8eb21b840179af49938e3d7a6b0482666b3aa6210747a9942a183aaa2a90d14c.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/images/8eb21b840179af49938e3d7a6b0482666b3aa6210747a9942a183aaa2a90d14c.jpg)

![ad4c04e8bab270b230640d3cb59fe072080c19654945b1094b26ea04a570facb.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/images/ad4c04e8bab270b230640d3cb59fe072080c19654945b1094b26ea04a570facb.jpg)

![b777a9b21b1951850f1476ff6c1191ee5ee883d2d978b574beb2440db9a7dfb6.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/images/b777a9b21b1951850f1476ff6c1191ee5ee883d2d978b574beb2440db9a7dfb6.jpg)

![cf1aa3b43f0fc733d6b96ede6c6586440fdc0b7275a0abb62495ab205feef225.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/images/cf1aa3b43f0fc733d6b96ede6c6586440fdc0b7275a0abb62495ab205feef225.jpg)

![e913c8caf4e1569a1c5c051a4389b958360d615fd8f8d35632bdde4c8efe2aac.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/images/e913c8caf4e1569a1c5c051a4389b958360d615fd8f8d35632bdde4c8efe2aac.jpg)

![ee3a751aefa6057809dc06e9a4858e40fa0c91d3835e3e688c677d0ecef604e5.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/images/ee3a751aefa6057809dc06e9a4858e40fa0c91d3835e3e688c677d0ecef604e5.jpg)

![f7ad7f0044303ec88d1ad70ffab7ef32a2c12d3da92f9e76adbeecc1232b5ecb.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/images/f7ad7f0044303ec88d1ad70ffab7ef32a2c12d3da92f9e76adbeecc1232b5ecb.jpg)

### Tables

![00f62a25d76cc34f41d2ba73f4d96570c9c9849aeffd8b9c253354adc2f7e04e.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/tables/00f62a25d76cc34f41d2ba73f4d96570c9c9849aeffd8b9c253354adc2f7e04e.jpg)

![4ac0d204bba6ec8f062e0643144581fd60c3b97f21c454a8ac4b37459d204809.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/tables/4ac0d204bba6ec8f062e0643144581fd60c3b97f21c454a8ac4b37459d204809.jpg)

![7109ef82ef0f0491fcdceae122a56370fbc6319369848db98117d642f5793dac.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/tables/7109ef82ef0f0491fcdceae122a56370fbc6319369848db98117d642f5793dac.jpg)

![7400dfde917d04a5993dcfc67f731f17649be46c780a58037fb3ae5083169812.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/tables/7400dfde917d04a5993dcfc67f731f17649be46c780a58037fb3ae5083169812.jpg)

![75a1e491da9e01e9837fd6bb763258a476f6d58d03c8c9ec90e05f4a91850386.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/tables/75a1e491da9e01e9837fd6bb763258a476f6d58d03c8c9ec90e05f4a91850386.jpg)

![7d31c82b4684aee3ea8efed59e8ef5d4fd5269ee85196ee2c41a84fa2d60be58.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/tables/7d31c82b4684aee3ea8efed59e8ef5d4fd5269ee85196ee2c41a84fa2d60be58.jpg)

![a71afb55e8555f5807ca7d87d2ba891f6e298a2f19a0dc56619a18948688629f.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/tables/a71afb55e8555f5807ca7d87d2ba891f6e298a2f19a0dc56619a18948688629f.jpg)

![b4ae09c20d93f6495096ba5a71450907de2557e0443c852d3e7d189a74803aa5.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/tables/b4ae09c20d93f6495096ba5a71450907de2557e0443c852d3e7d189a74803aa5.jpg)

![c48f5aac581924288e601262c5f321a467ec3ec2e95b0c3a692ed1d36f96f5e6.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/tables/c48f5aac581924288e601262c5f321a467ec3ec2e95b0c3a692ed1d36f96f5e6.jpg)

![fe4fca2bb3199eb398b9c9f5a0750bfae6b29cc69ae44d0977b6bca0b3bd452f.jpg](../iclr_results/676_HERO_ Human-Feedback Efficient Reinforcement Learning for Online Diffusion Model Finetuning/tables/fe4fca2bb3199eb398b9c9f5a0750bfae6b29cc69ae44d0977b6bca0b3bd452f.jpg)

## On the Price of Differential Privacy for Hierarchical Clustering


### Images

![12a9b74b5dcf566f112dda1620324a1a9cb1052f1cbb9c6c303225dd8fa31bb7.jpg](../iclr_results/677_On the Price of Differential Privacy for Hierarchical Clustering/images/12a9b74b5dcf566f112dda1620324a1a9cb1052f1cbb9c6c303225dd8fa31bb7.jpg)

![2cea993f5017c56d4b99337edbbb3c7187faaeee14578820eca00f0439d27061.jpg](../iclr_results/677_On the Price of Differential Privacy for Hierarchical Clustering/images/2cea993f5017c56d4b99337edbbb3c7187faaeee14578820eca00f0439d27061.jpg)

![4f4c400415b2092d5ae0ab65fa4ada79486f257f4aa2115fbcc918abf34cb613.jpg](../iclr_results/677_On the Price of Differential Privacy for Hierarchical Clustering/images/4f4c400415b2092d5ae0ab65fa4ada79486f257f4aa2115fbcc918abf34cb613.jpg)

![52c1ff91a8e9ccd778aa8c7d8a4ea8738234242f147a4a631be7bfa1906f2f7f.jpg](../iclr_results/677_On the Price of Differential Privacy for Hierarchical Clustering/images/52c1ff91a8e9ccd778aa8c7d8a4ea8738234242f147a4a631be7bfa1906f2f7f.jpg)

![a07018f4ededa36ec3df92a4744c3e9433ddf935f7dfdeccbadf42bc48c2841d.jpg](../iclr_results/677_On the Price of Differential Privacy for Hierarchical Clustering/images/a07018f4ededa36ec3df92a4744c3e9433ddf935f7dfdeccbadf42bc48c2841d.jpg)

![a28e4cf79bc3a34ac74d6a249790cbed91e7247c5841e93f2fd343670bfe2bc1.jpg](../iclr_results/677_On the Price of Differential Privacy for Hierarchical Clustering/images/a28e4cf79bc3a34ac74d6a249790cbed91e7247c5841e93f2fd343670bfe2bc1.jpg)

![ab30d160fff89cd746fa3452b8a78bd4e43304a7bb17da5309c7f04477b81fdd.jpg](../iclr_results/677_On the Price of Differential Privacy for Hierarchical Clustering/images/ab30d160fff89cd746fa3452b8a78bd4e43304a7bb17da5309c7f04477b81fdd.jpg)

![fa6bae51a27e5d45b67f1eda426f5429ee44fde6775de63cea7a54363101664a.jpg](../iclr_results/677_On the Price of Differential Privacy for Hierarchical Clustering/images/fa6bae51a27e5d45b67f1eda426f5429ee44fde6775de63cea7a54363101664a.jpg)

### Tables

![3198a6d1c08f1ebe1cb7ccccafdddd8e5976029c919984201b57c7f6319343e8.jpg](../iclr_results/677_On the Price of Differential Privacy for Hierarchical Clustering/tables/3198a6d1c08f1ebe1cb7ccccafdddd8e5976029c919984201b57c7f6319343e8.jpg)

![e645b686b0cd7759a6bc87776f4ce0c2836a903f138b0076c7b9db947f511aaa.jpg](../iclr_results/677_On the Price of Differential Privacy for Hierarchical Clustering/tables/e645b686b0cd7759a6bc87776f4ce0c2836a903f138b0076c7b9db947f511aaa.jpg)

## Brain Mapping with Dense Features: Grounding Cortical Semantic Selectivity in Natural Images With Vision Transformers


### Images

![012203165a6cfee0c71fe3dc9fa2b908db689a5819497f855954159797d8dc4b.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/012203165a6cfee0c71fe3dc9fa2b908db689a5819497f855954159797d8dc4b.jpg)

![070aa7b9109d01e4e8e29b1b4246b8f1363c17c87fd80d24d6a15209ac3d91e5.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/070aa7b9109d01e4e8e29b1b4246b8f1363c17c87fd80d24d6a15209ac3d91e5.jpg)

![0c89ad9c92a74588dfac3025fabc9bf3652bb7f78af649d6aa8374339f0a9a38.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/0c89ad9c92a74588dfac3025fabc9bf3652bb7f78af649d6aa8374339f0a9a38.jpg)

![1589dfdf1a8907d3b4d5b068096ba3923ad0a0431e7acd6f35105e9ac0376083.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/1589dfdf1a8907d3b4d5b068096ba3923ad0a0431e7acd6f35105e9ac0376083.jpg)

![22bebbde8aad25114aaa33074cea5c9bab7e30ba83604710c2a28a18cb66ba7f.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/22bebbde8aad25114aaa33074cea5c9bab7e30ba83604710c2a28a18cb66ba7f.jpg)

![2934a6c6f2aecec914aedbf017ddd5638b177f20ecbe71b02be0bb31aeb29897.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/2934a6c6f2aecec914aedbf017ddd5638b177f20ecbe71b02be0bb31aeb29897.jpg)

![39760d81dbdd59e4f0efd724aea3651a430865aa37dc6ced01b97257ec0f0160.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/39760d81dbdd59e4f0efd724aea3651a430865aa37dc6ced01b97257ec0f0160.jpg)

![492ec3ecbd8dcf855de3911e1fb9dcdc526204d6dece7b3e036f651e631f3f25.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/492ec3ecbd8dcf855de3911e1fb9dcdc526204d6dece7b3e036f651e631f3f25.jpg)

![4bf379ec056a2fa50c1f32c08e7cc708818d26d93c28530167ced4ac72ed3636.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/4bf379ec056a2fa50c1f32c08e7cc708818d26d93c28530167ced4ac72ed3636.jpg)

![4d5b633cf87991857d08f75f21c9328f3d6950d0a968e84d8baa2baa23e09933.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/4d5b633cf87991857d08f75f21c9328f3d6950d0a968e84d8baa2baa23e09933.jpg)

![4dd414679983e3ea9925f2faedde18ae3517a998005632f8a43c53fa18b16dbc.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/4dd414679983e3ea9925f2faedde18ae3517a998005632f8a43c53fa18b16dbc.jpg)

![526237fee8f24afa4d7aebd3984d91d1a01de1fd70cc50746b96e915ae4cb830.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/526237fee8f24afa4d7aebd3984d91d1a01de1fd70cc50746b96e915ae4cb830.jpg)

![638c98f46c14bbefc0004ed36b1d6ee3cd21bc16cf5d09606b807cf4b72a7566.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/638c98f46c14bbefc0004ed36b1d6ee3cd21bc16cf5d09606b807cf4b72a7566.jpg)

![735ebdccc35112377f7824e0510d63454936f931a307bd243e4f36e88280e965.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/735ebdccc35112377f7824e0510d63454936f931a307bd243e4f36e88280e965.jpg)

![7653fba511f0e5b76a8d1c2f41c19cf7931177e84357d1e8d2c3ebe9003e5f57.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/7653fba511f0e5b76a8d1c2f41c19cf7931177e84357d1e8d2c3ebe9003e5f57.jpg)

![76a43345733bacc37d621fad932bbb52cf77e8a5f5a7f49a4a710b694cc73652.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/76a43345733bacc37d621fad932bbb52cf77e8a5f5a7f49a4a710b694cc73652.jpg)

![7859b3399014625734587fcce0bf3b4e516c3d7dc755e83052b675521a58e5b7.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/7859b3399014625734587fcce0bf3b4e516c3d7dc755e83052b675521a58e5b7.jpg)

![78b7bde99ecf6bd390dc0ca210bc21056ee7cbeb4ddc5c423a15e66d58ec2cf1.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/78b7bde99ecf6bd390dc0ca210bc21056ee7cbeb4ddc5c423a15e66d58ec2cf1.jpg)

![8a2eb2a80341a5d24ec62a1210aa194c26ebe6e8522b1372fe97d1e2a60e30e6.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/8a2eb2a80341a5d24ec62a1210aa194c26ebe6e8522b1372fe97d1e2a60e30e6.jpg)

![8cd2afbf32073416caf0f11ee4056c1371ba2e2d60af880429357f58d1c6c20b.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/8cd2afbf32073416caf0f11ee4056c1371ba2e2d60af880429357f58d1c6c20b.jpg)

![9d1497604661e5b588f0f012f61728e00ac44ca586e4b4607673fa4bd92d1f64.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/9d1497604661e5b588f0f012f61728e00ac44ca586e4b4607673fa4bd92d1f64.jpg)

![9f70f91386d5e2bf310515c2ab1791f75dfd5edcfec65a80c1cb397a2af6d919.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/9f70f91386d5e2bf310515c2ab1791f75dfd5edcfec65a80c1cb397a2af6d919.jpg)

![a754a398c6d3ce15bc17fcc39b1276332765f843a135567930ddb0a7643a88a8.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/a754a398c6d3ce15bc17fcc39b1276332765f843a135567930ddb0a7643a88a8.jpg)

![aaa59a80ae20a759f9aecf302cfdd1018982eb66f5c95fc34a7c427fa88e7d20.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/aaa59a80ae20a759f9aecf302cfdd1018982eb66f5c95fc34a7c427fa88e7d20.jpg)

![ab8b235a1c415417d76d7d58d7887f760d97d6d7ad5a9e1e0263c545e517d512.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/ab8b235a1c415417d76d7d58d7887f760d97d6d7ad5a9e1e0263c545e517d512.jpg)

![b424b2b179ee7984c8974c4d9a2c11b3da84ec25ece772b5a7e21c5fb5398165.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/b424b2b179ee7984c8974c4d9a2c11b3da84ec25ece772b5a7e21c5fb5398165.jpg)

![b75a5010fcd0a332c0f3c124a8d989f1043aa65cc97167c5f10b1c7417bd3ca7.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/b75a5010fcd0a332c0f3c124a8d989f1043aa65cc97167c5f10b1c7417bd3ca7.jpg)

![c553e2ee4f601ad6ab4ee94e01c9d6d19e23ac968e659d3d4f9022452808b387.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/c553e2ee4f601ad6ab4ee94e01c9d6d19e23ac968e659d3d4f9022452808b387.jpg)

![c6504043af4ca86c21bc218cf647a1fca8419fda181194835d615e3b0ecaf4c6.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/c6504043af4ca86c21bc218cf647a1fca8419fda181194835d615e3b0ecaf4c6.jpg)

![cd0a893f67bf9ff167b4913f54727f5135db39a14b6ee7b210339ad19fb47eef.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/cd0a893f67bf9ff167b4913f54727f5135db39a14b6ee7b210339ad19fb47eef.jpg)

![d12d4e9806c4818303ab29c3f53e8f361cd283e396d054f5633646d0455981a5.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/d12d4e9806c4818303ab29c3f53e8f361cd283e396d054f5633646d0455981a5.jpg)

![dcdf25c19a9c670cb942ae5425d0094ae4d19e3f275a4d2620373a2d905d5d77.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/dcdf25c19a9c670cb942ae5425d0094ae4d19e3f275a4d2620373a2d905d5d77.jpg)

![e0f68a932142f127ac59d7a0f821c815a0fdda420e938a41d25f786bdccac8de.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/e0f68a932142f127ac59d7a0f821c815a0fdda420e938a41d25f786bdccac8de.jpg)

![e717317060b4471bdafb0052d0b3e79abebdf3800160709299a58db122410a53.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/e717317060b4471bdafb0052d0b3e79abebdf3800160709299a58db122410a53.jpg)

![f1b178ba03e18440818c6e0089d0297f24cffa72f94aad9a66fd17f048cdd08f.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/f1b178ba03e18440818c6e0089d0297f24cffa72f94aad9a66fd17f048cdd08f.jpg)

![fce8bf5702e8789eeec5d4375391e4f716473d011f9f9dd230a500c738adfe2a.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/images/fce8bf5702e8789eeec5d4375391e4f716473d011f9f9dd230a500c738adfe2a.jpg)

### Tables

![8ff49fe07c95f66b25203dbc2ca386fafb276366234aebbb3746b04501362154.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/tables/8ff49fe07c95f66b25203dbc2ca386fafb276366234aebbb3746b04501362154.jpg)

![a44fa70782a48e04b6bbfb22eba4f6c4572e48313d3297895d64f95d24ab9928.jpg](../iclr_results/678_Brain Mapping with Dense Features_ Grounding Cortical Semantic Selectivity in Natural Images With Vi/tables/a44fa70782a48e04b6bbfb22eba4f6c4572e48313d3297895d64f95d24ab9928.jpg)

## A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Translations


### Images

![014e05d69fae89c346adcb80c0f444b3cbbd3a71edf93b401c68c24336157148.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/014e05d69fae89c346adcb80c0f444b3cbbd3a71edf93b401c68c24336157148.jpg)

![04648b5f4c0ade14fd601fc6da8fc21435e0e75bafaa240c9e5d1b17fa258636.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/04648b5f4c0ade14fd601fc6da8fc21435e0e75bafaa240c9e5d1b17fa258636.jpg)

![10f82c2c74e4a141f638ea55093ab335cdb32b1cc01e18feeb69cc6b6effb982.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/10f82c2c74e4a141f638ea55093ab335cdb32b1cc01e18feeb69cc6b6effb982.jpg)

![13e5bd930c2a4ad78b1135983ace79bcad520c088b1f643a66b8fed08ea5aec3.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/13e5bd930c2a4ad78b1135983ace79bcad520c088b1f643a66b8fed08ea5aec3.jpg)

![171102f0569de07893afbf76c9ed898c0289eb7f076d3be55c67f8a02de4a726.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/171102f0569de07893afbf76c9ed898c0289eb7f076d3be55c67f8a02de4a726.jpg)

![21fd982b610346c89c930c9a6bad833cf7cbee2c51ea19c2534a705c91bf6395.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/21fd982b610346c89c930c9a6bad833cf7cbee2c51ea19c2534a705c91bf6395.jpg)

![253f996d248737a670b4b1ad3ad939991bb7d2b3234b55c7142ee15040ca13e3.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/253f996d248737a670b4b1ad3ad939991bb7d2b3234b55c7142ee15040ca13e3.jpg)

![25dbf62066868ecad080d484ce87b73702ce8b81272dbf2e2228833b24ca92f1.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/25dbf62066868ecad080d484ce87b73702ce8b81272dbf2e2228833b24ca92f1.jpg)

![2b89ecf29b46b8f41a039443631e74b9d3507ec6c92d2fdef3814671aa59f149.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/2b89ecf29b46b8f41a039443631e74b9d3507ec6c92d2fdef3814671aa59f149.jpg)

![32eb94089b6f71ad5188c94c503c1e2b6829aa5ddf90d00d95952ff4f869351a.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/32eb94089b6f71ad5188c94c503c1e2b6829aa5ddf90d00d95952ff4f869351a.jpg)

![42d3286e17bf208484b5950d9a5ef839bc389fb6171712d973de6db3e6e96acc.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/42d3286e17bf208484b5950d9a5ef839bc389fb6171712d973de6db3e6e96acc.jpg)

![50f49a6366bda450b7a111589f1f086daa8e90f81a76286c28b75fd71316b6c1.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/50f49a6366bda450b7a111589f1f086daa8e90f81a76286c28b75fd71316b6c1.jpg)

![64123f72cd981a2ed28d7e01a3f3e3167ee472667f0f5b187b802242bf9f8ef7.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/64123f72cd981a2ed28d7e01a3f3e3167ee472667f0f5b187b802242bf9f8ef7.jpg)

![6675b8d45cf9a3dffa6ebc9f45c11d6596fabe92e758720bf781c6e2fa3b11ea.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/6675b8d45cf9a3dffa6ebc9f45c11d6596fabe92e758720bf781c6e2fa3b11ea.jpg)

![6aee66baa931d00629751109337fc5b737f3b49454915d50fd2440f1f1211c6a.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/6aee66baa931d00629751109337fc5b737f3b49454915d50fd2440f1f1211c6a.jpg)

![6df53a6c77558ebfcb24b210d387ccbaa677e2092dbf5662d2861b8e07efab12.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/6df53a6c77558ebfcb24b210d387ccbaa677e2092dbf5662d2861b8e07efab12.jpg)

![73e832f1544a27de61c40f3857a9165a0354bbeb71064dee3073df977a5aed55.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/73e832f1544a27de61c40f3857a9165a0354bbeb71064dee3073df977a5aed55.jpg)

![778113b57c42d0ab6c250c72fd00a93d1edac121dce386330470d4ad1eaf1699.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/778113b57c42d0ab6c250c72fd00a93d1edac121dce386330470d4ad1eaf1699.jpg)

![79f5ead6134bc76d0f6edfb9a24c12694963f07266edaf43f3222d1d507dfe69.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/79f5ead6134bc76d0f6edfb9a24c12694963f07266edaf43f3222d1d507dfe69.jpg)

![7acea8911745eef345d4857ac688ece4fb2882b683f7c78f81dbface01dc534f.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/7acea8911745eef345d4857ac688ece4fb2882b683f7c78f81dbface01dc534f.jpg)

![818e4ba1c444d37eecb8b4dad613cd8fb1e46c006855acc4c6ab0e9e1d94a7d0.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/818e4ba1c444d37eecb8b4dad613cd8fb1e46c006855acc4c6ab0e9e1d94a7d0.jpg)

![872d4dec1e4fdb9439fd6991453d2620cca62718870b6e103e5e2981e1a7b8ac.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/872d4dec1e4fdb9439fd6991453d2620cca62718870b6e103e5e2981e1a7b8ac.jpg)

![8c8801d55cb1dd63c9c7d6bdfbfdd0cc33d2c595fb7756744359585a1c8747fe.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/8c8801d55cb1dd63c9c7d6bdfbfdd0cc33d2c595fb7756744359585a1c8747fe.jpg)

![8ea3cbae60cb92a6fa0669ba4271e01655e02535f6f9c812ed20d132a45ea9ca.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/8ea3cbae60cb92a6fa0669ba4271e01655e02535f6f9c812ed20d132a45ea9ca.jpg)

![9704faaf1c1d1059338a52e7a89932fcd04fecc397fb5f22d75d905881ebdf99.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/9704faaf1c1d1059338a52e7a89932fcd04fecc397fb5f22d75d905881ebdf99.jpg)

![9fe1f6fce50695142d96f0307cd760cf445491123fccad87764e2504732573da.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/9fe1f6fce50695142d96f0307cd760cf445491123fccad87764e2504732573da.jpg)

![ad7048da10d669e20787fbbca25b912abe8e43634a3e3ab5c5db83c19b214a83.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/ad7048da10d669e20787fbbca25b912abe8e43634a3e3ab5c5db83c19b214a83.jpg)

![b256622e0878aaff9d2fb79ecfc4369e431f62f2857ffb17b8a75519be7120bd.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/b256622e0878aaff9d2fb79ecfc4369e431f62f2857ffb17b8a75519be7120bd.jpg)

![b28b0e2f9a34836344569b3b9e641cf82360f52e9d35dd5edd92f71ea5f6d2e1.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/b28b0e2f9a34836344569b3b9e641cf82360f52e9d35dd5edd92f71ea5f6d2e1.jpg)

![c096a2846551e43f9dd43c44ce41da24e90fd00b367627897c47004e69ead1b5.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/c096a2846551e43f9dd43c44ce41da24e90fd00b367627897c47004e69ead1b5.jpg)

![c2e3cb04cba7712d262b205a3b427f0293a106fbc6692e2b8465970fddd82ebb.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/c2e3cb04cba7712d262b205a3b427f0293a106fbc6692e2b8465970fddd82ebb.jpg)

![d6e335c157a2f5076bd02a3fd3f79e7b2e43ba74cff35ece7978e5b600f4bb0f.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/d6e335c157a2f5076bd02a3fd3f79e7b2e43ba74cff35ece7978e5b600f4bb0f.jpg)

![db85786e1c371860858186b58c68a3c6729415e9f9f5c92b4d3bcdf53c95c53d.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/db85786e1c371860858186b58c68a3c6729415e9f9f5c92b4d3bcdf53c95c53d.jpg)

![fc8bda021931ef7b498258ebe5a5fc97ee809b3ae8fdf8955c20613a96b3d916.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/fc8bda021931ef7b498258ebe5a5fc97ee809b3ae8fdf8955c20613a96b3d916.jpg)

![fccb1f5abbbb1c48bfbcf7441e74d7f2755f873c42ecd5825ab1ec47c0913f50.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/images/fccb1f5abbbb1c48bfbcf7441e74d7f2755f873c42ecd5825ab1ec47c0913f50.jpg)

### Tables

![01626a8065056f6da927f13b11a86a6e1e8aa1368e745a9cf08163ff8d9ffbd6.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/tables/01626a8065056f6da927f13b11a86a6e1e8aa1368e745a9cf08163ff8d9ffbd6.jpg)

![1394eb7896cd59265337e73c9adbea9dbcef35a5b37290ec425d1f1b8e4c9c3d.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/tables/1394eb7896cd59265337e73c9adbea9dbcef35a5b37290ec425d1f1b8e4c9c3d.jpg)

![2e42e86721a480fc83e6c77d321616b4ca6b4a550be998f3cccb806fc3e63346.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/tables/2e42e86721a480fc83e6c77d321616b4ca6b4a550be998f3cccb806fc3e63346.jpg)

![3cb978e2d3943156715cde9284ea742ed75ddfc3f21d1abad927f3ce9e712ce6.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/tables/3cb978e2d3943156715cde9284ea742ed75ddfc3f21d1abad927f3ce9e712ce6.jpg)

![50fd1471b1fc2249d8092c68ef825dc75a57766e6fef0eb1f0a511c9238e7bf6.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/tables/50fd1471b1fc2249d8092c68ef825dc75a57766e6fef0eb1f0a511c9238e7bf6.jpg)

![5b4e2b84a7d5cceccb58dd2305f0889bbc85b7e3a6d82845248e56372989b327.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/tables/5b4e2b84a7d5cceccb58dd2305f0889bbc85b7e3a6d82845248e56372989b327.jpg)

![69ff8abd42a7529bfdd7fd3cc8082c45c643aae20198d9bc569f7ec0081ac80f.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/tables/69ff8abd42a7529bfdd7fd3cc8082c45c643aae20198d9bc569f7ec0081ac80f.jpg)

![72c25839b10c0ff5a0ff496c38fb237298b730083e871149d8d737e6c32df648.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/tables/72c25839b10c0ff5a0ff496c38fb237298b730083e871149d8d737e6c32df648.jpg)

![81712311fa2e33f11efb9aa4e32ef2adbc38814dc58ab4c6d2ba70bf3f48556e.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/tables/81712311fa2e33f11efb9aa4e32ef2adbc38814dc58ab4c6d2ba70bf3f48556e.jpg)

![8319ef61b379ac48e67dc3b27b1e9426f8469f65de1dbc20d0bc28bea4e0e6fc.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/tables/8319ef61b379ac48e67dc3b27b1e9426f8469f65de1dbc20d0bc28bea4e0e6fc.jpg)

![c7b805a58f65f28c772e0ecfe5ffafd3eb78c6268e0078280d09170c27644143.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/tables/c7b805a58f65f28c772e0ecfe5ffafd3eb78c6268e0078280d09170c27644143.jpg)

![c994a996ef1f6f66ab1bc6ab255f784a2ab9e0a66481b57695111aa6a28b1672.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/tables/c994a996ef1f6f66ab1bc6ab255f784a2ab9e0a66481b57695111aa6a28b1672.jpg)

![da2318e863ccf3846867e66d9a2a2dfa2aa37c49fded653d9ad0ae7ef897e724.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/tables/da2318e863ccf3846867e66d9a2a2dfa2aa37c49fded653d9ad0ae7ef897e724.jpg)

![e14a789cd6da47ccffdc0b387286c80945fcec63b28347ca7dc34a4d20ef0033.jpg](../iclr_results/679_A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Transl/tables/e14a789cd6da47ccffdc0b387286c80945fcec63b28347ca7dc34a4d20ef0033.jpg)

## Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding


### Images

![093dc405108e8d86ab3a7204ea42cb216eddc498ea703416e4adc4891ee18678.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/images/093dc405108e8d86ab3a7204ea42cb216eddc498ea703416e4adc4891ee18678.jpg)

![173e8c198a176b2ab893fa30681273cfcfa6a8d5a6bc974a9dc22512613e71c0.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/images/173e8c198a176b2ab893fa30681273cfcfa6a8d5a6bc974a9dc22512613e71c0.jpg)

![184a3e7c0a39a24e00a6b1582d418f76a7f6d938b90b5f02a298ca9b1a5c5224.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/images/184a3e7c0a39a24e00a6b1582d418f76a7f6d938b90b5f02a298ca9b1a5c5224.jpg)

![201bb80bb53631f13e46af95db7835f9c6dae8c15d1d0704545d9d02bc47338f.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/images/201bb80bb53631f13e46af95db7835f9c6dae8c15d1d0704545d9d02bc47338f.jpg)

![4c57f0d44e2c9f3985f3a0f5efb5d048796ea456495bb01de520bcf945197aa6.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/images/4c57f0d44e2c9f3985f3a0f5efb5d048796ea456495bb01de520bcf945197aa6.jpg)

![7cee10870c1f66709b9f5953afab9e77c19ff215905e84d848d79e4df39192cb.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/images/7cee10870c1f66709b9f5953afab9e77c19ff215905e84d848d79e4df39192cb.jpg)

![ac85c644c718dd362ae8c0d6797c1cc4ad94bc26341c1c038825c4e0f8ba2481.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/images/ac85c644c718dd362ae8c0d6797c1cc4ad94bc26341c1c038825c4e0f8ba2481.jpg)

![bb3b530e199e5922e1b3144c5c4a68d37d35465667aaf4ce7e6e06cfebdde419.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/images/bb3b530e199e5922e1b3144c5c4a68d37d35465667aaf4ce7e6e06cfebdde419.jpg)

![ce2342a32c425a0db34e9bbe3d0f709e39fac6c91bb582ad87db9b15d8a8faa3.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/images/ce2342a32c425a0db34e9bbe3d0f709e39fac6c91bb582ad87db9b15d8a8faa3.jpg)

![d9b4461b794affe9b660b7daa3f9b2807e5626a4ff78114a18a9c180a33e6850.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/images/d9b4461b794affe9b660b7daa3f9b2807e5626a4ff78114a18a9c180a33e6850.jpg)

![ef0ea82d2b4fdee37a8c94e88a494bf9dec435ce000ca2b7ab1d3da3c33aa2f8.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/images/ef0ea82d2b4fdee37a8c94e88a494bf9dec435ce000ca2b7ab1d3da3c33aa2f8.jpg)

![f38f40dab4ceda7e5ceaed282074469996c8efe3e46fbb5ed792908126c396a3.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/images/f38f40dab4ceda7e5ceaed282074469996c8efe3e46fbb5ed792908126c396a3.jpg)

![faa41579344a286de2316dfa4691c5fbc712cd56e038270df53a83a6ee128fe7.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/images/faa41579344a286de2316dfa4691c5fbc712cd56e038270df53a83a6ee128fe7.jpg)

### Tables

![00709a8186417eef44a026d5b25da75e26c63abe00353e29d7ebf6fa517cda57.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/tables/00709a8186417eef44a026d5b25da75e26c63abe00353e29d7ebf6fa517cda57.jpg)

![0a6d2d1e2b1ce8ddeb54f32e67a39569255069cca5ee8dbb55150e7194092dff.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/tables/0a6d2d1e2b1ce8ddeb54f32e67a39569255069cca5ee8dbb55150e7194092dff.jpg)

![122b16228a8481de58ee9a29aed38efe55b142296a1a6d095eac976a4f8afdcf.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/tables/122b16228a8481de58ee9a29aed38efe55b142296a1a6d095eac976a4f8afdcf.jpg)

![13d220605bc60b17291b2f271e59c12684df91868bde55be07ba83d9e5fa7baa.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/tables/13d220605bc60b17291b2f271e59c12684df91868bde55be07ba83d9e5fa7baa.jpg)

![180f00f17e1ea189e9abff78c1592f278daff06244c62fe2aca5346eec1a4264.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/tables/180f00f17e1ea189e9abff78c1592f278daff06244c62fe2aca5346eec1a4264.jpg)

![22f2f73fd341cb754a0640528fff981fe9dc38b72e706b17d2a7617ae179e07a.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/tables/22f2f73fd341cb754a0640528fff981fe9dc38b72e706b17d2a7617ae179e07a.jpg)

![2eae71852f14ccff5e718fc2642367ee888d7ef6708b2def0143334227df701a.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/tables/2eae71852f14ccff5e718fc2642367ee888d7ef6708b2def0143334227df701a.jpg)

![3f93ced35ac7096dedba60a376975429be518595fcea6e681eda59abec45e18e.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/tables/3f93ced35ac7096dedba60a376975429be518595fcea6e681eda59abec45e18e.jpg)

![6668f9541591f8add3f77938c6014cd8aef5d41939fd4be52bbc81832fbdac34.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/tables/6668f9541591f8add3f77938c6014cd8aef5d41939fd4be52bbc81832fbdac34.jpg)

![8bef07a58f9508cf11a4ac507b068745b676b5749f2f1ee60d8310df90ad9801.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/tables/8bef07a58f9508cf11a4ac507b068745b676b5749f2f1ee60d8310df90ad9801.jpg)

![a406f613490398edac3da91b2ba6de4334062d568815ffb3b1e803b44e81560b.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/tables/a406f613490398edac3da91b2ba6de4334062d568815ffb3b1e803b44e81560b.jpg)

![ab1d15a0f7bea45d1729caa4d3db9784f554a67d601b9de87667218c6b8d99a6.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/tables/ab1d15a0f7bea45d1729caa4d3db9784f554a67d601b9de87667218c6b8d99a6.jpg)

![af721f1d6a6a909ce98fdb18d1dd3beaae3d0cdcf7048c05f7c84088abc26bc4.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/tables/af721f1d6a6a909ce98fdb18d1dd3beaae3d0cdcf7048c05f7c84088abc26bc4.jpg)

![bb04f57b4dd765225682a8d62b855242bd2f673e32f0fff37deb0a5d13fa6705.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/tables/bb04f57b4dd765225682a8d62b855242bd2f673e32f0fff37deb0a5d13fa6705.jpg)

![be1f455cd0a51bd634c2c7ef7dcf0686911bddbf0f718304b76e7eef505c8d37.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/tables/be1f455cd0a51bd634c2c7ef7dcf0686911bddbf0f718304b76e7eef505c8d37.jpg)

![c8cd31c2b769068b0c5e26cf07687c3283af35e57a2b17a54dc41520d9d39785.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/tables/c8cd31c2b769068b0c5e26cf07687c3283af35e57a2b17a54dc41520d9d39785.jpg)

![e2be18edc5b4ad4a7dcb74eea2bf1ffe49e7ce7c2c37340a2c3e6304d5ba0c2a.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/tables/e2be18edc5b4ad4a7dcb74eea2bf1ffe49e7ce7c2c37340a2c3e6304d5ba0c2a.jpg)

![e669e8702b19625221b14092cde76793823526ceece26ed5f48f0f6c31e89630.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/tables/e669e8702b19625221b14092cde76793823526ceece26ed5f48f0f6c31e89630.jpg)

![fd1f3dec9e2b81dc622fa0a5e6b0cda09da7ab32b8a5d438f1ccb851cd6eb46c.jpg](../iclr_results/680_Contextual Self-paced Learning for Weakly Supervised Spatio-Temporal Video Grounding/tables/fd1f3dec9e2b81dc622fa0a5e6b0cda09da7ab32b8a5d438f1ccb851cd6eb46c.jpg)

## Image and Video Tokenization with Binary Spherical Quantization


### Images

![0559782b1187f879349e2c13063796ee356fa33161de0e518b8ad2872d5fdd1e.jpg](../iclr_results/681_Image and Video Tokenization with Binary Spherical Quantization/images/0559782b1187f879349e2c13063796ee356fa33161de0e518b8ad2872d5fdd1e.jpg)

![3943482eb9bd41999776bf8524f0181ff5b94b330fafd192a57ddab1eddd059c.jpg](../iclr_results/681_Image and Video Tokenization with Binary Spherical Quantization/images/3943482eb9bd41999776bf8524f0181ff5b94b330fafd192a57ddab1eddd059c.jpg)

![48fcf48ccd8ecb3820bec7c365ede3e008ec152281fe959b42631a6565567a23.jpg](../iclr_results/681_Image and Video Tokenization with Binary Spherical Quantization/images/48fcf48ccd8ecb3820bec7c365ede3e008ec152281fe959b42631a6565567a23.jpg)

![68a1a84d53df363b54bd68e45a13df616ac119a45c2d116ccac6ca181e8df6b1.jpg](../iclr_results/681_Image and Video Tokenization with Binary Spherical Quantization/images/68a1a84d53df363b54bd68e45a13df616ac119a45c2d116ccac6ca181e8df6b1.jpg)

![7ed49bcfd4423966469342a230491fdecff618967bbd76491cb31c46b673c45b.jpg](../iclr_results/681_Image and Video Tokenization with Binary Spherical Quantization/images/7ed49bcfd4423966469342a230491fdecff618967bbd76491cb31c46b673c45b.jpg)

![8116fabc400a5844a557d36bd48169b052c8454f67ef95e30c91022890de66f0.jpg](../iclr_results/681_Image and Video Tokenization with Binary Spherical Quantization/images/8116fabc400a5844a557d36bd48169b052c8454f67ef95e30c91022890de66f0.jpg)

![a36510a52534145a804c8fb0e43ec6d67e77230ccaa81c7ba162b7f61b03ef80.jpg](../iclr_results/681_Image and Video Tokenization with Binary Spherical Quantization/images/a36510a52534145a804c8fb0e43ec6d67e77230ccaa81c7ba162b7f61b03ef80.jpg)

![a729ea2f872266889fd5c246f3a43363f35d4e81576950854a3b48ca50fcff25.jpg](../iclr_results/681_Image and Video Tokenization with Binary Spherical Quantization/images/a729ea2f872266889fd5c246f3a43363f35d4e81576950854a3b48ca50fcff25.jpg)

![c22496f3e1ca8ae953a9fbfc13a8e70d233628b101dd99cab4eefc7bb745bf63.jpg](../iclr_results/681_Image and Video Tokenization with Binary Spherical Quantization/images/c22496f3e1ca8ae953a9fbfc13a8e70d233628b101dd99cab4eefc7bb745bf63.jpg)

![c46b3d79823cece0b84a9240d1f55154d86b73fbba60c18579555c221bfcdd78.jpg](../iclr_results/681_Image and Video Tokenization with Binary Spherical Quantization/images/c46b3d79823cece0b84a9240d1f55154d86b73fbba60c18579555c221bfcdd78.jpg)

![ca01319bf25174129b5c3189f9f8eb1ea0990f8c80aabf1a9e328619b36578eb.jpg](../iclr_results/681_Image and Video Tokenization with Binary Spherical Quantization/images/ca01319bf25174129b5c3189f9f8eb1ea0990f8c80aabf1a9e328619b36578eb.jpg)

![fd0595096a92796057448e0fea8607c3aa4c37783115f41802d2f7b4fa06b83c.jpg](../iclr_results/681_Image and Video Tokenization with Binary Spherical Quantization/images/fd0595096a92796057448e0fea8607c3aa4c37783115f41802d2f7b4fa06b83c.jpg)

### Tables

![0999c7863c332f0b97e823a7c75e8b40b39beeda07f01475960ed83214cfba70.jpg](../iclr_results/681_Image and Video Tokenization with Binary Spherical Quantization/tables/0999c7863c332f0b97e823a7c75e8b40b39beeda07f01475960ed83214cfba70.jpg)

![27d09d27fc4553409576a4e10ab99f67f50a409489e54b94d0b23bdd2c62ae02.jpg](../iclr_results/681_Image and Video Tokenization with Binary Spherical Quantization/tables/27d09d27fc4553409576a4e10ab99f67f50a409489e54b94d0b23bdd2c62ae02.jpg)

![49631303b6a0a2b0d4f5797c3b142d9963cd1d9769afcb1116106cb0e55c4028.jpg](../iclr_results/681_Image and Video Tokenization with Binary Spherical Quantization/tables/49631303b6a0a2b0d4f5797c3b142d9963cd1d9769afcb1116106cb0e55c4028.jpg)

![49ba52ab02de190c863f724ddbd8f702be7c61f105fe0ac04ec36eddbeff40bd.jpg](../iclr_results/681_Image and Video Tokenization with Binary Spherical Quantization/tables/49ba52ab02de190c863f724ddbd8f702be7c61f105fe0ac04ec36eddbeff40bd.jpg)

![59e80e6497663da8a06ff113702c934b5a919e840ba0d5a69dbf79386527e014.jpg](../iclr_results/681_Image and Video Tokenization with Binary Spherical Quantization/tables/59e80e6497663da8a06ff113702c934b5a919e840ba0d5a69dbf79386527e014.jpg)

![60c0e8a3d5eed252105f91297608891c1bcc09f2aad419ece120e87d2b21c053.jpg](../iclr_results/681_Image and Video Tokenization with Binary Spherical Quantization/tables/60c0e8a3d5eed252105f91297608891c1bcc09f2aad419ece120e87d2b21c053.jpg)

![6309db4fa636bd6097a09bad9c5f3257f1e5639ef8c5e96a7f33457d829de365.jpg](../iclr_results/681_Image and Video Tokenization with Binary Spherical Quantization/tables/6309db4fa636bd6097a09bad9c5f3257f1e5639ef8c5e96a7f33457d829de365.jpg)

![6ab356d8cd9b16794ba76d485a69a5c82dc0224c2d82bd4994ec2b0ad63e59c0.jpg](../iclr_results/681_Image and Video Tokenization with Binary Spherical Quantization/tables/6ab356d8cd9b16794ba76d485a69a5c82dc0224c2d82bd4994ec2b0ad63e59c0.jpg)

![990204c61c1ded0e39fd6e2c6d36d512365d3ce2065c029044dd9f207bc773b6.jpg](../iclr_results/681_Image and Video Tokenization with Binary Spherical Quantization/tables/990204c61c1ded0e39fd6e2c6d36d512365d3ce2065c029044dd9f207bc773b6.jpg)

![e3c442cca4a624fae4a05f83f173c36d7b14f4f7375237f5efcf37c7a057d02f.jpg](../iclr_results/681_Image and Video Tokenization with Binary Spherical Quantization/tables/e3c442cca4a624fae4a05f83f173c36d7b14f4f7375237f5efcf37c7a057d02f.jpg)

![ef1173c5f6701f63933262a048b9d180d6c7c2798d1de8592bd9797924196a05.jpg](../iclr_results/681_Image and Video Tokenization with Binary Spherical Quantization/tables/ef1173c5f6701f63933262a048b9d180d6c7c2798d1de8592bd9797924196a05.jpg)

## Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment


### Images

![01a6a9be5688790e1710e77eebaf6814583b7035dd038f34c62d8de68688ee1b.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/01a6a9be5688790e1710e77eebaf6814583b7035dd038f34c62d8de68688ee1b.jpg)

![057dacaefcf6bce18c81b303165ab6b7324c0d0e8a0a2a2c4f661de2dcc12408.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/057dacaefcf6bce18c81b303165ab6b7324c0d0e8a0a2a2c4f661de2dcc12408.jpg)

![078d9ed718ba7011349535cdf1c751d26af48cd009c29141d691c4075215fb0e.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/078d9ed718ba7011349535cdf1c751d26af48cd009c29141d691c4075215fb0e.jpg)

![103d055a35346058cded81e87cfee91ddb4b3eaed61eeb32bd5e3b4d70de59d3.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/103d055a35346058cded81e87cfee91ddb4b3eaed61eeb32bd5e3b4d70de59d3.jpg)

![1805d7f6e300da49287eb3c209f6df133a773e948fcfbfbafa15bf9e76d828f8.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/1805d7f6e300da49287eb3c209f6df133a773e948fcfbfbafa15bf9e76d828f8.jpg)

![2a54906778f1181d545dbf93d1051ebe848b4e29f2027d56b3d44db004dc06cb.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/2a54906778f1181d545dbf93d1051ebe848b4e29f2027d56b3d44db004dc06cb.jpg)

![2e0a3486af56d04a3dc3c60dc085d2f75d3a002138796ee8e916d5cb1e8e8dc6.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/2e0a3486af56d04a3dc3c60dc085d2f75d3a002138796ee8e916d5cb1e8e8dc6.jpg)

![2fd77c28625ce40898178a337f2118e0c3ae196c548746d5e46e287a43334f53.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/2fd77c28625ce40898178a337f2118e0c3ae196c548746d5e46e287a43334f53.jpg)

![307ec7afec5a2c6a8e453dc55422493e9cb6a213c77ead2bf85269c981a30610.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/307ec7afec5a2c6a8e453dc55422493e9cb6a213c77ead2bf85269c981a30610.jpg)

![38f943e924214ab96bc104d1712ec02efa92ac8819fdd4b2f7aef8197ee9ce1c.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/38f943e924214ab96bc104d1712ec02efa92ac8819fdd4b2f7aef8197ee9ce1c.jpg)

![3b5fb3de84e82fe3205c633cb2ce0ce5a17aa509da92db230399429073ff7aa0.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/3b5fb3de84e82fe3205c633cb2ce0ce5a17aa509da92db230399429073ff7aa0.jpg)

![3fb62149f6249b1310b0d1c932bfe282f236ae22bbef73af3b478eb4b2058456.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/3fb62149f6249b1310b0d1c932bfe282f236ae22bbef73af3b478eb4b2058456.jpg)

![42fc6292a4f3fdccb2bfa49cc6ea919d975e6e97ef57c33e9da925c51c5aab7f.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/42fc6292a4f3fdccb2bfa49cc6ea919d975e6e97ef57c33e9da925c51c5aab7f.jpg)

![5503e8990249c1ad3b8eac5e4736ad455c99dcbbba2022784499e416f218fdbc.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/5503e8990249c1ad3b8eac5e4736ad455c99dcbbba2022784499e416f218fdbc.jpg)

![553089cf658d4988ef954907bf82a99e19806546cbe9cf5aea979118e2ed2336.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/553089cf658d4988ef954907bf82a99e19806546cbe9cf5aea979118e2ed2336.jpg)

![68f35ec97430980198810c044c895af9b4c9536c6ff2ff9a26a21ea12a66a1e3.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/68f35ec97430980198810c044c895af9b4c9536c6ff2ff9a26a21ea12a66a1e3.jpg)

![6d1671fdf1ab03c5454a552b36f127b2346886171a97a9a59b14d096a4a20c9b.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/6d1671fdf1ab03c5454a552b36f127b2346886171a97a9a59b14d096a4a20c9b.jpg)

![7516263dc32c74fb201b9eed1c569282a94994195d37f217273c6c9566ec05b3.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/7516263dc32c74fb201b9eed1c569282a94994195d37f217273c6c9566ec05b3.jpg)

![784ac5de4b524d7b42520e3f5310a5c841574265d24543036fc8a261127b9105.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/784ac5de4b524d7b42520e3f5310a5c841574265d24543036fc8a261127b9105.jpg)

![7aa88ac54be54eea275f6c00bce00c1a03ed8ba775d61b52bad75d32e4b093c7.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/7aa88ac54be54eea275f6c00bce00c1a03ed8ba775d61b52bad75d32e4b093c7.jpg)

![7b55ee27b2289193336691d1d78b0638cb69c47c0e3b24906661d5e64bbbda9f.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/7b55ee27b2289193336691d1d78b0638cb69c47c0e3b24906661d5e64bbbda9f.jpg)

![80f523f7ebb9a4072a7e32933724359aacb1a3ff0be31cba6738a8010919b941.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/80f523f7ebb9a4072a7e32933724359aacb1a3ff0be31cba6738a8010919b941.jpg)

![838b3f0329c5919663e5320cb483e7843c2fd9b0838d004ee1759c6557b91c90.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/838b3f0329c5919663e5320cb483e7843c2fd9b0838d004ee1759c6557b91c90.jpg)

![8936d63b99f744d209a7d7db27d96fcfa309fb163cbc1d0e06fe014fbf47e0d5.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/8936d63b99f744d209a7d7db27d96fcfa309fb163cbc1d0e06fe014fbf47e0d5.jpg)

![9f0448f215bdfd2ca8cde490f79d9819c4da8c836fa67ad08cac5c2c96458eaf.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/9f0448f215bdfd2ca8cde490f79d9819c4da8c836fa67ad08cac5c2c96458eaf.jpg)

![a3f37c15f86b9488004be82145edee703b268eacd58ddf84d1c255b1fd62d55b.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/a3f37c15f86b9488004be82145edee703b268eacd58ddf84d1c255b1fd62d55b.jpg)

![b8eddddc68fc760cc1e43bdbf5df52c19ddefb62b72b1763ee430717d224684a.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/b8eddddc68fc760cc1e43bdbf5df52c19ddefb62b72b1763ee430717d224684a.jpg)

![b91cfb6bbf01343487141a54b86891cd8deb6e0ae8206487b85edfaf7d6d0ba4.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/b91cfb6bbf01343487141a54b86891cd8deb6e0ae8206487b85edfaf7d6d0ba4.jpg)

![bc428a9a3010d105ac87a7849477a41fbe214d3469b11604d4aeb67a91c8f13b.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/bc428a9a3010d105ac87a7849477a41fbe214d3469b11604d4aeb67a91c8f13b.jpg)

![c2a1a1942b308c607cca1f0a26e7a531b8f6de77cb5bdcdf0416a04f040e8ed4.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/c2a1a1942b308c607cca1f0a26e7a531b8f6de77cb5bdcdf0416a04f040e8ed4.jpg)

![ca88d20bd7199b30814ec3bc689ab7ddebfcd6bad39e6291f77d9a43a0f4af70.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/ca88d20bd7199b30814ec3bc689ab7ddebfcd6bad39e6291f77d9a43a0f4af70.jpg)

![d58062b6d10868442228b602bbe1436b679df8a1d12524f355783faea4ac635a.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/d58062b6d10868442228b602bbe1436b679df8a1d12524f355783faea4ac635a.jpg)

![deeb225ecbd81cce47ff15906059c01d0fa222e7c7a84ed0503080f84a76df93.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/deeb225ecbd81cce47ff15906059c01d0fa222e7c7a84ed0503080f84a76df93.jpg)

![e664e19fe3ef48bfbdb4e7417bdcd443883bc4c8bfa8c0751e7a6733c9302d06.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/e664e19fe3ef48bfbdb4e7417bdcd443883bc4c8bfa8c0751e7a6733c9302d06.jpg)

![ee4548047c2c3933c0a2efd46f0ae67a8d03faae9ac6ff10418b5372add87a03.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/ee4548047c2c3933c0a2efd46f0ae67a8d03faae9ac6ff10418b5372add87a03.jpg)

![ee8d49d053d1b0e2b75a76333569312288c311c780f68a7f31fa4282741001df.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/ee8d49d053d1b0e2b75a76333569312288c311c780f68a7f31fa4282741001df.jpg)

![f18653b3e6d0727ce07ab2374cc660ddd923c0ede05413fc2a5499ff3d335b3e.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/f18653b3e6d0727ce07ab2374cc660ddd923c0ede05413fc2a5499ff3d335b3e.jpg)

![f4b4965d81fd34985138ad5a898bf0bc96f9338d795ca1d98793671447771db3.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/f4b4965d81fd34985138ad5a898bf0bc96f9338d795ca1d98793671447771db3.jpg)

![f4e19e6ac7ed4a5998acb753bd36b3f30096e69941b4de82e47e5e9dac959fee.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/f4e19e6ac7ed4a5998acb753bd36b3f30096e69941b4de82e47e5e9dac959fee.jpg)

![f6c045b3f2431fcb6c888760681a960dfcbfc0f7d9d9cc36e85b2a530c9936f9.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/f6c045b3f2431fcb6c888760681a960dfcbfc0f7d9d9cc36e85b2a530c9936f9.jpg)

![fb6b3057b653c4518ddc0959bac704872361f49f9f82ed7d4666753c66b1be25.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/images/fb6b3057b653c4518ddc0959bac704872361f49f9f82ed7d4666753c66b1be25.jpg)

### Tables

![00cfb9fc20a8c0c4569c85629e8e14b769daa28a085949b6c0b265dc4a32a6ef.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/tables/00cfb9fc20a8c0c4569c85629e8e14b769daa28a085949b6c0b265dc4a32a6ef.jpg)

![0bf65d1ed60c745d57574b8507171cfd8046924202d303ec17649038f5f73afa.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/tables/0bf65d1ed60c745d57574b8507171cfd8046924202d303ec17649038f5f73afa.jpg)

![1382d36dbd1642aa4bb0773009566e32bcd3dbdffe8eee0438597f70dec4adce.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/tables/1382d36dbd1642aa4bb0773009566e32bcd3dbdffe8eee0438597f70dec4adce.jpg)

![2cf4ac761bd4e8331910178a981f950097587426877e1ea71f373fae8276964f.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/tables/2cf4ac761bd4e8331910178a981f950097587426877e1ea71f373fae8276964f.jpg)

![2f369946d8a10ff3b48d284bdf0c2d818229dc949a5fc64059b1b1cb1f9ed335.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/tables/2f369946d8a10ff3b48d284bdf0c2d818229dc949a5fc64059b1b1cb1f9ed335.jpg)

![339231b25d1b89594d9188567df7ef8d6b754db69421a47409e85c9b1414f81f.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/tables/339231b25d1b89594d9188567df7ef8d6b754db69421a47409e85c9b1414f81f.jpg)

![35276619895b9da7fe5f2158af1dd14faac877e1a1e527eee3e501f2c405e4bd.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/tables/35276619895b9da7fe5f2158af1dd14faac877e1a1e527eee3e501f2c405e4bd.jpg)

![3af934b7ca2ea206c9ac3b974b3911a41b3688ef8f731c054287cfcb95c649b0.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/tables/3af934b7ca2ea206c9ac3b974b3911a41b3688ef8f731c054287cfcb95c649b0.jpg)

![3eff803f34e2b3cb5b8ced50de8d1049051dc2ca7918286306dbf311d348ab21.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/tables/3eff803f34e2b3cb5b8ced50de8d1049051dc2ca7918286306dbf311d348ab21.jpg)

![42d532f5161eb33a1420663d6d7c28ad917f2220c22128c94df41430b9b09401.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/tables/42d532f5161eb33a1420663d6d7c28ad917f2220c22128c94df41430b9b09401.jpg)

![436190f629881a315082e9aa46d23cc01ba935864d6c89db1bcd61b178dbd91a.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/tables/436190f629881a315082e9aa46d23cc01ba935864d6c89db1bcd61b178dbd91a.jpg)

![5c57fdfbd36664af03937efd8a17da174df5917e91b595f722746f6931ce6929.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/tables/5c57fdfbd36664af03937efd8a17da174df5917e91b595f722746f6931ce6929.jpg)

![8885aa689c9c5337989d82e39942be42fac6310563c725825ad60189ef22ed1d.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/tables/8885aa689c9c5337989d82e39942be42fac6310563c725825ad60189ef22ed1d.jpg)

![97d2119a278c61b9641a9dee5a7e57d3773fece41b0ed2e44efa39326e249a5c.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/tables/97d2119a278c61b9641a9dee5a7e57d3773fece41b0ed2e44efa39326e249a5c.jpg)

![a21434c54bc3734eccc479a8d36cac68f2a616d2f952958f54ce75278d242e62.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/tables/a21434c54bc3734eccc479a8d36cac68f2a616d2f952958f54ce75278d242e62.jpg)

![a8bf576f04ee4f35901297572207ad5fee9dab3033f1920317650577f8a61e61.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/tables/a8bf576f04ee4f35901297572207ad5fee9dab3033f1920317650577f8a61e61.jpg)

![b490f95785fcd159a22b2e08b6feeb78c1e417351045c473c75dc44f3e336d7f.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/tables/b490f95785fcd159a22b2e08b6feeb78c1e417351045c473c75dc44f3e336d7f.jpg)

![bcf439e8d6f3a5ef76092dc718842407e76c470de25c55e609d392fbf4747fe8.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/tables/bcf439e8d6f3a5ef76092dc718842407e76c470de25c55e609d392fbf4747fe8.jpg)

![d09dc694e77769869d5c0e7a34c6a03d22bc47ac70e851594d79404f6b7a6712.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/tables/d09dc694e77769869d5c0e7a34c6a03d22bc47ac70e851594d79404f6b7a6712.jpg)

![d60410909cc1912e6b2525ae1838ff029f07d0dd8d279823cb40bcb79fe9c873.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/tables/d60410909cc1912e6b2525ae1838ff029f07d0dd8d279823cb40bcb79fe9c873.jpg)

![e2607da4298886eab2f5ae9b5ac4b85f277a16868fd4088132dbed779c0f514f.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/tables/e2607da4298886eab2f5ae9b5ac4b85f277a16868fd4088132dbed779c0f514f.jpg)

![ea4cc81d2ffb21f41f9ff496557b19ddea004a06aae41d92125baa73229f8c61.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/tables/ea4cc81d2ffb21f41f9ff496557b19ddea004a06aae41d92125baa73229f8c61.jpg)

![f37e8b3481c3e0242fd0730093fc1a7127fcd44ad30afa7affb3fab26d2f50f6.jpg](../iclr_results/682_Mitigating Object Hallucination in MLLMs via Data-augmented Phrase-level Alignment/tables/f37e8b3481c3e0242fd0730093fc1a7127fcd44ad30afa7affb3fab26d2f50f6.jpg)

## Simple, Good, Fast: Self-Supervised World Models Free of Baggage


### Images

![25db490ba106f5db05d72cb33ce0343bbc58ab7c4efc817fcd2ef634f5c2d572.jpg](../iclr_results/683_Simple, Good, Fast_ Self-Supervised World Models Free of Baggage/images/25db490ba106f5db05d72cb33ce0343bbc58ab7c4efc817fcd2ef634f5c2d572.jpg)

![2888bec7d413253851fde7b9a061500fb922b89fee9da869a06a4f776afb68e9.jpg](../iclr_results/683_Simple, Good, Fast_ Self-Supervised World Models Free of Baggage/images/2888bec7d413253851fde7b9a061500fb922b89fee9da869a06a4f776afb68e9.jpg)

![4a3204c321ccede35fb17e1ddd876a719a2dfc5d998c92a5da990412055ecab2.jpg](../iclr_results/683_Simple, Good, Fast_ Self-Supervised World Models Free of Baggage/images/4a3204c321ccede35fb17e1ddd876a719a2dfc5d998c92a5da990412055ecab2.jpg)

![4ad6f57c23329f94ee746b0c499b78161c5082c06dc4303072bc6835ce14e653.jpg](../iclr_results/683_Simple, Good, Fast_ Self-Supervised World Models Free of Baggage/images/4ad6f57c23329f94ee746b0c499b78161c5082c06dc4303072bc6835ce14e653.jpg)

![500aaba02caeb9ac96cae510164045c1d69a1bd51998a250f6bf0f8c920abb1e.jpg](../iclr_results/683_Simple, Good, Fast_ Self-Supervised World Models Free of Baggage/images/500aaba02caeb9ac96cae510164045c1d69a1bd51998a250f6bf0f8c920abb1e.jpg)

![50a15f5e000b5103e0cb10c50c3c429341b40cdb22209ba239aa9a8d1fa3198a.jpg](../iclr_results/683_Simple, Good, Fast_ Self-Supervised World Models Free of Baggage/images/50a15f5e000b5103e0cb10c50c3c429341b40cdb22209ba239aa9a8d1fa3198a.jpg)

![9dd24b0553fd88cc7009668cba6c3352e1a5d4536a3e8714e0c8eec5d9618ac1.jpg](../iclr_results/683_Simple, Good, Fast_ Self-Supervised World Models Free of Baggage/images/9dd24b0553fd88cc7009668cba6c3352e1a5d4536a3e8714e0c8eec5d9618ac1.jpg)

![affdae5769459d619fc45f8f43e7a84b81619372ef7deff641f175c1c7effb25.jpg](../iclr_results/683_Simple, Good, Fast_ Self-Supervised World Models Free of Baggage/images/affdae5769459d619fc45f8f43e7a84b81619372ef7deff641f175c1c7effb25.jpg)

![b0c9f9bccbe3dc7ab0647bfe7e5a9912d53a7008bad2e4acc46e56860bed6306.jpg](../iclr_results/683_Simple, Good, Fast_ Self-Supervised World Models Free of Baggage/images/b0c9f9bccbe3dc7ab0647bfe7e5a9912d53a7008bad2e4acc46e56860bed6306.jpg)

![db49e875d3d4c35c3ba6741aaebe9aeb4f58592761a56ffd749f7f491804faa9.jpg](../iclr_results/683_Simple, Good, Fast_ Self-Supervised World Models Free of Baggage/images/db49e875d3d4c35c3ba6741aaebe9aeb4f58592761a56ffd749f7f491804faa9.jpg)

### Tables

![05fd06b9e0221bcb92d34570aaeaf84270382b43649d18041f06689b9f69fde2.jpg](../iclr_results/683_Simple, Good, Fast_ Self-Supervised World Models Free of Baggage/tables/05fd06b9e0221bcb92d34570aaeaf84270382b43649d18041f06689b9f69fde2.jpg)

![401ede51d551050133f08b70137aa841870dde8f69275af998e9dd64a89fc814.jpg](../iclr_results/683_Simple, Good, Fast_ Self-Supervised World Models Free of Baggage/tables/401ede51d551050133f08b70137aa841870dde8f69275af998e9dd64a89fc814.jpg)

![7e4dec04b02acc9570af14b5ef80f017611edd1a4e6970d4a78b5d109618eaea.jpg](../iclr_results/683_Simple, Good, Fast_ Self-Supervised World Models Free of Baggage/tables/7e4dec04b02acc9570af14b5ef80f017611edd1a4e6970d4a78b5d109618eaea.jpg)

![946dcd97545d7bb339fe420411c18071dd29095e0caf9e8ff3b435db0595e12d.jpg](../iclr_results/683_Simple, Good, Fast_ Self-Supervised World Models Free of Baggage/tables/946dcd97545d7bb339fe420411c18071dd29095e0caf9e8ff3b435db0595e12d.jpg)

![aeee350141ce7660cef4a8cfa178d5eff0d818e86b8bcafb15ca3a6c8956a72d.jpg](../iclr_results/683_Simple, Good, Fast_ Self-Supervised World Models Free of Baggage/tables/aeee350141ce7660cef4a8cfa178d5eff0d818e86b8bcafb15ca3a6c8956a72d.jpg)

![e905bb92aada7a422f1fde4b8a071fd9e72f3d1e06223310e21734791a0ed5df.jpg](../iclr_results/683_Simple, Good, Fast_ Self-Supervised World Models Free of Baggage/tables/e905bb92aada7a422f1fde4b8a071fd9e72f3d1e06223310e21734791a0ed5df.jpg)

![ed89da6166fd0d86cc353c257bf3c2255e206a09adb9d87c2ed941a2384af6d4.jpg](../iclr_results/683_Simple, Good, Fast_ Self-Supervised World Models Free of Baggage/tables/ed89da6166fd0d86cc353c257bf3c2255e206a09adb9d87c2ed941a2384af6d4.jpg)

![fced2ca12c8338f457d9556e344a7eff8b28c6c3b7891cdf06f2a1788b5c294b.jpg](../iclr_results/683_Simple, Good, Fast_ Self-Supervised World Models Free of Baggage/tables/fced2ca12c8338f457d9556e344a7eff8b28c6c3b7891cdf06f2a1788b5c294b.jpg)

## MDSGen: Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Generation


### Images

![198e33a5cf29318fe9ffbe1e22e8699e7898297fb143eab66f6160b8f23797eb.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/images/198e33a5cf29318fe9ffbe1e22e8699e7898297fb143eab66f6160b8f23797eb.jpg)

![19df9e7e82a0e86de1548187e6535c5d7a196115970354027f38576cae18ecc9.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/images/19df9e7e82a0e86de1548187e6535c5d7a196115970354027f38576cae18ecc9.jpg)

![27550933b9700c9d7f67e1886acf1173bc0d4a3ae501c3c0b77c8bb188f93df7.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/images/27550933b9700c9d7f67e1886acf1173bc0d4a3ae501c3c0b77c8bb188f93df7.jpg)

![31229e84da449055c370fb1094d7204bd89a5b128bb62eea714dad153d4073a7.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/images/31229e84da449055c370fb1094d7204bd89a5b128bb62eea714dad153d4073a7.jpg)

![3f655759854dfb531337251bd152424c2f08496548dde4b0c151ac06ac8e2cb1.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/images/3f655759854dfb531337251bd152424c2f08496548dde4b0c151ac06ac8e2cb1.jpg)

![4efeb81728d9272a09d1836b84c02f6babcd2983a47ba7a132dd4699c7344366.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/images/4efeb81728d9272a09d1836b84c02f6babcd2983a47ba7a132dd4699c7344366.jpg)

![56a125a0132c7542e1193a4f0b53fef2748aa5bdda49b57c1143372192c72336.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/images/56a125a0132c7542e1193a4f0b53fef2748aa5bdda49b57c1143372192c72336.jpg)

![68f03ec969fc09b5f35176775df89dfc3bbcca08ce69770cbdbb713cef7f58cf.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/images/68f03ec969fc09b5f35176775df89dfc3bbcca08ce69770cbdbb713cef7f58cf.jpg)

![6fb8e86f79eb9736c57da6e8422993c6a78111f8fda2af29507d3ebb16498906.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/images/6fb8e86f79eb9736c57da6e8422993c6a78111f8fda2af29507d3ebb16498906.jpg)

![8156ca902818f9d02665d390834effc9f13cb5c5e5ed1df2d87d59036ac3744d.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/images/8156ca902818f9d02665d390834effc9f13cb5c5e5ed1df2d87d59036ac3744d.jpg)

![8a689cf1c9eefb333e9eb74373c24adcbfa371b47d7311e0dc437b15bc5dccd2.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/images/8a689cf1c9eefb333e9eb74373c24adcbfa371b47d7311e0dc437b15bc5dccd2.jpg)

![8ad8e6106945f3d846c5e4845f4506666a99dd393e6a020a758b6e68765bc0f6.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/images/8ad8e6106945f3d846c5e4845f4506666a99dd393e6a020a758b6e68765bc0f6.jpg)

![9823da8fb1636ca6d9030770ba11ceaf210a4ddf84861e4e62b042a76c147fd6.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/images/9823da8fb1636ca6d9030770ba11ceaf210a4ddf84861e4e62b042a76c147fd6.jpg)

![a70c7d7c7446a5afbfabb72667284358da2d80faded9aaa7b36e7737a10b68c0.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/images/a70c7d7c7446a5afbfabb72667284358da2d80faded9aaa7b36e7737a10b68c0.jpg)

![c1f1b6ea0225344303141c90cfa4314cb4c225f13a609778f2f18394b342872e.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/images/c1f1b6ea0225344303141c90cfa4314cb4c225f13a609778f2f18394b342872e.jpg)

![c6f0e3520535bbdc5a7f2c4aa09ecbefdae6ba477011ea0d4fc9710a558926a7.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/images/c6f0e3520535bbdc5a7f2c4aa09ecbefdae6ba477011ea0d4fc9710a558926a7.jpg)

![eed81e8e70679400144c30eb5905a4ed9d9d07858d691e3baf884533b94b56bd.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/images/eed81e8e70679400144c30eb5905a4ed9d9d07858d691e3baf884533b94b56bd.jpg)

![fc47e4d8f948dd337c42d5c8f7e0f356800159ab8bc90b52dfcca37d53512748.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/images/fc47e4d8f948dd337c42d5c8f7e0f356800159ab8bc90b52dfcca37d53512748.jpg)

### Tables

![0c0e5392bf6c5d0cd25dd72bb1eda3abea0b353932ca8e787d24ba6e7ca9ee0e.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/tables/0c0e5392bf6c5d0cd25dd72bb1eda3abea0b353932ca8e787d24ba6e7ca9ee0e.jpg)

![2f103894b9076e27c516a1e295a255daa908c59315e807e3063542a1eaafef27.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/tables/2f103894b9076e27c516a1e295a255daa908c59315e807e3063542a1eaafef27.jpg)

![38f93302455de48bead69c61d5d5c2e15bca98a2501365891e501afaac454117.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/tables/38f93302455de48bead69c61d5d5c2e15bca98a2501365891e501afaac454117.jpg)

![3916558005533d2a8204a32cfb526a14d03f4b2f0bee4a0c25733b6ac272247a.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/tables/3916558005533d2a8204a32cfb526a14d03f4b2f0bee4a0c25733b6ac272247a.jpg)

![49a83e895b9e95ed839057015419fd045e4c16f65d5ede4093a5976c708d3302.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/tables/49a83e895b9e95ed839057015419fd045e4c16f65d5ede4093a5976c708d3302.jpg)

![4c372f42986a371ae1b46de7b6634edf4ed8a6509d6624a365d20a0d1ec6a291.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/tables/4c372f42986a371ae1b46de7b6634edf4ed8a6509d6624a365d20a0d1ec6a291.jpg)

![5f1e3986a4b254db3ade3a89d82bb3abc8b4c604d7e19c898b13de59d55fd1da.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/tables/5f1e3986a4b254db3ade3a89d82bb3abc8b4c604d7e19c898b13de59d55fd1da.jpg)

![675b80272358ccec641ba4ea72560280ef15d68f6f85b8808dafafdf48861375.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/tables/675b80272358ccec641ba4ea72560280ef15d68f6f85b8808dafafdf48861375.jpg)

![6fbf5500660f81de0a1be0e9caa884599bdb8c0f520670db701c44b20b65e5e7.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/tables/6fbf5500660f81de0a1be0e9caa884599bdb8c0f520670db701c44b20b65e5e7.jpg)

![8bd13a9f0fb0eb0d2b0e1f83a9b22fdf5ac897056bd15c7eaf130caaf76ac898.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/tables/8bd13a9f0fb0eb0d2b0e1f83a9b22fdf5ac897056bd15c7eaf130caaf76ac898.jpg)

![a43c10ef37baef48de208788aea02df630f387c598eb205e905bbd9d904e9610.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/tables/a43c10ef37baef48de208788aea02df630f387c598eb205e905bbd9d904e9610.jpg)

![ba0882dda91cee4e677a86c4ba2cc654839564bb9ac930fc74e8975d466e99ef.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/tables/ba0882dda91cee4e677a86c4ba2cc654839564bb9ac930fc74e8975d466e99ef.jpg)

![ca537cdae3412c248f7c9a5bf5ce3dbfc56d21df9aee9fe035726e120ec5631e.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/tables/ca537cdae3412c248f7c9a5bf5ce3dbfc56d21df9aee9fe035726e120ec5631e.jpg)

![cf66fbd12e28ae51893e74241bad95da060e0f295a2ae252c09b9fa67843c28c.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/tables/cf66fbd12e28ae51893e74241bad95da060e0f295a2ae252c09b9fa67843c28c.jpg)

![cfa5a281ada9d5fadf413d4913899542cb748c2c880279634a5fd6397b9f5007.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/tables/cfa5a281ada9d5fadf413d4913899542cb748c2c880279634a5fd6397b9f5007.jpg)

![ea0d05652c610cbfa3aaf5933b3d9b83d57c4feb4045a5a334228d5b1b224d0c.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/tables/ea0d05652c610cbfa3aaf5933b3d9b83d57c4feb4045a5a334228d5b1b224d0c.jpg)

![fb7e755dd487f9ba3e56b4cf65934c899ee736e6d69bd5ee9b319ba3c074ed13.jpg](../iclr_results/684_MDSGen_ Fast and Efficient Masked Diffusion Temporal-Aware Transformers for Open-Domain Sound Genera/tables/fb7e755dd487f9ba3e56b4cf65934c899ee736e6d69bd5ee9b319ba3c074ed13.jpg)

## UniCO: On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP


### Images

![0866fd74786a17ae943e67cdc3afcebbc4523a20b70ee5e30ebcb44cbf08dabb.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/images/0866fd74786a17ae943e67cdc3afcebbc4523a20b70ee5e30ebcb44cbf08dabb.jpg)

![5ebb2ab579fd65d6ea79af94d9cd0767ca1ae922f662b9e0edb2d1f64fe6a384.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/images/5ebb2ab579fd65d6ea79af94d9cd0767ca1ae922f662b9e0edb2d1f64fe6a384.jpg)

![6ff3bd43b062983102b9be4b1d8f516ecc3f5e9590774603859c8586f7a12fbf.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/images/6ff3bd43b062983102b9be4b1d8f516ecc3f5e9590774603859c8586f7a12fbf.jpg)

![845f2a32df8d474be35c9cc9717ee5354224a899b930582215756f3a9e324554.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/images/845f2a32df8d474be35c9cc9717ee5354224a899b930582215756f3a9e324554.jpg)

![848ef5ba0bb52eb7de5a59ecc5d1218ea94496064d36e1bb16e3baa5083da196.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/images/848ef5ba0bb52eb7de5a59ecc5d1218ea94496064d36e1bb16e3baa5083da196.jpg)

![d88f8ba23160e935a70f4f235003d19ad1f0dd9ebaaf288c0f1e962773b24b7e.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/images/d88f8ba23160e935a70f4f235003d19ad1f0dd9ebaaf288c0f1e962773b24b7e.jpg)

![f50443beba9f2422940ef3c043ddf0361876a2d482c27f787669ce017d240f4b.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/images/f50443beba9f2422940ef3c043ddf0361876a2d482c27f787669ce017d240f4b.jpg)

![fd585089039f36b44ffb541f085a17b715867f1928a9a3490c51a95ae0f10806.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/images/fd585089039f36b44ffb541f085a17b715867f1928a9a3490c51a95ae0f10806.jpg)

### Tables

![1d7a63c0d76342a9cb15c74d9eb263de3b36eac95978e37d2a1346eb9bd06499.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/tables/1d7a63c0d76342a9cb15c74d9eb263de3b36eac95978e37d2a1346eb9bd06499.jpg)

![2115eaa2cb26ffad8bdf529c5aa829361c467ff8b3c3420c0e93a15d320e442e.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/tables/2115eaa2cb26ffad8bdf529c5aa829361c467ff8b3c3420c0e93a15d320e442e.jpg)

![29dec199eb586d39a08ef4be0467d69ddfc053ce682b3b03a3aa0facadade79c.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/tables/29dec199eb586d39a08ef4be0467d69ddfc053ce682b3b03a3aa0facadade79c.jpg)

![36809c5cced6000b3cf2b05d0eff08bdf88941a3c85e40b87cdb65b221294ef3.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/tables/36809c5cced6000b3cf2b05d0eff08bdf88941a3c85e40b87cdb65b221294ef3.jpg)

![3e47252a35fe9bd2040a45bffc58c6f610bedeaa2ab101f710fbf3d410c3957c.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/tables/3e47252a35fe9bd2040a45bffc58c6f610bedeaa2ab101f710fbf3d410c3957c.jpg)

![5b6163521c07b497e04c41c3986063ba384513fdcf78a2b35d8b10cd993d5cc8.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/tables/5b6163521c07b497e04c41c3986063ba384513fdcf78a2b35d8b10cd993d5cc8.jpg)

![63463e88537aa013b6bc259a82154c1f8da488bec35d376240299204cb38a3c3.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/tables/63463e88537aa013b6bc259a82154c1f8da488bec35d376240299204cb38a3c3.jpg)

![78cbbeed296184e50826665dc9dcfaf5ed7329e5c79feee92e66abb2915a7332.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/tables/78cbbeed296184e50826665dc9dcfaf5ed7329e5c79feee92e66abb2915a7332.jpg)

![8071e887e7b51e4dc45463aaec9a6a7e00dd2efd956a0b33e6ba91d8230daf55.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/tables/8071e887e7b51e4dc45463aaec9a6a7e00dd2efd956a0b33e6ba91d8230daf55.jpg)

![815d2686773d2611a1c7de7dfa4926d1b8edb5db6fabd1ebf56fe961913d9f7a.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/tables/815d2686773d2611a1c7de7dfa4926d1b8edb5db6fabd1ebf56fe961913d9f7a.jpg)

![8d5a90521ad2fb2ef60c359ca92c2cd69586e5eee11141eb763610505ad937eb.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/tables/8d5a90521ad2fb2ef60c359ca92c2cd69586e5eee11141eb763610505ad937eb.jpg)

![9606ab62f32baba1279ea8f05448a007239c2f9f0d5089ff0f916a981dd31766.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/tables/9606ab62f32baba1279ea8f05448a007239c2f9f0d5089ff0f916a981dd31766.jpg)

![a0124c7589fd03cc4cb75bc102975499e88c41e9da32b4671a9f5b55495b82d2.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/tables/a0124c7589fd03cc4cb75bc102975499e88c41e9da32b4671a9f5b55495b82d2.jpg)

![a10511173c9d1dc2d292f8409a1b0aadd5a1748be166ac3a6b9aef19e90fac04.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/tables/a10511173c9d1dc2d292f8409a1b0aadd5a1748be166ac3a6b9aef19e90fac04.jpg)

![acf9ce61fd6ee21bbea835a3035c06b56739b6ec428ef8881f001cc93d8b890e.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/tables/acf9ce61fd6ee21bbea835a3035c06b56739b6ec428ef8881f001cc93d8b890e.jpg)

![d208427d5924ff59d74627c7e87d1ae92684423e597f3a6401aa01a48f0b1110.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/tables/d208427d5924ff59d74627c7e87d1ae92684423e597f3a6401aa01a48f0b1110.jpg)

![d38042c8c86bf02ec32ac96b2115a5fb0dc8175938bfe98455d98d2f96ae0f00.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/tables/d38042c8c86bf02ec32ac96b2115a5fb0dc8175938bfe98455d98d2f96ae0f00.jpg)

![d85b90ee981965074afc612e1170ed99f3302f6cb5927bb21395c13b6b549ce4.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/tables/d85b90ee981965074afc612e1170ed99f3302f6cb5927bb21395c13b6b549ce4.jpg)

![dc16feb9b084b6287f3eb81b0a673cee92772f99c9e77b1ec89e9bd875fd4b27.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/tables/dc16feb9b084b6287f3eb81b0a673cee92772f99c9e77b1ec89e9bd875fd4b27.jpg)

![e3706ae020ca3a175f57402e5d39ede5732aea51942937f12afcbd3baae53562.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/tables/e3706ae020ca3a175f57402e5d39ede5732aea51942937f12afcbd3baae53562.jpg)

![effb67f2610dfad5fd2c17468441c6192f4265d545951e0bb6444039f5732176.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/tables/effb67f2610dfad5fd2c17468441c6192f4265d545951e0bb6444039f5732176.jpg)

![f96f0cb6052ea16a18d6ea2609c36b980f7b49dcc712490e7dddbb83ae95dbec.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/tables/f96f0cb6052ea16a18d6ea2609c36b980f7b49dcc712490e7dddbb83ae95dbec.jpg)

![f9763e58c828c811dcaee5866203e6e5f89d7986ca812ad3d3ad6219a842a339.jpg](../iclr_results/685_UniCO_ On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP/tables/f9763e58c828c811dcaee5866203e6e5f89d7986ca812ad3d3ad6219a842a339.jpg)

## Grammar Reinforcement Learning: path and cycle counting in graphs with a Context-Free Grammar and Transformer approach


### Images

![12fbbf32022ce9966cad2234e1c6e236ec8d2461d72eb5296fffb18450c406b5.jpg](../iclr_results/686_Grammar Reinforcement Learning_ path and cycle counting in graphs with a Context-Free Grammar and Tr/images/12fbbf32022ce9966cad2234e1c6e236ec8d2461d72eb5296fffb18450c406b5.jpg)

![2fb23260c14c298c35f29a223dc7e5812fc37c463ba3697bf9a9543d6bb8966e.jpg](../iclr_results/686_Grammar Reinforcement Learning_ path and cycle counting in graphs with a Context-Free Grammar and Tr/images/2fb23260c14c298c35f29a223dc7e5812fc37c463ba3697bf9a9543d6bb8966e.jpg)

![346facbaa03091d60117b372ba618279c1a4757655f69dd148fc71cd72e5cc5c.jpg](../iclr_results/686_Grammar Reinforcement Learning_ path and cycle counting in graphs with a Context-Free Grammar and Tr/images/346facbaa03091d60117b372ba618279c1a4757655f69dd148fc71cd72e5cc5c.jpg)

![5638ee712d299ddbb98a9c5d17573254dccfbdf889d1bca90cca56e9cc428a78.jpg](../iclr_results/686_Grammar Reinforcement Learning_ path and cycle counting in graphs with a Context-Free Grammar and Tr/images/5638ee712d299ddbb98a9c5d17573254dccfbdf889d1bca90cca56e9cc428a78.jpg)

![5cbfa6fcdc861c22b6a07654987037f0a7d169be48214cda86c6b65d66005b46.jpg](../iclr_results/686_Grammar Reinforcement Learning_ path and cycle counting in graphs with a Context-Free Grammar and Tr/images/5cbfa6fcdc861c22b6a07654987037f0a7d169be48214cda86c6b65d66005b46.jpg)

![732b6dcbbb51929b8f687445a78acc81686b8f557484958b3db63f486acc8def.jpg](../iclr_results/686_Grammar Reinforcement Learning_ path and cycle counting in graphs with a Context-Free Grammar and Tr/images/732b6dcbbb51929b8f687445a78acc81686b8f557484958b3db63f486acc8def.jpg)

![7d6ea687ca627b6930eedeea4f442fc0062cc0afae95170c9195a860f303c5cd.jpg](../iclr_results/686_Grammar Reinforcement Learning_ path and cycle counting in graphs with a Context-Free Grammar and Tr/images/7d6ea687ca627b6930eedeea4f442fc0062cc0afae95170c9195a860f303c5cd.jpg)

![b7451691333f75dea7abbd265c77b721cfa4f848fba2882cdccd990c2bb2ecf6.jpg](../iclr_results/686_Grammar Reinforcement Learning_ path and cycle counting in graphs with a Context-Free Grammar and Tr/images/b7451691333f75dea7abbd265c77b721cfa4f848fba2882cdccd990c2bb2ecf6.jpg)

![d3b2578a2c87f9228df56f152b104661272a21867397bfad307d204a929ad726.jpg](../iclr_results/686_Grammar Reinforcement Learning_ path and cycle counting in graphs with a Context-Free Grammar and Tr/images/d3b2578a2c87f9228df56f152b104661272a21867397bfad307d204a929ad726.jpg)

![e4d4cd72a6e52383a7ceb11adb2df510d2ca709859e75890dafc3dfda6bac459.jpg](../iclr_results/686_Grammar Reinforcement Learning_ path and cycle counting in graphs with a Context-Free Grammar and Tr/images/e4d4cd72a6e52383a7ceb11adb2df510d2ca709859e75890dafc3dfda6bac459.jpg)

![e87f38842e0d09f67df5cc24b9de2812c154d4a4e874796136296ac15b983466.jpg](../iclr_results/686_Grammar Reinforcement Learning_ path and cycle counting in graphs with a Context-Free Grammar and Tr/images/e87f38842e0d09f67df5cc24b9de2812c154d4a4e874796136296ac15b983466.jpg)

![ee1a0337213cbf9c6fcf987fcb6439fcc5507c7b750485c30f1f31c8213c4990.jpg](../iclr_results/686_Grammar Reinforcement Learning_ path and cycle counting in graphs with a Context-Free Grammar and Tr/images/ee1a0337213cbf9c6fcf987fcb6439fcc5507c7b750485c30f1f31c8213c4990.jpg)

![f4882e6e0fe8613257df72cd4be8e0f1136109cbedbd6dc1aac3e9356dcd4092.jpg](../iclr_results/686_Grammar Reinforcement Learning_ path and cycle counting in graphs with a Context-Free Grammar and Tr/images/f4882e6e0fe8613257df72cd4be8e0f1136109cbedbd6dc1aac3e9356dcd4092.jpg)

![feff55eb9edbf3111dcfb79698ed770461e871897f303ec541d7c4223fe21557.jpg](../iclr_results/686_Grammar Reinforcement Learning_ path and cycle counting in graphs with a Context-Free Grammar and Tr/images/feff55eb9edbf3111dcfb79698ed770461e871897f303ec541d7c4223fe21557.jpg)

## GPromptShield: Elevating Resilience in Graph Prompt Tuning Against Adversarial Attacks


### Images

![32faa2a1e2585541db6571a6059ddc6fdcf1a7fb4da3c7475de224b7ca5a7dba.jpg](../iclr_results/687_GPromptShield_ Elevating Resilience in Graph Prompt Tuning Against Adversarial Attacks/images/32faa2a1e2585541db6571a6059ddc6fdcf1a7fb4da3c7475de224b7ca5a7dba.jpg)

![5d7883188e20850685b9bf934013e4202d4bd3bebe277d40906d63c7b612a341.jpg](../iclr_results/687_GPromptShield_ Elevating Resilience in Graph Prompt Tuning Against Adversarial Attacks/images/5d7883188e20850685b9bf934013e4202d4bd3bebe277d40906d63c7b612a341.jpg)

![bb7addbaca59170ab0d25d6ef8a831a629dd76d391cbd435ce7c14f1ddd8cb83.jpg](../iclr_results/687_GPromptShield_ Elevating Resilience in Graph Prompt Tuning Against Adversarial Attacks/images/bb7addbaca59170ab0d25d6ef8a831a629dd76d391cbd435ce7c14f1ddd8cb83.jpg)

![cee9d35a60e2e74d82e340192a964cb31a5002bd8b2eec3f33997cec8304ffd3.jpg](../iclr_results/687_GPromptShield_ Elevating Resilience in Graph Prompt Tuning Against Adversarial Attacks/images/cee9d35a60e2e74d82e340192a964cb31a5002bd8b2eec3f33997cec8304ffd3.jpg)

### Tables

![0435c766af5088cf01f76f5e832305313788db053bded783e5001bcf8b4efab3.jpg](../iclr_results/687_GPromptShield_ Elevating Resilience in Graph Prompt Tuning Against Adversarial Attacks/tables/0435c766af5088cf01f76f5e832305313788db053bded783e5001bcf8b4efab3.jpg)

![256e5fe14f5b15fd7107b550f271c13521cabebcde725c449e85a4ae58380eba.jpg](../iclr_results/687_GPromptShield_ Elevating Resilience in Graph Prompt Tuning Against Adversarial Attacks/tables/256e5fe14f5b15fd7107b550f271c13521cabebcde725c449e85a4ae58380eba.jpg)

![4b55074c1ad1f290491b3724956090528fc9e751dd86d32d989576b40161e5ac.jpg](../iclr_results/687_GPromptShield_ Elevating Resilience in Graph Prompt Tuning Against Adversarial Attacks/tables/4b55074c1ad1f290491b3724956090528fc9e751dd86d32d989576b40161e5ac.jpg)

![5441c62bf31ec56be5a63a32074dde9462642b9b07417cef5dce51279c4b96ba.jpg](../iclr_results/687_GPromptShield_ Elevating Resilience in Graph Prompt Tuning Against Adversarial Attacks/tables/5441c62bf31ec56be5a63a32074dde9462642b9b07417cef5dce51279c4b96ba.jpg)

![6430c965b95aac841bfcb6d584273412cba609bc941b6c4846c8f8d711cf5f54.jpg](../iclr_results/687_GPromptShield_ Elevating Resilience in Graph Prompt Tuning Against Adversarial Attacks/tables/6430c965b95aac841bfcb6d584273412cba609bc941b6c4846c8f8d711cf5f54.jpg)

![6cd8a8673a83ab3b215bce2a30f48b963f0c70039793f563432fe41bf5129abe.jpg](../iclr_results/687_GPromptShield_ Elevating Resilience in Graph Prompt Tuning Against Adversarial Attacks/tables/6cd8a8673a83ab3b215bce2a30f48b963f0c70039793f563432fe41bf5129abe.jpg)

![73501e40600788852fc9d71160dd1b9f24ab1d0181e3be934fecad86ae6dcbb0.jpg](../iclr_results/687_GPromptShield_ Elevating Resilience in Graph Prompt Tuning Against Adversarial Attacks/tables/73501e40600788852fc9d71160dd1b9f24ab1d0181e3be934fecad86ae6dcbb0.jpg)

![833d0d58fada41710049c92d00176614d1f5dd3b8fca48c814b122d873bb477b.jpg](../iclr_results/687_GPromptShield_ Elevating Resilience in Graph Prompt Tuning Against Adversarial Attacks/tables/833d0d58fada41710049c92d00176614d1f5dd3b8fca48c814b122d873bb477b.jpg)

![8b2c8d7d867970169124050d5953fffe7ec002b72cca640fae146be3c12978f1.jpg](../iclr_results/687_GPromptShield_ Elevating Resilience in Graph Prompt Tuning Against Adversarial Attacks/tables/8b2c8d7d867970169124050d5953fffe7ec002b72cca640fae146be3c12978f1.jpg)

![8c5168954f68f3ab734b8116f240510199423c8cac391a3f8e3c61aff34854c6.jpg](../iclr_results/687_GPromptShield_ Elevating Resilience in Graph Prompt Tuning Against Adversarial Attacks/tables/8c5168954f68f3ab734b8116f240510199423c8cac391a3f8e3c61aff34854c6.jpg)

![df6e3e178a33a182482aac6cf13fae83a6acc2d9f6d3d457f176ecb14d291ffd.jpg](../iclr_results/687_GPromptShield_ Elevating Resilience in Graph Prompt Tuning Against Adversarial Attacks/tables/df6e3e178a33a182482aac6cf13fae83a6acc2d9f6d3d457f176ecb14d291ffd.jpg)

![e7015de7ecf34d0b8fbd595cb9d20f4758b5afdeaac3674316b0c1d1f9996cb4.jpg](../iclr_results/687_GPromptShield_ Elevating Resilience in Graph Prompt Tuning Against Adversarial Attacks/tables/e7015de7ecf34d0b8fbd595cb9d20f4758b5afdeaac3674316b0c1d1f9996cb4.jpg)

## WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling


### Images

![50b60cac2b82490c526e0f0f89cb9b29017bc50951c675f09d036f46e6adaab6.jpg](../iclr_results/688_WavTokenizer_ an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling/images/50b60cac2b82490c526e0f0f89cb9b29017bc50951c675f09d036f46e6adaab6.jpg)

![9961cf4ddad8cb7872b72c47e2940276ec1ab58d3920091565c4ea8c41be5c4e.jpg](../iclr_results/688_WavTokenizer_ an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling/images/9961cf4ddad8cb7872b72c47e2940276ec1ab58d3920091565c4ea8c41be5c4e.jpg)

### Tables

![1e3cef68f9b8e8607132e807ab4d3dd34958899c0891f59123b0792d87d2e559.jpg](../iclr_results/688_WavTokenizer_ an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling/tables/1e3cef68f9b8e8607132e807ab4d3dd34958899c0891f59123b0792d87d2e559.jpg)

![37fa952978b418ff822ca8123f7cb880ca879e063a26c603580354e879690432.jpg](../iclr_results/688_WavTokenizer_ an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling/tables/37fa952978b418ff822ca8123f7cb880ca879e063a26c603580354e879690432.jpg)

![3b0863d2a164bac150875ecf44a07fa714cd4b7efa94dbd6a55327349843aec9.jpg](../iclr_results/688_WavTokenizer_ an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling/tables/3b0863d2a164bac150875ecf44a07fa714cd4b7efa94dbd6a55327349843aec9.jpg)

![3fefe247c0f3e828af9415cb066fb9de9fe1fa51f5acb4c88bed3b075810fefc.jpg](../iclr_results/688_WavTokenizer_ an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling/tables/3fefe247c0f3e828af9415cb066fb9de9fe1fa51f5acb4c88bed3b075810fefc.jpg)

![5f1378b819101267a706ed0d87a565b70f8150a47355f436615f6f1b5572577f.jpg](../iclr_results/688_WavTokenizer_ an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling/tables/5f1378b819101267a706ed0d87a565b70f8150a47355f436615f6f1b5572577f.jpg)

![7c5137853b46faa8c85adf04061745f7eaff99a72f5f780548fc635d0cde0191.jpg](../iclr_results/688_WavTokenizer_ an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling/tables/7c5137853b46faa8c85adf04061745f7eaff99a72f5f780548fc635d0cde0191.jpg)

![811e396ec47cdfe4aafa8d8d6a9638b34bd140c675bbaebbb15457d8fdb62b5b.jpg](../iclr_results/688_WavTokenizer_ an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling/tables/811e396ec47cdfe4aafa8d8d6a9638b34bd140c675bbaebbb15457d8fdb62b5b.jpg)

![83743465e4f5155bb5b42307028beda96d45d13a9960fa9cf1921e5b7d05cd4d.jpg](../iclr_results/688_WavTokenizer_ an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling/tables/83743465e4f5155bb5b42307028beda96d45d13a9960fa9cf1921e5b7d05cd4d.jpg)

![c3ae350ac769da133292315e3c8768d796531c4400fe6d9972d0455abed5efc3.jpg](../iclr_results/688_WavTokenizer_ an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling/tables/c3ae350ac769da133292315e3c8768d796531c4400fe6d9972d0455abed5efc3.jpg)

![c545285dc53e84479f0487f78210003fff9396694cb8e1957afe42ead204143f.jpg](../iclr_results/688_WavTokenizer_ an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling/tables/c545285dc53e84479f0487f78210003fff9396694cb8e1957afe42ead204143f.jpg)

![edf7d1adb309d108ed0167739d00759661478871ce62060846f5845361434c7e.jpg](../iclr_results/688_WavTokenizer_ an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling/tables/edf7d1adb309d108ed0167739d00759661478871ce62060846f5845361434c7e.jpg)

![ef1037df09c6356803e752646ae7ea6c940bf542691babc92492276d173b8797.jpg](../iclr_results/688_WavTokenizer_ an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling/tables/ef1037df09c6356803e752646ae7ea6c940bf542691babc92492276d173b8797.jpg)

## RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation


### Images

![1082047c07e529d641805e5a3d7977e4c88f83c8a1bc55c5a544b5c42d37daf2.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/images/1082047c07e529d641805e5a3d7977e4c88f83c8a1bc55c5a544b5c42d37daf2.jpg)

![2646fccab77827a3170820c07cadb2af82d61769cc587d71a2628172a7823c67.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/images/2646fccab77827a3170820c07cadb2af82d61769cc587d71a2628172a7823c67.jpg)

![3945ea940b525bf043586aaa02077b07de6b7eaa4d43d64f74fac975c9658c5b.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/images/3945ea940b525bf043586aaa02077b07de6b7eaa4d43d64f74fac975c9658c5b.jpg)

![442ffd6070d6af931a35d1ec42a0313b2ec7b57304c8b71318402d52b4de32e5.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/images/442ffd6070d6af931a35d1ec42a0313b2ec7b57304c8b71318402d52b4de32e5.jpg)

![45f391b098c9f6b75fc4cb3b1b19848a2df41c05925bd357f75c0dc1809d753b.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/images/45f391b098c9f6b75fc4cb3b1b19848a2df41c05925bd357f75c0dc1809d753b.jpg)

![5c37f0a5e24809cf0b00767a61e949ed8f637b0e847dd9d715fd6f14ec0d9fcc.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/images/5c37f0a5e24809cf0b00767a61e949ed8f637b0e847dd9d715fd6f14ec0d9fcc.jpg)

![63f3602aa26596127add12e7d9b0037659bc2a333269711e6e2b5afbdd6f6036.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/images/63f3602aa26596127add12e7d9b0037659bc2a333269711e6e2b5afbdd6f6036.jpg)

![86dd4eb06a85bff1a19604fbc03bd2e1ca0599525f6ce67199a446359912966c.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/images/86dd4eb06a85bff1a19604fbc03bd2e1ca0599525f6ce67199a446359912966c.jpg)

![8ef98f7f5c601bd2e11df5eb393e599776507e5b06f2a35e0175ded7da8bb97c.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/images/8ef98f7f5c601bd2e11df5eb393e599776507e5b06f2a35e0175ded7da8bb97c.jpg)

![95aaa0d4b62aac02f24cd919ad7054eec3ae838db85bceaa21f152fd62b0583c.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/images/95aaa0d4b62aac02f24cd919ad7054eec3ae838db85bceaa21f152fd62b0583c.jpg)

![95acf3f430a0d0afa9826fd19f454b84e9822538ab7eed7b5cc8ea8aa0846639.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/images/95acf3f430a0d0afa9826fd19f454b84e9822538ab7eed7b5cc8ea8aa0846639.jpg)

![b8bf77b8409af0558d67972a1129d4d76091144ca309dabb858735798a713099.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/images/b8bf77b8409af0558d67972a1129d4d76091144ca309dabb858735798a713099.jpg)

![d0198f878e6347114afa213924182c7be1567893fe6725ff4ab1ca0f32c0b5c0.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/images/d0198f878e6347114afa213924182c7be1567893fe6725ff4ab1ca0f32c0b5c0.jpg)

![d9225d6f84c51b5214315d3b826ec7e261771603a3cc134d5c7bc59336c6656c.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/images/d9225d6f84c51b5214315d3b826ec7e261771603a3cc134d5c7bc59336c6656c.jpg)

![f19582c4cabec3b71e1adb71c2bce570d8e917bd6a1adb55e0112bae49585eca.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/images/f19582c4cabec3b71e1adb71c2bce570d8e917bd6a1adb55e0112bae49585eca.jpg)

### Tables

![0b1d8def243a5c452f2f7a4dbf575b2991e8f4a2ae581efa1472647dc6697bc2.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/tables/0b1d8def243a5c452f2f7a4dbf575b2991e8f4a2ae581efa1472647dc6697bc2.jpg)

![1fd852f34a5ee88f1a6c9bdc91b3f94c45df1d66e521d7803bfb9afe33948fd9.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/tables/1fd852f34a5ee88f1a6c9bdc91b3f94c45df1d66e521d7803bfb9afe33948fd9.jpg)

![376a55d78f25787fd01b2cefddb1e5bb16e8f4f0d00bc8f3f60bbb671a720055.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/tables/376a55d78f25787fd01b2cefddb1e5bb16e8f4f0d00bc8f3f60bbb671a720055.jpg)

![38823227719029254844c05af70ff7c47e6b84422659dde8468383903821318a.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/tables/38823227719029254844c05af70ff7c47e6b84422659dde8468383903821318a.jpg)

![433cd8e64b9302c035a95c741c30a10fa23830e1a7fa89fe05479a2636b4481d.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/tables/433cd8e64b9302c035a95c741c30a10fa23830e1a7fa89fe05479a2636b4481d.jpg)

![5465d30049e5c7f2fd4ac1ea0b736b9dcc3a4d9e046385a10e6e33d6e006e3de.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/tables/5465d30049e5c7f2fd4ac1ea0b736b9dcc3a4d9e046385a10e6e33d6e006e3de.jpg)

![5cacaac6101a76c2854125811e693051e69c4f3baa5fde9a4f5be7aa8343d466.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/tables/5cacaac6101a76c2854125811e693051e69c4f3baa5fde9a4f5be7aa8343d466.jpg)

![7925caff0a1cea5d30287c83b8af9ea258c0861d2b2e815854e6996e60370c67.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/tables/7925caff0a1cea5d30287c83b8af9ea258c0861d2b2e815854e6996e60370c67.jpg)

![9196f6e1469ae35500eddbff8837abc8f375ead735fb7eaffa02f4a54ead33d0.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/tables/9196f6e1469ae35500eddbff8837abc8f375ead735fb7eaffa02f4a54ead33d0.jpg)

![b2b969d7ddaec18798881e363d81ee5518be9e6437fc1ba1d8c6c6b2d26020a1.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/tables/b2b969d7ddaec18798881e363d81ee5518be9e6437fc1ba1d8c6c6b2d26020a1.jpg)

![d610970d4790566b4ad176b725d5ac0d12d6593396ff068a20fc9427958673d5.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/tables/d610970d4790566b4ad176b725d5ac0d12d6593396ff068a20fc9427958673d5.jpg)

![dc45b71ee7d65620beffe5fe434fa575f294956f3a8394b5bc953479974907e3.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/tables/dc45b71ee7d65620beffe5fe434fa575f294956f3a8394b5bc953479974907e3.jpg)

![f11fca94e1bdfc3d53d77f6ace325cadde99bd2f27e3a9a39034792d82d02c4c.jpg](../iclr_results/689_RDT-1B_ a Diffusion Foundation Model for Bimanual Manipulation/tables/f11fca94e1bdfc3d53d77f6ace325cadde99bd2f27e3a9a39034792d82d02c4c.jpg)

## Quality over Quantity in Attention Layers: When Adding More Heads Hurts


### Images

![50d714fb78dcca6cded8dac1d35a79dbd14204d39e9890ffd1a360266fd33959.jpg](../iclr_results/690_Quality over Quantity in Attention Layers_ When Adding More Heads Hurts/images/50d714fb78dcca6cded8dac1d35a79dbd14204d39e9890ffd1a360266fd33959.jpg)

![57e34a1a8c4befc9999dec0c1a79b994fdc92f6fbf16c5d164c7244dbb662f8b.jpg](../iclr_results/690_Quality over Quantity in Attention Layers_ When Adding More Heads Hurts/images/57e34a1a8c4befc9999dec0c1a79b994fdc92f6fbf16c5d164c7244dbb662f8b.jpg)

![5bd6c38c5ecd362dfdea8f3faf037c58b78442005c2b51a01981b6b1ed2052a2.jpg](../iclr_results/690_Quality over Quantity in Attention Layers_ When Adding More Heads Hurts/images/5bd6c38c5ecd362dfdea8f3faf037c58b78442005c2b51a01981b6b1ed2052a2.jpg)

![ae5069f7bab2a509271b7cc536f2340d86bf3c05174315d797494c0510781891.jpg](../iclr_results/690_Quality over Quantity in Attention Layers_ When Adding More Heads Hurts/images/ae5069f7bab2a509271b7cc536f2340d86bf3c05174315d797494c0510781891.jpg)

![b2627d09d35a95132d2c849cd1d9a32e7649c7d546fa9cd8bb853c52c5c1700a.jpg](../iclr_results/690_Quality over Quantity in Attention Layers_ When Adding More Heads Hurts/images/b2627d09d35a95132d2c849cd1d9a32e7649c7d546fa9cd8bb853c52c5c1700a.jpg)

![b955ba01f9371690dd419c5bd897730f8d2454d23d8251422ddb382bd68ef58b.jpg](../iclr_results/690_Quality over Quantity in Attention Layers_ When Adding More Heads Hurts/images/b955ba01f9371690dd419c5bd897730f8d2454d23d8251422ddb382bd68ef58b.jpg)

![bc6bd55f72a9309dd5cc551e43c7f0d12cee052ff8b0c12dbd6ad41cd3ce7218.jpg](../iclr_results/690_Quality over Quantity in Attention Layers_ When Adding More Heads Hurts/images/bc6bd55f72a9309dd5cc551e43c7f0d12cee052ff8b0c12dbd6ad41cd3ce7218.jpg)

![da2fda897851a273d8ab147fd6ec846bb3bf6af1dcc351aa98060aea75b6a65d.jpg](../iclr_results/690_Quality over Quantity in Attention Layers_ When Adding More Heads Hurts/images/da2fda897851a273d8ab147fd6ec846bb3bf6af1dcc351aa98060aea75b6a65d.jpg)

### Tables

![8e78632e63f7976bccca06e19b29dae276faefaaf9df10d469c594bb4f004cc6.jpg](../iclr_results/690_Quality over Quantity in Attention Layers_ When Adding More Heads Hurts/tables/8e78632e63f7976bccca06e19b29dae276faefaaf9df10d469c594bb4f004cc6.jpg)

## Language Agents Meet Causality -- Bridging LLMs and Causal World Models

### Images

![040420e17c3fe972176e09d2fad2146e401723d0555ee219d2006b149c861369.jpg](../iclr_results/691_Language Agents Meet Causality -- Bridging LLMs and Causal World Models/images/040420e17c3fe972176e09d2fad2146e401723d0555ee219d2006b149c861369.jpg)

![6646228f3cc606f82f5c678a4c248b7a81dfca771230142c6d1446d418eb5e2b.jpg](../iclr_results/691_Language Agents Meet Causality -- Bridging LLMs and Causal World Models/images/6646228f3cc606f82f5c678a4c248b7a81dfca771230142c6d1446d418eb5e2b.jpg)

![882ac5251365206845226ba73d98c177bf332f2086df42f1ebed84cf8c310f77.jpg](../iclr_results/691_Language Agents Meet Causality -- Bridging LLMs and Causal World Models/images/882ac5251365206845226ba73d98c177bf332f2086df42f1ebed84cf8c310f77.jpg)

![a3600762d0dd481708262ba587f911a0c0aa80b95f3158a145d38dcbeec474ea.jpg](../iclr_results/691_Language Agents Meet Causality -- Bridging LLMs and Causal World Models/images/a3600762d0dd481708262ba587f911a0c0aa80b95f3158a145d38dcbeec474ea.jpg)

### Tables

![35567f9c3e7f1ec8a35924e1d0f549a8844a81f1cb509e88c646a6839a78ccde.jpg](../iclr_results/691_Language Agents Meet Causality -- Bridging LLMs and Causal World Models/tables/35567f9c3e7f1ec8a35924e1d0f549a8844a81f1cb509e88c646a6839a78ccde.jpg)

![5905516e7d763978b19a30555229d0d33e26fbcc66b77c2f6aeb122dcce6f1d8.jpg](../iclr_results/691_Language Agents Meet Causality -- Bridging LLMs and Causal World Models/tables/5905516e7d763978b19a30555229d0d33e26fbcc66b77c2f6aeb122dcce6f1d8.jpg)

![b594a145ae72f902ace133e0074e4c86a0d92abcd17994240ffd628cc96ef2ed.jpg](../iclr_results/691_Language Agents Meet Causality -- Bridging LLMs and Causal World Models/tables/b594a145ae72f902ace133e0074e4c86a0d92abcd17994240ffd628cc96ef2ed.jpg)

![d5163ec10cbd34f91d9a1c0f0b6141d73c946c2d95ff547fc7fff255afeb0fe3.jpg](../iclr_results/691_Language Agents Meet Causality -- Bridging LLMs and Causal World Models/tables/d5163ec10cbd34f91d9a1c0f0b6141d73c946c2d95ff547fc7fff255afeb0fe3.jpg)

![ede819d2533d002af43e094ca24de7edf80a9fde44f5deb77b7afba6d643ff52.jpg](../iclr_results/691_Language Agents Meet Causality -- Bridging LLMs and Causal World Models/tables/ede819d2533d002af43e094ca24de7edf80a9fde44f5deb77b7afba6d643ff52.jpg)
