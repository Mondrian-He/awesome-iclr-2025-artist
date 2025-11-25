# ICLR 2025 Main Conference Papers

**Summary:** 37 papers with extracted content:
- 📊 Total images: 46210
- 📋 Total tables: 34695
- 📄 Total files: 80905

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 5 of 100

## 目录 (Table of Contents)

1. [Accelerated training through iterative gradient propagation along the residual path](#Accelerated-training-through-iterative-gradient-propagation-along-the-residual-path)
2. [Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to Refuse](#Measuring-and-Enhancing-Trustworthiness-of-LLMs-in-RAG-through-Grounded-Attributions-and-Learning-to-Refuse)
3. [Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models](#Sparse-Feature-Circuits-Discovering-and-Editing-Interpretable-Causal-Graphs-in-Language-Models)
4. [AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models](#AlphaEdit-Null-Space-Constrained-Knowledge-Editing-for-Language-Models)
5. [STAR: Synthesis of Tailored Architectures](#STAR-Synthesis-of-Tailored-Architectures)
6. [MMIE: Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models](#MMIE-Massive-Multimodal-Interleaved-Comprehension-Benchmark-for-Large-Vision-Language-Models)
7. [SAM 2: Segment Anything in Images and Videos](#SAM-2-Segment-Anything-in-Images-and-Videos)
8. [Data Shapley in One Training Run](#Data-Shapley-in-One-Training-Run)
9. [Oscillatory State-Space Models](#Oscillatory-State-Space-Models)
10. [Restructuring Vector Quantization with the Rotation Trick](#Restructuring-Vector-Quantization-with-the-Rotation-Trick)
11. [MMQA: Evaluating LLMs with Multi-Table Multi-Hop Complex Questions](#MMQA-Evaluating-LLMs-with-Multi-Table-Multi-Hop-Complex-Questions)
12. [GridMix: Exploring Spatial Modulation for Neural Fields in PDE Modeling](#GridMix-Exploring-Spatial-Modulation-for-Neural-Fields-in-PDE-Modeling)
13. [More RLHF, More Trust? On The Impact of Preference Alignment On Trustworthiness](#More-RLHF-More-Trust-On-The-Impact-of-Preference-Alignment-On-Trustworthiness)
14. [Population Transformer: Learning Population-level Representations of Neural Activity](#Population-Transformer-Learning-Population-level-Representations-of-Neural-Activity)
15. [Inference Scaling for Long-Context Retrieval Augmented Generation](#Inference-Scaling-for-Long-Context-Retrieval-Augmented-Generation)
16. [Proxy Denoising for Source-Free Domain Adaptation](#Proxy-Denoising-for-Source-Free-Domain-Adaptation)
17. [Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs](#Turning-Up-the-Heat-Min-p-Sampling-for-Creative-and-Coherent-LLM-Outputs)
18. [Topological Blindspots: Understanding and Extending Topological Deep Learning Through the Lens of Expressivity](#Topological-Blindspots-Understanding-and-Extending-Topological-Deep-Learning-Through-the-Lens-of-Expressivity)
19. [Retrieval Head Mechanistically Explains Long-Context Factuality](#Retrieval-Head-Mechanistically-Explains-Long-Context-Factuality)
20. [MIND over Body: Adaptive Thinking using Dynamic Computation](#MIND-over-Body-Adaptive-Thinking-using-Dynamic-Computation)
21. [How much of my dataset did you use? Quantitative Data Usage Inference in Machine Learning](#How-much-of-my-dataset-did-you-use-Quantitative-Data-Usage-Inference-in-Machine-Learning)
22. [SymmetricDiffusers: Learning Discrete Diffusion on Finite Symmetric Groups](#SymmetricDiffusers-Learning-Discrete-Diffusion-on-Finite-Symmetric-Groups)
23. [Cut Your Losses in Large-Vocabulary Language Models](#Cut-Your-Losses-in-Large-Vocabulary-Language-Models)
24. [Interpreting Emergent Planning in Model-Free Reinforcement Learning](#Interpreting-Emergent-Planning-in-Model-Free-Reinforcement-Learning)
25. [Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think](#Representation-Alignment-for-Generation-Training-Diffusion-Transformers-Is-Easier-Than-You-Think)
26. [Progressive Compression with Universally Quantized Diffusion Models](#Progressive-Compression-with-Universally-Quantized-Diffusion-Models)
27. [High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian Representation](#High-Dynamic-Radar-Sequence-Prediction-for-Weather-Nowcasting-Using-Spatiotemporal-Coherent-Gaussian-Representation)
28. [Training Language Models to Self-Correct via Reinforcement Learning](#Training-Language-Models-to-Self-Correct-via-Reinforcement-Learning)
29. [Open-YOLO 3D: Towards Fast and Accurate Open-Vocabulary 3D Instance Segmentation](#Open-YOLO-3D-Towards-Fast-and-Accurate-Open-Vocabulary-3D-Instance-Segmentation)
30. [What should a neuron aim for? Designing local objective functions based on information theory](#What-should-a-neuron-aim-for-Designing-local-objective-functions-based-on-information-theory)
31. [Backtracking Improves Generation Safety](#Backtracking-Improves-Generation-Safety)
32. [Spread Preference Annotation: Direct Preference Judgment for Efficient LLM Alignment](#Spread-Preference-Annotation-Direct-Preference-Judgment-for-Efficient-LLM-Alignment)
33. [Global Convergence in Neural ODEs: Impact of Activation Functions](#Global-Convergence-in-Neural-ODEs-Impact-of-Activation-Functions)
34. [Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces](#Geometry-of-Neural-Reinforcement-Learning-in-Continuous-State-and-Action-Spaces)
35. [The Hidden Cost of Waiting for Accurate Predictions](#The-Hidden-Cost-of-Waiting-for-Accurate-Predictions)
36. [DeepLTL: Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL](#DeepLTL-Learning-to-Efficiently-Satisfy-Complex-LTL-Specifications-for-Multi-Task-RL)
37. [The Complexity of Two-Team Polymatrix Games with Independent Adversaries](#The-Complexity-of-Two-Team-Polymatrix-Games-with-Independent-Adversaries)

---


## Accelerated training through iterative gradient propagation along the residual path

### Images

![126a5608e7981739d0e85ca038d12b113543625a6a27c7712ff098fe80f7e9b2.jpg](../iclr_results/150_Residual Deep Gaussian Processes on Manifolds/images/126a5608e7981739d0e85ca038d12b113543625a6a27c7712ff098fe80f7e9b2.jpg)

![13fc288fc2d8839e14263e3a39ea71dd3f64e8bac6bd4c319ca8d0d6ad19467f.jpg](../iclr_results/150_Residual Deep Gaussian Processes on Manifolds/images/13fc288fc2d8839e14263e3a39ea71dd3f64e8bac6bd4c319ca8d0d6ad19467f.jpg)

![250069ded2cde11ff8eb686bd7bfc3833bf605e8860ca8bf3baeec59f0080c6b.jpg](../iclr_results/150_Residual Deep Gaussian Processes on Manifolds/images/250069ded2cde11ff8eb686bd7bfc3833bf605e8860ca8bf3baeec59f0080c6b.jpg)

![2c649b46e4e6f001a27d4f185ac4e3e8232531bc2ab19913ec7a2177ca172b8d.jpg](../iclr_results/150_Residual Deep Gaussian Processes on Manifolds/images/2c649b46e4e6f001a27d4f185ac4e3e8232531bc2ab19913ec7a2177ca172b8d.jpg)

![335162ab93dde37553476e9158b1ab3be77bcdfdf531723588687f41fbfd3c87.jpg](../iclr_results/150_Residual Deep Gaussian Processes on Manifolds/images/335162ab93dde37553476e9158b1ab3be77bcdfdf531723588687f41fbfd3c87.jpg)

![33849bc242e49cd5d641520469ca065a1d0b3e4fad2a6ddfd999ad0e1279b410.jpg](../iclr_results/150_Residual Deep Gaussian Processes on Manifolds/images/33849bc242e49cd5d641520469ca065a1d0b3e4fad2a6ddfd999ad0e1279b410.jpg)

![376ef941676e4687608e87c5b676c3cc56fb351209a6f901c6126a48223468f6.jpg](../iclr_results/150_Residual Deep Gaussian Processes on Manifolds/images/376ef941676e4687608e87c5b676c3cc56fb351209a6f901c6126a48223468f6.jpg)

![41ab5fe992e8c82ae782b0970b09395f63e7afe6b969b96bf470812fe9dca663.jpg](../iclr_results/150_Residual Deep Gaussian Processes on Manifolds/images/41ab5fe992e8c82ae782b0970b09395f63e7afe6b969b96bf470812fe9dca663.jpg)

![4d5524e9e026a098cbc11d62a273aaf5537393143d7f3af1b0d66e3c332a3aac.jpg](../iclr_results/150_Residual Deep Gaussian Processes on Manifolds/images/4d5524e9e026a098cbc11d62a273aaf5537393143d7f3af1b0d66e3c332a3aac.jpg)

![4fd25481975b16ad8c803822a08cf00cea00335c41d7d4d14289523605eee56a.jpg](../iclr_results/150_Residual Deep Gaussian Processes on Manifolds/images/4fd25481975b16ad8c803822a08cf00cea00335c41d7d4d14289523605eee56a.jpg)

![5a0be59e397693e80f20bf36e24edee8803429806a7a960c8ecb9e936a401d89.jpg](../iclr_results/150_Residual Deep Gaussian Processes on Manifolds/images/5a0be59e397693e80f20bf36e24edee8803429806a7a960c8ecb9e936a401d89.jpg)

![651e8631b28b358dd673136b479751db280f6ea4725a24e14fa2143275816b3e.jpg](../iclr_results/150_Residual Deep Gaussian Processes on Manifolds/images/651e8631b28b358dd673136b479751db280f6ea4725a24e14fa2143275816b3e.jpg)

![654351f153b4790dfc00fd2035542628032b865df2385ddb49f8579779d5c4aa.jpg](../iclr_results/150_Residual Deep Gaussian Processes on Manifolds/images/654351f153b4790dfc00fd2035542628032b865df2385ddb49f8579779d5c4aa.jpg)

![6c1092120846bf07c457be1bc9f3ab2a155ad2361414efb55b487ccb4640790a.jpg](../iclr_results/150_Residual Deep Gaussian Processes on Manifolds/images/6c1092120846bf07c457be1bc9f3ab2a155ad2361414efb55b487ccb4640790a.jpg)

![771c2b86526aa1e87689f942a92acf6172ad5056f2e6efa56676a52940fa3cce.jpg](../iclr_results/150_Residual Deep Gaussian Processes on Manifolds/images/771c2b86526aa1e87689f942a92acf6172ad5056f2e6efa56676a52940fa3cce.jpg)

![9a7a6f4999b8b2d3bd8c2773842b9b04daed38a3e7b0b009e979c8642fe2181c.jpg](../iclr_results/150_Residual Deep Gaussian Processes on Manifolds/images/9a7a6f4999b8b2d3bd8c2773842b9b04daed38a3e7b0b009e979c8642fe2181c.jpg)

![c8749ccbfede705b063d6de12e91c307580eef5894b611c9c62ca2cad932f97d.jpg](../iclr_results/150_Residual Deep Gaussian Processes on Manifolds/images/c8749ccbfede705b063d6de12e91c307580eef5894b611c9c62ca2cad932f97d.jpg)

![ce1d645c0bdbeac90fed8a57ebd5d044dd45a25a80eee3d980172e8c3e49feb0.jpg](../iclr_results/150_Residual Deep Gaussian Processes on Manifolds/images/ce1d645c0bdbeac90fed8a57ebd5d044dd45a25a80eee3d980172e8c3e49feb0.jpg)

![cf53e0ebb4f0987c86c3155950258285fc9b7ef3ff4987d736acff7d6ad012a3.jpg](../iclr_results/150_Residual Deep Gaussian Processes on Manifolds/images/cf53e0ebb4f0987c86c3155950258285fc9b7ef3ff4987d736acff7d6ad012a3.jpg)

![e49e7c9be7e99efc9513895f62d0e9be6a4f9e5224e5fbeea62fbc6feee1bf04.jpg](../iclr_results/150_Residual Deep Gaussian Processes on Manifolds/images/e49e7c9be7e99efc9513895f62d0e9be6a4f9e5224e5fbeea62fbc6feee1bf04.jpg)

![fa7beab63146974b925bac24f804147923141d99cc4086a1448b7a964dc4c877.jpg](../iclr_results/150_Residual Deep Gaussian Processes on Manifolds/images/fa7beab63146974b925bac24f804147923141d99cc4086a1448b7a964dc4c877.jpg)

## Accelerated training through iterative gradient propagation along the residual path


### Images

![06ae2016c31bccc89b3aa687e024bf296ea6bf8fdbbab87694be5f48ad674998.jpg](../iclr_results/151_Accelerated training through iterative gradient propagation along the residual path/images/06ae2016c31bccc89b3aa687e024bf296ea6bf8fdbbab87694be5f48ad674998.jpg)

![36d956b8779c099497112e19c68dd703fb2f4e7eb4d4df81c171be4529245a27.jpg](../iclr_results/151_Accelerated training through iterative gradient propagation along the residual path/images/36d956b8779c099497112e19c68dd703fb2f4e7eb4d4df81c171be4529245a27.jpg)

![7114dca90bf7746cc5d3eb38f21a467572376d4312fc5f286d58a97f8568f9b0.jpg](../iclr_results/151_Accelerated training through iterative gradient propagation along the residual path/images/7114dca90bf7746cc5d3eb38f21a467572376d4312fc5f286d58a97f8568f9b0.jpg)

![9d3581838880d6fab75b623db958b94ef7edf26a4a00b995f8c53628bd90526b.jpg](../iclr_results/151_Accelerated training through iterative gradient propagation along the residual path/images/9d3581838880d6fab75b623db958b94ef7edf26a4a00b995f8c53628bd90526b.jpg)

![cbe429ee0cc1db1fb733a8e5b4e86beb09d96f131341608930dd6a222e985864.jpg](../iclr_results/151_Accelerated training through iterative gradient propagation along the residual path/images/cbe429ee0cc1db1fb733a8e5b4e86beb09d96f131341608930dd6a222e985864.jpg)

![cbff928eabc7565be7c71cc150c10c89df1512c770ee27d0054028a6fbfd2472.jpg](../iclr_results/151_Accelerated training through iterative gradient propagation along the residual path/images/cbff928eabc7565be7c71cc150c10c89df1512c770ee27d0054028a6fbfd2472.jpg)

### Tables

![0ed98cb397d5fc0d9fdd0831596db0e39a55fe34ad5a245b243ec879b7b836dc.jpg](../iclr_results/151_Accelerated training through iterative gradient propagation along the residual path/tables/0ed98cb397d5fc0d9fdd0831596db0e39a55fe34ad5a245b243ec879b7b836dc.jpg)

![2510de6e9f7b4754e2ab6f20c71786dfc6ab27f4848e3abb70ad45629a740406.jpg](../iclr_results/151_Accelerated training through iterative gradient propagation along the residual path/tables/2510de6e9f7b4754e2ab6f20c71786dfc6ab27f4848e3abb70ad45629a740406.jpg)

![2cadca6d771c0381ac891f9c937b8cf094278deb66151e477bb7daa55ac93983.jpg](../iclr_results/151_Accelerated training through iterative gradient propagation along the residual path/tables/2cadca6d771c0381ac891f9c937b8cf094278deb66151e477bb7daa55ac93983.jpg)

![48b790f4ca6bd8a4aa4afc9f440bb57af890b9db7579fc0dbc12e6defc90a119.jpg](../iclr_results/151_Accelerated training through iterative gradient propagation along the residual path/tables/48b790f4ca6bd8a4aa4afc9f440bb57af890b9db7579fc0dbc12e6defc90a119.jpg)

![afda3ce44dfaf491cbb1ab880c14bd432f16f5378364771cba9a920691f5b402.jpg](../iclr_results/151_Accelerated training through iterative gradient propagation along the residual path/tables/afda3ce44dfaf491cbb1ab880c14bd432f16f5378364771cba9a920691f5b402.jpg)

![d7bf7f672cc1e8740bb109525c3f5adbb6e8d199efb4ac395beab57a39ee756e.jpg](../iclr_results/151_Accelerated training through iterative gradient propagation along the residual path/tables/d7bf7f672cc1e8740bb109525c3f5adbb6e8d199efb4ac395beab57a39ee756e.jpg)

## Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to Refuse


### Images

![3c311b4f927573370c9d93ce4eea05e7d17d72b0bb17a0e25e17b16c540bb01d.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/images/3c311b4f927573370c9d93ce4eea05e7d17d72b0bb17a0e25e17b16c540bb01d.jpg)

![768d4a3e96fa325835ddb26d54771c528912b31ae4f5e4c9bd683bf5f6675d84.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/images/768d4a3e96fa325835ddb26d54771c528912b31ae4f5e4c9bd683bf5f6675d84.jpg)

![8db4897ce96aa6185efdf8a54d2d1aa0c7697c82c376b5dc05288c72b7c81cfb.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/images/8db4897ce96aa6185efdf8a54d2d1aa0c7697c82c376b5dc05288c72b7c81cfb.jpg)

![8e9c2cd0e0d4e2370ca66dc39599e8611d8353f2f3634a4c107b92be15d7ee98.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/images/8e9c2cd0e0d4e2370ca66dc39599e8611d8353f2f3634a4c107b92be15d7ee98.jpg)

![b019ab54bf0a9bc6baac05897657735696c5a7efcda623558a9a02119d80dc69.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/images/b019ab54bf0a9bc6baac05897657735696c5a7efcda623558a9a02119d80dc69.jpg)

![c00015b4cf8a11e33fe37dc2614bd56c2e5a83c3d957276a4a8b4b3ccf4ac379.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/images/c00015b4cf8a11e33fe37dc2614bd56c2e5a83c3d957276a4a8b4b3ccf4ac379.jpg)

![c2b81919a5df8ffa83c54fb3071237a674b5331c860ae0db056d8d1ddc9b743c.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/images/c2b81919a5df8ffa83c54fb3071237a674b5331c860ae0db056d8d1ddc9b743c.jpg)

### Tables

![0eba679d412e72845e0bcbaf7dcec837c579539b4e63b5b9435dae9edba434fb.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/0eba679d412e72845e0bcbaf7dcec837c579539b4e63b5b9435dae9edba434fb.jpg)

![2a4f154becaca3c6cb9bbaa24fb1900969287132e3264cea09ee35834b7d87a7.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/2a4f154becaca3c6cb9bbaa24fb1900969287132e3264cea09ee35834b7d87a7.jpg)

![4541721faaf530bb67b29b4d76673365a1b0c128c049370f6a7563cff00e47c1.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/4541721faaf530bb67b29b4d76673365a1b0c128c049370f6a7563cff00e47c1.jpg)

![487fd5259628586d547339acf597540f33f90a845282d15e1de1cfd4760a4f07.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/487fd5259628586d547339acf597540f33f90a845282d15e1de1cfd4760a4f07.jpg)

![5053fd61af7011a6921d731fcfb00e8bb25d22a619c21467434991d933352d05.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/5053fd61af7011a6921d731fcfb00e8bb25d22a619c21467434991d933352d05.jpg)

![5e35147a60a5c6ad8b9d35545fc28a2c7b87794ead12089a18179fd1c69607ca.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/5e35147a60a5c6ad8b9d35545fc28a2c7b87794ead12089a18179fd1c69607ca.jpg)

![6699e71a57cec47e01f064024818d575c52ca12d80e9b0b35b3fd81accfe0d22.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/6699e71a57cec47e01f064024818d575c52ca12d80e9b0b35b3fd81accfe0d22.jpg)

![6bbf39df98497eff2d13b4f12b7344a8c8ad9dd47677f7da0ed995eddef2f10e.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/6bbf39df98497eff2d13b4f12b7344a8c8ad9dd47677f7da0ed995eddef2f10e.jpg)

![9138dc69bedae068d1a2dc23752861c43a8555b2c791f645b57e321a52b76d4b.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/9138dc69bedae068d1a2dc23752861c43a8555b2c791f645b57e321a52b76d4b.jpg)

![93c028cfc3dc0b19f3c27a31a9aa269487447ea94fbb8cbab0513b27e55678db.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/93c028cfc3dc0b19f3c27a31a9aa269487447ea94fbb8cbab0513b27e55678db.jpg)

![987adabc459d1ab65001442a5e3f63931eff85f44052923ff8e27a044ffa0952.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/987adabc459d1ab65001442a5e3f63931eff85f44052923ff8e27a044ffa0952.jpg)

![9d905f998266855b60421b1caa190f557c646d3d3a86bc44dfeb715b9b56bd89.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/9d905f998266855b60421b1caa190f557c646d3d3a86bc44dfeb715b9b56bd89.jpg)

![a6c312cad79e7363ad2726d4a992e538664fec343afc0c49e167f073980f1cbf.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/a6c312cad79e7363ad2726d4a992e538664fec343afc0c49e167f073980f1cbf.jpg)

![ba216ec07f34139319c45a94bd3c609200f40a73b924a4f85db07184100952f1.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/ba216ec07f34139319c45a94bd3c609200f40a73b924a4f85db07184100952f1.jpg)

![c43a944edd7b213994be630ea4ba89448edf11e4c9d135748fe8edb35b5510ef.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/c43a944edd7b213994be630ea4ba89448edf11e4c9d135748fe8edb35b5510ef.jpg)

![c5b4f8d19ffb208167b2438b6119ed1734b4f1cab7bec960690393447252a2a2.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/c5b4f8d19ffb208167b2438b6119ed1734b4f1cab7bec960690393447252a2a2.jpg)

![c76cf04388727c3efee16897c1c1fc3a3e0c6caa23c9fe5c73d0d49b3b1fbd10.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/c76cf04388727c3efee16897c1c1fc3a3e0c6caa23c9fe5c73d0d49b3b1fbd10.jpg)

![c77407b381e9eb7dd7ef2a2f2b3f8f81d3e39830ce24d335be736f86851ac396.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/c77407b381e9eb7dd7ef2a2f2b3f8f81d3e39830ce24d335be736f86851ac396.jpg)

![c8303a931ac26b9b263faa364ea01682682f58c03665a7b1ba77f98fbcfe5c46.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/c8303a931ac26b9b263faa364ea01682682f58c03665a7b1ba77f98fbcfe5c46.jpg)

![ce602a9f870d29bfd8b3238b10a73d576d4cb01bc0c472c576a24023932459b3.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/ce602a9f870d29bfd8b3238b10a73d576d4cb01bc0c472c576a24023932459b3.jpg)

![d4eb2a1b16c7bd959989b15dd954ee280e03c83e8975a7036e992a83783fb5cc.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/d4eb2a1b16c7bd959989b15dd954ee280e03c83e8975a7036e992a83783fb5cc.jpg)

![e5c4423fde6082c011120f2d84d25b34c63dd1bc8f77df260ce5d1caa368351d.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/e5c4423fde6082c011120f2d84d25b34c63dd1bc8f77df260ce5d1caa368351d.jpg)

![efc258ca9d1330b1106b4914bc0c51ae1c01249524e8caee659b4611b2b77ad3.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/efc258ca9d1330b1106b4914bc0c51ae1c01249524e8caee659b4611b2b77ad3.jpg)

![fb04ac0baa3bbb460c5503f6b85d40a96104ba5b8061c66a180589e13f5d6173.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/fb04ac0baa3bbb460c5503f6b85d40a96104ba5b8061c66a180589e13f5d6173.jpg)

![fd26430094e1d0b64084318092bd31164507832e7585a61da50f3ec4967d1af8.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/fd26430094e1d0b64084318092bd31164507832e7585a61da50f3ec4967d1af8.jpg)

![ffae537059da5177518c9abe1d250dc7925469e7cfe1cf89c03bd9ab6b2d9192.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/ffae537059da5177518c9abe1d250dc7925469e7cfe1cf89c03bd9ab6b2d9192.jpg)

![ffc97fd492a881d102dccbf3b73a5703d1d4e04c95d09c21d0cdbefc6743c343.jpg](../iclr_results/152_Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to/tables/ffc97fd492a881d102dccbf3b73a5703d1d4e04c95d09c21d0cdbefc6743c343.jpg)

## Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models


### Images

![11b40561b7481fbc6cd0f4151d9d7df12e4f2885a9f0f9e30674426f921322e4.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/11b40561b7481fbc6cd0f4151d9d7df12e4f2885a9f0f9e30674426f921322e4.jpg)

![28d4ffb75896db91745cd7749b98f21d66910d8369a1b92baf7c91282b3064a3.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/28d4ffb75896db91745cd7749b98f21d66910d8369a1b92baf7c91282b3064a3.jpg)

![30ad6d44969b9f1315996d170ef116b1dfa667bb4adaa412b6550215dfa0bc2d.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/30ad6d44969b9f1315996d170ef116b1dfa667bb4adaa412b6550215dfa0bc2d.jpg)

![35f940de07fd366dad20e373787126d86721fde43d760e5df069cc6064c4b15d.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/35f940de07fd366dad20e373787126d86721fde43d760e5df069cc6064c4b15d.jpg)

![3dde4b3a0416eb49ee58c40d32ccac61d0d383c48038fa02f703c290f2ac7c96.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/3dde4b3a0416eb49ee58c40d32ccac61d0d383c48038fa02f703c290f2ac7c96.jpg)

![40dae6b008186b19ac5d74c0c73fa88fc72ddd3f7ed9fcfb4fbe6192ce91ced9.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/40dae6b008186b19ac5d74c0c73fa88fc72ddd3f7ed9fcfb4fbe6192ce91ced9.jpg)

![48b82d531e46263447d9f170661cef884919236ce5ef0229bfcac7c8bea58e0b.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/48b82d531e46263447d9f170661cef884919236ce5ef0229bfcac7c8bea58e0b.jpg)

![4a380833aea02d0c03d2c6429e244ea1fe68625a0f9ebab4c907fdae6dddd924.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/4a380833aea02d0c03d2c6429e244ea1fe68625a0f9ebab4c907fdae6dddd924.jpg)

![64486f1acef434a18dd9781b49d4da80e470f850a3dd6c26175f4fcce03a1778.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/64486f1acef434a18dd9781b49d4da80e470f850a3dd6c26175f4fcce03a1778.jpg)

![69f82ff255b06700eab68aa4c64bf3999c6bdb747c647d7e4db623892135c6ae.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/69f82ff255b06700eab68aa4c64bf3999c6bdb747c647d7e4db623892135c6ae.jpg)

![7501950739161dc45ddf372a8b173d1f9d6667378e28c558232b22ada60840a0.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/7501950739161dc45ddf372a8b173d1f9d6667378e28c558232b22ada60840a0.jpg)

![83cc7cb6716710f63558cbd9f933ee3a67f3184b88ce53eac7becd3197c571db.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/83cc7cb6716710f63558cbd9f933ee3a67f3184b88ce53eac7becd3197c571db.jpg)

![872991c467c6d941a6b2436387a316257cbc861dc4110c3cdc3a46a26bbefdce.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/872991c467c6d941a6b2436387a316257cbc861dc4110c3cdc3a46a26bbefdce.jpg)

![90619e0edb968b5779947f7ef9d469315ecfa06c40a496bafb7a093c50e8344b.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/90619e0edb968b5779947f7ef9d469315ecfa06c40a496bafb7a093c50e8344b.jpg)

![9cad62e165637b368f197f0fc91f2a724b57a367642c1c3534404f9301997df5.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/9cad62e165637b368f197f0fc91f2a724b57a367642c1c3534404f9301997df5.jpg)

![aa2c069296b8f9afc5ecc1a6767c1fb33cd9cd75f6aa8e3781666e07896f8801.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/aa2c069296b8f9afc5ecc1a6767c1fb33cd9cd75f6aa8e3781666e07896f8801.jpg)

![add21d772eb9327ba06eb17265e9315a80f9a9d8622d97866b5e5d1456d5f933.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/add21d772eb9327ba06eb17265e9315a80f9a9d8622d97866b5e5d1456d5f933.jpg)

![c20f927da66639b31eb2bd31bfb396e160a4f4137e51ff019646e78c4b4d90af.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/c20f927da66639b31eb2bd31bfb396e160a4f4137e51ff019646e78c4b4d90af.jpg)

![cdb8b70a4280b1e63c50bb4b1b31b60cef899b180f1d8f79a564b04ca7c5bee8.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/cdb8b70a4280b1e63c50bb4b1b31b60cef899b180f1d8f79a564b04ca7c5bee8.jpg)

![d0bdf93f03594b08fa13d4da243f426ec931853483ad33e7792d06bc939fc80e.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/d0bdf93f03594b08fa13d4da243f426ec931853483ad33e7792d06bc939fc80e.jpg)

![d5cb2e12a5d596d34c9c3cd241c6c766be98e7f84006f48569ed124c106bd39a.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/d5cb2e12a5d596d34c9c3cd241c6c766be98e7f84006f48569ed124c106bd39a.jpg)

![dd79a38a20cc75c2ad7d7360f2f49be7d173b1605e8ec60bd28c0a4b397bc2e3.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/dd79a38a20cc75c2ad7d7360f2f49be7d173b1605e8ec60bd28c0a4b397bc2e3.jpg)

![e966125aff02be7fed677fd3d7172333cac6569cae80a92bc66685c9261aa48b.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/e966125aff02be7fed677fd3d7172333cac6569cae80a92bc66685c9261aa48b.jpg)

![f071f789a9e93a38b6d6d1c4cabd63c0f64dcc464af6ffcc6150910b0718ffc3.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/f071f789a9e93a38b6d6d1c4cabd63c0f64dcc464af6ffcc6150910b0718ffc3.jpg)

![ff29b7f199ae1a9e30753a934d62aa82e969fce1038c3d7a4ae504f3c8ac8053.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/images/ff29b7f199ae1a9e30753a934d62aa82e969fce1038c3d7a4ae504f3c8ac8053.jpg)

### Tables

![51f3ff337175ac2cc1c5827a41aa0c4c8adca6cc7284b913e5f3f62cbfd07a4d.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/tables/51f3ff337175ac2cc1c5827a41aa0c4c8adca6cc7284b913e5f3f62cbfd07a4d.jpg)

![5f0b94c96c4c35cd692d1512edc62b7f27bb65c4c1b8ef07276347a7e66077f9.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/tables/5f0b94c96c4c35cd692d1512edc62b7f27bb65c4c1b8ef07276347a7e66077f9.jpg)

![5fe0b93dfaedef01e2d90f962fc3112b97104ec93ed38b1d9b150bb345cb3601.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/tables/5fe0b93dfaedef01e2d90f962fc3112b97104ec93ed38b1d9b150bb345cb3601.jpg)

![7fd229fe7f4b7ae6f57f53cc81c5492ee988e3902fed42d7d243b47ad91e4a91.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/tables/7fd229fe7f4b7ae6f57f53cc81c5492ee988e3902fed42d7d243b47ad91e4a91.jpg)

![985fba554bbdbd33543137c77e72d38b4b95660706ffb90a3df2847b75c3ff1f.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/tables/985fba554bbdbd33543137c77e72d38b4b95660706ffb90a3df2847b75c3ff1f.jpg)

![c1730a375fb843008babc2e9c1de249c0f7d0d4d0973fbef55b47157e59e6667.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/tables/c1730a375fb843008babc2e9c1de249c0f7d0d4d0973fbef55b47157e59e6667.jpg)

![e2d97105dd1eb0af7f19bfafb08df969e33c23c3bc106237d3e6124de4dd2d3e.jpg](../iclr_results/153_Sparse Feature Circuits_ Discovering and Editing Interpretable Causal Graphs in Language Models/tables/e2d97105dd1eb0af7f19bfafb08df969e33c23c3bc106237d3e6124de4dd2d3e.jpg)

## AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models


### Images

![011b2a6e7b63f0369ba92092c92bd5c6cb7f4fc5b8f9199c6464b542ec581c5d.jpg](../iclr_results/154_AlphaEdit_ Null-Space Constrained Knowledge Editing for Language Models/images/011b2a6e7b63f0369ba92092c92bd5c6cb7f4fc5b8f9199c6464b542ec581c5d.jpg)

![106bcf0f1ff3dca824ab1ab29c8edcd8ba569553582e9300cd1c9fde0bc3a226.jpg](../iclr_results/154_AlphaEdit_ Null-Space Constrained Knowledge Editing for Language Models/images/106bcf0f1ff3dca824ab1ab29c8edcd8ba569553582e9300cd1c9fde0bc3a226.jpg)

![332db377626f15f7ef2768346413e13fe1f13a99c9a11e371300ade0f55750a6.jpg](../iclr_results/154_AlphaEdit_ Null-Space Constrained Knowledge Editing for Language Models/images/332db377626f15f7ef2768346413e13fe1f13a99c9a11e371300ade0f55750a6.jpg)

![5e3be9cc27e62e2396c600501acd0e3873da5994041da090bb9b05f6345b0200.jpg](../iclr_results/154_AlphaEdit_ Null-Space Constrained Knowledge Editing for Language Models/images/5e3be9cc27e62e2396c600501acd0e3873da5994041da090bb9b05f6345b0200.jpg)

![6a3f05c8fb0ae3685a406071231e42de017b596daecab07389cf756d7ffca4d8.jpg](../iclr_results/154_AlphaEdit_ Null-Space Constrained Knowledge Editing for Language Models/images/6a3f05c8fb0ae3685a406071231e42de017b596daecab07389cf756d7ffca4d8.jpg)

![8385c278d0633a24a2184a7d5ab3ab9e98fdfaa1c239d331009291374789d9ec.jpg](../iclr_results/154_AlphaEdit_ Null-Space Constrained Knowledge Editing for Language Models/images/8385c278d0633a24a2184a7d5ab3ab9e98fdfaa1c239d331009291374789d9ec.jpg)

![8c21549405cbeff081158f2d8c057032711cb0addb9c3ef1aebccd9b77dd0b41.jpg](../iclr_results/154_AlphaEdit_ Null-Space Constrained Knowledge Editing for Language Models/images/8c21549405cbeff081158f2d8c057032711cb0addb9c3ef1aebccd9b77dd0b41.jpg)

![97d60555e8bddc187549fc10ac58ae45b20d9ca7600e70190c6db9c529a8b89c.jpg](../iclr_results/154_AlphaEdit_ Null-Space Constrained Knowledge Editing for Language Models/images/97d60555e8bddc187549fc10ac58ae45b20d9ca7600e70190c6db9c529a8b89c.jpg)

![9973aea3f0a175d39f448824e806bb4998c1bf9eab0d1ed940df2df7194237c2.jpg](../iclr_results/154_AlphaEdit_ Null-Space Constrained Knowledge Editing for Language Models/images/9973aea3f0a175d39f448824e806bb4998c1bf9eab0d1ed940df2df7194237c2.jpg)

![9c90d5d58f4c615288ed4d395cac68ad8f13c36830fc477a5a589deea89108a4.jpg](../iclr_results/154_AlphaEdit_ Null-Space Constrained Knowledge Editing for Language Models/images/9c90d5d58f4c615288ed4d395cac68ad8f13c36830fc477a5a589deea89108a4.jpg)

![a5c08349ace24d2144373c4468ce6945869b818d292c094e1c26761c5c7735e0.jpg](../iclr_results/154_AlphaEdit_ Null-Space Constrained Knowledge Editing for Language Models/images/a5c08349ace24d2144373c4468ce6945869b818d292c094e1c26761c5c7735e0.jpg)

![c7f6ff7946abfa534ecc45ce0296ba7260ae1a285bd7ce690aca9c908046c106.jpg](../iclr_results/154_AlphaEdit_ Null-Space Constrained Knowledge Editing for Language Models/images/c7f6ff7946abfa534ecc45ce0296ba7260ae1a285bd7ce690aca9c908046c106.jpg)

### Tables

![06a4aaa134dae22c254b9e6cebde7b7ecb2c56cbc35880a737e6930266ddd008.jpg](../iclr_results/154_AlphaEdit_ Null-Space Constrained Knowledge Editing for Language Models/tables/06a4aaa134dae22c254b9e6cebde7b7ecb2c56cbc35880a737e6930266ddd008.jpg)

![105d16728003d2ab0db30adfa9009b69c0190c0231e3825ca7bcdc86dab0ceaf.jpg](../iclr_results/154_AlphaEdit_ Null-Space Constrained Knowledge Editing for Language Models/tables/105d16728003d2ab0db30adfa9009b69c0190c0231e3825ca7bcdc86dab0ceaf.jpg)

![40ea4d6c31875825787ca2ee3093cc68487cd17d021c508e901ec171827d2b86.jpg](../iclr_results/154_AlphaEdit_ Null-Space Constrained Knowledge Editing for Language Models/tables/40ea4d6c31875825787ca2ee3093cc68487cd17d021c508e901ec171827d2b86.jpg)

![6947daa9479110c1ce8a246163c866da0688ef103e7cb3bea3a22e197b1e7e47.jpg](../iclr_results/154_AlphaEdit_ Null-Space Constrained Knowledge Editing for Language Models/tables/6947daa9479110c1ce8a246163c866da0688ef103e7cb3bea3a22e197b1e7e47.jpg)

![8a9e45d5725a98bcdc96900e596e1460ed2ffdef649eba3144306c5b22f368a8.jpg](../iclr_results/154_AlphaEdit_ Null-Space Constrained Knowledge Editing for Language Models/tables/8a9e45d5725a98bcdc96900e596e1460ed2ffdef649eba3144306c5b22f368a8.jpg)

![a3754789b8ab61b426f9ccbd904e9187b9e1f83d015423f93191d5f77d0b9bd3.jpg](../iclr_results/154_AlphaEdit_ Null-Space Constrained Knowledge Editing for Language Models/tables/a3754789b8ab61b426f9ccbd904e9187b9e1f83d015423f93191d5f77d0b9bd3.jpg)

![b5403c8a51248c40224310228e44b616c0fde81c3788425c3ec96c783cfb43ae.jpg](../iclr_results/154_AlphaEdit_ Null-Space Constrained Knowledge Editing for Language Models/tables/b5403c8a51248c40224310228e44b616c0fde81c3788425c3ec96c783cfb43ae.jpg)

![b6aa9780459b2e6c81b3ef967939e9153867ad95a9bbaf1c64db58d76c85a00a.jpg](../iclr_results/154_AlphaEdit_ Null-Space Constrained Knowledge Editing for Language Models/tables/b6aa9780459b2e6c81b3ef967939e9153867ad95a9bbaf1c64db58d76c85a00a.jpg)

![bbb55726f0e70fffb3d9e19badabeaa597e55060cda8705c9b10ccac19146b27.jpg](../iclr_results/154_AlphaEdit_ Null-Space Constrained Knowledge Editing for Language Models/tables/bbb55726f0e70fffb3d9e19badabeaa597e55060cda8705c9b10ccac19146b27.jpg)

![c57290dac1670a34f648608a37a2dcc5a0e61eca56da2ebb48f4ac3b6a7cce6e.jpg](../iclr_results/154_AlphaEdit_ Null-Space Constrained Knowledge Editing for Language Models/tables/c57290dac1670a34f648608a37a2dcc5a0e61eca56da2ebb48f4ac3b6a7cce6e.jpg)

## STAR: Synthesis of Tailored Architectures


### Images

![03f4cf7c61ceeb290b557ecfc2b708c76f9b79007ebcb529ad7d80cd83f7750f.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/03f4cf7c61ceeb290b557ecfc2b708c76f9b79007ebcb529ad7d80cd83f7750f.jpg)

![095b2e752416cb4ab62e254698bb191b4a1345e8a3973150267373a66a27705e.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/095b2e752416cb4ab62e254698bb191b4a1345e8a3973150267373a66a27705e.jpg)

![0e84a847bd01f61dfa438c2ae45f57a01c4321efe761f0b13adcb72a647d5945.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/0e84a847bd01f61dfa438c2ae45f57a01c4321efe761f0b13adcb72a647d5945.jpg)

![125006ff48bee4787b98554a8687132f6c73b7b2d1287a33b7a95988f25bba19.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/125006ff48bee4787b98554a8687132f6c73b7b2d1287a33b7a95988f25bba19.jpg)

![2683060c990a574b6c6e77f10536ba3668434212a133b56a7c23033cecf0482f.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/2683060c990a574b6c6e77f10536ba3668434212a133b56a7c23033cecf0482f.jpg)

![283ce4f2798b740c1ad2c06799ff3cbc498ac10f5723db3030936b765e7fac26.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/283ce4f2798b740c1ad2c06799ff3cbc498ac10f5723db3030936b765e7fac26.jpg)

![2c3ec8162e34a7709ef8c5616d9f6e9884c033e973ea7d2e7c9d5fced0bbd0ad.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/2c3ec8162e34a7709ef8c5616d9f6e9884c033e973ea7d2e7c9d5fced0bbd0ad.jpg)

![362a70e37ecd056349dbfd17ba4280e4659ae35dee6a78cdd49f545ede041290.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/362a70e37ecd056349dbfd17ba4280e4659ae35dee6a78cdd49f545ede041290.jpg)

![3e6b949841bf55c391370bfb56e7912910310858fa2a3ea37a26c836cc594a1d.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/3e6b949841bf55c391370bfb56e7912910310858fa2a3ea37a26c836cc594a1d.jpg)

![3f86e68e291d7431701f3f39f8317261abac4f3c94c17fcde65d85876650c6d7.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/3f86e68e291d7431701f3f39f8317261abac4f3c94c17fcde65d85876650c6d7.jpg)

![408a7344dd6c2881b52ad01a7b28ae3b6db3b1d1889919f3ff61d32fde456cee.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/408a7344dd6c2881b52ad01a7b28ae3b6db3b1d1889919f3ff61d32fde456cee.jpg)

![482f4caec062746d1ba5fcb13f6e71ec705c9d3689f626e8ccf171b082623897.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/482f4caec062746d1ba5fcb13f6e71ec705c9d3689f626e8ccf171b082623897.jpg)

![4b922863272d89523ef985f96d06b75f1609ab1c93f2107bcb138a3e2c08c8ee.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/4b922863272d89523ef985f96d06b75f1609ab1c93f2107bcb138a3e2c08c8ee.jpg)

![56ea599a8eb3593c8971bbc11283964e6bbbc70f44b658111304dcd0d589e008.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/56ea599a8eb3593c8971bbc11283964e6bbbc70f44b658111304dcd0d589e008.jpg)

![5e2a10cd22f35436f8db9137a07d73ecb4f472ab95b938a9eb6416e31124199f.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/5e2a10cd22f35436f8db9137a07d73ecb4f472ab95b938a9eb6416e31124199f.jpg)

![60cb7820d40e3f5fc0b629a1fe45b0d3662c0ff6f522523a4ca07900448f4981.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/60cb7820d40e3f5fc0b629a1fe45b0d3662c0ff6f522523a4ca07900448f4981.jpg)

![6468d626523f3641b050147830f94022e4752849b23850fa1774d4743aa4cb52.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/6468d626523f3641b050147830f94022e4752849b23850fa1774d4743aa4cb52.jpg)

![6547b697fb59f94ade816a63093a052126f7de08174995f70bbd9ebf450eaa16.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/6547b697fb59f94ade816a63093a052126f7de08174995f70bbd9ebf450eaa16.jpg)

![666f52b9a18eca2f22c71d0776e4f118885e3992e00e52166fa891453fb6938a.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/666f52b9a18eca2f22c71d0776e4f118885e3992e00e52166fa891453fb6938a.jpg)

![6d250781c5cae88e04be4bcf9465fecef963424a9bfde2c9224ac6e2abfcb134.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/6d250781c5cae88e04be4bcf9465fecef963424a9bfde2c9224ac6e2abfcb134.jpg)

![762a6c92654ab2d23a35c63559d2cce58dcc4b9eedb37a5ec47f6f90f370d2e8.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/762a6c92654ab2d23a35c63559d2cce58dcc4b9eedb37a5ec47f6f90f370d2e8.jpg)

![77bb6db0f865e52eebaac429fac6ce469a0ec060ddbf9dd63f089e3f0269760b.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/77bb6db0f865e52eebaac429fac6ce469a0ec060ddbf9dd63f089e3f0269760b.jpg)

![7fa0846fffdd07f5297b60e221ab343dd974635a7ddf697a672fca9aee174259.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/7fa0846fffdd07f5297b60e221ab343dd974635a7ddf697a672fca9aee174259.jpg)

![8a641f45a1889c218d14f56c4f54b73b80ceea35d2ec8fb6ba70587a0f8228a1.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/8a641f45a1889c218d14f56c4f54b73b80ceea35d2ec8fb6ba70587a0f8228a1.jpg)

![8b2f23716da572fd2b7b35988584189878d37f68f5d0e18660a2b6a4ddabaadf.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/8b2f23716da572fd2b7b35988584189878d37f68f5d0e18660a2b6a4ddabaadf.jpg)

![93ce75e422c5b0cbe484e7421c7ff98303190804f197da76795ddaf9f623d574.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/93ce75e422c5b0cbe484e7421c7ff98303190804f197da76795ddaf9f623d574.jpg)

![953792bc064b381e1d761135a20fc6754638d896b1c7e6c14393c31ae20a185a.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/953792bc064b381e1d761135a20fc6754638d896b1c7e6c14393c31ae20a185a.jpg)

![9c9ef073ca8eb98a177c0a6af962fb9f6f92c08178494aa43a2439fa20b3ee8d.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/9c9ef073ca8eb98a177c0a6af962fb9f6f92c08178494aa43a2439fa20b3ee8d.jpg)

![9e9c5d4f36bd5ab79ba6da54c47e30e1a41c1623b4b116eb94a92c8590945454.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/9e9c5d4f36bd5ab79ba6da54c47e30e1a41c1623b4b116eb94a92c8590945454.jpg)

![abd4a5e2ce4766eafd514d5d3f0d3441bda2f958e96142f07a3f4519076587ac.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/abd4a5e2ce4766eafd514d5d3f0d3441bda2f958e96142f07a3f4519076587ac.jpg)

![b58a5cc96cc63ae8d60508af505904a9b0d8552edae3ad8d1d1539ea95cf342d.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/b58a5cc96cc63ae8d60508af505904a9b0d8552edae3ad8d1d1539ea95cf342d.jpg)

![b637cd61b7589d41ffb375418129ebac61cecfaf6ef3ce6ded434eb591b7c42f.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/b637cd61b7589d41ffb375418129ebac61cecfaf6ef3ce6ded434eb591b7c42f.jpg)

![b7dd0323aae25b24ed20f0eb344c5828d5c77fed36f2cfd0ef1c218b0dac665b.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/b7dd0323aae25b24ed20f0eb344c5828d5c77fed36f2cfd0ef1c218b0dac665b.jpg)

![bac9aa39f7df164c6ee610e15ffa278a6cbd06f1d72934c9c193643071d72148.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/bac9aa39f7df164c6ee610e15ffa278a6cbd06f1d72934c9c193643071d72148.jpg)

![bbc852fed88858b11cfd50b745c6442b70a08d9a19ddd2834e5bd9da66bae6d2.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/bbc852fed88858b11cfd50b745c6442b70a08d9a19ddd2834e5bd9da66bae6d2.jpg)

![c4dea872b145f4382011efc94f8e9e0e35ba7494df1b9797f38078d8bce6084b.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/c4dea872b145f4382011efc94f8e9e0e35ba7494df1b9797f38078d8bce6084b.jpg)

![cfb5917842a7b4d9d8b09872dede7b7222cb6398dcab4044a3f910055f9245b1.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/cfb5917842a7b4d9d8b09872dede7b7222cb6398dcab4044a3f910055f9245b1.jpg)

![d09d3bb06b54469255cef3f8ce951ea20251b1f05ca49db4a55805dd50b4fb89.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/d09d3bb06b54469255cef3f8ce951ea20251b1f05ca49db4a55805dd50b4fb89.jpg)

![d794ed078655fb9f866b0fd946d353bf06c6501f9f597b0a826127f72585885d.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/d794ed078655fb9f866b0fd946d353bf06c6501f9f597b0a826127f72585885d.jpg)

![d8bd8c0cc51e9b081b27edf3bfe0416b02e09713cb02d0bac6118c6a65afa54c.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/d8bd8c0cc51e9b081b27edf3bfe0416b02e09713cb02d0bac6118c6a65afa54c.jpg)

![eb647aec34f2b44fb26af76ac5202a3d2101b17ebd3e38f828345ce079ca1b25.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/eb647aec34f2b44fb26af76ac5202a3d2101b17ebd3e38f828345ce079ca1b25.jpg)

![ecfa553d3806acbf1b58a1060b00da4f4ee92894bc30dbb22515cccec0f07621.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/ecfa553d3806acbf1b58a1060b00da4f4ee92894bc30dbb22515cccec0f07621.jpg)

![f2f6b01dc48eace4f49ed95eae1bf103aa988e1303abd53a030e40f35f86ba86.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/f2f6b01dc48eace4f49ed95eae1bf103aa988e1303abd53a030e40f35f86ba86.jpg)

![fa6600e79e4fd6c9ee276bf854ae58584db5d1f31e6a3944471ad61b03ffd61a.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/fa6600e79e4fd6c9ee276bf854ae58584db5d1f31e6a3944471ad61b03ffd61a.jpg)

![fb5d2860aacb98eed9ad5eb5500b7083844b6d212c8962ab51209a700c3e2874.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/images/fb5d2860aacb98eed9ad5eb5500b7083844b6d212c8962ab51209a700c3e2874.jpg)

### Tables

![255c9be19fb86a19fcbe942b4a314c3bf08e0710a5eccbfedf723e4487f73a96.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/tables/255c9be19fb86a19fcbe942b4a314c3bf08e0710a5eccbfedf723e4487f73a96.jpg)

![3cb434e0a51325d3196c2b4f846d9179ff517bd4d34610e8e7c6d7682531a60d.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/tables/3cb434e0a51325d3196c2b4f846d9179ff517bd4d34610e8e7c6d7682531a60d.jpg)

![3e8a7f53c21c482a43a678f834ead27037cfac3bda65b5492dc28e77adbe7d28.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/tables/3e8a7f53c21c482a43a678f834ead27037cfac3bda65b5492dc28e77adbe7d28.jpg)

![59b84e5686152a5f9c7124540331e5b5f49d9fe4a2cb339f592d27fbc4d28ea1.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/tables/59b84e5686152a5f9c7124540331e5b5f49d9fe4a2cb339f592d27fbc4d28ea1.jpg)

![622edbdc8933d9de5844f2ec3b5b2567cc9e4ada1112488097dd650bc3bd713c.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/tables/622edbdc8933d9de5844f2ec3b5b2567cc9e4ada1112488097dd650bc3bd713c.jpg)

![9620879648e1312e36803b30825419322244ccbb284b2d3f00e9d5866cdd4952.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/tables/9620879648e1312e36803b30825419322244ccbb284b2d3f00e9d5866cdd4952.jpg)

![9bb566dc1e4ce40cdbf12bcc5e23fd8c7613c61186d0efadaacf488d725e7b82.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/tables/9bb566dc1e4ce40cdbf12bcc5e23fd8c7613c61186d0efadaacf488d725e7b82.jpg)

![a2e769b83e3dd68009dd7587c784d57600606c9b5f65ee078987fdb7a0612e7c.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/tables/a2e769b83e3dd68009dd7587c784d57600606c9b5f65ee078987fdb7a0612e7c.jpg)

![e4baa2ebbbcead2aec54d146b82c71ed16c4d296e71ee6af34f460867e186b3d.jpg](../iclr_results/155_STAR_ Synthesis of Tailored Architectures/tables/e4baa2ebbbcead2aec54d146b82c71ed16c4d296e71ee6af34f460867e186b3d.jpg)

## MMIE: Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models


### Images

![0128a26ccf1513ef104237e21f9e3eaabe773e02e5ae5adc03698cf5188ab2bf.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/0128a26ccf1513ef104237e21f9e3eaabe773e02e5ae5adc03698cf5188ab2bf.jpg)

![115688a91108a600d7a38de0a2860fd6a0292e044892134712abbe8b7e5db1ed.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/115688a91108a600d7a38de0a2860fd6a0292e044892134712abbe8b7e5db1ed.jpg)

![1a943e49f0bdc3b7618fee3e7998223f97ad3390b523c08e4ccd6633abc193fa.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/1a943e49f0bdc3b7618fee3e7998223f97ad3390b523c08e4ccd6633abc193fa.jpg)

![1e4ebe1b443ccd60a561d5221f73503f74060e2adf24aca6f533969708aaa7f9.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/1e4ebe1b443ccd60a561d5221f73503f74060e2adf24aca6f533969708aaa7f9.jpg)

![208ff0ed486c3958937a8e648b2f424f62ef8f0679d5d93b690225f70d71e683.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/208ff0ed486c3958937a8e648b2f424f62ef8f0679d5d93b690225f70d71e683.jpg)

![2fd8cd169accffdee0824d8fd69a1e2985639dca49fa88c0b694243937cd1354.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/2fd8cd169accffdee0824d8fd69a1e2985639dca49fa88c0b694243937cd1354.jpg)

![4742e74bf72b76db755a290228e63aeac198c7ddd6da3915957fdb106b6294e3.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/4742e74bf72b76db755a290228e63aeac198c7ddd6da3915957fdb106b6294e3.jpg)

![5afb70018432109a9a828e06fac8e946081c877028a49e8323d4fc4028e97d39.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/5afb70018432109a9a828e06fac8e946081c877028a49e8323d4fc4028e97d39.jpg)

![64965719de41f7a25b275798ced37ba99031868ef998d7edcb2000ff7a6636bd.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/64965719de41f7a25b275798ced37ba99031868ef998d7edcb2000ff7a6636bd.jpg)

![6772126dcf6abc19efc299e95b8d983602778a7f5d00a2b057788d72587c1fca.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/6772126dcf6abc19efc299e95b8d983602778a7f5d00a2b057788d72587c1fca.jpg)

![6dab56f9ccdd9126db5e136a8fca21430ea46ed959ae5df0c20fddd6968c1e7c.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/6dab56f9ccdd9126db5e136a8fca21430ea46ed959ae5df0c20fddd6968c1e7c.jpg)

![6e85bbe43e955d4dd6e3bcd8adb232e8108f4a8be1eb551172e600346925277b.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/6e85bbe43e955d4dd6e3bcd8adb232e8108f4a8be1eb551172e600346925277b.jpg)

![6ebbeea09dc2e254923cf912101846f1996d48fd1b0c16a2505c29dd23e5c9aa.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/6ebbeea09dc2e254923cf912101846f1996d48fd1b0c16a2505c29dd23e5c9aa.jpg)

![8d893e1c4143a0b7169703741c87e7e3cef61c801737e66f86d04787aeb0a562.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/8d893e1c4143a0b7169703741c87e7e3cef61c801737e66f86d04787aeb0a562.jpg)

![93570a6f1882a8f976b7427e6ffe2ccdcc5e7369aee61f560a72cd6f4c84cc07.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/93570a6f1882a8f976b7427e6ffe2ccdcc5e7369aee61f560a72cd6f4c84cc07.jpg)

![93dec7175e20d66cf1c705ab27bbba26fea8b7ba1d4da754de40b6a8a28d53d7.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/93dec7175e20d66cf1c705ab27bbba26fea8b7ba1d4da754de40b6a8a28d53d7.jpg)

![946f7f3b5a64369e9da0881f67b065238734929679b6b091eeaa2489cfe61104.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/946f7f3b5a64369e9da0881f67b065238734929679b6b091eeaa2489cfe61104.jpg)

![9d12203036c0d73ed877567bb7f30c60960080ceeef5bd2b1abae1b0a6204320.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/9d12203036c0d73ed877567bb7f30c60960080ceeef5bd2b1abae1b0a6204320.jpg)

![9fa1f9488dc027587bb74777678808e2bd88ca3411a5bcea44c7a14870e2ae44.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/9fa1f9488dc027587bb74777678808e2bd88ca3411a5bcea44c7a14870e2ae44.jpg)

![a1e1e6e0624434b97689b60821274a78744210f441fd36930f3ae553ac213622.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/a1e1e6e0624434b97689b60821274a78744210f441fd36930f3ae553ac213622.jpg)

![ad7fc09bb0775730c8cc8e032815b4efa20a917a6e6717508d8d2992b4570436.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/ad7fc09bb0775730c8cc8e032815b4efa20a917a6e6717508d8d2992b4570436.jpg)

![adbf5912274f8d44cc3194ae22f38377fae542e723740e1e48d6131819f930ce.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/adbf5912274f8d44cc3194ae22f38377fae542e723740e1e48d6131819f930ce.jpg)

![b0320712898ce0ce38515291cfb218d53bdac26508d7b3c040c5942341513da3.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/b0320712898ce0ce38515291cfb218d53bdac26508d7b3c040c5942341513da3.jpg)

![d03f811871a15a0b9b897cac0595f2462add2d6ab8bae1ce34227d6755531174.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/d03f811871a15a0b9b897cac0595f2462add2d6ab8bae1ce34227d6755531174.jpg)

![d4e697dfd67e55bcf351e7693c61d251e8a6b5e30e9f3af9a08ed5719f8f11b7.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/d4e697dfd67e55bcf351e7693c61d251e8a6b5e30e9f3af9a08ed5719f8f11b7.jpg)

![d518900a98886ee5d86510fe9f9a36ee524193bcc40e7ced96b328eaf83c72e7.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/d518900a98886ee5d86510fe9f9a36ee524193bcc40e7ced96b328eaf83c72e7.jpg)

![d62c3eee19ece7542ee957b18152ecd21970b0ad021395597c4cb67325e9ebc5.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/d62c3eee19ece7542ee957b18152ecd21970b0ad021395597c4cb67325e9ebc5.jpg)

![e1d1e912600176a203c684b7a52c8c255bf7cca9d447317945bc3cdcdf4b8127.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/e1d1e912600176a203c684b7a52c8c255bf7cca9d447317945bc3cdcdf4b8127.jpg)

![e6648748662ec6e230b74d178b242f92783a36bfa2cb3f6199ef68b674e1e88a.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/e6648748662ec6e230b74d178b242f92783a36bfa2cb3f6199ef68b674e1e88a.jpg)

![e73123753eff93cb609bd57e006dec60864c40df3c3dcb86f700a9f04a115f30.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/e73123753eff93cb609bd57e006dec60864c40df3c3dcb86f700a9f04a115f30.jpg)

![f20181785c7a5e1a1a529a8a1d68924ec6f7b51bc1ca6892a9dd36f5c52ebb7b.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/f20181785c7a5e1a1a529a8a1d68924ec6f7b51bc1ca6892a9dd36f5c52ebb7b.jpg)

![f668cc471e997a26b4a250e810f0d5c6b845e1a34c7b4924e97472c080c12ec2.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/images/f668cc471e997a26b4a250e810f0d5c6b845e1a34c7b4924e97472c080c12ec2.jpg)

### Tables

![0ac4e1fef55fa8bfc3aee0cfbb4c35ebf55d8b986a065732b938fd79bbf554f3.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/tables/0ac4e1fef55fa8bfc3aee0cfbb4c35ebf55d8b986a065732b938fd79bbf554f3.jpg)

![5c436b54d9c7a27100e5ae656c4844a487935867e2f24364ffadabccd0b4eeab.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/tables/5c436b54d9c7a27100e5ae656c4844a487935867e2f24364ffadabccd0b4eeab.jpg)

![672c378123c5ab79a69bc6858c41cdde95615397c58bc530b5210b35e608cd79.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/tables/672c378123c5ab79a69bc6858c41cdde95615397c58bc530b5210b35e608cd79.jpg)

![68555217b74be3058e468d38f2286568a1b0f6e09c7a58d9b528c2ae9a1da4a9.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/tables/68555217b74be3058e468d38f2286568a1b0f6e09c7a58d9b528c2ae9a1da4a9.jpg)

![74267223dd9d290dd1b962839a71e8ee72930a0e89791d0e725b28675d67c5e7.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/tables/74267223dd9d290dd1b962839a71e8ee72930a0e89791d0e725b28675d67c5e7.jpg)

![760d6edbf3c40192c39f520d1b1a3becb0ac76ec15c39d051cbeab8317443df7.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/tables/760d6edbf3c40192c39f520d1b1a3becb0ac76ec15c39d051cbeab8317443df7.jpg)

![9c161ddad3eaeb0db23f6bc1b265b871a788a98ffaad4cb27b6c546a7813b4fd.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/tables/9c161ddad3eaeb0db23f6bc1b265b871a788a98ffaad4cb27b6c546a7813b4fd.jpg)

![ae6d48bd5cb8ead95692e794a3933b4f94a1835a002d02b137b59e2f233cb835.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/tables/ae6d48bd5cb8ead95692e794a3933b4f94a1835a002d02b137b59e2f233cb835.jpg)

![b5e0ee915ace457452143e9b18964d22f1db019dc46e6c389579b926929fcf58.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/tables/b5e0ee915ace457452143e9b18964d22f1db019dc46e6c389579b926929fcf58.jpg)

![e4264b34110e87844365007aa88b64dc5927a1db01d3aeafef2a8891bef4a8a1.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/tables/e4264b34110e87844365007aa88b64dc5927a1db01d3aeafef2a8891bef4a8a1.jpg)

![e7bb32e6ea9597e91d50b666fc63396afa6d10103c3989c6bab3b9bb519ceb22.jpg](../iclr_results/156_MMIE_ Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models/tables/e7bb32e6ea9597e91d50b666fc63396afa6d10103c3989c6bab3b9bb519ceb22.jpg)

## SAM 2: Segment Anything in Images and Videos


### Images

![15fb36a276a0fc0c7cf88d73841283581486e2fdf33a76fa6a5f6fdf964c3f08.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/images/15fb36a276a0fc0c7cf88d73841283581486e2fdf33a76fa6a5f6fdf964c3f08.jpg)

![286d47d4b7b7969f15bbdfa0d5c7690a96ebce9f3c354092119d9928a7e8f425.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/images/286d47d4b7b7969f15bbdfa0d5c7690a96ebce9f3c354092119d9928a7e8f425.jpg)

![2ba226c16a52e648d7deeb9706cc680e564c32b49564f616df42504fdd876a03.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/images/2ba226c16a52e648d7deeb9706cc680e564c32b49564f616df42504fdd876a03.jpg)

![2c26add040729d27d777492ddee09af9f65c62212f1e20e791458102e73815c6.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/images/2c26add040729d27d777492ddee09af9f65c62212f1e20e791458102e73815c6.jpg)

![3b281f16f57fbdb299150df896ae57cb72e2fa7d359fe4f0b0548c4b259e03e5.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/images/3b281f16f57fbdb299150df896ae57cb72e2fa7d359fe4f0b0548c4b259e03e5.jpg)

![4054e24126fbd54e45a916ad1b0b9c25b39426450bd77e7d38a4b14c60d62437.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/images/4054e24126fbd54e45a916ad1b0b9c25b39426450bd77e7d38a4b14c60d62437.jpg)

![592036a3eae225f464fe707739b68ab28b15cf202ce6640f8466ced4d275e92b.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/images/592036a3eae225f464fe707739b68ab28b15cf202ce6640f8466ced4d275e92b.jpg)

![6400fa498f8e7afc4c33201f0b75ae810d58e547d1f531d50df708503607d0f0.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/images/6400fa498f8e7afc4c33201f0b75ae810d58e547d1f531d50df708503607d0f0.jpg)

![6e61673879af4cc09e6e0ddb78e463b60dcfa8cfaf246d74cd7ccb0527fd4a1e.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/images/6e61673879af4cc09e6e0ddb78e463b60dcfa8cfaf246d74cd7ccb0527fd4a1e.jpg)

![747f20d91334090f470ddf3455caa1445f025eaf18ff30fd0771000d4d8c522b.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/images/747f20d91334090f470ddf3455caa1445f025eaf18ff30fd0771000d4d8c522b.jpg)

![75370ff1a6e83f1e19897d37ebff083aebffc30e27c96c113c7c2b053766caf8.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/images/75370ff1a6e83f1e19897d37ebff083aebffc30e27c96c113c7c2b053766caf8.jpg)

![7d8d63fbe94631704add4d73027cc4233b5a91eeca90594fa236f8e1b0fa701e.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/images/7d8d63fbe94631704add4d73027cc4233b5a91eeca90594fa236f8e1b0fa701e.jpg)

![8a6a242efc3d91706b723408a407e31d32d8baf0541eee00347d49d102e2f408.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/images/8a6a242efc3d91706b723408a407e31d32d8baf0541eee00347d49d102e2f408.jpg)

![9093f69954035184d03f017b6a4f57111a2d5b6e756607d75e8f92ed80a45498.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/images/9093f69954035184d03f017b6a4f57111a2d5b6e756607d75e8f92ed80a45498.jpg)

![98cf0912ace4c391424b817659b000cd7fc602a9656dfd5fc5785e6f0493a703.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/images/98cf0912ace4c391424b817659b000cd7fc602a9656dfd5fc5785e6f0493a703.jpg)

![9ce86cc90957421d1cd0dc097647c83e93d9c60a5a7d5408817653a252de08de.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/images/9ce86cc90957421d1cd0dc097647c83e93d9c60a5a7d5408817653a252de08de.jpg)

![9e9c40657361d4cb0ede9fd01c995cab528f07b0681a4edb2d8c16459c5a141c.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/images/9e9c40657361d4cb0ede9fd01c995cab528f07b0681a4edb2d8c16459c5a141c.jpg)

![ac5a68d2175fd8f542bcba5fbf270105db39c4601ac4929c72e00f4338e4bc3f.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/images/ac5a68d2175fd8f542bcba5fbf270105db39c4601ac4929c72e00f4338e4bc3f.jpg)

![c5cf75e1b6e915eb8594534231108c24604d7fe5f082bd58f22075117ed6a43d.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/images/c5cf75e1b6e915eb8594534231108c24604d7fe5f082bd58f22075117ed6a43d.jpg)

### Tables

![0164a5f77d491e038a379f8d7b55cd43fccb12f9cb11387c46368e974cefa2d1.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/0164a5f77d491e038a379f8d7b55cd43fccb12f9cb11387c46368e974cefa2d1.jpg)

![04b4188be7a263f8fc20580404db708b0be4a0534d2c45c9c14f3822460b36a8.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/04b4188be7a263f8fc20580404db708b0be4a0534d2c45c9c14f3822460b36a8.jpg)

![20a1f45aee23c5b0c9aca803834e030ca701038ad3a2ec701f7439e5193010b5.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/20a1f45aee23c5b0c9aca803834e030ca701038ad3a2ec701f7439e5193010b5.jpg)

![30051cf9e4fdbb9ff2ea64436ce7daa8c77c70fdd14d01245f3b9c3aaf949679.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/30051cf9e4fdbb9ff2ea64436ce7daa8c77c70fdd14d01245f3b9c3aaf949679.jpg)

![34d0e9716ee3e7df1726cf54771a905434178f805fb6d8408c4a226882e68007.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/34d0e9716ee3e7df1726cf54771a905434178f805fb6d8408c4a226882e68007.jpg)

![3cb5e4f98064551ba971ba960015ca0f97778cecd4326a1104bf508e498e6d13.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/3cb5e4f98064551ba971ba960015ca0f97778cecd4326a1104bf508e498e6d13.jpg)

![433e2fc201fba31d38e3c36f7ff12a4c141774dccd8de22c8aead55d2aeb92d5.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/433e2fc201fba31d38e3c36f7ff12a4c141774dccd8de22c8aead55d2aeb92d5.jpg)

![4e1e4d282bf94be5d49da79cf8baaead5f92067143fb454d794a1bd0e2cb78d8.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/4e1e4d282bf94be5d49da79cf8baaead5f92067143fb454d794a1bd0e2cb78d8.jpg)

![612d66f162b3119288ee654ff5e4228718a8f8532c650348e32c9c03b1c3644e.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/612d66f162b3119288ee654ff5e4228718a8f8532c650348e32c9c03b1c3644e.jpg)

![81873f90bfe6b05b8a24a3bc29a24298f951392358cd20ff85682206205f92f7.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/81873f90bfe6b05b8a24a3bc29a24298f951392358cd20ff85682206205f92f7.jpg)

![8a858aeeea2643f02ea5a87376b06560f4f0f069e1e2b0029a7904a66ba6fd2a.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/8a858aeeea2643f02ea5a87376b06560f4f0f069e1e2b0029a7904a66ba6fd2a.jpg)

![93e78d5f1f2f0d8493728ced5f0c377f8b2196ab4380f0fa55b1e6b78d562018.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/93e78d5f1f2f0d8493728ced5f0c377f8b2196ab4380f0fa55b1e6b78d562018.jpg)

![a0df2ebed21888de68fb79b03b05f0dbd99e3e8f5e693511e8fbd943f9298c13.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/a0df2ebed21888de68fb79b03b05f0dbd99e3e8f5e693511e8fbd943f9298c13.jpg)

![a5925c9ca3c861adee1d8ca5753ae306e7180dbd2c0e1b8627a8647f4984a86f.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/a5925c9ca3c861adee1d8ca5753ae306e7180dbd2c0e1b8627a8647f4984a86f.jpg)

![a59d358fa91d9514478c222e778ecf3052767dfc060f615a2bec64550062d942.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/a59d358fa91d9514478c222e778ecf3052767dfc060f615a2bec64550062d942.jpg)

![a90fa7c4f9a4b26bbd7d915eded85a660313b2c57af2d39fe5ab94e817ded9ca.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/a90fa7c4f9a4b26bbd7d915eded85a660313b2c57af2d39fe5ab94e817ded9ca.jpg)

![c2bb4b0f0b5f9c7c0e97cfdcd799edff580955e994034ff0d3ddd2c99705ee3a.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/c2bb4b0f0b5f9c7c0e97cfdcd799edff580955e994034ff0d3ddd2c99705ee3a.jpg)

![c91ae9e76ab215b04803239d69bd01ce7f6515b9747e6d5e076911b94680cb60.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/c91ae9e76ab215b04803239d69bd01ce7f6515b9747e6d5e076911b94680cb60.jpg)

![d9ba055acb9a2fac5baa2b992d0d4c5ab801b490997c23f2402ba3c1f5b1a418.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/d9ba055acb9a2fac5baa2b992d0d4c5ab801b490997c23f2402ba3c1f5b1a418.jpg)

![db9e51666eac6effae96b2586b248c150aa19e3f592bab72348987d34c284bc7.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/db9e51666eac6effae96b2586b248c150aa19e3f592bab72348987d34c284bc7.jpg)

![dc7fecf5db9338aeb19c8ff84a1fb96704e0491d8e041a453489791ee014f30f.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/dc7fecf5db9338aeb19c8ff84a1fb96704e0491d8e041a453489791ee014f30f.jpg)

![e621968d1c4731eda95dc093640b7c0e0c7441b1a029f24591245068ed719716.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/e621968d1c4731eda95dc093640b7c0e0c7441b1a029f24591245068ed719716.jpg)

![ee225fb75e2ac159e88dd6a82bfd4a8762a7393831ebf68b6ab51eff630280a6.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/ee225fb75e2ac159e88dd6a82bfd4a8762a7393831ebf68b6ab51eff630280a6.jpg)

![f7c50db361128882547ba26e487279aa21695e0fe5eac066e65de0d75be41c66.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/f7c50db361128882547ba26e487279aa21695e0fe5eac066e65de0d75be41c66.jpg)

![fdeabe352cb19bd9ac37acbe05957f7e5e113f0e1518699a98cb65829abde661.jpg](../iclr_results/157_SAM 2_ Segment Anything in Images and Videos/tables/fdeabe352cb19bd9ac37acbe05957f7e5e113f0e1518699a98cb65829abde661.jpg)

## Data Shapley in One Training Run


### Images

![01e2f739a32269c9ae92675a5013c11084a5e2be0192c66a9cd4c183a5b95d22.jpg](../iclr_results/158_Data Shapley in One Training Run/images/01e2f739a32269c9ae92675a5013c11084a5e2be0192c66a9cd4c183a5b95d22.jpg)

![1688559749f90714f63b9d4aa86a7af6a57c9f7fe44fdb492431c4ce9a304654.jpg](../iclr_results/158_Data Shapley in One Training Run/images/1688559749f90714f63b9d4aa86a7af6a57c9f7fe44fdb492431c4ce9a304654.jpg)

![1c342146253d56f802a0d2fda000b4673b9021db9bc17e7395af1037cfa4e23e.jpg](../iclr_results/158_Data Shapley in One Training Run/images/1c342146253d56f802a0d2fda000b4673b9021db9bc17e7395af1037cfa4e23e.jpg)

![682f38864ccf4b5c86eb1862ac60cc3deb1187f906f34d39230be926e2c6f2dc.jpg](../iclr_results/158_Data Shapley in One Training Run/images/682f38864ccf4b5c86eb1862ac60cc3deb1187f906f34d39230be926e2c6f2dc.jpg)

![8238f3fb9d11ed305440c29c84dda87a92abcfca49fad81000cb6062e149fab0.jpg](../iclr_results/158_Data Shapley in One Training Run/images/8238f3fb9d11ed305440c29c84dda87a92abcfca49fad81000cb6062e149fab0.jpg)

![8478ffdce988121741a113ae3c7f6947131193b42b0211642da635fd93767d84.jpg](../iclr_results/158_Data Shapley in One Training Run/images/8478ffdce988121741a113ae3c7f6947131193b42b0211642da635fd93767d84.jpg)

![8a1318095f216df0af117c397265e6489da9735bf668e6194a2e762520b770c4.jpg](../iclr_results/158_Data Shapley in One Training Run/images/8a1318095f216df0af117c397265e6489da9735bf668e6194a2e762520b770c4.jpg)

![9298756d56f4f099d0ccb5fdec4687ca179284ff8eae8c79960aa2eeca0fc928.jpg](../iclr_results/158_Data Shapley in One Training Run/images/9298756d56f4f099d0ccb5fdec4687ca179284ff8eae8c79960aa2eeca0fc928.jpg)

![98b7e9cc751321562db46a3af5f249ed6d75459065dfa6b640fffdd5c9d5b8c9.jpg](../iclr_results/158_Data Shapley in One Training Run/images/98b7e9cc751321562db46a3af5f249ed6d75459065dfa6b640fffdd5c9d5b8c9.jpg)

![9b246d83c76b9d760b20f188c4a360050e4206f59d8b9e78e7788a93eb29fced.jpg](../iclr_results/158_Data Shapley in One Training Run/images/9b246d83c76b9d760b20f188c4a360050e4206f59d8b9e78e7788a93eb29fced.jpg)

![c85f9950b24346e5312c448a92265ac0239ca70be05df7c50369e6acc4fff9c8.jpg](../iclr_results/158_Data Shapley in One Training Run/images/c85f9950b24346e5312c448a92265ac0239ca70be05df7c50369e6acc4fff9c8.jpg)

![e8f726b1311d0045e03ceeca3653cd33dda93503a62e89f9aba1a4305add1d28.jpg](../iclr_results/158_Data Shapley in One Training Run/images/e8f726b1311d0045e03ceeca3653cd33dda93503a62e89f9aba1a4305add1d28.jpg)

### Tables

![204a86435800dd1dd32298260fb94005ea314cbd1b8e13d240e8dcfb996318d4.jpg](../iclr_results/158_Data Shapley in One Training Run/tables/204a86435800dd1dd32298260fb94005ea314cbd1b8e13d240e8dcfb996318d4.jpg)

![3c719dabda7c326bb98fd9403fd107aea96de730d6fb5b0c4265f71f9627cec8.jpg](../iclr_results/158_Data Shapley in One Training Run/tables/3c719dabda7c326bb98fd9403fd107aea96de730d6fb5b0c4265f71f9627cec8.jpg)

![5221639afb3551a28c45119f35b15bf736d16150a9cf36789afb1cad4fbdb323.jpg](../iclr_results/158_Data Shapley in One Training Run/tables/5221639afb3551a28c45119f35b15bf736d16150a9cf36789afb1cad4fbdb323.jpg)

![894e3821014c5355b16c6c98c8af61dcc7a7afaac1eb250c02b9d9b03e8fc07f.jpg](../iclr_results/158_Data Shapley in One Training Run/tables/894e3821014c5355b16c6c98c8af61dcc7a7afaac1eb250c02b9d9b03e8fc07f.jpg)

![9d871b6587d04e34917d1bb59ded5e92698b4c7b1cb0c25d7d47b832a5a86df1.jpg](../iclr_results/158_Data Shapley in One Training Run/tables/9d871b6587d04e34917d1bb59ded5e92698b4c7b1cb0c25d7d47b832a5a86df1.jpg)

![e5fe2cff2de742f42220f2195f2a2236668c6af9f72ece62385f353b0868d8ed.jpg](../iclr_results/158_Data Shapley in One Training Run/tables/e5fe2cff2de742f42220f2195f2a2236668c6af9f72ece62385f353b0868d8ed.jpg)

## Oscillatory State-Space Models


### Images

![4eec9a73ffcb24c7db1ddee50fbc6716b687cf66e61d4398505d3f2674e5b6a2.jpg](../iclr_results/159_Oscillatory State-Space Models/images/4eec9a73ffcb24c7db1ddee50fbc6716b687cf66e61d4398505d3f2674e5b6a2.jpg)

![9fb76e6a9db20f295d02530acf4b5f27da2da066a7071d2c63379449cccacb3b.jpg](../iclr_results/159_Oscillatory State-Space Models/images/9fb76e6a9db20f295d02530acf4b5f27da2da066a7071d2c63379449cccacb3b.jpg)

![ccf7fd4016d831916450653203f6f415b13d1d3a539f189b31c7a70951b229e1.jpg](../iclr_results/159_Oscillatory State-Space Models/images/ccf7fd4016d831916450653203f6f415b13d1d3a539f189b31c7a70951b229e1.jpg)

### Tables

![2b9ef9333f1e7b631ecef8c59ee6ef2615b07054ddde3df7bf46d730cb4de3a4.jpg](../iclr_results/159_Oscillatory State-Space Models/tables/2b9ef9333f1e7b631ecef8c59ee6ef2615b07054ddde3df7bf46d730cb4de3a4.jpg)

![5ec9b00ade81d32d6d564849e6465e07d3f025e8bcd306aa79873a8d157730fc.jpg](../iclr_results/159_Oscillatory State-Space Models/tables/5ec9b00ade81d32d6d564849e6465e07d3f025e8bcd306aa79873a8d157730fc.jpg)

![9cc363559fe3a7dc1522bd13120304b0d3ea0e325e80effe910cfd8fcba61ebc.jpg](../iclr_results/159_Oscillatory State-Space Models/tables/9cc363559fe3a7dc1522bd13120304b0d3ea0e325e80effe910cfd8fcba61ebc.jpg)

![b352f114fd0e63d36d52fe4fab2a1398a01d9fd08f9afda59dec25df4c408698.jpg](../iclr_results/159_Oscillatory State-Space Models/tables/b352f114fd0e63d36d52fe4fab2a1398a01d9fd08f9afda59dec25df4c408698.jpg)

![cba891c8994ddae9d6661e752fccf63028f3db3fe06f3327c82448458d68d013.jpg](../iclr_results/159_Oscillatory State-Space Models/tables/cba891c8994ddae9d6661e752fccf63028f3db3fe06f3327c82448458d68d013.jpg)

![d496a43454ac9f4e57a4229ec4a17b54894121596b889368461b6bdde4d6b3df.jpg](../iclr_results/159_Oscillatory State-Space Models/tables/d496a43454ac9f4e57a4229ec4a17b54894121596b889368461b6bdde4d6b3df.jpg)

![d9b8b5a9fa2a6c0ea9009c2d8053810b134d89839582cdd8c159b0618e9d22a2.jpg](../iclr_results/159_Oscillatory State-Space Models/tables/d9b8b5a9fa2a6c0ea9009c2d8053810b134d89839582cdd8c159b0618e9d22a2.jpg)

## Restructuring Vector Quantization with the Rotation Trick


### Images

![03e288dd84f8958ce7b880a53a75b24a19ed8b4e2e7539487282d6e4ec5ce903.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/03e288dd84f8958ce7b880a53a75b24a19ed8b4e2e7539487282d6e4ec5ce903.jpg)

![1feae79487618ddf512bd824448f13bac6484564ec6c9c7fd8204ad8d0c29c84.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/1feae79487618ddf512bd824448f13bac6484564ec6c9c7fd8204ad8d0c29c84.jpg)

![20f31ea4e68b177b2af8977e6160143b692aea99c51346559e6111e6e43dacf9.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/20f31ea4e68b177b2af8977e6160143b692aea99c51346559e6111e6e43dacf9.jpg)

![32b20ab0355bf43b447e45d6018506836cc5a6e004c2d9f369dad0fcdc783ec6.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/32b20ab0355bf43b447e45d6018506836cc5a6e004c2d9f369dad0fcdc783ec6.jpg)

![3bbe18e6c22b7cf2c6387ee2c95ae469828117697bf29aebbfd20bdec095ad7f.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/3bbe18e6c22b7cf2c6387ee2c95ae469828117697bf29aebbfd20bdec095ad7f.jpg)

![3eb7d75e45a94dbd186f074599be265340f9800817b5529468c3ce1be467835b.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/3eb7d75e45a94dbd186f074599be265340f9800817b5529468c3ce1be467835b.jpg)

![49b527882df359ed0d9928298660ac6a8b82f9275a11cd0e3ae66e7c172cd04c.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/49b527882df359ed0d9928298660ac6a8b82f9275a11cd0e3ae66e7c172cd04c.jpg)

![541f1a0ff15f385109cea9c8ee8ba9f581b68a3c74224cbf838237061038ff3c.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/541f1a0ff15f385109cea9c8ee8ba9f581b68a3c74224cbf838237061038ff3c.jpg)

![6990941e6f6ab7fa462d441b99d3865ce55a7ae3a2499042249cdadecd06f9ff.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/6990941e6f6ab7fa462d441b99d3865ce55a7ae3a2499042249cdadecd06f9ff.jpg)

![710044e32a73e6118cd54c8741a55bfa14aa82d3499fa054ad2b2823333ac134.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/710044e32a73e6118cd54c8741a55bfa14aa82d3499fa054ad2b2823333ac134.jpg)

![88355bd853a2ccfb2e3801376189155931061148cfcfd84bfeb1a9920e4683f7.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/88355bd853a2ccfb2e3801376189155931061148cfcfd84bfeb1a9920e4683f7.jpg)

![8e3931b7627a1ba6700adfa0c23fad91ba364338cccfe2d2ef24b2518911cc41.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/8e3931b7627a1ba6700adfa0c23fad91ba364338cccfe2d2ef24b2518911cc41.jpg)

![8e60e783dff32ba7be8450ab64dab8f72745bb4f86702ca81b6eaed6c51dba11.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/8e60e783dff32ba7be8450ab64dab8f72745bb4f86702ca81b6eaed6c51dba11.jpg)

![9187357d404dba4fabdb959b8cd518a052f7c4024a85ae9fcd09ace1574e544f.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/9187357d404dba4fabdb959b8cd518a052f7c4024a85ae9fcd09ace1574e544f.jpg)

![962e4ad86ab812fd222fe531d290a936ab580ef2eea4768ab366393478849fa9.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/962e4ad86ab812fd222fe531d290a936ab580ef2eea4768ab366393478849fa9.jpg)

![c66ac648a1e5282a93888f71e7230ec4adbfa34c32c1c32e0298b981b5a7c9ae.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/c66ac648a1e5282a93888f71e7230ec4adbfa34c32c1c32e0298b981b5a7c9ae.jpg)

![ec24147a39ce1485899bff68a77ab19990ff9be0e9e4f2e42c54c120f7761a70.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/ec24147a39ce1485899bff68a77ab19990ff9be0e9e4f2e42c54c120f7761a70.jpg)

![f11e4cb489829c5aa6cadac761122914d9634ed723677b570a9a9057ed3a22ac.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/f11e4cb489829c5aa6cadac761122914d9634ed723677b570a9a9057ed3a22ac.jpg)

![f27815022b19b90d2b172ad3561e88ddcd8f183ed9fed26ad8a3a1aa66c551fa.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/f27815022b19b90d2b172ad3561e88ddcd8f183ed9fed26ad8a3a1aa66c551fa.jpg)

![f9a2d5d8f05bf7f7e9bdd199960b17a92421e3c2ea807a8aaa83f9a822db6bcb.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/f9a2d5d8f05bf7f7e9bdd199960b17a92421e3c2ea807a8aaa83f9a822db6bcb.jpg)

![fa0365bef1b5584fde79f320eb31bfcbb2db400cfdc8e6ddc4cce41e0c581b3b.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/images/fa0365bef1b5584fde79f320eb31bfcbb2db400cfdc8e6ddc4cce41e0c581b3b.jpg)

### Tables

![06f6f7a5ff2c4a9755ca2d78f5e9f68b78f5052eb6b54fa19bc99de4730dc575.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/tables/06f6f7a5ff2c4a9755ca2d78f5e9f68b78f5052eb6b54fa19bc99de4730dc575.jpg)

![119a1c6d3416510b14230e64bd643e19338d9754b350f02a72da46d3f928e15d.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/tables/119a1c6d3416510b14230e64bd643e19338d9754b350f02a72da46d3f928e15d.jpg)

![266011ea677388d2c2ac566351ee877960cac1d9fe8f3ed54740986f2f3c4da0.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/tables/266011ea677388d2c2ac566351ee877960cac1d9fe8f3ed54740986f2f3c4da0.jpg)

![3059ff2580aa96b443c8a5ea093b9f78ef67b988543bd75b667236ace0244543.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/tables/3059ff2580aa96b443c8a5ea093b9f78ef67b988543bd75b667236ace0244543.jpg)

![35acb11496a16967aa790375fe161e980876b393d8e02b0124f101fb078239d3.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/tables/35acb11496a16967aa790375fe161e980876b393d8e02b0124f101fb078239d3.jpg)

![50443de74989a149cb7c39d00cad53ae43aec678e724cb2a528e74f62e650ed9.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/tables/50443de74989a149cb7c39d00cad53ae43aec678e724cb2a528e74f62e650ed9.jpg)

![62b77991999c9aa492a7fa906674a901fa7a1b2e762bdf3081dbe9a8138276ef.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/tables/62b77991999c9aa492a7fa906674a901fa7a1b2e762bdf3081dbe9a8138276ef.jpg)

![6e14ad1ec0ca0a4a06340e2993194fb7874518122d8cb291a55e498eb84eecd0.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/tables/6e14ad1ec0ca0a4a06340e2993194fb7874518122d8cb291a55e498eb84eecd0.jpg)

![8ebb8207fb37706157323dd71f1a0cd351dee675da862caebd493f853934efc3.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/tables/8ebb8207fb37706157323dd71f1a0cd351dee675da862caebd493f853934efc3.jpg)

![9c35dc3d15cdd508eca7ff3e323bdbe664317f192e15575d409ca895130c454a.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/tables/9c35dc3d15cdd508eca7ff3e323bdbe664317f192e15575d409ca895130c454a.jpg)

![e657d22619d23f6f05659a2e560a34a01c947018639079d52402a53ad01941a4.jpg](../iclr_results/160_Restructuring Vector Quantization with the Rotation Trick/tables/e657d22619d23f6f05659a2e560a34a01c947018639079d52402a53ad01941a4.jpg)

## MMQA: Evaluating LLMs with Multi-Table Multi-Hop Complex Questions


### Images

![03b5b200b42b04629f810025de0cc1068884fb42e15b01a5323f7bceef73e0f8.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/images/03b5b200b42b04629f810025de0cc1068884fb42e15b01a5323f7bceef73e0f8.jpg)

![8995eea1b753dfce505faf40a84588272049da8c6fddf13d2526bf3eb0c53446.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/images/8995eea1b753dfce505faf40a84588272049da8c6fddf13d2526bf3eb0c53446.jpg)

![a2954073f69de99bd566d0740ee7012879444ca1f1f645ae2fecbc10f19fe587.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/images/a2954073f69de99bd566d0740ee7012879444ca1f1f645ae2fecbc10f19fe587.jpg)

![d412d106720ecde080cd9342da027818220521f76b9fd54741d951bc71b02eb3.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/images/d412d106720ecde080cd9342da027818220521f76b9fd54741d951bc71b02eb3.jpg)

![dc0efb0e0f7ad02bb4582804192c7a4c8517db9bb827ab8b1856a6252c83c8bf.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/images/dc0efb0e0f7ad02bb4582804192c7a4c8517db9bb827ab8b1856a6252c83c8bf.jpg)

### Tables

![09e62e9f480cabe63be3e3bbc47620b570075f9c0ab6de4483e741e41a365079.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/tables/09e62e9f480cabe63be3e3bbc47620b570075f9c0ab6de4483e741e41a365079.jpg)

![3b380534259904368a09066d75a203da0e2c21b0faaf4fdaf585c16ce5ac8624.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/tables/3b380534259904368a09066d75a203da0e2c21b0faaf4fdaf585c16ce5ac8624.jpg)

![3cf76892c42482bf467fe27adf6fb7e89c30da8acf4ed92cbe4a2d5fe1c348e3.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/tables/3cf76892c42482bf467fe27adf6fb7e89c30da8acf4ed92cbe4a2d5fe1c348e3.jpg)

![7daf007b565f4955797787e25f703ebdc1cea7727023009984c7ca6ccc77b6ff.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/tables/7daf007b565f4955797787e25f703ebdc1cea7727023009984c7ca6ccc77b6ff.jpg)

![a4163f472d515833c36cbc0f1224978f89a2954c6224fae2d2c86c01bd9dd1df.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/tables/a4163f472d515833c36cbc0f1224978f89a2954c6224fae2d2c86c01bd9dd1df.jpg)

![a7995ce61e617f0bbf2234b00ee0f009997427b00a31d3133334d86b70372e67.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/tables/a7995ce61e617f0bbf2234b00ee0f009997427b00a31d3133334d86b70372e67.jpg)

![cae21f492c88eb86b54b326642e7acac0e501756dc1bda72230575833082d6d2.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/tables/cae21f492c88eb86b54b326642e7acac0e501756dc1bda72230575833082d6d2.jpg)

![d7ec6f4e932a3f49744874873186ac42e5164777a3dea9d47975c539471fe3b7.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/tables/d7ec6f4e932a3f49744874873186ac42e5164777a3dea9d47975c539471fe3b7.jpg)

![e0c225111838e3dc38b9261082060afa2a040b5c639362556bfe9f3badf0d92c.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/tables/e0c225111838e3dc38b9261082060afa2a040b5c639362556bfe9f3badf0d92c.jpg)

![f98770ae4db0964d5bbd7ff406e30515813fbe856b9b422fad9dc546f2963a52.jpg](../iclr_results/161_MMQA_ Evaluating LLMs with Multi-Table Multi-Hop Complex Questions/tables/f98770ae4db0964d5bbd7ff406e30515813fbe856b9b422fad9dc546f2963a52.jpg)

## GridMix: Exploring Spatial Modulation for Neural Fields in PDE Modeling


### Images

![02fab3a47781bba0185daa5468b17799d7722c5dd32b09cd491291c5d5a74bd3.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/images/02fab3a47781bba0185daa5468b17799d7722c5dd32b09cd491291c5d5a74bd3.jpg)

![2be275b1f2d4d2c63f2e1d38cf3b2ad70bfc70e73e1497bc6d69e697cbc98105.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/images/2be275b1f2d4d2c63f2e1d38cf3b2ad70bfc70e73e1497bc6d69e697cbc98105.jpg)

![3fc453c8ddc8faa1f5dabfcc46a9289c6a903895a89a5c9905565da082ac7b2e.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/images/3fc453c8ddc8faa1f5dabfcc46a9289c6a903895a89a5c9905565da082ac7b2e.jpg)

![45598a475c74892e6d0759584474a125512a56153a0b1c01ac693f05247ce04f.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/images/45598a475c74892e6d0759584474a125512a56153a0b1c01ac693f05247ce04f.jpg)

![b4dc9702aee179507a03c317cf5e2e9c26ba1583fd518b15b85721236cf8877c.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/images/b4dc9702aee179507a03c317cf5e2e9c26ba1583fd518b15b85721236cf8877c.jpg)

![d2c6ac627d0f89c22643206d8cc3c94143302752790615bc2dda9e56d8eddf93.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/images/d2c6ac627d0f89c22643206d8cc3c94143302752790615bc2dda9e56d8eddf93.jpg)

![d912b6eac9345a1a73880a627f3f952f43442dc849892c71256d402e8bc70839.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/images/d912b6eac9345a1a73880a627f3f952f43442dc849892c71256d402e8bc70839.jpg)

![f7eb617476ad7ca74682f951097ad54e1562f4b5e3066013afed1b9b516bd1e1.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/images/f7eb617476ad7ca74682f951097ad54e1562f4b5e3066013afed1b9b516bd1e1.jpg)

### Tables

![158caeaf1971069ce3323f7ce07520a290f13ce2857d33609f5e137683f5fe9a.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/tables/158caeaf1971069ce3323f7ce07520a290f13ce2857d33609f5e137683f5fe9a.jpg)

![184a86cd1768c09f182f529b73a824e00817c7640c2198d4cb51bc1e1b0b7780.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/tables/184a86cd1768c09f182f529b73a824e00817c7640c2198d4cb51bc1e1b0b7780.jpg)

![355ce52f18c9d919a30da2d45b5df72e8c8ea2ced7551f1da9ac832567059bab.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/tables/355ce52f18c9d919a30da2d45b5df72e8c8ea2ced7551f1da9ac832567059bab.jpg)

![4a88fbd37b20b55bcb8b3b49fe27c4ff7f748434430864d6db78395eb4cdd014.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/tables/4a88fbd37b20b55bcb8b3b49fe27c4ff7f748434430864d6db78395eb4cdd014.jpg)

![a4ac65f43fc71ffca82e07d8b6dedfedd0c2d08cd4e86e2ddcdad40f2810ed4b.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/tables/a4ac65f43fc71ffca82e07d8b6dedfedd0c2d08cd4e86e2ddcdad40f2810ed4b.jpg)

![b1d80f7b39fc301fc7844784561fecd3f3c0904129a2b800ccf3980dfbb853ed.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/tables/b1d80f7b39fc301fc7844784561fecd3f3c0904129a2b800ccf3980dfbb853ed.jpg)

![d53fb7e00b833a4d365c04261b9342384122274a356ca8cda7b86fec126fc2f5.jpg](../iclr_results/162_GridMix_ Exploring Spatial Modulation for Neural Fields in PDE Modeling/tables/d53fb7e00b833a4d365c04261b9342384122274a356ca8cda7b86fec126fc2f5.jpg)

## More RLHF, More Trust? On The Impact of Preference Alignment On Trustworthiness


### Images

![4e2998fae48494157c3087f7b4b8215f9e1b15cf65d49656d5c4ba449c741a1c.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/4e2998fae48494157c3087f7b4b8215f9e1b15cf65d49656d5c4ba449c741a1c.jpg)

![56f52fa5027255b3a1d094f635bfedecffffa47466ad995744deb6bc9bc866be.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/56f52fa5027255b3a1d094f635bfedecffffa47466ad995744deb6bc9bc866be.jpg)

![82df7ca1d29dd8240ceeae1f986d5acc771ccf0f8c6d300d3516616e8245054b.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/82df7ca1d29dd8240ceeae1f986d5acc771ccf0f8c6d300d3516616e8245054b.jpg)

![8368c589c959f9b145dff63443afaea2751c209c42123d72b3729e046dc9a316.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/8368c589c959f9b145dff63443afaea2751c209c42123d72b3729e046dc9a316.jpg)

![917f21bd43f677efc309ff4d7b5e3d9fd03fd1c7c5ec48a6c943d52c40c75696.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/917f21bd43f677efc309ff4d7b5e3d9fd03fd1c7c5ec48a6c943d52c40c75696.jpg)

![93216c243bebe1678a60d4921d2077fcd5e152a3989570d30ee5606fc1d54541.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/93216c243bebe1678a60d4921d2077fcd5e152a3989570d30ee5606fc1d54541.jpg)

![97fbcce6d09b8138d54009c3aaff891bb2fdb987f23fe29bcb9f08590614aea4.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/97fbcce6d09b8138d54009c3aaff891bb2fdb987f23fe29bcb9f08590614aea4.jpg)

![9c6976c198ec9054f0b0d71bad5a0931cf4361db4d42f7f784499b30a3ed529a.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/9c6976c198ec9054f0b0d71bad5a0931cf4361db4d42f7f784499b30a3ed529a.jpg)

![9e448bcbff1c18f4e334eab2329044bf7132077c660395d25da4961ce6467d9a.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/9e448bcbff1c18f4e334eab2329044bf7132077c660395d25da4961ce6467d9a.jpg)

![a501393907b0c4dd0fb8a8166257b0e1ab23f1128aaf50334668fc5573883271.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/a501393907b0c4dd0fb8a8166257b0e1ab23f1128aaf50334668fc5573883271.jpg)

![d9803e0c8b06b099f217787d9053430b254d52d1b91c55af49d2cda3e47fb1ab.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/d9803e0c8b06b099f217787d9053430b254d52d1b91c55af49d2cda3e47fb1ab.jpg)

![e3944eb6441eca406e140fae4dda0a70ef672cc79526a6c7c41178c96132f0bb.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/images/e3944eb6441eca406e140fae4dda0a70ef672cc79526a6c7c41178c96132f0bb.jpg)

### Tables

![4143bcb77d074a2039380ebbfccb91b6a64d047412e9d17276299aec2c386917.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/tables/4143bcb77d074a2039380ebbfccb91b6a64d047412e9d17276299aec2c386917.jpg)

![79852e024cacb350892e171e0963115be88d57cda2fe5997593a3015c2dc643e.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/tables/79852e024cacb350892e171e0963115be88d57cda2fe5997593a3015c2dc643e.jpg)

![abbbb730da7520b4060861746c47f409d1a960bb38e2f6915330bb9b3b225e53.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/tables/abbbb730da7520b4060861746c47f409d1a960bb38e2f6915330bb9b3b225e53.jpg)

![ca2961f115a83ba7b0ecf7869f821f7a8a1bbae6bacb3537f718d6f6473e5a4e.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/tables/ca2961f115a83ba7b0ecf7869f821f7a8a1bbae6bacb3537f718d6f6473e5a4e.jpg)

![db9e743d79992db4f865827a7c75c94a02c8815f3b5ffe9776d01c9a03851eb3.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/tables/db9e743d79992db4f865827a7c75c94a02c8815f3b5ffe9776d01c9a03851eb3.jpg)

![eb97ff628bf71a7d2d4cc54e4f93c6c252f8df55841dd17f9aed0f76455c9488.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/tables/eb97ff628bf71a7d2d4cc54e4f93c6c252f8df55841dd17f9aed0f76455c9488.jpg)

![f3b4e32e8a254bc0072c2d9bc16394013a5b1d7d9f479c250e4a3c5082d56948.jpg](../iclr_results/163_More RLHF, More Trust_ On The Impact of Preference Alignment On Trustworthiness/tables/f3b4e32e8a254bc0072c2d9bc16394013a5b1d7d9f479c250e4a3c5082d56948.jpg)

## Population Transformer: Learning Population-level Representations of Neural Activity


### Images

![0026827419f5faaf831b2061ae8082a164c1f38d4b756cdfce88f8b1d5b9124c.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/0026827419f5faaf831b2061ae8082a164c1f38d4b756cdfce88f8b1d5b9124c.jpg)

![02d9fd0df02819bd91d003342a0c4c63731eeb9982bf03128cb87e90437d2a28.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/02d9fd0df02819bd91d003342a0c4c63731eeb9982bf03128cb87e90437d2a28.jpg)

![133e8ad6f930f56a59eb047e15ab15f8c170989fdc769d3c21c6a908a6e79953.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/133e8ad6f930f56a59eb047e15ab15f8c170989fdc769d3c21c6a908a6e79953.jpg)

![17360ccc8177b4d9ee33406277908c10dd328a50e83584fd6149e0dba697ab7c.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/17360ccc8177b4d9ee33406277908c10dd328a50e83584fd6149e0dba697ab7c.jpg)

![211f7cc53c5d8b1603bb91200e083d9fb3f2c10c0ecafcea814f699f21fdd0f9.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/211f7cc53c5d8b1603bb91200e083d9fb3f2c10c0ecafcea814f699f21fdd0f9.jpg)

![32da252ed3206be3d825c24be5dff8c1303b04258f81452649cc9472644c36e0.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/32da252ed3206be3d825c24be5dff8c1303b04258f81452649cc9472644c36e0.jpg)

![36a26f5d3f40bb7d03ce94c0c6f2f7e60133096f361311975aed8c2beeef9930.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/36a26f5d3f40bb7d03ce94c0c6f2f7e60133096f361311975aed8c2beeef9930.jpg)

![5b78ddbd2a723701d743a9009c65a7daf1d2462a5dd3448c51b023da843e2bc0.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/5b78ddbd2a723701d743a9009c65a7daf1d2462a5dd3448c51b023da843e2bc0.jpg)

![667ce13f39690869020e351b9bae8fee5097ea53d1a44d4de79e267df209363f.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/667ce13f39690869020e351b9bae8fee5097ea53d1a44d4de79e267df209363f.jpg)

![84ef38a6bcfc55df3cf94b6c434c5a496032eef6bf66738265ded72add5e36e3.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/84ef38a6bcfc55df3cf94b6c434c5a496032eef6bf66738265ded72add5e36e3.jpg)

![89aedd78e2d3d263e59200be37e1d53300db4daae03b924978e07605e9d2548d.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/89aedd78e2d3d263e59200be37e1d53300db4daae03b924978e07605e9d2548d.jpg)

![89f78afa37bf85b3c85e1ea16a6aceb49652189afa4d55da2b0207213fb5d611.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/89f78afa37bf85b3c85e1ea16a6aceb49652189afa4d55da2b0207213fb5d611.jpg)

![94ff4d640d67ff9cb7a6ec011c3158e59df6c71381e903f2686a04ac8a3a5c2a.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/94ff4d640d67ff9cb7a6ec011c3158e59df6c71381e903f2686a04ac8a3a5c2a.jpg)

![97e01887d0712a305a15f4eea6141bc5e7c9a051e603307f655e03816d340705.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/97e01887d0712a305a15f4eea6141bc5e7c9a051e603307f655e03816d340705.jpg)

![cee4f8417e2dfc0e100cf4d06f76f762d88d0ccacd2833054980d924888d377b.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/cee4f8417e2dfc0e100cf4d06f76f762d88d0ccacd2833054980d924888d377b.jpg)

![e19dc5812c9f947ea11b2541927dd6036176914aa6ade7f31ad8d13e59e68e17.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/e19dc5812c9f947ea11b2541927dd6036176914aa6ade7f31ad8d13e59e68e17.jpg)

![e7d0c1526ea714671afff6102ae2b4f101f820c3503886f8d1fac0e24de4ce1f.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/images/e7d0c1526ea714671afff6102ae2b4f101f820c3503886f8d1fac0e24de4ce1f.jpg)

### Tables

![0d96f5c83e2b9d74f1d9c279b5764eaa6ec453ec71fcb6b7a2689413c66d9c05.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/tables/0d96f5c83e2b9d74f1d9c279b5764eaa6ec453ec71fcb6b7a2689413c66d9c05.jpg)

![1c812769c3c699b5a9ceed93028082a9c5d763937a43fdba1fd11651cbf370d0.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/tables/1c812769c3c699b5a9ceed93028082a9c5d763937a43fdba1fd11651cbf370d0.jpg)

![2992f7df585830ae4da8dc2bc20cf4f0be1e991fb699b677c9b73b37a4fea3ba.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/tables/2992f7df585830ae4da8dc2bc20cf4f0be1e991fb699b677c9b73b37a4fea3ba.jpg)

![3fef66415bcf22ab56663b8ba485f0f3285e1f449f640625927a4a1c1baaf51b.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/tables/3fef66415bcf22ab56663b8ba485f0f3285e1f449f640625927a4a1c1baaf51b.jpg)

![45c1ec550d7af1d7113da6e4b8556d0166546c29d9a11cba5d08f1168aeb8630.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/tables/45c1ec550d7af1d7113da6e4b8556d0166546c29d9a11cba5d08f1168aeb8630.jpg)

![5139ceaac2979a6232102746c1c714971a30b0802d3915a03704f109865c6d3d.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/tables/5139ceaac2979a6232102746c1c714971a30b0802d3915a03704f109865c6d3d.jpg)

![90ca54b6fdb663cec44bf2015270e7b796cc1c05e9b34f0209b8d97ab126f3c8.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/tables/90ca54b6fdb663cec44bf2015270e7b796cc1c05e9b34f0209b8d97ab126f3c8.jpg)

![b8f77a8d5c02a94d5a959d6948044c4294f1af70462d36bd2cc0ce7328683ebe.jpg](../iclr_results/164_Population Transformer_ Learning Population-level Representations of Neural Activity/tables/b8f77a8d5c02a94d5a959d6948044c4294f1af70462d36bd2cc0ce7328683ebe.jpg)

## Inference Scaling for Long-Context Retrieval Augmented Generation


### Images

![28c2c94a3b8446e6714638bb23eb8b3388649531d437366bd3cfbaff2bbe3dd5.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/28c2c94a3b8446e6714638bb23eb8b3388649531d437366bd3cfbaff2bbe3dd5.jpg)

![2a71d46f49942c18341d0b78d0c52fdae14bcc006ecae4c382a5bc656c6c4c07.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/2a71d46f49942c18341d0b78d0c52fdae14bcc006ecae4c382a5bc656c6c4c07.jpg)

![32e74e07aeb0f3eea5db4dcbe079ee8cc07e7e1a488df01843ced36c1a76534d.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/32e74e07aeb0f3eea5db4dcbe079ee8cc07e7e1a488df01843ced36c1a76534d.jpg)

![479dc87b6803ffbc54323ff4ff16646ecef52089a2999a5cfb9ee353e22acb3e.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/479dc87b6803ffbc54323ff4ff16646ecef52089a2999a5cfb9ee353e22acb3e.jpg)

![48135f2ef8370d885ee2ab2e53c83b2bc9f96a8a000a0f1e9c414f417d64be0d.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/48135f2ef8370d885ee2ab2e53c83b2bc9f96a8a000a0f1e9c414f417d64be0d.jpg)

![4b088d47e0c44bc397ab50e9689a290f570ee401c47d277bd23e3f1130b0c7dc.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/4b088d47e0c44bc397ab50e9689a290f570ee401c47d277bd23e3f1130b0c7dc.jpg)

![71fa70a6c684dd95d107cd47eb4fbf8c1ca233acc9b0c1a0908b0145eef5d63b.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/71fa70a6c684dd95d107cd47eb4fbf8c1ca233acc9b0c1a0908b0145eef5d63b.jpg)

![73173d6babaeb5544b114e987381eefd35b0db7d8b5aa1c4bc2035da776b7ac6.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/73173d6babaeb5544b114e987381eefd35b0db7d8b5aa1c4bc2035da776b7ac6.jpg)

![87883559c36e332acdc25171ba31d81d106ca123ee3d27456e06d42ea43c2ef5.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/87883559c36e332acdc25171ba31d81d106ca123ee3d27456e06d42ea43c2ef5.jpg)

![921128ff15a36f4f80f1708d06888da7b6143b7d916bd6176125c183b6cfed86.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/921128ff15a36f4f80f1708d06888da7b6143b7d916bd6176125c183b6cfed86.jpg)

![999ddb862b2dd3e0219b6917788940f66ea58eae81ddecb0e39c8b033870ee7e.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/999ddb862b2dd3e0219b6917788940f66ea58eae81ddecb0e39c8b033870ee7e.jpg)

![a6d5cc7db2c27a98c47e5e41e2184c17927385e2849d21a121171ba4a4ddeb8a.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/a6d5cc7db2c27a98c47e5e41e2184c17927385e2849d21a121171ba4a4ddeb8a.jpg)

![e1ec600f05cb3676b7d9ce21de69e18e7f2213fc5503241511f8eff197274aae.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/e1ec600f05cb3676b7d9ce21de69e18e7f2213fc5503241511f8eff197274aae.jpg)

![fb465289991d6084a5c0123f65f471274c292ee944e7075ce42a349d9c3097e6.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/images/fb465289991d6084a5c0123f65f471274c292ee944e7075ce42a349d9c3097e6.jpg)

### Tables

![09007137aea978bd5d005cbe6f7f879d7e23243dac8b34d9af0fa528f05ef6d1.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/tables/09007137aea978bd5d005cbe6f7f879d7e23243dac8b34d9af0fa528f05ef6d1.jpg)

![21eaa16d4d44a8fe7dce452c36d2ce6798dc26dd612b12921174565a73f9d866.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/tables/21eaa16d4d44a8fe7dce452c36d2ce6798dc26dd612b12921174565a73f9d866.jpg)

![255ef9d9e495bff15419c614bca0836661d9c606dafdce09b214445d4850b92a.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/tables/255ef9d9e495bff15419c614bca0836661d9c606dafdce09b214445d4850b92a.jpg)

![39d84d9416f7e1b75e5261fe1577ae7effbdb4f576d80ef7cc3eaea8e13137f9.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/tables/39d84d9416f7e1b75e5261fe1577ae7effbdb4f576d80ef7cc3eaea8e13137f9.jpg)

![60d98f555ca6099b7898483feeb33763d668a32e44532be0c9fec15c8c28243c.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/tables/60d98f555ca6099b7898483feeb33763d668a32e44532be0c9fec15c8c28243c.jpg)

![842088c31b7aeaf4f52f3cbd787b22b9cbfc595454622cb8c545500784b69fc3.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/tables/842088c31b7aeaf4f52f3cbd787b22b9cbfc595454622cb8c545500784b69fc3.jpg)

![b7baa6923da013a5981394118126efb7b2c8d4506f55df3f6c6702ac7adf47d9.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/tables/b7baa6923da013a5981394118126efb7b2c8d4506f55df3f6c6702ac7adf47d9.jpg)

![bbbf03a2d0095a2ea7e80382f6e5a906575ca95f124917dd07314feb3dc83fc4.jpg](../iclr_results/165_Inference Scaling for Long-Context Retrieval Augmented Generation/tables/bbbf03a2d0095a2ea7e80382f6e5a906575ca95f124917dd07314feb3dc83fc4.jpg)

## Proxy Denoising for Source-Free Domain Adaptation


### Images

![29cb841c31a342df913ab974e9c42d4890d31f495b283e8227064655e45b4036.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/images/29cb841c31a342df913ab974e9c42d4890d31f495b283e8227064655e45b4036.jpg)

![4df0a5c16a26fda98e97f2f7f36d0b1a49a688ec17faa958e9ffe52447119e8c.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/images/4df0a5c16a26fda98e97f2f7f36d0b1a49a688ec17faa958e9ffe52447119e8c.jpg)

![66b84c40dfe16d90cebff235ea454d7823fbb24c8cba5a8736d2d39f01586f43.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/images/66b84c40dfe16d90cebff235ea454d7823fbb24c8cba5a8736d2d39f01586f43.jpg)

![748071375be8a5fa9bca223888c0c4531e8fed89da437c2cf0e6f00be0c7cc6c.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/images/748071375be8a5fa9bca223888c0c4531e8fed89da437c2cf0e6f00be0c7cc6c.jpg)

![85e24e08e013cd945a816ba4e3d97c64706e21fe23b64656cebf8dbaf341dcf0.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/images/85e24e08e013cd945a816ba4e3d97c64706e21fe23b64656cebf8dbaf341dcf0.jpg)

### Tables

![22be632aac7d77a975a9a60772adc42beba6e49ae8caca2221acf25e6b7e7752.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/22be632aac7d77a975a9a60772adc42beba6e49ae8caca2221acf25e6b7e7752.jpg)

![24ec33dfaedef288923c083ac06d99f05cd30c4efbf8072d3af1db4b5a5b77fe.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/24ec33dfaedef288923c083ac06d99f05cd30c4efbf8072d3af1db4b5a5b77fe.jpg)

![3038b91dea79572714255b7c0c49b407b0234f8f2eb0e76233190183890cb87e.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/3038b91dea79572714255b7c0c49b407b0234f8f2eb0e76233190183890cb87e.jpg)

![37e501b1fe06a0b8a641621834241b86c752c23d9edca3a40f683d2410090bf0.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/37e501b1fe06a0b8a641621834241b86c752c23d9edca3a40f683d2410090bf0.jpg)

![48a679f7921c70758b74e1cb35b1670e885300c6265da41df0fc1217042ab17b.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/48a679f7921c70758b74e1cb35b1670e885300c6265da41df0fc1217042ab17b.jpg)

![4adf07fdb2b4ada31d9381ca3b19432405343da1ad00b6c7671ddf57396222b6.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/4adf07fdb2b4ada31d9381ca3b19432405343da1ad00b6c7671ddf57396222b6.jpg)

![4e0a25c74e578f65510ce1caf45120ce87da11fc980b3b39c3359bfd7bb4cd40.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/4e0a25c74e578f65510ce1caf45120ce87da11fc980b3b39c3359bfd7bb4cd40.jpg)

![4fd68616d21c9e3aba648c73f88075b80d4ce071e4f8fcc9bfced0a5817a3b36.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/4fd68616d21c9e3aba648c73f88075b80d4ce071e4f8fcc9bfced0a5817a3b36.jpg)

![6906c5603d470b88d355ae29cc12c53b7db1c9a3e41a4857299ed733d3d1c306.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/6906c5603d470b88d355ae29cc12c53b7db1c9a3e41a4857299ed733d3d1c306.jpg)

![82b11ffadfd25b835e12a1eaabddde8154f3cc76621b3ae28559b522995931fe.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/82b11ffadfd25b835e12a1eaabddde8154f3cc76621b3ae28559b522995931fe.jpg)

![946ac1ece91afbe3c18e58862b7aa2590fda735a8400292f5238f0eec59f8f8f.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/946ac1ece91afbe3c18e58862b7aa2590fda735a8400292f5238f0eec59f8f8f.jpg)

![979944e2faaaa0286bbe99c4ad316146d5aa5c5bf36e6f51f410081f6e018def.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/979944e2faaaa0286bbe99c4ad316146d5aa5c5bf36e6f51f410081f6e018def.jpg)

![9e9db8d031d500ff67336f9eea79ce32594728a37cf3cff5fa9f031ced6df0af.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/9e9db8d031d500ff67336f9eea79ce32594728a37cf3cff5fa9f031ced6df0af.jpg)

![a3cbbbd5ab4d09f7ccc71ffaaffc076792cf7cca0d2644b18216af7591c5c48a.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/a3cbbbd5ab4d09f7ccc71ffaaffc076792cf7cca0d2644b18216af7591c5c48a.jpg)

![aa930560d6fa20aacf57a9e6695d26150bc6207be7c5ce5bb92cf8a56391fd19.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/aa930560d6fa20aacf57a9e6695d26150bc6207be7c5ce5bb92cf8a56391fd19.jpg)

![b10b3fc9c6b46cf458615202ec8df517b37dea8da261558382602837dcdd9408.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/b10b3fc9c6b46cf458615202ec8df517b37dea8da261558382602837dcdd9408.jpg)

![d8dc9c103319d5a32eac39fb80049c9edc851ca204e11bc26381b0068db56e38.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/d8dc9c103319d5a32eac39fb80049c9edc851ca204e11bc26381b0068db56e38.jpg)

![e2f88c486e870369f8b0e31814f148224720e868e28fcdf9ce3bf111aaeae587.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/e2f88c486e870369f8b0e31814f148224720e868e28fcdf9ce3bf111aaeae587.jpg)

![e4256cefe52b8216111185c4556cd184594abcd6bca449dbd15ac6c7feba13b3.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/e4256cefe52b8216111185c4556cd184594abcd6bca449dbd15ac6c7feba13b3.jpg)

![e63991143f10947a46c65a43150ff4ff40d0f12150956596c411ea0b3ca782b4.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/e63991143f10947a46c65a43150ff4ff40d0f12150956596c411ea0b3ca782b4.jpg)

![ed451dbb6a376aafc3e9a95e397321306405ab46e7f26f9ad785be65b840ca8f.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/ed451dbb6a376aafc3e9a95e397321306405ab46e7f26f9ad785be65b840ca8f.jpg)

![eda266d9f198878ce2ca7081300438124b98d32d16ee095234d3c407cc2a7ca3.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/eda266d9f198878ce2ca7081300438124b98d32d16ee095234d3c407cc2a7ca3.jpg)

![f1f2130276e930d06f934656c40d76c11a096fa57de60b036acf395af716a4ef.jpg](../iclr_results/166_Proxy Denoising for Source-Free Domain Adaptation/tables/f1f2130276e930d06f934656c40d76c11a096fa57de60b036acf395af716a4ef.jpg)

## Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs


### Images

![9e2ac9a6596e224675dab8e1286e2b9401bbdb549d0cc5622923f20734edd242.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/images/9e2ac9a6596e224675dab8e1286e2b9401bbdb549d0cc5622923f20734edd242.jpg)

![fa218f3a435d488296a93edd578f911d844fb3a7f761f9159fec0b8e1fc99856.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/images/fa218f3a435d488296a93edd578f911d844fb3a7f761f9159fec0b8e1fc99856.jpg)

### Tables

![05fa5bcc6cbb45dde997c5d70a292f8a4cb734e51372ac7463c38fab4a3e2435.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/05fa5bcc6cbb45dde997c5d70a292f8a4cb734e51372ac7463c38fab4a3e2435.jpg)

![1395fdb359b5fc4c971cd8cebdb94aabb1fe5f1ed5d70f7e8c7f943c0a869a0a.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/1395fdb359b5fc4c971cd8cebdb94aabb1fe5f1ed5d70f7e8c7f943c0a869a0a.jpg)

![1768f61b97a8eed0323dd973e71a89661723d239a1a8cd805aa2a1b92cb59aac.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/1768f61b97a8eed0323dd973e71a89661723d239a1a8cd805aa2a1b92cb59aac.jpg)

![18fc2fb31b572b4f6d2f9ff60c6e33282c5966034640ded6e31f2d58b6f2018d.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/18fc2fb31b572b4f6d2f9ff60c6e33282c5966034640ded6e31f2d58b6f2018d.jpg)

![1a771a6af61487dcc5f354a765f8b6b93c4ba7218aa90064e5fe5d1812b03cfd.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/1a771a6af61487dcc5f354a765f8b6b93c4ba7218aa90064e5fe5d1812b03cfd.jpg)

![1f65577826aebfb2fdd138f7d43799ee19c8cfe0afd6f1900e1164ba15397708.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/1f65577826aebfb2fdd138f7d43799ee19c8cfe0afd6f1900e1164ba15397708.jpg)

![24138fe87320eca49f2b3b727b6d8fa2c0c5e7aa52572e701344974923d55014.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/24138fe87320eca49f2b3b727b6d8fa2c0c5e7aa52572e701344974923d55014.jpg)

![27bf30d3d292c70731f2107820e9d1535f87686fe5885f23f918ee995b9f52df.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/27bf30d3d292c70731f2107820e9d1535f87686fe5885f23f918ee995b9f52df.jpg)

![3c21f351a5859cc558ac0d0c477b943419247621799e636eb060f6d5a4ae539b.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/3c21f351a5859cc558ac0d0c477b943419247621799e636eb060f6d5a4ae539b.jpg)

![3e92ecb54298534af89e8901a0702ed0c4b5967d4679c8ac5288c50e240f129b.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/3e92ecb54298534af89e8901a0702ed0c4b5967d4679c8ac5288c50e240f129b.jpg)

![405d3993336080130ae76068058c0bc715f87495df1dfd8388b398c09e2997e3.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/405d3993336080130ae76068058c0bc715f87495df1dfd8388b398c09e2997e3.jpg)

![4a9d15661844fa8a01354daaf7901a291b1e577de158784a939c8063130c5d43.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/4a9d15661844fa8a01354daaf7901a291b1e577de158784a939c8063130c5d43.jpg)

![4e1fb5496832db44917fecfffaf410ee303ad4ceb2cadb2e5328ec546d998b57.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/4e1fb5496832db44917fecfffaf410ee303ad4ceb2cadb2e5328ec546d998b57.jpg)

![65f09e97ebc9d2ac5f79ce8e8d4e4731c7eb547b19e557003cb4b9c740d3bea8.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/65f09e97ebc9d2ac5f79ce8e8d4e4731c7eb547b19e557003cb4b9c740d3bea8.jpg)

![66b21606f0462bee1513b36a4ab5a4d6eea0ae60dd05136f64ea2515108c4776.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/66b21606f0462bee1513b36a4ab5a4d6eea0ae60dd05136f64ea2515108c4776.jpg)

![7013673224fafec10e4a4c3fcbd5d2d55595c2cba11142f03e3943cb8a138b09.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/7013673224fafec10e4a4c3fcbd5d2d55595c2cba11142f03e3943cb8a138b09.jpg)

![7b9bbdeb27707b096b32031eadddfe9f03e21d2726da544314b7040935476668.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/7b9bbdeb27707b096b32031eadddfe9f03e21d2726da544314b7040935476668.jpg)

![8c1cc8ac8c099f9b1ad1835615ff1fd2fceebd29c489f74ee76a3ed9cd9334dd.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/8c1cc8ac8c099f9b1ad1835615ff1fd2fceebd29c489f74ee76a3ed9cd9334dd.jpg)

![8d6ef97b7b45970e91ec30e58ee69c748acc7bff8ffa4647c112b36336c28e4a.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/8d6ef97b7b45970e91ec30e58ee69c748acc7bff8ffa4647c112b36336c28e4a.jpg)

![9437c5c90a53eb0662ba6dd66136f18b04c02ecc8e1a781bcbf9afa8513822f8.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/9437c5c90a53eb0662ba6dd66136f18b04c02ecc8e1a781bcbf9afa8513822f8.jpg)

![97c72001865f38a2f5e2696669f2615e83808a713d1753b82d56b6c303ecc954.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/97c72001865f38a2f5e2696669f2615e83808a713d1753b82d56b6c303ecc954.jpg)

![9a77fce2174fa64223943b98272b04b2447b927d05f4dfc56e9ddd564c23a572.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/9a77fce2174fa64223943b98272b04b2447b927d05f4dfc56e9ddd564c23a572.jpg)

![b95a900fd2641f725246bbec37143ac5f8974e8c26d1bc8f5032c6c72174827a.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/b95a900fd2641f725246bbec37143ac5f8974e8c26d1bc8f5032c6c72174827a.jpg)

![c7f2602919bcf6154b8c9c6b4285d8221fb73b3b12e9821fbb1a4169f15a7117.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/c7f2602919bcf6154b8c9c6b4285d8221fb73b3b12e9821fbb1a4169f15a7117.jpg)

![ca36ebd55b6fbe22e995e9d894546dfea9425dfed43366988d3f37a1dfc31464.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/ca36ebd55b6fbe22e995e9d894546dfea9425dfed43366988d3f37a1dfc31464.jpg)

![d1515be9fe7b75d57d65bd0a18d5fdf0369cb85e75b4536783c7265bf3a99f9e.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/d1515be9fe7b75d57d65bd0a18d5fdf0369cb85e75b4536783c7265bf3a99f9e.jpg)

![e484dddfb22c8db98e693696f2c30361cb1b6310bd1998439cf9f8529b1572d9.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/e484dddfb22c8db98e693696f2c30361cb1b6310bd1998439cf9f8529b1572d9.jpg)

![ec2559d545c82bea6201e0b1fdd90b67515aedaba854fba1ff07a3f84387183b.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/ec2559d545c82bea6201e0b1fdd90b67515aedaba854fba1ff07a3f84387183b.jpg)

![f3565ee719fbc86a48445fec062cc6b2200a175694caa5a100fdeaa0a8bde29b.jpg](../iclr_results/167_Turning Up the Heat_ Min-p Sampling for Creative and Coherent LLM Outputs/tables/f3565ee719fbc86a48445fec062cc6b2200a175694caa5a100fdeaa0a8bde29b.jpg)

## Topological Blindspots: Understanding and Extending Topological Deep Learning Through the Lens of Expressivity


### Images

![028503409a7a35b476a8107743ba4dde3ed790b45093a567567f631ec9a3624c.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/028503409a7a35b476a8107743ba4dde3ed790b45093a567567f631ec9a3624c.jpg)

![21a5f4f0cc90ee1da62cbc5d2b787066f130c7852195f327e98ac2e17bab73af.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/21a5f4f0cc90ee1da62cbc5d2b787066f130c7852195f327e98ac2e17bab73af.jpg)

![3bcf7efe85971119469ea21f6550a31b959f2870b9eb6a5c6d7f463f66768a3b.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/3bcf7efe85971119469ea21f6550a31b959f2870b9eb6a5c6d7f463f66768a3b.jpg)

![3cbbca8f506b2c5113f13504e54d391be9fad7efd321c8134fd51c82373ebe57.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/3cbbca8f506b2c5113f13504e54d391be9fad7efd321c8134fd51c82373ebe57.jpg)

![4e61aff506ff5b221588e682cf608b68510a0c4a277bc4c9c1d5a9a012a78fe6.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/4e61aff506ff5b221588e682cf608b68510a0c4a277bc4c9c1d5a9a012a78fe6.jpg)

![6cbe45eeb96308600ef6a98798a5d72ad31f0fe9b7f91f672edd06fa7944eaa6.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/6cbe45eeb96308600ef6a98798a5d72ad31f0fe9b7f91f672edd06fa7944eaa6.jpg)

![7a3846835461e72dfbe7ca79c600ba3315198ac2b37ce07005cabab468f4c6d2.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/7a3846835461e72dfbe7ca79c600ba3315198ac2b37ce07005cabab468f4c6d2.jpg)

![7d247f0ff1d1fdc93da4aeb05338fe297c57d1bfad98592471aca6cdc8986461.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/7d247f0ff1d1fdc93da4aeb05338fe297c57d1bfad98592471aca6cdc8986461.jpg)

![93f95928bbc76bf0fb2a609acd8434935952a1a09f2a56235ac75a413d48aab3.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/93f95928bbc76bf0fb2a609acd8434935952a1a09f2a56235ac75a413d48aab3.jpg)

![9cdfa1fef2bcb63fba5064a94775e14058748faac2ecafa0e033c18399776d13.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/9cdfa1fef2bcb63fba5064a94775e14058748faac2ecafa0e033c18399776d13.jpg)

![9e310d33f094e795c20e599910051df560df7221176b28f6b3062b14b79ff652.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/9e310d33f094e795c20e599910051df560df7221176b28f6b3062b14b79ff652.jpg)

![a42e369e2ac7128c1ed3cc196727ff8594170117c5f66be89df1e1829ae7cd3c.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/a42e369e2ac7128c1ed3cc196727ff8594170117c5f66be89df1e1829ae7cd3c.jpg)

![b04d4d330997ac03c7c41e2e53b9b0e178a127515d7f531d650a0e291569d69c.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/b04d4d330997ac03c7c41e2e53b9b0e178a127515d7f531d650a0e291569d69c.jpg)

![b7e7a460a130beaec24841047c3a93563cd3b225a7043a6bac934b0007230332.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/b7e7a460a130beaec24841047c3a93563cd3b225a7043a6bac934b0007230332.jpg)

![e641645820d10987c66f8dea93b18dc7c011fa06c3a04ab1f699fa69640b8353.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/e641645820d10987c66f8dea93b18dc7c011fa06c3a04ab1f699fa69640b8353.jpg)

![e82ad587cc2e7145884a82663121a72cff228b6235b5cf162c87c3164552c416.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/images/e82ad587cc2e7145884a82663121a72cff228b6235b5cf162c87c3164552c416.jpg)

### Tables

![4f721a60bc26cec21bb3334fffda8eb910472c9b2a5bba91ad8a51edab7532cd.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/tables/4f721a60bc26cec21bb3334fffda8eb910472c9b2a5bba91ad8a51edab7532cd.jpg)

![659b615c5d137ac13e250b4048e2741117e12e6fb99fa52e00cabf22ecf658c8.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/tables/659b615c5d137ac13e250b4048e2741117e12e6fb99fa52e00cabf22ecf658c8.jpg)

![888d51fd5d45761e8ea990233078dac69ece8abb17d6bd8b4180690e7d041cd5.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/tables/888d51fd5d45761e8ea990233078dac69ece8abb17d6bd8b4180690e7d041cd5.jpg)

![a6b7ef521f9af475047088096c97dd17f8513f6ef4dfb1fbbac36b9993a7bb28.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/tables/a6b7ef521f9af475047088096c97dd17f8513f6ef4dfb1fbbac36b9993a7bb28.jpg)

![f333cb5041d1e046671502e79c27640ed48f2b7b9d8eaf8cc2ab881d25ed746a.jpg](../iclr_results/168_Topological Blindspots_ Understanding and Extending Topological Deep Learning Through the Lens of Ex/tables/f333cb5041d1e046671502e79c27640ed48f2b7b9d8eaf8cc2ab881d25ed746a.jpg)

## Retrieval Head Mechanistically Explains Long-Context Factuality


### Images

![06b380daed50da9ed1faae6fa7571eb949ad2bdf362703e84d0373dfcf82227f.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/06b380daed50da9ed1faae6fa7571eb949ad2bdf362703e84d0373dfcf82227f.jpg)

![07f9f47bb5ace396e6e5ccc9f0c8265c37b3948088c06f8f32754b7bd54ae59c.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/07f9f47bb5ace396e6e5ccc9f0c8265c37b3948088c06f8f32754b7bd54ae59c.jpg)

![0c66e07dd153efef2a345d9e1b54e8a30357384ef8da71328cddff8ef1dd1de2.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/0c66e07dd153efef2a345d9e1b54e8a30357384ef8da71328cddff8ef1dd1de2.jpg)

![14bbb593682c4ed6889d35d59c31d3c5187a9c57f8c28f43a99fb7371223efe9.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/14bbb593682c4ed6889d35d59c31d3c5187a9c57f8c28f43a99fb7371223efe9.jpg)

![52184b46989c51024b522d681bfd50b09e341d71325d304e8fce9f8a943b152f.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/52184b46989c51024b522d681bfd50b09e341d71325d304e8fce9f8a943b152f.jpg)

![56368c4858ecbf4a8dc74d99bc476423432d18fd0c0e92f012c79b7e3b686453.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/56368c4858ecbf4a8dc74d99bc476423432d18fd0c0e92f012c79b7e3b686453.jpg)

![563938d528f459f9caee435c379e09add1254220764a02c29b3de8fdc57ca766.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/563938d528f459f9caee435c379e09add1254220764a02c29b3de8fdc57ca766.jpg)

![6b7ac60d53f7b805bbed403c5d08bcda5b5c2be208853721c2f79d9d965a21cb.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/6b7ac60d53f7b805bbed403c5d08bcda5b5c2be208853721c2f79d9d965a21cb.jpg)

![7345be229e71a85dafd18a6547ddcc36ee02c0d89540101f5da1d2d78946cb29.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/7345be229e71a85dafd18a6547ddcc36ee02c0d89540101f5da1d2d78946cb29.jpg)

![873df00d38b53ccd13c17158d5c48e1ca0934673603b5608896457bb5eb7c43d.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/873df00d38b53ccd13c17158d5c48e1ca0934673603b5608896457bb5eb7c43d.jpg)

![a51a4ccd761014d05c0a44a2b96d129d1dccce9b1aac47b78472bd03b03e9e9f.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/a51a4ccd761014d05c0a44a2b96d129d1dccce9b1aac47b78472bd03b03e9e9f.jpg)

![c5f52e9e8230244ed25de2a0d175006f729b351743004537d176c029c7112b69.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/c5f52e9e8230244ed25de2a0d175006f729b351743004537d176c029c7112b69.jpg)

![c82012c8ab057bb445dfbd51edf6c287fd441a766e6a7563dbde6a65a0ce3447.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/c82012c8ab057bb445dfbd51edf6c287fd441a766e6a7563dbde6a65a0ce3447.jpg)

![cf9ee2000f874c9de9cd3182df3438a02877a2e516fcad1f257612ff3397bbf2.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/images/cf9ee2000f874c9de9cd3182df3438a02877a2e516fcad1f257612ff3397bbf2.jpg)

### Tables

![0d48b98dc8e58d430fcff540b4da91986015b83487a1aab4e8d23ae2be77ef83.jpg](../iclr_results/169_Retrieval Head Mechanistically Explains Long-Context Factuality/tables/0d48b98dc8e58d430fcff540b4da91986015b83487a1aab4e8d23ae2be77ef83.jpg)

## MIND over Body: Adaptive Thinking using Dynamic Computation


### Images

![287858bd88f39f5aaae222acd99bde2d7777918d403e464381978746eebdd8ab.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/images/287858bd88f39f5aaae222acd99bde2d7777918d403e464381978746eebdd8ab.jpg)

![4f79daf8b422e9560867327686154a6b39f38536271c60771fb2c32a3b6352bf.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/images/4f79daf8b422e9560867327686154a6b39f38536271c60771fb2c32a3b6352bf.jpg)

![54f979b1c1471bb8e7fe5ea56f84977ddace30ba81bce3eb834686e3a2f8b4ff.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/images/54f979b1c1471bb8e7fe5ea56f84977ddace30ba81bce3eb834686e3a2f8b4ff.jpg)

![66371ea5cdd870d85ad5275be1f2ae6fdf1d60f83d5deb705d3cfbd50c66d349.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/images/66371ea5cdd870d85ad5275be1f2ae6fdf1d60f83d5deb705d3cfbd50c66d349.jpg)

![6749cd688e3a81b97362885c75570892e3b565fc0ed7b427e1bc8eca004e5596.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/images/6749cd688e3a81b97362885c75570892e3b565fc0ed7b427e1bc8eca004e5596.jpg)

![94cc16b9c955d830512d7085f620f8a864e9abcc8be07c112fa527e4b500ce25.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/images/94cc16b9c955d830512d7085f620f8a864e9abcc8be07c112fa527e4b500ce25.jpg)

![a01f988dcf3d653574045d7bf6d9312c8b451c116b2b95969abf2ac8cd0cd54d.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/images/a01f988dcf3d653574045d7bf6d9312c8b451c116b2b95969abf2ac8cd0cd54d.jpg)

![a2f69eba4fafd43682351d5a6d5d00b20881f21f87e906132a3e48aea5c89e28.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/images/a2f69eba4fafd43682351d5a6d5d00b20881f21f87e906132a3e48aea5c89e28.jpg)

![ba52e17ad0fd05b671ddad7ba5e61737a1a1815a1eec9662c827dcd166242adc.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/images/ba52e17ad0fd05b671ddad7ba5e61737a1a1815a1eec9662c827dcd166242adc.jpg)

![bfcc17f783fb2b1ecbaf3ce72c7a9c7c97e80505241450933f043e3c5b6f32c2.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/images/bfcc17f783fb2b1ecbaf3ce72c7a9c7c97e80505241450933f043e3c5b6f32c2.jpg)

![e2d5eefc4ee33215f2dab5de9d3f9b08ab0b8f35c60b83f8ed6d5b78618d7f98.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/images/e2d5eefc4ee33215f2dab5de9d3f9b08ab0b8f35c60b83f8ed6d5b78618d7f98.jpg)

### Tables

![008141789ca2d91e32ddd4970adcd8b8fc91b5fe48bda0dabf334872039f3fe0.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/008141789ca2d91e32ddd4970adcd8b8fc91b5fe48bda0dabf334872039f3fe0.jpg)

![083c1a5ae22185cf6a19867e3d2c42cb5848faa7ade51c04517f17d37bf756bf.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/083c1a5ae22185cf6a19867e3d2c42cb5848faa7ade51c04517f17d37bf756bf.jpg)

![4125ad532e57579cb7bb768c43c7c7daf859f7265d5054d12b7b5c44c104941a.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/4125ad532e57579cb7bb768c43c7c7daf859f7265d5054d12b7b5c44c104941a.jpg)

![5b9ad05d30bcb9212d382a364f3a58f8328b79af7d16857b00957c4df2e3af00.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/5b9ad05d30bcb9212d382a364f3a58f8328b79af7d16857b00957c4df2e3af00.jpg)

![5c4e10526d0912717f8fa2979312364e044aae3ac72f409f237d3ce9dea1d97c.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/5c4e10526d0912717f8fa2979312364e044aae3ac72f409f237d3ce9dea1d97c.jpg)

![66ce0136eaed9c8399062b216854a44ec6e312b9bdf80efd74fd30b31e2cf9a1.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/66ce0136eaed9c8399062b216854a44ec6e312b9bdf80efd74fd30b31e2cf9a1.jpg)

![6a8ab888f8866f9e67a505a5052ecd6701825528934a02b8ae66f17aa794f0cd.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/6a8ab888f8866f9e67a505a5052ecd6701825528934a02b8ae66f17aa794f0cd.jpg)

![796d381023ccc5ed3ac19496756a76203bc66aa04598b09e3e171d8e92a52fce.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/796d381023ccc5ed3ac19496756a76203bc66aa04598b09e3e171d8e92a52fce.jpg)

![8aba64769f771314c7a962ae7d063ccf14d2229dcf2b46c7deb3c76fcab6d496.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/8aba64769f771314c7a962ae7d063ccf14d2229dcf2b46c7deb3c76fcab6d496.jpg)

![9284b68ca70c41e3fed7bb0b1925e1a144283894197581718f1068fe1dfdc6b8.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/9284b68ca70c41e3fed7bb0b1925e1a144283894197581718f1068fe1dfdc6b8.jpg)

![a110a2803a4365b2a0cdc737c8670878e355021026149014cd1e35303d076a62.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/a110a2803a4365b2a0cdc737c8670878e355021026149014cd1e35303d076a62.jpg)

![d5d7a6014b829f54bd047fc71fea58a67eca9613209df258401ffc978eeba40f.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/d5d7a6014b829f54bd047fc71fea58a67eca9613209df258401ffc978eeba40f.jpg)

![d914bb01992902ff8a68ae23f8e3d7d8aa32667968e58f41333ceaab97b3633f.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/d914bb01992902ff8a68ae23f8e3d7d8aa32667968e58f41333ceaab97b3633f.jpg)

![da9e896cbe49f3357fde8044b93a698e7f44a1a5ff9bc66dd1105eb5160016ab.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/da9e896cbe49f3357fde8044b93a698e7f44a1a5ff9bc66dd1105eb5160016ab.jpg)

![e09c11e36edaa27c062f92a3c6905a0f1c1172e336e2eccb741bcb54c8c371a4.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/e09c11e36edaa27c062f92a3c6905a0f1c1172e336e2eccb741bcb54c8c371a4.jpg)

![eb253e6bd2a7cc5dd7527adcfcdcc99b3cdc6e2d0b9ed5f32fcb928d6dddcc53.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/eb253e6bd2a7cc5dd7527adcfcdcc99b3cdc6e2d0b9ed5f32fcb928d6dddcc53.jpg)

![eeaed12669049e5379e6d1e846571d251918882cf6274f9d46135494bc598c7d.jpg](../iclr_results/170_MIND over Body_ Adaptive Thinking using Dynamic Computation/tables/eeaed12669049e5379e6d1e846571d251918882cf6274f9d46135494bc598c7d.jpg)

## How much of my dataset did you use? Quantitative Data Usage Inference in Machine Learning


### Images

![0d291cb30106439c7be63fe3828798ccc96512a14c239139adfa3abc69b207db.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/images/0d291cb30106439c7be63fe3828798ccc96512a14c239139adfa3abc69b207db.jpg)

![2f53419d75d1b2caf2826ae1e93e268142c6b0e7954ee3607d28eadc25fb0d55.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/images/2f53419d75d1b2caf2826ae1e93e268142c6b0e7954ee3607d28eadc25fb0d55.jpg)

![591dabb5f17d42e351d63edb22ac1b1ada8b13f7025e40b1151eba5082188588.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/images/591dabb5f17d42e351d63edb22ac1b1ada8b13f7025e40b1151eba5082188588.jpg)

![84c31ac02c485ab02fad939f9172a8b123f676a5444da748ebaad8a51b892a91.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/images/84c31ac02c485ab02fad939f9172a8b123f676a5444da748ebaad8a51b892a91.jpg)

![8a5c0586e641ad3a4cc177a599be1a2603cbcc2d7143b9889eec59b0eaa2e021.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/images/8a5c0586e641ad3a4cc177a599be1a2603cbcc2d7143b9889eec59b0eaa2e021.jpg)

![9c3d1810df5b6b2203466023ee76e4fe18656cf54d876ffa8617f013d8dbb31b.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/images/9c3d1810df5b6b2203466023ee76e4fe18656cf54d876ffa8617f013d8dbb31b.jpg)

![babe6a03758cf2d64ceff51674b9ac21c7d506ab931bb80370c2eb74554408ef.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/images/babe6a03758cf2d64ceff51674b9ac21c7d506ab931bb80370c2eb74554408ef.jpg)

![e23fde6c56eb4e6ddf76eea7f2707168783ad9b97bfb44793a5b41180e629dc0.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/images/e23fde6c56eb4e6ddf76eea7f2707168783ad9b97bfb44793a5b41180e629dc0.jpg)

![f84c357a5d8d1c32807c960cc32f8573df61876df2de37c6ddc69c098211a6cd.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/images/f84c357a5d8d1c32807c960cc32f8573df61876df2de37c6ddc69c098211a6cd.jpg)

![f86d3d620b7549fa66b65d73c9bbb3e22acf35e4197f16b21a0dfbdceb35867c.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/images/f86d3d620b7549fa66b65d73c9bbb3e22acf35e4197f16b21a0dfbdceb35867c.jpg)

### Tables

![2a75b0cc0e1ef213350fed612f8498f6166c59e91865b019b58fdf9c43267ebd.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/tables/2a75b0cc0e1ef213350fed612f8498f6166c59e91865b019b58fdf9c43267ebd.jpg)

![3225ba20638dacac277a4e2be79eaead60b10b1435ef4e17f6e6a63ae633a398.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/tables/3225ba20638dacac277a4e2be79eaead60b10b1435ef4e17f6e6a63ae633a398.jpg)

![bda891fe05f415e031cf04d3eaf0141ef7f93c7f14437ac53f5a0cd8d95b9514.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/tables/bda891fe05f415e031cf04d3eaf0141ef7f93c7f14437ac53f5a0cd8d95b9514.jpg)

![dc2a3b798f7b0db2cdeedcfe9edd4dde1ba726d12b3e120cc1a30d3ec6ea8404.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/tables/dc2a3b798f7b0db2cdeedcfe9edd4dde1ba726d12b3e120cc1a30d3ec6ea8404.jpg)

![e1ad51921553f9fe038a5678e42f24f565bf9a31fb61b2b0e16c0a9ac3c996dd.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/tables/e1ad51921553f9fe038a5678e42f24f565bf9a31fb61b2b0e16c0a9ac3c996dd.jpg)

![e3197568c97f6dfe6c166f9b5095079999232f9564423b2202bfecb078b930e2.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/tables/e3197568c97f6dfe6c166f9b5095079999232f9564423b2202bfecb078b930e2.jpg)

![e3295c8372519faaf7d8304349cdc65108c87e67b49963d0cb61fce06355d280.jpg](../iclr_results/171_How much of my dataset did you use_ Quantitative Data Usage Inference in Machine Learning/tables/e3295c8372519faaf7d8304349cdc65108c87e67b49963d0cb61fce06355d280.jpg)

## SymmetricDiffusers: Learning Discrete Diffusion on Finite Symmetric Groups


### Images

![48f45f86fa0656447f6d22c7c65edaba2d835beaba6a6936ba86ca5374610f04.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/images/48f45f86fa0656447f6d22c7c65edaba2d835beaba6a6936ba86ca5374610f04.jpg)

![a069ba5a70804e07bc5eed3eb65235ebaaa2eac07779555279719c57adc7fd1a.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/images/a069ba5a70804e07bc5eed3eb65235ebaaa2eac07779555279719c57adc7fd1a.jpg)

![f59f285cfab9d78378e8d0455beb3f68cc4557620397f03a681168ce3b5f8c8e.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/images/f59f285cfab9d78378e8d0455beb3f68cc4557620397f03a681168ce3b5f8c8e.jpg)

### Tables

![1313fca8ddc6e106dcd542e45a0d21d91a1909152ddb7b4d454eac093077ce3b.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/1313fca8ddc6e106dcd542e45a0d21d91a1909152ddb7b4d454eac093077ce3b.jpg)

![1c184844bd33d92748a15cfb911229470595daba7f64925a11d10db8bfcb7933.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/1c184844bd33d92748a15cfb911229470595daba7f64925a11d10db8bfcb7933.jpg)

![1c1ecf85e919a8a546c26ff928f5935bbdb75444ab1d0d8e6f61252dea53a975.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/1c1ecf85e919a8a546c26ff928f5935bbdb75444ab1d0d8e6f61252dea53a975.jpg)

![36f96b9de4c0bf2e8c46221bf759ebe77250e27e78a1c3050f1ece1f4f7c81ce.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/36f96b9de4c0bf2e8c46221bf759ebe77250e27e78a1c3050f1ece1f4f7c81ce.jpg)

![38c7eadaf239f9e7d90d57fd9cd71b2af348fae42cce8134565b8c7c205e5553.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/38c7eadaf239f9e7d90d57fd9cd71b2af348fae42cce8134565b8c7c205e5553.jpg)

![4ff2685427b8de14782770ae85b9b7462acb79a525bbb9bdaeb8e297c6de6816.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/4ff2685427b8de14782770ae85b9b7462acb79a525bbb9bdaeb8e297c6de6816.jpg)

![7df70981d5401a1495311da893eb0115a055f60d396c6dcb93e7932c208585a3.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/7df70981d5401a1495311da893eb0115a055f60d396c6dcb93e7932c208585a3.jpg)

![94fa9aaf92260e967ff70cd9d3b17c970e7940d1f5772b3472949b3b8e1ba1a7.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/94fa9aaf92260e967ff70cd9d3b17c970e7940d1f5772b3472949b3b8e1ba1a7.jpg)

![95182a9e10480c3b2d42440d65ad944a12ef97d3edd0acab02f02223dfefe4fb.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/95182a9e10480c3b2d42440d65ad944a12ef97d3edd0acab02f02223dfefe4fb.jpg)

![9c408020edf89e33d506cea0cb1af524a6120634714a443eddc7b1029166d438.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/9c408020edf89e33d506cea0cb1af524a6120634714a443eddc7b1029166d438.jpg)

![a00b1ae36c335adfaa1eed98f4d4072b9ea1ed2c79d63ba5c384454d270efd46.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/a00b1ae36c335adfaa1eed98f4d4072b9ea1ed2c79d63ba5c384454d270efd46.jpg)

![e8cac9325f2b3d89954299f8db639a161b6757ba5fee849ba6916a5a3a4f6c3a.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/e8cac9325f2b3d89954299f8db639a161b6757ba5fee849ba6916a5a3a4f6c3a.jpg)

![f55114ced3fad26eb6dea6002ecaee91ecac7d4b0033f3fe4020fe929d3e7c96.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/f55114ced3fad26eb6dea6002ecaee91ecac7d4b0033f3fe4020fe929d3e7c96.jpg)

![fa147f9024bf7e64bba4f623aa197a1cc06e0259be8b8042541923556f327e95.jpg](../iclr_results/172_SymmetricDiffusers_ Learning Discrete Diffusion on Finite Symmetric Groups/tables/fa147f9024bf7e64bba4f623aa197a1cc06e0259be8b8042541923556f327e95.jpg)

## Cut Your Losses in Large-Vocabulary Language Models


### Images

![1fe57519dad0f2313e7217ff3e7fc3841c5c63ad694b1099a5e200533b85a986.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/images/1fe57519dad0f2313e7217ff3e7fc3841c5c63ad694b1099a5e200533b85a986.jpg)

![36b3ab3a964ea79779db9b7aa1a251e6841a95d3aaf232d0e5236fcc0127bb77.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/images/36b3ab3a964ea79779db9b7aa1a251e6841a95d3aaf232d0e5236fcc0127bb77.jpg)

![64a05bbf2c8829ab9c3389d9f1df959b10e8a1ee4712278ea5de7a5079e59bf8.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/images/64a05bbf2c8829ab9c3389d9f1df959b10e8a1ee4712278ea5de7a5079e59bf8.jpg)

![84f586e51a67fd4b27ea124be8aa876ef586c837fe807f3d1940f6ee3c3f15cb.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/images/84f586e51a67fd4b27ea124be8aa876ef586c837fe807f3d1940f6ee3c3f15cb.jpg)

![95d5f0de2360b08d1f9615dad9922ceb80cc5111112a420a06650233b7f1963c.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/images/95d5f0de2360b08d1f9615dad9922ceb80cc5111112a420a06650233b7f1963c.jpg)

![9a35190106ebce0d570ba986197c90e30934b2320960f7d7dd66f6c0528b30a6.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/images/9a35190106ebce0d570ba986197c90e30934b2320960f7d7dd66f6c0528b30a6.jpg)

![c4ef1ac05c0fe9df5f589fa1fad84c9a7241f1d36a83a219c2a8c47a2a2a9307.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/images/c4ef1ac05c0fe9df5f589fa1fad84c9a7241f1d36a83a219c2a8c47a2a2a9307.jpg)

### Tables

![1f63097fe9ce12f0583faf9b38e4baadc7d4a6c6398e91475f76ad39a81ed3b1.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/tables/1f63097fe9ce12f0583faf9b38e4baadc7d4a6c6398e91475f76ad39a81ed3b1.jpg)

![454b6538f37da4bf53382bfd9a947a78989503917ab5e6ab6ef18e8fe15ab9e6.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/tables/454b6538f37da4bf53382bfd9a947a78989503917ab5e6ab6ef18e8fe15ab9e6.jpg)

![4a69cba34de8d167bb39bcb2edbf4e922b7a6cf04204f54c0b7e3937dbf22c36.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/tables/4a69cba34de8d167bb39bcb2edbf4e922b7a6cf04204f54c0b7e3937dbf22c36.jpg)

![6a414752cc60b5c352300e1c5fa17129b645f844f8da94e53854c77cd0a93091.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/tables/6a414752cc60b5c352300e1c5fa17129b645f844f8da94e53854c77cd0a93091.jpg)

![756563ad190dcb2d590f77b4d60751b7b3f35aad6888c426ba7bc01e56f16aea.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/tables/756563ad190dcb2d590f77b4d60751b7b3f35aad6888c426ba7bc01e56f16aea.jpg)

![766757a12a2a6cbd42e54557c6cd70ce1a27a5a703f7146171847100a9c6966e.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/tables/766757a12a2a6cbd42e54557c6cd70ce1a27a5a703f7146171847100a9c6966e.jpg)

![88303db36270dacca1cac10e5a30a0a37262533bedf78bbbf95e87964571b191.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/tables/88303db36270dacca1cac10e5a30a0a37262533bedf78bbbf95e87964571b191.jpg)

![8fa39e4270bd92620a5b7f1de1a6627745c135cd7634ed17e8f74b0e5f00312f.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/tables/8fa39e4270bd92620a5b7f1de1a6627745c135cd7634ed17e8f74b0e5f00312f.jpg)

![952d1a748a7b448ed4ba74626317e924aceee811fa147b64b55f67d4f37314a2.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/tables/952d1a748a7b448ed4ba74626317e924aceee811fa147b64b55f67d4f37314a2.jpg)

![d17485c452baee7cc029e774ca6dab8e6fef40a7ee50cebb776678f390d05f57.jpg](../iclr_results/173_Cut Your Losses in Large-Vocabulary Language Models/tables/d17485c452baee7cc029e774ca6dab8e6fef40a7ee50cebb776678f390d05f57.jpg)

## Interpreting Emergent Planning in Model-Free Reinforcement Learning


### Images

![01376b29a6f0cdb2178d890ebf40321c478d560d23a913633352daf7eff2a74c.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/01376b29a6f0cdb2178d890ebf40321c478d560d23a913633352daf7eff2a74c.jpg)

![0229fcea13ac3f049803def424803f0ebf8c9feb15e68e2ce4a1818980b557dd.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/0229fcea13ac3f049803def424803f0ebf8c9feb15e68e2ce4a1818980b557dd.jpg)

![05c3f0a7e49b4d13905eeb76cea299dd6e5d30234f507af619df54b37cf65e89.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/05c3f0a7e49b4d13905eeb76cea299dd6e5d30234f507af619df54b37cf65e89.jpg)

![05e2c9bbff6df46927079b52645d045a80e56361f9a75ced74de9a0b9f6817a0.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/05e2c9bbff6df46927079b52645d045a80e56361f9a75ced74de9a0b9f6817a0.jpg)

![082362ebfc29a7b248559b76049a8c0ab16d8446adb3115d114e31a5dc850f2f.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/082362ebfc29a7b248559b76049a8c0ab16d8446adb3115d114e31a5dc850f2f.jpg)

![0c1a825f53ec20f0b5903c13dad5537d9e13c1a2a8fa592e92f8d9ce659dfb33.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/0c1a825f53ec20f0b5903c13dad5537d9e13c1a2a8fa592e92f8d9ce659dfb33.jpg)

![0c3c9243910269a176ad1dc04ab1c3ad83bae1400516b46cafb3ae37593042b3.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/0c3c9243910269a176ad1dc04ab1c3ad83bae1400516b46cafb3ae37593042b3.jpg)

![0c4ddcde80357a22f06bda3a797e3753cae50e504f481204560dc4a4b2381f01.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/0c4ddcde80357a22f06bda3a797e3753cae50e504f481204560dc4a4b2381f01.jpg)

![0f064d62c8d12505185591422eeabddf3c35099886fbe09ae1668d35790ed372.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/0f064d62c8d12505185591422eeabddf3c35099886fbe09ae1668d35790ed372.jpg)

![0f8be8c8da76c322bace52d3d952827fd6276240caaf8a2e04a592062c9adff9.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/0f8be8c8da76c322bace52d3d952827fd6276240caaf8a2e04a592062c9adff9.jpg)

![150bf70908a7f1bb116930284738179a10fadb8da50be4c1447e5099902c8ba5.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/150bf70908a7f1bb116930284738179a10fadb8da50be4c1447e5099902c8ba5.jpg)

![153145122f7bee48c1d9e8e03b4b606cf55688c13969be51651ec0f82723b813.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/153145122f7bee48c1d9e8e03b4b606cf55688c13969be51651ec0f82723b813.jpg)

![15b571ee9ecb2e92c638d40f93f34f233fc99d43054b3f68452779d57a5d92b9.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/15b571ee9ecb2e92c638d40f93f34f233fc99d43054b3f68452779d57a5d92b9.jpg)

![16270d034526f679319e7787beafed1a7a7303e10f05d9d518433596909c1325.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/16270d034526f679319e7787beafed1a7a7303e10f05d9d518433596909c1325.jpg)

![17147258dc9e13ba1d624656d18645ba127dd3459183356885f8e47bf80afc91.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/17147258dc9e13ba1d624656d18645ba127dd3459183356885f8e47bf80afc91.jpg)

![17d8db35194cb61c87f929b9ed8dcd7e873e2d13f4f4186a8c729663de7a929b.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/17d8db35194cb61c87f929b9ed8dcd7e873e2d13f4f4186a8c729663de7a929b.jpg)

![18662e34c332c735dcb37def7622711ed74b77bcfc8f11524d9cb7e6a6f70743.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/18662e34c332c735dcb37def7622711ed74b77bcfc8f11524d9cb7e6a6f70743.jpg)

![1cf030f57606a6200859a0b28f91297c4d5e50022bf5bd3f204eaa28457047dc.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/1cf030f57606a6200859a0b28f91297c4d5e50022bf5bd3f204eaa28457047dc.jpg)

![1e95506b1dd0b66d04d9ec8df41791eaa70d1527156416c1136cdeef02be5849.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/1e95506b1dd0b66d04d9ec8df41791eaa70d1527156416c1136cdeef02be5849.jpg)

![1efe5ca1d974c3f5fc4b16912a26e286f108cd470593b001fb0015fe4f6c2b8d.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/1efe5ca1d974c3f5fc4b16912a26e286f108cd470593b001fb0015fe4f6c2b8d.jpg)

![211cae2a10bd6edb294c9314886e06fb840269a6613d1cab63cd8e67560b39e1.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/211cae2a10bd6edb294c9314886e06fb840269a6613d1cab63cd8e67560b39e1.jpg)

![26d55ad6b162f3676a33494febf2467bb4dfe8f7f6a8997485b3eb99695d69ea.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/26d55ad6b162f3676a33494febf2467bb4dfe8f7f6a8997485b3eb99695d69ea.jpg)

![2802b12131fc7496928c8c2372607732ea8b736b213ec27d064161408f8da24d.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/2802b12131fc7496928c8c2372607732ea8b736b213ec27d064161408f8da24d.jpg)

![2b7d07429cd65106564d66a18916cda8e31e486b06758f9c96df72b3deb65d33.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/2b7d07429cd65106564d66a18916cda8e31e486b06758f9c96df72b3deb65d33.jpg)

![2d836d1ab25cbdb71b671ee4a03aca2e65a31ae0e8bb4b2886c901322519d1c9.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/2d836d1ab25cbdb71b671ee4a03aca2e65a31ae0e8bb4b2886c901322519d1c9.jpg)

![2ec25d89ca9e6bd0fa5b2ceb0e6dc6d45cc384b3b9dd564e6061fadab16333ab.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/2ec25d89ca9e6bd0fa5b2ceb0e6dc6d45cc384b3b9dd564e6061fadab16333ab.jpg)

![304534e82c0e21d4e6629bcf4ea60e44fe912374b8d1b2eafd3cea63e35c6352.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/304534e82c0e21d4e6629bcf4ea60e44fe912374b8d1b2eafd3cea63e35c6352.jpg)

![328a91e72b1720eef85e4414cd594d00eed5dea70c7b0c2ba0c2b8fcffbd7dfa.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/328a91e72b1720eef85e4414cd594d00eed5dea70c7b0c2ba0c2b8fcffbd7dfa.jpg)

![330020ec57b3898219ee7e653c1117b630b349ad8c13fbca35ccb88b270cb158.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/330020ec57b3898219ee7e653c1117b630b349ad8c13fbca35ccb88b270cb158.jpg)

![35e6f236215c76e64647cfe1bf295cb54f31583404269d1fdefdc1d5ced52a3e.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/35e6f236215c76e64647cfe1bf295cb54f31583404269d1fdefdc1d5ced52a3e.jpg)

![39d212f6259955aba4d50e7e575714c607d1a923addf8781ee79e6462b596711.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/39d212f6259955aba4d50e7e575714c607d1a923addf8781ee79e6462b596711.jpg)

![3d1f30667e18788c1272e5c5e2e48211e607497cc758001edf9556e38303afc7.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/3d1f30667e18788c1272e5c5e2e48211e607497cc758001edf9556e38303afc7.jpg)

![3d5ceff7d271ad0d2e8a6779d166d46b433b7d09d8b3d1d86dfa0cbac4913da7.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/3d5ceff7d271ad0d2e8a6779d166d46b433b7d09d8b3d1d86dfa0cbac4913da7.jpg)

![3fa9aacd23c9dea714e55e863acefb65db098731c40e65b62548273c8a42233e.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/3fa9aacd23c9dea714e55e863acefb65db098731c40e65b62548273c8a42233e.jpg)

![4019df224b5caf3151851f13ad42417500e68eddccda1fb09eb521cbb01a4703.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/4019df224b5caf3151851f13ad42417500e68eddccda1fb09eb521cbb01a4703.jpg)

![409a6d4eb25de2cd867bc79949f71e7c6db51abaac0e5d855578c4b1f6c179b3.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/409a6d4eb25de2cd867bc79949f71e7c6db51abaac0e5d855578c4b1f6c179b3.jpg)

![40a97c685308c286aa10ece0aca0c7c2cf410c9705e484865e0aac406b94ba48.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/40a97c685308c286aa10ece0aca0c7c2cf410c9705e484865e0aac406b94ba48.jpg)

![40cb92e8a76219b5f578764751ff456f0131b2746581bddfaa8ab1d14c44d712.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/40cb92e8a76219b5f578764751ff456f0131b2746581bddfaa8ab1d14c44d712.jpg)

![40e80a28edb19730447bd019fc82ab035390391b4ac937c7da5fdc4dc5b27508.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/40e80a28edb19730447bd019fc82ab035390391b4ac937c7da5fdc4dc5b27508.jpg)

![41f77ccdc552b73d1506c91686d889eb2bb4624ada0bb3917a82b8d4b78f4db7.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/41f77ccdc552b73d1506c91686d889eb2bb4624ada0bb3917a82b8d4b78f4db7.jpg)

![44c43668ef21ef87e922395323f74cb80ad855b45c87538ee6748a81c0cf6dc5.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/44c43668ef21ef87e922395323f74cb80ad855b45c87538ee6748a81c0cf6dc5.jpg)

![4577fe1698b75fe1253661636157fefaafb14d0f9c0510ea856f981d351cbb09.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/4577fe1698b75fe1253661636157fefaafb14d0f9c0510ea856f981d351cbb09.jpg)

![494ae637e69d7d5f93d6beb47e7221a844f41b30c617cb542ee139ca13c2b665.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/494ae637e69d7d5f93d6beb47e7221a844f41b30c617cb542ee139ca13c2b665.jpg)

![4ab34da8696a722e45a0fe9ca1b6fd2a4293cc290067f9c4eb5e890b68210de7.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/4ab34da8696a722e45a0fe9ca1b6fd2a4293cc290067f9c4eb5e890b68210de7.jpg)

![4ba8023dcd8f83b190343487c88d87e182b132899192d85a90a47291a79ef49b.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/4ba8023dcd8f83b190343487c88d87e182b132899192d85a90a47291a79ef49b.jpg)

![4c5cb4d137324820756dc94662624d93afaad55fad2dbc577b3fc26711117dae.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/4c5cb4d137324820756dc94662624d93afaad55fad2dbc577b3fc26711117dae.jpg)

![4d723afc3cf0defb7cde71ffd5b436671d01b905e81ff08d4f31db824e9d5775.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/4d723afc3cf0defb7cde71ffd5b436671d01b905e81ff08d4f31db824e9d5775.jpg)

![51923328281f488a50702e875900f34ab079f1e5195b2c8b43fce7f9ec3f5f4e.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/51923328281f488a50702e875900f34ab079f1e5195b2c8b43fce7f9ec3f5f4e.jpg)

![54642fe1b5775e3b187aeefe3dd0d7019dd7fed049d59984ff3458a562566c30.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/54642fe1b5775e3b187aeefe3dd0d7019dd7fed049d59984ff3458a562566c30.jpg)

![56a94806c747fbd20f5dc94381227f92057cfcc8d270e503658dd4ae8b53a86b.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/56a94806c747fbd20f5dc94381227f92057cfcc8d270e503658dd4ae8b53a86b.jpg)

![587e93535b6922c3cd2f6d21d3109e2111d8b298d4ae8cdd802fff0a79151a5d.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/587e93535b6922c3cd2f6d21d3109e2111d8b298d4ae8cdd802fff0a79151a5d.jpg)

![610818dec51424c43c017bc292f42839660ca35d932ad4b271bcf57d88ebc6c0.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/610818dec51424c43c017bc292f42839660ca35d932ad4b271bcf57d88ebc6c0.jpg)

![61b3f696bd8384aced3d250e9832da4ad3a8dc1f7a34de0b962ea06277e63d8d.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/61b3f696bd8384aced3d250e9832da4ad3a8dc1f7a34de0b962ea06277e63d8d.jpg)

![6be1d66d59820a84356ca86ae2c459b3f4e591e05499b91f95b53248f517450e.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/6be1d66d59820a84356ca86ae2c459b3f4e591e05499b91f95b53248f517450e.jpg)

![6cf3aac069cee5992ceeb8d9d958c6f6a63658a49cef1338adbdc73b822a8839.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/6cf3aac069cee5992ceeb8d9d958c6f6a63658a49cef1338adbdc73b822a8839.jpg)

![6d205eab200a39b4e2ad7a529c03e271c2f23de5ae289975d039098d9d994e63.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/6d205eab200a39b4e2ad7a529c03e271c2f23de5ae289975d039098d9d994e63.jpg)

![6d2aa70882bdb1220f2713ee5b35472e926f8fda16da948fa034d4241fcceaae.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/6d2aa70882bdb1220f2713ee5b35472e926f8fda16da948fa034d4241fcceaae.jpg)

![6f8c27d749e2c83386d40b4c9ca590ad6366767df1d31325c995c4d089decac5.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/6f8c27d749e2c83386d40b4c9ca590ad6366767df1d31325c995c4d089decac5.jpg)

![7176a2041ecad328a8d0023210954e39ab1d1140b819ec6a94a8bffa719cc930.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/7176a2041ecad328a8d0023210954e39ab1d1140b819ec6a94a8bffa719cc930.jpg)

![746d24843bb566aafb844ef1076682202d174128890698bed492dfb670a61e08.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/746d24843bb566aafb844ef1076682202d174128890698bed492dfb670a61e08.jpg)

![77ab030f65f825bbe1853bc5e8b242ef87d64876b3afc88b2be92ab42ae30ec2.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/77ab030f65f825bbe1853bc5e8b242ef87d64876b3afc88b2be92ab42ae30ec2.jpg)

![7b48952ffa502b5613c8c25c6e396a301489cbbf094ed0587acbe361b791d56a.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/7b48952ffa502b5613c8c25c6e396a301489cbbf094ed0587acbe361b791d56a.jpg)

![826d05d978fe680a31bd7df203422de4bd65569cdf98ef7ac0c76f64ebd0391d.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/826d05d978fe680a31bd7df203422de4bd65569cdf98ef7ac0c76f64ebd0391d.jpg)

![82fb5709a7c150a324e3687f01241230924f288a518e62fec68f165812fd9a52.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/82fb5709a7c150a324e3687f01241230924f288a518e62fec68f165812fd9a52.jpg)

![86bc5f56f8c9f0ce542561ae4f38fe60343b93ea384faf3d877f000402100340.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/86bc5f56f8c9f0ce542561ae4f38fe60343b93ea384faf3d877f000402100340.jpg)

![86eda7cbf5e6d97fe30b5e7fa459d1e4f023f397edc39cae54580a484f09421a.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/86eda7cbf5e6d97fe30b5e7fa459d1e4f023f397edc39cae54580a484f09421a.jpg)

![8800e96cb447cdef467dcab694b23440ce3d9edbab7825084947aedb62908e50.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/8800e96cb447cdef467dcab694b23440ce3d9edbab7825084947aedb62908e50.jpg)

![89b61b6f0e5638dd4f549ab07074facf3bf0b8a2f858e118caeec6aed349caaf.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/89b61b6f0e5638dd4f549ab07074facf3bf0b8a2f858e118caeec6aed349caaf.jpg)

![8ad742a8bdc1a6736b2d3cfdab384dfc53bfc708f20a368cf5f46d321662c3e3.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/8ad742a8bdc1a6736b2d3cfdab384dfc53bfc708f20a368cf5f46d321662c3e3.jpg)

![8c4cb4efefd3fd21b74e185d730b352403559a7512c4fd106efdb6bf46948ee2.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/8c4cb4efefd3fd21b74e185d730b352403559a7512c4fd106efdb6bf46948ee2.jpg)

![8d7fa403ab125ff847d8786c39f6647847e997493a7a82c52628429771ac8bab.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/8d7fa403ab125ff847d8786c39f6647847e997493a7a82c52628429771ac8bab.jpg)

![9100110906b68ee13dd0f4a53229fd1f10e23aa049fb6f59459c5d170393080d.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/9100110906b68ee13dd0f4a53229fd1f10e23aa049fb6f59459c5d170393080d.jpg)

![912da1602a20bfbe4a76861aa082756d44f883291d0afd93c94b46fd3c022c4c.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/912da1602a20bfbe4a76861aa082756d44f883291d0afd93c94b46fd3c022c4c.jpg)

![93e4e918a97545d02573a04b48388bc5c46d131de244e19cc897cca6f2dbdb51.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/93e4e918a97545d02573a04b48388bc5c46d131de244e19cc897cca6f2dbdb51.jpg)

![94739d612a563cffdbde471b10315a520f1674bb8d8bbae545976e701d5eba03.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/94739d612a563cffdbde471b10315a520f1674bb8d8bbae545976e701d5eba03.jpg)

![9aaf08863584d40af857a214af0ca63630fa10ec52647f7b79c898551bb264c7.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/9aaf08863584d40af857a214af0ca63630fa10ec52647f7b79c898551bb264c7.jpg)

![9df7b6f3f37f8f58ee414689378b3717d2ff9055d9584996be268c0db41ff0a0.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/9df7b6f3f37f8f58ee414689378b3717d2ff9055d9584996be268c0db41ff0a0.jpg)

![9f71323ba88ff1cee9399c8f5eaf83a96260e1caeeb16abafec87c1702266074.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/9f71323ba88ff1cee9399c8f5eaf83a96260e1caeeb16abafec87c1702266074.jpg)

![a03f97991a8cf139a26a24972648b18622935103b7192d0446f4911e7db1cd61.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/a03f97991a8cf139a26a24972648b18622935103b7192d0446f4911e7db1cd61.jpg)

![a05f3b69f385be6496ba60c98b7059b43663dce3c6429e277f1265dc399f18ec.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/a05f3b69f385be6496ba60c98b7059b43663dce3c6429e277f1265dc399f18ec.jpg)

![a090a426a5f07517a02fd7a0e4833a1b5e57731ace22df6de2d594fd4be64ae6.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/a090a426a5f07517a02fd7a0e4833a1b5e57731ace22df6de2d594fd4be64ae6.jpg)

![a1cd9b627fd52eaebb4ee60973c8d7485ae050b6a21c62f1bac5b02297f24025.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/a1cd9b627fd52eaebb4ee60973c8d7485ae050b6a21c62f1bac5b02297f24025.jpg)

![a2f0773ea86e45da1b0ecfe1f88ed59aa5a7d60f3b72f9ce4b31bb7cf36c6a61.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/a2f0773ea86e45da1b0ecfe1f88ed59aa5a7d60f3b72f9ce4b31bb7cf36c6a61.jpg)

![a982b685b6355b35333bcfe01c9e06ac4dfdae2e2399c919ab58292c0ecf47ba.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/a982b685b6355b35333bcfe01c9e06ac4dfdae2e2399c919ab58292c0ecf47ba.jpg)

![ac5cc63ef95753a6cfb3f0a422a49df4466bb2b76b54a69964e8b27d89453ee1.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/ac5cc63ef95753a6cfb3f0a422a49df4466bb2b76b54a69964e8b27d89453ee1.jpg)

![ad469c4af3263f6bb1289a5f5975ff312400c9a5dcbe84381a12065ac88c029b.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/ad469c4af3263f6bb1289a5f5975ff312400c9a5dcbe84381a12065ac88c029b.jpg)

![ad90678846eee2761f9de7221e980972f143b29649375b7c24dcb1ded72b9611.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/ad90678846eee2761f9de7221e980972f143b29649375b7c24dcb1ded72b9611.jpg)

![addde885099758428352ec46ade878af2272f618b2ddbd3cec06518b3491f8b5.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/addde885099758428352ec46ade878af2272f618b2ddbd3cec06518b3491f8b5.jpg)

![ae4b8f1efd89e1c30ce8fdb1e8452e56b08efcd2b84862dc47b6615386fdcde1.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/ae4b8f1efd89e1c30ce8fdb1e8452e56b08efcd2b84862dc47b6615386fdcde1.jpg)

![b0b08632c624d25b786a247656c2b243c9227cb81eba62f3290be60861021912.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/b0b08632c624d25b786a247656c2b243c9227cb81eba62f3290be60861021912.jpg)

![b5611283c658310d7e683786790216959a11c8b092c44eed371e7ef16b83344d.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/b5611283c658310d7e683786790216959a11c8b092c44eed371e7ef16b83344d.jpg)

![b6e9919780c9521801af9d468f6273b106d3347c901cf6fc6bf5b10a90b956d6.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/b6e9919780c9521801af9d468f6273b106d3347c901cf6fc6bf5b10a90b956d6.jpg)

![b6ea0b76e31bc218e82c43d796a040e45754b94d4ab829a872db70ac59990094.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/b6ea0b76e31bc218e82c43d796a040e45754b94d4ab829a872db70ac59990094.jpg)

![b713eae540ad44d194f2440682e737430329de06559402389403617ed24027f0.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/b713eae540ad44d194f2440682e737430329de06559402389403617ed24027f0.jpg)

![b84928a116097f8e807629f8682fe0fc505a105965834e0606569973f2524344.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/b84928a116097f8e807629f8682fe0fc505a105965834e0606569973f2524344.jpg)

![b8c80ffff49dc30d9e9053f27be0103e4b11f5cd52d6e12f45887fbaba626ef0.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/b8c80ffff49dc30d9e9053f27be0103e4b11f5cd52d6e12f45887fbaba626ef0.jpg)

![b9d5cce93ca4f696093c6792082a1239e4dba6270382bed8081b423ac452ae0c.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/b9d5cce93ca4f696093c6792082a1239e4dba6270382bed8081b423ac452ae0c.jpg)

![bfcae45abf5057a152428e1ee7bb629dee2d976c9a98b3f8638347f6a0d3f703.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/bfcae45abf5057a152428e1ee7bb629dee2d976c9a98b3f8638347f6a0d3f703.jpg)

![c054468320c6324f7cfabf6c652c428812ac638e57f20d2edf9d131b87113619.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/c054468320c6324f7cfabf6c652c428812ac638e57f20d2edf9d131b87113619.jpg)

![c3edeb5d84f2bd60742dfa8b5ff5ed76e26550a21a9ddc6bf66cb2d726eeb9fa.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/c3edeb5d84f2bd60742dfa8b5ff5ed76e26550a21a9ddc6bf66cb2d726eeb9fa.jpg)

![c51c00638a7fb92bbad53a382ed830c8f0f5e634cdcdfe87b509002b98570db6.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/c51c00638a7fb92bbad53a382ed830c8f0f5e634cdcdfe87b509002b98570db6.jpg)

![c5b85fdfe56cea9eef8df3a15b6220155d159070cb95e5f4daba4caca9adaf3e.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/c5b85fdfe56cea9eef8df3a15b6220155d159070cb95e5f4daba4caca9adaf3e.jpg)

![c74fb932744fff4dd9c679c69573050a053bdad19bf56d5e638f1103a27f9dc5.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/c74fb932744fff4dd9c679c69573050a053bdad19bf56d5e638f1103a27f9dc5.jpg)

![c847c74f0e3ab3208b5eb84c1334ea03594ade381e9d82f3f04fe180d4b8a8ea.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/c847c74f0e3ab3208b5eb84c1334ea03594ade381e9d82f3f04fe180d4b8a8ea.jpg)

![ca9cbe0933fd7efc790b50215f4e3eaaa84792bcc60b5b3b9fbe84249ee8af79.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/ca9cbe0933fd7efc790b50215f4e3eaaa84792bcc60b5b3b9fbe84249ee8af79.jpg)

![cc392434249763212745cf492b8699933629c4cfc6e0c4f65a7f150238e03340.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/cc392434249763212745cf492b8699933629c4cfc6e0c4f65a7f150238e03340.jpg)

![ce18318130b568b14a3f99d635a4bc5ba788fc287da2124f0ea08eb5e44cf2e3.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/ce18318130b568b14a3f99d635a4bc5ba788fc287da2124f0ea08eb5e44cf2e3.jpg)

![d0a5aa69c145ef3b61335103246378ee7bb11392962bdcbd4ea298a4e9c4819f.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/d0a5aa69c145ef3b61335103246378ee7bb11392962bdcbd4ea298a4e9c4819f.jpg)

![d0bc88296b7cbc6dca3f11cbb0be9184c38442253734176a0a7dc485fc600d55.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/d0bc88296b7cbc6dca3f11cbb0be9184c38442253734176a0a7dc485fc600d55.jpg)

![d0fca59043609e04ccbc6e8498396f835e4c24f10ce198bcda3ae2e3c7a43aaf.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/d0fca59043609e04ccbc6e8498396f835e4c24f10ce198bcda3ae2e3c7a43aaf.jpg)

![d21efdcb50460dba19bb6abe9e1bcd88533c02fa0a678cde0f788a5d3fcf9916.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/d21efdcb50460dba19bb6abe9e1bcd88533c02fa0a678cde0f788a5d3fcf9916.jpg)

![d3b142c5755cd96a1aeefc1479886388435a79f4f23698d491c50e78da1d0008.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/d3b142c5755cd96a1aeefc1479886388435a79f4f23698d491c50e78da1d0008.jpg)

![d56443afe920f538eae3920b7f1c6b03b72cdbb31478d7e4dc261204c22e406e.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/d56443afe920f538eae3920b7f1c6b03b72cdbb31478d7e4dc261204c22e406e.jpg)

![d6cf4f29d3e2b94ee36e046e9b9c9837e74c435a7d85e9204ef589028452d2ed.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/d6cf4f29d3e2b94ee36e046e9b9c9837e74c435a7d85e9204ef589028452d2ed.jpg)

![d8383d70be6ce43b9d15ae4eecb7cdd04036972418273f3c4de185c88c7c268a.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/d8383d70be6ce43b9d15ae4eecb7cdd04036972418273f3c4de185c88c7c268a.jpg)

![db20f5eb48bc00743d6101dbdc649842779792fa04aa8963fbffcc2794ea9008.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/db20f5eb48bc00743d6101dbdc649842779792fa04aa8963fbffcc2794ea9008.jpg)

![ddc46936e70ae93ae758683dceb0c2d860df5ab7f7ae660ca104cbfe303e421b.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/ddc46936e70ae93ae758683dceb0c2d860df5ab7f7ae660ca104cbfe303e421b.jpg)

![de2999eadd2acf9242cb567b930b92c89b72699cf3da149d47a2f29afb10c54b.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/de2999eadd2acf9242cb567b930b92c89b72699cf3da149d47a2f29afb10c54b.jpg)

![de9b2e83c3171ed7c54064803d1e48e591e7c93b8fe400b0e1f4d7daad8fe683.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/de9b2e83c3171ed7c54064803d1e48e591e7c93b8fe400b0e1f4d7daad8fe683.jpg)

![deb5034ed5d1bc4f9bdd0da86c8881998785eb54073b8a19f2d2158fb338d4b1.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/deb5034ed5d1bc4f9bdd0da86c8881998785eb54073b8a19f2d2158fb338d4b1.jpg)

![e123825a2d33d2a17057a004ff0fda487e8551ca65be0ed3a1863968bea6e07c.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/e123825a2d33d2a17057a004ff0fda487e8551ca65be0ed3a1863968bea6e07c.jpg)

![e7e29f9e02c29d8564332fc22b7f83300ab5f8e402789b99a47fa96bd6b1518f.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/e7e29f9e02c29d8564332fc22b7f83300ab5f8e402789b99a47fa96bd6b1518f.jpg)

![e980e1ed9328505dbda8d94b56d2c368411cdf0d32180755764d9d180b0eb0ea.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/e980e1ed9328505dbda8d94b56d2c368411cdf0d32180755764d9d180b0eb0ea.jpg)

![e9d9a689661127fceebc03643e36bd218fe583db27c4cf0147812c92c43b32b5.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/e9d9a689661127fceebc03643e36bd218fe583db27c4cf0147812c92c43b32b5.jpg)

![ecdf9eb1a6336b7629aa409396be0e691740fc837510131c580a0d75bb794d40.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/ecdf9eb1a6336b7629aa409396be0e691740fc837510131c580a0d75bb794d40.jpg)

![ee018925b52e76b0908b59794b7ffe940907d1a0cd38235b24c4adcdf936088c.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/ee018925b52e76b0908b59794b7ffe940907d1a0cd38235b24c4adcdf936088c.jpg)

![f3088319b6d44f2a8cafe076c403bf44e666e2c6fa590904b7db863baf7e2122.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/f3088319b6d44f2a8cafe076c403bf44e666e2c6fa590904b7db863baf7e2122.jpg)

![f58044c3d7b1380a738a57a6e848c899410a21e1aa3e1c2c7ad2dab87b7ed03d.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/f58044c3d7b1380a738a57a6e848c899410a21e1aa3e1c2c7ad2dab87b7ed03d.jpg)

![f762a82227e5c0357cbcf0050b5426575f4df061d8a6d82039a3c17be87645ad.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/f762a82227e5c0357cbcf0050b5426575f4df061d8a6d82039a3c17be87645ad.jpg)

![f7aa66bd69e01add2479abd6e93ab6c4f23db060a2fe36687b231bf43d4c7b92.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/f7aa66bd69e01add2479abd6e93ab6c4f23db060a2fe36687b231bf43d4c7b92.jpg)

![f7afe24a4434846da6954cc59465260dc976a88cb4426e89fa7105d5cc0e5ed5.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/f7afe24a4434846da6954cc59465260dc976a88cb4426e89fa7105d5cc0e5ed5.jpg)

![f8d8bcbda005c1bb4ce71683cec949b1154db3ccca7d554ad666d41b8b320353.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/f8d8bcbda005c1bb4ce71683cec949b1154db3ccca7d554ad666d41b8b320353.jpg)

![f90090287a739330f8b6a7e4e52788231c06e826c2eb4087a7814ca121f70df9.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/f90090287a739330f8b6a7e4e52788231c06e826c2eb4087a7814ca121f70df9.jpg)

![f9d459050b73872f2f1b37c4f016feff390c734dd10a09a6a798305ba4e6ff76.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/f9d459050b73872f2f1b37c4f016feff390c734dd10a09a6a798305ba4e6ff76.jpg)

![fb1702c9bda1b6b2d85d3063e23a6be50e2caf6efc7745c3bfc2851b5d93fce7.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/fb1702c9bda1b6b2d85d3063e23a6be50e2caf6efc7745c3bfc2851b5d93fce7.jpg)

![fc4d2009a31dd39008887a8f4dd7143f95a075b68b4fcbcb8be822689dd835b6.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/fc4d2009a31dd39008887a8f4dd7143f95a075b68b4fcbcb8be822689dd835b6.jpg)

![fd897daa3e3cfa0b96154b165683064cf7f54bf9d4dad80c08c63f58c4c3bfab.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/images/fd897daa3e3cfa0b96154b165683064cf7f54bf9d4dad80c08c63f58c4c3bfab.jpg)

### Tables

![1763e899e1c5b64885da21194c950452738ede9d5e1a66387f3af9604b5040a6.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/tables/1763e899e1c5b64885da21194c950452738ede9d5e1a66387f3af9604b5040a6.jpg)

![223e61cd622921625424aaa8ba42c092ef22237547c973169f79855fe367af50.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/tables/223e61cd622921625424aaa8ba42c092ef22237547c973169f79855fe367af50.jpg)

![34c3e9527fbd4cf19b34fff9551e08b569a0212a4b6c1b7abe6b8238a99584ef.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/tables/34c3e9527fbd4cf19b34fff9551e08b569a0212a4b6c1b7abe6b8238a99584ef.jpg)

![68cc4ec3793e7a6cc653d14a3f4d5a969dbaaf9149e98fb5cd382325c225b23f.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/tables/68cc4ec3793e7a6cc653d14a3f4d5a969dbaaf9149e98fb5cd382325c225b23f.jpg)

![91c50ba0e5fef3e1000dbb5755b3794f6c8dda2e1f33929d306819026161baef.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/tables/91c50ba0e5fef3e1000dbb5755b3794f6c8dda2e1f33929d306819026161baef.jpg)

![e2ddddf382b05742b629f02e6c61999ff441d49f128983779b3e07211e7df256.jpg](../iclr_results/174_Interpreting Emergent Planning in Model-Free Reinforcement Learning/tables/e2ddddf382b05742b629f02e6c61999ff441d49f128983779b3e07211e7df256.jpg)

## Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think


### Images

![0107d0f831daeb3473012ac6a2badb093bc4417c98f0576869f4d31b3b16e71d.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/0107d0f831daeb3473012ac6a2badb093bc4417c98f0576869f4d31b3b16e71d.jpg)

![09211f130957b86df6045a7ebbb2624fe948196e2021f678a8328f40933597f1.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/09211f130957b86df6045a7ebbb2624fe948196e2021f678a8328f40933597f1.jpg)

![0a950e1f9eb444337dbe210d5ea79ee4886b31a57f0af47a8c5d8c37748c0429.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/0a950e1f9eb444337dbe210d5ea79ee4886b31a57f0af47a8c5d8c37748c0429.jpg)

![1817c02fb3edd3617415725bbd5d3630a91e5fcc0495ca798a6ba1963eb3820a.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/1817c02fb3edd3617415725bbd5d3630a91e5fcc0495ca798a6ba1963eb3820a.jpg)

![1d49557ebb8812ef09dc7372be5dadf5f9e6feccc27ea14d98c081b0394a0cc0.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/1d49557ebb8812ef09dc7372be5dadf5f9e6feccc27ea14d98c081b0394a0cc0.jpg)

![1db5c6b8bcbebc367e25b22d6f1cb69b713a72d45dbe0d57611dd120bad137cf.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/1db5c6b8bcbebc367e25b22d6f1cb69b713a72d45dbe0d57611dd120bad137cf.jpg)

![1ff3db34963785076f86d51ed45903dac8c5af1f801a5ebf22b7c6141982758a.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/1ff3db34963785076f86d51ed45903dac8c5af1f801a5ebf22b7c6141982758a.jpg)

![20aaea43be9a020d9e732f3bab6c24293121ef406012ff6064d3f39179598b7c.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/20aaea43be9a020d9e732f3bab6c24293121ef406012ff6064d3f39179598b7c.jpg)

![2d11bf0b8d350bd96c5b7213ea15508aeb87ca48a52480b60927129312abc254.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/2d11bf0b8d350bd96c5b7213ea15508aeb87ca48a52480b60927129312abc254.jpg)

![2d2e87933fa0511cbc2a6250e285809f6abd7c1734f27e3d1c168fffbcd7dc40.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/2d2e87933fa0511cbc2a6250e285809f6abd7c1734f27e3d1c168fffbcd7dc40.jpg)

![3e5d9bc9677e1199faafa06984943cab195d8118f821fa18e92d8f1444ecd9e8.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/3e5d9bc9677e1199faafa06984943cab195d8118f821fa18e92d8f1444ecd9e8.jpg)

![44b527faac2acc418f529144b4678bb0e9e403c154b10766edd83cf3d010edb2.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/44b527faac2acc418f529144b4678bb0e9e403c154b10766edd83cf3d010edb2.jpg)

![44b9c7e9e7b8a734af762f5f637982ab4223a6304c5b2c4989b8c446c8fd2419.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/44b9c7e9e7b8a734af762f5f637982ab4223a6304c5b2c4989b8c446c8fd2419.jpg)

![46c79d1723eaffc0145e05a7116b1ba38baaae171a325201af47552cb3cf93cc.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/46c79d1723eaffc0145e05a7116b1ba38baaae171a325201af47552cb3cf93cc.jpg)

![542535cd109a284c91db7564a91a5c120fcf806467bd8c79626e69d7e185fbe0.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/542535cd109a284c91db7564a91a5c120fcf806467bd8c79626e69d7e185fbe0.jpg)

![5eaffb4614df9dbca0dcfd97f3586f5922032fd4e43998669f0d3d29fbcf6264.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/5eaffb4614df9dbca0dcfd97f3586f5922032fd4e43998669f0d3d29fbcf6264.jpg)

![64dc6891a5a1a40c30ee64d99107eb664547b03b0b9ef227d4dd8bdf3f1167ef.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/64dc6891a5a1a40c30ee64d99107eb664547b03b0b9ef227d4dd8bdf3f1167ef.jpg)

![6ce63cddcdf0d4db37d37240666eecc2c10497cc30d5c746532d09ca66ed9d2c.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/6ce63cddcdf0d4db37d37240666eecc2c10497cc30d5c746532d09ca66ed9d2c.jpg)

![6f157585b67f6e96f8302b0171955ebac75ee1c0a0971760f1532da2e4147d18.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/6f157585b67f6e96f8302b0171955ebac75ee1c0a0971760f1532da2e4147d18.jpg)

![76d5a362784ddf6a60d02d2c6e82cdc7c63c4957f9ca3f152d47ec401b5c3fd8.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/76d5a362784ddf6a60d02d2c6e82cdc7c63c4957f9ca3f152d47ec401b5c3fd8.jpg)

![79afe0b0f8869ee4bb750ff6993963a7da08cf0aef792d9ba6d8a77402b6d6f7.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/79afe0b0f8869ee4bb750ff6993963a7da08cf0aef792d9ba6d8a77402b6d6f7.jpg)

![9a70af58fc4eb8d4c410ad3ffb178d496a3e9c0c8c4d9a39ca16d68d247f5b54.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/9a70af58fc4eb8d4c410ad3ffb178d496a3e9c0c8c4d9a39ca16d68d247f5b54.jpg)

![a08cca8f8003de075b1e423a8d1848be20357047f041d45a76d9e3f356cc8154.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/a08cca8f8003de075b1e423a8d1848be20357047f041d45a76d9e3f356cc8154.jpg)

![b13a0c3c649c1355f189be129a429c4d7c5dcaad7f9ddfa317db21fda44f0d27.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/b13a0c3c649c1355f189be129a429c4d7c5dcaad7f9ddfa317db21fda44f0d27.jpg)

![b1f86d5f2e7821e2bb9e29c126c487cf5a3b09adb5a5cd8c5f104a32133e4495.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/b1f86d5f2e7821e2bb9e29c126c487cf5a3b09adb5a5cd8c5f104a32133e4495.jpg)

![b23e86a43873ead2f52e3d8295f77908ef2a1d59210e1cea958d0c0aa6cd0944.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/b23e86a43873ead2f52e3d8295f77908ef2a1d59210e1cea958d0c0aa6cd0944.jpg)

![cc566a23015d971bd675a4cdd43ea236041809dfb991e58c70a149c5eac0ec53.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/cc566a23015d971bd675a4cdd43ea236041809dfb991e58c70a149c5eac0ec53.jpg)

![cd08935b4220ace09458fc49024451a6bc153992b1e298b79da3662db3a7a663.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/cd08935b4220ace09458fc49024451a6bc153992b1e298b79da3662db3a7a663.jpg)

![d771a2342b358ea089d136e7177aab622933ac3a24b8a899da5b6129a4cca650.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/d771a2342b358ea089d136e7177aab622933ac3a24b8a899da5b6129a4cca650.jpg)

![d8b83c1bdfebe08c68a403d3fee5be573bc46f0293a2a9323b6945f3ac090fae.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/d8b83c1bdfebe08c68a403d3fee5be573bc46f0293a2a9323b6945f3ac090fae.jpg)

![d8d521725786762f65e6790149d7d45e07d6dc7019e609ced9a98a8fa68ed1e4.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/d8d521725786762f65e6790149d7d45e07d6dc7019e609ced9a98a8fa68ed1e4.jpg)

![dbb356174454229e323fef57a1a23f22e05548458defca28d2d3451bf70b5ee2.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/dbb356174454229e323fef57a1a23f22e05548458defca28d2d3451bf70b5ee2.jpg)

![dfc079e5815a2d67de9eedfaf476f1fd243b52220fabbd2b8f60ea5fcf61a63b.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/dfc079e5815a2d67de9eedfaf476f1fd243b52220fabbd2b8f60ea5fcf61a63b.jpg)

![e5ffb98578468d30d86d9b61945beaac6d3c9668774e55b4871235a9bed9763a.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/e5ffb98578468d30d86d9b61945beaac6d3c9668774e55b4871235a9bed9763a.jpg)

![f4305ef79a6803d83ff1899ccdf1af4399bac41f09b28c71df0877fbd958bead.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/f4305ef79a6803d83ff1899ccdf1af4399bac41f09b28c71df0877fbd958bead.jpg)

![f5a3bc8c6c98f10f29afe4d4e3a4d4350a12bba1942eda5c9855bec76b45ca90.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/f5a3bc8c6c98f10f29afe4d4e3a4d4350a12bba1942eda5c9855bec76b45ca90.jpg)

![fe2aa458ef6272bf131deded8bda73a7d724d9a5a2ae5c2fc7c6b7264b60b81f.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/images/fe2aa458ef6272bf131deded8bda73a7d724d9a5a2ae5c2fc7c6b7264b60b81f.jpg)

### Tables

![4e46777d1f091e6f8233186d8733d01f21d647e22c1d0e311695227f11f17b8f.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/tables/4e46777d1f091e6f8233186d8733d01f21d647e22c1d0e311695227f11f17b8f.jpg)

![63adc2c87fc6e36ebc35fbfda62dcf95298a0c84a1bef1b03605f8ee65f32fe0.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/tables/63adc2c87fc6e36ebc35fbfda62dcf95298a0c84a1bef1b03605f8ee65f32fe0.jpg)

![7881f65624c14b3f7ba1112cdd2cbd4c5ddd63ff4e17c78d1865ece269715408.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/tables/7881f65624c14b3f7ba1112cdd2cbd4c5ddd63ff4e17c78d1865ece269715408.jpg)

![7c5a31cb17f579d0432fa352a0f6f7ede3ab3f92acd17d776c5a9c3eb7c63c39.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/tables/7c5a31cb17f579d0432fa352a0f6f7ede3ab3f92acd17d776c5a9c3eb7c63c39.jpg)

![9874416571973aad40fc6f9cc7defb76f4b668572016365113f0d831a9498f07.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/tables/9874416571973aad40fc6f9cc7defb76f4b668572016365113f0d831a9498f07.jpg)

![a604bcbc829fe6b58367c26fd7a710b64cf22be09cede1c102c1c910b21e2034.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/tables/a604bcbc829fe6b58367c26fd7a710b64cf22be09cede1c102c1c910b21e2034.jpg)

![b00eefd5ca6dd3c06ccf57bef9e13389cb1bb2a3429c187f17c3cadabec454a4.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/tables/b00eefd5ca6dd3c06ccf57bef9e13389cb1bb2a3429c187f17c3cadabec454a4.jpg)

![d4455c1d001a07b3c65bd92d5de811c569404816adee323d887d9f4d7cb69693.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/tables/d4455c1d001a07b3c65bd92d5de811c569404816adee323d887d9f4d7cb69693.jpg)

![e372873ce2279f2c81c1cc121412e99083006e03d7f3ccb6d443cea09d9a1562.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/tables/e372873ce2279f2c81c1cc121412e99083006e03d7f3ccb6d443cea09d9a1562.jpg)

![e6a69ce960c261540074d8d1abb9576928df0fdff6a084a89e3075b3f716d568.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/tables/e6a69ce960c261540074d8d1abb9576928df0fdff6a084a89e3075b3f716d568.jpg)

![f05b77651074925f37dc0cc406627eeb91b01ae63f2aa4abd5438f7ff0ad974e.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/tables/f05b77651074925f37dc0cc406627eeb91b01ae63f2aa4abd5438f7ff0ad974e.jpg)

![fd9709cb06a6156f679925ca7279622337c719f622658ca7c0d53d702c8a474b.jpg](../iclr_results/175_Representation Alignment for Generation_ Training Diffusion Transformers Is Easier Than You Think/tables/fd9709cb06a6156f679925ca7279622337c719f622658ca7c0d53d702c8a474b.jpg)

## Progressive Compression with Universally Quantized Diffusion Models


### Images

![02193f1b031adf6155367104a67d54da589264d169cc2f9e91d1eb750630439f.jpg](../iclr_results/176_Progressive Compression with Universally Quantized Diffusion Models/images/02193f1b031adf6155367104a67d54da589264d169cc2f9e91d1eb750630439f.jpg)

![023c98bdc20e37f1b7ed28353676bb20e67081801f091f8bb608f513b610269b.jpg](../iclr_results/176_Progressive Compression with Universally Quantized Diffusion Models/images/023c98bdc20e37f1b7ed28353676bb20e67081801f091f8bb608f513b610269b.jpg)

![074353674b0cd6854162fe3d43a1a610b6f9515b46c51d123144e492fac552ea.jpg](../iclr_results/176_Progressive Compression with Universally Quantized Diffusion Models/images/074353674b0cd6854162fe3d43a1a610b6f9515b46c51d123144e492fac552ea.jpg)

![0d6318261afccae5ed7cccc55736dc7e591e5c1ef1fa04858a4a54860ba09b4b.jpg](../iclr_results/176_Progressive Compression with Universally Quantized Diffusion Models/images/0d6318261afccae5ed7cccc55736dc7e591e5c1ef1fa04858a4a54860ba09b4b.jpg)

![1508bb9aafce5f0b74820ed70395355e945f892c0be4e0d554be51f8058c8b12.jpg](../iclr_results/176_Progressive Compression with Universally Quantized Diffusion Models/images/1508bb9aafce5f0b74820ed70395355e945f892c0be4e0d554be51f8058c8b12.jpg)

![1bf230ef64e508c9eed7aaabb89761fc15b18d68cda0e8f07bc7a91877db1f85.jpg](../iclr_results/176_Progressive Compression with Universally Quantized Diffusion Models/images/1bf230ef64e508c9eed7aaabb89761fc15b18d68cda0e8f07bc7a91877db1f85.jpg)

![279bf76c481f5ad1181bdc02b78bba0f06135fc6ebe61a9f6ede67b168f63855.jpg](../iclr_results/176_Progressive Compression with Universally Quantized Diffusion Models/images/279bf76c481f5ad1181bdc02b78bba0f06135fc6ebe61a9f6ede67b168f63855.jpg)

![3cb508e1da0d8d4abbe48a3b1ed20ec5da6bd3f6de2b798d67819c37412ceec6.jpg](../iclr_results/176_Progressive Compression with Universally Quantized Diffusion Models/images/3cb508e1da0d8d4abbe48a3b1ed20ec5da6bd3f6de2b798d67819c37412ceec6.jpg)

![4b66aa06f081c8084f80aa229e9cf5d4b9868c18ec3b8aeae818573351901809.jpg](../iclr_results/176_Progressive Compression with Universally Quantized Diffusion Models/images/4b66aa06f081c8084f80aa229e9cf5d4b9868c18ec3b8aeae818573351901809.jpg)

![ebf217c37b7d2ab4922d22fbbe7177c6622f63399457d590743bd54ae9429871.jpg](../iclr_results/176_Progressive Compression with Universally Quantized Diffusion Models/images/ebf217c37b7d2ab4922d22fbbe7177c6622f63399457d590743bd54ae9429871.jpg)

## High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian Representation


### Images

![1e03b7a37a7857cb306a5b09faf2cad63ce69c47412507642344c58097021e30.jpg](../iclr_results/177_High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian/images/1e03b7a37a7857cb306a5b09faf2cad63ce69c47412507642344c58097021e30.jpg)

![1e9029a1ac6ebc94ff94bc586b1d198c0541fe379814a2095d27306dd130a43d.jpg](../iclr_results/177_High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian/images/1e9029a1ac6ebc94ff94bc586b1d198c0541fe379814a2095d27306dd130a43d.jpg)

![393ecbad23be9e34291e3523ad7fc6a4b462449b519972b3aeb6c08873b55942.jpg](../iclr_results/177_High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian/images/393ecbad23be9e34291e3523ad7fc6a4b462449b519972b3aeb6c08873b55942.jpg)

![753ab7ac44967ce281ac41d7c514f3d9251502f330b40729dc94d1e2ee80052a.jpg](../iclr_results/177_High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian/images/753ab7ac44967ce281ac41d7c514f3d9251502f330b40729dc94d1e2ee80052a.jpg)

![88cff284e93a0562aa1f8995845edc7a08f9ef350ad29b3461755279d5aa2203.jpg](../iclr_results/177_High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian/images/88cff284e93a0562aa1f8995845edc7a08f9ef350ad29b3461755279d5aa2203.jpg)

![9418754d52570ead0e6d98ad9f605ba179ad3c0769cddad1edae6365d95f610c.jpg](../iclr_results/177_High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian/images/9418754d52570ead0e6d98ad9f605ba179ad3c0769cddad1edae6365d95f610c.jpg)

![ac3cf20d5deda4bba1c27d81cfff51d3325c807b058d282c0f62fddb60d225a0.jpg](../iclr_results/177_High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian/images/ac3cf20d5deda4bba1c27d81cfff51d3325c807b058d282c0f62fddb60d225a0.jpg)

![ae940d9707b0bc913d53a0a60c45a72fcfd380242d49bc654565370391978b86.jpg](../iclr_results/177_High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian/images/ae940d9707b0bc913d53a0a60c45a72fcfd380242d49bc654565370391978b86.jpg)

![ca1450cf75aedbfe6f469514fb06d56d2935c06da4fe0913a6da87f05456e73d.jpg](../iclr_results/177_High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian/images/ca1450cf75aedbfe6f469514fb06d56d2935c06da4fe0913a6da87f05456e73d.jpg)

![cad45b6470a7691c90738ef587fab59d591cd18177bf137319e60279ebbb7cdc.jpg](../iclr_results/177_High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian/images/cad45b6470a7691c90738ef587fab59d591cd18177bf137319e60279ebbb7cdc.jpg)

![d29b77b0bcf16b90fa472892793bb48e0049723e18e93af88f30eaf93929a7ce.jpg](../iclr_results/177_High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian/images/d29b77b0bcf16b90fa472892793bb48e0049723e18e93af88f30eaf93929a7ce.jpg)

![d527d83aed6db3a232324acfb6f4d38e3bf476d5890cebd83126c4a87c7a95ef.jpg](../iclr_results/177_High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian/images/d527d83aed6db3a232324acfb6f4d38e3bf476d5890cebd83126c4a87c7a95ef.jpg)

![e6475e45c5885b50e6ff0d4f7b837f313beef946f86be0bd664ba654453e3ad9.jpg](../iclr_results/177_High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian/images/e6475e45c5885b50e6ff0d4f7b837f313beef946f86be0bd664ba654453e3ad9.jpg)

### Tables

![29d83d6363246c5e4db4205d7c633971ad0331855a5d7569888a7757d6443dbc.jpg](../iclr_results/177_High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian/tables/29d83d6363246c5e4db4205d7c633971ad0331855a5d7569888a7757d6443dbc.jpg)

![53a93840c1df78a7f853f47300bb4cd902edac24c154a9911c576dceb9adb8b0.jpg](../iclr_results/177_High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian/tables/53a93840c1df78a7f853f47300bb4cd902edac24c154a9911c576dceb9adb8b0.jpg)

![bb10e13a717c7723da5fddb61050e4d603daf9fb5bfca516a4a2916c592a9881.jpg](../iclr_results/177_High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian/tables/bb10e13a717c7723da5fddb61050e4d603daf9fb5bfca516a4a2916c592a9881.jpg)

![bb5a9032a85851a0866cbe6b8b454f33dbebdedbcead239d54fe20454437af13.jpg](../iclr_results/177_High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian/tables/bb5a9032a85851a0866cbe6b8b454f33dbebdedbcead239d54fe20454437af13.jpg)

![ce73090dcc433424e8a6ac611e91ffd2735a9dbe909fbaef4e923e725b031bb6.jpg](../iclr_results/177_High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian/tables/ce73090dcc433424e8a6ac611e91ffd2735a9dbe909fbaef4e923e725b031bb6.jpg)

![ebb67aa8c2db4a78c1136358a05109dd45e691d542a0589be211a8d2da233a36.jpg](../iclr_results/177_High-Dynamic Radar Sequence Prediction for Weather Nowcasting Using Spatiotemporal Coherent Gaussian/tables/ebb67aa8c2db4a78c1136358a05109dd45e691d542a0589be211a8d2da233a36.jpg)

## Training Language Models to Self-Correct via Reinforcement Learning


### Images

![0aeae14cbb110cdb9df17c4e28c66ddeebfd87aa2323e65c25cc26ae9a3168c9.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/images/0aeae14cbb110cdb9df17c4e28c66ddeebfd87aa2323e65c25cc26ae9a3168c9.jpg)

![4624ffc1fa4c18542031d73883a1b221eb6156f8ef1360bc89a8d10b4e32d55a.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/images/4624ffc1fa4c18542031d73883a1b221eb6156f8ef1360bc89a8d10b4e32d55a.jpg)

![489a513448410a1fed339e78b4cd5b6b977892e138dee984ce4b137353e9f4a1.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/images/489a513448410a1fed339e78b4cd5b6b977892e138dee984ce4b137353e9f4a1.jpg)

![4c57b00f6c35fc25038f08959904ea16d080327885feeb0e449e614f10ee142c.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/images/4c57b00f6c35fc25038f08959904ea16d080327885feeb0e449e614f10ee142c.jpg)

![64adacab4ad16cf173fb7cfb03b672d975ee88e0d71225a8814c8f2bb24541af.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/images/64adacab4ad16cf173fb7cfb03b672d975ee88e0d71225a8814c8f2bb24541af.jpg)

![6fcd00183463da0a358cda3fb25a0b71899ec5364a8c50db4e0a99c1591dcff8.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/images/6fcd00183463da0a358cda3fb25a0b71899ec5364a8c50db4e0a99c1591dcff8.jpg)

![7063fb0b5e617b4b1e4dd5fc0ee513e257cb9de7f59ca93a804eab1ef9aae7b6.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/images/7063fb0b5e617b4b1e4dd5fc0ee513e257cb9de7f59ca93a804eab1ef9aae7b6.jpg)

![800f3ae09c603d5b8a19a09d35388c64158d83ffbed37e098c1e8aee1696db98.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/images/800f3ae09c603d5b8a19a09d35388c64158d83ffbed37e098c1e8aee1696db98.jpg)

![82ef2b50b1facf6012f793a91b12c6eecca58cbeaffd647dd06347cbcf2c3d8c.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/images/82ef2b50b1facf6012f793a91b12c6eecca58cbeaffd647dd06347cbcf2c3d8c.jpg)

![bb4d561d4992913185a894052836df49be0325d7a7f1300c1c6f517744400153.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/images/bb4d561d4992913185a894052836df49be0325d7a7f1300c1c6f517744400153.jpg)

![f0bdc23e122e5aff0797564c0d893bd9d2c5054c6305b5d4351bd452a859fcf1.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/images/f0bdc23e122e5aff0797564c0d893bd9d2c5054c6305b5d4351bd452a859fcf1.jpg)

### Tables

![1284a74d21a8fb3c78ce0b54a07981119be5da1d3a565d17d2614a5c8d9e0687.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/tables/1284a74d21a8fb3c78ce0b54a07981119be5da1d3a565d17d2614a5c8d9e0687.jpg)

![1fb6598a128cba680cfe0e3f8805f814a4923f50785af3e9ef271d69c4041a09.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/tables/1fb6598a128cba680cfe0e3f8805f814a4923f50785af3e9ef271d69c4041a09.jpg)

![27a1b28b245cf5cd2b4b0751eb0f7aab70edbd828ed2198700ac11647977626e.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/tables/27a1b28b245cf5cd2b4b0751eb0f7aab70edbd828ed2198700ac11647977626e.jpg)

![29fe97a1cf6ff53d748282090540676addc75365c0b1e48145a21ac7d9f568a3.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/tables/29fe97a1cf6ff53d748282090540676addc75365c0b1e48145a21ac7d9f568a3.jpg)

![2db320b86a954ea433b128c81ec9328dba0f42707875cdbe550d42fdc5c59724.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/tables/2db320b86a954ea433b128c81ec9328dba0f42707875cdbe550d42fdc5c59724.jpg)

![71e46951340cdf38f8207eb1fa65c9f9a4d20c8ead4ad8c151d58ac911af46dc.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/tables/71e46951340cdf38f8207eb1fa65c9f9a4d20c8ead4ad8c151d58ac911af46dc.jpg)

![72c17a289b394b58e85bc8653e2fb7288a7a4aa2e3bb5225ff749069908d4975.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/tables/72c17a289b394b58e85bc8653e2fb7288a7a4aa2e3bb5225ff749069908d4975.jpg)

![7e0fbffa441703fef269d053a7a51fd8c079f39f33bf99f17c9e5f1f8ecda087.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/tables/7e0fbffa441703fef269d053a7a51fd8c079f39f33bf99f17c9e5f1f8ecda087.jpg)

![fb5656de444d1e0b33d62216873d5a49164d778a0572856d4d09dc7a4762690c.jpg](../iclr_results/178_Training Language Models to Self-Correct via Reinforcement Learning/tables/fb5656de444d1e0b33d62216873d5a49164d778a0572856d4d09dc7a4762690c.jpg)

## Open-YOLO 3D: Towards Fast and Accurate Open-Vocabulary 3D Instance Segmentation


### Images

![56e7b80e7ff5b90cdcd8b671daae787d20ee8cc3c64f768e8537094eb825c76c.jpg](../iclr_results/179_Open-YOLO 3D_ Towards Fast and Accurate Open-Vocabulary 3D Instance Segmentation/images/56e7b80e7ff5b90cdcd8b671daae787d20ee8cc3c64f768e8537094eb825c76c.jpg)

![e407f145a3784fd5e66e872468961d8dfda8325fc3655f692f179f20aa3962bf.jpg](../iclr_results/179_Open-YOLO 3D_ Towards Fast and Accurate Open-Vocabulary 3D Instance Segmentation/images/e407f145a3784fd5e66e872468961d8dfda8325fc3655f692f179f20aa3962bf.jpg)

![eb846dab3132af188e0629f22dddde78d60e705116a6a5fe9c6093630adc68fc.jpg](../iclr_results/179_Open-YOLO 3D_ Towards Fast and Accurate Open-Vocabulary 3D Instance Segmentation/images/eb846dab3132af188e0629f22dddde78d60e705116a6a5fe9c6093630adc68fc.jpg)

![f2cb97243b8cff717227af2a7a9684c4a2d6b149fd87a3b28af92edd784a788f.jpg](../iclr_results/179_Open-YOLO 3D_ Towards Fast and Accurate Open-Vocabulary 3D Instance Segmentation/images/f2cb97243b8cff717227af2a7a9684c4a2d6b149fd87a3b28af92edd784a788f.jpg)

### Tables

![0357964705001993624d88f1805045e7fd891470a76c853eae6e1c27e7564633.jpg](../iclr_results/179_Open-YOLO 3D_ Towards Fast and Accurate Open-Vocabulary 3D Instance Segmentation/tables/0357964705001993624d88f1805045e7fd891470a76c853eae6e1c27e7564633.jpg)

![0f18029a23aca47ea73d36be6e30c06da12b2a97f5c08d00b58337bef20ad098.jpg](../iclr_results/179_Open-YOLO 3D_ Towards Fast and Accurate Open-Vocabulary 3D Instance Segmentation/tables/0f18029a23aca47ea73d36be6e30c06da12b2a97f5c08d00b58337bef20ad098.jpg)

![246cc3cff459159d323b4fb4f99c2d58d359d7380c879a02d1a5158c02afde9d.jpg](../iclr_results/179_Open-YOLO 3D_ Towards Fast and Accurate Open-Vocabulary 3D Instance Segmentation/tables/246cc3cff459159d323b4fb4f99c2d58d359d7380c879a02d1a5158c02afde9d.jpg)

![532a354155020ce5d970de2219e764600828c8030753091337ceb23c798352d2.jpg](../iclr_results/179_Open-YOLO 3D_ Towards Fast and Accurate Open-Vocabulary 3D Instance Segmentation/tables/532a354155020ce5d970de2219e764600828c8030753091337ceb23c798352d2.jpg)

![68420cff377d47caaf6e200b105d99ec14e4d3149466e4d6c4088141995861c0.jpg](../iclr_results/179_Open-YOLO 3D_ Towards Fast and Accurate Open-Vocabulary 3D Instance Segmentation/tables/68420cff377d47caaf6e200b105d99ec14e4d3149466e4d6c4088141995861c0.jpg)

![75b4eb85ce5aa8c3a8951eba2ea95ad590387d4579627eb816285a76bf911fd9.jpg](../iclr_results/179_Open-YOLO 3D_ Towards Fast and Accurate Open-Vocabulary 3D Instance Segmentation/tables/75b4eb85ce5aa8c3a8951eba2ea95ad590387d4579627eb816285a76bf911fd9.jpg)

![98f898889462682e31060ed89f6450b04c630eca5a7b2fcc491af38894155f05.jpg](../iclr_results/179_Open-YOLO 3D_ Towards Fast and Accurate Open-Vocabulary 3D Instance Segmentation/tables/98f898889462682e31060ed89f6450b04c630eca5a7b2fcc491af38894155f05.jpg)

![af35d08707794a59137844c96759bcfea3637c97f33aee76fd1e0e6bcbbddb05.jpg](../iclr_results/179_Open-YOLO 3D_ Towards Fast and Accurate Open-Vocabulary 3D Instance Segmentation/tables/af35d08707794a59137844c96759bcfea3637c97f33aee76fd1e0e6bcbbddb05.jpg)

## What should a neuron aim for? Designing local objective functions based on information theory


### Images

![37936f2b456c36c33256b135bbf6ab8193a981384903fa723c63c1a61e056677.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/images/37936f2b456c36c33256b135bbf6ab8193a981384903fa723c63c1a61e056677.jpg)

![670eb9820cdae0881dfba23d3ff84648cc256a62e19c0788e612dc2e2f532864.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/images/670eb9820cdae0881dfba23d3ff84648cc256a62e19c0788e612dc2e2f532864.jpg)

![71c0cd8cbc7494416425739984af1b81aa7906af475205b99069583e0d088154.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/images/71c0cd8cbc7494416425739984af1b81aa7906af475205b99069583e0d088154.jpg)

![8022d88722306b62da4f8c902c85d3a548ba4ed92ee715ffc03abf877c30ec56.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/images/8022d88722306b62da4f8c902c85d3a548ba4ed92ee715ffc03abf877c30ec56.jpg)

![a2274fddc204d7aaad8af67e9298d58014af70a5e8bf17a525ab1bc6138589ac.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/images/a2274fddc204d7aaad8af67e9298d58014af70a5e8bf17a525ab1bc6138589ac.jpg)

![b33f39ba253eb1c20c5956d9d8b55eea6af9d6fbcde1cad4c4c08467aada6c4d.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/images/b33f39ba253eb1c20c5956d9d8b55eea6af9d6fbcde1cad4c4c08467aada6c4d.jpg)

![c4a3fb3d583e9e03a449daedbb01aee5bc237c8caaa9d6fb57c33a13c443000d.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/images/c4a3fb3d583e9e03a449daedbb01aee5bc237c8caaa9d6fb57c33a13c443000d.jpg)

![c58f81d601c76252b17776fb2427ffeb038042cf223548402f02db15c892a674.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/images/c58f81d601c76252b17776fb2427ffeb038042cf223548402f02db15c892a674.jpg)

![ce4079bfc07d3dcd44255a6d69999d94d270c7a6b8a9dc6e113e2b2cd334a9a4.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/images/ce4079bfc07d3dcd44255a6d69999d94d270c7a6b8a9dc6e113e2b2cd334a9a4.jpg)

![dd6ec113dd3d125935398a7611cc044c935f560c9fc04a1b14bef918ee05e033.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/images/dd6ec113dd3d125935398a7611cc044c935f560c9fc04a1b14bef918ee05e033.jpg)

![fd93276eb6c31483b5a9a53bbd0022f23296dcb1787004e9d04a78b669ab18db.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/images/fd93276eb6c31483b5a9a53bbd0022f23296dcb1787004e9d04a78b669ab18db.jpg)

### Tables

![4da5f56c98b446181290c1c9874dfe942c03bec561eee0bed44a6fc025de4003.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/tables/4da5f56c98b446181290c1c9874dfe942c03bec561eee0bed44a6fc025de4003.jpg)

![5ce7207d73f87596fe3b5e0150a53e63e46bd5d02f01960beb540081ece4a7fd.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/tables/5ce7207d73f87596fe3b5e0150a53e63e46bd5d02f01960beb540081ece4a7fd.jpg)

![d22f9573efc96a455dd828b20ad160c9b2b18b55b0161ef24e04d99ab67a8786.jpg](../iclr_results/180_What should a neuron aim for_ Designing local objective functions based on information theory/tables/d22f9573efc96a455dd828b20ad160c9b2b18b55b0161ef24e04d99ab67a8786.jpg)

## Backtracking Improves Generation Safety


### Images

![2445f557107f12823e641b891588d02a0ef61474f938c7eb97d1f83c4fc9a25a.jpg](../iclr_results/181_Backtracking Improves Generation Safety/images/2445f557107f12823e641b891588d02a0ef61474f938c7eb97d1f83c4fc9a25a.jpg)

![45d1c6cb4a110c6db536ef58f257b522f29197b6ff3bf5fd11ebcef14028bc6a.jpg](../iclr_results/181_Backtracking Improves Generation Safety/images/45d1c6cb4a110c6db536ef58f257b522f29197b6ff3bf5fd11ebcef14028bc6a.jpg)

![913878472d751a23ed59ca831ca4224d743de2ce7c4e1251647873612c7465d6.jpg](../iclr_results/181_Backtracking Improves Generation Safety/images/913878472d751a23ed59ca831ca4224d743de2ce7c4e1251647873612c7465d6.jpg)

![d19ce3b915946fdd98cf886709829b64c948b1f5a2a452a58ee7b54d9cf5ad8e.jpg](../iclr_results/181_Backtracking Improves Generation Safety/images/d19ce3b915946fdd98cf886709829b64c948b1f5a2a452a58ee7b54d9cf5ad8e.jpg)

![fa165a4c03a41a0432f5a3e82f26f265104b8a003b3413a739eed4a0ed971f17.jpg](../iclr_results/181_Backtracking Improves Generation Safety/images/fa165a4c03a41a0432f5a3e82f26f265104b8a003b3413a739eed4a0ed971f17.jpg)

### Tables

![2590216ed4e8e0c039c1e3784b53f0b593b855a629368e30101762c9b9d2b372.jpg](../iclr_results/181_Backtracking Improves Generation Safety/tables/2590216ed4e8e0c039c1e3784b53f0b593b855a629368e30101762c9b9d2b372.jpg)

![3e967c4725075df13ba442be6d8b8301f7dee0ad4425361dfd2ce9f30eef56f9.jpg](../iclr_results/181_Backtracking Improves Generation Safety/tables/3e967c4725075df13ba442be6d8b8301f7dee0ad4425361dfd2ce9f30eef56f9.jpg)

![4d801a4a9fd11d4eee722125d827caf2a6de2739eb8b4857cdb2f3b71a2c7665.jpg](../iclr_results/181_Backtracking Improves Generation Safety/tables/4d801a4a9fd11d4eee722125d827caf2a6de2739eb8b4857cdb2f3b71a2c7665.jpg)

![5dc0f1b4448be98f343e337e315fee8939402e0891f5bdae9ad431d0310fda41.jpg](../iclr_results/181_Backtracking Improves Generation Safety/tables/5dc0f1b4448be98f343e337e315fee8939402e0891f5bdae9ad431d0310fda41.jpg)

![7c54bbb3fa5e7d2f89e7f705a319b213fafe4b0f412e21c9cbab3ac9316532b1.jpg](../iclr_results/181_Backtracking Improves Generation Safety/tables/7c54bbb3fa5e7d2f89e7f705a319b213fafe4b0f412e21c9cbab3ac9316532b1.jpg)

![8b2c0b2d7e0ee9ce11a0a72cff028516de45633027b2f267e3477d573ca875a8.jpg](../iclr_results/181_Backtracking Improves Generation Safety/tables/8b2c0b2d7e0ee9ce11a0a72cff028516de45633027b2f267e3477d573ca875a8.jpg)

![b3ec8b66e52418dce7e4bf1924fcdcb77d3d4a974066cfb0f1c49695f6c85050.jpg](../iclr_results/181_Backtracking Improves Generation Safety/tables/b3ec8b66e52418dce7e4bf1924fcdcb77d3d4a974066cfb0f1c49695f6c85050.jpg)

![f960b90324618787bbc23235e722446a482cac586ddcd6b8c831d113dca99b2b.jpg](../iclr_results/181_Backtracking Improves Generation Safety/tables/f960b90324618787bbc23235e722446a482cac586ddcd6b8c831d113dca99b2b.jpg)

## Spread Preference Annotation: Direct Preference Judgment for Efficient LLM Alignment


### Images

![5caebd593c76d75e380ed8fb2d31c3f8c1ce1a51a91db8362fe8b4ec7b3a67e4.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/images/5caebd593c76d75e380ed8fb2d31c3f8c1ce1a51a91db8362fe8b4ec7b3a67e4.jpg)

![672541a7f6652e32d1a05f4ccfb2c900ee95ef9259cf06a3cec255343dc46768.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/images/672541a7f6652e32d1a05f4ccfb2c900ee95ef9259cf06a3cec255343dc46768.jpg)

![749583c63d948046e4fa788bd22963fbd80ba97bf111069e45ec03f427ad524f.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/images/749583c63d948046e4fa788bd22963fbd80ba97bf111069e45ec03f427ad524f.jpg)

![bafe650fff391cc97dfc0c3d7cf602aeb7c6d6beffadebaee481c892dc47fe41.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/images/bafe650fff391cc97dfc0c3d7cf602aeb7c6d6beffadebaee481c892dc47fe41.jpg)

![e6016821480e999ef2060946795f70b61af52d438129997ec8f33e7355d64884.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/images/e6016821480e999ef2060946795f70b61af52d438129997ec8f33e7355d64884.jpg)

### Tables

![16ac9d79c3714cfb87ab0cb82e06672ba057b35b93dfb0a092a7c47a8ae44b0b.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/16ac9d79c3714cfb87ab0cb82e06672ba057b35b93dfb0a092a7c47a8ae44b0b.jpg)

![1e17367fc1255cd75abfa94c41aa678aacae19948e07be0f212a216e88e2ce1f.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/1e17367fc1255cd75abfa94c41aa678aacae19948e07be0f212a216e88e2ce1f.jpg)

![2ff2161276b2affd587e322095b74910bb3787d92673fd4f2ae66ff3ea3c317f.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/2ff2161276b2affd587e322095b74910bb3787d92673fd4f2ae66ff3ea3c317f.jpg)

![45524f7a5598cc51c2920560f43108640d2dc97759081a37b793c18ee218e86b.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/45524f7a5598cc51c2920560f43108640d2dc97759081a37b793c18ee218e86b.jpg)

![98015511aab3af65502bb04e0c209e86b60641741b61c5e78f4825481c399139.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/98015511aab3af65502bb04e0c209e86b60641741b61c5e78f4825481c399139.jpg)

![9b0e9eac22512cfb41365edc21a00a2e26e0cb3b304ff01581c68285913eb924.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/9b0e9eac22512cfb41365edc21a00a2e26e0cb3b304ff01581c68285913eb924.jpg)

![a828cafc6d4850d2d7909e6efdb810db72bd249fd664d161f9abe129d6533c18.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/a828cafc6d4850d2d7909e6efdb810db72bd249fd664d161f9abe129d6533c18.jpg)

![aa1559571563c2f5a33433459d2c80f955c3cd99d20524436d3e7d49d307be20.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/aa1559571563c2f5a33433459d2c80f955c3cd99d20524436d3e7d49d307be20.jpg)

![c8f0419b3c232d8189a651c41df66a0fa24b83684dd52ee246870fb39e94f810.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/c8f0419b3c232d8189a651c41df66a0fa24b83684dd52ee246870fb39e94f810.jpg)

![e8277a98207fd30bb5fd7531896071e4d37494a9026392f6328ff34999690403.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/e8277a98207fd30bb5fd7531896071e4d37494a9026392f6328ff34999690403.jpg)

![f772416d808f6d65038352f1e6fb7d2cf71ab3582d32ba63aa69dcbf2a316fb0.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/f772416d808f6d65038352f1e6fb7d2cf71ab3582d32ba63aa69dcbf2a316fb0.jpg)

![fe346b0cce1d86c4a326f5bce31cfe82dad003119bc59d5423e8582ff01662c4.jpg](../iclr_results/182_Spread Preference Annotation_ Direct Preference Judgment for Efficient LLM Alignment/tables/fe346b0cce1d86c4a326f5bce31cfe82dad003119bc59d5423e8582ff01662c4.jpg)

## Global Convergence in Neural ODEs: Impact of Activation Functions


### Images

![1183715f61d1ef0c0a8ed7d0246a26ba8cdfdd2bfe7fd91a24f0123ee63f9e9e.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/1183715f61d1ef0c0a8ed7d0246a26ba8cdfdd2bfe7fd91a24f0123ee63f9e9e.jpg)

![145880ead73f6fff0f635963e32322d65865483078b5c340591f2ae0d7baabc4.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/145880ead73f6fff0f635963e32322d65865483078b5c340591f2ae0d7baabc4.jpg)

![1801d51ef4be3e511fb6d3b75fdadf18f5bc3193a0e347aad7c3ccb0dbf6921f.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/1801d51ef4be3e511fb6d3b75fdadf18f5bc3193a0e347aad7c3ccb0dbf6921f.jpg)

![2bf5e5ed7f1e5c5d02110aa2a5c6eb5fae6309820889515a7e34459823b6d14b.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/2bf5e5ed7f1e5c5d02110aa2a5c6eb5fae6309820889515a7e34459823b6d14b.jpg)

![33aec0a08b858478d39a7fe49a2d4e0a003bd8248aec0c1c8ae5a95142b3b930.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/33aec0a08b858478d39a7fe49a2d4e0a003bd8248aec0c1c8ae5a95142b3b930.jpg)

![437a0c999aa5601205626618f0610dfa7ba164c96bc994b97d0d56e321d3be55.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/437a0c999aa5601205626618f0610dfa7ba164c96bc994b97d0d56e321d3be55.jpg)

![55d3ebcf1f979b5bc8da5b8f0f61832d73270979d6cd0b646ca4b03313554e42.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/55d3ebcf1f979b5bc8da5b8f0f61832d73270979d6cd0b646ca4b03313554e42.jpg)

![64b08d72f8ec3ec13a63af1e715d2b173aa0a9ea0bc414ca8af7cf586882236f.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/64b08d72f8ec3ec13a63af1e715d2b173aa0a9ea0bc414ca8af7cf586882236f.jpg)

![ab61856a0d0793472ab157964724172b76b74c0bcb2170038ee0ae634ef89b15.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/ab61856a0d0793472ab157964724172b76b74c0bcb2170038ee0ae634ef89b15.jpg)

![b33dd2561267ec251ca44008faf582ad8874e3707b590f71263a7d767bf82a23.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/b33dd2561267ec251ca44008faf582ad8874e3707b590f71263a7d767bf82a23.jpg)

![bb61ddb1cb3564215431d3e0bc0ca674b12c6715ea4e9d48897541826bc174cc.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/bb61ddb1cb3564215431d3e0bc0ca674b12c6715ea4e9d48897541826bc174cc.jpg)

![c19f5247aaa17c0f7b667917cb40fbd0b40fbac5ad85c2e082977a2fb8fa4817.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/c19f5247aaa17c0f7b667917cb40fbd0b40fbac5ad85c2e082977a2fb8fa4817.jpg)

![eaf8e03024bc72d0d25e01eb1a953af0e010161f821052d24f2abfe85d464938.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/eaf8e03024bc72d0d25e01eb1a953af0e010161f821052d24f2abfe85d464938.jpg)

![f0e573dce18b94d5cec97da5db50d3a4b5c7764b8095a39e573ba5d3c3e68550.jpg](../iclr_results/183_Global Convergence in Neural ODEs_ Impact of Activation Functions/images/f0e573dce18b94d5cec97da5db50d3a4b5c7764b8095a39e573ba5d3c3e68550.jpg)

## Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces


### Images

![03210abf672c06ee754e1d8031b669c0a2eb450c8f606804a970623006721d0e.jpg](../iclr_results/184_Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces/images/03210abf672c06ee754e1d8031b669c0a2eb450c8f606804a970623006721d0e.jpg)

![35d74592c55d9c7fed460cd452a355e7073f3fe28ebbfaefff2178afe0ba7874.jpg](../iclr_results/184_Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces/images/35d74592c55d9c7fed460cd452a355e7073f3fe28ebbfaefff2178afe0ba7874.jpg)

![3f2dede7a2bf0763a4161a227067fc5c61d89335452ba2ba98f9c046e49df53a.jpg](../iclr_results/184_Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces/images/3f2dede7a2bf0763a4161a227067fc5c61d89335452ba2ba98f9c046e49df53a.jpg)

![456cbb7c68e2135c596dac55c96d4f56aea3621f24b39bce95560b1941bba0d8.jpg](../iclr_results/184_Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces/images/456cbb7c68e2135c596dac55c96d4f56aea3621f24b39bce95560b1941bba0d8.jpg)

![4c5f7363afabfa144f107901ff541714aabd53b7dd201d7b3721c945617d4cbd.jpg](../iclr_results/184_Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces/images/4c5f7363afabfa144f107901ff541714aabd53b7dd201d7b3721c945617d4cbd.jpg)

![598d093867c29fba886c32131aa3a380b063cf85e30dd21700b178cf3f5f965d.jpg](../iclr_results/184_Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces/images/598d093867c29fba886c32131aa3a380b063cf85e30dd21700b178cf3f5f965d.jpg)

![988dfd4d23bd4486a59d293350b29236b89c80bd7bf10b3960aaea6965adb9cf.jpg](../iclr_results/184_Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces/images/988dfd4d23bd4486a59d293350b29236b89c80bd7bf10b3960aaea6965adb9cf.jpg)

![98e8965a308e9c886b3d9bf524444f78ab519c7c0da60a60b0951ac4de22fff9.jpg](../iclr_results/184_Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces/images/98e8965a308e9c886b3d9bf524444f78ab519c7c0da60a60b0951ac4de22fff9.jpg)

![ce826d89c0987a02d6515bd2f29ada224974903d242704eb31efca71698b8825.jpg](../iclr_results/184_Geometry of Neural Reinforcement Learning in Continuous State and Action Spaces/images/ce826d89c0987a02d6515bd2f29ada224974903d242704eb31efca71698b8825.jpg)

## The Hidden Cost of Waiting for Accurate Predictions


### Images

![780743247ec9dffc45b27f485624d3e8e95b0e4a37e5193c39a1c00cdde3deef.jpg](../iclr_results/185_The Hidden Cost of Waiting for Accurate Predictions/images/780743247ec9dffc45b27f485624d3e8e95b0e4a37e5193c39a1c00cdde3deef.jpg)

![844e5bae84376681ca17d991659dccf7bdfcfc21a4c462cee5f6e7f1f039ed98.jpg](../iclr_results/185_The Hidden Cost of Waiting for Accurate Predictions/images/844e5bae84376681ca17d991659dccf7bdfcfc21a4c462cee5f6e7f1f039ed98.jpg)

![d5dd6018700a14406c3b38a59cf9a95ea9738f6678a9b317a2f65da48e7ddf49.jpg](../iclr_results/185_The Hidden Cost of Waiting for Accurate Predictions/images/d5dd6018700a14406c3b38a59cf9a95ea9738f6678a9b317a2f65da48e7ddf49.jpg)

### Tables

![943219d113fc99ea4c11d8cd17976a6b08424559bbd78f2626b5da52f0dea553.jpg](../iclr_results/185_The Hidden Cost of Waiting for Accurate Predictions/tables/943219d113fc99ea4c11d8cd17976a6b08424559bbd78f2626b5da52f0dea553.jpg)

## DeepLTL: Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL


### Images

![016ff054581855709233076159fa584182428b004b4838cb8dac014642e2a511.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/016ff054581855709233076159fa584182428b004b4838cb8dac014642e2a511.jpg)

![2337acf869f2a789cbf645898ac4c4f352823289d36e5ad1cc440d2d289f5204.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/2337acf869f2a789cbf645898ac4c4f352823289d36e5ad1cc440d2d289f5204.jpg)

![2cd0f138393fab0c9e369fee6e6ebfe95895b714f829a0fb99e72e721b78669a.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/2cd0f138393fab0c9e369fee6e6ebfe95895b714f829a0fb99e72e721b78669a.jpg)

![48e9d920cfdff22e7cbb1e18f0504aa40c178101fe4d2236de79fac49d164c74.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/48e9d920cfdff22e7cbb1e18f0504aa40c178101fe4d2236de79fac49d164c74.jpg)

![5c3b24a486555f5b6ee48ef2bb7114231f9e276e9f9bdb60c5eceeade62e77b8.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/5c3b24a486555f5b6ee48ef2bb7114231f9e276e9f9bdb60c5eceeade62e77b8.jpg)

![6f1b6dd5da06d467538be618c6e98f40df467c8cc37afac8e278aacb785fa54b.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/6f1b6dd5da06d467538be618c6e98f40df467c8cc37afac8e278aacb785fa54b.jpg)

![a6ab5bf98f82c722cb24ee2620c9e6d04fd578d65c203d2933b0fd854af69550.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/a6ab5bf98f82c722cb24ee2620c9e6d04fd578d65c203d2933b0fd854af69550.jpg)

![b6d50978ec1e013ef78f22250a01033b50b70bc79de529535472397153313233.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/b6d50978ec1e013ef78f22250a01033b50b70bc79de529535472397153313233.jpg)

![c1a62d4a86e9b76531237df61660ccbe5fc6a5b503e019b2139982e3fec2d0b1.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/c1a62d4a86e9b76531237df61660ccbe5fc6a5b503e019b2139982e3fec2d0b1.jpg)

![c5b2fd1215e733a40c2dcc4f5814a558349f6703d189e6d80a4a679111b66e6d.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/c5b2fd1215e733a40c2dcc4f5814a558349f6703d189e6d80a4a679111b66e6d.jpg)

![cb2562d14ce12540976cc7f66c6a2ebdbcbc0459d6b32a85c3d9fabad70290eb.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/cb2562d14ce12540976cc7f66c6a2ebdbcbc0459d6b32a85c3d9fabad70290eb.jpg)

![e3cc494d65055ac29a939fc2363e9b4d094cc58a2a9cc84649f8a572bb1366c1.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/images/e3cc494d65055ac29a939fc2363e9b4d094cc58a2a9cc84649f8a572bb1366c1.jpg)

### Tables

![53e067feafbc6091f79d740caa371ebc76fb4cde29eca773e2918a7454a9ceb1.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/tables/53e067feafbc6091f79d740caa371ebc76fb4cde29eca773e2918a7454a9ceb1.jpg)

![5d367277f549897554eebe85d6e47348d24ff8823865b266a0624dc941610f09.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/tables/5d367277f549897554eebe85d6e47348d24ff8823865b266a0624dc941610f09.jpg)

![609e5515a8bf41cea49678f9e61463c9ff8f8f1648629b82f01ed2ff2b1da3f4.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/tables/609e5515a8bf41cea49678f9e61463c9ff8f8f1648629b82f01ed2ff2b1da3f4.jpg)

![73641d4f0240c791b2712895d7569f445d013d6c7f54a3cd38395dce04f73558.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/tables/73641d4f0240c791b2712895d7569f445d013d6c7f54a3cd38395dce04f73558.jpg)

![8a0d5fb2247dc8ed97c63a02177e1709087627c45c711ba88a6c29608098f5eb.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/tables/8a0d5fb2247dc8ed97c63a02177e1709087627c45c711ba88a6c29608098f5eb.jpg)

![9afef1d23f06eeb9b61635bae0cb3228cf59500ccbdba1090d891cc1c493665f.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/tables/9afef1d23f06eeb9b61635bae0cb3228cf59500ccbdba1090d891cc1c493665f.jpg)

![e90b3a054a948608d7a2c4640fb4500aa44c071603d11448a0a9a89d03a5baa4.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/tables/e90b3a054a948608d7a2c4640fb4500aa44c071603d11448a0a9a89d03a5baa4.jpg)

![ed5ad9cbdcd47bd069c2e37546618795390ff3b1ec1079b73f7adedcefbcbc65.jpg](../iclr_results/186_DeepLTL_ Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL/tables/ed5ad9cbdcd47bd069c2e37546618795390ff3b1ec1079b73f7adedcefbcbc65.jpg)

## The Complexity of Two-Team Polymatrix Games with Independent Adversaries

### Images

![4e5a38f27eb21622427a720a76e7a0d41c33931fd136356276a8dcc1980b07ef.jpg](../iclr_results/187_The Complexity of Two-Team Polymatrix Games with Independent Adversaries/images/4e5a38f27eb21622427a720a76e7a0d41c33931fd136356276a8dcc1980b07ef.jpg)

![824a8e74da5858b8c0843917e6637a658af0fefe5ed636c799042c4ada79232f.jpg](../iclr_results/187_The Complexity of Two-Team Polymatrix Games with Independent Adversaries/images/824a8e74da5858b8c0843917e6637a658af0fefe5ed636c799042c4ada79232f.jpg)
