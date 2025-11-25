# ICLR 2025 Main Conference Papers

**Summary:** 36 papers with extracted content:
- 📊 Total images: 44031
- 📋 Total tables: 33468
- 📄 Total files: 77499

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 12 of 100

## 目录 (Table of Contents)

1. [SRSA: Skill Retrieval and Adaptation for Robotic Assembly Tasks](#SRSA-Skill-Retrieval-and-Adaptation-for-Robotic-Assembly-Tasks)
2. [Towards General-Purpose Model-Free Reinforcement Learning](#Towards-General-Purpose-Model-Free-Reinforcement-Learning)
3. [The Computational Complexity of Circuit Discovery for Inner Interpretability](#The-Computational-Complexity-of-Circuit-Discovery-for-Inner-Interpretability)
4. [Learning from End User Data with Shuffled Differential Privacy over Kernel Densities](#Learning-from-End-User-Data-with-Shuffled-Differential-Privacy-over-Kernel-Densities)
5. [VisualPredicator: Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning](#VisualPredicator-Learning-Abstract-World-Models-with-Neuro-Symbolic-Predicates-for-Robot-Planning)
6. [Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement](#Spectral-Compressive-Imaging-via-Unmixing-driven-Subspace-Diffusion-Refinement)
7. [Topograph: An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation](#Topograph-An-Efficient-Graph-Based-Framework-for-Strictly-Topology-Preserving-Image-Segmentation)
8. [LoCoDL: Communication-Efficient Distributed Learning with Local Training and Compression](#LoCoDL-Communication-Efficient-Distributed-Learning-with-Local-Training-and-Compression)
9. [Let SSMs be ConvNets: State-space Modeling with Optimal Tensor Contractions](#Let-SSMs-be-ConvNets-State-space-Modeling-with-Optimal-Tensor-Contractions)
10. [SoftCVI: Contrastive variational inference with self-generated soft labels](#SoftCVI-Contrastive-variational-inference-with-self-generated-soft-labels)
11. [Adam Exploits $\ell_\infty$-geometry of Loss Landscape via Coordinate-wise Adaptivity](#Adam-Exploits-ell_infty-geometry-of-Loss-Landscape-via-Coordinate-wise-Adaptivity)
12. [DailyDilemmas: Revealing Value Preferences of LLMs with Quandaries of Daily Life](#DailyDilemmas-Revealing-Value-Preferences-of-LLMs-with-Quandaries-of-Daily-Life)
13. [u-$\mu$P: The Unit-Scaled Maximal Update Parametrization](#u-muP-The-Unit-Scaled-Maximal-Update-Parametrization)
14. [Instance-dependent Early Stopping](#Instance-dependent-Early-Stopping)
15. [Learning vector fields of differential equations on manifolds with geometrically constrained operator-valued kernels](#Learning-vector-fields-of-differential-equations-on-manifolds-with-geometrically-constrained-operator-valued-kernels)
16. [Programming Refusal with Conditional Activation Steering](#Programming-Refusal-with-Conditional-Activation-Steering)
17. [Samba: Synchronized Set-of-Sequences Modeling for Multiple Object Tracking](#Samba-Synchronized-Set-of-Sequences-Modeling-for-Multiple-Object-Tracking)
18. [A Second-Order Perspective on Model Compositionality and Incremental Learning](#A-Second-Order-Perspective-on-Model-Compositionality-and-Incremental-Learning)
19. [Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes](#Signature-Kernel-Conditional-Independence-Tests-in-Causal-Discovery-for-Stochastic-Processes)
20. [Formation of Representations in Neural Networks](#Formation-of-Representations-in-Neural-Networks)
21. [RAG-SR: Retrieval-Augmented Generation for Neural Symbolic Regression](#RAG-SR-Retrieval-Augmented-Generation-for-Neural-Symbolic-Regression)
22. [Towards Automated Knowledge Integration From Human-Interpretable Representations](#Towards-Automated-Knowledge-Integration-From-Human-Interpretable-Representations)
23. [TOP-ERL: Transformer-based Off-Policy Episodic Reinforcement Learning](#TOP-ERL-Transformer-based-Off-Policy-Episodic-Reinforcement-Learning)
24. [Multi-Draft Speculative Sampling: Canonical Decomposition and Theoretical Limits](#Multi-Draft-Speculative-Sampling-Canonical-Decomposition-and-Theoretical-Limits)
25. [Anti-Exposure Bias in Diffusion Models](#Anti-Exposure-Bias-in-Diffusion-Models)
26. [Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs](#Continuous-Exposure-Learning-for-Low-light-Image-Enhancement-using-Neural-ODEs)
27. [WildBench: Benchmarking LLMs with Challenging Tasks from Real Users in the Wild](#WildBench-Benchmarking-LLMs-with-Challenging-Tasks-from-Real-Users-in-the-Wild)
28. [D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement](#D-FINE-Redefine-Regression-Task-of-DETRs-as-Fine-grained-Distribution-Refinement)
29. [DynamicCity: Large-Scale 4D Occupancy Generation from Dynamic Scenes](#DynamicCity-Large-Scale-4D-Occupancy-Generation-from-Dynamic-Scenes)
30. [CubeDiff: Repurposing Diffusion-Based Image Models for Panorama Generation](#CubeDiff-Repurposing-Diffusion-Based-Image-Models-for-Panorama-Generation)
31. [A Periodic Bayesian Flow for Material Generation](#A-Periodic-Bayesian-Flow-for-Material-Generation)
32. [Generalized Principal-Agent Problem with a Learning Agent](#Generalized-Principal-Agent-Problem-with-a-Learning-Agent)
33. [Efficient and Accurate Explanation Estimation with Distribution Compression](#Efficient-and-Accurate-Explanation-Estimation-with-Distribution-Compression)
34. [Nonlinear multiregion neural dynamics with parametric impulse response communication channels](#Nonlinear-multiregion-neural-dynamics-with-parametric-impulse-response-communication-channels)
35. [POTEC: Off-Policy Contextual Bandits for Large Action Spaces via Policy Decomposition](#POTEC-Off-Policy-Contextual-Bandits-for-Large-Action-Spaces-via-Policy-Decomposition)
36. [LoRA3D: Low-Rank Self-Calibration of 3D Geometric Foundation models](#LoRA3D-Low-Rank-Self-Calibration-of-3D-Geometric-Foundation-models)

---


## SRSA: Skill Retrieval and Adaptation for Robotic Assembly Tasks

### Images

![0a386caf3f65e4fca26c07a6b7a73ac463896bb264fbe269e224fd1c8cde8a7d.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/images/0a386caf3f65e4fca26c07a6b7a73ac463896bb264fbe269e224fd1c8cde8a7d.jpg)

![14670497d98a19d963ab35ff6d5aa9a1b57b489798cbd915b8ac00d4b0272222.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/images/14670497d98a19d963ab35ff6d5aa9a1b57b489798cbd915b8ac00d4b0272222.jpg)

![36f751a84d6e9c0dbb04824da2a76a65b4c9c826d7d63b0079e14dc303954467.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/images/36f751a84d6e9c0dbb04824da2a76a65b4c9c826d7d63b0079e14dc303954467.jpg)

![3a2225b1fa2a3c9de7270d66048d9e0f85384a48ba91d2d5155068376e1293ef.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/images/3a2225b1fa2a3c9de7270d66048d9e0f85384a48ba91d2d5155068376e1293ef.jpg)

![50f5d4084df155c92651add795fdc97dc60c38a7159b6fb35d7deaaf2c95af53.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/images/50f5d4084df155c92651add795fdc97dc60c38a7159b6fb35d7deaaf2c95af53.jpg)

![5170cda51326bba999337a97766795ad34b5fbe28bbbf54b517bbe7ec4cb62b5.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/images/5170cda51326bba999337a97766795ad34b5fbe28bbbf54b517bbe7ec4cb62b5.jpg)

![6c1f0652e3cfa9d2f655443dc92638b0514bc0f853ba9c9d20b686af7b474bfa.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/images/6c1f0652e3cfa9d2f655443dc92638b0514bc0f853ba9c9d20b686af7b474bfa.jpg)

![6dface3d5aedba6fe4f8577b253d9f4a1456ce2a29654ed4dd7d7296d926dca0.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/images/6dface3d5aedba6fe4f8577b253d9f4a1456ce2a29654ed4dd7d7296d926dca0.jpg)

![742e65e1dab2d3612ddb6bff30baa986d16ddecbaee4399796a93585bc2d7317.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/images/742e65e1dab2d3612ddb6bff30baa986d16ddecbaee4399796a93585bc2d7317.jpg)

![790fcd9800dcfbf05b16e3c71ed651e9cdab9d5b7abff9cdd8467907219a4e27.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/images/790fcd9800dcfbf05b16e3c71ed651e9cdab9d5b7abff9cdd8467907219a4e27.jpg)

![7bc5146c8401811d945ac5f9878fa95d97698e03946953c8b13b942dacc39ebd.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/images/7bc5146c8401811d945ac5f9878fa95d97698e03946953c8b13b942dacc39ebd.jpg)

![8b71c357f23ba0948d9ea1ac471726c9e13aa4459f7b988bd7aee07a2cf2b562.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/images/8b71c357f23ba0948d9ea1ac471726c9e13aa4459f7b988bd7aee07a2cf2b562.jpg)

![acbd3eafcee05da36beca7537ce501f70ffe4a2642f05ea9487d9b1fa3d9bf5f.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/images/acbd3eafcee05da36beca7537ce501f70ffe4a2642f05ea9487d9b1fa3d9bf5f.jpg)

![d5889c5fbfffa28aba561b177d3ea15d30083f0aeb65c1ec31164edc04059f8a.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/images/d5889c5fbfffa28aba561b177d3ea15d30083f0aeb65c1ec31164edc04059f8a.jpg)

![dfca38c91bfb89b9efa5eff98401d1a6ea8318b0dffed2ecb620013dec44acd1.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/images/dfca38c91bfb89b9efa5eff98401d1a6ea8318b0dffed2ecb620013dec44acd1.jpg)

![e98cc4d74f75b0d739cef08d359cd26e6e08cb40142e82711b5e684ab826e8dd.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/images/e98cc4d74f75b0d739cef08d359cd26e6e08cb40142e82711b5e684ab826e8dd.jpg)

![f472543e0b1272fa05158857bc0485669134c63b41db3acb8734990432ea4832.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/images/f472543e0b1272fa05158857bc0485669134c63b41db3acb8734990432ea4832.jpg)

### Tables

![525086f1ed8ce2053d49845c3145ef1eed60ff010edc4c10beb45775f1a1f5ff.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/tables/525086f1ed8ce2053d49845c3145ef1eed60ff010edc4c10beb45775f1a1f5ff.jpg)

![723aadaba15a1fb0df3cf4f7e5f8bcc2634ebfa291a0686f34ff2d5545d97c4e.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/tables/723aadaba15a1fb0df3cf4f7e5f8bcc2634ebfa291a0686f34ff2d5545d97c4e.jpg)

![7a802c5a7c4aeddacb9b0462a0deac366d6857954bd6cf1fdd25effabd0f77ae.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/tables/7a802c5a7c4aeddacb9b0462a0deac366d6857954bd6cf1fdd25effabd0f77ae.jpg)

![ae1d55c68186a63ce266d3aa51ce57b6abdf6c43023bed798c768465ca76d7a7.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/tables/ae1d55c68186a63ce266d3aa51ce57b6abdf6c43023bed798c768465ca76d7a7.jpg)

![c36f306a0f63a29c6aa71f2d05f79853f3eef9a6e3f41bd25ae0ac53a24c626b.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/tables/c36f306a0f63a29c6aa71f2d05f79853f3eef9a6e3f41bd25ae0ac53a24c626b.jpg)

![c8f6468dbdb3d32ec2294b95afaf7475b382c024f1956715090d2fa6b9d34506.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/tables/c8f6468dbdb3d32ec2294b95afaf7475b382c024f1956715090d2fa6b9d34506.jpg)

![d077be2960595c8eaf326012018763edda650dfdb6c38889a13a224d12adbc96.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/tables/d077be2960595c8eaf326012018763edda650dfdb6c38889a13a224d12adbc96.jpg)

![ea8129c07ad87c2c24aa85310a1c14e2d5068c44f10a5560615a5f7473f641e4.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/tables/ea8129c07ad87c2c24aa85310a1c14e2d5068c44f10a5560615a5f7473f641e4.jpg)

![ecec8f51a509519f27046e3818af2898de55e1ae8829bee55beb6e1385b489e7.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/tables/ecec8f51a509519f27046e3818af2898de55e1ae8829bee55beb6e1385b489e7.jpg)

![fb841a5b7ec2808badce75de7fe052dda47347b3306d2e3802a44271735a0066.jpg](../iclr_results/425_FairMT-Bench_ Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs/tables/fb841a5b7ec2808badce75de7fe052dda47347b3306d2e3802a44271735a0066.jpg)

## SRSA: Skill Retrieval and Adaptation for Robotic Assembly Tasks


### Images

![13940ffe038c653d056520479393fae6720dd20e6e23b73ae1fd8319f121c34e.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/images/13940ffe038c653d056520479393fae6720dd20e6e23b73ae1fd8319f121c34e.jpg)

![21c17aaf84f2efce75b458c7b77a0a443db8a7b6a6bbc51624c45f4d71f090da.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/images/21c17aaf84f2efce75b458c7b77a0a443db8a7b6a6bbc51624c45f4d71f090da.jpg)

![2cb6aac5612fa9ec54c5ce68f871fc3248fa0a91226ce86c416708f9e05f1935.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/images/2cb6aac5612fa9ec54c5ce68f871fc3248fa0a91226ce86c416708f9e05f1935.jpg)

![34aabbb55c4fb0fee276735b934cc17b797c5fada561b4ed252498e49a029ed7.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/images/34aabbb55c4fb0fee276735b934cc17b797c5fada561b4ed252498e49a029ed7.jpg)

![3729f63686ed14e76c5da19b2615bd39940276d393829951ff8456581769658d.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/images/3729f63686ed14e76c5da19b2615bd39940276d393829951ff8456581769658d.jpg)

![4559b96927aa7cc1001718a8090c259dd48232cd7eda83bbe6d8028b463924e0.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/images/4559b96927aa7cc1001718a8090c259dd48232cd7eda83bbe6d8028b463924e0.jpg)

![619c0001ce7b2ba9fbd799314f7345f4720d240814f304703636f9dfa9f767d4.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/images/619c0001ce7b2ba9fbd799314f7345f4720d240814f304703636f9dfa9f767d4.jpg)

![669342af3e2a545aacbdabc093db8559f9e8b7cf14d66446879eb83ca1242475.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/images/669342af3e2a545aacbdabc093db8559f9e8b7cf14d66446879eb83ca1242475.jpg)

![7d4ccdbd6713fb606da6bed4cb3dfdb716251c416e8072a5d7c667d0a438f814.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/images/7d4ccdbd6713fb606da6bed4cb3dfdb716251c416e8072a5d7c667d0a438f814.jpg)

![99377635c4331691d64ae40c6fec4379fee64e10612bb7c690f8f53a833d74f8.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/images/99377635c4331691d64ae40c6fec4379fee64e10612bb7c690f8f53a833d74f8.jpg)

![a97d8abceb58f3f3b289e6f97dbeee37f41b7be4e2b841c94f5204b3e553e3b4.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/images/a97d8abceb58f3f3b289e6f97dbeee37f41b7be4e2b841c94f5204b3e553e3b4.jpg)

![b3bd3d98091006b8cba4bb7f7329ee65394f9d294704737f96fb63b248dc66a9.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/images/b3bd3d98091006b8cba4bb7f7329ee65394f9d294704737f96fb63b248dc66a9.jpg)

![b7f9bc78056b1267248e2bf1673bf78c637aeedcf5c8791cec45cfad4ed4e9df.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/images/b7f9bc78056b1267248e2bf1673bf78c637aeedcf5c8791cec45cfad4ed4e9df.jpg)

![d58e4b2c2f4ab3f80633e4372cb6da997220ae5424c7af8792d452b7bc593845.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/images/d58e4b2c2f4ab3f80633e4372cb6da997220ae5424c7af8792d452b7bc593845.jpg)

![dfae1ede9d267bc52d7a3efbc4457ca0ddb29c84212d1cbd472aa69e738fbdc0.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/images/dfae1ede9d267bc52d7a3efbc4457ca0ddb29c84212d1cbd472aa69e738fbdc0.jpg)

![e0f4ece638b336f11097234ff14fcdb416c6008a5099c348a89b3994509201b2.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/images/e0f4ece638b336f11097234ff14fcdb416c6008a5099c348a89b3994509201b2.jpg)

![f5d9d0ab1c9155dba1b7f79da8d2ad453a2c98fab2dac9e0221c3232b51fc66b.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/images/f5d9d0ab1c9155dba1b7f79da8d2ad453a2c98fab2dac9e0221c3232b51fc66b.jpg)

![fbbd7283957589c99a6fa84ad01b4e0cb09452c26eca841ef29d93aa802fec92.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/images/fbbd7283957589c99a6fa84ad01b4e0cb09452c26eca841ef29d93aa802fec92.jpg)

### Tables

![1473bf21a0fd05addaf558320c7230bb7ee6db65aef949382ea5a1548f1b448a.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/tables/1473bf21a0fd05addaf558320c7230bb7ee6db65aef949382ea5a1548f1b448a.jpg)

![3904cf757f844253726fad1c68bc178c2b0f0adfb9333255e7734fecbe248504.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/tables/3904cf757f844253726fad1c68bc178c2b0f0adfb9333255e7734fecbe248504.jpg)

![44e8da9721f592e7ab4c07af94afd85c9c0f128115068617564d66490e0c71d6.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/tables/44e8da9721f592e7ab4c07af94afd85c9c0f128115068617564d66490e0c71d6.jpg)

![515f976ed0db162209d34310f4a7e4cf19477ddf8536a48d3546df86500ca6ee.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/tables/515f976ed0db162209d34310f4a7e4cf19477ddf8536a48d3546df86500ca6ee.jpg)

![55e89a048b573031a797f44e74814803b0b9607d4abbc86ee3768d3466b133aa.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/tables/55e89a048b573031a797f44e74814803b0b9607d4abbc86ee3768d3466b133aa.jpg)

![626102412bb93aa1d5fd655c4837d09411480d48e19294d5e96c2a74f70cad58.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/tables/626102412bb93aa1d5fd655c4837d09411480d48e19294d5e96c2a74f70cad58.jpg)

![741733e8785ee6c80f20404e2ec2d87d6224696d37917f1721236fe79e9a15e4.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/tables/741733e8785ee6c80f20404e2ec2d87d6224696d37917f1721236fe79e9a15e4.jpg)

![79a704d723303888f19949ca46b596e072e050f7f196a4c3bbb544ca794c8232.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/tables/79a704d723303888f19949ca46b596e072e050f7f196a4c3bbb544ca794c8232.jpg)

![7e44b34f661d70d134867cacc8c55d34cee6f52e4bb9a6e0ba58a0f5dda38cfb.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/tables/7e44b34f661d70d134867cacc8c55d34cee6f52e4bb9a6e0ba58a0f5dda38cfb.jpg)

![985842af7fb6257c0e05fcbf402f781bfe2a266b1102ef076776c5bb9e000d01.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/tables/985842af7fb6257c0e05fcbf402f781bfe2a266b1102ef076776c5bb9e000d01.jpg)

![b996f501c1a8b61d3cbe92d90b77d27c10d8a082c22ac54793769a54819ff366.jpg](../iclr_results/426_SRSA_ Skill Retrieval and Adaptation for Robotic Assembly Tasks/tables/b996f501c1a8b61d3cbe92d90b77d27c10d8a082c22ac54793769a54819ff366.jpg)

## Towards General-Purpose Model-Free Reinforcement Learning


### Images

![75bc120f4f753aa74f9ce5a1fbb5c6a0ee71384c89340c9645784552dab2a793.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/images/75bc120f4f753aa74f9ce5a1fbb5c6a0ee71384c89340c9645784552dab2a793.jpg)

![8a82bc562df8360e99d46980f6e6e18574f78e62387ac52eaab5ee65d6b7745c.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/images/8a82bc562df8360e99d46980f6e6e18574f78e62387ac52eaab5ee65d6b7745c.jpg)

![8e7e9e84880ab2e4a411e4db0ba61bd23b4de8ff9421ff93c7dcea65f1e3ae23.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/images/8e7e9e84880ab2e4a411e4db0ba61bd23b4de8ff9421ff93c7dcea65f1e3ae23.jpg)

![ac176fa00d7fa45b3dd5420a0b42459c978c1ba5aa2f4da16e403e0ca9d8aa11.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/images/ac176fa00d7fa45b3dd5420a0b42459c978c1ba5aa2f4da16e403e0ca9d8aa11.jpg)

![acf906a4c6d5576db408710494109e992e598edca447f33ccf26f0d2bf6a319a.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/images/acf906a4c6d5576db408710494109e992e598edca447f33ccf26f0d2bf6a319a.jpg)

![f708650868617cf019330ba3eccce1edb8aab63b67e7fb1f632ef49b555c2d61.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/images/f708650868617cf019330ba3eccce1edb8aab63b67e7fb1f632ef49b555c2d61.jpg)

### Tables

![051ea39f153b24751917899a3dcebb0c3a36addfc3092a9603d626c05a03e938.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/051ea39f153b24751917899a3dcebb0c3a36addfc3092a9603d626c05a03e938.jpg)

![1ee89a91efd6624a992b28390c6f075072703cafa4fba75ff15f0903c8c533ac.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/1ee89a91efd6624a992b28390c6f075072703cafa4fba75ff15f0903c8c533ac.jpg)

![2287f4cc40954240dad66e523d5dd2a435a4eea790f6563d430eff13cb0d30e2.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/2287f4cc40954240dad66e523d5dd2a435a4eea790f6563d430eff13cb0d30e2.jpg)

![2794b2f270c91215b7b44294f7b81cd4d3dd2aa107aa38531e1c48858e6ad79e.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/2794b2f270c91215b7b44294f7b81cd4d3dd2aa107aa38531e1c48858e6ad79e.jpg)

![2a8ee03213ea09daff3f917abc0a2eaf40dd1ded697e4d54e75200f2098654cf.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/2a8ee03213ea09daff3f917abc0a2eaf40dd1ded697e4d54e75200f2098654cf.jpg)

![311308c3ce07e2efbbf556ba3a15ca9156b52f97c9d6420735c6333eb9876049.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/311308c3ce07e2efbbf556ba3a15ca9156b52f97c9d6420735c6333eb9876049.jpg)

![42532f91718a2c8c67dbc6b5c51ef451470ae696078b65e439736a2d9490b08f.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/42532f91718a2c8c67dbc6b5c51ef451470ae696078b65e439736a2d9490b08f.jpg)

![42852b592d23eb28dc6a58f6c0983fc1e79a9f75b6fa6e2e7a3c41bdec0762b6.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/42852b592d23eb28dc6a58f6c0983fc1e79a9f75b6fa6e2e7a3c41bdec0762b6.jpg)

![63ee2110a3ec47ba092ac35b80ade78a15f73be9a4a40227b214af41dfde3a56.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/63ee2110a3ec47ba092ac35b80ade78a15f73be9a4a40227b214af41dfde3a56.jpg)

![74395b51632e96377437c8ab11cac4db47765347e41354340eb02ccc2cd7a5a5.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/74395b51632e96377437c8ab11cac4db47765347e41354340eb02ccc2cd7a5a5.jpg)

![74cbc1a718905d725b37d4dbc8905b6eb9a0cf03e9f946383b9aba40be25599b.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/74cbc1a718905d725b37d4dbc8905b6eb9a0cf03e9f946383b9aba40be25599b.jpg)

![8b015c36b749d731c4e3d336db4d8ff4c5a1956f9ea44768daa58740fd2c47fd.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/8b015c36b749d731c4e3d336db4d8ff4c5a1956f9ea44768daa58740fd2c47fd.jpg)

![8c8927cbe5d4a65284cee08881d7d54c9c4e931c34f9c4c3ce861f8dc730d08a.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/8c8927cbe5d4a65284cee08881d7d54c9c4e931c34f9c4c3ce861f8dc730d08a.jpg)

![9fcd80dda8dc8e25784070ff044d8a7889941ebfb0ff8ddb84dbd3a5cba09d35.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/9fcd80dda8dc8e25784070ff044d8a7889941ebfb0ff8ddb84dbd3a5cba09d35.jpg)

![a990b17b08f65950246a2e7e055e3fc5b76eea4564fbcecb7038126773414a21.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/a990b17b08f65950246a2e7e055e3fc5b76eea4564fbcecb7038126773414a21.jpg)

![bbe1ab38b9e720746c6adf8f3df56620f21f888e794a34dfe5b2e345111f31a4.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/bbe1ab38b9e720746c6adf8f3df56620f21f888e794a34dfe5b2e345111f31a4.jpg)

![bff08b5cceb27f9b71103fbcba889d252e5127f98e0a874623f891d0b1cfbd00.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/bff08b5cceb27f9b71103fbcba889d252e5127f98e0a874623f891d0b1cfbd00.jpg)

![d0935964f96256c3c9b1065a89d7d185a059df214ab958d7180c377eaaae3779.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/d0935964f96256c3c9b1065a89d7d185a059df214ab958d7180c377eaaae3779.jpg)

![d9d6c198f47c687e0f1c6c25d54caf07aacea295f757482821b1868ea3618e31.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/d9d6c198f47c687e0f1c6c25d54caf07aacea295f757482821b1868ea3618e31.jpg)

![dca38158b3a894169440ab1ec5620475fe549789ef582fc7065b3104053c93bc.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/dca38158b3a894169440ab1ec5620475fe549789ef582fc7065b3104053c93bc.jpg)

![dfd7d650c68994dafcaaee435daa0e5a502561d91cf97ae0f8e0ea7a23997e13.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/dfd7d650c68994dafcaaee435daa0e5a502561d91cf97ae0f8e0ea7a23997e13.jpg)

![e65b17018e658a478cee509f7d973bbd6a75fb79ee823c63f2f42192665ec43e.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/e65b17018e658a478cee509f7d973bbd6a75fb79ee823c63f2f42192665ec43e.jpg)

![ec71cf4fc7afd1e394a0ea0f15c719d4a793e37292824aa7ffa701d620b8ee25.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/ec71cf4fc7afd1e394a0ea0f15c719d4a793e37292824aa7ffa701d620b8ee25.jpg)

![eeeb95c2be77411d9f19853a843cbee393d1edc22cb697e6f029b1fd509e29c3.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/eeeb95c2be77411d9f19853a843cbee393d1edc22cb697e6f029b1fd509e29c3.jpg)

![fae2096f36e4cc36cd197bfa04ab6a92b2aeb51febae23726cc993ba7b6efaa7.jpg](../iclr_results/427_Towards General-Purpose Model-Free Reinforcement Learning/tables/fae2096f36e4cc36cd197bfa04ab6a92b2aeb51febae23726cc993ba7b6efaa7.jpg)

## The Computational Complexity of Circuit Discovery for Inner Interpretability


### Images

![1b82fb9b55ebf54b252aa3bac6c0e85408da7e1119426694e691ee1175213929.jpg](../iclr_results/428_The Computational Complexity of Circuit Discovery for Inner Interpretability/images/1b82fb9b55ebf54b252aa3bac6c0e85408da7e1119426694e691ee1175213929.jpg)

### Tables

![2c2e4cd958ef8a0378de4bd31f801e8a621ce441c810350fbaf214a0139f0475.jpg](../iclr_results/428_The Computational Complexity of Circuit Discovery for Inner Interpretability/tables/2c2e4cd958ef8a0378de4bd31f801e8a621ce441c810350fbaf214a0139f0475.jpg)

![42e25d374249422833036f1b6ff7d692b87f5778e6f1c9731da66eabfc9256e5.jpg](../iclr_results/428_The Computational Complexity of Circuit Discovery for Inner Interpretability/tables/42e25d374249422833036f1b6ff7d692b87f5778e6f1c9731da66eabfc9256e5.jpg)

![7d8fc7f97c1bec88eccadf8b29459b723f339afba31b83921c3634121f251455.jpg](../iclr_results/428_The Computational Complexity of Circuit Discovery for Inner Interpretability/tables/7d8fc7f97c1bec88eccadf8b29459b723f339afba31b83921c3634121f251455.jpg)

![cc91253cb515836e7ddb656070ce28dc5c7d1239843af9bf74f5cf15b0c150db.jpg](../iclr_results/428_The Computational Complexity of Circuit Discovery for Inner Interpretability/tables/cc91253cb515836e7ddb656070ce28dc5c7d1239843af9bf74f5cf15b0c150db.jpg)

## Learning from End User Data with Shuffled Differential Privacy over Kernel Densities


### Images

![08b1a1ad71a542b5fcd1c6e51c1a6dc2b37913950351fd9d964b3ed6acd42326.jpg](../iclr_results/429_Learning from End User Data with Shuffled Differential Privacy over Kernel Densities/images/08b1a1ad71a542b5fcd1c6e51c1a6dc2b37913950351fd9d964b3ed6acd42326.jpg)

![14d2bcc715387324df5504af1691b2fc9fb74c80bed6b3b7c822414e0f989a4c.jpg](../iclr_results/429_Learning from End User Data with Shuffled Differential Privacy over Kernel Densities/images/14d2bcc715387324df5504af1691b2fc9fb74c80bed6b3b7c822414e0f989a4c.jpg)

![16420be748561709dcff929f888f3add62bdf859396ac052b940a5b3375397a6.jpg](../iclr_results/429_Learning from End User Data with Shuffled Differential Privacy over Kernel Densities/images/16420be748561709dcff929f888f3add62bdf859396ac052b940a5b3375397a6.jpg)

![19749adfe54a895d3d378244075335ea8ae67193f7fc43f91d71f2454642985a.jpg](../iclr_results/429_Learning from End User Data with Shuffled Differential Privacy over Kernel Densities/images/19749adfe54a895d3d378244075335ea8ae67193f7fc43f91d71f2454642985a.jpg)

![2e974a2f76ac36eae5355ce4ac219c5d443f1ed2e2ec07481455c7bb3e866648.jpg](../iclr_results/429_Learning from End User Data with Shuffled Differential Privacy over Kernel Densities/images/2e974a2f76ac36eae5355ce4ac219c5d443f1ed2e2ec07481455c7bb3e866648.jpg)

![6a19b70ab80544ba0c028a89b0e114c517efdf1b692c6fef0837f0743bcb3d27.jpg](../iclr_results/429_Learning from End User Data with Shuffled Differential Privacy over Kernel Densities/images/6a19b70ab80544ba0c028a89b0e114c517efdf1b692c6fef0837f0743bcb3d27.jpg)

![b43c718ea9bf281c4a513be3a9809f2375beb587373bbda39da4a0bc45e9c82c.jpg](../iclr_results/429_Learning from End User Data with Shuffled Differential Privacy over Kernel Densities/images/b43c718ea9bf281c4a513be3a9809f2375beb587373bbda39da4a0bc45e9c82c.jpg)

![b4747590fa3e34b70283e1e384261074bdddd03f4d877752940c490203546422.jpg](../iclr_results/429_Learning from End User Data with Shuffled Differential Privacy over Kernel Densities/images/b4747590fa3e34b70283e1e384261074bdddd03f4d877752940c490203546422.jpg)

### Tables

![2addc54ec8cac91e479108f921686223f12dba73943308cceafb977916fd5f32.jpg](../iclr_results/429_Learning from End User Data with Shuffled Differential Privacy over Kernel Densities/tables/2addc54ec8cac91e479108f921686223f12dba73943308cceafb977916fd5f32.jpg)

![91cbccf02bc4abfce04d1a45d479fb694add5f5db8899d3684eadbee8c058582.jpg](../iclr_results/429_Learning from End User Data with Shuffled Differential Privacy over Kernel Densities/tables/91cbccf02bc4abfce04d1a45d479fb694add5f5db8899d3684eadbee8c058582.jpg)

![b508dc1af2a441b75aa0f3c6ea9fec67321f311a086fa03f1b7f9c11be516b8c.jpg](../iclr_results/429_Learning from End User Data with Shuffled Differential Privacy over Kernel Densities/tables/b508dc1af2a441b75aa0f3c6ea9fec67321f311a086fa03f1b7f9c11be516b8c.jpg)

![b5e4f3a866bb6f16962d712ef0fa99adfc6d6d248e18af4a773d68b58f6d5634.jpg](../iclr_results/429_Learning from End User Data with Shuffled Differential Privacy over Kernel Densities/tables/b5e4f3a866bb6f16962d712ef0fa99adfc6d6d248e18af4a773d68b58f6d5634.jpg)

![d3e55d2ba5a8d9caf018ee7d470b6dc2d554c58bc5bc37bee9094fb2bbccd527.jpg](../iclr_results/429_Learning from End User Data with Shuffled Differential Privacy over Kernel Densities/tables/d3e55d2ba5a8d9caf018ee7d470b6dc2d554c58bc5bc37bee9094fb2bbccd527.jpg)

![eee91817a057ab1d5ae9303aea0d2198700a7602fe50b0d8a2ae95bf2b312e4e.jpg](../iclr_results/429_Learning from End User Data with Shuffled Differential Privacy over Kernel Densities/tables/eee91817a057ab1d5ae9303aea0d2198700a7602fe50b0d8a2ae95bf2b312e4e.jpg)

![f1a74c4a41c237f6239eeff6f4518e4d0727bd537d98cbd236bf427379f9956d.jpg](../iclr_results/429_Learning from End User Data with Shuffled Differential Privacy over Kernel Densities/tables/f1a74c4a41c237f6239eeff6f4518e4d0727bd537d98cbd236bf427379f9956d.jpg)

![ff76850356282d2c25f45b904f3e298935ad320b742aea931ef555317afec921.jpg](../iclr_results/429_Learning from End User Data with Shuffled Differential Privacy over Kernel Densities/tables/ff76850356282d2c25f45b904f3e298935ad320b742aea931ef555317afec921.jpg)

## VisualPredicator: Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning


### Images

![08a2fdc071fd232eb0affe63829249b63e8d5b57f88d41a489c4849f69b34dcd.jpg](../iclr_results/430_VisualPredicator_ Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning/images/08a2fdc071fd232eb0affe63829249b63e8d5b57f88d41a489c4849f69b34dcd.jpg)

![13b893dbbec85c597f5a2006fdcc90308ab72eba432a257a386801373032221c.jpg](../iclr_results/430_VisualPredicator_ Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning/images/13b893dbbec85c597f5a2006fdcc90308ab72eba432a257a386801373032221c.jpg)

![41fdc8a7e656c6b1348960af3cfe93aa8826d971465f41ff84a20a2525dad961.jpg](../iclr_results/430_VisualPredicator_ Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning/images/41fdc8a7e656c6b1348960af3cfe93aa8826d971465f41ff84a20a2525dad961.jpg)

![46b1448508282ac4828e03a3894ab4c3212c59ae43983aefe6b3161a4d3d5f69.jpg](../iclr_results/430_VisualPredicator_ Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning/images/46b1448508282ac4828e03a3894ab4c3212c59ae43983aefe6b3161a4d3d5f69.jpg)

![5d74aaee416dd7d03a93ccf909e33300ea6823c5ad1647e0eee683f2955210d7.jpg](../iclr_results/430_VisualPredicator_ Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning/images/5d74aaee416dd7d03a93ccf909e33300ea6823c5ad1647e0eee683f2955210d7.jpg)

![6cc2098fa103b7a13227c52aabd6b8e264e7ab553820d34b00fc78dbcf1be634.jpg](../iclr_results/430_VisualPredicator_ Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning/images/6cc2098fa103b7a13227c52aabd6b8e264e7ab553820d34b00fc78dbcf1be634.jpg)

![7bce5bdea850fa513a3da02a4942a3ca2b45f40afcc90e2ff2d04c7084dd9def.jpg](../iclr_results/430_VisualPredicator_ Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning/images/7bce5bdea850fa513a3da02a4942a3ca2b45f40afcc90e2ff2d04c7084dd9def.jpg)

![842c884632bf79edd00a82b4d93fa8e1279cc81aeedb547319d3ed9ce97b5005.jpg](../iclr_results/430_VisualPredicator_ Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning/images/842c884632bf79edd00a82b4d93fa8e1279cc81aeedb547319d3ed9ce97b5005.jpg)

![9a3d7201821106258b2037ffd8fd57664a8db2ae4b6d34e95fd4d74f8008a78b.jpg](../iclr_results/430_VisualPredicator_ Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning/images/9a3d7201821106258b2037ffd8fd57664a8db2ae4b6d34e95fd4d74f8008a78b.jpg)

![ae4d23656d39cbe75955be78169b17a366aca8b99368c0dc05a249418745086f.jpg](../iclr_results/430_VisualPredicator_ Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning/images/ae4d23656d39cbe75955be78169b17a366aca8b99368c0dc05a249418745086f.jpg)

![b69185c2db257a15c195b087cdc41857932d9fce7a42e52e5416088fcba9a128.jpg](../iclr_results/430_VisualPredicator_ Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning/images/b69185c2db257a15c195b087cdc41857932d9fce7a42e52e5416088fcba9a128.jpg)

![b73714d8c868eff99b20e8e1e65ac5186a462e2224505c72eccf051be136b573.jpg](../iclr_results/430_VisualPredicator_ Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning/images/b73714d8c868eff99b20e8e1e65ac5186a462e2224505c72eccf051be136b573.jpg)

![b79cf6e8dd8a11a09f0aae14220788ce0d3fda50f724d09f64ed931e3f89312e.jpg](../iclr_results/430_VisualPredicator_ Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning/images/b79cf6e8dd8a11a09f0aae14220788ce0d3fda50f724d09f64ed931e3f89312e.jpg)

![c0e1f3d3ab63af59bd813c1557dc108da77efbb6a3a2646e88ee24b9012275d3.jpg](../iclr_results/430_VisualPredicator_ Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning/images/c0e1f3d3ab63af59bd813c1557dc108da77efbb6a3a2646e88ee24b9012275d3.jpg)

![d5d8d343dd968cfbae2e851dc1f740c6d25bd3bd531470a2685b17971e5f45a8.jpg](../iclr_results/430_VisualPredicator_ Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning/images/d5d8d343dd968cfbae2e851dc1f740c6d25bd3bd531470a2685b17971e5f45a8.jpg)

### Tables

![30925329d6dab54608415bbe320e6c6c61f43676668d72525ea857f2b2a8f36f.jpg](../iclr_results/430_VisualPredicator_ Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning/tables/30925329d6dab54608415bbe320e6c6c61f43676668d72525ea857f2b2a8f36f.jpg)

![a32cb14217da580257d82e0791424cbfc23f54c424653929aa58eb2f1278c3bb.jpg](../iclr_results/430_VisualPredicator_ Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning/tables/a32cb14217da580257d82e0791424cbfc23f54c424653929aa58eb2f1278c3bb.jpg)

![d9ba79d2ef344c0b058264bd0f1db33d15d754ff6fb41ec5c17f80acf181bc98.jpg](../iclr_results/430_VisualPredicator_ Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning/tables/d9ba79d2ef344c0b058264bd0f1db33d15d754ff6fb41ec5c17f80acf181bc98.jpg)

## Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement


### Images

![03ef0e92fc988f7caaf906c00ad8b729d53554fa9e252d0c866c88500f5ecc56.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/images/03ef0e92fc988f7caaf906c00ad8b729d53554fa9e252d0c866c88500f5ecc56.jpg)

![127bda72ef0d8be846af70f497b7428114170e95aa4d7adbe996f475d5dc9609.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/images/127bda72ef0d8be846af70f497b7428114170e95aa4d7adbe996f475d5dc9609.jpg)

![143c49101e588274b0baeab44a5183f6c26f30940f7adb22448758ab8881e172.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/images/143c49101e588274b0baeab44a5183f6c26f30940f7adb22448758ab8881e172.jpg)

![314f033184a1dd7f6454097a12fb5cff21b81e5776e59966c4372ae41b9d5b2d.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/images/314f033184a1dd7f6454097a12fb5cff21b81e5776e59966c4372ae41b9d5b2d.jpg)

![329389f494676e994b77ea26f4936e870951f3e2d1305432c6e687cd88aa060f.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/images/329389f494676e994b77ea26f4936e870951f3e2d1305432c6e687cd88aa060f.jpg)

![32f215fb8dc6a906a52ac2b87f18c713d176a978d6e040c8f941de403e1edc92.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/images/32f215fb8dc6a906a52ac2b87f18c713d176a978d6e040c8f941de403e1edc92.jpg)

![3f2e538a6107049053a75ddefd934306d6dbde897bd0a017163615127ad7c07e.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/images/3f2e538a6107049053a75ddefd934306d6dbde897bd0a017163615127ad7c07e.jpg)

![5a0f13defb2b1ec5e9b2c46f787566a2e68408976739679a17f2704ae5141338.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/images/5a0f13defb2b1ec5e9b2c46f787566a2e68408976739679a17f2704ae5141338.jpg)

![6bd5793c40e87aa2cdbcdb12b0ba739e49fe0426d750a28c432db63c7532462a.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/images/6bd5793c40e87aa2cdbcdb12b0ba739e49fe0426d750a28c432db63c7532462a.jpg)

![74361196a1faf0cfeb149b6490fef259dd9868081a834e56857eebe17e0acd86.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/images/74361196a1faf0cfeb149b6490fef259dd9868081a834e56857eebe17e0acd86.jpg)

![7a955b9ea0b5f8e02cb9adbafd7d8ff595079c7856980cbe88a7994a8df168df.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/images/7a955b9ea0b5f8e02cb9adbafd7d8ff595079c7856980cbe88a7994a8df168df.jpg)

![93dd3913da5f109269385522bfae7d0ce65fac790c4d59433b3e4cd6a7be2f5d.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/images/93dd3913da5f109269385522bfae7d0ce65fac790c4d59433b3e4cd6a7be2f5d.jpg)

![96c594f21cd5deaaca73fd219ee63550e5967fc1c2b7f42e55a9d0d03a22a43c.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/images/96c594f21cd5deaaca73fd219ee63550e5967fc1c2b7f42e55a9d0d03a22a43c.jpg)

![a42cde4d522e452fc3248432a331ca747575b90d28fa9f714f18271119b76c98.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/images/a42cde4d522e452fc3248432a331ca747575b90d28fa9f714f18271119b76c98.jpg)

![a549fb418aaa63f32e375a14063a0ae46c46b7d45234b1702116900242f2b1b4.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/images/a549fb418aaa63f32e375a14063a0ae46c46b7d45234b1702116900242f2b1b4.jpg)

![bf04232d5b2240d190698c980963de18d394f6ed92584284dc5f48786edf2f70.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/images/bf04232d5b2240d190698c980963de18d394f6ed92584284dc5f48786edf2f70.jpg)

![bf06b28c4abbcee8e681461455c36f0cf53ed7f44e0b04174d7e9bb854f208c5.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/images/bf06b28c4abbcee8e681461455c36f0cf53ed7f44e0b04174d7e9bb854f208c5.jpg)

![d1bb727d55bc26542034eb3e18bb68ca6abb8c04f4e71c627dbb9125072eadeb.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/images/d1bb727d55bc26542034eb3e18bb68ca6abb8c04f4e71c627dbb9125072eadeb.jpg)

![d8216edf1d5df06e22ed19bcc73907007c2092b74ce1266a7a48edaf478cb711.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/images/d8216edf1d5df06e22ed19bcc73907007c2092b74ce1266a7a48edaf478cb711.jpg)

![dda0ba6d7c3ec1809507d70f5b3b77f8c33f09f163e2e03cb2c8f259a6813717.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/images/dda0ba6d7c3ec1809507d70f5b3b77f8c33f09f163e2e03cb2c8f259a6813717.jpg)

![e9fbfd48733ed9cacc4cf614a6e92fb086e7435977992846ab1e8796f7b558f6.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/images/e9fbfd48733ed9cacc4cf614a6e92fb086e7435977992846ab1e8796f7b558f6.jpg)

![f9884048fe20182fabea36ba4d440c5eb3e5e054e890279ba1e0895d612fbe71.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/images/f9884048fe20182fabea36ba4d440c5eb3e5e054e890279ba1e0895d612fbe71.jpg)

![fb8cc015a6d505d30cab2c6b5a1a42c30ac3047e27a9c18b992fe878cec7b295.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/images/fb8cc015a6d505d30cab2c6b5a1a42c30ac3047e27a9c18b992fe878cec7b295.jpg)

### Tables

![09c717c822f3b3ccc6b3135d8bfe7f4ced99d17b292de3a9b8ce7cb8a26226e9.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/tables/09c717c822f3b3ccc6b3135d8bfe7f4ced99d17b292de3a9b8ce7cb8a26226e9.jpg)

![0f34c87950bdfbe7f2a6538e3efed5fc8a7d9eb8eb40b91ad0b7842fa81e8795.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/tables/0f34c87950bdfbe7f2a6538e3efed5fc8a7d9eb8eb40b91ad0b7842fa81e8795.jpg)

![12a048de0781ec80470ad9beb9914a5a5b7ac1017452dc9c93693f9085e4ef14.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/tables/12a048de0781ec80470ad9beb9914a5a5b7ac1017452dc9c93693f9085e4ef14.jpg)

![29c4a9e2267d6e260b1c2aa0c9d69524b7b14eaa77ca918955d1f3dfd219e3c3.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/tables/29c4a9e2267d6e260b1c2aa0c9d69524b7b14eaa77ca918955d1f3dfd219e3c3.jpg)

![35b61777608f18e76b21e6392ea6f8ea56f8a8f9903dd6947522f46c46942298.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/tables/35b61777608f18e76b21e6392ea6f8ea56f8a8f9903dd6947522f46c46942298.jpg)

![4eac4237a20020bcdc142834d25090c43b8cdd98e1fb01796b511bb2bdbf82e7.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/tables/4eac4237a20020bcdc142834d25090c43b8cdd98e1fb01796b511bb2bdbf82e7.jpg)

![518cfd1c66ef8f801bccf6a15bbcf8185b6cf2d513c9756fe2f6f157c8168ae1.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/tables/518cfd1c66ef8f801bccf6a15bbcf8185b6cf2d513c9756fe2f6f157c8168ae1.jpg)

![5f1c64563b6840ba70a0296d9e954857f1af5a4d25a4141894620d6fc568dc6c.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/tables/5f1c64563b6840ba70a0296d9e954857f1af5a4d25a4141894620d6fc568dc6c.jpg)

![9f688b3b0247012e93b14ed5bcbb146be05ee001d647ff5f8b12e227797591e2.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/tables/9f688b3b0247012e93b14ed5bcbb146be05ee001d647ff5f8b12e227797591e2.jpg)

![c8d2bc2c151f248c1377c96da5876baf73410bb0ed9452a4b5b4d51e29d51be3.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/tables/c8d2bc2c151f248c1377c96da5876baf73410bb0ed9452a4b5b4d51e29d51be3.jpg)

![dfbae26d3472cbf512637cf191b38438a6d7fb00474c076cb2c71a106cc3dc18.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/tables/dfbae26d3472cbf512637cf191b38438a6d7fb00474c076cb2c71a106cc3dc18.jpg)

![f1528e1988032c256ee1652fc5462059d35ad97e26988c3000dfac01d52a09b4.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/tables/f1528e1988032c256ee1652fc5462059d35ad97e26988c3000dfac01d52a09b4.jpg)

![f1d1550ef8fdc2fe8866ce4b15fbf040ebe62d88c259143010a555311cffe107.jpg](../iclr_results/431_Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement/tables/f1d1550ef8fdc2fe8866ce4b15fbf040ebe62d88c259143010a555311cffe107.jpg)

## Topograph: An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation


### Images

![1c88fdb22581d1c8a1698be32306dd784787db41aed0db913d1de365fb53f8fc.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/images/1c88fdb22581d1c8a1698be32306dd784787db41aed0db913d1de365fb53f8fc.jpg)

![28d1822911159960e24a0f661200ce38bb0991daeb03ecb81cdfd1e5ef70611a.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/images/28d1822911159960e24a0f661200ce38bb0991daeb03ecb81cdfd1e5ef70611a.jpg)

![29c43d252cd2717103b794f2e974e6daffed48b78eec47300363fed6a1ccfd4e.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/images/29c43d252cd2717103b794f2e974e6daffed48b78eec47300363fed6a1ccfd4e.jpg)

![2e3ee2f4085897b7525c996b4571d27b6ae2a772edb3a0aad0639fc18e36aa81.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/images/2e3ee2f4085897b7525c996b4571d27b6ae2a772edb3a0aad0639fc18e36aa81.jpg)

![2f1d35562cf1e50e85fe4c5974123f37bfd43ffcd173cf24e445f5f095be1cad.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/images/2f1d35562cf1e50e85fe4c5974123f37bfd43ffcd173cf24e445f5f095be1cad.jpg)

![3030ac9941c16391e7f52351ef42070b4fc65a4a81f80c45d360ffa6e3d3fa13.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/images/3030ac9941c16391e7f52351ef42070b4fc65a4a81f80c45d360ffa6e3d3fa13.jpg)

![33e7371e33bb0a9382b00ade8e622f1ade858fe41bc3d27c422ae09b12d438c5.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/images/33e7371e33bb0a9382b00ade8e622f1ade858fe41bc3d27c422ae09b12d438c5.jpg)

![57f1f69125ffe7322ab083c40b4eafa57ee56596d5786bdf223d7fcbc1705028.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/images/57f1f69125ffe7322ab083c40b4eafa57ee56596d5786bdf223d7fcbc1705028.jpg)

![580535abed464e6f2416563b383bdcfa63caacd7430f4ea0c04156569c4a3186.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/images/580535abed464e6f2416563b383bdcfa63caacd7430f4ea0c04156569c4a3186.jpg)

![70bab4c4dac8aabb55341732e4453aaa83c0eb1f71a8bbb8a50b7c51debb0945.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/images/70bab4c4dac8aabb55341732e4453aaa83c0eb1f71a8bbb8a50b7c51debb0945.jpg)

![72b9ee30acc39c026e6e3392ae9f051d7d3e2faf40ec724ed10636313874278e.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/images/72b9ee30acc39c026e6e3392ae9f051d7d3e2faf40ec724ed10636313874278e.jpg)

![89c9cdc92d2c0095347bef1ace486daedae5e92eed787e5b8db32afb256d9c38.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/images/89c9cdc92d2c0095347bef1ace486daedae5e92eed787e5b8db32afb256d9c38.jpg)

![98aad04060b8bd23894f2794ac4d792262ebebe64659b3d8b0b7b79ac56a9c96.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/images/98aad04060b8bd23894f2794ac4d792262ebebe64659b3d8b0b7b79ac56a9c96.jpg)

![9e1b5e9bdb4fa6519e24c75442e6a6f4558ec4a2c17a44f9cb7bee265ff1c357.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/images/9e1b5e9bdb4fa6519e24c75442e6a6f4558ec4a2c17a44f9cb7bee265ff1c357.jpg)

![a2ec8de4b24541e297da3955548ad0033ddded33b07904852f1cfe6cc0f06a04.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/images/a2ec8de4b24541e297da3955548ad0033ddded33b07904852f1cfe6cc0f06a04.jpg)

![a6db3c5fd128236ab992ac1e6ab76c4713bcf0dbb4cdc048129af5aad94bfb2a.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/images/a6db3c5fd128236ab992ac1e6ab76c4713bcf0dbb4cdc048129af5aad94bfb2a.jpg)

![b4bbf20bbbbde2300ff6fe2cfca081e417fab5899ff5f0cc6ecbebbf51e78857.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/images/b4bbf20bbbbde2300ff6fe2cfca081e417fab5899ff5f0cc6ecbebbf51e78857.jpg)

![bf97e564f0656d6aad89f55814ff3f4617c6ff43675923d20607303dc334125b.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/images/bf97e564f0656d6aad89f55814ff3f4617c6ff43675923d20607303dc334125b.jpg)

![d833356c88fa6d44336b38e2cce2f89969f2584d4c4639b77d492853b3ae49ed.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/images/d833356c88fa6d44336b38e2cce2f89969f2584d4c4639b77d492853b3ae49ed.jpg)

![e29295bf509c3093ea5b3f235ec266afdd2767f4b0d9cfb922ee7ff2dc7fa9e3.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/images/e29295bf509c3093ea5b3f235ec266afdd2767f4b0d9cfb922ee7ff2dc7fa9e3.jpg)

![f86908040c6a781381a5ee2a2baa01ed3021ee520bd52db5887d8a4c4716adca.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/images/f86908040c6a781381a5ee2a2baa01ed3021ee520bd52db5887d8a4c4716adca.jpg)

### Tables

![0d6d97805bb557409ae3032412db3e1c4e3a271be55b002d561daf13c5bab26d.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/tables/0d6d97805bb557409ae3032412db3e1c4e3a271be55b002d561daf13c5bab26d.jpg)

![144a08e073dd1ad11118a5791ac5ce86131b21b5a6557ecf6467fd8e4e4bd838.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/tables/144a08e073dd1ad11118a5791ac5ce86131b21b5a6557ecf6467fd8e4e4bd838.jpg)

![1bc4cb41f8230c73170476b4da349e543ba6e72193e5e920b627265bda767844.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/tables/1bc4cb41f8230c73170476b4da349e543ba6e72193e5e920b627265bda767844.jpg)

![1be5c9d969bacd343e23b78a3cf5fc53aa35ee6cfc6d4a8e1ebe1c2a673de622.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/tables/1be5c9d969bacd343e23b78a3cf5fc53aa35ee6cfc6d4a8e1ebe1c2a673de622.jpg)

![8bdaa32a44fb5d0ff8a1df912ac390497ffe500339e7ffb482a15c8d3375b5d5.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/tables/8bdaa32a44fb5d0ff8a1df912ac390497ffe500339e7ffb482a15c8d3375b5d5.jpg)

![ad58c9336fa9428ce306cf2733d4b23d31e2615b7abd362451a24285409856ab.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/tables/ad58c9336fa9428ce306cf2733d4b23d31e2615b7abd362451a24285409856ab.jpg)

![cc79bace1209387499b8c0d265b43f42fbaff8bf124affa154f4f54930aeae35.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/tables/cc79bace1209387499b8c0d265b43f42fbaff8bf124affa154f4f54930aeae35.jpg)

![de0f5e0f31fba8f231aca9e0b1ed1e247ae44d92c516048d0ed67d6008c0c030.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/tables/de0f5e0f31fba8f231aca9e0b1ed1e247ae44d92c516048d0ed67d6008c0c030.jpg)

![fff4454c8ba734ec08a48751e1a7769124e2d56e45e3f8e9440bd0c406d16285.jpg](../iclr_results/432_Topograph_ An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation/tables/fff4454c8ba734ec08a48751e1a7769124e2d56e45e3f8e9440bd0c406d16285.jpg)

## LoCoDL: Communication-Efficient Distributed Learning with Local Training and Compression


### Images

![015fa6695ecb69a147b8dd94e7e60288844506cc1dbd2ca688391ea5c3ba5dbc.jpg](../iclr_results/434_LoCoDL_ Communication-Efficient Distributed Learning with Local Training and Compression/images/015fa6695ecb69a147b8dd94e7e60288844506cc1dbd2ca688391ea5c3ba5dbc.jpg)

![33a26e0aa6a66c7f7531f008b53f1fc7b06aab77af801c5875ae2efadb55e758.jpg](../iclr_results/434_LoCoDL_ Communication-Efficient Distributed Learning with Local Training and Compression/images/33a26e0aa6a66c7f7531f008b53f1fc7b06aab77af801c5875ae2efadb55e758.jpg)

![50d3d908456902211e7e4458828bbda616997f92cb3e947bf5ceb701ec359582.jpg](../iclr_results/434_LoCoDL_ Communication-Efficient Distributed Learning with Local Training and Compression/images/50d3d908456902211e7e4458828bbda616997f92cb3e947bf5ceb701ec359582.jpg)

![5d43a96bebea43fcdf326dfe3824f54d63e4240e71576d3690da66b4dddc2c04.jpg](../iclr_results/434_LoCoDL_ Communication-Efficient Distributed Learning with Local Training and Compression/images/5d43a96bebea43fcdf326dfe3824f54d63e4240e71576d3690da66b4dddc2c04.jpg)

![65e6143c7e935536e1924b596499323b630650f168b330617f33d636ce2e108d.jpg](../iclr_results/434_LoCoDL_ Communication-Efficient Distributed Learning with Local Training and Compression/images/65e6143c7e935536e1924b596499323b630650f168b330617f33d636ce2e108d.jpg)

![c6377e56e6e327a1c5b908c84ce13282d978d5e0c069e12497beff28c59effee.jpg](../iclr_results/434_LoCoDL_ Communication-Efficient Distributed Learning with Local Training and Compression/images/c6377e56e6e327a1c5b908c84ce13282d978d5e0c069e12497beff28c59effee.jpg)

![e45b53ec461a4e30aa98484ed5906a2bbdb61bf78a07c08474d4a6c4e0896fc8.jpg](../iclr_results/434_LoCoDL_ Communication-Efficient Distributed Learning with Local Training and Compression/images/e45b53ec461a4e30aa98484ed5906a2bbdb61bf78a07c08474d4a6c4e0896fc8.jpg)

### Tables

![1fea52ec3d0bdaca45018794e5b313cf3b4fa876290a9748bb8807d44817a14c.jpg](../iclr_results/434_LoCoDL_ Communication-Efficient Distributed Learning with Local Training and Compression/tables/1fea52ec3d0bdaca45018794e5b313cf3b4fa876290a9748bb8807d44817a14c.jpg)

![7810423b8d955e76d4cb10bcaecfa43e5e9df3acccebf266ddc23dd94730c906.jpg](../iclr_results/434_LoCoDL_ Communication-Efficient Distributed Learning with Local Training and Compression/tables/7810423b8d955e76d4cb10bcaecfa43e5e9df3acccebf266ddc23dd94730c906.jpg)

## Let SSMs be ConvNets: State-space Modeling with Optimal Tensor Contractions


### Images

![0d89e4c5e146607418f83538051ebe4f01764890b195e9d3d6e5d1cb56630804.jpg](../iclr_results/435_Let SSMs be ConvNets_ State-space Modeling with Optimal Tensor Contractions/images/0d89e4c5e146607418f83538051ebe4f01764890b195e9d3d6e5d1cb56630804.jpg)

![32c45b0a2e0821d08f11c5c799c5500f9848cf4f16094c108ab82aea955c9da9.jpg](../iclr_results/435_Let SSMs be ConvNets_ State-space Modeling with Optimal Tensor Contractions/images/32c45b0a2e0821d08f11c5c799c5500f9848cf4f16094c108ab82aea955c9da9.jpg)

![4d9073dab32c7077043a2d73cf4116d197c28c6e44c2510af3efaaadf51bcda0.jpg](../iclr_results/435_Let SSMs be ConvNets_ State-space Modeling with Optimal Tensor Contractions/images/4d9073dab32c7077043a2d73cf4116d197c28c6e44c2510af3efaaadf51bcda0.jpg)

![65e8140771c224966f85a9fae44bc8ddbd04065190fea51ed805748d95ee93cb.jpg](../iclr_results/435_Let SSMs be ConvNets_ State-space Modeling with Optimal Tensor Contractions/images/65e8140771c224966f85a9fae44bc8ddbd04065190fea51ed805748d95ee93cb.jpg)

![928b4cf1f8622dfe6dc1b1c2b6fe641e5f91fc8dfaa02199da6e1bbbd64a2a0c.jpg](../iclr_results/435_Let SSMs be ConvNets_ State-space Modeling with Optimal Tensor Contractions/images/928b4cf1f8622dfe6dc1b1c2b6fe641e5f91fc8dfaa02199da6e1bbbd64a2a0c.jpg)

![9b872ed67c2c6afe71628d50ca62e13ddf65b3488f8fc46f74173b88be349dec.jpg](../iclr_results/435_Let SSMs be ConvNets_ State-space Modeling with Optimal Tensor Contractions/images/9b872ed67c2c6afe71628d50ca62e13ddf65b3488f8fc46f74173b88be349dec.jpg)

![dd0d7e5cc4b87e9aa9d9a5f89887cf77fac11a63f980a4032bc624f6096d8030.jpg](../iclr_results/435_Let SSMs be ConvNets_ State-space Modeling with Optimal Tensor Contractions/images/dd0d7e5cc4b87e9aa9d9a5f89887cf77fac11a63f980a4032bc624f6096d8030.jpg)

### Tables

![354196110389036d82c72b79e6c0415c8d772daa098d96252618fdbdc7403258.jpg](../iclr_results/435_Let SSMs be ConvNets_ State-space Modeling with Optimal Tensor Contractions/tables/354196110389036d82c72b79e6c0415c8d772daa098d96252618fdbdc7403258.jpg)

![4c9c563b28d50db0f1ec7e0df780151db28954864c2c0d72269fe2eefc7c5dc7.jpg](../iclr_results/435_Let SSMs be ConvNets_ State-space Modeling with Optimal Tensor Contractions/tables/4c9c563b28d50db0f1ec7e0df780151db28954864c2c0d72269fe2eefc7c5dc7.jpg)

![617d6a21422ec83e47c309e35085784bacaa7da0f4612cfa1ca632c0d88db3be.jpg](../iclr_results/435_Let SSMs be ConvNets_ State-space Modeling with Optimal Tensor Contractions/tables/617d6a21422ec83e47c309e35085784bacaa7da0f4612cfa1ca632c0d88db3be.jpg)

![aa0174a1ae81713e6db97520b2d9352c97c32874662ab98e89e109734724a539.jpg](../iclr_results/435_Let SSMs be ConvNets_ State-space Modeling with Optimal Tensor Contractions/tables/aa0174a1ae81713e6db97520b2d9352c97c32874662ab98e89e109734724a539.jpg)

![c7acd34f05343bc19e8807baf1ecf9baf38d5a30a294e481e4967ea9cfb0d105.jpg](../iclr_results/435_Let SSMs be ConvNets_ State-space Modeling with Optimal Tensor Contractions/tables/c7acd34f05343bc19e8807baf1ecf9baf38d5a30a294e481e4967ea9cfb0d105.jpg)

![e305ba0664930b98e33124e881ae5a7ee2f9b0e268663ad119cdec7dd428d4e7.jpg](../iclr_results/435_Let SSMs be ConvNets_ State-space Modeling with Optimal Tensor Contractions/tables/e305ba0664930b98e33124e881ae5a7ee2f9b0e268663ad119cdec7dd428d4e7.jpg)

![ffc0e016eb5c67a7219b4889a0dd77582dd6b0d0156f0f4012aa3273cdcf9b97.jpg](../iclr_results/435_Let SSMs be ConvNets_ State-space Modeling with Optimal Tensor Contractions/tables/ffc0e016eb5c67a7219b4889a0dd77582dd6b0d0156f0f4012aa3273cdcf9b97.jpg)

## SoftCVI: Contrastive variational inference with self-generated soft labels


### Images

![6ae4df69a5fe28425365771c8058223386073aa73813da74b6b11905d6364f2a.jpg](../iclr_results/436_SoftCVI_ Contrastive variational inference with self-generated soft labels/images/6ae4df69a5fe28425365771c8058223386073aa73813da74b6b11905d6364f2a.jpg)

![7c0b0fa0abc748e4febf61be6b96a95df86e2f706200da52732d53897aaac615.jpg](../iclr_results/436_SoftCVI_ Contrastive variational inference with self-generated soft labels/images/7c0b0fa0abc748e4febf61be6b96a95df86e2f706200da52732d53897aaac615.jpg)

![9e49556fcbf5cb80ba89e5fe47d9d80e95957c2d88faa9c6d643a66e49d1b827.jpg](../iclr_results/436_SoftCVI_ Contrastive variational inference with self-generated soft labels/images/9e49556fcbf5cb80ba89e5fe47d9d80e95957c2d88faa9c6d643a66e49d1b827.jpg)

![9e961ecbe8b5d03278c452770ee2c62f5b09ac68b8331d38b4a61895a20887a4.jpg](../iclr_results/436_SoftCVI_ Contrastive variational inference with self-generated soft labels/images/9e961ecbe8b5d03278c452770ee2c62f5b09ac68b8331d38b4a61895a20887a4.jpg)

![bab762dc4e51263ab9e6b94c4a6acedbadf703b109542f52184d6dbe21c523ea.jpg](../iclr_results/436_SoftCVI_ Contrastive variational inference with self-generated soft labels/images/bab762dc4e51263ab9e6b94c4a6acedbadf703b109542f52184d6dbe21c523ea.jpg)

![bbbe55bf097134b70cbf47a30b500ceda2aaf696c956d2a1b9f1903e6a5a47af.jpg](../iclr_results/436_SoftCVI_ Contrastive variational inference with self-generated soft labels/images/bbbe55bf097134b70cbf47a30b500ceda2aaf696c956d2a1b9f1903e6a5a47af.jpg)

![c3aab4adbc4fae6359ccf28b798207edf4ee11a10486cba64214bb89b47eadc7.jpg](../iclr_results/436_SoftCVI_ Contrastive variational inference with self-generated soft labels/images/c3aab4adbc4fae6359ccf28b798207edf4ee11a10486cba64214bb89b47eadc7.jpg)

![c5e62d38c17b2803365b0380153339b333cf63cc718622e0f38dcde25fca101f.jpg](../iclr_results/436_SoftCVI_ Contrastive variational inference with self-generated soft labels/images/c5e62d38c17b2803365b0380153339b333cf63cc718622e0f38dcde25fca101f.jpg)

![cb18c87c20d47fb3060de017ee5b174600cac6c0c855e4bf05f1a5427871d7d4.jpg](../iclr_results/436_SoftCVI_ Contrastive variational inference with self-generated soft labels/images/cb18c87c20d47fb3060de017ee5b174600cac6c0c855e4bf05f1a5427871d7d4.jpg)

![eafd3bf9d47b511ddc313d8a3bd098388b3ed0b175281e6678205ec2e87f56ab.jpg](../iclr_results/436_SoftCVI_ Contrastive variational inference with self-generated soft labels/images/eafd3bf9d47b511ddc313d8a3bd098388b3ed0b175281e6678205ec2e87f56ab.jpg)

### Tables

![11daf0758ea0f375d12dd15b834ac8cf9c3fe4c4e8f1050b1bd4f4f8edc5e2e0.jpg](../iclr_results/436_SoftCVI_ Contrastive variational inference with self-generated soft labels/tables/11daf0758ea0f375d12dd15b834ac8cf9c3fe4c4e8f1050b1bd4f4f8edc5e2e0.jpg)

![6f4b7e1b2fbfc6edda7513471294b03e963b14783c5c148c5486ce4ed4b1ac93.jpg](../iclr_results/436_SoftCVI_ Contrastive variational inference with self-generated soft labels/tables/6f4b7e1b2fbfc6edda7513471294b03e963b14783c5c148c5486ce4ed4b1ac93.jpg)

## Adam Exploits $\ell_\infty$-geometry of Loss Landscape via Coordinate-wise Adaptivity


### Images

![41fad15bdfbf096330742402552bf6870696729e58eca3355f567445a8b82407.jpg](../iclr_results/437_Adam Exploits $_ell__infty$-geometry of Loss Landscape via Coordinate-wise Adaptivity/images/41fad15bdfbf096330742402552bf6870696729e58eca3355f567445a8b82407.jpg)

![87cc452614edb941f2eab95c55bf35ba80e2d1941967de2afbbee3c393e983b9.jpg](../iclr_results/437_Adam Exploits $_ell__infty$-geometry of Loss Landscape via Coordinate-wise Adaptivity/images/87cc452614edb941f2eab95c55bf35ba80e2d1941967de2afbbee3c393e983b9.jpg)

![f6b3855d323ad5f6d463e0f760611e4450528b22fbb8966fd4b28041c33709d4.jpg](../iclr_results/437_Adam Exploits $_ell__infty$-geometry of Loss Landscape via Coordinate-wise Adaptivity/images/f6b3855d323ad5f6d463e0f760611e4450528b22fbb8966fd4b28041c33709d4.jpg)

### Tables

![439462626e1705a76d822e2ecd63d0028d29e1168a3799ed59e89ff9cfe8b20f.jpg](../iclr_results/437_Adam Exploits $_ell__infty$-geometry of Loss Landscape via Coordinate-wise Adaptivity/tables/439462626e1705a76d822e2ecd63d0028d29e1168a3799ed59e89ff9cfe8b20f.jpg)

![8519b34d9c9927c5d53a709ca8b18be53737db2de564fc214e92b46c27693851.jpg](../iclr_results/437_Adam Exploits $_ell__infty$-geometry of Loss Landscape via Coordinate-wise Adaptivity/tables/8519b34d9c9927c5d53a709ca8b18be53737db2de564fc214e92b46c27693851.jpg)

![aa71e8907ade965400e554977cfd04f6e38b665a8bc093b0ccfbf2056e648ce4.jpg](../iclr_results/437_Adam Exploits $_ell__infty$-geometry of Loss Landscape via Coordinate-wise Adaptivity/tables/aa71e8907ade965400e554977cfd04f6e38b665a8bc093b0ccfbf2056e648ce4.jpg)

![e699cbc4f4f03193a77d813b6a702b21c71fc7b55a809e1da48b5528981df38f.jpg](../iclr_results/437_Adam Exploits $_ell__infty$-geometry of Loss Landscape via Coordinate-wise Adaptivity/tables/e699cbc4f4f03193a77d813b6a702b21c71fc7b55a809e1da48b5528981df38f.jpg)

## DailyDilemmas: Revealing Value Preferences of LLMs with Quandaries of Daily Life


### Images

![08185f7f7b6b2f60b61ee497aeed48be682aa152fbf053df6a74271c32a7c2fa.jpg](../iclr_results/438_DailyDilemmas_ Revealing Value Preferences of LLMs with Quandaries of Daily Life/images/08185f7f7b6b2f60b61ee497aeed48be682aa152fbf053df6a74271c32a7c2fa.jpg)

![26b31b3ced32eb2ca4b13be6e70870f1809b10fad8a15575d006c6cb415d55f1.jpg](../iclr_results/438_DailyDilemmas_ Revealing Value Preferences of LLMs with Quandaries of Daily Life/images/26b31b3ced32eb2ca4b13be6e70870f1809b10fad8a15575d006c6cb415d55f1.jpg)

![26ea9e5f5f7db4b8c0c385ffed8e6eefdf44fa2d4e0cfac4a24160d3c4635e8f.jpg](../iclr_results/438_DailyDilemmas_ Revealing Value Preferences of LLMs with Quandaries of Daily Life/images/26ea9e5f5f7db4b8c0c385ffed8e6eefdf44fa2d4e0cfac4a24160d3c4635e8f.jpg)

![3684912b85ed68918c9e5c03cc0fd1f8cc1b92985298382581258f28e72e46aa.jpg](../iclr_results/438_DailyDilemmas_ Revealing Value Preferences of LLMs with Quandaries of Daily Life/images/3684912b85ed68918c9e5c03cc0fd1f8cc1b92985298382581258f28e72e46aa.jpg)

![8427329aa0cc8c2189f2155da8e3e01a95c1fb590bb90b89ceca4f06f88c028c.jpg](../iclr_results/438_DailyDilemmas_ Revealing Value Preferences of LLMs with Quandaries of Daily Life/images/8427329aa0cc8c2189f2155da8e3e01a95c1fb590bb90b89ceca4f06f88c028c.jpg)

![b956cc731d5ab5e1512a8f8d7faba5e8250bcadfcf6b8d07cb55b43492ccc4af.jpg](../iclr_results/438_DailyDilemmas_ Revealing Value Preferences of LLMs with Quandaries of Daily Life/images/b956cc731d5ab5e1512a8f8d7faba5e8250bcadfcf6b8d07cb55b43492ccc4af.jpg)

### Tables

![0298134c4467fb64c36b467e878dabc05fa40ae494c4277f6522193eec85923d.jpg](../iclr_results/438_DailyDilemmas_ Revealing Value Preferences of LLMs with Quandaries of Daily Life/tables/0298134c4467fb64c36b467e878dabc05fa40ae494c4277f6522193eec85923d.jpg)

![231ce581d45db989ca92ed77ac37dc3d85204f3798ec06e5879185f028492783.jpg](../iclr_results/438_DailyDilemmas_ Revealing Value Preferences of LLMs with Quandaries of Daily Life/tables/231ce581d45db989ca92ed77ac37dc3d85204f3798ec06e5879185f028492783.jpg)

![55565a028eeb6299c19c653b1dd5842fe7b5a09910862c81d6f5d702094b06d6.jpg](../iclr_results/438_DailyDilemmas_ Revealing Value Preferences of LLMs with Quandaries of Daily Life/tables/55565a028eeb6299c19c653b1dd5842fe7b5a09910862c81d6f5d702094b06d6.jpg)

![61157229733e6590d4ae6b3f7c1ccb922c4a9b8a3c31fc4834f2a229c266dad2.jpg](../iclr_results/438_DailyDilemmas_ Revealing Value Preferences of LLMs with Quandaries of Daily Life/tables/61157229733e6590d4ae6b3f7c1ccb922c4a9b8a3c31fc4834f2a229c266dad2.jpg)

![65d6849a8c06f6f6b9cd815a2571a953e4a30e062dd98edc1707637ad6e153bd.jpg](../iclr_results/438_DailyDilemmas_ Revealing Value Preferences of LLMs with Quandaries of Daily Life/tables/65d6849a8c06f6f6b9cd815a2571a953e4a30e062dd98edc1707637ad6e153bd.jpg)

![67aa3f1ff6a3d9c77b701af6999fae074e013abfec78fe18faeec64e3c2b5d9f.jpg](../iclr_results/438_DailyDilemmas_ Revealing Value Preferences of LLMs with Quandaries of Daily Life/tables/67aa3f1ff6a3d9c77b701af6999fae074e013abfec78fe18faeec64e3c2b5d9f.jpg)

![6a0e66bd8b942b9fc4db18b50c795eff50aef9d638c6d9c7a4999feb2afb698c.jpg](../iclr_results/438_DailyDilemmas_ Revealing Value Preferences of LLMs with Quandaries of Daily Life/tables/6a0e66bd8b942b9fc4db18b50c795eff50aef9d638c6d9c7a4999feb2afb698c.jpg)

![724143f38e32ed6cd8f8db98ca211407fccef0bcd5765d4636dd0a54e9aa2eb8.jpg](../iclr_results/438_DailyDilemmas_ Revealing Value Preferences of LLMs with Quandaries of Daily Life/tables/724143f38e32ed6cd8f8db98ca211407fccef0bcd5765d4636dd0a54e9aa2eb8.jpg)

![7dd39eae8e5ed04381b6e983202b7ec4916d27336cf6d22611f1787a22d40ace.jpg](../iclr_results/438_DailyDilemmas_ Revealing Value Preferences of LLMs with Quandaries of Daily Life/tables/7dd39eae8e5ed04381b6e983202b7ec4916d27336cf6d22611f1787a22d40ace.jpg)

![b66219b1e4ead0b30eabf50e89338009ffa18a88eeaebf6a2f56439a36b62d92.jpg](../iclr_results/438_DailyDilemmas_ Revealing Value Preferences of LLMs with Quandaries of Daily Life/tables/b66219b1e4ead0b30eabf50e89338009ffa18a88eeaebf6a2f56439a36b62d92.jpg)

![b9890b2b9545d2ea14dc34a5959dfd35dfacf6deb3a69eb0d84ed19a9e1d98ad.jpg](../iclr_results/438_DailyDilemmas_ Revealing Value Preferences of LLMs with Quandaries of Daily Life/tables/b9890b2b9545d2ea14dc34a5959dfd35dfacf6deb3a69eb0d84ed19a9e1d98ad.jpg)

![d3bdb5d0bd72c73f23f3c9cea695215fed5f69f836ec58ea3228afaaee2365d7.jpg](../iclr_results/438_DailyDilemmas_ Revealing Value Preferences of LLMs with Quandaries of Daily Life/tables/d3bdb5d0bd72c73f23f3c9cea695215fed5f69f836ec58ea3228afaaee2365d7.jpg)

![eaa59a9c758804d6ec157d4fdcc236377d319c7e61486a6bb57f5f931fb5f961.jpg](../iclr_results/438_DailyDilemmas_ Revealing Value Preferences of LLMs with Quandaries of Daily Life/tables/eaa59a9c758804d6ec157d4fdcc236377d319c7e61486a6bb57f5f931fb5f961.jpg)

![f137688aacc87d7c3ab89925018fbb90aa3488bc6262c4611bffbea9b28517eb.jpg](../iclr_results/438_DailyDilemmas_ Revealing Value Preferences of LLMs with Quandaries of Daily Life/tables/f137688aacc87d7c3ab89925018fbb90aa3488bc6262c4611bffbea9b28517eb.jpg)

## u-$\mu$P: The Unit-Scaled Maximal Update Parametrization


### Images

![0497804e7e7680988b23475bfc33822269c989102162a2b5f92bed9483b506ea.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/images/0497804e7e7680988b23475bfc33822269c989102162a2b5f92bed9483b506ea.jpg)

![08251de4b3d65aa68915c624fcd7b3feddf2e3b5239ec62168708fafb52ed78c.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/images/08251de4b3d65aa68915c624fcd7b3feddf2e3b5239ec62168708fafb52ed78c.jpg)

![0d54871a6c335ac66e890c9344ca1dc1841eeaa01a3f79a628cb0f4a3b49a7f6.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/images/0d54871a6c335ac66e890c9344ca1dc1841eeaa01a3f79a628cb0f4a3b49a7f6.jpg)

![133adc89defc9c6574c353344c7fbc7ceab01b4e4e09290888931f48383f5333.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/images/133adc89defc9c6574c353344c7fbc7ceab01b4e4e09290888931f48383f5333.jpg)

![24737a9377d6cbdd17997ca67d2ce77f6c1e900b562625847232eda26f5b5964.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/images/24737a9377d6cbdd17997ca67d2ce77f6c1e900b562625847232eda26f5b5964.jpg)

![283f994169810386f750774c16959b7f1602840f74cf926cfa4bbea97fbe9850.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/images/283f994169810386f750774c16959b7f1602840f74cf926cfa4bbea97fbe9850.jpg)

![2ba5d5dae60785995d6ccb50755d670a75994243e54e9e9bd514e0c55aa9c3cd.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/images/2ba5d5dae60785995d6ccb50755d670a75994243e54e9e9bd514e0c55aa9c3cd.jpg)

![3781ef50eda473c4a484d00f38fcca9a8f37a58e20cd5dff171d7026324a6f37.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/images/3781ef50eda473c4a484d00f38fcca9a8f37a58e20cd5dff171d7026324a6f37.jpg)

![3a9ab9b3cae5e5b62d4fc9f4be5c3661ad1a5ce40521a176efc6711eab41774d.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/images/3a9ab9b3cae5e5b62d4fc9f4be5c3661ad1a5ce40521a176efc6711eab41774d.jpg)

![424d05a9997efa515a80cd4adc1f54ac2fcd07f8098395b5c01761c464fe6e73.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/images/424d05a9997efa515a80cd4adc1f54ac2fcd07f8098395b5c01761c464fe6e73.jpg)

![48cce9a8c7db05052a00b7a5ea3e0392b5afa42bfba5eb936b3b1d5cbdb362e3.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/images/48cce9a8c7db05052a00b7a5ea3e0392b5afa42bfba5eb936b3b1d5cbdb362e3.jpg)

![5707a6b37765f89bf4414555f76b1f96bbde9b4c1363f216cd6b7c00c050dba9.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/images/5707a6b37765f89bf4414555f76b1f96bbde9b4c1363f216cd6b7c00c050dba9.jpg)

![5d0959e59bf8f794c0e9d4380cc3bed2288617379b7a6c568c21f03749ae9ef6.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/images/5d0959e59bf8f794c0e9d4380cc3bed2288617379b7a6c568c21f03749ae9ef6.jpg)

![7d5af2352bd8595b9aefb0cfbc82824a38d1936d2b502c0075320ea173cc0bb0.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/images/7d5af2352bd8595b9aefb0cfbc82824a38d1936d2b502c0075320ea173cc0bb0.jpg)

![8f6eb3a97a6157a83f6e2d58b5eacabc2c188e68b64a5f3fa19aaa6978723e80.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/images/8f6eb3a97a6157a83f6e2d58b5eacabc2c188e68b64a5f3fa19aaa6978723e80.jpg)

![974a7b1dedcc6b70cb75e7d6d7c430592ae62c37da48bfd5a3cbe0ce38ef3386.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/images/974a7b1dedcc6b70cb75e7d6d7c430592ae62c37da48bfd5a3cbe0ce38ef3386.jpg)

![a29e9037798bcd59c2dc6124a3f3feb3bd6bbfa385c18819348b1e0816cadd5e.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/images/a29e9037798bcd59c2dc6124a3f3feb3bd6bbfa385c18819348b1e0816cadd5e.jpg)

![ac72a44031ef1e5a4ba87e90a8234bcb81752fda05dbde71c544849f109b8d21.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/images/ac72a44031ef1e5a4ba87e90a8234bcb81752fda05dbde71c544849f109b8d21.jpg)

![bb384c8a70b4940278b017d506bb4e988406d545452b1e9305f55729e31e01fb.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/images/bb384c8a70b4940278b017d506bb4e988406d545452b1e9305f55729e31e01fb.jpg)

![be4e3648c5fecf20e965d54be8485f460def22cff0273c2e4b4ec1382f17bb04.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/images/be4e3648c5fecf20e965d54be8485f460def22cff0273c2e4b4ec1382f17bb04.jpg)

![c5b77247f62800aaf00ebad134bd2a08f2d79937953f00bfa493eafe63fb45bf.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/images/c5b77247f62800aaf00ebad134bd2a08f2d79937953f00bfa493eafe63fb45bf.jpg)

![cae1f4f7eeda1db7bcea0555cd3a597d4258e05f683a95171c9e30b6df57b3d4.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/images/cae1f4f7eeda1db7bcea0555cd3a597d4258e05f683a95171c9e30b6df57b3d4.jpg)

![e0ab4bf40d424d3ef436b8206b9a1fd03f7e0a564d644d505267c5821c1d2770.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/images/e0ab4bf40d424d3ef436b8206b9a1fd03f7e0a564d644d505267c5821c1d2770.jpg)

### Tables

![008b6175fa1b167b5fcd072e92f6565d6ec3a08f381d3d80735d3307970917f6.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/tables/008b6175fa1b167b5fcd072e92f6565d6ec3a08f381d3d80735d3307970917f6.jpg)

![2ef90a24865a2856a2a681ce8d49d542db4784c6fac24c8ab836a37d5cf63225.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/tables/2ef90a24865a2856a2a681ce8d49d542db4784c6fac24c8ab836a37d5cf63225.jpg)

![427cae7aadbf55c98fafac8ee23b47d24d456e04fe40a6754c8ec92072e4dc53.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/tables/427cae7aadbf55c98fafac8ee23b47d24d456e04fe40a6754c8ec92072e4dc53.jpg)

![514d2a32886b0499fcd33ed56fbfdbe39b72822daecca963ffe09c889207ab01.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/tables/514d2a32886b0499fcd33ed56fbfdbe39b72822daecca963ffe09c889207ab01.jpg)

![76c7243f3c2304662a5c8f26a63586f9bc253352d2e87675a57b4e3aa9ba6b91.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/tables/76c7243f3c2304662a5c8f26a63586f9bc253352d2e87675a57b4e3aa9ba6b91.jpg)

![9fff617fd31efe603eaaf4311d78093d1f9e245a0ab5280b749ad9b6f902537e.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/tables/9fff617fd31efe603eaaf4311d78093d1f9e245a0ab5280b749ad9b6f902537e.jpg)

![a635bfe99be80f78ecb7ede93ae1e929e84bba15f9494bb62dfc7412ef3f0c87.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/tables/a635bfe99be80f78ecb7ede93ae1e929e84bba15f9494bb62dfc7412ef3f0c87.jpg)

![be8c5fd9f29d9fe57da2771e03a14b6d7a0cce42faf79f3c988e962b44a232f9.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/tables/be8c5fd9f29d9fe57da2771e03a14b6d7a0cce42faf79f3c988e962b44a232f9.jpg)

![d7a6123f5ebec1006a88d9268772f0d0daad6b76b6fc329f14c711afae2e9659.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/tables/d7a6123f5ebec1006a88d9268772f0d0daad6b76b6fc329f14c711afae2e9659.jpg)

![e5f1b2856804318919c77c87ae8244f29f72f277d75528df6a63a1af1e8c4d69.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/tables/e5f1b2856804318919c77c87ae8244f29f72f277d75528df6a63a1af1e8c4d69.jpg)

![ec265c2a9dcec05a1fa3ff00bfec244fd43a493a3b9328a1869f2c099dcd4c0d.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/tables/ec265c2a9dcec05a1fa3ff00bfec244fd43a493a3b9328a1869f2c099dcd4c0d.jpg)

![ede1d67ffe1bfd706898368a9e6ef1ec61acb5ae2f3a20dafd162c30343b6830.jpg](../iclr_results/439_u-$_mu$P_ The Unit-Scaled Maximal Update Parametrization/tables/ede1d67ffe1bfd706898368a9e6ef1ec61acb5ae2f3a20dafd162c30343b6830.jpg)

## Instance-dependent Early Stopping


### Images

![2cc9df80b557d0e54d6fb5905271f274d1757c1864dfcd99714ecf61b409a188.jpg](../iclr_results/440_Instance-dependent Early Stopping/images/2cc9df80b557d0e54d6fb5905271f274d1757c1864dfcd99714ecf61b409a188.jpg)

![30618b5ced80e38f806fa74f402ad783346f423ec1f05df05016a1d01379bb7e.jpg](../iclr_results/440_Instance-dependent Early Stopping/images/30618b5ced80e38f806fa74f402ad783346f423ec1f05df05016a1d01379bb7e.jpg)

![3406f958a795ea12f754adc950d850f0ee4297ba0c4a6d45b5b8b87b7086a1cf.jpg](../iclr_results/440_Instance-dependent Early Stopping/images/3406f958a795ea12f754adc950d850f0ee4297ba0c4a6d45b5b8b87b7086a1cf.jpg)

![64e77951c80ff71ef4674d83732c77c1a94806dc8f3a2e36fe5704cddb581519.jpg](../iclr_results/440_Instance-dependent Early Stopping/images/64e77951c80ff71ef4674d83732c77c1a94806dc8f3a2e36fe5704cddb581519.jpg)

![7fa3902aa97720e61785e9024609516dc93a26a471f96f7d25c72027e88ebf6d.jpg](../iclr_results/440_Instance-dependent Early Stopping/images/7fa3902aa97720e61785e9024609516dc93a26a471f96f7d25c72027e88ebf6d.jpg)

![959e0540915022baef7fd2d81e09604279027697c6d7717f88e1f84b04a97de5.jpg](../iclr_results/440_Instance-dependent Early Stopping/images/959e0540915022baef7fd2d81e09604279027697c6d7717f88e1f84b04a97de5.jpg)

![b8b6443a9bea469b7d148ced75f71c42e7ae40beb3a14383b185249ed3b42982.jpg](../iclr_results/440_Instance-dependent Early Stopping/images/b8b6443a9bea469b7d148ced75f71c42e7ae40beb3a14383b185249ed3b42982.jpg)

![bc72915ad8809741043a5f6f830d87c44306eec1919c8fec06211bd9dcd25164.jpg](../iclr_results/440_Instance-dependent Early Stopping/images/bc72915ad8809741043a5f6f830d87c44306eec1919c8fec06211bd9dcd25164.jpg)

### Tables

![0548587e36a00739b25308d5df9bbd2483d3c7f94b87ba053eaa7516b190f5a8.jpg](../iclr_results/440_Instance-dependent Early Stopping/tables/0548587e36a00739b25308d5df9bbd2483d3c7f94b87ba053eaa7516b190f5a8.jpg)

![06c1f1654765c30028c04659b50a79215cf2dae2f12a9b52e27c36be7ec938a0.jpg](../iclr_results/440_Instance-dependent Early Stopping/tables/06c1f1654765c30028c04659b50a79215cf2dae2f12a9b52e27c36be7ec938a0.jpg)

![23c6c13739dab67f2e73b047799a431244d7dec481e500b1329b639d78ee4754.jpg](../iclr_results/440_Instance-dependent Early Stopping/tables/23c6c13739dab67f2e73b047799a431244d7dec481e500b1329b639d78ee4754.jpg)

![2a2c244903c8be79029d45b68b07fe2aa4879ae5636f7f9cc69ab964e9074947.jpg](../iclr_results/440_Instance-dependent Early Stopping/tables/2a2c244903c8be79029d45b68b07fe2aa4879ae5636f7f9cc69ab964e9074947.jpg)

![2e134e459111e622d15d8eb406e4293b6758c080e6d986760cba615fefc360c3.jpg](../iclr_results/440_Instance-dependent Early Stopping/tables/2e134e459111e622d15d8eb406e4293b6758c080e6d986760cba615fefc360c3.jpg)

![3c05496466b141d11b17f5ac3d8fdddcb4685c1a889a83e2f1e3115e58d63931.jpg](../iclr_results/440_Instance-dependent Early Stopping/tables/3c05496466b141d11b17f5ac3d8fdddcb4685c1a889a83e2f1e3115e58d63931.jpg)

![42cb0283ee511e17ef761fd639178eff144a80c73dfae457b8ac8beb37be5257.jpg](../iclr_results/440_Instance-dependent Early Stopping/tables/42cb0283ee511e17ef761fd639178eff144a80c73dfae457b8ac8beb37be5257.jpg)

![7a0abb6e7d49616a4998e98e4d00c72fafc288393fe79bf8a67cb86d1eb4dcd7.jpg](../iclr_results/440_Instance-dependent Early Stopping/tables/7a0abb6e7d49616a4998e98e4d00c72fafc288393fe79bf8a67cb86d1eb4dcd7.jpg)

![92753a541ff07b7d39c66cf1d549e4d03261916f5d819f6c1d4c351e83e4ae40.jpg](../iclr_results/440_Instance-dependent Early Stopping/tables/92753a541ff07b7d39c66cf1d549e4d03261916f5d819f6c1d4c351e83e4ae40.jpg)

![9b98044943e428d4dc24900bc059cd9144fa0ae232319dbcf8ee33735d5467e7.jpg](../iclr_results/440_Instance-dependent Early Stopping/tables/9b98044943e428d4dc24900bc059cd9144fa0ae232319dbcf8ee33735d5467e7.jpg)

![9cf3385a4c6282be7abf3b1a04be288ae27e6d023f3bc2d4a893b3af09c45a35.jpg](../iclr_results/440_Instance-dependent Early Stopping/tables/9cf3385a4c6282be7abf3b1a04be288ae27e6d023f3bc2d4a893b3af09c45a35.jpg)

![a4b0a44af980ff47e1d86d70a16e34bece87e7b0abe0659d6f9a5a66b608f052.jpg](../iclr_results/440_Instance-dependent Early Stopping/tables/a4b0a44af980ff47e1d86d70a16e34bece87e7b0abe0659d6f9a5a66b608f052.jpg)

![ca4b9dd4554b81f8b1916e426de0aca228537179d8d7ddbd8299d3342873f52f.jpg](../iclr_results/440_Instance-dependent Early Stopping/tables/ca4b9dd4554b81f8b1916e426de0aca228537179d8d7ddbd8299d3342873f52f.jpg)

![e732d1aa4bcbb3d59c20ebb39307c459bdad8bba21dc21f9a688c9751a15bc41.jpg](../iclr_results/440_Instance-dependent Early Stopping/tables/e732d1aa4bcbb3d59c20ebb39307c459bdad8bba21dc21f9a688c9751a15bc41.jpg)

![fb1e21c24298068c8da034e2ac270704a78c0e9e8d0fdca8bb75a0d9344771e7.jpg](../iclr_results/440_Instance-dependent Early Stopping/tables/fb1e21c24298068c8da034e2ac270704a78c0e9e8d0fdca8bb75a0d9344771e7.jpg)

![fd295dc49a4aa3015be5b813200d07b6e03bd508423c6417e8b7ed1456c3dc58.jpg](../iclr_results/440_Instance-dependent Early Stopping/tables/fd295dc49a4aa3015be5b813200d07b6e03bd508423c6417e8b7ed1456c3dc58.jpg)

## Learning vector fields of differential equations on manifolds with geometrically constrained operator-valued kernels


### Images

![01499c47fd918ded34145a937a11065d676dc2181e3caacad4ebc83cfc6705fd.jpg](../iclr_results/441_Learning vector fields of differential equations on manifolds with geometrically constrained operato/images/01499c47fd918ded34145a937a11065d676dc2181e3caacad4ebc83cfc6705fd.jpg)

![1d0f5cac0834d679af7c31db44a6453c990d5a61411675b419654a3c88d56962.jpg](../iclr_results/441_Learning vector fields of differential equations on manifolds with geometrically constrained operato/images/1d0f5cac0834d679af7c31db44a6453c990d5a61411675b419654a3c88d56962.jpg)

![231e5ade6b309eef2e36ba39946cb3bb9375c71b84316e506959d154d8190735.jpg](../iclr_results/441_Learning vector fields of differential equations on manifolds with geometrically constrained operato/images/231e5ade6b309eef2e36ba39946cb3bb9375c71b84316e506959d154d8190735.jpg)

![2851f1ea73d43e93325f8abeede850c8aa64f1ce70cf488bfc375927b3861342.jpg](../iclr_results/441_Learning vector fields of differential equations on manifolds with geometrically constrained operato/images/2851f1ea73d43e93325f8abeede850c8aa64f1ce70cf488bfc375927b3861342.jpg)

![43d22d6e361a2d5146ceab728c70103ffd49a46170d27be813f3521a6dc881ad.jpg](../iclr_results/441_Learning vector fields of differential equations on manifolds with geometrically constrained operato/images/43d22d6e361a2d5146ceab728c70103ffd49a46170d27be813f3521a6dc881ad.jpg)

![54fde8155ff065950d3e4b9ca7112909503421d350c710060b0c66fb1845acf2.jpg](../iclr_results/441_Learning vector fields of differential equations on manifolds with geometrically constrained operato/images/54fde8155ff065950d3e4b9ca7112909503421d350c710060b0c66fb1845acf2.jpg)

![6783617378184b874b7d37b415608857f4accfe5b2b03349bb7c8b6148aa80c5.jpg](../iclr_results/441_Learning vector fields of differential equations on manifolds with geometrically constrained operato/images/6783617378184b874b7d37b415608857f4accfe5b2b03349bb7c8b6148aa80c5.jpg)

![67911ce81a75c45839e5268973c79a3d4643062400c09d1e1461f318170dafac.jpg](../iclr_results/441_Learning vector fields of differential equations on manifolds with geometrically constrained operato/images/67911ce81a75c45839e5268973c79a3d4643062400c09d1e1461f318170dafac.jpg)

![948a4e5ed28b4cd988e40933b91b3a3441e269c72560ac3930559107b2c1324c.jpg](../iclr_results/441_Learning vector fields of differential equations on manifolds with geometrically constrained operato/images/948a4e5ed28b4cd988e40933b91b3a3441e269c72560ac3930559107b2c1324c.jpg)

![b0489f7725e6542a4e135b4c4ec7a3920fe435fc15faa5ff4cd2d0ec3b23cd77.jpg](../iclr_results/441_Learning vector fields of differential equations on manifolds with geometrically constrained operato/images/b0489f7725e6542a4e135b4c4ec7a3920fe435fc15faa5ff4cd2d0ec3b23cd77.jpg)

![e7a9f57378bef0b7695bdfc45f45f2fd22594eee497227c5c5a185a8d9443b9e.jpg](../iclr_results/441_Learning vector fields of differential equations on manifolds with geometrically constrained operato/images/e7a9f57378bef0b7695bdfc45f45f2fd22594eee497227c5c5a185a8d9443b9e.jpg)

![f53e602b890580ba6ff55e8a291eac6149057ccc11f47805fb093bb5909e59ba.jpg](../iclr_results/441_Learning vector fields of differential equations on manifolds with geometrically constrained operato/images/f53e602b890580ba6ff55e8a291eac6149057ccc11f47805fb093bb5909e59ba.jpg)

### Tables

![2406cbd4dd312dd5ee82599f32a4b476009674e0ca14d0d2449ab90cebcea441.jpg](../iclr_results/441_Learning vector fields of differential equations on manifolds with geometrically constrained operato/tables/2406cbd4dd312dd5ee82599f32a4b476009674e0ca14d0d2449ab90cebcea441.jpg)

![496179584d68352c294e847cbe581ba8e8abb345fdc2497f2ceebec8f2a31c9b.jpg](../iclr_results/441_Learning vector fields of differential equations on manifolds with geometrically constrained operato/tables/496179584d68352c294e847cbe581ba8e8abb345fdc2497f2ceebec8f2a31c9b.jpg)

![7dd3da6de6181d219d9cd5b6adafea3d3ff8ed45a4c9ae8bb6adbe6f83d9f319.jpg](../iclr_results/441_Learning vector fields of differential equations on manifolds with geometrically constrained operato/tables/7dd3da6de6181d219d9cd5b6adafea3d3ff8ed45a4c9ae8bb6adbe6f83d9f319.jpg)

![998fcc83531aacca5b6a41b62727c8384680f584c81b503eb75c44dac0f8e730.jpg](../iclr_results/441_Learning vector fields of differential equations on manifolds with geometrically constrained operato/tables/998fcc83531aacca5b6a41b62727c8384680f584c81b503eb75c44dac0f8e730.jpg)

![9f81c7b39b61ba1c6d361f2a47a7b61014731a4b64565e483cfa68ced8d71e9f.jpg](../iclr_results/441_Learning vector fields of differential equations on manifolds with geometrically constrained operato/tables/9f81c7b39b61ba1c6d361f2a47a7b61014731a4b64565e483cfa68ced8d71e9f.jpg)

![bf0340070da7b7d3f2c20032a93c868cc5b01ba67589aecd98245b190caa001a.jpg](../iclr_results/441_Learning vector fields of differential equations on manifolds with geometrically constrained operato/tables/bf0340070da7b7d3f2c20032a93c868cc5b01ba67589aecd98245b190caa001a.jpg)

## Programming Refusal with Conditional Activation Steering


### Images

![03c239381a91ed4682ce87af69e61992cdf16e241c808e342d39428615431361.jpg](../iclr_results/442_Programming Refusal with Conditional Activation Steering/images/03c239381a91ed4682ce87af69e61992cdf16e241c808e342d39428615431361.jpg)

![12ff45b4dda3be580862584cd5b100fb8470555445af9250ce5c259b2e182eef.jpg](../iclr_results/442_Programming Refusal with Conditional Activation Steering/images/12ff45b4dda3be580862584cd5b100fb8470555445af9250ce5c259b2e182eef.jpg)

![1b5849f295c15581f07dcf0fbe2fa6c7b9deb03c4d6e10819df2225ea75c9d8b.jpg](../iclr_results/442_Programming Refusal with Conditional Activation Steering/images/1b5849f295c15581f07dcf0fbe2fa6c7b9deb03c4d6e10819df2225ea75c9d8b.jpg)

![4d271308f166a5e708ae84453b28a10da4ab14ecb22f0490e1e6d82706b376b0.jpg](../iclr_results/442_Programming Refusal with Conditional Activation Steering/images/4d271308f166a5e708ae84453b28a10da4ab14ecb22f0490e1e6d82706b376b0.jpg)

![52aab731d0f1ebd7208c2efacb98b4b8b0625ff6837d3fa134b5c1dd8d9bf68b.jpg](../iclr_results/442_Programming Refusal with Conditional Activation Steering/images/52aab731d0f1ebd7208c2efacb98b4b8b0625ff6837d3fa134b5c1dd8d9bf68b.jpg)

![5de1f9f7a4d541307e025a7ce865952f1f016c5bf0237f0c68714d8a4bc92c7a.jpg](../iclr_results/442_Programming Refusal with Conditional Activation Steering/images/5de1f9f7a4d541307e025a7ce865952f1f016c5bf0237f0c68714d8a4bc92c7a.jpg)

![7b1e82586087673d67419a183f1807ae87d121a6092876a10ec5dcfa92642b0b.jpg](../iclr_results/442_Programming Refusal with Conditional Activation Steering/images/7b1e82586087673d67419a183f1807ae87d121a6092876a10ec5dcfa92642b0b.jpg)

![7bd58cd96153f790a81fc137d08ad2f7fd8394373b78dd8338e3edbd8938dd5d.jpg](../iclr_results/442_Programming Refusal with Conditional Activation Steering/images/7bd58cd96153f790a81fc137d08ad2f7fd8394373b78dd8338e3edbd8938dd5d.jpg)

![b8263f210f162e5ff5c22cbb413954f1981366b9902bf88460eea1bbd2d83918.jpg](../iclr_results/442_Programming Refusal with Conditional Activation Steering/images/b8263f210f162e5ff5c22cbb413954f1981366b9902bf88460eea1bbd2d83918.jpg)

![c2505c7f07c5bfd65bcc52140376dde66e1764ef3d95bc07142d3c1977603baf.jpg](../iclr_results/442_Programming Refusal with Conditional Activation Steering/images/c2505c7f07c5bfd65bcc52140376dde66e1764ef3d95bc07142d3c1977603baf.jpg)

![ec8da52a72bb4edad25b482e2a6df1dc5a6430d60f311f6ea2779fa4ffaa5d0f.jpg](../iclr_results/442_Programming Refusal with Conditional Activation Steering/images/ec8da52a72bb4edad25b482e2a6df1dc5a6430d60f311f6ea2779fa4ffaa5d0f.jpg)

![f2b859423d439cbea2c65416f52dbaf32eba179428d0ecd29276708bade9b822.jpg](../iclr_results/442_Programming Refusal with Conditional Activation Steering/images/f2b859423d439cbea2c65416f52dbaf32eba179428d0ecd29276708bade9b822.jpg)

![f3ed55d25fcfc1cd3c4d0253502b747b37ec498994315b830914af143b821081.jpg](../iclr_results/442_Programming Refusal with Conditional Activation Steering/images/f3ed55d25fcfc1cd3c4d0253502b747b37ec498994315b830914af143b821081.jpg)

### Tables

![26e5ab5987f9c43309d7043fa36d9df3de3685e3693579a378e54bd6c72dd946.jpg](../iclr_results/442_Programming Refusal with Conditional Activation Steering/tables/26e5ab5987f9c43309d7043fa36d9df3de3685e3693579a378e54bd6c72dd946.jpg)

![3d2a5def49d5e0783bac943ac486e1f781e66e751bfb247636afb355aacd9de3.jpg](../iclr_results/442_Programming Refusal with Conditional Activation Steering/tables/3d2a5def49d5e0783bac943ac486e1f781e66e751bfb247636afb355aacd9de3.jpg)

![51441d559b56bfcc77bbadc86f653264839def2628f74ee9b30222a9d50d43ae.jpg](../iclr_results/442_Programming Refusal with Conditional Activation Steering/tables/51441d559b56bfcc77bbadc86f653264839def2628f74ee9b30222a9d50d43ae.jpg)

![b0604c023780ec3e99b53809e0d3d0e5bf6d56b64ce6360e5ba96fb66bd96333.jpg](../iclr_results/442_Programming Refusal with Conditional Activation Steering/tables/b0604c023780ec3e99b53809e0d3d0e5bf6d56b64ce6360e5ba96fb66bd96333.jpg)

![bfa61ac9eaeb616009c97db13267f31f1ab4ff92a63a53c74081b4011874feee.jpg](../iclr_results/442_Programming Refusal with Conditional Activation Steering/tables/bfa61ac9eaeb616009c97db13267f31f1ab4ff92a63a53c74081b4011874feee.jpg)

## Samba: Synchronized Set-of-Sequences Modeling for Multiple Object Tracking


### Images

![20ca83a101af2c77cd8676654d2efe016810aef8d6d29522ed83b0833d598789.jpg](../iclr_results/443_Samba_ Synchronized Set-of-Sequences Modeling for Multiple Object Tracking/images/20ca83a101af2c77cd8676654d2efe016810aef8d6d29522ed83b0833d598789.jpg)

![cb5f03e470e8e8e48b748b22e327f7a2a67f8bfca61df8bf2a7334c9f8fdfd36.jpg](../iclr_results/443_Samba_ Synchronized Set-of-Sequences Modeling for Multiple Object Tracking/images/cb5f03e470e8e8e48b748b22e327f7a2a67f8bfca61df8bf2a7334c9f8fdfd36.jpg)

![e5fe7bed823d614563b6fe4bf85f17c59df080aa7f47e83c660d06db7ae06dc3.jpg](../iclr_results/443_Samba_ Synchronized Set-of-Sequences Modeling for Multiple Object Tracking/images/e5fe7bed823d614563b6fe4bf85f17c59df080aa7f47e83c660d06db7ae06dc3.jpg)

### Tables

![16defa617985057682aed592064ba9060c36f1f1a6ba3885b0c1fb565954e0b1.jpg](../iclr_results/443_Samba_ Synchronized Set-of-Sequences Modeling for Multiple Object Tracking/tables/16defa617985057682aed592064ba9060c36f1f1a6ba3885b0c1fb565954e0b1.jpg)

![3fed4d15efae2a38859a75130b933b515d796d53dc715a1e439c120b089f9fdb.jpg](../iclr_results/443_Samba_ Synchronized Set-of-Sequences Modeling for Multiple Object Tracking/tables/3fed4d15efae2a38859a75130b933b515d796d53dc715a1e439c120b089f9fdb.jpg)

![c6d53b6f5f552febe34dddb8fc5eaa72b089678b0b6e08ab87022aa7f788e0e9.jpg](../iclr_results/443_Samba_ Synchronized Set-of-Sequences Modeling for Multiple Object Tracking/tables/c6d53b6f5f552febe34dddb8fc5eaa72b089678b0b6e08ab87022aa7f788e0e9.jpg)

![f7e773171edfddea3770b9d0dfecb2c5d4a6c59912478a2fc1ec0c37ff89f8b0.jpg](../iclr_results/443_Samba_ Synchronized Set-of-Sequences Modeling for Multiple Object Tracking/tables/f7e773171edfddea3770b9d0dfecb2c5d4a6c59912478a2fc1ec0c37ff89f8b0.jpg)

## A Second-Order Perspective on Model Compositionality and Incremental Learning


### Images

![952fb0f17ec1e86ef19b2d5193ef9335f744bc867df7596aac7d1e6181fca1f8.jpg](../iclr_results/446_A Second-Order Perspective on Model Compositionality and Incremental Learning/images/952fb0f17ec1e86ef19b2d5193ef9335f744bc867df7596aac7d1e6181fca1f8.jpg)

![a61971ae277f7a24ffc07cf4ffe704a02e1cdc6cf2ead1ad5764ffc06388328c.jpg](../iclr_results/446_A Second-Order Perspective on Model Compositionality and Incremental Learning/images/a61971ae277f7a24ffc07cf4ffe704a02e1cdc6cf2ead1ad5764ffc06388328c.jpg)

![db0b5140f415f5cdeab40c6c3049fabcdba5f7ef40c3fea266bc171073ffd3da.jpg](../iclr_results/446_A Second-Order Perspective on Model Compositionality and Incremental Learning/images/db0b5140f415f5cdeab40c6c3049fabcdba5f7ef40c3fea266bc171073ffd3da.jpg)

### Tables

![07dce05ee022a504fb33ea4b56e6c60eb0b87bf1f2ae6d53daee1071f19f0738.jpg](../iclr_results/446_A Second-Order Perspective on Model Compositionality and Incremental Learning/tables/07dce05ee022a504fb33ea4b56e6c60eb0b87bf1f2ae6d53daee1071f19f0738.jpg)

![147809e1ae118767380855dd8b7288c4cff286b209503a442c7a8d44b3d084af.jpg](../iclr_results/446_A Second-Order Perspective on Model Compositionality and Incremental Learning/tables/147809e1ae118767380855dd8b7288c4cff286b209503a442c7a8d44b3d084af.jpg)

![62d5b8591a109588b621eda151fa68befc07f0a53eb027c773d3e428ee6f6b57.jpg](../iclr_results/446_A Second-Order Perspective on Model Compositionality and Incremental Learning/tables/62d5b8591a109588b621eda151fa68befc07f0a53eb027c773d3e428ee6f6b57.jpg)

![7a89d4c1e1f7d3db67edcff69f16b1af0b5d99ada0f7af297eded9dae3c23cd9.jpg](../iclr_results/446_A Second-Order Perspective on Model Compositionality and Incremental Learning/tables/7a89d4c1e1f7d3db67edcff69f16b1af0b5d99ada0f7af297eded9dae3c23cd9.jpg)

![818357bff18fc28d80165e914b441a40abe0ef39485a12df2e45eb0c4dbb1dd6.jpg](../iclr_results/446_A Second-Order Perspective on Model Compositionality and Incremental Learning/tables/818357bff18fc28d80165e914b441a40abe0ef39485a12df2e45eb0c4dbb1dd6.jpg)

![944b1ddabf936aaeb7c5ec16d3333e043bcf8652982a1d433881196f35a39a66.jpg](../iclr_results/446_A Second-Order Perspective on Model Compositionality and Incremental Learning/tables/944b1ddabf936aaeb7c5ec16d3333e043bcf8652982a1d433881196f35a39a66.jpg)

![9cf96daf0a314b826e10435bd79dd485aad2a451a7a6327c919851017a268bfd.jpg](../iclr_results/446_A Second-Order Perspective on Model Compositionality and Incremental Learning/tables/9cf96daf0a314b826e10435bd79dd485aad2a451a7a6327c919851017a268bfd.jpg)

![9fe325aa1abc806c8b47cbc072a5a7949bceb429da4785df5ab94c36e55ed354.jpg](../iclr_results/446_A Second-Order Perspective on Model Compositionality and Incremental Learning/tables/9fe325aa1abc806c8b47cbc072a5a7949bceb429da4785df5ab94c36e55ed354.jpg)

![bd4a3bc2a0aeb8a4cf8a28d1bf593f64062e95cf70b923235f837d1c8e293013.jpg](../iclr_results/446_A Second-Order Perspective on Model Compositionality and Incremental Learning/tables/bd4a3bc2a0aeb8a4cf8a28d1bf593f64062e95cf70b923235f837d1c8e293013.jpg)

## Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes


### Images

![07f574464157a6ae1e493ab938690fbbb859f7e2b62201542497c4a8b4f78914.jpg](../iclr_results/447_Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes/images/07f574464157a6ae1e493ab938690fbbb859f7e2b62201542497c4a8b4f78914.jpg)

![129f2435292e2e2bc7ed74fe03aa32af14fdc5bd6f019d24a26b5cf578e29133.jpg](../iclr_results/447_Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes/images/129f2435292e2e2bc7ed74fe03aa32af14fdc5bd6f019d24a26b5cf578e29133.jpg)

![23c6a8e445de9766f76df2e50f5f62571484197d824ad9a0dd2913f2d79e04e5.jpg](../iclr_results/447_Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes/images/23c6a8e445de9766f76df2e50f5f62571484197d824ad9a0dd2913f2d79e04e5.jpg)

![330c39c348e44f7d053155d41f074bd09525ecdbf79b3b4f9ba36068e9f1c5e0.jpg](../iclr_results/447_Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes/images/330c39c348e44f7d053155d41f074bd09525ecdbf79b3b4f9ba36068e9f1c5e0.jpg)

![6ed7a1b075cf0240c6382ccc64c0d69cf1c0f9c3fe0bc40b3e6d821d36910950.jpg](../iclr_results/447_Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes/images/6ed7a1b075cf0240c6382ccc64c0d69cf1c0f9c3fe0bc40b3e6d821d36910950.jpg)

![9a895bb43ed79e1aecdcafd09632c2f7e73ba50d56356eb2777b12dd22ed223b.jpg](../iclr_results/447_Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes/images/9a895bb43ed79e1aecdcafd09632c2f7e73ba50d56356eb2777b12dd22ed223b.jpg)

![9f7664d1dd5c292da94df40309f20334232086977fc9256958ce80d614937c60.jpg](../iclr_results/447_Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes/images/9f7664d1dd5c292da94df40309f20334232086977fc9256958ce80d614937c60.jpg)

![bcc0f26f12671fd3a48dd9cb3683f66070aef976dd8495a31341a52da9808aa6.jpg](../iclr_results/447_Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes/images/bcc0f26f12671fd3a48dd9cb3683f66070aef976dd8495a31341a52da9808aa6.jpg)

![fbd27707ccaeef78f6b2b90019ac850551a7637b5644296df550ddbc669f3e24.jpg](../iclr_results/447_Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes/images/fbd27707ccaeef78f6b2b90019ac850551a7637b5644296df550ddbc669f3e24.jpg)

### Tables

![19610ea2b60b87d8b24704b6d6e424789b6148b4005bba575b80e8d1debeb2c9.jpg](../iclr_results/447_Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes/tables/19610ea2b60b87d8b24704b6d6e424789b6148b4005bba575b80e8d1debeb2c9.jpg)

![2c72957b59b7a1865d8efc55d6d937ed2cb622ea08e150dd48c218018fdebd42.jpg](../iclr_results/447_Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes/tables/2c72957b59b7a1865d8efc55d6d937ed2cb622ea08e150dd48c218018fdebd42.jpg)

![354162512b279bca8da85cb5017b2e6f73fe6fc4bd3880870e2d94562063a1f6.jpg](../iclr_results/447_Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes/tables/354162512b279bca8da85cb5017b2e6f73fe6fc4bd3880870e2d94562063a1f6.jpg)

![395f661748ede5aff6b680ace5535ca908a76b2e3d28ddc958c8c42976519b2c.jpg](../iclr_results/447_Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes/tables/395f661748ede5aff6b680ace5535ca908a76b2e3d28ddc958c8c42976519b2c.jpg)

![559ead941fc462869afea72028896f528ed0944e6ed85de2e94f3a200d17472c.jpg](../iclr_results/447_Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes/tables/559ead941fc462869afea72028896f528ed0944e6ed85de2e94f3a200d17472c.jpg)

![9bd62e35b4187eb0da79282dfd01f6712d053a192431031d489cdfdcca9e2e32.jpg](../iclr_results/447_Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes/tables/9bd62e35b4187eb0da79282dfd01f6712d053a192431031d489cdfdcca9e2e32.jpg)

![a7b7530bcd5dce422f6343b4f7da9e71fb3f692bd3491f2b4ab0d614aa03fba6.jpg](../iclr_results/447_Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes/tables/a7b7530bcd5dce422f6343b4f7da9e71fb3f692bd3491f2b4ab0d614aa03fba6.jpg)

![b3bee4a8dff7695df94c84ac3c7dc202e11fee407cab7412601f2c0313da5d34.jpg](../iclr_results/447_Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes/tables/b3bee4a8dff7695df94c84ac3c7dc202e11fee407cab7412601f2c0313da5d34.jpg)

![c0bf5141877afd20aa2009276d39c5a1b8c684039e286e981c27017d15a967d4.jpg](../iclr_results/447_Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes/tables/c0bf5141877afd20aa2009276d39c5a1b8c684039e286e981c27017d15a967d4.jpg)

![c3aee39ccd1a12efc4ad8acd986049e55ba1c67531e4556487fb65a6c992f51c.jpg](../iclr_results/447_Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes/tables/c3aee39ccd1a12efc4ad8acd986049e55ba1c67531e4556487fb65a6c992f51c.jpg)

![fa43a24be5c3d1f298cedc814c1908f6c2ed103388658b9acf3597dee1c03703.jpg](../iclr_results/447_Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes/tables/fa43a24be5c3d1f298cedc814c1908f6c2ed103388658b9acf3597dee1c03703.jpg)

## Formation of Representations in Neural Networks


### Images

![00f9a8debc528b81fd0ef1ed264a4b9b7294cd1755d7f6a5aae737f4b573408c.jpg](../iclr_results/448_Formation of Representations in Neural Networks/images/00f9a8debc528b81fd0ef1ed264a4b9b7294cd1755d7f6a5aae737f4b573408c.jpg)

![19e66b2fcfccbbdf25ceacc638d48b3706c6bf5d7204ce95ad29fabd80b0bb44.jpg](../iclr_results/448_Formation of Representations in Neural Networks/images/19e66b2fcfccbbdf25ceacc638d48b3706c6bf5d7204ce95ad29fabd80b0bb44.jpg)

![1a75f09d3b8e822254bf2a707fa0b6b6d6f27ba068854e3af52e83a7d3873bf0.jpg](../iclr_results/448_Formation of Representations in Neural Networks/images/1a75f09d3b8e822254bf2a707fa0b6b6d6f27ba068854e3af52e83a7d3873bf0.jpg)

![2401bdcc9565e19d8a0e0851f95cfe88e84cfc4d2b60c295e3dea9324715bdbb.jpg](../iclr_results/448_Formation of Representations in Neural Networks/images/2401bdcc9565e19d8a0e0851f95cfe88e84cfc4d2b60c295e3dea9324715bdbb.jpg)

![2afeebf013aeba142c241d9a68ac1f008cf48027fc0d5724e88e9ad9258e3160.jpg](../iclr_results/448_Formation of Representations in Neural Networks/images/2afeebf013aeba142c241d9a68ac1f008cf48027fc0d5724e88e9ad9258e3160.jpg)

![46e3badb0e7324d7a445e2755319de1943918850b03e17b3dc622ddfc0b40fe7.jpg](../iclr_results/448_Formation of Representations in Neural Networks/images/46e3badb0e7324d7a445e2755319de1943918850b03e17b3dc622ddfc0b40fe7.jpg)

![4763d2c354d2564aaefb130c710efbbb473b995336f92565e3a9c8dc066516dd.jpg](../iclr_results/448_Formation of Representations in Neural Networks/images/4763d2c354d2564aaefb130c710efbbb473b995336f92565e3a9c8dc066516dd.jpg)

![58704dc2d821bc61615b7e2380fe3fd4af7a193e3a5cbbc5f9d2f21dfbe89540.jpg](../iclr_results/448_Formation of Representations in Neural Networks/images/58704dc2d821bc61615b7e2380fe3fd4af7a193e3a5cbbc5f9d2f21dfbe89540.jpg)

![61e7a99f08c31dde9486da663d5a54b74c6ebc8eb85c50e843c2cfad35274eaa.jpg](../iclr_results/448_Formation of Representations in Neural Networks/images/61e7a99f08c31dde9486da663d5a54b74c6ebc8eb85c50e843c2cfad35274eaa.jpg)

![66a693deced28a782c619225a511c30b7c58be102e154ee505c59073e6000871.jpg](../iclr_results/448_Formation of Representations in Neural Networks/images/66a693deced28a782c619225a511c30b7c58be102e154ee505c59073e6000871.jpg)

![6e9b60216eb7e337dcda27a1f8abb9cbb87d38f92282122266bdbdf1b55874d2.jpg](../iclr_results/448_Formation of Representations in Neural Networks/images/6e9b60216eb7e337dcda27a1f8abb9cbb87d38f92282122266bdbdf1b55874d2.jpg)

![9651a94274e31e05e65b20543b7ee396eb9ebe0d3beca38017b5a75c95e79a8a.jpg](../iclr_results/448_Formation of Representations in Neural Networks/images/9651a94274e31e05e65b20543b7ee396eb9ebe0d3beca38017b5a75c95e79a8a.jpg)

![97f42f692c369ca4edd0c9e6409e0477e618cf1657672c594773c8acf1c1ce15.jpg](../iclr_results/448_Formation of Representations in Neural Networks/images/97f42f692c369ca4edd0c9e6409e0477e618cf1657672c594773c8acf1c1ce15.jpg)

![9aa8bc8ca0156cfe447ceb788d29cb3380613286db96aad9c6751bde869fa3e5.jpg](../iclr_results/448_Formation of Representations in Neural Networks/images/9aa8bc8ca0156cfe447ceb788d29cb3380613286db96aad9c6751bde869fa3e5.jpg)

![a3e600fcc943f41e35628edcedadb179a85da68aeb75df00977400f156cd6446.jpg](../iclr_results/448_Formation of Representations in Neural Networks/images/a3e600fcc943f41e35628edcedadb179a85da68aeb75df00977400f156cd6446.jpg)

![a525f489c442bd79938b7c7c8c8f2ee7ab2888414bb921c12d8ece058ffe4dcf.jpg](../iclr_results/448_Formation of Representations in Neural Networks/images/a525f489c442bd79938b7c7c8c8f2ee7ab2888414bb921c12d8ece058ffe4dcf.jpg)

![a57c35a28f251fe89b9cebd66efea1ae0cf76858dfa3d714abff68add1939f81.jpg](../iclr_results/448_Formation of Representations in Neural Networks/images/a57c35a28f251fe89b9cebd66efea1ae0cf76858dfa3d714abff68add1939f81.jpg)

![b34a21780d3f65de26b0d1a212f9a8cdd1a2f65ffe24c9d9f4a891116743b8de.jpg](../iclr_results/448_Formation of Representations in Neural Networks/images/b34a21780d3f65de26b0d1a212f9a8cdd1a2f65ffe24c9d9f4a891116743b8de.jpg)

![b4871a8f9608d710f30e74cbdf09abfbdb13d8a70d5dcce4ec9de18202bd92dd.jpg](../iclr_results/448_Formation of Representations in Neural Networks/images/b4871a8f9608d710f30e74cbdf09abfbdb13d8a70d5dcce4ec9de18202bd92dd.jpg)

![c3ad00da9a57f2f85de47c861c56678ea19d5d9f4db8a9904b2037e211ecc473.jpg](../iclr_results/448_Formation of Representations in Neural Networks/images/c3ad00da9a57f2f85de47c861c56678ea19d5d9f4db8a9904b2037e211ecc473.jpg)

![ca746b5a189b9bd4edff6b9ef6c532f3c9b70f8bd1a822aa0b470bbebfecf746.jpg](../iclr_results/448_Formation of Representations in Neural Networks/images/ca746b5a189b9bd4edff6b9ef6c532f3c9b70f8bd1a822aa0b470bbebfecf746.jpg)

![e3359c2f8cab8f2b344b296e7b544fcd852610b4197ed106bbdb83ff306f1a5f.jpg](../iclr_results/448_Formation of Representations in Neural Networks/images/e3359c2f8cab8f2b344b296e7b544fcd852610b4197ed106bbdb83ff306f1a5f.jpg)

### Tables

![9f09288d8ad8f6134a94541cb0fe0bfcbb8921d6d8758bf07e7d12b5f2e04a26.jpg](../iclr_results/448_Formation of Representations in Neural Networks/tables/9f09288d8ad8f6134a94541cb0fe0bfcbb8921d6d8758bf07e7d12b5f2e04a26.jpg)

## RAG-SR: Retrieval-Augmented Generation for Neural Symbolic Regression


### Images

![001a3753e6f1c51d5dd6a45ca639c10c21cf70a338845d4818b2bfd0ff06cb8f.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/001a3753e6f1c51d5dd6a45ca639c10c21cf70a338845d4818b2bfd0ff06cb8f.jpg)

![0d912c8366f769ed3b31ee750a2b2dd7ccaad29707dffb8a0ca592ae40244769.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/0d912c8366f769ed3b31ee750a2b2dd7ccaad29707dffb8a0ca592ae40244769.jpg)

![0e3ccd99b94300f6b337817799848fd4bea3869edb04c02bde01ca872bcdcc89.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/0e3ccd99b94300f6b337817799848fd4bea3869edb04c02bde01ca872bcdcc89.jpg)

![18f900f57aecb972296793fd42b990069acff7e48fde8eb83df351cad300436d.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/18f900f57aecb972296793fd42b990069acff7e48fde8eb83df351cad300436d.jpg)

![21979a62af35c44dc3622652a9fe6534e776327101c2f429a282d407dea55c4e.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/21979a62af35c44dc3622652a9fe6534e776327101c2f429a282d407dea55c4e.jpg)

![22867f19e48b20fa01f2aec8941fe7bb6d9d8203a296a831b68eeae0852aeb61.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/22867f19e48b20fa01f2aec8941fe7bb6d9d8203a296a831b68eeae0852aeb61.jpg)

![2d345926418d80c6fe0709f63b54eb04f864846384150bf09862bbae48e88fe5.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/2d345926418d80c6fe0709f63b54eb04f864846384150bf09862bbae48e88fe5.jpg)

![2e8a379c8b2c07c9d2ec67b5965ba336e545de28935bcf0bd7e2f621917c385a.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/2e8a379c8b2c07c9d2ec67b5965ba336e545de28935bcf0bd7e2f621917c385a.jpg)

![36eef77a781fce15ca3d90468c479ec43c3315e29a42aca6952535844d3b19b4.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/36eef77a781fce15ca3d90468c479ec43c3315e29a42aca6952535844d3b19b4.jpg)

![3e2a6a24369a71e6caad070f5e38fc5807a0f5046277262198e1aceab65efb00.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/3e2a6a24369a71e6caad070f5e38fc5807a0f5046277262198e1aceab65efb00.jpg)

![65a178f2984e8081df17e4a0772610082c886ffb8e034e312be22ddd4a5d8c06.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/65a178f2984e8081df17e4a0772610082c886ffb8e034e312be22ddd4a5d8c06.jpg)

![6ffdb0dba8ccef789e9fd559e0999160a95c351023fb0b397045e6481ae6ffc5.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/6ffdb0dba8ccef789e9fd559e0999160a95c351023fb0b397045e6481ae6ffc5.jpg)

![718902bb3be08b95b28184279d68f3ab5dbcad28e3d0537d0e9ebf30327ac202.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/718902bb3be08b95b28184279d68f3ab5dbcad28e3d0537d0e9ebf30327ac202.jpg)

![84a5c0aa136ef09a002d6dfd8cc2b267172f3270053b02bd2f968f627bc81e2b.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/84a5c0aa136ef09a002d6dfd8cc2b267172f3270053b02bd2f968f627bc81e2b.jpg)

![8562a5273a516ffd2898cfafbb265ee45842496a8e264d7c518c3ab2b3d49737.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/8562a5273a516ffd2898cfafbb265ee45842496a8e264d7c518c3ab2b3d49737.jpg)

![86a12ff39572720cec4113b6614d69e074824b042678aab0531aeb118f3fb796.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/86a12ff39572720cec4113b6614d69e074824b042678aab0531aeb118f3fb796.jpg)

![8c69b337eb9690c68c52959c966336d5700de9ba46766dee1edb3b11b5598741.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/8c69b337eb9690c68c52959c966336d5700de9ba46766dee1edb3b11b5598741.jpg)

![8fd0f0cd0f9b2a550ace1d8a3089ed5ddd33c2a0964f7e76dbd977d36b6e3bac.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/8fd0f0cd0f9b2a550ace1d8a3089ed5ddd33c2a0964f7e76dbd977d36b6e3bac.jpg)

![9f8784d3bbfc33c2e69801f525138f6eac8079a8e63f7cd0bb7aa1405b8dd6d8.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/9f8784d3bbfc33c2e69801f525138f6eac8079a8e63f7cd0bb7aa1405b8dd6d8.jpg)

![9ff46255689e12ebde21f2f42011c093543b4b00079f77defa143dcae8e328be.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/9ff46255689e12ebde21f2f42011c093543b4b00079f77defa143dcae8e328be.jpg)

![ac9c518091eef91db685598418803656e67ce2dde6596fbca7f1744159282dbf.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/ac9c518091eef91db685598418803656e67ce2dde6596fbca7f1744159282dbf.jpg)

![ada5468c6c9a86c97cc2bd5c3860c178286b57f357771f87bfc5aca0bc94b625.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/ada5468c6c9a86c97cc2bd5c3860c178286b57f357771f87bfc5aca0bc94b625.jpg)

![adaaa6fd80a719d1d3f6f4abfaa1b65c715dcb5f46e71ef3be447ff7d605b203.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/adaaa6fd80a719d1d3f6f4abfaa1b65c715dcb5f46e71ef3be447ff7d605b203.jpg)

![af0de81e273539aa69ea9efdf43a35f8d5ea953cf5eb02903bf60698c84de515.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/af0de81e273539aa69ea9efdf43a35f8d5ea953cf5eb02903bf60698c84de515.jpg)

![b0ddab72a8d5ef1220a20b8e92415d70d70e5df8de8fa6f1155a277ded60a1ce.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/b0ddab72a8d5ef1220a20b8e92415d70d70e5df8de8fa6f1155a277ded60a1ce.jpg)

![b9f5df6b62e503f84273d48903ecbe74c3e79d01f0ce25207f36b9f8d23fc3c9.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/b9f5df6b62e503f84273d48903ecbe74c3e79d01f0ce25207f36b9f8d23fc3c9.jpg)

![bea637c68a13e41a3a78b95cc2cf0066821736a41c02e9046f94295be081aa10.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/bea637c68a13e41a3a78b95cc2cf0066821736a41c02e9046f94295be081aa10.jpg)

![bfcb2eb46996f4fb50b01e0898971f51b89d57c82c58977aa6c9fe84eddd0edf.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/bfcb2eb46996f4fb50b01e0898971f51b89d57c82c58977aa6c9fe84eddd0edf.jpg)

![c810a0514f5ac7a321c1d9c1eaff11853e2f6d3142e84dc344f9d0afa05add88.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/c810a0514f5ac7a321c1d9c1eaff11853e2f6d3142e84dc344f9d0afa05add88.jpg)

![c93919f925161d87156f5e1236188e2ec8cf4c4492ab344e506a6779c7c1c276.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/c93919f925161d87156f5e1236188e2ec8cf4c4492ab344e506a6779c7c1c276.jpg)

![cc7b3b07afbd7bf8ab95747afe7fa2da8729d1ac9552abc98c16572d7033a1b1.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/cc7b3b07afbd7bf8ab95747afe7fa2da8729d1ac9552abc98c16572d7033a1b1.jpg)

![cef872f5e0918949a93daeccc1b21eefded3f026480d0b7be49dad62133c06f2.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/cef872f5e0918949a93daeccc1b21eefded3f026480d0b7be49dad62133c06f2.jpg)

![eae4710544e154c209f6e15389234d8ad1a2d924b74ef1a802bc4882dc2130aa.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/eae4710544e154c209f6e15389234d8ad1a2d924b74ef1a802bc4882dc2130aa.jpg)

![edf315ea263fad0dca374712651897fb40a57bfe22d953d68524b3f8ec32629a.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/edf315ea263fad0dca374712651897fb40a57bfe22d953d68524b3f8ec32629a.jpg)

![efa84c27dbf0f20e65d9dfeb628023cce0db6086ececc190ed50af66ed7655fb.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/efa84c27dbf0f20e65d9dfeb628023cce0db6086ececc190ed50af66ed7655fb.jpg)

![f85c628bec2370affd4599368beeb658fa13533c6a871f80bd62e7570e39a23d.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/images/f85c628bec2370affd4599368beeb658fa13533c6a871f80bd62e7570e39a23d.jpg)

### Tables

![03a777e6ee944adc928bbed6c60aad9e36889a5aced25c533d5654626fed094b.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/tables/03a777e6ee944adc928bbed6c60aad9e36889a5aced25c533d5654626fed094b.jpg)

![048671852c4ad482b12666445639f5c62a109587640eb2ec1ab2985174353222.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/tables/048671852c4ad482b12666445639f5c62a109587640eb2ec1ab2985174353222.jpg)

![7ce50ac532660dc8198d500de4450088eb4ea4080e47127e495881b7775b6ccc.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/tables/7ce50ac532660dc8198d500de4450088eb4ea4080e47127e495881b7775b6ccc.jpg)

![a4d18027df376122725883c931c3da6bcdd15dbc2879b5695655250c82bcdba6.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/tables/a4d18027df376122725883c931c3da6bcdd15dbc2879b5695655250c82bcdba6.jpg)

![b311ecf90fee568aa833cd3111e2441ae0dca7c084cd6a9258eacc7d133afd6a.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/tables/b311ecf90fee568aa833cd3111e2441ae0dca7c084cd6a9258eacc7d133afd6a.jpg)

![cd4c50ef76159c4d6ec5609878567e107d015d120ac364120887372f17cadb55.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/tables/cd4c50ef76159c4d6ec5609878567e107d015d120ac364120887372f17cadb55.jpg)

![cf1ce840096be4344b3541e0dbf29a0f0b414e6627cf58153ecfc56c52228f27.jpg](../iclr_results/449_RAG-SR_ Retrieval-Augmented Generation for Neural Symbolic Regression/tables/cf1ce840096be4344b3541e0dbf29a0f0b414e6627cf58153ecfc56c52228f27.jpg)

## Towards Automated Knowledge Integration From Human-Interpretable Representations


### Images

![0762332209b8ca5278ae0b6ac3117005619daa638e8a635413cdfefc9e8818b1.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/images/0762332209b8ca5278ae0b6ac3117005619daa638e8a635413cdfefc9e8818b1.jpg)

![41278ff9233ad27c688d6920a14f0d35a9e1ab7b1acb865dd04e7453cb35be28.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/images/41278ff9233ad27c688d6920a14f0d35a9e1ab7b1acb865dd04e7453cb35be28.jpg)

![68b487461781e45fb727866155557a5abfb42c4923a438f9f92b12d33cc96962.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/images/68b487461781e45fb727866155557a5abfb42c4923a438f9f92b12d33cc96962.jpg)

![7616cfbb2ce428c6eb45b44d13f973cb403169eab923c57bbc29257be7dda0cf.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/images/7616cfbb2ce428c6eb45b44d13f973cb403169eab923c57bbc29257be7dda0cf.jpg)

![7f22c89edba7ee80631d92477e367349df568d4175feb6a60ef627ba75d624d1.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/images/7f22c89edba7ee80631d92477e367349df568d4175feb6a60ef627ba75d624d1.jpg)

![99ee594bc4b4d450f6a7f246017551311d9eb71e932f59b3ec8461f0b738cbac.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/images/99ee594bc4b4d450f6a7f246017551311d9eb71e932f59b3ec8461f0b738cbac.jpg)

![9a2d9bf6fef754e5beb2cd51541c3203f5234d88e9f6b751db1e8497c395a147.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/images/9a2d9bf6fef754e5beb2cd51541c3203f5234d88e9f6b751db1e8497c395a147.jpg)

![9bb0752dd8091702b709ff06f6abd91665912510b55adabb256001ce33c897e9.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/images/9bb0752dd8091702b709ff06f6abd91665912510b55adabb256001ce33c897e9.jpg)

![a566e863f3903cb3ac2a6c24fa0e008fe1a96951daeaeeff5912c7bc63096c0d.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/images/a566e863f3903cb3ac2a6c24fa0e008fe1a96951daeaeeff5912c7bc63096c0d.jpg)

![badf1a64df30f147175508bf02d3f9450794da1a968485974221def2a796ef8f.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/images/badf1a64df30f147175508bf02d3f9450794da1a968485974221def2a796ef8f.jpg)

![bd5588cb7faf6e834d39f68caf06f57eff797386ce63e0e826454a4956a94141.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/images/bd5588cb7faf6e834d39f68caf06f57eff797386ce63e0e826454a4956a94141.jpg)

![d6591a89f9dd3bee7aeb72bcefa71473a1373da426491e90765d7e8257a56db7.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/images/d6591a89f9dd3bee7aeb72bcefa71473a1373da426491e90765d7e8257a56db7.jpg)

![e15cbe60252183d001b45670a66c4692d2e27ba324b94c82c80393b7c55d3708.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/images/e15cbe60252183d001b45670a66c4692d2e27ba324b94c82c80393b7c55d3708.jpg)

![ef1a19e294a4c283b859c9ea476c33a35ab6a271797440ace9fd38f3d522aa37.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/images/ef1a19e294a4c283b859c9ea476c33a35ab6a271797440ace9fd38f3d522aa37.jpg)

![f565b0687cc7979a720fc9b733f988e20801470ab87f4f788dbf36bdca237828.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/images/f565b0687cc7979a720fc9b733f988e20801470ab87f4f788dbf36bdca237828.jpg)

![f89bfdc6f91cb462723593e0c2c1dd99880afc6e93fca37f468ac8d5748054e1.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/images/f89bfdc6f91cb462723593e0c2c1dd99880afc6e93fca37f468ac8d5748054e1.jpg)

![fb325183546602b05840bb25cb629471b91aeb91813bb338506aca550513ab73.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/images/fb325183546602b05840bb25cb629471b91aeb91813bb338506aca550513ab73.jpg)

### Tables

![003d4fe4bd9fa12ca0b148556bd6afa2c3207adc64bc736492551b10ba74ba4c.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/tables/003d4fe4bd9fa12ca0b148556bd6afa2c3207adc64bc736492551b10ba74ba4c.jpg)

![48a971f7a3c3e0b56b51b8ae6494ae63ea60fe0ad7ac3d1ea8d90e97a96d534a.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/tables/48a971f7a3c3e0b56b51b8ae6494ae63ea60fe0ad7ac3d1ea8d90e97a96d534a.jpg)

![4907c4ffc8d5e6060043090602272aab856bc744211ef59fd10334f7d429bc40.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/tables/4907c4ffc8d5e6060043090602272aab856bc744211ef59fd10334f7d429bc40.jpg)

![5bbdd90530cc8317891c32d606c8f77124b61a92c66a268a7fa1305e13ee3d72.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/tables/5bbdd90530cc8317891c32d606c8f77124b61a92c66a268a7fa1305e13ee3d72.jpg)

![934a741259eee10d39b9c8b0b81602317f04040549148eedf5388cd6e3c1ff3d.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/tables/934a741259eee10d39b9c8b0b81602317f04040549148eedf5388cd6e3c1ff3d.jpg)

![95d07cf6f8afb4c6f06ce7ddec74dafbc124e4edfd64312e8122846bb64bcaa9.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/tables/95d07cf6f8afb4c6f06ce7ddec74dafbc124e4edfd64312e8122846bb64bcaa9.jpg)

![a7b964f3d5164c903ceae51fcf1df6ac665beb59869e873e8b799dabf0e032f8.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/tables/a7b964f3d5164c903ceae51fcf1df6ac665beb59869e873e8b799dabf0e032f8.jpg)

![e6b7e4f6ccfd291d916bb241ba0dd25f6d369545ab202503b975d10b2b93b572.jpg](../iclr_results/450_Towards Automated Knowledge Integration From Human-Interpretable Representations/tables/e6b7e4f6ccfd291d916bb241ba0dd25f6d369545ab202503b975d10b2b93b572.jpg)

## TOP-ERL: Transformer-based Off-Policy Episodic Reinforcement Learning


### Images

![2b844aa92c09d84fbd35dcbb98d0c69b57522727dd4ce1cc84d4afea69051a9d.jpg](../iclr_results/453_TOP-ERL_ Transformer-based Off-Policy Episodic Reinforcement Learning/images/2b844aa92c09d84fbd35dcbb98d0c69b57522727dd4ce1cc84d4afea69051a9d.jpg)

![36484bee0b2f688ed8d747008c671c4834c8bd935f3df72627e7be429e8a7bc5.jpg](../iclr_results/453_TOP-ERL_ Transformer-based Off-Policy Episodic Reinforcement Learning/images/36484bee0b2f688ed8d747008c671c4834c8bd935f3df72627e7be429e8a7bc5.jpg)

![41fa08f822a3c49d0ab963b973587b42f40647abf9db35946a10394099306bb7.jpg](../iclr_results/453_TOP-ERL_ Transformer-based Off-Policy Episodic Reinforcement Learning/images/41fa08f822a3c49d0ab963b973587b42f40647abf9db35946a10394099306bb7.jpg)

![4488996dd1e743490c33d8da88b46ed7e0c71d096f9583b03f30950acf81fad7.jpg](../iclr_results/453_TOP-ERL_ Transformer-based Off-Policy Episodic Reinforcement Learning/images/4488996dd1e743490c33d8da88b46ed7e0c71d096f9583b03f30950acf81fad7.jpg)

![6724c057ed1d46c76cbcbdfba52cec534f150e76aaa26bd201693ab16fd61708.jpg](../iclr_results/453_TOP-ERL_ Transformer-based Off-Policy Episodic Reinforcement Learning/images/6724c057ed1d46c76cbcbdfba52cec534f150e76aaa26bd201693ab16fd61708.jpg)

![6f5140e1215800462ea683920dab2b9730cec2216eba15adc8a6aded84ba2fe6.jpg](../iclr_results/453_TOP-ERL_ Transformer-based Off-Policy Episodic Reinforcement Learning/images/6f5140e1215800462ea683920dab2b9730cec2216eba15adc8a6aded84ba2fe6.jpg)

![746722a3869fc334261c42547f0b9c71627a6c39d46e80b8bd73d835a7263981.jpg](../iclr_results/453_TOP-ERL_ Transformer-based Off-Policy Episodic Reinforcement Learning/images/746722a3869fc334261c42547f0b9c71627a6c39d46e80b8bd73d835a7263981.jpg)

![a509449aac3cf6cd9c62b6f0a5cfc5cc1e47120790f408dc075dbad15ca5efab.jpg](../iclr_results/453_TOP-ERL_ Transformer-based Off-Policy Episodic Reinforcement Learning/images/a509449aac3cf6cd9c62b6f0a5cfc5cc1e47120790f408dc075dbad15ca5efab.jpg)

![b89fcff81c8d5a2551dec2c00941feedae72547035fca0834876175157ae52bb.jpg](../iclr_results/453_TOP-ERL_ Transformer-based Off-Policy Episodic Reinforcement Learning/images/b89fcff81c8d5a2551dec2c00941feedae72547035fca0834876175157ae52bb.jpg)

![c8958b468c97f31d8b8ba6cc54f17539ebc8706e509468373af2fa1b91138e95.jpg](../iclr_results/453_TOP-ERL_ Transformer-based Off-Policy Episodic Reinforcement Learning/images/c8958b468c97f31d8b8ba6cc54f17539ebc8706e509468373af2fa1b91138e95.jpg)

![cdea16df3de2b306f25858b44b1da68dd6bc48dce80a9727b67e9d90b7acb62e.jpg](../iclr_results/453_TOP-ERL_ Transformer-based Off-Policy Episodic Reinforcement Learning/images/cdea16df3de2b306f25858b44b1da68dd6bc48dce80a9727b67e9d90b7acb62e.jpg)

![f08564cebb387cbbbe19f8b3afaea2341ef8ce12e0a28856115a8aba11f156e0.jpg](../iclr_results/453_TOP-ERL_ Transformer-based Off-Policy Episodic Reinforcement Learning/images/f08564cebb387cbbbe19f8b3afaea2341ef8ce12e0a28856115a8aba11f156e0.jpg)

### Tables

![352311b3c331bd0db109944368aad6719fa7edfa5fb284fe6866b4834f224fcc.jpg](../iclr_results/453_TOP-ERL_ Transformer-based Off-Policy Episodic Reinforcement Learning/tables/352311b3c331bd0db109944368aad6719fa7edfa5fb284fe6866b4834f224fcc.jpg)

![a41df34b124a3f1953a8497f72bda97e4f9f93a4c500a29f5b8be2c09fd8b76e.jpg](../iclr_results/453_TOP-ERL_ Transformer-based Off-Policy Episodic Reinforcement Learning/tables/a41df34b124a3f1953a8497f72bda97e4f9f93a4c500a29f5b8be2c09fd8b76e.jpg)

![a7694fb0ccf27a86c187f47e030159de13bd36e01a6b95c7fa23b22da46fe423.jpg](../iclr_results/453_TOP-ERL_ Transformer-based Off-Policy Episodic Reinforcement Learning/tables/a7694fb0ccf27a86c187f47e030159de13bd36e01a6b95c7fa23b22da46fe423.jpg)

![b4ff4387397c2b05be6a5e578405483169944c7e329566ba1b72e6d77a1a9059.jpg](../iclr_results/453_TOP-ERL_ Transformer-based Off-Policy Episodic Reinforcement Learning/tables/b4ff4387397c2b05be6a5e578405483169944c7e329566ba1b72e6d77a1a9059.jpg)

![ba6baa7783c2ec8e4baa26badda4d3140f820013d9b1d587295745dbcb7771c8.jpg](../iclr_results/453_TOP-ERL_ Transformer-based Off-Policy Episodic Reinforcement Learning/tables/ba6baa7783c2ec8e4baa26badda4d3140f820013d9b1d587295745dbcb7771c8.jpg)

![ff0ea5de0ca8f27821faa5642f14e0bb090baaeb465917d06e9ae48ff5e1b13a.jpg](../iclr_results/453_TOP-ERL_ Transformer-based Off-Policy Episodic Reinforcement Learning/tables/ff0ea5de0ca8f27821faa5642f14e0bb090baaeb465917d06e9ae48ff5e1b13a.jpg)

## Multi-Draft Speculative Sampling: Canonical Decomposition and Theoretical Limits


### Images

![46a46e8d5a478ef4df506d7f682967e0304f225194ee948d50c0d426d64ec6ef.jpg](../iclr_results/454_Multi-Draft Speculative Sampling_ Canonical Decomposition and Theoretical Limits/images/46a46e8d5a478ef4df506d7f682967e0304f225194ee948d50c0d426d64ec6ef.jpg)

![5bf891b3e3c2bf5587c3af0830572ff3cdde76646935408667a418aaf879baed.jpg](../iclr_results/454_Multi-Draft Speculative Sampling_ Canonical Decomposition and Theoretical Limits/images/5bf891b3e3c2bf5587c3af0830572ff3cdde76646935408667a418aaf879baed.jpg)

![637badc3ff0114af0ecd9b8dd34746e407eceeb94936bf97c3d831dfbc884fb5.jpg](../iclr_results/454_Multi-Draft Speculative Sampling_ Canonical Decomposition and Theoretical Limits/images/637badc3ff0114af0ecd9b8dd34746e407eceeb94936bf97c3d831dfbc884fb5.jpg)

![80670f8ec9f5571b7b85a140a973a12adf94aff5bbf225bd3ca2aa1402629ff6.jpg](../iclr_results/454_Multi-Draft Speculative Sampling_ Canonical Decomposition and Theoretical Limits/images/80670f8ec9f5571b7b85a140a973a12adf94aff5bbf225bd3ca2aa1402629ff6.jpg)

![9b9cebff5fd0b9e0d5bc74e30096d2f607a42462a07f0a6e40b67dc1653d1d38.jpg](../iclr_results/454_Multi-Draft Speculative Sampling_ Canonical Decomposition and Theoretical Limits/images/9b9cebff5fd0b9e0d5bc74e30096d2f607a42462a07f0a6e40b67dc1653d1d38.jpg)

![a42f4a6b07aa867266791743346fbde472757e80b5edabd31f693740bde8906b.jpg](../iclr_results/454_Multi-Draft Speculative Sampling_ Canonical Decomposition and Theoretical Limits/images/a42f4a6b07aa867266791743346fbde472757e80b5edabd31f693740bde8906b.jpg)

![b013ba7f7c19cb5f2e91f1cb7ddf16dac3949a3581a10fe103aab58f3e93c8d1.jpg](../iclr_results/454_Multi-Draft Speculative Sampling_ Canonical Decomposition and Theoretical Limits/images/b013ba7f7c19cb5f2e91f1cb7ddf16dac3949a3581a10fe103aab58f3e93c8d1.jpg)

![c0802bb1cdad28ee8e38f17eb6fa197b162c9a614993d7c2e33f934ba8459876.jpg](../iclr_results/454_Multi-Draft Speculative Sampling_ Canonical Decomposition and Theoretical Limits/images/c0802bb1cdad28ee8e38f17eb6fa197b162c9a614993d7c2e33f934ba8459876.jpg)

![f0a66d443c8abff8642deb6322807d313e45f3a1fc93db10f852b86ec7cfd894.jpg](../iclr_results/454_Multi-Draft Speculative Sampling_ Canonical Decomposition and Theoretical Limits/images/f0a66d443c8abff8642deb6322807d313e45f3a1fc93db10f852b86ec7cfd894.jpg)

### Tables

![3ad49c840b578ac0a4cace4856b5f50fd163f805efc29dde11767a3778448ca7.jpg](../iclr_results/454_Multi-Draft Speculative Sampling_ Canonical Decomposition and Theoretical Limits/tables/3ad49c840b578ac0a4cace4856b5f50fd163f805efc29dde11767a3778448ca7.jpg)

![3f85db85c405ef88f585b3c5ee03f17fac5db6cfc7615b53e3be5721de0841e3.jpg](../iclr_results/454_Multi-Draft Speculative Sampling_ Canonical Decomposition and Theoretical Limits/tables/3f85db85c405ef88f585b3c5ee03f17fac5db6cfc7615b53e3be5721de0841e3.jpg)

![4226c384e1aa495a5ab17c5adffb56e097a111bb644d99784f9ff9509b6b4c0a.jpg](../iclr_results/454_Multi-Draft Speculative Sampling_ Canonical Decomposition and Theoretical Limits/tables/4226c384e1aa495a5ab17c5adffb56e097a111bb644d99784f9ff9509b6b4c0a.jpg)

![4ddd56ac8d1a173814fb2b50af01a03a1b59877d01eed41018f525f462d42582.jpg](../iclr_results/454_Multi-Draft Speculative Sampling_ Canonical Decomposition and Theoretical Limits/tables/4ddd56ac8d1a173814fb2b50af01a03a1b59877d01eed41018f525f462d42582.jpg)

![7c0dab80f24db29fd3ba59b5fd771041310412238db6654168c6e474d7b92e9f.jpg](../iclr_results/454_Multi-Draft Speculative Sampling_ Canonical Decomposition and Theoretical Limits/tables/7c0dab80f24db29fd3ba59b5fd771041310412238db6654168c6e474d7b92e9f.jpg)

![7f93afbd67c59667b3867528d09b9517318e5a9ea323e076a070529f0a8dd9fd.jpg](../iclr_results/454_Multi-Draft Speculative Sampling_ Canonical Decomposition and Theoretical Limits/tables/7f93afbd67c59667b3867528d09b9517318e5a9ea323e076a070529f0a8dd9fd.jpg)

![8ba0e8282ecfab147eec3dee29d37fc363d6a5b38e75812269f1c8f64b35b9ae.jpg](../iclr_results/454_Multi-Draft Speculative Sampling_ Canonical Decomposition and Theoretical Limits/tables/8ba0e8282ecfab147eec3dee29d37fc363d6a5b38e75812269f1c8f64b35b9ae.jpg)

![a2a583f2b1bc5827ebea75e2d89b407dc34a1532b6593807be86398905d4cb10.jpg](../iclr_results/454_Multi-Draft Speculative Sampling_ Canonical Decomposition and Theoretical Limits/tables/a2a583f2b1bc5827ebea75e2d89b407dc34a1532b6593807be86398905d4cb10.jpg)

![b800c1da7a916e16e09bf308a1206705407082b4ef284457232a78d4fec5ba5d.jpg](../iclr_results/454_Multi-Draft Speculative Sampling_ Canonical Decomposition and Theoretical Limits/tables/b800c1da7a916e16e09bf308a1206705407082b4ef284457232a78d4fec5ba5d.jpg)

## Anti-Exposure Bias in Diffusion Models


### Tables

![13f24e735ce6d765f9b989e6a7d467ee7e3ef347f7d335b108a99c73e0e5e1da.jpg](../iclr_results/455_Anti-Exposure Bias in Diffusion Models/tables/13f24e735ce6d765f9b989e6a7d467ee7e3ef347f7d335b108a99c73e0e5e1da.jpg)

![245f23bc5d53f84f26cdf8d3276926a452738a4545eff8114a341d2d9557e5a2.jpg](../iclr_results/455_Anti-Exposure Bias in Diffusion Models/tables/245f23bc5d53f84f26cdf8d3276926a452738a4545eff8114a341d2d9557e5a2.jpg)

![5de70fd22f9608f1108d5f94a3fd71a3051b26353deb3eff99ee8cdb602c66e1.jpg](../iclr_results/455_Anti-Exposure Bias in Diffusion Models/tables/5de70fd22f9608f1108d5f94a3fd71a3051b26353deb3eff99ee8cdb602c66e1.jpg)

![6ccf6ad4cbfb011b947468aa0ba943b04d32fa63135122b9e9f48305ee8a8780.jpg](../iclr_results/455_Anti-Exposure Bias in Diffusion Models/tables/6ccf6ad4cbfb011b947468aa0ba943b04d32fa63135122b9e9f48305ee8a8780.jpg)

![81ca06eff2f432f1f8300f2e105cb68d2fa6e2195296ffaf8df649cb782b4c16.jpg](../iclr_results/455_Anti-Exposure Bias in Diffusion Models/tables/81ca06eff2f432f1f8300f2e105cb68d2fa6e2195296ffaf8df649cb782b4c16.jpg)

![898355fbc941505c7ee150f8837639f3ccb934b33870bd3835475c676e69a4d3.jpg](../iclr_results/455_Anti-Exposure Bias in Diffusion Models/tables/898355fbc941505c7ee150f8837639f3ccb934b33870bd3835475c676e69a4d3.jpg)

![ba4b05fbfe9e018bc4a4c2020d1e18048ce6ebbaa7ebc3bddd32fadbedc24d60.jpg](../iclr_results/455_Anti-Exposure Bias in Diffusion Models/tables/ba4b05fbfe9e018bc4a4c2020d1e18048ce6ebbaa7ebc3bddd32fadbedc24d60.jpg)

![d7da77c57fadf98f1f60eb88f9b4f6030c36b2009396b3bd04d9a0f9193c2de0.jpg](../iclr_results/455_Anti-Exposure Bias in Diffusion Models/tables/d7da77c57fadf98f1f60eb88f9b4f6030c36b2009396b3bd04d9a0f9193c2de0.jpg)

![e1e196a5c8aebd0530634d4f05c51a53ddbfa85ac3a018b671b58fa39fdf6483.jpg](../iclr_results/455_Anti-Exposure Bias in Diffusion Models/tables/e1e196a5c8aebd0530634d4f05c51a53ddbfa85ac3a018b671b58fa39fdf6483.jpg)

## Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs


### Images

![0065892c88f672ecadacdfbaf9410c12cbfa0122d571937010d0a8c76d03708e.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/images/0065892c88f672ecadacdfbaf9410c12cbfa0122d571937010d0a8c76d03708e.jpg)

![3f2da595ceb6ba61a2892b12ada3caa34aaaef1979e1f24d70aa103a8b6dd435.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/images/3f2da595ceb6ba61a2892b12ada3caa34aaaef1979e1f24d70aa103a8b6dd435.jpg)

![4ef366cf6ae7617ea70f81634e93c5883934d6d27a99cdc5f279c3837c805244.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/images/4ef366cf6ae7617ea70f81634e93c5883934d6d27a99cdc5f279c3837c805244.jpg)

![53921f5cdc2a44b3e35a5e39f2b223d2a2369f78c461b31bf8f024954848ceda.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/images/53921f5cdc2a44b3e35a5e39f2b223d2a2369f78c461b31bf8f024954848ceda.jpg)

![57452ad65df384b95271350b5dca124a513708ea46f969274b8679a6e5501514.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/images/57452ad65df384b95271350b5dca124a513708ea46f969274b8679a6e5501514.jpg)

![57aae2c655bf4e4dd6229e0bad4c7fd420857cd50a9b13888fab08bd2adb105c.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/images/57aae2c655bf4e4dd6229e0bad4c7fd420857cd50a9b13888fab08bd2adb105c.jpg)

![57e5f82b138281af817a267abbdc04ff5dd024fe834eac364057ffac65aa996f.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/images/57e5f82b138281af817a267abbdc04ff5dd024fe834eac364057ffac65aa996f.jpg)

![74aa83a1c1bc4a2614c83997e73686b9c1b8e12913bd62e91b7b681db649ceab.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/images/74aa83a1c1bc4a2614c83997e73686b9c1b8e12913bd62e91b7b681db649ceab.jpg)

![7bad1f66d43310f5d5bc88eabbc51bda84fed1c8292dc44ea75acf0811ae98b4.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/images/7bad1f66d43310f5d5bc88eabbc51bda84fed1c8292dc44ea75acf0811ae98b4.jpg)

![84629e4bd0c722d61308656a2feb7a9739d4417dd3b55ce34c4bc94c6943031c.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/images/84629e4bd0c722d61308656a2feb7a9739d4417dd3b55ce34c4bc94c6943031c.jpg)

![8efcbf93146c4fe65e6fc5b202ec8c00341d757be02c122b84a2e4cc88c36c02.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/images/8efcbf93146c4fe65e6fc5b202ec8c00341d757be02c122b84a2e4cc88c36c02.jpg)

![a1a6d6bedcf5e5f317b4fdb01429b81c0d7a709d8701171d690efa8e8f0e52f4.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/images/a1a6d6bedcf5e5f317b4fdb01429b81c0d7a709d8701171d690efa8e8f0e52f4.jpg)

![cb5b276e9e1f4285a5cf462b33aa1a901a8053f7799d3bd5153a0d3cf85bcec2.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/images/cb5b276e9e1f4285a5cf462b33aa1a901a8053f7799d3bd5153a0d3cf85bcec2.jpg)

![d9dc4f092a7cd66d5b6ac811b16912e1b8fb421ae8cc734490dc3d5ccdaf4c99.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/images/d9dc4f092a7cd66d5b6ac811b16912e1b8fb421ae8cc734490dc3d5ccdaf4c99.jpg)

![e5207be83cfe49f251b82d06a6112ba1629d7258af05ab8924445563992e9121.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/images/e5207be83cfe49f251b82d06a6112ba1629d7258af05ab8924445563992e9121.jpg)

![ed49e23763a7a1948679da80c99088c8571b74d9db37477219aff6dbb0c76670.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/images/ed49e23763a7a1948679da80c99088c8571b74d9db37477219aff6dbb0c76670.jpg)

![ed9f52a3a8c1d16f96518af0a8d041d75ab7f27bf157f3ba9940fc6b239c001a.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/images/ed9f52a3a8c1d16f96518af0a8d041d75ab7f27bf157f3ba9940fc6b239c001a.jpg)

![f3220e44c28ebd5a8c1df5b23319de209ac49c636159d55c49b144b8bbbe9438.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/images/f3220e44c28ebd5a8c1df5b23319de209ac49c636159d55c49b144b8bbbe9438.jpg)

### Tables

![03a0b17fc1a8b4904e06bbbcd1a9ef828802adbafffb0f71d5faf9c850cfa4d0.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/tables/03a0b17fc1a8b4904e06bbbcd1a9ef828802adbafffb0f71d5faf9c850cfa4d0.jpg)

![088314cd864923545ed758dff181b9927e19d22e5faedba9d9af6eb3ed668560.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/tables/088314cd864923545ed758dff181b9927e19d22e5faedba9d9af6eb3ed668560.jpg)

![1e742a79ff1b1281919817141af625a2a93b62e7bb5227d5c2011bc48cc8ac09.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/tables/1e742a79ff1b1281919817141af625a2a93b62e7bb5227d5c2011bc48cc8ac09.jpg)

![374a6817ebd2bb39f744e53d5afa5218143ed4483d771328f879afa677c83800.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/tables/374a6817ebd2bb39f744e53d5afa5218143ed4483d771328f879afa677c83800.jpg)

![52971fa0135ae320d670dab26f0acbd0e2240315b7fbdc7939d6390c9c7eaec3.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/tables/52971fa0135ae320d670dab26f0acbd0e2240315b7fbdc7939d6390c9c7eaec3.jpg)

![5d10ea79d2ae1ee90c3f3e73e97c230fa366c680ca1b0d6f9c281d037cd89154.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/tables/5d10ea79d2ae1ee90c3f3e73e97c230fa366c680ca1b0d6f9c281d037cd89154.jpg)

![7b3557cf65c5b1e067f88ba20dc9322028b007d9b7daf1322cfc83da39ad7d4e.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/tables/7b3557cf65c5b1e067f88ba20dc9322028b007d9b7daf1322cfc83da39ad7d4e.jpg)

![803567b8996d89233fcaa6b3765660cf4aee3adf1ce75a1eaa00070600ec602e.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/tables/803567b8996d89233fcaa6b3765660cf4aee3adf1ce75a1eaa00070600ec602e.jpg)

![9277e659967077f1cdab1806a7fbb626cf0af24376ee13c85552b23dd7f3a872.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/tables/9277e659967077f1cdab1806a7fbb626cf0af24376ee13c85552b23dd7f3a872.jpg)

![9d95016ee50d3aa7e2d24de9782902e324de9790f226fca614400b2895bcb4c4.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/tables/9d95016ee50d3aa7e2d24de9782902e324de9790f226fca614400b2895bcb4c4.jpg)

![e8d7f6846891aaffa03461d7b5bbb946c97a15ee1cbaf36f289ad2fdd585672a.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/tables/e8d7f6846891aaffa03461d7b5bbb946c97a15ee1cbaf36f289ad2fdd585672a.jpg)

![ecddbcc27dd6a3f7e836d34e78962e02b921688cea237a8952eaeb111e95ede5.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/tables/ecddbcc27dd6a3f7e836d34e78962e02b921688cea237a8952eaeb111e95ede5.jpg)

![ff10d098a655ae19ac42fccd22e9954d6bac2549c386ed54c756f309080a92c2.jpg](../iclr_results/456_Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs/tables/ff10d098a655ae19ac42fccd22e9954d6bac2549c386ed54c756f309080a92c2.jpg)

## WildBench: Benchmarking LLMs with Challenging Tasks from Real Users in the Wild


### Images

![03f3381c8a4fc2ac96fbaae5d23d24e744ddedb6779b77bc7e31c6c7e077e581.jpg](../iclr_results/458_WildBench_ Benchmarking LLMs with Challenging Tasks from Real Users in the Wild/images/03f3381c8a4fc2ac96fbaae5d23d24e744ddedb6779b77bc7e31c6c7e077e581.jpg)

![173e4fef3dc61ac818eecb1292f2dde7b5d7aad41983f547536be23d3773008e.jpg](../iclr_results/458_WildBench_ Benchmarking LLMs with Challenging Tasks from Real Users in the Wild/images/173e4fef3dc61ac818eecb1292f2dde7b5d7aad41983f547536be23d3773008e.jpg)

![30a5e52b96069098cec56074e66f220f3fcd4b9ac4906876845ede4d05423e73.jpg](../iclr_results/458_WildBench_ Benchmarking LLMs with Challenging Tasks from Real Users in the Wild/images/30a5e52b96069098cec56074e66f220f3fcd4b9ac4906876845ede4d05423e73.jpg)

![5650f900485b5094f6eb4d2d390431e8b7e81df5fe98dac3b6901abc428611a2.jpg](../iclr_results/458_WildBench_ Benchmarking LLMs with Challenging Tasks from Real Users in the Wild/images/5650f900485b5094f6eb4d2d390431e8b7e81df5fe98dac3b6901abc428611a2.jpg)

![6a56c2b54a20a7e172835ca0827d1d9550fc280e2fb1601daa8bc05cb15810c2.jpg](../iclr_results/458_WildBench_ Benchmarking LLMs with Challenging Tasks from Real Users in the Wild/images/6a56c2b54a20a7e172835ca0827d1d9550fc280e2fb1601daa8bc05cb15810c2.jpg)

![95d05be5afb1ffed5acfd5d673c0ddca914e05b45f8aaba8bab45a23a0ad4b8e.jpg](../iclr_results/458_WildBench_ Benchmarking LLMs with Challenging Tasks from Real Users in the Wild/images/95d05be5afb1ffed5acfd5d673c0ddca914e05b45f8aaba8bab45a23a0ad4b8e.jpg)

![ae4583eb597b7a03c8a995b4cba2370eedc1a15fd3996eea01a83f330036aa90.jpg](../iclr_results/458_WildBench_ Benchmarking LLMs with Challenging Tasks from Real Users in the Wild/images/ae4583eb597b7a03c8a995b4cba2370eedc1a15fd3996eea01a83f330036aa90.jpg)

![f7e24aea8dd5c905e394d5b271556d3c5f3250e29f11597e4acc52765c3f4cf4.jpg](../iclr_results/458_WildBench_ Benchmarking LLMs with Challenging Tasks from Real Users in the Wild/images/f7e24aea8dd5c905e394d5b271556d3c5f3250e29f11597e4acc52765c3f4cf4.jpg)

![f7f7312740dd5ce2ceecd8f0b951b8f8accf8b367a39fd72084f6628386ad746.jpg](../iclr_results/458_WildBench_ Benchmarking LLMs with Challenging Tasks from Real Users in the Wild/images/f7f7312740dd5ce2ceecd8f0b951b8f8accf8b367a39fd72084f6628386ad746.jpg)

### Tables

![05a6a75ca51e8b317f914ad569f5cb71b9f1273ecb91e55b34ef428e74b33786.jpg](../iclr_results/458_WildBench_ Benchmarking LLMs with Challenging Tasks from Real Users in the Wild/tables/05a6a75ca51e8b317f914ad569f5cb71b9f1273ecb91e55b34ef428e74b33786.jpg)

![27554265f2a1f4137eaeedc3f5d238e05c82760cf58f704a0d095996763ced1a.jpg](../iclr_results/458_WildBench_ Benchmarking LLMs with Challenging Tasks from Real Users in the Wild/tables/27554265f2a1f4137eaeedc3f5d238e05c82760cf58f704a0d095996763ced1a.jpg)

![97ec482ff0ca675d5ae7a100a70f3f091a8d78fdb566c7af246efba965b12e03.jpg](../iclr_results/458_WildBench_ Benchmarking LLMs with Challenging Tasks from Real Users in the Wild/tables/97ec482ff0ca675d5ae7a100a70f3f091a8d78fdb566c7af246efba965b12e03.jpg)

![c4c9e300021471c8866f3d89da61ec8853d0c736174ac86faabf83b14ecef151.jpg](../iclr_results/458_WildBench_ Benchmarking LLMs with Challenging Tasks from Real Users in the Wild/tables/c4c9e300021471c8866f3d89da61ec8853d0c736174ac86faabf83b14ecef151.jpg)

## D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement


### Images

![14c49f68a10b971249fe145fcb746fd8cbd0ad4872c2a00c38a7983a3551ffac.jpg](../iclr_results/460_D-FINE_ Redefine Regression Task of DETRs as Fine-grained Distribution Refinement/images/14c49f68a10b971249fe145fcb746fd8cbd0ad4872c2a00c38a7983a3551ffac.jpg)

![2419b21cee78b970e5dfe58e294b8ce6a10c8b9e601a3d3fa60592e38bf47294.jpg](../iclr_results/460_D-FINE_ Redefine Regression Task of DETRs as Fine-grained Distribution Refinement/images/2419b21cee78b970e5dfe58e294b8ce6a10c8b9e601a3d3fa60592e38bf47294.jpg)

![764dc6218fb2d0a2867ec79001e96035c4e35cadbea042289b444da6b428eac0.jpg](../iclr_results/460_D-FINE_ Redefine Regression Task of DETRs as Fine-grained Distribution Refinement/images/764dc6218fb2d0a2867ec79001e96035c4e35cadbea042289b444da6b428eac0.jpg)

![8f5778399e037495cd05ade3c18b5ece8b14ce2503ed62213268bc90d15da814.jpg](../iclr_results/460_D-FINE_ Redefine Regression Task of DETRs as Fine-grained Distribution Refinement/images/8f5778399e037495cd05ade3c18b5ece8b14ce2503ed62213268bc90d15da814.jpg)

![91949b8fea56b95c3c10126bdef0b24820d33908af92ec91f48910a6f1f7f5c8.jpg](../iclr_results/460_D-FINE_ Redefine Regression Task of DETRs as Fine-grained Distribution Refinement/images/91949b8fea56b95c3c10126bdef0b24820d33908af92ec91f48910a6f1f7f5c8.jpg)

### Tables

![175e143c5f062153f61ec7e9d3ea225deb305e4cf81fa9b0c054e6b64fa7722b.jpg](../iclr_results/460_D-FINE_ Redefine Regression Task of DETRs as Fine-grained Distribution Refinement/tables/175e143c5f062153f61ec7e9d3ea225deb305e4cf81fa9b0c054e6b64fa7722b.jpg)

![298d6127479c80dd6a8e200125e82d0ced588605be82a6084bfb40dd49f567c7.jpg](../iclr_results/460_D-FINE_ Redefine Regression Task of DETRs as Fine-grained Distribution Refinement/tables/298d6127479c80dd6a8e200125e82d0ced588605be82a6084bfb40dd49f567c7.jpg)

![5373321161e106909865f147bea7b9e23911933e59e78ef116fff5d659c9a37e.jpg](../iclr_results/460_D-FINE_ Redefine Regression Task of DETRs as Fine-grained Distribution Refinement/tables/5373321161e106909865f147bea7b9e23911933e59e78ef116fff5d659c9a37e.jpg)

![86ab6fe8cb34fab0fca592f17dad402c56f082cc800e9ee28ff50089b758ac53.jpg](../iclr_results/460_D-FINE_ Redefine Regression Task of DETRs as Fine-grained Distribution Refinement/tables/86ab6fe8cb34fab0fca592f17dad402c56f082cc800e9ee28ff50089b758ac53.jpg)

![8797e8a1f8216729afd240847b2a1b8f58b270ec2f86757df79390ce0642dc2f.jpg](../iclr_results/460_D-FINE_ Redefine Regression Task of DETRs as Fine-grained Distribution Refinement/tables/8797e8a1f8216729afd240847b2a1b8f58b270ec2f86757df79390ce0642dc2f.jpg)

![a9370a7a8ee5aee6ed8ae1899161727cf49db1d01a9c3f29eb96f0627bb76402.jpg](../iclr_results/460_D-FINE_ Redefine Regression Task of DETRs as Fine-grained Distribution Refinement/tables/a9370a7a8ee5aee6ed8ae1899161727cf49db1d01a9c3f29eb96f0627bb76402.jpg)

![aa0b8a8f711be41fbe2d23b33b6b9987ef035cf5aa3659989ff4c297e0992a5d.jpg](../iclr_results/460_D-FINE_ Redefine Regression Task of DETRs as Fine-grained Distribution Refinement/tables/aa0b8a8f711be41fbe2d23b33b6b9987ef035cf5aa3659989ff4c297e0992a5d.jpg)

![da88aa36cccebcc7ce3c59789d2558fe6cfcc61d42f1476727a263c08dfda3e4.jpg](../iclr_results/460_D-FINE_ Redefine Regression Task of DETRs as Fine-grained Distribution Refinement/tables/da88aa36cccebcc7ce3c59789d2558fe6cfcc61d42f1476727a263c08dfda3e4.jpg)

![e968322b821087c573c0ea26ed92b7629167b2638421dd14ac351f44298095f6.jpg](../iclr_results/460_D-FINE_ Redefine Regression Task of DETRs as Fine-grained Distribution Refinement/tables/e968322b821087c573c0ea26ed92b7629167b2638421dd14ac351f44298095f6.jpg)

## DynamicCity: Large-Scale 4D Occupancy Generation from Dynamic Scenes


### Images

![22a95491896e8e9bfead4494c4a6cd3107cc2f23bdb4890209164918f3ba876b.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/images/22a95491896e8e9bfead4494c4a6cd3107cc2f23bdb4890209164918f3ba876b.jpg)

![2c0f9eed6568bfa2a23fee3866b52e6737219366e67c017dd5405d62f4542b60.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/images/2c0f9eed6568bfa2a23fee3866b52e6737219366e67c017dd5405d62f4542b60.jpg)

![3989a38d3573253d6d40735a6281e961e87e05c85e43c278b4e4073a98dd79e4.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/images/3989a38d3573253d6d40735a6281e961e87e05c85e43c278b4e4073a98dd79e4.jpg)

![51cf85ea42e6385b5f56edaac8b49a6be0a3ac17aefdfea2d5ca7d8256402c21.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/images/51cf85ea42e6385b5f56edaac8b49a6be0a3ac17aefdfea2d5ca7d8256402c21.jpg)

![54036bd3625d7b527e9eb2e0667957d6df48fea70e74701580ba758aab488370.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/images/54036bd3625d7b527e9eb2e0667957d6df48fea70e74701580ba758aab488370.jpg)

![67b6107663c2e95b86b3a429ae2e4cf5777eca47f68abab28e112437657d64f9.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/images/67b6107663c2e95b86b3a429ae2e4cf5777eca47f68abab28e112437657d64f9.jpg)

![6f482333587152989081598ea3cf3e79afb861e21d5d6b68385fdc29e9c22df2.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/images/6f482333587152989081598ea3cf3e79afb861e21d5d6b68385fdc29e9c22df2.jpg)

![70281fbd2d6db39197a85fb171a3b869451903667d9c06b3995fd28c5e03166e.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/images/70281fbd2d6db39197a85fb171a3b869451903667d9c06b3995fd28c5e03166e.jpg)

![74b55f35da18a9fb09fffc87be788317aebb018f7251fa147da66a8aec7674d9.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/images/74b55f35da18a9fb09fffc87be788317aebb018f7251fa147da66a8aec7674d9.jpg)

![838eef5a9f5a163eaef88cbfef2f1ccf0adbd61bff4659a214794d2339523a36.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/images/838eef5a9f5a163eaef88cbfef2f1ccf0adbd61bff4659a214794d2339523a36.jpg)

![98f6edab14fedb5ad0c8b9c1083b6e18cc9927a62646381224450a3d4ef9929e.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/images/98f6edab14fedb5ad0c8b9c1083b6e18cc9927a62646381224450a3d4ef9929e.jpg)

![9ae55e2b3a9355e90f2bbc5f8bcd2707afe3d3f79581c257af0a38c5ada9ee02.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/images/9ae55e2b3a9355e90f2bbc5f8bcd2707afe3d3f79581c257af0a38c5ada9ee02.jpg)

![a2c047315ad4e55b68a3264f16ade3acc71ec771319f20c8a67d560887fe51df.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/images/a2c047315ad4e55b68a3264f16ade3acc71ec771319f20c8a67d560887fe51df.jpg)

![a4633f59550e454ab78d5855dd8438e4db1f735f0bde09d84e5983b65a23f6f1.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/images/a4633f59550e454ab78d5855dd8438e4db1f735f0bde09d84e5983b65a23f6f1.jpg)

![c1682168325890c8c1486914d9bbe8a526a869e404757a22b015b4bd4f006a26.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/images/c1682168325890c8c1486914d9bbe8a526a869e404757a22b015b4bd4f006a26.jpg)

![e3ddb8c29ce06c6e4f656b5c8db9f01af83a5a0ee91e48ff593d0a62cfc4e14e.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/images/e3ddb8c29ce06c6e4f656b5c8db9f01af83a5a0ee91e48ff593d0a62cfc4e14e.jpg)

![fa8b64d3563f0d7be1d3f517d07cfb333c8951284cbf75e301bab3a9afe754f9.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/images/fa8b64d3563f0d7be1d3f517d07cfb333c8951284cbf75e301bab3a9afe754f9.jpg)

![fcbcd7d3e960e66168a643328cd8fee2df1b3c109fe3e36ccef018ab98a2e2d8.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/images/fcbcd7d3e960e66168a643328cd8fee2df1b3c109fe3e36ccef018ab98a2e2d8.jpg)

### Tables

![0106b2fe24eb7f66f459cdb320e441018aefd92d598078470fffef7ba6934030.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/tables/0106b2fe24eb7f66f459cdb320e441018aefd92d598078470fffef7ba6934030.jpg)

![0924d67b0a9776e84d0f83c217af3fd29e04258e1fec463f65117594c9083dd6.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/tables/0924d67b0a9776e84d0f83c217af3fd29e04258e1fec463f65117594c9083dd6.jpg)

![2fd0e24adac7d2dcd90cae7132fdacfa327a018e09b6a87c9f920c13c366593c.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/tables/2fd0e24adac7d2dcd90cae7132fdacfa327a018e09b6a87c9f920c13c366593c.jpg)

![32975f71d3c2fa53cf241b12915b8fa49c110ae54b6dacf745f36926e5f71756.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/tables/32975f71d3c2fa53cf241b12915b8fa49c110ae54b6dacf745f36926e5f71756.jpg)

![3d9c509e621cf02d0fe15893955d2cc1a038d9ea4f85ee30575906f2bd9bcfb7.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/tables/3d9c509e621cf02d0fe15893955d2cc1a038d9ea4f85ee30575906f2bd9bcfb7.jpg)

![4b787b4b06fee79510cf8d578b998d0a9db83b682cdbdea2a35d5984f86a954d.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/tables/4b787b4b06fee79510cf8d578b998d0a9db83b682cdbdea2a35d5984f86a954d.jpg)

![5a22086b76237f77f620c9370e30630124057ccd5d817661dbe7959b6ccc7a99.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/tables/5a22086b76237f77f620c9370e30630124057ccd5d817661dbe7959b6ccc7a99.jpg)

![7db7e4848aced3fab0640669afd0dc67d2c8139d53f4fcc92988ce34323523cc.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/tables/7db7e4848aced3fab0640669afd0dc67d2c8139d53f4fcc92988ce34323523cc.jpg)

![849e04324e003877a08588dabb4fa736298acc337b458ce6940a8fc576ba5b2c.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/tables/849e04324e003877a08588dabb4fa736298acc337b458ce6940a8fc576ba5b2c.jpg)

![aeae8f0ca2676135c2a057d81c65894e8990d87d4e06f1bcacd558f1d07cc9a0.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/tables/aeae8f0ca2676135c2a057d81c65894e8990d87d4e06f1bcacd558f1d07cc9a0.jpg)

![b7ae820a3ac0d31109e70becc3b566d727a7ee09e4c811523442b5ba73e66a24.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/tables/b7ae820a3ac0d31109e70becc3b566d727a7ee09e4c811523442b5ba73e66a24.jpg)

![c47e1d0aef6de98499bb409a26724aedba7c5dc4773a0298ed2e072a032d5d61.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/tables/c47e1d0aef6de98499bb409a26724aedba7c5dc4773a0298ed2e072a032d5d61.jpg)

![dd440c0b70aa96672ac042e464035ffd359a340fbfebe25fabe7fba083d88de6.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/tables/dd440c0b70aa96672ac042e464035ffd359a340fbfebe25fabe7fba083d88de6.jpg)

![df73f3a91312e16ff3b298f35a03882c394d688650de4bb152146a5d1bfdc6f9.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/tables/df73f3a91312e16ff3b298f35a03882c394d688650de4bb152146a5d1bfdc6f9.jpg)

![f45389ddf3d39666552bde27b24683b7789df139d5f60376b9e60dda5eddddc8.jpg](../iclr_results/462_DynamicCity_ Large-Scale 4D Occupancy Generation from Dynamic Scenes/tables/f45389ddf3d39666552bde27b24683b7789df139d5f60376b9e60dda5eddddc8.jpg)

## CubeDiff: Repurposing Diffusion-Based Image Models for Panorama Generation


### Images

![2493e2339c0c4fef2b347aa76f467474136b260049f6fff2ac62deb2860d1ee9.jpg](../iclr_results/463_CubeDiff_ Repurposing Diffusion-Based Image Models for Panorama Generation/images/2493e2339c0c4fef2b347aa76f467474136b260049f6fff2ac62deb2860d1ee9.jpg)

![37612ed376f5d1fc27410efaa69028e823cae877f656f3677502fc4de0159a52.jpg](../iclr_results/463_CubeDiff_ Repurposing Diffusion-Based Image Models for Panorama Generation/images/37612ed376f5d1fc27410efaa69028e823cae877f656f3677502fc4de0159a52.jpg)

![3eae7186bcfc943ebb88e70003a29198786f9782a80c7e0b61f541270b2e7be9.jpg](../iclr_results/463_CubeDiff_ Repurposing Diffusion-Based Image Models for Panorama Generation/images/3eae7186bcfc943ebb88e70003a29198786f9782a80c7e0b61f541270b2e7be9.jpg)

![5ac19fad02414c2502c09bac47bf69ebe229c8a743862d246fd0fa31199ac6e2.jpg](../iclr_results/463_CubeDiff_ Repurposing Diffusion-Based Image Models for Panorama Generation/images/5ac19fad02414c2502c09bac47bf69ebe229c8a743862d246fd0fa31199ac6e2.jpg)

![9b96c7a6c6b932e26a0904d64376376c191d67171bffc77c4f77fba42a1ee2d9.jpg](../iclr_results/463_CubeDiff_ Repurposing Diffusion-Based Image Models for Panorama Generation/images/9b96c7a6c6b932e26a0904d64376376c191d67171bffc77c4f77fba42a1ee2d9.jpg)

![c60d27b39e85d53932cb8734c2b96b481b0d0f74a90c4ea094f7acae4a0357fe.jpg](../iclr_results/463_CubeDiff_ Repurposing Diffusion-Based Image Models for Panorama Generation/images/c60d27b39e85d53932cb8734c2b96b481b0d0f74a90c4ea094f7acae4a0357fe.jpg)

### Tables

![c87464d93506c0e02fe9772a03cc5e7d8387d6ad2407e9bd3fa166b6f522d472.jpg](../iclr_results/463_CubeDiff_ Repurposing Diffusion-Based Image Models for Panorama Generation/tables/c87464d93506c0e02fe9772a03cc5e7d8387d6ad2407e9bd3fa166b6f522d472.jpg)

## A Periodic Bayesian Flow for Material Generation


### Images

![0991794add8e613fba7bba656dfc4750af1da13242221414dc330a87394b7eab.jpg](../iclr_results/464_A Periodic Bayesian Flow for Material Generation/images/0991794add8e613fba7bba656dfc4750af1da13242221414dc330a87394b7eab.jpg)

![1d74828b78e27d7a69da89dba21fa5b4685c05ef9347c25935086fbf19062394.jpg](../iclr_results/464_A Periodic Bayesian Flow for Material Generation/images/1d74828b78e27d7a69da89dba21fa5b4685c05ef9347c25935086fbf19062394.jpg)

![2389fb71c691a056df6ca5d8568bf954fd933a6b9b85fddefb84202f96260fb7.jpg](../iclr_results/464_A Periodic Bayesian Flow for Material Generation/images/2389fb71c691a056df6ca5d8568bf954fd933a6b9b85fddefb84202f96260fb7.jpg)

![2aebfcfb9c1245ca31e015b512bf668f8cb7cdda34f1323d4ab47738879a97c0.jpg](../iclr_results/464_A Periodic Bayesian Flow for Material Generation/images/2aebfcfb9c1245ca31e015b512bf668f8cb7cdda34f1323d4ab47738879a97c0.jpg)

![2e7c0fa791bf28d7c779efb9fc2abee433ba14a4b776226cfa508e3e1ed4d920.jpg](../iclr_results/464_A Periodic Bayesian Flow for Material Generation/images/2e7c0fa791bf28d7c779efb9fc2abee433ba14a4b776226cfa508e3e1ed4d920.jpg)

![6d34827c52c53552112ca547b6bcbd56a7c37edb0d387f813ffe23b35e21554f.jpg](../iclr_results/464_A Periodic Bayesian Flow for Material Generation/images/6d34827c52c53552112ca547b6bcbd56a7c37edb0d387f813ffe23b35e21554f.jpg)

![8bf93bc7115e4c68e6e80d3bb007a87d5bfbc671c25b6fe9b08a9ee61ac51a50.jpg](../iclr_results/464_A Periodic Bayesian Flow for Material Generation/images/8bf93bc7115e4c68e6e80d3bb007a87d5bfbc671c25b6fe9b08a9ee61ac51a50.jpg)

### Tables

![26cdd7b0bd4ffcec83aa5ff026bcbf38f51118c69846765bbeeb6f7fde91b514.jpg](../iclr_results/464_A Periodic Bayesian Flow for Material Generation/tables/26cdd7b0bd4ffcec83aa5ff026bcbf38f51118c69846765bbeeb6f7fde91b514.jpg)

![28ec5646d84176fbbbe6a721154ed41a9affd3b94e802b043404b43f4e0ae2a1.jpg](../iclr_results/464_A Periodic Bayesian Flow for Material Generation/tables/28ec5646d84176fbbbe6a721154ed41a9affd3b94e802b043404b43f4e0ae2a1.jpg)

![2d54c25ce1943e621a5a05faddbe46240ef7fd9c46d3796c485e64aabaa770fa.jpg](../iclr_results/464_A Periodic Bayesian Flow for Material Generation/tables/2d54c25ce1943e621a5a05faddbe46240ef7fd9c46d3796c485e64aabaa770fa.jpg)

![5df15896ea40c12abb9769497eacf5096b85bbf8e12538cc5fb713e2ccf45bab.jpg](../iclr_results/464_A Periodic Bayesian Flow for Material Generation/tables/5df15896ea40c12abb9769497eacf5096b85bbf8e12538cc5fb713e2ccf45bab.jpg)

![7abf2ca58df5f93524ba16154a71364c39bdab497f37cd3cdb029d2337663f0b.jpg](../iclr_results/464_A Periodic Bayesian Flow for Material Generation/tables/7abf2ca58df5f93524ba16154a71364c39bdab497f37cd3cdb029d2337663f0b.jpg)

![8adf952e0a309f275285efb0127316359810752eb002b38e76a6c8ca5a1a6349.jpg](../iclr_results/464_A Periodic Bayesian Flow for Material Generation/tables/8adf952e0a309f275285efb0127316359810752eb002b38e76a6c8ca5a1a6349.jpg)

![96cae75585ee1b7b04f841a918f11f48a999ef1738f625d84253a42ddc444a2f.jpg](../iclr_results/464_A Periodic Bayesian Flow for Material Generation/tables/96cae75585ee1b7b04f841a918f11f48a999ef1738f625d84253a42ddc444a2f.jpg)

![9db95f2e914d46324b57581a36158fe29f0ea0c7339c569180e4afff1726a393.jpg](../iclr_results/464_A Periodic Bayesian Flow for Material Generation/tables/9db95f2e914d46324b57581a36158fe29f0ea0c7339c569180e4afff1726a393.jpg)

## Generalized Principal-Agent Problem with a Learning Agent


### Tables

![2dc2ec5ba3b5dc00210e2afcf2cf9f53f31dc339321c9519a5724a5a2be2af90.jpg](../iclr_results/465_Generalized Principal-Agent Problem with a Learning Agent/tables/2dc2ec5ba3b5dc00210e2afcf2cf9f53f31dc339321c9519a5724a5a2be2af90.jpg)

![5df994ea8fb4ec9522715f8ebd7550096b95823a3a6162d4a1560d034386d32d.jpg](../iclr_results/465_Generalized Principal-Agent Problem with a Learning Agent/tables/5df994ea8fb4ec9522715f8ebd7550096b95823a3a6162d4a1560d034386d32d.jpg)

![bf22c14e3a20e0267153cf4b517a8f7fccdda6b43dd7b6c71f978beb4002db37.jpg](../iclr_results/465_Generalized Principal-Agent Problem with a Learning Agent/tables/bf22c14e3a20e0267153cf4b517a8f7fccdda6b43dd7b6c71f978beb4002db37.jpg)

![fcc26eedaae8e05ce43f51751216e575972b012925a0d4e82a379115af232cf7.jpg](../iclr_results/465_Generalized Principal-Agent Problem with a Learning Agent/tables/fcc26eedaae8e05ce43f51751216e575972b012925a0d4e82a379115af232cf7.jpg)

## Efficient and Accurate Explanation Estimation with Distribution Compression


### Images

![0c582030ecbd27d83e22343488920346711e416238ead529acddf32ea9620fb2.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/0c582030ecbd27d83e22343488920346711e416238ead529acddf32ea9620fb2.jpg)

![0e779abe241b9d60dcf9f0d711577430e06530aef3de35ddb6a1b172a5c37dac.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/0e779abe241b9d60dcf9f0d711577430e06530aef3de35ddb6a1b172a5c37dac.jpg)

![125f58cd26b53b599438ff2e2619ed9474a47ea2c620ea0b0c4378b928c28bcf.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/125f58cd26b53b599438ff2e2619ed9474a47ea2c620ea0b0c4378b928c28bcf.jpg)

![1ca61967b9db00fd93d5635f812c546a58cf63b6933e7880441979bb1e315b10.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/1ca61967b9db00fd93d5635f812c546a58cf63b6933e7880441979bb1e315b10.jpg)

![2eca01d8dda278f3ea3f9d8c10d51312749a2f740e19eded094cecf978a11a1f.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/2eca01d8dda278f3ea3f9d8c10d51312749a2f740e19eded094cecf978a11a1f.jpg)

![314a51f59be9b73c6f4141e3e2f73d8ec9d90f0f356210b43e9470d31042a6a5.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/314a51f59be9b73c6f4141e3e2f73d8ec9d90f0f356210b43e9470d31042a6a5.jpg)

![325681a592028b6ad31116bdc09409e400ff3fffc203a38c3fa4441c78448b21.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/325681a592028b6ad31116bdc09409e400ff3fffc203a38c3fa4441c78448b21.jpg)

![36681d5b8b3e0388d20c1322c6ddd5578d96dbde984f1609f7ff5e7a004d7855.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/36681d5b8b3e0388d20c1322c6ddd5578d96dbde984f1609f7ff5e7a004d7855.jpg)

![39451278de10952c8f576f15b0adf37ea110cfcf9ad0b45aa6c72ad46af01b35.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/39451278de10952c8f576f15b0adf37ea110cfcf9ad0b45aa6c72ad46af01b35.jpg)

![4fd89284274122809a4f574b0d55ce2038339ec1fd24ca61363d8f020b5d8e61.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/4fd89284274122809a4f574b0d55ce2038339ec1fd24ca61363d8f020b5d8e61.jpg)

![5384602b959c7873b37091f7028a3102e02ed2b4a517af572153fb779bf02c7e.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/5384602b959c7873b37091f7028a3102e02ed2b4a517af572153fb779bf02c7e.jpg)

![5e33d22d5f8a29931773d0585c6ac4c496409b3ef8767fcbfd91533c04abcdb3.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/5e33d22d5f8a29931773d0585c6ac4c496409b3ef8767fcbfd91533c04abcdb3.jpg)

![6233133ae4aff2a74119cbe7b74d6bc7d23f4955ebab4001e98f58dc0a810c46.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/6233133ae4aff2a74119cbe7b74d6bc7d23f4955ebab4001e98f58dc0a810c46.jpg)

![6aa93a429ecc760c049c8b49962543d61e1dea97dbe6bee3467e003620aa035f.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/6aa93a429ecc760c049c8b49962543d61e1dea97dbe6bee3467e003620aa035f.jpg)

![7ade358d088dcf048d88d6324bbdcaf868121ef806dc59ab8e0ba11abbc8c7ec.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/7ade358d088dcf048d88d6324bbdcaf868121ef806dc59ab8e0ba11abbc8c7ec.jpg)

![7d5c0c52f761e5ff7498eb7a0573ce792d7a9967ec96dc7bcf5c98e792ef6bf6.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/7d5c0c52f761e5ff7498eb7a0573ce792d7a9967ec96dc7bcf5c98e792ef6bf6.jpg)

![7ebcc2973b8f0e9722f0563c3625facb35d8ec023aa7c58c2deed1dc9f7e2530.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/7ebcc2973b8f0e9722f0563c3625facb35d8ec023aa7c58c2deed1dc9f7e2530.jpg)

![85ae3d87d813c081fe479664f86c51292d2bbe6041abed2b6c11887de1e0c6ce.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/85ae3d87d813c081fe479664f86c51292d2bbe6041abed2b6c11887de1e0c6ce.jpg)

![90a221c30d2aa0e15daf4a038a0a592f73127e91eeef2372faeaf186d7f814ef.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/90a221c30d2aa0e15daf4a038a0a592f73127e91eeef2372faeaf186d7f814ef.jpg)

![9553f85176cab739a97c06d700832eb6afb256070dd8e4d76809ae2e66ed03f5.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/9553f85176cab739a97c06d700832eb6afb256070dd8e4d76809ae2e66ed03f5.jpg)

![97671404faf67cd9cda750edcbc1002e3bd81f6ae2fa6b51d8d88fe8ca9fece2.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/97671404faf67cd9cda750edcbc1002e3bd81f6ae2fa6b51d8d88fe8ca9fece2.jpg)

![983bc69598a40af63f3d63b42d23306da66643802863529f7bf2fc7bbe0b1fe8.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/983bc69598a40af63f3d63b42d23306da66643802863529f7bf2fc7bbe0b1fe8.jpg)

![9b660757708fb59ac8980323a77dfbf090844b6c6e255b655bb5ec1d074c0618.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/9b660757708fb59ac8980323a77dfbf090844b6c6e255b655bb5ec1d074c0618.jpg)

![a033a35fc350ebb05083762c09e849cb85f28d5afd711a31f10a188e940c770a.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/a033a35fc350ebb05083762c09e849cb85f28d5afd711a31f10a188e940c770a.jpg)

![a45e27a7a254a97882e75080e9b368515a1342e68a86c5b4ef79f2b14323c409.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/a45e27a7a254a97882e75080e9b368515a1342e68a86c5b4ef79f2b14323c409.jpg)

![a860c91b69fc5c28e1cc774434b2cdd22275905e25ce45634cee1df793d2465d.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/a860c91b69fc5c28e1cc774434b2cdd22275905e25ce45634cee1df793d2465d.jpg)

![b0aa9b8d7060c169fe6f40c5d192f4d70608d05a518273f83b2fbbaf9db7e09c.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/b0aa9b8d7060c169fe6f40c5d192f4d70608d05a518273f83b2fbbaf9db7e09c.jpg)

![c4d971178b998242b0003fb0c0d9bf882d618038b93046353482f3f0365041be.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/c4d971178b998242b0003fb0c0d9bf882d618038b93046353482f3f0365041be.jpg)

![cf1d06fdf748b523a32c2ed5511c0ae14ef476bdc1303c6a259ef9455e4991c8.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/cf1d06fdf748b523a32c2ed5511c0ae14ef476bdc1303c6a259ef9455e4991c8.jpg)

![d0539a23e4f879c5035693cc21b7aa2c1a960bb0395831ddaa76affbd0303800.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/d0539a23e4f879c5035693cc21b7aa2c1a960bb0395831ddaa76affbd0303800.jpg)

![d37a212b87ffc53cca1f3c901d20909c84a10359fc2e7a62b2a8e74897926a51.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/d37a212b87ffc53cca1f3c901d20909c84a10359fc2e7a62b2a8e74897926a51.jpg)

![d4c202027eeb20fda80766ee332d13ff31f41ee1ddd8642afe28821a5fd16af0.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/d4c202027eeb20fda80766ee332d13ff31f41ee1ddd8642afe28821a5fd16af0.jpg)

![d75b4f62f055d04fbd6f80f6e59d515a56454edaaf8f8ca69afd3d4b737bd0a8.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/d75b4f62f055d04fbd6f80f6e59d515a56454edaaf8f8ca69afd3d4b737bd0a8.jpg)

![e1f58ce1e2be487ab77681a35347862a2f4852463f8d2d9e5e3cfc32ed96d094.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/e1f58ce1e2be487ab77681a35347862a2f4852463f8d2d9e5e3cfc32ed96d094.jpg)

![e2b08b22d758b1c42562443e80d47ec1f5b4c8bdd4669de33e04401cc8482d60.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/e2b08b22d758b1c42562443e80d47ec1f5b4c8bdd4669de33e04401cc8482d60.jpg)

![e5e2d08f21e8341bf98059ddbc71db7aa531b7bce3f7c8ade964d19db3f693e7.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/e5e2d08f21e8341bf98059ddbc71db7aa531b7bce3f7c8ade964d19db3f693e7.jpg)

![ea290601a04fe9156a09bd207411a78323f48880a1545407a1d8861c3fd62a00.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/ea290601a04fe9156a09bd207411a78323f48880a1545407a1d8861c3fd62a00.jpg)

![ea7d69bf2aedfc8a7204152d0dc20675904e25e01044461345155a13232a1b4a.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/ea7d69bf2aedfc8a7204152d0dc20675904e25e01044461345155a13232a1b4a.jpg)

![f62d06d33e24a22a232a6d3db15927fd5b43495dfa30dd7003b408f274f69f8e.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/images/f62d06d33e24a22a232a6d3db15927fd5b43495dfa30dd7003b408f274f69f8e.jpg)

### Tables

![2c86774eeb7996c1b7f65bf6a5161ad9fcbf13f785fc6182b28ecff9b4cad227.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/tables/2c86774eeb7996c1b7f65bf6a5161ad9fcbf13f785fc6182b28ecff9b4cad227.jpg)

![e3a88d1515789b9e08e0b5b04e646dff90834617c28733c1bbcd761af98d437b.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/tables/e3a88d1515789b9e08e0b5b04e646dff90834617c28733c1bbcd761af98d437b.jpg)

![fe1da882cfef7c81b04b9a01c1efd6d8713ad74d914d4f3677dab9fe08da6b94.jpg](../iclr_results/466_Efficient and Accurate Explanation Estimation with Distribution Compression/tables/fe1da882cfef7c81b04b9a01c1efd6d8713ad74d914d4f3677dab9fe08da6b94.jpg)

## Nonlinear multiregion neural dynamics with parametric impulse response communication channels


### Images

![01205ee07c4cd7fbabac880ac170d55a5d3c24765d2d2d9487c534b61e6ddf51.jpg](../iclr_results/467_Nonlinear multiregion neural dynamics with parametric impulse response communication channels/images/01205ee07c4cd7fbabac880ac170d55a5d3c24765d2d2d9487c534b61e6ddf51.jpg)

![3978416b39ec2459ae000d4521f48a4858f7182947c74afcdd56f2da6b21dffc.jpg](../iclr_results/467_Nonlinear multiregion neural dynamics with parametric impulse response communication channels/images/3978416b39ec2459ae000d4521f48a4858f7182947c74afcdd56f2da6b21dffc.jpg)

![5c2297d5b91c2b8691eea447879eabdc8a200bfe2e2216e257fe62f000212833.jpg](../iclr_results/467_Nonlinear multiregion neural dynamics with parametric impulse response communication channels/images/5c2297d5b91c2b8691eea447879eabdc8a200bfe2e2216e257fe62f000212833.jpg)

![6b4c4d2f82cb90ba338a50c0d5348d7f83daffd48bc1772f27a0d4d5917e34e0.jpg](../iclr_results/467_Nonlinear multiregion neural dynamics with parametric impulse response communication channels/images/6b4c4d2f82cb90ba338a50c0d5348d7f83daffd48bc1772f27a0d4d5917e34e0.jpg)

![90441a74e725835cadf68ff73acc3cb865c9f7e7827881c93d3d2d09e6d1ea9c.jpg](../iclr_results/467_Nonlinear multiregion neural dynamics with parametric impulse response communication channels/images/90441a74e725835cadf68ff73acc3cb865c9f7e7827881c93d3d2d09e6d1ea9c.jpg)

![a3fc2a172041e7dc4dd032d1e6f1b736628a2b225b2d9349b58c722515b8f283.jpg](../iclr_results/467_Nonlinear multiregion neural dynamics with parametric impulse response communication channels/images/a3fc2a172041e7dc4dd032d1e6f1b736628a2b225b2d9349b58c722515b8f283.jpg)

![b02aa63976cae3ee4a9474b597d266c95eee70541a037fd941782ff21c14fce3.jpg](../iclr_results/467_Nonlinear multiregion neural dynamics with parametric impulse response communication channels/images/b02aa63976cae3ee4a9474b597d266c95eee70541a037fd941782ff21c14fce3.jpg)

### Tables

![27c8312f1801d13ca82c6275f515e098e1bb53cee53abbcee6ee610fd530476e.jpg](../iclr_results/467_Nonlinear multiregion neural dynamics with parametric impulse response communication channels/tables/27c8312f1801d13ca82c6275f515e098e1bb53cee53abbcee6ee610fd530476e.jpg)

## POTEC: Off-Policy Contextual Bandits for Large Action Spaces via Policy Decomposition


### Images

![114cf58f719ea55516f19f31d03f7106411ca38799f501606917e7c2b75b46e7.jpg](../iclr_results/468_POTEC_ Off-Policy Contextual Bandits for Large Action Spaces via Policy Decomposition/images/114cf58f719ea55516f19f31d03f7106411ca38799f501606917e7c2b75b46e7.jpg)

![21aec9f24f96c72b6aacbac863418ed63789c686fcedd156a5f4427a9421cb4c.jpg](../iclr_results/468_POTEC_ Off-Policy Contextual Bandits for Large Action Spaces via Policy Decomposition/images/21aec9f24f96c72b6aacbac863418ed63789c686fcedd156a5f4427a9421cb4c.jpg)

![2b48211b9a6df27672bc3e2400aa480fd6492e6ef2b6855668f1d3730194c041.jpg](../iclr_results/468_POTEC_ Off-Policy Contextual Bandits for Large Action Spaces via Policy Decomposition/images/2b48211b9a6df27672bc3e2400aa480fd6492e6ef2b6855668f1d3730194c041.jpg)

![45bef8ebed38170d4ec9cc7a2966f3ef36ba9c85385c09000a6decbcb9f6c14e.jpg](../iclr_results/468_POTEC_ Off-Policy Contextual Bandits for Large Action Spaces via Policy Decomposition/images/45bef8ebed38170d4ec9cc7a2966f3ef36ba9c85385c09000a6decbcb9f6c14e.jpg)

![7799f358b6188942e34a09b9308459e3fd7a2c84650061851557f6dd5349a2a8.jpg](../iclr_results/468_POTEC_ Off-Policy Contextual Bandits for Large Action Spaces via Policy Decomposition/images/7799f358b6188942e34a09b9308459e3fd7a2c84650061851557f6dd5349a2a8.jpg)

![ee1650b9942eded1eb01edeb2e64ae871c1927c0f2067fe42d9879179879ed9c.jpg](../iclr_results/468_POTEC_ Off-Policy Contextual Bandits for Large Action Spaces via Policy Decomposition/images/ee1650b9942eded1eb01edeb2e64ae871c1927c0f2067fe42d9879179879ed9c.jpg)

![f30dd6396da884d1fc82f88852665d81b2d1648053f3679d026739f0791a671f.jpg](../iclr_results/468_POTEC_ Off-Policy Contextual Bandits for Large Action Spaces via Policy Decomposition/images/f30dd6396da884d1fc82f88852665d81b2d1648053f3679d026739f0791a671f.jpg)

### Tables

![09f8b93a9dd312e39b014f95b08a9185fabdf5c92c3c73ddf30ff1ab95443396.jpg](../iclr_results/468_POTEC_ Off-Policy Contextual Bandits for Large Action Spaces via Policy Decomposition/tables/09f8b93a9dd312e39b014f95b08a9185fabdf5c92c3c73ddf30ff1ab95443396.jpg)

![867ae1601bc3a3457746adb11b39e8e92067aca425979df77fb7740489e4ad22.jpg](../iclr_results/468_POTEC_ Off-Policy Contextual Bandits for Large Action Spaces via Policy Decomposition/tables/867ae1601bc3a3457746adb11b39e8e92067aca425979df77fb7740489e4ad22.jpg)

![9289d8e31d5e0d9d1b8ed6883c221648b4483cca5d487b9a1463821a262ac176.jpg](../iclr_results/468_POTEC_ Off-Policy Contextual Bandits for Large Action Spaces via Policy Decomposition/tables/9289d8e31d5e0d9d1b8ed6883c221648b4483cca5d487b9a1463821a262ac176.jpg)

![a30825034071d3815e4f065e70901efcf5fce89ed8f4ceba88c5498217307908.jpg](../iclr_results/468_POTEC_ Off-Policy Contextual Bandits for Large Action Spaces via Policy Decomposition/tables/a30825034071d3815e4f065e70901efcf5fce89ed8f4ceba88c5498217307908.jpg)

![a5882cb7debc759f574e9f97eb84a6d0618218f3f6a8f84da358ca19bab0aa0a.jpg](../iclr_results/468_POTEC_ Off-Policy Contextual Bandits for Large Action Spaces via Policy Decomposition/tables/a5882cb7debc759f574e9f97eb84a6d0618218f3f6a8f84da358ca19bab0aa0a.jpg)

![c6d2aeea780cedbb974ad9a925bb3692b348995d89d6b3ee1acad24cd747c1dd.jpg](../iclr_results/468_POTEC_ Off-Policy Contextual Bandits for Large Action Spaces via Policy Decomposition/tables/c6d2aeea780cedbb974ad9a925bb3692b348995d89d6b3ee1acad24cd747c1dd.jpg)

![e1cde120177f5614888f4a46cd22a6a85c7eb490f6ce67b14944476747416257.jpg](../iclr_results/468_POTEC_ Off-Policy Contextual Bandits for Large Action Spaces via Policy Decomposition/tables/e1cde120177f5614888f4a46cd22a6a85c7eb490f6ce67b14944476747416257.jpg)

## LoRA3D: Low-Rank Self-Calibration of 3D Geometric Foundation models

### Images

![0f3c6995788ab02c273adb5ee7b50bb2f93a4b381725870326992fe6d1135a0c.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/images/0f3c6995788ab02c273adb5ee7b50bb2f93a4b381725870326992fe6d1135a0c.jpg)

![1b82e8b482b4a5a41b43594c9080e1edbf29031bc593d4e71cb9be4dbe74dfe7.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/images/1b82e8b482b4a5a41b43594c9080e1edbf29031bc593d4e71cb9be4dbe74dfe7.jpg)

![26b0ea2a695ce1b0e04d6f8d9928007fa4ad106a0eafb850228bb9b45d28f40e.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/images/26b0ea2a695ce1b0e04d6f8d9928007fa4ad106a0eafb850228bb9b45d28f40e.jpg)

![8714406ba9e940b36966e9abf6001a355ddf62e6e6f185fda983c30c4f38ef34.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/images/8714406ba9e940b36966e9abf6001a355ddf62e6e6f185fda983c30c4f38ef34.jpg)

![88a20f95ab025d6e71971c6d5e40f1b009f68ad0fbcfd0a204d3519265512d78.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/images/88a20f95ab025d6e71971c6d5e40f1b009f68ad0fbcfd0a204d3519265512d78.jpg)

![a871d07c2accc3c36501ef0856a725f04cf24fc3e192b7b3e1a6b233be9c3229.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/images/a871d07c2accc3c36501ef0856a725f04cf24fc3e192b7b3e1a6b233be9c3229.jpg)

![ac69276c29c82e6085f26b3ee48dd2881d5017f936f3f6367aa456bbfbf37d5b.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/images/ac69276c29c82e6085f26b3ee48dd2881d5017f936f3f6367aa456bbfbf37d5b.jpg)

![d1eed78376c861e5780854823a73a8fe5f974e39d98c430e73036d2c6f595c16.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/images/d1eed78376c861e5780854823a73a8fe5f974e39d98c430e73036d2c6f595c16.jpg)

![d88220538dc70d6d6e7d739eed26415efc7306c8d17a899292a665020fc4b7e3.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/images/d88220538dc70d6d6e7d739eed26415efc7306c8d17a899292a665020fc4b7e3.jpg)

![f57a66c4e6b79f14424ebf52fab17d701422ac21e02e4c05fb222ef14afa99e7.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/images/f57a66c4e6b79f14424ebf52fab17d701422ac21e02e4c05fb222ef14afa99e7.jpg)

![f9f6bc14312704b2657433fe11ebbad5b96c62e1ff01849780d736a779b1c748.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/images/f9f6bc14312704b2657433fe11ebbad5b96c62e1ff01849780d736a779b1c748.jpg)

![fc6094f1ee93cd269f6613736941be80842d467ee4d76470d62257c1dcd3fe9b.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/images/fc6094f1ee93cd269f6613736941be80842d467ee4d76470d62257c1dcd3fe9b.jpg)

### Tables

![0816919ce03b2b113c3c2209595f89fe3ce0d71564cf64a4ac311dc50812e608.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/tables/0816919ce03b2b113c3c2209595f89fe3ce0d71564cf64a4ac311dc50812e608.jpg)

![1907df70ef65bec4dfa582232b0b9367132c2c791080a70c1fd1e5e8a03799cf.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/tables/1907df70ef65bec4dfa582232b0b9367132c2c791080a70c1fd1e5e8a03799cf.jpg)

![1d6e661ebb6c6e4d2fefe42c4e78d242190e22f73d0bd58dd198bc35c5a6e645.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/tables/1d6e661ebb6c6e4d2fefe42c4e78d242190e22f73d0bd58dd198bc35c5a6e645.jpg)

![246c6e1fb0e40347439b44f3c96072729e6ce03433fd58a9f4f6f1ed8e9234ab.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/tables/246c6e1fb0e40347439b44f3c96072729e6ce03433fd58a9f4f6f1ed8e9234ab.jpg)

![2e469613ae9c1e9c135168294dcc8dc99b9e6b89580b6bc1b142c0b1757e0222.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/tables/2e469613ae9c1e9c135168294dcc8dc99b9e6b89580b6bc1b142c0b1757e0222.jpg)

![3b73847dea54d6fc77ecfceb5bf4a3419ed8ef6a6bc5660ff1a31917accecc8f.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/tables/3b73847dea54d6fc77ecfceb5bf4a3419ed8ef6a6bc5660ff1a31917accecc8f.jpg)

![70cae2a5e346e9f162dd45ce6abc0fcccb14085a6eef9178b103fd11964fe71f.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/tables/70cae2a5e346e9f162dd45ce6abc0fcccb14085a6eef9178b103fd11964fe71f.jpg)

![7e3d9aa3d5ced2e95a5a30766410e614e426ab33b0c0611ccdd1e6a7ad92a652.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/tables/7e3d9aa3d5ced2e95a5a30766410e614e426ab33b0c0611ccdd1e6a7ad92a652.jpg)

![7f3c95d84ee80044323ee0da4b8c85b6164f3830fba295fb18cc768efeb32210.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/tables/7f3c95d84ee80044323ee0da4b8c85b6164f3830fba295fb18cc768efeb32210.jpg)

![abbb8979576d150a43a56be5e5dd47c0014a3b7e6f266b6e70643716540483dd.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/tables/abbb8979576d150a43a56be5e5dd47c0014a3b7e6f266b6e70643716540483dd.jpg)

![b0e86a111ef31e838069f1a9a8f92b9d7d4b87b5332c09a48b5b17c6abc5e009.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/tables/b0e86a111ef31e838069f1a9a8f92b9d7d4b87b5332c09a48b5b17c6abc5e009.jpg)

![c4b2657c71c6a18845ecbf608b4622869691f2c32420414529277b6ad517075b.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/tables/c4b2657c71c6a18845ecbf608b4622869691f2c32420414529277b6ad517075b.jpg)

![d1870cb0cb080f708f1e0d645ba7cb78fc698a08629101b8549c59cad2cd208a.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/tables/d1870cb0cb080f708f1e0d645ba7cb78fc698a08629101b8549c59cad2cd208a.jpg)

![f9c8bdae688e1d3e7c7681ae9ae2ead8e18c3774101850130ba936df1e55ca15.jpg](../iclr_results/469_LoRA3D_ Low-Rank Self-Calibration of 3D Geometric Foundation models/tables/f9c8bdae688e1d3e7c7681ae9ae2ead8e18c3774101850130ba936df1e55ca15.jpg)
