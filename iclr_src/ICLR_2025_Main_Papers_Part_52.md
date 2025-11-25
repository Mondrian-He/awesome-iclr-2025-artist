# ICLR 2025 Main Conference Papers

**Summary:** 36 papers with extracted content:
- 📊 Total images: 44031
- 📋 Total tables: 33468
- 📄 Total files: 77499

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 52 of 100

## 目录 (Table of Contents)

1. [Diff3DS: Generating View-Consistent 3D Sketch via Differentiable Curve Rendering](#Diff3DS-Generating-View-Consistent-3D-Sketch-via-Differentiable-Curve-Rendering)
2. [Self-Updatable Large Language Models by Integrating Context into Model Parameters](#Self-Updatable-Large-Language-Models-by-Integrating-Context-into-Model-Parameters)
3. [SaMer: A Scenario-aware Multi-dimensional Evaluator for Large Language Models](#SaMer-A-Scenario-aware-Multi-dimensional-Evaluator-for-Large-Language-Models)
4. [Investigating Pattern Neurons in Urban Time Series Forecasting](#Investigating-Pattern-Neurons-in-Urban-Time-Series-Forecasting)
5. [Retrieval Augmented Diffusion Model for Structure-informed Antibody Design and Optimization](#Retrieval-Augmented-Diffusion-Model-for-Structure-informed-Antibody-Design-and-Optimization)
6. [CityGaussianV2: Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes](#CityGaussianV2-Efficient-and-Geometrically-Accurate-Reconstruction-for-Large-Scale-Scenes)
7. [Pangea: A Fully Open Multilingual Multimodal LLM for 39 Languages](#Pangea-A-Fully-Open-Multilingual-Multimodal-LLM-for-39-Languages)
8. [Self-Play Preference Optimization for Language Model Alignment](#Self-Play-Preference-Optimization-for-Language-Model-Alignment)
9. [One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning](#One-for-All-Few-Shot-Anomaly-Detection-via-Instance-Induced-Prompt-Learning)
10. [Rethinking Spiking Neural Networks from an Ensemble Learning Perspective](#Rethinking-Spiking-Neural-Networks-from-an-Ensemble-Learning-Perspective)
11. [DGQ: Distribution-Aware Group Quantization for Text-to-Image Diffusion Models](#DGQ-Distribution-Aware-Group-Quantization-for-Text-to-Image-Diffusion-Models)
12. [TLDR: Token-Level Detective Reward Model for Large Vision Language Models](#TLDR-Token-Level-Detective-Reward-Model-for-Large-Vision-Language-Models)
13. [PvNeXt: Rethinking Network Design and Temporal Motion for Point Cloud Video Recognition](#PvNeXt-Rethinking-Network-Design-and-Temporal-Motion-for-Point-Cloud-Video-Recognition)
14. [Enhancing Language Model Agents using Diversity of Thoughts](#Enhancing-Language-Model-Agents-using-Diversity-of-Thoughts)
15. [SELF-EVOLVED REWARD LEARNING FOR LLMS](#SELF-EVOLVED-REWARD-LEARNING-FOR-LLMS)
16. [Synergy and Diversity in CLIP: Enhancing Performance Through Adaptive Backbone Ensembling](#Synergy-and-Diversity-in-CLIP-Enhancing-Performance-Through-Adaptive-Backbone-Ensembling)
17. [LDAdam: Adaptive Optimization from Low-Dimensional Gradient Statistics](#LDAdam-Adaptive-Optimization-from-Low-Dimensional-Gradient-Statistics)
18. [InstantPortrait: One-Step Portrait Editing via Diffusion Multi-Objective Distillation](#InstantPortrait-One-Step-Portrait-Editing-via-Diffusion-Multi-Objective-Distillation)
19. [What Makes Large Language Models Reason in (Multi-Turn) Code Generation?](#What-Makes-Large-Language-Models-Reason-in-Multi-Turn-Code-Generation)
20. [Generalized Consistency Trajectory Models for Image Manipulation](#Generalized-Consistency-Trajectory-Models-for-Image-Manipulation)
21. [Complexity Lower Bounds of Adaptive Gradient Algorithms for Non-convex Stochastic Optimization under Relaxed Smoothness](#Complexity-Lower-Bounds-of-Adaptive-Gradient-Algorithms-for-Non-convex-Stochastic-Optimization-under-Relaxed-Smoothness)
22. [Federated Few-Shot Class-Incremental Learning](#Federated-Few-Shot-Class-Incremental-Learning)
23. [Theory, Analysis, and Best Practices for Sigmoid Self-Attention](#Theory-Analysis-and-Best-Practices-for-Sigmoid-Self-Attention)
24. [KLay: Accelerating Arithmetic Circuits for Neurosymbolic AI](#KLay-Accelerating-Arithmetic-Circuits-for-Neurosymbolic-AI)
25. [Add-it: Training-Free Object Insertion in Images With Pretrained Diffusion Models](#Add-it-Training-Free-Object-Insertion-in-Images-With-Pretrained-Diffusion-Models)
26. [Step-by-Step Reasoning for Math Problems  via Twisted Sequential Monte Carlo](#Step-by-Step-Reasoning-for-Math-Problems-via-Twisted-Sequential-Monte-Carlo)
27. [ASTrA: Adversarial Self-supervised Training with Adaptive-Attacks](#ASTrA-Adversarial-Self-supervised-Training-with-Adaptive-Attacks)
28. [SpikeLLM: Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking](#SpikeLLM-Scaling-up-Spiking-Neural-Network-to-Large-Language-Models-via-Saliency-based-Spiking)
29. [Actions Speak Louder Than Words: Rate-Reward Trade-off in Markov Decision Processes](#Actions-Speak-Louder-Than-Words-Rate-Reward-Trade-off-in-Markov-Decision-Processes)
30. [Diff-2-in-1: Bridging Generation and Dense Perception with Diffusion Models](#Diff-2-in-1-Bridging-Generation-and-Dense-Perception-with-Diffusion-Models)
31. [Truncated Consistency Models](#Truncated-Consistency-Models)
32. [From Lazy to Rich: Exact Learning Dynamics in Deep Linear Networks](#From-Lazy-to-Rich-Exact-Learning-Dynamics-in-Deep-Linear-Networks)
33. [A Policy-Gradient Approach to Solving Imperfect-Information Games with Best-Iterate Convergence](#A-Policy-Gradient-Approach-to-Solving-Imperfect-Information-Games-with-Best-Iterate-Convergence)
34. [ARB-LLM: Alternating Refined Binarizations for Large Language Models](#ARB-LLM-Alternating-Refined-Binarizations-for-Large-Language-Models)
35. [Radar: Fast Long-Context Decoding for Any Transformer](#Radar-Fast-Long-Context-Decoding-for-Any-Transformer)
36. [Self-Improving Robust Preference Optimization](#Self-Improving-Robust-Preference-Optimization)

---


## Diff3DS: Generating View-Consistent 3D Sketch via Differentiable Curve Rendering

### Images

![00aef4baa1d6e160663689f243df266ad694e13f152090b195101930b3885137.jpg](../iclr_results/1906_Wasserstein-Regularized Conformal Prediction under General Distribution Shift/images/00aef4baa1d6e160663689f243df266ad694e13f152090b195101930b3885137.jpg)

![0951b9b09f5a1be22fdcb0e2f3a3c69811978fad4924b081262fbbe7889e28e6.jpg](../iclr_results/1906_Wasserstein-Regularized Conformal Prediction under General Distribution Shift/images/0951b9b09f5a1be22fdcb0e2f3a3c69811978fad4924b081262fbbe7889e28e6.jpg)

![18c0b24778d2fcc172ced66284fe4a92c432073bee5477132fc72d26b17556ba.jpg](../iclr_results/1906_Wasserstein-Regularized Conformal Prediction under General Distribution Shift/images/18c0b24778d2fcc172ced66284fe4a92c432073bee5477132fc72d26b17556ba.jpg)

![1b283afbe9d7a989f8f651a26a9da44d01a60c6b4ef066096400eaab6555682f.jpg](../iclr_results/1906_Wasserstein-Regularized Conformal Prediction under General Distribution Shift/images/1b283afbe9d7a989f8f651a26a9da44d01a60c6b4ef066096400eaab6555682f.jpg)

![1f89c8414535d345227552589011b8a60e8ff58e849da85cda2d4e28613626a0.jpg](../iclr_results/1906_Wasserstein-Regularized Conformal Prediction under General Distribution Shift/images/1f89c8414535d345227552589011b8a60e8ff58e849da85cda2d4e28613626a0.jpg)

![22dc90e0512a6a8556cd2a8e402743a38d1f2e067eda271485143c53f988f6d9.jpg](../iclr_results/1906_Wasserstein-Regularized Conformal Prediction under General Distribution Shift/images/22dc90e0512a6a8556cd2a8e402743a38d1f2e067eda271485143c53f988f6d9.jpg)

![3101129126bcbf48af3d4b02ebaa96669123e0e59e39b9f30330e3e8566cb7df.jpg](../iclr_results/1906_Wasserstein-Regularized Conformal Prediction under General Distribution Shift/images/3101129126bcbf48af3d4b02ebaa96669123e0e59e39b9f30330e3e8566cb7df.jpg)

![34ab4faa44d9ceaa7be199e80b5ddd04f83936e6880f06ecc47ff682197b459c.jpg](../iclr_results/1906_Wasserstein-Regularized Conformal Prediction under General Distribution Shift/images/34ab4faa44d9ceaa7be199e80b5ddd04f83936e6880f06ecc47ff682197b459c.jpg)

![69e970c9941fa94240c536fc3e139afbfdc453522d04102a0563ccf1f28f145c.jpg](../iclr_results/1906_Wasserstein-Regularized Conformal Prediction under General Distribution Shift/images/69e970c9941fa94240c536fc3e139afbfdc453522d04102a0563ccf1f28f145c.jpg)

![7edd902a57fbd5c5cd37ae401c819cb6ff1b828be941c94c508a21faf417ceec.jpg](../iclr_results/1906_Wasserstein-Regularized Conformal Prediction under General Distribution Shift/images/7edd902a57fbd5c5cd37ae401c819cb6ff1b828be941c94c508a21faf417ceec.jpg)

![8782d595edea322997e5d7b118b0aaec7f60c0714b2fb0213759a69935ed9b9d.jpg](../iclr_results/1906_Wasserstein-Regularized Conformal Prediction under General Distribution Shift/images/8782d595edea322997e5d7b118b0aaec7f60c0714b2fb0213759a69935ed9b9d.jpg)

![8aa414667a55b2f02c32ab5f7b92c9cabb3a226562cdbef02aa95ac4c0e26f92.jpg](../iclr_results/1906_Wasserstein-Regularized Conformal Prediction under General Distribution Shift/images/8aa414667a55b2f02c32ab5f7b92c9cabb3a226562cdbef02aa95ac4c0e26f92.jpg)

![9c07282ba8d7c4783d4ead65bed72e9b8e746445041b0f6e517b7f3b55ed9d14.jpg](../iclr_results/1906_Wasserstein-Regularized Conformal Prediction under General Distribution Shift/images/9c07282ba8d7c4783d4ead65bed72e9b8e746445041b0f6e517b7f3b55ed9d14.jpg)

![ad41de4b10d3d780e410778eb3e14e1d427634b894adc6e2bc606c380d4908e6.jpg](../iclr_results/1906_Wasserstein-Regularized Conformal Prediction under General Distribution Shift/images/ad41de4b10d3d780e410778eb3e14e1d427634b894adc6e2bc606c380d4908e6.jpg)

![ae24c6e085d72bf2d660663a2328e2ed442da745dbe821c50e667e084c54d288.jpg](../iclr_results/1906_Wasserstein-Regularized Conformal Prediction under General Distribution Shift/images/ae24c6e085d72bf2d660663a2328e2ed442da745dbe821c50e667e084c54d288.jpg)

![cbd13aa0a77bc7d1603ae662385da0cb3409c9b3beb89402be3ed76a9e3eb646.jpg](../iclr_results/1906_Wasserstein-Regularized Conformal Prediction under General Distribution Shift/images/cbd13aa0a77bc7d1603ae662385da0cb3409c9b3beb89402be3ed76a9e3eb646.jpg)

![d92c48a78b5165a710f85c0157e9620787a3afa9ecbffa5c439ba0851f721a57.jpg](../iclr_results/1906_Wasserstein-Regularized Conformal Prediction under General Distribution Shift/images/d92c48a78b5165a710f85c0157e9620787a3afa9ecbffa5c439ba0851f721a57.jpg)

![e067af6aa46a4c4a26c2296639bffacc212f60b09cbfd7d9697cfb6fb846326d.jpg](../iclr_results/1906_Wasserstein-Regularized Conformal Prediction under General Distribution Shift/images/e067af6aa46a4c4a26c2296639bffacc212f60b09cbfd7d9697cfb6fb846326d.jpg)

![e3671da47b0f53aa26dc3e4bfea8e25288ba752273ced519211343f11c06476c.jpg](../iclr_results/1906_Wasserstein-Regularized Conformal Prediction under General Distribution Shift/images/e3671da47b0f53aa26dc3e4bfea8e25288ba752273ced519211343f11c06476c.jpg)

![e3f691905a58bc60b879f66e1dac3c951d8aa4d02f20cc2ab7049c20603c514a.jpg](../iclr_results/1906_Wasserstein-Regularized Conformal Prediction under General Distribution Shift/images/e3f691905a58bc60b879f66e1dac3c951d8aa4d02f20cc2ab7049c20603c514a.jpg)

### Tables

![acef3d4304c6b8676090c12898b56e48005beec1dfe2c334f0df7b3d4f823b22.jpg](../iclr_results/1906_Wasserstein-Regularized Conformal Prediction under General Distribution Shift/tables/acef3d4304c6b8676090c12898b56e48005beec1dfe2c334f0df7b3d4f823b22.jpg)

![b41361c40cfcf96472b3f26f3cdb7b40b9266cac93960a211fded8183d4da66d.jpg](../iclr_results/1906_Wasserstein-Regularized Conformal Prediction under General Distribution Shift/tables/b41361c40cfcf96472b3f26f3cdb7b40b9266cac93960a211fded8183d4da66d.jpg)

## Diff3DS: Generating View-Consistent 3D Sketch via Differentiable Curve Rendering


### Images

![05ee018c28789f3d9405f339fd0fe8241731773fb9d0b84d22901443d83118e1.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/05ee018c28789f3d9405f339fd0fe8241731773fb9d0b84d22901443d83118e1.jpg)

![27a7708d10f65b97012c22c4e94a096c9beb92a6e37ed0ef31db52f669936db5.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/27a7708d10f65b97012c22c4e94a096c9beb92a6e37ed0ef31db52f669936db5.jpg)

![2a5a9fedd1dcb73cf878d149b33342f1e05f258db778d92937e39e13998af42b.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/2a5a9fedd1dcb73cf878d149b33342f1e05f258db778d92937e39e13998af42b.jpg)

![3a54e241289ff00976ab2744bbb5f6f15c6821b78d2f88921633c4dd9a621a77.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/3a54e241289ff00976ab2744bbb5f6f15c6821b78d2f88921633c4dd9a621a77.jpg)

![3f01c8cef4c7456d53130e8bba3b65c72be8d3f1711062fc98625e9482dd454d.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/3f01c8cef4c7456d53130e8bba3b65c72be8d3f1711062fc98625e9482dd454d.jpg)

![47c080e4ca16d5b3fa189d09d776352858871f15cbe738959d177908b01dcdc9.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/47c080e4ca16d5b3fa189d09d776352858871f15cbe738959d177908b01dcdc9.jpg)

![584ea1850538288459674c6901fc0f4a6669a94586836df63c6ad78235284756.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/584ea1850538288459674c6901fc0f4a6669a94586836df63c6ad78235284756.jpg)

![5b7758c307595c493d5ed7ecddec82c2f645514878a47c1125bbdb13032686b9.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/5b7758c307595c493d5ed7ecddec82c2f645514878a47c1125bbdb13032686b9.jpg)

![6087960dc88a51a19b640a63af973eaa4624715b9a8adb1231e87e0dd8986976.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/6087960dc88a51a19b640a63af973eaa4624715b9a8adb1231e87e0dd8986976.jpg)

![69cf8c145b9338b66549df63e9fe6b7f6706ffd8daa3180238eb6413b11e0ad7.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/69cf8c145b9338b66549df63e9fe6b7f6706ffd8daa3180238eb6413b11e0ad7.jpg)

![6ad9b4602e457af23659a299283fc57e68e99a40946ea1e6203839c4cf89896f.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/6ad9b4602e457af23659a299283fc57e68e99a40946ea1e6203839c4cf89896f.jpg)

![78a66f58fa806dbd1502a76a307f85a5560dfcc683bf996bb0f1ddd978c8fe9c.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/78a66f58fa806dbd1502a76a307f85a5560dfcc683bf996bb0f1ddd978c8fe9c.jpg)

![83446a83ad8c0293a0d3bfca781ca47c0ffbf828e6110ecbfb113afae6be0c81.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/83446a83ad8c0293a0d3bfca781ca47c0ffbf828e6110ecbfb113afae6be0c81.jpg)

![9dea1066e68bb24bcba83c5e929efbfaa5fedb4495668fd0991e4840a30382cf.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/9dea1066e68bb24bcba83c5e929efbfaa5fedb4495668fd0991e4840a30382cf.jpg)

![aa758070582fb66d0d2ca0d03334caf43d939bb7dd1b4d03c755390c8de6feff.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/aa758070582fb66d0d2ca0d03334caf43d939bb7dd1b4d03c755390c8de6feff.jpg)

![bc36da973a42634b0ae875821b1b4d99619807db00b10ecc1a406087f446b57b.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/bc36da973a42634b0ae875821b1b4d99619807db00b10ecc1a406087f446b57b.jpg)

![c0ca30c3704688a57060ae3ce1170b31ba1fc69710ed667e190643dbe19e0c75.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/c0ca30c3704688a57060ae3ce1170b31ba1fc69710ed667e190643dbe19e0c75.jpg)

![c5642cb7f4a09a8e7c0e3cee3d561a9a78cf7e8c5eb4606dd5f36ff40734b41b.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/c5642cb7f4a09a8e7c0e3cee3d561a9a78cf7e8c5eb4606dd5f36ff40734b41b.jpg)

![c89942a435b40f4979565293383ea3945d66d863c90c1d8e0471185a374e3f71.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/c89942a435b40f4979565293383ea3945d66d863c90c1d8e0471185a374e3f71.jpg)

![d2046cca04120e76494f5e9a9570d8e7768b9c08b25ae38e3154c9623c6e617d.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/d2046cca04120e76494f5e9a9570d8e7768b9c08b25ae38e3154c9623c6e617d.jpg)

![df2590db950cd723b5f5c2cb718b2a2314725f53972e79d297cfa8ae179540d4.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/df2590db950cd723b5f5c2cb718b2a2314725f53972e79d297cfa8ae179540d4.jpg)

![f04b5b54fb6303278d4c2b8196a1b87ab95d919517bf3cb7465650381abd5e1b.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/f04b5b54fb6303278d4c2b8196a1b87ab95d919517bf3cb7465650381abd5e1b.jpg)

![faa1596ac0603f1a22980079a2288995b990feb3bb8536ff4b583a679a6572e8.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/faa1596ac0603f1a22980079a2288995b990feb3bb8536ff4b583a679a6572e8.jpg)

![faf314f0b31558af5d084ffe6f8b5f06c69710260c059bd767f15c6f03b5cddf.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/faf314f0b31558af5d084ffe6f8b5f06c69710260c059bd767f15c6f03b5cddf.jpg)

![fdef6838d462f8fa4e113e828ee9bc67e13f3ab57a4a98ebe5e994a9698b3fca.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/images/fdef6838d462f8fa4e113e828ee9bc67e13f3ab57a4a98ebe5e994a9698b3fca.jpg)

### Tables

![5f0f45bc608fe99316bcd70ee66af4c15b8dae44f5bf4fbc37603ca1b862b8ce.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/tables/5f0f45bc608fe99316bcd70ee66af4c15b8dae44f5bf4fbc37603ca1b862b8ce.jpg)

![971b24e205b73f98ad046ab9a882a38fc82284f59226be021567eb3980d332ac.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/tables/971b24e205b73f98ad046ab9a882a38fc82284f59226be021567eb3980d332ac.jpg)

![feae51f6cd8d5bfff0b60f9c35281a83a88a34d120e904d19d373f515e9c9856.jpg](../iclr_results/1907_Diff3DS_ Generating View-Consistent 3D Sketch via Differentiable Curve Rendering/tables/feae51f6cd8d5bfff0b60f9c35281a83a88a34d120e904d19d373f515e9c9856.jpg)

## Self-Updatable Large Language Models by Integrating Context into Model Parameters


### Images

![04c763f653e882d07772c45586ec3f47deaa62e316c011470f7e8c3d35d25393.jpg](../iclr_results/1908_Self-Updatable Large Language Models by Integrating Context into Model Parameters/images/04c763f653e882d07772c45586ec3f47deaa62e316c011470f7e8c3d35d25393.jpg)

![592fbae40f56c17261a8b7f646ee68a19d05f72c2437d3354ab547dc23523ff7.jpg](../iclr_results/1908_Self-Updatable Large Language Models by Integrating Context into Model Parameters/images/592fbae40f56c17261a8b7f646ee68a19d05f72c2437d3354ab547dc23523ff7.jpg)

![a1cfc4c1e9b559a5199810dd9757358c0e79b2ea6b26d5d85e4cf7430c4edb90.jpg](../iclr_results/1908_Self-Updatable Large Language Models by Integrating Context into Model Parameters/images/a1cfc4c1e9b559a5199810dd9757358c0e79b2ea6b26d5d85e4cf7430c4edb90.jpg)

![b27c4b36e8da1f226350577a076a835e7c8b0be8a5c75df0fecade2dcef21f02.jpg](../iclr_results/1908_Self-Updatable Large Language Models by Integrating Context into Model Parameters/images/b27c4b36e8da1f226350577a076a835e7c8b0be8a5c75df0fecade2dcef21f02.jpg)

![c7916de28ae48593033b5c47a59635610f9cf8cf8dca07618b66fba7b45b1297.jpg](../iclr_results/1908_Self-Updatable Large Language Models by Integrating Context into Model Parameters/images/c7916de28ae48593033b5c47a59635610f9cf8cf8dca07618b66fba7b45b1297.jpg)

### Tables

![1e50ed74923e55b6321932789ae3bdb51581e378c238f43d507f46c9504dc8db.jpg](../iclr_results/1908_Self-Updatable Large Language Models by Integrating Context into Model Parameters/tables/1e50ed74923e55b6321932789ae3bdb51581e378c238f43d507f46c9504dc8db.jpg)

![282126181aa1ca9fc2cd4ce2e7f0576b08a8b049778cef5cb5565da12dd6dc67.jpg](../iclr_results/1908_Self-Updatable Large Language Models by Integrating Context into Model Parameters/tables/282126181aa1ca9fc2cd4ce2e7f0576b08a8b049778cef5cb5565da12dd6dc67.jpg)

![567369428428d1a4612045017ed2c128e8aa7cb2cb375097356c42e8172fe2a9.jpg](../iclr_results/1908_Self-Updatable Large Language Models by Integrating Context into Model Parameters/tables/567369428428d1a4612045017ed2c128e8aa7cb2cb375097356c42e8172fe2a9.jpg)

![bccf5bebf807f5e61097300651ef2aa595ebafb381e4ceaf1cb1fd23a5dfda87.jpg](../iclr_results/1908_Self-Updatable Large Language Models by Integrating Context into Model Parameters/tables/bccf5bebf807f5e61097300651ef2aa595ebafb381e4ceaf1cb1fd23a5dfda87.jpg)

![cbbd24b2ca50474eaca0c0eeeb909f4cf69ffc4fc3b6480d913299362a8294af.jpg](../iclr_results/1908_Self-Updatable Large Language Models by Integrating Context into Model Parameters/tables/cbbd24b2ca50474eaca0c0eeeb909f4cf69ffc4fc3b6480d913299362a8294af.jpg)

![e4ad906d98097af083656a08e226f8e636d1eef25750e8765c710504655325e7.jpg](../iclr_results/1908_Self-Updatable Large Language Models by Integrating Context into Model Parameters/tables/e4ad906d98097af083656a08e226f8e636d1eef25750e8765c710504655325e7.jpg)

## SaMer: A Scenario-aware Multi-dimensional Evaluator for Large Language Models


### Images

![3b755913bb406238fdbed588a5c88e016322da7186c9506c2556248883da4e6f.jpg](../iclr_results/1909_SaMer_ A Scenario-aware Multi-dimensional Evaluator for Large Language Models/images/3b755913bb406238fdbed588a5c88e016322da7186c9506c2556248883da4e6f.jpg)

![9efb557eabc5110e77007f4b2f6cd32b269400dbac2155fbc8172e93afa4763b.jpg](../iclr_results/1909_SaMer_ A Scenario-aware Multi-dimensional Evaluator for Large Language Models/images/9efb557eabc5110e77007f4b2f6cd32b269400dbac2155fbc8172e93afa4763b.jpg)

![c6993ce71c6fb0549c4900954d0b29a0ca27d8c15c8dfc12f87a800e6a3def83.jpg](../iclr_results/1909_SaMer_ A Scenario-aware Multi-dimensional Evaluator for Large Language Models/images/c6993ce71c6fb0549c4900954d0b29a0ca27d8c15c8dfc12f87a800e6a3def83.jpg)

![eac3a04e845922bf7f8b4736ec8717523de7de7078cd7f678b360c982e056eb5.jpg](../iclr_results/1909_SaMer_ A Scenario-aware Multi-dimensional Evaluator for Large Language Models/images/eac3a04e845922bf7f8b4736ec8717523de7de7078cd7f678b360c982e056eb5.jpg)

### Tables

![16729478b434f1a1adada9b56aa88a039c43fc431092c0125448d3285a8d843c.jpg](../iclr_results/1909_SaMer_ A Scenario-aware Multi-dimensional Evaluator for Large Language Models/tables/16729478b434f1a1adada9b56aa88a039c43fc431092c0125448d3285a8d843c.jpg)

![176fd26e13aab47700c22ee2a6fe40a78279692a2c70f568b46263f508d626c6.jpg](../iclr_results/1909_SaMer_ A Scenario-aware Multi-dimensional Evaluator for Large Language Models/tables/176fd26e13aab47700c22ee2a6fe40a78279692a2c70f568b46263f508d626c6.jpg)

![3ee4570a8bf1dcf735809507aa3336547741b1e5afe4e493c85ec09339aad9f1.jpg](../iclr_results/1909_SaMer_ A Scenario-aware Multi-dimensional Evaluator for Large Language Models/tables/3ee4570a8bf1dcf735809507aa3336547741b1e5afe4e493c85ec09339aad9f1.jpg)

![45491a0cd1ccbea876688d8987649a9d99d118511fbcd5a444daa081e9601201.jpg](../iclr_results/1909_SaMer_ A Scenario-aware Multi-dimensional Evaluator for Large Language Models/tables/45491a0cd1ccbea876688d8987649a9d99d118511fbcd5a444daa081e9601201.jpg)

![5311df23f04a6c16e3819331573cd3029071f5ba8c02afc53d9b4715e9c4163c.jpg](../iclr_results/1909_SaMer_ A Scenario-aware Multi-dimensional Evaluator for Large Language Models/tables/5311df23f04a6c16e3819331573cd3029071f5ba8c02afc53d9b4715e9c4163c.jpg)

![69a4abb89127af1195e4781b3cbb52b0c01f9f8b7670a5fff7ff0af6708634ec.jpg](../iclr_results/1909_SaMer_ A Scenario-aware Multi-dimensional Evaluator for Large Language Models/tables/69a4abb89127af1195e4781b3cbb52b0c01f9f8b7670a5fff7ff0af6708634ec.jpg)

![a2e5f006826ca3912d16fe9516c5babf5cb947eec33edc396d38df1fe7406dfe.jpg](../iclr_results/1909_SaMer_ A Scenario-aware Multi-dimensional Evaluator for Large Language Models/tables/a2e5f006826ca3912d16fe9516c5babf5cb947eec33edc396d38df1fe7406dfe.jpg)

![b13904f7aa799af1a007fc48be2a5d812ffca574326f9106697d0645e55797e8.jpg](../iclr_results/1909_SaMer_ A Scenario-aware Multi-dimensional Evaluator for Large Language Models/tables/b13904f7aa799af1a007fc48be2a5d812ffca574326f9106697d0645e55797e8.jpg)

![e5aac3882a8d63cab1e3b82297a08ff39cca15a63b622160c53ae07f7991fffb.jpg](../iclr_results/1909_SaMer_ A Scenario-aware Multi-dimensional Evaluator for Large Language Models/tables/e5aac3882a8d63cab1e3b82297a08ff39cca15a63b622160c53ae07f7991fffb.jpg)

![e918b5e74020d79a0c9a07ff2a35c68ab578315096edb870cf4d07077d028faf.jpg](../iclr_results/1909_SaMer_ A Scenario-aware Multi-dimensional Evaluator for Large Language Models/tables/e918b5e74020d79a0c9a07ff2a35c68ab578315096edb870cf4d07077d028faf.jpg)

## Investigating Pattern Neurons in Urban Time Series Forecasting


### Images

![257b412c2ba69cd39c94671d351a8ee2c7452c2e5141b994d309b17190997863.jpg](../iclr_results/1910_Investigating Pattern Neurons in Urban Time Series Forecasting/images/257b412c2ba69cd39c94671d351a8ee2c7452c2e5141b994d309b17190997863.jpg)

![5005365d0769286d10552442959bfdc00fd047e888c3c684080fa01896d6f1ec.jpg](../iclr_results/1910_Investigating Pattern Neurons in Urban Time Series Forecasting/images/5005365d0769286d10552442959bfdc00fd047e888c3c684080fa01896d6f1ec.jpg)

![5272c5ea711d30545b627cbfb93fd39d084339c004048eab2048656042314552.jpg](../iclr_results/1910_Investigating Pattern Neurons in Urban Time Series Forecasting/images/5272c5ea711d30545b627cbfb93fd39d084339c004048eab2048656042314552.jpg)

![8b7b5904d41622a25549c95d6269ad1ab23fbe4a810f24f6e3758fd988a3a75b.jpg](../iclr_results/1910_Investigating Pattern Neurons in Urban Time Series Forecasting/images/8b7b5904d41622a25549c95d6269ad1ab23fbe4a810f24f6e3758fd988a3a75b.jpg)

![b694145226fb9f9e4de21b39714f37a53944de3882c81a5e059f3ee39737fe86.jpg](../iclr_results/1910_Investigating Pattern Neurons in Urban Time Series Forecasting/images/b694145226fb9f9e4de21b39714f37a53944de3882c81a5e059f3ee39737fe86.jpg)

![bfa9ab3840061104e80fb33eefa529604d7e2577f8e8c554247713da35ec2041.jpg](../iclr_results/1910_Investigating Pattern Neurons in Urban Time Series Forecasting/images/bfa9ab3840061104e80fb33eefa529604d7e2577f8e8c554247713da35ec2041.jpg)

![fd3750597f585fc51659f3f0d2d9abf02fa798d388ff9a520ca97eab6817dc44.jpg](../iclr_results/1910_Investigating Pattern Neurons in Urban Time Series Forecasting/images/fd3750597f585fc51659f3f0d2d9abf02fa798d388ff9a520ca97eab6817dc44.jpg)

### Tables

![1a56a5e1ac8ad87e8e9d97b98d47fa1a8dd2b6e3c6b992f135f2fcc9bf1fccb2.jpg](../iclr_results/1910_Investigating Pattern Neurons in Urban Time Series Forecasting/tables/1a56a5e1ac8ad87e8e9d97b98d47fa1a8dd2b6e3c6b992f135f2fcc9bf1fccb2.jpg)

![3f969e12dd9cdc77508a4d90a0fd6231c5068ab50caa9c753ed95267bea01d70.jpg](../iclr_results/1910_Investigating Pattern Neurons in Urban Time Series Forecasting/tables/3f969e12dd9cdc77508a4d90a0fd6231c5068ab50caa9c753ed95267bea01d70.jpg)

![57d893a7333411d9c38985c48a587e33bf0cf04d59c92e0256fad7983d35d015.jpg](../iclr_results/1910_Investigating Pattern Neurons in Urban Time Series Forecasting/tables/57d893a7333411d9c38985c48a587e33bf0cf04d59c92e0256fad7983d35d015.jpg)

![d093cae3b22aacdb1b1795ad60a7e699a30682b28e0d290f7628e03282af2e09.jpg](../iclr_results/1910_Investigating Pattern Neurons in Urban Time Series Forecasting/tables/d093cae3b22aacdb1b1795ad60a7e699a30682b28e0d290f7628e03282af2e09.jpg)

![d160f7529bfcc03df814d26d990500c42b065dcc75d95e4405fd584b82a48ebc.jpg](../iclr_results/1910_Investigating Pattern Neurons in Urban Time Series Forecasting/tables/d160f7529bfcc03df814d26d990500c42b065dcc75d95e4405fd584b82a48ebc.jpg)

![d31eff45be692e93ab26aa3d3afbd7ee427510c7c9203f3253d6a01109034212.jpg](../iclr_results/1910_Investigating Pattern Neurons in Urban Time Series Forecasting/tables/d31eff45be692e93ab26aa3d3afbd7ee427510c7c9203f3253d6a01109034212.jpg)

![daf549c08b447d2842c758cf45b432d1280b2d564467e5f53ac6f589e316cc7e.jpg](../iclr_results/1910_Investigating Pattern Neurons in Urban Time Series Forecasting/tables/daf549c08b447d2842c758cf45b432d1280b2d564467e5f53ac6f589e316cc7e.jpg)

## Retrieval Augmented Diffusion Model for Structure-informed Antibody Design and Optimization


### Images

![23db07e78dbf0016eb628d9dbd13895bc9379efd2ad8905cd9fa9efe717836e0.jpg](../iclr_results/1911_Retrieval Augmented Diffusion Model for Structure-informed Antibody Design and Optimization/images/23db07e78dbf0016eb628d9dbd13895bc9379efd2ad8905cd9fa9efe717836e0.jpg)

![26d486b8e99ace460e32eee63e0c6af5119fab425ce3ddd2ca65d55cc75c1513.jpg](../iclr_results/1911_Retrieval Augmented Diffusion Model for Structure-informed Antibody Design and Optimization/images/26d486b8e99ace460e32eee63e0c6af5119fab425ce3ddd2ca65d55cc75c1513.jpg)

![3f92530c48d2602ef1776a86e3ffdb84d5dab3a23e60a85c493a24ffe2d8839a.jpg](../iclr_results/1911_Retrieval Augmented Diffusion Model for Structure-informed Antibody Design and Optimization/images/3f92530c48d2602ef1776a86e3ffdb84d5dab3a23e60a85c493a24ffe2d8839a.jpg)

![4050e71e77d63bcdd323edb5d4b7fd7b42be468169ac851e79d30f9117937ae9.jpg](../iclr_results/1911_Retrieval Augmented Diffusion Model for Structure-informed Antibody Design and Optimization/images/4050e71e77d63bcdd323edb5d4b7fd7b42be468169ac851e79d30f9117937ae9.jpg)

![a933c0baa75fca4492cd25470c96183539e51ef69743338d2c16af00c6c961fd.jpg](../iclr_results/1911_Retrieval Augmented Diffusion Model for Structure-informed Antibody Design and Optimization/images/a933c0baa75fca4492cd25470c96183539e51ef69743338d2c16af00c6c961fd.jpg)

![d9be19e2e59b16f30dcf058faa3882a4902b3d65bdfc277d5fb3abee7def1bc2.jpg](../iclr_results/1911_Retrieval Augmented Diffusion Model for Structure-informed Antibody Design and Optimization/images/d9be19e2e59b16f30dcf058faa3882a4902b3d65bdfc277d5fb3abee7def1bc2.jpg)

### Tables

![11921259a02c9a89f4d6ac26010da2ff3fb2acb2132323d92a7980aac3252f3e.jpg](../iclr_results/1911_Retrieval Augmented Diffusion Model for Structure-informed Antibody Design and Optimization/tables/11921259a02c9a89f4d6ac26010da2ff3fb2acb2132323d92a7980aac3252f3e.jpg)

![43edc143460a2a46c58863f88bea4521692eb1a841a765573fabdefefef104b2.jpg](../iclr_results/1911_Retrieval Augmented Diffusion Model for Structure-informed Antibody Design and Optimization/tables/43edc143460a2a46c58863f88bea4521692eb1a841a765573fabdefefef104b2.jpg)

![601f96616834e6182d1756f00794583efc2f58701dfcbe9720649b7293b5d0fa.jpg](../iclr_results/1911_Retrieval Augmented Diffusion Model for Structure-informed Antibody Design and Optimization/tables/601f96616834e6182d1756f00794583efc2f58701dfcbe9720649b7293b5d0fa.jpg)

![7b8284883fb95a0c53f59961802cc22c92af47846fab8dae3186381e34c7ac98.jpg](../iclr_results/1911_Retrieval Augmented Diffusion Model for Structure-informed Antibody Design and Optimization/tables/7b8284883fb95a0c53f59961802cc22c92af47846fab8dae3186381e34c7ac98.jpg)

![a160cba3f3a148950fcb029af9450fb5f367df3aaa9e9d4ac40be92f8a7d46e9.jpg](../iclr_results/1911_Retrieval Augmented Diffusion Model for Structure-informed Antibody Design and Optimization/tables/a160cba3f3a148950fcb029af9450fb5f367df3aaa9e9d4ac40be92f8a7d46e9.jpg)

## CityGaussianV2: Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes


### Images

![11a158d2c7b2bbcaf1bc2472dd9f512e5c249ef920aaf488468aa75107882cf0.jpg](../iclr_results/1912_CityGaussianV2_ Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes/images/11a158d2c7b2bbcaf1bc2472dd9f512e5c249ef920aaf488468aa75107882cf0.jpg)

![1526de7a7ced2f5cbb4fc133a72cb7d8ca84384a067b8c717ec2f21786c82b8f.jpg](../iclr_results/1912_CityGaussianV2_ Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes/images/1526de7a7ced2f5cbb4fc133a72cb7d8ca84384a067b8c717ec2f21786c82b8f.jpg)

![15eb5c18d806f071129e456657fa5458a99bfe702d3e0d9e50d1e8e91b376861.jpg](../iclr_results/1912_CityGaussianV2_ Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes/images/15eb5c18d806f071129e456657fa5458a99bfe702d3e0d9e50d1e8e91b376861.jpg)

![1e43941760adc633bb08116df1b4b6bdc4e6ee3d95b172394bb52c1b9faf2a5e.jpg](../iclr_results/1912_CityGaussianV2_ Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes/images/1e43941760adc633bb08116df1b4b6bdc4e6ee3d95b172394bb52c1b9faf2a5e.jpg)

![29a5b611d801fee2dd2e67f91f85942d9ad346017856d94c1209919041af1ba2.jpg](../iclr_results/1912_CityGaussianV2_ Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes/images/29a5b611d801fee2dd2e67f91f85942d9ad346017856d94c1209919041af1ba2.jpg)

![4ff2fd6118778179a4d081180910729dffc6d3e6fc9e8c821efa61a701800563.jpg](../iclr_results/1912_CityGaussianV2_ Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes/images/4ff2fd6118778179a4d081180910729dffc6d3e6fc9e8c821efa61a701800563.jpg)

![8846ca018df43de347383ac2086c5f692f12c57b33ffb91e664c1d0c76d53135.jpg](../iclr_results/1912_CityGaussianV2_ Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes/images/8846ca018df43de347383ac2086c5f692f12c57b33ffb91e664c1d0c76d53135.jpg)

![a212ac8f34ae2b2f29e13dfff00dcdd5345ff10c9f69c93cafcf397008325b02.jpg](../iclr_results/1912_CityGaussianV2_ Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes/images/a212ac8f34ae2b2f29e13dfff00dcdd5345ff10c9f69c93cafcf397008325b02.jpg)

![b0417f9ce52afb86499a4f7d5d4405a4c30c48b67d7e510326a274504167971c.jpg](../iclr_results/1912_CityGaussianV2_ Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes/images/b0417f9ce52afb86499a4f7d5d4405a4c30c48b67d7e510326a274504167971c.jpg)

![c5032b79f1719988864c251b1ded5df58730db44ead46c7ffb8bcc1b4d7eb6be.jpg](../iclr_results/1912_CityGaussianV2_ Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes/images/c5032b79f1719988864c251b1ded5df58730db44ead46c7ffb8bcc1b4d7eb6be.jpg)

![fbf5c8c7575f5fced4f96aa94b272502b1254850ac5a6c8239a4ed4fe842ce81.jpg](../iclr_results/1912_CityGaussianV2_ Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes/images/fbf5c8c7575f5fced4f96aa94b272502b1254850ac5a6c8239a4ed4fe842ce81.jpg)

### Tables

![051a06e255155a7a1fc08768df2b84edaee2740a8d75e23e508062f49159a677.jpg](../iclr_results/1912_CityGaussianV2_ Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes/tables/051a06e255155a7a1fc08768df2b84edaee2740a8d75e23e508062f49159a677.jpg)

![2794cda75c4761587ec55037d06880cb2c595fe51169cc97076d56af98b9811f.jpg](../iclr_results/1912_CityGaussianV2_ Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes/tables/2794cda75c4761587ec55037d06880cb2c595fe51169cc97076d56af98b9811f.jpg)

![3f90f527d7182293d32b247fa35c74f2618b75c00e7a6f82ae9b89ca71089dfe.jpg](../iclr_results/1912_CityGaussianV2_ Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes/tables/3f90f527d7182293d32b247fa35c74f2618b75c00e7a6f82ae9b89ca71089dfe.jpg)

![4fac49130456b5d14a9993ed131c36e5eaef2314628a97e03bb3e16beadfc911.jpg](../iclr_results/1912_CityGaussianV2_ Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes/tables/4fac49130456b5d14a9993ed131c36e5eaef2314628a97e03bb3e16beadfc911.jpg)

![527b8fa86f0fd3ba319159a4abfceb501653eb5b7d5b0fb9cd2b1862aafc5a2b.jpg](../iclr_results/1912_CityGaussianV2_ Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes/tables/527b8fa86f0fd3ba319159a4abfceb501653eb5b7d5b0fb9cd2b1862aafc5a2b.jpg)

![5aa6f722e2fa1c94815f0a7eb23a3fde7f71cf966c0dde1b2d5feb30ba193b28.jpg](../iclr_results/1912_CityGaussianV2_ Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes/tables/5aa6f722e2fa1c94815f0a7eb23a3fde7f71cf966c0dde1b2d5feb30ba193b28.jpg)

![f34dfa950c52b7a77b04da17380fc366864fe3757c3bd2904f73f94b1a07386f.jpg](../iclr_results/1912_CityGaussianV2_ Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes/tables/f34dfa950c52b7a77b04da17380fc366864fe3757c3bd2904f73f94b1a07386f.jpg)

## Pangea: A Fully Open Multilingual Multimodal LLM for 39 Languages


### Images

![1646d3131702f61c153bc4479cfc4ddd9a652d5a903b1725b49f8ebcfe05880a.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/images/1646d3131702f61c153bc4479cfc4ddd9a652d5a903b1725b49f8ebcfe05880a.jpg)

![31d28201bec82c915a8a485b31ba4ed6e358fcbb36c1f9541cd970ae3134075c.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/images/31d28201bec82c915a8a485b31ba4ed6e358fcbb36c1f9541cd970ae3134075c.jpg)

![3657aa1812fa0d88c910737a46b0ec8554b1047ae9e8659f12a1e8fa9b8930bc.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/images/3657aa1812fa0d88c910737a46b0ec8554b1047ae9e8659f12a1e8fa9b8930bc.jpg)

![38e8113d370e0dfc5fc18aa17860b91c424ffc595ee7f8681d14642a6ebf2593.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/images/38e8113d370e0dfc5fc18aa17860b91c424ffc595ee7f8681d14642a6ebf2593.jpg)

![3eb819b5a793c77d8a3065944113d2807e79177bebe7859c64dc72230b1d6593.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/images/3eb819b5a793c77d8a3065944113d2807e79177bebe7859c64dc72230b1d6593.jpg)

![45c78f9f06810b8f7c15e6148a7be4ff83c3e209eb24234300fca7a43044aa9c.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/images/45c78f9f06810b8f7c15e6148a7be4ff83c3e209eb24234300fca7a43044aa9c.jpg)

![4e3dc2c9685ce48f130414c70565baa963da7709bb494b8e3e7b25b9c2403743.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/images/4e3dc2c9685ce48f130414c70565baa963da7709bb494b8e3e7b25b9c2403743.jpg)

![538d4f1ee6a22f79f5fd60f25f7c4e603a6e683e057452fed282be651aaa332b.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/images/538d4f1ee6a22f79f5fd60f25f7c4e603a6e683e057452fed282be651aaa332b.jpg)

![69cb6db4291cf0d6fb44a3ba0845332460b7f31141d224191a85a11099eff1b7.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/images/69cb6db4291cf0d6fb44a3ba0845332460b7f31141d224191a85a11099eff1b7.jpg)

![6ebb573e456973a59a5a9b7cbbcefbf005a6c9a73f7703b4b274c16493e9f4cf.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/images/6ebb573e456973a59a5a9b7cbbcefbf005a6c9a73f7703b4b274c16493e9f4cf.jpg)

![82bab287395571b41f0bc2fd83c40bb4ac729b0d09b140f9fa1b55a6e520c214.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/images/82bab287395571b41f0bc2fd83c40bb4ac729b0d09b140f9fa1b55a6e520c214.jpg)

![83eea8c6e1d95c1eadf5e34915cef0ad20a0f6e497aa03b9adee777bc101f7c0.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/images/83eea8c6e1d95c1eadf5e34915cef0ad20a0f6e497aa03b9adee777bc101f7c0.jpg)

![89e1e266e16a936ea3ad26b8e7bb54a458191ddf68523b4336a5d5cdb2440ca3.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/images/89e1e266e16a936ea3ad26b8e7bb54a458191ddf68523b4336a5d5cdb2440ca3.jpg)

![8eedc948ae6a8eacb9875b697be448a01fad26a141b9220894f75093a688fc27.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/images/8eedc948ae6a8eacb9875b697be448a01fad26a141b9220894f75093a688fc27.jpg)

![a4b5f8b38c41da7c87e148aa1a7c1343539c05df1a566deec2657114db1bf28f.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/images/a4b5f8b38c41da7c87e148aa1a7c1343539c05df1a566deec2657114db1bf28f.jpg)

![a7309cc1f12b87e779fc9fe2c2bf6b9c49ab7f4fd2c7a11ce5da0575c085b581.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/images/a7309cc1f12b87e779fc9fe2c2bf6b9c49ab7f4fd2c7a11ce5da0575c085b581.jpg)

![ba25f523287ad6bf1997944474f12b29445ea97a0aaeabea77b42229995e3799.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/images/ba25f523287ad6bf1997944474f12b29445ea97a0aaeabea77b42229995e3799.jpg)

![ba88dd60716d15aa04e8633f0f4d73669b8aabf76be2e0282170d9b82e3d840a.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/images/ba88dd60716d15aa04e8633f0f4d73669b8aabf76be2e0282170d9b82e3d840a.jpg)

![bcbdabcedd20adc710e8d60d07a9abbaa5d158a79b8f15f2a52bb28119912f55.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/images/bcbdabcedd20adc710e8d60d07a9abbaa5d158a79b8f15f2a52bb28119912f55.jpg)

![cc68aee109b81b3d4bc66b52359c7e9002b300b2686f836d82e4636aa0d38beb.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/images/cc68aee109b81b3d4bc66b52359c7e9002b300b2686f836d82e4636aa0d38beb.jpg)

![d2561dba80d333e7ee6dab48655d46f2d147ee5e4e206357b01921bf11c25947.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/images/d2561dba80d333e7ee6dab48655d46f2d147ee5e4e206357b01921bf11c25947.jpg)

![ecbc1481a13a1ab377e2903e694aad28383d3d0f231c0df0fe04443db3c724ee.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/images/ecbc1481a13a1ab377e2903e694aad28383d3d0f231c0df0fe04443db3c724ee.jpg)

![fb48b99585cbd94116a1f32478f29886f518904a326de5cd2cbfaab7bf05d34e.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/images/fb48b99585cbd94116a1f32478f29886f518904a326de5cd2cbfaab7bf05d34e.jpg)

### Tables

![18fcddea882b7a9f152a707da31dc72168df2222de9cbd8e75e4e7b98bd28a9f.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/tables/18fcddea882b7a9f152a707da31dc72168df2222de9cbd8e75e4e7b98bd28a9f.jpg)

![2cb01e7875e57149e861170ea502a096f8019359aa81fe113481a3f2b929e6c9.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/tables/2cb01e7875e57149e861170ea502a096f8019359aa81fe113481a3f2b929e6c9.jpg)

![303e05da95241fae950e7b8707d0ed7861d2342ead252537e6636b7d3a7a3068.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/tables/303e05da95241fae950e7b8707d0ed7861d2342ead252537e6636b7d3a7a3068.jpg)

![4672e9e7625e0ec34c36a5f34e43d6e417efb9bbbf81aecc5de5b28a9b7957f2.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/tables/4672e9e7625e0ec34c36a5f34e43d6e417efb9bbbf81aecc5de5b28a9b7957f2.jpg)

![48fd96defee148ff8eea72cf61d981fce1d7a43f715d1457a784aa5ebe2c594f.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/tables/48fd96defee148ff8eea72cf61d981fce1d7a43f715d1457a784aa5ebe2c594f.jpg)

![51560e047ad13968ae966f24fd6be9c6537b0f3995770cd9701f1b6dfb47f83b.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/tables/51560e047ad13968ae966f24fd6be9c6537b0f3995770cd9701f1b6dfb47f83b.jpg)

![52e4d06d6b4b96232f6244cc0a32b3f2ab775b972c87f03ed25b2d1a4cd413ea.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/tables/52e4d06d6b4b96232f6244cc0a32b3f2ab775b972c87f03ed25b2d1a4cd413ea.jpg)

![550887a2b599c80ec8f2bb4a5388d6e3e1dc8b1dc104abf528e6e002c4278320.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/tables/550887a2b599c80ec8f2bb4a5388d6e3e1dc8b1dc104abf528e6e002c4278320.jpg)

![563b4097e8b24d47b2c7b0b9b31a34884adb68a03b5effa3ffceaf879268b621.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/tables/563b4097e8b24d47b2c7b0b9b31a34884adb68a03b5effa3ffceaf879268b621.jpg)

![5642a7a43c9fb711c8df17da8fee9f7dbd41a451531894f2ab2edef24ad38b61.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/tables/5642a7a43c9fb711c8df17da8fee9f7dbd41a451531894f2ab2edef24ad38b61.jpg)

![611e36d3d2491bcfdc8ecaf0c436284f77cb5b73dde04045f8889c4ae70a023f.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/tables/611e36d3d2491bcfdc8ecaf0c436284f77cb5b73dde04045f8889c4ae70a023f.jpg)

![7a4fb89093268ee1fe32a5aae38fb3652c06055e92e3ff597c6722ec27a7e7d9.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/tables/7a4fb89093268ee1fe32a5aae38fb3652c06055e92e3ff597c6722ec27a7e7d9.jpg)

![8b8dfb69dd324f4214e025020b0ac6bacafc335d534c2cc305dbea1db9ce379c.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/tables/8b8dfb69dd324f4214e025020b0ac6bacafc335d534c2cc305dbea1db9ce379c.jpg)

![91e756d7b4eaeb3c61c6e994485a8be22aadaf03430eab2a7e50ff13b4fe2160.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/tables/91e756d7b4eaeb3c61c6e994485a8be22aadaf03430eab2a7e50ff13b4fe2160.jpg)

![a27fb5375cc7771ced65bf9741b198050133437e2fd90fa3172f0dd35a47bce0.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/tables/a27fb5375cc7771ced65bf9741b198050133437e2fd90fa3172f0dd35a47bce0.jpg)

![ac0159e5fc62b6442a5bb7dd04c0d9a2725d59a9b6c30285a10195ff9533bd3f.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/tables/ac0159e5fc62b6442a5bb7dd04c0d9a2725d59a9b6c30285a10195ff9533bd3f.jpg)

![b4720c6fdd4f1d18c65838dd14f7395e14328d4f489d0411228c83b08f3b2d97.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/tables/b4720c6fdd4f1d18c65838dd14f7395e14328d4f489d0411228c83b08f3b2d97.jpg)

![bf2b4909fb5ea3eafed0dd4b8ab810f70e1532e6b3da95433fb44627dad8c80a.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/tables/bf2b4909fb5ea3eafed0dd4b8ab810f70e1532e6b3da95433fb44627dad8c80a.jpg)

![eb6f682dcb10d3c4332c5bd4f87cf3333888d7644ff6179c36b6c87b7cb47afa.jpg](../iclr_results/1913_Pangea_ A Fully Open Multilingual Multimodal LLM for 39 Languages/tables/eb6f682dcb10d3c4332c5bd4f87cf3333888d7644ff6179c36b6c87b7cb47afa.jpg)

## Self-Play Preference Optimization for Language Model Alignment


### Images

![037d2a149cfccefb9591332aaa29325c12fee5e0008e9c10071b17829d8427f0.jpg](../iclr_results/1914_Self-Play Preference Optimization for Language Model Alignment/images/037d2a149cfccefb9591332aaa29325c12fee5e0008e9c10071b17829d8427f0.jpg)

![3ac9bab5f507156874584d93528e5e18d833630b029b57b52dbd16f9661ec46a.jpg](../iclr_results/1914_Self-Play Preference Optimization for Language Model Alignment/images/3ac9bab5f507156874584d93528e5e18d833630b029b57b52dbd16f9661ec46a.jpg)

![4efc6d766de710f084a305bf539f42706339e3eff16e8b95c887895faa7e7acb.jpg](../iclr_results/1914_Self-Play Preference Optimization for Language Model Alignment/images/4efc6d766de710f084a305bf539f42706339e3eff16e8b95c887895faa7e7acb.jpg)

![78af9ad205f486ab069ca72bc3c14532c6b500e050016baa0faadd1c9f817b20.jpg](../iclr_results/1914_Self-Play Preference Optimization for Language Model Alignment/images/78af9ad205f486ab069ca72bc3c14532c6b500e050016baa0faadd1c9f817b20.jpg)

### Tables

![193eb5dc7c8f61e34920b8bde98eceffa48c85250910f2b951873d30b31e1ebb.jpg](../iclr_results/1914_Self-Play Preference Optimization for Language Model Alignment/tables/193eb5dc7c8f61e34920b8bde98eceffa48c85250910f2b951873d30b31e1ebb.jpg)

![67aeeb58cd809401cabf02b22438a5a0b68b1a564cb9c22d51129c793b3cfcbc.jpg](../iclr_results/1914_Self-Play Preference Optimization for Language Model Alignment/tables/67aeeb58cd809401cabf02b22438a5a0b68b1a564cb9c22d51129c793b3cfcbc.jpg)

![80408c504f424d33d4f66b0fa49764018e1c9ae0dcdf5606fc0944427bd500e6.jpg](../iclr_results/1914_Self-Play Preference Optimization for Language Model Alignment/tables/80408c504f424d33d4f66b0fa49764018e1c9ae0dcdf5606fc0944427bd500e6.jpg)

![869ee866316c3a24ccd6ffda8a52ae635d4b32c51ad99c89940325d570e1a2ff.jpg](../iclr_results/1914_Self-Play Preference Optimization for Language Model Alignment/tables/869ee866316c3a24ccd6ffda8a52ae635d4b32c51ad99c89940325d570e1a2ff.jpg)

![b814c28bdb6cd8196dac80609e6fe237f34ab8785058541812da56bc3cee61bf.jpg](../iclr_results/1914_Self-Play Preference Optimization for Language Model Alignment/tables/b814c28bdb6cd8196dac80609e6fe237f34ab8785058541812da56bc3cee61bf.jpg)

![dcdf39c3e3a1b0e08b8cbecd8df6ecb9d1224941893538f882117207a573edec.jpg](../iclr_results/1914_Self-Play Preference Optimization for Language Model Alignment/tables/dcdf39c3e3a1b0e08b8cbecd8df6ecb9d1224941893538f882117207a573edec.jpg)

## One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning


### Images

![39c16345e6a6810da8faa08d0cf05b5d003e1c66af174f1b1b49c967ded22aa0.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/images/39c16345e6a6810da8faa08d0cf05b5d003e1c66af174f1b1b49c967ded22aa0.jpg)

![805fe7fff26043e033b9b844bda7f222cc0689b1969f0cd85ca60e98226d4c1c.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/images/805fe7fff26043e033b9b844bda7f222cc0689b1969f0cd85ca60e98226d4c1c.jpg)

![8321a7f2a532f7764143bee831b65e25ec958f4b4f7fa2c8735364da9744536e.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/images/8321a7f2a532f7764143bee831b65e25ec958f4b4f7fa2c8735364da9744536e.jpg)

![f671c39025e46efe6e5f65f0dad20af8bee902d94b5416e0df51d6079b2eb531.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/images/f671c39025e46efe6e5f65f0dad20af8bee902d94b5416e0df51d6079b2eb531.jpg)

### Tables

![042485073158c43f1e5e93b6d81a2954eb9b57c3554b7e027e10767813bdaf23.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/tables/042485073158c43f1e5e93b6d81a2954eb9b57c3554b7e027e10767813bdaf23.jpg)

![2f48d1fc0741d3fa7d897005b05405a1d384174897eca75de3d826e1e14bf5ab.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/tables/2f48d1fc0741d3fa7d897005b05405a1d384174897eca75de3d826e1e14bf5ab.jpg)

![35423525b0bc5a16a2ab800900733a25f59811f47808f88a3b7c25b6f01d8912.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/tables/35423525b0bc5a16a2ab800900733a25f59811f47808f88a3b7c25b6f01d8912.jpg)

![391dac5702909ff77f81af4a1df9d058f937fe9e16180956f40bbe7a2e5a15f5.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/tables/391dac5702909ff77f81af4a1df9d058f937fe9e16180956f40bbe7a2e5a15f5.jpg)

![3922e67d3ca4559188d68d453ec5c234e0172d1b5cdf2327859bf1ea390bd4d3.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/tables/3922e67d3ca4559188d68d453ec5c234e0172d1b5cdf2327859bf1ea390bd4d3.jpg)

![548435893ed1f9e3412bf58530a7596ffa896b155aec606be367e073db703167.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/tables/548435893ed1f9e3412bf58530a7596ffa896b155aec606be367e073db703167.jpg)

![6593dd0180efdb8253309e5eaca76b3037e6ebaf219b21c23becc51e89a9502c.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/tables/6593dd0180efdb8253309e5eaca76b3037e6ebaf219b21c23becc51e89a9502c.jpg)

![6d899102373f4cbd66dbbc200079cbb34fbd143b40adfa92a5e92ad606987a34.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/tables/6d899102373f4cbd66dbbc200079cbb34fbd143b40adfa92a5e92ad606987a34.jpg)

![71c8334ae3353fa6889ad138cb1e76b16fdd43233101df722e26d27f22c4e857.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/tables/71c8334ae3353fa6889ad138cb1e76b16fdd43233101df722e26d27f22c4e857.jpg)

![8f118b73cdbbaab20ab2fdb1dc87dbef0a95e5e8ef6b7917f5926d73561d16e2.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/tables/8f118b73cdbbaab20ab2fdb1dc87dbef0a95e5e8ef6b7917f5926d73561d16e2.jpg)

![a43540d4b2450340e3341f0be47ffbe9eae3084423a85a012345ab9c6bfe0e0f.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/tables/a43540d4b2450340e3341f0be47ffbe9eae3084423a85a012345ab9c6bfe0e0f.jpg)

![a4840cfd4318574175f5602878c9000a8eecf93df87862e60929bd44b857fb04.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/tables/a4840cfd4318574175f5602878c9000a8eecf93df87862e60929bd44b857fb04.jpg)

![afe0e0f5ddc2ee40db81dcba420bdeebdee031ed9891960b7f293c6e113b7405.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/tables/afe0e0f5ddc2ee40db81dcba420bdeebdee031ed9891960b7f293c6e113b7405.jpg)

![b536651d6d251188e5d659ba900eed09b21c56e8a773f2167e6de705645f585f.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/tables/b536651d6d251188e5d659ba900eed09b21c56e8a773f2167e6de705645f585f.jpg)

![b85bd2a5dc60001aaef4a546f129274ab9ccae72808e3fcf1d9bc9d1152735d3.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/tables/b85bd2a5dc60001aaef4a546f129274ab9ccae72808e3fcf1d9bc9d1152735d3.jpg)

![ccbad461de2b533e6d590babffcf2d0cee9f47b9bf31ba0b0e71791afdbbb8ae.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/tables/ccbad461de2b533e6d590babffcf2d0cee9f47b9bf31ba0b0e71791afdbbb8ae.jpg)

![d09c506a42cfd4a81443ae10fbe98b4c73b078a9767b36490f9f2e3c0869bba7.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/tables/d09c506a42cfd4a81443ae10fbe98b4c73b078a9767b36490f9f2e3c0869bba7.jpg)

![dae3ed868081d4217e8b49da97dc7ca1fe125355e37bc4331c4f59c3de48d06d.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/tables/dae3ed868081d4217e8b49da97dc7ca1fe125355e37bc4331c4f59c3de48d06d.jpg)

![f366b350c867a1efa58d2b6be74b4bc2fff998a13ae72f6ebe2eb1a2005fc028.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/tables/f366b350c867a1efa58d2b6be74b4bc2fff998a13ae72f6ebe2eb1a2005fc028.jpg)

![fb9dcbee496fa901faa4d2f06c508569ae26f929817be80b1b56cf4a31817f15.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/tables/fb9dcbee496fa901faa4d2f06c508569ae26f929817be80b1b56cf4a31817f15.jpg)

![ffde720c148fe0c6bf74a4846c2a64452dde0f72986db7f74e58f00d6f29173b.jpg](../iclr_results/1915_One-for-All Few-Shot Anomaly Detection via Instance-Induced Prompt Learning/tables/ffde720c148fe0c6bf74a4846c2a64452dde0f72986db7f74e58f00d6f29173b.jpg)

## Rethinking Spiking Neural Networks from an Ensemble Learning Perspective


### Images

![32dcc54335b04bce9c764576f21d55a14731f3a117cfefb28dd384e1d7ad4f7d.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/images/32dcc54335b04bce9c764576f21d55a14731f3a117cfefb28dd384e1d7ad4f7d.jpg)

![5deb02a56bb9e08f34f1bab0a23a9a60b37d8e4cf528c0d91897489fc05ae78f.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/images/5deb02a56bb9e08f34f1bab0a23a9a60b37d8e4cf528c0d91897489fc05ae78f.jpg)

![6d26a74289e33552a3beaa92cffd97609d69d7fbac4ae639768c92c816e7b595.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/images/6d26a74289e33552a3beaa92cffd97609d69d7fbac4ae639768c92c816e7b595.jpg)

![7264c8b5a6e4ca4e1583b517184a6db58cf594d8bbe3e8515437b943dae2459b.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/images/7264c8b5a6e4ca4e1583b517184a6db58cf594d8bbe3e8515437b943dae2459b.jpg)

![7316559de1905d6f48a3f70bea2084d014f65097c1491f9f9b7e191b8be5f226.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/images/7316559de1905d6f48a3f70bea2084d014f65097c1491f9f9b7e191b8be5f226.jpg)

![819dce6dc6db07f3a2b64afa6772b90a6a8f8305ae37a8f78f0eaec03dd0fae6.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/images/819dce6dc6db07f3a2b64afa6772b90a6a8f8305ae37a8f78f0eaec03dd0fae6.jpg)

![a97aeeb8688123b8061b21b36c1a70d64be4f65bc28afea4891a8313f7031280.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/images/a97aeeb8688123b8061b21b36c1a70d64be4f65bc28afea4891a8313f7031280.jpg)

![c4a625d73c74bd6418b9cacb485ced1fa9d3a0fee44c660b704df31394000ae2.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/images/c4a625d73c74bd6418b9cacb485ced1fa9d3a0fee44c660b704df31394000ae2.jpg)

![c9f99277c79b5511ee52993c389f07731c396b323ae5d2a3cd86296bb15d646f.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/images/c9f99277c79b5511ee52993c389f07731c396b323ae5d2a3cd86296bb15d646f.jpg)

### Tables

![06b6ada713664180d3b0f2c5b77ce63fc4534e09180d355f42e73c96b9178708.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/tables/06b6ada713664180d3b0f2c5b77ce63fc4534e09180d355f42e73c96b9178708.jpg)

![12d5ebf7266abb0d09bf92d67396a76692e8008085754e74c563cfb4a2a9d569.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/tables/12d5ebf7266abb0d09bf92d67396a76692e8008085754e74c563cfb4a2a9d569.jpg)

![2390d7edf0601cc2fc3db1fc97be3a045e362ceae292e0b1a8a040c158fa82ea.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/tables/2390d7edf0601cc2fc3db1fc97be3a045e362ceae292e0b1a8a040c158fa82ea.jpg)

![324255fc237b8c2bc683c14d2cc5cfd830501345bd877322e9b1b94f1af3c789.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/tables/324255fc237b8c2bc683c14d2cc5cfd830501345bd877322e9b1b94f1af3c789.jpg)

![5dd5d3cae6ba4e9a058bdf4ad5ca97f67863d35eb2a0998a0a3e8f16028f35f6.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/tables/5dd5d3cae6ba4e9a058bdf4ad5ca97f67863d35eb2a0998a0a3e8f16028f35f6.jpg)

![610031744041bba42d2dc9572593bf19d11d48373fd3c65f57e05fb7ed6c2b62.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/tables/610031744041bba42d2dc9572593bf19d11d48373fd3c65f57e05fb7ed6c2b62.jpg)

![79399630c5a0f0a0683d8655fbc8d28b744181783126a742dcdef26483a6f956.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/tables/79399630c5a0f0a0683d8655fbc8d28b744181783126a742dcdef26483a6f956.jpg)

![7cbf40d74ced0a7345d1d1592f155f658bcbabbd773244774dd5ad9707a14530.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/tables/7cbf40d74ced0a7345d1d1592f155f658bcbabbd773244774dd5ad9707a14530.jpg)

![81dd3bfb9812159b39a811074c0100b46a35d260e9ed4118f595c49b80e5b93d.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/tables/81dd3bfb9812159b39a811074c0100b46a35d260e9ed4118f595c49b80e5b93d.jpg)

![8c5fc1fae89e0be816c2c568537b56115a981691b579627e92d118a203cb80c1.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/tables/8c5fc1fae89e0be816c2c568537b56115a981691b579627e92d118a203cb80c1.jpg)

![91db4a3f531c1a9953b3b257e385b4d6bf859d3adcb37eab31a8b134997adb90.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/tables/91db4a3f531c1a9953b3b257e385b4d6bf859d3adcb37eab31a8b134997adb90.jpg)

![96fbd8716921e2e5a3bd0ddba90b6f740abe018292412714932fcfb553e43d60.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/tables/96fbd8716921e2e5a3bd0ddba90b6f740abe018292412714932fcfb553e43d60.jpg)

![adcb00a9db02b5ca25735dcb0719f0affb7fc65a71be5a1262ad17259ee7b6fe.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/tables/adcb00a9db02b5ca25735dcb0719f0affb7fc65a71be5a1262ad17259ee7b6fe.jpg)

![c4ca4621a60c71aecf13ee673302474d2b70eb8daf75a3f9b5d1c83c5b161dcc.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/tables/c4ca4621a60c71aecf13ee673302474d2b70eb8daf75a3f9b5d1c83c5b161dcc.jpg)

![db63807095e30f5cd95b3fa978c1db874ead61f6b402758c09d7322cc1d19a1a.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/tables/db63807095e30f5cd95b3fa978c1db874ead61f6b402758c09d7322cc1d19a1a.jpg)

![e6a06d309c7ebf7211a7d11b801653ba72bf5d26ed1448acad10116162a058e3.jpg](../iclr_results/1916_Rethinking Spiking Neural Networks from an Ensemble Learning Perspective/tables/e6a06d309c7ebf7211a7d11b801653ba72bf5d26ed1448acad10116162a058e3.jpg)

## DGQ: Distribution-Aware Group Quantization for Text-to-Image Diffusion Models


### Images

![23d409dfeb71c01e7606ca12fe536a352a55763514b5addc9df4f30018c0beac.jpg](../iclr_results/1917_DGQ_ Distribution-Aware Group Quantization for Text-to-Image Diffusion Models/images/23d409dfeb71c01e7606ca12fe536a352a55763514b5addc9df4f30018c0beac.jpg)

![2793d58c591e313242a48c76c8b7f9a67fa9b874c229f8cd49284ed950ecbb76.jpg](../iclr_results/1917_DGQ_ Distribution-Aware Group Quantization for Text-to-Image Diffusion Models/images/2793d58c591e313242a48c76c8b7f9a67fa9b874c229f8cd49284ed950ecbb76.jpg)

![389ffec08722a46f98697ccebc1f5171aabd779f2e761f6f62dfa4a919f0232a.jpg](../iclr_results/1917_DGQ_ Distribution-Aware Group Quantization for Text-to-Image Diffusion Models/images/389ffec08722a46f98697ccebc1f5171aabd779f2e761f6f62dfa4a919f0232a.jpg)

![420c5fd3603c3ad587dd2108c9f7a60df8f3b8bc7c15b552a76c4e7a2fa23530.jpg](../iclr_results/1917_DGQ_ Distribution-Aware Group Quantization for Text-to-Image Diffusion Models/images/420c5fd3603c3ad587dd2108c9f7a60df8f3b8bc7c15b552a76c4e7a2fa23530.jpg)

![44ba25ca214db9c9ffe8566c4b85aeae69ad8214f2ca3d7bbdad4a3fc7448c7b.jpg](../iclr_results/1917_DGQ_ Distribution-Aware Group Quantization for Text-to-Image Diffusion Models/images/44ba25ca214db9c9ffe8566c4b85aeae69ad8214f2ca3d7bbdad4a3fc7448c7b.jpg)

![4dbe34acc2e1d6648a3a87731e5cee7f875cf34ab6affedb63efcd8d8f641a6e.jpg](../iclr_results/1917_DGQ_ Distribution-Aware Group Quantization for Text-to-Image Diffusion Models/images/4dbe34acc2e1d6648a3a87731e5cee7f875cf34ab6affedb63efcd8d8f641a6e.jpg)

![61ba62f5fa5c2d0fdcb9e08603dd7ee4773d3347fd5e45630eb28cbd49bac763.jpg](../iclr_results/1917_DGQ_ Distribution-Aware Group Quantization for Text-to-Image Diffusion Models/images/61ba62f5fa5c2d0fdcb9e08603dd7ee4773d3347fd5e45630eb28cbd49bac763.jpg)

![81a432e8992662f74637344d073c1f37616717e891a3e3b11c5aaa04573ca6bf.jpg](../iclr_results/1917_DGQ_ Distribution-Aware Group Quantization for Text-to-Image Diffusion Models/images/81a432e8992662f74637344d073c1f37616717e891a3e3b11c5aaa04573ca6bf.jpg)

![acc0ccb3a0934f7831a557edcd7eff45ef93128d40ccbfbc9c4e8ac7e3d090d6.jpg](../iclr_results/1917_DGQ_ Distribution-Aware Group Quantization for Text-to-Image Diffusion Models/images/acc0ccb3a0934f7831a557edcd7eff45ef93128d40ccbfbc9c4e8ac7e3d090d6.jpg)

![b4a6625744b183db4788ff6632f794713da94932be7691c84eb8dd79251ab672.jpg](../iclr_results/1917_DGQ_ Distribution-Aware Group Quantization for Text-to-Image Diffusion Models/images/b4a6625744b183db4788ff6632f794713da94932be7691c84eb8dd79251ab672.jpg)

![d5b6423f14929b7aee959abc4ad67c4055d17875b7e29420e9904283dafb7afb.jpg](../iclr_results/1917_DGQ_ Distribution-Aware Group Quantization for Text-to-Image Diffusion Models/images/d5b6423f14929b7aee959abc4ad67c4055d17875b7e29420e9904283dafb7afb.jpg)

![dc782b7e84602382325d294960da857c27f146fb25ed703d3350c5168db3a89b.jpg](../iclr_results/1917_DGQ_ Distribution-Aware Group Quantization for Text-to-Image Diffusion Models/images/dc782b7e84602382325d294960da857c27f146fb25ed703d3350c5168db3a89b.jpg)

![f43fffd58d69850294fdc488b2252fc1b9ec01cd3b138c17eae4e6e20e96cd94.jpg](../iclr_results/1917_DGQ_ Distribution-Aware Group Quantization for Text-to-Image Diffusion Models/images/f43fffd58d69850294fdc488b2252fc1b9ec01cd3b138c17eae4e6e20e96cd94.jpg)

![f44d1b7debfa564011c9e67fd5810ffefd9d6727581c099aa70f346138162167.jpg](../iclr_results/1917_DGQ_ Distribution-Aware Group Quantization for Text-to-Image Diffusion Models/images/f44d1b7debfa564011c9e67fd5810ffefd9d6727581c099aa70f346138162167.jpg)

### Tables

![136661feba51f34f1f702e5ba67407e2625646534638faf66b8e1c7926257e96.jpg](../iclr_results/1917_DGQ_ Distribution-Aware Group Quantization for Text-to-Image Diffusion Models/tables/136661feba51f34f1f702e5ba67407e2625646534638faf66b8e1c7926257e96.jpg)

![4e892ddd9aef83f4e2efbaff1466d38a302de0709bf146c7f9c4b0123c19d181.jpg](../iclr_results/1917_DGQ_ Distribution-Aware Group Quantization for Text-to-Image Diffusion Models/tables/4e892ddd9aef83f4e2efbaff1466d38a302de0709bf146c7f9c4b0123c19d181.jpg)

![904ab4170576e14e5534a6b9e23e72b96403c2467b6f059d2f59be9b8a97aabe.jpg](../iclr_results/1917_DGQ_ Distribution-Aware Group Quantization for Text-to-Image Diffusion Models/tables/904ab4170576e14e5534a6b9e23e72b96403c2467b6f059d2f59be9b8a97aabe.jpg)

![a3b156e2a395b5755cb8acd2bfe4271c417301760aa66064f1b3d00dc61c785c.jpg](../iclr_results/1917_DGQ_ Distribution-Aware Group Quantization for Text-to-Image Diffusion Models/tables/a3b156e2a395b5755cb8acd2bfe4271c417301760aa66064f1b3d00dc61c785c.jpg)

![aa2976df798a8ccd177870e1505074d09075fe779bd33c56213972c8963161ef.jpg](../iclr_results/1917_DGQ_ Distribution-Aware Group Quantization for Text-to-Image Diffusion Models/tables/aa2976df798a8ccd177870e1505074d09075fe779bd33c56213972c8963161ef.jpg)

![b03424c72144efb9c4f446c598534b69edf9889674c04c643483a54c02e93c5f.jpg](../iclr_results/1917_DGQ_ Distribution-Aware Group Quantization for Text-to-Image Diffusion Models/tables/b03424c72144efb9c4f446c598534b69edf9889674c04c643483a54c02e93c5f.jpg)

![c6de1750f6fcd43c5b56ddd4f17f78d1f06054dacb7fba9671478d5664be78af.jpg](../iclr_results/1917_DGQ_ Distribution-Aware Group Quantization for Text-to-Image Diffusion Models/tables/c6de1750f6fcd43c5b56ddd4f17f78d1f06054dacb7fba9671478d5664be78af.jpg)

![e44573dca1b78d20210f1252e22e73c3e8b877539758d7007b306daefd33ccd9.jpg](../iclr_results/1917_DGQ_ Distribution-Aware Group Quantization for Text-to-Image Diffusion Models/tables/e44573dca1b78d20210f1252e22e73c3e8b877539758d7007b306daefd33ccd9.jpg)

## TLDR: Token-Level Detective Reward Model for Large Vision Language Models


### Images

![153843dfcc05ca950357bbde10ca4fe36b288e2ec0c7950756579de841f9b396.jpg](../iclr_results/1918_TLDR_ Token-Level Detective Reward Model for Large Vision Language Models/images/153843dfcc05ca950357bbde10ca4fe36b288e2ec0c7950756579de841f9b396.jpg)

![1e6bf71ab341038aa8f62620cfed932c9d62d68301043af8ee9b6444161525c3.jpg](../iclr_results/1918_TLDR_ Token-Level Detective Reward Model for Large Vision Language Models/images/1e6bf71ab341038aa8f62620cfed932c9d62d68301043af8ee9b6444161525c3.jpg)

![3e561b452132c3d3e585334a686e62afd3e717fa25d5168c30196e9e94b06109.jpg](../iclr_results/1918_TLDR_ Token-Level Detective Reward Model for Large Vision Language Models/images/3e561b452132c3d3e585334a686e62afd3e717fa25d5168c30196e9e94b06109.jpg)

![6446db2afc44997e5aeef6e97ac6535b0d08e03315aafdce3f645dd7a302a8ab.jpg](../iclr_results/1918_TLDR_ Token-Level Detective Reward Model for Large Vision Language Models/images/6446db2afc44997e5aeef6e97ac6535b0d08e03315aafdce3f645dd7a302a8ab.jpg)

![93802cef728b5ea4b3dcabba72420e6ba41c3ede40968662762abdda4c914191.jpg](../iclr_results/1918_TLDR_ Token-Level Detective Reward Model for Large Vision Language Models/images/93802cef728b5ea4b3dcabba72420e6ba41c3ede40968662762abdda4c914191.jpg)

![b76f0b5030d7d0f659aa985ddc2c2ab3c9944cfc9afd793d08153465bde6c2b0.jpg](../iclr_results/1918_TLDR_ Token-Level Detective Reward Model for Large Vision Language Models/images/b76f0b5030d7d0f659aa985ddc2c2ab3c9944cfc9afd793d08153465bde6c2b0.jpg)

![e65f09de1e9321ab967aebd948f9119b8ad703deacf5a65faf90cfccbd82806e.jpg](../iclr_results/1918_TLDR_ Token-Level Detective Reward Model for Large Vision Language Models/images/e65f09de1e9321ab967aebd948f9119b8ad703deacf5a65faf90cfccbd82806e.jpg)

![eba800428a926ce50f6afad2df3c67ab76e7dd6a64a79103f116ee998da3677d.jpg](../iclr_results/1918_TLDR_ Token-Level Detective Reward Model for Large Vision Language Models/images/eba800428a926ce50f6afad2df3c67ab76e7dd6a64a79103f116ee998da3677d.jpg)

### Tables

![089a0063188120aa24b92a33fbeb34fe31630544e96db99fc80908d88da1c59c.jpg](../iclr_results/1918_TLDR_ Token-Level Detective Reward Model for Large Vision Language Models/tables/089a0063188120aa24b92a33fbeb34fe31630544e96db99fc80908d88da1c59c.jpg)

![129fe4697ca29dd13fd08f54a977d1a77c76d598f31819675d63e2569373aafa.jpg](../iclr_results/1918_TLDR_ Token-Level Detective Reward Model for Large Vision Language Models/tables/129fe4697ca29dd13fd08f54a977d1a77c76d598f31819675d63e2569373aafa.jpg)

![1540fdc6563d5498a2e3f89a171053ae0455febef75b87da3725ce78291f9ef8.jpg](../iclr_results/1918_TLDR_ Token-Level Detective Reward Model for Large Vision Language Models/tables/1540fdc6563d5498a2e3f89a171053ae0455febef75b87da3725ce78291f9ef8.jpg)

![2d15020be12044d0e08e482e9fee9648fac094340b7a0fe81558e7dd03dfdbfc.jpg](../iclr_results/1918_TLDR_ Token-Level Detective Reward Model for Large Vision Language Models/tables/2d15020be12044d0e08e482e9fee9648fac094340b7a0fe81558e7dd03dfdbfc.jpg)

![83b2c0c4dc9faa78f4598c345ac5bd66dad3e97dd9b95d5f0537c1cfc9253552.jpg](../iclr_results/1918_TLDR_ Token-Level Detective Reward Model for Large Vision Language Models/tables/83b2c0c4dc9faa78f4598c345ac5bd66dad3e97dd9b95d5f0537c1cfc9253552.jpg)

![ace396ace1c77a57fae2bf1b257074738d91ed2e90554743a403eb469a55a253.jpg](../iclr_results/1918_TLDR_ Token-Level Detective Reward Model for Large Vision Language Models/tables/ace396ace1c77a57fae2bf1b257074738d91ed2e90554743a403eb469a55a253.jpg)

![b34c7388d9f387c13aa15fc42bd89e2527f29f32916aecb91aa1aaac9bcdb65d.jpg](../iclr_results/1918_TLDR_ Token-Level Detective Reward Model for Large Vision Language Models/tables/b34c7388d9f387c13aa15fc42bd89e2527f29f32916aecb91aa1aaac9bcdb65d.jpg)

![b431a56aa697d86bdc727674bca181e38ca4fc86bffa76cc2b7f38bfaea73eb0.jpg](../iclr_results/1918_TLDR_ Token-Level Detective Reward Model for Large Vision Language Models/tables/b431a56aa697d86bdc727674bca181e38ca4fc86bffa76cc2b7f38bfaea73eb0.jpg)

![d657ddbe18a62067cf9ac30895e52e0dc9ecff6c229b821bcbbc67fb255dc5f4.jpg](../iclr_results/1918_TLDR_ Token-Level Detective Reward Model for Large Vision Language Models/tables/d657ddbe18a62067cf9ac30895e52e0dc9ecff6c229b821bcbbc67fb255dc5f4.jpg)

![e064c67e4d65f2671653c8847197c9e9da85c1060e5cbb71585ee654a9a6f769.jpg](../iclr_results/1918_TLDR_ Token-Level Detective Reward Model for Large Vision Language Models/tables/e064c67e4d65f2671653c8847197c9e9da85c1060e5cbb71585ee654a9a6f769.jpg)

![f78b648b800d8e990b4a632ea00436bcda4ae334eded3f3d83d24134ad942856.jpg](../iclr_results/1918_TLDR_ Token-Level Detective Reward Model for Large Vision Language Models/tables/f78b648b800d8e990b4a632ea00436bcda4ae334eded3f3d83d24134ad942856.jpg)

## PvNeXt: Rethinking Network Design and Temporal Motion for Point Cloud Video Recognition


### Images

![48826780058a5beaf5792f04f761e5f644fd3ec8579ed2e76a399c2a3d29d50e.jpg](../iclr_results/1919_PvNeXt_ Rethinking Network Design and Temporal Motion for Point Cloud Video Recognition/images/48826780058a5beaf5792f04f761e5f644fd3ec8579ed2e76a399c2a3d29d50e.jpg)

![7d0d5af20df24995ba35e54a5f924891277c76aea2e5426ec31f673a289e180b.jpg](../iclr_results/1919_PvNeXt_ Rethinking Network Design and Temporal Motion for Point Cloud Video Recognition/images/7d0d5af20df24995ba35e54a5f924891277c76aea2e5426ec31f673a289e180b.jpg)

![9d115347505335ab387a858f8cfdaaf7a4d51cc7f70e55c39524bfd02688a476.jpg](../iclr_results/1919_PvNeXt_ Rethinking Network Design and Temporal Motion for Point Cloud Video Recognition/images/9d115347505335ab387a858f8cfdaaf7a4d51cc7f70e55c39524bfd02688a476.jpg)

![f7887c10f93a62bf1748d03570ad110656cc349e82926ea16877e1f43a5da912.jpg](../iclr_results/1919_PvNeXt_ Rethinking Network Design and Temporal Motion for Point Cloud Video Recognition/images/f7887c10f93a62bf1748d03570ad110656cc349e82926ea16877e1f43a5da912.jpg)

### Tables

![10cc324bc271fd3e0c711061f54d6143fb13ce8517f16d96a4fe92e93bbc5e8b.jpg](../iclr_results/1919_PvNeXt_ Rethinking Network Design and Temporal Motion for Point Cloud Video Recognition/tables/10cc324bc271fd3e0c711061f54d6143fb13ce8517f16d96a4fe92e93bbc5e8b.jpg)

![1c5e9290c58945b6dcae5f42e2dd9e7293727c39eb3a6b9157c843d185ef3554.jpg](../iclr_results/1919_PvNeXt_ Rethinking Network Design and Temporal Motion for Point Cloud Video Recognition/tables/1c5e9290c58945b6dcae5f42e2dd9e7293727c39eb3a6b9157c843d185ef3554.jpg)

![3dd88915de0053b77b84d8c97e64943bc45dcfece5339c8619965313de52b8bf.jpg](../iclr_results/1919_PvNeXt_ Rethinking Network Design and Temporal Motion for Point Cloud Video Recognition/tables/3dd88915de0053b77b84d8c97e64943bc45dcfece5339c8619965313de52b8bf.jpg)

![8aa6ec49b9358c736187b3a05b690b7db4d3a1ccff08581f4b78dcb34ef83893.jpg](../iclr_results/1919_PvNeXt_ Rethinking Network Design and Temporal Motion for Point Cloud Video Recognition/tables/8aa6ec49b9358c736187b3a05b690b7db4d3a1ccff08581f4b78dcb34ef83893.jpg)

![92b0f1918c51dc2ae22164e2628c924a815ef171b3ebf2ddf92888dda0ce94f4.jpg](../iclr_results/1919_PvNeXt_ Rethinking Network Design and Temporal Motion for Point Cloud Video Recognition/tables/92b0f1918c51dc2ae22164e2628c924a815ef171b3ebf2ddf92888dda0ce94f4.jpg)

![97801f2f9727783e6a89dc078f7adec92f4211b47c41ff21c97f5421db0bdc5d.jpg](../iclr_results/1919_PvNeXt_ Rethinking Network Design and Temporal Motion for Point Cloud Video Recognition/tables/97801f2f9727783e6a89dc078f7adec92f4211b47c41ff21c97f5421db0bdc5d.jpg)

![97c14e9c17d318aa0401d50cc6286e110308a63a8e6b6c632be2ad735c354241.jpg](../iclr_results/1919_PvNeXt_ Rethinking Network Design and Temporal Motion for Point Cloud Video Recognition/tables/97c14e9c17d318aa0401d50cc6286e110308a63a8e6b6c632be2ad735c354241.jpg)

![a4964edc45de8abd52a5e3fd1c12214008352f66b3c3745d1d786546074196d8.jpg](../iclr_results/1919_PvNeXt_ Rethinking Network Design and Temporal Motion for Point Cloud Video Recognition/tables/a4964edc45de8abd52a5e3fd1c12214008352f66b3c3745d1d786546074196d8.jpg)

![b13d84cd307cc7460627670f569c1d64c9abd3bf3af70f5dffdf293913585056.jpg](../iclr_results/1919_PvNeXt_ Rethinking Network Design and Temporal Motion for Point Cloud Video Recognition/tables/b13d84cd307cc7460627670f569c1d64c9abd3bf3af70f5dffdf293913585056.jpg)

![c6dc2187b1023cfdf14bcad9dd4db9a0b336a8e72b0f1a8a0ac8d9a80d87d8d8.jpg](../iclr_results/1919_PvNeXt_ Rethinking Network Design and Temporal Motion for Point Cloud Video Recognition/tables/c6dc2187b1023cfdf14bcad9dd4db9a0b336a8e72b0f1a8a0ac8d9a80d87d8d8.jpg)

![cc25562f91025040fd7925c30d16f309da88a3b0f5a2e21a17944f0ac43dc425.jpg](../iclr_results/1919_PvNeXt_ Rethinking Network Design and Temporal Motion for Point Cloud Video Recognition/tables/cc25562f91025040fd7925c30d16f309da88a3b0f5a2e21a17944f0ac43dc425.jpg)

![e0aa9dfce796bb13caafda08af2156de2518ca3749a28f184b22c016f405a0b4.jpg](../iclr_results/1919_PvNeXt_ Rethinking Network Design and Temporal Motion for Point Cloud Video Recognition/tables/e0aa9dfce796bb13caafda08af2156de2518ca3749a28f184b22c016f405a0b4.jpg)

![e8ead1291a18cdb79182970ecf9bc1f4974bb90765467b2ca38a5be39a6770d8.jpg](../iclr_results/1919_PvNeXt_ Rethinking Network Design and Temporal Motion for Point Cloud Video Recognition/tables/e8ead1291a18cdb79182970ecf9bc1f4974bb90765467b2ca38a5be39a6770d8.jpg)

## Enhancing Language Model Agents using Diversity of Thoughts


### Images

![257eec13f2e1f5679775d4fdf5ff50bb9115f9e9c35912f6353b3c4c5d59dbad.jpg](../iclr_results/1920_Enhancing Language Model Agents using Diversity of Thoughts/images/257eec13f2e1f5679775d4fdf5ff50bb9115f9e9c35912f6353b3c4c5d59dbad.jpg)

![45682beb7deb4723ea9de5775700ead13cfd476a33dfcbd7dacf1dc723aa5a64.jpg](../iclr_results/1920_Enhancing Language Model Agents using Diversity of Thoughts/images/45682beb7deb4723ea9de5775700ead13cfd476a33dfcbd7dacf1dc723aa5a64.jpg)

![e63ba814796bcf41f2b66dddd9b55bcd8eea1b1b6a53590fae05dcc209e59401.jpg](../iclr_results/1920_Enhancing Language Model Agents using Diversity of Thoughts/images/e63ba814796bcf41f2b66dddd9b55bcd8eea1b1b6a53590fae05dcc209e59401.jpg)

![e81af45a2204bd81addbd48b8fbbd228495ab58a0e09a5df47f179a5da212fdc.jpg](../iclr_results/1920_Enhancing Language Model Agents using Diversity of Thoughts/images/e81af45a2204bd81addbd48b8fbbd228495ab58a0e09a5df47f179a5da212fdc.jpg)

### Tables

![0fb046e6f7003395b1685004ae86ef327c96dd3ff6cbbdb7c3624c63815d188e.jpg](../iclr_results/1920_Enhancing Language Model Agents using Diversity of Thoughts/tables/0fb046e6f7003395b1685004ae86ef327c96dd3ff6cbbdb7c3624c63815d188e.jpg)

![123682c6efdf1737a63903cd51e8bf1725efd5771feeeebde4710ee09b8fb9c0.jpg](../iclr_results/1920_Enhancing Language Model Agents using Diversity of Thoughts/tables/123682c6efdf1737a63903cd51e8bf1725efd5771feeeebde4710ee09b8fb9c0.jpg)

![3cd71212471a22799226c19f079ba2840ef469c3ddb82a1d1df39aa4a52858b6.jpg](../iclr_results/1920_Enhancing Language Model Agents using Diversity of Thoughts/tables/3cd71212471a22799226c19f079ba2840ef469c3ddb82a1d1df39aa4a52858b6.jpg)

![3f974c3e2237cbe1f81c2a540ba57ec5a250b11889feb4391f8285d4d3a6490e.jpg](../iclr_results/1920_Enhancing Language Model Agents using Diversity of Thoughts/tables/3f974c3e2237cbe1f81c2a540ba57ec5a250b11889feb4391f8285d4d3a6490e.jpg)

![404c46dfa80725868e75f02cf19d226d330ef412f99ab543e88348cd39f19f18.jpg](../iclr_results/1920_Enhancing Language Model Agents using Diversity of Thoughts/tables/404c46dfa80725868e75f02cf19d226d330ef412f99ab543e88348cd39f19f18.jpg)

![50f70ed7c3c983893ab034e99d4118b1e6743823d6eb096abc493554c592c31d.jpg](../iclr_results/1920_Enhancing Language Model Agents using Diversity of Thoughts/tables/50f70ed7c3c983893ab034e99d4118b1e6743823d6eb096abc493554c592c31d.jpg)

![5c353d560dfedc639947da5bfdff3798e9346b9710c77c2b584b93de838d091d.jpg](../iclr_results/1920_Enhancing Language Model Agents using Diversity of Thoughts/tables/5c353d560dfedc639947da5bfdff3798e9346b9710c77c2b584b93de838d091d.jpg)

![6f0bfffafc01c890d48774dc14f625816cdf30f898911387158424b7214fe2a0.jpg](../iclr_results/1920_Enhancing Language Model Agents using Diversity of Thoughts/tables/6f0bfffafc01c890d48774dc14f625816cdf30f898911387158424b7214fe2a0.jpg)

![97a5e47d3665ef24248599c38480748d6c7a418320093bd977bc336ab164988a.jpg](../iclr_results/1920_Enhancing Language Model Agents using Diversity of Thoughts/tables/97a5e47d3665ef24248599c38480748d6c7a418320093bd977bc336ab164988a.jpg)

![97c4d61ae850bd328a7bb9a66750c4e1cafb374053c736e661b72c970dc4694d.jpg](../iclr_results/1920_Enhancing Language Model Agents using Diversity of Thoughts/tables/97c4d61ae850bd328a7bb9a66750c4e1cafb374053c736e661b72c970dc4694d.jpg)

![9d5a14eeb8dcbfbd9e35375e0743fa4efa74ad75a62292ed8fe04e683c529e34.jpg](../iclr_results/1920_Enhancing Language Model Agents using Diversity of Thoughts/tables/9d5a14eeb8dcbfbd9e35375e0743fa4efa74ad75a62292ed8fe04e683c529e34.jpg)

![afcfd129dcd0049eb656acf3f704cc00371bd3c8c1d5a6fd98fde6f4e34a241a.jpg](../iclr_results/1920_Enhancing Language Model Agents using Diversity of Thoughts/tables/afcfd129dcd0049eb656acf3f704cc00371bd3c8c1d5a6fd98fde6f4e34a241a.jpg)

![e60c0d63e7ef32a1db0364b72dabfdc22e61c7a041b34a1a9df7d992acb1a9de.jpg](../iclr_results/1920_Enhancing Language Model Agents using Diversity of Thoughts/tables/e60c0d63e7ef32a1db0364b72dabfdc22e61c7a041b34a1a9df7d992acb1a9de.jpg)

![eb5bba1d5abe4f2300b1628613c9869545e664e068a3b38bb7494fa5fe1f067a.jpg](../iclr_results/1920_Enhancing Language Model Agents using Diversity of Thoughts/tables/eb5bba1d5abe4f2300b1628613c9869545e664e068a3b38bb7494fa5fe1f067a.jpg)

![ec14e50caf6bfe8040a5ec1d01f3d9b182f1e6586ed6e4de3f70fa7f204b94f4.jpg](../iclr_results/1920_Enhancing Language Model Agents using Diversity of Thoughts/tables/ec14e50caf6bfe8040a5ec1d01f3d9b182f1e6586ed6e4de3f70fa7f204b94f4.jpg)

![ec4505260581a2a3e75e0c98285640229201fdceca08edcf01c837aa0e8a100f.jpg](../iclr_results/1920_Enhancing Language Model Agents using Diversity of Thoughts/tables/ec4505260581a2a3e75e0c98285640229201fdceca08edcf01c837aa0e8a100f.jpg)

## SELF-EVOLVED REWARD LEARNING FOR LLMS


### Images

![33ca7965e41120618d2ae13962c75a9d4ee3f762afa61c494257f54bd27a88d4.jpg](../iclr_results/1921_SELF-EVOLVED REWARD LEARNING FOR LLMS/images/33ca7965e41120618d2ae13962c75a9d4ee3f762afa61c494257f54bd27a88d4.jpg)

![61a8c32636fa0ba626a117e74d5e784b329c38dc0ee22be81e5e46bd1f1f3e80.jpg](../iclr_results/1921_SELF-EVOLVED REWARD LEARNING FOR LLMS/images/61a8c32636fa0ba626a117e74d5e784b329c38dc0ee22be81e5e46bd1f1f3e80.jpg)

![6add6b4e09a32ee79dc42508982111bac56283ab0bc46625a915ebc5305aa567.jpg](../iclr_results/1921_SELF-EVOLVED REWARD LEARNING FOR LLMS/images/6add6b4e09a32ee79dc42508982111bac56283ab0bc46625a915ebc5305aa567.jpg)

![baa399a2f66cf3ba0614e71affa3b45fcef40034145ad9b2f4138d4794253476.jpg](../iclr_results/1921_SELF-EVOLVED REWARD LEARNING FOR LLMS/images/baa399a2f66cf3ba0614e71affa3b45fcef40034145ad9b2f4138d4794253476.jpg)

![d0d1433b093fa00b2612c80ba72ca78920b01e508160425bae2c863be95bf998.jpg](../iclr_results/1921_SELF-EVOLVED REWARD LEARNING FOR LLMS/images/d0d1433b093fa00b2612c80ba72ca78920b01e508160425bae2c863be95bf998.jpg)

![ea22e71974158064473eae20a0c6685e7882eef9762184518f438735815a6d0f.jpg](../iclr_results/1921_SELF-EVOLVED REWARD LEARNING FOR LLMS/images/ea22e71974158064473eae20a0c6685e7882eef9762184518f438735815a6d0f.jpg)

### Tables

![03e275f6e6e7d888f6bfb4ed01ea5509c24d96e3ba31f77b968f46d938fe92d2.jpg](../iclr_results/1921_SELF-EVOLVED REWARD LEARNING FOR LLMS/tables/03e275f6e6e7d888f6bfb4ed01ea5509c24d96e3ba31f77b968f46d938fe92d2.jpg)

![1e8770ce79cb228c40f36cbc4eb6878097ffd12bd16f50216fa273af84f78491.jpg](../iclr_results/1921_SELF-EVOLVED REWARD LEARNING FOR LLMS/tables/1e8770ce79cb228c40f36cbc4eb6878097ffd12bd16f50216fa273af84f78491.jpg)

![314c133caf6471629c650d6c7cf8103614182556ce73aeeae2d179bab59c2823.jpg](../iclr_results/1921_SELF-EVOLVED REWARD LEARNING FOR LLMS/tables/314c133caf6471629c650d6c7cf8103614182556ce73aeeae2d179bab59c2823.jpg)

![3eb1edc82770a5626bd9c0f2bac8593cac6ecd2303e0dec70a28bbd9be7ef2d0.jpg](../iclr_results/1921_SELF-EVOLVED REWARD LEARNING FOR LLMS/tables/3eb1edc82770a5626bd9c0f2bac8593cac6ecd2303e0dec70a28bbd9be7ef2d0.jpg)

![7d0b094a32e50fa598bf78aa0a0827dd09e808b17a00426df8a68293f1820be9.jpg](../iclr_results/1921_SELF-EVOLVED REWARD LEARNING FOR LLMS/tables/7d0b094a32e50fa598bf78aa0a0827dd09e808b17a00426df8a68293f1820be9.jpg)

![860e72c69d221b2af3b8922702f1eb5d34ef789b2a5fcb73ec976e182e21ee96.jpg](../iclr_results/1921_SELF-EVOLVED REWARD LEARNING FOR LLMS/tables/860e72c69d221b2af3b8922702f1eb5d34ef789b2a5fcb73ec976e182e21ee96.jpg)

![aae212fb7a56ed9ba3a404e06efcc27292571be3f5c7c2ffc429caebb82756ee.jpg](../iclr_results/1921_SELF-EVOLVED REWARD LEARNING FOR LLMS/tables/aae212fb7a56ed9ba3a404e06efcc27292571be3f5c7c2ffc429caebb82756ee.jpg)

![afad0e621f476bdd5d6a987213c8bb094036c680d984bcf64e84ebaeb30846dd.jpg](../iclr_results/1921_SELF-EVOLVED REWARD LEARNING FOR LLMS/tables/afad0e621f476bdd5d6a987213c8bb094036c680d984bcf64e84ebaeb30846dd.jpg)

## Synergy and Diversity in CLIP: Enhancing Performance Through Adaptive Backbone Ensembling


### Images

![007394932bd4375677e4c196a30c53cc2a3d220c61a1160e9d19db3d470ba5c3.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/007394932bd4375677e4c196a30c53cc2a3d220c61a1160e9d19db3d470ba5c3.jpg)

![012ebbb41f1cb212d1e43bc8ca38a22a498a8486b11bf663439cded25eaa7bef.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/012ebbb41f1cb212d1e43bc8ca38a22a498a8486b11bf663439cded25eaa7bef.jpg)

![07950c068db0b6a7f861e28fba216804e066ee30377fde7873e1020a3ef50b20.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/07950c068db0b6a7f861e28fba216804e066ee30377fde7873e1020a3ef50b20.jpg)

![0ca92b7df93404e6677707f1b5eb7e61fcd92feb658791d9ec78a70698b14be1.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/0ca92b7df93404e6677707f1b5eb7e61fcd92feb658791d9ec78a70698b14be1.jpg)

![11f9566bde8320d16903a8622b4f615828a774f7056add9c1320f72c7f0dbcd5.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/11f9566bde8320d16903a8622b4f615828a774f7056add9c1320f72c7f0dbcd5.jpg)

![20a0b58f844b16d4a08b8c1a0635d14440d0740bc16d9e31583e72a93884770d.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/20a0b58f844b16d4a08b8c1a0635d14440d0740bc16d9e31583e72a93884770d.jpg)

![221369b0013be80fe1cee94ffd55727e45bf52262e8c835fc1dcc0339ee3ff8a.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/221369b0013be80fe1cee94ffd55727e45bf52262e8c835fc1dcc0339ee3ff8a.jpg)

![242fafdcee994bfdc3c3ff0ac36944770e4066306ac58107c3ffe0ad14259b26.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/242fafdcee994bfdc3c3ff0ac36944770e4066306ac58107c3ffe0ad14259b26.jpg)

![2738241c548e4d0157b918f1c258412b948ea05b7994cd45c3552814c91df3c0.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/2738241c548e4d0157b918f1c258412b948ea05b7994cd45c3552814c91df3c0.jpg)

![2aeb7c4e6b701ab8176ead6e957b7a458da6fce4f9dbd489fe222daec4a2f0ff.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/2aeb7c4e6b701ab8176ead6e957b7a458da6fce4f9dbd489fe222daec4a2f0ff.jpg)

![4c5de1eb9c5315bc8071415158edfef3ff42a2ee292bed768d37aa008294891e.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/4c5de1eb9c5315bc8071415158edfef3ff42a2ee292bed768d37aa008294891e.jpg)

![4cae91ea69d7123f6a246bbed57ff0677bb47b121a7dbbbb345d3547c2f0d03b.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/4cae91ea69d7123f6a246bbed57ff0677bb47b121a7dbbbb345d3547c2f0d03b.jpg)

![5950505916e3c1400fdd3b69b9a8ced22bfd027c32746c9103b28fa0ebff8d66.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/5950505916e3c1400fdd3b69b9a8ced22bfd027c32746c9103b28fa0ebff8d66.jpg)

![5d92ff3635f6c8b1d890eeac067481447173cdeed5802ddd3d8220b83fee3d04.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/5d92ff3635f6c8b1d890eeac067481447173cdeed5802ddd3d8220b83fee3d04.jpg)

![60d49fe57d53371bfefc0cc5719bb5a055dc66b90cbe7b99b2107039683bb313.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/60d49fe57d53371bfefc0cc5719bb5a055dc66b90cbe7b99b2107039683bb313.jpg)

![62dfd6dfe564654ebaa61790a82009f8ddce1b01c4f780606520d1f954bde85d.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/62dfd6dfe564654ebaa61790a82009f8ddce1b01c4f780606520d1f954bde85d.jpg)

![64666fbf049ea30be817c582c9cd7fbf81c947e139f401606bea3f8033d12495.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/64666fbf049ea30be817c582c9cd7fbf81c947e139f401606bea3f8033d12495.jpg)

![72e23a68d9c26eadab3fd60b1ab506acdb9da848b4faee329ea582302e770ddd.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/72e23a68d9c26eadab3fd60b1ab506acdb9da848b4faee329ea582302e770ddd.jpg)

![7a9a4212fb4e0ebe848c7c23e59f2689bcd8c6aa8b9b89892fe0f2eba4df75d7.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/7a9a4212fb4e0ebe848c7c23e59f2689bcd8c6aa8b9b89892fe0f2eba4df75d7.jpg)

![88212c6ab54d756b38ef2cd4169f742081de2c583bbbe9fd002b77bc71fda210.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/88212c6ab54d756b38ef2cd4169f742081de2c583bbbe9fd002b77bc71fda210.jpg)

![8c900ab0d33a2752c3ed2f63cbdea3ab6fb2311c22d97d6042475ec004eaf0ba.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/8c900ab0d33a2752c3ed2f63cbdea3ab6fb2311c22d97d6042475ec004eaf0ba.jpg)

![8cdd44083b4f21a17103f5b9128c960c6ce262f0ede1202c137a5313e0276d3a.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/8cdd44083b4f21a17103f5b9128c960c6ce262f0ede1202c137a5313e0276d3a.jpg)

![94725af0779630a7aeda43f58d8a5dba086162d043a04d1357a9564f87e04ed9.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/94725af0779630a7aeda43f58d8a5dba086162d043a04d1357a9564f87e04ed9.jpg)

![99660ef7066b1a5345939a5fe77bf8600d5202ae0ae6b7cd93de51737eeb3cd3.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/99660ef7066b1a5345939a5fe77bf8600d5202ae0ae6b7cd93de51737eeb3cd3.jpg)

![a13d54c7c14e778d92d58ad845f97d8438a64acfac04b1109cf9f4b26e812cf7.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/a13d54c7c14e778d92d58ad845f97d8438a64acfac04b1109cf9f4b26e812cf7.jpg)

![a51378ebf111bf75c48b22c20c26b08c70df30c5bac5cf64f4e2bff0101546bd.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/a51378ebf111bf75c48b22c20c26b08c70df30c5bac5cf64f4e2bff0101546bd.jpg)

![af2445b62e59cd69b9ed2be512575e67d71c4190ec58b743d97e590e7f902ecb.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/af2445b62e59cd69b9ed2be512575e67d71c4190ec58b743d97e590e7f902ecb.jpg)

![b8b87dc1e0395855c38234eef07b0531c507a2a58682a240f133ff2154c94813.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/b8b87dc1e0395855c38234eef07b0531c507a2a58682a240f133ff2154c94813.jpg)

![c0ede1793cbb745892e3e84288be26becdc7da93c72f088837603d1f0d6b5fc4.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/c0ede1793cbb745892e3e84288be26becdc7da93c72f088837603d1f0d6b5fc4.jpg)

![c3fe4a20d8b8190b6849330a47c07348b879bec338da0890d04e22a5194c6646.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/c3fe4a20d8b8190b6849330a47c07348b879bec338da0890d04e22a5194c6646.jpg)

![c6cf5fa3d73e08a61dbaf0ed6bfb3f8cf83af99fb0ec588d14baffee3aa4a078.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/c6cf5fa3d73e08a61dbaf0ed6bfb3f8cf83af99fb0ec588d14baffee3aa4a078.jpg)

![d79b5ee2ec3292e7ad2b3af027d43d3516d4678c46d03930d4513c775e6ef558.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/d79b5ee2ec3292e7ad2b3af027d43d3516d4678c46d03930d4513c775e6ef558.jpg)

![eac4a60f99c5017d39799fea795b6f0c20dc7fc94759f679e79328e0f3c1eea9.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/eac4a60f99c5017d39799fea795b6f0c20dc7fc94759f679e79328e0f3c1eea9.jpg)

![f3e21d545104ae78c369e38b3d753e42fcee30f626b1f2d283402a9590804479.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/images/f3e21d545104ae78c369e38b3d753e42fcee30f626b1f2d283402a9590804479.jpg)

### Tables

![02ecd8347ee704e4d741a2d391e97dd5fced25fc686966f941ac5bf56ad54ec4.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/02ecd8347ee704e4d741a2d391e97dd5fced25fc686966f941ac5bf56ad54ec4.jpg)

![0d378279f0208b6897790bd0ad253afb68e35d3efbeadbfb29404cfc263289c8.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/0d378279f0208b6897790bd0ad253afb68e35d3efbeadbfb29404cfc263289c8.jpg)

![190cf10a7c4edbee8b8234cc879a2478e74fb4e6f18bbcbdf931cfa01016f195.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/190cf10a7c4edbee8b8234cc879a2478e74fb4e6f18bbcbdf931cfa01016f195.jpg)

![19104c5808dd1553e6643197724457eba9c20de8aa7fc7451a1aee2e4a52787f.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/19104c5808dd1553e6643197724457eba9c20de8aa7fc7451a1aee2e4a52787f.jpg)

![1c4eed8953e3c58884997f313bc3d35eab70cb5a67bde15ae9ee32e62bffe2af.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/1c4eed8953e3c58884997f313bc3d35eab70cb5a67bde15ae9ee32e62bffe2af.jpg)

![1f62324f958844aec34b4088238c7eb62b6ff5b27bb229a7f791473dabd3a9a0.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/1f62324f958844aec34b4088238c7eb62b6ff5b27bb229a7f791473dabd3a9a0.jpg)

![1ff1c93052bbf37f06e278333a3f22f7f3cc65aa46e6a58d66bc2e3e313d8f89.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/1ff1c93052bbf37f06e278333a3f22f7f3cc65aa46e6a58d66bc2e3e313d8f89.jpg)

![2a2c1ea3eac614301c3ad81b7fa57d713fc6d5948c0d0aa363dc859eadc601ea.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/2a2c1ea3eac614301c3ad81b7fa57d713fc6d5948c0d0aa363dc859eadc601ea.jpg)

![365df8a2d1e67730485f28f5ce6187989a7faaaaba45f9fb91fd0c78ad26ffc7.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/365df8a2d1e67730485f28f5ce6187989a7faaaaba45f9fb91fd0c78ad26ffc7.jpg)

![3aa55552df702b886d271011ee2adb56ed6e306d7e441c1dc55e4fccaeef622a.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/3aa55552df702b886d271011ee2adb56ed6e306d7e441c1dc55e4fccaeef622a.jpg)

![44f33e0496a18bc8d7f9c90f879a02e5b41ee2879064faae803ffbd5ab6612b7.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/44f33e0496a18bc8d7f9c90f879a02e5b41ee2879064faae803ffbd5ab6612b7.jpg)

![45e739783c84f61927d8bbdd17b1831ab4e511dce497561516dd0eb471e34a70.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/45e739783c84f61927d8bbdd17b1831ab4e511dce497561516dd0eb471e34a70.jpg)

![470595252d448ff547fa226331199e34655b531d280b2e3f917ddcf57015005a.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/470595252d448ff547fa226331199e34655b531d280b2e3f917ddcf57015005a.jpg)

![4cf5150b22c016f8176567213f1a91d541d82b611987eca75f88d20ef63176bf.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/4cf5150b22c016f8176567213f1a91d541d82b611987eca75f88d20ef63176bf.jpg)

![4d1871ef7d98e76e170e13ac93b38fa1a04b7272c80313c068987e1cd25e3eee.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/4d1871ef7d98e76e170e13ac93b38fa1a04b7272c80313c068987e1cd25e3eee.jpg)

![4da3c142a8200feb920a45c24eb3136c38d0cfdc0a39dbd80bdf62c60f64f3a1.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/4da3c142a8200feb920a45c24eb3136c38d0cfdc0a39dbd80bdf62c60f64f3a1.jpg)

![4e6fa0d339a5a6b744e0e60174a4d437444b32aa7d3e7e4a2dba22622d28c305.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/4e6fa0d339a5a6b744e0e60174a4d437444b32aa7d3e7e4a2dba22622d28c305.jpg)

![549223479dc2f26c75e4973c6ee028303000148379a599266a38606c8c8f3fb1.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/549223479dc2f26c75e4973c6ee028303000148379a599266a38606c8c8f3fb1.jpg)

![5e33974b906f0f43b73d8706efd30e28f79957d1258a573090a76acbdf22d6a1.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/5e33974b906f0f43b73d8706efd30e28f79957d1258a573090a76acbdf22d6a1.jpg)

![708d9a35e9532fa5dc97a6cb96ce2f227818d8be95a46fba71a170c4ff208b26.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/708d9a35e9532fa5dc97a6cb96ce2f227818d8be95a46fba71a170c4ff208b26.jpg)

![767838ccb423c6a47835868dfda59567d48371e783e26156be051f0a53ad0c5e.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/767838ccb423c6a47835868dfda59567d48371e783e26156be051f0a53ad0c5e.jpg)

![7cd0c79e80b9ce7e7ee4f846f350299227c70e4b3308d0bb99d2008c828a9db0.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/7cd0c79e80b9ce7e7ee4f846f350299227c70e4b3308d0bb99d2008c828a9db0.jpg)

![85e76010f5ee64bb62cea7b3590349273a78ba635797c004a1e1c5a1e0613858.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/85e76010f5ee64bb62cea7b3590349273a78ba635797c004a1e1c5a1e0613858.jpg)

![98306f065deb711bc965fe2b5d5fecf13e202a0fda4ad23912fc2ab7576deee6.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/98306f065deb711bc965fe2b5d5fecf13e202a0fda4ad23912fc2ab7576deee6.jpg)

![a0e1652ef588a27bf2780104409bd22b31258474f549196e27a0c7e0808da814.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/a0e1652ef588a27bf2780104409bd22b31258474f549196e27a0c7e0808da814.jpg)

![c261af827281422b28053c4ea560d51acb3612d30f02485c9c4e98b538e95aab.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/c261af827281422b28053c4ea560d51acb3612d30f02485c9c4e98b538e95aab.jpg)

![c2c8ee0c7373688088606f595ab6eb6c65621488793142e1978127646e182a65.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/c2c8ee0c7373688088606f595ab6eb6c65621488793142e1978127646e182a65.jpg)

![c66bd268a12cea8827b764e9f4ecf1b0a08ec1e3d7f77bd642822ab1ce88e97c.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/c66bd268a12cea8827b764e9f4ecf1b0a08ec1e3d7f77bd642822ab1ce88e97c.jpg)

![d4dab82ceb9c104dc14c827232258bb562ae382b189f1217f3c70ab9d87dafc1.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/d4dab82ceb9c104dc14c827232258bb562ae382b189f1217f3c70ab9d87dafc1.jpg)

![de0fc65474e7566e2655936e510b9a9bba4c2d60f6b92b449f86f09b04502f70.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/de0fc65474e7566e2655936e510b9a9bba4c2d60f6b92b449f86f09b04502f70.jpg)

![e4073e5416e0f09962148c5a20334e1c60e2e1f692debd66d036fd025e66817e.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/e4073e5416e0f09962148c5a20334e1c60e2e1f692debd66d036fd025e66817e.jpg)

![f7648ed85683cf5371c6c0b42caeee6d0c82238772c6345b60823987038771d2.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/f7648ed85683cf5371c6c0b42caeee6d0c82238772c6345b60823987038771d2.jpg)

![fa36ed8b5b734dbf303d73cfa61190d0ce6f6642bf6899aca102e2619b891e08.jpg](../iclr_results/1922_Synergy and Diversity in CLIP_ Enhancing Performance Through Adaptive Backbone Ensembling/tables/fa36ed8b5b734dbf303d73cfa61190d0ce6f6642bf6899aca102e2619b891e08.jpg)

## LDAdam: Adaptive Optimization from Low-Dimensional Gradient Statistics


### Images

![1a6414647fffdb99f35ca3dfd52e2bf808d20eed43e5ea1a5ed20003a7879d6d.jpg](../iclr_results/1923_LDAdam_ Adaptive Optimization from Low-Dimensional Gradient Statistics/images/1a6414647fffdb99f35ca3dfd52e2bf808d20eed43e5ea1a5ed20003a7879d6d.jpg)

![454a77c7c284414d3718965a7f4e311a71c8604d8099e30f7ea378e73fc694fb.jpg](../iclr_results/1923_LDAdam_ Adaptive Optimization from Low-Dimensional Gradient Statistics/images/454a77c7c284414d3718965a7f4e311a71c8604d8099e30f7ea378e73fc694fb.jpg)

![75e971e3933ebc3b8d6affa8d602a041fcb53b7dacad79254a90c002ee42b8cf.jpg](../iclr_results/1923_LDAdam_ Adaptive Optimization from Low-Dimensional Gradient Statistics/images/75e971e3933ebc3b8d6affa8d602a041fcb53b7dacad79254a90c002ee42b8cf.jpg)

![7b5be033b5b60ad57c38ad7c472540a3f687659c3c24d1a84cac736eaca3502c.jpg](../iclr_results/1923_LDAdam_ Adaptive Optimization from Low-Dimensional Gradient Statistics/images/7b5be033b5b60ad57c38ad7c472540a3f687659c3c24d1a84cac736eaca3502c.jpg)

![90984ad8da0ae7b4266af1328c450b7773ebdca03ce25df61976366581a83be9.jpg](../iclr_results/1923_LDAdam_ Adaptive Optimization from Low-Dimensional Gradient Statistics/images/90984ad8da0ae7b4266af1328c450b7773ebdca03ce25df61976366581a83be9.jpg)

### Tables

![04b48642ecc53b0c6d0f911e9a3791847586700579f9c2b62c9229ded63afd8e.jpg](../iclr_results/1923_LDAdam_ Adaptive Optimization from Low-Dimensional Gradient Statistics/tables/04b48642ecc53b0c6d0f911e9a3791847586700579f9c2b62c9229ded63afd8e.jpg)

![07f52b90486c81e15a019d2d6380b7b9da174392ac8403a9822f3915fc7d9329.jpg](../iclr_results/1923_LDAdam_ Adaptive Optimization from Low-Dimensional Gradient Statistics/tables/07f52b90486c81e15a019d2d6380b7b9da174392ac8403a9822f3915fc7d9329.jpg)

![2ca9c735995080f90733a2b0cf0b5573ccd87d41af52ff23041c20f92e182c54.jpg](../iclr_results/1923_LDAdam_ Adaptive Optimization from Low-Dimensional Gradient Statistics/tables/2ca9c735995080f90733a2b0cf0b5573ccd87d41af52ff23041c20f92e182c54.jpg)

![2f67a2c9ea189950a2ffb1aca634432dc902a6fbe9911c00a1e8dab815301d7a.jpg](../iclr_results/1923_LDAdam_ Adaptive Optimization from Low-Dimensional Gradient Statistics/tables/2f67a2c9ea189950a2ffb1aca634432dc902a6fbe9911c00a1e8dab815301d7a.jpg)

![523c80c97918a918dae95566c62fc22653c9d14ca6cd484b951c127c7905b6fe.jpg](../iclr_results/1923_LDAdam_ Adaptive Optimization from Low-Dimensional Gradient Statistics/tables/523c80c97918a918dae95566c62fc22653c9d14ca6cd484b951c127c7905b6fe.jpg)

![5d720e110dceed2736ec016fafd31580f363e3f232f2d6d44a98d77cc601e07d.jpg](../iclr_results/1923_LDAdam_ Adaptive Optimization from Low-Dimensional Gradient Statistics/tables/5d720e110dceed2736ec016fafd31580f363e3f232f2d6d44a98d77cc601e07d.jpg)

![669d159099dab2aed3e2a40ac9cbc724fef8eed761294fe5b37b0a71029e0735.jpg](../iclr_results/1923_LDAdam_ Adaptive Optimization from Low-Dimensional Gradient Statistics/tables/669d159099dab2aed3e2a40ac9cbc724fef8eed761294fe5b37b0a71029e0735.jpg)

![7cbaa4e944014637cd2bbbe1e0477c93e189857fcfbe55139f4e5b76a46067f7.jpg](../iclr_results/1923_LDAdam_ Adaptive Optimization from Low-Dimensional Gradient Statistics/tables/7cbaa4e944014637cd2bbbe1e0477c93e189857fcfbe55139f4e5b76a46067f7.jpg)

![ae07e70473b019a7276fcb24e1ec4bd8755ec7448ddef21d25eeede995db0ba2.jpg](../iclr_results/1923_LDAdam_ Adaptive Optimization from Low-Dimensional Gradient Statistics/tables/ae07e70473b019a7276fcb24e1ec4bd8755ec7448ddef21d25eeede995db0ba2.jpg)

![b922384b1cd599d55eb1fb07eebcb3bad22155a63acd5d99de3d68dc95609baa.jpg](../iclr_results/1923_LDAdam_ Adaptive Optimization from Low-Dimensional Gradient Statistics/tables/b922384b1cd599d55eb1fb07eebcb3bad22155a63acd5d99de3d68dc95609baa.jpg)

![e26a1a5b01f44676c4c3329ef512c4b6dd186634bfe34ace8e51bee5b5057461.jpg](../iclr_results/1923_LDAdam_ Adaptive Optimization from Low-Dimensional Gradient Statistics/tables/e26a1a5b01f44676c4c3329ef512c4b6dd186634bfe34ace8e51bee5b5057461.jpg)

![fca5c942a16026a44cc3d557d5413d8b9d6fbb5bed39b799a925d9c81381fe72.jpg](../iclr_results/1923_LDAdam_ Adaptive Optimization from Low-Dimensional Gradient Statistics/tables/fca5c942a16026a44cc3d557d5413d8b9d6fbb5bed39b799a925d9c81381fe72.jpg)

## InstantPortrait: One-Step Portrait Editing via Diffusion Multi-Objective Distillation


### Images

![06ece6794b97a6b3fa5fe05caf241b0159c2097045b3cb333daf4a879b1972bb.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/images/06ece6794b97a6b3fa5fe05caf241b0159c2097045b3cb333daf4a879b1972bb.jpg)

![420945bb8829749c8ea55147c20051e82d6a09fd9b6d7de696d88907ca42e65d.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/images/420945bb8829749c8ea55147c20051e82d6a09fd9b6d7de696d88907ca42e65d.jpg)

![444bec1ba30012187d838278e0dce6480dab16e5613de1d5079e62edefc1e0b0.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/images/444bec1ba30012187d838278e0dce6480dab16e5613de1d5079e62edefc1e0b0.jpg)

![4e66a29ab1ec2110827bdefa2927112ad7c0cd0b0f43c29c1b2a89c7edc85703.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/images/4e66a29ab1ec2110827bdefa2927112ad7c0cd0b0f43c29c1b2a89c7edc85703.jpg)

![4f5a2f56c4c67d27c9933e99e5b14befe3c2aec3ae4c995b80451dbbf3ad793c.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/images/4f5a2f56c4c67d27c9933e99e5b14befe3c2aec3ae4c995b80451dbbf3ad793c.jpg)

![5c42011de943449a3433e597bad9c437cc84ee169100b766230109869bec412b.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/images/5c42011de943449a3433e597bad9c437cc84ee169100b766230109869bec412b.jpg)

![74ae09f9465b0879eee40f6894e0f22425ec98c7b9cd1ce0220c2a64e1840d2e.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/images/74ae09f9465b0879eee40f6894e0f22425ec98c7b9cd1ce0220c2a64e1840d2e.jpg)

![9576a634386bd4ece5eb893dbc2ec79859ab9255947c01391c07cfeba470317a.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/images/9576a634386bd4ece5eb893dbc2ec79859ab9255947c01391c07cfeba470317a.jpg)

![962594b11abc3915adf1f0cf3b6c96b5238e5dda5dbcf06ec531a96d9f5e453d.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/images/962594b11abc3915adf1f0cf3b6c96b5238e5dda5dbcf06ec531a96d9f5e453d.jpg)

![97eb6aac5a8c84600871ff7289bc937713e5e09197f6dc78a21a5035d5cafb74.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/images/97eb6aac5a8c84600871ff7289bc937713e5e09197f6dc78a21a5035d5cafb74.jpg)

![9981a1d74d3b89b4bea04a6c33ed4cc11533c387b0d1ddb74e6ac899a7d794c8.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/images/9981a1d74d3b89b4bea04a6c33ed4cc11533c387b0d1ddb74e6ac899a7d794c8.jpg)

![b8a0c182d28cbe64c42831c2d67a2fafd707b1aafb6ceb85badf1589c344423f.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/images/b8a0c182d28cbe64c42831c2d67a2fafd707b1aafb6ceb85badf1589c344423f.jpg)

![b91ebb32534335be57a039c54081a2499c920c21e0afb92599b724ce5eedf5b3.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/images/b91ebb32534335be57a039c54081a2499c920c21e0afb92599b724ce5eedf5b3.jpg)

![ba8fa65d614ae9c84e3bc48331f66aea58e2a8975429b17fd7096b9f0030e6e7.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/images/ba8fa65d614ae9c84e3bc48331f66aea58e2a8975429b17fd7096b9f0030e6e7.jpg)

![c8b32a8086a8e578eedff49db9edc91728d11621da83a009f94628640c1c3644.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/images/c8b32a8086a8e578eedff49db9edc91728d11621da83a009f94628640c1c3644.jpg)

![d662f2009e8b6a1c888507348c72cb200aa806deaaa3f1a228c39cb6e4d706b8.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/images/d662f2009e8b6a1c888507348c72cb200aa806deaaa3f1a228c39cb6e4d706b8.jpg)

![dbc5414c1db7583d3756e9b7290a008e9e40ce9042a01ae8be1e28b428ba4392.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/images/dbc5414c1db7583d3756e9b7290a008e9e40ce9042a01ae8be1e28b428ba4392.jpg)

![e0b45e4101a195a7ec5d509b51df1b6781920132c4e37d10077031b3d43c62c4.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/images/e0b45e4101a195a7ec5d509b51df1b6781920132c4e37d10077031b3d43c62c4.jpg)

![eb6b9eff5ec0db4ffd8840efde02c195209b81079d0c862baf7b1438d88d7521.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/images/eb6b9eff5ec0db4ffd8840efde02c195209b81079d0c862baf7b1438d88d7521.jpg)

![ed99720c05132d9cfee8b0ae101b9f119565613c32c77ea7dfc3cf457bc8a11a.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/images/ed99720c05132d9cfee8b0ae101b9f119565613c32c77ea7dfc3cf457bc8a11a.jpg)

![f5e532fdb9e6a99109270c06cf9c6726ebdc4a738dc9c4ed058cb7babe7997a3.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/images/f5e532fdb9e6a99109270c06cf9c6726ebdc4a738dc9c4ed058cb7babe7997a3.jpg)

![fe0f17883f2f6771e45604bb0897e01d671ca3ce04f852edaf071ec3a175b123.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/images/fe0f17883f2f6771e45604bb0897e01d671ca3ce04f852edaf071ec3a175b123.jpg)

### Tables

![1bcdc18dfa0b27f20b75d837f27e1296353843123bd3e68c2d625ccf7998066f.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/tables/1bcdc18dfa0b27f20b75d837f27e1296353843123bd3e68c2d625ccf7998066f.jpg)

![2b58f03cd7b45658cf3918b2224899203d3fff5d55cb47fc3006cd39b9a8a8f8.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/tables/2b58f03cd7b45658cf3918b2224899203d3fff5d55cb47fc3006cd39b9a8a8f8.jpg)

![2bc1ed6f17568b10126076e012955680f3d0d29e6432133599ba2b58928e4bd2.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/tables/2bc1ed6f17568b10126076e012955680f3d0d29e6432133599ba2b58928e4bd2.jpg)

![6f6edac28907232d07e0bf548504d2b0d85020faa98da38a45603e701b7ac8d4.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/tables/6f6edac28907232d07e0bf548504d2b0d85020faa98da38a45603e701b7ac8d4.jpg)

![9df36d6f710f0d285c57df0fb670a38291e7c11387325eee19cbf46e7422fbc4.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/tables/9df36d6f710f0d285c57df0fb670a38291e7c11387325eee19cbf46e7422fbc4.jpg)

![aa0250c6e7c0c213246fec67264c9530ee3b4162045a98fe3b455f269cec03b6.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/tables/aa0250c6e7c0c213246fec67264c9530ee3b4162045a98fe3b455f269cec03b6.jpg)

![ee3683a2b2ce94cfdfe507862ee8dc89d9a4801c502226310b1e402ab4a6520f.jpg](../iclr_results/1924_InstantPortrait_ One-Step Portrait Editing via Diffusion Multi-Objective Distillation/tables/ee3683a2b2ce94cfdfe507862ee8dc89d9a4801c502226310b1e402ab4a6520f.jpg)

## What Makes Large Language Models Reason in (Multi-Turn) Code Generation?


### Images

![03c5e682a868eda056fb6d9a8adc654141637c2a754d825475005f9b2b1e485f.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/03c5e682a868eda056fb6d9a8adc654141637c2a754d825475005f9b2b1e485f.jpg)

![05bcbcf67d3c8a4336fc451d9f690a15a6de4a830fd9d9f28962a54f268a758b.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/05bcbcf67d3c8a4336fc451d9f690a15a6de4a830fd9d9f28962a54f268a758b.jpg)

![12ba3ce8c53d4afb2098702c9efa4a23dc6efad7a77145168a9672cc4c2300b6.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/12ba3ce8c53d4afb2098702c9efa4a23dc6efad7a77145168a9672cc4c2300b6.jpg)

![2234ba414d431cd917845064eca40871d39205fb0f1595cc894df447fb9870a4.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/2234ba414d431cd917845064eca40871d39205fb0f1595cc894df447fb9870a4.jpg)

![23004d634c953a67ea5f822246a7efe3e895bfec3397d3de050da052eaee5cff.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/23004d634c953a67ea5f822246a7efe3e895bfec3397d3de050da052eaee5cff.jpg)

![2ca6d552e964ec94789a5cae51709b4fd1e587dcc59b5046794430bc425567b6.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/2ca6d552e964ec94789a5cae51709b4fd1e587dcc59b5046794430bc425567b6.jpg)

![35b9286919f4afd5ffef47cc06f97935e67de3e34de60ba5cc438322caf6b643.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/35b9286919f4afd5ffef47cc06f97935e67de3e34de60ba5cc438322caf6b643.jpg)

![4407a56bc4d8235afb596e8e07d7a47e942c9c1e4cfc11dd1bc842b51222d4b2.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/4407a56bc4d8235afb596e8e07d7a47e942c9c1e4cfc11dd1bc842b51222d4b2.jpg)

![5253a1e77bb16d9825b5e89bf731d3ca6f5bc8c6d2bcf6b6a08931e380e1ed98.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/5253a1e77bb16d9825b5e89bf731d3ca6f5bc8c6d2bcf6b6a08931e380e1ed98.jpg)

![5fac5ed8a3b6650fd0f06dd1d05fc7e4b176a5859469014e51b0dccfdc2bca7d.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/5fac5ed8a3b6650fd0f06dd1d05fc7e4b176a5859469014e51b0dccfdc2bca7d.jpg)

![6da61c32079dc29531ca0b8586902c51c097dda06d024efecba9753ae873d8aa.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/6da61c32079dc29531ca0b8586902c51c097dda06d024efecba9753ae873d8aa.jpg)

![72d5afacdb53f8754e85b3d0888bf2561d93ebdd8ca2b7fd11ff39a2db0f9304.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/72d5afacdb53f8754e85b3d0888bf2561d93ebdd8ca2b7fd11ff39a2db0f9304.jpg)

![74b684d26abee0920f6afa0439ba856562d5b966b6fc5da96524795dc01b1ec7.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/74b684d26abee0920f6afa0439ba856562d5b966b6fc5da96524795dc01b1ec7.jpg)

![757021faabe4b996c858a97625af4c3f20579d9f4128133dfceb0a3c20c5454b.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/757021faabe4b996c858a97625af4c3f20579d9f4128133dfceb0a3c20c5454b.jpg)

![a8bbaffac9a01ad973ae6a3c9d960e2a9cce52a58717a3c4eaf226fe3b9dd35f.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/a8bbaffac9a01ad973ae6a3c9d960e2a9cce52a58717a3c4eaf226fe3b9dd35f.jpg)

![aea49d285b738c9b48cdd5f1856865c0c3a73567c61c12d35527f20b16755ed6.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/aea49d285b738c9b48cdd5f1856865c0c3a73567c61c12d35527f20b16755ed6.jpg)

![b4b3a607848d723a08055844eefcb82099dd00d90920feb2e55d3b9c4f47a26a.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/b4b3a607848d723a08055844eefcb82099dd00d90920feb2e55d3b9c4f47a26a.jpg)

![cfcfe34a223ebd54d088903c199824687ee4fffa23915ee0b65fe1d0801e0135.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/cfcfe34a223ebd54d088903c199824687ee4fffa23915ee0b65fe1d0801e0135.jpg)

![d58874beab52799582a303ddb89d6427c69bc2d591d754471f45d186097eee87.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/d58874beab52799582a303ddb89d6427c69bc2d591d754471f45d186097eee87.jpg)

![e99e587f1cef43da177320308856c9ce02e20966e7a4280f8a76405fda9ff986.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/e99e587f1cef43da177320308856c9ce02e20966e7a4280f8a76405fda9ff986.jpg)

![ebfe69c3648bd9157a7db62534a555e8ac833f0168872f30f51b0c83c00c86f2.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/ebfe69c3648bd9157a7db62534a555e8ac833f0168872f30f51b0c83c00c86f2.jpg)

![ef1a76a1199262d5a6bf42d9866ebbe80c4eed089beec005d18876a87ec744d8.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/ef1a76a1199262d5a6bf42d9866ebbe80c4eed089beec005d18876a87ec744d8.jpg)

![f6d12184f53fe04e55843a8b3ed4c5827a6ff14bf4936fb3e98b5669b4187297.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/f6d12184f53fe04e55843a8b3ed4c5827a6ff14bf4936fb3e98b5669b4187297.jpg)

![fc453e75ccc486c7c71a549131ab2ddc2c1076ec53c8541173076ca033eb6634.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/images/fc453e75ccc486c7c71a549131ab2ddc2c1076ec53c8541173076ca033eb6634.jpg)

### Tables

![25368b0a9300724468f0b10d1401fe4691ff4aa232c9e4b62aa05ddfecbd1d4b.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/tables/25368b0a9300724468f0b10d1401fe4691ff4aa232c9e4b62aa05ddfecbd1d4b.jpg)

![4605b44c8aca3183b3cc52e0a7c6a340841d412a57fe8f3d36b538d0259d9c57.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/tables/4605b44c8aca3183b3cc52e0a7c6a340841d412a57fe8f3d36b538d0259d9c57.jpg)

![52a95c7f22dbc72aebf40ef68841f74f01e37ba168bf43fbcb99ae1176b86976.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/tables/52a95c7f22dbc72aebf40ef68841f74f01e37ba168bf43fbcb99ae1176b86976.jpg)

![695d5ab827c034630de3717be92b6d8d3d1e10f3122fe582b63b2929c273e14c.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/tables/695d5ab827c034630de3717be92b6d8d3d1e10f3122fe582b63b2929c273e14c.jpg)

![785a206222ff2e25b366897097c18f8786c43bbffb0fdee5a431be7ce5496e1e.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/tables/785a206222ff2e25b366897097c18f8786c43bbffb0fdee5a431be7ce5496e1e.jpg)

![92a3693e49ecf0f73e8b0cb1c9d6a1c98c7b68c3cfd0965a062f2e7d2aef7ead.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/tables/92a3693e49ecf0f73e8b0cb1c9d6a1c98c7b68c3cfd0965a062f2e7d2aef7ead.jpg)

![95ae8da4d31b54d6c536e45aeed991218ba6fa4e1caccae41937266d0455b488.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/tables/95ae8da4d31b54d6c536e45aeed991218ba6fa4e1caccae41937266d0455b488.jpg)

![9c78375772b3d000d134a4a5fd1f8f5fd3b93cacd186db39dfbc391b006adb30.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/tables/9c78375772b3d000d134a4a5fd1f8f5fd3b93cacd186db39dfbc391b006adb30.jpg)

![a15c461f53df722fcd044466681a8af0581afc8395776357953777246b57ee96.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/tables/a15c461f53df722fcd044466681a8af0581afc8395776357953777246b57ee96.jpg)

![a265138ddff2aa30da1e0fb601bd49b508715878c56d02c3ee293b369de2ed1e.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/tables/a265138ddff2aa30da1e0fb601bd49b508715878c56d02c3ee293b369de2ed1e.jpg)

![c69ce80ade0411cc97c248891e3ec649b5770a9da03f89c74b0c2d0bca5db596.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/tables/c69ce80ade0411cc97c248891e3ec649b5770a9da03f89c74b0c2d0bca5db596.jpg)

![d3886d6e285f4245afde1d62c9790188b284b1be0a3423c6563b2507af55be6e.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/tables/d3886d6e285f4245afde1d62c9790188b284b1be0a3423c6563b2507af55be6e.jpg)

![dbd1a23a65e1c35b4aea3e5d055701e4386678916a34c6c2cdeb8c57745eb9fe.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/tables/dbd1a23a65e1c35b4aea3e5d055701e4386678916a34c6c2cdeb8c57745eb9fe.jpg)

![df4af5d16490b81d9f462da3b742471adb3482144b85fca33fd9d457954193b6.jpg](../iclr_results/1925_What Makes Large Language Models Reason in (Multi-Turn) Code Generation_/tables/df4af5d16490b81d9f462da3b742471adb3482144b85fca33fd9d457954193b6.jpg)

## Generalized Consistency Trajectory Models for Image Manipulation


### Images

![09e7b014898c92278caca6d87cee9680b2695a70d92aed4f0429f417ee44f066.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/images/09e7b014898c92278caca6d87cee9680b2695a70d92aed4f0429f417ee44f066.jpg)

![1c2593750924e83614070fbb2f517bb91db350916a81ce77cd7430bbf5d3e839.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/images/1c2593750924e83614070fbb2f517bb91db350916a81ce77cd7430bbf5d3e839.jpg)

![27b91f548dc278b07b3965b00dbb9248f3baf964e61e2dd9aba5f61de6543758.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/images/27b91f548dc278b07b3965b00dbb9248f3baf964e61e2dd9aba5f61de6543758.jpg)

![316273e9aee99083ca25c51341f8bbef61a204c877cd6eedb8e05836c603c193.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/images/316273e9aee99083ca25c51341f8bbef61a204c877cd6eedb8e05836c603c193.jpg)

![3f004294d17187c33b42a2e9d5eb6370d3b9d69ed67c6bb8495c8d7609169787.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/images/3f004294d17187c33b42a2e9d5eb6370d3b9d69ed67c6bb8495c8d7609169787.jpg)

![6a60bf690a79cd2ab3a04ccc45e906f4d1d8a67de46e2433a65065cc53f032c4.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/images/6a60bf690a79cd2ab3a04ccc45e906f4d1d8a67de46e2433a65065cc53f032c4.jpg)

![7200f5e591476bf3b95f801c0b650ee1f3682350c3ca16fb48d1a71d82d3d318.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/images/7200f5e591476bf3b95f801c0b650ee1f3682350c3ca16fb48d1a71d82d3d318.jpg)

![8bb71473b64460fd1b3b441dfb6cd42bde093e6cf35b9fe68659ae6aa0149d39.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/images/8bb71473b64460fd1b3b441dfb6cd42bde093e6cf35b9fe68659ae6aa0149d39.jpg)

![9089662260ec2745146fd8d67a892491c0031f7b4b88b58e2daee38356aa8d72.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/images/9089662260ec2745146fd8d67a892491c0031f7b4b88b58e2daee38356aa8d72.jpg)

![a4553f6ce5a1e333c596fa58ed91f2a5426d6977e985b524c349d51b3edf9ac2.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/images/a4553f6ce5a1e333c596fa58ed91f2a5426d6977e985b524c349d51b3edf9ac2.jpg)

![c22c664accc22ae5bd25206982eace238f5394c6e48c13bbe2e790ccdf4643bc.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/images/c22c664accc22ae5bd25206982eace238f5394c6e48c13bbe2e790ccdf4643bc.jpg)

![c255c65c71712e75357af0e22af385fefc1d8bc6246cf3c7a7502ef892faf236.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/images/c255c65c71712e75357af0e22af385fefc1d8bc6246cf3c7a7502ef892faf236.jpg)

![d10aac018ac827142e813513c260dd652e0c6112ead40ddad737dea1d5761d65.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/images/d10aac018ac827142e813513c260dd652e0c6112ead40ddad737dea1d5761d65.jpg)

![eded3f8f83e1e2fe456663d7c80a5fce91132ecdc10afdcb636482fe2d11b5ad.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/images/eded3f8f83e1e2fe456663d7c80a5fce91132ecdc10afdcb636482fe2d11b5ad.jpg)

![f7286719a0b74a54e8a6b2ec9c000c3d3005822e4aa13496a3d059d7202a08fd.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/images/f7286719a0b74a54e8a6b2ec9c000c3d3005822e4aa13496a3d059d7202a08fd.jpg)

![fd92efffe708a56b1d6138be9f12fe64d3c8ae287c3a0b1b63ca464a6437acd8.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/images/fd92efffe708a56b1d6138be9f12fe64d3c8ae287c3a0b1b63ca464a6437acd8.jpg)

### Tables

![34ab30247857b7d854069baba9d329240af7f1e9f3568094f0a14e5fd48cb248.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/tables/34ab30247857b7d854069baba9d329240af7f1e9f3568094f0a14e5fd48cb248.jpg)

![3e30f91e8487ae282cfc13e511a5a76156ee583b4328dcf3e1f3a0becf47f84e.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/tables/3e30f91e8487ae282cfc13e511a5a76156ee583b4328dcf3e1f3a0becf47f84e.jpg)

![73c1066a10fb1ac343a6f6d2c22cc602a0214f759a4654f192a25c2d7e4acf85.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/tables/73c1066a10fb1ac343a6f6d2c22cc602a0214f759a4654f192a25c2d7e4acf85.jpg)

![8666364dd6ccf3476c852be5cd186255e0d343403946b2717a804106728a3c41.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/tables/8666364dd6ccf3476c852be5cd186255e0d343403946b2717a804106728a3c41.jpg)

![895b30c20cb41a8c10229f47244459c018a5d0eb8803e63bb4ab6e293d36a12f.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/tables/895b30c20cb41a8c10229f47244459c018a5d0eb8803e63bb4ab6e293d36a12f.jpg)

![904a38d18e4196bdeccdfa70e4d645b9154f781463e561a4a3c73a80e72a73c6.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/tables/904a38d18e4196bdeccdfa70e4d645b9154f781463e561a4a3c73a80e72a73c6.jpg)

![c478ae0e285affb3fea87eca919593f40ddd0762d98ad026ebeb88d60db1a3f4.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/tables/c478ae0e285affb3fea87eca919593f40ddd0762d98ad026ebeb88d60db1a3f4.jpg)

![de50cd03c3f91408cc39cae8ca947322f719cbadb540aa8293fde31886f6b347.jpg](../iclr_results/1926_Generalized Consistency Trajectory Models for Image Manipulation/tables/de50cd03c3f91408cc39cae8ca947322f719cbadb540aa8293fde31886f6b347.jpg)

## Complexity Lower Bounds of Adaptive Gradient Algorithms for Non-convex Stochastic Optimization under Relaxed Smoothness


### Images

![2170da4e8af636ab9cb9efb2b13d042de9a115f348711baf0b9dd2daaac1a652.jpg](../iclr_results/1927_Complexity Lower Bounds of Adaptive Gradient Algorithms for Non-convex Stochastic Optimization under/images/2170da4e8af636ab9cb9efb2b13d042de9a115f348711baf0b9dd2daaac1a652.jpg)

### Tables

![ac3c5c5dd8ac50f6785253da02769a6fa441d5febd9ce73ca3927a10989b14b9.jpg](../iclr_results/1927_Complexity Lower Bounds of Adaptive Gradient Algorithms for Non-convex Stochastic Optimization under/tables/ac3c5c5dd8ac50f6785253da02769a6fa441d5febd9ce73ca3927a10989b14b9.jpg)

## Federated Few-Shot Class-Incremental Learning


### Images

![4bf6f50d39438ac022c3403701f3c9a447b5057c0b05fdd468089e98a01692cb.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/images/4bf6f50d39438ac022c3403701f3c9a447b5057c0b05fdd468089e98a01692cb.jpg)

![5ba651b4b14c51e58b3a5c41a2b1dc45330c46dcc78a54f5000c2f30cf9d368b.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/images/5ba651b4b14c51e58b3a5c41a2b1dc45330c46dcc78a54f5000c2f30cf9d368b.jpg)

![c90e5ba09fe94db02e1d43d1d7aa0d3446ff9a335b11044a6fb3ec6f16d6fb8a.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/images/c90e5ba09fe94db02e1d43d1d7aa0d3446ff9a335b11044a6fb3ec6f16d6fb8a.jpg)

![d1274ea0208f899d38222df9f8be3279f12ef112c9892f05c57dbde22d5c663e.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/images/d1274ea0208f899d38222df9f8be3279f12ef112c9892f05c57dbde22d5c663e.jpg)

![f2a4c0cd6d9798ffbc6519fce4903baedb12e60392d88a924c62d4395748e8d3.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/images/f2a4c0cd6d9798ffbc6519fce4903baedb12e60392d88a924c62d4395748e8d3.jpg)

![f83d3400a75ff6dcd25a48ad1307cb006a5da37bf87571898808e7021b2bc7bd.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/images/f83d3400a75ff6dcd25a48ad1307cb006a5da37bf87571898808e7021b2bc7bd.jpg)

### Tables

![00c75efd88bb16832b262178be8c63bc03f622fc6ff8d7db537c1d35fb559fa4.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/tables/00c75efd88bb16832b262178be8c63bc03f622fc6ff8d7db537c1d35fb559fa4.jpg)

![1a883f5c53d978f2b2b9d36a09c5606598062d6a0f76e38231c65820237b560f.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/tables/1a883f5c53d978f2b2b9d36a09c5606598062d6a0f76e38231c65820237b560f.jpg)

![22499e6516cce355790d36744fe1c3d846060e854f0b8be1dfa1e4fd02355a0d.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/tables/22499e6516cce355790d36744fe1c3d846060e854f0b8be1dfa1e4fd02355a0d.jpg)

![27b2b9d9728d4f432924a3e06aa7d4223367cb128c9ba024bbfd8b2c56d1e26f.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/tables/27b2b9d9728d4f432924a3e06aa7d4223367cb128c9ba024bbfd8b2c56d1e26f.jpg)

![2ff19fa4ecaa96fd94103effdd481d630c42ed58ac02c5fdb6e358abd9ab4ce6.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/tables/2ff19fa4ecaa96fd94103effdd481d630c42ed58ac02c5fdb6e358abd9ab4ce6.jpg)

![302f0bc81c53e8015ebc27247631a5524540c7f7af174bc8716984dc16583958.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/tables/302f0bc81c53e8015ebc27247631a5524540c7f7af174bc8716984dc16583958.jpg)

![43c8c624c0653452ba97a557a0bce98c1ca13b9b8d891c29a210136a52fad28e.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/tables/43c8c624c0653452ba97a557a0bce98c1ca13b9b8d891c29a210136a52fad28e.jpg)

![55b2346cbbf24530553434967e5f19aa50a528e356f2b138803ba1a0edee0b27.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/tables/55b2346cbbf24530553434967e5f19aa50a528e356f2b138803ba1a0edee0b27.jpg)

![5b8f673bff9388e9bb50e31a41ed68fa699227bf26abcc8483380d12802055af.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/tables/5b8f673bff9388e9bb50e31a41ed68fa699227bf26abcc8483380d12802055af.jpg)

![5dd3f1b9c114d5a1bc4046fdfe3217587ba7f2d60bde99a3b3d716a33f152d00.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/tables/5dd3f1b9c114d5a1bc4046fdfe3217587ba7f2d60bde99a3b3d716a33f152d00.jpg)

![8307a09ebecbba2a8b4307205b71bf2976613b4d5ee9a811c362a5b25da03835.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/tables/8307a09ebecbba2a8b4307205b71bf2976613b4d5ee9a811c362a5b25da03835.jpg)

![8d5445b71933c0b98fb1a9a32958badd97cf1deb216c12ab7e7d9ce5b9e8323b.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/tables/8d5445b71933c0b98fb1a9a32958badd97cf1deb216c12ab7e7d9ce5b9e8323b.jpg)

![abd41aa5aa7ac823b7edcdb09e6e254211f2492bd62c62866ec0766a7105ed4d.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/tables/abd41aa5aa7ac823b7edcdb09e6e254211f2492bd62c62866ec0766a7105ed4d.jpg)

![b49f26bbad66db66fb93996c1fc1c1f4e4b411f118ce545043cd9a270b44c78c.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/tables/b49f26bbad66db66fb93996c1fc1c1f4e4b411f118ce545043cd9a270b44c78c.jpg)

![b9e6ccb5634b508d42db74829407ae9c3cdde52ed52974f15156a7c929109384.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/tables/b9e6ccb5634b508d42db74829407ae9c3cdde52ed52974f15156a7c929109384.jpg)

![c0ada58d85b44dd6414e53916c12dc5377891894d5d44b270849b28af8ac788f.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/tables/c0ada58d85b44dd6414e53916c12dc5377891894d5d44b270849b28af8ac788f.jpg)

![d8ea807c2737a0ddf5b7c0dc1c0a380d780688f9e3a542f6bf8affdbcd59ba6e.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/tables/d8ea807c2737a0ddf5b7c0dc1c0a380d780688f9e3a542f6bf8affdbcd59ba6e.jpg)

![f43c342a66459b27c2b7272d9e4e115b76cd04ae1ae1b1481b6420628d4b9591.jpg](../iclr_results/1928_Federated Few-Shot Class-Incremental Learning/tables/f43c342a66459b27c2b7272d9e4e115b76cd04ae1ae1b1481b6420628d4b9591.jpg)

## Theory, Analysis, and Best Practices for Sigmoid Self-Attention


### Images

![05fdd89cec724813c3d6d660e4409cf3f4732646dcd1bb25363e13060282a722.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/05fdd89cec724813c3d6d660e4409cf3f4732646dcd1bb25363e13060282a722.jpg)

![0b5c8f5006233f692afbd3a96a67275e8f17be50c9eb4c19678fed41c87ff0ac.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/0b5c8f5006233f692afbd3a96a67275e8f17be50c9eb4c19678fed41c87ff0ac.jpg)

![12be30f6d3144c5360e117f0b40d88c3b5f716cbcd4820aa6e3e585438d593ac.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/12be30f6d3144c5360e117f0b40d88c3b5f716cbcd4820aa6e3e585438d593ac.jpg)

![203b2c733e350f7e42783dc2577c9ad452ec5646df38a92e7ce64a0d6f0b3887.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/203b2c733e350f7e42783dc2577c9ad452ec5646df38a92e7ce64a0d6f0b3887.jpg)

![2d2076008a4c6a293d6eabf21cc83b9fcd9bf5d5303bc67d14567728c65c7490.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/2d2076008a4c6a293d6eabf21cc83b9fcd9bf5d5303bc67d14567728c65c7490.jpg)

![349d6cc5ad55fac42c4f223e824bac42a8aff976b655e4df0d28b70e0b29dd62.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/349d6cc5ad55fac42c4f223e824bac42a8aff976b655e4df0d28b70e0b29dd62.jpg)

![34bb0ee3ea55bfeb52492e62ad69963f354eb098786678dcebb4df9b6fb19fed.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/34bb0ee3ea55bfeb52492e62ad69963f354eb098786678dcebb4df9b6fb19fed.jpg)

![34e55407cc25525da676e1594d16bff78dbc8b2ab9022dfd8eb30c231a8377ad.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/34e55407cc25525da676e1594d16bff78dbc8b2ab9022dfd8eb30c231a8377ad.jpg)

![3d1e91bfc2324186dff8ba333d4f426748c95122ae2709685b6b43171da1b32d.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/3d1e91bfc2324186dff8ba333d4f426748c95122ae2709685b6b43171da1b32d.jpg)

![451a1307cc602872c5ab3243168e9fbcc760974706f4aac19db5c2f1bb96c017.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/451a1307cc602872c5ab3243168e9fbcc760974706f4aac19db5c2f1bb96c017.jpg)

![4ca371c9f9bad36cfbbfae56d5400fd5c991daf83b7c3f672c7170652b880384.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/4ca371c9f9bad36cfbbfae56d5400fd5c991daf83b7c3f672c7170652b880384.jpg)

![518bb0c4fe52b9f5a9b596080d4fa1b0e3dfcd88d173a6bbb9be51b7fb84f62a.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/518bb0c4fe52b9f5a9b596080d4fa1b0e3dfcd88d173a6bbb9be51b7fb84f62a.jpg)

![5553eb9111de2d02ef42862d4e1d9447452590c446d6e37cec6709a80655c26f.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/5553eb9111de2d02ef42862d4e1d9447452590c446d6e37cec6709a80655c26f.jpg)

![5b8ab40f3d1856a1c87a7b78a0b2d64421a27bfefa03f8eb45cc4793d9277dc2.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/5b8ab40f3d1856a1c87a7b78a0b2d64421a27bfefa03f8eb45cc4793d9277dc2.jpg)

![5d0276f1255909ebfdf84f42b85504b2c4b1b2d7f6c3c4911f902dbf7507819b.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/5d0276f1255909ebfdf84f42b85504b2c4b1b2d7f6c3c4911f902dbf7507819b.jpg)

![5f86e291820cc0809d960dce04a476fed862211a2d31ff3b299f2e2108f09492.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/5f86e291820cc0809d960dce04a476fed862211a2d31ff3b299f2e2108f09492.jpg)

![63aa2cec9906b225af391bbbc6cf0a6c4b82c063dd8f10701b26eb02b4c8b95b.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/63aa2cec9906b225af391bbbc6cf0a6c4b82c063dd8f10701b26eb02b4c8b95b.jpg)

![66b62326d4cb0c9a4ddd336cc066929090c67decb293c44d4c5891c84e933ccc.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/66b62326d4cb0c9a4ddd336cc066929090c67decb293c44d4c5891c84e933ccc.jpg)

![6ac948f57e7f0e7f0255fb9c845abcb4f524aab6d35e3d13f8f883551a7090cd.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/6ac948f57e7f0e7f0255fb9c845abcb4f524aab6d35e3d13f8f883551a7090cd.jpg)

![6e6ad6b5d42182812eb27c9d4f4d79c0cc520b92447b4d9991958ccc346b5e0e.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/6e6ad6b5d42182812eb27c9d4f4d79c0cc520b92447b4d9991958ccc346b5e0e.jpg)

![79760b5d5ecbbbef51a85448109a5635c70291f0dc94ed7ef94857cfa7f95ba5.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/79760b5d5ecbbbef51a85448109a5635c70291f0dc94ed7ef94857cfa7f95ba5.jpg)

![81c21257d0e593dde56a9fa986adcb4cdd1f25a765fc57b3b209ada4ebdf1fbe.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/81c21257d0e593dde56a9fa986adcb4cdd1f25a765fc57b3b209ada4ebdf1fbe.jpg)

![828ea082eb103e545727b4fca5da63e18f66a6e76ec01b4b00e3926d0c7c45ad.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/828ea082eb103e545727b4fca5da63e18f66a6e76ec01b4b00e3926d0c7c45ad.jpg)

![874fa8b32c0d487b98e078ebcf24f61c1d306253e16f069c96a12e997bee435e.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/874fa8b32c0d487b98e078ebcf24f61c1d306253e16f069c96a12e997bee435e.jpg)

![87f3cd6998f92a5f24134b925aa540485cec49d00641a69894f958ac96c70463.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/87f3cd6998f92a5f24134b925aa540485cec49d00641a69894f958ac96c70463.jpg)

![b695778483713ddcca846bfa1815d599b6ef4a612e1baed2ed6ac31904acb3ca.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/b695778483713ddcca846bfa1815d599b6ef4a612e1baed2ed6ac31904acb3ca.jpg)

![c66cadd61a5395ffafca236a79036e038e32ed6f20de1256b677e9a7acfc141f.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/c66cadd61a5395ffafca236a79036e038e32ed6f20de1256b677e9a7acfc141f.jpg)

![caefa65200d8acdff7f16f05ddc49bedd2e3233a9f6378112d2eea34a1f30099.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/caefa65200d8acdff7f16f05ddc49bedd2e3233a9f6378112d2eea34a1f30099.jpg)

![d2aaa4fd02c255390d22c23a75d86a52ffc5c41727e973605e8c2838e19ee7f6.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/d2aaa4fd02c255390d22c23a75d86a52ffc5c41727e973605e8c2838e19ee7f6.jpg)

![e48a206f792e6945424cc69155e3fb286f9ca10b68e39ddb88de85c6852adeb3.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/e48a206f792e6945424cc69155e3fb286f9ca10b68e39ddb88de85c6852adeb3.jpg)

![e4c07424b74b0633f5ce61e1f27f9917f3ec9b73021bc9ab30666ad0e2ecf56f.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/e4c07424b74b0633f5ce61e1f27f9917f3ec9b73021bc9ab30666ad0e2ecf56f.jpg)

![e637cfe6d20add9cb002032ff09ae449c64925beb0db60d1e85840a1cf000d52.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/e637cfe6d20add9cb002032ff09ae449c64925beb0db60d1e85840a1cf000d52.jpg)

![f4b9ef02d0ca8fc5a375157ae001be94c80af38947ea9a533e64826cfe01dfb1.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/f4b9ef02d0ca8fc5a375157ae001be94c80af38947ea9a533e64826cfe01dfb1.jpg)

![fd7d446270ead975e2080c5a4a36b7c92b82f27d4dc09618673ed1ed9276262f.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/images/fd7d446270ead975e2080c5a4a36b7c92b82f27d4dc09618673ed1ed9276262f.jpg)

### Tables

![035a310cb7e97e6fbc8a86bab74d7c0964b4ccd294a7c4fd056db5da7641269a.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/tables/035a310cb7e97e6fbc8a86bab74d7c0964b4ccd294a7c4fd056db5da7641269a.jpg)

![1a53b0ccbe039ebf8cf1f4b9d2eea370186ec44546911144f5de1306c7e70093.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/tables/1a53b0ccbe039ebf8cf1f4b9d2eea370186ec44546911144f5de1306c7e70093.jpg)

![25a46dbd645d3e0de2e53bda5bd39a541b8b65d4ed3514212336bc62818aeac1.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/tables/25a46dbd645d3e0de2e53bda5bd39a541b8b65d4ed3514212336bc62818aeac1.jpg)

![3ced835519c2dc97adb07511b9fa15ba8bda4fa8883393932a715465f349ba60.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/tables/3ced835519c2dc97adb07511b9fa15ba8bda4fa8883393932a715465f349ba60.jpg)

![506247290e21d146a1500a6bbbb3c3e0f242d13abd811433ef6f2be27946c932.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/tables/506247290e21d146a1500a6bbbb3c3e0f242d13abd811433ef6f2be27946c932.jpg)

![5d13b69d1dfbeb45116d518b80fd890df23b14dcac73dba36b8561fc2176bbfe.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/tables/5d13b69d1dfbeb45116d518b80fd890df23b14dcac73dba36b8561fc2176bbfe.jpg)

![61fadb058e1c935b6146fc2cd4dd4def114409bf71e6a9a53960983cc11b49df.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/tables/61fadb058e1c935b6146fc2cd4dd4def114409bf71e6a9a53960983cc11b49df.jpg)

![682fc2d360e7309f951c1c3a488a8a02f58ea1fce0d3d1def7ec9919d32f9793.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/tables/682fc2d360e7309f951c1c3a488a8a02f58ea1fce0d3d1def7ec9919d32f9793.jpg)

![7b7c6f2dcd30a9e36964ef5eb3889e2683bb488ad205461c78370e24ee11fe75.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/tables/7b7c6f2dcd30a9e36964ef5eb3889e2683bb488ad205461c78370e24ee11fe75.jpg)

![7c890daa86de4a68881b5a26eec20cbae1d10671570aae4906705dd4e7a51abe.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/tables/7c890daa86de4a68881b5a26eec20cbae1d10671570aae4906705dd4e7a51abe.jpg)

![7df9cdbf213808f0429833e78e7af8145282121f01d5eb53c9af365fd9626413.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/tables/7df9cdbf213808f0429833e78e7af8145282121f01d5eb53c9af365fd9626413.jpg)

![8c17345b3f386f13315d4a55ea088ab9315105e474e25860810a4a02f9676fef.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/tables/8c17345b3f386f13315d4a55ea088ab9315105e474e25860810a4a02f9676fef.jpg)

![9e7cdd04f34eb47319a3749a2e36c8c40d49e79a3b25034ad38068e49091d4c0.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/tables/9e7cdd04f34eb47319a3749a2e36c8c40d49e79a3b25034ad38068e49091d4c0.jpg)

![a8d3818d4b36a7530023ce1cbd74a0a3f4f448123cc89d81e53d7e596b3c632e.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/tables/a8d3818d4b36a7530023ce1cbd74a0a3f4f448123cc89d81e53d7e596b3c632e.jpg)

![aa8a352a3203208b7ca19dacd7945917cc95a6174278f42f235fa56208eb4f6b.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/tables/aa8a352a3203208b7ca19dacd7945917cc95a6174278f42f235fa56208eb4f6b.jpg)

![b66bb9c7105ecbd93cfddadc29ea3eb0e45ee989bee205b72fec217f3923ebf7.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/tables/b66bb9c7105ecbd93cfddadc29ea3eb0e45ee989bee205b72fec217f3923ebf7.jpg)

![c6ca544ef10c60b430e6a2285a615a37a8bf7829f088cb3ba2da849f79fc0d3a.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/tables/c6ca544ef10c60b430e6a2285a615a37a8bf7829f088cb3ba2da849f79fc0d3a.jpg)

![d813f310e0314186912cd471b6df78ce818c1cac11506eacd7f4ea6d2b83e552.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/tables/d813f310e0314186912cd471b6df78ce818c1cac11506eacd7f4ea6d2b83e552.jpg)

![f199fdc73397ff81074a591f3b7ed7d783e219b0f3856e05303a590b23e4d78b.jpg](../iclr_results/1929_Theory, Analysis, and Best Practices for Sigmoid Self-Attention/tables/f199fdc73397ff81074a591f3b7ed7d783e219b0f3856e05303a590b23e4d78b.jpg)

## KLay: Accelerating Arithmetic Circuits for Neurosymbolic AI


### Images

![2b304592985d8fc48e4990b639b2898e9c3673042d238104459ce15dc5f97db7.jpg](../iclr_results/1930_KLay_ Accelerating Arithmetic Circuits for Neurosymbolic AI/images/2b304592985d8fc48e4990b639b2898e9c3673042d238104459ce15dc5f97db7.jpg)

![617ad0c20900920829dbe3eef80a8830e8cde692e7cfdae4c5d6f1f7a8403f1a.jpg](../iclr_results/1930_KLay_ Accelerating Arithmetic Circuits for Neurosymbolic AI/images/617ad0c20900920829dbe3eef80a8830e8cde692e7cfdae4c5d6f1f7a8403f1a.jpg)

![6991acbed39b981c4c85271af1d99bb868fb247cf42034d68992b95a4e12c7d1.jpg](../iclr_results/1930_KLay_ Accelerating Arithmetic Circuits for Neurosymbolic AI/images/6991acbed39b981c4c85271af1d99bb868fb247cf42034d68992b95a4e12c7d1.jpg)

![72c9a4797e00836d585f009325a2302ab9d311b419a2fde411e350fa603125cd.jpg](../iclr_results/1930_KLay_ Accelerating Arithmetic Circuits for Neurosymbolic AI/images/72c9a4797e00836d585f009325a2302ab9d311b419a2fde411e350fa603125cd.jpg)

![79be65d57c5a591981169f17cd9259bc9c526cc4a7722070c58fe1e9d83b557b.jpg](../iclr_results/1930_KLay_ Accelerating Arithmetic Circuits for Neurosymbolic AI/images/79be65d57c5a591981169f17cd9259bc9c526cc4a7722070c58fe1e9d83b557b.jpg)

![8e9a64c8f8a6fecc04333dc0e7c4c127801b9f55d1e63801df9b45ec30f6aaca.jpg](../iclr_results/1930_KLay_ Accelerating Arithmetic Circuits for Neurosymbolic AI/images/8e9a64c8f8a6fecc04333dc0e7c4c127801b9f55d1e63801df9b45ec30f6aaca.jpg)

![a8432482ee5714f20e2fc500a15ed7ef0d329442db8817d8a7d06ae064a3c891.jpg](../iclr_results/1930_KLay_ Accelerating Arithmetic Circuits for Neurosymbolic AI/images/a8432482ee5714f20e2fc500a15ed7ef0d329442db8817d8a7d06ae064a3c891.jpg)

![cfb9e46c86eaa64efbbbb00418b623a810a89cff8f0e15a2f23b7319b408249c.jpg](../iclr_results/1930_KLay_ Accelerating Arithmetic Circuits for Neurosymbolic AI/images/cfb9e46c86eaa64efbbbb00418b623a810a89cff8f0e15a2f23b7319b408249c.jpg)

![f8c81423628f1a82e011ca3460af250cbf2be18b7b7eb2f36f8253fc04088968.jpg](../iclr_results/1930_KLay_ Accelerating Arithmetic Circuits for Neurosymbolic AI/images/f8c81423628f1a82e011ca3460af250cbf2be18b7b7eb2f36f8253fc04088968.jpg)

### Tables

![342689a22585de7343d1bbb5d775a9fca0a3ef447dcb5a0db612e03ac6e75b40.jpg](../iclr_results/1930_KLay_ Accelerating Arithmetic Circuits for Neurosymbolic AI/tables/342689a22585de7343d1bbb5d775a9fca0a3ef447dcb5a0db612e03ac6e75b40.jpg)

![b316983c49f22bd40f652a9e52dc1d5a80aeee55a3816470aa11dd8bc9be0593.jpg](../iclr_results/1930_KLay_ Accelerating Arithmetic Circuits for Neurosymbolic AI/tables/b316983c49f22bd40f652a9e52dc1d5a80aeee55a3816470aa11dd8bc9be0593.jpg)

![ff3a2b797dbb0da1f8668e8f90ec4e8e3368c599ca6bfe2fd87d36cf71a049d0.jpg](../iclr_results/1930_KLay_ Accelerating Arithmetic Circuits for Neurosymbolic AI/tables/ff3a2b797dbb0da1f8668e8f90ec4e8e3368c599ca6bfe2fd87d36cf71a049d0.jpg)

## Add-it: Training-Free Object Insertion in Images With Pretrained Diffusion Models


### Images

![149e157a24329378b14172211ea012ff7ee6754e6ac5094b73e24b70eab59783.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/images/149e157a24329378b14172211ea012ff7ee6754e6ac5094b73e24b70eab59783.jpg)

![1cd69ecc19c053349b4d47de16809972c7f50838979d5b73860eae6b0ed044eb.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/images/1cd69ecc19c053349b4d47de16809972c7f50838979d5b73860eae6b0ed044eb.jpg)

![25c45489657af8a2793fdd90741785ffd519357e96d1af5194ab41e59070918d.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/images/25c45489657af8a2793fdd90741785ffd519357e96d1af5194ab41e59070918d.jpg)

![2be4e0958ddce15a79934fdc23df86c85dab842e3e76ba8f9b9c6817103bd364.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/images/2be4e0958ddce15a79934fdc23df86c85dab842e3e76ba8f9b9c6817103bd364.jpg)

![32e93f95c5791d93e2e4b18b14b85d65619e84622b7fdf80d2768bc3f59175c4.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/images/32e93f95c5791d93e2e4b18b14b85d65619e84622b7fdf80d2768bc3f59175c4.jpg)

![5156fe1a1757353538be4940841ca36fd9d24060d0e2b6568e76d52bcc35b9d0.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/images/5156fe1a1757353538be4940841ca36fd9d24060d0e2b6568e76d52bcc35b9d0.jpg)

![5dd226c289cb82f23b7dede90107b3d6eaa807d6cb5e2ab13ccc9e6b072f7c57.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/images/5dd226c289cb82f23b7dede90107b3d6eaa807d6cb5e2ab13ccc9e6b072f7c57.jpg)

![69ebaf582b137cd20011683927613a64c5c7bc9b05f4a25ce1c2353ebfd363ff.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/images/69ebaf582b137cd20011683927613a64c5c7bc9b05f4a25ce1c2353ebfd363ff.jpg)

![6dc066338203e7b4fbe95949fea198eea67e7343f698a08098f934f1f54c7df8.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/images/6dc066338203e7b4fbe95949fea198eea67e7343f698a08098f934f1f54c7df8.jpg)

![725b0176d77fac0b900b141be387e321018d1741d64d575f8fada717dea68699.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/images/725b0176d77fac0b900b141be387e321018d1741d64d575f8fada717dea68699.jpg)

![795b72f572d63085943ac4bd492489c8eaffcdc8879115c2b554b22ce17dd772.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/images/795b72f572d63085943ac4bd492489c8eaffcdc8879115c2b554b22ce17dd772.jpg)

![7bfae77880920551235ce73bddcf8d4be89e8a58730183d7ebb48cfa62d26b43.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/images/7bfae77880920551235ce73bddcf8d4be89e8a58730183d7ebb48cfa62d26b43.jpg)

![7d0a45777ddf426ccde1599f606acb5b7926f7835ea6e5f3546ffd6358be9402.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/images/7d0a45777ddf426ccde1599f606acb5b7926f7835ea6e5f3546ffd6358be9402.jpg)

![830cc3173eb627c6e368d6b89a961af1be3184699c91f03270af8f88bba4ead0.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/images/830cc3173eb627c6e368d6b89a961af1be3184699c91f03270af8f88bba4ead0.jpg)

![b41152338a957d5029e0c5478c283e76ac844e6ae3201bf368839da753d878e6.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/images/b41152338a957d5029e0c5478c283e76ac844e6ae3201bf368839da753d878e6.jpg)

![c7bdfe22f80cfe77daa642bc36572bd874074b186bd2c904beb5b92e0a363480.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/images/c7bdfe22f80cfe77daa642bc36572bd874074b186bd2c904beb5b92e0a363480.jpg)

![c9a577a2560c5bef24c638c1db19bc0d6756cf8f2018e83516177d5c48d9647e.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/images/c9a577a2560c5bef24c638c1db19bc0d6756cf8f2018e83516177d5c48d9647e.jpg)

![d67d78f0b463bcdf380435d9d59e2cde49d0bdf9529e00d70cd9cf64f97d9249.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/images/d67d78f0b463bcdf380435d9d59e2cde49d0bdf9529e00d70cd9cf64f97d9249.jpg)

![e7af4a9d09ac87ed9d7434caf2c12d6bed3be22b0b2b1e72da43aba97aeba140.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/images/e7af4a9d09ac87ed9d7434caf2c12d6bed3be22b0b2b1e72da43aba97aeba140.jpg)

![f4150444cc56c366352ba18400c803061c396773eea7e75389ba9c8db53d2690.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/images/f4150444cc56c366352ba18400c803061c396773eea7e75389ba9c8db53d2690.jpg)

### Tables

![0b2137f31925cb96c0fb28b55441fd4c155160d8f7d9a8484271b6e2c273da01.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/tables/0b2137f31925cb96c0fb28b55441fd4c155160d8f7d9a8484271b6e2c273da01.jpg)

![475723186d42a0d333f80fc4f992b0d96ca3ac1cb0047e222eb49ad19f8b3449.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/tables/475723186d42a0d333f80fc4f992b0d96ca3ac1cb0047e222eb49ad19f8b3449.jpg)

![4b26502343b3dd2d6f191bda9d685ba74e8edc3c7cb415f252d64decb5b101a1.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/tables/4b26502343b3dd2d6f191bda9d685ba74e8edc3c7cb415f252d64decb5b101a1.jpg)

![96a1e4d3456b0d6751c73402307e716af76d6b197dde557f8a8766f9c190326d.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/tables/96a1e4d3456b0d6751c73402307e716af76d6b197dde557f8a8766f9c190326d.jpg)

![a1d270c7b94579501ae63cfaacdd2f9f0fa87562d1f9612d5453383d12150c7e.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/tables/a1d270c7b94579501ae63cfaacdd2f9f0fa87562d1f9612d5453383d12150c7e.jpg)

![a1dc93c1d238ce842a7ac4d96cc313dfd18e2f27ce9e52c6ff0c0a2b7a539428.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/tables/a1dc93c1d238ce842a7ac4d96cc313dfd18e2f27ce9e52c6ff0c0a2b7a539428.jpg)

![c44c4889a5c959a70b9d80d042b7df5cd87383c269b7da420d4ec9ce28e44d35.jpg](../iclr_results/1931_Add-it_ Training-Free Object Insertion in Images With Pretrained Diffusion Models/tables/c44c4889a5c959a70b9d80d042b7df5cd87383c269b7da420d4ec9ce28e44d35.jpg)

## Step-by-Step Reasoning for Math Problems  via Twisted Sequential Monte Carlo


### Images

![082ec02a7f42c82ee81e53338d504020b2f7f7ae3e499f89e1c0fba3b9140629.jpg](../iclr_results/1932_Step-by-Step Reasoning for Math Problems  via Twisted Sequential Monte Carlo/images/082ec02a7f42c82ee81e53338d504020b2f7f7ae3e499f89e1c0fba3b9140629.jpg)

![8938ca8096bfbae4c7544ecfb0eb83acfc79b810e317e94763c188d99c4a1125.jpg](../iclr_results/1932_Step-by-Step Reasoning for Math Problems  via Twisted Sequential Monte Carlo/images/8938ca8096bfbae4c7544ecfb0eb83acfc79b810e317e94763c188d99c4a1125.jpg)

![983b342569729bf926390a69be00ae03d671be504b6892f11e7a66276d8f1958.jpg](../iclr_results/1932_Step-by-Step Reasoning for Math Problems  via Twisted Sequential Monte Carlo/images/983b342569729bf926390a69be00ae03d671be504b6892f11e7a66276d8f1958.jpg)

![b404a18ffa47bf09c68894b7bd94df195891bb4a35c2c8a5c3f0ee0bacb1af88.jpg](../iclr_results/1932_Step-by-Step Reasoning for Math Problems  via Twisted Sequential Monte Carlo/images/b404a18ffa47bf09c68894b7bd94df195891bb4a35c2c8a5c3f0ee0bacb1af88.jpg)

![ec76ce07c9374591f5b31059bf00022905a7b50c71f39757493cd97c5313260c.jpg](../iclr_results/1932_Step-by-Step Reasoning for Math Problems  via Twisted Sequential Monte Carlo/images/ec76ce07c9374591f5b31059bf00022905a7b50c71f39757493cd97c5313260c.jpg)

### Tables

![01b4b1a71b40500aa38f938637c4461240c62f316193e60b84d61c4fda035f07.jpg](../iclr_results/1932_Step-by-Step Reasoning for Math Problems  via Twisted Sequential Monte Carlo/tables/01b4b1a71b40500aa38f938637c4461240c62f316193e60b84d61c4fda035f07.jpg)

![3059c79c079d2ac58c3f67243214e41cafa82b5ed4b31476796eb54855184494.jpg](../iclr_results/1932_Step-by-Step Reasoning for Math Problems  via Twisted Sequential Monte Carlo/tables/3059c79c079d2ac58c3f67243214e41cafa82b5ed4b31476796eb54855184494.jpg)

![684623aa985f96585af1e9e6a7ccc11a2b10221110e204e7185f722328ac3246.jpg](../iclr_results/1932_Step-by-Step Reasoning for Math Problems  via Twisted Sequential Monte Carlo/tables/684623aa985f96585af1e9e6a7ccc11a2b10221110e204e7185f722328ac3246.jpg)

![8ab0394ca79ff43a934ca00339b2a4f76dec5adbbb9f671e7ccd13c0623c8df3.jpg](../iclr_results/1932_Step-by-Step Reasoning for Math Problems  via Twisted Sequential Monte Carlo/tables/8ab0394ca79ff43a934ca00339b2a4f76dec5adbbb9f671e7ccd13c0623c8df3.jpg)

![8e7d197bdc29d77c2edadd09048864be305ed951ddc55679121cd1cd5a316811.jpg](../iclr_results/1932_Step-by-Step Reasoning for Math Problems  via Twisted Sequential Monte Carlo/tables/8e7d197bdc29d77c2edadd09048864be305ed951ddc55679121cd1cd5a316811.jpg)

![98a22560cd44b0dbc67bdccc12a6f97c81dea8aa0516ac25544c319db1eb0a85.jpg](../iclr_results/1932_Step-by-Step Reasoning for Math Problems  via Twisted Sequential Monte Carlo/tables/98a22560cd44b0dbc67bdccc12a6f97c81dea8aa0516ac25544c319db1eb0a85.jpg)

![9a7502b4eaf38e01ab292684dfa1652c82f1d647075c971402adda424a61184b.jpg](../iclr_results/1932_Step-by-Step Reasoning for Math Problems  via Twisted Sequential Monte Carlo/tables/9a7502b4eaf38e01ab292684dfa1652c82f1d647075c971402adda424a61184b.jpg)

## ASTrA: Adversarial Self-supervised Training with Adaptive-Attacks


### Images

![14c5518d12a2e47b1cf9f16d83cdb23a1b8c2f53ea8603f025708ca525094fa6.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/images/14c5518d12a2e47b1cf9f16d83cdb23a1b8c2f53ea8603f025708ca525094fa6.jpg)

![4166d844043f766a9eda47c5f1bbe0fb5c7e4d271c7eafddd7b1c01f7b63ac71.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/images/4166d844043f766a9eda47c5f1bbe0fb5c7e4d271c7eafddd7b1c01f7b63ac71.jpg)

![46d4a7e3bb564d47c453a36b2f68164f28beed018c03e676ad2ca2a18acd8850.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/images/46d4a7e3bb564d47c453a36b2f68164f28beed018c03e676ad2ca2a18acd8850.jpg)

![4738109a743fe603e50f84c084eec390763b6a774e49c2b62fd422b2dd7532e6.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/images/4738109a743fe603e50f84c084eec390763b6a774e49c2b62fd422b2dd7532e6.jpg)

![5ec3ae9c09d5cb53eb1256023be8c2b54a8bf120a7b498c8ba4ef06fa9465a3d.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/images/5ec3ae9c09d5cb53eb1256023be8c2b54a8bf120a7b498c8ba4ef06fa9465a3d.jpg)

![62b370bd188cc7ce936264764904ee0570bbd66898d0e8cb43da3c6682317f55.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/images/62b370bd188cc7ce936264764904ee0570bbd66898d0e8cb43da3c6682317f55.jpg)

![685e1d94786a080115d9ec33391ec0ddd4a2678a8b46a683dfe2edb00977bce2.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/images/685e1d94786a080115d9ec33391ec0ddd4a2678a8b46a683dfe2edb00977bce2.jpg)

![6ef85ebc7231595cd5ae3a67045cc0454e1509c039f18f36dd54acaace20aebc.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/images/6ef85ebc7231595cd5ae3a67045cc0454e1509c039f18f36dd54acaace20aebc.jpg)

![9c67d45c794ebb7adb672c0cdd573e9d1b4e8fea072c84f603916f15e0088a7a.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/images/9c67d45c794ebb7adb672c0cdd573e9d1b4e8fea072c84f603916f15e0088a7a.jpg)

![b16124b5230e45e15d8903ef0156237a1c021fb0c0b3d2679a00ef0c1beeced3.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/images/b16124b5230e45e15d8903ef0156237a1c021fb0c0b3d2679a00ef0c1beeced3.jpg)

![b8bc7148203e1b567a2889e0321144598081375a3ada55f3a6a2039211ab2848.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/images/b8bc7148203e1b567a2889e0321144598081375a3ada55f3a6a2039211ab2848.jpg)

![fdc9d0995982a72b743571da4350854a47182a67b5cb14f21c3be8dfa1007eb8.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/images/fdc9d0995982a72b743571da4350854a47182a67b5cb14f21c3be8dfa1007eb8.jpg)

### Tables

![02037df055ce9ce4394ccd0054a755cc3477705c3ae8d7eda8e916916db0eaef.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/tables/02037df055ce9ce4394ccd0054a755cc3477705c3ae8d7eda8e916916db0eaef.jpg)

![2ba88a516b3d762aa74a11aec1f308ac4ac04b6d6562dad9b9d4dcbaf848f3e0.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/tables/2ba88a516b3d762aa74a11aec1f308ac4ac04b6d6562dad9b9d4dcbaf848f3e0.jpg)

![2d162dcbdcf51c01a7deb12cc00fe5a28af3f3a0966e9087460c588677676a9d.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/tables/2d162dcbdcf51c01a7deb12cc00fe5a28af3f3a0966e9087460c588677676a9d.jpg)

![343fc03e39ff713e48bf4d929e95415bec191fedddc8556524246a6596ceb93c.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/tables/343fc03e39ff713e48bf4d929e95415bec191fedddc8556524246a6596ceb93c.jpg)

![37bc60789d74edd427e7ff92fa95068b8b7ace1d5a6a54838f09bf61d3ac8811.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/tables/37bc60789d74edd427e7ff92fa95068b8b7ace1d5a6a54838f09bf61d3ac8811.jpg)

![4ba175725ecec65df03ccb2a8465bd8f1f0254e86f879e53fa7784ec3a5d21b0.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/tables/4ba175725ecec65df03ccb2a8465bd8f1f0254e86f879e53fa7784ec3a5d21b0.jpg)

![4c9715e72bd3233e160ebe848fd793c506a0e424f826b19119652ac0eecc2715.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/tables/4c9715e72bd3233e160ebe848fd793c506a0e424f826b19119652ac0eecc2715.jpg)

![695d70e98c7525005627718294ecea8c056b233b8517e11528afc65052318a30.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/tables/695d70e98c7525005627718294ecea8c056b233b8517e11528afc65052318a30.jpg)

![6e4c43071b25f43c555fb3656ead99c15affe3c8eedb62fe4b192edd5db7ef2a.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/tables/6e4c43071b25f43c555fb3656ead99c15affe3c8eedb62fe4b192edd5db7ef2a.jpg)

![9742eb376b7a9faada2fadb46c90dbd6e748dbb1677e115d7b9055e3fe5500ad.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/tables/9742eb376b7a9faada2fadb46c90dbd6e748dbb1677e115d7b9055e3fe5500ad.jpg)

![b2ff6b44bd06578498d5de957621c1d95fd610a0aea169c29ec6f33200a95a1a.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/tables/b2ff6b44bd06578498d5de957621c1d95fd610a0aea169c29ec6f33200a95a1a.jpg)

![c6f5fe84fd6b726b7ca23f0b66d0402e213cdd2e379ce18eb481023a82414937.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/tables/c6f5fe84fd6b726b7ca23f0b66d0402e213cdd2e379ce18eb481023a82414937.jpg)

![cf64d21bdc6eddae86f9cc7fe72419a370630401b609dade3005467c520a7554.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/tables/cf64d21bdc6eddae86f9cc7fe72419a370630401b609dade3005467c520a7554.jpg)

![de5ab650ba8ea5a5deaaa3296527de1cb42f9f6946873b882c04e80a9e07dc99.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/tables/de5ab650ba8ea5a5deaaa3296527de1cb42f9f6946873b882c04e80a9e07dc99.jpg)

![eda40df645aa2b149fd381a5fae43d6c47624dd7b1779d65760244c96ab98088.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/tables/eda40df645aa2b149fd381a5fae43d6c47624dd7b1779d65760244c96ab98088.jpg)

![fd06e29c21fcf09a0ae2d4a6060ce72c74dc972fa8b3561f33f476ec625c4bde.jpg](../iclr_results/1933_ASTrA_ Adversarial Self-supervised Training with Adaptive-Attacks/tables/fd06e29c21fcf09a0ae2d4a6060ce72c74dc972fa8b3561f33f476ec625c4bde.jpg)

## SpikeLLM: Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking


### Images

![216e648c5a14001b5cddb8dc80cc0fe39db696ee99fecaeca6bf18329a410332.jpg](../iclr_results/1934_SpikeLLM_ Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking/images/216e648c5a14001b5cddb8dc80cc0fe39db696ee99fecaeca6bf18329a410332.jpg)

![4c157d0473979aa3e4a506aa0f2729b1a35ba566153de2509550e1ac0443b1c0.jpg](../iclr_results/1934_SpikeLLM_ Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking/images/4c157d0473979aa3e4a506aa0f2729b1a35ba566153de2509550e1ac0443b1c0.jpg)

![7d398aabf44b5f607a5dec48b9b20dfed6f972c7cbe02e09271e305934ed9268.jpg](../iclr_results/1934_SpikeLLM_ Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking/images/7d398aabf44b5f607a5dec48b9b20dfed6f972c7cbe02e09271e305934ed9268.jpg)

![87773a3ee673b61be7c48c6ab315207065fdc1e2a6321941f75fa80b349095bc.jpg](../iclr_results/1934_SpikeLLM_ Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking/images/87773a3ee673b61be7c48c6ab315207065fdc1e2a6321941f75fa80b349095bc.jpg)

![c14e0edc085beebfe2ec6ab1253b6cab7fe4a9235de3f95d547a7236b304f8e2.jpg](../iclr_results/1934_SpikeLLM_ Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking/images/c14e0edc085beebfe2ec6ab1253b6cab7fe4a9235de3f95d547a7236b304f8e2.jpg)

![c1ff428d3c3220aceb1fa0bcd830b8874b36c6bb5b8a7e5cd998c2cfa6eebd29.jpg](../iclr_results/1934_SpikeLLM_ Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking/images/c1ff428d3c3220aceb1fa0bcd830b8874b36c6bb5b8a7e5cd998c2cfa6eebd29.jpg)

![c67ee068ded8b44c257e63da7c1c39ad24ac599e18352ab9089417a4ddf8b454.jpg](../iclr_results/1934_SpikeLLM_ Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking/images/c67ee068ded8b44c257e63da7c1c39ad24ac599e18352ab9089417a4ddf8b454.jpg)

![c8dbd47e9a9691833e0891fd3b4c37c1f543d74e08f14e1e0b0577f13e9dbb66.jpg](../iclr_results/1934_SpikeLLM_ Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking/images/c8dbd47e9a9691833e0891fd3b4c37c1f543d74e08f14e1e0b0577f13e9dbb66.jpg)

![cfbe46a1b5b672ba1da09b896e992baec80b5dc13fa3296dc48c2fb59856bedd.jpg](../iclr_results/1934_SpikeLLM_ Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking/images/cfbe46a1b5b672ba1da09b896e992baec80b5dc13fa3296dc48c2fb59856bedd.jpg)

### Tables

![299fdd8dec9a5f7d1fb42bbbb4b9eaa7f3a85dde425f5fbe8b694aa7a5762e2a.jpg](../iclr_results/1934_SpikeLLM_ Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking/tables/299fdd8dec9a5f7d1fb42bbbb4b9eaa7f3a85dde425f5fbe8b694aa7a5762e2a.jpg)

![4b3066230547c3a100d808dd021fbcb01c9b25c041629659dd33c3f22fda7359.jpg](../iclr_results/1934_SpikeLLM_ Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking/tables/4b3066230547c3a100d808dd021fbcb01c9b25c041629659dd33c3f22fda7359.jpg)

![52a5460e1b0d349e6fb0753137d5507694643a8aaafe8cd5246324d151b25782.jpg](../iclr_results/1934_SpikeLLM_ Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking/tables/52a5460e1b0d349e6fb0753137d5507694643a8aaafe8cd5246324d151b25782.jpg)

![535e1daa25f1b80b75fe3a5dcb445f6cad395ad8e3afbf59fafbe4a7ae323531.jpg](../iclr_results/1934_SpikeLLM_ Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking/tables/535e1daa25f1b80b75fe3a5dcb445f6cad395ad8e3afbf59fafbe4a7ae323531.jpg)

![5872bd879435faffdb34fec123b8b9a2f13c91f5c00cfb9d8d726131b526c553.jpg](../iclr_results/1934_SpikeLLM_ Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking/tables/5872bd879435faffdb34fec123b8b9a2f13c91f5c00cfb9d8d726131b526c553.jpg)

![6cb7a6df7dccaebc1e054ab4d7ca18c190b745ed8ec0467df20e73ba2da6c1cd.jpg](../iclr_results/1934_SpikeLLM_ Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking/tables/6cb7a6df7dccaebc1e054ab4d7ca18c190b745ed8ec0467df20e73ba2da6c1cd.jpg)

![82fe2d89a3cb2b6141f4f550873f51461db1554ba9da02969b4a5973ad156f51.jpg](../iclr_results/1934_SpikeLLM_ Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking/tables/82fe2d89a3cb2b6141f4f550873f51461db1554ba9da02969b4a5973ad156f51.jpg)

![89a9df42a0e32b98525dd7f23aa222f51b9624d2a62f60f031f30ef4824f0465.jpg](../iclr_results/1934_SpikeLLM_ Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking/tables/89a9df42a0e32b98525dd7f23aa222f51b9624d2a62f60f031f30ef4824f0465.jpg)

![9b43520e16fe8333cff4c198656f854ca4e9f7522e19101f7903ca02eefef39d.jpg](../iclr_results/1934_SpikeLLM_ Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking/tables/9b43520e16fe8333cff4c198656f854ca4e9f7522e19101f7903ca02eefef39d.jpg)

![a641734abfc2399f3a7517b3af351b8bed12ded8fb592b748bde8fc485bd8d88.jpg](../iclr_results/1934_SpikeLLM_ Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking/tables/a641734abfc2399f3a7517b3af351b8bed12ded8fb592b748bde8fc485bd8d88.jpg)

![b1bfae2044acfbd577b5a126e75c187479437708b450457e63e4ea3b1537e0d0.jpg](../iclr_results/1934_SpikeLLM_ Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking/tables/b1bfae2044acfbd577b5a126e75c187479437708b450457e63e4ea3b1537e0d0.jpg)

![cf88a1bf139f329231282e16bff1d2c7e6f4f2cfd149ccab4af7d38ede70adaf.jpg](../iclr_results/1934_SpikeLLM_ Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking/tables/cf88a1bf139f329231282e16bff1d2c7e6f4f2cfd149ccab4af7d38ede70adaf.jpg)

## Actions Speak Louder Than Words: Rate-Reward Trade-off in Markov Decision Processes


### Images

![1595718e1031a65bf16c18128bfecb226c583b140d79393ff0224d5216fa90af.jpg](../iclr_results/1935_Actions Speak Louder Than Words_ Rate-Reward Trade-off in Markov Decision Processes/images/1595718e1031a65bf16c18128bfecb226c583b140d79393ff0224d5216fa90af.jpg)

![1a57ace4d08ef500bed756c6a9216d1bdb28661e20b8e4fd7c1d94539410476f.jpg](../iclr_results/1935_Actions Speak Louder Than Words_ Rate-Reward Trade-off in Markov Decision Processes/images/1a57ace4d08ef500bed756c6a9216d1bdb28661e20b8e4fd7c1d94539410476f.jpg)

![32ca422bf9c48d8a97dc4abc10b07c6ff27c5ab31db04a7c8082ab5bfb351d6b.jpg](../iclr_results/1935_Actions Speak Louder Than Words_ Rate-Reward Trade-off in Markov Decision Processes/images/32ca422bf9c48d8a97dc4abc10b07c6ff27c5ab31db04a7c8082ab5bfb351d6b.jpg)

![38de84ce7b3ec54533aed08f4aa294abef5c27eeec091c87ef94bc690ac034f4.jpg](../iclr_results/1935_Actions Speak Louder Than Words_ Rate-Reward Trade-off in Markov Decision Processes/images/38de84ce7b3ec54533aed08f4aa294abef5c27eeec091c87ef94bc690ac034f4.jpg)

![41802b9e675afab939aeb52406323263c51d235030cb309b2311af8e34faa955.jpg](../iclr_results/1935_Actions Speak Louder Than Words_ Rate-Reward Trade-off in Markov Decision Processes/images/41802b9e675afab939aeb52406323263c51d235030cb309b2311af8e34faa955.jpg)

![48908114f0e8ff4bcac0912ed59fe551e29e22cc9bae969ce26539a1b1c30363.jpg](../iclr_results/1935_Actions Speak Louder Than Words_ Rate-Reward Trade-off in Markov Decision Processes/images/48908114f0e8ff4bcac0912ed59fe551e29e22cc9bae969ce26539a1b1c30363.jpg)

![4cdc9a48e80a3394ddbe230a3c1d0c22504b22ee041464507270aa99028ae328.jpg](../iclr_results/1935_Actions Speak Louder Than Words_ Rate-Reward Trade-off in Markov Decision Processes/images/4cdc9a48e80a3394ddbe230a3c1d0c22504b22ee041464507270aa99028ae328.jpg)

![6789b5991c573a6801382e6e36ae6b351c8d27507827f61ef20385f734dcb7d9.jpg](../iclr_results/1935_Actions Speak Louder Than Words_ Rate-Reward Trade-off in Markov Decision Processes/images/6789b5991c573a6801382e6e36ae6b351c8d27507827f61ef20385f734dcb7d9.jpg)

![7297f23039a3bfbe02e10766a708e2106259742376082b8c5ea15e6a35c248cf.jpg](../iclr_results/1935_Actions Speak Louder Than Words_ Rate-Reward Trade-off in Markov Decision Processes/images/7297f23039a3bfbe02e10766a708e2106259742376082b8c5ea15e6a35c248cf.jpg)

![7b6ae4bce9d2aeb63179aa8f544a975881cc632b22fa80ee2cb775526e21a5b9.jpg](../iclr_results/1935_Actions Speak Louder Than Words_ Rate-Reward Trade-off in Markov Decision Processes/images/7b6ae4bce9d2aeb63179aa8f544a975881cc632b22fa80ee2cb775526e21a5b9.jpg)

![a17217a9b3130a5ce78c14ce6a43a568f8274504e687fedffdbd09f9a70e3388.jpg](../iclr_results/1935_Actions Speak Louder Than Words_ Rate-Reward Trade-off in Markov Decision Processes/images/a17217a9b3130a5ce78c14ce6a43a568f8274504e687fedffdbd09f9a70e3388.jpg)

![cf65786105fc4fd5537c446fd06548dcd86712bc4291c7ee9fc2be8232d2d398.jpg](../iclr_results/1935_Actions Speak Louder Than Words_ Rate-Reward Trade-off in Markov Decision Processes/images/cf65786105fc4fd5537c446fd06548dcd86712bc4291c7ee9fc2be8232d2d398.jpg)

### Tables

![5d99a3f200ac56eed16d9a40441480ffb7b3a3dd1a4f577961343d6ddfe03c00.jpg](../iclr_results/1935_Actions Speak Louder Than Words_ Rate-Reward Trade-off in Markov Decision Processes/tables/5d99a3f200ac56eed16d9a40441480ffb7b3a3dd1a4f577961343d6ddfe03c00.jpg)

![721f3c80e3ffa3986479e03ee5abdc34363a1d9ffbd707796ad68b6f0ff24823.jpg](../iclr_results/1935_Actions Speak Louder Than Words_ Rate-Reward Trade-off in Markov Decision Processes/tables/721f3c80e3ffa3986479e03ee5abdc34363a1d9ffbd707796ad68b6f0ff24823.jpg)

![80b21070850f68a15808dd21f2b69c7a2d9c67e3fe9fa9c25a7c982a7d6a0066.jpg](../iclr_results/1935_Actions Speak Louder Than Words_ Rate-Reward Trade-off in Markov Decision Processes/tables/80b21070850f68a15808dd21f2b69c7a2d9c67e3fe9fa9c25a7c982a7d6a0066.jpg)

![a9ebfb9497d992e3a232f684d5c485f3e68ca4a6f24c24706c280281aa89c73b.jpg](../iclr_results/1935_Actions Speak Louder Than Words_ Rate-Reward Trade-off in Markov Decision Processes/tables/a9ebfb9497d992e3a232f684d5c485f3e68ca4a6f24c24706c280281aa89c73b.jpg)

![c254b2a2ffcca52e47478045adf524dafff3a2477d2554278a22d39a8cc5ba5c.jpg](../iclr_results/1935_Actions Speak Louder Than Words_ Rate-Reward Trade-off in Markov Decision Processes/tables/c254b2a2ffcca52e47478045adf524dafff3a2477d2554278a22d39a8cc5ba5c.jpg)

![d241188c2bf7e4d8ab2b4d1244367be48d21e992dae74d8bc59a6df2aee2c270.jpg](../iclr_results/1935_Actions Speak Louder Than Words_ Rate-Reward Trade-off in Markov Decision Processes/tables/d241188c2bf7e4d8ab2b4d1244367be48d21e992dae74d8bc59a6df2aee2c270.jpg)

![ee177e8637bb7d1c7fa38630ddf3d2ee62a17d37dccaa86bcd9b4b7f8e0cc23a.jpg](../iclr_results/1935_Actions Speak Louder Than Words_ Rate-Reward Trade-off in Markov Decision Processes/tables/ee177e8637bb7d1c7fa38630ddf3d2ee62a17d37dccaa86bcd9b4b7f8e0cc23a.jpg)

## Diff-2-in-1: Bridging Generation and Dense Perception with Diffusion Models


### Images

![06ae6b2cdad3232a93c5de851cb7c653c32f55d90b0a36cf34401b6b23fff517.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/images/06ae6b2cdad3232a93c5de851cb7c653c32f55d90b0a36cf34401b6b23fff517.jpg)

![0fa96a1eda8dc63075ee683574c8346a651a53ef1a68b2af18c61dd0dc833785.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/images/0fa96a1eda8dc63075ee683574c8346a651a53ef1a68b2af18c61dd0dc833785.jpg)

![25b2c711c889339da90611a313afcc72f4cf2a8dec517f7cb37be0b8fae2c89f.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/images/25b2c711c889339da90611a313afcc72f4cf2a8dec517f7cb37be0b8fae2c89f.jpg)

![2d2385e5293c9823371a0a8898c76a505e0e6309cc6abfe3ac7b5b3b82a0e680.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/images/2d2385e5293c9823371a0a8898c76a505e0e6309cc6abfe3ac7b5b3b82a0e680.jpg)

![3155f0fe43346e08657ac313ed70a4b0b2a771b6db031625cb6eef19f773a388.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/images/3155f0fe43346e08657ac313ed70a4b0b2a771b6db031625cb6eef19f773a388.jpg)

![5db5e88173bbcf94e29beff27eae700b44e80a73fa934c83f41252473d3e86ff.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/images/5db5e88173bbcf94e29beff27eae700b44e80a73fa934c83f41252473d3e86ff.jpg)

![60d71d5afc41f1a6b15fdb2a2ec0d815ae3ad8615d34be49f37fc7ea9623bc67.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/images/60d71d5afc41f1a6b15fdb2a2ec0d815ae3ad8615d34be49f37fc7ea9623bc67.jpg)

![6fc368507cdb8493f75cd7b270770ad7bf5c43d77e9001cd83e81498dfa69dd5.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/images/6fc368507cdb8493f75cd7b270770ad7bf5c43d77e9001cd83e81498dfa69dd5.jpg)

![71eaf606a1c9e803a47b0823dffe14e16f27d816f2ec2d5f34a6933dad97fba4.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/images/71eaf606a1c9e803a47b0823dffe14e16f27d816f2ec2d5f34a6933dad97fba4.jpg)

![78b12648acc9b3913ec634ac1c8f6630798a16e4c73bc94e63c637f18bd8dd06.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/images/78b12648acc9b3913ec634ac1c8f6630798a16e4c73bc94e63c637f18bd8dd06.jpg)

![7927cf7bb54798d7f65a3b04c361e148b2516d152df900ab2fb682e70464aec8.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/images/7927cf7bb54798d7f65a3b04c361e148b2516d152df900ab2fb682e70464aec8.jpg)

![947d56d04b76f2c3b2198326e3fbca0ebb641fdc683210bc92b77b4941960ef6.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/images/947d56d04b76f2c3b2198326e3fbca0ebb641fdc683210bc92b77b4941960ef6.jpg)

![bf4650132fdb677cdba640a72d1f8ecaed81485266b6429770c8354d5d0e5ff2.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/images/bf4650132fdb677cdba640a72d1f8ecaed81485266b6429770c8354d5d0e5ff2.jpg)

![d0b45e01ccb890937299e92dcfb381a4fda1aa63c6a7e157d70581eafdf0c5fa.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/images/d0b45e01ccb890937299e92dcfb381a4fda1aa63c6a7e157d70581eafdf0c5fa.jpg)

![fc1c4f8e19c8f8fde7dbb3a610792ba865a9123033c269e7dd89345582d41b40.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/images/fc1c4f8e19c8f8fde7dbb3a610792ba865a9123033c269e7dd89345582d41b40.jpg)

### Tables

![001826c4ec3b4bd60744d7f94657a92d4d78b2a2d91a02e712adb01d28a2ebc6.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/tables/001826c4ec3b4bd60744d7f94657a92d4d78b2a2d91a02e712adb01d28a2ebc6.jpg)

![31c8b363fbbfcbfa81d3e84911e3a806b7835a11cf7d67b5b4515e57c8f19570.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/tables/31c8b363fbbfcbfa81d3e84911e3a806b7835a11cf7d67b5b4515e57c8f19570.jpg)

![480a126d64ff84c11b4f5eb41f2813b06ec8f216aac01db4ab513dd0d749f55b.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/tables/480a126d64ff84c11b4f5eb41f2813b06ec8f216aac01db4ab513dd0d749f55b.jpg)

![4e0095f0b249e5253ca39da0908270df936896d28d0c82d32ef8b2f22a581290.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/tables/4e0095f0b249e5253ca39da0908270df936896d28d0c82d32ef8b2f22a581290.jpg)

![52d5444f2b695aa426801cb8ba2bf8e0776c05d4b5122dd7fd0c694841665298.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/tables/52d5444f2b695aa426801cb8ba2bf8e0776c05d4b5122dd7fd0c694841665298.jpg)

![52f2c18d1d89bf64173305b6c5daf3f19dbe80bfff599f029ad36eaffd589505.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/tables/52f2c18d1d89bf64173305b6c5daf3f19dbe80bfff599f029ad36eaffd589505.jpg)

![6b9f09f17446a889fdef4c6ebc9173a195cca9692bd570e62205641fb9a27ec2.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/tables/6b9f09f17446a889fdef4c6ebc9173a195cca9692bd570e62205641fb9a27ec2.jpg)

![7a27806431d51864bbc101c7dd70e7aceac9b94a579ddfaaf70672957a50344d.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/tables/7a27806431d51864bbc101c7dd70e7aceac9b94a579ddfaaf70672957a50344d.jpg)

![7effa15b7c9fcd3ac98d90bdd0049efb9b5056045122919880c14e7735cb2b28.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/tables/7effa15b7c9fcd3ac98d90bdd0049efb9b5056045122919880c14e7735cb2b28.jpg)

![8d05e30caa918af0be42f77d78152c9cf3910a891a49c497f104a34cf3d3ad60.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/tables/8d05e30caa918af0be42f77d78152c9cf3910a891a49c497f104a34cf3d3ad60.jpg)

![b4d2b6df0e2eb6e25f21d28c64639cc088236eaa8f77f404aca9f1976e593bed.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/tables/b4d2b6df0e2eb6e25f21d28c64639cc088236eaa8f77f404aca9f1976e593bed.jpg)

![b85cc5d69112415679992e1aa77daeecf29fb569d57ade694e379d94fbf5e8bc.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/tables/b85cc5d69112415679992e1aa77daeecf29fb569d57ade694e379d94fbf5e8bc.jpg)

![bd159ad27ee9c1acc665af6f8658671fb6b041bdf8907076729e13e3587b231a.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/tables/bd159ad27ee9c1acc665af6f8658671fb6b041bdf8907076729e13e3587b231a.jpg)

![be5493ca73a1ebf315a1d43551f2513f31a5af6a78d0534ac1ba4977b46caf0e.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/tables/be5493ca73a1ebf315a1d43551f2513f31a5af6a78d0534ac1ba4977b46caf0e.jpg)

![eba1364a0f55fefff2a9c77c4865ab9b10e88aad20c92a44e1e032af159caffa.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/tables/eba1364a0f55fefff2a9c77c4865ab9b10e88aad20c92a44e1e032af159caffa.jpg)

![ed4175c4412a73de0e31b852badc669f201f0c833e92d51302ab802385196fa9.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/tables/ed4175c4412a73de0e31b852badc669f201f0c833e92d51302ab802385196fa9.jpg)

![f9c1082b5093cee674aea891ba0fdd11b75ebd9866e38f39ada2714370657d24.jpg](../iclr_results/1936_Diff-2-in-1_ Bridging Generation and Dense Perception with Diffusion Models/tables/f9c1082b5093cee674aea891ba0fdd11b75ebd9866e38f39ada2714370657d24.jpg)

## Truncated Consistency Models


### Images

![13f67585255b2b764b5d474e9a27ee1a2b8388e616fe670049411a3ade53db50.jpg](../iclr_results/1937_Truncated Consistency Models/images/13f67585255b2b764b5d474e9a27ee1a2b8388e616fe670049411a3ade53db50.jpg)

![218244679692a749f183e341b06ea9b9abedb9e24aadca4cc6d495851ed104c6.jpg](../iclr_results/1937_Truncated Consistency Models/images/218244679692a749f183e341b06ea9b9abedb9e24aadca4cc6d495851ed104c6.jpg)

![27d6f7de2be7d647937a83b916e663fdcc77ae144f609763e4ec90248a2ec898.jpg](../iclr_results/1937_Truncated Consistency Models/images/27d6f7de2be7d647937a83b916e663fdcc77ae144f609763e4ec90248a2ec898.jpg)

![599a30142e5c6e0ed725cc0fbba5f1f82ef5a38b33869a802d1806e9ddfa7d1c.jpg](../iclr_results/1937_Truncated Consistency Models/images/599a30142e5c6e0ed725cc0fbba5f1f82ef5a38b33869a802d1806e9ddfa7d1c.jpg)

![66c98707fc920211b04a348083ff10355dd5fb45af3f5668e2305660c98439a8.jpg](../iclr_results/1937_Truncated Consistency Models/images/66c98707fc920211b04a348083ff10355dd5fb45af3f5668e2305660c98439a8.jpg)

![7036cf4be01e9bf466b3c20452a768e49844fd109b53365eed8db79a9fa4a15b.jpg](../iclr_results/1937_Truncated Consistency Models/images/7036cf4be01e9bf466b3c20452a768e49844fd109b53365eed8db79a9fa4a15b.jpg)

![708ff4c213ffa512a65953fae71662783d4eae681e8864d9b524dd0ced4ea9f7.jpg](../iclr_results/1937_Truncated Consistency Models/images/708ff4c213ffa512a65953fae71662783d4eae681e8864d9b524dd0ced4ea9f7.jpg)

![70d0741042c284c4aeedb3bdfe841ce741339db09030da0468888663919b3abb.jpg](../iclr_results/1937_Truncated Consistency Models/images/70d0741042c284c4aeedb3bdfe841ce741339db09030da0468888663919b3abb.jpg)

![70d753b3c1f22778892975d0758a212ebb9c036636b933b891fa678f7bafdeaf.jpg](../iclr_results/1937_Truncated Consistency Models/images/70d753b3c1f22778892975d0758a212ebb9c036636b933b891fa678f7bafdeaf.jpg)

![95c606dc53c5d13039f3a719922e57a3e8d355c23ed8312bc8176a9d99014251.jpg](../iclr_results/1937_Truncated Consistency Models/images/95c606dc53c5d13039f3a719922e57a3e8d355c23ed8312bc8176a9d99014251.jpg)

![dc62d6a349dc33f4a143b12c1e550839fafdbf4fba70f7f13a5329b4329de713.jpg](../iclr_results/1937_Truncated Consistency Models/images/dc62d6a349dc33f4a143b12c1e550839fafdbf4fba70f7f13a5329b4329de713.jpg)

### Tables

![5e1553b23e492ee5eb7be41587c029631c9797248407e89e3c91262b2a2da5be.jpg](../iclr_results/1937_Truncated Consistency Models/tables/5e1553b23e492ee5eb7be41587c029631c9797248407e89e3c91262b2a2da5be.jpg)

![8b834ca755eac5cdf077dff309b10d7cb9bce1f54a471162aa13fb1cb3d9ae11.jpg](../iclr_results/1937_Truncated Consistency Models/tables/8b834ca755eac5cdf077dff309b10d7cb9bce1f54a471162aa13fb1cb3d9ae11.jpg)

![a41c172ed18b2fc67e08ce234bb0537267d41878d31cc5718ea6f87de2218ecb.jpg](../iclr_results/1937_Truncated Consistency Models/tables/a41c172ed18b2fc67e08ce234bb0537267d41878d31cc5718ea6f87de2218ecb.jpg)

![af7e8193e7380a1c05375419df5dfb2c82f0cec037044b605b86b3b3c5ed607d.jpg](../iclr_results/1937_Truncated Consistency Models/tables/af7e8193e7380a1c05375419df5dfb2c82f0cec037044b605b86b3b3c5ed607d.jpg)

![c2d6a9a2542ebb6989da14782e180153a8d75f1c841cb42b92ed8930066f8f6b.jpg](../iclr_results/1937_Truncated Consistency Models/tables/c2d6a9a2542ebb6989da14782e180153a8d75f1c841cb42b92ed8930066f8f6b.jpg)

![ffaa1d94b445daccd1cc72269f656c0a708c717b96eb20983a2e9e7fed13d565.jpg](../iclr_results/1937_Truncated Consistency Models/tables/ffaa1d94b445daccd1cc72269f656c0a708c717b96eb20983a2e9e7fed13d565.jpg)

## From Lazy to Rich: Exact Learning Dynamics in Deep Linear Networks


### Images

![15144785155d2c4caf4e0f9ef0706cd2a7c35706f66e0b0d88b525ad9b1605a1.jpg](../iclr_results/1938_From Lazy to Rich_ Exact Learning Dynamics in Deep Linear Networks/images/15144785155d2c4caf4e0f9ef0706cd2a7c35706f66e0b0d88b525ad9b1605a1.jpg)

![1d0216b22bebed7f7d2858db12b7128b2190006007ba8558b871f404bfb9ae94.jpg](../iclr_results/1938_From Lazy to Rich_ Exact Learning Dynamics in Deep Linear Networks/images/1d0216b22bebed7f7d2858db12b7128b2190006007ba8558b871f404bfb9ae94.jpg)

![27344635c28c0b31094e0402ab4a2ec4d16a34aeb249291630dcc70fe0ebcda2.jpg](../iclr_results/1938_From Lazy to Rich_ Exact Learning Dynamics in Deep Linear Networks/images/27344635c28c0b31094e0402ab4a2ec4d16a34aeb249291630dcc70fe0ebcda2.jpg)

![363a1e4d5c9e0f41c3739b6fc9cc6927453069afed049d905be5f174cdad21d4.jpg](../iclr_results/1938_From Lazy to Rich_ Exact Learning Dynamics in Deep Linear Networks/images/363a1e4d5c9e0f41c3739b6fc9cc6927453069afed049d905be5f174cdad21d4.jpg)

![3c95a93502e18095ca9480dd6277db04340dc30241598174c9967809ef115abf.jpg](../iclr_results/1938_From Lazy to Rich_ Exact Learning Dynamics in Deep Linear Networks/images/3c95a93502e18095ca9480dd6277db04340dc30241598174c9967809ef115abf.jpg)

![49104bc6459489bd9c49ae352e3c6fb68b6aafa20d8e0fdd264320309c842c75.jpg](../iclr_results/1938_From Lazy to Rich_ Exact Learning Dynamics in Deep Linear Networks/images/49104bc6459489bd9c49ae352e3c6fb68b6aafa20d8e0fdd264320309c842c75.jpg)

![4e1cab01df32cf43389b14210522aa1aa953472af9c4af6c443ad490b09e8061.jpg](../iclr_results/1938_From Lazy to Rich_ Exact Learning Dynamics in Deep Linear Networks/images/4e1cab01df32cf43389b14210522aa1aa953472af9c4af6c443ad490b09e8061.jpg)

![5520101f9a41d89c978ace4636fdf018644b4a024843342b6b8d1bec351e8f4b.jpg](../iclr_results/1938_From Lazy to Rich_ Exact Learning Dynamics in Deep Linear Networks/images/5520101f9a41d89c978ace4636fdf018644b4a024843342b6b8d1bec351e8f4b.jpg)

![76ee158c60d6c028ef440e7213a60840b69c69567546bc2a03582731a085f90f.jpg](../iclr_results/1938_From Lazy to Rich_ Exact Learning Dynamics in Deep Linear Networks/images/76ee158c60d6c028ef440e7213a60840b69c69567546bc2a03582731a085f90f.jpg)

![846fcbc7b104660d667c5b6bd4236e035d21e97868b53eb70a4d94190b8f5c6a.jpg](../iclr_results/1938_From Lazy to Rich_ Exact Learning Dynamics in Deep Linear Networks/images/846fcbc7b104660d667c5b6bd4236e035d21e97868b53eb70a4d94190b8f5c6a.jpg)

![9fd78c137d5256b624b3598d16c430bf01fe3273f24224bc7a1a83ee0ab0944d.jpg](../iclr_results/1938_From Lazy to Rich_ Exact Learning Dynamics in Deep Linear Networks/images/9fd78c137d5256b624b3598d16c430bf01fe3273f24224bc7a1a83ee0ab0944d.jpg)

![af72161bfd9c4133f6b79d51549077c10b8ca7bf5a511e53a14bbc161b9ef702.jpg](../iclr_results/1938_From Lazy to Rich_ Exact Learning Dynamics in Deep Linear Networks/images/af72161bfd9c4133f6b79d51549077c10b8ca7bf5a511e53a14bbc161b9ef702.jpg)

![c528a2164118f5d9302aa909ab68f828e3e62f5f9968ee7002ebf2facd1702c7.jpg](../iclr_results/1938_From Lazy to Rich_ Exact Learning Dynamics in Deep Linear Networks/images/c528a2164118f5d9302aa909ab68f828e3e62f5f9968ee7002ebf2facd1702c7.jpg)

![d4c64d214d14064a8707bdc3d1af23486b7359c94c63b2f00e4f5eca4ed40729.jpg](../iclr_results/1938_From Lazy to Rich_ Exact Learning Dynamics in Deep Linear Networks/images/d4c64d214d14064a8707bdc3d1af23486b7359c94c63b2f00e4f5eca4ed40729.jpg)

### Tables

![2aa1c75116f6d00ecb23eac429a3a612ca312bec104a73d0b4075335f71e975b.jpg](../iclr_results/1938_From Lazy to Rich_ Exact Learning Dynamics in Deep Linear Networks/tables/2aa1c75116f6d00ecb23eac429a3a612ca312bec104a73d0b4075335f71e975b.jpg)

![474e2b61106155b88fb558882ce4725eb87ff6c6ff0834508212f789c45c1382.jpg](../iclr_results/1938_From Lazy to Rich_ Exact Learning Dynamics in Deep Linear Networks/tables/474e2b61106155b88fb558882ce4725eb87ff6c6ff0834508212f789c45c1382.jpg)

![58467ef2339de6deb9410eb98b82e59b979571c69e906a1f63f912e480ab2812.jpg](../iclr_results/1938_From Lazy to Rich_ Exact Learning Dynamics in Deep Linear Networks/tables/58467ef2339de6deb9410eb98b82e59b979571c69e906a1f63f912e480ab2812.jpg)

![f16ccc6b697b91e1071aa4f5dc237a81fb46be2e850c65121a0782d082ef17c1.jpg](../iclr_results/1938_From Lazy to Rich_ Exact Learning Dynamics in Deep Linear Networks/tables/f16ccc6b697b91e1071aa4f5dc237a81fb46be2e850c65121a0782d082ef17c1.jpg)

## A Policy-Gradient Approach to Solving Imperfect-Information Games with Best-Iterate Convergence


### Images

![350b1598f27c6ab2a603a6517e2243dad27fcc944ed386875dbf25d926d7a80b.jpg](../iclr_results/1939_A Policy-Gradient Approach to Solving Imperfect-Information Games with Best-Iterate Convergence/images/350b1598f27c6ab2a603a6517e2243dad27fcc944ed386875dbf25d926d7a80b.jpg)

![5fe1284d5e0112895381fe582d85351806c69a0207aedc10b43b1d10688205e2.jpg](../iclr_results/1939_A Policy-Gradient Approach to Solving Imperfect-Information Games with Best-Iterate Convergence/images/5fe1284d5e0112895381fe582d85351806c69a0207aedc10b43b1d10688205e2.jpg)

![630490932e16efd10d25e3974f88763b778b0df10fbdc39329e0244052c57d2f.jpg](../iclr_results/1939_A Policy-Gradient Approach to Solving Imperfect-Information Games with Best-Iterate Convergence/images/630490932e16efd10d25e3974f88763b778b0df10fbdc39329e0244052c57d2f.jpg)

### Tables

![0533cd32c704c0cac4abee62a27d61432a94aaf7eec87791615bf85fbbdbabe8.jpg](../iclr_results/1939_A Policy-Gradient Approach to Solving Imperfect-Information Games with Best-Iterate Convergence/tables/0533cd32c704c0cac4abee62a27d61432a94aaf7eec87791615bf85fbbdbabe8.jpg)

![1bd5561554c9940cb67ef2c2364cc49b5b987746490706c6a7df4a078d9c5ca6.jpg](../iclr_results/1939_A Policy-Gradient Approach to Solving Imperfect-Information Games with Best-Iterate Convergence/tables/1bd5561554c9940cb67ef2c2364cc49b5b987746490706c6a7df4a078d9c5ca6.jpg)

## ARB-LLM: Alternating Refined Binarizations for Large Language Models


### Images

![1264a3e9e1727aa0a740312400960760ddc55ef5cb702bcc0da14342b20135c6.jpg](../iclr_results/1940_ARB-LLM_ Alternating Refined Binarizations for Large Language Models/images/1264a3e9e1727aa0a740312400960760ddc55ef5cb702bcc0da14342b20135c6.jpg)

![3f2e4326e05e1a9b8d9141dbe6ea1553716dac0d35caa30cec81e46ea2ff7fa4.jpg](../iclr_results/1940_ARB-LLM_ Alternating Refined Binarizations for Large Language Models/images/3f2e4326e05e1a9b8d9141dbe6ea1553716dac0d35caa30cec81e46ea2ff7fa4.jpg)

![8548379d3f2ce6fbb0058afabd9c920604e61b1e04d2b936d0ee54d15d780c2b.jpg](../iclr_results/1940_ARB-LLM_ Alternating Refined Binarizations for Large Language Models/images/8548379d3f2ce6fbb0058afabd9c920604e61b1e04d2b936d0ee54d15d780c2b.jpg)

![b41eabcaef46324d22acc6eea40fafdbf54412c0fa5ab2ba00754d67369df19a.jpg](../iclr_results/1940_ARB-LLM_ Alternating Refined Binarizations for Large Language Models/images/b41eabcaef46324d22acc6eea40fafdbf54412c0fa5ab2ba00754d67369df19a.jpg)

![cd045ffca24f677ad4a9384377eab10aece1d48185805ce3b73b0932642c7007.jpg](../iclr_results/1940_ARB-LLM_ Alternating Refined Binarizations for Large Language Models/images/cd045ffca24f677ad4a9384377eab10aece1d48185805ce3b73b0932642c7007.jpg)

![dbd507a96dc9f2191108193c1940a645fffa4b56b98bedd43fd0cbfe20abbdde.jpg](../iclr_results/1940_ARB-LLM_ Alternating Refined Binarizations for Large Language Models/images/dbd507a96dc9f2191108193c1940a645fffa4b56b98bedd43fd0cbfe20abbdde.jpg)

### Tables

![139589fbc715ec8f4d6076248959ae1139b3cfd2405762fd84415e1770fb7874.jpg](../iclr_results/1940_ARB-LLM_ Alternating Refined Binarizations for Large Language Models/tables/139589fbc715ec8f4d6076248959ae1139b3cfd2405762fd84415e1770fb7874.jpg)

![210547a360d7704badc532935adaa212eaa7d770fd925980bc1a91d50fea0d17.jpg](../iclr_results/1940_ARB-LLM_ Alternating Refined Binarizations for Large Language Models/tables/210547a360d7704badc532935adaa212eaa7d770fd925980bc1a91d50fea0d17.jpg)

![3fc0ac37d487712f69d888f3e478de2df2c9108eb368f04135a1e15cce594251.jpg](../iclr_results/1940_ARB-LLM_ Alternating Refined Binarizations for Large Language Models/tables/3fc0ac37d487712f69d888f3e478de2df2c9108eb368f04135a1e15cce594251.jpg)

![7a11a3f1a7724986bada2ea5a9b860ccd6871923d0a19afb331089c69fa18ef6.jpg](../iclr_results/1940_ARB-LLM_ Alternating Refined Binarizations for Large Language Models/tables/7a11a3f1a7724986bada2ea5a9b860ccd6871923d0a19afb331089c69fa18ef6.jpg)

![a2b7d73f295ceeeb186bd59ce200b01e74808ffb02b62eed69ea8d85a1501ce1.jpg](../iclr_results/1940_ARB-LLM_ Alternating Refined Binarizations for Large Language Models/tables/a2b7d73f295ceeeb186bd59ce200b01e74808ffb02b62eed69ea8d85a1501ce1.jpg)

![a7a8929143bd552e8836866c16eec88eb5ddaa7c125c9c89e1b3c75f74d2c290.jpg](../iclr_results/1940_ARB-LLM_ Alternating Refined Binarizations for Large Language Models/tables/a7a8929143bd552e8836866c16eec88eb5ddaa7c125c9c89e1b3c75f74d2c290.jpg)

![ab4433fcbb02018ca408df2c5a808cd2f12e6e3dfad96e8a046a9366346f13f0.jpg](../iclr_results/1940_ARB-LLM_ Alternating Refined Binarizations for Large Language Models/tables/ab4433fcbb02018ca408df2c5a808cd2f12e6e3dfad96e8a046a9366346f13f0.jpg)

![baa5f90494b4e0636f244903686570f86b291f38c2b35ab09ccdf31eb5ff4173.jpg](../iclr_results/1940_ARB-LLM_ Alternating Refined Binarizations for Large Language Models/tables/baa5f90494b4e0636f244903686570f86b291f38c2b35ab09ccdf31eb5ff4173.jpg)

![bc54ddecba526414ec4722dbfcb4ad844d369cad5d4c1867509c4041391d4dc6.jpg](../iclr_results/1940_ARB-LLM_ Alternating Refined Binarizations for Large Language Models/tables/bc54ddecba526414ec4722dbfcb4ad844d369cad5d4c1867509c4041391d4dc6.jpg)

![cf2a7537d5617c5f9e95d7a3f02f12ad1e9905a10a9e4655e0c4a91488b24627.jpg](../iclr_results/1940_ARB-LLM_ Alternating Refined Binarizations for Large Language Models/tables/cf2a7537d5617c5f9e95d7a3f02f12ad1e9905a10a9e4655e0c4a91488b24627.jpg)

![d90ebe956264c53b9940ec848f394de8a8af277be8a72f25f79a1179721b3fc0.jpg](../iclr_results/1940_ARB-LLM_ Alternating Refined Binarizations for Large Language Models/tables/d90ebe956264c53b9940ec848f394de8a8af277be8a72f25f79a1179721b3fc0.jpg)

![f6f6a1894ac5ca2cfa0dff274769962699dcfc5d0a9f3008aefc9d5958374043.jpg](../iclr_results/1940_ARB-LLM_ Alternating Refined Binarizations for Large Language Models/tables/f6f6a1894ac5ca2cfa0dff274769962699dcfc5d0a9f3008aefc9d5958374043.jpg)

## Radar: Fast Long-Context Decoding for Any Transformer


### Images

![085fc5cbc0923138c3b9129516c20d6c642dcfc8ee1ea1617b8c111038db9f8f.jpg](../iclr_results/1941_Radar_ Fast Long-Context Decoding for Any Transformer/images/085fc5cbc0923138c3b9129516c20d6c642dcfc8ee1ea1617b8c111038db9f8f.jpg)

![102da3aec50d475415f57b62243d1bb5721dcc22323d82034b1281b654d7c048.jpg](../iclr_results/1941_Radar_ Fast Long-Context Decoding for Any Transformer/images/102da3aec50d475415f57b62243d1bb5721dcc22323d82034b1281b654d7c048.jpg)

![3b5820c1ec2ad0de4edda32d4f31dcc639622b3a8c37a55f3e9d1570cec0d2f5.jpg](../iclr_results/1941_Radar_ Fast Long-Context Decoding for Any Transformer/images/3b5820c1ec2ad0de4edda32d4f31dcc639622b3a8c37a55f3e9d1570cec0d2f5.jpg)

![3b888342381e8c36d6b19c141679fffc6d7ea073f93d43192b8a4610f2871a7f.jpg](../iclr_results/1941_Radar_ Fast Long-Context Decoding for Any Transformer/images/3b888342381e8c36d6b19c141679fffc6d7ea073f93d43192b8a4610f2871a7f.jpg)

![793f548b530ccdab16ac62f62a38ea21bd6b7abf1b174583ae06e1676635bd05.jpg](../iclr_results/1941_Radar_ Fast Long-Context Decoding for Any Transformer/images/793f548b530ccdab16ac62f62a38ea21bd6b7abf1b174583ae06e1676635bd05.jpg)

![8b52831a5c56e1d19609b09c7b77c862de53574c9ff84f61f288abdb2ecd98a2.jpg](../iclr_results/1941_Radar_ Fast Long-Context Decoding for Any Transformer/images/8b52831a5c56e1d19609b09c7b77c862de53574c9ff84f61f288abdb2ecd98a2.jpg)

![dda544a7cdf9b0aa10c5a1fd33e5d18a6480aad49aa8b286d1648eb4040f7ce6.jpg](../iclr_results/1941_Radar_ Fast Long-Context Decoding for Any Transformer/images/dda544a7cdf9b0aa10c5a1fd33e5d18a6480aad49aa8b286d1648eb4040f7ce6.jpg)

### Tables

![3cb73fb33a531b7bba5333042243dad4ad18195becd99c365cb8652a78deac15.jpg](../iclr_results/1941_Radar_ Fast Long-Context Decoding for Any Transformer/tables/3cb73fb33a531b7bba5333042243dad4ad18195becd99c365cb8652a78deac15.jpg)

![64ee6d1c724f0131387e21c27450db35a0d009a0cb44acc10108e45ea9b3fd9e.jpg](../iclr_results/1941_Radar_ Fast Long-Context Decoding for Any Transformer/tables/64ee6d1c724f0131387e21c27450db35a0d009a0cb44acc10108e45ea9b3fd9e.jpg)

![7c44ef41af4d646094cf573ca7abc7cd344b178a98ed6a202eccafe3e0a5be87.jpg](../iclr_results/1941_Radar_ Fast Long-Context Decoding for Any Transformer/tables/7c44ef41af4d646094cf573ca7abc7cd344b178a98ed6a202eccafe3e0a5be87.jpg)

![efb6d74bda5b1c4571d31451eb9d9028efdc1968b69898fe583b4e675f80f8e2.jpg](../iclr_results/1941_Radar_ Fast Long-Context Decoding for Any Transformer/tables/efb6d74bda5b1c4571d31451eb9d9028efdc1968b69898fe583b4e675f80f8e2.jpg)

## Self-Improving Robust Preference Optimization

### Images

![26f4aa6201dfb39ab1d04840ae5cf0d6e39e5235b034b798ad2a3f013766145b.jpg](../iclr_results/1942_Self-Improving Robust Preference Optimization/images/26f4aa6201dfb39ab1d04840ae5cf0d6e39e5235b034b798ad2a3f013766145b.jpg)

![2a1857bc03e45acb509c14247fecf74578854b2e46082951df2700a51aca9ab2.jpg](../iclr_results/1942_Self-Improving Robust Preference Optimization/images/2a1857bc03e45acb509c14247fecf74578854b2e46082951df2700a51aca9ab2.jpg)

![2e434474dfb3a0a6387f5b7db226813f06476a962b429a2bff778485d5534769.jpg](../iclr_results/1942_Self-Improving Robust Preference Optimization/images/2e434474dfb3a0a6387f5b7db226813f06476a962b429a2bff778485d5534769.jpg)

![be4f4273439a099f5871d50bbaf1d409bedbe38950eeddb581ba08681d0f11df.jpg](../iclr_results/1942_Self-Improving Robust Preference Optimization/images/be4f4273439a099f5871d50bbaf1d409bedbe38950eeddb581ba08681d0f11df.jpg)

![d7d2aa7029ba268752b8fe8f726332891da78f25095bf559a3ab4dfd3332207f.jpg](../iclr_results/1942_Self-Improving Robust Preference Optimization/images/d7d2aa7029ba268752b8fe8f726332891da78f25095bf559a3ab4dfd3332207f.jpg)

### Tables

![7f115c4e6bd04bca1446f144071ce0ebbce7ed87332046bbad2f6ade4baa70af.jpg](../iclr_results/1942_Self-Improving Robust Preference Optimization/tables/7f115c4e6bd04bca1446f144071ce0ebbce7ed87332046bbad2f6ade4baa70af.jpg)

![7f34286302d001134c81e9df9761a098ef89ffba4dba322bb148860423cc51a8.jpg](../iclr_results/1942_Self-Improving Robust Preference Optimization/tables/7f34286302d001134c81e9df9761a098ef89ffba4dba322bb148860423cc51a8.jpg)

![f539b00a209c22cd70e5c3b187f37bdfcbc085aff7700e6ec8326d79dc84d8a6.jpg](../iclr_results/1942_Self-Improving Robust Preference Optimization/tables/f539b00a209c22cd70e5c3b187f37bdfcbc085aff7700e6ec8326d79dc84d8a6.jpg)
