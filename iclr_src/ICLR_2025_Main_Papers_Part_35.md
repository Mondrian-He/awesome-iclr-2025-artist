# ICLR 2025 Main Conference Papers

**Summary:** 36 papers with extracted content:
- 📊 Total images: 44031
- 📋 Total tables: 33468
- 📄 Total files: 77499

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 35 of 100

## 目录 (Table of Contents)

1. [Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity](#Zeroth-Order-Fine-Tuning-of-LLMs-with-Transferable-Static-Sparsity)
2. [Atomas: Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Generation](#Atomas-Hierarchical-Adaptive-Alignment-on-Molecule-Text-for-Unified-Molecule-Understanding-and-Generation)
3. [From Layers to States: A State Space Model Perspective to Deep Neural Network Layer Dynamics](#From-Layers-to-States-A-State-Space-Model-Perspective-to-Deep-Neural-Network-Layer-Dynamics)
4. [TIGeR: Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models](#TIGeR-Unifying-Text-to-Image-Generation-and-Retrieval-with-Large-Multimodal-Models)
5. [Towards Federated RLHF with Aggregated Client Preference for LLMs](#Towards-Federated-RLHF-with-Aggregated-Client-Preference-for-LLMs)
6. [Subtask-Aware Visual Reward Learning from Segmented Demonstrations](#Subtask-Aware-Visual-Reward-Learning-from-Segmented-Demonstrations)
7. [From Isolated Conversations to Hierarchical Schemas: Dynamic Tree Memory Representation for LLMs](#From-Isolated-Conversations-to-Hierarchical-Schemas-Dynamic-Tree-Memory-Representation-for-LLMs)
8. [Web Agents with World Models: Learning and Leveraging Environment Dynamics in Web Navigation](#Web-Agents-with-World-Models-Learning-and-Leveraging-Environment-Dynamics-in-Web-Navigation)
9. [Quamba: A Post-Training Quantization Recipe for Selective State Space Models](#Quamba-A-Post-Training-Quantization-Recipe-for-Selective-State-Space-Models)
10. [Active Learning for Continual Learning: Keeping the Past Alive in the Present](#Active-Learning-for-Continual-Learning-Keeping-the-Past-Alive-in-the-Present)
11. [Ready-to-React: Online Reaction Policy for Two-Character Interaction Generation](#Ready-to-React-Online-Reaction-Policy-for-Two-Character-Interaction-Generation)
12. [SimXRD-4M: Big Simulated X-ray Diffraction Data and Crystal Symmetry Classification Benchmark](#SimXRD-4M-Big-Simulated-X-ray-Diffraction-Data-and-Crystal-Symmetry-Classification-Benchmark)
13. [Associative memory and dead neurons](#Associative-memory-and-dead-neurons)
14. [Preble: Efficient Distributed Prompt Scheduling for LLM Serving](#Preble-Efficient-Distributed-Prompt-Scheduling-for-LLM-Serving)
15. [From Tokens to Lattices: Emergent Lattice Structures in Language Models](#From-Tokens-to-Lattices-Emergent-Lattice-Structures-in-Language-Models)
16. [HQ-Edit: A High-Quality Dataset for Instruction-based Image Editing](#HQ-Edit-A-High-Quality-Dataset-for-Instruction-based-Image-Editing)
17. [Topological Zigzag Spaghetti for Diffusion-based Generation and Prediction on Graphs](#Topological-Zigzag-Spaghetti-for-Diffusion-based-Generation-and-Prediction-on-Graphs)
18. [E(3)-equivariant models cannot learn chirality: Field-based molecular generation](#E3-equivariant-models-cannot-learn-chirality-Field-based-molecular-generation)
19. [A General Framework for Off-Policy Learning with Partially-Observed Reward](#A-General-Framework-for-Off-Policy-Learning-with-Partially-Observed-Reward)
20. [Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection](#Pursuing-Feature-Separation-based-on-Neural-Collapse-for-Out-of-Distribution-Detection)
21. [CipherPrune:  Efficient and Scalable Private Transformer Inference](#CipherPrune-Efficient-and-Scalable-Private-Transformer-Inference)
22. [OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data](#OpenMathInstruct-2-Accelerating-AI-for-Math-with-Massive-Open-Source-Instruction-Data)
23. [Last Iterate Convergence of Incremental Methods as a Model of Forgetting](#Last-Iterate-Convergence-of-Incremental-Methods-as-a-Model-of-Forgetting)
24. [Visually Guided Decoding: Gradient-Free Hard Prompt Inversion with Language Models](#Visually-Guided-Decoding-Gradient-Free-Hard-Prompt-Inversion-with-Language-Models)
25. [AgentSquare: Automatic LLM Agent Search in Modular Design Space](#AgentSquare-Automatic-LLM-Agent-Search-in-Modular-Design-Space)
26. [GS-CPR: Efficient Camera Pose Refinement via 3D Gaussian Splatting](#GS-CPR-Efficient-Camera-Pose-Refinement-via-3D-Gaussian-Splatting)
27. [DRESSing Up LLM: Efficient Stylized Question-Answering via Style Subspace Editing](#DRESSing-Up-LLM-Efficient-Stylized-Question-Answering-via-Style-Subspace-Editing)
28. [Fragment and Geometry Aware Tokenization of Molecules for Structure-Based Drug Design Using Language Models](#Fragment-and-Geometry-Aware-Tokenization-of-Molecules-for-Structure-Based-Drug-Design-Using-Language-Models)
29. [ParetoFlow: Guided Flows in Multi-Objective Optimization](#ParetoFlow-Guided-Flows-in-Multi-Objective-Optimization)
30. [GOFA: A Generative One-For-All Model for Joint Graph Language Modeling](#GOFA-A-Generative-One-For-All-Model-for-Joint-Graph-Language-Modeling)
31. [Apollo-MILP: An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear Programming](#Apollo-MILP-An-Alternating-Prediction-Correction-Neural-Solving-Framework-for-Mixed-Integer-Linear-Programming)
32. [Diffusion Policy Policy Optimization](#Diffusion-Policy-Policy-Optimization)
33. [Image Watermarks are Removable using Controllable Regeneration from Clean Noise](#Image-Watermarks-are-Removable-using-Controllable-Regeneration-from-Clean-Noise)
34. [Coreset Selection via Reducible Loss in Continual Learning](#Coreset-Selection-via-Reducible-Loss-in-Continual-Learning)
35. [Efficient Distribution Matching of Representations via Noise-Injected Deep InfoMax](#Efficient-Distribution-Matching-of-Representations-via-Noise-Injected-Deep-InfoMax)
36. [Is Your Video Language Model a Reliable Judge?](#Is-Your-Video-Language-Model-a-Reliable-Judge)

---


## Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity

### Images

![0b5f5987c0bbf5ce9b80f8e6705f7586c1f72fe5d5aab43fb36f0ec8ab21163f.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/images/0b5f5987c0bbf5ce9b80f8e6705f7586c1f72fe5d5aab43fb36f0ec8ab21163f.jpg)

![17ef9426029b95a75503c9fae79ede38b1e63059d8ad9aea938f8316451a67c6.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/images/17ef9426029b95a75503c9fae79ede38b1e63059d8ad9aea938f8316451a67c6.jpg)

![2a57825ba457b809c572f076d69967b2c05bf807d2908be4fcccc9781acbaf2a.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/images/2a57825ba457b809c572f076d69967b2c05bf807d2908be4fcccc9781acbaf2a.jpg)

![2d7bdfbfe3ed2420614f51c1f282c16370e0d86bbfc220dc78ad8e090c87a1c7.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/images/2d7bdfbfe3ed2420614f51c1f282c16370e0d86bbfc220dc78ad8e090c87a1c7.jpg)

![2df79cc1a2b63dd81c6fa328da71842b2cf1b348cac23f78932e5cc0331f41d2.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/images/2df79cc1a2b63dd81c6fa328da71842b2cf1b348cac23f78932e5cc0331f41d2.jpg)

![48656ef0fe6e8f53a2b230674b6bd163c3eeda978cd7397bc89c924b2628da2c.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/images/48656ef0fe6e8f53a2b230674b6bd163c3eeda978cd7397bc89c924b2628da2c.jpg)

![611a9eb6fa446b81276065aafc9a69177ffc9ca0c67458982c866e430771761b.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/images/611a9eb6fa446b81276065aafc9a69177ffc9ca0c67458982c866e430771761b.jpg)

![6745d29e449acf630c66b60d9be0f680d5447f259196cff9853ef9beab368e9f.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/images/6745d29e449acf630c66b60d9be0f680d5447f259196cff9853ef9beab368e9f.jpg)

![6929f38d21dc9ff0f92e02562ee3d413342ca2fa09ba535760da5d5de7fb328c.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/images/6929f38d21dc9ff0f92e02562ee3d413342ca2fa09ba535760da5d5de7fb328c.jpg)

![71586ab885038b213956115cbf044e34338e164fe7404e84f9b1800496d8dd55.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/images/71586ab885038b213956115cbf044e34338e164fe7404e84f9b1800496d8dd55.jpg)

![772ba5ec7d58eb05cf7fba28a23d693f27aef21b0cfc032087bbe41a82f5cb8b.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/images/772ba5ec7d58eb05cf7fba28a23d693f27aef21b0cfc032087bbe41a82f5cb8b.jpg)

![8395570190f8a40e229f7895124026634c4589cd3f8c49c61a2d923d008a7da2.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/images/8395570190f8a40e229f7895124026634c4589cd3f8c49c61a2d923d008a7da2.jpg)

![93221a3788484277fe5d980e7da528e8758b72d2039c5d8e11d0026babae89f8.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/images/93221a3788484277fe5d980e7da528e8758b72d2039c5d8e11d0026babae89f8.jpg)

![9885160c9a635201e0d87aa7297d58012631b163511fd19e2735efca093c8177.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/images/9885160c9a635201e0d87aa7297d58012631b163511fd19e2735efca093c8177.jpg)

![b96677a8fd9a4c2742f51420e25dd1a7edc3dd6223fe6fe1c1e482cf45c8b3de.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/images/b96677a8fd9a4c2742f51420e25dd1a7edc3dd6223fe6fe1c1e482cf45c8b3de.jpg)

![bf7666a5ef31453284cb2569e2fea37bc926f514b745614f7de3ea25b01b95e9.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/images/bf7666a5ef31453284cb2569e2fea37bc926f514b745614f7de3ea25b01b95e9.jpg)

![ccefe89ae0a3e333c579d952997fda8823b61c96f1e14f97d17dd0365eb53cb7.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/images/ccefe89ae0a3e333c579d952997fda8823b61c96f1e14f97d17dd0365eb53cb7.jpg)

![d009ce86cb50f7d8dca201df4e5be2bed318a3aaac0ee88c63ec4e20d5acba0b.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/images/d009ce86cb50f7d8dca201df4e5be2bed318a3aaac0ee88c63ec4e20d5acba0b.jpg)

![dbe51b64f4e4efa10e0fbd4029ba2dab83adc701721c304577e23539332b51ff.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/images/dbe51b64f4e4efa10e0fbd4029ba2dab83adc701721c304577e23539332b51ff.jpg)

![fbbf073cb6fab74ed5b2937d7cbbfc3326948113af2649519f22d157633b47ef.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/images/fbbf073cb6fab74ed5b2937d7cbbfc3326948113af2649519f22d157633b47ef.jpg)

### Tables

![062a11c4079311c4039421667f614b68e3bc8803e3ec45552154f69ad37c0cfc.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/tables/062a11c4079311c4039421667f614b68e3bc8803e3ec45552154f69ad37c0cfc.jpg)

![228f25d911a9a70560b20ecfa8d04ef9b4c82f5e91045dd06527ccf3bb7322d1.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/tables/228f25d911a9a70560b20ecfa8d04ef9b4c82f5e91045dd06527ccf3bb7322d1.jpg)

![4f438dd22ab4d0c4fd267cf916f927d36150df37cede12bacaa82ec2005da6f4.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/tables/4f438dd22ab4d0c4fd267cf916f927d36150df37cede12bacaa82ec2005da6f4.jpg)

![54e0c9b3346b927a42567018177df61272c72f547588ee7e5d1d9b78211ccf41.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/tables/54e0c9b3346b927a42567018177df61272c72f547588ee7e5d1d9b78211ccf41.jpg)

![88ee1268e3222bfdd7e2e7df551e6e26ecd8388d32e199b1243745b480800999.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/tables/88ee1268e3222bfdd7e2e7df551e6e26ecd8388d32e199b1243745b480800999.jpg)

![c774e3d8befe664363f8f482be709e1b5b51359940debc7d9af5f4f18d05a1b0.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/tables/c774e3d8befe664363f8f482be709e1b5b51359940debc7d9af5f4f18d05a1b0.jpg)

![ce685eea6a845320c59e2a3c5f0b686709751c71b83ac273bbb6a90b20fa0700.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/tables/ce685eea6a845320c59e2a3c5f0b686709751c71b83ac273bbb6a90b20fa0700.jpg)

![dc05b95767f9df3ead24cbe6ff69c8422a8027cde9fc9abad55184324bc62fe2.jpg](../iclr_results/1273_Layout-your-3D_ Controllable and Precise 3D Generation with 2D Blueprint/tables/dc05b95767f9df3ead24cbe6ff69c8422a8027cde9fc9abad55184324bc62fe2.jpg)

## Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity


### Images

![1a0fe757c30b3dec3025d83731e8fd558b546fb1ef34f01d64e97dd4a137c910.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/images/1a0fe757c30b3dec3025d83731e8fd558b546fb1ef34f01d64e97dd4a137c910.jpg)

![1e9fdfae34c0ded0b7cf3967bfc67571755c99964269578811ac4669d3cf0974.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/images/1e9fdfae34c0ded0b7cf3967bfc67571755c99964269578811ac4669d3cf0974.jpg)

![4f7960b87a0c4f402a6b283f61bef82bfe7cb369b994579ec15faefa02e854d9.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/images/4f7960b87a0c4f402a6b283f61bef82bfe7cb369b994579ec15faefa02e854d9.jpg)

![50e592f8abdbcfd4601475aeed13d4d8ff2190ab616614c5c3aca52eab6ca50d.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/images/50e592f8abdbcfd4601475aeed13d4d8ff2190ab616614c5c3aca52eab6ca50d.jpg)

![5e3427d03ed7f544e72f107fb49f6dea10ac869045828ed0e54a52e5a7a35d45.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/images/5e3427d03ed7f544e72f107fb49f6dea10ac869045828ed0e54a52e5a7a35d45.jpg)

![756d74ac56b638ed506e53e3b96f99b966441974d5e1c316e9a586244f62751c.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/images/756d74ac56b638ed506e53e3b96f99b966441974d5e1c316e9a586244f62751c.jpg)

![779f2eee9c91adeb1721b46015f2d5722a2d690fd7107e999ec3eb4d93b5b322.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/images/779f2eee9c91adeb1721b46015f2d5722a2d690fd7107e999ec3eb4d93b5b322.jpg)

![7c46c7d8f6ee3556a07d6b5a89ed94136be299e08cbaa60822f6808ba6c5d6e5.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/images/7c46c7d8f6ee3556a07d6b5a89ed94136be299e08cbaa60822f6808ba6c5d6e5.jpg)

![8fa507511884086e4eccbbdc48140ae3019085c5b5c84c33928ea51bb3a699d1.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/images/8fa507511884086e4eccbbdc48140ae3019085c5b5c84c33928ea51bb3a699d1.jpg)

![b69d3cd4064a28aeed5b1fa938b233eb13b9b1c86e4489d74091128f7639fc0c.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/images/b69d3cd4064a28aeed5b1fa938b233eb13b9b1c86e4489d74091128f7639fc0c.jpg)

![c0526ea13954c9c81dc275e63a1c299e79ccd47f657b31a8b5b0d0150cf84776.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/images/c0526ea13954c9c81dc275e63a1c299e79ccd47f657b31a8b5b0d0150cf84776.jpg)

![cf8525c5b8aeb5736f12c9c8f62de92f936d9ba93b6d3e2fb9ac83071d085fee.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/images/cf8525c5b8aeb5736f12c9c8f62de92f936d9ba93b6d3e2fb9ac83071d085fee.jpg)

![def0497ecc1ae28ba00ecf8cd57631bc1fb13912395015410c08a7935f33f980.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/images/def0497ecc1ae28ba00ecf8cd57631bc1fb13912395015410c08a7935f33f980.jpg)

![f17e96a1656bcf424a260c583987cc48c862b6c29015e86e68b62f603e2d20dd.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/images/f17e96a1656bcf424a260c583987cc48c862b6c29015e86e68b62f603e2d20dd.jpg)

![f4f7bfc1ffea4812bf54b11c3ac686e135bf8d952d0bc76d3802076e7844c1c3.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/images/f4f7bfc1ffea4812bf54b11c3ac686e135bf8d952d0bc76d3802076e7844c1c3.jpg)

![fa81b9eab892a8338f08745c7cd665b0ed0a936f15adced7e36e1413d33ba7c0.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/images/fa81b9eab892a8338f08745c7cd665b0ed0a936f15adced7e36e1413d33ba7c0.jpg)

### Tables

![125a2397012f7aaaf56840e5f790681116f9c5b75c9c8e96c4b92de2f2f7472b.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/tables/125a2397012f7aaaf56840e5f790681116f9c5b75c9c8e96c4b92de2f2f7472b.jpg)

![13cbd2cac664da800654349d2b4d03cdb9a76973b1e07d7468d1ead9fc989d54.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/tables/13cbd2cac664da800654349d2b4d03cdb9a76973b1e07d7468d1ead9fc989d54.jpg)

![34fdd2ad9e56161d1f8495935d0d2bc8e5dc75bfd8f9c460150639fe8f2e5ea6.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/tables/34fdd2ad9e56161d1f8495935d0d2bc8e5dc75bfd8f9c460150639fe8f2e5ea6.jpg)

![3552e9025a46c1c09e32a013179690fd5488de89b050045aaf5c285ccdf3eb0a.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/tables/3552e9025a46c1c09e32a013179690fd5488de89b050045aaf5c285ccdf3eb0a.jpg)

![378e8c618391fe50411409d5d268abee354902f1ef0d0dc7392f46ccafa7f4e4.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/tables/378e8c618391fe50411409d5d268abee354902f1ef0d0dc7392f46ccafa7f4e4.jpg)

![6d61faef99b9dc6228bc86d188113494daf0d6ca1180ac695932eed86e9e1602.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/tables/6d61faef99b9dc6228bc86d188113494daf0d6ca1180ac695932eed86e9e1602.jpg)

![73a1418e0acb780220372e93c09111fbca2b8c59332660e608e288d89f7cd329.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/tables/73a1418e0acb780220372e93c09111fbca2b8c59332660e608e288d89f7cd329.jpg)

![949b3611b3150f20711a3616b71760e39988888f8842ac487986467f75abd3b1.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/tables/949b3611b3150f20711a3616b71760e39988888f8842ac487986467f75abd3b1.jpg)

![a1432d6e191d8c9adb4bfe3b3bd1888cb819659585c1412c48272f9b1d3d3981.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/tables/a1432d6e191d8c9adb4bfe3b3bd1888cb819659585c1412c48272f9b1d3d3981.jpg)

![c5260d2766421433a3595dcdf74fb8d2bf05680b1ec5b49258f92c322a9d4127.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/tables/c5260d2766421433a3595dcdf74fb8d2bf05680b1ec5b49258f92c322a9d4127.jpg)

![d8e450d2fd1c4265e7dbdf667a1aec5bf23cc84d83710c14cf2b1b81b25a629c.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/tables/d8e450d2fd1c4265e7dbdf667a1aec5bf23cc84d83710c14cf2b1b81b25a629c.jpg)

![dc6e9129760ad654f7be884f22ad2e9896b690785f557824695afc0973bc3871.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/tables/dc6e9129760ad654f7be884f22ad2e9896b690785f557824695afc0973bc3871.jpg)

![ecf8c6a5dc6afe4688c68c1f6ef13a953195ef8aedf1dcbab7d56b85854e9f7d.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/tables/ecf8c6a5dc6afe4688c68c1f6ef13a953195ef8aedf1dcbab7d56b85854e9f7d.jpg)

![f92c82a35b4cbaac7b5e3aacad54124ef4248e4dc9729ddbb92c0cdb85fedecd.jpg](../iclr_results/1274_Zeroth-Order Fine-Tuning of LLMs with Transferable Static Sparsity/tables/f92c82a35b4cbaac7b5e3aacad54124ef4248e4dc9729ddbb92c0cdb85fedecd.jpg)

## Atomas: Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Generation


### Images

![21bb8b8c749abd6d5bb2f3853aeb38ff8b7c71e4c0fc7ed2c51ce9abcb369f94.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/images/21bb8b8c749abd6d5bb2f3853aeb38ff8b7c71e4c0fc7ed2c51ce9abcb369f94.jpg)

![575099d0cc627b1fb7e9faff3520651c683f3a2fbec3afd9aea22dc00bd0883b.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/images/575099d0cc627b1fb7e9faff3520651c683f3a2fbec3afd9aea22dc00bd0883b.jpg)

![68b9c2392b41e7c294a8decfaba2044d4a1d20745aab5a34e5b696b04516bd93.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/images/68b9c2392b41e7c294a8decfaba2044d4a1d20745aab5a34e5b696b04516bd93.jpg)

![714c97f8ecf26cbe23c83ce7f1144299a0775acd473b7cd7540a2fed36b7263b.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/images/714c97f8ecf26cbe23c83ce7f1144299a0775acd473b7cd7540a2fed36b7263b.jpg)

![87fb65ee648059046aa2883b81ccb2c1f791a5caf6158785e18f5425fe86ab95.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/images/87fb65ee648059046aa2883b81ccb2c1f791a5caf6158785e18f5425fe86ab95.jpg)

![a2a8e4ee284eb1acae460ab29f8f69fed157cb6d9df86b7923d9686ea47598c3.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/images/a2a8e4ee284eb1acae460ab29f8f69fed157cb6d9df86b7923d9686ea47598c3.jpg)

![b1871e316f9dfb6622934409c0f12d27d1add929375f2998f8f08a014f7dd3a6.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/images/b1871e316f9dfb6622934409c0f12d27d1add929375f2998f8f08a014f7dd3a6.jpg)

![bd8b7005b267b75b49b44ac854d454173548d1a991f7041d21ebd9d7c4f7edfc.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/images/bd8b7005b267b75b49b44ac854d454173548d1a991f7041d21ebd9d7c4f7edfc.jpg)

![c2c8ae6cf492d6b6fba443775b0585537a27fa36bdc314d5567b3e51afcca05b.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/images/c2c8ae6cf492d6b6fba443775b0585537a27fa36bdc314d5567b3e51afcca05b.jpg)

![c6ed7b01d714c832fefa22c613954bd211079e04f2b4bf427f43daa7ec631c80.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/images/c6ed7b01d714c832fefa22c613954bd211079e04f2b4bf427f43daa7ec631c80.jpg)

![cd5da687a3318f3e7bc9b623ca15275a20821be2963683f30dca30e5213f732c.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/images/cd5da687a3318f3e7bc9b623ca15275a20821be2963683f30dca30e5213f732c.jpg)

![ce7ef1224f60132bad47857d1c47c69ab58afa2410fef6b202db2b1d3af4ac2b.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/images/ce7ef1224f60132bad47857d1c47c69ab58afa2410fef6b202db2b1d3af4ac2b.jpg)

![d80c5fe43304e0d3e108aec57bcb4aafd1bf83d275370ebcd8c9d43375e9737a.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/images/d80c5fe43304e0d3e108aec57bcb4aafd1bf83d275370ebcd8c9d43375e9737a.jpg)

![dd96266f4634223f79a4a93bfaecddfc7d779b39551a3f02355e5d3d0c577320.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/images/dd96266f4634223f79a4a93bfaecddfc7d779b39551a3f02355e5d3d0c577320.jpg)

![ea16cd4ae3b0e473a69ad42e228fb66b9e1e5ccf81d700c5df6717f972ca88de.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/images/ea16cd4ae3b0e473a69ad42e228fb66b9e1e5ccf81d700c5df6717f972ca88de.jpg)

![f20404890e3853c81960198a662428e165885b2dd3979830e778dcb11193d3dd.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/images/f20404890e3853c81960198a662428e165885b2dd3979830e778dcb11193d3dd.jpg)

### Tables

![1331ef6e9df9c0b2a9862be423f21769816d1927caafbab94d1fe6f71b16f41c.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/tables/1331ef6e9df9c0b2a9862be423f21769816d1927caafbab94d1fe6f71b16f41c.jpg)

![1b5e2e015a549bc660bbe5951305b618931595224be931dede38d0984f07b9c5.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/tables/1b5e2e015a549bc660bbe5951305b618931595224be931dede38d0984f07b9c5.jpg)

![1da1f16470392f1e257b6abdfcbd3bec729eaae218e57f2a6885c68af41b195e.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/tables/1da1f16470392f1e257b6abdfcbd3bec729eaae218e57f2a6885c68af41b195e.jpg)

![240b050ca1d6e6509279580379b91fc6b32f4f9b3c03d468f3bbf9a8b0c622d7.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/tables/240b050ca1d6e6509279580379b91fc6b32f4f9b3c03d468f3bbf9a8b0c622d7.jpg)

![423d7ea4bfd58b09cb1fc7885f3d3ac7172666bedea92bce799625010cb6cd71.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/tables/423d7ea4bfd58b09cb1fc7885f3d3ac7172666bedea92bce799625010cb6cd71.jpg)

![4646b494f003f109b1f8bedfd6c276fcfbef40acf7a3b1bc8598024cc67e70ba.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/tables/4646b494f003f109b1f8bedfd6c276fcfbef40acf7a3b1bc8598024cc67e70ba.jpg)

![4edc923a1977102e66d6d674594a78936007ae771aa78a23245e2be4e7100285.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/tables/4edc923a1977102e66d6d674594a78936007ae771aa78a23245e2be4e7100285.jpg)

![5f0f041d3a20c317878970f6041e0701fab0f9a08aefae90b3079f54da762652.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/tables/5f0f041d3a20c317878970f6041e0701fab0f9a08aefae90b3079f54da762652.jpg)

![64179462bed321ab9031b2f62e5585a2e91e8e9c4354c934604881acc6848f96.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/tables/64179462bed321ab9031b2f62e5585a2e91e8e9c4354c934604881acc6848f96.jpg)

![8281b49cbc36fa34e5b6e4381964fae0f0d2aa33c3702ef2a8113691d34a1bed.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/tables/8281b49cbc36fa34e5b6e4381964fae0f0d2aa33c3702ef2a8113691d34a1bed.jpg)

![82f5367b56b011d93340134bf17df8535194455021abce4b15de901399d4492f.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/tables/82f5367b56b011d93340134bf17df8535194455021abce4b15de901399d4492f.jpg)

![8afd0fdd84dc97867aca0ebafb47fe6b5923dfd971f633e5e0ed038663e988ee.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/tables/8afd0fdd84dc97867aca0ebafb47fe6b5923dfd971f633e5e0ed038663e988ee.jpg)

![9335ab5be8735222a25fa6a0d3fc826f2b48891a77fa8316e061f92bbd7d75b7.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/tables/9335ab5be8735222a25fa6a0d3fc826f2b48891a77fa8316e061f92bbd7d75b7.jpg)

![9552ce33d6bf6e0ae35ccbfd77a841fcad3ab3af132239df4c2158b58a93ec0b.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/tables/9552ce33d6bf6e0ae35ccbfd77a841fcad3ab3af132239df4c2158b58a93ec0b.jpg)

![9b329d1b855424a187439d996f1b8ac88796918ce80a4626bf79bad41b3b22aa.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/tables/9b329d1b855424a187439d996f1b8ac88796918ce80a4626bf79bad41b3b22aa.jpg)

![9be6ea4e674bf2843bbf6045e1d6b0fdf0e160118114bcb3ff2b1321908f265a.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/tables/9be6ea4e674bf2843bbf6045e1d6b0fdf0e160118114bcb3ff2b1321908f265a.jpg)

![9c5a5f99238556dee38d88b16f032fd4d002a39f83816dd7e6c7d856ff1744e2.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/tables/9c5a5f99238556dee38d88b16f032fd4d002a39f83816dd7e6c7d856ff1744e2.jpg)

![a3c2ab2616606df7c42357fd10b9dcae7d492af7151acf2ccc82c8667c47f90a.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/tables/a3c2ab2616606df7c42357fd10b9dcae7d492af7151acf2ccc82c8667c47f90a.jpg)

![cbb280d78b0f8958c214e379f5d5bb511b417f66a02aada30ee64b740fd60e20.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/tables/cbb280d78b0f8958c214e379f5d5bb511b417f66a02aada30ee64b740fd60e20.jpg)

![cd9aa79eaede90e1837ff0a51687956471cb1d8a58bdd7607c748d8fe1fc2c8b.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/tables/cd9aa79eaede90e1837ff0a51687956471cb1d8a58bdd7607c748d8fe1fc2c8b.jpg)

![e2cd8605aa1cb245761009b19a5a05a629939ad357d0d5e4a15c187f4e4e3409.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/tables/e2cd8605aa1cb245761009b19a5a05a629939ad357d0d5e4a15c187f4e4e3409.jpg)

![ff674ec32c3dc89230d21ce5f4b742c6b6dca9300dd0464abbcbd42e2a3f00f4.jpg](../iclr_results/1275_Atomas_ Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Gene/tables/ff674ec32c3dc89230d21ce5f4b742c6b6dca9300dd0464abbcbd42e2a3f00f4.jpg)

## From Layers to States: A State Space Model Perspective to Deep Neural Network Layer Dynamics


### Images

![1489aa708d0f5885937cf23237605ad462033c33903d4f6d44456928e2038092.jpg](../iclr_results/1276_From Layers to States_ A State Space Model Perspective to Deep Neural Network Layer Dynamics/images/1489aa708d0f5885937cf23237605ad462033c33903d4f6d44456928e2038092.jpg)

![1a43cc47d1f53d4698206549f3976761ed3c0264da95c54b7d14f6e85b5015d4.jpg](../iclr_results/1276_From Layers to States_ A State Space Model Perspective to Deep Neural Network Layer Dynamics/images/1a43cc47d1f53d4698206549f3976761ed3c0264da95c54b7d14f6e85b5015d4.jpg)

![5621d2bbe8ceb18127eac08d27989d1c127cf3a256f81f0c566d4904631cc891.jpg](../iclr_results/1276_From Layers to States_ A State Space Model Perspective to Deep Neural Network Layer Dynamics/images/5621d2bbe8ceb18127eac08d27989d1c127cf3a256f81f0c566d4904631cc891.jpg)

![608e2ca569d7043e41d8c03fb548fe8e0902f2cbd80a751d67e3d6dc93920fa3.jpg](../iclr_results/1276_From Layers to States_ A State Space Model Perspective to Deep Neural Network Layer Dynamics/images/608e2ca569d7043e41d8c03fb548fe8e0902f2cbd80a751d67e3d6dc93920fa3.jpg)

![68c2d031e429c751e2a681ff504ce082c944e8a709dcad488fc96ad3c8232150.jpg](../iclr_results/1276_From Layers to States_ A State Space Model Perspective to Deep Neural Network Layer Dynamics/images/68c2d031e429c751e2a681ff504ce082c944e8a709dcad488fc96ad3c8232150.jpg)

![ac9fb9dfea95dec541639373273f213c52cac4b43105fe011c870c515a81cdb1.jpg](../iclr_results/1276_From Layers to States_ A State Space Model Perspective to Deep Neural Network Layer Dynamics/images/ac9fb9dfea95dec541639373273f213c52cac4b43105fe011c870c515a81cdb1.jpg)

![ad836c08ff79d7efcac1db3e58bee883bddd0473adf0e870e413a1d2f5d7bb99.jpg](../iclr_results/1276_From Layers to States_ A State Space Model Perspective to Deep Neural Network Layer Dynamics/images/ad836c08ff79d7efcac1db3e58bee883bddd0473adf0e870e413a1d2f5d7bb99.jpg)

![cc77b012d7fede00634ba47ce8d532650620700f21e873ad8652677cf1b3cb8a.jpg](../iclr_results/1276_From Layers to States_ A State Space Model Perspective to Deep Neural Network Layer Dynamics/images/cc77b012d7fede00634ba47ce8d532650620700f21e873ad8652677cf1b3cb8a.jpg)

![ff12fb8fa1c71782a17d6a8eed1853bcdf3eae74a917e5b7fff81899489d6c83.jpg](../iclr_results/1276_From Layers to States_ A State Space Model Perspective to Deep Neural Network Layer Dynamics/images/ff12fb8fa1c71782a17d6a8eed1853bcdf3eae74a917e5b7fff81899489d6c83.jpg)

### Tables

![73cce299ae9f774f1aa1ee167d31e97d29c87c87abc91eb747c9c72464f3fba8.jpg](../iclr_results/1276_From Layers to States_ A State Space Model Perspective to Deep Neural Network Layer Dynamics/tables/73cce299ae9f774f1aa1ee167d31e97d29c87c87abc91eb747c9c72464f3fba8.jpg)

![84c993c4ec6c4a12c08048b03772e1016b79607d58f7ddc6201a87129103f7dd.jpg](../iclr_results/1276_From Layers to States_ A State Space Model Perspective to Deep Neural Network Layer Dynamics/tables/84c993c4ec6c4a12c08048b03772e1016b79607d58f7ddc6201a87129103f7dd.jpg)

![885c0c37899ca4d70a8bd98531779ea7e3ab74996ba5c04321ba879149068cce.jpg](../iclr_results/1276_From Layers to States_ A State Space Model Perspective to Deep Neural Network Layer Dynamics/tables/885c0c37899ca4d70a8bd98531779ea7e3ab74996ba5c04321ba879149068cce.jpg)

![8a2eed41573190497c07d1ffdd3c53b8e575e979ca6241b7af38b7366e04f4a2.jpg](../iclr_results/1276_From Layers to States_ A State Space Model Perspective to Deep Neural Network Layer Dynamics/tables/8a2eed41573190497c07d1ffdd3c53b8e575e979ca6241b7af38b7366e04f4a2.jpg)

![9f818aca127b0d9e89f3caf2ce049685a39bfeeb1c237608c8a97aa1e2569ae1.jpg](../iclr_results/1276_From Layers to States_ A State Space Model Perspective to Deep Neural Network Layer Dynamics/tables/9f818aca127b0d9e89f3caf2ce049685a39bfeeb1c237608c8a97aa1e2569ae1.jpg)

![adb15cba34d888291ef011d9d60e943b6757039394b0766e8d957dcfae2ffe93.jpg](../iclr_results/1276_From Layers to States_ A State Space Model Perspective to Deep Neural Network Layer Dynamics/tables/adb15cba34d888291ef011d9d60e943b6757039394b0766e8d957dcfae2ffe93.jpg)

![d61ea9c7277a232859e5a856161232c17624354f8a7d8cabd30e15dcf9228a97.jpg](../iclr_results/1276_From Layers to States_ A State Space Model Perspective to Deep Neural Network Layer Dynamics/tables/d61ea9c7277a232859e5a856161232c17624354f8a7d8cabd30e15dcf9228a97.jpg)

![d9bf0fd2ce6ef83f645ee59e78fbf6a6c6b134eabfcdd9cdb87315ed94098407.jpg](../iclr_results/1276_From Layers to States_ A State Space Model Perspective to Deep Neural Network Layer Dynamics/tables/d9bf0fd2ce6ef83f645ee59e78fbf6a6c6b134eabfcdd9cdb87315ed94098407.jpg)

![df03eba27486077cf335175f44a93ee0350d4c986d247694be9b95306ce0c33b.jpg](../iclr_results/1276_From Layers to States_ A State Space Model Perspective to Deep Neural Network Layer Dynamics/tables/df03eba27486077cf335175f44a93ee0350d4c986d247694be9b95306ce0c33b.jpg)

## TIGeR: Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models


### Images

![224b2f86f1fc67de0df8f8eaec08792f6195a0f43166ff9cbe0116a48a2eefd5.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/images/224b2f86f1fc67de0df8f8eaec08792f6195a0f43166ff9cbe0116a48a2eefd5.jpg)

![298aca82f186940d6d22f30fd577829c87e60e7aa91b3761206bb64ad3a310e6.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/images/298aca82f186940d6d22f30fd577829c87e60e7aa91b3761206bb64ad3a310e6.jpg)

![39bdc699acd099896da556f0b31fbad022ca8825d234dbc8677201c82c13f1f4.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/images/39bdc699acd099896da556f0b31fbad022ca8825d234dbc8677201c82c13f1f4.jpg)

![45445b904d645690f135a10fa4e2774ac0bee0d092e017d80226e2528a355ea5.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/images/45445b904d645690f135a10fa4e2774ac0bee0d092e017d80226e2528a355ea5.jpg)

![5400c35efa015465650ef34baedfc0bde8f6ff61f28f445110a58b2166519d1f.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/images/5400c35efa015465650ef34baedfc0bde8f6ff61f28f445110a58b2166519d1f.jpg)

![64ed98ca28ea69c979d70ba5812824cd2044c6a3e87bf63476d630ffdddc81d2.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/images/64ed98ca28ea69c979d70ba5812824cd2044c6a3e87bf63476d630ffdddc81d2.jpg)

![67e33b371b055d6f23bd72591b40d3b27ae72c07cc78cbb4f691f3af9ad9ca93.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/images/67e33b371b055d6f23bd72591b40d3b27ae72c07cc78cbb4f691f3af9ad9ca93.jpg)

![72cee0c271f6e31f8c74131a0e5221cd630a472e3f95b5408f1eb3613f7878e4.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/images/72cee0c271f6e31f8c74131a0e5221cd630a472e3f95b5408f1eb3613f7878e4.jpg)

![74edb11764e023aa949c781706b2e945e5f5088020650424aa824b42c7cacd39.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/images/74edb11764e023aa949c781706b2e945e5f5088020650424aa824b42c7cacd39.jpg)

![797a09fce3415eb186d9feb2320ab2bc42d21a29dd81a750c6b105174d6d90c4.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/images/797a09fce3415eb186d9feb2320ab2bc42d21a29dd81a750c6b105174d6d90c4.jpg)

![8b8f8f1cb2025cf18f42b34a6f3982d89d0ab2ad94d702d29f8869c690ba8726.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/images/8b8f8f1cb2025cf18f42b34a6f3982d89d0ab2ad94d702d29f8869c690ba8726.jpg)

![9ee2d8e64a598850a96ff5e752fa81debea205965a11284d8d349ce082ec50b0.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/images/9ee2d8e64a598850a96ff5e752fa81debea205965a11284d8d349ce082ec50b0.jpg)

![b4e5ec3d127e5c878eb31782b19ba09414429116fec819fbfec962095cd9e665.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/images/b4e5ec3d127e5c878eb31782b19ba09414429116fec819fbfec962095cd9e665.jpg)

![c5b38477f6dc6de860f65b81a340ac2b372e3cdc9b405951372e29ecd7fee3b5.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/images/c5b38477f6dc6de860f65b81a340ac2b372e3cdc9b405951372e29ecd7fee3b5.jpg)

![edb31390c16af4130451d8c3e25fcd48caef5da152f5b2a6b7ddfa62615c8f71.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/images/edb31390c16af4130451d8c3e25fcd48caef5da152f5b2a6b7ddfa62615c8f71.jpg)

![f6f217f418cc74b505af8805381078fd29c132e43e5fcfe86bda9945431c2fec.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/images/f6f217f418cc74b505af8805381078fd29c132e43e5fcfe86bda9945431c2fec.jpg)

![f84fda227c57279944976929b65d90052e7619487c059125e4a45948ed454d57.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/images/f84fda227c57279944976929b65d90052e7619487c059125e4a45948ed454d57.jpg)

### Tables

![0552fa60d853fd06a0bf2e00a3eac3154dc221943ccc69068ab0213664bf2079.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/tables/0552fa60d853fd06a0bf2e00a3eac3154dc221943ccc69068ab0213664bf2079.jpg)

![26814cc732f1337704ab14f30ab40af99d1d3bdf269276b8647d26a4492279ee.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/tables/26814cc732f1337704ab14f30ab40af99d1d3bdf269276b8647d26a4492279ee.jpg)

![2ecc23ac1dcd05a2d8a0833d337fbffc6c8d0a3b0d0e8f13c3910356e156499b.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/tables/2ecc23ac1dcd05a2d8a0833d337fbffc6c8d0a3b0d0e8f13c3910356e156499b.jpg)

![4a3ea239d7c505d3ac25bca769ff7057daca18c64942661abd6759c21bbd91a1.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/tables/4a3ea239d7c505d3ac25bca769ff7057daca18c64942661abd6759c21bbd91a1.jpg)

![515ca188e3c7373b18f319c419bd092b4285d9cb895a2b22eabcf1bea5bcd77a.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/tables/515ca188e3c7373b18f319c419bd092b4285d9cb895a2b22eabcf1bea5bcd77a.jpg)

![653e6e89e1cb756bf0b10aa9b3bc2e9f31a0d001ffe94f0e09987a4d547322bb.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/tables/653e6e89e1cb756bf0b10aa9b3bc2e9f31a0d001ffe94f0e09987a4d547322bb.jpg)

![6c423ed9f1b273dd5456cf249468fb43251601c8e20bd8e0fb5bde6831ad4759.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/tables/6c423ed9f1b273dd5456cf249468fb43251601c8e20bd8e0fb5bde6831ad4759.jpg)

![6cd07310be545424da088b52727d4cc7b13e92977fd28f700b72a6531312231d.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/tables/6cd07310be545424da088b52727d4cc7b13e92977fd28f700b72a6531312231d.jpg)

![713b6fdf90470f79f7e4dbdc713dd5ed8f676637193c8ffefa662c9b65b60211.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/tables/713b6fdf90470f79f7e4dbdc713dd5ed8f676637193c8ffefa662c9b65b60211.jpg)

![741bc6a497d7e3527395e879de5edb77164733765d06b08e5f77bd8bddc28799.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/tables/741bc6a497d7e3527395e879de5edb77164733765d06b08e5f77bd8bddc28799.jpg)

![7aa730d629a9976403d653c00ac986b92d494be26335be1bd9d852672f2360df.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/tables/7aa730d629a9976403d653c00ac986b92d494be26335be1bd9d852672f2360df.jpg)

![82b11ff734eecbfcb4523685abcdb498b506f873b2dac963cd48907ac59d909a.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/tables/82b11ff734eecbfcb4523685abcdb498b506f873b2dac963cd48907ac59d909a.jpg)

![8fa85f973a2a1487bccb45298caf7337178334a31779556ac4925cffdb047704.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/tables/8fa85f973a2a1487bccb45298caf7337178334a31779556ac4925cffdb047704.jpg)

![906cdfb1e084a7a298dfdf129b6504d34ac960cd7773bdaae66f68d6059ac9e4.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/tables/906cdfb1e084a7a298dfdf129b6504d34ac960cd7773bdaae66f68d6059ac9e4.jpg)

![97636884efb9941f157634615004a6b763105b6ab4514678f2e9fe41e785b403.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/tables/97636884efb9941f157634615004a6b763105b6ab4514678f2e9fe41e785b403.jpg)

![cb13260378b34c4315ddc8ffdbe0d976162c2b8c74b1c01346fd7eddd7b9e5e0.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/tables/cb13260378b34c4315ddc8ffdbe0d976162c2b8c74b1c01346fd7eddd7b9e5e0.jpg)

![d7c1b5dd15b47ebea7fe21a6f70ad5509a97a764112e51e788c0a0961fb0456c.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/tables/d7c1b5dd15b47ebea7fe21a6f70ad5509a97a764112e51e788c0a0961fb0456c.jpg)

![e1679b566301791f2131447045b7387fc33167bdf34249f052c7dbc92d2a6d2e.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/tables/e1679b566301791f2131447045b7387fc33167bdf34249f052c7dbc92d2a6d2e.jpg)

![fe2e848b85657da45371b4b34b0d2694dc2fb62ec9b92e384b45a4ca5daa3fc7.jpg](../iclr_results/1277_TIGeR_ Unifying Text-to-Image Generation and Retrieval with Large Multimodal Models/tables/fe2e848b85657da45371b4b34b0d2694dc2fb62ec9b92e384b45a4ca5daa3fc7.jpg)

## Towards Federated RLHF with Aggregated Client Preference for LLMs


### Images

![02c346ddee27970ccf06067e32db17439b3232c23441ae1fd8d5027ca6aa73bf.jpg](../iclr_results/1278_Towards Federated RLHF with Aggregated Client Preference for LLMs/images/02c346ddee27970ccf06067e32db17439b3232c23441ae1fd8d5027ca6aa73bf.jpg)

![169d7552ab7bcf1e24d232738034ccfb860e46bfafe5c60304321f1eea93da5f.jpg](../iclr_results/1278_Towards Federated RLHF with Aggregated Client Preference for LLMs/images/169d7552ab7bcf1e24d232738034ccfb860e46bfafe5c60304321f1eea93da5f.jpg)

![409887d64a8bf8ca4aa88517e040065ec32bcbb0e375876b04fc5637aca2fa72.jpg](../iclr_results/1278_Towards Federated RLHF with Aggregated Client Preference for LLMs/images/409887d64a8bf8ca4aa88517e040065ec32bcbb0e375876b04fc5637aca2fa72.jpg)

![9e0eb681196def142b6bbd6c67c8b008b72e323ab4cc572d52e442162aeb0d81.jpg](../iclr_results/1278_Towards Federated RLHF with Aggregated Client Preference for LLMs/images/9e0eb681196def142b6bbd6c67c8b008b72e323ab4cc572d52e442162aeb0d81.jpg)

### Tables

![02e9e023a555f2a917e7a8a7ee8cc674d8a9b3c488794e9c15b51d5f53e8d7ad.jpg](../iclr_results/1278_Towards Federated RLHF with Aggregated Client Preference for LLMs/tables/02e9e023a555f2a917e7a8a7ee8cc674d8a9b3c488794e9c15b51d5f53e8d7ad.jpg)

![0b3f5872802ff7ab3dbf9593cc801d452c1eebffbcbb18997bb118020afbcd5b.jpg](../iclr_results/1278_Towards Federated RLHF with Aggregated Client Preference for LLMs/tables/0b3f5872802ff7ab3dbf9593cc801d452c1eebffbcbb18997bb118020afbcd5b.jpg)

![1423cc6c67e003f8f402d8a32a72daeea720b4c8aecc465de50dc6bb76fd7850.jpg](../iclr_results/1278_Towards Federated RLHF with Aggregated Client Preference for LLMs/tables/1423cc6c67e003f8f402d8a32a72daeea720b4c8aecc465de50dc6bb76fd7850.jpg)

![74f07e095837da95f66cf4045eedad13b7e823959c2af30cb76678f2efcd5c0d.jpg](../iclr_results/1278_Towards Federated RLHF with Aggregated Client Preference for LLMs/tables/74f07e095837da95f66cf4045eedad13b7e823959c2af30cb76678f2efcd5c0d.jpg)

![79520d13381f3e92331b7d65405fd9fe3d191ddaa5765e485eec02eb2b376e2d.jpg](../iclr_results/1278_Towards Federated RLHF with Aggregated Client Preference for LLMs/tables/79520d13381f3e92331b7d65405fd9fe3d191ddaa5765e485eec02eb2b376e2d.jpg)

![9f37e1678df179372828e13748fe1386f0fdf1adc9ddc9080913cba553d1f1a3.jpg](../iclr_results/1278_Towards Federated RLHF with Aggregated Client Preference for LLMs/tables/9f37e1678df179372828e13748fe1386f0fdf1adc9ddc9080913cba553d1f1a3.jpg)

![f4328eb43471ad4887c2bc38866ae72e6053a6b7df6655efc6526769f503d28f.jpg](../iclr_results/1278_Towards Federated RLHF with Aggregated Client Preference for LLMs/tables/f4328eb43471ad4887c2bc38866ae72e6053a6b7df6655efc6526769f503d28f.jpg)

![fce713f276b93ab23a127491659d7abae83c1eb172ef2656ad2340353bf2eb99.jpg](../iclr_results/1278_Towards Federated RLHF with Aggregated Client Preference for LLMs/tables/fce713f276b93ab23a127491659d7abae83c1eb172ef2656ad2340353bf2eb99.jpg)

![fd9aafb06034c8e8add24bf5b28db23c23b11e67326a00332f315faf0ed703c2.jpg](../iclr_results/1278_Towards Federated RLHF with Aggregated Client Preference for LLMs/tables/fd9aafb06034c8e8add24bf5b28db23c23b11e67326a00332f315faf0ed703c2.jpg)

## Subtask-Aware Visual Reward Learning from Segmented Demonstrations


### Images

![0f612a71f894cf84d17cc7fbc02a208e0039a7d1b4ccf82b9ad5ce3ef27b2afc.jpg](../iclr_results/1279_Subtask-Aware Visual Reward Learning from Segmented Demonstrations/images/0f612a71f894cf84d17cc7fbc02a208e0039a7d1b4ccf82b9ad5ce3ef27b2afc.jpg)

![39c885a934085874a0d13288101aaa077994b52e40788e355537ca68d6c676ce.jpg](../iclr_results/1279_Subtask-Aware Visual Reward Learning from Segmented Demonstrations/images/39c885a934085874a0d13288101aaa077994b52e40788e355537ca68d6c676ce.jpg)

![540197d854fd3589107cb1af0574f0f90d7e8b82a8f29212dd03ec62477569d8.jpg](../iclr_results/1279_Subtask-Aware Visual Reward Learning from Segmented Demonstrations/images/540197d854fd3589107cb1af0574f0f90d7e8b82a8f29212dd03ec62477569d8.jpg)

![5a2166b99b0de9d7fc680f6e5a75c95347a1d0b6b8f5a204e7ef9c965ea81a3f.jpg](../iclr_results/1279_Subtask-Aware Visual Reward Learning from Segmented Demonstrations/images/5a2166b99b0de9d7fc680f6e5a75c95347a1d0b6b8f5a204e7ef9c965ea81a3f.jpg)

![74d91ba8805c97e91d2407006a6101b049675bd14b72d24a4140dc871ef1f4d0.jpg](../iclr_results/1279_Subtask-Aware Visual Reward Learning from Segmented Demonstrations/images/74d91ba8805c97e91d2407006a6101b049675bd14b72d24a4140dc871ef1f4d0.jpg)

![9588761002ea19d1435d7d290e1c645e8f758111b557796ae80b2285c99059c3.jpg](../iclr_results/1279_Subtask-Aware Visual Reward Learning from Segmented Demonstrations/images/9588761002ea19d1435d7d290e1c645e8f758111b557796ae80b2285c99059c3.jpg)

![a14c47a30a1fb184687fdc6f89fe76dfaa0df8c6fa0edcaf9f3d1079d72e549f.jpg](../iclr_results/1279_Subtask-Aware Visual Reward Learning from Segmented Demonstrations/images/a14c47a30a1fb184687fdc6f89fe76dfaa0df8c6fa0edcaf9f3d1079d72e549f.jpg)

![ae09e30a51cec1a093893363859105ba78ae2d42dd4d32db88832ecb494e9c58.jpg](../iclr_results/1279_Subtask-Aware Visual Reward Learning from Segmented Demonstrations/images/ae09e30a51cec1a093893363859105ba78ae2d42dd4d32db88832ecb494e9c58.jpg)

![b1cbf2e6c4ae5138b0b90a8c4dce5067306c586e73206bb816e854ecb8b86898.jpg](../iclr_results/1279_Subtask-Aware Visual Reward Learning from Segmented Demonstrations/images/b1cbf2e6c4ae5138b0b90a8c4dce5067306c586e73206bb816e854ecb8b86898.jpg)

![b814cfe35cf47d9ff923f657957b104cae196e93fcbaec240cc0fc7d23766558.jpg](../iclr_results/1279_Subtask-Aware Visual Reward Learning from Segmented Demonstrations/images/b814cfe35cf47d9ff923f657957b104cae196e93fcbaec240cc0fc7d23766558.jpg)

![db8514cad66cb895966b795ee47b0fd8aa5b3375873e81fc291e9a5e9b30db65.jpg](../iclr_results/1279_Subtask-Aware Visual Reward Learning from Segmented Demonstrations/images/db8514cad66cb895966b795ee47b0fd8aa5b3375873e81fc291e9a5e9b30db65.jpg)

![dc6310897b6bf665442705e4fe9b2d2b112aa39eb4dea6875285c254288a7a22.jpg](../iclr_results/1279_Subtask-Aware Visual Reward Learning from Segmented Demonstrations/images/dc6310897b6bf665442705e4fe9b2d2b112aa39eb4dea6875285c254288a7a22.jpg)

### Tables

![0d2f2922db1faf1ecb3665d3b3ffcb69ae8ee4f92f7febfe5767c5b3ef7e9b84.jpg](../iclr_results/1279_Subtask-Aware Visual Reward Learning from Segmented Demonstrations/tables/0d2f2922db1faf1ecb3665d3b3ffcb69ae8ee4f92f7febfe5767c5b3ef7e9b84.jpg)

![2684e6fc95cfc413bcacfab2b5799cc83f080e3f11bf3d79ad8ffa932dbf48d8.jpg](../iclr_results/1279_Subtask-Aware Visual Reward Learning from Segmented Demonstrations/tables/2684e6fc95cfc413bcacfab2b5799cc83f080e3f11bf3d79ad8ffa932dbf48d8.jpg)

![61a49b783709bef1db844ed697511a1f59d39f886f6998860c0af6ddb46bf4c4.jpg](../iclr_results/1279_Subtask-Aware Visual Reward Learning from Segmented Demonstrations/tables/61a49b783709bef1db844ed697511a1f59d39f886f6998860c0af6ddb46bf4c4.jpg)

![6c92eaf81c44f5ef39a9484fe57d81bd493d80a5d58b9e9b8aef8d7a10fe4552.jpg](../iclr_results/1279_Subtask-Aware Visual Reward Learning from Segmented Demonstrations/tables/6c92eaf81c44f5ef39a9484fe57d81bd493d80a5d58b9e9b8aef8d7a10fe4552.jpg)

![8b1c24d7c99ea03ba7e58f2d8c98b7c5b75d30ee6e175d7510644030d8507de0.jpg](../iclr_results/1279_Subtask-Aware Visual Reward Learning from Segmented Demonstrations/tables/8b1c24d7c99ea03ba7e58f2d8c98b7c5b75d30ee6e175d7510644030d8507de0.jpg)

![c66f15f4724df7916cc45ed341f964a2ced3a466e679a841af455a7cf570822c.jpg](../iclr_results/1279_Subtask-Aware Visual Reward Learning from Segmented Demonstrations/tables/c66f15f4724df7916cc45ed341f964a2ced3a466e679a841af455a7cf570822c.jpg)

![dc0a02a28dc31bf680174b084d3b4aeb320a6ca11b2a7fbc300673058242739f.jpg](../iclr_results/1279_Subtask-Aware Visual Reward Learning from Segmented Demonstrations/tables/dc0a02a28dc31bf680174b084d3b4aeb320a6ca11b2a7fbc300673058242739f.jpg)

![e642f71fd8ee1a4216cecce1cb58dbeafcdd8c4faadaca8a53280cb6d77f3670.jpg](../iclr_results/1279_Subtask-Aware Visual Reward Learning from Segmented Demonstrations/tables/e642f71fd8ee1a4216cecce1cb58dbeafcdd8c4faadaca8a53280cb6d77f3670.jpg)

![f72d5f8aa9ea0aa490826c9f9f6f0ae2ab72907f5af7c63db6eb971d82eeec01.jpg](../iclr_results/1279_Subtask-Aware Visual Reward Learning from Segmented Demonstrations/tables/f72d5f8aa9ea0aa490826c9f9f6f0ae2ab72907f5af7c63db6eb971d82eeec01.jpg)

## From Isolated Conversations to Hierarchical Schemas: Dynamic Tree Memory Representation for LLMs


### Images

![1026ecfd2ee0e6cf2b88a518c96a8bf1f96c50793ef7205c95cdccf71712ff5f.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/images/1026ecfd2ee0e6cf2b88a518c96a8bf1f96c50793ef7205c95cdccf71712ff5f.jpg)

![18b1af7271f870e9d91ece980e8bd7f13658dfdb599dff67041fe175c56ac7bc.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/images/18b1af7271f870e9d91ece980e8bd7f13658dfdb599dff67041fe175c56ac7bc.jpg)

![4ced027ee5d405afa54c7d14ef4a287976eed39aaa4e8cdde0bb6f3e49cbfe5e.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/images/4ced027ee5d405afa54c7d14ef4a287976eed39aaa4e8cdde0bb6f3e49cbfe5e.jpg)

![50e6ef29cc16b405d7849d3d362216305ded486f062ce8a2a4b88ed91a8db60b.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/images/50e6ef29cc16b405d7849d3d362216305ded486f062ce8a2a4b88ed91a8db60b.jpg)

![531d9decb730962250500ce5dd4a652f6121d7e82d7de5bffe2acc456fe38181.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/images/531d9decb730962250500ce5dd4a652f6121d7e82d7de5bffe2acc456fe38181.jpg)

![64fb5479e7f4b51832e808e8429565f5dc7dc43713b3a10b095da2d831dc0a2f.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/images/64fb5479e7f4b51832e808e8429565f5dc7dc43713b3a10b095da2d831dc0a2f.jpg)

![7b8471c1f1ead1593ffd121bbe947c37b2f37ed1d3996dd5bc9e74d2adc8f8f2.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/images/7b8471c1f1ead1593ffd121bbe947c37b2f37ed1d3996dd5bc9e74d2adc8f8f2.jpg)

![7f1dbeeecc4b90bf75a9cfec3cc4950d57896f80f3c752cebe5c2156437331ee.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/images/7f1dbeeecc4b90bf75a9cfec3cc4950d57896f80f3c752cebe5c2156437331ee.jpg)

![855dc9b9feb72cdd5e47cf46c4c1544c928718c8a86f9104216c7ef8b6c99689.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/images/855dc9b9feb72cdd5e47cf46c4c1544c928718c8a86f9104216c7ef8b6c99689.jpg)

![a9c9102ad581609d1f1be8997d9e77c72bb54ba6ca4d60cdbdc63d8b650f070b.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/images/a9c9102ad581609d1f1be8997d9e77c72bb54ba6ca4d60cdbdc63d8b650f070b.jpg)

![ebb1aaf99c57335d8a782826d8dcfaccc17870a8750989e00c5b6ef566447a68.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/images/ebb1aaf99c57335d8a782826d8dcfaccc17870a8750989e00c5b6ef566447a68.jpg)

![f198bc69ca17e81304c22434bde95eef8c3e6f2ee4d371764cf0acc4e7ac363e.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/images/f198bc69ca17e81304c22434bde95eef8c3e6f2ee4d371764cf0acc4e7ac363e.jpg)

![f6974286f0bdcd1cdd8c107c712555f532aac6b6abea1e9a8da2dbb88a5acb07.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/images/f6974286f0bdcd1cdd8c107c712555f532aac6b6abea1e9a8da2dbb88a5acb07.jpg)

![fd3972c08a172324536b022486b17ec179340bcc0b274daa4edc1e37c1bbabff.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/images/fd3972c08a172324536b022486b17ec179340bcc0b274daa4edc1e37c1bbabff.jpg)

### Tables

![1931260f955104d666e2d017f83496e52e5e3914db0a7dcd44fd904282c5e44c.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/tables/1931260f955104d666e2d017f83496e52e5e3914db0a7dcd44fd904282c5e44c.jpg)

![1f835665e4e1aea1f6150c6bc0325c0207b1e2167112ca58405eb8caa43c5226.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/tables/1f835665e4e1aea1f6150c6bc0325c0207b1e2167112ca58405eb8caa43c5226.jpg)

![2eb7c50eac14f54624fea4c2f48205998b0d936accbad4c0d7f4c5ef3dcb1c55.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/tables/2eb7c50eac14f54624fea4c2f48205998b0d936accbad4c0d7f4c5ef3dcb1c55.jpg)

![2f42ea9b11ecf766f17117c9db59bae0811643df0e6f9dda435f66bc8e63a77a.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/tables/2f42ea9b11ecf766f17117c9db59bae0811643df0e6f9dda435f66bc8e63a77a.jpg)

![4126921562aefe4687c44b78c7988b6fabf3f134578ff3b170c5cf4fcf230b6d.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/tables/4126921562aefe4687c44b78c7988b6fabf3f134578ff3b170c5cf4fcf230b6d.jpg)

![58a93ac302bb3d12f279fe0f7618a3c8ab166b951f9d84b12797e0c8a7e40d6a.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/tables/58a93ac302bb3d12f279fe0f7618a3c8ab166b951f9d84b12797e0c8a7e40d6a.jpg)

![7ed7cd66fa35b469cfce416d97759efe59b0c3814b948aa12c672cdf7ba8c526.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/tables/7ed7cd66fa35b469cfce416d97759efe59b0c3814b948aa12c672cdf7ba8c526.jpg)

![b94835e5308e494da2ca2e471874ed7bc8c2daa87c61f5962d76445480b73a9a.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/tables/b94835e5308e494da2ca2e471874ed7bc8c2daa87c61f5962d76445480b73a9a.jpg)

![bb752725a19e9f17fc8082ea464fa15e60babd8c69e64c8b6df128071f9e3be1.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/tables/bb752725a19e9f17fc8082ea464fa15e60babd8c69e64c8b6df128071f9e3be1.jpg)

![c45ea535aec1fa921e1d30d0c0f357c0de28c2a540cd71ff5f40f02c42db0108.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/tables/c45ea535aec1fa921e1d30d0c0f357c0de28c2a540cd71ff5f40f02c42db0108.jpg)

![d488fd62ba2cbdcf4838c74d053c2696fdbcd8d0ef66685c086bbc3041890702.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/tables/d488fd62ba2cbdcf4838c74d053c2696fdbcd8d0ef66685c086bbc3041890702.jpg)

![e160e5b721924c64065cbcf060c1c95b672074f253a611ddfee380f3fc60c127.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/tables/e160e5b721924c64065cbcf060c1c95b672074f253a611ddfee380f3fc60c127.jpg)

![ef8fcc62ae23351297333e61fbc395a6bd159dd6f7d368724d81681ab2a4f4c5.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/tables/ef8fcc62ae23351297333e61fbc395a6bd159dd6f7d368724d81681ab2a4f4c5.jpg)

![f5fc01b426686f2a46b762e67a1bae6f6b11a05393f2779bfb066ba25369bec5.jpg](../iclr_results/1280_From Isolated Conversations to Hierarchical Schemas_ Dynamic Tree Memory Representation for LLMs/tables/f5fc01b426686f2a46b762e67a1bae6f6b11a05393f2779bfb066ba25369bec5.jpg)

## Web Agents with World Models: Learning and Leveraging Environment Dynamics in Web Navigation


### Images

![0be30ee81e517b36ac2b4841a064d67f77edeaf7f5a44712842f108803f0a414.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/images/0be30ee81e517b36ac2b4841a064d67f77edeaf7f5a44712842f108803f0a414.jpg)

![1eebacfe684fb21e1ab0b0572fd3cbc96e444da2cdd950f55922ed21aceb3f9c.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/images/1eebacfe684fb21e1ab0b0572fd3cbc96e444da2cdd950f55922ed21aceb3f9c.jpg)

![2745b7cdb6d5a4233d7c1ed6d7c5a2e841f8bdeabb6c1083686856dbc38a1c1a.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/images/2745b7cdb6d5a4233d7c1ed6d7c5a2e841f8bdeabb6c1083686856dbc38a1c1a.jpg)

![305260342b514028861164656bd1b955009891612c96bce2cb03135176cd74da.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/images/305260342b514028861164656bd1b955009891612c96bce2cb03135176cd74da.jpg)

![46dec5a5eec40d607bb959dea8cdadfb726a268850da5e0ac540f9fab76bb0a9.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/images/46dec5a5eec40d607bb959dea8cdadfb726a268850da5e0ac540f9fab76bb0a9.jpg)

![5851c729ad8181cdf73de83768e28d2908811822c7ac8fe318a315649a64712b.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/images/5851c729ad8181cdf73de83768e28d2908811822c7ac8fe318a315649a64712b.jpg)

![5b7a3f8154fed944d6b16f2e887f908a4290f47f0a79edfc4cf0921d03c658e2.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/images/5b7a3f8154fed944d6b16f2e887f908a4290f47f0a79edfc4cf0921d03c658e2.jpg)

![5de1e7b4e78e99a5344956e26b34122306fc6ffe5479935f63afef1200d6b83f.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/images/5de1e7b4e78e99a5344956e26b34122306fc6ffe5479935f63afef1200d6b83f.jpg)

![603d9e73ae75f325cd1a2a22b2c306d82b73e3b11d6d8523016b07e5a5a3761f.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/images/603d9e73ae75f325cd1a2a22b2c306d82b73e3b11d6d8523016b07e5a5a3761f.jpg)

![6693deba77c86481d3165693c22f505bed4b488d2a3079e952f8489177e6d89d.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/images/6693deba77c86481d3165693c22f505bed4b488d2a3079e952f8489177e6d89d.jpg)

![6c263922a66ba439adf5ab77bd0e4f7c9c7534d257a9a703573b6deb5892ab43.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/images/6c263922a66ba439adf5ab77bd0e4f7c9c7534d257a9a703573b6deb5892ab43.jpg)

![77b01ccaf7cf9364875081b6081d5e790c1c48d01867da7fd0d5be986eba3fa7.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/images/77b01ccaf7cf9364875081b6081d5e790c1c48d01867da7fd0d5be986eba3fa7.jpg)

![7ea4e55bbb3466cacb90d3a6dcd14b0b693c7dcfd97a52c009b9ea9810e6cbf2.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/images/7ea4e55bbb3466cacb90d3a6dcd14b0b693c7dcfd97a52c009b9ea9810e6cbf2.jpg)

![9beb3f76c2665db8a48d92f08fdc282c4daa1641108e1fd8c5495eb572b882cb.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/images/9beb3f76c2665db8a48d92f08fdc282c4daa1641108e1fd8c5495eb572b882cb.jpg)

![b9fb4482d8bf4bbb03f1cf612ed81dac9420e5f289549cf74a5a4050c2e23901.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/images/b9fb4482d8bf4bbb03f1cf612ed81dac9420e5f289549cf74a5a4050c2e23901.jpg)

![bb258174d19b8bc3691b522cdcfb6ab85ec838fc474da5dab334246fa43f7071.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/images/bb258174d19b8bc3691b522cdcfb6ab85ec838fc474da5dab334246fa43f7071.jpg)

![bd45083e91046eeee7abef85def1b0db376bd7db7147785483c46657eefb83e6.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/images/bd45083e91046eeee7abef85def1b0db376bd7db7147785483c46657eefb83e6.jpg)

![be0d3c6a372d9099d1623fce92b0a28407d8ca4c4f3ed2434a3da6a0b8e23953.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/images/be0d3c6a372d9099d1623fce92b0a28407d8ca4c4f3ed2434a3da6a0b8e23953.jpg)

![d45e89d15de8525c49f60475fecd6812bbd7535554343abf7491f005a32be7ed.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/images/d45e89d15de8525c49f60475fecd6812bbd7535554343abf7491f005a32be7ed.jpg)

![df1ec0852655e6f3ced84409d0cabff11317584f61a776f115fdc357e88d43f6.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/images/df1ec0852655e6f3ced84409d0cabff11317584f61a776f115fdc357e88d43f6.jpg)

### Tables

![0b1cde7b58d4fb77c315f210a1fd590e03be34b920e5f8550102931a43e0b61a.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/tables/0b1cde7b58d4fb77c315f210a1fd590e03be34b920e5f8550102931a43e0b61a.jpg)

![142d5b74b4ef3689b2a924575b73b46d4c527375d7e4059d2bde298b89396796.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/tables/142d5b74b4ef3689b2a924575b73b46d4c527375d7e4059d2bde298b89396796.jpg)

![2210bd552e3fb189bec49cd65b5873936f309f101c6e85b211397533c3f94347.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/tables/2210bd552e3fb189bec49cd65b5873936f309f101c6e85b211397533c3f94347.jpg)

![236d18e265d23acd86341b1e25f91b117b7f76eb9f9d339a2984955e76215800.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/tables/236d18e265d23acd86341b1e25f91b117b7f76eb9f9d339a2984955e76215800.jpg)

![5b933a1aa531437654df4d347e5ba295b7f0bc13d004e7f816035c6c13a44d03.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/tables/5b933a1aa531437654df4d347e5ba295b7f0bc13d004e7f816035c6c13a44d03.jpg)

![6c7cc48dc77c6d49030aeda022a96228c5a80fcaaad53b1ef96adca46827c605.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/tables/6c7cc48dc77c6d49030aeda022a96228c5a80fcaaad53b1ef96adca46827c605.jpg)

![72d803487af92746f1248f4bc249144c84e98d89b6e2aaca5760fe91367602c9.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/tables/72d803487af92746f1248f4bc249144c84e98d89b6e2aaca5760fe91367602c9.jpg)

![967b0d84c099af88d6f184142996aeea847b8e85000e08ee0dcf3ac16a46ce59.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/tables/967b0d84c099af88d6f184142996aeea847b8e85000e08ee0dcf3ac16a46ce59.jpg)

![c077c263d3bdbea656a82b5fad8d98ed4f09c5017c6246029231b2ceccea2fad.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/tables/c077c263d3bdbea656a82b5fad8d98ed4f09c5017c6246029231b2ceccea2fad.jpg)

![cb29e21c6fc63e9a768bd7d2e85a67c764e2a9eaf1066055100ae374adf68d2a.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/tables/cb29e21c6fc63e9a768bd7d2e85a67c764e2a9eaf1066055100ae374adf68d2a.jpg)

![ccf28c9cdb69550755ad37a5c9bf24f8d61c440ed83e371f661a7f7571605fb0.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/tables/ccf28c9cdb69550755ad37a5c9bf24f8d61c440ed83e371f661a7f7571605fb0.jpg)

![f1e611754521bf14d3dda525246c07649af29bdbfd4de84187d0243b577e82c3.jpg](../iclr_results/1281_Web Agents with World Models_ Learning and Leveraging Environment Dynamics in Web Navigation/tables/f1e611754521bf14d3dda525246c07649af29bdbfd4de84187d0243b577e82c3.jpg)

## Quamba: A Post-Training Quantization Recipe for Selective State Space Models


### Images

![1d52f3f27fbbf949c4c4d8dfee1a3af65c286b892f17e7fbb46c086370f3d6c8.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/images/1d52f3f27fbbf949c4c4d8dfee1a3af65c286b892f17e7fbb46c086370f3d6c8.jpg)

![22ede0ffbf1f8efa7dddda18bbbe5eb0757ae9cbe98e0a6a68df6f7123fc2385.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/images/22ede0ffbf1f8efa7dddda18bbbe5eb0757ae9cbe98e0a6a68df6f7123fc2385.jpg)

![387fd2b13e62ed532f421b9e90780be403ac832ba50f744ef39823106524c05f.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/images/387fd2b13e62ed532f421b9e90780be403ac832ba50f744ef39823106524c05f.jpg)

![6ee050e7358823e1042401b6d1207277e3571781485a7c3e52ff7e475ff65638.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/images/6ee050e7358823e1042401b6d1207277e3571781485a7c3e52ff7e475ff65638.jpg)

![8047c41a8385bc1cf18addacfbd5ac8442c65ffd61a57ffe30fd0360ce04478e.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/images/8047c41a8385bc1cf18addacfbd5ac8442c65ffd61a57ffe30fd0360ce04478e.jpg)

![987d232e0400a10800efb621640301bc9d168f63ba0501ef9d903d2a9ab05d2d.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/images/987d232e0400a10800efb621640301bc9d168f63ba0501ef9d903d2a9ab05d2d.jpg)

![a0e47fcabf76db95a38b7964286534ef6e4607eeb7966d3cf53dbc46a0dcddeb.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/images/a0e47fcabf76db95a38b7964286534ef6e4607eeb7966d3cf53dbc46a0dcddeb.jpg)

![a1033a9772e4e8ad4eca61fbc0e01aa0fdcdcfc33249dbc6023b2758effdbca2.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/images/a1033a9772e4e8ad4eca61fbc0e01aa0fdcdcfc33249dbc6023b2758effdbca2.jpg)

![a56019a19c186491f374cabde546118aa024c73ee11c8875a4eb8cb18d98c874.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/images/a56019a19c186491f374cabde546118aa024c73ee11c8875a4eb8cb18d98c874.jpg)

![b2789429bde71b3c5649f3ab378515c5b86fd2ca07fb2268ca0cb93c2ffee349.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/images/b2789429bde71b3c5649f3ab378515c5b86fd2ca07fb2268ca0cb93c2ffee349.jpg)

![bc90f1ca3e6aa17daee04bb09e034242a302bfb0424463eabc9069c7fdc41231.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/images/bc90f1ca3e6aa17daee04bb09e034242a302bfb0424463eabc9069c7fdc41231.jpg)

![c43419bb1aeb6ee74e0db29fe425093fe2787b9ad155b7d05edfbe017419a953.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/images/c43419bb1aeb6ee74e0db29fe425093fe2787b9ad155b7d05edfbe017419a953.jpg)

![c7585ce7ee20f6a0a3276a775c6af762bce314e31229165dde7f98d97ed0b892.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/images/c7585ce7ee20f6a0a3276a775c6af762bce314e31229165dde7f98d97ed0b892.jpg)

![d2c800ddd4c353936b508871f360b6062fa5eb4685cd6e71c764e4d46d3cb7e1.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/images/d2c800ddd4c353936b508871f360b6062fa5eb4685cd6e71c764e4d46d3cb7e1.jpg)

![da907080f9b69795770a6c596d16d4235351cb7b22d5715f44b8e5b8ed24ebc3.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/images/da907080f9b69795770a6c596d16d4235351cb7b22d5715f44b8e5b8ed24ebc3.jpg)

![dc7a0a1e98d5d45ec35d3d10267d276814ae3b0bab8a668081bc952d5a2504e4.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/images/dc7a0a1e98d5d45ec35d3d10267d276814ae3b0bab8a668081bc952d5a2504e4.jpg)

### Tables

![10250bae916d896c4c36a4df79e968f641d2cb45dd8e54efea32dde5d2c215ce.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/tables/10250bae916d896c4c36a4df79e968f641d2cb45dd8e54efea32dde5d2c215ce.jpg)

![15cfdcfa69ee61d1447086b14e9df87a2b1a11758798cad9aa2b062815fe4b4a.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/tables/15cfdcfa69ee61d1447086b14e9df87a2b1a11758798cad9aa2b062815fe4b4a.jpg)

![2fb2b738ae2b5cec67274a4ae61b07878c15569a1103c9387d42f18ecb36373d.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/tables/2fb2b738ae2b5cec67274a4ae61b07878c15569a1103c9387d42f18ecb36373d.jpg)

![67d475545f40b96a2e861397c805f5b4dac7f667f93ea34a46d16bea06217e4d.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/tables/67d475545f40b96a2e861397c805f5b4dac7f667f93ea34a46d16bea06217e4d.jpg)

![7742c0aba17a07d6025e00fe6419968dfbaa253f9a1a3dfad56ceca5eec441ee.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/tables/7742c0aba17a07d6025e00fe6419968dfbaa253f9a1a3dfad56ceca5eec441ee.jpg)

![7b8657b6e73d86ae750313613a1e4f14497d6422e39662631f44d4b2c63c83cd.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/tables/7b8657b6e73d86ae750313613a1e4f14497d6422e39662631f44d4b2c63c83cd.jpg)

![8229cbe55962fec10f4ea65d2f2f2a3c849f836d52e9e9c0114aaec91d2df221.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/tables/8229cbe55962fec10f4ea65d2f2f2a3c849f836d52e9e9c0114aaec91d2df221.jpg)

![8edd87333aaa2c93009945ba3980d6c54cd0294145c98c83c91c070be8c103db.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/tables/8edd87333aaa2c93009945ba3980d6c54cd0294145c98c83c91c070be8c103db.jpg)

![9411e6213f3f72485ceee08cff53a65d300b7f0a635502ef7abaff6dccff60bf.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/tables/9411e6213f3f72485ceee08cff53a65d300b7f0a635502ef7abaff6dccff60bf.jpg)

![c7736432f568d46c28d4cfd7ba126fe14f33ad4fc3f646ea1b0b322ba40985ca.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/tables/c7736432f568d46c28d4cfd7ba126fe14f33ad4fc3f646ea1b0b322ba40985ca.jpg)

![e2689440644ef5fa8e4b4e11adb1d742e4102f5f4aa7c56986b27f3f6ed4dc78.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/tables/e2689440644ef5fa8e4b4e11adb1d742e4102f5f4aa7c56986b27f3f6ed4dc78.jpg)

![fa5941778cb4a2074d659ec1c0c300dc5f4b4750734c9d7050736a97324f4356.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/tables/fa5941778cb4a2074d659ec1c0c300dc5f4b4750734c9d7050736a97324f4356.jpg)

![fd9fc0f22a71d3b7a928f2632f1638b1c0200f09fd49618db61d0d5b1f101d73.jpg](../iclr_results/1282_Quamba_ A Post-Training Quantization Recipe for Selective State Space Models/tables/fd9fc0f22a71d3b7a928f2632f1638b1c0200f09fd49618db61d0d5b1f101d73.jpg)

## Active Learning for Continual Learning: Keeping the Past Alive in the Present


### Images

![2bdf9316831ef7a394971c7cfe962311d90fe35ec591dad58fe0de29230c7ac2.jpg](../iclr_results/1283_Active Learning for Continual Learning_ Keeping the Past Alive in the Present/images/2bdf9316831ef7a394971c7cfe962311d90fe35ec591dad58fe0de29230c7ac2.jpg)

![8bf61a1e491440fac1ecf78e649b24cb47a44c65146e79b803fa50453164f921.jpg](../iclr_results/1283_Active Learning for Continual Learning_ Keeping the Past Alive in the Present/images/8bf61a1e491440fac1ecf78e649b24cb47a44c65146e79b803fa50453164f921.jpg)

![b60de9107bbb78277de89e01740db4a10fe62e4d0586fee293c5538a3b001542.jpg](../iclr_results/1283_Active Learning for Continual Learning_ Keeping the Past Alive in the Present/images/b60de9107bbb78277de89e01740db4a10fe62e4d0586fee293c5538a3b001542.jpg)

![e5b9e29665bba34cb1f7287f393997217a9c90f0fd0c927695aed7c29ba3ae9c.jpg](../iclr_results/1283_Active Learning for Continual Learning_ Keeping the Past Alive in the Present/images/e5b9e29665bba34cb1f7287f393997217a9c90f0fd0c927695aed7c29ba3ae9c.jpg)

### Tables

![017c198fe16133b0606e4a570c485c3b74ff8966398142cf6b9f56cb52e1b7fe.jpg](../iclr_results/1283_Active Learning for Continual Learning_ Keeping the Past Alive in the Present/tables/017c198fe16133b0606e4a570c485c3b74ff8966398142cf6b9f56cb52e1b7fe.jpg)

![2fd6a80bada33119a48c028ae38a1f6cb7aa2dc78918b877e2c1bc32d7eb39a4.jpg](../iclr_results/1283_Active Learning for Continual Learning_ Keeping the Past Alive in the Present/tables/2fd6a80bada33119a48c028ae38a1f6cb7aa2dc78918b877e2c1bc32d7eb39a4.jpg)

![a7e3e197d59712d1a729daca4b157fe879ae0762b46327611636b9faab9c9b65.jpg](../iclr_results/1283_Active Learning for Continual Learning_ Keeping the Past Alive in the Present/tables/a7e3e197d59712d1a729daca4b157fe879ae0762b46327611636b9faab9c9b65.jpg)

![ba6ba99d0d778709172e7dce84afc02368046a3fa4454b36c226aec20d2d1820.jpg](../iclr_results/1283_Active Learning for Continual Learning_ Keeping the Past Alive in the Present/tables/ba6ba99d0d778709172e7dce84afc02368046a3fa4454b36c226aec20d2d1820.jpg)

![d9fbb5cf7dc793fbff454c12fe87022b592bdb1e5a46b9265c079052b7e59e9c.jpg](../iclr_results/1283_Active Learning for Continual Learning_ Keeping the Past Alive in the Present/tables/d9fbb5cf7dc793fbff454c12fe87022b592bdb1e5a46b9265c079052b7e59e9c.jpg)

![de56d0fb6457fb88095116d981446d07e44b52f438300e26bfe7a705bd9fc62f.jpg](../iclr_results/1283_Active Learning for Continual Learning_ Keeping the Past Alive in the Present/tables/de56d0fb6457fb88095116d981446d07e44b52f438300e26bfe7a705bd9fc62f.jpg)

![df71c7d65515c54d7314faab1daa12fdc06d501af3933309e5c2001e4206f760.jpg](../iclr_results/1283_Active Learning for Continual Learning_ Keeping the Past Alive in the Present/tables/df71c7d65515c54d7314faab1daa12fdc06d501af3933309e5c2001e4206f760.jpg)

![ff94c9d9147dc5268f6376907c7e5a4301368a8d13ecb5da8eaa75a1ea714b68.jpg](../iclr_results/1283_Active Learning for Continual Learning_ Keeping the Past Alive in the Present/tables/ff94c9d9147dc5268f6376907c7e5a4301368a8d13ecb5da8eaa75a1ea714b68.jpg)

## Ready-to-React: Online Reaction Policy for Two-Character Interaction Generation


### Images

![0fa340c67c2f098c41abf5f614a0e45eea3b97bb11bbe78b05fb00d2666c24b7.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/images/0fa340c67c2f098c41abf5f614a0e45eea3b97bb11bbe78b05fb00d2666c24b7.jpg)

![2384df880aefd16708c4d1e482743838b4730d53333a1b471f7634f8df2e1893.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/images/2384df880aefd16708c4d1e482743838b4730d53333a1b471f7634f8df2e1893.jpg)

![2c77a08fe50a3e55de8610ca7030a4befac6d78e141fcb3f32c4df60cea5d0d0.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/images/2c77a08fe50a3e55de8610ca7030a4befac6d78e141fcb3f32c4df60cea5d0d0.jpg)

![35580e6c13ac87f410bdacd3731e6e73b8a3de051bb8f1eff53da2315fb549ec.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/images/35580e6c13ac87f410bdacd3731e6e73b8a3de051bb8f1eff53da2315fb549ec.jpg)

![48ad52811390a8cb76f3efaec9ef25ad843d76ed481490c8ebc5d7c43745cbfa.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/images/48ad52811390a8cb76f3efaec9ef25ad843d76ed481490c8ebc5d7c43745cbfa.jpg)

![5d22347dd5c3ed775125592d534c9a4a00d5960ed13e847c48f5ed975dbdfa28.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/images/5d22347dd5c3ed775125592d534c9a4a00d5960ed13e847c48f5ed975dbdfa28.jpg)

![71e829b0d16a3f52e140ab5da19bf86d88f80afdb7aed42811e6f21e8a772915.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/images/71e829b0d16a3f52e140ab5da19bf86d88f80afdb7aed42811e6f21e8a772915.jpg)

![7e7063b138e5e8e333636b2eda3d220d136b45833d70c311344c3a6225a59fb7.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/images/7e7063b138e5e8e333636b2eda3d220d136b45833d70c311344c3a6225a59fb7.jpg)

![941e6002b5c364f375df1f41b39e8f3300fcd1c97e15d21fff32ae7cf0f5e786.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/images/941e6002b5c364f375df1f41b39e8f3300fcd1c97e15d21fff32ae7cf0f5e786.jpg)

### Tables

![06d2f7dead8b6b1f32a8212824f2ea5357437ca4ca70018cd06ef3f5671adab7.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/tables/06d2f7dead8b6b1f32a8212824f2ea5357437ca4ca70018cd06ef3f5671adab7.jpg)

![0cb802a8ca36d11d99a819b7063e76d18dd25a5ddb589c43f6d79fbc82f85ae7.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/tables/0cb802a8ca36d11d99a819b7063e76d18dd25a5ddb589c43f6d79fbc82f85ae7.jpg)

![0f44fcd1a354e225064cb299d25009cd0c802440fe50f618834b52b658b06984.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/tables/0f44fcd1a354e225064cb299d25009cd0c802440fe50f618834b52b658b06984.jpg)

![1510a51fe7fe356e63526948a28f41975f398426f6794a175a39f81b9cca516c.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/tables/1510a51fe7fe356e63526948a28f41975f398426f6794a175a39f81b9cca516c.jpg)

![214cde764f79c0ee9e8641c551c1a14b8f9ed4e13a82486496054387a89cc2c8.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/tables/214cde764f79c0ee9e8641c551c1a14b8f9ed4e13a82486496054387a89cc2c8.jpg)

![2a75c88c8d97363d5965054310f96b9c450f34b1b21b150e381aade95d3e1734.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/tables/2a75c88c8d97363d5965054310f96b9c450f34b1b21b150e381aade95d3e1734.jpg)

![31777939cd40080de36c5915e51e7d8e1f25db95ecf097e416a46c910f4ef53e.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/tables/31777939cd40080de36c5915e51e7d8e1f25db95ecf097e416a46c910f4ef53e.jpg)

![378364d06c48fdf848b785d40d9b3e89aeae7a6b03f4f57ef49d299066ba37b7.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/tables/378364d06c48fdf848b785d40d9b3e89aeae7a6b03f4f57ef49d299066ba37b7.jpg)

![3a21e67d160d6369a52269db6ef44f661c1f02bba68cc8bb9d5829b62c2e31cc.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/tables/3a21e67d160d6369a52269db6ef44f661c1f02bba68cc8bb9d5829b62c2e31cc.jpg)

![3f5b15f0442b7109087f9a122ba8b7497dfddd769fd5d17e687b1a7dad077df6.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/tables/3f5b15f0442b7109087f9a122ba8b7497dfddd769fd5d17e687b1a7dad077df6.jpg)

![5f94c47ac5a8bb1a53a9879ac8d1ea9b636797572c0b7cdb6df5aadc83a5577f.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/tables/5f94c47ac5a8bb1a53a9879ac8d1ea9b636797572c0b7cdb6df5aadc83a5577f.jpg)

![7b6ae4ba258cb598e6ee042984738590661d0a6eb585840461eb6480750102a4.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/tables/7b6ae4ba258cb598e6ee042984738590661d0a6eb585840461eb6480750102a4.jpg)

![96e34739454cfb1182284a696442f2111ac724c0097a48ed7bb6b8375be52ca4.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/tables/96e34739454cfb1182284a696442f2111ac724c0097a48ed7bb6b8375be52ca4.jpg)

![a33bbf0a6392c38e855cc3be7ad7ae1943b6214611bd316a78bf21f215ee2d3a.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/tables/a33bbf0a6392c38e855cc3be7ad7ae1943b6214611bd316a78bf21f215ee2d3a.jpg)

![faa0aa4ac5f450d4ebf01ffbf4a6c754a9b57099b1efefb67860fdd3cf5b7554.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/tables/faa0aa4ac5f450d4ebf01ffbf4a6c754a9b57099b1efefb67860fdd3cf5b7554.jpg)

![fccb7b009a2131391c23a517af961dc5692c6800c7d66e47338e32db5e152867.jpg](../iclr_results/1284_Ready-to-React_ Online Reaction Policy for Two-Character Interaction Generation/tables/fccb7b009a2131391c23a517af961dc5692c6800c7d66e47338e32db5e152867.jpg)

## SimXRD-4M: Big Simulated X-ray Diffraction Data and Crystal Symmetry Classification Benchmark


### Images

![35905a239b107b07770c50e3ed0cc898e3ab82e3a8aaa58eba5369d7807ab45c.jpg](../iclr_results/1285_SimXRD-4M_ Big Simulated X-ray Diffraction Data and Crystal Symmetry Classification Benchmark/images/35905a239b107b07770c50e3ed0cc898e3ab82e3a8aaa58eba5369d7807ab45c.jpg)

![4169fbcad6901c94ed54f40d6adb0188f95635070b29dc44413d8032d7c63937.jpg](../iclr_results/1285_SimXRD-4M_ Big Simulated X-ray Diffraction Data and Crystal Symmetry Classification Benchmark/images/4169fbcad6901c94ed54f40d6adb0188f95635070b29dc44413d8032d7c63937.jpg)

![68c767703dd6617413c31fa70167437cc09102d490aaf3ec92137077015f5428.jpg](../iclr_results/1285_SimXRD-4M_ Big Simulated X-ray Diffraction Data and Crystal Symmetry Classification Benchmark/images/68c767703dd6617413c31fa70167437cc09102d490aaf3ec92137077015f5428.jpg)

![6a11fc20aa1440a7728ea5c0666914fcfc16e9a875a088f8a648eee6192a599c.jpg](../iclr_results/1285_SimXRD-4M_ Big Simulated X-ray Diffraction Data and Crystal Symmetry Classification Benchmark/images/6a11fc20aa1440a7728ea5c0666914fcfc16e9a875a088f8a648eee6192a599c.jpg)

![7b08bfb220a39d14dc2a94a9dc31cbd7291f66785d747cea9e712df1056a5df9.jpg](../iclr_results/1285_SimXRD-4M_ Big Simulated X-ray Diffraction Data and Crystal Symmetry Classification Benchmark/images/7b08bfb220a39d14dc2a94a9dc31cbd7291f66785d747cea9e712df1056a5df9.jpg)

![832df598c69275399be060d9412263a8258e00a8e6503fba66f83131cdb4d369.jpg](../iclr_results/1285_SimXRD-4M_ Big Simulated X-ray Diffraction Data and Crystal Symmetry Classification Benchmark/images/832df598c69275399be060d9412263a8258e00a8e6503fba66f83131cdb4d369.jpg)

![906634b10c9f519b77f605975c245a48cac7c6f918f5574fbd4fa58147644f22.jpg](../iclr_results/1285_SimXRD-4M_ Big Simulated X-ray Diffraction Data and Crystal Symmetry Classification Benchmark/images/906634b10c9f519b77f605975c245a48cac7c6f918f5574fbd4fa58147644f22.jpg)

![d659084b235d818d54c33af8b704cbadfcce4fe486cf1c623170b71a226a749e.jpg](../iclr_results/1285_SimXRD-4M_ Big Simulated X-ray Diffraction Data and Crystal Symmetry Classification Benchmark/images/d659084b235d818d54c33af8b704cbadfcce4fe486cf1c623170b71a226a749e.jpg)

![dcb12a76eb9a005eed7cbd04aca27d983f14020be502d707fb1ab42eaec9f7ca.jpg](../iclr_results/1285_SimXRD-4M_ Big Simulated X-ray Diffraction Data and Crystal Symmetry Classification Benchmark/images/dcb12a76eb9a005eed7cbd04aca27d983f14020be502d707fb1ab42eaec9f7ca.jpg)

![f67f2a56052c678ddba2962d5539324a240da9f1362fd4b5ee731d19bbd9ec77.jpg](../iclr_results/1285_SimXRD-4M_ Big Simulated X-ray Diffraction Data and Crystal Symmetry Classification Benchmark/images/f67f2a56052c678ddba2962d5539324a240da9f1362fd4b5ee731d19bbd9ec77.jpg)

### Tables

![0b7cd4c8c6dccb6b7a57df3518912c22900ada768251098b1401549d3c359d39.jpg](../iclr_results/1285_SimXRD-4M_ Big Simulated X-ray Diffraction Data and Crystal Symmetry Classification Benchmark/tables/0b7cd4c8c6dccb6b7a57df3518912c22900ada768251098b1401549d3c359d39.jpg)

![2eab7410570ea3f23e46a09e47e908b8cbdd721829b7382fcaa3fab52c14c05d.jpg](../iclr_results/1285_SimXRD-4M_ Big Simulated X-ray Diffraction Data and Crystal Symmetry Classification Benchmark/tables/2eab7410570ea3f23e46a09e47e908b8cbdd721829b7382fcaa3fab52c14c05d.jpg)

![73b5a1bfae93c872f618789a5969c75e6692ed70c83c254c3aebc15b687cc94f.jpg](../iclr_results/1285_SimXRD-4M_ Big Simulated X-ray Diffraction Data and Crystal Symmetry Classification Benchmark/tables/73b5a1bfae93c872f618789a5969c75e6692ed70c83c254c3aebc15b687cc94f.jpg)

![9c62115d9979e2e8a13ec61d209ad6d509dcf8b8f006263e55d458fb3b0a6567.jpg](../iclr_results/1285_SimXRD-4M_ Big Simulated X-ray Diffraction Data and Crystal Symmetry Classification Benchmark/tables/9c62115d9979e2e8a13ec61d209ad6d509dcf8b8f006263e55d458fb3b0a6567.jpg)

![e5327260110956d31f83ef66a108350a115332937fc11ccecf94ffd84e80092d.jpg](../iclr_results/1285_SimXRD-4M_ Big Simulated X-ray Diffraction Data and Crystal Symmetry Classification Benchmark/tables/e5327260110956d31f83ef66a108350a115332937fc11ccecf94ffd84e80092d.jpg)

![f1b1c2670b2d0db8aa29b1f5088d72ebd6d24f9b7bcfeb34dbf46924b37bdea9.jpg](../iclr_results/1285_SimXRD-4M_ Big Simulated X-ray Diffraction Data and Crystal Symmetry Classification Benchmark/tables/f1b1c2670b2d0db8aa29b1f5088d72ebd6d24f9b7bcfeb34dbf46924b37bdea9.jpg)

![fd3c4c858d6072983db12b9410d8fd0ae1b27513f5f88dc0bc5b94db34e6ebea.jpg](../iclr_results/1285_SimXRD-4M_ Big Simulated X-ray Diffraction Data and Crystal Symmetry Classification Benchmark/tables/fd3c4c858d6072983db12b9410d8fd0ae1b27513f5f88dc0bc5b94db34e6ebea.jpg)

## Associative memory and dead neurons


### Images

![b0696484f329b414d243a5a8c8557d594d3bfd91d3f083fb64918a310ef937fb.jpg](../iclr_results/1286_Associative memory and dead neurons/images/b0696484f329b414d243a5a8c8557d594d3bfd91d3f083fb64918a310ef937fb.jpg)

![cc0f4f0156259482288169ea66933425ef2a4b7ff3b568984684bd8c1729fc45.jpg](../iclr_results/1286_Associative memory and dead neurons/images/cc0f4f0156259482288169ea66933425ef2a4b7ff3b568984684bd8c1729fc45.jpg)

## Preble: Efficient Distributed Prompt Scheduling for LLM Serving


### Images

![2fe7acf416518e32a84bdbcf859a87b2c538c1598485cde48ffa6d2c0a960532.jpg](../iclr_results/1287_Preble_ Efficient Distributed Prompt Scheduling for LLM Serving/images/2fe7acf416518e32a84bdbcf859a87b2c538c1598485cde48ffa6d2c0a960532.jpg)

![56069bb8c7a75e2d92eba141f40d632401353fdfa2f242dda59485267cc381de.jpg](../iclr_results/1287_Preble_ Efficient Distributed Prompt Scheduling for LLM Serving/images/56069bb8c7a75e2d92eba141f40d632401353fdfa2f242dda59485267cc381de.jpg)

![607dd2687b207c38d0327b8c80ac264a86da169c0a42ff4e779a19c20d1f5261.jpg](../iclr_results/1287_Preble_ Efficient Distributed Prompt Scheduling for LLM Serving/images/607dd2687b207c38d0327b8c80ac264a86da169c0a42ff4e779a19c20d1f5261.jpg)

![63191243e397c0e76dbf5b72e1f586fcdf3de243e5ca023001b529931d3fb81d.jpg](../iclr_results/1287_Preble_ Efficient Distributed Prompt Scheduling for LLM Serving/images/63191243e397c0e76dbf5b72e1f586fcdf3de243e5ca023001b529931d3fb81d.jpg)

![6375f300daaf6a80e9ff4bcd296e53ef2eccefdbbcf534edec632466138d127c.jpg](../iclr_results/1287_Preble_ Efficient Distributed Prompt Scheduling for LLM Serving/images/6375f300daaf6a80e9ff4bcd296e53ef2eccefdbbcf534edec632466138d127c.jpg)

![6cf082354f1706392ab6d28ee2d050c991a57b287d280550a0d4667b70d0d763.jpg](../iclr_results/1287_Preble_ Efficient Distributed Prompt Scheduling for LLM Serving/images/6cf082354f1706392ab6d28ee2d050c991a57b287d280550a0d4667b70d0d763.jpg)

![70b33772566b747ada78a2c2c479e80eec21d5377fd48090155aa2192d7344ed.jpg](../iclr_results/1287_Preble_ Efficient Distributed Prompt Scheduling for LLM Serving/images/70b33772566b747ada78a2c2c479e80eec21d5377fd48090155aa2192d7344ed.jpg)

![748c4b8b02c54e0a85a83b49fbc185df8d97536f6dcc5d593f774e7ae6df7e48.jpg](../iclr_results/1287_Preble_ Efficient Distributed Prompt Scheduling for LLM Serving/images/748c4b8b02c54e0a85a83b49fbc185df8d97536f6dcc5d593f774e7ae6df7e48.jpg)

![75c8c7bd77d8e1ee3dae6d960ca5a182fecf401cf6aaee89fce0d1c51cc9c694.jpg](../iclr_results/1287_Preble_ Efficient Distributed Prompt Scheduling for LLM Serving/images/75c8c7bd77d8e1ee3dae6d960ca5a182fecf401cf6aaee89fce0d1c51cc9c694.jpg)

![b0bf0b592b6d590f7abbe872b117398062580cee75d20e1559e06e37de086f28.jpg](../iclr_results/1287_Preble_ Efficient Distributed Prompt Scheduling for LLM Serving/images/b0bf0b592b6d590f7abbe872b117398062580cee75d20e1559e06e37de086f28.jpg)

![d2007bbe40565b9b9cde01fdc9d8e92c1f9043307e8a59ac9fae24ab8ae9078f.jpg](../iclr_results/1287_Preble_ Efficient Distributed Prompt Scheduling for LLM Serving/images/d2007bbe40565b9b9cde01fdc9d8e92c1f9043307e8a59ac9fae24ab8ae9078f.jpg)

![d4d664d933a86c139d7bcdd6430d6ae8ed4ce95f9e2cdce78f066afdfd4879ac.jpg](../iclr_results/1287_Preble_ Efficient Distributed Prompt Scheduling for LLM Serving/images/d4d664d933a86c139d7bcdd6430d6ae8ed4ce95f9e2cdce78f066afdfd4879ac.jpg)

![d68694dc5a6c8ebcfa9737bcd2eeb645195fee2038ca9be487b17c2d4d42daf0.jpg](../iclr_results/1287_Preble_ Efficient Distributed Prompt Scheduling for LLM Serving/images/d68694dc5a6c8ebcfa9737bcd2eeb645195fee2038ca9be487b17c2d4d42daf0.jpg)

![d808fc784e0d3ad77357fd828b431d1cbd81ec92d24d0c9be9a70c4215619fde.jpg](../iclr_results/1287_Preble_ Efficient Distributed Prompt Scheduling for LLM Serving/images/d808fc784e0d3ad77357fd828b431d1cbd81ec92d24d0c9be9a70c4215619fde.jpg)

![db53d3c13b39dc8138f4e1200506facc8640a231b61f1122d51b5dcf9d287bf8.jpg](../iclr_results/1287_Preble_ Efficient Distributed Prompt Scheduling for LLM Serving/images/db53d3c13b39dc8138f4e1200506facc8640a231b61f1122d51b5dcf9d287bf8.jpg)

![e006e17fd90f9abedb564c1a1ef06867db4f24ecd2354b6438dd9dbdb3c56c58.jpg](../iclr_results/1287_Preble_ Efficient Distributed Prompt Scheduling for LLM Serving/images/e006e17fd90f9abedb564c1a1ef06867db4f24ecd2354b6438dd9dbdb3c56c58.jpg)

![e4d4754f4b2f324ca001b783810eecbb89c5e10cc4835fec6d38e589e8d5f30f.jpg](../iclr_results/1287_Preble_ Efficient Distributed Prompt Scheduling for LLM Serving/images/e4d4754f4b2f324ca001b783810eecbb89c5e10cc4835fec6d38e589e8d5f30f.jpg)

### Tables

![2365cf22bd2ffd6785ed4736877d907e0bc09d83cd691e8913107c5e57677ec8.jpg](../iclr_results/1287_Preble_ Efficient Distributed Prompt Scheduling for LLM Serving/tables/2365cf22bd2ffd6785ed4736877d907e0bc09d83cd691e8913107c5e57677ec8.jpg)

![665a676cf8a337ecaacca8ecc32e8052ba2728cbb7f50289946ce1d241428088.jpg](../iclr_results/1287_Preble_ Efficient Distributed Prompt Scheduling for LLM Serving/tables/665a676cf8a337ecaacca8ecc32e8052ba2728cbb7f50289946ce1d241428088.jpg)

![be24697586c77445b86214e5a964b8522531b96ef1ba89818ad6de78ee304159.jpg](../iclr_results/1287_Preble_ Efficient Distributed Prompt Scheduling for LLM Serving/tables/be24697586c77445b86214e5a964b8522531b96ef1ba89818ad6de78ee304159.jpg)

## From Tokens to Lattices: Emergent Lattice Structures in Language Models


### Images

![165df1642b6fd5e0bc6d6a773ac0fc06aeba9747747021c32bd731448d4e1e85.jpg](../iclr_results/1288_From Tokens to Lattices_ Emergent Lattice Structures in Language Models/images/165df1642b6fd5e0bc6d6a773ac0fc06aeba9747747021c32bd731448d4e1e85.jpg)

![2a3a75f3258bfdc8d2f492b210d86e96b1eaa9b950a01a6f3999545ff8fb9441.jpg](../iclr_results/1288_From Tokens to Lattices_ Emergent Lattice Structures in Language Models/images/2a3a75f3258bfdc8d2f492b210d86e96b1eaa9b950a01a6f3999545ff8fb9441.jpg)

![86841dc788d6d71168b0bfdd5ebfc746d74c9825974bcbe02c4d7eaaf4cb25f8.jpg](../iclr_results/1288_From Tokens to Lattices_ Emergent Lattice Structures in Language Models/images/86841dc788d6d71168b0bfdd5ebfc746d74c9825974bcbe02c4d7eaaf4cb25f8.jpg)

![8ecedb075c379f67824220958bdc49d555046148b482457b254289f8ca2164c0.jpg](../iclr_results/1288_From Tokens to Lattices_ Emergent Lattice Structures in Language Models/images/8ecedb075c379f67824220958bdc49d555046148b482457b254289f8ca2164c0.jpg)

![d1c94a6c9e2c0ee25d4c593bb824254986d95dc68bb1939e0c43b5a63dd23056.jpg](../iclr_results/1288_From Tokens to Lattices_ Emergent Lattice Structures in Language Models/images/d1c94a6c9e2c0ee25d4c593bb824254986d95dc68bb1939e0c43b5a63dd23056.jpg)

### Tables

![503114138e28bdd6f3f9687ca4718bf198dc2c338173174e7f464600d2caee92.jpg](../iclr_results/1288_From Tokens to Lattices_ Emergent Lattice Structures in Language Models/tables/503114138e28bdd6f3f9687ca4718bf198dc2c338173174e7f464600d2caee92.jpg)

![9951e4bfa7374002313a8452d0e858090fa63196cc90b2daff7352dbd3e5dd3d.jpg](../iclr_results/1288_From Tokens to Lattices_ Emergent Lattice Structures in Language Models/tables/9951e4bfa7374002313a8452d0e858090fa63196cc90b2daff7352dbd3e5dd3d.jpg)

![b8bc16b3aa7550a825dff3765143b2bf8abdb1033245eef4cd16f8b386b09ea9.jpg](../iclr_results/1288_From Tokens to Lattices_ Emergent Lattice Structures in Language Models/tables/b8bc16b3aa7550a825dff3765143b2bf8abdb1033245eef4cd16f8b386b09ea9.jpg)

![dc31db160a5204ecbe8569c7538b227abf8a634e5057b1bf7254bbaeb9c93f80.jpg](../iclr_results/1288_From Tokens to Lattices_ Emergent Lattice Structures in Language Models/tables/dc31db160a5204ecbe8569c7538b227abf8a634e5057b1bf7254bbaeb9c93f80.jpg)

![fb78daad99a7b5a434cd083cf504ccbfccb78ba09fb1f886e1fe3792c89d5654.jpg](../iclr_results/1288_From Tokens to Lattices_ Emergent Lattice Structures in Language Models/tables/fb78daad99a7b5a434cd083cf504ccbfccb78ba09fb1f886e1fe3792c89d5654.jpg)

## HQ-Edit: A High-Quality Dataset for Instruction-based Image Editing


### Images

![00a2164d7b8a81508f08c3d5b775057a146c8c5aa3fc287b995487e6a7d4f35f.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/00a2164d7b8a81508f08c3d5b775057a146c8c5aa3fc287b995487e6a7d4f35f.jpg)

![0b12141a76f27c0b87f51c4f433733376283de4e4b777d9b6813e9ca0fc6e067.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/0b12141a76f27c0b87f51c4f433733376283de4e4b777d9b6813e9ca0fc6e067.jpg)

![18d82d53f79bf7f03454563333c83dcb508733fd0dfdc15dc01e45df043dcb42.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/18d82d53f79bf7f03454563333c83dcb508733fd0dfdc15dc01e45df043dcb42.jpg)

![2010bd7799327155db56d2b3618f55b7f5442ee2b5b1eac524ae0f3a4efdaffb.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/2010bd7799327155db56d2b3618f55b7f5442ee2b5b1eac524ae0f3a4efdaffb.jpg)

![351d6501b006e94c760fae52f1ff7f738c56e2818eb16e9138e0a3528457e27d.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/351d6501b006e94c760fae52f1ff7f738c56e2818eb16e9138e0a3528457e27d.jpg)

![42cfc5b4018ebf768edad72bafa1656bba5ae395aeb02cc27a9e384a276ac778.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/42cfc5b4018ebf768edad72bafa1656bba5ae395aeb02cc27a9e384a276ac778.jpg)

![5219db8acd15e1114835c4f0babcac1f00b5b24c8245b491888f0deb75870971.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/5219db8acd15e1114835c4f0babcac1f00b5b24c8245b491888f0deb75870971.jpg)

![58db0545e1de33864c4c9a14db4360ebace8f63b40f2af5c4ff48b1c1a6bbbef.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/58db0545e1de33864c4c9a14db4360ebace8f63b40f2af5c4ff48b1c1a6bbbef.jpg)

![59969fd7791b8bae12f469fb87ff3469efc815162b141d07f6ae6fb3dfa7a307.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/59969fd7791b8bae12f469fb87ff3469efc815162b141d07f6ae6fb3dfa7a307.jpg)

![64e327c28fd546d589749033ff353002f5eb221f1c294e91fa4e715dd45fbc33.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/64e327c28fd546d589749033ff353002f5eb221f1c294e91fa4e715dd45fbc33.jpg)

![666e62244e1ec4c77a8f2c3783d5924e85dc0dc3903b8a7bdf6307494fcf51db.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/666e62244e1ec4c77a8f2c3783d5924e85dc0dc3903b8a7bdf6307494fcf51db.jpg)

![6cfc3b5ae616a3de7ce061e0ea4d82bf3ebc31701b379573950b35c2f50a2378.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/6cfc3b5ae616a3de7ce061e0ea4d82bf3ebc31701b379573950b35c2f50a2378.jpg)

![6d3dba7eb778446ae879f81c2d2e564cfffdad63adae8cc7ce63867c84ec0ab5.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/6d3dba7eb778446ae879f81c2d2e564cfffdad63adae8cc7ce63867c84ec0ab5.jpg)

![7ba368d4db33785f00e173f8c2b04baca2a2b7cdad0d10144126ff024e4bf693.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/7ba368d4db33785f00e173f8c2b04baca2a2b7cdad0d10144126ff024e4bf693.jpg)

![803086c6cee7a5797921bb60da892d1fcd6bd98395ff76cf4e160f4976223a0b.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/803086c6cee7a5797921bb60da892d1fcd6bd98395ff76cf4e160f4976223a0b.jpg)

![828686ce681fa461828efd6156cb0f90c658aac6f77271f55d7eb562b3b4a023.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/828686ce681fa461828efd6156cb0f90c658aac6f77271f55d7eb562b3b4a023.jpg)

![8bcfcf7c156db80d2c8620876a288cf2860a064869b6589a4218d78993d14e99.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/8bcfcf7c156db80d2c8620876a288cf2860a064869b6589a4218d78993d14e99.jpg)

![ac5633f46cb7a602d7fe910a9e86b529aba334b5702e25e8635ca419f2010b20.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/ac5633f46cb7a602d7fe910a9e86b529aba334b5702e25e8635ca419f2010b20.jpg)

![b0790d29aad21fe7c75bded859887a9d530a1b67a30c95660223836a16c9ac86.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/b0790d29aad21fe7c75bded859887a9d530a1b67a30c95660223836a16c9ac86.jpg)

![b1e7442831632eada64a9442e973e7109eeaf774c5e318461122eadfde8c113e.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/b1e7442831632eada64a9442e973e7109eeaf774c5e318461122eadfde8c113e.jpg)

![ca1f37bc26489605746a83aa3792c5d3d73d778dc1490ac84e856c0a93936f76.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/ca1f37bc26489605746a83aa3792c5d3d73d778dc1490ac84e856c0a93936f76.jpg)

![cebef3f7be86602284e8bca1a43670b7737fcae800c1efde80424747477df1ce.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/cebef3f7be86602284e8bca1a43670b7737fcae800c1efde80424747477df1ce.jpg)

![dbe54ae7190d5e4d512725aa79f819998a3d82d06fe158ce61189d2aa6c1f29f.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/dbe54ae7190d5e4d512725aa79f819998a3d82d06fe158ce61189d2aa6c1f29f.jpg)

![dc9907dcc56425b7442eea191667e925779eb5050e23aeda2faab4ba344df42b.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/dc9907dcc56425b7442eea191667e925779eb5050e23aeda2faab4ba344df42b.jpg)

![e065c290979aa1b3c03c8342f4649c08c89694001be55c9bbbdde0779758882c.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/e065c290979aa1b3c03c8342f4649c08c89694001be55c9bbbdde0779758882c.jpg)

![f7853fe4a5741099d4419c4dc4e941fb0fa07854a48450b0516fe02472c376f7.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/f7853fe4a5741099d4419c4dc4e941fb0fa07854a48450b0516fe02472c376f7.jpg)

![f87ecc913c4275279f106f869940eac754d500b5ff5f62ce84f2cfb1d42d1f7c.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/images/f87ecc913c4275279f106f869940eac754d500b5ff5f62ce84f2cfb1d42d1f7c.jpg)

### Tables

![0777ccbffb5858711517072bbb8d145a4b393bdf2528d727a5f947076c972144.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/tables/0777ccbffb5858711517072bbb8d145a4b393bdf2528d727a5f947076c972144.jpg)

![15f75570ec65a69f4474292ef461e2f29d4fa6aebe143754322c1e79f2ed4c8f.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/tables/15f75570ec65a69f4474292ef461e2f29d4fa6aebe143754322c1e79f2ed4c8f.jpg)

![1ade9fcf202d3d45b68917e1f8a4b41aaef540a64265808bff17c9f662b880db.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/tables/1ade9fcf202d3d45b68917e1f8a4b41aaef540a64265808bff17c9f662b880db.jpg)

![2490152c67f91b668a35e3af44f7de9962369f90fe5d58fdd87a9e15e2494f28.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/tables/2490152c67f91b668a35e3af44f7de9962369f90fe5d58fdd87a9e15e2494f28.jpg)

![3023374c390123cd92ddc275e88f12d1b5945f6f1990d7fdf90026cf0e452843.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/tables/3023374c390123cd92ddc275e88f12d1b5945f6f1990d7fdf90026cf0e452843.jpg)

![51676020b7402b0091fabd919d103e11ec4b0b23e6223e4a05599ebdffe04367.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/tables/51676020b7402b0091fabd919d103e11ec4b0b23e6223e4a05599ebdffe04367.jpg)

![a544f0f0b4a170e788f78effffd1887479c0a5ce2ef4defb3a646ab29cd9f0fa.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/tables/a544f0f0b4a170e788f78effffd1887479c0a5ce2ef4defb3a646ab29cd9f0fa.jpg)

![ea1b8754e02e2331b54cb764b1f297702be734caafe9070d8c807c917397a98d.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/tables/ea1b8754e02e2331b54cb764b1f297702be734caafe9070d8c807c917397a98d.jpg)

![eff211acea46e33f34f67fc76d173826a3a3f4f70b245494d7ec9d1491df791c.jpg](../iclr_results/1290_HQ-Edit_ A High-Quality Dataset for Instruction-based Image Editing/tables/eff211acea46e33f34f67fc76d173826a3a3f4f70b245494d7ec9d1491df791c.jpg)

## Topological Zigzag Spaghetti for Diffusion-based Generation and Prediction on Graphs


### Images

![99a166fd8bad6984c1131d92cb979575707bbee4e8e39cbb6890be3d4e2a7b14.jpg](../iclr_results/1291_Topological Zigzag Spaghetti for Diffusion-based Generation and Prediction on Graphs/images/99a166fd8bad6984c1131d92cb979575707bbee4e8e39cbb6890be3d4e2a7b14.jpg)

![c1f9046469de2622e91400d92b917dee8619b6bdb7987616fa01c628098a940b.jpg](../iclr_results/1291_Topological Zigzag Spaghetti for Diffusion-based Generation and Prediction on Graphs/images/c1f9046469de2622e91400d92b917dee8619b6bdb7987616fa01c628098a940b.jpg)

![ef591b2ceac41cdd055112b9b3947af297da50cedabca5aa978a3560fc84e743.jpg](../iclr_results/1291_Topological Zigzag Spaghetti for Diffusion-based Generation and Prediction on Graphs/images/ef591b2ceac41cdd055112b9b3947af297da50cedabca5aa978a3560fc84e743.jpg)

### Tables

![00414d548ebf54788f22a5152f1bc086c20f5c3238d57def9e7ed8ad5bfcfa1a.jpg](../iclr_results/1291_Topological Zigzag Spaghetti for Diffusion-based Generation and Prediction on Graphs/tables/00414d548ebf54788f22a5152f1bc086c20f5c3238d57def9e7ed8ad5bfcfa1a.jpg)

![02eb8e10e2c7843c28b6b4381106b0bb707e26bd6f05893898942609b84909c6.jpg](../iclr_results/1291_Topological Zigzag Spaghetti for Diffusion-based Generation and Prediction on Graphs/tables/02eb8e10e2c7843c28b6b4381106b0bb707e26bd6f05893898942609b84909c6.jpg)

![138796a9af42c2267b73c2404144fa65b705c8c1db06ad344b1db6b3aa27a8d6.jpg](../iclr_results/1291_Topological Zigzag Spaghetti for Diffusion-based Generation and Prediction on Graphs/tables/138796a9af42c2267b73c2404144fa65b705c8c1db06ad344b1db6b3aa27a8d6.jpg)

![1414ede99129c8920050f53140a04fc040e71dab162eb2d0baaf014c07a29734.jpg](../iclr_results/1291_Topological Zigzag Spaghetti for Diffusion-based Generation and Prediction on Graphs/tables/1414ede99129c8920050f53140a04fc040e71dab162eb2d0baaf014c07a29734.jpg)

![33d3c12e5462f3558fb41ce67c2048e750c5e11e0c054aca620b522714848864.jpg](../iclr_results/1291_Topological Zigzag Spaghetti for Diffusion-based Generation and Prediction on Graphs/tables/33d3c12e5462f3558fb41ce67c2048e750c5e11e0c054aca620b522714848864.jpg)

![41c65067261d0830a6de2ef04434a2536e7eb2d4e8fdf8a43509c05c0cbfcfc8.jpg](../iclr_results/1291_Topological Zigzag Spaghetti for Diffusion-based Generation and Prediction on Graphs/tables/41c65067261d0830a6de2ef04434a2536e7eb2d4e8fdf8a43509c05c0cbfcfc8.jpg)

![530ae9c3a95e0ffccba06a84ab8c299a0e3f48fe2df289e498facbbfccd2d0cb.jpg](../iclr_results/1291_Topological Zigzag Spaghetti for Diffusion-based Generation and Prediction on Graphs/tables/530ae9c3a95e0ffccba06a84ab8c299a0e3f48fe2df289e498facbbfccd2d0cb.jpg)

![67a26f74773f645798c0a681451984950b8793494d0b2f0c020188d4efe635a8.jpg](../iclr_results/1291_Topological Zigzag Spaghetti for Diffusion-based Generation and Prediction on Graphs/tables/67a26f74773f645798c0a681451984950b8793494d0b2f0c020188d4efe635a8.jpg)

![6fb427081aab72295c4275940ec48a00d107779dabedeefc8495593dbc53b880.jpg](../iclr_results/1291_Topological Zigzag Spaghetti for Diffusion-based Generation and Prediction on Graphs/tables/6fb427081aab72295c4275940ec48a00d107779dabedeefc8495593dbc53b880.jpg)

![86233eeb4a70b50287e9ae82a1781c0f7322112c8344a644d3ec818633a3a6c3.jpg](../iclr_results/1291_Topological Zigzag Spaghetti for Diffusion-based Generation and Prediction on Graphs/tables/86233eeb4a70b50287e9ae82a1781c0f7322112c8344a644d3ec818633a3a6c3.jpg)

![a7b19a6548de201a72e7d498cf3c7689e00334e6b67efb09199d036d22e390e1.jpg](../iclr_results/1291_Topological Zigzag Spaghetti for Diffusion-based Generation and Prediction on Graphs/tables/a7b19a6548de201a72e7d498cf3c7689e00334e6b67efb09199d036d22e390e1.jpg)

![add7663f5e3e398c4960027ca1874b4777576323e02386efefb483b31f313498.jpg](../iclr_results/1291_Topological Zigzag Spaghetti for Diffusion-based Generation and Prediction on Graphs/tables/add7663f5e3e398c4960027ca1874b4777576323e02386efefb483b31f313498.jpg)

![cf9a81f799290bf05804e4a2a1a130b9e6aec88af93470eafcd9cc8a40058631.jpg](../iclr_results/1291_Topological Zigzag Spaghetti for Diffusion-based Generation and Prediction on Graphs/tables/cf9a81f799290bf05804e4a2a1a130b9e6aec88af93470eafcd9cc8a40058631.jpg)

![f558d68dbda5095f6922fda749c3659c46c5effeb28c76297f04d0ebac8e3070.jpg](../iclr_results/1291_Topological Zigzag Spaghetti for Diffusion-based Generation and Prediction on Graphs/tables/f558d68dbda5095f6922fda749c3659c46c5effeb28c76297f04d0ebac8e3070.jpg)

## E(3)-equivariant models cannot learn chirality: Field-based molecular generation


### Images

![08b19909c0586a66a53bd55315d71ffaaa68cd7e619bbff4de26f930084f4fde.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/images/08b19909c0586a66a53bd55315d71ffaaa68cd7e619bbff4de26f930084f4fde.jpg)

![14983228b14ad495af786aa201db6be1b2e6b3f5bd48a9f14b1146639d6678e0.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/images/14983228b14ad495af786aa201db6be1b2e6b3f5bd48a9f14b1146639d6678e0.jpg)

![2e2ca730456c675e28383e50faf579e490944cbfb47f4869b7829ff2b7e0b3fe.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/images/2e2ca730456c675e28383e50faf579e490944cbfb47f4869b7829ff2b7e0b3fe.jpg)

![33c59a519caf3c9de5030d3fe2f1452abbbb3251b653857f7c403c69983cece0.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/images/33c59a519caf3c9de5030d3fe2f1452abbbb3251b653857f7c403c69983cece0.jpg)

![35b4a6664c9e8ac705fb7fbaee1224026c6abe66f7589405b311f6ac0c55f3c1.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/images/35b4a6664c9e8ac705fb7fbaee1224026c6abe66f7589405b311f6ac0c55f3c1.jpg)

![444e7b99e5f7de9369afc1fb14178a8e3ff34e740c691db36763d2f127d71d6d.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/images/444e7b99e5f7de9369afc1fb14178a8e3ff34e740c691db36763d2f127d71d6d.jpg)

![46550b252abb0e6add806e30fc39c4c6608ab6de0aac8e03f61e04a5a885330b.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/images/46550b252abb0e6add806e30fc39c4c6608ab6de0aac8e03f61e04a5a885330b.jpg)

![4d57a6fa6bd3831aa16e7fa4e846bf18bcacd657c7e3d44a06e952e611eac6c9.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/images/4d57a6fa6bd3831aa16e7fa4e846bf18bcacd657c7e3d44a06e952e611eac6c9.jpg)

![4f3a362d2eef61c797d2106593f671de3c5a28b0b538ee897d32a08be5feaac2.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/images/4f3a362d2eef61c797d2106593f671de3c5a28b0b538ee897d32a08be5feaac2.jpg)

![607138a70f514c1d8b5d3a6c522d97a007d299ab5e86c8c87b72782275fac8f2.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/images/607138a70f514c1d8b5d3a6c522d97a007d299ab5e86c8c87b72782275fac8f2.jpg)

![925ceba5bf48794050819aece3f0e03c7cb7a88d93d6bb9925dbe5f87eced9c7.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/images/925ceba5bf48794050819aece3f0e03c7cb7a88d93d6bb9925dbe5f87eced9c7.jpg)

![95c607d4356741f8f1c05a3b5a4b121d7fc938b6c9283b04b00ee7379eebd79a.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/images/95c607d4356741f8f1c05a3b5a4b121d7fc938b6c9283b04b00ee7379eebd79a.jpg)

![9d10f14b53cb350f3cb00b26a044fdaa7beedc11301f1f63e1f1b89721fa63ce.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/images/9d10f14b53cb350f3cb00b26a044fdaa7beedc11301f1f63e1f1b89721fa63ce.jpg)

![b03ec8b64f799f574009d27e05d0451d0c026b1f8b60f84978926f61d1fff481.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/images/b03ec8b64f799f574009d27e05d0451d0c026b1f8b60f84978926f61d1fff481.jpg)

![ba9eb23de713a7e589407a0f4ff671c79d3564c6bd66846b4ee2d27c834c4da5.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/images/ba9eb23de713a7e589407a0f4ff671c79d3564c6bd66846b4ee2d27c834c4da5.jpg)

![c2cd35148a5cc09571a543a7ceba22922df702fafe71a1fd8b5fd024de5075cb.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/images/c2cd35148a5cc09571a543a7ceba22922df702fafe71a1fd8b5fd024de5075cb.jpg)

![cb5b857eceb108e583be92d67a9d538077089b23ee8b2eebd1e7263f6ab438f4.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/images/cb5b857eceb108e583be92d67a9d538077089b23ee8b2eebd1e7263f6ab438f4.jpg)

![ce22d322e7e4a9ea6fe3af7ec8e9264fa7e1085ef4f9281120190809b1c4c1a9.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/images/ce22d322e7e4a9ea6fe3af7ec8e9264fa7e1085ef4f9281120190809b1c4c1a9.jpg)

![ced07ed445b60bbf735e431b56967ff537d88199709d9f6a9b12da7106ed5015.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/images/ced07ed445b60bbf735e431b56967ff537d88199709d9f6a9b12da7106ed5015.jpg)

![d0803498df436ebb8f6ad6b2c4fb69a5b242c2e97cc23aa449510105fa80390c.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/images/d0803498df436ebb8f6ad6b2c4fb69a5b242c2e97cc23aa449510105fa80390c.jpg)

![eacbb480937f38d4070cfa3f55cfbe9a7e04f87a7e97a7eae7504b223394971e.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/images/eacbb480937f38d4070cfa3f55cfbe9a7e04f87a7e97a7eae7504b223394971e.jpg)

![f7c78f9742f781d5b624749dc985d21ce3ba29213de0c59496c39bf09017bc4e.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/images/f7c78f9742f781d5b624749dc985d21ce3ba29213de0c59496c39bf09017bc4e.jpg)

### Tables

![0dde9d351f0de5f82992d4358871be421d853658b84693815e7848841fc451b7.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/tables/0dde9d351f0de5f82992d4358871be421d853658b84693815e7848841fc451b7.jpg)

![39dd92f55cea272e2141888c3e58670534777d134c74cad024b325f6dc1aeace.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/tables/39dd92f55cea272e2141888c3e58670534777d134c74cad024b325f6dc1aeace.jpg)

![749a4254da220ba22235f4496fe3b04daa51d0d98fd4b2821d04cc03d251792d.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/tables/749a4254da220ba22235f4496fe3b04daa51d0d98fd4b2821d04cc03d251792d.jpg)

![b9692324b94e066bd54b3ce0cc57d64094ab4cd6f9a630e50f805ddcb13accd3.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/tables/b9692324b94e066bd54b3ce0cc57d64094ab4cd6f9a630e50f805ddcb13accd3.jpg)

![d8782beda780dfc2ae8457bef06fb4fb86ddcd6e97266d2c307cf1963f76c16f.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/tables/d8782beda780dfc2ae8457bef06fb4fb86ddcd6e97266d2c307cf1963f76c16f.jpg)

![dd922df395ca04acb1c86d83b452c78cb8d47b8feac9ba5012dbba95abe2072f.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/tables/dd922df395ca04acb1c86d83b452c78cb8d47b8feac9ba5012dbba95abe2072f.jpg)

![f7399518570ba12202714f7217f9b87799a5b30547f9918859962f8506649712.jpg](../iclr_results/1292_E(3)-equivariant models cannot learn chirality_ Field-based molecular generation/tables/f7399518570ba12202714f7217f9b87799a5b30547f9918859962f8506649712.jpg)

## A General Framework for Off-Policy Learning with Partially-Observed Reward


### Images

![06ee0f6dd91a11528c5148b7ceaa87e5a7218e0caebe347bdfc5edbcff34f851.jpg](../iclr_results/1293_A General Framework for Off-Policy Learning with Partially-Observed Reward/images/06ee0f6dd91a11528c5148b7ceaa87e5a7218e0caebe347bdfc5edbcff34f851.jpg)

![267f107bfbce86f85f240373b9a3a5ad98643f2e22663ee0314a921cab05fa16.jpg](../iclr_results/1293_A General Framework for Off-Policy Learning with Partially-Observed Reward/images/267f107bfbce86f85f240373b9a3a5ad98643f2e22663ee0314a921cab05fa16.jpg)

![33f4501985af6b1666f0981253a5c3317fab73652d9618a7109724a1685d0bd4.jpg](../iclr_results/1293_A General Framework for Off-Policy Learning with Partially-Observed Reward/images/33f4501985af6b1666f0981253a5c3317fab73652d9618a7109724a1685d0bd4.jpg)

![3ca117b923a9a4d020646d510946f2506d0e3d9daae66e98eebd884e779787b7.jpg](../iclr_results/1293_A General Framework for Off-Policy Learning with Partially-Observed Reward/images/3ca117b923a9a4d020646d510946f2506d0e3d9daae66e98eebd884e779787b7.jpg)

![46d8b378c246a2458e8d27b428efa525b2a236504549255c9a0cf432e7401e95.jpg](../iclr_results/1293_A General Framework for Off-Policy Learning with Partially-Observed Reward/images/46d8b378c246a2458e8d27b428efa525b2a236504549255c9a0cf432e7401e95.jpg)

![54df3606e80d183bbfc369e5fd0975f38c0faa905ca1cb7bf2ed53fd93514d99.jpg](../iclr_results/1293_A General Framework for Off-Policy Learning with Partially-Observed Reward/images/54df3606e80d183bbfc369e5fd0975f38c0faa905ca1cb7bf2ed53fd93514d99.jpg)

![5d75a11a02885d13bd2d29aec41f773647cf4521218af05cb598f4ace55718a0.jpg](../iclr_results/1293_A General Framework for Off-Policy Learning with Partially-Observed Reward/images/5d75a11a02885d13bd2d29aec41f773647cf4521218af05cb598f4ace55718a0.jpg)

![b41b2c9beba728f3053543e36c36d04dea98aec56cac0159cde790d8522a45ba.jpg](../iclr_results/1293_A General Framework for Off-Policy Learning with Partially-Observed Reward/images/b41b2c9beba728f3053543e36c36d04dea98aec56cac0159cde790d8522a45ba.jpg)

![b7bdd1a311a6494c0615ab0fb624cf5d383245eb096816a4a91a15a4b74ab781.jpg](../iclr_results/1293_A General Framework for Off-Policy Learning with Partially-Observed Reward/images/b7bdd1a311a6494c0615ab0fb624cf5d383245eb096816a4a91a15a4b74ab781.jpg)

![c043b09ff3cbb0a8945c6500cc298881ac269556763eec8684ea66cfdc772850.jpg](../iclr_results/1293_A General Framework for Off-Policy Learning with Partially-Observed Reward/images/c043b09ff3cbb0a8945c6500cc298881ac269556763eec8684ea66cfdc772850.jpg)

![c3d76ee3f688485b0b472876d63256f171b02e3753c7c33fd33ac57e2b6482bb.jpg](../iclr_results/1293_A General Framework for Off-Policy Learning with Partially-Observed Reward/images/c3d76ee3f688485b0b472876d63256f171b02e3753c7c33fd33ac57e2b6482bb.jpg)

![cfd3e60aae1ccb510a7a9cdf8227a02024203f1355d155915df1a2d552451abe.jpg](../iclr_results/1293_A General Framework for Off-Policy Learning with Partially-Observed Reward/images/cfd3e60aae1ccb510a7a9cdf8227a02024203f1355d155915df1a2d552451abe.jpg)

![d2510421144fb49acda4dca838a39bd5d1e78531fc5d3d9a3fdbf78ca9cde380.jpg](../iclr_results/1293_A General Framework for Off-Policy Learning with Partially-Observed Reward/images/d2510421144fb49acda4dca838a39bd5d1e78531fc5d3d9a3fdbf78ca9cde380.jpg)

![dd0b1ab900167918eceae1d9b9f86320f8945d02fbcba7ec1e89b91e7053147b.jpg](../iclr_results/1293_A General Framework for Off-Policy Learning with Partially-Observed Reward/images/dd0b1ab900167918eceae1d9b9f86320f8945d02fbcba7ec1e89b91e7053147b.jpg)

![e75ed5dfd1cfa253165679f1f7e4687ef9f0cdc4828191f88a44f043c6e8c595.jpg](../iclr_results/1293_A General Framework for Off-Policy Learning with Partially-Observed Reward/images/e75ed5dfd1cfa253165679f1f7e4687ef9f0cdc4828191f88a44f043c6e8c595.jpg)

![f7d922f2204ced3de24ec30f69a9fabda6e4e24d5986e52a1d8abb7f8bd976c2.jpg](../iclr_results/1293_A General Framework for Off-Policy Learning with Partially-Observed Reward/images/f7d922f2204ced3de24ec30f69a9fabda6e4e24d5986e52a1d8abb7f8bd976c2.jpg)

### Tables

![0056115411b4468c968a0e4f225c563afc14eb29f1c65ed952bcec20cc563faf.jpg](../iclr_results/1293_A General Framework for Off-Policy Learning with Partially-Observed Reward/tables/0056115411b4468c968a0e4f225c563afc14eb29f1c65ed952bcec20cc563faf.jpg)

![48565b8f8b6237fea56acda3d9b9133ef6ca907e6b4e2b964fabefd11068b659.jpg](../iclr_results/1293_A General Framework for Off-Policy Learning with Partially-Observed Reward/tables/48565b8f8b6237fea56acda3d9b9133ef6ca907e6b4e2b964fabefd11068b659.jpg)

![5a23c243a3a562215c5be4d17888a5a547c525aa59c92adef056dd625b3baafb.jpg](../iclr_results/1293_A General Framework for Off-Policy Learning with Partially-Observed Reward/tables/5a23c243a3a562215c5be4d17888a5a547c525aa59c92adef056dd625b3baafb.jpg)

![8cd2c950db20aebe394f656a3e3127dbd013b1a544f5bdc54670181f4621c7e8.jpg](../iclr_results/1293_A General Framework for Off-Policy Learning with Partially-Observed Reward/tables/8cd2c950db20aebe394f656a3e3127dbd013b1a544f5bdc54670181f4621c7e8.jpg)

![be037346449ba322b889dcda69abd569870b2a7f7deaf739892463c550cb3e99.jpg](../iclr_results/1293_A General Framework for Off-Policy Learning with Partially-Observed Reward/tables/be037346449ba322b889dcda69abd569870b2a7f7deaf739892463c550cb3e99.jpg)

![c480a46fdc06bb14d88dff11bc788b93a3e67834e652e25fbb34ace884aa76a3.jpg](../iclr_results/1293_A General Framework for Off-Policy Learning with Partially-Observed Reward/tables/c480a46fdc06bb14d88dff11bc788b93a3e67834e652e25fbb34ace884aa76a3.jpg)

## Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection


### Images

![297f79a7de84402661afef7a9ade8c1fd4889d49394307becd8abd233e591146.jpg](../iclr_results/1294_Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection/images/297f79a7de84402661afef7a9ade8c1fd4889d49394307becd8abd233e591146.jpg)

![3e9b0a2300e793fabc77ab66c67ef757338a95b85c76a28acab1f05fb668f292.jpg](../iclr_results/1294_Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection/images/3e9b0a2300e793fabc77ab66c67ef757338a95b85c76a28acab1f05fb668f292.jpg)

![8edbb09be654504500e087dae9b13288aa765d69118869c2d9b77c1172e37981.jpg](../iclr_results/1294_Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection/images/8edbb09be654504500e087dae9b13288aa765d69118869c2d9b77c1172e37981.jpg)

![bffb2167b1db25d09e5077129ef5ad9a2f037ab47d8b7041cdc581761403d098.jpg](../iclr_results/1294_Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection/images/bffb2167b1db25d09e5077129ef5ad9a2f037ab47d8b7041cdc581761403d098.jpg)

![c7942e28caf2a15664a320d495059143b9a2a90189ba1e22f032124537c16822.jpg](../iclr_results/1294_Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection/images/c7942e28caf2a15664a320d495059143b9a2a90189ba1e22f032124537c16822.jpg)

![d2812d583ff9c8a5f2f0c2a3f83e902f0560f95432ca06635e80add7d59f846b.jpg](../iclr_results/1294_Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection/images/d2812d583ff9c8a5f2f0c2a3f83e902f0560f95432ca06635e80add7d59f846b.jpg)

![f8d9c102863f97152df059209b47c2e279dc05b18aca38376458b18eb69e7e1c.jpg](../iclr_results/1294_Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection/images/f8d9c102863f97152df059209b47c2e279dc05b18aca38376458b18eb69e7e1c.jpg)

### Tables

![08871ef7195ed9a41d25151029647eddfd7ea01f7a75550e8f1b5c2061774b14.jpg](../iclr_results/1294_Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection/tables/08871ef7195ed9a41d25151029647eddfd7ea01f7a75550e8f1b5c2061774b14.jpg)

![0f8bf18b7398b403109c0ffd1b8a429bc1424f4dd0be437d1388f4fc9caa1bcd.jpg](../iclr_results/1294_Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection/tables/0f8bf18b7398b403109c0ffd1b8a429bc1424f4dd0be437d1388f4fc9caa1bcd.jpg)

![22c597c7758f4cc2f35e2278bdd162b050ed038d063ab8ef6645a998f73e9949.jpg](../iclr_results/1294_Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection/tables/22c597c7758f4cc2f35e2278bdd162b050ed038d063ab8ef6645a998f73e9949.jpg)

![250a91da04bd3219c7eef0bb3bc0429e99215fdf09c6ee61b160eeabda107b51.jpg](../iclr_results/1294_Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection/tables/250a91da04bd3219c7eef0bb3bc0429e99215fdf09c6ee61b160eeabda107b51.jpg)

![3d8cdad1a9335d4a3fb5481cb6728fac3c3348eda1a727c841531dcad53c220f.jpg](../iclr_results/1294_Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection/tables/3d8cdad1a9335d4a3fb5481cb6728fac3c3348eda1a727c841531dcad53c220f.jpg)

![5b325b5f944491251a109471aca1dee5bd2621a1f0614699fe2b7dc3fadb18cb.jpg](../iclr_results/1294_Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection/tables/5b325b5f944491251a109471aca1dee5bd2621a1f0614699fe2b7dc3fadb18cb.jpg)

![5cdaa6c3b7635304307b943d7ea55e2e02f002a47ae8e5ee2631f4bcad00f58a.jpg](../iclr_results/1294_Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection/tables/5cdaa6c3b7635304307b943d7ea55e2e02f002a47ae8e5ee2631f4bcad00f58a.jpg)

![5de91e5a47a7109ef22b4cee6c05673728458ae1c09e71ca0bc29a79b58d32d0.jpg](../iclr_results/1294_Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection/tables/5de91e5a47a7109ef22b4cee6c05673728458ae1c09e71ca0bc29a79b58d32d0.jpg)

![5f7c0f83f5355b77bb18ece830119bc2239a286c38afc87303c7d555722055e4.jpg](../iclr_results/1294_Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection/tables/5f7c0f83f5355b77bb18ece830119bc2239a286c38afc87303c7d555722055e4.jpg)

![61470cc3cb17eeac0108afda0f1b64b04b6564729368531c366d6a6707e1be6c.jpg](../iclr_results/1294_Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection/tables/61470cc3cb17eeac0108afda0f1b64b04b6564729368531c366d6a6707e1be6c.jpg)

![7710e96a2ae240ec0bab56f5973bfb3c7d95b29fcf1db250c7bf52e4113aae8c.jpg](../iclr_results/1294_Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection/tables/7710e96a2ae240ec0bab56f5973bfb3c7d95b29fcf1db250c7bf52e4113aae8c.jpg)

![7b8532e22647f4a725d0ff31d236e56e2a6b69411dbe3d064e1e2f24dd72cb81.jpg](../iclr_results/1294_Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection/tables/7b8532e22647f4a725d0ff31d236e56e2a6b69411dbe3d064e1e2f24dd72cb81.jpg)

![e2730f027329e1348669806802082ab1b7e403a71bae8922655b0e508c6702b1.jpg](../iclr_results/1294_Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection/tables/e2730f027329e1348669806802082ab1b7e403a71bae8922655b0e508c6702b1.jpg)

![ee7f98aab2b2b0c9fa980f5d8a16119f2ca8ee3177f76d4eba60dc240a0acc83.jpg](../iclr_results/1294_Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection/tables/ee7f98aab2b2b0c9fa980f5d8a16119f2ca8ee3177f76d4eba60dc240a0acc83.jpg)

![f0bc4a4160a9401a1a238da1ed8909c97fc1b23d201f9f3e322e432ca8257451.jpg](../iclr_results/1294_Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection/tables/f0bc4a4160a9401a1a238da1ed8909c97fc1b23d201f9f3e322e432ca8257451.jpg)

![f0f3ca5fb6b4e6315cf36996da270446ae46ba9327b120ab72576b519e2084b3.jpg](../iclr_results/1294_Pursuing Feature Separation based on Neural Collapse for Out-of-Distribution Detection/tables/f0f3ca5fb6b4e6315cf36996da270446ae46ba9327b120ab72576b519e2084b3.jpg)

## CipherPrune:  Efficient and Scalable Private Transformer Inference


### Images

![09bfcd63759ece171e1a0c13bec922a49bebe4c4ffaa0f6120e7c797adea0c24.jpg](../iclr_results/1295_CipherPrune_  Efficient and Scalable Private Transformer Inference/images/09bfcd63759ece171e1a0c13bec922a49bebe4c4ffaa0f6120e7c797adea0c24.jpg)

![107c66f9cadfed2763d4312dbd056c1439ffe243fe39abb13aa53291998e768c.jpg](../iclr_results/1295_CipherPrune_  Efficient and Scalable Private Transformer Inference/images/107c66f9cadfed2763d4312dbd056c1439ffe243fe39abb13aa53291998e768c.jpg)

![108fa6cdfda00fa6537547ba412c93646abbf476cd280768b35fa51b7177b5b2.jpg](../iclr_results/1295_CipherPrune_  Efficient and Scalable Private Transformer Inference/images/108fa6cdfda00fa6537547ba412c93646abbf476cd280768b35fa51b7177b5b2.jpg)

![2ddfbabb5d41a098aff006358442e76f316b2dbea5d7c2851aa69b717d8ad9f8.jpg](../iclr_results/1295_CipherPrune_  Efficient and Scalable Private Transformer Inference/images/2ddfbabb5d41a098aff006358442e76f316b2dbea5d7c2851aa69b717d8ad9f8.jpg)

![399f7eec28e2a984a6d1f7de392cf4765fbf451e86ca8d9f9b151875677d1d4c.jpg](../iclr_results/1295_CipherPrune_  Efficient and Scalable Private Transformer Inference/images/399f7eec28e2a984a6d1f7de392cf4765fbf451e86ca8d9f9b151875677d1d4c.jpg)

![3a312d1de7a0e5c82fac89e67ad92f9b8036ed3c11d2d9397b2b06ac2db646d1.jpg](../iclr_results/1295_CipherPrune_  Efficient and Scalable Private Transformer Inference/images/3a312d1de7a0e5c82fac89e67ad92f9b8036ed3c11d2d9397b2b06ac2db646d1.jpg)

![4ba5f9b53dec1aab0f4b09d8dc5254af3631e763399cbd2d6b9a9cef97cb2847.jpg](../iclr_results/1295_CipherPrune_  Efficient and Scalable Private Transformer Inference/images/4ba5f9b53dec1aab0f4b09d8dc5254af3631e763399cbd2d6b9a9cef97cb2847.jpg)

![4c2c756606768e47a3d807850149f0c502282a5c3d76e884f94c2a78f7918524.jpg](../iclr_results/1295_CipherPrune_  Efficient and Scalable Private Transformer Inference/images/4c2c756606768e47a3d807850149f0c502282a5c3d76e884f94c2a78f7918524.jpg)

![731e14bbd2eb90697fef30e85b6c626b96b50ac4ae8df2bd9d092acdfef3c861.jpg](../iclr_results/1295_CipherPrune_  Efficient and Scalable Private Transformer Inference/images/731e14bbd2eb90697fef30e85b6c626b96b50ac4ae8df2bd9d092acdfef3c861.jpg)

![765c9c0fb5f47e638bde85d2eceb36d03e6634bab22aa1c3956987612705a857.jpg](../iclr_results/1295_CipherPrune_  Efficient and Scalable Private Transformer Inference/images/765c9c0fb5f47e638bde85d2eceb36d03e6634bab22aa1c3956987612705a857.jpg)

![8552499fd48dd53455075b9bb651a390b10f51a7f4d6fa8dbc3e7ad6b1e16a0f.jpg](../iclr_results/1295_CipherPrune_  Efficient and Scalable Private Transformer Inference/images/8552499fd48dd53455075b9bb651a390b10f51a7f4d6fa8dbc3e7ad6b1e16a0f.jpg)

![8e84e35519d3129db9aa5ba6df6b9bbcaa9850cfbae1bca35442a251b227c604.jpg](../iclr_results/1295_CipherPrune_  Efficient and Scalable Private Transformer Inference/images/8e84e35519d3129db9aa5ba6df6b9bbcaa9850cfbae1bca35442a251b227c604.jpg)

![b3fcc5a0f343100637d31ee0736288f4ecc61a26b8907f44f3650f40c4dd6146.jpg](../iclr_results/1295_CipherPrune_  Efficient and Scalable Private Transformer Inference/images/b3fcc5a0f343100637d31ee0736288f4ecc61a26b8907f44f3650f40c4dd6146.jpg)

![c5d876d151f6f7a8a8c4527194fd334933d0e1f14d1ed34d0102028324d4481c.jpg](../iclr_results/1295_CipherPrune_  Efficient and Scalable Private Transformer Inference/images/c5d876d151f6f7a8a8c4527194fd334933d0e1f14d1ed34d0102028324d4481c.jpg)

![d3104ad5a1b140384d794efde8e8e25dd0c42dff9d7203e8ea0c3d60bc51833e.jpg](../iclr_results/1295_CipherPrune_  Efficient and Scalable Private Transformer Inference/images/d3104ad5a1b140384d794efde8e8e25dd0c42dff9d7203e8ea0c3d60bc51833e.jpg)

![dfd0610c3e65bb19b668dbf37299488cb53b9fa34cd98d666ad70ea850654cb8.jpg](../iclr_results/1295_CipherPrune_  Efficient and Scalable Private Transformer Inference/images/dfd0610c3e65bb19b668dbf37299488cb53b9fa34cd98d666ad70ea850654cb8.jpg)

![f527eb7e13e845928c2f1d9b58820c801c6cd92ca8bff0dce138326302b3a615.jpg](../iclr_results/1295_CipherPrune_  Efficient and Scalable Private Transformer Inference/images/f527eb7e13e845928c2f1d9b58820c801c6cd92ca8bff0dce138326302b3a615.jpg)

![f8b5f03e8d323f58613efb1d755bba079c36e194856723d7e17c590b8caf0c20.jpg](../iclr_results/1295_CipherPrune_  Efficient and Scalable Private Transformer Inference/images/f8b5f03e8d323f58613efb1d755bba079c36e194856723d7e17c590b8caf0c20.jpg)

### Tables

![5da9f3a0244b994a3af68df19eab23b73dcaf02831fe994808f0914788f43555.jpg](../iclr_results/1295_CipherPrune_  Efficient and Scalable Private Transformer Inference/tables/5da9f3a0244b994a3af68df19eab23b73dcaf02831fe994808f0914788f43555.jpg)

![de4553a4b051027dc5785e5a283f51e47992e9831e330d67d57ee683b4e380a0.jpg](../iclr_results/1295_CipherPrune_  Efficient and Scalable Private Transformer Inference/tables/de4553a4b051027dc5785e5a283f51e47992e9831e330d67d57ee683b4e380a0.jpg)

![ea9c0828735f8ff14c7a5d89319d7353af8270b8b69b27a238296f3c3a8cee2d.jpg](../iclr_results/1295_CipherPrune_  Efficient and Scalable Private Transformer Inference/tables/ea9c0828735f8ff14c7a5d89319d7353af8270b8b69b27a238296f3c3a8cee2d.jpg)

## OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data


### Images

![1b1c91ad3dfb02c85b3772cf0beccbce69fcc0c8e014d5071df3f0f72839ae0f.jpg](../iclr_results/1296_OpenMathInstruct-2_ Accelerating AI for Math with Massive Open-Source Instruction Data/images/1b1c91ad3dfb02c85b3772cf0beccbce69fcc0c8e014d5071df3f0f72839ae0f.jpg)

![23b0e2a590c9e3320fa515d3cdf41261f7e0969e7ac9ca7acde936ae7e32f7d5.jpg](../iclr_results/1296_OpenMathInstruct-2_ Accelerating AI for Math with Massive Open-Source Instruction Data/images/23b0e2a590c9e3320fa515d3cdf41261f7e0969e7ac9ca7acde936ae7e32f7d5.jpg)

![2686da0946f95fe4645b0b02f994fb2d7986148d227d96d8646ae3fe8471cb39.jpg](../iclr_results/1296_OpenMathInstruct-2_ Accelerating AI for Math with Massive Open-Source Instruction Data/images/2686da0946f95fe4645b0b02f994fb2d7986148d227d96d8646ae3fe8471cb39.jpg)

![4ed40ace8d69527fb0e07c2879da43f6412601be56dff61d5793f8a6d695a531.jpg](../iclr_results/1296_OpenMathInstruct-2_ Accelerating AI for Math with Massive Open-Source Instruction Data/images/4ed40ace8d69527fb0e07c2879da43f6412601be56dff61d5793f8a6d695a531.jpg)

![68f211ead7a47ba220c9152b94486d2452353f1833f3ba64d83c49e7807e0bdc.jpg](../iclr_results/1296_OpenMathInstruct-2_ Accelerating AI for Math with Massive Open-Source Instruction Data/images/68f211ead7a47ba220c9152b94486d2452353f1833f3ba64d83c49e7807e0bdc.jpg)

![88ae6036ab34b705e0273acfb72f6f0d982b093e6c58f2567e7f1843cc2613c6.jpg](../iclr_results/1296_OpenMathInstruct-2_ Accelerating AI for Math with Massive Open-Source Instruction Data/images/88ae6036ab34b705e0273acfb72f6f0d982b093e6c58f2567e7f1843cc2613c6.jpg)

![cf30635d50bda2023c3f8a184bc59ec1cd8955ba8fc25e0ea21915710f790852.jpg](../iclr_results/1296_OpenMathInstruct-2_ Accelerating AI for Math with Massive Open-Source Instruction Data/images/cf30635d50bda2023c3f8a184bc59ec1cd8955ba8fc25e0ea21915710f790852.jpg)

![d9587af5d034bb297934a513d0eefaa47e50588a3a7e4fcbf77f07b55d3d0b05.jpg](../iclr_results/1296_OpenMathInstruct-2_ Accelerating AI for Math with Massive Open-Source Instruction Data/images/d9587af5d034bb297934a513d0eefaa47e50588a3a7e4fcbf77f07b55d3d0b05.jpg)

![ebe950cae3736f8f6ae0130176d9661ebe62c006209025653f94e284aa9277dd.jpg](../iclr_results/1296_OpenMathInstruct-2_ Accelerating AI for Math with Massive Open-Source Instruction Data/images/ebe950cae3736f8f6ae0130176d9661ebe62c006209025653f94e284aa9277dd.jpg)

### Tables

![04e2f97ab2b4000c0f65a85638e43c5c69154aabb9fdb796d630e9c4fb1854a3.jpg](../iclr_results/1296_OpenMathInstruct-2_ Accelerating AI for Math with Massive Open-Source Instruction Data/tables/04e2f97ab2b4000c0f65a85638e43c5c69154aabb9fdb796d630e9c4fb1854a3.jpg)

![1ca33751737796231953abef9153d09f325e498454585fa6edeac5b6c61d8267.jpg](../iclr_results/1296_OpenMathInstruct-2_ Accelerating AI for Math with Massive Open-Source Instruction Data/tables/1ca33751737796231953abef9153d09f325e498454585fa6edeac5b6c61d8267.jpg)

![2106a169cc5ecfb44dcf4704f275c4da5e109dbe1f3078811065b81e392dc69a.jpg](../iclr_results/1296_OpenMathInstruct-2_ Accelerating AI for Math with Massive Open-Source Instruction Data/tables/2106a169cc5ecfb44dcf4704f275c4da5e109dbe1f3078811065b81e392dc69a.jpg)

![26e3ee22a7c93bc592175cb02f39d737331f3be4b7d450280a7e6abb3dad9a1b.jpg](../iclr_results/1296_OpenMathInstruct-2_ Accelerating AI for Math with Massive Open-Source Instruction Data/tables/26e3ee22a7c93bc592175cb02f39d737331f3be4b7d450280a7e6abb3dad9a1b.jpg)

![534956ec42c7b29979b76f9773d57955107b649badeeb3d4501c7eddd7b78a53.jpg](../iclr_results/1296_OpenMathInstruct-2_ Accelerating AI for Math with Massive Open-Source Instruction Data/tables/534956ec42c7b29979b76f9773d57955107b649badeeb3d4501c7eddd7b78a53.jpg)

![81c1cecc76539c3b5aaeb3bd38b778db4954b9329be918a2475156ff8e0a893e.jpg](../iclr_results/1296_OpenMathInstruct-2_ Accelerating AI for Math with Massive Open-Source Instruction Data/tables/81c1cecc76539c3b5aaeb3bd38b778db4954b9329be918a2475156ff8e0a893e.jpg)

![ac85938e52033b4d23cf46c0659f65b0f672657da8d0b299cc77c4c68fbad36e.jpg](../iclr_results/1296_OpenMathInstruct-2_ Accelerating AI for Math with Massive Open-Source Instruction Data/tables/ac85938e52033b4d23cf46c0659f65b0f672657da8d0b299cc77c4c68fbad36e.jpg)

![bb5c4760e713d072dddbb53dea1593296b61f8390488088a8df4b32439abe028.jpg](../iclr_results/1296_OpenMathInstruct-2_ Accelerating AI for Math with Massive Open-Source Instruction Data/tables/bb5c4760e713d072dddbb53dea1593296b61f8390488088a8df4b32439abe028.jpg)

![d7d951c2c4d856c59f25fe6b449adbffa2a9ce1a5c863a807a7c16c0bd31bc6f.jpg](../iclr_results/1296_OpenMathInstruct-2_ Accelerating AI for Math with Massive Open-Source Instruction Data/tables/d7d951c2c4d856c59f25fe6b449adbffa2a9ce1a5c863a807a7c16c0bd31bc6f.jpg)

![e3214b56d4eacf1ef5b38ea3870003bf09b4c6ea787523f9d2de731126b48b95.jpg](../iclr_results/1296_OpenMathInstruct-2_ Accelerating AI for Math with Massive Open-Source Instruction Data/tables/e3214b56d4eacf1ef5b38ea3870003bf09b4c6ea787523f9d2de731126b48b95.jpg)

![e3cdff6ed71c642222a557835bb748c42af0e0c2ac4243db9a19620fb2e14355.jpg](../iclr_results/1296_OpenMathInstruct-2_ Accelerating AI for Math with Massive Open-Source Instruction Data/tables/e3cdff6ed71c642222a557835bb748c42af0e0c2ac4243db9a19620fb2e14355.jpg)

## Last Iterate Convergence of Incremental Methods as a Model of Forgetting


### Images

![3ddba6b9cd88f5e37bb265de75847cdaf5eb6f2fc342388df4179c607acbecba.jpg](../iclr_results/1297_Last Iterate Convergence of Incremental Methods as a Model of Forgetting/images/3ddba6b9cd88f5e37bb265de75847cdaf5eb6f2fc342388df4179c607acbecba.jpg)

![648c760a5a8276afb27d51098ee9c13afbe210f5eb182c2f5eb610528b48096a.jpg](../iclr_results/1297_Last Iterate Convergence of Incremental Methods as a Model of Forgetting/images/648c760a5a8276afb27d51098ee9c13afbe210f5eb182c2f5eb610528b48096a.jpg)

### Tables

![0b75d4adf681ac7ca162bd8f637d5ef9f55d1ba42ab3e0ebaa549869884b2725.jpg](../iclr_results/1297_Last Iterate Convergence of Incremental Methods as a Model of Forgetting/tables/0b75d4adf681ac7ca162bd8f637d5ef9f55d1ba42ab3e0ebaa549869884b2725.jpg)

![72c77ccfa0392fb741a163895edb23e5d4b9e0d93c6b4ac627f614ca0dd08f84.jpg](../iclr_results/1297_Last Iterate Convergence of Incremental Methods as a Model of Forgetting/tables/72c77ccfa0392fb741a163895edb23e5d4b9e0d93c6b4ac627f614ca0dd08f84.jpg)

## Visually Guided Decoding: Gradient-Free Hard Prompt Inversion with Language Models


### Images

![0a668b8a0ad4abf7409f9e8289e4e3628f1e2f0cc9f17c71e1b3b11add2fe22b.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/images/0a668b8a0ad4abf7409f9e8289e4e3628f1e2f0cc9f17c71e1b3b11add2fe22b.jpg)

![399b7d51d48224b0d6095cdaf3849dbbaa2c7bc490cf6ff8f2a160a88b5aac35.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/images/399b7d51d48224b0d6095cdaf3849dbbaa2c7bc490cf6ff8f2a160a88b5aac35.jpg)

![3c9059bab75ce95f325c0cb83a63aac02823bdac712c83b1ad49bf9c80ba12bd.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/images/3c9059bab75ce95f325c0cb83a63aac02823bdac712c83b1ad49bf9c80ba12bd.jpg)

![4a205fa1e63d715a8469c70418b784685ec43b921f3e4a76a1411c0bc3b2a648.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/images/4a205fa1e63d715a8469c70418b784685ec43b921f3e4a76a1411c0bc3b2a648.jpg)

![5f5904df8370b602a5809f083090f16636985a94402574d46b54358419528033.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/images/5f5904df8370b602a5809f083090f16636985a94402574d46b54358419528033.jpg)

![6a30262a3bb02c48930e6103f47cf8dabff27070aa23382c38097eafff8b6b5c.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/images/6a30262a3bb02c48930e6103f47cf8dabff27070aa23382c38097eafff8b6b5c.jpg)

![7864da1446621f5297433593dcc53e61861ab94417980e6bbcdef52e8338599d.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/images/7864da1446621f5297433593dcc53e61861ab94417980e6bbcdef52e8338599d.jpg)

![7d00f0ea79aa7352270fa17ca95f1bf1a13b46b1bc46f35295db0c9b1e85d7df.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/images/7d00f0ea79aa7352270fa17ca95f1bf1a13b46b1bc46f35295db0c9b1e85d7df.jpg)

![8f5818b382e2234c5f425a5eafc6e9650128397cc48ef5b062deb565002ec187.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/images/8f5818b382e2234c5f425a5eafc6e9650128397cc48ef5b062deb565002ec187.jpg)

![8f7e13ca69ed8d0db0992046d2d1b9c82075cd9f3c2018f69df63261ff39c4b6.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/images/8f7e13ca69ed8d0db0992046d2d1b9c82075cd9f3c2018f69df63261ff39c4b6.jpg)

![a66e3e0aee4dafb5a0df1fdccd64e9e41a06bfecc674389d82113aef582af768.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/images/a66e3e0aee4dafb5a0df1fdccd64e9e41a06bfecc674389d82113aef582af768.jpg)

![a9ba43b78d839e7de69c3e277f4ef3795c35f321bd7619d46789e383efa08cb1.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/images/a9ba43b78d839e7de69c3e277f4ef3795c35f321bd7619d46789e383efa08cb1.jpg)

![ad0ee45dece3658f13517f16097bea873c46f63c5b07162960f605385af595f5.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/images/ad0ee45dece3658f13517f16097bea873c46f63c5b07162960f605385af595f5.jpg)

![b5869d09211eaea72f820d9f572163771da6bb81a1f6a32e361b5385eb5b08f6.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/images/b5869d09211eaea72f820d9f572163771da6bb81a1f6a32e361b5385eb5b08f6.jpg)

![bc68e225137673d19d4cb529f085188bfe36486f5235f1228833be4c18afc4c7.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/images/bc68e225137673d19d4cb529f085188bfe36486f5235f1228833be4c18afc4c7.jpg)

![c05934f4187cd4d1136045e238b1d768b45dfa87c643d20efa36a62f51f1310d.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/images/c05934f4187cd4d1136045e238b1d768b45dfa87c643d20efa36a62f51f1310d.jpg)

![de67f1e4a9b3eb116a671036ca5a7c9809006bec36780248c52e8be65e5cafd1.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/images/de67f1e4a9b3eb116a671036ca5a7c9809006bec36780248c52e8be65e5cafd1.jpg)

![e953c6eeea7901a55d0680b1d0fc1fd64d61b98c7e8509bdf1e7f692bcb817a4.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/images/e953c6eeea7901a55d0680b1d0fc1fd64d61b98c7e8509bdf1e7f692bcb817a4.jpg)

![ed9cc31059c0414348c304b5be0b7de590031c87d13363554aa0cd881abe64e2.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/images/ed9cc31059c0414348c304b5be0b7de590031c87d13363554aa0cd881abe64e2.jpg)

![febdefb1f18158568c0a4b91c29c00d43d86a6c7636f3fa0fab9224a77269eba.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/images/febdefb1f18158568c0a4b91c29c00d43d86a6c7636f3fa0fab9224a77269eba.jpg)

![ff35e5640a30d640ae60f21d43d44002f2193bd1bc42562058817277c26be03c.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/images/ff35e5640a30d640ae60f21d43d44002f2193bd1bc42562058817277c26be03c.jpg)

### Tables

![08ab56f523cae090ef30032ad215b454589d0d51f81a4f297f45279633ca4bce.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/tables/08ab56f523cae090ef30032ad215b454589d0d51f81a4f297f45279633ca4bce.jpg)

![4835e897edf9842d3e6799bb6f8a3b78e0f51cd7fa067b54d16199cef5b45543.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/tables/4835e897edf9842d3e6799bb6f8a3b78e0f51cd7fa067b54d16199cef5b45543.jpg)

![afdff17ae38b5802049da304c17666323474e5dce2817e10409928aa75e203a1.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/tables/afdff17ae38b5802049da304c17666323474e5dce2817e10409928aa75e203a1.jpg)

![ca6f184b1527e07b140ebca07b2bd1f9873a420e70ce4c88bfb92b1228eae466.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/tables/ca6f184b1527e07b140ebca07b2bd1f9873a420e70ce4c88bfb92b1228eae466.jpg)

![d1841a72d619131c9e28f27f7c19fabc5d179a8562bdbd949874e8c42d10bd97.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/tables/d1841a72d619131c9e28f27f7c19fabc5d179a8562bdbd949874e8c42d10bd97.jpg)

![d9a9675ba62f1a7942854864083051607c3f71e640547aca865a4e5f2c34d7f5.jpg](../iclr_results/1298_Visually Guided Decoding_ Gradient-Free Hard Prompt Inversion with Language Models/tables/d9a9675ba62f1a7942854864083051607c3f71e640547aca865a4e5f2c34d7f5.jpg)

## AgentSquare: Automatic LLM Agent Search in Modular Design Space


### Images

![07c4d378264de16a7a2c51d1df19606d8d5641b16238145c8a2ffecece42ac16.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/images/07c4d378264de16a7a2c51d1df19606d8d5641b16238145c8a2ffecece42ac16.jpg)

![0a4470949b372dfde4526af96aea098d008202c6fcc8659042331454e822a017.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/images/0a4470949b372dfde4526af96aea098d008202c6fcc8659042331454e822a017.jpg)

![0cb994bf42ce7a1b97d9ab4c9d9734ad52fc322f7da86144b8c561982ff18429.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/images/0cb994bf42ce7a1b97d9ab4c9d9734ad52fc322f7da86144b8c561982ff18429.jpg)

![1ac2b80638c6b17c1947fe9996704f1dbd25333a226175a9ee0c922df8c1a564.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/images/1ac2b80638c6b17c1947fe9996704f1dbd25333a226175a9ee0c922df8c1a564.jpg)

![2bcb5ee54e571a9fe16af6bd5d1c8ff00d1a7cddcd5e4f136923e842bc785a68.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/images/2bcb5ee54e571a9fe16af6bd5d1c8ff00d1a7cddcd5e4f136923e842bc785a68.jpg)

![35750892fa1810024d8aeca72801430b5d6e4ca4271a8af18f70e2a02921bd0b.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/images/35750892fa1810024d8aeca72801430b5d6e4ca4271a8af18f70e2a02921bd0b.jpg)

![40d6fd77fcc25bc087c9653d4c78a8d0cc565abac791b4e2c98026550efe66b4.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/images/40d6fd77fcc25bc087c9653d4c78a8d0cc565abac791b4e2c98026550efe66b4.jpg)

![56aa1d4f86290be01e289cb8ee55f85a3fd097f86881899eb9fcd5fc6be1a4f8.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/images/56aa1d4f86290be01e289cb8ee55f85a3fd097f86881899eb9fcd5fc6be1a4f8.jpg)

![5d327939e7ab1933ac438f2a8fcefe4ff000253ae338e65c0be72959f8f3c6c8.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/images/5d327939e7ab1933ac438f2a8fcefe4ff000253ae338e65c0be72959f8f3c6c8.jpg)

![6c4c1fe3fe59319f4db3e9a249bd2439dd0daf1664f8497ffa5f578b58de65e2.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/images/6c4c1fe3fe59319f4db3e9a249bd2439dd0daf1664f8497ffa5f578b58de65e2.jpg)

![6da96608e7d81e18a5ed3f39e0ba2c9e84713438c78aad322926ffd4470fa254.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/images/6da96608e7d81e18a5ed3f39e0ba2c9e84713438c78aad322926ffd4470fa254.jpg)

![7bf65f0c79f16bb858f4672db08750468475c1d634e0580c878f5d051765c67f.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/images/7bf65f0c79f16bb858f4672db08750468475c1d634e0580c878f5d051765c67f.jpg)

![83c48726d6f6faba7f94ea54670b551a9cdb5f10c923090ec8b29f20fc541a5d.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/images/83c48726d6f6faba7f94ea54670b551a9cdb5f10c923090ec8b29f20fc541a5d.jpg)

![ab7f7b13e25de099914ce2c695d84601d64808863bec33f6651a42c92b407b9a.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/images/ab7f7b13e25de099914ce2c695d84601d64808863bec33f6651a42c92b407b9a.jpg)

![ad26a91fb7710ddf51f43c61853d70fee05a3f6941dee8c007b3ee3258c54b90.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/images/ad26a91fb7710ddf51f43c61853d70fee05a3f6941dee8c007b3ee3258c54b90.jpg)

![b3ba833ffe7ca330605eed88b0f9eac57c15e6c301662b70daa2c93e2c4c868d.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/images/b3ba833ffe7ca330605eed88b0f9eac57c15e6c301662b70daa2c93e2c4c868d.jpg)

![da60b4ce153c9c2754cef6dec9bfff90b987089b4bd7cdcab1b8acab0a080860.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/images/da60b4ce153c9c2754cef6dec9bfff90b987089b4bd7cdcab1b8acab0a080860.jpg)

![de7001c109c31ff24c1c89de285e55ee0380b8ec0bc8db588c797d83cb535d1b.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/images/de7001c109c31ff24c1c89de285e55ee0380b8ec0bc8db588c797d83cb535d1b.jpg)

![dff96ad4d805fd6678d126a61b1da0dc156bde886b1497745e5700a2d0f43ac0.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/images/dff96ad4d805fd6678d126a61b1da0dc156bde886b1497745e5700a2d0f43ac0.jpg)

![e244559dc28ad7a763cfcb5e10fa0ec1613ab46ee651b30aaef95d770a3bea37.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/images/e244559dc28ad7a763cfcb5e10fa0ec1613ab46ee651b30aaef95d770a3bea37.jpg)

![eb8bd6c1a0aa74991b7d7ee66d576ecc0614cee829706f92885437b0972db276.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/images/eb8bd6c1a0aa74991b7d7ee66d576ecc0614cee829706f92885437b0972db276.jpg)

![f83153175e5c09784f28582682517f89adc4b3f3a52bc8e5b67aedb8c9d0bdce.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/images/f83153175e5c09784f28582682517f89adc4b3f3a52bc8e5b67aedb8c9d0bdce.jpg)

### Tables

![1bb24e4967abb433dbcd5dad9b9e2e398d4c64020d20516faec51893fad76fbc.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/tables/1bb24e4967abb433dbcd5dad9b9e2e398d4c64020d20516faec51893fad76fbc.jpg)

![3c363a5afdf01ae2a6c4fd5f34e75192632adba5fff7290309fd1cb05535cd62.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/tables/3c363a5afdf01ae2a6c4fd5f34e75192632adba5fff7290309fd1cb05535cd62.jpg)

![45b20814adc371b4952ff7f669b6c13e29d81f4a10cd53de9702083b1d8ab29e.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/tables/45b20814adc371b4952ff7f669b6c13e29d81f4a10cd53de9702083b1d8ab29e.jpg)

![542f9ad1b7498ba6971789f4e897e47a1e6583cc3bcf8e7587f4b6df89616231.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/tables/542f9ad1b7498ba6971789f4e897e47a1e6583cc3bcf8e7587f4b6df89616231.jpg)

![604ef7f5ded28e4f4455a95bbcf053d5666b50a14fd4b825c54ff068024c3ea8.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/tables/604ef7f5ded28e4f4455a95bbcf053d5666b50a14fd4b825c54ff068024c3ea8.jpg)

![70e9f617bac6e7d44f4a14d8e4cd9a1a9cac3d2a7df44e92a4ca55c2405f124d.jpg](../iclr_results/1299_AgentSquare_ Automatic LLM Agent Search in Modular Design Space/tables/70e9f617bac6e7d44f4a14d8e4cd9a1a9cac3d2a7df44e92a4ca55c2405f124d.jpg)

## GS-CPR: Efficient Camera Pose Refinement via 3D Gaussian Splatting


### Images

![05b3a83769318f9e6c7d7c43fdec166c3590408e0affb3f693cf22c3dea4ca20.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/images/05b3a83769318f9e6c7d7c43fdec166c3590408e0affb3f693cf22c3dea4ca20.jpg)

![05ef8a1cf5913ecac83acadf68a3701e04374404652b04529639767a9c0c7468.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/images/05ef8a1cf5913ecac83acadf68a3701e04374404652b04529639767a9c0c7468.jpg)

![135a8f0055fa9ea172a5ec74348ace4b8b51d5b10873116d623d9d576af2be3d.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/images/135a8f0055fa9ea172a5ec74348ace4b8b51d5b10873116d623d9d576af2be3d.jpg)

![197ba7e201768f3b632ecfb2d8c5c7421330090ad738d62d29469b4c5ef22122.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/images/197ba7e201768f3b632ecfb2d8c5c7421330090ad738d62d29469b4c5ef22122.jpg)

![316221009c73059cb3e55fafa14d22cd3c7972d5a873f542f74c2441194d41a6.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/images/316221009c73059cb3e55fafa14d22cd3c7972d5a873f542f74c2441194d41a6.jpg)

![6c8bbdff44ebfc0a206e01e49a850c0c97aa960e3b8bf6485eacf3db778d9b8b.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/images/6c8bbdff44ebfc0a206e01e49a850c0c97aa960e3b8bf6485eacf3db778d9b8b.jpg)

![8f1abfc275c418e3d6cbbaf0c042c63294adcc5e995b3fba385690cf8816049f.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/images/8f1abfc275c418e3d6cbbaf0c042c63294adcc5e995b3fba385690cf8816049f.jpg)

![98b2326aabbc95a42075fbafb0f9e5039d5233b47b9452e22d3bd6750f94440d.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/images/98b2326aabbc95a42075fbafb0f9e5039d5233b47b9452e22d3bd6750f94440d.jpg)

![99d8d8c7a5d048a7ac1f7dabf2bb9d7288b138780474fce58cfc1e359ed94dc0.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/images/99d8d8c7a5d048a7ac1f7dabf2bb9d7288b138780474fce58cfc1e359ed94dc0.jpg)

![9cad5babbe2b626c74a60be43cf43d3c2983b76a33164719d83af45ff69861ad.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/images/9cad5babbe2b626c74a60be43cf43d3c2983b76a33164719d83af45ff69861ad.jpg)

![c7adf086b80e8b903eea52a13643df12dc3aab9aaa18b88316b4d46a98132448.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/images/c7adf086b80e8b903eea52a13643df12dc3aab9aaa18b88316b4d46a98132448.jpg)

![f8bc7bc27a170efb22e46472b8d3ddd179760fc9840b8ccf799bfb579615127d.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/images/f8bc7bc27a170efb22e46472b8d3ddd179760fc9840b8ccf799bfb579615127d.jpg)

### Tables

![158c7cda61308d14312da966e75e88fedf910afa3a4d14ded432eb96c1f5230a.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/tables/158c7cda61308d14312da966e75e88fedf910afa3a4d14ded432eb96c1f5230a.jpg)

![31b3910bbc0ee90a7f601fb25ce36864bbac20f37d1da7f3cb3d3155b553efee.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/tables/31b3910bbc0ee90a7f601fb25ce36864bbac20f37d1da7f3cb3d3155b553efee.jpg)

![338ee5ce88519e4719eadeb3582939fb2b621534f5865032fc1cff5d8b041e53.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/tables/338ee5ce88519e4719eadeb3582939fb2b621534f5865032fc1cff5d8b041e53.jpg)

![3f9875fffac371f2ec08fac0985fcb3b11443ccc39174c9f38af7f9d643b94f6.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/tables/3f9875fffac371f2ec08fac0985fcb3b11443ccc39174c9f38af7f9d643b94f6.jpg)

![43526a6573876f59582aeee0fbd3a44bb2176c3bc27e669da5f94d44dd48bd31.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/tables/43526a6573876f59582aeee0fbd3a44bb2176c3bc27e669da5f94d44dd48bd31.jpg)

![49bf1b09dda884b2b9397e62495040f5443be30d8a35173662fb507dd8317469.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/tables/49bf1b09dda884b2b9397e62495040f5443be30d8a35173662fb507dd8317469.jpg)

![6855be746ad80623c954bb1c4dda32e5cc5a9378be968c899d159ee380c9bea4.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/tables/6855be746ad80623c954bb1c4dda32e5cc5a9378be968c899d159ee380c9bea4.jpg)

![8eaaafc30dd9392317d6da08661d36a767ef8ff8e61ddfb016db98926dd74940.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/tables/8eaaafc30dd9392317d6da08661d36a767ef8ff8e61ddfb016db98926dd74940.jpg)

![ab91f23e1d8d5ea365a054268009a2fe584abae5ffcacb8cd9808d5ab44b117f.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/tables/ab91f23e1d8d5ea365a054268009a2fe584abae5ffcacb8cd9808d5ab44b117f.jpg)

![c3a07bcbf1ebff00e8530e746a8f1c98f3493b2dc0fa217b6e02cef5623d72d9.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/tables/c3a07bcbf1ebff00e8530e746a8f1c98f3493b2dc0fa217b6e02cef5623d72d9.jpg)

![c694fbeae96bb62d6c6c6fd3aeaa10cedae14d5eb46629a7caf3f4d2eac5f612.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/tables/c694fbeae96bb62d6c6c6fd3aeaa10cedae14d5eb46629a7caf3f4d2eac5f612.jpg)

![ef64ab77b8a3040ef7ab42449f250715ff3918bceed75040abc6228507d56acb.jpg](../iclr_results/1300_GS-CPR_ Efficient Camera Pose Refinement via 3D Gaussian Splatting/tables/ef64ab77b8a3040ef7ab42449f250715ff3918bceed75040abc6228507d56acb.jpg)

## DRESSing Up LLM: Efficient Stylized Question-Answering via Style Subspace Editing


### Images

![1d0433ae6248db3576cf7e96cfdecce19d528e53862e51ee0cf93df88ba9f1c1.jpg](../iclr_results/1301_DRESSing Up LLM_ Efficient Stylized Question-Answering via Style Subspace Editing/images/1d0433ae6248db3576cf7e96cfdecce19d528e53862e51ee0cf93df88ba9f1c1.jpg)

![591ccd47eb1feef7196c3a28927c78839f69d9e3996f3ee19dc9f5b1cecb2c2c.jpg](../iclr_results/1301_DRESSing Up LLM_ Efficient Stylized Question-Answering via Style Subspace Editing/images/591ccd47eb1feef7196c3a28927c78839f69d9e3996f3ee19dc9f5b1cecb2c2c.jpg)

![6bf36879ddab29030d69e19be12d153b708d98d745ecdbc5f77c387bec9b0efe.jpg](../iclr_results/1301_DRESSing Up LLM_ Efficient Stylized Question-Answering via Style Subspace Editing/images/6bf36879ddab29030d69e19be12d153b708d98d745ecdbc5f77c387bec9b0efe.jpg)

![ac33cf3e35bd96d8de10f0cf6917ad43274fc619f783131890db44f3fd826386.jpg](../iclr_results/1301_DRESSing Up LLM_ Efficient Stylized Question-Answering via Style Subspace Editing/images/ac33cf3e35bd96d8de10f0cf6917ad43274fc619f783131890db44f3fd826386.jpg)

![be35c92d560912ad03ab79688b1e261f9cb4c5a017167228aea0506397a0dc29.jpg](../iclr_results/1301_DRESSing Up LLM_ Efficient Stylized Question-Answering via Style Subspace Editing/images/be35c92d560912ad03ab79688b1e261f9cb4c5a017167228aea0506397a0dc29.jpg)

![f404563f97b8a3e877089f2cc34968518e7fd4dd8493e114243e598a948d5823.jpg](../iclr_results/1301_DRESSing Up LLM_ Efficient Stylized Question-Answering via Style Subspace Editing/images/f404563f97b8a3e877089f2cc34968518e7fd4dd8493e114243e598a948d5823.jpg)

### Tables

![0535215ef85e6d6ab055d0ed44bdc55b9447bf155f96e6efc04d0fa79146a57c.jpg](../iclr_results/1301_DRESSing Up LLM_ Efficient Stylized Question-Answering via Style Subspace Editing/tables/0535215ef85e6d6ab055d0ed44bdc55b9447bf155f96e6efc04d0fa79146a57c.jpg)

![2de1ad4711fdb059fdd0c9bbb93746948659f402806e0035a6db77f90aa89f72.jpg](../iclr_results/1301_DRESSing Up LLM_ Efficient Stylized Question-Answering via Style Subspace Editing/tables/2de1ad4711fdb059fdd0c9bbb93746948659f402806e0035a6db77f90aa89f72.jpg)

![69dabdec728886c954612396308798e981ceabfc5208244015e3ac9cc0d59f71.jpg](../iclr_results/1301_DRESSing Up LLM_ Efficient Stylized Question-Answering via Style Subspace Editing/tables/69dabdec728886c954612396308798e981ceabfc5208244015e3ac9cc0d59f71.jpg)

![87c9328e01852b6b3f190ff7ff87a8323c26a15ebebe26f2173bd69e114c6a4e.jpg](../iclr_results/1301_DRESSing Up LLM_ Efficient Stylized Question-Answering via Style Subspace Editing/tables/87c9328e01852b6b3f190ff7ff87a8323c26a15ebebe26f2173bd69e114c6a4e.jpg)

![aed69e32ef559b525963c138500c257882446303774ae74baabd9c2689c721e2.jpg](../iclr_results/1301_DRESSing Up LLM_ Efficient Stylized Question-Answering via Style Subspace Editing/tables/aed69e32ef559b525963c138500c257882446303774ae74baabd9c2689c721e2.jpg)

![bdce0d10fb92aefc9afa49bbe42a43715577c14fc7fd723e639b794c4701b780.jpg](../iclr_results/1301_DRESSing Up LLM_ Efficient Stylized Question-Answering via Style Subspace Editing/tables/bdce0d10fb92aefc9afa49bbe42a43715577c14fc7fd723e639b794c4701b780.jpg)

![d9c9bba7e652f1ef9e7df6919c112a93aba49ffc70613ea75879b18f922c7a32.jpg](../iclr_results/1301_DRESSing Up LLM_ Efficient Stylized Question-Answering via Style Subspace Editing/tables/d9c9bba7e652f1ef9e7df6919c112a93aba49ffc70613ea75879b18f922c7a32.jpg)

## Fragment and Geometry Aware Tokenization of Molecules for Structure-Based Drug Design Using Language Models


### Images

![54d6542540bb836f8bc6a0db4b567db002c794e2dae0b4877ec98172007f7ac3.jpg](../iclr_results/1302_Fragment and Geometry Aware Tokenization of Molecules for Structure-Based Drug Design Using Language/images/54d6542540bb836f8bc6a0db4b567db002c794e2dae0b4877ec98172007f7ac3.jpg)

![837c8231fe73c429d55e9ded508882f7d1f1422dcb8928e48c7c186fc24a3cd7.jpg](../iclr_results/1302_Fragment and Geometry Aware Tokenization of Molecules for Structure-Based Drug Design Using Language/images/837c8231fe73c429d55e9ded508882f7d1f1422dcb8928e48c7c186fc24a3cd7.jpg)

![98f6a4f519e4ef9866e0d2fc4d0ce2a926b0eb2465b533bc8445d12969dee39e.jpg](../iclr_results/1302_Fragment and Geometry Aware Tokenization of Molecules for Structure-Based Drug Design Using Language/images/98f6a4f519e4ef9866e0d2fc4d0ce2a926b0eb2465b533bc8445d12969dee39e.jpg)

![b487b5e99e34e20423de78f5c28fb9268ca10aa877123850ac6a7c22064d38b4.jpg](../iclr_results/1302_Fragment and Geometry Aware Tokenization of Molecules for Structure-Based Drug Design Using Language/images/b487b5e99e34e20423de78f5c28fb9268ca10aa877123850ac6a7c22064d38b4.jpg)

![f73456530643d8da73d33ec99c909fe2dd0e995fa27541ae9e46ee8523cc2125.jpg](../iclr_results/1302_Fragment and Geometry Aware Tokenization of Molecules for Structure-Based Drug Design Using Language/images/f73456530643d8da73d33ec99c909fe2dd0e995fa27541ae9e46ee8523cc2125.jpg)

### Tables

![5a25bc8784a05c3a3455900c8dba9ecd494cf7fdc3b9675101cca20302c68c85.jpg](../iclr_results/1302_Fragment and Geometry Aware Tokenization of Molecules for Structure-Based Drug Design Using Language/tables/5a25bc8784a05c3a3455900c8dba9ecd494cf7fdc3b9675101cca20302c68c85.jpg)

![7235b4df361915f62536093ee563c6442b74de8fd39ba65ace92390e2669bfac.jpg](../iclr_results/1302_Fragment and Geometry Aware Tokenization of Molecules for Structure-Based Drug Design Using Language/tables/7235b4df361915f62536093ee563c6442b74de8fd39ba65ace92390e2669bfac.jpg)

![828a08b8717a6ac644ce5392f4015e1972664960be472c2011b3ab5de47469c8.jpg](../iclr_results/1302_Fragment and Geometry Aware Tokenization of Molecules for Structure-Based Drug Design Using Language/tables/828a08b8717a6ac644ce5392f4015e1972664960be472c2011b3ab5de47469c8.jpg)

![ab1e9eb987a31447a7e92e85a8c86a2181f76de52a6a88000b0395c8acf2921c.jpg](../iclr_results/1302_Fragment and Geometry Aware Tokenization of Molecules for Structure-Based Drug Design Using Language/tables/ab1e9eb987a31447a7e92e85a8c86a2181f76de52a6a88000b0395c8acf2921c.jpg)

![cc82ad1ea8afedcccbbc773fb6fc025f279f92ab883c209dfae9b163997e2433.jpg](../iclr_results/1302_Fragment and Geometry Aware Tokenization of Molecules for Structure-Based Drug Design Using Language/tables/cc82ad1ea8afedcccbbc773fb6fc025f279f92ab883c209dfae9b163997e2433.jpg)

![dfeeffe493ccc6ff97f8845c4760b4317de853c5b2469ff54f1b273bb23001c4.jpg](../iclr_results/1302_Fragment and Geometry Aware Tokenization of Molecules for Structure-Based Drug Design Using Language/tables/dfeeffe493ccc6ff97f8845c4760b4317de853c5b2469ff54f1b273bb23001c4.jpg)

![f1906a9728ee3e06fd62d6a3fbfc937d38798a55a9269d016d4d61e8825c8b4e.jpg](../iclr_results/1302_Fragment and Geometry Aware Tokenization of Molecules for Structure-Based Drug Design Using Language/tables/f1906a9728ee3e06fd62d6a3fbfc937d38798a55a9269d016d4d61e8825c8b4e.jpg)

![f21f8b9aac7d28a0ea1492c0e67982cc1bae2b1915979556175e38c25a5b3093.jpg](../iclr_results/1302_Fragment and Geometry Aware Tokenization of Molecules for Structure-Based Drug Design Using Language/tables/f21f8b9aac7d28a0ea1492c0e67982cc1bae2b1915979556175e38c25a5b3093.jpg)

## ParetoFlow: Guided Flows in Multi-Objective Optimization


### Images

![06361c0c03fbf29f81462e9abec1031c1c332d805824f6a120c943e673bc4ad1.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/images/06361c0c03fbf29f81462e9abec1031c1c332d805824f6a120c943e673bc4ad1.jpg)

![12dea94d37246ea85dbd643c423d2bb35be58b193d165a5d94e8cacd0a81ed8a.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/images/12dea94d37246ea85dbd643c423d2bb35be58b193d165a5d94e8cacd0a81ed8a.jpg)

![2e33155997afbce6dcf74a1d71c35a6fc5bb5c1c360755d110bb06d217f7ec27.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/images/2e33155997afbce6dcf74a1d71c35a6fc5bb5c1c360755d110bb06d217f7ec27.jpg)

![4593a19dc4350368a285fb3c8bc3c90f488a3b0f7c57b6261aa9883ffb177553.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/images/4593a19dc4350368a285fb3c8bc3c90f488a3b0f7c57b6261aa9883ffb177553.jpg)

![59a4666e97c103f74e5fe4980bf7718b62666080a57fabeb929840d58f760ff2.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/images/59a4666e97c103f74e5fe4980bf7718b62666080a57fabeb929840d58f760ff2.jpg)

![70e107b965392ddfeea312f901f649ff97e2b88687a45d7d62eac0f90151d359.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/images/70e107b965392ddfeea312f901f649ff97e2b88687a45d7d62eac0f90151d359.jpg)

![78981e1d0fb12f8796e732c89ccac11d91316b52cce1d973ff60cd20908cb43b.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/images/78981e1d0fb12f8796e732c89ccac11d91316b52cce1d973ff60cd20908cb43b.jpg)

![a90c03eccb5e0be059f2602018620deced70355b7d880704537315fc842e1a3a.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/images/a90c03eccb5e0be059f2602018620deced70355b7d880704537315fc842e1a3a.jpg)

![b9ccdc1ac6188a02726be73f0282e0e3b6c4b29686c8189c7380437027b2edca.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/images/b9ccdc1ac6188a02726be73f0282e0e3b6c4b29686c8189c7380437027b2edca.jpg)

![b9e22ee210cbba180e2e59a30df50b1145b058c3b83c70b80fea74218de372fb.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/images/b9e22ee210cbba180e2e59a30df50b1145b058c3b83c70b80fea74218de372fb.jpg)

![d9da81d2cf88b755e185e673241ad725f964b71479af54914285ca7313d66694.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/images/d9da81d2cf88b755e185e673241ad725f964b71479af54914285ca7313d66694.jpg)

![ed39714b7d05052ebfecd284c51564a7dd3d82c56fad71dcd94b02a12f8e09f9.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/images/ed39714b7d05052ebfecd284c51564a7dd3d82c56fad71dcd94b02a12f8e09f9.jpg)

### Tables

![071dacd67d193c1a4b8928041ae67f4757cb9e7a5a6cd34d23fb9cb0cce9384a.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/tables/071dacd67d193c1a4b8928041ae67f4757cb9e7a5a6cd34d23fb9cb0cce9384a.jpg)

![0ef53128b4e58e3124f0b5c7374bf57ca00d285a3427c7ea1435252ccd16becb.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/tables/0ef53128b4e58e3124f0b5c7374bf57ca00d285a3427c7ea1435252ccd16becb.jpg)

![1686c000809ec6a9a222df5d1214f4ca5c2930a555c7e8fcd58a3aa0a270121a.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/tables/1686c000809ec6a9a222df5d1214f4ca5c2930a555c7e8fcd58a3aa0a270121a.jpg)

![209cc2d2f6232217a597ad86e2cd17dba71f489f7506229dc193e5a111aba0be.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/tables/209cc2d2f6232217a597ad86e2cd17dba71f489f7506229dc193e5a111aba0be.jpg)

![2538632f7858f18d44ded722603817a9cf3b8ff7eef6a7b5b6c69fc517fa86b1.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/tables/2538632f7858f18d44ded722603817a9cf3b8ff7eef6a7b5b6c69fc517fa86b1.jpg)

![27a74a60c535b215ff90b2e34e096c5cc8047468330fd0bc77ed5fd6ee8e13d9.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/tables/27a74a60c535b215ff90b2e34e096c5cc8047468330fd0bc77ed5fd6ee8e13d9.jpg)

![36be5724caa7db5a51e000e2b2130aea122344adff75c41823980474c72b3bde.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/tables/36be5724caa7db5a51e000e2b2130aea122344adff75c41823980474c72b3bde.jpg)

![490653b5173d6a08d3321b9ee245951e44fb138e3c973aeecca28c3ade7cd13f.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/tables/490653b5173d6a08d3321b9ee245951e44fb138e3c973aeecca28c3ade7cd13f.jpg)

![519f1c850ee9d53ce6681b0aa8354eb968f88d0f68a5e1edb58ef28e37df12fe.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/tables/519f1c850ee9d53ce6681b0aa8354eb968f88d0f68a5e1edb58ef28e37df12fe.jpg)

![52c487995ea901c078ef9db3f864cc0bc0d93952a519d78854af2546e40c3921.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/tables/52c487995ea901c078ef9db3f864cc0bc0d93952a519d78854af2546e40c3921.jpg)

![59811d367435868e12b2411d82c79c72b9cb2ef4a3b7d81e92c07d6b21adac5a.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/tables/59811d367435868e12b2411d82c79c72b9cb2ef4a3b7d81e92c07d6b21adac5a.jpg)

![5fbbac975d03f70c45b8abf979e320652a2fa7525c8e68dcd61abc2604686fe2.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/tables/5fbbac975d03f70c45b8abf979e320652a2fa7525c8e68dcd61abc2604686fe2.jpg)

![6327a6065955f2d5d7b7b525679b669bee7e3c85796535cb342a46e99fd58625.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/tables/6327a6065955f2d5d7b7b525679b669bee7e3c85796535cb342a46e99fd58625.jpg)

![7bad01c64c657ce021a22301eb3e5e73141ba9f9d397d756f5dc569b45f5d98b.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/tables/7bad01c64c657ce021a22301eb3e5e73141ba9f9d397d756f5dc569b45f5d98b.jpg)

![9a558877d2e0054da30f6ad7cb8c9be83ec83a2da02ae2cb73d9551acbf710da.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/tables/9a558877d2e0054da30f6ad7cb8c9be83ec83a2da02ae2cb73d9551acbf710da.jpg)

![a9dd01dcba94ed41395f47eaad12843b1547f001b6f8c45c0a3fd453ea09301c.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/tables/a9dd01dcba94ed41395f47eaad12843b1547f001b6f8c45c0a3fd453ea09301c.jpg)

![ac5a7b77fe84bcf9e29ce48057cd1d1f1b5ae1440d799657e711c7a0c513126a.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/tables/ac5a7b77fe84bcf9e29ce48057cd1d1f1b5ae1440d799657e711c7a0c513126a.jpg)

![b5962e600822e99ba637bd15614177a6ee12455e786e3bc1603c8797697e012e.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/tables/b5962e600822e99ba637bd15614177a6ee12455e786e3bc1603c8797697e012e.jpg)

![ccaae43dae1fb35302e928762828e1f28fb6a254155363abf60912a61ebae0dc.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/tables/ccaae43dae1fb35302e928762828e1f28fb6a254155363abf60912a61ebae0dc.jpg)

![dbf469fbc4da83103482031f27b673b464f98caf6476aa9fe9b6ca0fc319de64.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/tables/dbf469fbc4da83103482031f27b673b464f98caf6476aa9fe9b6ca0fc319de64.jpg)

![de9f71c42c7ff67ae04589c8a9dc7b4be0a5e6c7af7da9447ab198e11522f6d6.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/tables/de9f71c42c7ff67ae04589c8a9dc7b4be0a5e6c7af7da9447ab198e11522f6d6.jpg)

![e396d79d86e48705bc1de6a0d27d43bd6cd9f6465a2a031311cdbecd2c149c23.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/tables/e396d79d86e48705bc1de6a0d27d43bd6cd9f6465a2a031311cdbecd2c149c23.jpg)

![f014cc37eeed343a5b426e4b394a3c4c2da46f6b57bfda3d952627471f81327c.jpg](../iclr_results/1303_ParetoFlow_ Guided Flows in Multi-Objective Optimization/tables/f014cc37eeed343a5b426e4b394a3c4c2da46f6b57bfda3d952627471f81327c.jpg)

## GOFA: A Generative One-For-All Model for Joint Graph Language Modeling


### Images

![1e1c00bc4173e99111697e34f19ff9e20ef15683721709f69194e91fbe170916.jpg](../iclr_results/1304_GOFA_ A Generative One-For-All Model for Joint Graph Language Modeling/images/1e1c00bc4173e99111697e34f19ff9e20ef15683721709f69194e91fbe170916.jpg)

![31b6a7b3a78f5fed7a89ec0f9cb98117e2451e78c567988d8e392413dd57a2e2.jpg](../iclr_results/1304_GOFA_ A Generative One-For-All Model for Joint Graph Language Modeling/images/31b6a7b3a78f5fed7a89ec0f9cb98117e2451e78c567988d8e392413dd57a2e2.jpg)

![336b5c997aa1729cbeb7c58be2ef8a950dbb3b878ff69b95bc9e3ba0e03079d1.jpg](../iclr_results/1304_GOFA_ A Generative One-For-All Model for Joint Graph Language Modeling/images/336b5c997aa1729cbeb7c58be2ef8a950dbb3b878ff69b95bc9e3ba0e03079d1.jpg)

![4912047720fbde9f8ca2028944e5518852d717d58c63ac1707d18c70ed10ab1a.jpg](../iclr_results/1304_GOFA_ A Generative One-For-All Model for Joint Graph Language Modeling/images/4912047720fbde9f8ca2028944e5518852d717d58c63ac1707d18c70ed10ab1a.jpg)

![811a05c99192dd397eea26d6dab1fcb493633987c0abb19247a1a07d560c3c28.jpg](../iclr_results/1304_GOFA_ A Generative One-For-All Model for Joint Graph Language Modeling/images/811a05c99192dd397eea26d6dab1fcb493633987c0abb19247a1a07d560c3c28.jpg)

![c51baede71eea8a945f9ba9ed5e314201bb099ee9d415e046c32d9f3e6813cbc.jpg](../iclr_results/1304_GOFA_ A Generative One-For-All Model for Joint Graph Language Modeling/images/c51baede71eea8a945f9ba9ed5e314201bb099ee9d415e046c32d9f3e6813cbc.jpg)

### Tables

![19ae08579b26a1fe63c714f66d140a0d8d84844f0a20d2fc0b06174083e111e7.jpg](../iclr_results/1304_GOFA_ A Generative One-For-All Model for Joint Graph Language Modeling/tables/19ae08579b26a1fe63c714f66d140a0d8d84844f0a20d2fc0b06174083e111e7.jpg)

![29944fa8648a671c0343dfc6cae27008bf8a48af4784f85c259430b05c5a7304.jpg](../iclr_results/1304_GOFA_ A Generative One-For-All Model for Joint Graph Language Modeling/tables/29944fa8648a671c0343dfc6cae27008bf8a48af4784f85c259430b05c5a7304.jpg)

![2bb7dc419f6231a28561be4d420ad3b2e25771054d42520d319558b27b82c8e3.jpg](../iclr_results/1304_GOFA_ A Generative One-For-All Model for Joint Graph Language Modeling/tables/2bb7dc419f6231a28561be4d420ad3b2e25771054d42520d319558b27b82c8e3.jpg)

![36fc68b3dfabb4f5c22f8d85261b301a203077f041fa9c588492ac64cbfd50dd.jpg](../iclr_results/1304_GOFA_ A Generative One-For-All Model for Joint Graph Language Modeling/tables/36fc68b3dfabb4f5c22f8d85261b301a203077f041fa9c588492ac64cbfd50dd.jpg)

![4759ad09e550bb80a81e67c6908cfee36eb4b792009f2ba4a78f25b98995b6b6.jpg](../iclr_results/1304_GOFA_ A Generative One-For-All Model for Joint Graph Language Modeling/tables/4759ad09e550bb80a81e67c6908cfee36eb4b792009f2ba4a78f25b98995b6b6.jpg)

![5b542ef416cbf28e1f39f58723930a55cfdf88be31a78e2966ac8b2de46d0081.jpg](../iclr_results/1304_GOFA_ A Generative One-For-All Model for Joint Graph Language Modeling/tables/5b542ef416cbf28e1f39f58723930a55cfdf88be31a78e2966ac8b2de46d0081.jpg)

![5bcd76b0eaab9f2be77a0562e5ceef5c992f6d182748d846f5e54567ee659dc7.jpg](../iclr_results/1304_GOFA_ A Generative One-For-All Model for Joint Graph Language Modeling/tables/5bcd76b0eaab9f2be77a0562e5ceef5c992f6d182748d846f5e54567ee659dc7.jpg)

![5f2de0a2853e6ad1cebcedcc5a712ca8dd8d0b495d5bc2fd341af738989cf50d.jpg](../iclr_results/1304_GOFA_ A Generative One-For-All Model for Joint Graph Language Modeling/tables/5f2de0a2853e6ad1cebcedcc5a712ca8dd8d0b495d5bc2fd341af738989cf50d.jpg)

![6b85459557ae217a36be73c4cc9d737606ee65116696df5f65cb509e2fc6c982.jpg](../iclr_results/1304_GOFA_ A Generative One-For-All Model for Joint Graph Language Modeling/tables/6b85459557ae217a36be73c4cc9d737606ee65116696df5f65cb509e2fc6c982.jpg)

![740eaeb7896a686befb3c67068c5e4d32aa8f7da82ce0285a4bf1c59a46f89e2.jpg](../iclr_results/1304_GOFA_ A Generative One-For-All Model for Joint Graph Language Modeling/tables/740eaeb7896a686befb3c67068c5e4d32aa8f7da82ce0285a4bf1c59a46f89e2.jpg)

![7ba597062c9e30e9e17f584e4624b9a99d29dfe6a248dea76f401b35ac43545b.jpg](../iclr_results/1304_GOFA_ A Generative One-For-All Model for Joint Graph Language Modeling/tables/7ba597062c9e30e9e17f584e4624b9a99d29dfe6a248dea76f401b35ac43545b.jpg)

![8f4fdaab2fe1068daed60e6bc0f7f2d1b0419a6add7950ea9919687a9c8b4ee3.jpg](../iclr_results/1304_GOFA_ A Generative One-For-All Model for Joint Graph Language Modeling/tables/8f4fdaab2fe1068daed60e6bc0f7f2d1b0419a6add7950ea9919687a9c8b4ee3.jpg)

![a7612b46849fcb2dae0066ef53e0a2ac810b67a2815cb3db4dffaadcbcb66774.jpg](../iclr_results/1304_GOFA_ A Generative One-For-All Model for Joint Graph Language Modeling/tables/a7612b46849fcb2dae0066ef53e0a2ac810b67a2815cb3db4dffaadcbcb66774.jpg)

![cf7e07256d2684983b57ff7808c8379e6dd337f3ad62f916bd1ba6ed57b3a731.jpg](../iclr_results/1304_GOFA_ A Generative One-For-All Model for Joint Graph Language Modeling/tables/cf7e07256d2684983b57ff7808c8379e6dd337f3ad62f916bd1ba6ed57b3a731.jpg)

![e5aba3dc643acb04a78212d6ad14b8ca2af97ec3160a42556979880e833d93a2.jpg](../iclr_results/1304_GOFA_ A Generative One-For-All Model for Joint Graph Language Modeling/tables/e5aba3dc643acb04a78212d6ad14b8ca2af97ec3160a42556979880e833d93a2.jpg)

![ef874eaa3c1255f5a9ee5613614457b9713b3d769a6ed6ccf5132a3dd350cb28.jpg](../iclr_results/1304_GOFA_ A Generative One-For-All Model for Joint Graph Language Modeling/tables/ef874eaa3c1255f5a9ee5613614457b9713b3d769a6ed6ccf5132a3dd350cb28.jpg)

![f74481b42d1699ee6ba40eaeeb1694b63cd32f7b667678197d17a30e28cfe78e.jpg](../iclr_results/1304_GOFA_ A Generative One-For-All Model for Joint Graph Language Modeling/tables/f74481b42d1699ee6ba40eaeeb1694b63cd32f7b667678197d17a30e28cfe78e.jpg)

## Apollo-MILP: An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear Programming


### Images

![734ef92171ce2c4e4ea741e4d388cbabac191a6186a6b6b1a48d473aa5d67ffc.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /images/734ef92171ce2c4e4ea741e4d388cbabac191a6186a6b6b1a48d473aa5d67ffc.jpg)

![87907f274c0a31b00dbc5d19a49bd4a1870d2f5dcc2af02c4f06561d543da565.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /images/87907f274c0a31b00dbc5d19a49bd4a1870d2f5dcc2af02c4f06561d543da565.jpg)

![b8e4a4725431b6ea73153e8731b3102e46cf3075344592ab94bdce61761227e1.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /images/b8e4a4725431b6ea73153e8731b3102e46cf3075344592ab94bdce61761227e1.jpg)

![cc5b19687165d085a79567874f27e5104c3abd3eb997eb8f36c9bd5d5f9091be.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /images/cc5b19687165d085a79567874f27e5104c3abd3eb997eb8f36c9bd5d5f9091be.jpg)

![f0d88c718f68ff86be53a25221f7c610e01945c8b2a2c27a277c71568747f763.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /images/f0d88c718f68ff86be53a25221f7c610e01945c8b2a2c27a277c71568747f763.jpg)

### Tables

![11e9205a7db1b199c967c596110ca7d1011efd672ae44e788aebb4878ab4c9d5.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /tables/11e9205a7db1b199c967c596110ca7d1011efd672ae44e788aebb4878ab4c9d5.jpg)

![1656781f8f5133c11543ecc9e5a50d0a07de58bbf244674804415e16f2f57e17.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /tables/1656781f8f5133c11543ecc9e5a50d0a07de58bbf244674804415e16f2f57e17.jpg)

![16897700eebf29aa6563b698278bc7927a7020de1482a511af5caef01ec278d9.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /tables/16897700eebf29aa6563b698278bc7927a7020de1482a511af5caef01ec278d9.jpg)

![180414af90c677247ad32b58e924b43360b9738691a21f4b1d39c4be35a387f7.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /tables/180414af90c677247ad32b58e924b43360b9738691a21f4b1d39c4be35a387f7.jpg)

![1f07fec8a20215d9fed6344055587e244f7f9548c3e6d51815a519880444791d.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /tables/1f07fec8a20215d9fed6344055587e244f7f9548c3e6d51815a519880444791d.jpg)

![1fbc1d18ab68d34d2044c8ae1169478b786c1e47a346b39163b24f4d05519226.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /tables/1fbc1d18ab68d34d2044c8ae1169478b786c1e47a346b39163b24f4d05519226.jpg)

![3a52d7b8e12c221ac5fc6e4a2e38375cc783f8f10e61ad6ab6817702612cfc09.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /tables/3a52d7b8e12c221ac5fc6e4a2e38375cc783f8f10e61ad6ab6817702612cfc09.jpg)

![483de413c7687ae8b315c2bc1385ab28482ac8cd3c7673f54f27a2003fd07105.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /tables/483de413c7687ae8b315c2bc1385ab28482ac8cd3c7673f54f27a2003fd07105.jpg)

![5d6e442c10ce9c034135f5f949578fe6835631f5856eae932f6446da044a8d18.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /tables/5d6e442c10ce9c034135f5f949578fe6835631f5856eae932f6446da044a8d18.jpg)

![6eaee0ac8d45d91e811bc08140209684aeaf7b6cb173f2900bb97fc5742b172e.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /tables/6eaee0ac8d45d91e811bc08140209684aeaf7b6cb173f2900bb97fc5742b172e.jpg)

![7cee12f35f013827f1ac91c57f07f4955d3d526c09804401f53444b3d20627fc.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /tables/7cee12f35f013827f1ac91c57f07f4955d3d526c09804401f53444b3d20627fc.jpg)

![96be3d89e6fd190c2a7ec4a909ad40a11a9575db987a561c9c61c90a5f8feef0.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /tables/96be3d89e6fd190c2a7ec4a909ad40a11a9575db987a561c9c61c90a5f8feef0.jpg)

![9e2e9152f4e136bb7c70ca6788bab97626ce47acc99cbfee1579bdb70c2b5766.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /tables/9e2e9152f4e136bb7c70ca6788bab97626ce47acc99cbfee1579bdb70c2b5766.jpg)

![a689d8d7928c1757fbc4b75aa8c76fd1feaf286ccc339da3b3ec8a2b83e65de9.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /tables/a689d8d7928c1757fbc4b75aa8c76fd1feaf286ccc339da3b3ec8a2b83e65de9.jpg)

![b00b1b3dded50637f00933e7af1854c25821955d185be3bca8d0ae3c48a988e8.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /tables/b00b1b3dded50637f00933e7af1854c25821955d185be3bca8d0ae3c48a988e8.jpg)

![b6b35ce0f0e96da370fe198cfd4c263ba87e56f002c8a4ba4ef9b79a0e1f2479.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /tables/b6b35ce0f0e96da370fe198cfd4c263ba87e56f002c8a4ba4ef9b79a0e1f2479.jpg)

![cb833f431635cfe21f5e5c8245b4940c7842cf3ff032edc8d95c0f118e9509bc.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /tables/cb833f431635cfe21f5e5c8245b4940c7842cf3ff032edc8d95c0f118e9509bc.jpg)

![f99d23806aed26225f5f2e6ae109f5f7d87ba447e6d64eae5d94306f3c846e52.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /tables/f99d23806aed26225f5f2e6ae109f5f7d87ba447e6d64eae5d94306f3c846e52.jpg)

![fb399b2342274495ccc8ae9b35060418823410fd68c94e8be4981adbfdcb150e.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /tables/fb399b2342274495ccc8ae9b35060418823410fd68c94e8be4981adbfdcb150e.jpg)

![fe2718b2169884ecfcb7df2b0a90bb9974e72f9a0993c63169df722a9f7d0dee.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /tables/fe2718b2169884ecfcb7df2b0a90bb9974e72f9a0993c63169df722a9f7d0dee.jpg)

![ff62a36ebe6468dca1cbe805dd11cb23a324fa522abfdfd9683f82e2213b37b0.jpg](../iclr_results/1305_Apollo-MILP_ An Alternating Prediction-Correction Neural Solving Framework for Mixed-Integer Linear /tables/ff62a36ebe6468dca1cbe805dd11cb23a324fa522abfdfd9683f82e2213b37b0.jpg)

## Diffusion Policy Policy Optimization


### Images

![00cafb69a8c6f879b752c88bde167616c40a3b954f3f2c6583e6a46e8d426ad8.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/00cafb69a8c6f879b752c88bde167616c40a3b954f3f2c6583e6a46e8d426ad8.jpg)

![0c8ddf4bbd9a91a6321a02d2d1f2d591a274793e38a62f1d9e7e81fe24303f39.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/0c8ddf4bbd9a91a6321a02d2d1f2d591a274793e38a62f1d9e7e81fe24303f39.jpg)

![14d237cd4183386aa060f318e95de20128b225ad64ebe4f668e5b00e1b9718c1.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/14d237cd4183386aa060f318e95de20128b225ad64ebe4f668e5b00e1b9718c1.jpg)

![15e32d55c2d34c910a596a6d0bc574f0584f9108b57630bb8dc602fb58dbca76.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/15e32d55c2d34c910a596a6d0bc574f0584f9108b57630bb8dc602fb58dbca76.jpg)

![1a189d35f9dba6a9aa4ddfdc69fde877ae0aa74bf98307c1391106c391a05e93.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/1a189d35f9dba6a9aa4ddfdc69fde877ae0aa74bf98307c1391106c391a05e93.jpg)

![1c67cd7860ce9057aad7743b3f8237986e7311fa9b709a28f5364da52993c7c8.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/1c67cd7860ce9057aad7743b3f8237986e7311fa9b709a28f5364da52993c7c8.jpg)

![204a8dcc8e9bdd0663fc7d5398844a788ec3cc6b1efe279a80fab98f8d9367d7.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/204a8dcc8e9bdd0663fc7d5398844a788ec3cc6b1efe279a80fab98f8d9367d7.jpg)

![2fd03323e257075ce703aa82fbfe27e1fe433e1f1616f78e1d613a860a50d01f.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/2fd03323e257075ce703aa82fbfe27e1fe433e1f1616f78e1d613a860a50d01f.jpg)

![3746240ec6994377ee25166e70ec01b913a0603deeadf7a4476e52353f8eee7e.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/3746240ec6994377ee25166e70ec01b913a0603deeadf7a4476e52353f8eee7e.jpg)

![4074ce081fe7bbaeaa93006b0875743fdf9e1fb225d415623ffc2bd08e86dd5c.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/4074ce081fe7bbaeaa93006b0875743fdf9e1fb225d415623ffc2bd08e86dd5c.jpg)

![436b943e45d08a7bd9d1b615d4f339e6490e186cd9ff4cc7b142b445903f50f4.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/436b943e45d08a7bd9d1b615d4f339e6490e186cd9ff4cc7b142b445903f50f4.jpg)

![58a9f365459b6a0710352414bfa1ad051b5ef6b7e1bf211f372c01db29881637.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/58a9f365459b6a0710352414bfa1ad051b5ef6b7e1bf211f372c01db29881637.jpg)

![63b9e2173b9fbec3355d340aff8eacf095830116e273db4b5e1d233a69f19a03.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/63b9e2173b9fbec3355d340aff8eacf095830116e273db4b5e1d233a69f19a03.jpg)

![654756c13a0c6d7e338962ac1ed9a57c7dd0ace1079af579dc4ffdec71249521.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/654756c13a0c6d7e338962ac1ed9a57c7dd0ace1079af579dc4ffdec71249521.jpg)

![7ad76b215fa30525092353cb582606b922da1ed1f75d50fd46582a897a617d57.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/7ad76b215fa30525092353cb582606b922da1ed1f75d50fd46582a897a617d57.jpg)

![875f5ba10f9f11cac3129aadd6d051350c16ade21372a9a4f8e29e179f5a2248.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/875f5ba10f9f11cac3129aadd6d051350c16ade21372a9a4f8e29e179f5a2248.jpg)

![949f999886a9d261cbb1b79f4bbca0867180d2a03c45aeacc671e1c8c84a380d.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/949f999886a9d261cbb1b79f4bbca0867180d2a03c45aeacc671e1c8c84a380d.jpg)

![9a57fc11f93f958bb79daa506d95a0fe26c38a02dac9e6da6df03ef4dea52973.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/9a57fc11f93f958bb79daa506d95a0fe26c38a02dac9e6da6df03ef4dea52973.jpg)

![a904a1f447193501c5df825c3cf68c28709ff994f1e12f313637dcc8f09cc24d.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/a904a1f447193501c5df825c3cf68c28709ff994f1e12f313637dcc8f09cc24d.jpg)

![ba4fa43f26f4ab6d7882707a0306c895b148bd149cf1233e2bcf01465a1133d1.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/ba4fa43f26f4ab6d7882707a0306c895b148bd149cf1233e2bcf01465a1133d1.jpg)

![c0222dc3c2aac0d3bb70478e01f14801104ce8a64f75408c31ead34ca24800dc.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/c0222dc3c2aac0d3bb70478e01f14801104ce8a64f75408c31ead34ca24800dc.jpg)

![c8348d72eb1c8f22302f783e335c8c9885ebcb00ef1ef8cc29be537d91e982be.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/c8348d72eb1c8f22302f783e335c8c9885ebcb00ef1ef8cc29be537d91e982be.jpg)

![d7f77b48817a72d6508a88cd2a38ffb104dc259ac5bd531ebba362c793b913cd.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/d7f77b48817a72d6508a88cd2a38ffb104dc259ac5bd531ebba362c793b913cd.jpg)

![fbcc8ea94b6478bd520da33f7b76c6a06181b8396a745d88cdd3b400cef1337b.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/images/fbcc8ea94b6478bd520da33f7b76c6a06181b8396a745d88cdd3b400cef1337b.jpg)

### Tables

![00579fbee1c78fb77e1fb97d58076f517598333beddebc32de318f7c7bace07c.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/tables/00579fbee1c78fb77e1fb97d58076f517598333beddebc32de318f7c7bace07c.jpg)

![2bc57b3c6e92ca34a13fa941e3b8ca03291fad021650aec48fab25864eb1e405.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/tables/2bc57b3c6e92ca34a13fa941e3b8ca03291fad021650aec48fab25864eb1e405.jpg)

![43dceb74200fedf0ba159508959a8e6d671e59ffa2b39f19aa477b57a98df938.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/tables/43dceb74200fedf0ba159508959a8e6d671e59ffa2b39f19aa477b57a98df938.jpg)

![48017a23e239750bd261dd68242782e223874db4b45e19fc86c73e3dfd30344e.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/tables/48017a23e239750bd261dd68242782e223874db4b45e19fc86c73e3dfd30344e.jpg)

![4f149022fcce7d795fb039e5eaab76154db5ddc034df822981beeef9c6000df0.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/tables/4f149022fcce7d795fb039e5eaab76154db5ddc034df822981beeef9c6000df0.jpg)

![713d240ca92036f1dc8a692a11ce9d97b78eb29afe3ef0ede45cbd4a3a56ec9b.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/tables/713d240ca92036f1dc8a692a11ce9d97b78eb29afe3ef0ede45cbd4a3a56ec9b.jpg)

![89c7a41ba93b9c5d217ba63bbe7e08a9092269860ce6933da034afbf6a22abb7.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/tables/89c7a41ba93b9c5d217ba63bbe7e08a9092269860ce6933da034afbf6a22abb7.jpg)

![a210200d3e352154dee5ce9415c6d4a05d28512f81d93823d2c0fe0daa2fdecc.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/tables/a210200d3e352154dee5ce9415c6d4a05d28512f81d93823d2c0fe0daa2fdecc.jpg)

![c845094f0abce3ce767613aff51ecf7eb83d7ff8b8b70af5eed7dc702bff72de.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/tables/c845094f0abce3ce767613aff51ecf7eb83d7ff8b8b70af5eed7dc702bff72de.jpg)

![d9953a1f218e7ac5a795fd79923e1e426f472cc2ea48ab33b71c9d465b8960e0.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/tables/d9953a1f218e7ac5a795fd79923e1e426f472cc2ea48ab33b71c9d465b8960e0.jpg)

![dd7807acb55f24dc4706f2280e4d0b86826b320f175975900d36ed4016a69569.jpg](../iclr_results/1306_Diffusion Policy Policy Optimization/tables/dd7807acb55f24dc4706f2280e4d0b86826b320f175975900d36ed4016a69569.jpg)

## Image Watermarks are Removable using Controllable Regeneration from Clean Noise


### Images

![1431313dad0678e8944902d762f1aae587b724f7c1e1b279c33c9d86432c216b.jpg](../iclr_results/1307_Image Watermarks are Removable using Controllable Regeneration from Clean Noise/images/1431313dad0678e8944902d762f1aae587b724f7c1e1b279c33c9d86432c216b.jpg)

![16b89b9ac5da0f0a0478065d3f5422f1dd955aa881ecffbd9be47c171bf825dc.jpg](../iclr_results/1307_Image Watermarks are Removable using Controllable Regeneration from Clean Noise/images/16b89b9ac5da0f0a0478065d3f5422f1dd955aa881ecffbd9be47c171bf825dc.jpg)

![16cbea7136e0d68b5f805fb05a3cff987e5f3214251113ec4d30911be6868ece.jpg](../iclr_results/1307_Image Watermarks are Removable using Controllable Regeneration from Clean Noise/images/16cbea7136e0d68b5f805fb05a3cff987e5f3214251113ec4d30911be6868ece.jpg)

![182b3eb81c000fe12ce4afaa69d3f62f8097e51d431d3f5a974d8e4ae1991ae5.jpg](../iclr_results/1307_Image Watermarks are Removable using Controllable Regeneration from Clean Noise/images/182b3eb81c000fe12ce4afaa69d3f62f8097e51d431d3f5a974d8e4ae1991ae5.jpg)

![33326aa5820a2de55c9b2e91127809a6facea70e0b62eb72d104747360199118.jpg](../iclr_results/1307_Image Watermarks are Removable using Controllable Regeneration from Clean Noise/images/33326aa5820a2de55c9b2e91127809a6facea70e0b62eb72d104747360199118.jpg)

![750d352b144b0643bf36839f67f81755581d800d76eb92aa1dc00a1c202fbb5c.jpg](../iclr_results/1307_Image Watermarks are Removable using Controllable Regeneration from Clean Noise/images/750d352b144b0643bf36839f67f81755581d800d76eb92aa1dc00a1c202fbb5c.jpg)

![870b2f07edec41ad6642ae75e58d03d4f23b3f0265ec5ccf531374825d27940e.jpg](../iclr_results/1307_Image Watermarks are Removable using Controllable Regeneration from Clean Noise/images/870b2f07edec41ad6642ae75e58d03d4f23b3f0265ec5ccf531374825d27940e.jpg)

![b29a040ab6921a0c21d10222547b925fb95332e4bf18bd55c26d3781d00d8ae8.jpg](../iclr_results/1307_Image Watermarks are Removable using Controllable Regeneration from Clean Noise/images/b29a040ab6921a0c21d10222547b925fb95332e4bf18bd55c26d3781d00d8ae8.jpg)

### Tables

![2c64b4e4ea6fd8b2fb01cef9257d28a4e4e7a732379b45bf89fa67dc242ecd4c.jpg](../iclr_results/1307_Image Watermarks are Removable using Controllable Regeneration from Clean Noise/tables/2c64b4e4ea6fd8b2fb01cef9257d28a4e4e7a732379b45bf89fa67dc242ecd4c.jpg)

![71e9d78286f0080f1b35017cd0353f28e41d654d4de6e72f2cfcb24491bda686.jpg](../iclr_results/1307_Image Watermarks are Removable using Controllable Regeneration from Clean Noise/tables/71e9d78286f0080f1b35017cd0353f28e41d654d4de6e72f2cfcb24491bda686.jpg)

![8f9cb94662861f8bd354ab083daf7c8694515b713d4dccc01211dcf2e24a2ccd.jpg](../iclr_results/1307_Image Watermarks are Removable using Controllable Regeneration from Clean Noise/tables/8f9cb94662861f8bd354ab083daf7c8694515b713d4dccc01211dcf2e24a2ccd.jpg)

![a924973cbf854b6dffe3bfb35f381e8877615b2a653ead5490ab142492a24a18.jpg](../iclr_results/1307_Image Watermarks are Removable using Controllable Regeneration from Clean Noise/tables/a924973cbf854b6dffe3bfb35f381e8877615b2a653ead5490ab142492a24a18.jpg)

## Coreset Selection via Reducible Loss in Continual Learning


### Images

![122070cc608be9924b83ddd761ea046dc0646377cae93b872c4e42aaa95df595.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/images/122070cc608be9924b83ddd761ea046dc0646377cae93b872c4e42aaa95df595.jpg)

![141c0a65aa7af6258d5580ad8ed7904e667a6217e5ba76172f84da20560b2980.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/images/141c0a65aa7af6258d5580ad8ed7904e667a6217e5ba76172f84da20560b2980.jpg)

![24b858ad1a0972a8cc60bece8017df6b42d0b220469353f57355a46145a9532e.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/images/24b858ad1a0972a8cc60bece8017df6b42d0b220469353f57355a46145a9532e.jpg)

![3ec13b12388df9a2b78a4d5b0b0d2e0b8a548f31deb561c44ff28b99e3ab6344.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/images/3ec13b12388df9a2b78a4d5b0b0d2e0b8a548f31deb561c44ff28b99e3ab6344.jpg)

![43cfe6a474fc206cc4eb90fbaae4e2f45cfaec24164cfbcd10bef15807b69b36.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/images/43cfe6a474fc206cc4eb90fbaae4e2f45cfaec24164cfbcd10bef15807b69b36.jpg)

![48527067d7bcd8e0c3d8153cb96750408ffc100174ce590397db57d8e884cc6b.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/images/48527067d7bcd8e0c3d8153cb96750408ffc100174ce590397db57d8e884cc6b.jpg)

![4f9dfa2505f140f3c51a3bbca6ca53c13b9e5590af28407fec0152e667876c20.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/images/4f9dfa2505f140f3c51a3bbca6ca53c13b9e5590af28407fec0152e667876c20.jpg)

![509a0bff4af964098725e430c789ffac9ef83947731f96a3060a4c9b9b505d59.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/images/509a0bff4af964098725e430c789ffac9ef83947731f96a3060a4c9b9b505d59.jpg)

![5b1497171c7441aeae204c0790a289d8b6eef787d113e3b257302ef30c2946bc.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/images/5b1497171c7441aeae204c0790a289d8b6eef787d113e3b257302ef30c2946bc.jpg)

![5f5937010423eb65ddc615af7464c139daebb70b70a363f0902552911dab1cef.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/images/5f5937010423eb65ddc615af7464c139daebb70b70a363f0902552911dab1cef.jpg)

![62510e0f2e232f7cf0af69577a952c6555d05a1bd5f8c1b8d066bc45571cef5f.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/images/62510e0f2e232f7cf0af69577a952c6555d05a1bd5f8c1b8d066bc45571cef5f.jpg)

![75d598be62536a7591384b1ebe829bac5caa1e039d53ef2f35cd5237b60081fa.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/images/75d598be62536a7591384b1ebe829bac5caa1e039d53ef2f35cd5237b60081fa.jpg)

![7fcba44405bc2fdfa6664a8210e1c85c8af6ab9f40993f47647d3929628680f9.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/images/7fcba44405bc2fdfa6664a8210e1c85c8af6ab9f40993f47647d3929628680f9.jpg)

![8f9e4c974a7306daf5f728d128cc011ab26c0cd5d604909b64f788160829872f.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/images/8f9e4c974a7306daf5f728d128cc011ab26c0cd5d604909b64f788160829872f.jpg)

![a403e5a5790ef834b5bf9db7c936e0e786eb54605a287007c8863142f5544f1d.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/images/a403e5a5790ef834b5bf9db7c936e0e786eb54605a287007c8863142f5544f1d.jpg)

![af2d2edda6fbfc92a1b5d931fb8ffb94c98ec0d751baef2714b9bdb6e1678f70.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/images/af2d2edda6fbfc92a1b5d931fb8ffb94c98ec0d751baef2714b9bdb6e1678f70.jpg)

![b5f0b55eeff3e4848358339f09277ec070186955d656a35a44c8be456d8beda6.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/images/b5f0b55eeff3e4848358339f09277ec070186955d656a35a44c8be456d8beda6.jpg)

![d54c7afd4b96d36e2cd883f46aa328cb017fe026406f6f95c77a95361fd31555.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/images/d54c7afd4b96d36e2cd883f46aa328cb017fe026406f6f95c77a95361fd31555.jpg)

![db2b72e690a6bc0a344981e7860470688af0ccc1272a5b835706567ca5776f40.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/images/db2b72e690a6bc0a344981e7860470688af0ccc1272a5b835706567ca5776f40.jpg)

![dfccf96214b5dc04b69c967055f3ff5aead12693e215b209005d9fac51adcac6.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/images/dfccf96214b5dc04b69c967055f3ff5aead12693e215b209005d9fac51adcac6.jpg)

![eb9aa3495d4e4ff2e8c1943b5161cf797da03410fa3e14a94d336ca03ed9c3bd.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/images/eb9aa3495d4e4ff2e8c1943b5161cf797da03410fa3e14a94d336ca03ed9c3bd.jpg)

![f1925756fcb7a7fc04405847e0cb5a3d5097e54a8f3c6887c81885a298d5c81e.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/images/f1925756fcb7a7fc04405847e0cb5a3d5097e54a8f3c6887c81885a298d5c81e.jpg)

### Tables

![26e9a68d9e0fc487a66c4f3478e89232049264cd321432b7fed6f81c9a5ab72d.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/tables/26e9a68d9e0fc487a66c4f3478e89232049264cd321432b7fed6f81c9a5ab72d.jpg)

![46d42d44d67d9de7cacdcab8f7eba7b19281ae9fda99e1a6422a7167d4d72818.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/tables/46d42d44d67d9de7cacdcab8f7eba7b19281ae9fda99e1a6422a7167d4d72818.jpg)

![4af1fd5d572129ba43b7d9681493c8b946bb90cd0bea6d28550ed7539c334f9f.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/tables/4af1fd5d572129ba43b7d9681493c8b946bb90cd0bea6d28550ed7539c334f9f.jpg)

![54b32444fd2e41ada3a8842d7bae917bc7e4e75411dd38fc4c1d5aef96e8ede2.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/tables/54b32444fd2e41ada3a8842d7bae917bc7e4e75411dd38fc4c1d5aef96e8ede2.jpg)

![5818db1bad67cad980b038a088d0f7c407eb3b4fbde7e8d5d21b97fc5eb8a8e7.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/tables/5818db1bad67cad980b038a088d0f7c407eb3b4fbde7e8d5d21b97fc5eb8a8e7.jpg)

![60f318eb8b6d65b6ca94ed4f8d4d5af5f005d534dbd9ae02e8555906d04ac54b.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/tables/60f318eb8b6d65b6ca94ed4f8d4d5af5f005d534dbd9ae02e8555906d04ac54b.jpg)

![66ac85ded92cad2a651af07477dc6b9471ae3a9bb88697ba282e4e465cf01ddb.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/tables/66ac85ded92cad2a651af07477dc6b9471ae3a9bb88697ba282e4e465cf01ddb.jpg)

![672f810245deb27f3d2db5148a15ec3e0334fd13cfb29dcfff52a86610e45724.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/tables/672f810245deb27f3d2db5148a15ec3e0334fd13cfb29dcfff52a86610e45724.jpg)

![9252308f0fef62e0dd1520e0529f02ce3cf7bb2f6393d0ad13b7e91120382fbd.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/tables/9252308f0fef62e0dd1520e0529f02ce3cf7bb2f6393d0ad13b7e91120382fbd.jpg)

![9b649834f215d95f7866cfca7a524b49e68c3b6920f97d11244993d94cc28065.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/tables/9b649834f215d95f7866cfca7a524b49e68c3b6920f97d11244993d94cc28065.jpg)

![9d57941a6077bd8256df4f4f2a2bb859f356b4cbec851e694d5b760cbe2f5828.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/tables/9d57941a6077bd8256df4f4f2a2bb859f356b4cbec851e694d5b760cbe2f5828.jpg)

![a21265f4d88a6459d236343b4e01986e91da869941c168f3e170b1f1bd06f51c.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/tables/a21265f4d88a6459d236343b4e01986e91da869941c168f3e170b1f1bd06f51c.jpg)

![aa5cf361b5db2253db8de49454ec7f2b87004efbb8b292a25785459e24623468.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/tables/aa5cf361b5db2253db8de49454ec7f2b87004efbb8b292a25785459e24623468.jpg)

![c423d524850f0709ed37dbf2a9224e9f9070a926959a10215efe1fcfae4bcd98.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/tables/c423d524850f0709ed37dbf2a9224e9f9070a926959a10215efe1fcfae4bcd98.jpg)

![ef4ee91a17e6f12d9650872a7e5008e7faa28e322904c6c9ee0b0fa8e45d5c7b.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/tables/ef4ee91a17e6f12d9650872a7e5008e7faa28e322904c6c9ee0b0fa8e45d5c7b.jpg)

![f116fe37fd693771b5ed817ef7391ee25a4046f7958dbd3bcb64cb21267a6726.jpg](../iclr_results/1308_Coreset Selection via Reducible Loss in Continual Learning/tables/f116fe37fd693771b5ed817ef7391ee25a4046f7958dbd3bcb64cb21267a6726.jpg)

## Efficient Distribution Matching of Representations via Noise-Injected Deep InfoMax


### Images

![12a5f92a03fa6693591530894dd5629b0c15dba24310f310b60660bb545b78b5.jpg](../iclr_results/1309_Efficient Distribution Matching of Representations via Noise-Injected Deep InfoMax/images/12a5f92a03fa6693591530894dd5629b0c15dba24310f310b60660bb545b78b5.jpg)

![35655b5a2821cbbe8f979fc66e32927c0356777360d23fbe955d1b26b694d08c.jpg](../iclr_results/1309_Efficient Distribution Matching of Representations via Noise-Injected Deep InfoMax/images/35655b5a2821cbbe8f979fc66e32927c0356777360d23fbe955d1b26b694d08c.jpg)

![3879386b984679ed1dde26ecd45c2c345d4eac6f4b70f337aadeea6f1aaa861c.jpg](../iclr_results/1309_Efficient Distribution Matching of Representations via Noise-Injected Deep InfoMax/images/3879386b984679ed1dde26ecd45c2c345d4eac6f4b70f337aadeea6f1aaa861c.jpg)

![396dafed0681013ab01a03c6f3fbda7c5cdcda84aaf8ae56fdc6b6befd4a0ca5.jpg](../iclr_results/1309_Efficient Distribution Matching of Representations via Noise-Injected Deep InfoMax/images/396dafed0681013ab01a03c6f3fbda7c5cdcda84aaf8ae56fdc6b6befd4a0ca5.jpg)

![a2d22e951e78661328a438df887de260282f9bbe8b147e4b2be7a6f6a2c00b96.jpg](../iclr_results/1309_Efficient Distribution Matching of Representations via Noise-Injected Deep InfoMax/images/a2d22e951e78661328a438df887de260282f9bbe8b147e4b2be7a6f6a2c00b96.jpg)

![c7206144ad21da31e6a51938891656156ce88e588bf48ed6fff74b770b30c93f.jpg](../iclr_results/1309_Efficient Distribution Matching of Representations via Noise-Injected Deep InfoMax/images/c7206144ad21da31e6a51938891656156ce88e588bf48ed6fff74b770b30c93f.jpg)

![e8d36240c8aca4a2d7a5f6df7d8af7fa3e195ed8756daac473dfd3abfe9a8acc.jpg](../iclr_results/1309_Efficient Distribution Matching of Representations via Noise-Injected Deep InfoMax/images/e8d36240c8aca4a2d7a5f6df7d8af7fa3e195ed8756daac473dfd3abfe9a8acc.jpg)

### Tables

![2c269d605bab356e6734e0213917e41f091d51e05c515ba7d1414583e152a2c3.jpg](../iclr_results/1309_Efficient Distribution Matching of Representations via Noise-Injected Deep InfoMax/tables/2c269d605bab356e6734e0213917e41f091d51e05c515ba7d1414583e152a2c3.jpg)

![6e15581d9ae3584a9bc57975748e1f20441d339bbfed9d1a8f736c1fbccb4ecd.jpg](../iclr_results/1309_Efficient Distribution Matching of Representations via Noise-Injected Deep InfoMax/tables/6e15581d9ae3584a9bc57975748e1f20441d339bbfed9d1a8f736c1fbccb4ecd.jpg)

![875bd1d5577b22f592e81cc5c2ae61bae55367493508a26eac46157a5fffa553.jpg](../iclr_results/1309_Efficient Distribution Matching of Representations via Noise-Injected Deep InfoMax/tables/875bd1d5577b22f592e81cc5c2ae61bae55367493508a26eac46157a5fffa553.jpg)

## Is Your Video Language Model a Reliable Judge?

### Images

![03ecd2a9e0e0059dd1112fd98f4e14444d9c7e99f057c098c6e2d48b043cd4d3.jpg](../iclr_results/1310_Is Your Video Language Model a Reliable Judge_/images/03ecd2a9e0e0059dd1112fd98f4e14444d9c7e99f057c098c6e2d48b043cd4d3.jpg)

![0df7be8f5c7fbe8f34ca518fe5c9c92cb758dfd1113401ce51c8aa0a2676f415.jpg](../iclr_results/1310_Is Your Video Language Model a Reliable Judge_/images/0df7be8f5c7fbe8f34ca518fe5c9c92cb758dfd1113401ce51c8aa0a2676f415.jpg)

![118faa5fed91c64d3f9643a13110a79e04e29976d9f8052d03958813959446b8.jpg](../iclr_results/1310_Is Your Video Language Model a Reliable Judge_/images/118faa5fed91c64d3f9643a13110a79e04e29976d9f8052d03958813959446b8.jpg)

![2b9ffe4c38f0bb5997a59a21aa7c1db46a57ceb7ded9b1351d6a64389fab38c9.jpg](../iclr_results/1310_Is Your Video Language Model a Reliable Judge_/images/2b9ffe4c38f0bb5997a59a21aa7c1db46a57ceb7ded9b1351d6a64389fab38c9.jpg)

![af83e226f37aab078aaacd268cf2a159b5c3d5939856026aabce15aed0d539b4.jpg](../iclr_results/1310_Is Your Video Language Model a Reliable Judge_/images/af83e226f37aab078aaacd268cf2a159b5c3d5939856026aabce15aed0d539b4.jpg)

![bbf22cad29a291e67ed52e0eb5acdbb7695758c6d29698fe2b97ff650db75259.jpg](../iclr_results/1310_Is Your Video Language Model a Reliable Judge_/images/bbf22cad29a291e67ed52e0eb5acdbb7695758c6d29698fe2b97ff650db75259.jpg)

### Tables

![3048b574e244392fc44fd4f2ce781619a36797d5f1042ce1331d339992db9c10.jpg](../iclr_results/1310_Is Your Video Language Model a Reliable Judge_/tables/3048b574e244392fc44fd4f2ce781619a36797d5f1042ce1331d339992db9c10.jpg)

![3f4099c88eb38ffb8302d26d287f7846b6c5ba0ca9a2111ecc114849cfda4a0f.jpg](../iclr_results/1310_Is Your Video Language Model a Reliable Judge_/tables/3f4099c88eb38ffb8302d26d287f7846b6c5ba0ca9a2111ecc114849cfda4a0f.jpg)

![89324d56cc919e11b64cfea66b3b1cf118c1d29eaad9902bf50b179a07d1efa4.jpg](../iclr_results/1310_Is Your Video Language Model a Reliable Judge_/tables/89324d56cc919e11b64cfea66b3b1cf118c1d29eaad9902bf50b179a07d1efa4.jpg)

![9f17e21205ee0c498e4a168bc595af653ebfdf83c9fe6646edc2a10877a2f06f.jpg](../iclr_results/1310_Is Your Video Language Model a Reliable Judge_/tables/9f17e21205ee0c498e4a168bc595af653ebfdf83c9fe6646edc2a10877a2f06f.jpg)

![a2f3f5da190a8e14fbd40483dd07b4005741c90940402df5d9d885d9eb996c10.jpg](../iclr_results/1310_Is Your Video Language Model a Reliable Judge_/tables/a2f3f5da190a8e14fbd40483dd07b4005741c90940402df5d9d885d9eb996c10.jpg)

![ae972d2776aadb8402f7806573d8e8da6ac9c86e029ff5b0193abf018a1567f5.jpg](../iclr_results/1310_Is Your Video Language Model a Reliable Judge_/tables/ae972d2776aadb8402f7806573d8e8da6ac9c86e029ff5b0193abf018a1567f5.jpg)

![b0a1d983a23263a4b212d1df9acef599fba3b0a92c387c22c0d2636479974314.jpg](../iclr_results/1310_Is Your Video Language Model a Reliable Judge_/tables/b0a1d983a23263a4b212d1df9acef599fba3b0a92c387c22c0d2636479974314.jpg)

![d363535807458844dc6a2fc42d458433e6616c2cfa08b9eb32869412ddca34e9.jpg](../iclr_results/1310_Is Your Video Language Model a Reliable Judge_/tables/d363535807458844dc6a2fc42d458433e6616c2cfa08b9eb32869412ddca34e9.jpg)

![eb114146bfadb56cedb3e96e7c22b6ab49ac8a2ce255404d69cedee7844f8b26.jpg](../iclr_results/1310_Is Your Video Language Model a Reliable Judge_/tables/eb114146bfadb56cedb3e96e7c22b6ab49ac8a2ce255404d69cedee7844f8b26.jpg)

![f889b3600533c42ccf7e2a0d55e927dfa02eb5fdc3afed651234e725b422fa56.jpg](../iclr_results/1310_Is Your Video Language Model a Reliable Judge_/tables/f889b3600533c42ccf7e2a0d55e927dfa02eb5fdc3afed651234e725b422fa56.jpg)

![ff2950590b69ba54a344501f215df314ab588f475ec1df91ff06cc68557defc5.jpg](../iclr_results/1310_Is Your Video Language Model a Reliable Judge_/tables/ff2950590b69ba54a344501f215df314ab588f475ec1df91ff06cc68557defc5.jpg)
