# ICLR 2025 Main Conference Papers

**Summary:** 36 papers with extracted content:
- 📊 Total images: 44031
- 📋 Total tables: 33468
- 📄 Total files: 77499

*Note: Equations have been filtered out and are not included.*

---

# ICLR 2025 Main Papers - Part 31 of 100

## 目录 (Table of Contents)

1. [$\gamma-$MoD: Exploring Mixture-of-Depth Adaptation for Multimodal Large Language Models](#gamma-MoD-Exploring-Mixture-of-Depth-Adaptation-for-Multimodal-Large-Language-Models)
2. [Forgetting Transformer: Softmax Attention with a Forget Gate](#Forgetting-Transformer-Softmax-Attention-with-a-Forget-Gate)
3. [CofCA: A STEP-WISE Counterfactual Multi-hop QA benchmark](#CofCA-A-STEP-WISE-Counterfactual-Multi-hop-QA-benchmark)
4. [Understanding Matrix Function Normalizations in Covariance Pooling through the Lens of Riemannian Geometry](#Understanding-Matrix-Function-Normalizations-in-Covariance-Pooling-through-the-Lens-of-Riemannian-Geometry)
5. [Rethinking Invariance in In-context Learning](#Rethinking-Invariance-in-In-context-Learning)
6. [Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-smoothing](#Understanding-and-Mitigating-Bottlenecks-of-State-Space-Models-through-the-Lens-of-Recency-and-Over-smoothing)
7. [On the Optimization Landscape of Low Rank Adaptation Methods for Large Language Models](#On-the-Optimization-Landscape-of-Low-Rank-Adaptation-Methods-for-Large-Language-Models)
8. [G-LLaVA: Solving Geometric Problem with Multi-Modal Large Language Model](#G-LLaVA-Solving-Geometric-Problem-with-Multi-Modal-Large-Language-Model)
9. [AdaFisher: Adaptive Second Order Optimization via Fisher Information](#AdaFisher-Adaptive-Second-Order-Optimization-via-Fisher-Information)
10. [Learning from Imperfect  Human Feedback: A Tale from Corruption-Robust Dueling](#Learning-from-Imperfect-Human-Feedback-A-Tale-from-Corruption-Robust-Dueling)
11. [mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models](#mPLUG-Owl3-Towards-Long-Image-Sequence-Understanding-in-Multi-Modal-Large-Language-Models)
12. [UV-Attack: Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping](#UV-Attack-Physical-World-Adversarial-Attacks-on-Person-Detection-via-Dynamic-NeRF-based-UV-Mapping)
13. [Convergence of Score-Based Discrete Diffusion Models: A Discrete-Time Analysis](#Convergence-of-Score-Based-Discrete-Diffusion-Models-A-Discrete-Time-Analysis)
14. [Logicbreaks: A Framework for Understanding Subversion of Rule-based Inference](#Logicbreaks-A-Framework-for-Understanding-Subversion-of-Rule-based-Inference)
15. [ESE: Espresso Sentence Embeddings](#ESE-Espresso-Sentence-Embeddings)
16. [Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling](#Tackling-Data-Corruption-in-Offline-Reinforcement-Learning-via-Sequence-Modeling)
17. [Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories](#Adaptive-Shrinkage-Estimation-for-Personalized-Deep-Kernel-Regression-in-Modeling-Brain-Trajectories)
18. [Scalable Discrete Diffusion Samplers: Combinatorial Optimization and Statistical Physics](#Scalable-Discrete-Diffusion-Samplers-Combinatorial-Optimization-and-Statistical-Physics)
19. [Random-Set Neural Networks](#Random-Set-Neural-Networks)
20. [SOREL: A Stochastic Algorithm for Spectral Risks Minimization](#SOREL-A-Stochastic-Algorithm-for-Spectral-Risks-Minimization)
21. [Encryption-Friendly LLM Architecture](#Encryption-Friendly-LLM-Architecture)
22. [TEOChat: A Large Vision-Language Assistant for Temporal Earth Observation Data](#TEOChat-A-Large-Vision-Language-Assistant-for-Temporal-Earth-Observation-Data)
23. [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](#LongMemEval-Benchmarking-Chat-Assistants-on-Long-Term-Interactive-Memory)
24. [ICLR: In-Context Learning of Representations](#ICLR-In-Context-Learning-of-Representations)
25. [Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges](#Explanations-of-GNN-on-Evolving-Graphs-via-Axiomatic-Layer-edges)
26. [Streamlining Prediction in Bayesian Deep Learning](#Streamlining-Prediction-in-Bayesian-Deep-Learning)
27. [Think Thrice Before You Act: Progressive Thought Refinement in Large Language Models](#Think-Thrice-Before-You-Act-Progressive-Thought-Refinement-in-Large-Language-Models)
28. [TEASER: Token Enhanced Spatial Modeling for Expressions Reconstruction](#TEASER-Token-Enhanced-Spatial-Modeling-for-Expressions-Reconstruction)
29. [Leveraging Flatness to Improve Information-Theoretic Generalization Bounds for SGD](#Leveraging-Flatness-to-Improve-Information-Theoretic-Generalization-Bounds-for-SGD)
30. [Boltzmann priors for Implicit Transfer Operators](#Boltzmann-priors-for-Implicit-Transfer-Operators)
31. [Diverse Preference Learning for Capabilities and Alignment](#Diverse-Preference-Learning-for-Capabilities-and-Alignment)
32. [CONTRA: Conformal Prediction Region via Normalizing Flow Transformation](#CONTRA-Conformal-Prediction-Region-via-Normalizing-Flow-Transformation)
33. [Bayesian WeakS-to-Strong from Text Classification to Generation](#Bayesian-WeakS-to-Strong-from-Text-Classification-to-Generation)
34. [Multimodal Lego: Model Merging and Fine-Tuning Across Topologies and Modalities in Biomedicine](#Multimodal-Lego-Model-Merging-and-Fine-Tuning-Across-Topologies-and-Modalities-in-Biomedicine)
35. [Distribution-Free Data Uncertainty for Neural Network Regression](#Distribution-Free-Data-Uncertainty-for-Neural-Network-Regression)
36. [Jump Your Steps: Optimizing Sampling Schedule of Discrete Diffusion Models](#Jump-Your-Steps-Optimizing-Sampling-Schedule-of-Discrete-Diffusion-Models)

---


## $\gamma-$MoD: Exploring Mixture-of-Depth Adaptation for Multimodal Large Language Models

### Images

![0ba71efae3a34f67afd1063ad89e9a15023a44d65c78a4ab55d6bd9a36ef43c3.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/images/0ba71efae3a34f67afd1063ad89e9a15023a44d65c78a4ab55d6bd9a36ef43c3.jpg)

![3de52e0800c8b5226f97c6ae363e3d2449aa4509e1e0f241723898cef344ac00.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/images/3de52e0800c8b5226f97c6ae363e3d2449aa4509e1e0f241723898cef344ac00.jpg)

![450033bd3dbe2a1f2453149f79c4d136f0a53b33f8ac83c07311001192b84aeb.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/images/450033bd3dbe2a1f2453149f79c4d136f0a53b33f8ac83c07311001192b84aeb.jpg)

![457b6808f684f2a54b61d2311744e532623ab84579d90b894e6ea4e11ef2eeb9.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/images/457b6808f684f2a54b61d2311744e532623ab84579d90b894e6ea4e11ef2eeb9.jpg)

![48e31ab6455902dc805d3e08e8eb9b2203e93e3a8b2db31f0b5e79ab7c70293f.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/images/48e31ab6455902dc805d3e08e8eb9b2203e93e3a8b2db31f0b5e79ab7c70293f.jpg)

![5891e5cb3bfca99515ac1355e390a924fca44c1aa1d0f120da25f60641a0af0d.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/images/5891e5cb3bfca99515ac1355e390a924fca44c1aa1d0f120da25f60641a0af0d.jpg)

![58f6220bd8f9b19813def8cdf82871efcf91ff06910719a4b1db95687ebf433d.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/images/58f6220bd8f9b19813def8cdf82871efcf91ff06910719a4b1db95687ebf433d.jpg)

![6e11f8aa52d2e2cc4226bdae3bab30d67d28e9e889c64509bb3c92da8e740a0d.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/images/6e11f8aa52d2e2cc4226bdae3bab30d67d28e9e889c64509bb3c92da8e740a0d.jpg)

![79a01df9c28ca0c8347cdc292cbabd09f3a1bcc9e009d534a4820941f4759e8b.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/images/79a01df9c28ca0c8347cdc292cbabd09f3a1bcc9e009d534a4820941f4759e8b.jpg)

![88795fa30c5d378a56371ac7b11279f4e5abae50d65f291d63ee3c900c5e7992.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/images/88795fa30c5d378a56371ac7b11279f4e5abae50d65f291d63ee3c900c5e7992.jpg)

![a2f52db2157a0fd8e90b5b330487fb552c617d485898532c70ad22207ced1f34.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/images/a2f52db2157a0fd8e90b5b330487fb552c617d485898532c70ad22207ced1f34.jpg)

![ac4c69753902e617ce33ab89d8eb2074936724ff1d8af8779563adfefbd26deb.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/images/ac4c69753902e617ce33ab89d8eb2074936724ff1d8af8779563adfefbd26deb.jpg)

![b99b2e9b391ed6bf1c6e860b02c35a0995b41a40f68d33765646131b48894d56.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/images/b99b2e9b391ed6bf1c6e860b02c35a0995b41a40f68d33765646131b48894d56.jpg)

![ba0ebbc3796ea113c339f2531af6bda7bf646eed617a83604ebb9e5bf66dcf57.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/images/ba0ebbc3796ea113c339f2531af6bda7bf646eed617a83604ebb9e5bf66dcf57.jpg)

![d42d8b62d9f65929d3ae3aa9f8b325ae505bc3ebd5d96592186201a051672767.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/images/d42d8b62d9f65929d3ae3aa9f8b325ae505bc3ebd5d96592186201a051672767.jpg)

![d43b7e0004f60695602e7d39927ef41f16b9f821e702ee6770509e94b713a28c.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/images/d43b7e0004f60695602e7d39927ef41f16b9f821e702ee6770509e94b713a28c.jpg)

![d9625c2cefe3d0d86e57af7cf15e6151f680bf301770599628a107f5d8bd1793.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/images/d9625c2cefe3d0d86e57af7cf15e6151f680bf301770599628a107f5d8bd1793.jpg)

![e1dcaac16531e02193062052dc6a163c7a9f537686800ac8d1dc2ca19dddd9da.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/images/e1dcaac16531e02193062052dc6a163c7a9f537686800ac8d1dc2ca19dddd9da.jpg)

![eaa5aeba0e26603ab53adc4ffed02ad289a052250fbc9e8978806795f76a5594.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/images/eaa5aeba0e26603ab53adc4ffed02ad289a052250fbc9e8978806795f76a5594.jpg)

![f0639fc0725ae9ac536450424373d907dd2605733ad6c484f7bd2174950cba9e.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/images/f0639fc0725ae9ac536450424373d907dd2605733ad6c484f7bd2174950cba9e.jpg)

### Tables

![19471a158f76933bce61ef578468f4ef360e3bf49733647bb979b8c14a9c9780.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/tables/19471a158f76933bce61ef578468f4ef360e3bf49733647bb979b8c14a9c9780.jpg)

![1e02028a44abcd555397a860c9d9449adc7247566a595d79c336bab9a8999582.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/tables/1e02028a44abcd555397a860c9d9449adc7247566a595d79c336bab9a8999582.jpg)

![30f0672ccce93d9f99b9dbbe746b4600e31f8e29b7bdabb01f27814a2d460a48.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/tables/30f0672ccce93d9f99b9dbbe746b4600e31f8e29b7bdabb01f27814a2d460a48.jpg)

![343536ef624a3e76237f34cc448ccdd21dd84e79e369d1e3afc9b428a31dfc9c.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/tables/343536ef624a3e76237f34cc448ccdd21dd84e79e369d1e3afc9b428a31dfc9c.jpg)

![41823b8e262f00a8d78d19a29ee9c357597bb4f5a4a6badddcd04e0880dde562.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/tables/41823b8e262f00a8d78d19a29ee9c357597bb4f5a4a6badddcd04e0880dde562.jpg)

![43f7bb30a5a594acd73e03801bdb932f24d41b27c1c63ca6b7509bc625e9b636.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/tables/43f7bb30a5a594acd73e03801bdb932f24d41b27c1c63ca6b7509bc625e9b636.jpg)

![4a9b69def541c9295ea70fc5760ee47812bc5a668b7890d19d055cda9efdabe9.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/tables/4a9b69def541c9295ea70fc5760ee47812bc5a668b7890d19d055cda9efdabe9.jpg)

![72544b22838a1b778ae6c642e2059c37a7b923d51b69bcc4aa8903f57f545cfc.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/tables/72544b22838a1b778ae6c642e2059c37a7b923d51b69bcc4aa8903f57f545cfc.jpg)

![730ca3d614e1b590e4d0979c84eb573a32c25c7fe955f0f8378f53ea0e252cb6.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/tables/730ca3d614e1b590e4d0979c84eb573a32c25c7fe955f0f8378f53ea0e252cb6.jpg)

![730ee25e4f5bc2d732dc788e31f44c0f390998c1e63235401de1adfdd2d65c06.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/tables/730ee25e4f5bc2d732dc788e31f44c0f390998c1e63235401de1adfdd2d65c06.jpg)

![802bf147d727d962a184f5f1b77d0d325ede4d056a3d46084dbf03ce5228a2a5.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/tables/802bf147d727d962a184f5f1b77d0d325ede4d056a3d46084dbf03ce5228a2a5.jpg)

![8d770d4add7c20dba830c1f6f187a39f9f26021625832d60cf93a9c550036db1.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/tables/8d770d4add7c20dba830c1f6f187a39f9f26021625832d60cf93a9c550036db1.jpg)

![9b8e375bbc275e5ecc7c71d4bc94dc8bcf33c166795259e40734078dd77687db.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/tables/9b8e375bbc275e5ecc7c71d4bc94dc8bcf33c166795259e40734078dd77687db.jpg)

![a1392e3996989239c50ad4997e9c241da8f22ea1b8a6d1d7a90fdb62880789a8.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/tables/a1392e3996989239c50ad4997e9c241da8f22ea1b8a6d1d7a90fdb62880789a8.jpg)

![ae53fb9c8b609fdfb4312678431ae519bd0e8007a2a5c218f04c0737aaad11f2.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/tables/ae53fb9c8b609fdfb4312678431ae519bd0e8007a2a5c218f04c0737aaad11f2.jpg)

![dabc6e2b215ae2c5a0654addc50f13659ba4da233d67f7918ab5ccbcc7c64b03.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/tables/dabc6e2b215ae2c5a0654addc50f13659ba4da233d67f7918ab5ccbcc7c64b03.jpg)

![dd2e5f27a5a124d3d1388e318e0b119f7b61d20f4da00250a506fdffda66e574.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/tables/dd2e5f27a5a124d3d1388e318e0b119f7b61d20f4da00250a506fdffda66e574.jpg)

![e5335d74355a7c34ae73c7e7af9f5622f0a7ec60655435df44c1ef6d0be9fe5c.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/tables/e5335d74355a7c34ae73c7e7af9f5622f0a7ec60655435df44c1ef6d0be9fe5c.jpg)

![ed4b6da734540662f61c54feaa402f50cda72666c658cea2763c60fc4f65de30.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/tables/ed4b6da734540662f61c54feaa402f50cda72666c658cea2763c60fc4f65de30.jpg)

![f9ba6fd5dc76bf5d80f5b6b3a12851a58faa5c62794b912fefdc795b18530972.jpg](../iclr_results/1129_VOILA_ Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning/tables/f9ba6fd5dc76bf5d80f5b6b3a12851a58faa5c62794b912fefdc795b18530972.jpg)

## $\gamma-$MoD: Exploring Mixture-of-Depth Adaptation for Multimodal Large Language Models


### Images

![44c1634f851ab2d4f2884d51c0c6e40c65aa415c8fdcc34e982b5e23aad5d6ad.jpg](../iclr_results/1130_$_gamma-$MoD_ Exploring Mixture-of-Depth Adaptation for Multimodal Large Language Models/images/44c1634f851ab2d4f2884d51c0c6e40c65aa415c8fdcc34e982b5e23aad5d6ad.jpg)

![6b2f92038834e074e877ce5859d094773cbda425734383b671b9325a3f6af93e.jpg](../iclr_results/1130_$_gamma-$MoD_ Exploring Mixture-of-Depth Adaptation for Multimodal Large Language Models/images/6b2f92038834e074e877ce5859d094773cbda425734383b671b9325a3f6af93e.jpg)

![82ccfa2576d58b49298ae57c84aed0dc0f3425461ab97e386f4631f11777bb73.jpg](../iclr_results/1130_$_gamma-$MoD_ Exploring Mixture-of-Depth Adaptation for Multimodal Large Language Models/images/82ccfa2576d58b49298ae57c84aed0dc0f3425461ab97e386f4631f11777bb73.jpg)

![85a78cff4b84fedd1a73127130f9b5272d20000f626ff14fe57829758b38bcbd.jpg](../iclr_results/1130_$_gamma-$MoD_ Exploring Mixture-of-Depth Adaptation for Multimodal Large Language Models/images/85a78cff4b84fedd1a73127130f9b5272d20000f626ff14fe57829758b38bcbd.jpg)

### Tables

![2fb61a890469e8dde7d0f028e004fb9e270d38a807110c5abdda24fc10fd3a60.jpg](../iclr_results/1130_$_gamma-$MoD_ Exploring Mixture-of-Depth Adaptation for Multimodal Large Language Models/tables/2fb61a890469e8dde7d0f028e004fb9e270d38a807110c5abdda24fc10fd3a60.jpg)

![318cc8715c1a4726aca1ba0ffbbae52aad428a79ad5dc433c1d249d88548be91.jpg](../iclr_results/1130_$_gamma-$MoD_ Exploring Mixture-of-Depth Adaptation for Multimodal Large Language Models/tables/318cc8715c1a4726aca1ba0ffbbae52aad428a79ad5dc433c1d249d88548be91.jpg)

![7064e98e5fc6e5156a68fcce136568328ece7382b84cb59a7ca84d047ab525d6.jpg](../iclr_results/1130_$_gamma-$MoD_ Exploring Mixture-of-Depth Adaptation for Multimodal Large Language Models/tables/7064e98e5fc6e5156a68fcce136568328ece7382b84cb59a7ca84d047ab525d6.jpg)

![91e75930dea4d333c324ce2c33af9d999c4b7e984dc198b2b4252a7e2e949a95.jpg](../iclr_results/1130_$_gamma-$MoD_ Exploring Mixture-of-Depth Adaptation for Multimodal Large Language Models/tables/91e75930dea4d333c324ce2c33af9d999c4b7e984dc198b2b4252a7e2e949a95.jpg)

![9e7b393e03af54fa2b989c6a7a978e29009a9b902314c38ca05a20902790950f.jpg](../iclr_results/1130_$_gamma-$MoD_ Exploring Mixture-of-Depth Adaptation for Multimodal Large Language Models/tables/9e7b393e03af54fa2b989c6a7a978e29009a9b902314c38ca05a20902790950f.jpg)

![e747611538a8318032de879eab3a5d31b05b1fc6bd410cc36ce612d4334d63bc.jpg](../iclr_results/1130_$_gamma-$MoD_ Exploring Mixture-of-Depth Adaptation for Multimodal Large Language Models/tables/e747611538a8318032de879eab3a5d31b05b1fc6bd410cc36ce612d4334d63bc.jpg)

## Forgetting Transformer: Softmax Attention with a Forget Gate


### Images

![16e7ee8cb6ab4e05ee3cc33e86fbaa138881d917afff611e1a73f0857c02f589.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/16e7ee8cb6ab4e05ee3cc33e86fbaa138881d917afff611e1a73f0857c02f589.jpg)

![1cf8517b29443c9a1c0fae6077551cf92d26af7a59de666d4e936468c52a03a1.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/1cf8517b29443c9a1c0fae6077551cf92d26af7a59de666d4e936468c52a03a1.jpg)

![27a29b98e9e6a7fa55a5168b8851749f986d788c079341de878e8af16b4bb096.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/27a29b98e9e6a7fa55a5168b8851749f986d788c079341de878e8af16b4bb096.jpg)

![28ee831b86a9cbf2f63ff72f87aee4a26943e9e443d7dc2b172ce11873101fcf.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/28ee831b86a9cbf2f63ff72f87aee4a26943e9e443d7dc2b172ce11873101fcf.jpg)

![2d1cae0ff22da2cdd287acf40d47af2afa75fb29c421657bc7814ab40d5345bd.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/2d1cae0ff22da2cdd287acf40d47af2afa75fb29c421657bc7814ab40d5345bd.jpg)

![342d216f116aef78df8e0de34878fc47ff3cd4d54c6e1398f0506bc89e70b304.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/342d216f116aef78df8e0de34878fc47ff3cd4d54c6e1398f0506bc89e70b304.jpg)

![459e79b562e1f8c7be9da6019742d1e97da6b9b592729933f41ff91f40c17830.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/459e79b562e1f8c7be9da6019742d1e97da6b9b592729933f41ff91f40c17830.jpg)

![4d53e2adbb241429efc59fe599d49a35a25c685db1177c7c04eeb23800b3981c.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/4d53e2adbb241429efc59fe599d49a35a25c685db1177c7c04eeb23800b3981c.jpg)

![63b47cf3fb138a1df58043c44966deefe1281dd64f9d404f3349d961ad6d0173.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/63b47cf3fb138a1df58043c44966deefe1281dd64f9d404f3349d961ad6d0173.jpg)

![679e3720590dedc0f01c412024c5a217374f76352372adcee61708116822121f.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/679e3720590dedc0f01c412024c5a217374f76352372adcee61708116822121f.jpg)

![72f7ea01a50e6ab0a4af85cd09901674b114e480f432edb4aeac308605da0dd7.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/72f7ea01a50e6ab0a4af85cd09901674b114e480f432edb4aeac308605da0dd7.jpg)

![75cd671beab1d3dc0e0b62052628672509d310f101cb4f822a3e179e70306e9a.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/75cd671beab1d3dc0e0b62052628672509d310f101cb4f822a3e179e70306e9a.jpg)

![76709c197f8cac3b10388617eee992ea478226ee6749df03489703ac130b1521.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/76709c197f8cac3b10388617eee992ea478226ee6749df03489703ac130b1521.jpg)

![77a8edc4f0ab6dffc501e69257d4523b63d3ebdcd2e4324d752479924c4ba456.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/77a8edc4f0ab6dffc501e69257d4523b63d3ebdcd2e4324d752479924c4ba456.jpg)

![8a29aea8780203e7ceb01e083ca0d8dbb546ad0354c7cde7cb0d1fcba0fee95d.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/8a29aea8780203e7ceb01e083ca0d8dbb546ad0354c7cde7cb0d1fcba0fee95d.jpg)

![8ec64b770728b599b3b13ca033b1d876a7a5b03ceb15f46ca3f31f207818a2f6.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/8ec64b770728b599b3b13ca033b1d876a7a5b03ceb15f46ca3f31f207818a2f6.jpg)

![98c32ece81e06c1df418e8a129d0fcc0c23dceffdeca444bc308f8a447e6b0bc.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/98c32ece81e06c1df418e8a129d0fcc0c23dceffdeca444bc308f8a447e6b0bc.jpg)

![99164ea5473e8f61c699569933bbe90ef739d42d71ca661f1e9e81359f3ed346.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/99164ea5473e8f61c699569933bbe90ef739d42d71ca661f1e9e81359f3ed346.jpg)

![9d6a482d880cf0578d41e100df1c068e2a90f9c5d74a6d2bc0e97f621d9eed99.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/9d6a482d880cf0578d41e100df1c068e2a90f9c5d74a6d2bc0e97f621d9eed99.jpg)

![a45c9c337179736179570866213a1628b3227282731025ef07c7584b78c32b4c.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/a45c9c337179736179570866213a1628b3227282731025ef07c7584b78c32b4c.jpg)

![bbe39e8ba03a213a37764a70083626c8f228297c6ee5c298c5018dc22495c0b5.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/bbe39e8ba03a213a37764a70083626c8f228297c6ee5c298c5018dc22495c0b5.jpg)

![c43a13b7c8c2b4427cc915e3b2d6c94885e9a7a7013a2c9e4c837dbd617aa918.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/c43a13b7c8c2b4427cc915e3b2d6c94885e9a7a7013a2c9e4c837dbd617aa918.jpg)

![cf0905e93ee767b5b82f59a6a7834880602b297de9a9b667720f93fcee511217.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/cf0905e93ee767b5b82f59a6a7834880602b297de9a9b667720f93fcee511217.jpg)

![cfee11405c56fe61c924a55a899bbfa653fabfb6ecefd4ad5c6265e8ae73abac.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/cfee11405c56fe61c924a55a899bbfa653fabfb6ecefd4ad5c6265e8ae73abac.jpg)

![d1cdf6380b40414456ab44e813501dd809b64583103c58c63dee7f8d0cdf7403.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/d1cdf6380b40414456ab44e813501dd809b64583103c58c63dee7f8d0cdf7403.jpg)

![e4a51698e3a168f3b0cd93e6cbf73c7855b3823da425aa37411b5f40a293d3ad.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/e4a51698e3a168f3b0cd93e6cbf73c7855b3823da425aa37411b5f40a293d3ad.jpg)

![e537f659180d7dd5bee6ba14b6549ff69e3f106992ffc15d2b5910e91ab0419f.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/e537f659180d7dd5bee6ba14b6549ff69e3f106992ffc15d2b5910e91ab0419f.jpg)

![e9a83d444a22ecc049d48f127ad3464b96e287d7db7f154e45953db80174e536.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/images/e9a83d444a22ecc049d48f127ad3464b96e287d7db7f154e45953db80174e536.jpg)

### Tables

![08ed4234cd296c331c9f7f2775eb418d8ecf874075e5ccac99041418325cc30b.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/tables/08ed4234cd296c331c9f7f2775eb418d8ecf874075e5ccac99041418325cc30b.jpg)

![1ad548fc77fecacede0d90faddc01507fa0358189a374728c1b6c37ad15cdff1.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/tables/1ad548fc77fecacede0d90faddc01507fa0358189a374728c1b6c37ad15cdff1.jpg)

![6bbd9912a2d9678800983bf4eba7ee1247f8c66c807611bc2474f2c2f0d7f5f2.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/tables/6bbd9912a2d9678800983bf4eba7ee1247f8c66c807611bc2474f2c2f0d7f5f2.jpg)

![82fc3abca8f37a00dfadff63e04172318eb2906767a34f093aa3316d0b161304.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/tables/82fc3abca8f37a00dfadff63e04172318eb2906767a34f093aa3316d0b161304.jpg)

![9fa7535ff53dd6ac90723c682760f329b4b5a822e1d13e216ae7009a27020516.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/tables/9fa7535ff53dd6ac90723c682760f329b4b5a822e1d13e216ae7009a27020516.jpg)

![a8d7a789cd58998ae9de318dd718ec3e618a2a930dfa67826cbd81a2d48b3f44.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/tables/a8d7a789cd58998ae9de318dd718ec3e618a2a930dfa67826cbd81a2d48b3f44.jpg)

![d1278baa7b315a0b3f16b1f87223f54ebfeecc880fa7302c5c20c062d8a6c914.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/tables/d1278baa7b315a0b3f16b1f87223f54ebfeecc880fa7302c5c20c062d8a6c914.jpg)

![d74cd5cee823e90787afbb05a5f4a2644f19259b25f5aed73c125d5e8bd5de62.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/tables/d74cd5cee823e90787afbb05a5f4a2644f19259b25f5aed73c125d5e8bd5de62.jpg)

![f07e91852c37e9d18ac5178dfca6b1db8aae864c7233a67c178e0a2ddf3f3342.jpg](../iclr_results/1131_Forgetting Transformer_ Softmax Attention with a Forget Gate/tables/f07e91852c37e9d18ac5178dfca6b1db8aae864c7233a67c178e0a2ddf3f3342.jpg)

## CofCA: A STEP-WISE Counterfactual Multi-hop QA benchmark


### Images

![2b0be31cb416b873b0e755f3a174d084be1813177a1d1696c5cdcf65233a4483.jpg](../iclr_results/1132_CofCA_ A STEP-WISE Counterfactual Multi-hop QA benchmark/images/2b0be31cb416b873b0e755f3a174d084be1813177a1d1696c5cdcf65233a4483.jpg)

![35e99ca8699136d878b2b0419fbe4622b9605f0eb869b89e72fdef0ea099a681.jpg](../iclr_results/1132_CofCA_ A STEP-WISE Counterfactual Multi-hop QA benchmark/images/35e99ca8699136d878b2b0419fbe4622b9605f0eb869b89e72fdef0ea099a681.jpg)

![8aab901575a7683ca994d88c496ac7cceee0e324362f5809de16db65181278fb.jpg](../iclr_results/1132_CofCA_ A STEP-WISE Counterfactual Multi-hop QA benchmark/images/8aab901575a7683ca994d88c496ac7cceee0e324362f5809de16db65181278fb.jpg)

![a57b0150e17002e280b2d3342d1fd97fbd321b83397239b3fed3c2b10e4b4ad7.jpg](../iclr_results/1132_CofCA_ A STEP-WISE Counterfactual Multi-hop QA benchmark/images/a57b0150e17002e280b2d3342d1fd97fbd321b83397239b3fed3c2b10e4b4ad7.jpg)

![baebd148dfde873dd385037ef465f62d63c8b3ffc0cb7c9695f151b8b63ca92b.jpg](../iclr_results/1132_CofCA_ A STEP-WISE Counterfactual Multi-hop QA benchmark/images/baebd148dfde873dd385037ef465f62d63c8b3ffc0cb7c9695f151b8b63ca92b.jpg)

![e2f60402bbebafa01f97fd0e2153948d30f8400186511fd4549b81ba228e1344.jpg](../iclr_results/1132_CofCA_ A STEP-WISE Counterfactual Multi-hop QA benchmark/images/e2f60402bbebafa01f97fd0e2153948d30f8400186511fd4549b81ba228e1344.jpg)

### Tables

![28da254486845d64907f3d42e432378f3a5f43f0ef41bc3bd1511c900af525e8.jpg](../iclr_results/1132_CofCA_ A STEP-WISE Counterfactual Multi-hop QA benchmark/tables/28da254486845d64907f3d42e432378f3a5f43f0ef41bc3bd1511c900af525e8.jpg)

![2c1c7d28b3bb029a7d97db93c642dcac33cb41ac44de004b053fd869e5a0c535.jpg](../iclr_results/1132_CofCA_ A STEP-WISE Counterfactual Multi-hop QA benchmark/tables/2c1c7d28b3bb029a7d97db93c642dcac33cb41ac44de004b053fd869e5a0c535.jpg)

![2fe64e83177d6a82ce8ab5efcfb6014f860dd768b43ae96aeac831f4d5dbd924.jpg](../iclr_results/1132_CofCA_ A STEP-WISE Counterfactual Multi-hop QA benchmark/tables/2fe64e83177d6a82ce8ab5efcfb6014f860dd768b43ae96aeac831f4d5dbd924.jpg)

![30b6f558c35f43dcf54494e179bb0be369af01848b9d6bc5bc95091feca02365.jpg](../iclr_results/1132_CofCA_ A STEP-WISE Counterfactual Multi-hop QA benchmark/tables/30b6f558c35f43dcf54494e179bb0be369af01848b9d6bc5bc95091feca02365.jpg)

![31215c5a5e9859331b9f329a30123a3a5073958da7e2ece8842c6151385a73e2.jpg](../iclr_results/1132_CofCA_ A STEP-WISE Counterfactual Multi-hop QA benchmark/tables/31215c5a5e9859331b9f329a30123a3a5073958da7e2ece8842c6151385a73e2.jpg)

![63ad755731cf9712342de8e5d3c9bc0eb3e4bc2e0b728b265176f352ed0549b9.jpg](../iclr_results/1132_CofCA_ A STEP-WISE Counterfactual Multi-hop QA benchmark/tables/63ad755731cf9712342de8e5d3c9bc0eb3e4bc2e0b728b265176f352ed0549b9.jpg)

![65e18d72f5327473b4af4844bc23a5e33048764ab6d62e663f0ddf8af3207a24.jpg](../iclr_results/1132_CofCA_ A STEP-WISE Counterfactual Multi-hop QA benchmark/tables/65e18d72f5327473b4af4844bc23a5e33048764ab6d62e663f0ddf8af3207a24.jpg)

![7351b09f396096e086b42572fc9db2479de573cdfe99ec2736e6486bda32a87d.jpg](../iclr_results/1132_CofCA_ A STEP-WISE Counterfactual Multi-hop QA benchmark/tables/7351b09f396096e086b42572fc9db2479de573cdfe99ec2736e6486bda32a87d.jpg)

![8be019d6cece33127c311130881d8f14109657c8952f3d5b268205cabafc27dc.jpg](../iclr_results/1132_CofCA_ A STEP-WISE Counterfactual Multi-hop QA benchmark/tables/8be019d6cece33127c311130881d8f14109657c8952f3d5b268205cabafc27dc.jpg)

![94f41677fbf165aaa18e45d8d58148e14ada2b40186f29ce55594416201c3b0e.jpg](../iclr_results/1132_CofCA_ A STEP-WISE Counterfactual Multi-hop QA benchmark/tables/94f41677fbf165aaa18e45d8d58148e14ada2b40186f29ce55594416201c3b0e.jpg)

![9ee600eeb6ecee07512bc803bbdf224cd3c4a33960767705e0c2a5f8f9d1a193.jpg](../iclr_results/1132_CofCA_ A STEP-WISE Counterfactual Multi-hop QA benchmark/tables/9ee600eeb6ecee07512bc803bbdf224cd3c4a33960767705e0c2a5f8f9d1a193.jpg)

![acbd2d521de61588c0135ea963f98277ba1be12ad1a0787f3f840c24500fe02e.jpg](../iclr_results/1132_CofCA_ A STEP-WISE Counterfactual Multi-hop QA benchmark/tables/acbd2d521de61588c0135ea963f98277ba1be12ad1a0787f3f840c24500fe02e.jpg)

![c49f0f3a358d69add334fee0a53e273adb8bf4dd377c33510cca8de3f9398dfc.jpg](../iclr_results/1132_CofCA_ A STEP-WISE Counterfactual Multi-hop QA benchmark/tables/c49f0f3a358d69add334fee0a53e273adb8bf4dd377c33510cca8de3f9398dfc.jpg)

![cb250bd48a965e6c9ab2c01688cc6176e3f6d3ce50cbf4c37e66569b1b647fee.jpg](../iclr_results/1132_CofCA_ A STEP-WISE Counterfactual Multi-hop QA benchmark/tables/cb250bd48a965e6c9ab2c01688cc6176e3f6d3ce50cbf4c37e66569b1b647fee.jpg)

![d4443183ac2d8ac205692ae39e5f48fa85cd7b0ab72b7e78d356479ed97c2ef7.jpg](../iclr_results/1132_CofCA_ A STEP-WISE Counterfactual Multi-hop QA benchmark/tables/d4443183ac2d8ac205692ae39e5f48fa85cd7b0ab72b7e78d356479ed97c2ef7.jpg)

## Understanding Matrix Function Normalizations in Covariance Pooling through the Lens of Riemannian Geometry


### Images

![cd323825ee5457b02c3f29cbc159b72473789075a500e14fd625ad020c8fde19.jpg](../iclr_results/1133_Understanding Matrix Function Normalizations in Covariance Pooling through the Lens of Riemannian Ge/images/cd323825ee5457b02c3f29cbc159b72473789075a500e14fd625ad020c8fde19.jpg)

![e227dd5f5fe04f02c2487710204958ffe8e86d9a5c78372ed3db5db661d3a2ce.jpg](../iclr_results/1133_Understanding Matrix Function Normalizations in Covariance Pooling through the Lens of Riemannian Ge/images/e227dd5f5fe04f02c2487710204958ffe8e86d9a5c78372ed3db5db661d3a2ce.jpg)

![ef0fe179d4ffaa0f6c93dab9f8fd12d27d9801f77368180052ef6b02d208568a.jpg](../iclr_results/1133_Understanding Matrix Function Normalizations in Covariance Pooling through the Lens of Riemannian Ge/images/ef0fe179d4ffaa0f6c93dab9f8fd12d27d9801f77368180052ef6b02d208568a.jpg)

### Tables

![17feba15a0d9c65c48ab1c19feafa6d1574ab23a648420f1ed97858cdf2a2b9d.jpg](../iclr_results/1133_Understanding Matrix Function Normalizations in Covariance Pooling through the Lens of Riemannian Ge/tables/17feba15a0d9c65c48ab1c19feafa6d1574ab23a648420f1ed97858cdf2a2b9d.jpg)

![2226b712ed734d7a6cd752429b11921e93a1ec4d580ea5f73f78f18366ceb844.jpg](../iclr_results/1133_Understanding Matrix Function Normalizations in Covariance Pooling through the Lens of Riemannian Ge/tables/2226b712ed734d7a6cd752429b11921e93a1ec4d580ea5f73f78f18366ceb844.jpg)

![2e3ac27f945bb3f2159fa5001b334a5c997a2b2046fac4cd2a2f3d2100c8a338.jpg](../iclr_results/1133_Understanding Matrix Function Normalizations in Covariance Pooling through the Lens of Riemannian Ge/tables/2e3ac27f945bb3f2159fa5001b334a5c997a2b2046fac4cd2a2f3d2100c8a338.jpg)

![3302aa50094132ea1ef2e08a978bb9dca1b409796fb0b6bb2cf2cb31bd31d998.jpg](../iclr_results/1133_Understanding Matrix Function Normalizations in Covariance Pooling through the Lens of Riemannian Ge/tables/3302aa50094132ea1ef2e08a978bb9dca1b409796fb0b6bb2cf2cb31bd31d998.jpg)

![77816c521867e3fa35d6ac99f5c10df74491756dbe07d7a38553918e47385a6a.jpg](../iclr_results/1133_Understanding Matrix Function Normalizations in Covariance Pooling through the Lens of Riemannian Ge/tables/77816c521867e3fa35d6ac99f5c10df74491756dbe07d7a38553918e47385a6a.jpg)

![97095520ff98d8bf35ce40d26b2bc9a5e6c3a34dd625f427d7dec9c6cee98fc2.jpg](../iclr_results/1133_Understanding Matrix Function Normalizations in Covariance Pooling through the Lens of Riemannian Ge/tables/97095520ff98d8bf35ce40d26b2bc9a5e6c3a34dd625f427d7dec9c6cee98fc2.jpg)

![ac425aeda8f27646de7e6166c42446c6d98755c7f382894473554acd9dbb502a.jpg](../iclr_results/1133_Understanding Matrix Function Normalizations in Covariance Pooling through the Lens of Riemannian Ge/tables/ac425aeda8f27646de7e6166c42446c6d98755c7f382894473554acd9dbb502a.jpg)

![cea1f6f200693fffd751612704c1904ee4e519db8a0084875b2bb8ede510ebd6.jpg](../iclr_results/1133_Understanding Matrix Function Normalizations in Covariance Pooling through the Lens of Riemannian Ge/tables/cea1f6f200693fffd751612704c1904ee4e519db8a0084875b2bb8ede510ebd6.jpg)

![d82d4f38c4639122ed5f86cf3d2082a1c19b37cd11e4444afa7747fef6a5f10f.jpg](../iclr_results/1133_Understanding Matrix Function Normalizations in Covariance Pooling through the Lens of Riemannian Ge/tables/d82d4f38c4639122ed5f86cf3d2082a1c19b37cd11e4444afa7747fef6a5f10f.jpg)

![d945da5fb7af36128b048daad414e5410bd28cbdf5af68e0ffca8fe57a2f6e54.jpg](../iclr_results/1133_Understanding Matrix Function Normalizations in Covariance Pooling through the Lens of Riemannian Ge/tables/d945da5fb7af36128b048daad414e5410bd28cbdf5af68e0ffca8fe57a2f6e54.jpg)

![f6a2538280115e689197bbefbcc6f514a54c7460a42780be4331dc36e4a66465.jpg](../iclr_results/1133_Understanding Matrix Function Normalizations in Covariance Pooling through the Lens of Riemannian Ge/tables/f6a2538280115e689197bbefbcc6f514a54c7460a42780be4331dc36e4a66465.jpg)

## Rethinking Invariance in In-context Learning


### Images

![1c503943d42722647f7703a37e6838c911a01b40feff33d79eadce0926328354.jpg](../iclr_results/1134_Rethinking Invariance in In-context Learning/images/1c503943d42722647f7703a37e6838c911a01b40feff33d79eadce0926328354.jpg)

![336e3876b23647efbd2a70f849da32ff8503fa4c5b857abfeda95dca8153e618.jpg](../iclr_results/1134_Rethinking Invariance in In-context Learning/images/336e3876b23647efbd2a70f849da32ff8503fa4c5b857abfeda95dca8153e618.jpg)

![3429708ea7786ce83dcf9f449e75bb3a2fac4fe50f2b088e4f55a1784f18f681.jpg](../iclr_results/1134_Rethinking Invariance in In-context Learning/images/3429708ea7786ce83dcf9f449e75bb3a2fac4fe50f2b088e4f55a1784f18f681.jpg)

![34cde22197a9e404ce033798e229f3a159923627bca09709a0f798103795f226.jpg](../iclr_results/1134_Rethinking Invariance in In-context Learning/images/34cde22197a9e404ce033798e229f3a159923627bca09709a0f798103795f226.jpg)

![3f37d932be91d3c0f95c5723b27b9b48530c334d0f395d233f79f877ea1d05a4.jpg](../iclr_results/1134_Rethinking Invariance in In-context Learning/images/3f37d932be91d3c0f95c5723b27b9b48530c334d0f395d233f79f877ea1d05a4.jpg)

![40ebb096980eafe450d49aee2bf82d3b65ba0dfa3f53e7aa0a845a74dd9bce8f.jpg](../iclr_results/1134_Rethinking Invariance in In-context Learning/images/40ebb096980eafe450d49aee2bf82d3b65ba0dfa3f53e7aa0a845a74dd9bce8f.jpg)

![56dfdccae123b6646f86a2695c351c502edf7917227a6b14eb251fa31d82cc78.jpg](../iclr_results/1134_Rethinking Invariance in In-context Learning/images/56dfdccae123b6646f86a2695c351c502edf7917227a6b14eb251fa31d82cc78.jpg)

![65ba617f05561677f25bf19d09f7a82b36955076c59db05e54edb901945d6d7d.jpg](../iclr_results/1134_Rethinking Invariance in In-context Learning/images/65ba617f05561677f25bf19d09f7a82b36955076c59db05e54edb901945d6d7d.jpg)

![96cd9472f1e91f4e94550ef8ce7025533d48c6a535687f8f62846c2224625470.jpg](../iclr_results/1134_Rethinking Invariance in In-context Learning/images/96cd9472f1e91f4e94550ef8ce7025533d48c6a535687f8f62846c2224625470.jpg)

![d454f764304b166a8987f9e39d2ebe58d2d76a0538439800a69063855996ca2e.jpg](../iclr_results/1134_Rethinking Invariance in In-context Learning/images/d454f764304b166a8987f9e39d2ebe58d2d76a0538439800a69063855996ca2e.jpg)

### Tables

![30b8c799d59e0ccbbba8adab3406f09d8577eae9626cdeca968cd47437d63ad8.jpg](../iclr_results/1134_Rethinking Invariance in In-context Learning/tables/30b8c799d59e0ccbbba8adab3406f09d8577eae9626cdeca968cd47437d63ad8.jpg)

![6634aac9acf0248f70c783fbe7793d20ae02df8a079ab0c4031123ef6755281a.jpg](../iclr_results/1134_Rethinking Invariance in In-context Learning/tables/6634aac9acf0248f70c783fbe7793d20ae02df8a079ab0c4031123ef6755281a.jpg)

![80bada511a65742f007d986f2c3c5f8a6c7b1ffa934bb35ff51f562d63b182e1.jpg](../iclr_results/1134_Rethinking Invariance in In-context Learning/tables/80bada511a65742f007d986f2c3c5f8a6c7b1ffa934bb35ff51f562d63b182e1.jpg)

![9f21a81e9bdcfade038493a6b28428be6cde0f7a3ab66a89ee2b876c9de4cd78.jpg](../iclr_results/1134_Rethinking Invariance in In-context Learning/tables/9f21a81e9bdcfade038493a6b28428be6cde0f7a3ab66a89ee2b876c9de4cd78.jpg)

![bc0dfd8ff3c13a3e4b8ac6250dbec7b35b2136fb6855f8652a4572f7913efc37.jpg](../iclr_results/1134_Rethinking Invariance in In-context Learning/tables/bc0dfd8ff3c13a3e4b8ac6250dbec7b35b2136fb6855f8652a4572f7913efc37.jpg)

![cec40e57afffcfe311d781dca554c65f20900bbc04b5c628bcf52356c123e1c6.jpg](../iclr_results/1134_Rethinking Invariance in In-context Learning/tables/cec40e57afffcfe311d781dca554c65f20900bbc04b5c628bcf52356c123e1c6.jpg)

![d49910522840be92d0a137f2fefc01dc930f8df89f6cc6d74ef96f8a2429449e.jpg](../iclr_results/1134_Rethinking Invariance in In-context Learning/tables/d49910522840be92d0a137f2fefc01dc930f8df89f6cc6d74ef96f8a2429449e.jpg)

![eab9d07a122aab2107471239b53e20974c91861d0edb5fcfa169696fd49c8e91.jpg](../iclr_results/1134_Rethinking Invariance in In-context Learning/tables/eab9d07a122aab2107471239b53e20974c91861d0edb5fcfa169696fd49c8e91.jpg)

## Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-smoothing


### Images

![1d9c34bd78a4982ed5fb06eed1ec29eb6654f9e91bfac5cc5dd4eb95e9dd6c81.jpg](../iclr_results/1135_Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-/images/1d9c34bd78a4982ed5fb06eed1ec29eb6654f9e91bfac5cc5dd4eb95e9dd6c81.jpg)

![7219c928b0dc47e48ef49d82732661fc5aedbe36d737f778d17b98520399f57d.jpg](../iclr_results/1135_Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-/images/7219c928b0dc47e48ef49d82732661fc5aedbe36d737f778d17b98520399f57d.jpg)

![82e6fc454e8a3c5ac61878a93e306577aec4d81d30aa32b2e7f5b972fa9568ec.jpg](../iclr_results/1135_Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-/images/82e6fc454e8a3c5ac61878a93e306577aec4d81d30aa32b2e7f5b972fa9568ec.jpg)

![9a3fe85fba830850aadc6f54f506836da59b4400d924b6b6ff7fd43066b94bf0.jpg](../iclr_results/1135_Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-/images/9a3fe85fba830850aadc6f54f506836da59b4400d924b6b6ff7fd43066b94bf0.jpg)

![a2afc7a61ec8f689f93cff75fd18f3a8424b64d65cb6f90d06dc4a28d913174b.jpg](../iclr_results/1135_Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-/images/a2afc7a61ec8f689f93cff75fd18f3a8424b64d65cb6f90d06dc4a28d913174b.jpg)

![bac37204bc0b8d0f0728d0d5aacae30086748e6b007dd38fa92164bf02f032ca.jpg](../iclr_results/1135_Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-/images/bac37204bc0b8d0f0728d0d5aacae30086748e6b007dd38fa92164bf02f032ca.jpg)

![c485b16adc395e8af1c055eeaf4c13de32042bfb6fa7637c676c2f576372df7b.jpg](../iclr_results/1135_Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-/images/c485b16adc395e8af1c055eeaf4c13de32042bfb6fa7637c676c2f576372df7b.jpg)

![c9a76d8e54ba33d97c99a64276d6585c06beb1118e2fcd557fa556e5aa7b8328.jpg](../iclr_results/1135_Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-/images/c9a76d8e54ba33d97c99a64276d6585c06beb1118e2fcd557fa556e5aa7b8328.jpg)

![e531b18c7ce8cc8489a13b1a0ef917ac66fe438d31125de9ad3f226e0b4710ff.jpg](../iclr_results/1135_Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-/images/e531b18c7ce8cc8489a13b1a0ef917ac66fe438d31125de9ad3f226e0b4710ff.jpg)

### Tables

![558b2a3fc68bc4513dec25e11fd49e71172c00c7674d8e023ff03da249114cff.jpg](../iclr_results/1135_Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-/tables/558b2a3fc68bc4513dec25e11fd49e71172c00c7674d8e023ff03da249114cff.jpg)

![99c880e11f564539ec044d25f5eac42b251f432f2e53544712cdafc7fbcd3c93.jpg](../iclr_results/1135_Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-/tables/99c880e11f564539ec044d25f5eac42b251f432f2e53544712cdafc7fbcd3c93.jpg)

![aaee49c4dcf3307673fb5f42c53ba9b8f7433c30c62a01edf78c1b91ed0eadc4.jpg](../iclr_results/1135_Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-/tables/aaee49c4dcf3307673fb5f42c53ba9b8f7433c30c62a01edf78c1b91ed0eadc4.jpg)

![b6b9bcaf1d23144a5711277a619a23e5c1929b948f0d06db90e5aa663dbf41fe.jpg](../iclr_results/1135_Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-/tables/b6b9bcaf1d23144a5711277a619a23e5c1929b948f0d06db90e5aa663dbf41fe.jpg)

![d7559e8b814f6ead192ef61541aca83219a071f2c644e701ac4146c4c58444bc.jpg](../iclr_results/1135_Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-/tables/d7559e8b814f6ead192ef61541aca83219a071f2c644e701ac4146c4c58444bc.jpg)

## On the Optimization Landscape of Low Rank Adaptation Methods for Large Language Models


### Images

![0eadc9ab650ec9f7a20b872286cb217bd04f7391bfae208de5f0a0d8febcde62.jpg](../iclr_results/1136_On the Optimization Landscape of Low Rank Adaptation Methods for Large Language Models/images/0eadc9ab650ec9f7a20b872286cb217bd04f7391bfae208de5f0a0d8febcde62.jpg)

![184f202c64eeecb3ea4fe70df75efe51e7063f726101ca13fda03d4a1bb44cb9.jpg](../iclr_results/1136_On the Optimization Landscape of Low Rank Adaptation Methods for Large Language Models/images/184f202c64eeecb3ea4fe70df75efe51e7063f726101ca13fda03d4a1bb44cb9.jpg)

![5a7c2110bb7eb89c117c0bda8298eabf0511fa1d0a34431d8b54bcc7cc02a03c.jpg](../iclr_results/1136_On the Optimization Landscape of Low Rank Adaptation Methods for Large Language Models/images/5a7c2110bb7eb89c117c0bda8298eabf0511fa1d0a34431d8b54bcc7cc02a03c.jpg)

![5e3ce56b67435239ec8d72fae7b2ac1e7b406468d9db2a7f362b5d8d581e8d0b.jpg](../iclr_results/1136_On the Optimization Landscape of Low Rank Adaptation Methods for Large Language Models/images/5e3ce56b67435239ec8d72fae7b2ac1e7b406468d9db2a7f362b5d8d581e8d0b.jpg)

### Tables

![035840ddc2f2d0abbd7d7cd25442f9be0a1a6594777ae7c1df1cc3023a67ae07.jpg](../iclr_results/1136_On the Optimization Landscape of Low Rank Adaptation Methods for Large Language Models/tables/035840ddc2f2d0abbd7d7cd25442f9be0a1a6594777ae7c1df1cc3023a67ae07.jpg)

![0cbb3099addb18d381ea389c214d393d7cc7c92bc62f7cc862419328fe07a69c.jpg](../iclr_results/1136_On the Optimization Landscape of Low Rank Adaptation Methods for Large Language Models/tables/0cbb3099addb18d381ea389c214d393d7cc7c92bc62f7cc862419328fe07a69c.jpg)

![1e39d5c4c87bb39413e8423e66a3f51b6aefda2f72d641b9cfbbe5861d0366e1.jpg](../iclr_results/1136_On the Optimization Landscape of Low Rank Adaptation Methods for Large Language Models/tables/1e39d5c4c87bb39413e8423e66a3f51b6aefda2f72d641b9cfbbe5861d0366e1.jpg)

![34a0e883c227b77bdd3d6bdf51f8622b1f88cffda9a71763847335730b7a3287.jpg](../iclr_results/1136_On the Optimization Landscape of Low Rank Adaptation Methods for Large Language Models/tables/34a0e883c227b77bdd3d6bdf51f8622b1f88cffda9a71763847335730b7a3287.jpg)

![4e7750ac7e40e01d2a6718f36eed2fc77d4d5ea601b923158bb454d496940f83.jpg](../iclr_results/1136_On the Optimization Landscape of Low Rank Adaptation Methods for Large Language Models/tables/4e7750ac7e40e01d2a6718f36eed2fc77d4d5ea601b923158bb454d496940f83.jpg)

![5d86a5b5703f0151c92142fdb597cd27607c027b752d14b4816d0331d01f28e7.jpg](../iclr_results/1136_On the Optimization Landscape of Low Rank Adaptation Methods for Large Language Models/tables/5d86a5b5703f0151c92142fdb597cd27607c027b752d14b4816d0331d01f28e7.jpg)

![79de7d1553795916aef7fd636dfad0e41686440b62db74ad90e356c3c3afd51f.jpg](../iclr_results/1136_On the Optimization Landscape of Low Rank Adaptation Methods for Large Language Models/tables/79de7d1553795916aef7fd636dfad0e41686440b62db74ad90e356c3c3afd51f.jpg)

![9c47fd15c9ef9c7b8e3353482ef9804a5a9bd256f7f8f4e7fc2b28ef745b9c45.jpg](../iclr_results/1136_On the Optimization Landscape of Low Rank Adaptation Methods for Large Language Models/tables/9c47fd15c9ef9c7b8e3353482ef9804a5a9bd256f7f8f4e7fc2b28ef745b9c45.jpg)

![caf3f172234226fbb3f928898270863d182a82e04aa75557012790635ada422e.jpg](../iclr_results/1136_On the Optimization Landscape of Low Rank Adaptation Methods for Large Language Models/tables/caf3f172234226fbb3f928898270863d182a82e04aa75557012790635ada422e.jpg)

![d090724b806d884f68251bdd7a7527c40ed0b272d828531953f54ba2d6d3da27.jpg](../iclr_results/1136_On the Optimization Landscape of Low Rank Adaptation Methods for Large Language Models/tables/d090724b806d884f68251bdd7a7527c40ed0b272d828531953f54ba2d6d3da27.jpg)

![e54a9a3af43794684b5ff48fe59912c4c69f891aa5ba01e91f71c28705b94295.jpg](../iclr_results/1136_On the Optimization Landscape of Low Rank Adaptation Methods for Large Language Models/tables/e54a9a3af43794684b5ff48fe59912c4c69f891aa5ba01e91f71c28705b94295.jpg)

![f7f829a67bbaf68d33423a5b6f48421e881ac8aa3032846b369a4701867d99d7.jpg](../iclr_results/1136_On the Optimization Landscape of Low Rank Adaptation Methods for Large Language Models/tables/f7f829a67bbaf68d33423a5b6f48421e881ac8aa3032846b369a4701867d99d7.jpg)

![fbf0bcadda03f3133655a5032f14edaf4f497cfec6a3d0a922fad8b01b67b177.jpg](../iclr_results/1136_On the Optimization Landscape of Low Rank Adaptation Methods for Large Language Models/tables/fbf0bcadda03f3133655a5032f14edaf4f497cfec6a3d0a922fad8b01b67b177.jpg)

## G-LLaVA: Solving Geometric Problem with Multi-Modal Large Language Model


### Images

![0ec225fd0e8cfad518e80ee3fc69fe00c2d3375b5b98ac9bc6813bacc048a44b.jpg](../iclr_results/1137_G-LLaVA_ Solving Geometric Problem with Multi-Modal Large Language Model/images/0ec225fd0e8cfad518e80ee3fc69fe00c2d3375b5b98ac9bc6813bacc048a44b.jpg)

![0f4ab4665858c9d04514481a905560901b906e35eec1b331113c69f1adc76e50.jpg](../iclr_results/1137_G-LLaVA_ Solving Geometric Problem with Multi-Modal Large Language Model/images/0f4ab4665858c9d04514481a905560901b906e35eec1b331113c69f1adc76e50.jpg)

![1aedc82f152be4b58793e40c48ac78e84b3661f8b751300013bd61332afeefe8.jpg](../iclr_results/1137_G-LLaVA_ Solving Geometric Problem with Multi-Modal Large Language Model/images/1aedc82f152be4b58793e40c48ac78e84b3661f8b751300013bd61332afeefe8.jpg)

![1b95670995f2b1be8367cbdc253e80e76b018159ba7996aeab9c7aa897be5b3f.jpg](../iclr_results/1137_G-LLaVA_ Solving Geometric Problem with Multi-Modal Large Language Model/images/1b95670995f2b1be8367cbdc253e80e76b018159ba7996aeab9c7aa897be5b3f.jpg)

![2def8b92fcf80d03dee5d1295cbbb2ae78b2ce790243963f7ffa1146bc39ef2c.jpg](../iclr_results/1137_G-LLaVA_ Solving Geometric Problem with Multi-Modal Large Language Model/images/2def8b92fcf80d03dee5d1295cbbb2ae78b2ce790243963f7ffa1146bc39ef2c.jpg)

![39f48ffb999396f85cba4e0e7f36c62dc196b2f7bb0a4d06fac00848a7eabe86.jpg](../iclr_results/1137_G-LLaVA_ Solving Geometric Problem with Multi-Modal Large Language Model/images/39f48ffb999396f85cba4e0e7f36c62dc196b2f7bb0a4d06fac00848a7eabe86.jpg)

![4b86f00f7ffd11f9498496efff0d1c983f68ca77bd05506a604a53a6a6c7e5c4.jpg](../iclr_results/1137_G-LLaVA_ Solving Geometric Problem with Multi-Modal Large Language Model/images/4b86f00f7ffd11f9498496efff0d1c983f68ca77bd05506a604a53a6a6c7e5c4.jpg)

![4e621a52fe166fb2bdb84bed4a367aac1a54c6fd982879b3b9b00e62ba3ee601.jpg](../iclr_results/1137_G-LLaVA_ Solving Geometric Problem with Multi-Modal Large Language Model/images/4e621a52fe166fb2bdb84bed4a367aac1a54c6fd982879b3b9b00e62ba3ee601.jpg)

![7698192e412cea9dd549b10df910b0b37b25a2ddc16db5ae56fec5245734a895.jpg](../iclr_results/1137_G-LLaVA_ Solving Geometric Problem with Multi-Modal Large Language Model/images/7698192e412cea9dd549b10df910b0b37b25a2ddc16db5ae56fec5245734a895.jpg)

![9c0b96dcfb53cc7241cb4dff528c6f113a1f26c58c8715271038ab3dc58426c4.jpg](../iclr_results/1137_G-LLaVA_ Solving Geometric Problem with Multi-Modal Large Language Model/images/9c0b96dcfb53cc7241cb4dff528c6f113a1f26c58c8715271038ab3dc58426c4.jpg)

![9c2bc65791d03cfce3ef9ed4a474c34d3b13bca83eb562bd9e0174e40b46f439.jpg](../iclr_results/1137_G-LLaVA_ Solving Geometric Problem with Multi-Modal Large Language Model/images/9c2bc65791d03cfce3ef9ed4a474c34d3b13bca83eb562bd9e0174e40b46f439.jpg)

![b241774e2dcbc6023c07945308ea9404cca8ac57d7c7b18deed1b2559d5d0e5b.jpg](../iclr_results/1137_G-LLaVA_ Solving Geometric Problem with Multi-Modal Large Language Model/images/b241774e2dcbc6023c07945308ea9404cca8ac57d7c7b18deed1b2559d5d0e5b.jpg)

![f07d6ef5492b31860a36e005a6cacb9189bdf09d95dec279f59392660ab49c9f.jpg](../iclr_results/1137_G-LLaVA_ Solving Geometric Problem with Multi-Modal Large Language Model/images/f07d6ef5492b31860a36e005a6cacb9189bdf09d95dec279f59392660ab49c9f.jpg)

### Tables

![21f2771a6781788d66721187c5d3772667b604382b9413473bacced12b5526e1.jpg](../iclr_results/1137_G-LLaVA_ Solving Geometric Problem with Multi-Modal Large Language Model/tables/21f2771a6781788d66721187c5d3772667b604382b9413473bacced12b5526e1.jpg)

![2829e31e80dc61638ea6d9912a521aad59d7dbc9a5b32a7567e7669eeb221198.jpg](../iclr_results/1137_G-LLaVA_ Solving Geometric Problem with Multi-Modal Large Language Model/tables/2829e31e80dc61638ea6d9912a521aad59d7dbc9a5b32a7567e7669eeb221198.jpg)

![446b28cd8d8881852497f616f71ccacd37806b02c9f2bf5d83b0b0e95c9468c2.jpg](../iclr_results/1137_G-LLaVA_ Solving Geometric Problem with Multi-Modal Large Language Model/tables/446b28cd8d8881852497f616f71ccacd37806b02c9f2bf5d83b0b0e95c9468c2.jpg)

![4b2049180585bd077c0882aa175bcf007fe00900517510d1011902cfa0b0344d.jpg](../iclr_results/1137_G-LLaVA_ Solving Geometric Problem with Multi-Modal Large Language Model/tables/4b2049180585bd077c0882aa175bcf007fe00900517510d1011902cfa0b0344d.jpg)

![8bf99e2fe920f230fcbeacb523dc07a3c51d50c294ce4b6b2bc9feec1e3ae2d3.jpg](../iclr_results/1137_G-LLaVA_ Solving Geometric Problem with Multi-Modal Large Language Model/tables/8bf99e2fe920f230fcbeacb523dc07a3c51d50c294ce4b6b2bc9feec1e3ae2d3.jpg)

![947b3a2d725f305ea004f62228ff4dbd82caaad4ccf292fe3752c3579c0c9bf5.jpg](../iclr_results/1137_G-LLaVA_ Solving Geometric Problem with Multi-Modal Large Language Model/tables/947b3a2d725f305ea004f62228ff4dbd82caaad4ccf292fe3752c3579c0c9bf5.jpg)

![95d227ac46ef19b43c5f0526b88e1f629fc88d08542cbadc1c72bf8321c50599.jpg](../iclr_results/1137_G-LLaVA_ Solving Geometric Problem with Multi-Modal Large Language Model/tables/95d227ac46ef19b43c5f0526b88e1f629fc88d08542cbadc1c72bf8321c50599.jpg)

![d58c580383b174b469df6b1b1debde0edbbd2cd3b0825adf174ffdf9df5589d4.jpg](../iclr_results/1137_G-LLaVA_ Solving Geometric Problem with Multi-Modal Large Language Model/tables/d58c580383b174b469df6b1b1debde0edbbd2cd3b0825adf174ffdf9df5589d4.jpg)

![da04748d8b0aea50b5f5847227c5e97f473438553753cc52239c40ede778533b.jpg](../iclr_results/1137_G-LLaVA_ Solving Geometric Problem with Multi-Modal Large Language Model/tables/da04748d8b0aea50b5f5847227c5e97f473438553753cc52239c40ede778533b.jpg)

## AdaFisher: Adaptive Second Order Optimization via Fisher Information


### Images

![1a0a4811b8dd52546a4287690865d836f8420b6e80d774bb426392dfe4485bcf.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/1a0a4811b8dd52546a4287690865d836f8420b6e80d774bb426392dfe4485bcf.jpg)

![1f39f0945db852cf687e8ed68a3d1919c20d6ab4211646855e5583a9ba463234.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/1f39f0945db852cf687e8ed68a3d1919c20d6ab4211646855e5583a9ba463234.jpg)

![20026642b690a97e7ae849a102b96f5023c04b4427555573cedb8fb74a4e8706.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/20026642b690a97e7ae849a102b96f5023c04b4427555573cedb8fb74a4e8706.jpg)

![24fae553e97f90fd07e8e153b423f17ec14e633d267e4f7ffb40e8632b2c59f8.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/24fae553e97f90fd07e8e153b423f17ec14e633d267e4f7ffb40e8632b2c59f8.jpg)

![35546a2bcb83c639845e93b250f7e4ba5cb264677bda55bf401d3350f1c65247.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/35546a2bcb83c639845e93b250f7e4ba5cb264677bda55bf401d3350f1c65247.jpg)

![391765225d752505d82e428247f696cccd6c8f57afeb4a1d02143650bb3cdaec.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/391765225d752505d82e428247f696cccd6c8f57afeb4a1d02143650bb3cdaec.jpg)

![462c7362690b9464036e7402987b5a62f94f4927d61f81b67b26c6baf19037d2.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/462c7362690b9464036e7402987b5a62f94f4927d61f81b67b26c6baf19037d2.jpg)

![60cd1cb12272e8a616a09f630c974c569f088884d9c65fb0097b65a1a84c1147.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/60cd1cb12272e8a616a09f630c974c569f088884d9c65fb0097b65a1a84c1147.jpg)

![7021d3c26e3f48217563c1656814e76861b86932264804ae2007934b1351d747.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/7021d3c26e3f48217563c1656814e76861b86932264804ae2007934b1351d747.jpg)

![71c48ab83f843b811de853cbfcb44f9a9e201d5b758c8e4e2d84cb8ef061dfd4.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/71c48ab83f843b811de853cbfcb44f9a9e201d5b758c8e4e2d84cb8ef061dfd4.jpg)

![773d1add7b65e4064fe10de42c7668317773ae416ae8f784396904c7d3b65a14.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/773d1add7b65e4064fe10de42c7668317773ae416ae8f784396904c7d3b65a14.jpg)

![78e6c41795ba7f42153810600798e96c9026810b9fb8501ba83c9ee587c4aede.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/78e6c41795ba7f42153810600798e96c9026810b9fb8501ba83c9ee587c4aede.jpg)

![8b7f4f3f2886614075d73569a45a22229ff52c6a99c0b6438bcdbcab116230cf.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/8b7f4f3f2886614075d73569a45a22229ff52c6a99c0b6438bcdbcab116230cf.jpg)

![8cf6801ee5994b53c8040b0baa54bfe5ff83c74b884e676a9ead92b35c4672de.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/8cf6801ee5994b53c8040b0baa54bfe5ff83c74b884e676a9ead92b35c4672de.jpg)

![bf6a647b2e867a32fe13e043376ad376cb07d01d861d0cc3380464600f15ab90.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/bf6a647b2e867a32fe13e043376ad376cb07d01d861d0cc3380464600f15ab90.jpg)

![c913de9653fe86e6ca4f5d65487de966841480605b9f29fc284c35f71bebb89c.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/c913de9653fe86e6ca4f5d65487de966841480605b9f29fc284c35f71bebb89c.jpg)

![d7979bcf25301f395b05c4af73933da3ef38a1359d3201d2bd70b26842933248.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/d7979bcf25301f395b05c4af73933da3ef38a1359d3201d2bd70b26842933248.jpg)

![dda032aed41859c276904d4607d0b173dd7127aee90ed4172f098a1110a765bb.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/dda032aed41859c276904d4607d0b173dd7127aee90ed4172f098a1110a765bb.jpg)

![de6797194a50e696590d743947ca91f843ade9382506ba0cfe9a3d1bf4a93fbe.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/de6797194a50e696590d743947ca91f843ade9382506ba0cfe9a3d1bf4a93fbe.jpg)

![e25fe3c3611aad5d54ba7e524b3b785b7c2fcf320286a23eabb2ab2842b7b8eb.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/e25fe3c3611aad5d54ba7e524b3b785b7c2fcf320286a23eabb2ab2842b7b8eb.jpg)

![e94581b877c89cc095132dcfa220b15c0568e8de0892e19d223a8fc3ba787fc0.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/e94581b877c89cc095132dcfa220b15c0568e8de0892e19d223a8fc3ba787fc0.jpg)

![f78ab10236e686db1cd2b5a4e0e8311f3799e9cdf92969f1e1972dc43fca280b.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/f78ab10236e686db1cd2b5a4e0e8311f3799e9cdf92969f1e1972dc43fca280b.jpg)

![f8c4d707633b65c877bb1c03896008535eeb199d1ed08864a5677cbd0aeb1047.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/f8c4d707633b65c877bb1c03896008535eeb199d1ed08864a5677cbd0aeb1047.jpg)

![ff8d45577f5cb7be099176b1b12c5548cf2e8395685a883ea990abbcef10dc6a.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/ff8d45577f5cb7be099176b1b12c5548cf2e8395685a883ea990abbcef10dc6a.jpg)

![ff962892069db8828f419cd420f33b0bb5add2190455a2a43de6b7c7a6d29232.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/images/ff962892069db8828f419cd420f33b0bb5add2190455a2a43de6b7c7a6d29232.jpg)

### Tables

![3353d2b987ee187e1c7a43c104ea0b3312e7696059c0c0b60ec4e7ae58b6869f.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/tables/3353d2b987ee187e1c7a43c104ea0b3312e7696059c0c0b60ec4e7ae58b6869f.jpg)

![3757c512c26319ac5ace53b2c4d348aa5a35ad098d6152200f0ef992cd0f11b8.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/tables/3757c512c26319ac5ace53b2c4d348aa5a35ad098d6152200f0ef992cd0f11b8.jpg)

![3b81da92a50b351c88dd59f3ebc36c826b84935a806a85ec92f76a46f4c8560c.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/tables/3b81da92a50b351c88dd59f3ebc36c826b84935a806a85ec92f76a46f4c8560c.jpg)

![3bd1c06cb80aff88f6cf64fcc53c65c08aef9f8f3d370753ed444ed805fed406.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/tables/3bd1c06cb80aff88f6cf64fcc53c65c08aef9f8f3d370753ed444ed805fed406.jpg)

![492292fd1db44e19b36c7703ac175433447240182f3f2434f3ea0cc39f407899.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/tables/492292fd1db44e19b36c7703ac175433447240182f3f2434f3ea0cc39f407899.jpg)

![4e09e599a883030e4e932fa5a1a51666328deb38bd72c5a0c68b9a743706f346.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/tables/4e09e599a883030e4e932fa5a1a51666328deb38bd72c5a0c68b9a743706f346.jpg)

![6d65efac5ac109613ae15ad86824204303bf340a60495104f0c2643f808e5f07.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/tables/6d65efac5ac109613ae15ad86824204303bf340a60495104f0c2643f808e5f07.jpg)

![7929448bd7a2b7848082e0b7e3f09c3baacbf5e1418e8b89e6e8fcc1fbeb1830.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/tables/7929448bd7a2b7848082e0b7e3f09c3baacbf5e1418e8b89e6e8fcc1fbeb1830.jpg)

![83e486eaddecfc81ffe003e2dcf1cb001de8f82357b850d09eea51c1299b42ef.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/tables/83e486eaddecfc81ffe003e2dcf1cb001de8f82357b850d09eea51c1299b42ef.jpg)

![8ca684f895cadf5f6e1e3d95898bdbb2dfbef6fe2e976269127f6f28a1d8fe83.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/tables/8ca684f895cadf5f6e1e3d95898bdbb2dfbef6fe2e976269127f6f28a1d8fe83.jpg)

![a8545d45f787232cfa655dd67e0d62bee19ef93f7d73c90c520fae0bc03b9158.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/tables/a8545d45f787232cfa655dd67e0d62bee19ef93f7d73c90c520fae0bc03b9158.jpg)

![aa284c3fc139926e64563a61ef97810af38d05896a0367d501e444bde740d8b3.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/tables/aa284c3fc139926e64563a61ef97810af38d05896a0367d501e444bde740d8b3.jpg)

![be4b33de898c4592221a6df758da7edc6153928689036085dc819b187a3c44f8.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/tables/be4b33de898c4592221a6df758da7edc6153928689036085dc819b187a3c44f8.jpg)

![c0de682bda59da259c58bb3dfe10f00441fca678853414abdf8e8cb9be451b30.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/tables/c0de682bda59da259c58bb3dfe10f00441fca678853414abdf8e8cb9be451b30.jpg)

![c75a33ce92ac236842b6ac179171cf2516a31c02f9cd3250b667a03d3c7936bc.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/tables/c75a33ce92ac236842b6ac179171cf2516a31c02f9cd3250b667a03d3c7936bc.jpg)

![dcf1a9b9d2bab513865e767ff4891e6751b0216d44fb7ad643e6047abde7f396.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/tables/dcf1a9b9d2bab513865e767ff4891e6751b0216d44fb7ad643e6047abde7f396.jpg)

![efb2b2d3aaf4a04b3ece7f329ffc30837d6f0b32765421e17a95158b8073ba28.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/tables/efb2b2d3aaf4a04b3ece7f329ffc30837d6f0b32765421e17a95158b8073ba28.jpg)

![fdca334daf58bcaa095debfe80eeaec58637b6ab5c03a53b9be2d7cdd979339a.jpg](../iclr_results/1138_AdaFisher_ Adaptive Second Order Optimization via Fisher Information/tables/fdca334daf58bcaa095debfe80eeaec58637b6ab5c03a53b9be2d7cdd979339a.jpg)

## Learning from Imperfect  Human Feedback: A Tale from Corruption-Robust Dueling


### Images

![08a07b0d05a0c532aba7af31a08cdd363495fcc9ca9a9681c159bc29dc96faad.jpg](../iclr_results/1139_Learning from Imperfect  Human Feedback_ A Tale from Corruption-Robust Dueling/images/08a07b0d05a0c532aba7af31a08cdd363495fcc9ca9a9681c159bc29dc96faad.jpg)

![bbaddabf78bea11b5954fbcf60148a76ff13f2a94b5177430dc462b28fbec426.jpg](../iclr_results/1139_Learning from Imperfect  Human Feedback_ A Tale from Corruption-Robust Dueling/images/bbaddabf78bea11b5954fbcf60148a76ff13f2a94b5177430dc462b28fbec426.jpg)

![d3cea88aca49b25f149b8d0864722b130000638e01fdae055e1fa85285bf4713.jpg](../iclr_results/1139_Learning from Imperfect  Human Feedback_ A Tale from Corruption-Robust Dueling/images/d3cea88aca49b25f149b8d0864722b130000638e01fdae055e1fa85285bf4713.jpg)

## mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models


### Images

![20569ef3f01b1f2fd082d0691929a20a4f3e262a232db1820ead51fc6d52382d.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/images/20569ef3f01b1f2fd082d0691929a20a4f3e262a232db1820ead51fc6d52382d.jpg)

![2559da0c53599dd6155513c53cc2cc48c6dbda0c698f5ee59409ecc60b806fc9.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/images/2559da0c53599dd6155513c53cc2cc48c6dbda0c698f5ee59409ecc60b806fc9.jpg)

![2a2434ad77ab1c3b5a7b57523fc3f00279bba1195052ea67905e3917efb3f07d.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/images/2a2434ad77ab1c3b5a7b57523fc3f00279bba1195052ea67905e3917efb3f07d.jpg)

![2d59da1c7f69fa4599914a992d468f21bb95aa73b5fb21d812231dfd04f7b025.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/images/2d59da1c7f69fa4599914a992d468f21bb95aa73b5fb21d812231dfd04f7b025.jpg)

![596b8b857b73fe916befbcb10686012acbd0cc7446726cfacc6446ffe9fef762.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/images/596b8b857b73fe916befbcb10686012acbd0cc7446726cfacc6446ffe9fef762.jpg)

![5c42e530c4f7673d6cdd9749095fa9a6b7f9ee5f1a0d9aa0cc72d240c5917169.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/images/5c42e530c4f7673d6cdd9749095fa9a6b7f9ee5f1a0d9aa0cc72d240c5917169.jpg)

![76d300fe7af40bee5967b1069abfbb0623570cf069c9e936f40fa87f3530c61b.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/images/76d300fe7af40bee5967b1069abfbb0623570cf069c9e936f40fa87f3530c61b.jpg)

![789d2774a20ed75e0427c8c044e52b531cbf988910689ed2006f9b3c14dc2ec4.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/images/789d2774a20ed75e0427c8c044e52b531cbf988910689ed2006f9b3c14dc2ec4.jpg)

![8ffab79870f6862ab4d4ae0bf645cb6c478cdf1b77290cd6afd331a5079da02d.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/images/8ffab79870f6862ab4d4ae0bf645cb6c478cdf1b77290cd6afd331a5079da02d.jpg)

![a6c33c550d79494d3b43651bdf3cd4fe7b9fa5bd5b8d169389eae5011f508014.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/images/a6c33c550d79494d3b43651bdf3cd4fe7b9fa5bd5b8d169389eae5011f508014.jpg)

![b49de68e6db5a8d51ee87c784121600b1e5f75b407bc5004d9cb5763f3c4bd1a.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/images/b49de68e6db5a8d51ee87c784121600b1e5f75b407bc5004d9cb5763f3c4bd1a.jpg)

![b56390051d17976b5aa8982692d245aec0d61ba6519b0cae12bacdd8b120ad14.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/images/b56390051d17976b5aa8982692d245aec0d61ba6519b0cae12bacdd8b120ad14.jpg)

![bd42f79f0a40208708d84b94cc64741292ca57789ca21a1bf97850960bfaf034.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/images/bd42f79f0a40208708d84b94cc64741292ca57789ca21a1bf97850960bfaf034.jpg)

![cd22da8a1f656cbf6ec6da968f4f9e838e39b329fd6d006c0b791cb406d6a268.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/images/cd22da8a1f656cbf6ec6da968f4f9e838e39b329fd6d006c0b791cb406d6a268.jpg)

![d15621f6f3e2e5d06d8b089a34fc268e9aae375cf079fd215c14c8701939c0f2.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/images/d15621f6f3e2e5d06d8b089a34fc268e9aae375cf079fd215c14c8701939c0f2.jpg)

![d5b892419190fed4b5cc1937d62c725ac2e9acd963640817e6e1a6b0bbd8c732.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/images/d5b892419190fed4b5cc1937d62c725ac2e9acd963640817e6e1a6b0bbd8c732.jpg)

### Tables

![2fb5f8a9b96bd2f156a514d4f24d462531041fd7c6438630445a2a3bb397b430.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/tables/2fb5f8a9b96bd2f156a514d4f24d462531041fd7c6438630445a2a3bb397b430.jpg)

![3c5bea7f666d0648b50bf7ecfcbc77b7488dc2513ca695d4f868d8a2fb83e368.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/tables/3c5bea7f666d0648b50bf7ecfcbc77b7488dc2513ca695d4f868d8a2fb83e368.jpg)

![5867be3ad8359e554fe8a259ee90b2c483570f8eb6ec45a55bc1d7ac641a7df9.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/tables/5867be3ad8359e554fe8a259ee90b2c483570f8eb6ec45a55bc1d7ac641a7df9.jpg)

![590fa0f7775f4a08d375e89781365ce1f07e161c4ce5e91b2dd2216690660417.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/tables/590fa0f7775f4a08d375e89781365ce1f07e161c4ce5e91b2dd2216690660417.jpg)

![5ef2d8c66f3767f412343789fc3803efea75b0f08765dbfaf4e1894782c2d10c.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/tables/5ef2d8c66f3767f412343789fc3803efea75b0f08765dbfaf4e1894782c2d10c.jpg)

![812cfe71205eef577d2af1706579adbd80af94bb7d87ac8d4aad1fc91dc79a88.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/tables/812cfe71205eef577d2af1706579adbd80af94bb7d87ac8d4aad1fc91dc79a88.jpg)

![937d36ed20f5a915b15f49b012efab829fc0c7896258b0ac74be85c88633f84b.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/tables/937d36ed20f5a915b15f49b012efab829fc0c7896258b0ac74be85c88633f84b.jpg)

![a5329b0eeab52fb013629cdd8a4254f4f83113e33367d9d5f9e607b5e10b5e45.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/tables/a5329b0eeab52fb013629cdd8a4254f4f83113e33367d9d5f9e607b5e10b5e45.jpg)

![aa35e4aef1e8b5d67da0475f98c90963ddc6aa75673519e9d6d9fb5cd1f15c72.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/tables/aa35e4aef1e8b5d67da0475f98c90963ddc6aa75673519e9d6d9fb5cd1f15c72.jpg)

![ca314a5b6d315fe66987719827c94111613a9959eda3ff8d67041c50d5dab4d5.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/tables/ca314a5b6d315fe66987719827c94111613a9959eda3ff8d67041c50d5dab4d5.jpg)

![d10d73851909eb8e834bd3e4f6607e4d65106ff2c648ef1a9283c4dac40c4f1e.jpg](../iclr_results/1140_mPLUG-Owl3_ Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models/tables/d10d73851909eb8e834bd3e4f6607e4d65106ff2c648ef1a9283c4dac40c4f1e.jpg)

## UV-Attack: Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping


### Images

![1164cf98121c05445854bad28d7f40d92aacfbe7bbbd33b263b88e0b9f922ce7.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/images/1164cf98121c05445854bad28d7f40d92aacfbe7bbbd33b263b88e0b9f922ce7.jpg)

![136657ca5c96bc539969673ac09b98fa230d004a5a3da6c4b63bced9054d9990.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/images/136657ca5c96bc539969673ac09b98fa230d004a5a3da6c4b63bced9054d9990.jpg)

![1f83c43d7e3b7acc081b2a08fbee910f7904ea41d13379394ea066eb781313f9.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/images/1f83c43d7e3b7acc081b2a08fbee910f7904ea41d13379394ea066eb781313f9.jpg)

![254757ffe2c8a25b9e00d8722c33bce450c2f854fa408657e6ec7ffcc59efe47.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/images/254757ffe2c8a25b9e00d8722c33bce450c2f854fa408657e6ec7ffcc59efe47.jpg)

![3078a786c38927c69fc1a6bcd05688087812d144817b730aa09811562c9d3126.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/images/3078a786c38927c69fc1a6bcd05688087812d144817b730aa09811562c9d3126.jpg)

![3fcedda8edb4249f224dd0d9cf8fe17b054ffa95caefcf00bc47d30fb91783f6.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/images/3fcedda8edb4249f224dd0d9cf8fe17b054ffa95caefcf00bc47d30fb91783f6.jpg)

![45be3c7185f2b965761bf7534885fb3d7a13814595ca2157ea569f6e4fc26c95.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/images/45be3c7185f2b965761bf7534885fb3d7a13814595ca2157ea569f6e4fc26c95.jpg)

![4d1c5a1b1ece5594048ce61ef2c005ac69ff42c74579cdbdc4b47bb7e1ebfea5.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/images/4d1c5a1b1ece5594048ce61ef2c005ac69ff42c74579cdbdc4b47bb7e1ebfea5.jpg)

![57a7b356d83f678a661b69e950ec9d47d74c5e2684cc7b565754c6eda2f6ab98.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/images/57a7b356d83f678a661b69e950ec9d47d74c5e2684cc7b565754c6eda2f6ab98.jpg)

![69672d7548910f4032ca1e6186c422fbed15f28eeb7600119d1096dd681808f9.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/images/69672d7548910f4032ca1e6186c422fbed15f28eeb7600119d1096dd681808f9.jpg)

![6dc7e615e4b1dceb9ece8caa80d0902bfe8bbb6182036c23a3ffbc1e2c146bb4.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/images/6dc7e615e4b1dceb9ece8caa80d0902bfe8bbb6182036c23a3ffbc1e2c146bb4.jpg)

![70f457467bc217801460c16405d4d093f3ad224c76de9ea976050d4f7c684e53.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/images/70f457467bc217801460c16405d4d093f3ad224c76de9ea976050d4f7c684e53.jpg)

![7acf03f299b879e7577b98d38e674ddf9c21e71cafaba46883578d9818d99d44.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/images/7acf03f299b879e7577b98d38e674ddf9c21e71cafaba46883578d9818d99d44.jpg)

![813e2f58b8d137a7937c6fb2046314b87f45aeaca332a59bcf874af118fb7f87.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/images/813e2f58b8d137a7937c6fb2046314b87f45aeaca332a59bcf874af118fb7f87.jpg)

![977f58a705a782f55db4dd099fca3464971d02edbbc12df1b053fb2033005e71.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/images/977f58a705a782f55db4dd099fca3464971d02edbbc12df1b053fb2033005e71.jpg)

![af20a546fe1671b180063d1527099f77633e68947a62dd047f97f3afbeebedad.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/images/af20a546fe1671b180063d1527099f77633e68947a62dd047f97f3afbeebedad.jpg)

![bdb390a0ab3906b21385a5bcd4e795e6b2139530281ec4a0b68434ebd7e606fe.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/images/bdb390a0ab3906b21385a5bcd4e795e6b2139530281ec4a0b68434ebd7e606fe.jpg)

![c0f293686a968b32d0fb9481a899784b26801bbe628755528270a34eacb807d4.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/images/c0f293686a968b32d0fb9481a899784b26801bbe628755528270a34eacb807d4.jpg)

![c5a609580e7c70ff025823efbcaeded76e9f0c798a0bafb3a24c3fb8529dfbdb.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/images/c5a609580e7c70ff025823efbcaeded76e9f0c798a0bafb3a24c3fb8529dfbdb.jpg)

![d0b4dea3499a8ecdf3a29fd5e242d1b733df10a44f1bf56e3bd82b6ac4e5633d.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/images/d0b4dea3499a8ecdf3a29fd5e242d1b733df10a44f1bf56e3bd82b6ac4e5633d.jpg)

![d4bbd2ab800e25ef7d85f9a9d6074d925dd1e8cb99b3d7778ed70a327920158b.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/images/d4bbd2ab800e25ef7d85f9a9d6074d925dd1e8cb99b3d7778ed70a327920158b.jpg)

![e1d9abf05969afd4609d22e60ce393e92983fd8557e9ff5e2c74ae232ed88809.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/images/e1d9abf05969afd4609d22e60ce393e92983fd8557e9ff5e2c74ae232ed88809.jpg)

### Tables

![593f578763a0c12072f17bc055fb75a9cc5e8862842597bddac61cf0c5e6ef95.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/tables/593f578763a0c12072f17bc055fb75a9cc5e8862842597bddac61cf0c5e6ef95.jpg)

![a0c82f7e87b142b000d8006d8c7689d79a9b1b222cc3f632c77e37daed59bd73.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/tables/a0c82f7e87b142b000d8006d8c7689d79a9b1b222cc3f632c77e37daed59bd73.jpg)

![ae5b5601c9ceec91e9e616a4ec323cd8cb05619d20a991e5dc753939c017586f.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/tables/ae5b5601c9ceec91e9e616a4ec323cd8cb05619d20a991e5dc753939c017586f.jpg)

![b49a6dcb8ba0582b81ba114d9bdba6096977758dfdef10927bffd0e73c92ae2b.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/tables/b49a6dcb8ba0582b81ba114d9bdba6096977758dfdef10927bffd0e73c92ae2b.jpg)

![ea8e30ba393e3a5d580c04c004c0c04a6dbe97c5dbaec5db2e65dacaad0e8461.jpg](../iclr_results/1141_UV-Attack_ Physical-World Adversarial Attacks on Person Detection via Dynamic-NeRF-based UV Mapping/tables/ea8e30ba393e3a5d580c04c004c0c04a6dbe97c5dbaec5db2e65dacaad0e8461.jpg)

## Convergence of Score-Based Discrete Diffusion Models: A Discrete-Time Analysis


### Images

![9e6e9c15ba5cbde7f81780fc9b04a672ef1fe0bd14d0f67c32057670d3b45eea.jpg](../iclr_results/1142_Convergence of Score-Based Discrete Diffusion Models_ A Discrete-Time Analysis/images/9e6e9c15ba5cbde7f81780fc9b04a672ef1fe0bd14d0f67c32057670d3b45eea.jpg)

### Tables

![a289a72c6eea148a696ce696ad68825d388d6b0e2303fb9d1a05e9fd9aea811e.jpg](../iclr_results/1142_Convergence of Score-Based Discrete Diffusion Models_ A Discrete-Time Analysis/tables/a289a72c6eea148a696ce696ad68825d388d6b0e2303fb9d1a05e9fd9aea811e.jpg)

## Logicbreaks: A Framework for Understanding Subversion of Rule-based Inference


### Images

![01da6baab504492ef38d9504cb40f055aad9702d57fde06f54c5c3e2499d6506.jpg](../iclr_results/1143_Logicbreaks_ A Framework for Understanding Subversion of Rule-based Inference/images/01da6baab504492ef38d9504cb40f055aad9702d57fde06f54c5c3e2499d6506.jpg)

![10f959c077b0020650126097f2ee9981a5fc8c5e7676e5aa042cd192e2d81ff8.jpg](../iclr_results/1143_Logicbreaks_ A Framework for Understanding Subversion of Rule-based Inference/images/10f959c077b0020650126097f2ee9981a5fc8c5e7676e5aa042cd192e2d81ff8.jpg)

![158e69f1a2482451c3a714c078ead7329f6d6740002d41df46ad9ed21ffd5ad9.jpg](../iclr_results/1143_Logicbreaks_ A Framework for Understanding Subversion of Rule-based Inference/images/158e69f1a2482451c3a714c078ead7329f6d6740002d41df46ad9ed21ffd5ad9.jpg)

![7542290d281a2298667458760d3f8b370db8edde0a09e2e53b49956bfce00d2b.jpg](../iclr_results/1143_Logicbreaks_ A Framework for Understanding Subversion of Rule-based Inference/images/7542290d281a2298667458760d3f8b370db8edde0a09e2e53b49956bfce00d2b.jpg)

![84368822661b34473eb84cc63d0e96a054d9cb21522a16d63989e968e28d75f5.jpg](../iclr_results/1143_Logicbreaks_ A Framework for Understanding Subversion of Rule-based Inference/images/84368822661b34473eb84cc63d0e96a054d9cb21522a16d63989e968e28d75f5.jpg)

![87120fabf6860735a5ccf075eaf8f5a9ab987fb8074b527cc1e11e7b234ca21d.jpg](../iclr_results/1143_Logicbreaks_ A Framework for Understanding Subversion of Rule-based Inference/images/87120fabf6860735a5ccf075eaf8f5a9ab987fb8074b527cc1e11e7b234ca21d.jpg)

![9ed2e8e0e04e790824f136c6c59b4f89c7b4b352b2f8fc4fd7b452d9b3a66bb8.jpg](../iclr_results/1143_Logicbreaks_ A Framework for Understanding Subversion of Rule-based Inference/images/9ed2e8e0e04e790824f136c6c59b4f89c7b4b352b2f8fc4fd7b452d9b3a66bb8.jpg)

![a45df2549fc5813d82a5b268bea06d31353b6ca7645e5da8681ab77f4d70982c.jpg](../iclr_results/1143_Logicbreaks_ A Framework for Understanding Subversion of Rule-based Inference/images/a45df2549fc5813d82a5b268bea06d31353b6ca7645e5da8681ab77f4d70982c.jpg)

![da60391121385076ec843c2aca1c301439acdcc3284464df083e00cc0f724060.jpg](../iclr_results/1143_Logicbreaks_ A Framework for Understanding Subversion of Rule-based Inference/images/da60391121385076ec843c2aca1c301439acdcc3284464df083e00cc0f724060.jpg)

### Tables

![00ec0f2adf701d2cdfc01a776e7ca9b52ad55a7fe1f671e74c7eeff691576b4a.jpg](../iclr_results/1143_Logicbreaks_ A Framework for Understanding Subversion of Rule-based Inference/tables/00ec0f2adf701d2cdfc01a776e7ca9b52ad55a7fe1f671e74c7eeff691576b4a.jpg)

![860a80d34aee52a4dfd548a29ddfe4c04269f362cb8e6472d13f947cf1489cd2.jpg](../iclr_results/1143_Logicbreaks_ A Framework for Understanding Subversion of Rule-based Inference/tables/860a80d34aee52a4dfd548a29ddfe4c04269f362cb8e6472d13f947cf1489cd2.jpg)

![ac2278e1f36a1a3dc490b0b36e894b58c1cb5924173e874cd47d6ce9e28bac68.jpg](../iclr_results/1143_Logicbreaks_ A Framework for Understanding Subversion of Rule-based Inference/tables/ac2278e1f36a1a3dc490b0b36e894b58c1cb5924173e874cd47d6ce9e28bac68.jpg)

![c844aca4f9fe83227fb1fbd9f2a50ddfb6cc69466c59a1970b71ecbdd3002592.jpg](../iclr_results/1143_Logicbreaks_ A Framework for Understanding Subversion of Rule-based Inference/tables/c844aca4f9fe83227fb1fbd9f2a50ddfb6cc69466c59a1970b71ecbdd3002592.jpg)

![e86a8d91efcba7d95b93044b40d821cd28ac46fe965b595c05299390a92b44f7.jpg](../iclr_results/1143_Logicbreaks_ A Framework for Understanding Subversion of Rule-based Inference/tables/e86a8d91efcba7d95b93044b40d821cd28ac46fe965b595c05299390a92b44f7.jpg)

![eca786224199114d16bd31ca4fc7955e51dbb94c7842e8a918b2be2b905a1498.jpg](../iclr_results/1143_Logicbreaks_ A Framework for Understanding Subversion of Rule-based Inference/tables/eca786224199114d16bd31ca4fc7955e51dbb94c7842e8a918b2be2b905a1498.jpg)

## ESE: Espresso Sentence Embeddings


### Images

![43b833d5b9304da22e3e1ac95489e8c85fb21c00e0a9d5b00e3898aab95305bf.jpg](../iclr_results/1144_ESE_ Espresso Sentence Embeddings/images/43b833d5b9304da22e3e1ac95489e8c85fb21c00e0a9d5b00e3898aab95305bf.jpg)

![542097709d5ec6f9f55928f97ccd0c9923537c31976bbbbb5250e4a3f5411fe1.jpg](../iclr_results/1144_ESE_ Espresso Sentence Embeddings/images/542097709d5ec6f9f55928f97ccd0c9923537c31976bbbbb5250e4a3f5411fe1.jpg)

![68800f80cedd886a0bc8f384ef07ce823bc363130191f08e6c9eaf36fc0a9d57.jpg](../iclr_results/1144_ESE_ Espresso Sentence Embeddings/images/68800f80cedd886a0bc8f384ef07ce823bc363130191f08e6c9eaf36fc0a9d57.jpg)

![74e102f6efb713a802257081a7be1174a37553e1c7d00c6ab928c82447b3dae8.jpg](../iclr_results/1144_ESE_ Espresso Sentence Embeddings/images/74e102f6efb713a802257081a7be1174a37553e1c7d00c6ab928c82447b3dae8.jpg)

![862529d035dc6af61d697e58333b1d7a62c818e4d373b51d1619ed5fb6873222.jpg](../iclr_results/1144_ESE_ Espresso Sentence Embeddings/images/862529d035dc6af61d697e58333b1d7a62c818e4d373b51d1619ed5fb6873222.jpg)

![bf428e89209f10bfb7c77620b102a4ffe802d8d643eaa0c49f2da399970a42da.jpg](../iclr_results/1144_ESE_ Espresso Sentence Embeddings/images/bf428e89209f10bfb7c77620b102a4ffe802d8d643eaa0c49f2da399970a42da.jpg)

![e3262b82d623c412506718052e7f7f372bf72be39e5a91781bceae93ecf74104.jpg](../iclr_results/1144_ESE_ Espresso Sentence Embeddings/images/e3262b82d623c412506718052e7f7f372bf72be39e5a91781bceae93ecf74104.jpg)

### Tables

![84cf1ac8b49c8747331737526afdddc54edd3a9bc231e03bb9e7663c58da0e0a.jpg](../iclr_results/1144_ESE_ Espresso Sentence Embeddings/tables/84cf1ac8b49c8747331737526afdddc54edd3a9bc231e03bb9e7663c58da0e0a.jpg)

![c942552e50828c2b7dbd5109f0f376bbb990038ec8b8daef89f731725723c02f.jpg](../iclr_results/1144_ESE_ Espresso Sentence Embeddings/tables/c942552e50828c2b7dbd5109f0f376bbb990038ec8b8daef89f731725723c02f.jpg)

![fef8b02c95e62cea19130e99cef09421dd2684e9653a0535c6533902731527f6.jpg](../iclr_results/1144_ESE_ Espresso Sentence Embeddings/tables/fef8b02c95e62cea19130e99cef09421dd2684e9653a0535c6533902731527f6.jpg)

## Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling


### Images

![07558b5251c7e7f8eaaea12875bf5208e358eac8460a52ddfca758f5c94ac619.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/images/07558b5251c7e7f8eaaea12875bf5208e358eac8460a52ddfca758f5c94ac619.jpg)

![18f0a85aad3e71725a24ae16cc5e885a95f8c7ae115c6663e021c4ef317f3797.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/images/18f0a85aad3e71725a24ae16cc5e885a95f8c7ae115c6663e021c4ef317f3797.jpg)

![1db069258c163cb37a979b7a519b184938c504f7f9553636ad0667407e8cefa6.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/images/1db069258c163cb37a979b7a519b184938c504f7f9553636ad0667407e8cefa6.jpg)

![23e3d496b5e529d2d28853c334baa087180b8c9b47d6e518c86b6008425b4804.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/images/23e3d496b5e529d2d28853c334baa087180b8c9b47d6e518c86b6008425b4804.jpg)

![2a9ec172b508cb05eace75093ed08e6908084f17e447c5faeae2d1fee904fd6a.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/images/2a9ec172b508cb05eace75093ed08e6908084f17e447c5faeae2d1fee904fd6a.jpg)

![30da49eaa9a3d567bee3738acc9a51da003b17f94323c466fe1926cdd6d99810.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/images/30da49eaa9a3d567bee3738acc9a51da003b17f94323c466fe1926cdd6d99810.jpg)

![3e627881d7e425c61e5f2e5c9341e0b169ca1b7069d091b5375e3ceed8144d0f.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/images/3e627881d7e425c61e5f2e5c9341e0b169ca1b7069d091b5375e3ceed8144d0f.jpg)

![51ddc98d1632e91da38d835ec322ece2c3944df14a27f81a2d66622ef6422d8c.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/images/51ddc98d1632e91da38d835ec322ece2c3944df14a27f81a2d66622ef6422d8c.jpg)

![69fed11920cda2d337c528e5afe6716e07def009857f5104276b784d8f7cdacd.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/images/69fed11920cda2d337c528e5afe6716e07def009857f5104276b784d8f7cdacd.jpg)

![830926118ed927eb7f38017de417e3649658d4b71d2ae71fcdad75a64446c7f5.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/images/830926118ed927eb7f38017de417e3649658d4b71d2ae71fcdad75a64446c7f5.jpg)

![a0c804b15cb6418a912093599ea854a0684744e6c2176ef7e7f6a7b923d8c0a2.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/images/a0c804b15cb6418a912093599ea854a0684744e6c2176ef7e7f6a7b923d8c0a2.jpg)

![acebbfdda3f82ab41eaa7cd63b7e998e6a03fbc2fd2b442abf07df128a1a1055.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/images/acebbfdda3f82ab41eaa7cd63b7e998e6a03fbc2fd2b442abf07df128a1a1055.jpg)

![b4faa34e5ba94e19a035122fed72431301871dcb3ca632341badcd5704a0e21f.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/images/b4faa34e5ba94e19a035122fed72431301871dcb3ca632341badcd5704a0e21f.jpg)

![b577756566856ed2b85d34ce0de40c79d417565a4257b4e22b917a1e688a4fd0.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/images/b577756566856ed2b85d34ce0de40c79d417565a4257b4e22b917a1e688a4fd0.jpg)

![bd4cfd33634ecdb48ff65106b6abf1e5ded5605034952773252b3afad232555a.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/images/bd4cfd33634ecdb48ff65106b6abf1e5ded5605034952773252b3afad232555a.jpg)

![be8bb8e69d573fe6ad432c920112e828746f1f544208b0d015c5330f017d3737.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/images/be8bb8e69d573fe6ad432c920112e828746f1f544208b0d015c5330f017d3737.jpg)

![c595a4ca50b83adfcee2707451c41c35fb8eb39466a14a18be691a65727650c4.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/images/c595a4ca50b83adfcee2707451c41c35fb8eb39466a14a18be691a65727650c4.jpg)

![e5840dac53d10f75b7b764c32137156fc700dce45ffb0713fcfc70019b9fb50c.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/images/e5840dac53d10f75b7b764c32137156fc700dce45ffb0713fcfc70019b9fb50c.jpg)

![e7ef1b738135b5d8c97f50475c21d4eba5cc61d960f607a5de35f317cdd5717f.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/images/e7ef1b738135b5d8c97f50475c21d4eba5cc61d960f607a5de35f317cdd5717f.jpg)

![e9486a7c1869d5e90d53145490ba23835d9d2a7b4e8e589325349a603554af2f.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/images/e9486a7c1869d5e90d53145490ba23835d9d2a7b4e8e589325349a603554af2f.jpg)

### Tables

![27aa4777f1b7238e180e8f1fb064cdedba3158bcaaaf8d60380bbbb17e09e395.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/tables/27aa4777f1b7238e180e8f1fb064cdedba3158bcaaaf8d60380bbbb17e09e395.jpg)

![463aaed9821d893f6332d565600ba9c4f04b0a1291ffa780be444d748a275d02.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/tables/463aaed9821d893f6332d565600ba9c4f04b0a1291ffa780be444d748a275d02.jpg)

![54d018d311b798f2ef4c05fcff3e22ca97a36fa112aff7788832ebb4ee612b9f.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/tables/54d018d311b798f2ef4c05fcff3e22ca97a36fa112aff7788832ebb4ee612b9f.jpg)

![77293ff7acfdc1b53fd64e893b950b32b5fd0394fd2030656f56dd7077c8f5e2.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/tables/77293ff7acfdc1b53fd64e893b950b32b5fd0394fd2030656f56dd7077c8f5e2.jpg)

![9410983613315d3381952416d2d14cf186fd7cf8c2395a653c620b85bfb680a5.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/tables/9410983613315d3381952416d2d14cf186fd7cf8c2395a653c620b85bfb680a5.jpg)

![a6a882dd76fe3bac66c14e4b56728e2290269b19ebdb38c8ee89846d26c2241c.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/tables/a6a882dd76fe3bac66c14e4b56728e2290269b19ebdb38c8ee89846d26c2241c.jpg)

![a9612f3fe730158aab17fae9b341994d8854d4682b89b66e614b455b2e80019c.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/tables/a9612f3fe730158aab17fae9b341994d8854d4682b89b66e614b455b2e80019c.jpg)

![cc6015352349c481fc76a9c1612a01c06373c7437d1ff6475c8a7f9514f296f6.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/tables/cc6015352349c481fc76a9c1612a01c06373c7437d1ff6475c8a7f9514f296f6.jpg)

![d60e18273dfafa4bcf3f8fd4bbd99f408eac630ffbc06a070bfb6c645df3eee3.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/tables/d60e18273dfafa4bcf3f8fd4bbd99f408eac630ffbc06a070bfb6c645df3eee3.jpg)

![d7d7cef92d26cf598e2b917e9f58fd9615c9d49804f5ed470aa650105d3a908b.jpg](../iclr_results/1145_Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling/tables/d7d7cef92d26cf598e2b917e9f58fd9615c9d49804f5ed470aa650105d3a908b.jpg)

## Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories


### Images

![0226832a02310099a1b63fc00aff4ff3c8488e60119b7506d6b81619948a254c.jpg](../iclr_results/1146_Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories/images/0226832a02310099a1b63fc00aff4ff3c8488e60119b7506d6b81619948a254c.jpg)

![0b2ca41559c96cf7ac3b1b5de67e43677e4f99dfff0bd0885ca674b1754ba874.jpg](../iclr_results/1146_Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories/images/0b2ca41559c96cf7ac3b1b5de67e43677e4f99dfff0bd0885ca674b1754ba874.jpg)

![21d00ed3ce39544189b9490f06daea4622e3a74c8a087750b56367601f950ebc.jpg](../iclr_results/1146_Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories/images/21d00ed3ce39544189b9490f06daea4622e3a74c8a087750b56367601f950ebc.jpg)

![516103dbb395a5e44354b166cdc5dbf9fce908b00ad6e48984c8812a170ed34a.jpg](../iclr_results/1146_Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories/images/516103dbb395a5e44354b166cdc5dbf9fce908b00ad6e48984c8812a170ed34a.jpg)

![534d0cd4c36e219ef39e281eab123f9ba57060d646e187337ad1ca5714798727.jpg](../iclr_results/1146_Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories/images/534d0cd4c36e219ef39e281eab123f9ba57060d646e187337ad1ca5714798727.jpg)

![607ddb5d55f3293c28394408362f989167cfd4ca581329313c0075e1d7375441.jpg](../iclr_results/1146_Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories/images/607ddb5d55f3293c28394408362f989167cfd4ca581329313c0075e1d7375441.jpg)

![74e9219fe7538f49b9df2534e3c2bd759bdb3a0877aafd74b330d898256995f3.jpg](../iclr_results/1146_Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories/images/74e9219fe7538f49b9df2534e3c2bd759bdb3a0877aafd74b330d898256995f3.jpg)

![aa76ac6e9e42616d5b3699fc4fe78b08c74ab51d648e5872796c9def02b3ea5e.jpg](../iclr_results/1146_Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories/images/aa76ac6e9e42616d5b3699fc4fe78b08c74ab51d648e5872796c9def02b3ea5e.jpg)

![bf6bafc1a8d91a8cba0e9bbf8e08cec4ac5fd98e1d605fa190946de3ae283a1d.jpg](../iclr_results/1146_Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories/images/bf6bafc1a8d91a8cba0e9bbf8e08cec4ac5fd98e1d605fa190946de3ae283a1d.jpg)

![d09965c82f49e248e59941911edf1991d9d5442260ddb3f5674215730b2ca5b5.jpg](../iclr_results/1146_Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories/images/d09965c82f49e248e59941911edf1991d9d5442260ddb3f5674215730b2ca5b5.jpg)

![d2d988d08c37b8b4d6aaf620df4ae82ff1385012bb57b0dc7b62b515b5ea3a4f.jpg](../iclr_results/1146_Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories/images/d2d988d08c37b8b4d6aaf620df4ae82ff1385012bb57b0dc7b62b515b5ea3a4f.jpg)

![e920c092068f5f56e80d5abdb3d0f23c627f1f41e283ea445ef6297040beefc3.jpg](../iclr_results/1146_Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories/images/e920c092068f5f56e80d5abdb3d0f23c627f1f41e283ea445ef6297040beefc3.jpg)

![f567ec9444d3585cbc2054a8d629dfaff5155c667037fff365024b8c6c82963f.jpg](../iclr_results/1146_Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories/images/f567ec9444d3585cbc2054a8d629dfaff5155c667037fff365024b8c6c82963f.jpg)

![fd799bb0c877a0bcba161b6decbf6d6ff1c5e7f8c79172f9f3f425db4822f575.jpg](../iclr_results/1146_Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories/images/fd799bb0c877a0bcba161b6decbf6d6ff1c5e7f8c79172f9f3f425db4822f575.jpg)

### Tables

![13c40b1b3f352196c6c18a8a70eb95d5b5c8ac05b6d6c318863f3dbb78b7bcd4.jpg](../iclr_results/1146_Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories/tables/13c40b1b3f352196c6c18a8a70eb95d5b5c8ac05b6d6c318863f3dbb78b7bcd4.jpg)

![2a91c5797e070b1e3242a0c694a2fcac6d7b92b78782633b5b6803dad38befcb.jpg](../iclr_results/1146_Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories/tables/2a91c5797e070b1e3242a0c694a2fcac6d7b92b78782633b5b6803dad38befcb.jpg)

![9f9edaadf24a82c70aee528ec323cf7a4d6422f71b4111ef46c8b3382733e08e.jpg](../iclr_results/1146_Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories/tables/9f9edaadf24a82c70aee528ec323cf7a4d6422f71b4111ef46c8b3382733e08e.jpg)

![aea35157505ee6833a2ea126b552a174e7b29f74c61b2fd60a77daf1cb485c0c.jpg](../iclr_results/1146_Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories/tables/aea35157505ee6833a2ea126b552a174e7b29f74c61b2fd60a77daf1cb485c0c.jpg)

![cfa03875c618c8fd02bc8420d5682e3a385fce2286d607cbb310c30334342f97.jpg](../iclr_results/1146_Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories/tables/cfa03875c618c8fd02bc8420d5682e3a385fce2286d607cbb310c30334342f97.jpg)

![d3cb48689a7ceb100bd33db968c0866b51ce5843970a531bf0ce138cd213a0c1.jpg](../iclr_results/1146_Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories/tables/d3cb48689a7ceb100bd33db968c0866b51ce5843970a531bf0ce138cd213a0c1.jpg)

![d98854cf3b9a2ee84e411411965fb8ca439c90db78007a10d12791e77dfc9f69.jpg](../iclr_results/1146_Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories/tables/d98854cf3b9a2ee84e411411965fb8ca439c90db78007a10d12791e77dfc9f69.jpg)

![ede973f94e9d903643d47d6d64c043a6500c85b86933450685c2536904365bed.jpg](../iclr_results/1146_Adaptive Shrinkage Estimation for Personalized Deep Kernel Regression in Modeling Brain Trajectories/tables/ede973f94e9d903643d47d6d64c043a6500c85b86933450685c2536904365bed.jpg)

## Scalable Discrete Diffusion Samplers: Combinatorial Optimization and Statistical Physics


### Images

![fd703012ee92563a0f8c54e79d8cea3093088c74fe9725137a8304b3b5e718b0.jpg](../iclr_results/1147_Scalable Discrete Diffusion Samplers_ Combinatorial Optimization and Statistical Physics/images/fd703012ee92563a0f8c54e79d8cea3093088c74fe9725137a8304b3b5e718b0.jpg)

### Tables

![0d55bb5cf5d69bf79e3c65783de594cb5f9df8a3018b3770f70a3813ac4a0f81.jpg](../iclr_results/1147_Scalable Discrete Diffusion Samplers_ Combinatorial Optimization and Statistical Physics/tables/0d55bb5cf5d69bf79e3c65783de594cb5f9df8a3018b3770f70a3813ac4a0f81.jpg)

![0f12abc63421eac2bd360b5ef09da8d10a3a4fb53d02c7f92af2bae5a61acac5.jpg](../iclr_results/1147_Scalable Discrete Diffusion Samplers_ Combinatorial Optimization and Statistical Physics/tables/0f12abc63421eac2bd360b5ef09da8d10a3a4fb53d02c7f92af2bae5a61acac5.jpg)

![11a5a34aae1f14b43f2ed9fcc259cb9d8502eedf25924e949d687ee90cd4799a.jpg](../iclr_results/1147_Scalable Discrete Diffusion Samplers_ Combinatorial Optimization and Statistical Physics/tables/11a5a34aae1f14b43f2ed9fcc259cb9d8502eedf25924e949d687ee90cd4799a.jpg)

![1cb8be3c6084abfab3c43b5c026b2415b75faea0a1aebd3b7929fc0ad7c3a718.jpg](../iclr_results/1147_Scalable Discrete Diffusion Samplers_ Combinatorial Optimization and Statistical Physics/tables/1cb8be3c6084abfab3c43b5c026b2415b75faea0a1aebd3b7929fc0ad7c3a718.jpg)

![29182892f68c31c1d4673763593238f1fc3d0a660d337ddb0e981476e533b2c3.jpg](../iclr_results/1147_Scalable Discrete Diffusion Samplers_ Combinatorial Optimization and Statistical Physics/tables/29182892f68c31c1d4673763593238f1fc3d0a660d337ddb0e981476e533b2c3.jpg)

![7717d7fe080a568d6d7b808b056b7738d0bfda2a6cc7d9c0f8f77c64ea995d60.jpg](../iclr_results/1147_Scalable Discrete Diffusion Samplers_ Combinatorial Optimization and Statistical Physics/tables/7717d7fe080a568d6d7b808b056b7738d0bfda2a6cc7d9c0f8f77c64ea995d60.jpg)

![772f466ad0f8b58df0deeee0b38776a553945eaafd4f6c9078ff9d58395146fe.jpg](../iclr_results/1147_Scalable Discrete Diffusion Samplers_ Combinatorial Optimization and Statistical Physics/tables/772f466ad0f8b58df0deeee0b38776a553945eaafd4f6c9078ff9d58395146fe.jpg)

![85f3107ecc1c57ca497c8a4f25e98ff8b0d4a6bd92cbbb33b1c31d8432fa4c83.jpg](../iclr_results/1147_Scalable Discrete Diffusion Samplers_ Combinatorial Optimization and Statistical Physics/tables/85f3107ecc1c57ca497c8a4f25e98ff8b0d4a6bd92cbbb33b1c31d8432fa4c83.jpg)

![9defd9019f4a51ac4b83bf3f37241ba63b8fc43000e6dc45ad0c161a853ce9f6.jpg](../iclr_results/1147_Scalable Discrete Diffusion Samplers_ Combinatorial Optimization and Statistical Physics/tables/9defd9019f4a51ac4b83bf3f37241ba63b8fc43000e6dc45ad0c161a853ce9f6.jpg)

![bb83dc557c25b186eed37b0b0e7d81d35bff4ae81009ed95a32bdd607148b741.jpg](../iclr_results/1147_Scalable Discrete Diffusion Samplers_ Combinatorial Optimization and Statistical Physics/tables/bb83dc557c25b186eed37b0b0e7d81d35bff4ae81009ed95a32bdd607148b741.jpg)

![d48719e1624b53384d2edf85446f2704ea15b87ff3e46f1bab2903b0f83a3171.jpg](../iclr_results/1147_Scalable Discrete Diffusion Samplers_ Combinatorial Optimization and Statistical Physics/tables/d48719e1624b53384d2edf85446f2704ea15b87ff3e46f1bab2903b0f83a3171.jpg)

![eecd5d6d44903d912da3ed51db6257987414c6fb016dab47412545f5c3d9cf31.jpg](../iclr_results/1147_Scalable Discrete Diffusion Samplers_ Combinatorial Optimization and Statistical Physics/tables/eecd5d6d44903d912da3ed51db6257987414c6fb016dab47412545f5c3d9cf31.jpg)

![f6fb89fcf0f11468bfdd4366ad79318da2f17474c6f8603bba34efffc97c3460.jpg](../iclr_results/1147_Scalable Discrete Diffusion Samplers_ Combinatorial Optimization and Statistical Physics/tables/f6fb89fcf0f11468bfdd4366ad79318da2f17474c6f8603bba34efffc97c3460.jpg)

## Random-Set Neural Networks


### Images

![133620ac9070c003694412ebc014d987e6512b3e4d90929450dc0f1f62e279a0.jpg](../iclr_results/1148_Random-Set Neural Networks/images/133620ac9070c003694412ebc014d987e6512b3e4d90929450dc0f1f62e279a0.jpg)

![2e9089696dd39bc1b97e81a4d7533200d59facd0081f801ca11758bffcc343b6.jpg](../iclr_results/1148_Random-Set Neural Networks/images/2e9089696dd39bc1b97e81a4d7533200d59facd0081f801ca11758bffcc343b6.jpg)

![33906313b31c3e57b58009d88c2bfa4afa7c332b5f85787f11e4cb8743d5abdd.jpg](../iclr_results/1148_Random-Set Neural Networks/images/33906313b31c3e57b58009d88c2bfa4afa7c332b5f85787f11e4cb8743d5abdd.jpg)

![373dec8ae2a76f8a181d9a15a36a7b076aa84cdbd050dd985b0397f827d3e934.jpg](../iclr_results/1148_Random-Set Neural Networks/images/373dec8ae2a76f8a181d9a15a36a7b076aa84cdbd050dd985b0397f827d3e934.jpg)

![3978c72749b72854bce4af6add6712c437bafd58f0aca942fba8967c48af949d.jpg](../iclr_results/1148_Random-Set Neural Networks/images/3978c72749b72854bce4af6add6712c437bafd58f0aca942fba8967c48af949d.jpg)

![4425884fda34b162bc5d7ab805338e2fbac63c5a967b8981d4d65a1e4c8b14d5.jpg](../iclr_results/1148_Random-Set Neural Networks/images/4425884fda34b162bc5d7ab805338e2fbac63c5a967b8981d4d65a1e4c8b14d5.jpg)

![4566393405c92938320d6100653d347d873ace288837d7bd68271ac19b454ec0.jpg](../iclr_results/1148_Random-Set Neural Networks/images/4566393405c92938320d6100653d347d873ace288837d7bd68271ac19b454ec0.jpg)

![4711f26f8f8db5a1f1d9239b1d54d14715d54a0e3935eb21d802d5931e2dbe9e.jpg](../iclr_results/1148_Random-Set Neural Networks/images/4711f26f8f8db5a1f1d9239b1d54d14715d54a0e3935eb21d802d5931e2dbe9e.jpg)

![4cb6f3f4b119c1e99028fc1782899b51c7557c284a9709da83c141234cece1d6.jpg](../iclr_results/1148_Random-Set Neural Networks/images/4cb6f3f4b119c1e99028fc1782899b51c7557c284a9709da83c141234cece1d6.jpg)

![600cfd7633f54d295a1aff10d6c378d40df1eb7bafbf8d5e65245f09ff803bd2.jpg](../iclr_results/1148_Random-Set Neural Networks/images/600cfd7633f54d295a1aff10d6c378d40df1eb7bafbf8d5e65245f09ff803bd2.jpg)

![66879a31f2eba8d909f22da91283aa425d406542222d61db3fe5d01f8014098b.jpg](../iclr_results/1148_Random-Set Neural Networks/images/66879a31f2eba8d909f22da91283aa425d406542222d61db3fe5d01f8014098b.jpg)

![7136f34be0127f263cc3a5a1f0b71cd01d56cdfa27f971b93cbe3db495489176.jpg](../iclr_results/1148_Random-Set Neural Networks/images/7136f34be0127f263cc3a5a1f0b71cd01d56cdfa27f971b93cbe3db495489176.jpg)

![74fb9e8859d56d6e40353fe545216b27297eac0c93691a7b182ce70f76323266.jpg](../iclr_results/1148_Random-Set Neural Networks/images/74fb9e8859d56d6e40353fe545216b27297eac0c93691a7b182ce70f76323266.jpg)

![763bf4b510cca37d02d2487338881fe3dd6e54d04cac58ac9d431df767b3bef0.jpg](../iclr_results/1148_Random-Set Neural Networks/images/763bf4b510cca37d02d2487338881fe3dd6e54d04cac58ac9d431df767b3bef0.jpg)

![7eb82c8e08861a7055e9ab89ee22e992d2cbb703cf6112ce75382a3a67fea4b5.jpg](../iclr_results/1148_Random-Set Neural Networks/images/7eb82c8e08861a7055e9ab89ee22e992d2cbb703cf6112ce75382a3a67fea4b5.jpg)

![82028f3ffdeb15117166cc1db7ce7f185f888b8940d81cb14b208854ed601435.jpg](../iclr_results/1148_Random-Set Neural Networks/images/82028f3ffdeb15117166cc1db7ce7f185f888b8940d81cb14b208854ed601435.jpg)

![8e31eb2f9519090d9b3e385b895ceefc1f3416a9dd132f6aded1f19fa7ee5de8.jpg](../iclr_results/1148_Random-Set Neural Networks/images/8e31eb2f9519090d9b3e385b895ceefc1f3416a9dd132f6aded1f19fa7ee5de8.jpg)

![980e68f756d2432f4deb51e737cdac550219a65c96ebeceae92315cd216b9858.jpg](../iclr_results/1148_Random-Set Neural Networks/images/980e68f756d2432f4deb51e737cdac550219a65c96ebeceae92315cd216b9858.jpg)

![99dc491e52641a1369236fb0d3abcc5074c1bd5e3b8246416c88041fe6b8706f.jpg](../iclr_results/1148_Random-Set Neural Networks/images/99dc491e52641a1369236fb0d3abcc5074c1bd5e3b8246416c88041fe6b8706f.jpg)

![a3eddbfd2422afb27e4e1d014ec5dc721b4741b1bde66d4551237c1ecdcfe17b.jpg](../iclr_results/1148_Random-Set Neural Networks/images/a3eddbfd2422afb27e4e1d014ec5dc721b4741b1bde66d4551237c1ecdcfe17b.jpg)

![a702ca6d84c808e469575a47e86aea22278020a27b2f4ca487d3cf62a479bc3c.jpg](../iclr_results/1148_Random-Set Neural Networks/images/a702ca6d84c808e469575a47e86aea22278020a27b2f4ca487d3cf62a479bc3c.jpg)

![acbb9ed32dd9d14824ab52da93749e30592bfeee0f3032d34ac268d9ab9f9786.jpg](../iclr_results/1148_Random-Set Neural Networks/images/acbb9ed32dd9d14824ab52da93749e30592bfeee0f3032d34ac268d9ab9f9786.jpg)

![b4628d6178a5e1ca25cb83bf5df3dd14322082d8a94ee40271df8f7706111b44.jpg](../iclr_results/1148_Random-Set Neural Networks/images/b4628d6178a5e1ca25cb83bf5df3dd14322082d8a94ee40271df8f7706111b44.jpg)

![b79e5d55e0383789f2b7ff86d2fc3ff8fcde20fb66761c10752d68d88691614b.jpg](../iclr_results/1148_Random-Set Neural Networks/images/b79e5d55e0383789f2b7ff86d2fc3ff8fcde20fb66761c10752d68d88691614b.jpg)

![c65d8f9e09d41685542523c99d170b233b41d4c9d6f4348160cbfdf6432c23d7.jpg](../iclr_results/1148_Random-Set Neural Networks/images/c65d8f9e09d41685542523c99d170b233b41d4c9d6f4348160cbfdf6432c23d7.jpg)

![cf71e36bb01688cab37bc76357bf3add1769de5a2d7f3a0a59e66f9ca9056605.jpg](../iclr_results/1148_Random-Set Neural Networks/images/cf71e36bb01688cab37bc76357bf3add1769de5a2d7f3a0a59e66f9ca9056605.jpg)

![d38abcb5a647dea2eeb1567b6d767fb6dc05c3d64db9dd1228c559331abaa60c.jpg](../iclr_results/1148_Random-Set Neural Networks/images/d38abcb5a647dea2eeb1567b6d767fb6dc05c3d64db9dd1228c559331abaa60c.jpg)

![ddea93ffaed33408a127eb741f38f4e3bda5b595ab90e164684c7dff8dba13f4.jpg](../iclr_results/1148_Random-Set Neural Networks/images/ddea93ffaed33408a127eb741f38f4e3bda5b595ab90e164684c7dff8dba13f4.jpg)

![ec232fa87c72d7eba349154a0fbcb13528142b7dc194b14d319b0ff1f459f5c6.jpg](../iclr_results/1148_Random-Set Neural Networks/images/ec232fa87c72d7eba349154a0fbcb13528142b7dc194b14d319b0ff1f459f5c6.jpg)

### Tables

![183e3533f192b0cdc2e6e5668ca228d9d78d5a3fd655d869478bccc55997b9a9.jpg](../iclr_results/1148_Random-Set Neural Networks/tables/183e3533f192b0cdc2e6e5668ca228d9d78d5a3fd655d869478bccc55997b9a9.jpg)

![3041e01e8c6485e283bf17c3900836fb0c6ef48c83a242906f31c08195252c39.jpg](../iclr_results/1148_Random-Set Neural Networks/tables/3041e01e8c6485e283bf17c3900836fb0c6ef48c83a242906f31c08195252c39.jpg)

![407acc208438ffbdb922c182833c5de545973f967431c25096b7d0cd8a5753cf.jpg](../iclr_results/1148_Random-Set Neural Networks/tables/407acc208438ffbdb922c182833c5de545973f967431c25096b7d0cd8a5753cf.jpg)

![602d38234c9dac7447dd7a7aa272ceaa52e08feac7d53851e89d1a5afeabed73.jpg](../iclr_results/1148_Random-Set Neural Networks/tables/602d38234c9dac7447dd7a7aa272ceaa52e08feac7d53851e89d1a5afeabed73.jpg)

![82f93fa6dcc2f3ac4659773dd71f1a27383b40f6d3e99558b2a1d39811cfe1f6.jpg](../iclr_results/1148_Random-Set Neural Networks/tables/82f93fa6dcc2f3ac4659773dd71f1a27383b40f6d3e99558b2a1d39811cfe1f6.jpg)

![8dcec8c7aa95330720293c7c2116fac4edee5ad0bcf7fb602cea34c3b332b78d.jpg](../iclr_results/1148_Random-Set Neural Networks/tables/8dcec8c7aa95330720293c7c2116fac4edee5ad0bcf7fb602cea34c3b332b78d.jpg)

![94d2ea8302ab6cfd9d4552ee38c6ba44be7665681005d79c4b95688198a66754.jpg](../iclr_results/1148_Random-Set Neural Networks/tables/94d2ea8302ab6cfd9d4552ee38c6ba44be7665681005d79c4b95688198a66754.jpg)

![ac18820c8508bc9a7e3a4e33fedfdcf9794a96f7e8b6370fdb2dc2c84a412f92.jpg](../iclr_results/1148_Random-Set Neural Networks/tables/ac18820c8508bc9a7e3a4e33fedfdcf9794a96f7e8b6370fdb2dc2c84a412f92.jpg)

![b53b36474687af713837dfb49a24fc9ae0d3346e2531edf87c275924ff804070.jpg](../iclr_results/1148_Random-Set Neural Networks/tables/b53b36474687af713837dfb49a24fc9ae0d3346e2531edf87c275924ff804070.jpg)

![bbbe30b7b791016087b69ce45171537f38f6f89f6d9a1231bc07786c4eb958aa.jpg](../iclr_results/1148_Random-Set Neural Networks/tables/bbbe30b7b791016087b69ce45171537f38f6f89f6d9a1231bc07786c4eb958aa.jpg)

![c5b52e5d82020fd21c30f42cb1ea8c0df27e1c8d05e75ab7cf9b161c8c2736c0.jpg](../iclr_results/1148_Random-Set Neural Networks/tables/c5b52e5d82020fd21c30f42cb1ea8c0df27e1c8d05e75ab7cf9b161c8c2736c0.jpg)

![d44d4c817ee589550b3236c3b5b50bdc845d51be734ebeea3e64ac9137e5eb7b.jpg](../iclr_results/1148_Random-Set Neural Networks/tables/d44d4c817ee589550b3236c3b5b50bdc845d51be734ebeea3e64ac9137e5eb7b.jpg)

![dbe5dc7bf5b6a4ca3b12b38618b18f3d39fb5d93c6eb0331133a435994545cdf.jpg](../iclr_results/1148_Random-Set Neural Networks/tables/dbe5dc7bf5b6a4ca3b12b38618b18f3d39fb5d93c6eb0331133a435994545cdf.jpg)

![deb112d30bfb6fa51285a4e4cdc162a0597f020bf92f91744a9e1f0b537206dd.jpg](../iclr_results/1148_Random-Set Neural Networks/tables/deb112d30bfb6fa51285a4e4cdc162a0597f020bf92f91744a9e1f0b537206dd.jpg)

![e5ea2d5126dad7fa94668e03db474077b6e1a988b969e6b767aadc800b97d90f.jpg](../iclr_results/1148_Random-Set Neural Networks/tables/e5ea2d5126dad7fa94668e03db474077b6e1a988b969e6b767aadc800b97d90f.jpg)

![ec4a1696889c4801848775e355e1deeb8e510cd033fcc9d9be458ce013036805.jpg](../iclr_results/1148_Random-Set Neural Networks/tables/ec4a1696889c4801848775e355e1deeb8e510cd033fcc9d9be458ce013036805.jpg)

![f744e9a992221b4da00013bb5e6f283e8fa41fdaac24f26bb7bdb942fcc1b9e9.jpg](../iclr_results/1148_Random-Set Neural Networks/tables/f744e9a992221b4da00013bb5e6f283e8fa41fdaac24f26bb7bdb942fcc1b9e9.jpg)

![f759c1b6784d19a29777835774fc29fa8108e70bce93c1c2bf490bed5fdb2b1e.jpg](../iclr_results/1148_Random-Set Neural Networks/tables/f759c1b6784d19a29777835774fc29fa8108e70bce93c1c2bf490bed5fdb2b1e.jpg)

![fbaf7279782fd1d5715108c993dac7a89892ae4cd7789bca12022ecf10a99d2a.jpg](../iclr_results/1148_Random-Set Neural Networks/tables/fbaf7279782fd1d5715108c993dac7a89892ae4cd7789bca12022ecf10a99d2a.jpg)

## SOREL: A Stochastic Algorithm for Spectral Risks Minimization


### Images

![0c85179600c613234e8b27c535a0f631113748495fe68fa669046784ce729f8c.jpg](../iclr_results/1149_SOREL_ A Stochastic Algorithm for Spectral Risks Minimization/images/0c85179600c613234e8b27c535a0f631113748495fe68fa669046784ce729f8c.jpg)

![17aa26707967c8032fc1d40174866745e5ef31b726adaa344100e279729f6834.jpg](../iclr_results/1149_SOREL_ A Stochastic Algorithm for Spectral Risks Minimization/images/17aa26707967c8032fc1d40174866745e5ef31b726adaa344100e279729f6834.jpg)

![299d91cd7e16a7e4848d5c9102b2c95fb96d1b927173a8510d8e44815470d7c5.jpg](../iclr_results/1149_SOREL_ A Stochastic Algorithm for Spectral Risks Minimization/images/299d91cd7e16a7e4848d5c9102b2c95fb96d1b927173a8510d8e44815470d7c5.jpg)

![6fa9596462ff4f498215371d8eabe4decb6ef802a213d40e2a4768afa1ae5c7f.jpg](../iclr_results/1149_SOREL_ A Stochastic Algorithm for Spectral Risks Minimization/images/6fa9596462ff4f498215371d8eabe4decb6ef802a213d40e2a4768afa1ae5c7f.jpg)

![d7f2e9a095607ac628e1656f64df768615076cdc51dc394c90b12d6a0a9f71ae.jpg](../iclr_results/1149_SOREL_ A Stochastic Algorithm for Spectral Risks Minimization/images/d7f2e9a095607ac628e1656f64df768615076cdc51dc394c90b12d6a0a9f71ae.jpg)

![d8027a83a0983eb9c1ae6fae57ede3be357526ec45b0162e71dea733b0830b91.jpg](../iclr_results/1149_SOREL_ A Stochastic Algorithm for Spectral Risks Minimization/images/d8027a83a0983eb9c1ae6fae57ede3be357526ec45b0162e71dea733b0830b91.jpg)

![edfe17ff3a7d08219594eb4efc195ab02cdc088e82b8da5acccd524e839e0f25.jpg](../iclr_results/1149_SOREL_ A Stochastic Algorithm for Spectral Risks Minimization/images/edfe17ff3a7d08219594eb4efc195ab02cdc088e82b8da5acccd524e839e0f25.jpg)

### Tables

![15c4a99fd34077c9dc5e04c13fd7de2a8814cad52fd6ae49ca9029c6d7738041.jpg](../iclr_results/1149_SOREL_ A Stochastic Algorithm for Spectral Risks Minimization/tables/15c4a99fd34077c9dc5e04c13fd7de2a8814cad52fd6ae49ca9029c6d7738041.jpg)

![264eab0c2324e5c767ecef91f8cee7a90d879f5f161449600764845e343eb2cd.jpg](../iclr_results/1149_SOREL_ A Stochastic Algorithm for Spectral Risks Minimization/tables/264eab0c2324e5c767ecef91f8cee7a90d879f5f161449600764845e343eb2cd.jpg)

![5449d5d6fed98aebfb4aeed83e152987ecf95bfdb36f99e054a13330486ec3c4.jpg](../iclr_results/1149_SOREL_ A Stochastic Algorithm for Spectral Risks Minimization/tables/5449d5d6fed98aebfb4aeed83e152987ecf95bfdb36f99e054a13330486ec3c4.jpg)

![641999c6ca99adefe61cc245288eed0c433548fa6a455ac0b0ad302723aba1be.jpg](../iclr_results/1149_SOREL_ A Stochastic Algorithm for Spectral Risks Minimization/tables/641999c6ca99adefe61cc245288eed0c433548fa6a455ac0b0ad302723aba1be.jpg)

![91f88259a9e498e0a6c3e052c78c024cc96be9e0d9c6a1662e609a092fd2978b.jpg](../iclr_results/1149_SOREL_ A Stochastic Algorithm for Spectral Risks Minimization/tables/91f88259a9e498e0a6c3e052c78c024cc96be9e0d9c6a1662e609a092fd2978b.jpg)

## Encryption-Friendly LLM Architecture


### Images

![021bf05411ff224072d65c0e6b02bd7dadd1e8f73e3e5b3c5269994c5bbb937b.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/images/021bf05411ff224072d65c0e6b02bd7dadd1e8f73e3e5b3c5269994c5bbb937b.jpg)

![177d335d15ffd81608cf0fb78915221b891dff4977af5b5b706d1cf8cd8b4f2e.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/images/177d335d15ffd81608cf0fb78915221b891dff4977af5b5b706d1cf8cd8b4f2e.jpg)

![259f2ef51cc9f144b83fd9b32fb56f6d4ff9f021c58de03b52805ecc1df2d3cc.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/images/259f2ef51cc9f144b83fd9b32fb56f6d4ff9f021c58de03b52805ecc1df2d3cc.jpg)

![42c79d9df2c8b8e55d1f7b6c4d48fed6023acc426e2740dfdc927914b867f6f0.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/images/42c79d9df2c8b8e55d1f7b6c4d48fed6023acc426e2740dfdc927914b867f6f0.jpg)

![527324f0f9b94c65960bb739f6fd1c5cb158be2f26870ef798ca7cdc0ac8f0d7.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/images/527324f0f9b94c65960bb739f6fd1c5cb158be2f26870ef798ca7cdc0ac8f0d7.jpg)

![6929f022b80c06612d779acc7463c1cf652e91c1941ae68ed4ddedcd96ad3032.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/images/6929f022b80c06612d779acc7463c1cf652e91c1941ae68ed4ddedcd96ad3032.jpg)

![bec8d719a0d829ce8cc37d303e3388e4d39eb4d09dbd5d71aa54f45c3d4e0f9a.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/images/bec8d719a0d829ce8cc37d303e3388e4d39eb4d09dbd5d71aa54f45c3d4e0f9a.jpg)

![d0b985c2773f8659c705c21c7e4ffd6b3f3a23db73c7a07d14dbe9591c4e68f9.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/images/d0b985c2773f8659c705c21c7e4ffd6b3f3a23db73c7a07d14dbe9591c4e68f9.jpg)

![d808db6327857a0aaded2ebdeeab89f0807ac10ba0cc40ea059d00a5b9384778.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/images/d808db6327857a0aaded2ebdeeab89f0807ac10ba0cc40ea059d00a5b9384778.jpg)

![ec34a295fd238d32a4d9fc9bb3e3a7b5977e0d2c5f29883e328faf6cb247f50d.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/images/ec34a295fd238d32a4d9fc9bb3e3a7b5977e0d2c5f29883e328faf6cb247f50d.jpg)

### Tables

![0ca3753ae72e99eaa30876547ef49565ad72a357d64e759841df59f3fa5e0293.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/tables/0ca3753ae72e99eaa30876547ef49565ad72a357d64e759841df59f3fa5e0293.jpg)

![12c2e70eb4da1f595fa14658d33bb6c86ceb8f23a8495fb07b01d0e680eb2033.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/tables/12c2e70eb4da1f595fa14658d33bb6c86ceb8f23a8495fb07b01d0e680eb2033.jpg)

![22749023cffca8d1e9ef6691b1142506d14e6c4f99dade8e1e33e26d208be8fa.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/tables/22749023cffca8d1e9ef6691b1142506d14e6c4f99dade8e1e33e26d208be8fa.jpg)

![23559fce516f69c6258ae517b6b4d4f304a3684cc2b5215142e01f25779c0afb.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/tables/23559fce516f69c6258ae517b6b4d4f304a3684cc2b5215142e01f25779c0afb.jpg)

![2bd7adf07a0ccbf657abc0860cc0d6731442ac4a66e6dd7f6138263505acec72.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/tables/2bd7adf07a0ccbf657abc0860cc0d6731442ac4a66e6dd7f6138263505acec72.jpg)

![2d8eaffe5f4ee6b938449f94ee32b70acbdad519a21c8e1cb027954900cb9df8.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/tables/2d8eaffe5f4ee6b938449f94ee32b70acbdad519a21c8e1cb027954900cb9df8.jpg)

![33fec4be970d09c4bc1e7718af6ff081afc333c3d3036efd97c830e30cd2f2b8.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/tables/33fec4be970d09c4bc1e7718af6ff081afc333c3d3036efd97c830e30cd2f2b8.jpg)

![3c0ce219b56a773c47a14c0fc42af671d72b67f84fef4cd966509349ce1c547b.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/tables/3c0ce219b56a773c47a14c0fc42af671d72b67f84fef4cd966509349ce1c547b.jpg)

![3e4bf19bf4c34844f5ea60fecadcfd2ba5f5a41ab009bf13919ee9ee64c175f7.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/tables/3e4bf19bf4c34844f5ea60fecadcfd2ba5f5a41ab009bf13919ee9ee64c175f7.jpg)

![538a320a42a75c8f59912eb4d6191d6e7923bc91537415521ebe55acd23b1e62.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/tables/538a320a42a75c8f59912eb4d6191d6e7923bc91537415521ebe55acd23b1e62.jpg)

![5c02065aa21bf5a207b1b96e6062d835f6cac5f6ccc5af6e9ba11deeb6923fed.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/tables/5c02065aa21bf5a207b1b96e6062d835f6cac5f6ccc5af6e9ba11deeb6923fed.jpg)

![8f35668440fe515a84e20ba330008c9500c0924dcab0d68b1165b61f27669acb.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/tables/8f35668440fe515a84e20ba330008c9500c0924dcab0d68b1165b61f27669acb.jpg)

![94ceac180adcf3e27e37f3fddc11ef0e30a84fffe7d30668c78459fa12947906.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/tables/94ceac180adcf3e27e37f3fddc11ef0e30a84fffe7d30668c78459fa12947906.jpg)

![9b93ce9914000b2a9184fd7ccf092846fd46eef753cc97a324d0a6925c6cf279.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/tables/9b93ce9914000b2a9184fd7ccf092846fd46eef753cc97a324d0a6925c6cf279.jpg)

![a4f79965a43c0cc9ccfd4dcb369b356846cca92ad5a0625a86163783636e10f1.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/tables/a4f79965a43c0cc9ccfd4dcb369b356846cca92ad5a0625a86163783636e10f1.jpg)

![bb6b15c0d3516af9add40c3ec12f1ae55b05d9c39656bafe4ef82e2eb8c0c824.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/tables/bb6b15c0d3516af9add40c3ec12f1ae55b05d9c39656bafe4ef82e2eb8c0c824.jpg)

![bfbe6c1b0998da93657c05108d9616f040499f305da9ba9b5b7910948373e274.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/tables/bfbe6c1b0998da93657c05108d9616f040499f305da9ba9b5b7910948373e274.jpg)

![c633c035ddad0d11f32ebdf0d604aecb77910b88d92df5841bf8aae3b0f905ef.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/tables/c633c035ddad0d11f32ebdf0d604aecb77910b88d92df5841bf8aae3b0f905ef.jpg)

![d5f6bfc5cf82ee883f688b57c41e42216d1a3a025a2e9d900d017bdfec8ca070.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/tables/d5f6bfc5cf82ee883f688b57c41e42216d1a3a025a2e9d900d017bdfec8ca070.jpg)

![eaf1fc1530ee0f03629c2dfa37446472c379f5ad14b6de7796a39c3096233d83.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/tables/eaf1fc1530ee0f03629c2dfa37446472c379f5ad14b6de7796a39c3096233d83.jpg)

![ef9dbaebcc8da573beb2ee2a4779b981796769ed9c3a55cf0a993ae3c1b7290d.jpg](../iclr_results/1150_Encryption-Friendly LLM Architecture/tables/ef9dbaebcc8da573beb2ee2a4779b981796769ed9c3a55cf0a993ae3c1b7290d.jpg)

## TEOChat: A Large Vision-Language Assistant for Temporal Earth Observation Data


### Images

![00c6a4eaf2026a2e94ccdb0cfee20c71340b3d80b8593be24be6b475536c81e0.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/images/00c6a4eaf2026a2e94ccdb0cfee20c71340b3d80b8593be24be6b475536c81e0.jpg)

![1c9050ce29c78a1d3c0272ee78801fafc6879293a62c344b19f76abe62c2b8e3.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/images/1c9050ce29c78a1d3c0272ee78801fafc6879293a62c344b19f76abe62c2b8e3.jpg)

![68cee7629f4cf2e2b2619d84054ef1949f09bc2494e6b7318f4de45da7e131f6.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/images/68cee7629f4cf2e2b2619d84054ef1949f09bc2494e6b7318f4de45da7e131f6.jpg)

![80ebbabc445710115eb3d051ae4153174ecb61192f5eea78ad08163719583141.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/images/80ebbabc445710115eb3d051ae4153174ecb61192f5eea78ad08163719583141.jpg)

![936a05e9ed785424a301d8858e4f81d8397a2fd54c102f31425ccd415c21bf9d.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/images/936a05e9ed785424a301d8858e4f81d8397a2fd54c102f31425ccd415c21bf9d.jpg)

![b6541fe56ab5c48dbc3661da380dcb641f5f5a73488eaed4202d3427916470a9.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/images/b6541fe56ab5c48dbc3661da380dcb641f5f5a73488eaed4202d3427916470a9.jpg)

![cef49c3e4386bba209129d5c2bc44131296cf5bbd0886872e3f7caa379c3b3c5.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/images/cef49c3e4386bba209129d5c2bc44131296cf5bbd0886872e3f7caa379c3b3c5.jpg)

![dcf60d228b33cdd8c2611a8d9ff6ceef8277958a4f1af75405776948fdb8520a.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/images/dcf60d228b33cdd8c2611a8d9ff6ceef8277958a4f1af75405776948fdb8520a.jpg)

### Tables

![0b67be7439a823fa00eeb134048f2499423e07ef2d5802673d9dee9d1d563327.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/tables/0b67be7439a823fa00eeb134048f2499423e07ef2d5802673d9dee9d1d563327.jpg)

![1220f95ca37a182b07e1ee2fbf4259ff2cc46e0368b850f92a116ea10bdf7f95.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/tables/1220f95ca37a182b07e1ee2fbf4259ff2cc46e0368b850f92a116ea10bdf7f95.jpg)

![14a90538c259550a5b79d05a77f500efd2c40347c5ca799989f78cb53e6e5fe2.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/tables/14a90538c259550a5b79d05a77f500efd2c40347c5ca799989f78cb53e6e5fe2.jpg)

![2fb622355d3eb634c9235155d9abb350ebfb9f43d5d85b48464dcd0e1cf5f5a6.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/tables/2fb622355d3eb634c9235155d9abb350ebfb9f43d5d85b48464dcd0e1cf5f5a6.jpg)

![314816e7fbe6ab8447c767b9b16d1c10a00a36561dbd83796780e8cac2bd9824.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/tables/314816e7fbe6ab8447c767b9b16d1c10a00a36561dbd83796780e8cac2bd9824.jpg)

![3b05de74376f29a5fb3eae7c14a159c45cbe20ed6c585686f2cf0ccc18b22313.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/tables/3b05de74376f29a5fb3eae7c14a159c45cbe20ed6c585686f2cf0ccc18b22313.jpg)

![3d5c4ebafeb165dce84fe272dcb9dd7fc3ce6d0cd8c25560bb79e83f02b68493.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/tables/3d5c4ebafeb165dce84fe272dcb9dd7fc3ce6d0cd8c25560bb79e83f02b68493.jpg)

![469fbb42e57676ce39c020449d9a9a4d88ec278998d48cf698bcad91114b64b7.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/tables/469fbb42e57676ce39c020449d9a9a4d88ec278998d48cf698bcad91114b64b7.jpg)

![541a89e5be72c809b569d2bbe72b86b38667177cba730843cf2554495b504079.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/tables/541a89e5be72c809b569d2bbe72b86b38667177cba730843cf2554495b504079.jpg)

![6b078c238b1e8a09648f47cc176855890a838c3e84c3fc253a809bcc0fed5de9.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/tables/6b078c238b1e8a09648f47cc176855890a838c3e84c3fc253a809bcc0fed5de9.jpg)

![70cb73821a60a11701939887a0ba22946bfdfcf9a861344ceac3779cf467c26d.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/tables/70cb73821a60a11701939887a0ba22946bfdfcf9a861344ceac3779cf467c26d.jpg)

![819eeabfbf75c3e43975ce98865f31c54a94bcfb93e7ceb4efa58ee1ea157041.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/tables/819eeabfbf75c3e43975ce98865f31c54a94bcfb93e7ceb4efa58ee1ea157041.jpg)

![822500046bd4b19752f8246c06ff189c7c94da65fdfb74175360b00835f8e9a1.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/tables/822500046bd4b19752f8246c06ff189c7c94da65fdfb74175360b00835f8e9a1.jpg)

![86ec2321c56888b7648863dc6434beb04a533e99f2cf55f5bffd83ebb3d680c5.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/tables/86ec2321c56888b7648863dc6434beb04a533e99f2cf55f5bffd83ebb3d680c5.jpg)

![8d78fc9d634075b74b695defeecfe96b3b8665d21744133a37f525abe72d90ac.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/tables/8d78fc9d634075b74b695defeecfe96b3b8665d21744133a37f525abe72d90ac.jpg)

![96ba813190ef4e467c5a0b15ae17e16631fc0ab0b68059ca3b668e126c6478ec.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/tables/96ba813190ef4e467c5a0b15ae17e16631fc0ab0b68059ca3b668e126c6478ec.jpg)

![b6bf997043b8862ce076b6828bf426dacc486acb7d17e2e0d3c60e4403403c64.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/tables/b6bf997043b8862ce076b6828bf426dacc486acb7d17e2e0d3c60e4403403c64.jpg)

![dbdbad9ee2c9aa01fa8878629c6d98a2d3a0c841f9bef50153b11c18cf98562e.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/tables/dbdbad9ee2c9aa01fa8878629c6d98a2d3a0c841f9bef50153b11c18cf98562e.jpg)

![efb02f8ec98406452ea721aad584f0be49e4e5ed98f6abf32e236e7efd2970cb.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/tables/efb02f8ec98406452ea721aad584f0be49e4e5ed98f6abf32e236e7efd2970cb.jpg)

![ffd8372e3a46fba0c529cc013553744910dd49201ce26df6435fff6d4775eb52.jpg](../iclr_results/1151_TEOChat_ A Large Vision-Language Assistant for Temporal Earth Observation Data/tables/ffd8372e3a46fba0c529cc013553744910dd49201ce26df6435fff6d4775eb52.jpg)

## LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory


### Images

![23b929bb0e3a3a6c3eea1a79c9b0a884efaa850564ed86b6d0142462661d8109.jpg](../iclr_results/1152_LongMemEval_ Benchmarking Chat Assistants on Long-Term Interactive Memory/images/23b929bb0e3a3a6c3eea1a79c9b0a884efaa850564ed86b6d0142462661d8109.jpg)

![3f57496a62c7c852f959eab62ec2bf0d26ea74b5a5041875627ac2e532e63125.jpg](../iclr_results/1152_LongMemEval_ Benchmarking Chat Assistants on Long-Term Interactive Memory/images/3f57496a62c7c852f959eab62ec2bf0d26ea74b5a5041875627ac2e532e63125.jpg)

![4a0065abf8642b5af262442d03bd94eaab14edc3b1ab8f1778a4050cc080b6c4.jpg](../iclr_results/1152_LongMemEval_ Benchmarking Chat Assistants on Long-Term Interactive Memory/images/4a0065abf8642b5af262442d03bd94eaab14edc3b1ab8f1778a4050cc080b6c4.jpg)

![5b4d83330785979aa3ef40a7017b26b31a651d387c9d57ea1f0fda294cab9d7b.jpg](../iclr_results/1152_LongMemEval_ Benchmarking Chat Assistants on Long-Term Interactive Memory/images/5b4d83330785979aa3ef40a7017b26b31a651d387c9d57ea1f0fda294cab9d7b.jpg)

![5db0e4b62eb68a7885817098ccdfcc1e50dc98d1e61add90127c54bc2e4e1a22.jpg](../iclr_results/1152_LongMemEval_ Benchmarking Chat Assistants on Long-Term Interactive Memory/images/5db0e4b62eb68a7885817098ccdfcc1e50dc98d1e61add90127c54bc2e4e1a22.jpg)

![5e5b4d541477fbeef31f1bc1fada7b6c71b077823e9c034551e0af9906e7c623.jpg](../iclr_results/1152_LongMemEval_ Benchmarking Chat Assistants on Long-Term Interactive Memory/images/5e5b4d541477fbeef31f1bc1fada7b6c71b077823e9c034551e0af9906e7c623.jpg)

![63837d0a3abfe9f958530ac65ab5798a830dc7baea8aa9429804d869261b56c0.jpg](../iclr_results/1152_LongMemEval_ Benchmarking Chat Assistants on Long-Term Interactive Memory/images/63837d0a3abfe9f958530ac65ab5798a830dc7baea8aa9429804d869261b56c0.jpg)

![8f4337c61b90cfc4267c615467e48b61f567dfeb83e3a965b35fd7b7b5ebc7c8.jpg](../iclr_results/1152_LongMemEval_ Benchmarking Chat Assistants on Long-Term Interactive Memory/images/8f4337c61b90cfc4267c615467e48b61f567dfeb83e3a965b35fd7b7b5ebc7c8.jpg)

![a4330e9bb3989b60d29cb816b940b284a31bf3ee07f42849f5e735cfdd63cae9.jpg](../iclr_results/1152_LongMemEval_ Benchmarking Chat Assistants on Long-Term Interactive Memory/images/a4330e9bb3989b60d29cb816b940b284a31bf3ee07f42849f5e735cfdd63cae9.jpg)

![a94ba5c18cb673e08f1a9f5b51e29950a7cc9b13b547b1f9f945fbc2135b98d3.jpg](../iclr_results/1152_LongMemEval_ Benchmarking Chat Assistants on Long-Term Interactive Memory/images/a94ba5c18cb673e08f1a9f5b51e29950a7cc9b13b547b1f9f945fbc2135b98d3.jpg)

### Tables

![2d088a7b882a7183c3d05db10897f8c5f300f4986e3b951f12bcdbcbcaffe129.jpg](../iclr_results/1152_LongMemEval_ Benchmarking Chat Assistants on Long-Term Interactive Memory/tables/2d088a7b882a7183c3d05db10897f8c5f300f4986e3b951f12bcdbcbcaffe129.jpg)

![3433b806330a1555f5e2ddb34cac7a94a14bf8a8e1bb675a4b86c9ec7fa63755.jpg](../iclr_results/1152_LongMemEval_ Benchmarking Chat Assistants on Long-Term Interactive Memory/tables/3433b806330a1555f5e2ddb34cac7a94a14bf8a8e1bb675a4b86c9ec7fa63755.jpg)

![5e302a83ae541861d18e89a29c9153d3f9e7a5269c6442865f2589a3562e10d7.jpg](../iclr_results/1152_LongMemEval_ Benchmarking Chat Assistants on Long-Term Interactive Memory/tables/5e302a83ae541861d18e89a29c9153d3f9e7a5269c6442865f2589a3562e10d7.jpg)

![7c246ac9c9e0528b4ad81206daea1b216254d1dededef0b3d2d0503ceb27e1b9.jpg](../iclr_results/1152_LongMemEval_ Benchmarking Chat Assistants on Long-Term Interactive Memory/tables/7c246ac9c9e0528b4ad81206daea1b216254d1dededef0b3d2d0503ceb27e1b9.jpg)

![7cd21ca82db2f824627b69363a5b290d2a63ff71677c71aa8175156cdf1fb1f9.jpg](../iclr_results/1152_LongMemEval_ Benchmarking Chat Assistants on Long-Term Interactive Memory/tables/7cd21ca82db2f824627b69363a5b290d2a63ff71677c71aa8175156cdf1fb1f9.jpg)

![83c3d6d2e040c552f0ef3aa049642672b5ef997cd44972613a1832110974c41e.jpg](../iclr_results/1152_LongMemEval_ Benchmarking Chat Assistants on Long-Term Interactive Memory/tables/83c3d6d2e040c552f0ef3aa049642672b5ef997cd44972613a1832110974c41e.jpg)

![a5fae1bddf72d555bafc618ba777cf8d75a821a5f1b6aaacbff9684a2ce85a8e.jpg](../iclr_results/1152_LongMemEval_ Benchmarking Chat Assistants on Long-Term Interactive Memory/tables/a5fae1bddf72d555bafc618ba777cf8d75a821a5f1b6aaacbff9684a2ce85a8e.jpg)

![ae584d69845e755c57df7176681a4ea8c39c9a2f7572e515d504b7869d844b70.jpg](../iclr_results/1152_LongMemEval_ Benchmarking Chat Assistants on Long-Term Interactive Memory/tables/ae584d69845e755c57df7176681a4ea8c39c9a2f7572e515d504b7869d844b70.jpg)

![c8250a7e0812bc8e755e8e91624686bf4052bfac585fe90aea8146bcc3a7cbb4.jpg](../iclr_results/1152_LongMemEval_ Benchmarking Chat Assistants on Long-Term Interactive Memory/tables/c8250a7e0812bc8e755e8e91624686bf4052bfac585fe90aea8146bcc3a7cbb4.jpg)

![dae6ad21997415850876b2c2eba90540f4e3c7584f663813d0dd73b6ad841842.jpg](../iclr_results/1152_LongMemEval_ Benchmarking Chat Assistants on Long-Term Interactive Memory/tables/dae6ad21997415850876b2c2eba90540f4e3c7584f663813d0dd73b6ad841842.jpg)

![df58370336a5dba68e74e60508d0a8672f4371edafdf8ee0ffa4f04a2e15d4cf.jpg](../iclr_results/1152_LongMemEval_ Benchmarking Chat Assistants on Long-Term Interactive Memory/tables/df58370336a5dba68e74e60508d0a8672f4371edafdf8ee0ffa4f04a2e15d4cf.jpg)

![e78937f329e4dad6ad87e52eba2740074763bbc76de82f590c28d45ee1398eed.jpg](../iclr_results/1152_LongMemEval_ Benchmarking Chat Assistants on Long-Term Interactive Memory/tables/e78937f329e4dad6ad87e52eba2740074763bbc76de82f590c28d45ee1398eed.jpg)

![f99992f7655733e7181ada4bb74d2128a362b398bbbdd14442d5d630c6331eff.jpg](../iclr_results/1152_LongMemEval_ Benchmarking Chat Assistants on Long-Term Interactive Memory/tables/f99992f7655733e7181ada4bb74d2128a362b398bbbdd14442d5d630c6331eff.jpg)

## ICLR: In-Context Learning of Representations


### Images

![06e1d20846d9a91767a7f7555e395da0240366270cbcec8b2c6dce33889c87a8.jpg](../iclr_results/1153_ICLR_ In-Context Learning of Representations/images/06e1d20846d9a91767a7f7555e395da0240366270cbcec8b2c6dce33889c87a8.jpg)

![088e63d5c93c77376d98d65aa436b1efc2dde5bace90980a0679f27156f43c0a.jpg](../iclr_results/1153_ICLR_ In-Context Learning of Representations/images/088e63d5c93c77376d98d65aa436b1efc2dde5bace90980a0679f27156f43c0a.jpg)

![08eab746a31df29bcf2aa88f88819ddcfd200558265aa395d3897f09d7e9d675.jpg](../iclr_results/1153_ICLR_ In-Context Learning of Representations/images/08eab746a31df29bcf2aa88f88819ddcfd200558265aa395d3897f09d7e9d675.jpg)

![2b64e0d807cf8db94f6405c7620bdca14a1afb3e9626a5e3a27912896e31c557.jpg](../iclr_results/1153_ICLR_ In-Context Learning of Representations/images/2b64e0d807cf8db94f6405c7620bdca14a1afb3e9626a5e3a27912896e31c557.jpg)

![356d687ce21e093f6089fb4d5628a09aa252ebcc006c65fe40e51032406fd87f.jpg](../iclr_results/1153_ICLR_ In-Context Learning of Representations/images/356d687ce21e093f6089fb4d5628a09aa252ebcc006c65fe40e51032406fd87f.jpg)

![3799dd3c12c9d1f3d24a99ffbdd8d69b69692c2a62fd6a3e11f760b419a05ea2.jpg](../iclr_results/1153_ICLR_ In-Context Learning of Representations/images/3799dd3c12c9d1f3d24a99ffbdd8d69b69692c2a62fd6a3e11f760b419a05ea2.jpg)

![3917bac91c1486d0d0ca7aea5e9cd1a44b53bf9142646f6fdb641e34923cc630.jpg](../iclr_results/1153_ICLR_ In-Context Learning of Representations/images/3917bac91c1486d0d0ca7aea5e9cd1a44b53bf9142646f6fdb641e34923cc630.jpg)

![4867683132d5064060834c8c7be8483d657ba9670fdfa08c103da008f71bdca1.jpg](../iclr_results/1153_ICLR_ In-Context Learning of Representations/images/4867683132d5064060834c8c7be8483d657ba9670fdfa08c103da008f71bdca1.jpg)

![4bf627c27af527e8d7eb8e6aa4373e438ddb736622b091220be0abf811f7d8c0.jpg](../iclr_results/1153_ICLR_ In-Context Learning of Representations/images/4bf627c27af527e8d7eb8e6aa4373e438ddb736622b091220be0abf811f7d8c0.jpg)

![5a9ca4c159f88bc24e680ffda2fa17b0120f3c096595953041ab22c75adb1d75.jpg](../iclr_results/1153_ICLR_ In-Context Learning of Representations/images/5a9ca4c159f88bc24e680ffda2fa17b0120f3c096595953041ab22c75adb1d75.jpg)

![62a19905b2f339814b9e73395c9b24f4063eb4c585e4e6ae5f96853bf411bb7b.jpg](../iclr_results/1153_ICLR_ In-Context Learning of Representations/images/62a19905b2f339814b9e73395c9b24f4063eb4c585e4e6ae5f96853bf411bb7b.jpg)

![66728f0c50a8f9908fbb0a88f99576998e8d7a9a775c68a7076fb53986ae27ec.jpg](../iclr_results/1153_ICLR_ In-Context Learning of Representations/images/66728f0c50a8f9908fbb0a88f99576998e8d7a9a775c68a7076fb53986ae27ec.jpg)

![6a1659af22cf270b3ced8e1b580307ab061229ba8893efaee576229bd2a244df.jpg](../iclr_results/1153_ICLR_ In-Context Learning of Representations/images/6a1659af22cf270b3ced8e1b580307ab061229ba8893efaee576229bd2a244df.jpg)

![99f5ac46eadc28fdb39a6fca398e3d6127eab3e94be6dd393ff3b447e66938b8.jpg](../iclr_results/1153_ICLR_ In-Context Learning of Representations/images/99f5ac46eadc28fdb39a6fca398e3d6127eab3e94be6dd393ff3b447e66938b8.jpg)

![b9b0a9277a5a706c238b03c3585a60f70b683d85b2c450db866f039efaa7dbe3.jpg](../iclr_results/1153_ICLR_ In-Context Learning of Representations/images/b9b0a9277a5a706c238b03c3585a60f70b683d85b2c450db866f039efaa7dbe3.jpg)

![c87ad6b44c5956c6b313361820805d5a0885a5367fde8b0d47116e5aeed55e91.jpg](../iclr_results/1153_ICLR_ In-Context Learning of Representations/images/c87ad6b44c5956c6b313361820805d5a0885a5367fde8b0d47116e5aeed55e91.jpg)

![caa4f0339ec1e6742e636ea4efc220446f004faae84d4d6103d0fa0c898a3df2.jpg](../iclr_results/1153_ICLR_ In-Context Learning of Representations/images/caa4f0339ec1e6742e636ea4efc220446f004faae84d4d6103d0fa0c898a3df2.jpg)

![d973978b5c7fdc7198bbd1ba7e2eb54f3b8c8079a0fe916cbe8762c3c4528f7e.jpg](../iclr_results/1153_ICLR_ In-Context Learning of Representations/images/d973978b5c7fdc7198bbd1ba7e2eb54f3b8c8079a0fe916cbe8762c3c4528f7e.jpg)

![dc9ee8450d8f34ff24611f92b2d69eef02ebb1deb751a644953436d844a9090c.jpg](../iclr_results/1153_ICLR_ In-Context Learning of Representations/images/dc9ee8450d8f34ff24611f92b2d69eef02ebb1deb751a644953436d844a9090c.jpg)

![fa555d995f1e0e8258dda414032bb63dc64bc489d99d0c341284b73b7a6ee556.jpg](../iclr_results/1153_ICLR_ In-Context Learning of Representations/images/fa555d995f1e0e8258dda414032bb63dc64bc489d99d0c341284b73b7a6ee556.jpg)

### Tables

![061a047673118aecb642ee62204d4965d22f8e7689f47b69574ac55fbba944ba.jpg](../iclr_results/1153_ICLR_ In-Context Learning of Representations/tables/061a047673118aecb642ee62204d4965d22f8e7689f47b69574ac55fbba944ba.jpg)

![be14cbbc626f08a75f9c35f9a25ae174131a71e2762a3df4c0803ce38474e221.jpg](../iclr_results/1153_ICLR_ In-Context Learning of Representations/tables/be14cbbc626f08a75f9c35f9a25ae174131a71e2762a3df4c0803ce38474e221.jpg)

## Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges


### Images

![0f85ae01dafe7cc1479d332c5a82abefb9e12c340d225300b0a184e045dfb140.jpg](../iclr_results/1154_Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges/images/0f85ae01dafe7cc1479d332c5a82abefb9e12c340d225300b0a184e045dfb140.jpg)

![1592e848f8a31457345e944176730421c268bf3f5cc52129ce05284ed3d62a78.jpg](../iclr_results/1154_Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges/images/1592e848f8a31457345e944176730421c268bf3f5cc52129ce05284ed3d62a78.jpg)

![24c312f7df0c0232ef0892a927a53350301289e0c972c1b3499c872445723adf.jpg](../iclr_results/1154_Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges/images/24c312f7df0c0232ef0892a927a53350301289e0c972c1b3499c872445723adf.jpg)

![40babb519ed5b81acb662a2f92441104aaffc3678f71e725455e1aeaa163ed72.jpg](../iclr_results/1154_Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges/images/40babb519ed5b81acb662a2f92441104aaffc3678f71e725455e1aeaa163ed72.jpg)

![52b0473bf02dcd52ff52792085bff0c29bcabd83a8b544056806178f057d77fb.jpg](../iclr_results/1154_Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges/images/52b0473bf02dcd52ff52792085bff0c29bcabd83a8b544056806178f057d77fb.jpg)

![6d729ca8831a6f185f34bf397b13687f44d5edfff0b80b06f8653f03cd529909.jpg](../iclr_results/1154_Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges/images/6d729ca8831a6f185f34bf397b13687f44d5edfff0b80b06f8653f03cd529909.jpg)

![76760e52b27c6a71047a23c131b98c353c0e74089895b8688cf52ca5083cd3cb.jpg](../iclr_results/1154_Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges/images/76760e52b27c6a71047a23c131b98c353c0e74089895b8688cf52ca5083cd3cb.jpg)

![76d606318778acf230ade62ed75c1003385fbdea58c74ecccd5e52bc20d65b95.jpg](../iclr_results/1154_Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges/images/76d606318778acf230ade62ed75c1003385fbdea58c74ecccd5e52bc20d65b95.jpg)

![ac6a527648f2d58323c4110ccf95e986d31f4316afa780e69c129cfc75472835.jpg](../iclr_results/1154_Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges/images/ac6a527648f2d58323c4110ccf95e986d31f4316afa780e69c129cfc75472835.jpg)

![d5cdd3a5ca651eeec77a7167d537476f9961f4a96ce5a308859854fef6aebfcc.jpg](../iclr_results/1154_Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges/images/d5cdd3a5ca651eeec77a7167d537476f9961f4a96ce5a308859854fef6aebfcc.jpg)

![f1d88aa3030f0af81d4a820005cdc11f6ea08740b4cc5dc1fc3c7984ee0b9dea.jpg](../iclr_results/1154_Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges/images/f1d88aa3030f0af81d4a820005cdc11f6ea08740b4cc5dc1fc3c7984ee0b9dea.jpg)

![f46ad33d971c814697bb91dc49f18a87105eab02ef7857e8e01c6bd610759a76.jpg](../iclr_results/1154_Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges/images/f46ad33d971c814697bb91dc49f18a87105eab02ef7857e8e01c6bd610759a76.jpg)

### Tables

![0ea88bceb65a34267799c725eec6bcb80cad0a4b57d50f265f4767163133b7a9.jpg](../iclr_results/1154_Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges/tables/0ea88bceb65a34267799c725eec6bcb80cad0a4b57d50f265f4767163133b7a9.jpg)

![31f20ab78df73ea41067a03274557fdbf09ec98809d6f50ac0403074af5cb41d.jpg](../iclr_results/1154_Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges/tables/31f20ab78df73ea41067a03274557fdbf09ec98809d6f50ac0403074af5cb41d.jpg)

![3e88e407f494b15b4cfc32c2a7ba8005ae3805a274d957110c6428480e075e56.jpg](../iclr_results/1154_Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges/tables/3e88e407f494b15b4cfc32c2a7ba8005ae3805a274d957110c6428480e075e56.jpg)

![5d5c5ac1bba3d02833d165a49b28813530bb79646177e90f23a7697a0e82a968.jpg](../iclr_results/1154_Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges/tables/5d5c5ac1bba3d02833d165a49b28813530bb79646177e90f23a7697a0e82a968.jpg)

![745e590e076bc7b23584ed315c4ff6fd21e3bd7b0756c916f1d54d7421205a94.jpg](../iclr_results/1154_Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges/tables/745e590e076bc7b23584ed315c4ff6fd21e3bd7b0756c916f1d54d7421205a94.jpg)

![9729231f2d77fe6e2903e8377fae77a66a9f2ddb1fbfea9c7abe26c72500e986.jpg](../iclr_results/1154_Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges/tables/9729231f2d77fe6e2903e8377fae77a66a9f2ddb1fbfea9c7abe26c72500e986.jpg)

![ca4da122d9046f5ecd98218ed5ca0cdc908da468b961c485152e9bbf730d657d.jpg](../iclr_results/1154_Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges/tables/ca4da122d9046f5ecd98218ed5ca0cdc908da468b961c485152e9bbf730d657d.jpg)

![f91b181ae1958678dbb8ff263b109ca8e63ab1f7ebba215046615880db771ca6.jpg](../iclr_results/1154_Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges/tables/f91b181ae1958678dbb8ff263b109ca8e63ab1f7ebba215046615880db771ca6.jpg)

## Streamlining Prediction in Bayesian Deep Learning


### Images

![1705bf0a6198f4142d5921b64a0b17e2e59ddec294f6ce8e43a0853bff9cad89.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/images/1705bf0a6198f4142d5921b64a0b17e2e59ddec294f6ce8e43a0853bff9cad89.jpg)

![3390111c3362e3616d0cce177a8fe0a0336debf4faf86265b56673ec7a678f81.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/images/3390111c3362e3616d0cce177a8fe0a0336debf4faf86265b56673ec7a678f81.jpg)

![554102c48b7f3c305c7382b96b99759b3940c3a086626a59670b70aa7e61acd0.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/images/554102c48b7f3c305c7382b96b99759b3940c3a086626a59670b70aa7e61acd0.jpg)

![79da49bcd09f028e23c3219f08783896506418c37d90daeee0b5bb0db0a2ba47.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/images/79da49bcd09f028e23c3219f08783896506418c37d90daeee0b5bb0db0a2ba47.jpg)

![85130758f0fc0f545b195e9f4f203e67c5381e94be0dec32290f31f63a432790.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/images/85130758f0fc0f545b195e9f4f203e67c5381e94be0dec32290f31f63a432790.jpg)

![96b150ae736674a4be705be98aba3ae5030025b207428f920f5d6c703496aa91.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/images/96b150ae736674a4be705be98aba3ae5030025b207428f920f5d6c703496aa91.jpg)

![a085fbfdb4193fefbd682520c56f2be0d4fd0697be0a50592e1f29f768a478b6.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/images/a085fbfdb4193fefbd682520c56f2be0d4fd0697be0a50592e1f29f768a478b6.jpg)

![a526614f3ed65a5668ac76925db2c775e8a861724cc16aab1481b564d25cc96f.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/images/a526614f3ed65a5668ac76925db2c775e8a861724cc16aab1481b564d25cc96f.jpg)

![cefe2976d4af3352e5fe40a49eded560d50be0b862917136a5c781e3329dea1f.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/images/cefe2976d4af3352e5fe40a49eded560d50be0b862917136a5c781e3329dea1f.jpg)

![d0a216af18a1a5082b857bc722ca23253b413541077e8a4f415116eb825226ff.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/images/d0a216af18a1a5082b857bc722ca23253b413541077e8a4f415116eb825226ff.jpg)

![df304d5ceba92f2690edab1f9382843ae49dc8fa5263283f6b923d651bcd7620.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/images/df304d5ceba92f2690edab1f9382843ae49dc8fa5263283f6b923d651bcd7620.jpg)

![f1748c40e628e22b81ca1a8e2ad2f85b607fc4e2419ca0b0bdc96ad1cdf11e8b.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/images/f1748c40e628e22b81ca1a8e2ad2f85b607fc4e2419ca0b0bdc96ad1cdf11e8b.jpg)

### Tables

![01eb948615b0a1e98fb657ae03180cb40a050c3089b549014752c4a754e56bb9.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/tables/01eb948615b0a1e98fb657ae03180cb40a050c3089b549014752c4a754e56bb9.jpg)

![244c22dcd837c866df254fda23233f992d63168fbe2e451369177d2a55be58ef.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/tables/244c22dcd837c866df254fda23233f992d63168fbe2e451369177d2a55be58ef.jpg)

![26029bef7f325c9192c02a7abf23ed70d8de0e482140b2ca50129be58876aa7b.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/tables/26029bef7f325c9192c02a7abf23ed70d8de0e482140b2ca50129be58876aa7b.jpg)

![2cdaf6a63bdebbf2ce36ea1a21aeb59a124d866d1daaf3707ea721dbcbbf0c73.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/tables/2cdaf6a63bdebbf2ce36ea1a21aeb59a124d866d1daaf3707ea721dbcbbf0c73.jpg)

![2d9aa6b830ea65844cb5d45bf310002d30b010bdb873fc45faa025c12a5089ca.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/tables/2d9aa6b830ea65844cb5d45bf310002d30b010bdb873fc45faa025c12a5089ca.jpg)

![4ef4d1db31c3fa84eac84cf9c6b6f48fda5166f8e9bcafbe1ed26ea531090490.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/tables/4ef4d1db31c3fa84eac84cf9c6b6f48fda5166f8e9bcafbe1ed26ea531090490.jpg)

![79e204c2011c0264db9c26d98a4d227b98470b7691e02168281795fddf6c773e.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/tables/79e204c2011c0264db9c26d98a4d227b98470b7691e02168281795fddf6c773e.jpg)

![7c8e29801049f03446df2fe1066f81b40e97ddae10d93e4ef5238f0ccab80114.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/tables/7c8e29801049f03446df2fe1066f81b40e97ddae10d93e4ef5238f0ccab80114.jpg)

![9d3e92850cf52c8c5c1d8e63d0a3c10c9c53258f9ef4c6e3e8c778a3e27c6e8f.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/tables/9d3e92850cf52c8c5c1d8e63d0a3c10c9c53258f9ef4c6e3e8c778a3e27c6e8f.jpg)

![a416db300808eeb95b1cab806198e29854996931bc5a5e53239ea7739074443d.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/tables/a416db300808eeb95b1cab806198e29854996931bc5a5e53239ea7739074443d.jpg)

![a8859e1ba28ffbae96ce6806f6efd14ac8f9510be2e93501536fd2abcccc5b6a.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/tables/a8859e1ba28ffbae96ce6806f6efd14ac8f9510be2e93501536fd2abcccc5b6a.jpg)

![cf9baa9b3352532274c742cfc31f6bcd638b6ca5a304f4706f6388e0ffde67b3.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/tables/cf9baa9b3352532274c742cfc31f6bcd638b6ca5a304f4706f6388e0ffde67b3.jpg)

![dc772d9f75b5276b88c83a7fa824fec175a23154048132a29bf5a7cf9d0614e4.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/tables/dc772d9f75b5276b88c83a7fa824fec175a23154048132a29bf5a7cf9d0614e4.jpg)

![deadfe8d1ef291b10475ddc556d4a21221180563bd5348857f2edecf44aa574d.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/tables/deadfe8d1ef291b10475ddc556d4a21221180563bd5348857f2edecf44aa574d.jpg)

![e6265e024b17d1b450f7d92ef55770474cabf71d64beb84c1e2e16122736a62b.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/tables/e6265e024b17d1b450f7d92ef55770474cabf71d64beb84c1e2e16122736a62b.jpg)

![ebb112680d89e5c04a30c2c726023d0e78c0e4692755b9d61c93ebf87b9e48fa.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/tables/ebb112680d89e5c04a30c2c726023d0e78c0e4692755b9d61c93ebf87b9e48fa.jpg)

![ebe014a0f7be0d12a1eaec7e01e2f07dbbf3a688cc25799b2088d3034bd64b2d.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/tables/ebe014a0f7be0d12a1eaec7e01e2f07dbbf3a688cc25799b2088d3034bd64b2d.jpg)

![fa583f778b46f780bdfaa4136e598bb0bee684ebe7e7e7bb7fb7b2ad007f6324.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/tables/fa583f778b46f780bdfaa4136e598bb0bee684ebe7e7e7bb7fb7b2ad007f6324.jpg)

![ff115274a83a95eed50cca5b8b0c211cb453b1af198e26ab690602658852c001.jpg](../iclr_results/1155_Streamlining Prediction in Bayesian Deep Learning/tables/ff115274a83a95eed50cca5b8b0c211cb453b1af198e26ab690602658852c001.jpg)

## Think Thrice Before You Act: Progressive Thought Refinement in Large Language Models


### Images

![6d7d054be5e66b0e347cd9f903bfc55e7520e0c9e192a8a8396f53638cb5a9bc.jpg](../iclr_results/1156_Think Thrice Before You Act_ Progressive Thought Refinement in Large Language Models/images/6d7d054be5e66b0e347cd9f903bfc55e7520e0c9e192a8a8396f53638cb5a9bc.jpg)

![741a32ed1c67708728b088d96ab953013af1ae8f7703383dc24cca3ba20e4c00.jpg](../iclr_results/1156_Think Thrice Before You Act_ Progressive Thought Refinement in Large Language Models/images/741a32ed1c67708728b088d96ab953013af1ae8f7703383dc24cca3ba20e4c00.jpg)

![b3016d702b8f3ebf08c93ecbae197d64b5b0cdd207f5f31154a947416eac611e.jpg](../iclr_results/1156_Think Thrice Before You Act_ Progressive Thought Refinement in Large Language Models/images/b3016d702b8f3ebf08c93ecbae197d64b5b0cdd207f5f31154a947416eac611e.jpg)

![b97d17a74d68e7711fe780c7342576a788a429d49a8b6b8d4f363ef4038fb9bf.jpg](../iclr_results/1156_Think Thrice Before You Act_ Progressive Thought Refinement in Large Language Models/images/b97d17a74d68e7711fe780c7342576a788a429d49a8b6b8d4f363ef4038fb9bf.jpg)

![be1b9bd062a25eac181218ae7beebf6b8181c11e56a0dd2f7372e4e951424ddf.jpg](../iclr_results/1156_Think Thrice Before You Act_ Progressive Thought Refinement in Large Language Models/images/be1b9bd062a25eac181218ae7beebf6b8181c11e56a0dd2f7372e4e951424ddf.jpg)

![ed82656cd0f4ace04e8f08a35420d9eb5be71bdda5ed4cd0ffa2574b55116e4d.jpg](../iclr_results/1156_Think Thrice Before You Act_ Progressive Thought Refinement in Large Language Models/images/ed82656cd0f4ace04e8f08a35420d9eb5be71bdda5ed4cd0ffa2574b55116e4d.jpg)

![f154cb952dfce90bed4fb6b55c46d5ee9cf6db7d8f6532a455359c20abf02558.jpg](../iclr_results/1156_Think Thrice Before You Act_ Progressive Thought Refinement in Large Language Models/images/f154cb952dfce90bed4fb6b55c46d5ee9cf6db7d8f6532a455359c20abf02558.jpg)

### Tables

![232cfcca621054dc1cbb24543e17f6a01f3164e813b1bf67865eb38ce745ada2.jpg](../iclr_results/1156_Think Thrice Before You Act_ Progressive Thought Refinement in Large Language Models/tables/232cfcca621054dc1cbb24543e17f6a01f3164e813b1bf67865eb38ce745ada2.jpg)

![2daf3e3ebfb3f45972060b6f0874d7ee77c6025e0ea738931a3bf2729e26c588.jpg](../iclr_results/1156_Think Thrice Before You Act_ Progressive Thought Refinement in Large Language Models/tables/2daf3e3ebfb3f45972060b6f0874d7ee77c6025e0ea738931a3bf2729e26c588.jpg)

![466f09cc00f8d0f61d61d2dd563424da5a3110e377b03f7bc6e2476461a1e78b.jpg](../iclr_results/1156_Think Thrice Before You Act_ Progressive Thought Refinement in Large Language Models/tables/466f09cc00f8d0f61d61d2dd563424da5a3110e377b03f7bc6e2476461a1e78b.jpg)

![54155065d757494f23d187bd5c41b072f11119e979b7073c8dd66efefff2c812.jpg](../iclr_results/1156_Think Thrice Before You Act_ Progressive Thought Refinement in Large Language Models/tables/54155065d757494f23d187bd5c41b072f11119e979b7073c8dd66efefff2c812.jpg)

![642d6d73f25202b246943be5a439da83343e2c1ac51a2c7ae18271ad6adec011.jpg](../iclr_results/1156_Think Thrice Before You Act_ Progressive Thought Refinement in Large Language Models/tables/642d6d73f25202b246943be5a439da83343e2c1ac51a2c7ae18271ad6adec011.jpg)

![65e3c4f722bbf8b657fc2e1286edf447f52b3ae30afeed91f4dff90ab25df3f1.jpg](../iclr_results/1156_Think Thrice Before You Act_ Progressive Thought Refinement in Large Language Models/tables/65e3c4f722bbf8b657fc2e1286edf447f52b3ae30afeed91f4dff90ab25df3f1.jpg)

![949940a894b77007609e8a6a17c22c89636f5f9b3cb1e97769c6d3a3cc0aa36d.jpg](../iclr_results/1156_Think Thrice Before You Act_ Progressive Thought Refinement in Large Language Models/tables/949940a894b77007609e8a6a17c22c89636f5f9b3cb1e97769c6d3a3cc0aa36d.jpg)

![e3325c5bf67d410c89d2df76b647ce0d1d70ddff4bed26888c3d44c2e3e8713a.jpg](../iclr_results/1156_Think Thrice Before You Act_ Progressive Thought Refinement in Large Language Models/tables/e3325c5bf67d410c89d2df76b647ce0d1d70ddff4bed26888c3d44c2e3e8713a.jpg)

![e6a49d54428098e65555d4184ccf930c4bb82034efd908baf484a0eeda8661f0.jpg](../iclr_results/1156_Think Thrice Before You Act_ Progressive Thought Refinement in Large Language Models/tables/e6a49d54428098e65555d4184ccf930c4bb82034efd908baf484a0eeda8661f0.jpg)

![f4584035b4029d30301ea594b739a1288c0684fed1dccf7572f08d324f25af68.jpg](../iclr_results/1156_Think Thrice Before You Act_ Progressive Thought Refinement in Large Language Models/tables/f4584035b4029d30301ea594b739a1288c0684fed1dccf7572f08d324f25af68.jpg)

## TEASER: Token Enhanced Spatial Modeling for Expressions Reconstruction


### Images

![0478860c3ceb998dd297031004f89a774f1fa98d639e9412582995b9b9e2b5e3.jpg](../iclr_results/1157_TEASER_ Token Enhanced Spatial Modeling for Expressions Reconstruction/images/0478860c3ceb998dd297031004f89a774f1fa98d639e9412582995b9b9e2b5e3.jpg)

![0ea2845cbb21f4d75d8b1801596b477d01f9d198a18d9fb504f83b82df3934c3.jpg](../iclr_results/1157_TEASER_ Token Enhanced Spatial Modeling for Expressions Reconstruction/images/0ea2845cbb21f4d75d8b1801596b477d01f9d198a18d9fb504f83b82df3934c3.jpg)

![0f70284229b0d91c40e5c002d1419bb51d60baf8d487498b215e750ad3839e81.jpg](../iclr_results/1157_TEASER_ Token Enhanced Spatial Modeling for Expressions Reconstruction/images/0f70284229b0d91c40e5c002d1419bb51d60baf8d487498b215e750ad3839e81.jpg)

![314135cf8e6834e16ea77bfc3422ab443ce964c80beb84174c42a635ba1ecc44.jpg](../iclr_results/1157_TEASER_ Token Enhanced Spatial Modeling for Expressions Reconstruction/images/314135cf8e6834e16ea77bfc3422ab443ce964c80beb84174c42a635ba1ecc44.jpg)

![375d8698ceb9243cc2fcb5518d1eb8ea24f3f3ebf1e9027274e7e4220c48f621.jpg](../iclr_results/1157_TEASER_ Token Enhanced Spatial Modeling for Expressions Reconstruction/images/375d8698ceb9243cc2fcb5518d1eb8ea24f3f3ebf1e9027274e7e4220c48f621.jpg)

![483bff718208e416235d39a472fb73e728e6f83a987b324ea401dbe42765d23b.jpg](../iclr_results/1157_TEASER_ Token Enhanced Spatial Modeling for Expressions Reconstruction/images/483bff718208e416235d39a472fb73e728e6f83a987b324ea401dbe42765d23b.jpg)

![793821f726c219783a9e942bd45acc621d41cd046d87cc31a032545395676b38.jpg](../iclr_results/1157_TEASER_ Token Enhanced Spatial Modeling for Expressions Reconstruction/images/793821f726c219783a9e942bd45acc621d41cd046d87cc31a032545395676b38.jpg)

![7bd63e9f73517e7f31464fcf00af4c5633b005d1748b3aefeee4ec6776e78597.jpg](../iclr_results/1157_TEASER_ Token Enhanced Spatial Modeling for Expressions Reconstruction/images/7bd63e9f73517e7f31464fcf00af4c5633b005d1748b3aefeee4ec6776e78597.jpg)

![831d027924740ecc40234a6408c7a22a976d6d89c8315a8db53c8331be45f6af.jpg](../iclr_results/1157_TEASER_ Token Enhanced Spatial Modeling for Expressions Reconstruction/images/831d027924740ecc40234a6408c7a22a976d6d89c8315a8db53c8331be45f6af.jpg)

![9b97cafdc6c7fa9a045e86c8de0c21aca4cf3ff33278123c4d864d834fdc7245.jpg](../iclr_results/1157_TEASER_ Token Enhanced Spatial Modeling for Expressions Reconstruction/images/9b97cafdc6c7fa9a045e86c8de0c21aca4cf3ff33278123c4d864d834fdc7245.jpg)

![a44afbf15b79a368afcf3daf1b26f087789145d0cbcbd3548a622a65e8932f12.jpg](../iclr_results/1157_TEASER_ Token Enhanced Spatial Modeling for Expressions Reconstruction/images/a44afbf15b79a368afcf3daf1b26f087789145d0cbcbd3548a622a65e8932f12.jpg)

![b632b912bdb87d5a5aa30b4856c4d3062749e63d3bfa012f9ac544ed0b66e445.jpg](../iclr_results/1157_TEASER_ Token Enhanced Spatial Modeling for Expressions Reconstruction/images/b632b912bdb87d5a5aa30b4856c4d3062749e63d3bfa012f9ac544ed0b66e445.jpg)

![efc453161a5b23675665815a5e0220d30919ac19f996561a8c24de4c2cc5999d.jpg](../iclr_results/1157_TEASER_ Token Enhanced Spatial Modeling for Expressions Reconstruction/images/efc453161a5b23675665815a5e0220d30919ac19f996561a8c24de4c2cc5999d.jpg)

![fe050b9913b76add759d02b2dfd76b708819b43c40ac0123e3a160cac913ac5b.jpg](../iclr_results/1157_TEASER_ Token Enhanced Spatial Modeling for Expressions Reconstruction/images/fe050b9913b76add759d02b2dfd76b708819b43c40ac0123e3a160cac913ac5b.jpg)

### Tables

![2ba63f34370932925087247cf0944687dfd68ba5e2208ce1ce803a3c6f63a1cf.jpg](../iclr_results/1157_TEASER_ Token Enhanced Spatial Modeling for Expressions Reconstruction/tables/2ba63f34370932925087247cf0944687dfd68ba5e2208ce1ce803a3c6f63a1cf.jpg)

![4b292f10234d0662b25eb9afeef5301a0fc63964180bda11df084a6c2cf03731.jpg](../iclr_results/1157_TEASER_ Token Enhanced Spatial Modeling for Expressions Reconstruction/tables/4b292f10234d0662b25eb9afeef5301a0fc63964180bda11df084a6c2cf03731.jpg)

![92924f65dedf288442654d70591f7d033dd2746c74ed2c4cdb4bc4bd270b1fa6.jpg](../iclr_results/1157_TEASER_ Token Enhanced Spatial Modeling for Expressions Reconstruction/tables/92924f65dedf288442654d70591f7d033dd2746c74ed2c4cdb4bc4bd270b1fa6.jpg)

![c98850d289b71f1e7158f83ac4038eb7139e9c10822e949679cbd53b25cc34f3.jpg](../iclr_results/1157_TEASER_ Token Enhanced Spatial Modeling for Expressions Reconstruction/tables/c98850d289b71f1e7158f83ac4038eb7139e9c10822e949679cbd53b25cc34f3.jpg)

![d6383029b9974a987edac09ab2bb7cd7681952c7b0566948bff2b756aaa2ccae.jpg](../iclr_results/1157_TEASER_ Token Enhanced Spatial Modeling for Expressions Reconstruction/tables/d6383029b9974a987edac09ab2bb7cd7681952c7b0566948bff2b756aaa2ccae.jpg)

## Leveraging Flatness to Improve Information-Theoretic Generalization Bounds for SGD


### Images

![0bce7b314979528e64e65fc1c01984b39fd0d923ad7ffce52dbfe904e2706087.jpg](../iclr_results/1158_Leveraging Flatness to Improve Information-Theoretic Generalization Bounds for SGD/images/0bce7b314979528e64e65fc1c01984b39fd0d923ad7ffce52dbfe904e2706087.jpg)

![1e01f196ea0d3af93c16bbcd62445e2d516e6789c76375e877fd4865be7bd543.jpg](../iclr_results/1158_Leveraging Flatness to Improve Information-Theoretic Generalization Bounds for SGD/images/1e01f196ea0d3af93c16bbcd62445e2d516e6789c76375e877fd4865be7bd543.jpg)

![297da1e802348df393b0190a0e99d0cd57e56fd3ce8eed2bafce4fa322c9e2b5.jpg](../iclr_results/1158_Leveraging Flatness to Improve Information-Theoretic Generalization Bounds for SGD/images/297da1e802348df393b0190a0e99d0cd57e56fd3ce8eed2bafce4fa322c9e2b5.jpg)

![5323a689613859adc7bfc4584748cfdd76fcf709dc87ffdc2e7aeb0f8f7bd715.jpg](../iclr_results/1158_Leveraging Flatness to Improve Information-Theoretic Generalization Bounds for SGD/images/5323a689613859adc7bfc4584748cfdd76fcf709dc87ffdc2e7aeb0f8f7bd715.jpg)

![7501991b382043370020ee2432e98e3c77937a88f04b60e66fa2242c6f173595.jpg](../iclr_results/1158_Leveraging Flatness to Improve Information-Theoretic Generalization Bounds for SGD/images/7501991b382043370020ee2432e98e3c77937a88f04b60e66fa2242c6f173595.jpg)

![7d67c8edfa7a4acb328f0cba629a775967cd1eb7f1ef157ef593de75446c5b0c.jpg](../iclr_results/1158_Leveraging Flatness to Improve Information-Theoretic Generalization Bounds for SGD/images/7d67c8edfa7a4acb328f0cba629a775967cd1eb7f1ef157ef593de75446c5b0c.jpg)

![9381da52cc831cf72a163ec268b8d9506b13e44eab46c5bdeaa3b94284c0e8a2.jpg](../iclr_results/1158_Leveraging Flatness to Improve Information-Theoretic Generalization Bounds for SGD/images/9381da52cc831cf72a163ec268b8d9506b13e44eab46c5bdeaa3b94284c0e8a2.jpg)

![a8455006e5c7b40c28816ec888967a6e09193345e4c09475f0223448b7e09b5b.jpg](../iclr_results/1158_Leveraging Flatness to Improve Information-Theoretic Generalization Bounds for SGD/images/a8455006e5c7b40c28816ec888967a6e09193345e4c09475f0223448b7e09b5b.jpg)

![b30b02c1f285f2b2e6730ee2cb0bf159f74a53db779a764e6ec4377cbee2f08d.jpg](../iclr_results/1158_Leveraging Flatness to Improve Information-Theoretic Generalization Bounds for SGD/images/b30b02c1f285f2b2e6730ee2cb0bf159f74a53db779a764e6ec4377cbee2f08d.jpg)

![bbcba98fe0798812de29ddb9a6eac3b69107154db4b91de438181833003da93d.jpg](../iclr_results/1158_Leveraging Flatness to Improve Information-Theoretic Generalization Bounds for SGD/images/bbcba98fe0798812de29ddb9a6eac3b69107154db4b91de438181833003da93d.jpg)

![e037119898e606880db707332630424cee8cb8e45e15e51a4534dbc3857f86a5.jpg](../iclr_results/1158_Leveraging Flatness to Improve Information-Theoretic Generalization Bounds for SGD/images/e037119898e606880db707332630424cee8cb8e45e15e51a4534dbc3857f86a5.jpg)

![eb504386611214a37f9a145acbb154fa08c7f2702dd2f6c6e5d536bd890cbca2.jpg](../iclr_results/1158_Leveraging Flatness to Improve Information-Theoretic Generalization Bounds for SGD/images/eb504386611214a37f9a145acbb154fa08c7f2702dd2f6c6e5d536bd890cbca2.jpg)

![f0e1986faa008ac2b0e496c240d9202d5e4129d3830eb7ab2058402a142ba958.jpg](../iclr_results/1158_Leveraging Flatness to Improve Information-Theoretic Generalization Bounds for SGD/images/f0e1986faa008ac2b0e496c240d9202d5e4129d3830eb7ab2058402a142ba958.jpg)

### Tables

![e1994136e3f52fa3ba6ffe0d547faec200771e0d2d27f7387648856d9d71eeb4.jpg](../iclr_results/1158_Leveraging Flatness to Improve Information-Theoretic Generalization Bounds for SGD/tables/e1994136e3f52fa3ba6ffe0d547faec200771e0d2d27f7387648856d9d71eeb4.jpg)

## Boltzmann priors for Implicit Transfer Operators


### Images

![051cd607cca1f51c94957cb8737b3f3b37b75384823492d80f9c44bd6205e280.jpg](../iclr_results/1159_Boltzmann priors for Implicit Transfer Operators/images/051cd607cca1f51c94957cb8737b3f3b37b75384823492d80f9c44bd6205e280.jpg)

![051d6aa406065906cdd97bf5d61f182a4b24929cacc99ee281065ba1049f2e1f.jpg](../iclr_results/1159_Boltzmann priors for Implicit Transfer Operators/images/051d6aa406065906cdd97bf5d61f182a4b24929cacc99ee281065ba1049f2e1f.jpg)

![12661fac2de05989a9b2e9c2c7054a41876abf792376a85059c24aaf1078de31.jpg](../iclr_results/1159_Boltzmann priors for Implicit Transfer Operators/images/12661fac2de05989a9b2e9c2c7054a41876abf792376a85059c24aaf1078de31.jpg)

![1c1c6d8ea0edc3b419f1bdace1f555484fc5a15fe019c3a1004cc27677dc8ff1.jpg](../iclr_results/1159_Boltzmann priors for Implicit Transfer Operators/images/1c1c6d8ea0edc3b419f1bdace1f555484fc5a15fe019c3a1004cc27677dc8ff1.jpg)

![27f9f4c0e62998320505624eb99086aab7e689483573b0d1b0f7e084f947fa89.jpg](../iclr_results/1159_Boltzmann priors for Implicit Transfer Operators/images/27f9f4c0e62998320505624eb99086aab7e689483573b0d1b0f7e084f947fa89.jpg)

![34b787977f39cc201435c57feddc7ab22333d999f7fcf97855777754a32eb70a.jpg](../iclr_results/1159_Boltzmann priors for Implicit Transfer Operators/images/34b787977f39cc201435c57feddc7ab22333d999f7fcf97855777754a32eb70a.jpg)

![4c58e4fa7c07d446088ba5275bbc4b506e57df2346da704b9727f00c1510e473.jpg](../iclr_results/1159_Boltzmann priors for Implicit Transfer Operators/images/4c58e4fa7c07d446088ba5275bbc4b506e57df2346da704b9727f00c1510e473.jpg)

![57f7265b3b25212a1138f0483d2b1552f8e131ef344656b7e2ff2ffca4823b5f.jpg](../iclr_results/1159_Boltzmann priors for Implicit Transfer Operators/images/57f7265b3b25212a1138f0483d2b1552f8e131ef344656b7e2ff2ffca4823b5f.jpg)

![97b2f38c8e9abf9b6ae1259aa47bb40d26fd141bc739d36d7c649b5f83feca64.jpg](../iclr_results/1159_Boltzmann priors for Implicit Transfer Operators/images/97b2f38c8e9abf9b6ae1259aa47bb40d26fd141bc739d36d7c649b5f83feca64.jpg)

![b43ad77ae2e2e1b44c4298578e6a5646f5e30f5446fe0ca4c7eb167001affb79.jpg](../iclr_results/1159_Boltzmann priors for Implicit Transfer Operators/images/b43ad77ae2e2e1b44c4298578e6a5646f5e30f5446fe0ca4c7eb167001affb79.jpg)

![c3597bda6e9b24b42fb52ecc2315c0a3ad3a3e5514a8fbc48fe1b612dc3959ef.jpg](../iclr_results/1159_Boltzmann priors for Implicit Transfer Operators/images/c3597bda6e9b24b42fb52ecc2315c0a3ad3a3e5514a8fbc48fe1b612dc3959ef.jpg)

![c580395d20ce6dd1662e3d87207d5d88183a579dd9be9c762e583870d0aaf8a4.jpg](../iclr_results/1159_Boltzmann priors for Implicit Transfer Operators/images/c580395d20ce6dd1662e3d87207d5d88183a579dd9be9c762e583870d0aaf8a4.jpg)

![e47c29fc9cf871f03c5bb562e77eaf5d41f5a79512afb660ad34c1f334dffa3f.jpg](../iclr_results/1159_Boltzmann priors for Implicit Transfer Operators/images/e47c29fc9cf871f03c5bb562e77eaf5d41f5a79512afb660ad34c1f334dffa3f.jpg)

![fee4f828459e152f0d4309fa0f936d71d158c852478756e7183254b59587be99.jpg](../iclr_results/1159_Boltzmann priors for Implicit Transfer Operators/images/fee4f828459e152f0d4309fa0f936d71d158c852478756e7183254b59587be99.jpg)

### Tables

![6c887623161615ca9a813b0c02c2884258c5a5f1e4ef5cce418c87d5404e2a71.jpg](../iclr_results/1159_Boltzmann priors for Implicit Transfer Operators/tables/6c887623161615ca9a813b0c02c2884258c5a5f1e4ef5cce418c87d5404e2a71.jpg)

![b95e84a8254279c7eecf705c7ce7124b4fa33959ed3de3970aeabca456522248.jpg](../iclr_results/1159_Boltzmann priors for Implicit Transfer Operators/tables/b95e84a8254279c7eecf705c7ce7124b4fa33959ed3de3970aeabca456522248.jpg)

![daae6d193b8eb0edcfd510013b18fde49a1177a6e6ffcbced432ab7ec897918c.jpg](../iclr_results/1159_Boltzmann priors for Implicit Transfer Operators/tables/daae6d193b8eb0edcfd510013b18fde49a1177a6e6ffcbced432ab7ec897918c.jpg)

![f11a4d357cdff6bcaf68e91818de0df020ac6e9e5569561748ad1bd3f542b272.jpg](../iclr_results/1159_Boltzmann priors for Implicit Transfer Operators/tables/f11a4d357cdff6bcaf68e91818de0df020ac6e9e5569561748ad1bd3f542b272.jpg)

## Diverse Preference Learning for Capabilities and Alignment


### Images

![0cc803076dc28251cfaa62abe087c759ceccc40e59013ad3c0206327f442e650.jpg](../iclr_results/1160_Diverse Preference Learning for Capabilities and Alignment/images/0cc803076dc28251cfaa62abe087c759ceccc40e59013ad3c0206327f442e650.jpg)

![160d381daff6708d590d0bdba22e275b02ce64ba1625130aa3573a297fe9aefe.jpg](../iclr_results/1160_Diverse Preference Learning for Capabilities and Alignment/images/160d381daff6708d590d0bdba22e275b02ce64ba1625130aa3573a297fe9aefe.jpg)

![3767a8f8f27ec413792afbdf319711a12987b31ab08e860cc1162192a0cd5a73.jpg](../iclr_results/1160_Diverse Preference Learning for Capabilities and Alignment/images/3767a8f8f27ec413792afbdf319711a12987b31ab08e860cc1162192a0cd5a73.jpg)

![3f71412ea949070b445d821cd525f902ae179d2d3959f4a7de8c34279627636e.jpg](../iclr_results/1160_Diverse Preference Learning for Capabilities and Alignment/images/3f71412ea949070b445d821cd525f902ae179d2d3959f4a7de8c34279627636e.jpg)

![40511ee3191deeac79cadcaab469f5c992b634f1dff7ef3a6de59badb7d6a4aa.jpg](../iclr_results/1160_Diverse Preference Learning for Capabilities and Alignment/images/40511ee3191deeac79cadcaab469f5c992b634f1dff7ef3a6de59badb7d6a4aa.jpg)

![4bfcec4e97ca986d131d24d33884574a8f02f52c964341582cd2c6726536f9d4.jpg](../iclr_results/1160_Diverse Preference Learning for Capabilities and Alignment/images/4bfcec4e97ca986d131d24d33884574a8f02f52c964341582cd2c6726536f9d4.jpg)

![6cc37accab0cde674bf6d2f06cfa75d921e9ad6811cc29f25eaf83c3757ff137.jpg](../iclr_results/1160_Diverse Preference Learning for Capabilities and Alignment/images/6cc37accab0cde674bf6d2f06cfa75d921e9ad6811cc29f25eaf83c3757ff137.jpg)

![888d49c2427bf6aa848ace1de95468f8e5c7e4ccaea083acb2b998207f8df0d8.jpg](../iclr_results/1160_Diverse Preference Learning for Capabilities and Alignment/images/888d49c2427bf6aa848ace1de95468f8e5c7e4ccaea083acb2b998207f8df0d8.jpg)

![a8c0502164d817a7dc40a63753228fe72fdca5514fab0f45baa76dfebd3488ee.jpg](../iclr_results/1160_Diverse Preference Learning for Capabilities and Alignment/images/a8c0502164d817a7dc40a63753228fe72fdca5514fab0f45baa76dfebd3488ee.jpg)

![b4bf8697d3d53e019411e5b6dc0d7792c2802f7f7a17e361ec826bfd5442d912.jpg](../iclr_results/1160_Diverse Preference Learning for Capabilities and Alignment/images/b4bf8697d3d53e019411e5b6dc0d7792c2802f7f7a17e361ec826bfd5442d912.jpg)

![d7e726487297a2907a1b18ac0e5db7be411fddc4d8e98e2397bc42cd22967f34.jpg](../iclr_results/1160_Diverse Preference Learning for Capabilities and Alignment/images/d7e726487297a2907a1b18ac0e5db7be411fddc4d8e98e2397bc42cd22967f34.jpg)

![e4ca429adfd721e05953d534e7a7fa384c997d46a8ce2050eeeda2b914b2b279.jpg](../iclr_results/1160_Diverse Preference Learning for Capabilities and Alignment/images/e4ca429adfd721e05953d534e7a7fa384c997d46a8ce2050eeeda2b914b2b279.jpg)

![f2dbffabcb2a61d2b5535fa275112d33267f7371f2fed25212642aa5073a6d41.jpg](../iclr_results/1160_Diverse Preference Learning for Capabilities and Alignment/images/f2dbffabcb2a61d2b5535fa275112d33267f7371f2fed25212642aa5073a6d41.jpg)

### Tables

![04c58c3b6d6540cea1bd0780ebb411dd6ac887194e3f9ca2217db1cdb4a12b1d.jpg](../iclr_results/1160_Diverse Preference Learning for Capabilities and Alignment/tables/04c58c3b6d6540cea1bd0780ebb411dd6ac887194e3f9ca2217db1cdb4a12b1d.jpg)

![4319713c4884be39e36122f9fb6abf9b9b28d20389a911df2487a8eabba0222d.jpg](../iclr_results/1160_Diverse Preference Learning for Capabilities and Alignment/tables/4319713c4884be39e36122f9fb6abf9b9b28d20389a911df2487a8eabba0222d.jpg)

![4b108fa37ceb80f5ea770fd3e32a68c7f50a6b4843cf800057be0df7538814c4.jpg](../iclr_results/1160_Diverse Preference Learning for Capabilities and Alignment/tables/4b108fa37ceb80f5ea770fd3e32a68c7f50a6b4843cf800057be0df7538814c4.jpg)

## CONTRA: Conformal Prediction Region via Normalizing Flow Transformation


### Images

![00c152b2f40ccd2fae5eff2e2c6f5fe8aa7062bbc3dfe0cbbccee3024f7196bf.jpg](../iclr_results/1161_CONTRA_ Conformal Prediction Region via Normalizing Flow Transformation/images/00c152b2f40ccd2fae5eff2e2c6f5fe8aa7062bbc3dfe0cbbccee3024f7196bf.jpg)

![352cc290b5e80b04d852c44eb509b4d471cb4225284c034fc27b9db3fcf82190.jpg](../iclr_results/1161_CONTRA_ Conformal Prediction Region via Normalizing Flow Transformation/images/352cc290b5e80b04d852c44eb509b4d471cb4225284c034fc27b9db3fcf82190.jpg)

![35f5308e7e3e5a8e5e8fbcb958a23349205c8a870f804d809666ecaa06e06268.jpg](../iclr_results/1161_CONTRA_ Conformal Prediction Region via Normalizing Flow Transformation/images/35f5308e7e3e5a8e5e8fbcb958a23349205c8a870f804d809666ecaa06e06268.jpg)

![79f732716ac73c1230e31890282acb750e9067fefaedeeb96127abd59ab9842b.jpg](../iclr_results/1161_CONTRA_ Conformal Prediction Region via Normalizing Flow Transformation/images/79f732716ac73c1230e31890282acb750e9067fefaedeeb96127abd59ab9842b.jpg)

![b6be1a3be25c9e551690bbaec271e4f905bcc00d8f1c4290b843919c7379c2a8.jpg](../iclr_results/1161_CONTRA_ Conformal Prediction Region via Normalizing Flow Transformation/images/b6be1a3be25c9e551690bbaec271e4f905bcc00d8f1c4290b843919c7379c2a8.jpg)

![c48b8386d209352f095c5ff97cd2ca510bd66a41ed2bba82a36771583e7379ff.jpg](../iclr_results/1161_CONTRA_ Conformal Prediction Region via Normalizing Flow Transformation/images/c48b8386d209352f095c5ff97cd2ca510bd66a41ed2bba82a36771583e7379ff.jpg)

![d8fe4b9937c551710421a4ed83a970bdd3f8bc15bab7003d1d7c059ca4442f71.jpg](../iclr_results/1161_CONTRA_ Conformal Prediction Region via Normalizing Flow Transformation/images/d8fe4b9937c551710421a4ed83a970bdd3f8bc15bab7003d1d7c059ca4442f71.jpg)

### Tables

![2c0809d367cfc4c39c2ad5ddb6e305354fe270aad7cb1f99ae16e191b6c3e777.jpg](../iclr_results/1161_CONTRA_ Conformal Prediction Region via Normalizing Flow Transformation/tables/2c0809d367cfc4c39c2ad5ddb6e305354fe270aad7cb1f99ae16e191b6c3e777.jpg)

![535c363b18e2988335221932d4749645c00575d2606dbcdb37ff37fabe097b9a.jpg](../iclr_results/1161_CONTRA_ Conformal Prediction Region via Normalizing Flow Transformation/tables/535c363b18e2988335221932d4749645c00575d2606dbcdb37ff37fabe097b9a.jpg)

![6947ddab74d44c458da0e6902fc40b19d7360bbb1cc4bb669c435a96f0369dbb.jpg](../iclr_results/1161_CONTRA_ Conformal Prediction Region via Normalizing Flow Transformation/tables/6947ddab74d44c458da0e6902fc40b19d7360bbb1cc4bb669c435a96f0369dbb.jpg)

![9bde761f61eb141abd1c17b70664d5d6961bb432ddde80ca06b03dc83af00b0a.jpg](../iclr_results/1161_CONTRA_ Conformal Prediction Region via Normalizing Flow Transformation/tables/9bde761f61eb141abd1c17b70664d5d6961bb432ddde80ca06b03dc83af00b0a.jpg)

![f479ce5f11c6f6333400b948e066438b0abf93816342368901653311881c99e6.jpg](../iclr_results/1161_CONTRA_ Conformal Prediction Region via Normalizing Flow Transformation/tables/f479ce5f11c6f6333400b948e066438b0abf93816342368901653311881c99e6.jpg)

## Bayesian WeakS-to-Strong from Text Classification to Generation


### Images

![2206573867edd9e9f616a8db35f9e1cb3975d8747be70d13eebfeca729e61ca8.jpg](../iclr_results/1162_Bayesian WeakS-to-Strong from Text Classification to Generation/images/2206573867edd9e9f616a8db35f9e1cb3975d8747be70d13eebfeca729e61ca8.jpg)

![360dce275f4a63b44411087b1b215d5565cf3bbe3c6b83e8d0402b940be84724.jpg](../iclr_results/1162_Bayesian WeakS-to-Strong from Text Classification to Generation/images/360dce275f4a63b44411087b1b215d5565cf3bbe3c6b83e8d0402b940be84724.jpg)

![85495a0d6d5c545d9c7c1ce51cefc9cb6b0084d0c70406cbdda207652c836b93.jpg](../iclr_results/1162_Bayesian WeakS-to-Strong from Text Classification to Generation/images/85495a0d6d5c545d9c7c1ce51cefc9cb6b0084d0c70406cbdda207652c836b93.jpg)

![880182825f16d8e77800efbc05542e54752f0e5c29545afe956eaf0b7bc089dc.jpg](../iclr_results/1162_Bayesian WeakS-to-Strong from Text Classification to Generation/images/880182825f16d8e77800efbc05542e54752f0e5c29545afe956eaf0b7bc089dc.jpg)

### Tables

![00ea7ff84aa6299e82b31df8a3ab51155f0d5d30764d3ea829306bb9f392eb5a.jpg](../iclr_results/1162_Bayesian WeakS-to-Strong from Text Classification to Generation/tables/00ea7ff84aa6299e82b31df8a3ab51155f0d5d30764d3ea829306bb9f392eb5a.jpg)

![3544e286ade28d0f3aa710418807fa01482249df078b434e34bc44977397102d.jpg](../iclr_results/1162_Bayesian WeakS-to-Strong from Text Classification to Generation/tables/3544e286ade28d0f3aa710418807fa01482249df078b434e34bc44977397102d.jpg)

![42cfdb6642a423483374c7517c6dd45804f6f4ad34f66626b76229ec103397d1.jpg](../iclr_results/1162_Bayesian WeakS-to-Strong from Text Classification to Generation/tables/42cfdb6642a423483374c7517c6dd45804f6f4ad34f66626b76229ec103397d1.jpg)

![4fae6160a61f2d87ab6ce176159e1a8dc7190f52d7493c41acf2fa88bf1d1ce7.jpg](../iclr_results/1162_Bayesian WeakS-to-Strong from Text Classification to Generation/tables/4fae6160a61f2d87ab6ce176159e1a8dc7190f52d7493c41acf2fa88bf1d1ce7.jpg)

![587ad73a8d2ccfebd43548a9babf97a1d42dd8f1c0951bbbf203a8b848950dfb.jpg](../iclr_results/1162_Bayesian WeakS-to-Strong from Text Classification to Generation/tables/587ad73a8d2ccfebd43548a9babf97a1d42dd8f1c0951bbbf203a8b848950dfb.jpg)

![669f00a984ababbfd3108c960ad65ae150cab643d1177b93f048d78e0f11720a.jpg](../iclr_results/1162_Bayesian WeakS-to-Strong from Text Classification to Generation/tables/669f00a984ababbfd3108c960ad65ae150cab643d1177b93f048d78e0f11720a.jpg)

![77ae9ce6e6712ef7ec63a3a5db87895d35f9b0bd8d529b0c968a126cef6165eb.jpg](../iclr_results/1162_Bayesian WeakS-to-Strong from Text Classification to Generation/tables/77ae9ce6e6712ef7ec63a3a5db87895d35f9b0bd8d529b0c968a126cef6165eb.jpg)

![824e7b49579a1a622a243da7753300da13dd03ba6056297f50ec39a0a65ae086.jpg](../iclr_results/1162_Bayesian WeakS-to-Strong from Text Classification to Generation/tables/824e7b49579a1a622a243da7753300da13dd03ba6056297f50ec39a0a65ae086.jpg)

![8914fd8abfcb5f5c6ed59e3b95b3c1bac78bddc055731b07c92af7c972be3bac.jpg](../iclr_results/1162_Bayesian WeakS-to-Strong from Text Classification to Generation/tables/8914fd8abfcb5f5c6ed59e3b95b3c1bac78bddc055731b07c92af7c972be3bac.jpg)

![a01aa9f9b95b78791a72af664af8e9c960c70d4445e62efa12efe1872a8ec6c9.jpg](../iclr_results/1162_Bayesian WeakS-to-Strong from Text Classification to Generation/tables/a01aa9f9b95b78791a72af664af8e9c960c70d4445e62efa12efe1872a8ec6c9.jpg)

![e394a5f78c312c1ff52a752b980c87a04859d05533d5230e483b7600b7e803c1.jpg](../iclr_results/1162_Bayesian WeakS-to-Strong from Text Classification to Generation/tables/e394a5f78c312c1ff52a752b980c87a04859d05533d5230e483b7600b7e803c1.jpg)

![fd220f284f9138d9809c63748a1e541fedd20d76bd9b64f6910b6caaf4823084.jpg](../iclr_results/1162_Bayesian WeakS-to-Strong from Text Classification to Generation/tables/fd220f284f9138d9809c63748a1e541fedd20d76bd9b64f6910b6caaf4823084.jpg)

## Multimodal Lego: Model Merging and Fine-Tuning Across Topologies and Modalities in Biomedicine


### Images

![2d95c0e9ee0498d81d4f6e97c0af38560d6b9fe4a7245a792ef615c30a026153.jpg](../iclr_results/1163_Multimodal Lego_ Model Merging and Fine-Tuning Across Topologies and Modalities in Biomedicine/images/2d95c0e9ee0498d81d4f6e97c0af38560d6b9fe4a7245a792ef615c30a026153.jpg)

![42b3daf1737230aadc55ba91214469b40e7828add8b3a42d5215754a6b6d355f.jpg](../iclr_results/1163_Multimodal Lego_ Model Merging and Fine-Tuning Across Topologies and Modalities in Biomedicine/images/42b3daf1737230aadc55ba91214469b40e7828add8b3a42d5215754a6b6d355f.jpg)

![59b52180af43449e86c410206596bac4d5f75389e2fd827c7a0318611b223a1f.jpg](../iclr_results/1163_Multimodal Lego_ Model Merging and Fine-Tuning Across Topologies and Modalities in Biomedicine/images/59b52180af43449e86c410206596bac4d5f75389e2fd827c7a0318611b223a1f.jpg)

![741f85b24e521bb97c7727713f0807319daa27315f7a5d21315763e606102b85.jpg](../iclr_results/1163_Multimodal Lego_ Model Merging and Fine-Tuning Across Topologies and Modalities in Biomedicine/images/741f85b24e521bb97c7727713f0807319daa27315f7a5d21315763e606102b85.jpg)

![886f9f636df4779e859c5c009c92003720ca25377c9f869e6bf6e846e2258094.jpg](../iclr_results/1163_Multimodal Lego_ Model Merging and Fine-Tuning Across Topologies and Modalities in Biomedicine/images/886f9f636df4779e859c5c009c92003720ca25377c9f869e6bf6e846e2258094.jpg)

![b5a6af6a33fa76759b7a0752a2a0e48de37719dcb7daee4d0ab2fa18151632f8.jpg](../iclr_results/1163_Multimodal Lego_ Model Merging and Fine-Tuning Across Topologies and Modalities in Biomedicine/images/b5a6af6a33fa76759b7a0752a2a0e48de37719dcb7daee4d0ab2fa18151632f8.jpg)

![c83af1c82164cc6e6aef523d1976ea9adb8252ce7661422ea644a53b1da3586f.jpg](../iclr_results/1163_Multimodal Lego_ Model Merging and Fine-Tuning Across Topologies and Modalities in Biomedicine/images/c83af1c82164cc6e6aef523d1976ea9adb8252ce7661422ea644a53b1da3586f.jpg)

### Tables

![2a3b43c86b842d7c478d1fa9eb9d8ab4e0e70076c90cf60d8769439e90b38d3b.jpg](../iclr_results/1163_Multimodal Lego_ Model Merging and Fine-Tuning Across Topologies and Modalities in Biomedicine/tables/2a3b43c86b842d7c478d1fa9eb9d8ab4e0e70076c90cf60d8769439e90b38d3b.jpg)

![84111b72f9b18e03f0765c3aec71f91c98c7314e7c3ac4481726d4b0b4903f96.jpg](../iclr_results/1163_Multimodal Lego_ Model Merging and Fine-Tuning Across Topologies and Modalities in Biomedicine/tables/84111b72f9b18e03f0765c3aec71f91c98c7314e7c3ac4481726d4b0b4903f96.jpg)

![f1e88899ab6d9bc5d1a7eb634693dc6391296109fe7ba5ef86528890519445b0.jpg](../iclr_results/1163_Multimodal Lego_ Model Merging and Fine-Tuning Across Topologies and Modalities in Biomedicine/tables/f1e88899ab6d9bc5d1a7eb634693dc6391296109fe7ba5ef86528890519445b0.jpg)

![f63560f458648d59b69a4544a9c268448f7428a88fbc99d3ea612ed4356c9982.jpg](../iclr_results/1163_Multimodal Lego_ Model Merging and Fine-Tuning Across Topologies and Modalities in Biomedicine/tables/f63560f458648d59b69a4544a9c268448f7428a88fbc99d3ea612ed4356c9982.jpg)

## Distribution-Free Data Uncertainty for Neural Network Regression


### Images

![4d485e17d75602a030461978d9dfbd77fa476cf0162801e3793277e8f033070a.jpg](../iclr_results/1164_Distribution-Free Data Uncertainty for Neural Network Regression/images/4d485e17d75602a030461978d9dfbd77fa476cf0162801e3793277e8f033070a.jpg)

![5fee8361e46f1304a3554f124c84a25e3337eb00ff3a2b8623a2f03de56c5812.jpg](../iclr_results/1164_Distribution-Free Data Uncertainty for Neural Network Regression/images/5fee8361e46f1304a3554f124c84a25e3337eb00ff3a2b8623a2f03de56c5812.jpg)

![9378a3a034354fa6e723d9da53a4890d6159cee2e312cd26c976fe7e14974df4.jpg](../iclr_results/1164_Distribution-Free Data Uncertainty for Neural Network Regression/images/9378a3a034354fa6e723d9da53a4890d6159cee2e312cd26c976fe7e14974df4.jpg)

![9a7067dec7df50e4cd53f7137dc17935675a504b2bd24e187693cea27a5fcb1f.jpg](../iclr_results/1164_Distribution-Free Data Uncertainty for Neural Network Regression/images/9a7067dec7df50e4cd53f7137dc17935675a504b2bd24e187693cea27a5fcb1f.jpg)

![9c0ff6a9a255a547fa8d8ea8cad59a32f847c2bf54596480b3a2b3d52fb08063.jpg](../iclr_results/1164_Distribution-Free Data Uncertainty for Neural Network Regression/images/9c0ff6a9a255a547fa8d8ea8cad59a32f847c2bf54596480b3a2b3d52fb08063.jpg)

![b875a687b3ae910671d3c064fb4478a6644c2b4eba1a2b27d18dcb8c752a286f.jpg](../iclr_results/1164_Distribution-Free Data Uncertainty for Neural Network Regression/images/b875a687b3ae910671d3c064fb4478a6644c2b4eba1a2b27d18dcb8c752a286f.jpg)

![d0d585b329260640b58333f9fc64ca4386056ac7be12d0317cdce8c08fe8795a.jpg](../iclr_results/1164_Distribution-Free Data Uncertainty for Neural Network Regression/images/d0d585b329260640b58333f9fc64ca4386056ac7be12d0317cdce8c08fe8795a.jpg)

![fbdd4024c002714fe0c78b80a89dd8314730f2ed13825e18d7585517e02756b2.jpg](../iclr_results/1164_Distribution-Free Data Uncertainty for Neural Network Regression/images/fbdd4024c002714fe0c78b80a89dd8314730f2ed13825e18d7585517e02756b2.jpg)

### Tables

![0472d1a3da3baf6d3ff83e22862a5d36b42765a2b7a76732054b9360dd2c4e51.jpg](../iclr_results/1164_Distribution-Free Data Uncertainty for Neural Network Regression/tables/0472d1a3da3baf6d3ff83e22862a5d36b42765a2b7a76732054b9360dd2c4e51.jpg)

![115fc2cbebc6cf2ad08e5098fb76e3fb8b7ca7244661e540eeafbf32d05d004c.jpg](../iclr_results/1164_Distribution-Free Data Uncertainty for Neural Network Regression/tables/115fc2cbebc6cf2ad08e5098fb76e3fb8b7ca7244661e540eeafbf32d05d004c.jpg)

![1dfc448ef34c744be4008e0666f26d12f6b33772f7a72db84453268ebf0e18c7.jpg](../iclr_results/1164_Distribution-Free Data Uncertainty for Neural Network Regression/tables/1dfc448ef34c744be4008e0666f26d12f6b33772f7a72db84453268ebf0e18c7.jpg)

![44c72dc0a456867f3f6c1ed89b2dcd6d707a2c90f4b4ce36caedecd6751a1cea.jpg](../iclr_results/1164_Distribution-Free Data Uncertainty for Neural Network Regression/tables/44c72dc0a456867f3f6c1ed89b2dcd6d707a2c90f4b4ce36caedecd6751a1cea.jpg)

![589603a26c24dae1d61e99c31af33d128fc443751b8e9231c4a9f34f31129d1d.jpg](../iclr_results/1164_Distribution-Free Data Uncertainty for Neural Network Regression/tables/589603a26c24dae1d61e99c31af33d128fc443751b8e9231c4a9f34f31129d1d.jpg)

![602ea344877e73e4043d4c58a16fddb415c00eb8ab31cdf75600b6fc2663af09.jpg](../iclr_results/1164_Distribution-Free Data Uncertainty for Neural Network Regression/tables/602ea344877e73e4043d4c58a16fddb415c00eb8ab31cdf75600b6fc2663af09.jpg)

![64dd6382b1af16794dcd8f437b961bdcdd008ddd7887227ca69d58216abb533e.jpg](../iclr_results/1164_Distribution-Free Data Uncertainty for Neural Network Regression/tables/64dd6382b1af16794dcd8f437b961bdcdd008ddd7887227ca69d58216abb533e.jpg)

![8a1dcbaabc5266c00c63a7716cc2f2b8085e8d810771d725d01736be1388d69b.jpg](../iclr_results/1164_Distribution-Free Data Uncertainty for Neural Network Regression/tables/8a1dcbaabc5266c00c63a7716cc2f2b8085e8d810771d725d01736be1388d69b.jpg)

![9db71cc15e11e5fb0c31a213aba46582ebd12bbee01cffef4611f9f65e19980e.jpg](../iclr_results/1164_Distribution-Free Data Uncertainty for Neural Network Regression/tables/9db71cc15e11e5fb0c31a213aba46582ebd12bbee01cffef4611f9f65e19980e.jpg)

![a3ec8ac7f342861b0949ebbc24c7483a0391e352405f2ed8c0e8ea889142aaa9.jpg](../iclr_results/1164_Distribution-Free Data Uncertainty for Neural Network Regression/tables/a3ec8ac7f342861b0949ebbc24c7483a0391e352405f2ed8c0e8ea889142aaa9.jpg)

![b05633ce0b0012ee0f5d166ebcd561f622807b4da0e499dd5edebbcf1c22d334.jpg](../iclr_results/1164_Distribution-Free Data Uncertainty for Neural Network Regression/tables/b05633ce0b0012ee0f5d166ebcd561f622807b4da0e499dd5edebbcf1c22d334.jpg)

![cfa488092a1ee2e7aae107bb82a180393b9d9b503dcceda1b960733e7448d680.jpg](../iclr_results/1164_Distribution-Free Data Uncertainty for Neural Network Regression/tables/cfa488092a1ee2e7aae107bb82a180393b9d9b503dcceda1b960733e7448d680.jpg)

![d619163f644e7cbfeeea0f117398d62061848eddff1481aa9de9041d1df6f830.jpg](../iclr_results/1164_Distribution-Free Data Uncertainty for Neural Network Regression/tables/d619163f644e7cbfeeea0f117398d62061848eddff1481aa9de9041d1df6f830.jpg)

![e965b37c548127f397d124b493cb97f2cfc07bf93cbddf53415c682e2db2de8f.jpg](../iclr_results/1164_Distribution-Free Data Uncertainty for Neural Network Regression/tables/e965b37c548127f397d124b493cb97f2cfc07bf93cbddf53415c682e2db2de8f.jpg)

## Jump Your Steps: Optimizing Sampling Schedule of Discrete Diffusion Models

### Images

![26325d27e77331d1624cbd83c48a553d2949866cf6409044112e0be4da49d416.jpg](../iclr_results/1165_Jump Your Steps_ Optimizing Sampling Schedule of Discrete Diffusion Models/images/26325d27e77331d1624cbd83c48a553d2949866cf6409044112e0be4da49d416.jpg)

![276f1fef9b4fa162dd7a9098d4113836cade9973ed1c3b6c2e8191e6d3134586.jpg](../iclr_results/1165_Jump Your Steps_ Optimizing Sampling Schedule of Discrete Diffusion Models/images/276f1fef9b4fa162dd7a9098d4113836cade9973ed1c3b6c2e8191e6d3134586.jpg)

![2d8291cccb300e1bd8541f8f5b69ce082b60eac30865c5f801723bd608a28d51.jpg](../iclr_results/1165_Jump Your Steps_ Optimizing Sampling Schedule of Discrete Diffusion Models/images/2d8291cccb300e1bd8541f8f5b69ce082b60eac30865c5f801723bd608a28d51.jpg)

![3c959a6250fe18d718942d774f10588e56b52bd38adc45384fc50896f4ddfdd9.jpg](../iclr_results/1165_Jump Your Steps_ Optimizing Sampling Schedule of Discrete Diffusion Models/images/3c959a6250fe18d718942d774f10588e56b52bd38adc45384fc50896f4ddfdd9.jpg)

![533fcf99e302415e55f6d01dd0fc9f82d29dcebfa47f37e9f439e2b434ec2775.jpg](../iclr_results/1165_Jump Your Steps_ Optimizing Sampling Schedule of Discrete Diffusion Models/images/533fcf99e302415e55f6d01dd0fc9f82d29dcebfa47f37e9f439e2b434ec2775.jpg)

![54ce50c3c1926494ef49bface72d8d591c435573d1f8d526945131262b870292.jpg](../iclr_results/1165_Jump Your Steps_ Optimizing Sampling Schedule of Discrete Diffusion Models/images/54ce50c3c1926494ef49bface72d8d591c435573d1f8d526945131262b870292.jpg)

![617760c9c4645fbfb84156419d6dfe4b68d045d0e596d1323ccaf609bce92518.jpg](../iclr_results/1165_Jump Your Steps_ Optimizing Sampling Schedule of Discrete Diffusion Models/images/617760c9c4645fbfb84156419d6dfe4b68d045d0e596d1323ccaf609bce92518.jpg)

![6a08be8829f146a5adc698bc3fb009f94a907332db78a016997047d6487abfaa.jpg](../iclr_results/1165_Jump Your Steps_ Optimizing Sampling Schedule of Discrete Diffusion Models/images/6a08be8829f146a5adc698bc3fb009f94a907332db78a016997047d6487abfaa.jpg)

![8c2230f1992a7557776b859c168d037adaa9784d845942eafd130c1b83946b06.jpg](../iclr_results/1165_Jump Your Steps_ Optimizing Sampling Schedule of Discrete Diffusion Models/images/8c2230f1992a7557776b859c168d037adaa9784d845942eafd130c1b83946b06.jpg)

![8d5e0c3915f002f42f4c2a89336c0e076ef53f898e657e824425ee5a682461c3.jpg](../iclr_results/1165_Jump Your Steps_ Optimizing Sampling Schedule of Discrete Diffusion Models/images/8d5e0c3915f002f42f4c2a89336c0e076ef53f898e657e824425ee5a682461c3.jpg)

![98f7824e0821637553afbfc36768eab8a76e39019838f3236d2ac01f6e4b93cc.jpg](../iclr_results/1165_Jump Your Steps_ Optimizing Sampling Schedule of Discrete Diffusion Models/images/98f7824e0821637553afbfc36768eab8a76e39019838f3236d2ac01f6e4b93cc.jpg)

![9c45db2fd89ffdf3a8730c5184ae21a9b95f09374c16923f527cbdb242ef618a.jpg](../iclr_results/1165_Jump Your Steps_ Optimizing Sampling Schedule of Discrete Diffusion Models/images/9c45db2fd89ffdf3a8730c5184ae21a9b95f09374c16923f527cbdb242ef618a.jpg)

![9cff1d4995f3200e9b31a3824856889264873a2537115359e857f16cbf51490d.jpg](../iclr_results/1165_Jump Your Steps_ Optimizing Sampling Schedule of Discrete Diffusion Models/images/9cff1d4995f3200e9b31a3824856889264873a2537115359e857f16cbf51490d.jpg)

![a50147998583da23b3f0adbe4c41af1741f056fa5d6844cf3820414c21a37ea5.jpg](../iclr_results/1165_Jump Your Steps_ Optimizing Sampling Schedule of Discrete Diffusion Models/images/a50147998583da23b3f0adbe4c41af1741f056fa5d6844cf3820414c21a37ea5.jpg)

![c1d4a8cccd312ae3993245b7f9ffd0c036634495bb273848d682ba40aa777e2e.jpg](../iclr_results/1165_Jump Your Steps_ Optimizing Sampling Schedule of Discrete Diffusion Models/images/c1d4a8cccd312ae3993245b7f9ffd0c036634495bb273848d682ba40aa777e2e.jpg)

![d84f4409c26afc15fd9700d723415bca5ab2408cc0afe8732f969593269c0cea.jpg](../iclr_results/1165_Jump Your Steps_ Optimizing Sampling Schedule of Discrete Diffusion Models/images/d84f4409c26afc15fd9700d723415bca5ab2408cc0afe8732f969593269c0cea.jpg)

![e665f982808dd613ac4999585430fbcfe24b314c0f7de2dec7648513f8675e11.jpg](../iclr_results/1165_Jump Your Steps_ Optimizing Sampling Schedule of Discrete Diffusion Models/images/e665f982808dd613ac4999585430fbcfe24b314c0f7de2dec7648513f8675e11.jpg)

![f53ccefbdd6382ae95af1d4c7b77b5220455c158c3bda970f4e61e93e4629a1e.jpg](../iclr_results/1165_Jump Your Steps_ Optimizing Sampling Schedule of Discrete Diffusion Models/images/f53ccefbdd6382ae95af1d4c7b77b5220455c158c3bda970f4e61e93e4629a1e.jpg)

![fc9c90ef7a41cb6141966b173c5453a893123af522c24212a5c025896dadd9e3.jpg](../iclr_results/1165_Jump Your Steps_ Optimizing Sampling Schedule of Discrete Diffusion Models/images/fc9c90ef7a41cb6141966b173c5453a893123af522c24212a5c025896dadd9e3.jpg)
